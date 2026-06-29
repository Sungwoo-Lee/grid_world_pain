"""
Configuration loader for JAX Environment.

Translates YAML config files into JAX-compatible EnvParams.

v2.0 changes (CP1 — unified animal entity):
  - `_load_animals()` replaces the two legacy `predators:` / `neutral_animals:` paths.
  - The `predator_enabled` YAML key is fully removed; configs carrying it after the
    migration sweep raise ValueError with a clear "removed in v2.0" message.
  - `predator_tags` / `neutral_tags` no longer passed to EnvParams constructor (M2 fix);
    they are derived via @property accessors on EnvParams (B3 / M1 fix in state.py).
"""
import re as _re
import logging
import os as _os
import yaml
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple
from src.environment.state import EnvParams

from src.utils.config import Config
import warnings

# ── `extends:` config layering ───────────────────────────────────────────────
# Root of all YAML configs in the repo; used to resolve `extends:` targets.
_CONFIGS_ROOT = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "configs")
)


def _resolve_extends(config_path: str, _seen: frozenset) -> Config:
    """Recursively resolve `extends:` chains and return a merged Config.

    Rules:
    - If the YAML has a top-level ``extends:`` key (str or list[str]),
      each named base is loaded (recursively resolving its own ``extends:``),
      deep-merged in declared order, and THIS file's keys are merged on top.
    - If there is no ``extends:`` key the file is loaded STANDALONE —
      byte-for-byte today's behaviour (what every archived/full config relies on).
    - ``extends`` targets are repo-relative paths under ``configs/`` with an
      implicit ``.yaml`` suffix, e.g. ``extends: environment/default``
      resolves to ``configs/environment/default.yaml``.
    - The ``extends:`` key itself is stripped from the merged result before
      return (it is meta, not an env param).
    - Cycle detection: a config that (transitively) extends itself raises
      ``ValueError``.
    """
    abs_path = _os.path.abspath(config_path)
    if abs_path in _seen:
        raise ValueError(
            f"Config `extends:` cycle detected at {config_path!r}"
        )
    _seen = _seen | {abs_path}

    raw = Config.load_yaml(config_path).to_dict()
    extends = raw.pop("extends", None)  # strip meta key — not an env param

    if extends is None:
        # STANDALONE — byte-for-byte today's behaviour.
        return Config(raw)

    bases = [extends] if isinstance(extends, str) else list(extends)
    merged = Config({})
    for base_rel in bases:
        base_path = _os.path.join(_CONFIGS_ROOT, base_rel + ".yaml")
        if not _os.path.exists(base_path):
            raise ValueError(
                f"Config `extends:` target {base_rel!r} not found "
                f"(looked for {base_path!r})"
            )
        merged.merge(_resolve_extends(base_path, _seen))
    merged.merge(Config(raw))  # this file's keys win
    return merged


def load_env_config(config_path: str) -> Config:
    """Resolve a config file to a fully-merged Config, honouring ``extends:``.

    - A config with ``extends: environment/default`` (or a list) gets the
      named base(s) deep-merged underneath it; this file's keys win.
    - A config **without** ``extends:`` loads exactly as today (standalone,
      byte-identical to ``Config.load_yaml(config_path)``).
    - ``get_mandatory`` validation in ``load_env_params`` runs on the merged
      result — the no-fallback contract is satisfied post-merge.

    Authoring note (list-replace semantics):
      ``Config.merge`` / ``deep_update`` replaces list values wholesale — the
      override list wins; elements are NOT merged.  To *suppress* a base list
      block (e.g. ``entities:`` or ``resources:``) a sparse config MUST
      declare it explicitly as an empty list (``entities: []``).  Omitting the
      key entirely causes the base's list to survive the merge unchanged.

    See plan docs/develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md
    for design rationale.
    """
    return _resolve_extends(config_path, _seen=frozenset())

_log = logging.getLogger(__name__)

# Allowed characters in entity tags (used in WandB key 'Episode/MeanDist*_<tag>').
# Slashes / spaces / dots break the WandB namespace or log key.
_TAG_RE = _re.compile(r'^[A-Za-z0-9_-]+$')

# ── Animal entity constants (v2.0) ────────────────────────────────────────────
ANIMAL_CLASS_TO_INT = {"predator": 0, "neutral": 1}
ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator": 5, "neutral": 7}
ANIMAL_DAMAGING_CLASSES = {"predator"}
ANIMAL_BEHAVIOUR_TO_INT = {"wander": 0, "hunt": 1, "static": 2}
DISTRIBUTIONAL_FIELDS = (
    "detection_range",
    "max_stamina",
    "stamina_recovery_rate",
    "hunt_stamina_threshold",
    "lose_interest_multiplier",
)

# ---------------------------------------------------------------------------
# Behavior-measure toolkit v1 schema loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BehaviorMeasureCfg:
    """Validated configuration for the behavior-measure toolkit v1."""
    enabled: bool
    cue_radius: float
    obs_window: int
    eval_n_episodes: int
    eval_seeds: Tuple[int, ...]
    eval_policy_mode: str            # "deterministic" | "stochastic"
    eval_max_steps: int
    eval_obs_noise: str              # "training" | "zero" | "custom"
    motif_window_K: int
    motif_features: Tuple[str, ...]
    motif_kmeans_k: int
    motif_kmeans_seed: int
    motif_standardise: str           # "zscore_pooled" | "zscore_per_agent" | "none"
    eval_output_root: str


_ALLOWED_POLICY_MODES = {"deterministic", "stochastic"}
_ALLOWED_NOISE_MODES = {"training", "zero", "custom"}
_ALLOWED_STANDARDISE = {"zscore_pooled", "zscore_per_agent", "none"}
_DEFAULT_FEATURE_NAMES = (
    "net_displacement", "path_length", "threat_distance_change_rate",
    "min_threat_distance", "bush_occupancy_fraction", "eat_events_per_window",
    "action_entropy", "mode_action_fraction", "stay_in_place_fraction",
    "drive_injury_change",
)


def load_behavior_measure_cfg(config) -> "BehaviorMeasureCfg | None":
    """Load behavior_measures: from the YAML config.

    Returns None if the top-level key is absent (backwards compatibility — existing
    configs that pre-date this feature load unchanged).  If the block IS present,
    every leaf key is mandatory and missing keys raise ValueError.
    """
    if config.get("behavior_measures") is None:
        return None  # backwards-compat: feature off, online accumulators no-op.

    enabled         = config.get_mandatory("behavior_measures.enabled")
    cue_radius      = config.get_mandatory("behavior_measures.cue_radius", float)
    obs_window      = config.get_mandatory("behavior_measures.obs_window", int)
    eval_n_eps      = config.get_mandatory("behavior_measures.eval_n_episodes", int)
    eval_seeds_raw  = config.get_mandatory("behavior_measures.eval_seeds")
    eval_pol_mode   = config.get_mandatory("behavior_measures.eval_policy_mode")
    eval_max_steps  = config.get_mandatory("behavior_measures.eval_max_steps", int)
    eval_obs_noise  = config.get_mandatory("behavior_measures.eval_obs_noise")
    motif_K         = config.get_mandatory("behavior_measures.motif_window_K", int)
    motif_features  = config.get_mandatory("behavior_measures.motif_features")
    motif_k         = config.get_mandatory("behavior_measures.motif_kmeans_k", int)
    motif_seed      = config.get_mandatory("behavior_measures.motif_kmeans_seed", int)
    motif_std       = config.get_mandatory("behavior_measures.motif_standardise")
    eval_output     = config.get_mandatory("behavior_measures.eval_output_root")

    # ----- eval_seeds: accept list/tuple (legacy) OR dict generator spec -----
    if isinstance(eval_seeds_raw, dict):
        # Generator spec form: { rng: <int>, sort: <bool> }
        # `rng` is mandatory — missing it would silently produce wrong seeds.
        if 'rng' not in eval_seeds_raw:
            raise ValueError(
                "behavior_measures.eval_seeds dict spec is missing mandatory key 'rng'. "
                "Expected form: {rng: <int>, sort: <bool>}."
            )
        _rng_seed = int(eval_seeds_raw['rng'])
        # `sort: true` is the default and is load-bearing: episode i uses eval_seeds[i],
        # so the order must stay fixed for cross-run / cross-cell episode-index comparability.
        _do_sort = bool(eval_seeds_raw.get('sort', True))
        _generated = np.random.default_rng(_rng_seed).integers(
            0, 2**31, size=eval_n_eps, dtype=np.int64
        ).tolist()
        eval_seeds_raw = sorted(_generated) if _do_sort else _generated

    # ----- validation (runs for both list/tuple and generated-from-spec paths) -----
    if cue_radius <= 0:
        raise ValueError(f"behavior_measures.cue_radius must be > 0; got {cue_radius}.")
    if obs_window < 1:
        raise ValueError(f"behavior_measures.obs_window must be >= 1; got {obs_window}.")
    if eval_n_eps < 1:
        raise ValueError(f"behavior_measures.eval_n_episodes must be >= 1; got {eval_n_eps}.")
    if not isinstance(eval_seeds_raw, (list, tuple)):
        raise ValueError(f"behavior_measures.eval_seeds must be a list/tuple or dict spec; got {type(eval_seeds_raw)}.")
    if len(eval_seeds_raw) != eval_n_eps:
        raise ValueError(
            f"behavior_measures.eval_seeds length ({len(eval_seeds_raw)}) != eval_n_episodes ({eval_n_eps})."
        )
    eval_seeds = tuple(int(s) for s in eval_seeds_raw)
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("behavior_measures.eval_seeds contains duplicates.")
    if eval_pol_mode not in _ALLOWED_POLICY_MODES:
        raise ValueError(f"behavior_measures.eval_policy_mode must be in {_ALLOWED_POLICY_MODES}; got {eval_pol_mode!r}.")
    if eval_max_steps < 1:
        raise ValueError(f"behavior_measures.eval_max_steps must be >= 1; got {eval_max_steps}.")
    if eval_obs_noise not in _ALLOWED_NOISE_MODES:
        raise ValueError(f"behavior_measures.eval_obs_noise must be in {_ALLOWED_NOISE_MODES}; got {eval_obs_noise!r}.")
    if motif_K < 1:
        raise ValueError(f"behavior_measures.motif_window_K must be >= 1; got {motif_K}.")
    if not motif_features or not all(isinstance(f, str) for f in motif_features):
        raise ValueError("behavior_measures.motif_features must be a non-empty list of strings.")
    unknown_features = set(motif_features) - set(_DEFAULT_FEATURE_NAMES)
    if unknown_features:
        raise ValueError(
            f"behavior_measures.motif_features contains unknown names: {sorted(unknown_features)}. "
            f"Allowed v1 features: {_DEFAULT_FEATURE_NAMES}."
        )
    if motif_k < 2:
        raise ValueError(f"behavior_measures.motif_kmeans_k must be >= 2; got {motif_k}.")
    if motif_std not in _ALLOWED_STANDARDISE:
        raise ValueError(f"behavior_measures.motif_standardise must be in {_ALLOWED_STANDARDISE}; got {motif_std!r}.")

    return BehaviorMeasureCfg(
        enabled=bool(enabled),
        cue_radius=float(cue_radius),
        obs_window=int(obs_window),
        eval_n_episodes=int(eval_n_eps),
        eval_seeds=eval_seeds,
        eval_policy_mode=str(eval_pol_mode),
        eval_max_steps=int(eval_max_steps),
        eval_obs_noise=str(eval_obs_noise),
        motif_window_K=int(motif_K),
        motif_features=tuple(str(f) for f in motif_features),
        motif_kmeans_k=int(motif_k),
        motif_kmeans_seed=int(motif_seed),
        motif_standardise=str(motif_std),
        eval_output_root=str(eval_output),
    )


def _normalise_tag(raw, idx, entity_label):
    """Return a valid metric-suffix string.  Empty / missing → f'idx{idx}'.

    The tag field is optional with a documented default, so this function uses
    entry.get('tag', None) and normalises in Python.  Do NOT use
    config.get_mandatory — the field is optional by design.
    """
    if raw is None or raw == "":
        return f"idx{idx}"
    s = str(raw)
    if not _TAG_RE.match(s):
        raise ValueError(
            f"{entity_label} tag {s!r} must match [A-Za-z0-9_-]+ "
            f"(used in WandB key suffix; slashes / spaces / dots break the namespace)."
        )
    return s


def _read_properties(entry, entity_label):
    """Read olfactory signature, preferring `properties` (plural)."""
    if 'properties' in entry:
        return entry['properties']
    if 'property' in entry:
        warnings.warn(
            f"{entity_label}: YAML key 'property' is deprecated — rename to 'properties'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return entry['property']
    raise ValueError(f"{entity_label}: missing required key 'properties'.")

def _one_hot_list(channel: int, V: int) -> list:
    """Return a Python list encoding one-hot(channel, V)."""
    v = [0.0] * V
    v[channel] = 1.0
    return v


def _read_visual_properties(entry: dict, default_channel: int, V: int, entity_label: str) -> list:
    """Read optional visual_properties from a config entry.

    If `visual_properties` is present, validates it has length V and returns it.
    If absent, returns one_hot(default_channel, V) as the default appearance vector.
    Raises ValueError if the length does not match V.
    """
    if 'visual_properties' in entry:
        vp = list(entry['visual_properties'])
        if len(vp) != V:
            raise ValueError(
                f"{entity_label}: 'visual_properties' has length {len(vp)} "
                f"but visual_vector_size is {V}. "
                f"Length must equal visual_vector_size."
            )
        return [float(x) for x in vp]
    # Default: one-hot of the entity's current channel
    return _one_hot_list(default_channel, V)


def _read_properties_std(entry, entity_label):
    """Same, for the `*_std` variant."""
    if 'properties_std' in entry:
        return entry['properties_std']
    if 'property_std' in entry:
        warnings.warn(
            f"{entity_label}: YAML key 'property_std' is deprecated — rename to 'properties_std'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return entry['property_std']
    raise ValueError(f"{entity_label}: missing required key 'properties_std'.")


def _read_visual_properties_std(entry: dict, V: int, entity_label: str) -> list:
    """Read optional visual_properties_std from a config entry.

    If `visual_properties_std` is present, validates it has length V and returns it.
    If absent (including all archived/pre-std configs), returns zeros of length V so
    that std=0 is the default and observations are byte-identical to pre-change.
    Raises ValueError if the length does not match V.
    """
    if 'visual_properties_std' in entry:
        vps = list(entry['visual_properties_std'])
        if len(vps) != V:
            raise ValueError(
                f"{entity_label}: 'visual_properties_std' has length {len(vps)} "
                f"but visual_vector_size is {V}. "
                f"Length must equal visual_vector_size."
            )
        return [float(x) for x in vps]
    # Default: zeros of length V → deterministic (std=0 → sampled == mean exactly)
    return [0.0] * V

def _load_animals(config: Config, visual_vector_size: int = 8):
    """Build unified animal arrays from the YAML config (v2.0).

    Supports two YAML paths, detected automatically:
      1. `environment.predators:` + `environment.neutral_animals:` (legacy) — all 86
         pre-v2.0 configs. Predators are re-projected with class='predator',
         behaviour='hunt'; neutrals with class='neutral', behaviour='wander'.
         The five distributional fields are read as scalars from the legacy YAML
         and stored as degenerate ranges [s, s]. `attack_delay` and `damage` for
         neutrals are internally auto-filled to 0 and [0.0, 0.0] (NC-1 fix —
         the legacy schema never carried them on neutral entries).
      2. `environment.entities:` (new schema) — not used by any config yet; CP3
         activates this path fully. If detected, a deprecation warning is emitted
         when the legacy sections are also present.

    Returns a flat tuple of all unified `animal_*` arrays and dispatch tuples,
    ordered as expected by the `load_env_params` caller.

    YAML cadence notes:
      - `damage: [lo, hi]` is per-event (re-sampled on every collision).
      - `detection_range: [lo, hi]` (and four siblings in DISTRIBUTIONAL_FIELDS)
        are per-episode (re-sampled at reset). Cadence is determined by field name.

    Mandatory-key rule by behaviour:
      - `behaviour: hunt` — all five distributional fields are MANDATORY.
        Missing → ValueError (no fallback default, per project rule).
      - `behaviour: wander` / `static` — distributional fields are OPTIONAL.
        If missing, loader auto-fills [0, 0] (internal projection detail;
        the wander/static code path never reads these arrays at runtime).
      - Legacy `neutral_animals:` re-projection always behaves as wander/optional.

    v1.x → v2.0 semantic change note: `lose_interest_multiplier` was previously
    optional in the predator loader (soft default 2.0). It is now MANDATORY for
    `behaviour: hunt`. All 86 pre-v2.0 predator entries carry an explicit value,
    so no existing config breaks; this change is flagged for reference.
    """
    h = config.get_mandatory('environment.height')
    w = config.get_mandatory('environment.width')

    def _parse_area(area, default_h=h, default_w=w):
        a = area if area is not None else [[1, 1], [default_h, default_w]]
        return [a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]]

    def _parse_distributional(entry, field, mandatory, entity_label, idx):
        """Read a scalar or [lo, hi] distributional field.
        Returns (low, high) as floats.
        """
        val = entry.get(field)
        if val is None:
            if mandatory:
                raise ValueError(
                    f"Animal entity {entity_label!r} (index {idx}) is missing mandatory "
                    f"distributional field '{field}' (required for behaviour='hunt'). "
                    f"No fallback default exists."
                )
            else:
                _log.debug(
                    "Animal entity %r (index %d): '%s' not specified; auto-filling [0, 0] "
                    "(wander/static — field is unused at runtime).", entity_label, idx, field
                )
                return 0.0, 0.0
        if isinstance(val, list):
            if len(val) != 2:
                raise ValueError(
                    f"Animal entity {entity_label!r} (index {idx}): '{field}' must be a scalar "
                    f"or a 2-element list [low, high]; got {val!r}."
                )
            lo, hi = float(val[0]), float(val[1])
            if hi < lo:
                raise ValueError(
                    f"Animal entity {entity_label!r} (index {idx}): '{field}' range "
                    f"must satisfy low <= high; got [{lo}, {hi}]."
                )
        else:
            lo = hi = float(val)
        return lo, hi

    # ── Build the expanded entry list ──────────────────────────────────────────
    entries = []  # list of dicts with unified fields

    # Check for new entities: schema (CP3 will add full support)
    has_entities = config.get('environment.entities') is not None
    has_legacy_predators = config.get('environment.predators') is not None
    has_legacy_neutrals = config.get('environment.neutral_animals') is not None

    if has_entities:
        # CP3 path — parse unified entities: list directly
        if has_legacy_predators or has_legacy_neutrals:
            warnings.warn(
                "Config has both 'environment.entities:' and legacy "
                "'environment.predators:'/'environment.neutral_animals:'. "
                "The unified 'entities:' schema takes precedence; legacy sections are ignored.",
                DeprecationWarning,
                stacklevel=3,
            )
        raw_entities = config.get('environment.entities') or []
        for i_raw, ent in enumerate(raw_entities):
            _lo, count = _resolve_count_range(ent, f'Entity[{i_raw}]')
            for _ in range(count):  # allocate count_high slots
                cls = ent.get('class')
                if cls is None:
                    raise ValueError(f"entities[{i_raw}]: missing required field 'class'.")
                beh = ent.get('behaviour')
                if beh is None:
                    raise ValueError(f"entities[{i_raw}]: missing required field 'behaviour'.")
                # Validate behaviour string FIRST (before building index tuples)
                if beh not in ANIMAL_BEHAVIOUR_TO_INT:
                    raise ValueError(
                        f"Unknown behaviour {beh!r} for entity tag={ent.get('tag', '?')}. "
                        f"Must be one of {list(ANIMAL_BEHAVIOUR_TO_INT)}."
                    )
                mandatory_dist = (beh == 'hunt')
                tag_raw = ent.get('tag')
                tag_label = str(tag_raw) if tag_raw else f'entity{len(entries)}'
                entries.append({
                    'class': cls,
                    'behaviour': beh,
                    'tag_raw': tag_raw,
                    'tag_label': tag_label,
                    'type_label': f'Entity[{i_raw}]',
                    'property': _read_properties(ent, f'Entity[{i_raw}]'),
                    'property_std': _read_properties_std(ent, f'Entity[{i_raw}]'),
                    'nociception': ent.get('nociception_intensity', 0.0),
                    'move_interval': ent.get('move_interval'),
                    'damage': ent.get('damage'),
                    'attack_delay': ent.get('attack_delay'),
                    'disengage_on_contact': bool(ent.get('disengage_on_contact', False)),
                    'spawn_area': ent.get('spawn_area'),
                    'patrol_area': ent.get('patrol_area'),
                    'mandatory_dist': mandatory_dist,
                    'dist_source': ent,
                })
    else:
        # Legacy path: re-project predators (hunt) + neutrals (wander)
        raw_predators = config.get('environment.predators') or []
        for i_raw, p in enumerate(raw_predators):
            _lo, count = _resolve_count_range(p, f'Predator[{i_raw}]')
            for _ in range(count):  # allocate count_high slots
                tag_raw = p.get('tag')
                tag_label = str(tag_raw) if tag_raw else f'pred{len(entries)}'
                def _p_get(key, _p=p, _label=tag_label):
                    val = _p.get(key)
                    if val is None:
                        raise ValueError(
                            f"Strict Config: Predator (tag={_label!r}) field '{key}' is required."
                        )
                    return val
                entries.append({
                    'class': 'predator',
                    'behaviour': 'hunt',
                    'tag_raw': tag_raw,
                    'tag_label': tag_label,
                    'type_label': 'Predator',
                    'property': _read_properties(p, 'Predator'),
                    'property_std': _read_properties_std(p, 'Predator'),
                    'nociception': p.get('nociception_intensity', 0.9),
                    'move_interval': _p_get('move_interval'),
                    'damage': _p_get('damage'),
                    'attack_delay': _p_get('attack_delay'),
                    'disengage_on_contact': bool(p.get('disengage_on_contact', False)),
                    'spawn_area': p.get('spawn_area'),
                    'patrol_area': p.get('patrol_area'),
                    'mandatory_dist': True,  # hunt — all dist fields mandatory
                    'dist_source': p,
                })

        raw_neutrals = config.get('environment.neutral_animals') or []
        for i_raw, n in enumerate(raw_neutrals):
            _lo, count = _resolve_count_range(n, f'NeutralAnimal[{i_raw}]')
            for _ in range(count):  # allocate count_high slots
                tag_raw = n.get('tag')
                tag_label = str(tag_raw) if tag_raw else f'rabbit{len(entries)}'
                def _n_get(key, _n=n, _label=tag_label):
                    val = _n.get(key)
                    if val is None:
                        raise ValueError(
                            f"Strict Config: Neutral Animal (tag={_label!r}) field '{key}' is required."
                        )
                    return val
                entries.append({
                    'class': 'neutral',
                    'behaviour': 'wander',
                    'tag_raw': tag_raw,
                    'tag_label': tag_label,
                    'type_label': 'Rabbit',
                    'property': _read_properties(n, 'Neutral Animal'),
                    'property_std': _read_properties_std(n, 'Neutral Animal'),
                    'nociception': n.get('nociception_intensity', 0.0),
                    'move_interval': _n_get('move_interval'),
                    # NC-1 fix: legacy neutrals never had damage/attack_delay;
                    # auto-fill internally (not user-facing fallback defaults).
                    'damage': [0.0, 0.0],
                    'attack_delay': 0,
                    'disengage_on_contact': bool(n.get('disengage_on_contact', False)),
                    'spawn_area': n.get('spawn_area'),
                    'patrol_area': n.get('patrol_area'),
                    'mandatory_dist': False,  # wander — dist fields optional
                    'dist_source': n,
                })

    N = len(entries)
    chem_dim = 5  # default; updated below if entries exist

    if N == 0:
        # Zero-animal case (M6 smoke) — return empty arrays
        animal_property = jnp.zeros((0, chem_dim))
        animal_property_std = jnp.zeros((0, chem_dim))
        animal_nociception = jnp.zeros(0)
        animal_move_int = jnp.zeros(0, dtype=jnp.int32)
        animal_move_int_low = jnp.zeros(0, dtype=jnp.int32)
        animal_move_int_high = jnp.zeros(0, dtype=jnp.int32)
        animal_damage = jnp.zeros((0, 2))
        animal_attack_delay = jnp.zeros(0, dtype=jnp.int32)
        animal_attack_delay_low = jnp.zeros(0, dtype=jnp.int32)
        animal_attack_delay_high = jnp.zeros(0, dtype=jnp.int32)
        animal_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        animal_patrol = jnp.zeros((0, 4), dtype=jnp.int32)
        animal_detect_low = jnp.zeros(0)
        animal_detect_high = jnp.zeros(0)
        animal_max_stamina_low = jnp.zeros(0)
        animal_max_stamina_high = jnp.zeros(0)
        animal_recovery_low = jnp.zeros(0)
        animal_recovery_high = jnp.zeros(0)
        animal_hunt_thresh_low = jnp.zeros(0)
        animal_hunt_thresh_high = jnp.zeros(0)
        animal_lose_interest_low = jnp.zeros(0)
        animal_lose_interest_high = jnp.zeros(0)
        animal_classes_int = jnp.zeros(0, dtype=jnp.int32)
        animal_behaviours_int = jnp.zeros(0, dtype=jnp.int32)
        animal_is_damaging = jnp.zeros(0, dtype=jnp.bool_)
        animal_disengage_on_contact = jnp.zeros(0, dtype=jnp.bool_)
        animal_visual_channel = jnp.zeros(0, dtype=jnp.int32)
        animal_visual_property = jnp.zeros((0, visual_vector_size), dtype=jnp.float32)
        animal_visual_property_std = jnp.zeros((0, visual_vector_size), dtype=jnp.float32)
        animal_classes = ()
        animal_behaviours = ()
        animal_tags = ()
        hunt_idx = ()
        wander_idx = ()
        static_idx = ()
        predator_indices = ()
        neutral_indices = ()
        pred_spawn_area_for_placement = jnp.zeros((0, 4), dtype=jnp.int32)
        neutral_spawn_area_for_placement = jnp.zeros((0, 4), dtype=jnp.int32)
        return (
            animal_property, animal_property_std, animal_nociception,
            animal_move_int, animal_damage, animal_attack_delay,
            animal_spawn_area, animal_patrol,
            animal_detect_low, animal_detect_high,
            animal_max_stamina_low, animal_max_stamina_high,
            animal_recovery_low, animal_recovery_high,
            animal_hunt_thresh_low, animal_hunt_thresh_high,
            animal_lose_interest_low, animal_lose_interest_high,
            animal_classes_int, animal_behaviours_int,
            animal_is_damaging, animal_disengage_on_contact, animal_visual_channel,
            animal_visual_property, animal_visual_property_std,
            animal_classes, animal_behaviours, animal_tags,
            hunt_idx, wander_idx, static_idx,
            predator_indices, neutral_indices,
            pred_spawn_area_for_placement, neutral_spawn_area_for_placement,
            animal_move_int_low, animal_move_int_high,
            animal_attack_delay_low, animal_attack_delay_high,
        )

    # ── Build per-field arrays ─────────────────────────────────────────────────
    props_list = [_read_properties(e, e['tag_label']) if 'property' in e and not isinstance(e.get('property'), list) else e['property'] for e in entries]
    stds_list = [e['property_std'] for e in entries]
    chem_dim = len(props_list[0])

    noc_list = [float(e['nociception']) for e in entries]

    # move_interval: scalar OR [lo, hi] integer range (per-episode sampling).
    # A scalar s is stored as degenerate range (s, s) → always samples s (backward compat).
    move_int_list = []  # list of (lo_int, hi_int) tuples
    for i, e in enumerate(entries):
        mi = e['move_interval']
        if mi is None:
            raise ValueError(
                f"Animal entity {e['tag_label']!r} (index {i}) is missing mandatory field 'move_interval'."
            )
        if isinstance(mi, list):
            if len(mi) != 2:
                raise ValueError(
                    f"Animal entity {e['tag_label']!r} (index {i}): 'move_interval' must be a scalar "
                    f"or a 2-element list [low, high]; got {mi!r}."
                )
            lo, hi = int(mi[0]), int(mi[1])
            if hi < lo:
                raise ValueError(
                    f"Animal entity {e['tag_label']!r} (index {i}): 'move_interval' range "
                    f"must satisfy low <= high; got [{lo}, {hi}]."
                )
        else:
            lo = hi = int(mi)
        move_int_list.append((lo, hi))

    damage_list = []
    for i, e in enumerate(entries):
        d = e['damage']
        if d is None:
            raise ValueError(
                f"Animal entity {e['tag_label']!r} (index {i}) is missing mandatory field 'damage'."
            )
        damage_list.append(d if isinstance(d, list) else [float(d), float(d)])

    # attack_delay: scalar OR [lo, hi] integer range (per-episode sampling).
    # A scalar s is stored as degenerate range (s, s) → always samples s (backward compat).
    attack_delay_list = []  # list of (lo_int, hi_int) tuples
    for i, e in enumerate(entries):
        a = e['attack_delay']
        if a is None:
            raise ValueError(
                f"Animal entity {e['tag_label']!r} (index {i}) is missing mandatory field 'attack_delay'."
            )
        if isinstance(a, list):
            if len(a) != 2:
                raise ValueError(
                    f"Animal entity {e['tag_label']!r} (index {i}): 'attack_delay' must be a scalar "
                    f"or a 2-element list [low, high]; got {a!r}."
                )
            lo, hi = int(a[0]), int(a[1])
            if hi < lo:
                raise ValueError(
                    f"Animal entity {e['tag_label']!r} (index {i}): 'attack_delay' range "
                    f"must satisfy low <= high; got [{lo}, {hi}]."
                )
        else:
            lo = hi = int(a)
        attack_delay_list.append((lo, hi))

    spawn_list = [_parse_area(e['spawn_area'], h, w) for e in entries]
    patrol_list = [_parse_area(e['patrol_area'], h, w) for e in entries]

    # Distributional fields (per entry)
    dist_field_names = (
        ("detection_range", "animal_detect"),
        ("max_stamina", "animal_max_stamina"),
        ("stamina_recovery_rate", "animal_recovery"),
        ("hunt_stamina_threshold", "animal_hunt_thresh"),
        ("lose_interest_multiplier", "animal_lose_interest"),
    )
    dist_lows = {k: [] for _, k in dist_field_names}
    dist_highs = {k: [] for _, k in dist_field_names}
    for i, e in enumerate(entries):
        for yaml_key, arr_key in dist_field_names:
            lo, hi = _parse_distributional(
                e['dist_source'], yaml_key,
                mandatory=e['mandatory_dist'],
                entity_label=e['tag_label'], idx=i
            )
            dist_lows[arr_key].append(lo)
            dist_highs[arr_key].append(hi)

    # Class / behaviour codes
    classes_int_list = []
    behaviours_int_list = []
    is_damaging_list = []
    disengage_on_contact_list = []
    visual_channel_list = []
    visual_property_list = []
    visual_property_std_list = []
    classes_tuple = []
    behaviours_tuple = []
    tags_tuple = []
    hunt_idx_list = []
    wander_idx_list = []
    static_idx_list = []
    predator_idx_list = []
    neutral_idx_list = []

    V = visual_vector_size  # shorthand

    for i, e in enumerate(entries):
        cls = e['class']
        beh = e['behaviour']
        # Validate behaviour string (for entities: path; legacy path is already validated)
        if beh not in ANIMAL_BEHAVIOUR_TO_INT:
            raise ValueError(
                f"Unknown behaviour {beh!r} for entity tag={e['tag_label']!r}. "
                f"Must be one of {list(ANIMAL_BEHAVIOUR_TO_INT)}."
            )
        if cls not in ANIMAL_CLASS_TO_INT:
            raise ValueError(
                f"Unknown class {cls!r} for entity tag={e['tag_label']!r}. "
                f"Must be one of {list(ANIMAL_CLASS_TO_INT)}."
            )
        classes_int_list.append(ANIMAL_CLASS_TO_INT[cls])
        behaviours_int_list.append(ANIMAL_BEHAVIOUR_TO_INT[beh])
        is_damaging_list.append(cls in ANIMAL_DAMAGING_CLASSES)
        disengage_on_contact_list.append(bool(e['disengage_on_contact']))
        default_vis_ch = ANIMAL_CLASS_TO_VIS_CHANNEL[cls]
        visual_channel_list.append(default_vis_ch)
        # Visual properties live in the raw YAML source (dist_source), not the normalised entry dict.
        _raw_src = e['dist_source']
        if V != 8 and 'visual_properties' not in _raw_src:
            raise ValueError(
                f"Animal entity {e['tag_label']!r}: 'visual_properties' is required "
                f"when visual_vector_size={V} (only the default V=8 can auto-generate "
                f"one-hot defaults from the class→channel map)."
            )
        visual_property_list.append(_read_visual_properties(_raw_src, default_vis_ch, V, e['tag_label']))
        visual_property_std_list.append(_read_visual_properties_std(_raw_src, V, e['tag_label']))
        classes_tuple.append(cls)
        behaviours_tuple.append(beh)
        tags_tuple.append(_normalise_tag(e['tag_raw'], i, e.get('type_label', e['tag_label'])))

        if beh == 'hunt':
            hunt_idx_list.append(i)
        elif beh == 'wander':
            wander_idx_list.append(i)
        else:
            static_idx_list.append(i)

        if cls == 'predator':
            predator_idx_list.append(i)
        elif cls == 'neutral':
            neutral_idx_list.append(i)

    # Build JAX arrays
    animal_property = jnp.array(props_list, dtype=jnp.float32)
    animal_property_std = jnp.array(stds_list, dtype=jnp.float32)
    animal_nociception = jnp.array(noc_list, dtype=jnp.float32)
    # move_int_list and attack_delay_list are now lists of (lo, hi) tuples.
    # Keep legacy fixed arrays using lo (degenerate range = lo == hi for scalar configs).
    animal_move_int = jnp.array([lo for lo, hi in move_int_list], dtype=jnp.int32)
    animal_move_int_low = jnp.array([lo for lo, hi in move_int_list], dtype=jnp.int32)
    animal_move_int_high = jnp.array([hi for lo, hi in move_int_list], dtype=jnp.int32)
    animal_damage = jnp.array(damage_list, dtype=jnp.float32)
    animal_attack_delay = jnp.array([lo for lo, hi in attack_delay_list], dtype=jnp.int32)
    animal_attack_delay_low = jnp.array([lo for lo, hi in attack_delay_list], dtype=jnp.int32)
    animal_attack_delay_high = jnp.array([hi for lo, hi in attack_delay_list], dtype=jnp.int32)
    animal_spawn_area = jnp.array(spawn_list, dtype=jnp.int32)
    animal_patrol = jnp.array(patrol_list, dtype=jnp.int32)

    animal_detect_low = jnp.array(dist_lows['animal_detect'], dtype=jnp.float32)
    animal_detect_high = jnp.array(dist_highs['animal_detect'], dtype=jnp.float32)
    animal_max_stamina_low = jnp.array(dist_lows['animal_max_stamina'], dtype=jnp.float32)
    animal_max_stamina_high = jnp.array(dist_highs['animal_max_stamina'], dtype=jnp.float32)
    animal_recovery_low = jnp.array(dist_lows['animal_recovery'], dtype=jnp.float32)
    animal_recovery_high = jnp.array(dist_highs['animal_recovery'], dtype=jnp.float32)
    animal_hunt_thresh_low = jnp.array(dist_lows['animal_hunt_thresh'], dtype=jnp.float32)
    animal_hunt_thresh_high = jnp.array(dist_highs['animal_hunt_thresh'], dtype=jnp.float32)
    animal_lose_interest_low = jnp.array(dist_lows['animal_lose_interest'], dtype=jnp.float32)
    animal_lose_interest_high = jnp.array(dist_highs['animal_lose_interest'], dtype=jnp.float32)

    animal_classes_int = jnp.array(classes_int_list, dtype=jnp.int32)
    animal_behaviours_int = jnp.array(behaviours_int_list, dtype=jnp.int32)
    animal_is_damaging = jnp.array(is_damaging_list, dtype=jnp.bool_)
    animal_disengage_on_contact = jnp.array(disengage_on_contact_list, dtype=jnp.bool_)
    animal_visual_channel = jnp.array(visual_channel_list, dtype=jnp.int32)
    animal_visual_property = jnp.array(visual_property_list, dtype=jnp.float32)      # [N, V]
    animal_visual_property_std = jnp.array(visual_property_std_list, dtype=jnp.float32)  # [N, V]

    animal_classes = tuple(classes_tuple)
    animal_behaviours = tuple(behaviours_tuple)
    animal_tags = tuple(tags_tuple)

    hunt_idx = tuple(hunt_idx_list)
    wander_idx = tuple(wander_idx_list)
    static_idx = tuple(static_idx_list)
    predator_indices = tuple(predator_idx_list)
    neutral_indices = tuple(neutral_idx_list)

    # Per-class spawn areas for placement (N1 fix: jax_reset uses [res, pred, obs, neutral] order)
    pred_spawn_area_for_placement = (
        animal_spawn_area[jnp.array(list(predator_indices), dtype=jnp.int32)]
        if predator_indices else jnp.zeros((0, 4), dtype=jnp.int32)
    )
    neutral_spawn_area_for_placement = (
        animal_spawn_area[jnp.array(list(neutral_indices), dtype=jnp.int32)]
        if neutral_indices else jnp.zeros((0, 4), dtype=jnp.int32)
    )

    return (
        animal_property, animal_property_std, animal_nociception,
        animal_move_int, animal_damage, animal_attack_delay,
        animal_spawn_area, animal_patrol,
        animal_detect_low, animal_detect_high,
        animal_max_stamina_low, animal_max_stamina_high,
        animal_recovery_low, animal_recovery_high,
        animal_hunt_thresh_low, animal_hunt_thresh_high,
        animal_lose_interest_low, animal_lose_interest_high,
        animal_classes_int, animal_behaviours_int,
        animal_is_damaging, animal_disengage_on_contact, animal_visual_channel,
        animal_visual_property, animal_visual_property_std,
        animal_classes, animal_behaviours, animal_tags,
        hunt_idx, wander_idx, static_idx,
        predator_indices, neutral_indices,
        pred_spawn_area_for_placement, neutral_spawn_area_for_placement,
        animal_move_int_low, animal_move_int_high,
        animal_attack_delay_low, animal_attack_delay_high,
    )


def _resolve_count_range(entry: dict, entity_label: str):
    """Return (count_low, count_high) from a YAML entity entry.

    Rules (per-episode count-range feature — PER_EPISODE_ENV_VARIANCE plan):
    - Both ``count_low`` + ``count_high`` present → use them.
      Validates 0 <= low <= high.
    - Only ``count`` present → degenerate range (count_low = count_high = count).
      Backward-compatible: byte-identical to pre-feature behaviour.
    - Both styles present simultaneously → ValueError (ambiguous).
    - Neither → degenerate (1, 1), matching the old r.get('count', 1) default.

    This is an optional-fallback (not get_mandatory) because count_low/count_high are
    genuinely optional; absence means "use the scalar count or default of 1".
    """
    has_range  = ('count_low'  in entry) or ('count_high' in entry)
    has_scalar = 'count' in entry

    if has_range and has_scalar:
        raise ValueError(
            f"{entity_label}: 'count' and 'count_low'/'count_high' are mutually exclusive. "
            "Use either the scalar 'count: N' OR the range 'count_low: L / count_high: H'."
        )
    if has_range:
        lo = int(entry.get('count_low', 0))
        hi = int(entry.get('count_high', 0))
        if not (0 <= lo <= hi):
            raise ValueError(
                f"{entity_label}: count_low={lo} and count_high={hi} must satisfy "
                "0 <= count_low <= count_high."
            )
        return lo, hi
    else:
        n = int(entry.get('count', 1))
        return n, n


def load_env_params(config: Config) -> EnvParams:
    """Loads environment parameters from a Config object with strict retrieval."""

    # ── Visual vector size (V) ────────────────────────────────────────────────
    # Read-site default of 8: preserves byte-parity for all ~86 archived configs
    # that do not declare this key. The one permitted read-site default in this
    # plan (see CONFIGURABLE_VISUAL_PROPERTIES_PLAN.md §D1).
    _vis_v = config.get('sensory.visual_vector_size')
    visual_vector_size: int = int(_vis_v) if _vis_v is not None else 8

    # Build resource arrays
    raw_resources = config.get_mandatory('environment.resources')
    expanded_resources = []
    # Per-entry count-range metadata (NEW — PER_EPISODE_ENV_VARIANCE)
    res_count_low_list = []   # per-entry int32 lower bound
    res_count_high_list = []  # per-entry int32 upper bound
    res_entry_id_list = []    # per-slot → entry index
    if raw_resources:
        for i_entry, r in enumerate(raw_resources):
            lo, hi = _resolve_count_range(r, f'Resource[{i_entry}]')
            res_count_low_list.append(lo)
            res_count_high_list.append(hi)
            # Allocate count_high slots (not count); activate K in jax_reset
            for slot_i in range(hi):
                expanded_resources.append(r)
                res_entry_id_list.append(i_entry)
    
    if expanded_resources:
        def r_get(r, key):
            val = r.get(key)
            if val is None: raise ValueError(f"Strict Config: Resource field '{key}' is required.")
            if key == 'type' and val == 'danger':
                warnings.warn(
                    "Resource type 'danger' is deprecated — rename to 'hiding_predator'.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return val

        # 0: food, 1: hiding_predator
        res_type = jnp.array([0 if r_get(r, 'type') == 'food' else 1 for r in expanded_resources], dtype=jnp.int32)
        res_property = jnp.array([_read_properties(r, 'Resource') for r in expanded_resources])
        chem_dim = res_property.shape[-1]
        res_property_std = jnp.array([_read_properties_std(r, 'Resource') for r in expanded_resources])
        # Subtract 1 for minval (0-based) but keep maxval as is for JAX's exclusive upper bound
        res_spawn_area = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [r_get(r, 'spawn_area') for r in expanded_resources]])
        res_max_cons = jnp.array([r_get(r, 'max_consumption') for r in expanded_resources], dtype=jnp.int32)
        res_reg_delay = jnp.array([r_get(r, 'regeneration_delay') for r in expanded_resources], dtype=jnp.int32)

        # Damage can be scalar (Feb 12) or range [min, max] (tuningEnv)
        raw_damage = [r_get(r, 'damage') for r in expanded_resources]
        res_damage = jnp.array([d if isinstance(d, list) else [d, d] for d in raw_damage])

        res_nociception = jnp.array([r.get('nociception_intensity', 0.9 if r_get(r, 'type') in ('hiding_predator', 'danger') else 0.0) for r in expanded_resources])

        # Visual property vectors: food→channel 3, hiding_predator→channel 4
        _res_vis_list = []
        _res_vis_std_list = []
        for r in expanded_resources:
            _rtype = r_get(r, 'type')
            _default_ch = 3 if _rtype == 'food' else 4  # food=3, hiding_predator=4
            if visual_vector_size != 8 and 'visual_properties' not in r:
                raise ValueError(
                    f"Resource type={_rtype!r}: 'visual_properties' is required "
                    f"when visual_vector_size={visual_vector_size} (only V=8 can "
                    f"auto-generate one-hot defaults from the channel map)."
                )
            _res_vis_list.append(_read_visual_properties(r, _default_ch, visual_vector_size, f'Resource({_rtype})'))
            _res_vis_std_list.append(_read_visual_properties_std(r, visual_vector_size, f'Resource({_rtype})'))
        res_visual_property = jnp.array(_res_vis_list, dtype=jnp.float32)
        res_visual_property_std = jnp.array(_res_vis_std_list, dtype=jnp.float32)
    else:
        res_type = jnp.zeros(0, dtype=jnp.int32)
        res_property = jnp.zeros((0, 5))
        res_property_std = jnp.zeros((0, 5))
        res_nociception = jnp.zeros(0)
        res_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        res_max_cons = jnp.zeros(0, dtype=jnp.int32)
        res_reg_delay = jnp.zeros(0, dtype=jnp.int32)
        res_damage = jnp.zeros((0, 2))
        res_visual_property = jnp.zeros((0, visual_vector_size), dtype=jnp.float32)
        res_visual_property_std = jnp.zeros((0, visual_vector_size), dtype=jnp.float32)

    # ── Guard against stale `predator_enabled` key (removed in v2.0) ──────────
    # The 86 migrated configs have this key stripped by the CP1 migration sweep.
    # Any config that still carries it after migration raises a clear error.
    if config.get('environment.predator_enabled') is not None:
        raise ValueError(
            "Config key 'environment.predator_enabled' was removed in v2.0. "
            "Strip this line from your YAML (it was always True for 85 of 86 configs; "
            "for the neutral-only case, use 'predators: []' which is already present)."
        )

    # ── Build unified animal arrays via _load_animals() ──────────────────────
    (
        animal_property, animal_property_std, animal_nociception,
        animal_move_int, animal_damage, animal_attack_delay,
        animal_spawn_area, animal_patrol,
        animal_detect_low, animal_detect_high,
        animal_max_stamina_low, animal_max_stamina_high,
        animal_recovery_low, animal_recovery_high,
        animal_hunt_thresh_low, animal_hunt_thresh_high,
        animal_lose_interest_low, animal_lose_interest_high,
        animal_classes_int, animal_behaviours_int,
        animal_is_damaging, animal_disengage_on_contact, animal_visual_channel,
        animal_visual_property, animal_visual_property_std,
        animal_classes, animal_behaviours, animal_tags,
        hunt_idx, wander_idx, static_idx,
        predator_indices, neutral_indices,
        pred_spawn_area_for_placement, neutral_spawn_area_for_placement,
        animal_move_int_low, animal_move_int_high,
        animal_attack_delay_low, animal_attack_delay_high,
    ) = _load_animals(config, visual_vector_size=visual_vector_size)

    # ── Animal count-range metadata (NEW — PER_EPISODE_ENV_VARIANCE) ──────────
    # Re-read the raw entity entries to extract per-entry count_low / count_high.
    # This mirrors the expansion done in _load_animals (but here we only need counts).
    animal_count_low_list = []   # per-entry int32 lower bound
    animal_count_high_list = []  # per-entry int32 upper bound
    animal_entry_id_list = []    # per-slot → entry index
    _has_entities = config.get('environment.entities') is not None
    if _has_entities:
        _raw_ents = config.get('environment.entities') or []
        for i_raw, ent in enumerate(_raw_ents):
            lo, hi = _resolve_count_range(ent, f'Entity[{i_raw}]')
            animal_count_low_list.append(lo)
            animal_count_high_list.append(hi)
            for _s in range(hi):
                animal_entry_id_list.append(i_raw)
    else:
        # Legacy predators: each raw entry is a single-slot entry (no count_low/count_high)
        _raw_preds = config.get('environment.predators') or []
        for i_raw, p in enumerate(_raw_preds):
            lo, hi = _resolve_count_range(p, f'Predator[{i_raw}]')
            animal_count_low_list.append(lo)
            animal_count_high_list.append(hi)
            for _s in range(hi):
                animal_entry_id_list.append(i_raw)
        # Legacy neutrals: same pattern, offset entry indices
        _raw_neutrals = config.get('environment.neutral_animals') or []
        _n_pred_entries = len(_raw_preds)
        for i_raw, n in enumerate(_raw_neutrals):
            lo, hi = _resolve_count_range(n, f'NeutralAnimal[{i_raw}]')
            animal_count_low_list.append(lo)
            animal_count_high_list.append(hi)
            for _s in range(hi):
                animal_entry_id_list.append(_n_pred_entries + i_raw)

    # Build Obstacle arrays
    raw_obstacles = config.get_mandatory('environment.obstacles')
    expanded_obstacles = []
    # Per-entry count-range metadata (NEW — PER_EPISODE_ENV_VARIANCE)
    obs_count_low_list = []   # per-entry int32 lower bound
    obs_count_high_list = []  # per-entry int32 upper bound
    obs_entry_id_list = []    # per-slot → entry index
    for i_entry, o in enumerate(raw_obstacles):
        lo, hi = _resolve_count_range(o, f'Obstacle[{i_entry}]')
        obs_count_low_list.append(lo)
        obs_count_high_list.append(hi)
        # Allocate count_high slots (not count); activate K in jax_reset
        for slot_i in range(hi):
            expanded_obstacles.append(o)
            obs_entry_id_list.append(i_entry)
            
    if expanded_obstacles:
        def obs_get(o, key):
            val = o.get(key)
            if val is None: raise ValueError(f"Strict Config: Obstacle field '{key}' is required.")
            return val
        obs_blocking = jnp.array([o.get('blocking', True) for o in expanded_obstacles], dtype=jnp.bool_)
        obs_hides_agent = jnp.array([o.get('hides_agent', False) for o in expanded_obstacles], dtype=jnp.bool_)
        obs_blocks_animals = jnp.array([o.get('blocks_animals', False) for o in expanded_obstacles], dtype=jnp.bool_)
        
        # Obstacle damage ranges
        raw_obs_damage = [o.get('damage', 0.0) for o in expanded_obstacles]
        obs_damage = jnp.array([d if isinstance(d, list) else [d, d] for d in raw_obs_damage])
        
        obs_nociception = jnp.array([o.get('nociception_intensity', 0.3) for o in expanded_obstacles], dtype=jnp.float32)
        # Unified: Obstacles can have properties too
        chem_dim = res_property.shape[-1]
        obs_property = jnp.array([_read_properties(o, 'Obstacle') for o in expanded_obstacles])
        obs_property_std = jnp.array([_read_properties_std(o, 'Obstacle') for o in expanded_obstacles])
        # Adjust for 0-based min and exclusive max
        obs_spawn_area = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [obs_get(o, 'area') for o in expanded_obstacles]])
        
        # Obstacle types for visual sensor
        obstacle_names = tuple(sorted(list(set([o.get('name', 'rock') for o in expanded_obstacles]))))
        name_to_idx = {name: i for i, name in enumerate(obstacle_names)}
        obs_type = jnp.array([name_to_idx[o.get('name', 'rock')] for o in expanded_obstacles], dtype=jnp.int32)

        # Visual property vectors: rock→channel 6
        _obs_vis_list = []
        _obs_vis_std_list = []
        for o in expanded_obstacles:
            _oname = o.get('name', 'rock')
            if visual_vector_size != 8 and 'visual_properties' not in o:
                raise ValueError(
                    f"Obstacle name={_oname!r}: 'visual_properties' is required "
                    f"when visual_vector_size={visual_vector_size} (only V=8 can "
                    f"auto-generate one-hot defaults from the channel map)."
                )
            _obs_vis_list.append(_read_visual_properties(o, 6, visual_vector_size, f'Obstacle({_oname})'))
            _obs_vis_std_list.append(_read_visual_properties_std(o, visual_vector_size, f'Obstacle({_oname})'))
        obs_visual_property = jnp.array(_obs_vis_list, dtype=jnp.float32)
        obs_visual_property_std = jnp.array(_obs_vis_std_list, dtype=jnp.float32)
    else:
        obs_blocking = jnp.zeros(0, dtype=jnp.bool_)
        obs_hides_agent = jnp.zeros(0, dtype=jnp.bool_)
        obs_blocks_animals = jnp.zeros(0, dtype=jnp.bool_)
        obs_damage = jnp.zeros((0, 2), dtype=jnp.float32)
        obs_nociception = jnp.zeros(0, dtype=jnp.float32)
        chem_dim = res_property.shape[-1]
        obs_property = jnp.zeros((0, chem_dim))
        obs_property_std = jnp.zeros((0, chem_dim))
        obs_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        obs_type = jnp.zeros(0, dtype=jnp.int32)
        obstacle_names = ("rock",)
        obs_visual_property = jnp.zeros((0, visual_vector_size), dtype=jnp.float32)
        obs_visual_property_std = jnp.zeros((0, visual_vector_size), dtype=jnp.float32)
    
    # Build Grid Location Types
    import numpy as np
    height = config.get_mandatory('environment.height')
    width = config.get_mandatory('environment.width')
    grid_np = np.zeros((height, width), dtype=np.int32)
    location_areas = config.get_mandatory('environment.location_areas')
    for area_config in location_areas:
        a_type = area_config.get('type')
        type_idx = 1 if a_type == 'grass' else 2 if a_type == 'sand' else 0
        area = area_config.get('area')
        if area:
            # 1-based conversion: [r1, c1] to [r2, c2] inclusive maps to grid[r1-1:r2, c1-1:c2]
            r1, c1, r2, c2 = area[0][0], area[0][1], area[1][0], area[1][1]
            grid_np[r1-1:r2, c1-1:c2] = type_idx
    grid_location_type = jnp.array(grid_np)

    # ── Visual background property table [3, V] ───────────────────────────────
    # Rows: grass (loc 1 → index 0), sand (loc 2 → index 1), plain (loc 0 → index 2)
    # Default: one-hot(0/1/2, V). At V≠8, require explicit sensory.visual_background_properties.
    _vbg_raw = config.get('sensory.visual_background_properties')
    if visual_vector_size != 8:
        if _vbg_raw is None:
            raise ValueError(
                "'sensory.visual_background_properties' is required when "
                f"visual_vector_size={visual_vector_size} (a 3×V table of floats, "
                "rows = [grass, sand, plain])."
            )
        _vbg = list(_vbg_raw)
        if len(_vbg) != 3 or any(len(row) != visual_vector_size for row in _vbg):
            raise ValueError(
                f"'sensory.visual_background_properties' must be a 3×{visual_vector_size} "
                f"table; got shape {len(_vbg)}×{[len(r) for r in _vbg]}."
            )
        visual_background_property = jnp.array(
            [[float(x) for x in row] for row in _vbg], dtype=jnp.float32
        )
    else:
        if _vbg_raw is not None:
            # User explicitly supplied it at V=8 — validate and use it.
            _vbg = list(_vbg_raw)
            if len(_vbg) != 3 or any(len(row) != 8 for row in _vbg):
                raise ValueError(
                    f"'sensory.visual_background_properties' must be a 3×8 table when "
                    f"visual_vector_size=8; got shape {len(_vbg)}×{[len(r) for r in _vbg]}."
                )
            visual_background_property = jnp.array(
                [[float(x) for x in row] for row in _vbg], dtype=jnp.float32
            )
        else:
            # Default: one-hot rows for grass(0), sand(1), plain(2)
            visual_background_property = jnp.eye(8, dtype=jnp.float32)[:3]  # [3, 8]

    # ── Type-Level Placement: Group entities by spawn area ──
    # Preserve today's [res, pred, obs, neutral] index order for type_entity_map
    # (N1 fix: jax_reset placement uses this ordering for the resolve-scan).
    all_spawn_areas_np = np.concatenate([
        np.array(res_spawn_area), np.array(pred_spawn_area_for_placement),
        np.array(obs_spawn_area), np.array(neutral_spawn_area_for_placement)
    ], axis=0)  # [N, 4]
    num_total_entities = all_spawn_areas_np.shape[0]
    
    # Group by unique spawn area
    groups_by_area = {}  # tuple(area) -> list of entity global indices
    for eidx in range(num_total_entities):
        area_key = tuple(all_spawn_areas_np[eidx].tolist())
        groups_by_area.setdefault(area_key, []).append(eidx)
    
    # Build type-level arrays
    area_keys_list = list(groups_by_area.keys())
    num_types = len(area_keys_list)
    type_counts_list = [len(groups_by_area[k]) for k in area_keys_list]
    max_per_type = max(type_counts_list) if type_counts_list else 1
    
    type_areas_np = np.array([list(k) for k in area_keys_list], dtype=np.int32)
    type_entity_map_np = np.full((num_types, max_per_type), 0, dtype=np.int32)
    for tidx, k in enumerate(area_keys_list):
        ents = groups_by_area[k]
        type_entity_map_np[tidx, :len(ents)] = ents
    
    # Parse placement mode
    placement_mode = config.get('environment.placement.mode', 'per_entity')
    assert placement_mode in ('per_entity', 'per_type'), f"Unknown placement mode: {placement_mode}"

    _log.debug("=" * 60)
    _log.debug("ENTITY PLACEMENT STRATEGY: %s", placement_mode)
    _log.debug("=" * 60)
    _log.debug("Grid: %d×%d (%d cells)", height, width, height * width)
    _log.debug("Total entities: %d", num_total_entities)
    if placement_mode == 'per_type':
        _log.debug("Type groups: %d (one lax.scan step each)", num_types)
        for tidx, k in enumerate(area_keys_list):
            area = list(k)
            cnt = type_counts_list[tidx]
            area_cells = (area[2] - area[0]) * (area[3] - area[1])
            _log.debug(
                "  Group %d: area %s → %d entities / %d cells (%.0f%%)",
                tidx, area, cnt, area_cells, 100 * cnt / area_cells if area_cells else 0
            )
        _log.debug("Max entities per group: %d", max_per_type)
        _log.debug("Sequential steps: %d", num_types)
    else:
        _log.debug("Sequential steps: %d (one per entity)", num_total_entities)
    _log.debug("=" * 60)
    
    # Hidden-state observability flags
    injury_observable = bool(config.get_mandatory('sensory.injury_observable'))
    nutrition_observable = bool(config.get_mandatory('sensory.nutrition_observable'))

    # Interoceptive nociception (delayed-peak perception of hidden injury)
    interoceptive_nociception_enabled = bool(config.get_mandatory('sensory.interoceptive_nociception_enabled'))
    interoceptive_convolution_enabled = bool(config.get_mandatory('sensory.interoceptive_convolution_enabled'))
    interoceptive_kernel_length = int(config.get_mandatory('sensory.interoceptive_kernel_length'))
    interoceptive_kernel_tau = float(config.get_mandatory('sensory.interoceptive_kernel_tau'))

    if interoceptive_convolution_enabled:
        # Build normalized alpha kernel: k_raw[i] = (i/τ)·exp(1 - i/τ); k[i] = k_raw[i] / Σk_raw
        _k_idx = np.arange(interoceptive_kernel_length, dtype=np.float32)
        _k_raw = (_k_idx / interoceptive_kernel_tau) * np.exp(1.0 - _k_idx / interoceptive_kernel_tau)
        _k_sum = float(_k_raw.sum())
        if _k_sum <= 0.0:
            raise ValueError(
                f"interoceptive_kernel produced non-positive sum ({_k_sum}). "
                f"Check tau ({interoceptive_kernel_tau}) and length ({interoceptive_kernel_length})."
            )
        interoceptive_kernel = jnp.array(_k_raw / _k_sum, dtype=jnp.float32)
    else:
        # Passthrough mode — kernel is unused but kept as zeros for shape stability.
        interoceptive_kernel = jnp.zeros(interoceptive_kernel_length, dtype=jnp.float32)

    # Configurable initial-state ranges (Fork B2: conditional-mandatory).
    # Range keys are required ONLY when the matching random_start_* flag is true;
    # when the flag is false, sentinels are used (values never read at runtime).
    _rand_nutr = bool(config.get_mandatory('body.random_start_nutrition'))
    _rand_inj  = bool(config.get_mandatory('body.random_start_injury'))
    _max_nutr  = float(config.get_mandatory('body.max_nutrition'))
    _max_inj   = float(config.get_mandatory('body.max_injury'))

    if _rand_nutr:
        start_nutrition_low  = float(config.get_mandatory('body.start_nutrition_low'))
        start_nutrition_high = float(config.get_mandatory('body.start_nutrition_high'))
        if not (0.0 <= start_nutrition_low <= start_nutrition_high <= _max_nutr):
            raise ValueError(
                f"body.start_nutrition_low/high must satisfy 0 <= low <= high <= max_nutrition "
                f"({_max_nutr}); got low={start_nutrition_low}, high={start_nutrition_high}")
    else:
        # Sentinel: unused when random_start_nutrition is False.
        # Keeps params pytree shape static across configs (traced floats).
        start_nutrition_low  = 0.0
        start_nutrition_high = _max_nutr

    if _rand_inj:
        start_injury_low  = float(config.get_mandatory('body.start_injury_low'))
        start_injury_high = float(config.get_mandatory('body.start_injury_high'))
        if not (0.0 <= start_injury_low <= start_injury_high <= _max_inj):
            raise ValueError(
                f"body.start_injury_low/high must satisfy 0 <= low <= high <= max_injury "
                f"({_max_inj}); got low={start_injury_low}, high={start_injury_high}")
    else:
        # Sentinel: unused when random_start_injury is False.
        start_injury_low  = 0.0
        start_injury_high = _max_inj / 2.0

    return EnvParams(
        height=height,
        width=width,
        max_steps=config.get_mandatory('environment.max_steps'),
        grid_location_type=grid_location_type,
        res_type=res_type,
        res_property=res_property,
        res_property_std=res_property_std,
        res_visual_property=res_visual_property,
        res_visual_property_std=res_visual_property_std,
        res_nociception=res_nociception,
        res_spawn_area=res_spawn_area,
        res_max_cons=res_max_cons,
        res_reg_delay=res_reg_delay,
        res_damage=res_damage,
        # Unified animal arrays (M2 fix: no predator_tags= / neutral_tags= kwargs)
        animal_property=animal_property,
        animal_property_std=animal_property_std,
        animal_nociception=animal_nociception,
        animal_move_int=animal_move_int,
        animal_move_int_low=animal_move_int_low,
        animal_move_int_high=animal_move_int_high,
        animal_damage=animal_damage,
        animal_attack_delay=animal_attack_delay,
        animal_attack_delay_low=animal_attack_delay_low,
        animal_attack_delay_high=animal_attack_delay_high,
        animal_spawn_area=animal_spawn_area,
        animal_patrol=animal_patrol,
        animal_detect_low=animal_detect_low,
        animal_detect_high=animal_detect_high,
        animal_max_stamina_low=animal_max_stamina_low,
        animal_max_stamina_high=animal_max_stamina_high,
        animal_recovery_low=animal_recovery_low,
        animal_recovery_high=animal_recovery_high,
        animal_hunt_thresh_low=animal_hunt_thresh_low,
        animal_hunt_thresh_high=animal_hunt_thresh_high,
        animal_lose_interest_low=animal_lose_interest_low,
        animal_lose_interest_high=animal_lose_interest_high,
        animal_classes_int=animal_classes_int,
        animal_behaviours_int=animal_behaviours_int,
        animal_is_damaging=animal_is_damaging,
        animal_disengage_on_contact=animal_disengage_on_contact,
        animal_visual_channel=animal_visual_channel,
        animal_visual_property=animal_visual_property,
        animal_visual_property_std=animal_visual_property_std,
        animal_classes=animal_classes,
        animal_behaviours=animal_behaviours,
        animal_tags=animal_tags,
        hunt_idx=hunt_idx,
        wander_idx=wander_idx,
        static_idx=static_idx,
        predator_indices=predator_indices,
        neutral_indices=neutral_indices,
        # Per-episode count-range metadata (NEW — PER_EPISODE_ENV_VARIANCE)
        res_count_low=jnp.array(res_count_low_list, dtype=jnp.int32) if res_count_low_list else jnp.zeros(0, dtype=jnp.int32),
        res_count_high=jnp.array(res_count_high_list, dtype=jnp.int32) if res_count_high_list else jnp.zeros(0, dtype=jnp.int32),
        res_entry_id=jnp.array(res_entry_id_list, dtype=jnp.int32) if res_entry_id_list else jnp.zeros(0, dtype=jnp.int32),
        animal_count_low=jnp.array(animal_count_low_list, dtype=jnp.int32) if animal_count_low_list else jnp.zeros(0, dtype=jnp.int32),
        animal_count_high=jnp.array(animal_count_high_list, dtype=jnp.int32) if animal_count_high_list else jnp.zeros(0, dtype=jnp.int32),
        animal_entry_id=jnp.array(animal_entry_id_list, dtype=jnp.int32) if animal_entry_id_list else jnp.zeros(0, dtype=jnp.int32),
        obs_count_low=jnp.array(obs_count_low_list, dtype=jnp.int32) if obs_count_low_list else jnp.zeros(0, dtype=jnp.int32),
        obs_count_high=jnp.array(obs_count_high_list, dtype=jnp.int32) if obs_count_high_list else jnp.zeros(0, dtype=jnp.int32),
        obs_entry_id=jnp.array(obs_entry_id_list, dtype=jnp.int32) if obs_entry_id_list else jnp.zeros(0, dtype=jnp.int32),
        has_res_range=any(lo < hi for lo, hi in zip(res_count_low_list, res_count_high_list)),
        has_animal_range=any(lo < hi for lo, hi in zip(animal_count_low_list, animal_count_high_list)),
        has_obs_range=any(lo < hi for lo, hi in zip(obs_count_low_list, obs_count_high_list)),
        obs_blocking=obs_blocking,
        obs_hides_agent=obs_hides_agent,
        obs_blocks_animals=obs_blocks_animals,
        obs_damage=obs_damage,
        obs_property=obs_property,
        obs_property_std=obs_property_std,
        obs_visual_property=obs_visual_property,
        obs_visual_property_std=obs_visual_property_std,
        obs_nociception=obs_nociception,
        obs_spawn_area=obs_spawn_area,
        obs_type=obs_type,
        obstacle_names=obstacle_names,
        type_areas=jnp.array(type_areas_np, dtype=jnp.int32),
        type_counts=jnp.array(type_counts_list, dtype=jnp.int32),
        type_entity_map=jnp.array(type_entity_map_np, dtype=jnp.int32),
        max_per_type=max_per_type,
        num_types=num_types,
        num_entities=num_total_entities,
        placement_mode=placement_mode,
        max_satiation=config.get_mandatory('body.max_satiation'),
        max_nutrition=config.get_mandatory('body.max_nutrition'),
        max_injury=config.get_mandatory('body.max_injury'),
        food_nutrition_gain=config.get_mandatory('body.food_nutrition_gain'),
        setpoint=config.get_mandatory('body.satiation_setpoint'),
        start_satiation=config.get_mandatory('body.start_satiation'),
        start_nutrition=config.get_mandatory('body.start_nutrition'),
        start_nutrition_low=start_nutrition_low,
        start_nutrition_high=start_nutrition_high,
        start_injury_low=start_injury_low,
        start_injury_high=start_injury_high,
        metabolic_cost=config.get_mandatory('body.metabolic_cost'),
        nutrition_to_satiation_scaling_factor=config.get_mandatory('body.nutrition_to_satiation_scaling_factor'),
        recovery_base_rate=config.get_mandatory('body.recovery_base_rate'),
        recovery_accel_rate=config.get_mandatory('body.recovery_accel_rate'),
        smoothing_duration=config.get_mandatory('body.injury_smoothing_duration'),
        death_penalty=config.get_mandatory('body.death_penalty'),
        overeating_death=config.get_mandatory('body.overeating_death'),
        use_homeostatic_reward=config.get_mandatory('body.use_homeostatic_reward'),
        with_satiation=config.get_mandatory('body.with_satiation'),
        with_nutrition=config.get_mandatory('body.with_nutrition'),
        with_injury=config.get_mandatory('body.with_injury'),
        random_start_satiation=config.get_mandatory('body.random_start_satiation'),
        random_start_nutrition=config.get_mandatory('body.random_start_nutrition'),
        random_start_injury=config.get_mandatory('body.random_start_injury'),
        random_start_pos=config.get_mandatory('environment.random_start_pos'),
        start_pos=jnp.array(config.get_mandatory('environment.start_pos')) - 1,
        rest_action_enabled=config.get_mandatory('environment.rest_action_enabled'),
        eat_action_enabled=config.get_mandatory('environment.eat_action_enabled'),
        eating_nutrition_cost=config.get_mandatory('body.eating_nutrition_cost'),
        eating_reward_penalty=config.get_mandatory('body.eating_reward_penalty'),
        sensor_radius=config.get_mandatory('sensory.sensor_radius'),
        sensor_decay=config.get_mandatory('sensory.decay_power'),
        sensor_range=config.get_mandatory('sensory.collision_sensor_range'),
        visual_sensor_enabled=config.get_mandatory('sensory.visual_sensor_enabled'),
        visual_sensor_range=config.get_mandatory('sensory.visual_sensor_range'),
        local_view_size=config.get_mandatory('visualization.local_view_size'),
        proprioception_enabled=config.get_mandatory('sensory.proprioception_enabled'),
        olfactory_enabled=config.get_mandatory('sensory.olfactory_enabled'),
        nociception_enabled=config.get_mandatory('sensory.nociception_enabled'),
        location_sensor_enabled=config.get_mandatory('sensory.location_sensor'),
        olfactory_vector_size=config.get_mandatory('sensory.vector_size'),
        visual_vector_size=visual_vector_size,
        visual_background_property=visual_background_property,
        nociception_size=config.get_mandatory('sensory.nociception_size'),
        action_dim=4 + int(config.get_mandatory('environment.rest_action_enabled')) + int(config.get_mandatory('environment.eat_action_enabled')),

        # Hidden-state observability flags
        injury_observable=injury_observable,
        nutrition_observable=nutrition_observable,

        # Interoceptive nociception (delayed-peak perception of hidden injury)
        interoceptive_nociception_enabled=interoceptive_nociception_enabled,
        interoceptive_convolution_enabled=interoceptive_convolution_enabled,
        interoceptive_kernel_length=interoceptive_kernel_length,
        interoceptive_kernel=interoceptive_kernel,

        # Perceptual Noise Configuration
        perceptual_noise_enabled=config.get('perceptual_noise.enabled', False),
        **_parse_noise_config(config)
    )

_YAML_KEY_TO_SENSOR_NAME = {
    "injury":                    "Injury",
    "nutrition":                 "Nutrition",
    "satiation":                 "Satiation",
    "extero_nociception":        "Extero Nociception",
    "interoceptive_nociception": "Interoceptive Nociception",
    "olfaction":                 "Olfaction",
    "collision":                 "Collision",
    "proprioception":            "Proprioception",
    "visual":                    "Visual",
    "location":                  "Location",
}

def _parse_noise_config(config: Config):
    modalities_cfg = config.get('perceptual_noise.modalities') or {}
    
    def _parse_mode(s):
        return 2 if s == 'state_dependent' else 1 if s == 'constant' else 0

    noise_modality_order = tuple(
        _YAML_KEY_TO_SENSOR_NAME[k]
        for k in modalities_cfg
        if k in _YAML_KEY_TO_SENSOR_NAME
    )
    pad = max(0, 13 - len(noise_modality_order))

    noise_modes = jnp.pad(jnp.array([
        _parse_mode(modalities_cfg[k].get('mode', 'none'))
        for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME
    ], dtype=jnp.int32), (0, pad))
    
    noise_sigmas = jnp.pad(jnp.array([
        modalities_cfg[k].get('sigma', 0.0)
        for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME
    ], dtype=jnp.float32), (0, pad))
    
    noise_injury_scales = jnp.pad(jnp.array([
        modalities_cfg[k].get('injury_noise_scale', 0.0)
        for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME
    ], dtype=jnp.float32), (0, pad))
    
    noise_clip_min = jnp.pad(jnp.array([
        modalities_cfg[k].get('clip_min', -100.0)
        for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME
    ], dtype=jnp.float32), (0, pad))
    
    noise_clip_max = jnp.pad(jnp.array([
        modalities_cfg[k].get('clip_max', 100.0)
        for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME
    ], dtype=jnp.float32), (0, pad))

    return {
        "noise_modality_order": noise_modality_order,
        "noise_modes": noise_modes,
        "noise_sigmas": noise_sigmas,
        "noise_injury_scales": noise_injury_scales,
        "noise_clip_min": noise_clip_min,
        "noise_clip_max": noise_clip_max,
    }
