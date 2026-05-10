"""
Configuration loader for JAX Environment.

Translates YAML config files into JAX-compatible EnvParams.
"""
import re as _re
import yaml
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple
from src.environment.state import EnvParams

from src.utils.config import Config
import warnings

# Allowed characters in entity tags (used in WandB key 'Episode/MeanDist*_<tag>').
# Slashes / spaces / dots break the WandB namespace or log key.
_TAG_RE = _re.compile(r'^[A-Za-z0-9_-]+$')

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

    # ----- validation -----
    if cue_radius <= 0:
        raise ValueError(f"behavior_measures.cue_radius must be > 0; got {cue_radius}.")
    if obs_window < 1:
        raise ValueError(f"behavior_measures.obs_window must be >= 1; got {obs_window}.")
    if eval_n_eps < 1:
        raise ValueError(f"behavior_measures.eval_n_episodes must be >= 1; got {eval_n_eps}.")
    if not isinstance(eval_seeds_raw, (list, tuple)):
        raise ValueError(f"behavior_measures.eval_seeds must be a list/tuple; got {type(eval_seeds_raw)}.")
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

def load_env_params(config: Config) -> EnvParams:
    """Loads environment parameters from a Config object with strict retrieval."""
    
    # Build resource arrays
    raw_resources = config.get_mandatory('environment.resources')
    expanded_resources = []
    if raw_resources:
        for r in raw_resources:
            count = r.get('count', 1) 
            for _ in range(count):
                expanded_resources.append(r)
    
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
    else:
        res_type = jnp.zeros(0, dtype=jnp.int32)
        res_property = jnp.zeros((0, 5))
        res_property_std = jnp.zeros((0, 5))
        res_nociception = jnp.zeros(0)
        res_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        res_max_cons = jnp.zeros(0, dtype=jnp.int32)
        res_reg_delay = jnp.zeros(0, dtype=jnp.int32)
        res_damage = jnp.zeros((0, 2))

    # Build predator arrays
    raw_predators = config.get_mandatory('environment.predators')
    expanded_predators = []
    if raw_predators:
        for p in raw_predators:
            count = p.get('count', 1)
            for _ in range(count):
                expanded_predators.append(p)
                
    if expanded_predators:
        def p_get(p, key):
            val = p.get(key)
            if val is None: raise ValueError(f"Strict Config: Predator field '{key}' is required.")
            return val

        pred_property = jnp.array([_read_properties(p, 'Predator') for p in expanded_predators])
        chem_dim = pred_property.shape[-1]
        pred_property_std = jnp.array([_read_properties_std(p, 'Predator') for p in expanded_predators])
        pred_nociception = jnp.array([p.get('nociception_intensity', 0.9) for p in expanded_predators])
        pred_move_int = jnp.array([p_get(p, 'move_interval') for p in expanded_predators], dtype=jnp.int32)
        
        # Predator damage ranges
        raw_p_damage = [p_get(p, 'damage') for p in expanded_predators]
        pred_damage = jnp.array([d if isinstance(d, list) else [d, d] for d in raw_p_damage])
        
        h = config.get_mandatory('environment.height')
        w = config.get_mandatory('environment.width')
        # Adjust for 0-based min and exclusive max
        pred_patrol = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [p.get('patrol_area', [[1,1],[h,w]]) for p in expanded_predators]])
        pred_spawn_area = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [p.get('spawn_area', [[1,1],[h,w]]) for p in expanded_predators]])
        pred_detect = jnp.array([p_get(p, 'detection_range') for p in expanded_predators])
        pred_max_stamina = jnp.array([p_get(p, 'max_stamina') for p in expanded_predators])
        pred_recovery = jnp.array([p_get(p, 'stamina_recovery_rate') for p in expanded_predators])
        pred_hunt_thresh = jnp.array([p_get(p, 'hunt_stamina_threshold') for p in expanded_predators])
        pred_attack_delay = jnp.array([p_get(p, 'attack_delay') for p in expanded_predators], dtype=jnp.int32)
        pred_lose_interest_mult = jnp.array([p.get('lose_interest_multiplier', 2.0) for p in expanded_predators], dtype=jnp.float32)
        predator_enabled = config.get_mandatory('environment.predator_enabled')
        predator_tags = tuple(
            _normalise_tag(p.get('tag'), i, 'Predator')
            for i, p in enumerate(expanded_predators)
        )
    else:
        pred_property = jnp.zeros((0, 5))
        pred_property_std = jnp.zeros((0, 5))
        pred_nociception = jnp.zeros(0)
        pred_move_int = jnp.zeros(0, dtype=jnp.int32)
        pred_damage = jnp.zeros((0, 2))
        pred_patrol = jnp.zeros((0, 4), dtype=jnp.int32)
        pred_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        pred_detect = jnp.zeros(0)
        pred_max_stamina = jnp.zeros(0)
        pred_recovery = jnp.zeros(0)
        pred_hunt_thresh = jnp.zeros(0)
        pred_attack_delay = jnp.zeros(0, dtype=jnp.int32)
        pred_lose_interest_mult = jnp.zeros(0, dtype=jnp.float32)
        predator_enabled = config.get_mandatory('environment.predator_enabled')
        predator_tags = tuple()

    # Build Obstacle arrays
    raw_obstacles = config.get_mandatory('environment.obstacles')
    expanded_obstacles = []
    for o in raw_obstacles:
        count = o.get('count', 1)
        for _ in range(count):
            expanded_obstacles.append(o)
            
    if expanded_obstacles:
        def obs_get(o, key):
            val = o.get(key)
            if val is None: raise ValueError(f"Strict Config: Obstacle field '{key}' is required.")
            return val
        obs_blocking = jnp.array([o.get('blocking', True) for o in expanded_obstacles], dtype=jnp.bool_)
        obs_hides_agent = jnp.array([o.get('hides_agent', False) for o in expanded_obstacles], dtype=jnp.bool_)
        
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
    else:
        obs_blocking = jnp.zeros(0, dtype=jnp.bool_)
        obs_hides_agent = jnp.zeros(0, dtype=jnp.bool_)
        obs_damage = jnp.zeros((0, 2), dtype=jnp.float32)
        obs_nociception = jnp.zeros(0, dtype=jnp.float32)
        chem_dim = res_property.shape[-1]
        obs_property = jnp.zeros((0, chem_dim))
        obs_property_std = jnp.zeros((0, chem_dim))
        obs_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        obs_type = jnp.zeros(0, dtype=jnp.int32)
        obstacle_names = ("rock",)
    
    # Build Neutral Animal arrays (Decoys)
    raw_neutral = config.get_mandatory('environment.neutral_animals')
    expanded_neutral = []
    if raw_neutral:
        for n in raw_neutral:
            count = n.get('count', 1)
            for _ in range(count):
                expanded_neutral.append(n)
                
    if expanded_neutral:
        def n_get(n, key):
            val = n.get(key)
            if val is None: raise ValueError(f"Strict Config: Neutral Animal field '{key}' is required.")
            return val
        
        neutral_property = jnp.array([_read_properties(n, 'Neutral Animal') for n in expanded_neutral])
        chem_dim = neutral_property.shape[-1]
        neutral_property_std = jnp.array([_read_properties_std(n, 'Neutral Animal') for n in expanded_neutral])
        neutral_nociception = jnp.array([n.get('nociception_intensity', 0.0) for n in expanded_neutral])
        neutral_move_int = jnp.array([n_get(n, 'move_interval') for n in expanded_neutral], dtype=jnp.int32)
        h = config.get_mandatory('environment.height')
        w = config.get_mandatory('environment.width')
        # Adjust for 0-based min and exclusive max
        neutral_patrol = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [n.get('patrol_area', [[1,1],[h,w]]) for n in expanded_neutral]])
        neutral_spawn_area = jnp.array([[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]] for a in [n.get('spawn_area', [[1,1],[h,w]]) for n in expanded_neutral]])
        neutral_tags = tuple(
            _normalise_tag(n.get('tag'), i, 'Rabbit')
            for i, n in enumerate(expanded_neutral)
        )
    else:
        neutral_property = jnp.zeros((0, 5))
        neutral_property_std = jnp.zeros((0, 5))
        neutral_nociception = jnp.zeros(0)
        neutral_move_int = jnp.zeros(0, dtype=jnp.int32)
        neutral_patrol = jnp.zeros((0, 4), dtype=jnp.int32)
        neutral_spawn_area = jnp.zeros((0, 4), dtype=jnp.int32)
        neutral_tags = tuple()

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
    
    # ── Type-Level Placement: Group entities by spawn area ──
    all_spawn_areas_np = np.concatenate([
        np.array(res_spawn_area), np.array(pred_spawn_area),
        np.array(obs_spawn_area), np.array(neutral_spawn_area)
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
    
    # Log placement strategy
    print("="*60)
    print(f"ENTITY PLACEMENT STRATEGY: {placement_mode}")
    print("="*60)
    print(f"Grid: {height}×{width} ({height*width} cells)")
    print(f"Total entities: {num_total_entities}")
    if placement_mode == 'per_type':
        print(f"Type groups: {num_types} (one lax.scan step each)")
        for tidx, k in enumerate(area_keys_list):
            area = list(k)
            cnt = type_counts_list[tidx]
            area_cells = (area[2]-area[0]) * (area[3]-area[1])
            print(f"  Group {tidx}: area {area} → {cnt} entities / {area_cells} cells ({100*cnt/area_cells:.0f}%)")
        print(f"Max entities per group: {max_per_type}")
        print(f"Sequential steps: {num_types}")
    else:
        print(f"Sequential steps: {num_total_entities} (one per entity)")
    print("="*60)
    
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

    return EnvParams(
        height=height,
        width=width,
        max_steps=config.get_mandatory('environment.max_steps'),
        grid_location_type=grid_location_type,
        res_type=res_type,
        res_property=res_property,
        res_property_std=res_property_std,
        res_nociception=res_nociception,
        res_spawn_area=res_spawn_area,
        res_max_cons=res_max_cons,
        res_reg_delay=res_reg_delay,
        res_damage=res_damage,
        pred_property=pred_property,
        pred_property_std=pred_property_std,
        pred_nociception=pred_nociception,
        pred_move_int=pred_move_int,
        pred_damage=pred_damage,
        pred_patrol=pred_patrol,
        pred_detect=pred_detect,
        pred_max_stamina=pred_max_stamina,
        pred_recovery=pred_recovery,
        pred_hunt_thresh=pred_hunt_thresh,
        pred_attack_delay=pred_attack_delay,
        pred_lose_interest_mult=pred_lose_interest_mult,
        predator_enabled=predator_enabled,
        pred_spawn_area=pred_spawn_area,
        predator_tags=predator_tags,
        obs_blocking=obs_blocking,
        obs_hides_agent=obs_hides_agent,
        obs_damage=obs_damage,
        obs_property=obs_property,
        obs_property_std=obs_property_std,
        obs_nociception=obs_nociception,
        obs_spawn_area=obs_spawn_area,
        obs_type=obs_type,
        obstacle_names=obstacle_names,
        neutral_property=neutral_property,
        neutral_property_std=neutral_property_std,
        neutral_nociception=neutral_nociception,
        neutral_move_int=neutral_move_int,
        neutral_patrol=neutral_patrol,
        neutral_spawn_area=neutral_spawn_area,
        neutral_tags=neutral_tags,
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
