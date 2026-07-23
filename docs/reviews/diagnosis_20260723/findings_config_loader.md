# Diagnosis findings — src/environment/config_loader.py (+ src/utils/config.py)

Date: 2026-07-23. Reviewer: config-loader deep-diagnosis pass.
Severity scale: P0 = corrupts runs now / P1 = wrong results in realistic configs / P2 = latent-quality.
All claims verified by reading the actual code path and, where noted, by executing read-only checks
with `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` against real configs.

---

## Finding 1 — P1: base `entities:` in default.yaml silently clobbers legacy `predators:`/`neutral_animals:` configs under train.py

- **Where**: `src/environment/config_loader.py:429-443` (`has_entities` precedence in `_load_animals`), interacting with `train.py:306` (`get_default_config()` underlay) and `configs/environment/default.yaml:58` (which since v3.0 declares a real `environment.entities:` list).
- **Claim**: Any pre-v2.0 config that describes its scene with the legacy `predators:` / `neutral_animals:` schema gets its animals **silently replaced by default.yaml's entity list** when run through `train.py` (and `dreamer_srl_main.py`, which underlays the same default). `_load_animals` sees `entities` (from the merged-in default.yaml base) as non-None, takes the entities path, and ignores the user's legacy sections — emitting only a `DeprecationWarning` whose text ("Config has both...") is misleading, because the *user's file* does not have both; the merge created the conflict.
- **Concrete failure scenario**: Re-running the archived experiment `configs/environment/experiment/archive/2X2_area.yaml` (2 predators in specific quadrants) via `train.py --config ...` trains on default.yaml's scene (1-2 full-grid predators + 1-2 rabbits) instead. Worse, the mixed merge is incoherent: the user's `resources:`/`obstacles:` lists **do** win (list-replace semantics), so the run is user resources + default animals — a scene nobody authored. 81 config files under `configs/environment` (all in `archive/`) carry `predators:`; every one of them is affected on rerun.
- **Entry-point divergence**: `scripts/eval/eval_rollout.py:849` loads via bare `load_env_config(config_path)` with **no** default underlay → the same file evaluates with its *true* legacy scene. So train and eval disagree about what scene a legacy config describes.
- **Evidence** (executed): `get_default_config(); cfg.merge(load_env_config('configs/environment/experiment/archive/2X2_area.yaml')); _load_animals(cfg)` → `animal_tags == ('pred','pred','rabbit','rabbit')` (default.yaml's entities), merged config holds *both* `entities` (2, from base) and `predators` (2, from user); only a DeprecationWarning fired.
- **Fix direction**: escalate the both-schemas case to `ValueError` when the *user-supplied* file (pre-merge) carries legacy sections while the merged result carries `entities:` — or make `_load_animals` prefer the schema present in the user file; at minimum promote the warning to an error per the no-fallback rule.

## Finding 2 — P1: unknown perceptual-noise `mode` strings silently map to 0 (= no noise)

- **Where**: `src/environment/config_loader.py:1509-1510` (`_parse_mode`).
- **Claim**: `_parse_mode` returns `2` for `'state_dependent'`, `1` for `'constant'`, and **0 (noise off) for anything else** — including typos and unknown values — with no error.
- **Concrete failure scenario**: A noise-experiment config writes `mode: "state-dependent"` (hyphen) or `mode: "gaussian"`; the run trains with that modality's noise silently disabled. The whole experiment cell measures nothing, and the saved config looks like noise was configured.
- **Evidence** (executed): `_parse_noise_config` on `{'injury': {'mode': 'state-dependent', 'sigma': 0.5}}` → `noise_modes[0] == 0`.
- **Fix direction**: validate against `{'none','constant','state_dependent'}` and raise `ValueError` on anything else.

## Finding 3 — P1: unknown modality keys under `perceptual_noise.modalities` are silently dropped

- **Where**: `src/environment/config_loader.py:1512-1542` (every comprehension filters `if k in _YAML_KEY_TO_SENSOR_NAME`).
- **Claim**: A modality key not in `_YAML_KEY_TO_SENSOR_NAME` (e.g. `olfactory:` instead of `olfaction:`, `visual_noise:`, `nociception:`) is filtered out of all five noise arrays without any warning or error.
- **Concrete failure scenario**: User adds `olfactory: {mode: constant, sigma: 0.2}` to enable smell noise; the key is dropped; the base default.yaml `olfaction` entry (sigma 0.0) survives the deep merge; training runs noise-free on that channel. Silently wrong results in exactly the class of experiment this project runs (obs↔noise sync is already a named audit concern).
- **Evidence** (executed): same run as Finding 2 — modality dict with keys `{injury, olfactory}` produced `noise_modality_order == ('Injury',)`; `olfactory` vanished.
- **Fix direction**: raise `ValueError` listing allowed keys when `set(modalities_cfg) - set(_YAML_KEY_TO_SENSOR_NAME)` is non-empty.

## Finding 4 — P2: silent fallback defaults throughout the noise block

- **Where**: `src/environment/config_loader.py:1489` (`config.get('perceptual_noise.enabled', False)`), `:1520` (`.get('mode','none')`), `:1525` (`.get('sigma', 0.0)`), `:1530` (`.get('injury_noise_scale', 0.0)`), `:1535/:1540` (`clip_min/max` ±100).
- **Claim**: Every noise leaf has a read-site default, contra the no-fallback rule. `enabled` missing → noise silently off; `sigma` missing under `mode: constant` → noise silently zero.
- **Concrete failure scenario**: A standalone config (bare `load_env_config` path, e.g. a probe/eval load without the default.yaml underlay) declares a `perceptual_noise.modalities` block but omits `enabled: true` → all noise silently off. Under train.py the default.yaml underlay masks most of these, which is why it is P2 not P1.
- **Fix direction**: when the `perceptual_noise` block is present, make `enabled` and per-modality `mode`/`sigma` mandatory (mirror `load_behavior_measure_cfg`'s block-present ⇒ all-leaves-mandatory pattern).

## Finding 5 — P2: `nociception_intensity` silent default differs by schema path — 0.0 for `entities:` predators vs 0.9 for legacy predators

- **Where**: `src/environment/config_loader.py:470` (entities path: `ent.get('nociception_intensity', 0.0)`) vs `:503` (legacy predator: `p.get('nociception_intensity', 0.9)`); siblings at `:535` (neutral 0.0), `:1011` (resource 0.9/0.0 by type), `:1137` (obstacle 0.3).
- **Claim**: The same conceptual entity (a predator) gets nociception 0.9 by default under the legacy schema but **0.0** under the unified `entities:` schema. Nociception is this project's core research variable; a silently-zeroed predator pain signal is a construct-invalidating config error. Additionally the obstacle default of 0.3 means an undamaging rock still carries a nonzero pain signature by silent default.
- **Concrete failure scenario**: Author migrates a legacy config to `entities:` and drops the `nociception_intensity` line (it "had a default before") → predator contact produces zero nociceptive input; agent behaviour analyses of pain responses are measuring nothing. Verified all *current* live predator-entity configs do carry the key explicitly, so this is latent (P2), not active.
- **Fix direction**: make `nociception_intensity` mandatory for `class: predator` (and arguably for all entities), per the no-fallback rule.

## Finding 6 — P2: `get_mandatory` type handling — TypeError leaks, and most numeric env keys are read with no converter (PyYAML string-float trap)

- **Where**: `src/utils/config.py:76-80`; consumers at `src/environment/config_loader.py:1342-1476` (the bulk of `load_env_params` `get_mandatory` calls pass no `type_converter`).
- **Claim**: (a) The converter wrapper catches only `ValueError`; `float([1,2])` raises a bare `TypeError` that escapes without the "Strict Config" framing (verified). (b) PyYAML parses `1e-4`, `1e5`, and even `1.0e4` (no exponent sign) as **strings** (verified: `yaml.safe_load("v: 1e-4") == '1e-4'`). Keys read without a converter (`metabolic_cost`, `death_penalty`, `max_steps`, ...) pass such strings straight into `EnvParams`, failing later at JAX trace time with an error far from the config, or — worse — silently: a quoted boolean (`with_injury: "false"`) survives `get_mandatory` and the call-site `bool(...)` casts at `:1285-1290` turn `"false"` into `True`.
- **Concrete failure scenario**: `death_penalty: -1e2` in a hand-edited YAML → string → cryptic trace-time crash; `injury_observable: "false"` (e.g. from a templating script) → observability silently ON.
- **Fix direction**: catch `TypeError` too in `get_mandatory`; pass explicit converters (and a strict bool parser) at every numeric/bool read site in `load_env_params`.

## Finding 7 — P2: `Config.merge`/`deep_update` — crash on scalar←dict override, and list-reference aliasing with the source config

- **Where**: `src/utils/config.py:93-101`.
- **Claim**: (a) If the base holds a scalar/None at a key and the override supplies a dict, `deep_update(scalar, dict)` raises a bare `TypeError: 'int' object does not support item assignment` (verified) — the real failure (schema mismatch between layers) is swallowed behind an unrelated message. (b) Non-dict values — including **lists** — are assigned by reference (`d[k] = v`), so the merged config shares list objects with the source dict; mutating one mutates the other (verified: appending to the merged config's list changed the source dict). No config object is currently cached cross-run, so this is latent, but any future in-place mutation of e.g. an `entities` list would corrupt sibling configs (continual stages are protected only by train.py's explicit YAML round-trip deep copy at `train.py:197`).
- **Fix direction**: raise a descriptive `ValueError` when layer types conflict; `copy.deepcopy(v)` on assignment in `deep_update`.

## Finding 8 — P2: unknown `location_areas` type silently becomes plain terrain; missing `area` silently skipped

- **Where**: `src/environment/config_loader.py:1186-1193`.
- **Claim**: `type_idx = 1 if 'grass' else 2 if 'sand' else 0` — any other string (typo `Grass`, future `water`) silently maps to plain(0); an entry without `area` is silently ignored (`if area:`).
- **Concrete failure scenario**: `type: "Grass"` capitalization → the whole grass region silently becomes plain; background visual/location observations differ from the intended design with no error.
- **Fix direction**: validate `type` against `{'grass','sand','plain'}` and make `area` mandatory per entry.

## Finding 9 — P2: `placement.mode` has a silent read-site default and is guarded by `assert`

- **Where**: `src/environment/config_loader.py:1260-1261`.
- **Claim**: `config.get('environment.placement.mode', 'per_entity')` is a silent fallback on a perf-relevant key, and the validity check is an `assert` (stripped under `python -O`) rather than the project-standard `ValueError`.
- **Concrete failure scenario**: `mode: per-type` (hyphen) → AssertionError with no allowed-values list; under `-O`, the typo'd string would flow into `EnvParams.placement_mode` and downstream string comparisons silently pick per-entity.
- **Fix direction**: `get_mandatory`-style read (or documented default) + `ValueError` with the allowed set.

## Finding 10 — P2: `count_high`-only entries silently default `count_low` to 0

- **Where**: `src/environment/config_loader.py:944-952` (`_resolve_count_range`).
- **Claim**: Supplying only `count_high: 5` yields `(0, 5)` silently (verified) — the entity can vanish entirely some episodes without the author ever writing `count_low: 0`. Asymmetric: `count_low`-only errors out (via the `0 <= lo <= hi` check with `hi=0`), so one half-specified form is loud and the other silent.
- **Fix direction**: require both keys when either is present.

## Finding 11 — P2: dead/confused branch in the animal properties comprehension

- **Where**: `src/environment/config_loader.py:619`.
- **Claim**: `_read_properties(e, e['tag_label']) if 'property' in e and not isinstance(e.get('property'), list) else e['property']` — every entry dict always has `'property'` (set during entry construction), and when it is a non-list the fallback calls `_read_properties` **on the normalized entry dict** (which has key `'property'`, not `'properties'`), triggering the deprecation-warning path and returning the same non-list value. The branch is a no-op that can only emit a spurious deprecation warning; a genuinely non-list `properties` value then crashes at `chem_dim = len(props_list[0])` (line 621) with a bare `TypeError`.
- **Fix direction**: simplify to `[e['property'] for e in entries]` and validate list-ness where the entry is built.

## Finding 12 — P2: `extends:` in agent configs is silently ignored (entry-point asymmetry for the same syntax)

- **Where**: `train.py:397` and `src/algorithms/dreamer_srl/dreamer_srl_main.py:541` load `--agent_config` with plain `Config.load_yaml`; only env configs go through `load_env_config`.
- **Claim**: If an agent config ever declares `extends:`, nothing resolves it and nothing rejects it — the key sits in the merged config as dead data and the intended base layers are silently absent. This is the exact failure shape of the FIXED "extends ignored by train.py" bug, one file class over. Verified no agent/train/eval config currently uses `extends:` (repo-wide grep), so latent.
- **Fix direction**: route agent configs through `load_env_config` (rename appropriately) or raise on a present-but-unresolved `extends:` key in `Config.load_yaml` consumers.

## Finding 13 — P2: no cross-validation of `sensory.vector_size` against actual property-vector lengths

- **Where**: `src/environment/config_loader.py:1472` (`olfactory_vector_size` read) vs property arrays built at `:999-1001` (resources), `:621` (animals `chem_dim`), `:1140-1141` (obstacles).
- **Claim**: The loader never checks that resource/animal/obstacle `properties` lists share one length, nor that this length equals `sensory.vector_size`. Mismatches surface (if at all) as shape errors deep in JAX tracing, or as an inconsistent zero-animal placeholder (`chem_dim=5` hard-coded at `:549` regardless of resource dim).
- **Fix direction**: after building all property arrays, assert one common `chem_dim` and compare against `sensory.vector_size` with a plain-English `ValueError`.

---

## Fixed-bug regression check

- **Typo'd `--config` silently using default — FIX PRESENT.** `src/utils/config.py:30-39`: `Config.load_yaml` raises `FileNotFoundError` on a missing path (with the H3 rationale comment). All entry points (`train.py:389`, `dreamer_srl_main.py:539`, `eval_rollout.py:849`) route user paths through it. Optional layer merges guard with `os.path.exists` as documented — note (mild) that a *deleted* `configs/train/default.yaml` would silently skip merging that layer (`train.py:310-352`), relying on downstream `get_mandatory` to catch missing keys.
- **`extends:` ignored by train.py loader — FIX PRESENT, all entry points consistent.** Every env-config load site resolves `extends:` through the same `load_env_config`/`_resolve_extends`: `train.py:389` (single run), `train.py:198` (continual stages), `dreamer_srl_main.py:539` (single) and `_load_stage_env_cfg` at `dreamer_srl_main.py:133` (continual), `eval_rollout.py:849` and probe path `:753`. `_CONFIGS_ROOT` (config_loader.py:28-30) is CWD-independent. Child-over-parent merge order and cycle detection verified correct. Caveat: entry points still differ in what they layer *underneath* the resolved config (train/dreamer underlay `get_default_config()` + train/eval/vis defaults; eval_rollout does not) — that divergence is what activates Finding 1. Agent configs remain outside the `extends:` machinery (Finding 12).
- **Fractional attack/detection range inclusive-integer fix — FIX PRESENT.** Whole-number bound rejection at `config_loader.py:700-709` (`attack_range`) and `:741-747` (`detection_range`).

## Reviewed but clean

- `_resolve_extends`: cycle detection (per-branch `_seen`), declared-order base merging, `extends` meta-key stripping, missing-target `ValueError`, diamond-inheritance behavior (double-merge is idempotent).
- Documented list-replace merge semantics (`load_env_config` docstring) — matches actual `deep_update` behavior.
- `load_behavior_measure_cfg`: block-absent → None (documented backwards-compat), block-present → every leaf mandatory; seed-spec `rng` mandatory; duplicate/length/enum validation all loud.
- Conditional-mandatory `body.start_nutrition/injury_low/high` (Fork B2) with sentinel values when the matching flag is off.
- `visual_vector_size` read-site default of 8 — explicitly the one permitted default (documented), with strict `visual_properties` requirements at V≠8 for entities, resources, obstacles, and background table.
- Count expansion parity: the `_load_animals` slot expansion and the `load_env_params` re-read (`:1074-1106`) use the same `_resolve_count_range`; legacy neutral entry-id offset correct.
- `_normalise_tag` regex rejection of WandB-breaking tags; empty→`idxN` documented optional default.
- Noise application maps modalities **by name** (`sensor.py:253`), so YAML dict order is not load-bearing at consumption; an enabled-but-missing modality fails loudly (KeyError at trace).
- Continual stage configs deep-copied via YAML round-trip (`train.py:197`) — no cross-stage aliasing.
- `_parse_area` whole-grid default for animal spawn/patrol areas — intentional (`None` handled by design); resources/obstacles keep their areas mandatory.
- Interoceptive kernel: non-positive-sum guard; zeros kernel in passthrough mode keeps pytree shape static.
- `predator_enabled` v2.0 removal guard (`config_loader.py:1044-1049`) — loud with migration guidance.
