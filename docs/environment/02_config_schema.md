# 02 — Config Schema

> **Source**: `src/environment/config_loader.py`, `src/environment/state.py`, `configs/environment/default.yaml` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview — what this document is about

This document describes how a YAML configuration file is translated into the typed `EnvParams` data structure that the JAX-based GridWorld uses at runtime. The translation is performed by `load_env_params(config)` in `src/environment/config_loader.py`.

**Why this matters for a new reader:** The environment is configured entirely through YAML. The loader enforces a strict "no silent failure" policy: any field declared mandatory raises `ValueError` if absent. Several fields that existed in v1.x have been removed or renamed (most notably, the old separate `predators:` and `neutral_animals:` sections have been merged into a single "unified animal entity" system). If your config was written before v2.0, read the migration notes in the entity section below.

`EnvParams` is a JAX Flax struct used inside JIT-compiled functions. Fields marked `pytree_node=False` (called **static** below) are baked into the compiled function — changing them triggers a full JIT recompilation.

---

## Transformation Pipeline (v2.0)

`load_env_params(config: Config) → EnvParams` (`config_loader.py:614`) executes in this order:

1. **Guard** — if `environment.predator_enabled` is present, raise `ValueError` immediately (`config_loader.py:666`). This key was removed in v2.0.
2. **Resources** — parse `environment.resources` list, expand each entry by its `count` field, build `res_*` arrays (`config_loader.py:617–661`).
3. **Unified animals** — call `_load_animals(config)` (`config_loader.py:673–689`), which detects whether the config uses the legacy dual-section schema (`environment.predators:` + `environment.neutral_animals:`) or the new unified `environment.entities:` schema, then expands, validates, and builds all `animal_*` arrays. See §Unified Animal Entity for details.
4. **Obstacles** — parse `environment.obstacles` list, expand by `count`, build `obs_*` arrays (`config_loader.py:692–732`).
5. **Location grid** — parse `environment.location_areas` list → fill `grid_location_type [height, width]` numpy array with type codes (0: plain, 1: grass, 2: sand) (`config_loader.py:735–748`).
6. **Placement grouping** — concatenate spawn areas in `[resources | predator-class animals | obstacles | neutral-class animals]` order, group by identical spawn-area bounding box, build `type_areas`, `type_counts`, `type_entity_map`, `max_per_type`, `num_types`, `num_entities` (`config_loader.py:751–800`).
7. **Interoceptive sensor** — read `sensory.injury_observable`, `sensory.nutrition_observable`, `sensory.interoceptive_*` keys; build normalized alpha kernel if convolution is enabled (`config_loader.py:803–826`).
8. **Noise config** — call `_parse_noise_config(config)` → build five noise arrays of length 13 (`config_loader.py:957–1002`).
9. **Construct `EnvParams`** — assemble all arrays and scalars into the Flax struct (`config_loader.py:828–941`).

---

## YAML Top-Level Structure

```
environment:
  height, width, max_steps
  start_pos, random_start_pos
  rest_action_enabled, eat_action_enabled
  placement.mode                       # "per_entity" | "per_type"
  resources: [ list of resource defs ]
  predators: [ list — LEGACY; auto-projected as class=predator, behaviour=hunt ]
  neutral_animals: [ list — LEGACY; auto-projected as class=neutral, behaviour=wander ]
  entities: [ list — NEW unified schema (CP3); takes precedence when present ]
  obstacles: [ list of obstacle defs ]
  location_areas: [ list of area type defs ]
  # NOTE: predator_enabled was REMOVED in v2.0 — raises ValueError if present

body:
  max_satiation, max_nutrition, max_injury
  metabolic_cost, food_nutrition_gain, eating_nutrition_cost, eating_reward_penalty
  nutrition_to_satiation_scaling_factor, satiation_setpoint
  start_satiation, start_nutrition
  recovery_base_rate, recovery_accel_rate, injury_smoothing_duration
  use_homeostatic_reward, death_penalty, overeating_death
  with_satiation, with_nutrition, with_injury
  random_start_satiation, random_start_nutrition, random_start_injury

sensory:
  olfactory_enabled, sensor_radius, vector_size, decay_power
  collision_sensor_range
  nociception_enabled, nociception_size
  visual_sensor_enabled, visual_sensor_range
  proprioception_enabled
  location_sensor
  injury_observable, nutrition_observable
  interoceptive_nociception_enabled, interoceptive_convolution_enabled
  interoceptive_kernel_tau, interoceptive_kernel_length

visualization:
  local_view_size

perceptual_noise:
  enabled: bool
  modalities:
    injury / nutrition / satiation / interoceptive_nociception /
    extero_nociception / olfaction / collision / proprioception /
    visual / location:
      mode, sigma, injury_noise_scale, clip_min, clip_max

behavior_measures:           # optional block; absent → feature off
  enabled, cue_radius, obs_window, eval_n_episodes, eval_seeds
  eval_policy_mode, eval_max_steps, eval_obs_noise
  motif_window_K, motif_features, motif_kmeans_k, motif_kmeans_seed
  motif_standardise, eval_output_root
```

---

## Integer Encoding Maps (v2.0 constants)

Defined at `config_loader.py:32–41` and used throughout the animal-entity builder.

| Concept | String | Integer | Field(s) in EnvParams |
|---------|--------|---------|----------------------|
| Animal class | `"predator"` | `0` | `animal_classes_int` |
| Animal class | `"neutral"` | `1` | `animal_classes_int` |
| Animal behaviour | `"wander"` | `0` | `animal_behaviours_int` |
| Animal behaviour | `"hunt"` | `1` | `animal_behaviours_int` |
| Animal behaviour | `"static"` | `2` | `animal_behaviours_int` |
| Visual channel | predator class | `5` | `animal_visual_channel` |
| Visual channel | neutral class | `7` | `animal_visual_channel` |
| Damaging classes | `{"predator"}` | derived bool | `animal_is_damaging` |
| Resource type | food | `0` | `res_type` |
| Resource type | hiding\_predator | `1` | `res_type` |
| Location area | plain (default) | `0` | `grid_location_type` |
| Location area | grass | `1` | `grid_location_type` |
| Location area | sand | `2` | `grid_location_type` |

---

## Unified Animal Entity (v2.0 refactor)

The old separate `predators:` + `neutral_animals:` YAML sections that produced distinct `pred_*` and `neutral_*` arrays **no longer exist at the array level**. They are replaced by a single unified `animal_*` array family. `_load_animals()` (`config_loader.py:202`) handles both the old (legacy) and new schema automatically.

### Two supported YAML schemas

**Schema A — Legacy (all pre-v2.0 configs)**

```yaml
environment:
  predators:
    - name: "predator"
      count: 1
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.4, 0.4, 0.0, 0.0]
      move_interval: 1
      damage: [15.0, 45.0]
      nociception_intensity: 0.9
      spawn_area: [[1,1],[10,10]]
      patrol_area: [[1,1],[10,10]]
      detection_range: 5          # mandatory for predator/hunt
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      attack_delay: 3
      lose_interest_multiplier: 1.5
  neutral_animals:
    - name: "rabbit"
      count: 1
      properties: [0.0, 0.5, 0.7, 0.0, 0.0]
      properties_std: [0.0, 0.4, 0.4, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.1
      spawn_area: [[1,1],[5,5]]
      patrol_area: [[1,1],[5,5]]
```

Legacy predators are re-projected as `class='predator'`, `behaviour='hunt'`. Legacy neutrals are re-projected as `class='neutral'`, `behaviour='wander'`. The five distributional fields (`detection_range` etc.) are read as scalars from legacy entries and stored as degenerate ranges `[s, s]`. Legacy neutrals never carried `damage`/`attack_delay` — the loader auto-fills `[0.0, 0.0]` and `0` internally (NC-1 fix; not a user-facing default).

**Schema B — Unified (CP3 and later)**

```yaml
environment:
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "alpha"
      count: 1
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.4, 0.4, 0.0, 0.0]
      move_interval: 1
      damage: [15.0, 45.0]
      nociception_intensity: 0.9
      attack_delay: 3
      spawn_area: [[1,1],[10,10]]
      patrol_area: [[1,1],[10,10]]
      detection_range: [3, 7]     # distributional: [low, high]
      max_stamina: [20, 40]
      stamina_recovery_rate: [0.5, 1.5]
      hunt_stamina_threshold: [0.5, 0.9]
      lose_interest_multiplier: [1.0, 2.0]
    - class: "neutral"
      behaviour: "wander"
      tag: "rabbit"
      count: 2
      ...
```

If `environment.entities:` is present it takes precedence; legacy sections are ignored with a `DeprecationWarning`.

### Mandatory / optional per entity

| Field | `behaviour: hunt` | `behaviour: wander` / `static` |
|-------|--------------------|-------------------------------|
| `class` | **mandatory** | **mandatory** |
| `behaviour` | **mandatory** | **mandatory** |
| `move_interval` | **mandatory** | **mandatory** |
| `damage` | **mandatory** | **mandatory** (unified schema); auto-filled for legacy neutrals |
| `attack_delay` | **mandatory** | **mandatory** (unified schema); auto-filled for legacy neutrals |
| `properties` | **mandatory** | **mandatory** |
| `properties_std` | **mandatory** | **mandatory** |
| `detection_range` | **mandatory** | optional (auto-filled `[0,0]`) |
| `max_stamina` | **mandatory** | optional (auto-filled `[0,0]`) |
| `stamina_recovery_rate` | **mandatory** | optional (auto-filled `[0,0]`) |
| `hunt_stamina_threshold` | **mandatory** | optional (auto-filled `[0,0]`) |
| `lose_interest_multiplier` | **mandatory** | optional (auto-filled `[0,0]`) |
| `nociception_intensity` | optional (default `0.9`) | optional (default `0.0`) |
| `spawn_area` | optional (default full grid) | optional (default full grid) |
| `patrol_area` | optional (default full grid) | optional (default full grid) |
| `tag` | optional (default `idx{i}`) | optional (default `idx{i}`) |
| `count` | optional (default `1`) | optional (default `1`) |

### Distributional fields (per-episode sampling)

The five fields in `DISTRIBUTIONAL_FIELDS` (`config_loader.py:36–41`) can be a scalar or a `[low, high]` list. A scalar `s` is stored as `[s, s]` (deterministic). At each episode reset, a value is sampled uniformly from `[low, high]` for each animal individually.

| YAML key | `_low` array | `_high` array | Cadence |
|----------|-------------|--------------|---------|
| `detection_range` | `animal_detect_low` | `animal_detect_high` | per-episode |
| `max_stamina` | `animal_max_stamina_low` | `animal_max_stamina_high` | per-episode |
| `stamina_recovery_rate` | `animal_recovery_low` | `animal_recovery_high` | per-episode |
| `hunt_stamina_threshold` | `animal_hunt_thresh_low` | `animal_hunt_thresh_high` | per-episode |
| `lose_interest_multiplier` | `animal_lose_interest_low` | `animal_lose_interest_high` | per-episode |

`damage` is separately per-event (re-sampled on every collision hit) and uses `[lo, hi]` format stored in `animal_damage [N, 2]`.

### Index dispatch tuples (static)

After expansion the loader builds index tuples that allow `jax_reset` and `update_animals` to slice behaviour/class subsets without dynamic shapes:

| Tuple field | Content | Use |
|-------------|---------|-----|
| `hunt_idx` | global indices where `behaviour == 'hunt'` | hunt-specific update loop |
| `wander_idx` | global indices where `behaviour == 'wander'` | wander update loop |
| `static_idx` | global indices where `behaviour == 'static'` | static update (no-op) |
| `predator_indices` | global indices where `class == 'predator'` | placement, damage masking |
| `neutral_indices` | global indices where `class == 'neutral'` | placement |

All five are `pytree_node=False` (static).

### Canonical entity concat order for placement

`pred_spawn_area_for_placement` and `neutral_spawn_area_for_placement` are derived from `animal_spawn_area` by indexing with `predator_indices` / `neutral_indices` (`config_loader.py:587–594`). The global placement array is then built as:

```
all_spawn_areas = concat([res_spawn_area, pred_spawn_area_for_placement,
                          obs_spawn_area, neutral_spawn_area_for_placement])
```

This `[resources | predator-class | obstacles | neutral-class]` order determines each entity's **global index** in `type_entity_map` and must match `jax_reset`'s `all_positions` split.

### Legacy `predator_tags` / `neutral_tags`

These fields were removed from `EnvParams` as struct fields (M1/M2 fix). They are now derived via `@property` accessors on the struct (`state.py:234–247`) that filter `animal_tags` by class. Existing consumer code (e.g. `dreamer_srl_main.py:522–523`) continues to work until the accessors are removed in a future release.

---

## Entity Count Expansion

Each entity definition in YAML carries an optional `count` field (default `1`). The loader expands each definition `count` times into a flat list before building JAX arrays. `count: 0` produces zero expansions — a clean way to disable an entity kind without deleting its YAML block.

When an entity list is empty (e.g. no animals configured), the loader builds zero-size arrays of the correct dtype and shape so `jax.vmap` does not need entity-count conditionals.

---

## Resource Entity Fields

Each entry under `environment.resources` (after `count` expansion):

| YAML key | EnvParams field | Mandatory | Notes |
|----------|----------------|-----------|-------|
| `type` | `res_type [N]` int | yes | `"food"` → 0, `"hiding_predator"` → 1; `"danger"` → DeprecationWarning then 1 |
| `properties` | `res_property [N, V]` | yes | olfactory chemical signature |
| `properties_std` | `res_property_std [N, V]` | yes | std dev for per-episode sampling |
| `spawn_area` | `res_spawn_area [N, 4]` | yes | 1-based → 0-based exclusive |
| `max_consumption` | `res_max_cons [N]` | yes | steps until resource depletes |
| `regeneration_delay` | `res_reg_delay [N]` | yes | steps before respawn |
| `damage` | `res_damage [N, 2]` | yes | `[lo, hi]` per-event; scalar → `[s, s]` |
| `nociception_intensity` | `res_nociception [N]` | optional | default `0.9` for hiding\_predator/danger, `0.0` for food |
| `count` | (expansion only) | optional | default `1` |

---

## Obstacle Entity Fields

Each entry under `environment.obstacles` (after `count` expansion):

| YAML key | EnvParams field | Mandatory | Default | Notes |
|----------|----------------|-----------|---------|-------|
| `area` | `obs_spawn_area [N, 4]` | yes | — | 1-based → 0-based exclusive |
| `properties` | `obs_property [N, V]` | yes | — | olfactory signature |
| `properties_std` | `obs_property_std [N, V]` | yes | — | std dev |
| `blocking` | `obs_blocking [N]` bool | optional | `True` | blocks movement |
| `hides_agent` | `obs_hides_agent [N]` bool | optional | `False` | bush-type concealment |
| `damage` | `obs_damage [N, 2]` | optional | `0.0` → `[0,0]` | per-event damage range |
| `nociception_intensity` | `obs_nociception [N]` | optional | `0.3` | |
| `name` | `obs_type [N]` int | optional | `"rock"` | index into `obstacle_names` |
| `count` | (expansion only) | optional | `1` | |

`obstacle_names` is the **sorted unique** tuple of all obstacle `name` values (`config_loader.py:720`). Renaming an obstacle can shift its index — don't hardcode indices outside the config.

---

## Placement Configuration

Controlled by `environment.placement.mode` (`config_loader.py:779`):

| Mode | Description | Best for |
|------|-------------|----------|
| `per_entity` | vmap sample per entity → `resolve_overlaps_global` sequential scan | Small grids (≤100 cells) |
| `per_type` | `lax.scan` over spawn-area groups via `place_in_area` | Large grids or high entity density |

`placement_mode` is a **static** field — changing it forces JIT recompilation.

**Type group construction** (always built, regardless of mode): entities are grouped by their spawn-area bounding box. Entities with identical bounding boxes share a group, across all entity kinds.

| EnvParams field | Shape | Static | Description |
|----------------|-------|--------|-------------|
| `type_areas` | `[T, 4]` | no | one bounding box per group (0-based exclusive) |
| `type_counts` | `[T]` | no | entity count per group |
| `type_entity_map` | `[T, max_per_type]` | no | global entity indices (0-padded) |
| `max_per_type` | scalar | **yes** | maximum entities in any single group |
| `num_types` | scalar | **yes** | number of groups T |
| `num_entities` | scalar | **yes** | total entity count across all types |

---

## Coordinate Convention and Area Parsing

YAML area specifications use **1-based inclusive coordinates** in the format `[[row_min, col_min], [row_max, col_max]]`. The helper `_parse_area()` (`config_loader.py:241–243`) converts these to **0-based with exclusive upper bound** for JAX's `jax.random.randint`:

```python
[a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]]
```

Example: YAML `[[1,1],[5,5]]` → stored as `[0, 0, 5, 5]` (rows 0–4, cols 0–4).

`location_areas` uses the same convention applied via numpy slice (`grid_np[r1-1:r2, c1-1:c2]`, `config_loader.py:748`). Later entries overwrite earlier ones at overlapping cells (last writer wins).

---

## Mandatory Key Reference

All keys accessed via `config.get_mandatory()` — missing → `ValueError` with no fallback.

### `load_env_params` mandatory keys

```
environment.height                     environment.width
environment.max_steps                  environment.start_pos
environment.random_start_pos           environment.rest_action_enabled
environment.eat_action_enabled         environment.resources
environment.obstacles                  environment.location_areas

body.max_satiation                     body.max_nutrition
body.max_injury                        body.food_nutrition_gain
body.satiation_setpoint                body.start_satiation
body.start_nutrition                   body.metabolic_cost
body.nutrition_to_satiation_scaling_factor
body.recovery_base_rate                body.recovery_accel_rate
body.injury_smoothing_duration         body.death_penalty
body.overeating_death                  body.use_homeostatic_reward
body.with_satiation                    body.with_nutrition
body.with_injury                       body.random_start_satiation
body.random_start_nutrition            body.random_start_injury
body.eating_nutrition_cost             body.eating_reward_penalty

sensory.sensor_radius                  sensory.decay_power
sensory.collision_sensor_range         sensory.visual_sensor_enabled
sensory.visual_sensor_range            sensory.olfactory_enabled
sensory.nociception_enabled            sensory.proprioception_enabled
sensory.location_sensor                sensory.vector_size
sensory.nociception_size               sensory.injury_observable
sensory.nutrition_observable           sensory.interoceptive_nociception_enabled
sensory.interoceptive_convolution_enabled
sensory.interoceptive_kernel_length    sensory.interoceptive_kernel_tau

visualization.local_view_size
```

**Removed in v2.0 (raises `ValueError` if present):** `environment.predator_enabled`

**Not mandatory at top-level but required by `_load_animals`:**  
`environment.height`, `environment.width` (re-read for default area bounds, `config_loader.py:238–239`).

### `load_behavior_measure_cfg` mandatory keys (14 total)

Applies only when `behavior_measures:` block is present in the YAML. If absent, the function returns `None` (feature off, backward-compatible).

```
behavior_measures.enabled              behavior_measures.cue_radius
behavior_measures.obs_window           behavior_measures.eval_n_episodes
behavior_measures.eval_seeds           behavior_measures.eval_policy_mode
behavior_measures.eval_max_steps       behavior_measures.eval_obs_noise
behavior_measures.motif_window_K       behavior_measures.motif_features
behavior_measures.motif_kmeans_k       behavior_measures.motif_kmeans_seed
behavior_measures.motif_standardise    behavior_measures.eval_output_root
```

Validated enum fields:

| Key | Allowed values |
|-----|---------------|
| `eval_policy_mode` | `"deterministic"`, `"stochastic"` |
| `eval_obs_noise` | `"training"`, `"zero"`, `"custom"` |
| `motif_standardise` | `"zscore_pooled"`, `"zscore_per_agent"`, `"none"` |
| `motif_features` | non-empty subset of the 10 v1 names in `_DEFAULT_FEATURE_NAMES` (`config_loader.py:70–75`) |

Additional constraints: `cue_radius > 0`, `obs_window >= 1`, `eval_n_episodes >= 1`, `len(eval_seeds) == eval_n_episodes`, seeds unique, `eval_max_steps >= 1`, `motif_window_K >= 1`, `motif_kmeans_k >= 2`.

---

## Optional Keys and Defaults

| YAML key | Default | Source |
|----------|---------|--------|
| `environment.placement.mode` | `"per_entity"` | `config_loader.py:779` |
| `perceptual_noise.enabled` | `False` | `config_loader.py:940` |
| Per-resource `nociception_intensity` | `0.9` (hiding\_predator), `0.0` (food) | `config_loader.py:652` |
| Per-resource `count` | `1` | `config_loader.py:622` |
| Per-animal `nociception_intensity` | `0.9` (predator), `0.0` (neutral / unified entities) | `config_loader.py:356,387,324` |
| Per-animal `spawn_area` | full grid `[[1,1],[h,w]]` | `config_loader.py:242` |
| Per-animal `patrol_area` | full grid `[[1,1],[h,w]]` | `config_loader.py:242` |
| Per-animal `count` | `1` | `config_loader.py:299,337,368` |
| Per-animal `tag` | `"idx{i}"` | `_normalise_tag()` `config_loader.py:165` |
| Per-obstacle `blocking` | `True` | `config_loader.py:704` |
| Per-obstacle `hides_agent` | `False` | `config_loader.py:705` |
| Per-obstacle `damage` | `0.0` → `[0,0]` | `config_loader.py:708` |
| Per-obstacle `nociception_intensity` | `0.3` | `config_loader.py:711` |
| Per-obstacle `count` | `1` | `config_loader.py:695` |
| Per-obstacle `name` | `"rock"` | `config_loader.py:720,722` |

---

## Static vs Dynamic Fields in `EnvParams`

**Static** fields (`pytree_node=False`, `state.py:85–221`) are baked into JIT-compiled code; changing any one triggers full recompilation.

| Category | Static (`pytree_node=False`) | Dynamic (traced by JAX) |
|----------|------------------------------|------------------------|
| Grid shape | `height`, `width`, `max_steps` | `grid_location_type [H,W]` |
| Animal metadata | `animal_classes`, `animal_behaviours`, `animal_tags`, `hunt_idx`, `wander_idx`, `static_idx`, `predator_indices`, `neutral_indices` | `animal_property [N,V]`, `animal_property_std [N,V]`, `animal_nociception [N]`, `animal_move_int [N]`, `animal_damage [N,2]`, `animal_attack_delay [N]`, `animal_spawn_area [N,4]`, `animal_patrol [N,4]`, all ten `animal_*_low/high` arrays, `animal_classes_int [N]`, `animal_behaviours_int [N]`, `animal_is_damaging [N]`, `animal_visual_channel [N]` |
| Obstacles | `obstacle_names` | `obs_blocking [N]`, `obs_hides_agent [N]`, `obs_spawn_area [N,4]`, `obs_damage [N,2]`, `obs_property [N,V]`, `obs_property_std [N,V]`, `obs_nociception [N]`, `obs_type [N]` |
| Placement | `max_per_type`, `num_types`, `num_entities`, `placement_mode` | `type_areas [T,4]`, `type_counts [T]`, `type_entity_map [T,max_per_type]` |
| Body flags | `smoothing_duration`, `overeating_death`, `use_homeostatic_reward`, `with_satiation`, `with_nutrition`, `with_injury`, `random_start_satiation`, `random_start_nutrition`, `random_start_injury`, `random_start_pos`, `rest_action_enabled`, `eat_action_enabled` | `max_satiation`, `max_nutrition`, `max_injury`, `food_nutrition_gain`, `setpoint`, `start_satiation`, `start_nutrition`, `metabolic_cost`, `nutrition_to_satiation_scaling_factor`, `recovery_base_rate`, `recovery_accel_rate`, `death_penalty`, `eating_nutrition_cost`, `eating_reward_penalty`, `start_pos [2]` |
| Sensory flags | `sensor_range`, `visual_sensor_enabled`, `visual_sensor_range`, `local_view_size`, `olfactory_enabled`, `nociception_enabled`, `location_sensor_enabled`, `injury_observable`, `nutrition_observable`, `interoceptive_nociception_enabled`, `interoceptive_convolution_enabled`, `interoceptive_kernel_length`, `proprioception_enabled`, `action_dim`, `olfactory_vector_size`, `nociception_size` | `sensor_radius`, `sensor_decay`, `interoceptive_kernel [K]` |
| Noise | `perceptual_noise_enabled`, `noise_modality_order` | `noise_modes [13]`, `noise_sigmas [13]`, `noise_injury_scales [13]`, `noise_clip_min [13]`, `noise_clip_max [13]` |

---

## Olfactory YAML Keys

All entity types now uniformly use the **plural** form. The singular forms are deprecated and emit `DeprecationWarning` at load time but still function as a fallback (`config_loader.py:176–200`).

| Entity | Chemical signature key | Std dev key |
|--------|------------------------|-------------|
| Resource | `properties` | `properties_std` |
| Animal (predator / neutral / any) | `properties` | `properties_std` |
| Obstacle | `properties` | `properties_std` |

**Deprecated (rename these in your YAML):** `property` → `properties`, `property_std` → `properties_std`.

---

## Noise Configuration Parsing

Implemented in `_parse_noise_config()` (`config_loader.py:957`).

The YAML key order under `perceptual_noise.modalities` is the single source of truth for which index in the noise arrays corresponds to which modality. Unknown keys are **silently dropped** (the `if k in _YAML_KEY_TO_SENSOR_NAME` filter at `config_loader.py:965`). Typos are silent — double-check against the table below.

All five noise arrays are padded to length **13** (`pad = max(0, 13 - len(noise_modality_order))`, `config_loader.py:968`). This keeps the array shape static regardless of how many modalities are configured.

YAML key → sensor name mapping (`config_loader.py:944–955`):

| YAML key | Sensor name | Default index in `default.yaml` |
|----------|-------------|--------------------------------|
| `injury` | `"Injury"` | 0 |
| `nutrition` | `"Nutrition"` | 1 |
| `satiation` | `"Satiation"` | 2 |
| `interoceptive_nociception` | `"Interoceptive Nociception"` | 3 |
| `extero_nociception` | `"Extero Nociception"` | 4 |
| `olfaction` | `"Olfaction"` | 5 |
| `collision` | `"Collision"` | 6 |
| `proprioception` | `"Proprioception"` | 7 |
| `visual` | `"Visual"` | 8 |
| `location` | `"Location"` | 9 |

10 modalities defined (+ 3 spare slots = 13 total). `sensor.py` builds `modality_map = {name: i for i, name in enumerate(params.noise_modality_order)}` at observation-assembly time.

Per-modality optional keys (defaults apply when absent):

| Key | Default |
|-----|---------|
| `mode` | `"none"` (encoded as `0`) |
| `sigma` | `0.0` |
| `injury_noise_scale` | `0.0` |
| `clip_min` | `-100.0` |
| `clip_max` | `100.0` |

Mode encoding: `"none"` → `0`, `"constant"` → `1`, `"state_dependent"` → `2`.

Detail: see `docs/environment/10_perceptual_noise.md`.

---

## Default Values Reference

Full annotated listing from `configs/environment/default.yaml`:

```yaml
environment:
  height: 10                        # 10×10 grid
  width: 10
  start_pos: [5, 5]                 # 1-based; stored as [4, 4] (0-based)
  max_steps: 500
  rest_action_enabled: true         # action 4 = rest
  eat_action_enabled: true          # action 5 = eat
  random_start_pos: true
  placement:
    mode: per_entity                # fastest for 10×10

  resources:
    - type: "food"
      count: 2
      spawn_area: [[1,1],[5,5]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: 12
      regeneration_delay: 0
      damage: [0.0, 0.0]
      nociception_intensity: 0.0

    - type: "hiding_predator"       # static trap resource (not a moving animal)
      count: 1
      spawn_area: [[1,1],[5,5]]
      properties: [0.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: -1
      damage: [15.0, 45.0]
      regeneration_delay: 20
      nociception_intensity: 0.9

  predators:                        # legacy schema — re-projected as class=predator, behaviour=hunt
    - name: "predator"
      count: 1
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.4, 0.4, 0.0, 0.0]
      move_interval: 1
      damage: [15.0, 45.0]
      nociception_intensity: 0.9
      spawn_area: [[1,1],[10,10]]
      patrol_area: [[1,1],[10,10]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      attack_delay: 3
      lose_interest_multiplier: 1.5

  neutral_animals:                  # legacy schema — re-projected as class=neutral, behaviour=wander
    - name: "rabbit"
      count: 1
      properties: [0.0, 0.5, 0.7, 0.0, 0.0]
      properties_std: [0.0, 0.4, 0.4, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.1
      spawn_area: [[1,1],[5,5]]
      patrol_area: [[1,1],[5,5]]

  obstacles:
    - name: "rock"
      count: 3
      area: [[1,1],[5,5]]
      blocking: false
      damage: [1, 5]
      nociception_intensity: 0.9
      properties: [0.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
    - name: "bush"
      count: 5
      area: [[1,6],[5,10]]
      blocking: false
      hides_agent: true
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]

  location_areas:
    - type: "grass"
      area: [[1,1],[10,10]]         # whole grid is grass

body:
  with_satiation: true
  with_nutrition: true
  with_injury: true
  random_start_satiation: false
  random_start_nutrition: false
  random_start_injury: false
  max_satiation: 100
  max_nutrition: 100
  max_injury: 100
  metabolic_cost: 1.0
  food_nutrition_gain: 6
  eating_nutrition_cost: 1.0
  eating_reward_penalty: 0.0
  nutrition_to_satiation_scaling_factor: 1.0
  satiation_setpoint: 100
  start_satiation: 100
  start_nutrition: 100
  recovery_base_rate: 0.1
  recovery_accel_rate: 0.5
  injury_smoothing_duration: 3
  use_homeostatic_reward: true
  death_penalty: 100
  overeating_death: false

sensory:
  olfactory_enabled: true
  sensor_radius: 20                 # effectively whole 10×10 grid
  vector_size: 5                    # 5-dim chemical property
  decay_power: 2.0                  # inverse square distance decay
  collision_sensor_range: 1         # 5-cell Manhattan diamond
  location_sensor: false
  nociception_enabled: true
  nociception_size: 1
  visual_sensor_enabled: true
  visual_sensor_range: 0            # single-cell (agent's own cell only)
  proprioception_enabled: true
  injury_observable: false          # hidden (perceived via nociception only)
  nutrition_observable: false       # hidden (perceived via olfaction / intero)
  interoceptive_nociception_enabled: true
  interoceptive_convolution_enabled: true
  interoceptive_kernel_tau: 3.0
  interoceptive_kernel_length: 12

visualization:
  local_view_size: 5

perceptual_noise:
  enabled: false
  modalities:
    injury:
      mode: "state_dependent"
      sigma: 0.0
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 1.0
    interoceptive_nociception:
      mode: "state_dependent"
      sigma: 0.1
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 1.0
    extero_nociception:
      mode: "state_dependent"
      sigma: 0.1
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 100.0
    olfaction:
      mode: "state_dependent"
      sigma: 0.2
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 100.0
    collision:
      mode: "constant"
      sigma: 0.01
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    visual:
      mode: "state_dependent"
      sigma: 0.2
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 100.0
    location:
      mode: "constant"
      sigma: 0.01
      injury_noise_scale: 0.0
      clip_min: -1.0
      clip_max: 1.0
```

---

## Clarifications / FAQ

**Q: What happens if I set `count: 0` on an entity entry?**
A: `range(0)` produces zero expansions, so the entry is skipped entirely. It is a clean way to disable a specific entity kind without deleting its YAML block — useful for ablations.

**Q: What if I leave `count` out entirely?**
A: Defaults to `1` (`config_loader.py:299,337,368,622,695`). One entity is spawned.

**Q: Scalar vs range `damage` — which format is right?**
A: Both work. A scalar `damage: 10` expands to `[10, 10]` (deterministic). A list `damage: [5, 15]` specifies a uniform distribution. The `*_damage` array is always shape `[N, 2]`.

**Q: Can distributional fields like `detection_range` also be scalars?**
A: Yes. `detection_range: 5` is stored as `[5.0, 5.0]` (deterministic per episode). `detection_range: [3, 7]` samples uniformly from [3, 7] at each reset.

**Q: What happens if `location_areas` has overlapping entries?**
A: Later entries overwrite earlier ones at the overlapping cells (`config_loader.py:748`). Order your YAML so the desired foreground terrain appears last.

**Q: What does `placement.mode` actually affect at runtime?**
A: Only the reset-time placement algorithm — after step 0, the two modes produce identical behaviour. Changing the mode triggers JIT recompilation because `placement_mode` is static. See `docs/environment/03_entity_placement.md`.

**Q: When is a type group created?**
A: At config-load time, entities are grouped by the exact tuple of their 0-indexed spawn-area bounding box (`config_loader.py:762–764`). Entities from different kinds (food, predator, obstacle, neutral) with the same bounding box share a group.

**Q: How is `obstacle_names` built — does order matter?**
A: `obstacle_names` is the **sorted unique** set of obstacle `name` values (`config_loader.py:720`). `obs_type[o]` is the index of obstacle `o`'s name in this tuple. Renaming an obstacle may shift its index — don't hardcode indices.

**Q: If I omit `name` on an obstacle, what happens?**
A: Defaults to `"rock"` (`config_loader.py:720,722`). If all obstacles are nameless, `obstacle_names = ("rock",)`.

**Q: The old doc listed `environment.predator_enabled` as mandatory — is it still needed?**
A: No. It was **removed entirely in v2.0**. If your YAML still contains `environment.predator_enabled`, `load_env_params` raises `ValueError` with a migration message (`config_loader.py:666–671`). Strip that line from your YAML.

**Q: Are `body.start_satiation` and `body.random_start_satiation` actually used?**
A: They are mandatory in the schema and copied into `EnvParams`, but satiation at reset is derived from nutrition in `core.py`. These are legacy fields.

**Q: What does `sensor_radius: 20` do on a 10×10 grid?**
A: Olfaction scales intensity by distance with decay `decay_power`. `sensor_radius` is the normalisation distance. A radius ≥ max grid distance means every source is detectable (intensity still decays). See `docs/environment/09_sensors_and_observation.md`.

**Q: If `perceptual_noise.enabled: false`, does `modalities` still need to be present?**
A: No — `_parse_noise_config` safely returns zeros for absent modalities. Leaving `modalities` absent is fine.

**Q: Does `config.get(...)` raise on missing keys?**
A: No. `config.get('key', default)` returns the default. Only `config.get_mandatory('key')` raises `ValueError`.

**Q: Is `default.yaml` loaded automatically?**
A: No — you must explicitly point your training script at it (or merge it) via `Config.merge()`. There is no auto-loading.

**Q: The noise array is padded to 13. Why 13?**
A: 10 modalities defined + 3 spare slots. Padding keeps the array shape static regardless of how many modalities are configured, avoiding recompilation. The spare slots are always zero-valued and never indexed.

**Q: What happens if my YAML lists a modality name not in `_YAML_KEY_TO_SENSOR_NAME`?**
A: It is silently dropped from `noise_modality_order` (`config_loader.py:965`). No error is raised. Typos are silent — double-check the 10 valid keys in the table above.

**Q: Does `start_pos: [5, 5]` match row/col or x/y?**
A: Row/col, 1-indexed and inclusive. Stored as `[4, 4]` (0-indexed) in `EnvParams.start_pos` (`config_loader.py:910`).

**Q: The old doc says `predator_tags` and `neutral_tags` are passed to the EnvParams constructor — is that still true?**
A: No. In v2.0 these are `@property` accessors on the struct (`state.py:234–247`), not constructor arguments. They filter `animal_tags` by class on the fly. The corresponding struct fields have been removed (M1 fix).

**Q: Is `environment.predators` or `environment.neutral_animals` still required?**
A: Neither is mandatory. The loader uses `config.get()` (not `config.get_mandatory()`) for both (`config_loader.py:335,366`). An absent or empty list results in zero animals of that class. The new `environment.entities:` schema is entirely optional too — if all three sections are absent, the environment has zero animals.
