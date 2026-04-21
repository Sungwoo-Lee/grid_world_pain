# 02 — Config Schema

> **Source**: `src/environment/config_loader.py`, `configs/environment/default.yaml` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

`load_env_params(config: Config) → EnvParams` is the single entry point that turns a YAML config into a fully-typed `EnvParams` pytree ready for `jax_reset`. It uses a `Config` utility wrapper (dot-notation access, `merge()` support) and enforces a **strict mandatory-key policy**: any key accessed via `config.get_mandatory()` that is absent raises `ValueError` immediately — no silent defaults for critical fields.

The transformation pipeline:

1. Parse `environment.resources` list → expand by `count` → build `res_*` arrays
2. Parse `environment.predators` → expand → build `pred_*` arrays
3. Parse `environment.obstacles` → expand → build `obs_*` arrays
4. Parse `environment.neutral_animals` → expand → build `neutral_*` arrays
5. Build `grid_location_type` from `environment.location_areas`
6. Group entities by shared spawn area → build `type_areas`, `type_counts`, `type_entity_map`
7. Parse `perceptual_noise.modalities` → build noise arrays via `_parse_noise_config`
8. Construct and return `EnvParams`

---

## YAML Top-Level Structure

```
environment:
  height, width, max_steps, start_pos
  random_start_pos, rest_action_enabled, eat_action_enabled
  placement.mode
  resources: [ list of resource defs ]
  predators: [ list of predator defs ]
  obstacles: [ list of obstacle defs ]
  neutral_animals: [ list of neutral animal defs ]
  location_areas: [ list of area defs ]
  predator_enabled: bool

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

visualization:
  local_view_size

perceptual_noise:
  enabled: bool
  modalities:
    injury / nutrition / satiation / extero_nociception / olfaction /
    collision / proprioception / visual / location:
      mode, sigma, injury_noise_scale, clip_min, clip_max
```

---

## Entity Count Expansion

Each entity definition in YAML carries an optional `count` field. The config loader expands each definition `count` times into a flat list before building JAX arrays. For example:

```yaml
resources:
  - type: "food"
    count: 2
    spawn_area: [[1,1],[5,5]]
    ...
```

Becomes two identical resource entries in `expanded_resources`. The resulting `res_type`, `res_property`, `res_spawn_area`, etc. arrays have shape `[total_expanded_count, ...]`. Entity ordering in the flat arrays is: **resources first, then predators, then obstacles, then neutral animals** — this concatenation order is preserved in `jax_reset` when splitting `all_positions` back into per-type arrays (`core.py:703-707`).

When an entity list is empty (e.g. no predators), the loader builds zero-size arrays of the correct shape so `jax_step` and `jax.vmap` do not need entity-count conditionals.

---

## Placement Configuration

Controlled by `environment.placement.mode` (`config_loader.py:209`):

| Mode | Description | Best for |
|------|-------------|----------|
| `per_entity` | vmap sample per entity → `resolve_overlaps_global` sequential scan | Small grids (≤100 cells) |
| `per_type` | `lax.scan` over spawn-area groups via `place_in_area` | Large grids or high entity density |

**Type group construction** (always built, even in `per_entity` mode): entities are grouped by their spawn area bounding box. Entities with identical bounding boxes go into the same group.

- `type_areas [T, 4]` — one bounding box per group
- `type_counts [T]` — entity count per group
- `type_entity_map [T, max_per_type]` — global entity indices (padded to `max_per_type` with 0)
- `max_per_type`, `num_types`, `num_entities` — static scalars that fix loop bounds at compile time

`placement_mode` is a static field — changing it forces recompilation.

---

## Noise Configuration Parsing

Implemented in `_parse_noise_config()` (`config_loader.py:334`).

**Key invariant**: the YAML key order under `perceptual_noise.modalities` is the single source of truth for which index in the noise arrays corresponds to which modality. The loader iterates `modalities_cfg` (a Python dict preserving insertion order in Python ≥ 3.7) and emits:

```python
noise_modality_order = tuple(
    _YAML_KEY_TO_SENSOR_NAME[k]
    for k in modalities_cfg
    if k in _YAML_KEY_TO_SENSOR_NAME
)
```

This tuple is stored in `EnvParams.noise_modality_order` (a static pytree-False field). `sensor.py` builds `modality_map = {name: i for i, name in enumerate(params.noise_modality_order)}` to convert sensor names to array indices at observation-assembly time.

All five noise arrays (`noise_modes`, `noise_sigmas`, `noise_injury_scales`, `noise_clip_min`, `noise_clip_max`) are **zero-padded to length 12** (currently 9 defined modalities + 3 spare slots). This keeps the array shape static regardless of how many modalities are configured, avoiding recompilation when modalities are added or removed.

YAML key → sensor name mapping (`config_loader.py:322`):

| YAML key | Sensor name |
|----------|-------------|
| `injury` | `"Injury"` |
| `nutrition` | `"Nutrition"` |
| `satiation` | `"Satiation"` |
| `extero_nociception` | `"Extero Nociception"` |
| `olfaction` | `"Olfaction"` |
| `collision` | `"Collision"` |
| `proprioception` | `"Proprioception"` |
| `visual` | `"Visual"` |
| `location` | `"Location"` |

---

## Mandatory vs Optional Keys

**Mandatory** (missing → `ValueError` via `config.get_mandatory()`):

```
environment.height                    environment.width
environment.max_steps                 environment.predator_enabled
environment.random_start_pos          environment.rest_action_enabled
environment.eat_action_enabled        environment.start_pos
environment.resources                 environment.predators
environment.obstacles                 environment.neutral_animals
environment.location_areas

body.max_satiation                    body.max_nutrition
body.max_injury                       body.food_nutrition_gain
body.satiation_setpoint               body.start_satiation
body.start_nutrition                  body.metabolic_cost
body.nutrition_to_satiation_scaling_factor
body.recovery_base_rate               body.recovery_accel_rate
body.injury_smoothing_duration        body.death_penalty
body.overeating_death                 body.use_homeostatic_reward
body.with_satiation                   body.with_nutrition
body.with_injury                      body.random_start_satiation
body.random_start_nutrition           body.random_start_injury
body.eating_nutrition_cost            body.eating_reward_penalty

sensory.sensor_radius                 sensory.decay_power
sensory.collision_sensor_range        sensory.visual_sensor_enabled
sensory.visual_sensor_range           sensory.olfactory_enabled
sensory.nociception_enabled           sensory.proprioception_enabled
sensory.location_sensor               sensory.vector_size
sensory.nociception_size

visualization.local_view_size
```

**Optional** (safe defaults used):

| Key | Default | Where used |
|-----|---------|-----------|
| `environment.placement.mode` | `"per_entity"` | `config_loader.py:209` |
| `perceptual_noise.enabled` | `False` | `config_loader.py:318` |
| `perceptual_noise.modalities.*` | empty dict | `_parse_noise_config` |
| Per-resource `nociception_intensity` | `0.9` (danger) / `0.0` (food) | `config_loader.py:42` |
| Per-resource `properties_std` | `[0.0] * vector_size` | `config_loader.py:32` |
| Per-predator `nociception_intensity` | `0.9` | `config_loader.py:61` |
| Per-predator `property_std` | `[0.0] * vector_size` | `config_loader.py:60` |
| Per-predator `patrol_area` | full grid | `config_loader.py:71` |
| Per-predator `lose_interest_multiplier` | `2.0` | `config_loader.py:78` |
| Per-obstacle `blocking` | `True` | `config_loader.py:108` |
| Per-obstacle `hides_agent` | `False` | `config_loader.py:109` |
| Per-obstacle `damage` | `0.0` | `config_loader.py:112` |
| Per-obstacle `nociception_intensity` | `0.3` | `config_loader.py:115` |
| Per-obstacle `properties_std` | `[0.0] * vector_size` | `config_loader.py:118` |
| Per-neutral `nociception_intensity` | `0.0` | `config_loader.py:153` |
| Per-neutral `property_std` | `[0.0] * vector_size` | `config_loader.py:152` |

---

## Coordinate Convention

YAML area specifications use **1-based inclusive coordinates** in the format `[[row_min, col_min], [row_max, col_max]]`. The config loader converts these to **0-based with exclusive max** for JAX's `randint`:

```python
res_spawn_area = jnp.array([
    [a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]]
    for a in [r_get(r, 'spawn_area') for r in expanded_resources]
])
```

This means a YAML area of `[[1,1],[5,5]]` becomes `[0, 0, 5, 5]` — rows 0–4, cols 0–4 (5 is exclusive). The same conversion applies to all spawn and patrol areas. `location_areas` uses a different but analogous convention (see `config_loader.py:178`).

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

body:
  with_satiation: false             # satiation tracked but not in termination
  with_nutrition: true              # starvation death enabled
  with_injury: true                 # injury death enabled
  max_satiation: 100
  max_nutrition: 100
  max_injury: 100
  metabolic_cost: 1.0               # 1 nutrition/step decay
  food_nutrition_gain: 6            # net gain = 6 - 1 (eating_cost) = 5
  eating_nutrition_cost: 1.0
  eating_reward_penalty: 0.0
  nutrition_to_satiation_scaling_factor: 1.0   # linear mapping
  satiation_setpoint: 100           # target = fully satiated
  start_satiation: 100
  start_nutrition: 100
  recovery_base_rate: 0.1           # base injury healed per rest step
  recovery_accel_rate: 0.5          # 1.5× boost per additional rest streak step
  injury_smoothing_duration: 3      # damage spread over 3 steps
  use_homeostatic_reward: true      # drive-reduction reward
  death_penalty: 100
  overeating_death: false

sensory:
  olfactory_enabled: true
  sensor_radius: 20                 # effectively whole grid
  vector_size: 5                    # 5-dim chemical property
  decay_power: 2.0                  # inverse square distance decay
  collision_sensor_range: 1         # 5-cell Manhattan diamond
  location_sensor: false
  nociception_enabled: true
  nociception_size: 1
  visual_sensor_enabled: true
  visual_sensor_range: 0            # single-cell (agent's own cell only)
  proprioception_enabled: true

perceptual_noise:
  enabled: true
  modalities:
    injury:            mode: state_dependent, sigma: 0.1, injury_noise_scale: 1.5
    nutrition:         mode: state_dependent, sigma: 0.1, injury_noise_scale: 1.5
    satiation:         mode: state_dependent, sigma: 0.1, injury_noise_scale: 1.5
    extero_nociception: mode: state_dependent, sigma: 0.1, injury_noise_scale: 1.5
    olfaction:         mode: state_dependent, sigma: 0.2, injury_noise_scale: 1.5
    collision:         mode: constant, sigma: 0.01
    proprioception:    mode: constant, sigma: 0.05
    visual:            mode: state_dependent, sigma: 0.2, injury_noise_scale: 1.5
    location:          mode: constant, sigma: 0.01
```
