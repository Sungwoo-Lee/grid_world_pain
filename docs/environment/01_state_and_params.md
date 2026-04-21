# 01 — State & Parameters

> **Source**: `src/environment/state.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

The environment is split into two distinct data structures:

- **`EnvState`** — mutable, per-step data that changes every call to `jax_step`. Carries all the "what is happening right now" information: positions, timers, body values, RNG key.
- **`EnvParams`** — immutable, per-episode configuration. Locked at reset time and shared across all steps. Contains all rules, limits, and constant entity attributes.

Both are Flax `struct.dataclass` pytrees. This means JAX can trace and JIT them without recompilation when values change (only when structure changes). The split matters for `jax.jit` because `EnvParams` fields declared with `struct.field(pytree_node=False)` are treated as static — they affect the compiled graph shape and force recompilation if changed. `EnvState` fields are all pytree leaves and change freely without recompilation.

Updates always produce a new object via `.replace(**kwargs)` (which the code aliases as `._replace(**kwargs)` for compatibility). No in-place mutation ever occurs.

---

## EnvState — Mutable Per-Step

Defined in `src/environment/state.py:6`.

### Agent

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `agent_pos` | `[2]` | int32 | `[0, H-1] × [0, W-1]` | Agent's `(row, col)` position on the grid |
| `current_step` | `[]` | int32 | `[0, max_steps]` | Step counter within the episode |
| `last_action` | `[]` | int32 | `[0, action_dim-1]` | Action taken on the previous step (used by proprioception sensor) |

### Resources

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `res_pos` | `[num_res, 2]` | int32 | grid bounds | Current `(row, col)` of each resource |
| `res_active` | `[num_res]` | bool | `{0, 1}` | Whether the resource is available for interaction |
| `res_cons_count` | `[num_res]` | int32 | `[0, max_cons]` | How many times this resource has been consumed since last respawn |
| `res_reg_timer` | `[num_res]` | int32 | `[0, reg_delay]` | Countdown to respawn; decrements each step when inactive |
| `res_property_sampled` | `[num_res, 5]` | float32 | `[0, 1]` | Olfactory signature sampled at respawn |

### Predators

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `pred_pos` | `[num_pred, 2]` | int32 | grid bounds | Current `(row, col)` of each predator |
| `pred_state` | `[num_pred]` | int32 | `{0, 1, 2}` | FSM state: 0=Patrol, 1=Hunt, 2=Return |
| `pred_stamina` | `[num_pred]` | float32 | `[0, max_stamina]` | Current stamina; drains during Hunt, recovers otherwise |
| `pred_move_timer` | `[num_pred]` | int32 | `[0, move_int]` | Countdown to next movement step |
| `pred_attack_timer` | `[num_pred]` | int32 | `[0, attack_delay]` | Cooldown after an attack; blocks movement while nonzero |
| `pred_property_sampled`| `[num_pred, 5]` | float32 | `[0, 1]` | Olfactory signature sampled at reset |

### Neutral Animals

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `neutral_pos` | `[num_neutral, 2]` | int32 | grid bounds | Current `(row, col)` of each neutral animal |
| `neutral_move_timer` | `[num_neutral]` | int32 | `[0, move_int]` | Countdown to next move step |
| `neutral_property_sampled`| `[num_neutral, 5]`| float32 | `[0, 1]` | Olfactory signature sampled at reset |

### Obstacles

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `obs_pos` | `[num_obs, 2]` | int32 | grid bounds | `(row, col)` of each obstacle; fixed after reset |
| `obs_property_sampled`| `[num_obs, 5]` | float32 | `[0, 1]` | Olfactory signature sampled at reset |

### Body

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `satiation` | `[]` | float32 | `[0, max_satiation]` | Subjective fullness; derived nonlinearly from nutrition |
| `nutrition` | `[]` | float32 | `[0, max_nutrition]` | Objective energy reserve; decays each step |
| `injury_level` | `[]` | float32 | `[0, max_injury]` | Current accumulated injury after ring-buffer smoothing |
| `injury_buffer` | `[smoothing_duration]` | float32 | `[0, ...]` | Ring buffer holding incremental injury to be applied over future steps |
| `last_collision_noc` | `[]` | float32 | `[0, 1]` | Nociception intensity from the most recent obstacle collision bump; cleared next step |
| `rest_streak` | `[]` | int32 | `[0, ...]` | Consecutive steps the agent has been resting; boosts injury recovery rate |

### Meta

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `terminated` | `[]` | bool | `{0, 1}` | Whether this episode has ended (either done or truncated) |
| `key` | `PRNGKey` | uint32 | — | JAX PRNG key, split each step to produce all random events for that step |

---

## EnvParams — Immutable Per-Episode

Defined in `src/environment/state.py:51`. Fields marked `struct.field(pytree_node=False)` are **static** — they determine the compiled graph shape and trigger recompilation when changed.

### Grid

| Field | Type | Static | Description |
|-------|------|--------|-------------|
| `height` | int | yes | Grid height (rows) |
| `width` | int | yes | Grid width (cols) |
| `max_steps` | int | yes | Episode length before truncation |
| `grid_location_type` | `[H,W]` int32 | no | Tile type: 0=plain, 1=grass, 2=sand |

### Resources

| Field | Shape | Description |
|-------|-------|-------------|
| `res_type` | `[N]` int32 | 0=food, 1=danger |
| `res_property` | `[N, 5]` float32 | Olfactory chemical signature mean per resource |
| `res_property_std` | `[N, 5]` float32 | Olfactory chemical signature std dev per resource |
| `res_nociception` | `[N]` float32 | Nociception intensity emitted on contact |
| `res_spawn_area` | `[N, 4]` int32 | Bounding box `(min_r, min_c, max_r, max_c)` for respawn |
| `res_max_cons` | `[N]` int32 | Max consumptions before permanent deactivation; -1 = unlimited |
| `res_reg_delay` | `[N]` int32 | Steps to wait before respawning after consumption |
| `res_damage` | `[N, 2]` float32 | Damage `[min, max]` sampled uniformly on contact |

### Predators

| Field | Shape | Description |
|-------|-------|-------------|
| `pred_property` | `[P, 5]` float32 | Olfactory signature mean |
| `pred_property_std` | `[P, 5]` float32 | Olfactory signature std dev |
| `pred_nociception` | `[P]` float32 | Nociception intensity on contact |
| `pred_move_int` | `[P]` int32 | Steps per move (lower = faster) |
| `pred_damage` | `[P, 2]` float32 | Damage range `[min, max]` |
| `pred_patrol` | `[P, 4]` int32 | Patrol bounding box `(min_r, min_c, max_r, max_c)` |
| `pred_spawn_area` | `[P, 4]` int32 | Spawn bounding box |
| `pred_detect` | `[P]` float32 | Manhattan detection radius |
| `pred_max_stamina` | `[P]` float32 | Maximum stamina |
| `pred_recovery` | `[P]` float32 | Stamina recovered per non-hunt step |
| `pred_hunt_thresh` | `[P]` float32 | Fraction of max_stamina required before predator will switch to Hunt |
| `pred_attack_delay` | `[P]` int32 | Cooldown steps after an attack |
| `pred_lose_interest_mult` | `[P]` float32 | Multiplies detection range to compute the "lose interest" distance |
| `predator_enabled` | bool | Static flag; if False, predator update loop is a no-op |

### Obstacles

| Field | Shape | Description |
|-------|-------|-------------|
| `obs_blocking` | `[O]` bool | If True, agent is bounced back on collision |
| `obs_hides_agent` | `[O]` bool | If True and agent is on this cell, predators cannot detect the agent |
| `obs_spawn_area` | `[O, 4]` int32 | Spawn bounding box |
| `obs_damage` | `[O, 2]` float32 | Damage range on collision/overlap |
| `obs_property` | `[O, 5]` float32 | Olfactory chemical signature mean |
| `obs_property_std` | `[O, 5]` float32 | Olfactory chemical signature std dev |
| `obs_nociception` | `[O]` float32 | Nociception intensity on contact |
| `obs_type` | `[O]` int32 | Index into `obstacle_names` tuple for visual encoding |
| `obstacle_names` | `tuple[str]` | Static tuple of unique obstacle name strings (e.g. `("bush", "rock")`) |

### Neutral Animals

| Field | Shape | Description |
|-------|-------|-------------|
| `neutral_property` | `[M, 5]` float32 | Olfactory signature mean |
| `neutral_property_std`| `[M, 5]` float32 | Olfactory signature std dev |
| `neutral_nociception` | `[M]` float32 | Nociception on contact |
| `neutral_move_int` | `[M]` int32 | Steps per move |
| `neutral_patrol` | `[M, 4]` int32 | Patrol bounding box |
| `neutral_spawn_area` | `[M, 4]` int32 | Spawn bounding box |

### Placement

| Field | Type | Static | Description |
|-------|------|--------|-------------|
| `type_areas` | `[T, 4]` int32 | no | Spawn area per type group |
| `type_counts` | `[T]` int32 | no | Entity count per group |
| `type_entity_map` | `[T, max_per_type]` int32 | no | Global entity indices for each group |
| `max_per_type` | int | yes | Max entities in any single group (used as static loop bound) |
| `num_types` | int | yes | Number of type groups |
| `num_entities` | int | yes | Total entity count (res + pred + obs + neutral) |
| `placement_mode` | str | yes | `"per_entity"` or `"per_type"` |

### Body

| Field | Type | Static | Description |
|-------|------|--------|-------------|
| `max_satiation` | float | no | Upper bound on satiation |
| `max_nutrition` | float | no | Upper bound on nutrition |
| `max_injury` | float | no | Upper bound on injury (death at this value) |
| `food_nutrition_gain` | float | no | Nutrition gained per food consumption event |
| `setpoint` | float | no | Target satiation for homeostatic drive |
| `start_satiation` | float | no | Initial satiation if not randomised |
| `start_nutrition` | float | no | Initial nutrition if not randomised |
| `metabolic_cost` | float | no | Nutrition drained per step |
| `nutrition_to_satiation_scaling_factor` | float | no | Exponent `k` in `S = max_S * (N/max_N)^k` |
| `recovery_base_rate` | float | no | Base injury recovery amount per rest step |
| `recovery_accel_rate` | float | no | Exponential boost factor per rest streak step |
| `smoothing_duration` | int | yes | Length of injury ring buffer |
| `death_penalty` | float | no | Reward penalty applied on episode termination |
| `overeating_death` | bool | yes | If True, satiation ≥ max_satiation triggers death |
| `use_homeostatic_reward` | bool | yes | Toggles reward mode (see doc 06) |
| `with_satiation` | bool | yes | Enable satiation tracking |
| `with_nutrition` | bool | yes | Enable nutrition tracking and starvation |
| `with_injury` | bool | yes | Enable injury tracking; if False, any damage kills instantly |
| `random_start_satiation` | bool | yes | Randomise starting satiation |
| `random_start_nutrition` | bool | yes | Randomise starting nutrition (uniform in `[max/2, max]`) |
| `random_start_injury` | bool | yes | Randomise starting injury (uniform in `[0, max/2]`) |
| `random_start_pos` | bool | yes | Randomise agent start position |
| `start_pos` | `[2]` int32 | no | Fixed start position (used when `random_start_pos=False`) |
| `rest_action_enabled` | bool | yes | Adds action 4 = Rest |
| `eat_action_enabled` | bool | yes | Adds action 5 = Eat (otherwise eating is automatic on overlap) |
| `eating_nutrition_cost` | float | no | Nutrition cost deducted when eating |
| `eating_reward_penalty` | float | no | Reward penalty applied when eating |

### Sensors

| Field | Type | Static | Description |
|-------|------|--------|-------------|
| `sensor_radius` | float | no | Olfactory detection radius |
| `sensor_decay` | float | no | Distance decay power for olfaction |
| `sensor_range` | int | yes | Collision sensor diamond radius |
| `visual_sensor_enabled` | bool | yes | Enable visual sensor |
| `visual_sensor_range` | int | yes | Visual sensor diamond radius |
| `local_view_size` | int | yes | Renderer local view window size |
| `olfactory_enabled` | bool | yes | Enable olfaction sensor |
| `nociception_enabled` | bool | yes | Enable exteroceptive nociception sensor |
| `location_sensor_enabled` | bool | yes | Enable location sensor |
| `proprioception_enabled` | bool | yes | Enable proprioception sensor |
| `action_dim` | int | yes | Total number of actions (4 + rest + eat flags) |
| `olfactory_vector_size` | int | yes | Olfaction vector length (equals `res_property` width) |
| `nociception_size` | int | yes | Currently always 1 |

### Perceptual Noise

| Field | Type | Static | Description |
|-------|------|--------|-------------|
| `perceptual_noise_enabled` | bool | yes | Enable noise application |
| `noise_modality_order` | tuple | yes | Ordered modality names from YAML key order; used to index noise arrays |
| `noise_modes` | `[12]` int32 | no | Per-modality noise mode: 0=None, 1=Constant, 2=State-Dependent |
| `noise_sigmas` | `[12]` float32 | no | Base standard deviation per modality |
| `noise_injury_scales` | `[12]` float32 | no | Injury-scaling factor α per modality (used in mode 2) |
| `noise_clip_min` | `[12]` float32 | no | Per-modality observation lower bound after noise |
| `noise_clip_max` | `[12]` float32 | no | Per-modality observation upper bound after noise |

---

## Pytree Registration & Flax struct.dataclass

Flax `@struct.dataclass` automatically registers both classes as JAX pytrees. This means:

- `jax.jit`, `jax.vmap`, `jax.lax.scan` can accept them as inputs/outputs without manual tree registration.
- **`pytree_node=False` fields** (e.g. `height`, `width`, `placement_mode`) are treated as **static**. JAX traces a new compiled function any time these values change. They are safe to use in Python-level `if` statements inside JIT.
- **All other fields** are pytree leaves; their values can change freely without recompilation.
- **Immutability**: Flax structs are immutable after creation. To produce an updated state, call `state.replace(field=new_value)` (also exposed as `state._replace(...)` for API compatibility). This is zero-copy when possible under XLA.
- **`vmap` batching**: `ParallelEnv` vmaps over the leading dimension of `EnvState` arrays. Each env in the batch has its own independent state pytree — conceptually a batch of `EnvState` objects stacked along axis 0.
