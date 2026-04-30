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
| `nociception_history_buffer` | `[intero_length]` | float32 | `[0, max_injury]` | FIR history of `injury_level` values used by interoceptive sensor; slot 0 = most recent |
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
| `interoceptive_nociception_enabled` | bool | yes | Enable interoceptive nociception sensor |
| `interoceptive_convolution_enabled` | bool | yes | If False, interoceptive sensor emits `injury/max` directly (no delay) |
| `injury_observable` | bool | yes | If False, raw injury level is removed from observation |
| `nutrition_observable` | bool | yes | If False, raw nutrition level is removed from observation |
| `location_sensor_enabled` | bool | yes | Enable location sensor |
| `proprioception_enabled` | bool | yes | Enable proprioception sensor |
| `action_dim` | int | yes | Total number of actions (4 + rest + eat flags) |
| `olfactory_vector_size` | int | yes | Olfaction vector length (equals `res_property` width) |
| `nociception_size` | int | yes | Currently always 1 |
| `interoceptive_kernel_length` | int | yes | Static length for `nociception_history_buffer` |
| `interoceptive_kernel` | `[K]` float32 | no | Normalized alpha kernel used for interoceptive convolution |

### Perceptual Noise

| Field | Type | Static | Description |
|-------|------|--------|-------------|
| `perceptual_noise_enabled` | bool | yes | Enable noise application |
| `noise_modality_order` | tuple | yes | Ordered modality names from YAML key order; used to index noise arrays |
| `noise_modes` | `[13]` int32 | no | Per-modality noise mode: 0=None, 1=Constant, 2=State-Dependent |
| `noise_sigmas` | `[13]` float32 | no | Base standard deviation per modality |
| `noise_injury_scales` | `[13]` float32 | no | Injury-scaling factor α per modality (used in mode 2) |
| `noise_clip_min` | `[13]` float32 | no | Per-modality observation lower bound after noise |
| `noise_clip_max` | `[13]` float32 | no | Per-modality observation upper bound after noise |

---

## Pytree Registration & Flax struct.dataclass

Flax `@struct.dataclass` automatically registers both classes as JAX pytrees. This means:

- `jax.jit`, `jax.vmap`, `jax.lax.scan` can accept them as inputs/outputs without manual tree registration.
- **`pytree_node=False` fields** (e.g. `height`, `width`, `placement_mode`) are treated as **static**. JAX traces a new compiled function any time these values change. They are safe to use in Python-level `if` statements inside JIT.
- **All other fields** are pytree leaves; their values can change freely without recompilation.
- **Immutability**: Flax structs are immutable after creation. To produce an updated state, call `state.replace(field=new_value)` (also exposed as `state._replace(...)` for API compatibility). This is zero-copy when possible under XLA.
- **`vmap` batching**: `ParallelEnv` vmaps over the leading dimension of `EnvState` arrays. Each env in the batch has its own independent state pytree — conceptually a batch of `EnvState` objects stacked along axis 0.

---

## Reset Values (What's in `EnvState` at step 0)

Set in `core.py:749-778` during `jax_reset`. Use this as a single authoritative reference.

| Field | Value at reset | Source |
|-------|----------------|--------|
| `agent_pos` | `start_pos` **or** uniform random cell (if `random_start_pos=True`) | `core.py:634` |
| `current_step` | `0` | `core.py:751` |
| `last_action` | `4` if `rest_action_enabled` else `5` — i.e. Rest/Stay | `core.py:773` |
| `res_active` | all `True` | `core.py:753` |
| `res_cons_count` | all `0` | `core.py:754` |
| `res_reg_timer` | all `0` | `core.py:755` |
| `pred_state` | all `0` (Patrol) | `core.py:758` |
| `pred_stamina` | `pred_max_stamina[p]` (full) | `core.py:759` |
| `pred_move_timer` | all `0` | `core.py:760` |
| `pred_attack_timer` | all `0` | `core.py:761` |
| `nutrition` | `start_nutrition` **or** `Uniform(max_nutrition/2, max_nutrition)` if `random_start_nutrition=True` | `core.py:720-724` |
| `satiation` | **always derived** from nutrition: `max_satiation × (nutrition/max_nutrition)^k` | `core.py:727-728` |
| `injury_level` | `0.0` **or** `Uniform(0, max_injury/2)` if `random_start_injury=True` | `core.py:730-734` |
| `injury_buffer` | zeros of length `smoothing_duration` | `core.py:736` |
| `nociception_history_buffer` | zeros of length `interoceptive_kernel_length` | `core.py:739` |
| `last_collision_noc` | `0.0` | `core.py:769` |
| `rest_streak` | `0` | `core.py:770` |
| `terminated` | `False` | `core.py:771` |
| `*_property_sampled` | `clip(mean + std * N(0, 1), 0, 1)` — Gaussian draw per entity | `core.py:740-747` |

---

## Clarifications / FAQ

**Q: Why do entities have both `res_property` (in params) and `res_property_sampled` (in state)?**
A: `res_property` is the configured **mean** chemical signature (constant across episodes); `res_property_std` is the standard deviation. At reset/respawn, each entity samples `sampled = clip(mean + std * N(0,1), 0, 1)` (`core.py:740-742`). This gives per-entity per-episode olfactory variation — two food resources with the same mean signature can smell slightly different. The sampled vector is what sensors read.

**Q: The state tables show `vector_size = 5`. Is that hardcoded?**
A: `5` is whatever shape your YAML `property: [v1, v2, v3, v4, v5]` has; `olfactory_vector_size` is derived from the config (`config_loader.py`). All entities in one config must share the same vector length. In practice every shipped config uses 5, hence the table.

**Q: Are `start_satiation` and `random_start_satiation` actually used?**
A: **No.** They are loaded into `EnvParams` by `config_loader.py` but never read by `core.py`. Satiation at reset is always derived from nutrition via `S = max_S × (N/max_N)^k`. These fields are legacy — leaving them in YAML has no effect. Prefer controlling starting satiation indirectly through `start_nutrition` / `random_start_nutrition`.

**Q: Why is `noise_*` shape `[13]` when there are only 10 modalities?**
A: The 13 is a fixed padding size (`config_loader.py:401`: `pad = max(0, 13 - len(noise_modality_order))`). Active modalities fill the leading slots per `noise_modality_order`, and the remaining slots are zero-padded so the array shape stays static under JIT. Indexing must always go through `noise_modality_order` or `modality_map` — raw index positions are not semantically meaningful beyond whatever order your YAML declared.

**Q: How is resource respawn timing controlled?**
A: When a resource is consumed, `res_active` flips to `False` and `res_reg_timer` is set to `res_reg_delay` (see doc `08_resources_and_obstacles.md`). Each step, inactive resources with `res_reg_timer > 0` count down (`core.py:115-123`). When the timer hits 0, `respawn_mask` fires: `res_active` flips back to `True` and a fresh `res_property_sampled` is drawn. If `res_max_cons` has been exhausted, the resource stays inactive permanently (set `res_max_cons = -1` for unlimited respawns).

**Q: What are the integer values for `last_action`?**
A: `0–3` = movement (Up, Down, Left, Right), `4` = Rest (only if `rest_action_enabled`), `5` = Eat (only if `eat_action_enabled`). `action_dim = 4 + rest_enabled + eat_enabled` (`config_loader.py:330`). The proprioception sensor one-hot-encodes this index.

**Q: Does `terminated=True` get reset automatically?**
A: Only on an explicit call to `jax_reset` — the step function itself never flips `terminated` back. In `ParallelEnv`, the wrapper detects `terminated=True` and calls `reset` on those environments (see doc `11_parallel_env_wrapper.md`).

**Q: When does `injury_buffer` get populated?**
A: Whenever damage is applied in a step, the raw damage value is distributed across `smoothing_duration` slots of the ring buffer (a uniform-spread injection). Each step, `injury_buffer[0]` is consumed into `injury_level` and the buffer rotates. This models "pain over time" — a single hit slowly increases injury for `smoothing_duration` steps. Full mechanics in doc `05_body_homeostasis.md`.

**Q: What's the difference between `type_areas` and `grid_location_type`?**
A: They serve orthogonal purposes:
- `grid_location_type [H, W]` encodes **terrain** per cell (plain / grass / sand) — read by the location sensor and renderer.
- `type_areas [T, 4]` encodes **spawn region bounding boxes per entity type-group** — used only during entity placement at reset. It has no runtime effect after step 0.

**Q: What is `max_per_type` used for?**
A: It's the max entity count across any single type-group. Used as a **static loop bound** for the per-type placement pass (`per_type` mode, doc `03`). Making it static (`pytree_node=False`) means JAX compiles the loop once; changing the max_per_type triggers recompilation.

**Q: If `with_injury=False`, what happens on damage?**
A: The injury system is bypassed entirely and any nonzero damage terminates the episode immediately (no smoothing, no recovery). This is used in configs where the task is "touch-nothing-bad" survival, not pain modelling.

**Q: Does `overeating_death=True` kill at exactly `max_satiation` or above?**
A: At `satiation >= max_satiation` (i.e. equality triggers death). See `05_body_homeostasis.md`.

**Q: Is `EnvParams` immutable across an entire training run, or per-episode?**
A: Per-episode in principle, but most training setups reuse the same `EnvParams` for every episode in a run (it's built once from the YAML). Changing any static field (`pytree_node=False`) mid-run triggers JIT recompilation. Changing a non-static field (e.g. `max_nutrition`) is free at runtime but uncommon — the usual pattern is to rebuild `EnvParams` only for a new experiment.

**Q: Can I rely on `._replace(...)` vs `.replace(...)`?**
A: Yes — both work identically. The `._replace` alias exists for compatibility with earlier code that used `namedtuple._replace` semantics. Use either.
