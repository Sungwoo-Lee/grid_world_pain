# GridWorld Pain — Environment Reference

> **Status**: Complete | **Primary source**: `src/environment/` | **Last updated**: 2026-04-20

GridWorld Pain is a JAX-based RL environment for **Interoceptive AI** research — agents must balance external reward-seeking with internal homeostatic regulation (hunger/satiation, injury avoidance). All environment logic is pure-functional; all parallelism uses `jax.vmap`.

---

## Table of Contents

| # | Document | Topic |
|---|----------|-------|
| — | **This file** | Hub: conventions, step-flow diagram, observation table, config summary |
| 01 | [State & Parameters](01_state_and_params.md) | `EnvState`, `EnvParams` pytree definitions |
| 02 | [Config Schema](02_config_schema.md) | YAML → `EnvParams` loading, mandatory keys, expansion |
| 03 | [Entity Placement](03_entity_placement.md) | `jax_reset`, `per_entity`/`per_type`, overlap resolution |
| 04 | [Step Loop](04_step_loop.md) | `jax_step` pipeline stages, data flow, info-dict |
| 05 | [Body & Homeostasis](05_body_homeostasis.md) | Nutrition/satiation dynamics, injury ring buffer, drive |
| 06 | [Reward & Termination](06_reward_and_termination.md) | Survival vs homeostatic reward, termination codes 0–4 |
| 07 | [Predator AI](07_predator_ai.md) | State machine, stamina, pursuit, bush concealment |
| 08 | [Resources & Obstacles](08_resources_and_obstacles.md) | Food/danger, regen timers, neutral animals, obstacles |
| 09 | [Sensors & Observation](09_sensors_and_observation.md) | All sensor functions, assembly, `get_observation_breakdown` |
| 10 | [Perceptual Noise](10_perceptual_noise.md) | Noise modes, modality order, state-dependent σ |
| 11 | [Parallel Env Wrapper](11_parallel_env_wrapper.md) | `ParallelEnv`, `auto_reset_step`, vmap, PRNG |
| 12 | [Renderer](12_renderer.md) | Telemetry dashboard, video export |

---

## JAX / Pytree Design Conventions

**`EnvState` vs `EnvParams`**: `EnvState` is mutable per-step data (positions, timers, body values); `EnvParams` is immutable per-episode configuration. Both are Flax `@struct.dataclass` pytrees.

**Static fields** (`struct.field(pytree_node=False)`): integer/bool/string fields in `EnvParams` that determine array shapes and JIT trace structure. These are treated as compile-time constants — changing them forces XLA recompilation. Examples: `height`, `width`, `placement_mode`, `use_homeostatic_reward`, `predator_enabled`, `visual_sensor_enabled`.

**Immutability**: never mutate state in-place. Always use `.replace(**kwargs)` (aliased as `._replace(...)`) to produce a new object. This is enforced by Flax's struct semantics.

**Pure-functional signatures**: `jax_reset(params, key) → EnvState` and `jax_step(state, action, params) → (EnvState, reward, done, info)`. No side effects, no global mutable state. Both decorated with `@jax.jit`.

**`vmap` axis conventions**: `ParallelEnv` vmaps over the leading axis (axis 0 = env index). `EnvState` batched over N envs has all arrays with a leading `[N, ...]` shape. `EnvParams` is broadcast (not batched) — single copy shared across all envs.

**PRNG key threading**: each `EnvState.key` is an independent PRNG stream. `jax_step` splits it into 5 sub-keys per step and advances the main key forward. This ensures reproducibility: same initial key + same actions = same episode.

---

## End-to-End Step Flow Diagram

```
jax_step(state, action, params)
│
├── 0. Split PRNG key → respawn_key, predator_key, neutral_key, damage_key
│
├── 1. RESOURCE REGEN  ──────────────────────────────────────────────────────
│       update_resources(res_active, res_reg_timer, res_cons_count, params)
│       READ:  res_active, res_reg_timer, res_cons_count
│       WRITE: res_active, res_reg_timer, res_cons_count, res_pos (respawn)
│
├── 2. AGENT MOVEMENT  ──────────────────────────────────────────────────────
│       move_agent(agent_pos, action, obs_pos, obs_blocking, params)
│       READ:  agent_pos, action, obs_pos, obs_blocking
│       WRITE: agent_pos, last_action; produces: just_collided
│
├── 3. PREDATOR UPDATE  ─────────────────────────────────────────────────────
│       update_predators(..., new_agent_pos, ...)
│       READ:  pred_pos, pred_state, pred_stamina, pred_move_timer,
│              pred_attack_timer, new_agent_pos, obs_pos, obs_hides_agent
│       WRITE: pred_pos, pred_state, pred_stamina, pred_move_timer,
│              pred_attack_timer
│
├── 3.5 NEUTRAL ANIMAL UPDATE  ──────────────────────────────────────────────
│       update_neutral_animals(neutral_pos, neutral_move_timer, ...)
│       READ:  neutral_pos, neutral_move_timer
│       WRITE: neutral_pos, neutral_move_timer
│
├── 4. INTERACTION  ──────────────────────────────────────────────────────────
│       resource overlaps, predator overlaps, obstacle collision/overlap
│       READ:  res_pos/active/type, pred_pos, obs_pos, obs_blocking,
│              obs_nociception, new_agent_pos, just_collided
│       COMPUTES: ate_food, total_damage (res+pred+obs), collision_noc
│       WRITE: res_cons_count, res_active, res_reg_timer, pred_attack_timer
│
├── 5. BODY UPDATE  ──────────────────────────────────────────────────────────
│       update_body(state, info, params)
│       READ:  satiation, nutrition, injury_level, injury_buffer, rest_streak
│       COMPUTES: new_satiation, new_nutrition, new_injury, done_from_body
│       WRITE: satiation, nutrition, injury_level, injury_buffer, rest_streak
│
├── 6. TERMINATION  ──────────────────────────────────────────────────────────
│       reason codes 0-4, done = done_from_body OR truncated
│
├── 7. REWARD  ───────────────────────────────────────────────────────────────
│       homeostatic: prev_drive - curr_drive
│       survival:    +1 eat / -death_penalty terminal
│       eating_penalty applied in both modes
│
└── 8. STATE ASSEMBLY  ──────────────────────────────────────────────────────
        state._replace(all updated fields)
        returns: (new_state, reward, done, info)
```

---

## Observation Layout Reference Table

Authoritative sensor order (from `get_observation_breakdown()` in `sensor.py`).

| Slice | Sensor | Enabled by | Dim | Value range |
|-------|--------|-----------|-----|-------------|
| `[0]` | Injury | Always | 1 | `[0, 1]` |
| `[1]` | Nutrition | Always | 1 | `[0, 1]` |
| `[2]` | Satiation | Always | 1 | `[0, 1]` |
| `[3]` | Extero Nociception | `nociception_enabled` | 1 | `[0, 1]` |
| `[4:4+V]` | Olfaction | `olfactory_enabled` | V=`olfactory_vector_size` | `[0, ∞)` |
| `[4+V:4+V+C]` | Collision | Always | C=`2r²+2r+1` | `{0, 1}` |
| `+A` | Proprioception | `proprioception_enabled` | A=`action_dim` | `{0, 1}` |
| `+W` | Visual | `visual_sensor_enabled` | W=`(2r²+2r+1)×8` | `{0, 1}` |
| `+2` | Location | `location_sensor_enabled` | 2 | `[-1, 1]` |

Default config observation dimension (all sensors enabled, `r=1`, `vis_r=0`, `action_dim=6`):
- 1 + 1 + 1 + 1 + 5 + 5 + 6 + 8 + 0 = **28 dims** (location disabled by default)

---

## Config-to-EnvParams Mapping Summary

Quick-lookup for YAML path → `EnvParams` field:

| YAML path | EnvParams field | Type |
|-----------|----------------|------|
| `environment.height` | `height` | int (static) |
| `environment.width` | `width` | int (static) |
| `environment.max_steps` | `max_steps` | int (static) |
| `environment.placement.mode` | `placement_mode` | str (static) |
| `environment.predator_enabled` | `predator_enabled` | bool (static) |
| `environment.rest_action_enabled` | `rest_action_enabled` | bool (static) |
| `environment.eat_action_enabled` | `eat_action_enabled` | bool (static) |
| `environment.random_start_pos` | `random_start_pos` | bool (static) |
| `body.max_nutrition` | `max_nutrition` | float |
| `body.max_satiation` | `max_satiation` | float |
| `body.max_injury` | `max_injury` | float |
| `body.metabolic_cost` | `metabolic_cost` | float |
| `body.food_nutrition_gain` | `food_nutrition_gain` | float |
| `body.satiation_setpoint` | `setpoint` | float |
| `body.injury_smoothing_duration` | `smoothing_duration` | int (static) |
| `body.use_homeostatic_reward` | `use_homeostatic_reward` | bool (static) |
| `body.with_nutrition` | `with_nutrition` | bool (static) |
| `body.with_injury` | `with_injury` | bool (static) |
| `body.death_penalty` | `death_penalty` | float |
| `sensory.sensor_radius` | `sensor_radius` | float |
| `sensory.decay_power` | `sensor_decay` | float |
| `sensory.collision_sensor_range` | `sensor_range` | int (static) |
| `sensory.visual_sensor_enabled` | `visual_sensor_enabled` | bool (static) |
| `sensory.visual_sensor_range` | `visual_sensor_range` | int (static) |
| `sensory.olfactory_enabled` | `olfactory_enabled` | bool (static) |
| `sensory.nociception_enabled` | `nociception_enabled` | bool (static) |
| `sensory.proprioception_enabled` | `proprioception_enabled` | bool (static) |
| `sensory.location_sensor` | `location_sensor_enabled` | bool (static) |
| `sensory.vector_size` | `olfactory_vector_size` | int (static) |
| `perceptual_noise.enabled` | `perceptual_noise_enabled` | bool (static) |
| `perceptual_noise.modalities` (key order) | `noise_modality_order` | tuple (static) |
| `visualization.local_view_size` | `local_view_size` | int (static) |
