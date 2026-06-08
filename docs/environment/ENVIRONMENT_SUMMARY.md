---
aliases: [environment_summary]
---

# GridWorld Pain — Environment Reference

> **Status**: Complete | **Primary source**: `src/environment/` | **Last updated**: 2026-06-08

GridWorld Pain is a JAX-based RL environment for **Interoceptive AI** research — agents must balance external reward-seeking with internal homeostatic regulation (hunger/satiation, injury avoidance). All environment logic is pure-functional; all parallelism uses `jax.vmap`.

---

## How to read these docs

These documents double as a **tutorial track** for readers who are new to JAX, Flax, or Orbax.

**Embedded implementation.** Each numbered doc shows the actual source functions verbatim, with captions that give the file and line number (e.g., `src/environment/core.py:372`). You are reading the real code, not a paraphrase of it.

**API notes convention.** Whenever a code block uses an advanced JAX, Flax, or Orbax API — `jax.lax.scan`, `@struct.dataclass`, `orbax.checkpoint.CheckpointManager`, and so on — a blockquote immediately below the block explains it:

> **API notes** — links each unfamiliar API back to the section in `00_jax_primer.md` that teaches it from scratch.

**Recommended reading order.** If you are learning the codebase from scratch, follow this path:

1. `00_jax_primer.md` — JAX, Flax, and Orbax foundations: pure functions, pytrees, JIT, vmap, PRNG, `@struct.dataclass`, Orbax checkpointing. Read this once; all other docs assume it.
2. `01` (state/params pytrees) then `02` (config loading) then `03` (reset) then `04` (step loop) — these four form the spine of the environment.
3. Subsystem docs: `05` (body/homeostasis), `06` (reward/termination), `07` (predator AI), `08` (resources and obstacles), `09` (sensors), `10` (perceptual noise).
4. `11` (parallel wrapper and vmap batching), `12` (renderer), `13` (checkpoint scheduling).

If you only need one subsystem, jump directly to its doc — each is self-contained — but skim `00` first so the API notes make sense.

---

## Table of Contents

| # | Document | Topic |
|---|----------|-------|
| — | **This file** | Hub: conventions, step-flow diagram, observation table, config summary |
| 00 | [JAX & Advanced-API Primer](00_jax_primer.md) | Tutorial: every advanced JAX/Flax/Orbax API the env uses, taught once |
| 01 | [State & Parameters](01_state_and_params.md) | `EnvState`, `EnvParams` pytree definitions |
| 02 | [Config Schema](02_config_schema.md) | YAML → `EnvParams` loading, mandatory keys, expansion |
| 03 | [Entity Placement](03_entity_placement.md) | `jax_reset`, `per_entity`/`per_type`, overlap resolution |
| 04 | [Step Loop](04_step_loop.md) | `jax_step` pipeline stages, data flow, info-dict |
| 05 | [Body & Homeostasis](05_body_homeostasis.md) | Nutrition/satiation dynamics, injury ring buffer, drive |
| 06 | [Reward & Termination](06_reward_and_termination.md) | Survival vs homeostatic reward, termination codes 0–4 |
| 07 | [Predator AI](07_predator_ai.md) | State machine, stamina, pursuit, bush concealment |
| 08 | [Resources & Obstacles](08_resources_and_obstacles.md) | Food/hiding_predator, regen timers, neutral animals, obstacles |
| 09 | [Sensors & Observation](09_sensors_and_observation.md) | All sensor functions, assembly, `get_observation_breakdown` |
| 10 | [Perceptual Noise](10_perceptual_noise.md) | Noise modes, modality order, state-dependent σ |
| 11 | [Parallel Env Wrapper](11_parallel_env_wrapper.md) | `ParallelEnv`, `auto_reset_step`, vmap, PRNG |
| 12 | [Renderer](12_renderer.md) | Telemetry dashboard, video export |
| 13 | [Checkpoint Scheduling](13_checkpoint_scheduling.md) | Gate drift under vmap, milestone collapse, resume semantics |

---

## JAX / Pytree Design Conventions

**`EnvState` vs `EnvParams`**: `EnvState` is mutable per-step data (positions, timers, body values); `EnvParams` is immutable per-episode configuration. Both are Flax `@struct.dataclass` pytrees.

**Static fields** (`struct.field(pytree_node=False)`): integer/bool/string fields in `EnvParams` that determine array shapes and JIT trace structure. These are treated as compile-time constants — changing them forces XLA recompilation. Examples: `height`, `width`, `placement_mode`, `use_homeostatic_reward`, `animal_is_damaging`, `visual_sensor_enabled`.

**Immutability**: never mutate state in-place. Always use `.replace(**kwargs)` (aliased as `._replace(...)`) to produce a new object. This is enforced by Flax's struct semantics.

**Pure-functional signatures**: `jax_reset(params, key) → EnvState` and `jax_step(state, action, params) → (EnvState, reward, done, info)`. No side effects, no global mutable state. Both decorated with `@jax.jit`.

**`vmap` axis conventions**: `ParallelEnv` vmaps over the leading axis (axis 0 = env index). `EnvState` batched over N envs has all arrays with a leading `[N, ...]` shape. `EnvParams` is broadcast (not batched) — single copy shared across all envs.

**PRNG key threading**: each `EnvState.key` is an independent PRNG stream. `jax_step` splits it into 6 sub-keys per step (`key, respawn_key, hunt_key, wander_key, damage_key, property_key`) and advances the main key forward. `jax_reset` splits 5-way (`key, agent_key, placement_key, body_key, property_key`) and derives an additional `animal_episode_key` via `jax.random.fold_in(property_key, 0xAE1)`. This ensures reproducibility: same initial key + same actions = same episode. (`core.py:372`, `core.py:781–783`)

---

## End-to-End Step Flow Diagram

```
jax_step(state, action, params)
│
├── 0. Split PRNG key → key, respawn_key, hunt_key, wander_key, damage_key, property_key
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
├── 3. UNIFIED ANIMAL UPDATE  ───────────────────────────────────────────────
│       update_animals(state, new_agent_pos, params, hunt_key, wander_key)
│       READ:  animal_pos, animal_state, animal_stamina, animal_move_timer,
│              animal_attack_timer, new_agent_pos, obs_pos, obs_hides_agent
│       WRITE: animal_pos, animal_state, animal_stamina, animal_move_timer,
│              animal_attack_timer
│       (hunt subset uses hunt_key; wander subset uses wander_key)
│
├── 4. INTERACTION  ──────────────────────────────────────────────────────────
│       resource overlaps, animal overlaps, obstacle collision/overlap
│       READ:  res_pos/active/type, animal_pos, obs_pos, obs_blocking,
│              obs_nociception, new_agent_pos, just_collided
│       COMPUTES: ate_food, total_damage (res+animal+obs), collision_noc
│       WRITE: res_cons_count, res_active, res_reg_timer, animal_attack_timer
│
├── 5. BODY UPDATE  ──────────────────────────────────────────────────────────
│       update_body(state, info, params)
│       READ:  satiation, nutrition, injury_level, injury_buffer, rest_streak, nociception_history_buffer
│       COMPUTES: new_satiation, new_nutrition, new_injury, done_from_body, new_nociception_history
│       WRITE: satiation, nutrition, injury_level, injury_buffer, rest_streak, nociception_history_buffer
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
| `[0]` | Injury | `injury_observable` | 1 | `[0, 1]` |
| `[1]` | Nutrition | `nutrition_observable` | 1 | `[0, 1]` |
| `[2]` | Satiation | Always | 1 | `[0, 1]` |
| `[3]` | Interoceptive Nociception | `interoceptive_nociception_enabled` | 1 | `[0, 1]` |
| `[4]` | Extero Nociception | `nociception_enabled` | 1 | `[0, 1]` |
| `[5:5+V]` | Olfaction | `olfactory_enabled` | V=`olfactory_vector_size` | `[0, ∞)` |
| `[5+V:5+V+C]` | Collision | Always | C=`2r²+2r+1` | `{0, 1}` |
| `+A` | Proprioception | `proprioception_enabled` | A=`action_dim` | `{0, 1}` |
| `+W` | Visual | `visual_sensor_enabled` | W=`(2r²+2r+1)×8` | `{0, 1}` |
| `+2` | Location | `location_sensor_enabled` | 2 | `[-1, 1]` |

Default config observation dimension (all sensors enabled, `r=1`, `vis_r=0`, `action_dim=6`):
- 1 + 1 + 1 + 1 + 5 + 5 + 6 + 8 + 0 = **28 dims** (location disabled by default)

Olfaction pools 3 chemical signals: `res_chem + animal_chem + obs_chem` (resources, all animals unified, obstacles). The perceptual-noise system has 10 configured modalities; noise arrays are zero-padded to a fixed static shape of 13 for JIT stability (`state.py:222–226`). `interoceptive_nociception` sits at index 3 in the modality order. `injury` and `nutrition` noise are silenced by default (σ_base=0.0). See [10](10_perceptual_noise.md).

---

## Config-to-EnvParams Mapping Summary

Quick-lookup for YAML path → `EnvParams` field:

| YAML path | EnvParams field | Type |
|-----------|----------------|------|
| `environment.height` | `height` | int (static) |
| `environment.width` | `width` | int (static) |
| `environment.max_steps` | `max_steps` | int (static) |
| `environment.placement.mode` | `placement_mode` | str (static) |
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
| `sensory.interoceptive_nociception_enabled` | `interoceptive_nociception_enabled` | bool (static) |
| `sensory.interoceptive_convolution_enabled` | `interoceptive_convolution_enabled` | bool (static) |
| `sensory.injury_observable` | `injury_observable` | bool (static) |
| `sensory.nutrition_observable` | `nutrition_observable` | bool (static) |
| `sensory.proprioception_enabled` | `proprioception_enabled` | bool (static) |
| `sensory.location_sensor` | `location_sensor_enabled` | bool (static) |
| `sensory.vector_size` | `olfactory_vector_size` | int (static) |
| `sensory.interoceptive_kernel_length` | `interoceptive_kernel_length` | int (static) |
| `perceptual_noise.enabled` | `perceptual_noise_enabled` | bool (static) |
| `perceptual_noise.modalities` (key order) | `noise_modality_order` | tuple (static) |
| `visualization.local_view_size` | `local_view_size` | int (static) |

---

## Cross-Doc Clarifications / FAQ

These are cross-cutting gotchas that span multiple environment docs. For in-depth FAQs on a specific topic, see the FAQ section at the bottom of each numbered doc.

**Q: Which `body.*` fields are loaded but never used?**
A: `body.start_satiation` and `body.random_start_satiation` are mandatory in the schema but never read by `core.py`. Satiation at reset is always derived from nutrition via the power-law factor. Setting them has no effect. See [01](01_state_and_params.md#clarifications--faq) and [03](03_entity_placement.md#clarifications--faq).

**Q: Does `overeating_death=True` actually terminate the episode?**
A: **No — latent bug.** The flag only sets `termination_reason=3` but does not trigger `done=True`. The only terminators are nutrition≤0, injury≥max_injury, and truncation. See [05](05_body_homeostasis.md#clarifications--faq) and [06](06_reward_and_termination.md#clarifications--faq).

**Q: Why does the agent sometimes spawn on top of a predator?**
A: Agent placement at reset does NOT participate in the entity occupancy mask. A `random_start_pos=True` agent can overlap any entity. Contact effects fire on step 0. See [03](03_entity_placement.md#clarifications--faq).

**Q: Why can two entities of the same kind land on the same cell after step 0?**
A: Resource respawn (`core.py:300-305`) doesn't check occupancy. Animals also have no inter-entity collision. Placement uniqueness is only enforced at reset. See [03](03_entity_placement.md#clarifications--faq), [07](07_predator_ai.md#clarifications--faq), [08](08_resources_and_obstacles.md#clarifications--faq).

**Q: Why does my YAML key `property` do nothing for a resource?**
A: All entities now use `properties` (plural) as the canonical key. If you use the legacy `property` key, it still works but emits a `DeprecationWarning`. A missing key on any entity now hard-fails with a `ValueError`. See [02](02_config_schema.md#per-entity-olfactory-yaml-keys).

**Q: Why does my new sensor crash `apply_perceptual_noise` with a `KeyError`?**
A: Every sensor present in `get_observation_breakdown` must have a corresponding entry in `perceptual_noise.modalities`. Set the mode to `none` if you want to skip noise for that sensor. Silent omission crashes. See [10](10_perceptual_noise.md#clarifications--faq).

**Q: Why are noise arrays shape `[13]` when I only configured 10 modalities?**
A: Zero-padded to a fixed static shape for JIT stability. The extra 3 slots are unused. See [02](02_config_schema.md#clarifications--faq) and [10](10_perceptual_noise.md#clarifications--faq).

**Q: Why is there a `terminated` field on `EnvState` and a `done` return value?**
A: `terminated` is stored for next-step logic (e.g. `ParallelEnv.auto_reset_step`); `done` is the per-step return. They carry the same information. The wrapper's auto-reset checks `done`. See [01](01_state_and_params.md#clarifications--faq) and [11](11_parallel_env_wrapper.md#clarifications--faq).

**Q: Why don't my checkpoint directories have round-number step keys like `100/, 200/`?**
A: Per-iteration gate + cumulative episode counter = drift. With `num_envs=128` and `checkpoint_frequency=100`, expect names like `156, 224, 312, ...`. See [13](13_checkpoint_scheduling.md).

**Q: Why does a rest-action step sometimes fail to heal my agent?**
A: `can_recover = rested AND applied_inc <= 0`. If the agent took damage on this step (`inc > 0`), the front of the injury buffer blocks recovery. Rest streak continues; recovery resumes next step if no new damage. See [05](05_body_homeostasis.md#clarifications--faq).

**Q: Do predators see through walls?**
A: Yes. Detection is pure Manhattan distance — no line-of-sight, no obstacle blocking. The only concealment is an obstacle with `hides_agent: true` (bush). See [07](07_predator_ai.md#clarifications--faq).

**Q: Does the renderer work with a batched `EnvState` from `ParallelEnv`?**
A: No. Index into the batch first (`jax.tree.map(lambda x: x[i], batched)`) then call the renderer. It is CPU-only and not JIT-compatible. See [12](12_renderer.md#clarifications--faq).

**Q: How do I confirm my observation layout matches what the noise system expects?**
A: Call `get_observation_breakdown(params)` after loading config. This is the single source of truth for observation dim mapping, and it's what the noise system uses to build `modality_map`. See [09](09_sensors_and_observation.md#clarifications--faq).

**Q: Which renderer file should I use?**
A: `renderer.py` (V1) is the production default used by all eval and record scripts. `renderer_v2.py` (V2) is an experimental card/pod layout that re-exports `save_jax_video` from V1 unchanged. `grid_world.py` is a legacy renderer copy retained for reference — it does not contain step logic. See [12](12_renderer.md).

**Q: What are the info dict keys returned by `jax_step`?**
A: Core keys include: `ate_food`, `damage`, `damage_hiding_predator`, `damage_predator`, `damage_obstacle`, `rested`, `hit_hiding_predator`, `hit_predator`, `hit_neutral`, `agent_in_bush`, `termination_reason`, `reward_homeostatic`, `reward_extrinsic`, `drive_hunger`, `drive_injury`, `metabolic_drain`, `event_collided`, `dist_to_food`, `dist_to_pred`, `dist_to_neutral`, `dist_to_hiding_predator`, `dist_per_predator`, `dist_per_neutral`, `dist_per_animal`. Note: `interacted_this_step` is a local variable inside `jax_step`, NOT an info key. `info['termination_reason']` is unreliable when `with_nutrition`/`with_injury` are both disabled (codes 2 and 4 never fire). See [04](04_step_loop.md).

**Q: What is the canonical entity concat order used for placement / position-splitting?**
A: `[resources | predator-class animals | obstacles | neutral-class animals]` — verified at `core.py:831` and `core.py:886–889`. This order is preserved byte-for-byte across `jax_reset` and `update_animals`.

**Q: How do I access positions of just predators or just neutrals?**
A: Use the host-side helper `select_by_class(params, 'predator')` / `select_by_class(params, 'neutral')` (defined in `state.py:7–28`) to get a boolean NumPy mask, then index `state.animal_pos`. Inside JIT use `params.predator_indices` / `params.neutral_indices` (static index tuples).
