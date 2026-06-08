# 04 — Step Loop

> **Source**: `src/environment/core.py` (`jax_step`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## What this document is about

`jax_step` is the single function that advances the GridWorld Pain environment by one time step. It takes the current world state, the agent's chosen action, and a fixed set of environment parameters; it returns an updated state, a scalar reward, a done flag, and a diagnostic info dictionary.

This document describes exactly what happens inside that function, in order, with field-level precision. It was updated for v2.0, which merged the old separate predator and neutral-animal update pipelines into a single **unified animal entity** pipeline (`update_animals`). All state fields that previously used `pred_*` or `neutral_*` prefixes are now consolidated under `animal_*`. The `predator_enabled` runtime flag has been removed — whether predators exist is determined entirely by configuration.

---

## Signature and return

```python
@jax.jit
def jax_step(state: EnvState, action: int, params: EnvParams)
    -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict]
```
(`core.py:366`)

- `state` — the current `EnvState` (frozen Flax struct, immutable).
- `action` — scalar int32, range `[0, action_dim)`.
- `params` — the immutable `EnvParams` for this episode.
- Returns `(new_state, reward, done, info)`.

`new_state` is assembled via `state._replace(...)` at the very end. No intermediate variable mutates `state`; every stage reads from the same original snapshot and writes to local variables.

---

## ASCII Step-Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  state  (EnvState snapshot from previous step)                   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
        Stage 0 — PRNG key split (6 sub-keys)
                   │
        Stage 1 — Resource regeneration
                   │  reads: res_active, res_reg_timer, res_cons_count
                   │  writes: new_active, new_reg_timer, new_cons_count
                   │          res_pos_after_reg (respawn positions)
                   │          res_property_sampled_after_reg
                   │
        Stage 2 — Agent movement
                   │  reads: state.agent_pos, action, state.obs_pos, params.obs_blocking
                   │  writes: new_agent_pos, just_collided
                   │
        Stage 3 — Unified animal update (update_animals)
                   │  reads: state.animal_*, state.obs_pos, new_agent_pos
                   │  writes: new_animal_pos, new_animal_state, new_animal_stamina,
                   │          new_animal_mt, new_animal_at
                   │
        Stage 4 — Interaction & damage
                   │  sub: resource overlap → damage_res, ate_food, interacted_this_step
                   │  sub: animal damage    → damage_pred, new_animal_at (attack delay)
                   │  sub: obstacle damage  → damage_obs_overlap, damage_obs_collision
                   │       total_damage = sum of all three
                   │  builds early info dict (ate_food, damage, hit_*, hit_neutral ...)
                   │
        Stage 5 — Body / homeostasis update (update_body)
                   │  reads: info['ate_food'], info['damage'], info['rested']
                   │  writes: new_satiation, new_nutrition, new_injury,
                   │          next_injury_buffer, next_nociception_history,
                   │          new_rest_streak, done (body-driven)
                   │
  Termination: done = done_body OR truncated (next_step >= max_steps)
  info['termination_reason'] assigned here
                   │
        Stage 6 — Reward computation
                   │  writes: reward_homeostatic, reward_extrinsic, reward (scalar)
                   │          drive_hunger, drive_injury, metabolic_drain, event_collided,
                   │          dist_to_*, agent_in_bush appended to info
                   │
        Stage 7 — Final state assembly (state._replace)
                   │
┌──────────────────▼───────────────────────────────────────────────┐
│  (new_state, reward, done, info)                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## `move_agent` — Agent movement with branchless obstacle collision

`move_agent` computes where the agent ends up after applying `action`, clamping to the grid boundary, and checking every obstacle for a blocking collision — all without a single Python `if` branch. It returns the final position and a boolean `is_collision` flag.

Source: `src/environment/core.py:5–36`
```python
def move_agent(pos: jnp.ndarray, action: int, obs_pos: jnp.ndarray, obs_blocking: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    """Calculates New Agent position based on action, considering obstacles."""
    # 0: Up, 1: Right, 2: Down, 3: Left, 4+: Stay
    moves = jnp.array([
        [-1, 0], # Up
        [0, 1],  # Right
        [1, 0],  # Down
        [0, -1], # Left
        [0, 0],  # Rest/Stay
        [0, 0],  # Eat/Stay
    ], dtype=jnp.int32)
    
    # Clip action to valid range [0, 5]
    action = jnp.clip(action, 0, 5).astype(jnp.int32)
    move = moves[action]
    
    new_pos = pos + move
    # Clamp to grid boundaries
    new_pos = jnp.array([
        jnp.clip(new_pos[0], 0, params.height - 1),
        jnp.clip(new_pos[1], 0, params.width - 1)
    ])
    
    # Obstacle collision check
    is_collision = jnp.any(jnp.logical_and(
        jnp.all(obs_pos == new_pos, axis=-1),
        obs_blocking
    ))
    
    # If collision, stay at current position
    final_pos = jnp.where(is_collision, pos, new_pos)
    return final_pos, is_collision
```

> **API notes**
>
> - `moves[action]` is a **dynamic index** into a static 6×2 array. Under JIT the array is a compile-time constant; the `action` index is traced as a dynamic value. This is a `jnp.ndarray` gather, not a Python list subscript — see [primer: static vs. dynamic](00_jax_primer.md#static-dynamic).
> - `jnp.clip(new_pos[0], ...)` and `jnp.clip(new_pos[1], ...)` are branchless boundary clamps. There is no `if new_pos < 0` — the operation executes unconditionally on the traced value. See [primer: branchless](00_jax_primer.md#branchless).
> - `jnp.all(obs_pos == new_pos, axis=-1)` is a **masking** pattern: compare every obstacle's `[row, col]` against `new_pos` along the last axis to get a `[num_obs]` boolean. Then `jnp.logical_and(..., obs_blocking)` restricts to blocking obstacles. See [primer: masking](00_jax_primer.md#masking).
> - `jnp.where(is_collision, pos, new_pos)` selects the old position on a collision, the new position otherwise — no Python branch needed. See [primer: branchless](00_jax_primer.md#branchless).
> - The function has **no `@jax.jit`** of its own; it is called from inside the JIT-compiled `jax_step`. It does not need its own JIT boundary.

---

## `jax_step` — Full verbatim implementation

`jax_step` is the top-level step orchestrator. The `@jax.jit` decorator on line 366 means the entire body — all seven stages — is compiled to XLA once and then executed as a single GPU kernel. Every stage is presented below in order; cross-call dependencies (`update_resources`, `update_animals`, `update_body`) are covered in their own docs and shown only at their call sites here.

### Stage 0 — PRNG Key Split

Source: `src/environment/core.py:366–372`
```python
@jax.jit
def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Orchestrates a full environment step in JAX."""
    
    # 0. Split key for random events
    # Preserve today's 6-way split byte-for-byte (rename predator_key→hunt_key, neutral_key→wander_key).
    key, respawn_key, hunt_key, wander_key, damage_key, property_key = jax.random.split(state.key, 6)
```

> **API notes**
>
> - `@jax.jit` on the function definition is the JIT boundary for the whole step. JAX traces `jax_step` once (with abstract shapes), produces an XLA computation, and caches it. All subsequent calls hit the cache and run at compiled speed. See [primer: jit](00_jax_primer.md#jit).
> - `jax.random.split(state.key, 6)` produces **6 independent sub-keys** from one parent key in a single call. JAX's PRNG is stateless and functional — there is no global seed to mutate. The original `state.key` is never modified; `key` (the first element of the split) is what gets stored back into `new_state.key` at Stage 7, seeding the next step's split. See [primer: prng](00_jax_primer.md#prng).
> - Splitting 6 ways at once (rather than 6 separate `split` calls) is idiomatic and efficient: one PRNG kernel invocation, deterministic across all platforms.
> - `damage_key` is reused for all three damage-sample calls in Stage 4. The draws are correlated across sources within a step because they share the same seed. In practice only masked entries contribute, so this has no observable effect on game dynamics — but the samples are not statistically independent.

---

### Stage 1 — Resource Regeneration

Source: `src/environment/core.py:374–395`
```python
    # 1. Resource Regeneration (before agent moves)
    new_active, new_reg_timer, new_cons_count, respawn_mask = update_resources(
        state.res_active, state.res_reg_timer, state.res_cons_count, params
    )

    # Displace resources that just respawned
    num_res = params.res_type.shape[0]
    res_keys = jax.random.split(respawn_key, num_res)

    def sample_res_pos(rk, area):
        return jax.random.randint(rk, (2,), area[:2], area[2:])

    new_potential_pos = jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)
    # Only update position IF respawn_mask is true for that resource
    res_pos_after_reg = jnp.where(respawn_mask[:, None], new_potential_pos, state.res_pos)

    # Re-sample chemical property for respawned resources
    noise = jax.random.normal(property_key, shape=params.res_property.shape)
    new_sampled_prop = jnp.clip(params.res_property + params.res_property_std * noise, 0.0, 1.0)
    res_property_sampled_after_reg = jnp.where(
        respawn_mask[:, None], new_sampled_prop, state.res_property_sampled
    )
```

> **API notes**
>
> - `update_resources(...)` is documented in [08_resource_system.md](08_resource_system.md). Only its return values are used here.
> - `jax.random.split(respawn_key, num_res)` produces one sub-key per resource so each resource gets an independent random position draw. `num_res` is a **static** shape value (compile-time constant); JAX requires array shapes to be static at trace time. See [primer: static vs. dynamic](00_jax_primer.md#static-dynamic).
> - `jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)` maps `sample_res_pos` over the `(num_res,)` axis of both `res_keys` and `params.res_spawn_area` simultaneously — one random integer pair per resource with zero Python loops. See [primer: vmap](00_jax_primer.md#vmap).
> - `jnp.where(respawn_mask[:, None], new_potential_pos, state.res_pos)` is a **branchless conditional update**: `[:, None]` broadcasts the `(num_res,)` boolean mask to `(num_res, 2)` so it aligns with the position array. Resources whose `respawn_mask` is `False` keep their old position. See [primer: branchless](00_jax_primer.md#branchless) and [primer: masking](00_jax_primer.md#masking).
> - The same `jnp.where` pattern repeats for `res_property_sampled_after_reg`: only newly respawned resources receive a fresh chemical sample.

---

### Stage 2 — Agent Movement (call site)

Source: `src/environment/core.py:397–398`
```python
    # 2. Agent Movement
    new_agent_pos, just_collided = move_agent(state.agent_pos, action, state.obs_pos, params.obs_blocking, params)
```

> **API notes**
>
> - This is the call site for `move_agent`, which is fully embedded above. `obs_pos` comes from `state` (not `params`) because obstacle positions are randomised per episode at reset. `obs_blocking` is a `[num_obs] bool` from `params`.

---

### Stage 3 — Unified Animal Update (call site)

Source: `src/environment/core.py:400–405`
```python
    # 3. Unified Animal Update (per-subset call pattern — B1 fix)
    # hunt_key feeds hunt subset (N_pred draw shapes, byte-identical to old predator_key).
    # wander_key feeds wander subset (N_neutral draw shapes, byte-identical to old neutral_key).
    new_animal_pos, new_animal_state, new_animal_stamina, new_animal_mt, new_animal_at = update_animals(
        state, new_agent_pos, params, hunt_key, wander_key
    )
```

> **API notes**
>
> - `update_animals` is documented in [07_predator_ai.md](07_predator_ai.md). It dispatches `_hunt_step` and `_wander_step` per-behaviour subset using static index tuples frozen in `params`. The call signature passes `new_agent_pos` (Stage 2 output) so animals react to where the agent has already moved this step.
> - `hunt_key` and `wander_key` are the two sub-keys from Stage 0 allocated for animal movement; their draw shapes are byte-identical to the pre-v2.0 `predator_key` / `neutral_key` split to preserve PRNG reproducibility across the refactor.

---

### Stage 4 — Interaction and Damage

Source: `src/environment/core.py:407–530`

#### 4a. Resource overlap setup

```python
    # 4. Interaction Logic
    # Check overlaps with resources (using positions AFTER regeneration)
    at_resource = jnp.all(res_pos_after_reg == new_agent_pos, axis=-1)
    interact_resource = jnp.logical_and(at_resource, new_active)
    
    # Calculate attempted position for collision damage targeting
    moves_map = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0], [0, 0]], dtype=jnp.int32)
    attempted_pos = state.agent_pos + moves_map[jnp.clip(action, 0, 5).astype(jnp.int32)]
    # Clamp to grid boundaries
    attempted_pos = jnp.array([
        jnp.clip(attempted_pos[0], 0, params.height - 1),
        jnp.clip(attempted_pos[1], 0, params.width - 1)
    ])
```

> **API notes**
>
> - `jnp.all(res_pos_after_reg == new_agent_pos, axis=-1)` broadcasts the scalar `new_agent_pos` against the `[num_res, 2]` position array and reduces along `axis=-1` (the coordinate axis) to produce a `[num_res]` boolean. This is the canonical **masking** pattern for "which entities share my cell". See [primer: masking](00_jax_primer.md#masking).
> - `attempted_pos` recomputes the move delta independently inside `jax_step` (rather than reading it from `move_agent`) so the collision damage stage can target the specific obstacle the agent walked into. This is safe because `move_agent` used the same `moves_map` logic.

#### 4b. Resource damage and food interaction

Source: `src/environment/core.py:421–465`
```python
    # Hiding Predator interaction (Auto)
    is_hiding_predator = params.res_type == 1
    # Sample damage for each resource interaction
    sampled_res_damage = jax.random.uniform(damage_key, (params.res_type.shape[0],), 
                                           minval=params.res_damage[:, 0], 
                                           maxval=params.res_damage[:, 1])
    damage_res = jnp.sum(jnp.where(jnp.logical_and(interact_resource, is_hiding_predator), sampled_res_damage, 0.0))
    
    # Food interaction (Action-based or Auto)
    is_food = params.res_type == 0
    
    # Determine if eat action was triggered based on config
    eat_action_idx = jnp.where(params.rest_action_enabled, 5, 4)
    eat_action_triggered = jnp.logical_and(params.eat_action_enabled, action == eat_action_idx)
    
    # Auto-eat occurs if eat action is disabled and agent is on food
    ate_food_auto = jnp.logical_and(jnp.logical_not(params.eat_action_enabled), 
                                   jnp.logical_and(interact_resource, is_food))
    
    # Final 'ate_food' flag (used for satiation and lifecycle)
    ate_food = jnp.any(jnp.logical_or(
        ate_food_auto,
        jnp.logical_and(jnp.logical_and(interact_resource, is_food), eat_action_triggered)
    ))
    
    # Fix eat_triggered for lifecycle update (both auto and action)
    eat_lifecycle_triggered = jnp.logical_and(interact_resource, is_food)
    eat_lifecycle_triggered = jnp.logical_and(eat_lifecycle_triggered, 
                                             jnp.logical_or(jnp.logical_not(params.eat_action_enabled), eat_action_triggered))
    
    # Final interact mask for lifecycle update
    interacted_this_step = jnp.logical_or(
        jnp.logical_and(interact_resource, is_hiding_predator),
        eat_lifecycle_triggered
    )
    
    # Update Resource Lifecycle (Consumption)
    next_cons_count = new_cons_count + jnp.where(interacted_this_step, 1, 0)
    # Deactivate if exceeded max_cons (if max_cons > 0)
    # CRITICAL FIX: Only deactivate if it was active to avoid resetting timer during deactivation phase
    should_deactivate = jnp.logical_and(new_active, 
                                        jnp.logical_and(params.res_max_cons > 0, next_cons_count >= params.res_max_cons))
    final_active = jnp.where(should_deactivate, False, new_active)
    # Set reg timer
    next_reg_timer = jnp.where(should_deactivate, params.res_reg_delay, new_reg_timer)
```

> **API notes**
>
> - `jax.random.uniform(damage_key, (params.res_type.shape[0],), minval=..., maxval=...)` draws one sample per resource from its individual `[low, high]` range. The `minval` and `maxval` are `[num_res]` arrays sliced from `params.res_damage`. All `num_res` draws happen in one kernel call. See [primer: prng](00_jax_primer.md#prng).
> - `jnp.sum(jnp.where(..., sampled_res_damage, 0.0))` is the **mask-then-sum** idiom: fill non-matching entries with 0 before reducing, rather than filtering. This keeps the shape static. See [primer: masking](00_jax_primer.md#masking).
> - `eat_action_idx = jnp.where(params.rest_action_enabled, 5, 4)` selects the eat-action index branchlessly based on a `params` boolean. Even though `params.rest_action_enabled` is a static config value, writing it as `jnp.where` keeps the code JIT-safe if the param is ever traced. See [primer: branchless](00_jax_primer.md#branchless).
> - `interacted_this_step` is a **local variable only** — it never appears in the `info` dict. It is used solely to increment `next_cons_count` for resource lifecycle tracking.

#### 4c. Animal damage and obstacle damage

Source: `src/environment/core.py:470–530`
```python
    # ate_food is already calculated above
    
    # Predator Damage (unified — B5 fix: use at_damaging for damage + hit_predator)
    at_animal = jnp.all(new_animal_pos == new_agent_pos, axis=-1)         # POST-step positions
    at_damaging = jnp.logical_and(at_animal, params.animal_is_damaging)
    # Sample animal damage (preserving today's draw shape = all N animals, using damage_key)
    sampled_pred_damage = jax.random.uniform(damage_key, (params.animal_damage.shape[0],),
                                             minval=params.animal_damage[:, 0],
                                             maxval=params.animal_damage[:, 1])
    damage_pred = jnp.sum(jnp.where(at_damaging, sampled_pred_damage, 0.0))

    # Trigger Attack Delay for damaging animals that hit the agent
    new_animal_at = jnp.where(at_damaging, params.animal_attack_delay, new_animal_at)
    
    # Rock/Obstacle Damage
    # 1. Overlap damage (non-blocking rocks at current pos)
    at_obs = jnp.all(state.obs_pos == new_agent_pos, axis=-1)
    # Sample obstacle damage
    sampled_obs_damage = jax.random.uniform(damage_key, (params.obs_damage.shape[0],),
                                           minval=params.obs_damage[:, 0],
                                           maxval=params.obs_damage[:, 1])
    damage_obs_overlap = jnp.sum(jnp.where(jnp.logical_and(at_obs, jnp.logical_not(params.obs_blocking)), sampled_obs_damage, 0.0))
    
    # 2. Collision damage (blocking rocks)
    # Target the specific obstacle we hit
    at_attempted_obs = jnp.all(state.obs_pos == attempted_pos, axis=-1)
    damage_obs_collision = jnp.where(just_collided, jnp.max(jnp.where(at_attempted_obs, sampled_obs_damage, 0.0), initial=0.0), 0.0)
    
    # Calculate collision NOC intensity for sensing
    collision_noc = jnp.where(just_collided, jnp.max(jnp.where(at_attempted_obs, params.obs_nociception, 0.0), initial=0.0), 0.0)
    
    total_damage = damage_res + damage_pred + damage_obs_overlap + damage_obs_collision
    
    # 5. Body Update
    # rested is always action 4 if enabled
    rested = jnp.logical_and(params.rest_action_enabled, action == 4)
    
    damage_hiding_predator = damage_res
    
    # hit_neutral uses PRE-step animal positions (B5 fix: preserves today's pre/post asymmetry).
    # Today's code: hit_predator uses new_pred_pos (post-move); hit_neutral uses state.neutral_pos (pre-move).
    # We reproduce this exactly:
    #   at_damaging (above) → POST-step → hit_predator
    #   at_neutral_pre → PRE-step state.animal_pos masked by ~animal_is_damaging → hit_neutral
    at_neutral_pre = (
        jnp.logical_and(
            jnp.all(state.animal_pos == new_agent_pos, axis=-1),
            ~params.animal_is_damaging
        )
        if state.animal_pos.shape[0] > 0 else jnp.zeros(0, dtype=jnp.bool_)
    )

    info = {
        'ate_food': ate_food,
        'damage': total_damage,
        'damage_hiding_predator': damage_hiding_predator,
        'damage_predator': damage_pred,
        'damage_obstacle': damage_obs_overlap + damage_obs_collision,
        'rested': rested,
        'hit_hiding_predator': jnp.any(jnp.logical_and(interact_resource, is_hiding_predator)),
        'hit_predator': jnp.any(at_damaging),
        'hit_neutral': jnp.any(at_neutral_pre) if state.animal_pos.shape[0] > 0 else jnp.array(False),
    }
```

> **API notes**
>
> - `new_animal_at = jnp.where(at_damaging, params.animal_attack_delay, new_animal_at)` is a **scatter-like** branchless update: set the attack timer to `params.animal_attack_delay` for every animal that hit the agent this step, leave others unchanged. See [primer: branchless](00_jax_primer.md#branchless) and [primer: scatter-index](00_jax_primer.md#scatter-index).
> - `jnp.max(jnp.where(at_attempted_obs, sampled_obs_damage, 0.0), initial=0.0)` uses `initial=0.0` so the reduction is safe even when `at_attempted_obs` is all-False (empty input guard). Without `initial`, `jnp.max` on an all-masked array returns `-inf`.
> - The `if state.animal_pos.shape[0] > 0 else ...` guard is a **static shape check** — Python-level, evaluated at trace time, not at runtime. JAX traces each branch as a separate path and caches the compiled function for the specific shapes. This is the M3 zero-N fallback pattern used throughout the codebase. See [primer: static vs. dynamic](00_jax_primer.md#static-dynamic).
> - `damage_key` is reused (not re-split) for all three `jax.random.uniform` calls — resources, animals, and obstacles. The draws are correlated within a step but independent across steps because `damage_key` itself changes each step via the 6-way split in Stage 0. See [primer: prng](00_jax_primer.md#prng).

---

### Stage 5 — Body Update (call site)

Source: `src/environment/core.py:532`
```python
    new_satiation, new_nutrition, new_injury, next_injury_buffer, next_nociception_history, new_rest_streak, done = update_body(state, info, params)
```

> **API notes**
>
> - `update_body` is documented in [05_body_homeostasis.md](05_body_homeostasis.md). The `info` dict passed here contains the Stage 4 outputs (`ate_food`, `damage`, `rested`). `update_body` reads those keys and returns purely JAX arrays — no Python conditionals inside its body once tracing is done.
> - `done` returned here is `done_from_body` — it fires from starvation (`new_nutrition <= 0`) or lethal injury (`new_injury >= max_injury`). It is ORed with `truncated` in the termination block below.

---

### Termination Block

Source: `src/environment/core.py:534–548`
```python
    # Max Steps Truncation
    next_step = state.current_step + 1
    truncated = next_step >= params.max_steps
    
    # Termination Reason (Integer codes for JIT compatibility)
    # 0: active, 1: max_steps, 2: starvation, 3: overeating, 4: injury
    reason = jnp.array(0, dtype=jnp.int32)
    reason = jnp.where(truncated, 1, reason)
    reason = jnp.where(new_nutrition <= 0.0, 2, reason)
    if params.overeating_death:
        reason = jnp.where(new_satiation >= params.max_satiation, 3, reason)
    reason = jnp.where(new_injury >= params.max_injury, 4, reason)
    
    info['termination_reason'] = reason
    done = jnp.logical_or(done, truncated)
```

> **API notes**
>
> - Each `reason = jnp.where(cond, new_code, reason)` call **overwrites** the previous value if the condition is true. Because `injury` is assigned last, it wins over all earlier conditions if multiple termination criteria fire simultaneously.
> - `if params.overeating_death:` is a **static Python branch** — `params.overeating_death` is a Python bool, not a JAX array, so JAX sees only one of the two code paths at trace time. This is intentional: the JIT cache has a separate compiled function for `overeating_death=True` vs `False`. See [primer: static vs. dynamic](00_jax_primer.md#static-dynamic).
> - `jnp.logical_or(done, truncated)` is branchless OR. Both `done_from_body` and `truncated` can fire independently. See [primer: branchless](00_jax_primer.md#branchless).

---

### Stage 6 — Reward Computation

Source: `src/environment/core.py:550–624`
```python
    # 6. Reward (Homeostatic driven by Satiation)
    reward_homeostatic = 0.0
    reward_extrinsic = 0.0
    
    # Calculate components for analysis
    # drive = (1 - satiation/100)^2 + (injury/100)^2
    drive_hunger = jnp.power(1.0 - (new_satiation / params.max_satiation), 2)
    drive_injury = jnp.power(new_injury / params.max_injury, 2)
    
    if params.use_homeostatic_reward:
        prev_drive = calculate_drive(state.satiation, state.injury_level, params)
        curr_drive = calculate_drive(new_satiation, new_injury, params)
        reward_homeostatic = prev_drive - curr_drive
        # Death penalty based on Nutrition starvation
        reward_homeostatic = jnp.where(done, reward_homeostatic - params.death_penalty, reward_homeostatic)
    else:
        reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
        reward_extrinsic = jnp.where(done, -params.death_penalty, reward_extrinsic)
    
    reward = reward_homeostatic + reward_extrinsic
    # Apply eating penalty if ate food
    reward = jnp.where(ate_food, reward - params.eating_reward_penalty, reward)
    
    info['reward_homeostatic'] = reward_homeostatic
    info['reward_extrinsic'] = reward_extrinsic
    info['drive_hunger'] = drive_hunger
    info['drive_injury'] = drive_injury
    info['metabolic_drain'] = params.metabolic_cost
    info['event_collided'] = just_collided
    
    # GPU-side distance calculations for stats
    # M3 zero-N fallback: preserve today's `if state.pred_pos.shape[0] > 0 else 99.0` pattern.
    dist_to_food = jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 0), jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1), 99.0)) if state.res_pos.shape[0] > 0 else 99.0
    dist_to_hiding_predator = jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 1), jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1), 99.0)) if state.res_pos.shape[0] > 0 else 99.0

    # Unified animal distances
    dist_per_animal = (
        jnp.linalg.norm(state.animal_pos - new_agent_pos, axis=-1)
        if state.animal_pos.shape[0] > 0
        else jnp.zeros((0,), dtype=jnp.float32)
    )
    # Legacy aliases — kept for one release cycle (C5); computed from per-class masks.
    if len(params.predator_indices) > 0:
        pred_mask = params.animal_is_damaging  # predator_class ≡ damaging in current schema
        dist_to_pred = jnp.min(jnp.where(pred_mask, dist_per_animal, 99.0))
        dist_per_predator = dist_per_animal[jnp.array(params.predator_indices, dtype=jnp.int32)]
    else:
        dist_to_pred = 99.0
        dist_per_predator = jnp.zeros((0,), dtype=jnp.float32)

    if len(params.neutral_indices) > 0:
        neutral_mask = ~params.animal_is_damaging
        dist_to_neutral = jnp.min(jnp.where(neutral_mask, dist_per_animal, 99.0))
        dist_per_neutral = dist_per_animal[jnp.array(params.neutral_indices, dtype=jnp.int32)]
    else:
        dist_to_neutral = 99.0
        dist_per_neutral = jnp.zeros((0,), dtype=jnp.float32)

    info['dist_to_food'] = dist_to_food
    info['dist_to_pred'] = dist_to_pred
    info['dist_to_neutral'] = dist_to_neutral
    info['dist_to_hiding_predator'] = dist_to_hiding_predator
    info['dist_per_neutral'] = dist_per_neutral
    info['dist_per_predator'] = dist_per_predator
    info['dist_per_animal'] = dist_per_animal

    # Bush occupancy: True iff agent is standing on an obstacle marked hides_agent.
    # Mirrors the agent_hidden computation inside update_predators (line ~150);
    # recomputed here at minimal cost because EnvParams is in scope and we want it on `info`.
    # Uses new_agent_pos (post-step position) — correct for M2's "agent dives into bush" semantics.
    agent_in_bush = jnp.any(jnp.logical_and(
        jnp.all(state.obs_pos == new_agent_pos, axis=-1),
        params.obs_hides_agent
    )) if state.obs_pos.shape[0] > 0 else jnp.array(False)
    info['agent_in_bush'] = agent_in_bush
```

> **API notes**
>
> - `if params.use_homeostatic_reward:` is another **static Python branch** — the compiled function exists in two flavours (one per reward mode). Both branches use `jnp.where` internally rather than Python `if/else` for the death-penalty toggle so the scalar `done` flag remains a traced JAX value. See [primer: static vs. dynamic](00_jax_primer.md#static-dynamic).
> - `calculate_drive(state.satiation, state.injury_level, params)` is documented at `core.py:38`. It calls `jnp.linalg.norm` on a 2-element stack `[satiation, injury]`. See [primer: linalg](00_jax_primer.md#linalg).
> - `jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1)` broadcasts `new_agent_pos` (shape `(2,)`) against `state.res_pos` (shape `(num_res, 2)`) to get per-resource Euclidean distances in one call. See [primer: linalg](00_jax_primer.md#linalg).
> - `jnp.min(jnp.where(mask, distances, 99.0))` is the **masked minimum** idiom: unmasked entries become 99.0 (a sentinel meaning "no entity present") so `jnp.min` returns 99.0 when no active entity exists. See [primer: masking](00_jax_primer.md#masking).
> - `dist_per_animal[jnp.array(params.predator_indices, dtype=jnp.int32)]` is a **gather** by dynamic index: select the per-animal distances for the predator subset. `params.predator_indices` is a static Python list; wrapping it in `jnp.array` converts it to a JAX index tensor. See [primer: scatter-index](00_jax_primer.md#scatter-index).
> - The `if state.res_pos.shape[0] > 0 else 99.0` and `if len(params.predator_indices) > 0` guards are Python-level static shape checks at trace time (M3 zero-N fallback pattern). See [primer: static vs. dynamic](00_jax_primer.md#static-dynamic).

---

### Stage 7 — Final State Assembly

Source: `src/environment/core.py:626–659`
```python
    # 7. Final State
    new_state = state._replace(
        agent_pos=new_agent_pos,
        current_step=next_step,
        res_pos=res_pos_after_reg,
        res_active=final_active,
        res_reg_timer=next_reg_timer,
        res_cons_count=next_cons_count,
        res_property_sampled=res_property_sampled_after_reg,
        # Unified animal fields (positions + state mutate; sampled distributional fields unchanged in step)
        animal_pos=new_animal_pos,
        animal_state=new_animal_state,
        animal_stamina=new_animal_stamina,
        animal_move_timer=new_animal_mt,
        animal_attack_timer=new_animal_at,
        animal_property_sampled=state.animal_property_sampled,
        animal_detect_sampled=state.animal_detect_sampled,
        animal_max_stamina_sampled=state.animal_max_stamina_sampled,
        animal_recovery_sampled=state.animal_recovery_sampled,
        animal_hunt_thresh_sampled=state.animal_hunt_thresh_sampled,
        animal_lose_interest_sampled=state.animal_lose_interest_sampled,
        satiation=new_satiation,
        nutrition=new_nutrition,
        injury_level=new_injury,
        injury_buffer=next_injury_buffer,
        nociception_history_buffer=next_nociception_history,
        last_collision_noc=collision_noc,
        rest_streak=new_rest_streak,
        terminated=done,
        key=key,
        last_action=jnp.array(action, dtype=jnp.int32),
    )
    
    return new_state, reward, done, info
```

> **API notes**
>
> - `state._replace(...)` is the **immutable update** pattern for Flax `@struct.dataclass`. It returns a brand-new `EnvState` object with the named fields swapped; the original `state` is unchanged. This is what makes `jax_step` pure — no side effects, same inputs always produce same outputs. See [primer: immutability](00_jax_primer.md#immutability) and [primer: jax-pytrees](00_jax_primer.md#jax-pytrees).
> - The six `animal_*_sampled` fields (`animal_property_sampled`, `animal_detect_sampled`, `animal_max_stamina_sampled`, `animal_recovery_sampled`, `animal_hunt_thresh_sampled`, `animal_lose_interest_sampled`) are passed through as `state.<field>` — they were sampled at reset and do not change during a step. Listing them explicitly in `_replace` is required for JAX tracing fidelity; omitting them would silently drop them from the traced pytree.
> - `key=key` stores the **first** element of the 6-way Stage 0 split — not any of the named sub-keys. Sub-keys (`respawn_key`, `hunt_key`, etc.) are consumed within this step and discarded. See [primer: prng](00_jax_primer.md#prng).
> - `last_action=jnp.array(action, dtype=jnp.int32)` stores the scalar action as a JAX int32 array in state so it is available as a pytree leaf. Without the explicit `jnp.array` cast, `action` might be a traced abstract value whose type JAX infers differently across compilations. See [primer: jax-pytrees](00_jax_primer.md#jax-pytrees).

---

## Stage 0 — PRNG Key Split (`core.py:372`)

```python
key, respawn_key, hunt_key, wander_key, damage_key, property_key = jax.random.split(state.key, 6)
```

Six sub-keys, **6-way split in a single call**:

| Sub-key | Use |
|---------|-----|
| `key` | Stored back into `new_state.key`; seeds the next step's split. |
| `respawn_key` | Samples new `(row, col)` positions for resources that just respawned (vmap over `num_res` further splits). |
| `hunt_key` | Feeds `_hunt_step` inside `update_animals`; was `predator_key` pre-v2.0. Byte-identical PRNG draw shapes preserved. |
| `wander_key` | Feeds `_wander_step` inside `update_animals`; was `neutral_key` pre-v2.0. Byte-identical PRNG draw shapes preserved. |
| `damage_key` | **Reused** for all three damage-sample calls (resource, animal, obstacle). Uniform draws across kinds are correlated within a step because they share the same seed. In practice this is harmless — only the masked entries contribute — but it is not statistically independent sampling. |
| `property_key` | Re-samples chemical signatures (`res_property_sampled`) for resources that respawn this step. |

---

## Stage 1 — Resource Regeneration (`core.py:374`)

Runs **before** the agent moves, so a resource that respawns onto the agent's cell will be tested in Stage 4 on this same step.

**`update_resources(res_active, res_reg_timer, res_cons_count, params)`** (`core.py:119`):

1. For every inactive resource with `res_reg_timer > 0`: decrement `res_reg_timer` by 1.
2. Any inactive resource whose `new_reg_timer <= 0` fires `respawn_mask = True`: mark `new_active = True`, reset `new_cons_count = 0`.
3. Back in `jax_step`: for respawning resources, vmap-sample new `(row, col)` from their `res_spawn_area` (using `num_res` splits of `respawn_key`).
4. `res_pos_after_reg` = positions updated only where `respawn_mask` is True.
5. Re-sample `res_property_sampled` for respawning resources using `property_key`; others keep their existing sample.

**Fields written (local variables; assembled into `new_state` at Stage 7)**:
`new_active`, `new_reg_timer`, `new_cons_count`, `res_pos_after_reg`, `res_property_sampled_after_reg`.

**Note**: `update_resources` only touches inactive resources (`~res_active`). A resource with `active=True` is not decremented. A resource with `active=False AND timer==0` stays dead until `respawn_mask` fires (requires `timer <= 0` AND `~active`).

---

## Stage 2 — Agent Movement (`core.py:397`)

**`move_agent(state.agent_pos, action, state.obs_pos, params.obs_blocking, params)`** (`core.py:5`)

Action mapping:

| Action | Δ(row, col) | Label |
|--------|------------|-------|
| 0 | (-1, 0) | Up |
| 1 | (0, +1) | Right |
| 2 | (+1, 0) | Down |
| 3 | (0, -1) | Left |
| 4 | (0, 0) | Rest (stay) |
| 5 | (0, 0) | Eat (stay) |

Action is clipped to `[0, 5]` before lookup. Steps:

1. Look up the move delta.
2. Compute `new_pos = pos + delta`.
3. Clamp to grid boundaries `[0, H-1] × [0, W-1]`.
4. Check whether `new_pos` overlaps any obstacle with `obs_blocking=True` (using `state.obs_pos`).
5. If collision: `final_pos = pos`; `is_collision = True`.
6. Otherwise: `final_pos = new_pos`; `is_collision = False`.

**Returns**: `new_agent_pos`, `just_collided`.

**Fields read**: `state.agent_pos`, `state.obs_pos`, `params.obs_blocking`.

**Note**: obstacle positions come from `state.obs_pos`, not `params` — obstacles are stored in `EnvState`, not `EnvParams`, so they can be randomised per episode at reset.

---

## Stage 3 — Unified Animal Update (`core.py:400`)

**`update_animals(state, new_agent_pos, params, hunt_key, wander_key)`** (`core.py:283`)

This replaces the two separate calls (`update_predators` + `update_neutral_animals`) that existed before v2.0. All animals live in the same `animal_*` arrays. The function dispatches per-behaviour using static index tuples frozen in `params`:

| Index tuple | Behaviour | Helper |
|-------------|-----------|--------|
| `params.hunt_idx` | HUNT — FSM-driven pursuit of the agent | `_hunt_step` (`core.py:132`) |
| `params.wander_idx` | WANDER — random bounded jitter | `_wander_step` (`core.py:242`) |
| `params.static_idx` | STATIC — never moves | pass-through (no scatter) |

**Algorithm** (inside `update_animals`):

1. Slice the `(N,)` arrays to `(N_hunt,)` / `(N_wander,)` subsets using the index tuples.
2. Call `_hunt_step` with `hunt_key` → returns updated hunt subset positions, states, staminas, move timers, attack timers.
3. Call `_wander_step` with `wander_key` → returns updated wander subset positions, move timers.
4. Scatter results back into the full `(N,)` arrays at `h_idx` / `w_idx`.

**Fields read**: `state.animal_pos`, `state.animal_state`, `state.animal_stamina`, `state.animal_move_timer`, `state.animal_attack_timer`, `state.animal_detect_sampled`, `state.animal_max_stamina_sampled`, `state.animal_recovery_sampled`, `state.animal_hunt_thresh_sampled`, `state.animal_lose_interest_sampled`, `state.obs_pos`, `params.obs_blocking`, `params.obs_hides_agent`, `params.animal_patrol`, `params.animal_move_int`.

**Fields written (local)**: `new_animal_pos`, `new_animal_state`, `new_animal_stamina`, `new_animal_mt`, `new_animal_at`.

`_hunt_step` FSM states: `0 = PATROL` (wander in zone), `1 = HUNT` (chase agent), `2 = RETURN` (retreat to zone center). See [07_predator_ai.md](07_predator_ai.md) for the full state-transition logic.

**PRNG inside `_hunt_step`**: 4 sub-keys split from `hunt_key` — two for row/column jitter (when `state==PATROL`), one for diagonal tie-breaking, one internal resplit.

**PRNG inside `_wander_step`**: 2 sub-keys split from `wander_key` — one for row jitter, one for column jitter.

---

## Stage 4 — Interaction & Damage (`core.py:407`)

Determines what the agent collides with at `new_agent_pos` and accumulates all damage for this step.

### 4a. Resource Overlap

```python
at_resource      = jnp.all(res_pos_after_reg == new_agent_pos, axis=-1)   # [num_res] bool
interact_resource = jnp.logical_and(at_resource, new_active)
```

Uses `res_pos_after_reg` (post-respawn positions from Stage 1) and `new_active` (post-regeneration active flags).

**Hiding-predator (danger) resources** (`res_type == 1`):
- Damage sampled per resource: `Uniform(res_damage[i,0], res_damage[i,1])` (using `damage_key`).
- `damage_res = sum(sampled_res_damage where interact_resource AND is_hiding_predator)`.

**Food resources** (`res_type == 0`):
- Auto-eat (`eat_action_enabled=False`): triggered on any overlap with an active food resource.
- Eat-action (`eat_action_enabled=True`): triggered only if `action == eat_action_idx` where `eat_action_idx = 5 if rest_action_enabled else 4`.
- `ate_food = any(ate_food_auto OR (interact_resource AND is_food AND eat_action_triggered))`.

**Resource lifecycle** — `interacted_this_step` (local variable, NOT in info dict):
```python
interacted_this_step = hiding_predator_interacted OR eat_lifecycle_triggered
```
Tracks whether a resource was "touched" for consumption counting. Both auto-eat and eat-action interactions count. Used to increment `next_cons_count`; never exposed in the `info` dict.

```python
next_cons_count  = new_cons_count + jnp.where(interacted_this_step, 1, 0)
should_deactivate = new_active AND (res_max_cons > 0) AND (next_cons_count >= res_max_cons)
final_active     = where(should_deactivate, False, new_active)
next_reg_timer   = where(should_deactivate, res_reg_delay, new_reg_timer)
```

### 4b. Animal (Predator) Damage

Uses **post-step** animal positions (`new_animal_pos`) for damage and the `animal_is_damaging` flag to restrict damage to predator-class animals:

```python
at_animal   = jnp.all(new_animal_pos == new_agent_pos, axis=-1)    # [N] bool
at_damaging  = jnp.logical_and(at_animal, params.animal_is_damaging)
sampled_pred_damage = jax.random.uniform(damage_key, (N,), ...)    # all N animals, same damage_key
damage_pred = jnp.sum(jnp.where(at_damaging, sampled_pred_damage, 0.0))
```

Attack delay is set for any damaging animal that hit this step:
```python
new_animal_at = jnp.where(at_damaging, params.animal_attack_delay, new_animal_at)
```

**`hit_neutral`** uses **pre-step** positions (`state.animal_pos`) for non-damaging animals (B5 parity fix — preserves the asymmetry of the pre-v2.0 code where `hit_predator` used post-move and `hit_neutral` used pre-move):
```python
at_neutral_pre = jnp.logical_and(
    jnp.all(state.animal_pos == new_agent_pos, axis=-1),
    ~params.animal_is_damaging
)
```

### 4c. Obstacle Damage

Two modes, both using the same `damage_key` and the same `sampled_obs_damage` array:

1. **Overlap damage** (agent ends up on a non-blocking obstacle):
   - `at_obs = jnp.all(state.obs_pos == new_agent_pos, axis=-1)`
   - `damage_obs_overlap = sum(sampled_obs_damage where at_obs AND NOT obs_blocking)`

2. **Collision damage** (agent tried to walk into a blocking obstacle):
   - `attempted_pos = state.agent_pos + move_delta` (clamped to grid)
   - `at_attempted_obs = jnp.all(state.obs_pos == attempted_pos, axis=-1)`
   - `damage_obs_collision = where(just_collided, max(sampled_obs_damage where at_attempted_obs), 0.0)`
   - `collision_noc = where(just_collided, max(obs_nociception where at_attempted_obs), 0.0)` — stored to `state.last_collision_noc`.

Note: obstacle position checks use `state.obs_pos` (original, pre-step state) — obstacle positions do not change during a step.

### 4d. Damage Sum

```python
total_damage = damage_res + damage_pred + damage_obs_overlap + damage_obs_collision
```

### 4e. Early Info Dict Construction

At this point the following keys are populated:

```python
info = {
    'ate_food':               bool,
    'damage':                 float,
    'damage_hiding_predator': float,   # == damage_res
    'damage_predator':        float,   # == damage_pred
    'damage_obstacle':        float,   # == damage_obs_overlap + damage_obs_collision
    'rested':                 bool,    # action==4 AND rest_action_enabled
    'hit_hiding_predator':    bool,
    'hit_predator':           bool,
    'hit_neutral':            bool,
}
```

`rested = jnp.logical_and(params.rest_action_enabled, action == 4)` — computed here and placed in info before `update_body` is called.

---

## Stage 5 — Body / Homeostasis Update (`core.py:501`)

**`update_body(state, info, params)`** (`core.py:44`)

Reads `info['ate_food']`, `info['damage']`, `info['rested']` from the dict assembled in Stage 4.

**Nutrition** (linear decay):
```
new_nutrition = prev_nutrition - metabolic_cost
if ate_food: new_nutrition += (food_nutrition_gain - eating_nutrition_cost)
new_nutrition = clip(new_nutrition, 0, max_nutrition)
```
Guarded by `params.with_nutrition`; if disabled, `new_nutrition = prev_nutrition`.

**Satiation** (non-linear, derived from nutrition):
```
fullness_ratio = clip(new_nutrition / max_nutrition, 0, 1)
new_satiation  = max_satiation * fullness_ratio^k     (k = nutrition_to_satiation_scaling_factor)
```
Guarded by `params.with_satiation`; if disabled, `new_satiation = state.satiation`.

**Injury** (smoothing buffer — instant-start spreading):
1. Spread `damage` evenly across the `smoothing_duration`-length `injury_buffer`: `inc = damage / smoothing_duration`.
2. Add `inc` to every slot: `temp_buffer = injury_buffer + inc`.
3. Apply `temp_buffer[0]` immediately: `new_injury = prev_injury + temp_buffer[0]`.
4. Roll buffer left by 1 and zero the last slot (remaining `smoothing_duration-1` slices apply in future steps).
5. **Recovery**: if `rested AND applied_inc <= 0` (not absorbing net damage this step), subtract `recovery_amount`.
   - `recovery_amount = recovery_base_rate * (1 + recovery_accel_rate)^(rest_streak - 1)`
   - `new_rest_streak = rest_streak + 1` if rested, else 0.
6. `new_injury = clip(new_injury, 0, max_injury)`.

Guarded by `params.with_injury`; if disabled, injury is constant and `new_buffer = injury_buffer`.

**Nociception history** (always updated, regardless of `with_injury`):
```python
new_nociception_history = jnp.roll(nociception_history_buffer, 1).at[0].set(new_injury)
```
Slot 0 = most recent injury value. Used by the interoceptive sensor to convolve a perceived pain signal over time (see doc `06`).

**Termination from body**:
- `with_nutrition`: done if `new_nutrition <= 0.0`.
- `with_injury`: done if `new_injury >= max_injury`.
- If `with_injury=False` (legacy mode): done if `damage > 0`.

**Returns**: `(new_satiation, new_nutrition, new_injury, next_injury_buffer, next_nociception_history, new_rest_streak, done_from_body)`.

**Fields read (from state)**: `state.satiation`, `state.nutrition`, `state.injury_level`, `state.injury_buffer`, `state.nociception_history_buffer`, `state.rest_streak`.

---

## Termination Check (`core.py:535`)

```python
next_step  = state.current_step + 1
truncated  = next_step >= params.max_steps
reason = 0
reason = where(truncated,                         1, reason)
reason = where(new_nutrition <= 0.0,              2, reason)
reason = where(new_satiation >= max_satiation,    3, reason)   # only if overeating_death=True
reason = where(new_injury >= max_injury,          4, reason)
done = done_from_body OR truncated
```

Termination reason codes:

| Code | Name | Condition |
|------|------|-----------|
| 0 | Active | No termination condition met |
| 1 | Truncated | `next_step >= max_steps` |
| 2 | Starvation | `new_nutrition <= 0.0` |
| 3 | Overeating | `new_satiation >= max_satiation` AND `overeating_death=True` |
| 4 | Injury | `new_injury >= max_injury` |

`info['termination_reason']` is set here.

**Note on ordering**: reason assignments use `jnp.where` in sequence, so later conditions overwrite earlier ones. If both truncation and injury fire on the same step, `reason=4` (injury) is returned because it is assigned last.

**Note on `current_step`**: `next_step = state.current_step + 1` then `truncated = next_step >= max_steps`. Since `current_step` starts at 0, the episode runs for exactly `max_steps` transitions before truncation fires.

---

## Stage 6 — Reward Computation (`core.py:550`)

Two parallel modes controlled by `params.use_homeostatic_reward`:

**Homeostatic mode** (`use_homeostatic_reward=True`):
```
prev_drive = ‖(state.satiation − setpoint, state.injury_level)‖₂   ← previous step values
curr_drive = ‖(new_satiation − setpoint, new_injury)‖₂              ← new step values
reward_homeostatic = prev_drive − curr_drive   (positive if drive decreased)
if done: reward_homeostatic -= death_penalty
```

**Survival mode** (`use_homeostatic_reward=False`):
```
reward_extrinsic = +1.0 if ate_food else 0.0
if done: reward_extrinsic -= death_penalty
```

**Eating penalty** (applied in both modes):
```
reward -= eating_reward_penalty  if ate_food
```

**Final scalar**: `reward = reward_homeostatic + reward_extrinsic − (eating_reward_penalty if ate_food)`

**`calculate_drive(satiation, injury, params)`** (`core.py:38`):
```python
target  = [params.setpoint, 0.0]
current = [satiation, injury]
return jnp.linalg.norm(current - target)
```

Remaining info keys added in this stage: `reward_homeostatic`, `reward_extrinsic`, `drive_hunger`, `drive_injury`, `metabolic_drain`, `event_collided`, distance metrics, `agent_in_bush`.

---

## Stage 7 — Final State Assembly (`core.py:626`)

`new_state = state._replace(...)` with every computed value. Fields **explicitly passed through unchanged** (sampled at reset; do not mutate during a step):

- `animal_property_sampled`
- `animal_detect_sampled`, `animal_max_stamina_sampled`, `animal_recovery_sampled`
- `animal_hunt_thresh_sampled`, `animal_lose_interest_sampled`

`new_state.key` receives the first element of the 6-way split (the "leftover" key), **not** any of the named sub-keys. Sub-keys are consumed and discarded.

All written fields:

```
agent_pos, current_step,
res_pos, res_active, res_reg_timer, res_cons_count, res_property_sampled,
animal_pos, animal_state, animal_stamina, animal_move_timer, animal_attack_timer,
satiation, nutrition, injury_level, injury_buffer, nociception_history_buffer,
last_collision_noc, rest_streak, terminated, key, last_action
```

---

## Info Dict — Complete Key Reference

All keys returned by `jax_step` in the `info` dict, in construction order.

### Core event flags (Stage 4)

| Key | Type/Shape | Description |
|-----|-----------|-------------|
| `ate_food` | bool scalar | Agent consumed a food resource this step (either auto-eat or eat-action). |
| `damage` | float scalar | Total damage from all sources this step. |
| `damage_hiding_predator` | float scalar | Damage from hiding-predator resources (`res_type==1`). |
| `damage_predator` | float scalar | Damage from `animal_is_damaging` animals at the agent's post-step cell. |
| `damage_obstacle` | float scalar | Damage from obstacles (non-blocking overlap + blocking collision combined). |
| `rested` | bool scalar | Agent chose action 4 (Rest) and `rest_action_enabled=True`. |
| `hit_hiding_predator` | bool scalar | Agent overlapped at least one active danger resource this step. |
| `hit_predator` | bool scalar | Agent overlapped at least one `animal_is_damaging` animal (post-step positions used). |
| `hit_neutral` | bool scalar | Agent shared a cell with at least one non-damaging animal (pre-step positions used — asymmetric by design, B5 fix). |

### Termination (Stage 5 / termination block)

| Key | Type/Shape | Description |
|-----|-----------|-------------|
| `termination_reason` | int32 scalar | Code 0–4 (see table in "Termination Check" section). |

### Reward and drive (Stage 6)

| Key | Type/Shape | Description |
|-----|-----------|-------------|
| `reward_homeostatic` | float scalar | Drive-reduction reward component (`prev_drive − curr_drive`; 0 in survival mode). |
| `reward_extrinsic` | float scalar | Survival reward component (`+1 if ate_food`; 0 in homeostatic mode). |
| `drive_hunger` | float scalar | `(1 − satiation/max_satiation)²` — hunger component of the drive signal. |
| `drive_injury` | float scalar | `(injury/max_injury)²` — injury component of the drive signal. |
| `metabolic_drain` | float scalar | `params.metabolic_cost` (constant per step — nutrition cost of being alive). |
| `event_collided` | bool scalar | Agent attempted to move into a blocking obstacle this step. |

### Distance telemetry (Stage 6)

| Key | Type/Shape | Description |
|-----|-----------|-------------|
| `dist_to_food` | float scalar | Euclidean distance from `new_agent_pos` to nearest active food resource; 99.0 if none. Uses `state.res_pos` (pre-respawn). |
| `dist_to_pred` | float scalar | Euclidean distance to nearest `animal_is_damaging` animal; 99.0 if none. Uses `state.animal_pos` (pre-step). |
| `dist_to_neutral` | float scalar | Euclidean distance to nearest non-damaging animal; 99.0 if none. Uses `state.animal_pos` (pre-step). |
| `dist_to_hiding_predator` | float scalar | Euclidean distance to nearest active hiding-predator resource; 99.0 if none. |
| `dist_per_predator` | float `[N_pred]` | Per-predator-class distances from `new_agent_pos`. Shape `(0,)` if no predators. |
| `dist_per_neutral` | float `[N_neutral]` | Per-neutral-class distances from `new_agent_pos`. Shape `(0,)` if no neutrals. |
| `dist_per_animal` | float `[N]` | Per-animal distances for all N animals (all classes). |

### Bush occupancy (Stage 6)

| Key | Type/Shape | Description |
|-----|-----------|-------------|
| `agent_in_bush` | bool scalar | True iff `new_agent_pos` overlaps an obstacle with `obs_hides_agent=True`. Mirrors the `agent_hidden` computation in `_hunt_step` — when True, hunting animals cannot detect or cannot maintain interest in the agent. |

---

## Clarifications / FAQ

**Q: What's the effective action index for Rest and Eat in each config?**

Action indices shift based on which actions are enabled:

| `rest_enabled` | `eat_enabled` | Action 0–3 | Action 4 | Action 5 |
|----------------|---------------|-----------|---------|---------|
| False | False | Up/R/D/L | (n/a, `action_dim=4`) | (n/a) |
| True | False | Up/R/D/L | Rest | (n/a) |
| False | True | Up/R/D/L | Eat | (n/a) |
| True | True | Up/R/D/L | Rest | Eat |

`eat_action_idx = 5 if rest_action_enabled else 4` (`core.py:433`). `rested = rest_action_enabled AND action == 4` (`core.py:503`).

**Q: Are damage samples correlated across sources?**

Yes. A single `damage_key` (`core.py:372`) is passed to all three `jax.random.uniform` calls — one for resources (`core.py:424`), one for animals (`core.py:474`), one for obstacles (`core.py:486`). All three draw uniforms from the same PRNG seed, so the values are correlated across sources within a single step. In practice only masked entries contribute to the totals, so this has no observable effect on game dynamics, but the draws are not statistically independent.

**Q: Why does `hit_predator` use post-step positions but `hit_neutral` uses pre-step positions?**

B5 fix: pre-v2.0 code had `hit_predator` check against `new_pred_pos` (post-move) and `hit_neutral` check against `state.neutral_pos` (pre-move). The unified v2.0 refactor preserves this asymmetry exactly to keep training runs bit-for-bit identical. It is a known quirk, not a bug.

**Q: Does `interacted_this_step` appear in the info dict?**

No. `interacted_this_step` (`core.py:452`) is a local variable used only to increment `next_cons_count`. It is never added to the `info` dict. If you need this signal in a training callback, derive it from `hit_hiding_predator OR ate_food`.

**Q: What is `agent_in_bush` and how does it interact with predator detection?**

When the agent steps onto an obstacle with `obs_hides_agent=True` (a "bush"), `agent_in_bush` becomes True. Inside `_hunt_step`, hunting animals check `agent_hidden` using the same `obs_hides_agent` mask (`core.py:162`). A hidden agent cannot trigger `become_hunt` and can trigger `lose_interest` even if inside detection range. `agent_in_bush` in the info dict is a readout of this state for analysis; it does not feed back into any sensor.

**Q: What order are the termination reason codes applied?**

The four `jnp.where` calls execute in sequence: truncated=1, starvation=2, overeating=3, injury=4. Each overwrites the previous. If both truncation and injury fire on the same step, `reason=4` (injury) is returned.

**Q: Is Stage 1 (resource regeneration) affected by the agent's current position?**

Yes, implicitly. If a resource respawns onto the agent's current cell, Stage 4 will detect the overlap on the same step. `update_resources` does not check occupancy — a resource can respawn on the agent, on another resource, or on an obstacle. Use `max_consumption: -1` (unlimited) with a meaningful `regeneration_delay` to avoid edge cases.

**Q: What happens if the agent runs into a blocking obstacle?**

Three effects:
1. Position stays at `state.agent_pos` — `just_collided = True`.
2. Damage from the specific obstacle at `attempted_pos` is applied via `damage_obs_collision`.
3. `last_collision_noc = max(obs_nociception where at_attempted_obs)` — a nociception spike stored in `state.last_collision_noc`, cleared on the next step.

**Q: What if the agent moves onto a non-blocking obstacle (e.g. a bush)?**

Movement proceeds normally (`just_collided=False`). If the obstacle has `obs_damage > 0`, overlap damage (`damage_obs_overlap`) is applied. Bushes typically have `damage: 0.0`, so no injury is incurred. `agent_in_bush` will be True in the info dict.

**Q: Which state values are "before" vs "after" when homeostatic reward is computed?**

`calculate_drive(state.satiation, state.injury_level, params)` uses the **previous** step's values (the `state` input, unchanged). `calculate_drive(new_satiation, new_injury, params)` uses values from Stage 5. Drive reduction → positive reward.

**Q: Does `done=True` skip reward computation?**

No. Reward is computed regardless of `done`, then `−death_penalty` is added when `done=True`. The last-step reward is visible to the agent.

**Q: Do distance keys use pre-step or post-step positions?**

Mixed. Animal distances use `state.animal_pos` (pre-step — before `update_animals` moves them), but compare against `new_agent_pos` (post-move). Resource distances use `state.res_active` and `state.res_pos` (pre-respawn). These are telemetry only and are not fed to any sensor.

**Q: Where do per-episode sampled behavioural parameters (detection range, max stamina, etc.) come from?**

They are sampled at reset into `state.animal_detect_sampled`, `state.animal_max_stamina_sampled`, `state.animal_recovery_sampled`, `state.animal_hunt_thresh_sampled`, `state.animal_lose_interest_sampled` from `Uniform(low, high)` bounds in `EnvParams`. During a step these fields are **passed through unchanged** — read by `_hunt_step` but never written by `jax_step`. They are listed explicitly in `state._replace(...)` at Stage 7 for JAX tracing clarity.

**Q: What's in `new_state.key` after a step?**

The first element of the 6-way split (`key` from `core.py:372`). The five named sub-keys are consumed and discarded. The stored `key` seeds the next step's split.

**Q: What is `animal_is_damaging` and how is it set?**

`params.animal_is_damaging` (`state.py:123`) is a `[N] bool` array precomputed at config-load time from animal class: `True` for `'predator'`-class animals, `False` for `'neutral'`-class animals. It gates damage application (`at_damaging`) and splits `dist_to_pred` / `dist_to_neutral` computations.

**Q: `update_resources` fires for ALL inactive resources every step — doesn't that re-trigger already-pending respawns?**

No. Only decrements where `~res_active AND reg_timer > 0`. A resource with `active=False AND timer==0` stays at timer=0 and will fire `respawn_mask` again on the next step (re-activating it). If you want permanent deactivation, use `res_max_cons > 0` with a large `res_reg_delay`.

**Q: Does the `predator_enabled` flag exist in v2.0?**

No. The `predator_enabled` flag has been removed entirely. Whether predators exist is determined by config — if `hunt_idx` is empty (no animals configured with `behaviour: hunt`), `_hunt_step` is never called.
