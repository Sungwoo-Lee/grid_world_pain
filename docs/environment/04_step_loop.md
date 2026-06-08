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
