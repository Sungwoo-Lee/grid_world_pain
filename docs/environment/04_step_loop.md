# 04 — Step Loop

> **Source**: `src/environment/core.py` (`jax_step`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

`jax_step(state: EnvState, action: int, params: EnvParams) → (EnvState, reward, done, info)` (`core.py:285`) is the single pure-functional transition function for one environment step. It is decorated with `@jax.jit`.

The function:
- Takes the current state, a scalar integer action, and the immutable params.
- Runs 8 sequential stages — no stage mutates state; each reads from the same `state` snapshot and writes to intermediate variables.
- Assembles a new `EnvState` via `state._replace(...)` at the very end (Stage 7).
- Returns `(new_state, scalar_reward, bool_done, info_dict)`.

All random events for the step are derived from a single key split at the top (`core.py:289`):
```python
key, respawn_key, predator_key, neutral_key, damage_key, property_key = jax.random.split(state.key, 6)
```
- `respawn_key` — samples new resource positions on respawn.
- `predator_key` — diagonal tie-breaking and any other predator randomness.
- `neutral_key` — neutral animal jitter.
- `damage_key` — **reused** for all three damage samples (resource, predator, obstacle). Means damage magnitudes across kinds are drawn from correlated uniforms within a single step.
- `property_key` — re-samples chemical signatures on resource respawn (`core.py:307-312`).
- `key` — stored back into `new_state.key` for the next step.

---

## Stage 1 — Resource Regeneration (`core.py:291`)

Before the agent moves, inactive resources tick their regeneration timers.

**`update_resources(res_active, res_reg_timer, res_cons_count, params)`**:
1. Decrement `res_reg_timer` by 1 for all inactive resources.
2. Mark `new_active = True` and reset `res_cons_count = 0` for resources whose timer reached 0.
3. For resources that just respawned, sample a new `(row, col)` within their `res_spawn_area` via vmap over `respawn_key` splits.
4. Update `res_pos` only where `respawn_mask` is True.

**Fields written**: `res_active`, `res_reg_timer`, `res_cons_count`, `res_pos`.

---

## Stage 2 — Agent Movement (`core.py:308`)

**`move_agent(pos, action, obs_pos, obs_blocking, params)`**:

Action mapping:
| Action | Δ(row, col) |
|--------|------------|
| 0 | (-1, 0) — Up |
| 1 | (0, +1) — Right |
| 2 | (+1, 0) — Down |
| 3 | (0, -1) — Left |
| 4 | (0, 0) — Rest (stay) |
| 5 | (0, 0) — Eat (stay) |

Steps:
1. Look up the move delta from the action.
2. Compute `new_pos = pos + delta`.
3. Clamp to grid boundaries `[0, H-1] × [0, W-1]`.
4. Check if `new_pos` overlaps any obstacle with `obs_blocking=True`.
5. If collision: final position stays at `pos`; `just_collided=True`.
6. Otherwise: final position is `new_pos`; `just_collided=False`.

**Fields written**: `agent_pos`, `last_action`, `last_collision_noc` (set in Stage 5).

---

## Stage 3 — Predator Update (`core.py:311`)

Calls `update_predators(...)` with the **new** agent position (post-movement). See [07_predator_ai.md](07_predator_ai.md) for the full FSM description.

**Inputs consumed**: `pred_pos`, `pred_state`, `pred_stamina`, `pred_move_timer`, `pred_attack_timer`, `new_agent_pos`, `obs_pos`, `obs_blocking`.

**Fields written**: `pred_pos`, `pred_state`, `pred_stamina`, `pred_move_timer`, `pred_attack_timer`.

---

## Stage 3.5 — Neutral Animal Update (`core.py:317`)

Calls `update_neutral_animals(neutral_pos, neutral_move_timer, obs_pos, obs_blocking, params, neutral_key)`.

Each neutral animal moves randomly (jitter ±1 in row and col), bounded by its patrol area, with obstacle collision checking via `jax.vmap`. See [08_resources_and_obstacles.md](08_resources_and_obstacles.md) for details.

**Fields written**: `neutral_pos`, `neutral_move_timer`.

---

## Stage 4 — Interaction (`core.py:322`)

Determines what the agent collides with at `new_agent_pos` and computes all damage for this step.

### Resource Interaction

`at_resource = res_pos == new_agent_pos AND res_active`

**Danger resources** (auto-trigger on overlap):
- Damage sampled from `Uniform(res_damage[i,0], res_damage[i,1])` per resource.
- Total `damage_res = sum(sampled_damage where danger AND at_resource)`.

**Food resources** (two consumption modes):
- **Auto-eat** (`eat_action_enabled=False`): triggered automatically on overlap.
- **Eat-action** (`eat_action_enabled=True`): only triggered if `action == eat_action_idx` (4 if rest disabled, else 5).
- `ate_food` flag: True if food was consumed this step.

**Resource lifecycle update**:
- `res_cons_count` incremented for any interaction (food or danger).
- Resource deactivated when `res_cons_count >= res_max_cons` (only if `res_max_cons > 0`).
- `res_reg_timer` set to `res_reg_delay` on deactivation.

### Predator Damage (`core.py:385`)

`at_predator = pred_pos == new_agent_pos` (checked after predator move).
- Damage sampled from `Uniform(pred_damage[p,0], pred_damage[p,1])` per predator.
- `new_pred_attack_timer` set to `pred_attack_delay` for predators that hit the agent.

### Obstacle Damage (`core.py:396`)

Two cases:
1. **Overlap** (non-blocking obstacle at `new_agent_pos`): damage sampled and summed.
2. **Collision** (blocking obstacle at `attempted_pos`): only the hit obstacle's damage is applied; `collision_noc` intensity is stored to `last_collision_noc`.

`total_damage = damage_res + damage_pred + damage_obs_overlap + damage_obs_collision`

---

## Stage 5 — Body Update (`core.py:431`)

Calls `update_body(state, info, params)`. See [05_body_homeostasis.md](05_body_homeostasis.md) for full detail.

**`info` dict passed in**:
- `ate_food`: bool
- `damage`: total damage this step
- `rested`: bool — True iff `rest_action_enabled AND action == 4`

**Returns**: `(new_satiation, new_nutrition, new_injury, next_injury_buffer, new_rest_streak, done_from_body)`

---

## Stage 6 — Termination Check (`core.py:434`)

`truncated = next_step >= params.max_steps`

Termination reason codes:
| Code | Name | Condition |
|------|------|-----------|
| 0 | Active | No termination condition met |
| 1 | Truncated | `current_step >= max_steps` |
| 2 | Starvation | `nutrition <= 0` |
| 3 | Overeating | `satiation >= max_satiation` and `overeating_death=True` |
| 4 | Injury | `injury_level >= max_injury` |

`done = done_from_body OR truncated`

---

## Stage 7 — Reward Computation (`core.py:450`)

Two parallel reward modes, controlled by `params.use_homeostatic_reward`:

**Homeostatic mode** (`use_homeostatic_reward=True`):
```
prev_drive = ‖(satiation_prev − setpoint, injury_prev)‖₂
curr_drive = ‖(satiation_new − setpoint, injury_new)‖₂
reward_homeostatic = prev_drive − curr_drive   (positive if drive decreased)
if done: reward_homeostatic -= death_penalty
```

**Survival mode** (`use_homeostatic_reward=False`):
```
reward_extrinsic = +1.0 if ate_food else 0.0
if done: reward_extrinsic -= death_penalty
```

**Eating penalty** (both modes):
```
reward -= eating_reward_penalty  if ate_food
```

Final scalar: `reward = reward_homeostatic + reward_extrinsic`

---

## Stage 8 — Final State Assembly (`core.py:486`)

All computed values are assembled into a new `EnvState` via `state._replace(...)`. The original `state` is unchanged. The returned `new_state` carries the updated `key` (the top-level split from Stage 0, not the sub-keys used for random events).

---

## Info Dict

Complete set of keys returned by `jax_step` in the `info` dict:

| Key | Type/Shape | Description |
|-----|-----------|-------------|
| `ate_food` | bool | Agent consumed a food resource this step |
| `damage` | float | Total damage received (all sources) |
| `damage_danger` | float | Damage from danger resources only |
| `damage_predator` | float | Damage from predators only |
| `damage_obstacle` | float | Damage from obstacles (overlap + collision) |
| `rested` | bool | Agent chose the rest action |
| `hit_danger` | bool | Agent overlapped at least one danger resource |
| `hit_predator` | bool | Agent overlapped at least one predator |
| `termination_reason` | int32 scalar | Code 0–4 (see table above) |
| `reward_homeostatic` | float | Homeostatic component of reward |
| `reward_extrinsic` | float | Survival component of reward |
| `drive_hunger` | float | `(1 - satiation/max_satiation)²` |
| `drive_injury` | float | `(injury/max_injury)²` |
| `metabolic_drain` | float | `params.metabolic_cost` (constant) |
| `event_collided` | bool | Agent tried to move into a blocking obstacle |
| `dist_to_food` | float | Euclidean distance to nearest active food |
| `dist_to_pred` | float | Euclidean distance to nearest predator |

---

## Clarifications / FAQ

**Q: What's the effective action index for Rest and Eat in each config?**
A: Action indices slide based on which actions are enabled:

| `rest_enabled` | `eat_enabled` | Action 0–3 | Action 4 | Action 5 |
|----------------|---------------|-----------|---------|---------|
| False | False | Up/R/D/L | (n/a, `action_dim=4`) | (n/a) |
| True | False | Up/R/D/L | Rest | (n/a) |
| False | True | Up/R/D/L | Eat | (n/a) |
| True | True | Up/R/D/L | Rest | Eat |

`eat_action_idx` is computed at `core.py:354` as `5 if rest_enabled else 4`. `rested = rest_enabled AND action == 4` at `core.py:423` — so with both disabled, `rested` is always False.

**Q: Does `rested` imply the agent didn't move?**
A: Yes — action 4 has delta `(0, 0)` and is the *only* way to set `rested=True`. Action 5 (Eat) also has `(0, 0)` delta but is not counted as rest.

**Q: Can the agent eat and move in the same step?**
A: No. Eating requires the agent to be standing on a food cell at the end of movement. Since Eat (action 5) has `(0, 0)` delta, the agent must have arrived on food on a prior step or spawn. In auto-eat mode (`eat_action_enabled=False`), any movement action that lands on food triggers consumption.

**Q: Are damage samples correlated across kinds?**
A: Slightly, yes. A single `damage_key` is reused for the uniform draws against each damage array (`core.py:345, 394, 406`). Different arrays share the same PRNG seed, so the sample values are correlated across resource/predator/obstacle indices. In practice this doesn't matter because `jnp.where(at_*, ...)` masks out non-colliding entities — but it does mean the three damage distributions aren't statistically independent.

**Q: Is Stage 1 (resource regeneration) affected by the agent's current position?**
A: Yes, implicitly — if a resource respawns onto the agent's current cell, the interaction check at Stage 4 fires immediately on the same step. Respawn uses `res_spawn_area` with no occupancy check, so a resource can land on the agent, another resource, or an obstacle. Use `max_consumption: -1` (unlimited) with a reasonable `regeneration_delay` to avoid edge cases.

**Q: What happens if the agent runs into a blocking obstacle?**
A: Three effects:
1. Position stays at `state.agent_pos` (no movement) — `just_collided = True`.
2. The specific obstacle hit by `attempted_pos` provides the damage sample (`core.py:413-414`).
3. `last_collision_noc` is set to that obstacle's `obs_nociception`, cleared next step.

A wall-bump therefore costs damage *and* generates a transient nociception signal for the sensor.

**Q: What if the agent tries to move into a non-blocking obstacle (e.g. a bush)?**
A: Movement proceeds. `just_collided=False`. At Stage 4, the obstacle's overlap damage (if any) is applied via `damage_obs_overlap`. Bushes typically have `damage: 0.0`. Non-blocking = "walk through", not "bounce".

**Q: What's the order of damage application vs body update?**
A: All damage is summed into `total_damage` (`core.py:419`) then passed to `update_body`, which injects it into the ring buffer (`injury_buffer`). Injury level doesn't fully reflect the hit until `smoothing_duration` steps later. See doc `05`.

**Q: Which state values are "before" vs "after" when reward is computed?**
A: `calculate_drive(state.satiation, state.injury_level, params)` uses the **previous** step's values (the `state` input). `calculate_drive(new_satiation, new_injury, params)` uses the **new** values. Drive reduction = positive reward.

**Q: Are `ate_food` and `hit_danger` mutually exclusive?**
A: No. If the agent steps on a cell that has a food resource AND a danger resource at the same cell (possible since placement doesn't prevent this across kinds), both fire. `ate_food=True` *and* `damage_danger > 0` can appear in the same info dict.

**Q: Does `done=True` from termination skip reward computation?**
A: No. Reward is computed regardless of `done`, then `-death_penalty` is added when `done=True`. The last-step reward is visible to the agent.

**Q: Do `dist_to_food` and `dist_to_pred` use pre-step or post-step positions?**
A: Mixed. They use `state.res_pos` (pre-respawn) and `state.pred_pos` (pre-predator-move), but compare against `new_agent_pos` (post-move) — `core.py:487-488`. These are telemetry only (info dict) and aren't fed to any sensor.

**Q: Is `current_step` updated before or after termination check?**
A: Before. `next_step = state.current_step + 1` then `truncated = next_step >= max_steps`. So on step `max_steps - 1` (`current_step` starts at 0), the check fires because `next_step == max_steps`. An episode runs for exactly `max_steps` steps before truncation.

**Q: What's in `new_state.key` after a step — is it the new key or an old subkey?**
A: The first element of the 6-way split (`key`, renamed from the split tuple). The five subkeys (`respawn_key`, `predator_key`, etc.) are consumed and discarded. The stored `key` will seed the next step's split.

**Q: `update_resources` at Stage 1 fires for ALL inactive resources every step. Doesn't that re-trigger already-pending respawns?**
A: No. `update_resources` (`core.py:115-126`) only decrements timers for `~res_active AND res_reg_timer > 0`. A resource with `active=False AND timer==0` stays inactive unless `respawn_mask` fires (which resets `active=True` and clears the timer). So an un-respawnable resource (`max_consumption` exhausted) can stay dead forever — check that path carefully if you rely on permanent deactivation.
