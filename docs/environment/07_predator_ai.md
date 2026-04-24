# 07 — Predator AI

> **Source**: `src/environment/core.py` (`update_predators`, `core.py:128`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

Predators are hostile mobile agents that pursue and damage the player. They implement a 3-state finite state machine (FSM) with stamina-limited pursuit, configurable detection, and an attack cooldown. Multiple predators are updated in a single vectorised call — all state transitions and movements are computed simultaneously using `jnp.where` over the predator batch.

The entire predator system is gated by `params.predator_enabled` (a static flag). When False, the predator arrays are empty (shape `[0, ...]`) and predator update is a no-op.

> **Note on "Hiding Predators"**: This document describes *active* mobile predators. The environment also supports "hiding predators", which are currently implemented as a stationary, damage-dealing resource type. See [08 — Resources & Obstacles](08_resources_and_obstacles.md) for details on hiding predators.

---

## State Machine

Each predator has an integer state stored in `pred_state`:

| State | Value | Description |
|-------|-------|-------------|
| Patrol | 0 | Default — random jitter movement within `pred_patrol` bounding box |
| Hunt | 1 | Active pursuit of the agent; stamina drains each step |
| Return | 2 | Moving back toward the patrol area center after stamina depletes, the agent escapes, or the agent hides |

> **Initial state** (`core.py:758-761`): on reset, every predator starts in **Patrol (0)** with `pred_stamina = pred_max_stamina`, `pred_move_timer = 0`, `pred_attack_timer = 0`. The spawn position is sampled uniformly from the predator's `pred_spawn_area` (which may differ from `pred_patrol`).

### Transition Diagram

```
     become_hunt (and not Hunt)
  ┌──────────────────────────────┐
  │                              ▼
Patrol (0) ◄──reentered_home── Return (2) ──become_hunt──► Hunt (1)
                                  ▲                         │
                                  └─────── lose_interest ───┘
```

All three states can transition to **Hunt** via `become_hunt` (line 167: `pred_state != 1`). In particular, **Return → Hunt is possible** — a returning predator that recovers enough stamina while still within detection range of the agent will re-enter Hunt without first reaching Patrol.

### Transition Logic (`core.py:152`)

All transitions are evaluated **every step** regardless of `move_timer`. Only movement is gated by `move_timer <= 0` (see [Manhattan Pursuit](#manhattan-pursuit-corepy192)).

**Patrol → Hunt** *(also Return → Hunt)*:
```
dist = sum(abs(pred_pos - agent_pos), axis=-1)   # Manhattan distance (core.py:134-135)
become_hunt = (dist <= pred_detect)
           AND (pred_stamina >= max_stamina * hunt_thresh)
           AND NOT agent_hidden
next_state = 1  if (pred_state != 1) AND become_hunt   # (core.py:167)
```
The predator will only initiate a hunt if it has enough stamina and the agent is not concealed. `pred_hunt_thresh` (default 0.7) prevents a just-returned, exhausted predator from immediately starting a new chase. Because the guard is `pred_state != 1` (not `pred_state == 0`), a predator currently in Return can jump directly back to Hunt.

**Hunt → Return**:
```
lose_interest = (dist > pred_detect * pred_lose_interest_mult)
             OR (pred_stamina <= 0)
             OR agent_hidden
next_state = 2  if (pred_state == 1) AND lose_interest   # (core.py:169)
```
The predator gives up if the agent moves far enough away (detection range × multiplier), runs out of stamina, or hides in a bush obstacle.

**Return → Patrol**:
```
tr_return, tc_return = (min_r + max_r) // 2, (min_c + max_c) // 2   # patrol centroid (core.py:172-173)
dist_to_center = sum(abs(pred_pos - [tr_return, tc_return]), axis=-1)
reentered_home = (next_state == 2) AND (dist_to_center <= 2)   # (core.py:176-178)
```
The predator switches back to patrol once it is within 2 Manhattan steps of its patrol area's integer-division centroid. The threshold `2` is hardcoded.

---

## Return State — Full Definition

The Return state (value `2`) handles what the predator does after it gives up on a hunt. It is deliberately separated from Patrol so the predator has a "cool-down walk" back to its territory rather than immediately resuming random jitter far from home.

**Entry conditions** (from `Hunt → Return`, `core.py:169`): the predator enters Return whenever it is in Hunt and `lose_interest` fires — triggered by:
- The agent moving farther than `pred_detect × pred_lose_interest_mult` Manhattan steps away.
- Stamina reaching `0`.
- The agent stepping onto a bush (`agent_hidden`).

**Target**: the **integer-division centroid** of the predator's patrol box (`core.py:172-173`):
```
tr_return = (pred_patrol[p, 0] + pred_patrol[p, 2]) // 2     # midpoint row
tc_return = (pred_patrol[p, 1] + pred_patrol[p, 3]) // 2     # midpoint col
```
Because `//` is floor division, for an even-sized patrol box (e.g. rows `1..10`) the centroid is biased toward the smaller index (row `5`, not `5.5`).

**Behaviour during Return**:
- **Movement direction** (`core.py:201-202`): `dr = tr_return - pred_pos[row]`, `dc = tc_return - pred_pos[col]`, then resolved to a single-axis step via the diagonal tie-breaker (see [Manhattan Pursuit](#manhattan-pursuit-corepy192)).
- **Stamina recovery** (`core.py:244`): since `next_state != 1`, stamina regenerates at `pred_recovery` per step while returning — the longer the Return trip, the more stamina is available if the agent is re-sighted.
- **Spatial bounds**: same clamping as other states — new position is clipped to `pred_patrol[p]` and to the global grid bounds. If the predator starts outside its patrol box (e.g. spawned in a wider `spawn_area`), the first Return movements pull it toward the box.
- **Obstacle collision**: same rule — if the next cell is a blocking obstacle, the predator stays put.

**Exit conditions**:
- **Return → Patrol** (`core.py:177-178`): the predator reaches within `dist_to_center <= 2` Manhattan steps of the centroid.
- **Return → Hunt** (`core.py:167`): if the agent comes back into `pred_detect` range, the predator has `pred_stamina >= pred_max_stamina * pred_hunt_thresh`, and the agent is not hidden, Return is aborted and a fresh Hunt begins.

**Why the `hunt_thresh` guard matters**: without it, a predator that just lost interest (stamina = 0) would instantly re-enter Hunt on step N+1 after recovering 1 point of stamina. The fraction (default 0.7) forces the predator to walk most of the way home before it can chase again, giving the agent a real escape window.

---

## Stamina & Lose-Interest

`pred_stamina` (`core.py:244`):
```
if state == Hunt:
    new_stamina = prev_stamina - 1.0        (drained per step)
else:
    new_stamina = prev_stamina + pred_recovery
new_stamina = clip(new_stamina, 0, max_stamina)
```

- Stamina drains at rate 1.0 per hunt step (hardcoded).
- Stamina recovers at rate `pred_recovery` per non-hunt step (patrol or return).
- `pred_lose_interest_mult` (default 2.0): the effective "give up" distance is `pred_detect × 2.0`. A multiplier of 1.0 means the predator gives up as soon as the agent exits detection range; 2.0 allows the predator to continue chasing up to twice the detection range.

---

## Manhattan Pursuit (`core.py:192`)

Movement is throttled by `pred_move_int`: the predator only moves when `pred_move_timer <= 0`. The timer is reset to `pred_move_int` after each move.

**Direction computation**:
- Hunt: `dr = agent_pos[0] - pred_pos[row]`, `dc = agent_pos[1] - pred_pos[col]`
- Return: `dr, dc` toward patrol area center `((min_r + max_r)//2, (min_c + max_c)//2)`
- Patrol: `dr, dc` from random jitter `(-1, 0, 1)` per axis

**Movement resolution**:
```
step_r, step_c = sign(dr), sign(dc)
if both nonzero (diagonal):
    randomly choose either row-only or col-only with p=0.5
```
This random tie-breaking prevents diagonal oscillation and produces more natural pursuit paths.

**Spatial bounds**: after computing the new position, it is hard-clamped to:
1. The predator's patrol bounding box (`pred_patrol[p, 0:4]`).
2. The global grid bounds (`[0, H-1] × [0, W-1]`).

This means predators never leave their assigned territory even during Hunt state.

**Obstacle collision** (`core.py:233`): vmapped per predator — if the new position overlaps a blocking obstacle, the predator stays at its current position.

---

## Attack Logic

> **Location note**: damage application and attack-timer assignment live in `jax_step` (`core.py:392-400`), **not** inside `update_predators`. `update_predators` only propagates the pre-existing `pred_attack_timer` and consumes it to gate movement. The timer is *written* one step later, after the predator has finished moving and the overlap check runs against the new agent position.

When a predator's position equals the agent's position after the predator update, damage is applied:

```python
# core.py:392-397
at_predator = all(new_pred_pos == new_agent_pos, axis=-1)
sampled_pred_damage = Uniform(pred_damage[:, 0], pred_damage[:, 1], shape=[P])
damage_pred = sum(sampled_pred_damage where at_predator)
```

- `pred_damage` has shape `[P, 2]` — columns are `[min, max]` of a uniform distribution sampled fresh every step. A scalar `damage: 10` in YAML is expanded to `[10, 10]` (deterministic).
- Damage from all colliding predators is summed into a single `damage_pred` value passed to the injury system.

**Attack cooldown** (`core.py:400`): `pred_attack_timer` is set to `pred_attack_delay` whenever the predator hits the agent. While this timer is nonzero, `should_move` is False (line 181: `should_move = (new_move_timer <= 0) AND (new_attack_timer <= 0)`) — the predator freezes in place for the cooldown duration. This creates a pulsed attack pattern rather than continuous damage-on-overlap.

**Nociception**: `params.pred_nociception [P]` is the intensity value forwarded to the exteroceptive nociception sensor when the predator is at the agent's position.

---

## Bush Concealment (`core.py:146`)

```python
agent_hidden = any(all(obs_pos == agent_pos, axis=-1) AND obs_hides_agent)
```

When the agent is standing on any obstacle with `hides_agent=True` (bush-type), `agent_hidden=True`. This:
1. Blocks the **Patrol → Hunt** transition: `become_hunt = become_hunt AND NOT agent_hidden`.
2. Forces the **Hunt → Return** transition: `lose_interest = lose_interest OR agent_hidden`.

In other words, bushes give the agent a "stealth" mechanic — a predator in Hunt state will immediately lose interest if the agent reaches a bush, and a patrolling predator cannot initiate a chase at all while the agent is concealed.

The bush check uses `obs_hides_agent [O]` from `EnvParams`, which is set per-obstacle in the YAML config (`hides_agent: true`).

---

## vmap Parallelism

`update_predators` operates on full predator arrays and uses `jnp.where` over the predator batch axis for all conditional logic. One internal `jax.vmap` is used at `core.py:239` for the per-predator obstacle collision check (`check_collision`).

The outer `ParallelEnv` vmaps `jax_step` (and thus `update_predators`) over the environment batch axis. This means the predator arrays in a batched call have shape `[num_envs, num_pred, ...]`.

Each predator is independent — no predator-to-predator interaction exists.

**Output shapes** (single env):
- `pred_pos`: `[P, 2]`
- `pred_state`: `[P]`
- `pred_stamina`: `[P]`
- `pred_move_timer`: `[P]`
- `pred_attack_timer`: `[P]`

> **All parameters are per-predator.** Every `pred_*` param in `EnvParams` is an array of shape `[P]` (or `[P, k]`) — each predator has its own `detection_range`, `max_stamina`, `patrol_area`, etc. There are no global predator parameters.

---

## Configuration Reference (`config_loader.py:55-99`)

Each predator is defined as an entry under `environment.predators` in the experiment YAML. Example from `configs/experiment/labmeeting/basic-04.yaml`:

```yaml
environment:
  predator_enabled: true
  predators:
    - name: "predator"
      count: 1
      property: [0.0, 1.0, 0.0, 0.0, 0.0]       # chemical signature (olfaction)
      property_std: [0.0, 0.0, 0.0, 0.0, 0.0]   # per-episode property noise
      move_interval: 1                          # pred_move_int — steps between moves
      damage: [15.0, 45.0]                      # pred_damage — [min, max] uniform
      nociception_intensity: 0.9                # pred_nociception
      spawn_area: [[1, 1], [10, 10]]            # where predator can initially spawn (1-indexed, inclusive)
      patrol_area: [[1, 1], [10, 10]]           # movement bounds (1-indexed, inclusive)
      detection_range: 5                        # pred_detect (Manhattan cells)
      max_stamina: 30                           # pred_max_stamina
      stamina_recovery_rate: 1                  # pred_recovery (per non-hunt step)
      hunt_stamina_threshold: 0.7               # pred_hunt_thresh (fraction of max)
      attack_delay: 3                           # pred_attack_delay (cooldown steps)
      lose_interest_multiplier: 1.5             # pred_lose_interest_mult
```

**Index conversion**: YAML `[[1, 1], [10, 10]]` is parsed at `config_loader.py:76-77` as `[row_min-1, col_min-1, row_max, col_max] = [0, 0, 10, 10]` — the lower corner is shifted to 0-indexed, the upper corner is kept as-is (so the raw YAML reads as 1-indexed inclusive on both ends).

**`spawn_area` vs `patrol_area`**:
- `spawn_area` is used **only at reset** (`core.py:652-654`) to sample the predator's initial position.
- `patrol_area` defines the **hard movement bounds** enforced every step (`core.py:225-226`). A predator can spawn outside its `patrol_area` if the two areas differ, but it will be clipped back on its first move.

**`count: N`**: each YAML entry can spawn multiple identical predators. They share all parameters except for the randomly sampled spawn position.

**Disabling predators**: set `predator_enabled: false` OR use an empty `predators: []` list. Both cause all `pred_*` arrays to be length-0, and `update_predators` becomes a no-op on empty batches.

---

## Clarifications / FAQ

These are answers to questions that aren't obvious from the code alone.

**Q: Does the predator see the agent's old or new position?**
A: The **new** position. `jax_step` calls `move_agent` first, then passes `new_agent_pos` to `update_predators` (`core.py:315-320`). The predator reacts to the agent's position *after* the agent has moved this step.

**Q: Is detection omnidirectional?**
A: Yes. Detection is a pure Manhattan distance check — there is no line-of-sight, no facing direction, and obstacles do **not** block vision. A predator detects the agent through walls.

**Q: What happens if two predators try to occupy the same cell?**
A: Nothing prevents it. Predators have no inter-predator collision — the obstacle-collision check (`core.py:233-239`) only tests against `obs_pos` with `obs_blocking`. Two predators can stack on the same cell, and both will damage the agent if the agent is there.

**Q: What if the patrol area is a single cell?**
A: The centroid equals that cell, and the predator sits still in Patrol/Return. Jitter is clipped back to the single cell every step.

**Q: Does stamina recover during Return?**
A: Yes. Recovery applies whenever `next_state != 1`, which includes both Patrol (0) and Return (2). This is why a long Return trip can lead to a Return → Hunt re-engagement (see [Return State](#return-state--full-definition)).

**Q: Can stamina exceed `max_stamina`?**
A: No. `new_stamina = clip(new_stamina, 0.0, pred_max_stamina)` (`core.py:245`).

**Q: How does `attack_delay` interact with `move_interval`?**
A: Both must be at 0/negative for the predator to move: `should_move = (new_move_timer <= 0) AND (new_attack_timer <= 0)` (`core.py:181`). The longer of the two dominates. State transitions still happen every step regardless of either timer.

**Q: Is the hunt-threshold applied on entry only, or every step?**
A: Every step, but only for the Patrol/Return → Hunt transition (`become_hunt`). Once in Hunt, stamina can fall below the threshold — the predator exits Hunt only when stamina reaches `0` (not `max_stamina * hunt_thresh`).

**Q: What does `pred_lose_interest_mult = 1.0` produce?**
A: The predator gives up as soon as the agent exits detection range. Values above `1.0` create a hysteresis band: the predator pursues past its own detection range (up to `pred_detect × mult`) once a hunt has started, but cannot *start* a hunt beyond `pred_detect`.

**Q: Are the random diagonal tie-breaks reproducible?**
A: Yes. The PRNG key is threaded through `update_predators` and splits deterministically (`core.py:187, 213`). Given the same seed and state, the same moves occur.

**Q: What if `spawn_area` overlaps with an obstacle?**
A: Spawn positions are resolved via `all_positions` + a resolve step (`core.py:665-667`) that de-duplicates entities — but obstacle-on-predator overlap isn't guaranteed to be rejected here. Verify against the resolve logic in `config_loader.py` if this matters for your experiment.
