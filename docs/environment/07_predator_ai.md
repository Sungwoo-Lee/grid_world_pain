# 07 — Predator AI

> **Source**: `src/environment/core.py` (`update_predators`, `core.py:128`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

Predators are hostile mobile agents that pursue and damage the player. They implement a 3-state finite state machine (FSM) with stamina-limited pursuit, configurable detection, and an attack cooldown. Multiple predators are updated in a single vectorised call — all state transitions and movements are computed simultaneously using `jnp.where` over the predator batch.

The entire predator system is gated by `params.predator_enabled` (a static flag). When False, the predator arrays are empty (shape `[0, ...]`) and predator update is a no-op.

---

## State Machine

Each predator has an integer state stored in `pred_state`:

| State | Value | Description |
|-------|-------|-------------|
| Patrol | 0 | Default — random jitter movement within `pred_patrol` bounding box |
| Hunt | 1 | Active pursuit of the agent; stamina drains each step |
| Return | 2 | Moving back toward the patrol area center after stamina depletes |

### Transition Logic (`core.py:152`)

All transitions are evaluated every step (when `move_timer <= 0`):

**Patrol → Hunt**:
```
become_hunt = (dist_manhattan <= pred_detect)
           AND (pred_stamina >= max_stamina * hunt_thresh)
           AND NOT agent_hidden
```
The predator will only initiate a hunt if it has enough stamina and the agent is not concealed. `pred_hunt_thresh` (default 0.7) prevents a just-returned, exhausted predator from immediately starting a new chase.

**Hunt → Return**:
```
lose_interest = (dist_manhattan > pred_detect * pred_lose_interest_mult)
             OR (pred_stamina <= 0)
             OR agent_hidden
```
The predator gives up if the agent moves far enough away (detection range × multiplier), runs out of stamina, or hides in a bush obstacle.

**Return → Patrol**:
```
reentered_home = (next_state == 2) AND (dist_to_patrol_center <= 2)
```
The predator switches back to patrol once it is within 2 Manhattan steps of its patrol area center.

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

When a predator's position equals the agent's position after the predator update, damage is applied:

```python
at_predator = all(pred_pos == new_agent_pos, axis=-1)
damage_pred = sum(Uniform(pred_damage[:, 0], pred_damage[:, 1]) where at_predator)
```

**Attack cooldown** (`core.py:393`): `pred_attack_timer` is set to `pred_attack_delay` whenever the predator hits the agent. While this timer is nonzero, `should_move` is False — the predator freezes in place for the cooldown duration. This creates a pulsed attack pattern rather than continuous damage-on-overlap.

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

`update_predators` operates on full predator arrays and uses `jnp.where` over the predator batch axis for all conditional logic. There is no `jax.vmap` inside the function — it is written as pure vectorised array operations.

The outer `ParallelEnv` vmaps `jax_step` (and thus `update_predators`) over the environment batch axis. This means the predator arrays in a batched call have shape `[num_envs, num_pred, ...]`.

Each predator is independent — no predator-to-predator interaction exists.

**Output shapes** (single env):
- `pred_pos`: `[P, 2]`
- `pred_state`: `[P]`
- `pred_stamina`: `[P]`
- `pred_move_timer`: `[P]`
- `pred_attack_timer`: `[P]`
