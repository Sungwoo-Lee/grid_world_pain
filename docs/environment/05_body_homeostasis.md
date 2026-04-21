# 05 — Body & Homeostasis

> **Source**: `src/environment/core.py` (`update_body`, `calculate_drive`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

The body subsystem models the agent's internal physiological state. It provides the biological substrate for interoception — the agent must actively monitor and regulate three internal variables to survive:

- **Nutrition**: objective energy reserve, decaying each step due to metabolism.
- **Satiation**: subjective fullness derived non-linearly from nutrition.
- **Injury**: accumulated physical damage, smoothed and recoverable through rest.

Together these implement a homeostatic RL scenario: the agent receives drive-reduction reward for moving its body state toward healthy targets. This is implemented in `update_body()` (`core.py:44`), called from `jax_step` Stage 5. `calculate_drive()` (`core.py:38`) computes the Euclidean distance to the homeostatic setpoint for use in reward and analysis.

Each subsystem can be independently disabled via `with_nutrition`, `with_satiation`, and `with_injury` flags. When disabled, the corresponding field retains its value unchanged and is excluded from drive/reward/termination logic.

---

## Nutrition Dynamics

**Governing equation** (applied each step):
```
new_nutrition = clip(prev_nutrition - metabolic_cost + ate_food_gain, 0, max_nutrition)

where:
  ate_food_gain = food_nutrition_gain - eating_nutrition_cost   (if ate_food else 0)
```

- `metabolic_cost` (default 1.0): drained unconditionally every step.
- `food_nutrition_gain` (default 6): gross nutrition from consuming a food resource.
- `eating_nutrition_cost` (default 1.0): the physical cost of the eating act, subtracted from gain.
- Net gain from eating: `6 - 1 = 5` nutrition in the default config.
- Clamped to `[0, max_nutrition]` at the end.

**When `with_nutrition=False`**: `new_nutrition = prev_nutrition` — no decay, no gain, no starvation death.

**Starvation death**: triggered in the termination check when `new_nutrition <= 0` (code 2).

**Overeating death**: when `overeating_death=True`, satiation reaching `max_satiation` triggers death (code 3). In practice, satiation is bounded by the nutrition clamp, so this condition requires nutrition to exactly hit `max_nutrition`, which is prevented by the clamp.

---

## Satiation Dynamics

Satiation is **not independently tracked** — it is a deterministic function of nutrition, re-derived every step:

```
fullness_ratio = clip(new_nutrition / max_nutrition, 0.0, 1.0)
new_satiation = max_satiation * fullness_ratio ^ k
```

where `k = nutrition_to_satiation_scaling_factor` (default 1.0 = linear).

This power-law mapping allows different subjective hunger curves:
- `k = 1.0`: linear — satiation tracks nutrition directly.
- `k < 1.0`: sub-linear — agent feels relatively full even with low nutrition (optimistic subjective hunger).
- `k > 1.0`: super-linear — agent feels relatively empty unless nutrition is near-full (pessimistic subjective hunger).

Since satiation is derived from nutrition, `random_start_satiation` is effectively a no-op — the initial satiation is always computed from the initial nutrition value.

**When `with_satiation=False`**: `new_satiation = state.satiation` — value frozen.

---

## Injury Ring Buffer & Streak Recovery

**Ring buffer mechanism** (`core.py:70`): Rather than applying damage instantly, each step's damage is spread across `smoothing_duration` future steps. This prevents large single-hit deaths and creates a more biologically realistic pain experience.

```
Step-by-step:
  inc = damage / smoothing_duration              # per-step dose
  temp_buffer = injury_buffer + inc              # add to all future slots
  applied_inc = temp_buffer[0]                   # apply the front slot now
  new_injury = prev_injury + applied_inc
  new_buffer = roll(temp_buffer, -1)[:-1] + [0]  # shift left, zero the new tail
```

This means a single damage event of value `D` results in `D/K` injury added per step for the next `K` steps (where `K = smoothing_duration`). The front of the buffer is always applied immediately — there is no delay before the first increment.

**Streak-based recovery** (`core.py:84`): When the agent selects the rest action (action 4) and is not currently absorbing any buffered damage (`applied_inc <= 0`), injury recovers exponentially:

```
new_rest_streak = prev_rest_streak + 1    (if rested)  else 0
recovery_mult = (1 + recovery_accel_rate)^(max(streak, 1) - 1)
recovery_amount = recovery_base_rate * recovery_mult
```

Example with defaults (`base=0.1`, `accel=0.5`):
| Streak | Recovery per step |
|--------|-----------------|
| 1 | 0.1 |
| 2 | 0.15 |
| 3 | 0.225 |
| 4 | 0.3375 |

Recovery is gated by two conditions: agent must be resting (`info['rested']`), and no damage is being applied (`applied_inc <= 0`). If the agent is taking damage while resting, recovery does not activate.

Injury is clamped to `[0, max_injury]` after all updates.

**When `with_injury=False`**: injury frozen; `injury_buffer` unchanged. Damage does not accumulate. Instead, any nonzero `damage` in the info dict triggers instant death (`done = True`) directly in `update_body` (`core.py:111`). The injury death code (4) is still emitted, but termination is immediate on first contact.

---

## Drive Computation

`calculate_drive(satiation, injury, params)` (`core.py:38`):

```
target = (setpoint, 0.0)            # ideal: fully satiated, zero injury
current = (satiation, injury)
drive = ‖current - target‖₂
       = sqrt((satiation - setpoint)^2 + injury^2)
```

This is the Euclidean distance in a 2D homeostatic space. The setpoint for injury is fixed at 0 (no injury is ideal). The setpoint for satiation is `params.setpoint` (default 100 = max satiation).

The drive is **not normalised** — raw values of satiation (0–100) and injury (0–100) are used. This means the maximum possible drive is approximately `sqrt(100^2 + 100^2) ≈ 141.4`.

The reward components `drive_hunger` and `drive_injury` logged in the info dict are squared normalised values:
```
drive_hunger = (1 - satiation / max_satiation)^2
drive_injury = (injury / max_injury)^2
```
These are for logging only and are not used in the reward computation.

---

## Metabolic Cost

`metabolic_cost` (default 1.0) is deducted from nutrition every step, unconditionally — it applies even when the agent is resting. Resting does not reduce the metabolic rate; it only activates injury recovery.

In the default configuration with `max_nutrition=100` and `metabolic_cost=1.0`, the agent starves in exactly 100 steps without eating. With food providing a net gain of 5, the agent needs to eat at least every 5 steps to maintain nutrition.

---

## Body State Flags Summary

| Flag | Disabled Effect |
|------|----------------|
| `with_nutrition=False` | No decay, no gain, no starvation; nutrition frozen at start value |
| `with_satiation=False` | Satiation frozen; excluded from drive computation |
| `with_injury=False` | Injury frozen at 0; any damage → instant death; injury buffer unused |

All three flags are static (`pytree_node=False`), so changing them requires recompilation.
