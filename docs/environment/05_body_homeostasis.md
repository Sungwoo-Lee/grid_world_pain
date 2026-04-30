# 05 — Body & Homeostasis

> **Source**: `src/environment/core.py` (`update_body`, `calculate_drive`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

The body subsystem models the agent's internal physiological state. It provides the biological substrate for interoception — the agent must actively monitor and regulate three internal variables to survive:

- **Nutrition**: objective energy reserve, decaying each step due to metabolism.
- **Satiation**: subjective fullness derived non-linearly from nutrition.
- **Injury**: accumulated physical damage, smoothed and recoverable through rest.

Together these implement a homeostatic RL scenario: the agent receives drive-reduction reward for moving its body state toward healthy targets. This is implemented in `update_body()` (`core.py:44`), called from `jax_step` Stage 5. `calculate_drive()` (`core.py:38`) computes the Euclidean distance to the homeostatic setpoint for use in reward and analysis.

Each subsystem can be independently disabled via `with_nutrition`, `with_satiation`, and `with_injury` flags. Additionally, `injury_level` and `nutrition` can be masked from the agent's observation vector using `injury_observable` and `nutrition_observable` flags, while still serving as the authoritative ground truth for survival and reward logic.

---

## Hidden Body States

A core design principle is the decoupling of **Ground Truth Body State** from **Sensory Observation**:

1. **Ground Truth Authority**: The fields `injury_level` and `nutrition` in `EnvState` are the source of truth for the agent's survival. If `injury_level >= max_injury` or `nutrition <= 0`, the agent dies, regardless of whether it can "feel" or "see" these values.
2. **Masking (Hidden States)**:
   - `injury_observable = False`: The `Injury` slot is removed from the observation vector. The agent must rely on delayed, convolved interoceptive signals (see doc 09) to infer its state.
   - `nutrition_observable = False`: The `Nutrition` slot is removed. The agent must rely on `Satiation` (which is always observable) or behavior to manage its energy.
3. **Internal Dynamics**: These ground truth values always update normally behind the scenes, ensuring consistent physics and reward calculation even in "blind" configurations.

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

**Interoceptive Nociception History** (`core.py:102`): After `injury_level` is updated, it is pushed into the `nociception_history_buffer` in `EnvState`. This buffer acts as a "tonic memory" of ground truth injury, enabling temporal convolution in the sensory system (see doc 09).

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

---

## Clarifications / FAQ

**Q: Does `overeating_death=True` actually kill the agent?**
A: **No — this is a latent bug.** `core.py:449-450` only sets `reason=3` when `new_satiation >= max_satiation`, but **never sets `done=True`**. The only `done` triggers are nutrition ≤ 0, injury ≥ max_injury (via `update_body`), and truncation. An agent with `overeating_death=True` that reaches `satiation == max_satiation` will have `termination_reason=3` reported but the episode continues. Treat this as telemetry, not a real termination. If you need actual overeating death, add the check to `update_body` or add `jnp.where(satiation>=max_satiation, True, done)` after the reason is set.

**Q: What is `applied_inc` exactly and why does it gate recovery?**
A: `applied_inc = temp_buffer[0]` is the front of the ring buffer **after** this step's damage has been added (`core.py:73, 76`). If the agent took damage on this step, then `inc > 0` → `applied_inc > 0` → `can_recover=False`. The gate `applied_inc <= 0` prevents the recovery amount from cancelling fresh pain — resting through a predator attack still leaves you hurt.

**Q: Does `rest_streak` reset to 0 if the agent takes damage while resting?**
A: No — the streak check is purely on the action (`info['rested']`), not on damage. `new_rest_streak = prev + 1 if rested else 0` (`core.py:84`). So a rest-and-take-damage step **keeps the streak going** but skips recovery this step. On the next step (if no new damage), full streak-multiplied recovery resumes.

**Q: Is there a cap on `recovery_mult`?**
A: No. The exponential `(1 + recovery_accel_rate)^(streak-1)` grows without bound, so at `accel=0.5` and streak 20, recovery is `0.1 * 1.5^19 ≈ 222` per step. Practically the injury clamp at 0 prevents over-recovery, but be aware that long rest streaks make injury vanish very quickly — tune `recovery_accel_rate` for realistic behaviour.

**Q: What happens with `smoothing_duration=1`?**
A: Damage is applied in full on the step it's taken. `inc = damage / 1 = damage`, `temp_buffer[0] = damage`, `new_buffer[0] = 0` after the roll. Effectively disables smoothing — a single hit jumps injury by `damage`. Use this for deterministic "one-shot death" style tasks.

**Q: If I set `food_nutrition_gain = eating_nutrition_cost`, what happens?**
A: `ate_food_gain = 0`, so eating has no effect on nutrition — but `ate_food=True` still fires (lifecycle update consumes the resource, reward may include `+1` in survival mode minus `eating_reward_penalty`). Useful for training agents where food acquisition is a reward signal divorced from energy.

**Q: Can nutrition go negative?**
A: No — `clip(..., 0.0, max_nutrition)` at `core.py:56` clamps it to 0. Starvation death fires at `new_nutrition <= 0`, which includes exactly 0.

**Q: Does `metabolic_cost` apply when resting?**
A: Yes. Rest does NOT pause metabolism (`core.py:52` deducts unconditionally). Rest only enables injury recovery — it doesn't save nutrition.

**Q: What if damage comes from multiple sources in one step?**
A: All damage is summed into `info['damage']` before `update_body` sees it (`core.py:419`). The buffer receives the total, spread across `smoothing_duration` slots. There's no per-source tracking in the buffer.

**Q: Is the buffer applied before or after recovery each step?**
A: Damage is applied first (`new_injury = prev_injury + applied_inc` at `core.py:77`), then recovery subtracts (`core.py:94`) only if `can_recover`. Net effect: `new_injury = prev_injury + applied_inc - (rested ? recovery : 0)` where recovery only fires if `applied_inc <= 0`.

**Q: What's `state.satiation` when `with_satiation=False` at reset?**
A: It's still computed via the power-law from nutrition at reset (`core.py:727-728`). The flag only prevents re-derivation at *step* time — reset always initialises satiation from nutrition regardless.

**Q: What's the termination_reason when `with_injury=False` and damage kills the agent?**
A: The reason code will usually stay at `0` (active) or `1` (truncated) because the `reason = jnp.where(new_injury >= params.max_injury, 4, reason)` check fires against the frozen `new_injury = prev_injury` which is 0. The episode does end (`done=True` from `core.py:111`) but the reason is misleading. This is a minor labelling inconsistency.

**Q: Is `calculate_drive` always 2D (satiation + injury) regardless of `with_*` flags?**
A: Yes. The drive function doesn't know about the flags; it always takes `(satiation, injury)` and computes distance to `(setpoint, 0)`. If `with_satiation=False`, satiation is frozen at its reset value — so drive may not change as expected. The homeostatic reward path is designed for the all-flags-on default.

**Q: Does the reward ever become very large in a single step?**
A: Yes at termination: `reward = drive_delta - death_penalty`. With `death_penalty=100` (default) the reward can spike to ≈ -100 on death. Some algorithms (Dreamer-style) are sensitive to this spike — consider scaling or clipping `death_penalty` when reward variance matters.

**Q: Setpoint defaults — is it always `max_satiation`?**
A: Yes by convention (`configs/environment/default.yaml` has `satiation_setpoint: 100 == max_satiation`). Setting it lower models an agent whose "ideal" is partial fullness — rarely used.

**Q: Can `nutrition_to_satiation_scaling_factor = 0` work?**
A: `fullness_ratio^0 = 1` for all positive fullness, so `new_satiation = max_satiation` always. Non-useful — avoid `k=0`. The formula is undefined for `k<0` when nutrition is 0 (divide by zero in fullness^k), so keep `k > 0`.
