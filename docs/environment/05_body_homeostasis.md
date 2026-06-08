# 05 — Body & Homeostasis

> **Source**: `src/environment/core.py` (`update_body` `core.py:44–117`, `calculate_drive` `core.py:38–42`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## What this doc is about

The agent has an internal body — a small physiological simulation that runs inside every step of the environment. Three numbers matter: **nutrition** (how much energy the agent has stored), **satiation** (a subjective sense of fullness derived from nutrition), and **injury level** (accumulated physical damage). The agent dies of starvation when nutrition hits zero, or of injury when accumulated damage reaches its maximum. In between, drive-reduction reward pushes the agent to keep its body close to a healthy setpoint.

This doc describes exactly how those three numbers change each step, including all constants, clip bounds, and the two supporting buffers — the injury smoothing ring buffer and the nociception history buffer — that connect body state to the interoceptive sensor (doc 09).

---

## Body State Fields in `EnvState`

All fields are scalars (shape `[]`) unless noted. Declared in `src/environment/state.py:61–68`.

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `satiation` | `[]` | float32 | Subjective fullness derived each step from nutrition |
| `nutrition` | `[]` | float32 | Objective energy store; decays each step |
| `injury_level` | `[]` | float32 | Accumulated physical damage, smoothed and recoverable |
| `injury_buffer` | `[smoothing_duration]` | float32 | Ring buffer that spreads a single damage event across future steps |
| `nociception_history_buffer` | `[interoceptive_kernel_length]` | float32 | Sliding window of past `injury_level` values; idx 0 = most recent |
| `last_collision_noc` | `[]` | float32 | Nociception intensity of the most recent wall/obstacle collision |
| `rest_streak` | `[]` | int32 | Consecutive resting steps (used to accelerate recovery) |

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

`core.py:50–58`

**Governing equation** (applied each step when `with_nutrition=True`):

```
new_nutrition = clip(prev_nutrition - metabolic_cost + ate_food_gain, 0.0, max_nutrition)

where:
  ate_food_gain = food_nutrition_gain - eating_nutrition_cost   (if ate_food else 0)
```

- `metabolic_cost` (default 1.0): drained unconditionally every step, even while resting.
- `food_nutrition_gain` (default 6): gross nutrition from consuming a food resource.
- `eating_nutrition_cost` (default 1.0): the physical cost of the eating act, subtracted from gain.
- Net gain from eating: `6 - 1 = 5` nutrition in the default config.
- Clamped to `[0.0, max_nutrition]` at `core.py:56`.

**When `with_nutrition=False`**: `new_nutrition = prev_nutrition` — no decay, no gain, no starvation death (`core.py:57–58`).

**Starvation death**: triggered inside `update_body` at `core.py:108–110` when `new_nutrition <= 0.0` (termination reason code 2).

**Starvation timeline with defaults** (`max_nutrition=100`, `metabolic_cost=1.0`, net eating gain=5): the agent starves in exactly 100 steps without eating; needs to eat at least once every 5 steps to maintain nutrition.

**Overeating death**: when `overeating_death=True`, the termination-reason code 3 is set in `jax_step` at `core.py:543–544` when `new_satiation >= max_satiation`. **However, this does not set `done=True`** — it is telemetry only. See FAQ.

---

## Satiation Dynamics

`core.py:61–66`

Satiation is **not independently tracked** — it is re-derived deterministically from nutrition every step:

```
fullness_ratio = clip(new_nutrition / max_nutrition, 0.0, 1.0)
new_satiation  = max_satiation * fullness_ratio ^ k
```

where `k = nutrition_to_satiation_scaling_factor` (default 1.0 = linear).

This power-law mapping allows different subjective hunger curves:
- `k = 1.0`: linear — satiation tracks nutrition directly.
- `k < 1.0`: sub-linear — agent feels relatively full even with low nutrition (optimistic subjective hunger).
- `k > 1.0`: super-linear — agent feels relatively empty unless nutrition is near-full (pessimistic subjective hunger).

Since satiation is fully derived from nutrition, `random_start_satiation` is a no-op — the flag exists in `EnvParams` (`state.py:179`) but is **never read** during `jax_reset`. The initial satiation is always computed from the initial nutrition value (`core.py:915–916`).

**When `with_satiation=False`**: `new_satiation = state.satiation` — value frozen at its reset-derived level (`core.py:65–66`).

---

## Reset Initialisation of Body State

`core.py:907–925` (inside `jax_reset`)

| Field | Condition | Value |
|-------|-----------|-------|
| `nutrition` | `random_start_nutrition=True` | Uniform `[max_nutrition / 2, max_nutrition]` |
| `nutrition` | `random_start_nutrition=False` | `params.start_nutrition` |
| `satiation` | always | `max_satiation * clip(nutrition / max_nutrition, 0, 1) ^ k` |
| `injury_level` | `random_start_injury=True` | Uniform `[0, max_injury / 2]` |
| `injury_level` | `random_start_injury=False` | `0.0` |
| `injury_buffer` | always | `jnp.zeros(smoothing_duration)` |
| `nociception_history_buffer` | always | `jnp.zeros(interoceptive_kernel_length)` |
| `last_collision_noc` | always | `0.0` |
| `rest_streak` | always | `0` |

---

## Injury Ring Buffer & Streak Recovery

`core.py:68–100`

### Damage Smoothing (Ring Buffer)

Rather than applying damage instantly, each step's damage is spread across `smoothing_duration` future steps. This prevents large single-hit deaths and creates a biologically realistic ramping pain experience.

```
Step-by-step when with_injury=True:

  inc           = damage / smoothing_duration        # per-step dose from new damage
  temp_buffer   = state.injury_buffer + inc          # broadcast add to all future slots
  applied_inc   = temp_buffer[0]                     # front slot applied this step
  new_injury    = prev_injury + applied_inc

  new_buffer    = jnp.roll(temp_buffer, -1).at[-1].set(0.0)
                # shift entire buffer left by 1; zero the new tail slot
```

`core.py:72–80`

A single damage event of value `D` results in `D / smoothing_duration` injury added per step for the next `smoothing_duration` steps. The front of the buffer is applied **immediately on the same step as the damage** — there is no onset delay. After the roll, the tail of the buffer is zeroed so no phantom increments persist after the window closes.

**Example**: `D=20`, `smoothing_duration=10` → `inc=2.0` per step. At step 0 (damage step), `applied_inc = 0 + 2 = 2`; steps 1–9 each apply 2 more; step 10 applies nothing (buffer cleared). Net: `+20` total injury over 10 steps.

**Special case** `smoothing_duration=1`: `inc = damage`, `applied_inc = damage` immediately. Disables smoothing — a single hit jumps injury by the full damage value.

### Recovery (Streak-Based Exponential)

`core.py:82–96`

When the agent selects action 4 (Rest, if `rest_action_enabled=True`) and no buffered damage is being applied in this step (`applied_inc <= 0`), injury recovers with an exponentially accelerating rate:

```
# 1. Update streak (purely based on action, not damage)
new_rest_streak = prev_rest_streak + 1   if info['rested']
                = 0                      otherwise

# 2. Compute multiplier using the updated (already-incremented) streak
recovery_mult   = (1 + recovery_accel_rate) ^ (max(new_rest_streak, 1) - 1)
recovery_amount = recovery_base_rate * recovery_mult

# 3. Apply recovery only if resting AND no damage being absorbed this step
can_recover = info['rested'] AND (applied_inc <= 0)
new_injury  = new_injury - recovery_amount    if can_recover
            = new_injury                      otherwise

# 4. Clamp
new_injury = clip(new_injury, 0.0, max_injury)
```

`core.py:84, 89–94, 96`

Note: `new_rest_streak` (already incremented) is used when computing `recovery_mult`, not `prev_rest_streak`. This means the very first resting step uses streak=1 → multiplier=1.0 (base rate).

**Default recovery table** (`recovery_base_rate=0.1`, `recovery_accel_rate=0.5`):

| Consecutive rest streak | `recovery_mult` | Recovery per step |
|------------------------|-----------------|------------------|
| 1 | 1.0 | 0.1 |
| 2 | 1.5 | 0.15 |
| 3 | 2.25 | 0.225 |
| 4 | 3.375 | 0.3375 |
| 10 | ~57.7 | ~5.77 |

**Streak reset rule**: `new_rest_streak = prev + 1 if rested else 0` (`core.py:84`). The streak depends only on the *action*, not on damage. A step where the agent rests **and** takes damage extends the streak by 1 but skips recovery (because `applied_inc > 0`). On the next undamaged rest step, full streak-multiplied recovery resumes.

**No cap on `recovery_mult`**: the exponential grows without bound. At `recovery_accel_rate=0.5` and streak 20, recovery ≈ 222 per step — the injury clamp at 0 prevents over-subtraction, but very long streaks make injury vanish nearly instantaneously. Tune `recovery_accel_rate` for the desired recovery timescale.

**`with_injury=False`**: `new_injury = prev_injury`, `new_buffer = state.injury_buffer`, `new_rest_streak = prev_rest_streak` (all frozen, `core.py:97–100`). Any nonzero `damage > 0` triggers instant death directly at `core.py:115`. The injury death reason code (4) is not set because `new_injury` remains frozen at 0 — see FAQ.

---

## Nociception History Buffer

`core.py:102–104`

After `injury_level` is updated (whether `with_injury=True` or `False`), the new value is pushed into the front of `nociception_history_buffer`:

```python
new_nociception_history = jnp.roll(state.nociception_history_buffer, 1).at[0].set(new_injury)
```

- `jnp.roll(..., +1)`: shifts all existing entries one slot to the **right** (toward higher indices), so the oldest entry falls off the end.
- `.at[0].set(new_injury)`: places the latest `injury_level` at slot 0.
- Result: **slot 0 = most recent**, slot 1 = one step ago, slot `K-1` = oldest retained.

This buffer update runs **unconditionally** — it is not gated on `with_injury`. If `with_injury=False`, `new_injury` stays at `prev_injury` (0.0 from reset), so the buffer fills with zeros but the write still executes every step.

This buffer is the authoritative input for the interoceptive nociception sensor (doc 09), which convolves it with a discrete alpha kernel to produce a delayed, smoothed perception of internal injury.

**Shape**: `[interoceptive_kernel_length]` float32. Initialised to `jnp.zeros(interoceptive_kernel_length)` at reset (`core.py:925`).

---

## `last_collision_noc` — When It Is Set

`src/environment/core.py:497`, `core.py:652`

`last_collision_noc` is **not** updated inside `update_body`. It is set in `jax_step` (Stage 4, interaction logic) and written to the new state at `core.py:652`:

```python
# core.py:497
collision_noc = jnp.where(
    just_collided,
    jnp.max(jnp.where(at_attempted_obs, params.obs_nociception, 0.0), initial=0.0),
    0.0
)
# ... later in state._replace at core.py:652:
last_collision_noc = collision_noc,
```

`just_collided` is `True` when the agent attempted to move into a blocking obstacle and was rejected (`move_agent` returns `is_collision=True`, `core.py:29–36`). When a collision occurs, `collision_noc` is the `obs_nociception` value of the specific blocking obstacle that was hit (max over the obstacle array masked to the attempted cell). If no collision occurred this step, `last_collision_noc = 0.0`.

This value feeds the exteroceptive nociception sensor (doc 09, sensor #5) as one of four contact-based pain sources. It does **not** directly affect `injury_level` — collision damage flows through `info['damage']` → `update_body`, while `last_collision_noc` is a separate perceptual signal.

---

## Drive Computation

`calculate_drive(satiation, injury, params)` — `core.py:38–42`

```
target  = [params.setpoint, 0.0]          # ideal state: full satiation, zero injury
current = [satiation, injury]
drive   = ||current - target||_2
        = sqrt((satiation - setpoint)^2 + injury^2)
```

Euclidean distance in a 2D homeostatic space (satiation axis, injury axis). Both axes use raw values (0–100 range with defaults), so the maximum possible drive is approximately `sqrt(100^2 + 100^2) ≈ 141.4`.

**Default setpoint**: `params.setpoint = 100 = max_satiation` (fully satiated, zero injury = optimal).

**Inputs to `calculate_drive`**: called twice per step in `jax_step` (`core.py:560–562`) — once with the **previous** state, once with the **new** state — to compute reward as drive reduction:

```python
prev_drive = calculate_drive(state.satiation, state.injury_level, params)
curr_drive = calculate_drive(new_satiation,   new_injury,         params)
reward_homeostatic = prev_drive - curr_drive   # positive = moved toward homeostasis
```

**Logging components**: `drive_hunger` and `drive_injury` (written to `info`) are squared, normalised sub-components computed separately in `jax_step` at `core.py:556–558`:

```
drive_hunger = (1 - new_satiation / max_satiation)^2
drive_injury = (new_injury / max_injury)^2
```

These are for analysis logging **only** — they are not used in the reward formula. The actual reward uses the raw Euclidean drive via `calculate_drive`.

**Drive is not normalised**: raw values (0–100) are used. Both `drive_hunger` and `drive_injury` are dimensionless `[0, 1]` by construction, but the reward-driving `drive` value is in the same units as the body state variables. Reconfiguring `max_satiation` or `max_injury` changes the drive scale.

---

## Metabolic Cost

`metabolic_cost` (default 1.0) is deducted from nutrition every step, unconditionally — it applies even when the agent is resting (`core.py:52`). Resting does not pause metabolism; it only activates injury recovery.

---

## Body State Flags Summary

| Flag | Disabled Effect |
|------|----------------|
| `with_nutrition=False` | No decay, no gain, no starvation; nutrition frozen at start value (`core.py:57–58`) |
| `with_satiation=False` | Satiation frozen at reset-derived value; excluded from drive changes (`core.py:65–66`) |
| `with_injury=False` | Injury and injury_buffer frozen; `rest_streak` frozen; any `damage > 0` → instant death (`core.py:97–100, 115`) |

All three flags are static (`pytree_node=False` in `EnvParams`), so changing them requires recompilation.

**Note**: `nociception_history_buffer` updates are **not gated** by `with_injury`. The buffer rolls unconditionally each step; if `with_injury=False`, it fills with the frozen `prev_injury` value (0.0 from reset).

---

## Clarifications / FAQ

**Q: Does `overeating_death=True` actually kill the agent?**
A: **No — this is a latent bug.** `core.py:543–544` only sets `reason=3` when `new_satiation >= max_satiation`, but **never sets `done=True`**. The only `done` triggers are nutrition ≤ 0, injury ≥ max_injury (via `update_body`), and truncation. An agent with `overeating_death=True` that reaches `satiation == max_satiation` will have `termination_reason=3` reported but the episode continues. Treat this as telemetry, not a real termination. To add actual overeating death, add `done = jnp.where(new_satiation >= params.max_satiation, True, done)` inside `update_body` before the return statement.

**Q: What is `applied_inc` exactly and why does it gate recovery?**
A: `applied_inc = temp_buffer[0]` is the front of the ring buffer **after** this step's damage has been added (`core.py:73–76`). If the agent took damage on this step, then `inc > 0` → `applied_inc > 0` → `can_recover=False`. The gate `applied_inc <= 0` prevents recovery from cancelling fresh pain — resting through a predator attack still leaves you hurt.

**Q: Does `rest_streak` reset to 0 if the agent takes damage while resting?**
A: No — the streak check is purely on the action (`info['rested']`), not on damage. `new_rest_streak = prev + 1 if rested else 0` (`core.py:84`). A rest-and-take-damage step **extends the streak** but skips recovery (because `applied_inc > 0`). On the next step with no new damage absorbed, full streak-multiplied recovery resumes.

**Q: Does `recovery_mult` use the old or new rest_streak?**
A: The **new** (already incremented) streak (`core.py:89`). On the very first resting step, `new_rest_streak=1`, `recovery_mult = (1+accel)^0 = 1.0` — the base rate. On the second consecutive rest step, `new_rest_streak=2`, `recovery_mult = 1+accel`.

**Q: What happens with `smoothing_duration=1`?**
A: Damage is applied in full on the step it's taken. `inc = damage / 1 = damage`, `applied_inc = damage`, the buffer tail slot is zeroed after the roll. Effectively disables smoothing. Use this for deterministic "one-shot impact" style tasks.

**Q: If I set `food_nutrition_gain = eating_nutrition_cost`, what happens?**
A: `ate_food_gain = 0`, so eating has no effect on nutrition — but `ate_food=True` still fires (lifecycle update consumes the resource, reward may include `+1` in survival mode minus `eating_reward_penalty`). Useful when food acquisition is a reward signal divorced from energy.

**Q: Can nutrition go negative?**
A: No — `clip(..., 0.0, max_nutrition)` at `core.py:56` clamps it. Starvation death fires at `new_nutrition <= 0`, which includes exactly 0.

**Q: Does `metabolic_cost` apply when resting?**
A: Yes. Rest does NOT pause metabolism (`core.py:52` deducts unconditionally). Rest only enables injury recovery.

**Q: What if damage comes from multiple sources in one step?**
A: All damage is summed into `info['damage']` before `update_body` sees it (`core.py:499` aggregates `damage_res + damage_pred + damage_obs_overlap + damage_obs_collision`). The buffer receives the total spread across `smoothing_duration` slots. There is no per-source tracking in the buffer.

**Q: Is the buffer applied before or after recovery each step?**
A: Damage is applied first (`new_injury = prev_injury + applied_inc` at `core.py:77`), then recovery subtracts (`core.py:94`) only if `can_recover`. Net effect: `new_injury = prev_injury + applied_inc - (rested AND applied_inc<=0 ? recovery : 0)`, then clamped to `[0, max_injury]`.

**Q: What's `state.satiation` when `with_satiation=False` at reset?**
A: It's still computed via the power-law from nutrition at reset (`core.py:915–916`). The flag only prevents re-derivation at *step* time — reset always initialises satiation from nutrition regardless.

**Q: What's the termination_reason when `with_injury=False` and damage kills the agent?**
A: The reason code stays at `0` (active) or `1` (truncated) because `reason = jnp.where(new_injury >= params.max_injury, 4, reason)` fires against the frozen `new_injury = prev_injury` (0.0). The episode ends (`done=True` from `core.py:115`) but the reason code is misleading. This is a minor labelling inconsistency.

**Q: Is `calculate_drive` always 2D (satiation + injury) regardless of `with_*` flags?**
A: Yes. `calculate_drive` doesn't inspect the flags; it always takes `(satiation, injury)` and computes distance to `(setpoint, 0)`. If `with_satiation=False`, satiation is frozen at its reset value — drive changes only with injury.

**Q: Does the reward ever become very large in a single step?**
A: Yes at termination: `reward = drive_delta - death_penalty`. With `death_penalty=100` (default) the terminal step typically carries a large negative reward. The drive itself can jump up to ~141 on the death step, but the penalty dominates. Some algorithms (Dreamer-style) are sensitive to this spike — consider scaling `death_penalty` when reward variance matters.

**Q: Setpoint defaults — is it always `max_satiation`?**
A: Yes by convention (`configs/environment/default.yaml` has `satiation_setpoint: 100 == max_satiation`). Setting it lower models an agent whose "ideal" is partial fullness — rarely used.

**Q: Can `nutrition_to_satiation_scaling_factor = 0` work?**
A: `fullness_ratio^0 = 1` for all positive fullness, so `new_satiation = max_satiation` always. Not useful. The formula is undefined for `k<0` when `nutrition=0` (0^negative), so keep `k > 0`.

**Q: What random ranges apply at reset for nutrition and injury?**
A: `random_start_nutrition=True` → `uniform[max_nutrition/2, max_nutrition]` (`core.py:910–911`). `random_start_injury=True` → `uniform[0, max_injury/2]` (`core.py:919–920`). Both use independent sub-keys split from `body_key` at `core.py:907`.

**Q: Is there a cap on `recovery_mult`?**
A: No hard cap. At `recovery_accel_rate=0.5` and streak 20, recovery ≈ 222 per step. The injury clamp at 0 prevents over-recovery, but long rest streaks make injury vanish very quickly. Tune `recovery_accel_rate` for the desired recovery timescale.

**Q: Does `last_collision_noc` affect injury?**
A: No. `last_collision_noc` is a perceptual signal only — it feeds the exteroceptive nociception sensor (doc 09). Collision damage flows via `info['damage']` into `update_body` and accumulates in `injury_level`. The two are independent; you can have a collision that causes damage (if `obs_damage > 0`) and also sets `last_collision_noc` (if `obs_nociception > 0`), or configure either to be zero.
