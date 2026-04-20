# 06 — Reward & Termination

> **Source**: `src/environment/core.py` (reward/termination sections, `core.py:450–511`) | **Back to hub**: [README](README.md)

---

## Overview

`jax_step` returns a single scalar `reward` and a boolean `done`. Two parallel reward modes are available, selected by `params.use_homeostatic_reward`. Five integer termination codes cover all ways an episode can end. Both modes apply the same eating penalty and death penalty on top of their base signals.

The reward mode is a static flag — it determines the compiled graph shape and requires recompilation when changed.

---

## Survival Reward

**Mode**: `use_homeostatic_reward=False`

Sparse reward:

```
reward_extrinsic = +1.0  if ate_food
                 =  0.0  otherwise
if done:
    reward_extrinsic -= death_penalty
```

This is a classic foraging reward: the agent must learn to seek food while avoiding things that terminate the episode (starvation, predators, danger zones). No continuous feedback about body state is provided — the agent must infer survival needs from the sparse food signal.

---

## Homeostatic Reward

**Mode**: `use_homeostatic_reward=True`

Dense drive-reduction reward:

```
prev_drive = calculate_drive(state.satiation,     state.injury_level, params)
curr_drive = calculate_drive(new_satiation,        new_injury,         params)
reward_homeostatic = prev_drive - curr_drive
if done:
    reward_homeostatic -= death_penalty
```

Where `calculate_drive(sat, inj, params) = sqrt((sat - setpoint)^2 + inj^2)`.

**Sign convention**: positive reward means drive decreased (agent moved toward homeostasis). Negative reward means drive increased (agent got hungrier or more injured).

**Magnitude**: the raw drive values are in units of the state variables (0–100), so a single food-eating step might produce a reward of ~5–15 units (satiation gain from nutrition) while a large damage hit might produce a penalty of 30+.

**Death penalty interaction**: the penalty is subtracted from `reward_homeostatic` (not added to it), making `reward_homeostatic` more negative on the terminal step. Combined with the drive change itself, the terminal step usually has a large negative reward.

This mode is suited for research on homeostatic regulation: the agent receives continuous feedback about whether its actions improve or degrade its internal state, regardless of whether food was explicitly obtained.

---

## Combined Reward

Both modes emit their component to separate info keys and combine into a single scalar:

```python
reward = reward_homeostatic + reward_extrinsic
# Exactly one of these is nonzero depending on mode
if ate_food:
    reward -= eating_reward_penalty
```

The `eating_reward_penalty` applies in **both** modes — it is deducted whenever food is consumed, regardless of mode. Default is 0.0 (no penalty). Setting it to a positive value discourages excessive eating.

The info dict always contains both `reward_homeostatic` and `reward_extrinsic` for logging, even though one will always be zero depending on the active mode.

---

## Termination Codes

`info['termination_reason']` is an `int32` scalar:

| Code | Name | Trigger condition | Notes |
|------|------|------------------|-------|
| 0 | Active | — | Episode continues |
| 1 | Truncated | `current_step >= max_steps` | Standard episode length limit |
| 2 | Starvation | `new_nutrition <= 0` | Only fires if `with_nutrition=True` |
| 3 | Overeating | `new_satiation >= max_satiation` | Only fires if `overeating_death=True` |
| 4 | Injury | `new_injury >= max_injury` | When `with_injury=False`, any damage triggers this instantly |

**Priority**: codes are evaluated in order 1 → 2 → 3 → 4 using `jnp.where` overrides. The last triggered condition wins. In practice, the highest-priority meaningful code is reported because later `jnp.where` calls override earlier ones. Code 4 always overrides codes 1–3 if injury is maxed in the same step.

**`done` flag**: `done = done_from_body OR truncated`. It is a boolean that controls whether `ParallelEnv.auto_reset_step` triggers a reset. The termination reason code is in `info` for logging purposes.

---

## Death Penalty & Eating Penalty

**`death_penalty`** (default 100):
- Applied once on the terminal step.
- Deducted from `reward_homeostatic` (homeostatic mode) or `reward_extrinsic` (survival mode).
- Magnitude should dominate the per-step reward scale so that premature death is strongly penalised.
- In homeostatic mode, the terminal step also includes the final drive change, which is typically a large positive (drive went to max as nutrition/health hit 0), so the death penalty must exceed that to maintain negative terminal reward.

**`eating_reward_penalty`** (default 0.0):
- Deducted whenever `ate_food=True`, regardless of mode.
- Rationale: in the eat-action configuration, this discourages spamming the eat action when not hungry, since the agent might be on food and triggering `ate_food` every step at zero metabolic benefit.
- Recommended value if used: small positive (e.g. 0.1–1.0).
