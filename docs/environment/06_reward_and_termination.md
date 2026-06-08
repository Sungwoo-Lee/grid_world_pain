# 06 — Reward & Termination

> **Source**: `src/environment/core.py` (reward: lines 550–571; termination codes: lines 538–548; termination from body: lines 106–116) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Plain-language entry point

This document describes **how the environment scores the agent** and **when an episode ends**.

The environment offers two reward modes, selected once at config time:

- **Survival reward** (`use_homeostatic_reward=False`): sparse, food-only signal. The agent gets `+1` every time it eats food and `-death_penalty` when the episode ends. No continuous feedback about the internal body state — the agent must learn to seek food without being told it is hungry.
- **Homeostatic reward** (`use_homeostatic_reward=True`): dense, drive-reduction signal. Every step, the agent is rewarded for moving *closer to homeostasis* (less hungry, less injured) and penalised for moving further away. Death still adds an additional penalty.

**Performance is always measured in survival steps** (how long the agent stays alive), never cumulative reward. Reward shapes learning; survival steps measure success.

An episode can end in four ways (termination codes 1–4), or keep running (code 0). The integer code is in `info['termination_reason']` each step for logging.

---

## Reward — Survival Mode

**Mode**: `params.use_homeostatic_reward = False`

```python
# core.py:566–567
reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
reward_extrinsic = jnp.where(done, -params.death_penalty, reward_extrinsic)
```

**Important**: the `done` branch *replaces* the entire `reward_extrinsic` with `-params.death_penalty`. It does not subtract from an existing value. If food was eaten on the same terminal step, the `+1.0` is discarded — the terminal step yields exactly `-death_penalty` (before the eating penalty, see below).

The final combined reward (same formula applies in both modes):

```python
# core.py:569–571
reward = reward_homeostatic + reward_extrinsic
# Apply eating penalty whenever food was eaten (both modes)
reward = jnp.where(ate_food, reward - params.eating_reward_penalty, reward)
```

In survival mode, `reward_homeostatic` is always `0.0`, so:

```
reward = reward_extrinsic - (eating_reward_penalty  if ate_food else 0)
```

Typical per-step values:
- Alive, no food: `0.0`
- Alive, ate food: `1.0 - eating_reward_penalty`
- Terminal step (death or truncation): `-death_penalty` (eating penalty still applies if food was eaten on that step)

---

## Reward — Homeostatic Mode

**Mode**: `params.use_homeostatic_reward = True`

Dense drive-reduction reward based on `calculate_drive`:

```python
# core.py:38–42
def calculate_drive(satiation, injury, params):
    target  = jnp.array([params.setpoint, 0.0])
    current = jnp.stack([satiation, injury], axis=-1)
    return jnp.linalg.norm(current - target, axis=-1)
# i.e.: sqrt((satiation - setpoint)^2 + injury^2)
```

Per-step reward:

```python
# core.py:560–564
prev_drive = calculate_drive(state.satiation, state.injury_level, params)
curr_drive = calculate_drive(new_satiation, new_injury, params)
reward_homeostatic = prev_drive - curr_drive
# Death penalty applied on any terminal step (done=True):
reward_homeostatic = jnp.where(done, reward_homeostatic - params.death_penalty, reward_homeostatic)
```

Unlike survival mode, the death penalty here is **subtracted from** (not replaces) the existing drive-reduction term. The terminal step reward is:

```
reward_homeostatic_terminal = (prev_drive - curr_drive) - death_penalty
```

The eating penalty then applies on top if food was eaten:

```
reward = reward_homeostatic - (eating_reward_penalty  if ate_food else 0)
```

**Sign convention**: `reward_homeostatic > 0` means drive decreased (agent moved toward homeostasis — less hungry, less injured). `reward_homeostatic < 0` means drive increased (more hungry, more injured, or terminal step).

**Units**: drive is in the same units as `satiation` and `injury` (both range 0–`max_satiation`/`max_injury`, defaults 100). A typical satiation gain from eating might produce a drive-change of ~5–20; severe injury can produce a per-step penalty of 30+. The theoretical per-step maximum magnitude is `sqrt(max_satiation^2 + max_injury^2) ≈ 141` at default settings.

**Residual drive**: `calculate_drive` measures distance from `[setpoint, 0]`, not from `[max_satiation, 0]`. If `setpoint < max_satiation` (the normal case), there is always a drive ≥ 0. If `setpoint > max_satiation`, even a fully fed agent has a nonzero residual drive and can never reach zero drive.

**Note on logged drive components** (`drive_hunger`, `drive_injury` in `info`): these are normalised squared terms computed separately for logging only — they are **not** used to compute `reward_homeostatic`:

```python
# core.py:556–557  (info only, not part of reward)
drive_hunger = (1.0 - new_satiation / params.max_satiation) ** 2
drive_injury = (new_injury / params.max_injury) ** 2
```

---

## Combined Reward Formula (Both Modes)

```python
# core.py:569–571
reward = reward_homeostatic + reward_extrinsic   # exactly one is nonzero per mode
reward = jnp.where(ate_food, reward - params.eating_reward_penalty, reward)
```

The `info` dict always carries both `reward_homeostatic` and `reward_extrinsic`, even when one is `0.0`, to keep logging scripts mode-agnostic.

### Eating reward penalty

`params.eating_reward_penalty` (default `0.0`) is deducted whenever `ate_food = True`, in **both** modes. Setting it to a small positive value (e.g. `0.1–1.0`) discourages the agent from spamming the eat action when hunger is already satisfied. It affects only the reward signal — the actual body nutrition update (`food_nutrition_gain`, `eating_nutrition_cost`) is independent.

### Death penalty

`params.death_penalty` (default `100`) is applied on any terminal step (`done = True`), whether death is from starvation, injury, or `max_steps` truncation.

- In **survival mode**: replaces `reward_extrinsic` with `-death_penalty` (the step's food-eat bonus is lost).
- In **homeostatic mode**: subtracted from the drive-change term (the drive change from the final body update is still included).

FAQ: Is the death penalty also applied on truncation? **Yes.** `done = done_from_body OR truncated` (core.py:548), so any `done=True` triggers the penalty regardless of cause. If you want truncation to be reward-neutral, set `death_penalty=0`.

---

## Termination Codes

`info['termination_reason']` is an `int32` scalar set every step. The `EnvState.terminated` field stores the boolean `done` flag (not the integer code).

| Code | Name | Trigger condition | `done`? | Notes |
|------|------|-----------------|---------|-------|
| 0 | Active | — (default) | No | Episode is running normally |
| 1 | Truncated | `(state.current_step + 1) >= params.max_steps` | Yes | Standard episode length limit |
| 2 | Starvation | `new_nutrition <= 0.0` | Yes (via `update_body`) | No `with_nutrition` guard in `jax_step` — see note below |
| 3 | Overeating | `new_satiation >= params.max_satiation` | **No** | Only set if `params.overeating_death=True`; does NOT set `done=True` |
| 4 | Injury | `new_injury >= params.max_injury` | Yes (via `update_body`) | No `with_injury` guard in `jax_step` — see note below |

**Priority** (highest code wins): `4 > 3 > 2 > 1 > 0`. Codes are applied via sequential `jnp.where` — later checks overwrite earlier ones:

```python
# core.py:540–545
reason = jnp.array(0, dtype=jnp.int32)
reason = jnp.where(truncated,                              1, reason)   # lowest priority
reason = jnp.where(new_nutrition <= 0.0,                   2, reason)
if params.overeating_death:
    reason = jnp.where(new_satiation >= params.max_satiation, 3, reason)
reason = jnp.where(new_injury >= params.max_injury,        4, reason)   # highest priority
```

If starvation and truncation both fire in the same step, `reason=2` wins (starvation overwrites truncation). If injury and starvation both fire, `reason=4` wins.

### Code 3 / overeating: informational only

`reason=3` is **never** the direct cause of episode termination. The `done` flag comes from `update_body` (nutrition and injury only) and `truncated` — overeating does not contribute. Code 3 appears in `info['termination_reason']` for the triggering step, but the episode continues.

### `with_nutrition=False` and the starvation code

The starvation check at `core.py:542` (`reason = jnp.where(new_nutrition <= 0.0, 2, reason)`) has **no guard for `with_nutrition`**. When `with_nutrition=False`, `update_body` keeps nutrition frozen at its reset value (`start_nutrition`) and never decays it. If `start_nutrition=0.0`, this check fires every step, setting `reason=2` each step even though `update_body` does not set `done=True` for starvation. The `done` flag is unaffected, but `termination_reason` reads `2` spuriously. See Suspected Code Bugs below.

### `with_injury=False` and the injury code

Similarly, `core.py:545` has no `with_injury` guard. When `with_injury=False`, `new_injury` stays frozen at its reset value. If the reset value is `0.0` and `max_injury > 0`, the check never fires. However, damage *does* still terminate the episode via `update_body`'s instant-death branch (`done = jnp.where(damage > 0, True, done)` — core.py:115). In that case, `new_injury` never reaches `max_injury`, so `reason` will be `1` (truncation) or `0` — **not** `4`. The claim that code 4 fires for `with_injury=False` damage-deaths is incorrect.

---

## `done` Flag Assembly

```python
# core.py:548
done = jnp.logical_or(done, truncated)
```

Where `done` on the left is `done_from_body` returned by `update_body` (covers starvation and injury thresholds), and `truncated = (next_step >= params.max_steps)`.

`done` is returned as the third element of `jax_step` and stored in `state.terminated` (bool). The integer reason code is in `info['termination_reason']` only.

### Truncation vs body-death distinction

There is no separate truncation boolean in the step output. To distinguish truncation from body-death downstream, read `info['termination_reason']`: code `1` means pure truncation; codes `2` or `4` mean body death (noting that starvation+truncation on the same step reports `2`, not `1`).

---

## Death Conditions Summary

| Condition | `with_*` guard in `jax_step`? | `done=True` source | `reason` code |
|-----------|------------------------------|-------------------|--------------|
| `new_nutrition <= 0.0` | No (body does; `jax_step` does not) | `update_body` line 109 | 2 |
| `new_injury >= params.max_injury` | No (body does; `jax_step` does not) | `update_body` line 112 | 4 |
| `damage > 0` when `with_injury=False` | `with_injury=False` branch in `update_body` | `update_body` line 115 | 1 or 0 (not 4) |
| `next_step >= params.max_steps` | — | `jax_step` (truncated) | 1 |
| `new_satiation >= params.max_satiation` (overeating_death=True) | `params.overeating_death` | **none** (informational) | 3 |

---

## Clarifications / FAQ

**Q: Does `overeating_death=True` actually terminate the episode?**
A: **No.** The `reason=3` branch at `core.py:543–544` sets the reason code but does not set `done=True`. The `done` flag comes only from `update_body` (nutrition/injury) and `truncated`. Code 3 in telemetry is informational only.

**Q: When multiple termination conditions fire in the same step, which code wins?**
A: The highest-priority code by the `jnp.where` ordering: `4 > 3 > 2 > 1 > 0`. Example: starvation and truncation on the same step → `reason=2` (starvation overwrites truncation). Injury and starvation → `reason=4`.

**Q: Is the death penalty applied on truncation?**
A: Yes. `done = done_from_body OR truncated`, so truncation yields `done=True` and triggers the penalty in both reward modes. Set `death_penalty=0` for a truncation-neutral setup.

**Q: What is the reward on the very last step when no food was eaten (survival mode)?**
A: `reward = -death_penalty`. The `done` branch in `core.py:567` replaces `reward_extrinsic` entirely with `-death_penalty`.

**Q: What is the reward on a step where food is eaten AND the episode ends (survival mode)?**
A: `reward = -death_penalty - eating_reward_penalty`. The `+1.0` food bonus is discarded because the `done` branch replaces `reward_extrinsic` before the eating penalty is applied.

**Q: In homeostatic mode, what is the terminal step reward?**
A: `(prev_drive - curr_drive) - death_penalty - (eating_reward_penalty if ate_food else 0)`. Unlike survival mode, the drive-change term is retained and the penalty is additive.

**Q: Why do `reward_homeostatic` and `reward_extrinsic` both appear in the info dict?**
A: For logging separability — both are always written regardless of mode. In survival mode, `reward_homeostatic=0.0` every step; in homeostatic mode, `reward_extrinsic=0.0`. This lets a single analysis script handle both modes without mode-checking.

**Q: Does the reward sign tell me whether the agent is succeeding?**
A: Only in homeostatic mode, where `reward > 0` iff drive decreased this step. In survival mode, `reward ∈ {0, 1} - eating_reward_penalty` during the episode and `= -death_penalty` on termination.

**Q: How does `eating_reward_penalty` interact with `food_nutrition_gain`?**
A: Independently. `eating_reward_penalty` affects only the reward signal; `food_nutrition_gain` and `eating_nutrition_cost` affect the actual body state. You can configure eating as body-beneficial but reward-penalised.

**Q: Can I use homeostatic reward without injury (`with_injury=False`)?**
A: Yes. When `with_injury=False`, `new_injury` stays frozen at its reset value (typically `0.0`). Drive degenerates to `|satiation - setpoint|`. Reward depends only on satiation changes.

**Q: What's the reward shape?**
A: Scalar (`[]`). Single float per step. `ParallelEnv` wraps it to shape `[num_envs]` via vmap.

**Q: What's the bound on reward magnitude?**
A: Not strictly bounded. In homeostatic mode, the per-step drive change is bounded by `sqrt(max_satiation^2 + max_injury^2) ≈ 141` with defaults. Terminal step adds `-death_penalty`. Practical range: `[-(death_penalty + 141), +141]`. Survival mode: `[-death_penalty, 1.0]` per step (minus `eating_reward_penalty`).

**Q: Is reward clipped inside the environment?**
A: No. Apply clipping in the training loop if your algorithm requires it.

**Q: Does `info['termination_reason']` reliably indicate the true episode-ending cause?**
A: Mostly, with these caveats:
- `overeating_death=True` → reason=3, but episode continues (not terminal).
- `with_injury=False` → damage-triggered death produces reason=0 or 1, not 4.
- `with_nutrition=False` and `start_nutrition=0.0` → reason=2 every step (spurious).
Prefer checking `done` for the actual termination signal; use `reason` for diagnostic labelling.
