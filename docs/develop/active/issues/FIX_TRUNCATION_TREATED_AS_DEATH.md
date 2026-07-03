---
title: "Fix: surviving to the time limit is punished like death (Finding B)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Fix: surviving to the time limit is punished like death (Finding B)

> **Status**: PLANNED
> **Opened**: 2026-07-04
> **Related**: [[v3_pipeline_correctness_diagnosis]] (Finding B — the confirmed bug this plan fixes) · source path `docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md`

---

## Context

In this project an agent is scored by **how many steps it survives**. An episode ends in one
of two ways: the agent **dies** (it starves, over-eats, or is injured past a threshold), or the
clock runs out — it **survives all the way to the step limit** (`max_steps`). Surviving to the
step limit is the *success* outcome of a survival task.

The bug: the environment applies the **death penalty** (a large −100 reward) at the end of
**every** episode — including the ones where the agent survived to the time limit. A pipeline
audit drove the real environment to both kinds of ending and read out the ground-truth reward:
an agent that **survived to the limit** received **−100.19** on its final step (it should have
received about **−0.19**, the ordinary homeostatic step value), while an agent that **died of
starvation** received **−100.66**. In other words, doing the right thing (surviving) and doing
the worst thing (dying) are rewarded almost identically. Because every episode ends in death or
timeout, this spurious −100 lands at the end of essentially every episode. The −100 is roughly
**500× larger** than the normal per-step reward (~0.2), so it dominates the learning signal.

There is a second, related defect on the learning side. When the clock simply runs out, the
episode was **cut off, not genuinely over** — the standard reinforcement-learning treatment is
to keep estimating the value of where the agent would have gone next (a "bootstrap"). The trainer
currently throws that estimate away on timeouts, treating a cut-off episode as if the world truly
ended.

This is **not** a bug the recent v3.0 code wave introduced — the offending lines are byte-identical
to the shared `main` branch, so every past run shares the same distortion (run-to-run comparisons
stay valid). But it does distort the survival objective the six currently-training runs are
optimising, and it plausibly made the model-based DreamerV3 runs harder to train (a hypothesis,
see below). This plan fixes it in two parts with a crisp scope split so the reader can approve
**Part 1 only** or **both parts**.

## Analysis

### Mechanism (ground truth, from the diagnosis)

All in `src/environment/core.py`, inside `jax_step`:

- **Line 690** — `update_body(...)` returns `done`. At this point `done` means **real death only**
  (starvation / over-eating / injury); `update_body` knows nothing about the step clock.
- **Line 694** — `truncated = next_step >= params.max_steps` — the agent **survived** to the limit.
- **Lines 698–703** — a `termination_reason` integer is computed and **already distinguishes the
  outcomes**: `1` = timeout, `2` = starvation, `3` = over-eating, `4` = injury (`0` = still active).
- **Line 706** — `done = jnp.logical_or(done, truncated)` — timeout is folded into `done`. From
  here on `done` means "episode ended, for any reason".
- **Lines 722 / 725** — `reward = jnp.where(done, reward - params.death_penalty, reward)` — the death
  penalty fires on `done`, so it fires on **timeout too**. This is the bug.
- **Line 824** — `terminated=done` is written into the returned state, i.e. the state's `terminated`
  flag also treats timeout as termination. (Consumed only by the parity-fixture generator; see
  "Why line 824 is left alone" below.)

Ground-truth reward decomposition (driver `tmp/20260704_finding_b_driver.py`, `basic/06`,
`death_penalty=100`, `max_steps=500`):

| Episode ending | `termination_reason` | real death? | final reward | penalty applied? |
|---|---|---|---|---|
| **Survive to limit** (timeout) | `1` | no | **−100.186** | **YES — wrong** (should be −0.186) |
| **Starvation** (death) | `2` | yes | −100.660 | YES — correct |

### Value-target side (the bootstrap)

The trainer stores the merged `done` for each step and feeds it to Generalized Advantage
Estimation (`src/models/recurrent_ppo_trainer.py`, `compute_gae`, line 63):
`delta = reward + gamma * next_value * (1 - done) - value`. On a timeout step `done = True`, so the
`gamma * next_value` bootstrap is **zeroed** — the value head is trained to regress the truncated
single-step reward instead of the discounted continuation. The correct behaviour is to zero the
bootstrap only on **real termination** (death), and to **retain** it on truncation. Standard PPO
truncation handling (e.g. Stable-Baselines3, CleanRL) does exactly this.

The signal needed to fix this is **already carried** trainer-side: `termination_reason` is a field
of `StepInfo` (line 23) and is populated every step (line 212), so both trainers can derive
"real death" (`reason ∈ {2,3,4}`) and "truncation" (`reason == 1`) with **no new environment
plumbing**. What is *not* available is the **value of the true next state** at a truncation step —
the rollout auto-resets the env immediately after a `done` step (`collect_trajectories`,
`select_done`, lines 185–192), so the state stored at the next timestep is a **fresh episode start**,
not the true continuation. This is the crux of Part 2's cost (see Design).

### Scope of impact (which code paths)

- **Part 1 lives entirely in `core.py`**, so it fixes the reward for **every algorithm that calls
  `jax_step`** — RecurrentPPO (the six live runs), the non-recurrent PPO trainer, **and DreamerV3**.
- **Part 2 is trainer-specific.** This plan specs it for `recurrent_ppo_trainer.py` (the live-run
  path). `ppo_trainer.py` has the identical GAE pattern (`compute_gae`, lines 39–45) and DreamerV3
  uses a separate critic/continue-predictor path — both would need their own Part-2 follow-up,
  flagged but **out of scope here**.

### Hypothesis (label: HYPOTHESIS, not established) — DreamerV3 difficulty

DreamerV3 learns a **world model** and a **critic** by regressing observed rewards. Because Part 1's
bug fires a large −100 at the end of essentially every episode (including successful survivals),
Dreamer's reward predictor and critic must fit a huge, outcome-independent terminal spike. It is
**plausible but unproven** that this contributed to the DreamerV3 training difficulty seen on this
project. This plan does not test the hypothesis; it records it so a future analysis can. Fixing
Part 1 removes the spurious spike for Dreamer automatically (shared `jax_step`).

## Implementation Plan

### Design

**Part 1 — death penalty on real death only (PRIMARY, the explicit ask).**
Capture `update_body`'s `done` (real death) **before** it is merged with `truncated` on line 706,
into a new local `real_death`. Gate the `death_penalty` on `real_death` instead of the
timeout-inclusive `done`. Leave the merged `done` untouched so the episode still ends and resets on
both death and timeout. This is a small, self-contained, low-risk change with no config-schema
impact. On a timeout step the terminal reward becomes just the ordinary homeostatic step value
(≈ −0.19), matching the ground-truth target.

**Part 2 — correct the value bootstrap on truncation (companion, distinct scope).**
Distinguish "real termination" (bootstrap zeroed) from "truncation" (bootstrap retained) in GAE.
Two things are needed:

1. **The mask** — derive `terminated` (`reason ∈ {2,3,4}`) and use it for the delta bootstrap
   `(1 - terminated)`, keeping the merged `done` for the GAE accumulation reset `(1 - done)` (an
   episode boundary still resets advantage accumulation on both death and timeout). This is cheap
   and derivable from the already-carried `termination_reason`.
2. **The bootstrap value** — the harder part. Correct truncation bootstrapping needs `V(s')` of the
   **true** next state, but auto-reset overwrites it with a fresh episode start. The clean fix is to
   compute `V(next_state)` in the rollout **before** the reset and store it as a new `Transition`
   field (`next_value`), then feed those true next-values into GAE for every step. This adds one
   value-head forward per step in the rollout hot loop.

**Recommendation (see "Senior-developer recommendation" at the end):** approve and ship **Part 1
now**; treat **Part 2** as a distinct, benchmarked follow-up because (a) it touches the rollout hot
loop and will measurably change per-iteration speed, and (b) it changes value targets for all
future runs and so needs its own regression + speed sign-off. Part 1 alone removes the dominant
−100 corruption. The full Part 2 spec is included below so "both" can be approved in one go if
desired.

**Why line 824 (`terminated=done`) is left alone.** The state's `terminated` field is consumed only
by `scripts/fixtures/generate_parity_fixtures.py` (test golden snapshots) — not by any trainer. The
trainers read the 4th return value of `jax_step` (`done`) and derive death/truncation from
`termination_reason`. Changing line 824 would churn the parity fixtures for no functional gain, so
this plan only adds an explanatory comment there.

---

### File Changes — Part 1 (PRIMARY)

#### `src/environment/core.py` (lines 690, 705–706, 717–725)

Capture real death before the truncation merge, and gate the penalty on it.

```python
# BEFORE (line 690):
    new_satiation, new_nutrition, new_injury, next_injury_buffer, next_nociception_history, new_rest_streak, done = update_body(state, info, params)

# AFTER (line 690 + new comment/assignment):
    new_satiation, new_nutrition, new_injury, next_injury_buffer, next_nociception_history, new_rest_streak, done = update_body(state, info, params)
    # `done` here is REAL DEATH only (starvation / over-eating / injury). update_body does not know
    # about the step clock, so it never fires on a timeout. Capture it BEFORE the truncation merge
    # below so the death_penalty can be gated on real death and NOT on surviving to the step limit.
    # Finding B — see docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md and
    # docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md.
    real_death = done
```

```python
# BEFORE (lines 705–706):
    info['termination_reason'] = reason
    done = jnp.logical_or(done, truncated)

# AFTER (lines 705–706 + new comment):
    info['termination_reason'] = reason
    # `done` below is the EPISODE-END flag: real death OR timeout. It is used ONLY for episode reset,
    # hidden-state reset, and boundary bookkeeping. It must NOT gate the death_penalty — surviving to
    # max_steps (truncation, reason == 1) is the SUCCESS outcome of a survival task and must not be
    # punished like death. The penalty is gated on `real_death` (captured above). Finding B.
    done = jnp.logical_or(done, truncated)
```

```python
# BEFORE (lines 717–725):
    if params.use_homeostatic_reward:
        prev_drive = calculate_drive(state.satiation, state.injury_level, params)
        curr_drive = calculate_drive(new_satiation, new_injury, params)
        reward_homeostatic = prev_drive - curr_drive
        # Death penalty based on Nutrition starvation
        reward_homeostatic = jnp.where(done, reward_homeostatic - params.death_penalty, reward_homeostatic)
    else:
        reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
        reward_extrinsic = jnp.where(done, -params.death_penalty, reward_extrinsic)

# AFTER (lines 717–725):
    if params.use_homeostatic_reward:
        prev_drive = calculate_drive(state.satiation, state.injury_level, params)
        curr_drive = calculate_drive(new_satiation, new_injury, params)
        reward_homeostatic = prev_drive - curr_drive
        # Death penalty gated on REAL DEATH only (starvation / over-eating / injury), NOT on `done`.
        # Timeout / truncation (reason == 1) keeps just the normal homeostatic step value. Finding B.
        reward_homeostatic = jnp.where(real_death, reward_homeostatic - params.death_penalty, reward_homeostatic)
    else:
        reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
        # Death penalty gated on REAL DEATH only — NOT on timeout. Finding B.
        reward_extrinsic = jnp.where(real_death, -params.death_penalty, reward_extrinsic)
```

**No config-schema change.** `params.death_penalty` is still read; only its gate changes. No new
YAML keys, no `config.get_mandatory` additions.

#### `src/models/recurrent_ppo_trainer.py` (`compute_gae`, ~line 61–65) — comment only (Part 1)

Even under Part-1-only, add a durable comment at the bootstrap site recording the **known,
deferred** truncation-bootstrap limitation, so a future reader does not mistake it for correct or
silently reintroduce a conflation. No logic change under Part 1.

```python
# BEFORE (line 63):
        delta = reward + gamma * next_value * (1 - done) - value

# AFTER (add comment above line 63; logic unchanged under Part 1):
        # KNOWN LIMITATION (Finding B, Part 2 deferred): `done` here is real-death OR timeout, so the
        # (1 - done) bootstrap is zeroed on TIMEOUT too. Correct RL truncation handling would zero the
        # bootstrap only on real death and RETAIN gamma*V(s') on timeout. See
        # docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md (Part 2).
        delta = reward + gamma * next_value * (1 - done) - value
```

---

### File Changes — Part 2 (COMPANION, distinct scope — approve separately)

All in `src/models/recurrent_ppo_trainer.py`. If Part 2 is approved, the Part-1 comment above is
**replaced** by the real fix below.

#### 1. `Transition` struct (lines 28–38) — add a `next_value` field

```python
# ADD to Transition (after `value`):
    next_value: jnp.ndarray = None  # V(true next state) computed BEFORE auto-reset — for truncation bootstrap
```

#### 2. `collect_trajectories` `scan_fn` (after line 177, before the auto-reset at ~180) — compute and store `V(next_state)` pre-reset

```python
# ADD after `next_state, reward, done, info = jax.vmap(jax_step, ...)` (line 177):
        # Bootstrap value for correct truncation handling: value of the TRUE next state, computed
        # BEFORE auto-reset overwrites it with a fresh episode start. Uses h_new (post-step hidden).
        with jax.named_scope("rppo_bootstrap_value"):
            obs_next_true = jax.vmap(get_observation, in_axes=(0, None))(next_state, env_params)
            _, next_value, _, _, _ = jax.vmap(
                get_action_and_value_nnx, in_axes=(None, 0, h_axes, 0)
            )(model, obs_next_true, h_new, act_keys)
```

Add `next_value=next_value` to the `Transition(...)` constructor (line 217).

> Cheaper alternative to weigh: store `obs_next_true` + `h_new` in the transition and run ONE
> batched value forward over all `(T, num_envs)` steps after the scan. Same compute, heavier rollout
> memory. `developer` should pick whichever benchmarks better and note it in the report.

#### 3. `compute_gae` (lines 50–73) — add a `terminateds` argument; use it for the delta bootstrap, keep `dones` for the accumulation reset

```python
# BEFORE:
def compute_gae(rewards, values, values_next, dones, gamma, lmbda):
    def gae_scan(gae, x):
        reward, value, next_value, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lmbda * (1 - done) * gae
        return gae, gae
    _, advantages = jax.lax.scan(gae_scan, 0.0, (rewards, values, values_next, dones), reverse=True)
    return advantages

# AFTER:
def compute_gae(rewards, values, values_next, dones, terminateds, gamma, lmbda):
    # RL-correct truncation handling (Finding B, Part 2):
    #  - delta bootstrap gated on `terminated` (REAL death, reason in {2,3,4}) — retained on timeout.
    #  - accumulation reset gated on `done` (death OR timeout) — episode boundary still cuts the
    #    advantage chain on both.
    # `values_next` MUST be V(true next state) (Transition.next_value), NOT the auto-reset value.
    def gae_scan(gae, x):
        reward, value, next_value, done, terminated = x
        delta = reward + gamma * next_value * (1 - terminated) - value
        gae = delta + gamma * lmbda * (1 - done) * gae
        return gae, gae
    _, advantages = jax.lax.scan(
        gae_scan, 0.0, (rewards, values, values_next, dones, terminateds), reverse=True
    )
    return advantages
```

#### 4. `train_iteration` GAE branch (lines 292–303) — derive `terminated`, feed true next-values

```python
# BEFORE (GAE branch, lines 293–301):
            obs_final = jax.vmap(get_observation, in_axes=(0, None))(next_env_state, env_params)
            _, final_v, _, _ = jax.vmap(model)(obs_final, next_h_state)
            values_with_next = jnp.concatenate([trajectories.value, final_v.reshape(1, -1)], axis=0)
            advantages = jax.vmap(compute_gae, in_axes=(1, 1, 1, 1, None, None), out_axes=1)(
                trajectories.reward, trajectories.value, values_with_next[1:], trajectories.done, config.gamma, config.gae_lambda
            )

# AFTER:
            # Real-termination mask (Finding B): reason 2/3/4 = death; reason 1 = timeout; 0 = active.
            terminateds = (trajectories.step_info.termination_reason >= 2).astype(trajectories.done.dtype)
            # values_next is now the TRUE next-state value stored pre-reset (no auto-reset value, no
            # concatenate hack): Transition.next_value already holds V(s') for every step.
            advantages = jax.vmap(compute_gae, in_axes=(1, 1, 1, 1, 1, None, None), out_axes=1)(
                trajectories.reward, trajectories.value, trajectories.next_value,
                trajectories.done, terminateds, config.gamma, config.gae_lambda
            )
```

> Note: `trajectories.next_value` supersedes the `values_with_next[1:]` slice and the final-value
> forward (`obs_final` / `final_v`), which can be removed under Part 2. `developer` should confirm
> nothing else consumes `final_v` before deleting it.

---

### Regression test

Create **`tests/env/test_truncation_not_death.py`** (new file; env tests live under `tests/env/`).
This test must **fail on the current code and pass after the fix** — that is what proves the fix.

**Part 1 — environment reward gating (always required):**

- `test_timeout_no_death_penalty`: build resolved `EnvParams` with `use_homeostatic_reward=True`,
  a known `death_penalty` (e.g. 100) and a small `max_steps`; drive `jax_step` to the max-step
  timeout. Assert: (a) `info['termination_reason'] == 1`, and (b) the final `reward` does **not**
  include the penalty — e.g. `reward > -1.0` (its magnitude is the ~0.2 homeostatic step value),
  and equivalently `reward > -params.death_penalty / 2`. On the current code this fails (reward
  ≈ −100).
- `test_starvation_applies_death_penalty`: drive to starvation. Assert `termination_reason == 2`
  and `reward < -params.death_penalty / 2` (penalty present). Guards against over-correction.
- (If `overeating_death` / injury paths are cheap to reach) add analogous checks that `reason == 3`
  and `reason == 4` still apply the penalty.

**Part 2 — GAE truncation bootstrap (only if Part 2 is approved):**

Add **`tests/models/test_gae_truncation.py`** (unit test on `compute_gae`, no env needed):

- Construct a short sequence with a step where `done = True` **and** `terminated = False`
  (a truncation) followed by a non-terminal step. Assert the advantage/target at the truncation
  step **includes** the `gamma * next_value` bootstrap (compare against the hand-computed value),
  whereas a step with `terminated = True` (real death) **excludes** it. This pins the death-vs-
  truncation asymmetry.

---

### Documentation-context additions

1. **Diagnosis doc** (`docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md`,
   Finding-B section / E2E-1): append a short "Fix landed" note once merged, recording (a) the
   mechanism, (b) the ground-truth numbers (already present), (c) the fix (this plan, linked), and
   (d) — **clearly labelled HYPOTHESIS, not established** — that the spurious per-episode −100 may
   have contributed to DreamerV3 training difficulty (its critic/world-model must fit a large,
   outcome-independent terminal spike on every episode end, including timeouts). Frame per the
   project's plain-language documentation rule (English first, symbols after).
2. **Reward-regime break point** (backward-compat): once the fix commits, record the **commit hash
   and date** in the diagnosis doc's Finding-B section as the "reward-regime break point", so future
   analysis knows that post-fix absolute reward curves are not comparable to historical runs.
   `developer` fills this in the Implementation Report and mirrors it into the diagnosis doc.

### Backward-compat / comparability

- **Survival-step metric is unaffected.** Performance is measured in survival steps, not cumulative
  reward (per CLAUDE.md), so the headline metric definition does not change.
- **Learned reward changes**, so post-fix runs' **absolute reward curves** and learned behaviour are
  **not directly comparable** to historical runs. This is a deliberate, one-time reward-regime break;
  name the commit/date (above) as the boundary.
- **No config-schema change.** `death_penalty` key unchanged; `params.death_penalty` still consumed,
  just gated on real death.
- **Live-run decision (portfolio call, for the user / `pi`):** the six currently-training runs are
  distorted but *comparably* so (the bug predates `main`). Whether to restart them now vs. let them
  finish and fix before the next wave is a user call — a candidate for a `pi` consult before the
  next launch. This plan does not decide it.

### Risk assessment

| Part | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Over-correction — dropping the penalty on a step that is *both* death and timeout | Low | `real_death` = `update_body`'s pre-merge `done`, which is exactly reason ∈ {2,3,4}; death takes precedence in the reason logic, so a coincident death+timeout step correctly keeps the penalty. Regression test pins both directions. |
| 1 | JIT recompile / shape change | None | `real_death` is a scalar bool already in scope; no shape or static-field change. |
| 2 | **Speed** — extra value-head forward per rollout step (roughly doubles the network forward in collect) | **Medium–High** | `developer` MUST benchmark s/it before vs after on the same node/config/seed. Per the Verification Protocol: >5% slowdown warrants discussion, >15% is a blocker unless explicitly accepted. Weigh the cheaper batched-post-scan variant. |
| 2 | Value-target change for all future runs | Medium | This is the RL-correct behaviour, but it is a second reward-regime-adjacent change; ship with the GAE unit test and its own speed sign-off. |
| 2 | `ppo_trainer.py` and DreamerV3 still have the old bootstrap | Low (scope) | Flagged as out-of-scope follow-ups; Part 1 already fixes their *reward*. |

## Checkpoints

Part 1:
- [x] After the `core.py` edit, drive a timeout episode (reuse `tmp/20260704_finding_b_driver.py`
      logic) and confirm the final reward is ≈ −0.19 (not −100), `termination_reason == 1`. —
      confirmed via smoke test: seed=0, `max_steps=3`, reward = −0.1864, reason=1.
- [x] Drive a starvation episode and confirm the penalty is still applied (reward ≈ −100),
      `termination_reason == 2`. — confirmed: reward = −100.66, reason=2 (matches pre-fix ground
      truth exactly, since starvation deaths were never affected by the bug).
- [x] `test_truncation_not_death.py` fails on the pre-edit code, passes after. — confirmed: pre-fix
      `test_timeout_no_death_penalty` FAILED with reward=−100.18638 (matches the diagnosis doc's
      ground-truth number exactly); post-fix both tests PASS.
- [x] Existing env parity tests (`tests/env/test_unified_parity.py`) still pass — Part 1 changes
      reward, not state/positions, so golden snapshots of state must be unaffected. — confirmed:
      34 passed, 244 skipped (data-driven skips for missing configs/fixtures, pre-existing), 0
      failed. The parity fixtures do not store `reward`, only state/positions, so no fixture
      regeneration was needed or performed.

Part 2 (only if approved):
- [ ] `test_gae_truncation.py` pins bootstrap-retained-on-truncation, zeroed-on-death.
- [ ] Speed benchmark s/it recorded before/after on the same node/config/seed; delta reported.
- [ ] A short real training run produces finite advantages/targets (no NaN) and sane value loss.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-04

### Scope

**Part 1 ONLY**, exactly as scoped. Part 2 (GAE truncation-bootstrap change in
`recurrent_ppo_trainer.py` — `Transition.next_value`, `compute_gae` signature, `train_iteration` GAE
branch) was **NOT touched** — no logic changes anywhere in the Part-2 file-changes list. Confirmed by
inspecting the final diff (below): only a comment was added at the `compute_gae` bootstrap site.

### What was implemented, file-by-file

**`src/environment/core.py`** (`jax_step`) — three edits, all comment+logic, no shape/schema change:
1. After the `update_body(...)` call (was line 690), captured `real_death = done` with an explanatory
   comment — `done` at that point is real death only (starvation/over-eating/injury); `update_body`
   knows nothing about the step clock.
2. At the `done = jnp.logical_or(done, truncated)` merge (was lines 705–706), kept the merge as-is
   and added a durable comment explaining that the merged `done` now means "episode ended, for any
   reason" and must NOT gate the death penalty.
3. At the two reward-penalty sites (was lines 722 and 725), changed the gate from `done` to
   `real_death` in both the homeostatic branch (`reward_homeostatic = jnp.where(real_death, ...)`) and
   the extrinsic branch (`reward_extrinsic = jnp.where(real_death, ...)`), each with a one-line
   comment. No config-schema change — `params.death_penalty` is still consumed, just gated on real
   death instead of the timeout-inclusive `done`.

Full before/after diff:

```diff
@@ -688,7 +688,13 @@ def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState,
     }
     
     new_satiation, new_nutrition, new_injury, next_injury_buffer, next_nociception_history, new_rest_streak, done = update_body(state, info, params)
-    
+    # `done` here is REAL DEATH only (starvation / over-eating / injury). update_body does not know
+    # about the step clock, so it never fires on a timeout. Capture it BEFORE the truncation merge
+    # below so the death_penalty can be gated on real death and NOT on surviving to the step limit.
+    # Finding B — see docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md and
+    # docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md.
+    real_death = done
+
     # Max Steps Truncation
     next_step = state.current_step + 1
     truncated = next_step >= params.max_steps
@@ -703,8 +709,12 @@ def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState,
     reason = jnp.where(new_injury >= params.max_injury, 4, reason)
     
     info['termination_reason'] = reason
+    # `done` below is the EPISODE-END flag: real death OR timeout. It is used ONLY for episode reset,
+    # hidden-state reset, and boundary bookkeeping. It must NOT gate the death_penalty — surviving to
+    # max_steps (truncation, reason == 1) is the SUCCESS outcome of a survival task and must not be
+    # punished like death. The penalty is gated on `real_death` (captured above). Finding B.
     done = jnp.logical_or(done, truncated)
-    
+
     # 6. Reward (Homeostatic driven by Satiation)
     reward_homeostatic = 0.0
     reward_extrinsic = 0.0
@@ -718,11 +728,13 @@ def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState,
         prev_drive = calculate_drive(state.satiation, state.injury_level, params)
         curr_drive = calculate_drive(new_satiation, new_injury, params)
         reward_homeostatic = prev_drive - curr_drive
-        # Death penalty based on Nutrition starvation
-        reward_homeostatic = jnp.where(done, reward_homeostatic - params.death_penalty, reward_homeostatic)
+        # Death penalty gated on REAL DEATH only (starvation / over-eating / injury), NOT on `done`.
+        # Timeout / truncation (reason == 1) keeps just the normal homeostatic step value. Finding B.
+        reward_homeostatic = jnp.where(real_death, reward_homeostatic - params.death_penalty, reward_homeostatic)
     else:
         reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
-        reward_extrinsic = jnp.where(done, -params.death_penalty, reward_extrinsic)
+        # Death penalty gated on REAL DEATH only — NOT on timeout. Finding B.
+        reward_extrinsic = jnp.where(real_death, -params.death_penalty, reward_extrinsic)
```

This is a byte-exact match to the plan's specified before/after.

**`src/models/recurrent_ppo_trainer.py`** (`compute_gae`) — comment only, per Part-1 scope:
```diff
     def gae_scan(gae, x):
         reward, value, next_value, done = x
+        # KNOWN LIMITATION (Finding B, Part 2 deferred): `done` here is real-death OR timeout, so the
+        # (1 - done) bootstrap is zeroed on TIMEOUT too. Correct RL truncation handling would zero the
+        # bootstrap only on real death and RETAIN gamma*V(s') on timeout. See
+        # docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md (Part 2).
         delta = reward + gamma * next_value * (1 - done) - value
```
No logic change — confirmed no other lines in this file were touched.

**`tests/env/test_truncation_not_death.py`** (new file) — two tests, following the project's
`_run_episode`-driver pattern from `tmp/20260704_finding_b_driver.py`, against `basic/06`'s resolved
`EnvParams`:
- `test_timeout_no_death_penalty`: overrides `max_steps=3`, drives seeds until a clean timeout episode
  (`termination_reason == 1`), asserts the final reward is `> -1.0` and `> -death_penalty/2`.
- `test_starvation_applies_death_penalty`: drives a long-horizon wander episode to starvation
  (`termination_reason == 2`), asserts the final reward is `< -death_penalty/2` (guards against
  over-correction — the fix must not remove the penalty from real deaths too).

### Regression test — pre-fix / post-fix evidence

**Pre-fix** (before the `core.py` edit was applied): `test_timeout_no_death_penalty` **FAILED**:
```
AssertionError: Timeout step reward = -100.18638; expected ~ -0.2 (no death penalty).
death_penalty = 100.0 appears to have been applied on timeout.
assert Array(-100.18638, dtype=float32) > -1.0
```
This number (−100.18638) matches the diagnosis doc's ground-truth figure (−100.186) exactly, confirming
the test reproduces the documented bug, not a different issue. `test_starvation_applies_death_penalty`
passed both before and after (starvation deaths were never affected by the bug — correct, expected).

**Post-fix**: both tests PASS.
```
tests/env/test_truncation_not_death.py::test_timeout_no_death_penalty PASSED
tests/env/test_truncation_not_death.py::test_starvation_applies_death_penalty PASSED
2 passed in 49.27s
```

### Full test-suite results

- `pytest tests/env/` (686 collected): **194 passed, 492 skipped, 0 failed**, 748.81s. Skips are the
  pre-existing data-driven `pytest.skip(...)` calls for missing configs/fixtures scattered across the
  parametrized test files (`test_unified_parity.py`, `test_visual_parity.py`, etc.) — not new skips
  introduced by this change.
- `pytest tests/env/test_unified_parity.py -v` (isolated, to specifically confirm no parity
  regression): **34 passed, 244 skipped, 0 failed**, 287.92s. Inspected the parity test's comparison
  logic (`np.testing.assert_allclose` calls) — it compares `agent_pos` and other state/position
  fields, **not** `reward`, so this change (reward-gating only) was never going to perturb it; the
  pass confirms state/positions are byte-identical, as expected for a reward-only change.
- No `.npz` parity fixtures were regenerated or altered by this work. The pre-existing modified/
  untracked fixture churn noted in the task (animal_* rename) was left untouched — not staged, not
  committed.

### Smoke test

```
seed=0, max_steps=3: t=0 reward=-0.7637 done=False reason=0
                      t=1 reward=-0.1864 done=True  reason=1 (TIMEOUT)
SMOKE OK — no crash, forced-timeout reward = -0.1864 (not -100)
```
Confirmed no crash across reset+step and that a forced-timeout step shows the ordinary homeostatic
step value, not the death penalty.

### Speed check

**Skipped — comment-only + scalar-gate change, provably does not affect the hot path.** The `core.py`
edit changes which JAX array (`real_death` vs `done`, both scalars already in scope, same dtype/shape)
gates a `jnp.where`; it is not a new operation, branch, or shape change, so it cannot measurably affect
per-step compute or JIT recompilation. The `recurrent_ppo_trainer.py` edit is a comment only. No
before/after s/it benchmark was run.

### Deviations from the plan

None. All three `core.py` edits and the trainer comment match the plan's specified before/after
byte-for-byte. No file outside the plan's File Changes section (Part 1) was touched.

### Reward-regime break point (for the diagnosis doc)

Fix commit: `ef0fd25` (`fix(reward): 🐛 death penalty on real death only, not max_steps timeout`),
2026-07-04. Mirrored into `docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md`'s
Finding-B section (one-line note in the callout at the top of that section): runs trained before this
commit reward surviving-to-`max_steps` with the full `death_penalty`; runs trained after do not.

### Dreamer-difficulty hypothesis label check

Confirmed: the plan doc's `### Hypothesis (label: HYPOTHESIS, not established) — DreamerV3 difficulty`
section already states plainly that this is "plausible but unproven" and "This plan does not test the
hypothesis; it records it so a future analysis can." No diagnosis-doc edit was needed beyond the
one-line reward-regime break point (the diagnosis doc did not previously carry the Dreamer-difficulty
hypothesis text at all — it lives only in this plan doc, correctly labeled).

### Blockers / follow-ups

None for Part 1. Part 2 (GAE truncation bootstrap) remains a distinct, unapproved follow-up per the
plan's recommendation — flagged here for `senior-developer`/user to decide on separately, not
implemented, not started.

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]

---

<!--
NEW ISSUES: If a new issue is discovered during implementation/verification:
- If closely related: append as "## Issue #2: [title]" below with the same template sections.
- If independent: create a separate document and cross-reference.
-->
