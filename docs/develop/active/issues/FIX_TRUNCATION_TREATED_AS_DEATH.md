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

Part 2 (approved and implemented 2026-07-04):
- [x] `test_gae_truncation.py` pins bootstrap-retained-on-truncation, zeroed-on-death. — confirmed:
      fails pre-fix with `TypeError` (old 6-arg `compute_gae` signature) AND semantically (a
      hand-computed check against the pre-fix `compute_gae` on a timeout step returns 1.0 —
      the death-case number — instead of the correct 5.95); 5/5 pass post-fix.
- [x] Speed benchmark s/it recorded before/after on the same node/config/seed; delta reported. —
      confirmed: 0.00% delta on `basic/05` + default (MC-mode) `recurrent_ppo.yaml`, the config
      the task's speed gate specifies — see Implementation Report for the MC-vs-GAE gating that
      makes this true, plus a supplementary GAE-mode-only measurement (+4–9%, informative only,
      not gating since no live config uses GAE mode).
- [x] A short real training run produces finite advantages/targets (no NaN) and sane value loss. —
      confirmed in both MC mode (Loss values ~0.04–0.36, no NaN) and GAE mode (Loss values
      ~75–91, no NaN) — see Implementation Report.

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

None for Part 1. Part 2 (GAE truncation bootstrap) is now implemented — see the Part 2
Implementation Report below.

---

## Part 2 Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-04

### Plain-language summary

This is the second half of the Finding-B fix. Part 1 (already committed, `ef0fd25`) stopped the
environment from punishing "survived to the time limit" the same as "died" in the *reward*. Part 2
fixes the matching bug on the *learning* side: when an episode is cut off by the clock (a
"truncation"), the trainer's value-estimator (the "critic", which predicts how much future reward
is still coming) must keep counting the future it didn't get to see — not throw it away as if the
world had truly ended. Before this fix, the trainer's `compute_gae` function zeroed that future
estimate on BOTH real death and timeout, because it only ever saw a single merged `done` flag.
After this fix, it consults `termination_reason` (already carried into the trainer) to tell the two
apart: real death (reason 2/3/4) still zeroes the future estimate; timeout (reason 1) now retains
it, matching standard reinforcement-learning practice.

**Important finding not anticipated by the plan**: every currently-live `recurrent_ppo` config sets
`return_mode: "MC"` (Monte Carlo returns), not `"GAE"` — `compute_gae` (and therefore this whole
fix) is **not on the code path any live training run actually uses**. See "Deviation: MC-mode
gating" below for how this was handled and why it matters for the speed gate.

### What was implemented, file-by-file

**`src/models/recurrent_ppo_trainer.py`** — the primary Part-2 file, exactly as scoped:

1. **`Transition` struct**: added `next_value: jnp.ndarray = None` — V(true next state), computed
   pre-auto-reset (additive field, matches plan).
2. **`compute_gae`**: added a `terminateds` parameter. The delta bootstrap
   (`gamma * next_value * (1 - terminated)`) is now gated on `terminated` (real death only,
   RETAINED on truncation); the GAE accumulation-reset term (`gamma * lmbda * (1 - done) * gae`)
   is unchanged, still gated on the merged `done` (death OR timeout — an episode boundary still
   cuts the advantage chain either way, since a new episode starts on both).
3. **`collect_trajectories`**: after `jax_step` and before the auto-reset overwrites `next_state`,
   computes `V(true next state)` using `h_new` (this step's post-forward hidden state) and stores
   it as `Transition.next_value`. **Gated on `return_mode == "GAE"`** (see deviation below) — a
   plain Python `if` resolved at JIT trace time (zero runtime branch cost either way), since
   `return_mode` comes from `config`, which is itself a static `jit` argument
   (`nnx.jit(train_iteration, static_argnums=(6,))` in `train.py`).
4. **`train_iteration`** (GAE branch): derives `terminateds = (termination_reason >= 2)`, passes
   `trajectories.next_value` directly to `compute_gae` as `values_next`. This **supersedes and
   removes** the old `obs_final` / `final_v` / `values_with_next` concatenate-and-shift logic —
   `trajectories.next_value` already holds the correct value for every step, including the last,
   so no separate post-rollout forward pass is needed. Also removed the now-unused
   `from src.environment.sensor import get_observation` import at the top of `train_iteration`
   (still imported separately inside `collect_trajectories`, which needs its own copy).

**`src/models/ppo_trainer.py`** and **`src/models/dreamer_v3_trainer.py`** — one-line comment only,
per this session's explicit task delegation (not the plan doc's Part-2 File Changes list, which
scoped Part 2 to `recurrent_ppo_trainer.py` only — flagged here as directed, not a silent
expansion): both files share the same `done`-conflates-timeout-and-death pattern in their own
GAE/continue-predictor code and were left functionally untouched, with a comment marking the
deferred follow-up and linking this doc. `ppo_trainer.py`'s `compute_gae` docstring and
`dreamer_v3_trainer.py`'s continue-loss computation (`cont_target = 1.0 - terminal[..., None]`,
line ~229) were the two sites. No logic changed in either file.

### Deviation: MC-mode gating (not in the plan's literal snippet)

The plan's `collect_trajectories` snippet computes `next_value` **unconditionally**, every step,
regardless of `return_mode`. Investigating the actual config landscape before benchmarking
surfaced that **`return_mode: "MC"` is the default in `recurrent_ppo.yaml` and every other live
`recurrent_ppo_*.yaml` config** (checked via `grep -rn "return_mode" configs/models/recurrent_ppo/`
— 13 files, all `"MC"`, zero `"GAE"`). `compute_gae` (and its new `next_value` bootstrap) is only
consumed by the GAE branch of `train_iteration`'s "Compute Advantages and Targets" step — the MC
branch calls `compute_mc_returns`, which never reads `next_value`. Implementing the plan literally
would have added a full extra value-head forward pass, every rollout step, to **every currently
training rPPO run**, for a correctness benefit that mode never uses.

I gated the extra forward pass on `return_mode.upper() == "GAE"` (a static, trace-time Python `if`,
zero runtime cost either way — see point 3 above). Under MC mode, `next_value` is set to
`jnp.zeros_like(value)` (a cheap allocation, not a matmul) purely to keep the `Transition`
PyTree's leaf structure consistent for `jax.lax.scan`'s stacking. This deviates from the plan's
literal code (which didn't anticipate the MC/GAE split) but preserves 100% of its intent — Part 2's
correctness fix is fully present and correct whenever GAE mode is used — while eliminating the
speed cost for the mode actually in use today. Flagged here for `senior-developer` to confirm this
reading of the plan's intent is correct.

Also fixed a **transcription bug in the plan's own Part-2 code snippet**: the plan's
`get_action_and_value_nnx` unpacking example (`_, next_value, _, _, _ = ...`) would have assigned
`log_prob` (return position 1) to `next_value`, not `value` (return position 2) —
`get_action_and_value_nnx` returns `(action, log_prob, value, h_new, mod_info)`. I used a direct
`model(obs, h)` call instead (mirroring the existing `final_v` pattern this fix removes), which
avoids the bug entirely and also avoids an unnecessary categorical action sample on a step whose
action is never used.

### Regression test — fail-before / pass-after evidence

New file: **`tests/models/test_gae_truncation.py`** (5 tests, unit-level on `compute_gae`, no env
needed, per the plan's Part-2 test spec).

**Pre-fix** (Part-1-only code, via `git stash` isolating `recurrent_ppo_trainer.py` — confirmed
0-line diff against the stash before dropping it, so this is a faithful revert/restore, not an
approximation):
```
FAILED tests/models/test_gae_truncation.py::test_bootstrap_retained_on_truncation - TypeError: compute_gae() takes 6 positional arguments but 7 were given
FAILED tests/models/test_gae_truncation.py::test_bootstrap_zeroed_on_real_death - TypeError: ...
FAILED tests/models/test_gae_truncation.py::test_truncation_bootstrap_does_not_leak_across_episode_boundary - TypeError: ...
FAILED tests/models/test_gae_truncation.py::test_death_bootstrap_does_not_leak_across_episode_boundary - TypeError: ...
4 failed, 1 passed in 1.84s   # the 5th test (termination_reason mask formula) doesn't call compute_gae
```
The `TypeError`s alone prove old code lacks the death/truncation distinction, but to rule out "the
test just doesn't match the old API" as opposed to "the old code has the bug", I also ran the old
6-arg `compute_gae` directly against the truncation scenario:
```
OLD compute_gae on a timeout step (pre-fix): 1.0
Expected CORRECT (bootstrap retained):        5.95
Bug confirmed: True
```
This is the death-case number (1.0), not the truncation-case number (5.95) — i.e. old code
concretely conflates timeout with death, not just "doesn't have the parameter".

**Post-fix**: all 5 tests pass:
```
tests/models/test_gae_truncation.py::test_bootstrap_retained_on_truncation PASSED
tests/models/test_gae_truncation.py::test_bootstrap_zeroed_on_real_death PASSED
tests/models/test_gae_truncation.py::test_truncation_bootstrap_does_not_leak_across_episode_boundary PASSED
tests/models/test_gae_truncation.py::test_death_bootstrap_does_not_leak_across_episode_boundary PASSED
tests/models/test_gae_truncation.py::test_termination_reason_to_terminated_mask PASSED
5 passed in 2.41s
```

### Full test-suite results

- `pytest tests/models/` (5 collected, all new): **5 passed, 0 failed**, 2.2s.
- `pytest tests/env/` (686 collected): **194 passed, 492 skipped, 0 failed**, 735.11s (0:12:15) —
  identical pass/skip counts to Part 1's run, confirming Part 2 introduces no env-level regression
  (expected: Part 2 touches only trainer-side value/advantage computation, never environment state
  or reward).

### Speed benchmark (the mandatory gate)

**Setup**: `basic/05` (`configs/environment/experiment/basic/05-random_init_10x10.yaml`) +
non-modulated `recurrent_ppo.yaml` (default config, `return_mode: "MC"`), seed 0, identical
`--total-timesteps 1638400` (100 iterations at the config's default 128 envs × 128 steps/iter),
same GPU (`CUDA_VISIBLE_DEVICES=0`, local RTX 4090), same machine, sequential (not concurrent) runs.
Timing method: parsed tqdm's per-iteration elapsed-time stamps from the training log (`Training: 0it
[MM:SS, ..., Iter=N, ...]`), comparing iteration 10 → iteration 100 (skips early iterations so
one-time JIT-compile overhead — visible as a ~60s jump before iteration 1 — is excluded from the
steady-state measurement) to get s/it.

| Run | Command | Steady-state s/it (iter 10→100) |
|---|---|---|
| BEFORE (Part-1-only, stashed) | `tmp/20260704_speed_before.log` | 0.2667 s/it |
| AFTER (Part 1 + Part 2) | `tmp/20260704_speed_after.log` | 0.2667 s/it |

**Delta: 0.00%.** Cross-checked with several different warm-up cutoffs (k0 = 10, 20, 30, 50 —
`before`/`after` re-derived from the same two logs at each cutoff): deltas ranged −4.55% to +0.00%,
i.e. noise-floor level (±1s second-resolution rounding on a ~24s window), no consistent direction.
This result is a direct consequence of the MC-mode gating above: with `return_mode: "MC"` (every
live config), the new bootstrap-value forward pass never executes — the only change on this code
path is a cheap `jnp.zeros_like(value)` allocation for `Transition.next_value`, and removing the
old `obs_final`/`final_v` computation (which was itself GAE-branch-only, so also never ran in MC
mode) is a wash. **Verdict: PASS the speed gate cleanly** (<5% threshold, in fact indistinguishable
from zero).

**Supplementary (informative, non-gating) measurement — GAE mode**: since GAE mode is where Part
2's correctness fix actually changes behavior, I also ran a smaller before/after comparison with a
temporary `return_mode: "GAE"` copy of the agent config (`tmp/20260704_recurrent_ppo_gae.yaml`,
scratch-only, not committed), `num_envs=64`/`num_steps=64` (smaller for a faster supplementary
check), 200 iterations:

| Cutoff | BEFORE (Part-1-only GAE branch) | AFTER (Part 2 GAE branch) | Delta |
|---|---|---|---|
| iter 10→200 | 0.1263 s/it | 0.1316 s/it | **+4.17%** |
| iter 20→200 | 0.1222 s/it | 0.1333 s/it | **+9.09%** |

This +4–9% reflects the genuine extra value-head forward pass Part 2 adds when GAE mode is
actually exercised — consistent with the plan's own risk table ("Medium–High... roughly doubles
the network forward in collect", tempered because the forward pass is only part of the total
iteration cost, which also includes multiple PPO update epochs). This is **not part of the gate**
(no live config uses GAE mode today) but is recorded here so a future switch to GAE mode has the
real cost on record, per the plan's "5–15%: commit but flag it prominently" guidance.

### Smoke test — sane loss/value numbers, no NaN

Both extracted directly from the speed-benchmark training logs above (100/200 real training
iterations each, not a synthetic exercise):

- **MC mode** (`tmp/20260704_speed_after.log`, post-fix): `Iter=100, Loss=0.1234, Rew=-137.02` (and
  all 99 preceding iterations) — finite, no NaN/Inf, no crash, no traceback.
- **GAE mode** (`tmp/20260704_speed_gae_after.log`, post-fix, exercises the new bootstrap code path
  directly): `Iter=200, Loss=90.7972, Rew=-143.17` — finite (GAE-mode loss is on a different,
  unnormalized scale than MC, expected — not a regression signal), no NaN/Inf in the full log
  (checked via `grep -i "nan\|inf\|error\|traceback"`, zero matches beyond the word "info").

### Deviations from the plan (summary)

1. **MC-mode gating** of the extra forward pass in `collect_trajectories` (not in the plan's
   literal snippet) — see dedicated section above. Rationale: zero benefit + real cost for every
   currently-live config; implemented as a static trace-time branch, zero risk.
2. **Fixed a bug in the plan's own Part-2 code example** (tuple-unpacking off-by-one in the
   `get_action_and_value_nnx` snippet) by using a direct `model()` call instead — see above.
3. **Added one-line, comment-only follow-up flags** to `ppo_trainer.py` and `dreamer_v3_trainer.py`
   per this session's explicit task delegation, even though neither file is in the plan doc's
   Part-2 File Changes list. No logic changed in either file. Flagged here for `senior-developer`
   to fold into the plan doc's File Changes section if this is confirmed as intended scope.
4. Removed `obs_final` / `final_v` / `values_with_next` from `train_iteration`'s GAE branch, as the
   plan's own note anticipated ("`developer` should confirm nothing else consumes `final_v` before
   deleting it" — confirmed via grep, nothing else in the file references them).

### Blockers / follow-ups

None. `ppo_trainer.py` and DreamerV3's continue-predictor now carry explicit follow-up comments
(see above) but remain unimplemented — future work, not blocking this fix.

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-04
> **Scope**: Part 2 (commit `3c60f6f`). Part 1 (commit `ef0fd25`) verified at its own landing; this
> pass focuses on the value-target-changing Part 2 and re-confirms Part 1 via the unchanged env suite.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/recurrent_ppo_trainer.py` | `Transition.next_value` field; `compute_gae` gains `terminateds` arg; `collect_trajectories` computes `V(true next state)` pre-auto-reset (GAE-gated); `train_iteration` derives `terminated` mask + feeds true next-values | ✅ | Bootstrap correctly uses `(1 - terminated)` where `terminated = (termination_reason >= 2)`; accumulation reset correctly keeps `(1 - done)`. `next_value` read from `next_state` **before** the auto-reset block, with `h_new` — matches the removed `final_v` recurrence. Byte-correct against plan intent. |
| `src/models/ppo_trainer.py` | comment-only follow-up flag on `compute_gae` | ✅ | No logic change; deferred follow-up recorded. |
| `src/models/dreamer_v3_trainer.py` | comment-only follow-up flag on continue-head loss | ✅ | No logic change; DreamerV3 continue bootstrap **NOT** silently altered — confirmed. |
| `tests/models/test_gae_truncation.py` | new 5-test unit suite | ✅ | Re-ran: 5/5 pass. Fail-before claim credible (old 6-arg signature `TypeError` + semantic 1.0-vs-5.95 divergence). |
| `configs/models/recurrent_ppo/*.yaml` | (unchanged) | ✅ | Independently grep-confirmed: all 13 configs use `return_mode: "MC"` → `compute_gae` off the live path; Part 2 leaves the 6 live runs byte-identical. |
| `train.py` | (unchanged) | ✅ | Confirmed `nnx.jit(train_iteration, static_argnums=(6,))` (line 801) — `config`/`return_mode` static, so the MC/GAE branch resolves at trace time (no runtime cost). |

**Speed verdict**: ✅ no regression. 0.00% delta on the specified gate config (`basic/05` + default MC-mode
`recurrent_ppo.yaml`), same GPU/seed/step-budget, warm-up excluded. The MC/GAE gate is a JIT-static
branch, so the extra forward pass never executes in MC mode; the +4–9% measured under an explicit
GAE-mode config is informative only (no live config uses GAE) and within the plan's accepted band.

**Combined suite (cross-cutting, all four fixes together)**: `pytest tests/env/ tests/models/` →
**199 passed, 492 skipped, 0 failed** (720.64s). The 492 skips are the pre-existing fixture-presence
data-driven skips; the single warning is a pre-existing `entities:`/legacy-schema deprecation, not a
regression. The four commits (`3c60f6f`, `a3ab4cc`, `0bebe06`, `9eacf82`) do not interact badly.

**Out-of-scope changes**: none. Comment-only touches to `ppo_trainer.py` / `dreamer_v3_trainer.py`
were directed by the verification task (not silent scope creep) and change no logic. The plan doc's
Part-2 File Changes list scoped Part 2 to `recurrent_ppo_trainer.py`; the two comment flags are an
accepted, directed addition.

**Conclusion**: PASS. Part 2 is RL-correct (death vs. truncation asymmetry right, true next-state
bootstrap read pre-reset), the live MC-mode runs are provably unaffected (0.00% speed delta, GAE path
never traced), and the DreamerV3/ppo_trainer follow-ups remain open and are recorded as such in the
diagnosis doc. Signed: senior-developer.

---

<!--
NEW ISSUES: If a new issue is discovered during implementation/verification:
- If closely related: append as "## Issue #2: [title]" below with the same template sections.
- If independent: create a separate document and cross-reference.
-->
