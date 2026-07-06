---
title: "Fix plan H10 — DreamerV3 batch path in train.py corrupts online behavior metrics at episode boundaries"
topic: issues
status: active
created: 2026-07-06
last_updated: 2026-07-06
---

# Fix plan H10 — DreamerV3 batch driver mixes episodes in the online behavior measures

> **Status**: IMPLEMENTED + VERIFIED (2026-07-06 — work package WP-F; awaiting commit)
> **Opened**: 2026-07-06
> **Related**: [[07_behavior_measures]] (Finding 1 — source of record), [[00_combined_diagnosis]] (§2 row H10)

---

## Context

When `train.py` trains a DreamerV3 agent, it collects experience in whole batches of T
steps at a time, and a side channel turns those steps into the project's **online behavior
numbers** — "how often was feeding interrupted by a nearby threat" (M1), "how often did the
agent dive into a bush after a threat appeared" (M2), and "does the agent eat less when
threatened" (M5). The 2026-07-04 Fable 5 re-diagnosis found (finding H10) that this batch
driver feeds **all T steps** into the measure state machine first and only *afterwards*
closes out the episodes that ended mid-batch. Three corruptions follow whenever an episode
ends in the middle of a batch: the finished episode's numbers absorb steps from *after* its
death, the next episode loses its opening steps, and if the same environment slot finishes
twice in one batch the second episode's numbers come out empty (all-NaN) and are silently
dropped. The bug is **dormant today** — online measures are off by default since `9eacf82`
and live Dreamer work uses a different, correct driver — but any config that re-enables
them on this path gets wrong numbers with no error. The fix restructures the driver to the
already-correct pattern used by every other collection path: update the measures one step
at a time and close/reset each episode at the exact step it ends. The measures feed
*analysis*, not the training gradient — so this is a wrong-conclusions bug, not a
wrong-learning bug.

## Analysis

### Root cause

`train.py`'s DreamerV3 branch ("Site 2") violates the accumulator module's contract. The
contract (documented at `src/behavior/accumulators.py:157-160`) is that the `done_mask`
argument to `bm_step_update` is **unused** — the *caller* must call `bm_reset_env` at each
done boundary before feeding the next step. Every other driver honors this by interleaving
inside a per-step loop:

- **Reference pattern — rPPO Site 1, `train.py:1347-1415`**: for each step `t`:
  `_bm_step_update(info[t], done[t])` (line 1366) → for each env `i` done at `t`:
  `_bm_finalise_episode(i, ep_data)` (1396) → `_bm_reset_env(i)` (1415) → next `t`.
- Same interleaving at DQN (`train.py:1928/1957/1972`), DRQN (`2127/2156/2171`),
  PPO (`2301/2330/2345`), and `src/algorithms/dreamer_srl/dreamer_srl_main.py:1172` (update)
  followed by per-done finalise/reset in its done-handling block.

Site 2 instead runs the whole batch first, then handles dones:

1. `train.py:1639-1648` — `for t2 in range(T2): _bm_step_update(...)` over **all** T steps.
2. `train.py:1653-1698` — only then, per done event: `_bm_finalise_episode(i, ep_data)`
   (1681) and `_bm_reset_env(i)` (1697).

### Concrete failure trace (from [[07_behavior_measures]] Finding 1, verified against current code)

Batch of T=128; env 3's episode ends at t=17:

1. Steps t=18..127 (the **new** episode's threat exposure, eats, bush dives) land in env 3's
   still-uncleared accumulators.
2. `_bm_finalise_episode(3)` then reports the *old* episode's M1/M2/M5 over old **plus** 110
   new-episode steps. The M1/M2 aging clocks (`m1_candidate_age`, `m2_onset_age`) also tick
   across the boundary, so a pre-death candidate can resolve against post-death state.
3. `_bm_reset_env(3)` wipes everything — the new episode **loses** its first 110 steps.
4. A second done for env 3 in the same batch finalises a freshly-reset (empty) state → all
   denominators 0 → all-NaN rates → episode silently dropped from the WandB means
   (NaN-skipped by `_append_per_measure_mean`, `train.py:1051-1057`).

### Non-BM per-episode fields at Site 2 — checked, CORRECT, not in scope

I re-verified the episode-scalar handling around mid-batch dones (same conclusion as the
diagnosis doc):

- Episode reward/length segment correctly per done via `curr_start` slicing
  (`train.py:1657-1660`: `episode_returns[i] + np.sum(rew_steps[curr_start:d_idx+1, i])`,
  then `curr_start = d_idx + 1` at 1698).
- `termination_reason` is read at the done step itself (`info_steps['termination_reason'][d_idx, i]`,
  line 1668) — correct attribution.
- Behavior-event sums, dist sums, and per-tag dist sums all use the same
  `curr_start:d_idx+1` segmentation (1665-1678); leftover steps (1701-1712) and the
  no-done paths (1714-1740) carry forward correctly.

So **only the BM state machine is wrong**; the non-BM fields need no change and this plan
does not touch them. (Cosmetic note, unchanged by this fix: Site 2 appends episodes to
`iteration_episodes` env-major rather than time-major as rPPO does — irrelevant to the
logged means, which are order-independent.)

### No jit/scan involvement — code-reviewer pass NOT needed

The entire affected region operates on `transitions_np = jax.device_get(transitions)`
(numpy, `train.py:1579/1582`) with plain Python loops; the accumulator module is numpy-only.
The fix is a pure Python-loop reorder over already-materialised arrays: no `jit`, `scan`,
`vmap`, or PRNG surface. A `code-reviewer` (JAX-correctness) pass is not required;
`senior-developer` verification suffices.

### Why fix a dormant bug

`behavior_measures.enabled: false` is only a config default. The moment anyone flips it on
for a `train.py` DreamerV3 run (e.g. to cross-check the offline eval path), the numbers are
silently wrong. The fix is small, mechanically testable, and closes the last unsound driver
so all six BM call sites obey the same contract.

## Implementation Plan

### Design

**Chosen approach — extract the Site-2 BM driver into a testable shared function and make
it interleave.** Add `bm_drive_batch(...)` to `src/behavior/accumulators.py` (next to the
state machine it drives). It walks the `[T, B]` batch step-by-step; at each step it calls
`bm_step_update`, then for every env done *at that step* it calls `bm_finalise_episode` and
`bm_reset_env` before advancing to t+1 — exactly the rPPO Site-1 ordering
(`train.py:1366 → 1396 → 1415`), reusing the shared helpers exactly as the other drivers
do. It returns the finalised per-episode result dicts keyed `(t, env)`; `train.py` merges
each into the matching `ep_data` inside its existing (correct) vectorized done-handling
block, which iterates the same `done_steps` array, so the `(d_idx, i)` keys always align —
including the double-done case, where two distinct keys `(t_a, i)`, `(t_b, i)` yield two
valid finalisations.

Why extraction (vs. an inline reorder in `train.py`): the loop lives inline in `train.py`'s
main function, which cannot be imported in a test without executing the whole training
entry point. Extraction is the minimal change that makes the *production* driver directly
drivable by a regression test with synthetic arrays. This is the "extract-and-test the
shared helpers with the same call pattern" option, and `accumulators.py` already exists as
the shared home for exactly this kind of `train.py`/`dreamer_srl` common code.

**Rejected alternative**: rewriting the whole Site-2 stats block to the rPPO per-step
pattern (folding scalar accumulation into a per-t loop too). It would touch verified-correct
code, change the perf profile of the BM-disabled default path, and expand the diff for zero
behavioral gain — the scalar path already segments correctly.

**Aggregation shape is unchanged**: each `ep_data` in `iteration_episodes` receives the same
`*_raw` keys as before (they come from the same shared `bm_finalise_episode`), one dict per
done event, appended at the same place in the same order. `_bm_log_wandb` /
`_append_per_measure_mean` consume it unmodified.

**Fail-loud invariant**: after the done-handling block, assert the returned results dict has
been fully consumed — any leftover key means the BM driver and the scalar done block
disagreed about which dones exist, which must crash, not pass silently.

**One deliberate micro-change**: the old code called `_bm_finalise_episode` gated only on
`bm_enabled`, so if `agent_in_bush` were ever missing from the transitions the old path
finalised a never-updated state (all-NaN keys). The new merge is gated on
`bm_enabled and agent_in_bush_steps is not None` — matching the gate the step loop already
had — so in that degenerate case episodes simply carry no BM keys (NaN-equivalent
downstream, since `_append_per_measure_mean` skips missing keys the same as NaN).

### File Changes

#### 1. `src/behavior/accumulators.py` (append after `bm_finalise_episode`, ~line 410)

New function. No existing lines change.

```python
def bm_drive_batch(
    bm: BMState,
    ate_food_steps: np.ndarray,          # [T, B]
    agent_in_bush_steps: np.ndarray,     # [T, B]
    dist_per_predator_steps,             # [T, B, P] or None
    dist_per_neutral_steps,              # [T, B, N] or None
    done_steps: np.ndarray,              # [T, B]
    predator_tags: Tuple[str, ...],
    neutral_tags: Tuple[str, ...],
) -> dict:
    """Drive the BM state machine over a whole [T, B] collected batch
    (train.py DreamerV3 Site 2).

    Interleaves the per-step update with per-done finalise/reset — the same
    step -> finalise -> reset ordering as the per-step drivers (train.py
    Site 1 rPPO, dreamer_srl_main) — so no step after a done leaks into the
    finished episode, the next episode keeps its opening steps, and a second
    done for the same env within the batch yields a second valid finalisation.
    (H10 fix — see docs/develop/active/issues/diag_fable5_20260704/
    fix_plan_h10_dreamer_batch_bm.md.)

    Mutates ``bm`` (per-env reset at each done). Returns
    ``{(t, i): ep_data}`` — one finalised ``*_raw`` dict per done event at
    step ``t`` for env ``i``.
    """
    results: dict = {}
    T = done_steps.shape[0]
    for t in range(T):
        info_t = {
            'ate_food':      ate_food_steps[t].astype(bool),
            'agent_in_bush': agent_in_bush_steps[t].astype(bool),
        }
        if dist_per_predator_steps is not None:
            info_t['dist_per_predator'] = dist_per_predator_steps[t]
        if dist_per_neutral_steps is not None:
            info_t['dist_per_neutral'] = dist_per_neutral_steps[t]
        done_t = done_steps[t].astype(bool)
        bm_step_update(bm, info_t, done_t)
        for i in np.where(done_t)[0]:
            i = int(i)
            results[(t, i)] = bm_finalise_episode(bm, i, predator_tags, neutral_tags)
            bm_reset_env(bm, i)
    return results
```

#### 2. `train.py` — import (lines 68-69)

```python
# BEFORE:
    make_bm_state, bm_step_update, bm_reset_env,
    bm_finalise_episode as _bm_finalise_episode_shared,

# AFTER:
    make_bm_state, bm_step_update, bm_reset_env, bm_drive_batch,
    bm_finalise_episode as _bm_finalise_episode_shared,
```

#### 3. `train.py` — Site 2 step loop (lines 1637-1648) → single driver call

```python
# BEFORE:
                    # Behavior-measure toolkit v1: per-step sequential update for Site 2 (DreamerV3 batch)
                    # The K-buffer must be driven step-by-step so done-discarding is episode-accurate.
                    if bm_enabled and agent_in_bush_steps is not None:
                        T2, B2 = done_steps.shape
                        for t2 in range(T2):
                            _bm_info_t2 = {
                                'ate_food': transitions_np['ate_food'][t2].astype(bool),
                                'agent_in_bush': agent_in_bush_steps[t2].astype(bool),
                            }
                            if dist_per_predator_steps is not None: _bm_info_t2['dist_per_predator'] = dist_per_predator_steps[t2]
                            if dist_per_neutral_steps  is not None: _bm_info_t2['dist_per_neutral']  = dist_per_neutral_steps[t2]
                            _bm_step_update(_bm_info_t2, done_steps[t2].astype(bool))

# AFTER:
                    # Behavior-measure toolkit v1: per-step sequential update for Site 2 (DreamerV3 batch)
                    # H10 fix: bm_drive_batch interleaves the per-step update with
                    # per-done finalise/reset (rPPO Site-1 pattern) so no step after a
                    # mid-batch done leaks into the finished episode. Finalised results
                    # are keyed (t, env) and merged into ep_data in the done block below.
                    bm_ep_results = {}
                    if bm_enabled and agent_in_bush_steps is not None and _bm_state is not None:
                        bm_ep_results = bm_drive_batch(
                            _bm_state,
                            ate_food_steps=transitions_np['ate_food'],
                            agent_in_bush_steps=agent_in_bush_steps,
                            dist_per_predator_steps=dist_per_predator_steps,
                            dist_per_neutral_steps=dist_per_neutral_steps,
                            done_steps=done_steps,
                            predator_tags=predator_tags,
                            neutral_tags=neutral_tags,
                        )
```

#### 4. `train.py` — Site 2 finalisation (lines 1679-1681) → stash merge

```python
# BEFORE:
                                    # Behavior-measure toolkit v1: per-episode finalisation (Site 2)
                                    if bm_enabled:
                                        _bm_finalise_episode(i, ep_data)

# AFTER:
                                    # Behavior-measure toolkit v1 (H10): merge the result
                                    # that bm_drive_batch finalised at this done step.
                                    if bm_enabled and agent_in_bush_steps is not None:
                                        ep_data.update(bm_ep_results.pop((int(d_idx), int(i))))
```

#### 5. `train.py` — Site 2 per-env reset (lines 1695-1697) → DELETE

```python
# BEFORE:
                                # Behavior-measure toolkit v1: per-env reset (Site 2)
                                if bm_enabled:
                                    _bm_reset_env(i)

# AFTER: (removed — bm_drive_batch already reset env i at its done step;
#         resetting again here would be a harmless no-op but misleading)
```

Note: `curr_start = d_idx + 1` (currently line 1698) must remain the last statement of the
per-done loop body. The `_bm_reset_env` *definition* stays — Sites 1/3/4/5 and the continual
stage-transition loop (`train.py:1242`) still use it.

#### 6. `train.py` — fail-loud consumption invariant (after the done-handling `if/else`, i.e. after current line 1740, before `global_step += ...` at 1742)

```python
                    # H10 invariant: every episode bm_drive_batch finalised must have been
                    # consumed by the done block above (both iterate the same done_steps).
                    assert not bm_ep_results, \
                        f"BM driver / done-block mismatch, unconsumed keys: {sorted(bm_ep_results)}"
```

#### 7. `tests/environment/test_bm_dreamer_batch_driver.py` (NEW FILE)

Regression test for the driver, run with:
`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/environment/test_bm_dreamer_batch_driver.py -v`

No environment needed — construct synthetic `[T, B]` numpy arrays directly (same style as
the empirical confirmations in [[07_behavior_measures]]). Common fixture:
`make_bm_state(num_envs=2, num_predator_tags=1, num_neutral_tags=0, bm_R=2.0, bm_K=3)`;
predator distance arrays `[T, 2, 1]` where value `1.0` means "in radius" (≤ R) and `5.0`
means "safe" (> R); env 1 is a quiet control (never done, threat always far, never eats)
whose finalised counts must be unaffected by env 0's dones (guards cross-env bleed).

**Assert only on M5-family counters and rate/denominator keys**
(`eat_under_threat_rate_predator_raw`, `eat_safe_rate_predator_raw`,
`eat_under_threat_safe_steps_predator_raw`) plus NaN-ness — these are unambiguous
step-counting quantities. Do **not** assert on how a *pending* M1 candidate / M2 onset
resolves at episode end — that is the known-undecided B1 rule (scope fence below); the test
must pass under whatever end-of-episode semantics the shared helpers implement now.

Test cases:

- **`test_mid_batch_death_no_contamination`** — T=12, env 0 done at t=5 only.
  Episode 1 (t=0..5): threat far all 6 steps, eat at t=2. Post-death steps t=6..11: threat
  near all 6 steps, eat at t=8. Drive `bm_drive_batch`; assert on `results[(5, 0)]`:
  `eat_under_threat_safe_steps_predator_raw == 6`, `eat_safe_rate_predator_raw == 1/6`,
  `eat_under_threat_rate_predator_raw` is NaN (episode 1 had zero threat steps).
  *Pre-fix failure mode*: all 12 steps land in the state before finalise →
  `eat_under_threat_rate == 1/6` (not NaN).
- **`test_new_episode_opening_steps_counted`** — same arrays; after the batch, call
  `bm_finalise_episode(bm, 0, ...)` directly on the carried state (episode 2 still open):
  assert `eat_under_threat_rate_predator_raw == 1/6` (6 threat steps, 1 threat eat) and
  `eat_under_threat_safe_steps_predator_raw == 0`.
  *Pre-fix failure mode*: state was wiped after the batch-end finalise → all-NaN / zeros.
- **`test_double_done_two_valid_finalisations`** — T=12, env 0 done at t=3 **and** t=9.
  Segment 1 (t=0..3): threat near all 4 steps, eat at t=1. Segment 2 (t=4..9): threat far
  all 6 steps, eat at t=6. Assert `set(results) == {(3, 0), (9, 0)}`;
  `results[(3, 0)]['eat_under_threat_rate_predator_raw'] == 1/4`;
  `results[(9, 0)]['eat_safe_rate_predator_raw'] == 1/6` and its
  `eat_under_threat_safe_steps_predator_raw == 6` (i.e. NOT the empty all-NaN finalise).
  *Pre-fix failure mode*: first finalise mixes all 12 steps; second finalise runs on a
  fully-reset state → safe_steps 0, every rate NaN.
- **`test_no_done_batch_accumulates`** — no dones in two consecutive T=6 batches: assert
  both calls return `{}` and a manual finalise afterwards reflects all 12 steps (guards
  against over-eager resets).
- **`test_matches_interleaved_reference_pattern`** — consistency pin: drive the same
  arrays through a second, independent `BMState` using a hand-rolled per-step loop that
  mirrors the rPPO Site-1 call pattern verbatim (`bm_step_update` → per-done
  `bm_finalise_episode` + `bm_reset_env`); assert the per-episode dicts equal
  `bm_drive_batch`'s output key-for-key (NaN == NaN treated as equal). Pins the extracted
  driver to the reference pattern the other five sites use.
- **Control assertion in each scenario test**: env 1's counters (finalised manually at the
  end) match its own script and are independent of env 0's dones.

### Red-then-green discipline (mandatory ordering for `developer`)

The regression test must be shown to **fail on the pre-fix behavior**:

1. **Step 1 — verbatim extraction (still buggy)**: add `bm_drive_batch` implementing the
   *current* Site-2 ordering (all-T update loop first, then per-done finalise/reset), wire
   `train.py` changes #2-#6. Existing BM tests must stay green.
2. **Step 2 — red**: add the new test file; run it; capture the failing output (the three
   scenario tests must fail; the no-done test passes) into the Implementation Report.
3. **Step 3 — green**: change `bm_drive_batch`'s body to the interleaved ordering (as
   specified in File Change #1); all new tests pass.

This proves the failure lives in the driver ordering, on production code, not in test
scaffolding.

### Test plan

All via `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest`:

1. `tests/environment/test_bm_dreamer_batch_driver.py -v` — new, green post-fix (red at Step 2).
2. `tests/environment/test_behavior_measures.py -v` — existing T1-T8 accumulator tests, must stay green.
3. `tests/training/test_continual_bm_transition.py -v` — stage-transition reset loop uses `_bm_reset_env`, must stay green.
4. Full suite `tests/` — **known-red baseline (pre-existing, NOT caused by this change)**:
   4× A1 parity + 3× stale-config (`b093023`) + 1× dreamer_srl offline-WM smoke. Any
   failure beyond these 8 is a regression from this fix.

### Speed check

N/A with justification: with `behavior_measures.enabled: false` (the default, unchanged)
the new code path is behind the same `bm_enabled` gate as before — the only additions on
the default path are one empty-dict assignment and one assert on an empty dict per
iteration. When enabled, the added work is O(#dones) finalise calls that previously
happened anyway, just later. `developer` should state this justification in the
Implementation Report rather than benchmark.

### Scope fences (verbatim from the WP-F brief — violating any of these blocks the merge)

- **(a) WP-A territory is off-limits**: `train.py` also carries the just-landed
  resume/config-load fixes (commit `9ee2a30`, H1-H3). Do NOT touch the
  restore/stage-rebuild/config-loading code anywhere in the file. The diff must be confined
  to the Site-2 BM lines listed above (plus the import line) and the two new/appended files.
- **(b) B1 stays undecided**: the episode-end rule for in-progress M1/M2 events is a
  known-undecided user decision. This plan changes only *when* finalise/reset run, never
  *what* `bm_finalise_episode` / `bm_reset_env` compute — and the regression test must not
  encode assertions that depend on the pending-event end-of-episode rule.
- **(c) Online measures default stays OFF**: no changes under `configs/`
  (`behavior_measures.enabled: false` at `configs/environment/default.yaml:234` is untouched).
- No new config keys. No files under `scripts/` touched (SCRIPTS_DEPENDENCY_MAP not
  affected). No jit/scan surface (no `code-reviewer` pass needed — see Analysis).

## Checkpoints

- [x] CP1 — After Step 1 (verbatim extraction + wiring): `test_behavior_measures.py` and
      `test_continual_bm_transition.py` green; `git diff` confined to
      `src/behavior/accumulators.py`, `train.py` Site-2 region + import line.
      *(2026-07-06: 36 passed; diff stat = accumulators.py + train.py only.)*
- [x] CP2 — Step 2 red run captured: the three scenario tests fail with the predicted
      failure modes (contaminated rate `1/6`, wiped opening-steps state, all-NaN second
      finalise); paste the pytest output excerpt into the Implementation Report.
      *(2026-07-06: 4 failed / 1 passed exactly as predicted — excerpt in the report below.)*
- [x] CP3 — Step 3 green: all 5+ new tests pass.
      *(2026-07-06: 41 passed = 5 new + 36 existing BM tests.)*
- [x] CP4 — Grep check: within the DreamerV3 branch (`train.py` ~1523-1760) there are no
      remaining calls to `_bm_finalise_episode(` or `_bm_reset_env(` — both now happen only
      inside `bm_drive_batch`. (`_bm_reset_env` elsewhere — line 1242 and Sites 1/3/4/5 —
      untouched.)
      *(2026-07-06: grep over the branch region returns nothing.)*
- [x] CP5 — Full suite run: failures exactly match the 8-test known-red baseline.
      *(2026-07-06: 9 failed / 395 passed / 494 skipped; 8 = known-red baseline; the 9th,
      `test_continual_resume_rebuilds_stage_env` rc=-11, passes in isolation — transient
      contention with a parallel developer's concurrent pytest run. A parallel dev's
      untracked `test_terminal_step_data_reset.py` breaks collection and was `--ignore`d.)*
- [x] CP6 — Scope-fence audit: `git diff train.py` shows zero hunks in resume/restore/
      config-load code; `git status` shows no `configs/` or `scripts/` changes.
      *(2026-07-06: train.py diff = 5 hunks, all Site-2 BM + import; no configs/ or scripts/.)*

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-06

### Summary (file-by-file)

- **`src/behavior/accumulators.py`** — added `bm_drive_batch(...)` after
  `bm_finalise_episode` (~line 411). Per the red-then-green discipline it was first added
  with the *verbatim buggy* Site-2 ordering (all-T update loop, then per-done
  finalise/reset, env-major like the old inline code); after the red run was captured, the
  body was changed to the interleaved ordering (per-step `bm_step_update` → per-done
  `bm_finalise_episode` + `bm_reset_env`, the rPPO Site-1 pattern). Returns
  `{(t, i): ep_data}`. No existing lines changed.
- **`train.py`** — exactly the 5 planned hunks:
  1. Import line 68: added `bm_drive_batch`.
  2. Site-2 step loop (old 1637-1648) replaced with a single `bm_drive_batch` call, gated
     `bm_enabled and agent_in_bush_steps is not None and _bm_state is not None`;
     `bm_ep_results = {}` initialised unconditionally before the gate.
  3. Site-2 finalisation: `_bm_finalise_episode(i, ep_data)` →
     `ep_data.update(bm_ep_results.pop((int(d_idx), int(i))))`, gated
     `bm_enabled and agent_in_bush_steps is not None` (the plan's deliberate micro-change).
  4. Site-2 per-env reset deleted (`bm_drive_batch` already reset env `i` at its done
     step); `curr_start = d_idx + 1` remains the last statement of the per-done loop body.
  5. Fail-loud consumption invariant (`assert not bm_ep_results, ...`) after the
     done-handling `if/else`, before `global_step += ...`.
- **`tests/environment/test_bm_dreamer_batch_driver.py`** (NEW) — 5 tests per the plan:
  3 scenario tests (mid-batch-death contamination, lost opening steps, double-done),
  no-done accumulation guard, and the interleaved-reference-pattern consistency pin
  (NaN==NaN-aware key-for-key comparison). Env 1 is the quiet never-done control in every
  scenario, asserted independent of env 0's dones. Assertions confined to M5-family
  counters + NaN-ness (B1 episode-end rule for pending M1/M2 events deliberately NOT
  encoded, per scope fence b).

### Red-then-green evidence (CP2)

Regression tests added against the **verbatim buggy extraction** (Step 1), run pre-fix
(full log: `tmp/20260706_h10_step2_red.log`):

```
FAILED test_bm_dreamer_batch_driver.py::test_mid_batch_death_no_contamination - assert False
         where False = isnan(0.16666666666666666)
         (finished episode's eat_under_threat_rate == 1/6 — contaminated by 6 post-death
          threat steps + post-death eat — instead of NaN)
FAILED test_bm_dreamer_batch_driver.py::test_new_episode_opening_steps_counted - assert nan == (1 / 6)
         (carried state wiped by the batch-end reset → the new episode's opening 6 threat
          steps + 1 threat eat lost)
FAILED test_bm_dreamer_batch_driver.py::test_double_done_two_valid_finalisations - assert nan == (1 / 6)
         (results[(9, 0)]["eat_safe_rate_predator_raw"] NaN — second finalise ran on a
          fully-reset state)
FAILED test_bm_dreamer_batch_driver.py::test_matches_interleaved_reference_pattern
         (mismatch at (3, 0) / interrupted_feeding_rate_predator_raw: 1.0 != 0.0)
PASSED test_bm_dreamer_batch_driver.py::test_no_done_batch_accumulates
========================= 4 failed, 1 passed in 1.45s =========================
```

All three scenario tests failed with **exactly the failure modes the plan predicted**; the
no-done test passed pre-fix as predicted. (The reference-pattern pin also fails pre-fix by
construction — it compares the driver to the correct interleaving.)

**Post-fix (Step 3)**: `test_bm_dreamer_batch_driver.py` — 5/5 passed.

### Test results

Interpreter: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest`

| Run | Result |
|---|---|
| `tests/environment/test_bm_dreamer_batch_driver.py -v` (pre-fix, Step 2) | **4 failed, 1 passed** (red as predicted) |
| `tests/environment/test_bm_dreamer_batch_driver.py` + `test_behavior_measures.py` + `tests/training/test_continual_bm_transition.py` (post-fix) | **41 passed** |
| Full suite `tests/` (`--ignore=tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py`, see deviations) | **9 failed, 395 passed, 494 skipped** (38:55) |

Full-suite failure triage: 8 of 9 are exactly the known-red baseline — 4× A1 parity
(`test_unified_parity` S1-S4), 3× stale-config `FileNotFoundError`
(`test_inactive_animal_offgrid` ×1, `test_truncation_not_death` ×2, `b093023`), 1×
dreamer_srl offline-WM smoke. The 9th
(`test_continual_resume_rebuild.py::test_continual_resume_rebuilds_stage_env`, subprocess
rc=-11 SIGSEGV) **passes when rerun in isolation** (`1 passed in 58.41s`) — transient
GPU/resource contention with a parallel developer's pytest run that was executing
concurrently. Not a regression from this change.

### Speed check

N/A with justification (per plan): with `behavior_measures.enabled: false` (the default,
unchanged) the new code path sits behind the same `bm_enabled` gate as before — the only
additions on the default path are one empty-dict assignment and one assert on an empty
dict per collection iteration. When enabled, the added work is O(#dones) finalise calls
that previously happened anyway, just later in the same iteration. No hot-path (jit/scan/
env-step) surface touched.

### Deviations

None from the plan's File Changes. Two environmental notes for the verifier:

1. **Full-suite `--ignore`**: a parallel developer's *untracked* test file
   (`tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py`) breaks pytest
   collection outright (`ImportError: cannot import name '_reset_terminal_step_data'` —
   their code change is still in flight), aborting the whole suite. The full-suite run
   excludes that one file. Not this plan's scope; left untouched per the parallel-work
   fence.
2. **Transient rc=-11**: documented above; isolated rerun green.

### Blockers / follow-ups

None. Working tree left uncommitted for `senior-developer` verification.

> Signed — `Implemented by: developer`

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-06

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/behavior/accumulators.py` | `bm_drive_batch` appended after `bm_finalise_episode` (line 411) | ✅ | Pure insertion — single hunk, zero existing lines changed. Body matches plan File Change #1 exactly: per-step `bm_step_update` → per-done `bm_finalise_episode` + `bm_reset_env`, returns `{(t, i): ep_data}`. |
| `train.py` — import (line 68) | `bm_drive_batch` added | ✅ | As planned. |
| `train.py` — Site-2 step loop (1637-1653) | Replaced by single `bm_drive_batch` call | ✅ | `bm_ep_results = {}` initialised unconditionally before the gate; gate `bm_enabled and agent_in_bush_steps is not None and _bm_state is not None`. `bm_enabled` ⇒ `_bm_state` non-None (train.py:1024), so the extra check is redundant belt-and-braces, harmless. |
| `train.py` — Site-2 finalisation (1684-1687) | Stash-pop merge `ep_data.update(bm_ep_results.pop((int(d_idx), int(i))))` | ✅ | Gated `bm_enabled and agent_in_bush_steps is not None` (the plan's deliberate micro-change). `.pop()` without default is fail-loud on a missing key. Merge sits just before `ep_info_buffer.append` / `iteration_episodes.append` — same `*_raw` keys, same dict, same append position/order, so aggregation shape is unchanged. |
| `train.py` — per-env reset delete | Old `if bm_enabled: _bm_reset_env(i)` removed | ✅ | `curr_start = d_idx + 1` remains the last statement of the per-done loop body (line 1701). |
| `train.py` — consumption invariant (1745-1748) | `assert not bm_ep_results` | ✅ | Placed after the done-handling `if/else`, before `global_step +=` (1750), same indentation level as the init — no code path reaches the assert with the name undefined. |
| `tests/environment/test_bm_dreamer_batch_driver.py` | NEW — 5 tests | ✅ | All 5 planned tests present; scenario arrays hand-checked (see below). B1 fence honored: assertions confined to M5-family keys + NaN-ness; the reference pin compares driver-vs-reference relatively, encoding no pending-M1/M2 episode-end rule. |

**Independent checks performed** (2026-07-06, verifier's own runs):

1. **Diff scope** — `git diff HEAD -- train.py` = exactly 5 hunks, all Site-2 BM region + import
   line. Zero hunks in the WP-A resume/config-load code (commit `9ee2a30` territory). No
   `configs/` or `scripts/` changes attributable to this package (the `scripts/eval/eval_rollout.py`
   and dreamer_srl/trainer hunks in the tree belong to declared parallel packages, out of scope here).
2. **CP4 grep re-run** — `sed -n '1523,1760p' train.py | grep '_bm_finalise_episode(\|_bm_reset_env('`
   → no matches. `_bm_reset_env` survives at 1242 and Sites 1/3/4/5 (1415/1980/2179/2353) as planned.
3. **Hand-check of scenario arrays** — mid-batch-death: episode 1 = t=0..5, all FAR
   (`dist[6:] = NEAR` starts post-death), eat at t=2 → 6 safe steps, safe rate 1/6, threat rate
   NaN — asserted values are the correct ground truth. Double-done segments (4 NEAR steps / eat
   t=1 → 1/4; 6 FAR steps / eat t=6 → 1/6, safe_steps 6) likewise check out.
4. **Red evidence** — `tmp/20260706_h10_step2_red.log` exists (11 KB, 2026-07-06 18:51) and its
   tail matches the Implementation Report excerpt verbatim (4 failed / 1 passed, contamination
   `isnan(0.1666…)` failure, `nan == 1/6` failures, reference-pin mismatch at
   `(3, 0)/interrupted_feeding_rate_predator_raw`). Both the reference pin and the quiet-env
   control are non-vacuous: the pin failed pre-fix by construction, and the control asserts
   env 1's `safe_steps == T_total`, which a cross-env reset bleed would break.
5. **Test re-runs (verifier's own)** —
   `test_bm_dreamer_batch_driver.py` + `test_behavior_measures.py` + `test_continual_bm_transition.py`
   → **41 passed** (69 s), matching the report.
6. **Transient re-check** — `test_continual_resume_rebuild.py::test_continual_resume_rebuilds_stage_env`
   in isolation → **1 passed** (57 s). The rc=-11 in the developer's full-suite run was
   parallel-pytest contention, not a regression.
7. **Full-suite failure triage** — accepted as reported: 8 failures = the pre-existing known-red
   baseline (4× A1 parity, 3× stale-config `b093023`, 1× dreamer_srl offline-WM smoke); the
   `--ignore` of a parallel developer's in-flight `test_terminal_step_data_reset.py` is an
   environmental workaround, not a finding.
8. **Speed check** — ✅ no regression (N/A-with-justification accepted): default path
   (`behavior_measures.enabled: false`, unchanged at `configs/environment/default.yaml:234`)
   adds one dict assignment + one empty-dict assert per collection iteration; enabled path moves
   existing O(#dones) finalise work earlier in the same iteration. No jit/scan/env-step surface
   touched, so no benchmark required.

**Scope fences** — all three honored: (a) WP-A code untouched; (b) B1 not encoded in any
assertion; (c) online measures default stays OFF, no config changes.

**Conclusion**: PASS — implementation matches the plan file-for-file with no deviations; the
regression test is meaningful (red-then-green on production driver code with hand-verified
ground-truth values); all six BM call sites now obey the accumulator contract. Ready to commit.

> Signed — `Verified by: senior-developer`
