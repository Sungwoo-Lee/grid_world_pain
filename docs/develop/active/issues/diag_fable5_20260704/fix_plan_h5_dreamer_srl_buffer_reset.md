---
title: "Fix plan H5 — dreamer_srl replay-buffer terminal-reward/death-flag bleed"
topic: issues
status: active
created: 2026-07-06
last_updated: 2026-07-06
---

# Fix plan H5 — dreamer_srl: stop each episode's terminal reward + death flag bleeding into the next episode's first buffer row

> **Status**: IMPLEMENTED + VERIFIED (2026-07-06)
> **Opened**: 2026-07-06
> **Related**: [[04_dreamer_srl]] (Finding 1, the source of record) · [[00_combined_diagnosis]] §2 row H5 · work package **WP-C**

---

## Context

The project's from-scratch JAX/Flax port of the DreamerV3 world-model agent (the
`dreamer_srl` package) learns from a replay buffer. An independent code audit
(2026-07-04) found that the port drops one small block that its reference
implementation (the vendored PyTorch "sheeprl" DreamerV3) runs at every episode
boundary: when an episode ends, sheeprl **zeroes out the staged reward and
"episode ended" flags** before the next episode's first observation is written to
the buffer. The port never does this. As a result, **every episode's first stored
row carries the previous episode's final reward and death flag** while showing a
fresh, benign starting observation.

Why it matters (three effects, per the audit):

1. **Reward bias at episode starts** — the world model's reward predictor is trained
   to expect the previous episode's terminal reward on a fresh starting state, every
   episode. This fires even in the current food-only configs (where episodes end by
   time-out, not death).
2. **"Still alive?" head mislabeled after a death** — when the previous episode ended
   in death, the continue predictor is taught that fresh starting states are terminal.
3. **Imagination from starting states is silently discount-zeroed after a death** —
   the imagination rollout multiplies the entire imagined trajectory's learning weight
   by zero when its start row is one of these poisoned rows, so the actor and critic
   learn nothing from post-death restart states.

The fix is a small, well-understood port-completeness change: restore the missing
zeroing so the persistent per-step data dict is cleaned for done environments right
after the terminal row is written. This plan covers **only** that fix and its
regression test — no other `dreamer_srl` finding is in scope (see Scope fence below).

The bug lives at `src/algorithms/dreamer_srl/dreamer_srl_main.py:1242-1262`; the
missing reference block is `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:652-656`
("Reset already inserted step data").

---

## Analysis

### Why the poisoned row happens (data-flow trace)

The collection loop keeps one persistent dict, `step_data`, that stages the row to be
written on the *next* iteration. The write order is:

- **Top of iteration N** (`dreamer_srl_main.py:1124`): `buffer.add(step_data, ...)`
  writes the row that was staged at the end of iteration N-1.
- **Mid-iteration N**: env steps; the terminal step's values are staged into
  `step_data` at lines 1178-1182 —
  `step_data["obs"]` = the terminal/next observation,
  `step_data["rewards"]` = the terminal reward (line 1179),
  `step_data["terminated"]` = the terminal death flag (line 1180),
  `step_data["truncated"]` = the terminal time-out flag (line 1181),
  `step_data["is_first"]` = 0 (line 1182).
- **Done block** (`dones_idxes` non-empty, lines 1186-1278):
  - line 1251 `buffer.add(reset_data, done_mask=dones, ...)` writes the second
    ("reset_data") row holding the true terminal observation (correct — this is the CP7-P1 fix);
  - line 1257 `player.init_states(...)` resets the recurrent state for done envs;
  - lines 1259-1262 set `is_first = 1` for the done envs' **next** row;
  - lines 1287-1298 auto-reset the env and, because `step_data["obs"]` is a numpy
    view onto `next_obs`, overwrite `step_data["obs"]` for done envs with the **fresh
    reset observation**.
- **Top of iteration N+1** (`dreamer_srl_main.py:1124`): `buffer.add(step_data, ...)`
  now writes a row whose observation is the fresh reset obs and whose `is_first` is 1,
  **but whose `rewards`, `terminated`, `truncated` are still the terminal values from
  lines 1179-1181** — because nothing cleared them.

That row is the poisoned episode-start row: `(obs = fresh reset obs, is_first = 1,
rewards = previous episode's terminal reward, terminated = previous episode's death
flag, truncated = previous episode's time-out flag)`.

### What the reference does (the missing block)

`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:652-656` (pinned to
sheeprl@33b6366, the commit the port declares fidelity to):

```python
# Reset already inserted step data
step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
```

The port already performs the last line (the `is_first = 1` set, at
`dreamer_srl_main.py:1262`). The three **zeroing** lines (rewards / terminated /
truncated) are the missing piece.

### Zeroing semantics chosen (decision)

**Zero only the done-env columns (`dones_idxes`), not all envs.** sheeprl zeroes only
`[:, dones_idxes]`, and this is correct: envs that did **not** finish this step still
hold live, non-terminal reward/flag values in `step_data` that must survive to their
own next `buffer.add`. Zeroing all columns would corrupt the in-flight rows of
still-running envs. The port keeps `step_data` at fixed full width `[1, num_envs, 1]`
(unlike sheeprl's variable-width `reset_data`), so the column-subset assignment
`step_data["rewards"][:, dones_idxes] = 0.0` is a pure-numpy in-place write with no
shape variation and no JAX tracing — it matches the surrounding fixed-width style and
introduces no recompile risk.

### DEVIATION_LOG — no new row needed

The `dreamer_srl` package keeps a deviation log at
`docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md` (57 `D-NNN` rows). That log
ratifies deviations the port deliberately **keeps** — each row is a difference from
sheeprl that survives, with a sheeprl citation and a PI sign-off. This fix does the
opposite: it **eliminates** an (undeclared, accidental) divergence and restores exact
parity with the reference block. There is therefore **no new `D-NNN` row to add** —
this is purely a port-completeness fix. The audit finding ([[04_dreamer_srl]] Finding 1)
already records that the divergence existed; this plan's cross-link back into that doc
closes the loop. (If the developer finds the log has grown a "resolved/closed
divergences" convention since 2026-05-14, a one-line pointer there is optional and
harmless — but it is not required and is not part of the scope fence.)

---

## Implementation Plan

### Design

Restore sheeprl's three-line zeroing of the staged `step_data` for done envs, placed
in the same done block, grouped with the existing `is_first = 1` set so all four
"reset already inserted step data" operations are contiguous and mirror sheeprl's
L653-656 grouping.

**Testability decision.** The done-handling logic is inline in the ~500-line `main()`
collection loop and is not independently importable, so a pure-inline three-line patch
cannot be unit-tested without either (a) driving the whole `main()` loop, or (b)
extracting the block into a callable. The brief explicitly sanctions "a minimal harness
around the done-handling block", and the regression test it specifies (set a terminal
reward + `terminated=1` **by hand**, then assert the next buffer row is clean) is only
achievable deterministically if the reset step is callable. **Therefore the recommended
design factors the four `step_data`-reset operations into one small pure helper**,
`_reset_terminal_step_data(step_data, dones_idxes)`, called from the driver where the
inline `is_first` set currently sits. This is still "only the zeroing block" — the
zeroing lives in the helper, the driver calls it — and it lets the regression test
exercise the **real** production code (so it genuinely fails pre-fix and passes
post-fix, with no test/production drift).

A documented fallback (Option B, if the developer/user insists on a literal pure-inline
patch with no helper) is given in the Test Plan; it is heavier and less deterministic
and is **not** the recommendation.

### File Changes

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py`

**(1) Add the helper** near the other module-level helpers (e.g. just above `def main()`
at line 342). It performs sheeprl's L653-656 exactly, on the port's fixed-width
`step_data`:

```python
def _reset_terminal_step_data(step_data: dict, dones_idxes: list) -> None:
    """Zero the staged reward/termination flags and set is_first for done envs.

    Restores sheeprl's post-done "Reset already inserted step data" block, which
    the JAX port previously dropped (only the is_first set was ported). Without
    this, the next row written for a done env inherits the *previous* episode's
    terminal reward and death flag while showing the fresh reset observation.

    In-place, pure numpy. Zeroes ONLY the done-env columns — non-done envs keep
    their live in-flight reward/flag values for their own next buffer.add.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L653-L656.
    Fix for H5 — see docs/develop/active/issues/diag_fable5_20260704/
    fix_plan_h5_dreamer_srl_buffer_reset.md
    """
    step_data["rewards"][:, dones_idxes]    = 0.0
    step_data["terminated"][:, dones_idxes] = 0.0
    step_data["truncated"][:, dones_idxes]  = 0.0
    step_data["is_first"][:, dones_idxes]   = 1.0
```

**(2) Replace the inline `is_first` set with a call to the helper.** Current code
(lines 1259-1262):

```python
# BEFORE (dreamer_srl_main.py:1259-1262):
            # Set is_first=1 in step_data so the NEXT row written has is_first=1
            # (sheeprl L656: step_data["is_first"][:, dones_idxes] = ones_like(...))
            is_first_next[dones_idxes] = 1.0
            step_data["is_first"][:, dones_idxes] = 1.0
```

```python
# AFTER:
            # Reset the already-staged step_data for done envs so the NEXT row
            # written (buffer.add(step_data) at the top of the next iteration) does
            # NOT inherit this episode's terminal reward / death flag. The port
            # previously set only is_first here and dropped sheeprl's reward/
            # terminated/truncated zeroing (H5). Restored via the helper.
            # Ported from sheeprl@33b6366:dreamer_v3.py:L652-L656 ("Reset already
            # inserted step data"). Fix H5 — see fix_plan_h5_dreamer_srl_buffer_reset.md
            is_first_next[dones_idxes] = 1.0
            _reset_terminal_step_data(step_data, dones_idxes)
```

Notes for the developer:
- `is_first_next[dones_idxes] = 1.0` is a **separate** local array used later in the
  loop; keep it. The helper mutates only `step_data`.
- `step_data["rewards"]` is shape `[1, num_envs, 1]` (set at line 1179),
  `step_data["terminated"]`/`["truncated"]` are `[1, num_envs, 1]` (lines 1180-1181),
  `step_data["is_first"]` is `[1, num_envs, 1]` (line 1182) — so `[:, dones_idxes] = 0.0`
  / `= 1.0` broadcast correctly.
- Placement is equivalent to inserting immediately after `buffer.add(reset_data, ...)`
  at line 1251 (the brief's phrasing); grouping it with the existing `is_first` set at
  1262 keeps the four reset operations contiguous, matching sheeprl.

#### `tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py` (new)

Regression test on **buffer contents** (see Test Plan for the full spec).

### Speed

This adds three in-place numpy column-writes on tiny `[1, num_envs, 1]` arrays, only on
done boundaries (a small fraction of iterations), on the CPU-side collection path. It is
not on the JAX hot path and does not affect compilation. **No measurable runtime change
expected.** The developer should still record a before/after `s/it` (or env-SPS) note in
the Implementation Report per protocol; if for any reason it cannot be measured, state
why the change cannot affect runtime.

---

## Test Plan

### Recommended: Option A — helper + real buffer, deterministic

New file `tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py`, following the
conventions of the sibling tests (repo-root `sys.path` insert, run under the main conda
env — the `sheeprl_bridge` env is only for upstream-sheeprl runs, not our code). Header
docstring must state: **"Must FAIL on pre-fix code and PASS after."**

Run with:
```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    -m pytest tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py -v
```

**Test 1 — `test_helper_zeroes_done_env_terminal_fields` (helper unit).**
Build a `step_data` dict by hand with `num_envs = 2` where env 0 ends the episode with a
**nonzero terminal reward** and `terminated = 1`, and env 1 is mid-episode with a live
nonzero reward and `terminated = 0`:

```python
step_data = {
    "rewards":    np.array([[[ -5.0], [ 0.3]]], dtype=np.float32),  # [1, 2, 1]
    "terminated": np.array([[[ 1.0], [ 0.0]]], dtype=np.float32),
    "truncated":  np.array([[[ 0.0], [ 0.0]]], dtype=np.float32),
    "is_first":   np.array([[[ 0.0], [ 0.0]]], dtype=np.float32),
}
_reset_terminal_step_data(step_data, dones_idxes=[0])
```
Assert, for the **done** env (col 0): `rewards == 0`, `terminated == 0`,
`truncated == 0`, `is_first == 1`. Assert, for the **non-done** env (col 1), the live
values are **untouched**: `rewards == 0.3`, `terminated == 0`, `is_first == 0`. (The
non-done assertion guards the "zero only done columns" semantics.)

**Test 2 — `test_next_buffer_row_is_clean` (buffer contents, the brief's core assertion).**
Drive the real two-write sequence through a real `SequentialReplayBuffer` exactly as the
driver does, so the assertion is on **persisted buffer contents**:

1. Construct `SequentialReplayBuffer(buffer_size=..., n_envs=2, obs_keys=("obs",))`.
2. Stage the terminal `step_data` (env 0 done, nonzero terminal reward, `terminated=1`)
   and `buffer.add(step_data)` — the terminal row.
3. Write the `reset_data` row with `done_mask` for env 0 (mirror lines 1242-1251).
4. Call `_reset_terminal_step_data(step_data, [0])` and set `step_data["obs"]` to a
   fresh reset obs for env 0 (mirror the auto-reset).
5. `buffer.add(step_data)` — the episode-start row.
6. Read the persisted arrays back from `buffer._buf` at env-column 0 and locate the row
   with `is_first == 1`. Assert that row has `rewards == 0.0`, `terminated == 0.0`,
   `truncated == 0.0`.

**Pre-fix RED / post-fix GREEN:** with the fix reverted (helper absent, or its three
zeroing lines removed so only `is_first` is set), Test 1's done-env `rewards`/
`terminated`/`truncated` assertions and Test 2's buffer-content assertions fail (the row
carries `-5.0` / `1.0`). The developer must run once with the fix reverted to confirm RED,
then restore and confirm GREEN, per the suite's Lever-A gate rule
(`tests/algorithms/dreamer_srl/README.md`).

### Fallback: Option B — drive `main()` (only if pure-inline, no helper, is required)

If a literal pure-inline three-line patch (no helper) is mandated instead, the test must
drive the real collection loop: monkeypatch `SequentialReplayBuffer` in the
`dreamer_srl_main` namespace to capture the instance, run `main()` in pure-prefill mode
(`learning_starts` large so no model training — fast) with `--num-envs 2 --no-wandb
--quiet` on a small food-only config short enough that episodes complete within the run,
then inspect the captured `buffer._buf` and assert every `is_first == 1` row after the
first per env has `rewards == 0`. This covers only the reward-bleed effect deterministically
(food-only ends are time-outs, so `terminated == 0` regardless); asserting the death-flag
bleed would additionally require a deterministic-death fixture (e.g. near-zero
`start_satiation` with injury/death enabled). Option B is heavier, slower, and less
deterministic — hence not recommended.

---

## Comparability caveat

This fix **changes what dreamer_srl learns**: after the fix, episode-start buffer rows
carry `reward = 0`, `terminated = 0`, `truncated = 0` instead of the previous episode's
terminal values. The reward head, continue head, and imagination weighting all see
different targets at episode boundaries. **Therefore any dreamer_srl training run started
after this fix is NOT directly comparable to runs started before it.** Old runs were
trained on poisoned episode-start rows; do not pool pre-fix and post-fix runs in the same
comparison, and note the fix commit as a boundary when reading historical dreamer_srl
curves. This is expected and intended — the pre-fix behavior was a bug.

---

## Scope fence

**In scope:** the `step_data` zeroing block (via the helper) + the regression test, plus
the cross-link append into [[04_dreamer_srl]]. **Nothing else.**

Explicitly **out of scope** (each is its own future work package — do not touch here):
the observation-loss deviations (extra symlog + halved weight, Finding 2), missing
gradient clipping (Finding 3), the `learning_starts` iterations-vs-env-steps semantics
(Finding 6), the replay-ratio remainder dead code (Finding 4), the episode-logging
double-count (Finding 5), the train-gate `_full` check (Finding 7), the
`termination_reason >= 2` / `overeating_death` latent trap (Finding 8), and the
checkpoint optimizer-state omission (Finding 9 / A2). Do not "improve" adjacent code.

---

## Known-red baseline (not the developer's breakage)

The following tests are **already RED before this change** and are NOT caused by this fix
— do not attempt to fix them here, and do not let them block sign-off:

- **4× rPPO A1 parity gates** — pre-existing red.
- **3× stale-config tests** (introduced/known-red at commit `b093023`).
- **1× dreamer_srl offline-WM smoke** (`tests/scripts/test_dreamer_srl_offline_wm_test.py`)
  failing with "Only 36 valid starting states" — pre-existing red.

Record in the Implementation Report which tests were red **before** touching anything
(run the suite once up front), so the verifier can confirm the new test is the only delta.

---

## Checkpoints

- [x] Checkpoint 1 — Run the dreamer_srl test suite (and the known-red set above) **before**
  any edit; record the baseline red list in the Implementation Report. *(Done 2026-07-06:
  8 failed / 140 passed / 246 skipped — the 8 reds are exactly the known-red set.)*
- [x] Checkpoint 2 — With the new test written but the fix **reverted**, confirm the new
  test is **RED** (both Test 1 done-env assertions and Test 2 buffer-content assertions fail).
  *(Done: both tests failed with "reward not zeroed: -5.0" / "episode-start row inherited
  terminal reward: -5.0" under pre-fix semantics.)*
- [x] Checkpoint 3 — Apply the fix (helper + call site); confirm the new test is **GREEN**.
  *(Done: 2 passed.)*
- [x] Checkpoint 4 — Confirm the non-done env assertions pass (Test 1 col 1 untouched) —
  proves "zero only done columns" semantics. *(Done: col-1 assertions pass in both tests.)*
- [x] Checkpoint 5 — Re-run the full dreamer_srl suite; confirm no previously-GREEN test
  regressed (the known-red set stays red for its own reasons; nothing new turns red).
  *(Done: full suite post-fix — 8 failed / 407 passed / 494 skipped; the 8 reds are exactly
  the known-red set.)*
- [x] Checkpoint 6 — Record a before/after `s/it` (or env-SPS) note, or state why runtime
  cannot be affected. *(Done: before sps=13.8, after sps=13.4 — see Implementation Report.)*

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-06

### What was implemented (file-by-file)

- **`src/algorithms/dreamer_srl/dreamer_srl_main.py`** (+29/-3 lines)
  - Added module-level helper `_reset_terminal_step_data(step_data, dones_idxes)` just
    above `main()`, exactly per the plan's File Changes block: zeroes
    `rewards`/`terminated`/`truncated` and sets `is_first=1` for the done-env columns
    only, in-place, pure numpy. Docstring cites
    `sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L653-L656` and this plan doc.
  - Replaced the inline `step_data["is_first"][:, dones_idxes] = 1.0` in the done block
    (formerly lines 1259-1262) with a call to the helper, keeping the separate
    `is_first_next[dones_idxes] = 1.0` local-array set, per the plan's AFTER block.
    Comment cites sheeprl L652-L656 ("Reset already inserted step data") and H5.
- **`tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py`** (new, 2 tests)
  - Test 1 `test_helper_zeroes_done_env_terminal_fields` — helper semantics per plan
    spec (done env col 0 zeroed + `is_first=1`; live env col 1 untouched).
  - Test 2 `test_next_buffer_row_is_clean` — drives the driver's real two-`buffer.add`
    done-boundary sequence through a real `SequentialReplayBuffer` (terminal row →
    `reset_data` row via `done_mask` → helper + fresh obs → episode-start row), then
    asserts the persisted `is_first==1` row has `rewards == terminated == truncated == 0`
    (plus a fresh-obs row-identity sanity check and non-done-env col-1 preservation).

### Test results

- **Checkpoint 1 baseline (pre-fix, before any edit)** — dreamer_srl suite + known-red
  set (`tests/algorithms/dreamer_srl/` + `tests/env/test_unified_parity.py` +
  `tests/env/test_truncation_not_death.py` + `tests/env/test_inactive_animal_offgrid.py` +
  `tests/scripts/test_dreamer_srl_offline_wm_test.py`):
  **8 failed, 140 passed, 246 skipped** (log: `tmp/20260706_h5_prefix_targeted_baseline.log`).
  The 8 reds are exactly the pre-declared known-red set: 4× `test_unified_parity`
  `observability_gates_S1-S4`, 2× `test_truncation_not_death` (stale config path), 1×
  `test_inactive_animal_offgrid` (stale config path), 1× `test_dreamer_srl_offline_wm_test`
  ("Only 36 valid starting states"). All dreamer_srl-suite tests green pre-fix.
- **Checkpoint 2 pre-fix RED** — two stages:
  1. Helper absent (true pre-fix tree): new test file fails at collection with
     `ImportError: cannot import name '_reset_terminal_step_data'`.
  2. Helper present with the three zeroing lines removed (only `is_first` set — the
     plan's sanctioned revert form): **both tests FAIL on the core assertions** —
     Test 1: `AssertionError: done-env reward not zeroed: -5.0`;
     Test 2: `AssertionError: episode-start row inherited terminal reward: -5.0 (expected 0.0)`.
- **Checkpoint 3/4 post-fix GREEN** — `pytest tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py -v`:
  **2 passed**, including the non-done-env (col 1) preservation assertions.
- **Checkpoint 5 full suite post-fix** — `pytest tests/ -q`:
  **8 failed, 407 passed, 494 skipped** (log: `tmp/20260706_h5_postfix_full_suite.log`).
  The 8 failures are byte-for-byte the same known-red set as the pre-fix baseline —
  nothing previously green turned red; the only delta vs. baseline is the 2 new tests
  (RED pre-fix → GREEN post-fix).

### Speed check (Checkpoint 6)

Smoke run: `dreamer_srl_main.py --env-config configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml
--agent-config configs/models/dreamer_srl/01_food_only_buf50k.yaml --num-envs 4
--total-steps 8000 --seed 0 --no-wandb --log-interval 50`, local RTX 4090
(`CUDA_VISIBLE_DEVICES=1`), env-step mode so both runs execute identical iteration and
grad-step schedules.

| Variant | Final cumulative env-SPS | Log |
|---|---|---|
| Before (zeroing lines commented out — pre-fix semantics) | 13.8 | `tmp/20260706_h5_speed_before.log` |
| After (fix in place) | 13.4 | `tmp/20260706_h5_speed_after.log` |

Delta: **-2.9%**. Flagged per protocol, but almost certainly run-to-run noise on a shared
workstation: the change is three in-place numpy writes on `[1, 4, 1]` arrays executed only
at done boundaries, off the JAX hot path, with no compilation impact; the after-run was
measured first and the machine carried background load. The verifier may re-measure if
the delta matters. (`tests/algorithms/dreamer_srl/bench_sps.py` could not be used as-is —
its agent-config paths point at `configs/dreamer_srl/…`, which no longer exists; noted as
a pre-existing staleness, not fixed here per scope fence.)

### Deviations from the plan

- **None in the code change or test.** Helper body, call site, and both tests match the
  plan's File Changes / Test Plan (Option A) blocks.
- **Process note (Checkpoint 1 scope):** the pre-fix baseline was run as the plan's
  Checkpoint 1 specifies (dreamer_srl suite + the named known-red set) rather than the
  full 890-test suite twice; the full suite was run once post-fix and its red set matches
  the launcher-provided known-red list exactly, so the green-stays-green comparison holds.
- **Cross-link append into `04_dreamer_srl.md` not done** — the plan's scope fence lists
  it, but the implementation brief's scope fence restricts the developer to
  `dreamer_srl_main.py` + the new test file (docs edits limited to this report). Left for
  `senior-developer` to add during verification (one line pointing Finding 1 → this plan's
  Implementation Report).

### Blockers / follow-ups

- None blocking. Pre-existing staleness observed in passing (out of scope, not touched):
  `bench_sps.py` stale config paths; the known-red stale-config tests
  (`test_truncation_not_death`, `test_inactive_animal_offgrid`) still reference removed
  config files.
- Reminder for run bookkeeping: per the Comparability caveat, dreamer_srl runs started
  after this fix are not comparable to pre-fix runs.

**Implemented by: developer**

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-06

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | helper + call site | ✅ | `git diff HEAD`: exactly +29/−3, two hunks only (helper above `main()`, call-site swap at the done block). Helper body is byte-equivalent to the plan's File Changes block and semantically identical to `sheeprl@33b6366:dreamer_v3.py:L652-656` (re-read the vendored reference to confirm). Zeroing is column-subset (`[:, dones_idxes]`) — live envs keep in-flight values. `is_first_next` local set retained. Nothing else in the file changed. |
| `tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py` | new regression test | ✅ | Test 2 drives a **real** `SequentialReplayBuffer` through the driver's real two-`buffer.add` done-boundary sequence and asserts on the **persisted** `_buf` row (`is_first==1` → rewards/terminated/truncated all 0), with a fresh-obs row-identity check so the assertion targets the episode-start row, not the terminal row. Test 1's live-env assertion is non-vacuous (col 1 holds a nonzero staged reward 0.3 that must survive). Checkpoint 2's pre-fix RED ("episode-start row inherited terminal reward: -5.0") is behavioral, not import-shaped. Independently re-run: **2 passed** (0.65 s). |
| `docs/develop/active/issues/diag_fable5_20260704/04_dreamer_srl.md` | cross-link append (deferred by developer) | ✅ | Added by verifier: "Fix plan: implemented and verified" pointer under Finding 1, linking back to this doc's Implementation/Verification Reports. |

**Independent checks performed:**

1. **Ordering at the call site** — the helper runs at line 1288, after `buffer.add(reset_data, done_mask=dones, ...)` at line 1272, so the terminal and reset rows are persisted with the real terminal reward/flags before the staged dict is zeroed. Confirmed `buffer.add` **copies** into the preallocated `_buf` arrays (slice assignment, `buffers.py:268-296`), so the in-place zeroing cannot retroactively corrupt the already-persisted rows even though `reset_data["rewards"]/["terminated"]/["truncated"]` alias the same arrays the helper mutates.
2. **Aliasing side-effects ruled out** — `step_data["terminated"]/["truncated"]` are numpy **views** onto `terminated_np`/`truncated_np` (lines 1201-1202), so the helper writes through to them; verified neither (nor `dones`) is read after line 1288 within the iteration (last reads: 1272/1277), and all are rebuilt fresh next iteration. `step_data["rewards"]` is an `.astype` copy — no write-through to the `rewards` local used in episode logging.
3. **Test suite re-run** — `pytest tests/algorithms/dreamer_srl/` (incl. the new file): **110 passed, 2 skipped, 0 failed** — the dreamer_srl subset is fully green, matching the developer's baseline (the known-red offline-WM smoke lives in `tests/scripts/`, outside this subset; the 8-red full-suite set matches the pre-declared known-red list per the Implementation Report).
4. **Speed** — ✅ **no regression**. Independent re-run of the identical 8000-step smoke on the same machine/GPU/seed with the fix in place reached **15.5 env-SPS** — *faster* than both the developer's "before" (13.8) and "after" (13.4) — confirming the reported −2.9% is run-to-run noise on a shared workstation, as attributed. Mechanically the change (three in-place numpy writes on `[1, num_envs, 1]` arrays at done boundaries, off the JAX hot path) cannot plausibly cost 3%. Log: `tmp/20260706_h5_speed_verify.log`.
5. **Observation (not a blocker, not H5)** — the smoke runs show transient world-model-loss excursions to ~1e29-1e31 magnitude (developer's "after" log persistent through the 8000-step window; verifier's re-run spiked at iter ~1275 then recovered to O(1) by iter ~1525). This is run-level instability consistent with the audit's **Finding 3 (missing gradient clipping)** — out of scope here per the fence, but it strengthens the case for prioritizing that work package. The H5 fix itself is a verbatim port of the reference block and the regression test proves the buffer semantics.
6. **Scope** — no out-of-scope changes: the only uncommitted deltas attributable to WP-C are the two in-scope files plus this plan doc and the deferred `04_dreamer_srl.md` pointer (remaining tree dirt — `scripts/eval/render_recordings.py`, `train_command-agent.sh`, diary/index files — belongs to other sessions, as declared by the launcher).

**Conclusion**: WP-C verified — the H5 fix exactly restores sheeprl's "Reset already inserted step data" block with correct done-column-only, after-persist semantics; the regression test is behavioral and RED-pre/GREEN-post; no speed regression (independent re-run beat both baseline numbers); no out-of-scope changes.

**Verified by: senior-developer**
