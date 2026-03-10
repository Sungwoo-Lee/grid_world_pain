# Restore evaluation_core.py — Accidental Deletion by Gemini

> **Status**: COMPLETED
> **Opened**: 2026-03-10
> **Related**: [FIX_EVAL_TRUE_OBS_UNBOUND.md](FIX_EVAL_TRUE_OBS_UNBOUND.md), [VIDEO_TRUE_OBS_DIAGNOSIS.md](VIDEO_TRUE_OBS_DIAGNOSIS.md)

---

## Context

During training, evaluation fails with:
```
Warning: Evaluation failed: cannot import name 'evaluate_jax_checkpoint' from 'src.utils.evaluation_core'
```

This prevents all evaluation — both video generation and stats recording during training.

**Root cause**: On 2026-03-09, commit `1368408` emptied `src/utils/evaluation_core.py` to 0 bytes. The commit message claims this was intentional ("Remove evaluation_core.py to streamline codebase"), but the corresponding plan (`docs/FIX_EVAL_TRUE_OBS_UNBOUND.md`) only called for moving a single line to fix an `UnboundLocalError`. Gemini deleted the entire 694-line file instead of making the targeted fix.

Three files still import `evaluate_jax_checkpoint` from this module:
- `train.py` (lines 1697, 1704, 1719)
- `evaluation.py` (line 61, called at line 322)
- `main.py` (line 24, called at line 92)

No replacement module was created. The deletion is a breaking regression.

## Analysis

### What was lost

`evaluation_core.py` (694 lines) contained the entire evaluation framework:

| Function | Lines | Purpose |
|----------|-------|---------|
| `generic_inference()` | 23–42 | JIT-compiled inference for RecurrentPPO and DreamerV3 |
| `_write_episode_stats()` | 44–133 | CSV stats writing per episode |
| `evaluate_jax_checkpoint()` | 135–300 | Main entry point — orchestrates single/parallel eval |
| `_run_single_env_eval()` | 302–520 | Single-env evaluation with video rendering |
| `_run_parallel_env_eval()` | 522–688 | Parallel multi-env evaluation |
| `main()` | 690–694 | CLI entry point |

### The actual fix that should have been applied

Per `docs/FIX_EVAL_TRUE_OBS_UNBOUND.md`, the only change needed was:

**Move `true_obs` initialization before `if record_stats:` block** (around line 339):

```python
# BEFORE (lines 338-340):
ep_true_obs = [] if record_true_obs else None

if record_stats:

# AFTER:
ep_true_obs = [] if record_true_obs else None

# Compute true obs for step 0 (used by video render even when record_stats is off)
true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None

if record_stats:
```

And remove the duplicate at lines 360-361 inside the `if record_stats:` block:

```python
# BEFORE (lines 359-362):
            true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None
            if record_true_obs:
                ep_true_obs.append(true_obs)

# AFTER:
            if record_true_obs:
                ep_true_obs.append(true_obs)
```

---

## Implementation Plan

### Design

Two-step fix:

1. **Restore** `evaluation_core.py` from the last good commit (`2e4ac34`)
2. **Apply the original planned fix** from `FIX_EVAL_TRUE_OBS_UNBOUND.md` — move `true_obs` init before the `if record_stats:` block

### File Changes

#### 1. `src/utils/evaluation_core.py` — Restore from git

```bash
git show 2e4ac34:src/utils/evaluation_core.py > src/utils/evaluation_core.py
```

This restores the full 694-line file with all functions intact.

#### 2. `src/utils/evaluation_core.py` — Apply the true_obs fix (line ~339)

After restoring, apply the fix from `FIX_EVAL_TRUE_OBS_UNBOUND.md`:

**Step A — Add `true_obs` init before `if record_stats:` (after line 338):**

Find:
```python
        ep_true_obs = [] if record_true_obs else None

        if record_stats:
```

Replace with:
```python
        ep_true_obs = [] if record_true_obs else None

        # Compute true obs for step 0 (used by video render even when record_stats is off)
        true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None

        if record_stats:
```

**Step B — Remove duplicate inside `if record_stats:` block (lines ~359-361):**

Find (inside the `if record_stats:` block):
```python
            # Compute true obs once at step 0, gated by config, shared between CSV and video
            true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None
            if record_true_obs:
                ep_true_obs.append(true_obs)
```

Replace with:
```python
            if record_true_obs:
                ep_true_obs.append(true_obs)
```

#### 3. No other file changes needed

`train.py`, `evaluation.py`, and `main.py` all have correct imports — they just need `evaluation_core.py` to exist again with the `evaluate_jax_checkpoint` function.

---

## Checkpoints

- [x] **CP1**: After restoring the file, verify it has ~694 lines and `evaluate_jax_checkpoint` is defined: `grep -n "^def " src/utils/evaluation_core.py` [13:38:50]
- [x] **CP2**: After applying the fix, verify `true_obs` is initialized before the `if record_stats:` block — search for the line and confirm it's outside the block [13:39:00]
- [x] **CP3**: Run a short training session with evaluation enabled (`training.video_during_training: true`). Confirm no import error and a video is generated at the first eval checkpoint [13:54:00]
- [x] **CP4**: Run evaluation with `render_video=True` and `record_stats=False` — confirm no `UnboundLocalError` [13:58:00]
- [x] **CP5**: Run evaluation with both `render_video=True` and `record_stats=True` — confirm video and CSV are both generated correctly [13:58:00]

---

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-03-10 13:40:00

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-03-10

**Diff stats** (`git diff --stat HEAD`): `evaluation_core.py | 695 +++` — proportionate to restoring a 694-line file + 1-line fix. Total 6 files changed, 720 insertions, 17 deletions.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/utils/evaluation_core.py` | Restore from `2e4ac34` | ✅ | 695 lines restored. All 6 functions present (`generic_inference`, `_write_episode_stats`, `evaluate_jax_checkpoint`, `_run_single_env_eval`, `_run_parallel_env_eval`, `main`). |
| `src/utils/evaluation_core.py` | Apply `true_obs` init fix (Step A) | ✅ | Line 341: `true_obs = get_observation(...)` correctly placed **before** `if record_stats:` at line 343. Duplicate removed from inside the block (line 362 now just `ep_true_obs.append(true_obs)`). |
| `src/utils/evaluation_core.py` | Step loop `true_obs` | ✅ | Line 467: `true_obs` computed before both `render_video` (line 469) and `record_stats` (line 481) blocks. Shared correctly. |
| `configs/environment/default.yaml` | Predator count 4→3, bush counts 2→1 | ⚠️ | **Not in plan.** Environment balance changes — likely user's own edits or from another task. |
| `configs/models/neuromodulated_ppo.yaml` | Modulation type `Multiplicative`→`PreActivation` | ⚠️ | **Not in plan.** Model config change — likely user's own edits. |
| `train_command.sh` | Modified | ⚠️ | **Not in plan.** Training command changes — likely user's own edits. |

**Conclusion**: ✅ Core implementation correct — `evaluation_core.py` fully restored with the `true_obs` fix applied. Three out-of-scope config/script changes flagged (likely pre-existing user edits, not from Gemini).
