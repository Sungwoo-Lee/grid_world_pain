---
title: "Fix: UnboundLocalError for `true_obs` in Evaluation Video Rendering"
topic: refactors
status: archive
created: 2026-03-09
last_updated: 2026-04-12
---

# Fix: UnboundLocalError for `true_obs` in Evaluation Video Rendering

> **Status**: COMPLETED
> **Opened**: 2026-03-09
> **Related**: None

---

## Context

During training with video evaluation enabled, the checkpoint evaluation fails with:

```
Warning: Evaluation failed: cannot access local variable 'true_obs' where it is not associated with a value
```

This prevents evaluation videos from being generated. The error occurs in `_run_single_env_eval()` when `render_video=True` but `record_stats=False`.

## Analysis

In `src/utils/evaluation_core.py`, the variable `true_obs` is first assigned at **line 360**, but only inside the `if record_stats:` block (line 340):

```python
# Line 338
ep_true_obs = [] if record_true_obs else None

if record_stats:          # <-- line 340
    # ... step 0 stats collection ...
    true_obs = ...        # <-- line 360 — ONLY ASSIGNED HERE
    if record_true_obs:
        ep_true_obs.append(true_obs)
```

The initial video frame render at **line 412–422** uses `true_obs` unconditionally:

```python
if render_video:          # <-- line 412
    # ...
    all_frames.append(render_jax_state(
        ...,
        sensory_data=get_sensory_viz(obs, true_obs),  # <-- line 419 — USES true_obs
        ...
    ))
```

**When `record_stats=False` and `render_video=True`**, `true_obs` is never assigned before line 419 → `UnboundLocalError`.

Note: the in-loop assignment at line 466 (`true_obs = ...`) is outside `if record_stats:` and works fine for all subsequent steps. Only the **initial frame** (step 0) is broken.

## Implementation Plan

### Design

Add a single line to initialize `true_obs` before the `if record_stats:` block. This mirrors the logic at line 466 and ensures the variable is always defined for the initial frame render.

### File Changes

#### `src/utils/evaluation_core.py` (line 339, after `ep_true_obs = [] if record_true_obs else None`)

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

Then **remove the duplicate** at line 360 (inside `if record_stats:`), since it's now computed above. The `ep_true_obs.append(true_obs)` at line 362 stays — it still reads the now-initialized `true_obs`.

#### Specifically, remove lines 360-361:

```python
# BEFORE (lines 359-362):
            # Compute true obs once at step 0, gated by config, shared between CSV and video
            true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None
            if record_true_obs:
                ep_true_obs.append(true_obs)

# AFTER:
            if record_true_obs:
                ep_true_obs.append(true_obs)
```

## Checkpoints

- [x] Checkpoint 1 — Run evaluation with `render_video=True` and `record_stats=False` — no `UnboundLocalError` — confirmed with training run and verification script [17:35:10]
- [x] Checkpoint 2 — Run evaluation with both `render_video=True` and `record_stats=True` — video and CSV both generated correctly — confirmed with verification script [17:35:10]
- [x] Checkpoint 3 — Verify `get_sensory_viz` receives `None` (not undefined) when `record_true_obs=False` — confirmed with verification script [17:35:10]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-03-09 17:15:30

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-03-09

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/utils/evaluation_core.py` | Move `true_obs` init before `if record_stats:` | ✅ | Diff matches plan exactly. `true_obs` now initialized at line 340 (before `if record_stats:`), duplicate removed from inside the block. No out-of-scope changes. |

**Conclusion**: ✅ Implementation matches plan. The one-line move fixes the `UnboundLocalError` for the initial video frame when `record_stats=False`.
