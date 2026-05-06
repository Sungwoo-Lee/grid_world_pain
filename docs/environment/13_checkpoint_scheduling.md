# Checkpoint Scheduling Review

> **Status**: Description only — current behavior is **accepted as-is** per user direction (no change prescribed).
> **Primary source**: `train.py:1656-1694` | **Last updated**: 2026-04-20

This document describes how the checkpoint-save scheduler in `train.py` actually behaves under parallel-env training, to make the observed drift predictable and to serve as a reference for the continual-learning plan ([`docs/develop/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md)), which inherits the same drift for its stage-transition gate. **No scheduler change is recommended or planned.** The "Alternative Schemes" section is retained as a reference for future reviewers only.

---

## TL;DR

Under `N` parallel envs, `total_episodes_completed` can jump by up to `N` per training iteration. The checkpoint gate is evaluated **once per iteration**, not once per episode, so saved checkpoints land **at the first iteration whose cumulative episode count exceeds the next nominal milestone** — not at the milestone itself. With `num_envs=128` and `checkpoint_freq=100`, you typically observe Orbax step keys like `156, 224, 312, 467, …` instead of `100, 200, 300, 400, …`.

A second effect: if one iteration finishes more than `checkpoint_freq` episodes (possible when `num_envs ≥ checkpoint_freq`), multiple nominal milestones collapse into a single save, so some milestones are never represented on disk.

Both behaviors are mechanical consequences of the current gate code; neither is a JAX/Orbax quirk. They are **accepted** — the continual-learning plan uses the same per-iteration gate so stage transitions drift in the same way.

---

## How Checkpointing Is Implemented Today

### Call site and gate

```
# train.py:1656-1694  (abridged)
checkpoint_freq = args.checkpoint_frequency or config.get_mandatory('training.checkpoint_frequency')

if not hasattr(main, 'last_checkpoint_save'):
    main.last_checkpoint_save = 0

should_checkpoint = (total_episodes_completed >= main.last_checkpoint_save + checkpoint_freq)

if should_checkpoint:
    main.last_checkpoint_save = (total_episodes_completed // checkpoint_freq) * checkpoint_freq
    ...
    checkpointer.save(total_episodes_completed, args=ocp.args.StandardSave(ckpt_data))
    checkpointer.wait_until_finished()
```

Key facts:
- **Gate granularity = one training iteration**, i.e. one call to `jit_train(...)` covering `num_steps × num_envs` environment transitions.
- **Step key passed to Orbax = `total_episodes_completed`** — the *actual* cumulative episode count at the end of that iteration, not the nominal milestone.
- **Cursor update** snaps `last_checkpoint_save` **down** to the nearest multiple of `checkpoint_freq` below the current count: `(total // freq) * freq`.
- **`max_to_keep`** in the `CheckpointManagerOptions` (configured by `training.max_checkpoints_to_keep`). Once the limit is reached, the oldest is pruned — so early saves disappear silently.

### Episode counter advancement

`total_episodes_completed` is incremented inside a per-env loop, summed over dones within the iteration:

```
# train.py:855-889  (RecurrentPPO path; DreamerV3 path at 1575-1595 is analogous)
for t in range(num_steps):
    ...
    dones_t = done_np[t].astype(bool)
    if np.any(dones_t):
        completed_indices = np.where(dones_t)[0]
        for i in completed_indices:
            total_episodes_completed += 1
            ...
```

So a single iteration can produce up to `num_envs × num_steps` dones (in extreme short-episode / many-rollout-step regimes), though in practice `num_envs` is the binding cap in one iteration since each env can only finish once between resets done by the wrapper’s auto-reset.

### Orbax directory layout consequence

`checkpointer.save(step, ...)` creates `<models_dir>/<step>/...`. Because `step = total_episodes_completed`, directory names are exactly the *actual* episode count — **they look arbitrary** (`156/`, `224/`, `312/`, …) even though the user set a clean frequency of 100.

---

## Drift Analysis

Let `f = checkpoint_freq`, `N = num_envs`, and let `Δₖ` be the number of episodes completed in iteration `k`. Because the gate is only checked at iteration boundaries, the actual step key saved at the `k`-th save is:

```
step_k = min { E : E is the total after some iteration and E ≥ last_save_k + f }
```

### Regime 1 — normal drift ( `max(Δₖ) < f` )

One checkpoint per gate-crossing iteration. Saved step key overshoots the nominal milestone by up to `Δₖ - 1` episodes.

**Example**: `f=100`, `num_envs=128`, average episode length ≈ 200 steps, `num_steps=64`.
Typical iteration yields `Δₖ ≈ 40–80` dones.

```
iter    episodes   gate(last+f)      action          step_saved    last_save
 1      0 → 73     73 ≥ 100 ? no     —               —             0
 2      73 → 156   156 ≥ 100? yes    save            156           100
 3      156 → 224  224 ≥ 200? yes    save            224           200
 4      224 → 312  312 ≥ 300? yes    save            312           300
 5      312 → 395  395 ≥ 400? no     —               —             300
 6      395 → 467  467 ≥ 400? yes    save            467           400
 ...
```

Expected milestones `{100, 200, 300, 400}` became `{156, 224, 312, 467}`. The user never sees clean round-number directories.

### Regime 2 — milestone collapse ( `Δₖ ≥ f` )

When one iteration completes more episodes than the entire checkpoint interval, the gate fires **once** for many milestones at once:

**Example**: `f=100`, `num_envs=512`, short episodes. Iter 1 produces Δ₁=450 dones.

```
iter    episodes   gate            action    step    last_save
 1      0 → 450    yes (≥100)      save      450     400   ← milestones 100/200/300/400 all missed
 2      450 → 890  yes (≥500)      save      890     800   ← 500/600/700/800 all missed
```

On disk you see `{450, 890}` — the 100, 200, 300, 500, 600, 700 checkpoints simply never existed.

### Regime 3 — silent pruning by `max_checkpoints_to_keep`

Even when drift is small, the first few saves are deleted once the limit set by `training.max_checkpoints_to_keep` (default: 5) is reached. A 20k-episode run with `f=1000` leaves only the last 5 saves (episodes ≈ 16k, 17k, 18k, 19k, 20k) if the default is used. Early checkpoints that were correctly written are **gone** — not a scheduling bug but often mistaken for one. Set `training.max_checkpoints_to_keep: null` in your YAML to retain all checkpoints.

---

## Additional Concerns

### (A) Module-level singleton for `last_checkpoint_save`

```
if not hasattr(main, 'last_checkpoint_save'):
    main.last_checkpoint_save = 0
```

`main` is the function object; its attributes persist across calls within the same Python process. In notebook or test-runner contexts that call `main()` more than once, the cursor leaks into the second call, suppressing early checkpoints of the second run. Low-severity but worth fixing.

### (B) Step key vs. payload `episode` mismatch

`ckpt_data['episode'] = total_episodes_completed` is the *actual* episode count, and `checkpointer.save(total_episodes_completed, …)` uses the *same* value as the step key. These agree today, but any fix that makes the step key a clean milestone (option F1 below) must still store the actual count in `ckpt_data['episode']` so resume logic (`train.py:741, 794`) keeps working.

### (C) Checkpoint triggers post-iteration evaluation

`train.py:1697-1767` chains video and stats evaluation onto every successful save. If drift causes checkpoints to bunch up (Regime 2), evaluation is also skipped for those missed milestones.

### (D) Resume semantics

`train.py:725-727` uses `restore_mngr.latest_step()` to pick the checkpoint to resume. With drift, `latest_step()` returns the actual episode count (e.g. `19_467`) rather than a nominal milestone. Correct behavior, but the checkpoint *directory name* is non-obvious.

---

## Relevance to Continual Learning

See [`docs/develop/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md). The continual-learning plan uses the **same per-iteration gate** for stage transitions, so the same drift applies:

- A stage scheduled to end at episode `3000` will actually transition at the first iteration whose `total_episodes_completed ≥ 3000` — typically a few dozen episodes past the boundary when `num_envs = 128`.
- The last in-stage checkpoint won't align exactly with the stage boundary. This is **accepted**; no forced save is triggered at the boundary.
- `checkpoint_freq` becomes a per-stage list (`schedule.checkpoint_frequencies`). Everything else in the scheduler block (`train.py:1658-1694`) is unchanged.

---

## Summary of Observed Behavior

| Behavior | Cause | Status |
|---|---|---|
| Orbax step keys are actual episode counts (e.g. `156`), not nominal milestones (e.g. `100`) | Step key = `total_episodes_completed` | Accepted |
| Milestones may collapse when `num_envs ≥ freq` | Per-iteration gate, per-iteration Δ can exceed freq | Accepted (raise freq manually if tight alignment matters) |
| `main.last_checkpoint_save` persists across `main()` calls in the same process | Function-attribute singleton | Accepted (not an issue for single-run CLI usage) |
| `max_checkpoints_to_keep` prunes the oldest saves | YAML config `training.max_checkpoints_to_keep` | Accepted (set to null in config if full history needed) |

---

## Appendix: Alternative Schemes (reference only — not planned)

Retained for future reviewers. **None of these are being implemented as part of the continual-learning plan.**

Two axes: **WHEN** the save fires, and **WHAT step key** is written. Combinations:

| Option | WHEN | Step key | Cursor update | Notes |
|---|---|---|---|---|
| **F0** *(current, kept)* | First iter where `total ≥ last+f` | `total` (actual) | `floor(total / f) * f` | Drift + overshoot; round-number directories never occur. |
| **F1** | Same as F0 | `last+f` (nominal) | `last + f` | Directories become `100/, 200/, …`; requires `ckpt_data['episode']` to store actual count for resume accuracy. |
| **F2** | F1 **+ catch-up loop** for missed milestones | nominal | advance through all crossed milestones | Saves same model state under multiple step keys — guaranteed milestone coverage, wasteful on disk. Pair with higher `max_to_keep`. |
| **F3** | Require `f > max_expected_Δ` (config validation) | nominal | nominal | Guideline instead of fix. Rejects configs where `num_envs ≥ f`. |

---

## Clarifications / FAQ

**Q: What is `total_episodes_completed` counting — cumulative over the whole run, or per-iteration?**
A: **Cumulative across the entire training run**. It is initialised to 0 at start (or to the resumed value when loading a checkpoint), and only ever increments. Checkpoint step keys reflect this cumulative total — so the second training run of a 20k-episode resume will start with step keys above 20k.

**Q: Is the drift deterministic given the same config and seed?**
A: Yes. Under a fixed seed, env stepping is deterministic, so the per-iteration `Δₖ` sequence is fixed and the drift pattern reproduces exactly. Between runs with different seeds, drift values jitter but always exceed `last_save + f`.

**Q: How do I force a checkpoint at a specific episode milestone?**
A: Not supported by the current gate. Workaround: set `checkpoint_frequency` low (e.g. 1) and post-process by keeping only the savepoints nearest to your target milestones. Or implement **F2** (catch-up saves) — the plan explicitly does not.

**Q: Does `max_to_keep` apply per-run or per-resume?**
A: Per-`CheckpointManager` instance. A fresh run starts with an empty tracking list; resuming into the same `models_dir` re-attaches and Orbax re-discovers the existing saves up to the limit. Older saves that were already pruned stay gone.

**Q: Can I raise `max_to_keep` without re-training?**
A: Yes — change the value of `training.max_checkpoints_to_keep` in your YAML config. The new limit only affects future saves. Previously-pruned checkpoints aren't recovered.

**Q: Do WandB logs track nominal or actual episode counts?**
A: WandB step = `total_episodes_completed` (actual count). If you're cross-referencing WandB with checkpoint filenames, both use the same number — but neither corresponds to clean milestones. Use the actual count.

**Q: Why per-iteration and not per-episode gating?**
A: Per-episode gating would require pausing the vmapped env loop at every `done`, breaking JIT efficiency. The training loop is designed to complete a full `num_steps × num_envs` iteration atomically on GPU. Per-iteration gating is a consequence of this design choice — the trade-off between milestone precision and throughput.

**Q: What happens if `checkpoint_frequency=0` or negative?**
A: Untested. The gate `total >= last + 0` fires every iteration; a negative frequency would cause even more aggressive saving. Keep `checkpoint_frequency >= 1`.

**Q: Does the resume path reconstruct the correct `total_episodes_completed`?**
A: Yes — `ckpt_data['episode']` stores the actual count at save time. On resume, `train.py:741, 794` reads this value and seeds `total_episodes_completed`. The resumed run continues the cumulative count seamlessly.

**Q: Is Orbax's step key required to be monotonically increasing?**
A: Yes. The gate ensures `total >= last + f` strictly increases, and since `total` is cumulative, step keys are monotone. Orbax uses step keys for sorting and pruning.

**Q: Does a training crash mid-iteration lose the partial episode counts?**
A: Yes — there is no intra-iteration checkpoint. If the process dies during iteration `k`, all `Δₖ` episodes from that iteration are lost and not counted against the resume. The last saved `total_episodes_completed` is the recovery point.

**Q: Is this gate the same code path for RecurrentPPO and DreamerV3?**
A: Largely yes. Both paths increment `total_episodes_completed` the same way and hit the same gate block at `train.py:1656-1694`. The DreamerV3 episode counter (`train.py:1575-1595`) is a structural parallel of the RPPO version (`train.py:855-889`). Any gate change applies to both.
