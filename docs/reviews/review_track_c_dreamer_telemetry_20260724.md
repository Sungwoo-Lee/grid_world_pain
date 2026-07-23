---
title: "Code review — Dreamer-SRL eval telemetry fix (Track C)"
topic: dreamer
status: active
created: 2026-07-24
last_updated: 2026-07-24
---

# Code review — Dreamer-SRL eval telemetry fix (Track C)

> **Reviewed by**: code-reviewer · **Date**: 2026-07-24
> **Related**: [[DREAMER_SRL_EVAL_TELEMETRY_FIX]] (the plan) · [[findings_dreamer_main]] Finding 1

## What this review is about (plain language)

When the live "Dreamer" agent trains, it pauses at each checkpoint to test itself,
record a video, and post survival-step numbers to the WandB dashboard. Three
telemetry bugs made that dashboard wrong: eval videos were stamped with an
*episode count* instead of the *environment-step clock* the dashboard runs on (so
WandB silently dropped them as "backward in time"); a few logging calls had no
explicit timeline position and could eat the next 1-3 real training rows; and two
different eval passes (a cheap video pass and an optional expensive stats pass)
wrote the *same* metric keys, making the curve saw-tooth. The Track C diff fixes
all three by putting every eval/stage/video log on the env-step clock
(`policy_step`) and giving the video pass its own metric namespace when the stats
pass is also on. This review checks that the fix is JAX/WandB-correct, that all
call sites of the changed function were updated, that no step-less log survives,
and that the new tests would actually catch a regression.

**Verdict: APPROVE.** All five review targets pass. No blockers, no concerns. Two
green nits (test-robustness observations only). The diff is exactly the plan's
File Changes plus the one flagged, mechanically-forced test-call-site update.

## Per-target verdict

| # | Target | Verdict |
|---|--------|:------:|
| 1 | Step-clock correctness (`policy_step` captured at the right moment; same-step merges clean) | **OK** |
| 2 | `_eval_scalar_prefix` routing vs plan truth table; flag read matches the pass-driving key | **OK** |
| 3 | `_render_and_upload` signature change: all call sites updated, param genuinely required, failure paths intact | **OK** |
| 4 | No remaining step-less `wandb.log` in the two source files | **OK** |
| 5 | Test quality — T4 monotonicity would catch a regression to the old behavior | **OK** |

## Findings

None at blocker or concern severity. Two nits below.

| Severity | Location | Issue | Suggested fix |
|---|---|---|---|
| 🟢 nit | `test_eval_telemetry_wandb.py:129-131` | T4 assertion (c) `assert video_steps` depends on the render subprocess actually succeeding (producing the consolidated MP4 so `upload_video` fires the `eval/*` rows). In a runner without a working renderer, the test fails on "no eval/* rows captured" rather than on a telemetry regression — an environment-fragility, not a correctness gap. The scalar `Eval/Mean*` rows are robust (logged outside the `if auto_render` block), only the upload rows are conditional. | Optional: skip/xfail if the render subprocess is unavailable, or assert on the scalar-log step coincidence independent of the upload. Marked `@pytest.mark.slow` already, so blast radius is low. |
| 🟢 nit | `dreamer_srl_main.py:1755-1756`, `1780-1781` | When both passes run, the video and stats `wandb.log` payloads share the keys `iteration` and `timesteps` at the same `step=policy_step`. WandB merges same-step rows, so the second write overwrites the first — but both carry identical values (`iter_num`, `policy_step`), so the merge is a no-op. Confirmed harmless; noted only so a future reader who changes one payload's `iteration`/`timesteps` value knows the other pass silently wins the merge. | None required. |

## Target detail

### Target 1 — step-clock correctness: OK

`policy_step` advances exactly once per training iteration, at the top of the loop
(`dreamer_srl_main.py:1350`, `policy_step += num_envs`). Nothing between there and
the checkpoint-eval block (`:1712-1782`) advances it again, so at every eval log
site `policy_step` holds *the current iteration's* value — the same value passed
to `_save_checkpoint(..., policy_step=policy_step)` at `:1700`. The video upload,
both eval scalar logs, the stage log, and the checkpoint therefore all stamp the
identical `policy_step`. The diagnosis's specific worry ("eval running after the
counter has advanced within the same iteration") does not apply: there is no
intra-iteration advance.

Same-step logging is legal and merges into one WandB history row. Cross-pass key
collision was checked directly: with the stats pass on, the video pass writes
`Eval/video/MeanReward` / `Eval/video/MeanLength` while the stats pass writes
`Eval/MeanReward` / `Eval/MeanLength` — the authoritative `Eval/Mean*` keys are
**never written by two passes at once**, so the sawtooth is genuinely killed. The
only shared keys are `iteration`/`timesteps` with identical values (green nit
above).

### Target 2 — `_eval_scalar_prefix` routing: OK

Helper truth table (`dreamer_srl_main.py:409-419`) matches the plan's
"Decided semantics" table exactly:
`(True, False)→"Eval/"`, `(True, True)→"Eval/video/"`, `(False, *)→"Eval/"`.
The default config (`stats_during_training: false`) keeps the video pass writing
`Eval/Mean*`, so no existing dashboard panel goes silent — as required.

Flag-read correctness: the routing at `:1751` reads `stats_during_training`, which
is the **same** variable that gates the stats pass itself (`if
stats_during_training:` at `:1760`) and is resolved once from
`training.stats_during_training` via `get_mandatory` at `:629`. Because a single
variable drives both the pass execution and the key routing, they cannot desync —
there is no similarly-named decoy key in play. `video_during_training` (`:627`)
and the stats key are read side by side and are distinct.

### Target 3 — `_render_and_upload` signature: OK

`policy_step: int` is a **required** positional-or-keyword parameter with **no
default** (`eval.py:514`), placed before `quiet` per the plan. A missing default
is the correct choice: it makes any un-updated call site fail loudly at call time
rather than silently defaulting back to a broken clock (which is exactly how the
original bug hid). All call sites enumerated by grep are updated:

- Production: `dreamer_srl_main.py:1733` → passes `policy_step=policy_step`.
- Tests: `test_render_upload.py:37` (`policy_step=0`), `:83` (`policy_step=500`),
  `test_eval_telemetry.py:68` (`policy_step=1_000_000`).

The `test_render_upload.py` edits are the flagged, mechanically-forced deviation
from the plan's file list — correct and minimal. Failure paths are untouched: the
subprocess-render guard, the empty-`recordings_dir` early return, and the
post-render `os.path.exists` check are unchanged; the diff only threads the new
`step` value into the existing `upload_video(...)` call inside the already-guarded
`if wandb_enabled:` branch. `wandb_utils.py` and `evaluation_core.py` (the rPPO
path) are untouched, so rPPO stays byte-identical as the plan requires.

### Target 4 — no step-less `wandb.log` remaining: OK

Grep of every `.log(` in both source files:

- `dreamer_srl_main.py:1085` — `wandb.log(ep_log, step=step)` (per-episode row;
  explicit step; pre-existing, out of diff scope).
- `dreamer_srl_main.py:1671` — stage log, now `..., step=policy_step)` (`:1676`).
- `dreamer_srl_main.py:1752` — video eval scalar, `..., step=policy_step)`.
- `dreamer_srl_main.py:1777` — stats eval scalar, `..., step=policy_step)`.
- `dreamer_srl_main.py:2059` — training row, `wandb.log(log_dict, step=policy_step)`.

`eval.py` has no direct `wandb.log`; it routes through `upload_video`, which now
receives `step=policy_step`. No step-less log survives in either file.

### Target 5 — T4 catches a regression: OK

T4 (`test_eval_telemetry_wandb.py`) drives a real single-env `main()` with
`WANDB_MODE=offline`, patches `Run.log` at the *class* level (correct — the
plan/report note that `wandb.init()` rebinds the module-level `wandb.log`, so a
module-attr patch would be silently overwritten), and captures `(sorted(keys),
step)` per call. Its three assertions each bite on the pre-fix behavior:

- **(a) no `step=None`**: pre-fix the stage log (`:1671`) and both eval scalar
  logs (`:1752`,`:1777`) were step-less → `step=None` → assertion fails. This is
  the direct Item-2 guard.
- **(b) monotone non-decreasing**: pre-fix, the video *upload* stamped
  `step=checkpoint_pct` (a small episode count) after training rows had already
  advanced `policy_step` → a backward step → assertion fails. This is the Item-1
  guard. (Even in isolation — regress only the upload step back to
  `checkpoint_pct` while keeping the eval scalar logs explicit — assertion (c)
  still fails because the upload's `eval/*` step no longer coincides with the
  `Eval/Mean*` step.)
- **(c) video row and `Eval/Mean*` row share a step**: locks the two onto the
  same checkpoint clock.

I confirmed the `startswith('eval/')` vs `startswith('Eval/Mean')` split is
case-correct: `upload_video` emits lowercase `eval/video` / `eval/checkpoint_episode`,
the scalar logs emit capitalized `Eval/...`, so the two buckets are disjoint and
correctly identified. T3 is honestly labelled a *locked-design* guard (passes pre
and post fix), not a regression test — appropriate.

## Conventions audit

Track C is a WandB-telemetry-only change in the training driver; the JAX-specific
conventions are largely N/A (no pytree/EnvState edits, no vmap, no new config
keys, no sensor/observation changes). Recorded for completeness:

- Pytree & immutability: N/A (no state-struct edits) — ✅ nothing violated
- JIT recompilation triggers: N/A (no static/traced field moves; edits are in the
  once-per-checkpoint `if use_wandb:` branch, outside the hot loop) — ✅
- vmap & batch conventions: N/A — ✅
- PRNG key threading: unchanged; the fixed per-checkpoint eval seed is a
  documented, deliberate paired-eval choice (plan §Decided semantics), not a bug — ✅
- Sensor / observation breakdown sync: N/A — ✅
- Config protocol: `stats_during_training` / `video_during_training` read via
  `get_mandatory` (`:627-630`); no new keys added — ✅

## Conclusion

The Track C diff correctly and minimally implements the telemetry fix; all five
review targets pass with only two green test-robustness nits — safe to commit.

Reviewed by: code-reviewer
