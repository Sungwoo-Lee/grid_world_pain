---
title: "Dreamer-SRL eval telemetry fix (Track C): rescue dropped videos, fix step clock, split the eval estimators"
topic: dreamer
status: active
created: 2026-07-24
last_updated: 2026-07-24
---

# Dreamer-SRL eval telemetry fix (Track C)

> **Status**: PLANNED
> **Opened**: 2026-07-24
> **Related**: [[findings_dreamer_main]] (Finding 1 + its "related minor") · [[findings_dreamer_eval_ckpt]] (regression-check depth note) · [[review_full_diagnosis_20260723]] · [[KNOWN_BUGS]] OPEN row "dreamer_srl eval curve mixes two estimators" (A3) · sibling logging doc [[DREAMER_SRL_LOGGING_RPPO_PARITY]]

---

## Context

When the live Dreamer agent (`dreamer_srl`) trains, it periodically pauses to *evaluate* itself — it plays a handful of test episodes, records a video, and writes the survival-step numbers to the experiment dashboard (WandB). Three related defects make that dashboard telemetry wrong or missing:

1. **The eval videos never show up in WandB.** The video is uploaded stamped with an *episode count* (a small number like 10,000), but the run's dashboard timeline is measured in *environment steps* (a huge number in the millions). WandB refuses any log that jumps *backward* on the timeline, so it silently throws the video away. The MP4 sits on disk; the dashboard panel stays empty. (This is Dreamer-only — the recurrent-PPO stack does not trip it, and we must keep it that way.)

2. **Every eval and stage-change event silently eats the next few training rows.** Those events log to WandB *without* specifying a timeline position, which nudges WandB's internal step counter forward; the next 1–3 real training rows then land on a step WandB has already passed and are dropped. Visible whenever the run uses a single environment.

3. **The headline eval curve blends two different estimators, and both replay the same fixed test worlds.** At each checkpoint the agent runs a small video pass (3 episodes) *and* — when enabled — a large stats pass (100 episodes), but **both write the same `Eval/MeanReward` / `Eval/MeanLength` keys at the same timeline position**, so the curve saw-tooths between a noisy 3-episode estimate and the real 100-episode one. Worse, both passes rebuild the exact same random seed, so the 3 video episodes are a bit-for-bit copy of the first 3 stats episodes (the video pass adds zero new information), and every checkpoint re-runs the identical fixed set of eval worlds.

This plan fixes all three by putting eval logging on one consistent env-step clock and by giving the small video pass its own metric namespace so it can never contaminate the authoritative survival-step curve. It is a telemetry-hygiene fix only: no change to what the agent learns or how it acts.

**Explicitly out of scope** (mentioned so it is clearly left untouched): the separate finding that `training.eval_stats_num_envs` is a mandatory-but-dead config key on the Dreamer path (the stats pass never routes through the batched rollout) — see [[findings_dreamer_main]] Finding 2. That is a different defect (a routing gap, not a telemetry gap) and is **not** addressed here.

---

## Analysis

### Item 1 — eval videos dropped (backward-step rejection)

Call chain: `dreamer_srl_main.py:1718-1726` (`_render_and_upload(...)`) → `eval.py:566-573` (`upload_video(..., step=checkpoint_pct)`) → `wandb_utils.py:90/113` (`wandb.log(log_dict, step=step)`).

`checkpoint_pct` is `total_episodes_completed` — an **episode** count. The run's WandB step axis is driven by the training loop, which logs with `step=policy_step` (an **env-step** count; `dreamer_srl_main.py:2042`, and the checkpoint save at `:1686`). By the first checkpoint (10,000 episodes) `policy_step` is already in the millions, so `upload_video(step=10_000)` is a **backward** step → WandB warns-and-drops. `eval/video` and `eval/checkpoint_episode` never appear.

**Why rPPO is safe and must stay safe:** the recurrent-PPO trainer (`train.py`) logs *step-less* throughout, so its internal counter stays tiny and its shared `upload_video(..., step=checkpoint_pct)` call at `evaluation_core.py:359` is a *forward* step. The two stacks reach the shared `upload_video` through **different call sites** — rPPO via `evaluation_core.py:359`, Dreamer via `eval.py:_render_and_upload`. Fixing the Dreamer call site alone leaves rPPO byte-identical. **We therefore do not touch `wandb_utils.py` or `evaluation_core.py`.**

Note: the existing comment at `dreamer_srl_main.py:1728-1731` blames the render subprocess (~13 s) for "advancing WandB's internal step counter." That is incorrect — a child subprocess cannot advance the parent process's WandB counter. The real cause is the episode-count `step=`. The fix removes the need for that comment.

### Item 2 — step-less sibling logs advance the counter past `policy_step`

Three sites log without an explicit `step=` while the rest of the loop uses `step=policy_step`:
- `dreamer_srl_main.py:1655-1662` — stage-transition log (`stage/*`, `Episode/Number`).
- `dreamer_srl_main.py:1733-1740` — video-pass `Eval/*` log.
- `dreamer_srl_main.py:1758-1765` — stats-pass `Eval/*` log.

A step-less `wandb.log` commits at (and nudges) WandB's internal counter; the next explicit-step training rows at the *same* `policy_step` are then rejected as backward. At `num_envs=1` (`policy_step` increments by 1 per iteration) this drops the next 1–3 training rows after every eval/stage event. Benign at the default `num_envs=16` (the gap between successive `policy_step` values absorbs the nudge), but still incorrect. All three sites already carry `timesteps: policy_step` in the payload for dashboard x-axis routing (`define_metric(..., step_metric="timesteps")`, `dreamer_srl_main.py:879`); adding the explicit `step=policy_step` makes the global counter monotone without changing any dashboard axis. `policy_step` is in scope at all three sites (loop variable, `:1230`/`:1336`).

Multiple `wandb.log` calls at the *same* `policy_step` within one loop iteration are allowed — WandB merges them into one history row. Across iterations `policy_step` only increases. So `step=policy_step` on every eval/stage/video log is monotone and safe.

### Item 3 — estimator mixing + fixed-seed reuse

Root cause (depth from [[findings_dreamer_main]] Finding 7 and [[findings_dreamer_eval_ckpt]]):
1. Both passes write the **identical** keys `Eval/MeanReward` / `Eval/MeanLength` at the **same** `timesteps=policy_step` with no pass label → the N=3 and N=100 estimators land at the same x → sawtooth.
2. Both passes rebuild `jax.random.PRNGKey(args.seed)` and split identically per episode, and the actor is deterministic argmax → the video pass's episodes are **bit-identical** to the first `eval_video_episodes` of the stats pass.
3. Every checkpoint reuses `seed=args.seed` → identical eval initial conditions across all checkpoints (a **paired**, variance-reduced eval).

**Critical config nuance that shapes the fix:** the default (`configs/train/default.yaml:49-50`) is **`video_during_training: true`, `stats_during_training: false`** — i.e. by default only the *video* pass runs and it is what currently populates `Eval/MeanReward` / `Eval/MeanLength` on the dashboard. The mixing sawtooth only appears when a user *also* enables the stats pass. Any fix must therefore keep `Eval/Mean*` populated in the default (video-only) configuration, or every existing default-config panel goes silent.

---

## Decided semantics for Item 3 (keys + seed policy)

**Keys — authoritative-estimator routing with video fallback:**

| Config | `Eval/MeanReward` / `Eval/MeanLength` written by | Video-pass scalar keys |
|---|---|---|
| stats **off** (default) | the **video** pass (fallback — unchanged from today) | *(none extra — video owns `Eval/Mean*`)* |
| stats **on** | the **stats** pass (N=100, authoritative) | `Eval/video/MeanReward` / `Eval/video/MeanLength` |

- *Rationale:* the authoritative `Eval/Mean*` keys are always attached to the **best available estimator** and are **never written by two passes at once**, killing the sawtooth. The default (video-only) run keeps writing `Eval/Mean*` exactly as before, so **no existing dashboard panel goes silent and no panel is renamed** — the only panel-visible change is the *new* `Eval/video/*` keys that appear solely when a user runs both passes. `Eval/video/*` is matched by the existing `define_metric("Eval/*", step_metric="timesteps")` glob (fnmatch `*` spans `/`), so no new `define_metric` is required (developer may add an explicit `define_metric("Eval/video/*", ...)` defensively — optional).

**Seed — keep the fixed per-checkpoint eval seed (`seed=args.seed` for both passes), documented as deliberate:**

- *Rationale:* holding the eval worlds fixed across checkpoints is the point of a training-time eval *curve* — it is a **paired comparison** so that checkpoint-to-checkpoint survival-step deltas reflect *learning*, not eval-world variance. Varying the seed per checkpoint would inject eval-world noise into the headline curve. We therefore keep the fixed seed. The residual "video episodes are a bit-identical subset of the stats episodes" property is now **harmless** (the two passes write *different* keys), and is in fact desirable for the video pass: the same eval worlds are rendered at every checkpoint, so the videos are directly comparable frame-for-frame across training. This is a variance-reduction design choice, recorded here so a future reader does not "fix" it. (Trade-off noted: the eval seed equals the training seed, so eval is not held-out in any train/test sense — this is unchanged by this plan and is a separate concern.)

---

## Implementation Plan

### Design

One consistent clock, one authoritative estimator:

1. **Thread `policy_step` into the video upload** and pass `step=policy_step` (not `step=checkpoint_pct`) to `upload_video`, keeping `episode=checkpoint_pct` so `eval/checkpoint_episode` still records the episode number. Fixes Item 1. `wandb_utils.py`/`evaluation_core.py` untouched → rPPO byte-identical.
2. **Add `step=policy_step`** to the stage-transition log and both eval logs. Fixes Item 2.
3. **Route the eval scalar keys** by a tiny pure helper so the video pass writes `Eval/Mean*` only when the stats pass is off, else `Eval/video/Mean*`; the stats pass always writes `Eval/Mean*`. Fixes Item 3. Keep `seed=args.seed` in both passes.

### File Changes

#### `src/algorithms/dreamer_srl/eval.py` — `_render_and_upload` (lines 508–575)

Add a `policy_step` parameter and use it as the WandB `step`. `checkpoint_pct` stays the `episode` label.

```python
# BEFORE (signature, 508-515):
def _render_and_upload(
    recordings_dir: str,
    results_dir: str,
    checkpoint_pct: int,
    fps: int,
    wandb_enabled: bool,
    quiet: bool = True,
) -> Optional[str]:

# AFTER:
def _render_and_upload(
    recordings_dir: str,
    results_dir: str,
    checkpoint_pct: int,
    fps: int,
    wandb_enabled: bool,
    policy_step: int,          # WandB step axis (env-step clock); checkpoint_pct is the episode label only
    quiet: bool = True,
) -> Optional[str]:
```

```python
# BEFORE (upload call, 564-571):
            from src.utils.wandb_utils import upload_video
            upload_video(
                consolidated,
                episode=checkpoint_pct,
                step=checkpoint_pct,
                caption=f'Episode {checkpoint_pct}',
                quiet=True,
            )

# AFTER:
            from src.utils.wandb_utils import upload_video
            upload_video(
                consolidated,
                episode=checkpoint_pct,   # -> eval/checkpoint_episode payload (episode label)
                step=policy_step,         # env-step clock: forward step, no longer dropped
                caption=f'Episode {checkpoint_pct}',
                quiet=True,
            )
```

Update the docstring `Args:` block to document `policy_step` and to note that `checkpoint_pct` is the episode *label* while `policy_step` is the WandB step axis. **Do not** alter `upload_video`'s own signature in `wandb_utils.py`.

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py`

**(a) New tiny pure helper (module scope, near the other eval helpers).** Makes the key-routing unit-testable and keeps the call sites one-liners.

```python
def _eval_scalar_prefix(is_video_pass: bool, stats_during_training: bool) -> str:
    """Namespace for a checkpoint-eval pass's scalar metrics.

    The many-episode stats pass owns the authoritative ``Eval/`` keys. The
    small video pass falls back to ``Eval/`` only when the stats pass is off
    (default config); otherwise it writes ``Eval/video/`` so it can never
    contaminate the authoritative survival-step curve. See
    docs/develop/active/dreamer/DREAMER_SRL_EVAL_TELEMETRY_FIX.md.
    """
    if is_video_pass and stats_during_training:
        return "Eval/video/"
    return "Eval/"
```

**(b) Stage-transition log (1655-1662)** — add explicit step:

```python
# BEFORE:
                        _wandb_stage.log({
                            "stage/index":         current_stage,
                            "stage/transition":    1,
                            "stage/buffer_cleared": _pre_size,
                            "Episode/Number":      total_episodes_completed,
                        })

# AFTER:
                        _wandb_stage.log({
                            "stage/index":         current_stage,
                            "stage/transition":    1,
                            "stage/buffer_cleared": _pre_size,
                            "Episode/Number":      total_episodes_completed,
                        }, step=policy_step)
```

**(c) Video-pass upload call (1718-1726)** — pass `policy_step`:

```python
# BEFORE:
                            _render_and_upload(
                                recordings_dir=_eval_result['recordings_dir'],
                                results_dir=results_dir,
                                checkpoint_pct=total_episodes_completed,
                                fps=viz_fps,
                                wandb_enabled=use_wandb,
                                quiet=args.quiet,
                            )

# AFTER:
                            _render_and_upload(
                                recordings_dir=_eval_result['recordings_dir'],
                                results_dir=results_dir,
                                checkpoint_pct=total_episodes_completed,
                                fps=viz_fps,
                                wandb_enabled=use_wandb,
                                policy_step=policy_step,
                                quiet=args.quiet,
                            )
```

**(d) Video-pass Eval log (1727-1740)** — replace the misleading comment, route the keys, add explicit step:

```python
# BEFORE:
                        # Log Eval/* to WandB (mirrors train.py:L2466-L2467)
                        # Note: no explicit step= kwarg — render subprocess (~13s) advances
                        # WandB's internal step counter during eval, so passing the old
                        # policy_step triggers "Tried to log to step N < current step M".
                        # define_metric("Eval/*", step_metric="timesteps") routes the
                        # X-axis via the "timesteps" key in the dict instead.
                        if use_wandb:
                            import wandb as _wandb
                            _wandb.log({
                                'Eval/MeanReward': _eval_result['mean_reward'],
                                'Eval/MeanLength': _eval_result['mean_length'],
                                'iteration':       iter_num,
                                'timesteps':       policy_step,
                            })

# AFTER:
                        # Log the video pass's scalars. When the stats pass is
                        # ALSO enabled it owns the authoritative Eval/Mean* keys,
                        # so the video pass writes Eval/video/* instead (no
                        # sawtooth). step=policy_step keeps the global WandB
                        # counter monotone; the render subprocess does NOT touch
                        # it (it is a child process).
                        if use_wandb:
                            import wandb as _wandb
                            _p = _eval_scalar_prefix(is_video_pass=True,
                                                     stats_during_training=stats_during_training)
                            _wandb.log({
                                f'{_p}MeanReward': _eval_result['mean_reward'],
                                f'{_p}MeanLength': _eval_result['mean_length'],
                                'iteration':       iter_num,
                                'timesteps':       policy_step,
                            }, step=policy_step)
```

**(e) Stats-pass Eval log (1758-1765)** — always authoritative `Eval/*`, add explicit step:

```python
# BEFORE:
                        if use_wandb:
                            import wandb as _wandb
                            _wandb.log({
                                'Eval/MeanReward': _stats_result['mean_reward'],
                                'Eval/MeanLength': _stats_result['mean_length'],
                                'iteration':       iter_num,
                                'timesteps':       policy_step,
                            })

# AFTER:
                        if use_wandb:
                            import wandb as _wandb
                            _wandb.log({
                                'Eval/MeanReward': _stats_result['mean_reward'],  # N=eval_stats_episodes (authoritative)
                                'Eval/MeanLength': _stats_result['mean_length'],
                                'iteration':       iter_num,
                                'timesteps':       policy_step,
                            }, step=policy_step)
```

#### `src/utils/wandb_utils.py` — **UNTOUCHED**

Listed only to state the constraint explicitly: `upload_video`'s signature and body must not change, so the rPPO path through `evaluation_core.py:359` stays byte-identical. No File Changes here.

#### `configs/` — **UNTOUCHED**

No new config keys. Routing is derived from the already-mandatory `training.stats_during_training`.

> Config-system / scripts-dependency-map maintenance contracts: **not triggered** — no config schema change, no file added/moved/renamed/deleted under `scripts/`. No update to `CONFIG_GUIDE.md`, `02_config_schema.md`, or `SCRIPTS_DEPENDENCY_MAP.md` required.

---

## Checkpoints

- [ ] After the `eval.py` change, grep confirms `upload_video(` in `_render_and_upload` passes `step=policy_step` and `episode=checkpoint_pct` (two distinct values).
- [ ] After the `dreamer_srl_main.py` changes, grep confirms **no** `wandb.log(` / `_wandb.log(` / `_wandb_stage.log(` in the eval/stage/checkpoint block (≈1650–1770) lacks an explicit `step=` kwarg.
- [ ] `_eval_scalar_prefix(True, False) == "Eval/"`, `_eval_scalar_prefix(True, True) == "Eval/video/"`, `_eval_scalar_prefix(False, *) == "Eval/"`.
- [ ] `python -c "import ast,sys; ast.parse(open('src/algorithms/dreamer_srl/dreamer_srl_main.py').read())"` and same for `eval.py` — files parse.
- [ ] `stats_during_training` is in scope at line ~1733 (resolved at `:615`) — confirm before referencing it in the video-pass log.

---

## Test plan

New tests under `tests/algorithms/dreamer_srl/` (existing home for `test_eval_rollout.py`, `test_eval_video_smoke.py`, etc.). Each must **fail on pre-fix code and pass after the fix**.

**T1 — `test_eval_telemetry.py::test_video_upload_uses_policy_step_clock` (unit, fast).**
Monkeypatch `subprocess.run` to return `rc=0` and create the expected consolidated MP4, and monkeypatch `src.utils.wandb_utils.upload_video` to capture kwargs. Call `_render_and_upload(..., checkpoint_pct=10_000, policy_step=1_000_000, wandb_enabled=True)`. Assert the captured `step == 1_000_000` (env-step clock) and `episode == 10_000` (episode label). *Pre-fix:* `_render_and_upload` has no `policy_step` param and passes `step=checkpoint_pct` → the call raises `TypeError` / the captured `step==10_000`, so the test fails. This is the Item-1 regression guard.

**T2 — `test_eval_telemetry.py::test_eval_scalar_prefix_routing` (unit, fast).**
Directly exercise the pure helper `_eval_scalar_prefix`: assert `(True, False)->"Eval/"`, `(True, True)->"Eval/video/"`, `(False, True)->"Eval/"`, `(False, False)->"Eval/"`. This is the Item-3 key-split guard (asserts the video pass no longer writes the authoritative `Eval/Mean*` keys when the stats pass is on). *Pre-fix:* the helper does not exist → import/collection error.

**T3 — `test_eval_telemetry.py::test_video_and_stats_share_seed_paired_eval` (unit, fast, no WandB).**
Call `dreamer_srl_eval_rollout(...)` twice on the same restored/tiny actor with the same `seed`, once with `num_episodes=3` and once with `num_episodes=8`, `render_video=False`. Assert `result3['episode_rewards'] == result8['episode_rewards'][:3]` (bit-identical prefix) and `result3['episode_lengths'] == result8['episode_lengths'][:3]`. This pins the **kept** seed policy: the deterministic paired-eval property is intentional and retained. (This is a property of the eval fn, so it passes both pre- and post-fix — it is a *lock* against a future accidental seed change, and documents the decision; flagged as such.)

**T4 — `test_eval_telemetry_wandb.py::test_eval_logs_monotone_explicit_steps` (stub-based integration, slower — may be marked `@pytest.mark.slow`).**
Because Items 1–2's log sites are inline in the main training loop, monotonicity is verified by driving a minimal run: monkeypatch `wandb.log` to append `(sorted(dict.keys()), step)` tuples, run `dreamer_srl` `main()` with a tiny single-env config (`num_envs=1`, a few hundred episodes, one checkpoint, `video_during_training=true`), then assert: (a) every recorded `step` is an `int` (no `step=None` among eval/stage/video/training rows), (b) the `step` sequence is monotone non-decreasing across the whole run, (c) the `eval/video` upload row and the `Eval/Mean*` row carry the same `step` as the checkpoint's `policy_step`. **Stated limitation:** this exercises real WandB step semantics only through a stub — a stub cannot reproduce WandB's actual backward-step *drop*, so it asserts the *inputs* (explicit, monotone steps) that the fix guarantees, not the live drop behavior. If running `main()` is too heavy for the suite, keep T1–T3 (which fully cover Items 1 and 3 at unit level) and downgrade T4 to a documented manual smoke check (one real `num_envs=1` run; confirm `eval/video`, `eval/checkpoint_episode`, and `Eval/MeanReward` all appear on the dashboard and no training rows are missing after a checkpoint).

Add `test_eval_telemetry.py` (and optionally `test_eval_telemetry_wandb.py`) to the dreamer_srl test package; both live entirely under `tests/` (no `scripts/` change → SCRIPTS_DEPENDENCY_MAP untouched).

---

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the developer agent. Record: actual diff summary, any deviation from
     the routing/seed decisions above, and — since these are wandb.log-timing changes
     that do not touch the hot loop — a one-line note on whether step throughput
     changed (expected: none; the added kwargs and helper are O(1) per checkpoint). -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/algorithms/dreamer_srl/eval.py` | `_render_and_upload` gains `policy_step`, uploads on env-step clock | | |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | helper + 4 log sites on `step=policy_step`, video/stats key split | | |
| `src/utils/wandb_utils.py` | UNTOUCHED (rPPO byte-identical) | | |
| `tests/algorithms/dreamer_srl/test_eval_telemetry*.py` | T1–T4 added | | |

**Conclusion**: [one-line summary]
</content>
</invoke>
