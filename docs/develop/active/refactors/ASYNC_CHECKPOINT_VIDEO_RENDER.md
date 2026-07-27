---
title: "Remove the blocking checkpoint-video render stall (dreamer_srl + rPPO)"
topic: refactors
status: active
created: 2026-07-27
last_updated: 2026-07-27
---

# Remove the blocking checkpoint-video render stall from the training loops

> **Status**: PLANNED
> **Opened**: 2026-07-27
> **Related**: [[dreamer_srl_h1_speed_investigation]] · [[SYNTHESIS_20260727]] ·
> [[EXPERIMENT_EVAL_DURING_TRAINING]] (the in-repo async idiom this plan ports)

---

## Context

Both of our training loops — the DreamerV3 trainer ("dreamer_srl") and the
recurrent-PPO trainer — periodically **stop training entirely to make a video**.
Every N episodes they save a checkpoint, play a few evaluation episodes, and then
sit idle while a separate CPU process draws every step of those episodes as a
matplotlib frame and encodes an MP4 for the WandB dashboard. The training process
*waits* for that render to finish (a blocking `subprocess.run`), so the GPU does
nothing for 1–4 minutes per event.

Measured on the live runs (2026-07-27): the two Dreamer arms currently training
lose **21% and 11% of their total wall-clock** to these stalls, and — the surprise
this plan documents — the recurrent-PPO runs, which were assumed to have solved
this, lose **19–25%** the same way. rPPO uses the *identical* blocking render
code; it only *looked* healthy because its videos fire 100× less often per
episode, and its far-higher episode rate cancels most of that advantage.

The fix: make the render fire-and-forget, using the asynchronous
dispatch/poll/drain pattern rPPO **already uses** for its during-training
behavior-probe evaluation — a proven in-repo idiom. Evaluation recordings
(`.rec.gz`) and checkpoints keep being written exactly as now; only the MP4
drawing and WandB upload move off the critical path. Expected recovery:
**+16–31% training throughput** on both algorithms, for future runs only (no
live run is touched). Users will see identical videos on WandB, appearing ~1–4
minutes later than the checkpoint that produced them.

---

## Analysis

### A. Measured stall cost — Dreamer vs rPPO (live runs, ground truth)

All numbers parsed from the live runs' own stdout logs and results directories on
2026-07-27 (method in Appendix A). "Stall" = wall-clock where the training
progress bar is frozen at a checkpoint event.

| Run (live) | Algorithm | Cadence (`checkpoint_frequency`, episodes) | Stall events | Stall duration (median/mean) | **Wall-clock lost** | Stall breakdown (median) |
|---|---|---|---|---|---|---|
| `dsrl_b03_XS_bins6_rr0p0625` (node 114, launched today) | dreamer_srl | 1,000 | 84 in 10.8 h (~1 per 7.7 min) | 102 s / — excess | **21.1%** | Orbax save 10 s · eval rollout 2–7 s · **render+upload 15–51 s (mean), up to 141 s** |
| `dsrl_b03_XS_bins6_rr0p25` (node 114, launched today) | dreamer_srl | 1,000 | 31 in 10.7 h (~1 per 21 min) | 151 s / — excess | **10.7%** | Orbax save 24 s · rollout ~0 s · render+upload 50–93 s |
| `dsrl_b03_M_dp1` / `dsrl_b03_XS_dp1` (2026-07-26, from [[dreamer_srl_h1_speed_investigation]]) | dreamer_srl | 1,000 | ~156–175 in 21 h | ~165 s mean | **34% / 44%** | render-dominant (same code path) |
| `rppo_b02_mc_dp1_n107` (node 107) | recurrent PPO | 100,000 | 123 in 35.2 h (~1 per 16 min) | 196 s / 194 s | **18.9%** | eval rollout 13 s · **render 183–184 s** (94% of the stall) |
| `rppo_b03_mc_dp1_n110` (node 110) | recurrent PPO | 100,000 | 257 in 35.6 h (~1 per 8.3 min) | 142 s / 127 s | **25.4%** | same mechanism; b03's shorter episodes double the event rate |

Key observations:

1. **rPPO has NOT solved this.** Its in-training video path is the same blocking
   `subprocess.run` (`src/utils/evaluation_core.py:346`), which the Dreamer code
   was ported from (`src/algorithms/dreamer_srl/eval.py:517` says "near-verbatim
   port of evaluation_core.py:L289-L321"). rPPO's lower *relative* cost is pure
   cadence arithmetic — and its per-event stall is actually *longer* (~195 s vs
   Dreamer's 43–119 s mean today) because its trained agents survive longer, so
   there are more frames to draw.
2. **Stall frequency scales with episode rate, not time** — the faster arm
   (`rr0p0625`, ~2× the episode rate of `rr0p25`) stalls 2.7× as often per hour.
   This is exactly the user-observed pattern, now quantified. It also means the
   cost *worsens as agents improve* (longer survival → more frames per video).
3. **The GPU is idle for the whole render.** The parent process is blocked inside
   `subprocess.run`; the child is explicitly pinned to CPU
   (`JAX_PLATFORMS=cpu`, `eval.py:554`, `evaluation_core.py:343`). This matches
   the 15–70%-never-pinned GPU utilization documented in
   [[dreamer_srl_h1_speed_investigation]] §1.
4. **The render is ~90%+ of the stall; the eval rollout is small.** Filesystem
   mtimes (run_meta.pkl → last `.rec.gz` → MP4) split the phases: rollout is
   2–13 s; render is 15–184 s. The remaining blocking item is the Orbax
   checkpoint save (10–26 s on Dreamer with `max_checkpoints_to_keep: 1000000`
   on NAS) — out of scope here (checkpoints must keep being written), noted as
   follow-up.

### B. Code trace — what blocks, exactly

**Dreamer** (`src/algorithms/dreamer_srl/dreamer_srl_main.py`):
- `:1776–1800` — Orbax checkpoint save (blocking, in scope only as context).
- `:1805–1824` — "Commit F" checkpoint-triggered eval: `dreamer_srl_eval_rollout`
  (`eval.py:28–204`, single-env eager Python loop) plays
  `eval_video_episodes: 3` episodes and writes the `.rec.gz` recordings +
  `run_meta.pkl` **synchronously** — these artifacts are consumed by the
  behavior-measure pipeline and `dreamer_srl_probe_eval.py` and must not change.
- `:1825–1834` — `_render_and_upload(...)` (`eval.py:508–582`): builds a command
  for `scripts/eval/render_recordings.py` and runs it via **blocking
  `subprocess.run`** (`eval.py:556–561`); on success uploads the MP4 to WandB
  from the parent (`eval.py:569–578`, `step=policy_step` per the
  backward-step-drop fix, commit `39f851b`).
- Gating config (live values): `training.video_during_training: true`,
  `training.checkpoint_frequency: 1000`, `testing.auto_render_after_eval: true`,
  `visualization.fps: 5`.

**rPPO** (`train.py`):
- `:2388–2399` — video pass: `evaluate_jax_checkpoint(render_video=True, num_envs=1)`.
- `src/utils/evaluation_core.py:326–360` — inside that function, after the
  rollout writes recordings, the same render command runs via **blocking
  `subprocess.run`** (`:346`), then uploads to WandB (`:357–360`, note: with
  `step=checkpoint_pct`, i.e. episode count — a per-algo difference this plan
  preserves as-is; see Risks).
- Live cadence: `training.checkpoint_frequency: 100000` episodes at ~104–210
  episodes/s → one ~195 s stall every 8–16 min.

**Renderer** (`scripts/eval/render_recordings.py`):
- Spawns a `ProcessPoolExecutor` with `--workers` defaulting to
  **`cpu_count − 1`** (`:142`) — on a shared node this competes with the
  co-resident training processes' host-side Python (a contention source the H1
  doc flagged during co-residency).
- Per frame: one full matplotlib figure (`render_jax_state`,
  `src/environment/renderer.py:355`, `figsize=(14,10)`), ~0.12 s/frame measured
  from rPPO's ~1,500-frame events.

**The in-repo async idiom to port** (rPPO's experiment-eval, `train.py`):
- `_maybe_dispatch_experiment_eval` (`:256–298`) — `subprocess.Popen`, log file,
  **at most one concurrent child; skip-if-busy** (skipped work is
  offline-recoverable, so never a data loss).
- `_poll_and_log_experiment_results` (`:301–358`) — cheap once-per-iteration
  poll, every step failure-isolated; parent is the **sole WandB writer**.
- `_drain_experiment_results` (`:361–385`) — bounded wait at exit (300 s normal
  / 10 s on Ctrl-C); on timeout the child is deliberately **left running**
  (on-node CPU-only, finite work) rather than killed.
- Call sites: dispatch at the checkpoint block, poll at `train.py:2313–2320`,
  drain at `:2465–2472` before `wandb.finish()`.

### C. Corrections to [[dreamer_srl_h1_speed_investigation]] (verified in code)

1. **The "300-DPI" claim is wrong.** `visualization.video_dpi: 300` is a **dead
   config key** — nothing in `src/` or `scripts/` reads `video_dpi`. The render
   worker calls `render_jax_state()` without a `dpi` argument
   (`render_recordings.py:120–128`), so frames render at the function default
   `dpi=100` (`renderer.py:355`). Consequence: "lower the DPI" is not an
   available lever (it's already 100), and the dead key should not be cited as a
   cost driver.
2. **The eval rollout is a minor cost, not a co-headliner**: 2–13 s of a 43–195 s
   event (mtime-split, Appendix A), so routing the video pass through the jitted
   `dreamer_srl_eval_rollout_batched` is *not* needed for this fix (and would
   change recorded trajectories — see Non-goals).
3. The headline finding (render-blocking stall, 34–44% on the dp1 runs) is
   **confirmed** by two independent methods on two new runs today.

---

## Options considered (ranked)

| # | Option | Expected recovery | Risk | Effort | Artifacts changed? |
|---|---|---|---|---|---|
| **1 (recommended)** | **Async render: port rPPO's dispatch/poll/drain idiom** to the render step, for BOTH algorithms. `subprocess.Popen` + at-most-one-concurrent + skip-if-busy + per-iteration poll (parent uploads MP4 to WandB on completion) + bounded drain at exit. | Dreamer: 21.1%→~8% stall (+16% throughput) on today's XS arm; ~34%→~8% (~1.4×) on the dp1-style M runs. rPPO: 18.9%→~1.5% (+21%), 25.4%→~2% (+31%). | Low — no training semantics touched; child already runs CPU-pinned in its own process; idiom is battle-tested in this repo. Orphan-at-exit policy matches the existing, documented behavior. | Medium (~1 day + tests) | **No.** `.rec.gz`, `run_meta.pkl`, checkpoints byte-identical and same cadence. MP4s identical content, appear 1–4 min later. WandB video arrives at a slightly later step (still forward-monotone). |
| 2 | **Defer rendering entirely offline** — set `testing.auto_render_after_eval: false` for training runs (recordings still written; `eval.py:1825` and `evaluation_core.py:328` already gate on it), render post-run with the existing `render_recordings.py`. | Same as #1 plus removes the dispatch residue (~0). **Zero code.** | None to artifacts; **loses live videos on WandB during training** (a real monitoring regression — checking mid-run behavior via video is current practice). Also `testing.*` is shared with standalone eval, so the flip must be per-experiment-config, not in `configs/evaluation/default.yaml`. | Zero (config-only, per launch) | Recordings/ckpts unchanged; no in-training MPGs/WandB videos. |
| 3 | **Decouple render cadence from checkpoint cadence** — new key `training.render_every_n_checkpoints` (default 1 = today's behavior): rollout + recordings still every checkpoint, MP4 only every Nth. | Divides stall count by N but each stall still blocks (~50–195 s). Partial. | Low | Low | Recordings/ckpts unchanged; fewer MP4s. |
| 4 | Reduce per-event render cost (fewer `eval_video_episodes`, fps, DPI) | Small. DPI is already 100 (dead key, §C.1); fps doesn't change frame count; dropping episodes 3→1 **changes the `.rec.gz` set per checkpoint** → violates the artifact constraint. | Medium (artifact drift) | Trivial | **Yes** (episode-count variant) — rejected. |
| 5 | Swap the video pass to the jitted batched eval rollout | Saves only the 2–13 s rollout share; different RNG convention → **different recorded trajectories** (documented in `eval.py:314–348`). | Medium | Medium | **Yes** — rejected for this plan. |

**Recommendation: Option 1**, with Option 3's knob folded in for free (it falls
out of the same dispatch function; default 1 keeps today's cadence), and Option 2
noted to `experiment-designer` as the zero-code stopgap for any launch that
happens before Option 1 lands. Options 4/5 rejected (artifact drift).

**Scope: both algorithms.** rPPO bleeds 19–25% through the identical code path;
fixing Dreamer only would leave the larger fleet-wide cost in place. The change
is one shared helper + two thin call-site edits, so the marginal cost of covering
rPPO is small.

### Non-goals (explicit)

- The `.rec.gz` recordings, `run_meta.pkl`, and Orbax checkpoints keep being
  produced **synchronously, on the same cadence, by the same code** — the
  behavior-measure pipeline and `scripts/eval/dreamer_srl_probe_eval.py` consume
  them.
- The eval rollout implementation is untouched (no batched-rollout swap).
- The Orbax save time (10–26 s per event on Dreamer, keep-all on NAS) is out of
  scope; flagged as a follow-up candidate (Orbax supports async save).
- Standalone/offline eval (`evaluation.py`, `scripts/eval/*`) stays blocking —
  a script must not exit before its render finishes.
- No change to the live runs on nodes 114 / 107 / 108 / 110 / 112.

---

## Proposed Solution

Port rPPO's experiment-eval async pattern (`train.py:256–385`) into a small
shared helper, and switch both in-training render call sites to it.

### New module: `src/utils/async_render.py` (~130 lines)

State dict (per run): `{"proc": None, "log_f": None, "pending": None}` where
`pending = {"mp4_path", "checkpoint_pct", "recordings_dir"}`.

1. `dispatch_render(state, recordings_dir, results_dir, checkpoint_pct, fps,
   workers, quiet) -> bool`
   - If `state["proc"]` is alive → **skip** (print one line naming the skipped
     checkpoint and that `render_recordings.py <recordings_dir>` can backfill it
     offline; recordings are on disk, so nothing is lost). Mirrors
     `_maybe_dispatch_experiment_eval`'s skip-if-busy.
   - Else `subprocess.Popen` the exact command `_render_and_upload` builds today
     (`--concat --skip-existing --cleanup-per-episode --fps N`, child env
     `JAX_PLATFORMS=cpu` **plus** `CUDA_VISIBLE_DEVICES=""` — adopt the
     belt-and-braces GPU isolation from `_experiment_eval_env`,
     `train.py:240–253`), stdout/stderr → `<results_dir>/videos/render_<ckpt>.log`.
   - If `workers` is not None, append `--workers <workers>` (bounds CPU
     contention with the co-resident training process — important *after* this
     fix, because the render now runs concurrently with training instead of
     during a stall).
2. `poll_render(state, wandb_enabled, step, upload_step_mode) -> None`
   - Cheap (`proc.poll()`), called once per training iteration, every branch
     wrapped in its own try/except (failure-isolated like
     `_poll_and_log_experiment_results`; a render failure must never crash
     training).
   - On completion rc==0 and MP4 exists → `upload_video(...)` from the parent
     (parent remains the sole WandB writer). Step stamping preserves each
     algorithm's existing semantics: Dreamer passes the *current* `policy_step`
     (forward-monotone, same guarantee as commit `39f851b`); rPPO passes
     `checkpoint_pct` exactly as `evaluation_core.py:359` does today.
   - On rc!=0 → print warning with the log-file path (recordings preserved).
3. `drain_render(state, interrupted=False) -> None`
   - Bounded `proc.wait()`: 300 s normal exit / 10 s interrupted (module
     constants, same values as the experiment-eval drain). On timeout, **leave
     the child running** — deliberate, matching `_drain_experiment_results`'s
     documented policy: the render is finite (~1–4 min), CPU-only, and
     `--skip-existing` makes any later re-render harmless. This is the same
     orphan class already known and accepted for SIGINT-killed runs (memory:
     `docs/memory/memories/cluster_ops/20260513_0018_train_py_orphan_render_workers_on_sigint.md`);
     after implementation, ask `bug-curator` to note the new dispatch site on
     that record.
   - Close `log_f`.

### Call-site changes

**Dreamer** (`src/algorithms/dreamer_srl/dreamer_srl_main.py`):
- Read new config keys next to the existing eval-config block (`:651–674`), via
  `get_mandatory` for the switch and `get` for the worker count (declared in
  YAML, see Config surface).
- At `:1825–1834`: if `training.async_video_render` and this checkpoint index is
  a multiple of `training.render_every_n_checkpoints` → `dispatch_render(...)`;
  else if the flag is false → current blocking `_render_and_upload(...)`
  unchanged (kill-switch).
- Per-iteration poll: one `poll_render(...)` call near the experiment-eval-style
  bottom of the loop (after the pbar update block).
- Drain: `drain_render(state)` immediately before the final-log section
  (`:2187`), ahead of `wandb.finish()` (`:2201`).

**rPPO** (`train.py` + `src/utils/evaluation_core.py`):
- `evaluate_jax_checkpoint(...)` gains keyword-only
  `async_render_state: dict | None = None` (default None → today's blocking
  behavior for every existing caller, including standalone eval). When not None
  and `render_video and testing.auto_render_after_eval` → `dispatch_render`
  instead of the blocking block at `:326–360`, and skip the inline upload.
- `train.py` video-pass call (`:2388–2399`) passes the state dict (created next
  to `experiment_state`, `:852`) only when `training.async_video_render` is
  true.
- Poll call added beside the experiment-eval poll (`:2313–2320`); drain beside
  `_drain_experiment_results` (`:2465–2472`), sharing its `interrupted` flag.

### Config surface

Existing keys reused (no new parallel keys invented): `testing.auto_render_after_eval`
(still the master on/off for rendering at all), `training.video_during_training`,
`training.eval_video_episodes`, `visualization.fps`, `testing.seed` — all
unchanged. Note `visualization.video_dpi` is dead (§C.1); this plan does **not**
revive it — the plan's cleanup is limited to a comment marking it dead where it
appears in configs (removal is a separate decision).

New keys (all three files per the self-contained convention documented in
`configs/train/dreamer_srl.yaml:27–34`):

| Key | `configs/train/default.yaml` (shared fallback, DQN/DRQN/PPO) | `configs/train/dreamer_srl.yaml` | `configs/train/recurrent_ppo.yaml` |
|---|---|---|---|
| `training.async_video_render` | `false` (preserves current behavior for out-of-scope algorithms) | `true` | `true` |
| `training.render_every_n_checkpoints` | `1` (= today's cadence) | `1` | `1` |
| `training.render_workers` | `null` (= renderer's own default, `cpu_count−1`) | `8` | `8` |

`true` for the two in-scope algorithms is a deliberate behavior change — it *is*
the fix, and it must land for future runs without per-launch config edits; the
`false` setting is the documented kill-switch. `render_workers: 8` bounds the
now-concurrent render's CPU footprint on shared nodes (render duration grows,
but it is off the critical path and cadence ≫ duration; skip-if-busy caps
pile-up at zero).

None of these are critical-settings-registry entries
(`docs/environment/CONFIG_CRITICAL_SETTINGS.md` registry untouched — verified:
no `video_*`/`checkpoint_frequency` rows are being changed by this plan), so no
registry change-log entry is required. Adding keys **is** a schema change →
`docs/environment/CONFIG_GUIDE.md` + `docs/environment/02_config_schema.md`
must be updated in the same change (Maintenance Contract).

---

## File Changes

| # | File | Change |
|---|---|---|
| 1 | `src/utils/async_render.py` | **NEW** — `dispatch_render` / `poll_render` / `drain_render` + drain-timeout constants, per Proposed Solution. Docstring must name `train.py:256–385` as the source idiom and state the leave-orphan-on-timeout policy. |
| 2 | `src/algorithms/dreamer_srl/dreamer_srl_main.py` | Read 3 new keys (`:651–674` block); branch at `:1825–1834` (dispatch vs legacy blocking call); per-iteration `poll_render` after the pbar-update block; `drain_render` before `:2187`. |
| 3 | `src/algorithms/dreamer_srl/eval.py` | `_render_and_upload` untouched (remains the blocking fallback). Add a cross-reference comment pointing at `async_render.py`. |
| 4 | `src/utils/evaluation_core.py` | `evaluate_jax_checkpoint` gains `async_render_state=None`; when set, `dispatch_render` replaces the `:326–360` blocking block (upload moves to the parent's poll). All existing callers unaffected (default None). |
| 5 | `train.py` | Create render state next to `:852`; pass into the `:2388–2399` video pass when `training.async_video_render`; poll beside `:2313–2320`; drain beside `:2465–2472` (reuse `training_interrupted`). |
| 6 | `configs/train/default.yaml` | Declare the 3 new keys (values per table) with FALLBACK-comment style matching the file's existing convention. |
| 7 | `configs/train/dreamer_srl.yaml` | Declare the 3 new keys (`true` / `1` / `8`). |
| 8 | `configs/train/recurrent_ppo.yaml` | Declare the 3 new keys (`true` / `1` / `8`). |
| 9 | `docs/environment/CONFIG_GUIDE.md` + `docs/environment/02_config_schema.md` | Document the 3 new keys (Maintenance Contract). |
| 10 | `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | Update `render_recordings.py` caller edges: `evaluation_core.py` and `dreamer_srl/eval.py` direct-call rows joined by `src/utils/async_render.py` (Popen path from both trainers). No `scripts/` file is added/moved/renamed. |
| 11 | `tests/training/test_async_render_dispatch.py` | **NEW** — see Test Plan. |

No other file may change. In particular: no change to `render_recordings.py`,
no change to the eval rollouts, no change to checkpoint saving.

## Test Plan

1. **Unit — non-blocking dispatch** (`tests/training/test_async_render_dispatch.py`):
   monkeypatch the render-script path to a stub that sleeps 5 s then writes the
   MP4 path; assert `dispatch_render` returns in <1 s; assert `poll_render` is a
   no-op while running and detects completion after; assert the WandB upload
   hook fires exactly once (mock `upload_video`).
2. **Unit — skip-if-busy**: dispatch twice while the stub sleeps; assert the
   second returns `False` without spawning (one child PID only).
3. **Unit — drain bounded**: stub sleeps 30 s, drain with a 2 s test timeout →
   returns in ~2 s with the child still alive (leave-orphan policy), log file
   closed.
4. **Unit — blocking fallback**: `evaluate_jax_checkpoint(async_render_state=None)`
   and Dreamer with `async_video_render: false` take the byte-identical legacy
   path (assert `subprocess.run` called — mock).
5. **Existing suite** passes untouched.

## Verification Plan (how we prove the stall is gone — without touching live runs)

1. **Smoke run A/B on a free node** (pick via `gpu-status`; NOT 114/107/108/110/112):
   dreamer_srl XS smoke config, `--episodes` small, `checkpoint_frequency`
   overridden low (e.g. 50) so ~10 render events fire in ~15 min. Run once with
   `async_video_render: false`, once `true`, same seed.
   - Parse both logs with the gap/segment method of Appendix A: the async run's
     stall segments must drop to the Orbax-save + rollout floor (≤ ~15 s/event);
     the sync run reproduces the render-length stalls.
   - **Artifact parity**: `recordings/**/*.rec.gz` checksums and checkpoint dir
     listings identical between the two runs (same seed ⇒ same eval
     trajectories; render is downstream of recording). MP4s present in both.
   - WandB (offline mode acceptable): video logged once, forward step.
2. **rPPO spot-check**: one short `train.py` run with a low
   `--checkpoint-frequency`, same A/B, same parser.
3. **Speed check** per the Verification Protocol: steady-state s/iter unchanged
   (the poll is a `proc.poll()` — nanoseconds); cumulative SPS improves by
   roughly the measured stall share. >5% steady-state regression ⇒ discuss;
   >15% ⇒ blocker.
4. Confirm no orphan child after a *normal* exit (drain succeeded) via `pgrep -f
   render_recordings` on the test node.
5. Post-landing (future live launch, owned by `experiment-analyzer`): compare
   the new arms' tqdm-gap share against the table in §A.

## Checkpoints

- [ ] 1. `async_render.py` written + unit tests 1–4 green
- [ ] 2. Dreamer call sites wired; smoke run A/B done; artifact parity confirmed
- [ ] 3. rPPO call sites wired; rPPO A/B done
- [ ] 4. Configs + CONFIG_GUIDE/schema + SCRIPTS_DEPENDENCY_MAP updated
- [ ] 5. Full test suite green; speed numbers recorded in the Implementation Report
- [ ] 6. `bug-curator` notified: annotate the orphan-render-workers record with the new dispatch site

## Risks & notes for the implementer

- **Do not restart or touch the live runs** (nodes 114, 107, 108, 110, 112). The
  fix is for future launches only.
- **WandB single-writer**: the upload must stay in the parent (poll site). Never
  let the child touch WandB.
- **rPPO upload-step semantics**: `evaluation_core.py:359` stamps
  `step=checkpoint_pct` (episode count) while Dreamer stamps `policy_step` (env
  steps, fix `39f851b`). This plan **preserves each as-is**. The rPPO convention
  may share the backward-step-drop hazard Dreamer had — out of scope; if
  confirmed, open a separate bug via `bug-curator`.
- **Skip-if-busy is expected to be rare** (render 1–4 min vs cadence 8–21 min)
  but is the safety valve if a future config tightens cadence; a skipped
  checkpoint's video is offline-recoverable from its recordings.
- **Ctrl-C on Dreamer**: the Dreamer driver has no `KeyboardInterrupt` handler
  around its loop; on Ctrl-C the drain is skipped and the child finishes on its
  own (accepted, same class as the known orphan behavior).

---

## Implementation Report

_To be filled by `developer`._

## Verification Report

_To be filled by `senior-developer` after implementation._

---

## Appendix A — Measurement method (reproducible)

All parsing used the conda interpreter
(`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`) on the NAS-shared
stdout logs; no process on any lab node was touched.

1. **rPPO stall detection** (`logs/20260726_053744.log` = b02/n107,
   `logs/20260726_051206.log` = b03/n110): normalize `\r`→`\n`; parse tqdm lines
   `| <episodes>/100000000 [H:MM:SS<`; rPPO's bar updates ~every second during
   training, so any ≥10 s jump in elapsed between consecutive updates is a stall;
   sum the jumps. (123 gaps / 23,906 s / 18.9% and 257 / 32,548 s / 25.4%.)
2. **Dreamer stall detection** (`logs/20260727_052328_dsrl_b03_XS_bins6_*.log`):
   Dreamer's bar refreshes only every 100 iterations (~22–50 s), so the ≥10 s
   rule misfires; instead parse `(elapsed, iter=N)` pairs, compute per-segment
   s/iter, take the median as baseline, and count as stall the excess time of
   segments running >2× baseline. (84 segments / 8,181 s / 21.1% and 31 /
   4,128 s / 10.7%.)
3. **Phase split within a stall**: bracket the `[CHECKPOINT] Saving` /
   `[CHECKPOINT] Saved.` / `video pass` / `[eval] ep 3/3` markers with the
   nearest tqdm elapsed stamps (each `pbar.write` forces a redraw) → Orbax-save
   and (rollout+render) durations.
4. **Rollout-vs-render split**: per checkpoint `<ckpt>`, mtime of
   `recordings/<ckpt>/run_meta.pkl` (rollout start) → newest
   `recordings/<ckpt>/*.rec.gz` (rollout end) → `videos/eval_<ckpt>.mp4` (render
   end).
