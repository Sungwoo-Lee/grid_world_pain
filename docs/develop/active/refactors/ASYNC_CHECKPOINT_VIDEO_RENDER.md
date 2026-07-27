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
     `docs/llm_wiki/entries/cluster_ops/20260513_0018_train_py_orphan_render_workers_on_sigint.md`);
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

- [x] 1. `async_render.py` written + unit tests 1–4 green — 11/11 tests in `tests/training/test_async_render_dispatch.py`, incl. both reviewer amendments (drain-final-poll, no-overwrite-on-late-finish)
- [x] 2. Dreamer call sites wired; smoke run A/B done; artifact parity confirmed — Dreamer end-to-end async covered by the new `test_eval_logs_monotone_async_render` (real `main()` loop, offline WandB: dispatch → drain → upload, monotone steps); artifact-parity A/B was run on the rPPO trainer locally (see Implementation Report — GPU-node Dreamer A/B left to the Verification Plan, per the no-lab-nodes constraint)
- [x] 3. rPPO call sites wired; rPPO A/B done — local CPU A/B, same seed: recordings decompressed-byte identical, checkpoint listings identical, shared MP4 byte-identical, training-loop time 84 s → 42 s
- [x] 4. Configs + CONFIG_GUIDE/schema + SCRIPTS_DEPENDENCY_MAP updated
- [x] 5. Full test suite green; speed numbers recorded in the Implementation Report
- [ ] 6. `bug-curator` notified: annotate the orphan-render-workers record with the new dispatch site — **open**: `developer` cannot spawn agents; hand to top-level Claude / `senior-developer` at verification

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

**Status: implemented, all tests green, ready for verification.** In plain
terms: both trainers now hand the checkpoint-video MP4 render to a background
CPU process instead of freezing training for it; the training loop polls once
per iteration and uploads the finished video itself. Recordings and checkpoints
are written exactly as before (verified byte-level, see Smoke A/B below), and a
config switch (`training.async_video_render: false`) restores the old blocking
behavior unchanged. No live run, lab node, or hot-read script was touched.

### Files changed (file-by-file)

| File | What was done |
|---|---|
| `src/utils/async_render.py` | **NEW** (~270 lines incl. docs). `new_render_state` / `dispatch_render` / `poll_render` / `drain_render` + `RENDER_DRAIN_TIMEOUT_S=300` / `RENDER_DRAIN_TIMEOUT_S_INTERRUPTED=10`. Docstring names `train.py:256–385` as the source idiom and states the leave-orphan-on-timeout policy. Child env adopts `_experiment_eval_env`'s belt-and-braces GPU isolation (`JAX_PLATFORMS=cpu` + `CUDA_VISIBLE_DEVICES=""`). The `render_every_n_checkpoints` gate lives inside `dispatch_render` (an `every_n` param + `state["ckpt_index"]`), per the plan's "falls out of the same dispatch function". |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | 3 new config reads next to the eval-config block; module-level import of the helper; `async_render_state` created before the pbar; dispatch-vs-legacy branch at the Commit-F video pass (kill-switch keeps the byte-identical `_render_and_upload` call); per-iteration `poll_render` after the step-log block; `drain_render` after `pbar.close()`, before §13/`wandb.finish()`. |
| `src/algorithms/dreamer_srl/eval.py` | Cross-reference comment added to `_render_and_upload`'s docstring (now the blocking kill-switch fallback). No behavior change. |
| `src/utils/evaluation_core.py` | `evaluate_jax_checkpoint(..., async_render_state=None)`; when set, `dispatch_render` replaces the blocking block (`upload_step_mode='checkpoint_pct'`, preserving rPPO's existing step stamping); the entire legacy blocking block is preserved verbatim in the `else` branch (re-indented only). All other callers (`evaluation.py:429`, `main.py:92`, the stats pass) unchanged via the None default. |
| `train.py` | Module import; `async_video_render` switch + `async_render_state` created next to `experiment_state`; video pass passes the state dict only when the switch is true; `poll_render` beside the experiment-eval poll; `drain_render` beside `_drain_experiment_results` (shares `training_interrupted`). |
| `configs/train/default.yaml` | 3 new keys (`false` / `1` / `null`) with FALLBACK-comment documentation per the file's convention. |
| `configs/train/dreamer_srl.yaml`, `configs/train/recurrent_ppo.yaml` | 3 new keys (`true` / `1` / `8`), self-contained-convention comments. |
| `configs/visualization/default.yaml` | Comment marking `video_dpi` as a dead key (plan §C.1). The other occurrence is in an **archived** experiment config (`configs/environment/experiment/archive/hypervigilance/testbed_cellC_native_v2.yaml`) — left untouched (archives are historical snapshots). |
| `docs/environment/CONFIG_GUIDE.md` | §7: new 3-key table (Maintenance Contract). |
| `docs/environment/02_config_schema.md` | `training:` stanza added to the YAML top-level structure listing the 3 keys, pointing at CONFIG_GUIDE §7. |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | `render_recordings.py` caller edges: third `src/` subprocess row (`async_render.py:56` Popen), §2/§3 rows and the move-rule updated; stale line numbers for the two blocking sites refreshed (`evaluation_core.py:365`, `eval.py:546`). |
| `tests/training/test_async_render_dispatch.py` | **NEW** — 11 tests, see below. |
| `tests/algorithms/dreamer_srl/test_eval_telemetry_wandb.py` | **DEVIATION (flagged, see below)** — reworked into a shared driver + 2 tests: the original sync-coupling assertion pinned to `async_video_render: false` (kill-switch path), plus a new async-mode end-to-end test. |

### The four 🟡 plan-reviewer amendments

1. **Drain must poll (finding 1)** — implemented: `drain_render` ends with a
   `poll_render` call after the bounded wait (mirrors `train.py:381–383`).
   Tested by `test_drain_uploads_render_that_finishes_within_window` (dispatch →
   straight to drain, no intervening poll → upload fires exactly once) and
   observed live in the rPPO smoke arm B log (`[render] checkpoint 54 render
   finished` printed at drain) and in the Dreamer async telemetry test (video
   row logged at drain with a monotone step).
2. **Dispatch must poll before overwriting (finding 2)** — implemented:
   `dispatch_render`'s first action is a failure-isolated `poll_render`, so a
   render that finished in the last-poll→next-dispatch window is uploaded
   before `pending` is replaced. Tested by
   `test_dispatch_does_not_overwrite_unpolled_completed_render` (child exits
   with NO poll; next dispatch must yield upload #1 for the old checkpoint and
   then dispatch the new one).
3. **Determinism baseline for artifact parity (finding 3)** — owner
   `senior-developer` (verification-plan amendment); noted here with one
   implementation-side datum: **raw `.rec.gz` checksums differ even for
   identical content because gzip embeds an mtime in its header** — parity
   checks must compare *decompressed* payloads (`zcat | cmp`), which is what
   the Smoke A/B below does.
4. **Make skip drops observable (finding 4)** — implemented:
   `state["skips"]` counter incremented on every skip-if-busy;
   `poll_render` surfaces it as **`Eval/video/render_skipped_total`** (parent
   is sole WandB writer; logged only when the counter advances; Dreamer mode
   stamps `step=policy_step`, rPPO mode logs step-less like its other rows).
   Tested by `test_skip_if_busy_single_child_and_skip_counter`.

🟢 findings: **5** — `render_workers: 8` kept per the plan's table, with the
rationale comment corrected in all three YAMLs + the module ("per-episode
parallelism, ≤ eval_video_episodes workers ever active; a cap, not a contention
lever"). **6** — **NOT applied as written** (deviation): `get_mandatory` raises
on a declared `null` (`Config.get_mandatory` treats `None` as missing,
`src/utils/config.py:73`), so `render_workers` *cannot* be read via
`get_mandatory` while `null` is a legitimate value; kept the plan's original
`.get('training.render_workers', None)` with an explanatory comment at both
read sites. **7** — implemented: the skip/backfill message tells the operator
to delete partial `episode_*.mp4` files before backfilling (`--skip-existing`
would keep a truncated file).

### Test results (all with `JAX_PLATFORMS=cpu`, conda interpreter)

| Suite | Command | Result |
|---|---|---|
| New unit tests | `pytest tests/training/test_async_render_dispatch.py -v` | **11 passed** (7.4 s) — non-blocking dispatch (<1 s) / poll-detects-completion / upload-exactly-once; skip-if-busy single-child + skip counter; bounded drain leaves orphan + closes log; drain-final-poll (amendment 1); no-overwrite race (amendment 2); every_n gate; rPPO `checkpoint_pct` step mode; rc≠0 failure path; blocking-fallback signature default; `_render_and_upload` still blocking `subprocess.run` with unchanged cmd; child-env GPU isolation |
| Reworked telemetry tests | `pytest tests/algorithms/dreamer_srl/test_eval_telemetry_wandb.py -q` | **2 passed** (382 s) — sync (kill-switch) invariant + new async end-to-end (real Dreamer `main()` loop, offline WandB: explicit monotone steps, video uploaded at/after its checkpoint's step) |
| Existing Dreamer suite | `pytest tests/algorithms/dreamer_srl/ -q` (first pass, `-x`) | **80 passed, 2 skipped (pre-existing skips), 1 failed** — the single failure was `test_eval_logs_monotone_explicit_steps`, whose assertion (c) hard-codes the *synchronous* video-step coupling this plan deliberately changes; reworked as flagged above, then **2/2 green**. Full combined re-run after the rework: see final line below. |
| `tests/training/` (pre-existing) | `pytest tests/training/ -q` (excl. new file) | **8 passed** (235 s) |
| Final combined re-run | `pytest tests/algorithms/dreamer_srl/ tests/training/ -q` | **161 passed, 9 skipped, 0 failed** (19 m 36 s; includes the 11 new + 2 reworked; skips are pre-existing conditional skips) |

### Smoke A/B + speed check (local CPU only — no lab node touched)

Setup: rPPO (`recurrent_ppo_gae.yaml` agent), default env with
`max_steps: 60`, `eval_video_episodes: 1`, `--episodes 60
--checkpoint-frequency 15 --seed 7 --num-envs 8 --no-wandb --device cpu`,
JAX_PLATFORMS=cpu; identical override configs except
`training.async_video_render` (arm A false / arm B true). Artifacts under
`tmp/20260727_async_smoke/` (A/, B/, A.log, B.log). Both arms hit checkpoints
at episodes 54 and 102 and exited rc=0.

- **Speed**: training-loop elapsed (tqdm) **84 s (sync) → 42 s (async)**, a
  **50% wall-clock reduction** in this render-dominated smoke (2 blocking
  renders ≈ 40 s removed from the critical path). Steady-state per-iteration
  cost of the new poll is a single `proc.poll()` — not measurable at smoke
  scale and architecturally negligible; the Verification Plan's GPU-node
  steady-state check remains the authoritative number.
- **Artifact integrity**: all 4 `.rec.gz` recordings **byte-identical after
  decompression** (`zcat | cmp`; raw gzip bytes differ only by the header
  mtime — see amendment 3 note), checkpoint directory listings identical, and
  the video both arms produced (`eval_54.mp4`) **byte-identical**.
- **Skip-if-busy observed as designed**: in arm B the checkpoint-102 dispatch
  found the checkpoint-54 render still running (this smoke's cadence is
  seconds, vs 8–21 min in production) and skipped; the drain then waited for
  and finished render 54 (`[render] checkpoint 54 render finished`). The 102
  recordings are on disk, offline-recoverable per design. No orphan
  `render_recordings` process after either arm's exit.
- Dreamer-side end-to-end (real loop, async on) is covered by
  `test_eval_logs_monotone_async_render`; a GPU-node Dreamer A/B was **not**
  run here (task constraint: local CPU only) — it is Verification Plan step 1.

### Deviations from the plan (all flagged, none silent)

1. **`tests/algorithms/dreamer_srl/test_eval_telemetry_wandb.py` modified** —
   not in the File Changes list ("No other file may change"). Unavoidable: its
   assertion (c) encodes the synchronous upload-step coupling the plan
   deliberately removes, so with the per-algo default flipped to async it fails
   by design. Rework keeps the original invariant verbatim on the kill-switch
   path and adds the async-mode invariant (upload at/after its checkpoint's
   step, still explicit + monotone) as a new test — a strict coverage increase.
2. **Plan-reviewer finding 6 not applied** (`get_mandatory` for
   `render_workers`) — factually incompatible with `Config.get_mandatory`,
   which raises on a declared `null`. Kept the plan's original `.get`; both
   read sites carry a comment saying why.
3. **Drain interrupted-timeout = 10 s** as the plan's §3 specifies — noting the
   plan's "same values as the experiment-eval drain" clause is slightly off
   (the experiment-eval constant is 5 s); the plan's explicit number wins.
4. **`video_dpi` dead-key comment** applied to `configs/visualization/default.yaml`
   only; the second occurrence is in an archived config, left untouched.
5. **State dict carries 3 extra keys** beyond the plan's minimal spec
   (`ckpt_index`, `skips`, `skips_logged`) — required by the every_n gate and
   amendment 4.

### Blockers / follow-ups

- **Checkpoint 6 open**: ask `bug-curator` to annotate the
  orphan-render-workers record
  (`docs/llm_wiki/entries/cluster_ops/20260513_0018_...sigint.md`) with the new
  dispatch site (`src/utils/async_render.py`) — `developer` cannot spawn agents.
- Verification Plan steps 1–5 (GPU-node A/B incl. the sync-vs-sync determinism
  control of finding 3, steady-state speed check, post-landing live comparison)
  → `senior-developer` / `experiment-analyzer`.
- A parallel session's `docs/memory/` → `docs/llm_wiki/` rename landed on disk
  mid-implementation; this plan doc's one memory-path citation was updated by
  that session, and `async_render.py`'s docstring uses the new path. The
  working tree contains that session's unrelated changes — this
  implementation's commit stages only its own files (and only its own hunks of
  `SCRIPTS_DEPENDENCY_MAP.md`, which the rename also touched).
- `scripts/eval/render_recordings.py` was **NOT modified** (hard invariant;
  `git status` shows no change to it or anything it imports). No lab node was
  touched; everything above ran on the local workstation, CPU-only.

Implemented by: `developer` (2026-07-27).

## Verification Report

_To be filled by `senior-developer` after implementation._

---

## Feedback from `plan-reviewer` (2026-07-27)

**Verdict: SOUND WITH CONCERNS** — safe to implement once the four 🟡 items below
are folded in (all are cheap spec/verification amendments, not redesigns). No
🔴 blocker found. In plain terms: the plan's measurements, code citations, and
config wiring were independently re-verified and are correct; the residual risks
are all of the "a video silently fails to appear on the dashboard" class, never
of the "training data or conclusions are wrong" class.

### Independently verified (so the implementer need not re-check)

- **Every code citation is accurate**: the blocking `subprocess.run` sites
  (`src/algorithms/dreamer_srl/eval.py:556`, `src/utils/evaluation_core.py:346`),
  the async idiom (`train.py:256–385`), the call sites
  (`dreamer_srl_main.py:1805–1834`, `train.py:2388–2399`), the renderer's
  `--workers` default (`render_recordings.py:142`), and the final-log/`wandb.finish`
  drain target (`dreamer_srl_main.py` §13).
- **Config wiring is correct for both trainers.** Dreamer merges
  `configs/train/default.yaml` + `configs/train/dreamer_srl.yaml` itself at
  startup in both single-config (`dreamer_srl_main.py:553–560`) and curriculum
  (`_load_stage_env_cfg`, `:123–124`) modes; `train.py` merges the same pair at
  `:493–512`. So the three new keys land where both trainers actually read
  `training.*`, and `get_mandatory` is safe because `default.yaml` declares them
  (the "new mandatory key breaks old configs" trap is avoided).
- **Critical-settings registry**: confirmed no `video_*` / render /
  `checkpoint_frequency` rows exist — the plan's "no registry entry needed"
  claim is true.
- **Live-run blast radius is genuinely zero — but only because of the "No other
  file may change" line.** Running trainers hold `train.py` /
  `evaluation_core.py` / `dreamer_srl_main.py` in memory; the files live runs
  *re-read from disk at every checkpoint* are the subprocess scripts
  (`render_recordings.py`, `src/environment/renderer.py`, and rPPO's
  `experiment_eval_checkpoint.py` → `run_sweep` → `eval_rollout.py` tree) — and
  I verified none of them imports `evaluation_core` (only comment mentions in
  `eval_rollout.py:207,235`). Implementer: treat that constraint as a **hard
  invariant** — this working tree is the live deployment for those scripts,
  hot-read every 8–21 minutes by 12 running jobs.
- **Prior-art / known-bug check** (targeted grep of the registry, in lieu of a
  `bug-curator` spawn): the orphan-render-workers-on-SIGINT record the plan
  cites exists and matches the plan's orphan policy; the WandB backward-step
  silent-drop row (fixed for Dreamer in `39f851b`) is correctly flagged by the
  plan as a possible latent rPPO issue. The 2026-07-26
  `subagent_bg_job_orphan_idle_ping` memory is about *Claude-session* background
  jobs pinging an idle session — a different mechanism entirely; trainer-spawned
  `Popen` orphans are finite CPU-only work and do **not** collide with that bug.
- `train.py --checkpoint-frequency` exists (`:414`), so Verification step 2 is
  runnable as written; the Dreamer smoke correctly uses `--episodes` (the
  single-config budget gotcha is accounted for).

### Findings

| # | Sev | Location | Issue | Suggested fix | Owner |
|---|---|---|---|---|---|
| 1 | 🟡 | Proposed Solution §3 `drain_render` | The drain spec is wait + close log — it never uploads. A render in flight at normal exit *completes during the drain's 300 s wait*, but the poll loop has already ended, so **the last video of every run silently never reaches WandB**. The source idiom's drain ends with a final poll (`train.py:381–383`); the port drops that step. | `drain_render` calls `poll_render` once after the wait; Test 3 asserts the upload fires when the child finishes within the drain window. | `developer` (spec amendment first: `senior-developer`) |
| 2 | 🟡 | Proposed Solution §1 `dispatch_render` | Dispatch-over-unpolled-completion race: if the render finishes in the window between the last per-iteration poll and the next checkpoint's dispatch, dispatch sees `proc` dead and overwrites `pending` → that completed video is never uploaded, silently. The experiment-eval idiom is immune (it discovers results by globbing `result.json` files); this port keys off in-memory state, so it inherits a race the original doesn't have. | `dispatch_render` runs the poll/upload logic first (or refuses to overwrite a completed-but-unuploaded `pending`). Add a unit test for this interleaving. | `developer` |
| 3 | 🟡 | Verification Plan §1 (artifact parity) | "Same seed ⇒ same eval trajectories" assumes **bitwise-deterministic training across two separate GPU runs** — cuDNN/XLA autotune can break that. If checksums differ for that reason, you either burn hours debugging a non-bug or, worse, rationalise a real difference. | Run a sync-vs-sync same-seed control first to establish the determinism baseline; if not bitwise-stable, fall back to structural parity (file set, cadence, per-file step counts) + the already-planned code inspection that the recording path is untouched. | `senior-developer` |
| 4 | 🟡 | Risks §"Skip-if-busy is expected to be rare" | The rarity claim extrapolates render durations measured **while training was paused** (renderer had the whole host). Post-fix, renders share the host with training and durations grow with agent survival; on the b03-style rPPO cadence (event every ~8.3 min, render already ~3 min) headroom is ~2.7×, not large. A skipped video today is one stdout line nobody greps. | Make drops observable where people look: count skips and log e.g. `Eval/video/render_skipped_total` from the parent at poll time (parent stays sole WandB writer). Then the drop policy is fine as designed. | `developer` |
| 5 | 🟢 | Config surface, `render_workers: 8` | Largely inert knob with an overstated rationale: `render_recordings.py` parallelises **per episode** (one task per `.rec.gz`, `:182–186`) and `eval_video_episodes: 3`, so at most 3 workers ever do work, and the serial concat/re-encode in the main process is unaffected by `--workers`. Harmless to keep, but don't credit it with bounding contention. | Keep or default to `null`; correct the rationale comment. | `developer` |
| 6 | 🟢 | Call-site changes (Dreamer) | `render_workers` read via `.get` while the key is declared in all three YAMLs — use `get_mandatory` (with `null` as a legitimate declared value) for consistency with the no-fallback rule. | One-word change. | `developer` |
| 7 | 🟢 | Proposed Solution §1 skip message / §3 orphan policy | An orphan killed mid-encode (node reboot, SIGKILL) leaves a truncated `episode_*.mp4`; the advertised backfill (`render_recordings.py <dir>` — and the in-repo habit of `--skip-existing`) would then *skip the corrupt file* and concat a broken video. Pre-existing hazard, videos-only. | One line in the skip/backfill message: delete partial MP4s before backfilling. | `developer` |

### Unstated assumptions (status)

- **A. Live subprocesses don't import the edited files** — now VERIFIED (above);
  load-bearing for the "future runs only" safety claim.
- **B. Same-seed bitwise determinism** — UNVERIFIED (finding 3).
- **C. Cadence ≫ render duration holds for future configs/agents** — plausible
  today, UNVERIFIED as agents' survival grows (finding 4 makes violation
  observable instead of silent).
- **D. rPPO videos currently reach WandB at `step=checkpoint_pct`** —
  UNVERIFIED, and the Dreamer analogue was a confirmed silent-drop bug. If rPPO
  uploads are *already* being dropped, the smoke A/B's "video logged once" check
  fails in **both** arms — anticipate that and don't attribute it to the async
  change (plan already scopes the fix out; agreed).

### Passes skipped

Experiment-plan specifics (controls/seeds/confounds/obs-noise sync) — this is an
engineering plan, not an experiment design; feasibility items (node choice via
`gpu-status`, not-on-live-nodes) are already handled by the plan.

### Cost of being wrong

If the concerns above materialise, the cost is **missing dashboard videos**
(final-video-per-run from #1, occasional raced/skipped checkpoints from #2/#4) —
lost monitoring convenience, recoverable offline from the untouched recordings —
plus, for #3, a few hours chasing a spurious parity failure. There is no path
from this plan to training-artifact corruption, data loss, or a wrong scientific
conclusion, provided the "No other file may change" invariant is honoured.

*Reviewed by: `plan-reviewer`, 2026-07-27.*

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
