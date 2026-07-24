---
title: "Experiment eval during rPPO training (async, live WandB)"
topic: behavior
status: active
created: 2026-07-23
last_updated: 2026-07-24
---

# Experiment eval during rPPO training (async, live WandB)

> **Status**: PLANNED
> **Opened**: 2026-07-23
> **Related**: [[run_sweep]] (offline dwell-sweep pipeline, `scripts/eval/dwell_sweep/run_sweep.py`); Dreamer during-training eval hook (`src/algorithms/dreamer_srl/dreamer_srl_main.py` L1695–1765) — prior art for the checkpoint-triggered eval + custom-step-metric pattern.

---

## Context

Today, to see how an agent's *avoidance behaviour* evolves over training — how much time it spends hiding in the bush, how close it lets a predator approach, how far it roams — we have to wait until the run finishes (or pause to babysit it) and then launch the offline "dwell-sweep" pipeline by hand. That pipeline rolls each saved checkpoint against a fixed battery of 12 scenarios ("a predator is present", "a harmless rabbit is present", "no animal", each at two injury levels) and computes 11 behaviour measures per scenario. It works, but it is a separate, after-the-fact, multi-node job.

This plan wires that same behaviour eval **into the rPPO training loop** so it runs automatically and its results appear **live on the run's WandB dashboard** while training continues. Concretely: at every Nth checkpoint save (N configurable; **default 1, i.e. every checkpoint** — changed from an earlier draft default of 5 once the actual eval cost was measured at ~36 seconds, see "Implementation Report"), the trainer kicks off the 12-scenario experiment eval on the checkpoint it just wrote — as a **background CPU job that does not block or slow the GPU training** — and, once that job finishes, the trainer plots a focused handful of series (time-in-bush, survival, roaming radius for the predator / rabbit / no-animal scenarios) against the checkpoint number.

The hard part is not the eval itself — that code already exists and we reuse it wholesale. The hard part is the **plumbing**: two processes (the trainer on the GPU, the eval on the CPU) must never write the same WandB run at the same time, and an eval result that lands *after* training has already advanced must still plot at the *correct* x-position. The design below solves that with a file hand-off plus a WandB custom step-metric, mirroring the pattern the Dreamer trainer already uses for its late-arriving async video eval.

This is a **planning document only** — no code is written here. Implementation is the `developer` agent's job after this plan is approved.

## Analysis

### What already exists (and is reused verbatim)

| Piece | Location | Role |
|---|---|---|
| Fast multi-condition rollout | `scripts/eval/eval_rollout.py --config-list` | One process builds the model + restores the checkpoint **once**, then loops all listed conditions (the "3.2×" fast path). For rPPO it reads the agent config from the checkpoint's sibling `models/config.yaml` — **no `--agent_config` needed** (confirmed at `eval_rollout.py` L720, `_resolve_continual_stage_config`). |
| 11 behaviour measures | `scripts/behavior_measures/avoidance_stats_heatmap.py` → `episode_measures(ep)`, `KEYS` | Given one recording, returns the 11 measures (`bush_use_rate, bush_entry_step, bush_dwell, fid, time_near_animal, closest_approach, time_moving, spatial_spread, pursuit_duration, injury_change, survival_steps`). |
| Recording → CSV aggregation | `scripts/eval/dwell_sweep/run_sweep.py` → `_measure_cell(step_dir)`, `HEAD` | Recursive-globs a checkpoint's `.rec.gz` recordings, averages the 11 measures across episodes, and produces one CSV row (`step, step_M, <11 measures>`). Byte-identical schema to `results/eval/avoidance/<name>/<label>/<cond>.csv`. |
| CPU thread-cap invocation | `scripts/eval/dwell_sweep/sweep_worker.sh` L29–31, L67–69 | The exact env vars (`JAX_PLATFORMS=cpu`, `XLA_FLAGS=... eigen=false intra_op=1`, `OMP/MKL/OPENBLAS/TF_*=1`) + the `eval_rollout.py --config-list ... --batched --device cpu --record` call that make the eval barely compete with a GPU-bound trainer. |
| The 12 core conditions | `configs/environment/experiment/behavior_probes/core/avoidance/avoid_*.yaml` | `avoid_{none,pred,rabbit,rabbit_olfzero,rabbitwander,rabbitwander_predsmell}_inj{00,70}.yaml` = 12 files. |

### The two integration points in `train.py`

- **Checkpoint save** — `train.py` L2081–2100. The checkpoint is written to `<results_dir>/models/<total_episodes_completed>/` via orbax, and `checkpointer.wait_until_finished()` (L2100) guarantees it is fully on disk before the next line. The orbax save key is `total_episodes_completed` — an integer that is **also the checkpoint directory name** and the `step` column the offline CSVs already use. This is our natural checkpoint key. There is already a checkpoint-triggered eval block right below (L2102–2177, video/stats) — the new experiment-eval dispatch slots in beside it.
- **WandB metric definition** — `train.py` L702–714. Note L712 already binds `Behavior/* → iteration` for the per-iteration behaviour stats. **We must NOT reuse the `Behavior/` prefix** (a prefix can carry only one step-metric). The experiment series get a distinct `Experiment/` prefix with its own step-metric.

### The single-writer / late-arrival problem, and the prior art

Two processes writing one WandB run concurrently corrupts the step counter. The Dreamer trainer already dodges a milder version of this: its checkpoint-triggered eval renders a video in a subprocess (~13 s) that advances WandB's internal step, so it logs `Eval/*` **without** an explicit `step=` and relies on `wandb.define_metric("Eval/*", step_metric="timesteps")` to place the point (see `dreamer_srl_main.py` L876–879 for the define, L1727–1740 for the log). The custom step-metric means the x-position is read from a key **in the logged dict**, not from WandB's monotonic step — so an out-of-order or late log still lands at the right x.

Our case is stronger: the eval runs **fully async** (the trainer does not wait for it at all), so a result can arrive many iterations later. Same cure, taken one step further:

1. The **eval process never touches WandB.** It computes results, writes CSVs, and writes a small `result.json` to a known path keyed by checkpoint.
2. The **trainer is the sole WandB writer.** Each training iteration it cheaply polls for finished `result.json` files and logs any it has not logged yet, using `Experiment/checkpoint_episode` (= that checkpoint's `total_episodes_completed`) as the x-value. `wandb.define_metric("Experiment/*", step_metric="Experiment/checkpoint_episode")` guarantees the point plots at the correct checkpoint no matter how far training has advanced.

## Implementation Plan

### Design

#### Data flow

```
 training loop (GPU)                         async eval (CPU, thread-capped)
 ───────────────────                         ───────────────────────────────
 every checkpoint save:
   checkpointer.save(ep)
   wait_until_finished()
   if experiment_eval enabled and
      (ckpt_index % N == 0):
        Popen(experiment_eval_checkpoint.py ──►   builds --config-list over 12 conds
              --checkpoint models/<ep>       eval_rollout.py --config-list
              --result-json <run>/experiment_eval          --batched --device cpu --record
                  /<ep>/result.json          (one build+restore, loop 12 conds)
              ...)                           aggregate .rec.gz → 11 measures × 12 conds
   (does NOT wait)                           write <run>/experiment_eval/<ep>/<cond>.csv  (×12)
                                             write result.json ATOMICALLY (tmp+os.replace)
 every iteration (cheap):
   _poll_and_log_experiment_results():
     glob <run>/experiment_eval/*/result.json
     for each ep not in logged set:
       read json; if status==ok:
         wandb.log({Experiment/<measure>/<cond>: v, ...,
                    Experiment/checkpoint_episode: ep,
                    iteration, timesteps})
       mark ep handled
 at training end:
   final drain-poll (bounded wait) then wandb.finish()
```

#### Key decisions

- **Checkpoint key = `total_episodes_completed`** (the orbax save key / checkpoint dir name / offline-CSV `step` column). Monotonic, already meaningful, and lets the eval process locate the checkpoint dir directly. The `result.json` **also** carries `global_step` and `iteration` so a future dashboard could switch the x-axis without re-running.
- **Distinct WandB namespace `Experiment/`.** Series names: `Experiment/<measure>/<cond_short>`, e.g. `Experiment/bush_dwell/pred_inj00`. Chosen because `Behavior/*` is already bound to `iteration` (L712) and `Eval/*` to `timesteps` (indirectly). **Settled** (was open at plan-time): the live WandB panel gets ONLY the focused **3 measures × 3 conditions = 9 series** (`bush_dwell`/`survival_steps`/`spatial_spread` × `pred_inj00`/`none_inj00`/`rabbit_inj00`); the full 11×12 always lands in the CSVs regardless.
- **Dispatch = on-node `subprocess.Popen` (recommended default).** Rationale: it matches the user's "training script activates eval" mental model, needs zero cross-node coordination, and — because the eval process inherits the dwell-sweep thread caps (`OMP=1`, eigen off, `JAX_PLATFORMS=cpu`) while the trainer is GPU-bound and mostly idle-waiting on the GPU — it barely competes for CPU. The alternative (dispatch to a free lab node via `run_command.py`) removes even that marginal CPU contention but adds "which node is free" logic, cross-node file coordination, and a new failure surface. Config key `training.experiment_eval_on_node` is added now with only `"self"` implemented; a node-id value is left as a documented future extension (the runner script is written node-agnostic so the extension is a dispatch-site change only). **GPU isolation (nit fix, implemented):** the `Popen` env sets `JAX_PLATFORMS=cpu` AND `CUDA_VISIBLE_DEVICES=""` belt-and-braces at BOTH the `train.py`→`experiment_eval_checkpoint.py` layer and the `experiment_eval_checkpoint.py`→`eval_rollout.py` layer, plus the CPU thread caps (`OMP/MKL/OPENBLAS/TF_NUM_INTRAOP/INTEROP=1`) exactly as `sweep_worker.sh` L29-31 — so no code path in the eval tree can ever reach the training GPU.
- **At most one concurrent experiment-eval.** The trainer keeps the live `Popen` handle; if a new checkpoint trigger fires while the previous eval is still running, it **skips** this checkpoint's eval and logs a one-line warning (the next trigger recovers). This is never a data-loss event: rPPO keeps every checkpoint on disk, and the offline `run_sweep.py` pipeline can incrementally backfill any skipped checkpoint later — a skip is offline-recoverable, not lost data.
- **Serial vs. parallel eval — settled by measurement (fix #3).** Implemented SERIAL first (one `eval_rollout.py --config-list` process, all 12 conditions, no `--npar`/parallelism flag). Measured wall-time for the FULL 12-condition/30-episode battery on a real checkpoint, on a 20-core node: **~36 seconds**. Far under the "few minutes" threshold that would have warranted K-way parallel splitting, so SERIAL stays the only path — no parallel-splitting code was added, and the plan's earlier `--npar <int, default 12>` CLI flag (a leftover from the multi-node offline sweep's xargs fan-out, which does not exist in this single-checkpoint path) is REMOVED. Given this cost, `experiment_eval_every_n_checkpoints` defaults to **1** (every checkpoint), not 5 — see the config-keys section below.
- **Reusable single-checkpoint entry point.** A new script `scripts/eval/experiment_eval_checkpoint.py` is the "eval one checkpoint → CSV + result.json" core. It **imports** `_measure_cell`, `HEAD`, `KEYS` from `run_sweep.py` (no reimplementation of the measure/aggregate logic) and reproduces `sweep_worker.sh`'s single `eval_rollout.py --config-list` invocation + thread caps. `run_sweep.py`'s multi-node offline path is left untouched (it is proven and orthogonal); this script is the shared *core*, callable standalone for debugging and by the trainer. `_measure_cell` returns CSV-formatted strings (`""` for non-finite, `"%.4f"` else) — the runner converts these explicitly to `float`/`None` when building the `result.json` `measures` dict (nit fix), so `result.json` is valid JSON with real numbers, not string-typed CSV cells.
- **Resume double-logging (fix #2).** At trainer startup (before the training loop, right after `results_dir`/`models_dir` are established), the trainer globs `<results_dir>/experiment_eval/*/result.json` ONCE and pre-seeds `experiment_state["logged"]` with every checkpoint episode already present. A resumed run (same `results_dir`, restarted process) therefore never re-logs a prior session's checkpoint as a duplicate WandB point.
- **Startup-time config validation (nit fix).** All `training.experiment_eval_*` keys are read via `get_mandatory` at trainer STARTUP (right after `algorithm = config.get_mandatory('agent.algorithm')`), not at the first checkpoint save. A bad/incomplete config raises in the first second of the run, not hours in. The gate also raises if `experiment_eval_during_training=true` is set for a non-RecurrentPPO algorithm (the feature only supports RecurrentPPO).
- **Ctrl-C + drain policy (fix #4).** Two different drain timeouts, both HARDCODED module constants in `train.py` (not config keys — see the nit-fix note under File Changes): `EXPERIMENT_EVAL_DRAIN_TIMEOUT_S = 300` for a NORMAL exit (bounded wait so a late-dispatched eval still gets its WandB point logged before `wandb.finish()`), and `EXPERIMENT_EVAL_DRAIN_TIMEOUT_S_INTERRUPTED = 5` for a Ctrl-C (`KeyboardInterrupt`) exit — the user asked training to stop NOW, not in up to 300s. In BOTH cases, if the in-flight subprocess is still running past its timeout, the trainer does **NOT kill it** — it is left running as an orphan (harmless, on-node CPU only) so its CSV still completes; only its final WandB point is missed.

### File Changes

#### NEW: `scripts/eval/experiment_eval_checkpoint.py`

Standalone runner — the reusable "eval one checkpoint → metrics + CSV + result.json" entry point. Invoked by the trainer via `Popen`, and runnable by hand for smoke tests.

**As implemented** (CLI, all mandatory unless noted — `--npar` from the original plan draft was REMOVED, see the Key decisions' fix #3 note; there is no xargs fan-out in this single-checkpoint path):
```
--checkpoint    <run>/models/<episode>          # the just-saved checkpoint dir
--result-json   <run>/experiment_eval/<episode>/result.json
--out-root      <run>/experiment_eval/<episode>      # per-condition CSVs land here
--conditions    all | avoid_pred_inj00,avoid_none_inj00,...   # default "all"
--episodes      30                              # default 30
--probe-dir     configs/environment/experiment/behavior_probes/core/avoidance   # default
--checkpoint-key <int>                          # = total_episodes_completed (echoed into result.json)
--global-step   <int>                           # echoed into result.json for alt x-axis
--iteration     <int>                           # echoed into result.json
```

Behaviour (as implemented):
```python
# 1. Resolve conditions -> list of (cond_stem, cfg_path). "all" -> sorted glob avoid_*.yaml.
# 2. Set CPU thread caps in the child env EXACTLY as sweep_worker.sh L29-31, PLUS
#    CUDA_VISIBLE_DEVICES="" (belt-and-braces GPU isolation, nit fix) and a per-run
#    persistent JAX compile cache (also from sweep_worker.sh) keyed by a hash of the
#    checkpoint's run dir.
# 3. Build a --config-list temp file: one "<cfg_path>\t<out_root>/<cond>/<ckpt_key>" line per cond.
# 4. subprocess.run([...eval_rollout.py, "--config-list", cl,
#                    "--checkpoint", ckpt, "--eval-n-episodes", eps,
#                    "--record", "--record-n-episodes", eps,
#                    "--device", "cpu", "--quiet", "--batched", "--seed", "0"],
#                   env=capped_env)   # mirrors sweep_worker.sh L67-69; ONE call, SERIAL (fix #3)
# 5. For each cond: reuse run_sweep._measure_cell(<out_root>/<cond>/<ckpt_key>) -> (step, vals).
#    Write <out_root>/<cond>.csv with header run_sweep.HEAD (same schema as offline CSVs).
#    _measure_cell's CSV-formatted strings ("" / "%.4f") are converted to None/float
#    EXPLICITLY here (nit fix) before going into result.json's measures dict.
# 6. Assemble result = {"status": "ok"|"failed", "checkpoint_key": K, "global_step": G,
#        "iteration": I, "error": <str-if-failed>,
#        "measures": {cond_stem: {measure: float_or_null for measure in KEYS} for cond}}.
# 7. Write result.json ATOMICALLY: json.dump to result.json.tmp, then os.replace(tmp, result.json).
# ALL of steps 3-6 wrapped so ANY exception -> status="failed", error=<traceback>, still writes json,
# and the script's own exit code is 0 either way.
```

Note: because this is a *separate process*, even a hard crash of the runner cannot touch the trainer — the only failure the trainer sees is "no `result.json` ever appeared", which the drain-poll handles by simply never logging that checkpoint.

**Measured wall-time (settles fix #3):** on a real rPPO checkpoint (`results/JAX_RecurrentPPO/20260721-045245_rppo_b04v03_lowdmg_128env_n112/models/29000074`), the FULL 12-condition/30-episode battery took **~36 seconds** on a 20-core node — the model is built and the checkpoint restored ONCE, then all 12 conditions run in that same process. This is well under the "few minutes" threshold that would have warranted splitting the 12 conditions across K=3-4 parallel `eval_rollout.py` processes, so the implementation stays SERIAL (single `--config-list` call, no parallel-splitting code, no `--npar` flag).

#### `configs/train/default.yaml` (training block, after L54)

Add five keys (mandatory-key discipline: they live on the shared training-defaults path so every algorithm that merges this file has them, exactly like `video_during_training`/`stats_during_training` above). **`experiment_eval_every_n_checkpoints` default changed from the plan draft's 5 to 1** (every checkpoint) given the ~36s measured cost above — cheap enough, and fully async/non-blocking, to run on every save rather than skip most of them:

```yaml
  # --- Experiment eval during training (async, on-node CPU; see
  #     docs/develop/active/behavior/EXPERIMENT_EVAL_DURING_TRAINING.md) ---
  experiment_eval_during_training: false   # master gate; rPPO runs opt in via their own config
  experiment_eval_every_n_checkpoints: 1   # run the experiment eval on every Nth checkpoint save (see measurement above)
  experiment_eval_conditions: "all"        # "all" (12 core avoidance conds) or a comma-list of stems
  experiment_eval_episodes: 30             # episodes per condition
  experiment_eval_on_node: "self"          # "self" = on-node Popen (only value implemented; node-id = future)
```

**As implemented (nit fix — startup validation):** the trainer reads `experiment_eval_during_training` as the gate at STARTUP (right after `algorithm = config.get_mandatory('agent.algorithm')`, well before the training loop or the first checkpoint), and **only if true** does it `get_mandatory` the other four — all at that same startup point, not lazily at the first checkpoint. This keeps a run that never opts in from being forced to declare experiment-eval values, while still failing loudly (in the first second of the run) if an opted-in run is missing a key. The gate also raises if `experiment_eval_during_training=true` is combined with a non-RecurrentPPO algorithm.

**No 6th config key (nit fix).** The plan draft's `training.experiment_eval_drain_timeout_s` (via `config.get(..., 300)`) would have been a fallback default on a mandatory-key-discipline path — REMOVED. The drain timeout is instead two HARDCODED module constants in `train.py`: `EXPERIMENT_EVAL_DRAIN_TIMEOUT_S = 300` (normal exit) and `EXPERIMENT_EVAL_DRAIN_TIMEOUT_S_INTERRUPTED = 5` (Ctrl-C exit) — see fix #4 below. Only the original 5 keys are added to the YAML schema.

#### `train.py` — as implemented

**(a) define_metric block.** Added right after the existing `stage/transition` define_metric call, inside `if wandb_enabled:`. Unchanged from the plan draft:
```python
wandb.define_metric("Experiment/checkpoint_episode")
wandb.define_metric("Experiment/*", step_metric="Experiment/checkpoint_episode")
```

**(b) startup gate + resume pre-seed (NEW vs. the plan draft — fixes #2 and the startup-validation nit).** Two blocks, both BEFORE the training loop:
- Right after `algorithm = config.get_mandatory('agent.algorithm')`: the gate + mandatory-key validation described above, producing `experiment_eval_enabled: bool` and `experiment_eval_cfg: dict | None`.
- Right after `models_dir` is created: `experiment_state = {"proc": None, "ckpt_index": 0, "logged": set()}`, then (if `experiment_eval_enabled`) ONE glob over `<results_dir>/experiment_eval/*/result.json`, pre-seeding `experiment_state["logged"]` with every checkpoint episode already present — the resume-double-logging fix (#2).

**(c) dispatch — inside the `if should_checkpoint:` block, right after `checkpointer.wait_until_finished()`, before the existing video/stats eval.** Guarded and failure-isolated exactly as the plan draft specified:
```python
if algorithm == "RecurrentPPO" and experiment_eval_enabled:
    try:
        _maybe_dispatch_experiment_eval(
            experiment_eval_cfg, experiment_state, results_dir, models_dir,
            total_episodes_completed, global_step, iteration, args.quiet,
        )
    except Exception as e:
        pbar.write(f"[experiment-eval] dispatch skipped (non-fatal): {e}")
```
`_maybe_dispatch_experiment_eval` increments a monotonic checkpoint index; returns early unless `index % experiment_eval_every_n_checkpoints == 0`; reaps `experiment_state["proc"]` if finished, and if an experiment eval is still running, **skips** (logs a one-line warning, offline-recoverable) rather than launching a second; otherwise builds the `Popen` of `scripts/eval/experiment_eval_checkpoint.py` with **absolute paths** (nit fix — `os.path.abspath` on `--checkpoint`/`--result-json`/`--out-root`, since `results_dir` may be relative to the trainer's cwd) and the GPU-isolation env (`_experiment_eval_env()`: `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=""`, thread caps), `stdout/stderr` redirected to `<results_dir>/experiment_eval/<ep>/log.txt`, and stores the handle.

**(d) poll+log — once per iteration, cheap.** Call site, right after the algorithm if/elif chain's per-iteration `pbar.refresh()` and before the (unconditional, shared) "Checkpoint Logic" comment — so it runs exactly once per iteration regardless of which algorithm branch executed that iteration:
```python
if experiment_eval_enabled:
    try:
        _poll_and_log_experiment_results(experiment_state, results_dir, iteration, global_step,
                                     wandb_enabled, args.quiet)
    except Exception as e:
        print(f"[experiment-eval] WARNING: poll skipped (non-fatal): {e}")
```
**Fix #1 (failure isolation) — implemented at BOTH layers**: the call site above wraps the whole helper call, AND every internal step inside `_poll_and_log_experiment_results` (the glob, each `json.load`, the WandB-log branch) has its OWN try/except, so a malformed `result.json` (`KeyError`/`json.JSONDecodeError`), a NAS glob hiccup, or a `wandb.log` raise on ONE checkpoint's result can never block or crash processing of the others, and can never propagate into the training loop. A read failure does NOT mark the checkpoint handled (so a write-in-progress race retries next iteration); a successful read (whether `status="ok"` or `"failed"`) always marks it handled.

The WandB log only includes the **focused 9 series** (settled — see Key decisions): `FOCUS_MEASURES = ("bush_dwell", "survival_steps", "spatial_spread")` × `FOCUS_CONDS = ("pred_inj00", "none_inj00", "rabbit_inj00")`, both module-level constants in `train.py`. The full 11×12 always lands in the CSVs regardless of this filter.

**(e) final drain — in the cleanup block, before `wandb.finish()`.** Fix #4 (Ctrl-C + drain policy): the `try/except KeyboardInterrupt` around the training loop now has an `else:` clause setting `training_interrupted = False` (the `except` branch sets it `True`), and the drain call passes that through:
```python
if experiment_eval_enabled:
    try:
        _drain_experiment_results(experiment_state, results_dir, iteration, global_step, wandb_enabled,
                              args.quiet, interrupted=training_interrupted)
    except Exception as e:
        print(f"[experiment-eval] WARNING: final drain skipped (non-fatal): {e}")
```
`_drain_experiment_results` waits up to `EXPERIMENT_EVAL_DRAIN_TIMEOUT_S_INTERRUPTED=5`s (Ctrl-C) or `EXPERIMENT_EVAL_DRAIN_TIMEOUT_S=300`s (normal exit) for `experiment_state["proc"]` to finish, then does one final `_poll_and_log_experiment_results`. In BOTH cases, a subprocess still running past its timeout is **left running** (not killed) — orphaned, harmless (on-node CPU only), its CSV still completes; only its final WandB point is missed.

#### `docs/environment/SCRIPTS_DEPENDENCY_MAP.md`

Added rows for the new `scripts/eval/experiment_eval_checkpoint.py`: (§1a) its bare-name `run_sweep` import; (§1b) `train.py`'s `Popen` of it, AND its own `subprocess.run` of `eval_rollout.py`; (§3) its full per-file roll-up entry; (§4) joined Cluster B (eval→record→render). Also updated the footer "last updated" line. Done — see that doc for the exact rows.

#### `docs/environment/02_config_schema.md` + `docs/environment/CONFIG_GUIDE.md` — DEVIATION (not edited)

**Flagged deviation from the plan draft.** `02_config_schema.md`'s own Overview states its scope explicitly: it documents `load_env_params(config) → EnvParams` — i.e. the `environment.*` / `behavior_measures.*` namespace read by `src/environment/config_loader.py`. The five `training.experiment_eval_*` keys live in a completely different namespace (`training.*`, read directly by `train.py` via `src/utils/config.py`'s plain `Config.get_mandatory`), which this schema doc does not cover at all — confirmed by grepping it for the FIVE PRE-EXISTING analogous keys (`video_during_training`, `stats_during_training`, `eval_video_episodes`, etc.): zero hits. There is no established doc home for `training.*` keys today (they are documented only as YAML comments in `configs/train/default.yaml`, which is what this change does too, for consistency with the 5 keys it sits beside). Editing `02_config_schema.md` for these keys would be scope-mismatched and would set a precedent of documenting an unrelated namespace in the wrong doc. **`developer` did not touch either doc; the only docs updated were `SCRIPTS_DEPENDENCY_MAP.md` and this plan doc itself** — flagged here for `senior-developer` to confirm at verification.

### Failure isolation (explicit)

| Failure | Contained by |
|---|---|
| Eval process crashes / hangs / OOMs | It is a **separate process**; the trainer never `wait()`s on it in the loop. A crash → no `result.json` → the checkpoint is silently never logged (drain-poll times out gracefully). A hang → the one-concurrent cap skips future evals until it exits; the drain has a bounded timeout (300s normal exit / 5s Ctrl-C) and does NOT kill the process even on timeout — it is left running, orphaned, harmless (on-node CPU only), its CSV still completes. |
| `eval_rollout.py` errors inside the runner | Caught in `experiment_eval_checkpoint.py`; it writes `result.json` with `status="failed"` + traceback, exit code 0. Trainer marks the checkpoint handled and logs nothing. |
| Dispatch (`Popen` construction) throws | Wrapped in `try/except` at the call site (edit **c**); logs a one-line warning, training continues. |
| Truncated / malformed `result.json` read | Poll helper `try/except` around `json.load`; on error it **skips without marking handled**, so the next iteration retries (the atomic `os.replace` write means a fully-written file is never truncated — this guard only covers a read racing an in-progress write on filesystems where `os.replace` visibility lags). |
| **(fix #1) Any other failure in the poll/log path** — a `KeyError` on `json["status"]`/`["measures"]`/`["checkpoint_key"]`, a NAS glob hiccup, or a `wandb.log` raise | Every internal step of `_poll_and_log_experiment_results` has its OWN try/except (glob, per-file read, per-file log), AND the main-loop call site wraps the ENTIRE helper call in try/except too — double-layered so nothing here can ever propagate into the training loop. One bad checkpoint's result never blocks processing of the others. |
| CPU contention slowing training | Thread caps + `CUDA_VISIBLE_DEVICES=""` GPU isolation set at BOTH the `train.py`→`experiment_eval_checkpoint.py` layer and the `experiment_eval_checkpoint.py`→`eval_rollout.py` layer (belt-and-braces); one-concurrent cap; GPU-bound trainer is CPU-idle most of the wall clock; measured ~36s per full 12-condition/30-episode eval on a 20-core node. |
| Resumed run re-logging prior checkpoints (fix #2) | Startup pre-seed: one glob over `<results_dir>/experiment_eval/*/result.json` before the training loop populates `experiment_state["logged"]`, so a resumed session's poll loop treats prior-session checkpoints as already handled. |
| Bad/incomplete experiment-eval config discovered hours into a run | All `training.experiment_eval_*` keys validated via `get_mandatory` at trainer STARTUP (algorithm-determination time), not at the first checkpoint — a missing key raises `ValueError` in the first second of the run. |
| Ctrl-C making the user wait for a slow background eval (fix #4) | `KeyboardInterrupt` sets `interrupted=True`, which drops the drain timeout to 5s (from 300s) — the process exits promptly; any still-running eval is left as a harmless orphan rather than blocking shutdown. |

## Checkpoints

- [ ] **Reuse, not reinvent** — `experiment_eval_checkpoint.py` imports `_measure_cell`, `HEAD`, `KEYS` from `run_sweep` and `episode_measures`/`KEYS` from `avoidance_stats_heatmap`; it does **not** copy the measure/aggregate math. Confirm the produced `<cond>.csv` header equals `run_sweep.HEAD` and columns match an existing `results/eval/avoidance/*/*/*.csv`.
- [ ] **Non-blocking** — add a temporary print of wall-clock time around the `should_checkpoint` block in a smoke run; dispatching the experiment eval must add **< ~50 ms** to the training iteration (it is only a `Popen` + a dict update).
- [ ] **Single WandB writer** — grep the diff: the eval process must contain **no** `import wandb` / `wandb.log`. Only `train.py` logs `Experiment/*`.
- [ ] **Correct x-placement** — in the smoke run, force two checkpoints to trigger evals; confirm on WandB that `Experiment/bush_dwell/pred_inj00` has two points at x = the two checkpoint episode numbers, even though many training iterations logged in between.
- [ ] **No double-log** — confirm each checkpoint episode appears once in `Experiment/checkpoint_episode` (the `logged` set prevents re-logging on subsequent polls).
- [ ] **Failure swallowed** — temporarily point `--probe-dir` at a non-existent path (or corrupt one condition config) in a smoke run; confirm the eval writes `status="failed"`, training **continues to completion**, and no traceback escapes into the trainer.
- [ ] **Gate + mandatory keys** — with `experiment_eval_during_training: false`, no `experiment_eval` dir is created and no `Popen` fires. With it `true` but one `experiment_eval_*` key removed, the run raises a clear `get_mandatory` `ValueError` **at startup** (not at dispatch — nit fix: hoisted validation).
- [ ] **Resume no double-log (fix #2)** — with a results_dir that already has `experiment_eval/<ep>/result.json` files from a prior session, a fresh process pre-seeds `experiment_state["logged"]` and does not re-log those checkpoints as new WandB points.
- [ ] **Serial-vs-parallel decision measured, not assumed (fix #3)** — the plan doc records an actual measured wall-clock time for the full 12-condition/30-episode battery, and the implementation (serial vs. K-way parallel) matches what that measurement calls for.
- [ ] **Ctrl-C drain is short; normal-exit drain never kills (fix #4)** — a Ctrl-C exit does not block for anywhere near 300s; a normal-exit drain timeout leaves the subprocess running rather than killing it.
- [ ] **GPU isolation belt-and-braces** — grep the Popen env-building code: both `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=""` are set at the `train.py`→runner layer (not just inside the runner's own subprocess call).

### Verification plan (no full training run needed)

A full run is unnecessary — the mechanism is exercisable in minutes:

1. **Runner unit smoke (standalone) — DONE, see Implementation Report.** Ran against a real checkpoint (`results/JAX_RecurrentPPO/20260721-045245_rppo_b04v03_lowdmg_128env_n112/models/29000074`): (a) a quick 3-condition/3-episode functional smoke (25.7s, `status:"ok"`, 3 CSVs + valid `result.json`); (b) the FULL 12-condition/30-episode battery for the timing measurement that settled fix #3 (~36s, `status:"ok"`, all 12 CSVs + `result.json`); (c) a bogus `--probe-dir` to confirm the failure path (`status:"failed"` + full traceback in `error`, exit code 0, no uncaught traceback).
2. **In-loop smoke training — DONE, see Implementation Report.** Launched a short rPPO run on lab node 101 (GPU 0) with `training.experiment_eval_during_training: true`, `experiment_eval_every_n_checkpoints: 1`, `experiment_eval_episodes: 3`, `experiment_eval_conditions: "avoid_pred_inj00,avoid_none_inj00,avoid_rabbit_inj00"`, WandB enabled. See the Implementation Report for the confirmed outcomes (training completion/slowdown, CSV+result.json appearance, WandB `Experiment/*` series placement, resume no-dup).
3. **Regression guard.** `run_sweep.py`'s own import surface (`_measure_cell`, `HEAD`, `KEYS`) is unchanged (not modified by this plan) and importable — confirmed by `experiment_eval_checkpoint.py` itself successfully importing it and using `_measure_cell`/`HEAD`/`KEYS` in the runner-unit smoke above (item 1). `run_sweep.py`'s own offline CSVs were not independently re-run (out of scope — `run_sweep.py` was not touched).

## Implementation Report

> **Implemented by**: [developer]
> **Date**: [date]

<!-- Filled by the implementing agent. Record: actual file line numbers touched, any
     deviation from this plan, the speed delta from the non-blocking checkpoint, and the
     smoke-run WandB run URL + the CSV paths produced. -->

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `scripts/eval/experiment_eval_checkpoint.py` | new runner | | |
| `train.py` | define_metric + dispatch + poll + drain | | |
| `configs/train/default.yaml` | 5 experiment-eval keys | | |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | new-script row | | |
| `docs/environment/02_config_schema.md` | **DEVIATION: not edited** — out of scope, see design section's "DEVIATION (not edited)" note | | |

**Conclusion**: [one-line summary]
