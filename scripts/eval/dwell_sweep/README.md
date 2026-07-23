# Dwell-history / behavior-metrics sweep pipeline

## What this is

For a set of training runs, this pipeline rolls out every saved checkpoint against a
fixed battery of "behavior probe" scenarios (e.g. "a predator is present" vs "no
animal", crossed with "agent starts injured" vs "not"), computes 11 behavior measures
per rollout (time spent hiding in the bush, how close the agent lets a predator get,
how much ground it covers, etc. -- see `scripts/behavior_measures/avoidance_stats_heatmap.py`
for the full list and definitions), and plots how those measures evolve over training.
One CSV row per checkpoint, one CSV per (run, probe condition), one stacked-row figure
per measure per run.

It works identically for both algorithms the project trains: rPPO (`JAX_RecurrentPPO`)
and Dreamer (`JAX_DreamerSRL`), via the same unified `scripts/eval/eval_rollout.py
--batched --record` call -- they differ only in the `--checkpoint` path convention and
Dreamer's extra `--agent_config`.

**As of 2026-07-23**, each worker call evaluates ONE checkpoint against ALL of its still-
pending probe conditions in a single `eval_rollout.py --config-list` process, instead of
one process per (checkpoint, condition) pair -- the model is built and the checkpoint
restored ONCE and reused across every condition. This matters because a single
`eval_rollout.py` call's wall time is dominated by a flat ~7s "build model + restore
checkpoint" cost that does not depend on episode count, so evaluating a checkpoint's 12
core probe conditions used to mean paying that ~7s twelve times over; now it's paid once
per checkpoint (~5.9x fewer core-seconds measured on a real rPPO checkpoint, see "Tuning
notes").

This was built as ad-hoc gitignored scripts under `tmp/` across ~7 real sweeps this
session (`tmp/dist_metrics_worker.sh`, `tmp/dist_dreamer_worker_batched.sh`,
`tmp/aggregate_dreamer_metrics.py`, `tmp/plot_metrics_summary.py`) and is promoted here
as one committed, reproducible, self-serve tool.

## Quick start

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    scripts/eval/dwell_sweep/run_sweep.py configs/eval_sweeps/basic04_variants_rppo.yaml
```

Before launching against real nodes: check which nodes are actually free
(`scripts/lab/gpu_status.py` or the `gpu-status` skill) and edit the spec's `nodes:`
list if any are busy -- the pipeline does not check this for you.

Add `--dry-run` to build the worklist and LPT partition and print the exact launch
commands without touching the cluster:

```bash
... run_sweep.py configs/eval_sweeps/basic04_variants_rppo.yaml --dry-run
```

Add `--max-checkpoints N` to cap each (run, condition) pair to its newest N *pending*
checkpoints -- useful for a quick smoke test before committing to a full sweep:

```bash
... run_sweep.py configs/eval_sweeps/basic04_variants_rppo.yaml --max-checkpoints 3
```

## Files

| File | Role |
|---|---|
| `run_sweep.py` | The driver. Reads a spec, enumerates + incrementally filters checkpoints, LPT-partitions work across nodes **by checkpoint** (see below), launches `sweep_worker.sh` on each node via `run_command.py`, polls for completion, aggregates recordings into CSVs, renders figures. |
| `sweep_worker.sh` | The unified per-node worker. Eval-only: for each line in its worklist (one CHECKPOINT + all its pending conditions), expands the conditions into a `--config-list` file and runs `eval_rollout.py --batched --record --config-list` ONCE, writing every condition's `.rec.gz` recordings to its own scratch subdir. No aggregation (that's the driver's job -- one path, not two). |
| `plot_summary.py` | Stacked-row history figures from a directory of `avoid_*.csv` files (one row per probe condition). Promoted as-is from `tmp/plot_metrics_summary.py`; `fig_for()` is imported directly by `run_sweep.py`. |

## Spec schema

A spec is a YAML file under `configs/eval_sweeps/`. Two worked examples ship in that
directory: `basic04_variants_rppo.yaml` (rPPO) and `basic04_rr_dreamer.yaml` (Dreamer).

```yaml
name: basic04_variants          # sweep name; also the default output subdir name
algo: rppo                      # rppo | dreamer -- mandatory, whole spec is one algorithm
output_dir: results/eval/avoidance/<name>   # optional; defaults to results/eval/avoidance/<name>
probe: clean                    # clean | noise -- clean = configs/.../core/avoidance,
                                 #   noise = configs/.../explore/avoidance_stat_noise
conditions: all                 # "all" (the 12 core avoidance probes) or an explicit
                                 #   list of condition names, e.g. [avoid_pred_inj00, avoid_none_inj00]
episodes: 30                    # eval + record episodes per checkpoint (default 30)
nodes: [106, 107, 108, 109]     # node IDs to fan work across (LPT-balanced by pending-checkpoint count)
npar: null                      # per-node parallelism; null -> tuned default per algo (see below)
max_checkpoints: null           # optional cap on newest-N PENDING checkpoints per (run,cond); also a CLI flag
x_axis: null                    # steps | episodes; null -> steps for rppo, episodes for dreamer
plot_measures: [bush_dwell, spatial_spread, survival_steps]   # any of the 11 measure columns
runs:
  - label: v01_slowmove         # subdirectory name under output_dir; also the figure title
    path: "results/JAX_RecurrentPPO/*rppo_b04v01_slowmove*"   # exact path OR a glob that
                                 #   must resolve to EXACTLY ONE directory (the driver
                                 #   raises if it resolves to 0 or >1)
    # agent_config: only meaningful for algo: dreamer; auto-detected as
    #   <run_dir>/models/agent_config.yaml if omitted (that's where train.py saves it)
```

A "condition" name like `avoid_pred_inj00` decodes as: animal = `pred` (a hunting
predator; other values are `none`, `rabbit`, `rabbit_olfzero`, `rabbitwander`,
`rabbitwander_predsmell`), starting injury = `00` (vs `70`). The 12 core conditions are
every animal x injury combination; see
`configs/environment/experiment/behavior_probes/core/avoidance/` for the actual YAML
files (each one fully specifies the probe scenario: grid size, bush location, predator
spawn point, etc.).

## Tuning notes (why the numbers are what they are)

- **NPAR defaults: rPPO=18, Dreamer(batched)=5.** The eval sweep is CPU-bound (each
  `eval_rollout.py` process spends the bulk of its per-checkpoint time in JAX import +
  XLA compile, and that compile step is itself multithreaded). rPPO's single-env
  rollout is light enough that ~1 process per core (NPAR~=core count, here 18) keeps
  load near the core count. Dreamer's `--batched` path vmaps over all eval episodes at
  once, and that vmap parallelises across cores on its own *even with the 1-thread
  caps below* -- running it at rPPO's NPAR drove a 20-core node to load 142
  (thrashing). ~5 keeps batched Dreamer near the core count instead. If you add a new
  algorithm or change the batch size, re-measure before trusting either default.
- **Thread caps + persistent compile cache.** Every worker process runs with
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1` and
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`, plus
  a per-node `JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache_dwellsweep_<node>` (persistent
  across the whole worklist -- the compiled program depends only on tensor *shape*,
  identical across a model's checkpoints, so only the first eval per node pays the
  ~7s compile). Together these took a cold NPAR=16 run from load 100 down to NPAR=18
  warm-cache+capped at load 29, at ~4x lower per-eval wall time. Do not drop the
  caps or the cache dir without re-measuring.
- **Incremental.** Before building the worklist, the driver reads each target CSV's
  max `step` column and only schedules checkpoints newer than that. Re-running a spec
  after a run has trained further only evaluates the new checkpoints and MERGES the
  new rows into the existing CSV (old rows are never dropped, even if a different
  condition's CSV is further behind).
- **Recursive-glob aggregation.** `eval_rollout.py --batched` writes recordings to
  `{output-root}/{run_tag}/{ckpt}/recordings/{ckpt}/episode_*.rec.gz` -- one directory
  level deeper than a naive per-checkpoint output-root. The aggregation step globs
  `{step_dir}/**/episode_*.rec.gz` (recursive) so it finds recordings regardless of
  that nesting, and this is the SAME code path for both algorithms (unlike the
  original tmp/ scripts, which had a separate rPPO aggregator baked into the worker and
  a standalone Dreamer aggregator script).
- **LPT partition -- CHECKPOINT-granularity, not condition-granularity.** Work is grouped
  by (run, checkpoint) -- e.g. "v01_slowmove step 8900007" is one cell carrying every
  condition still pending for that checkpoint (a checkpoint can be ahead on some
  conditions and behind on others under the incremental filter; only the actually-pending
  ones are attached). Cells are sorted by pending-condition count descending and each
  WHOLE cell is assigned to whichever node currently has the smallest running total --
  a cell is never split across nodes, because that's what lets one `eval_rollout.py
  --config-list` process build the model + restore that checkpoint once and loop over
  every pending condition (the entire point of the 2026-07-23 change -- see the top of
  this README). The worklist line format is `CHECKPOINT|AGENT|EPISODE|CFG1,OUT1;
  CFG2,OUT2;...`; `sweep_worker.sh` expands the `;`-separated pairs into a
  `--config-list` file per invocation.

## Node safety

`run_sweep.py` does not check which nodes are busy -- that's on you (or the
`training-runner` agent) before editing a spec's `nodes:` list. Never point a sweep at
a node running live GPU training; the eval workers run on CPU (`--device cpu`) so they
don't contend for GPU memory, but they do add CPU load that can starve a training run's
data pipeline. As of this pipeline's creation, nodes 106/107/111/112/114 had live
training and were excluded from validation.

## Output layout

```
results/eval/avoidance/<name>/
├── <run_label>/
│   ├── avoid_pred_inj00.csv       # step, step_M, + 11 measure columns
│   ├── avoid_none_inj00.csv
│   ├── ...
│   ├── FIG_bush_dwell.png
│   └── FIG_spatial_spread.png
└── _scratch/                       # transient; safe to delete after a run completes
    ├── _worklists/worklist_<node>.txt
    ├── _run_markers/{npar,prog,done,fail}_<node>
    └── <run_label>/<cond>/<step>/...recordings.../episode_*.rec.gz
```
