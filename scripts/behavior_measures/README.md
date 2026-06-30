# scripts/behavior_measures/

Experiment-specific analysis scripts for the **interoceptive behavior-measure study**
(see `docs/experiments/active/behavior_measures/interoceptive_behavior_measure_study.md`).
These post-process eval-rollout recordings into measures, statistics, and figures — kept
separate from the general-purpose scripts in `scripts/` so this study's tooling is organized
in one place.

All scripts run from the repo root with the project conda interpreter
(`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`).

## Scripts

- **`avoidance_stats_heatmap.py`** — computes the canonical avoidance measures
  (bush-use rate / bush-entry step / bush-dwell fraction / flight-initiation distance (FID) /
  animal-proximity fraction / closest approach / injury change / survival) per episode from
  `.rec.gz` recordings, aggregates mean±std over seeds per config, writes a stats CSV, and renders
  an annotated heatmap (rows = experiment, columns = criterion).

  ```bash
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    scripts/behavior_measures/avoidance_stats_heatmap.py \
    --results-root results/eval/avoidance_stat \
    --out-dir results/eval/avoidance_stat/STATS
  ```

  Expects the layout `<results-root>/<config>/models/<ckpt>/recordings/<ckpt>/episode_*.rec.gz`
  (as written by `scripts/eval_rollout.py --record`). Configs auto-discovered; `--ckpt`,
  `--configs`, `--title` optional. To generate the input recordings, run the stat-variant configs
  in `configs/environment/experiment/behavior_probes/explore/avoidance_stat/` with many seeds.


- **`_heatmap_style.py`** — shared journal-style heatmap renderer (`render(...)`). Card-style cells,
  per-criterion colour (sequential `crest` for magnitude metrics; diverging-at-0 `RdBu_r` for signed
  metrics like injury change), two-tier mean/std annotation, column-group headers, row-group
  separators, Helvetica-like font, and **PNG (300 dpi) + PDF (vector)** output for submission.
  Imported by `avoidance_stats_heatmap.py`; reuse for future behavior-measure matrices.
