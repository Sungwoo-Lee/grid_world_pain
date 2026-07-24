#!/usr/bin/env python
"""Eval ONE rPPO checkpoint against the 12-condition avoidance-probe battery -> per-condition
CSVs (offline-pipeline schema) + a single `result.json` summary.

This is the reusable "eval one checkpoint" core behind the during-training experiment-eval
feature (see docs/develop/active/behavior/EXPERIMENT_EVAL_DURING_TRAINING.md). It is invoked two
ways:
  1. By `train.py`, via `subprocess.Popen`, once per Nth checkpoint save (async, non-blocking;
     the trainer never waits on this process — it only polls for `result.json`).
  2. By hand, for smoke-testing / debugging a single checkpoint standalone.

It does NOT touch WandB (`train.py` is the sole WandB writer — see the plan doc's
single-writer design). It reuses `_measure_cell`/`HEAD`/`KEYS` from
`scripts/eval/dwell_sweep/run_sweep.py` (no reimplementation of the measure/aggregate math)
and reproduces `sweep_worker.sh`'s single `eval_rollout.py --config-list` invocation + CPU
thread caps (see SCRIPTS_DEPENDENCY_MAP.md for the reuse edges this creates).

DEFAULT is SERIAL: one `eval_rollout.py --config-list` process builds the model + restores
the checkpoint ONCE, then loops all requested conditions (measured ~2 min for the full
12-condition/30-episode battery on a 20-core node — see the plan doc's Implementation
Report for the measurement that settled this). No `--npar`/parallelism flag is exposed
(there is no xargs fan-out in this single-checkpoint path, unlike the multi-node offline
`run_sweep.py`).

Failure isolation: steps 2-6 below are wrapped so ANY exception -> `result.json` still gets
written, with `status="failed"` and the traceback, and this script's own exit code is 0
(the trainer only ever sees "no result.json" or "status=failed" -- never a raised exception
or a non-zero exit that could look like something to react to).

Usage (hand-run smoke test):
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
      scripts/eval/experiment_eval_checkpoint.py \\
      --checkpoint results/JAX_RecurrentPPO/<run>/models/<episode> \\
      --result-json /tmp/experiment_smoke/result.json \\
      --out-root /tmp/experiment_smoke \\
      --conditions avoid_pred_inj00,avoid_none_inj00,avoid_rabbit_inj00 \\
      --episodes 3 --checkpoint-key <episode> --global-step 0 --iteration 0
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]  # scripts/eval/experiment_eval_checkpoint.py -> repo root (2 up)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval" / "dwell_sweep"))
from run_sweep import _measure_cell, HEAD, KEYS  # noqa: E402 (bare-name import, see SCRIPTS_DEPENDENCY_MAP.md)

PY = sys.executable
EVAL_ROLLOUT = REPO_ROOT / "scripts" / "eval" / "eval_rollout.py"
DEFAULT_PROBE_DIR = REPO_ROOT / "configs" / "environment" / "experiment" / "behavior_probes" / "core" / "avoidance"


def _capped_env(run_dir_hint: str) -> dict:
    """Child env for the `eval_rollout.py` subprocess: CPU thread caps EXACTLY as
    `sweep_worker.sh` L29-31, plus `CUDA_VISIBLE_DEVICES=""` belt-and-braces (nit fix)
    so no code path in the eval tree can touch the training GPU, plus a per-run
    persistent JAX compile cache (sweep_worker.sh L36-40) so eval #2..N at later
    checkpoints skip the ~7s compile."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TF_NUM_INTRAOP_THREADS"] = "1"
    env["TF_NUM_INTEROP_THREADS"] = "1"
    cache_key = hashlib.md5(run_dir_hint.encode()).hexdigest()[:12]
    cache_dir = f"/tmp/jaxcache_experiment_eval_{cache_key}"
    os.makedirs(cache_dir, exist_ok=True)
    env["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    return env


def _resolve_conditions(conditions_arg: str, probe_dir: Path) -> list:
    if conditions_arg == "all":
        return sorted(p.stem for p in probe_dir.glob("avoid_*.yaml"))
    return [c.strip() for c in conditions_arg.split(",") if c.strip()]


def _run_eval_and_measure(args, conds, probe_dir, out_root: Path):
    """Steps 3-5 of the module docstring: build the --config-list file, run ONE
    eval_rollout.py --config-list process (serial -- see module docstring), then
    measure + write a CSV per condition. Returns {cond: {measure: float|None}}."""
    out_root.mkdir(parents=True, exist_ok=True)
    cl_lines = []
    cond_out_dirs = {}
    for cond in conds:
        cfg_path = probe_dir / f"{cond}.yaml"
        if not cfg_path.exists():
            raise ValueError(f"condition config not found: {cfg_path}")
        cond_out = out_root / cond / str(args.checkpoint_key)
        cond_out.mkdir(parents=True, exist_ok=True)
        cond_out_dirs[cond] = cond_out
        cl_lines.append(f"{cfg_path}\t{cond_out}")

    cl_file = out_root / f"config_list_{args.checkpoint_key}.txt"
    cl_file.write_text("\n".join(cl_lines) + "\n")

    env = _capped_env(str(Path(args.checkpoint).resolve()))
    cmd = [
        PY, str(EVAL_ROLLOUT), "--config-list", str(cl_file),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--eval-n-episodes", str(args.episodes), "--record",
        "--record-n-episodes", str(args.episodes),
        "--device", "cpu", "--quiet", "--batched", "--seed", "0",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval_rollout.py failed (rc={proc.returncode}):\n"
            f"stdout(tail): {proc.stdout[-2000:]}\n"
            f"stderr(tail): {proc.stderr[-2000:]}"
        )

    measures = {}
    for cond, cond_out in cond_out_dirs.items():
        cell = _measure_cell(cond_out)
        if cell is None:
            print(f"[experiment-eval] WARNING: no recordings found for condition {cond!r} "
                  f"at {cond_out}; skipping.", file=sys.stderr)
            measures[cond] = {k: None for k in KEYS}
            continue
        step, vals = cell  # vals = [step, step_M_str, <11 CSV-formatted strings>]
        # _measure_cell returns CSV-formatted strings ("" for non-finite, "%.4f" else) --
        # convert to float/None explicitly for the JSON measures dict (nit fix).
        measures[cond] = {
            k: (None if v == "" else float(v)) for k, v in zip(KEYS, vals[2:])
        }
        csv_path = out_root / f"{cond}.csv"
        with open(csv_path, "w", newline="") as f:
            import csv as csv_mod
            w = csv_mod.writer(f)
            w.writerow(HEAD)
            w.writerow(vals)
    return measures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="<run>/models/<episode> checkpoint dir.")
    ap.add_argument("--result-json", required=True, help="Path to write the result summary JSON.")
    ap.add_argument("--out-root", required=True, help="Per-condition CSVs + recordings land here.")
    ap.add_argument("--conditions", default="all",
                    help="'all' (12 core avoidance conds) or a comma-list of stems.")
    ap.add_argument("--episodes", type=int, default=30, help="Episodes per condition.")
    ap.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR),
                    help="Directory of avoid_*.yaml condition configs.")
    ap.add_argument("--checkpoint-key", type=int, required=True,
                    help="= total_episodes_completed; echoed into result.json.")
    ap.add_argument("--global-step", type=int, required=True, help="Echoed into result.json.")
    ap.add_argument("--iteration", type=int, required=True, help="Echoed into result.json.")
    args = ap.parse_args()

    result_json = Path(args.result_json)
    out_root = Path(args.out_root)
    result_json.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "ok",
        "checkpoint_key": args.checkpoint_key,
        "global_step": args.global_step,
        "iteration": args.iteration,
        "measures": {},
    }
    try:
        probe_dir = Path(args.probe_dir)
        conds = _resolve_conditions(args.conditions, probe_dir)
        if not conds:
            raise ValueError(f"No conditions resolved from --conditions={args.conditions!r} "
                              f"--probe-dir={args.probe_dir!r}.")
        result["measures"] = _run_eval_and_measure(args, conds, probe_dir, out_root)
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()

    tmp_path = result_json.with_suffix(result_json.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, result_json)


if __name__ == "__main__":
    main()
