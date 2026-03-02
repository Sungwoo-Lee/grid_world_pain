#!/usr/bin/env python3
"""
WandB Speed Benchmark Tool
===========================
Extracts training speed metrics from WandB logs for a list of training runs.

Usage:
    python scripts/benchmark_wandb_speed.py RUN_NAME1 RUN_NAME2 ...

Example:
    python scripts/benchmark_wandb_speed.py \
        20260226-141145_dreamer_v3_64env_replayRatio1_collectInterval1 \
        20260301-213626_rppoNMN_MC_relu_128hidden_GRU_hierarchical

The script extracts the YYYYMMDD-HHMMSS prefix from each run name and
matches it against WandB runs by comparing creation timestamps (±120s).
It then downloads the run's logged history to compute speed metrics.

Output: a markdown table with s/it, it/s, SPS, and total wall-clock time.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
import wandb


# ── WandB project coordinates ──────────────────────────────────────────────
WANDB_ENTITY = "sungwoolee"
WANDB_PROJECT = "grid_world_pain"
TIMESTAMP_TOLERANCE_SEC = 120  # max allowed drift between run name and WandB created_at


def parse_timestamp_from_name(run_name: str) -> datetime | None:
    """Extract YYYYMMDD-HHMMSS from the beginning of a run name."""
    # Format: 20260226-141145_dreamer_v3_...
    parts = run_name.split("_", 1)
    ts_str = parts[0]  # e.g. "20260226-141145"
    try:
        return datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def match_wandb_run(api_runs, run_name: str):
    """
    Find the WandB run matching `run_name`.

    Strategy (in order):
      1. Exact match on run.name  (WandB name == results dir name)
      2. Timestamp match: extract YYYYMMDD-HHMMSS from run_name and compare
         against each WandB run's created_at within ±TIMESTAMP_TOLERANCE_SEC.
    """
    # Strategy 1: exact name match
    for r in api_runs:
        if r.name == run_name:
            return r

    # Strategy 2: timestamp fuzzy match
    local_ts = parse_timestamp_from_name(run_name)
    if local_ts is None:
        return None

    best_run = None
    best_diff = timedelta(days=999)
    for r in api_runs:
        try:
            # WandB stores created_at as ISO string (UTC)
            wandb_ts = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
            # Convert our local timestamp to UTC+9 (KST) then to UTC for comparison
            local_utc = local_ts.replace(tzinfo=timezone(timedelta(hours=9))).astimezone(timezone.utc)
            diff = abs(wandb_ts - local_utc)
            if diff < best_diff and diff < timedelta(seconds=TIMESTAMP_TOLERANCE_SEC):
                best_diff = diff
                best_run = r
        except Exception:
            continue

    return best_run


def compute_speed_metrics(run) -> dict:
    """Download history and compute speed metrics for a single WandB run."""
    hist = run.history(keys=["_timestamp", "timesteps", "iteration"], samples=5000, pandas=True)

    if hist is None or len(hist) < 2:
        return {"error": "not enough data"}

    # Sort by _step (the internal WandB step counter)
    hist = hist.sort_values("_step").reset_index(drop=True)

    # ── Total wall-clock ────────────────────────────────────────────────
    total_time_s = hist["_timestamp"].iloc[-1] - hist["_timestamp"].iloc[0]

    # ── Time per logged step ────────────────────────────────────────────
    hist["dt"] = hist["_timestamp"].diff()
    hist["d_iter"] = hist["iteration"].diff()

    # Filter out anomalies (< 0.05s or > 300s between logs, or d_iter <= 0)
    valid = hist[(hist["dt"] > 0.05) & (hist["dt"] < 300) & (hist["d_iter"] > 0)].copy()

    if len(valid) == 0:
        return {"error": "no valid intervals"}

    # Normalise: seconds per single iteration (not per logged step)
    valid["s_per_iter"] = valid["dt"] / valid["d_iter"]
    mean_s_per_iter = float(valid["s_per_iter"].mean())

    # ── Env steps per second (SPS) ──────────────────────────────────────
    valid["d_timesteps"] = hist.loc[valid.index, "timesteps"].diff()
    sps_vals = valid[(valid["d_timesteps"] > 0)].copy()
    if len(sps_vals) > 0:
        sps_vals["sps"] = sps_vals["d_timesteps"] / sps_vals["dt"]
        mean_sps = float(sps_vals["sps"].mean())
    else:
        mean_sps = float("nan")

    # ── Final iteration count ───────────────────────────────────────────
    total_iters = int(hist["iteration"].max() - hist["iteration"].min())
    total_timesteps = int(hist["timesteps"].max())

    return {
        "s_per_iter": mean_s_per_iter,
        "it_per_s": 1.0 / mean_s_per_iter if mean_s_per_iter > 0 else float("nan"),
        "sps": mean_sps,
        "total_time_s": total_time_s,
        "total_time_human": format_duration(total_time_s),
        "total_iters": total_iters,
        "total_timesteps": total_timesteps,
    }


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Extract training speed metrics from WandB logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Results directory names (e.g. 20260226-141145_dreamer_v3_64env_...)",
    )
    parser.add_argument(
        "--entity", default=WANDB_ENTITY,
        help=f"WandB entity (default: {WANDB_ENTITY})",
    )
    parser.add_argument(
        "--project", default=WANDB_PROJECT,
        help=f"WandB project (default: {WANDB_PROJECT})",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Output CSV instead of markdown table",
    )
    args = parser.parse_args()

    # ── Fetch all WandB runs ────────────────────────────────────────────
    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    print(f"Fetching runs from {path}...", file=sys.stderr)
    try:
        api_runs = list(api.runs(path))
    except Exception:
        # Fallback: try without entity
        path = args.project
        api_runs = list(api.runs(path))
    print(f"Found {len(api_runs)} WandB runs.", file=sys.stderr)

    # ── Process each target run ─────────────────────────────────────────
    results = []
    for run_name in args.runs:
        matched = match_wandb_run(api_runs, run_name)
        if matched is None:
            results.append({"name": run_name, "error": "no WandB match"})
            print(f"  ✗ {run_name}: no WandB match found", file=sys.stderr)
            continue

        print(f"  ✓ {run_name} → WandB run '{matched.name}' ({matched.id})", file=sys.stderr)
        metrics = compute_speed_metrics(matched)
        metrics["name"] = run_name
        results.append(metrics)

    # ── Output ──────────────────────────────────────────────────────────
    if args.csv:
        print("run_name,s_per_iter,it_per_s,sps,total_time,total_iters,total_timesteps")
        for r in results:
            if "error" in r:
                print(f"{r['name']},ERROR: {r['error']},,,,")
            else:
                print(f"{r['name']},{r['s_per_iter']:.4f},{r['it_per_s']:.3f},"
                      f"{r['sps']:.1f},{r['total_time_human']},{r['total_iters']},"
                      f"{r['total_timesteps']}")
    else:
        # Markdown table
        print()
        print("| Run | s/it | it/s | SPS | Total Time | Iterations | Timesteps |")
        print("|:---|---:|---:|---:|:---|---:|---:|")
        for r in results:
            short = r["name"]
            if "error" in r:
                print(f"| {short} | — | — | — | {r['error']} | — | — |")
            else:
                print(f"| {short} | {r['s_per_iter']:.4f} | {r['it_per_s']:.2f} "
                      f"| {r['sps']:.0f} | {r['total_time_human']} "
                      f"| {r['total_iters']:,} | {r['total_timesteps']:,} |")
        print()


if __name__ == "__main__":
    main()
