#!/usr/bin/env python3
"""
WandB Training Metrics Comparison Tool
=======================================
Compares training health metrics across multiple WandB runs.
Pulls episode performance, world model, actor-critic, and system metrics.

Usage:
    python scripts/compare_wandb_runs.py RUN_NAME1 RUN_NAME2 [...]
    python scripts/compare_wandb_runs.py --labels "A,B" RUN_NAME1 RUN_NAME2

Example:
    python scripts/compare_wandb_runs.py \\
        --labels "batch=64,batch=16" \\
        20260302-183207_dreamer_v3_128envs_64batch_128collect_replay1_hierarchical_1e6buffer \\
        20260302-221100_dreamer_v3_128envs_16batch_128collect_replay1_hierarchical_1e6buffer

The script matches runs by YYYYMMDD-HHMMSS timestamp prefix against WandB
creation timestamps (±120s), same as benchmark_wandb_speed.py.

Output: markdown tables with steady-state (last 20%) and trajectory metrics.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
import wandb
import numpy as np

WANDB_ENTITY = "sungwoolee"
WANDB_PROJECT = "grid_world_pain"
TIMESTAMP_TOLERANCE_SEC = 120

# Separate metric groups (logged at different intervals in WandB)
EPISODE_METRICS = [
    "Episode/Steps", "Episode/Reward", "Episode/Number",
    "Episode/Reward_Max", "Episode/Reward_Min",
    "iteration", "timesteps",
]

TRAINING_METRICS = [
    "WorldModel/loss_recon", "WorldModel/loss_rew", "WorldModel/loss_cont",
    "WorldModel/loss_dyn_kl", "WorldModel/loss_rep_kl", "WorldModel/loss_kl",
    "WorldModel/loss_model",
    "WorldModel/model_latent_entropy", "WorldModel/model_reward_mae",
    "WorldModel/model_reward_mae_pos", "WorldModel/model_reward_mae_neg",
    "WorldModel/model_cont_acc",
    "Behavior/mean_entropy", "value_mae", "Behavior/mean_advantage",
    "Behavior/mean_return", "Behavior/mean_norm_return", "Behavior/mean_value",
    "Behavior/loss_actor", "Behavior/loss_actor_policy",
    "Behavior/loss_actor_entropy", "Behavior/loss_critic",
    "Params/effective_replay_ratio",
    "iteration",
]

# Eval metrics are logged at a different interval (only during evaluation.py runs)
# Must be pulled separately to avoid empty intersection with training metrics.
EVAL_METRICS = [
    "eval/checkpoint_episode",
]


def parse_timestamp(run_name):
    """Extract YYYYMMDD-HHMMSS from the beginning of a run name."""
    ts_str = run_name.split("_", 1)[0]
    try:
        return datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def match_run(api_runs, run_name):
    """Find a WandB run by exact name or timestamp fuzzy match."""
    # Strategy 1: exact match on full run_name
    for r in api_runs:
        if r.name == run_name:
            return r
    # Strategy 2: timestamp match (same as benchmark_wandb_speed.py)
    local_ts = parse_timestamp(run_name)
    if local_ts is None:
        return None
    best_run, best_diff = None, timedelta(days=999)
    for r in api_runs:
        try:
            wandb_ts = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
            local_utc = local_ts.replace(tzinfo=timezone(timedelta(hours=9))).astimezone(timezone.utc)
            diff = abs(wandb_ts - local_utc)
            if diff < best_diff and diff < timedelta(seconds=TIMESTAMP_TOLERANCE_SEC):
                best_diff = diff
                best_run = r
        except Exception:
            continue
    return best_run


def pull_metrics(run, keys, label):
    """Pull a group of metrics from a run."""
    hist = run.history(keys=keys, samples=10000, pandas=True)
    if hist is None or len(hist) == 0:
        print(f"  WARNING: No data for {label} with keys {keys[:3]}...", file=sys.stderr)
        return None
    hist = hist.sort_values("_step").reset_index(drop=True)
    return hist


def compute_stats(series):
    """Compute summary statistics for a pandas Series."""
    col = series.dropna()
    if len(col) == 0:
        return None
    n = len(col)
    last_20_start = int(n * 0.8)
    last_20 = col.iloc[last_20_start:]
    return {
        "full_mean": float(col.mean()),
        "full_std": float(col.std()),
        "last20_mean": float(last_20.mean()) if len(last_20) > 0 else float("nan"),
        "last20_std": float(last_20.std()) if len(last_20) > 0 else float("nan"),
        "first": float(col.iloc[0]),
        "last": float(col.iloc[-1]),
        "min": float(col.min()),
        "max": float(col.max()),
        "count": n,
    }


def analyze_run(run, label):
    """Pull all metrics and compute summary statistics."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Analyzing: {label} ({run.name}, {run.id})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    results = {}

    # Pull episode metrics
    ep_hist = pull_metrics(run, EPISODE_METRICS, f"{label}/episode")
    if ep_hist is not None:
        print(f"  Episode data: {len(ep_hist)} rows", file=sys.stderr)
        for m in EPISODE_METRICS:
            if m in ep_hist.columns:
                stats = compute_stats(ep_hist[m])
                if stats:
                    results[m] = stats

    # Pull training metrics
    train_hist = pull_metrics(run, TRAINING_METRICS, f"{label}/training")
    if train_hist is not None:
        print(f"  Training data: {len(train_hist)} rows", file=sys.stderr)
        for m in TRAINING_METRICS:
            if m in train_hist.columns:
                stats = compute_stats(train_hist[m])
                if stats:
                    results[m] = stats

    # Pull eval metrics (logged at different interval, must be separate)
    eval_hist = pull_metrics(run, EVAL_METRICS, f"{label}/eval")
    if eval_hist is not None:
        print(f"  Eval data: {len(eval_hist)} rows", file=sys.stderr)
        for m in EVAL_METRICS:
            if m in eval_hist.columns:
                stats = compute_stats(eval_hist[m])
                if stats:
                    results[m] = stats

    print(f"  Metrics found: {len(results)}", file=sys.stderr)
    return results


def fmt(val, precision=4):
    """Format a float for table display."""
    if val is None or np.isnan(val):
        return "—"
    if abs(val) > 1000:
        return f"{val:,.0f}"
    if abs(val) > 10:
        return f"{val:.2f}"
    if abs(val) > 1:
        return f"{val:.3f}"
    return f"{val:.{precision}f}"


def print_comparison(results_list, labels):
    """Print comparison tables in markdown format."""

    categories = {
        "Episode Performance": [
            ("Episode/Steps", "Ep Steps (survival)", "> is better"),
            ("Episode/Reward", "Ep Reward", "> is better"),
            ("Episode/Number", "Total Episodes", "info"),
        ],
        "World Model": [
            ("WorldModel/loss_model", "loss_model (total)", "↓ better"),
            ("WorldModel/loss_recon", "loss_recon", "↓ better"),
            ("WorldModel/loss_rew", "loss_rew", "↓ better"),
            ("WorldModel/loss_dyn_kl", "loss_dyn_kl", "> 1.0"),
            ("WorldModel/loss_rep_kl", "loss_rep_kl", "> 1.0"),
            ("WorldModel/loss_kl", "loss_kl", "stable"),
            ("WorldModel/model_latent_entropy", "latent_entropy", "1.0-2.5"),
            ("WorldModel/model_reward_mae_pos", "reward_mae_pos", "> 0"),
            ("WorldModel/model_reward_mae_neg", "reward_mae_neg", "info"),
            ("WorldModel/model_cont_acc", "cont_acc", "> 0.95"),
        ],
        "Actor-Critic": [
            ("Behavior/mean_entropy", "mean_entropy", "> 0.5"),
            ("value_mae", "value_mae", "< 5"),
            ("Behavior/mean_advantage", "mean_advantage", "non-trivial"),
            ("Behavior/mean_return", "mean_return", "info"),
            ("Behavior/mean_norm_return", "mean_norm_return", "info"),
            ("Behavior/mean_value", "mean_value", "info"),
            ("Behavior/loss_actor_policy", "loss_actor_policy", "info"),
            ("Behavior/loss_actor_entropy", "loss_actor_entropy", "meaningful"),
            ("Behavior/loss_critic", "loss_critic", "↓ better"),
        ],
        "System": [
            ("Params/effective_replay_ratio", "eff_replay_ratio", "≈ 1.0"),
            ("eval/checkpoint_episode", "eval_checkpoint", "info"),
        ],
    }

    for cat_name, metric_list in categories.items():
        print(f"\n### {cat_name}")
        print()
        header = "| Metric | Criterion"
        for label in labels:
            header += f" | {label} (steady-state) | {label} (last)"
        header += " |"
        print(header)
        sep = "|:---|:---"
        for _ in labels:
            sep += "|---:|---:"
        sep += "|"
        print(sep)

        for metric_key, display_name, criterion in metric_list:
            row = f"| `{display_name}` | {criterion}"
            for res in results_list:
                if res and metric_key in res:
                    m = res[metric_key]
                    row += f" | {fmt(m['last20_mean'])} ± {fmt(m['last20_std'])}"
                    row += f" | {fmt(m['last'])}"
                else:
                    row += " | — | —"
            row += " |"
            print(row)

    # Trajectory summary
    print("\n### Trajectory Summary (first → last)")
    print()
    header = "| Metric"
    for label in labels:
        header += f" | {label}"
    header += " |"
    print(header)
    sep = "|:---"
    for _ in labels:
        sep += "|:---"
    sep += "|"
    print(sep)

    trajectory_metrics = [
        ("Episode/Steps", "Ep_Steps"),
        ("Episode/Reward", "Ep_Reward"),
        ("Behavior/mean_entropy", "mean_entropy"),
        ("value_mae", "value_mae"),
        ("WorldModel/loss_recon", "loss_recon"),
        ("WorldModel/model_reward_mae_pos", "reward_mae_pos"),
        ("WorldModel/model_latent_entropy", "latent_entropy"),
    ]
    for mk, short in trajectory_metrics:
        row = f"| `{short}`"
        for res in results_list:
            if res and mk in res:
                m = res[mk]
                row += f" | {fmt(m['first'])} → {fmt(m['last'])} (n={m['count']})"
            else:
                row += " | —"
        row += " |"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Compare training health metrics across WandB runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Results directory names (e.g. 20260302-183207_dreamer_v3_...)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels for each run (e.g. 'batch=64,batch=16')",
    )
    parser.add_argument(
        "--entity", default=WANDB_ENTITY,
        help=f"WandB entity (default: {WANDB_ENTITY})",
    )
    parser.add_argument(
        "--project", default=WANDB_PROJECT,
        help=f"WandB project (default: {WANDB_PROJECT})",
    )
    args = parser.parse_args()

    # Generate labels if not provided
    if args.labels is not None:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = [f"Run{i+1}" for i in range(len(args.runs))]
    if len(labels) != len(args.runs):
        print(f"ERROR: {len(labels)} labels for {len(args.runs)} runs", file=sys.stderr)
        sys.exit(1)

    # Fetch WandB runs
    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    print(f"Fetching runs from {path}...", file=sys.stderr)
    try:
        api_runs = list(api.runs(path))
    except Exception:
        path = args.project
        api_runs = list(api.runs(path))
    print(f"Found {len(api_runs)} runs.", file=sys.stderr)

    # Match and analyze
    results_list = []
    for run_name, label in zip(args.runs, labels):
        matched = match_run(api_runs, run_name)
        if matched is None:
            print(f"  ✗ {run_name}: no match", file=sys.stderr)
            results_list.append(None)
            continue
        print(f"  ✓ {label} → {matched.name} ({matched.id})", file=sys.stderr)
        results_list.append(analyze_run(matched, label))

    print_comparison(results_list, labels)


if __name__ == "__main__":
    main()
