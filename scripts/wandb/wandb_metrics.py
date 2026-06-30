#!/usr/bin/env python3
"""
WandB Metrics Tool — General-Purpose Training Analysis
========================================================
Discover, extract, and compare metrics from any WandB training run,
regardless of algorithm (DreamerV3, PPO, SAC, etc.).

Subcommands:
    config      — Show run hyperparameters/config
    discover    — List all metrics logged in a run, grouped by prefix
    extract     — Pull time-series stats for a single run
    compare     — Compare metrics across multiple runs
    timeseries  — Export windowed or raw time-series data for temporal analysis

Usage:
    # Show what algorithm/config a run used
    python scripts/wandb_metrics.py config RUN_NAME

    # Discover what metrics a run has
    python scripts/wandb_metrics.py discover RUN_NAME

    # Extract metrics from a single run
    python scripts/wandb_metrics.py extract RUN_NAME
    python scripts/wandb_metrics.py extract RUN_NAME --metrics "Episode/*,loss_*"

    # Compare runs (auto-discovery)
    python scripts/wandb_metrics.py compare RUN1 RUN2 --labels "A,B"

    # Compare with metric filter
    python scripts/wandb_metrics.py compare RUN1 RUN2 --metrics "Episode/*,WorldModel/*"

    # Compare with a named preset
    python scripts/wandb_metrics.py compare RUN1 RUN2 --preset dreamer_v3

    # Time-series: windowed summary (default 10 windows)
    python scripts/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps"
    python scripts/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps" --windows 20

    # Time-series: raw downsampled data points
    python scripts/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps" --raw

    # Time-series: compare two runs side-by-side
    python scripts/wandb_metrics.py timeseries RUN1 RUN2 --metrics "Episode/Steps" --labels "A,B"
"""

import argparse
import fnmatch
import json
import sys

import numpy as np

from wandb_utils import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    compute_stats,
    fetch_wandb_runs,
    fmt,
    match_wandb_run,
)

# ── WandB internal keys to always exclude ─────────────────────────────────────
INTERNAL_PREFIXES = ("_", "system/")


# ── Presets ───────────────────────────────────────────────────────────────────
# Each preset maps category_name → [(metric_key, display_name, criterion), ...]
PRESETS = {
    "dreamer_v3": {
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
    },
    "recurrent_ppo": {
        "Episode Performance": [
            ("Episode/Steps", "Ep Steps", "> is better"),
            ("Episode/Reward", "Ep Reward", "> is better"),
            ("Episode/Number", "Total Episodes", "info"),
        ],
        "Policy": [
            ("Policy/entropy", "entropy", "> 0.5"),
            ("Policy/policy_loss", "policy_loss", "info"),
            ("Policy/value_loss", "value_loss", "↓ better"),
            ("Policy/approx_kl", "approx_kl", "< 0.03"),
            ("Policy/clip_fraction", "clip_fraction", "0.1-0.3"),
        ],
        "System": [
            ("iteration", "iteration", "info"),
            ("timesteps", "timesteps", "info"),
        ],
    },
}


# ── Metric discovery ──────────────────────────────────────────────────────────

def discover_metrics(run) -> dict[str, list[str]]:
    """
    Discover all metrics in a run by inspecting run.summary.
    Returns {group_name: [metric_key, ...]} grouped by prefix.
    """
    all_keys = sorted(run.summary.keys())
    # Filter out internal WandB keys
    keys = [k for k in all_keys if not any(k.startswith(p) for p in INTERNAL_PREFIXES)]

    groups: dict[str, list[str]] = {}
    for key in keys:
        if "/" in key:
            group = key.split("/", 1)[0]
        else:
            group = "General"
        groups.setdefault(group, []).append(key)

    return groups


def filter_metrics(all_keys: list[str], patterns: list[str]) -> list[str]:
    """Filter metric keys by glob patterns (e.g., 'Episode/*', 'loss_*')."""
    matched = set()
    for pattern in patterns:
        for key in all_keys:
            if fnmatch.fnmatch(key, pattern):
                matched.add(key)
    return sorted(matched)


# ── Pull & analyze ────────────────────────────────────────────────────────────

def pull_and_analyze(run, keys: list[str], label: str = "") -> dict[str, dict]:
    """
    Pull time-series for a list of metric keys and compute stats for each.
    Returns {metric_key: stats_dict}.
    """
    if not keys:
        return {}

    # WandB may log different metrics at different intervals.
    # Pull in batches grouped by prefix to reduce missing-data issues.
    prefix_groups: dict[str, list[str]] = {}
    for k in keys:
        prefix = k.split("/", 1)[0] if "/" in k else "_root"
        prefix_groups.setdefault(prefix, []).append(k)

    results = {}
    for prefix, group_keys in prefix_groups.items():
        hist = run.history(keys=group_keys, samples=10000, pandas=True)
        if hist is None or len(hist) == 0:
            if label:
                print(f"  WARNING: No data for {label}/{prefix}", file=sys.stderr)
            continue
        hist = hist.sort_values("_step").reset_index(drop=True)

        for mk in group_keys:
            if mk in hist.columns:
                stats = compute_stats(hist[mk])
                if stats:
                    results[mk] = stats

    return results


# ── Output formatters ─────────────────────────────────────────────────────────

def print_discover(groups: dict[str, list[str]], run_name: str):
    """Print discovered metrics in a readable format."""
    total = sum(len(v) for v in groups.values())
    print(f"\n## Metrics in `{run_name}` ({total} total)\n")
    for group_name in sorted(groups.keys()):
        keys = groups[group_name]
        print(f"### {group_name} ({len(keys)} metrics)")
        print()
        for k in keys:
            print(f"- `{k}`")
        print()


def print_extract(results: dict[str, dict], run_name: str, groups: dict[str, list[str]]):
    """Print extracted metrics for a single run."""
    print(f"\n## Metrics for `{run_name}`\n")

    # Organize by group
    for group_name in sorted(groups.keys()):
        group_keys = [k for k in groups[group_name] if k in results]
        if not group_keys:
            continue
        print(f"### {group_name}")
        print()
        print("| Metric | Steady-State (last 20%) | Final | Min | Max | N |")
        print("|:---|---:|---:|---:|---:|---:|")
        for mk in group_keys:
            s = results[mk]
            print(
                f"| `{mk}` "
                f"| {fmt(s['last20_mean'])} ± {fmt(s['last20_std'])} "
                f"| {fmt(s['last'])} "
                f"| {fmt(s['min'])} "
                f"| {fmt(s['max'])} "
                f"| {s['count']} |"
            )
        print()

    # Trajectory summary
    print("### Trajectory Summary (first → last)")
    print()
    print("| Metric | Trajectory |")
    print("|:---|:---|")
    for mk in sorted(results.keys()):
        s = results[mk]
        print(f"| `{mk}` | {fmt(s['first'])} → {fmt(s['last'])} (n={s['count']}) |")
    print()


def print_compare_auto(results_list: list[dict], labels: list[str],
                       groups: dict[str, list[str]]):
    """Print comparison tables using auto-discovered groups."""
    print(f"\n## Comparison: {' vs '.join(labels)}\n")

    # Collect all metric keys found across all runs
    all_found_keys = set()
    for res in results_list:
        if res:
            all_found_keys.update(res.keys())

    for group_name in sorted(groups.keys()):
        group_keys = [k for k in groups[group_name] if k in all_found_keys]
        if not group_keys:
            continue

        print(f"### {group_name}")
        print()
        header = "| Metric"
        for label in labels:
            header += f" | {label} (steady-state) | {label} (last)"
        header += " |"
        print(header)
        sep = "|:---"
        for _ in labels:
            sep += "|---:|---:"
        sep += "|"
        print(sep)

        for mk in group_keys:
            row = f"| `{mk}`"
            for res in results_list:
                if res and mk in res:
                    s = res[mk]
                    row += f" | {fmt(s['last20_mean'])} ± {fmt(s['last20_std'])}"
                    row += f" | {fmt(s['last'])}"
                else:
                    row += " | — | —"
            row += " |"
            print(row)
        print()

    # Trajectory summary
    sorted_keys = sorted(all_found_keys)
    print("### Trajectory Summary (first → last)")
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
    for mk in sorted_keys:
        row = f"| `{mk}`"
        for res in results_list:
            if res and mk in res:
                s = res[mk]
                row += f" | {fmt(s['first'])} → {fmt(s['last'])} (n={s['count']})"
            else:
                row += " | —"
        row += " |"
        print(row)
    print()


def print_compare_preset(results_list: list[dict], labels: list[str],
                         preset: dict[str, list]):
    """Print comparison tables using a named preset with criteria."""
    print(f"\n## Comparison: {' vs '.join(labels)}\n")

    for cat_name, metric_list in preset.items():
        print(f"### {cat_name}")
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
                    s = res[metric_key]
                    row += f" | {fmt(s['last20_mean'])} ± {fmt(s['last20_std'])}"
                    row += f" | {fmt(s['last'])}"
                else:
                    row += " | — | —"
            row += " |"
            print(row)
        print()

    # Trajectory summary from preset metrics
    print("### Trajectory Summary (first → last)")
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
    for cat_name, metric_list in preset.items():
        for mk, short, _ in metric_list:
            row = f"| `{short}`"
            for res in results_list:
                if res and mk in res:
                    s = res[mk]
                    row += f" | {fmt(s['first'])} → {fmt(s['last'])} (n={s['count']})"
                else:
                    row += " | —"
            row += " |"
            print(row)
    print()


# ── Subcommand handlers ──────────────────────────────────────────────────────

def cmd_config(args):
    """Handle the 'config' subcommand — show run hyperparameters."""
    api_runs = fetch_wandb_runs(args.entity, args.project)
    run_name = args.run
    matched = match_wandb_run(api_runs, run_name)
    if matched is None:
        print(f"ERROR: No WandB match for '{run_name}'", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ {run_name} → {matched.name} ({matched.id})", file=sys.stderr)

    config = matched.config
    print(f"\n## Config for `{run_name}`\n")
    print(f"- **Run name**: `{matched.name}`")
    print(f"- **Run ID**: `{matched.id}`")
    print(f"- **State**: {matched.state}")
    print(f"- **Created**: {matched.created_at}")
    if hasattr(matched, 'tags') and matched.tags:
        print(f"- **Tags**: {', '.join(matched.tags)}")
    print()

    if not config:
        print("_No config logged for this run._")
        return

    if args.json:
        print("```json")
        print(json.dumps(config, indent=2, default=str))
        print("```")
    else:
        # Print as a readable nested table
        print("| Key | Value |")
        print("|:---|:---|")
        for key in sorted(config.keys()):
            val = config[key]
            if isinstance(val, dict):
                # Flatten one level of nesting
                for subkey in sorted(val.keys()):
                    print(f"| `{key}.{subkey}` | `{val[subkey]}` |")
            else:
                print(f"| `{key}` | `{val}` |")
        print()


def cmd_discover(args):
    """Handle the 'discover' subcommand."""
    api_runs = fetch_wandb_runs(args.entity, args.project)
    run_name = args.run
    matched = match_wandb_run(api_runs, run_name)
    if matched is None:
        print(f"ERROR: No WandB match for '{run_name}'", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ {run_name} → {matched.name} ({matched.id})", file=sys.stderr)

    groups = discover_metrics(matched)
    print_discover(groups, run_name)


def cmd_extract(args):
    """Handle the 'extract' subcommand."""
    api_runs = fetch_wandb_runs(args.entity, args.project)
    run_name = args.run
    matched = match_wandb_run(api_runs, run_name)
    if matched is None:
        print(f"ERROR: No WandB match for '{run_name}'", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ {run_name} → {matched.name} ({matched.id})", file=sys.stderr)

    groups = discover_metrics(matched)
    all_keys = [k for keys in groups.values() for k in keys]

    # Apply metric filter if provided
    if args.metrics:
        patterns = [p.strip() for p in args.metrics.split(",")]
        all_keys = filter_metrics(all_keys, patterns)
        # Rebuild groups from filtered keys
        groups = {}
        for k in all_keys:
            g = k.split("/", 1)[0] if "/" in k else "General"
            groups.setdefault(g, []).append(k)

    print(f"  Pulling {len(all_keys)} metrics...", file=sys.stderr)
    results = pull_and_analyze(matched, all_keys, run_name)
    print(f"  Got stats for {len(results)} metrics.", file=sys.stderr)

    print_extract(results, run_name, groups)


def cmd_compare(args):
    """Handle the 'compare' subcommand."""
    api_runs = fetch_wandb_runs(args.entity, args.project)

    # Parse labels
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = [f"Run{i+1}" for i in range(len(args.runs))]
    if len(labels) != len(args.runs):
        print(f"ERROR: {len(labels)} labels for {len(args.runs)} runs", file=sys.stderr)
        sys.exit(1)

    # Determine mode: preset vs auto-discovery
    use_preset = args.preset and args.preset in PRESETS

    # Match runs
    matched_runs = []
    for run_name, label in zip(args.runs, labels):
        matched = match_wandb_run(api_runs, run_name)
        if matched is None:
            print(f"  ✗ {run_name}: no match", file=sys.stderr)
            matched_runs.append(None)
        else:
            print(f"  ✓ {label} → {matched.name} ({matched.id})", file=sys.stderr)
            matched_runs.append(matched)

    if use_preset:
        # Preset mode: pull only the metrics defined in the preset
        preset = PRESETS[args.preset]
        preset_keys = []
        for metric_list in preset.values():
            for mk, _, _ in metric_list:
                if mk not in preset_keys:
                    preset_keys.append(mk)

        results_list = []
        for matched, label in zip(matched_runs, labels):
            if matched is None:
                results_list.append(None)
                continue
            results_list.append(pull_and_analyze(matched, preset_keys, label))

        print_compare_preset(results_list, labels, preset)
    else:
        # Auto-discovery mode: discover metrics from the first available run
        ref_run = next((r for r in matched_runs if r is not None), None)
        if ref_run is None:
            print("ERROR: No runs matched.", file=sys.stderr)
            sys.exit(1)

        groups = discover_metrics(ref_run)
        all_keys = [k for keys in groups.values() for k in keys]

        # Apply metric filter if provided
        if args.metrics:
            patterns = [p.strip() for p in args.metrics.split(",")]
            all_keys = filter_metrics(all_keys, patterns)
            groups = {}
            for k in all_keys:
                g = k.split("/", 1)[0] if "/" in k else "General"
                groups.setdefault(g, []).append(k)

        print(f"  Auto-discovered {len(all_keys)} metrics.", file=sys.stderr)

        results_list = []
        for matched, label in zip(matched_runs, labels):
            if matched is None:
                results_list.append(None)
                continue
            results_list.append(pull_and_analyze(matched, all_keys, label))

        print_compare_auto(results_list, labels, groups)


# ── Time-series analysis ─────────────────────────────────────────────────────

def pull_timeseries(run, keys: list[str], samples: int = 10000) -> dict[str, list]:
    """
    Pull raw time-series data for given metric keys.
    Returns {metric_key: [(step, value), ...]}.
    """
    if not keys:
        return {}

    prefix_groups: dict[str, list[str]] = {}
    for k in keys:
        prefix = k.split("/", 1)[0] if "/" in k else "_root"
        prefix_groups.setdefault(prefix, []).append(k)

    results = {}
    for prefix, group_keys in prefix_groups.items():
        hist = run.history(keys=group_keys, samples=samples, pandas=True)
        if hist is None or len(hist) == 0:
            continue
        hist = hist.sort_values("_step").reset_index(drop=True)

        for mk in group_keys:
            if mk in hist.columns:
                series = hist[["_step", mk]].dropna(subset=[mk])
                results[mk] = list(zip(series["_step"].tolist(),
                                       series[mk].tolist()))
    return results


def compute_windowed_stats(data_points: list[tuple], n_windows: int) -> list[dict]:
    """
    Divide time-series into n_windows equal segments and compute stats per window.
    Returns list of dicts with: window, step_start, step_end, mean, std, min, max, count.
    """
    if not data_points:
        return []

    n = len(data_points)
    window_size = max(1, n // n_windows)
    windows = []

    for i in range(n_windows):
        start_idx = i * window_size
        if i == n_windows - 1:
            # Last window takes all remaining points
            end_idx = n
        else:
            end_idx = start_idx + window_size

        if start_idx >= n:
            break

        chunk = data_points[start_idx:end_idx]
        steps = [p[0] for p in chunk]
        values = [p[1] for p in chunk]
        arr = np.array(values)

        windows.append({
            "window": i + 1,
            "step_start": int(steps[0]),
            "step_end": int(steps[-1]),
            "mean": float(arr.mean()),
            "std": float(arr.std()) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": len(arr),
        })

    return windows


def print_timeseries_windowed(all_windows: dict[str, list[dict]],
                               labels: list[str],
                               run_names: list[str]):
    """Print windowed time-series tables."""
    multi_run = len(labels) > 1

    for mk in sorted(all_windows.keys()):
        runs_data = all_windows[mk]  # list of window-lists, one per run
        print(f"\n### `{mk}` — Windowed Summary\n")

        if not multi_run:
            windows = runs_data[0]
            if not windows:
                print("_No data._\n")
                continue
            print("| Window | Steps | Mean | Std | Min | Max | N |")
            print("|---:|:---|---:|---:|---:|---:|---:|")
            for w in windows:
                print(
                    f"| {w['window']} "
                    f"| {w['step_start']:,}–{w['step_end']:,} "
                    f"| {fmt(w['mean'])} "
                    f"| {fmt(w['std'])} "
                    f"| {fmt(w['min'])} "
                    f"| {fmt(w['max'])} "
                    f"| {w['count']} |"
                )
        else:
            # Side-by-side comparison: show mean per window for each run
            header = "| Window"
            for label in labels:
                header += f" | {label} (mean ± std)"
            header += " |"
            print(header)
            sep = "|---:"
            for _ in labels:
                sep += "|:---"
            sep += "|"
            print(sep)

            # Determine max windows across runs
            max_win = max(len(wd) for wd in runs_data) if runs_data else 0
            for wi in range(max_win):
                row = f"| {wi + 1}"
                for rd in runs_data:
                    if wi < len(rd):
                        w = rd[wi]
                        row += f" | {fmt(w['mean'])} ± {fmt(w['std'])} ({w['step_start']:,}–{w['step_end']:,})"
                    else:
                        row += " | —"
                row += " |"
                print(row)
        print()


def print_timeseries_raw(all_series: dict[str, list],
                          labels: list[str],
                          run_names: list[str]):
    """Print raw time-series data points."""
    multi_run = len(labels) > 1

    for mk in sorted(all_series.keys()):
        runs_data = all_series[mk]  # list of [(step, value), ...] per run
        print(f"\n### `{mk}` — Raw Time-Series\n")

        if not multi_run:
            points = runs_data[0]
            if not points:
                print("_No data._\n")
                continue
            print("| Step | Value |")
            print("|---:|---:|")
            for step, val in points:
                print(f"| {int(step):,} | {fmt(val)} |")
        else:
            header = "| Step"
            for label in labels:
                header += f" | {label}"
            header += " |"
            print(header)
            sep = "|---:"
            for _ in labels:
                sep += "|---:"
            sep += "|"
            print(sep)

            # Merge steps from all runs and align
            step_vals: dict[int, list] = {}
            for ri, points in enumerate(runs_data):
                for step, val in points:
                    s = int(step)
                    if s not in step_vals:
                        step_vals[s] = [None] * len(labels)
                    step_vals[s][ri] = val

            for step in sorted(step_vals.keys()):
                row = f"| {step:,}"
                for val in step_vals[step]:
                    row += f" | {fmt(val)}" if val is not None else " | —"
                row += " |"
                print(row)
        print()


def cmd_timeseries(args):
    """Handle the 'timeseries' subcommand — export windowed or raw time-series."""
    api_runs = fetch_wandb_runs(args.entity, args.project)

    run_names = args.runs
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = run_names if len(run_names) <= 2 else [f"Run{i+1}" for i in range(len(run_names))]

    if len(labels) != len(run_names):
        print(f"ERROR: {len(labels)} labels for {len(run_names)} runs", file=sys.stderr)
        sys.exit(1)

    # Parse metric patterns
    if not args.metrics:
        print("ERROR: --metrics is required for timeseries. Use discover to find metric names.", file=sys.stderr)
        sys.exit(1)
    patterns = [p.strip() for p in args.metrics.split(",")]

    # Match runs and pull data
    matched_runs = []
    for run_name, label in zip(run_names, labels):
        matched = match_wandb_run(api_runs, run_name)
        if matched is None:
            print(f"  ✗ {run_name}: no match", file=sys.stderr)
            matched_runs.append(None)
        else:
            print(f"  ✓ {label} → {matched.name} ({matched.id})", file=sys.stderr)
            matched_runs.append(matched)

    # Discover metrics from first available run to resolve glob patterns
    ref_run = next((r for r in matched_runs if r is not None), None)
    if ref_run is None:
        print("ERROR: No runs matched.", file=sys.stderr)
        sys.exit(1)

    groups = discover_metrics(ref_run)
    all_keys = [k for keys in groups.values() for k in keys]
    resolved_keys = filter_metrics(all_keys, patterns)

    if not resolved_keys:
        print(f"ERROR: No metrics matched patterns: {patterns}", file=sys.stderr)
        print(f"  Available: {all_keys[:20]}{'...' if len(all_keys) > 20 else ''}", file=sys.stderr)
        sys.exit(1)

    print(f"  Pulling {len(resolved_keys)} metrics from {len(run_names)} run(s)...", file=sys.stderr)

    samples = args.samples if args.samples else 10000

    if args.raw:
        # Raw mode: output all downsampled data points
        all_series: dict[str, list] = {mk: [] for mk in resolved_keys}
        for matched, label in zip(matched_runs, labels):
            if matched is None:
                for mk in resolved_keys:
                    all_series[mk].append([])
                continue
            ts = pull_timeseries(matched, resolved_keys, samples=samples)
            for mk in resolved_keys:
                all_series[mk].append(ts.get(mk, []))

        print(f"\n## Time-Series (raw, up to {samples:,} samples): {' vs '.join(labels)}")
        print_timeseries_raw(all_series, labels, run_names)
    else:
        # Windowed mode: aggregate into windows
        n_windows = args.windows
        all_windows: dict[str, list] = {mk: [] for mk in resolved_keys}
        for matched, label in zip(matched_runs, labels):
            if matched is None:
                for mk in resolved_keys:
                    all_windows[mk].append([])
                continue
            ts = pull_timeseries(matched, resolved_keys, samples=samples)
            for mk in resolved_keys:
                points = ts.get(mk, [])
                all_windows[mk].append(compute_windowed_stats(points, n_windows))

        print(f"\n## Time-Series ({n_windows} windows): {' vs '.join(labels)}")
        print_timeseries_windowed(all_windows, labels, run_names)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="General-purpose WandB metrics tool. Discover, extract, and compare training metrics from any algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- config --
    p_config = sub.add_parser(
        "config",
        help="Show run hyperparameters and config.",
    )
    p_config.add_argument("run", help="Run name")
    p_config.add_argument("--json", action="store_true",
                          help="Output raw JSON instead of table")
    p_config.add_argument("--entity", default=WANDB_ENTITY)
    p_config.add_argument("--project", default=WANDB_PROJECT)

    # -- discover --
    p_discover = sub.add_parser(
        "discover",
        help="List all metrics logged in a run, grouped by prefix.",
    )
    p_discover.add_argument("run", help="Run name (e.g., 20260302-183207_dreamer_v3_...)")
    p_discover.add_argument("--entity", default=WANDB_ENTITY)
    p_discover.add_argument("--project", default=WANDB_PROJECT)

    # -- extract --
    p_extract = sub.add_parser(
        "extract",
        help="Pull time-series stats for a single run.",
    )
    p_extract.add_argument("run", help="Run name")
    p_extract.add_argument("--metrics", type=str, default=None,
                           help="Comma-separated glob patterns (e.g., 'Episode/*,loss_*')")
    p_extract.add_argument("--entity", default=WANDB_ENTITY)
    p_extract.add_argument("--project", default=WANDB_PROJECT)

    # -- compare --
    p_compare = sub.add_parser(
        "compare",
        help="Compare metrics across multiple runs.",
    )
    p_compare.add_argument("runs", nargs="+", help="Run names")
    p_compare.add_argument("--labels", type=str, default=None,
                           help="Comma-separated labels (e.g., 'batch=64,batch=16')")
    p_compare.add_argument("--metrics", type=str, default=None,
                           help="Comma-separated glob patterns (e.g., 'Episode/*,WorldModel/*')")
    p_compare.add_argument("--preset", type=str, default=None,
                           choices=list(PRESETS.keys()),
                           help=f"Use a named preset: {', '.join(PRESETS.keys())}")
    p_compare.add_argument("--entity", default=WANDB_ENTITY)
    p_compare.add_argument("--project", default=WANDB_PROJECT)

    # -- timeseries --
    p_ts = sub.add_parser(
        "timeseries",
        help="Export windowed or raw time-series data for temporal analysis.",
    )
    p_ts.add_argument("runs", nargs="+", help="One or more run names")
    p_ts.add_argument("--metrics", type=str, required=True,
                      help="Comma-separated glob patterns (e.g., 'Episode/Steps,*loss*')")
    p_ts.add_argument("--labels", type=str, default=None,
                      help="Comma-separated labels for multi-run comparison")
    p_ts.add_argument("--windows", type=int, default=10,
                      help="Number of windows for windowed mode (default: 10)")
    p_ts.add_argument("--raw", action="store_true",
                      help="Output raw downsampled data points instead of windowed summary")
    p_ts.add_argument("--samples", type=int, default=None,
                      help="Max data points to fetch from WandB (default: 10000)")
    p_ts.add_argument("--entity", default=WANDB_ENTITY)
    p_ts.add_argument("--project", default=WANDB_PROJECT)

    args = parser.parse_args()

    if args.command == "config":
        cmd_config(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "timeseries":
        cmd_timeseries(args)


if __name__ == "__main__":
    main()
