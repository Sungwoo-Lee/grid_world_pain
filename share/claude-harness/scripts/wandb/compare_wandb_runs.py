#!/usr/bin/env python3
"""
WandB Training Metrics Comparison (DreamerV3 Preset)
=====================================================
Backward-compatible wrapper around wandb_metrics.py.
Runs comparison using the DreamerV3 preset by default.

For general-purpose usage across any algorithm, use wandb_metrics.py directly:
    python scripts/wandb_metrics.py discover RUN_NAME
    python scripts/wandb_metrics.py compare RUN1 RUN2 --labels "A,B"

This script is equivalent to:
    python scripts/wandb_metrics.py compare --preset dreamer_v3 RUN1 RUN2

Usage:
    python scripts/compare_wandb_runs.py RUN_NAME1 RUN_NAME2 [...]
    python scripts/compare_wandb_runs.py --labels "A,B" RUN_NAME1 RUN_NAME2
"""

import argparse
import sys

from wandb_utils import WANDB_ENTITY, WANDB_PROJECT, fetch_wandb_runs, match_wandb_run
from wandb_metrics import PRESETS, pull_and_analyze, print_compare_preset


def main():
    parser = argparse.ArgumentParser(
        description="Compare training health metrics across WandB runs (DreamerV3 preset).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "runs", nargs="+",
        help="Run names (e.g. 20260302-183207_dreamer_v3_...)",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Comma-separated labels (e.g. 'batch=64,batch=16')",
    )
    parser.add_argument(
        "--preset", type=str, default="dreamer_v3",
        choices=list(PRESETS.keys()),
        help="Metric preset (default: dreamer_v3)",
    )
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    args = parser.parse_args()

    # Labels
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = [f"Run{i+1}" for i in range(len(args.runs))]
    if len(labels) != len(args.runs):
        print(f"ERROR: {len(labels)} labels for {len(args.runs)} runs", file=sys.stderr)
        sys.exit(1)

    # Fetch & match
    api_runs = fetch_wandb_runs(args.entity, args.project)

    preset = PRESETS[args.preset]
    preset_keys = []
    for metric_list in preset.values():
        for mk, _, _ in metric_list:
            if mk not in preset_keys:
                preset_keys.append(mk)

    results_list = []
    for run_name, label in zip(args.runs, labels):
        matched = match_wandb_run(api_runs, run_name)
        if matched is None:
            print(f"  ✗ {run_name}: no match", file=sys.stderr)
            results_list.append(None)
            continue
        print(f"  ✓ {label} → {matched.name} ({matched.id})", file=sys.stderr)
        results_list.append(pull_and_analyze(matched, preset_keys, label))

    print_compare_preset(results_list, labels, preset)


if __name__ == "__main__":
    main()
