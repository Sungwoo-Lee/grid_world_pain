#!/usr/bin/env python3
"""
WandB Shared Utilities
======================
Common functions shared across WandB analysis scripts:
- Run matching (exact name + timestamp fuzzy match)
- Metric statistics computation
- Formatting helpers
"""

import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import wandb


# ── Defaults ──────────────────────────────────────────────────────────────────
WANDB_ENTITY = "sungwoolee"
WANDB_PROJECT = "grid_world_pain"
TIMESTAMP_TOLERANCE_SEC = 120

# Timezone offset (hours from UTC) for interpreting run-name timestamps.
# Default: +9 (KST). Override via TZ_OFFSET_HOURS env or pass explicitly.
import os
TZ_OFFSET_HOURS = int(os.environ.get("WANDB_TZ_OFFSET", "9"))


# ── Timestamp parsing ────────────────────────────────────────────────────────

def parse_timestamp_from_name(run_name: str) -> datetime | None:
    """Extract YYYYMMDD-HHMMSS from the beginning of a run name."""
    parts = run_name.split("_", 1)
    ts_str = parts[0]
    try:
        return datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


# ── Run matching ──────────────────────────────────────────────────────────────

def match_wandb_run(api_runs, run_name: str, tz_offset_hours: int | None = None):
    """
    Find the WandB run matching `run_name`.

    Strategy (in order):
      1. Exact match on run.name
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

    tz_hours = tz_offset_hours if tz_offset_hours is not None else TZ_OFFSET_HOURS
    best_run = None
    best_diff = timedelta(days=999)
    for r in api_runs:
        try:
            wandb_ts = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
            local_utc = local_ts.replace(
                tzinfo=timezone(timedelta(hours=tz_hours))
            ).astimezone(timezone.utc)
            diff = abs(wandb_ts - local_utc)
            if diff < best_diff and diff < timedelta(seconds=TIMESTAMP_TOLERANCE_SEC):
                best_diff = diff
                best_run = r
        except Exception:
            continue

    return best_run


def fetch_wandb_runs(entity: str = WANDB_ENTITY, project: str = WANDB_PROJECT):
    """Fetch all runs from a WandB project. Returns (api_runs, path)."""
    api = wandb.Api()
    path = f"{entity}/{project}"
    print(f"Fetching runs from {path}...", file=sys.stderr)
    try:
        api_runs = list(api.runs(path))
    except Exception:
        path = project
        api_runs = list(api.runs(path))
    print(f"Found {len(api_runs)} runs.", file=sys.stderr)
    return api_runs


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(series) -> dict | None:
    """Compute summary statistics for a pandas Series."""
    col = series.dropna()
    if len(col) == 0:
        return None
    n = len(col)
    last_20_start = int(n * 0.8)
    last_20 = col.iloc[last_20_start:]
    return {
        "full_mean": float(col.mean()),
        "full_std": float(col.std()) if n > 1 else 0.0,
        "last20_mean": float(last_20.mean()) if len(last_20) > 0 else float("nan"),
        "last20_std": float(last_20.std()) if len(last_20) > 1 else 0.0,
        "first": float(col.iloc[0]),
        "last": float(col.iloc[-1]),
        "min": float(col.min()),
        "max": float(col.max()),
        "count": n,
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt(val, precision=4) -> str:
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


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"
