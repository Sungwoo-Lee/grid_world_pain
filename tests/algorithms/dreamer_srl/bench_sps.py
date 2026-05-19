#!/usr/bin/env /home/vncuser/miniconda3/envs/grid_world_pain/bin/python
"""SPS micro-benchmark for dreamer-srl.

Runs the dreamer-srl trainer end-to-end with a fixed smoke config and CLI
overrides, capturing per-log-step (policy_step, wall_time) pairs from stdout.
Computes:
  - Cumulative SPS = final_policy_step / total_wall_time
  - Instantaneous SPS = mean of Δpolicy_step/Δwall_time over the last 25% of
    training (5-sample sliding window)

Writes a CSV trace to tmp/sps_bench_<label>_<YYYYMMDD_HHMMSS>.csv with columns:
  policy_step, wall_time, instantaneous_sps

Config used:
  Agent config: configs/dreamer_srl/01_food_only_smoke.yaml  (if present; else
                configs/dreamer_srl/01_food_only.yaml — see NOTE below)
  Env config:   configs/experiment/dreamer_curriculum/01_food_only.yaml

CLI overrides applied:
  --num-envs N        (default 16)
  --total-steps N     (default 50000)
  --seed N            (default 0)
  --no-wandb          (always; we want clean SPS without WandB I/O)

NOTE: 01_food_only_smoke.yaml is confirmed to exist at plan time (verified by
the developer agent). If the file is removed in the future, the script falls
back to 01_food_only.yaml automatically and logs a warning.

Usage:
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
      tests/algorithms/dreamer_srl/bench_sps.py \\
      --label step0_baseline --num-envs 16 --total-steps 50000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYTHON = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

SMOKE_AGENT_CFG = os.path.join(PROJECT_ROOT, "configs", "dreamer_srl", "01_food_only_smoke.yaml")
FALLBACK_AGENT_CFG = os.path.join(PROJECT_ROOT, "configs", "dreamer_srl", "01_food_only.yaml")
ENV_CFG = os.path.join(PROJECT_ROOT, "configs", "experiment", "dreamer_curriculum", "01_food_only.yaml")
TRAINER_SCRIPT = os.path.join(PROJECT_ROOT, "src", "algorithms", "dreamer_srl", "dreamer_srl_main.py")

TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")

# Regex that matches the per-step log line from dreamer_srl_main.py:
#   [iter 12/3125] policy_step=192 world_model_loss=... sps=31.4
_LOG_RE = re.compile(
    r"\[iter\s+\d+/\d+\]\s+"
    r"policy_step=(\d+)\s+"
    r".*?"
    r"sps=([\d.]+)"
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dreamer-srl SPS micro-benchmark")
    p.add_argument("--label", type=str, default="bench",
                   help="Label for the output CSV filename")
    p.add_argument("--num-envs", type=int, default=16,
                   help="Number of parallel environments (default: 16)")
    p.add_argument("--total-steps", type=int, default=50000,
                   help="Total environment steps (default: 50000)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--buffer-device", type=str, default="cpu", choices=["cpu", "gpu"],
                   help="Buffer storage device: 'cpu' (default) or 'gpu' (Step 2). "
                        "Passed to SequentialReplayBuffer via --buffer-device CLI flag.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_path(label: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(TMP_DIR, f"sps_bench_{label}_{ts}.csv")


def _pick_agent_cfg() -> str:
    if os.path.exists(SMOKE_AGENT_CFG):
        return SMOKE_AGENT_CFG
    print(
        f"[bench_sps] WARNING: smoke config not found at {SMOKE_AGENT_CFG!r}. "
        f"Falling back to {FALLBACK_AGENT_CFG!r}. "
        "The benchmark is using a non-smoke config; SPS numbers may differ "
        "from runs on other steps that use the smoke config."
    )
    return FALLBACK_AGENT_CFG


def _parse_trace(lines: list[str]) -> list[tuple[int, float]]:
    """Extract (policy_step, sps_cumulative) pairs from stdout lines."""
    trace = []
    for line in lines:
        m = _LOG_RE.search(line)
        if m:
            policy_step = int(m.group(1))
            sps_cum = float(m.group(2))
            trace.append((policy_step, sps_cum))
    return trace


def _compute_metrics(
    trace: list[tuple[int, float]],
    t_start: float,
    t_end: float,
) -> tuple[float, float, list[tuple[int, float, float]]]:
    """
    Returns (cumulative_sps, instantaneous_sps_last25pct, csv_rows).

    csv_rows: list of (policy_step, wall_time_since_start, instantaneous_sps)

    The trainer logs `sps = policy_step / (time.time() - t_start)`, i.e. the
    cumulative rate.  We recover wall_time from that:
        wall_time(i) = policy_step(i) / sps_cum(i)

    Then instantaneous SPS over window [i-1, i]:
        inst_sps(i) = (step(i) - step(i-1)) / (wt(i) - wt(i-1))
    """
    if not trace:
        return 0.0, 0.0, []

    # Recover (policy_step, wall_time) from cumulative sps
    pts: list[tuple[int, float]] = []
    for ps, sps_cum in trace:
        if sps_cum > 0:
            wt = ps / sps_cum
        else:
            wt = 0.0
        pts.append((ps, wt))

    # Cumulative SPS using actual measured start/end times
    final_ps = pts[-1][0]
    total_elapsed = t_end - t_start
    cum_sps = final_ps / max(total_elapsed, 1e-9)

    # Instantaneous SPS per log step (Δsteps / Δwall_time)
    inst_spss: list[float] = []
    csv_rows: list[tuple[int, float, float]] = []

    for i in range(1, len(pts)):
        ps_prev, wt_prev = pts[i - 1]
        ps_curr, wt_curr = pts[i]
        dps = ps_curr - ps_prev
        dwt = wt_curr - wt_prev
        inst = dps / max(dwt, 1e-9)
        inst_spss.append(inst)
        csv_rows.append((ps_curr, wt_curr, inst))

    # Instantaneous SPS: mean over last 25% of training
    # Use a 5-sample sliding window over the last-25% window of samples
    n = len(inst_spss)
    last_25_start = max(0, int(n * 0.75))
    last_25_sps = inst_spss[last_25_start:]

    if not last_25_sps:
        inst_sps_last25 = inst_spss[-1] if inst_spss else 0.0
    else:
        # 5-sample sliding window mean
        window = 5
        if len(last_25_sps) <= window:
            inst_sps_last25 = sum(last_25_sps) / len(last_25_sps)
        else:
            # Compute sliding window means and take the overall mean
            window_means = []
            for j in range(len(last_25_sps) - window + 1):
                window_means.append(sum(last_25_sps[j:j + window]) / window)
            inst_sps_last25 = sum(window_means) / len(window_means)

    return cum_sps, inst_sps_last25, csv_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    agent_cfg = _pick_agent_cfg()
    csv_out = _csv_path(args.label)

    cmd = [
        PYTHON,
        TRAINER_SCRIPT,
        "--env-config",    ENV_CFG,
        "--agent-config",  agent_cfg,
        "--num-envs",      str(args.num_envs),
        "--total-steps",   str(args.total_steps),
        "--seed",          str(args.seed),
        "--no-wandb",
        "--buffer-device", args.buffer_device,  # Step 2: "cpu" (default) or "gpu"
    ]

    print(f"[bench_sps] Label:         {args.label}")
    print(f"[bench_sps] Agent cfg:     {agent_cfg}")
    print(f"[bench_sps] Env cfg:       {ENV_CFG}")
    print(f"[bench_sps] num_envs:      {args.num_envs}")
    print(f"[bench_sps] total_steps:   {args.total_steps}")
    print(f"[bench_sps] seed:          {args.seed}")
    print(f"[bench_sps] buffer_device: {args.buffer_device}")
    print(f"[bench_sps] CSV out:       {csv_out}")
    print(f"[bench_sps] Command:       {' '.join(cmd)}")
    print()

    t_start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,          # line-buffered
        cwd=PROJECT_ROOT,
    )

    collected_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        collected_lines.append(line)

    proc.wait()
    t_end = time.time()

    if proc.returncode != 0:
        print(f"\n[bench_sps] ERROR: trainer exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    # Parse trace
    trace = _parse_trace(collected_lines)
    if not trace:
        print("\n[bench_sps] ERROR: no SPS log lines found in output. "
              "Did the training complete? Check the stdout above.")
        sys.exit(1)

    cum_sps, inst_sps_last25, csv_rows = _compute_metrics(trace, t_start, t_end)

    # Write CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy_step", "wall_time", "instantaneous_sps"])
        writer.writerows(csv_rows)

    # Print summary
    print()
    print("=" * 60)
    print(f"[bench_sps] === RESULTS: {args.label} ===")
    print(f"[bench_sps] Total wall time:                   {t_end - t_start:.1f}s")
    print(f"[bench_sps] Log steps captured:                {len(trace)}")
    print(f"[bench_sps] Cumulative SPS (whole run):        {cum_sps:.2f} env-steps/s")
    print(f"[bench_sps] Instantaneous SPS (last 25%):      {inst_sps_last25:.2f} env-steps/s")
    print(f"[bench_sps] CSV trace written to:              {csv_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
