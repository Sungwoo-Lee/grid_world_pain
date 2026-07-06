#!/usr/bin/env python3
"""parity_check_eval_rollout.py — Tier 2 bit-for-bit parity gate.

What this checks, in plain language
------------------------------------
`scripts/eval/eval_rollout.py` now has two ways to play the same evaluation
episodes: the old "one episode at a time" Python loop (the *legacy* path,
always correct, never touched by this change) and a new "play all episodes at
once on the accelerator" path (the *batched* path, enabled with `--batched`,
built for speed). Both are supposed to produce **exactly** the same behavior
per episode — same positions, same injury, same survival length — because an
episode's outcome is fully determined by its random seed and the (deterministic)
policy. This script is the referee: it runs the SAME checkpoint, config, and
seeds through both paths, computes the 11 standard avoidance-behavior measures
(bush use, distance to the animal, movement, injury, survival — see
`scripts/behavior_measures/avoidance_stats_heatmap.py::episode_measures`, which
this script imports rather than re-implementing) for every episode, and reports
a PASS/FAIL table. It exits non-zero if ANY episode's measures differ at all —
this is a hard gate: batching may not become the default sweep path until this
script is green (see docs/develop/active/refactors/EVAL_ROLLOUT_BATCHING_PERF.md
for the full plan and the correctness argument).

Two run modes
-------------
1. Default: re-run the legacy rollout fresh (a subprocess), then the batched
   rollout (a second subprocess), then diff.
2. `--golden-dir <root>`: skip re-running the legacy rollout and compare the
   batched rollout against a FROZEN set of legacy recordings captured earlier
   (e.g. from `git` HEAD before any batched code existed). This guards against
   an accidental edit to the legacy functions quietly moving the reference
   goalposts.

Usage
-----
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        scripts/eval/parity_check_eval_rollout.py \\
        --config configs/environment/experiment/behavior_probes/core/avoidance/avoid_pred_inj00.yaml \\
        --checkpoint results/JAX_RecurrentPPO/<run>/models/8500010 \\
        --n-episodes 30 --device cpu

    # Golden-baseline mode (compare against a frozen legacy capture):
    ... --golden-dir tmp/golden_b03_pred70_30 --n-episodes 30
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "behavior_measures"))

# Shared measure function — imported, NEVER re-derived, per the plan's Tier 2
# acceptance contract (the harness must not route both sides through new code,
# and must not silently drift from the canonical measure definitions).
from avoidance_stats_heatmap import episode_measures, KEYS  # noqa: E402
from src.utils.eval_recording import load_episode  # noqa: E402

EVAL_ROLLOUT = PROJECT_ROOT / "scripts" / "eval" / "eval_rollout.py"


def _run_eval_rollout(python, config, agent_config, checkpoint, output_root,
                       n_episodes, seeds, batched, device):
    cmd = [
        python, str(EVAL_ROLLOUT),
        "--config", config,
        "--checkpoint", checkpoint,
        "--output-root", str(output_root),
        "--eval-n-episodes", str(n_episodes),
        "--record", "--record-n-episodes", str(n_episodes),
        "--device", device,
        "--quiet",
    ]
    if agent_config:
        cmd += ["--agent_config", agent_config]
    if seeds:
        cmd += ["--eval-seeds"] + [str(s) for s in seeds]
    if batched:
        cmd += ["--batched"]
    return subprocess.run(cmd, capture_output=True, text=True)


def _output_dir(output_root, checkpoint):
    ckpt_path = Path(checkpoint).resolve()
    run_tag = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else ckpt_path.parent.parent.name
    return Path(output_root) / run_tag / ckpt_path.name


def _pct_label(checkpoint):
    name = Path(checkpoint).resolve().name
    try:
        return str(int(name))
    except ValueError:
        return "eval"


def _compare_episode(legacy_dir, batched_dir, pct, idx, tolerance):
    """Returns (status, list of (measure_name, legacy_val, batched_val)) for one episode."""
    leg_rec = legacy_dir / "recordings" / pct / f"episode_{idx:06d}.rec.gz"
    bat_rec = batched_dir / "recordings" / pct / f"episode_{idx:06d}.rec.gz"
    if not leg_rec.exists() or not bat_rec.exists():
        return "FAIL", [("recording_missing", leg_rec.exists(), bat_rec.exists())]

    m_leg = episode_measures(load_episode(leg_rec))
    m_bat = episode_measures(load_episode(bat_rec))

    mismatches = []
    for k in KEYS:
        vl, vb = m_leg[k], m_bat[k]
        both_nan = (isinstance(vl, float) and isinstance(vb, float)
                    and np.isnan(vl) and np.isnan(vb))
        if both_nan:
            continue
        ok = (vl == vb) if tolerance == 0.0 else (abs(vl - vb) <= tolerance)
        if not ok:
            mismatches.append((k, vl, vb))

    # Stronger check: diff the raw per-episode .npz arrays too (not just the 11
    # derived measures), per the plan's "Also diff the .npz per-episode arrays".
    npz_leg = np.load(legacy_dir / "episodes" / f"{idx:04d}.npz")
    npz_bat = np.load(batched_dir / "episodes" / f"{idx:04d}.npz")
    for k in npz_leg.files:
        if k == "seed":
            continue
        if not np.array_equal(npz_leg[k], npz_bat[k]):
            mismatches.append((f"npz:{k}", "differs", "differs"))

    return ("FAIL" if mismatches else "PASS"), mismatches


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent_config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Explicit seed override. Default: config's canonical "
                         "behavior_measures.eval_seeds, sliced to --n-episodes.")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="Max allowed abs diff per measure. Default 0.0 = exact "
                         "bit-for-bit equality (the required, expected outcome — "
                         "any non-zero value must be independently justified).")
    ap.add_argument("--golden-dir", default=None,
                    help="Frozen legacy output root (skip re-running legacy; "
                         "compares batched against this instead).")
    ap.add_argument("--work-dir", default=None,
                    help="Scratch root for fresh legacy/batched runs. "
                         "Default: tmp/parity_check_<timestamp>/.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter to launch eval_rollout.py subprocesses with.")
    args = ap.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else PROJECT_ROOT / "tmp" / f"parity_check_{int(time.time())}"
    batched_root = work_dir / "batched"

    if args.golden_dir is None:
        legacy_root = work_dir / "legacy"
        print(f"[parity] Running LEGACY rollout -> {legacy_root}", flush=True)
        t0 = time.time()
        r = _run_eval_rollout(args.python, args.config, args.agent_config, args.checkpoint,
                              legacy_root, args.n_episodes, args.seeds, batched=False,
                              device=args.device)
        legacy_wall = time.time() - t0
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr, file=sys.stderr)
            sys.exit(f"[parity] legacy rollout FAILED (exit {r.returncode})")
        print(f"[parity] legacy rollout done in {legacy_wall:.1f}s", flush=True)
    else:
        legacy_root = Path(args.golden_dir)
        legacy_wall = None
        print(f"[parity] Using GOLDEN baseline at {legacy_root} (not re-running legacy)", flush=True)

    print(f"[parity] Running BATCHED rollout -> {batched_root}", flush=True)
    t0 = time.time()
    r = _run_eval_rollout(args.python, args.config, args.agent_config, args.checkpoint,
                          batched_root, args.n_episodes, args.seeds, batched=True,
                          device=args.device)
    batched_wall = time.time() - t0
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(f"[parity] batched rollout FAILED (exit {r.returncode})")
    print(f"[parity] batched rollout done in {batched_wall:.1f}s", flush=True)

    legacy_dir = _output_dir(legacy_root, args.checkpoint)
    batched_dir = _output_dir(batched_root, args.checkpoint)
    pct = _pct_label(args.checkpoint)

    with open(batched_dir / "metadata.json") as f:
        batched_meta = json.load(f)
    seeds_used = batched_meta["seeds"]

    print(f"\n{'idx':>4}  {'seed':>12}  {'status':6}  mismatches")
    n_fail = 0
    for i, seed in enumerate(seeds_used):
        status, mismatches = _compare_episode(legacy_dir, batched_dir, pct, i, args.tolerance)
        if status == "FAIL":
            n_fail += 1
        detail = "" if not mismatches else "; ".join(
            f"{k}: legacy={vl} batched={vb}" for k, vl, vb in mismatches[:6]
        )
        print(f"{i:>4}  {seed:>12}  {status:6}  {detail}")

    n_total = len(seeds_used)
    print(f"\n[parity] {n_total - n_fail}/{n_total} episodes exact-match "
          f"(tolerance={args.tolerance}).")
    if legacy_wall is not None:
        speedup = legacy_wall / batched_wall if batched_wall > 0 else float("inf")
        print(f"[parity] wall-clock: legacy={legacy_wall:.1f}s batched={batched_wall:.1f}s "
              f"(speedup {speedup:.2f}x)")
    print(f"[parity] legacy dir:  {legacy_dir}")
    print(f"[parity] batched dir: {batched_dir}")

    if n_fail:
        print(f"\n[parity] FAIL — {n_fail} episode(s) mismatched. Batching is NOT "
              f"parity-safe for this checkpoint/config/seed set.", file=sys.stderr)
        sys.exit(1)

    print("\n[parity] PASS — bit-for-bit parity confirmed.")


if __name__ == "__main__":
    main()
