#!/usr/bin/env python
"""Avoidance behavior-probe statistics + heatmap (reusable).

Reads the eval recordings written by `scripts/eval_rollout.py --record`, computes
the canonical avoidance measures per episode, aggregates mean +/- std over all
seeds/episodes per config, writes a stats CSV, and renders an annotated heatmap
(rows = experiment config, columns = criterion; color = per-column min-max
normalized mean; cell text = mean +/- std).

Canonical measures (see the study's "Measures & definitions" block in
docs/experiments/active/behavior_measures/interoceptive_behavior_measure_study.md):
  bush_use_rate              fraction of episodes that ever enter the bush (whether)
  bush_entry_step            step of first entry onto the bush cell (when / latency)
  bush_dwell_frac            fraction of steps spent on the bush cell (how long)
  flight_initiation_distance animal-agent distance at the agent's first move (FID)
  animal_proximity_frac      fraction of steps the animal is within 1 cell of the agent
  closest_approach           smallest animal-agent distance over the episode
  injury_change              injury(last) - injury(first)  (+ harmed / - healed)
  survival_steps             episode length

Layout expected under --results-root:
  <results-root>/<config>/models/<ckpt>/recordings/<ckpt>/episode_*.rec.gz

Usage (run from repo root with the project conda interpreter):
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    scripts/behavior_measures/avoidance_stats_heatmap.py \
    --results-root results/eval/avoidance_stat \
    --out-dir results/eval/avoidance_stat/STATS
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path
from src.utils.eval_recording import load_episode

# (key, display label) -- order = column order in the heatmap
MEASURES = [
    ("bush_use_rate",              "bush-use\nrate (0-1)"),
    ("bush_entry_step",            "bush-entry\nstep"),
    ("bush_dwell_frac",            "bush-dwell\nfraction"),
    ("flight_initiation_distance", "flight-initiation\ndistance (FID)"),
    ("animal_proximity_frac",      "animal-proximity\nfraction"),
    ("closest_approach",           "closest approach\nto animal"),
    ("injury_change",              "injury change\n(end - start)"),
    ("survival_steps",             "survival\nsteps"),
]
KEYS = [k for k, _ in MEASURES]


def manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def episode_measures(ep):
    """Compute the 8 avoidance measures for one episode recording."""
    S = ep["snapshots"]
    T = len(S)
    ag = [np.asarray(s["agent_pos"]) for s in S]
    bush = np.asarray(S[0]["obs_pos"][0]) if len(S[0]["obs_pos"]) else None
    start = ag[0]
    in_bush = [bush is not None and np.array_equal(ag[t], bush) for t in range(T)]
    entry = next((t for t in range(T) if in_bush[t]), None)
    inj_change = float(S[-1]["injury_level"]) - float(S[0]["injury_level"])
    has_animal = len(S[0]["animal_pos"]) > 0
    fid = closest = prox = np.nan
    if has_animal:
        adist = [manhattan(ag[t], np.asarray(S[t]["animal_pos"][0])) for t in range(T)]
        closest = float(min(adist))
        prox = float(np.mean([d <= 1 for d in adist]))
        dep = next((t for t in range(T) if not np.array_equal(ag[t], start)), None)
        fid = float(adist[dep]) if dep is not None else np.nan
    return {
        "bush_use_rate": 1.0 if entry is not None else 0.0,
        "bush_entry_step": float(entry) if entry is not None else np.nan,
        "bush_dwell_frac": float(np.mean(in_bush)),
        "flight_initiation_distance": fid,
        "animal_proximity_frac": prox,
        "closest_approach": closest,
        "injury_change": inj_change,
        "survival_steps": float(T),
    }


def prettify(cfg):
    """avoid_pred_inj00 -> 'pred * inj0'; keep generic for any avoid_<...>_inj<NN>."""
    name = cfg[len("avoid_"):] if cfg.startswith("avoid_") else cfg
    if "_inj" in name:
        animal, inj = name.rsplit("_inj", 1)
        inj = str(int(inj)) if inj.isdigit() else inj
        return f"{animal.replace('_', ' ')} . inj{inj}"
    return name.replace("_", " ")


def discover_configs(results_root):
    return sorted(d.name for d in Path(results_root).iterdir()
                  if d.is_dir() and glob.glob(str(d / "models" / "*" / "recordings")))


def recordings_for(results_root, cfg, ckpt):
    base = Path(results_root) / cfg / "models"
    ckpts = [ckpt] if ckpt else [p.name for p in base.iterdir() if p.is_dir()]
    out = []
    for ck in ckpts:
        out += sorted(glob.glob(str(base / ck / "recordings" / ck / "episode_*.rec.gz")))
    return out


def aggregate(results_root, configs, ckpt):
    stats = {}
    for cfg in configs:
        acc = {k: [] for k in KEYS}
        for r in recordings_for(results_root, cfg, ckpt):
            m = episode_measures(load_episode(Path(r)))
            for k in KEYS:
                v = m[k]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    acc[k].append(v)
        stats[cfg] = {k: (np.mean(acc[k]) if acc[k] else np.nan,
                          np.std(acc[k]) if acc[k] else np.nan) for k in KEYS}
    return stats


def write_csv(stats, configs, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config"] + [f"{k}_mean" for k in KEYS] + [f"{k}_std" for k in KEYS])
        for c in configs:
            w.writerow([c]
                       + [f"{stats[c][k][0]:.3f}" if not np.isnan(stats[c][k][0]) else "" for k in KEYS]
                       + [f"{stats[c][k][1]:.3f}" if not np.isnan(stats[c][k][1]) else "" for k in KEYS])


def render_heatmap(stats, configs, path, title):
    nrow, ncol = len(configs), len(KEYS)
    mean = np.array([[stats[c][k][0] for k in KEYS] for c in configs])
    std = np.array([[stats[c][k][1] for k in KEYS] for c in configs])
    norm = np.full_like(mean, np.nan)
    for j in range(ncol):
        col = mean[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm[:, j] = 0.5 if hi == lo else (col - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(1.65 * ncol + 1.5, 0.7 * nrow + 1.5))
    cmap = plt.cm.YlGnBu
    cmap.set_bad("#dddddd")
    ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(ncol)); ax.set_xticklabels([l for _, l in MEASURES], fontsize=9)
    ax.set_yticks(range(nrow)); ax.set_yticklabels([prettify(c) for c in configs], fontsize=9)
    ax.set_title(title, fontsize=10.5)
    for i in range(nrow):
        for j in range(ncol):
            mu, sd = mean[i, j], std[i, j]
            txt = "-" if np.isnan(mu) else (f"{mu:.2f}\n±{sd:.2f}" if abs(mu) < 10 else f"{mu:.0f}\n±{sd:.0f}")
            color = "white" if (not np.isnan(norm[i, j]) and norm[i, j] > 0.6) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True, help="Dir of <config>/models/<ckpt>/recordings/...")
    ap.add_argument("--ckpt", default=None, help="Checkpoint step subdir (auto-detect if omitted).")
    ap.add_argument("--configs", nargs="+", default=None, help="Config subdir names (auto-discover if omitted).")
    ap.add_argument("--out-dir", default=None, help="Output dir for CSV+PNG (default: <results-root>/STATS).")
    ap.add_argument("--title", default="Avoidance experiments - criteria summary (mean ± std)\n"
                    "color = per-column min-max normalized mean; gray = N/A   ·   FID = flight-initiation distance; distances in grid cells")
    args = ap.parse_args()

    configs = args.configs or discover_configs(args.results_root)
    if not configs:
        raise SystemExit(f"No config subdirs with recordings under {args.results_root}")
    out_dir = Path(args.out_dir or (Path(args.results_root) / "STATS"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = aggregate(args.results_root, configs, args.ckpt)
    csv_path = out_dir / "avoidance_stats.csv"
    png_path = out_dir / "avoidance_stats_heatmap.png"
    write_csv(stats, configs, csv_path)
    render_heatmap(stats, configs, png_path, args.title)
    print(f"[avoidance_stats_heatmap] {len(configs)} configs -> {csv_path}  +  {png_path}")


if __name__ == "__main__":
    main()
