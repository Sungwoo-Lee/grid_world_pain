#!/usr/bin/env python
"""Avoidance behavior-probe statistics + journal-style heatmap (reusable).

Reads eval recordings written by `scripts/eval_rollout.py --record`, computes the
canonical avoidance measures per episode, aggregates mean +/- std over all
seeds/episodes per config, writes a stats CSV, and renders a polished heatmap
(rows = experiment config, columns = criterion) via _heatmap_style.render.

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
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))  # repo root  -> src.*
sys.path.insert(0, str(_here.parent))      # this dir   -> _heatmap_style
from _heatmap_style import render
from src.utils.eval_recording import load_episode

# (key, display label, group) -- order = column order
MEASURES = [
    ("bush_use_rate",              "bush-use\nrate", "Cover use"),
    ("bush_entry_step",            "bush-entry\nstep", "Cover use"),
    ("bush_dwell_frac",            "bush-dwell\nfraction", "Cover use"),
    ("flight_initiation_distance", "flight-initiation\ndistance (FID)", "Proximity to animal"),
    ("animal_proximity_frac",      "animal-proximity\nfraction", "Proximity to animal"),
    ("closest_approach",           "closest\napproach", "Proximity to animal"),
    ("injury_change",              "injury change\n(end − start)", "Outcome"),
    ("survival_steps",             "survival\nsteps", "Outcome"),
]
KEYS = [k for k, _, _ in MEASURES]
SIGNED = {"injury_change"}

ANIMAL_LABEL = {
    "none": "no animal", "pred": "predator", "rabbit": "rabbit · chase",
    "rabbit_olfzero": "rabbit · chase · olf-zeroed", "rabbitwander": "rabbit · wander",
    "rabbitwander_predsmell": "rabbit · wander · pred-smell",
}


def manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def episode_measures(ep):
    S = ep["snapshots"]; T = len(S)
    ag = [np.asarray(s["agent_pos"]) for s in S]
    bush = np.asarray(S[0]["obs_pos"][0]) if len(S[0]["obs_pos"]) else None
    start = ag[0]
    in_bush = [bush is not None and np.array_equal(ag[t], bush) for t in range(T)]
    entry = next((t for t in range(T) if in_bush[t]), None)
    inj = float(S[-1]["injury_level"]) - float(S[0]["injury_level"])
    has = len(S[0]["animal_pos"]) > 0
    fid = closest = prox = np.nan
    if has:
        ad = [manhattan(ag[t], np.asarray(S[t]["animal_pos"][0])) for t in range(T)]
        closest = float(min(ad)); prox = float(np.mean([d <= 1 for d in ad]))
        dep = next((t for t in range(T) if not np.array_equal(ag[t], start)), None)
        fid = float(ad[dep]) if dep is not None else np.nan
    return {"bush_use_rate": 1.0 if entry is not None else 0.0,
            "bush_entry_step": float(entry) if entry is not None else np.nan,
            "bush_dwell_frac": float(np.mean(in_bush)),
            "flight_initiation_distance": fid, "animal_proximity_frac": prox,
            "closest_approach": closest, "injury_change": inj, "survival_steps": float(T)}


def split_cfg(cfg):
    name = cfg[len("avoid_"):] if cfg.startswith("avoid_") else cfg
    animal, inj = name.rsplit("_inj", 1) if "_inj" in name else (name, "")
    return animal, (str(int(inj)) if inj.isdigit() else inj)


def row_label(cfg):
    a, inj = split_cfg(cfg)
    return f"{ANIMAL_LABEL.get(a, a.replace('_', ' '))}   ·   inj {inj}" if inj else ANIMAL_LABEL.get(a, a)


def discover_configs(root):
    return sorted(d.name for d in Path(root).iterdir()
                  if d.is_dir() and glob.glob(str(d / "models" / "*" / "recordings")))


def recordings_for(root, cfg, ckpt):
    base = Path(root) / cfg / "models"
    ckpts = [ckpt] if ckpt else [p.name for p in base.iterdir() if p.is_dir()]
    out = []
    for ck in ckpts:
        out += sorted(glob.glob(str(base / ck / "recordings" / ck / "episode_*.rec.gz")))
    return out


def aggregate(root, configs, ckpt):
    stats = {}
    for cfg in configs:
        acc = {k: [] for k in KEYS}
        for r in recordings_for(root, cfg, ckpt):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--title", default="Avoidance behaviour across animal × injury conditions")
    args = ap.parse_args()

    configs = args.configs or discover_configs(args.results_root)
    if not configs:
        raise SystemExit(f"No config subdirs with recordings under {args.results_root}")
    out_dir = Path(args.out_dir or (Path(args.results_root) / "STATS"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = aggregate(args.results_root, configs, args.ckpt)
    write_csv(stats, configs, out_dir / "avoidance_stats.csv")

    mean = np.array([[stats[c][k][0] for k in KEYS] for c in configs])
    std = np.array([[stats[c][k][1] for k in KEYS] for c in configs])
    signed_cols = [i for i, k in enumerate(KEYS) if k in SIGNED]
    seen, groups = [], []
    for i, (_, _, g) in enumerate(MEASURES):
        if g not in [n for n, _ in groups]:
            groups.append((g, [j for j, (_, _, gg) in enumerate(MEASURES) if gg == g]))
    bounds = [i for i in range(1, len(configs)) if split_cfg(configs[i])[0] != split_cfg(configs[i - 1])[0]]
    caption = ("n = 30 episodes/condition (seeds 0–29, small initial-state jitter; deterministic policy).   "
               "Cell = mean ± std.   Colour is per-criterion: sequential (min–max) for magnitude metrics, "
               "diverging at 0 for injury change (red = net harm, blue = net healed).   "
               "Distances in grid cells; FID = flight-initiation distance.   ‘–’ = not applicable (no animal).")
    render(mean, std, [row_label(c) for c in configs], [lab for _, lab, _ in MEASURES],
           signed_cols=signed_cols, col_groups=groups, row_group_bounds=bounds,
           title=args.title, caption=caption,
           out_png=str(out_dir / "avoidance_stats_heatmap.png"),
           out_pdf=str(out_dir / "avoidance_stats_heatmap.pdf"))
    print(f"[avoidance_stats_heatmap] {len(configs)} configs -> {out_dir}/avoidance_stats.{{csv,png,pdf}}")


if __name__ == "__main__":
    main()
