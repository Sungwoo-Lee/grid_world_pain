#!/usr/bin/env python
"""Avoidance behavior-probe statistics + journal-style heatmap (reusable).

Reads eval recordings written by `scripts/eval/eval_rollout.py --record`, computes the
canonical avoidance measures per episode, aggregates mean +/- std over all
seeds/episodes per config, writes a stats CSV, and renders a polished heatmap
(rows = experiment config, columns = criterion) via _heatmap_style.render.

Measures (see "Measures & definitions" in the study anchor doc):
  Bush use:
    bush_use_rate       % of EPISODES the agent ever enters the bush ("entered bush")
    bush_entry_step     steps until first entry onto the bush cell ('steps to reach bush')
    bush_hiding          % of the episode's STEPS spent on the bush cell ("time in bush")
  Distance to animal:
    fid                 animal-agent distance at the agent's first move (flight-initiation distance)
    time_near_animal    % of steps the animal is within 1 cell ("time near animal")
    closest_approach    smallest animal-agent distance over the episode
  Movement & chase:
    time_moving         % of steps the agent's position changes
    spatial_spread      radius of gyration: sqrt(var(row)+var(col)), in cells
    pursuit_duration    longest unbroken run of steps with the animal within <=2 cells ('longest chase')
  Outcome:
    injury_change       injury(last) - injury(first)  (+ harmed / - healed)
    survival_steps      episode length

Usage (run from repo root with the project conda interpreter):
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    scripts/behavior_measures/avoidance_stats_heatmap.py \
    --results-root results/eval/avoidance_stat --out-dir results/eval/avoidance_stat/STATS
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))
sys.path.insert(0, str(_here.parent))
from _heatmap_style import render
from src.utils.eval_recording import load_episode

# (key, display label, group, decimals, is_percent)
MEASURES = [
    ("bush_use_rate",    "entered bush\n(% of episodes)", "Bush use", 0, True),
    ("bush_entry_step",  "steps to\nreach bush",          "Bush use", 0, False),
    ("bush_hiding",       "time in bush\n(%)",             "Bush use", 0, True),
    ("fid",              "flight-initiation\ndistance (FID)", "Distance to animal", 1, False),
    ("time_near_animal", "time near\nanimal (%)",         "Distance to animal", 0, True),
    ("closest_approach", "closest\napproach",             "Distance to animal", 2, False),
    ("time_moving",      "time moving\n(%)",              "Movement & chase", 0, True),
    ("spatial_spread",   "spatial spread\n(R_g)",         "Movement & chase", 2, False),
    ("pursuit_duration", "longest chase\n(steps)",        "Movement & chase", 0, False),
    ("injury_change",    "injury change\n(end − start)",  "Outcome", 0, False),
    ("survival_steps",   "survival\nsteps",               "Outcome", 0, False),
]
KEYS = [m[0] for m in MEASURES]
SIGNED = {"injury_change"}
PCT = {k for k, _, _, _, p in MEASURES if p}

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
    # movement: fraction of steps the position changes
    moved = [not np.array_equal(ag[t], ag[t - 1]) for t in range(1, T)]
    time_moving = float(np.mean(moved)) if moved else 0.0
    # radius of gyration: sqrt(var(row)+var(col))
    rows = np.array([p[0] for p in ag], float); cols = np.array([p[1] for p in ag], float)
    spatial_spread = float(np.sqrt(rows.var() + cols.var()))
    has = len(S[0]["animal_pos"]) > 0
    fid = closest = near = np.nan; pursuit = np.nan
    if has:
        ad = [manhattan(ag[t], np.asarray(S[t]["animal_pos"][0])) for t in range(T)]
        closest = float(min(ad)); near = float(np.mean([d <= 1 for d in ad]))
        # longest unbroken run of steps with animal within <=2 cells
        best = cur = 0
        for d in ad:
            cur = cur + 1 if d <= 2 else 0
            best = max(best, cur)
        pursuit = float(best)
        dep = next((t for t in range(T) if not np.array_equal(ag[t], start)), None)
        fid = float(ad[dep]) if dep is not None else np.nan
    return {"bush_use_rate": 1.0 if entry is not None else 0.0,
            "bush_entry_step": float(entry) if entry is not None else np.nan,
            "bush_hiding": float(np.mean(in_bush)),
            "fid": fid, "time_near_animal": near, "closest_approach": closest,
            "time_moving": time_moving, "spatial_spread": spatial_spread,
            "pursuit_duration": pursuit, "injury_change": inj, "survival_steps": float(T)}


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
        out = {}
        for k in KEYS:
            mu = np.mean(acc[k]) if acc[k] else np.nan
            sd = np.std(acc[k]) if acc[k] else np.nan
            if k in PCT:  # 0-1 fraction -> percentage
                mu, sd = mu * 100, sd * 100
            out[k] = (mu, sd)
        stats[cfg] = out
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
    col_decimals = [m[3] for m in MEASURES]
    groups = []
    for _, _, g, _, _ in MEASURES:
        if g not in [n for n, _ in groups]:
            groups.append((g, [j for j, mm in enumerate(MEASURES) if mm[2] == g]))
    bounds = [i for i in range(1, len(configs)) if split_cfg(configs[i])[0] != split_cfg(configs[i - 1])[0]]
    caption = ("n = 30 episodes/condition (seeds 0–29, small initial-state jitter; deterministic policy).   "
               "Cell = mean ± std.   Percentages are of an episode's steps (or of episodes, for 'entered bush').   "
               "Colour is per-criterion: sequential (min–max) for magnitude metrics, diverging at 0 for injury "
               "change (red = net harm, blue = net healed).   FID = flight-initiation distance; spatial spread = "
               "radius of gyration; distances/spread in grid cells.   ‘–’ = not applicable (no animal).")
    render(mean, std, [row_label(c) for c in configs], [m[1] for m in MEASURES],
           signed_cols=signed_cols, col_groups=groups, row_group_bounds=bounds, col_decimals=col_decimals,
           title=args.title, caption=caption,
           out_png=str(out_dir / "avoidance_stats_heatmap.png"),
           out_pdf=str(out_dir / "avoidance_stats_heatmap.pdf"))
    print(f"[avoidance_stats_heatmap] {len(configs)} configs, {len(KEYS)} measures -> {out_dir}")


if __name__ == "__main__":
    main()
