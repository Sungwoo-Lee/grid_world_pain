#!/usr/bin/env python
"""Dwell-history / behavior-metrics sweep driver — the self-serve entry point for the
promoted-from-tmp/ dwell-history pipeline (see README.md for the full picture).

What this reproduces: this session built (and proved, across ~7 sweeps) a working but
gitignored pipeline in tmp/ that, for a set of training runs, rolls out each saved
checkpoint against a battery of fixed "behavior probe" scenarios (e.g. "a predator is
present" vs "no animal"), computes 11 behavior measures per rollout (time spent hiding in
the bush, how close the agent lets a predator get, etc.), and plots how those measures
evolve over training. This script is that pipeline, cleaned up, unified across the two
supported algorithms (rPPO and Dreamer), and committed so it can be re-run any time a
training run advances instead of hand-assembled per sweep.

End-to-end, given a YAML spec (see README.md "Spec schema"), this script:
  1. resolves each run's directory + enumerates its saved checkpoints,
  2. skips checkpoints already present in that run's output CSV (incremental),
  3. LPT-partitions the remaining (run, condition) work across the spec's nodes,
  4. launches one worker per node via run_command.py (`sweep_worker.sh`, eval-only),
  5. polls until all nodes finish,
  6. aggregates every node's recordings into `<out>/<run_label>/<condition>.csv`
     (ONE aggregation path shared by both algorithms — a recursive glob over each
     scratch checkpoint dir, since the underlying eval_rollout.py --batched path nests
     recordings one directory level deeper than plain per-checkpoint output),
  7. renders the requested measures via plot_summary.py.

Usage:
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
      scripts/eval/dwell_sweep/run_sweep.py configs/eval_sweeps/basic04_variants_rppo.yaml
  ... --dry-run   # build the worklist + LPT partition, print launch commands, don't launch
  ... --max-checkpoints 3   # cap each (run, condition) to its newest N pending checkpoints
"""
import argparse
import csv
import glob as globmod
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[3]  # scripts/eval/dwell_sweep/run_sweep.py -> repo root (3 up)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "behavior_measures"))
sys.path.insert(0, str(_HERE.parent))
from avoidance_stats_heatmap import episode_measures, KEYS  # noqa: E402 (bare-name import, see SCRIPTS_DEPENDENCY_MAP.md)
from src.utils.eval_recording import load_episode  # noqa: E402
import plot_summary  # noqa: E402

PY = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
RUN_COMMAND = REPO_ROOT / "run_command.py"
WORKER = _HERE.parent / "sweep_worker.sh"

CLEAN_PROBE_DIR = REPO_ROOT / "configs/environment/experiment/behavior_probes/core/avoidance"
NOISE_PROBE_DIR = REPO_ROOT / "configs/environment/experiment/behavior_probes/explore/avoidance_stat_noise"

# Tuned NPAR defaults (see README.md "Tuning notes"): rPPO's single-env eval is light
# enough for ~1 process/core; batched Dreamer's vmap parallelises across cores on its
# own even with thread caps, so it needs a much lower process count.
DEFAULT_NPAR = {"rppo": 18, "dreamer": 5}
X_AXIS_DEFAULT = {"rppo": "steps", "dreamer": "episodes"}
HEAD = ["step", "step_M"] + KEYS


def mandatory(d, key, ctx=""):
    if key not in d or d[key] is None:
        suffix = f" ({ctx})" if ctx else ""
        raise ValueError(f"eval-sweep spec missing mandatory key '{key}'{suffix}")
    return d[key]


def load_spec(path):
    with open(path) as f:
        spec = yaml.safe_load(f)
    for k in ("name", "algo", "runs", "conditions", "nodes"):
        mandatory(spec, k)
    if spec["algo"] not in ("rppo", "dreamer"):
        raise ValueError(f"spec.algo must be 'rppo' or 'dreamer', got {spec['algo']!r}")
    return spec


def resolve_run_dir(path_or_glob, algo):
    has_glob = any(c in path_or_glob for c in "*?[")
    if has_glob:
        matches = sorted(globmod.glob(str(REPO_ROOT / path_or_glob)))
        if len(matches) != 1:
            raise ValueError(
                f"runs[].path glob '{path_or_glob}' resolved to {len(matches)} directories "
                f"(need exactly 1): {matches}"
            )
        run_dir = Path(matches[0])
    else:
        p = Path(path_or_glob)
        run_dir = p if p.is_absolute() else REPO_ROOT / p
    expected_root = "JAX_RecurrentPPO" if algo == "rppo" else "JAX_DreamerSRL"
    if expected_root not in str(run_dir):
        print(f"  WARNING: run dir '{run_dir}' does not contain '{expected_root}' "
              f"(spec.algo={algo!r}) -- double check this run is the right algorithm.")
    return run_dir


def list_checkpoints(algo, run_dir):
    sub = "models" if algo == "rppo" else "checkpoints"
    d = run_dir / sub
    if not d.is_dir():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.name.isdigit())


def checkpoint_path(algo, run_dir, step):
    sub = "models" if algo == "rppo" else "checkpoints"
    return run_dir / sub / str(step)


def default_agent_config(run_dir):
    cand = run_dir / "models" / "agent_config.yaml"
    if not cand.exists():
        raise ValueError(
            f"Dreamer run {run_dir} has no models/agent_config.yaml and its spec entry "
            "gives no explicit 'agent_config' override -- one of the two is required."
        )
    return cand


def probe_dir(probe):
    if probe not in ("clean", "noise"):
        raise ValueError(f"spec.probe must be 'clean' or 'noise', got {probe!r}")
    return CLEAN_PROBE_DIR if probe == "clean" else NOISE_PROBE_DIR


def resolve_conditions(spec):
    pdir = probe_dir(spec.get("probe", "clean"))
    conds = mandatory(spec, "conditions")
    if conds == "all":
        return sorted(p.stem for p in pdir.glob("avoid_*.yaml"))
    return list(conds)


def read_existing_max_step(csv_path):
    if not csv_path.exists():
        return 0
    maxstep = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                maxstep = max(maxstep, int(row["step"]))
            except (KeyError, ValueError, TypeError):
                pass
    return maxstep


def build_groups(spec, output_dir, max_checkpoints):
    algo = spec["algo"]
    pdir = probe_dir(spec.get("probe", "clean"))
    conditions = resolve_conditions(spec)
    groups = []
    for run in spec["runs"]:
        label = mandatory(run, "label", "runs[]")
        path = mandatory(run, "path", "runs[]")
        run_dir = resolve_run_dir(path, algo)
        all_ckpts = list_checkpoints(algo, run_dir)
        agent_config = run.get("agent_config")
        if agent_config:
            agent_config = REPO_ROOT / agent_config if not Path(agent_config).is_absolute() else Path(agent_config)
        elif algo == "dreamer":
            agent_config = default_agent_config(run_dir)
        for cond in conditions:
            cfg_path = pdir / f"{cond}.yaml"
            if not cfg_path.exists():
                raise ValueError(f"condition config not found: {cfg_path}")
            out_csv = output_dir / label / f"{cond}.csv"
            maxdone = read_existing_max_step(out_csv)
            newck = [c for c in all_ckpts if c > maxdone]
            if max_checkpoints:
                newck = newck[-max_checkpoints:]
            if not newck:
                continue
            groups.append({
                "run_label": label, "cond": cond, "checkpoints": newck,
                "cfg_path": cfg_path, "agent_config": agent_config,
                "run_dir": run_dir, "out_csv": out_csv,
            })
    return groups


def lpt_partition(groups, nodes):
    """Longest-Processing-Time-first: sort (run,cond) groups by size descending, assign
    each whole group to whichever node currently has the smallest running total."""
    loads = {n: 0 for n in nodes}
    buckets = {n: [] for n in nodes}
    for g in sorted(groups, key=lambda g: -len(g["checkpoints"])):
        n = min(nodes, key=lambda n: loads[n])
        buckets[n].append(g)
        loads[n] += len(g["checkpoints"])
    return buckets, loads


def write_worklist(node, groups, scratch_root, algo):
    lines = []
    for g in groups:
        for step in g["checkpoints"]:
            ck = checkpoint_path(algo, g["run_dir"], step)
            out = scratch_root / g["run_label"] / g["cond"] / str(step)
            agent = str(g["agent_config"]) if g["agent_config"] else "-"
            lines.append(f'{g["cfg_path"]}|{agent}|{ck}|-|{out}')
    wl_dir = scratch_root / "_worklists"
    wl_dir.mkdir(parents=True, exist_ok=True)
    wl_path = wl_dir / f"worklist_{node}.txt"
    wl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return wl_path, len(lines)


def launch_node(node, wl_path, npar, episodes, dry_run):
    cmd = f"bash {WORKER} {wl_path} {node} {npar} {episodes}"
    full = [PY, str(RUN_COMMAND), str(node), cmd, "--no-tail"]
    print(f"  launch: {' '.join(full)}")
    if dry_run:
        return
    subprocess.run(full, check=True)


def poll_done(scratch_root, nodes, interval=15):
    mark = scratch_root / "_run_markers"
    pending = set(nodes)
    t0 = time.time()
    while pending:
        for n in list(pending):
            if (mark / f"done_{n}").exists():
                pending.discard(n)
                print(f"  node {n} done ({time.time() - t0:.0f}s elapsed)")
        if pending:
            time.sleep(interval)
    print(f"  all {len(nodes)} node(s) done in {time.time() - t0:.0f}s")


def aggregate(groups, scratch_root):
    """ONE aggregation path for both algorithms: recursive-glob each (run,cond,step)
    scratch dir for .rec.gz recordings, average the 11 measures, MERGE into the
    existing CSV (never drops prior rows)."""
    n_written = 0
    for g in groups:
        out_csv = g["out_csv"]
        by_step = {}
        if out_csv.exists():
            for r in csv.DictReader(open(out_csv)):
                by_step[int(r["step"])] = [r.get(h, "") for h in HEAD]
        scratch_dir = scratch_root / g["run_label"] / g["cond"]
        for step_dir in sorted(scratch_dir.glob("*")) if scratch_dir.is_dir() else []:
            if not step_dir.name.isdigit():
                continue
            step = int(step_dir.name)
            # layout-agnostic recursive glob: eval_rollout.py --batched nests recordings
            # one level deeper than a flat per-checkpoint output-root (see README.md).
            recs = sorted(step_dir.glob("**/episode_*.rec.gz"))
            if not recs:
                continue
            rows = [episode_measures(load_episode(str(r))) for r in recs]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                agg = {k: float(np.nanmean([row[k] for row in rows])) for k in KEYS}
            vals = [step, f"{step / 1e6:.4f}"] + [
                ("" if not np.isfinite(agg[k]) else f"{agg[k]:.4f}") for k in KEYS
            ]
            by_step[step] = vals
        if not by_step:
            continue
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [by_step[s] for s in sorted(by_step)]
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEAD)
            for v in rows:
                w.writerow(v)
        n_written += 1
    return n_written


def plot(spec, output_dir):
    algo = spec["algo"]
    x_axis = spec.get("x_axis") or X_AXIS_DEFAULT[algo]
    if x_axis == "steps":
        os.environ["XDIV"] = "1e6"
        os.environ["XLABEL"] = "training  (million steps)"
        os.environ.setdefault("XBOUNDARY", "10")
    elif x_axis == "episodes":
        os.environ["XDIV"] = "1e3"
        os.environ["XLABEL"] = "training  (thousand episodes)"
        os.environ["XBOUNDARY"] = ""
    else:
        raise ValueError(f"spec.x_axis must be 'steps' or 'episodes', got {x_axis!r}")
    # plot_summary reads these env vars at import time; re-apply them onto its module
    # globals since we imported it before the spec (and its x_axis) was known.
    plot_summary.XDIV = float(os.environ["XDIV"])
    plot_summary.XLABEL = os.environ["XLABEL"]
    plot_summary.XBND = os.environ["XBOUNDARY"]

    measures = spec.get("plot_measures", ["bush_dwell", "spatial_spread", "survival_steps"])
    figs = []
    for run in spec["runs"]:
        label = run["label"]
        level_dir = output_dir / label
        if not level_dir.is_dir():
            continue
        for m in measures:
            o = plot_summary.fig_for(str(level_dir), label, m)
            if o:
                figs.append(o)
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="Path to a sweep spec YAML (see README.md).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Build the worklist + LPT partition, print launch commands, don't launch.")
    ap.add_argument("--max-checkpoints", type=int, default=None,
                     help="Cap each (run,condition) to its newest N pending checkpoints "
                          "(overrides spec.max_checkpoints; useful for a quick smoke test).")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    algo = spec["algo"]
    output_dir = REPO_ROOT / (spec.get("output_dir") or f"results/eval/avoidance/{spec['name']}")
    scratch_root = output_dir / "_scratch"
    npar = spec.get("npar") or DEFAULT_NPAR[algo]
    episodes = spec.get("episodes", 30)
    nodes = spec["nodes"]
    max_checkpoints = args.max_checkpoints if args.max_checkpoints is not None else spec.get("max_checkpoints")

    print(f"=== dwell sweep: {spec['name']} ({algo}) ===")
    print(f"output: {output_dir}")

    groups = build_groups(spec, output_dir, max_checkpoints)
    total_ck = sum(len(g["checkpoints"]) for g in groups)
    print(f"{len(groups)} (run,condition) group(s) with pending work, {total_ck} checkpoint-eval(s) total")
    if total_ck == 0:
        print("Nothing to do -- every (run,condition) CSV is already up to date with the newest checkpoint.")
        return

    buckets, loads = lpt_partition(groups, nodes)
    for n in nodes:
        print(f"  node {n}: {len(buckets[n])} group(s), {loads[n]} checkpoint-eval(s)")

    wl_paths = {}
    for n in nodes:
        if not buckets[n]:
            continue
        wl_path, nlines = write_worklist(n, buckets[n], scratch_root, algo)
        wl_paths[n] = wl_path
        print(f"  wrote {wl_path} ({nlines} lines)")

    if args.dry_run:
        print("\n[DRY RUN] launch commands (not executed):")
        for n, wl in wl_paths.items():
            print(f"  {PY} {RUN_COMMAND} {n} \"bash {WORKER} {wl} {n} {npar} {episodes}\" --no-tail")
        print("\n[DRY RUN] complete -- nothing launched.")
        return

    t0 = time.time()
    print("\nLaunching...")
    for n, wl in wl_paths.items():
        launch_node(n, wl, npar, episodes, dry_run=False)
    print("Waiting for nodes to finish...")
    poll_done(scratch_root, list(wl_paths.keys()))

    print("Aggregating...")
    n_csv = aggregate(groups, scratch_root)
    print(f"  wrote/updated {n_csv} CSV(s)")

    print("Plotting...")
    figs = plot(spec, output_dir)
    print(f"  wrote {len(figs)} figure(s)")

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed:.0f}s ===")
    for run in spec["runs"]:
        label = run["label"]
        d = output_dir / label
        n_csv_files = len(list(d.glob("*.csv"))) if d.is_dir() else 0
        n_figs = len(list(d.glob("FIG_*.png"))) if d.is_dir() else 0
        print(f"  {label}: {n_csv_files} CSV(s), {n_figs} figure(s)")


if __name__ == "__main__":
    main()
