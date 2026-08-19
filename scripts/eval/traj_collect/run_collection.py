#!/usr/bin/env python3
"""run_collection.py — multi-node driver for the trajectory-collection pipeline.

Reads a spec YAML (see `configs/trajectory_collection/example.yaml`), expands it into
`(run x checkpoint x block-range)` cells, LPT-partitions those cells across nodes, writes
one worklist per node, launches the per-node workers **serially** via `run_command.py`,
polls completion markers, then validates every store and prints a per-run summary.

`run_command.py` is NOT parallel-safe — concurrent invocations race through a shared SSH
control socket and can return one node's answer to every caller. Nodes are therefore
launched strictly one `subprocess.run` at a time, exactly as
`scripts/eval/dwell_sweep/run_sweep.py:275` does.

Usage
-----
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
        scripts/eval/traj_collect/run_collection.py \
        configs/trajectory_collection/example.yaml [--dry-run] [--validate-only]
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # scripts/eval/traj_collect -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.trajectory_store import (  # noqa: E402
    completed_blocks, read_manifest, validate_store_draws, validate_store_shapes,
    validate_store_structure,
)

PY = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
RUN_COMMAND = PROJECT_ROOT / "run_command.py"
WORKER = PROJECT_ROOT / "scripts" / "eval" / "traj_collect" / "collect_worker.sh"

# Operational keys with documented defaults. Every SCIENTIFIC key is mandatory.
DEFAULTS = {"device": "cpu", "shard_episodes": 5000,
            "npar": {"cpu": 16, "gpu": 1}, "batch_size": {"cpu": 1024, "gpu": 8192},
            "blocks_per_cell": 10}
MANDATORY = ("name", "algo", "out_root", "episodes", "seed_base", "checkpoints",
             "nodes", "obs_precision", "runs")
OPTIONAL = ("device", "npar", "batch_size", "shard_episodes", "blocks_per_cell")
# The ONLY keys a per-run entry may carry.  `seed_base` is the only SCIENTIFIC override
# (plan §D13); `label` is cosmetic and `allow_ambiguous_scene` is an explicit escape hatch.
RUN_KEYS = ("path", "label", "seed_base", "allow_ambiguous_scene")


def _req(spec: dict, key: str):
    """Mandatory spec access — no fallback defaults for scientific parameters.

    `obs_precision` is mandatory specifically because it is LOSSY: a collection must not
    be launchable without someone stating the measurement precision they accepted.
    """
    if key not in spec or spec[key] is None:
        raise ValueError(
            f"Strict spec: required key {key!r} is missing from the collection spec. "
            f"Mandatory keys are {list(MANDATORY)}.")
    return spec[key]


def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: spec must be a YAML mapping, got {type(spec).__name__}")
    for k in MANDATORY:
        _req(spec, k)

    # UNKNOWN KEYS ARE A HARD ERROR, top-level and per-run.  Silently ignoring them is
    # how a typo becomes a scientific claim: `seed_bases:` instead of `seed_base:` on a
    # run entry would quietly inherit the batch-level base and manufacture exactly the
    # ILLUSORY PAIRING §D13 warns about — two runs whose episode i is asserted to face
    # the same environment draw when it does not.
    unknown = sorted(set(spec) - set(MANDATORY) - set(OPTIONAL))
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level spec key(s) {unknown}. Known keys are "
            f"{sorted(MANDATORY)} (mandatory) and {sorted(OPTIONAL)} (optional). An "
            "unknown key is far more likely to be a typo that silently changes nothing "
            "than a harmless annotation.")

    device = spec.get("device", DEFAULTS["device"])
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    # The two knobs move in OPPOSITE directions and must be set together (plan §D7);
    # derive both from `device` unless both are given explicitly.
    # NOTE (device: gpu limitation): the worker exports JAX_PLATFORMS=cuda but sets no
    # CUDA_VISIBLE_DEVICES, so every GPU worker lands on GPU 0 and the spec has no way to
    # name an index. Multi-GPU fan-out does NOT work today. Only reachable with
    # `device: gpu`; the default is cpu. Consult docs/environment/LAB_NODE_GPU_SPEC.md
    # before using it at all.
    spec["device"] = device
    # `is None`, NOT truthiness: an explicitly written `npar: 0` must reach the positivity
    # check below rather than being silently replaced by the default.
    for key, default in (("npar", DEFAULTS["npar"][device]),
                         ("batch_size", DEFAULTS["batch_size"][device]),
                         ("shard_episodes", DEFAULTS["shard_episodes"]),
                         ("blocks_per_cell", DEFAULTS["blocks_per_cell"])):
        if spec.get(key) is None:
            spec[key] = default
    if spec["algo"] != "rppo":
        raise ValueError(f"algo={spec['algo']!r}: only 'rppo' is supported in this change.")

    for key in ("episodes", "seed_base", "npar", "batch_size", "shard_episodes",
                "blocks_per_cell"):
        v = spec[key]
        if not isinstance(v, int) or isinstance(v, bool) or v < (0 if key == "seed_base" else 1):
            raise ValueError(
                f"{path}: {key} must be a positive integer"
                f"{' (or zero)' if key == 'seed_base' else ''}, got {v!r}")
    if spec["obs_precision"] not in ("float16", "float32"):
        raise ValueError(f"{path}: obs_precision must be 'float16' or 'float32', "
                         f"got {spec['obs_precision']!r}")
    if not isinstance(spec["runs"], list) or not spec["runs"]:
        raise ValueError(f"{path}: `runs` must be a non-empty list")

    for i, run in enumerate(spec["runs"]):
        if not isinstance(run, dict):
            raise ValueError(f"{path}: runs[{i}] must be a mapping with a `path` key, "
                             f"got {type(run).__name__}")
        if "path" not in run or not run["path"]:
            raise ValueError(f"{path}: runs[{i}] is missing the required `path` key "
                             f"(present keys: {sorted(run)})")
        bad = sorted(set(run) - set(RUN_KEYS))
        if bad:
            raise ValueError(
                f"{path}: runs[{i}] ({run.get('label', run['path'])}) carries unknown "
                f"key(s) {bad}. A per-run entry may only carry {list(RUN_KEYS)}. "
                "`seed_base` is the ONLY scientific per-run override; a misspelt one "
                "would silently inherit the batch-level base and manufacture an illusory "
                "paired comparison.")
        # Pre-flight the path here, once, rather than discovering a typo per-cell after
        # every node has already been launched.
        cfg = PROJECT_ROOT / run["path"] / "models" / "config.yaml"
        if not cfg.exists() and not Path(run["path"], "models", "config.yaml").exists():
            raise ValueError(
                f"{path}: runs[{i}] path {run['path']!r} has no models/config.yaml. The "
                "collector reads each run's OWN saved config, so this run cannot be "
                "collected. Fix the path before launching (this check exists so a typo "
                "fails here rather than on every node afterwards).")
    return spec


def expand_cells(spec: dict) -> list[dict]:
    """(run x checkpoint x block-range) cells. A cell is a block RANGE so JAX startup is
    amortised across several blocks in one process."""
    episodes = int(_req(spec, "episodes"))
    shard = int(spec["shard_episodes"])
    n_blocks = math.ceil(episodes / shard)
    per_cell = int(spec["blocks_per_cell"])
    cells = []
    for run in _req(spec, "runs"):       # shape + key validation happened in load_spec
        run_path = run["path"]
        label = run.get("label", Path(run_path).name)
        # `seed_base` is the only SCIENTIFIC per-run override (plan §D13): pairing is a
        # property of the ENVIRONMENT, not of the seed, so a run whose world differs
        # structurally must be given its own base rather than an illusory pairing.
        seed_base = int(run.get("seed_base", spec["seed_base"]))
        for ck in _req(spec, "checkpoints"):
            for lo in range(0, n_blocks, per_cell):
                cells.append({
                    "label": label, "run": run_path, "ckpt": str(ck),
                    "seed_base": seed_base, "episodes": episodes,
                    "blocks": f"{lo}:{min(lo + per_cell, n_blocks)}",
                    "weight": min(per_cell, n_blocks - lo),
                    "extra": "--allow-ambiguous-scene" if run.get("allow_ambiguous_scene") else "-",
                })
    return cells


def lpt_partition(cells, nodes):
    """Longest-Processing-Time-first: assign each whole cell to the node with the
    smallest running total. A cell is never split across nodes."""
    loads = {n: 0 for n in nodes}
    buckets = {n: [] for n in nodes}
    for c in sorted(cells, key=lambda c: -c["weight"]):
        n = min(nodes, key=lambda n: loads[n])
        buckets[n].append(c)
        loads[n] += c["weight"]
    return buckets, loads


def write_worklist(node, cells, scratch_root: Path, spec: dict) -> tuple[Path, int]:
    lines = [
        "|".join([
            c["run"], c["ckpt"], str(spec["out_root"]), str(c["seed_base"]),
            str(c["episodes"]), c["blocks"], spec["device"], str(spec["batch_size"]),
            str(spec["shard_episodes"]), spec["obs_precision"], c["extra"],
        ]) for c in cells
    ]
    wl_dir = scratch_root / "_worklists"
    wl_dir.mkdir(parents=True, exist_ok=True)
    p = wl_dir / f"worklist_{node}.txt"
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    return p, len(lines)


def launch_node(node, wl_path: Path, npar: int, dry_run: bool):
    cmd = f"bash {WORKER} {wl_path} {node} {npar}"
    full = [PY, str(RUN_COMMAND), str(node), cmd, "--no-tail"]
    print(f"  launch: {' '.join(full)}")
    if dry_run:
        return
    subprocess.run(full, check=True)     # SERIAL — run_command.py is not parallel-safe


def poll_done(scratch_root: Path, nodes, interval=15):
    mark = scratch_root / "_run_markers"
    pending, t0 = set(nodes), time.time()
    while pending:
        for n in list(pending):
            if (mark / f"done_{n}").exists():
                pending.discard(n)
                print(f"  node {n} done ({time.time() - t0:.0f}s elapsed)")
        if pending:
            time.sleep(interval)
    print(f"  all {len(nodes)} node(s) done in {time.time() - t0:.0f}s")


def store_dirs(spec: dict) -> list[Path]:
    """Every store directory the spec should have produced."""
    out = []
    root = Path(spec["out_root"])
    for run in _req(spec, "runs"):
        run_path = Path(run["path"] if isinstance(run, dict) else run)
        d = root / run_path.name
        if d.is_dir():
            out += [p.parent for p in d.glob("*/*/_manifest.json")]
    return sorted(set(out))


def validate_and_report(spec: dict) -> int:
    """The driver's final pass: structure + shapes + whole-store realised-draw checks,
    then a per-run summary. Returns a process exit code."""
    dirs = store_dirs(spec)
    if not dirs:
        print("  no stores found — nothing to validate", file=sys.stderr)
        return 1
    bad = 0
    print(f"\n{'run':<48} {'blocks':>7} {'episodes':>9} {'steprows':>10} {'MB':>8} {'B/step':>7}")
    for d in dirs:
        try:
            m = read_manifest(d)
            summary = validate_store_structure(d)
            validate_store_shapes(d)
            validate_store_draws(d)
            nbytes = sum(p.stat().st_size for p in d.glob("*.parquet"))
            per_step = nbytes / summary["step_rows"] if summary["step_rows"] else float("nan")
            print(f"{m['run_dir_name'][:48]:<48} {summary['blocks']:>7} "
                  f"{summary['episodes']:>9} {summary['step_rows']:>10} "
                  f"{nbytes/1e6:>8.1f} {per_step:>7.1f}")
        except Exception as e:
            bad += 1
            print(f"{str(d):<48} VALIDATION FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print launches, run nothing")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip launching; validate the stores the spec names")
    a = ap.parse_args(argv)

    spec = load_spec(a.spec)
    if a.validate_only:
        return validate_and_report(spec)

    cells = expand_cells(spec)
    nodes = [str(n) for n in _req(spec, "nodes")]
    buckets, loads = lpt_partition(cells, nodes)
    scratch = Path(spec["out_root"]) / "_scratch" / spec["name"]
    (scratch / "_run_markers").mkdir(parents=True, exist_ok=True)

    print(f"spec={a.spec}  cells={len(cells)}  nodes={nodes}  device={spec['device']}  "
          f"npar={spec['npar']}  batch_size={spec['batch_size']}")
    for n in nodes:
        wl, k = write_worklist(n, buckets[n], scratch, spec)
        print(f"  node {n}: {k} cells, {loads[n]} blocks -> {wl}")
        # Do NOT clear completion markers under --dry-run. A dry run against a spec name
        # that currently has a LIVE collection would otherwise delete the markers the live
        # driver is polling for, hanging its wait loop forever. A dry run must be
        # observably inert with respect to anything already running.
        if not a.dry_run:
            (scratch / "_run_markers" / f"done_{n}").unlink(missing_ok=True)

    for n in nodes:                       # STRICTLY SERIAL — see the module docstring
        launch_node(n, scratch / "_worklists" / f"worklist_{n}.txt", spec["npar"], a.dry_run)
    if a.dry_run:
        return 0

    poll_done(scratch, nodes)
    fails = sorted((scratch / "_run_markers").glob("fail_*"))
    for f in fails:
        print(f"  !! {f.name}:\n{f.read_text()}", file=sys.stderr)
    return validate_and_report(spec) or (1 if fails else 0)


if __name__ == "__main__":
    sys.exit(main())
