#!/usr/bin/env python3
"""make_manifest.py - name every cell of a neuromodulator grid and say where its data lives.

The sensor-ladder arm sweep (`studies/sensor_ladder/collect_arm_data.py`) computes survival,
bush hiding, how each episode ended, hiding against distance to the nearest predator and to the
nearest rabbit, and the odour grids. Every one of those questions applies unchanged to this grid,
so the grid runs that sweep rather than re-implementing it - a re-implementation that happened to
agree would still not be the same analysis.

What the sweep cannot infer is which run directory and which collected stores belong to a cell
called `t3rnn_I`. That mapping is this file's whole job, and it is written out as JSON so the run
that produced a set of aggregates can be read back months later without re-deriving it.

A cell's evaluation population may span SEVERAL store roots. That is not untidiness: `n_episodes`
is a guarded field of a store's manifest, so a store collected for N episodes cannot be reopened
and extended, and a top-up therefore goes to a fresh root with a continuing seed base. The sweep
treats the union as one population and asserts seed contiguity across it.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
os.chdir(ROOT)

# tag in the run directory name -> the store roots that hold that grid's evaluation episodes
GRIDS = {
    "nmnsite":     ["results/trajectories_nmnsite", "results/trajectories_nmnsite2"],
    "nmngaenorm":  ["results/trajectories_nmngae"],
}
# the five unmodulated reference runs each grid is compared against
BASELINE_TAG = {"nmnsite": "cmp10m_mc", "nmngaenorm": "cmp10m_gaenorm"}


def cell_name(run_dir: str, tag: str) -> str:
    """`20260907-050224_rppo_nmnsite_t3rnn_I_s42` -> `t3rnn_I`; a baseline -> `baseline_s44`."""
    base = os.path.basename(run_dir.rstrip("/"))
    if f"_{tag}_" in base:
        return base.split(f"_{tag}_", 1)[1].rsplit("_s", 1)[0]
    return "baseline_" + base.rsplit("_", 1)[1]          # cmp10m run -> baseline_s44


def build(tag: str) -> dict:
    roots = GRIDS[tag]
    runs = sorted(glob.glob(f"results/JAX_RecurrentPPO/*_{tag}_*/")) \
         + sorted(glob.glob(f"results/JAX_RecurrentPPO/*_{BASELINE_TAG[tag]}_*/"))
    out = {}
    for r in runs:
        base = os.path.basename(r.rstrip("/"))
        stores = []
        for root in roots:
            hits = sorted(glob.glob(f"{root}/{base}/*/*/"))
            if len(hits) > 1:
                sys.exit(f"{base}: {root} holds {len(hits)} stores, expected at most one")
            stores += hits
        if not stores:
            print(f"  skipped (no collected store yet): {base}")
            continue
        name = cell_name(r, tag)
        if name in out:
            sys.exit(f"two runs claim the cell name {name!r}: {out[name]['run']} and {r}")
        out[name] = {"run": r.rstrip("/"), "stores": stores}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=sorted(GRIDS), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = build(a.grid)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(man, open(a.out, "w"), indent=1)
    print(f"{len(man)} cells -> {a.out}")
    for k, v in man.items():
        print(f"  {k:16} {len(v['stores'])} store(s)")
