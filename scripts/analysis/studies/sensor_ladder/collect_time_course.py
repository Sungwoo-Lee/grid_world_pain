#!/usr/bin/env python3
"""collect_time_course.py - the step-by-step sweep, ported onto core/scan.

A faithful port of `scripts/analysis/ladder/build_time_course.py`. Gated against
`results/_golden_prerefactor_20260904/` by `scripts/analysis/core/golden.py`.

This is the sweep that asks "what happened at step t, averaged over episodes" rather than "what
happened in this episode", and it is the one that reconstructs the agent's FELT nociception - the
smoothed, delayed signal the interoceptive nociceptor actually hands the policy, which the store
does not record as a column. That reconstruction now comes from `core/env.perceived_nociception`,
whose reset boundary is pinned by `tests/analysis/test_core_env.py`: the buffer is zeroed at reset,
so an agent that wakes badly wounded feels nothing for two steps, and a version that leaks across
the boundary manufactures the very early response this study's central claim rests on.

ONE FILE PER ARM, deliberately. A single shared time_course.json was read-modify-written by each
arm, so running arms concurrently made the last writer clobber every other arm's entry - and the
result looked fine, because the clobbered entries were stale data from a previous run rather than
missing. Per-arm files make that failure structurally impossible.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis", "core"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis", "ladder"))
os.chdir(ROOT)

import env as ENV                    # noqa: E402
import scan as SCAN                  # noqa: E402
import store as STORE                # noqa: E402
import _ladder as L                  # noqa: E402

MAXT = 120
EP_COLS = ["episode_seed", "length"]
STEP_COLS = ["episode_seed", "t", "injury_level", "agent_in_bush", "nutrition", "ate_food"]


def sweep(arm: str, verbose: bool = True) -> dict:
    kernel = ENV.nociception_kernel(L.arm_config(L.arm_runs()[arm]))
    st = STORE.open_run(L.arm_stores(arm), EP_COLS)

    acc = {k: np.zeros((4, MAXT)) for k in ["injury", "noci", "bush", "nutrition", "ate", "n"]}
    dose = {"bush": np.zeros(4), "n": np.zeros(4)}
    inj0 = np.full(st.n_episodes, np.nan)

    def collect(fr, _acc, fi):
        gi, t = fr.episode_id, fr.t
        inj = fr.raw("injury_level")
        inj0[fr.episodes] = fr.at_initial(inj)
        noci = ENV.perceived_nociception(inj, t, fr.estart, kernel)

        m = t < MAXT
        if np.isnan(inj0[gi[m]]).any():
            raise SystemExit(f"{arm}: a step row precedes its episode's t=0 row - shards misordered")
        ib = np.digitize(inj0[gi[m]], L.INJ_EDGES)
        np.add.at(acc["injury"], (ib, t[m]), inj[m])
        np.add.at(acc["noci"], (ib, t[m]), noci[m])

        # The PREDICTOR comes from the PREVIOUS row, as everywhere else in this analysis: the action
        # that put the agent in a bush at row t was chosen while it was feeling row t-1's signal.
        # An earlier version paired noci[t] with bush[t] - the same row - which relates the state
        # AFTER the action to the action itself. It moved the published spread by 0.35 pp (the
        # signal is a twelve-step convolution, so adjacent rows barely differ), but it was wrong.
        # `> estart + 1` rather than fr.is_step: the previous row must itself be a step.
        idx = np.arange(fr.n)
        step = idx > fr.estart + 1
        nb = np.digitize(noci[idx[step] - 1], L.INJ_EDGES)
        np.add.at(dose["bush"], nb, fr.raw("agent_in_bush")[step])
        np.add.at(dose["n"], nb, 1.0)
        np.add.at(acc["bush"], (ib, t[m]), fr.raw("agent_in_bush")[m])
        np.add.at(acc["nutrition"], (ib, t[m]), fr.raw("nutrition")[m])
        np.add.at(acc["ate"], (ib, t[m]), fr.raw("ate_food")[m])
        np.add.at(acc["n"], (ib, t[m]), 1.0)

    SCAN.sweep(st, STEP_COLS, collect, verbose=verbose, label=arm)
    n = np.maximum(acc["n"], 1)
    return {k: (acc[k] / n).tolist() for k in ["injury", "noci", "bush", "nutrition", "ate"]} | \
           {"n": acc["n"].tolist(),
            "dose_bush": dose["bush"].tolist(), "dose_n": dose["n"].tolist(),
            "kernel": kernel.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(L.OUT_ROOT, exist_ok=True)
    for arm in (a.arms or L.ARM_ORDER):
        t0 = time.time()
        p = f"{L.OUT_ROOT}/time_course_{arm}.json"
        json.dump(sweep(arm), open(p, "w"))
        print(f"{arm:22} done ({time.time()-t0:.0f}s) -> {p}", flush=True)
