"""Second sweep: what happens step by step after the agent wakes up wounded.

WHY A SECOND SWEEP. `build_arm_data.py` bins over whole episodes or fixed windows, which answers
"does a wound change behaviour" but not "for how long". That distinction turned out to matter: a
reviewer pointed out that the injury effect shrinks as the outcome window is widened, which looks
like a fragile result chosen at a flattering window. It is not - the assigned wound HEALS, so the
cause disappears and the effect must go with it. Showing that requires the step-by-step trace, so
it gets its own pass.

WHAT IT ACCUMULATES, per arm, per starting-wound quarter, for the first 120 steps:
  the injury level still carried  (the dose)
  bush dwell                      (the response)
  nutrition and food eaten        (the cost the response incurs)

Writes results/analysis/ladder/time_course.json. Takes about a minute per arm.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ladder as L

MAXT = 120
COLS = ["episode_seed", "t", "injury_level", "nutrition", "ate_food", "agent_in_bush"]


def sweep(arm: str) -> dict:
    stores = L.arm_stores(arm)
    ep = pq.read_table(L.store_files(stores, "episodes"), columns=["episode_seed"])
    sd0 = ep.column("episode_seed").to_numpy()
    o = np.argsort(sd0); seed0 = int(sd0[o][0]); nep = len(o)
    acc = {k: np.zeros((4, MAXT)) for k in ["injury", "bush", "nutrition", "ate", "n"]}
    inj0 = np.full(nep, np.nan)
    for f in L.store_files(stores, "steps"):
        tb = pq.read_table(f, columns=COLS)
        sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy()
        gi = sd - seed0
        st = np.flatnonzero(t == 0)
        num = lambda c: tb.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        inj = num("injury_level")
        inj0[gi[st]] = inj[st]
        m = t < MAXT
        if np.isnan(inj0[gi[m]]).any():
            raise SystemExit(f"{arm}: a step row precedes its episode's t=0 row - shards misordered")
        ib = np.digitize(inj0[gi[m]], L.INJ_EDGES)
        np.add.at(acc["injury"], (ib, t[m]), inj[m])
        np.add.at(acc["bush"], (ib, t[m]), num("agent_in_bush")[m])
        np.add.at(acc["nutrition"], (ib, t[m]), num("nutrition")[m])
        np.add.at(acc["ate"], (ib, t[m]), num("ate_food")[m])
        np.add.at(acc["n"], (ib, t[m]), 1.0)
    n = np.maximum(acc["n"], 1)
    return {k: (acc[k] / n).tolist() for k in ["injury", "bush", "nutrition", "ate"]} | \
           {"n": acc["n"].tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    import json
    p = f"{L.OUT_ROOT}/time_course.json"
    out = json.load(open(p)) if os.path.exists(p) else {}
    for arm in (a.arms or L.ARM_ORDER):
        t0 = time.time(); out[arm] = sweep(arm)
        os.makedirs(L.OUT_ROOT, exist_ok=True); json.dump(out, open(p, "w"))
        print(f"{arm:22} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"written: {p}")
