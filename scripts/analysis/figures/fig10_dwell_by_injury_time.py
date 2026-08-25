#!/usr/bin/env python
"""FIGURE 10 — bush dwell by wound level and by elapsed time.

QUESTION. Injury accumulates over an episode and so does much else, so the wound-dwell link might
just be a proxy for "late in the episode". Splitting by elapsed time as well as wound level tests
that directly.

MEASURE. Bush dwell in each (elapsed-step band x injury band) cell.

CONDITIONING. Action steps only. No proximity condition — this figure is about the time confound,
not about isolating the wound from threat.

KNOWN LIMIT. Columns bin the CONTEMPORANEOUS wound, which the agent's own behaviour produces. This
shows the link is not merely a time proxy; it is not a causal estimate. The causal handle is the
randomised start injury, used in Figure 18.
"""
import sys, os, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import base_args, load_run, find_store, step_shards, save

TB = [0, 10, 25, 50, 100, 200, 500]
IE = [1e-9, 25.0, 50.0]
ILAB = ["no wound", "0-25", "25-50", ">=50"]

def main():
    a = base_args(__doc__).parse_args()
    cfg, lay = load_run(a.run)
    store = find_store(a.run, a.checkpoint, a.store_root)
    NT, NI = len(TB)-1, len(IE)+1
    n = np.zeros((NT, NI)); d = np.zeros((NT, NI))
    t0 = time.time(); files = step_shards(store)
    for fi, f in enumerate(files):
        t_ = pq.read_table(f, columns=["t", "agent_in_bush", "injury_level"])
        t = t_.column("t").to_numpy(); m = t >= 1
        inj = t_.column("injury_level").to_numpy(zero_copy_only=False)[m]
        bu = t_.column("agent_in_bush").to_numpy(zero_copy_only=False)[m].astype(float)
        tb = np.clip(np.digitize(t[m], TB[1:-1]), 0, NT-1)
        ib = np.digitize(inj, IE)
        k = tb*NI + ib
        n += np.bincount(k, minlength=NT*NI).reshape(NT, NI)
        d += np.bincount(k, weights=bu, minlength=NT*NI).reshape(NT, NI)
        if not a.quiet and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    grid = (100*d/np.maximum(n, 1)).round(1)
    print("\n=== Figure 10: bush dwell by elapsed time (rows) and wound (columns) ===")
    print(f"{'steps into episode':22}" + "".join(f"{c:>14}" for c in ILAB))
    for i in range(NT):
        print(f"{str(TB[i])+'-'+str(TB[i+1]):22}" +
              "".join(f"{grid[i,j]:>12.1f}% " for j in range(NI)))
    save("fig10_dwell_by_injury_time",
         {"figure": 10, "run": a.run, "time_bands": TB, "injury_edges": IE,
          "injury_labels": ILAB, "bush_dwell": grid.tolist(),
          "steps": (n/1e6).round(3).tolist()}, a.quiet)

if __name__ == "__main__":
    main()
