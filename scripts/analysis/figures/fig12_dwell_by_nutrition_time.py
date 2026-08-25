#!/usr/bin/env python
"""FIGURE 12 — bush dwell by hunger and by elapsed time.

QUESTION. The raw hunger-dwell association is enormous. Before interpreting it, check whether it
survives holding elapsed time constant — nutrition depletes steadily, so it partly encodes "late in
the episode".

MEASURE. Bush dwell in each (elapsed-step band x injury band) cell.

CONDITIONING. Action steps only. No proximity condition — this figure is about the time confound,
not about isolating the wound from threat.

KNOWN LIMIT. Columns bin CONTEMPORANEOUS nutrition, which the agent's own foraging produces — and
here the reverse causation is severe, because dwelling in a bush prevents eating. An earlier draft
misread this figure's first row as evidence about randomised hunger; it is not, and the randomised
contrast is a separate analysis.
"""
import sys, os, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import base_args, load_run, find_store, step_shards, save

TB = [0, 10, 25, 50, 100, 200, 500]
IE = [25.0, 50.0, 75.0]
ILAB = ["starving <25", "25-50", "50-75", "well fed >=75"]

def main():
    a = base_args(__doc__).parse_args()
    cfg, lay = load_run(a.run)
    store = find_store(a.run, a.checkpoint, a.store_root)
    NT, NI = len(TB)-1, len(IE)+1
    n = np.zeros((NT, NI)); d = np.zeros((NT, NI))
    t0 = time.time(); files = step_shards(store)
    for fi, f in enumerate(files):
        t_ = pq.read_table(f, columns=["t", "agent_in_bush", "nutrition"])
        t = t_.column("t").to_numpy(); m = t >= 1
        inj = t_.column("nutrition").to_numpy(zero_copy_only=False)[m]
        bu = t_.column("agent_in_bush").to_numpy(zero_copy_only=False)[m].astype(float)
        tb = np.clip(np.digitize(t[m], TB[1:-1]), 0, NT-1)
        ib = np.digitize(inj, IE)
        k = tb*NI + ib
        n += np.bincount(k, minlength=NT*NI).reshape(NT, NI)
        d += np.bincount(k, weights=bu, minlength=NT*NI).reshape(NT, NI)
        if not a.quiet and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    grid = (100*d/np.maximum(n, 1)).round(1)
    print("\n=== Figure 12: bush dwell by elapsed time (rows) and hunger (columns) ===")
    print(f"{'steps into episode':22}" + "".join(f"{c:>14}" for c in ILAB))
    for i in range(NT):
        print(f"{str(TB[i])+'-'+str(TB[i+1]):22}" +
              "".join(f"{grid[i,j]:>12.1f}% " for j in range(NI)))
    save("fig12_dwell_by_nutrition_time",
         {"figure": 10, "run": a.run, "time_bands": TB, "nutrition_edges": IE,
          "nutrition_labels": ILAB, "bush_dwell": grid.tolist(),
          "steps": (n/1e6).round(3).tolist()}, a.quiet)

if __name__ == "__main__":
    main()
