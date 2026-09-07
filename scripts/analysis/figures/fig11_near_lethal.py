#!/usr/bin/env python
"""FIGURE 11 — bush hiding as the wound nears fatal.

QUESTION. Protective behaviour rises with injury over most of the range. Does it keep rising as
death approaches, or give way?

MEASURE. Bush hiding by injury_level band, up to the lethal ceiling.

CONDITIONING. Steps with no predator within Chebyshev 2, so the pattern is not just proximity.

KNOWN LIMIT. Associational — injury during an episode is the agent's own doing. The top band is the
smallest cell reported anywhere in this analysis (~140,000 steps); everything else exceeds a
million.
"""
import sys, os, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (base_args, load_run, find_store, step_shards, episode_table,
                     listcol, save)

EDGES = [1e-9, 25.0, 50.0, 75.0, 90.0]
LABELS = ["no wound", "0-25", "25-50", "50-75", "75-90", "90-100 (fatal at 100)"]
NEAR_D = 2

def main():
    a = base_args(__doc__).parse_args()
    cfg, lay = load_run(a.run)
    store = find_store(a.run, a.checkpoint, a.store_root)
    tb, o = episode_table(store, ["episode_seed", "animal_active"])
    seed0 = int(tb.column("episode_seed").to_numpy()[o][0])
    act = np.array(tb.column("animal_active").to_pylist(), bool)[o]
    P = lay["pred"]; nb = len(EDGES) + 1
    n = np.zeros(nb); d = np.zeros(nb)
    t0 = time.time(); files = step_shards(store)
    for fi, f in enumerate(files):
        t_ = pq.read_table(f, columns=["episode_seed", "t", "agent_in_bush", "injury_level",
                                       "agent_row", "agent_col", "animal_row", "animal_col"])
        sd = t_.column("episode_seed").to_numpy(); t = t_.column("t").to_numpy()
        ar = t_.column("agent_row").to_numpy(); ac = t_.column("agent_col").to_numpy()
        AR = listcol(t_.column("animal_row"), lay["n_animal"])
        AC = listcol(t_.column("animal_col"), lay["n_animal"])
        pn = ((np.maximum(np.abs(AR-ar[:, None]), np.abs(AC-ac[:, None])) <= NEAR_D)
              & act[sd - seed0])[:, P].any(1)
        inj = t_.column("injury_level").to_numpy(zero_copy_only=False)
        bu = t_.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(float)
        m = (t >= 1) & (~pn)
        b = np.digitize(inj[m], EDGES)
        n += np.bincount(b, minlength=nb); d += np.bincount(b, weights=bu[m], minlength=nb)
        if not a.quiet and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    rows = [dict(band=LABELS[i], steps=float(n[i]),
                 bush_dwell=float(100*d[i]/max(n[i],1))) for i in range(nb)]
    print("\n=== Figure 11: bush hiding as the wound approaches the lethal ceiling ===")
    print(f"{'injury band':26}{'steps':>15}{'bush hiding':>13}")
    for r in rows:
        print(f"{r['band']:26}{r['steps']:>15,.0f}{r['bush_dwell']:>12.1f}%")
    save("fig11_near_lethal", {"figure": 11, "run": a.run, "edges": EDGES,
                               "condition": "no predator within 2 tiles", "rows": rows}, a.quiet)

if __name__ == "__main__":
    main()
