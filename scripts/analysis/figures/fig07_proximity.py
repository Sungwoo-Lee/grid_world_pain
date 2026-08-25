#!/usr/bin/env python
"""FIGURE 7 — what was nearby one step earlier.

QUESTION. Before crediting olfaction with anything, check the simpler driver: is a predator simply
close? And test the obvious reverse reading, that predators linger near an already-hidden agent
rather than the agent taking cover because one arrived.

MEASURE. Bush dwell at step t, split by what was within Chebyshev distance 2 at step t-1. Lagging
is the point: it means the agent's choice at t cannot have produced the classification.

CONDITIONING. None beyond the lag. All action steps of the run.

KNOWN LIMIT. Associational. Proximity is partly the agent's own doing, and consecutive steps are
autocorrelated, so the lagged version being stronger argues against the reverse reading without
eliminating it.
"""
import sys, os, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (base_args, load_run, find_store, step_shards, episode_table,
                     listcol, episode_starts, save)

NEAR_D = 2

def main():
    a = base_args(__doc__).parse_args()
    cfg, lay = load_run(a.run)
    store = find_store(a.run, a.checkpoint, a.store_root)
    tb, o = episode_table(store, ["episode_seed", "animal_active"])
    seed = tb.column("episode_seed").to_numpy()[o]
    act = np.array(tb.column("animal_active").to_pylist(), bool)[o]
    seed0 = int(seed[0])
    P, R = lay["pred"], lay["neutral"]
    same = np.zeros(3); dwell = np.zeros(3)          # 0 neither, 1 predator, 2 rabbit
    t0 = time.time(); files = step_shards(store)
    for fi, f in enumerate(files):
        t_ = pq.read_table(f, columns=["episode_seed", "t", "agent_in_bush",
                                       "agent_row", "agent_col", "animal_row", "animal_col"])
        sd = t_.column("episode_seed").to_numpy(); t = t_.column("t").to_numpy()
        ar = t_.column("agent_row").to_numpy(); ac = t_.column("agent_col").to_numpy()
        AR = listcol(t_.column("animal_row"), lay["n_animal"])
        AC = listcol(t_.column("animal_col"), lay["n_animal"])
        near = (np.maximum(np.abs(AR - ar[:, None]), np.abs(AC - ac[:, None])) <= NEAR_D) \
               & act[sd - seed0]
        pn = near[:, P].any(1); rn = near[:, R].any(1) & ~pn
        state = np.where(pn, 1, np.where(rn, 2, 0))
        lag = np.zeros(len(t), np.int64); lag[1:] = state[:-1]
        cont = np.zeros(len(t), bool); cont[1:] = sd[1:] == sd[:-1]
        bu = t_.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(float)
        m = (t >= 1) & cont
        same += np.bincount(lag[m], minlength=3)
        dwell += np.bincount(lag[m], weights=bu[m], minlength=3)
        if not a.quiet and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    lab = ["neither", "a predator within 2 tiles", "a rabbit within 2 tiles (no predator)"]
    rows = [dict(state=lab[i], steps=float(same[i]), bush_dwell=float(100*dwell[i]/max(same[i],1)))
            for i in (1, 2, 0)]
    print(f"\n=== Figure 7: bush dwell by what was nearby ONE STEP EARLIER ===")
    print(f"{'nearby at t-1':40}{'steps':>16}{'bush dwell':>13}")
    for r in rows:
        print(f"{r['state']:40}{r['steps']:>16,.0f}{r['bush_dwell']:>12.1f}%")
    save("fig07_proximity", {"figure": 7, "run": a.run, "near_distance": NEAR_D,
                             "lagged": True, "rows": rows}, a.quiet)

if __name__ == "__main__":
    main()
