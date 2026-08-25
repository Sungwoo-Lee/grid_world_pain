#!/usr/bin/env python
"""FIGURE 9 — the same nociception level, different origin.

QUESTION. This is the study's central question. The signal could drive behaviour two ways: as
something to react to in itself, or as evidence that a predator is nearby. Those are normally
indistinguishable — but the environment randomly assigns some agents a wound at reset, with nothing
having attacked them. That gives the signal WITHOUT the evidence, and lets the two come apart.

MEASURE. Bush dwell by (reconstructed nociception level x number of prior damage events this
episode). Prior-hit count is a memory quantity that cannot be recovered from the current signal
value, so if it predicts behaviour at matched signal, the signal alone does not determine the
response.

RECONSTRUCTION. The store records injury_level but not the observation vector, so the received
signal is rebuilt: a 12-slot buffer of injury LEVELS, zeroed at reset, convolved with the run's own
alpha kernel. Mirrors core.py:115 and :1112 and sensor.py. The reset row is NOT in the buffer —
including it was a real bug that fabricated an all-zero column in an earlier version.

PAIRING. Dwell at t against the signal at t-1, since the action producing row t was chosen on the
t-1 observation.

CONDITIONING. Steps with no predator within Chebyshev 2, so current proximity cannot drive it.

KNOWN LIMIT. Zero-prior-hit states at high signal are necessarily early in the episode, because a
start wound heals within ~20 steps. Origin is therefore confounded with elapsed time.
"""
import sys, os, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (base_args, load_run, find_store, step_shards, episode_table, listcol,
                     episode_starts, reconstruct_nociception, previous_row,
                     nociception_kernel, save)

PE = [1e-9, 8.0, 18.0, 32.0, 50.0]
PLAB = ["none", "0-8", "8-18", "18-32", "32-50", "50+"]
HB = [1, 2, 3, 5]
HLAB = ["0 hits", "1", "2", "3-4", "5+"]
NEAR_D = 2

def main():
    a = base_args(__doc__).parse_args()
    cfg, lay = load_run(a.run)
    ker = nociception_kernel(cfg)
    store = find_store(a.run, a.checkpoint, a.store_root)
    tb, o = episode_table(store, ["episode_seed", "animal_active"])
    seed0 = int(tb.column("episode_seed").to_numpy()[o][0])
    act = np.array(tb.column("animal_active").to_pylist(), bool)[o]
    P = lay["pred"]; NP, NH = len(PE)+1, len(HB)+1
    n = np.zeros((NP, NH)); d = np.zeros((NP, NH))
    t0 = time.time(); files = step_shards(store)
    for fi, f in enumerate(files):
        t_ = pq.read_table(f, columns=["episode_seed", "t", "agent_in_bush", "injury_level",
                                       "damage", "agent_row", "agent_col",
                                       "animal_row", "animal_col"])
        sd = t_.column("episode_seed").to_numpy(); t = t_.column("t").to_numpy()
        inj = t_.column("injury_level").to_numpy(zero_copy_only=False).astype(float)
        bu = t_.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(float)
        dm = t_.column("damage").to_numpy(zero_copy_only=False)
        sig = reconstruct_nociception(inj, t, ker)
        prev = previous_row(sig, t)
        st, ends, estart = episode_starts(t)
        hit = (dm > 0).astype(np.int64); cs = np.cumsum(hit)
        nh = cs - np.where(estart > 0, cs[estart-1], 0) - hit      # strictly prior
        ar = t_.column("agent_row").to_numpy(); ac = t_.column("agent_col").to_numpy()
        AR = listcol(t_.column("animal_row"), lay["n_animal"])
        AC = listcol(t_.column("animal_col"), lay["n_animal"])
        pn = ((np.maximum(np.abs(AR-ar[:, None]), np.abs(AC-ac[:, None])) <= NEAR_D)
              & act[sd - seed0])[:, P].any(1)
        m = (t >= 2) & (~pn)
        k = np.digitize(prev[m], PE)*NH + np.digitize(nh[m], HB)
        n += np.bincount(k, minlength=NP*NH).reshape(NP, NH)
        d += np.bincount(k, weights=bu[m], minlength=NP*NH).reshape(NP, NH)
        if not a.quiet and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    grid = (100*d/np.maximum(n, 1)).round(1)
    print("\n=== Figure 9: bush dwell by nociception level (rows) x prior hits (cols) ===")
    print(f"{'nociception':14}" + "".join(f"{c:>13}" for c in HLAB))
    for i in range(NP):
        print(f"{PLAB[i]:14}" + "".join(f"{grid[i,j]:>11.1f}% " for j in range(NH)))
    save("fig09_nociception_by_origin",
         {"figure": 9, "run": a.run, "noci_edges": PE, "noci_labels": PLAB,
          "hit_bins": HB, "hit_labels": HLAB, "bush_dwell": grid.tolist(),
          "steps": (n/1e6).round(3).tolist(),
          "condition": "no predator within 2 tiles; dwell at t vs signal at t-1"}, a.quiet)

if __name__ == "__main__":
    main()
