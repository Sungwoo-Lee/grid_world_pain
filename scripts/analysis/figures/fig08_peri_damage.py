#!/usr/bin/env python
"""FIGURE 8 — behaviour around a moment of damage.

QUESTION. Does being damaged change what the agent does, and on what timescale? Time-locking to the
event is the standard way to see a response emerge and decay, and lets us compare the behavioural
timing against the nociception kernel, which peaks three steps after an injury.

MEASURE. Bush dwell from lag -10 to +25 around every damage event, within the same episode. Two
series: ALL events, and ISOLATED events with no other damage in the window.

BASELINE. Lag -10, NOT lag -1. Lag -1 is mechanically depressed because the agent must be out of
cover to be damaged at all, so differencing against it inflates the response.

KNOWN LIMIT. The isolated filter is a COLLIDER: dwelling prevents damage, so filtering for "no
further damage" preferentially keeps cases where dwelling worked. Read the isolated series as an
upper bound and the all-events series as a lower bound. An earlier draft quoted +8 pp sustained to
lag +25 by combining the wrong baseline with the filtered series alone.
"""
import sys, os, time
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import base_args, load_run, find_store, step_shards, episode_starts, save

LAGS = np.arange(-10, 26)

def main():
    a = base_args(__doc__).parse_args()
    cfg, lay = load_run(a.run)
    store = find_store(a.run, a.checkpoint, a.store_root)
    n = np.zeros((len(LAGS), 2)); d = np.zeros((len(LAGS), 2))
    t0 = time.time(); files = step_shards(store)
    for fi, f in enumerate(files):
        t_ = pq.read_table(f, columns=["t", "agent_in_bush", "damage"])
        t = t_.column("t").to_numpy()
        bu = t_.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(float)
        dm = t_.column("damage").to_numpy(zero_copy_only=False)
        st, ends, estart = episode_starts(t)
        eend = np.repeat(ends, ends - st)
        ev = np.flatnonzero((dm > 0) & (t >= 1))
        if not len(ev): continue
        prev = np.r_[-10**9, ev[:-1]]; nxt = np.r_[ev[1:], 10**9]
        isolated = (ev - prev > 10) & (nxt - ev > 25)
        for li, lag in enumerate(LAGS):
            tgt = ev + lag
            ok = (tgt >= estart[ev]) & (tgt < eend[ev])
            for c, sel in ((0, ok), (1, ok & isolated)):
                if sel.any():
                    n[li, c] += sel.sum(); d[li, c] += bu[tgt[sel]].sum()
        if not a.quiet and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    allv = 100*d[:, 0]/np.maximum(n[:, 0], 1)
    isov = 100*d[:, 1]/np.maximum(n[:, 1], 1)
    ba, bi = allv[0], isov[0]
    print("\n=== Figure 8: bush dwell around a damage event ===")
    print(f"baseline at lag -10:  all {ba:.1f}%   isolated {bi:.1f}%\n")
    print(f"{'lag':>5}{'all events':>13}{'isolated':>12}{'all vs base':>13}{'iso vs base':>13}")
    for li, lag in enumerate(LAGS):
        if lag % 2 and abs(lag) > 2: continue
        print(f"{lag:>5}{allv[li]:>12.1f}%{isov[li]:>11.1f}%"
              f"{allv[li]-ba:>+13.1f}{isov[li]-bi:>+13.1f}")
    save("fig08_peri_damage",
         {"figure": 8, "run": a.run, "lags": LAGS.tolist(),
          "all_events": allv.round(2).tolist(), "isolated_events": isov.round(2).tolist(),
          "baseline_lag": -10, "n_all": n[:, 0].tolist(), "n_isolated": n[:, 1].tolist()}, a.quiet)

if __name__ == "__main__":
    main()
