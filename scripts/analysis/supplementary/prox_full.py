"""Per-episode cross-tab: bush occupancy x proximity state, same-step and lagged.
Proximity state per step: PRED (a predator within Chebyshev 2), RAB (a rabbit within 2
and no predator), NEITHER. Lagged variant classifies by the state at t-1, bush at t."""
import glob, time, numpy as np, yaml
import pyarrow.parquet as pq

S   = sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]
RUN = "results/JAX_RecurrentPPO/20260810-185749_rppo_restprem_a01_n106"
env = yaml.safe_load(open(f"{RUN}/models/config.yaml"))["environment"]
PRED, RAB, s = [], [], 0
for e in env["entities"]:
    hi = e["count_high"]
    (PRED if e["class"] == "predator" else RAB).extend(range(s, s+hi)); s += hi
D = 2

ep = pq.read_table(sorted(glob.glob(S+"episodes_*.parquet")),
                   columns=["episode_seed","animal_active"])
amap = {int(k): v for k, v in zip(ep.column("episode_seed").to_numpy(),
        np.array(ep.column("animal_active").to_pylist(), bool))}

# per episode: 12 counts = [n, nbush] x nothing... layout below
# 0 nP  1 bP  2 nR  3 bR  4 nN  5 bN      (same-step)
# 6 nP  7 bP  8 nR  9 bR 10 nN 11 bN      (lagged: state t-1, bush t)
acc = {}
t0 = time.time()
files = sorted(glob.glob(S+"steps_*.parquet"))
for i, f in enumerate(files):
    tb = pq.read_table(f, columns=["episode_seed","agent_row","agent_col",
                                   "animal_row","animal_col","agent_in_bush"])
    sd = tb.column("episode_seed").to_numpy()
    ar = tb.column("agent_row").to_numpy(); ac = tb.column("agent_col").to_numpy()
    AR = np.array(tb.column("animal_row").to_pylist())
    AC = np.array(tb.column("animal_col").to_pylist())
    bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.int64)
    A  = np.array([amap[int(x)] for x in sd])
    near = (np.maximum(np.abs(AR-ar[:,None]), np.abs(AC-ac[:,None])) <= D) & A
    pn = near[:, PRED].any(1)
    rn = near[:, RAB].any(1) & ~pn
    nn = ~pn & ~rn
    same = np.zeros(len(sd), bool); same[1:] = sd[1:] == sd[:-1]
    # lagged state, valid only where previous row is the same episode
    lp = np.zeros(len(sd), bool); lr = np.zeros(len(sd), bool); ln = np.zeros(len(sd), bool)
    lp[1:] = pn[:-1]; lr[1:] = rn[:-1]; ln[1:] = nn[:-1]
    lp &= same; lr &= same; ln &= same
    u, inv = np.unique(sd, return_inverse=True)
    def cnt(m): return np.bincount(inv, weights=m.astype(np.int64), minlength=len(u))
    cols = [cnt(pn), cnt(pn*bu), cnt(rn), cnt(rn*bu), cnt(nn), cnt(nn*bu),
            cnt(lp), cnt(lp*bu), cnt(lr), cnt(lr*bu), cnt(ln), cnt(ln*bu)]
    M = np.stack(cols, 1).astype(np.int64)
    for k, row in zip(u, M): acc[int(k)] = row
    if i % 40 == 0: print(f"  shard {i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

order = np.array(sorted(acc))
M = np.stack([acc[int(k)] for k in order])
np.savez_compressed("/home/vncuser/.claude/jobs/4efbe660/tmp/a01_prox_full.npz",
                    seed=order, M=M)
print(f"\ndone in {time.time()-t0:.0f}s for {len(order):,} episodes")
nm = ["nP","bP","nR","bR","nN","bN","LnP","LbP","LnR","LbR","LnN","LbN"]
T = M.sum(0)
print("pooled over all steps:")
for lab,(a,b) in [("predator near",(0,1)),("rabbit near  ",(2,3)),("neither      ",(4,5))]:
    print(f"  {lab}: {T[a]:>12,} steps, in bush {100*T[b]/T[a]:5.1f}%")
print("pooled, lagged (state t-1 -> bush at t):")
for lab,(a,b) in [("predator near",(6,7)),("rabbit near  ",(8,9)),("neither      ",(10,11))]:
    print(f"  {lab}: {T[a]:>12,} steps, in bush {100*T[b]/T[a]:5.1f}%")
