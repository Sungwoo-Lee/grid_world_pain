"""Crux: is the injury->hiding gradient independent of predator proximity, or a proxy for it?
Cross-tab dwell by injury bin x predator-near x recent-damage, plus time bin."""
import glob, time, numpy as np, pyarrow.parquet as pq
S = sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]
PREDSL, D = [0,1], 2
ep = pq.read_table(sorted(glob.glob(S+"episodes_*.parquet")), columns=["episode_seed","animal_active"])
o = np.argsort(ep.column("episode_seed").to_numpy())
AACT = np.array(ep.column("animal_active").to_pylist(), bool)[o]
NI, NP, NR = 4, 2, 2
C = np.zeros((NI,NP,NR)); B = np.zeros((NI,NP,NR))
t0 = time.time()
files = sorted(glob.glob(S+"steps_*.parquet"))
def L2(col, w):     # fast fixed-width list column -> (n, w) array
    ch = col.chunks if hasattr(col, "chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False) for c in ch]).reshape(-1, w)
for fi, f in enumerate(files):
    tb = pq.read_table(f, columns=["episode_seed","t","agent_in_bush","injury_level","damage",
                                   "agent_row","agent_col","animal_row","animal_col"])
    sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy()
    ar = tb.column("agent_row").to_numpy(); ac = tb.column("agent_col").to_numpy()
    AR = L2(tb.column("animal_row"), 4); AC = L2(tb.column("animal_col"), 4)
    A = AACT[sd - 1000000]
    near = (np.maximum(np.abs(AR-ar[:,None]), np.abs(AC-ac[:,None])) <= D) & A
    pn = near[:, PREDSL].any(1)
    dmg = tb.column("damage").to_numpy(zero_copy_only=False)
    # recent damage: any damage in the previous 12 steps of the same episode
    st = np.flatnonzero(t == 0); ends = np.append(st[1:], len(t))
    hit = (dmg > 0).astype(np.int64); cs = np.cumsum(hit)
    lo = np.maximum(np.arange(len(t)) - 12, np.repeat(st, ends-st))
    rec = (cs - np.where(lo > 0, cs[lo-1], 0)) > 0
    m = t >= 1
    bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False)[m].astype(float)
    inj = tb.column("injury_level").to_numpy(zero_copy_only=False)[m]
    ib = np.digitize(inj, [1e-9, 25.0, 50.0]); p = pn[m].astype(int); r = rec[m].astype(int)
    k = (ib*NP + p)*NR + r
    C += np.bincount(k, minlength=NI*NP*NR).reshape(NI,NP,NR)
    B += np.bincount(k, weights=bu, minlength=NI*NP*NR).reshape(NI,NP,NR)
    if fi % 50 == 0: print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

IL = ["injury = 0","injury 0-25","injury 25-50","injury >= 50"]
print("\n=== dwell % by injury x predator-near x recent-damage (n in millions) ===")
print(f"{'':16}" + "".join(f"{h:>26}" for h in ["NO predator near","predator NEAR"]))
print(f"{'':16}" + "".join(f"{h:>13}" for h in ["no recent dmg","recent dmg","no recent dmg","recent dmg"]))
for i in range(NI):
    row = f"{IL[i]:16}"
    for p in range(NP):
        for r in range(NR):
            n = C[i,p,r]; row += f"{100*B[i,p,r]/max(n,1):>8.1f}%({n/1e6:>3.1f}M)"
    print(row)
print("\n=== the crux: injury gradient WITHIN each proximity/damage cell ===")
for p, pl in enumerate(["no predator near","predator near   "]):
    for r, rl in enumerate(["no recent damage","recent damage   "]):
        v = [100*B[i,p,r]/max(C[i,p,r],1) for i in range(NI)]
        n = [C[i,p,r] for i in range(NI)]
        ok = n[1] > 1e5 and n[3] > 1e5
        print(f"  {pl} | {rl} : mild(0-25) {v[1]:5.1f}%  severe(>=50) {v[3]:5.1f}%  "
              f"diff {v[3]-v[1]:+6.1f} pp" + ("" if ok else "   [thin]"))
np.savez("/home/vncuser/.claude/jobs/4efbe660/tmp/a01_crux.npz", C=C, B=B)
