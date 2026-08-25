"""Decisive test of the false-alarm framing (reviewer's request).
If a predator-smelling rabbit causes a SENSORY false alarm, the extra hiding should
concentrate in the moments when THAT RABBIT is nearby. If instead it just raises general
vigilance, the extra hiding should be spread evenly across all moments."""
import glob, time, numpy as np, pyarrow.parquet as pq
S = sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]
T = "/home/vncuser/.claude/jobs/4efbe660/tmp/"
A = np.load(T+"a01_passA.npz", allow_pickle=True)
P, R, D, SEED0 = [0,1], [2,3], 2, 1000000
ep = pq.read_table(sorted(glob.glob(S+"episodes_*.parquet")), columns=["episode_seed","animal_active"])
o = np.argsort(ep.column("episode_seed").to_numpy())
AACT = np.array(ep.column("animal_active").to_pylist(), bool)[o]
# episode groups: exactly 1 rabbit, split by its randomised smell; predator count held fixed
rp = A["rab_predatorness"]
GRP = np.full(len(rp), -1, np.int8)
sub = (A["n_rab"] == 1) & (A["n_pred"] == 1) & np.isfinite(rp)
GRP[sub & (rp < 0.0)] = 0            # rabbit smells rabbit-like
GRP[sub & (rp >= 0.3)] = 1           # rabbit smells predator-like
# 3 proximity states x 2 groups
C = np.zeros((2,3)); B = np.zeros((2,3))
def L2(col, w):
    ch = col.chunks if hasattr(col,"chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False) for c in ch]).reshape(-1,w)
t0 = time.time()
for fi, f in enumerate(sorted(glob.glob(S+"steps_*.parquet"))):
    tb = pq.read_table(f, columns=["episode_seed","t","agent_in_bush","agent_row","agent_col",
                                   "animal_row","animal_col"])
    sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy()
    g = GRP[sd - SEED0]
    m = (t >= 1) & (g >= 0)
    if not m.any(): continue
    ar = tb.column("agent_row").to_numpy(); ac = tb.column("agent_col").to_numpy()
    AR = L2(tb.column("animal_row"),4); AC = L2(tb.column("animal_col"),4)
    near = (np.maximum(np.abs(AR-ar[:,None]), np.abs(AC-ac[:,None])) <= D) & AACT[sd-SEED0]
    pn = near[:,P].any(1); rn = near[:,R].any(1) & ~pn
    st = np.where(pn, 1, np.where(rn, 2, 0))[m]          # 0 neither, 1 predator, 2 rabbit
    bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False)[m].astype(float)
    k = g[m]*3 + st
    C += np.bincount(k, minlength=6).reshape(2,3)
    B += np.bincount(k, weights=bu, minlength=6).reshape(2,3)
    if fi % 50 == 0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)", flush=True)

SL = ["neither near","predator near","RABBIT near"]
print("\n=== dwell %% by proximity state, split by the rabbit's randomised smell ===")
print("(all episodes have exactly 1 predator and 1 rabbit)")
print(f"{'':16}" + "".join(f"{s:>26}" for s in SL))
for gi, gl in enumerate(["rabbit smells rabbit-like","rabbit smells PREDATOR-like"]):
    row = f"{gl:28}"
    for s in range(3):
        row += f"{100*B[gi,s]/max(C[gi,s],1):>10.1f}% ({C[gi,s]/1e6:>5.2f}M)"
    print(row)
print(f"\n{'difference (pp)':28}" + "".join(
    f"{100*B[1,s]/max(C[1,s],1) - 100*B[0,s]/max(C[0,s],1):>+10.1f}    {'':9}" for s in range(3)))
print(f"\n{'share of time in state':28}" + "".join(
    f"{100*C[1,s]/C[1].sum():>9.1f}% vs {100*C[0,s]/C[0].sum():>5.1f}%" for s in range(3)))
np.savez(T+"a01_falsealarm.npz", C=C, B=B)
