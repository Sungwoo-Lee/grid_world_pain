"""Truncation robustness: recompute the smell / detection effects on a FIXED early window
(steps 1-25) so that differences in episode length cannot drive the dwell rate."""
import glob, time, numpy as np, pyarrow.parquet as pq
S = sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]
SEED0, NEP, W = 1000000, 1000000, 25
bw = np.zeros(NEP); nw = np.zeros(NEP)
t0 = time.time()
for fi, f in enumerate(sorted(glob.glob(S+"steps_*.parquet"))):
    tb = pq.read_table(f, columns=["episode_seed","t","agent_in_bush"])
    sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy()
    bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(float)
    m = (t >= 1) & (t <= W); g = sd[m] - SEED0
    lo = g.min()
    nw[lo:lo+5000] += np.bincount(g-lo, minlength=5000)
    bw[lo:lo+5000] += np.bincount(g-lo, weights=bu[m], minlength=5000)
    if fi % 50 == 0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)", flush=True)

T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
A=np.load(T+"a01_passA.npz",allow_pickle=True); B=np.load(T+"a01_passB.npz")
full_L, full_Y = B["n_steps"], B["bush_steps"]
def tab(name, v, bins, sub):
    k = sub & np.isfinite(v) & (nw >= W)     # survived the whole window
    g = np.clip(np.digitize(v[k], bins[1:-1]), 0, len(bins)-2)
    print(f"\n{name}   (episodes surviving all {W} steps: {100*k.sum()/max((sub&np.isfinite(v)).sum(),1):.1f}%)")
    print(f"{'bin':12}{'n':>9}{'dwell steps1-25':>18}{'dwell whole-ep':>17}{'mean survival':>15}")
    for i in range(len(bins)-1):
        m = g == i; idx = np.flatnonzero(k)[m]
        d_w = 100*bw[idx].sum()/nw[idx].sum()
        d_f = 100*full_Y[idx].sum()/full_L[idx].sum()
        print(f"{bins[i]:g}..{bins[i+1]:g}".ljust(12)+f"{m.sum():>9,}{d_w:>17.1f}%{d_f:>16.1f}%"
              f"{full_L[idx].mean():>15.1f}")
sb=[-1,-0.2,0,0.2,0.4,0.6,1.01]
tab("PREDATOR smell predator-likeness", A["pred_predatorness"], sb, A["n_pred"]==1)
tab("RABBIT smell predator-likeness",   A["rab_predatorness"],  sb, A["n_rab"]==1)
tab("predator detection range", A["pred_detect"], [1,2,3,4,5,6,7,8], A["n_pred"]==1)
tab("randomised START injury", B["inj0"], [0,20,40,60,80,101], np.ones(NEP,bool))
tab("randomised START nutrition", B["nut0"], [0,20,40,60,80,101], np.ones(NEP,bool))
np.savez(T+"a01_window.npz", bw=bw, nw=nw)
