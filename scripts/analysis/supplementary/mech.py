"""Mechanism checks: (1) does hiding block eating? (2) injury gradient among INJURED states,
time-controlled and within-episode."""
import glob, time, numpy as np, pyarrow.parquet as pq
S = sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]
FOOD = list(range(4))
n_in = n_out = e_in = e_out = 0.0
# co-location: can a bush tile and an active food tile be the same cell?
co = 0; tot_ep = 0
# within-episode, time-matched: injury 0-25 vs >=50, per time bin
TB = [0,10,25,50,100,200,500]; NT = len(TB)-1
cM = np.zeros(NT); bM = np.zeros(NT); cS = np.zeros(NT); bS = np.zeros(NT)
t0 = time.time()
files = sorted(glob.glob(S+"steps_*.parquet"))
for fi, f in enumerate(files):
    tb = pq.read_table(f, columns=["t","agent_in_bush","ate_food","injury_level",
                                   "agent_row","agent_col","res_row","res_col","res_active"])
    t = tb.column("t").to_numpy(); m = t >= 1
    bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False)[m].astype(bool)
    at = tb.column("ate_food").to_numpy(zero_copy_only=False)[m].astype(np.float64)
    n_in += bu.sum(); n_out += (~bu).sum(); e_in += at[bu].sum(); e_out += at[~bu].sum()
    inj = tb.column("injury_level").to_numpy(zero_copy_only=False)[m]
    tt = t[m]; tbin = np.clip(np.digitize(tt, TB[1:-1]), 0, NT-1)
    mild = (inj > 0) & (inj < 25); sev = inj >= 50
    for M, Bb, sel in ((cM,bM,mild),(cS,bS,sev)):
        M += np.bincount(tbin[sel], minlength=NT)
        Bb += np.bincount(tbin[sel], weights=bu[sel].astype(float), minlength=NT)
    if fi == 0:   # co-location check on one shard's spawn rows
        st = np.flatnonzero(t == 0)
        ar = tb.column("agent_row").to_numpy()[st]
        rr = np.array(tb.column("res_row").take(st).to_pylist(), float)
        print(f"  (res slots per row: {rr.shape[1]})")
    if fi % 50 == 0: print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== does hiding block eating? ===")
print(f"  steps IN a bush     : {n_in:>13,.0f}   ate on {100*e_in/n_in:6.3f}% of them")
print(f"  steps NOT in a bush : {n_out:>13,.0f}   ate on {100*e_out/n_out:6.3f}% of them")
print(f"\n=== injury gradient among INJURED states, time-controlled ===")
print(f"{'step index':14}{'mild (0-25)':>18}{'severe (>=50)':>20}{'difference':>14}")
for i in range(NT):
    a = 100*bM[i]/max(cM[i],1); b = 100*bS[i]/max(cS[i],1)
    print(f"{TB[i]:>4}-{TB[i+1]:<9}{a:>16.1f}% {b:>18.1f}% {b-a:>+13.1f} pp")
a = 100*bM.sum()/cM.sum(); b = 100*bS.sum()/cS.sum()
print(f"{'ALL':14}{a:>16.1f}% {b:>18.1f}% {b-a:>+13.1f} pp")
