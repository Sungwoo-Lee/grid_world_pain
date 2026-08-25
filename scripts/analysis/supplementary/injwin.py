"""Reviewer point: randomised start-injury decays at 5/step, so it is gone within ~20 steps.
Measure its effect ONLY inside the window where the manipulation is still live, and do it
UNCONDITIONALLY (no conditioning on surviving the window), which is causally valid under
randomisation: rate = E[bush steps in window] / E[steps in window]."""
import glob, time, numpy as np, pyarrow.parquet as pq
S=sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]; SEED0,NEP=1000000,1000000
WINS=[10,20]
n={w:np.zeros(NEP) for w in WINS}; b={w:np.zeros(NEP) for w in WINS}; ii={w:np.zeros(NEP) for w in WINS}
t0=time.time()
for fi,f in enumerate(sorted(glob.glob(S+"steps_*.parquet"))):
    tb=pq.read_table(f,columns=["episode_seed","t","agent_in_bush","injury_level"])
    sd=tb.column("episode_seed").to_numpy(); t=tb.column("t").to_numpy()
    bu=tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(float)
    inj=tb.column("injury_level").to_numpy(zero_copy_only=False).astype(float)
    for w in WINS:
        m=(t>=1)&(t<=w); g=sd[m]-SEED0; lo=g.min()
        n[w][lo:lo+5000]+=np.bincount(g-lo,minlength=5000)
        b[w][lo:lo+5000]+=np.bincount(g-lo,weights=bu[m],minlength=5000)
        ii[w][lo:lo+5000]+=np.bincount(g-lo,weights=inj[m],minlength=5000)
    if fi%50==0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)",flush=True)
T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
B=np.load(T+"a01_passB.npz"); inj0=B["inj0"]; L=B["n_steps"]; Y=B["bush_steps"]
print("\n=== effect of RANDOMISED start injury, unconditional, by window ===")
print("(no conditioning on survival; rate = total bush steps / total steps in window)")
hdr=f"{'start injury':14}{'episodes':>10}"+"".join(f"{'dwell 1-'+str(w):>13}{'mean inj':>10}" for w in WINS)+f"{'dwell whole-ep':>16}"
print(hdr)
for lo,hi in [(0,20),(20,40),(40,60),(60,80),(80,101)]:
    m=(inj0>=lo)&(inj0<hi); row=f"{lo:>3}-{hi:<10}{m.sum():>10,}"
    for w in WINS:
        row+=f"{100*b[w][m].sum()/n[w][m].sum():>12.1f}%{ii[w][m].sum()/n[w][m].sum():>10.1f}"
    row+=f"{100*Y[m].sum()/L[m].sum():>15.1f}%"
    print(row)
np.savez(T+"a01_injwin.npz", **{f"n{w}":n[w] for w in WINS}, **{f"b{w}":b[w] for w in WINS})
