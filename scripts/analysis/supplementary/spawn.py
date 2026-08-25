"""Two remaining exogenous regressors: distance at spawn to the nearest food, and how far
from the centre of the map the agent wakes up. Both are set by the world before it acts."""
import glob, time, numpy as np, pyarrow.parquet as pq, pandas as pd, statsmodels.api as sm
from scipy import stats
S=sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]; SEED0,NEP=1000000,1000000
FOOD=list(range(4))
ep=pq.read_table(sorted(glob.glob(S+"episodes_*.parquet")),columns=["episode_seed","res_allocated"])
o=np.argsort(ep.column("episode_seed").to_numpy())
RA=np.array(ep.column("res_allocated").to_pylist(),bool)[o]
d_food=np.zeros(NEP); t0=time.time()
def L2(col,w):
    ch=col.chunks if hasattr(col,"chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False) for c in ch]).reshape(-1,w)
for fi,f in enumerate(sorted(glob.glob(S+"steps_*.parquet"))):
    tb=pq.read_table(f,columns=["episode_seed","t","agent_row","agent_col","res_row","res_col"])
    t=tb.column("t").to_numpy(); st=np.flatnonzero(t==0)
    g=tb.column("episode_seed").to_numpy()[st]-SEED0
    ar=tb.column("agent_row").to_numpy()[st][:,None]; ac=tb.column("agent_col").to_numpy()[st][:,None]
    RR=L2(tb.column("res_row"),16)[st][:,FOOD]; RC=L2(tb.column("res_col"),16)[st][:,FOOD]
    al=RA[g][:,FOOD]
    d=np.where(al,np.maximum(np.abs(RR-ar),np.abs(RC-ac)),99.)
    d_food[g]=d.min(1)
    if fi%50==0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)",flush=True)
T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
A=np.load(T+"a01_passA.npz",allow_pickle=True); B=np.load(T+"a01_passB.npz")
L,Y=B["n_steps"],B["bush_steps"]
d_cent=np.maximum(np.abs(B["arow0"]-5.5),np.abs(B["acol0"]-5.5))
np.savez(T+"a01_spawn.npz",d_food=d_food,d_cent=d_cent)
def uni(nm,v):
    k=np.isfinite(v)&(v<90)
    X=sm.add_constant(pd.DataFrame({nm:v[k]}),has_constant="add")
    m=sm.GLM(np.column_stack([Y[k],(L-Y)[k]]),X,family=sm.families.Binomial()).fit()
    disp=m.pearson_chi2/m.df_resid; pbar=Y[k].sum()/L[k].sum(); s=pbar*(1-pbar)*100
    print(f"  {nm:26}{m.params[1]*v[k].std()*s:>+9.2f} pp/SD{m.params[1]*s:>+9.3f} pp/unit  n={k.sum():,}")
print("\n=== two more exogenous regressors ===")
uni("spawn_dist_to_food",d_food); uni("spawn_dist_from_centre",d_cent)
for nm,v,bins in [("spawn dist to nearest food",d_food,[0,1,2,3,4,5,10]),
                  ("spawn dist from centre",d_cent,[0,1,2,3,4,5])]:
    k=np.isfinite(v)&(v<90); g=np.clip(np.digitize(v[k],bins[1:-1]),0,len(bins)-2)
    print(f"\n{nm}")
    print("  level   "+"".join(f"{bins[i]:g}-{bins[i+1]:g}".rjust(9) for i in range(len(bins)-1)))
    print("  dwell%  "+"".join(f"{100*Y[k][g==i].sum()/L[k][g==i].sum():>9.1f}" for i in range(len(bins)-1)))
    print("  survive "+"".join(f"{L[k][g==i].mean():>9.1f}" for i in range(len(bins)-1)))
