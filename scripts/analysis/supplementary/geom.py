"""Spatial geometry of the forage-vs-hide trade-off, measured at spawn.
Never tested: how far the FOOD sits from the nearest BUSH. If food and cover coincide the
agent can do both; if they are far apart it must choose. That is the trade-off made physical."""
import glob, time, numpy as np, pyarrow.parquet as pq
S=sorted(glob.glob('results/trajectories/*_a01_*/*/*/'))[0]; SEED0,NEP=1000000,1000000
FOOD=list(range(4)); BUSH=list(range(10))
ep=pq.read_table(sorted(glob.glob(S+'episodes_*.parquet')),columns=["episode_seed","res_allocated","obs_active"])
o=np.argsort(ep.column("episode_seed").to_numpy())
RA=np.array(ep.column("res_allocated").to_pylist(),bool)[o]
OA=np.array(ep.column("obs_active").to_pylist(),bool)[o]
d_fb=np.zeros(NEP); spawn_on_bush=np.zeros(NEP); d_pred_bush=np.zeros(NEP)
def L2(c,w):
    ch=c.chunks if hasattr(c,"chunks") else [c]
    return np.concatenate([x.flatten().to_numpy(zero_copy_only=False) for x in ch]).reshape(-1,w)
t0=time.time()
for fi,f in enumerate(sorted(glob.glob(S+'steps_*.parquet'))):
    tb=pq.read_table(f,columns=["episode_seed","t","agent_in_bush","res_row","res_col",
                                "obs_row","obs_col","animal_row","animal_col"])
    t=tb.column("t").to_numpy(); st=np.flatnonzero(t==0)
    g=tb.column("episode_seed").to_numpy()[st]-SEED0
    RR=L2(tb.column("res_row"),16)[st][:,FOOD]; RC=L2(tb.column("res_col"),16)[st][:,FOOD]
    OR=L2(tb.column("obs_row"),22)[st][:,BUSH]; OC=L2(tb.column("obs_col"),22)[st][:,BUSH]
    fa=RA[g][:,FOOD]; ba=OA[g][:,BUSH]
    # chebyshev distance from every allocated food to every active bush
    dr=np.abs(RR[:,:,None]-OR[:,None,:]); dc=np.abs(RC[:,:,None]-OC[:,None,:])
    dist=np.maximum(dr,dc).astype(float)
    dist[~(fa[:,:,None]&ba[:,None,:])]=np.inf
    per_food=dist.min(2)                      # nearest bush for each food
    per_food[~fa]=np.nan
    d_fb[g]=np.nanmean(per_food,axis=1)
    spawn_on_bush[g]=tb.column("agent_in_bush").to_numpy(zero_copy_only=False)[st]
    if fi%50==0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)",flush=True)
np.savez("/home/vncuser/.claude/jobs/4efbe660/tmp/a01_geom.npz",
         d_food_to_bush=d_fb, spawn_on_bush=spawn_on_bush)
import pandas as pd, statsmodels.api as sm
from scipy import stats
T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
A=np.load(T+"a01_passA.npz",allow_pickle=True); B=np.load(T+"a01_passB.npz")
L,Y=B["n_steps"],B["bush_steps"]
X=pd.DataFrame({"start_injury":B["inj0"],"start_nutrition":B["nut0"],"bush_count":A["n_bush"],
  "food_count":A["n_food"],"hiding_predator_count":A["n_ambush"],"predator_count":A["n_pred"],
  "spawn_dist_to_bush":B["d_bush0"],
  "food_to_bush_distance":d_fb,"spawns_inside_a_bush":spawn_on_bush})
k=np.isfinite(X.to_numpy()).all(1)
Xc=sm.add_constant(X[k],has_constant="add")
m=sm.GLM(np.column_stack([Y[k],(L-Y)[k]]),Xc,family=sm.families.Binomial()).fit()
disp=m.pearson_chi2/m.df_resid; se=m.bse*np.sqrt(disp); z=m.params/se
pbar=Y[k].sum()/L[k].sum(); s=pbar*(1-pbar)*100; sd=np.r_[1.0,X[k].std().to_numpy()]
print(f"\n=== spawn geometry (n={k.sum():,}) ===")
print(f"{'factor':26}{'Δpp/SD':>10}{'Δpp/unit':>11}{'z':>9}")
for nm,c,sdv,zz in zip(Xc.columns,m.params,sd,z):
    if nm=="const": continue
    print(f"{nm:26}{c*sdv*s:>+10.2f}{c*s:>+11.3f}{zz:>9.1f}")
print(f"\n  food-to-bush distance: mean {np.nanmean(d_fb):.2f} tiles, range "
      f"{np.nanmin(d_fb):.1f}-{np.nanmax(d_fb):.1f}")
print(f"  agent spawns inside a bush in {100*spawn_on_bush.mean():.1f}% of episodes")
