"""Cross-arm stability: same multivariate GLM in all 10 agents. Identical world draws
across arms (same seeds), so this is a paired comparison of agents, not of environments."""
import glob, numpy as np, pandas as pd, statsmodels.api as sm, json
from scipy import stats
T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
ARMS=[f"a{i:02d}" for i in range(1,11)]
import yaml
def _heal(arm):                      # read the real value; never assume a ramp
    g=sorted(glob.glob(f"results/JAX_RecurrentPPO/*_rppo_restprem_{arm}_*/models/config.yaml"))
    return yaml.safe_load(open(g[0]))["body"]["recovery_accel_rate"]
HEAL={a:_heal(a) for a in ARMS}
rows=[]; dwell={}
for arm in ARMS:
    try:
        A=np.load(T+f"{arm}_passA.npz", allow_pickle=True); B=np.load(T+f"{arm}_passB.npz")
    except FileNotFoundError: print("missing", arm); continue
    L=B["n_steps"]; Y=B["bush_steps"]; ONEP=A["n_pred"]==1; ONER=A["n_rab"]==1
    X=pd.DataFrame({
      "start_injury":B["inj0"],"start_nutrition":B["nut0"],
      "n_bushes":A["n_bush"],"n_rocks":A["n_rock"],"n_food":A["n_food"],
      "n_ambush_predators":A["n_ambush"],"spawn_dist_to_bush":B["d_bush0"],
      "pred_detection_range":A["pred_detect"],"pred_attack_delay":A["pred_delay"],
      "pred_attack_range":A["pred_range"],"pred_max_stamina":A["pred_stamina"],
      "pred_smell_predatorness":A["pred_predatorness"],
      "rab_smell_predatorness":A["rab_predatorness"],
      "spawn_dist_to_predator":np.where(B["d_pred0"]>90,np.nan,B["d_pred0"])})
    k=ONEP&ONER&np.isfinite(X.to_numpy()).all(1)
    m=sm.GLM(np.column_stack([Y[k],(L-Y)[k]]), sm.add_constant(X[k],has_constant="add"),
             family=sm.families.Binomial()).fit()
    disp=m.pearson_chi2/m.df_resid; se=m.bse*np.sqrt(disp)
    pbar=Y[k].sum()/L[k].sum(); s=pbar*(1-pbar)*100
    sd=np.r_[1.0, X[k].std().to_numpy()]
    dwell[arm]=dict(overall=100*Y.sum()/L.sum(), survival=float(L.mean()),
                    subset=100*pbar, n=int(k.sum()), heal=HEAL[arm])
    for t_,c_,sd_ in zip(sm.add_constant(X[k],has_constant="add").columns, m.params, sd):
        if t_=="const": continue
        rows.append(dict(arm=arm, heal=HEAL[arm], term=t_, dpp_sd=c_*sd_*s, dpp_unit=c_*s))
    print(f"{arm} heal={HEAL[arm]:.1f} dwell={100*Y.sum()/L.sum():5.2f}% survival={L.mean():6.1f} n={k.sum():,}")
R=pd.DataFrame(rows); R.to_csv(T+"allarms_glm.csv", index=False)
print("\n=== Δpp per SD, by arm (columns = healing acceleration rate) ===")
P=R.pivot(index="term",columns="arm",values="dpp_sd").reindex(
   R.groupby("term").dpp_sd.mean().abs().sort_values(ascending=False).index)
pd.set_option("display.width",200,"display.max_columns",20)
print(P.round(2).to_string())
print("\nsign-stable across all 10 arms:",
      [t for t in P.index if (P.loc[t]>0).all() or (P.loc[t]<0).all()])
print("sign-UNSTABLE:", [t for t in P.index if not((P.loc[t]>0).all() or (P.loc[t]<0).all())])
json.dump(dwell, open(T+"allarms_summary.json","w"), indent=1)
print("\n=== overall dwell + survival by arm ===")
for a,v in dwell.items():
    print(f"  {a} heal={v['heal']:.1f}  dwell {v['overall']:5.2f}%  mean survival {v['survival']:6.1f} steps")
