"""Factors the earlier GLMs never tested.

The store audit shows animal_property_sampled varies on channels 1 AND 2 for every animal,
independently. Earlier analyses only used the DIFFERENCE (ch1-ch2, "predator-likeness"), which
throws away a second quantity entirely: the SUM, i.e. how strongly the animal smells at all.
A faint predator and a pungent one have the same predator-likeness.

Also adds, for the 2-predator episodes that all trait analyses previously discarded, the
statistics that actually describe danger there: the KEENEST predator's detection range rather
than the average, and the spread between the two.
"""
import glob, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
A=np.load(T+"a01_passA.npz",allow_pickle=True); B=np.load(T+"a01_passB.npz")
G=np.load(T+"a01_design.npz")     # per-SLOT predator traits
L,Y=B["n_steps"],B["bush_steps"]

def fit(X,keep,title):
    k=keep&np.isfinite(X.to_numpy()).all(1)
    Xc=sm.add_constant(X[k],has_constant="add")
    m=sm.GLM(np.column_stack([Y[k],(L-Y)[k]]),Xc,family=sm.families.Binomial()).fit()
    disp=m.pearson_chi2/m.df_resid; se=m.bse*np.sqrt(disp); z=m.params/se
    p=2*stats.norm.sf(np.abs(z)); pbar=Y[k].sum()/L[k].sum(); s=pbar*(1-pbar)*100
    sd=np.r_[1.0,X[k].std().to_numpy()]
    print(f"\n=== {title} ===\nn = {k.sum():,}   mean bush dwell = {100*pbar:.2f}%")
    print(f"{'factor':30}{'Δpp/SD':>10}{'Δpp/unit':>11}{'z':>9}{'p':>11}")
    for nm,c,sdv,zz,pp in zip(Xc.columns,m.params,sd,z,p):
        if nm=="const": continue
        ps="<1e-300" if pp==0 else f"{pp:.1e}"
        print(f"{nm:30}{c*sdv*s:>+10.2f}{c*s:>+11.3f}{zz:>9.1f}{ps:>11}")

ONEP=A["n_pred"]==1; ONER=A["n_rab"]==1
base={"start_injury":B["inj0"],"start_nutrition":B["nut0"],"bush_count":A["n_bush"],
      "rock_count":A["n_rock"],"food_count":A["n_food"],"hiding_predator_count":A["n_ambush"],
      "spawn_dist_to_bush":B["d_bush0"],
      "detection_range":A["pred_detect"],"attack_delay":A["pred_delay"],
      "attack_range":A["pred_range"],"max_stamina":A["pred_stamina"]}

print("### 1. olfactory channels split apart, and their SUM added")
X=pd.DataFrame({**base,
  "pred_olf_ch1":A["pred_smellA"],"pred_olf_ch2":A["pred_smellB"],
  "rab_olf_ch1":A["rab_smellA"],"rab_olf_ch2":A["rab_smellB"]})
fit(X,ONEP&ONER,"individual olfactory channels (replaces the single difference term)")
X2=pd.DataFrame({**base,
  "pred_olf_identity":A["pred_smellA"]-A["pred_smellB"],
  "pred_olf_intensity":A["pred_smellA"]+A["pred_smellB"],
  "rab_olf_identity":A["rab_smellA"]-A["rab_smellB"],
  "rab_olf_intensity":A["rab_smellA"]+A["rab_smellB"]})
fit(X2,ONEP&ONER,"identity (ch1-ch2) vs INTENSITY (ch1+ch2) — intensity was never tested")

print("\n\n### 2. the 2-predator episodes, previously discarded entirely")
TWO=A["n_pred"]==2
det=G["det"]; adl=G["adly"]; arg=G["arng"]; stm=G["stam"]
mx=np.nanmax(np.where(G["pact"],det,np.nan),axis=1)
mn=np.nanmin(np.where(G["pact"],det,np.nan),axis=1)
X3=pd.DataFrame({"start_injury":B["inj0"],"start_nutrition":B["nut0"],
  "bush_count":A["n_bush"],"food_count":A["n_food"],
  "hiding_predator_count":A["n_ambush"],"spawn_dist_to_bush":B["d_bush0"],
  "detect_KEENEST":mx,"detect_LEAST_KEEN":mn,"detect_spread":mx-mn,
  "attack_delay_mean":A["pred_delay"],"max_stamina_mean":A["pred_stamina"]})
fit(X3,TWO,"2-predator episodes: is it the keenest predator that matters?")
print(f"\n  2-predator episodes: {TWO.sum():,} ({100*TWO.mean():.1f}% of all episodes) —"
      f" previously excluded from every trait model")
