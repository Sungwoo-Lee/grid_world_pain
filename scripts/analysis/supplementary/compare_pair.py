"""FiLM-modulated vs unmodulated agent. Same world, same 1,000,000 world draws, same seed 42,
same 45,700,001-step checkpoint. The two runs' environment / sensory / body / perceptual-noise
/ training configs are byte-identical; only the agent's 7-key modulation block differs."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
U,M=sys.argv[1],sys.argv[2]
du=dict(np.load(U,allow_pickle=True)); dm=dict(np.load(M,allow_pickle=True))
assert np.array_equal(du["seed"],dm["seed"]), "not seed-paired"
for k in ["n_pred","n_rab","n_bush","n_rock","n_food","n_ambush","inj0","nut0",
          "pred_detect","pred_predatorness","rab_predatorness","d_bush0"]:
    assert np.allclose(np.nan_to_num(du[k],nan=-9),np.nan_to_num(dm[k],nan=-9)), f"world {k} differs"
print("paired design confirmed: identical worlds, identical starting bodies\n")
def summ(d,n):
    L,Y,t=d["n_steps"],d["bush_steps"],d["term"].astype(int)
    print(f"  {n:14} dwell {100*Y.sum()/L.sum():6.2f}%  survival {L.mean():7.2f}  "
          f"killed {100*np.mean(t==4):5.2f}%  starved {100*np.mean(t==2):5.2f}%  "
          f"timeout {100*np.mean(t==1):5.2f}%  eat/step {d['n_ate'].sum()/L.sum():.4f}")
    return L,Y
print("=== headline ===")
Lu,Yu=summ(du,"unmodulated"); Lm,Ym=summ(dm,"FiLM")
Du,Dm=Yu/np.maximum(Lu,1),Ym/np.maximum(Lm,1)
print()
for nm,a,b in [("dwell fraction",Du,Dm),("survival steps",Lu.astype(float),Lm.astype(float))]:
    d=b-a; se=d.std(ddof=1)/np.sqrt(len(d))
    print(f"  paired FiLM-minus-unmod {nm:15} {d.mean():+9.4f}  95% CI "
          f"[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]  higher in {100*np.mean(d>0):.1f}% of worlds")
print()
for nm,A in [("dwell",np.stack([Du,Dm],1)),("survival",np.stack([Lu,Lm],1).astype(float))]:
    gm=A.mean(); wm=A.mean(1); am=A.mean(0); vt=A.var()
    vi=(A-wm[:,None]-am[None,:]+gm).var()
    print(f"  variance in {nm:9}: world {100*wm.var()/vt:5.2f}%   MODULATOR {100*am.var()/vt:6.3f}%"
          f"   modulator x world {100*vi/vt:5.2f}%")
def build(d):
    return pd.DataFrame({"start_injury":d["inj0"],"start_nutrition":d["nut0"],
      "n_bushes":d["n_bush"],"n_rocks":d["n_rock"],"n_food":d["n_food"],
      "n_ambush_predators":d["n_ambush"],"spawn_dist_to_bush":d["d_bush0"],
      "pred_detection_range":d["pred_detect"],"pred_attack_delay":d["pred_delay"],
      "pred_attack_range":d["pred_range"],"pred_max_stamina":d["pred_stamina"],
      "pred_smell_predatorness":d["pred_predatorness"],
      "rab_smell_predatorness":d["rab_predatorness"],
      "spawn_dist_to_predator":np.where(np.isfinite(d["d_pred0"]),d["d_pred0"],np.nan)})
def glm(d,keep):
    L,Y=d["n_steps"],d["bush_steps"]; X=build(d)
    k=keep&np.isfinite(X.to_numpy()).all(1)
    Xc=sm.add_constant(X[k],has_constant="add")
    m=sm.GLM(np.column_stack([Y[k],(L-Y)[k]]),Xc,family=sm.families.Binomial()).fit()
    disp=m.pearson_chi2/m.df_resid
    pbar=Y[k].sum()/L[k].sum(); s=pbar*(1-pbar)*100
    sd=np.r_[1.0,X[k].std().to_numpy()]
    return (pd.Series(m.params.values*sd*s,index=Xc.columns),
            pd.Series(m.bse.values*np.sqrt(disp)*sd*s,index=Xc.columns), int(k.sum()))
P1=du["n_pred"]==1; R1=du["n_rab"]==1
cu,su,n=glm(du,P1&R1); cm,sm_,_=glm(dm,P1&R1)
print(f"\n=== what each agent responds to (Δpp per SD), n={n:,} 1-predator 1-rabbit episodes ===")
print(f"{'factor':26}{'unmod':>10}{'FiLM':>10}{'diff':>10}{'z':>8}")
for t in cu.index:
    if t=="const": continue
    dd=cm[t]-cu[t]; z=dd/np.sqrt(su[t]**2+sm_[t]**2)
    print(f"{t:26}{cu[t]:>+10.2f}{cm[t]:>+10.2f}{dd:>+10.2f}{z:>8.1f}{'  *' if abs(z)>3 else ''}")
print("\n=== injury-dependent hiding ===")
print(f"{'':16}{'inj=0':>10}{'0-25':>10}{'25-50':>10}{'>=50':>10}{'severe-mild':>13}")
for n2,d in [("unmodulated",du),("FiLM",dm)]:
    v=[100*d["IBb"][:,j].sum()/max(d["IB"][:,j].sum(),1) for j in range(4)]
    print(f"  {n2:14}"+"".join(f"{x:>10.1f}" for x in v)+f"{v[3]-v[1]:>+12.1f}")
print("\n=== proximity ===")
for n2,d in [("unmodulated",du),("FiLM",dm)]:
    L=np.maximum(d["n_steps"],1).sum()
    print(f"  {n2:14} predator-near {100*d['n_pred_near'].sum()/L:5.2f}% of steps   "
          f"rabbit-near {100*d['n_rab_near'].sum()/L:5.2f}%")
