"""Dose-response curves for the report: dwell + survival vs each randomised factor."""
import numpy as np, pandas as pd, json
T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
A=np.load(T+"a01_passA.npz", allow_pickle=True); B=np.load(T+"a01_passB.npz")
L=B["n_steps"]; Y=B["bush_steps"]
out={}
def curve(name, v, bins=None, keep=None):
    k = np.isfinite(v) if keep is None else (keep & np.isfinite(v))
    x, y, l = v[k], Y[k], L[k]
    if bins is None:
        u = np.unique(x); g = np.searchsorted(u, x); lab=[f"{q:g}" for q in u]
    else:
        g = np.clip(np.digitize(x, bins[1:-1]),0,len(bins)-2)
        lab=[f"{bins[i]:g}-{bins[i+1]:g}" for i in range(len(bins)-1)]
    n=len(lab)
    dw = np.bincount(g,weights=y,minlength=n)/np.maximum(np.bincount(g,weights=l,minlength=n),1)
    sv = np.bincount(g,weights=l,minlength=n)/np.maximum(np.bincount(g,minlength=n),1)
    ct = np.bincount(g,minlength=n)
    out[name]={"labels":lab,"dwell":(100*dw).round(2).tolist(),
               "survival":sv.round(1).tolist(),"n":ct.tolist()}
    print(f"\n{name}")
    print("  level   " + "".join(f"{s:>10}" for s in lab))
    print("  dwell%  " + "".join(f"{q:>10.1f}" for q in 100*dw))
    print("  survive " + "".join(f"{q:>10.1f}" for q in sv))
ONEP = A["n_pred"]==1; ONER = A["n_rab"]==1
curve("predator detection range", A["pred_detect"], keep=ONEP)
curve("predator count", A["n_pred"])
curve("rabbit count", A["n_rab"])
curve("bushes available", A["n_bush"])
curve("rocks", A["n_rock"])
curve("food items", A["n_food"])
curve("ambush predators", A["n_ambush"])
curve("predator attack delay", A["pred_delay"], keep=ONEP)
curve("predator attack range", A["pred_range"], keep=ONEP)
curve("predator max stamina", A["pred_stamina"], bins=[30,50,70,90,110,130,151], keep=ONEP)
curve("spawn distance to nearest bush", B["d_bush0"], bins=[0,1,2,3,4,5,10])
curve("randomised START injury", B["inj0"], bins=[0,20,40,60,80,101])
curve("randomised START nutrition", B["nut0"], bins=[0,20,40,60,80,101])
curve("predator smell 'predator-likeness'", A["pred_predatorness"], bins=[-1,-0.2,0,0.2,0.4,0.6,1.01], keep=ONEP)
curve("RABBIT smell 'predator-likeness'", A["rab_predatorness"], bins=[-1,-0.2,0,0.2,0.4,0.6,1.01], keep=ONER)
json.dump(out, open(T+"a01_curves.json","w"), indent=1)
print("\nsaved", T+"a01_curves.json")
