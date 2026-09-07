#!/usr/bin/env python
"""Stacked-row history figures from metrics_history-style CSVs (all 11 measures available).

Promoted as-is from tmp/plot_metrics_summary.py (part of the dwell-history pipeline;
see README.md). `fig_for` is also imported directly by run_sweep.py's plotting step.

CLI usage: plot_summary.py <run_dir> <run_label> [measure1 measure2 ...]
Default measures: bush_hiding spatial_spread. Any of the 11 measure columns works.
Env knobs (unchanged from the tmp original): XDIV, XLABEL, XBOUNDARY.
"""
import sys, glob, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
XDIV=float(os.environ.get("XDIV","1e6")); XLABEL=os.environ.get("XLABEL","training  (million steps)")
XBND=os.environ.get("XBOUNDARY","10")  # empty disables the boundary line

NICE={"none":"No animal","pred":"Predator","rabbit":"Rabbit · chase",
      "rabbit_olfzero":"Rabbit · chase\nolf-zeroed","rabbitwander":"Rabbit · wander",
      "rabbitwander_predsmell":"Rabbit · wander\npred-smell"}
ORDER=["none","pred","rabbit","rabbit_olfzero","rabbitwander","rabbitwander_predsmell"]
# measure -> (nice ylabel, is_percent, subtitle)
MINFO={
 "bush_hiding":("time in bush (%)",True,"% of the 100-step episode spent sitting on the bush"),
 "bush_use_rate":("entered bush (%)",True,"fraction of episodes the agent ever stepped on the bush"),
 "spatial_spread":("spatial spread (R_g)",False,"radius of gyration of the path, grid cells (low=parked, high=roaming)"),
 "survival_steps":("survival (steps)",False,"steps survived, capped at 100"),
 "fid":("flight-init distance",False,"distance to animal when the agent first moves"),
 "time_near_animal":("time near animal (%)",True,"fraction of steps within 1 cell of the animal"),
 "closest_approach":("closest approach",False,"minimum distance to the animal"),
 "time_moving":("time moving (%)",True,"fraction of steps the agent changed cell"),
 "pursuit_duration":("longest chase (steps)",False,"longest unbroken run within 2 cells of the animal"),
 "injury_change":("injury change",False,"end injury minus start injury"),
 "bush_entry_step":("steps to reach bush",False,"first step the agent entered the bush"),
}
def cond_key(c): return c[len("avoid_"):].rsplit("_inj",1)[0]
def inj(c): return "inj "+c.rsplit("_inj",1)[1]
def load(f,m):
    xs=[]; ys=[]
    for r in csv.DictReader(open(f)):
        xs.append(float(r["step"])/XDIV)
        v=r.get(m,""); ys.append(float(v) if v not in ("","nan") else np.nan)
    o=np.argsort(xs); return np.array(xs)[o], np.array(ys)[o]
def roll(y,w):
    n=len(y);
    if n<3: return y
    w=max(3,(w|1)); h=w//2; out=np.empty(n)
    for i in range(n):
        a=max(0,i-h); b=min(n,i+h+1); seg=y[a:b]; out[i]=np.nanmean(seg) if np.any(~np.isnan(seg)) else np.nan
    return out
def fig_for(level_dir, level_label, m):
    files=sorted(glob.glob(f"{level_dir}/avoid_*.csv"))
    conds=[os.path.basename(f)[:-4] for f in files]
    conds=sorted(conds, key=lambda c:(ORDER.index(cond_key(c)) if cond_key(c) in ORDER else 99, c))
    if not conds: return None
    ylabel,pct,sub=MINFO.get(m,(m,False,m))
    data={c:load(f"{level_dir}/{c}.csv",m) for c in conds}
    scale=100.0 if pct else 1.0
    n=len(conds); xmax=max(d[0].max() for d in data.values() if len(d[0]))
    fig,ax=plt.subplots(n,1,figsize=(13,1.05*n+1.1),sharex=True)
    if n==1: ax=[ax]
    band={k:i for i,k in enumerate(ORDER)}
    raw_c="#5a9367" if m in("bush_hiding","bush_use_rate") else "#c98a5e"
    trend_c="#14532d" if m in("bush_hiding","bush_use_rate") else "#7c3a12"
    for i,c in enumerate(conds):
        xs,y=data[c]; y=y*scale; A=ax[i]
        if band.get(cond_key(c),0)%2==0: A.set_facecolor("#f5f6f7")
        A.fill_between(xs,y,alpha=.14,color=raw_c,lw=0)
        A.plot(xs,y,lw=.5,color=raw_c,alpha=.5)
        A.plot(xs,roll(y,max(5,n//3)),lw=1.7,color=trend_c)
        if XBND and xmax>float(XBND): A.axvline(float(XBND),ls=(0,(2,2)),color="#888",lw=.7,zorder=0)
        A.set_ylabel(f"{NICE.get(cond_key(c),cond_key(c))}\n{inj(c)}",rotation=0,ha="right",va="center",fontsize=7.5,linespacing=1.05)
        if pct: A.set_ylim(-4,104)
        for s in ("top","right"): A.spines[s].set_visible(False)
        A.tick_params(labelsize=7.5); A.grid(axis="y",alpha=.25,lw=.5); A.margins(x=.005)
    ax[-1].set_xlabel(XLABEL,fontsize=10)
    fig.legend(handles=[plt.Line2D([],[],color=trend_c,lw=1.7,label="rolling trend"),
                        plt.Line2D([],[],color=raw_c,lw=.8,alpha=.6,label="raw (per checkpoint)")],
               loc="upper right",fontsize=8,frameon=False,bbox_to_anchor=(.995,.99))
    fig.suptitle(f"{level_label} — {ylabel}",x=.5,y=.996,fontsize=13,fontweight="bold")
    fig.text(.5,.972,sub,ha="center",fontsize=8,color="#555")
    fig.tight_layout(rect=(0.01,0,1,0.965))
    out=f"{level_dir}/FIG_{m}.png"; fig.savefig(out,dpi=140); plt.close(fig)
    return out
if __name__=="__main__":
    level_dir, level_label = sys.argv[1], sys.argv[2]
    measures = sys.argv[3:] or ["bush_hiding","spatial_spread"]
    for m in measures:
        o=fig_for(level_dir, level_label, m)
        print(f"  {level_label} {m} -> {o}")
