"""How much does the MODEL matter, compared with the world it is dropped into?

All 10 arms replay the identical 1,000,000 world draws, and both policy and environment are
deterministic given (world, agent). So an outcome Y[world, agent] is a complete 1,000,000 x 10
table with NO noise term, and its variance decomposes exactly into:
    world main effect  +  agent main effect  +  agent x world interaction
"""
import glob, numpy as np, yaml
from scipy import stats
T="/home/vncuser/.claude/jobs/4efbe660/tmp/"
ARMS=[f"a{i:02d}" for i in range(1,11)]
def heal(a):
    g=sorted(glob.glob(f"results/JAX_RecurrentPPO/*_rppo_restprem_{a}_*/models/config.yaml"))
    return yaml.safe_load(open(g[0]))["body"]["recovery_accel_rate"]
H=np.array([heal(a) for a in ARMS])

L=np.zeros((1000000,10)); Yb=np.zeros((1000000,10)); AT=np.zeros((1000000,10))
EAT=np.zeros((1000000,10)); RST=np.zeros((1000000,10))
for j,a in enumerate(ARMS):
    B=np.load(T+a+"_passB.npz"); A=np.load(T+a+"_passA.npz",allow_pickle=True)
    L[:,j]=B["n_steps"]; Yb[:,j]=B["bush_steps"]; AT[:,j]=A["term"].astype(int)
    EAT[:,j]=B["n_ate"]; RST[:,j]=B["n_rest"]
D=Yb/np.maximum(L,1)                       # per-episode dwell fraction

def decompose(Y,name,unit=""):
    gm=Y.mean()
    wm=Y.mean(1); am=Y.mean(0)             # world main, agent main
    v_w=wm.var(); v_a=am.var()
    resid=Y-wm[:,None]-am[None,:]+gm
    v_i=resid.var(); v_t=Y.var()
    print(f"\n=== {name} ===")
    print(f"  grand mean {gm:.4f}{unit}   total variance {v_t:.4f}")
    print(f"  world  (which environment was drawn) {100*v_w/v_t:>6.2f}%   sd {np.sqrt(v_w):.3f}")
    print(f"  AGENT  (which of the 10 models)      {100*v_a/v_t:>6.2f}%   sd {np.sqrt(v_a):.4f}")
    print(f"  agent x world interaction            {100*v_i/v_t:>6.2f}%   sd {np.sqrt(v_i):.3f}")
    print(f"  agent means: " + "  ".join(f"{x:.3f}" for x in am))
    print(f"  spread across agents: {am.max()-am.min():.4f}{unit}"
          f"   (world sd is {np.sqrt(v_w)/max(am.max()-am.min(),1e-9):.1f}x that spread)")
    r=stats.spearmanr(H,am)
    print(f"  trend vs healing parameter: Spearman rho={r.statistic:+.3f}  p={r.pvalue:.3f}")
    return am

print("="*78); print("  MODEL EFFECT — 10 agents x 1,000,000 identical worlds"); print("="*78)
print(f"healing accel by arm: " + "  ".join(f"{a}={h:g}" for a,h in zip(ARMS,H)))
am_d=decompose(D,"bush-dwell fraction per episode")
am_l=decompose(L,"survival steps per episode"," steps")

print("\n=== paired agent-vs-agent differences (same worlds, so extremely precise) ===")
print("  vs a01 (the zero-premium anchor):")
for j,a in enumerate(ARMS):
    if j==0: continue
    d=D[:,j]-D[:,0]; se=d.std(ddof=1)/np.sqrt(len(d))
    dl=L[:,j]-L[:,0]; sel=dl.std(ddof=1)/np.sqrt(len(dl))
    print(f"   {a} heal={H[j]:<4g} dwell {100*d.mean():+7.3f} pp [{100*(d.mean()-1.96*se):+.3f},"
          f"{100*(d.mean()+1.96*se):+.3f}]   survival {dl.mean():+7.2f} steps [{dl.mean()-1.96*sel:+.2f},"
          f"{dl.mean()+1.96*sel:+.2f}]")

print("\n=== is the agent ranking stable across environments? ===")
A0=np.load(T+"a01_passA.npz",allow_pickle=True)
for lab,sel in [("0 predators",A0["n_pred"]==0),("1 predator",A0["n_pred"]==1),
                ("2 predators",A0["n_pred"]==2),
                ("keen predator (detect>=6)",np.nan_to_num(A0["pred_detect"],nan=0)>=6),
                ("many bushes (>=9)",A0["n_bush"]>=9)]:
    m=D[sel].mean(0); order=np.argsort(-m)
    print(f"  {lab:26} best->worst: " + " ".join(ARMS[i] for i in order))
print(f"  {'ALL episodes':26} best->worst: " + " ".join(ARMS[i] for i in np.argsort(-D.mean(0))))
rho=[stats.spearmanr(D.mean(0), D[sel].mean(0)).statistic for _,sel in
     [("",A0["n_pred"]==0),("",A0["n_pred"]==1),("",A0["n_pred"]==2)]]
print(f"  rank correlation of agent ordering, overall vs by predator count: "
      f"{rho[0]:+.2f} {rho[1]:+.2f} {rho[2]:+.2f}")

print("\n=== death causes by agent (% of episodes) ===")
print(f"{'arm':>5}{'heal':>6}{'killed':>9}{'starved':>9}{'survived':>10}{'dwell%':>9}{'survival':>10}")
for j,a in enumerate(ARMS):
    t=AT[:,j]
    print(f"{a:>5}{H[j]:>6g}{100*np.mean(t==4):>9.2f}{100*np.mean(t==2):>9.2f}"
          f"{100*np.mean(t==1):>10.2f}{100*am_d[j]:>9.2f}{am_l[j]:>10.1f}")
