"""Does the pain->hiding relationship differ across the 10 models?
The arms differ in healing rate, which directly sets how fast injury (and so felt pain) decays."""
import glob, time, numpy as np, pyarrow.parquet as pq, yaml
from scipy import stats
KL,TAU=12,3.0
k=np.arange(KL,dtype=np.float64); raw=(k/TAU)*np.exp(1.0-k/TAU); KER=raw/raw.sum()
PE=[1e-9,8,18,32,50]
ARMS=[f"a{i:02d}" for i in range(1,11)]
res={}
for arm in ARMS:
    S=sorted(glob.glob(f"results/trajectories/*_{arm}_*/*/*/"))[0]
    n=np.zeros(6); b=np.zeros(6); t0=time.time()
    for f in sorted(glob.glob(S+"steps_*.parquet")):
        tb=pq.read_table(f,columns=["episode_seed","t","agent_in_bush","injury_level"])
        t=tb.column("t").to_numpy(); N=len(t)
        inj=tb.column("injury_level").to_numpy(zero_copy_only=False).astype(np.float64)
        bu=tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.float64)
        st=np.flatnonzero(t==0); ends=np.append(st[1:],N)
        estart=np.repeat(st,ends-st); idx=np.arange(N)
        noci=np.zeros(N)
        for j in range(1,KL):
            src=idx-j; ok=src>estart
            noci[ok]+=KER[j]*inj[src[ok]]
        nprev=np.zeros(N); nprev[1:]=noci[:-1]; nprev[idx==estart]=0.0
        m=t>=2; pb=np.digitize(nprev[m],PE)
        n+=np.bincount(pb,minlength=6); b+=np.bincount(pb,weights=bu[m],minlength=6)
    res[arm]=(n,b); print(f"  {arm} done ({time.time()-t0:.0f}s)",flush=True)
np.savez("/home/vncuser/.claude/jobs/4efbe660/tmp/armpain.npz",**{f"{a}_n":res[a][0] for a in ARMS},
         **{f"{a}_b":res[a][1] for a in ARMS})
def heal(a):
    g=sorted(glob.glob(f"results/JAX_RecurrentPPO/*_rppo_restprem_{a}_*/models/config.yaml"))
    return yaml.safe_load(open(g[0]))["body"]["recovery_accel_rate"]
H=np.array([heal(a) for a in ARMS])
PL=["felt 0","0-8","8-18","18-32","32-50","50+"]
print(f"\n=== hiding % by felt pain, per model (all steps, unconditioned) ===")
print(f"{'arm':>5}{'heal':>6}"+"".join(f"{p:>10}" for p in PL)+f"{'slope':>9}")
slopes=[]
for i,a in enumerate(ARMS):
    n,b=res[a]; v=100*b/np.maximum(n,1)
    sl=v[5]-v[1]
    slopes.append(sl)
    print(f"{a:>5}{H[i]:>6g}"+"".join(f"{x:>10.1f}" for x in v)+f"{sl:>+9.1f}")
r=stats.spearmanr(H,slopes)
print(f"\n  pain->hiding slope (top bin minus 0-8 bin) vs healing rate:"
      f" rho={r.statistic:+.3f}  p={r.pvalue:.3f}")
print(f"  slope range across models: {min(slopes):+.1f} to {max(slopes):+.1f} pp")
