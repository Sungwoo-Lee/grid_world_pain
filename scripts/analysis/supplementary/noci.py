"""CORRECTION PASS. The agent's interoceptive nociceptor is a lagged, smoothed readout of
injury_level (buffer of injury levels, alpha kernel tau=3 len=12, zeroed at reset), NOT a
damage-event signal. So injury IS perceivable. Redo finding 3 against the signal the agent
actually receives.

(a) per-step decomposition of the randomised start-injury effect, t=1..20, full data
(b) hiding by PERCEIVED nociception, conditioned on predator proximity
"""
import glob, time, numpy as np, pyarrow.parquet as pq
S=sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]; SEED0=1000000
KL, TAU = 12, 3.0
k=np.arange(KL,dtype=np.float64); raw=(k/TAU)*np.exp(1.0-k/TAU); KER=raw/raw.sum()
PRED,D=[0,1],2
ep=pq.read_table(sorted(glob.glob(S+"episodes_*.parquet")),columns=["episode_seed","animal_active"])
o=np.argsort(ep.column("episode_seed").to_numpy())
AACT=np.array(ep.column("animal_active").to_pylist(),bool)[o]
TMAX=20
nT=np.zeros((5,TMAX)); bT=np.zeros((5,TMAX))          # start-injury bin x t
NB=6                                                   # perceived-noci bins
nP=np.zeros((NB,2)); bP=np.zeros((NB,2))               # perceived noci x predator-near
EDGES=[1e-9,10,20,35,55]
def L2(col,w):
    ch=col.chunks if hasattr(col,"chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False) for c in ch]).reshape(-1,w)
t0=time.time()
for fi,f in enumerate(sorted(glob.glob(S+"steps_*.parquet"))):
    tb=pq.read_table(f,columns=["episode_seed","t","agent_in_bush","injury_level",
                                "agent_row","agent_col","animal_row","animal_col"])
    sd=tb.column("episode_seed").to_numpy(); t=tb.column("t").to_numpy()
    inj=tb.column("injury_level").to_numpy(zero_copy_only=False).astype(np.float64)
    bu=tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.float64)
    st=np.flatnonzero(t==0); ends=np.append(st[1:],len(t))
    # reconstruct the perceived signal: sum_j KER[j] * injury[t-j], zero before episode start
    noci=np.zeros(len(t))
    idx=np.arange(len(t)); estart=np.repeat(st,ends-st)
    for j in range(1,KL):
        src=idx-j
        ok=src>estart   # reset row is NEVER written into the buffer (core.py:1112,115)
        noci[ok]+=KER[j]*inj[src[ok]]
    # (a) per-t by start-injury bin
    inj0=inj[st]; g0=np.clip(np.digitize(inj0,[20,40,60,80]),0,4)
    gi=np.repeat(np.arange(len(st)),ends-st)
    m=(t>=1)&(t<=TMAX)
    key=g0[gi[m]]*TMAX+(t[m]-1)
    nT+=np.bincount(key,minlength=5*TMAX).reshape(5,TMAX)
    bT+=np.bincount(key,weights=bu[m],minlength=5*TMAX).reshape(5,TMAX)
    # (b) perceived noci x predator-near
    ar=tb.column("agent_row").to_numpy(); ac=tb.column("agent_col").to_numpy()
    AR=L2(tb.column("animal_row"),4); AC=L2(tb.column("animal_col"),4)
    near=(np.maximum(np.abs(AR-ar[:,None]),np.abs(AC-ac[:,None]))<=D)&AACT[sd-SEED0]
    pn=near[:,PRED].any(1)
    m2=t>=1
    nb=np.digitize(noci[m2],EDGES); p=pn[m2].astype(int)
    key2=nb*2+p
    nP+=np.bincount(key2,minlength=NB*2).reshape(NB,2)
    bP+=np.bincount(key2,weights=bu[m2],minlength=NB*2).reshape(NB,2)
    if fi%50==0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)",flush=True)

print("\n=== (a) randomised start-injury effect, step by step ===")
print("perceived signal for a wound you woke up with: 0.0 at t=1, ~28 by t=4, ~58 by t=10")
print(f"{'t':>3}{'inj 0-20':>11}{'inj 80-100':>12}{'diff':>9}")
for i in range(TMAX):
    a=100*bT[0,i]/max(nT[0,i],1); b=100*bT[4,i]/max(nT[4,i],1)
    print(f"{i+1:>3}{a:>10.1f}%{b:>11.1f}%{b-a:>+8.1f}")
print("\n=== (b) hiding by PERCEIVED nociception (the signal the agent gets) ===")
lab=["0","0-10","10-20","20-35","35-55","55+"]
print(f"{'perceived pain':>16}{'no predator near':>20}{'predator near':>18}")
for i in range(NB):
    a=100*bP[i,0]/max(nP[i,0],1); b=100*bP[i,1]/max(nP[i,1],1)
    print(f"{lab[i]:>16}{a:>13.1f}% ({nP[i,0]/1e6:>4.1f}M){b:>11.1f}% ({nP[i,1]/1e6:>4.1f}M)")
np.savez("/home/vncuser/.claude/jobs/4efbe660/tmp/a01_noci.npz",nT=nT,bT=bT,nP=nP,bP=bP,KER=KER)
