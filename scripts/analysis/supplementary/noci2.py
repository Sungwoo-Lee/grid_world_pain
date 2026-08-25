"""Deep dive on interoceptive nociception and hiding. Step level.

T1 dissociation : at matched TRUE injury, does PERCEIVED pain still move hiding (and vice
                  versa)? The kernel makes them come apart, which identifies which one is used.
T2 peri-event   : hiding aligned on a damage event, t-10 .. t+25.
T3 lag profile  : hiding at t as a function of damage k steps ago -> the agent's effective
                  response kernel, comparable with the true alpha kernel.
T4 direction    : at matched perceived pain, does rising vs falling pain differ?
"""
import glob, time, numpy as np, pyarrow.parquet as pq
S=sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]; SEED0=1000000
KL,TAU=12,3.0
k=np.arange(KL,dtype=np.float64); raw=(k/TAU)*np.exp(1.0-k/TAU); KER=raw/raw.sum()
PRED,D=[0,1],2
ep=pq.read_table(sorted(glob.glob(S+"episodes_*.parquet")),columns=["episode_seed","animal_active"])
o=np.argsort(ep.column("episode_seed").to_numpy())
AACT=np.array(ep.column("animal_active").to_pylist(),bool)[o]

IE=[1e-9,10,25,45,70]           # true injury bins (5)
PE=[1e-9,8,18,32,50]            # perceived pain bins (5)
NBI=NBP=6
n2=np.zeros((NBI,NBP)); b2=np.zeros((NBI,NBP))
LAGS=np.arange(-10,26); nE=np.zeros((len(LAGS),2)); bE=np.zeros((len(LAGS),2))
nL=np.zeros(16); bL=np.zeros(16); nL0=np.zeros(16); bL0=np.zeros(16)
nD=np.zeros((NBP,3)); bD=np.zeros((NBP,3))
def L2(col,w):
    ch=col.chunks if hasattr(col,"chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False) for c in ch]).reshape(-1,w)
t0=time.time()
for fi,f in enumerate(sorted(glob.glob(S+"steps_*.parquet"))):
    tb=pq.read_table(f,columns=["episode_seed","t","agent_in_bush","injury_level","damage",
                                "agent_row","agent_col","animal_row","animal_col"])
    sd=tb.column("episode_seed").to_numpy(); t=tb.column("t").to_numpy()
    inj=tb.column("injury_level").to_numpy(zero_copy_only=False).astype(np.float64)
    bu=tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.float64)
    dmg=tb.column("damage").to_numpy(zero_copy_only=False).astype(np.float64)
    N=len(t); st=np.flatnonzero(t==0); ends=np.append(st[1:],N)
    estart=np.repeat(st,ends-st); eend=np.repeat(ends,ends-st)
    idx=np.arange(N)
    noci=np.zeros(N)
    for j in range(1,KL):
        src=idx-j; ok=src>=estart
        noci[ok]+=KER[j]*inj[src[ok]]
    ar=tb.column("agent_row").to_numpy(); ac=tb.column("agent_col").to_numpy()
    AR=L2(tb.column("animal_row"),4); AC=L2(tb.column("animal_col"),4)
    pn=((np.maximum(np.abs(AR-ar[:,None]),np.abs(AC-ac[:,None]))<=D)&AACT[sd-SEED0])[:,PRED].any(1)
    m=(t>=1)&(~pn)                                   # no predator within 2 tiles
    # T1 dissociation
    ib=np.digitize(inj[m],IE); pb=np.digitize(noci[m],PE)
    kk=ib*NBP+pb
    n2+=np.bincount(kk,minlength=NBI*NBP).reshape(NBI,NBP)
    b2+=np.bincount(kk,weights=bu[m],minlength=NBI*NBP).reshape(NBI,NBP)
    # T4 direction of change, at matched perceived level
    dn=np.zeros(N); dn[1:]=noci[1:]-noci[:-1]; dn[idx==estart]=0
    dirc=np.where(dn>1e-9,2,np.where(dn<-1e-9,0,1))
    kk4=pb*3+dirc[m]
    nD+=np.bincount(kk4,minlength=NBP*3).reshape(NBP,3)
    bD+=np.bincount(kk4,weights=bu[m],minlength=NBP*3).reshape(NBP,3)
    # T2 peri-event
    ev=np.flatnonzero((dmg>0)&(t>=1))
    if len(ev):
        clean=np.ones(len(ev),bool)                 # no other damage event within +-10
        prev=np.r_[-10**9,ev[:-1]]; nxt=np.r_[ev[1:],10**9]
        clean=(ev-prev>10)&(nxt-ev>25)
        for li,lag in enumerate(LAGS):
            tgt=ev+lag
            ok=(tgt>=estart[ev])&(tgt<eend[ev])
            for c in (0,1):
                sel=ok&(clean if c else np.ones(len(ev),bool))
                if sel.any():
                    nE[li,c]+=sel.sum(); bE[li,c]+=bu[tgt[sel]].sum()
    # T3 lag profile: hiding at t vs damage at t-k
    for kk3 in range(16):
        src=idx-kk3; ok=(src>=estart)&(t>=1)&(~pn)
        hit=np.zeros(N,bool); hit[ok]=dmg[src[ok]]>0
        nL[kk3]+=hit.sum(); bL[kk3]+=bu[hit].sum()
        base=ok&~hit
        nL0[kk3]+=base.sum(); bL0[kk3]+=bu[base].sum()
    if fi%50==0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)",flush=True)

np.savez("/home/vncuser/.claude/jobs/4efbe660/tmp/a01_noci2.npz",
         n2=n2,b2=b2,nE=nE,bE=bE,nL=nL,bL=bL,nL0=nL0,bL0=bL0,nD=nD,bD=bD,LAGS=LAGS,KER=KER)
IL=["0","0-10","10-25","25-45","45-70","70+"]
PL=["0","0-8","8-18","18-32","32-50","50+"]
print("\n=== T1. hiding %, TRUE injury (rows) x PERCEIVED pain (cols), no predator near ===")
print(f"{'true inj':>10}"+"".join(f"{c:>14}" for c in PL))
for i in range(NBI):
    row=f"{IL[i]:>10}"
    for j in range(NBP):
        row+= f"{100*b2[i,j]/n2[i,j]:>8.1f}%({n2[i,j]/1e6:>4.1f}M)" if n2[i,j]>2e4 else f"{'-':>14}"
    print(row)
print("\n  read ACROSS a row: true injury fixed, perceived pain varies")
print("  read DOWN a column: perceived pain fixed, true injury varies")
print("\n=== T2. hiding around a damage event (t=0 is the hit) ===")
print(f"{'lag':>5}{'all events':>14}{'isolated events':>18}")
for li,lag in enumerate(LAGS):
    if lag%2 and abs(lag)>2: continue
    a=100*bE[li,0]/max(nE[li,0],1); b=100*bE[li,1]/max(nE[li,1],1)
    print(f"{lag:>5}{a:>13.1f}%{b:>17.1f}%")
print("\n=== T3. effective response kernel: excess hiding k steps after a hit ===")
print(f"{'k':>3}{'hit at t-k':>12}{'no hit':>10}{'excess':>10}{'true kernel':>13}")
for kk3 in range(16):
    a=100*bL[kk3]/max(nL[kk3],1); b=100*bL0[kk3]/max(nL0[kk3],1)
    tk=KER[kk3] if kk3<KL else 0.0
    print(f"{kk3:>3}{a:>11.1f}%{b:>9.1f}%{a-b:>+9.1f}{tk:>13.4f}")
print("\n=== T4. at matched perceived pain: rising vs falling ===")
print(f"{'perceived':>12}{'falling':>12}{'flat':>12}{'rising':>12}")
for i in range(NBP):
    r=f"{PL[i]:>12}"
    for j in range(3):
        r+= f"{100*bD[i,j]/nD[i,j]:>11.1f}%" if nD[i,j]>2e4 else f"{'-':>12}"
    print(r)
