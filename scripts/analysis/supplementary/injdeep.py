"""Injury-dependent hiding, deeper.

Q1 PAIN vs MEMORY: at matched perceived pain, does the number of PRIOR HITS still predict
   hiding? Hit count is a memory variable not recoverable from the current pain value, so if it
   predicts, the agent is using more than the pain channel.
Q2 SAME PAIN, DIFFERENT AGE: the convolution means one big hit 3 steps ago and a small hit 8
   steps ago can give the same felt value. Does behaviour distinguish them?
Q3 APPROACHING DEATH: what happens as injury nears the lethal ceiling of 100?
"""
import glob, time, numpy as np, pyarrow.parquet as pq
S=sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]; SEED0=1000000
KL,TAU=12,3.0
k=np.arange(KL,dtype=np.float64); raw=(k/TAU)*np.exp(1.0-k/TAU); KER=raw/raw.sum()
PRED,D=[0,1],2
ep=pq.read_table(sorted(glob.glob(S+"episodes_*.parquet")),columns=["episode_seed","animal_active"])
o=np.argsort(ep.column("episode_seed").to_numpy())
AACT=np.array(ep.column("animal_active").to_pylist(),bool)[o]
PE=[1e-9,8,18,32,50]                      # 6 perceived-pain bins
HB=[1,2,3,5]                              # 5 cumulative-hit bins: 0,1,2,3-4,5+
AB=[2,4,7,12]                             # 5 age-since-hit bins
IB=[1e-9,25,50,75,90]                     # 6 injury bins incl. near-lethal
n1=np.zeros((6,5)); b1=np.zeros((6,5))
n2=np.zeros((6,5)); b2=np.zeros((6,5))
n3=np.zeros((6,2)); b3=np.zeros((6,2))
def L2(c,w):
    ch=c.chunks if hasattr(c,"chunks") else [c]
    return np.concatenate([x.flatten().to_numpy(zero_copy_only=False) for x in ch]).reshape(-1,w)
t0=time.time()
for fi,f in enumerate(sorted(glob.glob(S+"steps_*.parquet"))):
    tb=pq.read_table(f,columns=["episode_seed","t","agent_in_bush","injury_level","damage",
                                "agent_row","agent_col","animal_row","animal_col"])
    sd=tb.column("episode_seed").to_numpy(); t=tb.column("t").to_numpy(); N=len(t)
    inj=tb.column("injury_level").to_numpy(zero_copy_only=False).astype(np.float64)
    bu=tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.float64)
    dmg=tb.column("damage").to_numpy(zero_copy_only=False).astype(np.float64)
    st=np.flatnonzero(t==0); ends=np.append(st[1:],N)
    estart=np.repeat(st,ends-st); idx=np.arange(N)
    noci=np.zeros(N)
    for j in range(1,KL):
        src=idx-j; ok=src>estart
        noci[ok]+=KER[j]*inj[src[ok]]
    nprev=np.zeros(N); nprev[1:]=noci[:-1]; nprev[idx==estart]=0.0
    hit=(dmg>0)
    cs=np.cumsum(hit); base=np.where(estart>0,cs[estart-1],0)
    nhits=cs-base                                  # hits so far INCLUDING this row
    nprevhits=np.zeros(N,np.int64); nprevhits[1:]=nhits[:-1]; nprevhits[idx==estart]=0
    last=np.where(hit,idx,-1); last=np.maximum.accumulate(last)
    lastok=last>=estart
    age=np.where(lastok,idx-last,999)
    ageprev=np.zeros(N); ageprev[1:]=age[:-1]; ageprev[idx==estart]=999
    ar=tb.column("agent_row").to_numpy(); ac=tb.column("agent_col").to_numpy()
    AR=L2(tb.column("animal_row"),4); AC=L2(tb.column("animal_col"),4)
    pn=((np.maximum(np.abs(AR-ar[:,None]),np.abs(AC-ac[:,None]))<=D)&AACT[sd-SEED0])[:,PRED].any(1)
    m=(t>=2)&(~pn)
    pb=np.digitize(nprev[m],PE)
    hb=np.digitize(nprevhits[m],HB)
    k1=pb*5+hb; n1+=np.bincount(k1,minlength=30).reshape(6,5); b1+=np.bincount(k1,weights=bu[m],minlength=30).reshape(6,5)
    ab=np.digitize(np.minimum(ageprev[m],99),AB)
    k2=pb*5+ab; n2+=np.bincount(k2,minlength=30).reshape(6,5); b2+=np.bincount(k2,weights=bu[m],minlength=30).reshape(6,5)
    ib=np.digitize(inj[m],IB); pnm=pn[m].astype(int)*0
    k3=ib*2+pnm; n3+=np.bincount(k3,minlength=12).reshape(6,2); b3+=np.bincount(k3,weights=bu[m],minlength=12).reshape(6,2)
    if fi%50==0: print(f"  shard {fi}/200 ({time.time()-t0:.0f}s)",flush=True)
np.savez("/home/vncuser/.claude/jobs/4efbe660/tmp/a01_injdeep.npz",n1=n1,b1=b1,n2=n2,b2=b2,n3=n3,b3=b3)
PL=["felt 0","0-8","8-18","18-32","32-50","50+"]
HL=["0 hits","1","2","3-4","5+"]
AL=["1 step ago","2-3","4-6","7-11","12+/never"]
IL=["injury 0","0-25","25-50","50-75","75-90","90-100"]
def show(n,b,rows,cols,title,note):
    print(f"\n=== {title} ===\n{note}")
    print(f"{'':10}"+"".join(f"{c:>15}" for c in cols))
    for i,r in enumerate(rows):
        line=f"{r:>10}"
        for j in range(len(cols)):
            line+= f"{100*b[i,j]/n[i,j]:>9.1f}%({n[i,j]/1e6:>3.1f}M)" if n[i,j]>3e4 else f"{'-':>15}"
        print(line)
show(n1,b1,PL,HL,"Q1. hiding by FELT PAIN (rows) x PRIOR HITS (cols)",
     "no predator within 2 tiles. Read ACROSS a row: pain fixed, does hit count still matter?")
show(n2,b2,PL,AL,"Q2. hiding by FELT PAIN (rows) x TIME SINCE LAST HIT (cols)",
     "no predator within 2 tiles. Read ACROSS a row: same felt pain, different age of injury.")
print("\n=== Q3. hiding as injury approaches the lethal ceiling (no predator near) ===")
for i,l in enumerate(IL):
    if n3[i,0]>3e4: print(f"  {l:>12}: {100*b3[i,0]/n3[i,0]:>6.1f}%   ({n3[i,0]/1e6:.2f}M steps)")
