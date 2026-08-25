"""Time-controlled diagnostic: is the injury/nutrition-dwell gradient just a proxy for
'late in the episode'? Cross-tab dwell by (step-index bin) x (injury bin) and x (nutrition bin)."""
import glob, time, numpy as np, pyarrow.parquet as pq
S = sorted(glob.glob("results/trajectories/*_a01_*/*/*/"))[0]
TB = [0,10,25,50,100,200,500]           # step-index bins
NI, NN, NT = 4, 4, len(TB)-1
cI = np.zeros((NT,NI)); bI = np.zeros((NT,NI))
cN = np.zeros((NT,NN)); bN = np.zeros((NT,NN))
cR = np.zeros((NT,NI)); bR = np.zeros((NT,NI))   # rest x injury
t0 = time.time()
files = sorted(glob.glob(S+"steps_*.parquet"))
for fi, f in enumerate(files):
    tb = pq.read_table(f, columns=["t","injury_level","nutrition","agent_in_bush","rested"])
    t = tb.column("t").to_numpy(); m = t >= 1
    t = t[m]
    inj = tb.column("injury_level").to_numpy(zero_copy_only=False)[m]
    nut = tb.column("nutrition").to_numpy(zero_copy_only=False)[m]
    bu  = tb.column("agent_in_bush").to_numpy(zero_copy_only=False)[m].astype(np.float64)
    rs  = tb.column("rested").to_numpy(zero_copy_only=False)[m].astype(np.float64)
    tbin = np.clip(np.digitize(t, TB[1:-1]), 0, NT-1)
    ib = np.digitize(inj, [1e-9, 25.0, 50.0]); nb = np.digitize(nut, [25.0, 50.0, 75.0])
    for C, Bb, bn, W, NBn in ((cI,bI,ib,bu,NI),(cN,bN,nb,bu,NN),(cR,bR,ib,rs,NI)):
        k = tbin*NBn + bn
        C += np.bincount(k, minlength=NT*NBn).reshape(NT,NBn)
        Bb += np.bincount(k, weights=W, minlength=NT*NBn).reshape(NT,NBn)
    if fi % 50 == 0: print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

def show(C, B, labs, title):
    print(f"\n{title}  — cell = % of steps in a bush (n in millions)")
    print(f"{'step index':14}" + "".join(f"{l:>22}" for l in labs))
    for i in range(NT):
        row = f"{TB[i]:>4}-{TB[i+1]:<9}"
        for j in range(len(labs)):
            row += f"{100*B[i,j]/max(C[i,j],1):>14.1f}% ({C[i,j]/1e6:>4.1f}M)"
        print(row)
    print(f"{'ALL':14}" + "".join(f"{100*B[:,j].sum()/max(C[:,j].sum(),1):>14.1f}% ({C[:,j].sum()/1e6:>4.1f}M)"
                                  for j in range(len(labs))))
IL = ["injury=0","injury 0-25","injury 25-50","injury>=50"]
NL = ["nutr<25","nutr 25-50","nutr 50-75","nutr>=75"]
show(cI, bI, IL, "DWELL by step-index x INJURY")
show(cN, bN, NL, "DWELL by step-index x NUTRITION")
show(cR, bR, IL, "REST-ACTION RATE by step-index x INJURY")
np.savez(("/home/vncuser/.claude/jobs/4efbe660/tmp/a01_timectrl.npz"),
         cI=cI,bI=bI,cN=cN,bN=bN,cR=cR,bR=bR,TB=np.array(TB))
