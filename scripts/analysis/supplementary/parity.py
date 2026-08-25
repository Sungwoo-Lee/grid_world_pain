"""Does the CURRENT environment reproduce the pre-v3.1 environment, with the five new
sensory keys switched off?

Ground truth: the a01 trajectory store recorded real 27-dim observation vectors on 2026-08-20.
The v3.1 sensor change (0e8a4ef) landed 2026-08-21. So the store holds genuine pre-v3.1
observations produced by the system itself.

Test: replay the RECORDED ACTIONS for real episodes through today's code and compare the
observations step by step. This does not re-derive anything from the same suspect path — it
compares today's output against output the old code actually produced.
"""
import glob, sys, yaml, copy, numpy as np
sys.path.insert(0,'.')
import jax, jax.numpy as jnp
import pyarrow.parquet as pq
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation

RUN='results/JAX_RecurrentPPO/20260810-185749_rppo_restprem_a01_n106'
S=sorted(glob.glob('results/trajectories/*_a01_*/*/*/'))[0]
NEW={'sensory.olfactory_sensor_range':0,'sensory.visual_blur_enabled':False,
     'sensory.visual_blur_radial_scale':0.0,'sensory.visual_blur_anisotropy':0.0,
     'sensory.visual_blur_sigma_floor':0.0}
raw=yaml.safe_load(open(RUN+'/models/config.yaml'))
for k,v in NEW.items():
    parts=k.split('.'); d=raw
    for p in parts[:-1]: d=d.setdefault(p,{})
    d[parts[-1]]=v
params=load_env_params(Config(raw))
print(f"params built. olfactory_sensor_range={params.olfactory_sensor_range} "
      f"visual_blur_enabled={params.visual_blur_enabled}")

f=sorted(glob.glob(S+'steps_*.parquet'))[0]
tb=pq.read_table(f,columns=['episode_seed','t','action','obs_true','agent_row','agent_col','injury_level'])
sd=tb.column('episode_seed').to_numpy(); t=tb.column('t').to_numpy()
act=tb.column('action').to_numpy(zero_copy_only=False)
ot=np.array(tb.column('obs_true').to_pylist(),dtype=np.float64)
ar=tb.column('agent_row').to_numpy(); ac=tb.column('agent_col').to_numpy()
st=np.flatnonzero(t==0); ends=np.append(st[1:],len(t))

NEP=25
maxdiff=0.0; maxpos=0; nstep=0; obs_bad=0; pos_bad=0
for e in range(NEP):
    a,b=st[e],ends[e]; seed=int(sd[a])
    state=jax_reset(params,jax.random.PRNGKey(seed))
    o0=np.asarray(get_observation(state,params),dtype=np.float64)
    d=np.abs(o0-ot[a]).max(); maxdiff=max(maxdiff,d); obs_bad+=int(d>1e-5)
    if int(state.agent_pos[0])!=ar[a] or int(state.agent_pos[1])!=ac[a]: pos_bad+=1
    for i in range(a+1,b):
        state,rew,done,info=jax_step(state,jnp.int32(act[i]),params)
        o=np.asarray(get_observation(state,params),dtype=np.float64)
        d=np.abs(o-ot[i]).max(); maxdiff=max(maxdiff,d); obs_bad+=int(d>1e-5); nstep+=1
        if int(state.agent_pos[0])!=ar[i] or int(state.agent_pos[1])!=ac[i]: pos_bad+=1
        if done: break
print(f"\nreplayed {NEP} episodes, {nstep} steps against the recorded pre-v3.1 observations")
print(f"  max |obs difference| over all 27 channels and all steps : {maxdiff:.3e}")
print(f"  steps where any channel differs by > 1e-5               : {obs_bad}")
print(f"  steps where the agent ended on a different tile         : {pos_bad}")
print("\nVERDICT:", "EXACT PARITY — old runs replay faithfully in today's code"
      if maxdiff<1e-5 and pos_bad==0 else "MISMATCH — today's environment differs")
