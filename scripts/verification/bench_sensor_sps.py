"""Environment throughput (steps per second) for the directional-sensors work.

Measures the real ParallelEnv under a jitted lax.scan, which is how the training
loop actually drives it -- not a Python loop with per-step dispatch.

  python scripts/verification/bench_sensor_sps.py [--json out.json]

Each row is one sensor configuration. `--label` tags the run so before/after
results can be compared in one table.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import jax, jax.numpy as jnp, numpy as np
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.wrapper import ParallelEnv
from src.environment.core import jax_step

NUM_ENVS, SCAN, REPS = 128, 200, 7


def sps_for(overrides, label):
    cfg = Config.load_yaml('configs/environment/default.yaml')
    for k, v in overrides.items():
        cfg.set(k, v)
    try:
        params = load_env_params(cfg)
    except Exception as e:
        return {'label': label, 'error': str(e)[:120]}
    env = ParallelEnv(params)
    key = jax.random.PRNGKey(0)
    states, obs = env.reset(key, NUM_ENVS)
    obs_dim = int(obs.shape[-1])

    @jax.jit
    def run(states, key):
        def body(carry, _):
            st, k = carry
            k, sk = jax.random.split(k)
            a = jax.random.randint(sk, (NUM_ENVS,), 0, int(params.action_dim))
            st, o, r, d, i = env.step(st, a)
            return (st, k), o.sum()
        (st, k), acc = jax.lax.scan(body, (states, key), None, length=SCAN)
        return st, acc

    jax.block_until_ready(run(states, key))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); jax.block_until_ready(run(states, key))
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    return {'label': label, 'obs_dim': obs_dim,
            'sps': NUM_ENVS * SCAN / best,
            'us_per_step': best / SCAN * 1e6}


CASES = [
    ({}, 'baseline (all new features off)'),
    ({'sensory.olfactory_grid_range': 1}, 'olfaction range 1'),
    ({'sensory.visual_sensor_range': 2}, 'vision range 2, blur off'),
    ({'sensory.visual_sensor_range': 2, 'sensory.visual_blur_enabled': True}, 'vision range 2 + blur'),
    ({'sensory.olfactory_grid_range': 1, 'sensory.visual_sensor_range': 2,
      'sensory.visual_blur_enabled': True}, 'BOTH (chosen settings)'),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--json'); ap.add_argument('--label', default='')
    a = ap.parse_args()
    rows = [sps_for(o, l) for o, l in CASES]
    base = next((r for r in rows if r['label'].startswith('baseline')), None)
    print(f"\n{NUM_ENVS} envs x {SCAN} steps per scan, best of {REPS}\n")
    print(f"  {'configuration':<34} {'obs':>5} {'SPS':>12} {'us/step':>9} {'vs baseline':>12}")
    for r in rows:
        if 'error' in r:
            print(f"  {r['label']:<34}  ERROR: {r['error']}"); continue
        rel = f"{100*(r['sps']/base['sps']-1):+.1f}%" if base and 'sps' in base else '—'
        print(f"  {r['label']:<34} {r['obs_dim']:>5} {r['sps']:>12,.0f} "
              f"{r['us_per_step']:>8.1f}u {rel:>12}")
    if a.json:
        json.dump({'label': a.label, 'num_envs': NUM_ENVS, 'scan': SCAN, 'rows': rows},
                  open(a.json, 'w'), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == '__main__':
    main()
