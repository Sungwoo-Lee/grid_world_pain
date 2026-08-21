"""Capture / verify the pre-change observation baseline for the directional-sensors work.

CP1 of docs/develop/active/sensors/DIRECTIONAL_SENSORS_PLAN.md.

  --capture   dump observations from the CURRENT code to a .npz fixture
  (default)   re-run and compare against that fixture, bit-for-bit

The comparison is against a STORED artefact produced before the change, not a
fresh re-derivation from source -- per CLAUDE.md's rule on verifying actual state.
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Bit-exact float results are LOWERING-dependent: CPU and GPU legitimately differ,
# and GPU autotuning makes repeat GPU runs differ from each other. A parity fixture
# is only meaningful on a PINNED backend, so force CPU for BOTH capture and compare.
# Must happen before jax is imported.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax, jax.numpy as jnp, numpy as np
assert jax.default_backend() == 'cpu', (
    f"parity fixture must run on CPU, got {jax.default_backend()!r}")
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.wrapper import ParallelEnv

FIX = 'tests/env/fixtures/directional_sensors/obs_baseline.npz'
CONFIGS = ['configs/environment/default.yaml']
NUM_ENVS, STEPS, SEED = 32, 20, 12345


def rollout(cfg_path):
    cfg = Config.load_yaml(cfg_path)
    params = load_env_params(cfg)
    env = ParallelEnv(params)
    key = jax.random.PRNGKey(SEED)
    states, obs = env.reset(key, NUM_ENVS)
    frames = [np.asarray(obs)]
    for t in range(STEPS):
        key, sk = jax.random.split(key)
        acts = jax.random.randint(sk, (NUM_ENVS,), 0, int(params.action_dim))
        states, obs, _, _, _ = env.step(states, acts)
        frames.append(np.asarray(obs))
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', action='store_true')
    a = ap.parse_args()
    out = {}
    for c in CONFIGS:
        out[c] = rollout(c)
        print(f"  {c}: {out[c].shape}  (steps+1, envs, obs_dim)")
    if a.capture:
        np.savez_compressed(FIX, **{k.replace('/', '__'): v for k, v in out.items()})
        print(f"\ncaptured -> {FIX}")
        return 0
    ref = np.load(FIX)
    bad = 0
    for c, got in out.items():
        exp = ref[c.replace('/', '__')]
        if got.shape != exp.shape:
            print(f"  FAIL {c}: shape {exp.shape} -> {got.shape}"); bad += 1; continue
        if got.tobytes() == exp.tobytes():
            print(f"  OK   {c}: bit-identical")
        else:
            print(f"  FAIL {c}: max abs diff {np.max(np.abs(got-exp)):.3e}, "
                  f"{int((got != exp).sum())} of {got.size} elements differ")
            bad += 1
    print("\nPARITY HELD" if not bad else f"\nPARITY BROKEN in {bad} config(s)")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
