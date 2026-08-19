"""How often does the on-source rule actually fire, before and after the diamond?

Runs the REAL environment with the shipped config and counts, per step, whether
any olfactory sampling point sits exactly on a source that carries a non-zero
chemical signature. (Rocks and hiding predators ship all-zero signatures, so the
2.0 decay multiplies to nothing for them and they are excluded.)

Writes frequency.json for the figure/page builders.
"""
import sys, json, pathlib
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')
import jax, jax.numpy as jnp, numpy as np
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.wrapper import ParallelEnv

NUM_ENVS, STEPS, SEED = 256, 400, 0


def diamond(r):
    return np.array([(dr, dc) for dr in range(-r, r+1) for dc in range(-r, r+1)
                     if abs(dr) + abs(dc) <= r], dtype=np.int32)


def main():
    cfg = Config.load_yaml('configs/environment/default.yaml')
    params = load_env_params(cfg)
    env = ParallelEnv(params)
    key = jax.random.PRNGKey(SEED)
    states, _ = env.reset(key, NUM_ENVS)

    # which entities can smell at all (non-zero mean signature)
    smelly = {
        'res':    np.any(np.asarray(params.res_property) != 0, axis=-1),
        'animal': np.any(np.asarray(params.animal_property) != 0, axis=-1),
        'obs':    np.any(np.asarray(params.obs_property) != 0, axis=-1),
    }
    print("smelly entity counts:", {k: int(v.sum()) for k, v in smelly.items()},
          "of", {k: int(v.size) for k, v in smelly.items()})

    hits = {0: 0, 1: 0, 2: 0}
    total = 0
    for t in range(STEPS):
        key, sk = jax.random.split(key)
        acts = jax.random.randint(sk, (NUM_ENVS,), 0, int(params.action_dim))
        states, _, _, _, _ = env.step(states, acts)

        ap = np.asarray(states.agent_pos)                       # [B,2]
        pos, act = [], []
        for name, p_attr, a_attr in [('res', 'res_pos', 'res_active'),
                                     ('animal', 'animal_pos', 'animal_active'),
                                     ('obs', 'obs_pos', 'obs_active')]:
            P = np.asarray(getattr(states, p_attr))             # [B,N,2]
            A = np.asarray(getattr(states, a_attr)).astype(bool)
            if P.shape[1] == 0:
                continue
            pos.append(P); act.append(A & smelly[name][None, :])
        P = np.concatenate(pos, axis=1)                         # [B,E,2]
        A = np.concatenate(act, axis=1)                         # [B,E]

        for r in (0, 1, 2):
            cells = ap[:, None, :] + diamond(r)[None, :, :]     # [B,C,2]
            same = np.all(cells[:, :, None, :] == P[:, None, :, :], axis=-1)  # [B,C,E]
            hits[r] += int(np.any(same & A[:, None, :], axis=(1, 2)).sum())
        total += NUM_ENVS

    out = {'num_envs': NUM_ENVS, 'steps': STEPS, 'total_agent_steps': total,
           'pct': {str(r): 100.0 * hits[r] / total for r in hits},
           'smelly_counts': {k: int(v.sum()) for k, v in smelly.items()}}
    pathlib.Path('docs/develop/active/sensors/onsource_rule_study/frequency.json').write_text(
        json.dumps(out, indent=1))
    print(f"\nover {total:,} agent-steps, the on-source rule fires on:")
    for r in (0, 1, 2):
        n = 2*r*r + 2*r + 1
        print(f"  olfactory range {r} ({n:2d} cells): {out['pct'][str(r)]:5.2f}% of steps")


if __name__ == '__main__':
    main()
