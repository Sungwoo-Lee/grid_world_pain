"""Speed AND bit-exactness of three ways to compute the olfactory diamond.

The plan's choice (vmap over cells of the existing sense_resource) is the
conservative one: it preserves the per-cell computation verbatim, so the centre
cell stays bit-identical to today. This asks what the alternatives buy.

Run: python bench_olfaction.py
"""
import sys, time
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')
import jax, jax.numpy as jnp, numpy as np
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.sensor import sense_resource, get_visual_offsets
from src.environment.wrapper import ParallelEnv

NUM_ENVS, REPEAT, TIMEIT = 128, 64, 40


# ---------------------------------------------------------------- variants ---
def olf_today(state, params):
    """Baseline: exactly what get_observation does now (single point)."""
    p = state.agent_pos
    return (sense_resource(p, state.res_pos, state.res_active, state.res_property_sampled,
                           params.sensor_radius, params.sensor_decay)
          + sense_resource(p, state.animal_pos, state.animal_active, state.animal_property_sampled,
                           params.sensor_radius, params.sensor_decay)
          + sense_resource(p, state.obs_pos, state.obs_active, state.obs_property_sampled,
                           params.sensor_radius, params.sensor_decay))


def olf_vmap(state, params, r):
    """A: the plan's choice -- vmap the untouched sense_resource over cells."""
    cells = state.agent_pos + get_visual_offsets(r)
    return jax.vmap(lambda p: (
        sense_resource(p, state.res_pos, state.res_active, state.res_property_sampled,
                       params.sensor_radius, params.sensor_decay)
        + sense_resource(p, state.animal_pos, state.animal_active, state.animal_property_sampled,
                         params.sensor_radius, params.sensor_decay)
        + sense_resource(p, state.obs_pos, state.obs_active, state.obs_property_sampled,
                         params.sensor_radius, params.sensor_decay)))(cells).flatten()


def _pool_broadcast(cells, pos, active, prop, radius, gamma):
    diff = pos[None, :, :] - cells[:, None, :]                 # [C,N,2]
    dist = jnp.linalg.norm(diff, axis=-1)                      # [C,N]
    decay = jnp.where(dist < 0.001, 2.0, 1.0 / (jnp.power(dist, gamma) + 1e-10))
    mask = jnp.logical_and(active[None, :], dist <= radius)
    return jnp.sum(prop[None, :, :] * decay[:, :, None] * mask[:, :, None], axis=1)


def olf_broadcast(state, params, r):
    """B: explicit broadcast, three pools kept separate. No vmap."""
    cells = state.agent_pos + get_visual_offsets(r)
    return (_pool_broadcast(cells, state.res_pos, state.res_active, state.res_property_sampled,
                            params.sensor_radius, params.sensor_decay)
          + _pool_broadcast(cells, state.animal_pos, state.animal_active,
                            state.animal_property_sampled, params.sensor_radius, params.sensor_decay)
          + _pool_broadcast(cells, state.obs_pos, state.obs_active, state.obs_property_sampled,
                            params.sensor_radius, params.sensor_decay)).flatten()


def olf_matmul(state, params, r):
    """C: one concatenated [C,E] @ [E,V] matmul -- the vision-style form."""
    cells = state.agent_pos + get_visual_offsets(r)
    pos = jnp.concatenate([state.res_pos, state.animal_pos, state.obs_pos], 0)
    act = jnp.concatenate([state.res_active, state.animal_active, state.obs_active], 0)
    prp = jnp.concatenate([state.res_property_sampled, state.animal_property_sampled,
                           state.obs_property_sampled], 0)
    dist = jnp.linalg.norm(pos[None, :, :] - cells[:, None, :], axis=-1)
    decay = jnp.where(dist < 0.001, 2.0, 1.0 / (jnp.power(dist, params.sensor_decay) + 1e-10))
    W = decay * jnp.logical_and(act[None, :], dist <= params.sensor_radius)
    return (W @ prp).flatten()


# ------------------------------------------------------------------ timing ---
def timeit(fn, *a):
    jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(TIMEIT):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter()-t0)
    return float(np.min(ts))


def repeated(f):
    def g(states):
        return jax.lax.fori_loop(0, REPEAT, lambda i, acc: acc + jnp.sum(f(states)),
                                 jnp.float32(0.0))
    return jax.jit(g)


def main():
    cfg = Config.load_yaml('configs/environment/default.yaml')
    params = load_env_params(cfg)
    env = ParallelEnv(params)
    states, _ = env.reset(jax.random.PRNGKey(0), NUM_ENVS)
    E = states.res_pos.shape[1] + states.animal_pos.shape[1] + states.obs_pos.shape[1]

    base = jax.jit(lambda s: jax.vmap(lambda x: olf_today(x, params))(s))
    ref = np.asarray(base(states))

    print("=== BIT-EXACTNESS vs today's single-point sensor (centre cell, range 0) ===")
    for name, fn in [('A vmap', olf_vmap), ('B broadcast', olf_broadcast), ('C matmul', olf_matmul)]:
        got = np.asarray(jax.jit(lambda s: jax.vmap(lambda x: fn(x, params, 0))(s))(states))
        exact = got.tobytes() == ref.tobytes()
        maxdiff = float(np.max(np.abs(got - ref)))
        print(f"  {name:<12} bit-identical: {str(exact):<5}  max abs diff: {maxdiff:.3e}")

    step_fn = jax.jit(lambda s, a: env.step(s, a))
    t_step = timeit(step_fn, states, jnp.zeros(NUM_ENVS, jnp.int32))
    t_base = timeit(repeated(lambda s: jax.vmap(lambda x: olf_today(x, params))(s)), states) / REPEAT
    print(f"\n=== SPEED (batch of {NUM_ENVS} envs, E={E} sources, dispatch amortised) ===")
    print(f"  env.step                          {t_step*1e6:8.1f} us")
    print(f"  olfaction today (single point)    {t_base*1e6:8.1f} us\n")
    print(f"  {'range':>5} {'cells':>6} {'A vmap':>10} {'B broadcast':>13} {'C matmul':>11}   {'A cost vs env.step':>19}")
    for r in (0, 1, 2, 3):
        C = 2*r*r + 2*r + 1
        ts = {}
        for name, fn in [('A', olf_vmap), ('B', olf_broadcast), ('C', olf_matmul)]:
            ts[name] = timeit(repeated(
                lambda s, f=fn, rr=r: jax.vmap(lambda x: f(x, params, rr))(s)), states) / REPEAT
        print(f"  {r:>5} {C:>6} {ts['A']*1e6:>9.1f}us {ts['B']*1e6:>12.1f}us "
              f"{ts['C']*1e6:>10.1f}us   {100*(ts['A']-t_base)/t_step:>17.2f}%")


if __name__ == '__main__':
    main()
