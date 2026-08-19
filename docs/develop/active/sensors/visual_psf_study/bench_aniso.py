"""Cost of the anisotropic visual kernel, measured against the real environment.

Answers: over 10M+ training steps, how much does replacing the boolean match
matrix with a gaussian weight matrix actually cost?

Run:  python bench_aniso.py
"""
import sys, time, functools
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')
import jax, jax.numpy as jnp, numpy as np
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.sensor import sense_visual, get_observation, get_visual_offsets
from src.environment.wrapper import ParallelEnv

NUM_ENVS = 128          # configs/train/default.yaml
REPEAT   = 64           # inner repeats, to amortise dispatch overhead
TIMEIT   = 60


# --------------------------------------------------------------- proposed ---
def sense_visual_aniso(agent_pos, state, params, k_radial, rho, sigma_floor):
    """Anisotropic point-spread visual sensor.

    Identical in structure to sense_visual: one matmul of a [C, E] weight matrix
    against the [E, V] property matrix. The ONLY change is that the weight matrix
    holds gaussian weights instead of exact-match booleans.
    """
    V = params.visual_vector_size
    offsets = get_visual_offsets(params.visual_sensor_range)      # [C,2] static
    cell_coords = agent_pos + offsets                             # [C,2]

    is_in_bounds = jnp.all(jnp.logical_and(
        cell_coords >= 0,
        cell_coords < jnp.array([params.height, params.width])), axis=-1)
    safe_coords = jnp.where(is_in_bounds[:, None], cell_coords, 0)
    loc_types = params.grid_location_type[safe_coords[:, 0], safe_coords[:, 1]]
    bg_sel = jnp.where(loc_types == 1, 0, jnp.where(loc_types == 2, 1, 2))
    vis_background = params.visual_background_property[bg_sel] * is_in_bounds[:, None]

    parts_pos, parts_props, parts_active = [state.res_pos], [state.res_visual_property_sampled], [state.res_active]
    if state.animal_pos.shape[0] > 0:
        parts_pos.append(state.animal_pos)
        parts_props.append(state.animal_visual_property_sampled)
        parts_active.append(state.animal_active)
    parts_pos.append(state.obs_pos)
    parts_props.append(state.obs_visual_property_sampled)
    parts_active.append(state.obs_active)
    all_pos = jnp.concatenate(parts_pos, 0).astype(jnp.float32)   # [E,2]
    all_props = jnp.concatenate(parts_props, 0)                   # [E,V]
    all_active = jnp.concatenate(parts_active, 0).astype(jnp.float32)

    # ---- the kernel: everything below is the added work ----
    ap = agent_pos.astype(jnp.float32)
    rel = all_pos - ap                                            # [E,2]
    d = jnp.sqrt(jnp.maximum(jnp.sum(rel * rel, -1), 1e-12))      # [E]
    u = rel / d[:, None]                                          # [E,2]  (d>=1 on a grid, or entity on agent)
    u = jnp.where((d > 1e-6)[:, None], u, jnp.array([1.0, 0.0]))
    t = jnp.stack([-u[:, 1], u[:, 0]], -1)                        # [E,2]

    sp = jnp.maximum(k_radial * d, sigma_floor)                   # [E]
    st = jnp.maximum(sp / rho, sigma_floor)                       # [E]

    # v.u and v.t without ever forming the [C,E,2] tensor: two small matmuls
    cf = cell_coords.astype(jnp.float32)                          # [C,2]
    vpar = cf @ u.T - jnp.sum(all_pos * u, -1)[None, :]           # [C,E]
    vperp = cf @ t.T - jnp.sum(all_pos * t, -1)[None, :]          # [C,E]

    inv_p = 0.5 / (sp * sp)
    inv_t = 0.5 / (st * st)
    W = jnp.exp(-(vpar * vpar) * inv_p[None, :] - (vperp * vperp) * inv_t[None, :])
    W = W * (all_active / (2.0 * jnp.pi * sp * st))[None, :]      # mass-normalise + activity
    # ---- end kernel ----

    vis_entities = W @ all_props                                  # [C,V]  (was matches @ all_props)
    return ((vis_background + vis_entities) * is_in_bounds[:, None]).flatten()


# ------------------------------------------------------------------ timing ---
def timeit(fn, *args):
    jax.block_until_ready(fn(*args))          # warm-up / compile
    ts = []
    for _ in range(TIMEIT):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.min(ts))   # min, not median: these are launch-bound micro-kernels


def repeated(f):
    """Run f REPEAT times inside one jit so dispatch overhead is amortised."""
    def g(states):
        def body(i, acc):
            return acc + jnp.sum(f(states))
        return jax.lax.fori_loop(0, REPEAT, body, jnp.float32(0.0))
    return jax.jit(g)


def main():
    cfg = Config.load_yaml('configs/environment/default.yaml')
    rows = []
    for rng_range in (0, 1, 2, 3):
        cfg.set('sensory.visual_sensor_range', rng_range)
        params = load_env_params(cfg)
        env = ParallelEnv(params)
        states, _ = env.reset(jax.random.PRNGKey(0), NUM_ENVS)
        C = 2*rng_range**2 + 2*rng_range + 1
        E = (states.res_pos.shape[1] + states.animal_pos.shape[1] + states.obs_pos.shape[1])

        cur = repeated(lambda s: jax.vmap(lambda x: sense_visual(x.agent_pos, x, params))(s))
        ani = repeated(lambda s: jax.vmap(lambda x: sense_visual_aniso(
            x.agent_pos, x, params, jnp.float32(0.5), jnp.float32(3.0), jnp.float32(0.5)))(s))
        t_cur = timeit(cur, states) / REPEAT
        t_ani = timeit(ani, states) / REPEAT

        obs_fn = jax.jit(lambda s: jax.vmap(lambda x: get_observation(x, params, True))(s))
        t_obs = timeit(obs_fn, states)
        acts = jnp.zeros(NUM_ENVS, jnp.int32)
        step_fn = jax.jit(lambda s, a: env.step(s, a))
        t_step = timeit(step_fn, states, acts)
        obs_dim = int(get_observation(jax.tree.map(lambda x: x[0], states), params, True).shape[0])

        rows.append(dict(r=rng_range, C=C, E=E, obs_dim=obs_dim,
                         cur_us=t_cur*1e6, ani_us=t_ani*1e6,
                         obs_us=t_obs*1e6, step_us=t_step*1e6))
        print(f"range {rng_range}: C={C:2d} E={E} obs_dim={obs_dim:3d} | "
              f"current {t_cur*1e6:7.1f} us  aniso {t_ani*1e6:7.1f} us  "
              f"(delta {(t_ani-t_cur)*1e6:+7.1f} us) | full obs {t_obs*1e6:7.1f} us | "
              f"env.step {t_step*1e6:8.1f} us")

    calls = 10e6 / NUM_ENVS                       # batched calls for 10M agent-steps
    print(f"\nPer call for a batch of {NUM_ENVS} environments. 10M agent-steps = {calls:,.0f} calls.\n")
    print(f"{'range':>5} {'obs dim':>8} {'kernel delta':>13} {'% of env.step':>14} {'total over 10M':>16}")
    for r in rows:
        d = r['ani_us'] - r['cur_us']
        print(f"{r['r']:>5} {r['obs_dim']:>8} {d:>+11.1f} us {100.0*d/r['step_us']:>13.2f}% "
              f"{(d*1e-6)*calls:>13.1f} s")

    # ---- the other half: obs growth feeding the hierarchical encoder ----------
    print("\n--- network side: the visual encoder's first layer scales with the obs ---")
    print("hierarchical encoding, unimodal_overrides.visual = [128, 128]\n")
    key = jax.random.PRNGKey(0)
    B = NUM_ENVS * 128                            # sequence_length 128 -> one rollout
    rollouts = 10e6 / (NUM_ENVS * 128)            # sequence_length 128
    print(f"{'range':>5} {'vis dims':>9} {'layer params':>13} {'fwd+bwd':>10} {'over 10M x4 epochs':>20}")
    for r in rows:
        vdim = (2*r['r']**2 + 2*r['r'] + 1) * 8
        W = jax.random.normal(key, (vdim, 128), jnp.float32) * 0.05
        X = jax.random.normal(key, (B, vdim), jnp.float32)
        def loss(W, X):
            return jnp.sum(jnp.tanh(X @ W))
        f = jax.jit(jax.grad(loss))
        t = timeit(f, W, X)
        print(f"{r['r']:>5} {vdim:>9} {vdim*128:>13,} {t*1e6:>8.1f} us {t*rollouts*4:>17.2f} s")
    print(f"\n(one rollout of {NUM_ENVS} envs x 128 steps through the visual encoder's first layer only;")
    print(f" {rollouts:,.0f} rollouts in 10M steps, K_epochs=4)")
    print("\n--- rollout buffer, float32 observations ---")
    for r in rows:
        mb = NUM_ENVS * 128 * r['obs_dim'] * 4 / 1e6
        print(f"  range {r['r']}: obs_dim {r['obs_dim']:>3} -> {mb:6.2f} MB per rollout")


if __name__ == '__main__':
    main()
