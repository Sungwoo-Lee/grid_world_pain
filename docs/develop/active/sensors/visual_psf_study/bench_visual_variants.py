"""Visual sensor: parity of the OFF path, and speed of the ON path.

The two paths have different obligations:

  blur OFF -> must be BIT-IDENTICAL to today's sense_visual. The plan restructures
              this path (activity mask moves from all_props onto W, and a mask gate
              multiply is added), so parity is a claim to verify, not assume.
  blur ON  -> new behaviour, no parity constraint. Free to pick the fastest form.

Run: python bench_visual_variants.py
"""
import sys, time
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')
import jax, jax.numpy as jnp, numpy as np
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.sensor import sense_visual, get_visual_offsets
from src.environment.wrapper import ParallelEnv

NUM_ENVS, REPEAT, TIMEIT = 128, 64, 40
K_RAD, RHO, FLOOR = 0.5, 3.0, 0.5


def _gather(state, params):
    parts_pos, parts_props, parts_act = [state.res_pos], [state.res_visual_property_sampled], [state.res_active]
    if state.animal_pos.shape[0] > 0:
        parts_pos.append(state.animal_pos); parts_props.append(state.animal_visual_property_sampled)
        parts_act.append(state.animal_active)
    parts_pos.append(state.obs_pos); parts_props.append(state.obs_visual_property_sampled)
    parts_act.append(state.obs_active)
    return (jnp.concatenate(parts_pos, 0), jnp.concatenate(parts_props, 0),
            jnp.concatenate(parts_act, 0))


def _frame(state, params):
    offs = get_visual_offsets(params.visual_sensor_range)
    cells = state.agent_pos + offs
    inb = jnp.all(jnp.logical_and(cells >= 0,
                                  cells < jnp.array([params.height, params.width])), axis=-1)
    safe = jnp.where(inb[:, None], cells, 0)
    lt = params.grid_location_type[safe[:, 0], safe[:, 1]]
    bg = params.visual_background_property[jnp.where(lt == 1, 0, jnp.where(lt == 2, 1, 2))]
    return offs, cells, inb, bg * inb[:, None]


# --------------------------------------------------------- OFF path: parity ---
def off_restructured(state, params):
    """The plan's OFF path: activity mask on W, plus an all-none mask gate."""
    offs, cells, inb, bg = _frame(state, params)
    pos, props, act = _gather(state, params)
    W = jnp.all(cells[:, None, :] == pos[None, :, :], axis=-1).astype(jnp.float32)
    W = W * act.astype(jnp.float32)[None, :]
    W = W * jnp.ones((1, pos.shape[0]), jnp.float32)      # visual_mask: all none
    return ((bg + W @ props) * inb[:, None]).flatten()


# ----------------------------------------------------------- ON path: forms ---
def _widths(pos, agent):
    rel = pos - agent.astype(jnp.float32)
    d = jnp.sqrt(jnp.maximum(jnp.sum(rel * rel, -1), 1e-12))
    u = jnp.where((d > 1e-6)[:, None], rel / d[:, None], jnp.array([1.0, 0.0]))
    t = jnp.stack([-u[:, 1], u[:, 0]], -1)
    sp = jnp.maximum(K_RAD * d, FLOOR)
    st = jnp.maximum(sp / RHO, FLOOR)
    return u, t, sp, st


def blur_v1(state, params):
    """V1 reference: projections as two small matmuls, normalise W."""
    offs, cells, inb, bg = _frame(state, params)
    pos, props, act = _gather(state, params)
    u, t, sp, st = _widths(pos, state.agent_pos)
    cf = cells.astype(jnp.float32)
    vpar = cf @ u.T - jnp.sum(pos * u, -1)[None, :]
    vprp = cf @ t.T - jnp.sum(pos * t, -1)[None, :]
    W = jnp.exp(-(vpar * vpar) * (0.5 / (sp * sp))[None, :]
                - (vprp * vprp) * (0.5 / (st * st))[None, :])
    W = W * (act.astype(jnp.float32) / (2.0 * jnp.pi * sp * st))[None, :]
    return ((bg + W @ props) * inb[:, None]).flatten()


def blur_v2(state, params):
    """V2: materialise the [C,E,2] displacement tensor and contract."""
    offs, cells, inb, bg = _frame(state, params)
    pos, props, act = _gather(state, params)
    u, t, sp, st = _widths(pos, state.agent_pos)
    v = cells.astype(jnp.float32)[:, None, :] - pos[None, :, :]        # [C,E,2]
    vpar = jnp.einsum('ced,ed->ce', v, u)
    vprp = jnp.einsum('ced,ed->ce', v, t)
    W = jnp.exp(-(vpar * vpar) * (0.5 / (sp * sp))[None, :]
                - (vprp * vprp) * (0.5 / (st * st))[None, :])
    W = W * (act.astype(jnp.float32) / (2.0 * jnp.pi * sp * st))[None, :]
    return ((bg + W @ props) * inb[:, None]).flatten()


def blur_v3(state, params):
    """V3: fold the per-entity normalisation into props ([E,V]) instead of W ([C,E])."""
    offs, cells, inb, bg = _frame(state, params)
    pos, props, act = _gather(state, params)
    u, t, sp, st = _widths(pos, state.agent_pos)
    cf = cells.astype(jnp.float32)
    vpar = cf @ u.T - jnp.sum(pos * u, -1)[None, :]
    vprp = cf @ t.T - jnp.sum(pos * t, -1)[None, :]
    W = jnp.exp(-(vpar * vpar) * (0.5 / (sp * sp))[None, :]
                - (vprp * vprp) * (0.5 / (st * st))[None, :])
    scaled = props * (act.astype(jnp.float32) / (2.0 * jnp.pi * sp * st))[:, None]
    return ((bg + W @ scaled) * inb[:, None]).flatten()


def blur_v4(state, params):
    """V4: rsqrt for the unit vector, no divide."""
    offs, cells, inb, bg = _frame(state, params)
    pos, props, act = _gather(state, params)
    rel = pos - state.agent_pos.astype(jnp.float32)
    d2 = jnp.maximum(jnp.sum(rel * rel, -1), 1e-12)
    inv = jax.lax.rsqrt(d2)
    u = jnp.where((d2 > 1e-12)[:, None], rel * inv[:, None], jnp.array([1.0, 0.0]))
    t = jnp.stack([-u[:, 1], u[:, 0]], -1)
    d = d2 * inv                                             # == sqrt(d2)
    sp = jnp.maximum(K_RAD * d, FLOOR); st = jnp.maximum(sp / RHO, FLOOR)
    cf = cells.astype(jnp.float32)
    vpar = cf @ u.T - jnp.sum(pos * u, -1)[None, :]
    vprp = cf @ t.T - jnp.sum(pos * t, -1)[None, :]
    W = jnp.exp(-(vpar * vpar) * (0.5 / (sp * sp))[None, :]
                - (vprp * vprp) * (0.5 / (st * st))[None, :])
    W = W * (act.astype(jnp.float32) / (2.0 * jnp.pi * sp * st))[None, :]
    return ((bg + W @ props) * inb[:, None]).flatten()


def timeit(fn, *a):
    jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(TIMEIT):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter()-t0)
    return float(np.min(ts))


def repeated(f):
    return jax.jit(lambda s: jax.lax.fori_loop(
        0, REPEAT, lambda i, acc: acc + jnp.sum(f(s)), jnp.float32(0.0)))


def main():
    cfg = Config.load_yaml('configs/environment/default.yaml')
    for rng in (0, 1, 2):
        cfg.set('sensory.visual_sensor_range', rng)
        params = load_env_params(cfg)
        env = ParallelEnv(params)
        states, _ = env.reset(jax.random.PRNGKey(0), NUM_ENVS)
        vm = lambda f: jax.jit(lambda s: jax.vmap(lambda x: f(x, params))(s))

        if rng == 0:
            print("=== OFF PATH — must be bit-identical to today's sense_visual ===")
        ref = np.asarray(vm(lambda x, p: sense_visual(x.agent_pos, x, p))(states))
        got = np.asarray(vm(off_restructured)(states))
        print(f"  range {rng}: bit-identical: {str(got.tobytes()==ref.tobytes()):<5} "
              f" max abs diff: {float(np.max(np.abs(got-ref))):.3e}")

    print("\n=== ON PATH — new behaviour, no parity obligation; pick the fastest ===")
    cfg.set('sensory.visual_sensor_range', 2)
    params = load_env_params(cfg)
    env = ParallelEnv(params)
    states, _ = env.reset(jax.random.PRNGKey(0), NUM_ENVS)
    vm = lambda f: jax.jit(lambda s: jax.vmap(lambda x: f(x, params))(s))
    base = np.asarray(vm(blur_v1)(states))
    for name, f in [('V2 [C,E,2] einsum', blur_v2), ('V3 fold norm into props', blur_v3),
                    ('V4 rsqrt unit vector', blur_v4)]:
        g = np.asarray(vm(f)(states))
        print(f"  {name:<26} vs V1  bit-identical: {str(g.tobytes()==base.tobytes()):<5} "
              f" max abs diff: {float(np.max(np.abs(g-base))):.3e}")

    step = timeit(jax.jit(lambda s, a: env.step(s, a)), states, jnp.zeros(NUM_ENVS, jnp.int32))
    print(f"\n  env.step: {step*1e6:.1f} us   (batch of {NUM_ENVS})\n")
    print(f"  {'variant':<26} {'range 0':>9} {'range 1':>9} {'range 2':>9}")
    for name, f in [('exact match (today)', lambda x, p: sense_visual(x.agent_pos, x, p)),
                    ('V1 reference', blur_v1), ('V2 [C,E,2] einsum', blur_v2),
                    ('V3 fold norm into props', blur_v3), ('V4 rsqrt', blur_v4)]:
        row = []
        for rng in (0, 1, 2):
            cfg.set('sensory.visual_sensor_range', rng)
            p_i = load_env_params(cfg)
            e_i = ParallelEnv(p_i); st_i, _ = e_i.reset(jax.random.PRNGKey(0), NUM_ENVS)
            t = timeit(repeated(lambda s, ff=f, pp=p_i: jax.vmap(lambda x: ff(x, pp))(s)), st_i)
            row.append(t / REPEAT * 1e6)
        print(f"  {name:<26} " + " ".join(f"{v:8.1f}us" for v in row))


    # --- why V1 and V2 diverge: GPU matmul precision, not algebra -------------
    print("\n=== MATMUL PRECISION ===")
    print(f"  jax_default_matmul_precision = {jax.config.jax_default_matmul_precision}")
    for prec in (None, 'highest'):
        if prec:
            jax.config.update('jax_default_matmul_precision', prec)
        jax.clear_caches()
        a = np.asarray(vm(blur_v1)(states)); b = np.asarray(vm(blur_v2)(states))
        print(f"  precision={(prec or 'default'):<8} V1 vs V2 max abs diff: "
              f"{float(np.max(np.abs(a - b))):.3e}")
    print("  -> V1's [C,2]@[2,E] projection matmul runs in reduced precision by default.")


if __name__ == '__main__':
    main()
