# 11 — Parallel Env Wrapper

> **Source**: `src/environment/wrapper.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

`ParallelEnv` (`wrapper.py:7`) and `auto_reset_step` (`wrapper.py:35`) provide the standard interface for running many independent environments simultaneously. They wrap `jax_reset`, `jax_step`, and `get_observation` with `jax.vmap` so that a training loop can collect experience from N environments per GPU call.

All N environments share the same `EnvParams` (immutable, single copy). Each environment has its own `EnvState` pytree, stored as a batched pytree where all arrays have a leading axis of size N.

---

## `ParallelEnv`

```python
class ParallelEnv:
    def __init__(self, params: EnvParams)
    def reset(self, key: PRNGKey, num_envs: int) → (EnvState, obs)
    def step(self, states: EnvState, actions: jnp.ndarray) → (EnvState, obs, rewards, dones, infos)
```

**`__init__`** (`wrapper.py:12`): pre-builds three vmapped functions:
```python
self._v_reset = jax.vmap(jax_reset, in_axes=(None, 0))     # vmap over keys
self._v_step  = jax.vmap(jax_step,  in_axes=(0, 0, None))  # vmap over states/actions
self._v_obs   = jax.vmap(get_observation, in_axes=(0, None)) # vmap over states
```

- `params` is `in_axes=None` for both reset and step — the same params are broadcast to all envs.
- `states` and `actions` are `in_axes=0` — the leading axis is the env batch dimension.

**`reset(key, num_envs)`** (`wrapper.py:17`):
```python
keys = jax.random.split(key, num_envs)   # one key per env
states = self._v_reset(self.params, keys)  # [N, ...] batched state
obs = self._v_obs(states, self.params)    # [N, obs_dim]
return states, obs
```

**`step(states, actions)`** (`wrapper.py:23`):
```python
next_states, rewards, dones, infos = self._v_step(states, actions, self.params)
obs = self._v_obs(next_states, self.params)
return next_states, obs, rewards, dones, infos
```

Note: no auto-reset is done in `ParallelEnv.step` — it returns terminal states as-is. The training loop is responsible for handling episode boundaries.

---

## `auto_reset_step`

`auto_reset_step(states, actions, params, key)` (`wrapper.py:35`)

A functional alternative to `ParallelEnv.step` that resets terminated environments in-place within the same call:

```python
def step_fn(state, action):
    next_state, reward, done, info = jax_step(state, action, params)

    reset_key, _ = jax.random.split(state.key)
    reset_state = jax_reset(params, reset_key)

    # Pytree-level conditional select: reset_state if done, next_state if not
    final_state = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(done, x, y),
        reset_state, next_state
    )

    obs = get_observation(next_state, params)  # terminal obs returned, not reset obs
    return final_state, obs, reward, done, info

return jax.vmap(step_fn, in_axes=(0, 0))(states, actions)
```

**Key design decisions**:
- The returned observation is from `next_state` (the terminal state), not `reset_state`. This matches the Gymnax/CleanRL convention where the terminal observation is used for value bootstrapping.
- The returned `final_state` is the reset state for any terminated env — the next call to `step` will continue from the new episode.
- Reset key is derived from `state.key` (the current env key) — deterministic given the terminal state, no additional top-level key needed.

---

## PRNG Key Handling

Each `EnvState` carries its own `key` field, which is advanced every step:
```python
# Inside jax_step:
key, respawn_key, predator_key, neutral_key, damage_key = jax.random.split(state.key, 5)
# ...
new_state = state._replace(key=key)  # main key thread forward
```

This means each environment maintains an independent PRNG stream. Two envs that start with different initial keys will always diverge.

`reset(key, num_envs)` splits the top-level key into `num_envs` subkeys using `jax.random.split(key, num_envs)`. This guarantees all envs have distinct streams and the overall reset is reproducible given the same top-level key.

**Reproducibility**: given the same initial key and params, `jax_reset` and `jax_step` are deterministic (modulo floating point). A full episode is reproducible by replaying actions with the same initial key.

---

## Parallelism Notes

**Why `vmap` over loops**: `jax.vmap` transforms the function into a single vectorised computation that XLA can map to hardware SIMD units or distribute across accelerator cores. This avoids Python-level overhead per environment and produces a single compiled kernel for all N envs.

**Limitations**:
- All N envs must use the same `EnvParams`. Different configs require separate `ParallelEnv` instances and separate vmap calls.
- `jax.vmap` requires all batched arrays to have the same shape. If envs had different grid sizes or entity counts, they could not be batched this way.
- The number of environments `num_envs` can change between calls (by passing different-sized key arrays to `reset`) but this changes the vmap trace and may require recompilation.

**Batched state shapes**: all `EnvState` array fields gain a leading axis of size N. For example:
- `agent_pos`: `[2]` single env → `[N, 2]` batched
- `pred_pos`: `[P, 2]` single env → `[N, P, 2]` batched
- `key`: `PRNGKey` single env → `[N, PRNGKey]` batched

---

## Clarifications / FAQ

**Q: Does `ParallelEnv.step` auto-reset terminated environments?**
A: **No.** It is a plain vmapped `jax_step` + `get_observation` (`wrapper.py:26-33`). Terminal states persist until the caller explicitly resets them. Use `auto_reset_step` if you want in-call reset, or handle it in your training loop.

**Q: Does `auto_reset_step` have the same observation semantics as `ParallelEnv.step`?**
A: Almost. It returns the **terminal** observation from `next_state` (not the reset observation), matching Gymnax/CleanRL. The returned `final_state` is the reset state (for next step's input). The subtle invariant: after a terminal transition, `obs` describes `next_state` but `final_state = reset_state`. Bootstrap value estimates should use `obs` with `done=True` as a terminal marker.

**Q: How does `auto_reset_step` pick the reset PRNG key?**
A: `reset_key, _ = jax.random.split(state.key)` at `wrapper.py:45`. It consumes the current env's key to derive a reset key. Deterministic given the terminal state — no top-level key needed. But beware: this means reset reproducibility depends on the entire episode's key chain, not just a fresh seed.

**Q: Can I mix envs with different `num_envs` in the same `ParallelEnv` instance?**
A: No — `num_envs` is set at `reset` time and determines the batched shape. Changing it triggers recompilation. If you need variable batch sizes (e.g. for curriculum), create multiple `ParallelEnv` instances or pad to a fixed max.

**Q: Does `ParallelEnv` support heterogeneous `EnvParams` across envs?**
A: No. `params` is `in_axes=None` for both reset and step — broadcast to all envs. Heterogeneous configs require multiple `ParallelEnv` instances with separate `jax.vmap` calls, then concatenating outputs.

**Q: How does `info` get batched?**
A: Each field in `info` is vmapped independently. A scalar info field (e.g. `termination_reason`) becomes a `[N]` array. A nested dict would be batched recursively. Consumers must aggregate over the leading axis to summarise across envs.

**Q: Is the `apply_noise` arg honoured in `ParallelEnv.step`?**
A: `_v_obs = jax.vmap(get_observation, in_axes=(0, None))` — the `apply_noise` kwarg isn't surfaced. It defaults to `True`. To disable noise in eval, either call `get_observation` directly with `apply_noise=False`, or wrap with a separate vmapped eval-obs function.

**Q: Can I JIT-compile the whole training loop including `ParallelEnv.step`?**
A: Yes. Both `_v_step` and `_v_obs` are pure JAX. The enclosing class is trivial; `self.params` is a static reference that can be closed over. Common pattern: wrap `ParallelEnv.step` in `@jax.jit` or call inside a larger jitted training step.

**Q: Do terminated envs continue stepping with valid data?**
A: Yes, they just keep stepping. `jax_step` doesn't short-circuit on `terminated=True`. The env stays in its terminal state and produces valid (but possibly non-sensical) observations/rewards until explicitly reset. This is why `auto_reset_step` exists.

**Q: Does `auto_reset_step`'s `reset_state` get thrown away if `done=False`?**
A: Yes, the `reset_state` is always computed (no short-circuit in `jax.lax.select`) but only used if `done=True`. The cost is constant per step — you pay for the reset even on non-terminal transitions. For most training runs this is negligible.

**Q: What's the memory cost of a large `num_envs`?**
A: Roughly `num_envs × sizeof(EnvState)` on GPU. For a typical 10×10 grid with ~20 entities, `EnvState` is a few KB per env, so 1024 envs ≈ a few MB. Easily fits on any GPU.

**Q: Can I use `jax.pmap` instead of `jax.vmap` for multi-device parallelism?**
A: Not directly — `ParallelEnv` uses `vmap`. Multi-device scaling would need a separate wrapper that `pmap`s over a *batch* dimension and vmaps within each device. Not provided out-of-the-box.

**Q: Is the `key` in `EnvState` also vmapped?**
A: Yes. It's an array of shape `[N, 2]` after vmap (a `PRNGKey` is shape `[2]`). Each env has its own independent key stream. The top-level `reset(key, num_envs)` seeds them deterministically from a single top-level key.

**Q: What happens to `info['termination_reason']` after auto-reset?**
A: It reports the reason from the **terminal** transition (the one that caused `done=True`). The fresh reset state has `terminated=False` and will produce `reason=0` on its next step. Good for logging episode-end reasons.

**Q: Do I need to call `get_observation` myself after `ParallelEnv.reset`?**
A: No — `reset` returns `(states, obs)` already. Same for `step`. You only need to call `get_observation` directly if you want a custom apply_noise setting or different state snapshot.
