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
