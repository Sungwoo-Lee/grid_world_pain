# 11 — Parallel Env Wrapper

> **Source**: `src/environment/wrapper.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Plain-language entry point

This document describes how the GridWorld Pain environment is run as many simultaneous, independent copies ("parallel environments") during training. Running N copies in parallel is how RL training loops collect enough experience quickly — instead of stepping through episodes one at a time in Python, JAX's `vmap` ("vectorised map") transformation compiles the environment step into a single hardware kernel that processes all N copies at once, with no Python-level loop overhead.

Two tools are provided:

- **`ParallelEnv`** (`wrapper.py:7`): a class that wraps the core environment functions and exposes clean `reset` / `step` methods for N-environment batches. It does **not** auto-reset terminated environments — that is left to the training loop.
- **`auto_reset_step`** (`wrapper.py:35`): a standalone function that steps all N environments and, for any environment that just finished an episode, immediately resets it in-place. This keeps all N environments perpetually "running" without the training loop having to detect and handle episode boundaries.

**What "vmap" means in plain language**: `jax.vmap(f)` takes a function `f` that operates on a single environment and returns a new function that operates on a batch of N environments — all at once, in a single GPU/TPU kernel call. The `in_axes` argument tells JAX which argument axis is the "batch" dimension for each parameter. When `in_axes=(None, 0)` is set for `(params, key)`, it means: broadcast `params` (one copy for all envs) and slice `key` along axis 0 (each env gets its own key).

**What "PRNG key threading" means**: JAX's random number generator is purely functional — you must explicitly pass and advance a key. Each environment in the batch carries its own independent key inside its `EnvState`. At every step, `jax_step` splits that key into 6 sub-keys for different random events (resource respawn, animal movement, damage sampling, etc.), and stores the advanced main key back into the state. This ensures each environment's random events are independent and reproducible.

---

## `ParallelEnv`

```python
class ParallelEnv:
    def __init__(self, params: EnvParams)
    def reset(self, key: PRNGKey, num_envs: int) → (EnvState, obs)
    def step(self, states: EnvState, actions: jnp.ndarray) → (EnvState, obs, rewards, dones, infos)
```

### `__init__` (`wrapper.py:9–15`)

Pre-builds three vmapped functions at construction time, so JAX traces and compiles them once:

```python
self._v_reset = jax.vmap(jax_reset, in_axes=(None, 0))      # params broadcast; key batched
self._v_step  = jax.vmap(jax_step,  in_axes=(0, 0, None))   # state+action batched; params broadcast
self._v_obs   = jax.vmap(get_observation, in_axes=(0, None)) # state batched; params broadcast
```

**`in_axes` explained — why each argument is batched or broadcast:**

| Function | Arg 1 | Arg 2 | Arg 3 | Reason |
|---|---|---|---|---|
| `jax_reset(params, key)` | `None` (broadcast) | `0` (batched) | — | `params` is the same config for all envs; each env gets a distinct `key` from a `split` |
| `jax_step(state, action, params)` | `0` (batched) | `0` (batched) | `None` (broadcast) | Each env has its own `EnvState` and receives its own action; `params` is shared |
| `get_observation(state, params)` | `0` (batched) | `None` (broadcast) | — | Each env has its own `EnvState`; `params` is shared |

`EnvParams` is always `in_axes=None` — a single immutable copy is shared and broadcast to all N environments. This keeps memory usage proportional to `N × sizeof(EnvState)`, not `N × sizeof(EnvState) + N × sizeof(EnvParams)`.

### `reset(key, num_envs)` (`wrapper.py:17–21`)

```python
def reset(self, key: jax.random.PRNGKey, num_envs: int) -> tuple[EnvState, jnp.ndarray]:
    keys = jax.random.split(key, num_envs)   # shape [N, 2] — one independent key per env
    states = self._v_reset(self.params, keys) # batched EnvState, all arrays [N, ...]
    obs = self._v_obs(states, self.params)    # [N, obs_dim]
    return states, obs
```

`jax.random.split(key, num_envs)` produces `num_envs` independent subkeys from a single top-level key. This guarantees:
- All N environments have distinct PRNG streams from the first step.
- The entire reset is deterministic and reproducible: the same `key` and `num_envs` always produces the same starting states.

The returned `states` is a batched `EnvState` pytree where every array field has a leading axis of size `N`. For example:
- `states.agent_pos`: shape `[N, 2]`
- `states.animal_pos`: shape `[N, A, 2]` (A = number of animals)
- `states.key`: shape `[N, 2]` (each env's independent PRNG key)

`obs` is shape `[N, obs_dim]`.

### `step(states, actions)` (`wrapper.py:23–33`)

```python
def step(self, states: EnvState, actions: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Steps all environments in parallel."""
    next_states, rewards, dones, infos = self._v_step(states, actions, self.params)
    obs = self._v_obs(next_states, self.params)
    return next_states, obs, rewards, dones, infos
```

- `actions`: shape `[N]` (one integer action per environment).
- `rewards`: shape `[N]` float.
- `dones`: shape `[N]` bool.
- `infos`: dict of batched arrays. Each scalar info field (e.g. `termination_reason`) becomes shape `[N]`; per-entity arrays (e.g. `dist_per_predator`) gain a leading `[N, ...]` dimension.
- `next_states`: batched `EnvState`, all arrays `[N, ...]`.

**No auto-reset**: `ParallelEnv.step` returns terminal states as-is. Terminated environments continue to hold their terminal `EnvState` until the caller explicitly resets them. Use `auto_reset_step` if you need in-call reset, or handle episode boundaries in your training loop with `jax.lax.cond` / masking.

---

## `auto_reset_step` (`wrapper.py:35–61`)

A standalone function that steps all N environments and, for any that just terminated, immediately resets them so the next call to `auto_reset_step` continues from a fresh episode.

### Signature

```python
def auto_reset_step(
    states: EnvState,
    actions: jnp.ndarray,
    params: EnvParams,
    key: jax.random.PRNGKey
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]
```

Note: despite accepting `key` as an argument, the current implementation does **not** use it — each environment's reset key is derived from the environment's own `state.key` (see below). The `key` parameter is present in the signature but unused (`wrapper.py:35`).

### Inner function

```python
def step_fn(state, action):
    next_state, reward, done, info = jax_step(state, action, params)

    reset_key, _ = jax.random.split(state.key)     # derive reset key from current env key
    reset_state = jax_reset(params, reset_key)       # produce a fresh state

    # Pytree-level conditional: swap reset_state leaf-by-leaf where done=True
    final_state = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(done, x, y),   # x=reset_state, y=next_state
        reset_state, next_state
    )

    obs = get_observation(next_state, params)       # terminal obs, not reset obs
    return final_state, obs, reward, done, info

return jax.vmap(step_fn, in_axes=(0, 0))(states, actions)
```

### Key design decisions

**1. Auto-reset masking via `jax.lax.select` over pytree leaves** (`wrapper.py:50–53`):

`jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), reset_state, next_state)` walks every leaf array of the two `EnvState` pytrees in parallel and selects, element-by-element:
- `reset_state` leaf if `done=True` (the environment just terminated)
- `next_state` leaf if `done=False` (normal transition)

This is functionally a conditional branch but is compiled as a masked select — no branching in the JAX/XLA graph, which is necessary for `vmap` and `jit` compatibility. The `done` scalar is broadcast across all leaf shapes during the select.

**2. Terminal observation, not reset observation** (`wrapper.py:55`):

```python
obs = get_observation(next_state, params)
```

The returned `obs` always corresponds to the **terminal state** (`next_state`), not the reset state. This matches the Gymnax/CleanRL convention: on the terminal transition, `obs` carries the terminal observation that value estimates should bootstrap from (with `done=True` as the episode-end marker). The `final_state` returned is already the reset state, so the next call to `auto_reset_step` will step from the fresh episode — but the last observation in the current rollout is the terminal one.

This means after a terminal transition:
- `obs` describes `next_state` (terminal)
- `final_state` is `reset_state` (fresh episode for next step)
- `done=True` flags the boundary

**3. Reset PRNG key derivation** (`wrapper.py:45`):

```python
reset_key, _ = jax.random.split(state.key)
```

The reset key is derived from the **current** environment's key (the key at the moment of termination), not from the top-level `key` argument. This means:
- No external key management needed per env per episode.
- The reset is deterministic given the terminal `state.key`.
- Reproducibility of the reset depends on the entire episode's key chain, not just a fresh seed. If you need deterministic episode-N resets from a known seed, you should track the reset key at launch rather than replaying through the full episode chain.
- `reset_state` is always computed (no short-circuit in `jax.lax.select`) even on non-terminal transitions. The cost is one `jax_reset` call per env per step — paid even when `done=False`. For most training configurations this is negligible.

**4. `vmap` in `auto_reset_step`** (`wrapper.py:61`):

```python
return jax.vmap(step_fn, in_axes=(0, 0))(states, actions)
```

The vmap is constructed inline at call time (not pre-compiled in `__init__`). Each invocation re-applies vmap over the `(state, action)` batch axes. `params` is closed over from the outer scope — it is not batched.

---

## PRNG Key Handling

### Per-step key advancement (inside `jax_step`)

Each `EnvState` carries its own `key` field. Inside `jax_step` (`core.py:372`), it is split into 6 sub-keys at the top of every step:

```python
key, respawn_key, hunt_key, wander_key, damage_key, property_key = jax.random.split(state.key, 6)
```

These sub-keys are used for:
- `respawn_key`: resource position resampling on respawn (`core.py:381`)
- `hunt_key`: hunt-subset animal movement and state transitions (`core.py:403`)
- `wander_key`: wander-subset animal random jitter movement (`core.py:404`)
- `damage_key`: stochastic damage sampling for animals, resources, and obstacles (`core.py:424`, `core.py:474`, `core.py:487`)
- `property_key`: re-sampling chemical properties for respawned resources (`core.py:391`)
- `key`: advanced main key, stored back as `new_state.key` (`core.py:655`)

This means each environment's key advances by one `split(6)` operation per step — the key stream grows deterministically along the episode.

### Per-reset key seeding (inside `jax_reset`)

`jax_reset` (`core.py:762`) splits its incoming key into 5 sub-keys:

```python
key, agent_key, placement_key, body_key, property_key = jax.random.split(key, 5)
```

The `key` at this point is stored as `state.key` in the returned `EnvState` (`core.py:1010`). This seeds the per-step key chain for the new episode.

### Batch seeding in `ParallelEnv.reset`

```python
keys = jax.random.split(key, num_envs)   # [N, 2]
states = self._v_reset(self.params, keys)
```

`jax.random.split(key, num_envs)` produces `num_envs` subkeys, each independent. All N environments therefore start with distinct PRNG streams. The full reset is reproducible: same top-level `key` → same initial states for all N envs.

### Independent streams guarantee

Two environments that start with different keys (produced by `split`) will always diverge in their random event sequences, because JAX's PRNG is a counter-mode CSPRNG. There is no global shared state — all randomness flows through the key field in `EnvState`.

---

## Batched EnvState Shape Convention

When `ParallelEnv.reset` or `_v_reset` produces states for N environments, all `EnvState` array fields gain a leading axis of size N:

| Field | Single-env shape | N-env batched shape | Notes |
|---|---|---|---|
| `agent_pos` | `[2]` | `[N, 2]` | row, col |
| `animal_pos` | `[A, 2]` | `[N, A, 2]` | A = total animals (unified, predators-first) |
| `animal_state` | `[A]` | `[N, A]` | int: 0=PATROL, 1=HUNT, 2=RETURN |
| `animal_stamina` | `[A]` | `[N, A]` | float |
| `animal_move_timer` | `[A]` | `[N, A]` | int |
| `animal_attack_timer` | `[A]` | `[N, A]` | int |
| `animal_property_sampled` | `[A, V]` | `[N, A, V]` | chemical properties |
| `animal_detect_sampled` | `[A]` | `[N, A]` | per-episode sampled detection range |
| `animal_max_stamina_sampled` | `[A]` | `[N, A]` | per-episode sampled max stamina |
| `animal_recovery_sampled` | `[A]` | `[N, A]` | per-episode sampled recovery rate |
| `animal_hunt_thresh_sampled` | `[A]` | `[N, A]` | per-episode sampled hunt threshold |
| `animal_lose_interest_sampled` | `[A]` | `[N, A]` | per-episode sampled lose-interest factor |
| `res_pos` | `[R, 2]` | `[N, R, 2]` | R = number of resources |
| `res_active` | `[R]` | `[N, R]` | bool |
| `res_cons_count` | `[R]` | `[N, R]` | int |
| `res_reg_timer` | `[R]` | `[N, R]` | int |
| `res_property_sampled` | `[R, V]` | `[N, R, V]` | |
| `obs_pos` | `[O, 2]` | `[N, O, 2]` | O = number of obstacles |
| `obs_property_sampled` | `[O, V]` | `[N, O, V]` | |
| `satiation` | `[]` | `[N]` | float scalar per env |
| `nutrition` | `[]` | `[N]` | float scalar per env |
| `injury_level` | `[]` | `[N]` | float scalar per env |
| `injury_buffer` | `[S]` | `[N, S]` | S = `smoothing_duration` |
| `nociception_history_buffer` | `[K]` | `[N, K]` | K = `interoceptive_kernel_length` |
| `last_collision_noc` | `[]` | `[N]` | float |
| `rest_streak` | `[]` | `[N]` | int |
| `terminated` | `[]` | `[N]` | bool |
| `key` | `[2]` | `[N, 2]` | JAX PRNGKey is shape `[2]` |
| `last_action` | `[]` | `[N]` | int32 |
| `current_step` | `[]` | `[N]` | int32 |

**`EnvParams` is never batched.** It is `in_axes=None` in all three vmapped functions. A single copy is broadcast to all N environments. Fields that are static (`pytree_node=False`, e.g. `height`, `width`, `placement_mode`) are compile-time constants and not JAX arrays at all.

---

## Unified `animal_*` Arrays

As of v2.0, the separate `pred_*` / `neutral_*` arrays (e.g. `pred_pos`, `neutral_pos`, `pred_state`, etc.) have been merged into unified `animal_*` arrays under a **predators-first** ordering. This refactor is reflected throughout `EnvState` and `EnvParams`:

- `state.animal_pos` replaces `state.pred_pos` + `state.neutral_pos` — shape `[A, 2]` (or `[N, A, 2]` batched)
- `state.animal_state` replaces `state.pred_state` (wander-type animals always have state=0)
- `state.animal_stamina` replaces `state.pred_stamina` (zero for wander/static animals, kept for shape stability)
- `state.animal_move_timer` replaces `state.pred_move_timer` + `state.neutral_move_timer`
- `state.animal_attack_timer` replaces `state.pred_attack_timer` (zero for non-hunt animals)
- `state.animal_property_sampled` replaces `state.pred_property_sampled` + `state.neutral_property_sampled`
- The five per-episode-sampled behavioural fields (`animal_detect_sampled`, `animal_max_stamina_sampled`, `animal_recovery_sampled`, `animal_hunt_thresh_sampled`, `animal_lose_interest_sampled`) are new in v2.0 and have no pre-refactor equivalent.

In `EnvParams`, `animal_is_damaging` (`[A]` bool) identifies which animals deal damage — this is used in `jax_step` instead of the old class-based predator/neutral split. `params.predator_indices` and `params.neutral_indices` (static tuples, `pytree_node=False`) are used only during `jax_reset` for placement and property sampling; they are not iterated at step time.

When consuming batched states in analysis or rendering code, use `state.py:select_by_class(params, 'predator')` or `state.py:select_by_class(params, 'neutral')` to recover per-class masks (host-side NumPy, not JAX-traced).

---

## Parallelism Notes

**Why `vmap` over Python loops**: `jax.vmap` transforms the function into a single vectorised computation that XLA can map to hardware SIMD units or distribute across accelerator cores. This avoids Python-level overhead per environment and produces a single compiled kernel for all N envs. In practice, stepping 1024 envs with `vmap` takes nearly the same wall-clock time as stepping 1 env, up to GPU memory limits.

**Limitations**:
- All N environments must use the same `EnvParams`. Different configs (e.g. different grid sizes, different entity counts) require separate `ParallelEnv` instances and separate vmap calls. Shape heterogeneity is not supported within a single batch.
- `num_envs` can change between `reset` calls, but this changes the vmap trace and may require XLA recompilation. For training loops that fix `num_envs` at launch this is not a concern.
- `auto_reset_step` always computes `jax_reset` for every environment every step, even when `done=False`. The reset result is discarded via `jax.lax.select`, but the computation is not skipped. This is the standard JAX cost of in-graph branching.

**JIT compatibility**: both `ParallelEnv.step` and `auto_reset_step` are pure JAX (no Python side effects) and can be wrapped in `@jax.jit` or used inside a larger `jit`-compiled training step. The `ParallelEnv` instance is trivial; `self.params` can be closed over.

**`pmap` for multi-device**: `ParallelEnv` uses `vmap`, not `pmap`. To scale across multiple GPUs/TPUs, wrap the entire training step in `jax.pmap` over a device-batch axis and let each device run its own `ParallelEnv` with `num_envs` per device.

---

## Clarifications / FAQ

**Q: Does `ParallelEnv.step` auto-reset terminated environments?**
A: **No.** It is a plain vmapped `jax_step` + `get_observation` (`wrapper.py:26–33`). Terminal states persist until the caller explicitly resets them. Use `auto_reset_step` if you want in-call reset, or handle it in your training loop.

**Q: Does `auto_reset_step` have the same observation semantics as `ParallelEnv.step`?**
A: Almost. Both return the **terminal** observation from `next_state` (not the reset observation), matching Gymnax/CleanRL. The returned `final_state` from `auto_reset_step` is the reset state (for next step's input). The subtle invariant: after a terminal transition, `obs` describes `next_state` but `final_state = reset_state`. Bootstrap value estimates should use `obs` with `done=True` as a terminal marker.

**Q: How does `auto_reset_step` pick the reset PRNG key?**
A: `reset_key, _ = jax.random.split(state.key)` at `wrapper.py:45`. It derives the reset key from the current environment's key at the moment of termination. Deterministic given the terminal state — no top-level key needed. But reset reproducibility depends on the entire episode's key chain, not just a fresh seed.

**Q: Is the top-level `key` argument to `auto_reset_step` used?**
A: **No.** The current implementation (`wrapper.py:35–61`) accepts `key` in the signature but does not use it. All reset keys are derived per-env from `state.key`. Callers can pass `jax.random.PRNGKey(0)` safely.

**Q: Can I mix envs with different `num_envs` in the same `ParallelEnv` instance?**
A: No — `num_envs` is set at `reset` time and determines the batched shape. Changing it triggers recompilation. Create multiple `ParallelEnv` instances or pad to a fixed max if you need variable batch sizes.

**Q: Does `ParallelEnv` support heterogeneous `EnvParams` across envs?**
A: No. `params` is `in_axes=None` for both reset and step — broadcast to all envs. Heterogeneous configs require multiple `ParallelEnv` instances with separate vmap calls, then concatenating outputs.

**Q: How does `info` get batched?**
A: Each field in `info` is vmapped independently. A scalar info field (e.g. `termination_reason`) becomes a `[N]` array. Per-entity arrays (e.g. `dist_per_predator`, `dist_per_neutral`, `dist_per_animal`) gain a leading `[N, ...]` dimension. Consumers must aggregate over the leading axis to summarise across envs.

**Q: Is the `apply_noise` arg honoured in `ParallelEnv.step`?**
A: `_v_obs = jax.vmap(get_observation, in_axes=(0, None))` — the `apply_noise` kwarg is not surfaced through `ParallelEnv.step`. It defaults to `True` (from `sensor.py:270`: `@jax.jit(static_argnames=['apply_noise'])`). To disable noise in eval, either call `get_observation` directly with `apply_noise=False`, or build a separate vmapped eval-obs function.

**Q: Can I JIT-compile the whole training loop including `ParallelEnv.step`?**
A: Yes. Both `_v_step` and `_v_obs` are pure JAX. The enclosing class is trivial; `self.params` is a static reference that can be closed over. Common pattern: wrap `ParallelEnv.step` in `@jax.jit` or call inside a larger jitted training step.

**Q: Do terminated envs continue stepping with valid data?**
A: Yes, they just keep stepping. `jax_step` doesn't short-circuit on `terminated=True`. The env stays in its terminal state and produces valid (but possibly non-sensical) observations/rewards until explicitly reset. This is why `auto_reset_step` exists.

**Q: Does `auto_reset_step`'s `reset_state` get computed even when `done=False`?**
A: Yes. `jax.lax.select` is a data-select, not a control-flow branch. Both `reset_state` and `next_state` are always computed; the result is masked. You pay for one `jax_reset` per env per step regardless of termination. For most training runs this is negligible.

**Q: What's the memory cost of a large `num_envs`?**
A: Roughly `num_envs × sizeof(EnvState)` on GPU/TPU. For a typical 10×10 grid with ~20 entities, `EnvState` is a few KB per env (mostly the animal/resource arrays), so 1024 envs ≈ a few MB. Easily fits on any GPU.

**Q: Can I use `jax.pmap` instead of `jax.vmap` for multi-device parallelism?**
A: Not directly — `ParallelEnv` uses `vmap`. Multi-device scaling needs a separate wrapper that `pmap`s over a device-batch dimension and vmaps within each device. Not provided out-of-the-box.

**Q: Is the `key` in `EnvState` also vmapped?**
A: Yes. A JAX `PRNGKey` has shape `[2]`. After vmap over N envs it becomes shape `[N, 2]`. Each env has its own independent key stream seeded by `jax.random.split(key, num_envs)` at reset time.

**Q: What happens to `info['termination_reason']` after auto-reset?**
A: It reports the reason from the **terminal** transition. Reason codes: 0=active, 1=max_steps truncation, 2=starvation, 3=overeating, 4=injury. The fresh reset state will produce `reason=0` on its next step. Good for logging episode-end reasons.

**Q: Do I need to call `get_observation` myself after `ParallelEnv.reset`?**
A: No — `reset` returns `(states, obs)` already. Same for `step`. Only call `get_observation` directly if you need `apply_noise=False` or a custom state snapshot.

**Q: How does olfaction handle the unified animal array in the observation?**
A: `sensor.py:301` calls `sense_resource` once over `state.animal_pos` + `state.animal_property_sampled` (all animals, always treated as "active"). The old separate `pred_chem` + `neutral_chem` calls have been replaced by a single `animal_chem` call. When this is batched via `_v_obs`, the `[N, A, 2]` animal positions and `[N, A, V]` properties are processed correctly — vmap slices the leading N axis and the inner `sense_resource` sees single-env arrays.
