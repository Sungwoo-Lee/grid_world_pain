# JAX & Advanced-API Primer for GridWorld Pain

This document is the shared reference for every other doc in the `docs/environment/` series. It teaches each advanced JAX, Flax, and Orbax API **once**, grounded in how this codebase actually uses it. The other 14 docs link back here on first use of an API rather than re-explaining it.

## Who this is for

A competent Python and RL developer who is new to JAX. You know Python classes, NumPy, and how an RL step loop works. You do not need to know Flax, Orbax, or JAX's functional style. This primer does not teach basic Python — only the patterns that differ meaningfully from NumPy/PyTorch.

## How to use this document

Each section has an HTML anchor (`<a id="...">`) so you can deep-link from another doc. For example, `[[00_jax_primer#jit]]` (Foam/Obsidian wikilink) or `[jax.jit](00_jax_primer.md#jit)` (standard Markdown) both land at the JIT section.

Section order follows a learning curve: pytrees and immutability first (they underpin everything else), then compilation, vectorization, randomness, and the higher-level APIs built on top.

---

<a id="jax-pytrees"></a>
## Pytrees, Flax `@struct.dataclass`, and `.replace()`

### What a pytree is

JAX operates on **pytrees** — any nested container of arrays. A pytree is a tree whose leaves are JAX arrays and whose nodes are Python containers (lists, tuples, dicts) or registered custom types. `jax.jit`, `jax.vmap`, `jax.lax.scan`, and friends all accept and return pytrees, recursively mapping over every leaf array while leaving the structure intact.

Think of a pytree as "a bag of arrays that JAX knows how to walk". If you hand `jax.jit` a function whose argument is an `EnvState`, JAX sees through the struct and processes each array field independently. The code never needs to flatten and re-pack arrays by hand.

### `@struct.dataclass` makes a class a pytree node

Flax's `@struct.dataclass` decorator registers the class as a pytree node so JAX can walk its fields. Every field annotated with a JAX array type is a **leaf**. The struct itself is a **node**.

```python
# src/environment/state.py:30-81
@struct.dataclass
class EnvState:
    agent_pos: jnp.ndarray      # [2] (row, col)
    current_step: jnp.ndarray   # []
    res_pos: jnp.ndarray        # [num_res, 2]
    res_active: jnp.ndarray     # [num_res] bool
    # ... (more fields)
    key: jax.random.PRNGKey
    last_action: jnp.ndarray    # [] (int32 action index)

    def _replace(self, **kwargs):
        return self.replace(**kwargs)
```

`src/environment/state.py:30-81`

`EnvParams` is declared the same way (line 82). Both are pytrees, so `jax.jit(jax_reset)(params, key)` correctly traces through all their fields.

### Immutability and `.replace()`

JAX arrays are **immutable**. You cannot do `state.agent_pos = new_pos`. Instead, Flax structs expose a `.replace(**kwargs)` method that returns a **new** struct with the named fields swapped, leaving all other fields untouched. This is the functional equivalent of object mutation.

The codebase aliases `.replace()` as `._replace()` for compatibility with NamedTuple-style call sites. Both are identical:

```python
# src/environment/state.py:79-80
def _replace(self, **kwargs):
    return self.replace(**kwargs)
```

`src/environment/state.py:79-80`

In practice, every step ends with one large `.replace()` call that assembles the next state:

```python
# src/environment/core.py:627-657
new_state = state._replace(
    agent_pos=new_agent_pos,
    current_step=next_step,
    res_pos=res_pos_after_reg,
    res_active=final_active,
    # ... (remaining fields)
    key=key,
    last_action=jnp.array(action, dtype=jnp.int32),
)
```

`src/environment/core.py:627-657`

**Why it is written this way.** JAX's compiler traces functions as pure mathematical transformations. Mutation would break the trace. Returning a new struct is the functional pattern that lets JAX reason about data flow, apply transformations (vmap, scan), and compile efficiently.

---

<a id="static-dynamic"></a>
## Static vs. dynamic fields and JIT recompilation

### The problem: JIT needs to know shapes at compile time

When `jax.jit` compiles a function it traces through it once, treating every array as a symbolic placeholder (a **tracer**). It needs to know the shape and dtype of every array at trace time, but it does not need to know the values. This is what makes compilation possible.

Fields that control code structure — grid dimensions, a tuple of class names, the number of entities — cannot be array tracers because Python `if`, `for`, and `len()` branch on their values at trace time. These fields must be **static**: concrete Python objects that are baked into the compiled code.

### `struct.field(pytree_node=False)`

Flax's `struct.field(pytree_node=False)` marks a field as a static leaf. JAX excludes it from the pytree walk (it is not an array) and instead uses its Python value as part of the **cache key** for the compiled function. If the value changes, JAX re-compiles.

```python
# src/environment/state.py:85-87, 126-138, 155-158
class EnvParams:
    height: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)
    max_steps: int = struct.field(pytree_node=False)
    # ...
    animal_classes: tuple = struct.field(pytree_node=False)    # len N strings
    animal_behaviours: tuple = struct.field(pytree_node=False) # len N strings
    animal_tags: tuple = struct.field(pytree_node=False)       # len N strings
    hunt_idx: tuple = struct.field(pytree_node=False)   # tuple[int, ...], len N_hunt
    wander_idx: tuple = struct.field(pytree_node=False) # tuple[int, ...], len N_neutral
    static_idx: tuple = struct.field(pytree_node=False) # tuple[int, ...], len N_static
    predator_indices: tuple = struct.field(pytree_node=False)
    neutral_indices: tuple = struct.field(pytree_node=False)
    # ...
    max_per_type: int = struct.field(pytree_node=False)
    num_types: int = struct.field(pytree_node=False)
    num_entities: int = struct.field(pytree_node=False)
    placement_mode: str = struct.field(pytree_node=False)
```

`src/environment/state.py:85-87, 126-158`

### Why these specific fields are static

- `height`, `width`, `max_steps`: control array allocations and loop bounds inside JIT'd code. Changing them produces different shapes, which requires recompilation anyway.
- `animal_classes`, `animal_behaviours`, `animal_tags`: Python string tuples. String comparisons (`if c == 'predator'`) must be resolved at trace time so the compiler can eliminate dead branches.
- `hunt_idx`, `wander_idx`, `static_idx`, `predator_indices`, `neutral_indices`: index tuples that slice the unified animal array into behaviour subsets. Used in `if len(params.hunt_idx) > 0:` guards and as arguments to `jnp.array(params.hunt_idx)`. Their length must be known at trace time.
- `placement_mode`: selects between two entirely different placement code paths (`'per_entity'` vs `'per_type'`).

### The recompilation cost

Re-compiling is expensive (seconds to tens of seconds). This is why the codebase never changes static fields between calls that go through `jax.jit`. In production: one `EnvParams` instance is created at startup and reused for the entire training run. Changing grid size means a new params object and one cold recompile.

---

<a id="jit"></a>
## `jax.jit`: tracing, compilation, when it recompiles

### The "trace once" mental model

`jax.jit` wraps a function so that the first call **traces** it: JAX runs the Python function body with symbolic array placeholders (tracers) instead of real values, records every operation, and compiles the resulting graph to XLA. Subsequent calls with the same array shapes and dtypes skip Python entirely and execute the compiled binary.

```python
# src/environment/core.py:366-368
@jax.jit
def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Orchestrates a full environment step in JAX."""
```

`src/environment/core.py:366-368`

The decorator form `@jax.jit` is equivalent to `jax_step = jax.jit(jax_step)`. The function is compiled the first time it is called, not at decoration time.

### What triggers recompilation

JAX recompiles when the **cache key** changes. The cache key is formed from:

1. The shapes and dtypes of every array leaf in the arguments.
2. The Python value of every static field (those marked `pytree_node=False`).
3. The identity of the Python function itself.

Concretely:
- Pass a state whose `agent_pos` has shape `(2,)` once and shape `(3,)` another time → recompile.
- Change `params.height` from 10 to 20 → recompile (it is a static field; its value is part of the key).
- Change `params.animal_pos` values but not shape/dtype → no recompile (values are not part of the key for dynamic fields).

### The "you cannot branch on a traced value" constraint

Inside a JIT'd function, array values are tracers — they have no concrete value until execution time. This means you cannot write:

```python
# This will FAIL inside jax.jit — `pos[0]` is a tracer, not a Python int
if state.agent_pos[0] > 5:
    ...
```

The solution is to use `jnp.where` or `jax.lax.cond` (see the [branchless control flow](#branchless) section).

Python-level `if` is fine as long as it branches on a static value (a plain Python bool or int), because that branch is resolved at trace time.

`jax_reset` is also JIT-compiled:

```python
# src/environment/core.py:761-762
@jax.jit
def jax_reset(params: EnvParams, key: jax.random.PRNGKey) -> EnvState:
```

`src/environment/core.py:761-762`

---

<a id="static-argnames"></a>
## `jax.jit(static_argnames=...)`: Python flags that aren't struct fields

### The distinction from `struct.field(pytree_node=False)`

The [static vs. dynamic fields](#static-dynamic) section showed how to mark a *struct field* static with `struct.field(pytree_node=False)`. That mechanism is for fields living inside a pytree (`EnvParams`). But sometimes the value you need to be static is a **plain Python argument** passed directly to a JIT'd function — not a field of any struct. For that case, `jax.jit` takes a `static_argnames` argument: a list of parameter names whose values should be treated as static (baked into the compiled code) rather than traced as arrays.

Same underlying idea — the value becomes part of the compile cache key, and changing it triggers a recompile — but a different entry point: `pytree_node=False` is declared on the *field*, `static_argnames` is declared on the *jit wrapper* for a loose argument.

### The real example

```python
# src/environment/sensor.py:269-270
@jax.jit(static_argnames=['apply_noise'])
def get_observation(state: EnvState, params: EnvParams, apply_noise=True):
```

`src/environment/sensor.py:269-270`

### Why `apply_noise` must be static

`apply_noise` is a plain Python `bool` that controls **which sensors run** — concretely, whether the noise-injection branch executes:

```python
# src/environment/sensor.py:324
if apply_noise:
```

`src/environment/sensor.py:324`

That is a Python `if` branching on `apply_noise`. As the [JIT section](#jit) explained, you cannot branch on a traced value — so `apply_noise` cannot be an ordinary traced argument. Marking it static means its concrete `True`/`False` value is known at trace time, the compiler resolves the `if` and bakes in only the taken branch, eliminating the other entirely.

### The compile-per-value cost

The flip side: each distinct value of a static argument produces its own compiled variant. Training calls `get_observation(state, params)` with the default `apply_noise=True` → one compiled binary with the noise branch. Evaluation calls it with `apply_noise=False` → a **second** compiled binary with the noise branch stripped out. Two values, two compiles. This is cheap here because `apply_noise` only ever takes two values, so at most two variants are ever cached. Static arguments become a problem only when they take many distinct values (each one a fresh cold compile).

---

<a id="vmap"></a>
## `jax.vmap` and `in_axes`

### What vmap does

`jax.vmap` transforms a function that operates on a single example into one that operates on a batch, without writing an explicit loop. It adds a batch dimension to each input array and removes it from the output. The transformation is fused into the JIT-compiled code so no Python loop runs at runtime.

### `in_axes`: which arguments are batched

`in_axes` controls which arguments get the batch dimension and which are broadcast:

- `in_axes=0` (or just `0`): the first axis of this argument is the batch axis.
- `in_axes=None`: this argument is **broadcast** — the same value is used for every element in the batch. No batch axis is added.

### How the wrapper uses vmap

```python
# src/environment/wrapper.py:13-15
self._v_reset = jax.vmap(jax_reset, in_axes=(None, 0))
self._v_step  = jax.vmap(jax_step,  in_axes=(0, 0, None))
self._v_obs   = jax.vmap(get_observation, in_axes=(0, None))
```

`src/environment/wrapper.py:13-15`

Reading these line by line:

- `_v_reset(params, keys)`: `params` is `None`-batched (broadcast — one config for all envs), `keys` is `0`-batched (each env gets its own key). Produces a batch of `EnvState`.
- `_v_step(states, actions, params)`: `states` is batched (one per env), `actions` is batched (one per env), `params` is broadcast (shared config).
- `_v_obs(states, params)`: `states` batched, `params` broadcast.

The call site is straightforward:

```python
# src/environment/wrapper.py:18-20
keys = jax.random.split(key, num_envs)
states = self._v_reset(self.params, keys)
obs = self._v_obs(states, self.params)
```

`src/environment/wrapper.py:18-20`

### vmap inside a JIT'd function

`jax.vmap` is also used inside `jax_step` and `jax_reset` to process per-entity arrays without Python loops:

```python
# src/environment/core.py:386
new_potential_pos = jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)
```

`src/environment/core.py:386`

Here `sample_res_pos` takes a single key and a single area and returns a single position. `jax.vmap` maps it over `num_res` keys and areas simultaneously. No `in_axes` argument means the default `in_axes=0` is used for all arguments.

A second inner use of vmap maps a per-animal **predicate** over the moving animals. Inside `_hunt_step`, `check_collision` decides for a single animal whether its proposed move lands on a blocking obstacle, reverting to its old position if so:

```python
# src/environment/core.py:227-231
    def check_collision(p_pos, old_p_pos):
        is_coll = jnp.any(jnp.logical_and(jnp.all(obs_pos == p_pos, axis=-1), obs_blocking_for_collision))
        return jnp.where(is_coll, old_p_pos, p_pos)

    new_pos = jax.vmap(check_collision)(new_pos, hunt_pos)
```

`src/environment/core.py:227-231`

`check_collision` is written for one animal (one proposed position, one old position). `jax.vmap(check_collision)` lifts it over the whole `[N_hunt, 2]` predator subset at once — every predator's collision check runs in parallel, no Python loop. The neutral-animal path in `_wander_step` does the identical thing over the wander subset:

```python
# src/environment/core.py:271-275
    def check_collision(p_pos, old_p_pos):
        is_coll = jnp.any(jnp.logical_and(jnp.all(obs_pos == p_pos, axis=-1), obs_blocking))
        return jnp.where(is_coll, old_p_pos, p_pos)

    new_pos = jax.vmap(check_collision)(new_pos, wand_pos)
```

`src/environment/core.py:271-275`

So vmap appears at two scales: at the [wrapper level](#vmap) to fan one config across `num_envs` parallel environments, and *inside* a single env's step to fan a one-entity predicate across all animals of a behaviour subset.

---

<a id="prng"></a>
## Functional PRNG: `split`, `fold_in`, key threading

### Why JAX has no global RNG

PyTorch/NumPy maintain a global random state that you seed once and then draw from. JAX does not. The reason is reproducibility and parallelism: if a function has a hidden global state, you cannot vectorize it with `vmap` or compile it with `jit` in a way that gives deterministic, reproducible results.

Instead JAX uses **functional keys**: a pure-data object (a PRNGKey) that you explicitly pass into every function that needs randomness. The function returns any derived keys it produces. This makes the full random stream deterministic given only the initial key.

### `jax.random.split`: producing independent streams

`jax.random.split(key, n)` takes one key and returns `n` independent keys. The original key should not be used again after splitting (by convention).

```python
# src/environment/core.py:372
key, respawn_key, hunt_key, wander_key, damage_key, property_key = jax.random.split(state.key, 6)
```

`src/environment/core.py:372`

This is the opening move of every `jax_step` call: split the state's key into six independent streams — one for resource respawning, one for predator movement, one for neutral movement, one for damage sampling, one for property sampling, and one that becomes the new state key for the next step. The six streams are independent: draws from `hunt_key` are statistically uncorrelated with draws from `damage_key`.

The reset function follows the same pattern:

```python
# src/environment/core.py:781
key, agent_key, placement_key, body_key, property_key = jax.random.split(key, 5)
```

`src/environment/core.py:781`

### `jax.random.fold_in`: deterministic sub-streams without consuming splits

`fold_in(key, data)` mixes an integer `data` into `key` to produce a new key. Unlike `split`, it does not consume the parent key — `key` is still usable afterwards. This is useful when you need a derived key for a specific named purpose without disrupting an existing split budget.

```python
# src/environment/core.py:783
animal_episode_key = jax.random.fold_in(property_key, 0xAE1)
```

`src/environment/core.py:783`

The comment explains why: the existing 5-way split (`agent_key, placement_key, body_key, property_key`) was established to preserve byte-identical random draws with earlier code. Adding a 6th split would shift all downstream keys. `fold_in` derives a new stream from `property_key` without touching the other streams.

The observation pipeline uses the same trick:

```python
# src/environment/sensor.py:273
obs_key = jax.random.fold_in(state.key, 999)
```

`src/environment/sensor.py:273`

Here the integer `999` is an arbitrary domain tag that separates the observation-noise key from any other key derived from `state.key`.

### Other random draw functions in this codebase

| Function | Usage |
|---|---|
| `jax.random.uniform(key, shape, minval, maxval)` | Damage sampling, body start states, per-episode animal behaviour params |
| `jax.random.randint(key, shape, minval, maxval)` | Agent start position, entity placement positions, predator jitter |
| `jax.random.normal(key, shape)` | Property noise sampling |
| `jax.random.permutation(key, n)` | Global cell permutation in overlap resolver |

Example of `uniform` for per-episode behavioural sampling:

```python
# src/environment/core.py:963-964
animal_detect_sampled = jax.random.uniform(
    ep_keys[0], (N,), minval=params.animal_detect_low,
    maxval=jnp.maximum(params.animal_detect_high, params.animal_detect_low))
```

`src/environment/core.py:963-964`

---

<a id="immutability"></a>
## Immutable arrays and functional updates: `.at[idx].set()` / `.add()`

### Why arrays are immutable

In JAX, every array is a value, not a reference to a mutable buffer. This mirrors mathematical notation: `y = f(x)` produces a new value; it does not modify `x`. Immutability is what allows JAX to compile, differentiate, and vectorize functions safely.

The consequence: you cannot write `arr[i] = value`. Instead, use the `.at[].set()` syntax, which returns a **new** array with the update applied.

### The `.at[idx].set(value)` idiom

```python
# src/environment/core.py:337-341
new_pos     = new_pos.at[h_idx].set(new_hunt_pos)
new_state   = new_state.at[h_idx].set(new_hunt_state)
new_stamina = new_stamina.at[h_idx].set(new_hunt_stamina)
new_mt      = new_mt.at[h_idx].set(new_hunt_mt)
new_at      = new_at.at[h_idx].set(new_hunt_at)
```

`src/environment/core.py:337-341`

Here `h_idx` is an integer index array (the predator subset indices). Each line produces a new copy of the full `[N]` animal array with the predator slots updated. The original arrays (`new_pos`, etc.) are not modified — the variable names are rebound to new values.

The injury buffer uses `.at[-1].set()` to shift and zero the tail:

```python
# src/environment/core.py:80
new_buffer = jnp.roll(temp_buffer, -1).at[-1].set(0.0)
```

`src/environment/core.py:80`

And nociception history is updated by rolling and writing slot 0:

```python
# src/environment/core.py:104
new_nociception_history = jnp.roll(state.nociception_history_buffer, 1).at[0].set(new_injury)
```

`src/environment/core.py:104`

### Out-of-bounds behaviour

By default, JAX's `.at[].set()` **clamps** out-of-bounds indices to the nearest valid index rather than raising an error. This is because error handling at the array level would conflict with JIT compilation (errors are control flow). When you need bounds safety, add explicit `jnp.clip` on the index before the update.

---

<a id="branchless"></a>
## Branchless control flow: `jnp.where`, `jax.lax.select`, `jax.lax.cond`

### Why you cannot `if` on a traced value

Inside `jax.jit`, array values are symbolic tracers. A Python `if condition:` requires evaluating `condition` to a Python bool, which is impossible for a tracer. The solution is **branchless control flow**: both branches are computed, and the result is selected elementwise.

### `jnp.where`: elementwise conditional

`jnp.where(condition, on_true, on_false)` returns an array whose elements are taken from `on_true` where `condition` is true and from `on_false` elsewhere. All three arguments can be arrays; they are broadcast together.

Injury recovery:

```python
# src/environment/core.py:93-96
can_recover = jnp.logical_and(info['rested'], applied_inc <= 0)
new_injury = jnp.where(can_recover, new_injury - recovery_amount, new_injury)
new_injury = jnp.clip(new_injury, 0.0, params.max_injury)
```

`src/environment/core.py:93-96`

Position after obstacle collision check:

```python
# src/environment/core.py:35
final_pos = jnp.where(is_collision, pos, new_pos)
```

`src/environment/core.py:35`

`jnp.where` with a scalar condition and array branches is the standard replacement for a Python `if/else` in a JIT'd function. Both branches are always computed; only the selected values survive. This is fine for cheap operations, but wasteful when one branch is very expensive.

### `jax.lax.select`: scalar branch selection for pytrees

`jax.lax.select(condition, on_true, on_false)` is similar to `jnp.where` but operates on scalars or pytrees rather than arrays. In `auto_reset_step`, it selects between a reset state and a running state for each environment:

```python
# src/environment/wrapper.py:50-53
final_state = jax.tree_util.tree_map(
    lambda x, y: jax.lax.select(done, x, y),
    reset_state, next_state
)
```

`src/environment/wrapper.py:50-53`

`jax.tree_util.tree_map` walks both pytrees simultaneously and applies the lambda to each pair of matching leaves. The result is a new state pytree whose every leaf is selected from either `reset_state` or `next_state` depending on `done`.

### `jax.lax.cond`: expensive branches

`jax.lax.cond(condition, true_fn, false_fn, *operands)` calls exactly one branch function at runtime (unlike `jnp.where`, which evaluates both). Use it when one branch is significantly more expensive and you want to avoid paying for the unused branch. The tradeoff is that both functions must have identical input/output signatures.

This codebase uses `jnp.where` almost exclusively in the environment core because the branches are cheap comparisons and position updates.

### Cost summary

| Primitive | Both branches computed? | Works on arrays? | Works on pytrees? |
|---|---|---|---|
| `jnp.where` | Yes | Yes (elementwise) | No |
| `jax.lax.select` | Yes | Yes (scalar/array) | Via `tree_map` |
| `jax.lax.cond` | No (one branch) | Yes | Yes |

---

<a id="tree-map"></a>
## `jax.tree_util.tree_map`: leafwise transforms over a pytree

### Apply a function to every leaf of a pytree

`jax.tree_util.tree_map(f, tree)` walks a [pytree](#jax-pytrees) and applies `f` to every leaf, returning a new pytree with the identical structure but transformed leaves. The multi-argument form `tree_map(f, tree_a, tree_b)` walks **two (or more) pytrees of the same structure in lockstep**, calling `f(leaf_a, leaf_b)` on each matching pair of leaves and assembling the results back into one pytree. It is the pytree-level generalization of "map a function over a list".

### The real example

In `auto_reset_step`, after stepping every env one tick, environments that finished (`done`) must be swapped for a fresh reset state while the rest keep their stepped state. Both `reset_state` and `next_state` are full `EnvState` pytrees with dozens of array fields (`agent_pos`, `res_pos`, `res_active`, `key`, ...). `tree_map` selects between them field-by-field in a single call:

```python
# src/environment/wrapper.py:50-53
        final_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y),
            reset_state, next_state
        )
```

`src/environment/wrapper.py:50-53`

### Why this beats a field-by-field select

`EnvState` has many fields. Without `tree_map` you would write one `jax.lax.select(done, reset_state.agent_pos, next_state.agent_pos)` line per field, then re-assemble them with a giant `.replace(...)`. That is verbose, and — worse — it silently breaks the moment someone adds a new field to `EnvState`: the new field would keep its stepped value on done-envs because nobody remembered to add its select line. `tree_map` walks *whatever* leaves exist, so it stays correct as the struct grows. One line covers the entire state.

### How it composes with `lax.select`

`tree_map` supplies the *structure traversal*; the lambda supplies the *per-leaf operation*. Here the per-leaf op is [`jax.lax.select`](#branchless): `lambda x, y: jax.lax.select(done, x, y)` picks leaf `x` (from `reset_state`) where `done` is true and leaf `y` (from `next_state`) otherwise. Because `done` is a scalar, the same boolean drives the choice for every leaf — the whole env's state flips atomically. `tree_map` (which leaves to touch) and `lax.select` (how to combine each pair) are orthogonal, and snap together cleanly: traversal from one, branch logic from the other.

---

<a id="scan"></a>
## `jax.lax.scan`: carry + xs

### The problem with Python loops inside JIT

A Python `for i in range(N):` inside a `jax.jit`'d function is **unrolled** at trace time: JAX traces through every iteration separately and produces a compiled graph with N copies of the loop body. For small N this is fine; for large N it explodes compile time and binary size.

`jax.lax.scan` is JAX's solution: it compiles the loop body once and runs it N times, threading a **carry** (mutable state) forward through iterations.

### Signature

```python
final_carry, outputs = jax.lax.scan(f, init_carry, xs, length=None)
```

- `f(carry, x) -> (new_carry, y)`: the loop body, called once per iteration.
- `init_carry`: initial carry value (can be any pytree).
- `xs`: a pytree of arrays to slice along axis 0, one slice per iteration. Pass `None` for iterations with no input array.
- `length`: number of iterations (required when `xs=None`).
- Returns: final carry and stacked outputs (one `y` per iteration, stacked along a new axis 0).

### Overlap resolver scan (per-entity, fixed iterations)

The placement overlap resolver in `jax_reset` uses `scan` to process entities one at a time, threading an occupancy mask as the carry:

```python
# src/environment/core.py:704-707
init_carry = (occupancy, all_positions, jnp.array(0))
(_, all_positions, _), _ = jax.lax.scan(
    resolve_one, init_carry, None, length=num_entities
)
```

`src/environment/core.py:704-707`

The carry is `(occupancy, all_positions, i)`. Each call to `resolve_one` checks whether entity `i` overlaps an already-occupied cell, replaces it if so, marks the cell occupied, and returns the updated carry. Using `xs=None` with `length=num_entities` means there is no input array to slice — only the carry threads forward.

### Per-type placement scan

The `per_type` placement mode scans over entity type groups:

```python
# src/environment/core.py:879-883
(_, all_positions, _), _ = jax.lax.scan(
    place_type_group,
    (occupancy, all_positions, scan_key),
    jnp.arange(params.num_types)
)
```

`src/environment/core.py:879-883`

Here `xs = jnp.arange(params.num_types)` provides the type index for each iteration. `params.num_types` is a static field, so the compiled loop has a fixed iteration count known at compile time.

### Why not a Python loop?

Inside `jax.jit`, a Python `for i in range(num_entities):` would be unrolled: every iteration produces its own set of XLA ops. With 20 entities the compiled graph is 20x larger, compile time grows, and debugging becomes painful. `lax.scan` compiles the body once — the graph has the same size regardless of `num_entities`.

---

<a id="one-hot"></a>
## `jax.nn.one_hot`: visual-channel encoding

`jax.nn.one_hot(x, num_classes)` converts integer class indices to binary vectors. Given integer `x` and class count `num_classes`, it returns an array of shape `(*x.shape, num_classes)` with a 1.0 at position `x` and 0.0 everywhere else.

In the visual sensor, each entity type is assigned a channel index (0-7) and converted to an 8-dimensional one-hot vector:

```python
# src/environment/sensor.py:197-202
res_props = jax.nn.one_hot(jnp.where(params.res_type == 0, 3, 4), 8)  # [num_res, 8]
obs_props = jax.nn.one_hot(jnp.full((num_obs,), 6), 8)                 # [num_obs, 8]
# ...
animal_props = jax.nn.one_hot(params.animal_visual_channel, 8)          # [N, 8]
```

`src/environment/sensor.py:197-202`

The channel mapping is:
- 0: Grass, 1: Sand, 2: Plain (background terrain types)
- 3: Food resource, 4: Hiding predator resource
- 5: Predator animal, 6: Rock obstacle, 7: Neutral animal

Each entity's one-hot vector is multiplied by its activity mask (`all_active[:, None]`), zeroing inactive entities, before being summed into the visual observation via a matrix multiply:

```python
# src/environment/sensor.py:211-214
matches = jnp.all(cell_coords[:, None, :] == all_pos[None, :, :], axis=-1)
vis_entities = jnp.matmul(matches.astype(jnp.float32), all_props)
```

`src/environment/sensor.py:211-214`

The proprioception channel uses the same pattern to encode the previous action:

```python
# src/environment/sensor.py:310
obs_parts.append(jax.nn.one_hot(state.last_action, params.action_dim))
```

`src/environment/sensor.py:310`

**Why one-hot rather than an integer?** Neural networks treat input values as continuous. An integer encoding of class labels implies an ordinal relationship (predator "5" is closer to rock "6" than to grass "0") that does not exist. One-hot vectors place all classes equidistant.

---

<a id="masking"></a>
## Fixed-shape masking idioms

### Why fixed shapes and no boolean indexing

In NumPy you can write `arr[mask]` to extract a variable-length subset. In JAX, the output shape must be known at compile time. `arr[mask]` would return an array whose length equals the number of `True` values in `mask`, which is unknown until runtime — JAX disallows this inside JIT.

The fix: keep arrays at their maximum possible size and carry a boolean mask alongside them. Instead of "here are 3 active resources", store "here are 5 resource slots; this bool array says which ones are active". All operations run on all slots; the mask suppresses contributions from inactive ones.

### `jnp.logical_and` / `logical_or` / `logical_not`

Build compound conditions elementwise without Python operators (Python `and`/`or` on tracers calls `__bool__`, which JAX blocks):

```python
# src/environment/core.py:29-32
is_collision = jnp.any(jnp.logical_and(
    jnp.all(obs_pos == new_pos, axis=-1),
    obs_blocking
))
```

`src/environment/core.py:29-32`

### `jnp.all` / `jnp.any`

Reduce a boolean array along an axis. Common pattern: test whether two positions match:

```python
# src/environment/core.py:471
at_animal = jnp.all(new_animal_pos == new_agent_pos, axis=-1)
```

`src/environment/core.py:471`

`axis=-1` reduces along the last axis (the spatial dimension of a `[N, 2]` array), giving a `[N]` bool array: "which animals are at the agent's position?"

### Mask-then-`where`: the masked aggregation pattern

To sum only over active entries:

```python
# src/environment/core.py:477
damage_pred = jnp.sum(jnp.where(at_damaging, sampled_pred_damage, 0.0))
```

`src/environment/core.py:477`

`at_damaging` is a `[N]` bool. `jnp.where` maps it over `sampled_pred_damage`, producing 0.0 for non-damaging entries. `jnp.sum` then adds up only the genuine damage values.

To take a minimum while ignoring inactive resources (substituting a sentinel value 99.0):

```python
# src/environment/core.py:582-584
dist_to_food = jnp.min(jnp.where(
    jnp.logical_and(state.res_active, params.res_type == 0),
    jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1),
    99.0
))
```

`src/environment/core.py:582-584`

### `jnp.clip`

Clamp values to a range. Used for boundary enforcement and safety:

```python
# src/environment/core.py:18-21
new_pos = jnp.array([
    jnp.clip(new_pos[0], 0, params.height - 1),
    jnp.clip(new_pos[1], 0, params.width - 1)
])
```

`src/environment/core.py:18-21`

### `jnp.roll`

Shift array elements along an axis (wrapping around). Used for the injury smoothing buffer and nociception history:

```python
# src/environment/core.py:80
new_buffer = jnp.roll(temp_buffer, -1).at[-1].set(0.0)
```

`src/environment/core.py:80`

`jnp.roll(arr, -1)` shifts all elements left by one, moving the last element to the front. Then `.at[-1].set(0.0)` zeros the tail slot, effectively implementing a FIFO queue without dynamic memory allocation.

### `jnp.concatenate` / `jnp.stack`

Combine arrays along an existing (`concatenate`) or new (`stack`) axis. Used extensively in sensor assembly:

```python
# src/environment/sensor.py:321
obs = jnp.concatenate(obs_parts)
```

`src/environment/sensor.py:321`

`obs_parts` is a Python list of fixed-size array fragments built up during observation construction. `jnp.concatenate` joins them into the final observation vector. Each fragment's size is statically known (controlled by static params fields), so the output shape is known at trace time.

`jnp.stack` joins arrays along a new axis. It is used to build 2D position arrays from row and column components:

```python
# src/environment/core.py:214
move_vec = jnp.stack([final_move_r, final_move_c], axis=-1)
```

`src/environment/core.py:214`

### `jnp.cumsum` and `jnp.sort`

`jnp.cumsum` is used in the placement logic to select the first N valid cells from a permuted list:

```python
# src/environment/core.py:743-744
cumsum = jnp.cumsum(valid_in_perm)
selected = valid_in_perm & (cumsum <= num_entities)
```

`src/environment/core.py:743-744`

`cumsum` turns a boolean mask into a running count; comparing it to `num_entities` selects exactly the first `num_entities` True entries without dynamic indexing.

`jnp.sort` is used just after to sort the selected flat indices into a fixed-size array:

```python
# src/environment/core.py:748
selected_flat = jnp.sort(selected_flat)[:max_entities]
```

`src/environment/core.py:748`

This produces a fixed-size array (length `max_entities`) regardless of how many valid cells were found, satisfying JAX's shape-at-compile-time requirement.

---

<a id="linalg"></a>
## Distances with `jnp.linalg.norm`

`jnp.linalg.norm(x, axis=-1)` computes the Euclidean (L2) norm of each row of `x`. Applied to a `[N, 2]` array of positions, it returns a `[N]` array of distances.

The olfactory sensor computes a decay-weighted chemical gradient using per-resource distances:

```python
# src/environment/sensor.py:8-9
diff = res_pos - agent_pos
dist = jnp.linalg.norm(diff, axis=-1)
```

`src/environment/sensor.py:8-9`

The homeostatic drive (Euclidean distance to the body setpoint in satiation-injury space) uses the same function on a 2D state vector:

```python
# src/environment/core.py:40-42
target = jnp.array([params.setpoint, 0.0])
current = jnp.stack([satiation, injury], axis=-1)
return jnp.linalg.norm(current - target, axis=-1)
```

`src/environment/core.py:40-42`

In `jax_step`, distances from the agent to each animal class are computed for logging:

```python
# src/environment/core.py:586-589
dist_per_animal = (
    jnp.linalg.norm(state.animal_pos - new_agent_pos, axis=-1)
    if state.animal_pos.shape[0] > 0
    else jnp.zeros((0,), dtype=jnp.float32)
)
```

`src/environment/core.py:586-589`

The Python-level `if state.animal_pos.shape[0] > 0` guard is safe because `shape[0]` is a static integer known at trace time (the number of entities is fixed for a given `EnvParams`).

**Why not Manhattan distance?** The olfactory sensor uses L2 because chemical diffusion in continuous space is isotropic. The predator AI uses Manhattan distance for movement direction calculation (L1 is cheaper and natural for grid movement). The homeostatic drive uses L2 to define a smooth energy landscape.

---

<a id="scatter-index"></a>
## The static-index-tuple scatter pattern

This is the project's signature idiom for updating a **subset** of a fixed-size unified array. Understanding it is essential for reading the reset and step code.

### The problem

The environment has N animals stored in unified arrays of shape `[N]` or `[N, 2]`. Some animals are predators (hunt behaviour); others are neutral (wander behaviour); others are static. Different update functions must be applied to each subset. But the subset sizes are fixed at config time, so their indices can be stored statically.

### Predator/neutral index tuples on `EnvParams`

```python
# src/environment/state.py:130-138
predator_indices: tuple = struct.field(pytree_node=False)  # e.g. (0, 1) for 2 predators
neutral_indices: tuple = struct.field(pytree_node=False)   # e.g. (2, 3) for 2 neutrals
hunt_idx: tuple = struct.field(pytree_node=False)          # indices with hunt behaviour
wander_idx: tuple = struct.field(pytree_node=False)        # indices with wander behaviour
static_idx: tuple = struct.field(pytree_node=False)        # indices with static behaviour
```

`src/environment/state.py:130-138`

These are Python tuples of integers stored as static (non-pytree) fields. Their contents are fixed for the lifetime of the `EnvParams` object.

### Gather: slice the subset from the full array

```python
# src/environment/core.py:313-314
h_idx = jnp.array(params.hunt_idx, dtype=jnp.int32)
# animal_pos[h_idx] has shape [N_hunt, 2] — only the hunt-subset rows
```

`src/environment/core.py:313-314`

`jnp.array(params.hunt_idx)` converts the static tuple to a JAX array of indices. Indexing `animal_pos[h_idx]` performs a **gather**: it extracts the rows at those indices, producing an `[N_hunt, 2]` array.

### Process the subset

The gathered slice is passed to the behaviour-specific update function (`_hunt_step` or `_wander_step`), which returns arrays of shape `[N_hunt, ...]`.

### Scatter: write the results back

```python
# src/environment/core.py:337-341
new_pos     = new_pos.at[h_idx].set(new_hunt_pos)
new_state   = new_state.at[h_idx].set(new_hunt_state)
new_stamina = new_stamina.at[h_idx].set(new_hunt_stamina)
new_mt      = new_mt.at[h_idx].set(new_hunt_mt)
new_at      = new_at.at[h_idx].set(new_hunt_at)
```

`src/environment/core.py:337-341`

`.at[h_idx].set(new_hunt_pos)` performs a **scatter**: it writes the `[N_hunt, 2]` result back into the corresponding rows of the `[N, 2]` array. All other rows are unmodified. The same pattern repeats for the wander subset at lines 358-359.

### Why the index tuples are static

If `hunt_idx` were a dynamic JAX array, its values would be tracers at compile time. JAX would not know which cells of `new_pos` are being written to, preventing the compiler from reasoning about the update's structure. Storing the indices as a static Python tuple means the compiler sees concrete integers, enabling efficient indexed update code.

### Application in `jax_reset` for property sampling

The same gather-update-scatter pattern is used for property sampling during reset:

```python
# src/environment/core.py:943-947
p_idx = jnp.array(list(params.predator_indices), dtype=jnp.int32)
pred_prop_mean = params.animal_property[p_idx]
pred_prop_std  = params.animal_property_std[p_idx]
pred_prop_sampled = _sample_property(prop_key_pred, pred_prop_mean, pred_prop_std)
animal_property_sampled = animal_property_sampled.at[p_idx].set(pred_prop_sampled)
```

`src/environment/core.py:943-947`

The predator class properties are sampled with `prop_key_pred` (a different key than the neutral-class sampling key `prop_key_neutral`), then scattered back into the unified `[N, vector_size]` property array.

---

<a id="orbax"></a>
## Orbax checkpointing: `CheckpointManager`, `StandardSave`, `StandardRestore`

### Why a dedicated checkpoint library?

JAX model weights are nested pytrees of arrays spread across NumPy / accelerator memory. Writing them to disk naively (e.g., `pickle`) is fragile across JAX versions, slow for large models, and hard to resume from mid-training. Orbax is the standard JAX checkpoint library: it serializes any pytree reliably, manages a directory of numbered checkpoints, and handles atomic writes.

### `CheckpointManager`: the session handle

```python
# src/algorithms/dreamer_srl/checkpoint.py:36-41
manager = ocp.CheckpointManager(
    os.path.abspath(ckpt_dir),
    checkpointers=ocp.StandardCheckpointer(),
    options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True),
)
```

`src/algorithms/dreamer_srl/checkpoint.py:36-41`

`CheckpointManager` tracks a directory of checkpoint steps. `max_to_keep` controls how many past checkpoints are retained (older ones are deleted automatically). `create=True` creates the directory if it does not exist.

### `StandardSave`: writing a pytree

```python
# src/algorithms/dreamer_srl/checkpoint.py:77-102
ckpt_data = {
    'world_model': nnx.state(world_model, nnx.Param),
    'actor':       nnx.state(actor, nnx.Param),
    'key':         key,
    'iter_num':    jnp.array(iter_num, dtype=jnp.int32),
    # ...
}
manager.save(episode, args=ocp.args.StandardSave(ckpt_data))
manager.wait_until_finished()
```

`src/algorithms/dreamer_srl/checkpoint.py:77-102`

`nnx.state(module, nnx.Param)` extracts the trainable parameters from a Flax NNX module as a pytree. `ocp.args.StandardSave(ckpt_data)` wraps the pytree for the save call. `manager.wait_until_finished()` blocks until the write completes (saves are asynchronous by default).

The `episode` argument is the checkpoint step label — `manager.save(episode, ...)` creates a subdirectory named after `episode` inside `ckpt_dir`.

### `StandardRestore`: loading a pytree

```python
# src/algorithms/dreamer_srl/checkpoint.py:115
return manager.restore(episode)
```

`src/algorithms/dreamer_srl/checkpoint.py:115`

`manager.restore(episode)` reads the checkpoint saved at the given step and returns a raw pytree dict matching the structure passed to `StandardSave`. To update a live model, call `nnx.update(module, restored_dict['actor'])` (or the equivalent for each module).

### What is serialized

The `ckpt_data` dict contains:
- `'world_model'`, `'actor'`, `'critic'`, `'target_critic'`: model parameter pytrees (extracted via `nnx.state`).
- `'moments'`: a dict of JAX arrays (running statistics for return normalization).
- `'key'`, `'iter_num'`, `'policy_step'`, `'total_episodes_completed'`, `'cumulative_grad_steps'`: scalar bookkeeping state as 0-d JAX arrays (Orbax handles 0-d arrays natively).

**Why 0-d arrays for scalars?** Plain Python ints are not JAX pytree leaves; Orbax cannot serialize them directly. Wrapping them as `jnp.array(x, dtype=jnp.int32)` makes them first-class pytree leaves that Orbax can round-trip correctly.
