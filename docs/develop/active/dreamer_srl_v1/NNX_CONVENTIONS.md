---
title: "dreamer-srl v3 — Project NNX conventions reference"
topic: dreamer
status: active
created: 2026-05-13
last_updated: 2026-05-13
phase: 2
---

# dreamer-srl v3 — Project NNX conventions reference

**Purpose:** This doc is the reference the developer uses during CP1+ so they
don't accidentally pattern-match against the existing in-house Dreamer
implementation (`src/models/dreamer_v3_nnx.py` / `dreamer_v3_trainer.py`).
The new `src/algorithms/dreamer_srl/` module follows these conventions but
does **NOT** import from `src.models.dreamer_v3_*` (v2 Risks §13 — isolation).

Read this before any CP1 work. (~50 lines)

---

## 1. Rngs construction

The project constructs `nnx.Rngs` from a single `jax.random.PRNGKey` at the
call site (train script or test), then passes the `rngs` object down through
`__init__` signatures. Every `nnx.Module.__init__` in this project takes
`rngs: nnx.Rngs` as its last positional argument.

```python
rngs = nnx.Rngs(jax.random.PRNGKey(seed))
model = MyModule(..., rngs=rngs)
```

Inside a module, parameters are drawn from `rngs.params()` (for
`nnx.Param`-holding layers like `nnx.Linear`). The `rngs` object is **not
stored** on the module — it is only used during `__init__`.

**dreamer-srl rule:** Follow the same pattern. Never store `rngs` as an
attribute. Never pass `rngs` to `__call__`.

---

## 2. PRNG during inference / forward pass

Forward-pass randomness (sampling from latent distributions) uses pure-functional
`jax.random.PRNGKey` keys, NOT `rngs`. Keys are passed explicitly as arguments
to `step()`, `imagine_step()`, and `lax.scan` bodies. Keys are split at the
scan level:

```python
scan_rngs = jax.random.split(rng, T * B).reshape((T, B, -1))
```

**dreamer-srl rule:** All stochastic forward-pass operations receive an explicit
`key: jax.Array` argument. No implicit global PRNG state.

---

## 3. JIT boundaries — `nnx.split` / `nnx.merge`

The project JITs training steps via `@nnx.jit` on `nnx.Module` methods
(e.g. `@nnx.jit def train_step(self, batch, rng)`). For cases where the
module must be passed through a plain `jax.jit` boundary (e.g. multi-step
compiled loops), the pattern is:

```python
graphdef, state = nnx.split(self)           # extract pytree state

@jax.jit
def compiled_step(state, batch, rng):
    model = nnx.merge(graphdef, state)      # reconstruct inside jit
    # ... work ...
    new_state = nnx.state(model)            # extract updated state
    return new_state

final_state = compiled_step(state, ...)
nnx.update(self, final_state)               # write back to module
```

`nnx.split` / `nnx.merge` / `nnx.state` / `nnx.update` are the four-function
JIT-boundary idiom. Do not call them inside `@nnx.jit`-decorated methods.

**dreamer-srl rule:** Use `@nnx.jit` for all training methods. Only use the
`split/merge/state/update` pattern when a plain `jax.jit` boundary is required.

---

## 4. EMA (target critic)

The project updates the EMA target critic via `nnx.state` + arithmetic, then
writes back with `nnx.update`:

```python
current_st = nnx.state(self.agent.ac.critic, nnx.Param)
target_st   = nnx.state(self.target_critic, nnx.Param)
new_target_st = jax.tree.map(
    lambda c, t: (1 - tau) * t + tau * c, current_st, target_st
)
nnx.update(self.target_critic, new_target_st)
```

No mutation of `nnx.Variable` values inside a `jax.jit` — only `nnx.state`
(pytree extraction) and `nnx.update` (pytree write-back).

---

## 5. Isolation rule (v2 Risks §13)

`src/algorithms/dreamer_srl/` must NOT import from:
- `src.models.dreamer_v3_nnx`
- `src.models.dreamer_v3_trainer`
- `src.models.dreamer_v3_util`
- Any other file in `src/models/`

The only shared import allowed is `src.utils.config.Config` (for
`config.get_mandatory`) and `src.environment.*` (for env step / reset, if
needed in `train.py`). All Dreamer components (symlog, hafner_init, MLP,
RSSM, etc.) are re-implemented from scratch inside `src/algorithms/dreamer_srl/`
with sheeprl@33b6366 as the source-of-truth.

---

## 6. Key source files read

- `src/models/dreamer_v3_nnx.py` (735 lines) — in-house JAX Dreamer model
- `src/models/dreamer_v3_trainer.py` (1023 lines) — training loop with `@nnx.jit`
- `src/utils/config.py` — `Config.get_mandatory` (raises `ValueError` on missing key)

These files were read on 2026-05-13 as part of the pre-CP0 NNX-convention audit.
They are reference material only; dreamer-srl does not import from them.
