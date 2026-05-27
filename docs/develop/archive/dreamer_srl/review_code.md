---
title: "dreamer-srl Implementation Plan — JAX/Flax-NNX Correctness Review"
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-12
---

> **Superseded by**: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — PI pivot to sheeprl-direct (Option 1) shelves the dreamer-srl plan this review audits.

# dreamer-srl Implementation Plan — JAX/Flax-NNX Correctness Review

## Purpose

This is a JAX-correctness audit of [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — the senior-developer's plan to re-implement sheeprl's PyTorch DreamerV3 as a parallel JAX/Flax-NNX module under `src/algorithms/dreamer_srl/`. The user's goal is bit-identical algorithm semantics so a 5x5 food-only NoPred run reproduces sheeprl's ~500-step survival (the current in-house JAX Dreamer reaches ~106 on the same task). My scope is narrow: PRNG threading, vmap/scan axis hygiene, pytree mutation, NNX state semantics, replay-buffer sequence semantics, JIT recompilation triggers, target-critic aliasing, Configuration Protocol compliance. The algorithm and equation reviews are running in parallel under `professor-rl-bayesian-dl` and `math-reviewer`; I do not relitigate those.

I read the plan, the nine-file sheeprl walkthrough at `docs/project/references/sheeprl_dreamer_v3/`, the sheeprl source under `tmp/sheeprl/sheeprl/`, the existing NNX scaffolding at `src/models/dreamer_v3_nnx.py` / `src/models/dreamer_v3_trainer.py` / `src/models/dreamer_v3_util.py`, and `src/utils/config.py`'s `Config.get_mandatory` contract. The plan is largely correct on JAX idiom and the NNX decision is well-documented. There are however **eleven deviations** — eight blockers (🔴) and three concerns (🟡) — that need to be resolved before the developer starts. Most are missing translation steps where the plan defers to "the developer matches sheeprl exactly" without naming the JAX/NNX pattern. A few are arbitrary JAX idioms (in the plan's prose) that drift from sheeprl semantics.

## Verdict (one line)

**DEVIATIONS FOUND — 11 items (8 blockers, 3 concerns).** The plan is structurally sound but underspecified at the JAX-translation level on several load-bearing details. Fix the 🔴 items before implementation begins, then proceed.

---

## Conventions audit checklist

| Convention | Status | Notes |
|---|---|---|
| Pytree immutability (no in-place updates) | ✅ | Plan explicitly states "No mutation of pytrees in-place. Every `update_*` returns a new pytree" (§agent.py JAX/Flax discipline). |
| JIT recompilation triggers (static-vs-traced) | ✅ | Plan §agent.py discipline names every static shape that must be closed over at construction time; `Config.get_mandatory` path is correct. |
| vmap / scan axis discipline | 🟡 | Plan declares "operate on `[T, B, ...]` with scan over T, batch axis B implicit; avoid double-vmap" but does NOT name the carry signature for the dynamic-learning scan (see Deviation 1). |
| PRNG threading | 🟡 | Plan correctly says "every sampling site takes explicit `key`; never grab from `nnx.Rngs` at forward time." But does not name the sub-key split count per `one_train_step` invocation (see Deviation 5). |
| Target-critic aliasing | ✅ | Plan §agent.py discipline says target critic is a *second* `Critic` instance; explicitly warns "do NOT alias parameters across the two instances." |
| Sensor / observation breakdown sync | N/A | Plan is a fresh algorithm with `obs_keys=["state"]` only — no new sensor introduced. Verified the existing food-only env config wiring is not touched. |
| Configuration Protocol (no-fallback-defaults) | 🟡 | Plan §Risks #10 names every key as mandatory, BUT `defaults_from:` in `configs/models/dreamer_srl/01_food_only.yaml` is not a real `Config` mechanism (see Deviation 11). |
| `is_first` reset broadcast pattern | 🔴 | Plan says `jnp.where(is_first[..., None], learnable_init_state, current_state)`. Sheeprl actually uses `(1 - is_first) * current + is_first * initial` with `is_first` already shaped `[1, B, 1]` (see Deviation 3). |
| `lax.scan` over time axis | 🔴 | Plan does not specify the scan carry signature or how `is_first` enters the scan body (see Deviation 1). |
| Replay sequence semantics (straddle episode boundaries) | ✅ | Plan §buffers.py explicitly preserves "windows that DO NOT respect episode boundaries; trainer relies on stored `is_first` flag." |
| `Ratio` stays Python-side | ✅ | Plan §utils.py discipline correctly names this. Risk §2 flags the env-step counter ambiguity (legitimate). |

---

## Numbered deviation list

### 1. 🔴 Missing Translation — `lax.scan` carry signature for RSSM dynamic learning

**Plan location**: §Implementation Plan → File Changes → `agent.py` JAX/Flax discipline (line ~202) and §`train.py` `one_train_step` (line ~234).

**Walkthrough location**: `agent.md` Line 396 (`RSSM.dynamic` per-step contract); sheeprl source `dreamer_v3.py:205-244` (the Python for-loop over `sequence_length`).

**What's wrong**: The plan says "the standard pattern is `nnx.scan` (or pure-functional `lax.scan` after `nnx.split` / `nnx.merge`) over `T` with batch axis `B` implicit. Avoid mixing scan and vmap in the same dimension." But it never specifies the **carry signature** or how `RSSM.dynamic`'s five-output return becomes a scan body. Sheeprl's PyTorch loop carries `(recurrent_state, posterior)` and accumulates `(recurrent_states, priors_logits, posteriors, posteriors_logits)` per step. In JAX this must be:
- **Carry**: `(recurrent_state: [B, H_rec], posterior: [B, S, D])` — note posterior is shape `[B, S, D]` not flattened, because sheeprl's `dynamic` reshapes internally via `posterior.view(*posterior.shape[:-2], -1)` at line 428.
- **Scan input** (per-T slice): `(batch_action[t]: [B, A], embedded_obs[t]: [B, E], is_first[t]: [B, 1])`.
- **Scan output** (per-T): `(recurrent_state, posterior, posterior_logits, prior_logits)` — note `posterior` is collected unflattened with shape `[B, S, D]` for the loss to compute KL, then flattened to `[T, B, S*D]` for latent_states concatenation at sheeprl `dreamer_v3.py:245`.

**Proposed correction**: Add a paragraph to §agent.py spelling out the carry signature, OR add a section in §train.py for `one_train_step` that names it. Without this, the developer is likely to invent a different carry shape (e.g. flattened posterior) and silently regress KL computation. Pattern:

```python
def _rssm_dynamic_scan_body(carry, inputs):
    recurrent_state, posterior = carry  # [B, H_rec], [B, S, D]
    action, embed, is_first = inputs    # [B, A], [B, E], [B, 1]
    recurrent_state, posterior, prior, posterior_logits, prior_logits = \
        rssm.dynamic(posterior, recurrent_state, action, embed, is_first)
    new_carry = (recurrent_state, posterior)
    outputs = (recurrent_state, posterior, posterior_logits, prior_logits)
    return new_carry, outputs

(final_recurrent, final_posterior), (recurrent_states, posteriors, posteriors_logits, priors_logits) = \
    nnx.scan(_rssm_dynamic_scan_body, ...)(initial_carry, (batch_actions, embedded_obs, is_first))
```

### 2. 🔴 Wrong NNX Pattern — `LayerNormGRUCell` matmul structure differs from sheeprl

**Plan location**: §`agent.py` row `class LayerNormGRUCell` (line ~184).

**Walkthrough location**: `models.md` Line 351 (`LayerNormGRUCell.__init__`) and Line 370 (`.forward`). Sheeprl source `tmp/sheeprl/sheeprl/models/models.py:351-403`.

**What's wrong**: The plan's prose says "one projection from `[input, hidden]` to `3 * hidden`, then split into `(reset, update, cand_pre)`." Two issues:
- **Gate ordering is wrong.** Sheeprl `models.py:399` chunks as `reset, cand, update = torch.chunk(x, 3, -1)` — the order is `(reset, cand, update)`, not `(reset, update, cand)`. While the math is invariant to ordering (each gate's column slice is what matters), bit-identical replication of a pretrained sheeprl checkpoint (Checkpoint 8 in the plan) requires matching column order so a transferred `linear.weight` slice maps to the right gate. If the plan implements the gate ordering as `(reset, update, cand)`, the cross-framework parity check at Checkpoint 8 will fail to within 1e-4 because the column slices of the weight matrix line up with different gates.
- **The `bias`-trick is on `update`, not on `cand`.** Sheeprl: `update = torch.sigmoid(update - 1)` (the "keep old state" bias). The plan's prose says this correctly ("The fused-gate output of `update` is `sigmoid(update_raw - 1.0)`") — that part is fine. Flag this for the developer so they don't confuse with cascade fix #28 (which is the `tanh(reset * cand)` on the candidate).
- **The existing `LayerNormGRUCell` in `src/models/dreamer_v3_nnx.py:18-70` uses TWO Linears (`dense_ih`, `dense_hh`) and TWO LayerNorms (`ln_ih`, `ln_hh`) — that is the PyTorch `nn.GRUCell` convention, NOT sheeprl's.** Sheeprl uses **one** `nn.Linear(input_size + hidden_size, 3 * hidden_size)` and **one** `LayerNorm(3 * hidden_size)` applied to the joint projection. The plan explicitly says "Re-write the cell clean" (line 208), but the senior-developer's NNX cell definition needs to make this explicit so the developer doesn't pattern-match on `src/models/dreamer_v3_nnx.py` and reproduce the two-Linear structure. The two-Linear structure has a different parameter count and a different LayerNorm coupling and would silently break Checkpoint 8.

**Proposed correction**: Add to the `agent.py` table row for `LayerNormGRUCell`:

```
# Sheeprl-canonical fused-gate form:
self.linear = nnx.Linear(input_size + hidden_size, 3 * hidden_size, use_bias=True,
                         kernel_init=hafner_init(), rngs=rngs)
self.layer_norm = nnx.LayerNorm(3 * hidden_size, epsilon=1e-3, rngs=rngs)
# forward:
x = jnp.concatenate([hx, input], axis=-1)
x = self.layer_norm(self.linear(x))
reset, cand, update = jnp.split(x, 3, axis=-1)   # ORDER: reset, cand, update
reset = jax.nn.sigmoid(reset)
cand  = jnp.tanh(reset * cand)                    # cascade fix #28
update = jax.nn.sigmoid(update - 1.0)             # keep-old-state bias
hx_new = update * cand + (1 - update) * hx
```

Also explicitly tell the developer: "**Do NOT pattern-match on `src/models/dreamer_v3_nnx.py:LayerNormGRUCell` — that class uses a two-Linear / two-LayerNorm structure (PyTorch `nn.GRUCell` convention) that does NOT match sheeprl's fused-gate form. The dreamer-srl cell is one Linear + one LayerNorm.**"

### 3. 🔴 Missing Translation — `is_first` reset uses arithmetic-mask, not `jnp.where`

**Plan location**: §`agent.py` row `class RSSM` (line ~190): "The `is_first` handling at `agent.py:425-430` resets both posterior and recurrent state to the learnable initial state at first-step indices — replicate exactly." And §Your audit checklist context: "`jnp.where(is_first[..., None], learnable_init_state, current_state)` not a Python branch."

**Walkthrough location**: `agent.md` Line 396 ff., showing sheeprl's `dynamic` method. Sheeprl source `agent.py:423-429`:
```python
action = (1 - is_first) * action
initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
posterior = posterior.view(*posterior.shape[:-2], -1)
posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)
```

**What's wrong**: Sheeprl uses the **arithmetic mask** form, NOT `jnp.where`. The two are numerically equivalent when both sides are valid arrays, but they have different gradient implications:
- `jnp.where(cond, a, b)` propagates gradient through both `a` and `b` (the unselected branch's grad is computed but masked, except when wrapped in `lax.cond` which short-circuits).
- `(1 - mask) * a + mask * b` propagates gradient through both, weighted by the mask.

For bit-identical replication these should be the same, but the plan's claim that "the `[..., None]` broadcast trick is named (or equivalent)" misses that sheeprl's `is_first` arrives as `[T, B, 1]` from the buffer (see sheeprl `dreamer_v3.py:209`'s `data["is_first"][i : i + 1]` slicing — that pulls a `[1, B, 1]` slice with the trailing-singleton already baked in by the buffer schema; cf. plan §buffers.py "Required transition keys": `is_first (uint8 [1, n_envs, 1])`).

**Three concrete corrections needed**:
1. The plan must state that `is_first` is stored in the buffer with **shape `[T, B, 1]`** (trailing singleton), so the broadcast against `recurrent_state: [B, H_rec]` works without an extra `[..., None]`. The plan should NOT use `jnp.where` with `is_first[..., None]` — it should use sheeprl's arithmetic form.
2. The plan must name the **three** quantities reset: `action` (zeroed: `(1 - is_first) * action`), `recurrent_state`, AND `posterior`. The plan currently only mentions recurrent state and posterior.
3. The `posterior` must be reshaped from `[B, S, D]` to `[B, S*D]` BEFORE the mask is applied (sheeprl line 428: `posterior.view(*posterior.shape[:-2], -1)` then the mask). Otherwise the `[B, 1]` broadcast against `[B, S, D]` will not match shapes.

**Proposed correction**: Add a code block to the RSSM section of the plan:

```python
def dynamic(self, posterior, recurrent_state, action, embedded_obs, is_first):
    # is_first shape: [B, 1] (trailing singleton from buffer)
    # action shape:   [B, A]
    # posterior:      [B, S, D]  (categorical latent, not yet flattened)
    # recurrent_state:[B, H_rec]
    action = (1.0 - is_first) * action
    initial_recurrent, initial_posterior = self.get_initial_states((batch_size,))
    recurrent_state = (1.0 - is_first) * recurrent_state + is_first * initial_recurrent
    posterior = posterior.reshape(*posterior.shape[:-2], -1)                  # [B, S*D]
    initial_posterior = initial_posterior.reshape(*posterior.shape)           # [B, S*D]
    posterior = (1.0 - is_first) * posterior + is_first * initial_posterior   # [B, S*D]
    # ... rest of dynamic
```

### 4. 🔴 Missing Translation — `get_initial_states` requires `tanh` on the learnable parameter

**Plan location**: §`agent.py` row `class RSSM` (line ~190): "the learnable initial recurrent state is a `nn.Parameter` (sheeprl) → NNX `self.initial_recurrent_state = nnx.Param(jnp.zeros((recurrent_state_size,)))` declared in `__init__`. The recurrent state is initialised by applying `tanh` to the parameter (`agent.py:392`), broadcast over the batch — do not skip the tanh."

**Walkthrough location**: `agent.md` Line 391 and 593: `initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)`. Then `initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]`.

**What's wrong**: The plan correctly flags the `tanh`, but **does not flag the second step** — `initial_posterior` is computed by calling `self._transition(initial_recurrent_state, sample_state=False)` and **using its mode (the soft probabilities), not a sample**. The `sample_state=False` branch of `compute_stochastic_state` returns `.mode` (uniform-mix-applied softmax), not `.sample`. This matters because:
1. Computing the initial posterior via `sample` would consume a PRNG key in `get_initial_states`, but `get_initial_states` does NOT take a key in sheeprl — it must be deterministic.
2. The initial posterior is used at every `is_first=1` step (i.e. the start of every episode), so a non-deterministic initial posterior would break the `is_first` reset's determinism.

**Proposed correction**: Add to the RSSM row of the plan:

```
# get_initial_states: NO PRNG, deterministic.
def get_initial_states(self, batch_shape):
    initial_recurrent = jnp.tanh(self.initial_recurrent_state.value)         # [H_rec]
    initial_recurrent = jnp.broadcast_to(initial_recurrent, (*batch_shape, recurrent_state_size))
    # transition's mode (uniform-mix-applied softmax), NOT a sample. No key consumed.
    prior_logits = self.transition_model(initial_recurrent)
    prior_logits = self._uniform_mix(prior_logits)
    initial_posterior = jax.nn.softmax(prior_logits, axis=-1)  # mode, not sample
    return initial_recurrent, initial_posterior
```

This is also relevant to **Deviation 5** because if `get_initial_states` is mistakenly given a PRNG, the JIT trace of `one_train_step` will retrace on every call (key inputs are abstract values; nothing changes), but the developer may waste keys.

### 5. 🟡 PRNG Threading Error — sub-key split count for `one_train_step` is unspecified

**Plan location**: §`train.py` JAX/Flax discipline (line ~241): "every call to `one_train_step` consumes one PRNG key; split inside for (a) RSSM categorical sample, (b) Actor categorical sample, (c) any future stochastic op."

**Walkthrough location**: sheeprl source `dreamer_v3.py:48-358` is the entire `train` function; PRNG handling in PyTorch is implicit (CUDA RNG state).

**What's wrong**: The plan says "split inside for (a), (b), (c)" but does not name the **number of splits per gradient step**. Recall: dynamic-learning RSSM rollout samples one categorical per scan step (T splits). Imagination rollout samples one actor categorical per imagination step + one prior categorical per imagination step (H+1 splits each). Initial encoded obs is deterministic. So per `one_train_step`:
- T sub-keys for RSSM `compute_stochastic_state` during dynamic learning (carried inside scan via `lax.fori_loop`-style key generation, NOT split outside).
- H+1 sub-keys for prior categorical during imagination.
- H+1 sub-keys for actor categorical during imagination.

This is **2(H+1) + T = ~30 + 64 = ~94 sub-keys per gradient step** at the XS settings. The standard JAX pattern is to split the main key once into the number of distinct sampling sites and then `lax.scan` over T or H+1 with a key as part of the scan inputs (not the carry). The plan needs to name this explicitly so the developer doesn't:
1. Reuse the same key inside the scan body (would give all scan steps the same sample stream — Z3-class bug).
2. Take a Python-side loop and increment a counter (would force JIT retrace on every step).

**Proposed correction**: Add a `__PRNG threading discipline__` paragraph to §train.py JAX/Flax discipline:

```python
# Split inside one_train_step:
key, rssm_key, img_prior_key, img_actor_key = jax.random.split(key, 4)
# For the scan over T (RSSM dynamic learning):
rssm_keys = jax.random.split(rssm_key, T)   # shape [T, 2]; passed as scan input
# For the scan over H+1 (imagination):
img_prior_keys = jax.random.split(img_prior_key, H + 1)
img_actor_keys = jax.random.split(img_actor_key, H + 1)
# These keys-per-step arrays are scan INPUTS, not carries.
```

### 6. 🔴 Aliasing Risk — `Player` weight-tying claim is JAX-correct but checkpoint-reload protocol is unstated

**Plan location**: §`agent.py` row `class Player` (line ~195): "The PyTorch Player holds its own deep-copies of encoder/RSSM/actor with weights **aliased** to the trainable modules. In JAX/Flax there is no aliasing — params live in a separate pytree. The 'tied weights' pattern becomes: **always pass the current trainable params** into the Player's `__call__`. No deep-copy; no aliasing dance."

**Walkthrough location**: `agent.md` Line 596-693 (Player class); sheeprl source `agent.py:1184-1220` does `copy.deepcopy(world_model.encoder)` etc. then `player.actor.load_state_dict(actor.state_dict())` style aliasing.

**What's wrong**: The plan's JAX simplification is correct — there's no PyTorch deep-copy-then-alias dance in NNX because params are external pytree state. **BUT** the plan does not specify what `Player` *holds* in NNX. Two interpretations:
- (A) `Player` is a free-standing NNX module with its OWN params, updated by an explicit `nnx.update(player_state, source_state)` after every gradient step — this re-introduces aliasing risk and adds a sync cost.
- (B) `Player` is just a method namespace (`encode_obs`, `act`) that takes the trainable `WorldModel`/`Actor` as arguments. This is the simpler and correct JAX form.

The plan implies (B) ("always pass the current trainable params") but the file changes table says "`Player` (rollout-time wrapper that carries `(h, z)` state across env steps with weights tied to the trainable modules)" — suggesting (A).

This matters for **`build_state`** (`build_state` calls `state.player_initial_state(num_envs)` per plan line 423) and for **`collect_step`** (plan line 235: "Carry the `(h, z)` rollout state; encode obs; sample one-hot action via Actor"). If `Player` is interpretation (A), it needs its own pytree slot in `DreamerSrlState` AND a sync step after every gradient update. If (B), `(h, z)` lives in `DreamerSrlState` and `Player` is just a free function that closes over (or takes) the world model.

**Proposed correction**: The plan must explicitly pick (A) or (B). Recommendation: **(B)** — make `Player` a thin function-namespace, not a module. `(h, z)` lives in `DreamerSrlState` as `player_recurrent_state, player_stochastic_state`. The `collect_step` function takes the current `WorldModel` and `Actor` modules (which already hold the trainable params via NNX). No sync step needed; no aliasing.

```python
# In train.py:
@flax.struct.dataclass
class DreamerSrlState:
    world_model: nnx.GraphState         # holds WM params + nnx.Variable state
    actor: nnx.GraphState
    critic: nnx.GraphState
    target_critic: nnx.GraphState
    world_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    moments_state: MomentsState
    player_recurrent_state: jnp.ndarray  # [B, H_rec]
    player_stochastic_state: jnp.ndarray # [B, S, D]
    player_action: jnp.ndarray           # [B, A]
    train_step: jnp.ndarray
    env_step: jnp.ndarray
```

The `WorldModel`/`Actor` NNX modules ARE the trainable modules; the Player does not deep-copy them.

### 7. 🟡 Missing Translation — Polyak update cadence and `tau=1.0` first-step

**Plan location**: §`train.py` row `polyak_update` (line ~236): "First call uses `tau=1.0` (hard copy); subsequent use `tau=0.02` (sheeprl XS default)." And Checkpoint 7.

**Walkthrough location**: `dreamer_v3.md` Line 720-726 + the dreamer_v3.py source at `dreamer_v3.py:720-726`:
```python
if (cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq == 0):
    tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
    for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
        tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
```

**What's wrong**: Three problems with the plan:
1. **Polyak fires BEFORE `train()` in each gradient-step iteration.** Sheeprl applies polyak THEN calls `train(...)`. The plan does not name this ordering; the developer may put it after the critic gradient step (which would be wrong because the first iteration would then use a randomly-initialized target).
2. **The condition is on `cumulative_per_rank_gradient_steps`, not on `train_step` or `env_step`.** This is a gradient-step counter, not an env-step counter. The plan's `DreamerSrlState` pytree has `train_step` and `env_step` but `train_step` semantics are unspecified — does it count gradient steps or `one_train_step` calls? If multiple `one_train_step` calls fire per env step (replay_ratio=K>1), the counter must be incremented K times.
3. **The polyak formula is `tau * source + (1 - tau) * target`**, which matches the plan. Verified correct direction.

**Proposed correction**: Add to §`train.py`:

```
The top-level training loop applies Polyak update INSIDE the per-gradient-step
inner loop, BEFORE calling `one_train_step`. Like sheeprl `dreamer_v3.py:720-726`:

for _ in range(per_rank_gradient_steps):  # per env-step "outer" loop's K updates
    if state.train_step % target_update_freq == 0:
        tau = jnp.where(state.train_step == 0, 1.0, tau_default)
        target_params_new = jax.tree.map(
            lambda src, tgt: tau * src + (1 - tau) * tgt,
            nnx.state(critic), nnx.state(target_critic))
        nnx.update(target_critic, target_params_new)
    state, metrics = one_train_step(state, batch[i], key)
    state = state.replace(train_step=state.train_step + 1)
```

Also: `train_step` semantics must be **"cumulative gradient steps applied since training began"**, NOT "calls to `one_train_step` since restart" — name this in the `DreamerSrlState` docstring.

### 8. 🔴 Pytree Mutation — `Moments` in `dreamer_v3_util.py` mutates `nnx.Variable`, plan says `flax.struct.dataclass`

**Plan location**: §`utils.py` row `class Moments` (line ~142): "PyTorch uses `register_buffer` for mutable state. In our pure-functional setup, `Moments` becomes a `@flax.struct.dataclass` with `low: jnp.ndarray, high: jnp.ndarray`, and an `update(state, x) -> (new_state, (offset, invscale))` function."

**Walkthrough location**: `utils.md` Line 40 ff., showing sheeprl's `Moments` as a `nn.Module` with `register_buffer` for `low`/`high`.

**What's wrong**: The plan picks `flax.struct.dataclass` (pure pytree, immutable, `update` returns new pytree). The existing `src/models/dreamer_v3_util.py:Moments` uses `nnx.Variable` and mutates `self.low.value` in-place inside the `update` method (line 172-173). These are **two different patterns** and the developer needs to know which to use. The plan's choice (`flax.struct.dataclass`) is **better for JIT correctness** because it makes the state flow explicit through `one_train_step`'s return value. The existing `nnx.Variable` form requires `Moments` to be carried as part of the trainer NNX module and updated through `nnx.update` — workable but more fragile when nested in `one_train_step`.

Two concrete issues:
1. **Plan must explicitly say "do NOT pattern-match on `src/models/dreamer_v3_util.py:Moments`** — that class uses `nnx.Variable` mutation; dreamer-srl uses `flax.struct.dataclass` and returns a new state from `update`." Without this, the developer will copy the existing class.
2. **`Moments.update` return signature** must be `(new_moments_state, offset, invscale)`, not `(low, invscale)` (the existing class's return). The plan's signature `update(state, x) -> (new_state, (offset, invscale))` is correct; just make this explicit in the file table.

**Proposed correction**: Add to §`utils.py` Moments row:

```
# Sheeprl semantics, pure-functional:
@flax.struct.dataclass
class MomentsState:
    low: jnp.ndarray   # scalar; init 0.0
    high: jnp.ndarray  # scalar; init 0.0

def moments_update(state: MomentsState, x: jnp.ndarray, decay=0.99, p_low=0.05, p_high=0.95, max_=1.0
                   ) -> Tuple[MomentsState, jnp.ndarray, jnp.ndarray]:
    x_flat = x.ravel().astype(jnp.float32)
    low_p  = jnp.quantile(x_flat, p_low)
    high_p = jnp.quantile(x_flat, p_high)
    new_low  = decay * state.low  + (1 - decay) * low_p
    new_high = decay * state.high + (1 - decay) * high_p
    invscale = jnp.maximum(1.0 / max_, new_high - new_low)
    return MomentsState(low=new_low, high=new_high), new_low, invscale

# DO NOT pattern-match on src/models/dreamer_v3_util.py:Moments — that class uses
# nnx.Variable in-place mutation; dreamer-srl uses pure-functional return.
```

### 9. 🟡 Configuration Protocol Violation — `defaults_from:` in `01_food_only.yaml` is not a real Config mechanism

**Plan location**: §`configs/models/dreamer_srl/01_food_only.yaml` (line ~344): "`defaults_from: configs/experiment/dreamer_curriculum/01_food_only.yaml`". Plan flag at line 367 admits "the project does not have a `defaults_from`-style merge mechanism in `src/utils/config.py` (only `Config.merge(other)`)."

**Walkthrough location**: N/A (this is project-side).

**What's wrong**: The plan invents a `defaults_from:` YAML key that doesn't exist in `src/utils/config.py`. The plan acknowledges this and says "Step 0 of implementation is to confirm the right merge pattern by reading how `configs/experiment/dreamer_curriculum/01_food_only.yaml` is currently loaded." But this leaves an implementation-time decision: should the developer (a) add a new `defaults_from` to `Config`, or (b) drop it and use two `--config` flags, or (c) add a small loader helper? Without a decision, the developer will pick whichever is fastest, which may introduce a one-off pattern that drifts from how the rest of the project loads configs.

**Proposed correction**: The senior-developer should decide between (a), (b), or (c) **before** the developer starts. Recommendation: **(c)** add a 5-line helper in the new top-level branch that does `cfg = Config.load_yaml(exp_path); cfg.merge(Config.load_yaml(cfg.get_mandatory('agent_config')))`. No new YAML-level mechanism; no change to `src/utils/config.py`.

### 10. 🟡 Configuration Protocol Concern — `target_update_freq` config key naming

**Plan location**: `configs/models/dreamer_srl/agent_xs.yaml` (line ~318): `per_rank_target_network_update_freq: 1`. Plan §Risks §10: "every new key in `configs/models/dreamer_srl/agent_xs.yaml` must be read via `Config.get_mandatory`."

**Walkthrough location**: sheeprl `dreamer_v3.yaml:151`.

**What's wrong**: The plan uses sheeprl's exact key name `per_rank_target_network_update_freq` — a sheeprl-specific name that refers to per-rank under distributed training. We are single-device. The "per_rank" prefix is misleading and will confuse future maintainers. Two options:
- Keep `per_rank_target_network_update_freq` for bit-identical replication (good for the `jzgkcep4` analyzer to read both projects with the same key).
- Rename to `target_update_freq` (cleaner; more JAX-idiomatic).

The plan doesn't pick one. The rest of the config has the same issue (`per_rank_sequence_length`, `per_rank_batch_size`, `per_rank_pretrain_steps`).

**Proposed correction**: Pick one convention and apply it everywhere. Recommendation: **keep sheeprl names verbatim** for parity. Trade-off accepted: future maintainers will read a comment in the YAML header explaining "per_rank_*" is sheeprl's distributed-training prefix; we run single-device so per_rank=global.

### 11. 🟢 Arbitrary JAX Idiom — `discount` computation must use `lax.stop_gradient`, not `jax.lax.stop_gradient` wrapping the full block

**Plan location**: §`train.py` row `one_train_step` (line ~234): "Hazard: the `discount = cumprod(continues * gamma, axis=0) / gamma` computation is the unrolling-aware weighting; gradient must NOT flow through it (`with torch.no_grad()` in sheeprl `dreamer_v3.py:259`)."

**Walkthrough location**: `dreamer_v3.md` Line 292-293; sheeprl `dreamer_v3.py:292-293`.

**What's wrong**: The plan correctly identifies the gradient-isolation requirement but does not name the JAX form. `with torch.no_grad()` blocks ALL operations inside from contributing to the graph. The JAX equivalent for a single array is `jax.lax.stop_gradient(discount)` AFTER the cumprod. Applied to the whole cumprod block:

```python
discount = jax.lax.stop_gradient(jnp.cumprod(continues * gamma, axis=0) / gamma)
```

The plan should name this so the developer doesn't try `with jax.disable_jit()` or some other wrong pattern.

**Proposed correction**: Replace the prose in the `one_train_step` row with the explicit JAX form.

---

## Auxiliary observations (not numbered deviations; verification notes)

- **`Ratio` env-step counter** (Plan Risks §2): the plan correctly flags this as open. The fractional-debt accumulator (`_prev += repeats / _ratio`) is in `src/models/dreamer_v3_util.py:Ratio` already, line-for-line identical to sheeprl. The dreamer-srl `Ratio` should NOT import from the existing module (per plan line 152) but should replicate verbatim — keeping the existing per-env-step counter convention `train.py:789`. Verified the existing call passes `policy_step` (global env step). This is consistent with sheeprl's `ratio(ratio_steps / world_size)` at `dreamer_v3.py:708` where `world_size=1` for us.

- **`hafner_init` constant `0.87962566103423978`** (Plan Risks §5): existing `src/models/dreamer_v3_util.py:hafner_init` uses `scale=0.8796` (truncated to 4 decimals). Sheeprl's `utils.py:149` uses `0.87962566103423978` (full precision). For bit-identical replication, dreamer-srl `init_weights` must use the full-precision constant, NOT the truncated one. **Add this to the plan's `utils.py` row** as a one-line note: "Use `0.87962566103423978`, not the existing `0.8796`."

- **Plan §agent.py "actor MLP body gets `init_weights` Hafner-trunc-normal"** (line 193) plus sheeprl `agent.py:1168` `actor.apply(init_weights)`: verified. `actor.mlp_heads.apply(uniform_init_weights(1.0))` at sheeprl `agent.py:1171` overrides the actor heads' final Linear AFTER Hafner-init. The plan handles this correctly with `uniform_init_weights(1.0)` (NOT zero) — verified at plan §agent.py build_agent row line 197.

- **Plan §loss.py `kl_balanced` form**: the plan correctly says `dyn_loss = kl(post.detach, prior)` and `repr_loss = kl(post, prior.detach)`. Verified sheeprl `loss.py` does this. The `jax.lax.stop_gradient` translation is correct.

- **Sheeprl `Moments` default `max_=1e8`** but **YAML default `max: 1.0`** — the YAML's `1.0` is what's used; verified.

- **The plan says `learning_starts: 1024`** — this is `total env steps before the first gradient update fires`. In `Ratio.__call__`, the first call sees `step >= learning_starts` and returns `int((step - 0) * ratio) = step * ratio` updates. The plan correctly preserves this. Verified sheeprl `dreamer_v3.py:706` checks `iter_num >= learning_starts` against the iteration count (where `iter_num = policy_step / policy_steps_per_iter` and `policy_steps_per_iter = num_envs * action_repeat`). For us `num_envs=4`, `action_repeat=1`, so `learning_starts=1024` env steps = 256 iterations. **The plan should clarify whether `learning_starts` is counted in env steps or iterations** — sheeprl counts iterations.

---

## Sign-off

Reviewed by: code-reviewer

**Verdict**: DEVIATIONS FOUND — 11 items (8 🔴 blockers, 3 🟡 concerns).

**Blockers** (must resolve before implementation):
1. Missing `lax.scan` carry signature for RSSM dynamic learning.
2. Wrong `LayerNormGRUCell` matmul structure (one Linear + one LayerNorm, not two of each; gate order `(reset, cand, update)`).
3. `is_first` reset uses arithmetic mask, not `jnp.where`; three quantities reset; posterior must be reshape-flattened first.
4. `get_initial_states` must use transition mode (not sample); no PRNG.
6. `Player` interpretation (A vs B) must be picked explicitly — recommend B (free function, no aliased copies).
8. `Moments` pattern: `flax.struct.dataclass` form, explicit "do NOT copy from existing `dreamer_v3_util.py:Moments`".

**Concerns** (fixable inline by developer with senior-dev awareness):
5. PRNG sub-key count per `one_train_step` invocation.
7. Polyak update cadence (BEFORE `train`; `train_step` semantics = cumulative gradient steps).
9. `defaults_from:` invented YAML key — pick (a), (b), or (c) before start.
10. `per_rank_*` config key naming convention — pick verbatim sheeprl or rename, apply uniformly.
11. `lax.stop_gradient` form for `discount` — name the JAX expression.

The plan is structurally sound, the NNX decision is well-documented, and the file decomposition mirrors sheeprl correctly. With the 6 blockers addressed, the developer has a clean blueprint and the bit-identity gate (Checkpoint 8) is achievable to 1e-4. Without them, expect Checkpoint 8 to fail on at least the GRU cell and the RSSM initial-state determinism.
