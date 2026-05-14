---
title: "v2-CP8 — dreamer-srl wrapper modules re-audit (H1 hypothesis-locus)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
---

# v2-CP8 — dreamer-srl wrapper modules re-audit

## Verdict (plain-language entry point)

**What this review is.** A fresh line-by-line re-audit of the wrapper-class neural-network modules
in [`src/algorithms/dreamer_srl/agent.py`](../../src/algorithms/dreamer_srl/agent.py) — the
**Encoder**, **Decoder**, **ContinueHead**, **FullMLPHead** (used for reward / continue / critic),
**Actor**, **WorldModel** composite, and the **`build_agent`** factory — against the canonical
sheeprl analogs at `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py`, pinned at sheeprl commit
`33b6366`. **No v1 reviewer output is trusted** per the 2026-05-14 binding directive captured in
[`SHEEPRL_REFERENCE_AUDIT.md`](../../docs/develop/active/dreamer_srl_v2/SHEEPRL_REFERENCE_AUDIT.md) §1.
The audit is read-only — findings are flagged; no code is modified.

**Plain-language context for H1.** The 3-seed parity launch on 2026-05-14 collapsed the agent to
the random-policy floor (~104 survival steps versus a ~500-step sheeprl baseline). The leading
hypothesis (**H1**) blames a gradient-side bug in the *actor objective* — most likely a mis-port
of REINFORCE (the policy-gradient term `log_prob(action) × advantage` that drives the actor toward
actions that beat the baseline), the `unimix=0.01` mixture (which blends 1 % uniform into the
categorical action distribution to keep exploration alive), the entropy bonus, or a
`stop_gradient` site (the JAX equivalent of PyTorch's `.detach()`). The Actor wrapper class and the
imagination → actor-loss data flow are therefore the H1 epicenter.

**Headline verdict.** **FAIL — one blocker-class H1 finding in the Actor / actor-loss data flow,
plus two MED-class concerns.**

The dreamer-srl REINFORCE term evaluates `log_prob(a)` at a **freshly-sampled action** drawn
inside the actor-loss function, not at the **rollout action** that produced the lambda-return.
Sheeprl evaluates `p_new.log_prob(action_old.detach())` — distribution from the loss-time re-run,
action from the imagination rollout. Dreamer-srl evaluates `p_new.log_prob(action_new.detach())` —
distribution AND action both from the loss-time re-run, with the loss-time action sampled afresh
inside the actor module from a deterministic constant PRNG seed `jax.random.PRNGKey(0)`. The
rollout actions (collected in `WorldModel.imagine()`) are computed and returned in
`imag_outputs["imagined_actions"]` but then **discarded** — only `imagined_latents` is consumed
downstream. This is a substantive divergence that breaks the per-sample REINFORCE gradient: the
action being log-prob'd is no longer the action that earned the advantage, so the gradient pushes
the policy toward an action that was not actually taken in the trajectory that produced the
return. Combined with the constant `PRNGKey(0)`, the loss-time action is deterministic across
gradient steps but still wrong with respect to the rollout.

The two MED findings: (a) the actor-loss function uses a hard-coded `jax.random.PRNGKey(0)` for
the actor's straight-through Gumbel sampling — even if the action-mismatch (above) is fixed by
threading the rollout action in, the constant key is itself a PRNG-discipline regression; (b)
entropy in `Actor.__call__` uses `softmax(logits) * log(softmax(logits) + 1e-8)` instead of the
numerically-stable `softmax(logits) * log_softmax(logits)` — algebraically equivalent at unimix
probability floor (`0.01 / D` is well above `1e-8`), but a needless drift from sheeprl's
`Categorical.entropy()`.

The remaining wrappers — **Encoder**, **Decoder**, **ContinueHead**, **FullMLPHead**, **WorldModel
composite**, **`build_agent`** — pass the structural / trainability / gradient-flow audit.
Every weight is registered as `nnx.Param`, every body Linear has `use_bias=False` matching
sheeprl's `layer_args={"bias": layer_norm_cls == nn.Identity}` under the default LayerNorm config,
every layer-norm uses `eps=1e-3` (not the nnx default `1e-6`), every output head's zero-init
(reward + critic) and Hafner-scale-1 init (continue + decoder + actor) matches the sheeprl
`build_agent` two-phase init pattern at L1167–L1180, and the zero-init regression assertions at
`build_agent` L2108–L2115 are in place.

---

## 1. The Actor module (H1 epicenter)

### 1.1 Forward-pass structure — PASS

Side-by-side comparison against sheeprl `agent.py:L729-L845`:

| Sheeprl detail | Dreamer-srl detail | Match? |
|---|---|---|
| MLP body: `MLP(input_dims=latent, output_dim=None, hidden_sizes=[dense_units]*mlp_layers, layer_args={"bias": layer_norm_cls == nn.Identity}, norm_layer=LayerNorm, norm_args={"eps": 1e-3, "normalized_shape": dense_units}, activation=SiLU)` (L761-L770) | `mlp_layers` × `(Linear(use_bias=False) → LayerNorm(eps=1e-3) → SiLU)` (agent.py:L1448-L1455) | YES |
| Output heads: `mlp_heads = nn.ModuleList([nn.Linear(dense_units, action_dim) for action_dim in actions_dim])` (L774) — multi-discrete supported | `self.output_linear = nnx.Linear(dense_units, action_dim, use_bias=True)` (agent.py:L1459) — single discrete head only | YES (gridworld is single-discrete; the simplification is acceptable) |
| Hafner init: `actor.apply(init_weights)` (L1167) on body Linears + `actor.mlp_heads.apply(uniform_init_weights(1.0))` (L1171) on output head | `init_weights(I, O, k)` applied to each body Linear's kernel (agent.py:L1463-L1466); `uniform_init_weights(1.0, I, O, k)` applied to output_linear's kernel (agent.py:L1468-L1470) | YES |
| `_uniform_mix`: `probs = softmax(logits); probs = (1-unimix)*probs + unimix*ones/D; logits = probs_to_logits(probs)` (L839-L845) where `probs_to_logits` for Categorical = `log(probs)` | `probs = softmax(logits); probs = (1-unimix)*probs + unimix*ones/D; logits = log(probs)` (agent.py:L1477-L1481) | YES |

The cascade-fix-#27 zero-init for the **critic** output linear is enforced via `FullMLPHead(zero_init_output=True)` at `build_agent` L2090, and there is an explicit assertion at L2106-L2115 that fails build if the critic output kernel is non-zero. The Actor's output head correctly uses `uniform_init_weights(1.0)` (NOT zero-init) per sheeprl L1171 — there is a one-line comment chain in v1 reviews that mis-claimed the Actor's final layer should be zero-init; this review confirms the dreamer-srl implementation is correct per sheeprl.

### 1.2 Forward `__call__` — sampler, sg-placement, log-prob, entropy

This is the **H1-critical region**. Findings split by sub-question.

#### 1.2.1 Distribution / sampler type — PASS (with caveat)

| Sheeprl | Dreamer-srl |
|---|---|
| `OneHotCategoricalStraightThrough(logits=mixed_logits)` with `dist.rsample()` (L832-L834) — PyTorch's built-in straight-through estimator; forward = one-hot of argmax of `softmax(logits) + gumbel_noise`, backward = gradient through softmax | Custom JAX STE: `gumbel_noise = jax.random.gumbel(key, shape=logits.shape); perturbed = logits + gumbel_noise; hard = one_hot(argmax(perturbed)); soft = softmax(logits); actions = hard - sg(soft) + soft` (agent.py:L1520-L1526) |

**Verdict**: equivalent at infinite precision. The dreamer-srl construction `hard − sg(soft) + soft` IS the standard Hafner straight-through gradient — the forward pass returns `hard`, the backward pass propagates `∂soft/∂logits`. The PyTorch `OneHotCategoricalStraightThrough.rsample` does the same operation internally. **This is one of the cross-substrate boundaries explicitly enumerated in
[`SHEEPRL_REFERENCE_AUDIT.md`](../../docs/develop/active/dreamer_srl_v2/SHEEPRL_REFERENCE_AUDIT.md) §8 (D-009 / D-002 class)** — forward samples differ because of the PRNG mechanism, but backward gradient w.r.t. logits matches within 1e-5.

Caveat: sheeprl multiplies `gumbel_noise` and computes argmax INSIDE `rsample()` from a global PyTorch RNG; dreamer-srl uses `jax.random.gumbel(key, ...)` with the supplied key. This is the correct JAX idiom but means the per-step actions during `imagine()` are deterministic given the PRNG key — that is sheeprl-equivalent in semantics.

#### 1.2.2 The `_uniform_mix` site — PASS

Both sheeprl L839-L845 and dreamer-srl `Actor._uniform_mix` (agent.py:L1472-L1482) apply the mix
in **probability space** (`softmax → mix → log → logits`), NOT in logit space. Gradient flows
through `log(softmax(logits))` which has a different Jacobian than logit-space addition. The
RSSM's `_uniform_mix` (agent.py:L819-L856) also matches. **Both implementations use the
correct probability-space mix.**

The `probs_to_logits` numerical-floor concern from
[`SHEEPRL_REFERENCE_AUDIT.md`](../../docs/develop/active/dreamer_srl_v2/SHEEPRL_REFERENCE_AUDIT.md) §8 is moot in practice: at post-`unimix` the minimum
probability is `0.01 / D` (typically ~2e-3 for D=5 grid-world actions), well above `1e-8` /
`finfo(float32).tiny` ~1.18e-38. The dreamer-srl `jnp.log(probs)` at agent.py:L1481 omits an
explicit clip; sheeprl's `probs_to_logits` clips at `torch.finfo(probs.dtype).tiny`. Neither
clip is reachable in practice — this is a non-issue.

#### 1.2.3 BLOCKER — `log_prob(action)` is evaluated at the wrong action

This is the **H1 epicenter finding**.

**Sheeprl** (`dreamer_v3.py:L273-L291`):

```python
policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]      # L273
# ...
objective = (
    torch.stack(
        [
            p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]                    # L286
            for p, imgnd_act in zip(policies, torch.split(imagined_actions, ...))
        ],
        dim=-1,
    ).sum(dim=-1)
    * advantage.detach()                                                         # L291
)
```

- `policies` are the **distributions returned by the actor re-run at loss-time** on
  detached imagined trajectories — these carry the gradient into the actor params.
- `imgnd_act` is the **action sampled during the imagination rollout** at L219/L240
  (the rollout-time `actor(...)[0]`) — detached at log-prob time, since it serves as
  the *target* for the log-prob computation, not a path of gradient flow.
- The REINFORCE term is `p_NEW.log_prob(action_OLD.detach()) × sg(advantage)` —
  evaluated at the rollout action, with gradient through the NEW logits only.

**Dreamer-srl** (`agent.py:Actor.__call__` L1484-L1542 + `train.py:make_train_step` L836-L867):

In `Actor.__call__` the log-prob is computed **internally** from the actor's freshly-sampled
action:

```python
gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
perturbed = logits + gumbel_noise
hard_indices = jnp.argmax(perturbed, axis=-1)
hard = jax.nn.one_hot(hard_indices, self.action_dim)
soft = jax.nn.softmax(logits, axis=-1)
actions = hard - jax.lax.stop_gradient(soft) + soft               # [..., action_dim]

log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
sg_actions = jax.lax.stop_gradient(actions)
log_probs = jnp.sum(sg_actions * log_softmax_logits, axis=-1, keepdims=True)   # [..., 1]
```

Then the `actor_loss_fn` re-runs the actor inside the gradient context (`train.py:L840-L864`):

```python
actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)      # L837 — CONSTANT seed
sg_latents = jax.lax.stop_gradient(imagined_latents)                # L838

def actor_loss_fn(actor_module):
    all_log_probs = []
    all_entropies = []
    for h in range(H_plus_1):
        _, lp_h, ent_h = actor_module(sg_latents[h], actor_keys[h])   # L845 — fresh sample
        all_log_probs.append(lp_h)
        all_entropies.append(ent_h)
    # ... use log_probs in compute_actor_objective
```

Meanwhile `WorldModel.imagine()` does collect rollout-time actions, log-probs, and entropies in
`imag_outputs["imagined_actions"]`, `imag_outputs["imagined_log_probs"]`,
`imag_outputs["imagined_entropies"]` (`agent.py:L1755-L1808`) — but **a grep over
`train.py` confirms only `imag_outputs["imagined_latents"]` is consumed** (`train.py:L788` is
the only `imag_outputs[...]` access). The rollout actions are computed and immediately
discarded.

**Net effect**: the REINFORCE estimator `log_probs * sg(advantage)` in
`compute_actor_objective` (`train.py:L591`) evaluates `log_prob(a_freshly_sampled_at_loss_time)`,
not `log_prob(a_taken_during_rollout)`. Mathematically:

- **Sheeprl form**: `∇θ [ log p_θ(a_rollout) · A(s_rollout, a_rollout) ]` — the canonical REINFORCE
  estimator, where the advantage `A` was computed under the actions that were actually taken.
- **Dreamer-srl form**: `∇θ [ log p_θ(a_freshsample) · A(s_rollout, a_rollout) ]` — the advantage
  was computed under `a_rollout`, but the log-prob being incremented is for an action
  `a_freshsample` that may or may not equal `a_rollout`.

If the policy is approximately deterministic at the rollout-time latents (high probability mass on
one action), `a_freshsample ≈ a_rollout` often and the bug is silent. If the policy is exploratory
(uniform-ish under unimix), the freshly-sampled action differs from the rollout action ~`(D-1)/D`
of the time, and the gradient is **systematically wrong** — pushing the policy toward
*counterfactual* actions weighted by *factual* advantages. With `PRNGKey(0)` held constant
across gradient steps, the same wrong actions are reinforced repeatedly.

**Sheeprl-faithful fix sketch** (the developer agent should implement):

- Plumb `imag_outputs["imagined_actions"]` from `WorldModel.imagine()` through `train.py` into
  `actor_loss_fn`.
- In `actor_loss_fn`, instead of computing `log_probs` from the fresh sample, compute it
  directly: `log_probs = jnp.sum(sg(imagined_actions[h]) * log_softmax(logits_h), axis=-1)`,
  where `logits_h` comes from a re-run that returns the **distribution logits** (not a freshly
  sampled action).
- Equivalently, modify `Actor.__call__` to support a `target_action` argument: when provided,
  skip the Gumbel sample and compute `log_prob` against the supplied action — matching the
  sheeprl `actor(s)[1].log_prob(imgnd_act.detach())` pattern.

**Severity**: **BLOCKER** — this is the canonical H1 hypothesis-locus. The math is wrong, not
just numerically drifted.

#### 1.2.4 CONCERN — constant `PRNGKey(0)` for the actor re-run

Independent of finding 1.2.3, the line `actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)`
at `train.py:L837` is a PRNG-discipline regression:

- The actor's straight-through Gumbel-softmax sampler at `Actor.__call__` L1520 consumes the key.
- A constant `PRNGKey(0)` makes the loss-time action deterministic across gradient steps.
- Even if 1.2.3 is fixed (so log-prob is evaluated against the rollout action, not the
  loss-time sample), the entropy and the actor's softmax intermediates would be deterministic
  in a way that diverges from sheeprl's per-call RNG draw.
- The fix is trivial: thread the train-step key through into `actor_loss_fn` and split it
  per timestep, the same way `k_imag` is split at L785 for `world_model.imagine`.

**Severity**: **MED** — masked by 1.2.3 (the action is wrong regardless of the key), but
still a PRNG hygiene problem that would surface if 1.2.3 alone is patched.

#### 1.2.5 CONCERN — entropy numerical form

Dreamer-srl `Actor.__call__` L1539-L1540:

```python
probs = jax.nn.softmax(logits, axis=-1)
entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
```

Sheeprl `Categorical.entropy()` uses `-sum(probs * log_softmax(logits))` (the
PyTorch-numerically-stable form). The two are algebraically equivalent at infinite precision
when `probs > 1e-8`. At unimix=0.01 and D=5 (gridworld) the min prob = `0.01/5 = 0.002`, well
above `1e-8` — the `+1e-8` is therefore inert in practice. But:

- The drift is one of the cross-substrate boundaries unnecessary to incur — using
  `-(probs * log_softmax(logits)).sum(-1)` would be sheeprl-bitwise-identical (modulo softmax
  numerical drift, which is a class-D boundary already approved).
- Gradient w.r.t. logits differs by O(1e-8) at the entropy term — negligible but a needless
  drift from `[`SHEEPRL_REFERENCE_AUDIT.md`](../../docs/develop/active/dreamer_srl_v2/SHEEPRL_REFERENCE_AUDIT.md) §8`.

**Severity**: **NIT** (numerically negligible) — flagged for completeness.

### 1.3 Trainability — PASS

Verified live via `nnx.split(actor)` (interactive REPL during this audit, not memorised) that
the Actor exposes `hidden_linears`, `hidden_norms`, `output_linear` in the State pytree — every
weight is a registered `nnx.Param` and reachable by `nnx.value_and_grad(actor_loss_fn)(actor)`
at `train.py:L866`. No `nnx.Variable(...)` shadow weights; no `jax.lax.stop_gradient` in the
forward pass that would orphan a body Linear from the optimizer.

---

## 2. Encoder (MLPEncoder) — PASS

| Sheeprl `MLPEncoder` (`agent.py:L100-L151`) | Dreamer-srl `MLPEncoder` (`agent.py:L1115-L1195`) | Match? |
|---|---|---|
| `MLP(input_dim, output_dim=None, hidden_sizes=[dense_units]*mlp_layers, layer_args={"bias": layer_norm_cls == nn.Identity}, norm_layer=LayerNorm, norm_args={"eps": 1e-3, ...}, activation=SiLU)` (L137-L145) | `mlp_layers` × `(Linear(use_bias=False) → LayerNorm(eps=1e-3) → SiLU)` (L1163-L1170) | YES |
| `output_dim = dense_units` (L146) | `self.output_dim = dense_units` (L1154) | YES |
| `symlog(obs[k])` if `symlog_inputs` (L150) | `x = _symlog(obs)` (L1190) | YES |
| `init_weights(m)` applied at `build_agent` L1129 (`encoder.apply(init_weights)`) | `init_weights(I, O, k)` per body Linear at L1174-L1178 | YES |

**Caveat**: sheeprl supports `layer_norm_cls = nn.Identity` (bias-on case); dreamer-srl
hard-codes `use_bias=False`. Under the food-task config, sheeprl uses LayerNorm so the bias is
off — match. If a future config switches to Identity, dreamer-srl would not match. Out of scope
for the H1 / parity-failure question.

---

## 3. Decoder (MLPDecoder) — PASS

| Sheeprl `MLPDecoder` (`agent.py:L229-L278`) | Dreamer-srl `MLPDecoder` (`agent.py:L1203-L1290`) | Match? |
|---|---|---|
| Body: `MLP(latent, output_dim=None, hidden_sizes=[dense_units]*mlp_layers, ...)` (L265-L273) | `mlp_layers` × `(Linear(use_bias=False) → LayerNorm(eps=1e-3) → SiLU)` (L1252-L1259) | YES |
| Head: `heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in output_dims])` (L274) — multi-output supported | `self.output_head = nnx.Linear(dense_units, obs_dim, use_bias=True)` (L1262) — single-key only | YES (gridworld is single mlp_key) |
| `init_weights` on body + `uniform_init_weights(1.0)` on heads (L1129 + L1178) | `init_weights(I,O,k)` on body + `uniform_init_weights(1.0, I, O, k)` on output_head (L1267-L1274) | YES |

The decoder's output distribution (`SymlogDistribution` per sheeprl `loss.py`) is constructed
outside the decoder module — handled in dreamer-srl `train.py` `wm_loss_fn` (L709-L711:
`-0.5 * sum((symlog(pred) - symlog(target))**2)`). This is the **SymlogDist** form per
[`SHEEPRL_REFERENCE_AUDIT.md`](../../docs/develop/active/dreamer_srl_v2/SHEEPRL_REFERENCE_AUDIT.md) §3 — out of scope for this wrapper audit.

---

## 4. ContinueHead — PASS (but redundant with FullMLPHead)

`ContinueHead` (`agent.py:L1299-L1377`) and `FullMLPHead(out_dim=1, zero_init_output=False)`
(`agent.py:L1817-L1906`) are **functionally identical**. `build_agent` uses `FullMLPHead` for
the continue model at L2055-L2062, not the `ContinueHead` class. The `ContinueHead` class is
defined but never instantiated by `build_agent` — it's dead code. Verified by grep:

```bash
grep -n "ContinueHead(" /media/nas01/projects/Interoceptive-AI/grid_world_pain/src/algorithms/dreamer_srl/agent.py
# matches only the class def at L1299
```

The actually-instantiated continue head (`FullMLPHead` at `build_agent` L2055) correctly applies
`uniform_init_weights(1.0)` to the output linear (matching sheeprl L1176:
`world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))`). It outputs a single
logit per batch element; the `BernoulliSafeMode` / `IndependentBernoulli` wrap happens in
`train.py` (L719: `IndependentBernoulli(wm_outputs["continue_logits"])`). The `BernoulliSafeMode`
mode-handling per §S5 splice is in `compute_imagined_returns` — out of this audit's scope.

**Severity**: dead-code class is a NIT — flagged for completeness; not a correctness issue.

---

## 5. FullMLPHead — PASS

The shared head class used for **reward_model** (zero_init_output=True), **continue_model**
(zero_init_output=False), **critic** (zero_init_output=True), and **target_critic**
(zero_init_output=True). The two init branches at L1882-L1891:

```python
if zero_init_output:
    self.output_linear.kernel = nnx.Param(uniform_init_weights(0.0, I, O, k))
    self.output_linear.bias = nnx.Param(jnp.zeros((O,), dtype=jnp.float32))
else:
    self.output_linear.kernel = nnx.Param(uniform_init_weights(1.0, I, O, k))
    # bias left at nnx.Linear default (zeros)
```

Sheeprl's `uniform_init_weights` (`utils.py:L170-L186`) ALWAYS sets `m.bias.data.fill_(0.0)` for
Linear, both for scale=0.0 and scale=1.0. Dreamer-srl explicitly zeros the bias only in the
zero_init_output branch. Verified that `nnx.Linear`'s default bias initializer is
`flax.nnx.initializers.zeros_init` — so bias is also zero in the non-zero-init branch. **Match in
practice.**

The zero-init enforcement assertions at `build_agent` L2106-L2115 are present and active —
**any future regression on the cascade-fix-#27 zero-init for reward or critic output linears
will fail at agent construction**, not silently train.

---

## 6. WorldModel composite — PASS (gradient-flow verified)

`WorldModel` (`agent.py:L1551-L1808`) wraps encoder + RSSM + decoder + reward + continue. The
audit checks the gradient flow through `observe()` (training-time WM rollout) and `imagine()`
(behaviour-learning rollout).

### 6.1 `observe()` gradient flow

`observe()` (L1603-L1706) runs:

1. `embedded_obs = jax.vmap(self.encoder)(obs.reshape(T*B, -1)).reshape(T, B, -1)` — encoder
   gradient flows from WM loss.
2. Python loop of T steps calling `self.rssm.dynamic(...)` — RSSM gradient flows from WM
   loss.
3. `reconstructed_obs = jax.vmap(self.decoder)(latent_flat)` — decoder gradient flows from WM
   obs-reconstruction loss.
4. `reward_logits = jax.vmap(self.reward_model)(latent_flat)` — reward-head gradient flows from
   WM reward loss.
5. `continue_logits = jax.vmap(self.continue_model)(latent_flat)` — continue-head gradient
   flows from WM continue loss.

No `jax.lax.stop_gradient` inside `observe()`. The `wm_loss_fn` at `train.py:L700-L765` does the
appropriate stop_gradient on `log_post` / `log_prior` to implement the §S8 KL split
(`dyn_loss` stops gradient on posterior; `repr_loss` stops gradient on prior — sheeprl
`loss.py:L40-L62`). **Match.**

### 6.2 `imagine()` gradient flow

`imagine()` (L1708-L1808) is called **outside** any gradient context in `train.py:L786`. The
returned `imagined_latents` are then `jax.lax.stop_gradient(...)`'d inside `actor_loss_fn` at
L838 before being fed back into the loss-time actor re-run.

Internal structure:

1. `actor(init_latent, k_act)` at L1755 — actor returns actions / log_probs / entropy.
2. Loop H times: `recurrent_input = cat([prior_flat, action], ...); recurrent_feat =
   recurrent_mlp_linear(...) → recurrent_mlp_norm(...) → SiLU; recurrent_state =
   gru_cell(recurrent_feat, recurrent_state); prior_logits, imagined_prior = rssm._transition(...);
   new_latent = cat([prior_flat_new, recurrent_state]); actor(new_latent, k_act)`.

The RSSM imagination step in dreamer-srl uses `rssm.recurrent_mlp_linear` + `rssm.recurrent_mlp_norm`
+ `silu` + `rssm.gru_cell` + `rssm._transition` directly — NOT a dedicated `RSSM.imagination(...)`
method as in sheeprl `agent.py:L482-L498`. The unrolled equivalence is correct: sheeprl's
`imagination()` is just the `_transition`-only path (no representation_model call), which is
what dreamer-srl reconstructs inline. Match at the math level.

**Caveat**: a future RSSM refactor could centralize this into a `rssm.imagination()` method
matching sheeprl's API and reduce duplication. Out of scope for this audit; flagged as a
**hygiene NIT** in the cross-reference table below.

### 6.3 Stop-gradient placement

Sheeprl's `imagined_trajectories.detach()` at `dreamer_v3.py:L273` and `imagined_latent_state.detach()`
at L219/L240 — both block gradient from the rollout chain into the actor's loss-time graph.
Dreamer-srl's analog is `jax.lax.stop_gradient(imagined_latents)` at `train.py:L838`. Because
`world_model.imagine` is itself called OUTSIDE the gradient context (not inside
`actor_loss_fn`), the gradient never enters the rollout in the first place — the
`stop_gradient` at L838 is **defensive but not strictly required** for the imagine() output
specifically. **Match in effect.**

### 6.4 Trainability of composite — PASS

Verified live via `nnx.split(world_model)` that the composite's State pytree exposes top-level
keys `{continue_model, decoder, encoder, reward_model, rssm}` — every sub-module's params
are reachable by `nnx.value_and_grad(wm_loss_fn)(world_model)` at `train.py:L763-L765`. The
zero-init assertion for `reward_model.output_linear.kernel` and `critic.output_linear.kernel`
at `build_agent` L2106-L2115 is enforced.

---

## 7. `build_agent` factory — PASS

Construction order at `build_agent` L2009-L2102:

1. `MLPEncoder` (L2010-L2015)
2. `RSSM` (L2018-L2030)
3. `MLPDecoder` (L2033-L2039)
4. `FullMLPHead` reward_model with `zero_init_output=True` (L2045-L2052)
5. `FullMLPHead` continue_model with `zero_init_output=False` (L2055-L2062)
6. `WorldModel` composite (L2065-L2071)
7. `Actor` (L2074-L2081)
8. `FullMLPHead` critic with `zero_init_output=True` (L2085-L2092)
9. `FullMLPHead` target_critic with `zero_init_output=True` (L2095-L2102)

**Differences from sheeprl `build_agent` (`agent.py:L935-L1236`)** — all are functionally
equivalent for the single-device JAX port:

- Sheeprl wraps each module in `_FabricModule` for distributed-training abstraction;
  dreamer-srl does not — single-device, no DDP. **OK** (cross-substrate boundary).
- Sheeprl uses `copy.deepcopy(critic.module)` at L1217 for `target_critic`; dreamer-srl
  constructs a fresh `FullMLPHead` at L2095-L2102 with the same arch and zero-init. **Note**:
  this means at agent-construction time, `target_critic` is identical to `critic` (both have
  zero kernel, zero bias). Sheeprl's deepcopy makes them identical at init too. The first
  Polyak update at the driver-loop level uses `tau=1.0` (per sheeprl `dreamer_v3.py:L678`:
  `tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau`) — copying the
  critic into the target_critic. Both target_critic init flows converge to the same value at
  the first polyak. **Match.**
- Sheeprl ties weights between agent and player via `p.data = agent_p.data` at L1230-L1233;
  dreamer-srl's `Player` class (`dreamer_srl_main.py:L61-L152`) instead holds direct module
  references (`self.world_model = world_model; self.actor = actor`) — **no copy, no tie**, so
  by reference identity the player and agent share params. **Match in effect** (the tie is
  vacuous when there's only one copy).
- Zero-init assertions at `build_agent` L2108-L2115 — present and active.

The two `target_critic`-related deviations are documented in
[`SHEEPRL_REFERENCE_AUDIT.md`](../../docs/develop/active/dreamer_srl_v2/SHEEPRL_REFERENCE_AUDIT.md) §6 and §8 as expected substrate boundaries; the
Polyak-update is a v2-CP7 (driver) responsibility.

### 7.1 Optimizer wiring — out of audit scope

`build_agent` does NOT construct optimizers — those are wired in
`dreamer_srl_main.py` (`L207-L225`). The audit scope is wrapper-class correctness; optimizer
correctness (epsilon placement, gradient-clipping, learning rates) is v2-CP6 / v2-CP7 territory.

Confirmed via grep that **three separate optimizers** are used: `actor_opt`, `critic_opt`,
`wm_opt` — each with their own Adam state. Sheeprl matches (`dreamer_v3.py:L272, L313, L191`).

---

## 8. Findings table

| Severity | Module | File:line | Issue | Suggested fix |
|---|---|---|---|---|
| BLOCKER | Actor / actor-loss flow | [`agent.py:L1484-L1542`](../../src/algorithms/dreamer_srl/agent.py) + [`train.py:L836-L867`](../../src/algorithms/dreamer_srl/train.py) | REINFORCE's `log_prob(a)` is evaluated at a **freshly-sampled action** inside `actor_loss_fn`, not at the **rollout action** that produced the lambda-return / advantage. The rollout actions in `imag_outputs["imagined_actions"]` are computed but discarded. Sheeprl's pattern `p_NEW.log_prob(action_OLD.detach())` is broken; dreamer-srl computes `p_NEW.log_prob(action_NEW.detach())`. Per-sample gradient is systematically pushed toward counterfactual actions weighted by factual advantages — primary H1 root-cause candidate. | Plumb `imag_outputs["imagined_actions"]` through `train.py` into `actor_loss_fn`. Modify `Actor.__call__` to optionally accept a `target_action` arg; when provided, skip Gumbel sample and compute `log_prob` from `sg(target_action) · log_softmax(logits)`. Or compute `log_softmax(logits)` outside the actor module and apply the log-prob formula directly in `actor_loss_fn`. |
| CONCERN | Actor / actor-loss flow | [`train.py:L837`](../../src/algorithms/dreamer_srl/train.py) | `actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)` — constant PRNG seed for the actor's straight-through Gumbel sampling during the loss-time re-run. Even after the BLOCKER fix above, this is a PRNG-discipline regression. Loss-time actions are deterministic across gradient steps, which diverges from sheeprl's per-call global-RNG draw. | Thread the train-step key through into `actor_loss_fn`. Split per H+1 actor calls using a new sub-key, mirroring the `k_imag` pattern at L785. |
| NIT | Actor | [`agent.py:L1539-L1540`](../../src/algorithms/dreamer_srl/agent.py) | `entropy = -sum(probs * log(probs + 1e-8))` uses an explicit `+1e-8` floor; sheeprl `Categorical.entropy()` uses `-sum(probs * log_softmax(logits))` (numerically stable, no eps). Algebraically equivalent at unimix=0.01, D≥5 (min prob = 0.002, well above 1e-8). | Replace with `entropy = -jnp.sum(probs * jax.nn.log_softmax(logits, axis=-1), axis=-1)` for sheeprl-bitwise-identical form. Negligible numerical drift either way. |
| NIT | ContinueHead | [`agent.py:L1299-L1377`](../../src/algorithms/dreamer_srl/agent.py) | Class is defined but never instantiated by `build_agent` — replaced by `FullMLPHead(out_dim=1, zero_init_output=False)` at L2055-L2062. Dead code. | Delete `ContinueHead` class or document it as deprecated. No behavioural impact. |
| NIT | WorldModel.imagine | [`agent.py:L1766-L1789`](../../src/algorithms/dreamer_srl/agent.py) | Inlines the RSSM imagination step (`recurrent_mlp_linear → recurrent_mlp_norm → SiLU → gru_cell → _transition`) rather than calling a centralised `rssm.imagination(...)` method matching sheeprl `agent.py:L482-L498`. Math is correct; just duplicates RSSM internals. | Add `RSSM.imagination(prior, recurrent_state, action, key)` method and call it from `imagine()`. Pure refactor; no behavioural change. |

---

## 9. Conventions audit checklist

This is the project's JAX-conventions checklist from [`CLAUDE.md`](../../CLAUDE.md). Note that
several rows are non-applicable to dreamer-srl, which intentionally lives outside the gridworld
environment-pytree conventions — dreamer-srl uses raw JAX arrays, not Flax `@struct.dataclass`
EnvState pytrees.

- **Pytree immutability** — N/A. Dreamer-srl wrappers are `nnx.Module` instances (not Flax
  struct dataclasses). Module state is updated via `nnx.Optimizer.update(module, grads)` which
  is the NNX-correct mutation pattern.
- **JIT recompilation triggers** — PASS. Static fields (`mlp_layers`, `dense_units`, etc.) are
  Python ints baked into module construction; dynamic state (logits, hidden states) flows as
  JAX arrays. No traced→static leakage detected.
- **vmap conventions** — PASS. `jax.vmap(self.encoder)`, `jax.vmap(self.decoder)`, etc., at
  `WorldModel.observe` / `Player.get_actions` correctly batch over the leading axis. No
  EnvParams batching (not applicable — dreamer-srl modules are not env-state pytrees).
- **PRNG key threading** — **FAIL** at one site: `train.py:L837` uses `PRNGKey(0)` constant.
  All other PRNG sites (`world_model.imagine` L1754/L1767; `rssm.dynamic` L1093/L1102;
  `Player.get_actions` L129/L144) correctly split-and-advance. Flagged as MED.
- **Sensor / observation breakdown sync** — N/A (dreamer-srl is not gridworld env code).
- **Configuration protocol** — PASS at the wrapper-module level. `build_agent` extracts all
  config values via `cfg['algo']['world_model'][...]` etc. with direct dict access — missing
  key raises `KeyError`. The `get_mandatory` wrapper is enforced at the `dreamer_srl_main.py`
  driver layer (L207-L225); the wrappers themselves correctly fail-loud on missing keys.

---

## 10. Conclusion

**Wrappers verdict by module**:

- **Actor** — **FAIL** (1 BLOCKER, 1 CONCERN, 1 NIT) — H1 epicenter; REINFORCE log-prob is
  evaluated at the wrong action; constant `PRNGKey(0)`; entropy uses `log(p+1e-8)` instead of
  `log_softmax(logits)`.
- **Encoder (MLPEncoder)** — **PASS** — structural and init parity to sheeprl `agent.py:L100-L151`.
- **Decoder (MLPDecoder)** — **PASS** — structural and init parity to sheeprl `agent.py:L229-L278`.
- **ContinueHead** — **PASS** (with NIT: dead-code class; the actually-used continue model is
  `FullMLPHead(out_dim=1, zero_init_output=False)`).
- **FullMLPHead** — **PASS** — correctly implements zero-init / Hafner-init dual branches for
  reward / continue / critic / target_critic; matches sheeprl's two-phase `build_agent` init.
- **WorldModel composite** — **PASS** — gradient flow through `observe()` is correct; `imagine()`
  is called outside the gradient context so the actor doesn't get spurious gradient via the
  rollout chain. NIT: `imagine()` inlines RSSM internals.
- **`build_agent`** — **PASS** — construction order, zero-init enforcement assertions, separate
  target_critic instance, and module trainability are all correct.

**The single BLOCKER finding (1.2.3) is the most likely H1 root cause among the wrapper-class
surface**. The fix is mechanical: thread `imag_outputs["imagined_actions"]` through
`actor_loss_fn` and evaluate `log_prob` at the rollout action, not at a freshly-sampled action.
The CONCERN finding (1.2.4 — constant `PRNGKey(0)`) is masked by the BLOCKER but would still
need to be patched.

This audit does not modify code. Findings are handed off to the developer agent. The
fix should be paired with the v2-CP3 grad-parity methodology (when authored by
`professor-rl-bayesian-dl` under v2-CP2) to verify per-parameter gradient bit-identity against
sheeprl after the patch.

Reviewed by: code-reviewer

---

## Verification Report

> **Verified by**: `code-reviewer` (this memo). Read-only line-by-line audit of dreamer-srl
> wrapper classes against vendored sheeprl `33b6366`. No source-tree, `vendor/sheeprl/`, or
> config modifications.
> **Date**: 2026-05-14.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| [`docs/reviews/dreamer_srl_v2_cp8_wrappers_review.md`](dreamer_srl_v2_cp8_wrappers_review.md) | New review memo (this file) | ☑ authored | One BLOCKER finding, one CONCERN, three NITs. |

**Conclusion**: v2-CP8 wrapper-module re-audit complete. **FAIL** — the Actor's REINFORCE
log-prob evaluation at a freshly-sampled action (rather than the rollout action) is the
H1-epicenter candidate. Handoff: the `developer` agent should patch the BLOCKER + CONCERN
findings; the v2-CP3 grad-parity methodology (pending from `professor-rl-bayesian-dl`) should
then verify per-parameter gradient bit-identity against sheeprl on the patched code.
