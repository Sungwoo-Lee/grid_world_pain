---
title: "v2-CP6 — `one_train_step` orchestrator re-audit (integration-level)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
phase: 2
reviewer: code-reviewer
sheeprl_commit: 33b6366
dreamer_srl_commit: 197e212
---

# v2-CP6 — `one_train_step` orchestrator re-audit

## Verdict (plain-language entry point)

**What this doc is.** A line-by-line correctness re-audit of dreamer-srl's `make_train_step` / `one_train_step`
factory in [`src/algorithms/dreamer_srl/train.py:L613-L913`](../../src/algorithms/dreamer_srl/train.py)
— the **integration site** that wires together the world-model loss, imagined rollout, λ-return, actor objective,
critic two-term loss, and three separate optimizer steps into a single JIT'd training step. The reference is
sheeprl's `train()` function at `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L48-L358`.

**Why it exists.** The v2 audit chain (CP3 actor, CP4 critic, CP5 imagined-returns, CP7 driver, CP8 wrappers)
has already surfaced **5 P-blockers** that cleanly explain the v1 parity failure (random-policy floor at ~104
survival steps versus ~500 sheeprl baseline). v2-CP6 is the **last orthogonal check** before v2-CP9 lands the
fix bundle: it asks whether the orchestrator introduces **additional** integration-level issues that the
per-function audits could not see (sub-loss ordering, optimizer-state isolation, stop-gradient boundary
placement at the seams between modules, PRNG threading across the full step).

**Headline.** ❌ **FAIL** at the integration level — but **NO new P-class blockers beyond the known five**.
The five already-confirmed blockers (CP7-P1 reset-data write, CP7-P2 terminated/truncated conflation, CP7-P3
D-014 burst, CP8-P1/CP3-A1 resampled action in actor loss, CP3-A2 constant `PRNGKey(0)`, CP5-P1/CP3-A3 stale
`target_critic` for `predicted_values`) **all manifest at orchestrator lines 798 / 837 / 845** and are therefore
visible to a CP6 audit as well. Beyond these, the orchestrator's **sub-loss ordering**, **3-optimizer-state
isolation**, **stop-gradient boundary discipline at the WM↔rollout↔actor↔critic seams**, and **polyak
ordering** are all **structurally faithful to sheeprl**. The only NEW finding at this audit is one 🟡 **concern**
about WM-imagine using post-step WM parameters (which matches sheeprl, so likely fine — flagged for explicit
gradient-parity verification at v2-CP6's grad-test #3).

**Verdict trio.** Imagined-actions-threading: **❌ broken (CP8-P1/CP3-A1 — confirmed: `imag_outputs["imagined_actions"]`
is collected at `agent.py:L1799` and dropped at orchestrator `train.py:L788`; `actor_loss_fn` resamples)**.
Optimizer-state isolation: **✅ clean (three separate `nnx.Optimizer` instances passed in by the driver;
each `update()` consumes only its own gradient pytree; no shared moments)**. Stop-gradient seams: **✅ all six
canonical sites present (`sg(posteriors)`, `sg(recurrent_states)`, `sg(imagined_latents)` for actor input,
`sg(imagined_latents[:-1])` for critic input, `sg(lambda_values)` and `sg(predicted_values)` inside actor loss,
`sg(moments_offset)` / `sg(moments_invscale)`)**.

---

## 1. Scope and methodology

This audit reads dreamer-srl's `make_train_step` / `one_train_step`
([`train.py:L613-L913`](../../src/algorithms/dreamer_srl/train.py)) against sheeprl's `train()` function
([`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L48-L358`](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py)),
walking the 13 sub-steps in order. For each seam (WM↔rollout, rollout↔actor, rollout↔critic, actor↔critic,
moments↔actor, target↔critic), the audit verifies:

1. **Inputs / outputs match the sheeprl analog** (shape, semantic role, sg discipline).
2. **No cross-pollution of gradients** between the three separate optimizers.
3. **Stop-gradient boundary placement** matches sheeprl `.detach()` sites.
4. **PRNG threading** is consistent with sheeprl's implicit-global-RNG semantics (or flagged as a known
   cross-substrate boundary, D-009 class).
5. **Sub-loss ordering** matches sheeprl's WM → rollout → λ-return → actor → critic sequence.

Already-confirmed P-blockers from prior CPs are **noted as known** and cross-flagged at the integration site
where they materialise — they are not re-derived. The audit's primary deliverable is whether v2-CP6 surfaces
ANY new integration-class issue beyond the known 5.

---

## 2. Sub-loss ordering audit

Sheeprl's canonical ordering in `train()`:

| # | Sheeprl block | Lines | dreamer-srl analog | Lines |
|---|---|---|---|---|
| 1 | Force `is_first[0,:]=1` | L100 | §S1 force-set | [L686-L687](../../src/algorithms/dreamer_srl/train.py) |
| 2 | `batch_actions = action_shift(...)` | L104 | §S2 `action_shift` | [L693](../../src/algorithms/dreamer_srl/train.py) |
| 3 | WM forward (encoder + RSSM dynamic loop) | L113-L146 | `wm.observe(...)` | [L702](../../src/algorithms/dreamer_srl/train.py) |
| 4 | WM heads (decoder, reward, continue) | L148-L168 | Inside `wm.observe` | [L702](../../src/algorithms/dreamer_srl/train.py) |
| 5 | WM loss (reconstruction + KL) | L174-L190 | `wm_loss_fn` | [L700-L761](../../src/algorithms/dreamer_srl/train.py) |
| 6 | WM backward + optimizer step | L191-L200 | `nnx.value_and_grad` + `wm_opt.update` | [L763-L767](../../src/algorithms/dreamer_srl/train.py) |
| 7 | Rollout init from posterior | L203-L220 | Posterior reshape + `wm.imagine` | [L773-L786](../../src/algorithms/dreamer_srl/train.py) |
| 8 | Horizon-H imagined steps | L235-L241 | Inside `wm.imagine` | [`agent.py:L1766-L1796`](../../src/algorithms/dreamer_srl/agent.py) |
| 9 | Predict r/v/c on imagined trajectories | L244-L246 | vmap of `reward_model`, `target_critic`(!), `continue_model` | [L793-L806](../../src/algorithms/dreamer_srl/train.py) |
| 10 | §S5 true-continue splice | L247-L248 | Inside `compute_imagined_returns` | [L811-L818](../../src/algorithms/dreamer_srl/train.py) |
| 11 | λ-return computation | L251-L256 | Inside `compute_imagined_returns` | [L811-L818](../../src/algorithms/dreamer_srl/train.py) |
| 12 | Discount cumprod | L259-L260 | Inside `compute_imagined_returns` | [L811-L818](../../src/algorithms/dreamer_srl/train.py) |
| 13 | Moments update | (inside `moments(...)` at L276) | `moments_update` BEFORE actor loss | [L823-L830](../../src/algorithms/dreamer_srl/train.py) |
| 14 | Actor loss (REINFORCE + entropy + advantage normalization) | L272-L297 | `actor_loss_fn` | [L840-L864](../../src/algorithms/dreamer_srl/train.py) |
| 15 | Actor backward + optimizer step | L298-L304 | `nnx.value_and_grad` + `actor_opt.update` | [L866-L867](../../src/algorithms/dreamer_srl/train.py) |
| 16 | Critic loss (two-term: λ-return + target) | L307-L316 | `critic_loss_fn` | [L880-L892](../../src/algorithms/dreamer_srl/train.py) |
| 17 | Critic backward + optimizer step | L318-L327 | `nnx.value_and_grad` + `critic_opt.update` | [L894-L895](../../src/algorithms/dreamer_srl/train.py) |
| 18 | Polyak update | (fires BEFORE `train()`, sheeprl L678-L680) | (fires BEFORE `one_train_step`, [`dreamer_srl_main.py:L516-L524`](../../src/algorithms/dreamer_srl/dreamer_srl_main.py)) | driver |

**Ordering verdict: ✅ faithful.** Every sub-step is in the same relative position. No reorderings, no
missing steps, no extras.

**Subtlety (worth noting, not a bug):** sheeprl computes `predicted_values = ...(critic(imagined_trajectories), dims=1).mean`
at L244 using the **live critic**, BEFORE any actor or critic update happens in this train() call. The dreamer-srl
analog at [`train.py:L798`](../../src/algorithms/dreamer_srl/train.py) instead calls `target_critic` — this is
the already-known **CP5-P1 / CP3-A3 blocker**, materialising at the orchestrator. It is documented; not a new
finding.

---

## 3. The `imagined_actions` thread — the single most important integration question

### 3.1 Where the actions are collected

`WorldModel.imagine()` at [`agent.py:L1708-L1808`](../../src/algorithms/dreamer_srl/agent.py) collects per-step
actor outputs in three lists:

- `imagined_actions` ([L1744](../../src/algorithms/dreamer_srl/agent.py)) — populated at [L1756](../../src/algorithms/dreamer_srl/agent.py)
  (step 0) and [L1794](../../src/algorithms/dreamer_srl/agent.py) (step i ≥ 1) by calling `actor(...)` and taking
  the first return value.
- `imagined_log_probs` ([L1745, L1757, L1795](../../src/algorithms/dreamer_srl/agent.py)).
- `imagined_entropies` ([L1746, L1758, L1796](../../src/algorithms/dreamer_srl/agent.py)).

All three are stacked into arrays and returned in the dict at [L1803-L1807](../../src/algorithms/dreamer_srl/agent.py):

```python
return {
    "imagined_latents":   imagined_latents_arr,
    "imagined_actions":   imagined_actions_arr,    # shape [H+1, BT, action_dim]
    "imagined_log_probs": imagined_log_probs_arr,  # shape [H+1, BT, 1]
    "imagined_entropies": imagined_entropies_arr,  # shape [H+1, BT]
}
```

So the rollout DOES collect the imagined actions.

### 3.2 Where the orchestrator drops them

The orchestrator receives the dict at [`train.py:L786`](../../src/algorithms/dreamer_srl/train.py):

```python
imag_outputs = world_model.imagine(init_latent, actor, horizon, k_imag)
imagined_latents = imag_outputs["imagined_latents"]    # L788 — ONLY this key is consumed
```

The three other keys (`imagined_actions`, `imagined_log_probs`, `imagined_entropies`) are **never read again**.
A grep confirms zero references to `imag_outputs["imagined_actions"]` or `imag_outputs["imagined_log_probs"]`
or `imag_outputs["imagined_entropies"]` anywhere downstream in `train.py`.

### 3.3 What `actor_loss_fn` does instead

Inside `actor_loss_fn` at [`train.py:L840-L864`](../../src/algorithms/dreamer_srl/train.py):

```python
actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)   # L837 — CP3-A2 (constant seed)
sg_latents = jax.lax.stop_gradient(imagined_latents)              # L838

def actor_loss_fn(actor_module):
    all_log_probs = []
    all_entropies = []
    for h in range(H_plus_1):
        _, lp_h, ent_h = actor_module(sg_latents[h], actor_keys[h])   # L845 — RESAMPLES
        all_log_probs.append(lp_h)
        all_entropies.append(ent_h)
    ...
```

The actor is re-invoked on the same `sg_latents` it produced during rollout, but with **fresh PRNG keys**.
Inside `Actor.__call__` ([`agent.py:L1484-L1542`](../../src/algorithms/dreamer_srl/agent.py)), the keys drive a
Gumbel-softmax straight-through sample at [L1520-L1526](../../src/algorithms/dreamer_srl/agent.py):

```python
gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
perturbed = logits + gumbel_noise
hard_indices = jnp.argmax(perturbed, axis=-1)
hard = jax.nn.one_hot(hard_indices, self.action_dim)
soft = jax.nn.softmax(logits, axis=-1)
actions = hard - jax.lax.stop_gradient(soft) + soft
```

— a **fresh, decorrelated** one-hot sample. Then `log_probs = jnp.sum(sg_actions * log_softmax_logits, ...)`
at [L1536](../../src/algorithms/dreamer_srl/agent.py) is computed against this NEW sample. **The advantage,
however, was computed from the OLD action** (which determined the rollout trajectory → predicted_rewards →
λ-return → advantage). The product `log_prob(a_new) × sg(advantage(a_old))` is NOT a policy gradient. It is
the per-sample form of `E_{a~π}[log_prob(a)] × sg(advantage(rollout))` — pushing the policy toward whatever
action it happens to resample, weighted by the rollout-action's advantage. This is precisely the
**CP8-P1 / CP3-A1 blocker**.

### 3.4 The correct sheeprl pattern

Sheeprl's `train()` at L262-L297 does:

```python
policies = actor(imagined_trajectories.detach())[1]   # L273 — gets the DISTRIBUTION, not a fresh sample
...
objective = (
    torch.stack(
        [
            p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]                            # L286
            for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
        ],
        ...
    ).sum(dim=-1)
    * advantage.detach()
)
```

The key observations:

1. `actor(imagined_trajectories.detach())[1]` returns the **policy distribution** at each rollout latent.
   No new sample is drawn — the distribution is constructed from the current actor's logits.
2. `p.log_prob(imgnd_act.detach())` evaluates the log-probability of the **rollout-time action** (the same
   action that was sampled when `imagined_trajectories` was being computed) under the current policy. The
   `.detach()` on `imgnd_act` matches the §S7 sg(action) discipline.
3. The advantage is the gradient-stopped `sg(advantage)` per L291.

This is the **canonical policy gradient form**: `∇_θ E_{a~π_θ}[log π_θ(a) · sg(A(a))]` evaluated at the
specific sample `a = imagined_action`, with sg around `A` (advantage), no sg around `log π_θ` w.r.t. its own
parameters, and no resampling.

### 3.5 Pseudocode for the correct fix (for v2-CP9 developer)

```python
# In one_train_step, after wm.imagine returns:
imagined_latents = imag_outputs["imagined_latents"]
imagined_actions = imag_outputs["imagined_actions"]   # NEW: thread this downstream

sg_latents = jax.lax.stop_gradient(imagined_latents)
sg_imagined_actions = jax.lax.stop_gradient(imagined_actions)

def actor_loss_fn(actor_module):
    all_log_probs = []
    all_entropies = []
    for h in range(H_plus_1):
        # Compute logits at sg_latents[h], evaluate log_prob at the FIXED rollout action.
        # Either (a) refactor Actor to expose a forward_logits(latent) -> logits API,
        # or (b) keep Actor.__call__ but pass an `action_override` kwarg that skips the
        # Gumbel sample and uses sg_imagined_actions[h] for log_prob.
        # The PRNG key becomes ENTIRELY UNUSED for the loss path (since no sample is drawn);
        # the only remaining reason to thread a key would be for entropy regularizer's
        # noise injection, if any — sheeprl has none, so the key is purely a no-op.
        logits_h = actor_module.forward_logits(sg_latents[h])
        log_softmax_h = jax.nn.log_softmax(logits_h, axis=-1)
        log_prob_h = jnp.sum(sg_imagined_actions[h] * log_softmax_h, axis=-1, keepdims=True)
        probs_h = jax.nn.softmax(logits_h, axis=-1)
        entropy_h = -jnp.sum(probs_h * jnp.log(probs_h + 1e-8), axis=-1)
        all_log_probs.append(log_prob_h)
        all_entropies.append(entropy_h)
    log_probs_arr = jnp.stack(all_log_probs, axis=0)
    entropies_arr = jnp.stack(all_entropies, axis=0)
    log_probs_sliced = log_probs_arr[:-1]
    entropy_for_obj = entropies_arr[..., None]
    return compute_actor_objective(
        log_probs=log_probs_sliced,
        lambda_values=jax.lax.stop_gradient(lambda_values),
        predicted_values=jax.lax.stop_gradient(predicted_values),
        moments_offset=jax.lax.stop_gradient(moments_offset),
        moments_invscale=jax.lax.stop_gradient(moments_invscale),
        entropy=entropy_for_obj,
        discount=discount,
        ent_coef=ent_coef,
    )[0]
```

This eliminates **both** CP8-P1 (resampling) and CP3-A2 (constant PRNG seed) in one fix, because no PRNG is
needed for the loss path at all.

**Imagined-actions-threading verdict: ❌ broken at orchestrator [`train.py:L788`](../../src/algorithms/dreamer_srl/train.py)
(drop) + [`train.py:L845`](../../src/algorithms/dreamer_srl/train.py) (resample).** This is the
already-known CP8-P1 / CP3-A1 blocker, confirmed at the orchestrator audit surface.

---

## 4. Optimizer-step ordering and state isolation

### 4.1 Three separate optimizer instances

[`dreamer_srl_main.py:L264-L270`](../../src/algorithms/dreamer_srl/dreamer_srl_main.py) constructs three
`nnx.Optimizer` instances at driver startup:

```python
wm_opt     = nnx.Optimizer(world_model, optax.adam(wm_lr,     eps=wm_eps),     wrt=nnx.Param)
actor_opt  = nnx.Optimizer(actor,       optax.adam(actor_lr,  eps=actor_eps),  wrt=nnx.Param)
critic_opt = nnx.Optimizer(critic,      optax.adam(critic_lr, eps=critic_eps), wrt=nnx.Param)
```

Each `nnx.Optimizer` carries its own `optax` Adam state (`m`, `v`, `t` moments) keyed to its own `wrt=nnx.Param`
selection on its own module. There is **no shared optimizer state**, and `nnx.Optimizer.update(module, grads)`
applies the optax transformation in-place on the module's `nnx.Param` leaves only.

### 4.2 Each optimizer consumes its own gradient pytree

Inside `one_train_step`:

| Optimizer | Gradient source | Module updated | Line |
|---|---|---|---|
| `wm_opt` | `nnx.value_and_grad(wm_loss_fn, has_aux=True)(world_model)` | `world_model` only | [L763-L767](../../src/algorithms/dreamer_srl/train.py) |
| `actor_opt` | `nnx.value_and_grad(actor_loss_fn)(actor)` | `actor` only | [L866-L867](../../src/algorithms/dreamer_srl/train.py) |
| `critic_opt` | `nnx.value_and_grad(critic_loss_fn)(critic)` | `critic` only | [L894-L895](../../src/algorithms/dreamer_srl/train.py) |

Each `value_and_grad` call passes a **single module** (not a tuple of modules), so the gradient pytree
produced has the SAME structure as that module's params and contains gradients **only for that module's
leaves**. There is no possibility of cross-pollination unless one of the loss functions takes gradient w.r.t.
multiple modules — and none of them do.

### 4.3 Cross-pollution sanity check at each `value_and_grad` boundary

- **`wm_loss_fn(wm)`** at [L700-L761](../../src/algorithms/dreamer_srl/train.py) takes only `wm` as differentiable
  input; uses `batch` (data, no grad), `is_first` (data), `shifted_actions` (data), `k_wm` (PRNG). No actor /
  critic / target_critic forward call in this closure → no gradient leak.
- **`actor_loss_fn(actor_module)`** at [L840-L864](../../src/algorithms/dreamer_srl/train.py) takes `actor_module`;
  uses `sg_latents` (already sg'd), `actor_keys` (constant PRNG, no grad), `lambda_values` / `predicted_values`
  / `moments_offset` / `moments_invscale` (all sg'd inside `compute_actor_objective`'s call site at L856-L859),
  `discount` (already sg'd by `compute_discount`'s `jax.lax.stop_gradient`), `ent_coef` (Python scalar). No
  critic / WM forward call → no leak. **NOTE:** the call to `actor_module(sg_latents[h], actor_keys[h])` at
  L845 internally constructs the Gumbel sample and computes log_probs / entropy — all sg'd or differentiable
  through `actor_module`'s params only. Gradient correctly flows ONLY into actor params.
- **`critic_loss_fn(critic_module)`** at [L880-L892](../../src/algorithms/dreamer_srl/train.py) takes `critic_module`;
  uses `sg_latents_h` (sg'd at L873), `lambda_values` (sg'd inside `compute_critic_loss`), `target_critic_values`
  (sg'd inside `compute_critic_loss`), `discount` (already sg'd). The `target_critic_logits` at L875-L878 is
  **outside the closure** — it's evaluated against the current `target_critic` module which is NOT being
  differentiated. The closure only differentiates `critic_module`. No leak.

### 4.4 Polyak ordering

Sheeprl: polyak at L678-L680 fires BEFORE `train(...)` call at L682, inside the per-gradient-step inner loop.
This means within a single train step, the target_critic used inside `train()` is the **freshly updated** one.

dreamer-srl: polyak at [`dreamer_srl_main.py:L516-L524`](../../src/algorithms/dreamer_srl/dreamer_srl_main.py)
fires BEFORE `train_step(...)` call at L537-L541, inside the per-gradient-step inner loop. **Match.**

The initial-step `tau=1.0` special case (which makes target = online byte-identical at step 0) is preserved
at both sites: sheeprl L678 (`tau = 1 if cumulative... == 0 else cfg.algo.critic.tau`) and dreamer-srl L517
(`tau = 1.0 if cumulative_grad_steps == 0 else critic_tau`). **Match.**

**Optimizer-state isolation verdict: ✅ clean.** Three independent optimizer states, no shared moments, each
optimizer's gradient pytree comes from a single-module `value_and_grad` call → no cross-pollution possible.
Polyak fires in the correct relative position.

---

## 5. Gradient stop-flow boundaries in the orchestrator

The orchestrator carries **eight** sg sites at module seams. Per CP1 §3-§5 the sheeprl reference equivalents
are `.detach()` calls. Per-site verdict:

| # | Site | dreamer-srl line | Sheeprl analog | Verdict |
|---|---|---|---|---|
| 1 | `posteriors` for rollout init | [L773](../../src/algorithms/dreamer_srl/train.py) `jax.lax.stop_gradient(wm_outputs["posteriors"])` | L203 `posteriors.detach()` | ✅ |
| 2 | `recurrent_states` for rollout init | [L774](../../src/algorithms/dreamer_srl/train.py) `jax.lax.stop_gradient(wm_outputs["recurrent_states"])` | L204 `recurrent_states.detach()` | ✅ |
| 3 | `imagined_latents` for actor loss | [L838](../../src/algorithms/dreamer_srl/train.py) `jax.lax.stop_gradient(imagined_latents)` | L273 `imagined_trajectories.detach()` (passed to `actor`) | ✅ |
| 4 | `lambda_values` (input to actor loss) | [L856](../../src/algorithms/dreamer_srl/train.py) `jax.lax.stop_gradient(lambda_values)` | (implicit — `lambda_values` is computed under `torch.no_grad` for the discount, but the actor's normalization reads it un-detached at L277-L278; the critical detach is `advantage.detach()` at L291) | ✅ (defensive; not strictly required for the actor path since the rollout was already detached at L203/L204, but harmless and explicit) |
| 5 | `predicted_values` (input to actor loss) | [L857](../../src/algorithms/dreamer_srl/train.py) `jax.lax.stop_gradient(predicted_values)` | (implicit — sheeprl uses `predicted_values[:-1]` un-detached at L275, but the chain back to actor is blocked by L203/L204 detaches) | ✅ (defensive; harmless) |
| 6 | `moments_offset`, `moments_invscale` (input to actor loss) | [L858-L859](../../src/algorithms/dreamer_srl/train.py) `jax.lax.stop_gradient(...)` | `utils.py:L63` `self.low.detach(), invscale.detach()` | ✅ |
| 7 | `imagined_latents[:-1]` for critic loss | [L873](../../src/algorithms/dreamer_srl/train.py) `jax.lax.stop_gradient(imagined_latents[:-1])` | L307/L309 `imagined_trajectories.detach()[:-1]` | ✅ |
| 8 | Internal sg in `compute_actor_objective` (sg(advantage), sg(action) via Actor) | [`train.py:L591`](../../src/algorithms/dreamer_srl/train.py) + [`agent.py:L1535`](../../src/algorithms/dreamer_srl/agent.py) | L291 `advantage.detach()`, L286 `imgnd_act.detach()` | ✅ for sg(advantage); ⚠ for sg(action) — the action being sg'd is the RESAMPLED one (CP8-P1), not the rollout one |
| 9 | Internal sg in `compute_critic_loss` (sg(lambda_values), sg(target_critic_values)) | inside `compute_critic_loss` | L314 `lambda_values.detach()`, L315 `predicted_target_values.detach()` | ✅ (per CP4 review) |
| 10 | Discount (already sg'd by `compute_discount`) | [`train.py:L217-L219`](../../src/algorithms/dreamer_srl/train.py) | L259-L260 `with torch.no_grad():` | ✅ |

**Stop-gradient seams verdict: ✅ structurally faithful.** Every sheeprl `.detach()` site has a JAX equivalent
in the orchestrator. The site-#4 and site-#5 explicit `sg` calls are belt-and-braces (sheeprl relies on the
rollout init's `.detach()` to block the gradient chain; dreamer-srl explicitly re-applies it at the loss
boundary). Belt-and-braces is OK — gradients are already zero through these paths in both implementations.

**One subtle observation worth a 🟡 note:** site-#8 sg(action) is applied inside `Actor.__call__` at
[`agent.py:L1535`](../../src/algorithms/dreamer_srl/agent.py) to the RESAMPLED action. After the CP8-P1 fix,
sg(action) must instead be applied to `imagined_actions[h]` (which is what the fix at §3.5 does). The
sg-discipline placement is correct in principle; the action being sg'd is wrong (different P-blocker).

---

## 6. Loss aggregation — separate or merged?

Three SEPARATE `nnx.value_and_grad` calls, three SEPARATE `nnx.Optimizer.update` calls. **NOT merged.** This
matches sheeprl, which has three separate `world_optimizer.zero_grad / fabric.backward(rec_loss) /
world_optimizer.step` blocks (L175-L200), `actor_optimizer.zero_grad / fabric.backward(policy_loss) /
actor_optimizer.step` (L272-L304), `critic_optimizer.zero_grad / fabric.backward(value_loss) /
critic_optimizer.step` (L313-L327). **Match.**

A merged-loss form (e.g. `total_loss = wm_loss + policy_loss + value_loss` then `total_loss.backward()`) would
be a structural deviation — but the dreamer-srl orchestrator does NOT do this.

---

## 7. PRNG threading

The orchestrator splits the main key into sub-keys at three points:

1. **`k_wm`** at [`train.py:L698`](../../src/algorithms/dreamer_srl/train.py) — fed to `wm.observe(...)` at
   L702. Internally drives the RSSM's `_representation` / `_transition` sampling. **✅ correctly threaded.**
2. **`k_imag`** at [`train.py:L785`](../../src/algorithms/dreamer_srl/train.py) — fed to `world_model.imagine(...)`
   at L786. Internally split per-horizon-step inside `imagine` at [`agent.py:L1754, L1767`](../../src/algorithms/dreamer_srl/agent.py)
   for both the RSSM transition and the actor's Gumbel sample. **✅ correctly threaded.**
3. **`actor_keys` at [`train.py:L837`](../../src/algorithms/dreamer_srl/train.py) — `jax.random.split(jax.random.PRNGKey(0), H_plus_1)`.**
   **❌ BUG (CP3-A2 / known blocker).** Uses the constant seed `PRNGKey(0)` instead of splitting from the
   stepwise `key`. Under JIT, this deterministic constant means EVERY train step uses the same H+1 actor keys.
   Combined with CP8-P1's resampling, every train step resamples actor actions DETERMINISTICALLY identically.
   This is the **already-known CP3-A2 blocker, materialising at the orchestrator surface**. NOT a new finding.

After the §3.5 fix (no resampling at loss-time), `actor_keys` becomes irrelevant — the loss path no longer
needs a PRNG key. The fix removes both CP3-A2 and CP8-P1 in one move.

**PRNG threading verdict: ✅ for sites 1-2; ❌ at site 3 (known CP3-A2 blocker).**

---

## 8. Full-step gradient parity manifest (v2-CP6 grad-tests for v2-CP9 developer)

Per [`GRAD_PARITY_METHODOLOGY.md` §5.4](../develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md), v2-CP6's
5 grad-test sites are:

### 8.1 Test #1 — `∂total_loss / ∂(actor params)` after full train-step composition

```python
def test_full_step_actor_grad_matches_sheeprl():
    """v2-CP6 grad test #1 — full-step actor grad parity.

    Fixture seed: 0xD42AF for params, 0xD42B0 for inputs.
    Threshold: 2e-3 (per §5.4 row 1).
    Captures: spurious sg leak between modules, cross-module gradient interference.
    """
    seed_params = 0xD42AF
    seed_inputs = 0xD42B0

    # Build dreamer-srl modules at seed_params.
    jax_wm, jax_actor, jax_critic, jax_target = build_dreamer_srl_modules(seed_params)
    # Build sheeprl analogs at the same seed (per CP1 cross-substrate parity protocol).
    pt_wm, pt_actor, pt_critic, pt_target = build_sheeprl_modules(seed_params)

    # Build batch fixture at seed_inputs (obs, actions, rewards, terminated, is_first).
    batch_jax, batch_pt = build_fixture_batch(seed_inputs, T=16, B=4)

    # JAX side: take grad of TOTAL train-step loss w.r.t. actor params, AFTER fix.
    def jax_full_step_actor_loss(actor_params):
        # Wire actor_params into actor, run one_train_step's actor-loss path,
        # return the policy_loss scalar.
        ...
    jax_grad = jax.grad(jax_full_step_actor_loss)(nnx.state(jax_actor, nnx.Param))

    # Torch side: run sheeprl's train() through L297 (policy_loss); take autograd.grad.
    pt_grad = torch.autograd.grad(pt_policy_loss, pt_actor.parameters())

    max_abs_diff = max_abs(flatten(jax_grad), flatten(pt_grad))
    assert max_abs_diff < 2e-3, f"Full-step actor grad diff {max_abs_diff:.4e} > 2e-3"
```

### 8.2 Test #2 — `∂total_loss / ∂(critic params)`

Same skeleton, target = critic params, threshold = 2e-3, fixture `grad_cp6_full_critic.npz`. Captures:
sg leak from rollout into critic; missed `imagined_trajectories.detach()`.

### 8.3 Test #3 — `∂total_loss / ∂(world-model params)`

Same skeleton, target = WM params, threshold = **3e-3** (deeper stack — D-008 × 1.5 per §5.4). Captures: sg
leak from actor / critic into WM. **This is the test that catches the 🟡 concern about WM-imagine using
post-step WM params** (§9.1 below) — if the post-step WM use creates accumulated grad-stream divergence, the
threshold may need a one-time relaxation. Surface in the diff JSON's leaf breakdown.

### 8.4 Test #4 — Optimizer-step parity #1 (actor)

```python
def test_optax_adam_step_actor_matches_torch_adam_step():
    """v2-CP6 grad test #4 — optimizer-step parity (actor).
    
    Threshold: 5e-5 (per §5.4 row 4).
    Captures: eps placement, bias-correction step order.
    """
    seed = 0xD42AF
    actor_params_jax, actor_params_pt = build_aligned_params(seed)
    fake_grad = build_fixture_grad(0xD42B1)  # SAME grad fed to both sides

    # JAX side: optax.adam.update + apply_updates.
    opt_state = optax.adam(lr=8e-5, eps=1e-5).init(actor_params_jax)
    updates, new_opt_state = optax.adam(lr=8e-5, eps=1e-5).update(fake_grad, opt_state)
    new_params_jax = optax.apply_updates(actor_params_jax, updates)

    # Torch side: torch.optim.Adam.step on SAME grad.
    pt_optimizer = torch.optim.Adam(actor_params_pt, lr=8e-5, eps=1e-5)
    # ... inject fake_grad into .grad, then .step()

    max_abs_diff = max_abs(new_params_jax, new_params_pt)
    assert max_abs_diff < 5e-5
```

### 8.5 Test #5 — Optimizer-step parity #2 (critic)

Same as #4 with critic params and lr / eps. Threshold 5e-5.

### 8.6 Test seeds

Per §5.4: `0xD3EAF + 0x100 * 6 = 0xD42AF` for params, `+1` for inputs. Same convention as v2-CP3/CP4/CP5.

### 8.7 Test file

`tests/algorithms/dreamer_srl/test_train_grad.py` (shared with v2-CP5 per §6.1).

---

## 9. Findings table

| Severity | Site | Issue | Cross-flagged to |
|---|---|---|---|
| 🔴 **blocker (known)** | [`train.py:L788`](../../src/algorithms/dreamer_srl/train.py) + [`train.py:L845`](../../src/algorithms/dreamer_srl/train.py) | `imag_outputs["imagined_actions"]` collected at `agent.py:L1799` but **dropped** by orchestrator; `actor_loss_fn` **resamples** action via Gumbel-softmax at every train step. REINFORCE estimator broken. **Independently sufficient to drive policy to random floor.** | CP8-P1 / CP3-A1 |
| 🔴 **blocker (known)** | [`train.py:L837`](../../src/algorithms/dreamer_srl/train.py) | `actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)` uses CONSTANT seed under JIT → every train step uses identical actor PRNG keys, compounding the resample bug. After the CP8-P1 fix this line is removed entirely. | CP3-A2 |
| 🔴 **blocker (known)** | [`train.py:L798`](../../src/algorithms/dreamer_srl/train.py) | `predicted_values` for the λ-return bootstrap is computed from `target_critic` (slow Polyak EMA), but sheeprl uses the LIVE critic at L244. The bootstrap and the actor's advantage signal both lag the live critic by ~τ⁻¹≈50 grad steps. | CP5-P1 / CP3-A3 |
| 🟢 nit | [`train.py:L788-L807`](../../src/algorithms/dreamer_srl/train.py) | `imag_outputs["imagined_log_probs"]` and `imag_outputs["imagined_entropies"]` are also collected and dropped. After the CP8-P1 fix, `imag_outputs["imagined_log_probs"]` could be re-used directly instead of re-computing log_probs in `actor_loss_fn` — but the re-computation has the legitimate purpose of routing gradient through the CURRENT actor's logits (not the rollout-time actor's). So the rollout's log_probs / entropies remain genuinely unused; they could be deleted from `wm.imagine`'s return dict to make the code cleaner. Not a correctness issue. | (CP6 standalone) |
| 🟡 **concern (NEW — but expected behavior)** | [`train.py:L767`](../../src/algorithms/dreamer_srl/train.py) → [L786](../../src/algorithms/dreamer_srl/train.py) | `wm_opt.update(world_model, wm_grads)` mutates `world_model` IN PLACE (NNX semantics), and the subsequent `world_model.imagine(...)` call at L786 uses the **post-step** WM parameters for the rollout. Sheeprl does the SAME (`world_optimizer.step()` at L200, then rollout starts at L202 with the stepped world_model module). So this is a **match**, NOT a bug. Flagging as 🟡 only because: (a) this is the kind of subtle "in-place mutation between sub-losses" pattern that JIT/eager semantics can desync, and (b) the v2-CP6 grad-test #3 (`∂total_loss / ∂(WM params)`) is the test that empirically verifies this matches sheeprl — if it fails, this is one of the first sites to recheck. | (CP6 standalone) |
| 🟢 nit | [`train.py:L837`](../../src/algorithms/dreamer_srl/train.py) | `H_plus_1` is captured from the factory's closure as a Python int. Under `@nnx.jit`, the inner Python `for h in range(H_plus_1)` loop is unrolled at trace time (which is correct — H is static). But this means the actor is called H+1 = 16 times in the compiled graph (for horizon=15), producing a 16-deep unrolled chain. Consider `jax.lax.scan` for cleaner trace IR, especially after the §3.5 fix (which simplifies the inner body). Pure-performance / IR-readability concern, not correctness. | (CP6 standalone) |
| 🟢 nit | [`train.py:L823-L830`](../../src/algorithms/dreamer_srl/train.py) | `moments_update` is called BEFORE `actor_loss_fn` — fine — but `moments_update` returns `new_moments`, which `one_train_step` returns to the caller. The driver `dreamer_srl_main.py:L537` overwrites `moments = ...` correctly. **No issue**, just calling out: this is the one piece of mutable state that flows through the train-step boundary as a returned value (rather than via in-place `nnx.Optimizer` mutation). Sheeprl's `Moments` class has an in-place buffer (`self.low`, `self.high`) updated inside `Moments.forward`. The dreamer-srl pure-functional analog correctly threads it. | (none) |

---

## 10. Other integration-level checks

### 10.1 Shape consistency

- `imagined_latents`: `[H+1, BT, latent_dim]` = `[16, T*B, S*D + recurrent_state_size]`. Correctly consumed by
  `vmap(reward_model)`, `vmap(target_critic)`, `vmap(continue_model)`, `vmap(critic)` after a flatten to
  `[H+1 * BT, latent_dim]` at L791, L876, L883.
- `predicted_rewards`, `predicted_values`, `continues_predicted`: all `[H+1, BT, 1]` after reshape at L795-L806.
- `lambda_values`: `[H, BT, 1]` (one less than H+1, per `compute_lambda_values` semantics).
- `discount`: `[H+1, BT, 1]`, with `discount[:-1]` consumed in actor and critic loss.
- `log_probs_sliced = log_probs_arr[:-1]`: `[H, BT, 1]` matching sheeprl's `[:-1]` slice at L286.
- `entropy_for_obj = entropies_arr[..., None]`: `[H+1, BT, 1]` — entropy keeps H+1 dimension; `[:-1]` slice
  applied inside `compute_actor_objective` at L594.

**Shape consistency verdict: ✅ all shapes line up with sheeprl's analogs.**

### 10.2 Horizon convention (H vs H+1)

Sheeprl's rollout produces `imagined_trajectories[0..H]` (H+1 latents) and `imagined_actions[0..H]` (H+1
actions). dreamer-srl produces the same H+1-length arrays. The `[:-1]` slice (drop last step, which has no
defined log_prob target since `lambda_values` has length H) is consistently applied at:

- `compute_critic_loss` operates on `qv_logits[:-1]` (caller passes `[:-1]` slice at `train.py:L873`).
- `compute_actor_objective` operates on `log_probs[:-1]` (caller slices at `train.py:L851`).
- `compute_actor_objective` operates on `discount[:-1]` (sliced inside at `train.py:L600`).
- `compute_actor_objective` operates on `entropy[:-1]` (sliced inside at `train.py:L594`).

**Horizon convention verdict: ✅ consistent H+1 collection, [:-1] for loss.**

### 10.3 Gamma / discount application

`gamma` is applied at THREE sites:

1. Inside `compute_lambda_values` at `compute_imagined_returns:L486` — `continues_spliced[1:] * gamma`.
2. Inside `compute_discount` at `train.py:L218` — `jnp.cumprod(continues * gamma, axis=0) / gamma`.
3. Implicitly via the cumprod itself (the cumprod IS the discount cascade).

Sheeprl applies gamma at exactly the same three sites (L254, L260, and inside compute_lambda_values' recursion).
**Match.**

### 10.4 `learning_starts=0` vs `learning_starts=1024`

Per the user's checklist item:

- `learning_starts=0` (smoke config `01_food_only_smoke.yaml`): the train-gate at
  [`dreamer_srl_main.py:L496`](../../src/algorithms/dreamer_srl/dreamer_srl_main.py) `if iter_num >= learning_starts:`
  fires from iter 0 onwards. The `Ratio` class at L499 then returns `int(0 * replay_ratio) = 0` initially
  (well — actually it depends on `Ratio`'s self-correcting state; sheeprl D-014 burst is a known concern).
  No special branch for `learning_starts=0`.
- `learning_starts=1024` (parity config `01_food_only.yaml`): train-gate waits until iter 1024. The Ratio
  scheduler then returns `int(1024 * replay_ratio)` grad steps in the first firing — the D-014 ONE-SHOT BURST.

The orchestrator (`one_train_step`) does NOT branch on `learning_starts`. It is called by the driver per
gradient step, with batch already sampled, and is agnostic to the prefill / learning_starts schedule.
**The train_gate logic correctly handles BOTH cases at the driver level**, and the orchestrator is
schedule-agnostic. **Verdict: ✅** for the orchestrator's role; the D-014 BURST concern remains a CP7-P3
known blocker at the driver, NOT a CP6 issue.

### 10.5 `nnx.jit` + in-place module mutation

`one_train_step` is decorated with `@nnx.jit`. The function performs **three in-place module mutations** via
`nnx.Optimizer.update`:

1. `wm_opt.update(world_model, wm_grads)` at L767 → mutates `world_model.{Param leaves}`.
2. `actor_opt.update(actor, actor_grads)` at L867 → mutates `actor.{Param leaves}`.
3. `critic_opt.update(critic, critic_grads)` at L895 → mutates `critic.{Param leaves}`.

`nnx.jit` handles in-place NNX state mutation via its graph-tracking metaclass — the mutations are recorded as
state transitions and applied functionally inside the compiled graph. This is the standard NNX pattern and is
correct. The polyak update at the driver `dreamer_srl_main.py:L524` `nnx.update(target_critic, new_target_params)`
mutates `target_critic` BEFORE `train_step(...)` is called — so the `target_critic` module captured by reference
inside `train_step` already reflects the post-polyak state when L798 `jax.vmap(target_critic)(imag_flat)` is
evaluated. **Correct.**

Also: `target_critic` is NOT differentiated (no `nnx.value_and_grad(...)(target_critic)` anywhere), so no
optimizer state is associated with it. It is read-only inside `one_train_step` (in the sense that its params
are not mutated by `train_step` itself; only by the driver's polyak step before train_step is called). **Correct.**

---

## 11. Conventions audit checklist

This audit is dreamer-srl orchestrator-specific; not all v1 environment conventions apply. The applicable ones:

| Convention | Status | Note |
|---|---|---|
| Pytree immutability (NNX) | ✅ | `wm_opt.update` etc. via NNX graph-tracked in-place; not raw mutation. |
| JIT trace structure | ✅ | `H_plus_1` captured as Python int in closure; static. `@nnx.jit` decorator correct. |
| PRNG threading (key advancement) | ⚠ | `key, k_wm = jax.random.split(key)`, `key, k_imag = jax.random.split(key)` are correct. But `actor_keys = jax.random.split(jax.random.PRNGKey(0), ...)` at L837 IS the CP3-A2 blocker. |
| Loss-function signature purity | ✅ | `wm_loss_fn`, `actor_loss_fn`, `critic_loss_fn` are pure functions of their single module argument; closure-captured data is sg'd / non-differentiable. |
| Configuration mandatory keys | (n/a — driver-side) | Driver `dreamer_srl_main.py` uses `agent_cfg.get_mandatory(...)`; orchestrator receives static scalars from the factory. |
| Sub-loss ordering matches sheeprl | ✅ | WM → rollout → λ-returns → moments → actor → critic. |
| Optimizer-state isolation | ✅ | Three separate `nnx.Optimizer`, no shared state. |
| Stop-gradient seam discipline | ✅ | All 8 sg sites match sheeprl's `.detach()`. |
| Imagined-actions threading | ❌ | Known CP8-P1 / CP3-A1 blocker materialising at orchestrator. |
| `predicted_values` source | ❌ | Uses `target_critic` instead of live `critic` (CP5-P1 / CP3-A3). |

---

## 12. Conclusion

The v2-CP6 orchestrator re-audit confirms that **dreamer-srl's `one_train_step` is structurally faithful to
sheeprl's `train()` at the integration level** — sub-loss ordering, 3-optimizer isolation, polyak placement,
sg-seam discipline, shape consistency, horizon convention, and gamma application all match. The three known
P-blockers (CP3-A1 resample, CP3-A2 constant seed, CP3-A3 / CP5-P1 stale target_critic) materialise at three
specific orchestrator lines (L788, L837, L798), and the fix bundle for v2-CP9 is mechanical: thread
`imag_outputs["imagined_actions"]` into `actor_loss_fn`, drop the Gumbel resample (which also drops the
constant-seed PRNG), and swap `target_critic` → live `critic` at L798.

**No new P-class blockers surfaced** at the orchestrator audit surface. One 🟡 concern (WM-imagine using
post-step WM params, §9 row 5) is flagged for explicit verification by v2-CP6 grad-test #3 — but the pattern
matches sheeprl exactly, so it is likely fine.

**Verdict**: ❌ **FAIL** (because the three known P-blockers materialise here) — but the orchestrator's
**integration scaffolding is correct**, and the fix bundle's mechanical changes are localised to three lines.

---

## 13. Hand-off

- **For v2-CP9 developer**: implement the §3.5 pseudocode fix at `train.py:L788, L837, L845` (thread
  imagined_actions, remove resample, remove constant-seed PRNG line) and the L798 fix (target_critic → critic).
  After landing, run the 5 grad-parity tests per §8 (`test_train_grad.py`). Thresholds: `2e-3` for tests #1-#2,
  `3e-3` for test #3, `5e-5` for tests #4-#5.
- **For v2-CP10 experiment-analyzer**: once the fix bundle lands and grad parity passes, the policy-learning
  gate (`ep_len_avg > 200` over the 16k–20k window) is the empirical verification. If grad parity passes AND
  CP7-P1/P2/P3 driver fixes land AND CP10 still fails, the candidate cause is the aggregate substrate-class
  drift (D-001..D-014) per CP1 §9 retrospective #5.
- **For the parent agent**: this review's findings are duplicative of CP3 / CP5 / CP8 at the orchestrator
  surface — by design (re-audit of integration site). No new fixes implied beyond the already-planned v2-CP9
  bundle. Auto-commit + diary append per the standing protocol.

---

Reviewed by: code-reviewer
Date: 2026-05-14
Sheeprl pin: 33b6366
dreamer-srl HEAD: 197e212
