---
title: "v2-CP3 — compute_actor_objective forward + gradient re-audit (H1 hypothesis-locus)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
---

# v2-CP3 — compute_actor_objective re-audit

## Verdict (plain-language entry point)

**What this review is.** A fresh re-audit of the dreamer-srl actor-objective code
([`src/algorithms/dreamer_srl/train.py:L500-L604`](../../src/algorithms/dreamer_srl/train.py))
**and** the actor-loss orchestrator that calls it
([`src/algorithms/dreamer_srl/train.py:L832-L867`](../../src/algorithms/dreamer_srl/train.py)),
against the canonical sheeprl reference
([`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L272-L304`](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py)),
performed without trusting any v1 review under `docs/reviews/dreamer_srl_v3_cp7_*`. This is
the v2-CP3 deliverable and the **H1 hypothesis surface** in the v2 plan — H1 says the
random-policy-floor parity failure is rooted in a gradient-side bug in the actor objective
(missing `sg`, mis-ported entropy, mis-ported `unimix`, REINFORCE term mis-port).
v2-CP7 confirmed Actor.__call__ itself is structurally clean — the canonical `sg(action)`,
straight-through `hard - sg(soft) + soft`, and probability-space `_uniform_mix` are all in
place — so anything wrong with H1 must live in `compute_actor_objective` itself or in the
orchestrator that builds its inputs. That orchestrator block is in scope here.

**Headline verdict.** **❌ FAIL — substantive H1-class divergence found in the
orchestrator, not in `compute_actor_objective` itself.**

`compute_actor_objective` (the function body, `train.py:L500-L604`) is **structurally
correct** against sheeprl's `dreamer_v3.py:L274-L297`. Forward math matches term-by-term;
`sg(advantage)` at L591 is correct; entropy is added (not subtracted) and the negation is
at the outer `-jnp.mean(...)` (L602); the `[:-1]` slices on entropy and discount match
sheeprl L295-L296.

But the **call site** in `make_train_step` (`train.py:L832-L867`) commits two
H1-class structural bugs:

1. **A1 (🔴 BLOCKER): the actor-loss log-prob is computed on RE-SAMPLED actions, not
   on the imagined-rollout actions.** Sheeprl L286: `p.log_prob(imgnd_act.detach())`
   evaluates the actor's log-prob on the **action that was actually sampled during
   imagination** (i.e. `imagined_actions[i]` from the rollout). dreamer-srl does NOT do
   this. The orchestrator throws away `imag_outputs["imagined_log_probs"]` (which were
   computed inside `world_model.imagine` with the *rollout* PRNG keys) and instead
   re-runs `actor_module(sg_latents[h], actor_keys[h])` inside `actor_loss_fn` at
   `train.py:L845`, drawing **fresh Gumbel samples** with the orchestrator's keys. The
   action that REINFORCE multiplies against `sg(advantage)` is therefore a *different*
   action from the one that produced the advantage. The estimator is no longer
   REINFORCE — it is a biased Frankenstein that pairs `log_prob(a_new)` with
   `advantage(a_old)`. **This is sufficient on its own to drive the policy to the random
   floor**, because the gradient signal is decorrelated from the action that actually
   produced the value.

2. **A2 (🔴 BLOCKER): the PRNG key used for that re-sample is a hard-coded constant
   `jax.random.PRNGKey(0)`** at `train.py:L837`. Under JIT-tracing this becomes a baked-in
   constant. Every train step re-uses the same Gumbel noise; there is **zero PRNG
   diversity across train steps for the actor-loss resample**. Even setting aside A1, this
   means the actor sees the same fictitious action draw thousands of times in a row, with
   no decorrelation from the advantage signal. Combined with A1 this guarantees the
   REINFORCE estimator is broken.

A third, lower-severity finding is **A3 (🟡 concern): `predicted_values` used in the
advantage is from `target_critic`, not the live `critic`** (`train.py:L798`), whereas
sheeprl uses the live critic at `dreamer_v3.py:L244`. This is a CP4 / CP5 cross-cutting
concern (the orchestrator block is shared) but it cleanly enters CP3's surface because
the advantage formula consumes it.

Additionally, **A4 (🟡 concern): the entropy site uses `-Σ probs · log(probs + 1e-8)`**
at `agent.py:L1540` — sheeprl uses `Categorical.entropy()` which is `-Σ probs ·
log_softmax(logits)`. The two are algebraically equal (`log_softmax(logits) = log(probs)`)
but the `+1e-8` clamp introduces a small bias for very-low-probability classes; at
`unimix=0.01, num_classes=5` the smallest post-`unimix` probability floor is `0.01/5 =
0.002`, well above `1e-8`, so the clamp is **inactive in practice** but is still a
substrate-class deviation worth logging.

H1 is now elevated from MED-HIGH to **HIGH confidence**, with the root cause precisely
localised to A1 + A2 in the orchestrator call site, NOT inside `compute_actor_objective`
itself.

---

## 1. Forward parity (the bit-identity test that v1 already did)

`compute_actor_objective` (`train.py:L500-L604`) is line-for-line correct against sheeprl
`dreamer_v3.py:L274-L297`. The mapping:

| dreamer-srl line | sheeprl line | Verified |
|---|---|---|
| `train.py:L575` `baseline = predicted_values[:-1]` | `dreamer_v3.py:L275` `baseline = predicted_values[:-1]` | ✅ |
| `train.py:L579` `normed_lambda_values = (lambda_values - moments_offset) / moments_invscale` | `dreamer_v3.py:L277` `normed_lambda_values = (lambda_values - offset) / invscale` | ✅ |
| `train.py:L581` `normed_baseline = (baseline - moments_offset) / moments_invscale` | `dreamer_v3.py:L278` `normed_baseline = (baseline - offset) / invscale` | ✅ |
| `train.py:L585` `advantage = normed_lambda_values - normed_baseline` | `dreamer_v3.py:L279` `advantage = normed_lambda_values - normed_baseline` | ✅ |
| `train.py:L591` `objective = log_probs * jax.lax.stop_gradient(advantage)` | `dreamer_v3.py:L283-L291` `log_prob(action.detach()).sum * advantage.detach()` | ✅ |
| `train.py:L594` `entropy_term = ent_coef * entropy[:-1]` | `dreamer_v3.py:L294-L295` `ent_coef * sum(p.entropy()) ; .unsqueeze(-1)[:-1]` | ✅ |
| `train.py:L600` `discount_weights = discount[:-1]` | `dreamer_v3.py:L296` `discount[:-1].detach()` | ✅ |
| `train.py:L602` `policy_loss = -jnp.mean(discount_weights * (objective + entropy_term))` | `dreamer_v3.py:L297` `policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))` | ✅ |

The "offset cancels algebraically" docstring note at `train.py:L527-L532` is correct (per-term
form is preserved for floating-point bit-identity). Forward parity is **clean** — no
divergence inside `compute_actor_objective` itself.

---

## 2. The REINFORCE term gradient (high-priority H1 surface)

Sheeprl's actor objective is:

```
policy_loss = -E[ discount(t) * ( log_prob(a_t) * sg(advantage(t)) + ent_coef * H(t) ) ]
```

The gradient w.r.t. actor params flows ONLY through `log_prob(action)` and
`H(t) = entropy`. `advantage` is sg-detached; `discount` is sg-detached
(`compute_discount` already applies `jax.lax.stop_gradient`, `train.py:L216-L217`);
`action` is sg-detached (the actor returns `log_probs = sum(sg(action) * log_softmax(logits))`
at `agent.py:L1535-L1536`).

In `compute_actor_objective` proper, all four sg sites are correctly placed:

- **`sg(advantage)`** at `train.py:L591` ✅
- **`sg(action)` inside log_probs** at `agent.py:L1535` (the actor's `__call__`
  applies it before the `sum(action * log_softmax)` product) ✅
- **`sg(discount)`** at `train.py:L216-L217` (inside `compute_discount`) ✅
- **`sg(lambda_values)` / `sg(predicted_values)` / `sg(moments_offset)` /
  `sg(moments_invscale)`** at `train.py:L856-L859` — the orchestrator wraps each
  external input in `jax.lax.stop_gradient` before passing into `compute_actor_objective`
  ✅

So the function-internal gradient structure is clean. **But:**

### 2.1 The orchestrator-level mis-port (A1)

The H1 bug is *upstream* of `compute_actor_objective`, in `actor_loss_fn` itself
(`train.py:L840-L864`):

```python
def actor_loss_fn(actor_module):
    """Actor loss: REINFORCE + entropy, §S7 advantage normalization."""
    all_log_probs = []
    all_entropies = []
    for h in range(H_plus_1):  # Python loop — H_plus_1 is a static Python int
        _, lp_h, ent_h = actor_module(sg_latents[h], actor_keys[h])    # ← L845
        all_log_probs.append(lp_h)
        all_entropies.append(ent_h)
    log_probs_arr = jnp.stack(all_log_probs, axis=0)
    entropies_arr = jnp.stack(all_entropies, axis=0)

    log_probs_sliced = log_probs_arr[:-1]
    entropy_for_obj = entropies_arr[..., None]

    policy_loss, _, _ = compute_actor_objective(
        log_probs=log_probs_sliced,
        ...
    )
    return policy_loss
```

`actor_module(sg_latents[h], actor_keys[h])` returns `(actions, log_probs, entropy)`
where (per `agent.py:L1519-L1536`):

```python
# at L1519-L1526
gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
perturbed = logits + gumbel_noise
hard_indices = jnp.argmax(perturbed, axis=-1)
hard = jax.nn.one_hot(hard_indices, self.action_dim)
soft = jax.nn.softmax(logits, axis=-1)
actions = hard - jax.lax.stop_gradient(soft) + soft

# at L1535-L1536
sg_actions = jax.lax.stop_gradient(actions)
log_probs = jnp.sum(sg_actions * log_softmax_logits, axis=-1, keepdims=True)
```

So inside `actor_loss_fn`, `lp_h` is `sum(sg(NEW_action) * log_softmax(logits))` where
`NEW_action` is drawn from `gumbel(actor_keys[h]) + logits`. This is **a fresh sample
from the actor's current policy**, NOT the action that produced `lambda_values` /
`predicted_values` / `advantage`.

In sheeprl L283-L291, by contrast:

```python
objective = (
    torch.stack(
        [
            p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
            for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
        ],
        dim=-1,
    ).sum(dim=-1)
    * advantage.detach()
)
```

`imagined_actions` is the array of actions that **the rollout actually sampled**
(populated at `dreamer_v3.py:L219, L241`). `p.log_prob(imgnd_act.detach())` evaluates
the policy's log-prob ON THAT SPECIFIC ACTION TOKEN. The REINFORCE estimator pairs:

- the **log-prob** of the rollout-time action under the *current* policy,
- with the **advantage** that the rollout-time action produced.

Dreamer-srl pairs:

- the **log-prob** of a *new, freshly-sampled* action under the current policy (a
  Gumbel draw with a different key),
- with the **advantage** that the rollout-time (different) action produced.

This is **not REINFORCE**. The gradient `∂(log_prob(a_new) * sg(adv_old)) / ∂θ` does not
estimate the policy gradient `∇_θ E_a[ p_θ(a) * A(a) ]`, because `a_new` and the
advantage's `a_old` are independent samples. The expectation under
`a_new ~ p_θ(·|state)` of `log_prob(a_new)` is `-H(p_θ(·|state))`, so the dreamer-srl
"REINFORCE" term degenerates in expectation to `-E[H · advantage]`, which is a
**variance-weighted entropy gradient**, not a policy gradient. The actor learns to
collapse or expand entropy as a function of advantage sign, not to actually prefer
high-advantage actions.

`imag_outputs["imagined_log_probs"]` (`train.py:L788` returns `imag_outputs` but only
extracts `"imagined_latents"`) IS computed correctly inside `world_model.imagine`
(`agent.py:L1755, L1793`), using the rollout-time PRNG keys and the rollout-time
sampled actions. It is **silently discarded** by the orchestrator.

### 2.2 The PRNG-constant bug (A2)

At `train.py:L837`:

```python
actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)
```

`jax.random.PRNGKey(0)` is a literal constant. When `make_train_step` is JIT-compiled
(via `nnx.value_and_grad(actor_loss_fn)`), `actor_keys` becomes a baked-in constant
array. Every train step uses the **same H+1 keys** to draw Gumbel noise inside the
actor. The "fresh sample" in A1 is therefore deterministic across the entire training
run.

Two compounding effects:

1. **Zero exploration diversity in the actor-loss gradient computation.** Across 200k
   train steps, the same `hard_indices = argmax(logits + gumbel_noise_constant)` is
   computed; whichever action wins at step 0 keeps winning unless `logits` changes
   enough to flip the argmax. This is far from sheeprl's behaviour where
   `OneHotCategoricalStraightThrough.rsample()` draws fresh samples on every training
   call.
2. **Decorrelation guarantee.** The action that *would* match the imagined-rollout
   action is the one drawn with the rollout-time PRNG keys (`k_act` derived from
   `k_imag`, `train.py:L785-L786` and `agent.py:L1754, L1767`). The PRNGKey(0)
   constant has no path to those keys. So `a_new` and `a_old` are guaranteed
   uncorrelated.

### 2.3 The combined effect

A1+A2 together explain the random-floor parity failure WITHOUT requiring any of
H2's driver-side bugs (P1/P2/P3 from v2-CP7) to also be present. They are independent
sufficient causes. They also explain why v1's forward-only tests passed: forward
evaluation of `compute_actor_objective(log_probs_FROM_RESAMPLE, ...)` produces a
mathematically valid scalar; only the *gradient flowing through* `log_probs` reveals
that the gradient signal is uncorrelated from the advantage signal.

### 2.4 Suggested fix (for v2-CP9 developer)

The fix is to use the `imagined_log_probs` from the rollout, OR to re-run the actor
with the *rollout-time keys* and add a `sg(imagined_actions)`-equivalent constraint
(force the argmax to match the imagined action). The path-of-least-resistance fix
mirrors the sheeprl pattern:

```python
def actor_loss_fn(actor_module):
    # Re-run actor to get current-policy logits at sg(latents). We need NEW logits
    # because the actor params changed since rollout time. BUT we then compute
    # log_prob on the ALREADY-SAMPLED rollout actions, not on a fresh draw.
    all_log_probs = []
    all_entropies = []
    for h in range(H_plus_1):
        latent_h = sg_latents[h]
        # Compute current-policy log_softmax and entropy directly from the head,
        # without re-sampling.
        logits_h = actor_module.forward_logits(latent_h)  # NEW helper, no sampling
        log_softmax_h = jax.nn.log_softmax(logits_h, axis=-1)
        # log_prob of the rollout-time action under the current policy
        action_h = jax.lax.stop_gradient(imagined_actions[h])  # from imag_outputs
        lp_h = jnp.sum(action_h * log_softmax_h, axis=-1, keepdims=True)
        ent_h = -jnp.sum(jnp.exp(log_softmax_h) * log_softmax_h, axis=-1)
        all_log_probs.append(lp_h)
        all_entropies.append(ent_h)
    ...
```

This requires (a) extracting `imagined_actions` from `imag_outputs` (currently
discarded), (b) adding a `forward_logits` method to `Actor` that bypasses the
sampling branch, OR adapting the existing `__call__` to accept an
"`action_override`" parameter that skips Gumbel-sample and uses the override action.

---

## 3. The `sg(advantage)` leak diagnostic

Per `GRAD_PARITY_METHODOLOGY.md` §2.3, the canonical test is:

```python
def jax_loss_wrt_advantage(advantage):
    # advantage is the ONLY non-sg'd input
    return compute_actor_objective(
        log_probs=fixture_lp,
        lambda_values=fixture_lv,         # NOTE: lambda_values is sg'd inside the orchestrator,
        predicted_values=fixture_pv,       # but inside compute_actor_objective itself, the offset/invscale
        moments_offset=fixture_off,
        moments_invscale=fixture_inv,
        entropy=fixture_ent,
        discount=fixture_disc,
        ent_coef=3e-4,
    )[0]
grad_wrt_advantage = jax.grad(jax_loss_wrt_advantage)(...)
# Expected: exactly zero (sg(advantage) at train.py:L591 blocks the gradient)
```

**But:** `compute_actor_objective`'s `advantage` is **not an input**. Advantage is
*computed inside* the function from `lambda_values - predicted_values` (via the
normalization at L579-L585). So the proper grad-leak test takes the gradient w.r.t.
`lambda_values` or `predicted_values` directly:

```python
def jax_loss_wrt_lambda(lambda_values):
    return compute_actor_objective(
        log_probs=fixture_lp,
        lambda_values=lambda_values,
        predicted_values=fixture_pv,
        moments_offset=fixture_off,
        moments_invscale=fixture_inv,
        entropy=fixture_ent,
        discount=fixture_disc,
        ent_coef=3e-4,
    )[0]
grad_wrt_lambda = jax.grad(jax_loss_wrt_lambda)(fixture_lv)
# Expected: exactly zero, because:
#   advantage = (lambda - offset)/invscale - (baseline - offset)/invscale
#   policy_loss = -mean(disc * (log_probs * sg(advantage) + ent_coef * entropy))
#   Since sg(advantage) is at L591, gradient w.r.t. lambda_values is zero through that path.
#   No other path from lambda_values to policy_loss exists inside compute_actor_objective.
```

Inspection confirms `sg(advantage)` is correctly placed at `train.py:L591`. The
remaining advantage paths inside `compute_actor_objective` (L579, L581, L585) all
feed into the sg'd term at L591 — there is no second copy of `advantage` outside the
sg. Predicted: this gradient is exactly zero.

For the **end-to-end** loss (orchestrator + actor + compute_actor_objective), the
`sg_latents = jax.lax.stop_gradient(imagined_latents)` (`train.py:L838`) and the explicit
`jax.lax.stop_gradient(lambda_values)`, `jax.lax.stop_gradient(predicted_values)`,
`jax.lax.stop_gradient(moments_offset)`, `jax.lax.stop_gradient(moments_invscale)` at
L856-L859 are correct and consistent with sheeprl's `imagined_trajectories.detach()`
(L273) and the implicit detach on `Moments.forward` outputs at sheeprl `utils.py:L63`.

**Verdict: sg(advantage), sg(action), sg(discount), sg(lambda_values), sg(predicted_values),
sg(moments_offset/invscale) are ALL correctly placed in dreamer-srl.** The H1 bug is NOT a
missing sg. It is the orchestrator's choice of fresh-sample-rather-than-rollout-sample
(A1) and the PRNG-constant key (A2).

---

## 4. The advantage normalization (cross-cuts H3)

Sheeprl's normalization (`dreamer_v3.py:L275-L279`):

```python
baseline = predicted_values[:-1]
offset, invscale = moments(lambda_values, fabric)
normed_lambda_values = (lambda_values - offset) / invscale
normed_baseline = (baseline - offset) / invscale
advantage = normed_lambda_values - normed_baseline
```

where `moments(lambda_values, fabric)` returns `(low, invscale)` with `low = EMA-of-5th-quantile`,
`invscale = max(1/Moments._max, high_EMA - low_EMA)` (sheeprl `utils.py:L60-L63`). It is
**NOT** `max(1, sigma_returns)` (a per-batch std) — the brief's phrasing is sloppy
shorthand. The actual quantity is the **EMA of the 5th-to-95th-percentile range** of
`lambda_values`.

**dreamer-srl analog** (`utils.py:L217-L263`, `moments_update`):

```python
x_flat = x.astype(jnp.float32).ravel()
low_new = jnp.quantile(x_flat, percentile_low)         # 5th percentile
high_new = jnp.quantile(x_flat, percentile_high)       # 95th percentile
new_low = decay * state.low + (1 - decay) * low_new    # EMA
new_high = decay * state.high + (1 - decay) * high_new # EMA
new_state = MomentsState(low=new_low, high=new_high)
invscale = jnp.maximum(1.0 / max_, new_high - new_low)
return new_state, new_low, invscale
```

Formula match against sheeprl `utils.py:L40-L63`:
- ✅ `decay * state.low + (1 - decay) * low_new` matches sheeprl L60.
- ✅ `jnp.maximum(1.0/max_, new_high - new_low)` matches sheeprl L62 (`torch.max(1/self._max, self.high - self.low)`).
- ✅ Returned tuple shape `(new_state, offset, invscale)` consumed correctly at
  `train.py:L823-L830`.

**Where does the normalization happen?** Inside `compute_actor_objective`
(`train.py:L575-L585`), per the docstring at `train.py:L526-L532`. This matches
sheeprl's location at `dreamer_v3.py:L275-L279` (also inside the actor block).

**Cross-cut with v2-CP7 P2.** v2-CP7 confirmed that `terminated` and `truncated` are
conflated in the driver. That means when `lambda_values` is computed (via
`compute_imagined_returns` at `train.py:L811-L818`), the §S5 true-continue splice
`true_continue = 1 - terminated_observed` (`train.py:L475`) consumes a `terminated`
column that is `1` at every episode boundary (it should be `0` on a food-only
substrate where every done is a max-steps truncation). So `continues_spliced[0] = 0`
at every boundary, which zeros the discount and bootstrap at the first imagined step.

**Does `compute_actor_objective` re-zero the advantage at terminated steps?** No —
`compute_actor_objective` does not consume `terminated` directly. It receives
`lambda_values` (already with the §S5 boundary-zeroing applied upstream) and `discount`
(already with the §S6 cumprod applied upstream). It applies neither zeroing nor
re-zeroing. So there is **no double-zeroing**. The double-trouble is "P2 zeros at the
boundary upstream, advantage is then computed from the corrupted lambda_values
downstream" — a single zero, propagated, not a double zero.

This means CP3's surface is clean of the P2-double-zero concern. P2's amplification
of A1+A2 is via the *advantage-signal corruption*, not via a sg site inside
`compute_actor_objective`.

---

## 5. Entropy site

**Sheeprl's entropy** (`dreamer_v3.py:L294`):

```python
entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
```

where `p` is a `OneHotCategoricalStraightThrough(logits=unimix_logits)`. PyTorch's
`OneHotCategorical.entropy()` is `-sum(probs * log_softmax(logits), axis=-1)` (the
algebraically-clean form that uses `log_softmax(logits)` rather than `log(probs)`).

**dreamer-srl's entropy** (`agent.py:L1539-L1540`):

```python
probs = jax.nn.softmax(logits, axis=-1)
entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
```

**Algebraic equivalence**: `log_softmax(logits) ≡ log(probs)` for the same `logits`,
so the two forms are algebraically equal *in exact arithmetic*.

**Numerical divergence**: dreamer-srl uses `log(probs + 1e-8)`, sheeprl uses
`log_softmax(logits)`. At `unimix=0.01, num_classes=5`, the minimum post-unimix
probability is `0.01/5 = 0.002` (well above `1e-8`), so the clamp is inactive in
practice. But:

- For a categorical with `num_classes >> 100` (not the food task), the clamp could
  activate.
- `log_softmax(logits)` is numerically stabler at saturated logits (it subtracts the
  max before exp). `log(probs + 1e-8)` after `softmax` is two evaluations of `exp` +
  `log` that don't share the numerical hardening; substrate-class drift here is
  bounded by ~1-2 ULP at saturated logits.

**Sign convention** (`train.py:L594, L602`):

```python
entropy_term = ent_coef * entropy[:-1]
policy_loss = -jnp.mean(discount_weights * (objective + entropy_term))
```

This is `policy_loss = -(REINFORCE + ent_coef * H)`, so the entropy is **added** inside
the parentheses and **negated** by the outer `-jnp.mean`. Minimising `policy_loss`
therefore **maximises** entropy (correct: entropy is a regulariser that prevents
premature collapse). Sign matches sheeprl L294, L297. ✅

**Gradient through entropy.** Entropy at L1540 is computed from `probs` which is
`softmax(logits)`, and `logits` flow through the actor's MLP — so gradient correctly
flows from `entropy[:-1]` back into actor params. ✅

**Verdict: entropy site is structurally correct; only the `+1e-8` clamp is a
substrate-class deviation (D-class candidate; ULP-floor only, not H1 sufficient).**
Log as D-015 candidate at v2-CP3 verification time IF the entropy-only grad-parity
test (manifest site #3) exceeds the threshold.

---

## 6. Gradient bit-identity (the Lever-A test specification for v2-CP9)

Per `GRAD_PARITY_METHODOLOGY.md` §5.1, v2-CP3 owns 5 test sites. Below is the
specification for each; the developer at v2-CP9 implements them.

### 6.1 Fixture-seed convention

- **Param seed**: `0xD3EAF + 0x100 * 3 = 866223`
- **Input seed**: `0xD3EAF + 0x100 * 3 + 1 = 866224`
- **Path-A action draw seed**: `0xD3EAF + 0x100 * 3 + 2 = 866225`

### 6.2 Fixture-input file paths

All under `tests/fixtures/dreamer_srl/`:

- `grad_cp3_reinforce_lastlayer.npz` — params + inputs for test #1 (REINFORCE
  last-layer)
- `grad_cp3_reinforce_trunk.npz` — params + inputs for test #2 (REINFORCE trunk
  cascade)
- `grad_cp3_entropy_only.npz` — params + inputs for test #3 (entropy isolated by
  `advantage=0`)
- `grad_cp3_sg_advantage.npz` — params + inputs for test #4 (sg-leak check)
- `grad_cp3_sg_action.npz` — params + inputs for test #5 (sg-leak check)

Generator script: `scripts/fixtures/gen_dreamer_srl_grad_fixtures.py` (new, per
`GRAD_PARITY_METHODOLOGY.md` §5).

### 6.3 Test pseudocode (canonical template — adapt per site)

```python
# tests/algorithms/dreamer_srl/test_loss_grad.py

import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest

from src.algorithms.dreamer_srl.train import compute_actor_objective
from src.algorithms.dreamer_srl.agent import Actor

# v2-CP3 fixture seeds
FIXTURE_PARAM_SEED = 0xD3EAF + 0x100 * 3   # 866223
FIXTURE_INPUT_SEED = 0xD3EAF + 0x100 * 3 + 1
FIXTURE_ACTION_SEED = 0xD3EAF + 0x100 * 3 + 2

H = 15            # horizon
BT = 16 * 64      # batch_size * sequence_length
D_LATENT = 1024 + 32 * 32   # recurrent_state_size + stochastic_size
A_DIM = 5         # food-task action_dim
ENT_COEF = 3e-4   # XS recipe default
UNIMIX = 0.01


def _build_fixtures():
    """Construct (params_jax, inputs_jax, params_torch, inputs_torch) at fixed seeds.

    Path A: the action one-hot tensor is pre-sampled from numpy and passed
    identically to both sides. The straight-through Gumbel-softmax sampler is
    BYPASSED in this test — the test exercises the REINFORCE gradient through
    log_prob(fixed_action), not through the sample operation. Per
    GRAD_PARITY_METHODOLOGY §4.1 path A.
    """
    # Generate latent + advantage + lambda_values + entropy + discount fixtures
    np_rng = np.random.RandomState(FIXTURE_INPUT_SEED)
    latents = np_rng.normal(size=(H+1, BT, D_LATENT)).astype(np.float32)
    lambda_values = np_rng.normal(size=(H, BT, 1)).astype(np.float32)
    predicted_values = np_rng.normal(size=(H+1, BT, 1)).astype(np.float32)
    moments_offset = np.float32(0.0)
    moments_invscale = np.float32(1.0)
    entropy = np_rng.uniform(0.5, 1.5, size=(H+1, BT, 1)).astype(np.float32)
    discount = np.ones((H+1, BT, 1), dtype=np.float32)  # γ=1 fixture for cleanliness

    # Path-A action draw
    act_rng = np.random.RandomState(FIXTURE_ACTION_SEED)
    action_indices = act_rng.choice(A_DIM, size=(H+1, BT))
    action_one_hot = np.eye(A_DIM, dtype=np.float32)[action_indices]  # [H+1, BT, A_DIM]

    # JAX side
    inputs_jax = dict(
        latents=jnp.asarray(latents),
        action_one_hot=jnp.asarray(action_one_hot),
        lambda_values=jnp.asarray(lambda_values),
        predicted_values=jnp.asarray(predicted_values),
        moments_offset=jnp.asarray(moments_offset),
        moments_invscale=jnp.asarray(moments_invscale),
        entropy=jnp.asarray(entropy),
        discount=jnp.asarray(discount),
    )

    # JAX params
    actor_jax = Actor(
        latent_size=D_LATENT, action_dim=A_DIM, dense_units=1024,
        mlp_layers=5, unimix=UNIMIX,
        rngs=nnx.Rngs(FIXTURE_PARAM_SEED),
    )
    params_jax = nnx.state(actor_jax)

    # Torch side: mirror parameter-by-parameter
    actor_torch = _build_torch_actor(latent_size=D_LATENT, action_dim=A_DIM,
                                      dense_units=1024, mlp_layers=5, unimix=UNIMIX)
    _copy_jax_actor_params_to_torch(actor_jax, actor_torch)
    inputs_torch = {k: torch.from_numpy(np.asarray(v)) for k, v in inputs_jax.items()}

    return actor_jax, inputs_jax, actor_torch, inputs_torch


def test_actor_reinforce_lastlayer_grad_matches_sheeprl():
    """v2-CP3 #1 — ∂L_actor/∂(actor output_linear.kernel) via REINFORCE term.

    Threshold: 5e-5 (per GRAD_PARITY_METHODOLOGY §3.4 — D-006/D-010 substrate
    class for a single matmul + log_softmax cascade).
    """
    actor_jax, inputs_jax, actor_torch, inputs_torch = _build_fixtures()

    # --- JAX side ---
    def jax_loss(actor_module):
        # Compute log_probs at fixed action (path A — bypass sampler)
        all_log_probs = []
        all_entropies = []
        for h in range(H+1):
            logits = actor_module.forward_logits(inputs_jax["latents"][h])  # NEW helper
            log_softmax = jax.nn.log_softmax(logits, axis=-1)
            lp = jnp.sum(inputs_jax["action_one_hot"][h] * log_softmax, axis=-1, keepdims=True)
            probs = jax.nn.softmax(logits, axis=-1)
            ent = -jnp.sum(probs * log_softmax, axis=-1)
            all_log_probs.append(lp)
            all_entropies.append(ent)
        log_probs = jnp.stack(all_log_probs, axis=0)[:-1]   # [H, BT, 1]
        entropies = jnp.stack(all_entropies, axis=0)[..., None]  # [H+1, BT, 1]

        policy_loss, _, _ = compute_actor_objective(
            log_probs=log_probs,
            lambda_values=inputs_jax["lambda_values"],
            predicted_values=inputs_jax["predicted_values"],
            moments_offset=inputs_jax["moments_offset"],
            moments_invscale=inputs_jax["moments_invscale"],
            entropy=entropies,
            discount=inputs_jax["discount"],
            ent_coef=ENT_COEF,
        )
        return policy_loss

    jax_grads = nnx.grad(jax_loss)(actor_jax)
    jax_lastlayer_grad = np.asarray(jax_grads.output_linear.kernel.value)

    # --- PyTorch (sheeprl) side ---
    # Build the sheeprl actor-objective form using the SAME inputs.
    policy_loss_torch = _sheeprl_actor_objective(
        actor_torch=actor_torch,
        latents=inputs_torch["latents"],
        action_one_hot=inputs_torch["action_one_hot"],
        lambda_values=inputs_torch["lambda_values"],
        predicted_values=inputs_torch["predicted_values"],
        moments_offset=inputs_torch["moments_offset"],
        moments_invscale=inputs_torch["moments_invscale"],
        discount=inputs_torch["discount"],
        ent_coef=ENT_COEF,
    )
    torch_grads = torch.autograd.grad(
        policy_loss_torch,
        list(actor_torch.parameters()),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    torch_lastlayer_grad = torch_grads[-1].detach().numpy()  # last in module order

    max_abs_diff = float(np.max(np.abs(jax_lastlayer_grad - torch_lastlayer_grad)))
    threshold = 5e-5
    assert max_abs_diff < threshold, (
        f"REINFORCE last-layer grad FAIL: max_abs_diff={max_abs_diff:.3e} "
        f">= threshold={threshold:.3e}"
    )


def test_actor_reinforce_trunk_grad_matches_sheeprl():
    """v2-CP3 #2 — ∂L_actor/∂(actor MLP trunk weights) via REINFORCE.

    Threshold: 2e-3 (per GRAD_PARITY_METHODOLOGY §3.4 — D-008 class for
    cascaded 5-layer MLP + LayerNorm + SiLU + matmul + log_softmax).
    """
    # Same fixture + loss; assert on hidden_linears[0..4].kernel grads
    ...
    threshold = 2e-3


def test_actor_entropy_only_grad_matches_sheeprl():
    """v2-CP3 #3 — ∂L_actor/∂(actor params) via entropy term only.

    Set lambda_values = predicted_values so advantage = 0 → REINFORCE term
    zeroed; only entropy contributes. Verifies the entropy formula and
    gradient flow.

    Threshold: 5e-5 (D-006 class).

    Catches: ent_coef mis-port; entropy formula mis-port; entropy sign flip.
    """
    inputs_modified = dict(inputs_jax,
        lambda_values=inputs_jax["predicted_values"][:-1])
    ...
    threshold = 5e-5


def test_actor_sg_advantage_leak():
    """v2-CP3 #4 — sg-leak check: ∂L_actor/∂lambda_values should be exactly 0.

    Since `advantage = (lambda_values - offset)/invscale - (baseline - offset)/invscale`
    and `objective = log_probs * sg(advantage)`, no gradient should flow from
    policy_loss to lambda_values. Test takes grad w.r.t. lambda_values.

    Threshold: 1e-7 (tight — should be exactly zero).
    """
    def loss_wrt_lambda(lv):
        new_inputs = dict(inputs_jax, lambda_values=lv)
        return _full_actor_loss_with_fixed_action(actor_jax, **new_inputs)
    grad = jax.grad(loss_wrt_lambda)(inputs_jax["lambda_values"])
    max_leak = float(jnp.max(jnp.abs(grad)))
    assert max_leak < 1e-7, f"sg(advantage) LEAK: |∂L/∂lambda_values|={max_leak:.3e}"


def test_actor_sg_action_leak():
    """v2-CP3 #5 — sg-leak check: ∂L_actor/∂action_one_hot should be exactly 0.

    Since `log_probs = sum(sg(action) * log_softmax(logits))` per agent.py:L1535-L1536,
    no gradient should flow from log_probs into the action tensor. Test takes
    grad w.r.t. action_one_hot.

    Threshold: 1e-7.
    """
    def loss_wrt_action(a):
        new_inputs = dict(inputs_jax, action_one_hot=a)
        return _full_actor_loss_with_fixed_action(actor_jax, **new_inputs)
    grad = jax.grad(loss_wrt_action)(inputs_jax["action_one_hot"])
    max_leak = float(jnp.max(jnp.abs(grad)))
    assert max_leak < 1e-7, f"sg(action) LEAK: |∂L/∂action|={max_leak:.3e}"
```

### 6.4 Expected threshold per site (from `GRAD_PARITY_METHODOLOGY.md` §3.4)

| # | Site | Threshold | Anchor |
|---|---|---|---|
| 1 | REINFORCE last-layer (`output_linear.kernel`) | **`5e-5`** | D-006 / D-010 class (single matmul + log_softmax) |
| 2 | REINFORCE trunk (`hidden_linears[0..4].kernel`, `hidden_norms[*]`) | **`2e-3`** | D-008 class (5-layer MLP cascade) |
| 3 | Entropy-only on full actor params | **`5e-5`** | D-006 class (entropy isolated, single backward path) |
| 4 | sg(advantage) via `∂L/∂lambda_values` | **`1e-7`** | tight — must be ~0 |
| 5 | sg(action) via `∂L/∂action_one_hot` | **`1e-7`** | tight — must be ~0 |

### 6.5 What the test will reveal (predictions)

- **Test #1 (REINFORCE last-layer)**: If executed on the CURRENT dreamer-srl code with
  the path-A FIXED action passed through `Actor.__call__` (which `sg`s the action
  inside log_prob), the gradient should match sheeprl within `5e-5` because
  `compute_actor_objective` itself is bit-identity correct. **The test as
  specified above bypasses the orchestrator's A1/A2 bug** because it manually pairs
  `(fixed_action, lambda_values)` consistent with sheeprl. So this test PASSES on the
  current code, even though training fails. A separate, more targeted test (see §6.6
  below) is required to surface A1/A2.

- **Test #2 (REINFORCE trunk)**: Same as #1 — passes on current code.

- **Test #3 (entropy-only)**: Passes (entropy site is structurally clean; the `+1e-8`
  clamp may produce a marginal-class drift inside the threshold band).

- **Test #4 (sg-advantage leak)**: Passes (`sg(advantage)` is correctly at L591).

- **Test #5 (sg-action leak)**: Passes (`sg(actions)` is correctly at agent.py:L1535).

### 6.6 The A1/A2-surfacing test (the test that v2-CP9 MUST add)

The 5-site manifest does not catch A1/A2 because each site fixes the action and uses
`Actor` in isolation. A1/A2 are **orchestrator-level** bugs. To surface them, add a
**6th test**:

```python
def test_actor_loss_uses_imagined_actions_not_resampled():
    """v2-CP3 #6 (NEW) — Surfaces A1+A2 from the v2-CP3 review.

    The orchestrator's actor_loss_fn (train.py:L840-L864) must compute log_prob on
    the ACTION FROM THE ROLLOUT, not on a freshly-sampled action.

    Diagnostic: monkey-patch Actor.__call__ to record the action that was used
    for log_prob in TWO calls — first the imagination rollout, second the
    actor_loss_fn re-run — and assert they are EQUAL.

    Expected: FAIL on the current code (proves A1).
    """
    from src.algorithms.dreamer_srl.train import make_train_step
    # Capture two action tensors via monkeypatch
    captured = {"imagine_actions": None, "actor_loss_actions": None}
    original_call = Actor.__call__
    call_count = [0]
    def spy_call(self, latent, key):
        actions, lp, ent = original_call(self, latent, key)
        if call_count[0] < H + 1:
            captured["imagine_actions"] = actions  # accumulate? use a list
        else:
            captured["actor_loss_actions"] = actions
        call_count[0] += 1
        return actions, lp, ent
    # Run one train step with fixed seeds and inputs ...
    # Then assert captured["imagine_actions"] == captured["actor_loss_actions"]
    # On current code this assertion FAILS (different Gumbel keys → different argmax).
```

This test is **not in the v2-CP2 methodology manifest** because that manifest
predates v2-CP3's surfacing of the A1/A2 bug. The senior-developer should add this as
an extension to the manifest before the v2-CP9 fix.

---

## 7. Findings table

| ID | Severity | Site (dreamer-srl) | Site (sheeprl) | Issue | Suggested fix |
|---|---|---|---|---|---|
| **A1** | 🔴 **blocker** | `train.py:L845` | `dreamer_v3.py:L283-L291` | **REINFORCE log_prob is computed on a freshly-sampled action, not on the imagined-rollout action.** `actor_loss_fn` calls `actor_module(sg_latents[h], actor_keys[h])` which draws a NEW Gumbel sample for every train step. Sheeprl evaluates `p.log_prob(imgnd_act.detach())` on the rollout-time action. Consequence: the REINFORCE gradient `∂(log_prob(a_new) · sg(adv_old)) / ∂θ` is decorrelated from the action that produced the advantage. The estimator is no longer a policy gradient. **This is independently sufficient to drive the policy to the random floor.** | (a) Extract `imagined_actions` from `imag_outputs` (currently discarded). (b) Add an `Actor.forward_logits(latent)` helper (or extend `__call__` to accept an action override) so the orchestrator can compute `log_prob = sum(sg(imagined_action) * log_softmax(current_logits))` without re-sampling. (c) Remove the Gumbel-sample branch from the actor-loss path entirely. |
| **A2** | 🔴 **blocker** | `train.py:L837` | `dreamer_v3.py:L273` (uses module-implicit RNG) | **`actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)` uses a HARD-CODED constant PRNG key.** Under JIT this becomes baked-in; every train step uses the same Gumbel noise → zero stochastic diversity in the actor-loss resample. Compounds A1. | Subsumed by the A1 fix (no sampling inside the actor-loss path). Once log_prob is computed deterministically on the imagined action via `log_softmax`, no PRNG is needed at this site. |
| **A3** | 🟡 concern | `train.py:L798` | `dreamer_v3.py:L244` | **`predicted_values` is computed from `target_critic`, not the live `critic`.** Sheeprl uses the live critic to predict values during imagination; the target critic is reserved for the two-term critic loss (sheeprl L308-L310). Using the target critic for actor's advantage means the baseline lags behind the critic by ~0-50 polyak steps, which biases the advantage signal (advantage = λ_return − baseline → if baseline lags, advantage carries stale information). This is structurally distinct from H1 (it does not break the gradient estimator) but compounds the signal-to-noise problem. Out of strict scope for v2-CP3 — surfaces at the v2-CP5 imagined-returns surface — but flagged here because the orchestrator block is shared with the actor block. | Replace `jax.vmap(target_critic)(imag_flat)` at L798 with `jax.vmap(critic)(imag_flat)`. Keep `target_critic` only for the critic-loss two-term computation at L875-L878. Cross-flag for v2-CP4 and v2-CP5. |
| **A4** | 🟡 concern | `agent.py:L1539-L1540` | `dreamer_v3.py:L294` | **Entropy uses `-Σ probs · log(probs + 1e-8)` rather than `OneHotCategorical.entropy() = -Σ probs · log_softmax(logits)`.** Algebraically equal in exact arithmetic; numerically diverges by 1-2 ULP at saturated logits. At `unimix=0.01, num_classes=5` the `+1e-8` clamp is inactive in practice (smallest post-unimix prob is `0.002 >> 1e-8`), so this is substrate-class drift only. Log as D-015 candidate if test #3 (entropy-only grad) shows marginal diff in the threshold band. | (Optional) Replace with `entropy = -jnp.sum(probs * log_softmax_logits, axis=-1)` where `log_softmax_logits` is already computed at L1533. This removes the `+1e-8` clamp and matches sheeprl's `OneHotCategorical.entropy()` form numerically. |
| **A5** | 🟢 nit | `train.py:L526-L532` | n/a | The docstring's "max(1, sigma_returns)" hint in the brief is a sloppy paraphrase of the actual sheeprl quantity `max(1/Moments._max, high_EMA - low_EMA)`. The docstring at `train.py:L526-L532` correctly describes the per-term normalization. No fix needed; flag here to anchor the methodology spec's terminology. | (Documentation only — update the brief's wording, not the code.) |
| **A6** | 🟢 nit | `train.py:L838` | (n/a) | `sg_latents = jax.lax.stop_gradient(imagined_latents)` is correct but redundant: the inputs at `train.py:L856-L859` (`lambda_values`, `predicted_values`, `moments_offset`, `moments_invscale`) are also separately sg'd. This is *defensive overkill* — both are correct, but the double-sg costs a few unused XLA tracing nodes. | (Cosmetic — leave as-is; the defensive sg is clearer than the minimal form.) |

---

## 8. Conventions audit checklist

| Convention | Status | Notes |
|---|---|---|
| **Forward parity** of `compute_actor_objective` against sheeprl L274-L297 | ✅ | All 8 line-by-line correspondences verified. |
| **`sg(advantage)`** at REINFORCE multiplier | ✅ | `train.py:L591`. Algorithmic equivalent of sheeprl L291 `advantage.detach()`. |
| **`sg(action)`** before `log_prob` | ✅ | `agent.py:L1535`. Algorithmic equivalent of sheeprl L286 `imgnd_act.detach()`. |
| **`sg(discount)`** | ✅ | `train.py:L216-L217` inside `compute_discount`. Algorithmic equivalent of sheeprl L259 `torch.no_grad()`. |
| **`sg(moments_offset)` and `sg(moments_invscale)`** | ✅ | Explicit at `train.py:L858-L859`. Algorithmic equivalent of sheeprl L63 `.detach()`. |
| **`sg(lambda_values)` and `sg(predicted_values)`** at actor input | ✅ | Explicit at `train.py:L856-L857`. Plus `sg_latents` at L838 (defensive). |
| **Entropy site sign** | ✅ | Added (not subtracted) inside `(objective + entropy_term)` at L602; negated by outer `-jnp.mean(...)`. Equivalent to sheeprl L297. |
| **`unimix`** in probability space, then `log(probs)` conversion | ✅ | `agent.py:L1477-L1482`. Matches sheeprl `_uniform_mix` at `agent.py:L839-L845`. |
| **Action used in REINFORCE is the imagined-rollout action** | ❌ | **A1** — orchestrator uses a freshly-sampled action via re-running the actor. |
| **PRNG key for actor sampling is properly threaded from `train_step`'s key** | ❌ | **A2** — hard-coded `jax.random.PRNGKey(0)` at L837. |
| **`predicted_values` uses the live critic** (matching sheeprl L244) | ❌ | **A3** — uses `target_critic` at L798. |
| **Entropy is `OneHotCategorical.entropy()`-equivalent** | ⚠ | **A4** — uses `log(probs + 1e-8)` form; equivalent in exact arithmetic, ULP-class drift at saturated logits. |
| **`get_mandatory` for required configs** | ✅ | (Not directly applicable to `compute_actor_objective` — its `ent_coef` etc. are passed in by the orchestrator.) |
| **Pure-functional pytree semantics** | ✅ | `compute_actor_objective` is pure; takes inputs, returns outputs. No in-place mutation. |
| **JIT recompilation safety** | ⚠ | `H_plus_1` is a static Python int (per the comment at L844) and is folded into the trace — so the Python for-loop at L844 unrolls into a static graph of H+1 actor calls. JIT-safe but inefficient at large H. Not a finding. |
| **vmap conventions** | n/a | `compute_actor_objective` operates on `[H, BT, 1]` directly; no vmap surface inside the function. |
| **PRNG key threading** | ❌ | **A2** — the actor-loss path does not thread the `train_step`'s key; it uses a constant. |

---

## 9. H1 surface conclusion

**H1 is confirmed at HIGH confidence.** The bug is NOT inside `compute_actor_objective`
proper — that function is forward-parity-clean and has all four canonical sg sites in
place. The bug is in the **orchestrator's call-site construction of the log_probs
input** (`train.py:L840-L864`):

1. **A1 (the primary H1 root cause)**: `actor_loss_fn` re-runs `Actor.__call__` to
   sample fresh actions, then computes `log_prob` on those fresh actions, and
   multiplies by `sg(advantage)` where `advantage` was computed from the
   *original-rollout* actions. The REINFORCE estimator is broken — `log_prob(a_new)`
   and `advantage(a_old)` are independent random variables, so
   `E[log_prob(a_new) · advantage(a_old)] = E[log_prob(a_new)] · E[advantage(a_old)]`,
   which is not the policy gradient and carries no useful learning signal.

2. **A2 (the compounding bug)**: The PRNG key used to draw `a_new` is a constant
   `jax.random.PRNGKey(0)`, which under JIT bakes the same Gumbel noise into every
   train step. So `a_new` is **deterministic across training steps**, eliminating
   even the partial signal that random resampling might preserve.

**Mapping to the 5-site manifest from `GRAD_PARITY_METHODOLOGY.md` §5.1:**

| Manifest site | Verdict |
|---|---|
| **#1 REINFORCE last-layer** | Passes on current code (the manifest test fixes the action, bypassing A1/A2). The bug is invisible to this test as specified. |
| **#2 REINFORCE trunk** | Passes on current code (same as #1). |
| **#3 Entropy-only** | Passes (entropy site is structurally clean, A4 is sub-threshold). |
| **#4 sg(advantage) leak** | Passes (`sg(advantage)` is correctly placed at `train.py:L591`). |
| **#5 sg(action) leak** | Passes (`sg(action)` is correctly placed at `agent.py:L1535`). |

**The H1 bug does NOT live at any of the 5 manifest sites.** It lives at the
**call site that constructs the `log_probs` input to `compute_actor_objective`**.
The senior-developer must extend the v2-CP3 manifest with a **6th test** (the
"imagined-action vs resampled-action consistency" test, see §6.6 above) to make A1/A2
visible to the test suite. Without that 6th test, all five manifest tests would
PASS but the training would still fail.

The "sg(advantage) leak" descriptor in the user's prompt does not match the actual
finding here — the leak diagnostic at `train.py:L591` is clean. The actual H1 root
cause is **upstream of `compute_actor_objective`** in the orchestrator, manifesting
as a **broken estimator semantics** (wrong action paired with advantage), not as a
missing sg.

H2 (driver-side, v2-CP7) and H1 (actor-side, v2-CP3) are now **both confirmed at HIGH
confidence**, and they are **independent sufficient causes** of the parity failure. Both
must be fixed at v2-CP9 before re-launching the parity gate at v2-CP10.

---

## 10. Conclusion

**❌ FAIL.** `compute_actor_objective` (`train.py:L500-L604`) is **structurally correct** in
isolation — forward parity, sg sites, sign conventions, advantage normalization formula all
match sheeprl. But the **call site** in `make_train_step` (`train.py:L837-L867`) commits
two H1-class structural bugs (**A1** — REINFORCE log_prob on freshly-sampled action
rather than rollout action; **A2** — hard-coded `jax.random.PRNGKey(0)` for that sample),
plus one cross-cutting concern (**A3** — `target_critic` used where sheeprl uses live
`critic`) and one substrate-class deviation (**A4** — entropy `+1e-8` clamp). A1 + A2
together break the REINFORCE estimator and are independently sufficient to drive the
policy to the random floor.

The v2-CP3 manifest's 5 grad-parity tests as currently specified would all PASS on the
broken code, because each test fixes the action externally and thus bypasses A1/A2. A
**6th test** (the "imagined-action vs resampled-action consistency" test in §6.6) must
be added to the manifest to make A1/A2 visible to the test suite.

The v2-CP9 fix bundle must:

1. Extract `imagined_actions` from `imag_outputs` in the orchestrator.
2. Add an `Actor.forward_logits(latent)` helper that returns logits without sampling.
3. Replace the orchestrator's `actor_loss_fn` re-sample with
   `log_prob = sum(sg(imagined_actions[h]) * log_softmax(actor.forward_logits(sg_latents[h])))`.
4. Remove the `jax.random.PRNGKey(0)` constant (now unused).
5. Switch `predicted_values` source from `target_critic` to live `critic` at `train.py:L798`
   (A3 — cross-flag for v2-CP4/CP5).
6. (Optional, low priority) Switch entropy to `-Σ probs · log_softmax` form to remove the
   `+1e-8` clamp (A4 — substrate-class, log as D-015 if sub-threshold).

Reviewed by: code-reviewer
