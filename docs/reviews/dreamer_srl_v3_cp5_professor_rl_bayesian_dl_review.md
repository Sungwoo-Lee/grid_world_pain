---
title: "dreamer-srl v3 CP5 — professor-rl-bayesian-dl audit (historical-scar checkpoint)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/loss.py
---

# dreamer-srl v3 CP5 — professor-rl-bayesian-dl audit

## Plain-language verdict

This is the third and last technical gate on CP5 — the port of sheeprl's
`TwoHotEncodingDistribution` to the JAX side as `TwoHotEncoding`. CP5 is the
**historical-scar checkpoint**: in the v1 cascade the 255-bin grid was stored
in real reward space (with `symexp` mistakenly applied at storage); losses
dropped to the wrong basin and three static reviewers reading a 1000-line plan
never caught it. Code-reviewer and math-reviewer have already PASSED — bins
live in symlog space, the formulas match sheeprl term-for-term, and the
historical mutation is caught by `test_bins_not_symexp_at_storage` at a 14-OOM
margin. **My job is the algorithm-integration level**: does this distribution
behave correctly as a probability distribution that downstream CP6 (critic
regression — the negative log-likelihood of the bootstrapped `lambda_target`
under the critic's predicted bin distribution, plus the EMA-target slow self-
regression term) and CP7 (actor — uses the critic head's expected value as a
real-space baseline for advantage normalisation) can consume without surprise?

**It does.** The class is a proper discrete probability distribution on 255
symlog-space bins; `log_prob` accepts a real-space target and `symlog`-encodes
internally (the right side of the historical-scar contract); `mean`/`mode`
return real-space expected values via `symexp(E[bin])`; the shape contract
out of `log_prob` (event dims summed via `self.dims`; trailing keepdim removed
by `squeeze(-2)`) matches sheeprl's downstream call sites at
`dreamer_v3.py:L314-L315`. D-006 is a 1-ULP linspace-implementation drift in
the forward pass; gradients flow through `logits`, not through `log_prob`'s
absolute value, so the 1.8e-5 forward-pass drift is invisible to the
optimisation signal. The `sample()` method is correctly NOT implemented
(sheeprl's `TwoHotEncodingDistribution` has none; sheeprl's CP6/CP7 never
calls it). **Verdict: PASS.** Forward D-006 to PI with concurrence.

## Distribution-properties audit (the 8 numbered points)

### 1. `TwoHotEncoding` is a proper discrete probability distribution

| Property | Status | Reasoning |
|---|---|---|
| `softmax(logits)` on 255 bins sums to 1 | ✅ trivially | `jax.nn.softmax(logits, axis=-1)` at `loss.py:105` is the canonical normalised exponential; sum over `axis=-1` is exactly 1 by construction, modulo float32 rounding (`< 1e-7` relative). |
| `log_prob(value)` well-defined for any real-valued `value` | ✅ | The encoding sequence is: real `value` → `symlog(value)` in symlog space → `below = (bins <= x).sum(-1) - 1` → clamp `[0, N-1]` → two-hot weights → cross-entropy. For `value` *inside* `[symexp(-20), symexp(+20)]` ≈ `[-4.85e8, +4.85e8]`, `below` and `above` are distinct in-range bins; weights are convex. For `value` *outside*, the clamp forces `below == above`, the `equal` branch sets both distances to 1, and the one-hot scatter collapses to a single bin at the boundary with total mass 1. **No path produces `-inf` or NaN.** Math-reviewer derivation appendix (lines 418-461 of math-review) confirms the boundary case renormalises correctly. |
| `exp(log_prob(value))` sums to 1 over the 255-bin support | ✅ (in the limit; pseudo-discrete sense) | `TwoHotEncoding` is *not* a categorical over 255 bins with `log_prob(value)` returning `log p_value`. It is a **discrete-distribution-with-continuous-target** encoding: `log_prob(value)` returns `E_{p_target(value)}[log p_pred]` where `p_target(value)` is the two-hot lattice projection of `value`. So the right normalisation property is: for any fixed continuous `value`, `sum_i p_target_i(value) = 1` (where the sum is over the 255 bins). I verified analytically: `weight_below + weight_above = (dist_to_above + dist_to_below) / total = total/total = 1`; all other bins get 0 weight via the `one_hot` scatter. **The two-hot target is a proper distribution on the bin lattice.** ✅ |
| `mean`/`mode` return finite reals | ✅ | `symexp(s)` for `s ∈ [-20, +20]` returns values in `[-4.85e8, +4.85e8]` — finite float32 (`exp(20)-1 ≈ 4.85e8`, well below float32 overflow at `~3.4e38`). |

**Bayesian-DL interpretation.** This is a fixed-grid discretisation of a continuous reward distribution — the same idea as C51 (Bellemare et al. 2017), with `symlog` applied to the support so atoms are dense near 0 and exponentially sparse at the tails. The linear-interpolation cross-weight is the standard "categorical projection" operator (Bellemare §3.4). Grid locations are fixed (unlike QR-DQN / IQN); only the logits over bins are learned. **Stationary, well-behaved.**

### 2. `log_prob` consumes a real-space target

The historical-scar contract holds:

```
loss.py:192   x = symlog(x)   # encode target to symlog space before bin lookup
```

This is the **right side** of the contract — the symmetric position relative
to `bins = linspace(-20, +20, 255)` storage in symlog space. If `bins` lived
in real space, `symlog(target)` would be wrong; if `bins` lived in symlog
space and `target` were NOT `symlog`'d, the bin lookup would be wrong. Both
sides agree on **symlog-space lookup**. ✅

**CP6 consumption sanity check** (against vendored sheeprl
`dreamer_v3.py:L314-L315`):

```python
# sheeprl side
value_loss = -qv.log_prob(lambda_values.detach())
value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
```

- `lambda_values` comes from `compute_lambda_values(predicted_rewards[1:], predicted_values[1:], ...)` at L251-L256. `predicted_rewards` and `predicted_values` are both `.mean` of `TwoHotEncodingDistribution` — i.e., real reward space. So `lambda_values` is real-space. ✅ (`log_prob` will `symlog` it internally.)
- `predicted_target_values` is the EMA target critic's `.mean` — also real-space. ✅

The JAX `log_prob(x)` matches both call patterns: a real-space `x` is
`symlog`'d internally, bracketed in symlog space, and consumed via cross-
entropy. Both CP6 use-cases are signature-compatible.

### 3. `mean` returns a real-space expected value

`loss.py:134`:

```python
return symexp(jnp.sum(self.probs * self.bins, axis=self.dims, keepdims=True))
```

This is $\operatorname{symexp}(\mathbb{E}_{p}[b])$, **not**
$\mathbb{E}_{p}[\operatorname{symexp}(b)]$. The two are unequal by Jensen's
inequality for broad distributions; sheeprl deliberately uses the first form
(L247 `transbwd(sum)`). The JAX port matches sheeprl exactly.

**CP7 consumption sanity check** (sheeprl `dreamer_v3.py:L244-L275`):

```python
predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean
# ... compute lambda_values ...
baseline = predicted_values[:-1]
offset, invscale = moments(lambda_values, fabric)         # CP1's Moments
normed_lambda_values = (lambda_values - offset) / invscale
normed_baseline = (baseline - offset) / invscale
advantage = normed_lambda_values - normed_baseline
```

Sheeprl uses `predicted_values` directly in advantage computation — no further
`symexp` at the consumer. The Moments scaling at L276-L279 (CP1) normalises
the real-space lambda values and baseline against the EMA percentile bounds.
**Therefore CP7's actor expects `mean` in real reward space, exactly what the
JAX implementation returns.** ✅

### 4. The straight-through trick for sampling — NOT implemented, correctly

Sheeprl's `TwoHotEncodingDistribution` (`distribution.py:L224-L276`) has **no
`sample()` method**. I verified by direct file scan:

```
grep -n "def sample" vendor/sheeprl/sheeprl/utils/distribution.py
# Output: line 362, inside OneHotCategoricalValidateArgs — a different class.
```

The `OneHotCategoricalValidateArgs.sample()` (L362) is for the world model's
**posterior/prior categorical-stochastic state** (the 32×32 `z` latent) — a
different distribution entirely. `TwoHotEncodingDistribution` is consumed
only via `.mean` (CP7 advantage baseline; CP6 EMA target value) and
`.log_prob` (CP6 critic regression; CP6 EMA self-regression). No call site
samples from it.

**Why no sampling is needed.** The two-hot head is a regression head with a soft-histogram parameterisation. Critic regression needs only `log_prob`; actor baseline needs only `mean`. The class is essentially a heteroscedastic regression head with a fixed expressive prior over output space — not a generative source. **Deliberate gap, not missing feature.** Future-Claude should not hunt for a `TwoHotEncoding.sample` method; sheeprl has none and the cascade needs none. ✅

### 5. Numerical stability of `log_prob`

`loss.py:233`:

```python
log_pred = self.logits - jax.scipy.special.logsumexp(self.logits, axis=-1, keepdims=True)
```

This is the **stable log-softmax** form: `log_softmax(z)_i = z_i - logsumexp(z)`, where `logsumexp` is computed in the max-subtract trick by `jax.scipy.special.logsumexp` (the JAX function explicitly does the `max`-shifted accumulation; verified in `jax.scipy.special` source). The naïve form `log(softmax(z))` would compute `exp(z)` then `log`, which underflows for `z_i << max(z)` (returning `log(0) = -inf` after softmax saturates to a one-hot).

Sheeprl uses the same stable form at L275:

```
log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
```

Term-for-term match: `torch.logsumexp` ↔ `jax.scipy.special.logsumexp`,
`dim=-1, keepdims=True` ↔ `axis=-1, keepdims=True`. ✅

**Why this matters late-training.** As the critic becomes confident (one bin's logit $\gg$ others), naïve `log(softmax)` returns `-inf` on tail bins. The stable form returns finite values throughout. The training signal stays finite. ✅

### 6. Integration with CP6's critic regression

Sheeprl's CP6 critic loss at L313-L316:

```python
qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
predicted_target_values = TwoHotEncodingDistribution(
    target_critic(imagined_trajectories.detach()[:-1]), dims=1
).mean

value_loss = -qv.log_prob(lambda_values.detach())
value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
```

The JAX `log_prob` must support **both** call patterns:

| Call | Target source | Real-space? | Shape |
|---|---|---|---|
| `qv.log_prob(lambda_values.detach())` | bootstrapped λ-return from `compute_lambda_values` | YES (linear combinations of `.mean` values stay in real space) | `[T-1, B, 1]` |
| `qv.log_prob(predicted_target_values.detach())` | EMA-target critic `.mean` | YES (`symexp` was applied in `.mean`) | `[T-1, B, 1]` |

Both targets are real-space; both have shape `[T-1, B, 1]` (the trailing `1`
is the event dim that CP1's `compute_lambda_values` maintains via
`keepdims=True`); both go through the JAX `log_prob`, which `symlog`-encodes
to symlog space, does the bracket+weights, and returns shape `[T-1, B]` after
summing over `self.dims = (-1,)`. **Both patterns work.** ✅

**Gradient flow.** Sheeprl `.detach()` strips gradients from both targets; JAX CP6 will use `jax.lax.stop_gradient`. `log_prob` itself is purely forward — JAX backprop through `logsumexp` and `one_hot * weight` is standard. ✅

### 7. §S-rule cross-reference

See §S-rule table below. Summary:

- **§S8 (KL free-nats)** — NOT touched by CP5. Free-nats lives in `reconstruction_loss` (sheeprl `loss.py:L68-L75` — `torch.maximum(kl_loss, free_nats)`), separate from `TwoHotEncoding`. The reward head (`pr = TwoHotEncodingDistribution(...)` at sheeprl `dreamer_v3.py:L164`) IS consumed inside `reconstruction_loss` via `-pr.log_prob(rewards)` at sheeprl `loss.py:L62` — that's CP6's reward regression branch. The free-nats clamp is only applied to the **KL** terms (`dyn_loss`, `repr_loss`), never to `reward_loss`. ✅ CP5 substrate is correct; free-nats is a CP6/CP6.5 concern.

- **§S9 (`Independent(..., 1)` wrapping)** — CP5 does NOT need `Independent(TwoHotEncoding, 1)` wrapping. The `dims=1` constructor argument **internally handles the event-dim reduction**: `self.dims = tuple([-x for x in range(1, dims+1)])` = `(-1,)`, and `log_prob` reduces over `self.dims` at L236. This is sheeprl's design — `TwoHotEncodingDistribution(critic(...), dims=1)` is used directly without `Independent` at `dreamer_v3.py:L244, L245, L307, L308`. **Compare** to the continue-head pattern at sheeprl `dreamer_v3.py:L167`: `pc = Independent(BernoulliSafeMode(logits=...), 1)` — Bernoulli needs `Independent` wrapping because the underlying torch `Bernoulli` does not have a `dims` arg; `TwoHotEncodingDistribution` does. ✅

- **§S10 (continue target = `1 - terminated`)** — Not touched by CP5. Continue head uses `BernoulliSafeMode`, a separate distribution. CP5 only owns `TwoHotEncoding`. ✅

### 8. `Independent(TwoHotEncoding, 1)` shape contract

The JAX `log_prob` shape contract (from `loss.py:189-236`):

| Step | Shape | Notes |
|---|---|---|
| Input `x` (real reward target) | `[T, B, 1]` | event dim = 1; sheeprl convention for reward/value heads |
| After `x = symlog(x)` | `[T, B, 1]` | element-wise |
| `below = (bins <= x).sum(-1, keepdims=True) - 1` | `[T, B, 1]` | bins broadcasts as `[255]` against `x[..., :, None]` (implicit via `<=` broadcast); sum reduces the broadcast bin axis; keepdims preserves the trailing `1` |
| `target` after one-hot + scatter | `[T, B, 1, 255]` | `one_hot(below, 255)` adds a 255-axis at the end |
| `target = squeeze(target, axis=-2)` | `[T, B, 255]` | removes the kept `1` event dim from `below`/`above` |
| `log_pred = logits - logsumexp(logits)` | `[T, B, 255]` | element-wise on the bin axis |
| `(target * log_pred).sum(axis=self.dims)` with `self.dims = (-1,)` | `[T, B]` | sums over the 255-bin axis |

**Final shape `[T, B]`** — exactly what CP6's `-qv.log_prob(...)` expects when
multiplied by `discount[:-1].squeeze(-1)` (shape `[T-1, B]` after squeezing
the trailing `1`). The shape contract is sheeprl-compatible. ✅

**Counterfactual check.** With `dims=0`, `self.dims = ()` and `log_prob` would skip reduction. Sheeprl behaves identically (verified at L236-L243 vendored). The class is **dims-parametric** and behaves correctly across both regimes.

## §S-rule cross-reference table

| §S rule | Substrate provided by CP5? | Consumed by CP5? | Left to downstream CP? |
|---|---|---|---|
| **§S1 — symlog/symexp pair** | n/a (CP1 provides) | YES — `symlog(x)` at `loss.py:192` (target encoding); `symexp(...)` at L134/L146 (`mean`/`mode` decode) | n/a |
| **§S2 — `low=-20, high=+20, n_bins=255`** | YES — `__init__` defaults at `loss.py:89-90`; `bins = linspace(-20, +20, 255)` at L111 | YES — bins used throughout `log_prob` and `mean`/`mode` | n/a |
| **§S3 — bins live in symlog space (NOT real)** | YES — **THE HISTORICAL-SCAR LINE** at `loss.py:111`; bare `jnp.linspace`, no `symexp` wrap | n/a | n/a |
| **§S4 — cross-weight assignment** | YES — `weight_below = dist_to_above / total` at L219; `weight_above = dist_to_below / total` at L220 | n/a | n/a |
| **§S5 — `softmax` over bins** | YES — `jax.nn.softmax(logits, axis=-1)` at L105 | YES — `self.probs` used in `mean`/`mode` | n/a |
| **§S6 — stable `log_pred` via `logsumexp`** | YES — `loss.py:233` (sheeprl L275 form) | n/a | n/a |
| **§S7 — `mean = symexp(E[bin])`, NOT `E[symexp(bin)]`** | YES — `loss.py:134`; `mode = mean` per sheeprl convention at L146 | n/a | n/a |
| **§S8 — KL free-nats floor** | NO — not part of CP5; free-nats lives in reconstruction_loss KL terms only | NO — `reward_loss = -pr.log_prob(rewards)` is summed into `reconstruction_loss` *without* free-nats clamping (sheeprl `loss.py:L62, L80`) | **CP6 (reconstruction_loss)** owns free-nats on the KL terms |
| **§S9 — `Independent(..., 1)` wrapping** | n/a — `TwoHotEncoding` uses internal `dims=1` mechanism instead; never wrapped in `Independent` at any sheeprl call site | n/a | **CP9 (continue head)** owns `Independent(Bernoulli, 1)` for the continue-prediction distribution |
| **§S10 — continue target = `1 - terminated`** | n/a (CP5 owns reward/value heads, not continue) | n/a | **CP9** owns the continue target |

CP5 substrates §S1–S7 cleanly; leaves §S8/S9/S10 to downstream CPs (CP6
reconstruction loss, CP9 continue head). No §S-rule is silently violated or
mis-attributed.

## Hand-off notes for downstream CPs

### CP6 (critic regression — coming next)

The critic loss (sheeprl `dreamer_v3.py:L313-L316`) has **two log_prob terms**
and a discount-weighted mean:

```
qv = TwoHotEncoding(critic_logits, dims=1)         # JAX call
value_loss = -qv.log_prob(stop_grad(lambda_target))             # NLL on bootstrap target
value_loss = value_loss - qv.log_prob(stop_grad(target_critic_mean))  # EMA self-regression
value_loss = mean(value_loss * stop_grad(discount[:-1].squeeze(-1)))
```

**Action items for the CP6 developer**:

1. Both targets are real-space (`lambda_target` is a linear combination of
   `predicted_rewards.mean` and `predicted_values.mean`, both already
   `symexp`-decoded; `target_critic_mean` is `.mean` of the EMA target
   distribution, also `symexp`-decoded). **Do NOT `symlog` them at the call
   site** — `log_prob` does it internally.
2. Use `jax.lax.stop_gradient` on both targets (sheeprl uses `.detach()`).
   Gradient flows through `qv.logits`, not through `lambda_target` or
   `target_critic_mean`.
3. **The EMA self-regression term is the slow-target regulariser**
   (Hafner 2023 §3.3) — it stabilises the critic by anchoring it to a slowly-
   moving copy of itself, in addition to the bootstrap target. This is NOT a
   double-counting bug; it is deliberate. Both terms get equal weight.
4. Shape: each `-qv.log_prob(...)` term is `[T-1, B]`; `discount[:-1].squeeze(-1)`
   is `[T-1, B]`; element-wise product then `mean()` over both axes.

### CP7 (actor — coming after CP6)

The actor loss (sheeprl `dreamer_v3.py:L262-L297`) uses `TwoHotEncoding.mean`
as the **real-space critic baseline** for advantage normalisation through
CP1's `Moments`:

```
predicted_values = TwoHotEncoding(critic_logits, dims=1).mean   # REAL space
baseline = predicted_values[:-1]                                # REAL space
offset, invscale = Moments.update(lambda_values)                # CP1
normed_lambda = (lambda_values - offset) / invscale             # NORMALISED real space
normed_baseline = (baseline - offset) / invscale                # NORMALISED real space
advantage = normed_lambda - normed_baseline                     # NORMALISED real space
```

**Action items for the CP7 developer**:

1. Per the math-reviewer's clarification (§4 audit row in
   [dreamer_srl_v3_cp5_math_review.md](dreamer_srl_v3_cp5_math_review.md)):
   `TwoHotEncoding.mean == TwoHotEncoding.mode`, both implemented as
   `symexp(sum(probs * bins))` — **NOT** `symexp(bins[argmax(probs)])`.
   Sheeprl deliberately makes `mode == mean` for the two-hot distribution
   (the argmax-bin form would discard the linear-interpolation information).
   **The actor uses `mean`; never use `mode` as if it were
   `argmax`-based.** They are bit-identical in this implementation.
2. The output of `.mean` is already real-space; do not apply `symexp` at the
   actor call site. Sheeprl never does.
3. Use `jax.lax.stop_gradient(predicted_values)` when computing the
   advantage baseline — the actor's policy gradient should not flow back
   into the critic logits via the baseline term.

## Deviation review (D-006) — algorithm lens

### What D-006 actually is

JAX's `jnp.linspace(-20, +20, 255)` produces `bins[127] = 0.0` exactly.
PyTorch's `torch.linspace(-20, +20, 255)` produces `bins[127] = 7.45e-8`
(1 float32 ULP from zero). This cascades through the two-hot encoding chain:

- bins: `max_abs_diff = 1.907e-6`
- twohot encode: `max_abs_diff = 6.080e-6`
- twohot log_prob: `max_abs_diff = 1.812e-5`

Math-reviewer re-derived the 10× total amplification analytically (within
1.5×). Code-reviewer confirmed structural plausibility from arithmetic depth.

### Algorithm-level gradient-flow analysis

CP6 critic loss is `value_loss = -qv.log_prob(stop_grad(target))`. Gradient w.r.t. `qv.logits`:

$$
\frac{\partial}{\partial \text{logits}_i} \log p(x \mid \text{logits})
= p^{(\text{target})}_i - \text{softmax}(\text{logits})_i
$$

(standard softmax-cross-entropy gradient). The 1.8e-5 forward `log_prob` diff does **not** enter this gradient — it depends only on (1) `p^{(target)}_i` (carries the 6e-6 encode-level diff) and (2) `softmax(logits)_i` (no D-006 contamination, since logits are identical across backends given identical weights). The gradient signal carries **6e-6**, which is 3–5 OOM below typical critic-logit gradient magnitudes (O(1e-3)–O(1e-1)). Furthermore, optimisation depends on *differences* of `log_prob` across steps, not absolute values — a constant offset cancels. **D-006 is invisible to the optimisation trajectory.** ✅

### Comparison to D-003 (the precedent)

D-003 (CP1 `symexp` threshold relaxed to `2e-5`) was PI-approved 2026-05-13
on the same JAX-platform-drift basis: forward-pass float32 ULP, no semantic
divergence, relative error sub-ULP. D-006 is the same class:

| | D-003 | D-006 |
|---|---|---|
| Forward-pass diff | 1.526e-5 abs | 1.812e-5 abs |
| Relative diff | 2.1e-7 (< 0.25 ULP rel) | 2.4e-6 (< 0.25 ULP rel) |
| Mathematical formula | identical to sheeprl | identical to sheeprl |
| Gradient flow | downstream uses `symexp` output | downstream uses `log_prob`'s detached target |
| Threshold | `2e-5` (1.3× observed worst case) | `3e-5` (1.5× observed worst case) |

Same approval basis applies. **Recommend ✅ APPROVE at PI gate.**

### Risk register for D-006

| Risk | Mitigation | Status |
|---|---|---|
| Drift compounds across many training steps | Drift is forward-pass-only; gradients see ≤6e-6; optimization trajectory is unaffected | ✅ controlled |
| Symptom misclassified as ULP later masquerades as bug | Threshold `3e-5` is 1.5× current worst case; a true semantic bug (sign flip, off-by-one in clamp) would produce diff $> 0.1$ — 4 OOM above threshold | ✅ caught at margin |
| Numerical instability in `log_prob` for confident critic | Stable `logsumexp` form prevents underflow in late-training regime (see point 5 above) | ✅ structural |

## Verdict

✅ **PASS** — no blockers, no algorithm-level concerns.

### Summary

- `TwoHotEncoding` is a proper discrete probability distribution on 255 symlog-space bins; all 8 numbered properties verified.
- The historical-scar contract (bins in symlog, target `symlog`'d in `log_prob`) is maintained on both sides.
- CP6 (critic NLL + EMA self-regression) and CP7 (actor real-space baseline through CP1 Moments) consumption patterns are signature-compatible.
- `sample()` correctly NOT implemented (sheeprl has none; no consumer needs it).
- `Independent(..., 1)` wrapping correctly NOT applied (the `dims=1` constructor argument handles event reduction internally).
- §S-rule attribution is clean: CP5 substrates §S1–S7; §S8/S9/S10 belong to downstream CPs.
- D-006 is 1-ULP linspace-implementation drift in the forward pass; gradients carry ≤6e-6, dwarfed by optimization-scale gradients; precedent matches D-003. **Recommend ✅ APPROVE.**

### Findings table

| Severity | File:line | Issue | Suggested fix |
|---|---|---|---|
| none | n/a | None | n/a |

### Hand-off

- **PI (D-006 ratification)**: D-006 forwarded with professor-rl-bayesian-dl
  concurrence to ✅ APPROVE. Same class as D-003 (forward-pass JAX-platform
  ULP drift; sub-quarter-ULP relative error; mathematical formula identical
  to sheeprl; gradient flow not contaminated). After PI approval, CP5 gate
  closes and CP6 (reconstruction_loss) is the next checkpoint.
- **CP6 developer**: see "Hand-off notes for downstream CPs — CP6"
  above. Two `log_prob` terms; both targets are real-space; both need
  `stop_gradient`; the EMA self-regression term is deliberate (Hafner 2023
  §3.3 slow-target regulariser).
- **CP7 developer**: see "Hand-off notes for downstream CPs — CP7"
  above. `mean == mode`; both implemented as `symexp(E[bin])` not
  `symexp(bins[argmax])`; output is real-space; CP1's `Moments` normalises.

Reviewed by: professor-rl-bayesian-dl
