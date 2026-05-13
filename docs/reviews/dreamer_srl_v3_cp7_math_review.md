---
title: "dreamer-srl v3 CP7 — math review"
topic: dreamer
status: active
reviewer: math-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/train.py (CP7 additions; commits 3c5be0c + f71eecb)
---

# CP7 Math Review — PASS

## Plain-language verdict

**Question.** Is the math in CP7 of the dreamer-srl v3 rebuild — (i) the slow-target value-network update (Polyak EMA), (ii) the "grafting the real episode-end signal onto the imagined rollout" step (§S5 true-continue splice), and (iii) the policy-gradient objective with a running-scale advantage normalizer (§S7 REINFORCE) — written down correctly **and** implemented term-for-term in JAX as in the reference PyTorch implementation we are porting from (sheeprl)?

**Headline.** **PASS.** All four equations under review are algebraically correct, dimensionally consistent, and faithful to the sheeprl reference. The Polyak blend matches sheeprl's commuted form bit-identically in float32. The §S5 splice preserves the time axis correctly (the `[0:1]` slice keeps a rank-3 chunk so `jnp.concatenate(axis=0)` is well-defined). The §S7 per-term normalization is algebraically redundant — the offset cancels — but is preserved verbatim for bit-identity with sheeprl, which is the right call. The REINFORCE objective applies `stop_gradient` on both action (via the caller's `log_prob` site) and advantage (inside `compute_actor_objective`) and reduces by `mean` over $[T{-}1, B]$ with the discount weight. No sign errors, no missing factors, no shape mismatches.

**Recommendation.** Forward to PI for D-011 ratification. D-011 is the same structural class as D-001 (pure-functional return replacing in-place mutation, forced by JAX's lack of array mutation inside JIT) — same arithmetic, different mechanism, `max_abs_diff = 0.000e+00` measured. No code changes requested.

**Three sentences of plain-language context.** *Polyak EMA* keeps a slow shadow copy of the critic so the value bootstrap targets don't chase a moving estimator; the shadow drifts toward the live network by 2 % per update. The *§S5 true-continue splice* replaces the world model's guess for whether the first imagined step continues the episode with the actual flag from the replay buffer, so the discount at step 0 is grounded in reality. The *§S7 REINFORCE objective* turns the imagined trajectory's lambda-return minus a baseline into a policy gradient, with both terms divided by a running scale so the gradient size doesn't blow up as the value function grows.

## Scope

| Aspect | Path |
|---|---|
| Implementation | `src/algorithms/dreamer_srl/train.py` lines 329–602 (CP7 additions) |
| Sheeprl reference | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` L246–L297, L673–L697 |
| Code review (gate 1) | `docs/reviews/dreamer_srl_v3_cp7_code_review.md` (PASS) |
| Deviation log | `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` row D-011 |
| Commits under review | `3c5be0c` (impl), `f71eecb` (diary) |
| Precedent for D-011 | D-001 (CP1 `moments_update` — pure-functional dict replacing in-place buffer mutation, also `max_abs_diff = 0.000e+00`-class) |

## Equations under review

### (Eq. 1) Polyak EMA target-critic update

Sheeprl `dreamer_v3.py:L680`:

$$
\text{tcp} \;\gets\; \tau \cdot \text{cp} + (1 - \tau)\cdot\text{tcp}
$$

JAX `train.py:384-387`:

$$
\theta^\text{tgt}_k \;\gets\; (1 - \tau)\,\theta^\text{tgt}_k + \tau\,\theta^\text{on}_k \qquad \forall\,k \in \text{params}
$$

where $\tau = 1.0$ on the very first call (hard copy) and $\tau = 0.02$ thereafter (sheeprl XS default, `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml:152`).

### (Eq. 2) §S5 true-continue splice

Sheeprl `dreamer_v3.py:L247-L248`:

$$
\text{true\_continue} = (1 - \text{terminated})\big|_\text{reshape\ [1, BT, 1]}, \qquad c_{0:H+1} \;\gets\; \text{concat}\!\big(\text{true\_continue},\; c_{1:H+1}\big)
$$

JAX `train.py:473-477`:

$$
c^\text{spliced}_t = \begin{cases} 1 - \text{terminated}_\text{obs} & t = 0 \\ \hat c_t & 1 \le t \le H \end{cases}
$$

(`true_continue` is reshaped to `[1, BT, 1]`; `jnp.concatenate(axis=0)` produces shape `[H+1, BT, 1]`.)

### (Eq. 3) §S7 per-term advantage normalization

Sheeprl `dreamer_v3.py:L275-L279`:

$$
\text{baseline}_t = \hat V_t,\qquad
\tilde \Lambda_t = \frac{\Lambda_t - \mu}{\sigma_\text{inv}},\qquad
\tilde B_t = \frac{B_t - \mu}{\sigma_\text{inv}},\qquad
A_t = \tilde \Lambda_t - \tilde B_t
$$

where $\mu = \text{offset}$, $\sigma_\text{inv}^{-1} = \text{invscale}^{-1}$ (the `Moments` head's percentile-EMA scalars from CP1; see `MomentsState`), $\Lambda_t$ is the lambda-return target on the imagined trajectory, and $\hat V_t$ is the critic prediction $\text{predicted\_values}[:-1]$.

JAX `train.py:573-583` is verbatim.

### (Eq. 4) REINFORCE policy loss (discrete, grid-world)

Sheeprl `dreamer_v3.py:L283-L297`:

$$
J_t = \log\pi\!\big(\text{sg}[a_t]\big)\cdot \text{sg}[A_t],
\quad
\mathcal H_t = \beta_H \cdot \mathcal H[\pi_t],
\quad
\mathcal L_\pi = -\,\mathbb E_{[:T-1]}\!\big[\,d_t \cdot (J_t + \mathcal H_t)\big]
$$

where $\text{sg}[\cdot] \equiv \texttt{jax.lax.stop\_gradient}(\cdot)$ (PyTorch `.detach()`), $d_t$ is the §S6 discount (already `stop_gradient`'d in `compute_discount`), $\beta_H$ is `ent_coef` (sheeprl XS default $3\times 10^{-4}$), and the mean reduces over the $[T{-}1, B]$ axes (the `[:-1]` slice trims the last imagined step exactly as the critic loss does in CP6).

JAX `train.py:589-600` is verbatim.

## Per-equation audit

### Eq. 1 — Polyak EMA commutation: algebraically and bit-identically equivalent at float32

The user's question: is sheeprl's $\tau\cdot\text{cp} + (1-\tau)\cdot\text{tcp}$ bit-identical to JAX's $(1-\tau)\cdot\text{tcp} + \tau\cdot\text{cp}$ at float32, given that floating-point addition is commutative element-wise but compilers can reassociate?

**Algebraic equivalence.** Trivial: addition of real numbers commutes, and the two products are formed independently of each other.

**Bit-identity at float32.** Element-wise IEEE-754 binary32 addition is commutative *as a binary operation*: $a + b$ and $b + a$ are rounded identically because the operation is symmetric in its arguments before rounding (the rounding rule depends only on the unrounded exact sum and the rounding mode, both symmetric in $(a,b)$). The two scalar multiplications $\tau\cdot\text{cp}$ and $(1-\tau)\cdot\text{tcp}$ are independent — neither reuses the other — so the only operation whose order changes is the final sum, which is commutative. **Therefore Eq. 1's two forms produce bit-identical float32 output for matching inputs.** The measured `max_abs_diff = 0.000e+00` across the 3/3 CP7 runners confirms this empirically (Polyak is the only CP-step in the v3 series with a fully exact result — see code review §Verdict).

Caveat: this argument assumes the compiler (XLA / LLVM) does not fuse the two multiplies and the add into an FMA. JAX's default float32 path on CPU/GPU emits separate multiply and add ops here (no FMA on this expression because both multiplies feed the same add — FMA would require a different graph shape). If a future XLA pass were to enable strict-FMA on this expression, the bit-identity claim would need re-verification, but the measured `0.000e+00` indicates no such fusion is happening today. Not a concern for CP7.

**Verdict on Eq. 1**: ✅ correct, commuted, bit-identical, no findings.

### Eq. 2 — §S5 splice shape correctness

The user's specific concern: does `terminated_observed.reshape(1, continues_predicted.shape[1], 1)` preserve the time axis correctly so that `jnp.concatenate([true_continue, continues_predicted[1:]], axis=0)` works?

Walking through shapes:

- `continues_predicted` has shape $[H+1, BT, 1]$.
- `terminated_observed` arrives as either $[1, BT, 1]$, $[BT, 1]$, or $[BT]$ (the reshape is shape-flexible per the code-reviewer's nit).
- `(1 - terminated_observed).reshape(1, continues_predicted.shape[1], 1)` produces shape $[1, BT, 1]$ regardless of input layout, provided `size == BT`.
- `continues_predicted[1:]` has shape $[H, BT, 1]$ (the slice drops axis-0 element 0).
- `jnp.concatenate([true_continue, continues_predicted[1:]], axis=0)` requires all non-0 axes to match (✓ both have $BT$ on axis 1 and $1$ on axis 2) and stacks along axis 0: $1 + H = H+1$. ✓

The critical detail is the `axis=0` rank-preservation: a naive `(1 - terminated_observed).reshape(BT, 1)` would produce a rank-2 array, and `jnp.concatenate` with the rank-3 `continues_predicted[1:]` would raise a shape error. The explicit `reshape(1, BT, 1)` is correct.

The code-reviewer flagged the absence of a shape assertion as a 🟢 nit — I concur it's a nit and not a math-correctness blocker. A caller passing a wrongly-sized `terminated_observed` would either crash on the reshape (if `size != BT`) or, in the pathological case of `terminated_observed.size == BT` but wrong logical layout (e.g., a $[H, BT]$ array misread as a flat $[H \cdot BT/H]$), would produce silently wrong continues. This is a CP8-time integration concern, not a CP7 math concern.

**Cross-check against sheeprl L247.** PyTorch: `(1 - data["terminated"]).flatten().reshape(1, -1, 1)`. The `.flatten()` collapses any batch shape to a flat vector, then `.reshape(1, -1, 1)` produces `[1, BT, 1]`. The JAX form skips the explicit `.flatten()` and instead computes the target $BT$ as `continues_predicted.shape[1]`, then reshapes. Semantically equivalent for the contract `terminated_observed.size == BT`.

**Verdict on Eq. 2**: ✅ correct, shape-preserving, the `[0:1]`-style slice equivalent (reshape to length-1 leading axis) is in place.

### Eq. 3 — §S7 offset cancellation

The user's algebraic point: the offset $\mu$ cancels:

$$
A_t = \frac{\Lambda_t - \mu}{\sigma_\text{inv}} - \frac{B_t - \mu}{\sigma_\text{inv}} = \frac{\Lambda_t - B_t}{\sigma_\text{inv}}
$$

The two forms are algebraically identical for real arithmetic. **In float32, they may differ by 1 ULP at most**, because the per-term form performs the subtraction $\Lambda_t - \mu$ (which may shift the exponent of $\Lambda_t$), then divides, then subtracts again — while the cancelled form performs $\Lambda_t - B_t$ first, then divides once. The per-term form has three rounding events; the cancelled form has two. The two are not bit-identical in float32 except by accident.

**Why the per-term form is the right choice here.** The reviewer's audit is bit-identity against sheeprl. Sheeprl computes the per-term form at L277–L279. Therefore, to achieve `max_abs_diff = 0.000e+00`-class results on the actor-objective diff-tool runners (a CP8 target, not CP7 — CP7's diff-tool runners are Polyak-only), the JAX port must use the per-term form, not the cancelled form. The CP7 code does this verbatim. **This is the right call for the project's stated bit-identity goal**, even though a from-scratch implementation could use the cancelled form for one less rounding event.

The user's phrasing "per-term computation is for bit-identity with sheeprl, not numerical correctness" is precisely correct: numerically, both forms are correct; only the per-term form is sheeprl-faithful.

**Verdict on Eq. 3**: ✅ correct, per-term form preserved as required for bit-identity. No findings.

### Eq. 4 — REINFORCE objective, signs, slices, and stop-gradients

I verify each piece of the REINFORCE expression against sheeprl L283–L297:

| Component | Sheeprl | JAX `train.py` | Match? |
|---|---|---|---|
| Baseline = $\hat V_{:T-1}$ | L275 `baseline = predicted_values[:-1]` | L573 `baseline = predicted_values[:-1]` | ✓ |
| Advantage form | L277–L279 per-term normalized minus | L577–L583 per-term normalized minus | ✓ |
| `stop_gradient` on advantage | L291 `advantage.detach()` | L589 `jax.lax.stop_gradient(advantage)` | ✓ |
| `stop_gradient` on action (for `log_prob`) | L286 `p.log_prob(imgnd_act.detach())` | docstring at L540–L545 defers to caller (CP8 actor forward pass) | ✓ (deferred; flagged) |
| `log_probs` × `sg(advantage)` (discrete, no continuous branch) | L283–L290 `objective = sum_log_prob(...) * advantage.detach()` | L589 `objective = log_probs * sg(advantage)` | ✓ |
| Entropy term $\beta_H \cdot \mathcal H[:T-1]$ | L294–L295 `ent_coef * entropy[...,-1][:-1]` | L592 `ent_coef * entropy[:-1]` | ✓ |
| Discount weight $d_{:T-1}$ | L297 `discount[:-1].detach() * ...` | L598 `discount[:-1]` (already `sg`'d in `compute_discount`) | ✓ |
| Mean reduction | L297 `torch.mean(...)` over all axes | L600 `jnp.mean(...)` over all axes | ✓ |
| Sign | L297 `-torch.mean(...)` | L600 `-jnp.mean(...)` | ✓ |

Two subtleties worth flagging explicitly because they are common sign / shape pitfalls:

**(a) The sign and the reduction axis.** Sheeprl reduces with `torch.mean(...)` *without* a `dim=` argument, which averages over **all axes** of the input — in this case `[T-1, B, 1]` (or `[T-1, B, 1, A]` for discrete with $A$ action dims summed at L290, but for the grid-world single-action-head case the trailing axis is 1). JAX's `jnp.mean(...)` without an `axis` argument has the same semantics. The two are equivalent. The sign $-$ in front turns the policy-gradient objective (which we want to *maximize*) into a loss (which we *minimize*). ✓

**(b) The `[:-1]` slice on the entropy.** Sheeprl L295: `entropy.unsqueeze(dim=-1)[:-1]` — the `[:-1]` is applied **after** the `unsqueeze`, but since `[:-1]` slices axis 0, the order doesn't matter. The JAX form L592 `entropy[:-1]` assumes `entropy` is already shaped `[H+1, BT, 1]` (per the docstring at L553–L555), so `entropy[:-1]` is `[H, BT, 1]`. The shape of `entropy_term` then matches `objective` (both `[H, BT, 1]`), and the discount-weighted sum is well-defined. ✓ Caller (CP8) must produce `entropy` with the trailing-1 axis already in place — this is a CP8 concern, flagged in the docstring.

**Stop-gradient analysis.** The math-reviewer's concern on REINFORCE is always: *does the gradient flow through the right path?* The REINFORCE estimator's gradient comes from $\nabla_\theta \log\pi_\theta(a) \cdot A$ — the gradient flows **only** through $\log\pi$ (the policy distribution's parameters). For the gradient to be a valid score-function estimator:

1. $A$ must be treated as a constant (else we get a wrong, sample-dependent baseline term). $A$ is constructed from $\Lambda$ and $\hat V$, which both come from critic / value-network outputs *computed on detached states*. The `sg(advantage)` at L589 enforces this — even if the upstream `predicted_values` and `lambda_values` are not detached, the `sg` on `advantage` cuts the gradient cleanly here. ✓
2. The action $a$ must be treated as a constant: $\nabla_\theta \log\pi_\theta(a)$ must not also differentiate through how $\theta$ chose $a$. PyTorch sheeprl achieves this via `imgnd_act.detach()` inside `p.log_prob(...)` at L286. The JAX code defers this to the caller (CP8's actor forward pass). The docstring at L540–L545 explicitly states "log_probs is computed with stop_gradient already applied to the sampled action". This is correct mathematically; whether CP8 actually applies it is a CP8 review item (and was flagged as a 🟢 nit by the code-reviewer).

Both `sg` boundaries are at the right places.

**Verdict on Eq. 4**: ✅ correct sign, correct slices, correct reductions, correct `stop_gradient` discipline. The deferred-to-caller `sg(action)` is documented and consistent with sheeprl's structural location for that `detach()` (at the `log_prob` site, not the policy-loss site).

## D-011 review

D-011 logs that `polyak_update` returns a new dict rather than mutating in place. The math content of D-011:

- **Sheeprl form**: `tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)` (in-place mutation of the target params tensor).
- **JAX form**: `{k: (1.0 - tau) * target_params[k] + tau * online_params[k] for k in online_params}` (functional dict return).

**Mathematically identical** (Eq. 1 above; same blend, commuted form bit-identical at float32). Mechanism-only deviation: in-place tensor mutation vs functional dict construction, forced by JAX's prohibition on array mutation inside JIT-traced code. The measured `max_abs_diff = 0.000e+00` across the 3 CP7 runners confirms there is no numerical deviation introduced by the mechanism change.

**Substrate-class match with D-001.** D-001 logs the same kind of mechanism change for `moments_update` (PyTorch in-place EMA buffer update → JAX pure-functional `MomentsState` replacement); D-001 was approved by PI on 2026-05-13 with measured `max_abs_diff = 8.2e-8`. CP7's D-011 has an even cleaner `0.000e+00` measurement because the Polyak path is pure float32 arithmetic without the percentile-quantile chain that gave D-001 its 8.2e-8 ULP drift. **D-011 is unambiguously the same class and should ratify with no math-reviewer reservations.**

## Findings table

| Severity | Eq. | Location | Issue | Suggested action |
|---|---|---|---|---|
| 🟢 nit | Eq. 1 | `train.py:384-387` | Bit-identity of commuted EMA blend relies on no FMA fusion on this expression — true today (measured `0.000e+00`), worth re-checking if XLA passes change. | None for CP7. CP8 + CP9 should keep the strict 1e-6 threshold on Polyak runners; any future drift would be the canary. |
| 🟢 nit | Eq. 2 | `train.py:473` | The reshape accepts any input layout with `size == BT` — no assertion. A wrong-rank caller would silently reshape. | Optional CP8 hardening: add `assert terminated_observed.size == continues_predicted.shape[1]` at function entry. Concurs with code-reviewer's nit. |
| 🟢 nit | Eq. 4 | `train.py:540-545` | `stop_gradient` on the sampled action is deferred to the CP8 actor forward pass, not enforced inside `compute_actor_objective`. This is correct per sheeprl's structural location, but introduces a CP8 review dependency. | CP8 math-review checklist item: verify the actor forward pass applies `jax.lax.stop_gradient` to the sampled action before computing `log_prob`, matching sheeprl L286 `imgnd_act.detach()`. |

No 🔴 blockers. No 🟡 ambiguities. The math-reviewer's CP7 nits are all forward-looking integration checks for CP8, not corrections to CP7.

## Derivation appendix — §S7 offset cancellation worked out

For completeness, the offset-cancellation algebra the user cited:

$$
\begin{aligned}
A_t &= \tilde\Lambda_t - \tilde B_t \\
    &= \frac{\Lambda_t - \mu}{\sigma_\text{inv}} - \frac{B_t - \mu}{\sigma_\text{inv}} \\
    &= \frac{(\Lambda_t - \mu) - (B_t - \mu)}{\sigma_\text{inv}} \\
    &= \frac{\Lambda_t - B_t}{\sigma_\text{inv}}.
\end{aligned}
$$

The offset $\mu$ cancels exactly in real arithmetic. In float32, the cancelled form has fewer rounding events (two: one subtraction, one division) than the per-term form (three: two subtractions, two divisions, then one final subtraction — five rounding events). The two forms can therefore differ by up to a few ULPs.

The CP7 code preserves the **per-term** form to match sheeprl's rounding pattern bit-identically — the correct call for the project's bit-identity audit goal. A reader who saw only the cancelled form and asked "why isn't this simpler?" would have the right algebraic intuition but the wrong project-goal context. The code's inline comment at `train.py:582` ("NOTE: offset cancels algebraically; we keep per-term form for bit-identity") makes this explicit; the math reviewer endorses this comment.

## Conclusion

CP7 math is correct line-for-line. All four equations under review (Polyak EMA, §S5 splice, §S7 advantage normalization, §S7 REINFORCE objective) match the sheeprl reference algebraically, dimensionally, and — for Eq. 1 — bit-identically in float32. The §S5 splice shape contract is preserved correctly (`[1, BT, 1]` `concat` with `[H, BT, 1]` on `axis=0` produces `[H+1, BT, 1]`). The §S7 per-term form is the right choice for bit-identity even though offset cancels algebraically. The REINFORCE objective applies `stop_gradient` correctly on both the action (deferred to the caller per sheeprl's structural pattern) and the advantage (inline, matching sheeprl L291). D-011 is the same structural class as D-001 with measured `max_abs_diff = 0.000e+00` and should ratify.

**Verdict: ✅ PASS.** Forward to PI for D-011 ratification. No code changes requested.

Reviewed by: math-reviewer
