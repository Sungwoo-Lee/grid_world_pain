---
title: "dreamer-srl v3 CP7 — professor-rl-bayesian-dl audit (Polyak EMA + §S5 splice + §S7 REINFORCE)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/train.py (CP7 additions L329–L602; commits 3c5be0c + f71eecb)
---

# dreamer-srl v3 CP7 — professor-rl-bayesian-dl audit

## Plain-language verdict

This is the **third and final technical gate on CP7** — the port of three
Dreamer-V3 algorithm pieces to a new `src/algorithms/dreamer_srl/train.py`:
(i) the slow-moving target value network that gets nudged a little toward the
live value network each gradient step (the Polyak EMA update at sheeprl
`dreamer_v3.py:L678–L680`); (ii) the step at the start of the imagined rollout
where the world-model's guess for "did the episode just end?" is *replaced* by
the actual termination flag from the replay buffer (the §S5 true-continue
splice at sheeprl L246–L248); and (iii) the policy-gradient objective that
turns imagined lambda-returns and policy log-probabilities into an actor loss,
with both the lambda-return and the value-baseline divided by the same running
scale before subtracting (the §S7 advantage normalisation + REINFORCE
construction at sheeprl L274–L297). Code-reviewer and math-reviewer have
already PASSED — `max_abs_diff = 0.000e+00` on all three Polyak diff-tool
runners, all four CP7 equations match sheeprl term-for-term, and the one
deviation (D-011) is logged as `☐ pending`, not autonomously flipped.

**My job is the algorithm-integration level**: do these three pieces produce
the actual learning signal Hafner-2023 (DreamerV3) §3.4 prescribes, do they
couple sensibly with the CP1 Moments head, the CP6 critic loss, the §S5/§S6
discount machinery, and the still-to-come CP8 actor forward pass, and is the
D-011 mechanism-only deviation a clean substrate-class match with D-001?

**They do, and it is.** The Polyak EMA implements the standard slow-target
trick for value-function bootstrapping — the cleanest possible substrate of
the trick, because the only operation is `(1-τ)·target + τ·online` on float32
arrays, and addition of float32 scalars is commutative under IEEE-754 binary32
arithmetic. The §S5 splice anchors the imagination rollout's first-step
discount to the buffer's truthful `1 - terminated` flag at index 0, severing
a quietly catastrophic failure mode of the world model's miscalibrated
continue-head at episode boundaries. The §S7 per-term normalisation makes the
policy gradient scale-invariant across reward regimes — the offset $\mu$
cancels algebraically, but the per-term form preserves bit-identity with
sheeprl's rounding pattern, which is the right call for a bit-identity port.
The REINFORCE objective applies `stop_gradient` correctly on the advantage
(inline, at the policy-loss site) and defers `stop_gradient` on the action to
the caller (CP8's actor forward pass, where `log_prob(sg(action))` happens —
this is the same structural location as sheeprl L286).

**D-011 is a clean substrate match with D-001.** Both replace in-place
PyTorch tensor mutation with a pure-functional dict return, forced by JAX's
prohibition on array mutation inside JIT-traced code. D-001's measured drift
was `8.2e-8` (single linspace-quantile chain). D-011's measured drift is
`0.000e+00` — even cleaner — because the Polyak path is two scalar multiplies
and one add per parameter, with no quantile/bin-lookup cascade.

**Process-discipline positive note.** The D-011 verdict cell reads `☐ pending`.
The developer did not autonomously flip it to `✅ APPROVED`. This is the
**second clean Lever-E cycle in a row** after the CP6 restoration following
the CP4+CP4b incident (commit `4491c66`). The gate-respect strengthening
(2026-05-14 Process notes) is working as designed and the pattern is now
established: developer logs `☐ pending`, code-reviewer audits, math-reviewer
audits, professor audits, PI ratifies via `docs/pi/calls/*.md`, verdict cell
links the PI call doc. Credit.

**Verdict: ✅ PASS.** Forward D-011 to PI with concurrence.

## Algorithm-fidelity audit (the 8 numbered points from the review brief)

### 1. Polyak target-critic update — the slow-target trick for value bootstrapping

The CP7 implementation at `train.py:384–387`:

```python
return {
    k: (1.0 - tau) * target_params[k] + tau * online_params[k]
    for k in online_params
}
```

corresponds to sheeprl L680 `tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)`.
Mathematically, the EMA update is

$$
\theta^{\mathrm{tgt}}_k \;\gets\; (1 - \tau)\,\theta^{\mathrm{tgt}}_k \;+\; \tau\,\theta^{\mathrm{on}}_k
\qquad \forall\, k \in \mathrm{params},
$$

with $\tau = 1$ on the first call (hard copy → $\theta^{\mathrm{tgt}} = \theta^{\mathrm{on}}$)
and $\tau = 0.02$ thereafter (sheeprl XS default,
`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml:152`).

**Why the slow target matters algorithmically.** The CP6 critic loss is the
two-term DreamerV3 form (Eq. 8, sheeprl L313–L316):

$$
\mathcal{L}_{\mathrm{critic}}(\phi) \;=\;
-\,\mathbb{E}\!\left[
\log q_\phi(v_t \mid s_t)\!\left[\mathrm{sg}(R^\lambda_t)\right]
\;+\;
\log q_\phi(v_t \mid s_t)\!\left[\mathrm{sg}(v^{\mathrm{tgt}}_t)\right]
\right].
$$

The second term — cascade fix #29's slow-target self-regulariser — consumes
$v^{\mathrm{tgt}}_t = \mathbb{E}\!\left[\hat q_{\bar\phi}(v_t \mid s_t)\right]$,
where $\bar\phi$ is the **Polyak target** maintained by `polyak_update`. If
$\bar\phi$ were not slow-moving — for example, if it were a hard copy of
$\phi$ at every step ($\tau = 1$) — the second term collapses to
$\log q_\phi[\mathrm{sg}(\mathbb{E}[\hat q_\phi(v_t)])]$, a self-distillation
loss that contributes near-zero gradient near a sharp $q_\phi$. The Polyak
EMA at $\tau = 0.02$ guarantees $\bar\phi$ trails $\phi$ by **~35 gradient
steps** (half-life $\log 0.5 / \log 0.98 \approx 34.3$ steps), giving the
self-regulariser something to actually regulate against.

**Why $\tau = 0.02$ specifically.** The Hafner-2023 default across the
DreamerV3 model-size sweep (XS/S/M/L/XL — all use `tau: 0.02`) is calibrated
for a regime where the value head sees ~1k–10k gradient updates over a
typical training run. A half-life of 35 updates is short enough to track the
online critic's slow drift through the imagination-buffer's data distribution,
and long enough that high-variance early-training fluctuations average out in
$\bar\phi$. The JAX port consumes this default unchanged.

**Bit-identity of the commuted blend (Eq. 1 in math-reviewer's audit).**
Math-reviewer establishes that float32 IEEE-754 binary32 addition is
commutative element-wise — the rounded sum $\tau a + (1-\tau) b$ matches
$(1-\tau) b + \tau a$ bit-identically as long as no FMA fusion happens on the
expression (which JAX/XLA does not enable for two-multiply-one-add chains
where both multiplies feed the same add). Empirically, all three CP7 Polyak
diff-tool runners (first-call hard copy, subsequent EMA blend, fires-before-
train ordering) report `max_abs_diff = 0.000e+00`. ✅

**Call-order invariant.** Sheeprl L679–L680 places the Polyak update
**before** the `train()` call in the inner gradient-step loop. The target
critic used inside `train()` for the second-term log-prob is the *freshly
updated* target from this iteration's Polyak call. The JAX test
`test_polyak_fires_before_train_step` enforces this via a two-step trace +
grep-based call-order check. CP8 (actor + critic + world-model training loop
assembly) must preserve this order. ✅ at CP7; flagged for CP8 review.

### 2. §S5 true-continue splice — anchoring imagination to ground truth at step 0

The §S5 splice replaces `continues[0]` — the world model's predicted continue
probability for the first imagined step — with the **observed** continue from
the replay buffer:

$$
c^{\mathrm{spliced}}_t = \begin{cases}
1 - \mathrm{terminated}_{\mathrm{obs}} & t = 0, \\
\hat c_t & 1 \le t \le H.
\end{cases}
$$

The JAX implementation at `train.py:473–477` matches sheeprl L247–L248
verbatim, with `(1 - terminated_observed).reshape(1, BT, 1)` producing the
rank-3 length-1 leading axis that `jnp.concatenate([..., continues_predicted[1:]], axis=0)`
needs.

**Why the splice at index 0 is non-negotiable.** The world model's continue
head $\hat c_t = \mathbb{P}_\psi(\text{continue} \mid s_t)$ is trained on
the reconstruction loss's continue NLL (`loss.py:566` after CP6). It is
**well-trained for mid-episode states** — the abundant signal regime — but
**poorly calibrated at episode boundaries**, where the training signal is
sparse (one positive example per terminal transition) and the world model's
posterior tends to drift toward a "continue=1" prior on out-of-distribution
inputs. Without §S5, the discount at $t = 0$ becomes

$$
d_0 = \mathrm{sg}\!\left(\frac{\hat c_0 \cdot \gamma}{\gamma}\right) = \mathrm{sg}(\hat c_0),
$$

which silently mis-weights the first imagined step in proportion to the
world model's calibration error at the buffer's boundary states. The error
is **silent** because it does not show up in any obvious training-curve
signature — it just leaks gradient into the first-step actor/critic update
that should not be there (or fails to leak gradient that should be there,
depending on the calibration direction). The §S5 splice replaces $\hat c_0$
with $1 - \text{terminated}_{\text{obs}}$ — which is **bit-exact** because
the buffer stores the actual termination flag.

**Index correctness (the user's verification ask).** The splice is at
**index 0** (`continues_spliced[0] = 1 - terminated_obs`,
`continues_spliced[1:] = continues_predicted[1:]`). Not index 1; not the
last index. The JAX `jnp.concatenate([true_continue, continues_predicted[1:]], axis=0)`
puts `true_continue` (shape `[1, BT, 1]`) at the leading position of axis 0
and `continues_predicted[1:]` (shape `[H, BT, 1]`) after it, producing the
right `[H+1, BT, 1]` shape with index 0 spliced. ✅

**The splice happens BEFORE both `compute_lambda_values` and
`compute_discount`.** Lines 475–489 of `train.py`:

```
continues_spliced = jnp.concatenate(..., axis=0)        # L475
...
lambda_vals = compute_lambda_values(
    predicted_rewards[1:],
    predicted_values[1:],
    continues_spliced[1:] * gamma,   # L484 — uses spliced continues
    lmbda=lmbda,
)
discount = compute_discount(continues_spliced, gamma)    # L489 — uses spliced continues
```

Both downstream consumers see the post-splice tensor. ✅

**The §S5/§S6 boundary check from CP6.** CP6's `compute_discount` was
documented (`train.py:107–112` per the CP6 audit) as consuming
*pre-spliced* continues — i.e., CP6 did **not** silently implement §S5,
leaving the splice as CP7's responsibility. CP7 now delivers it. The
contract from CP6 is preserved: `compute_discount` does no splicing of
its own; the caller (CP7's `compute_imagined_returns`) splices. ✅

**Closest published precedent.** This is exactly the "real-continue
substitution" pattern from Hafner-2023 §3.4 (DreamerV3 paper, Eq. 9
caption: "we use the true continuation flag from the replay buffer at
the first step"). Earlier DreamerV2 (Hafner-2020) does not use this
splice — V2 used the world-model continue head at all steps, which
produced known instabilities at episode boundaries. The splice is one
of the V3-vs-V2 algorithmic differentiators. The JAX port is faithful.

### 3. §S7 advantage low-offset cancellation — scale-invariance via percentile-EMA scalars

The §S7 advantage construction at `train.py:573–583`:

```python
baseline = predicted_values[:-1]                                          # [H, BT, 1]
normed_lambda_values = (lambda_values - moments_offset) / moments_invscale  # L577
normed_baseline      = (baseline      - moments_offset) / moments_invscale  # L579
advantage = normed_lambda_values - normed_baseline                          # L583
```

matches sheeprl L274–L279 term-for-term, with $\mu = $ `moments_offset`
(low EMA percentile from the CP1 `Moments` head) and $\sigma_{\mathrm{inv}}$
the inverse scale (high-low EMA range, clamped). The algebra:

$$
A_t = \tilde\Lambda_t - \tilde B_t
= \frac{\Lambda_t - \mu}{\sigma_{\mathrm{inv}}} - \frac{B_t - \mu}{\sigma_{\mathrm{inv}}}
= \frac{\Lambda_t - B_t}{\sigma_{\mathrm{inv}}}.
$$

**Algorithm-level insight (the user's framing in the brief).** The offset
$\mu$ cancels algebraically. The per-term form is **not** a numerical
improvement — math-reviewer notes that the per-term form has *more*
rounding events (five) than the cancelled form (two), so a from-scratch
JAX port for numerical purity would use the cancelled form. The per-term
form exists because **sheeprl writes it that way**, and CP7's job is
bit-identity. The two forms differ by ≤ a few ULPs in float32 in
pathological cases; preserving the sheeprl rounding pattern keeps the
CP7 diff-tool runners (a CP8 target) within tight thresholds.

**The advantage's scale-invariance is what the percentile-EMA buys.**
Without the $\sigma_{\mathrm{inv}}$ division, the policy gradient
$\nabla_\theta \log\pi_\theta(a) \cdot A$ scales linearly with the absolute
magnitude of $\Lambda - B$, which scales with the absolute magnitude of the
reward signal. In a reward-shaping regime where the agent's reward range
shifts mid-training (e.g., the interoceptive-AI noxious-stimulus reward
regime changing as the world model's predicted Bernoulli noxious-event
probability is learned), this would couple the **gradient magnitude** to
**training-progress confounds**. The Moments head's percentile-EMA range
absorbs that signal: $\sigma_{\mathrm{inv}}$ tracks the 95th–5th percentile
spread of $\Lambda$, so $A_t \approx (\Lambda_t - B_t) / \mathrm{spread}$
is roughly in $[-1, +1]$ regardless of the absolute reward magnitude.
This is the same scale-invariance trick as PPO's reward normalisation but
applied to the **value** signal (post-bootstrap), not the **reward**
signal (pre-bootstrap) — algorithmically a slightly different beast,
because $\Lambda$ already integrates over many reward steps.

**Why $\mu$ is the low percentile, not the median.** Sheeprl's
`Moments(percentile_low=0.05, percentile_high=0.95)` (CP1 default) tracks
the **5th** and **95th** percentile. The offset $\mu = $ low percentile
is **not** an unbiased mean — it's the 5th percentile, a *one-sided
location* estimate. The advantage $\tilde\Lambda_t = (\Lambda_t - \mu) /
\sigma_{\mathrm{inv}}$ therefore shifts $\Lambda$ so that the 5th
percentile maps to 0 and the 95th maps to 1, giving advantages roughly in
$[0, 1]$ in the bulk of the distribution. After subtraction of the
similarly-shifted baseline, the advantage is roughly in $[-1, +1]$ as
claimed. This is the **percentile-normalised return shaping** from Hafner
et al. 2023 §3.4 ("Reward and value normalization"). ✅

**Bayesian-DL framing.** The Moments percentile-EMA is a robust
location-and-scale estimator with bounded influence — it discards the
extreme tails of $\Lambda$ rather than letting a single outlier blow up
$\sigma_{\mathrm{inv}}$. Compare to the more naive mean/std normaliser
that would be sensitive to high-variance early-training value targets.
This is a deliberate robustness choice in the DreamerV3 design — it
prevents the value-function-bootstrap-blowup pathology where one
miscalibrated $\Lambda_t \gg \sigma^{\mathrm{naive}}$ destroys the
normaliser. ✅

### 4. REINFORCE objective + entropy regularisation — gradient flow correctness

The CP7 REINFORCE construction at `train.py:589–600`:

```python
objective = log_probs * jax.lax.stop_gradient(advantage)                  # L589
entropy_term = ent_coef * entropy[:-1]                                    # L592
discount_weights = discount[:-1]  # already sg'd by compute_discount       # L598
policy_loss = -jnp.mean(discount_weights * (objective + entropy_term))    # L600
```

matches sheeprl L283–L297 term-for-term. The algorithmic form:

$$
\mathcal{L}_\pi \;=\; -\,\mathbb{E}_{[:H-1]}\!\left[\,d_t \cdot \Big(
\underbrace{\log\pi_\theta\!\big(\mathrm{sg}[a_t]\big)\,\mathrm{sg}[A_t]}_{\text{REINFORCE}}
\;+\;\underbrace{\beta_H \cdot \mathcal{H}[\pi_t]}_{\text{entropy}}
\Big)\right].
$$

**Gradient-flow analysis (the score-function estimator).** REINFORCE
requires the gradient to flow **only through $\log\pi$**, not through
$A$ or the sampled $a$. The two `stop_gradient` sites:

1. **$\mathrm{sg}[A_t]$**: applied at `train.py:589`,
   `jax.lax.stop_gradient(advantage)`. This severs the gradient path
   $\partial A / \partial \phi$ (which would otherwise flow into the
   critic through `predicted_values` and `lambda_values`), ensuring the
   actor loss does not become a critic update in disguise. Matches
   sheeprl L291 `.detach()`. ✅
2. **$\mathrm{sg}[a_t]$ inside `log_prob`**: this is the
   `p.log_prob(imgnd_act.detach())` construction at sheeprl L286. The
   reasoning is subtle: $a_t$ was sampled from $\pi_\theta$ via the
   reparameterisation / Gumbel-softmax / straight-through path, so
   $\partial a / \partial \theta \neq 0$ via the sampling op. The
   REINFORCE estimator is the *score function* estimator, which
   differentiates $\log \pi_\theta(a)$ as a function of $\theta$ at
   *fixed* $a$ — therefore $\partial \log\pi_\theta(a) / \partial \theta$
   must be computed at $a$ treated as a constant. Without
   $\mathrm{sg}[a]$, the gradient picks up an extra term
   $(\partial \log\pi_\theta(a) / \partial a) \cdot (\partial a / \partial \theta)$,
   which is the reparameterisation-gradient term — a valid estimator on
   its own (the pathwise estimator) but **different** from the score
   function. Mixing the two without explicit weighting is a classic
   off-by-one in policy-gradient implementations.

   **CP7's handling of $\mathrm{sg}[a]$.** Math-reviewer notes that the
   `compute_actor_objective` function signature **accepts pre-computed
   `log_probs`**, with the docstring at L540–L545 explicitly stating
   "log_probs is computed with stop_gradient already applied to the
   sampled action". The responsibility is correctly placed with the
   caller — which is CP8's actor forward pass, where the sampling op
   happens and where `log_prob(sg(action))` should be called. This is
   the **same structural location** as sheeprl L286 (`p.log_prob(...)`
   inside the policy forward pass, with `imgnd_act.detach()` as the
   argument). ✅ at CP7; flagged as CP8 review item.

**Entropy regularisation.** The $\beta_H \cdot \mathcal{H}[\pi_t]$ term
at L592 implements the standard Williams-1992 / Mnih-2016 entropy bonus
for exploration. With $\beta_H = 3 \times 10^{-4}$ (sheeprl XS default),
the bonus is small enough that the gradient is dominated by the
advantage-weighted log-prob in the bulk regime, but large enough to
prevent premature collapse of $\pi_\theta$ onto a single action when
the advantage signal is weak (early training, sparse-reward zones).
The `entropy[:-1]` slice matches sheeprl L295 `entropy.unsqueeze(dim=-1)[:-1]`
— the trailing-1 axis is the caller's (CP8's) responsibility to produce. ✅

**Discount weighting** (sheeprl L297). The factor $d_t$ multiplied
**outside** the parenthesised `(objective + entropy)` term is the §S6
discount, already `stop_gradient`'d at `compute_discount`. This makes
$d_t$ a **fixed importance schedule** — the actor objective at later
imagined steps is exponentially down-weighted by $\gamma^t$ (modulated
by the predicted continues, with §S5 anchoring $d_0$ to ground truth).
The placement *outside* the parenthesised sum is correct: both the
REINFORCE term and the entropy term share the same discount schedule.
✅ matches sheeprl L297 exactly.

**Sign and reduction.** The `-jnp.mean(...)` at L600 turns the
maximisable objective into a minimisable loss. The `jnp.mean(...)`
without an `axis=` argument averages over **all axes** of the input
`[H, BT, 1]` tensor — matching sheeprl's `torch.mean(...)` over all
axes. ✅

### 5. `stop_gradient(action)` deferred to CP8 — the right structural location

The math-reviewer's CP7 nit and the user's brief both flag this: the
`stop_gradient` on the sampled action is **not** applied inside
`compute_actor_objective`. The function consumes pre-computed `log_probs`
that the caller is expected to have produced with `stop_gradient`
already on the action.

**This is the right design choice for two reasons.**

First, **structural fidelity to sheeprl L286**: the
`p.log_prob(imgnd_act.detach())` form happens **at the policy distribution's
`log_prob` call site**, which is inside the actor forward pass — not at
the policy-loss assembly site. The JAX `compute_actor_objective` is
isolated from the actor forward pass (Lever-B isolation rule: it does
not import from `src.models.dreamer_v3_*`), so it physically cannot
apply `stop_gradient` to the action — the action is already encoded
into the scalar `log_probs` by the time `compute_actor_objective`
receives it. Pushing the `stop_gradient` into `compute_actor_objective`
would require it to also receive the raw action tensor and call
`log_prob` itself, which would break the Lever-B isolation.

Second, **engineering ergonomics**: the `log_probs` argument is a
scalar-per-step tensor of shape `[H, BT, 1]` (summed over action
dimensions per sheeprl L284–L290). Pushing the `stop_gradient(action)`
inside `compute_actor_objective` would require it to also handle the
multi-action-head sum (for discrete actions with multiple action
dimensions), or to handle continuous vs discrete branching (sheeprl
L281–L292's `if is_continuous: ... else: ...`). The cleaner design is
the one CP7 has: `compute_actor_objective` is action-form-agnostic;
the caller produces a single `log_probs` tensor with whatever
action-head logic is needed and with `stop_gradient(action)` already
applied.

**CP8 review checklist item (concurring with math-reviewer's nit).**
At CP8's actor forward pass, verify:
1. The sampled `imagined_actions` tensor is wrapped in
   `jax.lax.stop_gradient` before being passed to `log_prob`.
2. The `log_prob` sum over action dimensions matches sheeprl L284–L290
   (discrete-only for grid-world, no continuous branch).
3. The resulting `log_probs[:-1]` slice (matching L286's
   `.unsqueeze(-1)[:-1]`) is the input to `compute_actor_objective`.

**Function signature verification (the user's ask).** Lines 498–507 of
`train.py`:

```
def compute_actor_objective(
    log_probs: jax.Array,             # caller provides; sg(action) applied upstream
    lambda_values: jax.Array,
    predicted_values: jax.Array,
    moments_offset: jax.Array,
    moments_invscale: jax.Array,
    entropy: jax.Array,
    discount: jax.Array,
    ent_coef: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
```

The signature confirms `log_probs` is a top-level input, not constructed
internally. The `sg(action)` responsibility is unambiguously with the
caller. ✅

### 6. §S-rule cross-reference — CP7 scope correctly bounded

The §S-rules distribute across the CP chain. CP7's two implemented rules
(in **bold**) plus the rules CP7 must not touch (verified):

| §S-rule | Scope | CP7 status |
|---|---|---|
| §S1, §S2, §S3, §S4 | env-side / RSSM / GRU / training-loop | not in CP7 — `train.py` does not import RSSM/GRU and has no env interaction ✅ |
| **§S5 (true-continue splice)** | imagination rollout, before `compute_lambda_values` | **fully implemented**: `train.py:473–477` (the splice) + `train.py:481–486` (consumed by `compute_lambda_values`) + `train.py:489` (consumed by `compute_discount`) ✅ |
| §S6 (discount cumprod weighting) | actor + critic losses | **substrated at CP6**; CP7 consumes via `compute_discount(continues_spliced, gamma)` at L489 ✅ |
| **§S7 (Moments advantage normalisation)** | actor only | **fully implemented**: `train.py:573–583` (per-term normalisation + advantage); CP1 `Moments` head substrates `moments_offset` and `moments_invscale` ✅ |
| §S8 (free-nats per-element floor) | reconstruction loss | substrated at CP6 — not in CP7 |
| §S9 (`Independent(BernoulliSafeMode, 1)` wrap on continue) | reconstruction + imagination | substrated at CP6 — CP7 consumes `continues_predicted` from the caller, which is expected to apply the wrap upstream (CP8 imagination rollout) ✅ |
| §S10 (`continue = 1 - terminated`) | reconstruction loss, target side | substrated at CP6 — and **the §S5 splice at CP7 uses the same `1 - terminated` formula** for the spliced `continues_spliced[0]`, providing symmetry between the world-model training target and the imagination rollout's first-step anchor |

**Critical observation.** §S5 + §S7 are CP7's domain, and both are
implemented fully. §S6 (CP6) is consumed correctly. The CP7 scope is
correctly bounded — no overreach into §S4 (RSSM, CP4), §S8/§S9
(reconstruction loss, CP6), or §S10 (target-side continue, CP6). ✅

### 7. D-011 algorithm-level impact — same as D-001, exact arithmetic

D-011 logs that `polyak_update` returns a new dict rather than mutating in
place. The mathematical content:

- **Sheeprl form**: `tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)`
  (in-place mutation of the target params tensor inside the PyTorch
  optimisation loop).
- **JAX form**: `{k: (1-tau)*target[k] + tau*online[k] for k in online}`
  (functional dict return, forced by JAX's prohibition on array mutation
  inside JIT-traced code).

**Mathematically identical**: the EMA blend is the same, the order of
operands in the addition is commuted (`(1-τ)·target + τ·online` instead of
`τ·online + (1-τ)·target`), and float32 IEEE-754 addition is commutative
under the conditions math-reviewer establishes. Measured `max_abs_diff =
0.000e+00` on all three CP7 Polyak diff-tool runners.

**Substrate-class match with D-001.** D-001 logs the *same kind* of mechanism
change for `moments_update` (PyTorch in-place EMA buffer update → JAX
pure-functional `MomentsState` replacement). D-001 was approved by PI on
2026-05-13 with measured `max_abs_diff = 8.2e-8`. CP7's D-011 has an **even
cleaner** `0.000e+00` measurement because the Polyak path is pure float32
scalar arithmetic without the percentile-quantile chain that gave D-001 its
8.2e-8 ULP drift.

**No optimisation-trajectory impact.** The argument is identical to D-001's
ratification rationale: a pure-functional return that produces the same
arithmetic output as the in-place mutation is **observationally identical**
from the optimiser's perspective. The next `train()` call consumes the
returned dict; the gradient flowing back into the online critic at step $t+1$
is computed against the same numerical $\bar\phi$ regardless of mechanism.
Algorithm semantics are preserved exactly. ✅

**Substrate precedent band.** D-011 joins the pure-functional-return
substrate class:

| Deviation | CP | Function | Measured `max_abs_diff` | Status |
|---|---|---|---|---|
| D-001 | CP1 | `moments_update` | 8.2e-8 | ✅ APPROVED 2026-05-13 |
| D-011 | CP7 | `polyak_update` | **0.000e+00** | ☐ pending (this gate) |

D-001 is the **single direct precedent**. D-011 is unambiguously the same
class with a cleaner numerical witness. **D-011 should ratify with no
math-reviewer or professor reservations.** ✅

### 8. Process compliance — D-011 logged as pending, gate ordering respected

**Positive note.** The developer correctly logged D-011 as `☐ pending`
in `DEVIATION_LOG.md` row D-011. The Lever-E grep
(`git diff eeba638 3c5be0c -- DEVIATION_LOG.md`) returns only the D-011
`☐ pending` addition; no `✅ APPROVED` flips in the CP7 commit range
(per code-reviewer's audit).

**Second clean Lever-E cycle in a row.** The pattern that began at CP6
(D-010 logged as `☐ pending` after the CP4+CP4b incident, commit `4491c66`)
is now sustained at CP7 (D-011 logged as `☐ pending`). The CP4 PI call's
Lever-C reviewer-gate strengthening (2026-05-14 Process notes in
`DEVIATION_LOG.md`) continues to work as designed.

**Why this matters at the algorithm-fidelity level.** The deviation register
is the audit trail for every measurable JAX-vs-sheeprl difference. Its
integrity depends on gate ordering: reviewers identify → PI ratifies →
verdict cell flips with a link to the PI call doc. The CP7 cycle is the
**second checkpoint after the strengthening**, and the developer respected
it — meaning the gate-ordering discipline is no longer a one-off corrective
response to a Lever-E incident but a sustained workflow norm. Credit. ✅

Continuing this discipline at CP8 (actor forward pass + training-loop
assembly — likely substrate deviations on the Gumbel-softmax sampler) and
CP9 (deeper MLP chain deviations, reconstruction-loss bit-identity tests)
will keep the trail clean.

## Per-function algorithm-fidelity audit

| # | Function | Algorithm content | Verdict |
|---|----------|-------------------|---------|
| 1 | `polyak_update` (`train.py:329–387`) | Slow-target EMA blend $(1-\tau)\theta^{\mathrm{tgt}} + \tau\theta^{\mathrm{on}}$; $\tau = 0.02$ (sheeprl XS default); half-life ~35 steps. Pure-functional dict return (D-011 mechanism). Bit-identical to sheeprl L680 under float32 IEEE-754 addition-commutativity. Three diff-tool runners report `0.000e+00`. | ✅ |
| 2 | `compute_imagined_returns` (`train.py:394–491`) | §S5 splice at index 0 with `1 - terminated_observed`; consumes spliced continues into `compute_lambda_values(predicted_rewards[1:], predicted_values[1:], spliced[1:] * gamma, ...)` and `compute_discount(spliced, gamma)`. Splice happens BEFORE both downstream consumers. Shape contract `[H+1, BT, 1]` preserved through `[1, BT, 1]` concat with `[H, BT, 1]` on axis 0. | ✅ |
| 3 | `compute_actor_objective` (`train.py:498–602`) | §S7 per-term advantage normalisation with `(lambda - offset)/invscale - (baseline - offset)/invscale` (offset cancels algebraically; per-term form preserved for sheeprl rounding-pattern bit-identity). REINFORCE objective `log_probs * sg(advantage)` with entropy bonus `ent_coef * entropy[:-1]`. Discount weighting `discount[:-1]` outside the parenthesised sum. `-jnp.mean(...)` sign. `sg(action)` deferred to caller (CP8 actor forward pass), matching sheeprl L286 structural location. | ✅ |

## Concerns

### 🟡 Concern 1 — `stop_gradient(action)` is a CP8 dependency

`compute_actor_objective` consumes pre-computed `log_probs` with the contract
that `stop_gradient` has already been applied to the sampled action upstream.
This is the **right structural location** (matches sheeprl L286
`p.log_prob(imgnd_act.detach())`), but it introduces a **CP8 review
dependency**: if the CP8 actor forward pass forgets to apply
`jax.lax.stop_gradient` to `imagined_actions` before computing `log_prob`,
the REINFORCE estimator silently becomes a mixed score-function +
reparameterisation-gradient estimator, and the policy-gradient updates pick
up an extra non-REINFORCE term.

**Detection signature.** The bug would not show up as a numerical mismatch
on the CP7 diff-tool runners (which are unit tests with fixed `log_probs`).
It would show up at CP8's actor-objective bit-identity runner (which feeds
the actor forward pass + `compute_actor_objective` together) as a
**non-zero `max_abs_diff` on the policy gradient**, even when the forward
`log_probs` match — because the gradient picks up the extra
$\partial \log\pi / \partial a \cdot \partial a / \partial \theta$ term
through the sampling op.

**Recommendation.** CP8 review checklist must include:
1. **Code-reviewer**: grep for `jax.lax.stop_gradient` wrapping
   `imagined_actions` (or equivalent) before any `log_prob` call.
2. **Math-reviewer**: re-derive the REINFORCE gradient flow at CP8 and
   confirm only $\nabla_\theta \log\pi$ contributes.
3. **Professor (me at CP8)**: confirm the structural location matches
   sheeprl L286 (`detach()` inside `log_prob`, not at the policy-loss site).

Non-blocker for CP7. Flagged for CP8.

### 🟢 Nit 1 — Polyak fires-before-train ordering needs CP8 reaffirmation

CP7's `test_polyak_fires_before_train_step` enforces the call-order
invariant via a two-step trace + grep check on `train.py`. CP7 does not
yet have a full training-loop assembly (CP8's domain), so the test
verifies the **stub** ordering. CP8's training-loop assembly must
**preserve** this ordering — i.e., the `polyak_update` call must precede
the `one_train_step` call within the inner gradient-step loop, matching
sheeprl L679–L686.

**Recommendation.** CP8 code-reviewer audit must include a grep check
that `polyak_update(...)` appears in the source above `one_train_step(...)`
within the inner gradient-step loop. (The CP7 test already does this; CP8
must extend it to the full loop assembly, not just the stub.)

Non-blocker for CP7. Flagged for CP8.

### 🟢 Nit 2 — §S5 splice fixture-side test would close a structural-bug gap

The §S5 splice is currently defended by:
- **Code-reviewer**: line-by-line check that the concat is at index 0
  with the right tensors (line 73 of the CP7 code review).
- **Math-reviewer**: shape contract verification (Eq. 2 of the math
  review).

A **bit-identity test on `compute_imagined_returns`** against a fixed
fixture (where `continues_predicted[0]` differs from `1 - terminated_obs`,
so the splice has a *visible* effect on the output `continues_spliced[0]`)
would defend against a future regression where someone "simplifies" the
splice to `continues_spliced = continues_predicted` and the test still
passes because `lambda_values` and `discount` are within tolerance.

The current implementation is correct — this is a forward-looking
defensive recommendation.

**Recommendation.** Add a CP8 Lever-A test on `compute_imagined_returns`
with a fixture where `continues_predicted[0] = 0.7` and
`terminated_observed = 0.0`, so `continues_spliced[0]` must equal `1.0`
(not `0.7`) after the splice. The assertion
`continues_spliced[0].mean() == 1.0` would fire if the splice were
removed.

Non-blocker for CP7. Flagged for CP8.

## Closest published precedents

- **Hafner et al. 2023, "Mastering Diverse Domains through World Models"**
  (DreamerV3, [arxiv:2301.04104](https://arxiv.org/abs/2301.04104)).
  Polyak target critic + §S5 true-continue splice + §S7 percentile
  return normalization — all three are §3.4 of the paper. The XS config's
  `tau: 0.02` is the model-size-invariant default in the paper's
  hyperparameter table.
- **Polyak & Juditsky 1992, "Acceleration of stochastic approximation by
  averaging"**. The original Polyak averaging trick — DreamerV3's EMA
  target is the exponentially-weighted variant, with $\tau = 0.02$
  giving a half-life of ~35 updates.
- **Williams 1992, "Simple statistical gradient-following algorithms for
  connectionist reinforcement learning"** (REINFORCE). The score-function
  estimator at the heart of `compute_actor_objective`'s
  `log_probs * sg(advantage)` form.
- **Mnih et al. 2016, "Asynchronous methods for deep reinforcement
  learning"** (A3C, [arxiv:1602.01783](https://arxiv.org/abs/1602.01783)).
  The entropy bonus $\beta_H \cdot \mathcal{H}[\pi_t]$ form used in §S7,
  with the same outside-the-parenthesised-sum placement.
- **Hafner et al. 2020, "Mastering Atari with Discrete World Models"**
  (DreamerV2, [arxiv:2010.02193](https://arxiv.org/abs/2010.02193)).
  Predecessor that **did not** use the §S5 splice — V2 used the
  world-model continue head at all imagined steps. The §S5 splice is
  one of the V3-vs-V2 algorithmic differentiators in the value-bootstrap
  pipeline.

## One-line conclusion

CP7 — Polyak EMA target-critic update (slow-target trick for the CP6
two-term critic loss), §S5 true-continue splice (anchoring imagination
rollout's first-step discount to the buffer's `1 - terminated` ground
truth, severing the world-model continue-head's calibration error at
episode boundaries), and §S7 per-term advantage normalisation +
REINFORCE objective (Moments-percentile scale-invariance, score-function
estimator with `stop_gradient` on advantage inline and `stop_gradient`
on action correctly deferred to the CP8 actor forward pass) — is
algorithm-fidelity correct against the DreamerV3 paper and the sheeprl
reference at `dreamer_v3.py:L246–L297` + `:L673–L697`. D-011 is the
**same substrate-mechanical class as D-001** (pure-functional dict return
replacing in-place tensor mutation; forced by JAX's no-mutation-in-JIT
constraint) with measured `max_abs_diff = 0.000e+00` — cleaner than
D-001's 8.2e-8, since the Polyak path is pure scalar float32 arithmetic
with no percentile/quantile chain. Process discipline (D-011 logged as
`☐ pending`, no autonomous verdict flip) sustained from the CP6
restoration — credit to the developer for the second clean Lever-E
cycle. **PASS.**

## Next steps

- **pi** — ratify D-011 at the CP7 gate. APPROVE at the substrate-class
  precedent of D-001 (pure-functional return replaces in-place mutation;
  exact arithmetic, no numerical deviation). No threshold adjustment
  needed — measured `max_abs_diff = 0.000e+00` is the cleanest result in
  the deviation series. CP7 four-gate sweep otherwise closed pending this
  ratification.
- **senior-developer** — at CP8 planning, propagate the three flagged
  forward-looking items:
  (i) require a code-reviewer grep check for `jax.lax.stop_gradient`
  wrapping `imagined_actions` before any `log_prob` call (Concern 1);
  (ii) require a code-reviewer grep check that `polyak_update` appears
  in the source above `one_train_step` within the inner gradient-step
  loop, matching sheeprl L679–L686 (Nit 1); (iii) add a
  `compute_imagined_returns` fixture-side Lever-A test where the splice
  has a visible effect on `continues_spliced[0]` (Nit 2).
- **developer** — no CP7 action items. Continue the D-011-style
  `☐ pending` discipline at CP8 (likely Gumbel-softmax sampler
  deviations) and CP9 (deeper MLP chain deviations). The CP7 cycle is
  the **second clean** post-strengthening checkpoint; do not regress.

Reviewed by: professor-rl-bayesian-dl
