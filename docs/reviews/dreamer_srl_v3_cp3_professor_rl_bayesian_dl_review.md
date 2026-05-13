---
title: "dreamer-srl v3 CP3 — professor-rl-bayesian-dl audit (zero-init reward + critic heads)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/agent.py (RewardHead, CriticHead)
prior_gates:
  - docs/reviews/dreamer_srl_v3_cp3_code_review.md
  - docs/reviews/dreamer_srl_v3_cp3_math_review.md
---

# dreamer-srl v3 CP3 — professor-rl-bayesian-dl audit

## Plain-language verdict

This is the third and final technical gate on **CP3** of the dreamer-srl v3
rebuild — the port of sheeprl's "cascade fix #27": *force the output linear of
both the reward head and the critic head to literally zero at construction
time*. Both prior gates have PASSED. Code-reviewer (CP3 code review) confirmed
the implementation overwrites the kernel and bias of two new `nnx.Module`
classes (`RewardHead`, `CriticHead`) via CP1's already-tested
`uniform_init_weights(scale=0.0)` helper plus an explicit `jnp.zeros(...)`
bias, and that fixture parity is exact (`max_abs_diff = 0.000e+00`).
Math-reviewer (CP3 math review) confirmed the formula
$\ell = \sqrt{3s / d_\text{avg}}$ collapses cleanly to $\ell = 0$ at $s = 0$,
that `uniform(-0, 0)` is the constant-zero distribution (no NaN, no negative-
zero ambiguity), and that softmax of the resulting zero-logit vector is the
uniform $1/255$ over the two-hot bin support.

**My job is the algorithm-integration level**: does this zero-init produce the
training-dynamics behaviour DreamerV3 actually needs at iteration 0, and does
it compose cleanly with CP5's `TwoHotEncoding` (the categorical wrapper
downstream CP6 / CP7 will consume)?

**It does.** Zero-init the output linear, the categorical reward / value
distribution at init is the maximum-entropy uniform on the 255 two-hot bins,
the implied real-space expected value is *literally zero* (because the symlog-
space bin centres are symmetric around 0 by `linspace(-20, 20, 255)`
construction, so $\mathbb{E}_{\text{uniform}}[\text{bin}] = 0$ and
$\text{symexp}(0) = 0$), and the cross-entropy gradient at iteration 0 is
$p - y = \mathbf{1}/N - y$ — bounded, direction-free in logit space, and
identical for every state $h$ since the head produces the same zero logits
everywhere. This is the canonical operationalisation of "the critic believes
nothing yet" — uniform prior over bins, zero baseline in reward space, no
spurious random-init peak for the actor's REINFORCE update to chase. The
composition with CP5's `TwoHotEncoding(zeros)` is exactly what CP6/CP7 will
need at their first iteration. No deviations. **Verdict: PASS.** No PI gate
required for CP3 (no D-### entries). senior-developer flips CP3 → CP-PASS
directly.

## Why zero-init *here* matters (the algorithmic case)

DreamerV3 represents reward and value as **discrete categorical distributions
on 255 two-hot bins**, with bin centres at $\mathrm{linspace}(-20, +20, 255)$
in symlog space (CP5's contract — see math-review §Eq. 1 of CP5). The reward
head and the critic head each terminate in a 255-way linear layer that
produces logits $z \in \mathbb{R}^{255}$; the categorical distribution is
$p_k = \mathrm{softmax}(z)_k$, and the real-space expected reward / value is
$\hat r = \mathrm{symexp}(\sum_k p_k \cdot \mathrm{bin}_k)$.

If the output linear is *random-initialised* — even with a small-variance
Hafner-style uniform $\mathcal{U}[-\ell, +\ell]$ with $\ell \sim \mathcal{O}(\sqrt{1/d_\text{in}})$ —
then iteration-0 logits are *structured*: for hidden activation $h$, $z = W^\top h$
is a random linear projection of $h$ onto $\mathbb{R}^{255}$, and certain bins
systematically receive larger logits than others depending on the random
projection direction. Softmax of this non-uniform vector concentrates mass on
a few bins, and the implied expected value $\hat r = \mathrm{symexp}(\sum p_k \mathrm{bin}_k)$
becomes a **state-dependent random function of $h$** — non-zero, biased, and
varying across states purely because of the random init.

This is bad for three coupled reasons:

1. **Critic regression latches on early.** The categorical cross-entropy gradient
   $\partial \mathcal{L} / \partial z = p - y$ has a direction at init — it
   pushes mass *away* from the random-init's spurious peak and *toward* the
   bootstrapped target. The optimiser takes a real step in that direction
   before any environment reward has been credited; this step is "anti-random-
   init noise", not learning.
2. **Actor REINFORCE chases a spurious baseline.** CP7 uses the critic's
   real-space expected value $V_\phi(h)$ as the baseline in advantage
   normalisation $A = \mathrm{lambda\_target} - V_\phi(h)$ (§S7 in the v3
   plan). At init, a random-init critic gives a state-dependent random
   $V_\phi(h)$, which the actor's policy gradient then "corrects" — chasing
   noise.
3. **Reward head couples to the world-model gradient.** The world-model loss
   includes the reward head's NLL against `symlog(r_t)`. Random-init logits
   produce a non-zero, state-dependent gradient through the reward head into
   the RSSM posterior $z_t$ — which then biases the very representation the
   critic is being trained against.

Zero-init the output linear, all three pathologies vanish at iteration 0:

- $z = 0$ identically, for every state $h$, because $W = 0$ kills the
  dependence on $h$ entirely (the bias is also zero, so there is no constant
  offset either).
- $p = \mathbf{1}/255$ uniformly.
- $\hat r = \mathrm{symexp}(\sum_k (1/255) \cdot \mathrm{bin}_k) = \mathrm{symexp}(0) = 0$
  in real reward space, because the bin centres are symmetric around 0.
- The cross-entropy gradient at init is $p - y = \mathbf{1}/N - y$ — bounded
  in $[-1 + 1/N, 1/N]$, symmetric, and *state-independent* (same gradient for
  every $h$, just shifted by the target $y$).

The agent now starts from a clean "uninformative prior" — no random direction
to unlearn, no state-dependent spurious baseline, no biased coupling into the
world model. This is exactly Hafner's design intent.

## Composition with CP5's `TwoHotEncoding` at iteration 0

CP5 wraps the head's output logits in `TwoHotEncoding(logits)`. At init, with
zero-init heads, the composition behaves as follows (all derivations match
CP5 math-review §Eq. 1–§Eq. 3):

| Quantity | Expression at init | Numerical value |
|---|---|---|
| Logits $z$ | $W^\top h + b = 0$ identically | $\mathbf{0} \in \mathbb{R}^{255}$ |
| Softmax probabilities $p_k$ | $e^0 / \sum_j e^0 = 1/N$ | $1/255 \approx 3.92 \times 10^{-3}$ |
| Distribution entropy $H[p]$ | $-\sum_k p_k \log p_k = \log N$ | $\log 255 \approx 5.541$ nats |
| Expected bin (symlog space) $\mathbb{E}[\text{bin}]$ | $\sum_k (1/N) \cdot \mathrm{bin}_k$ | $0$ (bins symmetric around 0) |
| Expected reward / value (real space) | $\mathrm{symexp}(\mathbb{E}[\text{bin}])$ | $\mathrm{symexp}(0) = 0$ |
| `log_prob(value)` for any target | $-\sum_k y_k(\mathrm{value}) \log p_k$ where $\sum_k y_k = 1$ | $-\log(1/N) = \log 255 \approx 5.541$ nats |

The last row deserves a sanity check. CP5's `log_prob` does *not* return
$\log p_k$ for a single bin $k$ — it returns the cross-entropy of the two-hot
target distribution $y(\mathrm{value})$ against the predicted distribution $p$.
At init, $p_k = 1/N$ for all $k$, so $-\sum_k y_k \log p_k = -\log(1/N) \sum_k y_k = \log N$,
independent of $y$ (because the two-hot target sums to 1 by CP5's construction).
**Every target value produces the same NLL at init.** This is the iteration-0
signature CP6 will see — a flat scalar of $\approx 5.541$ nats for the critic
NLL and the reward NLL, regardless of which states are sampled. Any deviation
from this baseline at iteration 0 is a fingerprint of a zero-init failure.

The expected-value composition is the bit CP7 needs:

$$
V_\phi(h)\Big|_{\text{init}} \;=\; \mathrm{symexp}\!\Big(\textstyle\sum_k \mathrm{softmax}(0)_k \cdot \mathrm{bin}_k\Big) \;=\; \mathrm{symexp}(0) \;=\; 0.
$$

Critically, this is **not** a numerical coincidence — it relies on two
properties simultaneously:

1. **Softmax of zero logits is uniform** (any-bin-symmetric).
2. **The two-hot bin grid is symmetric around 0**: `linspace(-20, 20, 255)`
   has $\mathrm{bin}_{127} = 0$ (the median bin is exactly the origin) and
   $\mathrm{bin}_k + \mathrm{bin}_{254 - k} = 0$ for all $k$. So $\sum_k \mathrm{bin}_k = 0$,
   and the uniform-weighted mean is also $0$.

If a future regression were to break either property — e.g., changing the bin
grid to `linspace(0, 20, 255)` (one-sided, for a guaranteed-positive reward
signal), or shifting the median bin off zero — then *even with* zero-init
heads, $V_\phi(h)|_{\text{init}}$ would not be zero, and the algorithmic
guarantee CP3 establishes would silently lose. Worth a recurrence-test flag
for whoever touches CP5's bin grid in the future.

## §S-rule cross-reference

CP3 is upstream of the §S-rule semantic-fix family. None of §S5, §S6, §S7 are
touched by this checkpoint; they happen *downstream*, in CP5's distribution
math (already PASSED) and in CP6's loss assembly / CP7's actor update (not
yet landed). The check I run here is: *does zero-init enable each downstream
§S rule to take effect cleanly, or does it leak into them?*

| §S rule | Location | CP3 interaction | Verdict |
|---|---|---|---|
| §S2 — action-shift | CP2b `action_shift` | Independent of head init; operates on `actions` tensor before RSSM. | ✅ Not touched. |
| §S5 — true-continue splice | CP6 critic-target assembly | Uses `lambda_target` built from the critic's bootstrapped real-space value $V_\phi$. At iteration 0, zero-init makes $V_\phi = 0$ everywhere → bootstrap is zero → `lambda_target` is the discounted true reward sum on the splice path. This is the *correct* initial condition: bootstrap-from-nothing rather than bootstrap-from-random. | ✅ Enabled cleanly. |
| §S6 — discount weighting | CP6 critic loss | Discount weights multiply the per-step NLL; zero-init makes the per-step NLL a constant $\log 255$ across states (see §"Composition" above), so iteration-0 loss is `discount_sum × log(255)` — a clean, predictable scalar. | ✅ Enabled cleanly. |
| §S7 — advantage low-offset cancellation | CP7 actor advantage normalisation | Uses $A = \mathrm{lambda\_target} - V_\phi(h)$. Zero-init gives $V_\phi(h) = 0$, so iteration-0 $A = \mathrm{lambda\_target}$ — the actor sees the raw discounted return, not a noise-corrupted version. The low-offset cancellation §S7 enforces is moot at iteration 0 because the baseline is exactly $0$. | ✅ Enabled cleanly. |

CP3 is the **upstream enabler** of clean §S5/§S6/§S7 dynamics at iteration 0.
None of the downstream rules are silently bypassed; rather, they receive their
ideal initial conditions.

## Architectural-precedent placement

Zero-init of the output layer is not a DreamerV3-specific invention. Closest
precedents:

| Precedent | One-line summary | Relation to CP3 |
|---|---|---|
| **Hafner et al. 2023, "DreamerV3" §B "Network architectures"** (the actual paper this is ported from) | "We initialize the last layer of the reward predictor and critic to zeros, which accelerates early training." | CP3 ports exactly this. The paper does not derive the maximum-entropy interpretation; it asserts the choice empirically. |
| **Bellemare, Dabney & Munos 2017, "A Distributional Perspective on RL" (C51)** | Discrete-categorical value distribution on a fixed grid; final softmax over atoms. | The architectural family CP3 belongs to. C51 does not specifically zero-init the final layer — Hafner's contribution is to recognise the maximum-entropy uniform as the right *prior* over the bin grid. |
| **Andrychowicz et al. 2021, "What Matters in On-Policy RL?" §"value-function initialisation"** | Empirically, scaling down the final critic layer's initialisation by ~0.01 improves PPO on continuous-control benchmarks. | Same family of intervention ("damp the random init at the output to avoid an early-step value-prediction bias"); zero is the limit case. |
| **Henderson et al. 2018, "Deep RL that Matters"** | Documents init-sensitivity of policy-gradient methods. | Empirical motivation for the architectural choice; CP3 inherits it. |
| **Bishop 1994 / Williams 1996, on heteroscedastic regression** | "Initialise the precision-prediction head's bias to a high value, so the network begins maximally uncertain and only sharpens precision as evidence accumulates." | Conceptual analogue from Bayesian DL: a precision head that *starts uninformative* (high variance / low precision) is a closer analogue to a categorical head that *starts uniform*. Zero-init the output of a softmax-categorical head $\Leftrightarrow$ initialise the prior to maximum entropy. |

**The closest thing to what CP3 is doing is Hafner et al. 2023 §B itself.**
CP3 does not differ from the paper — it ports the prescription verbatim. The
algorithmic novelty is Hafner's; CP3's contribution is faithful translation
to JAX/NNX.

## Identifiability and the "wasted PRNG draw" question

A Bayesian-DL-flavoured concern: is the construct-then-overwrite idiom
(random `nnx.Linear` build at L258–L263, then `nnx.Param(...)` overwrite at
L270–L271) creating any *identifiability* issue — e.g., a hidden coupling
between the discarded random kernel and downstream RNG state?

**No.** NNX's `Rngs` container threads keys via `jax.random.split` on each
`make_rng(...)` call; the keys used to build `nnx.Linear`'s kernel are split
off, consumed in `nn.initializers.lecun_normal()(key, shape)`, and *the
resulting `kernel` array is the only artefact that leaks into module state* —
which is then overwritten on the next line. The PRNG stream state in `rngs`
*has advanced* (the split happened), but no downstream module construction
inside `RewardHead` / `CriticHead` reads from `rngs` after the overwrite, so
the advance is invisible to the rest of CP3's state. Any cross-module
correlation that *would* have existed in a hypothetical "no-overwrite" world
is destroyed by the overwrite. Identifiability is preserved.

The one residual cost is one PRNG split per head per construction — at
training time, constructors run once. Negligible.

## Failure-mode signatures (what would *refute* CP3)

For senior-developer / experiment-analyzer downstream: empirical signatures
that would indicate CP3 has silently failed (and the JAX side is not actually
zero-initialising the heads):

1. **Iteration-0 critic NLL $\ne \log 255$.** At the very first gradient
   step, before any updates, the categorical NLL for both reward and critic
   heads should be exactly $\log 255 \approx 5.541$ nats for every batch
   element. A deviation indicates either zero-init failed (kernel or bias
   non-zero) or CP5's `TwoHotEncoding` `log_prob` is mis-implemented. Since
   CP5 already PASSED at both code and math gates, a deviation here would
   point to CP3.
2. **Iteration-0 expected value $V_\phi(h) \ne 0$.** At iteration 0, the
   real-space critic prediction should be exactly $0$ for every state $h$.
   If `wandb` shows iteration-0 `critic_mean` as non-zero, zero-init failed.
3. **Iteration-0 expected reward $\hat r(h) \ne 0$.** Same logic for the
   reward head. CP6's training-loop log should show iteration-0 mean reward
   prediction exactly $0$.
4. **Iteration-0 critic / reward gradient is state-dependent.** Zero-init
   makes the iteration-0 gradient $\partial \mathcal{L}/\partial W$ equal to
   $h \otimes (p - y) = h \otimes (\mathbf{1}/N - y)$ — which *does* depend
   on $h$ (through the outer product), but in a structurally clean way. If
   the *direction* of the gradient varies across states more than this
   formula predicts, that points to non-zero logits (i.e., zero-init failure).

These signatures are easy to add as iteration-0 assertions in the CP6 / CP7
training-loop tests, and I recommend senior-developer flag them as
"recurrence-test signals" when CP6 lands. They are not part of CP3's required
deliverable.

## Findings

| Severity | File:Line | Issue | Suggested action |
|---|---|---|---|
| (none) | — | No findings at the algorithm-integration level. | — |

No 🔴 blockers. No 🟡 concerns. The two prior gates' 🟢 nits (key-arg
cosmetic, construct-then-overwrite wastage, test-file organisation) are
already documented in code-review §Findings; I have nothing to add to them.

## Verdict

✅ **PASS — algorithm-integration is sound, composition with CP5 is exact,
§S-rule downstreams receive ideal initial conditions.**

CP3 ports Hafner et al. 2023 §B's "zero-init the reward and critic output
layers" verbatim. The mechanism is a maximum-entropy prior on the 255 two-hot
bin support, which composes with CP5's `TwoHotEncoding` to give exactly zero
expected reward and zero expected value in real space at iteration 0 —
**because** the symlog-space bin grid `linspace(-20, 20, 255)` is symmetric
around the origin. The cross-entropy NLL is a state-independent constant
$\log 255 \approx 5.541$ nats at iteration 0, the categorical gradient
$\partial \mathcal{L}/\partial z = p - y$ is bounded and direction-free in
logit space, and the §S5/§S6/§S7 downstream rules each receive their ideal
"bootstrap-from-nothing" starting condition. The closest published precedent
is Hafner et al. 2023 §B itself; CP3 does not deviate.

No deviations to log. No D-### entries. **CP3 has no PI gate.**
senior-developer flips CP3 → CP-PASS directly.

Reviewed by: professor-rl-bayesian-dl
