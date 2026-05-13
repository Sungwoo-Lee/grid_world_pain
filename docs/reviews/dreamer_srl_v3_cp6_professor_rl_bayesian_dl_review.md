---
title: "dreamer-srl v3 CP6 — professor-rl-bayesian-dl audit (critic loss + cascade fix #29 + §S6/§S8/§S9)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/train.py, src/algorithms/dreamer_srl/loss.py (extensions)
---

# dreamer-srl v3 CP6 — professor-rl-bayesian-dl audit

## Plain-language verdict

This is the **third and final technical gate on CP6** — the port of sheeprl's
critic-loss assembly (the value-function update inside `dreamer_v3.py:L240-L320`)
to a new `src/algorithms/dreamer_srl/train.py`, plus bonus §S8 free-nats and §S9
`Independent(BernoulliSafeMode, 1)` plumbing inside the existing
`src/algorithms/dreamer_srl/loss.py`. Code-reviewer and math-reviewer have
already PASSED — both log-prob terms are present (cascade fix #29 lands), the
`/ gamma` post-cumprod gives `discount[0] = 1.0` exactly, the free-nats floor
is per-element BEFORE the mean (not the literalist post-mean form), and the
trailing event-dim sum in `IndependentBernoulli.log_prob` matches PyTorch's
`Independent(Bernoulli, 1)` semantics. **My job is the algorithm-integration
level**: does this loss compute the actual signal that Hafner §3.3 prescribes,
and does it couple sensibly with what CP7 will later contribute?

**It does.** The two NLL terms implement the *bootstrapped critic-regression
target* and the *EMA slow-target self-regulariser* — the two halves of the
DreamerV3 value head update. Both targets are `stop_gradient`'d, so gradient
flows only through the online critic's logits; the §S6 discount mask is
`stop_gradient`'d at construction, so the discount weighting acts as a *fixed
importance schedule* — not a learnable gate — which is the right computational
semantics (any gradient flow into `continues` from the critic loss would
couple the world-model's continue head to the value-update signal, a known
instability source not in the DreamerV3 prescription). The §S8 per-element
free-nats floor is the right Bayesian-DL form: it caps posterior-collapse
pressure at every `(t, b)` cell, not just on the mean. The §S9 `Independent`
wrap is shape-correctness machinery (sums over the trailing event-dim of
size 1) producing the `[T, B]` shape that downstream `discount * log_prob`
multiplication requires.

**D-010** (1.33× margin over the observed 3.099e-5) is the same
substrate-mechanical class as D-003 / D-006 / D-007 / D-008 — JAX-vs-PyTorch
float32 reduction-order ULP drift over an algebraically identical pipeline
(literally the same `TwoHotEncoding.log_prob` code path as D-006). The forward
drift is invisible to the optimisation signal: gradients flow through the
critic's *logits*, not through the symlog-encoded target's bin lookup; the
3.099e-5 forward-pass disagreement corresponds to relative drift ~4e-6, two
orders of magnitude below Adam's effective gradient noise floor.
Margin-band-wise, math-reviewer's recommendation to raise the threshold from
4e-5 to 5e-5 (1.61× margin, median of the precedent band) is reasonable; both
are defensible.

**Process-discipline positive note.** The D-010 verdict cell reads `☐ pending
— PI ratification at CP6 gate`. The developer did not autonomously flip it to
`✅ APPROVED`, in clear contrast to the CP4 + CP4b incident (commit
`4491c66`) where two cells were pre-flipped in the implementation commit
itself. This is the gate-respect strengthening (Process notes 2026-05-14)
working as designed and deserves credit.

**Verdict: PASS.** Forward D-010 to PI with concurrence.

## Algorithm-fidelity audit (the 8 numbered points from the review brief)

### 1. Cascade fix #29 — Hafner §3.3 slow-target self-regulariser is algorithmically present

The two `log_prob` terms in `compute_critic_loss` (`train.py:231` and
`train.py:235`) implement the value-head update from Hafner et al. (2023,
*DreamerV3*) Eq. (8):

$$
\mathcal{L}_{\mathrm{critic}}(\phi) \;=\;
-\,\mathbb{E}_{q_\phi}\!\left[
\log q_\phi(v_t \mid s_t)\big[\,R_t^\lambda\,\big]
\;+\;
\log q_\phi(v_t \mid s_t)\big[\,\mathrm{sg}(v^{\mathrm{tgt}}_t)\,\big]
\right]
$$

where $q_\phi(v_t \mid s_t)$ is the online critic's discrete two-hot
distribution over 255 symlog-space bins, $R_t^\lambda$ is the bootstrapped
TD-$\lambda$ return (sheeprl `lambda_values.detach()` at L314), and
$v^{\mathrm{tgt}}_t$ is the EMA target critic's expected value
(`predicted_target_values.detach()` at L315; this is `.mean` of
`TwoHotEncodingDistribution(target_critic(traj))`).

The first term is the standard **TD($\lambda$) regression** target. The
second is the **slow-target self-regulariser**, which makes the critic chase
a convex combination of its own bootstrap and its slow-moving EMA mean —
preventing the bootstrap from running away from the slow target during the
high-variance early-training phase. The v1 in-house DreamerV3 was missing
the second term (cascade fix #29). Symptom: a slow, silent training-quality
bug where the value head goes unhinged while the loss curve still looks
plausible.

**The JAX port has BOTH terms** at `train.py:231` (`neg_lp1`) and
`train.py:235` (`neg_lp2`), each `stop_gradient`'d on its respective target.
The test at `test_critic_loss_two_terms:160-166` asserts `max(|neg_lp2|) >
0.1` so the missing-second-term regression fires loudly.

**Gradient-flow analysis.** With both targets `stop_gradient`'d,

$$
\nabla_\phi \mathcal{L}_{\mathrm{critic}}
\;=\;
-\nabla_\phi \log q_\phi\!\big[R_t^\lambda\big]
\;-\;
\nabla_\phi \log q_\phi\!\big[v^{\mathrm{tgt}}_t\big],
$$

i.e., two cross-entropy gradients summed over the same online distribution
— exactly as DreamerV3 intends. The discount mask
`discount[:-1].squeeze(-1)` is `stop_gradient`'d at construction
(`train.py:138`) so it acts as a fixed per-`(t, b)` importance weight on
the combined loss — no spurious gradient flows from critic loss into the
world-model's continue head. ✅

### 2. §S6 discount weighting — `discount[0] = 1` invariant and the §S5/§S6 boundary

The §S6 trick

$$
d_t \;=\; \mathrm{sg}\!\left(\frac{\prod_{i=0}^{t}\,c_i \,\gamma}{\gamma}\right)
\;=\;
\mathrm{sg}\!\left(\gamma^{t} \prod_{i=0}^{t} c_i\right)
$$

gives $d_0 = c_0$ (the divide-by-$\gamma$ cancels the multiply-by-$\gamma$
at index 0). When $c_0 = 1$ (the §S5 *true-continue splice* — "this
trajectory started from a real env state that did not just terminate"),
$d_0 = 1$ exactly. Without the `/ gamma`, $d_0$ would equal $\gamma$
(≈ 0.997 default), under-weighting the very first imagined step. The
JAX port lands the trick at `train.py:138-140`.

**Step-weight structure.** With $\gamma \approx 1$ and $c_i = 1$ everywhere
(no predicted termination within horizon), $d_t = \gamma^t$ — standard
TD-$\lambda$ exponential decay. With $c_i \in [0, 1]$ (continue-head
probabilities), the discount additionally **zeros out imagined steps that
lie past a predicted termination**, preventing the critic from regressing
against bootstrap targets sampled in post-terminal hallucinated states.

**The §S5 vs §S6 boundary is correctly drawn.** §S5 (true-continue splice
of `1 - terminated` into `continues[0]`) lives at CP7, not CP6. The JAX
`compute_discount` **does not splice** — it consumes a `continues` tensor
that the caller is expected to have spliced. Docstring at `train.py:107-
112` makes the contract explicit. CP6 fixture pre-builds `continues[0] = 1`
to mimic the post-splice shape, which is the right isolation strategy for
a CP6 unit test. **CP6 does not accidentally implement §S5.** ✅

**Slice `[:-1]` semantics.** `discount` has shape `[H+1, BT, 1]`; the
critic loss applies `discount[:-1].squeeze(-1)` to get `[H, BT]`. The
H+1-th step is dropped because `compute_lambda_values` returns
λ-targets for $t = 0, \ldots, H-1$ only (the last step has no defined
λ-return — would require a value-bootstrap beyond the horizon). Matches
sheeprl L316 exactly. ✅

### 3. §S8 free-nats — per-element BEFORE mean, not the literalist post-mean trap

The Bayesian-DL-correct form is

$$
\mathcal{L}_{\mathrm{KL}}^{\,\mathrm{floor}} \;=\;
\mathbb{E}_{(t,b)}\!\left[\max\!\big(\mathrm{KL}_{tb},\,\nu\big)\right],
$$

with the $\max$ taken **element-wise** over the $[T, B]$ KL tensor before
the expectation. The JAX port (`loss.py:548`, `loss.py:556`):

```python
dyn_loss  = kl_dynamic        * jnp.maximum(dyn_loss,  kl_free_nats)  # [T, B]
repr_loss = kl_representation * jnp.maximum(repr_loss, kl_free_nats)  # [T, B]
kl_loss   = dyn_loss + repr_loss                                      # [T, B]
total     = (kl_regularizer * kl_loss + …).mean()                     # scalar
```

`jnp.maximum` broadcasts the scalar `kl_free_nats` against the `[T, B]`
tensor — element-wise max, matching sheeprl L68-L74 term-for-term.

**The literalist trap and why it matters.** A reader who takes
$\mathcal{L} = \max(\mathrm{KL}, \nu)$ at face value without watching the
shapes can collapse it to

$$
\mathcal{L}_{\mathrm{wrong}} \;=\;
\max\!\left(\mathbb{E}_{(t,b)}\!\left[\mathrm{KL}_{tb}\right],\,\nu\right),
$$

which is **silently broken in the regime where the trap matters most**.
The per-element form floors *every* $(t, b)$ cell at $\nu$, preventing
posterior collapse at any time-batch position; the post-mean form only
fires when the *mean* KL drops below $\nu$, which essentially never happens
once training is underway (mean KL is typically O(1)–O(10), well above
the default $\nu = 1.0$). The post-mean form therefore reduces to
unconstrained KL minimisation — and the world-model collapses its
posterior to the prior in regions of state space the policy doesn't visit
(sparse-reward zones, the "you've already won" terminal-corner of the
grid in this project's idiom). The JAX port avoids this trap. ✅

**Bayesian-DL framing.** Free-nats is the Higgins et al. (2017) β-VAE /
Burgess et al. (2018) information-bottleneck floor: it forbids the
posterior from being arbitrarily close to the prior on a per-element basis,
keeping the latent code informative. DreamerV3 repurposes it to keep the
RSSM's stochastic $z_t$ from collapsing onto the deterministic $h_t$'s
prior — making per-element flooring (not post-mean) the operative choice.

### 4. §S9 `Independent(BernoulliSafeMode, 1)` wrap — shape and semantics

The wrap re-interprets the trailing axis of size 1 as an *event dim*,
producing the right PyTorch semantics:

| Method | Input shape | Output shape | Reduction |
|---|---|---|---|
| `log_prob` | `[T, B, 1]` (binary targets) | `[T, B]` | sum over event-dim 1 |
| `mode` | (no input; on logits `[T, B, 1]`) | `[T, B, 1]` | no reduction |

The JAX `IndependentBernoulli` (`loss.py:357-417`) matches both: `log_prob`
calls `base.log_prob(value).sum(axis=-1)` (sum of one element over a
size-1 axis — numerically a no-op, but the **shape change** is what
downstream code needs); `mode` returns `self._base.mode` directly (no
reduction). The downstream CP6 call sites — `pc.log_prob(continue_targets)`
in `reconstruction_loss` (`loss.py:566`, producing `[T, B]`) and (future
CP7) `Independent(BernoulliSafeMode(logits=...), 1).mode` driving the
`continues` tensor — both depend on the shape contract. The wrap is
**shape-correctness machinery, not probability content** — the sum over a
size-1 axis is mathematically trivial, but it's the contract that lets
downstream code multiply against `[T, B]` masks without fragile
broadcasting. ✅

### 5. `BernoulliSafeMode.log_prob` numerical stability

The JAX port (`loss.py:351-354`) uses the stable form

$$
\log p(x \mid \ell) \;=\;
x \cdot \log\!\sigma(\ell) \;+\; (1 - x) \cdot \log\!\sigma(-\ell),
$$

via `jax.nn.log_sigmoid` (branch-stable: uses
$\log\sigma(\ell) = \ell - \mathrm{softplus}(\ell)$ when $\ell < 0$ to
avoid $\log(0)$ underflow at large positive $\ell$). This is
mathematically identical to PyTorch's
`F.binary_cross_entropy_with_logits(reduction='none')` negated, so
`BernoulliSafeMode.log_prob(x) = -BCE_with_logits(logits, x)` term-for-
term. The stable form is the right choice for continue logits that may run
to large magnitude once the world-model becomes confident about
"this state will not terminate" — the naive
$\log(\mathrm{sigmoid}(\ell))$ would underflow when $\ell \ll 0$. Both
code-reviewer (per-test table) and math-reviewer (re-derivation summarised
in the brief) confirmed the equivalence. ✅

### 6. §S-rule cross-reference — CP6 scope is correctly bounded

The §S-rules distribute across the CP chain. CP6's three implemented rules
(in **bold**) plus the four rules CP6 must not touch (verified):

| §S-rule | Scope | CP6 status |
|---|---|---|
| §S1, §S2, §S3, §S4 | env-side / RSSM / GRU / training-loop | not in CP6 — `train.py` does not import RSSM/GRU and has no env interaction ✅ |
| §S5 (true-continue splice) | imagination rollout, before `compute_lambda_values` | **NOT in CP6** — `compute_discount` consumes pre-spliced `continues`; the splice itself is CP7's responsibility ✅ |
| **§S6 (discount cumprod weighting)** | actor + critic losses | **fully implemented**: `compute_discount` (`train.py:88-141`) + `discount[:-1].squeeze(-1)` slice (`train.py:239`) ✅ |
| §S7 (Moments advantage normalisation) | actor only | not in CP6 — `compute_critic_loss` correctly uses raw `lambda_values`, NOT Moments-normed (documented at `train.py:180-187`) ✅ |
| **§S8 (free-nats per-element floor)** | reconstruction loss | **fully implemented**: `loss.py:548` + `loss.py:556` ✅ |
| **§S9 (`Independent` wrap on continue)** | reconstruction + imagination | **fully implemented**: `BernoulliSafeMode` (`loss.py:252-354`) + `IndependentBernoulli` (`loss.py:357-417`) ✅ |
| §S10 (`continue = 1 - terminated`) | reconstruction loss, target side | not in CP6 — `reconstruction_loss` consumes `continue_targets` from the caller |

**Critical observation.** §S5 is **not** touched by CP6, in line with the
review brief's instruction. The CP6 fixture pre-builds `continues[0] = 1`
to mimic the post-splice shape — right isolation strategy for an isolated
CP6 unit test. ✅

### 7. D-010 algorithm-level impact — same as D-006 / D-007 / D-008

Forward to PI with concurrence. D-010 is a 3.099e-5 `max_abs_diff` on
`qv.log_prob(stop_gradient(lambda_values))`, exceeding the CP5 `3e-5`
threshold by < 4%, contained within the new `4e-5` threshold at 1.33×
margin. The class is **bit-identical to D-006** — same
`TwoHotEncoding.log_prob` code path, same `jnp.linspace(-20, 20, 255)`
bin grid, same one-ULP-at-`bins[127]` JAX-vs-PyTorch midpoint
disagreement cascading through bin-lookup → two-hot weights → log-softmax
→ cross-entropy. The only difference from D-006 is the fixture's PRNG seed
(`0xD3EAF + 1` vs `0xD3EAF`), which produced target tensors landing
slightly closer to bin boundaries — more sensitive to the bin-grid ULP
drift.

**Gradient-flow invisibility argument.** The deciding analytical witness.
The critic loss is

$$
\mathcal{L}_{\mathrm{critic}} \;=\;
-\,\sum_{t,b} d_{tb}\,
\Big[\log q_\phi(R^\lambda_{tb} \mid s_{tb}) \;+\;
\log q_\phi(\mathrm{sg}(v^{\mathrm{tgt}}_{tb}) \mid s_{tb})\Big]
\Big/\,(T \cdot B).
$$

The gradient $\nabla_\phi \mathcal{L}$ flows through $\log q_\phi$'s
*logits* (`qv_logits`, the online critic's output), **not** through the
bin grid or the target's symlog-encoding. The 3.099e-5 forward-pass
disagreement is on the *value* of the log-probability after the bin
lookup, not on the *gradient signal* the optimiser sees. To bias the
optimisation signal the ULP drift would have to perturb the derivative
$\partial \log q_\phi / \partial \mathrm{logits} = \mathrm{softmax}(\mathrm{logits})
- \mathrm{one\_hot}(\mathrm{target\_bin})$, which depends on
`target_bin` (an integer index) and `softmax(logits)`, neither of which
is materially perturbed by a 1-ULP drift at `bins[127]`. The only entry
point would be the target bin index changing — and that requires the
symlog-encoded target to land within 7.45e-8 of a bin edge, probability
~$10^{-8}$ per element. Under Adam's running-second-moment normalisation
any such pathological perturbation is absorbed within ~10 steps. **The
3.099e-5 forward drift is invisible to optimisation.** ✅

**Threshold-margin recommendation.** Math-reviewer's soft recommendation
to raise to `5e-5` (1.61× margin, median of the D-003–D-008 precedent
band 1.5×–2.78×) is reasonable — `4e-5` at 1.33× is the tightest in the
series and gives the register slightly less headroom than the precedent
suggests. **My concurrence**: ratify at either `4e-5` or `5e-5`; mild
preference for `5e-5` on margin-band consistency grounds, but no block
on `4e-5`. The deviation class is unambiguous.

**XLA reduction non-determinism (math-reviewer's secondary finding).**
The diff-tool number drifts between runs (implementation report `1.287e-5`;
code-reviewer audit run `2.193e-5`; both within `4e-5`). Math-reviewer's
ULP-random-walk analysis predicts ~3.4e-5 scale for run-to-run variation
over the reduction tree; observed `1-2 × 10^{-5}` is consistent. This is
JAX-XLA JIT-cache invalidation behaviour across separate Python process
invocations (different cache states → different reduction-tree shapes for
the same sum), not non-determinism *within* a single process. Does not
threaten bit-identity. ✅

### 8. Process compliance — D-010 logged as pending, gate ordering respected

**Positive note.** The developer correctly logged D-010 as
`☐ pending — PI ratification at CP6 gate`. The Lever-C grep
(`git diff 8e201e5..1a4e51e -- DEVIATION_LOG.md | grep "^\+.*✅ APPROVED"`)
returns empty (per code-reviewer's audit). This contrasts with commit
`4491c66` (CP4 + CP4b implementation) where D-008 + D-009 verdict cells
were prematurely written as APPROVED *before* the PI gate had been opened
— corrected via PI ratification + cell replacement in commits `4563579`
and `8e201e5`.

**Why this matters.** The deviation register is the audit trail for
every measurable JAX-vs-sheeprl difference. Its integrity depends on
gate ordering: reviewers identify → PI ratifies → cell flips with a link
to the PI call doc. Pre-emptive flipping in the implementation commit
skips two steps and makes the PI's actual call invisible in the register.
The 2026-05-14 Process notes document the incident and corrective
strengthening; **the CP6 cycle is the first checkpoint after the
strengthening, and the developer respected it**. Credit. ✅ Continuing
this discipline at CP7 (Moments + EMA-polyak deviations likely) and CP9
(deeper MLP chain deviations likely) will keep the trail clean.

## Per-test algorithm-fidelity audit

| # | Test | Algorithm content | Verdict |
|---|------|-------------------|---------|
| 1 | `test_critic_loss_two_terms` | Both NLL terms present; `stop_gradient` on both targets; discount weighting applied; cascade-fix-#29 defended by `max(|neg_lp2|) > 0.1` (catches missing-second-term regression). Combined-mean form `mean((neg_lp1 + neg_lp2) * d_w)` matches sheeprl L316 bit-identically. | ✅ |
| 2 | `test_critic_target_lambda` | Critic regresses against raw `lambda_values`, NOT Moments-normed. Fixture-side distinguishability assertion (`diff_normed > 1e-4`) confirms raw vs. normed are *separable* in the fixture — defends the §S7 vs §S6 separation algorithmically. | ✅ |
| 3 | `test_discount_weighting` | Three sub-assertions: full discount matches reference; `discount[0].mean() ≈ 1.0` invariant (§S6 trick); sliced `[:-1].squeeze(-1)` shape `[H, BT]` matches downstream contract. The `jax.grad` flow-check (lines 317-328) is a real structural witness that `compute_discount` wraps in `stop_gradient` — a commented intent without the actual `lax.stop_gradient` call would fail at `max_grad > 1e-10`. | ✅ |
| 4 | `test_train_module_does_not_import_from_src_models` | Anchored regex isolation-rule check. | ✅ |

## Concerns

### 🟡 Concern 1 — Margin-band consistency for D-010 (forward to PI)

D-010's `4e-5` threshold gives a `1.33×` margin, the tightest in the
substrate-mechanical precedent band (D-003 1.5×, D-006 1.65×, D-007
1.68×, D-008 2.78×). Defensible on the same-code-path-as-D-006 identity
argument, but gives the register slightly less headroom than the
precedent suggests. **Recommendation (concurring with math-reviewer):**
PI may ratify at `5e-5` (1.61× margin, median of the band) for
margin-policy consistency, or at `4e-5` as logged. Either is acceptable
to me; the deviation class is unambiguous. Non-blocker.

### 🟡 Concern 2 — Reconstruction-loss path not bit-identity-tested at CP6

The §S8 free-nats and §S9 `IndependentBernoulli` extensions to
`reconstruction_loss` are landed correctly per code-reviewer's audit, but
do not yet have a CP6 Lever-A bit-identity test against
`sheeprl/loss.py:reconstruction_loss`. The developer's docstring
(`loss.py:497-501`) defers this to CP9. **Reasonable**: the free-nats
and Independent-wrap logic is two lines of code each, the trap surface is
small, the literalist `max(mean, ν)` pathology is the obvious failure
mode and is not present, and the §S9 wrap's only failure is a wrong-shape
output which CP9 will catch immediately. **Recommendation (senior-
developer at CP9 planning):** include explicit per-element bit-identity
assertions on `dyn_loss`, `repr_loss`, `kl_loss` *before* the mean — to
defend §S8 against future "simplification" to the post-mean form.
Non-blocker for CP6.

### 🟢 Nit 1 — Cascade-fix-#29 defensive assertion could be tightened

`test_critic_loss_two_terms:161` asserts `max(|neg_lp2|) > 0.1`, catching
the all-zeros regression. A more precise check would defend against the
copy-paste regression where a developer accidentally writes
`neg_lp2 = -qv.log_prob(stop_gradient(lambda_values))` (same as term 1) —
`neg_lp2` would then be non-zero (passes `> 0.1`) but identical to
`neg_lp1`. A one-line `assert |neg_lp1 - neg_lp2|.max() > 1e-3` would
close the gap. Probably overkill — the fixture-side distinguishability
check in Test 2 already defends a closely related regression — but
worth noting. Non-blocker.

## Closest published precedents

- **Hafner et al. 2023, "Mastering Diverse Domains through World Models"**
  (DreamerV3, [arxiv:2301.04104](https://arxiv.org/abs/2301.04104)).
  Two-term critic loss = Eq. (8); slow-target self-regulariser motivated
  in §3.3.
- **Bellemare, Dabney, Munos 2017, "A Distributional Perspective on RL"**
  (C51, [arxiv:1707.06887](https://arxiv.org/abs/1707.06887)).
  Two-hot-projection lemma (§3.4) — the categorical-projection operator
  used by TwoHotEncoding.
- **Higgins et al. 2017, β-VAE** + **Burgess et al. 2018, "Understanding
  disentangling in β-VAE"**
  ([arxiv:1804.03599](https://arxiv.org/abs/1804.03599)). Per-element-
  vs-post-mean KL floor distinction; DreamerV3 inherits the per-element
  form.
- **Hafner et al. 2020, "Mastering Atari with Discrete World Models"**
  (DreamerV2, [arxiv:2010.02193](https://arxiv.org/abs/2010.02193)). The
  KL balancing (`kl_dynamic`, `kl_representation`) split was introduced
  in V2.

## One-line conclusion

CP6 — two-term critic loss (cascade fix #29 + Hafner §3.3 EMA self-
regulariser), §S6 discount weighting (`discount[0] = 1` invariant), §S8
per-element free-nats floor, and §S9 `Independent(BernoulliSafeMode, 1)`
wrap — is algorithm-fidelity correct against the DreamerV3 paper and the
sheeprl reference at `dreamer_v3.py:L240-L320` + `loss.py:L9-L88`. D-010
is the same substrate-mechanical class as D-006 (literally the same
`TwoHotEncoding.log_prob` code path) and forwards to PI with concurrence
— gradient-flow invisibility argument applies identically; either `4e-5`
or `5e-5` threshold is defensible (mild preference for `5e-5` on
margin-band consistency). Process discipline (D-010 logged as `☐ pending`,
no autonomous verdict flip) restored after the CP4 incident — credit to
the developer. **PASS.**

## Next steps

- **pi** — ratify D-010 at the CP6 gate. APPROVE at either `4e-5` (as
  logged, 1.33× margin, defensible on same-code-path identity argument)
  or `5e-5` (1.61× margin, median of D-003–D-008 precedent band,
  recommended by math-reviewer). Both mathematically defensible; choose
  on margin-policy consistency grounds. CP6 three-gate sweep otherwise
  closed pending this ratification.
- **senior-developer** — at CP9 reconstruction-loss test planning,
  include explicit per-element bit-identity assertions on `dyn_loss`,
  `repr_loss`, `kl_loss` *before* the mean (defends §S8 against
  literalist post-mean regression). Also consider Nit 1's directional
  check (`|neg_lp1 - neg_lp2| > 1e-3`) if a future cleanup pass touches
  the critic loss.
- **developer** — no CP6 action items. Continue the D-010-style
  `☐ pending` discipline at CP7 (Moments + EMA-polyak deviations
  expected) and CP9 (deeper MLP chain deviations expected). The CP6
  cycle has established the post-process-strengthening baseline; do not
  regress.

Reviewed by: professor-rl-bayesian-dl
