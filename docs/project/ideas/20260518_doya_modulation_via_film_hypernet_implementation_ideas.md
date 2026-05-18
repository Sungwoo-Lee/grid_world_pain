---
title: "Doya's four neuromodulator-hyperparameter assignments — concrete implementation routes under the two-headed FiLM + hypernet architecture"
status: draft
author: professor-dl-theory
polished_by: "research-postdoc (2026-05-18)"
audience: user + pi + research-postdoc
date: 2026-05-18
last_updated: 2026-05-18
companion_to:
  - "docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md"
  - "docs/project/critiques/20260518_film_as_hyperparameter_modulator_audit_story.md"
one_line_summary: "Idea-level sketch of what each of Doya's four neuromodulator-hyperparameter assignments looks like in code under the audit's converged two-headed architecture — Head 1 emits FiLM scales/shifts to the agent's activations, Head 2 emits scalar coefficients to the loss / return estimator. Per knob: the one-line equation, where it lands, what claim is forfeited, what test confirms it."
---

## 1. Plain-language entry point

Doya (2002) proposed that four brain chemical systems each set one knob of an RL algorithm: **noradrenaline (NA)** controls the policy's greediness (*temperature*), **acetylcholine (ACh)** controls the *learning rate*, **dopamine (DA)** controls the *gain* on the reward-prediction error, and **serotonin (5-HT)** controls the *time-discount*. The companion audit concluded that a single FiLM block — a small modulator that emits a multiplier and an offset applied to another network's activations — cannot, alone, play all four, but a **two-headed** design can: one head emits FiLM scales/shifts (forward-pass / conditioning channel), and a second emits scalar coefficients to the *loss* and *return estimator* (loss-side / Bellman-side channel). This memo is the practical follow-up: for each knob, what does the modulator emit, where does it land, and what observation tells us the geometric story is doing real work rather than decorative re-labelling.

## 2. The converged architecture

The audit converged — from three independent directions — on a **two-headed modulator**. One network $m_\phi$ ingests context $c$ (interoceptive state, task signal, learned latent) and emits two structurally distinct streams. **Head 1** emits per-channel FiLM parameters $(\gamma(c), \beta(c))$ acting on the agent's hidden activations (forward-pass channel). **Head 2** emits scalar coefficients $(\kappa(c), \gamma_B(c), \ldots) \in \mathbb{R}^k$ inserted into the loss and the return estimator (loss-side / Bellman-side channel). Head 1 reaches the *activation-linear subgroup* of RL hyperparameters; Head 2 reaches the *operator-level* complement. The heads share a trunk so context representations are amortised, but their outputs sit in different parts of the computation graph.

```
                       context c
                          |
                     [ modulator m_phi ]
                       /          \
              Head 1                Head 2
       (gamma(c), beta(c))     (kappa(c), gamma_B(c), ...)
              |                         |
   FiLM on activations           scalar coeffs into
   (policy logits, critic        loss / return estimator
   hidden, sensory channels)     (advantage gain, target discount)
              |                         |
        forward pass               backward / Bellman
```

## 3. Per-hyperparameter implementation sketches

### NA $\to$ inverse temperature (Head 1 only; the cleanest case)

**Equation.** Apply a rank-1 broadcast FiLM at the policy-logit layer:
$$\pi(a \mid s, c) \;=\; \mathrm{softmax}\!\big(\gamma_{\text{NA}}(c) \cdot z(s)\big),\qquad \gamma_{\text{NA}}(c) \in \mathbb{R}_{>0}.$$

**Placement.** Head 1 output, restricted to a single scalar (broadcast across actions) at the final policy logits. $\beta_{\text{FiLM}}$ at the logits set to zero. This is identically a context-conditioned tempered softmax.

**Honesty caveat.** If you let $\gamma$ vary per action or let $\beta_{\text{FiLM}} \neq 0$ at the logit, the block silently also learns a **context-conditioned action prior**, which is canonically a DA-flavoured signal — see the LC-NA basket audit. The "this is NA" claim then leaks.

**Empirical signature.** Train with the broadcast/no-bias constraint and without; if returns are indistinguishable but the unconstrained $\beta_{\text{FiLM}}$ has large per-action variance, the system is learning action priors and the NA label is decorative (Perez et al. 2017 FiLM gives the architectural primitive).

### ACh $\to$ learning rate (Head 1 only, via the chain rule; coupled with feature gating)

**Equation.** A FiLM block at activation cut $k$ acts as a diagonal Jacobian in the backward pass — the *same* operator both ways:
$$\frac{\partial \mathcal{L}}{\partial W_{\text{upstream}}} \;=\; J_{\text{up}}^\top \,\mathrm{diag}(\gamma(c))\, \frac{\partial \mathcal{L}}{\partial h_k^{\text{post}}}.$$

**Placement.** Head 1 output, placed **upstream of the parameters whose effective learning rate should be modulated**. The gating is per-FiLM-channel and acts only on parameters upstream of the cut.

**Honesty caveat.** Forward-feature-gating and backward-gradient-gating are two views of one operator (Q3 of the ACh addendum). The project **cannot** claim ACh-style learning-rate modulation *without* simultaneously claiming feature gating — there is no FiLM setting that does one and not the other. The Doya scalar $\alpha(c)$ is recovered only in the degenerate broadcast-at-output limit; the generic case is a per-pathway gate of LSTM/GLU/Highway flavour. Loss-side $\kappa$ from Head 2 is *also* learning-rate-like (gauge), so the project cannot separate Head 2's $\kappa$ from Head 1's gradient gating without an external scale-pinning signal.

**Empirical signature.** Stratify upstream parameters by which FiLM channel they predominantly feed (row-norms of the matrix into the FiLM layer). Across two contexts on the same batch: gradient direction in parameter space should be preserved **within strata** and rotate **across strata**. Uniform global rotation ⇒ pure feature-conditioning; no stratified pattern ⇒ ACh-via-chain-rule is decorative.

### DA $\to$ TD-error gain (Head 2 only; loss-side scalar)

**Equation.** A loss-side context-dependent scalar coefficient on the advantage:
$$\mathcal{L}_\pi(\theta; c) \;=\; -\,\mathbb{E}\!\left[\,\kappa_{\text{DA}}(c)\,\delta_t\,\log\pi_\theta(a_t \mid s_t)\right].$$

**Placement.** Head 2 output, a positive scalar multiplied onto the advantage / TD residual before the policy gradient and (optionally) the critic loss.

**Honesty caveat.** Exact gauge with the global learning rate and with the ACh chain-rule channel: $(\kappa, \mathrm{lr}) \mapsto (\lambda\kappa, \mathrm{lr}/\lambda)$ leaves on-policy training invariant up to optimiser-preconditioner effects. The project **cannot claim separation between DA-gain and ACh-learning-rate from training curves alone** — they are the same operator under different names unless a second-order signal (curvature, Fisher, natural-gradient term) breaks the gauge.

**Empirical signature.** Explicit gauge sweep: train two agents with $(\kappa, \mathrm{lr})$ and $(\lambda\kappa, \mathrm{lr}/\lambda)$ for several $\lambda$. If learning curves coincide (under SGD) or differ only through Adam-normalisation, the gauge is real and "DA" is not a separately identifiable claim. If they differ meaningfully under SGD, you are implicitly relying on the optimiser preconditioner — surface this honestly. (Xu, van Hasselt & Silver 2018 *Meta-Gradient RL* is the closest precedent for treating $\kappa$ as a learned scalar.)

### 5-HT $\to$ time-discount (Head 1 + Head 2 jointly; γ-conditional UVFA)

**Equation.** Two pieces are required *together*. Head 1 conditions the critic on $c$; Head 2 emits the discount $\gamma_B(c) \in [0,1)$ used in the target:
$$\mathcal{L}(\phi; c) \;=\; \mathbb{E}\!\left[\big(V_\phi(s; c) - r - \gamma_B(c)\,V_\phi(s'; c)\big)^{\!2}\right],\quad \gamma_B(c) = (\text{Head 2}).$$

**Placement.** Head 1's FiLM scales/shifts modulate the critic's hidden activations (the *indexing* / UVFA mechanism). Head 2's scalar $\gamma_B(c)$ is inserted in the bootstrap target construction; the forward critic never multiplies by $\gamma_B(c)$. The critic learns the family $\{V^\pi_{\gamma_B(c)}\}_c$ as a γ-conditional UVFA (Schaul et al. 2015 UVFA; Sherstan et al. 2020 *γ-Nets* — AAAI 2020, 34(04), 5717–5725, arXiv:1911.07794; Fedus et al. 2019 hyperbolic discounting).

**Honesty caveat.** FiLM alone does **not** implement the discount — the project must say "the discount lives in the return estimator; FiLM provides the conditioning channel that lets one critic store the family." If only Head 1 is implemented, 5-HT is decorative. If $c$ varies *within* a trajectory, the right object is a time-inhomogeneous discounted return, equivalently the value in the enlarged MDP with state $(s, c)$ — flag the convention. A small residual gauge between $\gamma_B(c)$ and a critic-output rescale exists locally and degrades finite-sample conditioning unless context drives a visible spread in effective horizon.

**Empirical signature.** Calibrated-effective-horizon test: per context $c$, perturb a reward at delay $k$ and read $|\Delta V_\phi(s;c)|$. On log scale, slope $\approx \log \gamma_B(c)$ and **must match the loss-side $\gamma_B(c)$** used in training. Slope varying with $c$ and matching training-side $\gamma_B(c)$ confirms γ-conditional UVFA; same slope across all $c$ confirms FiLM-as-feature-conditioning with fixed discount; matching slope but rescaled magnitudes confirms post-hoc rescale — three distinct fingerprints.

## 4. What the modulator can and cannot claim

**Entitled to claim:** (i) Head 1 at the policy logit implements a context-conditioned temperature (NA); (ii) Head 1 upstream of a layer implements a coupled feature-and-gradient gate including per-pathway effective-learning-rate modulation (ACh, strict generalisation of Doya's $\alpha$); (iii) Head 2 on the advantage implements a loss-side RPE gain (DA, up to a gauge); (iv) Head 1 conditioning + Head 2 emitting $\gamma_B(c)$ implements a γ-conditional UVFA (5-HT).

**Not entitled to claim:** (a) that Head 2 alone separates DA-gain from ACh-learning-rate — the positive-scalar gauge means on-policy training curves cannot distinguish them without an explicit gauge-breaker (Adam preconditioning, Fisher / natural-gradient curvature, or a reward-magnitude baseline that pins the scale); (b) that ACh-via-Head-1 is pure learning-rate modulation — forward-feature-gating and backward-gradient-gating are two views of one operator, so any ACh claim is necessarily coupled with feature-conditioning; (c) that 5-HT is implemented by FiLM — the discount lives in the return estimator (Head 2); without $\gamma_B(c)$ implemented in the bootstrap target *separately from FiLM*, FiLM alone never "is" the discount.

## 5. Pointers

- Technical audit with full equations and identifiability conditions: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](../critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md).
- Plain-English story of the audit and the two pushbacks: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_audit_story.md`](../critiques/20260518_film_as_hyperparameter_modulator_audit_story.md).
- Direction memo this is feeding back into: [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md`](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) and successors.

**Next steps by agent.** `senior-developer`: scope a minimal two-headed modulator class with explicit Head 1 / Head 2 separation and a per-knob ablation flag. `experiment-designer`: design the three identifiability tests (broadcast-vs-unconstrained for NA; stratified gradient-direction for ACh; gauge sweep for DA; calibrated-horizon for 5-HT). `professor-rl`: weigh in on the DA-vs-ACh gauge breakers compatible with the project's optimiser (Adam preconditioning gives a partial break; explicit Fisher/KFAC would give a clean one).
