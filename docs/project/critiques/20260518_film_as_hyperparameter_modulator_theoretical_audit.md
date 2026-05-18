---
title: "FiLM / hypernetwork as RL hyperparameter modulator — independent theoretical audit"
status: draft
author: professor-dl-theory
audience: user + pi + research-postdoc
date: 2026-05-18
scope: "Independent theoretical audit (no project docs read). First-principles assessment of whether a FiLM-like operator $h \\mapsto \\gamma(c)\\odot h + \\beta(c)$, or a more general hypernetwork $\\theta = g_\\phi(c)$, can play the role of a learned modulator for the four RL hyperparameters that Doya (2002) mapped to ascending neuromodulators (NA→inverse-temperature, ACh→learning-rate, DA→TD-error gain, 5-HT→discount)."
one_line_summary: "Verdict — partial-with-a-category-warning: NA-temperature and DA-gain are clean forward-pass FiLM targets; ACh-learning-rate and 5-HT-discount live outside the forward pass and are reachable only via meta-gradient or unrolled-Bellman hypernetwork extensions, not by ordinary FiLM."
---

## Headline

A FiLM operator $h \mapsto \gamma(c)\odot h + \beta(c)$ is a section of a **forward-pass** parameter bundle. Two of Doya's four neuromodulator-hyperparameter assignments live in the forward pass and are reached cleanly; two live in the **learning operator** or the **temporal sum** and are *not* reachable by forward-pass FiLM without auxiliary machinery. Calling a single shared FiLM block a "neuromodulator analogue" across all four is therefore a partial claim at best and a category error at worst — but the partial claim is real and rests on a clean geometric structure.

## Plain-language entry point

Doya (2002) proposed that four brain chemical systems each control one knob of a reinforcement-learning algorithm: noradrenaline sets how greedy the policy is (the "temperature"), acetylcholine sets the learning rate, dopamine scales the reward-prediction error, and serotonin sets how far into the future the agent looks (the discount). The question of this memo is whether a single modern conditional-network trick — FiLM, which lets one network output a *multiplicative scale* and *additive shift* applied to another network's activations — can simultaneously play *all four* of those neuromodulator roles. The short answer is: it can play the first and third cleanly (they're just numbers that already multiply the policy logits and the TD error), it can only fake the second (the learning rate is a property of the gradient update, not of the forward pass), and it cannot play the fourth without unrolling the Bellman equation through the network (the discount is a temporal-sum property). The deeper point is that a FiLM operator is a section of a forward-pass bundle, and two of Doya's knobs simply don't live on that bundle.

## Q1 — Per-hyperparameter feasibility

### NA → inverse temperature $\beta$ in policy softmax

**Where it lives.** Strictly inside the forward pass, at the final policy head. The standard softmax policy is
$$\pi(a \mid s) \;=\; \frac{\exp(\beta\, z_a(s))}{\sum_{a'} \exp(\beta\, z_{a'}(s))}$$
where $z_a(s)$ are the action logits. The hyperparameter $\beta$ multiplies the logits pre-softmax. In tempered-softmax form, $\beta = 1/T$.

**FiLM reach.** Trivial — and in fact, FiLM applied to logits *is* a temperature head. Let the FiLM layer at the logit be $z \mapsto \gamma(c)\odot z + \beta_{\text{FiLM}}(c)$. If $\gamma(c) = \beta_{\text{Doya}}(c)\cdot \mathbf{1}$ (a single scalar broadcast over all action dimensions) and $\beta_{\text{FiLM}}(c) = 0$, then
$$\pi(a \mid s, c) \;=\; \frac{\exp(\gamma(c)\, z_a(s))}{\sum_{a'} \exp(\gamma(c)\, z_{a'}(s))}$$
which is exactly the context-conditioned tempered softmax with $T(c) = 1/\gamma(c)$.

**Form, rank, identifiability.** The mapping FiLM-at-logits $\to$ inverse-temperature is a *rank-1* slice of the FiLM bundle: a single shared scalar over the action axis. The unused degrees of freedom — non-broadcast $\gamma$ and any non-trivial $\beta_{\text{FiLM}}$ — implement context-conditioned **action priors** (the $\beta_{\text{FiLM}}$ shift) and **per-action temperatures** (the per-coordinate $\gamma$). These extra degrees of freedom are *identifiable from one another* only if the context $c$ produces sufficiently varied logits across actions; for a stateless or near-stationary policy they collapse. Practical implication: if you want FiLM-at-logits to *be* the NA system, you should constrain $\gamma$ to broadcast and $\beta_{\text{FiLM}}=0$ — otherwise the network silently learns a context-conditioned action prior under the same banner.

**Empirical signature distinguishing structure from metaphor.** Train with and without the broadcast constraint; if both give the same return profile but the unconstrained version has $\beta_{\text{FiLM}}$ with large per-action variance, the system is learning action priors, not a temperature. The Doya-NA reading is then decorative.

### DA → TD-error gain

**Where it lives.** In the loss / update, multiplying the reward-prediction error. The TD-error is $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, and DA-gain enters as a multiplicative scalar $\kappa_{\text{DA}}$ on $\delta_t$ in the actor / critic gradient: $\Delta\theta \propto \kappa_{\text{DA}}\, \delta_t\, \nabla_\theta \log\pi$ (or similar). It does not appear in the forward pass *unless* we equivalently let it scale the reward signal or the advantage.

**FiLM reach.** Indirect but clean. By the chain rule, multiplying the *advantage* by $\kappa(c)$ in the loss is equivalent to multiplying the *gradient* by $\kappa(c)$, which is equivalent — for the loss surface of the policy head only — to scaling the policy logits' effective learning rate by $\kappa(c)$. A FiLM scale on the value head's output (i.e., $V(s,c) = \gamma_V(c) \tilde V(s) + \beta_V(c)$) does *not* by itself give DA-gain; it gives a context-conditioned value rescaling that *does* propagate into $\delta$ but in a coupled way ($\gamma_V$ scales both $V(s_{t+1})$ and $V(s_t)$, so the relevant term is $\gamma_V (V(s_{t+1}) - V(s_t)) + r_t \cdot 1$, not $\kappa\delta$). The clean route is to expose $\kappa(c)$ as a learned scalar in the **loss** — strictly speaking this is no longer "FiLM" but a *hypernetwork emitting a loss-coefficient*. The architectural distinction matters.

**Form and identifiability.** The cleanest form is
$$\mathcal{L}_\pi(\theta; c) = -\,\mathbb{E}\big[\kappa(c)\, \delta_t\, \log\pi_\theta(a_t \mid s_t)\big].$$
This is identifiable up to a *positive scalar gauge*: $(\kappa, \text{lr}) \mapsto (\lambda\kappa, \text{lr}/\lambda)$ leaves the gradient invariant. So DA-gain and ACh-learning-rate are *not separately identifiable* from training dynamics alone — they only become identifiable when paired with a second-order signal (a curvature / Fisher / natural-gradient term that breaks the gauge), or when a non-trivial reward baseline pins the scale.

**Empirical signature.** Sweep $\lambda$ in $(\kappa, \text{lr}) \to (\lambda\kappa, \text{lr}/\lambda)$ explicitly; if training curves are invariant (or invariant up to second-order effects like Adam normalization), the gauge is real and "DA-gain" is not a *separately* meaningful quantity in your codebase. If they differ, you are implicitly relying on Adam-style preconditioning to break the gauge — worth surfacing.

### ACh → learning rate $\alpha$

**Where it lives.** In the **learning operator**, not the forward pass: $\theta \leftarrow \theta - \alpha(c)\,\nabla_\theta \mathcal{L}$. The learning rate is not an activation, not a weight, not a logit — it is a hyperparameter of the optimiser.

**FiLM reach.** *Not by ordinary FiLM.* A forward-pass operator cannot, by construction, modify the optimiser step size. What it *can* do is **emulate** a per-context learning rate via the gauge identity above (DA-gain $\equiv$ lr up to gauge); but this is the same gauge collapse — you are not modulating ACh, you are renaming DA. To actually reach ACh-lr requires one of:

1. **Meta-gradient / learned optimiser:** $\alpha = g_\phi(c)$ with $\phi$ trained by differentiating through one or more optimiser steps (Xu, van Hasselt & Silver 2018 *Meta-Gradient RL*; Andrychowicz et al. 2016 *Learning to Learn*; Metz et al. *Learned Optimizers*). The hypernetwork emits an *optimiser hyperparameter*, not a forward-pass scale.
2. **Hypernetwork weight emission with non-trivial Jacobian:** $\theta = g_\phi(c)$ followed by SGD on $\phi$. By the chain rule the *effective* learning rate of $\theta$ at context $c$ is $\alpha\, \|\partial_\phi g\|^2$. This is a *coupled* per-parameter, per-context learning rate, geometrically a pullback of the optimiser's metric through $g_\phi$. It is *not* the simple scalar $\alpha(c)$ Doya posited.
3. **Implicit second-order channels:** activation-norm rescaling can change effective learning rates through Adam's denominator. This is a well-known artifact (Bjorck et al. 2018 on BatchNorm and learning rate; van Laarhoven 2017) and is the route by which *FiLM as ACh* can be *accidentally* defended — the FiLM scale changes activation magnitudes, which through Adam's per-parameter normalisation changes the per-parameter effective step size. This is a *real* effect but it is a side channel, not the structure Doya specified.

**Form and identifiability.** Even with the cleanest route (1), $\alpha(c)$ is *not* identifiable from on-policy returns alone in finite training time — it is confounded with $\kappa$ (DA-gain), with the Fisher metric scale, and with the trust-region radius. Identifiability requires a controlled experiment that *holds the loss-gradient direction fixed and varies only the step size* — typically an offline replay sweep.

**Empirical signature.** A *FiLM-only* implementation claiming ACh-role should fail a *gradient-direction-preserving* perturbation test: take a trained agent, evaluate it at two contexts $c_1, c_2$ on *the same batch*, and check whether the *direction* of $\nabla_\theta\mathcal{L}$ differs (FiLM-as-ACh prediction: directions identical, only magnitude differs) or differs in direction (FiLM-as-feature-conditioning: directions differ; ACh metaphor is decorative).

### 5-HT → discount $\gamma_{\text{Bellman}}$

**Where it lives.** Inside the **temporal sum** in the Bellman backup: $V(s_t) = \mathbb{E}[\sum_{k\geq 0} \gamma^k r_{t+k}]$, equivalently $V(s_t) = \mathbb{E}[r_t + \gamma V(s_{t+1})]$. The discount $\gamma$ is a property of the *Bellman operator*, a non-local temporal recursion across the trajectory.

**FiLM reach.** *Not by forward-pass FiLM*, and the obstruction is deeper than for ACh. The discount controls a *fixed point of an infinite operator* (the Bellman backup); a per-timestep forward-pass scaling does not produce a global change to the fixed point of the same form. Concretely: if you scale the value-head output by $\gamma_V(c)$, you get $\tilde V(s,c) = \gamma_V(c) V(s)$, and the Bellman target becomes $r_t + \gamma\,\gamma_V(c)\, V(s_{t+1})$ — but the *current* value side is also $\gamma_V(c) V(s_t)$, so the residual is $r_t + \gamma\,\gamma_V(c) V(s_{t+1}) - \gamma_V(c) V(s_t)$, which is *not* the residual of any standard MDP with effective discount $\gamma_{\text{eff}}$. The algebra simply does not close — FiLM-at-V is not a discount change.

The next-simplest hypernetwork structure that *could* reach 5-HT is one that emits the **discount itself** to the Bellman backup: $\gamma_B(c) = g_\phi(c) \in (0,1)$, used in the value-target construction. This is a hypernetwork over the *return estimator*, not the forward pass. The closest published precedent is *Meta-Gradient RL* (Xu, van Hasselt & Silver 2018), which learns $\gamma$ by differentiating through the TD target — but even that learns a single scalar $\gamma$, not a $\gamma(c)$.

The deeper theoretical worry: a *context-conditioned discount* changes the very definition of "value at $s$" because the same state under two contexts has two different value functions over the *same* future. Either the context is part of the state — in which case there is no separate "context-conditioned discount", only a state-conditioned discount inside an enlarged MDP — or the context is exogenous, in which case the Bellman fixed point is no longer well-defined.

**Empirical signature.** A FiLM-only claim of 5-HT-role should fail an effective-horizon test: estimate the effective horizon of the trained policy (e.g., via the autocorrelation of the value function or the temporal sensitivity of the policy to delayed rewards) as a function of $c$. FiLM-as-5-HT prediction: effective horizon shifts with $c$. FiLM-as-feature-conditioning prediction: it does not, and any observed shift is mediated by changes in policy entropy or value-function shape.

## Q2 — Unifying theoretical statement

**The deepest single statement.** A FiLM operator is a *section of a low-dimensional locally-trivial parameter bundle over the forward-pass activation manifold*. Doya's four hyperparameters live in four *different* parts of the RL computation graph: (i) the forward pass (NA), (ii) the loss-weight (DA, modulo the lr-gauge), (iii) the optimiser (ACh), and (iv) the Bellman operator (5-HT). The *only* hyperparameters cleanly reachable by FiLM are those whose action on the loss is **representable as a forward-pass linear operator on activations**. NA satisfies this trivially; DA satisfies it up to a global scalar gauge with the learning rate; ACh satisfies it only via the implicit Adam-normalisation side channel; 5-HT does not satisfy it at any locality.

A precise reformulation: let $G \cdot \theta$ denote the action of the FiLM group $G = (\mathbb{R}_{>0}\rtimes \mathbb{R})^d$ on the activations of one layer. The orbit of $G$ over the parameter-functional manifold reaches *function-class equivalents* of NA-temperature modulation (when $G$ acts at the logit layer) but does *not* reach the operator class containing the Bellman discount or the gradient-descent step. This is a statement about the **commutator** of the FiLM group with the RL operators: FiLM commutes with the softmax-and-argmax composite (giving NA reach) and with linear loss-weighting (giving DA reach), but *not* with the gradient operator (ACh) or the Bellman fixed-point operator (5-HT). The category-theoretic flavour: FiLM is an endofunctor on the *activation* category; the optimiser and the Bellman operator are functors on different categories (the *gradient* category, the *value-function* category). One cannot replace the others without natural-transformation structure that ordinary FiLM lacks.

Information-geometrically: FiLM modulates the **mean** of the model's output distribution (and, through the softmax, also its precision at logits — that's the NA story). The *Fisher metric* on parameter space, which is what genuinely controls effective learning rates and natural-gradient preconditioning (Amari 1998), is *not* a function of activations alone — it depends on the Jacobian of the output with respect to *parameters*. A FiLM operator does not see this object. To reach ACh-flavoured modulation one needs either a *natural-gradient hypernetwork* (a hypernetwork emitting a Fisher-preconditioner, à la KFAC with context-conditioned blocks) or a meta-gradient learner.

## Q3 — Novel targets the Doya frame might miss

The Doya frame is forward-pass-shaped (NA), loss-shaped (DA), optimiser-shaped (ACh), and temporal-sum-shaped (5-HT) — but only one of those is FiLM-native. A more architecturally honest "hypernet-as-RL-modulator" frame should be organised by *what part of the computation graph the hypernet reaches*, not by neuromodulator. Promising forward-pass targets that FiLM can reach cleanly:

- **Per-state policy entropy temperature** (a refinement of NA): a per-state scalar $\beta(s, c)$ on logits, with empirical signature being a context-dependent entropy schedule. Cleanly FiLM-reachable. Relates to *Soft Actor-Critic* style entropy bonuses (Haarnoja et al. 2018).
- **Symexp / two-hot critic transform parameters** (Dreamer V3, Hafner et al. 2023): the symlog/symexp encoding has a width parameter; FiLM at the critic head could context-condition the encoding sharpness, with the empirical signature being a context-conditioned change in critic-target distribution shape.
- **Action prior / policy bias** (the orthogonal complement of NA-temperature): $\beta_{\text{FiLM}}$ on logits at non-zero values. Cleanly reachable; identifiability requires symmetry-breaking across actions.
- **Observation precision-weighting** (active-inference-flavoured): FiLM on early sensory layers with $\gamma$ as a per-channel precision, with empirical signature being a context-conditioned change in input-perturbation Jacobian norms. This is the closest *non-RL-hyperparameter* target that FiLM reaches naturally and is information-geometrically clean.

Targets that *look* like FiLM territory but require extended hypernet machinery:

- **GAE-$\lambda$**: lives in the return estimator, like 5-HT-discount; same Bellman-fixed-point obstruction. Reachable only by a hypernetwork emitting $\lambda$ to the return-estimator function, not by FiLM.
- **Trust-region KL coefficient / PPO clip ratio**: live in the loss-clipping operator. A hypernetwork emitting the clip ratio is possible (Hessel et al. *Discovered Policy Optimization* lineage). Not FiLM-shaped; clip is non-differentiable.
- **KFAC / natural-gradient preconditioner block scaling**: context-conditioned curvature is geometrically natural — this is *the* information-geometric extension of FiLM, with the hypernetwork emitting (rescalings of) blocks of the Fisher matrix. To my knowledge there is no clean published precedent for *context-conditioned KFAC*; this is a genuinely open architectural direction. I don't recall the specifics of any prior work here, but it would be the natural target if one wanted to take "ACh as natural-gradient scaling" seriously.
- **Replay-buffer importance weight / priority exponent** (PER, Schaul et al. 2015): lives in the sampling operator; reachable by a hypernet emitting priority exponents per (transition, context) pair, but the gradient through the sampling operation requires a Gumbel-style or score-function relaxation. Not FiLM-shaped.

The genuinely novel suggestion from a fiber-bundle viewpoint: rather than asking "can FiLM be all four Doya knobs", ask "what is the *largest* RL-hyperparameter subgroup whose action factors through forward-pass FiLM, and what is the *smallest* hypernetwork extension that reaches the complement". The first answer is roughly $\{$NA-temperature, action prior, observation precision, critic-transform sharpness$\}$. The second answer is a *two-headed* hypernetwork: one head emits FiLM scales and shifts to the forward pass (the "fast" channel — NA-like), and the other head emits scalar coefficients to the loss/return-estimator (the "slow" channel — DA/5-HT-like). ACh in this picture is *not* a separate head; it is the gauge dual of the DA head, broken by the choice of optimiser preconditioner.

## Q4 — Verdict

**Partial, with a category-warning.** A FiLM operator gives a clean, identifiable mapping for Doya's NA-inverse-temperature (it *is* a context-conditioned temperature, modulo a constraint on the FiLM shape); a clean mapping for DA-TD-error-gain only up to a positive-scalar gauge with the learning rate (so "DA" and "ACh" are not separately identifiable without auxiliary structure); a side-channel-only mapping for ACh-learning-rate (via Adam-normalisation, not as architecture); and *no* mapping at all for 5-HT-discount, which lives in the Bellman fixed-point operator and is unreachable by any forward-pass linear modulation. A unifying claim that "FiLM is a learned neuromodulator" is therefore a partial theorem for two of Doya's four and a metaphor for the other two — and the metaphor is load-bearing only if paired with a second hypernetwork channel reaching the loss / return-estimator. The cleanest re-statement is architectural, not neuromodulator-shaped: **forward-pass FiLM reaches the activation-linear subgroup of RL hyperparameters; the complement requires an extended hypernetwork over the loss, the optimiser, or the Bellman operator.**

## References used (canonical, by name)

- Doya, K. (2002). *Metalearning and neuromodulation*. Neural Networks 15, 495–506.
- Perez, E. et al. (2017). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018.
- Ha, D., Dai, A., Le, Q. (2016). *HyperNetworks*. ICLR 2017.
- Jacot, A., Gabriel, F., Hongler, C. (2018). *Neural Tangent Kernel*. NeurIPS.
- Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*. Neural Computation.
- Xu, Z., van Hasselt, H., Silver, D. (2018). *Meta-Gradient Reinforcement Learning*. NeurIPS.
- Andrychowicz, M. et al. (2016). *Learning to Learn by Gradient Descent by Gradient Descent*. NeurIPS.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic*. ICML.
- Hafner, D. et al. (2023). *Dreamer V3: Mastering Diverse Domains*. arXiv.
- Bronstein, M. et al. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. arXiv.
- Cohen, T. et al. *Gauge-Equivariant CNNs* line.
- Galanti, T., Wolf, L. (2020). *On the Modularity of Hypernetworks*. NeurIPS.
- Cao (2019); Courts & Kvinge (2021) BundleNet; Coda et al. (2022) BMNet — fiber-bundle / hypernet line of work, cited for the "section of a parameter bundle" framing only; I do not recall their specifics in detail and have not re-read them for this memo.
- Schaul, T. et al. (2015). *Prioritized Experience Replay*.
- Rodriguez-Garcia (2026) — I do not recall the specifics of this work and have not used it in this analysis.
