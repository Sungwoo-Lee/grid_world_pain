# Gamma Literature Review — Module C: Generative γ-Models

**Scope of this file.** This is the "C" sub-review of the broader `Gamma/` reference corpus. It covers Janner, Mordatch, and Levine (2020, NeurIPS), *γ-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction*. Two PDF files were supplied for this paper — both are versions of the same work. They are merged into a single review here, with version differences flagged in [Version notes (paper 1 vs. paper 2)](#version-notes-paper-1-vs-paper-2).

**Sister sub-reviews planned for the same corpus:** universal successor features (Borsa 2018), hyperbolic / multi-horizon discounting (Fedus 2019), separating value functions across timescales (Romoff 2019), universal value-function approximators (Schaul 2015), non-exponential discounting (Schultheis 2022), γ-nets (Sherstan 2020). Cross-paper synthesis is deferred to a follow-up curator pass.

---

## Plain-language entry point — what a γ-model is and why a precision-modulated agent might care

The standard "model" in model-based reinforcement learning is a **one-step predictor**: given state $s_t$ and action $a_t$, it returns a distribution over $s_{t+1}$. To plan ten steps ahead you feed the model into itself ten times — and every iteration injects a little prediction error that accumulates exponentially. This is the **compounding-error problem** that makes long-horizon model-based RL brittle.

A **γ-model** sidesteps the rollout entirely. Rather than predicting "where will I be one step from now?", it predicts "where will I tend to be over the entire discounted future, weighted geometrically by how soon I get there?" Concretely, a γ-model samples from the **discounted state-occupancy distribution**

$$
\mu_\gamma^\pi(s' \mid s, a) \;=\; (1-\gamma)\sum_{\Delta t = 1}^\infty \gamma^{\,\Delta t - 1}\, p\!\bigl(s_{t+\Delta t} = s' \mid s_t = s,\, a_t = a,\, \pi\bigr).
$$

In words: with probability $(1-\gamma)$ you sample where the agent will be next step, with probability $\gamma(1-\gamma)$ where it will be two steps from now, with probability $\gamma^2(1-\gamma)$ three steps, and so on. **One feedforward pass of a γ-model gives you a sample from this entire infinite-horizon distribution**, with the timestep itself marginalized out.

The trick that makes this trainable from one-step transitions is a **generative reinterpretation of TD learning**. The same Bellman recursion that lets $Q^\pi$ be learned from one-step rewards lets $\mu_\gamma^\pi$ be learned from one-step transitions:

$$
\mu_\gamma^\pi(s_e \mid s_t, a_t) \;=\; (1-\gamma)\, p(s_e \mid s_t, a_t) \;+\; \gamma\, \mathbb{E}_{s_{t+1} \sim p(\cdot \mid s_t, a_t)}\!\bigl[\mu_\gamma^\pi(s_e \mid s_{t+1})\bigr].
$$

Here $\mu^\pi$ plays the role of $Q$, the next-step occupancy plays the role of $V$, and the one-step transition plays the role of the reward — except every quantity is a **distribution over future states**, not a scalar. The γ-model can be a GAN (sample-only) or a normalizing flow (density-evaluable), and the bootstrap is stabilized with a delayed target network borrowed straight from DQN.

**How is this different from a value function?** A value function $V^\pi(s)$ collapses the entire discounted future into a single scalar by integrating reward against the occupancy. The γ-model keeps the **distribution itself**, with reward integrated out. So $V^\pi(s) = \frac{1}{1-\gamma}\, \mathbb{E}_{s_e \sim \mu_\gamma^\pi(\cdot \mid s)}[r(s_e)]$ — value pops out as a *single feedforward pass* of the γ-model followed by a reward evaluation. This decouples dynamics from reward: the same γ-model trained on policy $\pi$ supports value estimation for any reward function on the same MDP.

**How is this different from the successor representation?** The successor representation $M^\pi(s, s') = \mathbb{E}^\pi[\sum_t \gamma^t \mathbb{1}\{s_t = s'\}]$ stores the discounted *expected visitation count* of each state. In tabular settings, $\mu_\gamma^\pi$ and $M^\pi$ are the same object up to a normalization constant — Janner et al. prove this as their Proposition 1: $\mu_\theta(s_e \mid s_t, a_t) = (1-\gamma)\, M(s_e \mid s_t, a_t)$ at the global minimum. In continuous settings the SR cannot be directly normalized into a sampler; the γ-model is the continuous, **generatively-sampleable** analogue of the SR.

**Why a precision-modulated / multi-horizon agent might care.** The user's interest in γ-models for this project lives at three potential interfaces:

1. **Multi-horizon prediction without per-horizon rollouts.** A γ-model trained at $\gamma$ can be reweighted at test time to match any larger discount $\tilde\gamma \ge \gamma$ via the analytic mixture weights $\alpha_n = (1-\tilde\gamma)(\tilde\gamma - \gamma)^{n-1} / (1-\gamma)^n$. So a single network supports a family of effective horizons — a natural fit for an architecture where the agent must hold multiple temporal scales simultaneously.
2. **Precision as a modulator of γ.** Because γ itself is a hyperparameter governing the geometric horizon, modulating γ (or modulating which γ-head a controller reads from) is a candidate mechanism for "precision-driven horizon shortening" under interoceptive distress — the agent narrows its predictive lookahead when threatened.
3. **Decoupling dynamics from reward.** A γ-model trained once supports value estimation under arbitrary reward functions on the same MDP, which is structurally what the project's interoception ↔ nociception split wants from a representation.

The Janner paper itself is silent on all three threads — it frames γ-models as a generic dynamics-modeling tool for continuous-control RL. The connections above are speculative and would need to be developed in a directions memo, not asserted as findings from this paper.

---

## Table of contents

- [Plain-language entry point](#plain-language-entry-point--what-a-γ-model-is-and-why-a-precision-modulated-agent-might-care)
- [Bibliographic record](#bibliographic-record)
- [Version notes (paper 1 vs. paper 2)](#version-notes-paper-1-vs-paper-2)
- [Phase 1 — Foundational synthesis (undergrad-level)](#phase-1--foundational-synthesis-undergrad-level)
  - [The central question](#the-central-question)
  - [Three takeaways](#three-takeaways)
  - [Initial significance](#initial-significance)
- [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive)
  - [Setup and notation](#setup-and-notation)
  - [The generative TD operator](#the-generative-td-operator)
  - [The two training objectives: $\mathcal{L}_1$ (sample-only, GAN) and $\mathcal{L}_2$ (density-evaluable, flow)](#the-two-training-objectives-mathcall_1-sample-only-gan-and-mathcall_2-density-evaluable-flow)
  - [Proposition 1: γ-models recover the normalized successor representation at the global minimum](#proposition-1-γ-models-recover-the-normalized-successor-representation-at-the-global-minimum)
  - [Theorem 1: rolling out a γ-model and reweighting to a larger discount $\tilde\gamma$](#theorem-1-rolling-out-a-γ-model-and-reweighting-to-a-larger-discount-tildegamma)
  - [Theorem 2: the γ-MVE estimator (hybrid model-based / model-free value expansion)](#theorem-2-the-γ-mve-estimator-hybrid-model-based--model-free-value-expansion)
  - [The training-time vs. test-time compounding-error tradeoff](#the-training-time-vs-test-time-compounding-error-tradeoff)
  - [Stability machinery (target networks, single-step density model)](#stability-machinery-target-networks-single-step-density-model)
  - [Empirical findings](#empirical-findings)
  - [Limitations the paper itself flags](#limitations-the-paper-itself-flags)
- [Connections to the rest of the Gamma/ corpus](#connections-to-the-rest-of-the-gamma-corpus)
- [Open questions for follow-up](#open-questions-for-follow-up)
- [Appendix — Section-by-section backbone](#appendix--section-by-section-backbone)

---

## Bibliographic record

**Authors:** Michael Janner (UC Berkeley), Igor Mordatch (Google Brain), Sergey Levine (UC Berkeley + Google Brain).
**Venue:** Advances in Neural Information Processing Systems 33 (NeurIPS 2020), Vancouver.
**arXiv:** 2010.14496 (v1 Oct 2020; v4 Nov 2021).
**Project page / code:** released by the first author at Berkeley (γ-model GAN + neural-spline-flow code on GitHub at the time of publication).

**Source PDFs in this repository:**
- Paper 1: `docs/project/references/Gamma/sources/Janner et al. 2020 - Generative temporal difference learning for infinite-horizon prediction.pdf` (17 pages, arXiv v4, Nov 2021 — extended version with full appendices).
- Paper 2: `docs/project/references/Gamma/sources/Janner et al. 2020 -models - Generative temporal difference learning for infinite-horizon prediction.pdf` (12 pages, NeurIPS camera-ready, Jan 2021 — shorter version with Broader Impact, appendices absent from this extracted form).

---

## Version notes (paper 1 vs. paper 2)

The two PDFs are the **same paper in two versions**. The core method, equations, theorems, proofs, algorithms, experiments, and figures are identical between them. Only three differences are substantive:

| Aspect | Paper 1 (17 pp., arXiv v4) | Paper 2 (12 pp., NeurIPS camera) |
|---|---|---|
| Title | "Generative Temporal Difference Learning for Infinite-Horizon Prediction" | "γ-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction" (prepends "γ-Models:") |
| Compounding-error citation in intro | Talvitie (2014) | Janner et al. (2019) — i.e., MBPO, the first author's own earlier work |
| Section structure | No "Broader Impact" section; **full Appendices A–E** (rollout-weight derivation, γ-MVE derivation + actor-critic pseudocode, implementation/hyperparameter tables, environment details, GAN visualizations) | "Broader Impact" section present (NeurIPS 2020 camera-ready requirement at the time); **appendices absent from the extracted form** of this PDF |

There are also minor cosmetic differences — paper 2 uses `⇡` and `✓` glyphs (Computer Modern vs. cmsy) for $\pi$ and $\theta$ that render slightly differently in the OCR. These do not affect mathematical content.

**Practical rule for downstream citation in project docs:** cite the NeurIPS 2020 conference paper (paper 2's frame) for the publication record; cite paper 1 (the arXiv extended version) when the appendix derivations matter — the rollout-reweighting and γ-MVE proofs are only in paper 1's extracted form, and the actor-critic pseudocode in paper 1's Algorithm 3 is the cleanest plug-in template if the project ever wants to drop γ-MVE into a SAC-style critic.

---

## Phase 1 — Foundational synthesis (undergrad-level)

### The central question

> *Can we build a learned dynamics model whose prediction horizon is infinite — so we never have to roll it out step-by-step at test time — but that we can still train from ordinary one-step transition data?*

Standard model-based RL trains $p_\theta(s_{t+1} \mid s_t, a_t)$ by maximum likelihood on observed transitions $(s_t, a_t, s_{t+1})$, then iterates the model autoregressively to plan ahead. The cost is compounding error: $p_\theta$ is never quite right, so feeding it its own outputs amplifies the error exponentially in rollout depth.

A value function $V^\pi(s)$ already has an infinite horizon by virtue of geometric discounting, but it conflates reward structure with dynamics — change the reward function and you have to retrain. The γ-model is the answer: an infinite-horizon, *reward-independent* predictor of where the agent will be over the whole discounted future.

### Three takeaways

1. **The discounted state-occupancy distribution is the right object to model.** It is the dynamics-only counterpart to the value function. Just as $V^\pi$ summarizes the future via *expected reward*, $\mu_\gamma^\pi$ summarizes the future via *probability of state visitation*, weighted geometrically. The value of any reward function pops out by Monte-Carlo integration against samples from $\mu_\gamma^\pi$: $V^\pi(s) = \frac{1}{1-\gamma}\, \mathbb{E}_{s_e \sim \mu_\gamma^\pi(\cdot \mid s)}[r(s_e)]$. No rollout required.
2. **TD-style bootstrap turns it into a tractable training problem.** Because $\mu_\gamma^\pi$ has a self-referential Bellman-like recursion (next-step transition + γ-weighted next-state occupancy), it can be trained off-policy on one-step transitions, just like a Q-function. The novelty is that the target is now a *distribution* over states, not a scalar return. Two natural training objectives follow: an f-divergence (which becomes a GAN setup) and a log-density regression (which fits a normalizing flow).
3. **The γ-model is a continuous, sample-able successor representation.** Dayan's tabular successor representation $M^\pi(s, s')$ is exactly $\mu_\gamma^\pi$ scaled by $1/(1-\gamma)$. In tabular settings this scaling normalizes the SR into a distribution; in continuous settings the SR cannot be normalized directly, but the γ-model *is* normalizable by construction (it's a generative model). This gives the SR a continuous, deep-net realization that supports sampling, not just feature expectations.

### Initial significance

The paper's main contribution is **conceptual unification**, not raw benchmark wins. The γ-model demonstrates that:

- TD learning is not specific to scalar value functions — it is a generic bootstrap operator that applies to any quantity satisfying a discounted recurrence, *including distributions*.
- Sequential rollouts and bootstrapped training are two sides of one error budget. Standard single-step models pay all of their compounding error at *test time* (during rollouts); a TD-style infinite-horizon model pays it at *training time* (during bootstrap). γ-models give the practitioner a continuous knob (the model's training discount $\gamma$) to slide between these regimes.
- Multi-horizon control is achievable from a *single network*. Train one γ-model at some $\gamma$, then reweight its rollout steps analytically to match any target $\tilde\gamma \ge \gamma$. The familiar MBPO-style rollout (γ = 0) and the one-shot value estimate (γ = $\tilde\gamma$) are the two extremes of this same family.

Empirically: γ-MVE — γ-models plugged in as the value-expansion target inside SAC — matches MBPO on sample efficiency and retains SAC's asymptotic performance across four low-dimensional benchmark tasks (Acrobot, Mountain Car, Pendulum, Reacher), but the experiments are explicitly low-dimensional and the authors flag scaling to image-based / high-dimensional tasks as open.

---

## Phase 2 — Graduate-level deep dive

### Setup and notation

Consider an infinite-horizon MDP $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ with continuous state space, one-step transition kernel $p(s' \mid s, a)$, reward $r: \mathcal{S} \to \mathbb{R}$, and discount $\gamma \in (0, 1)$. A stationary policy $\pi(a \mid s)$ induces the **discounted state occupancy**

$$
\mu_\gamma^\pi(s \mid s_t, a_t) \;=\; (1-\gamma) \sum_{\Delta t = 1}^{\infty} \gamma^{\,\Delta t - 1}\, p\!\bigl(s_{t+\Delta t} = s \,\big|\, s_t, a_t, \pi\bigr). \tag{1}
$$

The $(1-\gamma)$ prefactor is the normalization that makes $\mu_\gamma^\pi$ a proper probability distribution over $s$ (it's the geometric series' inverse). Janner et al. write $\mu(s \mid s_t, a_t)$ when the discount is fixed, and $\mu(s_e \mid s_t; \gamma)$ when multiple discounts are in play; we follow that convention. The parametric γ-model is $\mu_\theta$.

A useful reframing: $\mu_\gamma^\pi$ is exactly the distribution of the **exit state** in a modified MDP where, at every timestep, the trajectory terminates with probability $1 - \gamma$. Sampling from $\mu_\gamma^\pi$ is then "run policy $\pi$, flip a coin with bias $1 - \gamma$ at every step, return the current state when the coin says terminate." The termination time $\Delta t \sim \mathrm{Geom}(1-\gamma)$.

### The generative TD operator

Sampling from $\mu_\gamma^\pi$ via the exit-state interpretation requires on-policy rollouts and is policy-specific. The key move is to split the geometric-mixture sum into "terminate at the next step" + "don't terminate, continue":

$$
p_{\text{targ}}(s_e \mid s_t, a_t)
\;=\; \underbrace{(1-\gamma)\, p(s_e \mid s_t, a_t)}_{\text{single-step distribution}}
\;+\; \underbrace{\gamma\, \mathbb{E}_{s_{t+1} \sim p(\cdot \mid s_t, a_t)}\!\bigl[\mu_\theta(s_e \mid s_{t+1})\bigr]}_{\text{model bootstrap}}, \tag{2}
$$

with the shorthand $\mu_\theta(s_e \mid s_{t+1}) := \mathbb{E}_{a_{t+1} \sim \pi(\cdot \mid s_{t+1})}[\mu_\theta(s_e \mid s_{t+1}, a_{t+1})]$ folding the next action over the policy.

**Verifying that (2) is the fixed point of the discounted occupancy.** Substituting (1) into the right-hand side of (2):

$$
\begin{aligned}
(1-\gamma)\, p(s_e \mid s_t, a_t) &+ \gamma\, \mathbb{E}_{s_{t+1}}\!\Bigl[(1-\gamma) \sum_{\Delta t = 1}^\infty \gamma^{\Delta t - 1}\, p(s_{t+1+\Delta t} = s_e \mid s_{t+1}, \pi)\Bigr] \\
&= (1-\gamma) \sum_{\Delta t = 1}^\infty \gamma^{\,\Delta t - 1}\, p\!\bigl(s_{t+\Delta t} = s_e \mid s_t, a_t, \pi\bigr) \;=\; \mu_\gamma^\pi(s_e \mid s_t, a_t),
\end{aligned}
$$

where the last equality re-indexes the inner sum by $\Delta t' = \Delta t + 1$ and absorbs the leading single-step term. So $\mu_\gamma^\pi$ is exactly the fixed point of the operator (2) — this is the TD-learning analogue of the Bellman equation, with a *distribution* in place of a scalar return.

**Why this is "TD on distributions" rather than "TD on values".** In standard TD,

$$
Q^\pi(s_t, a_t) \;=\; r(s_t) \;+\; \gamma\, \mathbb{E}_{s_{t+1}, a_{t+1}}\!\bigl[Q^\pi(s_{t+1}, a_{t+1})\bigr].
$$

The structural correspondence with (2) is exact under the substitutions: $Q^\pi \leftrightarrow \mu_\theta(\cdot \mid s_t, a_t)$, $V^\pi(s_{t+1}) \leftrightarrow \mu_\theta(\cdot \mid s_{t+1})$, $r(s_t) \leftrightarrow (1-\gamma)\, p(\cdot \mid s_t, a_t)$. Reward becomes a single-step *distribution*; the bootstrapped value becomes a bootstrapped *distribution*; the recursion is the same.

### The two training objectives: $\mathcal{L}_1$ (sample-only, GAN) and $\mathcal{L}_2$ (density-evaluable, flow)

Because the target (2) is itself a distribution, the training problem is to match $\mu_\theta(\cdot \mid s_t, a_t)$ to the mixture $(1-\gamma)\, p(\cdot \mid s_t, a_t) + \gamma\, \mathbb{E}[\mu_\theta(\cdot \mid s_{t+1})]$. Two paths:

**(1) Sample-only — f-divergence / GAN ($\mathcal{L}_1$).**

$$
\mathcal{L}_1(s_t, a_t, s_{t+1}) \;=\; D_f\Bigl(\mu_\theta(\cdot \mid s_t, a_t) \;\Big\|\; (1-\gamma)\, p(\cdot \mid s_t, a_t) \;+\; \gamma\, \mu_\theta(\cdot \mid s_{t+1})\Bigr). \tag{3}
$$

With Jensen-Shannon as $D_f$ and an auxiliary discriminator $D_\phi$, this becomes the standard saddle-point GAN objective

$$
\hat{\mathcal{L}}_1(s_t, a_t) \;=\; \mathbb{E}_{s_e^+ \sim p_{\text{targ}}(\cdot \mid s_t, a_t)}\!\bigl[\log D_\phi(s_e^+ \mid s_t, a_t)\bigr] \;+\; \mathbb{E}_{s_e^- \sim \mu_\theta(\cdot \mid s_t, a_t)}\!\bigl[\log(1 - D_\phi(s_e^- \mid s_t, a_t))\bigr],
$$

minimized over $\theta$ and maximized over $\phi$. Sampling from $p_{\text{targ}}$ is straightforward: with probability $(1-\gamma)$ draw $s_e^+ = s_{t+1}$ from the empirical transition, with probability $\gamma$ draw $s_e^+ \sim \mu_{\bar\theta}(\cdot \mid s_{t+1}, a_{t+1})$ from the *target* γ-model (see Stability below).

**(2) Density-evaluable — log-density regression / normalizing flow ($\mathcal{L}_2$).**

$$
\mathcal{L}_2(s_t, a_t, s_{t+1}) \;=\; \mathbb{E}_{s_e}\!\Bigl[\tfrac{1}{2}\bigl\|\log \mu_\theta(s_e \mid s_t, a_t) - \log\bigl((1-\gamma) p(s_e \mid s_t, a_t) + \gamma\, \mu_\theta(s_e \mid s_{t+1})\bigr)\bigr\|_2^2\Bigr]. \tag{4}
$$

Two subtleties for $\mathcal{L}_2$:

- The log target requires evaluating both $\mu_\theta$ *and* the single-step density $p$. Janner et al. take the cheap option: approximate $p(\cdot \mid s_t, a_t) \approx \mathcal{N}(s_{t+1}, \sigma^2 I)$ on-the-fly from the observed transition, with $\sigma^2 = 10^{-2}$ a fixed hyperparameter (Table 2 in paper 1's Appendix C). For deterministic dynamics this is unbiased; for stochastic dynamics this is a bound (Jensen's inequality on the log of an expectation).
- The flow is trained against a *log-mixture* target with a stop-gradient on the target (delayed network $\bar\theta$).

**Choice tradeoff.** Flows are more stable and were the authors' preferred parameterization in the experiments — particularly at high $\gamma$ where the GAN's bootstrap accumulates more variance and the discriminator's adversarial pressure can collapse modes (Appendix E of paper 1 shows the GAN visualizations degrading at $\gamma = 0.95$ on Mountain Car). The GAN remains useful when density evaluation is structurally impossible.

### Proposition 1: γ-models recover the normalized successor representation at the global minimum

**Statement.** The global minimum of both $\mathcal{L}_1$ and $\mathcal{L}_2$ is achieved iff

$$
\mu_\theta(s_e \mid s_t, a_t) \;=\; (1-\gamma)\, M(s_e \mid s_t, a_t),
$$

where $M$ is the successor representation defined by the tabular recurrence

$$
M(s_e \mid s_t, a_t) \;=\; \mathbb{E}_{s_{t+1} \sim p(\cdot \mid s_t, a_t)}\!\bigl[\mathbb{1}[s_e = s_{t+1}] \;+\; \gamma\, M(s_e \mid s_{t+1})\bigr]. \tag{5}
$$

**Proof sketch.** Both objectives are minimized when their bootstrap residual vanishes:

$$
\mu_\theta(s_e \mid s_t, a_t) \;=\; (1-\gamma)\, p(s_e \mid s_t, a_t) \;+\; \gamma\, \mathbb{E}_{s_{t+1} \sim p(\cdot \mid s_t, a_t)}\!\bigl[\mu_\theta(s_e \mid s_{t+1})\bigr].
$$

This is exactly the SR recurrence (5) multiplied through by $(1-\gamma)$. The $(1-\gamma)$ factor is what normalizes $\mu_\theta$ into a probability distribution over $s_e$: $\int M(s_e \mid s_t, a_t)\, ds_e = \sum_{\Delta t} \gamma^{\Delta t - 1} = 1/(1-\gamma)$, so $(1-\gamma) M$ integrates to 1.

**Why this matters.** Continuous successor representations have historically been forced to give up the "predict a distribution" interpretation: Barreto et al.'s successor *features* (2017, 2018) predict the *expected feature vector* $\psi^\pi(s) = \mathbb{E}^\pi[\sum_t \gamma^t \phi(s_t)]$ for a fixed featurizer $\phi$, not a distribution over next states. The γ-model recovers the original SR's predict-where-you'll-be interpretation in continuous spaces, by paying for it with a generative model (GAN or flow) rather than a featurizer. This is the cleanest available bridge from the cognitive-neuroscience SR literature (Dayan 1993; Momennejad et al. 2017; Gershman 2018) to deep-network continuous-state predictors.

### Theorem 1: rolling out a γ-model and reweighting to a larger discount $\tilde\gamma$

The γ-model trained at discount $\gamma$ can be rolled out auto-regressively (sample $s^{(1)} \sim \mu_\theta(\cdot \mid s_t)$, then $s^{(2)} \sim \mu_\theta(\cdot \mid s^{(1)})$, etc.). The marginal time at step $n$ is the sum of $n$ iid Geom$(1-\gamma)$ random variables, which is NegativeBinomial$(n, 1-\gamma)$.

**Statement.** For any target discount $\tilde\gamma \in [\gamma, 1)$, the reweighting

$$
\alpha_n \;=\; \frac{(1-\tilde\gamma)(\tilde\gamma - \gamma)^{n-1}}{(1-\gamma)^n}
$$

reproduces a $\tilde\gamma$-model from the $\gamma$-model rollout:

$$
\mu(s_e \mid s_t; \tilde\gamma) \;=\; \sum_{n=1}^\infty \alpha_n\, \mu^{(n)}(s_e \mid s_t; \gamma).
$$

**Proof structure (Appendix A of paper 1).** Induction on $n$. The pmf of the time at the $n$-th γ-model step is the negative-binomial $p_n(t) = \binom{t-1}{t-n}\gamma^{t-n}(1-\gamma)^n$, supported on $t \ge n$ (this is a slightly non-textbook form chosen so the support is $t \ge n$, simplifying the inductive bookkeeping). The induction sets $q(t) = (1-\tilde\gamma)\tilde\gamma^{t-1}$ (a Geom$(1-\tilde\gamma)$) as the target and solves $q(t) = \sum_{n \le t} \alpha_n\, p_n(t)$ for the $\alpha_n$.

- **Base case** ($n = 1$): $p_1$ is the only component supported at $t = 1$, so $\alpha_1 = (1-\tilde\gamma)/(1-\gamma)$.
- **Inductive step**: assume the formula for $\alpha_1, \dots, \alpha_{n-1}$, set $q(n) = (1-\tilde\gamma)\tilde\gamma^{n-1}$, isolate $\alpha_n$. The bookkeeping is laid out in paper 1 Appendix A and resolves cleanly because the geometric / negative-binomial sums telescope.

**Two special cases of interest.**
- **$\gamma = 0$** (standard single-step model): $\alpha_n = (1-\tilde\gamma)\tilde\gamma^{n-1}$. These are exactly the geometric-mixture weights of equation (1) — recovering the textbook one-step-rollout reweighting.
- **$\gamma = \tilde\gamma$**: $\alpha_n = 0^{n-1}$, i.e., $\alpha_1 = 1$ and $\alpha_{n \ge 2} = 0$. The γ-model already targets $\tilde\gamma$, so a single feedforward pass suffices.

**The rollout-length curve.** Figure 2(b) of the paper plots how many γ-model steps are needed to recover 95% of the probability mass under a target $\tilde\gamma$ across model discounts $\gamma$. At $\gamma = 0$, recovering $\tilde\gamma = 0.99$ takes **299 model steps** (this is just $\log(0.05) / \log(0.99) \approx 299$). At $\gamma = 0.95$, $\tilde\gamma = 0.99$ takes ~58 steps. The shape of the family is roughly $\propto 1/(1-\gamma)$ scaled by the residual gap $(\tilde\gamma - \gamma)$.

### Theorem 2: the γ-MVE estimator (hybrid model-based / model-free value expansion)

Standard model-based value expansion (MVE, Feinberg et al. 2018) plugs a short single-step model rollout into a terminal value function:

$$
V_{\text{MVE}}(s_t; \tilde\gamma) \;=\; \sum_{n=1}^H \tilde\gamma^{n-1}\, r(s_{t+n}) \;+\; \tilde\gamma^H\, V(s_{t+H}; \tilde\gamma).
$$

The hard horizon $H$ is a discrete cutoff between model-based and model-free regimes.

**Statement (γ-MVE).** Replace the single-step model with a γ-model and reweight via Theorem 1:

$$
\hat V_{\gamma\text{-MVE}}(s_t; \tilde\gamma) \;=\; \frac{1}{1-\tilde\gamma} \sum_{n=1}^H \alpha_n\, \mathbb{E}_{s_e \sim \mu^{(n)}(\cdot \mid s_t; \gamma)}[r(s_e)] \;+\; \Bigl(\tfrac{\tilde\gamma - \gamma}{1-\gamma}\Bigr)^H \mathbb{E}_{s_e \sim \mu^{(H)}(\cdot \mid s_t; \gamma)}\!\bigl[V(s_e; \tilde\gamma)\bigr].
$$

**Proof structure (Appendix B of paper 1).** Start from $V(s_t; \tilde\gamma) = \frac{1}{1-\tilde\gamma}\, \mathbb{E}_{s_e \sim \mu(\cdot \mid s_t; \tilde\gamma)}[r(s_e)]$ (equation 6 in the paper, which is the dynamics-only restatement of value: integrate reward against the γ-model's distribution and divide out the $(1-\gamma)$ normalizer). Substitute the rollout-reweighted expression from Theorem 1, split the infinite sum into $n \le H$ and $n > H$, and recognize the tail as a discounted expectation over the $H$-step γ-model state, weighted by

$$
\sum_{n=H+1}^\infty \alpha_n \;=\; \Bigl(\tfrac{\tilde\gamma - \gamma}{1-\gamma}\Bigr)^H \quad (\text{Lemma 1 in Appendix B}),
$$

which is the truncation-correction factor.

**Two limits worth checking.**
- **$\gamma = 0$**: $\alpha_n = (1-\tilde\gamma)\tilde\gamma^{n-1}$ and the truncation weight is $\tilde\gamma^H$, recovering standard MVE.
- **$H = 1, \gamma = 0.8, \tilde\gamma = 0.99$** (the configuration used in the SAC experiments): $\alpha_1 = (1-0.99)/(1-0.8) = 0.05$, truncation weight $= (0.19/0.2)^1 = 0.95$. So $\sim 5\%$ of the value mass comes from the one γ-model feedforward pass and $\sim 95\%$ from the terminal value function under one γ-model step. This is mostly a model-free estimator with a probabilistic-horizon "lookahead correction" of width $\sim 1/(1-0.8) = 5$ effective timesteps.

**Pseudocode (Algorithm 3 in paper 1's Appendix C).** γ-MVE drops in as a replacement for the value target inside SAC: keep SAC's Q, V, and π updates, replace the $V(s_{t+1})$ inside the Q-target with $\hat V_{\gamma\text{-MVE}}(s_{t+1})$. The γ-model update follows Algorithm 1 (GAN) or Algorithm 2 (flow) of the main paper.

### The training-time vs. test-time compounding-error tradeoff

This is the conceptual punchline of the paper and is worth stating cleanly.

- **Standard single-step models ($\gamma = 0$)**: trained by clean maximum-likelihood (no bootstrap), so training is stable. But long-horizon predictions require autoregressive rollouts at test time, so error compounds *during inference*.
- **γ-models with $\gamma > 0$**: amortize the long-horizon work into training via TD bootstrap, so inference is one feedforward pass. But the training is now approximate-dynamic-programming-style and inherits the bootstrap-error pathologies of off-policy Q-learning (Kumar et al. 2019).

The γ knob slides between these regimes. There is no free lunch — compounding error in *some* form is structural — but the practitioner gets to choose which side of the budget to pay it on, based on where the relative weak point of their pipeline is.

Paper 1's Figure 2 makes this visually concrete: at low $\gamma$ the rollouts are long but per-step variance is low; at high $\gamma$ the rollouts are short but the training distribution itself is harder to fit.

### Stability machinery (target networks, single-step density model)

Two engineering choices, both lifted from the model-free RL literature:

1. **Delayed target network** $\mu_{\bar\theta}$. The bootstrap target in equation (2) is constructed from a separate set of parameters $\bar\theta$ that lag $\theta$ via exponential moving average $\bar\theta \leftarrow \tau \theta + (1-\tau)\bar\theta$. Paper 1 Algorithm 1, line 9; Algorithm 2, line 9. $\tau = 5 \cdot 10^{-3}$ in both tables. This is the same trick DQN (Mnih et al. 2015) uses to stop the bootstrap target from chasing the model and causing oscillations.
2. **On-the-fly single-step Gaussian model** $p_\theta(s_e \mid s_t, a_t) \approx \mathcal{N}(s_{t+1}, \sigma^2 I)$ for the density-evaluable variant. Avoids the need to train and maintain a separate single-step model just to compute log-densities of the target's first mixture component. $\sigma^2 = 10^{-2}$.

### Empirical findings

**Prediction (Figure 3, paper 1 + paper 2).** Flow-based γ-models trained at $\gamma \in \{0, 0.5, 0.75, 0.85, 0.95\}$ on Acrobot and Pendulum. As $\gamma$ rises, the single-feedforward-pass predicted distribution visibly fans out to cover the agent's longer-horizon trajectory under the SAC policy. Ground truth ($\gamma = 0.95$ Monte Carlo from 100 rollouts) matches the flow's prediction qualitatively.

**Value visualization (Figure 4).** γ-model at $\gamma = 0.99$ on Pendulum + per-state reward → value-map estimate via equation (6). The estimated value map closely tracks the value-iteration ground truth on a fine-grained discretization. The fact that this works at $\gamma = 0.99$ in a single feedforward pass is the headline.

**Control (Figure 5).** γ-MVE inside SAC vs. SAC, PPO, MVE, MBPO across Acrobot / Mountain Car / Pendulum / Reacher. γ-MVE is the fastest-converging method on three of four tasks (matching MBPO on Pendulum); SAC retains its asymptotic level. Five seeds per condition, standard-deviation shading.

**GAN vs. Flow stability (Appendix E of paper 1).** Adversarial γ-models work at low-to-moderate $\gamma$ but degrade visibly at $\gamma = 0.95$, especially on Mountain Car where the discriminator's adversarial pressure interacts badly with the bootstrap to produce mode-collapse-like artifacts. The flow is the safer default.

### Limitations the paper itself flags

- All experiments are **low-dimensional, continuous-control** (state space ≤ 11D, no image inputs). The authors explicitly call out that scaling to high-dimensional / image-based tasks is an open challenge, and that the generative-model literature's progress on stable training is a prerequisite.
- The bootstrap is structurally tied to the **off-policy RL stability problem**. The authors cite Kumar et al. (2019) for bootstrap-error accumulation in deep Q-learning and concede that γ-model training inherits the same fragility.
- $\mathcal{L}_2$'s on-the-fly Gaussian single-step model **is biased for stochastic dynamics** (it's only a bound via Jensen's inequality on the log of a mixture). For environments with non-trivial transition noise, training a proper single-step density model alongside the γ-model is recommended but not done in the reported experiments.
- The **policy is treated as fixed** during γ-model training in the experiments. Using γ-models inside a full policy-improvement loop (instead of just inside value estimation against a fixed policy) is plausible but not demonstrated; the off-policy stability issues compound.

---

## Connections to the rest of the Gamma/ corpus

The `Gamma/` reference corpus assembles seven related works on horizon-aware value / dynamics modeling. Cross-paper synthesis is owed a separate curator pass, but the obvious bilateral connections are worth flagging here so a follow-up reviewer doesn't miss them.

- **Borsa et al. 2018 — Universal Successor Features Approximators (USFA).** USFA learns $\psi^\pi(s, w)$, the SR's *feature-expectation* form, parametric in a task vector $w$. γ-models are the *generative* (sampleable-distribution) analogue of the same underlying object that USFA models as a feature-expectation. A direct, formal bridge: $\mathbb{E}_{s_e \sim \mu_\gamma^\pi(\cdot \mid s)}[\phi(s_e)] = (1-\gamma)\, \psi^\pi(s)$ for any featurizer $\phi$ giving rise to the successor feature.
- **Sherstan et al. 2020 — γ-nets.** Networks parametric in $\gamma$ that output the value (or any GVF) at any queried discount — multi-horizon prediction via *input conditioning on $\gamma$*. γ-models share the multi-horizon goal but solve it differently: a γ-model trained at one $\gamma$ reweights *analytically* to any $\tilde\gamma \ge \gamma$ via Theorem 1; γ-nets train one network to *evaluate* multiple discounts. These are complementary mechanisms — γ-nets condition on $\gamma$, γ-models reweight from $\gamma$.
- **Fedus et al. 2019 — Hyperbolic discounting.** Builds value estimates that integrate over a *distribution* of discount factors, recovering hyperbolic discounting as a mixture of exponentials. The reweighting machinery of Theorem 1 is exactly the integration-over-discounts ingredient that hyperbolic / non-exponential value functions need. Note that the user has invited this connection to be developed in a separate professor-rl memo.
- **Romoff et al. 2019 — Separating value functions across timescales.** Decomposes $V$ into a sum of incremental contributions at different timescales. γ-models could replace each per-timescale value head with a sample-from-occupancy head; the timescale separation in Romoff lives at the level of value, not dynamics.
- **Schaul et al. 2015 — UVFA.** Universal value function approximators parametric in the *goal*. γ-models are reward-independent, so the UVFA construction (one network, many goals) trivially applies: the same γ-model serves any reward function on the MDP.
- **Schultheis et al. 2022 — RL with non-exponential discounting.** A theoretical companion to Fedus 2019; could share the reweighting machinery.

---

## Open questions for follow-up

(Questions for downstream agents, not findings from this paper.)

1. **Can γ be modulated by an interoceptive precision signal?** The γ-model machinery treats $\gamma$ as a fixed training hyperparameter. Theorem 1 says rollout-reweighting can promote a trained-at-$\gamma$ model to any $\tilde\gamma \ge \gamma$. Two paths to a modulated agent: (a) train one γ-model at a low $\gamma_{\min}$ and let a downstream controller pick a context-dependent $\tilde\gamma$ at inference (cheap, but capped above by training-time noise); (b) train several γ-models at different $\gamma$ values and let a FiLM / hypernet head modulate the read-out. Both are speculative; the paper does no modulation experiment.
2. **Does the single-feedforward-pass value estimate scale to recurrent agents?** All experiments use feedforward γ-models on MDPs. Whether the same approach works on the project's POMDP / partially-observed environments (where the state $s_t$ is itself a learned belief) is an open question the paper doesn't address.
3. **Is the γ-model the right object for "pain-shortens-horizon" claims?** A precision-modulated controller that tightens its effective $\tilde\gamma$ under nociceptive distress would, in the γ-model framework, narrow which rollout-reweighting weights it integrates against. Whether this gives the "behavioral myopia" the project is trying to model is a directions-memo-level question for `professor-pain-modeling`, not a finding here.
4. **Bayesian γ-models?** The paper trains a single point-estimate γ-model. A Bayesian / ensemble γ-model would carry epistemic uncertainty in the predicted occupancy. Whether epistemic uncertainty on $\mu_\gamma^\pi$ has the same disambiguation power as epistemic uncertainty on $Q$ is a question for `professor-bayesian-nn`.

These are pointers; the literature-reviewer does not pursue them. Recommendation: a follow-up `research-postdoc` or `professor-rl` memo could develop (1) and (3); `professor-bayesian-nn` could develop (4).

---

## Appendix — Section-by-section backbone

This is the literal section ordering of the paper, with the core claim of each section captured in 1–3 lines. Retained as a traceability anchor under the merged Phase 1/2 synthesis above.

**§1 Introduction.** Single-step dynamics models force a horizon choice; long horizons compound prediction error. Value functions sidestep horizon via discounting but entangle dynamics with reward. Proposes γ-models: discount-parameterized predictive models trained with a generative reinterpretation of TD learning. Three advantages flagged: constant-time long-horizon prediction; generalized rollouts and value estimation; omission of unnecessary timestep information.

**§2 Related Work.** Surveys model-free / model-based hybrids (initialization, model-augmented data, value-target sharpening, gradient-based model use, planning-without-prediction). Distinguishes from TDMs (Pong et al. 2018, fixed-horizon goal-conditioned), successor representation (Dayan 1993, tabular only), continuous SR variants (Barreto et al. 2017, expected features only), β-models (Sutton 1995, arbitrary mixture weights), option models (Sutton et al. 1999, state-dependent termination).

**§3 Preliminaries.** Infinite-horizon MDP $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$. Defines the discounted state occupancy $\mu(s \mid s_t, a_t) = (1-\gamma) \sum_{\Delta t = 1}^\infty \gamma^{\Delta t - 1} p(s_{t+\Delta t} = s \mid s_t, a_t, \pi)$ (equation 1).

**§4 Generative Temporal Difference Learning.** Introduces the bootstrapped target distribution $p_{\text{targ}}(s_e \mid s_t, a_t) = (1-\gamma) p(s_e \mid s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}}[\mu_\theta(s_e \mid s_{t+1})]$ (equation 2). States the structural analogy to scalar TD: $\mu_\theta(\cdot \mid s_t, a_t) \leftrightarrow Q$, $\mu_\theta(\cdot \mid s_{t+1}) \leftrightarrow V$, $p(s_e \mid s_t, a_t) \leftrightarrow r$. Gives the two training losses $\mathcal{L}_1$ (f-divergence, eq. 3) for sample-only models and $\mathcal{L}_2$ (log-density regression, eq. 4) for density-evaluable models.

**§5 Analysis and Applications of γ-Models.**
- **§5.1 γ-Models as a Continuous Successor Representation.** Proposition 1: at the global minimum of $\mathcal{L}_1$ or $\mathcal{L}_2$, $\mu_\theta = (1-\gamma) M$ where $M$ is the SR. The $(1-\gamma)$ factor normalizes the SR into a distribution.
- **§5.2 γ-Model Rollouts.** Theorem 1: a $\gamma$-model rollout, reweighted by $\alpha_n = (1-\tilde\gamma)(\tilde\gamma - \gamma)^{n-1} / (1-\gamma)^n$, recovers a $\tilde\gamma$-model for any $\tilde\gamma \in [\gamma, 1)$. Special cases: $\gamma = 0$ recovers the standard single-step weighting; $\gamma = \tilde\gamma$ recovers a single feedforward pass. Figure 2 quantifies rollout length needed to cover 95% of mass.
- **§5.3 γ-Model-Based Value Expansion.** Equation 6: $Q(s_t, a_t; \gamma) = \frac{1}{1-\gamma} \mathbb{E}_{s_e \sim \mu(\cdot \mid s_t, a_t; \gamma)}[r(s_e)]$ — values as expectations over a γ-model in one feedforward pass. γ-MVE: $\hat V_{\gamma\text{-MVE}}(s_t; \tilde\gamma) = \frac{1}{1-\tilde\gamma} \sum_{n=1}^H \alpha_n \mathbb{E}_{s_e \sim \mu^{(n)}(\cdot \mid s_t; \gamma)}[r(s_e)] + ((\tilde\gamma-\gamma)/(1-\gamma))^H \mathbb{E}_{s_e \sim \mu^{(H)}(\cdot \mid s_t; \gamma)}[V(s_e; \tilde\gamma)]$. MVE = $\gamma = 0$ special case. Single-shot Q estimate = $H = 0$ special case.

**§6 Practical Training of γ-Models.** Two algorithms. Algorithm 1 (without density evaluation): saddle-point GAN with delayed target $\bar\theta$. Algorithm 2 (with density evaluation): log-density regression with on-the-fly Gaussian single-step model $p \approx \mathcal{N}(s_{t+1}, \sigma^2 I)$ and delayed target. Both use exponential-moving-average target updates with $\tau = 5 \cdot 10^{-3}$.

**§7 Experiments.**
- **§7.1 Prediction.** Visualizes γ-model predictions on Acrobot and Pendulum across $\gamma \in \{0, 0.5, 0.75, 0.85, 0.95\}$ (Figure 3). At $\gamma = 0.99$ on Pendulum, computes values by integrating reward against the γ-model's single-feedforward-pass output (Figure 4); matches the ground-truth value-iteration map.
- **§7.2 Control.** γ-MVE inside SAC vs. SAC / PPO / MBPO / MVE on Acrobot, Mountain Car, Pendulum, Reacher. Model discount $\gamma = 0.8$, value discount $\tilde\gamma = 0.99$, $H = 1$. γ-MVE matches MBPO's sample efficiency and SAC's asymptote on all four (Figure 5).

**§8 Discussion, Limitations, and Future Work.** γ-models are a hybrid model-free / model-based mechanism: policy-conditioned and infinite-horizon like a value function, reward-independent like a single-step model. Experiments are low-dimensional; scaling to high-dimensional / image-based tasks is open. Future progress depends on continued improvement in generative-model training and off-policy RL stability.

**Broader Impact (paper 2 only).** γ-models offer a path to long-term modeling without test-time compounding error, at the cost of converting a maximum-likelihood problem into an approximate-dynamic-programming one — and the latter is substantially less well-understood with rigorous guarantees. Real-world safety-critical deployment will require sustained work on both generative-model training and dynamic-programming methods.

**Appendix A (paper 1 only): Derivation of γ-model-based rollout weights.** Induction proof of Theorem 1, with the non-textbook NegativeBinomial pmf form $p_n(t) = \binom{t-1}{t-n}\gamma^{t-n}(1-\gamma)^n$ chosen so support is $t \ge n$ rather than $t \ge 0$.

**Appendix B (paper 1 only): Derivation of γ-MVE.** Lemma 1: $1 - \sum_{n=1}^H \alpha_n = ((\tilde\gamma - \gamma)/(1-\gamma))^H$ — the truncation-correction factor. Theorem 2: the γ-MVE estimator (full statement and step-by-step derivation, including the time-invariance argument for the tail).

**Appendix C (paper 1 only): Implementation Details.** Algorithm 3 (γ-MVE inside SAC's actor-critic update). Hyperparameter Tables 1 (GAN: batch 128, 512 $s_e$ samples per $(s_t, a_t)$, $\tau = 5 \cdot 10^{-3}$, lr $10^{-4}$, replay $2 \cdot 10^5$) and 2 (Flow: batch 1024, 1 sample per $(s_t, a_t)$, $\sigma^2 = 10^{-2}$). Architectures: 2-layer MLPs (256 hidden, leaky-ReLU) for the GAN's generator and discriminator; 6-layer neural spline flow (16 knots on $[-10, 10]$, 3-layer MLP coupling network with 256 hidden) for the flow. GAN losses tested: standard GAN, least-squares GAN — both work.

**Appendix D (paper 1 only): Environment Details.** Acrobot-v1 (8-D obs, modified to 1-D continuous actions, shaped reward $-\cos\theta_0 - \cos(\theta_0 + \theta_1)$); MountainCarContinuous-v0 (2-D obs, shaped reward $= x$); Pendulum-v0 (3-D obs); Reacher-v2 (11-D obs, end-effector to goal). Shaped rewards available to model-free baselines for fair comparison.

**Appendix E (paper 1 only): Adversarial γ-Model Predictions.** Figure 6: GAN γ-model visualizations on Acrobot and Mountain Car at the same $\gamma$ values as Figure 3. GAN is competitive at low $\gamma$ but degrades visibly at $\gamma = 0.95$ on Mountain Car.

---

**End of Module C review.** Sister modules (Borsa USFA, Fedus hyperbolic, Romoff multi-timescale, Schaul UVFA, Schultheis non-exponential, Sherstan γ-nets) live as separate `Gamma_lit_review_*.md` files alongside this one; cross-paper synthesis is owed a follow-up `literature-curator` pass.
