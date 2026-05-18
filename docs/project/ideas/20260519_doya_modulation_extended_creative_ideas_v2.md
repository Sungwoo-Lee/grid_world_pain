---
title: "Doya's four neuromodulator-hyperparameter assignments — extended creative implementation ideas via FiLM and hypernetwork variants (v2)"
status: draft
authors:
  - "professor-dl-theory (NA, ACh sections)"
  - "professor-rl (DA, 5-HT sections)"
polished_by:
  - "research-postdoc (2026-05-19) — assembly and cross-cutting framing"
audience: user + pi + research-postdoc
date: 2026-05-19
last_updated: 2026-05-19
companion_to:
  - "docs/project/ideas/20260518_doya_modulation_via_film_hypernet_implementation_ideas.md"
references_integrated:
  - "docs/project/references/Temperature/ (4 reviews + sources/)"
  - "docs/project/references/Learning_Rate/ (4 reviews + sources/)"
  - "docs/project/references/TD/ (4 reviews + sources/)"
  - "docs/project/references/Gamma/ (4 reviews + sources/)"
one_line_summary: "Per-knob deep dive on Doya's four neuromodulator-hyperparameter assignments under the two-headed FiLM + hypernetwork architecture — each of NA, ACh, DA, 5-HT gets a TL;DR, background, three implementation candidates of increasing creativity, and a recommended first prototype; the cross-cutting synthesis argues that the richest design per knob outputs a vector or a function from Head 2 rather than a scalar, and that all four knobs can in principle share a single modulator trunk."
---

## 1. Plain-language entry point

Doya (2002) proposed a famously compact picture: four chemical systems in the brain each set one knob of a reinforcement-learning (RL) algorithm. **Noradrenaline (NA)** controls the agent's *policy temperature* — how greedy versus exploratory it is, a single dial between "always pick the best-scoring action" and "spread probability widely and keep trying things." **Acetylcholine (ACh)** controls the *learning rate* — how big a correction the agent makes after each experience. **Dopamine (DA)** controls the *gain on the reward-prediction error* — how strongly a surprise pushes the policy. **Serotonin (5-HT)** controls the *time-discount* — how heavily the agent weighs reward later versus reward now, equivalently how far ahead it "sees."

The companion v1 compact memo (`docs/project/ideas/20260518_doya_modulation_via_film_hypernet_implementation_ideas.md`) worked out the *minimum-viable* implementation of each of these four knobs under a particular architecture the project's prior audit converged on: a **two-headed modulator**. A small network ingests interoceptive context $c$ and emits two streams. **Head 1** emits **FiLM** parameters — a per-channel multiplicative scale and additive shift applied to another network's hidden activations; FiLM (feature-wise linear modulation, Perez et al. 2017) is the standard primitive for letting one network steer another's forward pass. **Head 2** emits scalar coefficients into the *loss* and the *return target* — the parts of the computation where the optimizer acts. Head 1 reaches everything in the forward pass; Head 2 reaches everything in the loss / Bellman backup.

This memo is the **creative companion** to v1. The v1 doc covered the boring-but-honest version of each knob; this one (v2) opens the design space. For each of NA / ACh / DA / 5-HT, four professor-authored sections lay out a TL;DR, background, three implementation candidates of increasing creativity, and a "what I'd try first" recommendation. The sneak-preview of what each section claims: **NA** — beyond the trivial scalar inverse-temperature, the modulator can emit a state-dependent temperature that preserves the Bellman contraction property (mellowmax) or that controls how strongly the agent leans on its own Q-history (Munchausen). **ACh** — FiLM's scale is *secretly* a per-pathway learning-rate gate via the chain rule, so naïve "ACh is α" generalizes to a much richer gating-and-routing operator. **DA** — a scalar gain on the reward-prediction error is gauge-equivalent to the global learning rate (a weak identifiability claim); upgrading Head 2 to emit a vector of reward weights, or a risk-quantile selector, or full critic weights via a hypernetwork breaks the gauge cleanly. **5-HT** — the discount can be conditioned in three architecturally distinct ways: as input to a Universal Value Function Approximator (UVFA), as weights over a basis of pre-trained horizon-specific value heads (hyperbolic discounting), or as the conditioning input to a generative model of the discounted future (γ-Models).

## 2. How to read this memo

This memo has four topic sections (NA, ACh, DA, 5-HT — Sections 3–6), each self-contained: TL;DR, background, candidates 1–3, and a "What I'd try first" recommendation. The topics are **independent** — you can read them in any order. Each lit-review citation resolves to one of four folders under `docs/project/references/`:

- **`Temperature/`** — four review files (SAC foundations, SAC variants, softmax operators, KL/Munchausen) plus a `sources/` folder with source PDFs. Anchors Section 3 (NA).
- **`Learning_Rate/`** — four review files (L2O foundations, learning-rate adaptation, meta-gradient RL, gated conditional architectures) plus `sources/`. Anchors Section 4 (ACh).
- **`TD/`** — four review files (distributional RL, successor features, hypernet RL, Decision Transformer) plus `sources/`. Anchors Section 5 (DA).
- **`Gamma/`** — four review files (universal value functions, multi-horizon, generative γ, non-exponential discounting) plus `sources/`. Anchors Section 6 (5-HT).

If you want the **synthesis** rather than the per-knob detail, jump straight to Section 7 — it argues a single architectural-honesty axis runs across all four knobs, and that what becomes possible when all four knobs share one modulator is the original NMN dream. Section 8 lists pointers to the companion v1 compact memo and the underlying audit / story docs that ground the two-headed architecture.

## 3. NA → policy temperature

### TL;DR

Noradrenaline (NA) controls policy temperature — the dial between "act greedily on whichever action looks best right now" and "spread probability widely and keep trying things." Architecturally this is the cleanest of Doya's four knobs: a single positive scalar emitted by Head 1 of the modulator, multiplied into the policy logits just before the softmax. Because the wiring is trivial, the design space *opens up*: the same one-scalar output can be made to embody three very different theoretical commitments. We sketch three candidates of increasing creativity: (1) the **baseline scalar inverse-temperature head**, (2) a **mellowmax-as-FiLM-target head** where the modulator learns to *be* a state-dependent temperature that preserves a contraction property of the Bellman backup, and (3) a **Munchausen-style log-policy bonus with FiLM-conditioned coefficient**, where the modulator's emitted scalar directly tunes how strongly the agent leans on the average of its own past Q-estimates.

### Background — what the knob is and why it matters

**What "policy temperature" means.** Most stochastic policies in deep RL convert a vector of action scores (logits) $z = (z_1, \ldots, z_n)$ into a probability distribution via the softmax $\pi(a_i) = \exp(\beta z_i) / \sum_j \exp(\beta z_j)$. The scalar $\beta$ is the **inverse temperature** (sometimes written $\beta = 1/\tau$). Large $\beta$ collapses the policy onto the best-scoring action (greedy); small $\beta$ flattens it toward uniform (exploratory). One scalar dial controls the entire exploration–exploitation balance.

**Doya 2002's mapping.** Doya proposed that noradrenaline — the brain's "arousal" / "vigilance" transmitter, released from the locus coeruleus in response to surprise or stress — sets this inverse temperature. High NA tone collapses behaviour onto the dominant response (focused, urgent, greedy); low NA tone broadens it (relaxed, exploratory). The mapping is appealing because it puts a one-dimensional brain signal in correspondence with a one-dimensional algorithmic dial.

**Where temperature lives in the RL computation graph.** Temperature acts on the **policy logits in the forward pass**, right at the output, before softmax. Unlike learning rates (which act on gradients) or discount factors (which act on the bootstrap target), temperature is a pure forward-pass scaling. There is no derivative-of-the-optimizer involved, no double-backprop, no Lagrangian.

**Why this opens creative space.** Architectural simplicity is *liberating*. Because plumbing a scalar onto the logits is essentially zero engineering cost, you can afford to ask: what if the modulator's scalar is interpreted not just as "make the policy greedier" but as the solution of a fixed-point equation that guarantees a Bellman contraction (Candidate 2)? Or as the weight on an implicit Q-averaging mechanism the agent applies to its own learning history (Candidate 3)? Both are still "the modulator emits a scalar that lands at the policy head" — but the *meaning* of that scalar changes radically, and so does the experimental story.

### Candidate 1: Basic FiLM-γ-at-logits temperature head (the audit's baseline)

**Algorithm intro (plain language).** This candidate borrows directly from Soft Actor-Critic (SAC) and its temperature-tuning variants. Plain SAC (Haarnoja 2018a, covered in `Temperature_lit_review_A_sac_foundations.md`) trains a stochastic policy that maximises *reward plus a bonus for being random*; the strength of the randomness bonus is the temperature $\alpha$. The follow-up SAC-v2 (Haarnoja 2018b, same lit-review file) makes $\alpha$ self-adjusting: it derives $\alpha$ as the Lagrange multiplier on a constraint "policy entropy must stay above a target floor $\bar{\mathcal{H}}$." The dual gradient on $\alpha$ has a beautiful one-line form — if current entropy is above target, push $\alpha$ down (let the policy sharpen); if below, push $\alpha$ up (force more randomness). This is the canonical machinery for a learned temperature. The variant most relevant to us — Cat-SAC (Lin 2020, `Temperature_lit_review_B_sac_variants.md`) — extends this to a **state-conditional** $\alpha(s)$, demonstrating that the proof of Bellman contraction goes through unchanged so long as $\alpha(s) > 0$.

**The proposed implementation.** Head 1 of the two-headed modulator emits a single positive scalar $\gamma_{\text{NA}}(c) \in \mathbb{R}_{>0}$ as a function of context $c$ (the interoceptive / task signal). This scalar is applied as a broadcast multiplier on the policy's logit vector $z(s) \in \mathbb{R}^{|\mathcal{A}|}$, immediately before the softmax:

$$\pi(a \mid s, c) \;=\; \mathrm{softmax}\!\big(\gamma_{\text{NA}}(c) \cdot z(s)\big).$$

The shift component $\beta_{\text{FiLM}}$ at this layer is pinned to zero (no per-action bias) — that constraint is what keeps the operation *purely* a temperature, rather than a temperature *and* an action prior. To keep $\gamma_{\text{NA}}(c) > 0$, the modulator emits $\log\gamma_{\text{NA}}(c)$ and exponentiates, the same trick SAC uses on $\log\alpha$.

```
   context c          (interoceptive / task signal)
       |
   [Head 1, restricted to 1 scalar output]
       |
   log γ_NA(c) -- exp --> γ_NA(c) > 0
                                     \
   logits z(s) ---------------------- * (broadcast) --> softmax --> π(·|s,c)
```

**Why it's creative / what it unlocks.** The vanilla version of this candidate is the audit's baseline — but the creativity lives in the **conditioning signal**. SAC's auto-$\alpha$ has *one* global scalar per training run; Cat-SAC conditions on a learned curiosity bin; the project's modulator conditions on an **interoceptive context** that may correspond to a biological NA signal (nociceptive load, energy state, novelty). That is a research story neither paper has told. Two specific unlocks: (i) **interpretable arousal**, where you can plot $\gamma_{\text{NA}}(c)$ as a function of pain level / surprise and read off a "vigilance curve" that is hypothesised to match LC-NA firing data; (ii) **dynamic exploration schedules without a target-entropy hyperparameter**, because the modulator is trained end-to-end by policy gradient through the temperature, not by a Lagrangian dual on an entropy floor. The result is a temperature schedule that emerges from the task itself.

**References.**
- Haarnoja et al. 2018a (SAC, ICML) and 2018b (SAC-v2, auto-$\alpha$) — `Temperature_lit_review_A_sac_foundations.md`. Closest precedent for the learned-scalar-temperature pattern; differs by being globally adaptive rather than context-conditioned.
- Lin et al. 2020 (Cat-SAC) — `Temperature_lit_review_B_sac_variants.md`. Closest precedent for state-conditional temperature; uses a discretised-curiosity lookup table rather than a continuous FiLM head.
- Perez et al. 2017 (FiLM) — provides the architectural primitive for the broadcast scalar modulation.

### Candidate 2: Mellowmax-as-FiLM-target — modulator learns a contraction-preserving temperature

**Algorithm intro (plain language).** This candidate borrows from a quieter but mathematically elegant line of work: Asadi & Littman's **mellowmax** operator (ICML 2017, covered in `Temperature_lit_review_C_softmax_operators.md`). The setup: in tabular Q-learning, the reason value iteration converges is that the max-based Bellman backup is a *contraction* — each iteration shrinks the gap to the true optimal Q-values by at least a factor of $\gamma$. If you replace the max with a *softmax average* of Q-values (the standard "Boltzmann softmax"), this contraction property is *lost* — value iteration can have multiple fixed points or even fail to converge. Mellowmax is a particular non-max soft operator built from a normalised log-sum-exp that *preserves contraction at every temperature*. The catch is that the corresponding "behaviour policy" — the actual softmax-style action distribution consistent with mellowmax — has a **state-dependent** inverse temperature $\beta(s)$, normally found by a 1-D root-finder. The creative idea here is: skip the root-finder and have the modulator emit $\beta(s)$ directly, trained to match the property mellowmax guarantees.

**The proposed implementation.** The modulator's Head 1 emits a state-conditional positive scalar $\beta(s, c)$ that plays the role of the mellowmax-consistent inverse temperature. The behaviour policy is then a state-conditional Boltzmann:

$$\pi(a \mid s, c) \;=\; \frac{\exp\big(\beta(s, c)\, Q(s, a)\big)}{\sum_{a'} \exp\big(\beta(s, c)\, Q(s, a')\big)}.$$

Two equivalent training signals are available. **Option A (self-consistency loss):** at each state, the policy should satisfy the mellowmax constraint $\sum_a \pi(a \mid s) \,Q(s,a) = \mathrm{mm}_\omega(Q(s, \cdot))$, where $\mathrm{mm}_\omega$ is the mellowmax operator with a *single fixed* hyperparameter $\omega$. Add an auxiliary loss:

$$\mathcal{L}_{\text{mm}}(\beta) \;=\; \mathbb{E}_s\!\left[\Big(\sum_a \pi(a \mid s; \beta(s,c)) Q(s,a) \;-\; \mathrm{mm}_\omega(Q(s, \cdot))\Big)^{\!2}\right].$$

The modulator is trained jointly on the RL loss and this self-consistency loss; the latter pins $\beta(s, c)$ to satisfy mellowmax's contraction constraint. **Option B (closed-form regression target):** at training time, run Brent's root-finder once per state in a minibatch to compute the *true* mellowmax-consistent $\beta^\star(s)$, then regress $\beta(s, c)$ onto $\beta^\star(s)$. At inference, the modulator replaces the root-finder.

```
   Q(s, ·) ------+--------> mellowmax target β*(s)  [training-only]
                 |                |
                 |                | regression target
   context c     |                v
       |         +-----> Head 1 emits β(s, c) ----+
                                                  |
   Q(s, ·) ----------- × β(s, c) ---- softmax --> π(·|s, c)
```

**Why it's creative / what it unlocks.** Three things. First, this candidate ties the modulator's output to a *mathematically meaningful object*: the unique $\beta(s)$ that makes the soft Bellman backup a contraction. The geometric reading is that the modulator is a *learned section of the bundle of contraction-preserving temperatures over state space* — which is the kind of geometric framing the project's broader research direction wants to lean on. Second, the framework lets you compare two regimes side-by-side: a vanilla FiLM-γ head (Candidate 1) versus a mellowmax-constrained head (Candidate 2), where the only difference is the auxiliary loss. If the constrained version produces empirically smoother $\beta$ profiles or more stable training, the contraction property is doing genuine work. Third — and most speculatively — it gives the **context $c$ a precise theoretical role**: the modulator picks among contraction-preserving temperatures *as a function of NA-coded context*, so different interoceptive regimes pick different mellowmax representatives.

**References.**
- Asadi & Littman 2017 (Mellowmax, ICML) — `Temperature_lit_review_C_softmax_operators.md`. The load-bearing precedent for a contraction-preserving soft operator and the state-dependent $\beta(s)$ behaviour policy.
- Song et al. 2018 — `Temperature_lit_review_C_softmax_operators.md`. Argues that in *function-approximation* deep RL the contraction story matters less and overestimation reduction matters more; useful as a falsification target — if mellowmax-constrained $\beta$ doesn't beat free $\beta$ in our setting, this paper predicts why.

### Candidate 3: Munchausen-style log-policy bonus with FiLM-conditioned coefficient

**Algorithm intro (plain language).** This candidate borrows from a remarkably clever trick: **Munchausen RL** (Vieillard, Pietquin & Geist, NeurIPS 2020), covered in `Temperature_lit_review_D_kl_munchausen.md`. The setup: an earlier theory paper by the same group (Vieillard 2020a, "Leverage the Average") proved that *KL-regularised* value iteration — where each new policy is penalised for moving too far from the previous policy — turns out to be mathematically equivalent to **bootstrapping decisions from a running average of all past Q-estimates**. That averaging is what gives KL-regularised RL its noise robustness: just as repeatedly averaging noisy measurements pulls toward the truth, averaging noisy Q-estimates cancels their idiosyncratic errors. The Munchausen trick collapses this elegant theory into a one-line code change: add $\alpha \tau \log\pi(a_t \mid s_t)$ — the log-probability of the action you actually took, scaled by two hyperparameters $\alpha$ and $\tau$ — to the reward signal. Nothing else changes; no policy network, no constraint optimisation. The resulting M-DQN beats Rainbow on Atari. The creative move here: have the modulator emit $\alpha(c)$ as a per-context coefficient, so the strength of implicit Q-averaging itself becomes NA-modulated.

**The proposed implementation.** Head 1 of the modulator emits a context-conditioned scalar $\alpha(c) \in [0, 1)$ via a sigmoid (or a clipped variant). The temperature $\tau$ is kept as a fixed hyperparameter (or, alternatively, jointly emitted; see ASCII diagram). The Munchausen-style modified reward in the bootstrap target is:

$$\tilde{r}_t(c) \;=\; r_t \;+\; \alpha(c)\, \tau\, \log\pi(a_t \mid s_t)\,.$$

The full M-DQN target (cribbing from Vieillard 2020b, Eq. 2):

$$\hat{q}(r_t, s_{t+1}; c) \;=\; \tilde{r}_t(c) \;+\; \gamma \sum_{a'} \pi(a' \mid s_{t+1}) \Big(q_{\bar\theta}(s_{t+1}, a') - \tau \log\pi(a' \mid s_{t+1})\Big).$$

The policy itself remains a vanilla softmax over Q-values with temperature $\tau$, so the *forward-pass* temperature is unchanged from a standard soft Q-learning setup. **The modulator's scalar $\alpha(c)$ acts on the bootstrap target only.** Per Vieillard 2020b's Theorem 1, this is *exactly* equivalent to running mirror-descent VI with KL coefficient $\alpha(c)\tau$ and entropy coefficient $(1-\alpha(c))\tau$. So as $\alpha(c) \to 1$, the agent leans heavily on its history (strong implicit Q-averaging); as $\alpha(c) \to 0$, it falls back to Soft-DQN (no averaging).

```
   context c
       |
   Head 1 ---> α(c) ∈ [0, 1)      ┐
                                  │
                                  v
   r_t  ─────── + ──── α(c)·τ·log π(a_t|s_t) ──── ̃r_t(c)
                                  |
                                  v
   bootstrap target  ̂q  =   ̃r_t(c)  + γ · soft-V(s_{t+1})
```

Optionally, the same modulator head can also emit a per-context $\tau(c)$ — but note that *this overlaps with the audit's NA route from Candidate 1 plus an ACh-flavoured loss-side scalar*, so it formally crosses into Head 2 / loss-side territory. Flag this architectural cross-talk explicitly: a context-conditioned Munchausen coefficient lives at the boundary of Doya's NA and ACh assignments.

**Why it's creative / what it unlocks.** This is the most theoretically loaded of the three candidates and arguably the most surprising. Three unlocks: (i) **NA as a controller of self-trust** — high $\alpha(c)$ means the agent says "I will weight my running history of Q-estimates strongly," low $\alpha(c)$ means "I will rely on the most recent estimate alone." This gives NA a *new* interpretive frame beyond "make policy greedier": NA-coded contexts (arousal, vigilance, pain) are now contexts where the agent's effective decision is informed by *more of its own past*. (ii) **Theoretical inheritance** — because Theorem 1 of Vieillard 2020b is an exact equivalence, our agent inherits the linear-horizon error-averaging bound of Vieillard 2020a. The modulator's output is not a heuristic — it controls a *quantified* implicit ensembling strength. (iii) **One-line implementation** — Munchausen is famously the rare deep-RL contribution where the modification is just a reward bonus. Adding context-conditioning to it is trivial engineering with rich theoretical payoff.

**References.**
- Vieillard, Pietquin & Geist 2020 (Munchausen RL, NeurIPS) — `Temperature_lit_review_D_kl_munchausen.md`. The one-line reward-bonus trick and its proven equivalence to MD-VI; the load-bearing precedent.
- Vieillard, Kozuno, Scherrer et al. 2020 ("Leverage the Average," NeurIPS) — same lit-review file. Establishes the implicit-Q-averaging interpretation our $\alpha(c)$ modulator inherits.
- Zhu et al. 2023 (Tsallis-KL Munchausen) — same lit-review file. Adjacent extension where $q$-divergence replaces KL; relevant if sparse-policy behaviour is desired.

### What I'd try first

I'd prototype **Candidate 1 first**, but instrumented for the eventual move to Candidate 3. Concretely: build the basic FiLM-γ-at-logits head with the broadcast-only / no-shift constraint and the $\log\gamma$ parameterisation. The reason for this ordering is that Candidate 1 is the smallest delta from the audit's existing two-headed architecture and gives you a *clean control* against which Candidates 2 and 3 must show improvement. The instrumentation matters: log the entropy of $\pi(\cdot \mid s, c)$ per context bin and the modulator's scalar output as a function of $c$ — these are the curves that turn Candidate 1 into a paper about "interoceptive arousal as learned temperature." From there, Candidate 3 (Munchausen with FiLM-conditioned $\alpha(c)$) is the natural next step because it requires only adding a reward bonus and one extra modulator output dimension — no architectural surgery — and unlocks the most theoretically loaded reading of NA as a controller of self-trust. Candidate 2 is the longer-horizon ambition: the mellowmax-constrained head needs an auxiliary loss and possibly a root-finder during training, which is more engineering, but pays off in the strongest geometric / contraction-theoretic story.

## 4. ACh → learning rate

### TL;DR

Doya (2002) proposed that acetylcholine (ACh) sets the brain's *learning rate* — how big a correction is made after each piece of experience. In a textbook RL update $\theta \leftarrow \theta - \alpha \nabla L$, ACh is the $\alpha$ knob. Naively this looks like the easy case: just have the modulator emit a number and multiply it into the optimizer. But the moment we look at how a FiLM (feature-wise linear modulation) block actually interacts with the chain rule, the picture becomes much richer — and arguably more interesting than Doya intended. We sketch three creative routes for plumbing ACh into the agent. **Candidate 1**: re-read the FiLM scale γ as a per-pathway learning-rate gate via the chain rule. **Candidate 2**: have a hypernetwork emit the critic's weights from context, so the geometry itself warps the effective step size. **Candidate 3 (recommended)**: a meta-gradient hybrid where the modulator emits a global scalar $\alpha(c)$ that the optimizer literally uses, plus FiLM gating for per-pathway sharpening.

### Background — what the knob is and why it matters

**What "learning rate" means.** In every gradient-based learner, the parameters $\theta$ get nudged toward better loss values via $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$. The scalar $\alpha$ — the *learning rate* or *step size* — controls how much the network changes per piece of experience. Pick it too high and the weights wobble or diverge; pick it too low and learning crawls. It is the single most consequential hyperparameter in deep learning (Bengio 2012).

**Doya's mapping.** In Doya's (2002) four-knob framework, ACh corresponds to $\alpha$. Biologically, high cholinergic tone is associated with novel or uncertain contexts where the agent should overwrite its current beliefs more aggressively; low tone is the consolidated regime where existing beliefs should be protected. Translated to RL: ACh is a *context-dependent gain on plasticity*.

**Where the learning rate lives.** Here is the subtlety that drives this whole memo. In a neural net, $\alpha$ does *not* live in the forward pass. The forward pass computes activations and a loss; $\alpha$ only appears in the **optimizer step**, after gradients have been computed via backprop. So a "learning-rate modulator" is naturally an object that talks to the optimizer, not to the activations.

**Why this knob is harder than it looks — the chain-rule channel.** A FiLM block sitting inside the network multiplies a hidden activation $h$ by a learned scale γ(c): $h' = \gamma(c) \cdot h$. By the chain rule, the gradient flowing *backward* through that block is also multiplied by γ(c). So if γ(c) is small, both the forward activations AND the upstream gradients shrink — and the gradient shrinkage IS, definitionally, a per-pathway learning-rate cut. This is the **chain-rule channel**: FiLM γ secretly couples feature gating and gradient gating, two effects you cannot separate. The canonical ACh story (a single scalar α) is the special case where γ broadcasts uniformly to all pathways; the generic case is much more expressive.

### Candidate 1: FiLM γ as a per-pathway gradient gate (chain-rule channel)

**Algorithm intro.** FiLM (feature-wise linear modulation, Perez et al. 2017) is a tiny conditioning primitive: a context-dependent scale γ(c) and shift β(c) are applied to a hidden layer's activations: $h' = \gamma(c) \cdot h + \beta(c)$. It was originally proposed for forward-pass *feature conditioning* — letting one network steer another's computation. The audit's central observation is that this same operator has a backward face.

A FiLM block at activation cut $k$ acts on the gradient flowing upstream by the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W_{\text{upstream}}} = J_{\text{up}}^\top \cdot \mathrm{diag}(\gamma(c)) \cdot \frac{\partial \mathcal{L}}{\partial h_k^{\text{post}}}.$$

The same diagonal $\mathrm{diag}(\gamma(c))$ that scales the forward activation appears in the backward gradient. If γ_i(c) = 0.5, then pathway i's effective step size halves; if γ_i(c) = 2.0, it doubles. The modulator IS a per-pathway learning-rate gate — without ever touching the optimizer. This is the cleanest version of what the audit calls "Head 1 only" ACh.

**Proposed implementation.** Insert a FiLM block immediately upstream of the parameters whose effective learning rate should be modulated. The modulator network $m_\phi(c)$ — a small MLP from interoceptive context $c$ to a $d$-dimensional γ vector — emits γ(c) with a sigmoid (bounded in $(0, 1)$) or softplus head. Forward: $h' = \gamma(c) \cdot h$. Backward: gradients flowing into upstream layers are automatically scaled by γ(c) via the chain rule — no special optimizer code needed.

A geometric reading: this is precisely the "gated linear unit" (GLU, Dauphin et al. 2017) family — a multiplicative gate $\sigma(W'x) \odot W x$ that simultaneously routes forward features and gates backward gradients. Highway Networks (Srivastava et al. 2015) are the depth-domain version, with a learned $T(x) \in (0,1)$ deciding per-layer how much to transform versus carry. An LSTM's forget gate is the recurrent-time-domain version. All three are instances of the same primitive: a learned multiplicative gate whose forward role (feature selection) and backward role (gradient routing) are two views of one operator. ACh-via-FiLM-γ is the FiLM-domain instance.

**Why creative.** The "honesty caveat" from the audit was that you cannot have *pure* learning-rate modulation via FiLM — you always also get feature gating, by the chain rule. The creative move is to *embrace* this coupling rather than apologize for it: the project's claim becomes "ACh in our model is a per-pathway plasticity-and-routing operator, strictly generalizing Doya's scalar α." This is biologically defensible: ACh in cortex is widely understood to do exactly this — both gate which sensory features reach cortex (forward routing) and modulate the magnitude of plastic change at those same synapses (backward gating). The Doya scalar α was always a cartoon; the chain-rule reading turns the cartoon into a richer mechanism.

A second creative payoff: the γ-as-gate framing connects ACh modulation directly to the mixture-of-experts (MoE) literature. MoE (Shazeer et al. 2017) is a network where a *gating module* softly routes each input to one of several "expert" subnetworks via a learned softmax weighting. FiLM γ at multiple cuts implements a *soft* MoE: each γ_i picks a fraction of pathway i's contribution. "ACh modulates a soft mixture of plasticity pathways" is a clean, publishable framing.

**References.** Perez et al. 2017 (FiLM original); `Learning_Rate_lit_review_D_gated_conditional_arch.md` (the lit review covering Highway Networks, GLU, HyperNetworks, all of which formalize the FiLM-γ-as-gate reading); Shazeer et al. 2017 (sparse MoE); Dauphin et al. 2017 (GLU).

### Candidate 2: Hypernetwork emits per-context weights (effective learning rate via Jacobian pullback)

**Algorithm intro.** A hypernetwork (Ha, Dai & Le 2017) is a small neural network whose *output* is the weights of another, larger network. Concretely, you replace the standard recipe "the critic is parameterized by $\theta$" with "the critic is parameterized by $\theta = g_\phi(c)$ where $g_\phi$ is a small MLP and $c$ is context". The big network does the actual task; the small network emits its weights as a function of context. Now the *learnable parameters* are $\phi$, the hypernetwork's weights, and the optimizer descends $\phi$ — not the main network's $\theta$ directly.

Where does the learning rate live in this setup? By the chain rule, if we update $\phi$ via $\phi \leftarrow \phi - \alpha \nabla_\phi L$, the *effective* update on $\theta$ is

$$\Delta\theta = \frac{\partial g_\phi}{\partial \phi}\Delta\phi \approx -\alpha \frac{\partial g_\phi}{\partial \phi}\frac{\partial g_\phi}{\partial \phi}^\top \nabla_\theta L.$$

The matrix $\frac{\partial g_\phi}{\partial \phi}\frac{\partial g_\phi}{\partial \phi}^\top$ is a context-dependent **Jacobian pullback** — a learned preconditioner. The effective learning rate on the main critic is not $\alpha$ but $\alpha$ times the local Jacobian norm of the hypernetwork at this context. Different contexts $c$ produce different effective step sizes, automatically.

**Proposed implementation.** Let the context $c$ be the interoceptive state plus any task signal. The hypernetwork is a small MLP $g_\phi: \mathbb{R}^{|c|} \to \mathbb{R}^{|\theta_{\text{critic}}|}$. To keep memory tractable, use the **row-scaling** form from Ha et al. 2017 (their Eq. 7): rather than emit the full critic weight matrix, emit a per-row scaling vector $d(c) \in \mathbb{R}^{N_h}$ that left-multiplies a baseline weight matrix $W^{\text{base}}$:

$$W^{\text{critic}}(c) = \mathrm{diag}(d(c)) \cdot W^{\text{base}}.$$

This recovers a form *structurally identical to FiLM* — but where the rows being scaled are the rows of a weight matrix rather than the activations of a hidden layer. The geometric interpretation: the hypernetwork's row-scaling output picks, *per context*, which rows of the critic's weight matrix are "plastic right now" and which are "frozen right now". The optimizer continues to descend $\phi$ at a constant base rate $\alpha$, but the *effective* plasticity of each critic row depends on $d(c)$.

**Why creative.** This is the most ambitious of the three candidates because it lifts the modulator from a *side-channel* (Candidate 1) to a *first-class generator* of the policy/critic. Three creative payoffs:

First, the framing reaches NeurIPS-quality theoretical territory. Sarafian et al. (ICML 2021) showed that "recomposing reinforcement learning with hypernetworks" — emitting policy and critic weights from a state-action embedding — gives genuinely better RL performance than concatenation-based conditioning. Our project's twist: the hypernetwork is the *neuromodulator*. ACh becomes the learned generator of the agent's instantaneous parameters.

Second, the Jacobian pullback gives a clean information-geometric interpretation. The matrix $\frac{\partial g_\phi}{\partial \phi}\frac{\partial g_\phi}{\partial \phi}^\top$ is the Gram matrix of the hypernetwork's output Jacobian, and natural-gradient theory (Amari) says this is exactly the right object to use as a learning-rate preconditioner if you want updates to respect the manifold geometry of the parameterized model. So our ACh-as-hypernet story can be read as: *the brain has discovered an amortized natural-gradient preconditioner, indexed by interoceptive context*.

Third, this candidate cleanly hosts the WarpGrad geometric framing (Flennerhag et al. 2020). WarpGrad inserts learnable "warp layers" between the task-learner's layers, reshaping the loss landscape so vanilla SGD on warped coordinates behaves correctly across tasks. If the warp layers are FiLM-conditioned on a modulator $m_t$, the induced metric $G$ becomes modulator-conditioned — the loss landscape itself is shaped by ACh. The Doya story becomes formally about *meta-Riemannian geometry of plasticity*.

**References.** Ha, Dai & Le 2017 (HyperNetworks, ICLR); Sarafian et al. 2021 (ICML, hypernet-based RL); `Learning_Rate_lit_review_A_l2o_foundations.md` (covers WarpGrad's geometric framing); `Learning_Rate_lit_review_D_gated_conditional_arch.md` (full HyperNetworks deep dive); Amari (1998, natural gradient).

### Candidate 3: Meta-gradient-style Head 2 emits a global α(c) (recommended)

**Algorithm intro.** Meta-gradient RL (Xu, van Hasselt & Silver, NeurIPS 2018) treats hyperparameters of the RL update as *learnable*: rather than picking the discount γ or the learning rate α once and freezing them, the agent updates them online via a meta-gradient on a held-out validation objective. The mathematical move: differentiate through one inner update step to compute $\partial L / \partial \alpha$, and run a second-order gradient descent on α itself. Sutton's IDBD (AAAI 1992) is the historical ancestor — per-feature step sizes adapted by stochastic meta-descent in linear function approximation. Andrychowicz et al.'s "Learning to learn by gradient descent by gradient descent" (NeurIPS 2016) is the deep generalization — an LSTM emits the full update rule, including the effective step size.

In our project's two-headed modulator architecture (per the audit), **Head 2** already emits scalar coefficients into the loss/return estimator. The proposal: Head 2 *also* emits a context-dependent scalar $\alpha(c) \in \mathbb{R}_{>0}$ that the optimizer uses literally as its step size for context $c$:

$$\theta_{t+1} = \theta_t - \alpha(c_t) \cdot \nabla L(\theta_t).$$

This is the most direct mapping of Doya's original story: ACh is the scalar α; the modulator emits α; the optimizer uses it. No chain-rule trickery, no Jacobian pullback — just a learned scalar.

**Proposed implementation.** Head 2 of the modulator emits $\alpha(c) = \alpha_0 \cdot \mathrm{softplus}(m_\phi^{\text{Head2}}(c))$ where $\alpha_0$ is a base learning rate and the softplus head ensures positivity. Two engineering routes for *how* α(c) actually updates the parameters:

(a) **Direct optimizer-step modulation.** The training loop multiplies $\alpha(c)$ into the SGD/Adam step for the current batch. Simple but somewhat outside the autograd graph. Best for SGD; Adam's preconditioning interacts awkwardly with a learned outer scale.

(b) **Meta-gradient via differentiable inner update.** Following Xu et al.'s "single-step approximation" ($A = 0$ shortcut, see `Learning_Rate_lit_review_C_meta_gradient_rl.md`), differentiate one inner update with respect to α(c) — one extra backward pass — and update the modulator parameters with the resulting meta-gradient against a held-out objective. This is more principled and integrates cleanly with JAX/Flax autograd.

For the project, route (b) is the right pick: it makes ACh's gain a *learned* quantity rather than a hand-tuned one, which is precisely what makes the architecture publishable as "biologically informed meta-RL".

**Why creative.** Three creative payoffs.

First, this candidate is the cleanest possible instance of "the brain has a learned meta-optimizer". Doya's 2002 framework was a brilliant cartoon — four chemical systems map to four hyperparameters — but until now, the field has only been able to *fix* those hyperparameters by hand. With Candidate 3, the hyperparameter is *learned by gradient descent*, which means the project can run an experiment prior work could not: "given a free-to-learn ACh head, does it *converge* on the values Doya's framework predicts?" That is a *prediction*, not a fit.

Second, the meta-gradient framing makes the project a strict descendant of IDBD → Schraudolph SMD → Xu et al. 2018, with an unusually clean place in the literature.

Third, and perhaps most creative: a **hybrid with Candidate 1** writes itself. Head 2 emits a *global* α(c) — the "cholinergic tone" Doya intended. Head 1 emits FiLM γ(c) at hidden layers — the "per-pathway sharpening" the chain-rule channel automatically gives us. The two are not redundant: Head 2 scales the magnitude of plastic change globally; Head 1 routes that plastic change to specific pathways. Biologically, this maps onto cortical-vs-subcortical ACh release with different effective spatial scales — basal-forebrain global tone vs local cortical modulation. The full architecture is then: $\theta_{t+1} = \theta_t - \alpha(c) \cdot J^\top \mathrm{diag}(\gamma(c)) \nabla L$. The honest version of Doya is *both* a scalar gain AND per-pathway routing — and the two-headed architecture lets us implement both without crosstalk.

**References.** Xu, van Hasselt & Silver 2018 (Meta-Gradient RL, NeurIPS); Andrychowicz et al. 2016 (Learning to learn, NeurIPS); Sutton 1992 (IDBD, AAAI); Kearney et al. 2018 (TIDBD, generalizing IDBD to TD); Baydin et al. 2018 (hypergradient descent, ICLR); `Learning_Rate_lit_review_B_lr_adaptation.md` and `Learning_Rate_lit_review_C_meta_gradient_rl.md` cover the full lineage.

### What I'd try first

The hybrid of Candidates 1 and 3. Stand up a two-headed modulator: Head 1 emits FiLM γ(c) at the critic's penultimate layer (single cut, single γ vector — the cheapest version); Head 2 emits a scalar $\alpha(c) \in (0.1 \alpha_0, 10 \alpha_0)$ via a softplus-and-clip on a logit. Train end-to-end with a meta-gradient on a held-out trajectory, following Xu et al.'s single-step approximation. Then run the *prediction* experiment: plot $\alpha(c)$ against an environmental novelty proxy (e.g., recent reward variance, or the agent's own surprise signal). If $\alpha(c)$ rises with novelty and falls with consolidation, *without ever being told to do so*, Doya 2002 has just been validated as a learned mechanism rather than a hand-tuned cartoon. That is the headline result the architecture is designed to make possible.

## 5. DA → TD-error gain

### TL;DR

Doya's mapping puts dopamine (DA) in charge of one specific knob — the multiplier $\kappa$ that sits in front of the temporal-difference error $\delta_t$ when the critic and actor get updated. The compact memo already worked out the boring version: emit $\kappa(c)$ as a scalar from Head 2 and multiply the advantage. That works, but it is gauge-equivalent to the learning rate, so as a *claim* about what DA is doing it is weak — and as an *opportunity* it leaves the whole rest of distributional/transfer/sequence-model RL on the table. This memo proposes three creative upgrades. **Candidate 1: Successor-features Head 2** — Head 2 emits a *reward weight vector* $w_r(c) \in \mathbb{R}^d$ instead of a scalar, so DA picks *which features* matter, not just how much. **Candidate 2: IQN-style risk distortion** — Head 2 emits a risk-preference parameter $\beta(c)$ that selects which part of the *return distribution* the agent optimises against, putting DA in the role of distributional risk knob. **Candidate 3 (recommended): Hypernet-emitted critic weights** — Head 2 *is* a hypernetwork that writes $\theta_Q(c)$, the most expressive option and a superset of the first two.

### Background — what the knob is and why it matters

In the standard actor-critic update, the policy gradient is
$$\nabla_\theta J(\theta) \;=\; \mathbb{E}\!\left[\,\delta_t \,\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right],\qquad \delta_t \;=\; r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).$$
The TD error $\delta_t$ is the "reward-prediction error" — the gap between what the critic predicted and what actually happened. Doya (2002) proposed that dopamine in the brain controls a multiplicative gain $\kappa$ on this error:
$$\Delta\theta \;\propto\; \kappa \,\delta_t \,\nabla_\theta \log\pi_\theta(a_t\mid s_t).$$
Plain-English reading: $\kappa$ says *how much* the agent should update its policy in response to a surprise. High dopamine → big response to surprise; low dopamine → blunt response.

Crucially, $\kappa$ lives on the **loss side** of the computation, not the forward side. It enters when you compute the gradient, not when the policy network produces an action. This is structurally different from noradrenaline (NA), which scales policy logits, and from serotonin (5-HT), which sits in the return target. NA changes *what action gets chosen* given fixed weights; DA changes *how the weights move* given the error. They act on orthogonal parts of the computation graph.

But there is a problem the compact memo flagged: a scalar $\kappa$ is gauge-equivalent to the global learning rate. The transformation $(\kappa, \mathrm{lr}) \mapsto (\lambda \kappa, \mathrm{lr}/\lambda)$ leaves on-policy SGD invariant. So Head-2-emits-a-scalar is the *minimal* DA implementation but a *weak* identifiability claim. The candidates below escape the gauge by emitting something richer than a scalar.

### Candidate 1: Successor-features Head 2 — DA emits a reward weight vector

**Algorithm intro.** Successor features (SFs) come from Barreto et al. (NeurIPS 2017) and a follow-up at ICML 2019. The trick is to break the value function into two pieces. Pretend for a moment that the reward at every step decomposes as a dot product:
$$r(s, a, s') \;=\; \boldsymbol{\phi}(s, a, s')^\top \mathbf{w}_r,$$
where $\boldsymbol{\phi} \in \mathbb{R}^d$ is a *feature vector* describing what salient things happened in this transition (the agent picked up a coin, hit a wall, saw red), and $\mathbf{w}_r \in \mathbb{R}^d$ is a *reward-weight vector* describing how much the agent cares about each of those things. Substituting this into the value function gives a clean factorisation:
$$V^\pi(s) \;=\; \boldsymbol{\psi}^\pi(s)^\top \mathbf{w}_r, \qquad \boldsymbol{\psi}^\pi(s) \;=\; \mathbb{E}^\pi\!\left[\sum_{i=t}^\infty \gamma^{i-t} \boldsymbol{\phi}_i \,\Big|\, s_t = s\right].$$
The vector $\boldsymbol{\psi}^\pi(s)$ is the **successor feature** of $s$ — the discounted future occurrence of each feature under $\pi$. It depends on dynamics and policy but *not* on what the agent currently wants. Change $\mathbf{w}_r$ and you instantly re-evaluate the same policy on a new reward function — no retraining of $\boldsymbol{\psi}$. See TD_lit_review_B (successor features) for the full derivation and the generalised policy improvement (GPI) theorem that lets multiple SFs be combined.

**Proposed implementation.** Replace the scalar $\kappa_{\text{DA}}(c)$ with a *vector* $\mathbf{w}_r(c) \in \mathbb{R}^d$ emitted by Head 2. The critic becomes
$$V_\phi(s; c) \;=\; \boldsymbol{\psi}_\phi(s)^\top \mathbf{w}_r(c),$$
where $\boldsymbol{\psi}_\phi$ is the successor-feature trunk (shared across contexts) and $\mathbf{w}_r(c)$ is the context-conditioned reward-weight vector. The TD error then takes the SF-Bellman form:
$$\boldsymbol{\delta}_t \;=\; \boldsymbol{\phi}_t + \gamma\, \boldsymbol{\psi}_\phi(s_{t+1}) - \boldsymbol{\psi}_\phi(s_t) \;\in\; \mathbb{R}^d,$$
which is a *vector* of TD errors, one per feature. The policy-gradient update uses the projected scalar $\boldsymbol{\delta}_t^\top \mathbf{w}_r(c)$ as the advantage signal. The advantage is now naturally context-dependent through which features DA flags as currently reward-relevant.

The choice of $\boldsymbol{\phi}$ is the next decision. Barreto's 2019 paper (TD_lit_review_B §2) provides a beautiful shortcut: **the base-task rewards themselves can serve as features**. If the project trains across $D$ contexts (e.g., baseline vs. high-pain vs. fasted), take $\boldsymbol{\phi} = (r_1, \dots, r_D)$ — the reward signal under each canonical context. Then $\boldsymbol{\psi}^\pi$ collapses to the stacked Q-values across the $D$ canonical tasks, and Head 2's $\mathbf{w}_r(c)$ becomes a context-conditioned *mixture weight over canonical reward regimes*. This is interpretable and avoids hand-designing $\boldsymbol{\phi}$.

**Why creative.** This breaks the DA/learning-rate gauge cleanly. A scalar $\kappa$ commutes with $\mathrm{lr}$; a *vector* $\mathbf{w}_r$ does not — a direction in feature space is not equivalent to a magnitude on the learning rate. The behavioural interpretation also becomes interesting: DA in this view is not "be more or less excited," it is "decide which of these reward sources currently counts." That maps naturally onto the project's interoceptive-state context: in a pain-on context, $\mathbf{w}_r(c)$ should over-weight pain-related features; in a hunger context, it should over-weight food-related features. The agent is not learning *a* value function; it is learning a *family* indexed by what DA is currently telling it to value. Combined with GPI (max over a library of context-trained policies, Barreto's Theorem 1), you get a zero-shot transfer mechanism across contexts for free.

**References.** Barreto et al. 2017 (NeurIPS) "Successor Features for Transfer in RL" — original SF&GPI framework, Theorems 1 and 2. Barreto et al. 2019 (ICML) — the rewards-as-features trick (TD_lit_review_B §2.2.3). Borsa et al. 2018 (ICLR) "Universal Successor Features Approximators" — extends to a single network conditioned on a policy descriptor $\mathbf{z}$, which is the closest published architecture to a context-conditioned SF critic (TD_lit_review_B §3). Dayan 1993 — the tabular successor representation, the SF ancestor.

### Candidate 2: IQN-style distortion preference — DA picks the risk quantile

**Algorithm intro.** Distributional RL learns the whole probability distribution over returns rather than just the mean. The line of work starts with C51 (Bellemare, Dabney, Munos 2017, ICML) which represents the return as a categorical distribution over 51 fixed atoms, continues through QR-DQN (Dabney et al. 2018, AAAI) which learns the locations of $N$ quantiles directly, and culminates in **Implicit Quantile Networks** (IQN — Dabney, Ostrovski, Silver, Munos 2018, ICML). IQN's trick is to make the value network take a *quantile level* $\tau \in [0, 1]$ as a forward-pass input and emit the return at that quantile:
$$Z_\theta(s, a; \tau) \;\approx\; F_{Z(s,a)}^{-1}(\tau),$$
where $F^{-1}$ is the inverse CDF of the return distribution. Plain-English: instead of "what is the expected return?", the network answers "what is the $\tau$-th percentile return?". Sampling many $\tau$'s and averaging recovers the mean; sampling $\tau$ from the bottom 10% recovers a risk-averse policy ("what's the 10th-percentile outcome — what would happen on a bad day?"). See TD_lit_review_A for the IQN derivation and the distortion-function framework.

A **distortion** is a map $\beta : [0,1] \to [0,1]$ that re-weights how much each quantile counts. The CVaR-$\alpha$ distortion is $\beta(\tau) = \tau / \alpha$ for $\tau < \alpha$ and constant after — meaning "average over the worst $\alpha$ fraction of outcomes." The Wang distortion, the CPW distortion, and many others encode different risk preferences (Dabney IQN §4). IQN's headline move is that you can pick *any* distortion at action time without retraining the critic.

**Proposed implementation.** Have Head 2 emit a context-conditioned distortion parameter $\beta(c) \in \mathbb{R}^k$ (typically 1- or 2-dimensional — a single CVaR level $\alpha(c)$, or a Wang-distortion parameter). The critic is an IQN trunk:
$$Z_\phi(s, a; \tau) \in \mathbb{R}, \qquad \tau \sim U[0,1].$$
The policy targets the distorted value
$$V^\beta_\phi(s, a; c) \;=\; \int_0^1 Z_\phi(s, a; \tau)\, d\beta(\tau; c) \;\approx\; \tfrac{1}{K}\sum_{k=1}^K Z_\phi\big(s, a;\, \beta^{-1}(\tau_k; c)\big),$$
and the advantage becomes $A^\beta(s, a; c) = V^\beta(s, a; c) - V^\beta(s; c)$. The TD-error gain is now context-dependent *through the shape of the return distribution it samples from*, not through a scalar multiplier. DA in this picture says: "are we in a context where I should care about the typical outcome, the worst-case outcome, or the best-case outcome?"

**Why creative.** Two reasons. First, this version of DA is non-trivially identifiable — a CVaR objective gives a measurably different policy from a mean-objective policy in any environment with reward variance, and the difference does not vanish under the $(\kappa, \mathrm{lr})$ gauge. Second, it connects to neuroscience: DA neurons in the VTA have been shown to code reward *distributions* (Dabney et al. Nature 2020 "A distributional code for value in dopamine-based reinforcement learning"), with different DA cells responding to different quantiles. The interpretation that DA controls *which quantile is read out* for the policy gradient is biologically plausible. In the project's interoceptive setting, this gives a clean mechanism for hypervigilance: a pain-on / threat context drives $\beta(c)$ toward the lower tail (CVaR), making the agent act as if optimising the worst-case return — exactly the behavioural signature the H5 chronic-pain analog wants.

**References.** Bellemare, Dabney, Munos 2017 (ICML) "A Distributional Perspective on RL" — C51, the foundation (TD_lit_review_A §1). Dabney, Rowland, Bellemare, Munos 2018 (AAAI) — QR-DQN, quantile representation (TD_lit_review_A §2). Dabney, Ostrovski, Silver, Munos 2018 (ICML) — IQN, $\tau$-conditioning (TD_lit_review_A §3). Lim & Malik 2022 — distributional RL for risk-sensitive policies, ties distortion to behavioural risk preference (TD_lit_review_A §4). Dabney et al. Nature 2020 — distributional code for value in dopamine neurons. **Note:** the TD_lit_review_A document explicitly flags IQN's $\tau$-conditioning as architecturally identical to FiLM-conditioning the value head on a risk-level scalar — the IQN architecture is the cleanest existing precedent for what Head 2 should look like in this candidate.

### Candidate 3 (recommended): Hypernet-emitted critic weights

**Algorithm intro.** A **hypernetwork** is a small network whose output is the weights of another network. Ha, Dai, Le (ICLR 2017) introduced the generic primitive. Sarafian, Keynan, Kraus (ICML 2021) showed that in off-policy actor-critic, treating the Q-function as a hypernetwork — let one network ingest $s$ and emit the weights of a small network that consumes $a$ — gives substantially better action-gradient quality than the concatenation baseline (TD_lit_review_C §1). Plain-English: instead of feeding the context $c$ to the critic as another input, let a small network *write the critic's weights* based on $c$. Each context value spawns a different critic. The agent is, in effect, re-parameterised on every forward pass by the modulator.

The structural payoff (Sarafian Proposition 1) is that the action-gradient $\nabla_a Q$ — the quantity the actor follows — becomes much more accurate when state-dependence enters through generated weights $W^\ell(s)$ rather than only through ReLU active-set masks. Rezaei-Shoshtari et al. (AAAI 2023, "HyperZero") showed that the same trick gives zero-shot transfer across MDP families when the hypernet ingests task parameters $(\psi, \mu)$ (TD_lit_review_C §2).

**Proposed implementation.** The modulator IS the hypernetwork. Head 2 ingests $c$ and emits the full weight vector $\theta_Q(c)$ of the critic's last block (or the entire critic):
$$\theta_Q(c) \;=\; H_\Theta(c), \qquad V_{\theta_Q(c)}(s) \in \mathbb{R}.$$
The critic loss and TD error then take their usual form, but every backward pass through $V$ now flows through the hypernet $H_\Theta$, so the gradient signal does double duty: it teaches the hypernet "which contexts need which critics" *and* it teaches the critic-architecture-as-a-whole how to fit values.

Following Sarafian's empirical findings, two structural choices matter: (i) the dynamic-layer form should be hypernet **plus** a FiLM-style gain, $h^{l+1} = \sigma((1 + g^l(c)) \odot h^l W^l(c) + b^l(c))$ (Sarafian Eq. 1, found necessary for stability via Littwin & Wolf 2019); and (ii) the primary net should be moderately deep (a small ResNet) while the *dynamic* critic stays small (one hidden layer × 256). The primary net bears the parameter cost; the generated critic is lean. Weight initialisation matters — Chang, Flokas, Lipson (2019) "Principled Weight Initialization for Hypernetworks" is load-bearing here.

**Why creative — and why recommended.** This is the **most expressive Head 2**, and it strictly contains Candidates 1 and 2 as special cases: a hypernet that always emits weights of the form "linear readout with reward weights $\mathbf{w}_r(c)$ on a fixed $\boldsymbol{\psi}$" is Candidate 1; a hypernet whose generated critic happens to be an IQN with $\beta(c)$-shaped distortion is Candidate 2. So an experimental program that starts at rung 3 (hypernet) and ablates *downward* into the simpler candidates can actually test which structural assumption is doing the work. The compute cost is real (hypernets are sensitive to init and need an LR re-sweep per Sarafian §2.7), but the payoff is the cleanest possible identifiability claim: "the modulator does not just multiply, shift, or scale the critic — it *writes* it." The gauge worry that plagued the scalar version evaporates, because a high-dimensional weight tensor is not invertibly absorbable into any scalar elsewhere in the graph.

This route also positions the project to make a real claim about what the *modulator* is — not a temperature, not a discount, but a *meta-learner over critic-architectures-given-context*. The Sarafian "state-as-meta" rule transfers cleanly: the modulator is the slow-varying low-dim "context" input and should be the **meta-variable** (input to the hypernet), with the agent's observation as the base. HyperZero's empirical zero-shot result is direct evidence that this structural choice generalises beyond training-distribution contexts.

**References.** Ha, Dai, Le 2017 (ICLR) "Hypernetworks" — the primitive. Sarafian, Keynan, Kraus 2021 (ICML) "Recomposing the RL Building Blocks with Hypernetworks" — SA-Hyper, action-gradient quality argument, ResNet primary + FiLM-augmented dynamic layer (TD_lit_review_C §1). Rezaei-Shoshtari et al. 2023 (AAAI) "HyperZero" — zero-shot transfer when the hypernet ingests task parameters (TD_lit_review_C §2). Chang, Flokas, Lipson 2019 — principled hypernet weight init. Galanti & Wolf 2020 — the modularity/abstraction argument that justifies why "context → weights" generalises better than "(state, context) → action."

### What I'd try first

If I had one prototype to spend, I would build Candidate 3 (hypernet critic) with the *option* to ablate down to Candidate 1 (successor features) by restricting the hypernet output to a low-rank factorisation $\theta_Q(c) = \mathrm{diag}(\mathbf{w}_r(c)) \cdot \boldsymbol{\Psi}_{\text{shared}}$. This buys two things simultaneously: a strong-form identifiability claim (the hypernet writes the critic, so DA-as-gain is not a gauge of the learning rate), and a fall-back to the SF interpretation if the full hypernet over-fits. The IQN candidate is the most biologically interesting (distributional DA codes are an active neuroscience finding) and I would keep it as the second prototype — particularly because the project's pain-on context is exactly the place where a CVaR-style risk knob has obvious behavioural meaning. The headline experiment I would design: train all three Head-2 variants on a context-shift schedule, and ask whether the policy at *test time* in a held-out context is best produced by GPI over an SF library (Candidate 1), by lowering the CVaR quantile (Candidate 2), or by simply letting the hypernet generate a fresh critic (Candidate 3). The answer will not be obvious.

## 6. 5-HT → time-discount

### TL;DR

Doya's mapping says serotonin (5-HT) sets the agent's **time-discount** — how heavily it weighs reward later vs. reward now. In RL this is the $\gamma$ inside $V(s) = \mathbb{E}[\sum_t \gamma^t r_t]$, a number between 0 and 1 that exponentially shrinks reward at distance $t$. We are exploring how to make $\gamma$ context-conditioned in a two-headed modulator: Head 1 emits FiLM scales/shifts to the critic's activations (the *conditioning* channel), Head 2 emits scalar coefficients into the return target (the *loss-side* channel). For 5-HT this is the richest of the four routes — the discount is architecturally more diverse than temperature, learning rate, or RPE gain because it can be (a) a single scalar input to a critic that learns a value-function *family*, (b) a weighting over a basis of pre-trained value heads at different horizons, or (c) the conditioning input to a generative model of the discounted future. We sketch three creative candidates. **Candidate 1**: a γ-conditional UVFA where Head 2 emits $\gamma_B(c)$ directly. **Candidate 2**: a Fedus-style multi-horizon ensemble where Head 2 emits the *shape* of the discount via weights $w(\gamma; c)$. **Candidate 3 (recommended)**: a γ-Model where Head 2 conditions a generative model of the discounted future state-occupancy.

### Background — what the knob is and why it matters

The **time-discount** $\gamma \in [0, 1)$ is the number that turns an infinite sum of future rewards into a finite quantity the agent can actually maximize. Concretely, in the standard RL objective

$$V^\pi(s) = \mathbb{E}^\pi\!\left[\sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0 = s\right],$$

the discount $\gamma$ does three jobs simultaneously: (i) keeps the sum finite when $r_t$ is bounded; (ii) tells the agent to prefer reward sooner over later; (iii) sets the **effective horizon** $\tau \approx 1/(1 - \gamma)$ — roughly, how many steps into the future the agent "sees". Setting $\gamma = 0.9$ gives $\tau \approx 10$ steps; $\gamma = 0.99$ gives $\tau \approx 100$; $\gamma = 0.999$ gives $\tau \approx 1000$. Small changes near $\gamma = 1$ produce huge changes in horizon.

Doya 2002 mapped $\gamma$ to serotonin: high 5-HT lengthens the agent's horizon; low 5-HT shortens it. This dovetails with behavioral findings — depleted serotonin causes impulsive choice; chronic pain induces narrow temporal focus.

Where does the discount live in the computational graph? **Inside the Bellman recursion**, specifically in the bootstrap target:

$$V^\pi(s) = r(s) + \gamma\, \mathbb{E}[V^\pi(s')].$$

It is *not* a forward-pass quantity in the standard setup. It multiplies the bootstrapped value of the next state when constructing the regression target for the critic. This makes it architecturally different from temperature (NA, lives at the policy logits), learning rate (ACh, lives in the optimizer), and RPE gain (DA, lives as a scalar on the advantage). $\gamma$ is a *global* property of the value function and of the entire trajectory's contribution to the loss — change it, and you change the meaning of the value head everywhere.

That global character is why 5-HT is the architecturally richest of Doya's four. Once $\gamma$ becomes context-conditioned, "where does $c$ enter?" admits three distinct answers — input to the critic, weights over a horizon basis, input to a generative dynamics model — each connecting to a separate RL literature (UVFA / γ-Nets, multi-horizon / hyperbolic discounting, γ-Models). The knob also touches **time perception** (effective horizon shifts with context), **risk** (Sozou-style $\gamma$ as posterior over an uncertain hazard rate), and **behavioral economics** (preference reversal). For a project framing 5-HT as a chronic-pain analog, this is the knob most directly tied to the clinical reading.

### Candidate 1: γ-conditional UVFA with Head 2 emitting $\gamma_B(c)$

**Algorithm intro.** A **UVFA** (Universal Value Function Approximator; Schaul et al. 2015) is a single neural network that takes both state $s$ and an extra parameter $g$ as input and returns $V(s, g)$. Plain English: instead of training one network per "goal" — one for the kitchen, one for the bedroom — train *one* network that takes "which goal?" as an input. The network learns a *family* of value functions, one per $g$, and generalizes to unseen goals. **γ-Nets** (Sherstan et al. 2020) is this construction with $g = \gamma$: the network takes the discount factor as input and learns the family $\{V^\pi_\gamma\}_\gamma$. One network, many horizons, queryable at run time.

**Proposed implementation.** Adapt γ-Nets to our two-headed modulator. Head 1 emits FiLM scales/shifts modulating the critic's hidden activations (the indexing mechanism). Head 2 emits scalar $\gamma_B(c) \in [0, 1)$ used in the bootstrap target:

$$\mathcal{L}(\phi; c) = \mathbb{E}\!\left[\big(V_\phi(s; c) - r - \gamma_B(c)\, V_\phi(s'; c)\big)^2\right], \qquad \gamma_B(c) = \mathrm{Head2}(c).$$

Head 1 conditions the critic; Head 2 emits the discount. The critic learns the family $\{V^\pi_{\gamma_B(c)}\}_c$ — γ-conditional UVFA with the "goal" axis replaced by the discount axis, indexed by $c$.

Two engineering details transfer from Sherstan ([Gamma_lit_review_B_multi_horizon.md](../../docs/project/references/Gamma/Gamma_lit_review_B_multi_horizon.md)): feed **both** $\gamma$ and $\tau = 1/(1-\gamma)$ into the modulator's context $c$ (each linearizes a different horizon range); use the loss-scaled TD error $\delta_t = (1-\gamma_B(c))[r_t + \gamma_B(c) V(s_{t+1}; c) - V(s_t; c)]$ to keep error magnitudes comparable as horizons stretch. From Schaul 2015 ([Gamma_lit_review_A_universal_vf.md](../../docs/project/references/Gamma/Gamma_lit_review_A_universal_vf.md)), the two-stage matrix-factorization warm-start is a pre-training accelerator: factorize a sparse $(s, \gamma)$ value table into target embeddings, regress the critic onto those targets, then attach Head 2 and fine-tune. Schaul reports order-of-magnitude speedup on grid-world.

**Why creative.** The standard γ-Nets construction samples $\gamma$ exogenously. Our version is *closed-loop*: Head 2 *emits* $\gamma_B(c)$ from the agent's current context. The modulator becomes a horizon controller — high pain / hazard shortens the horizon, safe context lengthens it. This is structurally what the Sozou (1998) hazard-rate framing predicts: an agent that updates its discount based on inferred hazard. With $c$ varying within a trajectory, the natural object is a time-inhomogeneous discounted return; equivalently, value in the augmented MDP with state $(s, c)$. The empirical signature — perturb a reward at delay $k$, read $|\Delta V_\phi(s; c)|$, slope $\approx \log \gamma_B(c)$ on log scale — is a clean falsifier: slope varying with $c$ and matching training-side $\gamma_B(c)$ confirms γ-conditional UVFA; flat slope across $c$ means the modulator is decorative on this knob.

**References.** Schaul et al. 2015 — *Universal Value Function Approximators* (ICML; introduces UVFA, two-stream architecture, two-stage warm-start). Sherstan et al. 2020 — *γ-Nets: Generalizing Value Estimation over Timescale* (AAAI; the operational template — $\gamma$ as input, both $\gamma$ and $\tau$ feeds, loss scaling). Xu, van Hasselt & Silver 2018 — *Meta-Gradient RL* (treats $\gamma$ as a learned scalar via meta-gradient, the closest precedent for a *learned* discount). Reviews: [Gamma_lit_review_A_universal_vf.md](../../docs/project/references/Gamma/Gamma_lit_review_A_universal_vf.md), [Gamma_lit_review_B_multi_horizon.md](../../docs/project/references/Gamma/Gamma_lit_review_B_multi_horizon.md).

### Candidate 2: Multi-horizon ensemble with Head 2 emitting the discount *shape*

**Algorithm intro.** Fedus et al. 2019 (*Hyperbolic Discounting and Learning over Multiple Horizons*) showed that **non-exponential discount functions can be assembled from a basis of exponential ones**. The load-bearing identity is

$$\frac{1}{1 + kt} = \int_0^1 \gamma^{kt}\, d\gamma$$

— hyperbolic discounting equals an integral over exponentials. More generally, any discount $d(t)$ expressible as $d(t) = \int_0^1 w(\gamma) \gamma^t\, d\gamma$ admits **Lemma 5.1**:

$$Q^d_\pi(s, a) = \int_0^1 w(\gamma)\, Q^\gamma_\pi(s, a)\, d\gamma.$$

Plain English: train value functions at many different discounts in parallel (the "bag of $\gamma$"), then combine them with weights $w(\gamma)$ to produce a value function with whatever discount *shape* you want — exponential, hyperbolic, U-shaped, anything. Architecturally this is one shared trunk with $n_\gamma$ small affine heads, each trained with a standard TD target at its own $\gamma_i$. Fedus's second, almost-independent finding: even when the *behavior policy* uses a single $\gamma$, the multi-horizon heads act as a powerful **auxiliary task** that improves representation learning — `Multi-Rainbow` (multi-horizon heads, single-$\gamma$ behavior) beats Rainbow on 14/19 Atari games, nearly matching `Hyper-Rainbow` (multi-horizon heads, hyperbolic behavior).

**Proposed implementation.** Train a fixed basis $\{Q^{\gamma_i}\}_{i=1}^{n_\gamma}$ at discounts $\gamma_1 < \dots < \gamma_{n_\gamma}$ (e.g., $\{0.5, 0.8, 0.9, 0.95, 0.99\}$), each an affine projection off a shared trunk, each updated with its own TD target. Head 1 FiLM-modulates the shared trunk on $c$. **Head 2 emits the weighting** $w(\gamma; c) \in \mathbb{R}^{n_\gamma}$ combining the heads at decision time:

$$Q_{\text{eff}}(s, a; c) = \sum_{i=1}^{n_\gamma} w(\gamma_i; c)\, Q^{\gamma_i}(s, a).$$

Head 2's output is *the shape of the discount function*, not a scalar. Constrain $w \geq 0$ via softplus or softmax. By Lemma 5.1 this is equivalent to a context-conditioned discount $d(t; c) = \sum_i w(\gamma_i; c) \gamma_i^t$ — fully shape-adaptive.

**Why creative.** This is much more expressive than Candidate 1. Candidate 1 shifts the agent along the *exponential* family ($\gamma = 0.9 \to \gamma = 0.99$); Candidate 2 moves the agent *across* discount-function shapes. The neuroscientific story: at baseline, the modulator emits a sharp $w$ concentrated near $\gamma = 0.99$ (exponential discounting); under sustained nociceptive load, $w$ broadens with mass at smaller $\gamma$, producing a hyperbolic-flavored curve (steep early decay, long tail). This directly instantiates Sozou's framing: $w(\gamma; c)$ *is* the posterior over hazard rates, sharper when safe, broader when uncertain. Preference-reversal phenomena fall out for free in the broad-$w$ regime.

A built-in fallback: even if Head 2's aggregation is decorative, the multi-horizon heads themselves function as a Fedus auxiliary task on the FiLM-modulated trunk — guaranteed representation-learning lift independent of whether the shape modulation does real work. The three-way ablation reveals which: (a) fixed-$\gamma$ baseline; (b) multi-horizon auxiliary with fixed-$\gamma$ behavior (Fedus floor); (c) Head-2-emits-$w(\gamma; c)$ shape modulation. If (c) > (b) on chronic-pain-analog windows, the shape modulation is load-bearing.

**References.** Fedus et al. 2019 — *Hyperbolic Discounting and Learning over Multiple Horizons* (arXiv:1902.06865; the integral identity, Lemma 5.1, the multi-horizon auxiliary finding). Romoff et al. 2019 — *Separating Value Functions across Time-Scales (TD(Δ))* (ICML; the cascade-of-differences alternative — could supply variance-reduction tricks for training the basis heads). Sozou 1998 — *On hyperbolic discounting and uncertain hazard rates* (provides the hazard-prior framing). Review: [Gamma_lit_review_B_multi_horizon.md](../../docs/project/references/Gamma/Gamma_lit_review_B_multi_horizon.md).

### Candidate 3: γ-Models — generative dynamics conditioned on γ  (recommended)

**Algorithm intro.** Janner, Mordatch & Levine 2020 (*γ-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction*, NeurIPS) introduced a model class that sidesteps the rollout machinery of standard model-based RL entirely. Instead of predicting "the next state given this state and action" — a one-step predictor whose error compounds when rolled out — a γ-Model predicts the **discounted state-occupancy distribution**:

$$\mu_\gamma^\pi(s' \mid s, a) = (1 - \gamma) \sum_{\Delta t = 1}^\infty \gamma^{\Delta t - 1}\, p(s_{t + \Delta t} = s' \mid s_t = s, a_t = a, \pi).$$

Plain English: rather than predicting where you'll be one step from now, predict the *distribution of all future states*, geometrically weighted by how soon you get there. With probability $(1-\gamma)$ you sample where you'll be one step ahead, $\gamma(1-\gamma)$ two steps ahead, $\gamma^2(1-\gamma)$ three steps ahead, and so on. One feedforward pass of a γ-Model returns one sample from this infinite-horizon distribution, with timestep marginalized out. The model is trained by a TD-style bootstrap on the recursion $\mu_\gamma = (1-\gamma) p(s' \mid s, a) + \gamma \mathbb{E}[\mu_\gamma(s' \mid s_{t+1})]$ — same Bellman skeleton as Q-learning, but with a *distribution* in place of a scalar return. The value function pops out as one extra forward pass: $V^\pi(s) = \frac{1}{1-\gamma}\, \mathbb{E}_{s_e \sim \mu_\gamma^\pi}[r(s_e)]$.

Critically, the γ-Model carries **dynamics**, not just value. It tells the agent what futures it expects to inhabit — the imagination horizon is set by $\gamma$. At $\gamma = 0$, it's a one-step world model; at $\gamma \to 1$, it's the full successor representation (where you'll spend all your time).

**Proposed implementation.** Train a γ-Model conditioned on $\gamma$ — call it $\mu_\theta(s_e \mid s, a, \gamma)$. Head 1 of the modulator FiLM-modulates the γ-Model's hidden activations on $c$ (the conditioning channel). **Head 2 emits $\gamma_B(c)$** which is fed as the discount argument to the γ-Model. The agent's imagined-future distribution then becomes:

$$\mu_\theta(s_e \mid s, a, \gamma_B(c), c) = (1 - \gamma_B(c))\, p(s_e \mid s, a) + \gamma_B(c)\, \mathbb{E}[\mu_\theta(s_e \mid s', \gamma_B(c), c)].$$

Operational details from Janner: train with the density-evaluable / normalizing-flow variant $\mathcal{L}_2$ (more stable than the GAN variant at high $\gamma$); use a delayed target network $\mu_{\bar\theta}$. Janner's **Theorem 1** gives a powerful trick: a γ-Model trained at one $\gamma_{\min}$ can be **rolled out and reweighted** to any $\tilde\gamma \geq \gamma_{\min}$ via analytic weights $\alpha_n$. So train one γ-Model at low $\gamma_{\min}$ (e.g., $0.5$); let Head 2 emit context-dependent $\tilde\gamma$ at inference; auto-regressively roll out and weight by $\alpha_n$ to recover $\mu^\pi_{\tilde\gamma(c)}$. Cheap, principled, single-network.

**Why creative.** This is the most distinctive of the three candidates. The γ-Model is the only one that gives the agent a **context-conditioned imagination horizon**: high modulator output ("safe") imagines far-future state distributions; low output ("threatened") imagines tight near-future distributions. This is not just value reweighting — it's a structural change to *what futures the agent simulates*. For a project also exploring Dreamer-style world models ([FiLM_ENSEMBLE_SENSORY_PRECISION.md](../../docs/develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md)), this candidate sits naturally on top of an existing world-model substrate — the γ-Model complements DreamerV3's imagination rollouts, with the modulator setting the horizon.

The clinical reading is the cleanest of the three: chronic pain narrows the patient's imagined-future window. In a γ-Model framework, a context-conditioned $\gamma_B(c)$ implements exactly that — contraction of the predictive horizon under nociceptive input. The agent doesn't just value the future less; it imagines less of it. This is structurally what Sozou predicted and what the active-inference literature calls contraction of the temporal prior. Two follow-on connections: (a) Janner's Proposition 1 shows γ-Models recover the normalized **Successor Representation** at the global minimum, so a context-conditioned γ-Model is a context-conditioned SR; (b) γ-Models decouple dynamics from reward — a γ-Model trained once supports value estimation under any reward function on the same MDP, useful for counterfactual analyses ("what would this agent do under modified pain signaling?") without retraining the dynamics.

The risk: γ-Models are the highest-cost to implement (GAN or normalizing flow, both nontrivial in JAX/Flax), and Janner's experiments are explicitly low-dimensional ($\leq 11$-D state). Scaling to the project's recurrent / POMDP setting is open. But the empirical payoff and structural neuroscientific narrative make this the recommended deeper-exploration candidate.

**References.** Janner, Mordatch & Levine 2020 — *γ-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction* (NeurIPS; the generative TD operator, Theorem 1 rollout-reweighting, γ-MVE). Barreto et al. 2017 — *Successor Features for Transfer in Reinforcement Learning* (NeurIPS; the structural parent — γ-Models are the generative continuous analogue). Sozou 1998 — *On hyperbolic discounting and uncertain hazard rates* (hazard-prior interpretation). Sutton 1995 — *β-models* and Sutton et al. 1999 — *Options framework* (earlier mixture-time-scale precedents). Review: [Gamma_lit_review_C_generative_gamma.md](../../docs/project/references/Gamma/Gamma_lit_review_C_generative_gamma.md).

### What I'd try first

If I had one experiment-cycle to allocate, I'd run the three-way ablation against the cheapest path first: **(a)** fixed-$\gamma$ baseline (no Head 2 on discount), **(b)** multi-horizon auxiliary heads at $\gamma \in \{0.5, 0.8, 0.9, 0.95, 0.99\}$ with a single-$\gamma$ behavior policy (the Fedus floor — guaranteed representation lift), **(c)** Candidate 1's γ-conditional UVFA with Head 2 emitting a single scalar $\gamma_B(c)$. This gives an immediate read on whether the modulator can do real horizon work, with calibrated-effective-horizon as the falsifier (perturb reward at delay $k$, slope of $\log|\Delta V|$ matches $\log \gamma_B(c)$ per-context). If (c) > (b) cleanly on chronic-pain-analog contexts, escalate to Candidate 3 (γ-Models). If (c) ≈ (b), the modulator is decorative on this knob — fall back to multi-horizon auxiliary as a free representation-learning upgrade, and consider Candidate 2's shape modulation as a higher-capacity alternative. The Schultheis et al. 2022 continuous-time HJB framework ([Gamma_lit_review_D_nonexponential.md](../../docs/project/references/Gamma/Gamma_lit_review_D_nonexponential.md)) is the theoretical anchor for what any of these candidates is approximating, and is worth citing in the eventual paper even if not directly implemented.

## 7. Cross-cutting synthesis

The four professor sections were written independently against four different lit-review corpora, but a single architectural-honesty axis runs across all of them — and once you see it, the v1 compact memo's "two-headed modulator" reading becomes the *floor*, not the ceiling.

**The unifying claim.** Across all four knobs, the same architecture recurs: Head 1 conditions the agent's forward pass (FiLM scales/shifts on activations), Head 2 reaches into the loss / return target. The recurrence is not an accident — it falls out of *where* in the computation graph each hyperparameter lives. Temperature lives at the policy logits, learning rate lives in the optimizer step, RPE gain lives on the advantage, discount lives in the bootstrap target. Heads 1 and 2 are the two natural reach-points; everything else is a composition of those two. **But the richest candidate per knob is almost never the scalar version of Head 2.** It is a *vector* (DA emits a reward-weight vector for successor features; ACh emits per-pathway γ via the chain rule), a *function* (5-HT emits a weighting $w(\gamma; c)$ over a horizon basis), or full *weights* (DA-as-hypernet writes the critic; ACh-as-hypernet writes the critic via Jacobian pullback). Scalar Head 2 is what makes each knob *implementable*; vector or functional Head 2 is what makes each knob *identifiable* — i.e., distinguishable from the global learning rate and from a feature-conditioning shortcut.

**The shared architectural-honesty axis.** All four sections place their candidates on the same axis. At one extreme: a single scalar from Head 2 (basic FiLM γ at logits for NA; scalar $\kappa$ for DA; scalar $\gamma_B$ for 5-HT; scalar $\alpha$ for ACh). At the other extreme: full hypernetwork weight emission (Sarafian-style critic generation, recommended for DA; ACh-as-hypernetwork via Jacobian pullback). In between sit the *structured* outputs: successor-feature reward vectors, IQN distortion parameters, mellowmax-consistent state-dependent temperatures, γ-conditional UVFAs, and Fedus horizon mixtures. The honesty axis sorts proposals by how cleanly they escape the gauge ambiguities Doya's original cartoon leaves open. **Scalar Head 2 is gauge-ambiguous with the learning rate; vector / function / weight outputs are not.** This is the deepest finding the four sections share — and it tells us that "implement Doya in code" is genuinely harder than the cartoon suggested, because the cartoon's identifiability is bought by the very simplicity that makes the scalar version weak.

**Comparison table — recommended first prototype per knob.**

| Knob | Section's "what I'd try first" | Head 2 output shape |
|---|---|---|
| NA (temperature) | Basic FiLM-γ-at-logits, instrumented for Munchausen extension | Single positive scalar (broadcast on logits) |
| ACh (learning rate) | Hybrid of FiLM γ at hidden layer + meta-gradient α(c) | Scalar α(c) emitted by Head 2 + γ vector via Head 1 |
| DA (RPE gain) | Hypernet critic with optional ablation to successor features | Critic weights $\theta_Q(c)$ (high-dim vector) |
| 5-HT (discount) | Three-way ablation: fixed γ vs multi-horizon auxiliary vs γ-conditional UVFA | Scalar $\gamma_B(c)$ in cheapest version; weight vector $w(\gamma; c)$ at higher capacity |

The table makes the asymmetry explicit. NA's recommended prototype lives at the simplest end (single scalar) because the gauge ambiguity is already mild for temperature — temperature acts on the forward pass at a unique location and cannot be absorbed into the learning rate. DA's recommended prototype lives at the most expressive end (hypernet weights) because the gauge ambiguity for scalar $\kappa$ is severe — only a high-dimensional output cleanly breaks it. ACh and 5-HT sit in between, with their richness coming from the structural complement (per-pathway gating + global α for ACh; horizon-axis conditioning for 5-HT).

**The four-knob unification.** What becomes possible if all four knobs are implemented together with the *same* modulator trunk? This is the original NMN dream: one small network ingests interoceptive context $c$, and from a shared trunk emits a structured output that simultaneously controls (i) the policy's exploration temperature, (ii) per-pathway plasticity gates and a global learning-rate scalar, (iii) the critic's weight tensor (or successor-feature reward vector, or risk-distortion parameter), and (iv) the discount or horizon-mixture. Because all four targets share an interoceptive context, the network's *representation of $c$* is amortised across all four roles. The downstream consequence is a strong identifiability claim that no per-knob prototype can make alone: the modulator's hidden representation now has to be *simultaneously consistent* with arousal, plasticity, valuation, and horizon — four constraints on one latent. If those four constraints concentrate on the same low-dimensional structure (and the v1 audit's geometric framing predicts they will), the resulting modulator is the most parsimonious account of a unified neuromodulatory state — far richer than four independent scalar emitters.

**Open creative questions.** The memo surfaces but does not resolve four questions. (1) When does the *gauge-breaking move per knob actually purchase identifiability versus just complexity*? An empirical answer requires the gauge-sweep experiments each section sketches — the project doesn't yet know which knobs need their richest variant and which are fine with scalar Head 2. (2) Is there a single computational principle (information geometry? distributional Bellman operators? successor representations?) that subsumes all four candidates at their richest? Section 5's recommendation (hypernet critic strictly contains SFs and IQN) hints that hypernetworks may be that principle, but the four sections didn't try to unify formally. (3) What is the right loss for *jointly* training all four knobs on one trunk — do they interfere, do they share gradients constructively, or do they need orthogonal subspace projections? (4) Where does Bayesian uncertainty enter — Fedus's $w(\gamma)$ as a posterior over hazard rates, IQN's distortion as a risk preference, the modulator's own emission as a point estimate vs distribution — and is there a unified Bayesian reading of the four-knob unification? These are the questions a follow-up could send to `professor-bayesian-nn` and `professor-bayesian-brain`.

## 8. Pointers

**Companion v1 (compact) memo.** The minimum-viable, audit-anchored version of these four implementation routes lives at [`docs/project/ideas/20260518_doya_modulation_via_film_hypernet_implementation_ideas.md`](20260518_doya_modulation_via_film_hypernet_implementation_ideas.md). Read it first if you want the bare-bones equations and the identifiability tests per knob.

**Underlying audit.** The two-headed architecture is grounded in two prior docs: the technical audit [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](../critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md), and its plain-English story [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_audit_story.md`](../critiques/20260518_film_as_hyperparameter_modulator_audit_story.md). These establish *why* Head 1 + Head 2 is the natural converged form.

**The four lit-review folders.**
- [`docs/project/references/Temperature/`](../references/Temperature/) — SAC foundations, SAC variants, softmax operators, KL / Munchausen. Anchors Section 3 (NA).
- [`docs/project/references/Learning_Rate/`](../references/Learning_Rate/) — L2O foundations, learning-rate adaptation, meta-gradient RL, gated conditional architectures. Anchors Section 4 (ACh).
- [`docs/project/references/TD/`](../references/TD/) — distributional RL, successor features, hypernet RL, Decision Transformer. Anchors Section 5 (DA).
- [`docs/project/references/Gamma/`](../references/Gamma/) — universal value functions, multi-horizon, generative γ, non-exponential discounting. Anchors Section 6 (5-HT).

Each folder contains four review files (one per sub-theme) plus a `sources/` directory with the underlying PDFs. The professor sections cite review files by canonical filename (e.g. `Temperature_lit_review_C_softmax_operators.md`); follow those filenames into the relevant folder to read the per-paper backbone.

**Next-step routing.** Recommendations that imply runnable experiments (the per-knob "What I'd try first" headlines, the three-way 5-HT ablation, the DA gauge sweep) should be handed to `experiment-designer`; recommendations that imply architectural code should be handed to `senior-developer`. The four-knob unification (Section 7) is upstream of either — it is a research-direction question best routed back through `pi` before any code or experiment commits.
