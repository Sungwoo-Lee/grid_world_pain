---
title: "FiLM and hypernetworks as a unifying framework for neuromodulatory-inspired algorithms — the potential to integrate"
status: draft
audience: user, pi, experiment-designer, senior-developer, math-reviewer
last_updated: 2026-05-16
classification: concept-memo
type: paper-shape standalone theoretical contribution
contribution_claim: "FiLM/hypernet family as mathematically-articulated taxonomy covering ~12/18 load-bearing neuromodulation-inspired algorithms, with 3 honestly named no-map cases as the substrate boundary."
target_behaviours:
  - context-switching / regime change (NA-style)
  - hypervigilance / sustained tonic vigilance (interoceptive)
  - plasticity gating / lifelong-learning resistance
  - exploration-exploitation control (precision / temperature)
inputs:
  - docs/project/references/FiLM/film_neuromod_integration_synthesis.md
  - docs/project/references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md
related:
  - docs/project/concepts/active_inference_hypervigilance.md
  - docs/project/concepts/pain_vs_nociception_construct.md
---

# FiLM and hypernetworks as a unifying framework for neuromodulatory-inspired algorithms — the potential to integrate

## §1 — Abstract / plain-English entry point

Two literatures have been developing in parallel for roughly two decades. On the deep-learning side, the **FiLM family** ("Feature-wise Linear Modulation" — a class of network operations in which a small side-network watches an input or context signal and produces two vectors, a multiplicative scale $\gamma$ and an additive shift $\beta$, that are applied to the main network's activations as $h \mapsto \gamma \odot h + \beta$) has grown from Perez et al. 2018 into roughly ten operationally distinct variants — vanilla FiLM, conditional batch/instance normalisation, AdaIN, temporal FiLM, full **hypernetworks** (Ha 2016 — small networks that *generate the weights* of a larger network rather than just modulating its activations), Bayesian hypernets, FiLM-Ensemble, BatchEnsemble, sparse mixtures-of-experts, attention-as-gating, and kernel modulation. On the neuroscience side, the **neuromodulator-inspired algorithm family** (Doya 2002 onward — a "neuromodulator" being a chemical signal in the brain like noradrenaline, acetylcholine, dopamine, or serotonin that diffusely changes how other neurons compute) has produced its own family of mechanisms: scaling cellular input-output slopes, shifting firing thresholds, scaling weight matrices uniformly, scaling them low-rank, scaling gradients during learning, hand-coded uncertainty-driven hyperparameters, recurrent-net excitability gain.

These two literatures use **the same mathematical operations under different names**, but no published paper has explicitly enumerated them as a single taxonomy. This memo does that. The headline claim is **not** "every neuromodulator paper IS FiLM in disguise" — that would over-reach. The claim is more careful: **the FiLM/hypernet family forms a mathematically articulated taxonomy that integrates roughly twelve of eighteen load-bearing neuromodulation-inspired algorithms in this project's reference corpus, with three honestly named no-map cases that draw the substrate boundary**. The taxonomy supports four behaviour-level benefits the neuromod literature already claims — fast context-switching after a regime change, sustained tonic vigilance (the project's interoceptive-pain anchor), plasticity gating that attenuates lifelong-learning forgetting, and exploration–exploitation control via precision or inverse temperature.

We make the integration explicit with mathematical bridges, name the gaps the unified picture exposes (six), and hand off the integration mapping (§5), the per-behaviour mathematical signatures (§6), and the substrate-choice rationale (§7) to the downstream design and implementation work. The honest no-map boundary in §5.5 is part of the claim, not a failure of it.

---

## §2 — Introduction and motivation

### §2.1 Two literatures, one operation

The **FiLM family**, as it appears in contemporary deep learning, traces a clean lineage. **Perez et al. 2018** named "Feature-wise Linear Modulation" and showed that an entire visual-reasoning benchmark could be tackled by letting a question-conditioned side network emit a per-channel $(\gamma, \beta)$ and applying $\gamma \odot F + \beta$ to a vision-CNN's feature map. The operation had already appeared, unnamed, in **Dumoulin et al. 2017** ("Conditional Instance Normalisation" — the same $\gamma, \beta$ but after per-example feature standardisation, indexed by a style code) and **Huang & Belongie 2017** ("Adaptive Instance Normalisation", AdaIN — same form, with $(\gamma, \beta)$ replaced by closed-form feature statistics of a second image). It had appeared in a different operator position — on weights rather than activations — in **Ha, Dai & Le 2016** ("HyperNetworks" — a small network that generates a larger network's weights). Subsequent work has produced ten or so operationally distinct relatives: time-varying FiLM whose conditioner is a recurrent network over the network's own pooled state (**Birnbaum et al. 2019**); multi-task FiLM with a continuous task vector (**Takeda et al. 2021**); FiLM-Ensemble — FiLM-Ensemble uses a discrete member index instead of a continuous code to produce $M$ different $(\gamma^m, \beta^m)$ per layer (**Turkoglu et al. 2022**); BatchEnsemble — per-member rank-1 weight modulation (**Wen et al. 2020**, used by **Gorishniy et al. 2025** as TabM); sparsely-gated mixtures-of-experts (**Shazeer et al. 2017**); attention as gating (**Vaswani et al. 2017**); per-weight Kernel Modulation that lifts FiLM's per-channel restriction (**Abdollahzadeh et al. 2021**).

The **neuromodulation-inspired algorithm family**, as it appears in computational neuroscience and biologically-inspired RL, traces a similarly clean lineage. **Doya 2002** proposed the foundational four-knob mapping: dopamine ↔ TD error, acetylcholine ↔ learning rate, noradrenaline ↔ inverse temperature, serotonin ↔ discount factor. **Vecoven et al. 2020** wrote the first explicit "neuromodulatory subnetwork" for meta-RL: a scalar $z$ from a side network produces per-feature $w_s, w_b$ in the main net's activation function. **Tsuda et al. 2021** showed that a scalar multiplicative gain on the weight matrix, $W \to f_{nm} \cdot W$, suffices to produce dissociable "hypertube" trajectories through activity space. **Costacurta et al. 2024** generalised this to a low-rank weight scaling. **Rodriguez-Garcia et al. 2026** moved the gain into the optimiser, producing a state-dependent effective learning rate via gradient reparametrisation. **Lee et al. 2024** hand-coded functional forms — $\alpha(s,a) = E/(E+A)$, $\beta^{-1}(s) = 1/\langle E(s,\hat a)\rangle$ — for the Doya knobs in terms of ensemble-derived expected and unexpected uncertainty. **Wainstein et al. 2025** showed that a trained-RNN's post-hoc activation gain $g$ correlates with pupil diameter and triggers perceptual switches. **Mei et al. 2022** synthesised the entire enterprise as "neuromodulation = DNN hyperparameter fine-tuning".

### §2.2 Why this is worth doing now

These two literatures cite each other rarely. The FiLM corpus does not cite Doya 2002; the neuromod corpus cites Vecoven 2020 and Lee 2024 but not Perez 2018 or Ha 2016 directly. Yet the operator forms collide constantly. Vecoven 2020's $\sigma_{NMN}(x, z) = \sigma\bigl((z^\top w_s) x + (z^\top w_b)\bigr)$ is, after a one-line expansion, vanilla FiLM with the FiLM-generator being a single linear read-out of $z$. Tsuda 2021's $W \to f_{nm} \cdot W$ is HyperNetwork (Ha 2016) restricted to a single scalar output. Lee 2024's hand-coded mapping is AdaIN's "closed-form $(\gamma, \beta)$ from feature statistics" idea applied to ensemble-uncertainty statistics instead of style-image statistics. Ben-Iwhiwhu 2022's activity-gating $h \to g(z) \odot h$ is vanilla FiLM with the additive arm clamped to zero.

Naming the integration explicitly does three things the present literature does not. It **provides** a unified mathematical vocabulary for what biologically-inspired-RL practitioners call "neuromodulation" and DL practitioners call "conditional modulation" — so that, for example, a benefit demonstrated in the FiLM literature (FiLM-Ensemble's cheap epistemic uncertainty at 1.3% parameter overhead, Turkoglu 2022) can be ported into a neuromod algorithm cleanly and vice versa. It **enumerates** what does and does not fit, so a downstream agent or experiment designer can pick the smallest FiLM-variant that supports the behaviour they want. And it **draws the substrate boundary** — three corpus papers (Rodriguez-Garcia 2026, Wainstein 2025, Osman 2024) operate at points in the optimisation/dynamics loop that no FiLM-variant covers, and naming these is essential because they are the strongest *complementary* mechanisms the integration claim must position against rather than absorb.

### §2.3 The headline claim, sharpened

The claim is that **the FiLM/hypernet family is the appropriate mathematically-articulated taxonomy for the *forward-pass* slice of the neuromodulation-inspired algorithm family**. Of eighteen load-bearing rows in the project's neuromod corpus, seven map cleanly onto a FiLM variant, five map with extension (the operator-position shifts, or the variant has a closed-form rather than learned generator, or the conditioner is structured-discrete rather than continuous), three map only partially (the paper supplies framing rather than a single mechanism), and three do not map (one operates at gradient level; one's gain is a post-hoc analytical knob, not a side-signal-conditioned operator; one operates on a recurrent attractor energy landscape). The seven + five = twelve cleanly-or-with-extension fitted rows constitute the empirical base for the claim that the FiLM-variant family **does** unify a substantial portion of neuromodulation-inspired algorithms. The three no-map rows constitute the substrate boundary: gradient-level, internal-trained-RNN-gain, and recurrent-energy-landscape gain are *complementary* mechanisms a FiLM substrate must name and not silently absorb.

---

## §3 — The FiLM-variant taxonomy

This section enumerates the operationally distinct FiLM-variants we will map onto in §5. Each variant has a canonical equation, a canonical paper, an operator position (activations, weights, gradient, attention, normalisation statistics), and a behaviour-level benefit demonstrated in the source paper. The variants are not mutually exclusive — AdaIN is CIN with a closed-form generator, FiLM-Ensemble is CIN with a discrete member-index conditioner, BatchEnsemble is FiLM one operator-position over (weights, not activations), Sparse MoE is a discrete hypernetwork. The taxonomy is a partial order, not a partition.

### §3.0 — Lineage callout: the family hierarchy

Before enumerating individual variants in §3.1–§3.12, this callout states the five formal lineage relations among the four families that §3.1–§3.12 collectively span: vanilla FiLM and its activation-level cousins (§3.1–§3.4), Hypernetworks (§3.5), Bayesian Hypernetworks (§3.6), Bayesian Neural Networks (a peer cell, not enumerated here but referenced in §5.4 Row EE-6 as the Kendall-Gal anchor), and FiLM-Ensemble / BatchEnsemble (§3.7–§3.8). The relations are derived in full in the follow-up investigation memo ([film_hypernet_bnn_lineage_and_critic_modulation.md §2](film_hypernet_bnn_lineage_and_critic_modulation.md)); here we summarise them so the §5 mapping inherits a clean ancestry.

**Claim 1 (FiLM ⊂ Hypernet — *rigorous*).** A FiLM layer (per-channel affine on activations) is a hypernetwork whose generated "weights" are restricted to per-channel diagonal scaling plus a constant translation. Independent routes: (i) Ha 2016 §3.2's scaling-vector trick produces FiLM at per-row granularity two years before FiLM was named (cited at [§3.5](#§35-hypernetwork-ha-dai--le-2016) below); (ii) Abdollahzadeh 2021 (this memo §3.11) shows vanilla FiLM is equivalent to per-output-channel-uniform kernel modulation, the most degenerate kernel-modulation in the KML hierarchy.

**Claim 2 (Hypernet ⊆ Bayesian-Hypernet-with-point-mass — *rigorous as a limit*).** A deterministic Hypernetwork is the limiting case of a Bayesian Hypernetwork (BHN; §3.6) whose implicit posterior over weights collapses to a point mass: $q_\phi(W \mid c) = \delta(W - g_\phi(c))$. This is a *limiting* inclusion, not a sub-architecture obtained by output-space restriction — calling a deterministic hypernet "a BHN" without the "as a limit" qualifier would over-state the lineage.

**Claim 3 (BNN ⊥ BHN — *rigorous, orthogonal*).** A Bayesian Neural Network places a posterior $p(W \mid \mathcal{D})$ on the main network's *leaf* weights *without* context-conditioning. A Bayesian Hypernetwork places a posterior on the *generator*, inducing a context-conditional $q_\phi(W \mid c)$ over main-network weights. The two architectures place uncertainty at different levels and are not nested. They can in principle combine — a hierarchical architecture with both posteriors active — but no paper in either of the project's corpora does so. The 2×2 taxonomy of {deterministic, stochastic} × {unconditional, context-conditional} weights has Standard-NN, Hypernet, BNN, BHN at its four corners respectively; vanilla FiLM lives in the Hypernet cell.

**Claim 4 (FiLM-Ensemble ≈ finite-mixture BHN approximation — *holds with caveat*).** Turkoglu 2022's FiLM-Ensemble (§3.7) produces $M$ deterministic settings of $(\gamma^m, \beta^m)$ and averages predictions. If we treat the member index as a categorical latent with $q(\gamma, \beta) = \frac{1}{M}\sum_m \delta(\gamma - \gamma^m)\delta(\beta - \beta^m)$, the ensemble's predictive distribution is a Monte Carlo estimate of $\mathbb{E}_q[\sigma(f_{\theta,\gamma,\beta}(x))]$ — same within-/between-member variance decomposition as Kendall & Gal 2017 Eq. 7. *Caveat*: members are trained jointly under one objective rather than sampled from a learned posterior $q_\phi$; there is no KL-to-prior; the $\rho$-gain init controls spread heuristically. Honest framing: **finite-mixture approximation of a posterior over FiLM scalars**, not a clean BHN.

**Claim 5 (Galanti & Wolf 2020 Thm. 4 modularity carries through — *rigorous*).** The hypernetwork modularity bound — $N_g = O(\epsilon^{-m_1/r})$ parameters for a hypernet to approximate any target in $W^{r,m}$ to error $\epsilon$, vs $\Omega(\epsilon^{-(m_1 + m_2)/r})$ for embedding-concatenation — applies to every member of the family. FiLM (Claim 1), deterministic-Hypernet, BHN (Claim 2's limit), and FiLM-Ensemble (Claim 4's approximation) all inherit the same parameter-efficient capacity allocation. This is the formal reason FiLM/hypernet outperforms a "just concatenate the context vector" baseline.

**Re-framing of §5.4 Row EE-6.** Under Claim 4, the memo's "FiLM-Ensemble + Kendall-Gal heteroscedastic precision compound" in Row EE-6 is more precisely framed as **the first instantiation of an approximate Bayesian Hypernetwork with a heteroscedastic likelihood for an RL policy**: cleanly positioned against Krueger 2017 (full BHN; no heteroscedastic head; not RL) and Kendall & Gal 2017 (BNN; no context-conditioning; not RL). The compound's novelty survives the lineage; what it gains is honest ancestry — every component has clear ancestors (Hypernet ← Ha 2016; BHN ← Krueger 2017; heteroscedastic likelihood ← Kendall & Gal 2017; FiLM-Ensemble approximation ← Turkoglu 2022), and their combination on an RL policy is new.

The §3.0 callout closes the BNN-as-peer gap in the §3 enumeration: this memo originally listed BHN (§3.6) without explicitly stating its relationship to BNN, and listed FiLM/Hypernet without explicitly stating the FiLM ⊂ Hypernet inclusion. §3.0 fixes both omissions in compact form. The detailed derivations live in the investigation memo §2.

### §3.1 Vanilla FiLM — per-channel affine on activations (Perez et al. 2018)

$$
\gamma_{i,c} = f_c(x_i),\quad \beta_{i,c} = h_c(x_i),
\qquad \mathrm{FiLM}(F_{i,c} \mid \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}.
$$

Where it acts: forward-pass activations, per channel. Demonstrated benefit: question-conditioned visual reasoning; (γ, β) parameter space is linearly composable for zero-shot attribute composition.

### §3.2 Conditional Instance Normalisation (Dumoulin et al. 2017)

$$
\mathrm{CIN}(x \mid s)_{n,c,h,w} = \gamma_{s,c} \cdot \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}} + \beta_{s,c},
$$

with $(\gamma_{s,c}, \beta_{s,c})$ a learned per-style lookup. Where it acts: activations, after per-example feature standardisation. Demonstrated benefit: 32 image styles in one network at 0.2% style-specific parameters.

### §3.3 Adaptive Instance Normalisation (Huang & Belongie 2017)

$$
\mathrm{AdaIN}(x, y) = \sigma^{\mathrm{IN}}(y) \cdot \frac{x - \mu^{\mathrm{IN}}(x)}{\sigma^{\mathrm{IN}}(x)} + \mu^{\mathrm{IN}}(y).
$$

Same shape as CIN but with $(\gamma, \beta)$ computed *closed-form* from a second image's feature statistics — zero learnable parameters in the modulation layer. Where it acts: activations, after standardisation. Demonstrated benefit: real-time arbitrary style transfer, including unseen styles at test time. **This is the formal precedent for "hand-coded modulator output" as a FiLM-family member** (load-bearing for §5 Row 3).

### §3.4 Temporal FiLM (Birnbaum et al. 2019)

A block-piecewise-constant $(\gamma_b, \beta_b)$ generated by an LSTM that observes the network's own pooled block-statistics:

$$
\bigl((\gamma_b, \beta_b), h_b\bigr) = \mathrm{LSTM}\bigl(F^{\mathrm{pool}}_{b,:}; h_{b-1}\bigr),
\qquad F^{\mathrm{norm}}_{b,t,c} = \gamma_{b,c} \cdot F^{\mathrm{blk}}_{b,t,c} + \beta_{b,c}.
$$

Where it acts: activations, time-block piecewise-constant; conditioner is self-recurrent (the LSTM observes the network's own pooled state). Demonstrated benefit: long-range sequence dependency at low compute cost; the *self-recurrent conditioner* template the project's modulator inherits.

### §3.5 HyperNetwork (Ha, Dai & Le 2016)

Full weight-generation:

$$
K^j = g(z^j) = \langle W_{\mathrm{out}}, W_i z^j + B_i \rangle + B_{\mathrm{out}}.
$$

Dynamic (HyperRNN) form uses a per-timestep embedding $z$. The memory-efficient "scaling-vector trick" replaces the dense weight tensor with per-row scaling — *structurally identical to FiLM at per-row granularity, two years before FiLM was named*. Where it acts: weight matrices. Demonstrated benefit: HyperLSTM beats LSTM on character-level language modelling, IAM handwriting, and WMT'14 En→Fr translation; weights change abruptly at word/phrase boundaries.

**Theoretical anchor.** Galanti & Wolf 2020 prove a modularity bound — for a hypernetwork approximating a target function family of complexity $W^{r,m}$, the parameter count is $N_g = O(\epsilon^{-m_1/r})$ versus $N_q = \Omega(\epsilon^{-(m_1 + m_2)/r})$ for naive embedding-concatenation. **Capacity is absorbed into the hypernetwork exponentially in the conditioning dimension.** This is the formal reason FiLM/hypernet outperforms a "just concatenate the context vector" baseline.

### §3.6 Bayesian Hypernetwork (Krueger et al. 2017)

Primary weights are sampled via an invertible normalising flow:

$$
\theta = h(\epsilon),\qquad \epsilon \sim \mathcal{N}(0, I_D), \qquad \log q(\theta) = \log q_\epsilon(\epsilon) - \log \bigl| \det \tfrac{\partial h(\epsilon)}{\partial \epsilon} \bigr|.
$$

Where it acts: weights (or, in the efficient parametrisation, only weight-norm scaling factors). Demonstrated benefit: full multi-modal correlated posterior over weights; better adversarial-example detection and active learning than mean-field VI / MC-dropout / deterministic baselines.

### §3.7 FiLM-Ensemble (Turkoglu et al. 2022)

Replace the continuous conditioner with a discrete member index $m \in \{1, \ldots, M\}$:

$$
\mathrm{FiLM}\!\bigl(F_n \mid \gamma^m_n, \beta^m_n\bigr) = \gamma^m_n \circ F_n + \beta^m_n.
$$

Where it acts: activations at every BatchNorm layer; conditioner is a member index applied $M$ times to the same input in parallel. Demonstrated benefit: cheap epistemic uncertainty — 16-member implicit ensemble at 1.3% parameter overhead vs. 1500% for an explicit ensemble of ResNet-18; *higher* member diversity than the explicit ensemble at $M=16$.

### §3.8 BatchEnsemble / TabM (Wen et al. 2020; Gorishniy et al. 2025)

Per-member rank-1 modulation of a shared weight matrix:

$$
W_i = W \odot (s_i\, r_i^\top), \qquad l_i(x_i) = s_i \odot \bigl(W (r_i \odot x_i)\bigr) + b_i.
$$

Where it acts: around the linear-layer weight (one operator-position over from FiLM). Demonstrated benefit: parameter-efficient implicit ensembling for tabular data; TabM Pareto-dominates 46-dataset benchmarks.

### §3.9 Sparsely-Gated Mixture-of-Experts (Shazeer et al. 2017)

$$
y = \sum_{i=1}^n G(x)_i \cdot E_i(x), \qquad G(x) = \mathrm{Softmax}(\mathrm{KeepTopK}(H(x), k)).
$$

Where it acts: discrete routing across expert sub-networks; the gate is a discrete-sparse hypernetwork. Demonstrated benefit: trillion-parameter models with bounded per-example compute; 24% lower perplexity than computational-budget-matched baseline.

### §3.10 Attention as gating (Vaswani et al. 2017)

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Bigl(\tfrac{Q K^\top}{\sqrt{d_k}}\Bigr) V.
$$

Where it acts: forward-pass positions (not channels). Empirical caveat from the FiLM corpus (Vuorio et al. 2019, cited in Turkoglu 2022 and Abdollahzadeh 2021): "FiLM outperforms attention-based modulation in this context [multimodal MAML] and is more stable" — when the side signal is low-dimensional, FiLM wins.

### §3.11 Kernel Modulation (Abdollahzadeh et al. 2021) — and the reinterpretation lemma

Replace per-channel scalar $\gamma$ with per-weight modulation matrix $M^l$ of the kernel's shape:

$$
\hat W^l_T = W^l \odot \bigl(J + M^l(\upsilon_T, \phi)\bigr), \qquad M^l(\upsilon_T, \phi) = g^l_{\phi_1}(\upsilon_T) \otimes g^l_{\phi_2}(\upsilon_T),
$$

with $J$ the all-ones matrix and the outer-product factorisation $\otimes$ keeping the generator small even when $M^l$ has the full kernel shape.

**Load-bearing reinterpretation lemma.** Abdollahzadeh's §3 shows — and this is the mathematically central result of this entire memo — that vanilla FiLM applied to a conv layer is **equivalent to per-channel-uniform kernel modulation**:

$$
\text{FiLM}(\text{Conv}(x)_c) = \gamma_c \cdot \text{Conv}(x)_c + \beta_c
\;\equiv\;
\text{Conv}(x; W'_c = \gamma_c W_c)_c + \beta_c,
$$

where $W_c$ is the $c$-th output-channel slice of the kernel and every entry in $W_c$ is scaled by the same scalar $\gamma_c$. **So vanilla FiLM is itself a degenerate kernel-modulation** — one degree of freedom per output channel per modulation. KML lifts this restriction to per-weight modulation.

The further degenerate restriction — every weight in the *entire* kernel scaled by *one* scalar — is Tsuda 2021's $W \to f_{nm} \cdot W$:

$$
\hat W = W \odot (J + (f_{nm} - 1) J) = f_{nm} \cdot W.
$$

So FiLM and Tsuda 2021 are **nested restrictions of a single kernel-modulation hierarchy**: Tsuda (one scalar for the kernel) $\subset$ vanilla FiLM (one scalar per output channel for the kernel) $\subset$ KML (one scalar per weight in the kernel). This is the load-bearing math result that sharpens the v3 / predictions-synthesis Tsuda framing — they are not independent mechanisms.

### §3.12 What we exclude from the FiLM family

**Symbolic-discrete neural module networks** (Andreas et al. 2016; Hu et al. 2017) — discrete composition of an entirely-new computation graph per input — is a *fork* of the FiLM family, not a member. We flag this so we do not silently absorb NMN into the integration.

---

## §4 — Neuromodulatory algorithms and their behaviour-level benefits

The neuromod corpus's load-bearing extraction lives in the predictions synthesis (the project's prior memo). Here we summarise, for each of the **four target behaviours** the integration mapping in §5 will be organised around, the corpus papers that demonstrate that behaviour and the cellular mechanism each invokes.

### §4.1 Context-switching / regime change

After a regime change (the environment switches task, distribution, or reward structure), a successful agent must rapidly detect the change and re-tune. The neuromod corpus offers several mechanisms. **Xing 2022** runs a context-conditioned modulator triggered by an environment-change detector and reports faster post-switch recovery with less catastrophic forgetting. **Ben-Iwhiwhu et al. 2022** add an activity-gating modulator $h \to g(z) \odot h$ before nonlinearities in CAVIA/PEARL-style meta-RL and show richer cross-task latent representations and faster meta-test adaptation. **Wang 2024 (NeuronML)** uses a per-task structure mask on weights, learned via bi-level optimisation, for fast meta-adaptation. **Wainstein et al. 2025** demonstrates a pupil-locked gain spike at perceptual switches in a trained RNN. **Tsuda et al. 2021** shows that context-dependent scalar weight-rescaling produces dissociable "hypertube" trajectories through activity space.

### §4.2 Hypervigilance / sustained tonic vigilance

The corpus is thin here — sustained tonic vigilance is mostly framed in the pain-modeling literature rather than the neuromod-algorithm literature. The anchors are **Eccleston & Crombez 1999** (attentional vigilance to pain as a defining feature of chronic pain), **Wiech 2016** (top-down expectations sustain pain experience), and **Tabor & Burr 2019**/**Büchel et al. 2014** (precision-weighted prediction error formulations of chronic pain). On the cellular side, **Ferguson & Cardin 2020** (Box 1) provides the multiplicative-vs-additive gain distinction: multiplicative gain (slope change) moves sensitivity, additive bias (rheobase shift) moves selectivity. Channel-selective hypervigilance — high gain on threat-relevant channels, unchanged or lowered gain on threat-irrelevant — requires a per-feature operator, not a scalar global gain. **Doya 2002**'s ACh-on-encoder branch supplies the substrate-level vocabulary (gain on the encoder as effective per-feature learning-rate adjustment).

### §4.3 Plasticity gating / lifelong-learning resistance

**Rodriguez-Garcia et al. 2026** is the clearest published result: a neuromodulator-gated stochastic gradient descent (NGM-SGD) attenuates the *stability gap* — the transient accuracy drop on old tasks at task boundaries — across Split MNIST, Split CIFAR, and mini-ImageNet. The mechanism is gradient-level: $W_{ij}(t) = g_i(t) w_{ij}(t)$ with $g$ tracked by a slow filter that responds to the network's output entropy. **Kudithipudi et al. 2022** surveys the biological mechanisms (synaptic-tagging, neurogenesis, neuromodulator-gated plasticity) and frames them as templates for lifelong-learning ML. **Costacurta et al. 2024** demonstrates dissociable-by-ablation multi-channel modulation (Fig. 3F) — different latent dimensions of the modulator $z$ are dedicated to different sub-computations; lesioning each produces a different behavioural deficit. **Lee et al. 2024** shows recovery in non-stationary multi-armed bandits with hand-coded uncertainty-driven hyperparameters. **Durstewitz et al. 2025** (cited by the project's predictions synthesis) supplies the dynamical-systems framing.

### §4.4 Exploration–exploitation control

**Doya 2002** §3.3 maps noradrenaline ↔ inverse temperature $\beta$ (high NA → low entropy → exploitation; low NA → high entropy → exploration). **Lee et al. 2024** sharpens this into a hand-coded functional form: $\beta^{-1}(s) = 1/\langle E(s,\hat a)\rangle$ where $E$ is aleatoric uncertainty from a value ensemble. **Wainstein et al. 2025** logs the pupil-locked gain-on-perceptual-switch chain. **Ferguson & Cardin 2020**'s multiplicative arm is the cellular substrate. **Shine 2021** formally defines neural gain as $g = dQ/dI$, the slope of a neuron's input-output mapping function. **Yu & Dayan 2005** (out of the project's primary corpus but cited repeatedly) maps acetylcholine to expected uncertainty / sensory precision — an ACh branch the project chose against (it picked Doya's $\alpha$-mapping) but which the integration mapping must acknowledge as a contested interpretation.

---

## §5 — The integration mapping

This is the load-bearing section. We reorganise the curator synthesis's eighteen rows by behaviour category. For each behaviour, we present (a) a table of (Neuromod algorithm, FiLM-variant, Mathematical bridge, Tag), (b) a per-row mathematical bridge in LaTeX, and (c) a mathematical sketch box for the cleanest claim per behaviour. The four tag levels — **CLEAN MAP**, **EXTENSION MAP**, **PARTIAL MAP**, **DOES NOT MAP** — express increasing degrees of distance between the neuromod algorithm and the FiLM substrate.

### §5.1 Context-switching / regime change

| Neuromod algorithm | FiLM-variant | Mathematical bridge | Tag |
|---|---|---|---|
| Vecoven 2020 NMN-A2C | Vanilla FiLM (§3.1) | $\sigma((z^\top w_s)x + (z^\top w_b)) \equiv \sigma(\gamma(z) x + \beta(z))$ | CLEAN |
| Ben-Iwhiwhu 2022 activity-gating | Vanilla FiLM with $\beta = 0$ (§3.1) | $g(z) \odot h$ ≡ $\gamma(z) \odot h + 0$ | CLEAN |
| Xing 2022 RL under env changes | Vanilla FiLM with change-detector conditioner (§3.1) | $z(t) = \text{ChangeDetector}(s_{t-k:t});\ (\gamma, \beta) = f(z(t))$ | CLEAN |
| Tsuda 2021 multiplicative-on-weights | HyperNet (§3.5) restricted to scalar; KML (§3.11) restricted to $J$ | $\hat W = W \odot (J + (f_{nm} - 1)J) = f_{nm} W$ | EXTENSION |
| Costacurta 2024 NM-RNN | HyperNet (§3.5) restricted to low-rank | $W_x(z) = \sum_k s_k(z) \ell_k r_k^\top$ | EXTENSION |
| AlKilany & Goodman 2025 (per-neuron) | Vanilla FiLM with $\beta = 0$ (§3.1) | $h_i \to g_i \cdot h_i$ | CLEAN (substrate caveat) |
| Wainstein 2025 trained-RNN gain | — | $g$ is an experimental dial, not a side-signal-conditioned operator | DOES NOT MAP |

**Per-row bridges.**

**Row CS-1 — Vecoven 2020 ↔ vanilla FiLM (CLEAN).** Plain-English: Vecoven's neuromodulatory subnetwork is mathematically the same machine as Perez 2018's FiLM, with the FiLM generator implemented as a single linear read-out from the modulator's scalar $z$. Formally,

$$
\sigma_{NMN}(x, z; w_s, w_b) = \sigma\bigl(z^\top(x w_s + w_b)\bigr) = \sigma\bigl((z^\top w_s) \odot x + (z^\top w_b)\bigr) = \sigma(\gamma(z) \odot x + \beta(z)),
$$

with $\gamma(z) = z^\top w_s,\ \beta(z) = z^\top w_b$. The preserved benefit is regime-conditioned per-feature affine modulation — the same mechanism Perez 2018 uses to re-purpose a CNN across CLEVR question types. Vecoven's "$z$-recruitment" diagnostic (Fig. 7) is the direct analogue of an effective-rank-of-$\gamma$ probe in a FiLM-conditioned policy.

**Row CS-2 — Ben-Iwhiwhu 2022 ↔ vanilla FiLM with $\beta = 0$ (CLEAN).** Plain-English: Ben-Iwhiwhu's activity-gating is FiLM with the additive arm clamped to zero. Formally, $\mathrm{FiLM}_{\beta = 0}(h \mid g(z)) = g(z) \odot h + \mathbf{0}$. Any FiLM-on-PPO architecture that ignores or freezes its additive arm degenerates exactly into Ben-Iwhiwhu's modulator — relevant for the project's $\beta$-clamp arm in v3 §5.1's headline falsifier.

**Row CS-3 — Xing 2022 ↔ vanilla FiLM with change-detector conditioner (CLEAN).** The operator is vanilla FiLM; what is task-specific is the conditioner, a learned change-point detector $z(t) = \text{ChangeDetector}(s_{t-k:t})$. The closest published precedent for an RL-domain neuromod algorithm that maps cleanly to FiLM with a *learned* conditioner.

**Row CS-4 — Tsuda 2021 ↔ HyperNet/KML degenerate-scalar restriction (EXTENSION).** Plain-English: Tsuda's mechanism — a single scalar that uniformly rescales the entire weight matrix — is the most degenerate possible hypernetwork, and (via the Abdollahzadeh reinterpretation lemma in §3.11) the most degenerate possible kernel modulation. **This is the load-bearing rewrite of v1's Tsuda framing**: Tsuda is *not* vanilla FiLM on activations (v1's mis-citation); it is one level down the kernel-modulation hierarchy from vanilla FiLM, on weights. The hypertube geometry Tsuda exhibits is an *effect* of weight-level scaling, not an operator FiLM-on-activations directly instantiates — though the project's v3 §5.1 Outcome T falsifier asks whether FiLM-on-activations with high effective-rank $\gamma$ can produce a Tsuda-like geometric signature through a different operator.

**Row CS-5 — Costacurta 2024 ↔ HyperNet restricted to low-rank scaling (EXTENSION).** Formally,

$$
W_x(z) = \sum_{k=1}^K s_k(z) \, \ell_k \, r_k^\top,\qquad s_k(z) = \sigma(A_z z + b_z)_k.
$$

This is a hypernetwork producing only the $K$ scalar coefficients of a fixed low-rank basis. Equivalently, a *generalised FiLM* where $\gamma_k(z) = s_k(z)$ acts on the rank-1 outer-product $\ell_k r_k^\top$ instead of on a per-channel scalar slot. Costacurta's Prop. 1 shows the LSTM forget-gate falls out as a special case. The dissociable-by-ablation multi-channel modulation (Fig. 3F) is the published precedent for the project's per-injection-site clamp tests.

**Row CS-6 — AlKilany & Goodman 2025 (CLEAN, substrate-translation caveat).** The per-neuron excitability multiplier in spiking networks is, in the rate-based / standard-DL surrogate, $h_i \to g_i \cdot h_i$ — vanilla FiLM with $\beta = 0$, identical form to Ben-Iwhiwhu 2022. The preserved benefit, "listening in the dips" — an emergent dynamic-gain-control behaviour that decreases reaction time in noisy listening — is evidence that the multiplicative-only FiLM mechanism produces measurable behavioural improvements *without* explicit training pressure on the gain itself.

**Row CS-7 — Wainstein 2025 (DOES NOT MAP).** Wainstein's $g$ is varied as an experimental dial on a *post-hoc* trained-RNN analysis, not as a side-signal-conditioned forward-pass operator. The "side signal" is pupil diameter, which *correlates* with $g$ but is not the *generator* of $g$. The integration memo treats Wainstein as the corpus's strongest empirical anchor for the cellular-gain ↔ behaviour-switch link, *not* as a FiLM-variant.

**Mathematical sketch box — context-switching headline claim.** Under FiLM-Ensemble (§3.7) with $M$ members and a regime-detector or self-recurrent conditioner producing $(\gamma_b^m, \beta_b^m)$ at each block $b$, context-switching with member-spread-as-uncertainty is implemented as

$$
\pi_b^m(a \mid s) = \mathrm{softmax}\bigl(\gamma_b^m(z) \cdot \ell_b(s) + \beta_b^m(z)\bigr), \qquad U_b(s) = \mathrm{Var}_m[\pi_b^m(\cdot \mid s)],
$$

with $U_b(s)$ functioning as a learnable epistemic-uncertainty estimate that gates whether the agent treats the post-switch evidence as a *new regime* (high $U$, raise exploration) or *noise around the old regime* (low $U$, exploit). **This is the unification the corpus's existing papers approach from different sides** — Vecoven 2020 lacks the uncertainty estimate, Lee 2024 has a hand-coded one, Wainstein 2025 has the empirical anchor but no operator. The combination is the project's substrate-level opportunity.

### §5.2 Hypervigilance / sustained tonic vigilance

| Neuromod algorithm | FiLM-variant | Mathematical bridge | Tag |
|---|---|---|---|
| Ferguson & Cardin 2020 (gain types) | Vanilla FiLM (multiplicative + additive arms) (§3.1) | $\gamma$ = slope change; $\beta$ = rheobase shift | CLEAN |
| Doya 2002 ACh-on-encoder branch | Vanilla FiLM at encoder (§3.1) | $h_A = \gamma_A(\text{mod}) \odot h_A + \beta_A(\text{mod})$ | EXTENSION |
| Shine 2021 cellular ↔ network gain | Vanilla FiLM (§3.1) | $g = dQ/dI = \gamma$ at post-ReLU operating point | CLEAN |
| Temporal FiLM (§3.4) for tonic timescale | Vanilla FiLM with self-recurrent conditioner | $(\gamma_b, \beta_b)$ piecewise-constant over many input timesteps | (architectural fit) |

**Per-row bridges.**

**Row HV-1 — Ferguson & Cardin 2020 ↔ vanilla FiLM, both arms (CLEAN).** Plain-English: Ferguson & Cardin's Box 1 distinguishes multiplicative gain (slope change — sensitivity moves, selectivity stays) from additive bias (rheobase shift — selectivity moves, sensitivity stays). Vanilla FiLM's $\gamma \odot h + \beta$ instantiates **both arms simultaneously**. Hypervigilance — high gain on threat-relevant channels (sensitivity ↑) with unchanged or lowered gain on threat-irrelevant — is a *channel-selective* signature, requiring per-feature $\gamma_i$ rather than a scalar global gain. Channel-selective hypervigilance is exactly what vanilla FiLM's per-channel $(\gamma, \beta)$ supplies. Divisive normalisation — gain scaled by a sum of competing-feature activations — requires content-dependent $\gamma$, i.e., AdaIN's $\sigma^{\mathrm{IN}}(y)$ where the gain is computed from feature statistics rather than a learnable lookup.

**Row HV-2 — Doya 2002 ACh-on-encoder ↔ FiLM at site A (EXTENSION).** Plain-English: Doya 2002's ACh-as-learning-rate mapping ($\alpha$ scalar) maps onto FiLM at the encoder pre-fusion site, with the ACh-side scalar fanned out by the FiLM read-out matrix to per-feature $\gamma_A, \beta_A$. The "fan-out matrix" is the receptor-density-analog from one nucleus to a population of post-synaptic features. The extension is *receptor-density fan-out*, which biology already commits to.

**Row HV-3 — Shine 2021 ↔ vanilla FiLM, cellular vocabulary anchor (CLEAN).** Plain-English: Shine 2021's formal definition $g = dQ/dI$ — slope of a neuron's input-output function — is mathematically identical to FiLM's multiplicative arm in the post-ReLU active regime. Formally, for $y = \phi(\gamma \cdot z + \beta)$ with $\phi$ a ReLU,

$$
\frac{dy}{dz} = \gamma \cdot \phi'(\gamma z + \beta) = \gamma \quad \text{(in the active regime)}.
$$

For sigmoid / softmax, $\phi'$ is the gain curve and $\gamma$ multiplies into it directly. **FiLM's multiplicative arm $\gamma$ literally IS Shine 2021's "neural gain"** at the post-ReLU operating point. The two are not analogues; they are the same scalar with different names.

**Mathematical sketch box — hypervigilance headline claim.** Under FiLM at the encoder site $A$ with a per-modality threat coupling, hypervigilance is implemented as

$$
h_A = \gamma_A(\text{mod\_h}, \text{inj}) \odot h_A^{\mathrm{pre}} + \beta_A(\text{mod\_h}, \text{inj}),
$$

with the channel-selective signature

$$
\Delta\gamma_{A,i}^{\mathrm{post-injury}} = \mathbb{E}_t[\gamma_{A,i}(t \mid \text{post-injury})] - \mathbb{E}_t[\gamma_{A,i}(t \mid \text{pre-injury})],
$$

where the prediction is $\Delta\gamma_{A,i} \ge 0.2$ for $i \in \mathcal{T}$ (threat-coupled channels — nociception, olfaction, vision-of-threat) and $\Delta\gamma_{A,i} \approx 0$ for $i \in \mathcal{N}$ (threat-irrelevant channels — proprioception, colour, location). **No paper in either corpus has logged this signature on a forward-pass FiLM-modulated agent.** The temporal FiLM (§3.4) layer supplies the tonic-timescale substrate: $(\gamma_b, \beta_b)$ piecewise-constant over many input-level timesteps gives the *hysteresis* signature (the modulator output stays elevated after the aversive regime ends) — the substrate-level analogue of chronic hypervigilance.

### §5.3 Plasticity gating / lifelong-learning resistance

| Neuromod algorithm | FiLM-variant | Mathematical bridge | Tag |
|---|---|---|---|
| Rodriguez-Garcia 2026 NGM-SGD | — | gradient-level $W_{ij}(t) = g_i(t) w_{ij}(t)$; effective $\lambda \to \lambda/g^2$ | DOES NOT MAP |
| Costacurta 2024 (NM-RNN, plasticity flavour) | HyperNet low-rank (§3.5) | $W_x(z) = \sum_k s_k(z) \ell_k r_k^\top$; LSTM-equivalent forget-gate (Prop. 1) | EXTENSION |
| Wang 2024 (NeuronML structure mask) | HyperNet-adjacent / MoE (§3.5/3.9) | $W_{\mathrm{eff}, T} = W \odot M_T$; structure-mask generator | EXTENSION |
| Tambaş 2025 (Krotov-Hopfield three-factor) | Bayesian-hypernet-adjacent (§3.6) | $\Delta w_{ij} \propto m \cdot \text{local}(x_i, x_j, y_i)$ | PARTIAL |
| Kudithipudi 2022 (lifelong-learning survey) | Taxonomic umbrella, not a single variant | — | PARTIAL (framing) |

**Per-row bridges.**

**Row PG-1 — Rodriguez-Garcia 2026 (DOES NOT MAP).** Plain-English: Rodriguez-Garcia's gain operates on the *gradient flow during training*; FiLM and all its corpus relatives operate on the *forward pass*. The two are at orthogonal points in the optimisation loop and **the integration paper's honest position is to name Rodriguez-Garcia 2026 as the canonical complement of forward-pass FiLM, not its instance**. Formally,

$$
W_{ij}(t) = g_i(t) \, w_{ij}(t), \qquad g(t+1) = \gamma g(t) + (1 - \gamma) g_0 + \eta H(y),
$$

with effective curvature $\lambda \to \lambda / g^2$. The closest formal cousin in the FiLM family would be HyperNetwork (§3.5) applied to the optimiser's preconditioning matrix rather than to the primary network's weights — but the FiLM corpus contains no such instantiation. **Diagnostic value**: the no-map row draws the substrate boundary. The two operators could in principle combine (forward-pass FiLM + gradient-level NGM-SGD); no paper has yet.

**Row PG-2 — Costacurta 2024 plasticity-flavour ↔ HyperNet low-rank (EXTENSION).** Already presented in §5.1 CS-5; relisted here because Costacurta's dissociation-by-ablation is the architectural-precedent template for forward-pass-only plasticity gating. The LSTM-equivalence (Prop. 1) is the formal bridge that says low-rank weight scaling is gated memory update — i.e., a learned forget-gate, i.e., a state-dependent effective discount.

**Row PG-3 — Wang 2024 ↔ HyperNet-adjacent structure mask (EXTENSION).** Plain-English: Wang's per-task binary or low-precision mask $M_T$ is a hypernetwork that produces structure rather than continuous weights. Formally,

$$
W_{\mathrm{eff}, T} = W \odot M_T, \qquad M_T = \text{StructureGenerator}(\text{task descriptor of } T).
$$

This is structurally close to Sparse MoE (§3.9) with $k$ exactly equal to the active sub-mask cardinality. **Relevance**: when FiLM's per-channel scalar modulation saturates (the v3 effective-rank-of-$\gamma$ probe signals saturation), structure-masks are the architectural ally.

**Row PG-4 — Tambaş 2025 ↔ structured-weight-modulation / Bayesian-hypernet-adjacent (PARTIAL).** The Krotov-Hopfield three-factor rule has a global modulator $m$ gating local Hebbian-flavoured plasticity: $\Delta w_{ij} \propto m \cdot \text{local}(x_i, x_j, y_i)$. The closest FiLM-family cousin is structured weight modulation under a multiplicative side signal — bridging Costacurta 2024 (low-rank) and Bayesian hypernet (§3.6) where the global signal plays the role of a prior over the weight-update distribution. **The map is partial**: the *form* (global multiplier on a local quantity) is FiLM-like, but the *operand* (a plasticity rule) is outside the FiLM corpus's substrate.

**Mathematical sketch box — plasticity-gating headline claim.** Under FiLM at the GRU update-gate site $B$ with a self-recurrent (Temporal-FiLM-style) conditioner over a slow timescale $\tau_T$, plasticity gating is implemented as

$$
z_t = \sigma\bigl(W_z [h_{t-1}, x_t] + b_z + \beta^{(B)}(\text{mod\_h}_t^T)\bigr),
$$

where $\beta^{(B)}$ shifts the "stay vs update" decision boundary additively — the rheobase analogue of Ferguson & Cardin 2020 Box 1c. The state-dependent effective discount over past memory is $\tilde\gamma_{\mathrm{Bellman}}(\beta^{(B)})$ — *distinct* from FiLM's $\gamma$, which we acknowledge as a notational collision the math-reviewer should flag. The forward-pass substrate's stability-gap attenuation is the project's R2 empirical signature; Rodriguez-Garcia 2026's gradient-level complement is the parallel mechanism the substrate boundary refuses to absorb.

### §5.4 Exploration–exploitation control

| Neuromod algorithm | FiLM-variant | Mathematical bridge | Tag |
|---|---|---|---|
| Doya 2002 NA-$\beta$ branch | Vanilla FiLM at policy site (§3.1) | $\pi' = \mathrm{softmax}(\gamma_C \ell_a + \beta_C)$; $\gamma_C \equiv 1/T$ | EXTENSION |
| Lee 2024 hand-coded $\beta^{-1}$ | AdaIN-style closed-form generator (§3.3) | $\beta^{-1}(s) = 1/\langle E(s,\hat a)\rangle$ | EXTENSION |
| Ferguson & Cardin 2020 (multiplicative arm) | Vanilla FiLM (§3.1) | $\gamma$-arm = slope of softmax = inverse temperature | CLEAN (mechanism inventory) |
| Shine 2021 ($g = dQ/dI$) | Vanilla FiLM (§3.1) | post-ReLU active regime, $dy/dz = \gamma$ | CLEAN (vocabulary) |
| Ben-Iwhiwhu 2022 / AlKilany 2025 | Vanilla FiLM with $\beta = 0$ (§3.1) | $h \to \gamma \odot h$ | CLEAN |
| FiLM-Ensemble (§3.7) + heteroscedastic precision (Kendall-Gal 2017) | Compound: ensemble produces uncertainty estimate, FiLM produces inverse-temperature | $\gamma_C = f(\mathrm{Var}_m[\pi^m]); \mathcal{L} = \tfrac{1}{2} e^{-s} \|y - \hat y\|^2 + \tfrac{1}{2} s$ | (project's unique combination) |
| Osman 2024 Hopfield + gain | — | recurrent-attractor energy-landscape gain | DOES NOT MAP |

**Per-row bridges.**

**Row EE-1 — Doya 2002 NA-$\beta$ ↔ FiLM at site C (EXTENSION).** Plain-English: scaling actor logits by FiLM's multiplicative arm $\gamma_C$ produces

$$
\ell'_a = \gamma_C(\text{mod\_h}) \cdot \ell_a + \beta_C(\text{mod\_h})
\;\;\Rightarrow\;\;
\pi'(a) = \mathrm{softmax}\bigl(\gamma_C \, \ell_a + \beta_C\bigr).
$$

Comparing to a temperature-scaled softmax $\pi(a) \propto \exp(\ell_a / T)$, the multiplicative $\gamma_C$ acts as an *inverse temperature* $1/T$. **FiLM at site C IS the Doya-NA-$\beta$ inverse-temperature mechanism, expressed as a learned forward-pass operator instead of a scalar global parameter.** The extension over a scalar global gain is the receptor-density-analog fan-out from a single nucleus's tone to per-feature $\gamma_C \in \mathbb{R}^d$.

**Row EE-2 — Lee 2024 ↔ AdaIN-style closed-form (EXTENSION).** Plain-English: Lee 2024's hand-coded mapping

$$
\alpha(s, a) = \frac{E(s, a)}{E(s, a) + A(s, a)}, \qquad \beta^{-1}(s) = \frac{1}{\langle E(s, \hat a)\rangle_{\hat a}},
$$

uses $E$ (aleatoric / expected uncertainty) and $A$ (epistemic / unexpected uncertainty) from a value ensemble. This is **AdaIN's structural move** ($(\gamma, \beta) = (\sigma^{\mathrm{IN}}(y), \mu^{\mathrm{IN}}(y))$ as closed-form functions of the *style image's* deep features) **applied with the ensemble's uncertainty statistics in place of the style image's feature statistics**. The closed-form generator is the formal class; what changes between Huang & Belongie 2017 and Lee 2024 is which "second image" the statistics come from.

**Row EE-3 — Ferguson & Cardin 2020 multiplicative arm (CLEAN, mechanism inventory).** The multiplicative arm of vanilla FiLM is the cellular substrate of the v3 γ-clamp vs β-clamp falsifier: $\gamma$ moves sensitivity (slope of the I/O curve), $\beta$ moves selectivity (the threshold). The four-arm probe at site C is the project's substrate-level operationalisation of Ferguson & Cardin Box 1b/c.

**Row EE-4 — Shine 2021 (CLEAN, vocabulary).** Already presented in §5.2 HV-3.

**Row EE-5 — Ben-Iwhiwhu 2022 / AlKilany 2025 (CLEAN).** Already presented in §5.1 CS-2 / CS-6.

**Row EE-6 — FiLM-Ensemble + heteroscedastic precision (the project's unique combination).** This row does not appear as a single neuromod-algorithm paper in the corpus — it is a *combination* the integration mapping uncovers as not yet attempted. **No paper in either corpus has combined FiLM-modulated $(\gamma, \beta)$ with Kendall-Gal heteroscedastic precision $\sigma^2(x)$ on the same regression head.** The combination produces a *learnable, end-to-end-trained precision-weighted gain*. Formally, let $M$ FiLM-Ensemble members produce $\{\pi^m\}_{m=1}^M$, define the epistemic uncertainty $U^{\mathrm{epist}}_C(s) = \mathrm{Var}_m[\pi^m(\cdot \mid s)]$, and let a Kendall-Gal heteroscedastic precision head output $\sigma^2(s)$. Then a *learnable Lee-2024-analogue* is

$$
\gamma_C(s) = f\bigl(U^{\mathrm{epist}}_C(s), \sigma^{-2}(s)\bigr), \qquad \mathcal{L} = \tfrac{1}{2} e^{-s} \|y - \hat y\|^2 + \tfrac{1}{2} s.
$$

This is the math-reviewer-relevant compound formula. **Math-reviewer flag**: the load-bearing claim is that the combination is operationally well-formed and that the gradient flows through both $\sigma^{-2}$ and the FiLM-Ensemble members coherently. The compound formula should be audited for (a) sign conventions matching Kendall-Gal Eq. 6, (b) the variance estimator $U^{\mathrm{epist}}$ being a proper post-softmax variance rather than logit-space, (c) the FiLM-Ensemble's gain-initialisation parameter $\rho$ (Turkoglu 2022 §3) being consistent with non-degenerate ensemble spread at training start.

**Row EE-7 — Osman 2024 Hopfield + gain (DOES NOT MAP).** Plain-English: Osman 2024's $W_{\mathrm{rec}} \to g \cdot W_{\mathrm{rec}}$ on a recurrent Hopfield network — low $g$ flattens the attractor energy landscape — has the same mathematical form as Tsuda 2021 (degenerate-scalar hypernet), but the *behavioural effect* operates through energy-landscape annealing in a recurrent attractor network, which is a substrate the FiLM corpus's operators do not produce. The integration memo names Osman 2024 as a **complementary architectural family** (recurrent-attractor-net + gain), not a FiLM-variant.

**Mathematical sketch box — exploration-exploitation headline claim.** Under FiLM-Ensemble at the policy site, exploration-exploitation control is implemented as

$$
\pi^m_C(a \mid s) = \mathrm{softmax}\bigl(\gamma_C^m(\text{mod\_h}) \cdot \ell(s) + \beta_C^m(\text{mod\_h})\bigr), \qquad m \in \{1, \ldots, M\},
$$

with the diagnostic signature $r(\gamma_C, |\delta|) \ge 0.5$ — the Pearson correlation between the multiplicative arm and the value-prediction-error magnitude over the recovery window — distinguishing Doya-NA-$\beta$-inverse-temperature (Outcome U) from Vecoven-task-identity (Outcome V) and Tsuda-hypertube (Outcome T) in the v3 §5.1 four-arm falsifier.

### §5.5 The no-map boundary — four honestly named cases

The integration claim is sharp because four lines of work **do not** map onto a FiLM-variant. Naming these is part of the contribution, not a failure of it: they draw the substrate boundary so the unification's footprint is honest. (No-map 4 was added after the [film_hypernet_bnn_lineage_and_critic_modulation.md](film_hypernet_bnn_lineage_and_critic_modulation.md) investigation: it covers the 5-HT / γ_Bellman channel that the math-reviewer audit flag (d) showed the v3 site-B framing was incorrectly conflating with plasticity gating.)

**No-map 1: Rodriguez-Garcia 2026 (gradient-level).** The gain $g_i(t)$ multiplies the *gradient*, not the forward activation. The effective learning rate becomes $\lambda \to \lambda/g^2$. The FiLM corpus contains no gradient-level operator. The closest formal cousin would be HyperNetwork applied to the optimiser's preconditioning matrix, but no FiLM-corpus paper has that instantiation. The two operators (forward-pass FiLM and gradient-level NGM-SGD) are orthogonal and could in principle combine — no paper has done so. The integration paper's position: **Rodriguez-Garcia 2026 is the canonical complement to forward-pass FiLM, not an instance of it.**

**No-map 2: Wainstein 2025 (trained-RNN internal gain).** Wainstein's $g$ is an experimenter-varied dial applied post-hoc to a trained RNN's activation function for analysis. It is *not* a side-signal-conditioned forward-pass operator — there is no learned generator producing $g$ as a function of an input or context signal. The pupil-diameter "side signal" correlates with $g$ but is not its generator: the RNN was trained without pupil input. The integration paper's position: **Wainstein 2025 is the corpus's strongest empirical anchor for the cellular-gain ↔ behavioural-switch link — and the substrate-level template (trained-RNN-gain-as-internal-state) is operationally distinct from the FiLM substrate-level template (side-signal-conditioned forward-pass affine).**

**No-map 3: Osman 2024 (recurrent-attractor energy-landscape gain).** Osman's $W_{\mathrm{rec}} \to g \cdot W_{\mathrm{rec}}$ on a Hopfield network has the same mathematical form as Tsuda 2021 (so it would CLEAN-MAP onto degenerate-scalar HyperNet), but the *operand* is a recurrent attractor network, not a feedforward / GRU policy substrate. The behavioural effect — arousal-modulated Bayesian-inference temperature via energy-landscape annealing — is not what FiLM-on-activations produces. The integration paper's position: **Osman 2024 is a complementary architectural family — recurrent-attractor-net + gain — that the FiLM corpus's operators do not subsume.**

**No-map 4: Doya 2002 §3.2 — 5-HT-as-discount (γ_Bellman) modulation.** The neuromod-corpus mapping of 5-HT to the Bellman value-function discount γ_Bellman has **no implementation in either corpus**: Lee 2024 §6 explicitly drops the 5-HT branch ("no convergent normative theory exists"); Xing 2022, Wang 2024, and Rodriguez-Garcia 2026 all leave γ_Bellman fixed. The mathematical reason is dimensional: γ_Bellman is a *global scalar inside a temporal sum* governing the value-function horizon, $V^\pi(s) = \mathbb{E}[\sum_t \gamma_{\text{Bellman}}^t r_t]$, whereas a forward-pass FiLM operator produces *per-feature affine on activations*. The clean architectural home for γ_Bellman modulation is the TD-target computation $y_t = r_t + \gamma(c) V(s_{t+1})$ — at the *loss / learning-update* layer, **not** the forward pass. The follow-up investigation memo ([film_hypernet_bnn_lineage_and_critic_modulation.md §3](film_hypernet_bnn_lineage_and_critic_modulation.md)) names four candidate modulation points (critic-head output FiLM as value-scale-not-horizon proxy; TD-target γ; GAE λ; a learned-discount network $\hat\gamma(c)$) and recommends Option A — drop γ_Bellman from substrate-level claims for v4. A forward-pass FiLM substrate cannot carry this channel without architectural extension into a hybrid system (forward-pass FiLM at A/C plus a learned-discount head feeding the TD target — the investigation memo's Option B). v4 of the direction memo ([20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md)) names this honestly and walks back v3's site-B "effective discount on past memory" framing to **plasticity gating** — per-unit hidden-state retention, measurable as recurrent-state autocorrelation $\tau_h$ — instead. *Future work*: the hybrid architecture (Option B in the investigation memo §3.5) is the route for a v5 project wanting the full 4-channel Doya mapping.

**The substrate boundary, stated once.** The FiLM substrate covers **forward-pass neuromodulation on feedforward / GRU policy networks via side-signal-conditioned affine operators on activations and weights**. It does not cover gradient-level (Rodriguez-Garcia 2026), trained-RNN-internal-as-dial (Wainstein 2025), recurrent-attractor energy-landscape (Osman 2024), or value-function-horizon (Doya 2002 §3.2 5-HT/γ_Bellman) mechanisms. The 7 + 5 = 12 in-fit rows are enough to support the unification claim; the 4 no-map rows draw the boundary — lineage-grounded and honest about which Doya channels the substrate genuinely carries (α, β, plasticity gating) versus which it cannot (γ_Bellman).

---

## §6 — Per-behaviour mathematical sketches with diagnostic signatures

For each of the four target behaviours, we state the formal claim, give the diagnostic mathematical signature that discriminates the FiLM-instance from alternatives, and name the expected reading under each FiLM-variant option. This section is the bridge from theoretical taxonomy to project-experiment design.

### §6.1 Context-switching / regime change

**Formal claim.** Under FiLM-Ensemble (§3.7) with a self-recurrent or change-detector conditioner, context-switching with behaviour-level recovery is implemented as state-dependent multi-member $(\gamma_b^m, \beta_b^m)$ at policy site $C$ and (optionally) encoder site $A$.

**Diagnostic mathematical signature 1: effective-rank of $\gamma$ at policy site.** Define

$$
\mathrm{erank}(\gamma_C) := \exp\biggl(-\sum_i \tfrac{\sigma_i^2}{\sum_j \sigma_j^2} \log \tfrac{\sigma_i^2}{\sum_j \sigma_j^2}\biggr),
$$

with $\sigma_i$ the singular values of the per-timestep $\gamma_C$ matrix over a regime window. Under vanilla FiLM (Row CS-1 Vecoven), $\mathrm{erank}(\gamma_C)$ shifts from rank-1 (single regime) to broader rank during recovery and re-collapses post-recovery; under FiLM-Ensemble with $M$ members, the *between-member spread* increases during recovery and collapses post. **Discriminating reading**: rank-1 throughout → Outcome V (Vecoven-task-identity, no Doya channel); rank-broadening at switch → Outcome U (Unification); rank-saturation throughout → architectural over-parametrisation, KML upgrade indicated (Row 7).

**Diagnostic mathematical signature 2: modulator-state PCA velocity around switch.** Project `mod_h` time-series onto a low-D PCA basis and compute $\|d\hat z/dt\|$ peri-switch. Under Wainstein 2025's empirical chain, this velocity peaks at switch; under FiLM-on-PPO, the same signature should appear if the modulator is operating in Doya-channel mode.

**Diagnostic mathematical signature 3: per-modulator-dimension ablation matrix.** Construct the 3×3 deficit matrix `eval/per_site_freeze_per_subtest_drop` (A-freeze, B-freeze, C-freeze × noisy-obs, long-dormancy R2/R3, regime-change recovery). **Discriminating reading**: diagonal dominance → Costacurta-2024-style dissociable channels → multi-channel reading holds; flat matrix → single-channel reading wins.

### §6.2 Hypervigilance / sustained tonic vigilance

**Formal claim.** Under vanilla FiLM at encoder site $A$ with channel-selective threat coupling, hypervigilance is implemented as differential post-injury $\gamma_{A,i}$ across threat-coupled $i \in \mathcal{T}$ vs threat-irrelevant $i \in \mathcal{N}$ channels.

**Diagnostic mathematical signature 1: channel-selective $\Delta\gamma_A$.** $\Delta\gamma_{A,i}^{\mathrm{post-injury}} = \mathbb{E}_t[\gamma_{A,i}(t \mid \text{post-injury})] - \mathbb{E}_t[\gamma_{A,i}(t \mid \text{pre-injury})]$. Prediction: $\Delta\gamma_{A, \mathcal{T}} \ge 0.2$, $\Delta\gamma_{A, \mathcal{N}} \approx 0$. **Discriminating reading**: uniform $\Delta\gamma_A$ across modalities → volume control, not hypervigilance; channel-selective $\Delta\gamma_A$ → hypervigilance signature.

**Diagnostic mathematical signature 2: temporal hysteresis.** Under Temporal FiLM (§3.4) with self-recurrent conditioner, the post-injury $\gamma_{A, \mathcal{T}}$ elevation should persist over $\tau_T \in [50, 200]$ episodes after the aversive regime ends — the chronic-pain hysteresis signature. **Discriminating reading**: rapid decay $\tau < 10$ episodes → phasic-only mechanism, no tonic vigilance; sustained elevation → tonic vigilance.

**Diagnostic mathematical signature 3: divisive-normalisation signature at site A.** Under AdaIN-style content-dependent $\gamma$ (Ferguson & Cardin 2020's divisive normalisation arm), high $\gamma_{A, i}$ on one channel should *reduce* relative weight on competing channels. Measure as $\mathrm{cov}(\gamma_{A,i}, \gamma_{A,j})$ for $i \ne j$ in the post-injury window. **Discriminating reading**: negative covariance → divisive normalisation; near-zero covariance → independent per-channel gain.

### §6.3 Plasticity gating / lifelong-learning resistance

**Formal claim.** Under FiLM at the GRU update-gate site $B$ with a slow-timescale conditioner (Temporal FiLM, §3.4), plasticity gating is implemented as additive $\beta^{(B)}$ shifting the update/stay decision boundary, attenuating the stability gap at task boundaries.

**Diagnostic mathematical signature 1: continual-evaluation stability gap (Hess-protocol).** Following Rodriguez-Garcia 2026 (Algorithm 1), evaluate old-task accuracy per batch every $\rho$ iterations through a task boundary. The stability gap is the transient accuracy drop; attenuation is the measure. **Discriminating reading**: forward-pass FiLM at site B attenuates stability gap while Rodriguez-Garcia 2026's gradient-level NGM-SGD operates at a different mechanism layer; both can attenuate; their combination has not been tested.

**Diagnostic mathematical signature 2: Hessian-eigenvalue probe peri-switch.** Compute the loss-curvature spectrum $\{\lambda_i\}$ via Hessian-vector product (e.g., `jax.jvp(jax.grad(L))`) peri-switch. Under Rodriguez-Garcia 2026, the effective curvature flattens: $\lambda_i \to \lambda_i / g^2$. Under forward-pass FiLM, the *forward-pass conditioning* changes but the loss curvature at the raw weight should not flatten through the same gradient-level mechanism. **Discriminating reading**: curvature flattening at the raw-weight Hessian → gradient-level mechanism dominant; curvature unchanged at raw-weight but conditioning of effective forward pass changes → FiLM-level mechanism. This is the project-substrate-level test that disambiguates the two complementary substrates.

**Diagnostic mathematical signature 3: per-batch deficit attenuation in the project's R2 readout.** The project's R2 win (~25× the seed-noise floor on second return to a previously-seen task stage) is plausibly an attenuated stability gap. Comparing per-batch evaluation against a fixed-hyperparameter baseline gives the attenuation magnitude.

### §6.4 Exploration-exploitation control

**Formal claim.** Under FiLM-Ensemble (§3.7) at the policy site $C$ with $M$ members, exploration-exploitation control is implemented as ensemble-uncertainty-conditioned multiplicative $\gamma_C^m$ on actor logits, with Kendall-Gal heteroscedastic precision (§3 of Kendall & Gal 2017) optionally compounding to give a learnable Lee-2024-analogue.

**Diagnostic mathematical signature 1: $\gamma_C$ ↔ $|\delta|$ correlation.** Pearson correlation of $\gamma_C$ output with value-prediction-error magnitude $|\delta|$ over the recovery window. Prediction (Outcome U from v3 §5.1): $r(\gamma_C, |\delta|) \ge 0.5$. **Discriminating reading**: $r \ge 0.5$ → Doya-NA-$\beta$ inverse-temperature work confirmed; $r \approx 0$ with both clamps roughly equally damaging → Vecoven-task-identity reading wins (Outcome V); $r \approx 0$ with $\beta$-clamp more damaging than $\gamma$-clamp → Tsuda-hypertube reading wins (Outcome T).

**Diagnostic mathematical signature 2: actor-entropy peri-switch spike amplitude.** Under Doya 2002 NA-$\beta$ branch, actor entropy should spike at regime change as inverse temperature drops to enable exploration. The spike amplitude should track $\gamma_C$ amplitude.

**Diagnostic mathematical signature 3: post-hoc Lee 2024 reorganisation.** Under the compound FiLM-Ensemble + heteroscedastic precision formula (Row EE-6),

$$
\gamma_C(s) = f\bigl(U^{\mathrm{epist}}_C(s), \sigma^{-2}(s)\bigr),
$$

regress the learned $\gamma_C$ on the closed-form Lee 2024 target $1/\langle E(s, \hat a)\rangle$. If the learned FiLM generator reorganises through training to approximate the Lee functional form, that is direct evidence that an end-to-end-trained FiLM substrate has *learned the Doya-NA-$\beta$ mapping* without being told it. This is the project's substrate-level claim and the cleanest paper-shape headline finding to aim for.

---

## §7 — Gaps and project positioning

Six concrete gaps the project's three-injection-site FiLM substrate is positioned to uniquely fill, distilled from the curator synthesis §5.

**§7.1 No FiLM paper logs the behaviour-level readouts the neuromod literature predicts.** The FiLM corpus measures task accuracy, calibration / OOD detection, robotic task success, anti-exploration bonus minimisation, and style-transfer quality. It does *not* measure actor-entropy peri-switch, $|\delta|$-tracking by the modulator, per-channel $\gamma_A$ statistics time-locked to interoceptive injury onset, or continual-evaluation stability-gap on a FiLM-modulated agent. **The project's substrate fills this gap directly** — none of the 23 FiLM papers logs these on a single forward-pass FiLM agent; none of the 18 neuromod papers logs them on a FiLM substrate.

**§7.2 No neuromod paper uses a FiLM-ensemble or Bayesian hypernet for uncertainty over the modulator output.** Lee 2024's hand-coded $\beta^{-1}(s) = 1/\langle E(s, \hat a)\rangle$ requires an ensemble. The two cheap implicit ensembles in the FiLM corpus — FiLM-Ensemble (1.3% parameter overhead) and BatchEnsemble — have never been used for this. The combination is **the project's Row EE-6 compound formula**; rl-bayesian-dl P-5 in v3 §5.7 names this as future work.

**§7.3 The FiLM-variant family is implicit but never explicitly enumerated as a unifying taxonomy.** This memo's §3 + §5 mapping is **the first explicit catalogue** of how the FiLM-variant family relates to the neuromodulator-style mechanism family — operator-level, with mathematical bridges, behaviour-tagged, honestly no-mapped where the substrate boundary lies. The paper-shape standalone gap.

**§7.4 No forward-pass FiLM substrate has been benchmarked against Rodriguez-Garcia 2026 NGM-SGD as the orthogonal complement.** The Hessian-eigenvalue probe (§6.3 signature 2) is the proposed discriminating measurement. **The project's substrate-level contribution** is the empirical disambiguation of forward-pass FiLM vs gradient-level NGM-SGD on the same agent on the same trials.

**§7.5 The bidirectional cellular-gain ↔ behavioural-hyperparameter unification test (gap 1 from the predictions synthesis).** No paper logs cellular-gain-style and behavioural-hyperparameter-style readouts on the same agent on the same trials. The project's three-injection-site FiLM + per-site $\gamma, \beta$ logging + actor-entropy / $|\delta|$-tracking is uniquely positioned to run this test — v3 §5.1's four-arm $\gamma$-clamp/$\beta$-clamp falsifier is the operational form.

**§7.6 Phasic-vs-tonic dissociation on a single-substrate forward-pass machine.** No paper in the FiLM corpus has two recurrent timescales with separable phasic and tonic readouts (Temporal FiLM has one LSTM-timescale; Vecoven 2020 has one; Rodriguez-Garcia 2026 has one). **The project's Phase 0 T/P split is the substrate-level architectural change that fills this gap** — v3 §5.3 names the autocorrelation-time-constant probes ($\tau_P, \tau_T$) as the diagnostic.

**Headline gap.** No published paper has run a symmetric bidirectional cellular-gain ↔ behavioural-hyperparameter unification test on a forward-pass FiLM-modulated agent.

---

## §8 — Conclusion and downstream uses

### §8.1 What this concept memo contributes

This memo names the **FiLM-variant taxonomy** as the appropriate mathematically-articulated substrate for the forward-pass slice of the neuromodulation-inspired algorithm family. It maps roughly twelve of eighteen load-bearing corpus papers onto a FiLM-variant via explicit mathematical bridges (§5). It names the substrate boundary via three honestly named no-map cases — Rodriguez-Garcia 2026 (gradient-level), Wainstein 2025 (trained-RNN-internal gain), Osman 2024 (recurrent-attractor energy-landscape gain). It establishes the **Abdollahzadeh 2021 reinterpretation lemma** as the load-bearing mathematical result that vanilla FiLM, Tsuda 2021's scalar weight gain, and KML are *nested restrictions of one kernel-modulation hierarchy*, not independent mechanisms — sharpening the previous v3 / predictions-synthesis Tsuda framing.

### §8.2 Downstream uses

- **The direction memo v4** ([20260516_nmn_scaling_shifting_as_hyperparameter_modulation v4 (forthcoming)](../directions/)) will cite this concept memo for its substrate choice. The v3 §3 unification claim is operator-level; this memo extends it from "FiLM at 3 sites" to "FiLM-family as unifying taxonomy across the 4 target behaviours".
- **`senior-developer`** can read the integration mapping (§5) for the operational equivalences. Specifically, the Row EE-6 compound formula (FiLM-Ensemble + heteroscedastic precision) is the most novel substrate-level combination the memo identifies and is a candidate for the project's Phase 3 precision head implementation work.
- **`experiment-designer`** can read the per-behaviour mathematical sketches (§6) for the diagnostic signatures, each of which translates directly to a WandB readout: `eval/erank_gamma_C`, `eval/per_site_freeze_per_subtest_drop`, `delta_gamma_A_threat_vs_irrelevant`, `r_filmarm_vs_abs_delta_per_clamp`, `mod_h_P_autocorr_tau`, `mod_h_T_autocorr_tau`, `eval/hessian_curvature_peri_switch`.
- **`math-reviewer`** should pay particular attention to: (a) the Abdollahzadeh 2021 reinterpretation lemma derivation (§3.11) — the load-bearing math that establishes FiLM/Tsuda nesting; (b) the FiLM-Ensemble + heteroscedastic-precision compound formula (Row EE-6 in §5.4) — the most novel mathematical combination, with sign-convention and gradient-flow concerns flagged in-line; (c) the §6.3 Hessian-eigenvalue probe formula for disambiguating forward-pass FiLM from Rodriguez-Garcia 2026 gradient-level NGM-SGD; (d) the §5.3 plasticity-gating sketch where the symbol $\gamma$ has a notational collision between FiLM's multiplicative arm and the Bellman discount — flagged explicitly.

### §8.3 What this memo is not

This memo is not a complete review of either literature (the curator synthesis at [`references/FiLM/film_neuromod_integration_synthesis.md`](../references/FiLM/film_neuromod_integration_synthesis.md) and the predictions synthesis at [`references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md`](../references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md) carry that work). It is not a derivation of any single mathematical claim at math-reviewer-final depth — the per-paper formal derivations live in the curator synthesis. It is not a clinical or biological-plausibility claim about pain or hypervigilance — the construct memos at [`active_inference_hypervigilance.md`](active_inference_hypervigilance.md) and [`pain_vs_nociception_construct.md`](pain_vs_nociception_construct.md) carry that work. **It is the project's first publication-direction theoretical contribution**: a freestanding paper-shape statement that the FiLM/hypernet family **is** the mathematically-articulated taxonomy that integrates the forward-pass slice of neuromodulation-inspired algorithms, and that the project's three-injection-site FiLM substrate is positioned to test this integration empirically.

---

## §9 — References + reading guide

### Primary inputs

- **FiLM ↔ neuromod integration synthesis** — [`docs/project/references/FiLM/film_neuromod_integration_synthesis.md`](../references/FiLM/film_neuromod_integration_synthesis.md). The ~10,500-word curator synthesis. §1 FiLM-variant taxonomy; §2 neuromod-algorithm corpus; §3 18-row mapping; §4 per-behaviour integration; §5 gaps; §6 reading guide.
- **Neuromodulatory algorithms predictions synthesis** — [`docs/project/references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md`](../references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md). §2 mechanism inventory; §3 predictions; §5 gaps; §6 unifications-already-tried.
- **Active direction memo v3** — [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md`](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md). §1–3 unification framing; §5.1 four-arm γ/β-clamp falsifier; §5.2 per-injection-site predictions.

### FiLM corpus (load-bearing only — others in curator §6)

- **Perez et al. 2018** ([review](../references/FiLM/reviews/perez_2018_film.md)) — §2.1 FiLM equation; §4.3 $\gamma$ carries 65% of conditioning weight.
- **Ha, Dai & Le 2016** ([review](../references/FiLM/reviews/ha_2016_hypernetworks.md)) — §3.1 static hypernet; §3.2 HyperRNN row-scaling trick.
- **Dumoulin et al. 2017 CIN** ([review](../references/FiLM/reviews/dumoulin_2017_cond_instance_norm.md)) — §2.1 CIN equation; §3.4 linear interpolation in $(\gamma, \beta)$ space.
- **Huang & Belongie 2017 AdaIN** ([review](../references/FiLM/reviews/huang_belongie_2017_adain.md)) — §5 closed-form $(\gamma, \beta) = (\sigma, \mu)$ from style image; precedent for Row EE-2 (Lee 2024).
- **Birnbaum et al. 2019 TFiLM** ([review](../references/FiLM/reviews/birnbaum_2019_temporal_film.md)) — §3 Algorithm 1 for block-piecewise-constant $(\gamma, \beta)$.
- **Turkoglu et al. 2022 FiLM-Ensemble** ([review](../references/FiLM/reviews/turkoglu_2022_film_ensemble.md)) — §2 FiLM-Ensemble equation; Table 1 parameter overhead. Load-bearing for Row EE-6.
- **Krueger et al. 2017 Bayesian hypernets** ([review](../references/FiLM/reviews/krueger_2017_bayesian_hypernets.md)) — §3.2 Bayesian hypernet via normalising flow; §3.3 weight-norm efficient parametrisation.
- **Galanti & Wolf 2020** ([review](../references/FiLM/reviews/galanti_wolf_2020_hypernet_modularity.md)) — Thm. 4 modularity bound. Theoretical anchor for §3.5.
- **Abdollahzadeh et al. 2021 KML** ([review](../references/FiLM/reviews/abdollahzadeh_2021_multimodal_meta.md)) — §3 reinterpretation lemma. **The load-bearing math result of this memo's §3.11.**
- **Kendall & Gal 2017 uncertainties** ([review](../references/FiLM/reviews/kendall_gal_2017_uncertainties.md)) — §3.1 Eq. 6 heteroscedastic loss. Load-bearing for Row EE-6 compound formula.

### Neuromod corpus (load-bearing only — others in curator §6)

- **Doya 2002** — §3 pp. 498-499 four-knob mapping; §3.3 NA-$\beta$; §3.4 ACh-$\alpha$. Foundational for all four behaviours.
- **Ferguson & Cardin 2020** — Box 1 panels b (multiplicative slope), c (additive rheobase). Cellular substrate for §5.2, §5.4.
- **Shine 2021** — Box 1 + Fig. 1 — $g = dQ/dI$. Cellular vocabulary for §5.2 Row HV-3.
- **Lee et al. 2024** — §3.1 Fig. 2 framework; Eq. 4 hand-coded mapping. Load-bearing for Row EE-2.
- **Vecoven et al. 2020** — §2 Fig. 1 $\sigma_{NMN}$ equation; Fig. 7 $z$-recruitment. Load-bearing for Row CS-1.
- **Tsuda et al. 2021** — Fig. 1b + Extended Data Appendix A multiplicative-on-weights. Load-bearing for Row CS-4 and §3.11 reinterpretation lemma.
- **Costacurta et al. 2024** — §3.1 Eq. 4 low-rank scaling; Prop. 1 LSTM-equivalence; Fig. 3F dissociation-by-ablation. Load-bearing for Row CS-5 / PG-2.
- **Rodriguez-Garcia et al. 2026** — §2.1 Eq. 2 fast-slow weight decomposition; Algorithm 1; Fig. 2 loss-landscape flattening. Load-bearing for §5.5 no-map 1.
- **Wainstein et al. 2025** — Fig. 1 pupil-switch chain; Figs. 4-5 RNN-trajectory velocity / energy-landscape flatness. Load-bearing for §5.5 no-map 2.
- **Osman et al. 2024** — Hopfield + gain; annealing-schedule interpretation. Load-bearing for §5.5 no-map 3.
- **Mei et al. 2022** — Fig. 1B canonical diagram of neuromodulation → DNN hyperparameter. Programmatic framing for §2.
- **Ben-Iwhiwhu et al. 2022** — §4 Fig. 1b activity-gating. Load-bearing for Row CS-2.
- **Xing et al. 2022** — context-conditioned modulator triggered by env change. Load-bearing for Row CS-3.
- **AlKilany & Goodman 2025** — Results §"listening in the dips". Load-bearing for Row CS-6.
- **Tambaş et al. 2025** — Krotov-Hopfield three-factor rule. Load-bearing for Row PG-4.
- **Wang et al. 2024 (NeuronML)** — §IV-V bi-level optimisation. Load-bearing for Row PG-3.
- **Kudithipudi et al. 2022** — plasticity-gating biological mechanisms survey. Load-bearing for §5.3 framing.

### Construct concept memos

- [`active_inference_hypervigilance.md`](active_inference_hypervigilance.md) — Bayesian-brain reading of hypervigilance; the precision-level account that the §5.2 hypervigilance section connects to.
- [`pain_vs_nociception_construct.md`](pain_vs_nociception_construct.md) — pain-construct guard-rail for the §5.2 channel-selective hypervigilance signature.

### Project anchors

- [`project_plan.md`](../project_plan.md) — G1 (the substrate gate), G2 (the construct gate). This memo positions the project to test G1 via §6 diagnostic signatures across all four behaviours.
- [`NEUROMODULATION_ALGORITHM.md` §1.4 H1–H5](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) — H1 (single-channel inverse temperature), H2 (multi-channel hyperparameter modulation), H3 (channel-selective hypervigilance), H4 (plasticity gating), H5 (precision-weighted exploration). §5 rows tag each H against a FiLM-variant.

---

## §10 — Math-reviewer audit

### Plain-English entry point

I checked every equation in §3 (the FiLM-variant taxonomy), §5 (the eighteen-row integration mapping), and §6 (the per-behaviour mathematical sketches) against the cited paper reviews, with special attention to the four flags the postdoc surfaced. **Headline verdict: the math is solid on three of the four flags and on the §3 taxonomy itself, but two issues are load-bearing enough to require a v2 revision before downstream agents consume the memo.** Flag (a) — the Abdollahzadeh reinterpretation lemma establishing the Tsuda ⊂ FiLM ⊂ KML nesting — holds with a small clarification about operand type (conv kernel vs RNN recurrent matrix). Flag (b) — the FiLM-Ensemble + heteroscedastic-precision compound formula in Row EE-6 — has correct sign convention but two unspecified design choices (variance-space ambiguity and an unspecified $f$ in $\gamma_C = f(U, \sigma^{-2})$) that the postdoc must close before the formula carries the "novel substrate-level contribution" claim. Flag (c) — the Hessian-eigenvalue probe in §6.3 signature 2 — has the **direction of the effect reversed**: under Rodriguez-Garcia 2026 the raw-weight Hessian is *amplified* by $g^2$, not flattened; the flattening lives in the effective-parameter-space Hessian. This must be fixed before the probe is logged. Flag (d) — the Bellman γ vs FiLM γ disambiguation — recognises the collision but conflates GRU-update-gate-induced state retention with Bellman value-function discount; the dimensional content does not survive close reading. Beyond the four flags, two minor §5 / §6 issues. Overall verdict: **math is solid with two revisions required** before the memo can carry the unification claim. The integration claim itself (12-of-18 rows mapped, 3 honestly no-mapped) survives.

### Audit of flag (a) — Abdollahzadeh reinterpretation lemma

**Verdict: holds, with a one-clarification widening.**

The memo's §3.11 derivation is faithful to Abdollahzadeh 2021's Eq. (5), reproduced in the per-paper review at `docs/project/references/FiLM/reviews/abdollahzadeh_2021_multimodal_meta.md` lines 128-142:

$$
\hat Y_i = \eta_i(W_i * X + b_i) + \gamma_i = (\eta_i W_i) * X + (\eta_i b_i + \gamma_i),
$$

so vanilla FiLM on per-output-channel $\eta_i$ is equivalent to convolving with the kernel $\hat W_i = \eta_i W_i$ where *every* spatial position and every input-channel slice of $W_i$ is rescaled by the same scalar $\eta_i$. This is one degree of freedom per output channel. Per-weight modulation (KML) lifts this to one degree of freedom per weight. So vanilla FiLM is the per-output-channel-uniform restriction of KML — **the derivation is correct**.

The memo then extends the nesting to Tsuda 2021: $W \to f_{nm} \cdot W$ is the further-degenerate case where every entry of $W$ is rescaled by the *same* scalar $f_{nm}$. The memo writes this as $\hat W = W \odot (J + (f_{nm} - 1)J) = f_{nm} W$, which algebraically holds: $J + (f_{nm} - 1) J = f_{nm} J$, and $W \odot (f_{nm} J) = f_{nm} \cdot W$.

**Clarification the memo should add (minor, not a blocker).** Abdollahzadeh's reinterpretation lemma is stated for vanilla FiLM applied to a **convolutional layer** — the "channel" indexing is over conv output channels. Tsuda 2021 applies $f_{nm}$ to an **RNN recurrent weight matrix** $W \in \mathbb{R}^{N \times N}$, which does not have a "conv channel" structure. The nesting Tsuda ⊂ vanilla FiLM ⊂ KML is rigorous if we abstract "channel" as "output index of the weight matrix" (i.e., row index of $W$): vanilla FiLM allows one scalar per row, Tsuda forces all row scalars equal, KML allows one scalar per (row, column) entry. The chain holds; the memo should add one sentence noting the operand is a generic weight matrix, not specifically a conv kernel, when Tsuda is included.

**What this means for the memo's claim.** The headline that "vanilla FiLM on conv activations IS a degenerate (per-channel-uniform) kernel modulation; Tsuda 2021 is the further-degenerate scalar instance; KML is the most expressive instance" is **mathematically defensible**. The Tsuda ⊂ FiLM ⊂ KML nesting is the load-bearing math contribution of §3.11 and it holds. The v3 / predictions-synthesis Tsuda framing can be sharpened on this basis.

### Audit of flag (b) — FiLM-Ensemble + heteroscedastic-precision compound formula

**Verdict: holds on sign convention, but two design ambiguities must be closed before the formula carries the "project's unique novel-architecture result" claim.**

**Sub-check 1: Kendall-Gal sign convention.** The memo's Row EE-6 writes the loss as
$$
\mathcal{L} = \tfrac{1}{2} e^{-s} \|y - \hat y\|^2 + \tfrac{1}{2} s.
$$
This matches Kendall & Gal 2017 Eq. 6 (boxed in the per-paper review at `docs/project/references/FiLM/reviews/kendall_gal_2017_uncertainties.md` line 109) **exactly** — $s = \log \sigma^2$, the $\exp(-s)$ factor downweights noisy data, the $\tfrac{1}{2} s$ regulariser prevents the network predicting infinite noise everywhere. Both terms have the right sign, no double-counting. **Verified.**

**Sub-check 2: variance estimator space.** The memo defines
$$
U^{\text{epist}}_C(s) = \text{Var}_m[\pi^m(\cdot \mid s)],
$$
where $\pi^m$ is the $m$-th member's policy distribution. The postdoc flagged this needs to be in **post-softmax (probability simplex) space**, not logit space — otherwise epistemic uncertainty is unbounded and not interpretable as variance over predictions. The memo's notation $\pi^m(\cdot \mid s)$ reads as a probability distribution (softmax output), so this is *implicitly* correct, but **the memo never states it explicitly**. **Required revision:** explicitly state that $U^{\text{epist}}$ is computed on the post-softmax outputs $\pi^m \in \Delta^{|A|-1}$, not on pre-softmax logits $\ell^m \in \mathbb{R}^{|A|}$. Without this, an implementer could read the formula either way and get a different quantity. The post-softmax variance is bounded $[0, 0.25]$ per component (since each $\pi^m_i \in [0,1]$), making it interpretable; the logit-space variance is unbounded and would not function as a precision proxy.

**Sub-check 3: FiLM-Ensemble's gain-init $\rho$.** Turkoglu 2022's Eq. (5) (reproduced in `docs/project/references/FiLM/reviews/turkoglu_2022_film_ensemble.md` line 130) initialises $\gamma^m, \beta^m \sim \mathcal{U}(-\sqrt{3/D_n}\rho, +\sqrt{3/D_n}\rho)$. The paper's default is $\rho = 2$. The memo's flagged concern — that $\rho \to 0$ collapses the ensemble — is correct per Turkoglu §2.1 and the review at lines 33-36. **The memo should state $\rho = 2$ as the recommended starting value** (Turkoglu's empirical default) and flag that the compound EE-6 formula presumes non-degenerate $\rho$ at training start. Without this, the heteroscedastic head sees zero epistemic signal and the compound collapses to a single-network heteroscedastic-regression baseline. Worth noting: Turkoglu initialises $\gamma^m$ centred at zero (not at one, which is BatchNorm's usual init), so the $\rho \to 0$ limit is *not* "all-ones $\gamma$" but rather "all-zeros $\gamma$ everywhere", which would collapse all features. This subtlety affects how $\rho$ trades accuracy for diversity and is worth flagging in EE-6.

**Sub-check 4: coherent gradient flow.** The memo's claim is that the gradient signal to the FiLM-Ensemble members and the heteroscedastic $\sigma^{-2}$ "flow through both coherently". Concretely: if the heteroscedastic loss applies the $\tfrac{1}{2} e^{-s} \|y - \hat y\|^2$ residual term, the gradient wrt $\hat y$ scales as $e^{-s} (y - \hat y)$ — meaning *high predicted noise downweights the gradient to $\hat y$*. Through the FiLM-Ensemble chain, $\hat y$ depends on $(\gamma^m, \beta^m)$. So the per-member $(\gamma^m, \beta^m)$ gradients are also downweighted in high-$\sigma^2$ regions — this is consistent. The gradient wrt $s$ is $-\tfrac{1}{2} e^{-s} \|y - \hat y\|^2 + \tfrac{1}{2}$, which finds the optimum at $s^* = \log\|y - \hat y\|^2$ — the standard heteroscedastic fixed point. Gradients do not "fight each other" in the bad sense; they share a coherent block structure where $\sigma^2$ acts as a learning-rate gate on $(\gamma^m, \beta^m, \theta_{\text{backbone}})$. **No structural issue.** Worth noting in the memo: gradient block-diagonality is *not* exact (because both $\sigma^2$ and the ensemble share the backbone $\theta$), but the cross-coupling is benign (the high-$\sigma^2$ region is precisely where we want both heads' gradients downweighted). This is *not* a load-bearing concern.

**Headline for flag (b).** The compound formula is mathematically well-formed; the project can claim it as a unique combination. The two required revisions: (1) explicit statement that $U^{\text{epist}}$ is post-softmax, (2) explicit $\rho = 2$ starting value with note that the formula presumes non-degenerate ensemble spread at training start. The functional form $f$ in $\gamma_C(s) = f(U^{\text{epist}}, \sigma^{-2})$ is currently unspecified — the memo should at minimum state whether $f$ is learned (a small MLP) or hand-coded (e.g., $\gamma_C = a \cdot \sigma^{-2} + b \cdot U^{\text{epist}}$). Without this, the formula is closer to a *family* than a specific architecture.

### Audit of flag (c) — Hessian-eigenvalue probe

**Verdict: derivation of $\lambda \to \lambda/g^2$ is correct, but the discriminator stated in §6.3 signature 2 reverses the direction of the effect at the raw-weight Hessian.**

**The Rodriguez-Garcia derivation, verified.** Following the per-paper review at `docs/project/references/neuromodulatory_algorithms/reviews/rodriguezgarcia_2026_ne_stability_gap.md` lines 117-133:

Define $\tilde{\mathcal L}(W) = \mathcal L(\Phi(W))$ with $\Phi(W) = G W$, $G = g I_n$. Lemma 1 (gradient): $\nabla_W \tilde{\mathcal L} = G^\top \nabla_{W_{\text{eff}}} \mathcal L = g \nabla_{W_{\text{eff}}} \mathcal L$. Lemma 2 (Hessian congruence):
$$
\nabla^2_W \tilde{\mathcal L}(W) = G^\top \nabla^2_{W_{\text{eff}}} \mathcal L(\Phi(W)) G = g^2 H_{W_{\text{eff}}}(W_{\text{eff}}).
$$
Both chain-rule applications check out. The corollary:
$$
H_W = g^2 H_{W_{\text{eff}}} \quad \Longleftrightarrow \quad H_{W_{\text{eff}}} = H_W / g^2.
$$
So $\lambda_i^{\text{eff}} = \lambda_i^{W} / g^2$ — eigenvalues of the **effective-parameter-space** Hessian $H_{W_{\text{eff}}}$ are $1/g^2$ times the eigenvalues of the **raw-weight** Hessian $H_W$. **The derivation is correct.**

**Where the memo §6.3 signature 2 goes wrong.** The memo writes: "Under Rodriguez-Garcia 2026, the effective curvature flattens: $\lambda_i \to \lambda_i / g^2$. Under forward-pass FiLM, the *forward-pass conditioning* changes but the loss curvature at the raw weight should not flatten through the same gradient-level mechanism. **Discriminating reading**: curvature flattening at the raw-weight Hessian → gradient-level mechanism dominant; curvature unchanged at raw-weight but conditioning of effective forward pass changes → FiLM-level mechanism."

This is **reversed**. Under Rodriguez-Garcia, the raw-weight Hessian $H_W$ is **amplified** by $g^2$ relative to baseline (since $H_W = g^2 H_{W_{\text{eff}}}$, with $H_{W_{\text{eff}}}$ being the curvature on the actual function the network computes). When $g > 1$ (NA-burst regime), $H_W$ grows. The "flattening" only appears in $H_{W_{\text{eff}}}$ — the effective-parameter-space Hessian — which is *not directly accessible* unless the optimiser explicitly logs the composite parameter $W_{\text{eff}} = gW$.

**What the memo should say (and what the project should log).** Three distinct Hessians are at issue:
1. $H_W$ — Hessian wrt raw weight $W$. Under Rodriguez-Garcia with $g > 1$, this is **amplified by $g^2$**. Under FiLM, $W$ is unchanged by modulation, so $H_W$ is **unchanged** (FiLM acts on activations, not weights).
2. $H_{W_{\text{eff}}}$ — Hessian wrt the effective composite weight $W_{\text{eff}} = gW$. Under Rodriguez-Garcia, this is the "flattened" quantity ($1/g^2$ relative to $H_W$). Under FiLM, there is no equivalent composite weight, so this quantity is undefined.
3. $H_{\theta_{\text{mod}}}$ — Hessian wrt the FiLM modulator parameters that produce $(\gamma, \beta)$. Under FiLM, this is the natural curvature probe; under Rodriguez-Garcia, $g$ is not parameterised by a side network so this is irrelevant.

The memo's discriminator collapses these three. The corrected discriminator should be: **(a) measure $H_W$ on the raw weights pre-injection and post-injection of either mechanism; (b) Rodriguez-Garcia 2026 should produce $H_W$ amplification at $g > 1$ (not flattening), accompanied by gradient-step amplification; (c) FiLM should produce $H_W$ approximately unchanged and instead show modulator-parameter-Hessian $H_{\theta_{\text{mod}}}$ shifts**. The two are dissociable on the *direction* of the $H_W$ shift, not on its flattening.

**Severity.** This is a load-bearing error for the §7.4 claim that the Hessian-eigenvalue probe is the discriminating measurement between forward-pass FiLM and gradient-level NGM-SGD. If the project logs $H_W$ expecting flattening under Rodriguez-Garcia, it will measure amplification and incorrectly read the result. **Required revision** before downstream agents consume the memo.

### Audit of flag (d) — γ notational collision

**Verdict: the disambiguation $\tilde\gamma_{\text{Bellman}}$ is consistent in §5.3 and §6.3, but the underlying conceptual claim that "additive β at the GRU update-gate ⇔ effective Bellman γ change" is dimensionally incoherent.**

**The notational sweep.** I read §5.3 and §6.3 line-by-line for instances of bare γ that might be ambiguous between FiLM and Bellman. The memo uses $\gamma_C$ (FiLM at policy site C), $\gamma_A$ (FiLM at encoder site A), $\gamma^{(B)}$/$\beta^{(B)}$ (FiLM at GRU update-gate B), and $\tilde\gamma_{\text{Bellman}}$ where it means the Bellman value-function discount. **The notation is consistent**: every γ in §5.3 / §6.3 either has a site subscript (FiLM) or carries the $\tilde\cdot_{\text{Bellman}}$ marker (RL discount). No silent collisions found.

**The dimensional incoherence.** The memo writes (§5.3, mathematical sketch box):
$$
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z + \beta^{(B)}(\text{mod\_h}_t^T)),
$$
"where $\beta^{(B)}$ shifts the 'stay vs update' decision boundary additively" and "the state-dependent effective discount over past memory is $\tilde\gamma_{\text{Bellman}}(\beta^{(B)})$".

This is dimensionally incoherent. The GRU update gate $z_t \in (0,1)^H$ is a **per-hidden-unit** retention scalar that produces the recurrent update $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde h_t$. The "discount" induced by $z_t$ is the *hidden-state decay factor*, a per-unit quantity bounded by $(0,1)$, with units of "retention probability per timestep of the inner recurrent loop". The Bellman value-function discount $\gamma_{\text{Bellman}}$ is a **global scalar** (typically ~0.99) that governs the value-function horizon: $V^\pi(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma_{\text{Bellman}}^t r_t]$. The two operate at different layers of the algorithm:
- GRU retention: at the policy network's recurrent loop, per-unit, per inner-network-timestep.
- Bellman γ: at the value-target computation, global, per environment-timestep.

A perturbation of the GRU's $z_t$ via additive $\beta^{(B)}$ changes how fast the hidden state decays. This affects *what the policy can remember*, not *how far into the future the value function looks*. The two effects can correlate empirically (longer hidden-state memory → effective longer-horizon credit assignment), but they are not the same quantity and one does not equal the other.

**Severity.** This is the same conflation `professor-rl-bayesian-dl` warned against in v1 §8 Q3 (per the user's brief). The memo's recognition that "the math-reviewer should flag a notational collision" is a half-recognition — the notation is fine; the *concept* is the issue. **Required revision**: rewrite the §5.3 sketch-box to drop the $\tilde\gamma_{\text{Bellman}}(\beta^{(B)})$ formulation entirely and instead describe the effect as "additive β at the GRU update gate shifts the per-unit hidden-state retention, which can be measured as the autocorrelation time constant $\tau$ of the recurrent state". The Bellman discount lives in the value-target computation and is *not* what β at the update gate modulates. The §6.3 signature 3 ("R2 readout") survives because it does not depend on the Bellman-γ framing.

### Sweep findings (other rows in §3, §5, §6)

**§3.2 CIN equation.** Correct — matches Dumoulin 2017 Eq. (1). No issue.

**§3.5 HyperNetwork equation.** The memo writes $K^j = g(z^j) = \langle W_{\text{out}}, W_i z^j + B_i \rangle + B_{\text{out}}$ with the angle-bracket notation. The Galanti-Wolf modularity-bound claim ($N_g = O(\epsilon^{-m_1/r})$ vs $N_q = \Omega(\epsilon^{-(m_1 + m_2)/r})$) is correctly attributed but the asymptotic-rate comparison should specify that this is for a target function family of complexity class $W^{r,m}$ and that $m_1, m_2$ are the conditioning and target dimensions respectively. Minor — not a blocker.

**§3.7 FiLM-Ensemble equation.** Matches Turkoglu Eq. (2). Verified.

**§3.9 Sparse MoE equation.** Matches Shazeer 2017 §3. Verified.

**§5.1 Row CS-1 Vecoven bridge.** The memo writes:
$$
\sigma_{\text{NMN}}(x, z) = \sigma(z^\top (x w_s + w_b)) = \sigma((z^\top w_s) \odot x + (z^\top w_b)) = \sigma(\gamma(z) \odot x + \beta(z)).
$$
The middle equality uses $\odot$ where the actual operation is scalar multiplication (since $x \in \mathbb{R}$ is per-neuron scalar). This is a notation-tightening matter; the substantive algebra is correct. Verified that the bridge holds.

**§5.1 Row CS-5 Costacurta bridge.** The memo writes $W_x(z) = \sum_{k=1}^K s_k(z) \ell_k r_k^\top$ with $s_k(z) = \sigma(A_z z + b_z)_k$. This matches the per-paper review's Section 3 equations precisely. The LSTM-equivalence (Prop. 1) holds. Verified.

**§5.4 Row EE-1 Doya bridge.** The memo writes $\ell'_a = \gamma_C(\text{mod\_h}) \cdot \ell_a + \beta_C(\text{mod\_h})$ and compares to temperature-scaled softmax $\pi(a) \propto \exp(\ell_a / T)$, identifying $\gamma_C = 1/T$. This is correct **provided $\beta_C = 0$**. When $\beta_C \ne 0$, the result is *not* a pure temperature rescaling — it is a temperature rescaling plus a logit shift (which biases the policy toward / away from specific actions independent of value). The memo should note that the Doya-NA-β-inverse-temperature mapping holds only on the multiplicative arm $\gamma_C$; the additive arm $\beta_C$ has a separate behavioural interpretation (action-prior shift, *not* inverse temperature). This is a minor revision.

**§5.4 Row EE-2 Lee 2024 bridge.** The hand-coded formula $\alpha = E/(E+A), \beta^{-1} = 1/\langle E\rangle$ matches Lee 2024 Eq. 4. The analogy to AdaIN's closed-form generator is structurally apt — both produce $(\gamma, \beta)$ from input statistics without learning. Verified.

**§5.4 Row EE-7 Osman no-map.** The memo notes Osman's $W_{\text{rec}} \to g \cdot W_{\text{rec}}$ has "the same mathematical form as Tsuda 2021". This is correct, and the memo correctly flags that the *operand* (recurrent attractor network, not feedforward / GRU policy) is what differentiates Osman from FiLM. The no-map verdict holds.

**§6.1 Effective-rank formula.** The formula
$$
\text{erank}(\gamma_C) := \exp\left(-\sum_i \frac{\sigma_i^2}{\sum_j \sigma_j^2} \log \frac{\sigma_i^2}{\sum_j \sigma_j^2}\right)
$$
is the **exponential of the Shannon entropy of the normalised squared singular values** — a standard effective-rank definition (Roy & Vetterli 2007). Correct.

**§5.5 NO-MAP boundary verification.** Wainstein 2025's $g$ is described as "an experimental dial on a *post-hoc* trained-RNN analysis, not as a side-signal-conditioned forward-pass operator". This matches the per-paper review's description (the RNN was trained without pupil input). The pupil-diameter "side signal" correlates with $g$ but does not generate it. The no-map verdict holds.

### Overall verdict

**The memo's headline integration claim is mathematically defensible — 12-of-18 rows mapped, 3 honestly no-mapped, with the Abdollahzadeh nesting result as the load-bearing math contribution — but the memo needs two required revisions and three minor revisions before downstream agents consume it.**

**Required revisions (must be fixed before `experiment-designer` or `senior-developer` use the memo):**

1. **§6.3 signature 2 Hessian-eigenvalue probe** — reverse the discriminator direction. Under Rodriguez-Garcia 2026, the raw-weight Hessian $H_W$ is **amplified** by $g^2$ (not flattened); the flattening lives in the effective-parameter-space Hessian $H_{W_{\text{eff}}} = H_W / g^2$. The corrected discriminator measures $H_W$ shift direction (amplification under Rodriguez-Garcia, no change under FiLM), not flattening at $H_W$. This is the §7.4 claim's foundation, so it must be right.

2. **§5.3 sketch-box Bellman-γ conflation** — drop the $\tilde\gamma_{\text{Bellman}}(\beta^{(B)})$ formulation. The GRU update gate $z_t$ produces per-unit hidden-state retention, which is dimensionally distinct from the global Bellman value-function discount. Rewrite as "additive β at the GRU update gate shifts the per-unit hidden-state retention $\tau_h$, measurable as recurrent-state autocorrelation".

**Minor revisions (not blockers, but should be tightened):**

3. **§5.4 Row EE-6 compound formula** — explicitly state (a) $U^{\text{epist}}$ is computed on post-softmax outputs $\pi^m \in \Delta^{|A|-1}$ (not pre-softmax logits); (b) the gain-initialisation $\rho = 2$ as starting value per Turkoglu 2022 §2.1; (c) the functional form of $f$ in $\gamma_C(s) = f(U^{\text{epist}}, \sigma^{-2})$ — learned (small MLP) or hand-coded.

4. **§3.11 nesting clarification** — add one sentence noting the Abdollahzadeh lemma originally targets conv kernels; the Tsuda 2021 extension generalises "channel" to "row index of a generic weight matrix", which is what makes the nesting Tsuda ⊂ FiLM ⊂ KML work for RNN recurrent matrices.

5. **§5.4 Row EE-1 Doya bridge** — note that $\gamma_C \equiv 1/T$ holds for the multiplicative arm only; the additive arm $\beta_C$ has a separate behavioural interpretation as an action-prior shift, not as part of the inverse-temperature mapping.

### Recommendations

The memo's central contribution — that the FiLM/hypernet family forms a mathematically articulated taxonomy that integrates roughly twelve of eighteen load-bearing neuromodulation-inspired algorithms — survives the audit. The Abdollahzadeh reinterpretation lemma is correctly derived. The compound EE-6 formula has correct sign conventions but unspecified design choices. The two required revisions (flag c and flag d) are both about *interpretation* rather than fundamental algebra — they reverse a direction and dismantle a conceptual conflation respectively. Neither requires re-derivation from scratch.

**Recommended next step:** the postdoc revises §5.3, §5.4, §6.3, and §3.11 with the five items above. A v2 of the memo can then carry the unification claim defensibly. **The postdoc does NOT need to be re-spawned**; the required revisions are surgical and targeted at four sub-sections.

The memo does *not* contain a math error so badly wrong that it invalidates the integration claim. The two required revisions are about Hessian-direction and Bellman-γ conflation respectively, neither of which undermines §3 (the taxonomy), §5 rows 1, 3, 4, 5, 6 (the cleanly-mapped neuromod algorithms), §5.5 (the no-map boundary), or §6.1 / §6.2 / §6.4 (the per-behaviour signatures other than §6.3 signature 2). Approximately 90% of the memo's mathematical content is unaffected.

### Sign-off

*--- Audit by `math-reviewer`, 2026-05-16.*

*[Reserved for math-reviewer feedback. Math-reviewer should attend particularly to: the Abdollahzadeh 2021 reinterpretation lemma derivation (§3.11) establishing the FiLM/Tsuda nesting; the FiLM-Ensemble + heteroscedastic-precision compound formula (Row EE-6 in §5.4); the Hessian-eigenvalue probe formula (§6.3 signature 2); the notational collision flagged in §5.3 mathematical sketch between FiLM's $\gamma$ and the Bellman-discount $\gamma$.]*
