---
title: "FiLM variants and neuromodulatory algorithms — integration synthesis"
topic: FiLM
status: draft
last_updated: 2026-05-16
audience: research-postdoc, professor-neuromodulation, professor-rl-bayesian-dl, professor-bayesian-brain, professor-pain-modeling, experiment-designer, senior-developer
related:
  - film_lit_review.md
  - film_synthesis.md
  - ../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md
  - ../neuromodulatory_algorithms/neuromodulatory_algorithms_lit_review.md
  - ../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md
purpose: "Evidence base for a paper-shape standalone concept memo (next author: research-postdoc). Question answered — which neuromodulatory algorithms in the corpus map onto a FiLM-variant operation, with what mathematical bridge, preserving which behaviour-level benefit. Section 3 carries the explicit row-by-row mapping. Honest about clean-fit, extension-fit, partial-fit, and no-fit cases."
---

# FiLM variants and neuromodulatory algorithms — integration synthesis

## Plain-English entry point

This synthesis is the **evidence base for a paper-shape claim**: that the FiLM family (Feature-wise Linear Modulation and its near relatives — hypernetworks, conditional batch / instance norm, FiLM-Ensemble, BatchEnsemble, mixture-of-experts, attention) provides a **single unifying architectural substrate** into which a wide swath of biologically-inspired *neuromodulator-style* algorithms can be re-cast as forward-pass operations on a learnable agent. The claim is **not** "every neuromodulator paper is FiLM in disguise". It is "the **potential** to integrate exists, here are the explicit mathematical bridges, here is what fits cleanly, what fits with extension, and what does not fit at all".

To support that claim, this document does five things. **Section 1** enumerates the operationally distinct FiLM-variants the FiLM corpus contains — vanilla FiLM, conditional instance / batch norm, AdaIN, temporal FiLM, full hypernetwork, Bayesian hypernet, FiLM-Ensemble, BatchEnsemble, sparse mixture-of-experts, attention-as-gating, multi-task feature modulation, kernel modulation — with each operation's canonical equation. **Section 2** lays out the load-bearing neuromodulatory algorithms from the project's neuromod corpus and the **four target behaviour-level benefits** the paper-shape memo will stake claims on: **context-switching / regime change**, **hypervigilance / sustained tonic vigilance**, **plasticity gating / lifelong-learning resistance**, and **exploration–exploitation control**. **Section 3** is the load-bearing row-by-row integration mapping — 18 neuromodulatory rows, each tagged CLEAN MAP / EXTENSION MAP / PARTIAL MAP / DOES NOT MAP, each with its mathematical bridge and target-behaviour tag. **Section 4** organises the result by behaviour-level benefit, ending each sub-section with a one-line headline claim. **Section 5** lists the gaps the FiLM corpus reveals and which the project's substrate is uniquely positioned to fill. **Section 6** is a service reading guide.

**Headline tag distribution from §3**: of the 18 rows, **7 are CLEAN MAP**, **5 are EXTENSION MAP**, **3 are PARTIAL MAP**, and **3 are DOES NOT MAP**. The three no-maps (Rodriguez-Garcia 2026's gradient-level gain, Wainstein 2025's trained-RNN internal gain, Osman 2024's Hopfield-attractor gain) are diagnostic, not failures: they reveal that the FiLM corpus's "side-signal → forward-pass affine" template does not cover *gradient-level optimiser-gain*, *trained-RNN-gain-as-internal-knob*, or *recurrent-energy-landscape-gain* without further machinery. The 12 cleanly-or-with-extension fitted rows are enough to support the paper-shape claim.

---

## Section 1 — FiLM-variant taxonomy

The FiLM corpus (23 papers, indexed in [`film_lit_review.md`](film_lit_review.md), synthesised in [`film_synthesis.md`](film_synthesis.md)) contains roughly **ten operationally distinct variants** of the "conditioning a network on a side signal" mechanism. Each variant below is named, equation-anchored, paper-anchored, characterised by *where* in the network it acts (activations / weights / normalisation statistics / attention positions / gradient — though gradient is not represented in this corpus), and tagged with the behaviour-level benefit demonstrated in the source paper.

The variants are not mutually exclusive. AdaIN is a special case of CIN with the lookup replaced by a closed-form function; FiLM-Ensemble is CIN with the conditioner replaced by a member index; BatchEnsemble is FiLM moved from the output of a normalisation layer to the *input* of a linear layer; sparse MoE is a discrete hypernetwork. The taxonomy is a partial order, not a partition.

### 1.1 Vanilla FiLM — per-channel affine on activations

**Equation.** For feature map $F_{i,c} \in \mathbb{R}^{H \times W}$ at channel $c$, example $i$, and a conditioning input $x_i$ (e.g., a question, a task embedding, a state vector):

$$
\gamma_{i,c} = f_c(x_i),\quad \beta_{i,c} = h_c(x_i),
\qquad
\mathrm{FiLM}(F_{i,c} \mid \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}.
$$

**Canonical paper.** Perez et al. 2018 ([review](reviews/perez_2018_film.md)).

**Where it acts.** Forward-pass activations, per channel, spatially uniform within the channel.

**Behaviour-level benefit in source paper.** Question-conditioned visual reasoning on CLEVR — halves prior state-of-the-art error using only per-channel affine modulation; demonstrates that the $(\gamma, \beta)$ parameter space is linearly composable (zero-shot attribute composition).

### 1.2 Conditional Instance Normalization (CIN)

**Equation.**

$$
\mathrm{CIN}(x \mid s)_{n,c,h,w} = \gamma_{s,c} \cdot \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}} + \beta_{s,c},
$$

with $(\gamma_{s,c}, \beta_{s,c})$ the $s$-th row of an $N \times C$ learned lookup table indexed by style $s$, and $(\mu_{n,c}, \sigma_{n,c})$ instance statistics computed per example, per channel.

**Canonical paper.** Dumoulin et al. 2017 ([review](reviews/dumoulin_2017_cond_instance_norm.md)).

**Where it acts.** Forward-pass activations, after per-example feature-statistic standardisation, per channel.

**Behaviour-level benefit in source paper.** Multi-style image stylisation — 32 painting styles in one network at 0.2% style-specific parameters; linear interpolation in $(\gamma, \beta)$ space produces smooth pastiches.

### 1.3 Adaptive Instance Normalization (AdaIN)

**Equation.**

$$
\mathrm{AdaIN}(x, y)_{n,c,h,w}
= \sigma_{n,c}^{\mathrm{IN}}(y) \cdot \frac{x_{n,c,h,w} - \mu_{n,c}^{\mathrm{IN}}(x)}{\sigma_{n,c}^{\mathrm{IN}}(x)} + \mu_{n,c}^{\mathrm{IN}}(y).
$$

The $(\gamma, \beta) = (\sigma^{\mathrm{IN}}(y), \mu^{\mathrm{IN}}(y))$ are *computed on the fly* from a second image's per-channel feature statistics — zero learnable parameters in the modulation layer itself.

**Canonical paper.** Huang & Belongie 2017 ([review](reviews/huang_belongie_2017_adain.md)).

**Where it acts.** Forward-pass activations, after per-example feature-statistic standardisation, per channel; $(\gamma, \beta)$ are *closed-form* functions of the side signal's own deep features.

**Behaviour-level benefit in source paper.** Real-time arbitrary style transfer — any style at test time, including styles never seen at training.

### 1.4 Temporal FiLM (TFiLM)

**Equation.** Split the time axis into $B$ blocks; pool each block per-channel; run an LSTM over the sequence of pooled vectors:

$$
\bigl((\gamma_b, \beta_b), h_b\bigr) = \mathrm{LSTM}\bigl(F^{\mathrm{pool}}_{b,:}; h_{b-1}\bigr),
\qquad F^{\mathrm{norm}}_{b,t,c} = \gamma_{b,c} \cdot F^{\mathrm{blk}}_{b,t,c} + \beta_{b,c}.
$$

$(\gamma_b, \beta_b)$ depend on the *entire history* of pooled blocks via the LSTM.

**Canonical paper.** Birnbaum et al. 2019 ([review](reviews/birnbaum_2019_temporal_film.md)).

**Where it acts.** Forward-pass activations, time-block piecewise-constant, conditioner is self-recurrent (the LSTM observes the network's own pooled state).

**Behaviour-level benefit in source paper.** Long-range sequence dependency at low compute cost — bridges 1-D CNN locality with RNN global context for audio super-resolution, text classification, ChIP-seq super-resolution.

### 1.5 STSM-FiLM (identity-initialised FiLM with scalar conditioner)

**Equation.**

$$
\hat f_t = (1 + \gamma_\alpha) \cdot f_t + \beta_\alpha,
\qquad (\gamma_\alpha, \beta_\alpha) = \mathrm{MLP}(\alpha),
$$

with $\alpha \in \mathbb{R}$ a scalar conditioner. The `1 +` ensures the model initialises at the identity transformation.

**Canonical paper.** Wisnu et al. 2025 ([review](reviews/wisnu_2025_stsm_film.md)).

**Where it acts.** Forward-pass activations, per channel; conditioner is a single scalar (a speed factor).

**Behaviour-level benefit in source paper.** Continuous control over a single behavioural axis (speech time-scale modification) via a scalar modulator.

### 1.6 Multi-task feature modulation (IN + FiLM at every layer)

**Equation.** Same as CIN, but the conditioner is a one-hot-or-mixture task vector $c \in [0,1]^n$ rather than a discrete style index:

$$
\gamma_i = f_{\gamma, i}(c),\quad \beta_i = f_{\beta, i}(c),
\qquad
\mathrm{FiLM}\!\bigl(F_i \mid \gamma_i, \beta_i\bigr) = \gamma_i \cdot F_i + \beta_i,
$$

inserted after every Instance Normalisation layer (IN + FiLM = AdaIN). Mixed-task training requires synthesised mixed-task ground-truth samples (Method 3) for the conditional vector to behave as a smooth mixture coordinate at inference.

**Canonical paper.** Takeda et al. 2021 ([review](reviews/takeda_2021_multi_task_feature_mod.md)).

**Where it acts.** Forward-pass activations after every IN layer; conditioner is a *continuous task vector* with $c = \mathbf{0}$ encoding the identity transformation.

**Behaviour-level benefit in source paper.** 6 heterogeneous image-to-image tasks (reconstruction, inpainting, denoising, segmentation, two styles) coexist on one shared backbone at <2% task-specific parameters; the conditional vector acts as a smooth mixture coordinate.

### 1.7 HyperNetwork (full weight generation)

**Equation.** Static (CNN) form: for each layer $j$, a learned embedding $z^j$ is fed through a two-layer linear hypernet that outputs the full conv kernel,

$$
K^j = g(z^j) = \langle W_{\mathrm{out}}, W_i z^j + B_i \rangle + B_{\mathrm{out}}.
$$

Dynamic (RNN / HyperRNN) form: the recurrent weights vary at every timestep as functions of a per-timestep embedding $z$. The memory-efficient "scaling-vector trick" replaces a dense weight tensor with per-row scaling:

$$
h_t = \phi\!\bigl( d_h(z_h) \odot (W_h h_{t-1}) + d_x(z_x) \odot (W_x x_t) + b(z_b) \bigr),
$$

with $d_h(z_h) = W_{hz} z_h$ — *structurally identical to FiLM at the per-row granularity*, two years before FiLM was named.

**Canonical paper.** Ha, Dai & Le 2016 ([review](reviews/ha_2016_hypernetworks.md)).

**Where it acts.** Weight matrices (static) or weight matrices per timestep (dynamic).

**Behaviour-level benefit in source paper.** Context-dependent recurrent dynamics — HyperLSTM beats LSTM and Layer-Norm-LSTM on character PTB, enwik8, IAM handwriting, and WMT'14 En→Fr translation; weights change abruptly at word/phrase boundaries.

**Theoretical anchor.** Galanti & Wolf 2020 ([review](reviews/galanti_wolf_2020_hypernet_modularity.md)) prove the modularity bound: for hypernet $h(x, I) = g(x; f(I; \theta_f))$ approximating $W^{r,m}$,

$$
N_g = O(\epsilon^{-m_1/r})
\quad\text{vs.}\quad
N_q = \Omega\bigl(\epsilon^{-(m_1 + m_2)/r}\bigr)
$$

for embedding-concatenation — capacity is absorbed into the hypernetwork, *exponentially* in the conditioning dimension $m_2$.

### 1.8 Bayesian Hypernetwork

**Equation.** The primary weights are sampled via an invertible normalising flow $h$:

$$
\theta = h(\epsilon),\qquad \epsilon \sim \mathcal{N}(0, I_D),
$$

$$
\log q(\theta) = \log q_\epsilon(\epsilon) - \log \Bigl| \det \frac{\partial h(\epsilon)}{\partial \epsilon} \Bigr|.
$$

**Canonical paper.** Krueger et al. 2017 ([review](reviews/krueger_2017_bayesian_hypernets.md)).

**Where it acts.** Weight matrices (or, in the efficient parametrisation, only weight-norm scaling factors $g$ in a weight-normalisation reparametrisation).

**Behaviour-level benefit in source paper.** A full multi-modal correlated posterior over weights — better adversarial-example detection and active learning than mean-field VI / MC-dropout / deterministic baselines.

### 1.9 FiLM-Ensemble

**Equation.** Replace the continuous conditioner $z$ with a discrete member index $m \in \{1, \ldots, M\}$, store per-member $(\gamma^m_n, \beta^m_n) \in \mathbb{R}^{D_n}$ as parameters per normalisation layer per member:

$$
\mathrm{FiLM}\!\bigl(F_n \mid \gamma^m_n, \beta^m_n\bigr) = \gamma^m_n \circ F_n + \beta^m_n.
$$

Initialise $\gamma^m_n, \beta^m_n$ from a Xavier-uniform with tunable gain $\rho$; $\rho \to 0$ collapses members, larger $\rho$ pushes them apart.

**Canonical paper.** Turkoglu et al. 2022 ([review](reviews/turkoglu_2022_film_ensemble.md)).

**Where it acts.** Forward-pass activations at every BatchNorm layer; conditioner is a member index, applied $M$ times to the same input in parallel.

**Behaviour-level benefit in source paper.** Cheap epistemic uncertainty — 16-member ensemble at 1.3% parameter overhead (vs. 1500% for an explicit ensemble of ResNet-18); *higher* member diversity than the explicit ensemble at $M=16$ on CIFAR-10 (9.2% pairwise disagreement vs. 6.8%).

### 1.10 BatchEnsemble / TabM

**Equation.** Per-member rank-1 modulation of a shared weight matrix:

$$
W_i = W \odot (s_i\, r_i^\top),
\qquad
l_i(x_i) = s_i \odot \bigl(W (r_i \odot x_i)\bigr) + b_i.
$$

**Canonical paper in this corpus.** Gorishniy et al. 2025 (TabM, [review](reviews/gorishniy_2025_tabm.md)); the BatchEnsemble underlier is Wen et al. 2020 (out of corpus, but cited and re-instantiated by TabM).

**Where it acts.** Around the linear-layer *weight* (not the normalisation-layer affine). Same multiplicative-gating inductive bias as FiLM, *one operator-position over*.

**Behaviour-level benefit in source paper.** Parameter-efficient implicit ensembling for tabular data — TabM Pareto-dominates 46-dataset benchmark over attention- and retrieval-based tabular DL methods, matches gradient-boosted trees.

### 1.11 Sparsely-Gated Mixture-of-Experts (MoE)

**Equation.** A bank of $n$ experts $E_1, \dots, E_n$ plus a gating network $G$:

$$
y = \sum_{i=1}^n G(x)_i \cdot E_i(x),
\qquad
G(x) = \mathrm{Softmax}\bigl(\mathrm{KeepTopK}(H(x), k)\bigr),
$$

$$
H(x)_i = (x W_g)_i + \mathcal{N}(0,1) \cdot \mathrm{Softplus}\!\bigl((x W_{\mathrm{noise}})_i\bigr).
$$

Most $G(x)_i$ are exactly zero; only $k$ experts compute per example.

**Canonical paper.** Shazeer et al. 2017 ([review](reviews/shazeer_2017_sparse_moe.md)).

**Where it acts.** Discrete routing across expert sub-networks; the gate is a *discrete-sparse* hypernetwork that selects which sub-architecture is active for each input.

**Behaviour-level benefit in source paper.** Trillion-parameter models with bounded per-example compute — 24% lower perplexity than computational-budget-matched baseline on the 1B-Word LM benchmark; 137B-parameter models on the 100B-Word Google News corpus.

### 1.12 Attention as gating

**Equation.** Scaled dot-product attention:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Bigl(\frac{Q K^\top}{\sqrt{d_k}}\Bigr) V.
$$

**Canonical paper.** Vaswani et al. 2017 ([review](reviews/vaswani_2017_attention.md)).

**Where it acts.** Forward-pass *positions* (not channels) — every position forms a softmax-weighted aggregation over positions in a source stream.

**Behaviour-level benefit in source paper.** Global context modelling with $O(1)$ path length between long-range dependencies; SOTA on WMT'14 En→De and En→Fr translation.

**Empirical comparison with FiLM in the corpus.** Vuorio et al. 2019 (cited in [Turkoglu 2022 review](reviews/turkoglu_2022_film_ensemble.md) and [Abdollahzadeh 2021 review](reviews/abdollahzadeh_2021_multimodal_meta.md)): "FiLM outperforms attention-based modulation in this context [multimodal MAML] and is more stable" — when the side signal is low-dimensional, FiLM wins.

### 1.13 Kernel Modulation (KML) — per-weight modulation

**Equation.** Replace the per-channel scalar $\gamma$ with a per-weight modulation matrix $M^l$ of the same shape as the convolutional kernel $W^l$:

$$
\hat W^l_T = W^l \odot \bigl(J + M^l(\upsilon_T, \phi)\bigr),
\qquad
M^l(\upsilon_T, \phi) = g^l_{\phi_1}(\upsilon_T) \otimes g^l_{\phi_2}(\upsilon_T),
$$

with $J$ the all-ones matrix and the outer-product factorisation $\otimes$ keeping the generator small even when $M^l$ has the full kernel shape.

**Canonical paper.** Abdollahzadeh et al. 2021 ([review](reviews/abdollahzadeh_2021_multimodal_meta.md)).

**Where it acts.** Conv-kernel weights, per weight (not per channel), residual offset from the un-modulated kernel.

**Behaviour-level benefit in source paper.** Highly multimodal meta-learning — KML beats FiLM-in-MMAML by ~5pp on 5-mode few-shot benchmarks. The mathematically load-bearing observation: **FiLM is equivalent to *uniform scalar rescaling of a conv kernel*** (the per-channel $\gamma$ scales the whole kernel slice into a channel by the same factor); KML lifts this restriction.

### 1.14 The Module-Networks family (Andreas 2016 / Hu 2017)

**Equation.** Discrete-symbolic — for input $w$ parsed into symbolic form $\sigma(w)$, the layout tree $T = P(w)$ is mapped recursively to a tree of typed neural modules and executed:

$$
p(y \mid w, x; \theta) = M_T\bigl(x; \{\theta_m\}_{m \in T}\bigr).
$$

**Canonical papers.** Andreas et al. 2016 ([review](reviews/andreas_2016_neural_module_networks.md)); Hu et al. 2017 ([review](reviews/hu_2017_e2e_module_networks.md)).

**Where it acts.** *Discrete-composition* of an entirely-new computation graph per input — a fork of the FiLM family rather than a member of it. We flag this here so we do not silently absorb it into the FiLM mapping. NMN is the *symbolic-discrete* cousin of the FiLM / hypernet / MoE family.

---

## Section 2 — Neuromodulatory algorithms and target behaviour-level benefits

The neuromod corpus's load-bearing extraction lives in [`neuromodulatory_algorithms_predictions_synthesis.md`](../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md). That document's §2 (mechanism inventory) and §3 (predictions per hyperparameter) are the canonical material we map from here.

The mechanism inventory contains ten variants of "neural gain", grouped by where each acts and how. The variants relevant to the FiLM mapping are summarised below together with the *target behaviour-level benefit* the source paper claims, and the *behaviour-category tag* it instantiates among the four locked categories — **CS** (context-switching / regime change), **HV** (hypervigilance / sustained tonic vigilance), **PG** (plasticity gating / lifelong-learning resistance), **EE** (exploration–exploitation control).

| Neuromod algorithm | Cellular mechanism (predictions-synthesis §2) | Behaviour-level benefit (source claim) | Target-behaviour tag |
|---|---|---|---|
| Doya 2002 four-knob mapping | One scalar per neuromodulator system retunes one RL hyperparameter (DA-δ, ACh-α, NA-β, 5-HT-γ) | Foundational framework; predicts hyperparameter movement under modulator tone | EE, CS, PG (overall framework) |
| Ferguson & Cardin 2020 (gain types) | Multiplicative on slope; additive on rheobase; divisive normalisation | Sensitivity vs selectivity dissociation in V1 / attention | EE (gain ↔ inverse-temperature), HV (selectivity) |
| Shine 2021 (cellular ↔ network) | $g = dQ/dI$ slope at the I/O curve; reaches into network-level dynamics | Yerkes-Dodson arousal–performance curve; energy-landscape flattening | EE, CS |
| Vecoven 2020 (NMN-A2C) | $\sigma_{NMN}(x, z; w_s, w_b) = \sigma(z^\top(x w_s + w_b))$ — FiLM on activations | Meta-RL navigation; $z$-recruitment varies with task | CS, EE |
| Tsuda 2021 ("hypertubes") | $W \to f_{nm} \cdot W$ uniform scalar on weights | Distinct context-relevant behaviours via PCA-space "hypertube" trajectory shifts | CS |
| Costacurta 2024 (NM-RNN) | $W_x(z) = \sum_k s_k(z) \ell_k r_k^\top$ — low-rank weight scaling; equivalent to LSTM forget gate (Prop. 1) | Dissociable-by-ablation multi-channel modulation (Fig. 3F) | CS, PG |
| Rodriguez-Garcia 2026 (NGM-SGD) | $g(t+1) = \gamma g(t) + (1-\gamma) g_0 + \eta H(y)$; effective curvature $\lambda \to \lambda / g^2$ | Stability-gap attenuation across Split MNIST / CIFAR / mini-ImageNet | PG |
| Lee 2024 (Doya–DaYu hand-coded) | $\alpha(s,a) = E/(E+A)$; $\beta^{-1}(s) = 1/\langle E(s,\hat a)\rangle$ — hand-coded functional forms from ensemble uncertainty | Recovery in non-stationary multi-armed bandit | CS, EE |
| Xing 2022 (RL under env changes) | Context-conditioned modulation triggered by environment change | Faster post-switch recovery, less catastrophic forgetting | CS, PG |
| Mei 2022 (review) | Multiscale modulation, hyperparameter analogy | Programmatic claim that "neuromodulation = hyperparameter fine-tuning" | (framing only) |
| Wainstein 2025 (perceptual switches) | Trained RNN gain $g$ modulating activation function in correlation with pupil | Pupil-locked perceptual switch latency; energy-landscape flattening at switch | CS, EE |
| Wang 2024 (NeuroNML) | Flexible-network-structure via bi-level optimisation; structure-mask | Meta-learning fast adaptation | PG (CL-flavoured) |
| Ben-Iwhiwhu 2022 (context meta-RL) | Activity-gating: $h \to g \odot h$ before nonlinearity | Richer latent representations; CAVIA / PEARL improvements | CS, EE |
| Driscoll 2022 (dynamical motifs) | Multi-task RNN reuses dynamical motifs | Architectural touchstone for shared dynamics across tasks | (framing only) |
| Kudithipudi 2022 (lifelong-learning survey) | Surveys plasticity-gating biological mechanisms | Programmatic framing for lifelong-learning in NN | PG |
| Osman 2024 (Hopfield + gain) | $W_{\mathrm{rec}} \to g \cdot W_{\mathrm{rec}}$; low $g$ flattens attractor landscape | Arousal-modulated Bayesian inference temperature schedule | EE (annealing-style) |
| AlKilany & Goodman 2025 | Per-neuron excitability multiplier in spiking networks | Reaction-time decrease ("listening in the dips") | EE, CS |
| Tambaş 2025 (Krotov-Hopfield) | Three-factor rule: global modulator gates local plasticity | Global signal modulates local learning | PG |

(Eighteen rows because we have folded Mei 2022 and Kudithipudi 2022 in as framing-only neuromodulation entries — they do not propose a new mechanism but supply the umbrella for several of the others. Doya 2002 is the foundational mapping that all four behaviour categories inherit from.)

For each of the four behaviour categories, the canonical readouts the corpus uses are:

- **Context-switching / regime change (CS)**: post-switch recovery half-life; actor-entropy spike at switch; $|\delta|$-tracking by the modulator; modulator-state PCA velocity around the switch boundary (Wainstein 2025; Lee 2024; Xing 2022; the project's R2 empirical anchor).
- **Hypervigilance / sustained tonic vigilance (HV)**: post-injury sustained elevation of channel-selective gain on threat-relevant inputs; tonic state's hysteresis after the aversive regime passes; near-injury vs far-from-injury action-entropy contrast (the project's pain-construct readouts per [`v3 §5.2`](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)).
- **Plasticity gating / lifelong-learning resistance (PG)**: stability-gap attenuation under continual evaluation; transient accuracy-drop at task boundaries; Rodriguez-Garcia 2026's Hess-protocol per-batch eval; the project's R2 return-after-dormancy win.
- **Exploration–exploitation control (EE)**: action-entropy / softmax-temperature regression on putative neuromodulator tone; ensemble disagreement as the signal that drives temperature (Lee 2024 $\beta^{-1}(s) = 1/\langle E(s,\hat a)\rangle$); pupil-locked perceptual switches (Wainstein 2025); precision-weighted gating of evidence under noisy observation (Yu & Dayan 2005-style ACh-as-precision).

The mapping in §3 will tag each row with the FiLM-variant it lands in and the behaviour category it inherits.

---

## Section 3 — The integration mapping (load-bearing)

This is the section the paper-shape memo will lean on. For each load-bearing neuromodulatory algorithm in §2, we name the **FiLM-variant from §1** it corresponds to, give the **explicit mathematical bridge**, name the **target-behaviour benefit** preserved across the bridge, and tag the row's fidelity as **CLEAN MAP**, **EXTENSION MAP**, **PARTIAL MAP**, or **DOES NOT MAP**. The eighteen rows are organised cellular-mechanism-first (forward-pass activation → weights → gradient / recurrent dynamics), not paper-first, so the operational distinctions stay legible.

### Row 1 — Vecoven 2020 ↔ Vanilla FiLM (Perez 2018)
**Tag**: **CLEAN MAP**. **Behaviour**: CS, EE.

**Mathematical bridge.** Vecoven 2020 defines

$$
\sigma_{NMN}(x, z; w_s, w_b) = \sigma\bigl(z^\top(x \, w_s + w_b)\bigr),
$$

with a scalar / low-D latent $z$ from a side network and per-feature weights $w_s, w_b$. Expanding,

$$
\sigma_{NMN}(x, z) = \sigma\bigl((z^\top w_s) \cdot x + (z^\top w_b)\bigr)
\;\equiv\; \sigma\bigl(\gamma(z) \cdot x + \beta(z)\bigr),
$$

with $\gamma(z) = z^\top w_s$ and $\beta(z) = z^\top w_b$. **This is vanilla FiLM with the FiLM-generator being a single linear read-out of $z$ into the per-feature $(\gamma, \beta)$ space**.

**Preserved benefit.** The regime-conditioned scalar $z$ produces regime-specific $(\gamma, \beta)$ shifts that re-purpose a shared feature stack across contexts — i.e., the same architectural mechanism that Perez 2018 uses to re-purpose a CNN across CLEVR question types. Vecoven Fig. 7's "$z$-recruitment" diagnostic — how many effective scalars the network uses — is the direct analogue of an effective-rank-of-$\gamma$ probe in a FiLM-conditioned policy.

**Why CLEAN MAP.** No extension to FiLM is needed; Vecoven 2020 *is* vanilla FiLM with a particular generator architecture. This is the architectural template the project's NMN-on-PPO inherits, and the closest published precedent at the activation level.

### Row 2 — Doya 2002 framework ↔ Vanilla FiLM at policy temperature / encoder
**Tag**: **EXTENSION MAP**. **Behaviour**: EE (NA-β), CS (ACh-α), PG (DA-δ, indirect).

**Mathematical bridge.** Doya 2002's mapping is *hyperparameter-level*, not *operator-level*. The bridge is not a re-write but a *substrate proposal*: each scalar modulator in Doya's mapping is read out by a per-site FiLM generator to produce per-feature $(\gamma, \beta)$ at the network site that implements the corresponding RL operation. Concretely, at a policy-temperature site (where actor logits $\ell$ become probabilities via $\pi(a) \propto \exp(\ell_a)$),

$$
\ell'_a = \gamma_C(\text{mod\_h}) \cdot \ell_a + \beta_C(\text{mod\_h})
\;\;\Rightarrow\;\;
\pi'(a) = \mathrm{softmax}\bigl(\gamma_C \, \ell_a + \beta_C\bigr).
$$

Comparing to a temperature-scaled softmax $\pi(a) \propto \exp(\ell_a / T)$, multiplicative $\gamma_C$ acts as an inverse temperature $1/T$ — the Doya-NA-β mapping in plain FiLM.

**Preserved benefit.** The four-knob mapping survives intact: scaling actor logits ↔ NA inverse-temperature (EE); scaling encoder pre-fusion activations ↔ ACh-as-effective-per-feature-learning-rate (CS); scaling memory-update gate ↔ effective discount on past memory (PG, indirect). All four hyperparameters become *forward-pass operations* on activations.

**Why EXTENSION MAP, not CLEAN.** Doya 2002 commits to scalar global neuromodulators; vanilla FiLM commits to per-feature $(\gamma, \beta) \in \mathbb{R}^d$. The bridge requires identifying the FiLM generator's read-out matrix with the *receptor-density-analog fan-out* from a one-nucleus scalar to a population-level $d$-dim vector (see [`v3 §3 "Scalar-to-population bridge"`](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)). This is the same FiLM operator, but the conditioner is *intra-system-generated* and one-dim per neuromodulator system, where Doya's mapping is scalar-per-nucleus.

### Row 3 — Lee 2024 (Doya–DaYu hand-coded) ↔ Vanilla FiLM with a *closed-form* generator
**Tag**: **EXTENSION MAP**. **Behaviour**: CS, EE.

**Mathematical bridge.** Lee 2024 defines hand-coded functional forms:

$$
\alpha(s, a) = \frac{E(s, a)}{E(s, a) + A(s, a)},
\qquad
\beta^{-1}(s) = \frac{1}{\langle E(s, \hat a)\rangle_{\hat a}},
$$

where $E$ is aleatoric (expected) and $A$ is epistemic (unexpected) uncertainty from a value ensemble. Fitting into the FiLM frame: $\alpha(s, a)$ *is* the modulation magnitude at a site where the gradient pre-multiplier acts on per-feature value updates; $\beta^{-1}(s)$ *is* the multiplicative $\gamma_C$ at the policy-temperature site. The "FiLM-variant" most closely matched is **vanilla FiLM with the FiLM generator replaced by a closed-form function of ensemble uncertainties** rather than a learned network. **AdaIN (§1.3) is the cleanest formal precedent**: AdaIN replaces the learned $(\gamma, \beta)$ lookup with $(\sigma^{\mathrm{IN}}(y), \mu^{\mathrm{IN}}(y))$ computed deterministically from the style image's own deep features. Lee 2024 is the same move: $(\gamma, \beta)$ as closed-form functions of an ensemble's deep features.

**Preserved benefit.** Recovery in non-stationary multi-armed bandits — Lee's Fig. 3 confirms the hand-coded uncertainty-driven $(\alpha, \beta)$ beat fixed-hyperparameter baselines at every context switch. The behaviour-level benefit is preserved; the project's extension is to *learn* the FiLM generator end-to-end rather than hand-code it.

**Why EXTENSION MAP.** Lee 2024 is a CLEAN MAP onto AdaIN-style FiLM if you accept "hand-coded closed-form" as a valid FiLM-generator class. The map is honest if and only if the paper-shape memo states that one of two things changes from Lee 2024: either the project replaces the closed-form generator with a learned one (making it vanilla FiLM with a learned generator — the project's actual proposal), or the project absorbs Lee's hand-coded forms as a baseline comparator. v3 §5.7 "future work" already flags rl-bayesian-dl P-5 ("Lee Component IV regression-recovery") as the test of whether the project's end-to-end-learned generator reorganises to approximate Lee's hand-coded forms.

### Row 4 — Ben-Iwhiwhu 2022 (activity-gating) ↔ Vanilla FiLM with $\beta = 0$
**Tag**: **CLEAN MAP**. **Behaviour**: CS, EE.

**Mathematical bridge.** Ben-Iwhiwhu 2022 applies an activity-gating modulator as $h \to g(z) \odot h$ before the nonlinearity, with $z$ a context vector from CAVIA / PEARL. This is **vanilla FiLM with the additive arm clamped to zero**:

$$
\mathrm{FiLM}_{\beta = 0}(h \mid g(z)) = g(z) \odot h + \mathbf{0} = \gamma \odot h.
$$

**Preserved benefit.** Richer latent representations across tasks and faster meta-test adaptation — same as Ben-Iwhiwhu's experimental result.

**Why CLEAN MAP.** No extension needed; Ben-Iwhiwhu is FiLM restricted to its multiplicative arm. This makes the link explicit: any FiLM-on-PPO architecture that ignores or freezes its additive arm degenerates *exactly* into Ben-Iwhiwhu's activity-gating modulator. The implication for the project's headline falsifier ([v3 §5.1 Outcome U](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)) — γ-clamp removing the inverse-temperature work — has a precedent here: clamping $\gamma = 1$ instead of $\beta = 0$ disables the Ben-Iwhiwhu mechanism entirely.

### Row 5 — Costacurta 2024 (NM-RNN low-rank) ↔ Hypernetwork with structured low-rank output
**Tag**: **EXTENSION MAP**. **Behaviour**: CS, PG.

**Mathematical bridge.** Costacurta 2024 defines

$$
W_x(z) = \sum_{k=1}^K s_k(z) \, \ell_k \, r_k^\top,
\qquad
s_k(z) = \sigma(A_z z + b_z)_k,
$$

i.e., the input-to-recurrent weight matrix is a sum of $K$ rank-1 outer products $\ell_k r_k^\top$ whose *scalar coefficients* $s_k(z)$ are generated from a low-D modulator $z$. This is a **hypernetwork (§1.7) restricted to producing only the $K$ scalar coefficients of a fixed low-rank basis** rather than the full weight matrix.

Equivalently, Costacurta's machine is a *generalised FiLM*: $\gamma_k(z) = s_k(z)$ acts on the rank-1 outer product $\ell_k r_k^\top$ instead of on a per-channel scalar slot. **The bridge**: FiLM is the special case $K = D$ with each $\ell_k r_k^\top$ being the rank-1 matrix that scales channel $k$ alone (i.e., $\ell_k = e_k$ a basis vector, $r_k = $ row-of-conv-kernel-for-channel-$k$). Costacurta with $K < D$ is *more parameter-efficient* than FiLM but *less expressive* per modulator dimension; Costacurta's Prop. 1 shows the LSTM forget-gate falls out as the special case where one of the $s_k$ is an LSTM gate.

**Preserved benefit.** Dissociable-by-ablation multi-channel modulation (Fig. 3F) — different latent dimensions of $z$ are dedicated to different sub-computations, lesioning each produces a dissociable behavioural deficit. **This is the published precedent for the project's per-site-clamp ([v3 §5.2 joint prediction](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md), target-specificity test).**

**Why EXTENSION MAP.** Costacurta is a hypernet that has been *restricted to a low-rank parametrisation* rather than the full kernel of §1.7. The same restriction is what makes it a "FiLM-like" rather than "vanilla FiLM" — the rank-1 outer products generalise the per-channel scalar slot.

### Row 6 — Tsuda 2021 (multiplicative-on-weights) ↔ HyperNetwork (scalar-gain restriction)
**Tag**: **EXTENSION MAP**. **Behaviour**: CS.

**Mathematical bridge.** Tsuda 2021's mechanism is

$$
W \to f_{nm} \cdot W,
$$

a uniform scalar multiplier $f_{nm} \in \mathbb{R}$ applied to a fixed weight matrix. This is the **HyperNetwork (§1.7) restricted to producing only a single scalar that uniformly rescales the kernel** — i.e., the *most degenerate* hypernet. Note this is **NOT vanilla FiLM** on activations, contrary to v1's mis-citation in the [direction memo](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md): the operator-position is at the weight matrix, not at the post-norm activation.

The mathematically clean re-statement under Abdollahzadeh 2021's reinterpretation lemma ($M^l \to J + M^l$ in §1.13): Tsuda is

$$
\hat W = W \odot (J + (f_{nm} - 1) J) = f_{nm} \cdot W,
$$

i.e., **Kernel Modulation (§1.13) restricted to the all-equal modulation matrix** — every weight in the kernel is scaled by the same scalar. KML is to Tsuda what vanilla FiLM is to Doya 2002: same operator-position (weights, in both Tsuda and KML), more degrees of freedom in KML.

**Preserved benefit.** Distinct activity-space "hypertubes" (Tsuda Fig. 3) — context-conditioned weight-level rescaling produces dissociable PCA-space trajectory shifts. **The bridge to FiLM is at the level of the geometric signature, not the mechanism**: the project's FiLM-on-activations would not reproduce Tsuda's hypertubes through the same machinery, but *the activity-space geometry produced by FiLM with high-effective-rank $\gamma$ at site C could resemble Tsuda's hypertubes* — exactly the falsification test in [v3 §5.1 Outcome T](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md).

**Why EXTENSION MAP.** Tsuda IS a hypernetwork (CLEAN MAP onto §1.7 if you accept "scalar gain on weights" as a degenerate hypernet), but its mechanism is at a *different operator-position* (weights, not activations) from the project's substrate. The map is honest only if the memo flags that Tsuda's mechanism is *not* directly instantiated by a forward-pass FiLM on activations; the effects can resemble each other (hypertubes-as-geometry), but the bridge passes through the hypernet variant, not vanilla FiLM.

### Row 7 — Abdollahzadeh 2021 KML ≡ Kernel Modulation
**Tag**: **CLEAN MAP**. **Behaviour**: (within-paper) generalization across multimodal tasks; PG-flavoured.

**Mathematical bridge.** Trivially CLEAN: KML is §1.13 verbatim. The interesting move is the reverse direction — KML's reinterpretation lemma shows **FiLM-on-conv-activations is equivalent to uniform scalar rescaling of a conv kernel per output channel**:

$$
\text{FiLM}(\text{Conv}(x)_c) = \gamma_c \cdot \text{Conv}(x)_c + \beta_c
\;\equiv\;
\text{Conv}(x; W'_c = \gamma_c W_c)_c + \beta_c,
$$

where $W_c$ is the $c$-th output-channel slice of the kernel and $W'_c$ is its uniformly rescaled version. **So vanilla FiLM is itself a degenerate kernel-modulation**: one degree of freedom per output channel per modulation. KML lifts this restriction to per-weight modulation.

**Preserved benefit.** Higher capacity for heterogeneous task distributions — KML beats FiLM-in-MMAML by ~5pp on highly multimodal benchmarks.

**Why CLEAN MAP.** No extension needed; KML is a strict superset of FiLM in the same family. **The project-relevance**: if the project's FiLM-modulator's effective rank saturates (the [v3 §5.4.1 effective-rank-of-$\gamma$ probe](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) signals saturation), KML is the named architectural upgrade. The bridge is mathematical, not heuristic.

### Row 8 — Rodriguez-Garcia 2026 (NGM-SGD) ↔ ... no FiLM-variant in this corpus (gradient-level)
**Tag**: **DOES NOT MAP** (under the FiLM family as defined in §1). **Behaviour**: PG.

**Mathematical bridge.** Rodriguez-Garcia 2026's mechanism

$$
W_{ij}(t) = g_i(t) \, w_{ij}(t),
\qquad g(t+1) = \gamma g(t) + (1 - \gamma) g_0 + \eta H(y),
$$

operates on the *gradient flow* during training — the effective curvature becomes $\lambda \to \lambda / g^2$. The FiLM corpus does not contain any gradient-level operator. The closest formal cousin is **Hypernetwork (§1.7) applied to the optimiser's preconditioning matrix** rather than to the primary network's weights — but the corpus has no such instantiation.

**Preserved benefit.** Stability-gap attenuation under continual training (Hess-protocol). The behaviour-level result is real and well-confirmed across Split MNIST / CIFAR / mini-ImageNet; *the project's R2 win is plausibly an attenuated stability gap of the same flavour* (see [v3 §3 "Differentiation from Rodriguez-Garcia 2026"](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)).

**Why DOES NOT MAP.** Rodriguez-Garcia's gain is on the *gradient*; FiLM and all its corpus relatives are on the *forward pass*. The two are orthogonal (this is the v3 §3 convergent professor-flagged differentiation). The integration paper's honest position is: **the FiLM substrate covers forward-pass neuromodulation, not gradient-level neuromodulation; Rodriguez-Garcia 2026 is the canonical *complement* the paper should name and not absorb**. The two operators could in principle combine (forward-pass FiLM + gradient-level NGM-SGD); none of the corpus has done this.

**Diagnostic value.** The no-map row is part of the paper-shape claim: it draws the substrate boundary. The paper-shape memo must state "of the corpus, only Rodriguez-Garcia 2026 acts at the gradient level, and the FiLM substrate does not absorb it — the architectural footprint of the unification claim is forward-pass only".

### Row 9 — Xing 2022 (RL under env changes) ↔ Vanilla FiLM with context-detector conditioner
**Tag**: **CLEAN MAP**. **Behaviour**: CS, PG.

**Mathematical bridge.** Xing 2022's machine is a context-conditioned modulator that is triggered when an environmental-change indicator fires. Operationally, it is **vanilla FiLM with the conditioner being a learned change-point detector's output**:

$$
z(t) = \text{ChangeDetector}(s_{t-k:t}),
\qquad
\gamma(t) = f_\gamma(z(t)),\;\; \beta(t) = f_\beta(z(t)).
$$

The change detector replaces a question (Perez 2018) or a task ID (Takeda 2021) or a sentence embedding (BC-Z) as the conditioner.

**Preserved benefit.** Faster post-switch recovery and less catastrophic forgetting — the same continual-learning benefit the project's R2 result instantiates.

**Why CLEAN MAP.** No extension needed; the conditioner identity is task-specific but the operator is vanilla FiLM. The relevance for the integration memo is that *this is the closest published precedent for an RL-domain neuromod algorithm that maps cleanly to FiLM with a learned conditioner*.

### Row 10 — Wang 2024 (NeuroNML / NeuronML) ↔ HyperNetwork-adjacent (flexible-network-structure mask)
**Tag**: **EXTENSION MAP**. **Behaviour**: PG.

**Mathematical bridge.** Wang 2024's "flexible network structure" is a per-task binary or low-precision mask on the network's weights, learned via bi-level optimisation:

$$
W_{\text{eff}, T} = W \odot M_T,
\qquad
M_T = \text{StructureGenerator}(\text{task descriptor of } T).
$$

This is **Hypernetwork (§1.7) restricted to producing structure masks rather than continuous weights** — the structurally-discrete cousin of full hypernet that ends up close to **Mixture-of-Experts (§1.11) with $k$ exactly equal to the active sub-mask cardinality**.

**Preserved benefit.** Fast meta-learning adaptation with task-specific structure; comparable to Wang's reported PEARL / CAVIA-beating results.

**Why EXTENSION MAP.** The map is not vanilla FiLM but its discrete cousin in the hypernet / MoE family. The paper-shape memo can name Wang 2024 as evidence that the *structure-mask* axis is a legitimate FiLM-family alternative to per-channel scalar modulation — and that for highly multimodal task distributions (where Abdollahzadeh 2021 flagged FiLM-saturation), structure-masks are an architectural ally.

### Row 11 — Wainstein 2025 (trained RNN gain ↔ pupil-locked switches) ↔ ... no clean FiLM-variant fit
**Tag**: **DOES NOT MAP** (under §1, as the *internal* gain knob), but **PARTIAL** as a pupil-time-lock empirical readout. **Behaviour**: CS, EE.

**Mathematical bridge.** Wainstein 2025's machine is a *trained* RNN with a *post-hoc* gain parameter $g$ applied to the activation function during analysis — the gain is varied to study how perceptual-switch latency moves. The internal knob is not a side-signal-conditioned operator; it is a per-experiment scalar manipulator.

Trying to fit this into the FiLM frame: $g$ multiplying the activation function is closest to **vanilla FiLM with $\gamma = g \mathbf{1}$ and $\beta = 0$ where $g$ is a global scalar applied uniformly across all features** — i.e., the most degenerate case of FiLM ($\gamma$ is a uniform scalar, not a per-channel vector). **But Wainstein's $g$ is not produced by a learned generator from a side signal**; it is an experimental dial. The "side signal" in Wainstein is *pupil diameter*, which correlates with $g$ but is not the *generator* of $g$ — the RNN was trained without pupil input.

**Why DOES NOT MAP.** Wainstein 2025 is the corpus's strongest empirical anchor for the cellular-gain ↔ behavioural-switch link, but its mechanism is *not a side-signal-conditioned forward-pass affine* — it is a *post-hoc analytical knob*. The integration memo should treat Wainstein as the *empirical evidence base* for the behaviour-level claim (gain modulation produces switch-latency changes), but should NOT claim it is a FiLM-variant.

**Diagnostic value.** This no-map row sharpens the integration: the FiLM substrate's "side signal → forward-pass affine" template is *operationally distinct* from Wainstein's *trained-RNN-gain-as-internal-state* template. The two intersect only when the side signal *is* an internal state of the network — which is exactly Birnbaum 2019's Temporal FiLM (§1.4, conditioner is self-pooled activations), or Yan & Guo 2025's CASA (conditioner is the batch's own feature mean). Wainstein's $g$ is closer to an *attention temperature* or a *gain-on-the-loss-Hessian* — neither covered by the FiLM corpus.

### Row 12 — Osman 2024 (Hopfield + gain) ↔ ... no clean FiLM-variant fit
**Tag**: **DOES NOT MAP** (under §1). **Behaviour**: EE (annealing-style).

**Mathematical bridge.** Osman 2024's mechanism is $W_{\mathrm{rec}} \to g \cdot W_{\mathrm{rec}}$ on a recurrent Hopfield network — low $g$ flattens the attractor energy landscape. The closest cousin is **Hypernetwork (§1.7) producing a single scalar that uniformly rescales the recurrent matrix** — i.e., same degenerate-scalar-hypernet as Tsuda 2021 (Row 6). But the *behaviour-level effect* in Osman is via Hopfield-net energy-landscape annealing, and the FiLM corpus has no analogue of energy-landscape modulation through a forward-pass operator.

**Why DOES NOT MAP.** The mechanism is a hypernet restriction (mappable to §1.7, like Tsuda); the behaviour-level effect (energy-landscape annealing à la statistical-physics Bayesian inference) is *not* what the FiLM corpus claims its operator produces. The FiLM corpus claims the operator produces feature-statistic shifts, conditional behaviour adaptation, or ensemble diversity — not attractor-landscape annealing. The integration memo should name Osman 2024 as a *complementary architectural family* (recurrent-attractor-net + gain) rather than as a FiLM-variant.

### Row 13 — AlKilany & Goodman 2025 (per-neuron excitability multiplier) ↔ Vanilla FiLM with $\beta = 0$
**Tag**: **CLEAN MAP** (with the caveat that the substrate is spiking-network, not standard DL). **Behaviour**: EE, CS.

**Mathematical bridge.** AlKilany & Goodman 2025 apply a per-neuron excitability multiplier — i.e., a multiplicative gain per spiking unit. In the rate-based / standard-DL surrogate, this is

$$
h_i \to g_i \cdot h_i = \gamma_i \cdot h_i + 0,
$$

i.e., **vanilla FiLM with $\beta = 0$ and a per-neuron $\gamma$** (the same form as Ben-Iwhiwhu 2022 in Row 4).

**Preserved benefit.** "Listening in the dips" — emergent dynamic-gain-control behaviour that decreases reaction time in noisy listening tasks. The behaviour-level benefit is comparable to attentional gain.

**Why CLEAN MAP.** With the substrate-translation caveat (spiking vs. rate-based), AlKilany & Goodman 2025 is FiLM with the additive arm clamped — the exact same form as Ben-Iwhiwhu 2022. The paper-shape memo can name AlKilany as evidence that the *multiplicative-only* FiLM mechanism produces measurable behavioural improvements even without explicit training pressure on the gain itself.

### Row 14 — Tambaş 2025 (Krotov-Hopfield three-factor rule) ↔ Bayesian Hypernet-adjacent / structured weight modulation
**Tag**: **PARTIAL MAP**. **Behaviour**: PG.

**Mathematical bridge.** The Krotov-Hopfield three-factor rule has a global modulator gating local plasticity. The plasticity rule reads, in shorthand, $\Delta w_{ij} \propto m \cdot \text{local}(x_i, x_j, y_i)$ where $m$ is a global modulator and `local` is a local Hebbian / anti-Hebbian rule. The closest FiLM-family cousin is **structured weight modulation under a multiplicative side signal** — i.e., something between Costacurta 2024 (§1.7 restricted to low-rank) and a Bayesian hypernet (§1.8) where the global signal $m$ plays the role of a *prior over the weight-update distribution*.

**Preserved benefit.** The three-factor framing — global signal modulates local learning — is the substrate-level argument for any "neuromodulator-modulates-plasticity" claim, including the lifelong-learning use of the project's modulator.

**Why PARTIAL MAP.** The cellular-mechanism level (a global multiplier on a local update) bridges to the hypernet / structured weight family. But the FiLM corpus has no machine that *gates plasticity globally via a side signal* — the closest is Rodriguez-Garcia 2026 (no-map, gradient-level). The bridge is partial: the *form* (global-multiplier on a local quantity) is FiLM-like, but the *operand* (a plasticity rule) is outside the FiLM corpus's substrate.

### Row 15 — Kudithipudi 2022 (lifelong-learning survey) ↔ taxonomic umbrella, not a single FiLM-variant
**Tag**: **PARTIAL MAP** (framing only). **Behaviour**: PG.

**Mathematical bridge.** Kudithipudi 2022 surveys plasticity-gating biological mechanisms; it does not propose a single mechanism. The bridge to FiLM is at the level of the *space of solutions* — the survey's "neuromodulated plasticity" sub-category is the umbrella that contains Rodriguez-Garcia 2026, Tambaş 2025, Costacurta 2024's continual-learning-flavoured Fig. 3F, and the project's R2-readout.

**Why PARTIAL MAP.** Kudithipudi names the *behavioural domain* (lifelong learning) but provides no single operator. The map onto §1 happens through each child paper of the survey, not through Kudithipudi itself.

### Row 16 — Mei 2022 (review) ↔ Vanilla FiLM (umbrella framing)
**Tag**: **PARTIAL MAP** (framing only). **Behaviour**: framing for all four.

**Mathematical bridge.** Mei 2022 Fig. 1B is the canonical diagram of "neuromodulation → hyperparameter fine-tuning" in a DNN — but it is *programmatic*, not a single mechanism. The framing maps cleanly onto vanilla FiLM as the "neuromodulation unit" that reconfigures DNN hyperparameters; Mei 2022 names the research programme that this synthesis is operationalising.

**Why PARTIAL MAP.** Mei does not run the experiment; the bridge is at the level of the proposal. The paper-shape memo can cite Mei 2022 Fig. 1B as the canonical statement of the unification thesis the project's substrate operationalises.

### Row 17 — Ferguson & Cardin 2020 (gain types) ↔ Vanilla FiLM (multiplicative + additive arms)
**Tag**: **CLEAN MAP** (mechanism inventory anchor). **Behaviour**: EE (multiplicative), HV (additive ↔ rheobase / selectivity).

**Mathematical bridge.** Ferguson & Cardin Box 1 distinguishes:

- **Multiplicative gain** — scaling the slope of $f(I)$. In FiLM, this is the multiplicative arm $\gamma \odot h$. Mathematically: $\gamma > 1$ steepens the I/O curve; $\gamma < 1$ flattens it.
- **Additive bias / rheobase shift** — shifting the input required to fire. In FiLM, this is the additive arm $h + \beta$. Mathematically: $\beta > 0$ pushes the activation closer to firing threshold (analogous to depolarisation); $\beta < 0$ moves it away.
- **Divisive normalisation** — gain scaled by a sum of competing-feature activations. In FiLM, this requires *content-dependent* $\gamma$ — e.g., AdaIN's $\sigma^{\mathrm{IN}}(y)$ where the gain is computed from feature statistics rather than a learnable lookup. Or, equivalently, FiLM with a generator that includes the *target stream's own pooled statistics* as input.

**Preserved benefit.** Sensitivity-vs-selectivity dissociation: $\gamma$ moves sensitivity (the slope), $\beta$ moves selectivity (the threshold). This is the cellular substrate of the [v3 §5.1 γ-clamp vs β-clamp falsifier](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md).

**Why CLEAN MAP.** Vanilla FiLM (§1.1) instantiates both arms simultaneously. The bridge is the most direct in the entire mapping: FiLM is *the artificial substrate for Ferguson & Cardin's cellular distinction*. Divisive normalisation requires a content-dependent generator (AdaIN-style or CASA-style); v3 §5.2 site A flags this as the `eval/gamma_A_divisive_index` probe.

### Row 18 — Shine 2021 (cellular ↔ network gain) ↔ Vanilla FiLM (cellular vocabulary anchor)
**Tag**: **CLEAN MAP** (substrate vocabulary). **Behaviour**: EE, CS.

**Mathematical bridge.** Shine 2021 Box 1 formally defines neural gain as $g = dQ/dI$ — the slope of a neuron's input-output mapping function. In a rate-based / standard-DL neuron $y = \phi(\gamma \cdot z + \beta)$, the slope at operating point $z$ is

$$
\frac{dy}{dz} = \gamma \cdot \phi'(\gamma z + \beta).
$$

For piecewise-linear $\phi$ (ReLU), $\phi' \in \{0, 1\}$, so the slope is exactly $\gamma$ in the active regime — **FiLM's multiplicative arm $\gamma$ literally IS Shine 2021's "neural gain"** at the post-ReLU operating point. For sigmoid / softmax, $\phi'$ is the gain's curve, and $\gamma$ multiplies into it directly.

**Preserved benefit.** Yerkes-Dodson arousal-performance inverted-U; macroscale energy-landscape flattening. Shine 2021 supplies the cellular vocabulary the unification claim depends on (gain = slope of I/O curve), which FiLM's $\gamma$ instantiates verbatim.

**Why CLEAN MAP.** No extension; FiLM's $\gamma$ is the artificial substrate for Shine 2021's $g$. The paper-shape memo should lead the integration argument with Shine 2021's gain-as-slope definition + Ferguson & Cardin 2020's multiplicative-vs-additive distinction + FiLM's $\gamma \odot h + \beta$ — those three citations together establish the cellular-to-FiLM operator equivalence.

### Tag-distribution summary

Of the 18 rows above, counted as primary tag per row:

- **7 CLEAN MAP** — Vecoven 2020 (Row 1), Ben-Iwhiwhu 2022 (Row 4), Abdollahzadeh 2021 KML (Row 7), Xing 2022 (Row 9), AlKilany & Goodman 2025 (Row 13), Ferguson & Cardin 2020 (Row 17), Shine 2021 (Row 18).
- **5 EXTENSION MAP** — Doya 2002 (Row 2), Lee 2024 (Row 3), Costacurta 2024 (Row 5), Tsuda 2021 (Row 6), Wang 2024 (Row 10).
- **3 PARTIAL MAP** — Tambaş 2025 (Row 14), Kudithipudi 2022 (Row 15), Mei 2022 (Row 16).
- **3 DOES NOT MAP** — Rodriguez-Garcia 2026 (Row 8), Wainstein 2025 (Row 11 — no-map mechanism with partial empirical readout caveat), Osman 2024 (Row 12).

Total: 7 + 5 + 3 + 3 = 18.

**The headline tag distribution is: 7 clean / 5 extension / 3 partial / 3 no-map.** Twelve of eighteen (67%) load-bearing neuromod algorithms map cleanly or with extension onto a FiLM-variant; three partial maps add framing without mechanism; three no-maps draw the substrate boundary at gradient-level (Rodriguez-Garcia 2026), recurrent-trained-RNN-internal gain (Wainstein 2025 as mechanism), and recurrent-energy-landscape gain (Osman 2024).

---

## Section 4 — Behaviour-level integration by target category

This section reorganises §3 by the four target behaviour-level benefits, ending each sub-section with the one-line headline claim the paper-shape memo can stake.

### 4.1 Context-switching / regime change

**Which FiLM-variants implement it.** Vanilla FiLM (§1.1, conditioner = task / regime ID), Multi-task feature modulation (§1.6, conditioner = task vector with $\mathbf{0}$ as identity), Temporal FiLM (§1.4, conditioner = self-pooled history, so the modulator detects regime changes from the network's own state).

**Which neuromod algorithms instantiate it.** Vecoven 2020 (Row 1, CLEAN), Doya 2002 / Lee 2024 (Rows 2–3, EXTENSION), Ben-Iwhiwhu 2022 (Row 4, CLEAN), Costacurta 2024 (Row 5, EXTENSION dissociable-channels), Tsuda 2021 (Row 6, EXTENSION via hypertube geometry), Xing 2022 (Row 9, CLEAN), AlKilany & Goodman 2025 (Row 13, CLEAN with substrate caveat), Wainstein 2025 (Row 11, no-map mechanism / partial readout — switch-latency).

**Diagnostic mathematical signatures.** (a) Modulator state burst time-locked to switch event ($|\delta|$-spike) — the [v3 §5.1 / §5.2 site-A](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) measure; (b) effective rank of $\gamma$ at policy site shifting rank-1 at switch and broadening through recovery (Vecoven 2020 Fig. 7 $z$-recruitment analogue); (c) per-modulator-dimension ablation producing dissociable behaviour deficits (Costacurta 2024 Fig. 3F precedent).

**Headline claim.** *Of the FiLM-variants in the corpus, vanilla FiLM with a learned regime-detector generator + a self-recurrent (TFiLM-style) conditioner instantiates the context-switching benefit demonstrated across Vecoven 2020, Lee 2024, Xing 2022, and the project's R2 empirical anchor.*

### 4.2 Hypervigilance / sustained tonic vigilance

**Which FiLM-variants implement it.** Vanilla FiLM (§1.1) with channel-selective $\gamma$ (high $\gamma$ on threat-relevant input channels, low $\gamma$ on threat-irrelevant); Temporal FiLM (§1.4) for the tonic / phasic timescale separation (block-piecewise-constant $(\gamma, \beta)$ extending over many input-level timesteps).

**Which neuromod algorithms instantiate it.** Ferguson & Cardin 2020 (Row 17, additive arm ↔ rheobase shift ↔ selectivity ↔ channel-selective hypervigilance signature); Doya 2002 (Row 2, ACh-α-on-encoder analogue); the project's pain-construct framing ([v3 §5.2 site A and site C, §5.5 four-dissociation gate](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)).

**Diagnostic mathematical signatures.** (a) Per-modality $\gamma_A$ statistics in event-locked window across the 9 input modalities — high on threat-relevant modalities, low or unchanged on threat-irrelevant (`professor-pain-modeling` P-Pain-1 in v3); (b) tonic state's *hysteresis* — modulator output stays elevated after the aversive regime ends (v3 §5.3.3); (c) near-injury vs far-from-injury action-entropy contrast post-recovery (v3 §5.2 site C, Vlaeyen-Linton fear-avoidance signature).

**Headline claim.** *Of the FiLM-variants in the corpus, vanilla FiLM with per-modality channel-selective $\gamma$ at the encoder (Ferguson & Cardin 2020's additive-arm-as-rheobase-shift instantiated as FiLM's additive arm $\beta$ at the encoder) is the architectural substrate that produces hypervigilance-like behaviour distinguishable from volume control; no corpus paper has yet logged this distinction on a forward-pass FiLM-modulated agent.*

### 4.3 Plasticity gating / lifelong-learning resistance

**Which FiLM-variants implement it.** *Forward-pass-only*: vanilla FiLM (§1.1) with conditioner that signals regime change; HyperNetwork (§1.7) with conditioner that produces fresh weights per task — Galanti & Wolf 2020's exponential parameter-efficiency bound is the theoretical anchor for why this works.

*Gradient-level cousin (not in this FiLM corpus)*: Rodriguez-Garcia 2026 NGM-SGD, which the integration memo must flag as the **complement** of forward-pass FiLM, not its instance.

**Which neuromod algorithms instantiate it.** Rodriguez-Garcia 2026 (Row 8, no-map — gradient-level); Costacurta 2024 (Row 5, EXTENSION via low-rank weight modulation); Wang 2024 (Row 10, EXTENSION via structure mask); Tambaş 2025 (Row 14, PARTIAL via three-factor rule); the project's R2 win (forward-pass-only, plausibly an attenuated stability gap).

**Diagnostic mathematical signatures.** (a) Continual-evaluation old-task accuracy curves with per-batch evaluation every $\rho$ iterations (Hess-protocol, Rodriguez-Garcia 2026); (b) per-stage transient drop at task boundary — magnitude is the stability gap, attenuation is the project's claim; (c) Hessian-eigenvalue probe of the loss curvature peri-switch (v3 §5.4.3 — discriminates forward-pass FiLM from gradient-level NGM-SGD).

**Headline claim.** *The FiLM substrate naturally instantiates plasticity gating at the forward-pass level (via state-dependent $(\gamma, \beta)$ that re-weights effective per-feature learning rates without touching the optimiser); the gradient-level instantiation (Rodriguez-Garcia 2026) is the orthogonal complement, not a special case, and the two could in principle combine to produce additive stability-gap attenuation.*

### 4.4 Exploration–exploitation control

**Which FiLM-variants implement it.** Vanilla FiLM (§1.1) at the policy site — scaling actor logits by $\gamma_C$ is mathematically inverse-temperature modulation. FiLM-Ensemble (§1.9) and Bayesian Hypernet (§1.8) supply the *uncertainty estimate* that the inverse-temperature could be conditioned on (Lee 2024 $\beta^{-1}(s) = 1/\langle E(s,\hat a)\rangle$).

**Which neuromod algorithms instantiate it.** Doya 2002 (Row 2, NA-β branch — EXTENSION); Lee 2024 (Row 3, hand-coded $\beta^{-1}$ from ensemble uncertainty — EXTENSION via AdaIN-style closed-form generator); Ferguson & Cardin 2020 (Row 17, multiplicative-arm slope-change ↔ inverse-temperature); Shine 2021 (Row 18, gain = $dQ/dI$ ↔ FiLM's $\gamma$ on softmax logits, slope-of-softmax = inverse-temperature in the sigmoid limit); Ben-Iwhiwhu 2022 / AlKilany & Goodman 2025 (Rows 4, 13 — CLEAN MAP with $\beta = 0$, supplying richer activity-gating).

**Diagnostic mathematical signatures.** (a) Multiplicative $\gamma_C$ on actor logits time-locked to ensemble-based unexpected-uncertainty $A(s, \hat a)$ (the Lee 2024 functional form recovered post-hoc from a learned FiLM generator — v3 P-5 future work); (b) actor-entropy peri-switch spike amplitude tracking $\gamma_C$ amplitude; (c) the [v3 §5.1 γ-clamp vs β-clamp falsifier](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) — $\gamma$-clamp more damaging than $\beta$-clamp at site C is the headline EE-mapping confirmation.

**Headline claim.** *Of the FiLM-variants in the corpus, vanilla FiLM with multiplicative $\gamma$ at the policy site (with conditioner driven by an ensemble-uncertainty signal à la FiLM-Ensemble or Bayesian hypernet) instantiates the Doya-NA-β inverse-temperature mechanism as a learned forward-pass operation, with Lee 2024's hand-coded $\beta^{-1}(s) = 1/\langle E(s,\hat a)\rangle$ as the AdaIN-style closed-form precedent the learned form should be benchmarked against.*

---

## Section 5 — Open gaps and what the project's substrate could uniquely fill

The integration mapping in §3 reveals a set of gaps the FiLM corpus *plus* the neuromod corpus jointly leave open — gaps the project's three-injection-site FiLM + heteroscedastic precision head substrate is uniquely positioned to fill. These are the load-bearing claims the paper-shape memo can stake.

### 5.1 No FiLM paper logs the behaviour-level readouts the neuromod literature predicts

The FiLM corpus measures FiLM's impact on **task accuracy** (Perez 2018, Takeda 2021), **calibration / OOD detection** (Turkoglu 2022, Gorishniy 2025), **robotic task success** (Jang 2022), **anti-exploration bonus minimisation** (Nikulin 2023), and **style-transfer quality** (Dumoulin 2017, Huang & Belongie 2017). It does *not* measure:

- Actor-entropy peri-switch spike (the Doya-NA-β signature, predicted by Doya 2002 §3.3, Lee 2024 §3.4, Mei 2022 Box 1).
- $|\delta|$-tracking by the modulator (the Wainstein 2025 pupil-locked-switch signature, the project's [v3 §5.1 falsifier](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)).
- Per-channel $\gamma_A$ statistics time-locked to interoceptive injury onset (the channel-selective hypervigilance signature, [v3 §5.2 site A](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)).
- Continual-evaluation stability-gap with per-batch evaluation across task boundaries (Rodriguez-Garcia 2026 Hess-protocol, applied to a forward-pass FiLM-modulated agent rather than to a gradient-level NGM-SGD agent).

**The project's substrate fills this gap directly**: the FiLM modulator + PPO + grid-world testbed + per-injection-site logging produce exactly these behavioural readouts on a single forward-pass FiLM-modulated agent. None of the 23 FiLM papers in the corpus does this; none of the 18 neuromod papers in the corpus does it on a FiLM substrate.

### 5.2 No neuromod paper uses a FiLM-ensemble or Bayesian hypernet for *uncertainty over the modulator output*

Lee 2024's hand-coded $\beta^{-1}(s) = 1/\langle E(s, \hat a)\rangle$ requires an ensemble to compute $E$. Lee uses a deep ensemble of Q-networks. The FiLM corpus contains two cheap implicit ensembles — FiLM-Ensemble (§1.9, 1.3% parameter overhead) and BatchEnsemble / TabM (§1.10, similar order). **No paper in the neuromod corpus has used a FiLM-style implicit ensemble to compute the uncertainty that drives the modulator's output**.

The two compose naturally: a per-modulator-output member-index $m$ at the modulator's read-out head (FiLM-Ensemble), producing $M$ different $(\gamma^m, \beta^m)$ vectors per timestep; the spread across $M$ is the epistemic uncertainty over the modulator's recommendation, which the value head could then weight. This is the unification of Lee 2024's $E(s, \hat a)$ with Turkoglu 2022's FiLM-Ensemble, and *no paper in either corpus has done it*. The project's three-injection-site substrate is the natural site to attempt this combination (rl-bayesian-dl P-5 in v3 §5.7).

### 5.3 The "FiLM-variant family" is implicit but never explicitly enumerated as a unifying taxonomy

The FiLM corpus's papers cite each other in clusters — CIN / CBN / AdaIN → FiLM (Perez 2018 §3); hypernet (Ha 2016) cited by FiLM-Ensemble (Turkoglu 2022 §4); KML reinterprets FiLM (Abdollahzadeh 2021 §3); CASA (Yan & Guo 2025) cites FiLM. *But no paper in the FiLM corpus enumerates the variants as a single taxonomy with explicit operator-level relations*. Galanti & Wolf 2020 §2 names "embedding method vs hypernetwork", which is a binary distinction; it does not catalogue the seven-to-ten variants of §1 above.

**This is the paper-shape standalone gap.** The taxonomy of §1, applied to neuromodulator-style mechanisms, *is* the load-bearing contribution. The integration memo's §3 mapping is the first published explicit catalogue of how the FiLM-variant family relates to the neuromodulator-style mechanism family — operator-level, with mathematical bridges, behaviour-tagged, and honesty-flagged for no-fits.

### 5.4 No forward-pass FiLM substrate has been benchmarked against Rodriguez-Garcia 2026 NGM-SGD as the orthogonal complement

Rodriguez-Garcia 2026's gradient-level gain is the corpus's clearest *alternative* to forward-pass FiLM as a stability-gap attenuator. The two operate at orthogonal points in the optimisation loop (forward pass vs gradient update). **No paper has run them head-to-head, and no paper has run them in conjunction.** v3 §5.4.3 names the Hessian-eigenvalue probe (`jax.hessian` or HVP via `jax.jvp(jax.grad(L))`) as the discriminating measurement — peri-switch curvature flattening tells you which level the gain mechanism is acting at. The integration memo can stake the headline claim that the *empirical disambiguation* of forward-pass FiLM vs gradient-level NGM-SGD is the project's substrate-level contribution, with the Hessian probe as the proposed measurement.

### 5.5 The bidirectional cellular-gain ↔ behavioural-hyperparameter unification test (gap 1 from the predictions synthesis)

[`neuromodulatory_algorithms_predictions_synthesis.md` §3.6 and §5.1](../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md) flagged this gap independently. The FiLM corpus does not close it (no FiLM paper logs cellular-gain-style and behavioural-hyperparameter-style readouts on the same agent on the same trials), and the neuromod corpus does not close it either (Lee 2024 logs the hyperparameter but uses a hand-coded form; Rodriguez-Garcia 2026 logs the loss-curvature but lacks an RL hyperparameter; Wainstein 2025 logs gain and pupil but the behaviour is perceptual switch latency). **The project's FiLM substrate + per-site $\gamma, \beta$ logging + actor-entropy / $|\delta|$-tracking is uniquely positioned to run this test** — v3 §5.1 names the four-arm γ-clamp / β-clamp falsifier as the operational form.

### 5.6 Forward-pass γ-as-precision-encoding under heteroscedastic loss is unexplored

Kendall & Gal 2017 ([review](reviews/kendall_gal_2017_uncertainties.md)) anchor the heteroscedastic loss

$$
\mathcal{L} = \frac{1}{2} \exp(-s) \|y - \hat y\|^2 + \frac{1}{2} s,\qquad s = \log \hat\sigma^2,
$$

as the project's precision-weighted regression loss. The corpus does not contain any paper that *combines this loss with a FiLM-modulated regression head*. The combination is the project's [Phase 3 precision head](../../../docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) — and it is the substrate the predictive-coding precision reading ([v3 §3 level-specific precision reading](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md), `professor-bayesian-brain` P3) operationalises. **No paper in either corpus has FiLM-modulated $(\gamma, \beta)$ together with Kendall-Gal heteroscedastic precision $\sigma^2(x)$ on the same regression head.** The combination produces a *learnable, end-to-end-trained precision-weighted gain* — the substrate-level integration of FiLM's conditional modulation with Bayesian-deep-learning aleatoric uncertainty.

### 5.7 Phasic-vs-tonic dissociation on a forward-pass FiLM machine

[v3 §5.3 and §5.5](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) and [predictions-synthesis §5.5 Gap 5](../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md) both name this gap. No paper in the FiLM corpus has *two recurrent timescales with separable phasic and tonic readouts at the conditioner*; Birnbaum 2019's TFiLM has one LSTM (one timescale). Rodriguez-Garcia 2026 has one time-constant. Vecoven 2020 has one time-constant. **The project's Phase 0 T/P split is the substrate-level architectural change that fills this gap**, and v3 §5.3 names the autocorrelation-time-constant probes ($\tau_P, \tau_T$) as the diagnostic measurements.

---

## Section 6 — Reading guide

A one-line gloss per cited paper, telling the postdoc and math-reviewer exactly which section / equation / figure to cite. Service section.

### FiLM corpus

- **Perez 2018** ([review](reviews/perez_2018_film.md)) — *§2.1 Eq. for FiLM ($y = \gamma F + \beta$); §4.3 ablations showing $\gamma$ carries 65% of conditioning weight vs $\beta$'s 1%; §4.2 (γ, β) algebra for zero-shot composition.* Cite as the canonical FiLM equation and the empirical anchor that $\gamma$ is the load-bearing arm.
- **Ha 2016** ([review](reviews/ha_2016_hypernetworks.md)) — *§3.1 static hypernet kernel-generation; §3.2 HyperRNN row-scaling trick (structurally FiLM at per-row); Fig. visualisations of weight changes at word boundaries.* Cite as FiLM's super-class; the HyperRNN row-scaling trick is the closest precedent to dynamic-FiLM.
- **Dumoulin 2017 CIN** ([review](reviews/dumoulin_2017_cond_instance_norm.md)) — *§2.1 Eq. for CIN; §3.4 linear interpolation in (γ, β) space.* Cite as the direct FiLM ancestor and the demonstration that (γ, β) carries semantic content.
- **Huang & Belongie 2017 AdaIN** ([review](reviews/huang_belongie_2017_adain.md)) — *§5 Eq. for AdaIN with (γ, β) = (σ, μ) of style image's deep features.* Cite as the closed-form-FiLM-generator precedent for Lee 2024's hand-coded mapping.
- **Birnbaum 2019 TFiLM** ([review](reviews/birnbaum_2019_temporal_film.md)) — *§3 Algorithm 1 for block-piecewise-constant (γ, β); §5.4 (γ, β) clustering by speaker gender.* Cite as the time-varying-FiLM precedent and the self-recurrent conditioner template.
- **Wisnu 2025 STSM-FiLM** ([review](reviews/wisnu_2025_stsm_film.md)) — *§3 Eq. for the (1 + γ) identity-init trick.* Cite as the identity-init precedent if the project wants the FiLM modulator to start at the identity transformation.
- **Takeda 2021 multi-task FM** ([review](reviews/takeda_2021_multi_task_feature_mod.md)) — *§3 Method 3 — synthesised mixed-task ground-truth as the only recipe that produces smooth mixture-coordinate interpolation.* Cite as the warning that the conditional vector behaves as a mixture only if training spans the corners.
- **Turkoglu 2022 FiLM-Ensemble** ([review](reviews/turkoglu_2022_film_ensemble.md)) — *§2 FiLM-Ensemble Eq.; §3.1 diversity-vs-explicit-ensemble; Table 1 parameter overhead.* Cite as the cheap epistemic-uncertainty channel that composes with FiLM-modulated regression.
- **Gorishniy 2025 TabM** ([review](reviews/gorishniy_2025_tabm.md)) — *§3.2 BatchEnsemble Eq. ($W_i = W \odot s_i r_i^\top$); §3.3 incremental construction.* Cite as the FiLM-adjacent rank-1 weight-modulation form for substrates without normalisation layers.
- **Krueger 2017 Bayesian hypernets** ([review](reviews/krueger_2017_bayesian_hypernets.md)) — *§3.2 Bayesian hypernet via normalising flow; §3.3 weight-norm-only efficient parametrisation.* Cite as the probabilistic generalisation of HyperNetwork and the precedent for a Bayesian-FiLM in the project's future work.
- **Galanti & Wolf 2020 hypernet modularity** ([review](reviews/galanti_wolf_2020_hypernet_modularity.md)) — *Thm. 4 modularity bound; Thm. 5 total parameter complexity.* Cite as the theoretical justification for the FiLM/hypernet family's parameter efficiency over embedding-concatenation.
- **Shazeer 2017 sparse MoE** ([review](reviews/shazeer_2017_sparse_moe.md)) — *§2 noisy top-$k$ gating; §4 importance / load auxiliary losses.* Cite as the discrete-sparse cousin of FiLM in the hypernet family.
- **Vaswani 2017 attention** ([review](reviews/vaswani_2017_attention.md)) — *§3.2 scaled dot-product Eq.; §3.2.2 multi-head.* Cite as the alternative conditioning primitive that modulates positions where FiLM modulates channels.
- **Kendall & Gal 2017 uncertainties** ([review](reviews/kendall_gal_2017_uncertainties.md)) — *§3.1 Eq. 6 the heteroscedastic loss $\mathcal{L} = \tfrac{1}{2} \exp(-s)\|y - \hat y\|^2 + \tfrac{1}{2} s$.* Cite as the project's anchor uncertainty formula and the precision-weighting precedent that combines with FiLM-modulated regression heads (gap 5.6).
- **Gawlikowski 2023 uncertainty survey** ([review](reviews/gawlikowski_2023_uncertainty_survey.md)) — *§3 4-branch taxonomy; §3.3.4 law-of-total-variance ensemble decomposition.* Cite as the canonical map of aleatoric / epistemic split for the integration with FiLM-Ensemble.
- **Santurkar 2018 BN optimisation** ([review](reviews/santurkar_2018_batchnorm_optimization.md)) — *§3 Thm. 4.1 BN loss-Lipschitz bound.* Cite as the formal justification for why FiLM-on-normalisation trains well (loss-landscape smoothing).
- **Nikulin 2023 SAC-RND** ([review](reviews/nikulin_2023_anti_exploration_rnd.md)) — *§4 FiLM-prior gradient-smoothing; §6.3 toy environment anti-gradient field.* Cite as the discovery that FiLM has a *gradient-landscape-shaping* property — relevant if the project's PPO inner loop shows curvature-flattening signatures from FiLM at the value head.
- **Jang 2022 BC-Z** ([review](reviews/jang_2022_bcz.md)) — *§5.3 FiLM at every ResNet block; §2.3 (γ, β) computation.* Cite as the canonical FiLM-on-policy template (closest published architecture to the project's substrate).
- **Moon 2023 hierarchical achievements** ([review](reviews/moon_2023_hierarchical_achievements.md)) — *§4.1 FiLM as action-on-state fuser in the contrastive head.* Cite as the small-role FiLM-as-fusion-gadget precedent.
- **Yan & Guo 2025 CASA** ([review](reviews/yan_guo_2025_context_aware_dg.md)) — *§3.2.2 CaFiLM 6-parameter shared 2×2 module on (z_c, μ_c).* Cite as the precedent for using the batch's own feature mean as a domain-context FiLM conditioner — substrate-level analogue of a self-recurrent modulator.
- **Abdollahzadeh 2021 KML** ([review](reviews/abdollahzadeh_2021_multimodal_meta.md)) — *§3 reinterpretation lemma (FiLM ≡ uniform scalar rescaling of conv kernel); KML's outer-product factorisation.* Cite as the named upgrade path if FiLM's $\gamma$ saturates the project's effective-rank probe.
- **Andreas 2016 / Hu 2017 NMN family** ([Andreas review](reviews/andreas_2016_neural_module_networks.md), [Hu review](reviews/hu_2017_e2e_module_networks.md)) — *§3-4 layout assembly; REINFORCE.* Cite as the symbolic-discrete fork of the conditional-modulation family (we do NOT claim NMN is FiLM in disguise).

### Neuromod corpus (canonical extraction in the prior predictions synthesis)

- **Doya 2002** — *§3 pp. 498-499 — four-knob mapping; §3.3 NA-β; §3.4 ACh-α.* Cite for the foundational hyperparameter mapping (Row 2).
- **Ferguson & Cardin 2020** — *Box 1 panels b (multiplicative slope), c (additive rheobase).* Cite for the cellular substrate of the [v3 §5.1 γ/β falsifier](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) (Row 17).
- **Shine 2021** — *Box 1 + Fig. 1 — neural gain as $dQ/dI$.* Cite for the formal definition of "neural gain = slope of I/O curve" that FiLM's $\gamma$ instantiates (Row 18).
- **Lee 2024** — *§3.1 Fig. 2 framework; Eq. 4 hand-coded mapping; §3.2 5-HT caveat.* Cite for the closest published precedent to the project's claim (Row 3).
- **Vecoven 2020** — *§2 Fig. 1 — $\sigma_{NMN}$ Eq.; Fig. 7 $z$-recruitment.* Cite for the architectural substrate (Row 1).
- **Tsuda 2021** — *Fig. 1b §Results — multiplicative on weights; Extended Data Appendix A explicit distinction from activation-level.* Cite as the multiplicative-weight-scaling story (Row 6).
- **Costacurta 2024** — *Eq. 4 §3.1 low-rank scaling; Prop. 1 LSTM-equivalence; Fig. 3F dissociation-by-ablation.* Cite for the structured-flexibility intermediate (Row 5).
- **Rodriguez-Garcia 2026** — *§2.1 Eq. 2 fast-slow weight decomposition; Algorithm 1; Fig. 2 loss-landscape flattening.* Cite for the optimiser-level gradient-gain unification — the no-map row (Row 8) that names the substrate boundary.
- **Wainstein 2025** — *Fig. 1 pupil-switch chain; Figs. 4-5 RNN-trajectory velocity / energy-landscape flatness.* Cite as the empirical anchor for cellular-gain ↔ switch-latency (Row 11, no-map mechanism / partial empirical readout).
- **Mei 2022** — *Fig. 1B canonical diagram of neuromodulation → hyperparameter; Box 1 actor-critic + Doya restatement.* Cite for the most direct statement of the unification thesis (Row 16, partial / framing).
- **Ben-Iwhiwhu 2022** — *§4 Fig. 1b activity-gating; §3 Doya recapitulation.* Cite for activity-gating in CAVIA / PEARL (Row 4).
- **Wang 2024 (NeuronML)** — *§I-III FNS motivation; §IV-V bi-level optimisation.* Cite as the hypernetwork-adjacent structure-mask alternative (Row 10).
- **Xing 2022** — *Context-conditioned modulation triggered by env change.* Cite for the RL-domain CLEAN-MAP precedent (Row 9).
- **Osman 2024** — *§3-4 Hopfield + gain; annealing-schedule interpretation.* Cite for the recurrent-attractor-net + gain family — the no-map row (Row 12) that names the recurrent-energy-landscape gain as a complementary family.
- **AlKilany & Goodman 2025** — *Results §"listening in the dips".* Cite for the multiplicative-only FiLM mechanism producing emergent attentional gain (Row 13).
- **Tambaş 2025** — *Krotov-Hopfield three-factor rule.* Cite for the global-modulator-gating-local-plasticity framing (Row 14).
- **Kudithipudi 2022** — *Plasticity-gating biological mechanisms survey.* Cite for the lifelong-learning umbrella (Row 15).

### Direction memo and predictions synthesis

- **v3 §3 + §5** ([direction memo](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)) — the unification claim, per-site dissociation scaffold, γ/β-clamp falsifier. The paper-shape memo extends v3 from "predictions" to "claim-shaped narrative with mathematical bridges". §3 cites the integration mapping for the *forward-pass* substrate; §4.3 cites it for the *complement* (Rodriguez-Garcia 2026) and the *boundary* (no-maps).
- **Predictions synthesis §3 + §5** ([predictions synthesis](../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md)) — the evidence base for v3 §5. §3 anchors the four behaviour-category tags used in this integration synthesis; §5 names the six gaps the project's substrate is positioned to fill, which §5 of this document inherits and translates into FiLM-variant operational form.

---

## Cross-references

- This document's primary write home: [`docs/project/references/FiLM/film_neuromod_integration_synthesis.md`](film_neuromod_integration_synthesis.md).
- FiLM corpus master index: [`film_lit_review.md`](film_lit_review.md).
- FiLM corpus cross-paper synthesis: [`film_synthesis.md`](film_synthesis.md).
- Neuromod corpus master review: [`../neuromodulatory_algorithms/neuromodulatory_algorithms_lit_review.md`](../neuromodulatory_algorithms/neuromodulatory_algorithms_lit_review.md).
- Neuromod corpus predictions synthesis (the canonical extraction for the mapping): [`../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md`](../neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md).
- Active direction memo: [`../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md`](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md).
- Project plan and gates G1, G2: [`../../project_plan.md`](../../project_plan.md).
- Project's null-result diagnosis series: [`../../../docs/develop/INDEX.md`](../../../develop/INDEX.md).
