---
title: "FiLM Corpus — Thematic and Historical Synthesis"
topic: FiLM
status: curated
last_updated: 2026-05-16
related:
  - film_lit_review.md
  - archive/film_conditional_modulation_review.md
scope: |
  Cross-paper thematic + historical synthesis across the 23 papers indexed
  in film_lit_review.md plus the De Vries 2017 CBN entry preserved from
  archive/film_conditional_modulation_review.md. Organized into seven
  clusters, a chronological narrative (2016 → 2026), a cross-cluster
  influence table, and project-specific open questions.
---

# FiLM Corpus — Thematic and Historical Synthesis

## 1. Plain-English entry point

### What is this corpus about?

The corpus is about **conditioning a neural network on a side signal**.
A "side signal" might be a natural-language question ("how many small
purple cylinders are there?"), a task identifier ("paint this image like
Van Gogh"), a desired playback speed (1.5x faster), a 512-dimensional
sentence embedding ("place the sponge in the tray"), a member index in
an ensemble ("you are member number 7 of 16"), a batch's domain-context
summary (the mean of this mini-batch's feature activations), or — in this
project's planned setting — a continuous physiological state ("the
agent's current interoceptive variable"). The architectural question is
the same in every case: **how should the side signal change what the
main network computes?**

### What are the main mechanisms?

Six families of answers appear in this corpus, all of which can be read
in plain English:

1. **FiLM** (Feature-wise Linear Modulation). For every channel of an
   intermediate feature map, multiply by a learned scale `γ` and add a
   learned shift `β`. The numbers `γ` and `β` are produced from the side
   signal by a small "FiLM generator" network. Two scalars per channel,
   applied as `y = γ·x + β`. Cheap, broadly applicable, decoupled from
   network resolution. *This is the corpus's namesake mechanism.*
2. **Conditional batch / instance / layer normalization**. The same
   per-channel `γ`, `β` numbers, except they're glued onto the affine
   step of a normalization layer (which standardizes the activations
   before applying the affine). Historically the precursor to FiLM —
   FiLM is what you get if you decouple the affine modulation from the
   normalization step.
3. **Hypernetworks**. One network's job is to *output the weights* of a
   second network. The first network is small (a "genotype"); the second
   is large (a "phenotype"); together, the system can generate
   context-dependent weight matrices on demand. FiLM is the special case
   where the hypernetwork outputs only per-channel scales and shifts.
4. **Mixture of Experts (MoE)**. Instead of one big network, build a
   bank of `n` small "experts" plus a tiny "gating network" that picks a
   sparse subset (say 2 out of 256) for each input. Only the selected
   experts compute. Total parameters can be enormous; per-example
   compute stays small.
5. **Neural Module Networks (NMN)**. Build a small library of typed
   neural modules (`find`, `transform`, `combine`, `describe`, `count`,
   ...) and *assemble a different network for each input* from those
   modules. The original version parses the input with an off-the-shelf
   grammar parser; the end-to-end version learns the layout with
   reinforcement learning.
6. **Attention**. Every position in a sequence can directly compare
   itself to every other position via dot-product similarity and form a
   weighted blend of their representations. Unlike FiLM (which
   modulates *channels*), attention modulates *positions*. It is a
   parallel, not competing, conditioning primitive.

### What are the open questions?

A few that recur across the corpus and matter to the project:
**(a)** When is FiLM's per-channel affine modulation enough? When is it
too narrow, and a richer mechanism (kernel modulation, full
hypernetwork) actually needed? **(b)** When can FiLM's `γ`/`β` channel
double as a source of ensemble diversity (FiLM-Ensemble) or as a
domain-context adapter (CASA)? **(c)** How do FiLM-conditioning
(per-channel scaling, used by the project) and the project's
heteroscedastic loss (per-input variance, also used by the project)
combine — are they redundant, complementary, or competing? **(d)** In
recurrent / RL settings without batch normalization, does FiLM still
work? **(e)** Does *attention* outperform FiLM as a conditioning
primitive when the side signal is itself a sequence? **(f)** What
distinguishes "FiLM as a controller signal" (BC-Z, Nikulin) from "FiLM
as a member-diversity signal" (Turkoglu, TabM)?

The rest of this document organizes the corpus into thematic clusters,
walks the history from 2016 to 2026, and ends with six specific open
questions the FiLM corpus poses for this project.

---

## 2. Thematic clusters

The 23 fresh per-paper reviews plus the archived De Vries 2017 entry are
organized into seven clusters. For each cluster: a plain-English
summary, the papers belonging to it (with a one-line "what this paper
contributes"), shared assumptions, disagreements / open questions, and
the 2–4 key equations that recur. Per-paper deep-dive math lives in
each `reviews/<slug>.md` — never re-pasted here.

### Cluster A — FiLM core and feature-wise affine modulation

**Plain-English summary.** Four papers that work directly on the FiLM
mechanism in its canonical "per-channel γ, β" form, varying what the
conditioner is (a question, block-pooled self-activations, a scalar
speed factor, a task-mixture vector) and where the modulation is
inserted (every residual block, every IN layer, only after the encoder,
along the time axis). The cluster's collective evidence: the same tiny
modulation operator — two numbers per channel — is expressive enough to
re-purpose a fixed feature stack across visual reasoning, audio
super-resolution, speech time-scale modification, and multi-task
image-to-image translation.

**Papers in this cluster.**

- [Perez et al. 2018 — FiLM](reviews/perez_2018_film.md) — names FiLM,
  halves CLEVR state-of-the-art error, decouples modulation from
  normalization, demonstrates (γ, β) algebra for zero-shot composition.
- [Birnbaum et al. 2019 — Temporal FiLM](reviews/birnbaum_2019_temporal_film.md)
  — TFiLM: block-piecewise-constant FiLM along the time axis, with an
  LSTM over block-pooled self-activations producing per-block (γ, β).
- [Wisnu et al. 2025 — STSM-FiLM](reviews/wisnu_2025_stsm_film.md) —
  FiLM with a *scalar* speed-factor conditioner α ∈ ℝ; uses the `(1 + γ)`
  init trick so the model starts at the identity transformation.
- [Takeda et al. 2021 — Multi-task feature modulation](reviews/takeda_2021_multi_task_feature_mod.md)
  — IN + FiLM at every conv layer for 6 heterogeneous image-to-image
  tasks on one backbone with <2% task-specific parameters; identifies
  Method-3 (synthesized mixed-GT) as the only recipe that lets the
  conditional vector double as a smooth mixture coordinate at inference.

**Shared assumptions.**

- The conditioner can be funneled into a low-dimensional vector that a
  small MLP / linear layer projects to per-layer (γ, β) values.
- Per-channel affine modulation has enough capacity to redirect the
  network's behavior — even when the tasks are qualitatively different
  (pixel segmentation vs. stylization, speech reasoning vs. ChIP-seq
  super-resolution).
- A normalization step in the host network is helpful but not required;
  Perez's ablations show FiLM-without-BN still reaches 93.7% on CLEVR
  vs. 97.4% with BN.

**Disagreements / open questions in this cluster.**

- *Where to insert FiLM.* Perez 2018 places FiLM after BN-without-affine
  inside every residual block. Takeda 2021 places IN+FiLM after every
  conv except the last, and shows EncIn (FiLM only in decoder) is worse
  than FiLM in both encoder and decoder. STSM-FiLM injects FiLM at
  multiple decoder layers in HiFi-GAN variants. The cluster has not
  settled a universal answer; the empirical rule is "more is better,
  except possibly the last layer".
- *Should modulation vary across time or stay uniform?* TFiLM lets (γ, β)
  vary per block via an LSTM on self-activations; STSM-FiLM applies the
  same (γ, β) at every time step and handles time stretching by
  separate linear interpolation. They occupy complementary corners of
  the FiLM design space (time-varying self-conditioning vs.
  time-uniform external conditioning).
- *Can the conditional vector be used as a continuous interpolation
  coordinate?* Takeda 2021's headline result is *only* if the network
  was trained on examples spanning the corners of the mixture space —
  Method 1 (train singles, expect mixtures) and Method 2 (summed loss)
  both fail. STSM-FiLM's continuous α ∈ [0.5, 2.0] succeeds with
  uniform sampling at training time. The Perez 2018 zero-shot trick via
  (γ, β) algebra works for *attribute composition*, not for arbitrary
  task mixing — suggesting compositionality is task-structure-dependent.

**Key equations.**

The base FiLM equation (from Perez et al. 2018):

$$
\gamma_{i,c} = f_c(x_i), \qquad \beta_{i,c} = h_c(x_i),
$$

$$
\mathrm{FiLM}(F_{i,c} \mid \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}.
$$

TFiLM extends this to time blocks (Birnbaum et al. 2019), where index
$b$ ranges over blocks rather than examples and the (γ, β) generator is
an LSTM over the block-pooled activations $F^{\text{pool}}$:

$$
\bigl((\gamma_b, \beta_b), h_b\bigr) = \mathrm{LSTM}\bigl(F^{\text{pool}}_{b,:}; h_{b-1}\bigr),
\qquad F^{\text{norm}}_{b,t,c} = \gamma_{b,c} \cdot F^{\text{blk}}_{b,t,c} + \beta_{b,c}.
$$

STSM-FiLM's identity-init trick (Wisnu et al. 2025):

$$
\hat f_t = (1 + \gamma_\alpha) \cdot f_t + \beta_\alpha,
\qquad (\gamma_\alpha, \beta_\alpha) = \mathrm{MLP}(\alpha).
$$

The `1 +` ensures $\gamma^{\text{eff}}|_{\alpha=\text{init}} \approx 1$
and $\beta|_{\alpha=\text{init}} \approx 0$, so the model initializes
to the identity.

### Cluster B — Normalization-conditioning lineage

**Plain-English summary.** Three papers that lay the conceptual track
FiLM later unified, plus one paper (Santurkar) that explains why the
underlying normalization-then-affine substrate trains well at all. The
lineage is **CIN → CBN → AdaIN → FiLM**: each paper takes the previous
one's "store a per-class (γ, β) lookup" idea and replaces the
class-indexed table with a richer generator (a language LSTM for CBN; a
deterministic style-image-statistics function for AdaIN; a learned
neural-network conditioner free of the normalization step for FiLM).
Together they establish the load-bearing empirical surprise: *a few
thousand scalars in the per-channel affine carry an enormous amount of
semantic content* (style identity, language semantics, task structure).

**Papers in this cluster.**

- [Dumoulin et al. 2017 — Conditional Instance Normalization (CIN)](reviews/dumoulin_2017_cond_instance_norm.md)
  — 32 painting styles in one network via an `N × C` lookup of
  per-style (γ, β); only 0.2% of parameters are style-specific.
  Demonstrates linear interpolation in (γ, β) space produces smooth
  pastiches.
- [Huang & Belongie 2017 — AdaIN](reviews/huang_belongie_2017_adain.md)
  — Drops the learned lookup table; computes (γ, β) on the fly as
  per-channel mean and std of a style image's deep features. Zero
  learnable parameters in the modulation layer; arbitrary styles at
  test time.
- [Santurkar et al. 2018 — How does BatchNorm help optimization?](reviews/santurkar_2018_batchnorm_optimization.md)
  — Demolishes the "internal covariate shift" explanation;
  demonstrates that BN's benefit is *loss-landscape smoothing* with
  formal Lipschitz / β-smoothness bounds.
- De Vries et al. 2017 — Conditional Batch Normalization (CBN) /
  MODERN — *no fresh per-paper review*; preserved from
  [`archive/film_conditional_modulation_review.md` §2](archive/film_conditional_modulation_review.md).
  Introduces residual-offset CBN that modulates ResNet-50's BN affines
  from a language LSTM; argues on neuroscience grounds (top-down
  modulation reaching early visual cortex) that conditioning should
  reach deep into the perceptual stack, not just the head.

**Shared assumptions.**

- The per-channel mean and std of a feature map carry "style" or
  "context" information; standardizing them is "style normalization"
  and re-applying conditional affines is "style injection". (Huang &
  Belongie 2017 makes this explicit; Dumoulin 2017 and De Vries 2017
  implicitly rely on it.)
- A normalization layer is the natural location for the conditional
  affine because it provides a calibrated input distribution for the
  affine to operate on. (Santurkar 2018 explains why this normalizes
  the loss landscape, smoothing optimization.)
- Initialization at the *identity transformation* matters: CBN
  zero-initializes its conditional-offset MLP so training begins at
  the pretrained ResNet behavior; Dumoulin's CIN warm-starts new
  styles from existing rows; STSM-FiLM uses the `(1 + γ)` trick.

**Disagreements / open questions.**

- *Is the normalization step doing the work, or the conditional
  affine?* Santurkar 2018 says BN's benefit is loss-landscape
  smoothing, *not* feature-statistic normalization specifically (any
  L_p-norm-based normalization works). Perez 2018 then shows that
  removing normalization entirely from a FiLM-ed CLEVR network still
  works — just 4 percentage points worse. The implication: in the
  FiLM era, the conditional affine is the load-bearing piece;
  normalization is a useful but non-essential pre-conditioner.
- *Should the conditional affine be a residual offset or a direct
  prediction?* De Vries 2017 chooses residual (`γ_c(z) = γ_c +
  Δγ_c(z)`) to anchor at pretrained behavior; FiLM (Perez 2018)
  abandons this and predicts (γ, β) directly. The residual form
  cannot *suppress* a channel that the pretrained network actively
  uses; the direct form can, at the cost of larger training signal
  requirements.
- *Per-image (IN) or per-batch (BN) normalization?* CIN and AdaIN use
  instance normalization specifically because per-image
  feature-statistic stripping interacts cleanly with per-image style
  re-injection. CBN uses batch normalization because the application
  (VQA on real images) doesn't need per-image style stripping. The
  choice is task-dependent.

**Key equations.**

Conditional Instance Normalization (Dumoulin et al. 2017):

$$
\mathrm{CIN}(x \mid s)_{n,c,h,w} = \gamma_{s,c} \cdot \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}} + \beta_{s,c},
$$

where the affine pair $(\gamma_{s,c}, \beta_{s,c})$ is the `s`-th row
of a learned `N × C` lookup table.

AdaIN (Huang & Belongie 2017): replace the lookup with deep-feature
statistics of a style image $y$:

$$
\mathrm{AdaIN}(x, y)_{n,c,h,w} = \sigma_{n,c}^{\mathrm{IN}}(y) \cdot \frac{x_{n,c,h,w} - \mu_{n,c}^{\mathrm{IN}}(x)}{\sigma_{n,c}^{\mathrm{IN}}(x)} + \mu_{n,c}^{\mathrm{IN}}(y).
$$

Conditional Batch Normalization (De Vries et al. 2017, residual-offset
form from `archive/film_conditional_modulation_review.md` Eq. 7–8):

$$
\gamma_c(z) = \gamma_c + \Delta\gamma_c(z), \quad \beta_c(z) = \beta_c + \Delta\beta_c(z),
$$

$$
\mathrm{CBN}(F_{b,c,h,w} \mid z) = \gamma_c(z) \cdot \frac{F_{b,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}} + \beta_c(z).
$$

Santurkar 2018's loss-Lipschitz bound (Theorem 4.1, simplified):

$$
\|\nabla_{y_j} \hat{\mathcal{L}}\|^2 \le \frac{\gamma^2}{\sigma_j^2} \left( \|\nabla_{y_j} \mathcal{L}\|^2 - \mathrm{(positive\ correction\ terms)} \right),
$$

so BN tightens the loss Lipschitz constant — the formal version of
"BN smooths the landscape".

### Cluster C — Hypernetworks

**Plain-English summary.** Three papers that frame FiLM's super-class:
**one network produces another's weights**. The static (Ha 2016) version
generates conv kernels from learned per-layer embedding vectors; the
dynamic version generates RNN weights at every timestep from the
current context. The Bayesian extension (Krueger 2017) lifts the
deterministic generator to a normalizing flow that produces *samples*
from a posterior over weights. The theoretical paper (Galanti & Wolf
2020) proves the resulting parameter efficiency advantage is
*exponential* in the conditioning dimension, justifying the entire
"capacity belongs in the conditioner, not in a giant shared primary"
design philosophy that FiLM, CBN, MoE, and module networks all
exemplify.

**Papers in this cluster.**

- [Ha, Dai & Le 2016 — HyperNetworks](reviews/ha_2016_hypernetworks.md)
  — The constructive paper. Static (CNN) and dynamic (RNN)
  hypernetworks; HyperRNN's row-scaling trick is structurally identical
  to FiLM.
- [Galanti & Wolf 2020 — On the modularity of hypernetworks](reviews/galanti_wolf_2020_hypernet_modularity.md)
  — The theory paper. Proves hypernetwork primary networks scale as
  $O(\epsilon^{-m_1/r})$ while embedding-method primaries scale as
  $\Omega(\epsilon^{-(m_1+m_2)/r})$.
- [Krueger et al. 2017 — Bayesian Hypernetworks](reviews/krueger_2017_bayesian_hypernets.md)
  — The probabilistic extension. Normalizing-flow hypernetwork with
  exact log-density via change of variables; matches MC-dropout on
  classification and beats it on adversarial-example detection.

**Shared assumptions.**

- The primary network's weights are *not* the optimization variables;
  instead, a hypernetwork's parameters are, and the primary's weights
  are *produced* by the hypernetwork.
- Capacity should be concentrated in the hypernetwork (the conditioner /
  router), and the primary network kept small. Galanti & Wolf 2020
  makes this rigorous.
- The hypernetwork can be invertible (Krueger 2017) for Bayesian
  treatment, or deterministic (Ha 2016) for parameter compression /
  context-dependent dynamics.

**Disagreements / open questions.**

- *How much hypernetwork capacity actually transfers to FiLM?* FiLM is
  a *restricted* hypernetwork — it outputs only per-channel (γ, β)
  rather than full weight matrices. Galanti & Wolf 2020 prove the
  modularity advantage for unrestricted hypernetworks; whether the
  same exponential gap holds for FiLM specifically is an *open
  theoretical question* (the Connections section of Galanti & Wolf
  explicitly flags this).
- *Static vs. dynamic vs. per-task hypernetworks?* Ha 2016 covers
  static (per-layer embedding) and dynamic (per-timestep context); FiLM
  with a task ID is something in between (per-task embedding, but a
  single embedding applied across the whole network). The corpus does
  not have a clean head-to-head comparison of these three regimes.
- *Bayesian over the modulator, or over the primary?* Krueger 2017
  makes the primary weights stochastic; a "Bayesian FiLM" would
  instead make (γ, β) stochastic. Krueger discusses this as a natural
  specialization but does not test it.

**Key equations.**

Hypernetwork (Ha 2016): for each layer $j$ of a CNN, an embedding
$z^j \in \mathbb{R}^{N_z}$ is fed through a two-layer linear hypernet
that outputs the full convolutional kernel $K^j$:

$$
K^j = g(z^j) = \langle W_{out}, W_i z^j + B_i \rangle + B_{out}.
$$

HyperRNN's memory-efficient row-scaling trick (which is structurally
identical to FiLM):

$$
h_t = \phi\Bigl( d_h(z_h) \odot (W_h h_{t-1}) + d_x(z_x) \odot (W_x x_t) + b(z_b) \Bigr),
$$

with $d_h(z_h) = W_{hz} z_h$ — exactly $\gamma$-style per-row scaling
driven by a context embedding.

Bayesian Hypernetwork (Krueger 2017): the weights of the primary
network are sampled via an invertible flow $h$:

$$
\theta = h(\epsilon), \qquad \epsilon \sim \mathcal{N}(0, I_D),
$$

$$
\log q(\theta) = \log q_\epsilon(\epsilon) - \log \Bigl| \det \frac{\partial h(\epsilon)}{\partial \epsilon} \Bigr|.
$$

Modularity bound (Galanti & Wolf 2020, Theorem 4): for a hypernetwork
$h(x, I) = g(x; f(I; \theta_f))$ approximating a target in $W^{r,m}$,

$$
N_g = O(\epsilon^{-m_1 / r}),
$$

where $m_1$ is the $x$-dimension and *not* the conditioning dimension
$m_2$ — capacity is absorbed into $f$.

### Cluster D — Modular reasoning and mixture of experts

**Plain-English summary.** Three papers offering **discrete-composition**
alternatives to FiLM's continuous modulation. Neural Module Networks
(NMN) parse the question into a symbolic tree of typed neural modules
(`find`, `transform`, `combine`, `describe`, `count`); End-to-End Module
Networks (N2NMN) learn the layout itself via REINFORCE; Mixture of
Experts (MoE) replaces a dense FFN with a bank of `n` small experts and
a sparse gate that picks `k`. All three break the "uniform-compute"
assumption of a regular network: only a sparse subset of the
architecture is active per input, and *which* subset is conditioned on
the input.

**Papers in this cluster.**

- [Andreas et al. 2016 — Neural Module Networks (NMN)](reviews/andreas_2016_neural_module_networks.md)
  — Parser-driven module assembly; compositional generalization to
  longer questions than seen at training.
- [Hu et al. 2017 — End-to-End Module Networks (N2NMN)](reviews/hu_2017_e2e_module_networks.md)
  — Replaces the parser with a learned seq2seq RNN trained by
  REINFORCE + behavioral cloning; reaches 83.7% on CLEVR.
- [Shazeer et al. 2017 — Sparsely-Gated Mixture-of-Experts](reviews/shazeer_2017_sparse_moe.md)
  — `n`-expert bank with noisy top-`k` gating + importance / load
  auxiliary losses for utilization balance. Up to 137-billion-parameter
  models with bounded per-example compute.

**Shared assumptions.**

- Different inputs should activate different parts of the network. The
  router (gate / layout policy / parser) is the part that decides
  which parts.
- The router's decisions must be *learnable* end-to-end. MoE uses
  smoothed noisy top-`k` softmax (gradient flows through the surviving
  `k`); N2NMN uses REINFORCE with behavioral-cloning bootstrap.
- Sparse routing creates utilization-imbalance failure modes; MoE
  solves this with auxiliary CV-of-importance + CV-of-load losses.

**Disagreements / open questions.**

- *Discrete or continuous routing?* MoE's gate is dense softmax then
  hard top-`k`; NMN's layout is fully discrete; FiLM's modulation is
  fully continuous. The empirical evidence is that the dense /
  continuous path (FiLM) actually beats the discrete-composition path
  (N2NMN) on CLEVR — 97.7% vs. 83.7%. But N2NMN's predicted layouts
  are *interpretable* in a way FiLM's (γ, β) histograms are not.
- *Sparse activation vs. dense modulation?* MoE makes effective
  parameter count >> per-example compute; FiLM makes effective
  behavioral repertoire >> trained parameter count. They reshape the
  capacity-vs-compute trade-off along orthogonal axes.
- *Compositional generalization?* NMN shows compositional
  generalization to longer questions (train ≤5 modules, test 6).
  Perez 2018 demonstrates an analogous "(γ, β) algebra" allowing
  zero-shot compositional generalization (cyan-cube from cyan-sphere
  + brown-cube − brown-sphere). The mechanisms are different (discrete
  module re-composition vs. continuous parameter-vector arithmetic)
  but both achieve the same end.

**Key equations.**

MoE layer (Shazeer 2017):

$$
y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x),
$$

with noisy top-`k` gating

$$
G(x) = \mathrm{Softmax}\bigl(\mathrm{KeepTopK}\bigl(H(x), k\bigr)\bigr),
\qquad H(x)_i = (x W_g)_i + \mathcal{N}(0,1) \cdot \mathrm{Softplus}\bigl((x W_{noise})_i\bigr).
$$

NMN composition (Andreas 2016): for an input string $w$ parsed into
symbolic form $\sigma(w)$, the layout tree $T = P(w)$ is mapped
recursively (leaves → `find`, root → `describe`/`measure`), and the
induced computation is

$$
p(y \mid w, x; \theta) = M_T\bigl(x; \{\theta_m\}_{m \in T}\bigr).
$$

N2NMN's REINFORCE objective (Hu 2017):

$$
\nabla_\theta L = \mathbb{E}_{l \sim p(l|q;\theta)}\bigl[\tilde L(\theta, l) \, \nabla_\theta \log p(l|q;\theta) + \nabla_\theta \tilde L(\theta, l)\bigr],
$$

where $\tilde L$ is the answer-prediction cross-entropy and $l$ is the
sampled discrete layout token sequence.

### Cluster E — Attention as a parallel conditioning primitive

**Plain-English summary.** One paper, but a load-bearing one: the
Transformer (Vaswani et al. 2017). Attention is a *different*
conditioning primitive than FiLM — it modulates *positions* (every
position in one sequence forms a softmax-weighted aggregation over
positions in another sequence), where FiLM modulates *channels*
(every channel of a feature map gets its own scale-and-shift). Both
mechanisms answer the question "how should the side signal change the
main stream's computation", but they answer it on orthogonal axes.

**Paper in this cluster.**

- [Vaswani et al. 2017 — Attention Is All You Need](reviews/vaswani_2017_attention.md)
  — The Transformer: scaled dot-product attention, multi-head, sinusoidal
  positional encoding. Beats RNN- and CNN-based translation while
  training in a fraction of the time.

**Shared assumptions** (with the rest of the corpus).

- A side signal should influence the main network's computation, not
  just be concatenated at the head.
- A learnable, differentiable, parallelizable conditioning primitive
  is preferable to a hand-designed one.

**Disagreements with FiLM-style modulation.**

- *Granularity of modulation.* FiLM modulates *channels* (one γ_c,
  β_c per feature map channel); attention modulates *positions* (one
  weight per source token, shared across all feature dimensions of
  that token's value).
- *Parameter cost.* FiLM costs $O(C)$ per modulation; cross-attention
  costs $O(n_q \cdot n_k)$ in compute and produces $n_q \cdot n_k$
  soft attention coefficients.
- *Inductive bias.* FiLM assumes the side signal affects *what
  features to amplify or suppress*; attention assumes the side signal
  provides *content to read from*. This makes FiLM the lighter choice
  when the side signal is low-dimensional (a task ID, a few scalars);
  attention becomes attractive when the side signal is itself a
  structured sequence.

The corpus contains one direct empirical comparison: Vuorio et al. 2019
(cited inside [`turkoglu_2022_film_ensemble.md`](reviews/turkoglu_2022_film_ensemble.md)
and [`abdollahzadeh_2021_multimodal_meta.md`](reviews/abdollahzadeh_2021_multimodal_meta.md))
report that **"FiLM outperforms attention-based modulation in this
context [multimodal MAML] and is more stable"**. This is the only
head-to-head FiLM-vs-attention modulation comparison in the corpus and
is suggestive but not definitive — task-conditioning with a
low-dimensional task identifier is exactly the regime where FiLM is
expected to win.

**Key equations** (from Vaswani 2017).

Scaled dot-product attention:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

Multi-head attention with $h = 8$ heads and $d_k = d_v = d_{\text{model}} / h$:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}\bigl(\mathrm{head}_1, \ldots, \mathrm{head}_h\bigr) W^O.
$$

### Cluster F — Probabilistic / uncertainty / ensemble FiLM

**Plain-English summary.** Four papers covering the line of work that
connects FiLM to Bayesian deep learning, uncertainty quantification,
and ensembles — the line directly relevant to the project's
heteroscedastic precision head. The cluster's central thread: the same
per-channel γ/β operator that FiLM uses for *context conditioning* can
double as a source of *ensemble diversity* (Turkoglu 2022) and the same
heteroscedastic loss that produces per-input aleatoric uncertainty
estimates (Kendall & Gal 2017) combines naturally with such implicit
ensembles to produce a full aleatoric / epistemic decomposition. The
survey (Gawlikowski 2023) provides the field-standard taxonomy in which
all of this sits.

**Papers in this cluster.**

- [Kendall & Gal 2017 — What uncertainties do we need?](reviews/kendall_gal_2017_uncertainties.md)
  — The project's *anchor heteroscedastic-regression loss*: $L =
  \tfrac{1}{2} \exp(-s) \|y - \hat y\|^2 + \tfrac{1}{2} s$ with $s =
  \log \sigma^2$. Combines learned per-input variance with MC-dropout
  for an aleatoric / epistemic split.
- [Gawlikowski et al. 2023 — A survey of uncertainty in deep neural networks](reviews/gawlikowski_2023_uncertainty_survey.md)
  — The 77-page map: 4-branch taxonomy (single deterministic, BNN,
  ensemble, test-time augmentation), uncertainty measures, calibration,
  applications.
- [Turkoglu et al. 2022 — FiLM-Ensemble](reviews/turkoglu_2022_film_ensemble.md)
  — Implicit ensemble via per-member γ/β; ~1.3% parameter overhead vs.
  1500% for an explicit ensemble of ResNet-18. Higher diversity than the
  explicit ensemble on CIFAR-10.
- [Gorishniy et al. 2025 — TabM](reviews/gorishniy_2025_tabm.md) —
  BatchEnsemble-based implicit ensembling on tabular MLPs;
  FiLM-adjacent (same multiplicative-gating bias, modulation at linear
  layers instead of normalization layers).

**Shared assumptions.**

- Useful uncertainty has two components: irreducible data noise
  (aleatoric) and reducible model ignorance (epistemic).
- A "diverse ensemble" — explicit or implicit — is the cleanest way to
  estimate epistemic uncertainty.
- The per-channel γ/β operator has enough expressive capacity to
  instantiate functionally distinct sub-networks from a shared
  backbone, not just to condition one backbone on a context.

**Disagreements / open questions.**

- *FiLM-Ensemble vs. explicit deep ensemble?* Turkoglu 2022 matches or
  beats the explicit ensemble on Expected Calibration Error, OOD
  detection, and *diversity* (9.2% pairwise disagreement vs. 6.8% for
  the explicit ensemble at $M=16$ on CIFAR-10), and is only slightly
  beaten on raw accuracy (79.4% vs. 81.6% on CIFAR-100 ResNet-18).
  Gorishniy 2025 rejects FiLM-Ensemble for tabular MLPs only because
  those backbones lack normalization layers — a pragmatic
  architectural constraint, not a principled rejection.
- *FiLM-Ensemble or evidential / Dirichlet?* Gawlikowski 2023 lists
  evidential / prior networks (Sensoy 2018, Malinin & Gales 2018,
  Amini 2020) as single-deterministic alternatives that give OOD
  detection in one forward pass. FiLM-Ensemble needs $M$ passes (but
  parallelizable). The cluster has not directly compared them.
- *How do MC-dropout and FiLM-Ensemble compose?* Kendall & Gal 2017
  uses MC-dropout for epistemic; Turkoglu 2022 uses FiLM-Ensemble.
  Combining them — MC-dropout over a FiLM-Ensemble — is unexplored in
  the corpus.
- *Where does the heteroscedastic loss go inside an ensemble?* In
  Lakshminarayanan 2017 (cited in Gawlikowski 2023), each ensemble
  member has its own heteroscedastic head; predictive variance
  decomposes via law of total variance into the average member
  variance (aleatoric) plus the spread across member means (epistemic).
  FiLM-Ensemble inherits this structure naturally.

**Key equations.**

The heteroscedastic regression loss (Kendall & Gal 2017 Eq. 6 — the
*project's anchor formula*):

$$
\mathcal{L}_{\text{BNN}}(\theta) = \frac{1}{D} \sum_i \frac{1}{2} \exp(-s_i) \cdot \|y_i - \hat y_i\|^2 + \frac{1}{2} s_i, \qquad s_i = \log \hat\sigma_i^2.
$$

The variance decomposition (from Lakshminarayanan 2017 as cited in
Gawlikowski 2023 §3.3.4):

$$
\bar\sigma^2(x) = \underbrace{\frac{1}{M} \sum_i \hat\sigma_i^2(x)}_{\text{aleatoric}} + \underbrace{\frac{1}{M} \sum_i \hat\mu_i^2(x) - \bar\mu^2(x)}_{\text{epistemic}}.
$$

FiLM-Ensemble (Turkoglu 2022): replace continuous conditioning $z$
with discrete member index $m \in \{1, \ldots, M\}$, store $\gamma_n^m,
\beta_n^m$ as parameters per layer per member:

$$
\mathrm{FiLM}\bigl(F_n \mid \gamma_n^m, \beta_n^m\bigr) = \gamma_n^m \circ F_n + \beta_n^m,
$$

with Xavier-uniform initialization of gain $\rho$ controlling the
ensemble diversity.

BatchEnsemble (Gorishniy 2025): per-member rank-1 modulation of a
shared weight matrix:

$$
W_i = W \odot (s_i\, r_i^\top), \qquad
l_i(x_i) = s_i \odot \bigl(W (r_i \odot x_i)\bigr) + b_i.
$$

This is the *same multiplicative-gating inductive bias* as FiLM, with
the modulation inserted around the linear layer's weight matrix
instead of around a normalization layer's output.

### Cluster G — Applications: multi-task, meta-learning, RL, robotics, domain generalization

**Plain-English summary.** Five papers that deploy FiLM-style
conditioning in concrete applied settings: multimodal meta-learning,
zero-shot robotic manipulation, hierarchical-achievement RL, offline RL
with anti-exploration bonuses, and domain generalization across painting
styles / photo subjects. The cluster's collective evidence: FiLM
generalizes well across domains and across the *scale* of its
deployment — it can serve as the global conditioning channel of a 7-DoF
visuomotor policy (BC-Z), a small fusion gadget inside a contrastive
representation head (Moon 2023), an OOD-detection prior that smooths
gradients for an actor (Nikulin 2023), a 6-parameter domain-context
adapter (CASA / Yan & Guo 2025), or a *too-narrow* modulator that
needs upgrading to per-weight kernel modulation when task distributions
are highly multimodal (Abdollahzadeh 2021).

**Papers in this cluster.**

- [Abdollahzadeh et al. 2021 — Kernel Modulation (KML)](reviews/abdollahzadeh_2021_multimodal_meta.md)
  — Proves FiLM ≡ uniform scalar rescaling of a conv kernel; proposes
  per-weight Kernel Modulation as a richer alternative. Imports the
  *transference* metric from multi-task learning.
- [Jang et al. 2022 — BC-Z](reviews/jang_2022_bcz.md) — Zero-shot
  robotic generalization via FiLM-conditioned ResNet-18 visuomotor
  policy and a frozen pretrained sentence encoder. The canonical
  "FiLM-on-policy" demonstration in real robotics.
- [Moon et al. 2023 — Hierarchical achievements via contrastive learning](reviews/moon_2023_hierarchical_achievements.md)
  — FiLM as a *small fusion gadget* (action → state-embedding) inside
  an auxiliary contrastive head; SOTA on Crafter at 9M parameters.
- [Nikulin et al. 2023 — SAC-RND](reviews/nikulin_2023_anti_exploration_rnd.md)
  — FiLM as an *RND-prior conditioner* with the discovered property of
  *gradient-landscape smoothing*; matches Q-ensemble SOTA on D4RL
  without ensembles.
- [Yan & Guo 2025 — CASA: Context-Aware Self-Adaptation (CaFiLM)](reviews/yan_guo_2025_context_aware_dg.md)
  — Domain generalization via a 6-parameter shared FiLM module that
  uses mini-batch feature mean as "domain context". SOTA on DomainBed.

**Shared assumptions.**

- The conditioner is a low-bandwidth signal (a task embedding, an
  action vector, a state vector, a batch's feature mean, a speed
  factor); a much larger feature stack receives it and changes its
  behavior accordingly.
- FiLM's *same-vector-space* property — modulation, not remapping — is
  often the inductive bias that makes the application work. This is
  most explicit in CASA (the classifier downstream is frozen) and in
  SAC-RND (the actor descends a gradient surface that needs to remain
  smooth across action space).
- FiLM works in both image (BC-Z, CASA), text-image (BC-Z), audio
  (TFiLM, STSM-FiLM), tabular (TabM, indirectly), and RL
  (BC-Z, Moon 2023, Nikulin 2023) settings.

**Disagreements / open questions.**

- *Is FiLM expressive enough for highly multimodal task distributions?*
  Abdollahzadeh 2021 says no — per-channel scalar modulation leaves
  "tasks fighting for capacity" — and proposes KML's per-weight
  modulation. Takeda 2021 says yes — 6 disparate image-translation
  tasks coexist on one backbone via FiLM at every IN layer. The
  resolution may be: FiLM's expressivity depends critically on (i)
  task heterogeneity, (ii) how richly FiLM is inserted (every layer
  vs. only the head), and (iii) whether the model is meta-learning a
  few-shot adapter (where capacity is tight) or training on many
  examples of each task (where capacity matters less). Takeda's
  6 tasks share an image-to-image structure; KML's 5-mode benchmark
  mixes Omniglot characters with Aircraft images.
- *FiLM as a controller signal vs. as a member-diversity signal?* BC-Z
  uses FiLM as the *primary* conditioning channel of a 7-DoF
  visuomotor policy; FiLM-Ensemble (Cluster F) uses FiLM channels as
  *ensemble heads*. Both work; they sit on opposite ends of a
  spectrum.
- *FiLM in the policy head vs. in the value head vs. in the
  representation head?* The applications cluster has all four
  variants. BC-Z places FiLM at every ResNet block of the policy
  network; Moon 2023 places it only inside the auxiliary contrastive
  head; Nikulin 2023 places it inside the RND prior network used to
  produce the actor's bonus; CASA places it between feature extractor
  and classifier. No paper offers a head-to-head comparison.

**Key equations.**

Kernel Modulation (Abdollahzadeh 2021), the per-weight generalization
of FiLM:

$$
\hat W^l_T = W^l \odot \bigl(J + M^l(\upsilon_T, \phi)\bigr),
\qquad M^l(\upsilon_T, \phi) = g^l_{\phi_1}(\upsilon_T) \otimes g^l_{\phi_2}(\upsilon_T).
$$

The outer-product factorization keeps the generator small even when
$M^l$ has the full kernel shape.

BC-Z's FiLM-on-policy (Jang 2022): for each of the four ResNet blocks
$k$ with $C_k$ channels and 512-dim task embedding $z$,

$$
\boldsymbol\gamma^{(k)}(z) = W_\gamma^{(k)} z + b_\gamma^{(k)},
\qquad \boldsymbol\beta^{(k)}(z) = W_\beta^{(k)} z + b_\beta^{(k)},
$$

$$
\widetilde F^{(k)}_{h,w,c} = \gamma^{(k)}_c(z) \cdot F^{(k)}_{h,w,c} + \beta^{(k)}_c(z).
$$

CaFiLM (Yan & Guo 2025): for feature $z_c$ and mini-batch mean $\mu_c$,
a *shared* 2×2 linear projection produces (γ_c, β_c):

$$
\begin{bmatrix} \gamma_c \\ \beta_c \end{bmatrix} = A \begin{bmatrix} z_c \\ \mu_c \end{bmatrix} + b,
\qquad \widetilde z_c = \gamma_c \cdot z_c + \beta_c.
$$

The 6 parameters $A \in \mathbb{R}^{2 \times 2}$, $b \in \mathbb{R}^2$
are shared across all `C` channels.

SAC-RND's FiLM-prior gradient-smoothing (Nikulin 2023): for action
hidden representation $h$ and state $s$,

$$
\boldsymbol\gamma(s) = W_\gamma s + b_\gamma,
\qquad \widetilde h = \boldsymbol\gamma(s) \odot h + \boldsymbol\beta(s),
$$

and the discovered property is that this multiplicative gating produces
a smoother $b(s, a)$ surface than $s$-and-$a$ concatenation — smoother
enough that the actor can follow the anti-exploration anti-gradient to
its global minimum.

---

## 3. Historical timeline

A chronological walk through the corpus's main influence flows.
Sub-sections use ~3-year windows.

### 2016 — The hypernetwork and the module network

Two papers from very different traditions plant the seeds the rest of
the corpus grows from.

**Ha, Dai & Le 2016** ([review](reviews/ha_2016_hypernetworks.md))
introduces hypernetworks: one network produces another's weights, in
either static (per-layer embedding) or dynamic (per-timestep context)
form. The dynamic variant's row-scaling trick — element-wise
multiplication of a fixed weight matrix's rows by context-dependent
scalars — is *structurally identical* to FiLM at the per-row
granularity, two years before FiLM is named. The hypernetwork framing
also names a design principle that the rest of the corpus will rely on:
the conditioner should be a network whose output is *the parameters
that vary with context*, not just a concatenated input.

**Andreas et al. 2016** ([review](reviews/andreas_2016_neural_module_networks.md))
introduces Neural Module Networks: parse a question into a symbolic
expression, map the expression to a tree of typed neural modules,
execute the tree end-to-end. Where Ha 2016's hypernetwork pushes
context into continuous weight generation, NMN pushes context into
discrete module composition. The two papers anchor opposite ends of
the conditioning spectrum (continuous vs. discrete) that the corpus
explores.

### 2017 — The three normalization-conditioning papers converge

A single year produces the three papers that together establish the
"normalize, then condition the affine" pattern.

**Dumoulin et al. 2017** ([review](reviews/dumoulin_2017_cond_instance_norm.md))
shows that 32 painting styles fit in one network with only 0.2%
style-specific parameters — those parameters being the per-style
(γ, β) rows of a `N × C` lookup table sitting on top of instance
normalization (CIN).

**De Vries et al. 2017** (preserved in [`archive/film_conditional_modulation_review.md` §2](archive/film_conditional_modulation_review.md))
makes the same move on batch normalization: residual-offset
conditional batch normalization (CBN) reparametrizes ResNet-50's BN
affines from a language LSTM, on neuroscience grounds (top-down
modulation reaching early visual cortex). The MODERN architecture
froze the convolutional weights and let CBN do all the work — a
template that BC-Z later inherits.

**Huang & Belongie 2017** ([review](reviews/huang_belongie_2017_adain.md))
removes the learned (γ, β) lookup entirely: compute them on the fly as
the per-channel mean and standard deviation of a style image's deep
features (AdaIN). Zero learnable parameters in the modulation layer;
arbitrary styles at test time.

Also in 2017:

**Vaswani et al. 2017** ([review](reviews/vaswani_2017_attention.md))
introduces the Transformer — attention-only sequence transduction.
Attention will become FiLM's main *competitor* conditioning primitive
once the field starts asking "channel modulation or position
aggregation?".

**Shazeer et al. 2017** ([review](reviews/shazeer_2017_sparse_moe.md))
publishes Sparsely-Gated Mixture-of-Experts, the load-balanced
discrete-routing answer to "how do you scale a model without scaling
per-example compute?". MoE is FiLM's *sparse-discrete* cousin in the
"conditional computation" family.

**Krueger et al. 2017** ([review](reviews/krueger_2017_bayesian_hypernets.md))
lifts Ha's hypernetwork to a Bayesian one by replacing the
deterministic generator with a normalizing flow. The change-of-variables
formula gives exact log-density on the implicit posterior.

**Kendall & Gal 2017** ([review](reviews/kendall_gal_2017_uncertainties.md))
publishes the heteroscedastic regression loss + MC-dropout combination
that the project later adopts as its *anchor uncertainty formula*. No
direct FiLM connection yet, but the loss formula
$L = \tfrac{1}{2} \exp(-s) \|y - \hat y\|^2 + \tfrac{1}{2} s$ will later
be combined with FiLM-Ensemble (Turkoglu 2022) to give a unified
aleatoric / epistemic decomposition.

### 2018 — FiLM is named

**Perez et al. 2018** ([review](reviews/perez_2018_film.md))
unifies CIN, CBN, AdaIN, dynamic-layer-norm, and squeeze-and-excitation
gating under one abstraction: **Feature-wise Linear Modulation**. Two
parameters per channel per layer, applied as `y = γ·x + β`, decoupled
from normalization. The paper does three things at once:

1. Halves CLEVR state-of-the-art error (from 4.5% to 2.3%) using only
   FiLM, no program-level supervision, no module networks, no
   compositional architecture.
2. Demonstrates that the (γ, β) parameter space is *meaningfully
   linearly composable* — adding (γ, β) of "cyan sphere" and "brown
   cube" minus "brown sphere" gives the (γ, β) of "cyan cube"
   zero-shot.
3. Decouples FiLM from normalization via ablation: moving FiLM after
   the post-norm ReLU, or removing batch norm entirely, both leave
   FiLM still working.

Also in 2018:

**Santurkar et al. 2018** ([review](reviews/santurkar_2018_batchnorm_optimization.md))
explains *why* the normalization step underlying CIN, CBN, AdaIN, and
optionally FiLM trains well at all — it's loss-landscape smoothing,
not internal-covariate-shift reduction. This is the citation that
later FiLM-on-normalization architectures rely on for their
optimization rationale.

### 2019 — FiLM moves to sequence and meta-learning

**Birnbaum et al. 2019** ([review](reviews/birnbaum_2019_temporal_film.md))
extends FiLM along the time axis: TFiLM applies FiLM
block-piecewise-constant in time, with an LSTM over block-pooled
self-activations producing per-block (γ, β). Applied to audio
super-resolution, text classification, and ChIP-seq.

Around this time (cited in [`abdollahzadeh_2021_multimodal_meta.md`](reviews/abdollahzadeh_2021_multimodal_meta.md)
and [`turkoglu_2022_film_ensemble.md`](reviews/turkoglu_2022_film_ensemble.md)),
**Vuorio et al. 2019** (MMAML) introduces FiLM as a multimodal
meta-learning conditioner, and reports the empirical observation
"FiLM outperforms attention-based modulation in this context, and is
more stable". This is the corpus's main FiLM-vs-attention empirical
data point.

### 2020 — Theory and modularity

**Galanti & Wolf 2020** ([review](reviews/galanti_wolf_2020_hypernet_modularity.md))
proves the modularity of hypernetworks: the primary network's
parameter count scales as $O(\epsilon^{-m_1/r})$ for hypernet vs.
$\Omega(\epsilon^{-(m_1+m_2)/r})$ for embedding-concatenation, where
$m_2$ is the conditioning dimension. Capacity is absorbed by the
hypernetwork (the conditioner / router) — the formal justification for
why FiLM, MoE, NMN, and conditional batch norm all work better than
"just concatenate the side signal".

### 2021–2022 — FiLM enters policy networks and ensembles

**Takeda et al. 2021** ([review](reviews/takeda_2021_multi_task_feature_mod.md))
demonstrates that 6 heterogeneous image-to-image tasks coexist on one
IN+FiLM-modulated backbone with <2% task-specific parameters, and that
the conditional vector smoothly interpolates over the task-mixture
space *if and only if* the network is trained on synthesized mixed-task
ground-truth (Method 3). Method 1 (train singles, expect mixtures) and
Method 2 (summed loss) both fail.

**Abdollahzadeh et al. 2021** ([review](reviews/abdollahzadeh_2021_multimodal_meta.md))
proves FiLM ≡ uniform scalar rescaling of a conv kernel and proposes
Kernel Modulation (KML), a per-weight generalization with an
outer-product factorization to keep the generator small. KML beats
FiLM by ~5pp on 5-mode few-shot. Also imports the *transference*
metric from multi-task learning into meta-learning.

**Jang et al. 2022** ([review](reviews/jang_2022_bcz.md)) publishes
BC-Z: zero-shot robotic generalization to 24 new manipulation tasks
via FiLM-conditioned ResNet-18 visuomotor policy with a frozen
pretrained sentence encoder. This is the canonical "FiLM-on-policy"
demonstration — the architectural template the project closely
resembles.

**Turkoglu et al. 2022** ([review](reviews/turkoglu_2022_film_ensemble.md))
makes the bridge between FiLM and Lakshminarayanan-style deep
ensembles: re-purpose FiLM γ/β as the source of diversity in an
*implicit* ensemble. Per-member γ/β tables modulate a shared backbone,
giving $M$ sub-networks for ~1.3% parameter overhead instead of
1500%. Diversity is *higher* than the explicit deep ensemble at
$M=16$ on CIFAR-10.

### 2023 — FiLM as a gradient-landscape primitive and as a contrastive fusion gadget

**Moon et al. 2023** ([review](reviews/moon_2023_hierarchical_achievements.md))
uses FiLM in a small role — fusing a discrete action vector into a
CNN state embedding inside an auxiliary contrastive-prediction head
for hierarchical-achievement RL. Crafter SOTA at 9M parameters vs.
DreamerV3's 201M.

**Nikulin et al. 2023** ([review](reviews/nikulin_2023_anti_exploration_rnd.md))
discovers a *new* property of FiLM: gradient-landscape shaping.
Replacing state-action concatenation with FiLM (state generates γ, β
that modulate action features) in an RND prior makes the
anti-exploration bonus's gradient field smooth across the action space
— smooth enough that the offline-RL actor can follow it to dataset-like
actions. SAC-RND matches Q-ensemble SOTA on D4RL without ensembles.

**Gawlikowski et al. 2023** ([review](reviews/gawlikowski_2023_uncertainty_survey.md))
publishes the 77-page survey that fits Kendall & Gal 2017, Turkoglu
2022, and the rest of the uncertainty literature into a 4-branch
taxonomy. The canonical citation for the aleatoric / epistemic split
as field-standard.

### 2025 — Tabular and domain-generalization deployment

**Yan & Guo 2025** ([review](reviews/yan_guo_2025_context_aware_dg.md))
proposes CASA / CaFiLM: a 6-parameter shared FiLM module that uses
mini-batch feature mean as "domain context" to adapt a frozen feature
extractor to unseen target domains. SOTA on DomainBed.

**Wisnu et al. 2025** ([review](reviews/wisnu_2025_stsm_film.md))
publishes STSM-FiLM: FiLM with a scalar speed-factor α ∈ ℝ as
conditioner, with the `(1 + γ)` identity-init trick. Speech
time-scale modification, beats WSOLA — its own training target — on
human-rated MOS.

**Gorishniy et al. 2025** ([review](reviews/gorishniy_2025_tabm.md))
publishes TabM: BatchEnsemble-based implicit ensembling on tabular
MLPs. Same multiplicative-gating bias as FiLM, modulation at linear
layers instead of normalization layers. Pareto-dominant on
46-dataset tabular DL benchmark. The authors explicitly compare to
FiLM-Ensemble in their appendix and prefer BatchEnsemble only because
tabular MLPs lack normalization layers.

---

## 4. Cross-cluster connections matrix

Rows: source cluster. Columns: target cluster. Cells: bridging papers
and what crosses between them. A blank cell means no bridging paper
appears in this corpus (which does *not* imply no connection in the
broader literature).

|   | A FiLM core | B Norm-cond. | C Hypernet | D MoE/NMN | E Attention | F Uncertainty | G Apps |
|---|---|---|---|---|---|---|---|
| **A FiLM core** | — | Perez 2018 cites CIN, CBN, AdaIN as unified family | Perez 2018 §2.2: FiLM is hypernetwork special case | Compositional FiLM = NMN + FiLM (open) | Vuorio 2019: FiLM > attention in MAML | Turkoglu 2022 builds on FiLM | BC-Z, Moon, Nikulin all deploy FiLM |
| **B Norm-cond.** | CIN/CBN/AdaIN → FiLM (decoupling) | — | CIN/CBN as restricted hypernet | — | — | Santurkar 2018 justifies FiLM-on-norm training | CASA's CaFiLM = FiLM with batch-mean conditioner |
| **C Hypernet** | Ha 2016 HyperRNN ≡ FiLM at per-row | Galanti & Wolf modularity covers norm-cond. | — | MoE = discrete hypernet; NMN = symbolic hypernet | — | Krueger 2017 = Bayesian hypernet → "Bayesian FiLM" (open) | Hypernet conditioning underlies BC-Z's task embedding |
| **D MoE/NMN** | NMN's `find` ≈ single FiLM layer | — | MoE/NMN as discrete hypernet | — | — | — | — |
| **E Attention** | Vuorio 2019: FiLM > attention | — | — | — | — | — | — |
| **F Uncertainty** | Turkoglu 2022 instantiates FiLM-Ensemble | Turkoglu 2022 depends on BN substrate | TabM = FiLM-adjacent on linear layers | — | — | — | Kendall-Gal anchor loss used across applied cluster |
| **G Apps** | BC-Z, Moon, Nikulin, CASA all FiLM-based | CASA conditioner = batch feature mean | KML = richer hypernet than FiLM | — | — | KML imports transference from MTL | — |

A few notes on the empty cells.

- *D ↔ E (MoE/NMN ↔ Attention):* The corpus has no direct
  empirical comparison; the broader literature does (Switch
  Transformers, etc.) but those papers are not in this corpus.
- *G ↔ F (Applications ↔ Uncertainty):* The applied papers (BC-Z,
  Moon, Nikulin, CASA) use FiLM for steering and gradient-shaping,
  not for uncertainty quantification. Bridging FiLM-Ensemble
  (Cluster F) into the applied settings is an open direction.

The matrix's most important takeaway is the **A column**: every other
cluster has at least one paper that connects back to FiLM core. The
corpus is shaped as a hub-and-spoke around Perez 2018, with the spokes
being the precursors (Cluster B), the super-class (Cluster C), the
alternatives (Clusters D and E), the probabilistic extensions
(Cluster F), and the application surface (Cluster G).

---

## 5. Open questions for this project

The project is building an interoceptive RL agent with a "modulator"
head that influences perception and action precision via FiLM γ/β
gating (see the NMN-as-hyperparameter direction memo and project
plan). The project's central uncertainty equation is Kendall & Gal
2017's heteroscedastic loss
$L = \tfrac{1}{2} \exp(-s) \|y - \hat y\|^2 + \tfrac{1}{2} s$. The
project has parallel interest in neuromodulator-inspired
architectures (see the sister corpus at
[`docs/project/references/neuromodulatory_algorithms/`](../neuromodulatory_algorithms/)).
Six open questions the FiLM corpus suggests for the project.

**Question 1. When does FiLM beat a hypernetwork?** Galanti & Wolf
2020 prove unrestricted hypernetworks have an exponential
parameter-efficiency advantage over embedding-concatenation methods.
FiLM is a *restricted* hypernetwork (output is per-channel affine
only). Whether FiLM inherits the same advantage in the project's
recurrent-PPO setting is not directly answered by the corpus.
*Concrete experiment:* compare FiLM-conditioned policy against (a)
concat-conditioned policy and (b) full-hypernetwork-conditioned
policy on the same task suite, holding total parameter count fixed.
Hand-off: `experiment-designer` and `senior-developer`.

**Question 2. When does FiLM-Ensemble beat heteroscedastic-loss-only
training?** The project currently has a heteroscedastic precision
head (aleatoric only). Turkoglu 2022 shows FiLM-Ensemble provides a
cheap epistemic-uncertainty channel on top — ~1.3% parameter overhead,
parallelizable on one GPU. The corpus does not directly answer when
the added epistemic estimate is worth the extra $M$-fold inference
cost. *Concrete experiment:* on the project's gridworld with novel
versus familiar pain regimes, measure whether FiLM-Ensemble's epistemic
variance correlates with OOD performance better than the
heteroscedastic head's aleatoric variance does. Hand-off:
`experiment-designer` and `professor-rl-bayesian-dl`.

**Question 3. Is attention a better conditioning primitive than FiLM
for the project's recurrent setting?** Vuorio 2019's finding — "FiLM
outperforms attention-based modulation in this context, and is more
stable" — is from few-shot meta-learning, not RL. The Transformer
literature would suggest attention should win when the conditioning
signal is itself a sequence (a history of past observations, a series
of physiology readings). The project's modulator-as-context-vector
setup is currently a single low-dimensional vector, so FiLM is
plausibly the right choice — but the corpus does not have a head-to-
head FiLM-vs-attention comparison in an RL recurrent setting.
*Concrete experiment:* replace FiLM γ/β at the recurrent head with
single-head cross-attention from the modulator state to the
recurrent hidden state. Hand-off: `professor-rl-bayesian-dl`.

**Question 4. How does the project's modulator-as-context-vector setup
relate to BC-Z's language-FiLM-on-policy setup?** BC-Z's architecture
is the closest published template to what the project is building: a
low-dimensional embedding (in BC-Z, 512 from a frozen sentence
encoder) injected via FiLM at every block of a ResNet visuomotor
policy. Two important asymmetries: (a) BC-Z's embedding is *static*
within an episode (the task command doesn't change), while the
project's modulator state can *change* within an episode in response
to interoceptive feedback; (b) BC-Z's controller is a feed-forward
ResNet, while the project's controller is recurrent (PPO with GRU /
LSTM hidden state). *Concrete reading:* the [BC-Z review's §2.3](reviews/jang_2022_bcz.md)
gives the exact (γ, β) computation that would need to be adapted for
within-episode modulator dynamics, and the Temporal FiLM mechanism
([Birnbaum 2019 §3](reviews/birnbaum_2019_temporal_film.md)) is the
closest precedent for time-varying γ/β. Hand-off: `senior-developer`
for design plan.

**Question 5. When does per-channel FiLM become too narrow, requiring
Kernel Modulation (KML) or a fuller hypernetwork?**
Abdollahzadeh 2021's re-interpretation lemma is sharp: FiLM ≡
uniform scalar rescaling of a conv kernel, leaving one degree of
freedom per channel per task. For the project's setting — a
low-dimensional modulator vector representing physiology — that
should be plenty. But if the project later expands the modulator state
to include several heterogeneous components (separate channels for
ACh, NE, DA, 5-HT, opioid; see the neuromodulation sister corpus),
KML's per-weight modulation may become the right architecture. *Concrete
diagnostic:* if a FiLM-modulated policy's training plateaus and
ablation shows the (γ, β) variance is saturated, swap in KML.
Hand-off: `professor-neuromodulation` for the modulator-state design,
`senior-developer` for the architectural plan.

**Question 6. How does FiLM's "gradient-landscape-smoothing" property
(Nikulin 2023) interact with the project's PPO inner loop?** Nikulin
2023's finding is that FiLM produces a smoother bonus surface than
concatenation in an RND prior — smooth enough that the actor can
follow gradients to the global minimum across the action space. The
project's PPO actor also descends a gradient surface; whether
FiLM-modulated value heads / advantage estimators have analogous
smoothing properties is an *unexplored question* in the corpus. The
project's null-result diagnoses ([NMN_PERFORMANCE_DIAGNOSIS_v1–v8](../../develop/INDEX.md))
are precisely the kind of "the policy can't follow the gradient
field" failure mode that Nikulin's mechanism would address.
*Concrete experiment:* a FiLM ablation against concat-conditioning
inside the value head, with the v8 diagnosis suite as the testbed.
Hand-off: `senior-developer`.

---

## Cross-references

- This document's index: [`film_lit_review.md`](film_lit_review.md).
- Per-paper reviews: [`reviews/`](reviews/).
- Source PDFs (read-only): [`sources/`](sources/).
- Preserved 5-paper master review (De Vries 2017 reference):
  [`archive/film_conditional_modulation_review.md`](archive/film_conditional_modulation_review.md).
- Sister corpus on neuromodulation:
  [`../neuromodulatory_algorithms/`](../neuromodulatory_algorithms/).
- Sister corpus on uncertainty:
  [`../uncertainty/`](../uncertainty/).
- Project plan: [`docs/project/project_plan.md`](../../project_plan.md).
- Project's null-result diagnosis series:
  [`docs/develop/INDEX.md`](../../../develop/INDEX.md).
