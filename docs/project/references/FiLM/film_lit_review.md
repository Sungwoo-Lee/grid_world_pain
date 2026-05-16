---
title: "FiLM Corpus — Master Table of Contents"
topic: FiLM
status: curated
last_updated: 2026-05-16
related:
  - film_synthesis.md
  - archive/film_conditional_modulation_review.md
scope: |
  Master index over 23 per-paper reviews in `reviews/` covering the FiLM
  (Feature-wise Linear Modulation) family, its precursors (hypernetworks,
  conditional normalization), its uncertainty / ensemble extensions, and
  its applications in multi-task learning, RL, robotics, and domain
  generalization. The thematic synthesis lives in `film_synthesis.md`.
---

# FiLM Corpus — Master Table of Contents

## Plain-English entry point

This corpus organizes 23 papers around a single architectural idea —
**conditioning a neural network's computation on a side signal**. The side
signal can be a natural-language question, a task identifier, a video, a
batch's domain-context summary, a desired playback speed, an ensemble-member
index, or a continuous physiological state. The mechanism that turns that
side signal into actual changes inside the network is the central object of
study. The corpus's namesake mechanism is **FiLM** (Feature-wise Linear
Modulation, Perez et al. 2018): for every channel of an intermediate
feature map, multiply by a learned scale `γ` and add a learned shift `β`,
where `γ` and `β` are produced from the side signal by a small "FiLM
generator" network. Two scalars per channel, applied as `y = γ·x + β`.
That is the whole equation. Around this equation a remarkably wide
literature has accumulated.

Why this corpus exists. The project is building an interoceptive
reinforcement-learning agent whose internal "modulator" head conditions a
policy network — most naturally via FiLM γ/β gating at the sensor encoder
and at recurrent / value heads. To make that design choice with eyes open,
we read across the FiLM lineage: where FiLM came from (HyperNetworks,
Conditional Instance Norm, Adaptive Instance Norm, Conditional Batch
Norm), where it goes (FiLM-Ensemble for uncertainty, Temporal FiLM and
STSM-FiLM for sequence modeling, BC-Z's FiLM-on-policy for robotics, CASA's
domain-context FiLM, Kernel Modulation as a richer alternative), and what
competing or complementary primitives sit alongside it (attention,
mixture-of-experts, neural module networks, BatchEnsemble).

How to read this corpus. Three entry points. *(a)* For a first
orientation, read this index's plain-English summary, pick a cluster from
the table below, and follow the link to its companion `film_synthesis.md`
section. *(b)* For a specific question — "what does FiLM-Ensemble do?" —
go directly to that paper's review in `reviews/`. *(c)* For the
cross-paper narrative — how Ha 2016's hypernetworks anticipated FiLM, how
Kendall & Gal 2017's heteroscedastic loss connected to Turkoglu 2022's
ensemble, how recent applications instantiate the FiLM primitive across
domains — read [`film_synthesis.md`](film_synthesis.md).

## Master Table of Contents

Each entry below links to a per-paper review in [`reviews/`](reviews/).
Papers are grouped by thematic cluster as defined in
[`film_synthesis.md`](film_synthesis.md).

### Cluster A — FiLM core and feature-wise affine modulation

The papers that name and extend FiLM along its core "per-channel scale and
shift" axis. Cross-cluster synthesis: [`film_synthesis.md` §2 Cluster A](film_synthesis.md#cluster-a--film-core-and-feature-wise-affine-modulation).

| Paper | One-line description |
|---|---|
| [Perez et al. 2018 — FiLM](reviews/perez_2018_film.md) | The paper that names FiLM and shows it halves prior CLEVR state-of-the-art error using only per-channel γ, β modulation. |
| [Birnbaum et al. 2019 — Temporal FiLM](reviews/birnbaum_2019_temporal_film.md) | Extends FiLM along the time axis: an LSTM over block-pooled self-activations produces per-block γ, β for 1-D conv networks; applied to audio super-resolution and ChIP-seq. |
| [Wisnu et al. 2025 — STSM-FiLM](reviews/wisnu_2025_stsm_film.md) | Applies FiLM with a scalar speed-factor conditioner (α ∈ ℝ) for time-scale modification of speech; uses the `(1 + γ)` stability trick at init. |
| [Takeda et al. 2021 — Multi-task feature modulation](reviews/takeda_2021_multi_task_feature_mod.md) | Shows IN+FiLM at every layer can host 6 heterogeneous image-translation tasks (segmentation, denoising, stylization) on one backbone with <2% task-specific parameters, and that the conditional vector smoothly interpolates between trained behaviors. |

### Cluster B — Normalization-conditioning lineage

The three "normalization + conditional affine" papers from which FiLM
emerged as the general abstraction, plus Santurkar's analysis of why
normalization helps optimization in the first place. Cross-cluster synthesis:
[`film_synthesis.md` §2 Cluster B](film_synthesis.md#cluster-b--normalization-conditioning-lineage).

| Paper | One-line description |
|---|---|
| [Dumoulin et al. 2017 — Conditional Instance Normalization (CIN)](reviews/dumoulin_2017_cond_instance_norm.md) | A single network captures 32 painting styles via an `N × C` lookup table of per-style (γ, β); only 0.2% of parameters are style-specific. The direct FiLM ancestor. |
| [Huang & Belongie 2017 — AdaIN](reviews/huang_belongie_2017_adain.md) | Computes (γ, β) on the fly as the per-channel mean and std of a style image's deep features — zero learnable parameters in the modulation layer, arbitrary styles at test time. |
| [Santurkar et al. 2018 — How does BatchNorm help optimization?](reviews/santurkar_2018_batchnorm_optimization.md) | Demonstrates that BN's benefit is *loss-landscape smoothing*, not internal covariate shift. The canonical reference for why FiLM-on-top-of-normalization trains well. |
| De Vries et al. 2017 — Conditional Batch Normalization (CBN) / MODERN | *No fresh per-paper review*; covered in [`archive/film_conditional_modulation_review.md` §2](archive/film_conditional_modulation_review.md). Introduces residual-offset CBN that modulates ResNet-50's BN affines from a language LSTM. Direct FiLM ancestor and biological-motivation precursor (top-down modulation reaching early visual cortex). |

### Cluster C — Hypernetworks

Hypernetworks — one network generating another's weights — are FiLM's
super-class: FiLM is a hypernetwork restricted to outputting per-channel
affine parameters. Cross-cluster synthesis:
[`film_synthesis.md` §2 Cluster C](film_synthesis.md#cluster-c--hypernetworks).

| Paper | One-line description |
|---|---|
| [Ha, Dai & Le 2016 — HyperNetworks](reviews/ha_2016_hypernetworks.md) | Introduces static and dynamic hypernetworks; HyperRNN's row-scaling trick is structurally identical to FiLM at the per-row granularity. |
| [Galanti & Wolf 2020 — On the modularity of hypernetworks](reviews/galanti_wolf_2020_hypernet_modularity.md) | Proves hypernetworks have an *exponential parameter-efficiency advantage* over embedding-concatenation methods: hypernet primary network scales as `O(ε^(−m₁/r))` instead of `O(ε^(−(m₁+m₂)/r))`. |
| [Krueger et al. 2017 — Bayesian Hypernetworks](reviews/krueger_2017_bayesian_hypernets.md) | Lifts the deterministic hypernetwork to a Bayesian one: a normalizing flow h(ε) produces samples of the primary network's weights with exact log-density via change of variables. |

### Cluster D — Modular reasoning and mixture of experts

Discrete-composition alternatives to FiLM's continuous modulation: assemble
a typed-module tree per question (NMN), or sparsely route through a bank of
experts (MoE). Cross-cluster synthesis:
[`film_synthesis.md` §2 Cluster D](film_synthesis.md#cluster-d--modular-reasoning-and-mixture-of-experts).

| Paper | One-line description |
|---|---|
| [Andreas et al. 2016 — Neural Module Networks (NMN)](reviews/andreas_2016_neural_module_networks.md) | Parses each question into a symbolic expression, assembles a typed tree of `find / transform / combine / describe / measure` modules, trains end-to-end. Compositional generalization to longer questions than seen at training. |
| [Hu et al. 2017 — End-to-End Module Networks (N2NMN)](reviews/hu_2017_e2e_module_networks.md) | Replaces NMN's off-the-shelf parser with a *learned* layout policy trained by REINFORCE + behavioral cloning; 83.7% on CLEVR. |
| [Shazeer et al. 2017 — Sparsely-Gated Mixture-of-Experts](reviews/shazeer_2017_sparse_moe.md) | Replaces a dense FFN with a bank of `n` experts plus a noisy top-`k` gate; importance + load auxiliary losses balance utilization. Trillion-parameter models become tractable. |

### Cluster E — Attention as conditioning

Attention is a *different* conditioning primitive than FiLM: it modulates
*positions* by a softmax-weighted aggregation across a source stream,
where FiLM modulates *channels* by per-channel affine. Cross-cluster
synthesis: [`film_synthesis.md` §2 Cluster E](film_synthesis.md#cluster-e--attention-as-a-parallel-conditioning-primitive).

| Paper | One-line description |
|---|---|
| [Vaswani et al. 2017 — Attention Is All You Need](reviews/vaswani_2017_attention.md) | Introduces the Transformer: scaled dot-product attention + multi-head + sinusoidal positional encoding. The alternative conditioning primitive that FiLM is most often compared to. |

### Cluster F — Probabilistic / uncertainty / ensemble FiLM

How FiLM connects to Bayesian deep learning, ensembles, and uncertainty
quantification — the line of work directly relevant to the project's
heteroscedastic precision head and any downstream epistemic-uncertainty
measurement. Cross-cluster synthesis:
[`film_synthesis.md` §2 Cluster F](film_synthesis.md#cluster-f--probabilistic--uncertainty--ensemble-film).

| Paper | One-line description |
|---|---|
| [Kendall & Gal 2017 — What uncertainties do we need?](reviews/kendall_gal_2017_uncertainties.md) | The project's anchor heteroscedastic-regression loss: `L = ½·exp(−s)·‖y − ŷ‖² + ½·s` with `s = log σ²`. Combines learned aleatoric noise with MC-dropout epistemic uncertainty. |
| [Gawlikowski et al. 2023 — A survey of uncertainty in deep neural networks](reviews/gawlikowski_2023_uncertainty_survey.md) | The 77-page map of the territory: 4-branch taxonomy (single deterministic, BNN, ensemble, test-time augmentation), measures, calibration, applications. The canonical citation for the aleatoric/epistemic split as field-standard. |
| [Turkoglu et al. 2022 — FiLM-Ensemble](reviews/turkoglu_2022_film_ensemble.md) | Re-purposes FiLM γ/β as the source of diversity in an *implicit* ensemble: per-member γ/β tables modulate a shared backbone, giving M sub-networks for ~1.3% parameter overhead. |
| [Gorishniy et al. 2025 — TabM](reviews/gorishniy_2025_tabm.md) | BatchEnsemble-based implicit ensembling on tabular MLPs (FiLM-adjacent: same multiplicative-gating inductive bias, modulation around linear layers instead of normalization layers). Pareto-dominant on 46-dataset tabular benchmark. |

### Cluster G — Applications

Multi-task, meta-learning, RL, robotics, and domain-generalization
applications of FiLM-style conditioning. Cross-cluster synthesis:
[`film_synthesis.md` §2 Cluster G](film_synthesis.md#cluster-g--applications-multi-task-meta-learning-rl-robotics-domain-generalization).

| Paper | One-line description |
|---|---|
| [Abdollahzadeh et al. 2021 — Multimodal meta-learning via Kernel Modulation (KML)](reviews/abdollahzadeh_2021_multimodal_meta.md) | Proves FiLM is equivalent to *uniform scalar rescaling of a conv kernel*, and proposes Kernel Modulation (KML) as a richer per-weight alternative. Imports the *transference* metric from multi-task learning. |
| [Jang et al. 2022 — BC-Z](reviews/jang_2022_bcz.md) | The canonical "FiLM-on-policy" demonstration: zero-shot robotic generalization to 24 new manipulation tasks via FiLM-conditioned ResNet-18 visuomotor policy and a frozen pretrained sentence encoder. |
| [Moon et al. 2023 — Hierarchical achievements via contrastive learning](reviews/moon_2023_hierarchical_achievements.md) | Uses FiLM as a *small fusion gadget* (action → state-embedding) inside a contrastive achievement-prediction head; SOTA on Crafter at 9M parameters vs. DreamerV3 at 201M. |
| [Nikulin et al. 2023 — Anti-Exploration by RND (SAC-RND)](reviews/nikulin_2023_anti_exploration_rnd.md) | Shows FiLM has a *gradient-landscape-shaping* property: replacing concat with FiLM in an RND prior yields smooth bonus gradients that the actor can follow. Matches Q-ensemble SOTA on D4RL without ensembles. |
| [Yan & Guo 2025 — CASA: Context-Aware Self-Adaptation (CaFiLM)](reviews/yan_guo_2025_context_aware_dg.md) | FiLM as a domain-generalization conditioner: a 6-parameter shared module adapts features using mini-batch feature mean as "domain context". SOTA on DomainBed (5 datasets, avg 68.8%). |

## Previously superseded master review

A 5-paper master review predates this curation:
[`archive/film_conditional_modulation_review.md`](archive/film_conditional_modulation_review.md)
(covering Perez 2018, De Vries 2017, Dumoulin 2017, Ha 2016, Turkoglu 2022).
It is superseded for the four papers that have fresh per-paper reviews
in `reviews/`, but the **De Vries et al. 2017 CBN / MODERN** entry has no
source PDF in the new corpus and therefore no fresh per-paper review.
For De Vries 2017, the archived review's §2 remains the authoritative
project reference and is cited inline in
[`film_synthesis.md`](film_synthesis.md) Cluster B and the historical
timeline (2017 window).

## Pointer to the cross-paper narrative

For the cross-paper narrative — thematic synthesis, historical timeline,
disagreement axes, the open questions this corpus poses for the project —
see [`film_synthesis.md`](film_synthesis.md). That document is the
reader-facing thematic synthesis; this index is the navigational table of
contents.
