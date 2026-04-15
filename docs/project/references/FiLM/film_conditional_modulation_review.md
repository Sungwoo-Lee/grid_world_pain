---
title: "Reference Review — FiLM & Conditional Modulation under Noise (Core Papers)"
status: DRAFT v1 — all 5 papers reviewed, synthesis complete
last_updated: 2026-04-15
target_depth: graduate-level — derivations in LaTeX
source: NotebookLM notebook 822f0187 + arXiv cross-reference
related:
  - ../../perceptual_noise_lit_review.md (§3, SQ2)
  - ../uncertainty/uncertainty_reference_review.md
  - ../../../develop/FiLM_ENSEMBLE_SENSORY_PRECISION.md
scope: |
  Per-paper review of the five core Section 7.2 references in
  perceptual_noise_lit_review.md — the papers that define the FiLM /
  conditional-modulation mechanism and its probabilistic extension used by
  the GridWorld Pain Phase 3 design. Each entry has a fixed 3-layer structure:
  Layer 1 (undergraduate) — basic introduction and concept; Layer 2 (graduate
  summary) — main results and algorithm; Layer 3 (graduate deep dive) —
  mathematical formulation with all critical equations in LaTeX and
  line-by-line derivations. Reviewed one paper at a time with a user
  checkpoint after each entry.
---

# Reference Review — FiLM & Conditional Modulation under Noise

> **How to read.** Each paper's entry has a fixed 3-layer structure. Skim
> Layer 1 for orientation, Layer 2 for the practical take-away, and Layer 3
> when you need to reimplement the method or understand why it behaves the
> way it does. Equation numbers in Layer 3 follow the original paper where
> possible; `(L.x)` marks derivation steps not present in the paper.

## Status Dashboard

| # | Paper (year) | Status | Core contribution (one line) |
|---|--------------|--------|------------------------------|
| 1 | Perez et al. (2018) — **FiLM** | DONE | Feature-wise affine modulation as a general conditioning primitive |
| 2 | De Vries et al. (2017) — **CBN / MODERN** | DONE | Conditional Batch Normalization: modulating vision by language |
| 3 | Dumoulin et al. (2017) — **Cond. Instance Norm** | DONE | A single network encodes many styles via conditional affine IN |
| 4 | Ha et al. (2017) — **HyperNetworks** | DONE | One network generates the weights of another — FiLM's super-class |
| 5 | Turkoglu et al. (2022) — **FiLM-Ensemble** | DONE | Implicit deep ensemble via per-member FiLM; shared backbone, cross-entropy training |

## Table of Contents

- [Notation](#notation-used-throughout)
- [1. Perez et al. (2018) — FiLM](#1-perez-et-al-2018--film-visual-reasoning-with-a-general-conditioning-layer)
- [2. De Vries et al. (2017) — Modulating Early Visual Processing by Language (CBN / MODERN)](#2-de-vries-et-al-2017--modulating-early-visual-processing-by-language-cbn--modern)
- [3. Dumoulin et al. (2017) — A Learned Representation for Artistic Style (CIN)](#3-dumoulin-et-al-2017--a-learned-representation-for-artistic-style-cin)
- [4. Ha, Dai & Le (2017) — HyperNetworks](#4-ha-dai--le-2017--hypernetworks)
- [5. Turkoglu et al. (2022) — FiLM-Ensemble](#5-turkoglu-et-al-2022--film-ensemble-probabilistic-deep-learning-via-feature-wise-linear-modulation)
- [Cross-Paper Synthesis](#cross-paper-synthesis)

## Notation (used throughout)

Common symbols reused across papers to keep LaTeX consistent:

- $x \in \mathcal{X}$ — input sample; $y$ — target.
- $F \in \mathbb{R}^{B \times C \times H \times W}$ — a feature tensor with batch $B$, channels $C$, spatial size $H \times W$. $F_{i,c}$ denotes sample $i$, channel $c$.
- $z \in \mathbb{R}^{d_z}$ — conditioning vector (e.g., question embedding).
- $\gamma_{i,c}, \beta_{i,c} \in \mathbb{R}$ — per-sample, per-channel FiLM scale and shift.
- $f_\theta, h_\phi$ — FiLM generator sub-networks mapping $z \mapsto (\gamma, \beta)$.
- $\mu_c, \sigma^2_c$ — batch statistics for channel $c$ (used by BN / CBN / CIN).
- $\theta$ — parameters of the primary (modulated) network; $\phi$ — parameters of the conditioner / hypernetwork.

---

## 1. Perez et al. (2018) — *FiLM: Visual Reasoning with a General Conditioning Layer*

**Venue:** AAAI 2018 (arXiv:1709.07871). **Domain:** Visual question answering
on CLEVR (synthetic compositional reasoning) and CLEVR-Humans / CoGenT
generalisation splits. **Why it anchors SQ2:** this is the paper that names
FiLM, shows it as a *general* conditioning primitive (not just a cond-BN
variant), and reports the empirical behaviours — feature suppression,
upregulation, spatial localisation — that make it the canonical target for
any "precision-as-gain" architectural story.

### Layer 1 — Basic introduction and concept

Standard deep nets struggle with *structured, multi-step visual reasoning* —
answering a compositional question like "what colour is the cube to the left
of the small red sphere?" requires the visual network to behave differently
depending on the question, not just to extract a fixed feature vector. The
failure mode is well known: big VQA models exploit language priors and
dataset statistics rather than learning the underlying logic.

The authors' move is to let the *question* — encoded by an RNN into a vector
$z$ — directly reshape the *visual* network's intermediate features. The
reshape is deliberately minimal: for each channel $c$ of an intermediate
feature map $F_{i,c}$, FiLM applies an affine transform
$\gamma_{i,c} F_{i,c} + \beta_{i,c}$ where $(\gamma_{i,c}, \beta_{i,c})$ are
produced by a small network reading $z$. Two parameters per channel per
layer; no dependence on spatial resolution. Cheap, general, and — critically
for the project — *the same mathematical object* that Active Inference calls
per-channel precision ($\gamma$ as a gain on a feature).

Empirically the paper shows FiLM does something stronger than mere
task-conditioning: it learns to *suppress*, *upregulate*, and *zero-out*
feature maps depending on the question; and it spatially localises the
referenced object without any localisation supervision. That qualitative
behaviour — selective attenuation driven by a side-channel — is what makes
FiLM the natural candidate for a precision gate.

### Layer 2 — Main results and algorithm

**Algorithm.** A FiLM layer takes a feature tensor $F$ and a conditioning
vector $z$, and outputs $\text{FiLM}(F \mid \gamma, \beta) = \gamma \odot F + \beta$,
broadcast per channel. The FiLM *generator* (a GRU over the tokenised
question, 4096 hidden units, 200-d word embeddings) produces one
$(\gamma^n, \beta^n)$ vector per residual block $n$ from affine projections
of its final hidden state. The *FiLM-ed network* is a small CNN: image
features ($128 \times 14 \times 14$, either trained from scratch or extracted
from a frozen ResNet-101), concatenated with two coordinate channels to
support spatial reasoning, processed by **4 FiLM-ed ResBlocks**, then a
classifier head. Inside each ResBlock the FiLM layer is inserted after a
batch-norm *whose affine parameters are turned off* — FiLM replaces them.

**Key results on CLEVR (test accuracy, raw-pixels / ResNet features):**

| Sub-task | From raw pixels | From features |
|----------|:---------------:|:-------------:|
| Overall | **97.6%** | **97.7%** |
| Count | 94.3% | 94.3% |
| Exist | 99.3% | 99.1% |
| Compare Numbers | 93.4% | 96.8% |
| Query Attribute | 99.3% | 99.1% |
| Compare Attribute | 99.3% | 99.1% |

This **roughly halved the prior state-of-the-art error** (4.5% → 2.3%) among
methods that do not use program-level supervision (e.g., the module-net
families of Andreas et al., 2016; Johnson et al., 2017). Two further
observations matter for downstream SQ2 reasoning:

1. **Architectural robustness.** Ablations — varying depth, moving the FiLM
   layer to different positions inside the ResBlock (including *after* the
   post-normalisation ReLU) — produce models that still exceed prior SOTA.
   This is the authors' central evidence for *decoupling* FiLM from
   normalisation: feature-wise affine modulation is expressive in its own
   right, not an artefact of being fused with batch statistics.
2. **Generalisation.** FiLM is data-efficient on CLEVR-Humans (natural human
   questions) and shows zero-shot compositional generalisation on
   CoGenT — the latter is specifically a test of whether the question-driven
   modulation is genuinely compositional rather than memorised.

**Take-away for the project.** FiLM's own paper establishes (i) the
mathematical form the project imports, (ii) the empirical observation that
$\gamma$ learns to go *negative* and *near-zero* on selected channels — i.e.
the primitive *can* express "suppress this modality" — and (iii) that the
mechanism is not intrinsically tied to normalisation, which matters because
GridWorld Pain's Phase 3 port sits in an RL setting where BN is unusual.

### Layer 3 — Graduate-level deep dive

#### 3.1 The FiLM layer, stated precisely

Let $F \in \mathbb{R}^{B \times C \times H \times W}$ be a feature tensor at
some point in the modulated network and let $x_i$ be the $i$-th sample's
conditioning input (in CLEVR, the tokenised question embedded by the GRU).
FiLM assumes a pair of conditioning functions $f_c, h_c : \mathcal{X} \to \mathbb{R}$
for each channel $c$, and defines

$$
\gamma_{i,c} = f_c(x_i), \qquad \beta_{i,c} = h_c(x_i), \tag{1}
$$

with the modulation itself

$$
\text{FiLM}(F_{i,c} \mid \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c}\, F_{i,c} + \beta_{i,c}. \tag{2}
$$

Equations (1)–(2) are the entire FiLM definition. In practice $f$ and $h$
share almost all their parameters — they are two linear heads on top of a
shared "FiLM generator" $g_\phi$:

$$
\begin{bmatrix}\gamma^n_i\\ \beta^n_i\end{bmatrix} = W^n\, g_\phi(x_i) + b^n,\qquad W^n \in \mathbb{R}^{2C \times d_z},\ b^n \in \mathbb{R}^{2C}, \tag{L.1}
$$

one $W^n$ per modulated block $n$. For CLEVR, $g_\phi$ is a
4096-d GRU over word embeddings and there are $n = 1,\ldots,4$ ResBlocks, so
the full conditioning network adds $4 \cdot 2 \cdot C \cdot 4096$ parameters
(plus the word-embedding + GRU parameters). With $C = 128$ this is
$\approx 4.2\mathrm{M}$ FiLM-specific parameters — cheap relative to the
backbone.

#### 3.2 How FiLM relates to Conditional Batch Normalisation

Conditional Batch Normalisation (Dumoulin et al., 2017; De Vries et al.,
2017 — papers 2–3 of this review) rewrites the standard BN affine
parameters $(\gamma, \beta)$ as *learned functions of a conditioning input*.
The full CBN layer is

$$
\mathrm{CBN}(F_{i,c} \mid z_i) = \gamma_c(z_i)\, \frac{F_{i,c} - \mu_c}{\sqrt{\sigma^2_c + \varepsilon}} + \beta_c(z_i), \tag{3}
$$

where $\mu_c, \sigma^2_c$ are computed over the batch and $\gamma_c, \beta_c$
are functions of $z_i$. Comparing (2) and (3) directly: FiLM is CBN with the
normalisation step *removed*,

$$
\text{FiLM}(F) = \mathrm{CBN}(F)\Big|_{\mu_c = 0,\; \sigma^2_c = 1 - \varepsilon}. \tag{L.2}
$$

Equivalently, CBN = normalise-then-FiLM. Perez et al.'s ablations — moving
the FiLM layer to positions in the ResBlock where no upstream normalisation
exists, with negligible accuracy loss — are the empirical content of the
claim that (L.2) is not load-bearing: the affine part alone is expressive
enough. This matters for the project: the DreamerV3 / PPO backbones in
`FiLM_ENSEMBLE_SENSORY_PRECISION.md` use LayerNorm rather than BN, so the
"FiLM without BN" regime of Perez et al. is exactly the regime relevant
here.

#### 3.3 ResBlock structure with FiLM

Each of the 4 ResBlocks has the schematic

$$
\begin{aligned}
  U &= \mathrm{Conv}_{1\times1}(F_{\text{in}}),\\
  V &= \mathrm{Conv}_{3\times3}(\mathrm{ReLU}(U)),\\
  V' &= \mathrm{BN}_{\text{no-affine}}(V),\\
  V'' &= \text{FiLM}(V' \mid \gamma^n, \beta^n),\\
  F_{\text{out}} &= \mathrm{ReLU}(V'') + F_{\text{in}}.
\end{aligned}
\tag{4}
$$

Two details. First, the BN preceding the FiLM layer has its affine
parameters disabled — FiLM replaces them, so stacking BN's own $(\gamma, \beta)$
with FiLM's would be redundant. Second, the residual add is *outside* the
FiLM layer, which means FiLM cannot directly scale the identity path —
consistent with the "modulate the transform, preserve the skip" pattern also
seen in later style-transfer and diffusion architectures.

#### 3.4 Coordinate feature maps

FiLM is channel-wise: $\gamma_{i,c}$ is a scalar, broadcast uniformly over
$H \times W$. A pure FiLM layer therefore cannot encode spatial instructions
like "focus on the left half". To recover that capacity cheaply, the input
feature map is augmented with two deterministic coordinate channels
$C_x[:,h,w] = (w / W) \cdot 2 - 1$ and $C_y[:,h,w] = (h / H) \cdot 2 - 1$.
After this augmentation, FiLM *can* express spatial selection because the
channels it modulates now carry positional information — the attention-like
localisation observed in the paper's figures is an interaction between
$(\gamma, \beta)$ and these coordinate channels, not a property of the
affine layer alone.

#### 3.5 What $\gamma$ and $\beta$ actually learn (empirical)

Three observations from Perez et al. that matter for the precision-gate
interpretation:

1. **Sign flip.** $\gamma$ takes negative values frequently; the distribution
   is well spread around 0, not concentrated near 1. A precision gate
   interpretation ($\gamma \approx 1/\sigma^2$) would constrain
   $\gamma \ge 0$ — FiLM's generality here exceeds what a strict precision
   story uses, which is relevant when porting to the project's Phase 3.
2. **Near-zero suppression.** Particular question types (e.g., "which
   colour?") produce $\gamma \approx 0$ on whole channels, effectively
   silencing them. This is the "suppression" behaviour invoked by the
   precision-weighting literature, observed here without a variance head.
3. **Conditioning reach.** Moving $(\gamma, \beta)$ into later ResBlocks
   produces qualitatively different error modes — earlier modulation helps
   spatial queries, later helps attribute queries — suggesting that the
   right *layer* of modulation depends on what information needs routing.

For Phase 3 of GridWorld Pain, these behaviours are the empirical
precedent that makes "$\gamma$ encodes a reliability gate" plausible; the
gap (§2 of the parent review) is that in Perez et al. the FiLM parameters
are supervised by a *clear task loss* with $\sim\!10^5$ labelled CLEVR
questions, whereas in the RL setting the only supervision is the sparse
return. This is the setup under which the project's v5–v8 series sees FiLM
collapse to near-identity.

#### 3.6 Connection to the broader conditioning landscape

FiLM sits inside a well-defined hierarchy of conditioning expressiveness:

$$
\underbrace{\text{bias-only}}_{\gamma \equiv 1} \;\subset\; \underbrace{\text{FiLM}}_{\text{eq. (2)}} \;\subset\; \underbrace{\text{hypernet, full } W(z)}_{\text{Ha et al., 2017}} \;\subset\; \underbrace{\text{input-dependent architecture}}_{\text{e.g., MoE gating}}. \tag{L.3}
$$

Each step up expands the modulator's expressivity — and the failure modes
of the lower levels (variance collapse, identity collapse) all appear, in
amplified form, at the higher levels without an uncertainty-aware training
signal. This hierarchy (papers 4 and 5 of this review extend it further) is
the reason FiLM-Ensemble and not FiLM-plus-noise is the target Phase 3
architecture.

---

---

## 2. De Vries et al. (2017) — *Modulating Early Visual Processing by Language* (CBN / MODERN)

**Venue:** NeurIPS 2017 (arXiv:1707.00683). **Domain:** Visual question
answering (VQAv1) and the *GuessWhat?!* object-identification dialogue game
(Oracle sub-task in particular). **Why it belongs in SQ2:** this is the
paper immediately preceding FiLM in the lineage — the one that argues, on
neuroscience grounds, that language should reach *into the visual network*
rather than being concatenated at the head; and that introduces the
Conditional Batch Normalisation (CBN) layer which FiLM (paper 1) then
decouples from normalisation. For the project it is the cleanest statement
of the "top-down modulation of early perception" motif that the precision
story rests on.

> **Source note.** The working NotebookLM notebook contains only summaries
> of this paper, not the full text. Layer 3 equations below are the
> canonical CBN form from the arXiv preprint; where an exact number is not
> pinned by the notebook or the preprint, it is called out explicitly.

### Layer 1 — Basic introduction and concept

Most VQA models at the time of writing used **late fusion**: a CNN encodes
the image into a fixed feature vector, an RNN encodes the question into
another, and a small head combines them to produce an answer. The linguistic
side therefore has no influence on *what the visual features are* — it only
decides how to read them off.

The authors argue this is biologically and computationally backwards.
Top-down signals (attention, task set, linguistic context) are well
documented to modulate even **early** visual cortex in humans and
non-human primates — effects have been reported as far back as V1 — which
suggests that when a task or a question primes the system, it should change
what the early feature extractors emphasise, not just how late features are
pooled. Their engineering expression of this idea is a **conditional
normalisation layer**: insert language into the visual network by making
its batch-normalisation affine parameters a function of the question
embedding, at every residual stage of a deep ResNet.

The architectural instantiation is called **MODERN** (MODulatEd ResNet).
The mechanism (the CBN layer) is what FiLM (paper 1) generalises.

### Layer 2 — Main results and algorithm

**Architecture.** A pre-trained ResNet-50 (ImageNet-trained) is the visual
backbone. All convolutional weights and all batch-norm *running statistics*
are frozen. The only trainable visual-path parameters are the **CBN offsets**
(see Layer 3 for the exact parameterisation), and a small classification
head. A language pipeline — an LSTM producing a question embedding
$z \in \mathbb{R}^{d_z}$ — feeds a small MLP that outputs per-channel
$(\Delta\gamma_c(z), \Delta\beta_c(z))$ for every BN layer being modulated.

**Which layers are modulated.** The paper's core ablation examines
modulating progressively more of the ResNet: only the last stage (stage 4),
then stages 3+4, then stages 2+3+4, then all four stages. The empirical
finding — which is the paper's rhetorical pay-off — is that modulating
*earlier* stages (not just late high-level features) gives a further
performance boost, supporting the biological analogy.

**Empirical results.** Two benchmarks:

- **VQAv1** (open-ended, real images). MODERN is *competitive* with the
  best dedicated VQA systems of its year despite freezing the entire
  visual backbone. The absolute number most often cited in follow-up work
  is that MODERN closes most of the gap between a frozen-ResNet baseline
  and an end-to-end fine-tuned ResNet, at a small fraction of the
  parameter cost.
- **GuessWhat?! Oracle.** The Oracle sub-task — given an image, a target
  object, and a yes/no question, predict the answer — is a cleaner
  language-conditions-vision benchmark. MODERN improves error over the
  prior baseline by a multi-point margin, and the improvement is largest
  when the target object is small and therefore demands early-stage
  spatial modulation.

> Exact numerical entries vary slightly between the arXiv v1/v2 and the
> NeurIPS camera-ready, and the notebook did not pin them. For decision
> purposes in this project the qualitative ranking — *modulating more
> stages helps, early-stage modulation contributes non-trivially* — is
> what carries over.

**Take-aways.** (i) The paper supplies the neuroscience-style motivation
for modulating *within* the perceptual encoder, which the project's
Phase 3 precision architecture reuses. (ii) It demonstrates that a
conditioning signal can be injected cheaply, by *reparameterising already
trained* BN affine parameters rather than adding new layers. (iii) The
freezing strategy is the conceptual ancestor of the project's
"lightweight modulator on top of a frozen encoder" variant in
`FiLM_ENSEMBLE_SENSORY_PRECISION.md`.

### Layer 3 — Graduate-level deep dive

#### 3.1 Standard batch normalisation (reference form)

For a pre-activation feature map $F_{b,c,h,w}$ with batch index $b$ and
channel $c$, vanilla BN computes per-channel batch statistics

$$
\mu_c = \frac{1}{B H W}\sum_{b,h,w} F_{b,c,h,w}, \qquad
\sigma^2_c = \frac{1}{B H W}\sum_{b,h,w} (F_{b,c,h,w} - \mu_c)^2, \tag{5}
$$

then normalises and applies a *learned* affine transform,

$$
\mathrm{BN}(F_{b,c,h,w}) = \gamma_c \cdot \frac{F_{b,c,h,w} - \mu_c}{\sqrt{\sigma^2_c + \varepsilon}} + \beta_c. \tag{6}
$$

The $(\gamma_c, \beta_c)$ pair is learned jointly with the rest of the
network and — crucially for this paper — is *where the only linear
degree of freedom per channel lives* once the convolutions are frozen.

#### 3.2 Conditional Batch Normalisation (CBN)

CBN replaces the per-channel affine pair by a function of a conditioning
embedding $z$ (the LSTM's final hidden state in MODERN). The paper's
parameterisation is a **residual offset** from the pre-trained affine
parameters:

$$
\gamma_c(z) = \gamma_c + \Delta\gamma_c(z), \qquad
\beta_c(z) = \beta_c + \Delta\beta_c(z), \tag{7}
$$

with the offsets produced by a small MLP $g_\phi$ applied to the language
embedding $z$ and split into two $C$-dimensional halves:

$$
[\Delta\gamma(z);\ \Delta\beta(z)] = W^{(2)}\,\tanh\!\bigl(W^{(1)} z + b^{(1)}\bigr) + b^{(2)}, \tag{L.3}
$$

$W^{(1)} \in \mathbb{R}^{d_h \times d_z}$, $W^{(2)} \in \mathbb{R}^{2C \times d_h}$.
The normalisation itself is unchanged — CBN simply substitutes the
conditional $(\gamma_c(z), \beta_c(z))$ into equation (6):

$$
\mathrm{CBN}(F_{b,c,h,w} \mid z) = \gamma_c(z)\, \frac{F_{b,c,h,w} - \mu_c}{\sqrt{\sigma^2_c + \varepsilon}} + \beta_c(z). \tag{8}
$$

Two design decisions inside (7) matter.

**Residual offset, not direct prediction.** Writing
$\gamma_c(z) = \gamma_c + \Delta\gamma_c(z)$ rather than
$\gamma_c(z) = g_c(z)$ anchors the layer at the pre-trained affine — if
$\Delta$ is small and centred on zero, the network starts at the
pre-trained behaviour. This is an inductive bias that the later FiLM
(paper 1) deliberately abandons; both choices have consequences. The
residual form cannot *suppress* a channel that the pre-trained ResNet
actively uses (because $\gamma_c$ dominates); the direct form can, at the
cost of larger training signal requirements.

**Zero-initialisation of $W^{(2)}$.** Initialising the final MLP weights
at zero forces $\Delta \equiv 0$ at step 0, so training starts from the
exact frozen-ResNet behaviour and CBN has to *earn* any deviation — this
is the standard warm-start trick and the analogue of FiLM-Ensemble's
identity-initialisation in paper 5.

#### 3.3 Freezing and parameter count

Let $L$ be the number of BN layers modulated and $C_\ell$ the number of
channels in layer $\ell$. The MLP parameter count is

$$
\bigl|g_\phi\bigr| = d_z d_h + d_h + d_h\bigl(2\textstyle\sum_\ell C_\ell\bigr) + 2\textstyle\sum_\ell C_\ell. \tag{L.4}
$$

In the full-modulation setting ($L =$ all BN layers of ResNet-50,
$\sum_\ell C_\ell \approx 2\times 10^4$), and $d_z = d_h = 1024$, this is
roughly $4\times 10^7$ parameters — a few per cent of the ResNet backbone,
and the *only* trainable parameters on the visual side.

The **freezing** regime is therefore asymmetric: the convolutional
weights and the BN running $(\mu_c, \sigma^2_c)$ are frozen at their
ImageNet-trained values; the BN affine $(\gamma_c, \beta_c)$ are *also*
left at their ImageNet-trained values (they are the anchor in (7));
only $g_\phi$ and the language / head networks are trained.

#### 3.4 Relation to FiLM and hierarchy

Dropping the normalisation step (5)–(6) from (8) recovers the FiLM
layer of paper 1:

$$
\text{FiLM}(F_{b,c,h,w} \mid z) = \gamma_c(z)\cdot F_{b,c,h,w} + \beta_c(z) = \mathrm{CBN}(F \mid z)\Big|_{\mu_c = 0,\ \sigma^2_c = 1-\varepsilon}. \tag{L.5}
$$

In the expressivity hierarchy (L.3) of paper 1, CBN sits *between* pure
FiLM and a full hypernetwork: it shares FiLM's per-channel rank-1
structure but carries the baked-in normalisation — an implicit
pre-conditioner that matters a lot in practice. The v7 observation in
`phase_1_noise_landscape.md` that "LayerNorm alone explains most of the
survival advantage" is consistent with (L.5): once the pre-conditioner
is present, the additional expressive gain from conditioning the affine
is smaller than it would otherwise seem.

#### 3.5 Stage-wise ablation and the early-modulation claim

The paper's key rhetorical move is the stage-wise sweep: modulate stage 4
only, then stages 3+4, then 2+3+4, then all stages of ResNet-50. The
monotonic improvement the paper reports — with the biggest marginal gain
coming from *adding* stage-2 modulation — is the empirical basis for the
"early modulation helps" claim. Mechanistically:

- Late-stage CBN shifts high-level, object-class-like features. This is
  cheap and largely equivalent to re-weighting a final-layer classifier.
- Middle-stage CBN shifts texture / part features — enough to reshape
  the representation of mid-level primitives on which late features
  condition.
- Early-stage CBN reaches gabor / edge features; shifting them changes
  what the rest of the network receives as input. This is where the
  biological analogy lives, and where the parameter count per channel
  is smallest (early stages have fewer channels).

For the project, the relevant lesson is that modulation needs to reach
the *perceptual* features, not just the late latent — which is why
Phase 3's FiLM insertion point is in the sensor encoder, not just in the
policy head.

#### 3.6 Relation to the project's Phase 3 plan

Three concrete carry-overs from De Vries et al. (2017) into
`FiLM_ENSEMBLE_SENSORY_PRECISION.md`:

1. **Residual anchoring (L.3 + L.5).** The project's gate should
   initialise at $\gamma \approx 1$, $\beta \approx 0$, by zero-init of
   the FiLM generator's final layer. This is the identity-start of CBN.
2. **Modulate the encoder, not the policy head.** MODERN puts CBN inside
   the ResNet because that is where "early visual processing" lives; the
   project's analogue is to place FiLM inside the sensor encoder.
3. **Per-stage modulation is cheap; late-only is insufficient.** The
   stage-wise ablation is the standard justification for modulating more
   than one depth — relevant for the project's choice to insert FiLM
   after multiple encoder stages rather than only at the output head.

---

---

## 3. Dumoulin et al. (2017) — *A Learned Representation for Artistic Style* (CIN)

**Venue:** ICLR 2017 (arXiv:1610.07629). **Domain:** Feed-forward neural
style transfer. **Why it belongs in SQ2:** this is where **Conditional
Instance Normalisation (CIN)** is introduced, and — more importantly for
the project — it is the paper that first empirically documents the
*surprising compression property* that later underwrites the whole
"conditioning via affine parameters" paradigm: essentially all of the
between-task specialisation can live in a tiny set of per-channel
$(\gamma, \beta)$ parameters, with the entire convolutional backbone shared.
That is the structural assumption precision-as-$\gamma$ arguments rely on.

### Layer 1 — Basic introduction and concept

Style transfer had two operating modes at the time. The flexible
optimisation-based recipes (Gatys et al., 2015) could render *any* style but
took minutes per image. The fast feed-forward recipes (Johnson et al.,
2016; Ulyanov et al., 2016) produced a stylised image in one forward pass
but required *one network per style*. Training and shipping thousands of
these — and ignoring that many styles share low-level primitives like
brush-strokes — was clearly wasteful.

The authors ask: how much of that separate-network machinery is really
necessary? Their answer — a single feed-forward network can produce
$N$ distinct styles if its convolutional weights are shared and only the
*per-channel affine parameters after normalisation* are specialised per
style — was not a foregone conclusion. Concretely, with $N = 32$ styles
they find a single network that matches per-style baselines, using only
$\sim 3{,}000$ parameters per style on top of a $\sim 1.6$M-parameter
trunk (0.2% overhead per style).

The conceptual pay-off: the $(\gamma_s, \beta_s)$ pairs function as a
*learned embedding* of styles, and convex combinations of them produce
visually coherent blends. Style becomes a vector in an affine-parameter
space — a point that generalises immediately to any "task / condition /
context" axis.

### Layer 2 — Main results and algorithm

**Architecture.** A standard feed-forward image-transformation network
(Johnson et al., 2016-style) with Instance Normalisation (IN) layers.
Every IN layer is replaced by CIN — a style-indexed version of the same
affine operation. Convolutional weights, biases, and all other parameters
are shared across styles; only the style-indexed $(\gamma^{(s)}, \beta^{(s)})$
vectors differ.

**Results.**

| Claim | Value | Notes |
|-------|-------|-------|
| Styles in one network | **32** distinct styles | single model; CIN-only per-style specialisation |
| Extended demo | 10 Monet paintings | focused within-artist set |
| Parameter overhead per style | $\approx 3{,}000$ / $\approx 0.2\%$ | scales as $O(N \cdot L)$, $L = \sum_\ell C_\ell$ |
| Style-loss vs per-style baseline | **$-8.9 \pm 16.5\%$** (lower) | N-style matches or beats separate nets |
| Content-loss vs per-style baseline | $+8.7 \pm 3.9\%$ (slightly higher) | small regression in content fidelity |
| New-style integration cost | freeze trunk, learn new $(\gamma, \beta)$ | converges much faster than training from scratch |

**Key qualitative finding.** The 10-style and 32-style models produce
stylisations "virtually indistinguishable" from the single-style baselines.
Since the convolutional trunk is *literally identical* across styles, this
is strong evidence that a tiny, rank-1, per-channel affine modulation is
*sufficient* to specialise an already-good feature extractor.

**Style-space interpolation.** Because the $(\gamma^{(s)}, \beta^{(s)})$
pairs form an embedding, a convex combination of two styles' parameters
produces a smooth stylistic blend. This is the earliest demonstration that
the conditioning-by-affine space has a meaningful geometry — a property
that FiLM (paper 1) inherits and that motivates the use of γ as an
*interpolable* precision signal downstream.

**Take-aways.** (i) The compression claim (specialisation can fit in
$\gamma, \beta$) is the empirical backbone of the whole SQ2 lineage.
(ii) The per-style parameter count formula $O(N \cdot L)$ is how the
project's own Phase 3 modulator should be sized. (iii) The
interpolation / fast-fine-tuning result predicts a benign property for
Phase 3: adapting an already-trained sensor encoder to a new noise regime
should be cheap if only $\gamma, \beta$ have to move.

### Layer 3 — Graduate-level deep dive

#### 3.1 Instance Normalisation (reference form)

Instance Normalisation differs from Batch Normalisation (paper 2, eqs.
(5)–(6)) in the *axis* of the moment statistic. For a feature tensor
$x_{nchw}$ with batch index $n$ and channel $c$, IN computes
**per-sample, per-channel** statistics across spatial dimensions only:

$$
\mu_{nc}(x) = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{nchw}, \tag{9}
$$

$$
\sigma_{nc}(x) = \sqrt{\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}\bigl(x_{nchw} - \mu_{nc}(x)\bigr)^2 + \varepsilon}. \tag{10}
$$

Compared to BN, there is no averaging across samples in the batch and no
running statistic — IN is a *deterministic* function of a single image.
For generative image networks this matters: BN's mixing across a batch
creates artefacts in style transfer (noticed originally by Ulyanov et al.
2016), which is why IN became the default normaliser for these
architectures.

#### 3.2 Conditional Instance Normalisation

The **CIN** layer replaces IN's single learned $(\gamma, \beta)$ pair by a
style-indexed pair. For style index $s \in \{1,\ldots,N\}$ and an input
activation $x_{nchw}$, CIN outputs

$$
z_{nchw} = \gamma^{(s)}_c \cdot \frac{x_{nchw} - \mu_{nc}(x)}{\sigma_{nc}(x)} + \beta^{(s)}_c. \tag{11}
$$

The full parameter set on the conditioning side is two matrices
$\Gamma \in \mathbb{R}^{N \times C}$ and $B \in \mathbb{R}^{N \times C}$
per CIN layer; for an $L$-layer network the total conditional parameter
budget is

$$
|\theta_{\mathrm{CIN}}| = 2 N \sum_{\ell=1}^{L} C_\ell = O(N \cdot L_\text{total}). \tag{L.6}
$$

With $\sum_\ell C_\ell \approx 1{,}500$ channels across the Johnson-style
trunk and $N = 32$, this gives the $\sim 3{,}000$-per-style figure quoted
above — the rest of the 1.6M parameters are in the shared convolutions.

#### 3.3 Relation to CBN and FiLM

Stacking up the three layers in axis/parameter form:

| Layer | Normalisation axis | Affine parameters |
|-------|--------------------|-------------------|
| BN (eq. 6) | batch + spatial | learned per channel |
| CBN (eq. 8) | batch + spatial | $\gamma_c(z), \beta_c(z)$ — function of $z$ |
| IN (eqs. 9–10 + std. affine) | spatial only | learned per channel |
| CIN (eq. 11) | spatial only | $\gamma^{(s)}_c, \beta^{(s)}_c$ — style-indexed |
| FiLM (eq. 2) | *none* | $\gamma_c(z), \beta_c(z)$ — function of $z$ |

Two facts matter. First, CIN is to IN as CBN is to BN — same move on a
different normaliser. Second, the conditioning signal in CIN is a
*discrete one-hot style index*, whereas CBN and FiLM accept a *continuous
embedding*. The CIN→CBN→FiLM progression is therefore both a
generalisation along the conditioning axis (one-hot → continuous) *and* a
relaxation along the normalisation axis (required → optional). Formally,

$$
\text{CIN}(x; s) \;\xrightarrow{\text{continuous } z}\; \text{CBN}(x; z) \;\xrightarrow{\text{no normalisation}}\; \text{FiLM}(x; z). \tag{L.7}
$$

This hierarchy explains a subtle fact about the project: in the DreamerV3
setting where LayerNorm is the canonical normaliser, the analogue of (11)
is *Conditional Layer Normalisation* — same affine logic, LayerNorm
statistics instead. Neither the paper 2 nor paper 3 variant is the right
template by itself, but (L.7) tells you the whole family is
interchangeable under the mild assumption that the pre-conditioner is
strong (this is the v7 LayerNorm-alone finding restated).

#### 3.4 Style-space interpolation: why it works

The paper reports that mixing $(\gamma, \beta)$ vectors produces smooth
stylistic blends. The mixing rule is a straight convex combination,

$$
\gamma_\alpha = \alpha\,\gamma^{(s_1)} + (1-\alpha)\,\gamma^{(s_2)}, \qquad
\beta_\alpha = \alpha\,\beta^{(s_1)} + (1-\alpha)\,\beta^{(s_2)}, \tag{12}
$$

and the resulting $z_\alpha = \gamma_\alpha \hat{x} + \beta_\alpha$ is itself
an affine function of $\alpha$. Since the rest of the network is shared,
and the per-layer output is (locally) Lipschitz in the affine parameters,
the pastiche varies continuously with $\alpha$. Two consequences:

1. **Affine parameters form an embedding.** The set
   $\{(\gamma^{(s)}, \beta^{(s)})\}_s$ is a *learned embedding of styles*
   in $\mathbb{R}^{2C}$, not just a collection of per-task settings. This
   is what paper 4 (HyperNetworks) promotes to a continuous hypernetwork
   output, and what paper 1 (FiLM) makes explicit by having a network
   produce $(\gamma, \beta)$ from a continuous $z$.
2. **New-style cost is sub-linear.** To add style $N+1$, the authors
   freeze everything and train only $(\gamma^{(N+1)}, \beta^{(N+1)})$ —
   $\sim 3{,}000$ parameters. Convergence is much faster than training
   from scratch because the trunk is already a good general-purpose
   style-transfer extractor. This is the same phenomenon paper 2 exploits
   for VQA (frozen ResNet + CBN) and that the project exploits for Phase
   3 (frozen encoder + trainable FiLM gate).

#### 3.5 Why IN, not BN, for this paper

A non-obvious detail: if CIN were built on BN, the affine transform would
be applied to *batch-mixed* statistics, and a single training batch could
contain images targeting different styles. Each style's target statistics
would corrupt the normalisation of the others — the CIN mechanism would
be fighting the normaliser. IN sidesteps this entirely because (9)–(10)
are per-sample. For BN-based conditioning (paper 2's VQA setting), the
analogous fix is to condition on a *single* language embedding *per
batch* or to use GroupNorm / LayerNorm; for IN-based conditioning no such
fix is needed. The $\gamma, \beta$ embedding story therefore depends on
the normaliser's compatibility with per-sample conditioning — worth
noting when porting to RL, where the "batch" is a rollout slice that is
not semantically i.i.d.

#### 3.6 Project carry-overs

1. **The compression claim is the project's structural prior.** If the
   sensor encoder is general enough, a small per-channel $(\gamma, \beta)$
   modulator is *sufficient* to specialise it to a noise context. This is
   Dumoulin et al.'s 32-styles-in-one-net observation, ported to a
   one-noise-regime-per-embedding setting.
2. **Parameter budget.** (L.6) is the sizing equation. With $\sim 10$
   modulated layers and $\sim 256$ channels each, the Phase 3 FiLM
   modulator carries $\sim 5{,}000$ parameters per noise regime — small
   enough to co-exist with a frozen DreamerV3 decoder.
3. **Normaliser compatibility matters.** In the DreamerV3 / PPO
   LayerNorm regime the project targets, the equivalent of (11) is a
   **Conditional LayerNorm** layer — same affine form, LN statistics.
   The v7 "LayerNorm-alone explains most" finding is the analogue of IN
   being the right pre-conditioner for CIN.

---

---

## 4. Ha, Dai & Le (2017) — *HyperNetworks*

**Venue:** ICLR 2017 (arXiv:1609.09106). **Domain:** CIFAR-10 image
classification (static hypernet on a Wide ResNet) and character-level
Penn Treebank language modelling (dynamic HyperLSTM). **Why it belongs in
SQ2:** hypernetworks are the *general* conditioning primitive of which
FiLM (paper 1) and CIN / CBN (papers 2–3) are rank-1 special cases. The
paper is also the origin of the "relaxed weight-sharing" framing that
makes all the downstream conditional-affine mechanisms feel principled
rather than ad-hoc.

### Layer 1 — Basic introduction and concept

A hypernetwork is a *small* network whose output *is the weights* of
another, larger main network. The main network maps raw inputs to
targets; the hypernetwork takes a conditioning input — a layer index, a
previous hidden state, a task id — and emits the parameters the main
network will use for the current forward pass.

The motivating framing is a spectrum of weight-sharing strategies. At one
end, a deep CNN has effectively *no* weight-sharing across layers —
every conv filter is a free parameter, which is both expensive and
redundant (filters in adjacent layers look alike). At the other end, a
vanilla RNN uses *strict* weight-sharing across every timestep, which is
efficient but rigid — the same matrix has to serve every context.
Hypernetworks sit between the two: you share the *meta-parameters* of the
hypernetwork, and let the generated per-layer or per-timestep weights
carry whatever specialisation is needed. The main-network weights are no
longer free parameters but a *function* of context.

This is the idea that paper 1 (FiLM) implements in its minimal form
(rank-1 multiplicative + additive shift per channel), and paper 3 (CIN)
had already instantiated in its rank-1 table-lookup form (one
$(\gamma, \beta)$ pair per style index). Hypernetworks generalise both to
arbitrary weight-matrix outputs.

### Layer 2 — Main results and algorithm

Two flavours are developed.

**Static hypernetworks for CNNs.** For a deep CNN with $D$ layers, a
per-layer embedding $z^j \in \mathbb{R}^{N_z}$ is *learned* (one embedding
per layer), and a shared hypernetwork $g$ maps $z^j$ to the full
convolutional kernel $K^j$ of that layer. The kernel is produced row by
row from per-row sub-embeddings, which keeps $g$ small. Empirically, on
CIFAR-10 with a Wide ResNet 40-2 backbone:

| Model | Params | Test error |
|-------|:------:|:----------:|
| Standard WRN 40-2 | 2.2 M | 5.33% |
| Hyper ResNet 40-2 | **0.148 M** (~15× smaller) | 7.23% |

The $\sim\!15\times$ compression costs $\sim\!1.5\%$ accuracy — a strong
demonstration that most of a CNN's filters live on a low-dimensional
manifold navigable by a tiny generator.

**Dynamic hypernetworks for RNNs — HyperLSTM.** A small auxiliary LSTM
(the hyper-cell) runs alongside the main LSTM. Its hidden state $\hat h_t$
drives *time-varying* weight-modulation signals — per-row scaling
vectors $d$ and a bias $b$ — that are applied to the main LSTM's weight
matrices, inside a LayerNorm wrap. On character-level Penn Treebank:

| Model | Params | Test BPC |
|-------|:------:|:--------:|
| 1000-unit LSTM | 4.25 M | 1.312 |
| 2-layer 1000-unit LSTM | 12.26 M | 1.281 |
| **HyperLSTM** (1000 units) | 4.91 M | **1.265** |
| HyperLSTM + LN | ~5 M | **1.250** |

A single-layer HyperLSTM with minimal parameter overhead beats a 2×
larger two-layer baseline. This is the result most often cited as
evidence that context-adaptive weights are a different *kind* of
capacity than simply adding depth.

**Take-aways.** (i) Hypernetworks frame weight-sharing as a design
continuum rather than a binary choice. (ii) The rank-1 special case —
generating only per-row *scalings* of a fixed matrix — is already enough
for HyperLSTM's improvements, and this is exactly the primitive that
FiLM (paper 1) and CIN (paper 3) use. (iii) The memory overhead scales
with $N_z$ (embedding dim) times hidden units, not with the full weight
matrix — the trick that makes the mechanism tractable at scale.

### Layer 3 — Graduate-level deep dive

#### 4.1 Static hypernetwork for a CNN

Let the main network's layer $j$ have a convolutional kernel
$K^j \in \mathbb{R}^{N_{\text{in}} \times N_{\text{out}} \times f_{\text{size}} \times f_{\text{size}}}$.
A static hypernetwork stores one learned embedding
$z^j \in \mathbb{R}^{N_z}$ per layer and generates $K^j$ from $z^j$
via a two-stage linear network.

**First stage — layer-embedding expansion.** For each "slice index"
$i \in \{1,\ldots,N_{\text{in}}\}$, produce an intermediate vector

$$
a_i^j = W_i\, z^j + B_i, \qquad i = 1,\ldots,N_{\text{in}},\ j = 1,\ldots,D. \tag{13}
$$

Here $W_i \in \mathbb{R}^{d \times N_z}$ and $B_i \in \mathbb{R}^{d}$ are
*shared across layers* (indexed by $i$ only), so adding a new layer $j$
only costs one new $z^j$.

**Second stage — kernel synthesis.** Each $a_i^j$ is projected into one
kernel slice $K_i^j \in \mathbb{R}^{N_{\text{out}} \times f_{\text{size}} \times f_{\text{size}}}$
via a shared tensor $W_{\text{out}}$ and bias $B_{\text{out}}$,

$$
K_i^j = \langle W_{\text{out}},\, a_i^j\rangle + B_{\text{out}}, \qquad i = 1,\ldots,N_{\text{in}}, \tag{14}
$$

with the full layer-$j$ kernel assembled by concatenating the slices

$$
K^j = \bigl(K_1^j,\, K_2^j,\, \ldots,\, K_{N_{\text{in}}}^j\bigr). \tag{15}
$$

The parameter count of this generator is

$$
\bigl|\theta_{\text{hyper}}\bigr| = \underbrace{N_{\text{in}}(N_z d + d)}_{\text{eq.}~13} + \underbrace{d \cdot N_{\text{out}} f_{\text{size}}^2 + N_{\text{out}} f_{\text{size}}^2}_{\text{eq.}~14} + \underbrace{D\, N_z}_{\text{layer embeddings}}, \tag{L.8}
$$

independent of $D$ in the dominant terms. With a Wide ResNet 40-2 this
gives the 0.148 M figure against 2.2 M for the standard model — the
$\sim\!15\times$ compression quoted above.

#### 4.2 Relation to FiLM

Specialising (13)–(15) to the case where the "generated weights" are a
*diagonal per-channel scaling* and a *per-channel bias* recovers exactly
the FiLM layer of paper 1. Concretely, if the main network's layer is a
point-wise affine $F \mapsto \mathrm{diag}(\gamma) F + \beta$, the
hypernet output collapses to

$$
\gamma = W_\gamma z + b_\gamma, \qquad \beta = W_\beta z + b_\beta, \tag{L.9}
$$

which is eq. (1) of paper 1 with shared $g_\phi$ replaced by a single
linear layer. FiLM is therefore the *rank-1 diagonal special case* of the
hypernetwork. Equivalently, dropping to even less expressivity — a
per-style *table* rather than a continuous map — gives CIN (paper 3,
eq. (11)). Schematically:

$$
\underbrace{\text{CIN}\,(s \to \gamma^{(s)}, \beta^{(s)})}_{\text{rank-1, table}} \;\subset\; \underbrace{\text{FiLM}\,(z \to \gamma(z), \beta(z))}_{\text{rank-1, continuous}} \;\subset\; \underbrace{\text{HyperNet}\,(z \to W(z))}_{\text{full rank, continuous}}. \tag{L.10}
$$

(L.10) is the rigorous form of (L.3) in paper 1. The reason this matters
for SQ2 is that *the failure modes propagate up the chain*: an
uncertainty-driven hypernet with no uncertainty-aware training signal
will collapse in the same way FiLM does, only with a much larger weight
pool to misuse.

#### 4.3 Dynamic HyperLSTM (abridged)

For the recurrent variant, an auxiliary "hyper-LSTM" with hidden state
$\hat h_t \in \mathbb{R}^{N_{\hat h}}$ runs in parallel with the main
LSTM. From $\hat h_t$ the hypernet produces three embeddings

$$
z_h = W_{\hat h_h}\, \hat h_{t-1} + b_{\hat h_h}, \quad
z_x = W_{\hat h_x}\, \hat h_{t-1} + b_{\hat h_x}, \quad
z_b = W_{\hat h_b}\, \hat h_{t-1}, \tag{16}
$$

and for each LSTM gate $y \in \{i, g, f, o\}$ the main update is

$$
y_t = \mathrm{LN}\!\Bigl(d^y_h(z_h) \odot W^y_h\, h_{t-1} + d^y_x(z_x) \odot W^y_x\, x_t + b^y(z_b)\Bigr), \tag{17}
$$

where the modulation vectors and bias are further linear in $z$,

$$
d^y_h(z_h) = W^y_{hz} z_h, \quad d^y_x(z_x) = W^y_{xz} z_x, \quad b^y(z_b) = W^y_{bz} z_b + b^y_0. \tag{18}
$$

The critical architectural choice is in (17): instead of generating the
full matrices $W^y_h, W^y_x$ from scratch, the hypernetwork produces
*per-row scaling vectors* $d$ that multiplicatively adjust a fixed
underlying weight matrix. Memory overhead is therefore
$O(N_z \cdot \text{hidden})$ rather than $O(\text{hidden}^2)$, and
training sees a main-network matrix with gradient-friendly statistics
while the hypernet contributes the context-dependence.

Rewriting (17) in the project's preferred vocabulary: $d^y_h$ is a
**per-row gain** and $b^y$ is a **per-row shift**, applied to a LayerNorm-
normalised pre-activation. That is precisely a FiLM layer acting on the
rows of a weight matrix instead of the channels of a feature tensor.

#### 4.4 Why "relaxed weight-sharing" is the right framing

The standard objection to FiLM-family mechanisms is that they are
"just" affine layers — they can be absorbed into the next matrix
multiply. (L.10) is the precise answer to that objection. FiLM is the
rank-1 compression of a hypernet; it inherits the *inductive bias* that
context influences the model's weights in a low-dimensional,
per-channel way. That bias is what you lose by absorbing FiLM into the
next layer: you retain the expressivity but reinstate $O(\text{hidden}^2)$
free parameters per context value, which both defeats the parameter
economy and — more importantly — removes the structural prior that
makes the modulator interpretable as "precision" or "style".

#### 4.5 Project carry-overs

1. **Identity-collapse is a family-level failure mode.** Paper 1 § 3.2
   of the parent review notes that FiLM can collapse to $\gamma \equiv 1,
   \beta \equiv 0$ when the training signal does not shape the
   conditioner. The hypernet analogue is a uniform $W(z) = W_0$ — same
   failure, larger pool. Project mitigation (an auxiliary loss) should
   therefore apply uniformly across (L.10).
2. **Rank-1 is the right default.** The project's Phase 3 design uses
   FiLM, not a full hypernet, for exactly the parameter-economy /
   inductive-bias reason above. Paper 4's CIFAR-10 compression result
   (15× with $-1.5\%$ accuracy) is the empirical precedent that *rank-1
   is enough* for most of the capacity benefit.
3. **Dynamic conditioning is available if needed.** If Phase 3 FiLM
   driven by a static precision head fails, the HyperLSTM dynamic form
   — $z_t$ updated at each time step from recurrent state — is the next
   rung up. The project need not jump straight to a full hypernet.

---

---

## 5. Turkoglu et al. (2022) — *FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation*

**Venue:** NeurIPS 2022. **Domain:** Image classification (CIFAR-10,
CIFAR-100, Retina Glaucoma) and genomics (6mA 1D-CNN). **Why it anchors
SQ1 + SQ2:** this is the paper the project's Phase 3 design directly
ports. It is the cleanest statement of *FiLM-as-ensemble-mechanism*: a
single shared backbone, M lookup-table FiLM members, and calibration
that approaches explicit deep ensembles at a tiny fraction of the memory
cost. Two findings from the actual paper are important to get right
because they correct a common misreading (including one in the parent
review, §3.3):
> **Correction to perceptual_noise_lit_review.md §3.3.** The parent
> review describes FiLM-Ensemble as trained with a "heteroscedastic NLL
> objective". Per Turkoglu et al., classification FiLM-Ensembles are
> trained with **standard cross-entropy**; the "auxiliary precision
> signal" is supplied by *inter-member* variance, not a per-member log-σ̂²
> head. This matters for the project — the Phase 3 design has to add
> that head itself if it wants heteroscedastic aleatoric uncertainty.

### Layer 1 — Basic introduction and concept

Epistemic uncertainty — the model not knowing what it does not know — is
essential for reliable deployment. The de facto standard for capturing
it is the **explicit deep ensemble** (Lakshminarayanan et al., 2017): train
$M$ independent copies of the model and combine their predictions. This
works but is brutal in practice — $M$× the training cost, $M$× the
inference cost, $M$× the memory. Prior "implicit" alternatives
(MC-Dropout, MIMO, BatchEnsemble) trade off accuracy or calibration
against that economy, and none matches explicit ensembles cleanly.

FiLM-Ensemble makes the observation that *most* of what distinguishes
ensemble members is a per-channel re-scaling of the same feature tensors
(the intuition of CIN, paper 3, applied across random seeds instead of
styles). So: keep a single shared backbone; give each member $m$ its own
lookup-table $(\gamma^m, \beta^m)$ FiLM pair at every normalisation
layer; train everything jointly. The conditioning variable $z$ collapses
to the member index $m \in \{1, \ldots, M\}$.

The result is a *shared-weight implicit ensemble* where adding a
16-member ensemble to a ResNet-18 costs only $+1.3\%$ parameters
(vs $+1500\%$ for an explicit ensemble), yet calibration at large $M$
actually *exceeds* the explicit ensemble (lower ECE).

### Layer 2 — Main results and algorithm

**Algorithm.** A single backbone $f_\theta$ (VGG-11, ResNet-18/34,
EfficientNet-B0, 1D-CNN for genomics). At every normalisation layer $n$
in the backbone, the standard BN/GN affine parameters are *replaced* by
$M$ per-member affine pairs $(\gamma^m_n, \beta^m_n)$ stored as lookup
tables. At each training step, a member index $m$ is drawn (or all $M$
are run jointly, depending on batch handling); the forward pass selects
that member's $(\gamma^m_n, \beta^m_n)$ and applies a plain FiLM
modulation — member selection therefore routes an $M$-way "affine
subnetwork" through a single convolutional trunk.

**Loss.** Classification: standard cross-entropy, averaged (or summed)
across members for joint training. No heteroscedastic NLL head in the
paper's published experiments.

**Initialisation.** Each $\gamma^m_n, \beta^m_n$ is sampled from
$\bigl[-\sqrt{3/D_n}\,\rho,\; +\sqrt{3/D_n}\,\rho\bigr]$ where $D_n$ is
the channel count of layer $n$ and $\rho$ is a tunable scalar "gain". The
bound is chosen so the per-member affine perturbation has variance
comparable to the backbone's own He-style initialisation, preventing any
one member from dominating at step 0.

**Ensemble aggregation.** The final prediction is a plain average of the
per-member outputs: $\hat y = \tfrac{1}{M}\sum_m y_m$. For classification
this is averaged in logit space (then softmax) or in probability space;
the paper uses the usual soft-voting.

**Key empirical results.**

| Method | CIFAR-100 ResNet-18 ($M{=}4$) Acc | ECE |
|--------|:---------------------------------:|:-----:|
| FiLM-Ensemble | **79.4%** | **0.038** |
| Deep Ensemble | 81.6% | 0.041 |
| BatchEnsemble | 77.7% | 0.052 |
| MC-Dropout | 75.5% | 0.064 |

Two qualitative patterns from the full sweep (all $M$ up to $\sim 16$):

1. **Accuracy.** FiLM-Ensemble is beaten only by the explicit deep
   ensemble; it beats BatchEnsemble and MC-Dropout consistently.
2. **Calibration.** FiLM-Ensemble's ECE *improves* as $M$ grows and at
   large $M$ drops below the explicit ensemble's. BatchEnsemble's
   accuracy degrades with $M$ — a pathology FiLM-Ensemble does not
   share.

**Parameter cost.** Adding 16 members to ResNet-18 is $+1.3\%$
parameters; an explicit 16-member ensemble is $+1500\%$. The asymmetry
between parameter and calibration/accuracy gains is the paper's
headline.

**Take-aways.** (i) FiLM is expressive enough to carry ensemble
diversity on top of a shared backbone — the CIN "compression" claim
extends from styles to seeds. (ii) Calibration from inter-member
variance is a real thing: a well-tuned implicit ensemble matches or
beats the explicit form. (iii) The paper does *not* establish FiLM as a
heteroscedastic mechanism — that interpretation has to be added via a
per-member log-σ̂² head (the project's Phase 3 modification).

### Layer 3 — Graduate-level deep dive

#### 5.1 The per-member FiLM layer

For a feature tensor $F_n \in \mathbb{R}^{B \times D_n \times H \times W}$
at layer $n$ and for ensemble member $m \in \{1,\ldots,M\}$, the
FiLM-Ensemble modulation is

$$
\mathrm{FiLM}(F_n \mid \gamma^m_n, \beta^m_n) = \gamma^m_n \circ F_n + \beta^m_n, \tag{19}
$$

where $\circ$ is a Hadamard product broadcast along the feature
dimension $D_n$ (channels, for a conv layer). The conditioning variable
has collapsed to the member index — the "generator" in the FiLM sense
(paper 1, eq. (1)) is a plain lookup table

$$
\gamma^m_n = \Gamma_n[m,:], \qquad \beta^m_n = B_n[m,:], \qquad \Gamma_n, B_n \in \mathbb{R}^{M \times D_n}. \tag{L.11}
$$

This is structurally *identical* to Conditional Instance Normalisation
(paper 3, eq. (11)), except that the conditioning axis is "ensemble
member index" rather than "style index", and the normaliser is whatever
the backbone uses (BN for ResNets, GN for some variants) rather than IN.

#### 5.2 Initialisation

Each entry of $\Gamma_n$ and $B_n$ is sampled from

$$
\gamma^m_{n,c},\ \beta^m_{n,c} \overset{\text{i.i.d.}}{\sim} \mathcal{U}\!\Bigl(-\sqrt{\tfrac{3}{D_n}}\,\rho,\; +\sqrt{\tfrac{3}{D_n}}\,\rho\Bigr). \tag{20}
$$

The bound $\sqrt{3/D_n}$ comes from demanding
$\mathrm{Var}[\gamma^m_{n,c}] = \rho^2/D_n$, which matches the
He / LeCun initialisation magnitude on the pre-modulation
pre-activation. $\rho$ is a global gain (the paper sweeps values up to
$\sim 1$); larger $\rho$ trades off ensemble diversity against
short-horizon training stability.

Note that (20) initialises $\gamma^m \sim 0$, *not* $\gamma^m \sim 1$ —
so the per-member FiLM layer *starts as an approximately noise-level
perturbation* of the feature tensor, with the backbone's own affine (or
lack thereof) doing the heavy lifting early in training. This is
different from CBN (paper 2, eq. (7)), which anchors at $\gamma \equiv 1$
by zero-init of the MLP. It is also different from the identity-start
that the project's Phase 3 FiLM gate should adopt — a deliberate
departure worth being explicit about.

#### 5.3 Joint training objective

Let $\theta$ be the shared backbone parameters and
$\Phi = \{\Gamma_n, B_n\}_{n=1}^{L}$ the per-member affine tables. For a
classification dataset $\{(x_i, y_i)\}_{i=1}^N$, training minimises the
sum of per-member cross-entropies,

$$
\mathcal{L}(\theta, \Phi) = \frac{1}{N M}\sum_{i=1}^{N}\sum_{m=1}^{M} \mathrm{CE}\!\bigl(y_i,\ \mathrm{softmax}(f_{\theta, \Phi_m}(x_i))\bigr), \tag{21}
$$

with gradients flowing into the *shared* $\theta$ from all $M$ members
and into $\Phi_m$ only from the $m$-th member's loss. No heteroscedastic
NLL, no per-member log-σ̂² head, no variance-reduction regulariser — (21)
is the entire training story.

At inference, the ensemble prediction is the uniform average

$$
\hat y(x) = \frac{1}{M}\sum_{m=1}^{M} f_{\theta, \Phi_m}(x). \tag{22}
$$

#### 5.4 How predictive uncertainty arises

Even though (21) contains no explicit variance head, FiLM-Ensemble
produces non-trivial predictive uncertainty through inter-member
disagreement. Decomposing the predictive distribution in the classic
deep-ensemble form (Lakshminarayanan et al., 2017),

$$
\mathrm{Var}_\text{pred}[y \mid x] \;=\; \underbrace{\frac{1}{M}\sum_{m=1}^{M} f_m(x)\,f_m(x)^\top - \hat y(x)\,\hat y(x)^\top}_{\text{epistemic (between-member)}} \;+\; \underbrace{0}_{\text{aleatoric (not modelled)}}. \tag{L.12}
$$

(L.12) is the precise form of the "auxiliary precision signal" the
parent review refers to: it is *epistemic* variance from member
disagreement, not aleatoric variance from a learned per-sample log σ̂².
FiLM-Ensemble earns its calibration by keeping the $M$ members diverse
enough that (L.12) is non-trivial on in-distribution inputs and grows
on OOD inputs.

The diversity engine is the random initialisation (20) plus the
per-member lookup-table gradient isolation — member $m$'s $\Phi_m$ only
sees member $m$'s loss, so early in training members inherit different
per-channel preferences, which the joint loss then sharpens into
different predictive distributions.

#### 5.5 Why FiLM-Ensemble calibrates better than BatchEnsemble

BatchEnsemble (Wen et al., 2020) modulates weights rather than
activations: each member multiplies the shared weight matrix $W$ by a
rank-1 perturbation $r_m s_m^\top$, producing member weight
$W_m = W \odot (r_m s_m^\top)$. The per-member perturbation has
$O(\text{in} + \text{out})$ parameters — slightly fewer than FiLM's
$O(\text{channels})$ per layer, but at the cost of passing the
perturbation through the *nonlinearity* that follows the weight
multiply. FiLM-Ensemble, by contrast, modulates the post-normalisation,
pre-nonlinearity activations directly, which preserves member
statistics under ReLU / SiLU in a way that rank-1 weight perturbations
do not. Empirically this shows up as BatchEnsemble's accuracy-versus-$M$
pathology: as $M$ grows, maintaining calibration forces increasingly
aggressive per-member weight perturbations that degrade the shared
backbone's ability to fit the data. FiLM-Ensemble does not have that
trade-off because its perturbation lives outside the normalisation.

#### 5.6 Parameter economy

With $L$ modulated layers of channel counts $D_1, \ldots, D_L$ and $M$
members, the FiLM-Ensemble parameter overhead is

$$
|\Phi| = 2 M \sum_{\ell=1}^{L} D_\ell = O(M L_{\text{total}}). \tag{L.13}
$$

This is CIN's cost formula (paper 3, eq. (L.6)) with $N \to M$. For
ResNet-18, $\sum_\ell D_\ell \approx 5{,}000$; at $M = 16$ this is
$\sim 160{,}000$ extra parameters, i.e. $+1.3\%$ of the $\sim 12\mathrm{M}$
ResNet-18 budget. The explicit 16-member deep ensemble has
$|{\theta}_{\text{explicit}}| = 16 \cdot 12\mathrm{M} = 192\mathrm{M}$ —
the $+1500\%$ figure.

#### 5.7 Project carry-overs (Phase 3)

1. **The Phase 3 port needs its own aleatoric head.** FiLM-Ensemble as
   published is *purely epistemic*. The project's
   `FiLM_ENSEMBLE_SENSORY_PRECISION.md` plan to use FiLM-Ensemble as a
   heteroscedastic precision mechanism therefore requires *adding* a
   per-member (or per-sample) log σ̂² head and a β-NLL training term —
   neither is inherited from Turkoglu et al. This is a real departure
   from the reference recipe, not a re-implementation.
2. **Initialisation choice.** The paper's near-zero init (eq. (20)) is
   optimised for ensemble diversity, *not* for precision-gate identity
   start. For a precision-modulation use of FiLM, zero-init of the γ
   head (so γ ≈ 1, β ≈ 0 initially) is the right warm-start — a
   conscious deviation the project documentation should state
   explicitly.
3. **Lookup-table conditioner is the minimum.** (L.11) shows that in
   FiLM-Ensemble the conditioning network is *trivial* (identity on an
   index). The project's Phase 3 uses a *learned* precision head
   feeding γ, which corresponds to moving up one rung in the (L.10)
   hierarchy — the added expressivity is what makes context-dependent
   precision possible, but it also reopens the FiLM-collapse failure
   mode that the paper's lookup-table form avoids by construction.
4. **Calibration is not free.** FiLM-Ensemble's calibration advantage
   rests on initialisation-driven member diversity plus gradient
   isolation between members. A single-stream FiLM-precision head has
   neither — so the Phase 3 design inherits the mechanism's structural
   form but not its calibration guarantee. This is the gap the β-NLL
   auxiliary loss (paper §2.3, Seitzer et al. 2022, in the sibling
   uncertainty review) has to close.

---

## Cross-Paper Synthesis

Reading the five papers as a single lineage, three structural facts
survive across domains (style transfer, VQA, compositional reasoning,
uncertainty-aware classification) and are the most robust bets for the
project's Phase 3 design.

**Fact 1 — The conditioning hierarchy is unambiguous.**
$$
\underbrace{\text{CIN}}_{\text{Dumoulin 2017}} \;\subset\; \underbrace{\text{CBN}}_{\text{De Vries 2017}} \;\subset\; \underbrace{\text{FiLM}}_{\text{Perez 2018}} \;\subset\; \underbrace{\text{HyperNet}}_{\text{Ha 2017}},
$$
with FiLM-Ensemble (Turkoglu 2022) a *constrained regression* back to
the CIN form (lookup-table conditioner) on a deeper substrate. The
project's Phase 3 lives in the FiLM tier — one step above FiLM-Ensemble
in conditioner expressivity, two steps below a full hypernet.

**Fact 2 — Rank-1 affine is empirically sufficient, up to the
training-signal boundary.** Across all five papers, a rank-1
per-channel $(\gamma, \beta)$ modulation is enough to specialise a
general backbone to the task at hand — styles (paper 3), questions
(papers 1–2), ensemble seeds (paper 5) — *provided the training loss
actively rewards differentiation between conditioning values*. The
failure modes (variance collapse, identity collapse) show up when that
reward is weak or indirect. The project's open risk is precisely this:
RL return is a weak, indirect reward for precision-conditioned
modulation, and the parent review's v5–v8 series already documents
γ ≈ identity collapse under that regime.

**Fact 3 — Normalisation is the silent load-bearer.** BN and IN are
the dominant mechanisms in papers 2 and 3; FiLM (paper 1) deliberately
detaches from normalisation and observes that decoupling costs little
as long as *some* normaliser is present upstream. The project's v7
finding ("LayerNorm alone explains most of the survival advantage") is
the same phenomenon in RL form. A modulator without a pre-conditioner
is expressive only on paper — practically, the pre-conditioner carries
most of the signal-to-noise improvement, and the modulator harvests
what remains.

**Three concrete prescriptions for Phase 3.**

1. **Pre-conditioner first.** LayerNorm inside the sensor encoder is
   non-optional; the affine modulator is secondary (papers 1–3).
2. **Anchor the modulator at identity, not noise.** Zero-init the FiLM
   generator's final layer so γ ≈ 1, β ≈ 0 at step 0 (paper 2's
   convention). FiLM-Ensemble's near-zero init (paper 5, eq. (20)) is
   for diversity, not precision.
3. **Supply an uncertainty-aware training signal.** FiLM-Ensemble
   earns calibration through inter-member disagreement (paper 5,
   eq. (L.12)); a single-stream FiLM-precision head has to earn it
   through an explicit aleatoric head and β-NLL (sibling uncertainty
   review, Seitzer et al. 2022). Without one or the other, (L.10) will
   collapse to γ ≡ 1.

---

*Review complete. Next steps (owner: implementation agent, not Claude):
(i) incorporate correction to parent review §3.3 (FiLM-Ensemble is
cross-entropy, not heteroscedastic NLL); (ii) update
`FiLM_ENSEMBLE_SENSORY_PRECISION.md` with identity-start init decision
and explicit aleatoric-head addition; (iii) extend this review to the
remaining six papers in §7.2 of the parent review (Parr et al. 2022;
Jang et al. 2022; Santurkar et al. 2018; Shazeer et al. 2017;
Vaswani et al. 2017; Galanti & Wolf 2020) if desired.*
