---
title: "FiLM: Visual Reasoning with a General Conditioning Layer"
authors: Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville
year: 2018
venue: AAAI-18
slug: perez_2018_film
source_pdf: sources/Perez et al. 2018 - FiLM - Visual Reasoning with a General Conditioning Layer.pdf
topic: FiLM
---

# Perez et al. 2018 — FiLM: Visual Reasoning with a General Conditioning Layer

## Plain-English entry point

This is the paper that named FiLM (Feature-wise Linear Modulation) and packaged a previously scattered family of "conditional modulation" tricks into a single, general-purpose neural-network primitive. The idea is simple: when one signal needs to influence the computation of another network (e.g., a written question telling a vision network how to look at an image), inject the influence by producing two numbers per channel — a multiplicative scale called **gamma** and an additive shift called **beta** — and apply them as a per-channel **affine transform** to the target network's intermediate feature maps. "Feature-wise" means each channel (feature map) of a convolutional layer gets its own (gamma, beta) pair; "linear" because the transformation is just `y = gamma * x + beta`. The conditioning signal (here, a question encoded by a GRU recurrent network) is mapped through a small "FiLM generator" network to produce all the (gamma, beta) values that get applied at every FiLM layer.

Why it matters: the authors show this minimal mechanism beats much more elaborate, hand-engineered "visual reasoning" architectures on the CLEVR benchmark — a synthetic image-question-answer dataset where each question requires multi-step composition (e.g., counting, comparing attributes across two objects). FiLM halves the prior state-of-the-art error (from 4.5% to 2.3%) without using program-level supervision, without specialized neural modules, and without scaling cost with image resolution. The paper also demonstrates that FiLM unifies a family of previously distinct methods (conditional batch norm, conditional instance norm, AdaIN), is largely insensitive to whether normalization is present at all, and that the learned (gamma, beta) parameters interpolate meaningfully — even supporting a zero-shot generalization trick by linearly combining the (gamma, beta) of related questions.

## Section-ordered backbone

### Abstract
Introduces FiLM as a general conditioning method built on a feature-wise affine transformation of intermediate features. Claims four results on CLEVR-style visual reasoning: (1) halves state-of-the-art error, (2) modulates features coherently, (3) is robust to ablations and architectural variations, (4) generalizes well from few or zero examples.

### 1. Introduction
Frames the problem as compositional, multi-step visual reasoning that has been hard for general deep models. Argues a general-purpose mechanism — one that lets a question (processed by an RNN) influence a CNN — could match or beat reasoning-specialized architectures. Positions FiLM as a generalization of Conditional Normalization that has worked across image stylization, speech recognition, and visual question answering.

### 2. Method
**2.1 Feature-wise Linear Modulation.** FiLM learns functions `f` and `h` that produce per-feature-map scalars `gamma_{i,c}` and `beta_{i,c}` from an input `x_i`, then applies `FiLM(F_{i,c} | gamma_{i,c}, beta_{i,c}) = gamma_{i,c} * F_{i,c} + beta_{i,c}` to feature map `c` of input `i`. In practice f and h are merged into a single "FiLM generator" that outputs a `(gamma, beta)` vector. Only two parameters per modulated feature map are needed, so it is cheap and resolution-independent.

**2.2 Model.** A GRU with 4096 hidden units encodes the question (200-dim word embeddings); from its final hidden state, linear projections in each residual block produce that block's (gamma, beta). The visual pipeline produces 128 channels of 14x14 feature maps (either trained-from-scratch CNN or fixed ResNet-101 conv4 features), then runs 4 FiLM-ed residual blocks, each consisting of a 1x1 conv, a 3x3 conv, batch-norm-without-affine, FiLM, ReLU, plus a coordinate-feature-map concatenation for spatial reasoning. A final 1x1 conv, global max-pool, and 2-layer MLP yields a 28-way softmax. Trained end-to-end with Adam, weight decay 1e-5, batch size 64, early stopping by validation accuracy.

### 3. Related Work
Positions FiLM as a generalization of Conditional Normalization (CN), unified across Conditional Instance Norm (Dumoulin), AdaIN (Huang & Belongie), Dynamic Layer Norm (Kim et al.), Conditional Batch Norm (de Vries et al.). Notes that prior CN literature included normalization in the method name for instructive or implementation reasons but never tested whether the affine transformation has to immediately follow normalization. The paper shows it does not. FiLM is also related to (a) concatenation-based conditioning, which is equivalent to FiLM with gamma=1; (b) hypernetworks (one network produces parameters of another); (c) mixture-of-experts / conditional computation, but at a feature-map level rather than sub-network level; (d) self-gating like LSTMs and Squeeze-and-Excitation, which use scaling-only and only between 0 and 1.

### 4. Experiments
**4.1 CLEVR.** FiLM hits 97.7% overall accuracy, beating the prior state-of-the-art (PG+EE with 700K program labels) at 96.9% and Relation Networks at 95.5%, all without program supervision. Performance is equally strong on raw pixels vs. pretrained features.

**4.2 What do FiLM layers learn?** Activation maps show the FiLM-ed CNN localizes question-relevant objects, suggesting feature modulation produces indirect spatial attention. Histograms of (gamma, beta) span large ranges (gamma in [-15, 19], beta in [-9, 16]); gamma has a sharp peak at zero (turning whole channels off) and 36% of gammas are negative (flipping sign of feature, which interacts with the downstream ReLU). t-SNE of (gamma, beta) at the first FiLM layer clusters questions by low-level reasoning function (query color, equal color), and at the last FiLM layer by high-level function (equal-X comparisons), suggesting a self-organized hierarchy.

**4.3 Ablations.** With gamma=1 (bias-only), accuracy drops 1.5%; with beta=0 (scale-only), it drops 0.5%; replacing gamma with its training mean at test time drops accuracy by 65.4%, while the same for beta drops it by only 1.0% — gamma carries most of the conditioning weight. Restricting gamma to (0,1) via sigmoid, to (-1,1) via tanh, or to (0,inf) via exp all hurt — FiLM benefits from unrestricted sign and magnitude. Moving FiLM to different points in the residual block (including after the post-norm ReLU) barely changes performance, decoupling FiLM from normalization. Removing FiLM entirely from all 4 residual blocks collapses accuracy to 21.4%, showing FiLM is doing the conditioning work.

**4.4 CLEVR-Humans.** 18K human-written questions with new words/concepts. Fine-tuning only the FiLM generator (not the visual pipeline) yields 75.9% test accuracy, beating PG+EE by 9.3%.

**4.5 CLEVR-CoGenT.** Tests compositional generalization (in Condition A, cubes are gray/blue/brown/yellow; in B, swapped). FiLM generalizes better than PG+EE and is more sample-efficient at fine-tuning. A novel zero-shot trick computes (gamma, beta) for unseen concept combinations by linear combination of related questions ("How many cyan spheres?" + "How many brown cubes?" - "How many brown spheres?" → (gamma, beta) for "How many cyan cubes?"), giving a 3.2% overall and 9.2% per-applicable-question gain on B.

### 5. Conclusion
FiLM achieves strong visual reasoning purely via feature-wise affine conditioning. It is robust, decoupled from normalization, generalizes well, and the (gamma, beta) parameter space supports meaningful linear manipulations. Opens FiLM up for applications in RNNs and RL where normalization is less common.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** Pick a feature map (a 2-D channel of a CNN layer) — call its activation `x`. Then transform it pointwise to `y = gamma * x + beta`, where `gamma` and `beta` are two scalars chosen *separately for each channel* and *computed from a separate "conditioning" input* (here, a question). That's FiLM. The conditioning input runs through a small "FiLM generator" network to produce all the gammas and betas needed for every FiLM layer in the main network.

**Worked example.** Given the question "How many small purple cylinders or yellow rubber things are there?" plus a CLEVR image:
1. The question goes through a GRU and produces a single 4096-dim hidden vector `h_q`.
2. For each FiLM-ed ResBlock (4 blocks, 128 channels each), a learned linear layer maps `h_q` to a (gamma, beta) vector of size 256 (128 gammas + 128 betas) — that's `128 * 2 * 4 = 1024` modulation scalars total.
3. The CNN processes the 224x224 image down to 14x14 with 128 channels. Each ResBlock convolves, batch-norms-without-affine, applies its own learned (gamma, beta) per channel, ReLUs, and adds the residual.
4. A final classifier head reads out an answer.

If a different question came in, the same CNN weights would be reused, but the (gamma, beta) vectors would change, scaling some feature maps up, others down, and possibly flipping signs. The CNN's *behavior* — not its parameters — adapts to the question.

**Why the result is non-trivial.** Before this paper, the assumption was that hard visual-reasoning tasks like CLEVR needed compositional architectures — explicit symbolic program generators, neural module networks, or pairwise relation networks. FiLM uses none of those. The (gamma, beta) ranges learned by the model — large, mostly negative beta, a sharp gamma peak at zero — show the model is using FiLM to selectively gate, sign-flip, and amplify features, doing on-the-fly the work that hand-designed architectures had attempted to encode explicitly.

## Phase 2 — Graduate-level deep dive

### The FiLM equation

Let $F_{i,c} \in \mathbb{R}^{H \times W}$ denote the $c$-th feature map of the $i$-th example (within a CNN intermediate activation) and $x_i$ the conditioning input (e.g., a question). FiLM learns two arbitrary functions $f$ and $h$ — typically merged into a single "FiLM generator" $g$ — such that

$$
\gamma_{i,c} = f_c(x_i), \qquad \beta_{i,c} = h_c(x_i),
$$

and the per-feature-map affine transformation is

$$
\mathrm{FiLM}(F_{i,c} \mid \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c} \, F_{i,c} + \beta_{i,c}.
$$

Crucially, $\gamma_{i,c}$ and $\beta_{i,c}$ are **scalars** (not maps), so the modulation is **spatially uniform within a feature map but feature-map-specific** — it shapes the per-channel distribution of activations without depending on spatial location $(h,w)$. Parameter count per FiLM layer scales as $2C$ where $C$ is the number of feature maps, independent of $H, W$, giving the resolution-independence claim.

### FiLM generator in the visual-reasoning model

Let $q = (q_1, \ldots, q_T)$ be the tokenized question. A GRU produces a final hidden state

$$
h_q = \mathrm{GRU}(q) \in \mathbb{R}^{4096}.
$$

For each FiLM-ed ResBlock $\ell \in \{1, \ldots, L\}$ with $C_\ell$ channels, learned weight matrices $W^\gamma_\ell \in \mathbb{R}^{C_\ell \times 4096}$ and $W^\beta_\ell \in \mathbb{R}^{C_\ell \times 4096}$ produce

$$
\gamma_\ell = W^\gamma_\ell h_q + b^\gamma_\ell, \qquad \beta_\ell = W^\beta_\ell h_q + b^\beta_\ell.
$$

In a FiLM-ed ResBlock, given input feature tensor $X_{\ell-1}$, the forward pass is roughly

$$
\begin{aligned}
U_\ell &= \mathrm{Conv}_{1 \times 1}(X_{\ell-1}) \\
V_\ell &= \mathrm{Conv}_{3 \times 3}(U_\ell) \\
\hat V_\ell &= \mathrm{BN}_{\text{no-affine}}(V_\ell) \\
\tilde V_{\ell,c} &= \gamma_{\ell,c} \, \hat V_{\ell,c} + \beta_{\ell,c} \quad \forall c \\
X_\ell &= U_\ell + \mathrm{ReLU}(\tilde V_\ell).
\end{aligned}
$$

Here $\mathrm{BN}_{\text{no-affine}}$ refers to batch-norm with the per-channel affine $(\hat\gamma, \hat\beta)$ disabled, since FiLM is taking over the role of those parameters (this is why FiLM "generalizes" conditional batch norm — conditional BN is the special case where FiLM is glued immediately after BN's normalization step).

### Why FiLM generalizes Conditional Normalization

Batch normalization computes, per channel $c$,

$$
\hat F_{c} = \frac{F_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad \mathrm{BN}(F_c) = \hat\gamma_c \hat F_c + \hat\beta_c.
$$

Conditional Batch Norm (de Vries et al. 2017) and Conditional Instance Norm (Dumoulin et al. 2017) replace the *learned, input-independent* affine $(\hat\gamma_c, \hat\beta_c)$ with a *learned function of a conditioning input* $(\gamma_c(x_i), \beta_c(x_i))$. FiLM strips away the requirement that the affine must immediately follow normalization, applying $\gamma F + \beta$ at any chosen point in the network. The CLEVR ablations confirm placement is largely irrelevant — moving FiLM to "after the post-norm ReLU" still gives 97.7%.

### Comparison with neighboring conditioning mechanisms

| Mechanism | FiLM form |
|---|---|
| Concatenation of constant feature maps with conv input (Conditional DCGAN) | Equivalent to FiLM with $\gamma = 1$ (bias-only) |
| WaveNet / Conditional PixelCNN additive conditioning | Same — $\gamma = 1$ |
| LSTM / Squeeze-and-Excitation / sigmoid gating | FiLM with $\gamma \in (0,1)$, $\beta = 0$ |
| Conditional Batch/Instance/Layer Norm | FiLM placed immediately after a normalization step |
| Hypernetworks (Ha et al. 2016) | FiLM is a low-rank special case: one network outputs the per-channel affine parameters of another |

### Empirical anatomy of (gamma, beta)

From the validation set:
- $\gamma \in [-15, 19]$, sharp peak at $0$ (channel-off), 36% negative.
- $\beta \in [-9, 16]$, 76% negative.
- Test-time ablation: replacing $\gamma$ with its training mean drops accuracy by 65.4%; replacing $\beta$ similarly drops it by 1.0% — gamma is the load-bearing parameter.
- Restricting $\gamma \in (0,1)$ (sigmoid), $\gamma \in (-1,1)$ (tanh), or $\gamma \in (0,\infty)$ (exp) all hurt similarly, demonstrating that large magnitudes and negative gammas matter.

The combination of `gamma < 0` followed by ReLU is particularly interesting: it sign-flips the feature before half-wave rectification, so the model effectively gets to choose at runtime which half of the activation distribution survives — a form of conditional gating that single-sign multiplicative gates (sigmoid, exp) cannot express.

### Zero-shot generalization via FiLM-parameter algebra

Inspired by word-vector analogy ("King - Man + Woman = Queen"), the paper computes for an unseen attribute combination (e.g., "cyan cube" never appears in Condition A training):

$$
(\gamma, \beta)_{\text{cyan cube}} \approx (\gamma, \beta)_{\text{cyan sphere}} + (\gamma, \beta)_{\text{brown cube}} - (\gamma, \beta)_{\text{brown sphere}}.
$$

Plugging this synthesized (gamma, beta) into the unchanged CNN yields a 3.2% overall and a 71.5% → 80.7% per-applicable-question accuracy gain on Condition B, with no training for zero-shot at all — evidence that the FiLM parameter space has learned a quasi-disentangled, linearly-composable representation of attributes.

## Connections to other papers in this corpus

- **`dumoulin_2017_cond_instance_norm.md`** — Conditional Instance Norm is the direct precursor; FiLM is its generalization (drops the normalization-coupling requirement). Perez et al. cite it as a key inspiration and one of the unified family members.
- **`huang_belongie_2017_adain.md`** — AdaIN computes (gamma, beta) as the statistics of a style image rather than as the output of a learned hypernetwork. FiLM is the learned-generator counterpart.
- **`santurkar_2018_batchnorm_optimization.md`** — Explains why the normalization step that FiLM sometimes sits next to is helpful (loss-landscape smoothing); FiLM's ablation that "no-batch-normalization → 93.7%" connects to this.
- **`birnbaum_2019_temporal_film.md`** — Temporal FiLM extends the FiLM idea to long-range sequence modeling, producing (gamma, beta) that vary along the time axis, citing Perez 2018 as the foundational reference.
- **`wisnu_2025_stsm_film.md`** — STSM-FiLM applies FiLM to time-scale modification of speech; the modulation signal is a tempo factor rather than a question.
- **Other corpus members not in this batch** — `Ha et al. 2016 - HyperNetworks`, `de Vries et al. 2017 - Modulating early visual processing by language`, and `Turkoglu et al. 2022 - FiLM-Ensemble` are all directly tied to this paper (hypernetwork generalization, CBN precursor, and ensembling extension respectively).
