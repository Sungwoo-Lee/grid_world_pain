---
title: "A Learned Representation For Artistic Style"
authors: Vincent Dumoulin, Jonathon Shlens, Manjunath Kudlur
year: 2017
venue: ICLR 2017
slug: dumoulin_2017_cond_instance_norm
source_pdf: sources/Dumoulin et al. 2017 - A Learned Representation For Artistic Style.pdf
topic: FiLM
---

# Dumoulin et al. 2017 — A Learned Representation For Artistic Style (Conditional Instance Normalization)

## Plain-English entry point

This paper introduces **Conditional Instance Normalization (CIN)** — the direct ancestor of FiLM and the moment when the field realized that a tiny set of per-channel scale-and-shift parameters can carry an enormous amount of *semantic* information (here, the identity of an entire painting style). The setting is **neural style transfer**, the task of repainting a content photograph (e.g., the Golden Gate Bridge) in the visual style of a reference painting (e.g., a Van Gogh). Before this paper, the dominant approach was to train *one feed-forward "style transfer network" per style*: a 1.6-million-parameter network entirely dedicated to one painting. That made on-device deployment of multi-style apps impossibly expensive.

The authors ask: what if almost all 1.6M weights are *shared* across all styles, and only a tiny per-style "specialization" is swapped in for each painting? They build this specialization on top of **instance normalization** — a normalization layer (similar to batch norm but per-image and per-channel rather than per-batch) that subtracts the spatial mean and divides by the spatial standard deviation of each feature map. Instance norm normally has a single learned scale (gamma) and shift (beta) per channel. Their twist: replace those with an `N x C` lookup table where row `s` holds the (gamma, beta) vector for style `s`. The convolutional backbone stays shared; only the row gets swapped.

The result is striking: 32 visually wildly different painting styles fit in one network, with only **0.2% of parameters** unique per style, and the resulting pastiches match those of single-style networks in quality. Even better, the learned (gamma, beta) vectors form an **embedding space** — linearly interpolating between them produces smooth, sensible blends of styles, and adding a new style requires fine-tuning only its tiny (gamma, beta) row, not the whole network. This embedding-of-styles-in-affine-parameters idea is the seed that grew into FiLM.

## Section-ordered backbone

### Abstract
A single feed-forward network can model a diversity of painting styles by reducing each style to a point in an embedding space. New styles can be obtained by combining learned ones.

### 1. Introduction
Reviews the history of style transfer (Efros-Leung non-parametric texture sampling; image analogies; Gatys et al.'s neural style transfer using VGG-19 features; Johnson/Ulyanov/Li-Wand's feed-forward transfer networks). Notes the key limitation: one network per style. Memory-limited mobile devices, and the implicit sharing across visually-similar styles, motivate a single conditional network.

### 2. Style Transfer with Deep Networks
Recaps the Gatys formulation: a pastiche image $p$ should be close in **content** (high-level VGG features) to a content image $c$ and close in **style** (Gram-matrix statistics of low-level VGG features) to a style image $s$. The combined loss is

$$
L(s, c, p) = \lambda_s L_s(p) + \lambda_c L_c(p),
$$

where the style loss compares Gram matrices of activations layer-by-layer and the content loss compares activations directly. A feed-forward style-transfer network $T$ is trained to map $c \to p$ via this same loss.

### 2.1 N-Styles Feed-Forward Style Transfer Networks
**Key claim and surprise:** to model a style, it suffices to specialize *only* the scaling-and-shifting parameters after a normalization step to each specific style. All convolutional weights can be shared. The mechanism: instance norm with `N x C` matrices for (gamma, beta), indexed by style label `s`. Includes the formula (Eq. 5; reproduced in the deep dive below) and notes that style transfer to N styles requires only `O(N * L)` extra parameters total (L = number of feature maps across the network). For a typical 1.6M-parameter network, only ~3K parameters specify individual styles.

### 3. Experimental Results

**3.1 Methodology.** Architecture follows Johnson et al. 2016 (5 residual blocks at 128 channels, 32->64->128 down-sampling, 64->32 up-sampling via nearest-neighbor + conv, output sigmoid). Mirror-padding and nearest-neighbor upsampling avoid border and checkerboard artifacts. Trained on ImageNet content images with Adam, 40K parameter updates, batch size 16.

**3.2 N-styles matches independent models.** Trained on 10 Monet impressionist paintings: content loss is 8.7±3.9% higher than single-style networks, style loss insignificantly different (8.9±16.5%). Visual quality is comparable.

**3.3 Flexibility across very different styles.** A single network captures all 32 stylistically-diverse paintings (impressionists, expressionists, cubists, ...) shown in Figure 1a.

**3.4 Generalization across styles.** Fine-tuning only the (gamma, beta) row for a new painting (Monet's *Plum Trees in Blossom*) converges much faster than training from scratch. After 5K fine-tune steps the new style matches a from-scratch model after 40K steps.

**3.5 Combining painting styles.** Linear interpolation between two styles' (gamma, beta) values produces smooth pastiches:

$$
\gamma = \alpha \gamma_1 + (1-\alpha) \gamma_2, \qquad \beta = \alpha \beta_1 + (1-\alpha) \beta_2.
$$

Style loss with respect to each endpoint varies monotonically with $\alpha$, confirming the linearly-blendable structure. Convex combinations over four corner styles produce a 2-D grid of intermediate pastiches.

### 4. Discussion
The fact that 0.2% of parameters dominate style identity is surprising. The interpretation offered: convolutional weights encode universal "elements of style" (paint-stroke filters, color-channel-mixing patterns); the per-channel affines select *which* elements each style activates. The paper raises future directions: predicting (gamma, beta) directly from a style image (which would later become AdaIN), and generative models over the embedding space.

### Appendix
Architecture details (Table 1) and pastiches for all 10 Monet and 32 varied styles.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** Take a normalization layer that divides each feature map by its spatial mean and standard deviation — its only learned parameters are a per-channel scale (gamma) and shift (beta). Now make those two scalars depend on a discrete style label: store an `N x C` table of (gamma, beta) entries, and at runtime look up the row for the desired style. Everything else in the network is shared across all N styles.

**Worked example.** Suppose you want one neural network that can stylize any photograph in any of 32 famous painting styles.
1. Build a feed-forward style-transfer network (5 residual blocks, ~1.6M parameters).
2. After every convolution, apply *instance normalization*: subtract the spatial mean and divide by spatial standard deviation, per feature map.
3. Then, instead of a single learned (gamma, beta) pair per channel, allocate a table with 32 rows (one per style) and C columns (one per channel).
4. To produce a Van-Gogh-styled image, fetch the Van Gogh row and use those gammas and betas in every CIN layer.
5. To produce a Lichtenstein-styled image with the *same* network weights, fetch the Lichtenstein row.

The whole 32-style network costs only ~3K extra parameters beyond a single-style network — a fraction of a percent of the model size. To add a 33rd style, freeze the entire backbone and learn just a 33rd (gamma, beta) row — a few thousand scalars total.

**Why the result is non-trivial.** Before this paper, conventional wisdom said a network's "knowledge" lives in its convolutional weights. This paper proved that a substantial fraction of *semantic* content (the visual identity of a Monet vs. a Klee vs. a Toulouse-Lautrec) lives in a handful of channel-wise affine parameters sitting on top of identical convolutions. That insight motivated FiLM (Perez 2018), AdaIN (Huang & Belongie 2017), Conditional Batch Norm (de Vries 2017), and the whole family of "feature-wise modulation" methods.

## Phase 2 — Graduate-level deep dive

### Instance normalization recap

For an input feature map $x_{n, c, h, w}$ (batch $n$, channel $c$, spatial position $h, w$), instance normalization (Ulyanov et al. 2016b) computes spatial statistics *per image, per channel*:

$$
\mu_{n,c} = \frac{1}{HW} \sum_{h,w} x_{n,c,h,w}, \qquad
\sigma_{n,c}^2 = \frac{1}{HW} \sum_{h,w} (x_{n,c,h,w} - \mu_{n,c})^2,
$$

then standardizes and applies a learned per-channel affine:

$$
\mathrm{IN}(x)_{n,c,h,w} = \gamma_c \cdot \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}} + \beta_c.
$$

The normalization is intentionally *per-image* (unlike batch norm, which averages across the batch) because in style transfer the per-image contrast statistics are themselves a form of style content that should be removed before re-injecting a target style's affine parameters.

### Conditional Instance Normalization (CIN)

Dumoulin et al.'s modification: replace the input-independent $(\gamma_c, \beta_c)$ with a *style-conditional* pair $(\gamma_{s,c}, \beta_{s,c})$, where $s \in \{1, \ldots, N\}$ is the style label. Concretely, we store learned matrices $\Gamma \in \mathbb{R}^{N \times C}$ and $\mathbf{B} \in \mathbb{R}^{N \times C}$ and define

$$
\mathrm{CIN}(x \mid s)_{n,c,h,w} = \gamma_{s,c} \cdot \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}} + \beta_{s,c},
$$

where $\gamma_{s,c} = \Gamma_{s,c}$ and $\beta_{s,c} = \mathbf{B}_{s,c}$ are simply the $s$-th row of the learned tables. The paper writes this compactly (Eq. 5 in the original) as

$$
z = \gamma_s \left( \frac{x - \mu}{\sigma} \right) + \beta_s.
$$

### Parameter counting

For a network with $L$ total feature maps summed across all CIN layers (e.g., $L \approx 1500$ for the Johnson et al. backbone with 5 residual blocks at 128 channels plus down/up-sampling stages), the style-specialization parameters number

$$
P_{\text{style}}(N) = 2 \cdot N \cdot L.
$$

For $N = 32$ styles and the architecture used in the paper, $P_{\text{style}} \approx 2 \cdot 32 \cdot 1500 \approx 96{,}000$ — but the *per-style* increment is $2L \approx 3000$, which is the figure quoted as "0.2% of 1.6M". The shared backbone parameters $P_{\text{shared}} \approx 1.6 \times 10^6$ dominate the total. Compare with $N$ independent networks at $1.6 \times 10^6 N$ parameters — savings scale linearly with $N$.

### Loss and training procedure

The conditional network $T(c, s)$ is trained on a stream of (content, style-label) pairs drawn from ImageNet content images and a fixed gallery of $N$ style images. The full loss aggregates Gatys-style content and style terms over all $N$ styles:

$$
L = \mathbb{E}_{c} \left[ \frac{1}{N} \sum_{s=1}^{N} \left( \lambda_s L_s(T(c, s), s) + \lambda_c L_c(T(c, s), c) \right) \right],
$$

with the per-style component-wise losses

$$
L_s(p, s) = \sum_{i \in S} \frac{1}{U_i} \|G(\phi_i(p)) - G(\phi_i(s))\|_F^2, \qquad
L_c(p, c) = \sum_{j \in C} \frac{1}{U_j} \|\phi_j(p) - \phi_j(c)\|_2^2,
$$

where $\phi_l$ are VGG-16 features at layer $l$, $G(\cdot)$ is the Gram matrix, $U_l$ is the number of activations at layer $l$, and $S, C$ are the chosen style/content layer sets. A useful implementation trick: with batch size $N$, a single forward pass can produce one image stylized in every style simultaneously.

### Linear interpolation in (gamma, beta) space

Given two style indices $s_1, s_2$ with learned $(\gamma_{s_1}, \beta_{s_1})$ and $(\gamma_{s_2}, \beta_{s_2})$, blended (gamma, beta) at mixing weight $\alpha \in [0,1]$ are

$$
\gamma(\alpha) = \alpha \, \gamma_{s_1} + (1 - \alpha) \, \gamma_{s_2}, \qquad
\beta(\alpha) = \alpha \, \beta_{s_1} + (1 - \alpha) \, \beta_{s_2}.
$$

Plug this $(\gamma(\alpha), \beta(\alpha))$ pair into every CIN layer of the network and run the forward pass. The result interpolates *smoothly* between the two pastiches, and the Gatys-style style loss with respect to each endpoint varies *monotonically* in $\alpha$ (Figure 7 right). This is the empirical evidence that the affine-parameter space is meaningfully linearly composable — the conceptual precursor to FiLM's "(gamma, beta) algebra" (zero-shot generalization via parameter-vector analogy in Perez 2018).

### Incremental style addition

To add an $(N+1)$-th style:
1. Freeze the backbone (all conv weights, all existing $(\gamma_{s}, \beta_{s})$ rows).
2. Initialize $(\gamma_{N+1}, \beta_{N+1})$, e.g., from a random or mean style.
3. Train only the new row on the Gatys content+style loss for the new style image.

Figure 6 shows convergence is ~8x faster than training a new single-style network from scratch, and pastiches of equivalent quality emerge after 5K vs. 40K parameter updates.

### Why instance norm rather than batch norm

The choice of instance norm (rather than batch norm or layer norm) is load-bearing: instance norm strips out per-image, per-channel mean and variance — exactly the kind of luminance/contrast statistics that constitute one major axis of "style". By zeroing these out, the conditional affine $(\gamma_s, \beta_s)$ is then injecting *only the target style's* statistics back in, with the shared convolutional backbone providing the structural feature filters. This insight is what AdaIN (Huang & Belongie 2017) later makes fully explicit: $\gamma_s$ and $\beta_s$ can in fact be computed directly as the spatial statistics of a style image, removing the need for an embedded lookup table altogether.

## Connections to other papers in this corpus

- **`perez_2018_film.md`** — Perez et al. explicitly identify Conditional Instance Norm as a special case of FiLM (FiLM placed immediately after instance norm). The (gamma, beta) interpolation idea here directly inspires FiLM's "(gamma, beta) algebra" for zero-shot generalization.
- **`huang_belongie_2017_adain.md`** — AdaIN drops the learned lookup table entirely and instead computes (gamma, beta) on-the-fly as the channel-wise mean and standard deviation of a style image's deep features, enabling arbitrary (not pre-trained) style transfer.
- **`santurkar_2018_batchnorm_optimization.md`** — Same year as this paper's normalization-as-affine-substrate insight; complements it by analyzing *why* normalization helps optimization in the first place.
- **`birnbaum_2019_temporal_film.md`** — Generalizes the (gamma, beta) idea to time-varying modulation in sequence models.
- **`wisnu_2025_stsm_film.md`** — Applies FiLM-style conditioning to audio time-scale modification; conceptually downstream of this CIN root.
