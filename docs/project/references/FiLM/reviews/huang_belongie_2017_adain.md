---
title: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
authors: Xun Huang, Serge Belongie
year: 2017
venue: ICCV 2017
slug: huang_belongie_2017_adain
source_pdf: sources/Huang and Belongie 2017 - Arbitrary style transfer in real-time with adaptive instance normalization.pdf
topic: FiLM
---

# Huang and Belongie 2017 — Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (AdaIN)

## Plain-English entry point

This paper introduces **AdaIN** (Adaptive Instance Normalization), the simplest and most surprising member of the FiLM family: a feature-wise affine modulation layer with **zero learnable parameters**. Where Conditional Instance Norm (`dumoulin_2017_cond_instance_norm.md`) stores a lookup table of (gamma, beta) vectors — one row per painting style learned during training — AdaIN computes its (gamma, beta) **on the fly** as the **per-channel mean and standard deviation of a deep-feature representation of a style image**. The consequence is dramatic: a single network can stylize a content photograph in **any** painting style at test time, including styles the network has never seen, at roughly 15 frames per second on a 512x512 image.

The technical claim is grounded in a clean reinterpretation of why instance normalization works for style transfer in the first place. Instance norm strips out each image's per-channel feature statistics — and those statistics, by the influential work of Gatys et al. and Li et al., **are** the style. So instance norm performs "style normalization": it removes the input's style. The natural next step: re-inject the *target* style's statistics by aligning the content features' channel-wise mean and variance to those of a style image. That's AdaIN, in one line: `(content_feature - mean_content) / std_content * std_style + mean_style`.

The paper bolts AdaIN between a frozen VGG-19 encoder (up to relu4_1) and a learned decoder trained to invert the AdaIN output back to the image space. The decoder uses **no normalization layers** — that's load-bearing, because instance norm or batch norm in the decoder would force a single output style and undo AdaIN's flexibility. The runtime supports content-style trade-off, style interpolation, color preservation, and spatial control, all without modifying training. AdaIN is roughly 720x faster than Gatys et al.'s optimization-based original and the first method to deliver real-time, arbitrary, learned style transfer.

## Section-ordered backbone

### Abstract
Introduces AdaIN, an instance-norm extension that aligns the channel-wise mean and variance of content features to those of style features. Achieves real-time arbitrary style transfer at speeds comparable to fixed-style feed-forward methods. Provides runtime controls (content-style trade-off, interpolation, color preservation, spatial masks) without re-training.

### 1. Introduction
Reviews the speed-vs-flexibility tension: Gatys et al.'s neural style transfer is flexible (any style at test time) but optimization-based and slow (minutes per image); Johnson/Ulyanov/Li-Wand feed-forward networks are fast but locked to one style; Dumoulin et al.'s Conditional Instance Norm handles 32 styles in one network but still can't generalize to unseen styles; Chen and Schmidt's "style swap" can handle arbitrary styles but is slow due to a patch-matching bottleneck. AdaIN claims to break this trilemma.

### 2. Related Work
Covers (a) style transfer history (texture synthesis, Gatys et al., feed-forward acceleration, multi-style networks); (b) deep generative models (VAEs, auto-regressive, GANs, cross-domain image generation).

### 3. Background

**3.1 Batch Normalization.** Recap: for input $x \in \mathbb{R}^{N \times C \times H \times W}$,
$$\mathrm{BN}(x) = \gamma \frac{x - \mu(x)}{\sigma(x)} + \beta,$$
where $\mu, \sigma$ are computed *across batch and spatial dimensions per channel*. Uses minibatch stats during training, population stats at inference.

**3.2 Instance Normalization (IN).** Identical equation but $\mu, \sigma$ are computed *per sample, per channel, across spatial dimensions only*. Has no train/test discrepancy. Ulyanov et al. found that simply replacing BN with IN dramatically improved feed-forward style transfer quality.

**3.3 Conditional Instance Normalization (CIN).** Dumoulin et al.'s style-indexed lookup table for (gamma, beta) — same as `dumoulin_2017_cond_instance_norm.md`.

### 4. Interpreting Instance Normalization
The interpretive contribution. Ulyanov's original explanation was that IN's effectiveness comes from contrast invariance. The authors test this with a three-way experiment:
- (a) IN beats BN substantially on original MS-COCO images.
- (b) When training images are first contrast-normalized (luminance histogram-equalized), IN still beats BN — so contrast invariance is not the full story.
- (c) When training images are first **style-normalized** (passed through a pre-trained feed-forward style transfer network), the IN-vs-BN gap shrinks dramatically.

This evidence reframes IN as performing **style normalization** in feature space — removing the per-channel feature statistics that, by Gatys et al. and Li et al., encode style. The (gamma, beta) of IN/CIN are then injecting target style statistics back in. The same explanation generalizes CIN: different (gamma_s, beta_s) values normalize different feature statistics, i.e., different output styles.

### 5. Adaptive Instance Normalization
If CIN normalizes to a finite set of styles via learned (gamma_s, beta_s), can we do it for *any* style by computing (gamma, beta) directly from a style image? Yes — and the natural choice is to use the style image's own per-channel mean and standard deviation in deep feature space:

$$\mathrm{AdaIN}(x, y) = \sigma(y) \cdot \frac{x - \mu(x)}{\sigma(x)} + \mu(y).$$

No learnable parameters in the AdaIN layer itself. Intuition: a feature channel that responds to a certain brushstroke will activate strongly on a style image with that brushstroke; AdaIN forces the content image's feature map at that channel to have the same average activation while preserving the spatial structure of the content image. Spatially homogeneous statistics map cleanly to a notion of style; the variance of each channel adds finer style detail.

### 6. Experimental Setup

**6.1 Architecture.** Content image $c$ and style image $s$ are encoded by a fixed VGG-19 (up to relu4_1) to give $f(c)$ and $f(s)$. AdaIN produces target feature maps $t = \mathrm{AdaIN}(f(c), f(s))$. A learned decoder $g$ mirrors the encoder (with pooling replaced by nearest-neighbor upsampling, reflection padding) and inverts $t$ back to the pixel-space stylized image $T(c, s) = g(t)$. **No normalization layers in the decoder** — IN or BN in the decoder would normalize the output to a single style, undoing AdaIN's whole point.

**6.2 Training.** MS-COCO as content images, WikiArt as style images (~80K each), Adam, batch size 8, 256x256 random crops from 512-resized inputs. Loss uses pre-trained VGG-19 features:
$$L = L_c + \lambda L_s,$$
$$L_c = \|f(g(t)) - t\|_2,$$
$$L_s = \sum_{i=1}^{L} \|\mu(\phi_i(g(t))) - \mu(\phi_i(s))\|_2 + \|\sigma(\phi_i(g(t))) - \sigma(\phi_i(s))\|_2,$$
with $L_s$ summing over $\phi_i$ = relu1_1, relu2_1, relu3_1, relu4_1. Note that $L_c$ matches the *AdaIN output* (not the raw content image features) — conceptually cleaner and slightly faster to converge. The style loss matches channel-wise mean and variance directly (the "IN statistics") rather than the Gram matrix — produces similar results.

### 7. Results

**7.1 Comparisons.** Visually competitive with Gatys et al.'s optimization-based and Ulyanov's single-style feed-forward methods, despite never having seen the test styles. Better than Chen and Schmidt's patch-matching style-swap, especially when most content patches map to non-representative style patches. Speed (256px / 512px in seconds): Gatys 14.17 / 46.75; Chen-Schmidt 0.171 / 3.214; AdaIN 0.018 / 0.065; CIN (32 styles) 0.011 / 0.038; single-style feed-forward 0.011 / 0.038. AdaIN is ~720x faster than Gatys, ~50x faster than Chen-Schmidt, and only ~2x slower than the most restrictive (single-style) baselines.

**7.2 Architecture ablations.** AdaIN cleanly beats: (a) concatenating content+style as a baseline fusion strategy (content leaks the style image's object outlines through into the output); (b) using BN in the decoder; (c) using IN in the decoder. IN-in-decoder is the worst because it normalizes away the style AdaIN just injected.

**7.3 Runtime controls.**
- **Content-style trade-off**: linearly interpolate feature maps fed to the decoder: $g((1 - \alpha) f(c) + \alpha \mathrm{AdaIN}(f(c), f(s)))$. Equivalent to interpolating AdaIN's affine parameters between identity and full style.
- **Style interpolation**: convex combination of multiple styles: $g(\sum_k w_k \mathrm{AdaIN}(f(c), f(s_k)))$.
- **Color preservation**: pre-match the style image's color distribution to the content image.
- **Spatial control**: apply different AdaIN statistics to different content regions via masks.

### 8. Discussion and Conclusion
Contrasts AdaIN with optimization-based and patch-matching methods. AdaIN aligns feature statistics in one shot and inverts back to pixels. Lists future directions: residual or skip-connection architectures, incremental training, replacing the simple mean+variance matching with correlation alignment or histogram matching for higher-order statistics, and applying AdaIN to texture synthesis.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** Pass both a content photo and a style image through a frozen VGG-19. At the relu4_1 layer, compute the **per-channel mean and standard deviation** of the style image's activations, then **rescale and shift** the content image's activations so they share the same per-channel mean and standard deviation. Feed those modified content activations through a learned decoder that paints them back into a stylized image.

**Worked example.** Suppose you want to stylize a Golden Gate Bridge photo in the style of Van Gogh's *The Starry Night*:
1. Run the bridge photo through VGG-19 up to relu4_1. You get a feature tensor with shape `[1, 512, H, W]`.
2. Run *The Starry Night* through the same VGG-19 up to relu4_1.
3. For each of the 512 channels, compute the spatial mean and standard deviation of the Starry-Night activations — that's a `[512]` vector of means and a `[512]` vector of standard deviations.
4. For each of the 512 channels of the bridge features, subtract its own spatial mean, divide by its own spatial standard deviation, then multiply by Starry-Night's std and add Starry-Night's mean.
5. Pass the modified tensor through a learned decoder. Out comes a Van-Gogh-styled bridge.

To restyle the same bridge in a Lichtenstein pop-art print: repeat steps 2–5 with the Lichtenstein image instead. No retraining, no per-style fine-tuning — just a fresh forward pass through VGG-19 to extract the style statistics.

**Why the result is non-trivial.** Previous "fast" style transfer methods learned one network per style or stored a lookup table of styles. AdaIN proves that the same encoder-decoder can handle infinitely many styles because the "style identity" lives entirely in two simple statistics (mean and std) of a feature tensor — a 2x512 = 1024-scalar fingerprint per style, computed instantly by a single forward pass through a fixed VGG-19. That's the same insight that makes FiLM (`perez_2018_film.md`) work: the channel-wise (gamma, beta) is enormously informative.

## Phase 2 — Graduate-level deep dive

### Recap of normalization variants

Let $x \in \mathbb{R}^{N \times C \times H \times W}$ be a feature tensor.

**Batch normalization.** Per-channel statistics computed across batch and spatial axes:
$$\mu_c^{\mathrm{BN}}(x) = \frac{1}{NHW} \sum_{n,h,w} x_{n,c,h,w}, \qquad
\sigma_c^{\mathrm{BN}}(x) = \sqrt{\frac{1}{NHW} \sum_{n,h,w} (x_{n,c,h,w} - \mu_c^{\mathrm{BN}})^2 + \epsilon}.$$

**Instance normalization.** Per-sample, per-channel statistics across spatial axes:
$$\mu_{n,c}^{\mathrm{IN}}(x) = \frac{1}{HW} \sum_{h,w} x_{n,c,h,w}, \qquad
\sigma_{n,c}^{\mathrm{IN}}(x) = \sqrt{\frac{1}{HW} \sum_{h,w} (x_{n,c,h,w} - \mu_{n,c}^{\mathrm{IN}})^2 + \epsilon}.$$

Both apply an output affine $\gamma_c \hat x + \beta_c$ with learned $(\gamma_c, \beta_c)$. CIN replaces the input-independent $(\gamma_c, \beta_c)$ with a style-indexed $(\gamma_{s,c}, \beta_{s,c})$ via an embedding table.

### The AdaIN equation

AdaIN takes a content feature tensor $x$ and a style feature tensor $y$ (both produced by the same VGG-19 encoder) and computes

$$
\boxed{\;\mathrm{AdaIN}(x, y)_{n,c,h,w} = \sigma_{n,c}^{\mathrm{IN}}(y) \cdot \frac{x_{n,c,h,w} - \mu_{n,c}^{\mathrm{IN}}(x)}{\sigma_{n,c}^{\mathrm{IN}}(x)} + \mu_{n,c}^{\mathrm{IN}}(y).\;}
$$

In the FiLM/CIN notation, the equivalent (gamma, beta) are

$$
\gamma_{n,c}^{\mathrm{AdaIN}} = \sigma_{n,c}^{\mathrm{IN}}(y), \qquad
\beta_{n,c}^{\mathrm{AdaIN}} = \mu_{n,c}^{\mathrm{IN}}(y) - \sigma_{n,c}^{\mathrm{IN}}(y) \cdot \frac{\mu_{n,c}^{\mathrm{IN}}(x)}{\sigma_{n,c}^{\mathrm{IN}}(x)}.
$$

Crucially, $(\gamma^{\mathrm{AdaIN}}, \beta^{\mathrm{AdaIN}})$ depend on **both** the content and the style — unlike CIN where (gamma, beta) depend only on the style index. But the $\sigma(y), \mu(y)$ contribution is the "style fingerprint", and rewriting AdaIN as "standardize content, then re-apply style stats" makes clear that the per-channel content statistics are simply being replaced.

### Why no normalization in the decoder

The decoder receives target feature maps $t = \mathrm{AdaIN}(f(c), f(s))$ whose per-channel statistics match the *style*. If the decoder contained instance-norm layers, those layers would re-standardize $t$ to zero mean and unit variance per channel, then re-apply their *own* learned (gamma, beta) — destroying the carefully-injected style statistics and forcing the decoder's affines to dominate the output style. Batch norm in the decoder is similarly bad: a single learned (gamma, beta) per channel forces all outputs toward one statistic, and the decoder cannot represent the full variety of styles it should produce. The ablation in Fig. 5–6 confirms both BN and IN in the decoder hurt qualitatively and quantitatively.

### Loss function in detail

Style is matched via channel-wise mean and standard deviation differences at multiple VGG-19 layers — equivalent to the Gram matrix loss up to off-diagonal terms, but conceptually closer to "matching IN statistics" and slightly cheaper:

$$
L_s = \sum_{i=1}^{L} \left( \|\mu(\phi_i(g(t))) - \mu(\phi_i(s))\|_2 + \|\sigma(\phi_i(g(t))) - \sigma(\phi_i(s))\|_2 \right).
$$

Content is matched against the AdaIN output $t$ rather than the raw content features $f(c)$:

$$
L_c = \|f(g(t)) - t\|_2.
$$

This is consistent with the decoder's job of inverting $t$ back to pixels and slightly accelerates convergence.

### Runtime controls as parameter-space operations

**Content-style trade-off.** The interpolation $g((1-\alpha) f(c) + \alpha \mathrm{AdaIN}(f(c), f(s)))$ is *equivalent* to feeding the decoder an AdaIN with shrunken affine parameters:

$$
\gamma^{\alpha} = (1-\alpha) + \alpha \sigma(f(s)) / \sigma(f(c)), \qquad
\beta^{\alpha} = \alpha (\mu(f(s)) - \sigma(f(s)) \mu(f(c))/\sigma(f(c))).
$$

**Style interpolation across K styles.** The weighted blend
$$
T(c, s_1, \ldots, s_K, w_1, \ldots, w_K) = g\left(\sum_{k=1}^{K} w_k \mathrm{AdaIN}(f(c), f(s_k))\right),
\quad \sum_k w_k = 1,
$$
is linear in the feature space and produces smooth blends — analogous to the linear (gamma, beta) interpolation in CIN, but now over *arbitrary* styles rather than only the 32 trained ones.

### Relationship to FiLM

In Perez et al.'s FiLM taxonomy, AdaIN is a FiLM layer where:
- The conditioning input is a style image $s$.
- The FiLM generator is the deterministic, parameter-free function $(\mu, \sigma) \circ \phi_{\text{relu4\_1}}$ applied to $s$, i.e., a frozen VGG-19 followed by per-channel mean and variance.
- The placement is *immediately after instance normalization* of the content features.

CIN is FiLM with a *learned embedding-table* generator. FiLM proper (Perez 2018) is FiLM with a *learned neural-network* generator (a GRU plus linear projection). AdaIN is FiLM with a *no-training, statistics-only* generator. All three sit on the same axis.

## Connections to other papers in this corpus

- **`dumoulin_2017_cond_instance_norm.md`** — Direct predecessor. AdaIN drops the learned (gamma_s, beta_s) lookup table and replaces it with on-the-fly per-channel statistics of a style image's deep features, enabling arbitrary (not pre-trained) styles.
- **`perez_2018_film.md`** — Perez et al. cite AdaIN as a member of the unified FiLM family (a style-conditional CN method); AdaIN is the parameter-free, statistics-based corner of the FiLM design space.
- **`santurkar_2018_batchnorm_optimization.md`** — Gives the loss-landscape rationale for why normalization (BN and IN both) helps training in the first place; complementary to AdaIN's *interpretive* claim that IN performs style normalization specifically.
- **`birnbaum_2019_temporal_film.md`** and **`wisnu_2025_stsm_film.md`** — Extend FiLM-style conditioning to the temporal domain; AdaIN's statistic-based modulation is conceptually upstream of any "natural-statistics-derived" conditioning generator.
