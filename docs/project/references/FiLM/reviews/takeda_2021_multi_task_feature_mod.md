---
title: "Training of Multiple and Mixed Tasks with a Single Network Using Feature Modulation"
authors: ["Mana Takeda", "Gibran Benitez", "Keiji Yanai"]
year: 2021
venue: "ICPR 2020 Workshops (Springer LNCS 12662)"
slug: takeda_2021_multi_task_feature_mod
source_pdf: "docs/project/references/FiLM/sources/Takeda et al. 2021 - Training of multiple and mixed tasks with a single network using feature modulation.pdf"
topic: FiLM
---

## Plain-English Entry Point

In image-to-image translation — turning a noisy photo into a clean one, a photo into a Van Gogh painting, a photo into its segmentation mask — the standard approach is one network per task. Multi-task learning (MTL) tries to do all tasks with one network, usually by **sharing the encoder** (the front half that compresses the image) and giving each task its own **decoder** (the back half that decodes to the output). That cuts costs partially, but task-specific decoders are still about half the network, so parameter count grows linearly with the number of tasks.

This paper proposes an alternative: **share both the encoder and decoder** entirely; switch tasks by feeding a small **task conditional vector** $c$ (a one-hot or mixture vector over tasks) through a **FiLM generator** that produces per-task per-channel scale-and-shift parameters $\gamma, \beta$. Those modulate the activations after every Instance Normalization layer (the combination IN+FiLM equals AdaIN, the standard style-transfer modulator). The only task-specific parameters are the FiLM generator's first FC layer; everything else is shared. With 6 tasks (reconstruction, inpainting, denoising, semantic segmentation, two artistic styles), the proposed network achieves better results than the prior SGN and Piggyback baselines while being about 6× smaller than running 6 independent networks.

Second, the paper tries **mixed-task learning** — combining heterogeneous tasks like "denoise then style-transfer" or "style-transfer only on horses". They show that simply training individual tasks and providing a mixture vector at inference time does *not* produce a clean mixture; nor does training with summed loss. What works is **explicitly creating synthetic mixed-task training samples** (e.g., apply denoising then style transfer in sequence to produce a target, then train on it). With these samples and L2 loss, the network smoothly interpolates over the **2D linear-combination space** of two tasks by varying the conditional vector at inference.

For this project, this paper is the **clearest demonstration that one FiLM-modulated backbone can host functionally diverse behaviors** — pixel-level segmentation versus stylization versus identity — and that the conditional vector smoothly mixes them. It is conceptually adjacent to the FiLM-Ensemble paper (Turkoglu 2022 in this batch), but on the **task-modulation** axis rather than the **member-modulation** axis.

## Section-Ordered Backbone

**Abstract.** Multi-task image translation with a *fully* shared (encoder + decoder) network; task-specific modulation via FiLM. Also tackles "mixed-task learning" (combining heterogeneous tasks like denoising + style transfer). Outperforms SGN and Piggyback baselines despite being smaller.

**1. Introduction.** Conventional MTL image translation uses shared encoder + per-task decoders → task-specific parameter cost grows with task count. Activation distributions differ per task; FiLM can adjust them. The paper introduces *mixed-task learning*, where a single network executes a *combination* of tasks specified by a mixture conditional vector. Two mixing types: (1) **sequential mixing** (apply task A then task B); (2) **mixing by region masking** (use task A to identify a region; apply task B only there).

**2. Related Work.**

- **MTL for image translation** — Cross-Stitch (Misra 2016), UberNet (Kokkinos 2017), Maninis 2019 (task-specific residual adapters + Squeeze-and-Excitation), Strezoski 2019 (task-specific binary masks per channel), Bragman 2019 (stochastic filter groups), MTAN (Liu 2019 attention). All require non-trivial task-specific parameters.
- **No prior mixed-task learning with heterogeneous tasks.** Closest: SGN (Chang 2019) which mixes style/domain transfer (both similar tasks) via dynamic weighted loss.
- **Continual learning** — Piggyback (Mallya 2018) freezes a base network and learns task-specific binary weight masks. Can handle multiple tasks but not mixed-task inference.
- **FiLM** — formalization of conditional affine transformations (Perez 2018, Dumoulin 2018 distill.pub). Recap: $\gamma_i = f_{\gamma, i}(c)$, $\beta_i = f_{\beta, i}(c)$, $\text{FiLM}(F_i \mid \gamma_i, \beta_i) = \gamma_i F_i + \beta_i$. Dumoulin 2017 showed FiLM can mix multiple **styles** via a mixture vector — the paper's mixed-task work extends this from styles to heterogeneous tasks.

**3. Proposed Method.**

- **3.1 Task conditional vector.** $n$-dim vector $c = [c_1, \ldots, c_n]$. Training: $c_i \in \{0, 1\}$ (one-hot for single tasks). Inference: $c_i \in [0, 1]$ (mixtures). Crucially, $c = \mathbf{0}$ encodes the **identity transformation** (Task 0 is reconstruction); this allows the conditional vector to control the *degree* of transformation continuously.
- **3.2 FiLM-based network architecture.** Backbone: Johnson 2016 encoder-decoder CNN (3 conv + 5 ResBlocks + 3 deconv) with **Instance Normalization** (IN) replacing BatchNorm. FiLM layers inserted after every IN layer except the last. IN + FiLM = AdaIN (Huang & Belongie 2017). FiLM generator: FC layers in the style of StyleGAN's mapping network (Karras 2019). Only the first FC layer's input dimension depends on the number of tasks $n$.
- **3.3 Training of mixed tasks.** Three candidate methods:
  - **Method 1**: train only single tasks; rely on implicit interpolation at inference (à la Dumoulin 2017 styles).
  - **Method 2**: train mixed tasks with a summed loss $L_{\text{mixed}} = L_{\text{task A}} + L_{\text{task B}}$, conditioning on $c = [1, 0, 1]$-type mix vectors.
  - **Method 3**: **synthesize mixed-task training samples** and train on them with L2 loss. Two synthesis ways: (a) **sequential** — apply task A network output to task B's input to produce a "task A then task B" ground truth; (b) **mask-based** — use semantic-segmentation masks to combine outputs of two task networks region-wise.

**4. Experiments.** Pascal VOC 2011, 8,498 train / 2,857 test. Six tasks: reconstruction, inpainting, denoising, semantic segmentation (3-channel cut-out form, L2 + adversarial loss), style transfer 1 (Van Gogh), style transfer 2 (Munk). Adam $1\!\times\!10^{-4}$, batch 32, 300n epochs.

- **4.2 Experiment 1: multiple distinct tasks.** Qualitative: single network successfully executes all 6 tasks by changing only $c$. No major task interference except mild on inpainting (data shortage).
- **4.3 Experiment 2: mixed-task learning.** Method 1 fails (only one task dominates the output). Method 2 also fails (output biased to whichever task has stronger loss gradient). Method 3 succeeds — outputs match synthesized ground truth. Critically, *changing $c$ at inference smoothly interpolates between identity, single-task, and mixed-task* (Figure 7).
- **4.4 Experiment 3: comparison to baselines.** Vs SGN (Conditional Channel Attention), SGN+bias, Piggyback1 (base=inpainting), Piggyback2 (base=segmentation), and three more baselines: Single (6 independent networks; ~10M params), SharedEnc (shared encoder + per-task decoders; ~9.6M params), EncIn (FiLM only in decoder; ~1.7M params). The proposed method is ~1.7M params (1.7× smaller than SharedEnc, ~6× smaller than Single), achieves best scores on most single tasks (MSE, SSIM, IoU, FID) among MTL methods, and is the only method capable of mixed-task learning at the heterogeneous-task level.

**5. Conclusion.** A FiLM-only single network can learn multiple heterogeneous image translation tasks and their mixtures. Mixed-task learning requires synthesized training samples (Method 3). The objective space is *linearly mixable* in the conditional vector — the network never sees true intermediate mixtures during training but can interpolate cleanly at inference.

## Phase 1 — Undergraduate-Level Synthesis

The setup: you have a CNN encoder-decoder that takes an image and outputs another image. You want it to do six different image-to-image translations (clean up noise, fill in missing patches, segment objects, paint Van Gogh-style, paint Munk-style, copy the input). Three approaches:

1. **Six separate networks.** Simple. Costs 6× the parameters. ~10M total.
2. **Shared encoder + six decoders.** The standard MTL approach. Costs ~half-of-6× parameters. ~9.6M total. Still grows linearly with tasks.
3. **One fully-shared network plus a tiny modulator** (this paper). Costs ~1.7M total — about the same regardless of task count.

The modulator is **FiLM**: a small network that reads a "which task?" vector $c$ and emits scale-and-shift parameters $(\gamma, \beta)$ for every channel at every layer. Those modulate the activations after Instance Normalization — exactly the trick AdaIN uses for style transfer. The intuition: **the same backbone weights can implement many different transformations, as long as the activation statistics are adjusted per-task**. FiLM is the activation-statistics knob.

Then they go further. Can the network do a *combination* of tasks — e.g., denoise *and* style-transfer in one shot, or style-transfer *only* the horse in the image? Two natural ideas fail:

- **"Just train individual tasks and feed a mixed conditional vector at inference."** No: the network never saw mixtures, doesn't know how to combine.
- **"Train with summed losses for mixed tasks."** No: gradients bias the network toward whichever task has stronger or faster-converging loss; the other task is suppressed.

What works:

- **"Synthesize mixed-task ground-truth images and train on them."** Apply network A then network B to produce a target for the mixed task; use L2 to teach the joint network. Now mixtures appear in training and the network learns to do them.

And the **surprising bonus**: once trained this way, varying the mixture vector $c$ continuously at inference produces a continuous interpolation through the objective space — even though training only saw discrete mixture targets. The network's **conditional vector $c$ becomes a linear-combination coordinate** over task outputs.

For this project, the headline lessons are:

1. **FiLM can switch a backbone between very different functional modes** (segmentation, denoising, stylization), not just slight task variants.
2. The conditional vector $c$ can be a **continuous mixture coordinate** at inference — the agent could in principle interpolate behaviors smoothly.
3. **Mixing heterogeneous behaviors at inference requires teaching mixtures during training** — there is no free lunch from "just feed a mixed condition".

## Phase 2 — Graduate-Level Deep Dive

### FiLM Mechanism (Recap)

For feature map $F_i$ at the $i$-th channel of a layer, with task conditional vector $c \in \mathbb{R}^n$:

$$
\gamma_i = f_{\gamma, i}(c), \qquad \beta_i = f_{\beta, i}(c), \tag{1}
$$

$$
\text{FiLM}(F_i \mid \gamma_i, \beta_i) = \gamma_i F_i + \beta_i . \tag{2}
$$

The FiLM generator $f_\gamma, f_\beta$ is implemented as a sequence of FC layers (StyleGAN-style mapping network) that maps $c \mapsto (\gamma, \beta)$ for every FiLM layer in the backbone. The combination

$$
\text{AdaIN}_i(F_i; c) = \gamma_i(c) \cdot \frac{F_i - \mu(F_i)}{\sigma(F_i) + \epsilon} + \beta_i(c) \tag{3}
$$

— Instance Normalization followed by FiLM — is the standard adaptive-instance-norm modulator (Huang & Belongie 2017). The paper uses exactly this pattern: IN followed by FiLM, inserted after every conv layer except the last.

### Per-Task Per-Layer Modulation Cost

Backbone has $L$ IN+FiLM layers; layer $\ell$ has channel count $D_\ell$. Per layer, FiLM needs $2 D_\ell$ parameters per task (γ and β). The FiLM generator's first FC layer maps $\mathbb{R}^n \to \mathbb{R}^h$ for some hidden width $h$, so its parameter count scales as $n \cdot h$; the remaining FC layers are task-independent. Thus the **only** task-specific parameters are the input weights of the first FC layer. Total parameter count is approximately:

$$
|\theta| \approx |\text{backbone}| + |\text{FiLM generator}| , \qquad |\text{task-specific}| = O(n \cdot h).
$$

Empirically, with $n = 6$ tasks the total is ~1.7M parameters vs ~10M for 6 independent networks — a 6× reduction.

### Conditional Vector as Mixture Coordinate

The task conditional vector $c \in \mathbb{R}^n$ is defined with two special properties:

1. **Identity at $c = \mathbf{0}$.** The reconstruction task (Task 0) is *the* identity. Training Task 0 with $c = \mathbf{0}$ teaches the network that "no modulation = identity", which then anchors continuous interpolation: as $c$ moves from $\mathbf{0}$ to a single one-hot, the output transitions from identity to the task.
2. **Mixtures with $\|c\|_1 > 1$.** For two-task mixtures, $c = [1, 1, 0]$-type vectors are used at training (Method 3 synthesizes mixed targets for these). At inference, $c$ can be any non-negative real vector; the network smoothly interpolates over the 2D plane spanned by $(c_a, c_b)$ for any two tasks $a, b$.

### Why Methods 1 and 2 Fail

- **Method 1** (individual single-task training, mixture at inference). The network has never received a $c$ with $\|c\|_1 > 1$ during training. The FiLM generator's output for such a $c$ is an extrapolation; the resulting $\gamma, \beta$ tend to amplify whichever task's signal is dominant. Empirically, Mix1 (denoising + style) retains scratches (style dominates); Mix3 only shows segmentation. The network has no inductive bias toward composition.
- **Method 2** (summed loss). The loss $L_{\text{mixed}} = L_A + L_B$ is computed on **separate** support data for tasks A and B respectively — there is no single ground truth that requires both A and B to be applied jointly. So the gradient on $c = [1, 1, 0]$ tries to make the output simultaneously match A's GT (when sampled from A) and B's GT (when sampled from B), and the optimum collapses toward whichever task has stronger gradient signal.

**Method 3** (synthesized mixed GT). For sequential mix: train single-task networks first, then for each (A, B) pair, generate $y_{\text{mixed}} = f_B(f_A(x))$ from inputs $x$. Train the unified network with $c = c_A + c_B$ to match $y_{\text{mixed}}$ via L2. Now the network sees true mixture targets and learns the composition.

### Smooth Interpolation Property (Section 4.3 + Figure 7)

Once Method 3 has trained mixed targets at $c = [c_a, c_b]$ for $c_a, c_b \in \{0, 1\}$, the network exhibits smooth interpolation: varying $c$ continuously from $\mathbf{0}$ to a one-hot to a mix produces:

- $c = [0.0, 0.0, 0.0]$ → identity.
- $c = [0.5, 0.0, 0.0]$ → half-strength denoising.
- $c = [1.0, 0.0, 0.0]$ → full denoising.
- $c = [1.0, 0.0, 0.5]$ → denoising + half-strength style transfer.
- $c = [1.0, 0.0, 1.0]$ → denoising + full style transfer (the trained mixture).

This is the **2D linear combination space** claim: training on the four corners $(0,0), (1,0), (0,1), (1,1)$ of the cube is enough for the FiLM generator to extrapolate the interior smoothly. The argument is implicit but compelling — the FiLM generator's first FC layer is *linear* in $c$, so its output $(\gamma, \beta)$ is a linear function of $c$ (before the FiLM generator's nonlinearities); the per-layer affine modulations compose into a smooth function of $c$.

### Quantitative Results (Table 3)

| Metric | Proposed | EncIn | SGN | SGN+bias | Piggyback1 | Single (6 nets) | SharedEnc |
|---|---|---|---|---|---|---|---|
| Task 0 reconstruction MSE | **0.141** | 0.186 | 0.418 | 0.419 | 0.160 | 0.122 | 0.180 |
| Task 1 inpainting MSE | **0.093** | 0.104 | 0.452 | 0.452 | 0.123 | 0.077 | 0.238 |
| Task 2 denoising MSE | **0.084** | 0.104 | 0.430 | 0.431 | 0.174 | 0.089 | 0.205 |
| Task 3 segmentation IoU | **0.591** | 0.569 | 0.430 | 0.445 | 0.403 | 0.564 | 0.555 |
| Task 4 Style 1 FID | **281.3** | 318.3 | 299.0 | 307.1 | 333.4 | 186.7 | 324.4 |
| Mix1 denoising+ST1 FID | **304.3** | 361.3 | 349.1 | 343.4 | — | — | — |
| Mix2 denoise+seg IoU | **0.576** | 0.575 | 0.470 | 0.463 | — | — | — |
| Params | **1.7M** | 1.7M | 1.8M | 1.9M | 1.7M | 10.1M | 9.6M |

Key observations:

- Proposed beats SGN, SGN+bias, Piggyback on every MTL metric except the task each Piggyback variant was directly trained on (its base network).
- Proposed matches Single (independently-trained networks) on most tasks at ~17% the parameter count.
- EncIn (FiLM only in decoder, IN in encoder) is consistently worse than Proposed (FiLM in both encoder and decoder). The paper concludes: **IN+FiLM should be applied to every conv layer except the last**.

### Comparison to FiLM-Ensemble (Turkoglu 2022, this batch)

The two papers use FiLM the same way mechanically (per-channel γ/β look-up tables) but condition on different axes:

- **Takeda 2021 (this paper)**: $c$ indexes the **task**. One backbone, many tasks. Goal: parameter efficiency at multi-task inference.
- **Turkoglu 2022**: $c$ indexes the **ensemble member**. One backbone, $M$ functionally distinct sub-networks. Goal: epistemic uncertainty estimation.

Both demonstrate that the per-channel γ/β knob has enough expressive power to instantiate **functionally distinct networks from a single backbone** — a key finding for the project's use of FiLM.

## Connections to the Project

- **The clearest demonstration that FiLM can switch heterogeneous behaviors.** Takeda 2021 shows IN+FiLM at every layer can host *qualitatively different* tasks (pixel segmentation versus stylization) on one backbone with $<2\%$ task-specific parameters. For the project, this is the empirical license to expect FiLM to suffice for tasks like switching between low/high tonic pain regimes, or between exploration-and-exploitation modes — provided the modulator's conditional vector is rich enough.
- **The "conditional vector as continuous mixture coordinate" finding** (Section 4.3, Figure 7) is the most generalizable insight. If the project's neuromodulator state is multi-dimensional (e.g., separate scalars for ACh / NE / DA / 5-HT, or a continuous physiology vector), Takeda 2021's result says the FiLM-modulated policy can in principle interpolate smoothly over the modulator state space — but **only if the policy was trained on examples spanning the corners of that state space**. Training only at single neuromodulator values and expecting smooth interpolation at intermediate values is exactly the Method-1 failure mode this paper rules out.
- **Where to insert FiLM layers.** The ablation EncIn vs Proposed answers a concrete architectural question: **IN+FiLM after every layer**, not just the decoder. Recommend `senior-developer` adopt this default if the project's policy network has multiple ConvBlocks / ResBlocks before the action head.
- **Identity at $c = \mathbf{0}$.** The trick of explicitly training the identity transformation as Task 0 with $c = \mathbf{0}$ is a useful regularizer — it grounds the modulator's "no modulation" point. If the project's modulator state has a meaningful zero (no neuromodulator → baseline policy), this offers a clean architectural recipe.
- **Mixed-task training requires synthesized GT.** Method 3 is the recipe — if the project ever wants the agent to smoothly interpolate between two trained behavioral regimes at deployment, it cannot just train at the corners and hope; it must synthesize intermediate-regime training data. This is a non-trivial design constraint to flag to `experiment-designer`.
- **Connections within the corpus.**
  - Perez et al. 2018 (Batch 1) — Takeda 2021 builds directly on FiLM's affine modulation.
  - Dumoulin et al. 2017 (Batch 1, conditional instance normalization for style) — Takeda 2021 extends Dumoulin's style-mixing observation from styles to heterogeneous tasks.
  - Huang & Belongie 2017 (AdaIN, in the FiLM corpus) — IN+FiLM = AdaIN; Takeda 2021 uses this combination layer-wise throughout.
  - Turkoglu et al. 2022 (`turkoglu_2022_film_ensemble.md`, this batch) — twin paper: same mechanism (per-channel γ/β look-up), different conditioning axis (member vs task). Together they show FiLM's modulation is broad enough for both purposes.
  - Abdollahzadeh et al. 2021 (`abdollahzadeh_2021_multimodal_meta.md`, this batch) — argues FiLM's per-channel scaling is too narrow for highly multimodal task distributions and proposes KML (per-weight modulation). Takeda 2021's success with 6 disparate tasks complicates that claim — the relevant variable seems to be (i) how many distinct tasks, and (ii) how richly inserted FiLM is (every layer vs only decoder).
  - Kendall & Gal 2017 (`kendall_gal_2017_uncertainties.md`, this batch) — Takeda 2021 cites Kendall-Gal as a MTL approach using uncertainty for loss weighting. The intersection in the project's stack: a Takeda-style FiLM-modulated policy could use Kendall-Gal's heteroscedastic loss as its training criterion when there are multiple noise regimes.
