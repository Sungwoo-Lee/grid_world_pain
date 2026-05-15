---
title: "A sparse quantized Hopfield network for online-continual memory"
authors: ["Nicholas Alonso", "Jeffrey L. Krichmar"]
year: 2023
venue: "Nature Communications 15:3722 (accepted 2024)"
slug: alonso_krichmar_2023_sparse_quantized_hopfield
source_pdf: "sources/Alonso and Krichmar 2023 - A sparse quantized hopfield network for online-continual memory.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

Most deep neural networks learn offline: you batch data, you randomize it (i.i.d.), and you do many passes over it with backpropagation. Brains can't do that. Brains see a stream of noisy data one observation at a time, often in a non-random order — many cats today, many dogs tomorrow — and yet manage to learn cats *and* dogs without forgetting one when they see the other. The brain also updates synapses using only *local* information: what the pre-synaptic and post-synaptic neurons did, not gradients backpropagated from far away. The question this paper attacks: can we build a deep network that learns under those constraints — **noisy, online, continual, local-rules-only** — without falling back on backpropagation?

The answer is **the Sparse Quantized Hopfield Network (SQHN)**: an energy-based model that interpolates between Hopfield networks and Bayesian directed graphical models. Each hidden node holds a *categorical* variable represented as a sparse **one-hot vector** (only one neuron in that node is active at a time — this is the "sparse quantized" part). The architecture is a tree (no loops): visible nodes at the bottom hold image patches, a chain of hidden nodes runs up to a single **memory/root node** at the top whose one-hot value is essentially a sparse code for which image the network is currently retrieving. Learning is **online maximum-a-posteriori (MAP)** — at each new observation, run a single feedforward + feedback sweep, find the most probable one-hot assignment at every node, and update the matrices column-locally by a Hebbian-style rule (Eq. 4). Crucially: weights start at zero, and new neurons / columns are **grown** as needed (neurogenesis-style), with a decaying threshold based on a Dirichlet prior (Eq. 5).

The headline claims, demonstrated across CIFAR-10, CIFAR-100, Tiny ImageNet, Caltech256, E-MNIST, F-MNIST, and SVHN: (1) on classic auto-associative and hetero-associative recall, SQHN matches or beats state-of-the-art predictive-coding networks (PCN, GPCN, BayesPCN) and Modern Hopfield Networks (MHN) — perfect recall MSE 0.0000 on all single-hidden-layer tasks; (2) in **online-continual** auto-association where data arrives one shot per iteration in non-i.i.d. order, SQHN does *one-shot memorization* and then forgets very gracefully, while BP-trained MHN baselines fail; (3) under noisy encoding (training only on noisy samples, recalling the clean original), SQHN substantially outperforms MHN-with-BP; (4) on a novel **episodic recognition** task (judge "have I seen this image before?"), SQHN's memory-node max-activity provides a natural decision threshold and beats MHN. The memory/root node has properties strikingly like the hippocampus — it grows new neurons throughout training, it indexes the rest of the network. Although the paper doesn't use the word "neuromodulator", the parameter-isolation + growth + decay-schedule mechanism plays the same role as the novelty/familiarity gating in [hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md): a structural answer to catastrophic forgetting under online-continual conditions.

## Section-ordered backbone

### Abstract
Brains learn online with local synaptic rules on noisy, non-i.i.d. data. Standard deep networks use offline batched i.i.d. backprop. The authors propose discrete graphical models with online MAP learning as a principled bridge, and implement them as the **Sparse Quantized Hopfield Network (SQHN)**. SQHN outperforms state-of-the-art neural networks on associative memory, online-continual settings, noisy encoding, and a new episodic-memory task.

### Introduction
The brain learns under noisy, online-continual conditions with local rules; deep networks use backprop offline. Recent work on bio-plausible deep learning (refs 8–11) is still tested offline; recent online-continual work (refs 12–17) still uses BP. Local online-continual learners without BP are missing. Contributions: (1) focus on the more general task of *associative memory* rather than classification; (2) propose discrete graphical models with online MAP as a strategy that naturally yields local rules and parameter isolation via sparse codes; (3) implement as SQHN; (4) introduce two new memory tasks (noisy encoding, episodic recognition); (5) extensive comparisons.

### Toward a foundation for local, online-continual memory models
Three pillars of the approach:
1. **Quantized neural codes** for associative recall: continuous-valued latent codes can drift to absorb noise; quantized codes (one-hot) cap the number of distinct mappings so many corrupted inputs deterministically collapse to the same code.
2. **Parameter isolation** via sparse activity: only the column corresponding to the active one-hot neuron is updated at each step, automatically isolating parameters across data points.
3. **Local MAP learning**: find the one-hot assignment maximizing the joint over hidden states, then make a small local update that increases the joint over all past data.

### The Sparse Quantized Hopfield Network
**Architecture (Fig. 1A).** Tree-structured directed acyclic graph. Visible nodes (bottom): image patches, continuous-valued. Hidden nodes: each holds a sparse one-hot vector $h_l^*$. Memory / root node at the top. Conditional $p(h_l^* | \text{pa}_l)$ parametrized by matrix $M_{\text{pa}_l, l}$.

**Energy** (sum, not product, of conditional probabilities — Eq. 1):

$$
E(h^*, \theta) = \frac{1}{L}\sum_{l=0}^{L} p(h_l^* | \text{pa}_l).
$$

The sum form (vs the usual product in Bayesian networks) lets the SQHN use standard neural network *additive* operations and admits an interpretation as an approximation to the joint marginal under parameter uncertainty.

**Recall (inference, Algorithm 1).** Single feedforward sweep up the tree (input to memory), then feedback sweep down, with hard winner-take-all picking the one-hot assignment that maximizes energy at each node.

**Learning (Algorithm 2).** After inference, update matrix columns by:

$$
\Delta M_{\text{pa}_l, l} = \frac{c^*_{\text{pa}_l}}{c^*_{\text{pa}_l} + 1}\,(h_l^* - M_{\text{pa}_l, l} h_{\text{pa}_l}^*)\, h_{\text{pa}_l}^{*\top}, \tag{4}
$$

where $c^*_{\text{pa}_l}$ counts how many iterations that parent value has been active. Because $h_{\text{pa}_l}^*$ is one-hot, only one column of $M$ updates at a time → *sparse update* → parameter isolation by construction.

**Neuron growth (Eq. 5).** Weights start at 0. A new neuron is grown at node $l$ on iteration $t$ if no existing neuron's activity exceeds a decaying threshold:

$$
\epsilon = \frac{\alpha}{t + \alpha},
$$

derived from a Dirichlet prior over neuron assignments. Together with learning-rate decay (Supplementary Fig. 2 — ablations show all three components matter), this preserves old memories while allowing new ones to allocate fresh capacity.

### Related works
- **Recursive Cortical Networks (RCN)** — tree-structured Bayesian-neural hybrid (similar architecture; different learning, max-pools, no energy function).
- **Sparse Hopfield-like networks**, **Hierarchical Temporal Memory** — sparse but not explicitly categorical / not energy-based.
- **Continual learning approaches** split into regularization (refs 35–39), memory/replay (refs 40–42), parameter isolation (refs 22–25). SQHN combines sparse-activity parameter isolation with learning-rate-decay regularization — distinct because it doesn't require global gradients to do the isolation.
- **Modern Hopfield Networks (MHN)** (Ramsauer et al. 2020) — `x_new = M softmax(β M^T x)` is a special case of the transformer attention layer. SQHN's hard WTA corresponds to MHN's $\beta \to \infty$ limit but with dynamic memory-matrix growth instead of a fixed random $M$ trained with BP.
- **Predictive Coding Networks (PCN)** — directed graphical models with Gaussian latent variables; minimize free energy $F = \sum_l \tfrac{1}{2}\|h_l - W_{\text{pa}_l,l} h_{\text{pa}_l}\|^2$ (Eq. 8); learning rule $\Delta W = -\alpha\,(h_l - W_{\text{pa}_l}h_{\text{pa}_l})h_{\text{pa}_l}^\top$ (Eq. 9). Identical to SQHN's Eq. 4 *up to* the one-hot quantization. Crucial difference: PCN inference is iterative gradient descent (dozens to hundreds of neuron updates per training step) — too slow for online learning; SQHN inference is one FF + one FB sweep.

### Experiments
**1. Auto-association and hetero-association comparison (Table 1).** 1024 images (moderate corruption) or 128 images (high corruption) from CIFAR-10 or Tiny ImageNet. Noise types: white noise, pixel dropout, masking. SQHN L1 / L2 / L3 (one / two / three hidden layers) compared against PCN, GPCN, BayesPCN, MHN, MHN-Manhattan, MHN-GradInf. SQHN L1 hits 0.0000 recall MSE across every condition. Multi-layer SQHNs slightly more sensitive to high noise on small CIFAR-10 images.

**2. Online-continual auto-association (Fig. 2).** Two settings: Online Class Incremental (OCI — each task = one class) and Online Domain Incremental (ODI — each task = one of four visually distinct datasets). SQHN performs one-shot memorization until capacity reached, then graceful decay; MHN-SGD / MHN-Adam / MHN-EWC++ / MHN-ER all learn too slowly to recall reliably in online passes and are highly order-sensitive.

**3. Noisy encoding (Fig. 3).** Train only on noisy versions of images (Gaussian noise or binary samples), test recall on the clean version. SQHN+ (variant where the one-hot latent is frozen across noisy samples of the same image) dramatically outperforms MHN-SGD / MHN-Adam.

**4. Episodic recognition (Fig. 4).** Novel task: after training on $X_{\text{train}}$ online, classify a test image as old / new. Test mix: training-distribution images, hold-out same-distribution, out-of-distribution. SQHN uses the **max activity at the memory root node**, compared against a moving average of activities, $\mu_{L,j}^t$ (Eq. 13). The threshold corresponds to $p(x_t = \text{old} | \theta, x_t) = 0.5$. SQHN performs perfectly until capacity then decays gracefully; MHN cannot do better than chance.

**5. SQHN architecture comparison (Fig. 5).** L1 / L2 / L3 SQHNs across CIFAR-100 / Tiny ImageNet / Caltech256. More layers → better recall under occlusion (compositional hierarchy helps ignore corrupted parts); more layers → slower decay after capacity reached (lower-layer feature primitives are reusable across data points); large bottom-layer receptive fields → better noise robustness (L1 has the largest, so wins on noise tasks for small images).

### Discussion
- Sparse quantized codes are advantageous for auto-association vs PCN's continuous codes.
- SQHN beats MHN-BP on online-continual: faster training, one-shot memorization, low order-sensitivity, no replay buffer.
- The memory node has hippocampus-like properties: grows neurons throughout training (cf. adult hippocampal neurogenesis), highly flexible synapses, indexes the rest of cortex (cf. hippocampal indexing theory, refs 56–58). These weren't engineered in; they emerged from the online-continual optimization.
- Future: SQHN for generalization (classification, self-supervised learning) will need more elaborate architectures.

### Methods (selected)
- Datasets: MNIST / F-MNIST / E-MNIST (1×28×28); SVHN / CIFAR-10 / CIFAR-100 (3×32×32); Tiny ImageNet (3×64×64); Caltech256 (cropped to 3×128×128).
- Recall threshold $\gamma = 0.01$ throughout.
- Feedforward operation at first hidden layer (Eqs. 14, 15): shifted cosine similarity between input patch $x_l^c$ and each memory column $m_{l,j}$, normalized to $[0, 1]$.
- At higher layers (Eq. 16): weighted average over child one-hot maxima.
- Final assignment (Eqs. 17, 18): argmax at memory root, then mixed argmax going down ($h_l^* = \arg\max(\lambda h_l + (1-\lambda) M_{\text{pa}_l, l} h_{\text{pa}_l}^*)$).

## Phase 1 — Undergraduate-level synthesis

**Key idea.** When you store memories in a Hopfield network, the network is a giant lookup table where each pattern lives at a local minimum of an energy landscape. Traditional Hopfield networks store patterns as *real-valued* vectors and use the *product* of pattern similarities. The SQHN does two simple things differently: (1) it stores patterns as *one-hot* vectors at hidden nodes — exactly one neuron is on at a time per node — and (2) it uses a *sum* of conditional probabilities instead of a product. The first change makes the network robust to noise: small perturbations of the input still snap to the same one-hot code. The second change makes the network easy to compute with standard add-and-multiply operations.

**Why one-hot helps.** Imagine a noisy image of a "3" and you want the network to clean it up. If the latent code is real-valued, the latent can drift slightly to absorb each noise pattern — different noise samples of the same "3" all push the latent into slightly different places. If the latent code is one-hot ("position 4 of 10"), every noisy version of the "3" must round to *the same* one-hot position. The cleanup is automatic.

**Why grow neurons.** The network starts with empty matrices and zero neurons at the hidden nodes. Each new image that doesn't activate any existing neuron above a threshold causes a new neuron to be born. The growth threshold *decays* over time (Dirichlet prior), so early images get their own neurons easily; later images have to compete harder for new slots. This is exactly how the hippocampus is believed to work in adult mammals — new dentate-gyrus neurons keep appearing throughout life.

**Setup.** Run two kinds of test. Standard test: present 1024 images one at a time, then test the network by showing it 75%-masked or heavily-noised versions of those images — does it reconstruct the originals? Continual test: present 4 datasets sequentially (1024 images each), one image at a time, never returning. Then test on a mix of old images. The "online continual" version of the task is the hard one for BP-based networks because each gradient step pulls the network toward the most recent image; the SQHN dodges this by *isolating* each image into its own column of memory.

**Result.** On the standard test, SQHN with one hidden layer perfectly reconstructs every image under every corruption (0.0000 MSE). On the continual test, the SQHN reaches one-shot memorization and forgets very slowly when the network runs out of slots — meanwhile MHN-with-SGD can't recall anything at all. On the new "old vs. new" recognition test (the network has to say whether it has seen each test image before), SQHN's recognition accuracy stays well above the 66% guess line even after the memory node is over-stuffed.

**Concrete instantiation.** Imagine the memory root node has 500 neurons. Image 1 comes in → no existing neuron lights up → grow neuron 1, store image 1 in its column. Image 2 comes in → neuron 1 doesn't fit → grow neuron 2. After 500 images, the node is full. Image 501 comes in → no growth, so the nearest existing neuron's column is *averaged* with the new image (Eq. 4). This averages a few similar memories together — they're "compressed" into shared neurons. Most older memories survive intact because their columns weren't averaged.

## Phase 2 — Graduate-level deep dive

### Energy function and its derivation

The SQHN energy is *the average of conditional probabilities*, not their product:

$$
E(h^*, \theta, x) = \frac{1}{L}\sum_{l=0}^{L} p(h_l^* \mid \text{pa}_l). \tag{1}
$$

In the supplementary derivation (Note 1.1), this is shown to approximate the joint $p(h^*, x | \theta)$ under a posterior over parameters: take $\log p(h^*, x | \theta) = \sum_l \log p(h_l^* | \text{pa}_l)$ in the usual product form, expand under parameter uncertainty $p(\theta | \mathcal{D})$, and apply a first-order Taylor expansion in log-space to convert a product into a sum. The sum form has two practical advantages:
1. **Standard neural-net ops**: summing inputs from children + parent, no need to log-transform (which can require representing very large negative numbers in fixed-precision hardware).
2. **Numerical stability**: no underflow risk from multiplying many small probabilities.

### MAP inference: one FF + one FB sweep

Inference solves $h^* = \arg\max_{h^*} E(h^*, \theta, x)$ (Eq. 2). The tree structure makes this tractable: a single bottom-up pass to the memory node followed by a single top-down pass back to the leaves. This is conceptually the max-product algorithm specialized for trees, but with two differences:
- **Sum not product**: corresponds to the modified energy form Eq. 1.
- **Hard WTA on the way down**: at every internal node, the one-hot assignment is chosen by $\arg\max$, not propagated as a distribution.

Total compute: one feedforward + one feedback sweep per training iteration. Compare to PCN inference, which requires dozens to hundreds of iterative neuron updates — SQHN inference is $\sim 100\times$ cheaper per step.

### Feedforward operations

**First hidden layer (visible inputs continuous, Eq. 14):**

$$
h_{l,j} = \frac{\langle m_{l,j} - 0.5,\, x_l^c - 0.5\rangle}{\|m_{l,j} - 0.5\| \cdot \|x_l^c - 0.5\|} \cdot 0.5 + 0.5.
$$

A shifted cosine similarity between input patch and each memory column. Shift by 0.5 because image pixels are in $[0, 1]$; renormalize to $[0, 1]$. Supplementary Note 2 shows this is a normalized weighted average of the conditional probabilities of each pixel value under a binary-pixel assumption.

**Hidden-to-hidden (Eq. 16):**

$$
h_l = \frac{1}{Z}\, M_l^\top\, h_{\max}^c, \qquad h_{\max}^c = \big[h_{\max,0}^c \mathbin\Vert h_{\max,1}^c \mathbin\Vert \dots \mathbin\Vert h_{\max,N}^c\big],
$$

where $h_{\max,n}^c$ is the WTA-max of child $n$'s activations and $Z = N\|h_{\max}^c\|$. This is a weighted average over the conditional probabilities of child assignments — a sum-version of the standard belief-propagation up-message.

### Feedback operations

After the root assigns $h_L^* = \arg\max(h_L)$ (Eq. 17), top-down assignments at intermediate nodes mix bottom-up and top-down inputs (Eq. 18):

$$
h_l^* = \arg\max\!\left(\lambda\, h_l + (1 - \lambda)\, M_{\text{pa}_l, l}\, h_{\text{pa}_l}^*\right).
$$

Here $\lambda$ tunes top-down vs bottom-up weighting; for pure recall (clean retrieval) $\lambda \to 0$ and the network simply replays the top-down decoding.

### Learning rule and its locality

The update (Eq. 4):

$$
\Delta M_{\text{pa}_l, l} = \frac{c^*_{\text{pa}_l}}{c^*_{\text{pa}_l} + 1}\,(h_l^* - M_{\text{pa}_l, l}\, h_{\text{pa}_l}^*)\, h_{\text{pa}_l}^{*\top}.
$$

Three structural features:
1. **Locally Hebbian**: depends only on activations of two adjacent nodes, $h_l^*$ and $h_{\text{pa}_l}^*$ — no gradient from far away.
2. **Sparse**: because $h_{\text{pa}_l}^*$ is one-hot, only the column indexed by the active parent value is touched.
3. **Running-average normalization**: $c^*_{\text{pa}_l}/(c^*_{\text{pa}_l} + 1)$ is the count-based learning rate. The first time parent value $j$ wins, $c^* = 0$ and the column is *set* to $h_l^*$ (full write). On the $k$-th win, $c^* = k$ and the column is updated by a $k/(k+1)$ fraction — so the column converges to the running mean of the child observations conditioned on parent value $j$. This is the MAP estimate under a Dirichlet prior.

### Neuron growth threshold

The threshold $\epsilon = \alpha/(t + \alpha)$ (Eq. 5) is derived in Supplementary Note 1.3 from a Dirichlet prior over the categorical distribution at each node:
- A Dirichlet with concentration $\alpha$ assigns probability $\alpha/(N+\alpha)$ to a previously unobserved category after $N$ observations.
- The decision "is the closest existing neuron's similarity above $\epsilon$?" amounts to "is the data more likely under an existing category than under a newly drawn one?"
- The decay over $t$ implements the Bayesian update of the categorical prior.

Ablations (Supplementary Fig. 2) show: removing growth → can't store anything new; removing the decay → grows too many neurons indiscriminately; removing learning-rate decay → forgets too fast.

### Equivalence with Modern Hopfield Networks at $\beta = \infty$

A one-hidden-layer SQHN (Eq. 7):

$$
x_{\text{new}} = M\,\arg\max\!\left(\frac{1}{Z}(M^\top - \mu)(x - \mu)\right),
$$

vs MHN with very large $\beta$ (Eq. 6):

$$
x_{\text{new}} = M\,\text{softmax}(\beta M^\top x) \xrightarrow{\beta \to \infty} M\,\arg\max(M^\top x).
$$

The two differ in: (1) SQHN normalizes both input and memories before the inner product (cosine-style, shift-by-0.5 to handle [0,1] images); (2) SQHN grows $M$ dynamically vs MHN's fixed random+BP-trained $M$. The first matters in masking tasks; the second matters in online-continual settings.

### Why discrete latents beat continuous latents under noise

Consider a single-image associative memory with stored datum $x$ and noisy query $\tilde x = x + \eta$. For a continuous latent code $h = f(\tilde x)$, $h$ varies smoothly with $\eta$: $\|h(\tilde x) - h(x)\| \approx \|\nabla f\|\cdot\|\eta\|$, so reconstructions also drift. For a discrete latent $h \in \{0,1\}^K$ assigned by $\arg\max$ over a discriminant function, $h(\tilde x) = h(x)$ as long as $\eta$ stays within the *decision basin* of the correct one-hot — a finite tolerance independent of $\|\eta\|$ once well inside the basin. The SQHN exploits this hard quantization to give *exact* reconstruction for moderate noise, vs PCN's gracefully degrading but always non-zero MSE.

### Memory-node max-activity as a recognition criterion

For episodic recognition, define $h_{L,j}^t$ as the activity of the most-active root neuron at iteration $t$. Maintain a running mean over the maximum-activity values:

$$
\mu_{L,j}^t = \frac{c_{L,j}^{t-1}}{c_{L,j}^t}\, \mu_{L,j}^{t-1} + \frac{1}{c_{L,j}^t}\, h_{L,j}^t. \tag{13}
$$

Supplementary Note 5 shows that $\mu_{L,j}^t$ is the natural threshold at which the posterior $p(x_t = \text{old} | \theta_{t-1}, x_t) = 0.5$. Decision rule: classify $x_t$ as "old" if its max-activity exceeds $\mu$, else "new". MHN cannot do this because BP-trained MHN doesn't maintain an explicit moving average of similarities.

### Parameter table (selected)

| Symbol | Meaning | Notes |
|---|---|---|
| $L$ | Number of layers (tree depth) | 1, 2, or 3 in experiments |
| $\alpha$ | Dirichlet concentration in growth threshold | Hyperparameter |
| $\epsilon$ | Growth threshold $= \alpha/(t+\alpha)$ | Decaying |
| $c^*_{\text{pa}_l}$ | Activation count of parent value | Used in learning rule |
| $\gamma$ | Recall MSE threshold for accuracy | 0.01 |
| $\beta$ | MHN inverse temperature (comparison) | Effectively $\infty$ in SQHN |
| $\lambda$ | Top-down / bottom-up mix on feedback | Tuned per task |
| $Z$ | Normalization in FF / FB | Activation-range normalizer |

### Connection to hippocampal indexing theory

The discussion lists three specific neural correspondences:
1. **Memory root node grows neurons throughout training** — adult hippocampal neurogenesis (refs 59).
2. **Memory node is at the top of the architectural hierarchy and indexes downstream cortex** — Teyler & DiScenna 1986 hippocampal indexing theory.
3. **Synapses at the memory node remain flexible** — hippocampal synapses retain plasticity into adulthood, unlike most cortical synapses (ref 60).

These are emergent, not engineered: the SQHN was designed to solve online-continual learning, not to look like the hippocampus, but the hippocampus-like properties fell out of the solution.

## Connections

**Direct references inside this corpus:**

- **[hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md)** — same lab, same anti-catastrophic-forgetting agenda, complementary mechanism. Hwu 2020 uses CHL + mPFC/HPC indexing + neuromodulated replay; Alonso 2023 uses MAP-learning + one-hot quantization + neurogenesis. Both target the same problem; SQHN is the more general (image-domain, scalable) version.
- **[krichmar_hwu_2022_design_principles_neurorobotics](krichmar_hwu_2022_design_principles_neurorobotics.md)** — design-principles companion. SQHN exemplifies the principles of "learning + memory" and "value systems" (parameter isolation as the algorithmic counterpart of neuromodulator-driven gating).
- **[osman_2024_hopfield_neuromodulatory_arousal](osman_2024_hopfield_neuromodulatory_arousal.md)** — sister paper applying Hopfield-network mechanics to arousal-state modulation, conceptually downstream.
- **[tambas_2025_neuromodulation_krotov_hopfield](tambas_2025_neuromodulation_krotov_hopfield.md)** — also uses Krotov-Hopfield variants with neuromodulation; thematically related.
- **[espino_2024_spiking_neural_network](espino_2024_spiking_neural_network.md)** — same lab program of rapid-adaptation + continual-learning, in a spiking-neural-net implementation.
- **[avery_krichmar_2017_models_neuromodulation](avery_krichmar_2017_models_neuromodulation.md)** — same lab's neuromodulation review, conceptually upstream.

**Forward / sibling references in the corpus.** Mei et al. 2022, Wainstein et al. 2025, Costacurta et al. 2024, Tsuda et al. 2021, Kudithipudi et al. 2022 ("Biological underpinnings for lifelong learning machines") are in the same conceptual cluster of biologically-inspired continual learning. Driscoll et al. 2022 (shared dynamical motifs in RNNs) is the contrasting motif-based view.

**External anchors.** Hopfield 1982 / Ramsauer 2020 (Modern Hopfield Networks); McClelland et al. 1995 (CLS); Teyler & DiScenna 1986 (hippocampal indexing); Yoo et al. (BayesPCN); Whittington & Bogacz 2017 (PCN); Friston 2010 (free energy); Movellan 1991 (CHL); Krotov & Hopfield variants.

**Note on neuromodulator framing.** This paper does **not** use the word "neuromodulator" in the main text — it's framed entirely in terms of energy-based graphical models. However, the parameter-isolation + growth + decay-schedule mechanism plays the same *functional* role that the ACh + NE neuromodulators play in Hwu & Krichmar 2020, Zou et al. 2020, and Xing et al. 2022: a structured, biologically-inspired answer to the catastrophic-forgetting problem under online-continual conditions. The curator may wish to flag this as a *mechanism-equivalent* paper rather than a *modulator-equivalent* paper.
