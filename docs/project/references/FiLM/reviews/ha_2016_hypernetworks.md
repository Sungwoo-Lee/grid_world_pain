---
title: "HyperNetworks"
authors: ["David Ha", "Andrew Dai", "Quoc V. Le"]
year: 2016
venue: "arXiv (later ICLR 2017)"
slug: ha_2016_hypernetworks
source_pdf: "sources/Ha et al. 2016 - HyperNetworks.pdf"
topic: FiLM
---

# Ha et al. 2016 — HyperNetworks

## Plain-English entry point

This paper introduces **hypernetworks**: the idea of using one neural network (the "hypernetwork") to *generate the weights* of a second, larger neural network (the "main network"). Instead of treating the main network's weights as the parameters you optimize directly, you optimize the much smaller hypernetwork, which outputs those weights for you. The authors borrow a biological analogy: the hypernetwork is like a **genotype** (a compact recipe) and the weights it produces are the **phenotype** (the network that actually does the work).

Two flavors are presented. A **static hypernetwork** generates the weights of a convolutional network from a small "layer embedding" vector — one embedding per layer — and is trained end-to-end with backpropagation. This gives a form of "relaxed weight-sharing" across layers of a deep CNN: weights are correlated through the shared hypernetwork but not literally tied. A **dynamic hypernetwork** (used for RNNs/LSTMs, called HyperRNN / HyperLSTM) generates *different* recurrent weights at every timestep, conditioned on the input and the previous hidden state. The dynamic version essentially makes the recurrent weight matrix a learned function of the current context.

The authors show this beats or matches strong baselines on CIFAR-10 image classification, character-level Penn Treebank and Wikipedia language modelling, IAM handwriting generation, and WMT'14 En→Fr machine translation, while using comparable or fewer learnable parameters. Hypernetworks are the *super-class* of which FiLM, MoE routing, and many later context-conditioned architectures are special cases.

## Section-ordered backbone

**1. Introduction.** A small "hypernetwork" generates weights for a larger "main network". Inputs to the hypernetwork are learned embedding vectors describing each layer (static) or dynamically computed context vectors (dynamic). Trained end-to-end with backprop, unlike HyperNEAT which uses evolution. Main contributions: (a) hypernetworks for CNNs as a form of weight factorization, (b) hypernetworks for RNNs as a form of relaxed/dynamic weight-sharing.

**2. Motivation and related work.** Inspired by evolutionary methods (HyperNEAT, Compressed Weight Search via DCT, DPPNs, ACDC-Networks) that search a small parameter space to produce a larger network. Differs by being end-to-end differentiable. Also related to Schmidhuber's classic "fast weights" idea (one network produces context-dependent weight changes for another), and to recent work on predicting CNN parameters with another network (Denil 2013, Bertinetto 2016, De Brabandere 2016) — but earlier work did not extend to recurrent networks.

**3. Methods.** Frames CNNs and RNNs as two extremes (no weight-sharing vs hard weight-sharing). Hypernetworks sit in between.

**3.1 Static hypernetwork (for CNNs).** For each layer $j$, a learned embedding $z^j \in \mathbb{R}^{N_z}$ is fed through a two-layer linear hypernetwork that outputs the full convolutional kernel $K^j \in \mathbb{R}^{N_{in} f_{size} \times N_{out} f_{size}}$. The kernel is built slice-by-slice ($N_{in}$ slices), each slice produced from $z^j$ via a per-slice linear map then a shared tensor projection. Kernels of varying sizes are handled by tiling several "basic" kernels (e.g., 16×16) together, each with its own embedding.

**3.2 Dynamic hypernetwork (HyperRNN / HyperLSTM).** A smaller "HyperRNN cell" runs alongside the main RNN, conditioned on the concatenation of $x_t$ and $h_{t-1}$. Its hidden state $\hat h_t$ is linearly projected into per-timestep embeddings $z_h, z_x, z_b$, which then parametrize the recurrent matrices and bias of the main RNN. To keep memory tractable, the authors use a **scaling-vector trick**: instead of constructing a full $N_h \times N_h$ matrix from $z$, they scale each row of a fixed weight matrix by a scalar function of $z$, equivalent to element-wise multiplicative modulation. This connects directly to multiplicative RNNs and to layer/batch normalization's affine scaling.

**4. Experiments.** Static hypernetworks reach 99.24% on MNIST (vs 99.28% baseline) using a 4-dim embedding for a 12,544-parameter kernel. On CIFAR-10 (WRN-40-1 / 40-2), one hypernetwork generates all 36 conv kernels, costing ~1.25–1.5% accuracy while sharply reducing learnable parameters. HyperLSTM beats LSTM and Layer-Norm LSTM on character Penn Treebank (1.265 vs 1.300 BPC), enwik8 (1.34 vs 1.40 BPC), IAM handwriting (-1162 vs -1106 nats log-loss), and improves a production GNMT translation model (40.03 vs 38.95 BLEU on WMT'14 En→Fr). Combining HyperLSTM with layer norm sometimes helps, sometimes hurts — the policies learned are not the same as statistical-moment normalization. Visualization shows the dynamic weights change abruptly at word/phrase boundaries.

**5. Conclusion.** End-to-end-trained hypernetworks offer a parameter-efficient, performant alternative across CNN and RNN architectures. Dynamic hypernetworks in particular challenge the recurrent weight-sharing paradigm.

## Phase 1 — undergraduate-level synthesis

**Key idea.** Don't train a big network's weights directly. Train a *small* network that, given a tiny "embedding" vector, outputs those weights for you. The small network is the **hypernetwork**; the big network is the **main network**.

**Why this is useful.** Two reasons. (1) **Parameter efficiency**: a 4-parameter embedding can produce a 12,544-parameter convolutional filter, so the model is much smaller. (2) **Context-dependent weights**: if you make the embedding a function of time or input, the main network's weights *change every timestep* — a recurrent network is no longer stuck with the same matrix multiplication forever.

**Setup.**
- **Static (CNN case)**: one learned embedding $z^j$ per layer. The hypernetwork is a tiny linear-linear stack that maps $z^j$ to the full kernel of layer $j$. Different layers share the hypernetwork but have different embeddings, so layers are correlated but not identical — "relaxed weight-sharing".
- **Dynamic (RNN case)**: a tiny "HyperRNN" runs alongside the main RNN. At every timestep, the HyperRNN's hidden state is mapped to embedding vectors $z_h, z_x, z_b$, which scale (row-wise) the main RNN's recurrent matrices. Each timestep effectively uses a slightly different recurrent weight matrix, tailored to the current word/character.

**Headline result.** Near state-of-the-art on Penn Treebank, enwik8, IAM handwriting, and WMT En→Fr translation, often with *fewer* parameters than the baseline LSTM. Visualizing the dynamic weights shows they change most at word/character boundaries — the network learned to modulate itself at semantic transitions.

**Initial takeaway.** Hypernetworks generalize many later context-conditioning ideas: FiLM is a hypernetwork that only generates per-channel affine parameters; conditional batch norm, adaptive instance norm, and even MoE routing are degenerate or sparse hypernetworks. This paper is one of the foundational references for the entire "conditional modulation" family.

## Phase 2 — graduate-level deep dive

### Static hypernetwork formulation

For each layer $j \in \{1, \dots, D\}$ of the main CNN, parameters are stored as $K^j \in \mathbb{R}^{N_{in} f_{size} \times N_{out} f_{size}}$. The hypernetwork is a function $g: \mathbb{R}^{N_z} \to \mathbb{R}^{N_{in} f_{size} \times N_{out} f_{size}}$ applied to a learned per-layer embedding $z^j \in \mathbb{R}^{N_z}$:

$$
K^j = g(z^j), \qquad \forall j = 1, \dots, D.
$$

The kernel decomposes as $N_{in}$ row-slices $K^j_i \in \mathbb{R}^{f_{size} \times N_{out} f_{size}}$. The first hypernetwork layer is $N_{in}$ separate linear maps $W_i \in \mathbb{R}^{d \times N_z}$, $B_i \in \mathbb{R}^d$; the second layer is a shared tensor $W_{out} \in \mathbb{R}^{f_{size} \times N_{out} f_{size} \times d}$ and bias $B_{out} \in \mathbb{R}^{f_{size} \times N_{out} f_{size}}$:

$$
\begin{aligned}
a^j_i &= W_i \, z^j + B_i, \qquad &\forall i = 1, \dots, N_{in}, \forall j = 1, \dots, D, \\
K^j_i &= \langle W_{out},\, a^j_i \rangle + B_{out}, &\forall i = 1, \dots, N_{in}, \forall j = 1, \dots, D, \\
K^j &= \big(\, K^j_1 \;\; K^j_2 \;\; \cdots \;\; K^j_{N_{in}} \,\big),
\end{aligned}
$$

where $\langle W, a \rangle \in \mathbb{R}^{m \times n}$ denotes a tensor dot product contracting the last $d$-axis of $W \in \mathbb{R}^{m \times n \times d}$ with $a \in \mathbb{R}^d$.

**Parameter count.** The total learnable parameter count of the hypernetwork is

$$
\underbrace{N_z \cdot D}_{\text{embeddings}} + \underbrace{d \cdot (N_z + 1) \cdot N_{in}}_{\text{first layer}} + \underbrace{f_{size} \cdot N_{out} \cdot f_{size} \cdot (d + 1)}_{\text{second layer}},
$$

compared to $D \cdot N_{in} \cdot N_{out} \cdot f_{size}^2$ for storing the kernels directly. With $d = N_z$ small, this is a large compression. The two-layer design (vs. a single linear map $z^j \to K^j$) is critical: a one-layer hypernet costs $N_z \cdot N_{in} \cdot N_{out} \cdot f_{size}^2$, whereas the two-layer factorization shares $W_{out}$ across slices.

**Kernel tiling for varying dimensions.** Real architectures (e.g., wide ResNets) use kernels whose $N_{in}, N_{out}$ are multiples of 16. The hypernetwork generates a basic $16 \times 16$ kernel; larger kernels are concatenated tilings, each tile generated from its own embedding $z$:

$$
K^{32 \times 64} = \begin{pmatrix} K_1 & K_2 & K_3 & K_4 \\ K_5 & K_6 & K_7 & K_8 \end{pmatrix}.
$$

### Dynamic hypernetwork formulation (HyperRNN)

Standard basic RNN:

$$
h_t = \phi(W_h h_{t-1} + W_x x_t + b),
$$

with $W_h \in \mathbb{R}^{N_h \times N_h}$, $W_x \in \mathbb{R}^{N_h \times N_x}$, $b \in \mathbb{R}^{N_h}$ *fixed across time*.

**Full dynamic formulation (memory-expensive).** Let $W_h, W_x, b$ vary across time as functions of per-timestep embeddings $z_h, z_x, z_b \in \mathbb{R}^{N_z}$:

$$
h_t = \phi\!\Big( W_h(z_h)\, h_{t-1} + W_x(z_x)\, x_t + b(z_b) \Big),
$$

where, with $W_{hz} \in \mathbb{R}^{N_h \times N_h \times N_z}$, $W_{xz} \in \mathbb{R}^{N_h \times N_x \times N_z}$, $W_{bz} \in \mathbb{R}^{N_h \times N_z}$, $b_0 \in \mathbb{R}^{N_h}$,

$$
W_h(z_h) = \langle W_{hz}, z_h \rangle, \qquad W_x(z_x) = \langle W_{xz}, z_x \rangle, \qquad b(z_b) = W_{bz} z_b + b_0.
$$

The HyperRNN cell itself is an RNN over the concatenated input $\hat x_t = (h_{t-1}, x_t)^\top$:

$$
\hat h_t = \phi(W_{\hat h} \hat h_{t-1} + W_{\hat x} \hat x_t + \hat b),
$$

whose state is linearly projected to embeddings:

$$
z_h = W_{\hat h h} \hat h_{t-1} + b_{\hat h h}, \quad z_x = W_{\hat h x} \hat h_{t-1} + b_{\hat h x}, \quad z_b = W_{\hat h b} \hat h_{t-1}.
$$

**Memory-efficient formulation (scaling-vector trick).** The full version costs $N_z \times$ baseline RNN memory because of the rank-$N_z$ tensors $W_{hz}, W_{xz}$. The authors replace the dense weight tensor with a per-row scaling:

$$
W(z) = W\!\big(d(z)\big) = \begin{pmatrix} d_0(z) W_0 \\ d_1(z) W_1 \\ \vdots \\ d_{N_h}(z) W_{N_h} \end{pmatrix},
$$

i.e., each row $W_i$ of a fixed matrix $W$ is scaled by a scalar $d_i(z)$. Equivalently, element-wise multiplication:

$$
h_t = \phi\!\Big( d_h(z_h) \odot (W_h h_{t-1}) + d_x(z_x) \odot (W_x x_t) + b(z_b) \Big),
$$

with

$$
d_h(z_h) = W_{hz} z_h, \qquad d_x(z_x) = W_{xz} z_x, \qquad b(z_b) = W_{bz} z_b + b_0.
$$

Here $W_{hz}, W_{xz} \in \mathbb{R}^{N_h \times N_z}$, $W_{bz} \in \mathbb{R}^{N_h \times N_z}$.

**Connection to normalization and to FiLM.** The element-wise multiplication $d(z) \odot W h$ followed by an additive bias $b(z)$ is *exactly* the structure of an affine modulation $\gamma \odot x + \beta$, where $\gamma = d(z)$ and $\beta = b(z)$ are conditioning-driven. This is precisely what FiLM (Perez et al. 2018) generalizes — FiLM is a *static-input-conditioned, per-channel* version of this dynamic per-row scaling. Layer normalization and recurrent batch normalization apply a *fixed* affine after normalizing by data statistics; HyperRNN learns the affine as a function of $(x_t, h_{t-1})$. The authors explicitly note both connections.

### Empirical anchor

| Task | Baseline | HyperLSTM (best) |
|---|---|---|
| Penn Treebank (BPC) | 1.300 (LN-LSTM) | 1.219 (2-Layer LN HyperLSTM) |
| enwik8 (BPC) | 1.402 (LN-LSTM, 1800u) | 1.340 (LN HyperLSTM, 2048u) |
| IAM handwriting (log-loss) | -1106 (LN-LSTM, 1000u) | -1162 (HyperLSTM, 900u) |
| WMT'14 En→Fr (BLEU) | 38.95 (GNMT LSTM) | 40.03 (GNMT HyperLSTM) |

## Connections

- **`perez_2018_film.md`** (other batch — FiLM): FiLM is a special case of a *static-context* dynamic hypernetwork where the generated weights are restricted to per-channel affine $(\gamma, \beta)$. HyperRNN's memory-efficient scaling-vector trick (Eq. 7–8 here) is structurally identical to FiLM at the per-row granularity. Ha 2016 is FiLM's super-class.
- **`shazeer_2017_sparse_moe.md`** (this batch — Sparse MoE): MoE and hypernetworks share the "many small specialists, one router" structure. MoE picks among a discrete bank of pre-existing expert weights via a gating network; a hypernetwork *synthesizes* weights from a continuous embedding. Both let the *effective* parameter count exceed the *trained* parameter count.
- **`galanti_wolf_2020_hypernet_modularity.md`** (this batch): Provides the theoretical justification (modularity / functional decomposition) for why hypernetworks generalize better than monolithic conditional networks on the same task family.
- **`krueger_2017_bayesian_hypernets.md`** (this batch): Lifts the deterministic hypernetwork here to a *Bayesian* one — the hypernetwork outputs samples from $q(\theta)$, an implicit variational posterior over main-network weights. Directly extends the static hypernetwork formulation in this paper.
- **`andreas_2016_neural_module_networks.md` / `hu_2017_e2e_module_networks.md`** (this batch): Module networks are a different compositional strategy — instead of one hypernetwork producing weights, a discrete program lays out a set of pre-trained modules. Conceptually an alternative to the hypernet path; both are precursors to compositional FiLM.
