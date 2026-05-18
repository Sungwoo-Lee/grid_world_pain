# Learning_Rate Reference Review — Section D: Gated / Conditional / Hypernetwork Architectures

**Reviewer:** literature-reviewer
**Date:** 2026-05-19
**Source folder:** [`docs/project/references/Learning_Rate/sources/`](sources/)
**Scope:** Four foundational papers on the architectural mechanisms by which one neural signal can modulate the forward pass of another network — gating (Highway, GLU) and weight generation (HyperNetworks).

---

## Why these papers belong in the "Learning_Rate" folder

A first-time reader who opens this folder will see titles like *"Learning Rate Adaptation"*, *"Meta-SGD"*, and *"Hypergradient Descent"* alongside the four papers reviewed here — *Highway Networks*, *Training Very Deep Networks*, *HyperNetworks*, and *Gated Convolutional Networks*. These four are not about learning-rate adaptation in the classical sense (no per-parameter step-size $\alpha_t$, no meta-gradient on the optimizer). They are filed here because **they are the architectural mechanisms by which a "modulator" signal — be it a neuromodulator scalar, a task embedding, a context vector, or a learned learning-rate-like quantity — can actually change what a network computes on a given forward pass**. The Learning_Rate corpus in this project is broader than "what is $\alpha$?"; it is "by what mechanism does *anything* adapt about the network at inference time?".

Three concepts the reader needs before the technical sections:

- **Gating** in a neural network means routing information through a learned, data-dependent **multiplicative switch**. A gate is a scalar (or vector) $g \in [0, 1]$ — typically the output of a sigmoid on some affine transform of an input — that is multiplied element-wise into another signal. If $g = 0$ the signal is blocked; if $g = 1$ it passes through; intermediate values let some fraction through. The same idea drives the forget/input/output gates in an LSTM cell, the carry/transform split in a Highway layer, and the GLU's pass-through stream. The key property is that **the gate is learned**, so the network discovers which paths to keep open based on the current input.
- **Hypernetwork** is a small neural network whose **outputs are the weights of another, usually larger, neural network**. The big network — the "main network" — does the actual task (e.g. image classification, language modelling). The small network — the "hypernetwork" — takes some compact input (a layer index embedding, a context vector, a previous hidden state) and emits the parameter matrix that the main network will use. Both are trained jointly with backprop. The biological intuition the authors offer is genotype-to-phenotype: the hypernetwork is a compressed description that *generates* the larger network on demand. For the project's purpose, this is the most direct path by which a modulator signal can literally **be the input that determines the main network's weights** on the current step.
- **Conditional architecture** is the umbrella term: a network whose forward pass is parameterised by some auxiliary signal, so the same network computes different functions depending on what that signal is. FiLM (feature-wise linear modulation, reviewed elsewhere in this corpus) is one instance; Highway gating is another (the gate is conditioned on the activation itself); HyperNetworks are the most general (the entire weight matrix is conditioned on an embedding).

**The conceptual progression in this review:** Highway (2015) shows that a learned gate can open a "skip path" through a very deep network so gradients propagate; Training Very Deep Networks (2015 NeurIPS) is the extended journal write-up of the same architecture with more experiments and the lesion analysis; HyperNetworks (2016) generalises beyond gating to having one network *generate* another's weights; GLU (2016/2017) brings gating into convolutional language models with a simplified gradient-friendly form $h = (X \ast W + b) \odot \sigma(X \ast V + c)$ that became standard in modern Transformer-era architectures (e.g. SwiGLU in LLaMA, GeGLU in T5-v1.1). For our project, all four are upstream of any plan in which a modulator signal *changes what the main policy network computes* at a given step — whether the signal is a Doya-style neuromodulator scalar, a task ID, or an interoceptive context.

---

## Table of Contents

1. [Paper 1 — Srivastava, Greff, Schmidhuber 2015. Highway Networks (ICML 2015 DL Workshop)](#paper-1--srivastava-greff-schmidhuber-2015-highway-networks-icml-2015-dl-workshop)
   - [Phase 1 — Foundational overview](#phase-1--foundational-overview-paper-1)
   - [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive-paper-1)
   - [Appendix — Section-by-section backbone](#appendix--section-by-section-backbone-paper-1)
2. [Paper 2 — Srivastava, Greff, Schmidhuber 2015. Training Very Deep Networks (NeurIPS 2015)](#paper-2--srivastava-greff-schmidhuber-2015-training-very-deep-networks-neurips-2015)
   - [Version notes — what is new vs. Paper 1](#version-notes--what-is-new-vs-paper-1)
3. [Paper 3 — Ha, Dai, Le 2016. HyperNetworks (ICLR 2017)](#paper-3--ha-dai-le-2016-hypernetworks-iclr-2017)
   - [Phase 1 — Foundational overview](#phase-1--foundational-overview-paper-3)
   - [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive-paper-3)
   - [Appendix — Section-by-section backbone](#appendix--section-by-section-backbone-paper-3)
4. [Paper 4 — Dauphin, Fan, Auli, Grangier 2016. Language Modeling with Gated Convolutional Networks (ICML 2017)](#paper-4--dauphin-fan-auli-grangier-2016-language-modeling-with-gated-convolutional-networks-icml-2017)
   - [Phase 1 — Foundational overview](#phase-1--foundational-overview-paper-4)
   - [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive-paper-4)
   - [Appendix — Section-by-section backbone](#appendix--section-by-section-backbone-paper-4)
5. [Cross-paper synthesis — gating → weight generation → modern GLU](#cross-paper-synthesis--gating--weight-generation--modern-glu)

---

## Paper 1 — Srivastava, Greff, Schmidhuber 2015. Highway Networks (ICML 2015 DL Workshop)

**PDF:** [`docs/project/references/Learning_Rate/sources/Srivastava et al. 2015 - Highway Networks.pdf`](sources/Srivastava%20et%20al.%202015%20-%20Highway%20Networks.pdf)
**Venue:** ICML 2015 Deep Learning Workshop (extended abstract, 6 pages)
**arXiv:** 1505.00387

### Phase 1 — Foundational overview (Paper 1)

**One-sentence summary.** A Highway layer is a normal feedforward layer **plus a learned scalar gate per unit** that lets the layer either transform its input or pass it through unchanged, and stacking these layers lets a network be trained at 100+ depth using plain SGD where ordinary feedforward networks of the same depth fail to optimise.

**Why it matters.** Before residual networks (He et al. 2015 ResNet, December 2015) and concurrent with them, this was the first architecture to demonstrate that *very* deep feedforward networks — up to 900 layers in the paper's experiments — could be trained directly with stochastic gradient descent without exotic two-stage procedures, auxiliary losses, or hand-tuned initialisation schemes. The mechanism the paper introduces is the **gate**: a sigmoid-output unit that learns, per layer and per dimension, how much of the layer's input to carry forward versus how much to transform. The same multiplicative-gating idea was already inside an LSTM cell; Highway Networks ported it from recurrent time-unrolling into feedforward depth-unrolling.

**Key empirical result.** On MNIST, plain feedforward networks of depth 10 train fine, but plain depth-100 networks essentially fail to optimise (final training cross-entropy ≈ 10⁻¹). Highway networks at the same depth-100 reach training cross-entropy ≈ 10⁻⁵, *with no loss of optimisability* as depth scales from 10 to 100 (Figure 1 in the paper). On CIFAR-10, Highway versions of the FitNet architectures match or exceed their performance trained directly with backpropagation, without the two-stage hint-based teacher distillation that FitNets required.

**Initial takeaway for the project.** Highway gating is the simplest possible "conditional architecture" in the multiplicative-gating family — the gate $T(x) \in [0,1]^n$ is itself conditioned on the layer input $x$, so different inputs route through the layer differently. If a modulator signal $m$ were appended to $x$ before the gate is computed (i.e. $T = \sigma(W_T [x; m] + b_T)$), the gate becomes a per-feature mask conditioned on $m$ — which is one of the simplest mechanisms a "modulator changes the policy's computation" thread can adopt. The carry-gate bias initialisation trick ($b_T \in \{-1, -2, -3\}$ to bias the network toward identity at start of training) is a robust piece of engineering wisdom that transfers directly to any gated design downstream.

### Phase 2 — Graduate-level deep dive (Paper 1)

#### Setup and notation

A plain feedforward layer applies a (typically affine + nonlinearity) transform $H$:
$$y = H(x, W_H), \qquad H : \mathbb{R}^n \to \mathbb{R}^n. \tag{P1.1}$$

For a network of $L$ such layers, $y_l = H_l(x_l, W_{H,l})$ with $x_{l+1} = y_l$. The pathology this paper attacks is that for $L \gtrsim 20$, gradients $\partial y_L / \partial x_1$ either vanish or explode through the composition $H_L \circ H_{L-1} \circ \cdots \circ H_1$, even with the variance-preserving Glorot/He initialisations of the time.

#### The Highway layer

Introduce **two extra learned transforms** that depend on the same input $x$:

- $T(x, W_T) \in (0,1)^n$ — the **transform gate**;
- $C(x, W_C) \in (0,1)^n$ — the **carry gate**.

The layer output is then a convex(-like) blend of the transformed signal and the input:
$$y = H(x, W_H) \odot T(x, W_T) + x \odot C(x, W_C), \tag{P1.2}$$
where $\odot$ is element-wise (Hadamard) product. For simplicity and parameter economy the paper **couples** the two gates as $C = 1 - T$:
$$\boxed{\; y = H(x, W_H) \odot T(x, W_T) + x \odot \bigl(1 - T(x, W_T)\bigr) \;} \tag{P1.3}$$

with
$$T(x, W_T) = \sigma(W_T x + b_T), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}. \tag{P1.4}$$

For (P1.3) to be well-typed, $\dim(x) = \dim(y) = \dim(H(x, W_H)) = \dim(T(x, W_T))$ — i.e. **the layer cannot change dimensionality**. When dimensionality change is needed, the authors insert a single non-highway plain layer and resume highway stacking afterwards.

The cost of one Highway layer over a plain layer is roughly **2× the parameters** of a plain layer of the same width (one set of weights for $H$, one for $T$) — modest, and the paper argues the depth-optimisation benefit dwarfs it.

#### Two limiting regimes

The point of the formulation in (P1.3) is that the layer can **smoothly interpolate** between two extreme behaviours, per dimension, based on $T$:
$$y_i = \begin{cases} x_i, & T_i(x, W_T) = 0 \quad \text{(carry: identity skip)} \\ H_i(x, W_H), & T_i(x, W_T) = 1 \quad \text{(transform: full nonlinear layer)} \end{cases} \tag{P1.5}$$

with the corresponding limits for the layer Jacobian:
$$\frac{\partial y}{\partial x} = \begin{cases} I, & T(x, W_T) = 0 \\ H'(x, W_H), & T(x, W_T) = 1. \end{cases} \tag{P1.6}$$

Strictly $T \in (0,1)$ (open interval) since $\sigma$ never saturates exactly, but at initialisation with $b_T \ll 0$, $T \approx 0$ uniformly and (P1.6) gives $\partial y / \partial x \approx I$, **at every layer**. Therefore at initialisation, $\partial y_L / \partial x_1 \approx I \cdot I \cdots I = I$, and gradients flow through the depth-$L$ stack as if the network were the identity map — **no vanishing, no exploding**. Training then has to *earn* the transformation: each layer must learn to drive $T$ towards 1 at the dimensions and inputs where it actually wants to compute something.

This is the architectural inductive bias that makes (P1.3) train: **identity is the default; transformation is the exception**. Contrast with a plain layer, where the default at initialisation is a random small linear transform that compounds chaotically over depth.

#### Initialisation: the negative bias trick

The transform-gate weight $W_T$ can be initialised by the same scheme as $W_H$ (zero-mean, variance-preserving). The crucial choice is $b_T$:
$$b_T \in \{-1, -2, -3, \ldots\} \quad \Longrightarrow \quad T(x, W_T)|_{\text{init}} \approx \sigma(b_T) \ll 0.5. \tag{P1.7}$$

For example $b_T = -3$ gives $T \approx 0.047$ at init, so the layer is doing ≈ 95% carry, ≈ 5% transform initially — the network looks like a stack of identity layers with small transformative perturbations. The authors find that any zero-mean initialisation of $W_H$ combined with $b_T \in [-1, -10]$ enables training. This contrasts with plain deep nets, where finding a working $W_H$ initialisation is essential and depends on the activation function (e.g. He init for ReLU, Glorot for tanh).

This trick is borrowed directly from Gers, Schmidhuber & Cummins 1999, who initialised LSTM forget gates with a negative bias to bias the cell toward retaining state in the early phase of training. The Highway paper makes the homology explicit: Highway depth-unrolling is to feedforward stacking as LSTM time-unrolling is to vanilla-RNN time-stacking.

#### Block structure and convolutional variant

The authors call a "block" the $i$-th unit of a Highway layer: it produces a scalar state $H_i(x)$ and a scalar gate output $T_i(x)$, and the layer output is the per-unit gated mix
$$y_i = H_i(x) \cdot T_i(x) + x_i \cdot (1 - T_i(x)). \tag{P1.8}$$

For convolutional Highway layers, $H$ and $T$ are both convolutions with weight-sharing and the same receptive field; zero-padding ensures the feature-map sizes of $H$, $T$, and the input all match for (P1.3) to apply.

#### Optimisation experiments — what "no depth dependence" means

The MNIST experiment (Figure 1 of the paper, summarised in the table on page 4 of the workshop version) trains plain networks (width 71) and Highway networks (width 50, roughly parameter-matched per layer) at depths $\{10, 20, 50, 100\}$, with a 40-config random search over learning rate, momentum, decay, activation ($\tanh$/ReLU), and (for Highway only) the transform-gate bias.

- Plain depth 10: trains well (cross-entropy $\sim 10^{-4}$).
- Plain depth 100: plateau at cross-entropy $\sim 10^{-1}$ — essentially fails.
- Highway depth 10–100: all reach $\sim 10^{-5}$, with the 100-layer Highway being roughly an *order of magnitude better* than its 10-layer counterpart and on par with the 10-layer plain network.

The authors also report training a 900-layer Highway on CIFAR-100 to 80 epochs with no sign of optimisation difficulty (though no full training curve is reported in the workshop version).

#### FitNet comparison (CIFAR-10)

Romero et al.'s FitNets are deep thin maxout networks that required a two-stage training procedure: pretrain a wide-shallow teacher (depth 5, 9M params), then train a thin-deep student (e.g. depth 19, 2.5M params) by hint-based distillation from the teacher's intermediate activations. The Highway paper shows that **Highway versions of these same architectures train directly with backprop** — no teacher, no two-stage procedure — and match or exceed FitNet test accuracy on CIFAR-10:

| Network | Layers | Params | Test acc. |
|---|---|---|---|
| Fitnet 1 (Romero) | 11 | ~250K | 89.01% |
| Highway 1 (this paper) | 11 | ~236K | 89.18% |
| Fitnet 4 (Romero) | 19 | ~2.5M | 91.61% |
| Highway 2 (this paper) | 19 | ~2.3M | 92.24% |

The point is methodological as much as numerical: **architecture replaces optimisation tricks**.

#### Inner-workings analysis

The paper inspects the transform-gate activity inside trained 50-layer Highway networks (MNIST and CIFAR-100). Three observations:

1. **Bias decreases further during training.** Initialised at $b_T = -2$ (MNIST) or $-4$ (CIFAR-100), the per-layer biases mostly become *more* negative through training, not less. The authors initially expected biases to be pushed positive (open the gates); the opposite happens.
2. **Average gate activity is low; per-example gate activity is sparse.** The mean transform-gate output over 10K samples is small (≈ 0.1–0.5), but for any single sample the gate vector is concentrated on a few dimensions — gates are **selective**, not uniformly open.
3. **Block outputs form "stripes" across depth.** Visualising block outputs as a depth × block heat-map for one sample shows long contiguous stripes — many units' outputs are nearly constant across many layers. The authors interpret this as the literal "information highway": once a value is computed in an early layer, it is carried unchanged through many later layers and used downstream.

The paper's framing: gating serves a dual role — *training-time* easier gradient flow, *inference-time* a learned routing policy where each input selects a few transformation paths.

### Appendix — Section-by-section backbone (Paper 1)

| Section | Core content |
|---|---|
| Abstract | Introduces Highway Networks: very deep nets trainable with SGD via learned gating, inspired by LSTM. Names the central mechanism: *information highways*. |
| 1. Introduction | Depth $\to$ accuracy on ImageNet; deep nets are theoretically more efficient; but training them is hard. Existing remedies: init schemes (Glorot, He), multi-stage training (FitNets), companion losses (DSN, GoogLeNet). This paper proposes a single-architecture solution. |
| 1.1 Notation | Bold for vectors/matrices, italic for transforms; $\sigma$ is sigmoid; $0$, $1$ are zero/one vectors. |
| 2. Highway Networks | Plain layer $y = H(x, W_H)$. Highway layer adds transform gate $T$ and carry gate $C$, coupled $C = 1 - T$: $y = H \odot T + x \odot (1 - T)$ (Eq. 3). Limit behaviours $T=0 \Rightarrow y = x$, $T=1 \Rightarrow y = H$ (Eq. 4). Jacobian likewise (Eq. 5). |
| 2.1 Constructing Highway Networks | Dimensions of $x, y, H, T$ must match; use a plain layer to change dim. Convolutional Highway: zero-pad to preserve spatial dims, share weights. |
| 2.2 Training Deep Highway Networks | Transform gate $T(x) = \sigma(W_T x + b_T)$; initialise $b_T \in \{-1, -3, \ldots\}$ to bias toward carry. Robust to choice of $W_H$ init and choice of activation in $H$. |
| 3. Experiments — 3.1 Optimization | MNIST, depths {10, 20, 50, 100}, plain (width 71) vs Highway (width 50). 40-config random search. Result: plain degrades sharply with depth, Highway essentially flat. 900-layer Highway on CIFAR-100 trained without optimisation issues. |
| 3.2 Comparison to Fitnets | CIFAR-10, Highway analogues of Fitnet 1 / Fitnet 4 trained directly with backprop match or exceed FitNet accuracies that required two-stage hint training. |
| 4. Analysis | Visualise gate biases, mean gate activity, per-example gate activity, block outputs across depth. Findings: biases drift more negative; gates are selective (sparse per-example); block outputs form "stripes" indicating information carried unchanged across many layers. |
| 5. Conclusion | Gating eases credit assignment and training of very deep nets; Highway opens depth experimentation without optimisation barriers. |

---

## Paper 2 — Srivastava, Greff, Schmidhuber 2015. Training Very Deep Networks (NeurIPS 2015)

**PDF:** [`docs/project/references/Learning_Rate/sources/Srivastava et al. 2015 - Training Very Deep Networks.pdf`](sources/Srivastava%20et%20al.%202015%20-%20Training%20Very%20Deep%20Networks.pdf)
**Venue:** Advances in Neural Information Processing Systems 28 (NeurIPS 2015), 9 pages
**Status:** Extended journal/conference version of Paper 1 (the workshop version's footnote on page 1 explicitly points to this paper as the "full paper extending this study", and Paper 2's footnote on page 2 confirms "This paper expands upon a shorter report on Highway Networks [31]").

### Version notes — what is new vs. Paper 1

The two papers describe **the same architecture**. Equations (P1.1)–(P1.7) above are identical in Paper 2 (Eqs. 1–5), the negative-bias initialisation scheme is identical, the convolutional Highway construction is identical, the depth-{10, 20, 50, 100} MNIST optimisation experiment is replicated with minor parameter changes (100-run search instead of 40-run, the headline finding is unchanged), and the FitNet-on-CIFAR-10 comparison is replicated with one extra row (a 32-layer "HighwayC"). The conceptual content needs no second review.

What is **new** in Paper 2 and worth pulling out:

**N1. Pilot MNIST results with full convolutional Highway (Table 1).** Two 10-layer convolutional Highway networks (width 16 and width 32) trained on MNIST give 99.43% and 99.55% test accuracy with only 39K and 151K parameters — competitive with Maxout (420K params, 99.55%) and DSN (350K, 99.61%). Demonstrates the architecture is not just a deep-toy result; it works as a small competitive convnet too.

**N2. State-of-the-art comparisons on CIFAR-10 and CIFAR-100 (Table 3).** A Highway convolutional network with global average pooling achieves 92.40% on CIFAR-10 and **67.76% on CIFAR-100**, both with typical data augmentation. The CIFAR-100 number was state-of-the-art for *that experimental regime* (no extra data, no extra-large model) at the time of NeurIPS 2015 — exceeding All-CNN (66.29%), DSN (65.43%), NiN (64.32%).

**N3. Information-routing analysis is class-conditional, not just sample-conditional (Section 4.1, Figure 3).** Paper 1 showed that the *mean* transform-gate activity is low and the *per-example* activity is sparse. Paper 2 goes one level deeper: it computes the **per-class mean** transform-gate activity and shows it differs visibly from the overall mean. For MNIST digits 0 and 7 the differences are concentrated in the first ≈ 15 layers; for CIFAR-100 they spread across more depth. The takeaway the authors emphasise:

> The gating system acts not just as a mechanism to ease training, but also as an important part of the computation in a trained network.

This is the key concession to gating-as-conditional-computation, not just gating-as-optimisation-aid. It is the most important *new* finding in Paper 2 and the one most relevant to the project's modulator thread.

**N4. Lesion analysis (Section 4.2, Figure 4).** The authors evaluate a trained 50-layer Highway network by **forcefully setting the transform gate of one layer at a time to $T = 0$** (i.e. making that single layer the identity map) and measuring how training-set error degrades:

- **MNIST**: closing any of layers 1–15 hurts performance noticeably; closing any of layers 15–45 has essentially no effect. About 60% of the layers contribute nothing measurable. Interpretation: MNIST is simple enough that the network only "uses" depth ≈ 15.
- **CIFAR-100**: closing any of layers 1–40 hurts performance noticeably. The network uses essentially its full depth for the harder dataset.

This is a strong piece of evidence: **a Highway network learns to use exactly as much depth as the problem requires, leaving the rest as identity passthrough**. The authors note this is impossible to do cleanly with plain networks — you cannot lesion a plain layer to identity without rewriting weights.

**N5. Discussion of related work expanded.** Paper 2 reviews additional optimiser work (Hessian-free, Adam-precursor, saddle-point analyses), additional initialisation work (random-walk init, exact-solution dynamics), competing strategies (deep supervision, hint-based distillation), and clarifies the relationship to LSTM and skip connections. Paper 2 also mentions a contemporaneous "similar LSTM-inspired model" — Grid LSTM (Kalchbrenner et al. 2015) — without merging into it. ResNet (He et al., December 2015) is not cited because Paper 2 was finalised before its release; ResNet is the obvious sequel where the carry gate is replaced by an unconditional skip ($C \equiv 1$, $T \equiv 1$, $y = x + H(x)$).

**N6. Engineering specifics for production setups.** Paper 2 includes the practical recipe: SGD with momentum, exponential LR decay, ReLU activation in $H$, batch-norm-friendly hyperparameter ranges, and the source-code release at idsia.ch/~rupesh/very_deep_learning/.

**Bottom line.** Paper 2 is the canonical citation for Highway Networks (the workshop paper has 6 pages and a footnote; the NeurIPS paper has 9 pages and the lesion/class-conditional analyses). If a reader wants the architecture, both papers say the same thing; if a reader wants empirical justification at competitive scale plus the analyses that distinguish "gating as ease-of-training" from "gating as conditional computation in a trained network", they want Paper 2.

---

## Paper 3 — Ha, Dai, Le 2016. HyperNetworks (ICLR 2017)

**PDF:** [`docs/project/references/Learning_Rate/sources/Ha et al. 2016 - HyperNetworks.pdf`](sources/Ha%20et%20al.%202016%20-%20HyperNetworks.pdf)
**Venue:** International Conference on Learning Representations 2017 (29 pages incl. appendices)
**arXiv:** 1609.09106

### Phase 1 — Foundational overview (Paper 3)

**One-sentence summary.** A hypernetwork is a *small* neural network that **outputs the weights** of a *larger* main network, where the main network does the actual task and the hypernetwork's input is either a learned per-layer embedding (giving a *static* hypernetwork for convnets) or the previous timestep's hidden state (giving a *dynamic* hypernetwork — HyperRNN/HyperLSTM — whose weights change every timestep based on the running state).

**Why it matters.** The paper does for *weight generation* what Highway did for *information gating*: it gives a clean, end-to-end-trainable mechanism by which one network's behaviour is controlled by another network's output. The static case is best read as a **weight-sharing relaxation**: instead of every convolutional layer having independent weights (the convnet extreme) or every recurrent step sharing one weight matrix (the RNN extreme), each layer gets its weights *generated* from a small per-layer embedding through a shared generator, so layers softly share structure through the generator while remaining distinguishable through their embeddings. The dynamic case is the more profound one: **the weight matrix of the main LSTM is regenerated at every timestep based on the input and the previous state**, so the main network is not a fixed function from sequences to predictions — it is a sequence-dependent family of functions.

For the project's interest, this is the canonical formulation of "a modulator signal generates the policy's parameters": the modulator is the input to the hypernetwork, the hypernetwork emits the weights, the main policy uses those weights. FiLM (reviewed elsewhere) is a rank-1 hypernetwork in disguise — it generates only per-channel scale and shift; the full hypernetwork formulation generates the whole weight matrix.

**Key empirical results.**

- **Static hypernet on MNIST (small convnet, 2nd-layer kernel $7 \times 7 \times 16 \times 16 = 12{,}544$ weights).** Conventional baseline: 0.72% test error. Hypernet generating the 2nd-layer kernel from a 4-dim embedding through a 4,244-parameter hypernet: 0.76% test error. *A 12,544-weight tensor is represented by a 4-dim vector + 4K-param generator.*
- **Static hypernet on CIFAR-10 with Wide ResNet 40-1 / 40-2 backbones.** Generating *all 36 layers* of conv2/3/4 from one shared hypernetwork: baseline WRN40-1 has 6.73% error with 0.563M params; HyperResNet40-1 has 8.02% error with **0.097M params** (5.8× compression for +1.29% error). HyperResNet40-2 is 7.23% vs 5.66% (15× compression for +1.57%). Strong compression for modest accuracy loss.
- **Dynamic HyperLSTM on Penn Treebank character LM.** 1000-unit HyperLSTM reaches 1.265 bits-per-character (test); 1000-unit LayerNorm-HyperLSTM reaches 1.250; the same-size LayerNorm LSTM baseline reaches 1.267. **HyperLSTM rivals layer normalisation — a state-of-the-art technique designed by hand — purely by letting the dynamic hypernetwork learn an analogous scaling policy.**
- **Dynamic HyperLSTM on enwik8.** 1800-unit LayerNorm-HyperLSTM reaches 1.353 bpc (vs 1.402 for the LayerNorm LSTM baseline); 2048-unit reaches 1.340, near state-of-the-art.
- **Handwriting generation (IAM)**, **neural machine translation (WMT En→Fr)**: HyperLSTM cells dropped into the GNMT-32K architecture give 40.03 BLEU vs 38.95 for the single-LSTM GNMT baseline, beating even the 8-LSTM ensemble (40.35) within a few tenths.

**Initial takeaway for the project.** Two design knobs are crucial. First, the **embedding dimension $N_z$** determines how much per-layer information the hypernet can encode; the paper uses $N_z = 4$ for MNIST static, $N_z = 64$ for CIFAR static, $N_z = 4, 16, 32$ for dynamic LSTMs. Second, the **memory-efficient "row-scaling" variant** (Eq. P3.7 below) is what makes dynamic hypernets practical for production-size LSTMs: rather than generating the full $W \in \mathbb{R}^{N_h \times N_h}$, the hypernet generates only a *row-wise scaling vector* $d(z) \in \mathbb{R}^{N_h}$, and the main network's weight is $W(z) = \text{diag}(d(z)) \cdot W$. This is structurally identical to FiLM-on-weights and to the gain modulation in LayerNorm — the difference is that the scales are emitted by a small RNN rather than computed from input statistics. This row-scaling form is the most directly applicable architecture for the project's modulator thread.

### Phase 2 — Graduate-level deep dive (Paper 3)

#### Static hypernetworks: weight factorisation for convnets

The setting: a convnet with $D$ layers, where layer $j$'s kernel is a matrix
$$K^j \in \mathbb{R}^{N_{\text{in}} f_{\text{size}} \times N_{\text{out}} f_{\text{size}}} \tag{P3.1}$$
(with $f_{\text{size}}$ the filter spatial size and $N_{\text{in}}, N_{\text{out}}$ the in/out channel counts). The hypernetwork's job is, for each layer $j$, to take a layer embedding $z^j \in \mathbb{R}^{N_z}$ and emit $K^j$:
$$K^j = g(z^j), \qquad j = 1, \ldots, D. \tag{P3.2}$$

**Two-layer linear generator.** The kernel matrix $K^j$ is sliced into $N_{\text{in}}$ smaller blocks $K_i^j \in \mathbb{R}^{f_{\text{size}} \times N_{\text{out}} f_{\text{size}}}$ ($i = 1, \ldots, N_{\text{in}}$), and each block is generated independently:
$$a_i^j = W_i z^j + B_i, \qquad i = 1, \ldots, N_{\text{in}}, \tag{P3.3a}$$
$$K_i^j = \langle W_{\text{out}}, a_i^j \rangle + B_{\text{out}}, \tag{P3.3b}$$
$$K^j = \bigl[\, K_1^j \;|\; K_2^j \;|\; \cdots \;|\; K_{N_{\text{in}}}^j \,\bigr]. \tag{P3.3c}$$

Here $W_i \in \mathbb{R}^{d \times N_z}$, $B_i \in \mathbb{R}^d$ (the first linear layer of the hypernet, with $N_{\text{in}}$ separate copies, one per output block); $W_{\text{out}} \in \mathbb{R}^{f_{\text{size}} \times N_{\text{out}} f_{\text{size}} \times d}$ and $B_{\text{out}} \in \mathbb{R}^{f_{\text{size}} \times N_{\text{out}} f_{\text{size}}}$ (the second linear layer, **shared across all $N_{\text{in}}$ slices**). The notation $\langle W_{\text{out}}, a \rangle$ is a tensor-mode product: contracting the 3-tensor $W_{\text{out}}$ along its last axis with the vector $a \in \mathbb{R}^d$ gives a matrix of shape $f_{\text{size}} \times N_{\text{out}} f_{\text{size}}$.

For dimensional intuition, take $d = N_z$ (the paper's default). Total trainable parameters in the hypernet:

- Per-layer embedding: $N_z \cdot D$ (one $z^j$ per layer)
- First hypernet layer (shared across layers but $N_{\text{in}}$ separate slices): $N_{\text{in}} \cdot d \cdot (N_z + 1)$
- Second hypernet layer (shared across layers and slices): $f_{\text{size}} \cdot N_{\text{out}} \cdot f_{\text{size}} \cdot (d + 1)$

Compared to the unshared baseline of $D \cdot N_{\text{in}} f_{\text{size}} \cdot N_{\text{out}} f_{\text{size}}$ for the main network's kernels, the hypernet's parameter count is roughly **factor-of-$D$ smaller** when $N_z, d \ll N_{\text{in}}, N_{\text{out}}$ — the compression scales with the depth.

**Why two-layer rather than one-layer?** A naive one-layer hypernet $K^j = \text{Reshape}(W_g z^j + b_g)$ would need $W_g \in \mathbb{R}^{(N_{\text{in}} f_{\text{size}} \cdot N_{\text{out}} f_{\text{size}}) \times N_z}$ — a single big matrix per layer pattern. The two-layer factored form (P3.3) keeps the $W_{\text{out}}$ tensor *shared* across all $N_{\text{in}}$ input slices, so the per-slice generators only differ in their first-layer $W_i$. This is structurally similar to the hierarchically semiseparable matrix decomposition (Xia et al. 2010) — a low-rank-plus-shared-basis factorisation.

**Variable kernel sizes.** Real ResNets use kernels of different shapes (e.g. $3 \times 3 \times 16 \times 16$, $3 \times 3 \times 32 \times 32$, etc., where channel counts grow with depth). The paper's solution: fix a basic size (16 in their experiments), generate basic-size kernels from individual embeddings, and **tile** them to form larger kernels. For example a $3 \times 3 \times 32 \times 64$ kernel is tiled from eight $3 \times 3 \times 16 \times 16$ basic kernels, each from its own embedding:
$$K_{32 \times 64} = \begin{pmatrix} K_1 & K_2 & K_3 & K_4 \\ K_5 & K_6 & K_7 & K_8 \end{pmatrix}. \tag{P3.4}$$
Larger kernels cost more embeddings; deep ResNets pay a per-channel-growth cost in embedding count.

#### Dynamic hypernetworks: HyperRNN

Now the input to the hypernet is not a static per-layer embedding but the *previous timestep's hidden state* of the main RNN. The plain RNN update is
$$h_t = \phi(W_h h_{t-1} + W_x x_t + b), \tag{P3.5}$$
with $W_h \in \mathbb{R}^{N_h \times N_h}$, $W_x \in \mathbb{R}^{N_h \times N_x}$, $b \in \mathbb{R}^{N_h}$, $\phi = \tanh$ (or relu). In the HyperRNN, **the weights $W_h, W_x, b$ are themselves time-varying functions of an embedding** $z = (z_h, z_x, z_b)$ generated by a small "HyperRNN cell" running alongside the main RNN:

$$h_t = \phi\bigl(W_h(z_h) h_{t-1} + W_x(z_x) x_t + b(z_b)\bigr), \tag{P3.6a}$$

with the full-matrix dependence
$$W_h(z_h) = \langle W_{hz}, z_h \rangle, \quad W_x(z_x) = \langle W_{xz}, z_x \rangle, \quad b(z_b) = W_{bz} z_b + b_0, \tag{P3.6b}$$

where $W_{hz} \in \mathbb{R}^{N_h \times N_h \times N_z}$, $W_{xz} \in \mathbb{R}^{N_h \times N_x \times N_z}$, $W_{bz} \in \mathbb{R}^{N_h \times N_z}$, and $\langle \cdot, \cdot \rangle$ contracts the last axis with $z \in \mathbb{R}^{N_z}$ to yield a 2-tensor. The embeddings $z_h, z_x, z_b$ are produced by the HyperRNN cell, which runs on the concatenation $\hat{x}_t = [h_{t-1}; x_t]$:
$$\hat{h}_t = \phi(W_{\hat{h}} \hat{h}_{t-1} + W_{\hat{x}} \hat{x}_t + \hat{b}), \tag{P3.6c}$$
$$z_h = W_{\hat{h}h} \hat{h}_{t-1} + b_{\hat{h}h}, \quad z_x = W_{\hat{h}x} \hat{h}_{t-1} + b_{\hat{h}x}, \quad z_b = W_{\hat{h}b} \hat{h}_{t-1}. \tag{P3.6d}$$

**Memory cost of (P3.6).** The full-tensor formulation has $N_h \cdot N_h \cdot N_z + N_h \cdot N_x \cdot N_z + N_h \cdot N_z$ parameters for the generator — i.e. $N_z$ times the memory of a plain RNN. For $N_h \approx 1000$ and $N_z \approx 16$, that's a 16× blow-up. Impractical.

#### Memory-efficient HyperRNN: row-wise weight scaling

Instead of generating the full $W$ from $z$, generate a per-row scaling vector $d(z) \in \mathbb{R}^{N_h}$ and apply it as a left-multiplying diagonal:
$$W(z) = W \odot d(z) = \begin{pmatrix} d_0(z) W_0 \\ d_1(z) W_1 \\ \vdots \\ d_{N_h}(z) W_{N_h} \end{pmatrix}, \tag{P3.7}$$
where $W_i$ is the $i$-th row of a baseline weight matrix $W$ (one set of baseline weights per matrix, learned end-to-end alongside the hypernet), and $d_i(z)$ is the $i$-th entry of a vector emitted by a linear projection of $z$.

The row-scaling form is **equivalent to an element-wise multiplication** in the matrix-vector product:
$$W(z) h = \bigl(W \odot d(z)\bigr) h = d(z) \odot (W h), \tag{P3.8}$$
which is *much* cheaper than generating and materialising a full $N_h \times N_h$ matrix at every timestep. The cost is $O(N_h)$ for the scaling vector versus $O(N_h^2)$ for the matrix.

The HyperRNN update with row-scaling is then
$$h_t = \phi\bigl(d_h(z_h) \odot W_h h_{t-1} + d_x(z_x) \odot W_x x_t + b(z_b)\bigr), \tag{P3.9a}$$
with
$$d_h(z_h) = W_{hz} z_h, \quad d_x(z_x) = W_{xz} z_x, \quad b(z_b) = W_{bz} z_b + b_0. \tag{P3.9b}$$

**The connection to other architectures.** Equation (P3.9a) is structurally identical to:

- **LayerNorm/BatchNorm + learned gains** — where the learned gain $\gamma$ is replaced by the dynamic $d(z)$ that varies with input and time.
- **FiLM** — where the per-channel scale $\gamma(c)$ and shift $\beta(c)$ are generated from conditioning $c$; HyperLSTM's row-scaling is the same idea applied to pre-activation values, with the scale generated by an *embedded* RNN rather than a feedforward MLP.
- **Multiplicative RNN / Multiplicative Integration RNN** (Sutskever et al. 2011; Wu et al. 2016) — earlier proposals where input and hidden state are multiplied rather than added; HyperRNN generalises this by letting the multiplicative factor itself be learned dynamically.

The paper's framing is that **HyperRNN learns its own scaling policy**, which empirically rivals — and sometimes beats — hand-engineered statistical-moment-based normalisation, suggesting that the optimal per-step rescaling policy is *not* simply "subtract mean, divide by std" but something the network discovers.

#### HyperLSTM

The HyperLSTM is the LSTM-version of HyperRNN: there are four gates ($i, g, f, o$) in a standard LSTM,
$$\begin{aligned} i_t &= W_h^i h_{t-1} + W_x^i x_t + b^i, \\ g_t &= W_h^g h_{t-1} + W_x^g x_t + b^g, \\ f_t &= W_h^f h_{t-1} + W_x^f x_t + b^f, \\ o_t &= W_h^o h_{t-1} + W_x^o x_t + b^o, \\ c_t &= \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \phi(g_t), \\ h_t &= \sigma(o_t) \odot \phi(c_t), \end{aligned} \tag{P3.10}$$
and the HyperLSTM replaces each of the eight weight matrices ($W_h^y, W_x^y$ for $y \in \{i, g, f, o\}$) and four biases ($b^y$) with a row-scaled version generated from a per-gate embedding $(z_h^y, z_x^y, z_b^y)$:
$$y_t = \text{LN}\bigl(d_h^y(z_h^y) \odot W_h^y h_{t-1} + d_x^y(z_x^y) \odot W_x^y x_t + b^y(z_b^y)\bigr), \tag{P3.11}$$
$$d_h^y = W_{hz}^y z_h^y, \quad d_x^y = W_{xz}^y z_x^y, \quad b^y = W_{bz}^y z_b^y + b_0^y, \tag{P3.12}$$
with $\text{LN}$ optionally LayerNorm. The embeddings are emitted by a small auxiliary HyperLSTM cell that itself runs LSTM dynamics on $\hat{x}_t = [h_{t-1}; x_t]$.

Key implementation specifics from Appendix A.2.3 of the paper:

- The HyperLSTM cell uses orthogonal initialisation for weights and zero for biases.
- The first two embedding-projection equations ($z_h, z_x$) have weights initialised to **zero** and biases initialised to **one** — so the row-scaling vectors $d_h, d_x$ are initially the identity (no scaling) and the HyperLSTM at init reduces to a standard LSTM.
- The third equation ($z_b$) has weights initialised as a small Gaussian (std 0.01), so the dynamic bias term is initially small.
- The row-scaling weight matrices $W_{hz}, W_{xz}$ are initialised to the constant $0.1/N_z$ to follow the Recurrent BatchNorm initialisation of 0.1 (rather than 1) which the paper notes helps gradient flow.

#### Analysis: what does the dynamic hypernet learn?

Two analyses in the paper are particularly relevant for the project.

**Histograms of $\phi(c_t)$ (Figure 5).** The vanilla LSTM saturates strongly (most cell activations at $\pm 1$). LayerNorm reduces saturation. HyperLSTM *increases* saturation relative to LayerNorm, but achieves equivalent or better validation loss. The authors interpret this as evidence that the dynamic scaling policy is doing something *qualitatively different* from statistical normalisation, even though both approaches reach similar perplexities.

**Weight-change visualisation during generation (Figures 4, 7).** When the HyperLSTM generates text or handwriting, plotting $\|W^y_{h,t} - W^y_{h,t-1}\|$ over time (where $W^y_{h,t}$ is the row-scaled weight at timestep $t$) reveals **regime changes**: the weights are nearly constant during the body of words and change sharply at word boundaries, between brackets, between handwriting strokes. The dynamic hypernet has learned to keep the main network's effective parameters fixed during "deterministic" generation and switch them at boundaries — exactly the kind of context-dependent reparametrisation that a modulator system would do.

### Appendix — Section-by-section backbone (Paper 3)

| Section | Core content |
|---|---|
| Abstract | Hypernetworks: small net generates weights of larger net. End-to-end trainable. Static for convnets, dynamic for recurrent nets. Achieves near-SOTA on character LM, handwriting, NMT; respectable on image classification with far fewer params. |
| 1. Introduction | Frames the genotype-phenotype analogy. Embedding-based static hypernets (compact per-layer encoding); dynamic hypernets generate input-dependent and time-varying weights. |
| 2. Motivation and Related Work | Roots in HyperNEAT (CPPN-based weight generation, evolved), Compressed Weight Search (DCT-based), DPPNs (differentiable pattern producers), ACDC nets, Schmidhuber's fast weights, Predicting Parameters (Denil 2013), Dynamic Filter Networks (De Brabandere 2016). Distinguishes from these by end-to-end gradient-based training of both nets. |
| 3. Methods | Two regimes: static (one embedding per layer of main convnet) and dynamic (input-dependent embedding for recurrent main net). |
| 3.1 Static Hypernetwork | Eq. (P3.2)–(P3.3): two-layer linear generator. Layer embeddings $z^j \in \mathbb{R}^{N_z}$ as learned per-layer codes. Tiling Eq. (P3.4) handles non-uniform kernel shapes in ResNets. |
| 3.2 Dynamic Hypernetwork | HyperRNN Eq. (P3.6): full-tensor weight generation from $(z_h, z_x, z_b)$ produced by a separate HyperRNN cell on $[h_{t-1}; x_t]$. Memory issue: factor-$N_z$ blow-up. |
| 3.2 (cont.) memory-efficient form | Eq. (P3.7)–(P3.9): row-scaling formulation $W(z) = W \odot d(z)$, equivalent to element-wise multiplication on $Wh$. Discussed in relation to LayerNorm, multiplicative RNN, multiplicative integration RNN. |
| 3.2 (cont.) HyperLSTM | Eq. (P3.10)–(P3.12): row-scaling applied to all 8 weight matrices of LSTM ($i, g, f, o$ × $h, x$), with optional LayerNorm. Implementation details in Appendix A.2. |
| 4. Experiments | Six experiment families covering image classification (static) and four sequence tasks (dynamic). |
| 4.1 Static + MNIST | Generates 2nd-layer conv kernel ($7 \times 7 \times 16 \times 16 = 12{,}544$ weights) from $N_z = 4$ embedding + 4,244-param hypernet. Test error 0.76% vs 0.72% baseline. |
| 4.2 Static + CIFAR-10 WideResNet | Generates all 36 conv layers of conv2/3/4 in WRN40-1/40-2 from a single shared hypernet with $N_z = 64$. WRN40-1: 6.73% → 8.02% (5.8× compression). WRN40-2: 5.66% → 7.23% (15× compression). |
| 4.3 HyperLSTM + Penn Treebank | 1000-unit HyperLSTM reaches 1.265 bpc test; LN-HyperLSTM 1.250; large-embed LN-HyperLSTM 1.233; 2-layer LN-HyperLSTM 1.219 (new SOTA at the time). |
| 4.4 HyperLSTM + enwik8 | 1800-unit HyperLSTM 1.391 bpc; LN-HyperLSTM 1.353; 2048-unit LN-HyperLSTM 1.340 (near SOTA). |
| 4.5 HyperLSTM + handwriting generation | 900-unit HyperLSTM (no LayerNorm) reaches log-loss -1162 vs -1055 for vanilla LSTM, -1096 for LayerNorm LSTM. Interestingly LayerNorm hurts the HyperLSTM here. |
| 4.6 HyperLSTM + neural machine translation | GNMT-32K + HyperLSTM: 40.03 BLEU on WMT En→Fr, beating single LSTM GNMT (38.95) and approaching 8-LSTM ensemble (40.35). |
| 5. Conclusion | Hypernetworks are efficient, scalable, end-to-end-trainable. Work for both static (convnet weight compression) and dynamic (sequence-conditional weight generation) regimes. |
| Appendix A.2 | HyperLSTM equations, weight initialisation recipe (orthogonal init, zero init for $z_h, z_x$ weights with ones bias so $d \approx 1$ at start, 0.1/N_z constant init for $W_{hz}, W_{xz}$). |
| Appendix A.3 | Per-experiment hyperparameters: optimiser, learning rates, dropout, sequence lengths, embedding sizes per task. |

---

## Paper 4 — Dauphin, Fan, Auli, Grangier 2016. Language Modeling with Gated Convolutional Networks (ICML 2017)

**PDF:** [`docs/project/references/Learning_Rate/sources/Dauphin et al. 2016 - Language modeling with gated convolutional networks.pdf`](sources/Dauphin%20et%20al.%202016%20-%20Language%20modeling%20with%20gated%20convolutional%20networks.pdf)
**Venue:** International Conference on Machine Learning 2017
**arXiv:** 1612.08083

### Phase 1 — Foundational overview (Paper 4)

**One-sentence summary.** A Gated Linear Unit (GLU) is a layer of the form $h = (X \ast W + b) \odot \sigma(X \ast V + c)$ — a linear convolution multiplied element-wise by a sigmoid-gated convolution of the *same input* — and stacks of GLUs (with residual connections, adaptive softmax, weight normalisation) match or beat the best LSTM-based language models on Google Billion Word and WikiText-103 while being parallelisable across sequence positions.

**Why it matters.** Two things. First, GLU shows that **gating works in convolutional language models** — recurrence is not required for state-of-the-art LM. Second, and more enduringly, the GLU formula — *one linear stream multiplied by one sigmoid-gated stream of the same input* — became the canonical "gated activation" of the Transformer era, generalised to SwiGLU (LLaMA), GeGLU (T5-v1.1), ReGLU, and the broader family of "X-GLU" feed-forward variants that have largely replaced the ReLU-FFN inside modern LLMs. The paper is therefore the canonical citation for what is now the default FFN sublayer in modern Transformers.

The critical theoretical contribution is the **gradient analysis** that explains why GLU works better than the contemporaneous "Gated Tanh Unit" $h = \tanh(X \ast W + b) \odot \sigma(X \ast V + c)$ of van den Oord et al.'s Conditional PixelCNN. The GLU's gradient has a **linear path** (Eq. P4.3 below) where activated gates pass gradient through *without downscaling* by the derivative of a saturating nonlinearity; the GTU's gradient (Eq. P4.2) has two saturating-derivative factors that compound across depth and vanish. Same observation as ReLU vs sigmoid for plain activations, here applied to the gating family.

**Key empirical results.**

- **Google Billion Word:** GCNN-13 reaches test perplexity 38.1 on a single GPU; GCNN-14 Bottleneck reaches 31.9 on 8 GPUs. Comparable LSTM-2048 baseline (Grave et al.): 43.9 with the same adaptive softmax. The much larger 2-layer LSTM-8192-1024 of Jozefowicz et al. (using full softmax, 32 GPUs, 3 weeks) reaches 30.6 — GCNN-14B is within 1.3 perplexity using 8 GPUs for 2 weeks.
- **WikiText-103:** GCNN-14 reaches 37.2 test perplexity on 4 GPUs, beating the 48.7 of the LSTM-1024 baseline. SOTA at time of writing.
- **GLU vs alternatives (ablation).** On WikiText-103: GLU outperforms GTU ($\tanh \odot \sigma$), Tanh-only, and ReLU-only, with GLU and ReLU converging fastest (both have a linear path); GTU and Tanh stagnate from vanishing gradient.
- **GLU vs Bilinear vs Linear (ablation).** On Google Billion Word: Linear plateaus at ~115 perplexity; Bilinear (= GLU without the sigmoid) reaches ~61; full GLU reaches ~41. The bilinear unit alone beats Kneser-Ney 5-gram — surprising — but the sigmoid gate on top adds ~20 perplexity points, confirming that the *non-linearity carried by the gate*, not just the multiplicative interaction, is what is doing work.
- **Responsiveness.** GCNN's responsiveness (per-token latency, no batching) is 20× the LSTM-2048, because convolutions parallelise across sequence positions while LSTMs are intrinsically sequential.

**Initial takeaway for the project.** GLU is the simplest gated nonlinearity with a healthy gradient. The formula generalises trivially: replace the second branch $X \ast W + b$ with a fully-connected projection $XW + b$, replace the sigmoid $\sigma$ with any squashing function (GeLU, Swish), and you have the modern "GLU variant" family. For the project, GLU is also the cleanest example of "gating as conditional computation" that doesn't require the bookkeeping of a Highway carry path or the parameter overhead of a hypernetwork — the gate is computed from the *same input* as the value branch, with separate weights $W$ (value) and $V$ (gate). If the gate were instead computed from a modulator signal $m$ (i.e. $\sigma(M \ast U + d)$ where $M$ is the modulator embedded as a feature map), GLU becomes a **modulator-conditioned multiplicative gate** — structurally identical to FiLM's $\gamma$-only branch.

### Phase 2 — Graduate-level deep dive (Paper 4)

#### The GLU layer

For input feature map $X \in \mathbb{R}^{N \times m}$ ($N$ sequence positions, $m$ input channels), a GLU layer with kernel width $k$ and output channels $n$ is
$$\boxed{\; h_l(X) = (X \ast W + b) \odot \sigma(X \ast V + c) \;} \tag{P4.1}$$
where $W \in \mathbb{R}^{k \times m \times n}$, $V \in \mathbb{R}^{k \times m \times n}$, $b, c \in \mathbb{R}^n$. The first term is the **value branch** — a plain linear convolution — and the second term is the **gate branch** — another linear convolution with its own weights, passed through a sigmoid. The two branches share the same input $X$ but have independent weights $W$ and $V$.

The output channel count $n$ is the same on both branches (otherwise element-wise multiplication is undefined). Total parameter count per layer is $2 \cdot k \cdot m \cdot n + 2n$ — double a plain conv layer of the same shape, comparable to a Highway layer.

**Causal padding.** For autoregressive language modelling, the kernel must not see future tokens. Pad $k-1$ zero tokens at the *beginning* of the sequence so that position $t$'s output depends only on positions $\le t$. (The Highway paper is non-causal; GCNN is causal.)

**Residual wrapping.** Each GLU + conv block is wrapped in a pre-activation residual block: $\text{output} = X + \text{GLU}(X)$. Deep stacks use bottleneck blocks $1 \times 1 \to k \times 1 \to 1 \times 1$ in the He et al. style for computational efficiency.

#### Gradient analysis — the linear-path argument

The Gated Tanh Unit (van den Oord et al. 2016b for Conditional PixelCNN) is
$$h_{\text{GTU}}(X) = \tanh(X \ast W + b) \odot \sigma(X \ast V + c). \tag{P4.GTU}$$
Its gradient with respect to a downstream loss, simplified to the single-input case $X$ (writing the two affine arguments as $X$ on both sides to focus on the gating structure), is
$$\nabla[\tanh(X) \odot \sigma(X)] = \underbrace{\tanh'(X) \nabla X \odot \sigma(X)}_{\text{value-path, downscaled by } \tanh'} + \underbrace{\sigma'(X) \nabla X \odot \tanh(X)}_{\text{gate-path, downscaled by } \sigma'}. \tag{P4.2}$$

**Both terms** carry a saturating-nonlinearity derivative factor ($\tanh' \in (0,1)$ or $\sigma' \in (0, 0.25)$). When the network is deep, products of such factors across layers compound to a vanishing gradient (same pathology as plain $\tanh$ networks).

The GLU's gradient with respect to $X$, replacing $\tanh$ with identity, is
$$\nabla[X \odot \sigma(X)] = \underbrace{\nabla X \odot \sigma(X)}_{\text{linear-path, no downscaling}} + \underbrace{\sigma'(X) \nabla X \odot X}_{\text{gate-path, downscaled by } \sigma'}. \tag{P4.3}$$

The first term has **no derivative factor**: when $\sigma(X) \approx 1$ (gate open), the gradient propagates back essentially unchanged. The authors describe this as a **"multiplicative skip connection"** — analogous to ResNet's additive skip, but the bypass is gated by $\sigma$. Where ResNet gives $\partial y / \partial x = I + \partial H / \partial x$ (additive identity in the Jacobian), GLU gives effectively $\partial y / \partial x \supset \text{diag}(\sigma(X)) \cdot I$ (multiplicative identity scaled by the gate). At init with $b, c$ small, $\sigma(X \ast V + c) \approx 0.5$ for typical $X$, so roughly half of the gradient propagates back through the linear path per layer — qualitatively much better than a tanh stack.

This argument is the paper's *theoretical* explanation for the empirical hierarchy GLU > ReLU > GTU > Tanh on WikiText-103.

#### Why simplified gating beats LSTM-style gating

Three reasons given by the paper:

1. **Linear gradient path (above).** GLU has it; GTU doesn't.
2. **No need for forget gates.** Convnets do not have the indefinite-horizon vanishing-gradient pathology of recurrent nets, because each layer's output depends on a *bounded* receptive field (at most $k$ positions back per layer, $L \cdot k$ across the stack). The paper's section 3 explicitly drops the input/forget machinery and keeps only an "output-style gate". This is a substantive design simplification, not just a re-parametrisation.
3. **Empirically (ablation, Section 5.2):** GLU beats GTU, ReLU, and Tanh on the same architecture with matched parameter counts. Linear < Bilinear < GLU on Google Billion Word, isolating that *both* the multiplicative interaction (bilinear vs linear) *and* the nonlinear gating (GLU vs bilinear) contribute.

#### Computational efficiency

Convnets parallelise both across batch and across sequence position; LSTMs only across batch. Concretely, for a model targeting 43.9 test perplexity on Google Billion Word (Table 4):

| Model | Throughput CPU (tok/s) | Throughput GPU (tok/s) | Responsiveness GPU (tok/s) |
|---|---|---|---|
| LSTM-2048 | 169 | 45,622 | 2,282 |
| GCNN-9 | 121 | 29,116 | 29,116 |
| GCNN-8 Bottleneck | 179 | 45,878 | 45,878 |

Throughput is similar (LSTM benefits from cuDNN); **responsiveness** — single-sequence latency — is **20× higher** for GCNN because each token's output is computed in parallel rather than sequentially.

#### Context-size analysis

Figure 4 of the paper tests GCNN-style models with varying maximum context size on Google Billion Word and WikiText-103. The finding: **performance improves with context up to ≈ 20–40 tokens, then plateaus**, even on WikiText-103 where full Wikipedia articles are available (~4000 tokens). The conclusion: **unbounded recurrent context is not necessary for LM**; bounded local context (≈ 40 tokens) is sufficient. This was prescient — Transformers later quantified the same observation through attention-window ablations.

#### Optimisation specifics

- **Initialisation:** Kaiming/He init on all conv layers.
- **Optimiser:** Nesterov momentum SGD with momentum 0.99, learning rate sampled uniformly from $[1, 2]$ at start of training, gradient clipping at 0.1.
- **Weight normalisation** (Salimans & Kingma 2016) and **gradient clipping** are both essential — Section 5.5 ablates each, showing they each roughly halve training time and combine super-linearly.
- **Adaptive softmax** for the output, since vocabularies are 200K–800K and full softmax is prohibitive.

The lesson, beyond GLU itself, is that gating-friendly architectures benefit greatly from the standard suite of optimisation tricks (clipping, weight norm), and the resulting recipe is mostly insensitive to dataset-specific tuning.

### Appendix — Section-by-section backbone (Paper 4)

| Section | Core content |
|---|---|
| Abstract | Gated convolutional networks for LM. Novel simplified gating mechanism (GLU). Outperforms van den Oord et al.'s GTU. SOTA on WikiText-103, competitive on Google Billion Word. 20× lower per-sentence latency than LSTM-2048. First non-recurrent approach competitive with strong recurrent LMs on large datasets. |
| 1. Introduction | Convnets stack to large receptive fields with $O(N/k)$ operations vs RNN's $O(N)$. Hierarchical (like linguistic grammar). Parallel-friendly. Gating shown essential in RNNs; GLU brings it to convnets with a gradient-friendly form. |
| 2. Approach | Convolutional LM architecture: lookup table → $L$ residual GLU blocks → adaptive softmax. GLU per Eq. P4.1. Causal padding for autoregressive generation. |
| 3. Gating Mechanisms | LSTM gating motivation. Convnets don't need forget gates. GLU motivated by Dauphin & Grangier 2015 "linearizing belief networks". Gradient comparison Eq. P4.2 vs P4.3: GLU has a non-downscaled linear path; GTU has two saturating-derivative factors. |
| 4. Experimental Setup — Datasets | Google Billion Word (1B tokens, 800K vocab) and WikiText-103 (100M tokens, 200K vocab); GBW is sentence-shuffled, WikiText-103 preserves paragraph context. |
| 4.2 Training | Torch + Tesla M40 GPUs. Nesterov momentum, gradient clipping, weight normalisation. Derives gradient clipping from spherical trust-region argument (Eq. P4.4 in paper text). |
| 4.3 Hyperparameters | Cross-validated random search. Architecture configs in Table 1: depths 8–14, widths 256–4096, kernel widths 1–6 with bottleneck variants. |
| 5. Results | Tables 2 (GBW) and 3 (WikiText-103): GCNN-13 38.1 / GCNN-14B 31.9 / GCNN-14 37.2 — competitive or SOTA. |
| 5.1 Computational Efficiency | Table 4: throughput parity with LSTM-2048, 20× higher responsiveness for GCNN-8B. Bottlenecks crucial for efficiency. |
| 5.2 Gating Mechanisms | Figure 3 ablation: GLU > ReLU > GTU > Tanh on both datasets. Gating units (sigmoid-modulated) clearly help over plain non-linearity (ReLU, Tanh). |
| 5.3 Non-linear Modeling | Figure 5: Linear (115 ppl) ≪ Bilinear (61 ppl) ≪ GLU (41 ppl). Bilinear alone beats Kneser-Ney 5-gram. |
| 5.4 Context Size | Figure 4: returns diminish past ≈ 20–40 tokens of context, even on WikiText-103. |
| 5.5 Training | Figure 6: weight norm and gradient clipping each roughly halve training time. |
| 6. Conclusion | Convolutional LM with GLU achieves SOTA on WikiText-103 and competitive results on GBW at lower compute cost. |

---

## Cross-paper synthesis — gating → weight generation → modern GLU

The four papers, read in chronological order (Highway workshop 2015 → Highway NeurIPS 2015 → HyperNetworks 2016 → GLU 2016) trace a coherent arc.

### Step 1 — Highway: gating as a gradient pathway

Highway's contribution is to **port LSTM-style multiplicative gating from time to depth**. The core equation (P1.3) defines a layer that is a learned, per-dimension, input-dependent convex combination of an identity skip and a nonlinear transform. The mechanism solves two problems simultaneously:

1. **Optimisation:** at init with $b_T \ll 0$, the network is approximately identity, gradients flow as $\partial y_L / \partial x_1 \approx I$, training proceeds.
2. **Conditional computation:** in the trained network, gates are *selective* (sparse per example, class-conditional), so each input routes through a different subset of layers. The Paper 2 lesion analysis confirms this: depth is *used* exactly to the extent the problem demands.

Two limitations of Highway that the later papers address:

- The gate is computed from the *same input* it gates — it cannot be conditioned on an external signal without modification.
- The dimensions of input and output must match — limits placement to interior of a fixed-width stack.

### Step 2 — HyperNetworks: from gating to weight generation

HyperNetworks generalise the "one signal controls another network's behaviour" idea from gating (multiplicative scale on activations) to **full weight generation** (the controlled network's *parameters* are emitted by the controlling network). The key generalisation steps:

- **Static hypernet (Eq. P3.2):** $K^j = g(z^j)$. A per-layer embedding $z^j$ replaces the per-layer weight, and a shared generator $g$ produces the actual weights. This is a *factorisation* of the parameter space — depth-$D$ network with $D \cdot \text{params/layer}$ becomes a $D \cdot N_z$ embedding + shared $g$.
- **Dynamic hypernet (Eq. P3.9):** the embedding $z$ is now a function of $(h_{t-1}, x_t)$, so the main network's weights depend on time and context. This is the most general formulation in the corpus.
- **Row-scaling efficient form (Eq. P3.7):** $W(z) = W \odot d(z)$, scaling rows by an emitted vector. This is **operationally identical** to FiLM, LayerNorm-with-learned-gain, and the Highway transform gate $T(x)$, with one key difference: the scales are emitted by a separate (possibly recurrent) network from a *context*, not a function of the layer input itself.

The conceptual ladder is therefore:
$$\underbrace{\text{Highway } T(x)}_{\text{gate from same input}} \;\to\; \underbrace{\text{FiLM } \gamma(c)}_{\text{gate from external context}} \;\to\; \underbrace{\text{HyperRNN } d(z(h_{t-1}, x_t))}_{\text{gate from running auxiliary network}} \;\to\; \underbrace{\text{Full hypernet } W(z)}_{\text{whole weight matrix generated}}.$$

Each step in this ladder is a strict generalisation; each later step contains the earlier as a special case. For a modulator that should change a policy's effective parameters, the architectural choice is *where to land on this ladder*: same-input gating (Highway), external scalar gain (FiLM), context-dependent row-scaling (HyperRNN-efficient), or full weight generation (HyperNetworks).

### Step 3 — GLU: gating made simple and gradient-friendly

GLU is in some sense a step *back* from HyperNetworks toward simplicity, but with a sharper gradient story. The contribution is twofold:

- **Simplified gate without forget machinery.** GLU drops the LSTM's input/forget gates and keeps only one "output-style" gate. For non-recurrent architectures (convnets, later Transformers) the forget-gate machinery is unnecessary because there is no indefinite-horizon hidden state to forget.
- **Linear-path gradient.** Eq. P4.3: the value branch is linear (not tanh), so $\partial y / \partial x$ contains a $\sigma(X) \cdot \nabla X$ term that doesn't pick up a saturating-derivative factor. The gate-branch term still has $\sigma'(X)$ but the linear path is the one carrying most of the signal at init. Across many layers, this preserves gradient where GTU loses it.

The **modern legacy** of GLU is that the formula $a \odot \sigma(b)$ (where $a$ and $b$ are two parallel projections of the same input) has been adopted, with various nonlinearities replacing $\sigma$, as the **default feed-forward sublayer of modern Transformers**. SwiGLU (Shazeer 2020, used in LLaMA, PaLM) replaces $\sigma$ with Swish; GeGLU (Shazeer 2020) replaces $\sigma$ with GeLU; ReGLU replaces $\sigma$ with ReLU. All are "X-GLU" variants of the original GLU formula. In this sense GLU is the *most directly inherited* of the four papers reviewed here — it lives inside today's frontier models in slightly modified form, whereas Highway lives on as ResNet (Highway with $C = T = 1$ fixed) and HyperNetworks live on as adapters / LoRA-style parameter-efficient fine-tuning (a hypernet that generates low-rank weight updates from a task embedding).

### Where the project lands on this ladder

For the project's neuromodulator-driven adaptation thread:

- If the modulator is a **scalar broadcast across all units** and should produce a per-unit gain — **FiLM** (reviewed elsewhere) is the right primitive.
- If the modulator should produce **per-row scales for specific weight matrices** of a recurrent policy — **HyperRNN row-scaling (Eq. P3.9)** is the right primitive.
- If the modulator should **route information through specific layers** (open or close certain processing pathways) — **Highway gating (Eq. P1.3)** with the modulator concatenated into the input that computes $T$ is the right primitive.
- If the modulator should be a **multiplicative element inside a feed-forward block** (e.g. inside a policy head) — **GLU (Eq. P4.1)** with the gate branch computed from the modulator rather than the input is the right primitive.

The four primitives are not exclusive — they can be composed (e.g. a GLU whose gate is computed by a small hypernetwork). The architectural choice depends on where the modulator's effect should land in the forward pass and how rich a transformation it should mediate.

---

*End of master review. Per project policy this doc lives at the topic root; source PDFs remain in `sources/`. Cross-references to FiLM, perceptual decision making, uncertainty corpora are available in their respective `_lit_review.md` files under `docs/project/references/`.*
