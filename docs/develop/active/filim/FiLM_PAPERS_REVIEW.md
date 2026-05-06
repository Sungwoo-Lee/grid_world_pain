---
title: FiLM (Feature-wise Linear Modulation) — Comprehensive Literature Review
topic: filim
status: active
created: 2026-03-18
last_updated: 2026-04-12
---

# FiLM (Feature-wise Linear Modulation) — Comprehensive Literature Review

> **Purpose**: Detailed paper-by-paper review for lab meeting presentation
> **Scope**: Original FiLM paper (Perez et al., 2018) + 9 subsequent works
> **Review Level**: Graduate / PhD — includes motivation, contribution, datasets, and detailed algorithm implementation
> **Last Updated**: 2026-03-18

---

## Table of Contents

1. [Perez et al., 2018 — FiLM: Visual Reasoning with a General Conditioning Layer](#1-perez-et-al-2018)
2. [Dumoulin et al., 2018 — Feature-wise Transformations](#2-dumoulin-et-al-2018)
3. [Birnbaum et al., 2019 — Temporal FiLM](#3-birnbaum-et-al-2019)
4. [Abdollahzadeh et al., 2021 — Revisit Multimodal Meta-Learning](#4-abdollahzadeh-et-al-2021)
5. [Takeda et al., 2021 — Multiple and Mixed Tasks with Feature Modulation](#5-takeda-et-al-2021)
6. [Turkoglu et al., 2022 — FiLM-Ensemble](#6-turkoglu-et-al-2022)
7. [Nikulin et al., 2023 — Anti-Exploration by Random Network Distillation](#7-nikulin-et-al-2023)
8. [Gorishniy et al., 2025 — TabM](#8-gorishniy-et-al-2025)
9. [Wisnu et al., 2025 — STSM-FiLM](#9-wisnu-et-al-2025)
10. [Yan & Guo, 2025 — Context-Aware Self-Adaptation (CaFiLM)](#10-yan--guo-2025)

---

## Summary Comparison Table

### A. Core FiLM Mechanism Comparison

| # | Paper | FiLM Variant | γ,β Source | Modulation Target | Normalization Coupling |
|---|-------|-------------|------------|-------------------|----------------------|
| 1 | Perez et al. | Original FiLM | GRU → affine projection | Feature maps (channel-wise) | After BN (but proven decoupled) |
| 2 | Dumoulin et al. | Taxonomy/Survey | Arbitrary functions f,h | Feature-wise (channel/element) | Can be decoupled |
| 3 | Birnbaum et al. | TFiLM | LSTM on pooled temporal blocks | 1D conv features (channel-wise) | With BN |
| 4 | Abdollahzadeh et al. | Kernel Modulation | Low-rank MLP from task embedding | Every kernel element (kernel-wise) | With BN |
| 5 | Takeda et al. | Task-conditional FiLM | FC from task one-hot vector | Feature maps after IN | After Instance Norm |
| 6 | Turkoglu et al. | FiLM-Ensemble | Learnable per-member params | BN affine params (channel-wise) | Replaces BN affine |
| 7 | Nikulin et al. | FiLM-conditioned RND | Linear from state encoding | MLP penultimate activations | None (MLP, no BN) |
| 8 | Gorishniy et al. | BatchEnsemble (not FiLM) | Learnable rank-1 adapters | Weight matrices (rank-1 perturbation) | None (avoids BN/IN) |
| 9 | Wisnu et al. | STSM-FiLM | MLP from scalar speed factor α | Decoder features (HiFi-GAN layers) | None specified |
| 10 | Yan & Guo | CaFiLM | 1D conv from (feature, batch mean) pairs | Per-dimension feature modulation | None (between extractor & classifier) |

### B. Task, Dataset, and Performance Comparison

| # | Paper | Year | Domain | Primary Benchmark | Key Result | Baseline Beaten |
|---|-------|------|--------|-------------------|------------|-----------------|
| 1 | Perez et al. | 2018 | Visual QA | CLEVR | 97.7% accuracy | PG+EE (96.9%), N2NMN (83.7%) |
| 2 | Dumoulin et al. | 2018 | Survey | N/A (taxonomy) | Unified framework | N/A |
| 3 | Birnbaum et al. | 2019 | Audio/Text/Genomics | VCTK, Yelp, ChIP-seq | +1.0 dB SNR; 95.6% Yelp-2 | CNN, LSTM, DNN baselines |
| 4 | Abdollahzadeh et al. | 2021 | Few-shot Meta-learning | 5-dataset multimodal | 56.72% (5-mode 1-shot) | MProtoNet (51.75%), ProtoNet (49.31%) |
| 5 | Takeda et al. | 2021 | Image Translation | Pascal VOC | Best FID/MSE/IoU, 1.69M params | Piggyback, SGN (9.5-10M params) |
| 6 | Turkoglu et al. | 2022 | Uncertainty | CIFAR-100, Retina | ECE 0.038, AUROC 79.85% | Deep Ensemble, BatchEnsemble, MC-Dropout |
| 7 | Nikulin et al. | 2023 | Offline RL | D4RL (Gym, AntMaze) | On par with EDAC/MSG/RORL | CQL, TD3+BC, IQL, naive RND |
| 8 | Gorishniy et al. | 2025 | Tabular Data | 46 tabular datasets | Mean rank 2.8 | CatBoost (3.2), XGBoost (3.3), FT-Trans (4.6) |
| 9 | Wisnu et al. | 2025 | Speech TSM | VCTK, TMHINT-QI | MOS 4.40, PESQ 2.034 | WSOLA (4.33), TSM-Net (1.89) |
| 10 | Yan & Guo | 2025 | Domain Generalization | DomainBed (5 datasets) | 68.8% avg accuracy | ERM (63.3%), SWAD (66.9%), EoA (68.0%) |

### C. Architectural and Training Comparison

| # | Paper | Backbone | # FiLM Layers | Ensemble Size | Optimizer | Key Hyperparams |
|---|-------|----------|---------------|---------------|-----------|-----------------|
| 1 | Perez et al. | CNN/ResNet-101 + GRU | 4 (per ResBlock) | — | Adam, LR 3e⁻⁴ | BS 64, 80 epochs |
| 2 | Dumoulin et al. | Various (survey) | Various | — | — | — |
| 3 | Birnbaum et al. | U-Net (1D conv) | Per conv layer | — | Adam, LR 3e⁻⁴ | 50 epochs, B=T/32 |
| 4 | Abdollahzadeh et al. | 4-layer CNN | 4 (per conv layer) | — | Adam, inner 0.05 / outer 1e⁻³ | 128-d task embedding |
| 5 | Takeda et al. | Encoder-Decoder (5 ResBlocks) | All conv layers (exc. last) | — | Adam, LR 1e⁻⁴ | BS 32, 300n epochs |
| 6 | Turkoglu et al. | VGG/ResNet/EfficientNet | All BN layers | M ∈ {2,4,8,16} | SGD, LR 0.1 cosine | BS 128, 200 epochs, ρ=2 |
| 7 | Nikulin et al. | 4-layer MLP (256 hidden) | 1 (penultimate layer) | — | Adam, LR 1e⁻³ | 3M steps, BS 1024 |
| 8 | Gorishniy et al. | MLP (variable depth) | All linear layers | k=32 | AdamW, grad clip 1.0 | Early stop patience 16 |
| 9 | Wisnu et al. | STFT/WavLM/Whisper + HiFi-GAN | Multiple decoder layers | — | Adam, LR 8e⁻⁴ | 200K steps, α∈[0.5,2.0] |
| 10 | Yan & Guo | ResNet-50 | 1 (between extractor & classifier) | |T| models ensembled | Adam, LR 5e⁻⁵/1e⁻³ | BS 32, λ trade-off |

---

## Paper Reviews

---

### 1. Perez et al., 2018

**Full Citation**: Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018.

#### 1.1 Motivation & Key Contribution

**Problem**: Standard deep learning methods struggle to perform multi-step, high-level visual reasoning (e.g., answering complex compositional questions about images). Instead of capturing the complex underlying structure of reasoning, these models tend to exploit biases in training data.

**Limitations of Prior Approaches**:
- Neural Module Networks (NMNs) and Program Generator + Execution Engine (PG+EE) models successfully tackle visual reasoning but rely on **strong priors**: explicitly hand-crafted architectures for relational computation, or step-by-step ground-truth program supervision to learn compositionality.
- A general conditioning mechanism built from general-purpose components — without restrictive architectural priors or extra supervision — would be far more widely applicable.

**Key Contributions**:
1. **State-of-the-Art Performance**: FiLM halves the SOTA error on CLEVR among models without extra supervision (error: 4.5% → 2.3%).
2. **Coherent Operation**: FiLM learns to selectively upregulate, downregulate, or shut off feature maps, and indirectly enables the CNN to localize question-referenced objects.
3. **Robustness**: Highly robust to architectural modifications and ablations. Crucially, conditional affine transformations do **not** need to be coupled with normalization layers.
4. **Strong Generalization**: Generalizes to human-generated questions (CLEVR-Humans), exhibits compositional generalization (CLEVR-CoGenT), and enables zero-shot generalization via linear arithmetic in FiLM parameter space.

#### 1.2 Datasets & Tasks

| Benchmark | Description | FiLM Accuracy | Best Baseline |
|-----------|-------------|---------------|---------------|
| **CLEVR** | Compositional visual QA on synthetic scenes | **97.7%** (pre-trained) / 97.6% (raw pixels) | PG+EE 96.9% (w/ 700K programs) |
| **CLEVR-Humans** | Free-form human questions on CLEVR scenes | 56.6% (zero-shot) → **75.9%** (fine-tuned) | PG+EE: 54.0% → 66.6% |
| **CLEVR-CoGenT** | Compositional generalization (swapped color palettes) | Cond A: **98.3%**, Cond B: 75.6% → **96.9%** (fine-tuned) | PG+EE: 73.7% (Cond B, no FT) |

- CNN+LSTM baseline: only 52.3% on CLEVR, 37.7%→43.2% on CLEVR-Humans
- N2NMN (End-to-End Module Networks): 83.7% on CLEVR

**Zero-Shot Composition**: Linear arithmetic in FiLM parameter space (e.g., γ,β for "cyan cubes" = "cyan spheres" + "brown cubes" − "brown spheres") improved Condition B accuracy from 71.5% → 80.7% on applicable questions without any fine-tuning.

#### 1.3 Detailed Algorithm & Implementation

**Core FiLM Formulation**:

The FiLM layer applies a feature-wise affine transformation conditioned on external input:

```
FiLM(F_{i,c} | γ_{i,c}, β_{i,c}) = γ_{i,c} · F_{i,c} + β_{i,c}
```

Where:
- `F_{i,c}` = the c-th feature map for the i-th input (spatial dimensions preserved)
- `γ_{i,c} = f_c(x_i)` — learned scaling function of conditioning input x_i
- `β_{i,c} = h_c(x_i)` — learned shifting function of conditioning input x_i
- f and h are implemented as affine projections from the question embedding

**Full Architecture**:

```
Question → Word Embeddings (200-d) → GRU (4096 hidden units) → Final hidden state
                                                                       ↓
                                                            Affine projections → (γ, β) per ResBlock
                                                                       ↓
Image (224×224) → Feature Extractor → 128 × 14×14 feature maps + 2 coordinate maps
                                                                       ↓
                                              4 FiLM-ed ResBlocks (each: 1×1 conv → 3×3 conv → FiLM → ReLU)
                                                                       ↓
                                              1×1 conv → 512 feature maps → Global Max Pool → MLP (1024) → Softmax
```

**Architecture Details**:
- **Feature Extractor**: Either a CNN trained from scratch (4 layers, 128 4×4 kernels) or pre-trained ResNet-101 (conv4 layer)
- **Coordinate Maps**: Two extra feature maps encoding relative x,y positions (−1 to 1) concatenated with image features to facilitate spatial reasoning
- **FiLM-ed ResBlocks**: 4 blocks, each containing 1×1 conv → 3×3 conv → BatchNorm (affine params OFF) → FiLM → ReLU, with 128 feature maps per block
- **Classifier**: 1×1 conv expanding to 512 maps → global max-pooling → 2-layer MLP (1024 hidden) → softmax over answers

**Training Configuration**:
- Optimizer: Adam, LR = 3e−4, Weight Decay = 1e−5
- Batch size: 64
- Training: end-to-end from scratch, image-question-answer triplets only, no data augmentation
- Early stopping on validation accuracy, max 80 epochs
- BatchNorm and ReLU activations throughout

**Key Ablation Findings**:
| Ablation | Accuracy | Insight |
|----------|----------|---------|
| Full model (pre-trained features) | 97.7% | — |
| Full model (raw pixels) | 97.6% | Pre-training barely helps |
| β = 0 (no shifting) | 96.9% | γ (scaling) is far more important |
| γ = 1 (no scaling) | 95.9% | Scaling > shifting for modulation |
| Replace γ with training mean (test time) | ~32% | **Catastrophic** — γ carries most information |
| Replace β with training mean (test time) | ~97% | β contributes relatively little |
| γ restricted to (0,1) via sigmoid | 95.9% | Negative γ and large magnitudes are crucial |
| Remove BatchNorm entirely | 93.7% | BN helps but is not required |
| FiLM after ReLU (not after BN) | 97.7% | FiLM placement is flexible |
| 1 ResBlock | 93.5% | Needs depth for iterative reasoning |
| 2–12 ResBlocks | 96.7–97.7% | Robust across depth |

**Visualization Insights**:
- **Parameter Distribution**: β peaks sharply at zero; 36% of γ and 76% of β values are negative. With ReLU after FiLM, this means features are selectively shut off.
- **t-SNE Clustering**: FiLM parameters cluster by question type. Early layers cluster low-level tasks (color, shape queries); deeper layers cluster high-level tasks (counting, comparison).
- **Semantic Sub-clusters**: Individual feature maps show sub-clusters based on specific concepts (e.g., "front" vs. "behind", or "matte/rubber" vs. "shiny/metallic"), demonstrating learned hierarchical, function-based modularity without hand-crafted priors.

---

### 2. Dumoulin et al., 2018

**Full Citation**: Dumoulin, V., Perez, E., Schucher, N., Strub, F., de Vries, H., Courville, A., & Bengio, Y. (2018). *Feature-wise Transformations*. Distill.

#### 2.1 Motivation & Key Contribution

**Problem**: Various conditioning mechanisms were being developed independently across different subfields (style transfer, VQA, RL, generative modeling), but lacked a **unified abstraction**. The paper addresses how to effectively integrate and fuse multiple sources of information — e.g., conditioning an image generator on a class label, or a visual pipeline on a linguistic question.

**Key Contribution**: The paper demonstrates that a simple family of approaches — **feature-wise transformations** — underlies the success of a surprisingly diverse set of models across seemingly unrelated problem domains. It provides a unifying taxonomy and framework.

#### 2.2 Datasets & Tasks

This is a **survey/overview paper** (published in Distill), not an empirical contribution. It catalogs feature-wise transformations across these domains:

| Domain | Example Tasks | Example Methods |
|--------|--------------|-----------------|
| **Visual QA** | CLEVR, GuessWhat?! | FiLM, CBN |
| **Style Transfer** | Arbitrary artistic style | AdaIN, Conditional IN |
| **Image Recognition** | ImageNet classification | Highway Networks, Squeeze-and-Excitation |
| **NLP** | Language modeling, machine translation | LSTMs, Gated Linear Units, Gated-Attention Readers |
| **Reinforcement Learning** | 3D instruction-following (VizDoom), multi-game Atari | Multimodal fusion policies |
| **Generative Modeling** | Class-conditional generation, raw audio/image | Conditional DCGAN, WaveNet, PixelCNN |
| **Speech Recognition** | Acoustic model adaptation | Utterance-based conditioning |
| **Domain Adaptation / Few-Shot** | Cross-domain transfer | Per-channel BN statistics adaptation |

#### 2.3 Detailed Algorithm & Implementation

**Unifying Formulation — Feature-wise Linear Modulation (FiLM)**:

```
γ = f(z)           # scaling parameters from conditioning input z
β = h(z)           # shifting parameters from conditioning input z
FiLM(x) = γ(z) ⊙ x + β(z)    # element-wise affine transform
```

Where f and h are arbitrary learned functions (the "FiLM generator"), z is the conditioning input, and x is the feature representation being modulated.

**Taxonomy of Feature-wise Transformation Types**:

| Type | Formula | Description |
|------|---------|-------------|
| **Concatenation** | `[x; z]` → linear layer | Equivalent to conditional biasing (proven mathematically) |
| **Conditional Biasing** | `x + β(z)` | Additive-only; γ fixed to 1 |
| **Conditional Scaling** | `γ(z) ⊙ x` | Multiplicative-only; β fixed to 0 |
| **Sigmoidal Gating** | `σ(g(z)) ⊙ x` | Scaling restricted to [0,1]; feature selection/gating |
| **Conditional Affine** | `γ(z) ⊙ x + β(z)` | Full FiLM; both scale and shift |

**How Existing Methods Map to FiLM**:

| Method | FiLM Interpretation |
|--------|-------------------|
| **Batch Normalization** | Self-conditioning: γ,β are learned constants (not input-dependent) |
| **Conditional Batch Norm** | FiLM where γ,β predicted from conditioning input, applied after BN |
| **Instance Normalization** | Like BN but per-instance statistics |
| **Adaptive Instance Norm (AdaIN)** | FiLM where γ,β are extracted as spatial std and mean of style image |
| **Squeeze-and-Excitation** | Self-conditioning: network predicts its own γ from its own activations |
| **LSTM gates** | Self-conditioning with sigmoidal gating |
| **PixelCNN / WaveNet conditioning** | FiLM with γ=1 (conditional biasing only) |
| **Highway Networks** | Self-conditioning with learned gating |

**Key Design Insights**:
1. **Parameter Efficiency**: Number of FiLM parameters scales **linearly** with channel count — avoids quadratic cost of spatial attention or full weight conditioning.
2. **Domain-Agnostic**: Minimal inductive bias → works across vision, language, RL, audio without architectural changes.
3. **Selective Routing**: FiLM learns to upregulate, downregulate, and completely shut off specific feature maps — emergent information routing.
4. **Structured Parameter Space**: Learned γ,β cluster semantically (t-SNE shows dense clusters by concept), enabling **zero-shot analogical arithmetic** via linear combinations of FiLM parameters.
5. **Decoupling from Normalization**: The affine transform does not need to be coupled with any normalization layer — can be applied anywhere in the network.

---

### 3. Birnbaum et al., 2019

**Full Citation**: Birnbaum, S., Kuleshov, V., Enam, S. Z., Koh, P. W., & Ermon, S. (2019). *Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulation*.

#### 3.1 Motivation & Key Contribution

**Problem**: Deep learning models for high-dimensional, long sequential data (audio, text, genomics) face significant challenges in capturing **long-range input dependencies** — interactions between symbols far apart in a sequence.

**Limitations of Existing Approaches**:
- **RNNs**: Naturally model sequences but are slow and suffer from vanishing gradients over long time series.
- **1D Dilated Convolutions**: Faster and easier to train, but fundamentally limited to a **finite receptive field** — each output depends only on a constrained input window.
- **Standard FiLM**: Prior FiLM applications to sequential data (e.g., speech recognition) used feed-forward models to modulate layer normalization, but lacked a mechanism to process long temporal histories efficiently.

**Key Contribution**: TFiLM modulates intermediate 1D-CNN activations using long-range context captured by an RNN that operates on **temporally pooled blocks**, dramatically reducing the RNN's sequence length. This combines CNN efficiency with RNN memory at minimal computational overhead.

#### 3.2 Datasets & Tasks

| Domain | Dataset | Task | Metric | TFiLM Result | Best Baseline |
|--------|---------|------|--------|--------------|---------------|
| **Text** | Yelp-2 (600k reviews) | Binary sentiment | Accuracy | **95.6%** | CNN 93.5%, LSTM 92.6% |
| **Text** | Yelp-5 (700k reviews) | 5-class sentiment | Accuracy | **62.3%** | SmallCNN 61.5% |
| **Audio** | VCTK (44h, 108 speakers) | Super-resolution (r=2,4,8) | SNR/LSD | +1.0 dB SNR avg | Conv baseline |
| **Audio** | Piano (10h, Beethoven) | Super-resolution | SNR/LSD | Consistent improvement | Conv baseline |
| **Genomics** | ChIP-seq (H3K4me1, etc.) | Signal super-resolution | Pearson r | **0.81** (H3K4me1) | CNN 0.59, Input 0.37 |

- On audio multispeaker r=4: TFiLM SNR 15.0 dB vs. cubic B-spline 13.2 dB, DNN 13.1 dB, Conv 13.3 dB
- On text: achieves competitive with BERT/VDCNN using **< 1.5M parameters**

#### 3.3 Detailed Algorithm & Implementation

**TFiLM Block — Core Mechanism**:

Given 1D convolutional activations `F ∈ ℝ^{T×C}` (T = temporal, C = channels):

```
Step 1 — Blocking:    F_blk ∈ ℝ^{B × T/B × C}     (reshape into B non-overlapping temporal blocks)
                      F_blk[b,t,c] = F[b×B + t, c]

Step 2 — Pooling:     F_pool ∈ ℝ^{B × C}            (max-pool within each block)
                      F_pool[b,c] = MaxPool(F_blk[b,:,c])

Step 3 — RNN:         (γ_b, β_b), h_b = LSTM(F_pool[b,:]; h_{b-1})    for b = 1,...,B
                      h_0 = 0

Step 4 — Modulate:    F_norm[b,t,c] = γ_{b,c} · F_blk[b,t,c] + β_{b,c}

Step 5 — Reshape:     F'[t,c] = F_norm[⌊t/B⌋, t mod B, c]    (back to ℝ^{T×C})
```

**Key insight**: The RNN processes only B pooled block representations (not T timesteps), making it computationally efficient while providing global temporal context.

**Overall Architecture (U-Net style for generative tasks)**:
- K=4 downsampling blocks (halve spatial dim, double features) → bottleneck → K=4 upsampling blocks (double spatial, halve features)
- Subpixel shuffling for upsampling (avoids checkerboard artifacts)
- Symmetric residual skip connections between corresponding down/up blocks
- Dilated convolutions with dilation factor 2
- TFiLM layers inserted after convolutions at each level

**Training Details**:
| Config | Audio | Text | Genomics |
|--------|-------|------|----------|
| Optimizer | Adam | Adam | Adam |
| Learning Rate | 3×10⁻⁴ | 1×10⁻³ | 3×10⁻⁴ |
| Batch Size | — | 128 | — |
| Epochs | 50 | 20 | 50 |
| Patch Length | 8192 | 256 tokens | 1000 |
| Block size B | T/32 (so always 32 blocks) | — | 2 |
| Pooling stride | 8 | — | — |
| Word Embeddings | — | 100-d GloVe | — |
| Conv layers | 4 blocks (U-Net) | 3 layers (SmallCNN) | Same as audio |

---

### 4. Abdollahzadeh et al., 2021

**Full Citation**: Abdollahzadeh, M., Malekzadeh, T., & Cheung, N.-M. (2021). *Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning*.

#### 4.1 Motivation & Key Contribution

**Problem**: Multimodal meta-learning extends few-shot learning to tasks drawn from **multiple diverse distributions** (e.g., mixing character recognition from Omniglot with natural object classification from mini-ImageNet). "Multimodal" here means multiple task distributions, not multiple sensor modalities.

**Key Issue — Negative Transfer**: Using "transference" analysis from multi-task learning (MTL), the authors show that existing meta-learners struggle because diverse tasks **fight for model capacity**, causing destructive interference (negative knowledge transfer), especially during early training iterations.

**Limitation of Standard FiLM**: Standard FiLM applies a single scalar γ and β per feature map — mathematically equivalent to scaling an entire convolutional kernel by one scalar. This severely restricts degrees of freedom, forcing diverse tasks to compete for the same capacity.

**Key Contribution — Kernel Modulation (KML)**:
- Generalizes FiLM from **feature-wise** to **kernel-wise**: generates a modulation parameter for **every element** within each convolutional kernel, not just one per feature map.
- Provides significantly more task-specific capacity while preserving parameter efficiency via low-rank factorization.
- From an MTL perspective: pseudo-task-specific layers on top of shared base kernels, reducing negative transference.

#### 4.2 Datasets & Tasks

**Multimodal meta-dataset**: 5 combined benchmarks — Omniglot, mini-ImageNet, FC100, CUB, Aircraft.

**Configurations**: 5-way 1-shot and 5-way 5-shot, evaluated in 2-mode, 3-mode, and 5-mode setups.

| Method | 5-mode 1-shot | 5-mode 5-shot |
|--------|---------------|---------------|
| ProtoNet (baseline) | 49.31% | 58.91% |
| Multi-ProtoNet | 50.69% | 59.88% |
| MProtoNet (MMAML variant) | 51.75% | 59.95% |
| **MProtoNet + KML** | **56.72%** | **64.91%** |

Substantial improvements also in 2-mode and 3-mode configurations.

#### 4.3 Detailed Algorithm & Implementation

**Kernel Modulation Formulation**:

Shared parameters: base kernels `W_l` and biases `b_l` for each layer l.
Task-specific parameters generated from task embedding `v_T`:

```
Ŵ_T^l = W_l ⊙ (J + M_l(v_T, ϕ))        # element-wise modulation of entire kernel
b̂_T^l = b_l + Δb_l(v_T, ϕ)               # bias offset

where J = all-ones matrix, ⊙ = Hadamard product
```

**Low-rank factorization** (prevents parameter explosion):
```
M_l(v_T, ϕ) = g^l_{ϕ1}(v_T) ⊗ g^l_{ϕ2}(v_T)    # outer product of two MLP outputs
```

**Comparison: FiLM vs KML**:
| Aspect | Standard FiLM | Kernel Modulation (KML) |
|--------|--------------|------------------------|
| Modulation granularity | 1 scalar per feature map | 1 scalar per kernel element |
| Parameters per layer | 2C (γ,β per channel) | Low-rank: 2 × MLP outputs |
| Capacity | Limited — all spatial kernel weights scaled identically | High — each kernel weight modulated independently |
| Interpretation | Feature-wise affine | Kernel-wise multiplicative + additive |

**Meta-Learning Objective**:
- **Inner loop**: Infer task embedding `v_T` from support set → generate modulation matrices → adapt shared params to task-specific params `θ̂_T`
- **Outer loop**: Evaluate on query set → update shared base params θ and generator params φ,ϕ:
  `θ ← θ − α ∇_θ Σ_T L_T(Q; θ̂_T, S)`

**Architecture**:
- **Backbone**: 4-layer CNN (3×3 conv, stride 2, BN, ReLU); channels: 32, 64, 128, 256
- **Task Encoder**: Same 4-layer CNN → avg pool → FC → 128-d task embedding
- **Optimizer**: Adam; inner LR = 0.05 (MAML variants), outer LR = 0.001
- **Modulation applied**: to base kernels of all 4 CNN layers during inner-loop adaptation

---

### 5. Takeda et al., 2021

**Full Citation**: Takeda, M., Benitez, G., & Yanai, K. (2021). *Training of Multiple and Mixed Tasks with a Single Network Using Feature Modulation*.

#### 5.1 Motivation & Key Contribution

**Problem**: Traditional multi-task learning (MTL) for image translation uses a shared encoder with **multiple task-specific decoders**. Since decoders can constitute up to half the network, parameter count scales linearly with the number of tasks. Training heterogeneous image translation tasks (denoising, segmentation, style transfer) in a single fully-shared network is challenging because activation distributions differ substantially across tasks.

**Mixed-Task Concept**: Beyond individual tasks, the paper targets **novel combinations** of tasks at inference time:
- **Sequential mixing**: e.g., denoising → style transfer
- **Mixing by masking**: e.g., segment an object, apply style transfer only to that region

Prior MTL approaches cannot handle such combinations and suffer from task interference.

**Key Contributions**:
1. **FiLM for full-network sharing**: Both encoder AND decoder are shared; FiLM layers dynamically adjust activations per task, eliminating task-specific decoders.
2. **Negligible parameter overhead**: Only the input dimension of the FiLM generator's first FC layer depends on task count — avoids parameter explosion (~1.69M vs 10M for single-task ensembles).
3. **Mixed-task learning**: Achieved via synthesized mixed-task training samples + L2 loss (simple summing of task losses fails for heterogeneous tasks).
4. **Continuous interpolation**: Degree of transformation between two tasks smoothly controllable by linearly interpolating the conditional vector.

#### 5.2 Datasets & Tasks

**Dataset**: Pascal VOC 2011 — 8,498 training, 2,857 test images, 20 classes.

**Individual Tasks**:
| Task ID | Task | Metric |
|---------|------|--------|
| 0 | Reconstruction (identity) | MSE / SSIM |
| 1 | Inpainting | MSE / SSIM |
| 2 | Denoising | MSE / SSIM |
| 3 | Semantic segmentation | IoU |
| 4 | Style transfer 1 (Van Gogh) | FID |
| 5 | Style transfer 2 (Munch) | FID |

**Mixed Tasks**: Mix1: Denoising + Style1; Mix2: Denoising + Segmentation; Mix3: Segmentation + Style1.

**Results**: With ~1.69M parameters, the FiLM-based network achieved **best evaluation scores in almost all tasks** compared to Piggyback, SGN, and standard shared-encoder MTL baselines (which use 9.5–10M parameters).

#### 5.3 Detailed Algorithm & Implementation

**FiLM Formulation with Instance Normalization**:

A task conditional vector `c = [c_1, ..., c_n]` (one-hot or multi-hot for mixed tasks) is passed through FC layer(s):

```
γ_i = f_{γ,i}(c)       # generated by FiLM generator (FC layers)
β_i = f_{β,i}(c)

x_i^norm = (x_i - μ_i) / σ_i          # Instance Normalization
z_i = γ_i · x_i^norm + β_i            # FiLM modulation
```

**Mixed-task handling**: Set multiple elements of c to 1.0 (e.g., c = [0,0,1,0,1,0] for denoising + style1). Training uses synthesized mixed-task samples (sequential application or mask-based composition) with L2 loss.

**Architecture**:
- **Backbone**: Encoder-Decoder CNN (Johnson et al. style transfer architecture): 3 conv layers → 5 residual blocks → 3 deconv layers
- **FiLM Insertion**: IN + FiLM after **all convolutional layers** except the last
- **FiLM Generator**: Single FC layer (keeps model compact)

**Training**:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e⁻⁴ |
| Batch Size | 32 |
| Epochs | 300n (n = number of tasks) |
| Task Sampling | One task randomly selected per mini-batch |
| Loss Weighting | Equal weight (no weighting) |
| Total Parameters | ~1.69M |

---

### 6. Turkoglu et al., 2022

**Full Citation**: Turkoglu, M. O., et al. (2022). *FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation*. NeurIPS 2022.

#### 6.1 Motivation & Key Contribution

**Problem**: Estimating **epistemic uncertainty** (uncertainty from insufficient training data) is critical for trustworthy ML, but explicit deep ensembles — the gold standard — require training, storing, and running inference on M independent networks. Cost scales linearly with M, making them impractical for constrained settings (mobile robotics, real-time).

**Limitations of Existing Implicit Ensembles**:
- **MC-Dropout**: Underperforms in accuracy and calibration vs. explicit ensembles.
- **BatchEnsemble**: Suffers from negative correlation between ensemble size and test accuracy — members are insufficiently diverse.
- **MIMO**: Struggles on complex tasks (48.0% on CIFAR-100 vs. 81.6% for deep ensembles).

**Key Contributions**:
1. **FiLM-Ensemble**: Shares all network weights across members; differentiates each member solely through unique FiLM parameters (γ,β) — creating distinct predictive functions in a single forward pass.
2. **Higher diversity than deep ensembles**: Achieves up to 9.2% disagreement on CIFAR-10 (vs. 6.8% for naive ensembles) via scaled Xavier initialization controlled by gain hyperparameter ρ.
3. **Best ECE among all methods**: Better calibrated than even explicit deep ensembles on several benchmarks.
4. **Parameter efficient**: Only ~0.1% additional parameters per ensemble member (just the FiLM γ,β per layer).

#### 6.2 Datasets & Tasks

| Benchmark | Task | Metric | FiLM-Ens (M=4) | Deep Ensemble | BatchEns | MC-Dropout |
|-----------|------|--------|-----------------|---------------|----------|------------|
| **CIFAR-100** (ResNet-18) | Classification | Accuracy | 79.4% | **81.6%** | 77.7% | 75.5% |
| **CIFAR-100** (ResNet-18) | Classification | ECE ↓ | **0.038** | 0.041 | 0.052 | 0.064 |
| **Retina→REFUGE** (M=16) | OOD Detection | AUROC | **79.85%** | 78.06% | 75.04% | — |
| **6mA Identification** | Genomic sequence | Classification | Competitive | — | — | — |

Additional benchmarks: CIFAR-10, Retina Glaucoma Detection, REFUGE 2020 challenge.

#### 6.3 Detailed Algorithm & Implementation

**Core Formulation**:

Each ensemble member m has its own learnable FiLM parameters (not generated by a secondary network):

```
FiLM(F_n | γ_n^m, β_n^m) = γ_n^m ∘ F_n + β_n^m     # member m, layer n
```

Where ∘ is the Hadamard (element-wise) product.

**Diversity Initialization** (key innovation):
```
γ, β ~ Uniform(±√(3/D_n) · ρ)     # bounded Xavier initialization
```
- `D_n` = number of features/channels at layer n
- `ρ` = tunable gain factor controlling initial diversity
- `ρ → 0`: ensemble collapses to single model
- `ρ = 2`: default for vision tasks; `ρ ∈ {4,8,16,32}` for genomic data

**Aggregation**:
```
ŷ = (1/M) Σ_{m=1}^{M} y_m     # average predictions of all members
```

**Training**: All shared params θ + member-specific (γ^m, β^m) optimized jointly with standard Cross-Entropy loss. Forward passes parallelized over batch dimension — all members processed simultaneously on single GPU.

**Integration Method**: Replace standard BatchNorm layers with **Conditional BatchNorm** layers that apply member-specific FiLM parameters.

**Architecture & Training Details**:
| Parameter | Value |
|-----------|-------|
| Base Networks | VGG-11, ResNet-18/34, EfficientNet-B0, 1D-CNN |
| Ensemble Sizes | M ∈ {2, 4, 8, 16} |
| FiLM Insertion | Replaces BN affine params in every BN layer |
| Optimizer | SGD, momentum 0.9, weight decay 5e⁻⁴ |
| LR Schedule | 0.1 → cosine annealing |
| Batch Size | 128 |
| Epochs | 200 |
| Diversity Gain ρ | 2 (vision), 4–32 (genomics) |

**Comparison with BatchEnsemble**: Both use per-member affine modulation, but BatchEnsemble uses rank-1 factors (outer product of two vectors) while FiLM-Ensemble uses full channel-wise γ,β with diversity-encouraging initialization. FiLM-Ensemble's ρ-controlled initialization is the key differentiator enabling higher diversity.

---

### 7. Nikulin et al., 2023

**Full Citation**: Nikulin, A., Kurenkov, V., Tarasov, D., & Kolesnikov, S. (2023). *Anti-Exploration by Random Network Distillation*.

#### 7.1 Motivation & Key Contribution

**Problem — Anti-Exploration in Offline RL**: Offline RL agents learn from a fixed dataset without environment interaction. They must avoid **out-of-distribution (OOD) actions** to prevent distributional shift and catastrophic Q-value overestimation for unseen actions.

**Random Network Distillation (RND) — Online vs. Offline**:
- **Online RL**: RND provides a novelty **bonus** (high prediction error = novel state → explore it).
- **Offline RL (this paper)**: RND prediction error becomes a novelty **penalty** — subtracted from the TD target to induce conservatism where OOD overestimation occurs.

**Prior work deemed RND insufficient** for continuous offline RL — the prediction error was thought to be insufficiently discriminative in continuous action spaces.

**Limitations of Existing Offline RL Methods**:
- **Ensemble-free** (CQL, TD3+BC, IQL): Regularize Q or constrain policy, but significantly underperform ensemble methods.
- **Ensemble-based** (SAC-N, EDAC, MSG): Use Q-network disagreement for uncertainty, but require up to 500 ensemble members — unscalable.

**Key Insight**: RND **is** discriminative enough, but naive state-action concatenation produces noisy anti-gradient fields. The actor cannot follow these gradients to minimize the penalty.

**Key Contribution — FiLM-conditioned RND**: Using FiLM to condition the RND prior on state creates **smooth anti-gradient fields** that consistently point toward the correct minimum across the entire action space. This makes the actor successfully conservative, achieving ensemble-level performance with **zero ensemble overhead**.

#### 7.2 Datasets & Tasks

**Benchmark**: D4RL (standard offline RL benchmark)

| Domain | Environments | Episodes for Eval |
|--------|-------------|-------------------|
| **Gym** | HalfCheetah, Walker2d, Hopper | 10 |
| **AntMaze** | umaze, medium, large (significantly harder) | 100 |

**Results**:
- **vs. Ensemble-free**: SAC-RND outperforms CQL, IQL, TD3+BC by a wide margin in both domains.
- **vs. Ensemble-based**: On par with EDAC, MSG, RORL (which use many Q-networks).
- **vs. Naive RND (no FiLM)**: FiLM conditioning completely solves the optimization bottleneck that made naive RND fail.

#### 7.3 Detailed Algorithm & Implementation

**RND Uncertainty Penalty**:
```
b(s,a) = ‖f_ψ(s,a) - f̄_ψ̄(s,a)‖²₂ / σ_running
```
Where `f_ψ` = predictor (trained), `f̄_ψ̄` = prior (fixed, random init). Normalized by running std of RND loss from pretraining.

**FiLM Conditioning on Prior Network**:
```
f(h, s) = γ(s) · h + β(s)
```
State s is encoded via linear layer → produces γ,β → applied to penultimate layer activations before nonlinearity. For predictor network: bilinear conditioning on first layer (empirically best).

**Integration into SAC**:

TD target (penalized):
```
y(r, s') = r + γ [min_{j=1,2} Q_ϕ̄_j(s', a') - β log π_θ(a'|s') - α · b(s', a')]
```

Policy objective (penalized):
```
max_θ  min_{j=1,2} Q_ϕ_j(s, ã_θ(s)) - β log π(ã_θ(s)|s) - α · b(s, ã_θ(s))
```

Where α = conservatism coefficient (swept per environment: 0.1–25.0), β = SAC temperature.

**Architecture & Training**:
| Component | Details |
|-----------|---------|
| RND Networks | 4-layer MLP, hidden dim 256, output embedding dim 32 |
| FiLM Location (Prior) | Penultimate layer, before nonlinearity |
| Conditioning (Predictor) | Bilinear on first layer |
| RND Pretraining | MSE between predictor/prior on offline dataset; prior gradients disabled |
| Post-pretraining | Both RND networks frozen; train actor + 2 critics |
| Optimizer | Adam; LR 1e⁻³ (Gym), 3e⁻⁴ (AntMaze) |
| Batch Size | 1024 (Gym), 256 (AntMaze) |
| Training Steps | 3M gradient steps |
| Conservatism α | Environment-specific sweep (0.1–25.0) |

**Why FiLM Matters Here**: With naive concatenation `[s,a]`, the anti-gradient ∇_a b(s,a) is noisy — only points to minimum in small neighborhood. FiLM conditioning makes the gradient field smooth and globally consistent, enabling the actor to effectively minimize the penalty.

---

### 8. Gorishniy et al., 2025

**Full Citation**: Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2025). *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*.

#### 8.1 Motivation & Key Contribution

**Problem**: Tabular deep learning remains challenging — tree-based methods (XGBoost, CatBoost) are still the go-to for practitioners due to reliability, efficiency, and resistance to overfitting. Advanced DL architectures (attention-based, retrieval-based) suffer from high training times and poor inference throughput. Standard deep ensembles improve DL performance substantially but scale linearly in cost.

**Prior attempts** to introduce ensemble-like elements into tabular DL were found unpromising. FiLM-Ensemble requires normalization layers, which aren't always beneficial in simple tabular MLPs.

**Key Contributions**:
1. **TabM**: Packs k implicit MLPs into a single model via **parameter-efficient ensembling** (BatchEnsemble-based, not FiLM-based — avoids normalization dependency).
2. **SOTA on 46 tabular datasets**: Mean rank 2.8 — beats CatBoost (3.2), XGBoost (3.3), FT-Transformer (4.6).
3. **Four variants** with different efficiency/expressivity tradeoffs:
   - **TabM_packed**: k fully independent MLPs, processed in parallel (no weight sharing)
   - **TabM_naive**: All BatchEnsemble adapters across all layers
   - **TabM_mini**: Only first adapter in first layer (maximally efficient)
   - **TabM (default)**: All adapters, but multiplicative ones initialized to 1 (TabM_mini at init, grows more expressive during training)

#### 8.2 Datasets & Tasks

**Benchmark**: 46 public tabular datasets — 18 regression + 28 classification, 37 random splits + 9 domain-aware splits.

| Method | Mean Rank (↓) | Relative Improvement over MLP |
|--------|---------------|-------------------------------|
| **TabM** | **2.8 ± 2.1** | +2.15% ± 2.8% |
| TabM_mini | 2.9 ± 2.2 | — |
| CatBoost | 3.2 ± 2.0 | — |
| XGBoost | 3.3 ± 2.1 | — |
| MLP | 3.8 ± 2.4 | baseline |
| FT-Transformer | 4.6 ± 2.9 | +0.39% ± 1.6% |
| SAINT | 4.9 ± 2.9 | — |

#### 8.3 Detailed Algorithm & Implementation

**BatchEnsemble Formulation** (rank-1 perturbations):

For a linear layer `l(x) = Wx + b`, the i-th ensemble member:
```
l_i(x_i) = s_i ⊙ (W(r_i ⊙ x_i)) + b_i
```

Where:
- `W ∈ ℝ^{d×d}` = shared weight matrix
- `r_i, s_i, b_i ∈ ℝ^d` = member-specific adapter vectors (non-shared)
- Equivalent to per-member weight matrix: `W_i = W ⊙ (s_i · r_i^T)` (rank-1 perturbation)

**Parallel execution** of all k members:
```
l_BE(X) = ((X ⊙ R) W) ⊙ S + B
```
Where `X ∈ ℝ^{k×d}`, `R,S,B ∈ ℝ^{k×d}` = stacked adapters.

**Loss**: Mean of k individual submodel losses. Inference: average k submodel outputs.

**Relationship to FiLM**: Authors explicitly chose BatchEnsemble over FiLM because FiLM requires normalization layers (BN/IN) which are not always present or beneficial in tabular MLPs. BatchEnsemble's multiplicative adapters work directly on weight matrices without normalization dependency.

**TabM Default Initialization Trick**: All multiplicative adapters (except first layer's r) initialized to **1** — model starts as TabM_mini but gains expressivity during training.

**Architecture & Training**:
| Parameter | Value |
|-----------|-------|
| Base Network | MLP: Linear → ReLU → Dropout, repeated N blocks |
| Ensemble Size k | 32 (fixed, not tuned) |
| Optimizer | AdamW, gradient clipping 1.0 |
| LR Schedule | None |
| Early Stopping | Patience 16 epochs, based on collective ensemble validation score |
| Feature Embeddings | Optional piecewise-linear embeddings for continuous features |
| Hyperparameter Tuning | TPE sampler (Optuna) |
| Batch Strategy | Shared batches — all k members see same objects (minimal perf loss, massive speedup) |

---

### 9. Wisnu et al., 2025

**Full Citation**: Wisnu, D. A. M. G., et al. (2025). *STSM-FiLM: A FiLM-conditioned neural architecture for time-Scale Modification of speech*.

#### 9.1 Motivation & Key Contribution

**Problem — Time-Scale Modification (TSM)**: Altering speech playback speed/duration without changing perceived pitch. Critical for speech synthesis, audio editing, hearing accessibility, and language learning.

**Limitations of Traditional TSM** (WSOLA, PSOLA, phase vocoder, Griffin-Lim):
- Rely on handcrafted heuristics (phase alignment, peak detection)
- Degrade under non-stationary speech dynamics
- Suffer from transient smearing, pitch/formant sensitivity at extreme speed factors

**Prior Neural TSM** (TSM-Net, DiffATSM): Trained on limited discrete stretch ratios, lack explicit continuous conditioning for playback rate.

**Key Contribution — STSM-FiLM**:
- First work to integrate FiLM into a neural TSM framework for **continuous speed control**
- FiLM maps a continuous speed factor α to affine parameters, enabling dynamic, smooth control during inference
- Encoder-conditioning-decoder architecture: encode audio → FiLM-condition on speed → decode
- WavLM-HiFiGAN variant **outperforms WSOLA** in subjective naturalness (MOS 4.40 vs. 4.33)

#### 9.2 Datasets & Tasks

**Training Data**: VCTK (English), TMHINT-QI (Mandarin), all at 16kHz.
**OOD Evaluation**: LibriSpeech, COSPRO.
**Speed Factors**: Training α ∈ [0.5, 2.0] step 0.1; Evaluation: 0.5, 0.75, 1.25, 1.5, 1.75, 2.0.

| Metric | STSM-FiLM (WavLM-HiFiGAN) | STSM-FiLM (STFT-HiFiGAN) | WSOLA | TSM-Net |
|--------|---------------------------|--------------------------|-------|---------|
| **MOS** (subjective) | **4.40** | — | 4.33 | 1.89 |
| **PESQ** (quality) | — | **2.034** | — | — |
| **STOI** (intelligibility) | — | **0.894** | — | — |

Additional metrics: DNSMOS (naturalness), WER/CER from pretrained ASR.

#### 9.3 Detailed Algorithm & Implementation

**FiLM Conditioning on Speed Factor**:

```
(γ_α, β_α) = MLP(α)                           # small MLP maps scalar speed factor to affine params
f̂_t = (1 + γ_α) · f_t + β_α                   # modulate intermediate features
```

**Note**: The `(1 + γ_α)` formulation (offset around 1) preserves identity mapping at initialization → improves training stability.

**Variable-Length Output**: FiLM modulates feature values but not temporal length. Linear interpolation along the temporal axis resizes the feature sequence to match target duration for desired speed.

**Loss Function** (WSOLA as teacher):
```
L = λ_L1 · ‖x̂_α − x^WSOLA_α‖₁ + λ_adv · L_GAN + λ_fm · L_FM
```
Where `x^WSOLA_α` = WSOLA-generated pseudo-ground truth targets.

**Architecture — Encoder-Conditioning-Decoder**:

| Encoder Variant | Feature Dim | Source |
|----------------|-------------|--------|
| STFT | 1024 (log-magnitude spectrogram) | Fixed |
| WavLM | 1024 (6th layer of WavLM-Large) | Pre-trained |
| Whisper | 1024 (last encoder layer, Whisper Medium) | Pre-trained |
| EnCodec | Variable | Pre-trained |

**Decoder**: HiFi-GAN for STFT/WavLM/Whisper variants; original EnCodec decoder for EnCodec variant.

**FiLM Insertion**:
- HiFi-GAN variants: FiLM at multiple layers within the decoder
- EnCodec variant: FiLM between encoder and quantizer (modulates latent codes before quantization)

**Training**:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 8×10⁻⁴ |
| Steps | 200,000 |
| Precision | 32-bit |
| Teacher Signal | WSOLA-generated targets |
| Speed Sampling | α ∈ [0.5, 2.0], step 0.1 (random per sample) |

---

### 10. Yan & Guo, 2025

**Full Citation**: Yan, H., & Guo, Y. (2025). *Context-Aware Self-Adaptation for Domain Generalization*.

#### 10.1 Motivation & Key Contribution

**Problem — Domain Generalization (DG)**: Train models on source domains that generalize to **completely unseen** target domains. Unlike domain adaptation (DA), DG has **zero access** to target domain data during training. Models fail on unseen domains due to distribution shift — the test data distribution deviates from training distribution.

**Limitations of Prior DG Methods**:
- **Data augmentation**: Enriches source distribution but cannot cover all possible target shifts.
- **Domain-invariant representations**: Discard domain-specific information that may be useful.
- **Meta-learning**: Mimics generalization process but trains static models that cannot dynamically adjust.
- **Key gap**: All prior methods process instances individually without understanding the broader domain context during inference.

**Key Contributions — CASA (Context-Aware Self-Adaptation)**:
1. **CaFiLM (Context-Aware FiLM)**: Uses **mini-batch feature mean** as lightweight domain context signal — no complex global calculations needed.
2. **Two-stage framework**: Stage 1 trains base model on meta-source; Stage 2 trains CaFiLM adaptation module to bridge meta-source → meta-target while preserving source performance.
3. **Test-time adaptation without labels**: At inference, the incoming test mini-batch provides context automatically. No target labels or fine-tuning required.
4. **Dimension-specific modulation**: Unlike standard FiLM (which maps an external signal via MLP to γ,β), CaFiLM concatenates each feature dimension with its mini-batch mean and learns modulation via shared 1D convolution — guarantees output stays in same vector space as input.

#### 10.2 Datasets & Tasks

**Benchmark**: DomainBed test-bed, 5 standard DG datasets.

| Dataset | Domains | Classes | CASA Accuracy | ERM Baseline | SWAD | DNA | EoA |
|---------|---------|---------|---------------|-------------|------|-----|-----|
| **PACS** | 4 | 7 | **89.7%** | — | — | — | — |
| **VLCS** | 4 | 5 | **81.5%** | — | — | — | — |
| **OfficeHome** | 4 | 65 | **73.5%** | — | — | — | — |
| **TerraIncognita** | 4 | 10 | **52.0%** | — | — | — | — |
| **DomainNet** | 6 | 345 | **47.2%** | — | — | — | — |
| **Average** | — | — | **68.8%** | 63.3% | 66.9% | 67.6% | 68.0% |

CASA outperforms ERM by +5.5 percentage points and establishes new SOTA.

#### 10.3 Detailed Algorithm & Implementation

**CaFiLM Formulation**:

Step 1 — Compute context from mini-batch:
```
μ = E_{x ∈ X_b} [f_i(x)]     # mini-batch feature mean as domain context
```

Step 2 — Generate γ,β per feature dimension (shared 1D conv):
```
[γ_c]   = A_{2×2} [z_c] + b_{2×1}     # z_c = instance feature dim c
[β_c]            [μ_c]                  # μ_c = context mean dim c
```
Where `A_{2×2}` and `b_{2×1}` are shared across all dimensions.

Step 3 — Modulate:
```
CaFiLM(z_c) = γ_c · z_c + β_c
```

**Difference from Standard FiLM**: Standard FiLM uses an MLP on an external signal (question, task ID) to produce global γ,β vectors. CaFiLM operates **dimension-by-dimension**, concatenating each feature with its mini-batch mean, using a shared tiny conv — ensures output stays in same feature space (critical for classifier compatibility).

**Meta-Learning Objective** (Stage 2):
```
min_{θ_g} L_adapt + λ · L_preserve
```
- `L_adapt`: Cross-entropy on meta-target domains (teaches CaFiLM to bridge domain gap)
- `L_preserve`: Cross-entropy on meta-source domains (prevents catastrophic forgetting)

**Test-Time Inference** (no labels needed):
```
ŷ = argmax [E_{i ∈ {1:|T|}} [h_i(g_{C(x)}(f_i(x)))]]    # ensemble over |T| adapted meta-source models
```
Context μ computed automatically from incoming test mini-batch.

**Architecture & Training**:
| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 |
| CaFiLM Insertion | Between feature extractor and classifier |
| Optimizer | Adam |
| Stage 1 LR | 5×10⁻⁵ |
| Stage 2 LR (CaFiLM) | 1×10⁻³ |
| Stage 2 LR (classifier FT) | 5×10⁻⁵ (only for TerraIncognita, DomainNet) |
| Batch Size | 32 (16 for DomainNet) |
| Classifier Freezing | Frozen for PACS, VLCS, OfficeHome; unfrozen for TerraIncognita, DomainNet |
| Ensemble | Average probability vectors across |T| adapted meta-source models |

---

*End of review document*
