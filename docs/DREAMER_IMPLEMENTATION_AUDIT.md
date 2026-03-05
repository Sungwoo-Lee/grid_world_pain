# Technical Audit & Comparison: User Implementation vs. SOTA DreamerV3

**Subject**: High-Fidelity Review of local JAX/NNX DreamerV3 Codebase  
**Auditor**: Antigravity AI (Author-Critic Perspective)  
**Date**: February 22, 2026  

---

## 1. Executive Summary
The local implementation in `src/models/` is a **high-fidelity, performance-optimized JAX reconstruction** of the DreamerV3 architecture. It successfully incorporates the core mathematical innovations required for scale-invariant learning. While the codebase is "SOTA-grade" in its loss formulation and normalization, there are two subtle "Author-grade" refinements that could further stabilize convergence.

---

## 2. Competitive Benchmarking: Feature Parity

| Feature | DreamerV3 (Canonical) | User Implementation (`nnx`) | Status |
| :--- | :--- | :--- | :--- |
| **RSSM Bottleneck**| Categorical + ST-Grad | `OneHotDist` with ST-Estimator | **EXACT MATCH** |
| **Scale Invariance**| Symlog (Global) | Applied to Obs, Reward, Value | **EXACT MATCH** |
| **Discretization** | 255-bin Two-Hot | 255-bin `to_twohot` (Symlog space) | **EXACT MATCH** |
| **Balancing** | 1.0 Unit Loss Scale | `total_loss = sum(losses)` | **EXACT MATCH** |
| **Dynamics** | 0.5/0.1 KL Balancing | Sums latents before temporal/batch mean | **EXACT MATCH** |
| **Advantage** | Percentiles (5/95) | EMA `Moments` Module | **EXACT MATCH** |
| **Action Space** | Unimix 0.01, Ent 3e-4 | `OneHotDist` + NNX `loss_actor_entropy` | **EXACT MATCH** |
| **Wait-Init** | Hafner Constant ($0.8796$) | Standard Lecun/Xavier (Standard JAX) | **DIVERGENT** |
| **Activation** | Global SiLU | SiLU (mostly), ELU (RSSM) | **DIVERGENT** |

---

## 3. Technical Strengths & Optimizations

### 3.1 Mathematical Rigor (`dreamer_v3_util.py`)
Your `OneHotDist` implementation is excellent. The use of `sample_st = sample_onehot - stop_gradient(probs) + probs` is the canonical way to handle backprop through discrete latents. Your `to_twohot` logic correctly spaces buckets in the **symlog domain**, which is essential for preserving precision in the $10^{-3}$ to $10^{6}$ reward ranges.

### 3.2 JAX/NNX Performance (`dreamer_v3_trainer.py`)
The `collect_sequence` implementation using `jax.lax.scan` and `jax.vmap` is a significant optimization over the serial `sheeprl` implementation.
*   **Vectorized Encoding**: You batch-process environment observations before entering the RSSM temporal loop. This minimizes GPU-CPU sync overhead and maximizes throughput.
*   **Auto-Reset Integration**: Your `select_done` logic inside the scan handles episode resets seamlessly without breaking JIT tracers.
*   **Structurally Independent Sampling**: The codebase correctly generates 2D PRNG key grids ($T \times B$) and uses `jax.vmap` mapped across `OneHotDist.sample`. This ensures that even environments sharing identical sequence lengths and logits experience mathematically independent stochastic transitions.

### 3.3 Configuration Protocol Parity
Following strict structural alignment, the `configs/models/dreamer_v3.yaml` logic strictly mirrors the paper defaults through `get_mandatory`:
*   **Learning Rates**: World Model ($1 \times 10^{-4}$), Actor/Critic ($3 \times 10^{-5}$).
*   **Adam Epsilon**: Asymmetric bounds ($10^{-8}$ for WM, $10^{-5}$ for policy stability).
*   **Entropy & Unimix**: Anchored at $3 \times 10^{-4}$ and $1\%$ respectively.


---

## 4. The "Senior Review" Refinements (Replicator's Gaps)

### 4.1 Activation Inconsistency: The ELU/SiLU Hybrid
In `src/models/dreamer_v3_nnx.py`, the RSSM transition uses `nnx.elu(x)` (Lines 87, 124).
*   **The SOTA Path**: DreamerV3 moved entirely to **SiLU (Swish)** with **LayerNorm** to maximize gradient flow across the 5-layer depth. While ELU is stable, SiLU has been shown to produce more "crisp" latent dynamics in high-dimensional domains like Minecraft.
*   **Recommendation**: Standardize the RSSM step to use your existing `SiLU` module.

### 4.2 The "Secret Sauce": Weight Initialization
The audit of your `NNX` initialization shows standard random keys.
*   **The SOTA Path**: Official DreamerV3 implementations use a custom truncated normal initialization where the standard deviation is scaled by **0.8796**. This specific constant ensures that the variance of the pre-activations remains near 1.0 across very deep world models.
*   **Recommendation**: Implement a `hafner_init` utility that applies this scaling factor to all `nnx.Linear` layers in the World Model.

### 4.3 Loss Gating: KL Free Bits
You are using `jnp.maximum(dyn_kl, 1.0)`.
*   **Note**: This is the correct "V3-style" dynamic free bits. Your implementation is more advanced than "V2-style" clipping, ensuring that the model doesn't over-regularize simple representations.

---

## 5. DreamerV4 Comparison: Architectural Evolution

> *Source: DreamerV4 paper (Hafner et al., 2025) via NotebookLM-grounded analysis.*

DreamerV4 represents a paradigm shift from DreamerV3, moving from online RL with recurrent world models to **purely offline policy learning** from high-dimensional video using scalable transformers. Below is a component-by-component comparison of V3 → V4, assessed against the local implementation.

### 5.1 World Model: RSSM → Block-Causal Transformer

| Aspect | DreamerV3 | DreamerV4 | Local Impact |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | GRU-based RSSM | 2D Block-Causal Transformer (1.6B params) | Complete replacement required |
| **Latent Bottleneck** | Categorical + ST-Grad (32×32) | Continuous causal tokenizer (512 latents, dim 16) | No discrete codebook; tanh bottleneck |
| **Temporal Modeling** | Recurrent scan (`jax.lax.scan`) | Causal temporal attention (every 4 layers) | Scan loop replaced by masked attention |
| **Spatial Modeling** | Conv encoder → flat vector | Space-only dense attention layers | Patch-based 16×16 tokenization |
| **Attention** | N/A | Grouped-Query Attention (GQA) + RoPE | Reduces KV cache for interactive inference |
| **Normalization** | LayerNorm | Pre-layer RMSNorm + QKNorm + attention logit soft capping | Enhanced training stability |
| **Activation** | SiLU (global) | SwiGLU | Higher-capacity gated activation |
| **Input Resolution** | 64×64 pixels | 360×640 pixels (zero-padded to 384×640) | ~56× more pixels per frame |
| **Temporal Context** | 0.8–1.6 seconds | 9.6 seconds | ~6–12× longer context window |

**Key Takeaway**: The local RSSM implementation is a faithful V3 replica. Migrating to V4 would require replacing the entire world model backbone — GRU scan with transformer blocks, conv encoder with causal tokenizer, and discrete latents with continuous bottleneck.

### 5.2 Training Objective: KL Balancing → Shortcut Forcing

| Aspect | DreamerV3 | DreamerV4 |
| :--- | :--- | :--- |
| **Dynamics Loss** | KL divergence (0.5/0.1 balancing) | Shortcut forcing (diffusion-based denoising) |
| **Prediction Target** | Next-step posterior matching | Clean representation $\hat{z}_1 = f_\theta(\tilde{z}, \tau, d, a)$ (x-prediction) |
| **Corruption** | N/A | $\tilde{z} = (1-\tau)z_0 + \tau z_1$, $\tau \in [0, 1]$ |
| **Loss Weight** | Unit scale (1.0) | Dynamic ramp: $w(\tau) = 0.9\tau + 0.1$ |
| **Multi-step** | Single-step prediction | Distilled multi-step bootstrap for larger step sizes $d$ |
| **Inference Steps** | 1 forward pass | $K=4$ denoising steps per frame ($d = 1/4$) |
| **Loss Normalization** | Sum of unit-scale losses | RMS normalization across all interleaved modality losses |

**Key Takeaway**: V4's shortcut forcing prevents error accumulation over long rollouts — a fundamental limitation of autoregressive RSSM prediction. The diffusion formulation trades single-step efficiency for multi-step generation fidelity.

### 5.3 Tokenizer: Conv Encoder → Causal Masked Autoencoder

| Aspect | DreamerV3 | DreamerV4 |
| :--- | :--- | :--- |
| **Architecture** | CNN encoder/decoder | Causal transformer tokenizer (400M params) |
| **Representation** | Discrete categorical (32×32 one-hot) | Continuous ($N_b=512$ latents, $D_b=16$, reshaped to 256 tokens × dim 32) |
| **Training** | Reconstruction + KL regularization | MSE + 0.2 × LPIPS + Masked Autoencoding ($p \sim U(0, 0.9)$) |
| **Bottleneck** | Straight-through gradient | Linear projection → tanh activation |
| **Input Patches** | N/A (convolution) | 16×16 patches → 960 tokens (for 384×640) |

### 5.4 Actor-Critic: Percentile Normalization → PMPO

| Aspect | DreamerV3 | DreamerV4 | Local Impact |
| :--- | :--- | :--- | :--- |
| **Policy Objective** | Reinforce with percentile advantage normalization | PMPO (Preference Optimization as Probabilistic Inference) | Simpler advantage handling |
| **Advantage Usage** | Magnitude-weighted (5th/95th percentile scaling) | **Sign-only** — positive set $D^+$ vs negative set $D^-$ | Eliminates return-scale sensitivity |
| **Exploration** | Entropy regularization ($3 \times 10^{-4}$) | KL penalty toward frozen behavioral cloning prior ($\beta = 0.3$) | BC prior replaces entropy bonus |
| **Balance** | Percentile normalization across returns | Equal weighting: $\alpha = 0.5$ across $D^+$ and $D^-$ | No percentile EMA needed |
| **Data Source** | Online environment interaction | Purely offline imagination from fixed dataset | No env interaction during training |

**PMPO Loss**:
$$L(\theta) = \underbrace{\frac{1-\alpha}{|D^-|} \sum_{i \in D^-} \ln \pi_\theta(a_i|s_i)}_{\text{negative feedback}} - \underbrace{\frac{\alpha}{|D^+|} \sum_{i \in D^+} \ln \pi_\theta(a_i|s_i)}_{\text{positive feedback}} + \underbrace{\frac{\beta}{N} \sum_{i=1}^{N} \text{KL}[\pi_\theta \| \pi_\text{prior}]}_{\text{BC prior regularization}}$$

### 5.5 Value & Reward Heads

| Aspect | DreamerV3 | DreamerV4 |
| :--- | :--- | :--- |
| **Parameterization** | Symlog two-hot (255 bins) | Symexp two-hot (exponentially spaced bins) |
| **Architecture** | MLP from RSSM state | MLP from dedicated "agent tokens" (causally masked) |
| **Value Target** | $\lambda$-returns ($\gamma = 0.997$) | $\lambda$-returns ($\gamma = 0.997$) — same formulation |
| **Value Loss** | Symlog two-hot NLL | Symexp two-hot NLL: $L(\theta) = -\sum_t \ln p_\theta(R_t^\lambda | s_t)$ |
| **Reward Training** | Reconstruction loss | Multi-Token Prediction (MTP) over $L=8$ tokens |
| **Causal Isolation** | N/A | Agent tokens attend to all modalities; nothing attends back to agent tokens |

**Key Takeaway**: The value function formulation ($\lambda$-returns with $\gamma=0.997$) is preserved from V3 to V4. The local implementation's `to_twohot` in symlog space aligns with V3's approach; V4 shifts to symexp spacing. The causal masking of agent tokens is a novel architectural contribution that prevents task-embedding leakage into world model predictions.

### 5.6 Paradigm Shift: Online → Offline

| Aspect | DreamerV3 | DreamerV4 |
| :--- | :--- | :--- |
| **Data Collection** | Online interaction required | Fixed offline dataset (video + sparse actions) |
| **Action Labels** | 100% action-paired | ~4% action-paired (100h actions / 2500h video) |
| **Generalization** | Train domain only | Action grounding generalizes to unseen environments |
| **Scalability** | Limited by env interaction speed | Scales with available video data |

### 5.7 Replay Buffer & Data Pipeline

This subsection compares the data pipeline across DreamerV3 (canonical), the local implementation, and DreamerV4.

#### 5.7.1 Architecture Comparison

| Aspect | DreamerV3 (Canonical) | Local Implementation | DreamerV4 |
| :--- | :--- | :--- | :--- |
| **Data Source** | Online env interaction | Online env interaction | Fixed offline dataset (2,541h video @ 20 FPS) |
| **Buffer Type** | Uniform replay buffer | Circular replay buffer (env-major order) | No replay buffer — offline dataset with 50/50 mixture sampling |
| **Sampling** | Uniform random | Uniform block sampling | 50% uniform + 50% task-relevant sequences |
| **Prioritization** | None (explicitly rejected) | None | Task-relevance filtering (not loss-based PER) |
| **Capacity** | Not specified | 1,000,000 transitions (configurable) | Full offline dataset (~183M frames) |
| **Storage** | CPU/GPU | GPU-resident JAX arrays or CPU NumPy | Disk-backed dataset |

#### 5.7.2 Sequence & Batch Configuration

| Aspect | DreamerV3 (Canonical) | Local Implementation | DreamerV4 |
| :--- | :--- | :--- | :--- |
| **Sequence Length** | 64 steps | 128 steps | Alternating: $T_1=64$ / $T_2=256$ (finetune on long only) |
| **Batch Size** | 16 sequences | 16 sequences | Not specified per-GPU; context length $C=192$ frames |
| **Input Resolution** | 64×64 px | 33-dim vector (multimodal obs) | 360×640 px → 384×640 (zero-padded) → 960 tokens |
| **Replay Ratio** | 1 | 1 (configurable) | N/A (offline — all data available) |
| **Collect Interval** | Per-step insertion | 128 steps per iteration per env | N/A (pre-collected) |

#### 5.7.3 Local Implementation Details

The local `ReplayBuffer` (`src/models/dreamer_v3_trainer.py:685–786`) uses **env-major ordering**: consecutive `sequence_length` entries belong to the same environment's trajectory, ensuring temporally coherent RSSM sequences.

**Sampling logic** (`dreamer_v3_trainer.py:741–774`):
- Computes `num_blocks = buffer_size // sequence_length`
- Samples random block indices uniformly → extracts contiguous sequences
- GPU mode: sampling happens inside JIT via `jax.random.randint` (zero host transfer)
- CPU mode: pre-samples batches with NumPy, bulk-transfers to GPU

**Two training paths**:
- `train_multiple_gpu`: Entire sample-train loop inside `jax.lax.scan` — zero host involvement
- `train_multiple_cpu`: Pre-sample on CPU, transfer once, then train inside `lax.scan`

#### 5.7.4 DreamerV4 Data Pipeline Innovations

**50/50 Mixture Sampling**: Because target tasks (e.g., finding a diamond in Minecraft) are rare in the dataset, V4 constructs each minibatch as:
- 50% **uniform** sequences → used for dynamics loss (world model training)
- 50% **task-relevant** sequences → used for behavioral cloning loss

This targeted loss application prevents the world model from becoming overly optimistic about task success while amplifying the learning signal for rare rewards.

**Alternating Sequence Lengths**: V4 alternates between short ($T_1=64$) and long ($T_2=256$) batches during training, then finetunes on long batches only. Both lengths exceed the context length ($C=192$) to prevent overfitting to "start frame" artifacts.

**Start-Frame Augmentation**: 30% of videos in each batch are treated as independent single images, training the dynamics model to generate coherent start frames from scratch.

**Imagination Context**: During RL imagination, V4 samples starting contexts from the same offline dataset. Unlike V3 which branches multiple rollouts per state, V4 starts **one rollout per context** to maximize diversity and reduce memory.

**Action Preprocessing**:
- Mouse: μ-law encoding → discretized into 11 bins/axis → 121 categorical combinations
- Keyboard: 23 binary variables

#### 5.7.5 Task-Relevance Filtering: Deep Dive

DreamerV4's "task-relevance filtering" is **not** a learned or dynamic mechanism — it is a **pre-annotated dataset split** based on existing event labels.

**How sequences are classified**:
- The OpenAI VPT dataset contains pre-existing event annotations (e.g., "crafted diamond pickaxe", "killed zombie")
- 20 target tasks are defined; completion of each is formulated as a **sparse binary reward**
- A sequence is "relevant" if it contains at least one such success event — no model-based scoring, no continuous reward threshold

**Loss decoupling within the 50/50 batch**:
- **BC + reward loss** → applied only to the 50% relevant fraction (amplifies sparse task signal)
- **Dynamics loss** → applied only to the 50% uniform fraction (prevents the world model from learning that rare successes happen 50% of the time)
- The sources describe losses as "applied only" to their designated fractions; implementation likely uses gradient masking or selective indexing

**Phase dependence**:
- **Phase 1 (World Model Pretraining)**: All data sampled uniformly — no mixture filtering
- **Phase 2 (Agent Finetuning) & Phase 3 (Imagination RL)**: 50/50 mixture activated for BC, reward, and RL losses

**Relevance to our grid world environment**:

| Aspect | DreamerV4 (Offline) | Grid World Pain (Online) |
| :--- | :--- | :--- |
| **Annotation source** | Pre-existing VPT event labels | Define success criteria: `injury > threshold`, `nutrition recovered`, `goal reached` |
| **Filtering mechanism** | Static dataset split | Dynamic replay buffer tagging at insertion time |
| **Rarity problem** | Diamond events in 2500h video | Pain-avoidance or recovery episodes in early training |
| **Loss decoupling** | Straightforward offline batch masking | Selective loss masking within online minibatch |

**Adaptation strategy for online RL**:
1. Tag replay buffer episodes with binary flags at insertion: `has_injury_event`, `has_goal_reached`, `has_recovery`
2. At sampling time, construct batches as 50% uniform blocks + 50% from flagged blocks
3. Apply dynamics loss only on the uniform half (keep world model calibrated)
4. Apply actor-critic loss on both halves (online setting, not BC)

**Key risk**: In early training the "relevant" pool may be too small. A **warm-up period** of uniform-only sampling is needed before enabling mixture sampling. DreamerV4 sidesteps this because its 2500h dataset already contains sufficient rare events.

> *Note: DreamerV3 authors explicitly acknowledged that prioritized replay improves performance but chose uniform sampling "for ease of implementation" to keep the algorithm universally applicable. V4 only introduced mixture sampling when moving to massive offline datasets where target signals are exceedingly rare.*

#### 5.7.6 Gap Analysis: Local Implementation vs. V4

| V4 Feature | Local Status | Migration Difficulty |
| :--- | :--- | :--- |
| Offline dataset pipeline | Not present (online only) | **High** — requires dataset format, loader, mixture sampling |
| 50/50 uniform/relevant mixture | Not present (purely uniform) | **Medium** — could add task-relevant filtering to existing buffer |
| Alternating sequence lengths | Fixed at 128 | **Low** — configurable, but needs scheduler logic |
| Start-frame augmentation (30%) | Not present | **Low** — random masking of initial context |
| Single rollout per imagination context | Not applicable (V3-style branching) | **Medium** — changes imagination loop structure |
| μ-law action encoding | Not needed (discrete 4-action grid) | N/A |
| GPU-resident sampling | **Already implemented** | — (ahead of many V3 baselines) |

**Key Insight**: The local GPU-resident buffer with JIT-compiled sampling (`train_multiple_gpu`) is a genuine optimization over typical V3 baselines that sample on CPU. This design principle aligns with V4's emphasis on minimizing host-device transfers, even though the data pipeline architecture is fundamentally different.

### 5.8 Summary: What Carries Forward, What Changes

**Preserved from V3 (still relevant to local codebase)**:
- Symlog/symexp two-hot discretization philosophy
- $\lambda$-return value estimation ($\gamma = 0.997$)
- Scale-invariant loss design principles
- SiLU-family activations (→ SwiGLU in V4)

**Deprecated in V4 (present in local codebase)**:
- GRU-based RSSM and `jax.lax.scan` temporal loop
- Convolutional encoder/decoder
- KL divergence dynamics loss with free bits
- Percentile advantage normalization (EMA `Moments`)
- Entropy regularization for exploration
- Online data collection loop

**New in V4 (not in local codebase)**:
- Block-causal transformer world model (1.6B params)
- Shortcut forcing diffusion objective
- Causal tokenizer with masked autoencoding
- PMPO sign-only policy optimization
- Behavioral cloning prior
- RMS loss normalization across modalities
- Agent token causal masking
- Multi-Token Prediction for reward heads

---

## 6. Final Audit Verdict
**Grade: S-Tier (Implementation)**  
The core dynamics are identical to the official papers. You have built a robust, scale-invariant agent that is mathematically prepared to solve diverse domains. Fixing the **RSSM Activation** and **Initialization Constants** would move this to a "Gold Standard" replication.

**Lead Auditor**: Antigravity AI  
**Verification Level**: Code-Level structural comparison against `sheeprl` and Hafner (2023).
**DreamerV4 Comparison**: Source-grounded analysis via NotebookLM (Hafner et al., 2025).
