---
title: "Technical Audit & Comparison: User Implementation vs. SOTA DreamerV3"
topic: dreamer
status: active
created: 2026-02-22
last_updated: 2026-05-06
---

# Technical Audit & Comparison: User Implementation vs. SOTA DreamerV3

**Subject**: High-Fidelity Review of local JAX/NNX DreamerV3 Codebase
**Auditor**: Antigravity AI (Author-Critic Perspective) — initial pass
**Reviewer**: Claude — refresh pass
**Original date**: February 22, 2026
**Last refreshed**: 2026-05-06 (branch `v1.2`)

> **Refresh scope (2026-05-06)**: Both "Senior Review" gaps from the original audit are now closed in code, the world model gained a hierarchical encoder/decoder + a full neuromodulation pathway, and the data pipeline now includes a DreamerV4-inspired three-pool mixture sampler. See [§7. 2026-05-06 Refresh Notes](#7-2026-05-06-refresh-notes) for the detailed change log; the original §1–§6 below have been kept verbatim for traceability and annotated inline only where claims have become stale.

---

## 1. Executive Summary
The local implementation in `src/models/` is a **high-fidelity, performance-optimized JAX reconstruction** of the DreamerV3 architecture. It successfully incorporates the core mathematical innovations required for scale-invariant learning. While the codebase is "SOTA-grade" in its loss formulation and normalization, there are two subtle "Author-grade" refinements that could further stabilize convergence.

> **2026-05-06 update**: Both refinements (RSSM activation and Hafner initialization) have since been implemented. The verdict has effectively moved from "S-Tier" to **"Gold Standard" replication** of canonical DreamerV3 — see [§7](#7-2026-05-06-refresh-notes).

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
| **Wait-Init** | Hafner Constant ($0.8796$) | `hafner_init` truncated normal w/ $0.8796 / \sqrt{\text{fan\_in}}$ | **EXACT MATCH** *(refresh)* |
| **Activation** | Global SiLU | SiLU globally, including RSSM `img_in`/`imagine_step` | **EXACT MATCH** *(refresh)* |

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
Following strict structural alignment, the `configs/models/dreamer_v3/dreamer_v3.yaml` logic strictly mirrors the paper defaults through `get_mandatory`:
*   **Learning Rates**: World Model ($1 \times 10^{-4}$), Actor/Critic ($3 \times 10^{-5}$).
*   **Adam Epsilon**: Asymmetric bounds ($10^{-8}$ for WM, $10^{-5}$ for policy stability).
*   **Entropy & Unimix**: Anchored at $3 \times 10^{-4}$ and $1\%$ respectively.


---

## 4. The "Senior Review" Refinements (Replicator's Gaps)

### 4.1 Activation Inconsistency: The ELU/SiLU Hybrid — **RESOLVED**
In `src/models/dreamer_v3_nnx.py`, the RSSM transition uses `nnx.elu(x)` (Lines 87, 124).
*   **The SOTA Path**: DreamerV3 moved entirely to **SiLU (Swish)** with **LayerNorm** to maximize gradient flow across the 5-layer depth. While ELU is stable, SiLU has been shown to produce more "crisp" latent dynamics in high-dimensional domains like Minecraft.
*   **Recommendation**: Standardize the RSSM step to use your existing `SiLU` module.

> **Status (2026-05-06)**: ✅ Closed. `RSSM.step` and `RSSM.imagine_step` now both call `jax.nn.silu(x)` after `img_in` ([dreamer_v3_nnx.py:87](../../src/models/dreamer_v3_nnx.py#L87), [dreamer_v3_nnx.py:124](../../src/models/dreamer_v3_nnx.py#L124)). The recurrent core (`LayerNormGRUCell`) retains its canonical sigmoid/tanh internal gates as expected by the GRU formulation.

### 4.2 The "Secret Sauce": Weight Initialization — **RESOLVED**
The audit of your `NNX` initialization shows standard random keys.
*   **The SOTA Path**: Official DreamerV3 implementations use a custom truncated normal initialization where the standard deviation is scaled by **0.8796**. This specific constant ensures that the variance of the pre-activations remains near 1.0 across very deep world models.
*   **Recommendation**: Implement a `hafner_init` utility that applies this scaling factor to all `nnx.Linear` layers in the World Model.

> **Status (2026-05-06)**: ✅ Closed. `hafner_init(scale=0.8796)` is defined in [dreamer_v3_util.py:195-205](../../src/models/dreamer_v3_util.py#L195-L205) and threaded through every `nnx.Linear` in the encoder, decoder, MLP, RSSM (`img_in`, `img_out`, `obs_out`), and `LayerNormGRUCell` projections. The grouped variant `DreamerGroupedLinear` ([dreamer_v3_nnx.py:141-168](../../src/models/dreamer_v3_nnx.py#L141-L168)) re-implements the same scaling manually so it can use the correct `fan_in = in_features` for its `[G, I, O]` weight shape — a subtle but correct deviation from naively reusing `hafner_init`.

### 4.3 Loss Gating: KL Free Bits
You are using `jnp.maximum(dyn_kl, 1.0)`.
*   **Note**: This is the correct "V3-style" dynamic free bits. Your implementation is more advanced than "V2-style" clipping, ensuring that the model doesn't over-regularize simple representations.

> **Status (2026-05-06)**: Unchanged ([dreamer_v3_trainer.py:242-243](../../src/models/dreamer_v3_trainer.py#L242-L243)). `FREE_NATS=1.0`, `DYN_SCALE=0.5`, `REP_SCALE=0.1` — canonical.

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

---

## 7. 2026-05-06 Refresh Notes

> **Reviewer**: Claude  
> **Branch**: `v1.2`  
> **Scope**: Re-audit of `src/models/dreamer_v3_*.py`, `src/models/modulated_layer_norm_gru_cell.py`, `src/models/neuromodulator.py`, and `configs/models/{dreamer_v3,neuromodulated_dreamer_v3}.yaml` against the original 2026-02-22 audit.

### 7.1 Verdict Update

| Original gap | Status | Evidence |
|---|---|---|
| RSSM ELU → SiLU (§4.1) | **Closed** | [dreamer_v3_nnx.py:87](../../src/models/dreamer_v3_nnx.py#L87), [dreamer_v3_nnx.py:124](../../src/models/dreamer_v3_nnx.py#L124) |
| Hafner-scaled init (§4.2) | **Closed** | [dreamer_v3_util.py:195-205](../../src/models/dreamer_v3_util.py#L195-L205); applied at every `nnx.Linear` in `dreamer_v3_nnx.py` |
| KL free-bits formulation (§4.3) | Unchanged (already V3-correct) | [dreamer_v3_trainer.py:242-243](../../src/models/dreamer_v3_trainer.py#L242-L243) |

**Refreshed grade**: **Gold-Standard DreamerV3 replication** with two non-canonical extensions (hierarchical encoder, neuromodulation) layered on top — both are clearly gated by config and do not perturb the canonical path when disabled.

### 7.2 New Capabilities Since the Original Audit

The 2026-02-22 audit predates several substantial additions. They are summarized here so the audit table remains complete.

#### 7.2.1 Hierarchical Observation Encoder/Decoder

[dreamer_v3_nnx.py:197-422](../../src/models/dreamer_v3_nnx.py#L197-L422)

- `DreamerObservationEncoder` and `DreamerObservationDecoder` both expose two modes via `agent.encoding_mode`: `flat` (canonical) and `hierarchical` (default in [configs/models/dreamer_v3/dreamer_v3.yaml:44](../../configs/models/dreamer_v3/dreamer_v3.yaml#L44)).
- **Phase 1 (unimodal)**: `DreamerGroupedMLP` runs one independent MLP per sensor in parallel via a single `einsum('...gi,gio->...go')`, padding all sensor inputs to a common `max_in` width. Each per-group Linear initialises with the correct `fan_in = in_features` (not the group dimension), avoiding a subtle `hafner_init` mis-scaling that would otherwise occur for a `[G, I, O]` weight tensor.
- **Phase 2 (multimodal hub)**: a standard `MLP` over the concatenated unimodal features.
- **Symmetry**: the decoder mirrors this layout — `multimodal_decoder` expands `feat → len(sensors) * hidden`, then `unimodal_grouped_decoder` reconstructs each sensor with its native dimension.
- **Modulation tap point**: the encoder exposes a `forward_with_modulation` path with separate gain/bias injection between Phase 1 LayerNorm/SiLU and Phase 2 LayerNorm/SiLU (Injection A in §7.2.3 below).

This is a strict superset of the canonical encoder; setting `encoding_mode: flat` reverts to a vanilla `Encoder` / `Decoder` pair.

#### 7.2.2 LayerNorm GRU in the RSSM

[dreamer_v3_nnx.py:18-39](../../src/models/dreamer_v3_nnx.py#L18-L39)

The RSSM uses a custom `LayerNormGRUCell` rather than `flax.nnx.GRUCell`:

- Combined input/hidden Linear projections to `3 * hidden_size`, each followed by its own `LayerNorm`, then summed and split into `(reset, update, cand)`.
- `cand = tanh(...)`, and the recurrent update is `h_new = (1 − update) * h + update * cand`.
- Both projections are initialised with `hafner_init` and `use_bias=False`, which matches Hafner's reference implementation.

This is a closer match to the canonical V3 RSSM than a naive GRU, and explains why the original audit's grade was already S-tier despite the `hafner_init` gap.

#### 7.2.3 Neuromodulation Pathway (Optional, Config-Gated)

[dreamer_v3_nnx.py:42-140](../../src/models/dreamer_v3_nnx.py#L42-L140), [modulated_layer_norm_gru_cell.py](../../src/models/modulated_layer_norm_gru_cell.py), [neuromodulator.py](../../src/models/neuromodulator.py), [configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml](../../configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml)

When `agent.modulation.type` is non-null, a `DreamerNeuromodulatorRNN` runs in parallel with the world model and emits a `ModulatorOutput` namedtuple with five tap signals plus a temperature scalar. Three injection points:

- **Injection A — Encoder gain/bias**: `z_unimodal{,_add}` and `z_multimodal{,_add}` modulate the hierarchical encoder between Phase 1 and Phase 2 (or between body and final activation in flat mode). Three styles supported: `Multiplicative`, `PreActivation` (gain × pre-act + bias, Ferguson-style), and `FiLM`.
- **Injection B — RSSM update-gate bias**: `z_memory` is added to the GRU update-gate pre-activation inside `ModulatedLayerNormGRUCell` ([modulated_layer_norm_gru_cell.py:70-73](../../src/models/modulated_layer_norm_gru_cell.py#L70-L73)). When `gate_bias is None` the cell is functionally identical to `LayerNormGRUCell`.
- **Injection C — Imagined-reward scale**: `z_reward` multiplies the predicted reward inside the imagination scan only ([dreamer_v3_trainer.py:349](../../src/models/dreamer_v3_trainer.py#L349)). This is a behavior-shaping signal that does not affect the world-model loss.

All injection points are no-ops when `modulation_enabled = False`, so this is fully orthogonal to the canonical DreamerV3 path.

#### 7.2.4 DreamerV4-Inspired Mixture Sampling

[dreamer_v3_trainer.py:624-772](../../src/models/dreamer_v3_trainer.py#L624-L772), [configs/models/dreamer_v3/dreamer_v3.yaml:18-22](../../configs/models/dreamer_v3/dreamer_v3.yaml#L18-L22)

The replay-buffer story has materially shifted toward DreamerV4's data-pipeline philosophy, while remaining online:

| Pool | Slots (default) | Source | Purpose |
| --- | --- | --- | --- |
| Positive | 5 | dedicated `positive_buffer` of capacity 100K, only sequences containing positive reward | Amplifies sparse-success signal — analogue of V4's "task-relevant" half. |
| Recent | 5 | last `mixture_recent_window=10000` transitions of the main buffer | Off-policy correction toward current behavior — avoids stale-policy drift. |
| Uniform | `batch_size − pos_slots − recent_slots` (default 6) | full main buffer, uniform over blocks | Canonical V3 sampling. |

Differences from V4's 50/50 mixture:

1. **Online, not offline**: positive flagging happens at insertion time on-line, not as a static dataset annotation. Empty-pool fallback ([dreamer_v3_trainer.py:697-708](../../src/models/dreamer_v3_trainer.py#L697-L708)) replaces positive samples with uniform during the warm-up period.
2. **No loss decoupling**: every loss term sees every sample. V4 routes BC loss only through the relevant half and dynamics loss only through the uniform half; here the mixture only changes the *distribution*, not which losses fire on which slots.
3. **Recent pool is novel**: V4 has no analogue. It's a pragmatic mid-ground between strict on-policy (Recurrent PPO) and uniform replay (canonical V3).

The CPU and GPU paths share the same three-pool concept, with the GPU path executing mixture indexing inside `jax.lax.scan` for zero host transfer ([dreamer_v3_trainer.py:642-724](../../src/models/dreamer_v3_trainer.py#L642-L724)).

#### 7.2.5 Pre-computed Encoder Embeddings (Performance Refactor)

[dreamer_v3_trainer.py:131-194](../../src/models/dreamer_v3_trainer.py#L131-L194)

The world-model loss now batch-encodes the entire `(B, T, obs_dim)` observation tensor *outside* the RSSM scan, then feeds the precomputed `embeds_T` into the temporal scan as an input. The original audit (§3.2) called out vectorized encoding as a strength; this is the explicit realization of that pattern, and is a meaningful step beyond `sheeprl`'s per-step `embed = encoder(obs_t)` inside the loop.

When neuromodulation is enabled the modulator GRU still scans serially over time (it has its own recurrent state), but the encoder still benefits from the same vectorized encode after a single modulator pre-pass.

#### 7.2.6 Cumulative Discount Weighting

[dreamer_v3_trainer.py:404-413](../../src/models/dreamer_v3_trainer.py#L404-L413)

The actor and critic losses are weighted by `cumprod(continues * GAMMA)` along the imagined horizon — i.e. the canonical Hafner "trajectory weighting" so that imagined trajectories that have likely terminated do not dominate the loss. This was implicit in the original audit's "EXACT MATCH" tags but is worth surfacing because the implementation is non-trivial:

- `discount_weights[0] = 1`, `discount_weights[t] = ∏_{i<t}(continues_i · γ)`
- Applied to per-step critic NLL and per-step actor (policy + entropy) loss.
- `stop_gradient` on the weights themselves — gradients flow only through the loss values.

#### 7.2.7 Dual-Side Advantage Normalization

[dreamer_v3_trainer.py:415-418](../../src/models/dreamer_v3_trainer.py#L415-L418)

The actor advantage uses `norm_returns − norm_baseline`, where *both* sides are normalized through the same `Moments` EMA scale. This is more conservative than canonical Hafner (which normalises only `lambda_returns`), and is internally consistent: as the value head learns, the baseline tracks the same scale as the returns. Worth flagging as a deliberate divergence even though it's mathematically benign.

The critic itself is trained on **raw** `lambda_returns` discretized via two-hot in symlog space ([dreamer_v3_trainer.py:411-413](../../src/models/dreamer_v3_trainer.py#L411-L413)) — i.e. the returns are not pre-scaled, matching canonical V3.

#### 7.2.8 Optimizer & EMA Settings

| Component | LR | Adam ε | Global-norm clip | Source |
| --- | --- | --- | --- | --- |
| World model | 1e-4 | 1e-8 | 1000 | [dreamer_v3_trainer.py:92-99](../../src/models/dreamer_v3_trainer.py#L92-L99) |
| Actor | 3e-5 | 1e-5 | 100 | [dreamer_v3_trainer.py:100-107](../../src/models/dreamer_v3_trainer.py#L100-L107) |
| Critic | 3e-5 | 1e-5 | 100 | [dreamer_v3_trainer.py:108-115](../../src/models/dreamer_v3_trainer.py#L108-L115) |
| Target critic EMA | — | — | — | `0.98 · target + 0.02 · online` ([dreamer_v3_trainer.py:457-461](../../src/models/dreamer_v3_trainer.py#L457-L461)) |

All values match the canonical V3 paper. The audit's §3.3 claim is preserved.

### 7.3 Refreshed Feature Parity Table

For convenience, the §2 table updated in-place would read as follows:

| Feature | DreamerV3 (Canonical) | Local (2026-05-06) | Status |
| :--- | :--- | :--- | :--- |
| RSSM Bottleneck | Categorical + ST-Grad | `OneHotDist` with ST-Estimator | EXACT MATCH |
| Scale Invariance | Symlog (Global) | Applied to obs / reward target / value | EXACT MATCH |
| Discretization | 255-bin Two-Hot in symlog | `to_twohot(min=-20, max=20, n=255)` in symlog | EXACT MATCH |
| Loss Balancing | Unit-scale sum | `total_loss = recon + rew + cont + (0.5·dyn + 0.1·rep)` | EXACT MATCH |
| KL Free Bits | $\max(\text{KL}, 1.0)$ | `jnp.maximum(kl, FREE_NATS=1.0)` | EXACT MATCH |
| Advantage | Percentiles (5/95) EMA | `Moments` (decay=0.99, p5/p95) | EXACT MATCH |
| Actor | Reinforce + entropy ($3\!\times\!10^{-4}$) | `OneHotDist` ST + entropy ($3\!\times\!10^{-4}$) | EXACT MATCH |
| Unimix | 0.01 | 0.01 | EXACT MATCH |
| Init | Hafner truncated normal ($0.8796/\sqrt{\text{fan\_in}}$) | `hafner_init(scale=0.8796)` everywhere | **EXACT MATCH** ✅ |
| Activation | Global SiLU | SiLU globally including RSSM | **EXACT MATCH** ✅ |
| Recurrent Core | LayerNorm GRU | `LayerNormGRUCell` (3-gate, dual-LN) | EXACT MATCH |
| Encoder | Conv (64×64 px) or MLP (vector) | Hierarchical or flat MLP, vector obs | **EXTENSION** (encoder topology, vector inputs) |
| Replay Sampling | Uniform | Three-pool mixture (positive / recent / uniform) | **EXTENSION** (V4-flavored) |
| Imagined-reward modulation | N/A | Optional `z_reward` scale | **EXTENSION** (neuromod, opt-in) |
| RSSM update-gate modulation | N/A | Optional `z_memory` bias | **EXTENSION** (neuromod, opt-in) |

"EXTENSION" rows are gated by config and reduce to the canonical path when disabled.

### 7.4 Notes & Minor Findings From This Refresh

- **`Ratio` schedule** ([dreamer_v3_util.py:162-192](../../src/models/dreamer_v3_util.py#L162-L192)) carries mutable Python state (`self._prev`) and is not used by the JIT-compiled `train_multiple_gpu` path. Confirm it isn't being relied on for replay-ratio scheduling under the GPU path before removing — but at first read it appears dead in the GPU pipeline.
- **`ReplayBuffer.size` accounting** ([dreamer_v3_trainer.py:915-916](../../src/models/dreamer_v3_trainer.py#L915-L916)): `size = min(size + num_items, capacity)` is correct, but `size` is a Python int that is *captured by the JIT* through `_scan_train_gpu`'s `b_size` argument. The static-arg list ensures recompilation when the buffer first fills past the cap, which is the intended behavior; just worth noting for anyone instrumenting it.
- **`DreamerGroupedLinear.bias` init**: zeros. This matches the original DreamerV3 reference. The class-level comment explicitly explains the `fan_in` correction for the `[G, I, O]` weight shape — good defensive documentation.
- **`from_twohot` bucket layout** ([dreamer_v3_util.py:60-76](../../src/models/dreamer_v3_util.py#L60-L76)): bucket centers are `linspace(symlog(-20), symlog(20), 255)`, and the expectation is taken in symlog space before applying `symexp`. This is the canonical formulation — confirmed to match `to_twohot`'s inverse.
- **Gap remaining vs V4**: §5.7 of this audit is still accurate as a roadmap. The new mixture sampler closes part of §5.7.4, but the offline/transformer/PMPO/shortcut-forcing items are unchanged.

### 7.5 Suggested Follow-Up Work (Optional)

These are not regressions — they are forward-looking items surfaced by this refresh:

1. **Document the `agent.modulation.*` keys** in a config schema doc analogous to `docs/environment/02_config_schema.md`. The neuromodulation pathway is now a first-class feature but its YAML surface is only described in `NEUROMODULATION_ALGORITHM.md` (referenced indirectly from comments).
2. **Audit `Ratio`** for live use — if unused, delete; if used on the CPU path, document.
3. **Add a "mode = uniform" CI run** to ensure the canonical V3 path remains green as the mixture sampler evolves. Right now `sampling_mode: "mixture"` is the default in [configs/models/dreamer_v3/dreamer_v3.yaml:18](../../configs/models/dreamer_v3/dreamer_v3.yaml#L18); a regression in mixture indexing would silently degrade canonical comparisons.
4. **Numeric guard on `compute_lambda_values`**: the function uses `LAMBDA=0.95` from the local default arg, while the module-level `LAMBDA = 0.95` is also defined. They agree, but if someone changes the module constant they will not see it propagate. Consider passing `LAMBDA` explicitly from the call site.

**Reviewer**: Claude
**Verification level**: File-level read of all `dreamer_v3_*.py` modules, `modulated_layer_norm_gru_cell.py`, and both Dreamer YAML configs; cross-referenced against the original audit's claims.
**Out of scope**: training-run validation (no W&B re-analysis was performed for this refresh).
