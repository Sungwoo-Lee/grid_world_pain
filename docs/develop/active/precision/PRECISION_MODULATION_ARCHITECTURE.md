---
title: "Precision Modulation Architecture: From FiLM to Prediction-Gated Neuromodulation"
topic: precision
status: active
created: 2026-03-23
last_updated: 2026-04-12
---

# Precision Modulation Architecture: From FiLM to Prediction-Gated Neuromodulation

> **Date**: 2026-03-20
> **Author**: Claude (analysis & synthesis), Sungwoo (direction)
> **Context**: Synthesizing Active Inference precision theory, neuromodulatory algorithms, FiLM-Ensemble analysis, and DreamerV3 world model capabilities into a concrete architectural proposal for sensory precision modulation
> **Sources**: [FiLM_ENSEMBLE_SENSORY_PRECISION.md](FiLM_ENSEMBLE_SENSORY_PRECISION.md), [PRECISION_MODULATION.md](PRECISION_MODULATION.md), [NEUROMODULATION_ALGORITHM.md](NEUROMODULATION_ALGORITHM.md), Kendall & Gal (2017), Friston (2023), Doya (2002), Costacurta et al. (2024), Ferguson & Cardin (2020), Shine et al. (2021), AlKilany & Goodman (2025)

---

## 1. The Problem: Three Converging Lines of Evidence

Three independent lines of work in this project converge on the same architectural requirement — **the agent needs a training signal for precision estimation**, and the current neuromodulatory architecture doesn't provide one.

### 1.1 From Active Inference: Precision Requires Prediction Error

Active Inference (Friston, 2023) defines precision as the inverse variance of sensory prediction errors: π = 1/σ². The critical equation is the belief update rule:

$$\dot{\mu} = D\mu - \Pi_s \epsilon_s \frac{\partial \epsilon_s}{\partial \mu}$$

Where Π_s is sensory precision and ε_s = o - g(μ) is the sensory prediction error. The key insight: **precision is not a static parameter but a dynamic variable optimized alongside beliefs**, driven by the statistics of prediction errors. Sustained large prediction errors → lower Π_s (Friston's precision dynamics: $\dot{\Pi}_s \propto \beta - \epsilon_s^2$).

Our environment implements the aleatoric side — state-dependent noise on each sensory channel (see [PRECISION_MODULATION.md](PRECISION_MODULATION.md)):

$$\sigma_{eff} = \sigma_{base} \cdot (1 + \alpha \cdot \hat{I})$$

But no component of the agent's architecture estimates or acts on this noise. The modulator generates γ,β signals that _could_ implement precision weighting, but has no objective that connects those signals to observation reliability.

### 1.2 From v7/v8 Experiments: The Modulator Learns Nothing

The NMN experiments (v7, v8) tested whether FiLM-style modulation — generating state-dependent γ,β from the GRU hidden state — could learn precision modulation:

| Experiment | Result |
|-----------|--------|
| v7 (NoNoise) | FiLM ≈ unmodulated LN baseline. Gates collapse to near-identity. |
| v8 (Noise) | FiLM provides zero benefit. Best FiLM (MC g16: 283.44) ties with unmodulated (MC: 283.64). |

**Root cause**: The PPO loss provides no gradient signal for precision. The modulator has no reason to produce non-identity gates because the RL objective doesn't reward knowing _which channels are unreliable_. This is not a representation problem (single FiLM can represent precision) — it is a **training signal problem**.

### 1.3 From Neuromodulation Architecture: Four Injection Sites, Zero Precision Objectives

The current bi-recurrent neuromodulator ([NEUROMODULATION_ALGORITHM.md](NEUROMODULATION_ALGORITHM.md)) has a rich multi-site architecture:

| Injection | Target | Signal | Training Signal |
|-----------|--------|--------|-----------------|
| **A (Perception)** | Encoder features | γ (gain), β (threshold shift) | PPO/WM loss only — no precision-specific objective |
| **B (Memory)** | GRU update gate bias | z_memory | PPO/WM loss only |
| **C-PPO (Exploration)** | Policy temperature | temperature | PPO loss (entropy bonus provides weak signal) |
| **C-DreamerV3 (Reward)** | Imagined reward scaling | z_reward | No direct gradient (imagination only) |

All four injections are trained end-to-end through the RL loss. This gives a training signal for _behavioral performance_ but not for _precision estimation_. The modulator can learn "be cautious after injury" (if it helps survival) but not "olfaction is 3× noisier when injured" — the latter requires prediction errors.

### 1.4 The Missing Piece

| What We Have | What We Need |
|-------------|-------------|
| Multi-channel perceptual noise (environment) | Agent mechanism to **estimate** per-channel noise |
| FiLM-style gain control (Injection A) | Training signal that **drives** the gain toward precision-appropriate values |
| Recurrent modulator with affective inertia (GRU) | Objective that rewards **accurate** precision estimation, not just survival |
| DreamerV3 world model with observation reconstruction | **Heteroscedastic** reconstruction that learns per-channel variance |

---

## 2. The Proposal: Prediction-Gated Neuromodulation

### 2.1 Core Idea

Add a **precision estimation objective** to the neuromodulatory architecture. The modulator doesn't just generate γ,β from hidden state — it also predicts how reliable each sensory channel is, trained by the statistics of observation prediction errors.

This is the marriage of two existing systems:
- **The neuromodulator** (NEUROMODULATION_ALGORITHM.md) provides the multi-site modulation architecture with affective inertia
- **The precision framework** (Friston, 2023; Kendall & Gal, 2017) provides the training objective — heteroscedastic prediction error

Neither alone is sufficient. The modulator without a precision objective learns identity gates (v7/v8). The precision objective without a modulation mechanism learns noise levels but can't act on them.

### 2.2 Architecture: Two Tracks

The architecture differs depending on whether we use the model-free (PPO) or model-based (DreamerV3) agent.

#### Track A: PPO + Precision Head (Model-Free)

The modulator gains an additional head — the **precision head** — that predicts per-channel observation reliability, trained by an auxiliary heteroscedastic prediction loss.

```
Observation o_t ───────────────────────────────→ Input Projection (Linear + act)
                                                          │
                     ┌────────────────────────────────────┤
                     │                                    │
               Modulator GRU                         Task GRU
               (slow, h_mod)                        (fast, h_task)
                     │                                    │
          ┌──────────┼──────────┬─────────┐              │
          │          │          │         │              │
     head_percept  head_mem  head_act  head_precision   ├── Actor
     (γ, β)       (z_mem)   (temp)   (ô_{t+1}, π̂_i)  └── Critic
          │                              │
          │         ◄────────────────────┘
          │         precision gates γ,β
          ▼
   Precision-Gated FiLM on encoder features
```

**New component — `head_precision`**: An MLP head on the modulator that outputs:
- ô_{t+1}: predicted next observation (obs_dim)
- s_i = log π̂_i: per-modality log-precision (9 values, one per modality)

**Precision loss** (Kendall & Gal, 2017):

$$L_{precision} = \sum_{i=1}^{9} \left[ \frac{1}{2} \exp(s_i) \cdot \|o_{t+1,i} - \hat{o}_{t+1,i}\|^2 - \frac{1}{2} s_i \right]$$

**Total loss**:

$$L_{total} = L_{PPO} + \lambda_{pred} \cdot L_{precision}$$

**How precision gates the existing modulator output**:

```python
# Existing modulator output (unchanged)
gamma = sigmoid(z_percept)        # gain from head_percept
beta = z_percept_add              # threshold shift from head_percept_add

# New precision head output (trained by L_precision)
pi_gate = sigmoid(log_precision)  # per-channel precision, squashed to [0, 1]
# Broadcast pi_gate from 9 modalities to full feature dimension
pi_gate_full = broadcast_modality_to_features(pi_gate)

# Precision-gated PreActivation modulation
# High precision → full FiLM effect (trust features)
# Low precision → bypass FiLM, pass features through (rely on GRU prior)
x_linear = input_proj(obs_t)
x_modulated = relu(x_linear * gamma + beta)
x_out = pi_gate_full * x_modulated + (1 - pi_gate_full) * relu(x_linear)
```

This preserves the full existing modulator architecture (Injection A PreActivation mode) but adds a precision gate that **controls how much the modulation affects downstream features**. When precision is low (noisy channel), the FiLM transform is suppressed — features pass through unmodulated, relying on temporal priors from the GRU hidden state.

#### Track B: DreamerV3 + Heteroscedastic Decoder (Model-Based)

In DreamerV3, the observation prediction is not auxiliary — it is the **core world model objective**. The precision head becomes part of the decoder.

```
                          WORLD MODEL (trained on real data)
                          ┌──────────────────────────────────────────────┐
Observation o_t ─→ Encoder ─→ Posterior z_t ─→ RSSM h_t
                     ↑                              │
              Injection A                           ├─→ Decoder ─→ ô_t (two-hot)
              (γ, β gated                           ├─→ PrecisionHead ─→ log π̂_i (9 modalities)
               by π̂)                               ├─→ Reward head
                                                    └─→ Continue head
                          └──────────────────────────────────────────────┘
                                                    │
                     Modulator GRU (observation mode)│
                     ├── head_percept (γ, β)        │
                     ├── head_memory (z_mem) ────────┘ (gate-bias on RSSM GRU)
                     └── head_reward (z_rew) ──→ imagination mode only
```

**Key difference from Track A**: The precision head is a separate MLP operating on the latent features `feat = concat(deter, stoch)`, trained by the heteroscedastic reconstruction loss as part of L_world:

$$L_{world} = L_{recon\_twohot} + \lambda_{\pi} \cdot L_{precision} + L_{KL} + L_{reward} + L_{continue}$$

The existing two-hot decoder remains unchanged (proven reconstruction quality). The precision head is a parallel continuous-space head trained specifically to learn per-channel precision.

**Imagination-mode precision**: During latent rollouts, the precision head can be queried on imagined states to produce imagined precision estimates. This allows precision to modulate reward interpretation:

```python
# During imagination
precision_imagined = precision_head(feat)           # per-channel precision from imagined state
mean_precision = mean(sigmoid(precision_imagined))  # aggregate precision

# Precision-modulated reward interpretation
rew_effective = rew * (1 - lambda_caution * (1 - mean_precision))
# Low precision → cautious reward interpretation → risk-averse planning
```

### 2.3 Precision Head Design

The precision head is architecturally simple but its design must align with the modality structure of the observation space.

**Per-modality precision (9 outputs)**:

| Index | Modality | Obs Dims | Noise Mode | Precision Output |
|-------|----------|----------|------------|-----------------|
| 0 | Injury | 1 | state_dependent (α=1.5) | 1 log π̂ |
| 1 | Nutrition | 1 | constant (σ=0.1) | 1 log π̂ |
| 2 | Satiation | 1 | constant (σ=0.1) | 1 log π̂ |
| 3 | Extero Nociception | 1 | constant (σ=0.01) | 1 log π̂ |
| 4 | Olfaction | 5 | state_dependent (α=2.0) | 1 log π̂ (shared) |
| 5 | Collision | variable | constant (σ=0.01) | 1 log π̂ |
| 6 | Proprioception | 5 | constant (σ=0.05) | 1 log π̂ (shared) |
| 7 | Visual | variable | state_dependent (α=3.0) | 1 log π̂ (shared) |
| 8 | Location | 2 | constant (σ=0.01) | 1 log π̂ (shared) |

Multi-element modalities (olfaction, visual, proprioception, location) share a single precision parameter within the modality. This matches the environment's noise configuration — all elements within a modality share the same noise regime — and is consistent with AlKilany & Goodman (2025)'s spatial grouping finding that macroscopic modulation is equally effective as fine-grained.

**Prediction target**: The full observation vector o_{t+1}. For the heteroscedastic loss, the per-modality precision π̂_i is broadcast to all elements within that modality:

```python
# For each modality i with dim d_i and precision log_pi_i:
# Prediction error contribution:
L_i = (1/2) * exp(log_pi_i) * sum(||o_{t+1}[start:end] - o_hat[start:end]||^2) / d_i  -  (1/2) * log_pi_i
```

Normalizing by d_i prevents high-dimensional modalities (visual) from dominating the loss.

### 2.4 Connection to Friston's Precision Dynamics

The heteroscedastic loss implements Friston's precision dynamics in a gradient-based form:

| Friston (2023) | Our Implementation |
|----------------|-------------------|
| $\dot{\Pi}_s \propto \beta - \epsilon_s^2$ | $\frac{\partial L}{\partial s_i} = \frac{1}{2}\exp(s_i) \cdot MSE_i - \frac{1}{2}$ |
| Sustained large ε_s → Π_s decreases | Large MSE_i → gradient pushes s_i (= log π̂_i) down |
| Tonic baseline β prevents collapse to zero | Regularization term $-\frac{1}{2}s_i$ prevents infinite suppression |
| ACh encodes expected sensory precision | π̂_i represents learned expected sensory precision per channel |

At equilibrium ($\frac{\partial L}{\partial s_i} = 0$): $\exp(s_i^*) = 1/MSE_i^*$, i.e., $\pi_i^* = 1/\sigma_{empirical}^2$ — precision converges to the inverse empirical variance of prediction errors. This is exactly the Active Inference definition of precision.

---

## 3. Integration with the Existing Neuromodulator

### 3.1 What Changes and What Stays

The existing bi-recurrent neuromodulator architecture is preserved almost entirely. The precision head is an **addition**, not a replacement.

| Component | Current Architecture | With Precision Head |
|-----------|---------------------|-------------------|
| Modulator GRU core | Shared GRU, h_mod | **Unchanged** |
| head_percept (γ) | Sigmoid gain, gating encoder features | **Unchanged** — but now gated by precision |
| head_percept_add (β) | Threshold shift (PreActivation mode) | **Unchanged** — but now gated by precision |
| head_memory (z_mem) | Gate-bias on task GRU update gate | **Unchanged** |
| head_action (temperature) | Bounded policy temperature (PPO) | **Unchanged** |
| head_reward (z_rew) | Imagined reward scale (DreamerV3) | **Unchanged** |
| **head_precision** (NEW) | N/A | Predicts ô_{t+1} + 9 log-precision values |
| **L_precision** (NEW) | N/A | Heteroscedastic prediction loss |
| **Precision gating** (NEW) | N/A | π̂_i gates Injection A influence |

### 3.2 The Modulator Now Has Two Roles

**Role 1 (existing): State-dependent modulation** — the modulator learns how to transform features based on the agent's internal state (injury, satiation, affective tone). This is controlled by γ, β, z_memory, temperature/z_reward. Trained by RL loss.

**Role 2 (new): Precision estimation** — the modulator learns how reliable each sensory channel is, based on its ability to predict observations. This is controlled by π̂_i. Trained by L_precision.

The two roles are coupled through the shared GRU hidden state h_mod. The same recurrent state that carries "affective tone" (Role 1) also integrates the information needed for precision estimation (Role 2). This is biologically correct: acetylcholine, which encodes sensory precision in the brain (Friston, 2023), is a neuromodulator with the same slow dynamics and global broadcast properties as the other neuromodulators (dopamine, serotonin, noradrenaline) that our modulator already models.

### 3.3 How Precision Gating Interacts with Existing Injections

**Injection A (Perception) — Now Precision-Gated**:

_Without precision (current v7/v8)_:
```
x_out = relu(gamma * x_linear + beta)     # PreActivation FiLM
```
The modulator controls _how_ features are transformed, but has no signal for _whether_ to transform them.

_With precision gating_:
```
x_modulated = relu(gamma * x_linear + beta)        # PreActivation FiLM (same)
x_bypass = relu(x_linear)                          # Unmodulated path
x_out = pi_gate * x_modulated + (1 - pi_gate) * x_bypass
```
π̂_i controls the **blending** between the modulated and unmodulated paths. When the modulator is uncertain about a channel (low π̂_i), features bypass the FiLM transform entirely.

**Injection B (Memory) — Unchanged but Informed**:

The memory gate-bias z_memory does not directly use precision. However, the shared GRU hidden state h_mod now integrates prediction error statistics (through the precision head's training signal). This means the memory injection becomes implicitly informed by observation reliability — the modulator may learn to increase memory retention (z_memory → negative, slower forgetting) when precision is low (observations unreliable → rely on memory).

**Injection C (Exploration/Reward) — Unchanged but Informed**:

Similarly, the temperature (PPO) or reward scaling (DreamerV3) is not directly gated by precision, but the shared h_mod state now carries precision information. The modulator may learn: low precision → increase temperature (explore more, since current observations are unreliable) or decrease reward scaling (be cautious about rewards in imprecise states).

### 3.4 Why the Shared GRU Is the Right Design

One might ask: should the precision head have its own separate recurrent state, independent of the modulator? The answer is no, for three reasons:

1. **Biological fidelity**: Acetylcholine (precision) is not independent of noradrenaline (arousal/exploration) and serotonin (temporal discounting). These neuromodulators interact through shared subcortical circuits. A single h_mod that drives all heads mirrors this coupling.

2. **Coherent multi-domain response (Hypothesis H4)**: The core hypothesis of the neuromodulation architecture is that perception, memory, and decision-making should respond coherently to injury — not independently. Precision estimation (knowing _which_ channels are noisy) should coordinate with memory persistence (holding threat information longer) and exploration suppression (acting cautiously). A shared GRU naturally produces this coherence.

3. **Training efficiency**: The precision head provides a strong, well-shaped gradient signal (heteroscedastic loss) that flows through h_mod to the GRU weights. This signal helps train the shared representations that other heads also use. The modulator GRU now gets gradient signal from _both_ the RL loss and the precision loss — more information for building a useful hidden state.

---

## 4. Mathematical Formulation

### 4.1 Notation

| Symbol | Definition |
|--------|-----------|
| o_t | Observation vector at time t (dim = obs_dim) |
| h_mod | Modulator GRU hidden state |
| h_task | Task GRU hidden state (PPO) or RSSM state (DreamerV3) |
| γ, β | Perceptual gain and threshold shift (from head_percept, head_percept_add) |
| z_mem | Memory gate-bias (from head_memory) |
| ô_{t+1} | Predicted next observation (from head_precision) |
| s_i = log π̂_i | Per-modality log-precision (from head_precision, 9 values) |
| π̂_i = exp(s_i) | Per-modality precision estimate |
| G | Spatial grouping size (AlKilany & Goodman) |
| M | Number of modalities (= 9) |

### 4.2 Modulator Forward Pass (PPO, Extended)

```python
# === Modulator GRU update ===
h_mod_new = GRU_mod(obs_t, h_mod_prev)

# === Existing heads (unchanged) ===
z_percept = head_percept(h_mod_new)                 # (hidden_size/G,) → broadcast to (hidden_size,)
z_percept_add = head_percept_add(h_mod_new)         # (hidden_size/G,) → broadcast (PreActivation only)
z_memory = head_memory(h_mod_new)                   # (hidden_size/G,) → broadcast to (hidden_size,)
temperature = clip(softplus(head_action(h_mod_new)) + 0.5, 0.1, 10.0)  # scalar

# === New precision head ===
precision_features = head_precision_hidden(h_mod_new)         # Linear → ReLU (hidden_size,)
o_hat = head_precision_obs(precision_features)                 # Linear (obs_dim,) — predicted next obs
log_precision = head_precision_pi(precision_features)           # Linear (9,) — per-modality log-precision

# === Compute precision gate ===
pi_gate = sigmoid(log_precision)                               # (9,) in [0, 1]
pi_gate_full = broadcast_modality_to_obs(pi_gate, breakdown)   # (obs_dim,) — broadcast per modality
pi_gate_features = broadcast_modality_to_features(pi_gate, hidden_size)  # (hidden_size,)
# Note: broadcasting from 9 modality-level precisions to feature-level
# requires a mapping from observation channels to hidden features.
# Simplest: average across modalities → single scalar. Better: per-modality
# groups in the encoder, aligned with spatial grouping (G).
```

### 4.3 Precision-Gated Injection A (PreActivation Mode)

```python
gamma = sigmoid(z_percept)          # existing gain (0, 1)
beta = z_percept_add                # existing threshold shift

x_linear = input_proj.linear(obs_t)              # pre-activation
x_modulated = activation(x_linear * gamma + beta) # PreActivation FiLM
x_bypass = activation(x_linear)                    # clean path

# Precision gating: blend modulated and bypass paths
x_out = pi_gate_features * x_modulated + (1 - pi_gate_features) * x_bypass
```

### 4.4 Precision Loss

```python
# At time t, given o_{t+1} (the actual next observation):
prediction_error = o_next - o_hat                  # (obs_dim,)

L_precision = 0.0
for i, (start, end) in enumerate(modality_slices):
    d_i = end - start                              # modality dimension
    mse_i = sum(prediction_error[start:end]**2) / d_i  # per-modality MSE
    L_precision += 0.5 * exp(log_precision[i]) * mse_i - 0.5 * log_precision[i]

# Total loss
L_total = L_RL + lambda_pred * L_precision
```

### 4.5 DreamerV3 Variant

For DreamerV3, the precision head operates on latent features rather than the modulator hidden state:

```python
# Precision head on latent state (world model decoder side)
feat = concat(deter, stoch_flat)                    # RSSM features
precision_features = precision_hidden(feat)          # Linear → SiLU
o_hat_continuous = precision_obs(precision_features)  # Linear (obs_dim,)
log_precision = precision_pi(precision_features)      # Linear (9,)

# Heteroscedastic reconstruction loss (added to L_world)
L_precision = heteroscedastic_loss(o_actual, o_hat_continuous, log_precision)
```

The two-hot reconstruction head is unmodified. The precision head is a parallel continuous-space pathway.

---

## 5. Biological Grounding

### 5.1 Mapping to Doya's Neuromodulatory Framework

Doya (2002) mapped four neuromodulators to four RL meta-parameters. Our extended architecture maps cleanly:

| Neuromodulator | Doya (2002) Role | Our Implementation | Head |
|---------------|------------------|-------------------|------|
| **Acetylcholine (ACh)** | Learning rate α | **Sensory precision** π̂_i — weights how much each observation channel contributes to belief updating | head_precision (NEW) |
| **Noradrenaline (NA)** | Exploration β | Policy temperature / exploration drive | head_action (PPO) |
| **Serotonin (5-HT)** | Temporal discount γ | Memory persistence / imagined reward scaling | head_memory / head_reward |
| **Dopamine (DA)** | TD error δ | PPO advantage signal (not modulated — computed by critic) | N/A (implicit) |

The missing piece in the current architecture was always acetylcholine — the neuromodulator specifically associated with sensory precision. The precision head fills this gap.

### 5.2 Friston's Precision as Synaptic Gain

Friston (2023, Section F.3 of NEUROMODULATION_ALGORITHM.md) identifies precision as **synaptic gain control** — modulating postsynaptic sensitivity. In our architecture:

- The modulator's γ (gain) directly implements synaptic gain on encoder features
- The precision head's π̂_i determines the _strength_ of this gain — how much the gain signal affects features
- This two-level structure (precision gates gain) mirrors the biological hierarchy: acetylcholine sets the _regime_ (how much to trust sensors), while the fast gain signal (γ, β) handles the specific _transform_ within that regime

### 5.3 Ferguson & Cardin Disinhibitory Circuit

The PreActivation modulation mode (γ * x + β) already implements the Ferguson & Cardin (2020) disinhibitory circuit:
- γ = sigmoid(z_percept) → VIP-SST disinhibitory gating (Shine's neural gain)
- β = z_percept_add → threshold shift (lower threshold = disinhibition; higher = inhibition)

Adding precision gating adds a **meta-level**: precision controls the overall _influence_ of the disinhibitory circuit. At low precision, even if the circuit says "amplify this channel," the precision gate says "but the channel is unreliable, so don't trust that amplification."

### 5.4 Costacurta's Dynamical Motifs

Costacurta et al. (2024) showed that neuromodulation can be understood as dynamically scaling rank-1 "motifs" in the recurrence matrix:

$$W_x(z(t)) = \sum_{k=1}^{K} s_k(z(t)) \cdot l_k r_k^\top$$

Our precision-gated FiLM is a special case: precision scales the _influence_ of the FiLM motif on feature processing. At high precision, the FiLM motif is fully active (γ,β have full effect). At low precision, the motif is suppressed (features pass through unmodulated).

The connection to Costacurta's LSTM-like forget gate analysis is also relevant: when precision is low, the agent should rely more on prior beliefs (memory) than current observations. The shared h_mod state, which drives both the precision head and the memory head, can learn this coordination — low π̂_i → z_memory biases toward retention.

---

## 6. Experimental Design

### 6.1 Phase 1: PPO + Precision Head (Proof of Concept)

**Goal**: Demonstrate that the precision head learns per-channel precision that correlates with environmental noise.

**Configuration**:
- Base: `neuromodulated_ppo.yaml` with `modulation.type = "PreActivation"`
- Add: `head_precision` with hidden_size → hidden_size → (obs_dim + 9)
- Loss: L_PPO + λ_pred · L_precision
- Sweep: λ_pred ∈ {0.1, 0.5, 1.0}
- MC return mode (GAE too sensitive per v7/v8)
- 128 parallel environments

**Ablation (2×2)**:

| Condition | Precision Head | Precision Gating | Expected Behavior |
|-----------|---------------|-----------------|-------------------|
| A: Baseline (current NMN, PreActivation) | No | No | ~284 survival under noise (v8-like) |
| B: Precision head, no gating | Yes (L_precision) | No | L_precision converges; π̂ correlates with noise; no survival change |
| C: Gating only, no precision training | No | Yes (random π̂) | Noise on gating → slightly worse survival |
| D: **Full: Precision head + gating** | Yes (L_precision) | Yes | π̂ correlates with noise AND survival improves |

**Key metrics**:
- `precision/log_pi_olfaction` vs injury level → should decrease with injury
- `precision/log_pi_location` vs injury level → should be constant
- `precision/prediction_error_per_modality` → olfaction errors should be highest under injury
- `survival/mean_steps` → D > A by ≥ 6 steps (290+ target)

### 6.2 Phase 2: DreamerV3 + Heteroscedastic Decoder

**Goal**: Demonstrate that precision learning improves when embedded in a generative model.

**Configuration**:
- Base: `neuromodulated_dreamer_v3.yaml` with modulator active
- Add: parallel continuous decoder + 9 precision outputs
- Loss: existing L_world + λ_π · L_precision (continuous head)
- Compare: standard DreamerV3 vs heteroscedastic DreamerV3 under noise

**Key question**: Does precision learned by the world model transfer to better actor-critic behavior during imagination?

### 6.3 Phase 3: Cross-Track Comparison & Ablation

Compare Track A (PPO + precision head) vs Track B (DreamerV3 + heteroscedastic decoder):
- Which learns per-channel precision more accurately?
- Which produces greater survival improvement under noise?
- Does the world model's richer training signal (reconstruction is core, not auxiliary) lead to better precision estimation?

### 6.4 Precision × Neuromodulation Interaction Study

The most scientifically interesting experiment: does precision interact with other neuromodulatory signals?

| Condition | Precision | Memory Mod | Temp/Reward Mod | Tests |
|-----------|----------|-----------|----------------|-------|
| Full NMN + precision | Yes | Yes | Yes | Coherent multi-domain response (H4) |
| Precision only | Yes | Disabled | Disabled | Pure sensory precision (predictive coding) |
| Memory + precision | Yes | Yes | Disabled | Precision-memory interaction |
| Full NMN, no precision | No (current) | Yes | Yes | Current v7/v8 replication |

The "full NMN + precision" condition tests Hypothesis H4: perception (precision-gated γ,β), memory (z_memory), and decision-making (temperature/z_reward) should respond **coherently** to injury — precision decreasing, memory increasing retention, and action becoming more conservative — driven by the shared h_mod state.

---

## 7. Success Criteria

| Metric | Current Best | Target | Interpretation |
|--------|-------------|--------|---------------|
| MC Survival (NoNoise) | ~357 | ≥ 355 | Precision head doesn't hurt baseline |
| MC Survival (Noise) | ~284 | **> 290** | Precision gating improves noise robustness |
| π̂_olfaction at Î=0 vs Î=1 | N/A | Significant decrease (p < 0.05) | Agent learns olfaction is unreliable under injury |
| π̂_location at Î=0 vs Î=1 | N/A | No significant change | Agent correctly identifies location as injury-invariant |
| π̂ rank order | N/A | Matches environment noise config | π̂_olfaction < π̂_visual < π̂_injury < ... < π̂_location |
| Prediction error convergence | N/A | L_precision decreasing over training | Prediction model is learning |
| Cross-correlation (γ, z_mem, π̂) | N/A | Significant after injury events | Coherent multi-domain response (H4) |

**The most important success criterion is the per-channel precision profile under injury.** If the 9-dimensional π̂ vector mirrors the environment's noise configuration — low precision for injury-scaled channels (olfaction α=2.0, visual α=3.0, injury α=1.5), stable precision for constant-noise channels (location, collision, nociception) — then precision modulation is demonstrated at the mechanistic level. This is a cleaner signal than survival improvement, which conflates many factors.

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **λ_pred too high → degrades RL performance** | Medium | Start low (0.1); monitor RL metrics independently |
| **Precision collapses to uniform** | Medium | Heteroscedastic loss regularizer prevents this; monitor π̂ variance across channels |
| **Prediction head too expressive → memorizes** | Low | Small hidden layer; L2 regularization on prediction weights |
| **Precision gating interferes with learned γ,β** | Medium | Ablation B vs D disambiguates; can disable gating if harmful |
| **Shared GRU overloaded by precision signal** | Low | GRU already handles 3-4 heads; one more is marginal. Monitor h_mod representation quality |
| **Observation-space prediction is trivial for some channels** | Low | Per-modality normalization in L_precision prevents trivial channels from dominating |
| **DreamerV3 two-hot head conflicts with continuous precision head** | Low | Separate heads, separate loss terms; two-hot unchanged |

---

## 9. Summary

| Question | Answer |
|----------|--------|
| What is the core problem? | The neuromodulator has the architecture for precision modulation (γ,β, gate-bias) but **no training signal** that drives these signals toward precision-appropriate values. v7/v8 showed identity gate collapse. |
| What is the proposed solution? | Add a **precision head** to the existing neuromodulator that predicts next observations with per-channel heteroscedastic loss. The learned precision gates the existing FiLM-style modulation. |
| What provides the training signal? | **Prediction error statistics** — channels with large, irreducible prediction errors (noisy channels) → low π̂. Channels with small, predictable errors → high π̂. This is Friston's precision dynamics, implemented as Kendall & Gal's heteroscedastic loss. |
| How does it integrate with the existing architecture? | Precision head is a new output of the shared modulator GRU. Existing heads (γ, β, z_mem, temp/z_rew) are unchanged. Precision gates Injection A (how much FiLM affects features). Shared h_mod coordinates all signals. |
| What is the biological analog? | Precision head = **acetylcholine** — the missing neuromodulator from Doya's framework. It encodes sensory precision, completing the mapping: ACh (precision) + NA (exploration) + 5-HT (temporal discount/memory) + DA (learning signal). |
| Two architectural tracks? | **Track A (PPO)**: precision head on modulator GRU, auxiliary L_precision loss. **Track B (DreamerV3)**: precision head on world model latent features, part of core L_world. Track B is theoretically superior (precision in the generative model = Active Inference). |
| Key success metric? | Per-channel precision profile mirrors environment noise config: low π̂ for injury-scaled channels, stable π̂ for constant-noise channels. This directly demonstrates learned precision modulation. |
| What about FiLM-Ensemble? | FiLM-Ensemble provides **epistemic** precision (model uncertainty via member disagreement). This proposal provides **aleatoric** precision (observation noise via prediction error). They can be combined but aleatoric is the dominant uncertainty at convergence (Kendall & Gal, 2017). |
| Key risk? | λ_pred tuning — too high hurts RL, too low gives weak precision signal. Mitigated by starting low and monitoring both RL and precision metrics independently. |
| First experiment? | PPO + precision head + precision-gated PreActivation FiLM. MC return, λ_pred={0.1, 0.5, 1.0}. 2×2 ablation: ±precision_head × ±precision_gating. Measure π̂ per channel vs injury. |
