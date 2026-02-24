# Network Structure & Observation Encoding Review

This document reviews the current network structure of the Recurrent PPO and DreamerV3 agents, specifically focusing on the observation encoding process and the hierarchical integration of multimodal inputs.

## 1. Observation Space Assembly
The observation space is defined in `src/environment/sensor.py`. It is a flat, concatenated vector of multiple modalities.

| Modality | Description | Dimension |
| :--- | :--- | :--- |
| Modality | Description | Encoding Detail |
| :--- | :--- | :--- |
| **Olfaction** | Chemosensory signal | Summed gradient of resource/predator/obstacle properties. |
| **Extero Nociception** | Phasic pain | Scalar value (0-1) representing immediate painful contact. |
| **Collision** | Tactical occupancy | Manhattan diamond binary grid (OOB or blocking rocks). |
| **Location** | Global position | Normalized (x, y) coordinates within grid bounds. |
| **Interoception** | Body state | 3D vector: [Satiation, Nutrition, Injury]. |
| **Visual** | Simplified semantic grid | N×N grid with 8 channels (semantic IDs, NOT RGB). |
| **Proprioception** | Efference copy | One-hot vector of the previous action taken. |

> [!IMPORTANT]
> **Visual Sensor Detail:** The visual sensor is **not RGB**. It is a semantic occupancy grid where each cell contains one-hot channels for object types:
> 1. Grass, 2. Sand, 3. Plain (Backgrounds)
> 4. Food, 5. Danger (Resources)
> 6. Predator, 7. Rock, 8. Neutral (Entities)

> [!NOTE]
> All modalities are flattened and concatenated in a fixed order within `sensor.get_observation`.

## 2. Recurrent PPO Encoding Structure
The Recurrent PPO agent (`ActorCriticRNN` in `src/models/recurrent_ppo_network.py`) uses a **flat fusion** approach.

```mermaid
graph TD
    Obs["Concatenated Observation (1D)"] --> InputProj["Input Projection (Linear)"]
    InputProj --> Act["ReLU Activation"]
    Act --> RNN["RNN Cell (GRU/LSTM)"]
    RNN --> Heads["Actor/Critic Heads"]
```

- **Hierarchy:** There is no hierarchical separation. All sensory inputs (vision, olfaction, interoception) are projected into the same hidden space simultaneously in the first linear layer.
- **Unimodal vs. Multimodal:** The network does not perceive individual "unimodal" streams. It treats the entire multimodal vector as a single unimodal input block.

## 3. DreamerV3 Encoding Structure
The DreamerV3 agent (`WorldModel` in `src/models/dreamer_v3_nnx.py`) follows a similar pattern but with a deeper MLP encoder.

```mermaid
graph TD
    Obs["Concatenated Observation (1D)"] --> EncoderMLP["Encoder MLP (Sequential)"]
    subgraph "Encoder Internal Structure"
        L1["Linear + LN + SiLU"]
        L2["Linear + LN + SiLU"]
        L_final["Final Linear + LN"]
    end
    EncoderMLP --> RSSM["RSSM (State Transition)"]
    RSSM --> AC["Actor/Critic MLP Heads"]
```

- **Encoder:** The `Encoder` class (`src/models/dreamer_v3_nnx.py:L142`) takes the full `obs_dim` and processes it through a standard feed-forward MLP.
- **Hierarchy:** Like PPO, DreamerV3 employs **Early Fusion**. Modalities are merged via concatenation before any feature extraction occurs.

## 4. Comparison & Analysis

### Unimodal vs. Multimodal Hierarchy
Current implementations lack an explicit hierarchical structure for multimodal processing. In many advanced RL architectures (especially those mimicking biological systems), unimodal hierarchy would involve:
1. **Unimodal Stream Processing:** Separate encoders for different modalities (e.g., a CNN for Visual, an MLP for Olfactory).
2. **Multimodal Integration:** Fusing the extracted features (Encodings) later in the network.

**Current Implementation Status:**
- **Recurrent PPO:** Minimal hierarchy. Single-layer linear projection followed by RNN.
- **DreamerV3:** Deeper encoding hierarchy (MLP), but still applies this hierarchy to the *entire* concatenated vector, not to individual modalities separately.

### Neuromodulation Injection Points
When enabled, both networks support neuromodulation (via `NeuromodulatorRNN` or `DreamerNeuromodulatorRNN`). These modules *do* act on the full observation, but their influence is injected at specific points:
- **Injection A (Perceptual):** Modulates the input projection (PPO) or encoder output (Dreamer).
- **Injection B (Memory):** Modulates internal RNN/GRU gates.
- **Injection C (Action):** Modulates output temperature (PPO only).

## 5. Proposed Hierarchical Integration (Brain-Inspired)

A biological brain does not fuse all signals at once. Information is processed through unimodal hierarchies and then merged at various stages (e.g., thalamic relay, association cortices).

### Conceptual Architecture

```mermaid
graph TD
    subgraph "Unimodal Feature Extraction (Lower Hierarchy)"
        V_raw["Visual Semantic Grid"] --> V_Enc["Visual Cortex (MLP)"]
        O_raw["Olfaction Gradient"] --> O_Enc["Olfactory Bulb (MLP)"]
        I_raw["Interoception (3D)"] --> I_Enc["Intero Encoder (MLP)"]
        EN_raw["Extero Nocicep (1D)"] --> EN_Enc["Nocicep Encoder (MLP)"]
        C_raw["Collision (Binary)"] --> C_Enc["Touch Encoder (MLP)"]
    end
    
    subgraph "Intermediate Body-State Integration"
        I_Enc --> B_Hub["Body-State Hub (MLP)"]
        EN_Enc --> B_Hub
        C_Enc --> B_Hub
    end

    subgraph "Multimodal Association (Higher Hierarchy)"
        V_Enc --> Assoc["Association Hub (MLP)"]
        O_Enc --> Assoc
        B_Hub --> Assoc
        P_raw["Proprioception (Previous Action)"] --> Assoc
        Loc_raw["Global Location"] --> Assoc
    end
    
    Assoc --> Core["RNN / RSSM Core"]
    Core --> Heads["Action/Value Heads"]
```

### 1. Implementation Strategy: Modality-Specific Encoders
Each modality will have its own dedicated encoder network. This allows for:
- **Spatial processing** for Visual and Collision sensors.
- **Signal compression** for high-dimensional vectors.
- **Independent capacity tuning** based on the evolutionary importance of the sensor.

> [!WARNING]
> **Implementation Constraint: Sensor Ablations**
> The current system allows for individual sensors to be toggled `on` or `off` via YAML configurations (e.g., `olfactory_enabled: false`). Any hierarchical implementation **MUST** be robust to these ablations:
> - The observation decomposition must dynamically skip indices of disabled sensors.
> - The integration hubs (Body-State Hub, Association Hub) must handle varying input dimensions (e.g., by using zero-padding for disabled streams or dynamically reconstructing the fusion MLP).
> - The code should not assume any sensor is always present.

### 2. Hierarchical Fusion Stages
Instead of a single concatenation, integration will happen in waves:
- **Phase 1 (Unimodal):** EVERY sensor modality (including each component of body state) $\to$ Individual specialized features via dedicated networks.
- **Phase 2 (Sub-Fusion):** Grouping related modalities. For example, merging the outputs of the Intero, Nocicep, and Touch encoders into a unified "Body-State Hub" representation (analogous to the somatosensory cortex/insula).
- **Phase 3 (Global Fusion):** Merging semantic environmental mappings (Visual, Olfactory) with the integrated body state and global spatial context (Location, Proprioception).

### 3. Comparison: Flat Fusion vs. Hierarchical Integration

| Feature | Current (Flat Fusion) | Proposed (Hierarchical Integration) |
| :--- | :--- | :--- |
| **Logic** | Concatenate all $\to$ Shared MLP. | (Unimodal MLP) $\times$ N $\to$ Sub-Fusions $\to$ Global MLP. |
| **Representational Power** | Generalist, but prone to "small signal wash-out". | Specialist, preserves small but critical signals (e.g., Pain). |
| **Modularity** | Rigid; changing one sensor requires retuning all. | High; unimodal encoders can be tuned/swapped independently. |
| **Interpretability** | Black-box hidden state. | Inspectable intermediate states (e.g., "Body State"). |
| **Compute Cost** | Lower (single projection). | Slightly Higher (multiple smaller networks). |

## 6. Configuration Schema Proposal

To maintain the project's goal of "simple yet flexible," the following YAML structure is proposed. This allows for a global default while permitting per-modality overrides.

### Proposed YAML Structure (`configs/models/*.yaml`)

```yaml
agent:
  # Toggle between "flat" and "hierarchical"
  encoding_mode: "hierarchical" 
  
  # GLOBAL DEFAULT: used for any encoder/hub not explicitly defined
  default_encoding_structure: [128, 128]
  
  # UNIMODAL OVERRIDES: (Optional)
  # Specify custom MLP structures for specific sensors
  unimodal_encoders:
    visual: [256, 256]    # Visual needs more capacity
    olfaction: [64]       # Olfaction is simple
    # intero, nocicep, touch, location, proprio automatically use 'default_encoding_structure'
    
  # MULTIMODAL HUB OVERRIDES: (Optional)
  multimodal_hubs:
    body_state: [128]     # Sub-fusion of intero/nocicep/touch
    association: [128]    # Final integration point
```

### 7. Planning Considerations for Implementation

#### DreamerV3 Adaptation
- Move from a monolithic `Encoder` class to a `HierarchicalEncoder` that contains sub-layer classes (`VisualEncoder`, `SomatoEncoder`, etc.).
- The `RSSM` deter-dim will act as the final multimodal integration point.

## 8. JAX-Specific Implementation & Potential Issues

The transition to hierarchical encoding in a JAX/Flax NNX environment introduces specific technical risks that must be managed.

### 1. JIT Compilation vs. Dynamic Shapes
JAX requires static shapes for `jit`-compiled functions.
- **Risk:** If the encoding hierarchy is reconstructed dynamically based on the observation vector size, it may trigger frequent re-compilations if the configuration changes.
- **Mitigation:** The architecture must be **static during the training/inference lifetime**. The configuration should be parsed once at initialization, and all sub-encoders must be created then. For ablations, either zero-pad inputs or use static "identity" networks for disabled paths.

### 2. Flattening and State Management (Flax NNX)
In Flax NNX, models are objects that store their own state.
- **Risk:** A deep hierarchy of sub-networks (`VisualEncoder`, `OlfactiveEncoder`, etc.) can make state tracking and gradient updates more complex if not properly registered in the parent `Module`.
- **Mitigation:** Ensure all sub-encoders are correctly assigned as attributes of the parent `Encoder` or `FeatureExtractor` class so that `nnx.split()` and `nnx.merge()` can see their parameters.

### 3. Vmap Compatibility
All encoding paths must support batching via `vmap`.
- **Risk:** Custom logic for "slicing" observations must be carefully implemented using JAX-native operations (like `jnp.split` or `jnp.take`) rather than Python loops or slicing that might break the vectorization of the first dimension.

### 4. Dimensionality "Wash-out" in Concatenation
- **Issue:** Even with hierarchy, the final `Association Hub` merges vectors of different sizes. If a compressed visual latent is 256 and injury is 1, the network might still ignore the 1.
- **Solution:** Use **balanced latent sizes** (e.g., all unimodal encoders output a 64D or 128D vector) or implement **weighted fusion** (Gain modulation) to ensure critical body-state signals have sufficient "volume" in the final latent.
### 6. Potential Training Speed (SPS) Decrease
Moving from a single linear layer to multiple specialized MLPs will impact performance:
- **Increased FLOPs:** More hidden layers across multiple encoders naturally increase computation.
- **Kernel Overhead:** JAX/XLA is highly optimized for large matrix multiplications. Splitting one large Matmul (Flat Fusion) into 7+ smaller Matmuls (Hierarchical) can reduce arithmetic intensity and increase kernel launch overhead.
- **Expected Impact:** For small hidden sizes (like `[128, 128]`), the SPS decrease is likely marginal (5-10%). However, if individual encoders are large, the cumulative overhead could become significant.
- **Mitigation:** Use **XLA-friendly structures**. Ensure that all sub-encoders are compiled into a single fused HLO graph during JIT.

### 5. Config-Driven Network Instantiation
- **Issue:** Creating MLPs dynamically based on a YAML list (e.g., `[256, 128]`) requires careful looping in `__init__`.
- **Constraint:** Use JAX-ready Flax NNX layers. Avoid using Python logic that depends on the *data value* within the forward pass; only use logic that depends on the *graph structure* defined in `__init__`.
