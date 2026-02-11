# Extensive Review: Neuromodulatory Algorithms & Hypernetworks

This document provides a one-by-one detailed review of the neuromodulatory-inspired algorithms found in the `Grid World Pain` library. The focus is on identifying "simplistic" mechanisms that mimic neural gain for perceptual modulation.

## I. Direct Modulation & Gating Techniques

### 1. Neuromodulated Activations (NA)
*   **Primary Source**: Vecoven et al. (2020)
*   **Mechanism**: Instead of modifying weights, a modulatory signal (z) is used to scale the activation functions of a target network. 
*   **Mathematical Trick**: $h = \phi(Wx + b) \odot \sigma(z)$, or more simply, modifying the slope of $\phi$.
*   **Gain Implementation**: It mimics "slope modulation" in biological neurons where a neuromodulator changes the input-output mapping without changing the underlying synaptic strengths.
*   **Pros**: Computationally very cheap; avoids the O(N^2) complexity of hypernetworks.
*   **Cons**: Less expressive than weight modulation.

### 2. Multiplicative Masking
*   **Primary Source**: Ben-Iwhiwhu et al. (2022)
*   **Mechanism**: A separate "modulator" network predicts a binary or continuous mask that is element-wise multiplied with the activations or weights of sections of the main network.
*   **Mathematical Trick**: $W_{effective} = W \odot M(c)$, where $M(c)$ is the mask generated from context $c$.
*   **Gain Implementation**: Acts as a "gate" rather than a continuous "gain control," but can be adapted for continuous modulation.

---

## II. Hypernetworks Variants

### 3. Contextual Hypernetworks (CH)
*   **Primary Source**: Beck et al. (2024)
*   **Mechanism**: A hypernetwork $H$ takes a context vector $c$ (representing task ID, uncertainty, or metabolic state) and generates the weights for a smaller task network $T$.
*   **Mathematical Trick**: $\theta_T = H(c; \phi_H)$.
*   **Gain Implementation**: This is the "ultimate gain control." By changing $c$, the hypernetwork can completely reshape the "energy landscape" (the loss function) of the task network.
*   **Pros**: Extreme flexibility; supports zero-shot generalization.
*   **Cons**: Training hypernetworks is notoriously unstable and computationally expensive.

### 4. HyperLSTM (Weight Scaling)
*   **Primary Source**: Ha et al. (2016)
*   **Mechanism**: A smaller RNN (HyperRNN) predicts a vector that scales the weights of a larger LSTM at every timestep.
*   **Mathematical Trick**: Instead of generating a full weight matrix $W$, it generates a scaling vector $d$ such that $W_{active} = \text{diag}(d) W$.
*   **Gain Implementation**: This is a direct implementation of "Dynamic Gain Control" in time. 

---

## III. Emerging Structures

### 5. Hierarchical Timescale Modules
*   **Primary Source**: Ichikawa and Kaneko (2024)
*   **Mechanism**: Separating a network into modules with strictly enforced timescales (Fast vs. Slow).
*   **Gain Implementation**: The slow module acts as a "modulator" that integrates information over long periods, effectively setting the "gain" (priors) for the fast module that processes immediate sensory flux.

---

## Block I: Neuroscience Foundations (Professional Deep-Dive)

This section details the theoretical and biological mechanisms that form the "first principles" of our perceptual modulation system.

### 1. Disinhibitory Gating & Signal-to-Noise (Ferguson & Cardin, 2020)
*   **Biological Mechanism**: Cortical gain is regulated by a "Disinhibitory Circuit" involving VIP, SST, and PV interneurons. VIP cells inhibit SST cells, which normally "gate" (inhibit) the dendritic inputs of pyramidal neurons.
*   **Computational Analog**: **Non-linear Disinhibitory Gating.** Rather than a linear gain, the system uses a "switch" that suppresses background noise (SST activity) only when a high-arousal or high-salience signal (VIP activity) is present.
*   **Implementation Hook**: In our Grid World, use an "Arousal Estimator" (based on prediction error) to trigger a disinhibitory gate that suddenly reveals high-resolution sensory details when the agent is "surprised."

### 2. Energy Landscape Flattening (Shine et al., 2021)
*   **Biological Mechanism**: Ascending neuromodulatory systems (like Noradrenaline) increase neural gain, pushing the brain from a segregated/linear regime to an integrated/non-linear regime.
*   **Computational Analog**: **Manifold Destabilization.** Increasing gain steepens the activation function slope, which mathematically flattens the energy landscape of the network's attractors.
*   **Implementation Hook**: Use a dynamic gain variable $g(t)$ to "shake" the agent out of repetitive behavioral loops (local minima) by destabilizing its current neural attractor state.

### 3. Precision-Weighted Belief Updating (Friston, 2023)
*   **Biological Mechanism**: Synaptic gain is the physiological implementation of "Precision"—the inverse variance of a belief. 
*   **Computational Analog**: **Uncertainty-Weighted Updates.** Errors are weighted by their estimated reliability. 
*   **Implementation Hook**: An interoceptive module monitors observation noise. If the "fog of war" is high, lower the gain on sensory prediction errors (Sensory Attenuation), causing the agent to rely more on internal "priors" (path integration).

### 4. Meta-Parameter Control (Doya, 2002)
*   **Biological Mechanism**: Specific neuromodulators map to specific RL hyperparameters: Serotonin $\rightarrow$ Discount Factor ($\gamma$); Noradrenaline $\rightarrow$ Exploration ($\beta$); Acetylcholine $\rightarrow$ Learning Rate ($\alpha$).
*   **Computational Analog**: **Dynamic Hyperparameter Tuning.**
*   **Implementation Hook**: Implement a "Brainstem" module that monitors TD-error variance to dynamically adjust the agent's myopia ($\gamma$) and exploration ($\beta$) in real-time.

### 5. Multiscale Plasticity & BTSP (Durstewitz et al., 2025)
*   **Biological Mechanism**: Behavioral Timescale Synaptic Plasticity (BTSP). High-salience events trigger global instructive signals that overlap with recent eligibility traces.
*   **Computational Analog**: **One-Shot Path Mapping.**
*   **Implementation Hook**: Enable "Zero-Shot" learning for traps or teleporters. When a massive negative/positive reward is hit, trigger a large-magnitude weight update across the entire recent trajectory trace.
