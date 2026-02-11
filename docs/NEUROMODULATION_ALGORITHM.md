# Extensive Review: Neuromodulatory Algorithms & Hypernetworks

> [!IMPORTANT]
> **Review Protocol & Objectives**
> 1. **Systematic Cataloging**: Review all 31 sources from the NotebookLM library.
> 2. **Categorization**: Group sources into 6 blocks (Foundations, Hypernetworks, Gating, Dynamics, Predictive Coding, Specialized).
> 3. **Deep-Dive Criteria**: For each paper, define:
>    - **Biological Mechanism**: The physical neural architecture.
>    - **Exhaustive Mathematical Formalism**: Full update rules, gain equations, and state-space derivations.
>    - **Project Insight**: Concrete implementation hooks for the grid-world RL agent.
> 4. **Additive Updates**: Preserve all previous entries; updates must be additive.
> 5. **Synthesis**: Final goal is a concrete architectural proposal for a "Simplistic Perceptual Modulation" system.

This document provides a one-by-one detailed review of the neuromodulatory-inspired algorithms found in the `Grid World Pain` library.

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

## Block I: Neuroscience Foundations (Exhaustive Technical Review)

This section details the theoretical and biological mechanisms that form the "first principles" of our perceptual modulation system. 

### 1. Disinhibitory Gating & Signal-to-Noise (Ferguson & Cardin, 2020)
*   **Biological Mechanism**: Cortical gain is regulated by a specific **Disinhibitory Circuit**. By default, SST interneurons inhibit the distal dendrites of pyramidal neurons. VIP interneurons inhibit SST cells.
*   **Mathematical Formalism**:
    The activity of a pyramidal neuron $r_P$ can be modeled as:
    $$r_P = f\left( \sum w_{syn} r_{in} - [I_{SST} - I_{VIP}]_+ \right)$$
    Where $I_{SST}$ is the gating signal and $I_{VIP}$ is the "un-gating" signal. When $I_{VIP} > I_{SST}$, the inhibitory gate is lifted, allowing sensory input to propagate.
*   **Project Insight**: Implement a hierarchical observation encoder where primary sensory input is gated by an SST-analog. High surprisal activates a VIP-analog to lift the gate.

### 2. Energy Landscape Flattening (Shine et al., 2021)
*   **Biological Mechanism**: Noradrenaline increases neural gain $g$, shifting the brain into an integrated regime.
*   **Mathematical Formalism**:
    In a gradient-flow system $\dot{x} = -\nabla E(x)$, the gain $g$ of the activation function effectively rescales the energy landscape $E$. 
    For a change in gain $\Delta g$, the stability of an attractor (eigenvalue $\lambda$) scales as:
    $$\lambda_{new} = \frac{\lambda_{old}}{g^2}$$
    Large $g$ flattens the potential wells, facilitating transitions between meta-stable states (behavioral modes).
*   **Project Insight**: Use a dynamic gain variable $g(t)$ to "shake" the agent out of local minima (habitual loops) by destabilizing its current neural attractor.

### 3. Precision-Weighted Belief Updating (Friston, 2023)
*   **Biological Mechanism**: Synaptic gain is the physiological implementation of **Precision** (inverse variance $\Pi = \sigma^{-2}$).
*   **Mathematical Formalism**:
    The update of an internal belief $\mu$ follows:
    $$\dot{\mu} = D\mu - \epsilon_p \Pi_p \frac{\partial \epsilon_p}{\partial \mu} - \epsilon_s \Pi_s \frac{\partial \epsilon_s}{\partial \mu}$$
    Where $\epsilon_s$ is the sensory prediction error and $\Pi_s$ is the precision. High sensory noise forces $\Pi_s \to 0$, making the agent rely exclusively on its "internal priors" $D\mu$.
*   **Project Insight**: An Interoceptive Module monitors observation noise; if the "Fog of War" is high, it lowers $\Pi_s$ (Sensory Attenuation), causing the agent to ignore unreliable data.

### 4. Meta-Parameter Control (Doya, 2002)
*   **Mathematical Formalism**:
    The RL policy $\pi(a|s)$ and value update $\Delta V$ are modulated via:
    - **Softmax Exploration**: $\pi(a|s) = \text{softmax}(\beta(t) Q(s,a))$
    - **Temporal Discount**: $\gamma(t)$ in $G_t = \sum \gamma^k r_{t+k}$
    - **Learning Rate**: $\alpha(t)$ in $\theta_{t+1} = \theta_t + \alpha(t) \delta_t$
    Where $\beta, \gamma, \alpha = H(\text{Statistics of } \delta)$.
*   **Project Insight**: If TD-error variance is high, inhibit the Serotonin variable to make the agent "myopic" (low $\gamma$), focusing on immediate survival.

### 5. Multiscale Plasticity & Behavioral Timescale (Durstewitz et al., 2025)
*   **Biological Mechanism**: Behavioral Timescale Synaptic Plasticity (BTSP). High-salience events trigger global instructive signals that bridge long temporal gaps (seconds) between actions and outcomes.
*   **Computational Analog**: **One-Shot eligibility traces.**
*   **Extended Project Insight**: 
    *   **Implementation**: Augment standard backprop with a BTSP "One-Shot" trigger.
    *   **Logic**: Maintain a decaying eligibility trace of the agent's recent path. If a massive reward or "pain" event is hit, trigger a global instructive signal that overlaps with the trace, performing a large-magnitude weight update to encode that specific path as a "hypertube" in a single trial.

---

## Block II: Hypernetwork & Dynamic Architectures (Exhaustive Technical Review)

This section explores "Meta-Architectures" where a secondary network modulates the parameters or connectivity of a primary task network, effectively decoupling *regulation* from *execution*.

### 6. Weight Manifold Mapping (HyperZero; Rezaei-Shoshtari et al., 2023)
*   **Mathematical Formalism**:
    The hypernetwork $H$ represents a mapping between two Riemannian manifolds: the environmental parameter space $\mathcal{M}_\psi$ and the optimal policy weight space $\mathcal{M}_\theta$.
    $$\theta^* = \text{arg min}_{\theta \in \mathcal{M}_\theta} \mathcal{L}(\theta; \psi)$$
    $$\theta_{generated} = H(\psi; \phi_H) \approx \theta^*(\psi)$$
*   **Project Insight**: If interoceptive sensors detect a "high motor noise" state $\psi_{noise}$, the hypernetwork instantly shifts the policy $\theta$ to a regime robust to that specific noise distribution.

### 7. Modular Expressivity (Galanti & Wolf, 2020)
*   **Mathematical Formalism**:
    Hypernetworks $H(z)$ are proven to be more expressive than embedding methods $f(x, z)$. Specifically, for any target function $y = g(x; z)$:
    $$\exists H \text{ such that } ||H(z)(x) - g(x; z)|| < \epsilon$$
    Standard networks with context-concatenation require exponentially more wide layers to achieve the same approximation for non-linear modulations.
*   **Project Insight**: Use a hyper-component to generate the weights of the sensory pre-processor. This allows the "Pain" signal to fundamentally re-wire how the grid is perceived.

### 8. Dynamic Transition Modulation (Jiang et al., 2021)
*   **Mathematical Formalism**:
    In a state-space model $x_t = A_t x_{t-1} + v_t$, the transition matrix $A_t$ is generated by a hypernetwork $H$:
    $$A_t = \text{reshape}\left( H(h_{t-1}; \phi) \right)$$
    Where $h_{t-1}$ is the high-level context or hidden state.
*   **Project Insight**: Modulate temporal expectations. Moving from "Deterministic Field" to "Stochastic Hallway" triggers the hypernetwork to re-generate $A_t$ with higher variance parameters.

### 9. Bayesian Posterior Generation (Borycki et al., 2022)
*   **Mathematical Formalism**:
    The hypernetwork outputs the parameters of a variational distribution $q(\theta|c)$:
    $$\mu, \Sigma = H(c; \phi)$$
    $$\theta \sim \mathcal{N}(\mu, \Sigma)$$
    The loss includes a KL-divergence term $D_{KL}(q(\theta|c) || p(\theta))$ to enforce uncertainty calibration.

---

## Block III: Gating & Direct Modulation (Exhaustive Technical Review)

This section focuses on computationally efficient scaling and gating of features, primarily via gain-control on activations or low-rank modifications.

### 10. Neuromodulated Networks (NMN; Vecoven et al., 2020)
*   **Biological Mechanism**: Mimics gain-control interneurons that adjust the slope of the input-output function of target neurons.
*   **Mathematical Formalism**:
    The activation of a neuron $h$ is multiplicatively gated by a signal $z$ from a parallel modulator network $G$:
    $$h = \sigma(z \odot (Wx + b))$$
    Where $z = G(c; \phi_G)$.
    Crucially, if $\sigma$ is a ReLU or Sigmoid, $z$ acts as a **Slope Modulator**:
    $$\frac{\partial h}{\partial (Wx+b)} \propto z$$
*   **Project Insight**: Use this for **Behavioral Mode Switching**. By adjusting only $z$, the agent can transition from "Pure Exploitation" to "High-Variance Exploration" without changing any core weights.

### 11. Structured Flexibility & Low-Rank Gating (Costacurta et al., 2024)
*   **Biological Mechanism**: Neuromodulators target specific "dynamical motifs" within a network, such as integrators or oscillators.
*   **Mathematical Formalism**:
    The recurrent weight matrix $W$ is decomposed into a static base and a modulated low-rank component:
    $W(t) = W_{base} + \sum_{k} s_k(t) \cdot (\mathbf{u}_k \mathbf{v}_k^T)$
    Where $s_k(t)$ is the modulatory gain for the $k$-th dynamical motif (rank-1).
*   **Project Insight**: **Programmable Memory Windows.** Scaling $s_k(t)$ can change the decay constant of the RNN's hidden state, effectively allowing the agent to "choose" how long to remember a specific sensory cue (e.g., 5s for a short-term trap vs. 60s for a goal location).

### 12. Biophysical Parameter Control & Spatial Grouping (AlKilany & Goodman, 2025)
*   **Biological Mechanism**: Rapid adjustment of threshold $v_{th}$ and membrane time constants $\tau_m$ in SNNs, via a separate Modulator Network that outputs dynamic biophysical adjustments.
*   **Mathematical Formalism (LIF Dynamics)**:
    $$\tau_m(t) \frac{dv}{dt} = -(v - v_{rest}) + R \cdot I(t)$$
    $$\text{Spike if } v(t) \geq v_{th}(t)$$
    Where $\tau_m(t)$ and $v_{th}(t)$ are dynamic outputs of a controller.

#### Spatial Grouping (Dimensionality Reduction for Modulation)
A key contribution is **Spatial Grouping**: instead of the Modulator producing a unique signal per neuron (G=1), neurons in the primary SNN are partitioned into sequential groups of size $G$. The modulator output dimension is reduced from $N_{hidden}$ to $N_{hidden}/G$.

*   **Additive Modulation (Preserving Heterogeneity)**:
    $$\Psi_i(t+1) = \text{Clip}\left(\Psi_i(0) + m_{\lfloor i/G \rfloor}(t)\right)$$
    Where:
    -   $\Psi_i(0)$ is the **initial, independently learned baseline** parameter for neuron $i$ (unique per neuron).
    -   $m_{\lfloor i/G \rfloor}(t)$ is the dynamic adjustment from the modulator for the group containing neuron $i$.
    -   $\text{Clip}(\cdot)$ bounds the parameter to biologically valid ranges.

    This is critically distinct from **Substitution** ($\Psi_i(t) = m_{\lfloor i/G \rfloor}(t)$), which would collapse all neurons in a group to identical states. Additive modulation preserves neuronal diversity through the unique baselines while correlating their *dynamics* within a group.

*   **Granularity Spectrum**:
    | Regime | $G$ Value | Modulator Output | Analogy |
    |---|---|---|---|
    | Fine-Grained | $G = 1$ | $N_{hidden}$ | Synaptic precision |
    | Spatial Grouping | $1 < G < N$ | $N_{hidden}/G$ | Volume transmission (mesoscale) |
    | Global | $G = N$ | $1$ | Uniform neuromodulatory bath |

*   **Key Empirical Finding**: Spatially extended modulation ($G=10$ or $G=20$) was **equally effective** as fine-grained modulation ($G=1$) across all tested tasks. This suggests that the computational benefit of neuromodulation comes from regulating the **macroscopic regime** of the network (global excitability, integration windows) rather than micromanaging individual neurons.

*   **Project Insight**: Implementation of **Signal-to-Noise Pumping**. In a high-noise environment (e.g., Grid World "Storm"), the agent can increase $v_{th}$ to "filter" low-intensity sensory noise. The spatial grouping finding is particularly relevant for our project: our Modulator's $z_{percept}$ head can output a **low-dimensional** vector (e.g., one gain per sensory modality group rather than per feature) without losing expressiveness.

---

## Block IV: Temporal Dynamics & RNNs (Exhaustive Technical Review)

This section focuses on the geometry of neural trajectories (manifolds) and how "gain" moves these trajectories in state space.

### 13. Activity Hypertube Shifting (Tsuda et al., 2021)
*   **Biological Mechanism**: Neuromodulators (e.g., Dopamine) globalize or localize the "attraction" of synaptic weight configurations.
*   **Mathematical Formalism**:
    Synaptic weights are globally scaled by a factor $f$: $W_{eff} = f \cdot W_{static}$.
    This scales the overall "speed" of the flow field $\dot{x} = F(x, W_{eff})$.
    Topologically, this shifts the entire neural trajectory manifold into a distinct, non-overlapping **Hypertube** in state space.
*   **Project Insight**: **Multi-Task Storage**. A single fixed weight matrix $W$ can store "Approach" and "Avoid" behaviors in separate hypertubes. The agent "interpolates" between behaviors simply by varying $f$.

### 14. Phasic Gain Bursts & Landscape Reset (Wainstein et al., 2025)
*   **Mathematical Formalism**:
    Phasic bursts of Noradrenaline induce a transient increase in gain $g$. 
    The "Stability" $\lambda$ of a neural belief (attractor) is inversely proportional to $g$: 
    $$\lambda_{barrier} \propto \frac{1}{g}$$
    High $g$ effectively "liquifies" the current belief state.
*   **Project Insight**: Use this for **Perceptual Switching**. When prediction error is massive, trigger a gain burst to "liquify" the old map interpretation so a new, correct one can form instantly.

### 15. Lifelong RL & Uncertainty Arbitration (Lee et al., 2024 / Tschantz et al., 2023)
*   **Mathematical Formalism**:
    Arbitrating between Fast (Amortized) and Slow (Iterative) inference:
    $$\mu_{final} = w_{fast} \mu_{fast} + w_{slow} \mu_{slow}$$
    Where weights $w$ are modulated by estimated **Aleatoric** (noise) and **Epistemic** (novelty) uncertainty.
*   **Project Insight**: In familiar rooms, the agent uses "Fast Gist" (low energy). In novel rooms, high uncertainty triggers "Slow Iterative" processing for higher accuracy.

---

## Block V: Predictive Coding & Uncertainty (Exhaustive Technical Review)

This section details how the brain (and our agent) manages the trade-off between internal expectations (priors) and external sensory data (likelihood) using precision-weighting.

### 20. Precision-Weighted Active Inference (Friston et al., 2023)
*   **Core Invention**: Using synaptic gain to represent the **Precision** (reliability) of sensory prediction errors.
*   **Mathematical Formalism**:
    The update rule for the internal state $\mu$ minimizes Variational Free Energy $\mathcal{F}$:
    $$\dot{\mu} = D\mu - \Pi_p \epsilon_p \frac{\partial \epsilon_p}{\partial \mu} - \Pi_s \epsilon_s \frac{\partial \epsilon_s}{\partial \mu}$$
    Where $\Pi_s$ is the **Sensory Precision Matrix**. 
    Neuromodulators (like Acetylcholine) effectively multiply $\Pi_s$, determining how much the agent "listens" to the world vs. its internal model $D\mu$.
*   **Project Insight**: **Dynamic SNR Control.** If the interoceptive sensor detects "high sensory noise," it should inhibit $\Pi_s$, forcing the agent to rely on its internal map (recurrent state) until reaching a high-precision landmark.

### 21. Hybrid Inference & Arbitration (Tschantz et al., 2023)
*   **Core Invention**: Arbitration between a fast feedforward "Reflex" and a slow iterative "Reflection" system based on surprisal.
*   **Mathematical Formalism**:
    1.  **Fast (Amortized)**: $z_{fast} = q_\phi(x)$.
    2.  **Slow (Iterative)**: $z_{t+1} = z_t - \eta \nabla_{z_t} \mathcal{F}(x, z_t)$.
    3.  **Arbitration Weight**: $w = \sigma(\mathcal{F}(x, z_{fast}) - \tau)$.
    Where $\mathcal{F}$ is the Reconstruction Error (Surprisal) and $\tau$ is the "Hesitation Threshold."
*   **Project Insight**: Implementation of **Biological Hesitation**. When the agent encounters a new trap pattern, the reconstruction error $\mathcal{F}$ spikes, $w \to 1$, and the agent "freezes" for a few ticks to iteratively converge on a correct perception before moving.

### 22. Gain-Modulated Perception & Bifurcation (Rodriguez-Garcia et al., 2026)
*   **Mathematical Formalism**:
    Bistable perception modeled via a competitive attractor network with gain $g$:
    $$\tau \dot{x}_i = -x_i + f(g \cdot x_i - \sum \beta_{ij} x_j + I_i)$$
    Increasing $g$ pushes the system through a **Pitchfork Bifurcation**, forcing the state to snap into one attractor (percept) and increasing the stability (depth) of that percept.
*   **Project Insight**: Use gain to "lock in" a perceptual decision. Once a doorway is identified with 90% confidence, spike the gain to prevent sensor jitter from causing the agent to "forget" the door's presence.

---

## Block VI: Specialized Applications (Exhaustive Technical Review)

This section covers domain-specific implementations that translate modulatory principles into concrete algorithmic "tricks" for robotics, vision, and navigation.

### 16. Zero-Shot Physics Transfer (HyperZero; Rezaei-Shoshtari et al., 2023)
*   **Biological Mechanism**: Meta-adaptation to internal and external physical states (e.g., metabolic fatigue or ground friction).
*   **Mathematical Formalism**:
    The hypernetwork $H$ maps a physics context vector $\psi$ to the policy manifold $\mathcal{M}_\theta$:
    $$\theta_{wet\_floor} = H(\psi=[0.5, \dots]; \phi)$$
    The policy is optimized via meta-gradients across a distribution of physics parameters $p(\psi)$.
*   **Project Insight**: Zero-shot robustness. If the agent moves from a deterministic to a stochastic hallway, the "Physics Monitor" updates $\psi$, and the hypernetwork instantly re-instantiates weights optimized for high-noise control.

### 17. Programmable Vision & FiLM (Perez et al., 2018)
*   **Mathematical Formalism**:
    Feature-wise Linear Modulation (FiLM) layers are inserted into the CNN encoder:
    $$FiLM(x; \gamma, \beta) = \gamma(c) \odot x + \beta(c)$$
    Where $\gamma, \beta$ are scaling and shifting vectors generated from a task-context $c$.
*   **Project Insight**: **Feature Sensitization.** If the agent is told "Avoid Red Lava," the modulator can generate a negative $\gamma$ for the red channel, effectively "blinding" the policy to irrelevant red features or treating them as maximal-intensity inhibitory signals.

### 18. Arousal-Modulated Hopfield Networks (Osman et al., 2024)
*   **Biological Mechanism**: Trust-arbitration between internal memory and external sensory drive.
*   **Mathematical Formalism**:
    The weight of recurrent memory $W_{rec}$ is suppressed by a global arousal gain $\alpha$:
    $$h_t = \sigma\left( \frac{W_{rec}}{\alpha} h_{t-1} + W_{in} x_t \right)$$
    As $\alpha \to \infty$, the system enters a "paramagnetic" state where state is driven purely by $x_t$, erasing prior beliefs.
*   **Project Insight**: **Network Reset.** Use high absolute prediction error (surprisal) to spike $\alpha$, liquifying the agent's "hallucinated map" of a familiar room when it has actually teleported to a new level.

### 19. Ranking-Based Feature Competition (Tambaş et al., 2025)
*   **Mathematical Formalism**:
    Anti-Hebbian competition for hidden representation:
    1.  Rank hidden units by activation: $r_1, r_2, \dots, r_N$.
    2.  Top-1 ($\mu_1$): $\Delta w \propto \eta \cdot (x - w)\mu_1$ (Hebbian).
    3.  Runners-up (2 to K): $\Delta w \propto -\eta \cdot (x - w)\mu_k$ (Anti-Hebbian).
*   **Project Insight**: Ensures non-overlapping, disentangled features (e.g., "Key" vs "Door") without needing explicit neuron-overlap, increasing robustness to catastrophic interference.

---

---

# Implementation Discussion: Critique of Simplistic Gating (Vecoven/Ben-Iwhiwhu Style)

Based on our recent review, here is the critical feedback regarding the implementation of the "Simplistic" neuromodulation style in the current project environment.

## 1. The "Identity Collapse" Concern (Context vs. Sensory)
In the Vecoven style, the Modulator $G(c)$ outputs a gain $z$. The primary risk is that if $c \equiv x$ (sensory input), the modulation becomes a simple feature extractor. 

### Integrated Observation (Full Observability per $c$ and $x$)
We acknowledge that in a practical RL implementation, **$c$ and $x$ can both be derived from the full observation set**. The model is expected to autonomously learn to extract regulatory features (for $c$) and task-specific features (for $x$). Furthermore, ablation experiments can be conducted to determine which specific combinations of information (e.g., visual vs. interoceptive) are sufficient to drive effective perceptual modulation and adaptive behavior.

The "Identity Collapse" is prevented not by *restricting* inputs, but by **Architectural Bottlenecking**:
*   **Context $c$ (The Modulator Path)**: Processes the full observation (Visual + Interoceptive + Proprioceptive) through a high-compression bottleneck (e.g., a small MLP or low-rank layer) to extract slowly-varying metabolic or environmental states.
*   **Sensory $x$ (The Task Path)**: Processes the same observation through the high-capacity feature encoder (MLP or RNN).

By forcing the Modulator to output a global coordination signal $z$ derived from all available information, we allow the agent to learn complex cross-modal modulations (e.g., "Mute Visual Noise because Interoceptive Pain is rising") without manual input partitioning.

### Branched Modulatory Targets (The Proposed Solution)
To resolve this while embracing the multifunctional role of neuromodulators (Doya, 2002), we propose a **Branched Modulator Head**. A single neuromodulatory network $G$ processes context $c$ and emits a high-dimensional vector $\mathbf{z}$, which is partitioned to control distinct functional blocks:

1.  **$z_{percept}$ (Early-stage Gating)**: Multiplicatively scales the input layer or first MLP hidden layer activations. This allows the model to "shut off" or "sensitize" specific input features (e.g., ignoring noisy sensory dimensions).
2.  **$z_{memory}$ (RNN Recurrence)**: Modulates the hidden-state decay or the 'forget gate' of the recurrent module (e.g., GRU or LSTM).
3.  **$z_{action}$ (Policy Temperature)**: Modulates the actor/critic heads, effectively controlling action entropy or value-certainty without changing the underlying percept.
4.  **$z_{reward}$ (Intrinsic Interpretation)**: Modulates the reward predictor, adjusting how "painfully" the agent interprets external reward signals or nociceptive inputs.

By having different modulatory targets for different outputs, we preserve the structural identity of the "Neuromodulator" as a global coordinator that reconciles internal states (Doya's metalevels) with specialized task subunits.

## 2. Gradient Sparsity (The "Shut-Off" Problem)
Multiplicative gating $z \odot h(x)$ is powerful but dangerous during early learning. If the modulator network predicts $z \approx 0$ for a specific sensory channel (e.g., "Ignore Blue Cells"), the task network's weights for that channel receive **zero gradient**.
*   **Potential Solution**: Implement a "Leaky Gate" (e.g., $z_{eff} = 0.1 + 0.9z$) to ensure base learning continues even during sensory suppression.

## 3. Perception vs. Action Arbitration
The Ben-Iwhiwhu model often applies modulation at the **hidden layers** of the policy. For a true study of "Perceptual Modulation," we should strictly separate functional layers:
*   **Encoder Modulation**: "I see the wall differently" (Strictly Perceptual).
*   **Policy Modulation**: "I react to the wall differently" (Strictly Behavioral).
*   **Recommendation**: Apply Vecoven gates early in the MLP encoder or input projection layer to validate perceptual claims.

## 4. The Recurrent Modulator (RNN-style Modulatory Path)
Instead of a feedforward $G(c)$, we propose making the Modulator network itself **Recurrent** (GRU or LSTM). This provides several critical advantages for studying perceptual modulation:

*   **Endogenous Affective State**: Biological neuromodulation (e.g., a persistent state of arousal or anxiety) doesn't just "switch off" the moment a stimulus vanishes; it decays slowly. An RNN modulator can maintain a "mood" across several time steps, sensitizing the agent to noise even after a painful event has ended.
*   **Timescale Separation**: We can design the Modulator-RNN to have a high recurrent "inertia" (slower decay) compared to the Task-RNN. This represents the distinction between **State** (slow, affective) and **Computation** (fast, reactive).
*   **Integration of History**: An RNN modulator can integrate a "History of Pain" to trigger a transition into a "Chronic Survival Mode," which a feedforward network would struggle to represent without manual feature engineering.

### Design Consideration: Latent Modification
With two coupled RNNs (Modulator and Task), we must decide if the modulation is:
1.  **Multiplicative (Gate)**: $h_{task} = \sigma(z_{mod}) \odot f(x)$. (Stable, prevents drift).
2.  **Additive (Bias)**: $h_{task} = f(x) + z_{mod}$. (Bio-inspired "Landscape Shift", more expressive but prone to instability).

---

# Implementation Details: Prototype Design for Grid-World RL

This section outlines the concrete software architecture for the neuromodulation prototype, derived from the actual codebase.

## 0. Current Observation Space (from `sensor.py`)
The observation is a **flat 1D vector** (not an image). Its composition is:

| Modality | Sensor Function | Dim | Description |
|---|---|---|---|
| Olfaction | `sense_resource()` | `property_vec_size` | Chemical gradient sum from Resources + Predators + Obstacles + Neutrals |
| Extero Nociception | `sense_extero_nociception()` | 1 | Max pain intensity from contact (Danger, Predator, Rock) |
| Collision | `sense_collision()` | $2r^2+2r+1$ | Manhattan-diamond binary map of blocking obstacles |
| Location | `sense_location()` | 2 | Normalized agent $(row, col)$ in $[-1, 1]$ |
| Satiation | (interoceptive) | 1 | $satiation / max\_satiation$ |
| Nutrition | (interoceptive) | 1 | $nutrition / max\_nutrition$ |
| Injury | (interoceptive) | 1 | $injury / max\_injury$ |
| Visual | `sense_visual()` | $N_{cells} \times 8$ | One-hot grid: [Grass, Sand, Plain, Food, Danger, Predator, Rock, Neutral] |
| Proprioception | (one-hot) | `action_dim` | One-hot encoding of previous action |

Total observation dim is variable based on config (sensor ranges, enabled flags). **Perceptual noise** (`apply_perceptual_noise`) adds state-dependent Gaussian noise per modality, scaled by injury level.

## 1. Current Agent Architectures

### A. Recurrent PPO (`ActorCriticRNN` in `recurrent_ppo_network.py`)
A single-file, compact architecture using Flax NNX:

```
obs (flat vector, dim=input_dim)
  │
  ├─► input_proj: Linear(input_dim → hidden_size) + ReLU
  │       │
  │       ▼
  │   rnn_cell: GRUCell(hidden_size → hidden_size) or LSTMCell
  │       │       ◄── h_prev (carry state)
  │       │
  │       ▼  x_h (RNN output)
  │       ├─► actor_fc1: Linear(hidden → hidden) + tanh/relu
  │       │       └─► actor_fc2: Linear(hidden → action_dim) → logits
  │       │
  │       └─► critic_fc1: Linear(hidden → hidden) + tanh/relu
  │               └─► critic_fc2: Linear(hidden → 1) → value
```

**Key properties**:
- Framework: **Flax NNX** (JAX)
- RNN type: Configurable (`LSTM` or `GRU`)
- Activation: Configurable (`tanh` or `relu`)
- Single hidden size for all layers
- No separate encoder — `input_proj` is the only feature transform

### B. DreamerV3 (`DreamerV3Agent` in `dreamer_v3_nnx.py`)
A world-model agent with explicit state decomposition:

```
obs (flat vector, dim=obs_dim)
  │
  ├─► Encoder: Sequential MLP [Linear → LayerNorm → SiLU] × N → embed (embed_dim)
  │
  ▼
RSSM (Recurrent State-Space Model):
  ┌──────────────────────────────────────────────┐
  │ img_in: Linear(stoch*discrete + action_dim   │
  │            → deter_dim) + ELU                │
  │     │                                        │
  │     ▼                                        │
  │ cell: LayerNormGRUCell(deter_dim)            │
  │     │    ◄── deter_prev (deterministic h)    │
  │     │                                        │
  │     ├─► img_out: Linear(deter → S*D)         │
  │     │       → prior_logits (S×D)             │
  │     │                                        │
  │     └─► obs_out: Linear(deter+embed → S*D)   │
  │             → post_logits (S×D)              │
  │             → OneHotDist → stoch sample      │
  └──────────────────────────────────────────────┘
  │
  ▼ feat = concat(deter, stoch_flat)  (feat_dim = deter + S*D)
  │
  ├─► Decoder:  MLP(feat_dim → obs_dim)   [LayerNorm + SiLU]
  ├─► Reward:   MLP(feat_dim → 255)        [TwoHot symlog]
  ├─► Continue: MLP(feat_dim → 1)          [Bernoulli]
  │
  ├─► Actor:    MLP(feat_dim → action_dim) [LayerNorm + SiLU]
  └─► Critic:   MLP(feat_dim → 255)        [TwoHot symlog]
```

**Key properties**:
- Framework: **Flax NNX** (JAX)
- GRU variant: Custom `LayerNormGRUCell` with separate LayerNorms on input/hidden gates
- Stochastic state: Categorical (S classes × D discrete), sampled via `OneHotDist` with straight-through gradients
- All MLPs use `Linear → LayerNorm → SiLU` blocks
- Reward/Critic use **TwoHot symlog** encoding (255 bins)
- Layer sizes fully configurable via YAML

## 2. The Neuromodulator Module (`NeuromodulatorRNN`)
A standalone recurrent module designed for high "affective inertia."

*   **Input ($c_t$)**: Full concatenated observation vector (same as task network input).
*   **Core**: 1-layer GRU (Flax `nnx.GRUCell`) with configurable hidden units (default: 64).
*   **Heads (Branched)**:
    - `head_percept`: Linear → Sigmoid (Size: `hidden_size` of Task network, gates the input projection output).
    - `head_memory`: Linear → Sigmoid (Size: `hidden_size` of Task-RNN, gates the carry state).
    - `head_action`: Linear → Softplus (Size: 1, scales policy temperature).
    - `head_reward`: Linear → Identity (Size: 1, scales reward signal).

## 3. Modulatory Injection Points

### A. Recurrent PPO: The Bi-Recurrent Actor-Critic
Two GRU/LSTMs run in parallel, coupled by the modulatory signal.

```python
# ── Modulator Path ──
h_mod_new = Modulator_GRU(obs_t, h_mod_prev)        # Slow affective state
z_perc, z_mem, z_act = Modulator_Heads(h_mod_new)

# ── Task Path (mirrors ActorCriticRNN.__call__) ──
# Layer 1: input_proj  (Linear: input_dim → hidden_size)
x_proj = relu(input_proj(obs_t))

# ◄◄ INJECTION A: Perceptual Gate ►►
x_proj = x_proj * sigmoid(z_perc)           # Gate input features

# Layer 2: rnn_cell  (GRUCell or LSTMCell: hidden → hidden)
# ◄◄ INJECTION B: Memory Gate ►►
h_gated = h_prev * sigmoid(z_mem)           # Gate carry state
h_new, x_h = rnn_cell(h_gated, x_proj)

# Layer 3: Actor/Critic Heads
logits = actor_fc2(activate(actor_fc1(x_h)))
# ◄◄ INJECTION C: Temperature ►►
logits = logits / jnp.exp(z_act)            # Scale exploration

value = critic_fc2(activate(critic_fc1(x_h)))
```

### B. DreamerV3: Modulating the World Model (RSSM)
The RSSM offers three structurally distinct injection sites.

```python
# ── Modulator Path ──
h_mod_new = Modulator_GRU(obs_t, h_mod_prev)
z_perc, z_mem, z_act, z_rew = Modulator_Heads(h_mod_new)

# ── World Model Path (mirrors RSSM.step) ──
# Layer 1: Encoder  (MLP: obs_dim → embed_dim)
embed = Encoder(obs_t)

# ◄◄ INJECTION A: Perceptual Gate ►►
embed = embed * sigmoid(z_perc)              # Gate encoded features

# Layer 2: img_in  (Linear: stoch*D + action → deter_dim)
x = elu(img_in(concat(stoch_prev, action)))

# Layer 3: LayerNormGRUCell  (deter_dim → deter_dim)
# ◄◄ INJECTION B: Memory Gate ►►
deter_gated = deter_prev * sigmoid(z_mem)    # Gate deterministic state
deter_new = cell(x, deter_gated)

# Layer 4: Posterior / Prior heads
post_logits = obs_out(concat(deter_new, embed))
# ... sample stoch from post_logits ...

# Layer 5: Reward Head  (MLP: feat_dim → 255)
feat = concat(deter_new, stoch)
reward_pred = reward_head(feat)
# ◄◄ INJECTION C: Reward Modulation ►►
reward_pred = reward_pred * sigmoid(z_rew)   # Scale nociceptive interpretation
```

## 4. Training Loop & Synergy
*   **Shared Objective**: Both the Task-RNN and the Modulator-RNN are trained end-to-end to minimize the global RL loss (PPO loss or Dreamer's variational loss).
*   **Decoupled Learning**: To prevent the modulator from over-fitting to task features, we can apply a **lower learning rate** or a **timescale penalty** to the Modulator-RNN, forcing it to focus on slow-moving regulatory trends.
*   **Ablation Hooks**:
    - `modulation.perceptual_only`: Disable memory/action/reward heads.
    - `modulation.static_context`: Use a feedforward modulator (MLP) as a baseline.
    - `modulation.type`: [None, 'Multiplicative', 'Additive', 'Affine'].
    - `modulation.context`: ['Full', 'VisualOnly', 'InteroOnly'].
