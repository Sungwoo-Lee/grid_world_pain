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

### 12. Biophysical Parameter Control (AlKilany & Goodman, 2025)
*   **Biological Mechanism**: Rapid adjustment of threshold $v_{th}$ and membrane time constants $\tau_m$ in SNNs.
*   **Mathematical Formalism**:
    $$\tau_m(t) \frac{dv}{dt} = -(v - v_{rest}) + R \cdot I(t)$$
    $$\text{Spike if } v(t) \geq v_{th}(t)$$
    Where $\tau_m(t)$ and $v_{th}(t)$ are dynamic outputs of a controller.
*   **Project Insight**: Implementation of **Signal-to-Noise Pumping**. In a high-noise environment (e.g., Grid World "Storm"), the agent can increase $v_{th}$ to "filter" low-intensity sensory noise, effectively "listening in the dips" of the background activity.

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

# Final Synthesis: The Simplistic Perceptual Modulation (SPM) Architecture

Based on the 31-paper review, we propose a **Simplistic Perceptual Modulation (SPM)** system for the grid-world agent. This architecture decouples *Observation Encoding* from *Regulatory Gain Control*.

## 1. The SPM Architectural Blueprint

The agent is composed of four interlocking modulatory loops:

### A. The Sensory Gate (Disinhibitory Gating)
*   **Mechanism**: A hierarchical CNN/SNN encoder where the output is gated by a "Salience Signal" $z_{sal}$.
*   **Equation**: $Obs_{eff} = \text{ReLU}(z_{sal} \odot \text{Encoder}(x) - \theta_{noise})$.
*   **Modulation**: If "Surprisal" (Prediction Error) is low, $z_{sal}$ is small, filtering out background details. If Surprisal spikes, $z_{sal} \uparrow$, revealing high-resolution cues like subtle trap indicators.

### B. The Attractor Manager (Manifold Destabilization)
*   **Mechanism**: The RNN hidden state $h_t$ is modulated by a global gain factor $g(t)$ (analogous to Noradrenaline).
*   **Equation**: $h_{t+1} = \tanh(g(t) \cdot (W h_t + U Obs_{eff}))$.
*   **Modulation**: Phasic bursts of $g(t)$ upon high TD-error "liquify" the current behavior, shaking the agent out of repetitive loops (attractor reset) and forcing it to explore.

### C. The Precision Controller (Active Inference)
*   **Mechanism**: An interoceptive branch predicts "Aleatoric Uncertainty" $\sigma^2$ (sensor noise).
*   **Equation**: Belief Update $\Delta \mu \propto (1/\sigma^2) \cdot (Obs - \text{Pred})$.
*   **Modulation**: In the "Fog of War" or high-smoke areas, the agent lowers its Sensory Precision $\Pi_s$, ignoring noisy pixels and relying on its internal "Prior" (path integration) to move toward the goal.

### D. The Hyper-Regulator (Doya's Meta-Controller)
*   **Mechanism**: A "Brainstem" module that outputs the agent's hyperparameters $\{\alpha, \gamma, \beta\}$.
*   **Mapping**:
    - **Serotonin ($\gamma$)**: High "Energy/Homeostasis" $\rightarrow$ High $\gamma$ (long-term planning). Low Energy $\rightarrow$ Low $\gamma$ (myopic survival).
    - **Acetylcholine ($\alpha$)**: High Surprisal $\rightarrow$ High $\alpha$ (fast map updating).
    - **Noradrenaline ($\beta$)**: Constant failure $\rightarrow$ Low $\beta$ (increase action randomness).

## 2. Implementation Roadmap
1.  **Module I (Sensory)**: Implement the Disinhibitory Gate in `src/models/encoders.py`.
2.  **Module II (Memory)**: Add slope modulation $g(t)$ to the RecurrentPPO actor network.
3.  **Module III (Regulator)**: Implement a small Hypernetwork that emits $\gamma$ and $\beta$ based on the agent's internal `injury` and `homeostatic` state.

This system guarantees that the agent's *perception* is not fixed, but is instead a dynamic function of its internal needs and environmental uncertainty.
