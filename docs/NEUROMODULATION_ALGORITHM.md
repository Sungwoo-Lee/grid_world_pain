# Unified Interoceptive Neuromodulation for Grid-World Pain

## Research Motivation & Hypotheses

The primary goal of this project is to study **perceptual precision modulation** in an RL agent equipped with interoceptive signals (injury, satiation, nutrition). The central phenomenon of interest is **pain hypervigilance** — the amplification of threat-related sensory processing following nociceptive experience, a hallmark of acute and chronic pain states in biological organisms.

However, the architecture we have implemented goes beyond pure perceptual modulation. A single recurrent neuromodulatory core (GRU), driven by the full observation vector (including interoceptive channels), simultaneously regulates **perception**, **memory**, and **decision-making** through branched output heads. This mirrors the multifunctional role of biological neuromodulators (Doya, 2002), which do not modulate a single cognitive domain in isolation.

### Architectural Scope: What Each Injection Controls

| Injection | Functional Domain | What It Modulates | Neuroscience Analog | Doya (2002) Metalevel |
|---|---|---|---|---|
| **A (Perception)** | Sensory gain / precision | Encoder features ($\gamma$: gain, $\beta$: threshold shift) | Acetylcholine: precision-weighting (Friston); VIP-SST disinhibitory gating (Ferguson & Cardin, 2020) | — |
| **B (Memory)** | Working memory persistence | GRU update gate bias (retention vs. forgetting) | Noradrenaline: integration window control (Shine et al., 2021; Costacurta et al., 2024) | Temporal discount $\gamma$ |
| **C-PPO (Exploration)** | Exploration-exploitation | Policy temperature (action entropy) | Dopamine: exploration drive | Exploration $\beta$ |
| **C-DreamerV3 (Reward)** | Nociceptive interpretation | Imagined reward scaling during planning | Serotonin: pain sensitivity; opioidergic modulation | Learning rate $\alpha$ |

All four domains are regulated by a **shared recurrent hidden state** — the modulator's GRU — which represents a slowly-evolving "affective tone." This state integrates the agent's history of interoceptive and exteroceptive experience, producing coordinated multi-domain modulation rather than independent per-domain adjustments.

### Core Hypothesis: Emergent Hypervigilance

> A single recurrent neuromodulatory core, driven by the full observation (including interoceptive signals like injury, satiation, nutrition), learns to simultaneously regulate perception, memory, and decision-making in a way that produces emergent phenomena analogous to pain hypervigilance — not through hand-designed rules, but through end-to-end optimization of the RL objective.

### Specific Predictions

**H1 — Perceptual amplification (Injection A):** After injury, the modulator should increase sensory gain ($\gamma$) on nociceptive-relevant features, making the agent more sensitive to threat cues. In PreActivation mode, the threshold shift ($\beta$) should become more negative in safe contexts (raising the activation threshold to filter noise) and more positive near danger zones (lowering the threshold — disinhibition — to let weak threat signals through). This mirrors the clinical observation that pain patients exhibit attentional bias toward threat-related stimuli.

**H2 — Memory persistence (Injection B):** An injured agent should hold onto threat information longer. The gate-bias on the GRU update gate controls how readily new information overwrites old state: negative $z_{\text{memory}}$ biases the update gate toward 0 (more retention, less forgetting), while positive bias promotes forgetting. The modulator should learn to reduce forgetting after pain events — "I remember the danger zone even after leaving it."

**H3 — Reward discounting / exploration suppression (Injection C):** In DreamerV3, $z_{\text{reward}}$ scales imagined rewards during planning. An injured agent could learn to discount expected rewards in dangerous regions — "even if there's food there, the pain risk makes it not worth it." This is a form of risk-aversion modulation that emerges from the nociceptive state. In PPO, the temperature head should lower action entropy after injury (conservative, deterministic behavior — the "freezing" response to threat), and increase entropy when the agent is healthy and satiated (free exploration).

**H4 — Coordinated multi-domain response:** The three injection sites should produce **coherent** responses to injury — perception amplifying, memory persisting, and decision-making becoming cautious simultaneously — rather than learning independent, uncoordinated strategies. This coherence would emerge from the shared GRU hidden state.

**H5 — Chronic pain analogs:** If the modulator GRU has sufficient temporal inertia, the "cautious mode" should persist well after the original injury has healed, resembling maladaptive chronic pain behavior. The duration of this persistence is controlled by the GRU's effective timescale relative to the environment's injury/healing dynamics.

### Experimental Validation Plan

| Hypothesis | Observable Metric | Comparison |
|---|---|---|
| H1 (Perceptual) | `mod_gamma_mean` increase after injury; `mod_beta` shift toward disinhibition near threats | Baseline vs. modulated; time-locked to injury events |
| H2 (Memory) | `mod_memory_mean` shift negative after injury (retention bias) | Baseline vs. modulated; memory probe tasks (delayed threat recall) |
| H3 (Reward/Exploration) | PPO: `temperature` decrease after injury. DreamerV3: `mod_z_reward_mean` decrease in dangerous regions | Baseline vs. modulated; injury vs. healthy episodes |
| H4 (Coherence) | Cross-correlation of $\gamma$, $z_{\text{memory}}$, $z_{\text{reward}}$/$\text{temperature}$ trajectories | Within-agent temporal analysis |
| H5 (Chronicity) | Duration of elevated modulator signals after injury recovery | Modulator GRU hidden state decay analysis; sweep `mod_hidden_size` |

### Why Not Just Perceptual Modulation?

Restricting the architecture to perception-only modulation (disabling Injections B and C) would test the "pure precision-weighting" hypothesis from predictive coding (Friston, 2023). However, the full multi-domain architecture is motivated by three observations:

1. **Biological neuromodulators are not domain-specific.** Noradrenaline simultaneously affects sensory gain, memory consolidation, and arousal. Acetylcholine modulates both sensory precision and attentional selection. Our branched-head architecture mirrors this multifunctionality.
2. **Hypervigilance is not just perceptual.** Clinical pain hypervigilance includes attentional bias (perception), catastrophizing/rumination (memory persistence), and behavioral avoidance (decision-making). A perception-only model cannot capture the full syndrome.
3. **Ablation is built in.** The `modulation.type = null` config produces an exact unmodulated baseline. Future config options like `modulation.perceptual_only` can selectively disable memory/action/reward heads to isolate the contribution of each domain, enabling controlled factorial experiments.

---

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

## 5. Training Stability Analysis for the Bi-Recurrent Architecture

The bi-recurrent design (Modulator-RNN + Task-RNN) introduces four concrete training stability risks that must be addressed before implementation.

### 5.1 🔴 Double-Gating the GRU Carry State (Critical)
The proposed Memory Injection externally gates the carry state before the GRU processes it:

```python
h_gated = h_prev * sigmoid(z_mem)           # External gate
h_new, x_h = rnn_cell(h_gated, x_proj)      # GRU's own update/reset gates
```

The GRU already has an internal update gate $u_t$ that controls memory retention:
$$h_t = (1 - u_t) \cdot h_{t-1} + u_t \cdot \tilde{h}_t$$

Adding an external multiplicative gate compounds the **vanishing gradient** problem. During BPTT, the gradient for timestep $t-k$ passes through:
$$\frac{\partial \mathcal{L}}{\partial h_{t-k}} \propto \prod_{j=0}^{k-1} \underbrace{(1 - u_{t-j})}_{\text{GRU gate}} \cdot \underbrace{\sigma(z_{mem,t-j})}_{\text{Modulator gate}}$$

This is a product of two numbers in $(0, 1)$ at every step — directly undermining the GRU's gradient highway.

**Recommendations** (choose one):
-   **(a) Internal Gate-Bias Injection** (Preferred, consistent with Ben-Iwhiwhu): Inject $z_{mem}$ as an additive bias to the GRU's reset or update gate *before* the sigmoid:
    $$u_t = \sigma(W_u x + U_u h_{t-1} + z_{mem})$$
    This shifts the gate's operating point without adding a second multiplicative bottleneck.
-   **(b) Switch to LSTM**: The LSTM separates cell state $c_t$ (long-term memory) from hidden output $h_t$. Externally gating $c_t$ is architecturally cleaner and analogous to modulating the forget gate, which has direct biological parallels with neuromodulatory memory persistence control.

### 5.2 🟠 Gate Initialization at 0.5 (Unfair Baseline Comparison)
All gates use `sigmoid(z)`. At initialization, $z \approx 0 \Rightarrow \sigma(z) = 0.5$.

**Consequence**: Every feature starts at 50% strength and every carry state at 50% retention. The baseline (non-modulated) network has no such attenuation, making comparison unfair and causing a slow training start as the network must first learn to "open" all gates.

**Recommendation**: Initialize gate head biases so that $\sigma(z) \approx 1.0$ (pass-through at start):
```python
# sigmoid(2.0) ≈ 0.88, sigmoid(3.0) ≈ 0.95
head_percept = Linear(mod_hidden, output_dim, bias_init=initializers.constant(2.0))
head_memory  = Linear(mod_hidden, output_dim, bias_init=initializers.constant(3.0))
```
The modulator starts as a **no-op** and gradually learns to deviate. This is consistent with residual gating best practices and the "Leaky Gate" concept from Section 2.

### 5.3 🟠 Unbounded Temperature Modulation (Policy Collapse Risk)
The current temperature injection is:
```python
logits = logits / jnp.exp(z_act)
```
-   If `z_act` → $+5$: temperature $= e^5 \approx 148$ → uniform random → no learning signal.
-   If `z_act` → $-5$: temperature $= e^{-5} \approx 0.007$ → deterministic → zero entropy → policy collapse.

PPO's entropy bonus provides some protection, but early in training `z_act` has no reason to stay bounded.

**Recommendations** (choose one):
-   **(a) Hard Clip**: `temp = jnp.clip(jnp.exp(z_act), 0.1, 10.0)`
-   **(b) Bounded Softplus**: `temp = 0.5 + softplus(z_act)` (always $\geq 0.5$)
-   **(c) Warmup Schedule**: Detach $z_{act}$ from the task loss for the first $N$ episodes, allowing the rest of the network to stabilize first.

### 5.4 🟡 Credit Assignment for Slow Modulator Under Truncated BPTT
In typical Recurrent PPO, rollouts are truncated to $T = 128$–$256$ steps. If the modulator is truly "slow" (high inertia GRU), the relevant credit assignment horizon might be $500+$ steps (e.g., "I was injured 200 steps ago, so I should still be cautious").

BPTT only backpropagates through $T$ steps. If the modulator's recurrent dynamics are too slow, gradients from early timesteps will be vanishingly small, and it will effectively learn nothing.

**Recommendations**:
-   **(a) Match Timescale to Truncation**: Ensure the modulator's effective temporal window fits within the truncation length.
-   **(b) Auxiliary Loss**: Add a self-supervised objective for the modulator that doesn't depend on long BPTT chains — e.g., predicting future interoceptive state from the current modulator hidden state. This is biologically plausible as interoceptive prediction.

### Summary of Risks

| Issue | Severity | Fix Effort | Recommendation |
|---|---|---|---|
| Double-gating GRU carry | 🔴 High | Medium | Inject as gate bias (5.1a), or switch to LSTM (5.1b) |
| Gate init at 0.5 | 🟠 Medium | Easy | Init biases to +2/+3 for pass-through (5.2) |
| Unbounded temperature | 🟠 Medium | Easy | Clip or softplus bound (5.3a/b) |
| Credit assignment | 🟡 Low-Med | Design | Match timescale to truncation (5.4a), or auxiliary loss (5.4b) |

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

## 2. The Neuromodulator Modules

Two agent-specific modulator classes share the same design principles (recurrent GRU core, branched heads, spatial grouping, pass-through initialization) but differ in their target dimensions and input handling.

### 2A. `NeuromodulatorRNN` (Recurrent PPO)
A standalone recurrent module designed for high "affective inertia," used with the `ActorCriticRNN` agent.

*   **Input ($c_t$)**: Full concatenated observation vector (same as task network input).
*   **Core**: 1-layer GRU (Flax `nnx.GRUCell`) with configurable hidden units (default: 64).
*   **Perceptual Modulation Type** (`modulation.type` in config):
    - **`"Multiplicative"`** — Post-activation gating (Ben-Iwhiwhu style). The gain signal gates features *after* the activation function. Simple, computationally cheap, well-tested in meta-RL.
    - **`"PreActivation"`** — Pre-activation gain + threshold shift (Ferguson & Cardin style). The gain ($\gamma$) and threshold shift ($\beta$) are applied *inside* the activation function, directly modulating the neuron's input-output curve. More biologically grounded — mirrors the disinhibitory circuit (Ferguson & Cardin, 2020) and energy landscape rescaling (Shine et al., 2021).
    - **`null`** — Disabled. No modulator is constructed (true baseline control, §5.2).
*   **Spatial Grouping** (AlKilany & Goodman, 2025):
    - Configurable grouping size $G$ (set via `modulation.grouping_size` in config).
    - Each head outputs $\lceil D_{target} / G \rceil$ values instead of $D_{target}$.
    - Each output is **broadcast** to $G$ consecutive units in the target layer.
    - Uses **Additive Modulation** to preserve heterogeneity: $z_{eff,i} = z_{baseline,i} + m_{\lfloor i/G \rfloor}$, where $z_{baseline,i}$ is a per-unit learned bias.
    - $G=1$: Fine-grained (per-neuron). $G=D_{target}$: Global (single scalar). Default: $G=1$.
*   **Heads (Branched)**:
    - `head_percept` ($\gamma$, gain): Linear → Sigmoid (Output: $\lceil hidden\_size / G \rceil$, broadcast to modulate input projection). **Bias init: +2.0** (§5.2: pass-through at start, $\sigma(2) \approx 0.88$). Used by both `Multiplicative` and `PreActivation`.
    - `head_percept_add` ($\beta$, threshold shift): Linear → Identity (Output: $\lceil hidden\_size / G \rceil$, broadcast as additive shift). **Bias init: 0.0** (no shift at start). **Only constructed when `modulation.type = "PreActivation"`**. This head produces the threshold-shift signal that mimics Ferguson & Cardin's disinhibitory gating — when $\beta > 0$, the effective activation threshold is lowered (disinhibition); when $\beta < 0$, it is raised (inhibition).
    - `head_memory`: Linear (Output: $\lceil hidden\_size / G \rceil$, broadcast as additive bias to GRU update gate). **No sigmoid** — injected pre-activation into the GRU's own gate (§5.1a). **Bias init: 0.0** (neutral).
    - `head_action`: Linear → Softplus + 0.5 (Size: 1, scales policy temperature). Output **clipped to [0.1, 10.0]** (§5.3a).
*   **Output** (`ModulatorOutput` NamedTuple):
    - `z_percept`: Perceptual gain signal ($\gamma$), shape `(hidden_size,)`.
    - `z_percept_add`: Perceptual additive signal ($\beta$), shape `(hidden_size,)`. Zeros when `type = "Multiplicative"`.
    - `z_memory`: Memory gate-bias signal, shape `(hidden_size,)`.
    - `temperature`: Bounded temperature scalar, shape `(1,)`.

### 2B. `DreamerNeuromodulatorRNN` (DreamerV3)
A dual-mode recurrent modulator for the DreamerV3 world model. It operates in two modes — **observation mode** during RSSM `step()` / `get_action()`, and **imagination mode** during `imagine_step()` — using separate input projections that feed a shared GRU core.

*   **Dual Input Projections** (avoids zero-padding; keeps GRU input distribution clean):
    - `proj_obs`: Linear(`obs_dim` → `mod_hidden_size`) — for observation mode. Input is the raw observation (symlog-encoded).
    - `proj_imagine`: Linear(`feat_dim + action_dim` → `mod_hidden_size`) — for imagination mode. Input is `concat(feat, action)` where `feat = concat(deter, stoch_flat)`. The imagined world state **and** the planned action jointly drive the modulatory response during planning. Lazily initialized via `set_imagine_input_dim()` since `feat_dim` depends on RSSM config (`deter_dim + stoch_dim * discrete`).
*   **Core**: 1-layer GRU (Flax `nnx.GRUCell`) with configurable hidden units (default: 64). Shared across both modes.
*   **Per-Head Target Dimensions**: Unlike PPO's single `target_hidden_size`, DreamerV3 heads target different network layers:
    - Perceptual heads ($\gamma$, $\beta$) → `embed_dim` (encoder output dimension, e.g. 128)
    - Memory head → `deter_dim` (RSSM deterministic state, e.g. 512)
    - Reward head → 1 (scalar)
*   **Heads (Branched)**:
    - `head_percept` ($\gamma$, gain): Same as PPO but targets `embed_dim`. **Only active in observation mode** — the encoder does not run during imagination.
    - `head_percept_add` ($\beta$, threshold shift): Same as PPO but targets `embed_dim`. Only constructed when `type = "PreActivation"`. Only active in observation mode.
    - `head_memory`: Linear → raw output targeting `deter_dim`. Injected as `gate_bias` into `ModulatedLayerNormGRUCell`. **Active in both observation and imagination modes** — the RSSM GRU dynamics are modulated during planning.
    - `head_reward`: Linear → **Sigmoid** (Size: 1). Scales the decoded imagined reward via `reward * sigmoid(z_reward)`. **Bias init: +2.0** ($\sigma(2) \approx 0.88$, near pass-through). **Active in imagination mode only** — the world model's reward head should learn to predict true rewards accurately, so reward modulation is NOT applied during world model training.
*   **Spatial Grouping**: Same as PPO — per-head grouping based on target dimension (`num_groups_percept = ceil(embed_dim / G)`, `num_groups_memory = ceil(deter_dim / G)`), with per-neuron learned baselines.
*   **Output** (`DreamerModulatorOutput` NamedTuple):
    - `z_percept`: Perceptual gain signal ($\gamma$), shape `(embed_dim,)`. Zeros during imagination.
    - `z_percept_add`: Perceptual additive signal ($\beta$), shape `(embed_dim,)`. Zeros during imagination or when `type = "Multiplicative"`.
    - `z_memory`: Memory gate-bias signal, shape `(deter_dim,)`. Active in both modes.
    - `z_reward`: Reward interpretation scale, shape `(1,)`. Sigmoid-bounded. Active in both modes (but only applied during imagination).

## 3. Modulatory Injection Points

### A. Recurrent PPO: The Bi-Recurrent Actor-Critic
Two GRU/LSTMs run in parallel, coupled by the modulatory signal. Injection A supports two perceptual modulation styles selected via `modulation.type`.

```python
# ── Modulator Path ──
h_mod_new = Modulator_GRU(obs_t, h_mod_prev)        # Slow affective state
mod_output, h_mod_new = Modulator(obs_t, h_mod_prev) # Returns ModulatorOutput

# mod_output contains (after spatial grouping + additive baselines):
#   z_percept      — perceptual gain gamma, shape (hidden_size,)
#   z_percept_add  — threshold shift beta, shape (hidden_size,) [zeros if Multiplicative]
#   z_memory       — GRU update gate bias, shape (hidden_size,)
#   temperature    — bounded scalar, shape (1,)

# ── Task Path (mirrors ActorCriticRNN.__call__) ──

# ◄◄ INJECTION A: Perceptual Modulation (type-dependent) ►►
#
# --- Option 1: "Multiplicative" (Ben-Iwhiwhu style) ---
# Post-activation gating: features are computed first, then scaled.
#   gamma = sigmoid(z_percept)   →  gain in (0, 1)
#   h = relu(Wx + b) * gamma
x_proj = relu(input_proj(obs_t))
x_proj = x_proj * sigmoid(mod_output.z_percept)      # Gate input features
#
# --- Option 2: "PreActivation" (Ferguson & Cardin style) ---
# Pre-activation gain + threshold shift: modulation applied INSIDE the activation.
#   gamma = sigmoid(z_percept)   →  multiplicative neural gain (Shine et al.)
#   beta  = z_percept_add        →  additive threshold shift (Ferguson disinhibition)
#   h = relu(gamma * (Wx + b) + beta)
#
# Biological interpretation:
#   gamma rescales the energy landscape (Shine et al., 2021): large gamma steepens
#   the neuron's I/O curve (increasing sensitivity), small gamma flattens it.
#   beta shifts the activation threshold (Ferguson & Cardin, 2020): positive beta
#   lowers threshold (disinhibition via VIP→SST circuit), negative beta raises it.
x_linear = input_proj(obs_t)                          # Raw pre-activation
gamma = sigmoid(mod_output.z_percept)                 # Gain
beta = mod_output.z_percept_add                       # Threshold shift
x_proj = relu(x_linear * gamma + beta)                # Modulated activation

# Layer 2: rnn_cell  (GRUCell or LSTMCell: hidden → hidden)
# ◄◄ INJECTION B: Internal Gate-Bias (§5.1a) ►►
# z_memory is injected INSIDE a custom ModulatedGRUCell (see §5 below).
# Internally: u_t = sigmoid(W_u @ x + U_u @ h_prev + z_memory)
# This preserves the GRU's gradient highway (no double-gating).
h_new, x_h = modulated_rnn_cell(h_prev, x_proj, gate_bias=mod_output.z_memory)

# Layer 3: Actor/Critic Heads
logits = actor_fc2(activate(actor_fc1(x_h)))
# ◄◄ INJECTION C: Bounded Temperature (§5.3a) ►►
logits = logits / mod_output.temperature              # Scale exploration (clipped)

value = critic_fc2(activate(critic_fc1(x_h)))
```

### B. DreamerV3: Modulating the World Model (RSSM)

The DreamerV3 modulator operates in **two modes** — observation (during `RSSM.step()` and `get_action()`) and imagination (during `imagine_step()`). The modulator's hidden state flows from the world model training scan into the imagination scan, so the agent's "affective state" accumulated from real observations persists and evolves during planning.

Three injection points are active across both modes:

| Injection | Target | Obs Mode | Imag Mode | Description |
|---|---|---|---|---|
| **A (Perceptual)** | Encoder output | ✅ Active | ❌ Inactive | Modulates encoded features before posterior |
| **B (Memory)** | RSSM GRU update gate | ✅ Active | ✅ Active | Gate-bias on `ModulatedLayerNormGRUCell` |
| **C (Reward)** | Decoded imagined reward | ❌ N/A | ✅ Active | Scales reward interpretation during planning |

#### Observation Mode (World Model Training & Action Selection)

```python
# ── Modulator: Observation Mode ──
# Input: raw observation (symlog-encoded)
x_mod = relu(proj_obs(obs_t))                       # Project to mod_hidden_size
h_mod_new, _ = mod_gru(h_mod_prev, x_mod)           # Shared GRU core
mod_output = compute_heads(h_mod_new, include_percept=True)

# ── World Model Path (mirrors RSSM.step) ──

# Layer 1: Encoder  (MLP body: obs_dim → embed_dim, split from final SiLU)
x_pre = encoder.body(obs_t)                          # Pre-activation embedding

# ◄◄ INJECTION A: Perceptual Modulation (type-dependent) ►►
#
# --- "Multiplicative" (post-activation) ---
# embed = SiLU(x_pre) * sigmoid(z_percept)
#
# --- "PreActivation" (Ferguson & Cardin style) ---
# gamma = sigmoid(z_percept)
# beta  = z_percept_add
# embed = SiLU(x_pre * gamma + beta)
embed = encoder.forward_with_modulation(obs_t, mod_output, modulation_type)

# Layer 2: img_in  (Linear: stoch*D + action → deter_dim)
x = elu(img_in(concat(stoch_prev, action)))

# Layer 3: ModulatedLayerNormGRUCell  (deter_dim → deter_dim)
# ◄◄ INJECTION B: Internal Gate-Bias (§5.1a) ►►
# z_memory is injected INSIDE the custom ModulatedLayerNormGRUCell.
# Internally: u_t = sigmoid(LN(W_u @ x) + LN(U_u @ deter_prev) + z_memory)
# This preserves the GRU's gradient highway (no double-gating).
deter_new = modulated_cell(x, deter_prev, gate_bias=mod_output.z_memory)

# Layer 4: Posterior / Prior heads (unchanged)
post_logits = obs_out(concat(deter_new, embed))
# ... sample stoch from post_logits ...
```

#### Imagination Mode (Behavior Learning via Imagined Rollouts)

During imagination, the modulator receives the **imagined world state and planned action** (`concat(feat, action)`) as input. This means the agent's pain sensitivity and memory dynamics evolve as it mentally plans trajectories through dangerous areas.

```python
# ── Modulator: Imagination Mode ──
# Input: concat(feat, action) where feat = concat(deter, stoch_flat)
mod_input = concat(feat, action)
x_mod = relu(proj_imagine(mod_input))                # Separate projection
h_mod_new, _ = mod_gru(h_mod_prev, x_mod)            # Same shared GRU core
mod_output = compute_heads(h_mod_new, include_percept=False)
# z_percept and z_percept_add are ZEROS (no encoder runs during imagination)
# z_memory and z_reward are ACTIVE

# ── Imagination Path (mirrors RSSM.imagine_step) ──

# No Injection A (encoder does not run during imagination)

# Layer 1: img_in + ELU (same as observation mode)
x = elu(img_in(concat(stoch, action)))

# ◄◄ INJECTION B: Internal Gate-Bias (active during imagination) ►►
deter_new = modulated_cell(x, deter_prev, gate_bias=mod_output.z_memory)

# Layer 2: Prior head (sample imagined stoch)
prior_logits = img_out(deter_new)
stoch = OneHotDist(prior_logits).sample(key)

# Layer 3: Reward prediction
feat = concat(deter_new, stoch)
rew = from_twohot(reward_head(feat))

# ◄◄ INJECTION C: Reward Interpretation Scale (imagination only) ►►
# Scales how "painfully" the agent interprets imagined rewards.
# sigmoid(z_reward) ∈ (0, 1) — modulates reward magnitude.
# NOT applied during world model training: the reward head learns true rewards.
rew = rew * sigmoid(mod_output.z_reward)
```

#### Hidden State Flow

The modulator's hidden state is carried through both scans as part of the `jax.lax.scan` carry:

```
World Model Training Scan (T steps):
  carry = (rssm_state, h_mod)
  for each timestep:
    mod_output, h_mod = modulator.forward_obs(obs_t, h_mod)
    embed = encoder.forward_with_modulation(obs_t, mod_output, ...)
    rssm_state = rssm.step(..., gate_bias=mod_output.z_memory)

  → outputs h_mod at each step: (T, B, mod_hidden)

Reshape for imagination:
  start_state: (B, T, ...) → (B*T, ...)
  h_mod_start: (B, T, mod_hidden) → (B*T, mod_hidden)
  Both stop_gradient'd — imagination doesn't backprop to world model.

Imagination Scan (HORIZON steps):
  carry = (rssm_state, h_mod)
  for each horizon step:
    feat = concat(deter, stoch)
    action = actor(feat).sample()
    mod_output, h_mod = modulator.forward_imagine(concat(feat, action), h_mod)
    rssm_state = rssm.imagine_step(..., gate_bias=mod_output.z_memory)
    rew = from_twohot(reward_head(feat)) * sigmoid(mod_output.z_reward)
```

**Key design decisions**:
- The modulator is trained as part of the **world model optimizer** — Injections A and B affect the reconstruction, reward prediction, and KL losses, providing gradient signal.
- `head_reward` weights receive **zero direct gradients** from the world model loss (z_reward is not used in the loss computation). However, the GRU backbone receives gradients from Injections A and B, so the shared hidden state becomes informative. The head_reward bias initialization (+2.0 → sigmoid ≈ 0.88) provides a mild near-pass-through default.
- During imagination, the modulator weights are NOT in the actor/critic `argnums`, so actor-critic gradients do not flow to the modulator. The modulator's effect on imagination is purely through its forward-pass outputs.

## 4. Training Loop & Synergy
*   **Shared Objective**: Both the Task-RNN and the Modulator-RNN are trained end-to-end to minimize the global RL loss (PPO loss or Dreamer's variational loss).
*   **Decoupled Learning**: To prevent the modulator from over-fitting to task features, we can apply a **lower learning rate** or a **timescale penalty** to the Modulator-RNN, forcing it to focus on slow-moving regulatory trends.
*   **Timescale Matching** (§5.4a): The modulator's effective temporal window must fit within the BPTT truncation length $T$. If $T=128$, the modulator should not need >128 steps to express its useful patterns. A modulator GRU with hidden size 64 and standard initialization naturally has an effective timescale of ~50–100 steps, which is suitable.
*   **Ablation Hooks**:
    - `modulation.perceptual_only`: Disable memory/action/reward heads.
    - `modulation.static_context`: Use a feedforward modulator (MLP) as a baseline.
    - `modulation.type`: [null, 'Multiplicative', 'PreActivation']. `null` = true baseline (no modulator). `Multiplicative` = post-activation gating (Ben-Iwhiwhu). `PreActivation` = pre-activation gain + threshold shift (Ferguson & Cardin).
    - `modulation.context`: ['Full', 'VisualOnly', 'InteroOnly'].
    - `modulation.grouping_size`: Integer $G$ controlling spatial grouping granularity (1 = per-neuron, $N$ = global scalar). Experiment sweep: [1, 8, 16, 32, hidden_size].
    - `modulation.temp_clip`: [min, max] bounds for temperature modulation (default: [0.1, 10.0]). PPO only.
    - `modulation.reward_bias_init`: Bias init for reward head (default: +2.0). DreamerV3 only.

## 5. Pre-Implementation Decisions (Resolved)

### 5.1 Custom Modulated GRU Cells
Flax NNX's built-in `nnx.GRUCell` does not accept a `gate_bias` argument. **Decision**: Write two custom Flax NNX modules:

1.  **`ModulatedGRUCell`** (`src/models/modulated_gru_cell.py`, for Recurrent PPO): Reimplements the standard GRU equations with an additional `gate_bias` input that is added to the update gate pre-activation:
    ```python
    u_t = sigmoid(W_u @ x + U_u @ h_prev + gate_bias)   # ← modulated
    r_t = sigmoid(W_r @ x + U_r @ h_prev)
    h_hat = tanh(W_h @ x + U_h @ (r_t * h_prev))
    h_new = (1 - u_t) * h_prev + u_t * h_hat
    ```
    When `gate_bias = None`, this is functionally identical to `nnx.GRUCell`.

2.  **`ModulatedLayerNormGRUCell`** (`src/models/modulated_layer_norm_gru_cell.py`, for DreamerV3): Extends the existing custom `LayerNormGRUCell` from `dreamer_v3_nnx.py` with the same optional `gate_bias` parameter, applied after the LayerNorm steps:
    ```python
    gates_ih = LN(W_ih @ x)
    gates_hh = LN(W_hh @ h)
    gates = gates_ih + gates_hh
    reset, update, cand = split(gates, 3)
    update = sigmoid(update + gate_bias)  # ← modulated (when gate_bias is not None)
    h_new = (1 - update) * h + update * tanh(cand)
    ```
    When `gate_bias = None`, this is functionally identical to `LayerNormGRUCell`. The RSSM conditionally constructs this cell (when `modulation_enabled=True`) or the original `LayerNormGRUCell` (when disabled).

### 5.2 Baseline Control Condition (`modulation.type = None`)
When `modulation.type` is set to `None` in config:
-   The `NeuromodulatorRNN` module is **not constructed** — no extra parameters, no extra compute.
-   The `ActorCriticRNN.__call__` (or `RSSM.step`) executes the **exact same code path** as the current unmodified agent.
-   This is the **true control condition** for ablation studies. Even a "no-op" modulator (initialized to pass-through) would add parameters and introduce numerical differences, making it unsuitable as a rigorous baseline.

### 5.3 Phased Implementation Strategy
**Phase 1: Recurrent PPO — ✅ Implemented.**
-   Implemented `NeuromodulatorRNN`, `ModulatedGRUCell`, and the modulated `ActorCriticRNN` variant.
-   Integrated into `train.py` with full config/ablation support.
-   Validated: (a) `modulation.type = None` reproduces baseline performance exactly, (b) modulated agent trains stably, (c) gate activations are interpretable via WandB logging.
-   Config: `configs/models/neuromodulated_ppo.yaml`.

**Phase 2: DreamerV3 — ✅ Implemented.**
-   Implemented `DreamerNeuromodulatorRNN` with dual input projections (`proj_obs` for observation mode, `proj_imagine` for imagination mode) feeding a shared GRU core.
-   Implemented `ModulatedLayerNormGRUCell` extending the existing `LayerNormGRUCell` with optional `gate_bias` on the update gate.
-   Modified `Encoder` to split `body` (pre-activation) from `final_act` (SiLU) for Injection A support, with `forward_with_modulation()` method supporting both `Multiplicative` and `PreActivation` styles.
-   Modified `RSSM.step()` and `RSSM.imagine_step()` to accept `gate_bias` parameter for Injection B.
-   Modified `WorldModel` and `DreamerV3Agent` to accept `modulation_config` and conditionally construct the modulator and modulated GRU cell.
-   Modified `DreamerTrainer.train_step` to carry `h_mod` through both the world model scan (Injections A+B) and imagination scan (Injections B+C). Modulator hidden state flows from the last step of the world model scan into the imagination scan initialization.
-   Modified `DreamerTrainer.get_action` to run the modulator in observation mode alongside the RSSM step and carry `mod_h` in the state dict.
-   Integrated into `train.py` DreamerV3 block with WandB logging for modulator metrics (gamma, beta, memory, z_reward).
-   Validated: (a) `modulation.type = None` reproduces baseline (unmodulated `LayerNormGRUCell`, identical code paths), (b) `Multiplicative` mode: `mod_gamma_mean ≈ 0.88` at init, (c) `PreActivation` mode: beta metrics included, (d) `train_step` and `get_action` produce correct shapes and metrics.
-   Config: `configs/models/neuromodulated_dreamer_v3.yaml`.

**Rationale for phasing**: PPO's architecture is compact (single `ActorCriticRNN`) and its training loop is much simpler than DreamerV3's world-model + actor-critic pipeline. Debugging modulator interactions was easier in the PPO setting before adapting to DreamerV3's dual-scan architecture.

---

# Paper Reviews: Source-Grounded Analysis (via NotebookLM)

> [!NOTE]
> The following reviews were generated by querying the project's NotebookLM library, which contains the full text of all referenced papers. All claims are **source-grounded** — citations refer to source indices within the NotebookLM document corpus.

## Paper Review A: Vecoven et al. (2020) — "Introducing Neuromodulation in Deep Neural Networks to Learn Adaptive Behaviours"

**Publication**: PLOS ONE, January 2020  
**Full Citation**: Vecoven, N., Ernst, D., Wehenkel, A., & Drion, G. (2020).

### A.1 Architecture: The Neuro-Modulated Network (NMN)

The NMN separates computation into two distinct, interacting neural networks:

1.  **Main Network**: A feed-forward DNN responsible for the primary input-output mapping (state → action/value). It is composed of neurons equipped with **parametric activation functions** whose slope and bias parameters are the targets of neuromodulation.
2.  **Neuromodulatory Network**: A separate network (typically an RNN/LSTM for meta-RL to handle dynamic history) that processes feedback and contextual data (context vector $c$). Its output is a **global neuromodulatory signal** $z \in \mathbb{R}^k$, broadcast to the main network.

In the Meta-RL setting, the authors employ **two separate NMNs** — one for the Actor and one for the Critic — with no shared parameters, allowing distinct modulatory signals for policy and value estimation.

### A.2 Mathematical Formulation: $\sigma_{NMN}$

The standard fixed activation $\sigma(x)$ is replaced by a context-dependent parametric version:

$$\sigma_{NMN}(x, z; w_s, w_b) = \sigma\left(z^\top (x \cdot w_s + w_b)\right)$$

Where:
-   $x \in \mathbb{R}$: Pre-activation input to the neuron (weighted sum from previous layer).
-   $z \in \mathbb{R}^k$: Global modulatory signal (shared across all neurons in the network).
-   $w_s \in \mathbb{R}^k$: **Per-neuron** learnable parameter vector controlling the **slope** (gain).
-   $w_b \in \mathbb{R}^k$: **Per-neuron** learnable parameter vector controlling the **bias** (offset).
-   $\sigma(\cdot)$: Base non-linearity (sReLU in practice).

**Dynamical Interpretation**: The inner product $z^\top w_s$ computes a **scalar gain factor** specific to each neuron based on context $z$, while $z^\top w_b$ computes a **dynamic bias shift**. This allows the network to silence ($z^\top w_s \approx 0$), amplify, or **invert** ($z^\top w_s < 0$) specific neurons based on the global context — effectively implementing per-neuron slope modulation.

### A.3 Modulatory Signal Computation & Sharing

-   **Computation**: $z = f(c)$, where $f$ is the neuromodulatory network and $c$ is the context input.
-   **Context Input**: In the Meta-RL framework, $c_t = h_t \setminus x_t = [x_0, a_0, r_0, \ldots, a_{t-1}, r_{t-1}]$ — the interaction history **excluding** the current state. This forces the modulator to infer task dynamics from past transitions and rewards, preventing it from becoming a simple feature extractor.
-   **Sharing**: The signal $z$ is **global** — identical across all neurons. However, each neuron's **reaction** is unique due to its specific learned $w_s, w_b$ vectors. Dimensionality $k$ is a free hyperparameter.

### A.4 sReLU Activation Function

The Saturated Rectified Linear Unit mimics bounded biological firing rates:

$$\text{sReLU}(x) = \min(1, \max(-1, x))$$

Used for all hidden layers. The output layer uses identity $\sigma(x) = x$. Empirical results showed sReLU outperformed sigmoidal activations in the NMN framework.

### A.5 Parameter Scaling

| Architecture | Modulates | Parameter Scaling |
|---|---|---|
| **NMN** | Activation functions (nodes) | $O(N_{\text{neurons}})$ |
| **Hypernetworks** | Synaptic weights (edges) | $O(N_{\text{neurons}}^2)$ |

NMNs scale linearly with the number of neurons, making them extensible to very large networks while providing context-dependent plasticity.

### A.6 Training Methodology

-   **Framework**: Meta-RL — agent interacts with a distribution of MDPs $\mathcal{D}$.
-   **Algorithm**: A2C (Advantage Actor-Critic) with GAE and PPO updates.
-   **Objective**: Maximize expected discounted reward over all tasks and episodes.

### A.7 Key Experimental Results

-   **Benchmarks**: Three custom continuous-control navigation tasks requiring task inference.
-   **Performance**: NMNs learned faster and achieved higher final rewards than baseline RNNs across all benchmarks.
-   **Near-Optimal**: On Benchmark 1 (1D target finding), NMN achieved cumulative reward 4534, approaching the theoretical Bayesian optimal of 4679.
-   **Adaptation Dynamics**: Analysis showed $z$ starts non-informative (uniform/exploration), then **converges to a stable task-specific value**, effectively "locking in" the adapted policy.
-   **Neuron Behavior**: The model learned to utilize the affine transformation to switch neurons off ($z^\top w_s \approx 0$) or invert their output (negative slope) depending on context.
-   **Robustness**: NMNs were surprisingly consistent with respect to the number of hidden layers, whereas baseline RNNs were sensitive to architecture choices.

---

## Paper Review B: Ben-Iwhiwhu et al. (2022) — "Context Meta-Reinforcement Learning via Neuromodulation"

**Publication**: Neural Networks, 2022  
**Full Citation**: Ben-Iwhiwhu, E., Dick, J., Ketz, N. A., Pilly, P. K., & Soltoggio, A. (2022).

### B.1 Architecture: The Neuromodulated Policy Network (NPN)

Unlike the NMN's separate modulator network, the NPN uses **intra-layer modulation** where each fully connected layer contains two co-located neural populations:

1.  **Standard Neurons**: Responsible for primary representation and processing of input (the "phenotype").
2.  **Neuromodulators**: A parallel population within the **same layer** that generates gating signals to alter the output of standard neurons (the "genotype/context").

These Neuromodulated Fully Connected Layers are stacked to form deep networks. The modulation is entirely **local** to each layer — there is no separate modulator network with its own recurrent state.

### B.2 Mathematical Formulation: Activity Gating

For a single Neuromodulated Layer with input $x$:

**Step 1 — Standard Pathway (Pre-activation)**:
$$h_s = W_s \cdot x$$

**Step 2 — Modulator Pathway (Context Generation)**:
$$g = \text{ReLU}(W_g \cdot x)$$
$$h_m = \tanh(W_m \cdot g)$$

Where $W_g$ connects input to neuromodulators, and $W_m$ projects modulatory activity onto the standard neurons. The $\tanh$ ensures the signal can be positive (excitatory) or negative (inhibitory/inverting).

**Step 3 — Modulated Output (Fusion)**:
$$h = \text{ReLU}(h_s \odot h_m)$$

Where $\odot$ denotes element-wise (Hadamard) multiplication.

**Step 4 — Discrete Control Variant (Binary Gating)**:
$$h = \text{ReLU}(h_s \odot \text{sign}(h_m))$$

This strictly turns neurons "on" or "off" rather than scaling them continuously, creating task-specific subnetworks within the larger policy.

### B.3 Modulation Targets

-   **Target**: Pre-nonlinearity activations ($h_s$) of standard neurons within the **same layer**.
-   **Mechanism**: Intra-layer modulation. The modulatory signal $h_m$ acts as a **gain on the weighted sum** $h_s$ before the final ReLU. This allows dynamic amplification, suppression (gating off), or inversion of specific features based on context.

### B.4 Signal Generation

-   **Type**: **Feedforward** — no recurrent state in the modulator.
-   **Input**: The neuromodulators receive the **same input** $x$ as the standard neurons in that layer.
-   **Process**: $x \xrightarrow{W_g} \text{ReLU} \xrightarrow{W_m} \tanh \rightarrow h_m$. Context is inferred implicitly from the layer's input. In deep networks, higher layers are modulated by complex features extracted by lower layers.

### B.5 Training Methodology

The NPN is used as a **drop-in replacement** for standard MLP policies within existing Meta-RL frameworks:

1.  **CAVIA** (Context Adaptation via Meta-Learning): Splits parameters into context parameters $\phi$ (inner loop) and network parameters $\theta$ (outer loop). NPN replaces the policy network.
2.  **PEARL** (Probabilistic Embeddings for Actor-Critic Meta-RL): Infers probabilistic latent context $z$ from experience history. Neuromodulation applied **only to the Actor** (not Critic or Inference network).

### B.6 Key Experimental Results

| Benchmark | Result |
|---|---|
| 2D Navigation, Half-Cheetah Dir/Vel | NPN ≈ SPN (simple tasks, static representations suffice) |
| **Meta-World ML1 & ML45** | NPN **significantly outperformed** SPN in success rate and return |
| ML45 (45 distinct tasks) | NPN achieved ~**2× higher success rates** than SPN with CAVIA |
| **CT-Graph** (Discrete, Depth 2–4) | NPN significantly outperformed SPN as complexity increased |

-   **CKA Analysis**: NPNs generate **dissimilar representations** for different tasks (low cross-task similarity), whereas SPNs learn averaged/overlapping representations.
-   **Mechanism**: The modulatory signal $h_m$ effectively acts as a **dynamic mask**, creating task-specific sub-networks by turning off subsets of neurons, allowing the single network to store conflicting optimal policies without interference.

### B.7 Related Work: Modulating Masks (Ben-Iwhiwhu et al., 2022b)

> [!NOTE]
> The full text of "Lifelong RL with Modulating Masks" (TMLR 2022) is **not in the current NotebookLM library**. The following is derived from the paper's abstract and citations within the existing sources.

The Modulating Masks paper extends the NPN concept to **Lifelong RL** with PPO and IMPALA agents. Key distinctions from the 2022a paper:

-   **Fixed Backbone**: The main network weights are frozen; only the masks are learned per task.
-   **Mask Composition**: A linear combination of previously learned masks is used to bootstrap learning on new tasks: $M_{\text{new}} = \sum_i \alpha_i M_i$, where $\alpha_i$ are learnable coefficients.
-   **Knowledge Reuse**: Mask composition solves tasks with extremely sparse rewards that cannot be solved from scratch.
-   **RL Algorithms**: PPO and IMPALA (rather than CAVIA/PEARL).
-   **Focus**: Lifelong/continual learning (anti-catastrophic-forgetting) rather than meta-learning.

---

# Comparative Analysis: Vecoven vs. Ben-Iwhiwhu vs. Our Implementation

## Architecture Comparison

| Dimension | Vecoven NMN (2020) | Ben-Iwhiwhu NPN (2022a) | **Our Bi-Recurrent Modulator** |
|---|---|---|---|
| **Modulator Type** | Separate network (RNN/LSTM) | Intra-layer parallel population (FF) | Separate recurrent network (GRU) |
| **Modulator Input** | History $c_t = h_t \setminus x_t$ (excludes current state) | Same input $x$ as task neurons | Full observation (same as task input) |
| **Modulator Architecture** | Any DNN/RNN | $W_g \rightarrow \text{ReLU} \rightarrow W_m \rightarrow \tanh$ per layer | GRU(obs\_dim → mod\_hidden) + branched heads |
| **Recurrent State** | ✅ Yes (RNN/LSTM in modulator) | ❌ No (feedforward only) | ✅ Yes (dedicated GRU in modulator) |
| **Temporal Dynamics** | Modulator accumulates task history | No memory — context from single timestep | "Affective inertia" — slow modulator GRU |

## Modulation Mechanism Comparison

| Dimension | Vecoven NMN (2020) | Ben-Iwhiwhu NPN (2022a) | **Our Bi-Recurrent Modulator** |
|---|---|---|---|
| **What is Modulated** | Activation function parameters (slope + bias) | Pre-activation features (hidden activations) | Input features, GRU update gate, policy temperature |
| **Core Equation** | $\sigma_{NMN}(x,z;w_s,w_b) = \sigma(z^\top(x \cdot w_s + w_b))$ | $h = \text{ReLU}(h_s \odot h_m)$ | See Injection A variants + B/C below |
| **Modulation Style** | **Parametric**: changes activation function shape | **Multiplicative mask**: gates activations | **Multi-site**: gating or gain+shift (configurable) + gate-bias + temperature |
| **Signal Sharing** | Global $z$ shared, per-neuron $w_s, w_b$ reaction | Per-layer modulator, per-neuron gating | Branched heads with spatial grouping |
| **Injection A (Perception)** | Implicit — slope modulation affects all layers | Implicit — gating at each layer | **Multiplicative**: $x = \text{relu}(Wx+b) \cdot \sigma(z_\gamma)$; **PreActivation**: $x = \text{relu}(\sigma(z_\gamma)(Wx+b) + z_\beta)$ |
| **Injection B (Memory)** | None (feedforward main network) | None (no RNN support) | $u_t = \sigma(W_u x + U_u h + z_{\text{memory}})$ (gate-bias) |
| **Injection C (Action)** | None (no temperature control) | None (no temperature control) | PPO: $\text{logits} = \text{logits} / \text{temperature}$ |
| **Injection D (Reward)** | None | None | DreamerV3: $\hat{r} = \hat{r} \cdot \sigma(z_{\text{reward}})$ (imagination only) |
| **Number of Injection Sites** | All hidden layers (homogeneous) | All hidden layers (homogeneous) | PPO: 3 sites (perception, memory, temperature). DreamerV3: 3 sites (perception, memory, reward) |

## Perceptual Modulation Variants (Injection A)

Our implementation supports two configurable perceptual modulation styles at Injection A. Both share the same modulator architecture (§2); the difference lies in **where and how** the gain signal is applied relative to the activation function.

| Dimension | **Multiplicative** (`type: "Multiplicative"`) | **PreActivation** (`type: "PreActivation"`) |
|---|---|---|
| **Equation** | $h = \text{relu}(Wx + b) \cdot \sigma(z_\gamma)$ | $h = \text{relu}(\sigma(z_\gamma) \cdot (Wx + b) + z_\beta)$ |
| **Biological Analogy** | Synaptic mask / activity gating (Ben-Iwhiwhu, 2022) | Disinhibitory gain circuit (Ferguson & Cardin, 2020) + energy landscape rescaling (Shine et al., 2021) |
| **Signals Used** | $z_\gamma$ (gain only) | $z_\gamma$ (gain) + $z_\beta$ (threshold shift) |
| **Extra Head** | None | `head_percept_add` ($\beta$) — only constructed in this mode |
| **Extra Parameters** | None | `head_percept_add` Linear + `z_perc_add_baseline` per-neuron params |
| **Modulation Target** | Post-activation features (output scaling) | Pre-activation signal (input-output curve reshaping) |
| **Effect of $\gamma \to 0$** | Feature silencing (output → 0) | Feature compression (slope → 0, but $\beta$ offset preserved) |
| **Effect of $\beta > 0$** | N/A | Threshold lowering (disinhibition — neuron fires more easily) |
| **Effect of $\beta < 0$** | N/A | Threshold raising (inhibition — neuron requires stronger input) |
| **Gradient Flow** | ⚠️ $\gamma \approx 0$ → zero gradient on $Wx+b$ (mitigated by bias init +2.0) | ✅ $\beta$ provides additive path: gradient always flows through the shift term |
| **Expressivity** | Scaling only (gain control) | Affine transformation (gain + shift = full I/O curve reshaping) |
| **Recommended Use** | Simpler baseline; sufficient for feature selection / sensory gating | Richer modulation; suitable for studying perceptual threshold changes and disinhibitory circuits |

## Training & Application Comparison

| Dimension | Vecoven NMN (2020) | Ben-Iwhiwhu NPN (2022a) | **Our Bi-Recurrent Modulator** |
|---|---|---|---|
| **Primary RL Algorithm** | A2C with GAE + PPO | CAVIA, PEARL | Recurrent PPO + DreamerV3 |
| **Learning Paradigm** | Meta-RL (task distribution) | Meta-RL (task distribution) | Single-task RL with interoceptive modulation |
| **Task Setting** | Multi-task navigation (infer target/wind) | Multi-task manipulation (Meta-World ML45) | Single environment with homeostatic drives |
| **End-to-End Training** | ✅ Joint optimization | ✅ Drop-in replacement for policy MLP | ✅ PPO: joint PPO loss. DreamerV3: modulator trained with world model optimizer, applied during imagination |
| **Baseline Control** | NMN vs vanilla RNN | NPN vs SPN (Standard Policy Network) | `modulation.type = None` (exact code path match for both PPO and DreamerV3) |

## Parameter Scaling & Expressivity

| Dimension | Vecoven NMN (2020) | Ben-Iwhiwhu NPN (2022a) | **Our Bi-Recurrent Modulator** |
|---|---|---|---|
| **Parameter Overhead** | $O(N_{\text{neurons}})$ — per-neuron $w_s, w_b$ | $O(N_{\text{neurons}})$ — per-layer $W_g, W_m$ | Fixed: GRU(mod\_hidden) + branched heads. PPO: 3–4 heads. DreamerV3: 3–4 heads + dual input projections + `head_reward` |
| **Spatial Grouping** | ❌ Per-neuron only | ❌ Per-neuron only | ✅ Configurable $G$ (AlKilany & Goodman, 2025) |
| **Dimensionality Reduction** | $z$ dimension $k$ is a free hyperparameter | No explicit control | Heads output $\lceil H/G \rceil$, broadcast to $H$ |
| **Per-Neuron Baselines** | Implicit via learned $w_s, w_b$ | ❌ No | ✅ Additive modulation preserves heterogeneity |

## Biological Fidelity

| Dimension | Vecoven NMN (2020) | Ben-Iwhiwhu NPN (2022a) | **Our Bi-Recurrent Modulator** |
|---|---|---|---|
| **Biological Analogy** | Cellular neuromodulation (gain control) | Activity gating (synaptic mask) | Neuromodulatory tone (affective state); PreActivation mode adds disinhibitory circuit analogy (Ferguson & Cardin) |
| **Temporal Inertia** | ✅ RNN accumulates history → "mood" | ❌ Instantaneous (no temporal memory) | ✅ GRU → persistent affective state |
| **Timescale Separation** | Possible (RNN vs FF main net) | ❌ No (both are feedforward) | ✅ Slow modulator GRU + fast task GRU |
| **Multi-Target Modulation** | ❌ Single target (activation shape) | ❌ Single target (hidden activations) | ✅ Perception + memory + exploration (PPO) / reward (DreamerV3) |
| **Pass-Through Init** | Not explicitly mentioned | Not explicitly mentioned | ✅ Bias +2.0 → $\sigma(2) \approx 0.88$ (near-identity start) |
| **Gradient Safety** | ✅ No double-gating | ⚠️ Multiplicative pre-ReLU can zero gradients | ✅ Gate-bias injection (§5.1a) preserves gradient highway |

## Summary of Key Differentiators

> [!IMPORTANT]
> **Our implementation is not a simple replication of either Vecoven or Ben-Iwhiwhu.** Rather, it synthesizes insights from both approaches while addressing their limitations:

| Innovation | Source | Our Adaptation |
|---|---|---|
| Parametric activation modulation | Vecoven (2020) | Replaced with **multi-site injection** (perception, memory, action/reward) for functional separation |
| Multiplicative activity gating | Ben-Iwhiwhu (2022a) | Available as **`"Multiplicative"` mode** for Injection A; combined with additive gate-bias (Injection B) to avoid double-gating |
| Pre-activation gain + threshold shift | Ferguson & Cardin (2020), Shine et al. (2021) | Available as **`"PreActivation"` mode** for Injection A; $\gamma$ rescales the I/O curve (energy landscape), $\beta$ shifts the activation threshold (disinhibitory gating). More expressive than pure multiplicative gating |
| Separate modulator network | Vecoven (2020) | Adopted, but with **dedicated GRU** instead of generic RNN, and **branched output heads**. DreamerV3 variant adds **dual input projections** for observation vs imagination modes |
| Intra-layer modulation | Ben-Iwhiwhu (2022a) | Rejected — we modulate at **distinct functional sites** rather than homogeneously across all layers |
| Context ≠ sensory input | Vecoven (2020) | Modified — we allow **full observation as context** but rely on **architectural bottleneck** (small GRU) to prevent identity collapse. DreamerV3 imagination mode uses `concat(feat, action)` as context |
| Spatial grouping | AlKilany & Goodman (2025) | Incorporated — configurable $G$ with additive modulation + per-neuron baselines. Both PPO and DreamerV3 modulators support grouping with per-head target dimensions |
| Bounded temperature | Doya (2002) / our design | PPO: **softplus + clip** prevents policy collapse. DreamerV3: replaced by **reward modulation** (sigmoid-bounded) for nociceptive interpretation |
| Pass-through initialization | Our design (§5.2) | Novel — bias init ensures modulator starts as **no-op**, critical for fair baseline comparison. Applied to all heads across both agents |
| Configurable modulation style | Our design | Novel — single architecture supports both **Multiplicative** and **PreActivation** perceptual modulation via config switch, enabling controlled ablation between post-activation gating and pre-activation curve reshaping |
| Imagination modulation | Our design | Novel — DreamerV3 modulator **continues running during imagination**, receiving `concat(feat, action)` as input. Injections B (memory) and C (reward) are active during planning, allowing the agent's "affective state" to influence imagined trajectories |
