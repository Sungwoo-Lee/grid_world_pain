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

---

## Paper Review C: AlKilany & Goodman (2025) — "Neuromodulation Enhances Dynamic Sensory Processing in Spiking Neural Network Models"

**Publication**: 2025
**Full Citation**: AlKilany, A. & Goodman, D. F. M. (2025).

### C.1 Architecture: Dual-Network SNN with External Modulator

The architecture consists of two interconnected networks: a **Primary SNN** (the plant) and a **Modulator Network** (the controller).

**Primary SNN**: A recurrent network with three layers:
1.  **Input layer**: Feedforward spiking layer ($N_{in} = 700$ for Spiking Heidelberg Digits, $4096$ for DVS tasks).
2.  **Hidden layer**: Fully connected recurrent layer of **Leaky Integrate-and-Fire (LIF)** neurons ($N_{hidden} = 200$).
3.  **Readout layer**: Fully connected non-spiking, leaky readout layer ($N_{out}$ depends on classification classes).

**Modulator Network — ANN Variant**: A two-layer MLP:
-   **Input**: Concatenation of (1) current values of all modulatable parameters, (2) recent spiking activity of the primary SNN hidden layer, and (3) raw input spike trains. If temporal grouping is used (modulator runs every $K$ steps), spiking activity is summed over the preceding $K$ time steps.
-   **Hidden**: Linear → ReLU.
-   **Output**: Linear → Sigmoid (for substitution mode) or Tanh (for addition mode), mapping to the parameter modulation space.

**Modulator Network — SNN Variant**: A single fully connected recurrent layer of LIF neurons. Continuous state inputs are injected as input currents. Discrete output spikes are segregated into groups representing positive or negative quantum adjustments for each primary network parameter.

### C.2 Mathematical Formulation: LIF Dynamics with Dynamic Parameters

The primary SNN neurons are current-based LIF units. The continuous-time dynamics:

$$\tau_m \dot{v} = -v + x$$
$$\tau_x \dot{x} = -x$$

When $v > v_{th}$: emit spike and reset $v \leftarrow v_r$.

For discrete-time implementation with integration step $dt$, time constants are parameterized as decay factors:

$$\alpha = e^{-dt/\tau_x}, \quad \beta = e^{-dt/\tau_m}$$

The modulator directly targets these parameters: $\alpha$, $\beta$, $v_{th}$, resting potential $v_0$, and reset potential $v_r$.

### C.3 Modulation Mechanisms: Substitution vs. Addition

The modulator output $m$ interacts with the primary network's parameter $p$ through two mechanisms:

**Substitution** (absolute replacement):
$$p \leftarrow m$$
Requires ANN variant with Sigmoid output to bound the parameter range. Static offsets are applied (e.g., $+0.5\text{V}$ for $v_{th}$, $-0.5\text{V}$ for resting/reset potentials).

**Addition** (relative adjustment):
$$p \leftarrow p + m$$
For ANN modulator: $m$ bounded via Tanh activation. For SNN modulator: two output neurons per parameter (one for $+\Delta$, one for $-\Delta$), where the magnitude of adjustment per spike is a learnable scalar.

### C.4 Spatial Grouping (Dimensionality Reduction)

Neurons in the primary SNN are clustered into spatial groups of size $G$. With additive modulation, each neuron $i$ maintains an independent, learnable baseline $\Psi_i(0)$. The modulator outputs a shared adjustment $m_{\lfloor i/G \rfloor}(t)$ for the $k$-th group:

$$\Psi_i(t+1) = \text{Clip}\left(\Psi_i(0) + m_{\lfloor i/G \rfloor}(t)\right)$$

This applies a uniform macroscopic shift to a group while preserving micro-level biophysical diversity through unique baselines.

**Granularity spectrum**: $G=1$ (per-neuron, synaptic precision) → $1 < G < N$ (volume transmission, mesoscale) → $G=N$ (global neuromodulatory bath).

**Key empirical finding**: Spatially extended modulation ($G=10$ or $G=20$) was **equally effective** as fine-grained modulation ($G=1$) across all tested tasks.

### C.5 Bounding and Clipping

Parameters are constrained to prevent numerical instability:

| Parameter | Range |
|---|---|
| $\tau_m, \tau_x$ | $[1, 18]$ ms |
| $v_{th}$ | $[0.5, 1.5]$ V |
| $v_0, v_r$ | $[-0.5, 0.5]$ V |

Under substitution: Sigmoid naturally restricts to $(0, 1)$, with static offsets mapping to valid ranges.
Under addition: Explicit clipping applied after every update step.

### C.6 Training Procedure

**End-to-end surrogate gradient descent**:

-   **Forward pass**: Spikes emitted using the Heaviside step function $H(x)$.
-   **Backward pass**: Non-differentiable derivative replaced with smooth surrogate:
$$H'(x) \approx \frac{1}{(|x| + 1)^2}$$

**Phased training**: (1) Pre-train primary SNN without modulation to establish initial representations. (2) Use pre-trained weights to initialize joint training of primary + modulator networks.

**Readout**: Maximum-over-time readout. Let $v_i^{max} = \max_t v_i(t)$ be the maximum membrane potential of the $i$-th output neuron. Class logits:
$$x_i = \text{softmax}(v^{max}) = \frac{\exp(v_i^{max})}{\sum_j \exp(v_j^{max})}$$

**Loss**: Cross-entropy over softmax logits, plus two regularization terms:
1.  **Firing rate penalty**: $\sum(r - 0.01)^2$ (penalizes deviation from target rate $0.01$).
2.  **Bursting suppression**: $\sum \text{ReLU}(r_{pop} - 100)^2$ (suppresses pathological population bursting).

**Optimizer**: Adam, $\text{lr} = 10^{-3}$ (or $2 \times 10^{-4}$ for specific tasks), batch size $64$.

---

## Paper Review D: Costacurta et al. (2024) — "Structured Flexibility in Recurrent Neural Networks via Neuromodulation"

**Publication**: 2024
**Full Citation**: Costacurta, J. C., Bhandarkar, S., Zoltowski, D., & Linderman, S. W. (2024).

### D.1 Architecture: The NM-RNN (Neuromodulated RNN)

Two coupled continuous-time recurrent subnetworks:

1.  **Output-Generating Subnetwork (Plant)**: A **low-rank RNN** of dimension $N$ that processes inputs and generates behavioral output.
2.  **Neuromodulatory Subnetwork (Controller)**: A smaller **full-rank RNN** of dimension $M$ ($M < N$), acting as a processing bottleneck.

**Timescale separation**: The controller time constant $\tau_z \gg \tau_x$ (the plant time constant), reflecting the biological reality that neuromodulatory signals evolve more slowly than sensory processing.

### D.2 Mathematical Formulation: Coupled ODEs

**Controller dynamics** — the neuromodulatory state $z(t) \in \mathbb{R}^M$:

$$\tau_z \frac{dz(t)}{dt} = -z(t) + W_z \phi(z(t)) + B_z u(t)$$

Where $W_z \in \mathbb{R}^{M \times M}$ are recurrent weights, $B_z \in \mathbb{R}^{M \times P}$ are input weights for external input $u(t) \in \mathbb{R}^P$, and $\phi(\cdot) = \tanh$.

**With feedback variant** (for tasks requiring state-dependent gating, e.g., Element Finder):

$$\tau_z \frac{dz(t)}{dt} = -z(t) + W_z \phi(z(t)) + (B_{zx} \phi(x(t)) + b_{zx}) + B_z u(t)$$

**Neuromodulatory signal extraction** — from hidden state $z(t)$, a $K$-dimensional signal:

$$s(z(t)) = \sigma(A_z z(t) + b_z) \in \mathbb{R}^K$$

Where $\sigma(\cdot)$ is the sigmoid, bounding each $s_k \in (0, 1)$.

**Plant dynamics** — the output-generating state $x(t) \in \mathbb{R}^N$:

$$\tau_x \frac{dx(t)}{dt} = -x(t) + W_x(z(t)) \phi(x(t)) + B_x u(t)$$

Where $B_x \in \mathbb{R}^{N \times P}$ are static input weights.

### D.3 Core Innovation: Dynamically Modulated Low-Rank Recurrence

The recurrent weight matrix $W_x(z(t))$ is **not static**. It is parameterized as a dynamically scaled rank-$K$ matrix:

$$W_x(z(t)) = \sum_{k=1}^{K} s_k(z(t)) \cdot l_k r_k^\top$$

Where:
-   $l_k, r_k \in \mathbb{R}^N$: Fixed rank-1 structural motifs (left and right factors).
-   $s_k(z(t)) \in (0, 1)$: Time-varying scalar gain for the $k$-th motif, computed by the controller.

Each $s_k$ acts as an independent, time-varying multiplicative gain on its corresponding rank-1 dynamical motif $l_k r_k^\top$.

### D.4 Readout

Linear readout of the output-generating state:

$$y(t) = C x(t) + d$$

Where $C \in \mathbb{R}^{O \times N}$, $d \in \mathbb{R}^O$.

### D.5 Theoretical Analysis: Connection to LSTM Forget Gates

Under constraints — linearized activation ($\phi(x) = x$), symmetric rank-1 components ($l_k = r_k$), orthonormal basis ($L^\top L = I$) — the system perfectly decouples. Reparameterizing as $w(t) = L^\top x(t)$:

$$w_k(t) = w_k(0) \exp\left(-\int_0^t \frac{1 - s_k(t')}{\tau_x} dt'\right)$$

Here $s_k(t)$ directly controls the exponential decay rate of each mode $w_k(t)$ — functionally equivalent to an LSTM forget gate acting on individual dynamical motifs.

### D.6 Training Procedure

**Loss**: MSE / $L_2$ loss between readout $y(t)$ and target sequence $\hat{z}(t)$.

**Optimization**: End-to-end via BPTT on Euler-discretized ODEs.

**Two-stage transfer learning protocol**:
1.  **Initial training**: Entire network (both subnetworks) trained on a base set of tasks.
2.  **Transfer**: Freeze plant weights ($l_k, r_k, B_x, C$) and controller recurrence ($W_z$). Train **only** the contextual input weights of the controller on new tasks.

This forces the controller to solve new tasks by discovering novel temporal sequences for $s(t)$ — recombining fixed rank-1 motifs without overwriting established synaptic weights.

---

## Paper Review E: Doya (2002) — "Metalearning and Neuromodulation"

**Publication**: Neural Networks, 2002
**Full Citation**: Doya, K. (2002).

### E.1 Core Framework: Neuromodulator → RL Hyperparameter Mapping

Four major neuromodulators are mapped to specific hyperparameters within an actor-critic RL architecture:

| Neuromodulator | Brain Region | RL Parameter | Function |
|---|---|---|---|
| **Dopamine (DA)** | VTA / SNc | TD error $\delta(t)$ | Global learning signal |
| **Serotonin (5-HT)** | Dorsal/Median Raphe | Discount factor $\gamma$ | Time scale of reward prediction |
| **Noradrenaline (NA)** | Locus Coeruleus | Inverse temperature $\beta$ | Exploration-exploitation trade-off |
| **Acetylcholine (ACh)** | Cholinergic Nuclei | Learning rate $\alpha$ | Speed of memory updates |

### E.2 Mathematical Formulation: Actor-Critic with Modulated Parameters

**Value function** (Critic): Parameterized as weighted sum of basis functions:
$$V(s) = \sum_j v_j b_j(s)$$

**Action-value function** (Actor):
$$Q(s, a) = \sum_k w_k c_k(s, a)$$

**TD error** (Dopamine):
$$\delta(t) = r(t) + \gamma V(s(t)) - V(s(t-1))$$

Alternative formulation suitable for basal ganglia circuitry:
$$\delta(t) = r(t) - (1 - \gamma) V(s(t)) + (V(s(t)) - V(s(t-1)))$$

Where $(1 - \gamma)V(s(t))$ acts as an immediate inhibitory signal.

**Weight updates** (modulated by learning rate $\alpha$ = Acetylcholine):
$$\Delta v_j = \alpha \cdot \delta(t) \cdot b_j(s(t-1))$$
$$\Delta w_k = \alpha \cdot \delta(t) \cdot c_k(s(t-1), a(t-1))$$

**Softmax policy** (modulated by inverse temperature $\beta$ = Noradrenaline):
$$P(a_i | s) = \frac{e^{\beta Q(s, a_i)}}{\sum_{j=1}^{m} e^{\beta Q(s, a_j)}}$$

-   $\beta \to 0$ (low NA): Random exploration.
-   $\beta \to \infty$ (high NA): Deterministic exploitation (winner-take-all).

### E.3 Meta-Learning: Heuristic Regulation of Meta-Parameters

Doya does **not** propose a unified differentiable meta-gradient objective. Instead, meta-parameters are regulated by heuristic rules based on statistical moments of the agent's experience:

**Discount factor $\gamma$ (Serotonin) — regulated by TD error variance**:
High variance in $\delta(t)$ → inhibit serotonergic system → lower $\gamma$ → bias toward reliable short-term predictions. Rationale: learning long-horizon predictions ($\gamma$ large) inherently produces high-variance TD errors.

**Inverse temperature $\beta$ (Noradrenaline) — regulated by Q-value variance**:
High variance in $Q(s, a)$ for a given state → suppress NA → decrease $\beta$ → promote wider exploration. Rationale: uncertain value estimates warrant more stochastic exploration.

**Learning rate $\alpha$ (Acetylcholine) — regulated by TD error oscillation**:
Frequent sign changes in $\delta(t)$ → inhibit cholinergic system → lower $\alpha$ → stabilize memory updates. Rationale: oscillating TD errors indicate the learning rate is too high for stable convergence.

### E.4 Neuroanatomical Computational Architecture

| Brain Structure | Computational Role |
|---|---|
| **Striatum (Patch/Striosome)** | State value function $V(s)$ |
| **Striatum (Matrix)** | Action-value functions $Q(s, a_i)$ |
| **VTA / SNc** (Dopaminergic) | Computes and broadcasts TD error $\delta(t)$ |
| **Dorsal/Median Raphe** (Serotonergic) | Modulates direct/indirect pathway balance → scales $\gamma$ |
| **Locus Coeruleus** (Noradrenergic) | Alters gain in SNr/GP competitive dynamics → scales $\beta$ |
| **Cholinergic Interneurons** | Modulates cortico-striatal plasticity → scales $\alpha$ |

---

## Paper Review F: Friston (2023) — "Computational Psychiatry: From Synapses to Sentience"

**Publication**: 2023
**Full Citation**: Friston, K. (2023).

> **Note**: This paper is a high-level expert review, not a derivation paper. The mathematical framework below synthesizes the conceptual definitions from Friston (2023) with standard Active Inference formalisms. Equations marked with (*) are from the broader Active Inference literature, not explicitly in this specific paper.

### F.1 Variational Free Energy

Under the Bayesian brain hypothesis, the brain minimizes variational free energy $\mathcal{F}$, an upper bound on surprise. For Gaussian distributions:

$$\mathcal{F}(\mu, x) = \frac{1}{2} \Pi_s \epsilon_s^2 + \frac{1}{2} \Pi_p \epsilon_p^2 - \frac{1}{2} \ln(\Pi_s \Pi_p) + C$$

Decomposition (*):
$$\mathcal{F} \approx \underbrace{D_{KL}[q(\mu) || p(\mu)]}_{\text{Complexity}} - \underbrace{\mathbb{E}_q[\ln p(x|\mu)]}_{\text{Accuracy}}$$

### F.2 Belief Update: Gradient Descent on Free Energy

Internal states $\mu$ (deep pyramidal cells) are updated based on prediction errors $\epsilon$ (superficial pyramidal cells):

-   **Sensory prediction error**: $\epsilon_s = x - g(\mu)$
-   **Prior prediction error**: $\epsilon_p = \mu - \mu_{prior}$

State update (negative gradient of $\mathcal{F}$):

$$\dot{\mu} = -\kappa \frac{\partial \mathcal{F}}{\partial \mu} = -\kappa \left(\Pi_p \epsilon_p - \nabla g(\mu)^\top \Pi_s \epsilon_s \right)$$

Where $\kappa$ is the integration rate, $\nabla g(\mu)$ is the Jacobian of the generative model.

### F.3 Precision as Synaptic Gain (Neuromodulation)

**Precision** $\Pi$ is the inverse variance: $\Pi = 1/\sigma^2$.

Physiologically, precision weighting = **synaptic gain control** — modulating postsynaptic sensitivity of superficial pyramidal cells broadcasting prediction errors. **Acetylcholine (ACh)** specifically encodes expected sensory precision $\Pi_s$.

Precision dynamics (*):
$$\dot{\Pi}_s \propto \beta - \epsilon_s^2$$

Where $\beta$ is a tonic baseline. Sustained high prediction errors suppress cholinergic precision gain, flattening the energy landscape to facilitate belief updating.

### F.4 Generalized Coordinates of Motion (*)

States are represented as vectors of higher-order temporal derivatives $\tilde{\mu} = (\mu, \mu', \mu'', \ldots)$:

$$\dot{\tilde{\mu}} = D\tilde{\mu} - \kappa \frac{\partial \mathcal{F}}{\partial \tilde{\mu}}$$

Where $D$ is a block-shift operator moving the system forward in time.

### F.5 Active Inference: Action Selection

**Continuous (motor reflexes)** — action descends the free energy gradient:
$$\dot{a} = -\kappa_a \frac{\partial \mathcal{F}}{\partial a} = -\kappa_a \frac{\partial x}{\partial a} \Pi_s \epsilon_s$$

**Discrete (planning)** (*) — policies $\pi$ selected to minimize Expected Free Energy $G$:
$$P(\pi) = \sigma(-\gamma \cdot G(\pi))$$

Where $\gamma$ is a precision parameter over policies (linked to Dopaminergic tone), and $G(\pi)$ balances epistemic value (information gain) and instrumental value (expected utility).

### F.6 Hierarchical Message Passing

Mapped onto canonical cortical microcircuits:

-   **Ascending** (bottom-up): Superficial pyramidal cells compute precision-weighted prediction errors ($\Pi \cdot \epsilon$) and broadcast them to the superordinate level.
-   **Descending** (top-down): Deep pyramidal cells encode posterior expectations ($\mu$) and send predictions ($g(\mu)$) to the level below to "explain away" prediction errors.

### F.7 Sensory Attenuation

To initiate movement, the brain transiently suspends sensory precision ($\Pi_s \to 0$) during active sensing:

$$\Pi_s(a) = \Pi_{baseline} \cdot \exp(-\lambda |a|) \quad (*)$$

As motor action amplitude $|a|$ increases, sensory precision drops, preventing proprioceptive prediction errors from overriding motor predictions.

---

## Paper Review G: Lee et al. (2024) — "Lifelong Reinforcement Learning via Neuromodulation"

**Publication**: 2024
**Full Citation**: Lee, S., Liebana, S., Clopath, C., & Dabney, W. (2024).

### G.1 Architecture: The Doya-DaYu Agent

The agent formalizes Doya's (2002) neuromodulator-to-hyperparameter mapping with **explicit uncertainty estimation**:

-   **Base algorithm**: Q-learning (tabular) or distributional RL (deep).
-   **Uncertainty estimation**: Ensemble of independent Q-learning agents (tabular) or ensemble of distributional RL agents (deep).
-   **Neuromodulatory integration**: Environmental uncertainties (estimated from ensemble) are continuously mapped to learning rate ($\alpha$) and softmax inverse temperature ($\beta$) at every timestep.

### G.2 Uncertainty Estimation

Two types of uncertainty, estimated from ensemble:

**Epistemic uncertainty** (unexpected — corresponds to Noradrenaline):
$$E(s, a) = \mathbb{E}_{i \sim \text{Unif}(1,N)} \left[\text{Var}_{\theta \sim P(\theta|D)} \left(y_i(\theta; s, a)\right)\right]$$

**Aleatoric uncertainty** (expected — corresponds to Acetylcholine):
$$A(s, a) = \text{Var}_{i \sim \text{Unif}(1,N)} \left[\mathbb{E}_{\theta \sim P(\theta|D)} \left(y_i(\theta; s, a)\right)\right]$$

Where $y_i$ is the $i$-th estimated quantile, $\theta$ are model parameters from ensemble $P(\theta|D)$.

**Tabular variant**: Aleatoric uncertainty estimated as return variance, updated via TD error $\delta$:
$$\text{Var}(G(a)) \leftarrow \text{Var}(G(a)) + \alpha_G [\delta^2 - \text{Var}(G(a))]$$

Epistemic uncertainty = variance of mean value estimates across ensemble members.

### G.3 Neuromodulatory Mappings

**Learning rate $\alpha$ (Acetylcholine)** — ratio of epistemic to total uncertainty:
$$\alpha(s, a) = \frac{E(s, a)}{E(s, a) + A(s, a)}$$

Naturally bounded in $(0, 1)$ as a ratio of positive variances. High epistemic uncertainty → high learning rate (rapid adaptation). High aleatoric uncertainty → low learning rate (stable, noise-robust).

**Inverse temperature $\beta$ (Noradrenaline)** — inversely proportional to average epistemic uncertainty:
$$\beta(s) = \frac{1}{\langle E(s, \hat{a}) \rangle_{\hat{a}}}$$

High epistemic uncertainty → low $\beta$ → exploratory policy. Low epistemic uncertainty → high $\beta$ → exploitative policy.

### G.4 Complete Update Equations

**Q-value update** with ACh-modulated learning rate:
$$Q(s, a) \leftarrow Q(s, a) + \alpha(s, a) \cdot \delta(t)$$

Where $\delta(t) = r(t) + \gamma \max_{a'} Q(s', a') - Q(s, a)$ (TD error = Dopamine).

**Action selection** with NA-modulated temperature:
$$P(a_i | s) = \frac{e^{\beta(s) Q(s, a_i)}}{\sum_j e^{\beta(s) Q(s, a_j)}}$$

### G.5 Continual Learning Setup

-   **Environment**: Non-stationary $k$-armed bandit ($k=5$ arms), $N$ contexts.
-   **Context duration**: Base $M=500$ steps, then switch with $p=0.4$.
-   **At context switch**: Gaussian payout distributions fully resampled ($\mu \in [-5, 5]$, $\sigma \in [0.001, 2]$).
-   **Task boundaries hidden**: Agent must autonomously detect non-stationarity.
-   **Mechanism**: Distribution shift at context switch → epistemic uncertainty spike → automatic $\alpha$ increase (rapid forgetting of obsolete values) + $\beta$ decrease (renewed exploration), without explicit boundary signals.

---

## Paper Review H: Osman et al. (2024) — "A Hopfield Network Model of Neuromodulatory Arousal State"

**Publication**: 2024
**Full Citation**: Osman, M. A. M., Fox, K., & Stern, J. I. (2024).

### H.1 Architecture: Arousal-Modulated Continuous Hopfield Network

A continuous Hopfield network where a scalar arousal parameter $\alpha$ controls the balance between internal memory attractors and external sensory drive.

### H.2 Mathematical Formulation: Network ODE

The state vector $y \in \mathbb{R}^N$ evolves according to:

$$\frac{dy}{dt} = -y + f\left(\frac{1}{\alpha} M y + W x\right)$$

Where:
-   $M \in \mathbb{R}^{N \times N}$: Symmetric, zero-diagonal recurrent connectivity matrix (stores memory patterns).
-   $W \in \mathbb{R}^{N \times D}$: Feedforward input weights.
-   $x \in \mathbb{R}^D$: External sensory stimulus.
-   $f(\cdot) = \tanh$ (element-wise).
-   $\alpha \in \mathbb{R}^+$: **Arousal parameter** — inversely scales recurrent interactions ($\frac{1}{\alpha} M y$), mediating internal memory vs. sensory drive.

### H.3 Phase Transition Analysis

| Phase | $\alpha$ Value | Behavior |
|---|---|---|
| **Ferromagnetic** (Deep Memory) | $\alpha \to 0$ | $\frac{dy}{dt} = -y + \text{sign}(My)$. Highly multistable, dominated by recurrent attractors. Sensory input irrelevant. |
| **Paramagnetic** (Sensory-Driven) | $\alpha \to \infty$ | $\frac{dy}{dt} = -y + f(Wx)$. Unistable, energy landscape flat. Network passively tracks sensory input. |
| **Critical Point** | $\alpha^* = \lambda_{max}(M)$ | Bifurcation from unistable to multistable dynamics (in absence of external input). |

### H.4 Energy Function (Lyapunov)

$$F(y | x; \alpha) = -\frac{1}{2\alpha} y^\top M y - \frac{1}{2} y^\top W x - \sum_{i=1}^{N} H_2^{(e)}\left(\frac{y_i + 1}{2}\right)$$

Where $H_2^{(e)}(p) = -p \log p - (1-p) \log(1-p)$ is the binary entropy function. This is equivalent to the mean-field variational free energy of a Boltzmann machine.

**Bayesian interpretation**: $\alpha$ scales the prior strength ($-\frac{1}{2} y^\top M y$) relative to likelihood ($-\frac{1}{2} y^\top W x$) and posterior entropy.

### H.5 Arousal Dynamics

$\alpha$ is an **exogenous control parameter** (no endogenous dynamics). Used as a "dynamic annealing schedule":
1.  Upon detecting stimulus change → momentarily set $\alpha$ high → flatten energy landscape → escape obsolete memory attractors.
2.  Gradually collapse $\alpha$ → network settles into new attractor corresponding to new stimulus.

### H.6 Simulation Details

Euler integration: $y(t + \Delta t) = y(t) + \Delta t \cdot \frac{dy(t)}{dt}$

Tested configurations: 2-unit networks with mutual inhibitory connections ($M_{ij} = -1$); 10-unit networks with dense inhibitory structure ($M = I - \mathbf{1}\mathbf{1}^\top$), identity feedforward weights ($W = I$), initial states uniform on $[-1, 1]$.

---

## Paper Review I: Rodriguez-Garcia et al. (2026) — "Noradrenergic-Inspired Gain Modulation Attenuates the Stability Gap in Joint Training"

**Publication**: 2026
**Full Citation**: Rodriguez-Garcia, A., Ghosh, A., & Ramaswamy, S. (2026).

### I.1 Algorithm: NGM-SGD (Noradrenergic Gain-Modulated SGD)

Modifies standard SGD by introducing a dynamic, non-learnable gain scalar $g(t)$ that scales weights during the forward pass, inspired by phasic noradrenaline bursts.

### I.2 Core Equations

**Effective weights** (forward pass):
$$W_{ij}^{eff}(t) = g_i(t) \cdot w_{ij}(t)$$

**Gradient scaling** (backpropagation — chain rule through $g$):
$$\frac{\partial L}{\partial w} = g(t) \frac{\partial L}{\partial W_{eff}}$$

**Weight update**:
$$w(t+1) = w(t) - \alpha \nabla_w L(f(x; g(t) w(t)), y_{target})$$

### I.3 Entropy-Driven Gain Dynamic

Uncertainty quantified as Shannon entropy of softmax output:
$$H(y_t) = -\sum_i \pi_i(y_t) \log(\pi_i(y_t))$$

Gain update (discrete-time leaky integrator):
$$g(t+1) = \gamma g(t) + (1 - \gamma) g_0 + \eta H(y_t)$$

Continuous-time equivalent:
$$\tau \frac{dg(t)}{dt} = (g_0 - g(t)) + \kappa H(y_t)$$

Where $\gamma \in (0,1)$ controls decay, $g_0$ is tonic baseline (typically $1$), $\eta > 0$ scales entropy impact.

### I.4 Two-Timescale Decomposition

The effective weight naturally decomposes:
$$W_{ij}(t) = \underbrace{g_0 w_{ij}(t)}_{\text{Slow (consolidated)}} + \underbrace{[g_i(t) - g_0] w_{ij}(t)}_{\text{Fast (contextual)}}$$

No dual weight storage needed — the two-timescale structure emerges from the multiplicative gain.

### I.5 Loss Landscape Flattening (Hessian Analysis)

Under reparameterization $\Phi(W) = gW$ ($g \geq 1$):
-   Gradient: $\nabla_W \tilde{L}(W) = g \nabla_{W_{eff}} L(W_{eff})$
-   Hessian: $\nabla_W^2 \tilde{L}(W) = g^2 \nabla_{W_{eff}}^2 L(W_{eff})$
-   Eigenvalue scaling: $\lambda \to \lambda / g^2$

At peak gain during task switch, curvature is maximally reduced, dampening sensitivity to distributional shifts.

### I.6 Algorithm Pseudocode

```
Initialize: W ← W_init; g ← g_init
for each context c_k ∈ C do
  for iteration i = 1 to I_C do
    (X, Ỹ) ~ D_k^B                              # Sample mini-batch
    π ← softmax(F_W(X; g))                        # Forward with g·W
    L ← (1/B) Σ l(π_j, ỹ_j)                      # Cross-entropy loss
    W ← W - α ∇_W L                               # Standard SGD update
    H ← -(1/B) Σ_j Σ_l π_{j,l} log π_{j,l}       # Batch entropy
    g ← γg + (1-γ)g_0 + ηH                        # Update gain
  end for
end for
```

### I.7 Hyperparameters

| Parameter | Symbol | Default | Sensitivity |
|---|---|---|---|
| Gain baseline | $g_0$ | 1 | Variations absorbed into $\alpha$ |
| Gain decay | $\gamma$ | 0.9 | Swept $\{0.85, 0.9, 0.95\}$; 0.9 optimal |
| Entropy scale | $\eta$ | Task-dependent | Swept $[0.1, 0.5]$; easy tasks need higher $\eta$, complex tasks need lower |

Key design: $\eta$ is **decoupled** from $\gamma$ (unlike standard EMA where $\eta = 1 - \gamma$), allowing slow decay + high reactivity simultaneously.

---

## Paper Review J: Tambaş et al. (2025) — "Neuromodulation via Krotov-Hopfield Improves Accuracy and Robustness of RBMs"

**Publication**: 2025
**Full Citation**: Tambaş, B., Subaşı, A. L., & Kabakçıoğlu, A. (2025).

### J.1 Standard RBM Equations

**Energy function** ($N$ visible units $v$, $M$ hidden units $h$):
$$E_\theta(v, h) = -v^\top W h - a^\top v - b^\top h$$

**Joint probability**: $p_\theta(v,h) = \frac{e^{-E_\theta(v,h)}}{Z_\theta}$

**Contrastive Divergence (CD) weight update**:
$$\delta W_{ij}^{CD} = \eta [\langle v_i h_j \rangle_d - \langle v_i h_j \rangle_m]$$

Where $\langle \cdot \rangle_d$ and $\langle \cdot \rangle_m$ are expectations over data and model (via $k$-step Gibbs sampling).

### J.2 Krotov-Hopfield Modulatory Step

**Input current** to each postsynaptic node $\nu$:
$$I_{r_\nu} = \langle W_\nu, x \rangle$$

Nodes ranked in descending order: $I_K \geq I_{K-1} \geq \cdots \geq I_1$.

**Global neuromodulatory signal** (rank-based lateral inhibition):
$$g_\nu(I) = \begin{cases} 1, & \text{if } r_\nu = K \text{ (Top-1 Winner)} \\ -\Delta, & \text{if } r_\nu \in [K-l, K-1] \text{ (Runners-up)} \\ 0, & \text{otherwise} \end{cases}$$

Where $\Delta > 0$ is the anti-Hebbian penalty magnitude, $l$ is the number of penalized runners-up.

### J.3 KH Weight Update

**Local update** (Hebbian + spherical regularization):
$$\Phi_{\mu\nu}(x, W) \equiv R^2 x_\mu - \langle W_\nu, x \rangle W_{\mu\nu}$$

This regularizes incoming weights onto a sphere of radius $R$: $\sum_\mu |W_{\mu\nu}|^2 = R^2$.

**Full KH weight modification** (normalized and modulated):
$$\delta W_{\mu\nu}^{KH} = \epsilon \frac{g_\nu(I) \Phi_{\mu\nu}(x, W)}{\max_{\mu\nu}[g_\nu(I) \Phi_{\mu\nu}(x, W)]}$$

-   **Winner** ($r_\nu = K$): Positive Hebbian update → weight vector moves toward input pattern.
-   **Losers** ($r_\nu \in [K-l, K-1]$): Negative anti-Hebbian update ($\times -\Delta$) → weight vectors pushed away from input.

### J.4 Training Loop (Interleaved CD + KH)

At each timestep $t$:
1.  Compute KH modulation: $\theta_t^{KH} = \theta_t + \delta\theta_t^{KH}$
2.  Compute CD gradient at intermediate state: $\nabla_\theta L(\theta_t^{KH})$
3.  Final update: $\theta_{t+1} = \theta_t^{KH} - \eta \nabla_\theta L(\theta_t^{KH})$

**Modulation direction**: Top-down ($\text{KH}_{TD}$, using $h \sim p_\theta(h|v)$ and $W^\top$) or Bottom-up ($\text{KH}_{BU}$, using $v \sim p_d(v)$ and $W$).

### J.5 Hyperparameters

| Parameter | Value |
|---|---|
| CD learning rate $\eta$ | 0.1, batch size 100 |
| KH step size $\epsilon$ | Annealed: $\epsilon(n) = \epsilon_0 (1 - n/S)^{3/2}$ |
| Schedule duration $S$ | $\{50, 100, 200, 300, 400, 500\}$ epochs |
| Runners-up $l$ | 1 |
| Anti-Hebbian penalty $\Delta$ | 0.4 |
| Spherical radius $R$ | 1.0 (std init) or 0.1 (LeCun init) |

### J.6 Effect on Representations

KH modulation forces strict feature competition → disentangled, non-overlapping receptive fields. Average cosine similarity between maximally overlapping features: Standard RBM = 0.41, $\text{KH}_{TD}$ = 0.37, $\text{KH}_{BU}$ = 0.35.

---

## Paper Review K: Tsuda et al. (2021) — "Neuromodulators Generate Multiple Context-Relevant Behaviors in a Recurrent Neural Network by Shifting Activity Hypertubes"

**Publication**: 2021
**Full Citation**: Tsuda, B., Pate, S. C., Tye, K. M., Siegelmann, H. T., & Sejnowski, T. J. (2021).

### K.1 Architecture: Continuous-Time Rate-Based RNN

-   **Network**: $N = 200$ units, sparse random connectivity ($p_{con} = 0.8$).
-   **Dale's Law**: Enforced — 80% excitatory, 20% inhibitory.
-   **Weight initialization**: $W \sim \mathcal{N}(0, g/\sqrt{N \cdot p_{con}})$, operating in chaotic regime ($g = 1.5$).
-   **Time constants**: $\tau$ sampled uniformly from $[20, 100]$ ms.
-   **Activation function**: Logistic sigmoid $r = \frac{1}{1 + e^{-x}}$.

### K.2 Mathematical Formulation: Network Dynamics

**Continuous-time ODE**:
$$\tau \frac{dx}{dt} = -x + Wr + W_{in} u + \mathcal{N}(0, 0.1)$$

**Discrete-time Euler integration** ($\Delta t = 5$ ms):
$$x_{i,t} = \left(1 - \frac{\Delta t}{\tau}\right) x_{i,t-1} + \frac{\Delta t}{\tau} \left(\sum_j W_{ji} r_{ji,t-1} + W_{ui} u_{t-1}\right) + \mathcal{N}(0, 0.1)$$

### K.3 Neuromodulatory Weight Scaling

The neuromodulator $f$ uniformly scales **outgoing weights** of targeted presynaptic subpopulations. Partitioning into non-modulated ($k$) and modulated ($q$) neurons:

$$\tau \dot{x}_i = -x_i + \sum_k W_{ki} r_{ki} + f \cdot \sum_q W_{qi} r_{qi} + W_{ui} u + \mathcal{N}(0, 0.1)$$

**Targeting options**: Global (100%), random subpopulations (10%–90%), or cell-type specific (excitatory-only / inhibitory-only).

### K.4 Hypertubes: State-Space Manifold Structure

A "hypertube" is the stereotyped, robust path that population activity traces through high-dimensional state space over time. Despite intrinsic noise, the vector flow fields constrain trajectories within isolated tubes.

Visualized via PCA on time-varying firing rates $r$; first 3 PCs capture 80–92% of activity variance.

### K.5 Producing Distinct Behavioral Manifolds

Different $f_{nm}$ values produce distinct manifolds by altering the internal flow field. Intermediate (untrained) $f_{nm}$ values shift hypertubes along a continuous "transition manifold." Output transitions are highly non-linear (sigmoidal/exponential), characterized by $EC_{50}$ (half-maximal transition level).

**Capacity**: Up to 9 distinct output behaviors embedded in a single network using 9 unique neuromodulated subpopulations.

### K.6 Training Procedure

-   **Loss**: Least Square Error between readout $O = W_{out} r + b_{out}$ and target trajectory.
-   **Optimizer**: BPTT with Adam.
-   **Stopping**: Average trial LSE over last $n \times 25$ trials $< 1.0$, or max 15,000 trials.

### K.7 Topological Separation Analysis

**Angle of Departure (AoD)** — measures how the manifold departs from a linear interpolation:

$$\vec{v}_1 = \vec{p}_F - \vec{p}_N, \quad \vec{u}_1 = \vec{p}_{L1} - \vec{p}_N$$
$$AoD = \cos^{-1} \frac{\vec{u}_1 \cdot \vec{v}_1}{|\vec{u}_1| |\vec{v}_1|}$$

Where $\vec{p}_N$ = no-modulation state, $\vec{p}_F$ = full-modulation state, $\vec{p}_{L1}$ = first intermediate level. Larger AoD correlates with lower sensitivity (higher $EC_{50}$).

---

## Paper Review L: Wainstein et al. (2025) — "Evidence from Pupillometry, fMRI, and RNN Modelling Shows That Gain Neuromodulation Mediates Task-Relevant Perceptual Switches"

**Publication**: 2025
**Full Citation**: Wainstein, G., Whyte, C. J., Ehgoetz Martens, K. A., Müller, E. J., Medel, V., Anderson, B., Stöttinger, E., Danckert, J., Munn, B. R., & Shine, J. M. (2025).

### L.1 Architecture: E/I-Constrained RNN with Dynamic Gain

-   **Network**: $N = 40$ units (32 excitatory, 8 inhibitory). Dale's Law enforced via static mask $W_{mask}$.
-   **Connectivity**: $W_{rec} = |W_{rec}^{plastic}| \odot W_{mask}$ (absolute value + mask ensures sign constraint).
-   **Input/Output**: $W_{in} \in \mathbb{R}^{40 \times 2}$, $W_{out} \in \mathbb{R}^{32 \times 2}$ (strictly positive). Only excitatory rates $r_E$ contribute to readout.

### L.2 Network Dynamics ODE

$$dx = \frac{1}{\tau}\left(-x(t) + W_{rec} r(t) + W_{in} u(t)\right) dt + dW$$

Euler-Maruyama discretization ($\tau = 100$ ms):
$$x(t + \Delta t) = (1 - \alpha) x(t) + \alpha(W_{rec} r(t) + W_{in} u(t)) + \sigma_{rec} \sqrt{\Delta t} \mathcal{N}(0, 1)$$

Where $\alpha = \Delta t / \tau$, $\sigma_{rec} = 0.01$.

### L.3 Gain-Parameterized Activation Function

$$r(t) = \frac{1}{1 + \exp(-g(t) \odot x(t))}$$

Where $g(t) \in \mathbb{R}^{40}$ is the time-varying neuronal gain vector. High $g$ → steep sigmoid (sensitive, decisive). Low $g$ → flat sigmoid (noisy, uncertain).

### L.4 Classification Uncertainty

**Readout**: $z = W_{out} r_E \in \mathbb{R}^2$

**Softmax** with inverse temperature $\omega = 0.25$:
$$p(z)_i = \frac{\exp(\omega z_i)}{\sum_j \exp(\omega z_j)}$$

**Shannon entropy**:
$$H(z) = -\sum_i p(z)_i \ln(p(z)_i)$$

### L.5 Gain Dynamics (Uncertainty-Driven)

$$\tau \frac{dg}{dt} = g_{tonic} - g(t) + \gamma H(z)$$

Where $g_{tonic} = 1$ (baseline), $\gamma$ scales the uncertainty forcing. Without ambiguity, $g \to 1$ exponentially. During high uncertainty, $H(z)$ drives phasic gain bursts.

### L.6 Training Procedure

-   **Task**: Change-detection — inputs morph linearly between categories over 1-second trials.
-   **Loss**: Cross-entropy on readout $z(t)$.
-   **Optimizer**: BPTT with Adam, 1000 iterations.
-   **During training**: Gain fixed at $g = 1$ (disabled), coarse $\Delta t = 200$ ms.
-   **At test time**: Gain dynamics enabled, fine $\Delta t$.

### L.7 Perceptual Switching Mechanism

1.  Training segregates E/I units into two stimulus-selective clusters.
2.  Gain burst at perceptual ambiguity selectively amplifies inhibitory units targeting the dominant excitatory population.
3.  The stable attractor for the prior percept is **destabilized** (bifurcation into oscillatory regime).
4.  Network escapes the old attractor; competing population establishes a new fixed point.

This is equivalent to **flattening the energy landscape** (reducing barrier height between attractors), allowing large state displacements.

### L.8 Pupillometry Connection

The model's simulated gain dynamics $g(t)$ closely mimic empirical pupillary responses. Pupil dilation reflects the biological execution of the uncertainty-driven gain forcing $\gamma H(z)$, validating the LC-noradrenaline account of perceptual switching.

---

## Paper Review M: Ha et al. (2016) — "HyperNetworks"

**Publication**: ICLR 2017 (arXiv 2016)
**Full Citation**: Ha, D., Dai, A. M., & Le, Q. V. (2016).

### M.1 Foundational Concept

A hypernetwork $H$ (parameterized by $\Phi$) generates weights $\Theta$ for a main network $M$:
$$\Theta_M = H(z; \Phi)$$

This relaxed weight-sharing mechanism allows adaptation without unique learnable parameters per layer/timestep.

### M.2 Static Hypernetworks (Feedforward/CNN)

For the $j$-th convolutional layer, a learned embedding $z_j \in \mathbb{R}^{N_z}$ generates kernel $K_j \in \mathbb{R}^{N_{in} f_{size} \times N_{out} f_{size}}$:

**Step 1** — Intermediate vectors (per input channel $i$):
$$a_i^j = W_i z_j + B_i \quad \forall i = 1, \ldots, N_{in}$$

**Step 2** — Kernel slice generation (shared $W_{out}$ across slices):
$$K_i^j = \langle W_{out}, a_i^j \rangle + B_{out}$$

**Step 3** — Concatenation: $K_j = (K_1^j \; K_2^j \; \ldots \; K_{N_{in}}^j)$

Where $W_i \in \mathbb{R}^{d \times N_z}$, $W_{out} \in \mathbb{R}^{f_{size} \times N_{out} f_{size} \times d}$.

### M.3 Dynamic Hypernetworks (HyperRNN) — Weight Factorization Trick

Instead of generating full matrices, the hypernetwork generates a **scaling vector** $d(z) \in \mathbb{R}^{N_h}$ that scales rows of a static base matrix:

$$W(z) = \text{diag}(d(z)) W_0 = d(z) \odot W_0$$

**Complete HyperRNN equations**:

1. HyperRNN state update: $\hat{x}_t = (h_{t-1}; x_t)$, $\hat{h}_t = \phi(W_{\hat{h}} \hat{h}_{t-1} + W_{\hat{x}} \hat{x}_t + \hat{b})$
2. Embedding generation: $z_h = W_{\hat{h}h} \hat{h}_{t-1} + b_{\hat{h}h}$, $z_x = W_{\hat{h}x} \hat{h}_{t-1} + b_{\hat{h}x}$, $z_b = W_{\hat{h}b} \hat{h}_{t-1}$
3. Scaling vectors: $d_h(z_h) = W_{hz} z_h$, $d_x(z_x) = W_{xz} z_x$, $b(z_b) = W_{bz} z_b + b_0$
4. Main RNN update: $h_t = \phi(d_h(z_h) \odot W_h h_{t-1} + d_x(z_x) \odot W_x x_t + b(z_b))$

### M.4 HyperLSTM Cell (Complete Equations)

A smaller HyperLSTM cell ($\hat{h}_t$, $\hat{c}_t$) generates per-gate scaling vectors for the main LSTM.

**For each gate** $y \in \{i, g, f, o\}$:
1. HyperLSTM computes embeddings: $z_h^y, z_x^y, z_b^y$
2. Scaling vectors: $d_h^y = W_{hz}^y z_h^y$, $d_x^y = W_{xz}^y z_x^y$, $b^y = W_{bz}^y z_b^y + b_0^y$

**Main LSTM update**:
$$y_t = \text{LN}(d_h^y \odot W_h^y h_{t-1} + d_x^y \odot W_x^y x_t + b^y) \quad \text{for } y \in \{i, g, f, o\}$$
$$c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \text{Dropout}(\phi(g_t))$$
$$h_t = \sigma(o_t) \odot \phi(\text{LN}(c_t))$$

### M.5 Parameter Efficiency

As depth $D$ increases, marginal cost = $N_z$ per layer (the embedding dimension). For CNNs with $D$ layers, the hypernetwork parameterizes the entire weight volume with:
$$N_z \times D + d \times (N_z + 1) \times N_{in} + f_{size} \times N_{out} \times f_{size} \times (d + 1)$$

### M.6 Training

End-to-end via standard backpropagation / BPTT. Loss: task-specific (cross-entropy for classification, BPC/log-loss for sequence modeling). Both hypernetwork and main network optimized jointly.

---

## Paper Review N: Beck et al. (2023) — "Hypernetworks in Meta-Reinforcement Learning"

**Publication**: 2023
**Full Citation**: Beck, J., Jackson, M. T., Vuorio, R., & Whiteson, S. (2023).

### N.1 The Problem: Initialization Instability

When naive initialization (Kaiming, Orthogonal, etc.) is applied to the hypernetwork, the **variance of generated base network weights is uncontrolled**, causing exploding/vanishing activations in the generated policy — severe training instability in meta-RL.

### N.2 Bias-HyperInit Algorithm

**Core insight**: If the hypernetwork's output reduces to a standard reliable initialization at step zero, the system avoids instability.

Let $W, b$ be the weight matrix and bias of the hypernetwork's **final linear layer**. Initialize:

$$W_{i,j} := 0 \quad \forall i, j$$
$$b := \phi_{shared} \sim f(\phi)$$

Where $f(\phi)$ is any standard initialization (Kaiming, Orthogonal, etc.) known to work for the base architecture.

**At initialization**, regardless of input $x$:
$$\phi_{init} = Wx + b = 0 \cdot x + \phi_{shared} = \phi_{shared}$$

All tasks share an identical, stable initialization. Gradients subsequently update $W$ for context-dependent parameter divergence. All preceding hidden layers use any default initialization.

### N.3 Hypernetwork Architecture for Meta-RL

1.  **Task encoder** $g$: Summarizes interaction history $\tau_t$ into embedding $e = g(\tau_t)$ (e.g., recurrent VAE in VariBAD).
2.  **Hypernetwork** $h_\theta$: Maps embedding to policy parameters: $\phi = h_\theta(e) = Wx + b$ (final layer).
3.  **Base policy**: $a_t \sim \pi_\phi(a | s_t)$. Policy has **no independent parameters** — purely a function of context.

### N.4 Meta-RL Training Loop

**Objective**:
$$\arg\max_\theta \mathbb{E}_{M \sim p(M)} \left[\mathbb{E}_{\tau \sim \pi_\theta(\cdot), M} [R(\tau)]\right]$$

-   **Inner loop**: Task encoder updates $e$ at every timestep from growing history.
-   **Outer loop**: Hypernetwork + encoder optimized via PPO (VariBAD) or pure RL (RL2).

### N.5 Advantage over Hyperfan-In (HFI)

HFI requires custom variance analysis per base architecture — brittle and architecture-dependent. Bias-HyperInit is **architecture-agnostic**: zeroing $W$ and drawing $b$ from the target distribution guarantees the exact variance profile of the base initialization. Empirically matches or exceeds HFI on Meta-World and MuJoCo benchmarks.

---

## Paper Review O: Borycki et al. (2022) — "Hypernetwork Approach to Bayesian MAML"

**Publication**: 2022
**Full Citation**: Borycki, M., Przybysz, P., Tabor, J., Zięba, M., & Spurek, P. (2022).

> **Note**: Limited detail available in NotebookLM sources. Summary based on available information.

### O.1 Core Concept

A hypernetwork maps task support sets to the **parameters of a probability distribution** over the target network's weights, capturing Bayesian uncertainty:

$$\mu, \Sigma = H(c; \phi)$$
$$\theta \sim \mathcal{N}(\mu, \Sigma)$$

### O.2 Training

The loss includes a KL-divergence regularization term:
$$\mathcal{L} = \mathcal{L}_{task} + D_{KL}(q(\theta | c) \| p(\theta))$$

This enforces uncertainty calibration — the generated weight distribution should not deviate arbitrarily from the prior.

---

## Paper Review P: Jiang et al. (2021) — "Dynamic Predictive Coding with Hypernetworks"

**Publication**: 2021
**Full Citation**: Jiang, L. P., Gklezakos, D. C., & Rao, R. P. N. (2021).

### P.1 Generative Spatiotemporal Model

Joint probability factorization:
$$p(I_{1:T}, r_{1:T}) = p(r_1) \prod_{t=1}^{T} p(I_t | r_t) \prod_{t=2}^{T} p(r_t | r_{1:t-1})$$

**Spatial model** (sparse coding): $I_t = U r_t + n$ (columns of $U$ are spatial filters, $n$ is Gaussian noise).

### P.2 Hypernetwork-Generated Transition Dynamics

At each timestep, hypernetwork $H$ computes mixing weights and updates its recurrent state:
$$w_{t+1}, h_{t+1} = H(r_t, h_t)$$

**Composite transition matrix** — linear combination of $K$ basis matrices:
$$V_{t+1} = \sum_{k=1}^{K} w_{t+1}^k V_k$$

**State transition**:
$$r_{t+1} = f(V_{t+1} r_t) + m$$

Where $f(\cdot) = \text{ReLU}$, $m$ is Gaussian noise.

### P.3 Dictionary of Basis Transition Matrices

-   $K = 5$ basis matrices $\{V_k\}_{k=1}^K$, each $V_k \in \mathbb{R}^{500 \times 500}$.
-   Randomly initialized, learned via gradient descent.

### P.4 Hypernetwork Architecture

-   **Input**: Recurrent hidden state $h_t$ (400-dimensional).
-   **MLP**: 4 hidden layers of 100 neurons each (bias + ReLU + batch normalization).
-   **Output**: 5 neurons (matching $K$) without activation function → mixing weights $w_{t+1} \in \mathbb{R}^K$.
-   Mixing weights are **unconstrained** (no softmax / simplex constraint).

### P.5 Inference (MAP via Bayesian Filtering)

**Initial step** ($t = 1$):
$$r_1 := \arg\min_{r_1} ||I_1 - U r_1||_2^2 + \lambda ||r_1||_1$$

**Subsequent steps** ($t > 1$):
$$r_t := \arg\min_{r_t} ||I_t - U r_t||_2^2 + ||r_t - \text{ReLU}(V_t r_{t-1})||_2^2 + \lambda ||r_t||_1$$

### P.6 Loss Function (Variational Free Energy)

$$\mathcal{L} = \sum_{t=1}^{T} \left(||I_t - U r_t||_2^2 + \lambda ||r_t||_1\right) + \sum_{t=1}^{T-1} ||r_{t+1} - \text{ReLU}(V_{t+1} r_t)||_2^2$$

**Predictive coding interpretation**:
-   $||I_t - U r_t||_2^2$: **Spatial prediction error** (top-down visual prediction vs. bottom-up sensory input).
-   $||r_{t+1} - \text{ReLU}(V_{t+1} r_t)||_2^2$: **Temporal prediction error** (hypernetwork-generated dynamics vs. actual next state).

### P.7 Training

Parameters updated via gradient descent: $U$ (SGD), $V_k$ (Adam), $H$ (Adam). States $r_t$ inferred first, then parameters updated.

---

## Paper Review Q: Rezaei-Shoshtari et al. (2023) — "Hypernetworks for Zero-Shot Transfer in Reinforcement Learning"

**Publication**: 2023
**Full Citation**: Rezaei-Shoshtari, S., Morissette, C., Hogan, F. R., Dudek, G., & Meger, D. (2023).

### Q.1 Core Concept: MDP Parameters → Policy Weights

HyperZero conceptualizes RL as a mapping from MDP parameters to near-optimal policy weights. Given a parameterized MDP family $M_i = (S, A, T_{\mu_i}, R_{\psi_i}, \gamma)$:

$$[\theta_i; \phi_i] = H_\Theta(\psi_i, \mu_i)$$

Where $\psi_i$ = reward parameters, $\mu_i$ = physics/dynamics parameters, $\Theta$ = hypernetwork parameters, $\theta_i$ = actor weights, $\phi_i$ = critic weights.

### Q.2 Dual-Objective Loss

**Prediction loss** (supervised from pre-collected optimal data):
$$\mathcal{L}_{pred.}(\Theta) = \mathbb{E}_{(\psi_i, \mu_i, s, a^*, q^*) \sim D} \left[(\hat{Q}_{\phi_i}(s, a^*) - q^*)^2\right] + \mathbb{E}_{(\psi_i, \mu_i, s, a^*) \sim D} \left[(\hat{\pi}_{\theta_i}(s) - a^*)^2\right]$$

**TD regularization loss** (enforces Bellman consistency):
$$\mathcal{L}_{TD}(\Theta) = \mathbb{E}_{(\psi_i, \mu_i, s, a^*, s', r, q^*) \sim D} \left[(r + \gamma \hat{Q}_{\phi_i}(s', \bar{a}') - q^*)^2\right]$$

Where $\bar{a}' = \hat{\pi}_{\theta_i}(s')$ (gradients stopped). This moves target estimates toward ground-truth (inverse of standard RL), ensuring generated actor-critic networks remain Bellman-consistent.

### Q.3 Meta-Training Procedure

1.  **Data collection**: Train independent TD3 agents on sampled MDP instances $M_i$. Collect rollouts $\tau_i^*$ into offline dataset $D$.
2.  **Hypernetwork optimization**: Sample mini-batch from $D$. Generate $[\theta_i; \phi_i] = H_\Theta(\psi_i, \mu_i)$. Update $\Theta$ via gradient descent on $\mathcal{L}_{pred.} + \mathcal{L}_{TD}$.

### Q.4 Zero-Shot Transfer

At test time, given novel $\psi_{test}, \mu_{test}$: single forward pass through $H_\Theta$ → generates $\theta_{test}$ → policy $\hat{\pi}_{\theta_{test}}(a|s)$ deployed directly. **No gradient updates, fine-tuning, or environment interactions** on target task.

### Q.5 Physics Context

Reward $\psi$: Desired speed of motion (positive/negative velocities). Dynamics $\mu$: Morphology parameters (torso length, finger length) — implicitly alters weight/inertia in physics engine. Uniformly sampled from defined prior distributions.

---

## Paper Review R: Schöpf et al. (2022) — "Hypernetwork-PPO for Continual Reinforcement Learning"

**Publication**: 2022
**Full Citation**: Schöpf, E., Hollenstein, J., Saveriano, M., Rodríguez-Sánchez, A., & Piater, J. (2022).

### R.1 Architecture: HN-PPO

Two variants:
-   **HN-PPO**: Hypernetwork generates both actor and critic weights.
-   **HN-PPO+fc**: Hypernetwork generates actor only; critic is a standard MLP re-initialized per task.

### R.2 Task Conditioning

Task-incremental setting. Task identity $C$ represented as a trainable embedding vector $t \in \mathbb{R}^8$.

**Weight generation**:
$$\Theta_t = h(t, \Theta_h)$$

Where $h$ is the hypernetwork, $\Theta_h$ are hypernetwork parameters, $\Theta_t$ are generated actor/critic parameters.

### R.3 Continual Learning via Functional Regularization

At the start of training a new task, generated parameters $\Theta_t$ are recorded for all previously learned task embeddings. During new-task training, an $L_2$ penalty prevents drift:

$$\mathcal{L}_{reg} = \beta \frac{1}{T-1} \sum_{t=0}^{T} ||\Theta_t - \Theta_{t,new}||_2^2$$

Where $\beta$ scales regularization strength, $\Theta_t$ are stored snapshots, $\Theta_{t,new}$ are current hypernetwork outputs for old task embeddings.

### R.4 PPO Loss with Hypernetwork

**Standard PPO clipped surrogate**:
$$L_t^{clip}(\theta) = \mathbb{E}_t \left[\min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t, \; \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right)\right]$$

**Total loss**:
$$L_t^{total}(\theta) = L_t^{clip}(\theta) + c_v L_t^{vf}(\theta) + c_e S_{\pi_\theta}(s_t)$$

Hypernetwork parameters $\Theta_h$ updated by differentiating through this total loss + $\mathcal{L}_{reg}$.

### R.5 Architecture Details

-   **Target networks** (actor/critic): MLP, 2 hidden layers × 64 neurons, tanh activations. Actor: 6-dim output. Critic: 1-dim.
-   **Hypernetwork**: MLP, 2 hidden layers × 640 neurons, ReLU activations. Multi-head linear output layer (no nonlinearity). Actor: 7 heads (weight/bias tensors + std dev). Critic: 6 heads.
-   **Scaling**: Model size constant — only an 8-dim embedding added per new task. Achieves remembering score of 1.00 (zero catastrophic forgetting) vs. severe forgetting in sequential PPO finetuning.

---

## Paper Review S: Ichikawa & Kaneko (2024) — "Bayesian Inference is Facilitated by Modular Neural Networks with Different Time Scales"

**Publication**: 2024
**Full Citation**: Ichikawa, K. & Kaneko, K. (2024).

### S.1 Architecture: Multi-Timescale Modular RNN

Vanilla continuous-time leaky RNN, ReLU activation, $N = 200$ total neurons:

-   **Main Module (Fast)**: $N_m = 150$ neurons. Receives sensory input, projects to output.
-   **Sub-Module (Slow)**: $N_s = 50$ neurons. **No direct input/output connections** — topologically insulated from environment.

### S.2 Module Dynamics

**Fast module** ($\alpha_m = 1$, rapid integration):
$$x_m(t+1) = (1 - \alpha_m) x_m(t) + \alpha_m \text{ReLU}(W_{in} u(t) + W_{main} x_m(t) + W_{s \to m} x_s(t)) + \sqrt{\alpha_m} \xi_m$$

**Slow module** ($\alpha_s \approx 0.1$, slow integration):
$$x_s(t+1) = (1 - \alpha_s) x_s(t) + \alpha_s \text{ReLU}(W_{sub} x_s(t) + W_{m \to s} x_m(t)) + \sqrt{\alpha_s} \xi_s$$

Where $\xi \sim \mathcal{N}(0, 0.05^2)$.

**Readout** (from main module only): $y(t) = W_{out} x_m(t)$

### S.3 Bayesian Inference Implementation

Optimal Bayesian estimate for signal from generator $\mathcal{N}(\mu_g, \sigma_g^2)$ with observation noise $\sigma_l^2$:

$$y_{opt} = \frac{\sigma_g^2}{\sigma_g^2 + \sigma_l^2} s + \frac{\sigma_l^2}{\sigma_g^2 + \sigma_l^2} \mu_g$$

-   **Likelihood** (fast module): Processes instantaneous noisy input $u(t)$.
-   **Prior** (slow module): Integrates history to estimate generator parameters. PCA reveals orthogonal axes encoding $\mu_g$ and $\sigma_g$ independently.
-   **Integration**: $W_{s \to m} x_s(t)$ shifts the main module's representation toward optimal Bayesian estimate.

### S.4 Training

-   **Loss**: MSE between readout and ground truth: $L = \frac{1}{T} \sum_t (y(t) - y_{true}(t))^2$
-   **Optimizer**: BPTT with Adam, lr = 0.001, weight decay = 0.0001, batch size = 50, 6000 iterations.
-   **Emergent structure**: If time constants $\alpha_i$ are made learnable, gradient descent **spontaneously** evolves a slow insulated sub-module.

### S.5 Task Setup

-   Hidden generator: $y_{true} \sim \mathcal{N}(\mu_g, \sigma_g^2)$ with sudden shifts ($p_t = 0.03$), $\mu_g \in [-0.5, 0.5]$, $\sigma_g \in [0, 0.8]$.
-   Observation: $s \sim \mathcal{N}(y_{true}, \sigma_l^2)$.
-   Input encoding: Probabilistic Population Code (PPC), 100 neurons with Poisson-distributed activity.

---

## Paper Review T: Perez et al. (2018) — "FiLM: Visual Reasoning with a General Conditioning Layer"

**Paper**: Perez, Strub, de Vries, Dumoulin, & Courville (2018)
**Core idea**: Feature-wise Linear Modulation (FiLM) — a general-purpose conditioning mechanism where one network (the "conditioning network") predicts per-feature affine transformation parameters (scale γ and shift β) that modulate intermediate feature maps of another network (the "modulated network"). Originally applied to visual question answering (VQA), but the mechanism is a general-purpose neural modulation primitive applicable to any architecture.

### T.1 Core FiLM Equation

The fundamental operation is a per-channel affine transformation of feature maps:

$$\text{FiLM}(F_{i,c} \mid \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}$$

Where:
-   $F_{i,c}$ = activation of the $c$-th feature map at the $i$-th network layer (a 2D spatial map for CNNs, or a scalar for MLPs)
-   $\gamma_{i,c}$ = learned scaling parameter for feature $c$ at layer $i$ (multiplicative modulation)
-   $\beta_{i,c}$ = learned shift parameter for feature $c$ at layer $i$ (additive modulation)
-   Modulation is **per-channel** — all spatial locations within a feature map share the same γ and β

### T.2 Conditioning Network (Generates γ, β)

In the VQA application, a **GRU-based language encoder** processes the question and produces modulation parameters:

1.  **Input**: Question tokens $q = (q_1, q_2, \ldots, q_L)$
2.  **GRU encoder**: Hidden state $h_t = \text{GRU}(q_t, h_{t-1})$, hidden size = 4096
3.  **Final hidden state**: $h_L$ (last GRU output) serves as the conditioning vector
4.  **Linear projection per block**: For each FiLM-ed ResBlock $i$:
    $$(\gamma_i, \beta_i) = W_i h_L + b_i$$
    where $\gamma_i, \beta_i \in \mathbb{R}^{128}$ (one scalar per feature map), $W_i \in \mathbb{R}^{256 \times 4096}$

**Key design choice**: Each ResBlock gets its **own** linear projection layer, so different layers can receive different modulation signals from the same conditioning vector.

### T.3 Modulated Network (FiLM-ed ResBlock Architecture)

The modulated network is a CNN with **4 FiLM-ed ResBlocks**, each containing 128 feature maps:

```
FiLM-ed ResBlock(x, γ, β):
    h = Conv1×1(x)          # 1×1 conv, 128 channels
    h = ReLU(h)
    h = Conv3×3(h)          # 3×3 conv, 128 channels
    h = BatchNorm(h)        # Batch normalization
    h = FiLM(h | γ, β)     # Apply γ·h + β per-channel
    h = ReLU(h)
    return h + x            # Residual connection
```

**Critical placement**: FiLM is applied **after** batch normalization and **before** the final ReLU activation. This means:
-   BN normalizes features to zero mean, unit variance
-   FiLM then re-scales (γ) and re-centers (β) per-channel
-   This effectively lets FiLM **override** the normalization statistics with task-conditioned values

### T.4 Full Pipeline

```
Image (224×224×3)
  → Conv(128, 3×3, stride 2, padding 1) + BN + ReLU     # Extract visual features
  → FiLM-ResBlock₁(γ₁, β₁)                              # Modulated by question
  → FiLM-ResBlock₂(γ₂, β₂)
  → FiLM-ResBlock₃(γ₃, β₃)
  → FiLM-ResBlock₄(γ₄, β₄)
  → Global Max Pooling
  → MLP classifier → answer
```

Total modulation parameters per question: $4 \times 2 \times 128 = 1024$ scalars (4 blocks × {γ, β} × 128 channels).

### T.5 Training Details

-   **End-to-end training**: Both conditioning network (GRU) and modulated network (CNN + ResBlocks) are trained jointly
-   **Optimizer**: Adam, learning rate $3 \times 10^{-4}$
-   **Loss**: Cross-entropy over answer vocabulary
-   **No pre-training**: The CNN is trained from scratch (not a pre-trained backbone)

### T.6 Relevance as Neuromodulation Primitive

FiLM is architecturally analogous to neuromodulation:
-   **γ (gain modulation)** ↔ neuromodulatory gain control (e.g., norepinephrine adjusting neural responsiveness)
-   **β (bias/shift)** ↔ tonic baseline shifts (e.g., serotonin shifting activation thresholds)
-   **Conditioning network** ↔ modulatory nuclei (e.g., locus coeruleus, raphe nuclei) that broadcast context-dependent signals
-   **Per-channel modulation** ↔ receptor-type-specific effects (different neurotransmitter receptors on different neuron populations)
-   **Layer-specific γ, β** ↔ laminar specificity of neuromodulatory innervation

FiLM subsumes several special cases:
-   γ = 1, β = learned → **Conditional bias** (additive-only modulation)
-   γ = learned, β = 0 → **Gain-only modulation** (multiplicative-only)
-   γ = 0 or 1 (binary) → **Feature gating / selection**

---

## Paper Review U: Tschantz et al. (2023) — "Hybrid Predictive Coding: Inferring, Fast and Slow"

**Paper**: Tschantz, Baltieri, Seth, & Buckley (2023)
**Core idea**: Hybrid Predictive Coding (HPC) unifies fast amortized inference (feedforward recognition model) with slow iterative inference (recurrent prediction-error minimization) within a single hierarchical generative model. A learned bottom-up encoder provides rapid initial beliefs, which are then refined by top-down iterative gradient descent on variational free energy. Both pathways are trained jointly via local Hebbian-like rules on a shared free energy objective.

### U.1 Variational Free Energy Objective

HPC frames perception as minimization of variational free energy $F$, an upper bound on surprise:

$$F = \mathbb{E}_{q_\lambda(z)} \left[ \ln q_\lambda(z) - \ln p(z, x) \right] \geq D_{KL}\left[ q_\lambda(z) \| p(z|x) \right]$$

Assuming Gaussian distributions:
-   Approximate posterior: $q_\lambda(z) = \mathcal{N}(z; \mu, \sigma^2)$
-   Prior: $p(z) = \mathcal{N}(z; \bar{\mu}, \sigma_p^2)$
-   Likelihood: $p(x|z) = \mathcal{N}(x; f_\theta(z), \sigma_l^2)$

The free energy decomposes into precision-weighted prediction errors:

$$F(\mu, x) = \frac{1}{2\sigma_l} \epsilon_l^2 + \frac{1}{2\sigma_p} \epsilon_p^2 + \frac{1}{2} \ln(\sigma_l \cdot \sigma_p)$$

Where $\epsilon_l$ is the sensory (likelihood) prediction error and $\epsilon_p$ is the prior prediction error.

### U.2 Hierarchical Architecture ($L$ layers)

Each layer $i$ maintains a state variable $\mu_i$ (mode of approximate posterior belief).

**Generative model (top-down)**: Higher layers predict lower layers:
$$\epsilon_i = \mu_{i-1} - f_{\theta_i}(\mu_i)$$

where $f_{\theta_i}$ is the generative function with parameters $\theta_i$.

**Recognition model (bottom-up / amortized)**: Lower layers predict higher layers:
$$\mu_i = f_{\phi_i}(\mu_{i-1}), \quad \mu_0 = f_{\phi_0}(x)$$

where $f_{\phi_i}$ is the amortized encoder with parameters $\phi_i$.

**Amortized prediction error** (discrepancy between feedforward guess and iteratively refined belief $\mu^*$):
$$\epsilon_i^\phi = \mu_{i+1}^* - f_{\phi_i}(\mu_i)$$

### U.3 Hybrid Inference Procedure

**Phase 1 — Fast amortized initialization**: At stimulus onset, sensory data $x$ propagates up the hierarchy via the recognition model $f_{\phi_i}(\cdot)$ to initialize all $\mu_i$ for $i \in \{0, \ldots, L-1\}$.

**Phase 2 — Slow iterative refinement**: Starting from the amortized initialization, beliefs $\mu_i$ are updated via gradient descent on free energy:

$$\dot{\mu}_i = -\kappa \left( \epsilon_p - \frac{\partial f_{\theta_i}(\mu_i)}{\partial \mu_i}^\top \epsilon_i \right)$$

where:
-   $\epsilon_p = \mu_i - f_{\theta_{i+1}}(\mu_{i+1})$ = error from the layer above (prior prediction error)
-   $\epsilon_i$ = error from the layer below (likelihood prediction error)
-   $\kappa$ = integration step size

This runs for $N$ iterations (or until convergence) to reach refined equilibrium $\mu^*$.

### U.4 Parameter Learning (Weight Updates)

After iterative convergence to $\mu^*$, synaptic weights update via local Hebbian-like rules:

**Generative parameters** (minimize generative prediction errors):
$$\dot{\theta}_i = -\alpha \left( \epsilon_i \cdot f_{\theta_i}(\mu_i)^\top \right)$$

**Amortized parameters** (learn to predict refined beliefs):
$$\dot{\phi}_i = -\alpha \left( \epsilon_i^\phi \cdot f_{\phi_i}(\mu_i)^\top \right)$$

The recognition model learns to approximate the result of iterative inference, so over training it produces increasingly accurate initializations, reducing the number of iterative steps needed.

### U.5 Neural Network Architecture (MNIST implementation)

-   **Hierarchy**: $L = 4$ layers
-   **Dimensions**: 784 (sensory) → 500 → 500 → 10 (label/prior)
-   **Generative activations** $f_\theta$: tanh for all layers except lowest (linear)
-   **Amortized activations** $f_\phi$: tanh for all layers except highest
-   **Weight normalization**: Applied to generative parameters for stability
-   **Optimizer**: Adam, $\alpha = 0.01$
-   **Inference step**: $\kappa = 0.01$, max $N = 100$ iterations per sample

### U.6 Biological Correspondence

| HPC Component | Neural Correlate |
|---|---|
| Amortized feedforward sweep | Initial cortical feedforward sweep (~100-150ms post-stimulus) |
| Iterative refinement | Recurrent cortical processing (>150ms) |
| Generative connections $\theta$ | Top-down cortical projections |
| Recognition connections $\phi$ | Bottom-up cortical projections |
| Prediction errors $\epsilon$ | Superficial pyramidal cell activity |
| Beliefs $\mu$ | Deep pyramidal cell activity |
| Precision weights $1/\sigma$ | Neuromodulatory gain control (e.g., attention) |

### U.7 Relevance to Neuromodulation

-   **Precision weighting** ($1/\sigma_l$, $1/\sigma_p$) directly maps to neuromodulatory gain modulation — a modulatory signal that scales the influence of prediction errors without changing their content
-   **Adaptive computation**: The framework naturally adjusts processing depth (number of iterative steps) based on stimulus ambiguity, analogous to arousal-modulated processing depth
-   **Active inference extension**: HPC naturally extends to active inference where actions minimize expected free energy, with precision over action policies modulated by dopaminergic signals

---

## Paper Review V: Wang et al. (2024) — "Neuromodulated Meta-Learning" (NeuronML)

**Paper**: Wang, Guo, Qiang, Li, Zheng, Xiong, & Hua (2024)
**Core idea**: NeuronML introduces a Flexible Network Structure (FNS) to meta-learning by learning a continuous structural mask $M$ that dynamically generates task-specific subnetworks via element-wise multiplicative gating. Unlike static-architecture meta-learning (e.g., MAML), NeuronML selectively activates distinct neuron populations per task, analogous to how biological neuromodulation routes information through specialized cortical regions. The mask is optimized via bi-level optimization with three biologically-inspired constraints: frugality, plasticity, and sensitivity.

### V.1 Flexible Network Structure (Multiplicative Mask)

Given base meta-learning parameters $\theta$, NeuronML introduces a learnable mask $M$ of identical dimensionality. The effective parameters for a forward pass are:

$$\theta_M \leftarrow M \odot \theta$$

where $\odot$ is the Hadamard (element-wise) product. Each element of $M$ represents the **activation probability** of its corresponding neuron — $M[i] = 0$ explicitly deactivates neuron $i$.

### V.2 Bi-Level Optimization

**First level (inner loop — weight adaptation)**: With mask $M$ fixed, adapt to task $\tau_i$ via gradient descent on support set $D_i^s$:

$$\theta_M^i \leftarrow \theta_M - \alpha \nabla_\theta L_{\text{weight}}(D_i^s, \theta_M)$$

Outer weight objective over query sets:

$$\arg\min_\theta \frac{1}{N_{tr}} \sum_{i=1}^{N_{tr}} L_{\text{weight}}(D_i^q, \theta_M^i)$$

**Second level (outer loop — structure optimization)**: With adapted weights fixed, optimize mask $M$ over all task data:

$$\arg\min_M \frac{1}{N_{tr}} \sum_{i=1}^{N_{tr}} L_{\text{structure}}(D_i, \theta_M^i)$$

### V.3 Structure Constraints

The structure loss combines three differentiable constraints:

$$L_{\text{structure}}(D_i, \theta_M^i) = \lambda_{fr} L_{fr}(\theta_i) + \lambda_{pl} L_{pl}(\theta_i, \theta_j) + \lambda_{se} L_{se}(\theta_i)$$

**Frugality** (sparsity — activate only necessary neurons):

$$L_{fr}(\theta_i) = \|\theta_i\|_1 \quad \text{s.t.} \quad \|\theta_i\|_1 \leq \max\{C, \gamma \cdot d \cdot \log(N_i / d)\}$$

where $C$ is a constant, $\gamma$ a scaling factor, $d$ the parameter dimensionality, and $N_i$ the task sample size. Uses $\ell_1$ relaxation of the NP-hard $\ell_0$ norm.

**Plasticity** (structural diversity across tasks — prevent collapse to static subnetwork):

$$L_{pl}(\theta_i, \theta_j) = \sum_{j \neq i} \sum_{\omega \in \theta_i} \mathbb{I}(\theta_i[\omega], \theta_j[\omega]) \cdot p_\omega$$

where $\mathbb{I}(\cdot)$ is an indicator function returning 1 if both tasks $\tau_i$ and $\tau_j$ activate neuron $\omega$, and $p_\omega$ is the **historical importance** (Hebbian-inspired softmax):

$$p_\omega = \frac{e^{\beta L_\omega}}{\sum_k e^{\beta L_k}}$$

where $L_\omega$ is the loss change caused by neuron $\omega$.

**Sensitivity** (ensure active neurons are maximally informative):

$$L_{se}(\theta_i) = \sum_{\omega=1}^{N_\omega} -\log\left(\frac{s(\omega)}{S}\right) \cdot \theta_i[\omega]$$

where the sensitivity score is the gradient magnitude: $s(\omega) = \left|\frac{\partial L(\theta_i, \tau_i)}{\partial \theta_i[\omega]}\right|$ and $S$ is the total sensitivity.

### V.4 Training Algorithm

```
Algorithm: NeuronML
Input: Task distribution P(T), model f_{θ_M} with θ_M = M ⊙ θ
Output: Trained model f_θ

1: while not done do
2:   Sample N_tr tasks {τ_i}_{i=1}^{N_tr} ~ P(T)
3:   for i = 1 to N_tr do
4:     Split τ_i data into support D_i^s and query D_i^q
5:     Compute L_weight(D_i^s, θ_M)
6:     Inner-loop adapt: θ_M^i ← θ_M - α∇_θ L_weight(D_i^s, θ_M)
7:     Compute query loss L_weight(D_i^q, θ_M^i)
8:     Compute structure loss L_structure(D_i, θ_M^i)
9:   end for
10:  Update θ using aggregated L_weight(D_i^q, θ_M^i)
11:  Update M using aggregated L_structure(D_i, θ_M^i)
12: end while
```

### V.5 Architecture Integration

NeuronML is **architecture-agnostic** — the mask $M$ is embedded directly over parameter tensors of any standard backbone. Validated on Conv4, VGG16, ResNet18/50/101, and DenseNet.

### V.6 Neuromodulatory Interpretation

| NeuronML Component | Biological Analogue |
|---|---|
| Structural mask $M$ | Neuromodulatory gating (selective neuron activation/deactivation) |
| Frugality constraint | Metabolic efficiency / sparse coding |
| Plasticity constraint | Task-dependent circuit reconfiguration |
| Sensitivity constraint | Hebbian relevance filtering |
| Hadamard product $M \odot \theta$ | Gain modulation of synaptic efficacy |
| Per-task subnetworks | Cortical specialization / functional segregation |

---

## Paper Review W: Wang et al. (2025) — "NEST: A Neuromodulated Small-world Hypergraph Trajectory Prediction Model for Autonomous Driving"

**Paper**: Wang, Chen, Wen, & Pan (2025)
**Core idea**: NEST uses artificial neuromodulation to dynamically adapt the topology of a small-world interaction hypergraph for multi-agent trajectory prediction in autonomous driving. Two neuromodulatory signals ($\alpha$, $\beta$) — computed from traffic density and clustering statistics — globally control hyperedge formation thresholds, allowing the graph structure to fluidly shift between dense local and sparse long-range interaction patterns depending on environmental conditions.

### W.1 Architecture Overview

NEST predicts future trajectories $Y = [Y_1, \ldots, Y_K]$ (K modal hypotheses) for traffic agents given historical data $X = [X_0, \ldots, X_n]$ (positions, velocities, accelerations over $t_h$ steps) and an HD map $M$. Four modules:

1.  **Hypergraph Forming**: Neuromodulator + Small-world Network → Interaction Hypergraph $G = (V, E)$
2.  **Hypergraph Pooling**: Extracts interaction features $F_i$ from $G$
3.  **Context Fusion**: Cross-attention fusing lane features $F_l$ (from HD map) with interaction features $F_i$
4.  **Multi-modal Predictor**: Decodes fused context into $K$ trajectory hypotheses with probabilities

### W.2 Neuromodulatory Signals ($\alpha$, $\beta$)

Two MLP-computed scalars control graph topology:

**Threshold $\alpha$** (from clustering coefficient distribution):
$$\alpha = \text{Sigmoid}\left(\text{Average}\left(\text{MLP}(\text{Coefficient Distribution})\right)\right)$$

**Connection probability $\beta$** (from spatial density of historical data):
$$\beta = \text{Sigmoid}\left(\text{Average}\left(\text{MLP}(\text{Density Feature})\right)\right)$$

Both are continuous values in $(0, 1)$ driven by external traffic stimuli (agent features $F_a$).

### W.3 Neuromodulated Hyperedge Formation

The hyperedge set is dynamically constructed:

$$E = \Omega(V, \alpha, \beta)$$

where $\Omega$ is the Small-world Network generator. For each pair of nodes, the local clustering coefficient $C$ is evaluated against the modulated threshold:

-   If $C > \alpha$: deterministic edge ($C_{i,j} = 1$)
-   If $C \leq \alpha$: probabilistic edge with probability $\beta$

This creates an **adaptive topology**: in dense traffic ($\alpha$ low), more deterministic edges form; in sparse traffic ($\alpha$ high), connections become probabilistic and long-range via $\beta$.

### W.4 Intention and Trajectory Generation

Agent interaction information: $I_a = \sum_{V_i \in E_j} \lambda_i V_i$ (learnable weights $\lambda_i$).

Intention via Gumbel-Softmax (handling discrete modal uncertainty):
$$I_i = \sigma\left(\frac{M_i(I_a) + \xi}{\tau}\right)$$

where $\sigma$ is softmax, $M_i$ is the intention encoder, $\xi \sim \text{Gumbel}(0, 1)$, and $\tau$ is temperature.

Additional streams:
-   Personality: $I_p = M_p(\sum_{V_i \in V} \lambda_i V_i)$
-   Willingness: $I_w = M_w(I_a)$

Context fusion via cross-attention:
$$F_c = \text{Attn}(Q = F_i, K = F_l, V = F_l)$$

### W.5 Output Parameterization

Each of the $K$ trajectory modes outputs Laplace-distributed predictions per timestep $t$:

$$Y_i^t = [x_i^t, y_i^t, b_{i,x}^t, b_{i,y}^t]$$

where $(x, y)$ are spatial coordinates and $(b_x, b_y)$ are Laplace scale parameters representing kinematic uncertainty.

### W.6 Neuromodulatory Interpretation

| NEST Component | Biological Analogue |
|---|---|
| Threshold $\alpha$ | Neural excitability threshold (modulated by tonic neuromodulator levels) |
| Connection probability $\beta$ | Stochastic synaptic transmission probability |
| Hyperedge formation $\Omega(V, \alpha, \beta)$ | Neuromodulatory gating of functional connectivity |
| Traffic density → $\beta$ | Arousal-driven modulation of network connectivity |
| Clustering → $\alpha$ | Local circuit regulation based on neighborhood statistics |
| Adaptive topology shift | State-dependent reconfiguration of functional networks |

### W.7 Relevance to Neuromodulation

-   **Structural modulation**: Unlike most neuromodulatory algorithms that modulate activations or weights, NEST modulates **graph topology** — which connections exist at all
-   **Environment-driven**: The modulatory signals are computed from environmental statistics (traffic density, clustering), analogous to how brainstem nuclei modulate cortical connectivity based on arousal and environmental demands
-   **Continuous control**: $\alpha$ and $\beta$ are continuous (sigmoid-bounded), providing graded rather than binary modulation
