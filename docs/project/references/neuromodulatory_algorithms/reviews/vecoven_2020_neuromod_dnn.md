---
title: "Introducing neuromodulation in deep neural networks to learn adaptive behaviours"
authors: "Nicolas Vecoven, Damien Ernst, Antoine Wehenkel, Guillaume Drion"
year: 2020
venue: "PLOS ONE 15(1):e0227922"
slug: "vecoven_2020_neuromod_dnn"
source_pdf: "sources/Vecoven et al. 2020 - Introducing neuromodulation in deep neural networks to learn adaptive behaviours.pdf"
topic: "neuromodulatory_algorithms"
---

## Plain-English entry point

This paper asks how to make a deep neural network change its own input/output behaviour on the fly when it is dropped into a new task, the way an animal can switch strategies when its environment changes. The authors' answer is a two-network architecture they call **NMN** ("Neuro-Modulated Network"). One subnetwork, the **main network**, does the usual job of mapping observations to actions or value predictions. A second subnetwork, the **neuromodulatory network**, watches a "context" stream (what has happened so far in the episode: past observations, actions and rewards) and emits a real-valued vector **z**. That vector reshapes the activation functions inside the main network — specifically, it rescales their slope and shifts their bias on every time-step. So instead of switching synaptic weights (slow, weight-update style) the network's *operating point* changes from moment to moment.

Why it matters: this is a concrete, very lightweight implementation of "cellular neuromodulation" inside an otherwise standard feed-forward network. The number of extra parameters scales with the number of *neurons* rather than the number of *connections*, so the trick is cheap. The authors test it inside a meta-reinforcement-learning loop on three custom navigation benchmarks against a vanilla RNN baseline. NMNs learn faster, plateau higher, are more stable across random seeds, and — on the simplest benchmark — come within 3 % of a Bayes-optimal policy. They also show, by freezing and unfreezing **z** mid-episode, that **z** behaves like a learned task-descriptor.

## Section-by-section backbone

### Abstract
Animals continuously re-tune intentions and attention via cellular neuromodulation, a mechanism distinct from synaptic plasticity. The authors graft an analogue of this mechanism onto a deep network and show it adapts better than a recurrent baseline on three meta-RL navigation benchmarks.

### 1. Introduction
Deep nets generalise poorly across tasks. In biology, *neuromodulation* — diffuse biochemical signals (dopamine, serotonin, acetylcholine, noradrenaline, neuropeptides) — dynamically retunes neuron input/output behaviour in a context-dependent way (refs [1–4]). This is qualitatively different from synaptic plasticity. The authors propose to capture this idea in a DNN.

### 2. NMN architecture
The architecture has two interacting subnetworks. The **neuromodulatory network** maps a context vector $c$ to a neuromodulatory signal $z \in \mathbb{R}^k$. The **main network** is a feed-forward DNN whose every activation is replaced by a "neuromodulation-capable" version

$$\sigma_{NMN}(x, z; w_s, w_b) = \sigma\big(z^\top (x\,w_s + w_b)\big)$$

with $w_s, w_b \in \mathbb{R}^k$ per-neuron parameters governing scale and offset. The signal $z$ is *shared* across all main-network neurons. Compared to a hypernetwork (which generates synaptic weights), the parameter count scales linearly in the number of *neurons*, not *connections*. Related work flagged: differentiable plasticity (Miconi et al. [5]), backpropamine (Miconi et al. [6]), hypernetworks (Ha et al. [7]), learned activation functions [8,9].

### 3. Experiments — 3.1 Setting
The architecture is evaluated in meta-RL as defined by Wang et al. [12]: a distribution $D$ over MDPs; each new episode draws a new MDP $M$; the agent has $T$ steps to figure out and exploit it, with information limited to the per-step trajectory. Goal: maximise expected discounted return over all episodes.

### 3.2 Training
A2C with generalised advantage estimation (GAE, ref [13]) and proximal policy update (PPO-style clipping, ref [14]). Both **actor** and **critic** are modelled as NMNs (one each, no parameter sharing — the authors argue the modulatory signals for policy vs. value may differ). For comparison, an RNN baseline matches NMN to within 2 % parameter count. Context $c_t = h_t \setminus x_t$ (history minus current observation) feeds the neuromodulatory network; the main network input is $x_t$. The neuromodulatory network is recurrent because $h_t$ grows over time. Main-network hidden layers use **saturated ReLU**: $\sigma(x)=\min(1, \max(-1,x))$ with the final layer linear, all neuromodulated.

### 3.3 Benchmarks
Three custom continuous-control meta-RL benchmarks.
- **Benchmark 1**: 1-D pursuit of a target whose observation is biased by an unknown offset $\alpha \sim U[-10, 10]$. Reward $r_t = 10$ if $|a_t - p_t| < 1$ (and target resampled), else $r_t = -|a_t - p_t|$.
- **Benchmark 2**: 2-D navigation to a target at $(\alpha_1, \alpha_2)$ through a wind cone of direction $\alpha_3 \sim U[-\pi, \pi]$. Per-step reward $-0.2$, $+100$ at target (then teleport-reset).
- **Benchmark 3**: 2-D navigation with two targets at $(\alpha_1, \alpha_2)$ and $(\alpha_3, \alpha_4)$, where $\alpha_5$ is a Bernoulli that flips which is rewarding (+100) vs. punishing ($-50$).

### 4. Results — Learning
NMNs learn faster (in episodes) and reach better terminal returns than RNNs on all three benchmarks; variance across 15 seeds is far smaller. On benchmark 1, NMNs reach 4534 expected return after 20 000 episodes vs. Bayes-optimal 4679 — a 3 % gap.

### 4. Results — Adaptation
The temporal evolution of $z$ is informative. Early in an episode $z$ is roughly $\alpha$-independent (the agent is exploring), then converges to an $\alpha$-dependent value (the agent has identified the task) and stays roughly time-constant from then on. Some main-network neurons have scale factors that *cross zero* between positive and negative across $\alpha$, which means the activation slope flips sign — these neurons effectively reverse polarity by task. Other neurons go silent (scale ≈ 0) for some tasks, equivalent to sub-network pruning per task.

A freeze/unfreeze experiment on benchmark 3: (a) freezing $z$ at the initial value yields the agent's exploration strategy; (b) unfreezing lets it solve the task; (c) re-freezing $z$ at a task-adapted value preserves most of the performance; (d) switching which target is rewarding while $z$ stays frozen sends the agent to the wrong target (no adaptation possible without updating $z$); (e) unfreezing re-fixes the problem. Conclusion: $z$ carries the task identity (mostly).

### 4. Results — Robustness
Replacing sReLU with sigmoid degrades RNNs more than NMNs (benchmark 2 especially). Varying main-network depth (0/1/4 hidden layers) on benchmark 1: NMNs are roughly insensitive to depth, RNNs are not.

### 5. Conclusions
NMNs improve adaptation on three meta-RL benchmarks. Directions: extend to supervised meta-learning / few-shot (where it would resemble a Conditional Neural Process [11]); use richer parametric activation functions or spiking neurons; share activation parameters per layer (especially in conv layers, where it would correspond to scaling filters); generate a per-layer rather than global modulatory signal; explore whether dissimilar task families would induce more separated sub-networks via scale-zero pruning.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** In a standard neural network you compute, for each neuron, $y = \sigma(x w + b)$ where $\sigma$ is a fixed nonlinearity (ReLU, sigmoid, etc.) and $w, b$ are weights and bias. Vecoven et al. notice that biological neurons don't keep a fixed input/output curve — a chemical signal can momentarily change their gain or threshold. They mimic this by adding a small extra "controller" network that produces a vector $z$, and they let $z$ rescale the slope and shift the bias of every activation function in the main network at every time-step. The controller takes as input everything the agent has seen so far (its history of states, actions, rewards), so $z$ is a learned summary of "what task am I in?". Crucially, the controller's signal does *not* change the network's weights — those stay fixed after training. What changes is the network's *operating point* on each step.

**The experimental setup.** They drop this architecture into a meta-RL setup: every episode the environment changes in a hidden way (a position-bias parameter, a wind direction, or which of two targets gives a reward). The agent has one episode to figure out the change and exploit it. Both the policy (actor) and the value estimator (critic) are NMN networks. The baseline is the same task with an ordinary RNN.

**The result.** NMNs learn faster, get higher final reward, and are way more stable across random seeds. They also visibly encode the task identity: plot the controller's output $z$ versus the hidden task variable $\alpha$ over time and you watch the agent's "belief" sharpen as it gathers data. The cleanest demonstration: freeze $z$ once the agent has adapted, then secretly swap which target is rewarding — the agent dutifully sails toward the wrong (now-punishing) target, because without updating $z$ it cannot re-adapt. Unfreezing $z$ rescues it.

**Worked example.** On benchmark 1 a Bayes-optimal agent (one that does exact Bayesian inference on $\alpha$ and acts optimally given the posterior) gets expected total reward 4679 over an episode. The NMN reaches 4534 — within 97 % of the Bayes optimum. The ordinary RNN baseline is meaningfully worse.

## Phase 2 — Graduate-level deep dive

### The neuromodulation-capable activation

Let $\sigma : \mathbb{R} \to \mathbb{R}$ be any activation. Vecoven replaces it with

$$\boxed{\;\sigma_{\text{NMN}}(x, z;\, w_s, w_b) \;=\; \sigma\!\big(\,z^{\top} \,(x\, w_s + w_b)\,\big)\;}$$

where $x \in \mathbb{R}$ is the pre-activation, $z \in \mathbb{R}^k$ is the neuromodulatory signal (vector, dimension $k$ is a free hyperparameter), and $w_s, w_b \in \mathbb{R}^k$ are per-neuron parameters trained with the rest of the network.

A useful unpacking. Let $u = x\, w_s + w_b \in \mathbb{R}^k$ — this is just the per-neuron parameters scaled by the input. Then $\sigma_{\text{NMN}} = \sigma(z^\top u)$. So:

- $z^\top w_s$ is the **effective slope** with respect to $x$: $\partial(z^\top u)/\partial x = z^\top w_s$.
- $z^\top w_b$ is the **effective bias** added to $x \cdot (z^\top w_s)$.

Concretely, $\sigma_{\text{NMN}}(x) = \sigma\big((z^\top w_s)\,x + (z^\top w_b)\big)$, which is exactly "slope $\times$ input + offset", but where both slope and offset are computed by *projecting* the modulatory vector $z$ onto a per-neuron learnable direction $(w_s, w_b)$. This is the mechanism by which the *same* shared $z$ controls *different* slopes and biases in different neurons.

### Parameter scaling

Number of NMN-specific parameters added: $2k$ per neuron in the main network (since each neuron has its own $w_s, w_b \in \mathbb{R}^k$). For a main network with $N$ neurons, that's $2kN$. Compare: a hypernetwork producing the *weights* of a layer with $N_{\text{in}}, N_{\text{out}}$ would need $O(k \cdot N_{\text{in}} N_{\text{out}})$ parameters — quadratic rather than linear in width.

### How $z$ is generated

$$z_t = f\big(c_t;\, \theta_{\text{neuromod}}\big), \qquad c_t = h_t \setminus x_t$$

where $h_t = [x_0, a_0, r_0, x_1, a_1, r_1, \ldots, x_{t-1}, a_{t-1}, r_{t-1}, x_t]$ is the full interaction history and $f$ is itself a recurrent network so it can ingest the growing history online. The main network input is $x_t$; the modulatory network input is the rest of the history. Note the explicit decoupling: state observation goes through the main network, context goes through the modulator.

### Activation choice in the main network

The authors specifically pick **saturated ReLU**:

$$\sigma(x) = \min(1, \max(-1, x))$$

except for the output layer ($\sigma(x)=x$). The bounded range $[-1, 1]$ is what makes the slope-scaling interpretation tight: $z^\top w_s$ moves the linear regime's slope; $z^\top w_b$ shifts where saturation kicks in. A negative effective slope literally inverts the neuron's polarity, which is what the authors observe in Fig. 7B for benchmark 1.

### Meta-RL objective

Following Wang et al. [12], the agent's objective averages over the MDP distribution $D$:

$$J(\pi) = \mathbb{E}_{M \sim D}\, \mathbb{E}_{\tau \sim \pi, M}\!\left[\sum_{t=0}^{T-1} \gamma^t\, r_t\right]$$

A2C with GAE [13] and PPO-style clipping [14] is used. The actor and the critic are independent NMNs (no shared modulator), motivated by the (intuitive but unproven) claim that the modulatory signal for action selection need not equal that for value estimation.

### Bayes-optimal benchmark-1 derivation (sketch)

For benchmark 1, $\alpha \sim U[-10, 10]$ and the observation at $t=0$ is $x_0 = p_0 + \alpha$ with $p_0 \sim U[-5-\alpha, 5-\alpha]$. The agent never directly observes $\alpha$; its posterior over $\alpha$ is updated whenever a positive reward is received (which informs $p_t$). The optimal policy is to (i) take an action that probes for $\alpha$ early, (ii) once the posterior is narrow enough, exploit by aiming at $\hat p_t = x_t - \mathbb{E}[\alpha \mid \text{history}]$. The supplementary material derives the expected return under this policy: 4679. The fact that NMN attains 4534 (97 %) suggests it has approximated near-Bayes-optimal posterior tracking through the dynamics of $z_t$.

### Per-neuron behavioural classes induced

Inspecting the trained network reveals three neuron classes:

1. **Polarity-flipping neurons**: $z^\top w_s$ changes sign across the task variable $\alpha$. The neuron computes a function and its negation, depending on context.
2. **Silent-for-some-tasks neurons**: $z^\top w_s \approx 0$ for some $\alpha$. The neuron is effectively pruned; its (constant) bias output can be absorbed into downstream neurons.
3. **Task-invariant neurons**: $z^\top w_s$ is constant in $\alpha$. These compute fixed features.

Class 2 implies an emergent sub-network selection: different sub-networks of the same main network handle different tasks. The conclusion section flags this as a generalisation of conditional computation.

### Information content of $z$

Empirically, on benchmark 3 some dimensions of $z$ never converge across an episode — they appear to encode *state* features, not only task identity. Freeze-experiments show this state-coding is *not* critical: a frozen task-converged $z$ still navigates well, just with slightly degraded avoidance of the wrong target. This decoupling suggests $z$ implicitly carries both belief-over-task and a (weaker) state representation.

## Connections

This paper is one of two central nodes for this corpus (alongside Ben-Iwhiwhu et al. 2022) and is repeatedly cited downstream.

- **[ben-iwhiwhu_2022_context_meta_rl](beniwhiwhu_2022_context_meta_rl.md)** — directly builds on Vecoven 2020 by replacing the recurrent neuromodulator with a learned *context* vector and a multiplicative modulation rule; reframes the architecture for the CT-graph meta-RL benchmark.
- **[wang_2024_neuromod_meta](wang_2024_neuromod_meta.md)** — uses a similar modulator-on-activations idea but pairs it with MAML-style outer-loop adaptation; cites Vecoven 2020 as motivation.
- **[lee_2024_lifelong_rl](lee_2024_lifelong_rl.md)** — applies the Vecoven NMN idea to lifelong RL (sequence of tasks rather than meta-RL within-episode).
- **[mei_2022_multiscale_neuromod](mei_2022_multiscale_neuromod.md)** — surveys multiscale neuromodulation principles; positions Vecoven-style "modulate the activation function" as a single point in a larger design space.
- **[ferguson_cardin_2020_gain_modulation](ferguson_cardin_2020_gain_modulation.md)** — provides the biological reference for *why* slope/bias modulation of single neurons is a meaningful target; cortical-circuit account of multiplicative gain.
- Cited inside Vecoven 2020: Miconi et al.'s differentiable plasticity and backpropamine (the closest "modulate the synapse instead" line), Ha et al.'s hypernetworks, Garnelo et al.'s Conditional Neural Processes, and Wang et al.'s "Learning to reinforcement learn" (the meta-RL framing).
