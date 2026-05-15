---
title: "Neuromodulation enhances dynamic sensory processing in spiking neural network models"
authors:
  - AbdalQader AlKilany
  - Dan F. M. Goodman
year: 2025
venue: "bioRxiv 2025.07.25.666748 (Imperial College London preprint, posted 31 July 2025)"
slug: alkilany_goodman_2025_snn_dynamic_sensory
source_pdf: "sources/AlKilany and Goodman 2025 - Neuromodulation enhances dynamic sensory processing in spiking neural network models.pdf"
topic: neuromodulatory_algorithms
---

# AlKilany & Goodman (2025) — Neuromodulation enhances dynamic sensory processing in spiking neural networks

## Plain-English entry point

In neuroscience, **neuromodulators** are chemicals (dopamine, noradrenaline, acetylcholine, serotonin, and others) that one neuron releases not into a single synapse but diffusely, so that *many* nearby neurons feel the signal at once. They are most often studied as slow learning signals over many minutes, but recent recordings (Hangya et al. 2015; Bang et al. 2020) show they also fluctuate on **sub-second timescales**, suggesting they have a fast role in moment-to-moment sensory processing — not just in learning.

This bioRxiv preprint from the Goodman lab at Imperial College tests that idea concretely: **add a small "modulator network" to a Spiking Neural Network (SNN) and let the modulator dynamically change the SNN's own biophysical parameters — firing threshold, reset and resting potentials, membrane and synaptic time constants — on the fly while the SNN processes input.** SNNs are brain-inspired models in which neurons communicate by all-or-nothing spikes; they are notoriously hard to train but extremely energy-efficient on **neuromorphic hardware** (custom chips like Intel Loihi).

The result, across three challenging real-world spiking datasets (Spiking Heidelberg Digits — spoken-digit recognition; Spiking Speech Commands — 35-word recognition; DVS128 Gestures — event-camera gesture recognition), is striking: the modulated SNN's accuracy improves substantially on every task. It also (a) makes decisions ~10–17% faster without being trained to, and (b) under noisy "speech-in-noise" conditions discovers, on its own, a strategy hypothesised in human hearing called **"listening in the dips"** — opportunistically increasing neural sensitivity when the background noise dips, and suppressing it when noise peaks. Performance gain at -17 dB SNR (very noisy) is ~25 percentage points. The paper matters to this corpus because it is the cleanest demonstration that **fast neuromodulation of intrinsic neural properties** (not weights) is a real, scalable lever for SNN performance and neuromorphic compute.

## Section-ordered backbone

**1. Introduction.** Neuromodulators broadcast signals to small or large groups of cells (Nadim & Bucher 2014; Mei et al. 2022). Slow timescales are well studied (Doya 2002; Grossman & Cohen 2022 — learning, decision-making). Sub-second roles in sensory processing recently demonstrated experimentally (Hangya et al. 2015 — cholinergic phasic responses; Bang et al. 2020 — sub-second dopamine / serotonin in human striatum during perceptual decisions). Previous SNN neuromodulation work (Krichmar 2012) was schematic; modern surrogate-gradient training (Neftci, Mostafa & Zenke 2019; Zenke et al. 2021) now allows realistic large-scale neuromodulation experiments. Builds on Perez-Nieves et al. 2021 *Nat. Commun.* showing heterogeneous neural time constants help SNNs. This paper takes the next step: make those time constants (and other intrinsic parameters) **dynamically self-modulated**.

**2. Results.**

- **2.1 Neuromodulation enhances sensory processing.** Architecture (Fig. 1):
  - Primary SNN: input spike layer → recurrent LIF hidden layer (200 neurons) → linear non-spiking readout. Each LIF neuron has *trainable* time constants, thresholds, resets, rests.
  - Modulator network: takes the hidden-layer activity over the past $K$ timesteps, feeds it through an MLP, and outputs new parameter values for the primary SNN over the next $K$ timesteps.
  - Two coupling modes: **substitution** ($p \leftarrow m$, the modulator outputs the full parameter value) and **addition** ($p \leftarrow p + m$, an additive correction; biologically more plausible).
  - Two grouping axes: *temporal* ($K = 1$ to a few hundred ms) and *spatial* ($G = 1$ to 200 — each modulator output controls $G$ neurons identically).

  Across SHD, SSC, DVS128: modulation improves accuracy at every setting. Best $K$ depends on dataset (sometimes per-step, sometimes 50–200 ms). Substitution wins by more than addition. Spatial grouping does not hurt much — even very broad neuromodulation works.

  Critically (Fig. 2D), increasing parameter count *without* modulation saturates accuracy quickly — the gain is not just from extra parameters.

  Replacing the ANN modulator with an SNN modulator yields equal or better performance (Fig. 3) — important for end-to-end neuromorphic deployment.

- **2.2 Neuromodulation improves reaction times.** Reaction time = $t$ at which output membrane potential peaks (since the argmax over $t$ determines the class). Modulated SNN responds ~10% faster on average (17% in median) than unmodulated, *without being trained to be faster*. The modulated SNN's reaction-time distribution becomes bimodal (~400 ms primary peak, ~900 ms secondary) — i.e. when informative content is early, it decides early; otherwise it waits.

- **2.3 Temporally dynamic gain control under noise.** Built a "SHD-in-noise" dataset adding (a) sinusoidally amplitude-modulated (SAM) Gaussian noise or (b) natural coffee-shop noise. Modulated SNN beats unmodulated by up to 25 pp at low SNR (-17 dB → 35% → ~60%).

  Mechanism (Fig. 6): in SAM noise, modulated SNN's firing rate is **suppressed during noise peaks** (red) and **boosted during noise dips** (blue) — the **listening-in-the-dips** strategy (Peters, Moore & Baer 1998; Lorenzi et al. 2006). Generalises to unseen noise frequencies (train at 5 Hz, test at 10 Hz). For natural coffee-shop noise, spike rate has $r = -0.60$ with noise amplitude, peak at lag = -47 ms.

  Parameter trajectories (Fig. 7): during noise peaks, **firing thresholds rise**, **resets fall** — i.e. neurons become less excitable when noise is high; the reverse at noise dips. All five modulated parameters (threshold, reset, rest, $\tau_m$, $\tau_s$) move significantly, but threshold/reset show the clearest noise-locked dynamics.

**3. Methods.**

- **Primary SNN.** Leaky integrate-and-fire (LIF) with continuous-time dynamics

$$\tau \dot v = -v + x, \qquad \tau_x \dot x = -x,$$

discretised via $\alpha = e^{-dt/\tau_x}$, $\beta = e^{-dt/\tau}$. Spike event $S = H(v - v_{\text{th}})$; reset $v \leftarrow v_r$.

- **Modulator architectures.**
  - **ANN modulator (Fig. 9C).** 2-layer MLP, ReLU hidden, sigmoid (substitution) or tanh (addition) output. Input = current parameter values + recent SNN activity + input spikes (summed over $K$ steps when $K > 1$).
  - **SNN modulator (Fig. 9D).** Recurrent LIF layer. Each output spike contributes a small $\pm$ change to one parameter via a learnable scalar.

- **Hyperparameters (Table 1):** 700 input × 200 hidden × {20, 35, 11} output (SHD, SSC, DVS); $\tau_m = 20$ ms baseline, $\tau_s = 10$ ms; $U_{\text{th}} = 1$ V, $U_0 = U_r = 0$; clipping ranges $\tau \in [1, 200]$ ms, $U_{\text{th}} \in [0.5, 1.5]$ V, $U_0, U_r \in [-0.5, 0.5]$ V.

- **Training.** Surrogate-gradient descent (Neftci et al. 2019) with surrogate $H'(x) = 1/(|x|+1)^2$ (SuperSpike, Zenke & Ganguli 2018). Two-phase training: (1) pre-train primary SNN alone; (2) jointly fine-tune primary SNN and modulator. Loss = cross-entropy on $\arg\max_t v_i(t)$ + rate regularisers $(r - 0.01)^2$ and $\mathrm{ReLU}(r - 100)^2$. Adam optimiser.

- **Datasets.** SHD: 700-channel cochleagram spike trains, 20 classes, 8156 train / 2264 test, two test-only speakers. SSC: 35 classes, 75 466 / 9981 / 20 382 samples. DVS128 Gestures: 128×128 event-camera, 11 gestures.

- **Reaction time:** $\mathrm{RT} = \arg\max_{i,t} v_i(t)$.

**4. Discussion.** Three implications. (1) For *neuromorphic computing*, modulation gives large accuracy gain at small parameter cost; either ANN or SNN modulator works; flexible spatial/temporal granularity allows hardware-specific tuning. (2) For *neuroscience*, fast (sub-second) neuromodulation has a robust general-purpose computational role — flexible decision timing and adaptive gain control / attention. (3) The "listening in the dips" result connects to a long-standing hypothesis about why human listeners outperform state-of-the-art ASR in noisy environments. Limitations: a single generic neuromodulatory mechanism — no different types, no target-cell-specific effects, no cross-modulator interactions. Future: add biophysical constraints, more sensory tasks.

## Phase 1 — Undergraduate-level synthesis

Spiking neural networks (SNNs) are models of the brain in which neurons communicate by all-or-nothing electrical spikes. They are interesting because they run very efficiently on neuromorphic chips (specialised brain-inspired hardware), but they are hard to train and currently lag behind regular deep nets on most benchmarks.

Each LIF neuron in an SNN has internal *biophysical parameters* — a threshold for firing, a reset value, time constants that decide how fast its membrane and synapses leak. In a standard trained SNN these parameters are fixed: they are set once during training and never change at inference time.

The paper's central idea is to make those parameters **dynamic**: every few milliseconds, a small "modulator network" looks at what the SNN is doing and *changes the SNN's own parameters on the fly*. This is what biological neuromodulators (dopamine, NA, ACh) do — they do not modify any single synapse precisely; they globally tune how excitable a chunk of cortex is.

The experiment: train this modulated SNN on three tough tasks — spoken-digit recognition (SHD), 35-word speech recognition (SSC), event-camera gesture recognition (DVS128).

Three results:

1. **Accuracy jumps**, on every task, by several percentage points. Crucially, an unmodulated SNN with the *same number* of total parameters cannot match it — the gain is in the *architecture*, not the parameter budget.
2. **Reaction times drop ~15%** without ever being asked to. The modulated SNN learns to make a decision early when the early part of the signal is informative.
3. **It learns to "listen in the dips"** in noisy speech. The modulated SNN automatically suppresses its own firing when background noise peaks, and amplifies it when noise dips. Humans are thought to do this; current speech-recognition systems do not. At -17 dB SNR (very noisy) the modulated SNN's accuracy is ~25 percentage points above unmodulated.

So the take-home: a tiny "neuromodulator" sub-network that controls intrinsic neural properties gives outsized improvements at low cost, generalises across tasks, and emergent strategies — like adaptive attention — fall out for free. This is useful both for understanding why brains may use neuromodulation in sensory processing (not just learning), and for making neuromorphic chips practical.

## Phase 2 — Graduate-level deep dive

### Leaky integrate-and-fire neuron dynamics

The primary SNN's hidden neurons are LIF (Eq. 1 of the paper):

$$
\tau \dot{v}(t) \;=\; -v(t) + x(t), \qquad \tau_x \dot{x}(t) \;=\; -x(t),
$$

with $v$ the membrane potential, $x$ the synaptic current, $\tau$ the membrane time constant, $\tau_x$ the synaptic time constant. An incoming spike via synapse of weight $w$ produces the event update (Eq. 2)

$$
v \;\leftarrow\; v + w.
$$

A spike is emitted (Eq. 3) when $v > v_{\text{th}}$, after which $v \leftarrow v_r$. For numerical implementation the time constants enter as decay factors (Eq. 4):

$$
\alpha = e^{-\Delta t / \tau_x}, \qquad \beta = e^{-\Delta t / \tau}.
$$

In the paper these are the *modulatable* parameters along with $v_{\text{th}}$, $v_r$, and $U_0$ (resting potential).

### Spike emission via surrogate gradient

In the forward pass (Eq. 11)

$$
S(t) \;=\; H\bigl(v(t) - v_{\text{th}}\bigr), \qquad H(x) = \mathbb{1}[x > 0]
$$

is non-differentiable. The backward pass replaces $H'(x)$ with the SuperSpike surrogate (Zenke & Ganguli 2018):

$$
\widetilde{H}'(x) \;=\; \frac{1}{(|x| + 1)^{2}}.
$$

This enables Adam-based gradient descent on all SNN parameters including the per-neuron $\{\alpha, \beta, v_{\text{th}}, v_r, U_0\}$.

### Neuromodulation: substitution and addition

The modulator outputs $m(t)$ are coupled to the primary SNN parameter $p$ via one of two rules (Eqs. 9–10):

**Substitution.**

$$
p(t) \;\leftarrow\; m(t),
$$

i.e. the modulator outputs the full new parameter value. Only feasible when $m$ can be constrained to the valid range (sigmoid output with a learned offset).

**Addition.**

$$
p(t) \;\leftarrow\; p(t) + m(t),
$$

i.e. modulator output is an additive correction. Used for both ANN modulator (tanh output) and SNN modulator (two output neurons per parameter: one for positive change, one for negative; each output spike adds $\pm \Delta p$ with $\Delta p > 0$ a learnable scalar). Clipping enforces biological bounds: $\tau \in [1, 200]$ ms, $v_{\text{th}} \in [0.5, 1.5]$ V, $v_r, U_0 \in [-0.5, 0.5]$ V.

### Temporal grouping

The modulator runs every $K$ timesteps (Fig. 1B). When $K > 1$:

$$
m_{[t, t+K)} \;=\; \mathrm{MLP}\!\Biggl( \sum_{s = t - K}^{t - 1} S(s), \; p(t-1) \Biggr),
$$

i.e. the MLP receives summed primary-SNN spike activity over the previous $K$ ms plus the current parameter vector. The new $m$ is held constant for the next $K$ ms. $K \in \{1, 50, 100, 200, 400\}$ tested; optimum is dataset-dependent (trade-off between integration time and update frequency).

### Spatial grouping

Group size $G$: each modulator output is broadcast to a contiguous block of $G$ neurons. With $G = 1$, every neuron is independently modulated; with $G$ large, modulation is shared across many cells. The paper finds performance roughly flat in $G$ for the tested tasks (Fig. 2E), implying broad neuromodulation is *almost* as effective as targeted — consistent with the diffuse anatomy of real neuromodulatory systems.

### Plasticity of intrinsic parameters as effective fast-weights

The mechanism here differs structurally from the gain-modulation papers in this corpus (Rodriguez-Garcia et al. 2026, Wainstein et al. 2025). Those modulate *output activations* multiplicatively. AlKilany & Goodman modulate **intrinsic biophysical parameters** — threshold, reset, time constants. This implements *plasticity of plasticity* (metaplasticity in the loose sense) at the millisecond scale: changing $v_{\text{th}}$ changes the neuron's input-output gain *and* its temporal integration window simultaneously.

A neuron's instantaneous *effective gain* (output spike rate per unit input current) for a constant input $I$ is approximately

$$
\mathrm{gain}_{\text{eff}}(I) \;\approx\; \frac{1}{\tau \cdot \log\!\bigl[(v_{\text{th}} - U_0 - I) / (v_r - U_0 - I)\bigr]} \cdot \mathbb{1}\bigl[ I > v_{\text{th}} - U_0 \bigr],
$$

so by jointly adjusting $\tau$ and $v_{\text{th}}$ the modulator can move the F-I curve along *both* axes — slope and threshold — yielding richer dynamic gain control than a pure multiplicative scaling.

### Reaction time

Defined as (Methods 3.6)

$$
\mathrm{RT} \;=\; \arg\max_{i, t} \; v_i(t),
$$

with $v_i(t)$ the output-neuron membrane potential. Since the predicted class is $\arg\max_i v_i^{\max}$, anything happening after RT cannot change the prediction — RT is the *earliest stable decision time*. The modulated SNN's RT distribution is bimodal (peaks at ~400 ms and ~900 ms) vs the unmodulated SNN's unimodal ~730 ms. The 17% median reduction is emergent, not loss-encoded.

### Loss

The training objective sums three terms:
1. **Task loss.** Softmax over $v_i^{\max} = \max_t v_i(t)$ followed by cross-entropy.
2. **Rate regulariser 1.** $\propto \sum_{i,b} (r_i - 0.01)^2$ encouraging low average firing.
3. **Rate regulariser 2.** $\propto \sum_b \mathrm{ReLU}(r_{\text{pop}} - 100)^2$ penalising population bursts.

### Listening in the dips — quantitative characterisation

In sinusoidally amplitude-modulated (SAM) noise the envelope is

$$
n(t) \;=\; \frac{A}{2}\bigl(1 + \sin(2\pi f_m t + \phi)\bigr) \cdot \xi(t),
$$

with $\xi(t)$ white Gaussian noise, $f_m \in \{5, 10\}$ Hz, $\phi$ uniform in $[0, 2\pi)$ at train and test. The modulated SNN's average spike rate $r(t)$ anti-correlates with $n(t)$ at lag ~50 ms (Pearson $r = -0.60$ on coffee-shop noise; minimum cross-correlation at $-47$ ms). This is the operational signature of **"listening in the dips"** as defined by Peters, Moore & Baer 1998 and Lorenzi et al. 2006 in psychophysics — neurally instantiated here without any direct training signal favouring it.

### Computational mechanism summary

The system is a hierarchical two-network architecture:

1. **Primary SNN** $F$ with parameter vector $\boldsymbol{\theta}(t) = (\boldsymbol{\tau}, \boldsymbol{\tau}_x, \boldsymbol{v}_{\text{th}}, \boldsymbol{v}_r, \boldsymbol{U}_0)$ — initial values learnable, *and* dynamically modulable.
2. **Modulator network** $M$ (MLP or SNN) producing $\boldsymbol{m}(t) = M\bigl(\mathrm{hist}_K[\text{spikes}], \boldsymbol{\theta}(t-1)\bigr)$, with $\boldsymbol{\theta}(t) \leftarrow \boldsymbol{m}(t)$ (substitution) or $\boldsymbol{\theta}(t) \leftarrow \boldsymbol{\theta}(t-1) + \boldsymbol{m}(t)$ (addition).
3. Joint surrogate-gradient training of $\boldsymbol{W}_{\text{primary}}$, $\boldsymbol{\theta}_{\text{init}}$, and $\boldsymbol{W}_M$ under cross-entropy + rate regularisers.

The result is a single end-to-end-trainable architecture that exposes the primary SNN's biophysical parameter trajectory as a learnable computational variable, expanding the model class far beyond static SNNs while keeping the spiking substrate.

## Connections to other papers in this corpus

- **Mei et al. 2022.** Cited directly. Mei et al. 2022 *Trends Neurosci.* (`mei_2022_multiscale_neuromodulation`, other batch) is the conceptual ancestor — "informing deep nets by multiscale neuromodulatory principles". This paper instantiates that idea in the SNN domain.
- **Doya 2002.** Cited as the foundational meta-learning-and-neuromodulation reference (`doya_2002_metalearning_neuromodulation`, other batch).
- **Krichmar 2012 / Krichmar-lab.** Cited as the prior schematic SNN-neuromodulation work. Direct ties to Cox & Krichmar 2009 (`cox_krichmar_2009_neuromodulation_robot`, other batch), Krichmar 2013 (`krichmar_2013_anxious_curious`, other batch), Avery & Krichmar 2017 (`avery_krichmar_2017_models_neuromodulation`, other batch — explicitly cited in the discussion), and Espino et al. 2024 (`espino_2024_snn_path_planning`, this batch — also an SNN + Krichmar-lab work).
- **Kudithipudi et al. 2025 neuromorphic-computing review** cited as the broader neuromorphic context — Kudithipudi is also the corresponding author of the 2022 lifelong-learning survey in this corpus (`kudithipudi_2022_lifelong_learning`, this batch).
- **Neuromorphic substrate.** Builds on Neftci, Mostafa & Zenke 2019 (surrogate gradients) and Zenke et al. 2021 (visualising a joint future of neuroscience and neuromorphic engineering). Thematically aligned with the other SNN paper in this batch — Espino et al. 2024 (`espino_2024_snn_path_planning`) — and the Hopfield-based RBM paper Tambaş et al. 2025 (`tambas_2025_krotov_hopfield_rbm`, this batch).
- **Heterogeneity prior.** Builds on the lab's own Perez-Nieves et al. 2021 *Nat. Commun.* on heterogeneous time constants — that paper established that *static* heterogeneity helps; this one makes the heterogeneity dynamic and self-modulated.
- **Gain modulation cluster.** The "listening in the dips" mechanism is **dynamic gain control via excitability**, the same general theme as:
  - Rodriguez-Garcia et al. 2026 (`rodriguezgarcia_2026_ne_stability_gap`, this batch) — gain modulation for stability-gap reduction.
  - Ferguson & Cardin 2020 (`ferguson_cardin_2020_gain_modulation`, other batch) — cortical gain-modulation mechanisms.
  - Wainstein et al. 2025 (`wainstein_2025_gain_perceptual_switches`, other batch) — gain modulation in perceptual switches.
  - Shine 2019 / Shine et al. 2021 (`shine_2021_cellular_to_dynamics`, other batch) — cellular mechanisms of neuromodulation to large-scale dynamics.
- **Cholinergic attention.** The discussion explicitly invokes the cholinergic attentional system (Avery & Krichmar 2017; Hangya et al. 2015). The framing connects to Driscoll et al. 2022 (`driscoll_2022_shared_dynamical_motifs`, other batch) and Tsuda et al. 2021 (`tsuda_2021_hypertube_shifts`, other batch) on context-relevant neuromodulator-driven RNN reorganisation.
- **Grossman & Cohen 2022.** Cited as the modern neuromodulation-and-decision-making review; bridges to the Krichmar-Cañamero behavioural robotics line in the corpus (Cañamero 1997, 2005; Blanchard & Cañamero 2006; Lewis & Cañamero 2016; Khan & Cañamero 2022; L'Haridon & Cañamero 2023).
- **Meta-learning neuromodulation.** The dynamic-parameter-modulation framing is the natural counterpart of Wang et al. 2024 (`wang_2024_neuromodulated_meta_learning`, other batch), Vecoven et al. 2020 (`vecoven_2020_neuromodulation_deep_nets`, other batch), and Ben-Iwhiwhu et al. 2022 (`ben-iwhiwhu_2022_context_meta_rl`, other batch).
- **Bang et al. 2020.** Sub-second human striatal dopamine / serotonin during perceptual decision-making — the experimental motivation; also relevant to the project's `perceptual_decision_making` corpus.
