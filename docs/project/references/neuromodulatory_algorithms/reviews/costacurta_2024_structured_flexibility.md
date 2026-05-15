---
title: "Structured flexibility in recurrent neural networks via neuromodulation"
authors: ["Julia C. Costacurta", "Shaunak Bhandarkar", "David Zoltowski", "Scott W. Linderman"]
year: 2024
venue: "NeurIPS 2024 (38th Conference on Neural Information Processing Systems)"
slug: costacurta_2024_structured_flexibility
source_pdf: "sources/Costacurta et al. 2024 - Structured flexibility in recurrent neural networks via neuromodulation.pdf"
topic: neuromodulatory_algorithms
---

# Costacurta et al. 2024 — Structured flexibility in recurrent neural networks via neuromodulation

## Plain-English entry point

This paper asks: standard recurrent neural networks (RNNs) — networks whose units feed back on each other to process time-varying input — have *fixed* connection weights once trained, but real brain synapses constantly change strength under chemical signals called **neuromodulators** (e.g., dopamine, serotonin). Can we build an RNN that captures this without losing analytical tractability? Costacurta et al. propose the **NM-RNN**, an RNN made of two pieces: (1) a small **neuromodulatory subnetwork** $z(t)$ that watches the inputs and emits a low-dimensional signal $s(t)$; (2) a larger **output-generating subnetwork** $x(t)$ whose recurrent weight matrix $W_x$ is **low-rank** (a sum of a few rank-1 outer products $\ell_k r_k^\top$). Crucially, the neuromodulatory signal $s_k(t)$ multiplicatively scales the $k$-th rank-1 component of $W_x$ on the fly. Three results: (a) the NM-RNN beats parameter-matched low-rank RNNs and vanilla RNNs at timing-extrapolation tasks (Measure-Wait-Go), and matches LSTM performance; (b) it admits an explicit mathematical equivalence to the LSTM forget gate, with $s(t)$ playing the role of the LSTM forget gate $f_t$ — a striking convergence of biology-motivated and engineering-motivated gating mechanisms; (c) on a multitask benchmark (DelayPro / DelayAnti / MemoryPro / MemoryAnti), retraining *only* the neuromodulatory subnetwork on a new task gives strong transfer learning, mirroring Driscoll et al. 2022's compositional motifs. The paper matters because it gives a *biologically motivated and analytically interpretable* form of "structured flexibility": the neuromodulator doesn't replace the recurrent weights, it multiplicatively reshapes them along a small number of structured directions — providing a clean dynamical-systems explanation for why gain modulation makes RNNs more flexible.

## Section-ordered backbone

**Introduction.** Traditional task-trained RNNs have fixed weights; biology has continuously-modulated synapses (Marder, Bargmann). Inputs to a fixed-weight RNN can only perturb the network state, not reshape its dynamics. The authors propose a small neuromodulatory subnetwork that scales a low-rank decomposition of the main recurrent matrix, providing the missing structured flexibility.

**Background.**
- *Modeling neuromodulation*: biophysical models (1980s onward, Abbott 1990 — neuromodulation adds long-term memory and learning gating, anticipating LSTMs); modern spiking-RNN approaches scale firing rates of subsets of neurons (Stroud 2018), incorporate arousal-mediated modulation (Munn / Shine), or use modulation for credit assignment.
- *Hypernetworks*: Ha 2017 generates RNN weights from a small hypernet; NM-RNN is similar but constrained to rank-1 multiplicative scaling for analytical traction. Von Oswald 2020 used hypernets for continual learning and explicitly suggested a "lower-dimensional modulatory signal" akin to NM-RNN.
- *Low-rank RNNs*: Mastrogiuseppe & Ostojic 2018, Beiran et al. 2021, Dubreuil et al. 2022 — low-rank $W$ admits a tractable theory of recurrent dynamics; perfect substrate for component-wise gain modulation.

**The NM-RNN model (Section 3, eqs. 1–4).** A coupled ODE:
- Neuromodulatory subnet: $\tau_z\, \dot z = -z + W_z\, \phi(z) + B_z\, u$.
- Output subnet: $\tau_x\, \dot x = -x + W_x(z)\, \phi(x) + B_x\, u$.
- Readout: $y = C x + d$.
- Time-varying weights: $W_x(z(t)) = \sum_{k=1}^K s_k(z(t))\, \ell_k r_k^\top$ where $s = \sigma(A_z\, z + b_z)$ (sigmoid).
- Hyperparameters: $\tau_z \gg \tau_x$ (slow modulator, fast output), $M < N$ (smaller modulator).

**Mathematical intuition (Section 3.1).** In a simplified symmetric/linear/no-input limit, change coordinates to $w = L^\top x$ where $L$ has columns $\ell_k$. The dynamics decouple into

$$\tau_x \dot w = -w + S(t) w, \qquad S(t) = \text{diag}(s(t)),$$

with explicit solution showing each mode decays at rate $(1 - s_k(t))/\tau_x$. So $s_k$ literally sets the time constant of mode $k$ — a "forgetting rate" knob per dynamical mode.

**Connection to LSTMs (Section 3.2).** The discretized NM-RNN cell-state update is $w_t = s_t \odot w_{t-1} + L^\top B_x u_t$, identical in form to the LSTM cell update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$. The supplementary material proves NM-RNN dynamics can be exactly reproduced by an LSTM under suitable assumptions (Proposition 1). This is a precise bridge between biology and engineering gates.

**Time-interval reproduction (Section 4, Fig. 3).** Measure-Wait-Go (MWG) task. For analytically tractable rank-1 linear NM-RNN, the optimal neuromodulatory signal has closed form $s(t) = (f(t) + \tau_x f'(t) - d) / ((\ell^\top r)(f(t) - d) + (c^\top \ell)(r^\top w_\perp(0) e^{-t/\tau_x}))$. Empirically, the trained rank-1 NM-RNN's signal matches this prediction. Rank-3 NM-RNNs trained on 4 intervals extrapolate to longer/shorter intervals better than parameter-matched LR-RNNs, vanilla RNNs, and (in some metrics) LSTMs. Ablation: each $s_k$ controls a specific computational sub-step (interval-setting, ramping, terminating).

**Multitask reusing dynamics (Section 5, Fig. 4).** Train NM-RNN on three tasks {DelayPro, DelayAnti, MemoryPro} feeding the *context* input only to the neuromodulatory subnet. Then freeze $W_x$ and retrain *only* the neuromodulatory subnet on the held-out MemoryAnti. The output-generating subnet shares a **ring attractor** for angle memory across tasks (PCs 1–2); the Pro/Anti distinction lives in PC5. Direct parallel to Driscoll 2022's shared motifs, with the rule input now mediated by neuromodulator rather than a static input weight.

**Element Finder Task (Section 6, Fig. 5).** Sequence task with long-range dependencies (recall the element at query index $q$ from a sequence of 25). NM-RNN with feedback coupling $x \to z$ matches LSTM performance and beats vanilla RNN. Analysis: $z(t)$ settles to a line attractor encoding $q$; $s(t)$ gates between modes at the queried timestep; $x(t)$ converges to a line attractor encoding the element value, with ordering preserved.

**Discussion.** Structured flexibility via modulation of singular values of $W_x$ enables both within-task generalization (timing) and across-task transfer (multitask). Gating-like dynamics from neuromodulation are mathematically equivalent to LSTM gates under conditions. Limits: small networks ($N \approx 100$); not compared to neural data; only post-training analyzed (not three-factor learning-rule role).

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Imagine an RNN whose recurrent weight matrix is built from just a few "basis ingredients" (rank-1 matrices) that you can mix in different proportions over time. The proportions are not fixed; a tiny "control RNN" computes them on the fly, based on the input. That control signal is the model's "neuromodulator". By turning the mixing knobs, the same set of ingredients can implement very different computations.

**The experimental setup.** Build two coupled networks: a small slow one (the modulator $z$, 5–10 units) and a larger fast one (the output network $x$, ~100 units). Constrain the output network's recurrent matrix to be a sum of $K$ rank-1 pieces, each scaled by a sigmoidal output of the modulator. Train end-to-end on canonical neuroscience tasks: a timing task (Measure-Wait-Go) where the network has to reproduce a measured interval; a multitask suite with Pro/Anti and Memory/Delay variants; and a sequence task with long-distance dependencies.

**The result.** (1) On timing: the NM-RNN extrapolates to longer intervals than vanilla low-rank RNNs, because the modulator can stretch the network's effective time constant by adjusting the gain on each mode. (2) On multitask: when you train on three tasks then retrain only the modulator on the fourth, the output network's geometry (a ring attractor for angle memory) is preserved and reused. (3) The neuromodulator turns out to be mathematically equivalent to an LSTM forget gate — biology and engineering converged on the same gating mechanism. (4) On long-range sequence tasks: NM-RNN matches LSTM and beats vanilla RNN, even with matched parameter counts.

## Phase 2 — Graduate-level deep dive

### 2.1 The NM-RNN coupled ODE (eqs. 1–4)

The two-subnet model with neuromodulator $z(t) \in \mathbb{R}^M$ and output state $x(t) \in \mathbb{R}^N$ ($M < N$):

$$
\tau_z \frac{dz(t)}{dt} = -z(t) + W_z\, \phi(z(t)) + B_z\, u(t),
$$

$$
\tau_x \frac{dx(t)}{dt} = -x(t) + W_x\!\big(z(t)\big)\, \phi(x(t)) + B_x\, u(t),
$$

$$
y(t) = C\, x(t) + d.
$$

The crucial coupling is the **rank-$K$ multiplicative gain decomposition**:

$$
W_x\!\big(z(t)\big) \;=\; \sum_{k=1}^{K} s_k\!\big(z(t)\big)\, \ell_k r_k^\top, \qquad s\!\big(z(t)\big) = \sigma\!\big(A_z\, z(t) + b_z\big).
$$

Here $\ell_k, r_k \in \mathbb{R}^N$ are learned left and right factors (column/row vectors of $W_x$), $\sigma$ is the sigmoid, $A_z \in \mathbb{R}^{K \times M}$ and $b_z \in \mathbb{R}^K$ are learned modulator-to-gain weights. $\phi$ is $\tanh$. $\tau_z \gg \tau_x$ enforces slow modulation.

### 2.2 The structured-flexibility decomposition

Compare to vanilla low-rank RNN where $W_x = \sum_k \ell_k r_k^\top$ is fixed. NM-RNN replaces this with the **time-varying rank-$K$ matrix**

$$
W_x(t) \;=\; L\, S(t)\, R^\top, \qquad L = [\ell_1, \dots, \ell_K], \;\; R = [r_1, \dots, r_K], \;\; S(t) = \text{diag}\!\big(s_1(t), \dots, s_K(t)\big).
$$

This is a low-rank SVD-like factorization with the singular values $s_k(t)$ controlled by the modulator. Each $s_k \in (0, 1)$ via sigmoid; $s_k \to 0$ "switches off" mode $k$, $s_k \to 1$ activates it fully.

### 2.3 Decoupled-mode derivation (eq. 5)

Under the assumptions (a) $W_x$ symmetric ($\ell_k = r_k$), (b) $\{\ell_k\}$ orthonormal, (c) $\phi(x) = x$ (no nonlinearity), (d) $u = 0$: define $w(t) = L^\top x(t)$ with $L^\top L = I_K$. Then

$$
\tau_x \frac{dw(t)}{dt} = L^\top \big(-x + W_x x\big) = -w + L^\top L\, S(t)\, L^\top x = -w + S(t)\, w.
$$

Componentwise:

$$
\tau_x \dot w_k = -(1 - s_k(t))\, w_k \;\;\Longrightarrow\;\; w_k(t) = w_k(0)\, \exp\!\left(-\int_0^t \frac{1 - s_k(t')}{\tau_x}\, dt'\right).
$$

So $s_k(t) = 1$ → zero decay (perfect memory along $\ell_k$); $s_k(t) = 0$ → decay rate $1/\tau_x$ (fastest forgetting). The neuromodulator literally schedules the timescale of each mode.

### 2.4 Equivalence to LSTM forget gate (Section 3.2)

Discretize eq. (5) with $\tau_x = 1$, $\Delta t = 1$:

$$
w_t = (1 - 1)\, w_{t-1} + s_t \odot w_{t-1} + L^\top B_x\, u_t = s_t \odot w_{t-1} + L^\top B_x\, u_t.
$$

Compare LSTM cell-state update:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t.
$$

Identifications: $w_t \leftrightarrow c_t$, $s_t \leftrightarrow f_t$ (forget gate), $L^\top B_x u_t \leftrightarrow i_t \odot \tilde c_t$ (input-gated transformation). The supplementary Proposition 1 proves the dynamics are exactly reproducible by an LSTM under suitable assumptions. **Biological forget-gating ≡ engineering forget-gating.**

### 2.5 Analytic timing-task signal (Section 4.1, eq. 8)

For rank-1 linear NM-RNN on MWG, with output $y(t) = c^\top x(t) + d$, target $f(t)$, no input during ramp:

$$
s(t) = \frac{f(t) + \tau_x f'(t) - d}{(\ell^\top r)\!\left(f(t) - c^\top w_\perp(0)\, e^{-t/\tau_x} - d\right) + (c^\top \ell)\!\left(r^\top w_\perp(0)\, e^{-t/\tau_x}\right)}.
$$

Under the approximation $w_\perp(0) \approx 0$ and small $\tau_x$:

$$
s(t) \approx \frac{1}{\ell^\top r}\!\left(1 + \frac{\tau_x f'(t)}{f(t) - d}\right).
$$

This is a closed-form prediction matched by the trained NM-RNN's signal in Fig. 3B, both for trained and extrapolated intervals — strong evidence that the network learns the analytically optimal gain schedule.

### 2.6 Element Finder Task and feedback coupling (eq. 9)

For long-range dependencies, the basic NM-RNN is augmented with feedback from output state $x$ into modulator $z$:

$$
\tau_z \frac{dz(t)}{dt} = -z(t) + W_z\, \phi(z(t)) + \big(B_{zx}\, \phi(x(t)) + b_{zx}\big) + B_z\, u(t).
$$

This closes the loop, mirroring the LSTM's $f_t = \sigma(W_f[h_{t-1}, u_t] + b_f)$ where the forget gate also depends on the prior hidden state. Empirically $z(t)$ settles on a line attractor encoding query index $q$; $s(t)$ gates between modes precisely at $t = q$; $x(t)$ ends on a line attractor encoding the element value.

### 2.7 Multitask transfer (Section 5)

Train NM-RNN on tasks $\{T_1, T_2, T_3\}$, feeding the context one-hot to $z$ only (not $x$). After training freeze $\{W_x, L, R, B_x\}$; retrain only $\{W_z, B_z, A_z, b_z\}$ on novel task $T_4$. Performance comparable to retraining input weights of an LR-RNN, vanilla RNN, or LSTM — but the modulator parameters are fewer, and crucially the *dynamical motifs* in $W_x$ (the ring attractor) are reused identically. The mechanism is mathematically equivalent to Driscoll 2022's rule-input retraining, but with the rule's effect mediated by the modulator subnet rather than by direct additive input bias.

### 2.8 Comparison to corpus mechanisms

| Mechanism | Parametrization of recurrence | Reference |
|---|---|---|
| Vanilla RNN | $W_x$ fixed | — |
| Low-rank RNN | $W_x = LR^\top$ fixed | Mastrogiuseppe-Ostojic 2018 |
| Driscoll input-bias | $W_x$ fixed; rule input shifts operating point | `driscoll_2022_dynamical_motifs.md` |
| Tsuda weight scaling | $W_x \mapsto f_{nm} W_x$ (single scalar) | `tsuda_2021_activity_hypertubes.md` |
| **NM-RNN (Costacurta)** | $W_x(t) = L\, S(z(t))\, R^\top$ ($K$-dim gain) | this paper |
| Hypernetwork (Ha 2017) | $W_x = \sum_k h_k(z) M_k$ (full-rank $M_k$) | superclass |

The NM-RNN is a structured, low-rank, biology-motivated specialization of hypernetworks, with the analytical traction needed for fixed-point analysis.

## Connections to other corpus papers

- **`tsuda_2021_activity_hypertubes.md`** (this batch) — direct conceptual ancestor: a scalar neuromodulator scales weights, here generalized to $K$ scalars scaling rank-1 components.
- **`driscoll_2022_dynamical_motifs.md`** (this batch) — multitask transfer in Costacurta directly cites and mirrors Driscoll's "rule input + frozen motifs" mechanism, with the modulator subnet playing the role of Driscoll's rule input.
- **`shine_2021_cellular_to_dynamics.md`** (this batch) — Shine provides the biological motivation for "gain controls dynamics"; Costacurta gives the cleanest analytical RNN instantiation. The "modulating singular values of $W_x$" framing is the discrete-RNN analog of Shine's "reshaping the population gain function".
- **`wainstein_2025_gain_perceptual_switches.md`** (this batch) — Wainstein uses a gain-modulated RNN driven by pupil; NM-RNN is the natural model class for that setup.
- **`stroud_2018_motor_primitives.md`** (other batch) — cited (ref. 15); Stroud uses balanced E/I RNNs with constant per-neuron multipliers; Costacurta generalizes to dynamic, rank-1-targeted multipliers.
- **`vecoven_2020_neuromodulation_dnn.md`** (other batch) — DNN with simple gain modulation; Costacurta is the structured RNN analog.
- **`ha_2017_hypernetworks.md`** (Ha et al. 2017 HyperRNN, ref. 21) — formal superclass; Costacurta specializes to rank-1 multiplicative + biological-time-constant separation.
- **`vonoswald_2020_hypernet_continual.md`** (other batch, ref. 22) — Costacurta cites von Oswald's suggestion that "a hypernetwork outputting lower-dimensional modulatory signals could assist task-specific mode-switching" — NM-RNN delivers exactly this.
- **`mastrogiuseppe_ostojic_lowrank.md`** (refs. 28–30) — formal foundation for low-rank RNN analysis.
- **`abbott_1990_modulation.md`** (ref. 11) — historical anchor: Abbott showed that adding a neuromodulatory parameter to a spiking-network ODE yields LSTM-like long+short-term memory, anticipating both LSTMs and NM-RNN.
- **`miconi_2019_backpropamine.md`** (other batch, ref. 14) — modulation for credit assignment / three-factor learning rule — orthogonal axis; Costacurta lists this as future work.
- **`avery_krichmar_2017_neuromodulation_models.md`** (other batch, refs. 6–8) — surveyed body of biophysical models that motivated Costacurta's abstraction.
- **`duncker_driscoll_2020_organize_dynamics.md`** (ref. 39) — multitask suite that Costacurta uses (DelayPro/Anti, MemoryPro/Anti).
- **`beiran_2021_shaping_dynamics.md`** (ref. 29 / 35) — direct comparison target on MWG; rank-3 LR-RNN with tonic context input is the baseline NM-RNN improves over.
