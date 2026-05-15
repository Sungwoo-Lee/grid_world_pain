---
title: "Neuromodulators generate multiple context-relevant behaviors in a recurrent neural network by shifting activity hypertubes"
authors: ["Ben Tsuda", "Stefan C. Pate", "Kay M. Tye", "Hava T. Siegelmann", "Terrence J. Sejnowski"]
year: 2021
venue: "bioRxiv 2021.05.31.446462 (Jan 2022 version)"
slug: tsuda_2021_activity_hypertubes
source_pdf: "sources/Tsuda et al. 2021 - Neuromodulators generate multiple context-relevant behaviors in a recurrent neural network by shifting activity hypertubes.pdf"
topic: neuromodulatory_algorithms
---

# Tsuda et al. 2021 — Neuromodulators generate multiple context-relevant behaviors in a recurrent neural network by shifting activity hypertubes

## Plain-English entry point

This paper asks: how can one fixed brain circuit produce *opposite* behaviors in different internal states — for example, approaching a glass of water when thirsty but ignoring it when sated? The biological answer is **neuromodulators**: chemicals (dopamine, serotonin, noradrenaline, acetylcholine) that broadcast a single scalar signal across many neurons and change how those neurons compute. Tsuda et al. build a simple but striking computational demonstration. They train a **recurrent neural network (RNN)** — a network whose neurons feed back into themselves, producing activity that evolves over time — on a "Go/No-Go" task (output 1 for a stimulus when behavior 1 is required, output 0 for the same stimulus when behavior 2 is required). The "neuromodulator" is modeled as the simplest possible mechanism: multiply *every* synaptic weight in the RNN by a scalar factor $f_{nm}$ (e.g., 0.5 or 5.0) at test time. The result: the *same* network, with the *same* inputs, produces *opposite* outputs depending on $f_{nm}$. Visualized in the network's high-dimensional state space, the activity trajectories under different $f_{nm}$ values lie in non-overlapping "tubes" — the authors call them **activity hypertubes**. Intermediate $f_{nm}$ values yield smoothly intermediate behaviors, and they reproduce a real Drosophila experiment in which dopamine signaling tunes sugar sensitivity by starvation level. The paper matters because it grounds the idea of "internal state controls behavior via gain on weights" in a concrete RNN model that anyone can replicate — and connects to clinical observations about why some patients respond differently to psychiatric drugs ("circuit-based sensitivity").

## Section-ordered backbone

**Introduction.** Standard RNN models of context-dependent behavior use an exogenous cue input that drives neurons (Mante et al. 2013); but cues are noise-sensitive. Neuromodulation is fundamentally different — a single scalar that modifies how neurons transduce *all* their inputs, supporting stable state changes (sleep, hunger, mood) robust to fluctuations. Most psychiatric disorders involve neuromodulator dysregulation, yet large-scale computational principles are unknown. Prior work (Yu & Dayan 2005, Stroud 2018, Vecoven 2020, Beaulieu 2020, Miconi 2019) targeted neural-activation modulation; Tsuda focuses on **synaptic weight modulation**, an experimentally established but computationally under-modeled mechanism.

**Results — neuromodulation creates multiple weight regimes (Fig. 1).** A 200-unit RNN with 80% excitatory / 20% inhibitory units (Song-style Dale's law) is trained by BPTT on a modified Go/No-Go: stimulus "+" → +1 output if $f_{nm}=1$, $0$ if $f_{nm}=0.5$. A single uniform multiplicative factor $f_{nm}$ applied to *all* weights $W$ at test time unlocks distinct behaviors. Individual neurons show non-linear, unpredictable activity transforms reminiscent of crustacean stomatogastric ganglion (STG) neurons under amine modulation.

**Targeted neuromodulation (Fig. 2).** Subpopulations from 10% to 100% of the network sustain the dichotomy. Modulating only excitatory or only inhibitory neurons works. Up to 9 distinct behaviors are stored in one RNN by modulating 9 non-overlapping 10% subpopulations (or, equivalently, applying 9 different scalar levels to the same subpopulation).

**Distinct activity hypertubes (Fig. 3).** PCA of trial activity reveals that trajectories under different $f_{nm}$ values occupy non-overlapping tubes ("hypertubes") in state space. Intermediate $f_{nm}$ values trace a smooth curved arc between tubes (the "transition manifold"), producing intermediate outputs. The output-vs-$f_{nm}$ curve is sigmoidal; networks are characterized by an **EC50** (half-maximal effective concentration) of $f_{nm}$. Across 29 independently trained RNNs, EC50 ranges 2.1–6.5 (3.1×) — "circuit-based sensitivity." EC50 correlates with **weight-distribution skewness** ($R=0.51$) and with the **angle of departure** of the transition arc from the linear interpolation path ($R=0.60$).

**Drosophila sugar sensitivity (Fig. 4).** Inagaki et al. 2012 showed that fly proboscis extension reflex (PER) increases with both starvation duration (1- vs 2-day) and L-DOPA dose (0, 3, 5 mg/ml). RNNs trained on the fed ($f_{nm}=1$) and 2-day-starved ($f_{nm}=5$) endpoints generalize to the unseen intermediate ($f_{nm}=3$), producing PER curves matching 1-day-starved flies. This is the paper's central biological validation — neuromodulation gives smooth handling of *never-experienced* intermediate states.

**Electrical modulation as an independent mechanism (Fig. 5).** External current injected into the same subpopulation can also shift behavior, but with no correlation between EC50 and the electrical e-mod50 ($R \approx 0$). Chemical and electrical modulation push activity along *different* transition manifolds. Clinically: this may explain why some patients who fail pharmacotherapy respond to DBS.

**Discussion.** Neuromodulation enables overlapping synaptic memories accessed by a scalar key; provides a continuous interpolation mechanism for unseen internal states; reveals "circuit-based sensitivity" as a source of clinical variability. Future directions: differential receptor-specific modulation, metamodulation, exceeding the Turing limit (Cabessa–Siegelmann), more flexible/compact ML architectures.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** An RNN is a brain-inspired network whose units feed back onto each other, so activity flows through time. A neuromodulator (e.g., dopamine in flies) is biologically a chemical that changes how synapses transmit signals — Tsuda models this as the simplest possible thing: multiply *every* connection weight in the RNN by the same scalar $f_{nm}$. The discovery: just by twisting this one knob, the same RNN with the same input can produce the opposite output.

**The experimental setup.** Train an RNN on a "Go/No-Go" task with a twist — under context A (no neuromodulator), give output 1 to stimulus "+", but under context B (neuromodulator present, weights scaled by 0.5), give output 0 to the same "+". After training with backpropagation through time, run the same trained RNN at intermediate scalings (e.g., 0.75, 0.6, 0.55) and watch what happens. Repeat for up to 9 behaviors in one network. Then replicate a real Drosophila experiment where dopamine, controlled by starvation level, tunes how reflexively a fly extends its proboscis to sugar.

**The result.** (1) Yes — a single scalar reconfigures the network into different behavioral modes. (2) The network's activity in those modes lives in non-overlapping "tubes" in state space — the *same* fixed weight matrix, but the *trajectory* the network takes through state space depends on the scalar. (3) Intermediate scalar values produce smooth intermediate behaviors, including ones never seen during training. (4) Different networks trained on the same task have wildly different sensitivities to the scalar (their EC50 varies 3×) — a candidate explanation for why psychiatric drugs work for some patients and not others. (5) Chemical (weight-scaling) and electrical (current-injecting) modulation work via independent geometries of state-space transition, predicting that DBS can rescue non-responders.

## Phase 2 — Graduate-level deep dive

### 2.1 The RNN dynamics (Methods, eq. 1)

The continuous-time RNN dynamics are governed by:

$$
\tau \frac{dx}{dt} = -x + W r + W_{\text{in}}\, u + \mathcal{N}(0, 0.1),
$$

where $x \in \mathbb{R}^{N}$ is the synaptic-current state of $N = 200$ units, $W \in \mathbb{R}^{N \times N}$ is the recurrent weight matrix, $W_{\text{in}} \in \mathbb{R}^{N}$ are input weights for stimulus $u$, $\tau \in \mathbb{R}^{N}$ are per-unit synaptic decay time constants (sampled uniformly in $[20, 100]$ ms), and $\mathcal{N}(0,0.1)$ is i.i.d. Gaussian noise. The firing rate is the elementwise logistic:

$$
r = \sigma(x) = \frac{1}{1 + e^{-x}}.
$$

Weights are initialized with Dale's law (80% excitatory, 20% inhibitory à la Song et al. 2016) from $\mathcal{N}\!\left(0, g^2 / (N p_{\text{con}})\right)$ with gain $g = 1.5$ (chaos regime). The discrete Euler update is

$$
x_{i,t} = \left(1 - \tfrac{\Delta t}{\tau_i}\right) x_{i,t-1} + \tfrac{\Delta t}{\tau_i}\!\left(\sum_j W_{ji}\, r_{j,t-1} + W_{u,i}\, u_{t-1}\right) + \mathcal{N}(0, 0.1).
$$

Output is $o_t = W_{\text{out}}^\top r_t + b_{\text{out}}$, trained by BPTT with Adam on squared error.

### 2.2 The "hypertube" / gain-shift formulation

The core intervention is the **uniform weight scaling**

$$
W \;\longmapsto\; \tilde{W} = f_{nm} \cdot W,
$$

applied at test time after training. For subpopulation modulation, only rows/columns of $W$ indexed by the subpopulation $\mathcal{S}$ are scaled:

$$
\tilde{W}_{ji} = \begin{cases} f_{nm}\, W_{ji} & i \in \mathcal{S} \text{ (or } j \in \mathcal{S}\text{)} \\ W_{ji} & \text{otherwise} \end{cases}.
$$

Substituting $\tilde W$ into the dynamics gives the modulated RNN

$$
\tau \frac{dx}{dt} = -x + f_{nm}\, W\, \sigma(x) + W_{\text{in}}\, u + \mathcal{N}(0,0.1).
$$

This is **not** a simple gain on the activation function: it rescales the **recurrent loop gain**, which changes both the location of fixed points and the local Jacobian. Linearizing around a fixed point $x^*$ with $\sigma'(x^*) = \text{diag}(r^*(1-r^*))$:

$$
J_{nm} = -\tfrac{1}{\tau} I + \tfrac{1}{\tau}\, f_{nm}\, W\, \sigma'(x^*).
$$

The leading eigenvalues of $J_{nm}$ scale with $f_{nm}$, so increasing $f_{nm}$ generically expands unstable directions and reshapes the slow manifold along which trajectories evolve — defining a *different attractor landscape* per $f_{nm}$.

### 2.3 The hypertube geometry

Trajectories $\{r_t(f_{nm})\}_{t=0}^T$ for different $f_{nm}$ values form low-dimensional bundles in state space. Empirically, three PCs capture 80–92% of variance. The transition from $f_{nm}=1$ to $f_{nm}=9$ is non-linear in state space: rather than a straight line between endpoint states, the network traces an *arc*. The **angle of departure** AoD quantifies the curvature:

$$
\vec v_1 = p_F - p_N, \qquad \vec u_1 = p_{L_1} - p_N, \qquad
\text{AoD} = \cos^{-1}\!\left( \frac{\vec u_1 \cdot \vec v_1}{\|\vec u_1\|\, \|\vec v_1\|} \right),
$$

where $p_N$, $p_F$, $p_{L_1}$ are the network states at no, full, and the first intermediate $f_{nm}$ levels. AoD correlates strongly ($R=0.60$) with EC50: more orthogonal departure → lower sensitivity.

### 2.4 EC50 and circuit-based sensitivity

The dose-response curve is fitted as a sigmoid:

$$
\text{output}(f_{nm}) = 1 - \frac{1}{1 + e^{a\, f_{nm} + b}}, \qquad \text{EC50} = -\frac{b}{a}.
$$

Across 29 networks, $\text{EC50} \in [2.1, 6.5]$, slope $\in [0.9, 26.3]$. EC50 correlates with the **skewness of the recurrent-weight distribution** ($R=0.51$, $p<0.01$):

$$
\text{skew}(W) = \frac{1}{N^2 \sigma_W^3} \sum_{i,j}(W_{ij} - \bar W)^3.
$$

Positively skewed weights (long tail of strong excitatory connections) → less sensitive (higher EC50).

### 2.5 Continuous-state generalization (Drosophila demo)

For the sugar-sensitivity task, MAT (median attractive threshold) is fitted from the PER curve:

$$
\text{PER}(x_{\text{sugar}}) = \frac{1}{1 + e^{-a\, \log_2(x_{\text{sugar}} / \text{MAT})}},
$$

and a normalized shift index is defined for $f_{nm} = 3$ (intermediate):

$$
\%\Delta\text{MAT} = \frac{\text{MAT}_{f_{nm}=3} - \text{MAT}_{f_{nm}=1}}{\text{MAT}_{f_{nm}=5} - \text{MAT}_{f_{nm}=1}}.
$$

This measures how far the intermediate neuromodulator level moves the network from baseline toward the fully starved state. Empirically, intermediate $f_{nm}$ produces fly-realistic intermediate sensitivity, even though only the endpoints were trained — a smooth interpolation through the curved transition manifold.

### 2.6 Independence of electrical modulation

External electrical input $I_e$ enters as an additive bias in the dynamics:

$$
\tau \frac{dx}{dt} = -x + W r + W_{\text{in}} u + \mathbf{1}_{\mathcal{S}}\, I_e + \mathcal{N}(0,0.1).
$$

This is mathematically distinct from weight scaling: it shifts the *fixed-point locus* (the operating point) without changing the **Jacobian eigenvalues**. Consequently the network state moves along a different curved manifold in state space, and the e-mod50 dose required for half-output is uncorrelated with EC50 ($R \approx 0$, $p = 0.61$). This is the formal basis for the clinical claim that DBS responders need not be drug responders.

## Connections to other corpus papers

- **`shine_2021_cellular_to_dynamics.md`** (this batch) — Shine 2021 provides the biophysical / mean-field motivation; Tsuda gives the dynamical-systems demonstration in a trainable RNN. Tsuda cites Marder, Hasselmo, Avery-Krichmar — the same neuromodulator literature underlying Shine.
- **`avery_krichmar_2017_neuromodulation_models.md`** (other batch) — Tsuda's ref. 5; Avery & Krichmar review computational neuromodulator models, of which Tsuda's RNN is a minimal instance.
- **`vecoven_2020_neuromodulation_dnn.md`** (other batch) — Tsuda's ref. 20; Vecoven scales the *activation function* with a neuromodulator scalar; Tsuda scales the *weights*. The paper explicitly contrasts mechanisms (Appendix A).
- **`beaulieu_2020_anml.md`** / "Learning to continually learn" (other batch) — Tsuda's ref. 21; Beaulieu masks effector networks with a separate modulator; Tsuda uses uniform scaling.
- **`stroud_2018_motor_primitives.md`** (other batch) — Tsuda's ref. 19 (Nature Neurosci); targeted gain modulation of cortical RNNs for motor primitives; closely related mechanism.
- **`miconi_2019_backpropamine.md`** (other batch) — Tsuda's ref. 22; differentiable neuromodulated plasticity; related ML thread.
- **`costacurta_2024_structured_flexibility.md`** (this batch) — extends Tsuda's "scalar reshapes RNN computation" idea with a structured low-rank factorization that yields structured (not random) flexibility.
- **`wainstein_2025_gain_perceptual_switches.md`** (this batch) — empirical (pupillometry + fMRI + RNN) demonstration of the activity-hypertube-shift hypothesis with humans on perceptual switch tasks.
- **`driscoll_2022_dynamical_motifs.md`** (this batch) — orthogonal mechanism: multitask RNNs use *shared dynamical motifs* across tasks without an external scalar; Driscoll's "compositional" mechanism is what neuromodulation might wire on top of.
- **`doya_2002_metalearning_neuromodulation.md`** (other batch) — historical anchor for the "one scalar per modulator → distinct hyperparameter / control mode" framing.
- **`ferguson_cardin_2020_gain_modulation.md`** (other batch) — cellular substrate that justifies the uniform-weight-scaling abstraction.
