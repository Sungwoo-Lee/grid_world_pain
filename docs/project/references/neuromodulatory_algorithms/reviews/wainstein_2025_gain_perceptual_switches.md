---
title: "Evidence from pupillometry, fMRI, and RNN modelling shows that gain neuromodulation mediates task-relevant perceptual switches"
authors: ["Gabriel Wainstein", "Christopher J. Whyte", "Kaylena A. Ehgoetz Martens", "Eli J. Müller", "Vicente Medel", "Britt Anderson", "Elisabeth Stöttinger", "James Danckert", "Brandon R. Munn", "James M. Shine"]
year: 2025
venue: "eLife 2024;13:RP93191 (Version of Record posted 20 June 2025)"
slug: wainstein_2025_gain_perceptual_switches
source_pdf: "sources/Wainstein et al. 2025 - Evidence from pupillometry, fMRI, and RNN modell ... that gain neuromodulation mediates task-relevant perceptual switches.pdf"
topic: neuromodulatory_algorithms
---

# Wainstein, Whyte et al. 2025 — Gain neuromodulation mediates task-relevant perceptual switches

## Plain-English entry point

When you look at an image that gradually morphs from one object (say a shark) into another (a plane), your conscious perception doesn't follow the morph smoothly — it "pops" from one interpretation to the other at some point. What controls *when* that pop happens? Wainstein et al. test the hypothesis that a chemical signal in the brain — **noradrenaline (NA)** released from a brainstem nucleus called the **locus coeruleus (LC)** — gates the switch by transiently making neurons more responsive to their inputs, a property called **gain**. Because the LC's activity is reflected in the size of the pupil (a well-established indirect proxy), they can measure when LC fires in awake humans. They show three converging pieces of evidence: **(1) pupillometry** — pupil diameter peaks at the moment of perceptual switch in 35 subjects performing the morph task; **(2) computational model** — a recurrent neural network (RNN, a brain-inspired network with feedback connections) trained on the same task and equipped with a *dynamic* gain knob that rises with the network's own classification uncertainty switches faster and more confidently; **(3) fMRI** — in 17 different subjects performing the same task in a scanner, the brain's low-dimensional activity trajectory peaks in **velocity** (rate of change) and the brain's **egocentric energy landscape** (a measure of how hard it is to move between brain states) flattens at the switch point — exactly the two predictions the RNN model made. The unifying mechanism: at moments of perceptual uncertainty, LC bursts boost cortex-wide gain, this destabilizes the currently-dominant perceptual attractor (the "valley" the brain has been sitting in), and lets the brain jump to the competing attractor. This is the **network-reset hypothesis** of Bouret-Sara 2005, made quantitative across three measurement modalities.

## Section-ordered backbone

**Introduction.** Ambiguous-figures tasks (Necker cube, binocular rivalry) classically produce switches in perception. Bayesian / active-inference accounts say resolution depends on balancing prior vs. sensory evidence — but the *mechanism* that re-weights them is unclear. The LC noradrenergic system, with diffuse cortex-wide projections, is the leading candidate for a "network reset" signal (Bouret & Sara 2005; Sara 2009). Hypothesis: phasic LC bursts under uncertainty raise gain, destabilize the brain's current attractor, allow switching.

**Pupillometry result (Fig. 1).** 35 subjects view 20 line-drawing morphs, 15 images each, button-press to indicate perceived object. Pupil diameter (eye-tracker, 1 kHz, blink-corrected) is locked to the perceptual-switch image (Δ=0). Pupil rises starting 3 images before the switch, peaks at the switch ($\beta = 0.22$, $t_{(32)} = 8.02$, $p = 2.3 \times 10^{-19}$). On a trial-by-trial basis: faster switches correlate with bigger pupil dilations.

**RNN model (Fig. 2).** 50 continuous-time Dale's-law RNNs ($N = 40$, 32 E / 8 I), trained on a 1-second analog of the morph task: linearly interpolate between two-dimensional inputs $u_1$ and $u_2$. Selectivity analysis after training shows two stimulus-selective E-clusters with associated I-clusters. **Gain** is implemented as the slope of the sigmoid activation function (the standard NA-mediated mechanism, Servan-Schreiber 1990, Shine 2021). **Static gain** sweep: higher gain → earlier switches. **Dynamic gain**: an ODE drives gain to track the network's own readout entropy (its uncertainty), with strength $\gamma$. Higher $\gamma$ → faster switches. Mechanism (lesion analysis): the gain-driven oscillatory regime *inhibits* the currently-dominant population via its associated I-cluster, rather than exciting the competing E-cluster directly.

**Dynamical-systems analysis (Fig. 3).** Map the 2D parameter space $(g, \Delta u)$ by running 100 simulations per grid cell, measuring convergence time and proportion to a fixed point. Three regimes: (i) stable fixed-point representing $u_1$; (ii) stable fixed-point representing $u_2$; (iii) **oscillatory regime** at intermediate $\Delta u$ and high gain. The dynamic-gain RNN's trajectory in $(g, \Delta u)$-space crosses through the oscillatory regime at the switch, then settles in the new attractor.

**Energy-landscape predictions (Fig. 4).** Two complementary energy frameworks:
- *Allocentric* (third-person, position in PC space): $E = -\ln p(\text{state})$ via kernel density.
- *Egocentric* (first-person, displacement from current position): $E = -\ln p(\text{MSD})$ at lag $\tau$.

Both show gain-driven *flattening* at the switch — large displacements become low-energy. A novel "neural work" metric: $W_t = s_t \cdot \frac{dE_t}{dx}$, peaks at the switch under high $\gamma$.

**fMRI test (Figs. 5–6).** 17 different subjects, same task in MRI. PCA on BOLD time series across 375 ROIs → PC1, PC2, PC3 spatial maps. PC2 correlates strongly with Neurosynth "switching" map ($r = 0.453$, $p < 10^{-18}$). GLM time-locked to perceptual switch: PC2 + PC3 evoked activity peaks at switch. **Egocentric energy** of BOLD (PC2 displacement vs. surprisal): minimum at the switch — energy landscape flattens. PC2 **velocity** (gradient of regression-coefficient time series): peaks at the switch. Both RNN predictions confirmed in human BOLD.

**Discussion.** Convergent evidence across three modalities for LC-NA → gain → network-reset hypothesis. Limits: pupil is non-specific (also superior colliculus, raphe, cholinergic system contribute); 17-subject fMRI is small; no causal NA manipulation. The mechanism makes a testable prediction about why pupil dilations vanish for task-irrelevant ambiguous stimuli: gain ramps only when the network's *task readout* is uncertain.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Your brain doesn't passively watch a morphing image — it actively decides when to "switch" its interpretation. A chemical called noradrenaline, released by a tiny brainstem nucleus (the locus coeruleus, LC), tunes how strongly cortical neurons respond to their inputs. When uncertainty rises, LC fires, gain goes up, the brain's current interpretation becomes unstable, and a switch happens. Pupil size goes up and down with LC firing, so it gives a window into when this is occurring.

**The experimental setup.** Three experiments + a model. (1) Show 35 people images that morph from object A to object B over 15 steps; they press buttons to report what they see while an eye-tracker measures pupil. (2) Train 50 small RNNs (40 units each) on a stripped-down version of the same task and add a gain knob that responds to the network's own uncertainty. Test what happens at different gain strengths $\gamma$. (3) Re-analyze fMRI data from 17 different people doing the same task — PCA the brain activity, look for predictions made by the RNN.

**The result.** Pupil peaks at the switch, signaling LC firing. The RNN with uncertainty-driven gain switches earlier, via a transient oscillatory regime where inhibitory neurons silence the currently-winning interpretation. The RNN predicts two specific brain signatures: brain-state velocity should spike, and the energy needed to move between brain states should flatten — both confirmed in the fMRI. The story: ambiguity → uncertainty → LC burst → cortical gain up → brain's "attractor landscape" flattens → switch.

## Phase 2 — Graduate-level deep dive

### 2.1 The gain-modulated continuous-time RNN

The output-generating subnetwork is a Dale's-law continuous-time RNN with $N_E = 32$, $N_I = 8$ ($N = 40$ total). State: firing rates $r_t$, currents $x_t$. The dynamics (Methods, Euler–Maruyama discretization):

$$
x_{t+\Delta t} = (1 - \alpha)\, x_t + \alpha\!\left( W_{\text{rec}}\, r_t + W_{\text{in}}\, u_t \right) + \sigma_{\text{rec}} \sqrt{\Delta t}\, \mathcal{N}(0, 1),
$$

with $\alpha = \Delta t / \tau$, $\sigma_{\text{rec}} = 0.01$, $\Delta t$ adaptive (200 ms for training, 1 ms for analysis). The firing rate is a **gain-modulated** sigmoid:

$$
r_t = \sigma_g(x_t) = \frac{1}{1 + e^{-g(t)\, x_t}},
$$

where $g(t)$ is the **gain** parameter — the slope of the activation function. This is the canonical Servan-Schreiber / Shine mechanism. Gain $g$ enters everywhere uniformly (single scalar across the network). Output is

$$
z_t = W_{\text{out}}\, r_E, \qquad W_{\text{out}} \in \mathbb{R}^{N_E \times 2},
$$

with Dale's law enforced via element-wise multiplication with a sign mask. Training: cross-entropy with ADAM, 1000 iterations of BPTT, gain fixed at $g = 1$ during training.

### 2.2 The uncertainty-driven gain dynamics

The gain is a slow ODE driven by the network's own classification entropy:

$$
\tau_g \frac{dg(t)}{dt} = g_{\text{tonic}} - g(t) + \gamma\, H\!\big(p(z(t))\big),
$$

where the softmax readout is

$$
p\!\big(z_i(t)\big) = \frac{\exp(\omega z_i(t))}{\sum_{j} \exp(\omega z_j(t))},
$$

with inverse temperature $\omega$, and entropy

$$
H\!\big(p(z)\big) = -\sum_i p(z_i) \ln p(z_i).
$$

Parameters: $g_{\text{tonic}} = 1$ (baseline gain), $\gamma$ is the uncertainty-coupling strength (the "phasic burst magnitude"). When $p$ is sharp (one class confident), $H \approx 0$ and $g$ decays to $g_{\text{tonic}}$. When $p$ is uniform (maximum uncertainty), $H = \ln 2 \approx 0.693$ and $g$ grows. This is the model's analog of the LC's known sensitivity to unexpected uncertainty (Yu & Dayan 2005; Sales et al. 2019).

### 2.3 Pupillometry-RNN link

The link is conceptual but explicit: the entropy-driven gain term $\gamma H(p(z))$ is the model analog of phasic LC bursts; the resulting transient elevation of $g(t)$ is the analog of NA-mediated gain modulation; the pupillary dilation tracked in the human task is the surface signature of the same hypothesized LC burst. Both reach their maximum at the switch point. Within-subject correlation: faster perceptual switches in humans correlate with bigger pupil dilations — same monotonic relationship the RNN shows between $\gamma$ and switch time.

### 2.4 Phase-space and fixed-point analysis (Fig. 3)

For each grid point $(g, \Delta u) \in \mathbb{R}^2$ ($\Delta u = u_1 - u_2$), run 100 simulations with random initial conditions $r_0 \sim U[0,1]^N$ and compute the convergence statistic

$$
\Delta r_i(t) = \frac{r_i(t) - r_i(t - \Delta t)}{\Delta t}.
$$

Convergence time $T_c(g, \Delta u)$ = first time at which $\|\Delta r(t)\| < \epsilon$; convergence proportion = fraction of initial conditions converging within 10 s. Result: a $V$-shaped basin in $(g, \Delta u)$ — at low gain and clear $|\Delta u|$, fast convergence to a category attractor; at high gain and $\Delta u \approx 0$, no convergence → **oscillatory regime** where the network does not settle but cycles. The dynamic-gain trajectory crosses through this oscillatory regime at the switch, then re-enters a basin on the other side.

### 2.5 Allocentric energy landscape

Reduce $r(t)$ to its first PC, $x_t = \text{PC}_1[r(t)]$. Time-windowed kernel-density estimate of $p(x | t \in W_k)$, then energy

$$
E(x; W_k) = -\ln p(x; W_k) \quad \text{(setting } \beta = 1, z = 1\text{)}.
$$

High-$\gamma$ trials show a flatter landscape (lower $E$ for large $|x|$) at the switch window — the "energy barrier" between attractors is reduced.

### 2.6 Egocentric energy landscape (key fMRI-translatable predictor)

For each unit $i$ and reference time $t_0$, compute the mean-squared displacement

$$
\text{MSD}_{\tau, t_0} = \left\langle \big( x_{t_0 + \tau} - x_{t_0} \big)^2 \right\rangle_n,
$$

then the energy of a displacement of size $\text{MSD}$ at lag $\tau$:

$$
E_{\text{MSD}, \tau} = -\ln p(\text{MSD}; \tau) = \ln \frac{1}{p(\text{MSD}; \tau)}.
$$

This is signal-magnitude-invariant — applicable to BOLD where absolute units don't have a physical scale. Result: $E_{\text{MSD}, \tau}$ for large $\text{MSD}$ drops as $\gamma$ increases (energy landscape flattens). The area under the AUC vs. $\gamma$ curve quantifies the flattening.

### 2.7 Neural work — a novel metric

By analogy with classical mechanics ($W = F \cdot d$, with $F = -dE/dx$):

$$
W_t = s_t \cdot \left(-\frac{dE_t}{dx}\right),
$$

where $s_t = |x_{t + \Delta t} - x_t|$ is PC1 displacement and $dE_t/dx$ is the energy gradient at $x_t$. Peaks at the switch under high $\gamma$ — captures the "kinetic-energy injection" interpretation.

### 2.8 fMRI translation (Figs. 5, 6)

375-region BOLD parcellation (Gordon + Harvard-Oxford subcortical + SUIT cerebellar). PCA across all subjects/trials → top 3 PCs explain ~30.6%. PC2 is the "switching" PC: GLM time-locked to perceptual switch yields significant $\beta$ for PC2 around $\Delta = 0$. Spatial map correlates with Neurosynth's "switching" map ($r = 0.453$, $p < 10^{-18}$).

**Egocentric energy of BOLD**:

$$
E_{\text{MBD}, r_e} = \ln\!\frac{1}{p(\text{MBD}, r_e)},
$$

where MBD is the mean BOLD displacement and $r_e$ is the regressor index (relative to switch). Switch trial has minimum $E / |\beta_{\text{PC}_2}|$ — landscape flattens. **PC2 velocity** (gradient of $\beta$ time series): peaks at $\Delta = 0$. Both RNN predictions confirmed.

### 2.9 Lesion experiment (mechanism)

Lesion the I-cluster associated with the *initially dominant* E-cluster (the one selective for $u_1$ in a $u_1 \to u_2$ trial) → switch time slows dramatically. Lesion the I-cluster associated with the *competing* E-cluster (selective for $u_2$) → only modest slowing. **Interpretation**: gain speeds switches by *enhancing inhibition of the dominant population*, not by boosting the competing population. This is the circuit-level mechanism behind the population-level oscillation.

## Connections to other corpus papers

- **`shine_2021_cellular_to_dynamics.md`** (this batch) — Shine is senior author; this paper is the empirical and computational follow-up that operationalizes Shine 2021's theoretical claim that NA-mediated gain modulation reshapes the population activation function and thereby flattens whole-brain energy landscapes.
- **`tsuda_2021_activity_hypertubes.md`** (this batch) — Tsuda's "hypertube shift via weight scaling" and Wainstein's "fixed-point reshape via activation gain" are dual operationalizations of the same neuromodulator-controls-dynamics idea. Tsuda scales $W$; Wainstein scales the slope of $\sigma$. Both produce dynamics-level reconfiguration but via different parametric routes.
- **`costacurta_2024_structured_flexibility.md`** (this batch) — Costacurta's NM-RNN is the natural model class for Wainstein-style experiments: dynamic gain scaling of low-rank components driven by a small modulator subnet. The Wainstein RNN is a simpler full-rank precursor.
- **`driscoll_2022_dynamical_motifs.md`** (this batch) — Wainstein's fixed-point analysis uses the same Sussillo dynamical-systems toolkit; the "oscillatory regime at intermediate $\Delta u$" is a transient dynamical motif in Driscoll's sense.
- **`aston_jones_cohen_2005_lc.md`** (other batch) — the adaptive-gain theory of LC; Wainstein directly extends it with a computational dynamical-systems implementation.
- **`yu_dayan_2005_uncertainty.md`** (other batch) — uncertainty drives neuromodulator release; Wainstein's $\gamma H(p)$ term is the explicit ODE realization.
- **`sales_2019_lc_active_inference.md`** (other batch) — active-inference model of LC tracking prediction errors; same theoretical lineage.
- **`munn_2021_energy_landscape.md`** (Wainstein refs Munn et al. 2021) — origin of the energy-landscape framework used here for BOLD analysis.
- **`ferguson_cardin_2020_gain_modulation.md`** (other batch) — cellular substrate for the sigmoid-slope gain manipulation.
- **`servan_schreiber_1990_catecholamines.md`** (other batch) — original gain-modulation model that Wainstein's sigmoid-slope formulation implements.
- **`doya_2002_metalearning_neuromodulation.md`** (other batch) — historical assignment of NA to "uncertainty" function, which Wainstein's RNN gain-ODE realizes mechanistically.
