---
title: "Neuromodulatory algorithms — predictions synthesis (cellular gain ↔ behavioural hyperparameter)"
status: draft
audience: research-postdoc, professor-neuromodulation, professor-rl-bayesian-dl, professor-bayesian-brain, professor-pain-modeling, experiment-designer
last_updated: 2026-05-16
related:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v2.md
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md
purpose: "Focused cross-paper synthesis that becomes the evidence base for v3 §5 (predictions). Question answered: what does the corpus say, paper-by-paper, about the link from cellular-level neural gain to behavioural-level RL-hyperparameter modulation?"
---

# Neuromodulatory algorithms — predictions synthesis (cellular gain ↔ behavioural hyperparameter)

## Plain-English entry point

This document collects, from the project's 18-paper neuromodulation reference corpus (at `docs/project/references/neuromodulatory_algorithms/sources/`), the published evidence for a single specific claim: **that a simple cellular-level operation — scaling the slope of a neuron's input-output curve, and shifting its operating point — is the same machine, mathematically, as a behavioural-level adjustment of reinforcement-learning hyperparameters (how fast the agent learns, how much it explores, how far into the future it plans, how much weight it puts on noisy evidence)**. The project's thesis is that the brain uses one simplified principle — neural gain — to produce a wide range of behavioural-level effects, and that an artificial agent built with a FiLM layer (a neural-network operation that multiplies activations by one signal and adds another) is the first explicit, learnable bridge between the two readings. This synthesis answers four questions: (1) which hyperparameters has the corpus claimed neural gain modulates, and through what behavioural readout; (2) what cellular mechanism each paper calls "gain"; (3) what each paper predicts about the cellular-to-behavioural link; (4) whether any paper has actually run the explicit unification test the project is proposing. The headline answer to (4): **no paper in the corpus has tested cellular-gain reading and behavioural-hyperparameter reading simultaneously on the same substrate** — Lee 2024 comes closest but uses a hand-coded functional form, not an emergent FiLM-on-activations machine; Rodriguez-Garcia 2026 unifies cellular gain with optimiser-level hyperparameters (learning rate via gradient reparameterisation), not forward-pass hyperparameters. The unification at *forward-pass* level on an end-to-end-trained policy is the project's untaken gap.

## Reader's note on the source state

The 18 PDFs at `docs/project/references/neuromodulatory_algorithms/sources/` have **not yet been processed by `literature-reviewer` into per-paper deep-dives**. There is no master `_lit_review.md` and no `reviews/<paper-key>_deepdive.md` files for this topic. This synthesis therefore relies on (a) direct extraction from the source PDFs, and (b) the two professor reviews carried in §8 of the superseded v1 direction memo (`20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md`). The convergent corrections that the two professors landed on (Tsuda mis-citation, Doya/Yu-Dayan branch confusion, Rodriguez-Garcia differentiation, Lee 2024 as closest precedent) are absorbed here. The reader should treat this synthesis as a stand-in for the missing master review *for the specific cellular-gain ↔ behavioural-hyperparameter question*. A broader thematic master review would still be valuable for the corpus as a whole.

## Section 1 — What "behavioural hyperparameter modulation" means in this corpus

The corpus is consistent in inheriting Doya 2002's mapping of four ascending neuromodulatory systems to four scalar hyperparameters of a standard RL agent. Doya 2002 (§3, pp. 498-499) puts the mapping in its sharpest form: dopamine (DA) signals the TD error $\delta$, serotonin (5-HT) controls the discount factor $\gamma$, noradrenaline (NA) controls the inverse temperature $\beta$, and acetylcholine (ACh) controls the learning rate $\alpha$. Mei et al. 2022 (Trends in Neurosciences, p. 238) restates this directly: "neuromodulatory processes may fine-tune hyperparameters to dynamically adapt learning strategies of DNNs", and Figure 1B in Mei is the canonical diagram of the framing. Lee et al. 2024 (Fig. 2, p. 3) gives the most explicit modern restatement, annotating the Q-learning update and softmax policy with neuromodulator labels on each hyperparameter.

The behavioural readouts that the corpus uses for each hyperparameter are summarised below. Each readout is what an experimenter or simulator records to *prove* that the relevant hyperparameter has moved.

| Hyperparameter | What it controls | Behavioural readout used by the corpus | Canonical citation |
|---|---|---|---|
| $\alpha$ (learning rate) | How fast the value or policy parameters change per TD step | Post-switch adaptation speed; trial-to-trial regression of $\Delta Q$ on $\delta$; recovery curve after rule change | Doya 2002 §3.4; Lee 2024 §3.1 (ACh-$\alpha$); Mei 2022 Box 1 |
| $\beta$ (inverse temperature) | Stochasticity of action selection — high $\beta$ ↔ low entropy / exploitation; low $\beta$ ↔ high entropy / exploration | Policy entropy spike at regime change; choice-variability around uncertainty peaks; pupil-dilation-linked behavioural switches | Doya 2002 §3.3; Lee 2024 §3.1 (NA-$\beta^{-1}$); Mei 2022 Box 1; Wainstein 2025 (pupil-linked switches) |
| $\gamma$ (discount factor) | Time-horizon over which reward is summed; high $\gamma$ ↔ long-horizon patient choice, low $\gamma$ ↔ impulsive immediate-reward choice | Delay-discounting tasks; impulsivity assays; effective-horizon measured by autocorrelation of value or memory states | Doya 2002 §3.2 (impulsivity in 5-HT-depleted rats); Lee 2024 §3.2 (caveats serotonin theory is "insufficient convergent evidence") |
| Precision / signal-to-noise weighting | How much sensory evidence is trusted vs. prior; per-feature gating of input | Performance under noisy observations; aleatoric-vs-epistemic estimation in ensemble agents | Yu & Dayan 2005 (out of corpus but cited by Lee 2024 §3.2); Lee 2024 (Doya-DaYu agent §4) |

Two things to flag. First, the **ACh branch is contested in the corpus**: Doya 2002 maps ACh to learning rate $\alpha$; Yu & Dayan 2005 (out-of-corpus but cited by Lee et al. 2024 §3.2 and Ben-Iwhiwhu et al. 2022 §3.1) maps ACh to expected uncertainty / precision. Lee 2024's "Doya-DaYu agent" (§4) deliberately combines both: ACh signals an *uncertainty balance* (the ratio of aleatoric to total uncertainty), which is then used as the learning rate. Both professors flagged this in v1's §8 — the project's v2 §3 resolves it by choosing the Doya-$\alpha$ branch, which makes Section 3 predictions (c) testable on the project's three-injection-site architecture. Second, **the 5-HT/$\gamma$ branch is the weakest** in the corpus: Lee 2024 §3.2 explicitly writes "there is insufficient convergent evidence on serotonin for a single suitable interpretation under our RL framework". The project should not stake claims on the 5-HT-$\gamma$ mapping without further work.

## Section 2 — Mechanism inventory: what does the corpus call "neural gain"?

The corpus does not use one definition of "gain". The mechanism variants below are the ones the project's unification claim must thread through. The order is: where each variant acts (cellular substrate), what the canonical citation is, what mathematical form it takes, and which behavioural readout has been demonstrated.

| Variant | Where it acts | Mathematical form | Canonical citation in corpus | Behavioural readout demonstrated |
|---|---|---|---|---|
| **(a) Multiplicative gain on input-output curve (slope)** | Single cell, cellular response function | $Q_{\text{out}} = g \cdot f(I_{\text{in}})$, $g$ scales $dQ/dI$ | Ferguson & Cardin 2020 (Box 1, panel b; "Multiplicative" vs "Divisive"); Shine 2021 (Box 1, Fig. 1 — "Neural gain is the gradient (slope) of the input-output mapping function for a neuron, $dQ/dI$") | Contrast invariance in V1; attentional enhancement; arousal-correlated gain in mouse V1 |
| **(b) Additive bias / rheobase shift** | Single cell, operating point | $I_{\text{eff}} = I_{\text{in}} + b$ (or $V_{\text{thr}} \to V_{\text{thr}} - b$) | Ferguson & Cardin 2020 (Box 1, panel c — "additive (blue arrows) or subtractive operation… maintain the sensitivity of a neuron to different inputs but alter the input required to reach the threshold for a response") | Selectivity change without sensitivity change; iceberg-effect tuning sharpening |
| **(c) Multiplicative-on-weights / synaptic-weight scaling** | Network connectivity matrix | $W \to f_{nm} \cdot W$ (uniform on a region) | Tsuda et al. 2021 (Fig. 1b, §Results, p. 3 — "a simple multiplicative factor applied to weights already acts as a powerful network control device"; Extended Data Appendix A explicitly distinguishes from neural-excitability models) | Distinct activity "hypertubes" (PCA-space shifts of trajectory geometry) ↔ distinct context-relevant behaviours in Drosophila and toy networks |
| **(d) Structured / low-rank weight modulation** | Low-rank decomposition of recurrent matrix | $W_x(z) = \sum_k s_k(z) \ell_k r_k^\top$, with $s_k = \sigma(A_z z + b_z)$ | Costacurta et al. 2024 (Eq. 4, §3 — "Neuromodulatory subnetwork scales each rank-1 component of the recurrent weight matrix"; Prop. 1 shows equivalence to LSTM forget gate) | Generalisation gain over fixed-recurrence low-rank RNNs on neuroscience tasks; LSTM-style gating emerges |
| **(e) Gradient-level / optimiser gain** | Weight updates during training | $W_{ij}(t) = g_i(t) w_{ij}(t)$, decomposing into $w_{\text{slow}} = g_0 w$ and $w_{\text{fast}} = (g - g_0) w$; effective curvature $\lambda \to \lambda / g^2$ | Rodriguez-Garcia et al. 2026 (Eq. 2, §2.1; Algorithm 1: $g(t+1) = \gamma g(t) + (1-\gamma) g_0 + \eta H(y)$, with $H$ the entropy of the network output) | Stability-gap attenuation on Split MNIST / Split CIFAR-10 / Domain CIFAR-100 / mini-ImageNet under joint-training continual learning |
| **(f) FiLM / activation-level scale-and-shift** | Forward-pass activations | $\sigma_{NMN}(x, z; w_s, w_b) = \sigma(z^\top(x w_s + w_b))$ — per-feature multiplicative $w_s$ and additive $w_b$ read out from a scalar/vector $z$ | Vecoven et al. 2020 (§2, Fig. 1; the scalar $z$ is shared across all neurons in the main net, with per-neuron $w_s, w_b$ giving the per-feature fan-out) | Meta-RL navigation benchmarks; the project's R2 continual-learning win sits in this family |
| **(g) Hand-engineered functional-form scalar** | RL hyperparameter directly | $\alpha(s, a) = E/(E + A)$, $\beta(s) = 1/\langle E \rangle_a$, where $E$ is expected (aleatoric) and $A$ is unexpected (epistemic) uncertainty from an ensemble | Lee et al. 2024 (Eq. 4, §3.4 — Component IV: "Finding functional forms back to hyper-parameters") | Non-stationary multi-armed bandit recovery on switching-context tasks |
| **(h) Generic gating / activity-multiplier** | Per-layer activations | Modulator outputs $g$, applied as $h \to g \odot h$ before nonlinearity | Ben-Iwhiwhu et al. 2022 (Fig. 1b — "activity-gating neuromodulator multiplies the activity of the target neurons before a non-linearity is applied"); Beaulieu et al. 2020 (cited by Ben-Iwhiwhu); generic in the meta-RL literature | CAVIA / PEARL meta-RL benchmarks |
| **(i) Recurrent-state gain (Hopfield)** | Recurrent weight matrix in attractor net | $W_{\text{rec}} \to g \cdot W_{\text{rec}}$, with low $g$ flattening attractor landscape | Osman et al. 2024 (continuous Hopfield with gain parameter — "gain parameter that mimics arousal state by suppressing recurrent interactions"); Tambaş et al. 2025 (Krotov-Hopfield three-factor rule) | Annealing-schedule-like adaptation; arousal-modulated Bayesian inference |
| **(j) Spike-rate / excitability gain** | Spiking-network membrane params | Per-neuron multiplier on input current or excitability | AlKilany & Goodman 2025 (excitability-multiplier in spiking networks); Shine 2021 §"Cellular mechanisms" Fig. 2c-e | Reaction-time decrease ("listening in the dips" attentional gain in speech-in-noise) |

The variants are not all the same machine. The two cleavages that matter for the project are:

- **Activation-level vs weight-level vs gradient-level.** Variants (a), (b), (f), (h), (j) act on activations or excitability *forward of* the weights. Variants (c), (d), (i) act on the weight matrix at inference time. Variant (e) acts on the gradient at training time. Tsuda 2021 (Extended Data Appendix A, cited by Costacurta 2024 §2.1) is emphatic that activation-level and weight-level "operate through independent mechanisms both biologically and computationally". Rodriguez-Garcia 2026's optimiser-level gain is a third, distinct mechanism — both professors flagged in v1 §8 that v1 conflated this with the project's activation-level FiLM. The v2 memo §3 resolves the conflation.
- **Multiplicative arm vs additive arm.** Ferguson & Cardin 2020's Box 1 is the canonical distinction. Multiplicative (panel b) changes the slope of $f(I)$ without changing the firing threshold — sensitivity moves, selectivity stays. Additive (panel c) shifts the input required to fire — selectivity moves, sensitivity (slope) stays. FiLM's $\gamma \odot h + \beta$ instantiates both arms simultaneously. Tsuda 2021's "hypertubes" arise from the *multiplicative arm on weights*; the v1 memo's mis-citation of Tsuda as additive was the convergent error both professors flagged.

The project's FiLM-on-policy-activations is in family (f), Vecoven-style. It is *not* Tsuda (which is (c), multiplicative on weights), it is *not* Costacurta (which is (d), structured low-rank), and it is *not* Rodriguez-Garcia (which is (e), gradient-level). The closest published precedents for the project's substrate are Vecoven 2020 (architecture) and Lee 2024 (the explicit cellular-to-hyperparameter framework, but with hand-coded functional forms instead of an end-to-end-learned FiLM).

## Section 3 — Predictions already made by the corpus

This is the evidence base for v3 §5. Each prediction below is a published or implied statement of the form "if cellular-level gain machine X is in the loop, then behaviour-level quantity Y should move in direction Z, measurable as W". I have grouped predictions by the hyperparameter (the behavioural-level variable) they are about, with the cellular mechanism cited alongside.

### 3.1 Predictions about learning rate ($\alpha$) — ACh and the Doya-α branch

- **(P3.1.1) Lee et al. 2024 §3.1 + Fig. 2** — *Cellular: ACh-modulated synaptic plasticity (Hasselmo & Bower 1993, cited p. 3); ACh signals an "uncertainty balance" (Marshall et al. 2016, expected vs unexpected uncertainty).* **Predicts**: in a non-stationary multi-armed bandit, an agent whose learning rate is $\alpha(s,a) = E(s,a) / (E(s,a) + A(s,a))$ — where $E$ is aleatoric (expected) and $A$ is epistemic (unexpected) uncertainty — should outperform a fixed-$\alpha$ Boltzmann agent and a Discounted-UCB baseline at every context switch. **Measure used**: per-block regret curves across context boundaries; the Doya-DaYu agent in Fig. 3. **The closest published precedent to the project's claim**. The project differs in (i) sequential RL (PPO) not bandits, (ii) FiLM with $\gamma, \beta \in \mathbb{R}^d$ per site instead of a single $\alpha$ scalar, (iii) the mapping is learned end-to-end, not hand-coded as Lee's Component IV.

- **(P3.1.2) Mei et al. 2022 Box 1 + Fig. 2** — *Cellular: ACh prospectively gates spike-timing-dependent plasticity (STDP) at sub-second to second timescales.* **Predicts**: a DNN with an artificial ACh-analogue modulator should adapt faster to task drift than a fixed-hyperparameter DNN; the modulator's activity should track post-switch instability. **Measure used**: the paper is a review and does not run the experiment, but Fig. 1B is the canonical diagram of "neuromodulation → hyperparameter fine-tuning" that the project's thesis instantiates.

- **(P3.1.3) Doya 2002 §3.4** — *Cellular: ACh modulates the strength of recurrent vs sensory input in cortex (Hasselmo); striatal ACh-DA balance modulates plasticity gating.* **Predicts**: behavioural memory speed (how fast old associations are overwritten by new) should track ACh tone. **Measure used**: rate-of-association-update in conditioning paradigms; the prediction is qualitative.

### 3.2 Predictions about inverse temperature ($\beta$) / exploration — NA and the adaptive-gain theory

- **(P3.2.1) Doya 2002 §3.3** — *Cellular: NA via Aston-Jones-Cohen adaptive-gain theory increases input-output gain of cortical neurons (Servan-Schreiber et al. 1990; Aston-Jones & Cohen 2005).* **Predicts**: higher NA → higher $\beta$ → lower action entropy → more exploitative behaviour. **Measure used**: action-entropy / softmax-temperature regression on putative NA tone.

- **(P3.2.2) Lee et al. 2024 §3.4, Eq. 4** — *Cellular: NA signals **unexpected** uncertainty (Yu & Dayan 2005); tonic NA mediates exploration (Aston-Jones & Cohen 2005).* **Predicts**: $\beta^{-1}(s) = 1/\langle E(s, \hat a)\rangle_{\hat a}$ — higher mean aleatoric uncertainty across actions raises temperature. Phasic NA bursts produce the opposite (focused exploitation). **Measure used**: per-trial $\beta$ as a function of ensemble-derived uncertainty; the Doya-DaYu agent.

- **(P3.2.3) Wainstein et al. 2025** — *Cellular: gain on a trained RNN's activation function, varied as a function of categorisation uncertainty (Shine 2021's adaptive-gain reading of NA).* **Predicts**: (i) at perceptual-switch moments, pupil diameter (a non-specific proxy for LC-NA tone) peaks; (ii) higher gain produces faster perceptual switches by destabilising the network's dynamical regime under maximal uncertainty; (iii) macroscale brain trajectories should show higher low-dimensional velocity and a flattened energy landscape around switch points. **Measures used**: pupil-diameter peri-switch time-locking (Fig. 1C); RNN-trajectory low-D velocity; fMRI-derived energy-landscape flatness. **All three predictions confirmed empirically**. This is the strongest published evidence that cellular-level gain causally moves a perceptual-switch behavioural quantity, *but the behavioural quantity Wainstein measures is "switch latency" — a perceptual-decision variable, not directly an RL hyperparameter like $\beta$*. The bridge to RL $\beta$ is a project-level extrapolation.

- **(P3.2.4) Ben-Iwhiwhu et al. 2022 §3** — *Cellular: activity-gating neuromodulator multiplies layer activations before nonlinearity.* **Predicts**: a context-meta-RL agent (CAVIA / PEARL backbone) augmented with a per-layer activity-gating modulator should produce more dissimilar (richer) latent representations across tasks than a non-modulated baseline, and adapt faster on held-out tasks. **Measures used**: latent-representation dissimilarity across tasks; per-task return curves. The connection to $\beta$ is implicit — richer representations enable a more selective policy — but is not directly tested as a temperature movement.

- **(P3.2.5) Mei et al. 2022 Box 1** — *Cellular: LC-NA tonic-phasic firing modes (Aston-Jones et al. 1994).* **Predicts**: a $\beta$-parameter softmax policy whose $\beta$ is modulated by an LC-analogue should oscillate between exploration (high tonic NA) and exploitation (high phasic NA) modes. **Measure used**: the paper is a review, not an experiment.

### 3.3 Predictions about discount factor ($\gamma$) — 5-HT and impulsivity

- **(P3.3.1) Doya 2002 §3.2** — *Cellular: 5-HT modulates the balance of direct vs indirect striatal pathways; experimentally, 5-HT depletion increases impulsive choice in rats (Mobini et al. 2000).* **Predicts**: lower 5-HT tone → lower effective $\gamma$ → choice of small immediate reward over large delayed reward. **Measure used**: delay-discounting paradigms; impulsivity assays.

- **(P3.3.2) Lee et al. 2024 §3.2 (the "we do not commit" prediction)** — Lee explicitly declines to commit to a 5-HT / $\gamma$ functional form: "there is insufficient convergent evidence on serotonin for a single suitable interpretation under our RL framework." **This is a negative prediction**: the corpus does not support a clean cellular-gain-to-effective-discount test as of 2024. The project should not stake claims on the 5-HT branch.

### 3.4 Predictions about precision / signal-to-noise — the Yu-Dayan ACh branch

- **(P3.4.1) Lee et al. 2024 §3.2** — *Cellular: ACh signals expected uncertainty (Yu & Dayan 2005); blocking ACh induces over-reliance on priors (Marshall et al. 2016).* **Predicts**: an agent whose precision (or learning rate) tracks expected uncertainty should outperform a fixed agent under stationary noisy observations; the gating direction reverses for unexpected uncertainty (NA territory). **Measure used**: Lee folds this into the Doya-DaYu $\alpha$ formula via "uncertainty balance"; not separately tested.

- **(P3.4.2) Osman et al. 2024** — *Cellular: gain parameter on a Hopfield network suppresses recurrent interactions and amplifies bottom-up input (the canonical arousal-as-gain hypothesis).* **Predicts**: behavioural arousal state should manifest as a Bayesian-inference temperature schedule; low gain → broad posterior, high gain → sharp posterior. **Measure used**: the paper connects the Hopfield gain dynamics to a statistical-physics annealing schedule and Boltzmann-machine inference; behavioural validation is by appeal to known arousal-behaviour curves.

### 3.5 Predictions about the unification *between* cellular and behavioural levels (the project's specific question)

These are the predictions that most directly anticipate the project's claim. They are the ones the four professors should cite when proposing predictions for v3 §5.

- **(P3.5.1) Mei et al. 2022 Fig. 1B** — *The most direct statement in the corpus of the unification thesis*: a single artificial "neuromodulation unit" should reconfigure DNN hyperparameters as a function of cognitive / behavioural state; the unit is conceptualised as a parallel of the four-system biological arrangement. **The paper does not run the experiment**; Mei et al. propose it as a research programme. **The project's v2 is the operationalisation of this proposal**.

- **(P3.5.2) Lee et al. 2024 framework (Components I–IV)** — *The most explicit unification framework in the corpus.* Lee's framework demands that any neuromodulation-inspired RL agent commit to (I) what the modulator does (the hyperparameter it tunes), (II) what it signals (the cellular / cognitive quantity it tracks), (III) what artificial analogue measures the same quantity, and (IV) the functional form mapping (III) to (I). **Lee's contribution at level (IV) is hand-coded**; the project's contribution is to learn (IV) end-to-end via a FiLM substrate. **Lee predicts**: a cleanly-defined hyperparameter modulator should outperform fixed-hyperparameter baselines specifically at non-stationarity boundaries — empirically confirmed in their bandit (Fig. 3).

- **(P3.5.3) Rodriguez-Garcia et al. 2026 §2.1** — *Implicit unification at the optimiser level.* The cellular gain $g(t)$ is shown to be mathematically equivalent to a fast-slow weight decomposition, and the effective learning rate is reparameterised as $\lambda \to \lambda / g^2$. **The cellular-to-hyperparameter bridge is the math itself**: a single time-varying gain on weights *is* a state-dependent learning rate. **Predicts**: under transient gain boosts at task boundaries, the stability gap should attenuate — empirically confirmed across MNIST / CIFAR / mini-ImageNet (their §3-4). **Differentiation from the project**: their unification is at the *optimiser* level (gradient × gain), not the *forward-pass* level (activation × gain). The two are orthogonal and could combine — see §6.

- **(P3.5.4) Costacurta et al. 2024 Prop. 1 + Fig. 3F** — *Implicit unification at the gating level.* The paper proves that a low-rank-weight-scaling NM-RNN is equivalent to an LSTM with a learned forget-gate. Behaviourally, ablating individual neuromodulatory dimensions has *dissociable* effects on different task computations (Fig. 3F). **Predicts**: if a multi-dimensional modulator learns to dedicate different latent dimensions to different sub-computations, lesioning each dimension should produce a dissociable behavioural deficit. **The corresponding project claim** (v2 prediction (c), per-site clamps) is the direct analogue; Costacurta's Fig. 3F is the closest published precedent for the dissociation-by-ablation experimental design.

- **(P3.5.5) Tsuda et al. 2021 Fig. 3-4** — *Cellular: multiplicative weight scaling; the only well-developed activity-space prediction in the corpus.* **Predicts**: the same recurrent network, under different scalar neuromodulator settings, should produce distinct context-relevant behaviours through *trajectory shifts in activity space — "hypertubes"*. **Measure used**: PCA of recurrent state space across neuromodulator levels; demonstration on a Drosophila behavioural paradigm. **Crucially**: Tsuda 2021's mechanism is *multiplicative on weights*, not on activations. Hypertubes are an *effect*, not the project's mechanism. The project should expect a Tsuda-analogue *if* the FiLM machinery learns to behave like a Costacurta-style structured weight scaler — testable via the effective-rank-of-$\gamma$ probe `professor-rl-bayesian-dl` proposed.

- **(P3.5.6) Shine et al. 2021** — *The bridge paper for the unification thesis.* Reviews the cellular biophysical mechanisms (G-protein cascades, ion channel modulation, dendritic apical amplification) and the systems-level effects (Yerkes-Dodson, energy-landscape flattening) of neuromodulation. **Implicit prediction**: cellular gain mediates large-scale brain-state shifts that in turn produce behavioural-state shifts. The paper is a *review* and proposes no falsifiable test of its own, but supplies the cellular vocabulary the project's claim depends on.

- **(P3.5.7) Vecoven et al. 2020 §3-4** — *Architectural unification.* A neuromodulatory subnetwork outputs a scalar / low-D signal that produces per-feature multiplicative $w_s$ and additive $w_b$ in the main net's activation function; this is the FiLM machine. **Predicts**: NMN-augmented A2C should outperform plain RNN-A2C on meta-RL navigation. **Measure used**: cumulative reward over meta-RL episodes; latent $z$-recruitment patterns (Fig. 7) show how many effective scalars the network uses — this is exactly the effective-rank probe `professor-rl-bayesian-dl` named.

- **(P3.5.8) Wainstein et al. 2025** — *The empirical anchor for the cellular-to-behavioural link*, but with the caveat that the behaviour measured is *perceptual switching*, not RL hyperparameter movement. Pupil-NA-gain-switch is a chain of observations the project's interoceptive-RL framing could extend; the extension is the project's contribution.

- **(P3.5.9) AlKilany & Goodman 2025** — *Cellular: excitability-multiplier in spiking networks; reaction-time decrease ("listening in the dips") emerges without explicit training.* **Predicts**: a per-neuron excitability multiplier in a spiking network produces dynamic-gain control that decreases reaction time, mimicking the cholinergic-attention reading. **Measure used**: reaction-time distributions in speech-in-noise tasks; performance under high-noise conditions.

### 3.6 The conspicuous absence: the bidirectional explicit-unification test

Section 3.1-3.5 lists 20+ published predictions linking cellular gain to behavioural quantities. None of them runs **the symmetric two-direction test**: take one substrate, log *both* the cellular gain signature (slope of effective input-output curve at a target site) *and* the behavioural hyperparameter signature (entropy spike / adaptation curve / per-feature precision movement) on the *same* trials, and check whether the two move together. Lee 2024 logs the bandit-recovery curve but not a cellular-style gain signature; Rodriguez-Garcia 2026 logs the loss-landscape flattening but not a behavioural entropy / temperature movement; Wainstein 2025 logs pupil-and-RNN-gain together but the behavioural readout is perceptual switch latency, not an RL hyperparameter; Costacurta 2024 demonstrates dissociation-by-ablation but does not show the dissociation corresponds to *named hyperparameter* movements. **The two-channel readout — gain at the activation level *and* effective hyperparameter at the behaviour level, on the same run, on the same agent — is the project's untaken gap.** See §5.

## Section 4 — Measures already tried by the corpus

For each behavioural-hyperparameter prediction in §3, here is the measurement the cited paper used. This is a service table for the postdoc and the four professors: anything operationally feasible on the project's testbed is worth proposing in v3 §5.

| Prediction ID | Measure | Operationalisation in source paper | Translatable to project's grid-world RL testbed? |
|---|---|---|---|
| P3.1.1 Lee — $\alpha$ from uncertainty balance | Recovery curve at block boundaries | Cumulative regret vs episode in non-stationary bandit; ensemble-based aleatoric / epistemic uncertainty per arm (Eqs. 2-3) | Yes — analogue is post-stage-switch survival-step recovery; ensemble disagreement on Q-values |
| P3.1.2 Mei — ACh modulates STDP | Conceptual / review | Not run | Not directly — STDP is not in PPO. Analogue is effective per-feature learning-rate-on-activations |
| P3.2.1 Doya — $\beta$ from NA gain | Action-entropy curve | Conceptual | Yes — actor entropy is already logged |
| P3.2.2 Lee — $\beta$ from ensemble | $\beta(s) = 1/\langle E(s, \hat a)\rangle$ | Ensemble-based $E$ averaged over actions | Yes — requires PPO ensemble or value-head variance proxy |
| P3.2.3 Wainstein — gain destabilises switch | Pupil-locked switch time; RNN low-D velocity; fMRI energy-landscape flatness | Peri-switch pupil-time-lock (Fig. 1C); RNN trajectory velocity around switch; second derivative of low-D Hamiltonian | Partial — pupil unavailable; **low-D velocity of `mod_h` around regime change is the direct project analogue** |
| P3.2.4 Ben-Iwhiwhu — activity-gating richness | Latent-representation dissimilarity across tasks (cosine across context vectors) | CAVIA / PEARL latent embeddings | Yes — `mod_h`-state dissimilarity across regimes |
| P3.3.1 Doya — 5-HT / $\gamma$ | Delay-discounting curves | Lever-press choice between immediate-small vs delayed-large reward | Hard — project has no native delay-reward dissociation in current grid-world |
| P3.4.1 Lee — ACh / expected uncertainty | Ensemble-based aleatoric uncertainty | Variance over the return distribution (Eq. 5 in Lee) | Yes — Q-value-distribution variance over an ensemble |
| P3.5.3 Rodriguez-Garcia — gain attenuates stability gap | Continual-evaluation old-task accuracy curves under joint training | Per-batch evaluation every $\rho$ iterations; measure transient accuracy drop at task boundary (Hess et al. 2023 protocol) | Yes — analogue is fine-grained survival-step evaluation across stage boundaries; **the project's R2 win is plausibly a stability-gap-attenuation signature** |
| P3.5.4 Costacurta — dissociable ablation | Per-modulator-dimension ablation with task-specific deficit (Fig. 3F) | Lesion each $s_k$ dimension; measure per-task return drop | Yes — **the project's per-injection-site clamps (v2 prediction c) are the direct analogue** |
| P3.5.5 Tsuda — hypertubes | PCA of recurrent activity across neuromodulator levels (Fig. 3-4) | RNN hidden-state PCA, coloured by modulator level | Yes — PCA of GRU hidden state colour-coded by `mod_h` |
| P3.5.7 Vecoven — $z$-recruitment | Distribution of $z$-component magnitudes across tasks (Fig. 7) | Per-dimension activation statistics of the modulator output | Yes — **the effective-rank-of-$\gamma$ probe `professor-rl-bayesian-dl` named** |
| P3.5.8 Wainstein — pupil-switch chain | Peri-switch pupil-time-lock + RNN-gain-modulated switch latency | Eye-tracker (1000 Hz) + fMRI | Partial — no pupil; **`mod_h` time-locked to value-prediction-error magnitude is the project analogue** |

The probes that translate cleanly: actor-entropy peri-switch (P3.2.1), `mod_h` low-D velocity around regime change (P3.2.3), `mod_h`-state dissimilarity across regimes (P3.2.4), per-injection-site ablation (P3.5.4), `mod_h` PCA coloured (P3.5.5), effective-rank of $\gamma$ (P3.5.7), `mod_h` time-locked to TD-error magnitude (P3.5.8), continual-evaluation transient accuracy drop at stage boundary (P3.5.3). The probes that don't: pupillometry, STDP, delay-discounting (no native delay-reward task), fMRI.

## Section 5 — Gaps and what v3 §5 could legitimately fill

The corpus has **partial** unifications but **no complete one**. Below are the specific gaps that the project's substrate is positioned to fill — i.e. the predictions v3 §5 can stake claims on without colliding with prior art.

### 5.1 Gap 1 — The bidirectional explicit-unification test

**No paper in the 18-paper corpus logs cellular-gain and behavioural-hyperparameter on the same agent at the same time and shows them moving together.** This is what §3.6 already flagged. Closest precedents:

- Lee 2024 logs the hyperparameter ($\alpha$, $\beta$) but specifies it by a hand-coded formula; the cellular-gain analogue is not separately logged.
- Rodriguez-Garcia 2026 logs the loss-landscape curvature (a cellular-analogue) and the stability-gap accuracy (a behavioural quantity), but the substrate is supervised classification, not RL — there is no $\alpha, \beta, \gamma$ to track separately.
- Wainstein 2025 logs the gain (in the trained RNN) and pupil-and-fMRI (the cellular-analogue + macroscale readouts), but the behavioural readout is perceptual-switch latency, not an RL hyperparameter.
- Costacurta 2024 logs the modulator's latent and shows dissociation-by-ablation, but does not measure the dissociation as named-hyperparameter movements (e.g., one ablated dimension producing a temperature shift, another a learning-rate shift).

The unification *as a single test of named hyperparameter movement co-varying with the cellular-style gain readout, on a single forward-pass-modulated agent*, **has not been published**. The project's three-injection-site FiLM is positioned to run it. **v3 §5 can legitimately stake the headline claim on this gap.**

### 5.2 Gap 2 — Forward-pass gain (not gradient, not weights) as a hyperparameter modulator

Rodriguez-Garcia 2026 is the corpus's clearest unification, but it is *optimiser-level*: the gain multiplies the gradient, not the forward pass. The professors' v1-§8 convergent note flagged this as the load-bearing differentiation:

> "Rodriguez-Garcia modulates the gradient (optimiser-level — gain reparameterises learning rate as $\lambda \to \lambda / g^2$); this project modulates the forward activation (PPO's Adam untouched)."

There is no published demonstration that forward-pass gain alone (no optimiser-level coupling) produces a measurable RL-hyperparameter signature. Lee 2024 is forward-pass-only but hand-coded; Vecoven 2020 is forward-pass-only and learned but does not log a hyperparameter-style readout. **v3 §5 can legitimately predict that the project's forward-pass FiLM produces a learned-but-explicit hyperparameter-style movement at each of its three injection sites.**

### 5.3 Gap 3 — Multi-site dissociation of named hyperparameter channels

Costacurta 2024 Fig. 3F shows dissociable ablation effects in a multi-dimensional neuromodulator, but does not assign each dimension to a named hyperparameter. The corpus does *predict* multi-channel dissociation (Doya 2002 §3 partitions across four anatomically distinct systems; Lee 2024 uses two separately-parameterised modulators) but no published paper has run the experiment "ablate site X → degrade hyperparameter Y" on a multi-site FiLM substrate. **v3 §5 prediction (c) — per-injection-site clamps degrading hyperparameter-distinguishable behaviours — fills this gap.**

### 5.4 Gap 4 — End-to-end-learned (vs hand-coded) mapping

Lee 2024 Component IV is the explicit mapping from cellular signal to hyperparameter (Eq. 4: $\alpha = E/(E+A)$ etc.). It is hand-coded. The corpus does not publish a result where this mapping is *learned* end-to-end via gradient descent on the policy objective. Vecoven 2020 learns the modulator end-to-end but does not interpret its outputs as named hyperparameters. **v3 §5 can legitimately predict that the project's `mod_h`-to-FiLM mapping reorganises through training to approximate a Doya-style hyperparameter modulator, with the mapping recoverable post-hoc**.

### 5.5 Gap 5 — Phasic vs tonic dissociation on a single-substrate forward-pass machine

Aston-Jones & Cohen 2005 phasic-vs-tonic NA is foundational background (cited in Lee 2024 §3.4, Wainstein 2025, Mei 2022, Rodriguez-Garcia 2026). The corpus consistently invokes the dissociation. But no published forward-pass FiLM-style machine in the corpus has *two recurrent timescales* with separable phasic and tonic readouts. Rodriguez-Garcia's $g(t) = \gamma g(t-1) + (1-\gamma) g_0 + \eta H$ is one-time-constant. Vecoven's RNN modulator is one-time-constant. Tsuda's $f_{nm}$ is held constant per simulation. **The project's Phase 0 T/P split, if implemented, fills this gap.** Both professors flagged this in v1 §8 as a *prerequisite* (not just an open question) — the unification claim is non-identifiable without it.

### 5.6 Gap 6 — Pain/interoceptive validation of neuromodulator-as-hyperparameter

None of the 18 papers ties the cellular-to-hyperparameter unification to nociception, pain, or interoception. The project's testbed (grid-world with predator-avoidance and noxious stimuli, per the project plan) is a domain-novel substrate for the unification claim. **v3 §5 can legitimately flag this as a downstream contribution** — but this is a positioning claim, not a corpus-grounded prediction.

## Section 6 — Methodological notes for the prediction designers

The four professors and the postdoc are about to propose predictions for v3 §5. This section lists what is operationally feasible on the project's testbed (partially-observed grid-world RL with PPO actor-critic, FiLM at three injection sites — A encoder, B GRU update gate, C policy temperature — and a modulator with single recurrent state).

### 6.1 Measures that translate cleanly

- **Actor entropy peri-switch (P3.2.1, Doya-NA-$\beta$).** Already logged. Window: 50-200 episodes pre-switch vs 50-200 episodes post-switch. Project should observe a phasic spike at $t=0$ followed by relaxation.
- **`mod_h` time-locked to value-prediction-error magnitude (P3.5.8, Wainstein-analogue).** Requires logging `mod_h` and $|\delta|$ at the same cadence. Correlation in regime-change windows is the falsifier for the Doya-channel reading vs the context-detector reading.
- **`mod_h` low-D velocity around regime change (P3.2.3, Wainstein-direct-analogue).** PCA on `mod_h` trajectory + first-derivative across switch window.
- **Effective rank of $\gamma$ across regime changes (P3.5.7, Vecoven Fig. 7 analogue).** SVD on the FiLM $\gamma$ matrix concatenated across episodes. Collapse to rank-1 at site C ↔ Doya-style 1-knob temperature claim. High rank ↔ overparameterised, non-Doya.
- **Per-injection-site clamp drop (P3.5.4, Costacurta Fig. 3F analogue).** Set $\gamma=1, \beta=0$ at one site at a time; measure survival-step drop on regime-change tasks. Dissociation across sites supports the multi-channel reading.
- **Continual-evaluation stability-gap measurement (P3.5.3, Rodriguez-Garcia Hess-protocol).** Fine-grained eval every $\rho$ episodes across stage boundaries; per-stage transient drop quantifies the stability gap. The project's R2 win is plausibly an attenuated stability gap.
- **$\gamma$-freeze vs $\beta$-freeze at Injection C (v1 §8 b′ falsifier from `professor-rl-bayesian-dl`).** Three-arm experiment, small cost. Direct test of multiplicative-arm-as-NA-temperature vs additive-arm-as-hypertube-shift.

### 6.2 Measures that don't translate

- **Pupillometry.** No analogue in a simulated agent. Wainstein 2025 used the pupil as a non-specific proxy for LC-NA; the project's `mod_h` is the direct cellular-analogue but without the noise the pupil signal carries.
- **fMRI / macroscale brain imaging.** Not applicable to a small actor-critic.
- **STDP / spike-timing.** PPO is batch-trained on advantage estimates, not spike-time-driven. The closest analogue is per-feature effective learning rate, which is what FiLM at site A already implements.
- **Delay-reward / impulsivity** (P3.3.1, Doya 5-HT-$\gamma$). The current grid-world does not natively dissociate immediate from delayed reward. Without a redesigned task, the 5-HT-$\gamma$ branch is not testable. This is *fine*: the corpus already calls 5-HT the weakest branch (Lee 2024 §3.2).

### 6.3 Identifiability prerequisites

Both professors converged on this in v1 §8: the **T/P split is a prerequisite**, not an open question. A single recurrent `mod_h` produces a non-identifiable read-out across the three injection sites — any rotation of the per-site read-out heads that preserves their projection onto policy-relevant subspaces is observationally equivalent. The Phase 0 T/P split (two recurrent states, separable timescales) is what makes the multi-channel claim testable. v3 §5 predictions should explicitly note this conditional.

### 6.4 Constraint on $\gamma$ — softplus / non-negativity

`professor-neuromodulation`'s v1 §8 Q7 minor flag: Rodriguez-Garcia 2026 explicitly bounds $g(t) \geq g_0 \geq 1$. If the project wants "gain" to be the headline word, the FiLM $\gamma$ should be constrained $\gamma \geq 0$ (softplus). This is a `senior-developer` / `experiment-designer` item; v3 §5 predictions that depend on multiplicative interpretation should flag the constraint as a prerequisite.

## Section 7 — Reading guide (compact)

A one-line gloss per load-bearing paper, telling each professor exactly which section / equation / figure to cite when proposing predictions.

- **Doya 2002** — *§3 (pp. 498-499), and especially Eq. 5 (the softmax with inverse temperature $\beta$) and §3.4 (ACh-$\alpha$).* Foundational hyperparameter mapping. Cite for the four-knob mapping and the Doya-$\alpha$ branch (rather than Yu-Dayan precision).
- **Ferguson & Cardin 2020** — *Box 1, panels b and c (p. 81).* Canonical multiplicative-vs-additive distinction. Cite for the additive-arm anchor (rheobase shift); the project's $\beta$ corresponds to the additive arm there.
- **Shine et al. 2021** — *Box 1 + Fig. 1 (p. 766); Fig. 3 (p. 768) for the cellular-to-systems bridge.* Cite for the formal definition of neural gain as $dQ/dI$, and for the bridge from cellular biophysics to systems-level dynamics.
- **Lee et al. 2024** — *§3.1 + Fig. 2 (p. 3) for the framework; Eq. 4 (p. 5) for the hand-coded mapping; §3.2 (p. 4) for the 5-HT caveat.* The closest published precedent to the project's claim. Cite for the four-component framework that v3 is operationalising.
- **Vecoven et al. 2020** — *§2 + Fig. 1 (p. 3) for the FiLM-on-activations form $\sigma_{NMN}(x, z; w_s, w_b) = \sigma(z^\top(x w_s + w_b))$; Fig. 7 for the $z$-recruitment / effective-rank evidence.* Cite for the architectural substrate.
- **Tsuda et al. 2021** — *Fig. 1b (p. 4), Results §1 (p. 3); Extended Data Appendix A for the explicit distinction from neural-excitability models.* Cite for the multiplicative-weight-scaling story, NOT for the additive arm. Hypertubes (Fig. 3) are an *effect* of the multiplicative arm, not the additive arm.
- **Costacurta et al. 2024** — *Eq. 4 + §3.1 (p. 4) for the low-rank scaling; Prop. 1 for the LSTM-equivalence; Fig. 3F for the dissociation-by-ablation.* Cite for the structured-flexibility intermediate, and for the multi-channel dissociation as a published precedent for v3 prediction (c).
- **Rodriguez-Garcia et al. 2026** — *§2.1 + Eq. 2 (p. 4) for the fast-slow weight decomposition; Algorithm 1 for the gain dynamics; Fig. 2 for the loss-landscape flattening.* Cite for the optimiser-level unification, and for the explicit differentiation from forward-pass FiLM (per v1 §8 convergent note).
- **Wainstein et al. 2025** — *Fig. 1 (p. 3) for the pupil-switch chain; Fig. 4-5 for the RNN-gain-modulated trajectory velocity and energy-landscape flatness.* Cite for the empirical anchor that cellular gain causally moves a switch-latency behavioural quantity.
- **Mei et al. 2022** — *Fig. 1B (p. 239) — the canonical diagram of "neuromodulation → hyperparameter fine-tuning"; Box 1 for the actor-critic + Doya restatement.* Cite for the most direct statement of the unification thesis in the corpus.
- **Durstewitz et al. 2025** — *§"Four major categories" of continual-learning solutions; the optimiser-dynamics category (where Rodriguez-Garcia 2026 sits).* Cite for the framing that gain-as-hyperparameter is one of four CL solutions, not "the" mechanism — supports v2's redirect from "NMN helps continual learning" to "continual learning is one readout of hyperparameter movement".
- **Ben-Iwhiwhu et al. 2022** — *§4 + Fig. 1b (p. 4) for the activity-gating modulator; §3 for the Doya-mapping recapitulation.* Cite for an activity-gating-style neuromodulator in CAVIA / PEARL, the closest precedent for activity-level gating in meta-RL.
- **Wang et al. 2024 (Neuromodulated Meta-Learning)** — *§I-III for the FNS (flexible network structure) motivation; §IV-V for the bi-level optimisation; the structure-mask formulation is a hypernetwork-adjacent alternative to FiLM.* Cite as a complementary architectural alternative — Wang 2024 is meta-learning, not sequential RL.
- **Driscoll et al. 2022** — *Figs. 2-3 for dynamical-motif reuse across tasks.* Cite as the architectural-touchstone for the "multi-task RNN reuses shared dynamical motifs" framing; relevant to the question of whether the project's three injection sites learn dissociable motifs.
- **Osman et al. 2024** — *§3-4 for the Hopfield + gain construction; the annealing-schedule interpretation.* Cite for an arousal-as-gain Bayesian-inference reading; complementary to the RL-hyperparameter reading.
- **AlKilany & Goodman 2025** — *Results §"listening in the dips" for the emergent-attentional-gain finding.* Cite for evidence that excitability-gain produces measurable behavioural improvements without explicit training pressure.
- **Tambaş et al. 2025** — *Krotov-Hopfield three-factor rule.* Cite for the global-modulatory-signal-modulates-local-plasticity framing.
- **Wang et al. 2025 (NEST)** — Trajectory prediction in autonomous driving. *Low relevance to the cellular-gain ↔ behavioural-hyperparameter unification.* Skip for v3 §5.

## Appendix — corpus state and recommendations

### Per-paper coverage in this synthesis (corpus state)

| Paper | Read in full | Predictions table 3.X | Mechanism in §2 | Measures in §4 | Reading-guide line |
|---|---|---|---|---|---|
| Doya 2002 | Yes | 3.1.3, 3.2.1, 3.3.1 | (Doya inherits all variants) | All four hyperparameters | Yes |
| Ferguson & Cardin 2020 | Yes (Box 1) | implicit in (a), (b) | (a), (b) | n/a | Yes |
| Shine 2021 | Yes (intro + Box 1) | 3.5.6 | (a) | n/a | Yes |
| Lee 2024 | Yes (§1-4) | 3.1.1, 3.2.2, 3.3.2, 3.4.1, 3.5.2 | (g) | All | Yes |
| Vecoven 2020 | Yes (§1-3) | 3.5.7 | (f) | $z$-recruitment | Yes |
| Tsuda 2021 | Yes (Abstract + Intro) | 3.5.5 | (c) | PCA hypertubes | Yes |
| Costacurta 2024 | Yes (Abstract + §1-3) | 3.5.4 | (d) | Dissociation-by-ablation Fig. 3F | Yes |
| Rodriguez-Garcia 2026 | Yes (Abstract + §1-2) | 3.5.3 | (e) | Stability-gap continual-evaluation | Yes |
| Wainstein 2025 | Yes (Abstract + §1-2 of Results) | 3.2.3, 3.5.8 | (a) on a trained RNN | Pupil + RNN-velocity + fMRI flatness | Yes |
| Mei 2022 | Yes (Intro + Box 1) | 3.1.2, 3.2.5, 3.5.1 | review of all | n/a (review) | Yes |
| Durstewitz 2025 | Yes (Abstract + §1-2) | (framing) | review | n/a (review) | Yes |
| Ben-Iwhiwhu 2022 | Yes (Abstract + §1-4) | 3.2.4 | (h) | Latent dissimilarity | Yes |
| Wang 2024 (NeuronML) | Yes (Abstract + §1-2) | (architectural alternative) | (related to (d)) | FNS measures | Yes |
| Driscoll 2022 | Abstract only | (architectural touchstone) | n/a | Dynamical-motif identification | Yes |
| Osman 2024 | Abstract only | 3.4.2 | (i) | n/a | Yes |
| AlKilany & Goodman 2025 | Abstract only | 3.5.9 | (j) | Reaction-time | Yes |
| Tambaş 2025 | Abstract only | (three-factor framing) | n/a | n/a | Yes |
| Wang 2025 (NEST) | Abstract only | n/a | n/a | n/a | Skip |

**Coverage assessment.** All 18 papers in the corpus are at least abstract-scanned. The 11 load-bearing papers (Doya, Ferguson & Cardin, Shine, Lee, Vecoven, Tsuda, Costacurta, Rodriguez-Garcia, Wainstein, Mei, Durstewitz) are read in their predictions-relevant sections.

### Flag for the postdoc

The v2 memo is internally consistent with this synthesis. Two minor refinements the postdoc could absorb into v3:

1. **Wainstein 2025 caveat in v2 §3.** The v2 memo cites Wainstein as evidence for gain-modulated regime-change behaviour, but does not flag that Wainstein's *behavioural* measure is perceptual-switch latency, not an RL hyperparameter. The project's extension from "perceptual switch" to "RL hyperparameter movement" is the project's contribution, not Wainstein's. v3 §5 should make the extrapolation explicit.

2. **Costacurta 2024 dissociation-by-ablation as the named precedent for prediction (c).** v2 prediction (c) (per-site clamps) is the project's analogue of Costacurta Fig. 3F. The synthesis suggests v3 cite Costacurta explicitly when stating prediction (c).

Neither refinement constitutes a factual error in v2. They are framing strengthenings. No append-to-v2 feedback is required — the postdoc can incorporate them in v3 directly.

### Recommended downstream hand-offs

- **`research-postdoc` / `professor-neuromodulation` / `professor-rl-bayesian-dl` / `professor-bayesian-brain` / `professor-pain-modeling`** — author v3 §5 predictions using §3 of this synthesis as the evidence base and §4 as the measure-translation table.
- **`literature-reviewer`** — separately, this synthesis substitutes for a missing master review on the cellular-gain ↔ behavioural-hyperparameter question. A complete `neuromodulatory_algorithms_lit_review.md` (the broader thematic master review) is still outstanding for the 18-paper corpus.
- **`senior-developer`** — confirm Phase 0 T/P split is a prerequisite (§5.5 / v2 §3); confirm FiLM $\gamma \geq 0$ constraint (§6.4 / v1 §8 Q7).
- **`experiment-designer`** — translate §6.1 measure list into a concrete probe program for v3 §5 predictions.
