---
title: "A Hopfield network model of neuromodulatory arousal state"
authors: ["Mohammed Abdal Monium Osman", "Kai Fox", "Joshua Isaac Stern"]
year: 2024
venue: "bioRxiv 2024.09.15.613134 (preprint, Sep 2024)"
slug: osman_2024_hopfield_arousal
source_pdf: "sources/Osman et al. 2024 - A Hopfield network model of neuromodulatory arousal state.pdf"
topic: neuromodulatory_algorithms
---

# Osman, Fox & Stern 2024 — A Hopfield network model of neuromodulatory arousal state

## Plain-English entry point

This short preprint asks: arousal — the brain's overall level of engagement, ranging from sleep to alert — is regulated by chemical signals (mainly acetylcholine and noradrenaline) that suppress how strongly neurons talk to each other internally while letting outside-world signals dominate. Can a *minimal* mathematical model capture this with a single tunable knob? The authors take a classic associative-memory network — the **Hopfield network**, a network of recurrently connected neurons whose stored memories are valleys ("attractors") in an energy landscape — and add one parameter $\alpha$ that **divisively suppresses the recurrent connections**. When $\alpha$ is small, recurrence dominates → the network behaves like associative memory, settling into stored patterns regardless of input (the sleep-like state). When $\alpha$ is large, recurrence is weak and the network's output is dominated by the bottom-up input — like a feedforward classifier (the alert state). Between these regimes there is a precise mathematical **bifurcation** (a phase transition) at $\alpha^* = \lambda_{\max}(M)$, the largest eigenvalue of the recurrent-weight matrix. They show three theoretical bridges: (1) the network's Lyapunov function is identical to the variational free energy of a **Boltzmann machine**, with $\alpha$ inversely scaling the prior; (2) without input, the model maps onto the **Ising model** with $\alpha$ as temperature; (3) the time course of $\alpha$ becomes an **annealing schedule** — when $\alpha$ briefly rises, the system can escape a local minimum and find a better attractor, mirroring how pupil dilation accompanies perceptual switches. The paper matters as a *theoretical compass*: it gives a one-parameter, analytically tractable model that unifies attractor neuroscience, Bayesian inference, statistical mechanics, and the neuromodulatory-arousal literature.

## Section-ordered backbone

**Introduction.** Brains interleave input-driven and internally-driven activity; arousal regulates the balance. Acetylcholine is the canonical neuromodulator that suppresses recurrent connectivity while preserving feedforward (Hasselmo & Bower 1992). The paper instantiates this in a continuous Hopfield network with a recurrence-suppressing gain $\alpha$.

**Related work.** Hasselmo's olfactory associative learning model (recurrent suppression by ACh); Papadopoulos 2024 (gain on background inputs explains inverted-U arousal-performance curve); Moran/Friston (ACh sets likelihood-vs-prior balance in active inference); Sompolinsky 1988 (random-connectivity RNNs undergo gain-driven chaos transition); deterministic annealing.

**Network dynamics (Section 3).** Continuous Hopfield with arousal-suppressed recurrence:

$$\frac{dy}{dt} = -y + f\!\left(\tfrac{1}{\alpha} M y + W x\right),$$

with $y \in \mathbb{R}^N$ (neural activations), $x \in \mathbb{R}^D$ (stimulus), $M \in \mathbb{R}^{N \times N}$ (symmetric zero-diagonal recurrent weights), $W \in \mathbb{R}^{N \times D}$ (feedforward weights), $f = \tanh$ (element-wise). The two limits:
- $\alpha \to 0$: $\dot y = -y + \text{sign}(My)$ → memory-dominated; attractors = stored patterns.
- $\alpha \to \infty$: $\dot y = -y + f(Wx)$ → input-dominated; effectively a feedforward network.

**Bifurcation (Fig. 2).** As $\alpha$ rises, the system goes through a pitchfork bifurcation from multistable (multiple memory attractors) to unistable (single input-driven attractor). A two-unit example with mutual inhibition: nullclines are sigmoids whose steepness depends on $\alpha$; at low $\alpha$, three intersections (two stable + saddle); at high $\alpha$, one intersection (global attractor). Closed form (no input):

$$\alpha^* = \lambda_{\max}(M).$$

Proof in Appendix A: linearize at the origin, find Jacobian $J|_0 = \tfrac{1}{\alpha} M - I$; require $\lambda_{\max}(J|_0) = 0$ ⇒ $\alpha^* = \lambda_{\max}(M)$.

**Energy function (Section 4).** The Lyapunov function

$$F(y \mid x; \alpha) = -\frac{1}{2\alpha} y^\top M y - \frac{1}{2} y^\top W x - \sum_{i=1}^N H_2^{(e)}\!\left(\frac{y_i + 1}{2}\right),$$

with $H_2^{(e)}(p) = -p \log p - (1-p)\log(1-p)$, equals the mean-field variational free energy of a Boltzmann machine with prior $\propto \exp(-\tfrac{1}{2\alpha} z^\top M z)$ and likelihood $\propto \exp(-\tfrac{1}{2} z^\top W x)$:

$$\mathcal{F}[q \mid x; \alpha] = \tfrac{1}{\alpha}\langle E^{(z)} \rangle_q + \langle E^{(x \mid z)} \rangle_q - \mathcal{S}[q] = F(y \mid x; \alpha).$$

As $\alpha \to \infty$, the prior contribution vanishes → maximum likelihood + entropy regularization → restricted Boltzmann machine. As $\alpha \to 0$, the prior dominates → samples from prior, no inference.

**Statistical-physics analogy.** Without input, $F[q \mid x; \alpha] = \tfrac{1}{\alpha}\langle E^{(z)}\rangle_q - \mathcal{S}[q]$ has $\alpha$ acting as temperature. With input, the model is isomorphic to the **Ising model** with $\alpha^{-1}$ as interaction strength: subcritical $\alpha^{-1}$ → magnetization tracks external field; supercritical → spontaneous symmetry-breaking and hysteresis.

**Annealing analogy (Section 5, Fig. 3).** Animals face ever-changing optimization problems. Treating $\alpha$ as a time-varying temperature schedule, transient $\alpha$ spikes act as an **annealing event** — flattening the energy landscape, allowing escape from local minima, then re-cooling into the new best attractor. This is offered as a mechanistic explanation for (a) pupil dilation at perceptual switches (Necker-cube flips), (b) pupil constriction during hippocampal/cortical replay (low $\alpha$ → strong recurrence → replay), (c) theta-rhythm cycles between encoding and retrieval (oscillating ACh).

**Conclusion.** Minimal one-parameter model recovers core arousal phenomenology, with rigorous links to attractor memory, Bayesian inference, and statistical mechanics. Suggests $\alpha$-style dynamic gain knobs could enhance AI flexibility.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Imagine a network that stores memories as low points (valleys) on a "potential" surface, like marbles rolling into wells. That's a Hopfield network. Now add one knob $\alpha$ that *flattens* the inside-network connections relative to the outside-world inputs. Low $\alpha$ = deep valleys, the marble settles into a memory; high $\alpha$ = shallow valleys, the marble follows whatever direction the input pushes. The network smoothly transitions between "sleeping" (memory replay) and "awake" (input following) by turning this one knob.

**The experimental setup.** No experiments — entirely theoretical. Start with the standard continuous Hopfield equations $\dot y = -y + f(My + Wx)$. Replace $M$ with $M/\alpha$. Analytically derive: (1) the bifurcation point where the multistable-to-unistable transition happens; (2) the energy/Lyapunov function and its identity with Boltzmann-machine free energy; (3) the equivalence to an Ising magnet with $\alpha$ as temperature. Then run small numerical simulations to confirm the phase portraits and visualize what time-varying $\alpha$ does.

**The result.** A single number $\alpha$ tunes the network between an associative-memory regime and an input-following regime; the transition is a clean pitchfork bifurcation at $\alpha^* = \lambda_{\max}(M)$. The network's energy descent is identical to variational free-energy minimization in a Boltzmann machine, so $\alpha$ also controls the weight of the Bayesian prior. The arousal-driven gain modulation is mathematically *exactly* the inverse temperature in an Ising model. Time-varying $\alpha$ implements simulated annealing — providing a principled explanation for why brief pupil dilations help the brain escape stale percepts.

## Phase 2 — Graduate-level deep dive

### 2.1 The continuous Hopfield network with arousal gain (eq. 1)

The state $y \in \mathbb{R}^N$ evolves according to

$$
\frac{dy}{dt} = -y + f\!\left(\frac{1}{\alpha} M y + W x\right),
$$

with $f(\cdot) = \tanh(\cdot)$ element-wise, $M \in \mathbb{R}^{N \times N}$ symmetric and zero-diagonal, $W \in \mathbb{R}^{N \times D}$, $\alpha \in \mathbb{R}_{+}$. Symmetric $M$ guarantees gradient-like dynamics; zero diagonal removes self-coupling.

### 2.2 Asymptotic limits

$\alpha \to 0$: the argument of $f$ saturates, so $f \to \text{sign}$:

$$
\frac{dy}{dt} = -y + \text{sign}(My).
$$

The fixed points are $y^* = \text{sign}(My^*)$ — the stored memory patterns. Input has zero effect.

$\alpha \to \infty$: the recurrent term $\tfrac{1}{\alpha} M y \to 0$:

$$
\frac{dy}{dt} = -y + f(Wx).
$$

The unique fixed point is $y^* = f(Wx)$ — a feedforward sigmoid of the input.

### 2.3 Bifurcation analysis (Appendix A)

Linearize around the origin $y = 0$ in the no-input case ($x = 0$). Jacobian entries:

$$
[J(\alpha)]_{ij} = \frac{\partial \dot y_i}{\partial y_j} = \begin{cases} -1, & i = j, \\ \tfrac{1}{\alpha} M_{ij}\, (1 - f([My]_i)^2), & i \neq j. \end{cases}
$$

At $y = 0$, $f(0) = 0$ so $1 - f([My]_i)^2 = 1$, giving

$$
J|_0(\alpha) = \frac{1}{\alpha} M - I.
$$

Critical point: $\lambda_{\max}(J|_0(\alpha^*)) = 0$ ⇒

$$
\boxed{\alpha^* = \lambda_{\max}(M).}
$$

Since $M$ is symmetric ($\lambda_i \in \mathbb{R}$) and trace-zero ($\sum_i \lambda_i = 0$), either all $\lambda_i = 0$ or at least one is positive, so $\alpha^* > 0$ is well-defined.

### 2.4 Energy function (Lyapunov, eq. 5)

The Hopfield Lyapunov function generalized to include arousal:

$$
F(y \mid x; \alpha) = -\frac{1}{2\alpha}\, y^\top M y - \frac{1}{2}\, y^\top W x - \sum_{i=1}^{N} H_2^{(e)}\!\left(\frac{y_i + 1}{2}\right),
$$

where $H_2^{(e)}(p) = -p \log p - (1-p)\log(1-p)$ is the binary entropy (in nats). One can verify $\dot F \leq 0$ along trajectories of the dynamics, so $F$ decreases monotonically to local minima (attractors).

### 2.5 Boltzmann machine equivalence (Appendix B)

A Boltzmann machine over binary spins $z \in \{-1, +1\}^N$ with prior $p(z) \propto \exp(\tfrac{1}{2\alpha} z^\top M z)$ and likelihood $p(x \mid z) \propto \exp(\tfrac{1}{2} z^\top W x)$ has posterior

$$
p(z \mid x; \alpha) = \frac{1}{Z(x; \alpha)}\, \exp\!\left( \tfrac{1}{2\alpha} z^\top M z + \tfrac{1}{2} z^\top W x \right).
$$

The mean-field variational approximation $q(z) = \prod_i q_i(z_i)$ with mean $y_i = \langle z_i \rangle_{q_i}$ gives variational free energy

$$
\mathcal{F}[q \mid x; \alpha] = \langle E(z \mid x; \alpha) \rangle_q - \mathcal{S}[q],
$$

with $\mathcal{S}[q] = \sum_i H_2^{(e)}\!\big(\tfrac{y_i + 1}{2}\big)$. Substituting $\langle z_i z_j\rangle_q = y_i y_j$ (factorized) and $\langle z_i \rangle_q = y_i$ recovers exactly the Lyapunov function $F(y \mid x; \alpha)$. Therefore the continuous Hopfield dynamics implement gradient descent on the variational free energy → **approximate Bayesian inference with $\alpha^{-1}$ as the prior weight**.

### 2.6 Ising-model isomorphism

Without input, the system is the Ising model with Hamiltonian $\mathcal{H} = -\tfrac{1}{2} \sum_{ij} J_{ij} z_i z_j$, $J_{ij} = M_{ij}/\alpha$, at unit temperature. Equivalently, fixing $M$ and varying $\alpha$ is fixing the interaction matrix and varying inverse temperature $\beta = 1/\alpha$. The Curie temperature $T_c = \lambda_{\max}(M)$ is the critical point above which the mean-field magnetization $\langle z \rangle = 0$ becomes the unique solution — i.e., $\alpha^* = T_c$ — matching the bifurcation analysis. With input, $Wx$ acts as an external field; subcritical ($\alpha$ large) → magnetization tracks field; supercritical ($\alpha$ small) → hysteresis.

### 2.7 Annealing schedule and pupillary correlates

A dynamic $\alpha(t)$ acts as a deterministic annealing schedule:

$$
\alpha(t) = \alpha_{\text{tonic}} + \gamma\, \mathbf{1}_{\text{ambiguous}}(t),
$$

where the indicator function fires during periods of high stimulus ambiguity. A transient $\alpha$ spike flattens the energy landscape, allowing the network state to escape an old attractor; as $\alpha$ relaxes back, the system "cools" into the new optimal attractor. Pupil dynamics:
- Pupil dilation during Necker-cube flips → $\alpha$ spike → barrier-crossing.
- Pupil constriction during hippocampal replay → low $\alpha$ → memory-pattern sampling.
- Theta oscillation in hippocampal ACh → fast $\alpha$ cycle → alternating encoding/retrieval phases (Hasselmo, Bodelón & Wyble 2002).

### 2.8 Connection to the Wainstein RNN

Wainstein 2025 (this batch) drives a *non-symmetric* Dale's-law RNN with an uncertainty-driven gain on the activation function and observes essentially the same phenomenology: gain transient at uncertainty → escape from current perceptual attractor. Osman 2024 strips this back to the cleanest symmetric Hopfield setting where the bifurcation is closed-form. The two models offer complementary readings of the same hypothesis.

### 2.9 Summary table of correspondences

| Concept | Osman 2024 parameter | Equivalent in… |
|---|---|---|
| Arousal level | $\alpha$ | LC firing rate (high $\alpha$) / sleep (low $\alpha$) |
| Bifurcation point | $\alpha^* = \lambda_{\max}(M)$ | Ising critical temperature $T_c$ |
| Prior weight | $1/\alpha$ | Bayesian-inference precision on prior |
| Likelihood weight | 1 | Bayesian-inference precision on sensory evidence |
| Energy function | $F(y \mid x; \alpha)$ | Variational free energy / Lyapunov / Ising Hamiltonian |
| Dynamic gain trajectory | $\alpha(t)$ | Annealing schedule / pupil-tracked LC bursts |

## Connections to other corpus papers

- **`wainstein_2025_gain_perceptual_switches.md`** (this batch) — explicitly cited (ref. 26); Osman's model formalizes the mechanism Wainstein's RNN and fMRI confirm empirically. Osman is the cleanest theoretical companion piece to Wainstein.
- **`shine_2021_cellular_to_dynamics.md`** (this batch) — Osman's "recurrent gain $\alpha$" is the mean-field abstraction of the population-gain mechanism Shine catalogued at the cellular level.
- **`tsuda_2021_activity_hypertubes.md`** (this batch) — Tsuda's uniform weight scaling $W \mapsto f_{nm} W$ in a trained RNN is the trained-network analog of Osman's $M \mapsto M/\alpha$ in a Hopfield. Tsuda gains memory-overlay; Osman gains a bifurcation.
- **`costacurta_2024_structured_flexibility.md`** (this batch) — Costacurta's NM-RNN structurally factors $W_x = \sum_k s_k(z) \ell_k r_k^\top$; Osman's uniform $\alpha$ is a one-parameter limit.
- **`hasselmo_bower_1992_cholinergic_suppression.md`** (other batch, ref. 9) — the empirical olfactory-piriform circuit that Osman's Hopfield abstracts.
- **`hasselmo_1995_neuromodulation_review.md`** (other batch, ref. 7) — broader cholinergic theory.
- **`papadopoulos_2024_metastable_arousal.md`** (other batch, ref. 20) — gain-driven multistable-to-unistable transition in auditory cortex; same phenomenon, different model.
- **`moran_2013_free_energy_acetylcholine.md`** (other batch, ref. 17) — active-inference account of ACh tuning likelihood-vs-prior; Osman gives the exact mathematical identification.
- **`sompolinsky_1988_chaos_in_random.md`** (other batch, ref. 25) — gain-driven transition to chaos in random non-symmetric RNNs; complement to Osman's symmetric bifurcation.
- **`hopfield_1984_graded_response.md`** (ref. 12) — the original continuous Hopfield network that Osman extends.
- **`hinton_sejnowski_ackley_1984_boltzmann.md`** (ref. 11) — the Boltzmann machine; Osman makes the Hopfield-Boltzmann equivalence parametric in $\alpha$.
- **`marder_2012_neuromodulation_circuits.md`** (other batch, ref. 15) — biological backdrop on neuromodulator flexibility.
- **`sara_2009_lc_modulation.md`** (other batch, ref. 23) — LC-NA modulation of cognition; foundational citation.
- **`shine_2019_integration_segregation.md`** (other batch, ref. 24) — neuromodulatory influence on whole-brain integration; conceptual companion.
