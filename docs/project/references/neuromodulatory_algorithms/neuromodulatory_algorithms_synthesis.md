---
title: "Neuromodulatory Algorithms — Cross-Paper Synthesis"
topic: neuromodulatory_algorithms
papers_synthesised: 41
last_curated: 2026-05-15
companion_index: neuromodulatory_algorithms_lit_review.md
---

# Neuromodulatory Algorithms — Cross-Paper Synthesis

## 1. Plain-English entry point — the field for newcomers

Suppose you wanted to build an artificial animal that survives in a world it
has never seen, that learns from a stream of unrepeatable events, that can
shift strategy when the situation changes, and that does *not* need a human
operator twisting knobs every time the weather, predator, or food source
changes. The classical recipe — train a neural network end-to-end on a big
dataset and freeze its weights — fails this test. Real animals win on this
test because their brains carry, in addition to point-to-point synaptic wiring,
a separate **broadcast layer** of slow chemical signals called
**neuromodulators**.

A *neuromodulator* is a chemical that one neuron releases into a wide volume
of tissue, where it tunes the behaviour of every neuron in its reach for a
period ranging from hundreds of milliseconds (a phasic burst) to hours (a
tonic level). Four "major" neuromodulators dominate the mammalian literature:

- **Dopamine (DA)** — released from the *substantia nigra* and *ventral
  tegmental area*; carries the brain's reward-surprise signal.
- **Serotonin (5-HT)** — released from the *raphe nuclei*; sets the brain's
  patience, harm-aversion, and how far into the future it cares about reward.
- **Noradrenaline (NA / norepinephrine, NE)** — released from the *locus
  coeruleus*; signals novelty / arousal / surprise; raises *gain* (the slope
  of a neuron's input-output curve) to increase responsiveness.
- **Acetylcholine (ACh)** — released from the *basal forebrain*; gates
  attention and plasticity ("how strongly should I update my memory right
  now?").

To these four, the corpus adds a *hormonal* layer — oxytocin, cortisol,
adrenaline, endorphin, and the catch-all "endocrine hormones" of the Cañamero
school — which operate on slower timescales (seconds-to-hours) and tie
neuromodulation explicitly to **homeostasis** (the maintenance of a body's
internal variables — energy, hydration, temperature — within safe bounds).

The field has organised around three big questions:

1. **What does each modulator compute?** Is dopamine "wanting" or "reward
   prediction error"? Is serotonin "patience" or "average reward" or
   "anxiety"? The corpus's answer is *both*: each modulator's role depends on
   what receptor it reaches and at what concentration.
2. **How does a modulator's diffuse signal produce specific behaviour?** The
   computational answer is **gain modulation**: a neuromodulator changes the
   slope of a neuron's response curve, and small per-neuron gain changes
   compound at the network level into large, sometimes opposite, behavioural
   shifts. This is *Film-style* multiplicative modulation in deep-learning
   language ("feature-wise linear modulation"), and *Servan-Schreiber-style*
   sigmoid-slope tuning in cognitive neuroscience.
3. **How do you put this into an artificial agent?** The corpus answers in
   seven distinct ways — implementing modulation as a learnable activation
   function, as a structural mask, as a low-rank scaling, as an arousal
   parameter, as a per-task confidence vector, as a temperature in a
   statistical-mechanics analogy, or as a hormonal modulation of a
   hand-designed control architecture. These are the seven clusters this
   synthesis is built around.

The open questions are equally three:

- **Specificity.** Are the four modulators *really* one-knob-each (Doya 2002),
  or do they all do the same thing (signal-to-noise sharpening, Krichmar 2008)
  and differ only in their triggers? Both views have empirical support.
- **Stability vs. flexibility.** If neuromodulators are how brains avoid
  catastrophic forgetting, why do artificial systems with neuromodulator-like
  mechanisms (NGM-SGD, NPN, ANML) still show stability gaps and trade off
  retention vs. plasticity? Are there *structural* prerequisites (manifold
  attractors, multi-scale plasticity) that the deep-learning recipes leave
  out?
- **Pain and interoception.** Almost the entire corpus treats neuromodulation
  as gating *external* attention, reward, and action. Only a handful of
  papers — L'Haridon-Cañamero 2023, Khan-Cañamero 2022, the broader
  Cañamero homeostatic tradition, Chiba-Krichmar 2020 — frame it as gating
  *internal-state* perception (felt pain, felt hunger, felt stress). This is
  the gap the present project tries to fill.

## 2. Thematic clusters

### Cluster A — Affective robotics & motivation/emotion lineage (Cañamero school)

**Members.** [canamero_1997](reviews/canamero_1997_motivations_emotions.md);
[canamero_2005](reviews/canamero_2005_emotion_understanding.md);
[blanchard_canamero_2006](reviews/blanchard_canamero_2006_affect_modulated.md);
[cos_2010](reviews/cos_2010_affordances_consummatory.md);
[lones_canamero_2013](reviews/lones_canamero_2013_epigenetic_hormones.md);
[lewis_canamero_2016](reviews/lewis_canamero_2016_hedonic_pleasure.md);
[lones_2018](reviews/lones_2018_hormone_epigenetic.md);
[khan_canamero_2022](reviews/khan_canamero_2022_social_buffering.md);
[scarinzi_canamero_2022](reviews/scarinzi_canamero_2022_affective_interactions.md);
[lharidon_canamero_2023](reviews/lharidon_canamero_2023_stress_pain.md).

**One-line per paper.** Each paper extends one piece of Cañamero 1997's
hormone-modulated, homeostatic, multi-motivation Abbott architecture:

- *canamero_1997* — the founding architecture: physiological variables, drives,
  ART-1 vigilance threshold $\rho$ set by emotion state, hormones modulate
  perceived state.
- *canamero_2005* — reflective methodological review splitting designed-emotion
  from emergent-emotion approaches; positions hormone modulation as one of two
  principled mechanism families.
- *blanchard_canamero_2006* — *affect* and *well-being* scalars on a Koala
  robot autonomously generate four behavioural modes via sign-flip of the
  motivation-to-continue.
- *cos_2010* — Gibsonian affordances learned through hormone-reinforced
  Hebbian updates; physiology-fluctuation as the supervisory signal.
- *lones_canamero_2013* — adds a 3-minute "epigenetic" developmental window
  where gland activity $\theta_h$ and receptor sensitivity $\text{sens}_h$
  drift to environment.
- *lewis_canamero_2016* — *pleasure ≠ reward*: a hedonic pleasure decoupled
  from need has its own adaptive value; the incentive-salience knob $\alpha$
  is hormone-modulated.
- *lones_2018* — IEEE-TCDS extension of Lones 2013 with the clean exponential
  receptor rule; six environments produce six distinct emergent phenotypes.
- *khan_canamero_2022* — oxytocin buffers cortisol in a six-agent society;
  social bonding is front-loaded.
- *scarinzi_canamero_2022* — opinion paper on what "affect" and "e-motion"
  mean for an HRI agent; no equations.
- *lharidon_canamero_2023* — cortisol-modulated pain perception
  (**pain = cortisol × damage**) in a survival robot; closest analogue to the
  present project.

**Shared assumptions.** Every paper assumes (i) the agent has a *body* with
physiological variables that drift away from healthy values, (ii) **drive** =
deviation from a homeostatic set point, (iii) **action selection** is
winner-takes-all over motivations, and (iv) hormones modulate perception,
incentive salience, and behaviour intensity — but cannot *trigger* a
motivation.

**Disagreements / open questions.** The school's biggest internal disagreement
is whether *pleasure* should be tied to need-satisfaction (Cañamero 1997, Cos
2010) or decoupled as a hedonic-quality signal (Lewis & Cañamero 2016). The
2016 paper is explicit that prior work *collapsed* pleasure into reward; the
unanswered question is whether this distinction has any value in a
reinforcement-learning agent that already has a separate reward signal. A
second disagreement concerns developmental plasticity: Lones 2013/2018 closes
the receptor-tuning window after a fixed early period, but the broader
Cañamero corpus mostly works with permanently plastic hormone systems.

**Key equations.** The cluster's recurring formal core:

The **drive** for motivation $m$ controlling variable $i(m)$ (Cañamero 1997):

$$
d_m(t) = f_m\!\big( x_{i(m)}(t) - \bar x_{i(m)} \big),
$$

with $\bar x$ the homeostatic set point.

**Hormone-modulated perception** (Cañamero 1997 ART-1 form):

$$
\rho(t) = \rho_0 + \delta\rho\!\big(\mathbf{e}(t)\big), \qquad
\tilde x_i(t) = x_i(t) - \kappa_{i,k}\, h_k(t).
$$

**Incentive salience as a hormonal knob** (Lewis & Cañamero 2016):

$$
\text{motivation}_i = d_i + (d_i \cdot \alpha \cdot \text{cue}_i),
\qquad \alpha = f(h_{\text{pleasure}}).
$$

**Epigenetic receptor update** (Lones et al. 2018):

$$
\text{Sens}_i(t+1) = \text{Sens}_i(t)^{\,E_h^i / \sigma}.
$$

**Cortisol-modulated pain** (L'Haridon & Cañamero 2023):

$$
\text{pain} = \text{cortisol} \times \text{damage}.
$$

---

### Cluster B — Neurorobotics & neuromodulator-controller lineage (Krichmar school)

**Members.** [cox_krichmar_2009](reviews/cox_krichmar_2009_neuromodulation_robot_controller.md);
[krichmar_2013](reviews/krichmar_2013_neurorobotic_anxiety_curiosity.md);
[avery_krichmar_2017](reviews/avery_krichmar_2017_models_neuromodulation.md);
[hwu_krichmar_2020](reviews/hwu_krichmar_2020_schemas_memory.md);
[chiba_krichmar_2020](reviews/chiba_krichmar_2020_self_monitoring.md);
[xing_2020](reviews/xing_2020_neuromodulated_patience.md);
[zou_2020](reviews/zou_2020_neuromodulated_attention.md);
[xing_2022](reviews/xing_2022_neuromodulation_rl_environment_changes.md);
[krichmar_hwu_2022](reviews/krichmar_hwu_2022_design_principles_neurorobotics.md).

**One-line per paper.**

- *cox_krichmar_2009* — foundational engineering paper: CARL-1 robot with
  VTA / Raphe / BF nuclei; phasic DA/5-HT drives Find/Flee with BCM gating;
  lesion methodology.
- *krichmar_2013* — adds tonic levels, 5-HT→DA inhibition, OFC/mPFC top-down
  inhibition; CarlRoomba reproduces open-field pharmacology phenotypes.
- *avery_krichmar_2017* — encyclopedia chapter; Doya 2002 mapping + Krichmar
  "decisiveness" framework; Hawk-Dove two-critic model.
- *hwu_krichmar_2020* — Tse-et-al-style schema network; novelty × familiarity
  product drives replay.
- *chiba_krichmar_2020* — three-level (sensory / homeostatic / cognitive)
  framework; interoception as the bridge to feelings/self-awareness.
- *xing_2020* — 5-HT scalar in Miyazaki-2018 Bayesian wait/quit rule on
  outdoor robot.
- *zou_2020* — c-EB + ACh+NE head for goal-driven attention.
- *xing_2022* — extension of Zou 2020 into deep RL (PPO/TD3); ACh-NE for
  task-switching.
- *krichmar_hwu_2022* — citation-hub position paper: 14 design principles.

**Shared assumptions.** (i) Each modulator has a *biologically-grounded
trigger* — DA = reward, 5-HT = harm/cost, ACh = effort/expected uncertainty,
NA = novelty/unexpected uncertainty. (ii) All modulators share a downstream
effect — *signal-to-noise sharpening* in target populations, i.e. "be
decisive". (iii) Lesion analysis is the canonical falsification tool. (iv)
Tonic levels set context; phasic bursts trigger specific actions.

**Disagreements / open questions.** Krichmar's *decisiveness framework*
(Krichmar 2008 → Cox & Krichmar 2009) says all modulators do the same thing
downstream and differ only in triggers — and explicitly contrasts itself with
Doya 2002 (Cluster C), which claims each modulator carries a *different*
algorithmic parameter. The corpus does not resolve this — Avery & Krichmar 2017
presents both as legitimate. A second open question is whether the iRobot /
HSR-scale robotic demonstrations (small networks, well-defined tasks) tell us
anything about the scaling of these mechanisms to deep-network agents — Xing
2022 ports Zou 2020 to deep RL successfully, but the corpus has no comparable
demonstration for, say, Hwu's schema network at scale.

**Key equations.**

**Hawk-Dove two-critic** (Asher, Zaldivar, Krichmar 2010, reviewed in Avery &
Krichmar 2017):

$$
\delta^{\text{reward}}_t = r_t + \gamma V^{\text{rew}}(s_{t+1}) - V^{\text{rew}}(s_t),
\qquad
\delta^{\text{cost}}_t = c_t + \gamma V^{\text{cost}}(s_{t+1}) - V^{\text{cost}}(s_t).
$$

**Miyazaki-2018 wait/quit Bayesian rule** (Xing 2020):

$$
p(\text{wait} \mid t) = \frac{1}{1 + \exp\!\big(\beta \cdot \text{5HT} \cdot L(t)\big)}.
$$

**ACh+NE expected/unexpected uncertainty signals** (Yu & Dayan 2005 → Zou 2020,
Xing 2022): ACh tracks per-task confidence; NE rises when softmax over ACh
fails to commit to a stored task.

---

### Cluster C — RL, meta-RL, and deep-learning neuromodulation

**Members.** [doya_2002](reviews/doya_2002_metalearning_neuromodulation.md);
[vecoven_2020](reviews/vecoven_2020_neuromod_dnn.md);
[beniwhiwhu_2022](reviews/beniwhiwhu_2022_context_meta_rl.md);
[mei_2022](reviews/mei_2022_multiscale_neuromod.md);
[lee_2024](reviews/lee_2024_lifelong_rl.md);
[wang_2024](reviews/wang_2024_neuromod_meta.md);
[wang_2025_nest](reviews/wang_2025_nest_hypergraph.md).

**One-line per paper.**

- *doya_2002* — foundational: DA = TD-error $\delta$; 5-HT = discount factor
  $\gamma$; NA = inverse temperature $\beta$; ACh = learning rate $\alpha$.
- *vecoven_2020* — **NMN**: slope-and-bias modulation of saturated ReLU
  activations by a shared $z$ vector from a context RNN; 97 % of Bayes-optimal.
- *beniwhiwhu_2022* — **NPN**: multiplicative gating ($\tanh$ over standard ×
  modulator) inside CAVIA/PEARL; CT-graph depth-4 win.
- *mei_2022* — TINS review proposing four-scale (hyperparameter / cell-type /
  weight / compartmental) integration of neuromodulation into DNNs.
- *lee_2024* — **Doya-DaYu agent**: $\alpha = E/(E+A)$ from ACh,
  $\beta = 1/\langle E\rangle$ from NA; closes the loop back to neuroscience.
- *wang_2024* — **NeuronML**: learnable per-task structural mask under
  frugality + plasticity + sensitivity constraints.
- *wang_2025_nest* — applied trajectory predictor; two-MLP module computes
  small-world graph parameters $\alpha, \beta$.

**Shared assumptions.** (i) The point of neuromodulation is to make a *single*
network express *different* effective functions in different contexts, without
re-training weights. (ii) The modulator subnet should receive *context* (task
identity, history, uncertainty) — not raw observations. (iii) Gating /
multiplicative interaction is preferred over additive bias.

**Disagreements / open questions.** Four implementations of "neuromodulation"
co-exist: activation-function modulation (Vecoven), per-layer gating
(Ben-Iwhiwhu), hyperparameter modulation (Lee), and structural masking
(Wang 2024). The corpus has no head-to-head benchmark of these against each
other; Mei 2022's four-scale proposal explicitly recommends *combining* them.
Whether Doya's one-modulator-per-parameter mapping holds at deep-network
scale, or collapses into the Krichmar "all modulators are gain knobs" view,
is the central unresolved issue.

**Key equations.**

**The Doya four-modulator equation** (Doya 2002):

$$
\theta_{t+1} = \theta_t + \underbrace{\alpha(\text{ACh}_t)}_{\text{learning rate}}
\cdot \underbrace{\delta(\text{DA}_t)}_{\text{TD error}} \cdot \phi(s_t, a_t),
\quad
a_t \sim \exp\!\big(\underbrace{\beta(\text{NA}_t)}_{\text{inverse temp}} Q_\theta(s_t, \cdot)\big),
\quad
V_\theta(s_t) = \mathbb{E}[r_{t+1} + \underbrace{\gamma(\text{5-HT}_t)}_{\text{discount}} V_\theta(s_{t+1})].
$$

**Vecoven's NMN activation** (Vecoven et al. 2020):

$$
\sigma_{\text{NMN}}(x, z; w_s, w_b) = \sigma\!\big(z^\top (x\, w_s + w_b)\big).
$$

**Ben-Iwhiwhu's NPN gating** (Ben-Iwhiwhu et al. 2022):

$$
h = \text{ReLU}\!\big(\underbrace{W_s x}_{\text{standard}}
\;\otimes\; \underbrace{\tanh(W_m\, \text{ReLU}(W_g x))}_{\text{modulator } h_m}\big).
$$

**Lee's Doya-DaYu hyperparameter mapping** (Lee et al. 2024):

$$
\alpha(s, a) = \frac{E(s, a)}{E(s, a) + A(s, a)}, \qquad
\beta(s) = \frac{1}{\big\langle E(s, \hat a)\big\rangle_{\hat a}},
$$

with $E$ epistemic and $A$ aleatoric uncertainty from a Q-quantile ensemble.

---

### Cluster D — Gain modulation & RNN dynamics

**Members.** [ferguson_cardin_2020](reviews/ferguson_cardin_2020_gain_modulation.md);
[shine_2021](reviews/shine_2021_cellular_to_dynamics.md);
[tsuda_2021](reviews/tsuda_2021_activity_hypertubes.md);
[driscoll_2022](reviews/driscoll_2022_dynamical_motifs.md);
[costacurta_2024](reviews/costacurta_2024_structured_flexibility.md);
[wainstein_2025](reviews/wainstein_2025_gain_perceptual_switches.md).

**One-line per paper.**

- *ferguson_cardin_2020* — *NRN* biology primer: divisive vs. additive
  modulation; PV / SST / VIP interneurons mediate; ACh / NA / 5-HT / DA
  effects.
- *shine_2021* — multi-scale review: five microscopic mechanisms map to four
  population-gain parameters (height / slope / threshold / width).
- *tsuda_2021* — multiplying every weight in an RNN by $f_{nm}$ shifts
  activity into a non-overlapping hypertube; fly PER reproduction; circuit-
  sensitivity varies 3× across networks.
- *driscoll_2022* — 15-task RNN reuses dynamical motifs (ring attractor,
  paired attractors, unstable saddles); rule input is the switch.
- *costacurta_2024* — **NM-RNN**: low-rank recurrent matrix scaled by a
  modulator subnet; mathematically equivalent to LSTM forget gates.
- *wainstein_2025* — pupillometry + fMRI + RNN; LC-NA gain modulation gates
  perceptual switches via attractor flattening.

**Shared assumptions.** (i) *Gain* (the slope of the I/O curve) is the
right common currency. (ii) Modulation reshapes the network's *dynamical
landscape* — attractors, ghost attractors, bifurcation manifolds — not just
its outputs. (iii) Mean-field reduction (population mean + variance) is the
right level to interpret cellular gain effects.

**Disagreements / open questions.** Tsuda 2021's mechanism scales *every*
recurrent weight uniformly; Costacurta 2024 scales *low-rank components*
selectively; Wainstein 2025 scales the *sigmoid slope* of every neuron
uniformly with one scalar. Are these three mechanisms three implementations of
the same principle, or biologically distinct? Shine 2021's five-mechanism
account suggests there are *more* gain parameters than the field has yet
exploited (specifically the width / temporal-aperture knob from AMPA/NMDA
balance, which no deep-learning paper has implemented).

**Key equations.**

**Single-neuron gain** (Ferguson & Cardin 2020; Shine 2021):

$$
g_{\text{neuron}}(I) = \frac{dQ}{dI}.
$$

**Population gain via mean-field reduction** (Shine 2021):

$$
F(V) = \int H(V - \theta)\, p(\theta)\, d\theta, \qquad
G(V) = \frac{dF}{dV}.
$$

**Tsuda uniform weight scaling** (Tsuda et al. 2021):

$$
\tilde W = f_{nm} \cdot W.
$$

**Costacurta low-rank modulated weight** (Costacurta et al. 2024):

$$
W_x(z(t)) = \sum_{k=1}^K s_k(z(t))\, \ell_k\, r_k^\top, \qquad
s(z(t)) = \sigma(A_z\, z(t) + b_z).
$$

**Wainstein uncertainty-driven gain** (Wainstein et al. 2025):

$$
\tau_g\, \frac{dg(t)}{dt} = g_{\text{tonic}} - g(t) + \gamma \cdot H\!\big(p(z(t))\big),
\qquad r_t = \frac{1}{1 + e^{-g(t)\, x_t}}.
$$

---

### Cluster E — Continual / lifelong learning via neuromodulation

**Members.** [kolouri_2019](reviews/kolouri_2019_attention_plasticity.md);
[kudithipudi_2022](reviews/kudithipudi_2022_lifelong_learning.md);
[durstewitz_2025](reviews/durstewitz_2025_neuroscience_continual_learning.md);
[rodriguezgarcia_2026](reviews/rodriguezgarcia_2026_ne_stability_gap.md).

**One-line per paper.**

- *kolouri_2019* — attention-based selective plasticity via contrastive
  Excitation Backprop + Oja's rule; biologically motivated stand-in for
  Elastic Weight Consolidation / Synaptic Intelligence.
- *kudithipudi_2022* — 31-author *Nature MI* Perspective; lays out biological
  ingredients (CLS, metaplasticity, neuromodulation, neurogenesis) for
  lifelong-learning machines.
- *durstewitz_2025* — 2025 companion piece from a dynamical-systems / PFC
  angle; **manifold attractors + multi-scale plasticity** as missing
  ingredients.
- *rodriguezgarcia_2026* — **NGM-SGD**: NA-inspired gain controlled by
  softmax entropy reduces the stability gap 2-15× across five CL benchmarks.

**Shared assumptions.** (i) Catastrophic forgetting is *not* purely a capacity
problem; it is a dynamics-of-optimisation problem and a representation problem.
(ii) Neuromodulator-style mechanisms gate which weights can change. (iii) The
right benchmark is *online-continual* with non-iid streams, not standard
multi-task fine-tuning.

**Disagreements / open questions.** Kolouri 2019 uses a *static* importance
tag $\gamma_k$ and a quadratic anchor regularisation; Rodriguez-Garcia 2026
uses a *dynamic* gain $g(t)$ driven by softmax entropy. These are not direct
competitors (one regularises weights, the other reshapes the loss landscape)
but the field has not yet integrated them. Durstewitz 2025 argues — and most
of the deep-learning cluster does not yet take seriously — that continual
learning needs the right *attractor structure* (manifold attractors, ghost
attractors) in addition to good gating, and that the Durstewitz lab's
dynamical-systems reconstruction (DSR) approach is the right path.

**Key equations.**

**Selective-plasticity loss** (Kolouri 2019, Elastic Weight Consolidation
family):

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \lambda \sum_k \gamma_k\,
(\theta_k - \theta_{A,k}^*)^2.
$$

**Attention-driven importance via c-EB + Oja** (Kolouri 2019):

$$
\gamma_{ji}^l \leftarrow \gamma_{ji}^l + \varepsilon\!\left(
P_c\!\big(f_j^{(l-1)}\big)\, P_c\!\big(f_i^{(l)}\big)
- \big[P_c(f_i^{(l)})\big]^2 \gamma_{ji}^l \right).
$$

**Rodriguez-Garcia entropy-driven gain** (Rodriguez-Garcia et al. 2026):

$$
g(t+1) = \gamma\, g(t) + (1 - \gamma)\, g_0 + \eta\, H(y),
\qquad \hat W = g\, W.
$$

Their Appendix B proof shows the effective Hessian satisfies
$\lambda_i^{\text{eff}} = \lambda_i / g^2$ — *gain boosts flatten the local
loss surface*.

---

### Cluster F — Hopfield / energy-based neuromodulation

**Members.** [osman_2024](reviews/osman_2024_hopfield_arousal.md);
[alonso_krichmar_2023](reviews/alonso_krichmar_2023_sparse_quantized_hopfield.md);
[tambas_2025](reviews/tambas_2025_krotov_hopfield_rbm.md).

**One-line per paper.**

- *osman_2024* — continuous Hopfield with divisive recurrence $M/\alpha$;
  pitchfork bifurcation at $\alpha^* = \lambda_{\max}(M)$; energy = Boltzmann
  free energy with $\alpha$ as inverse prior; Ising-temperature analogy.
- *alonso_krichmar_2023* — **SQHN**: tree-structured sparse-quantized Hopfield
  with a memory/root node doing one-shot online-continual learning; matches
  MHN+BP without backprop.
- *tambas_2025* — Krotov-Hopfield winner-take-all-with-runner-up-penalty rule
  as a neuromodulator layer for an RBM; reduces hidden-unit redundancy.

**Shared assumptions.** (i) Memories are attractors of an energy landscape;
modulation reshapes the landscape. (ii) A single scalar parameter (arousal,
temperature, sparsity) can interpolate between regimes (memory vs. inference,
storage vs. retrieval). (iii) Local Hebbian-style learning rules + a global
modulator can replace backprop for associative-memory tasks.

**Disagreements / open questions.** Osman 2024 is theoretical (no
demonstration of real task learning); Alonso & Krichmar 2023 is empirical
(strong demonstrations on CIFAR and EMNIST but no analytical bifurcation
story); Tambaş 2025 sits in between. None of these has been integrated with
the RNN-dynamics cluster (D) or the RL cluster (C), even though the
mathematical machinery (energy descent ↔ free-energy minimisation ↔ Boltzmann
machines) is shared.

**Key equations.**

**Osman arousal-modulated Hopfield**:

$$
\frac{dy}{dt} = -y + f\!\left(\frac{1}{\alpha}\, M y + W x\right), \qquad
\alpha^* = \lambda_{\max}(M).
$$

**SQHN local update** (Alonso & Krichmar 2023):

$$
\Delta M_{\text{pa}_l, l} = \frac{c^*_{\text{pa}_l}}{c^*_{\text{pa}_l} + 1}
\,(h_l^* - M_{\text{pa}_l, l}\, h^*_{\text{pa}_l})\, h_{\text{pa}_l}^{*\top},
$$

with **decaying neuron-growth threshold** $\epsilon = \alpha / (t + \alpha)$
from a Dirichlet prior.

**Krotov-Hopfield winner-take-all modulator** (Tambaş et al. 2025):

$$
g_\nu(I) = \begin{cases} 1 & \text{if } r_\nu = K \\ -\Delta & \text{if } r_\nu = K - \ell \\ 0 & \text{otherwise.} \end{cases}
$$

---

### Cluster G — Spiking neural networks + neuromodulation

**Members.** [espino_2024](reviews/espino_2024_snn_path_planning.md);
[alkilany_goodman_2025](reviews/alkilany_goodman_2025_snn_dynamic_sensory.md).

**One-line per paper.**

- *espino_2024* — Spiking Wavefront Planner (SNN cost map) + E-Prop online
  learning on Clearpath Jackal; flags neuromodulation for combining cost maps
  as future work.
- *alkilany_goodman_2025* — modulator network outputs LIF parameters
  (threshold, reset, $\tau_m$, $\tau_s$) on the fly; SNN discovers
  *listen-in-the-dips* under SAM noise.

**Shared assumptions.** (i) Spike-based computation is the right level to
deploy on neuromorphic chips. (ii) Local rules (E-Prop, surrogate gradients)
can do the learning that backprop usually does. (iii) Modulation of
*intrinsic biophysical parameters* (not just weights) is a viable lever.

**Disagreements / open questions.** Espino 2024 modulates *axonal delays*;
AlKilany & Goodman 2025 modulates *neuron-intrinsic time constants*. Neither
paper has demonstrated a *learned* modulator that exceeds a strong feedforward
SNN baseline by more than a few percentage points outside of the noise
regimes. Whether sub-second modulation will scale to large SNNs on Loihi or
SpiNNaker is the empirical open question.

**Key equations.**

**LIF dynamics under modulation** (AlKilany & Goodman 2025):

$$
\tau\, \dot v = -v + x, \qquad \tau_x\, \dot x = -x, \qquad
v_{\text{th}}(t), v_r(t), \tau_m(t), \tau_s(t) \;\;\text{all from modulator}.
$$

**Spiking Wavefront Planner with E-Prop delay update** (Espino et al. 2024):
spike propagation through grid neurons with axonal delays $D_{ij}$ encoding
edge cost; delays updated by eligibility-trace-gated E-Prop.

---

## 3. Historical timeline (1997 → 2026)

This section traces *influence flows*. Where a paper opens a thread that
later papers extend, the trace is shown explicitly. Citations like
"§Connections in *X*" point to the *X* per-paper review's Connections
subsection where the forward / backward edge is documented.

### 3.1 1997 → 2002 — Founding of the two parallel programmes

The corpus has *two* foundational papers from this window:

- **Cañamero 1997** ([review](reviews/canamero_1997_motivations_emotions.md))
  introduced the *Abbott / Gridland* hormone-modulated homeostatic
  architecture. The architecture's two key moves — (i) physiological deficit
  → drive → motivation winner-takes-all, (ii) emotion-released hormones
  modulate ART-1 vigilance, behaviour intensity, and *perceived* internal
  state — became the design template for every later Cañamero-school paper
  (Cluster A). The 1997 paper does *not* explicitly cite or anticipate the
  neuromodulator-as-RL-parameter view; it works at a coarser, hormone-as-
  control-signal level.

- **Doya 2002** ([review](reviews/doya_2002_metalearning_neuromodulation.md))
  proposed the one-modulator-per-RL-parameter mapping — DA = $\delta$,
  5-HT = $\gamma$, NA = $\beta$, ACh = $\alpha$ — by combining Schultz's DA
  recordings, 5-HT-depletion impulsivity, Aston-Jones's LC tonic/phasic
  inverted-U, and Hasselmo's hippocampal ACh model. The paper does not
  reference Cañamero 1997 — these two programmes start in *separate*
  literatures: Cañamero in autonomous-agents / affective computing
  (Minsky's *Society of Mind*, McFarland 1995 ethology); Doya in
  computational neuroscience (Schultz, Houk-Adams-Barto BG, Sutton & Barto
  RL).

Both will be cited as foundations by almost every later paper in the
corpus. The two programmes will not merge for another decade.

### 3.2 2005 → 2010 — Consolidation of the affective-robotics line

The Cañamero programme consolidates and extends:

- *Cañamero 2005* methodologically reviews the entire emergent-vs-designed
  emotion-modelling field, positioning hormone-modulated architectures as the
  designed-emotion mainstream.
- *Blanchard & Cañamero 2006* shows that **affect** as a scalar can autonomously
  generate four behavioural modes via sign-flip — this is the first
  Cañamero-school paper to make modulation *change the direction* of motivation,
  not just its intensity.
- *Cos, Cañamero & Hayes 2010* binds affordance learning to homeostatic
  feedback — a direct intellectual descendant of Cañamero 1997's Hebbian-
  hormone learning + Cañamero 2005's affordance discussion.
- *Cox & Krichmar 2009* — the **Krichmar programme begins**. CARL-1 implements
  Cañamero-style hormonal modulation but at the neural-circuit level (VTA,
  Raphe, BF nuclei) and explicitly cites Doya 2002 and Cañamero 1997 as joint
  foundations. The decisiveness framework starts here.

By 2010 the two programmes have *not yet integrated* — the Krichmar lab is
clearly aware of Cañamero 1997 (Cox & Krichmar 2009 cites it) but the
Cañamero school has not yet incorporated the cellular nuclei view.

### 3.3 2010 → 2017 — Maturation of the neurorobotic / Cañamero / RL threads

Three threads progress in parallel:

**Krichmar lab** — Krichmar 2013 adds tonic dynamics and 5-HT-DA opponency to
CARL; Avery & Krichmar 2017 produces the canonical encyclopedia chapter
synthesising Doya 2002 + Krichmar 2008 decisiveness; this chapter becomes the
single most-cited entry point for "what does each modulator do?"
([§Connections in avery_krichmar_2017](reviews/avery_krichmar_2017_models_neuromodulation.md)).

**Cañamero lab** — Lones & Cañamero 2013 adds developmental epigenetic
plasticity to the Abbott architecture, with a 3-minute critical window in
which gland activity and receptor sensitivity drift to environment; Lewis &
Cañamero 2016 makes the case that *pleasure ≠ reward* and adds a
hedonic-quality signal independent of need satisfaction.

**Cellular gain modulation** — Ferguson & Cardin 2020 (just outside this
window, but conceptually here) consolidates the cellular biology view in a
*Nature Reviews Neuroscience* primer; the Avery, Dutt & Krichmar 2014 ACh
model and Shine 2021's mean-field synthesis are the proximal antecedents.

### 3.4 2017 → 2022 — Two convergences and a Perspective wave

This window sees the **first integration of cellular gain modulation with
deep-learning RL**:

- *Ferguson & Cardin 2020* gives the canonical biology vocabulary
  ([review](reviews/ferguson_cardin_2020_gain_modulation.md)); every later
  deep-network paper (Rodriguez-Garcia 2026, Wainstein 2025, Costacurta 2024)
  cites it.
- *Shine 2021* provides the multi-scale link from cellular biophysics to
  whole-brain dynamics ([review](reviews/shine_2021_cellular_to_dynamics.md)),
  introducing the population-gain function as the right common currency.
- *Vecoven 2020* implements the cleanest deep-learning analogue of cellular
  gain modulation — slope-and-bias modulation of saturated ReLU via a shared
  $z$ vector; this becomes the architectural template that
  [Wang 2025 NEST](reviews/wang_2025_nest_hypergraph.md),
  [Mei 2022](reviews/mei_2022_multiscale_neuromod.md), and others all cite
  ([§Connections in vecoven_2020](reviews/vecoven_2020_neuromod_dnn.md)).
- *Tsuda et al. 2021* shows that even the *simplest* implementation — multiply
  every weight in an RNN by one scalar — produces non-trivial dynamical
  behaviour; this paper is the influence bridge from the cellular-biology
  (Ferguson & Cardin) cluster to the RNN-dynamics (Driscoll, Costacurta,
  Wainstein) cluster.
- *Ben-Iwhiwhu 2022* demonstrates that **multiplicative gating** inside
  CAVIA/PEARL beats parameter-matched standard MLPs on Meta-World and
  CT-graph, providing the first empirical evidence that the
  neuromodulator-as-gating-layer view scales to hard meta-RL.
- *Mei 2022* writes the canonical *Trends in Neurosciences* roadmap for
  putting neuromodulation into DNNs, with explicit four-scale framework
  ([review](reviews/mei_2022_multiscale_neuromod.md)).
- *Krichmar & Hwu 2022* writes the Krichmar-lab citation-hub position paper
  on 14 neurorobotics design principles.
- *Kudithipudi 2022* (31-author *Nature Machine Intelligence* Perspective)
  consolidates the **lifelong-learning** thread, naming neuromodulation as one
  of seven biological pillars.
- *Khan & Cañamero 2022* (in parallel) extends the Cañamero programme into the
  social-buffering domain — oxytocin and cortisol — and treats them as
  embodied biomarkers.

By 2022 the convergence is essentially complete: Cluster A (Cañamero), Cluster
B (Krichmar), and Cluster C (RL/DL) are no longer disjoint literatures.
Avery & Krichmar 2017, Krichmar & Hwu 2022, and Mei 2022 are the synthesising
documents that bridge them.

### 3.5 2023 → 2026 — The current frontier

The most recent papers fall into three threads:

**Continual-learning fixes via neuromodulation.**

- *Rodriguez-Garcia 2026* introduces NGM-SGD as the first algorithm to
  *specifically attack the stability gap* with a noradrenaline-inspired
  uncertainty-driven gain; this paper directly extends *Wainstein 2025* (it
  uses Wainstein's softmax-entropy uncertainty signal verbatim) and *Mei 2022*
  (which proposed gain modulation as a deep-learning lever) into the
  continual-learning setting ([§Connections in
  rodriguezgarcia_2026](reviews/rodriguezgarcia_2026_ne_stability_gap.md)).
- *Durstewitz 2025* is the 2025 *Nature Machine Intelligence* companion to
  Kudithipudi 2022, but from a dynamical-systems angle — arguing that
  manifold attractors and multi-scale plasticity are the missing ingredients
  that neuromodulation alone will not solve.

**Energy-based and Hopfield extensions.**

- *Alonso & Krichmar 2023* shows that an online-continual associative memory
  (SQHN) can be built without backprop.
- *Osman 2024* gives the analytical Hopfield+arousal story.
- *Tambaş 2025* injects a Krotov-Hopfield neuromodulator into an RBM.

**Applied / scalable demonstrations.**

- *Lee 2024* implements the Doya-DaYu agent on a non-stationary bandit and
  proposes a closed-loop mouse experiment to test the framework's
  predictions — the corpus's most explicit "AI ↔ neuroscience" loop.
- *Wang 2024 NeuronML* generalises neuromodulation to structural masking with
  formal regret bounds.
- *Wang 2025 NEST* shows the lightest possible neuromodulator — two MLPs
  computing graph parameters — already wins on real-world trajectory
  prediction benchmarks.
- *L'Haridon & Cañamero 2023* keeps the Cañamero programme alive at the
  embodied-pain frontier — and is the *closest analogue to the present
  project* in the entire corpus.
- *AlKilany & Goodman 2025* and *Espino 2024* demonstrate scalable spiking
  implementations on real hardware.

By 2026 the field is no longer about whether to use neuromodulation, but
about *which* implementation (activation-function, structural mask, low-rank
scaling, dynamical-gain, hopfield-arousal) and at *which* level (cellular,
mean-field, population, network, behavioural). The corpus has not converged
on an answer.

## 4. Cross-cluster connections matrix

Light marking only — cells with explicit bridging papers in the corpus are
named; empty cells mean no documented cross-pollination in the per-paper
*Connections* subsections we have read.

|         | **A** (Cañamero) | **B** (Krichmar) | **C** (RL/DL) | **D** (Gain/RNN) | **E** (CL) | **F** (Hopfield) | **G** (SNN) |
|---|---|---|---|---|---|---|---|
| **A** | — | Cañamero 1997 → Cox & Krichmar 2009; Krichmar & Hwu 2022; Chiba & Krichmar 2020 (interoception bridge) | Lones 2018 → Mei 2022 (cited as hormonal example) | Lewis 2016 → Wainstein 2025 (uncertainty/affect) (indirect) | Lones 2018 → Kudithipudi 2022 (epigenetic plasticity) | — | — |
| **B** | (see A row) | — | Cox & Krichmar 2009 → Doya 2002 (explicit foundation); Zou 2020 → Xing 2022 (RL extension); Avery & Krichmar 2017 → Lee 2024 | Krichmar & Hwu 2022 → Shine 2021 (gain modulation); Hwu & Krichmar 2020 → Driscoll 2022 (modularity) | Hwu & Krichmar 2020 → Kolouri 2019 (schemas + plasticity); Xing 2022 → Rodriguez-Garcia 2026 (NE/ACh CL) | Alonso & Krichmar 2023 → Hwu & Krichmar 2020 (hippocampal indexing) | Espino 2024 → Xing 2020 (cost-map combining) |
| **C** | Mei 2022 → Cañamero/Krichmar (cited in §Connections) | (see B row) | — | Vecoven 2020 → Costacurta 2024 (modulator subnet template); Wang 2025 NEST → Vecoven 2020 (cited) | Mei 2022 → Kudithipudi 2022; Lee 2024 → Durstewitz 2025 (continual-RL) | Doya 2002 → Osman 2024 (Boltzmann/inverse-temperature) | Vecoven 2020 → AlKilany 2025 (modulator template) |
| **D** | — | Wainstein 2025 → Krichmar lab (cited LC literature) | (see C row) | — | Wainstein 2025 → Rodriguez-Garcia 2026 (gain → entropy → CL); Driscoll 2022 → Durstewitz 2025 (attractor reuse) | Wainstein 2025 → Osman 2024 (energy landscape, perceptual switches) | — |
| **E** | (see A row) | (see B row) | (see C row) | (see D row) | — | Kolouri 2019 → Alonso & Krichmar 2023 (selective plasticity) | Kudithipudi 2022 → Espino 2024 (neuromorphic lifelong) |
| **F** | — | Alonso & Krichmar 2023 → Hwu & Krichmar 2020 (hippocampus indexing) | Osman 2024 → Doya 2002 (Boltzmann analogy) | Osman 2024 → Shine 2021 (mean-field, Sompolinsky) | (see E row) | — | — |
| **G** | — | Espino 2024 → Xing 2020 (Krichmar lab); AlKilany 2025 → Krichmar 2012 (early SNN modulation) | AlKilany 2025 → Vecoven 2020 (modulator architecture template); Espino 2024 → Mei 2022 (sub-second neuromodulation) | — | (see E row) | — | — |

Read across rows: "what does cluster X borrow from?" Read down columns: "what
does cluster X give to?" The diagonal is empty by convention.

The **three densest cells** (B↔C; D↔E; A↔B) are the corpus's main
intellectual highways. The **emptiest row** is G — spiking neural networks
have not yet been integrated with the RNN-dynamics or Hopfield clusters, even
though much of the analytical machinery (mean-field, attractors, energy
landscapes) is shared.

## 5. Open questions for this project

The present project is building an *interoceptive reinforcement-learning
agent* in a grid world. Its "modulator" head should influence the agent's
perception and action precision, and the project framings cited at intake
include **FiLM-style gating** (Feature-wise Linear Modulation, the deep-
learning version of multiplicative gain modulation), **precision-weighted
predictive coding** (active inference's view of attention), and
**risk-sensitive policies** (the corpus's nearest analogue: Krichmar 2013's
5-HT inhibits DA opponency, and Asher-Zaldivar-Krichmar's Hawk-Dove two-critic).

From the corpus, six actionable open questions stand out:

1. **Should the modulator scale activations (Vecoven 2020, Wainstein 2025),
   gate them multiplicatively (Ben-Iwhiwhu 2022, Costacurta 2024), or
   reshape per-task structure (Wang 2024 NeuronML)?** The corpus has no
   head-to-head comparison and the answer matters for which FiLM variant the
   project should default to. *Closest corpus analogue:* Mei 2022's
   four-scale framework explicitly recommends *combining* all three, but
   has no empirical benchmark.

2. **Is the right uncertainty signal for the modulator the softmax-output
   entropy (Wainstein 2025, Rodriguez-Garcia 2026), the
   epistemic-uncertainty average over actions (Lee 2024 Doya-DaYu), or the
   contrastive Excitation Backprop attention magnitude (Zou 2020, Kolouri
   2019)?** Each comes with a different physical interpretation (cortical
   gain, exploration-exploitation, attentional spotlight) and different
   computational cost. For a *pain-precision* modulator, the
   homeostatic-deficit signal of Cañamero 1997 / Lones 2018 / L'Haridon-
   Cañamero 2023 may be the right *primary* signal, with these uncertainty
   signals as modifiers.

3. **Should pain perception be modulated by a single hormonal scalar (L'Haridon
   & Cañamero 2023: pain = cortisol × damage), or should it sit inside a
   precision-weighted predictive-coding loop (Shine 2021's mapping of ACh ↔
   bottom-up precision)?** Both views are present in the corpus but never
   compared in the same agent.

4. **How does the modulator avoid catastrophic forgetting in the agent's
   replay-and-update loop?** The continual-learning cluster (E) gives three
   competing answers — selective-plasticity tags (Kolouri), entropy-driven
   gain (Rodriguez-Garcia), and attractor-structure regularisation
   (Durstewitz). For our agent, the bandwidth of the modulator's *gradient*
   into the precision head is the critical knob; we have no a-priori reason
   to pick one of these three.

5. **Should the modulator's training be a separate inner-loop (Vecoven 2020's
   z = f(history); Costacurta 2024's NM-RNN) or interleaved with the main
   policy (Wang 2024 NeuronML's bi-level optimisation)?** This is a
   FiLM-style architectural question that the corpus answers inconsistently.

6. **Is the right level of modulation per-neuron (Vecoven 2020), per-layer
   (Ben-Iwhiwhu 2022), per-rank-1-component (Costacurta 2024), or global
   (Tsuda 2021, Wainstein 2025)?** The corpus does not converge — and the
   answer probably depends on the size of the agent's network. For the
   small-RNN scale of our grid-world agent, Tsuda's "one global scalar"
   demonstration is the most encouraging existence proof; for any scale-up
   later, Ben-Iwhiwhu's per-layer or Costacurta's low-rank scaling will
   likely be needed.

These questions are exactly the kind that an `experiment-designer` or
`senior-developer` plan can convert into runnable comparisons. The synthesis
makes no recommendation about which to prioritise — that decision belongs to
the project plan, the PI, and the next round of experiment design.

---

**Companion documents.** This synthesis lives alongside the master index in
[`neuromodulatory_algorithms_lit_review.md`](neuromodulatory_algorithms_lit_review.md)
and the 41 per-paper reviews under [`reviews/`](reviews/). Per-paper deep
reviews have YAML frontmatter, plain-English entry points, section-by-section
backbones, undergraduate and graduate syntheses, and Connections subsections.
The curator did not modify any per-paper review during this synthesis pass.
