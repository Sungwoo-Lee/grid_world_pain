---
title: "Neuromodulatory Algorithms — Master Lit-Review Index"
topic: neuromodulatory_algorithms
papers: 41
last_curated: 2026-05-15
---

# Neuromodulatory Algorithms — Master Lit-Review Index

## Plain-English entry point

This file is the **table of contents** for the project's reference corpus on
*neuromodulatory algorithms* — the cluster of 41 papers we have collected on how
brains use diffuse chemical signals (dopamine, serotonin, noradrenaline,
acetylcholine, and hormones such as oxytocin and cortisol) to **change how a
network computes** without changing the network's wiring, and how artificial
agents can borrow that trick. *Neuromodulators* are slow, broadcast signals
released into wide volumes of tissue. Where ordinary synapses send a point-to-
point message, a neuromodulator nudges every neuron in its reach — turning gain
up or down, opening or closing plasticity, biasing exploration vs. exploitation,
and shifting whole-brain states like sleep, arousal, or pain.

The corpus was assembled to answer one project-level question: **how should the
"modulator" head of an interoceptive reinforcement-learning agent in our grid
world influence the agent's perception, action selection, and learning rules?**
Each per-paper review under `reviews/` already contains a plain-English summary,
a section-ordered backbone, an undergraduate-level synthesis, and a graduate
deep-dive with LaTeX. *This file does no summarising of its own* — it groups the
41 papers into seven thematic clusters, gives a one-line description per paper,
and points readers to:

- **the cross-paper synthesis** in
  [`neuromodulatory_algorithms_synthesis.md`](neuromodulatory_algorithms_synthesis.md),
  which carries the narrative thread (history, shared assumptions,
  disagreements, key equations, open questions);
- **the per-paper deep reviews** under [`reviews/`](reviews/), one file per
  paper, each kept intact by the curator.

Reading path: if you are new, read the synthesis end-to-end first (it leans
plain-English in every section), then drop into individual reviews from this
index. If you already know the field, use this index as a paper finder.

## Master table of contents (grouped by cluster)

The seven clusters were chosen by reading every per-paper review's *Connections*
subsection and by tracing influence flows across the 1997 → 2026 timeline. The
synthesis document defends the cluster boundaries and lists the bridging
papers; this index just enumerates the membership.

---

### Cluster A — Affective robotics & motivation/emotion lineage (Cañamero school)

Hand-designed homeostatic robots with simulated hormones modulating perception,
action selection, and (in later papers) developmental plasticity and social
state. Started by Cañamero 1997; extended through 2023.

- [canamero_1997_motivations_emotions](reviews/canamero_1997_motivations_emotions.md) —
  Foundational paper introducing the *Abbott / Gridland* hormone-modulated
  homeostatic action-selection architecture; ART-1 vigilance threshold $\rho$
  set by emotional state.
- [canamero_2005_emotion_understanding](reviews/canamero_2005_emotion_understanding.md) —
  Methodological review distinguishing **designed-emotion** (Abbott-style) from
  **emergent-emotion** (Braitenberg) architectures; positions Cañamero school
  inside the *neuromodulation-of-control-architecture* family.
- [blanchard_canamero_2006_affect_modulated](reviews/blanchard_canamero_2006_affect_modulated.md) —
  Koala robot whose *well-being* and *affect* scalars autonomously switch
  between stability-seeking, exploration, exploitation, and low-level
  imitation. Affect flips the sign of motivation-to-continue.
- [cos_2010_affordances_consummatory](reviews/cos_2010_affordances_consummatory.md) —
  Khepera robot learns Gibsonian affordances via hormone-reinforced Hebbian
  updates on a Growing-When-Required network; affordance × drive product gates
  behaviour.
- [lones_canamero_2013_epigenetic_hormones](reviews/lones_canamero_2013_epigenetic_hormones.md) —
  Adds a **3-minute developmental window** during which hormone-gland activity
  $\theta_h$ and receptor sensitivity $\text{sens}_h$ drift to match the
  environment.
- [lewis_canamero_2016_hedonic_pleasure](reviews/lewis_canamero_2016_hedonic_pleasure.md) —
  Argues **pleasure ≠ reward**: hedonic pleasure decoupled from need
  satisfaction still has homeostatic value via incentive-salience modulation
  $\alpha$.
- [lones_2018_hormone_epigenetic](reviews/lones_2018_hormone_epigenetic.md) —
  IEEE TCDS extension of Lones 2013: one-line exponential receptor-update rule
  $\text{Sens}_i(t{+}1) = \text{Sens}_i(t)^{E_h^i/\sigma}$ produces phenotypes
  tailored to six environments.
- [khan_canamero_2022_social_buffering](reviews/khan_canamero_2022_social_buffering.md) —
  Six-agent NetLogo society where oxytocin (OT) buffers cortisol (CT)-driven
  stress; bonded agents accumulate OT early and front-load resilience.
- [scarinzi_canamero_2022_affective_interactions](reviews/scarinzi_canamero_2022_affective_interactions.md) —
  Opinion piece distinguishing *affection* (bodily resonance) from *e-motion*
  (action readiness); argues for organismoid embodiment in HRI.
- [lharidon_canamero_2023_stress_pain](reviews/lharidon_canamero_2023_stress_pain.md) —
  Khepera-IV vs. Thymio-II predator setup; **pain = cortisol × damage** in a
  homeostatic-control loop. Closest analogue to the present project's pain
  modulation hypothesis.

---

### Cluster B — Neurorobotics & neuromodulator-controller lineage (Krichmar school)

Wheeled / humanoid robots whose controller is a small mean-firing-rate or
event-based neural network with explicit nuclei for DA / 5-HT / ACh / NE; the
canonical "decisiveness framework" (Krichmar 2008/2013): all modulators sharpen
the signal-to-noise ratio in downstream targets but differ in their triggers.

- [cox_krichmar_2009_neuromodulation_robot_controller](reviews/cox_krichmar_2009_neuromodulation_robot_controller.md) —
  CARL-1 robot with VTA / Raphe / BF nuclei; BCM Hebbian rule gated by
  neuromodulator activity; lesion methodology established.
- [krichmar_2013_neurorobotic_anxiety_curiosity](reviews/krichmar_2013_neurorobotic_anxiety_curiosity.md) —
  CarlRoomba in an open-field test; 5-HT inhibits DA, OFC / mPFC inhibit
  modulators, four parameter manipulations reproduce pharmacological
  phenotypes.
- [avery_krichmar_2017_models_neuromodulation](reviews/avery_krichmar_2017_models_neuromodulation.md) —
  Encyclopedia chapter: canonical mappings of DA / 5-HT / ACh / NA; Hawk-Dove
  two-critic model; Doya-vs-Krichmar synthesis.
- [hwu_krichmar_2020_schemas_memory](reviews/hwu_krichmar_2020_schemas_memory.md) —
  Schema-learning network reproducing Tse et al. 2007 rats; novelty × schema-
  familiarity drives replay epochs.
- [chiba_krichmar_2020_self_monitoring](reviews/chiba_krichmar_2020_self_monitoring.md) —
  Three-level (sensory / homeostatic / cognitive) self-monitoring framework;
  unifies the Krichmar and Cañamero schools via the *interoception bridge*.
- [xing_2020_neuromodulated_patience](reviews/xing_2020_neuromodulated_patience.md) —
  Android-Based Robot in outdoor parks; 5-HT scalar in Miyazaki-2018 wait/quit
  Bayesian rule.
- [zou_2020_neuromodulated_attention](reviews/zou_2020_neuromodulated_attention.md) —
  Contrastive Excitation Backprop with an ACh+NE head; ACh tracks per-goal
  confidence, NE resets when softmax fails to commit.
- [xing_2022_neuromodulation_rl_environment_changes](reviews/xing_2022_neuromodulation_rl_environment_changes.md) —
  Extension of Zou 2020 to deep RL (PPO + TD3): ACh task-confidence + NE
  novelty signal gives 100-1000× faster recall on returning tasks.
- [krichmar_hwu_2022_design_principles_neurorobotics](reviews/krichmar_hwu_2022_design_principles_neurorobotics.md) —
  Position paper: 14 design principles, organised into embodiment / adaptive
  behaviour / behavioural trade-offs; citation hub for the entire Krichmar-lab
  programme.

---

### Cluster C — RL, meta-RL, and deep-learning neuromodulation

Direct algorithmic descendants of Doya 2002: neuromodulators implemented as
learnable functions inside artificial networks that shape activations,
hyperparameters, or per-task structure. Mostly post-2020.

- [doya_2002_metalearning_neuromodulation](reviews/doya_2002_metalearning_neuromodulation.md) —
  Foundational paper: DA = $\delta$, 5-HT = $\gamma$, NA = $\beta$, ACh =
  $\alpha$; predicted modulator-interaction matrix (Fig. 9).
- [vecoven_2020_neuromod_dnn](reviews/vecoven_2020_neuromod_dnn.md) —
  Slope-and-bias modulation of saturated ReLU activations via a shared $z$
  vector from a context RNN; 97 % of Bayes-optimal on a bandit benchmark.
- [beniwhiwhu_2022_context_meta_rl](reviews/beniwhiwhu_2022_context_meta_rl.md) —
  Multiplicative gating layer (NPN) inserted into CAVIA / PEARL; wins on
  Meta-World ML45 and CT-graph depth-4.
- [mei_2022_multiscale_neuromod](reviews/mei_2022_multiscale_neuromod.md) —
  Trends-in-Neurosciences review proposing **four-scale framework**
  (hyperparameter / cell-type / weight / compartmental) for neuromodulation in
  DNNs.
- [lee_2024_lifelong_rl](reviews/lee_2024_lifelong_rl.md) —
  The **Doya-DaYu agent**: $\alpha = E/(E{+}A)$ from ACh, $\beta = 1/\langle
  E\rangle$ from NA; non-stationary bandit + proposed mouse experiment.
- [wang_2024_neuromod_meta](reviews/wang_2024_neuromod_meta.md) —
  **NeuronML**: learnable per-task structural mask under frugality + plasticity
  + sensitivity constraints; bi-level optimisation over weights and mask.
- [wang_2025_nest_hypergraph](reviews/wang_2025_nest_hypergraph.md) —
  **NEST**: small-world hypergraph trajectory predictor for autonomous driving;
  the "neuromodulator" is a two-MLP module computing the clustering threshold
  $\alpha$ and shortcut probability $\beta$.

---

### Cluster D — Gain modulation & RNN dynamics

Biology (Ferguson & Cardin) + theoretical/computational accounts (Shine 2021)
of how modulators reshape population gain functions, plus RNN-style
demonstrations (Tsuda, Driscoll, Costacurta, Wainstein) of how a single
scalar/low-rank gain knob switches a network's behaviour.

- [ferguson_cardin_2020_gain_modulation](reviews/ferguson_cardin_2020_gain_modulation.md) —
  *Nature Reviews Neuroscience* primer: multiplicative vs. additive
  modulation; PV / SST / VIP interneuron classes; ACh / NA / 5-HT / DA effects.
- [shine_2021_cellular_to_dynamics](reviews/shine_2021_cellular_to_dynamics.md) —
  Five microscopic mechanisms → mean-field population sigmoid → mesoscale
  dynamics → cognition (predictive coding / active inference). Maps modulators
  onto height / slope / threshold / width of the population gain function.
- [tsuda_2021_activity_hypertubes](reviews/tsuda_2021_activity_hypertubes.md) —
  Multiplying *every* weight in an RNN by a scalar $f_{nm}$ shifts activity
  into a non-overlapping "hypertube" in state space; reproduces Inagaki et al.
  2012 fly PER. "Circuit-based sensitivity" varies 3× across networks.
- [driscoll_2022_dynamical_motifs](reviews/driscoll_2022_dynamical_motifs.md) —
  15-task RNN reuses *dynamical motifs* (ring attractors, paired attractors,
  unstable saddles) across tasks; rule input is the switch. Foundation for
  Costacurta 2024.
- [costacurta_2024_structured_flexibility](reviews/costacurta_2024_structured_flexibility.md) —
  **NM-RNN**: a small modulator subnet $z(t)$ scales the rank-$K$ components
  of a low-rank recurrent matrix; mathematically equivalent to LSTM forget
  gates.
- [wainstein_2025_gain_perceptual_switches](reviews/wainstein_2025_gain_perceptual_switches.md) —
  Pupillometry + fMRI + RNN model; LC-NA-driven gain raises softmax entropy →
  gain rises → attractor flattens → perceptual switch at uncertainty maxima.

---

### Cluster E — Continual / lifelong learning via neuromodulation

How neuromodulator-style mechanisms can attenuate catastrophic forgetting,
close the *stability gap*, and support multi-task plasticity. Includes the two
big Perspectives (Kudithipudi 2022, Durstewitz 2025).

- [kolouri_2019_attention_plasticity](reviews/kolouri_2019_attention_plasticity.md) —
  Attention-based selective plasticity via contrastive Excitation Backprop +
  Oja's rule; biologically motivated stand-in for Elastic Weight Consolidation
  / Synaptic Intelligence.
- [kudithipudi_2022_lifelong_learning](reviews/kudithipudi_2022_lifelong_learning.md) —
  31-author *Nature Machine Intelligence* Perspective; catalogues biological
  ingredients (CLS, metaplasticity, neuromodulation, neurogenesis) for
  lifelong-learning machines.
- [durstewitz_2025_neuroscience_continual_learning](reviews/durstewitz_2025_neuroscience_continual_learning.md) —
  2025 companion piece to Kudithipudi 2022 from a dynamical-systems / PFC
  angle; argues for **manifold / ghost attractors** and **multi-scale
  plasticity** as the missing ingredients.
- [rodriguezgarcia_2026_ne_stability_gap](reviews/rodriguezgarcia_2026_ne_stability_gap.md) —
  **NGM-SGD**: noradrenaline-inspired gain controlled by softmax entropy
  reduces the stability gap 2-15× across five CL benchmarks.

---

### Cluster F — Hopfield / energy-based neuromodulation

Energy-based associative-memory networks where neuromodulator-style gain or
arousal parameters control the depth and topology of the energy landscape.

- [osman_2024_hopfield_arousal](reviews/osman_2024_hopfield_arousal.md) —
  Continuous Hopfield with divisive recurrence $M/\alpha$; pitchfork
  bifurcation at $\alpha^* = \lambda_{\max}(M)$; energy = Boltzmann-machine
  free energy with $\alpha$ as inverse prior; Ising-temperature analogy.
- [alonso_krichmar_2023_sparse_quantized_hopfield](reviews/alonso_krichmar_2023_sparse_quantized_hopfield.md) —
  **SQHN**: tree-structured sparse-quantized Hopfield with a memory/root node
  doing one-shot online-continual learning; matches MHN+BP without backprop.
- [tambas_2025_krotov_hopfield_rbm](reviews/tambas_2025_krotov_hopfield_rbm.md) —
  Krotov-Hopfield winner-take-all-with-runner-up-penalty rule as a
  neuromodulator layer for an RBM; reduces hidden-unit redundancy on MNIST.

---

### Cluster G — Spiking neural networks + neuromodulation

Neuromodulation in event-driven spiking models; both demonstrate that
sub-second modulation of intrinsic biophysical parameters (not weights) is a
scalable lever, and both target neuromorphic deployment (Loihi / SpiNNaker).

- [espino_2024_snn_path_planning](reviews/espino_2024_snn_path_planning.md) —
  Spiking Wavefront Planner (SNN cost map) + E-Prop online learning; Clearpath
  Jackal in Aldrich Park; flags neuromodulation for combining cost maps as
  future work.
- [alkilany_goodman_2025_snn_dynamic_sensory](reviews/alkilany_goodman_2025_snn_dynamic_sensory.md) —
  Modulator network outputs LIF parameters (threshold, reset, $\tau_m$,
  $\tau_s$) on the fly; SNN discovers *listen-in-the-dips* strategy under SAM
  noise.

---

## Pointer to the synthesis document

For the cross-paper narrative — *what is the field's consensus?*, *where do the
clusters disagree?*, *how did Cañamero 1997 → Krichmar 2009 → Doya 2002 →
Vecoven 2020 → Rodriguez-Garcia 2026 connect across thirty years?* — see:

**[`neuromodulatory_algorithms_synthesis.md`](neuromodulatory_algorithms_synthesis.md)**

The synthesis has six sections:

1. Plain-English entry point (the field for newcomers).
2. The seven thematic clusters, each with shared assumptions, disagreements,
   and 2-4 key equations in LaTeX.
3. Historical timeline 1997 → 2026 in five-year windows, tracing influence
   flows rather than publication order.
4. A cross-cluster connections matrix (which clusters borrow from which).
5. Open questions for this project — actionable directions for the
   grid-world-pain agent's modulator head.
