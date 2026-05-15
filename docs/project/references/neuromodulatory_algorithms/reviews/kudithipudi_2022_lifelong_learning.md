---
title: "Biological underpinnings for lifelong learning machines"
authors:
  - Dhireesha Kudithipudi
  - Mario Aguilar-Simon
  - Jonathan Babb
  - Maxim Bazhenov
  - Douglas Blackiston
  - Josh Bongard
  - Andrew P. Brna
  - Suraj Chakravarthi Raja
  - Nick Cheney
  - Jeff Clune
  - Anurag Daram
  - Stefano Fusi
  - Peter Helfer
  - Leslie Kay
  - Nicholas Ketz
  - Zsolt Kira
  - Soheil Kolouri
  - Jeffrey L. Krichmar
  - Sam Kriegman
  - Michael Levin
  - Sandeep Madireddy
  - Santosh Manicka
  - Ali Marjaninejad
  - Bruce McNaughton
  - Risto Miikkulainen
  - Zaneta Navratilova
  - Tej Pandit
  - Alice Parker
  - Praveen K. Pilly
  - Sebastian Risi
  - Terrence J. Sejnowski
  - Andrea Soltoggio
  - Nicholas Soures
  - Andreas S. Tolias
  - Darío Urbina-Meléndez
  - Francisco J. Valero-Cuevas
  - Gido M. van de Ven
  - Joshua T. Vogelstein
  - Felix Wang
  - Ron Weiss
  - Angel Yanguas-Gil
  - Xinyun Zou
  - Hava Siegelmann
year: 2022
venue: "Nature Machine Intelligence, 4, 196–210"
slug: kudithipudi_2022_lifelong_learning
source_pdf: "sources/Kudithipudi et al. 2022 - Biological underpinnings for lifelong learning machines.pdf"
topic: neuromodulatory_algorithms
---

# Kudithipudi et al. (2022) — Biological underpinnings for lifelong learning machines

## Plain-English entry point

This paper is a multi-institution **Perspective** in *Nature Machine Intelligence* that asks one practical question: **what would it take to build a machine that keeps learning for its entire service life — like an animal — rather than freezing after one training run?** The authors call this capability *lifelong learning* (L2 / L2M, "lifelong-learning machine").

The headline claim is that biology has already solved most of the sub-problems, and that today's artificial-intelligence research can borrow those solutions if it stops treating the brain as just "deep nets with backprop". The paper does three things:

1. It names **six features** any lifelong learner must have: knowledge transfer to new tasks, avoidance of *catastrophic forgetting* (the well-known phenomenon where a neural network trained on a new task overwrites what it learned on the old one), exploitation of similarities between tasks, *task-agnostic* operation (no oracle telling you when the task changed), noise tolerance, and resource efficiency.
2. It enumerates **nine biological mechanisms** that contribute to those features — **neurogenesis** (growth of new neurons), **episodic replay** (the hippocampus rehearsing past experience during sleep), **metaplasticity** (the "plasticity of plasticity": how easily a synapse can be modified itself varies with history), **neuromodulation** (broadcast chemical signals — dopamine, acetylcholine, noradrenaline, serotonin — that re-tune learning rates and behaviour), **context-dependent perception and gating**, **hierarchical distributed control**, **cognition outside the brain** (single cells and tissues that compute via bioelectric networks), **reconfigurable organisms**, and **multisensory integration**.
3. It maps existing bio-inspired AI work onto this 6-feature × 9-mechanism matrix and identifies the gaps.

The piece matters to this project because every other paper in the `neuromodulatory_algorithms` corpus is one of the puzzle-pieces this survey is trying to assemble — the survey is the canonical reading-list backbone for the corpus.

## Section-ordered backbone

**Introduction.** Lifelong learning (L2) is a current-generation AI gap: self-driving cars, autonomous drones, delivery robots, and wearable devices will all need to *keep* learning in the field while conserving compute, memory, and energy. Animals (invertebrates through humans) do this routinely. The paper limits scope to biologically inspired approaches and explicitly excludes pure-ML continual-learning schools (rehearsal-only, architectural-only, regularisation-only meta-learning) that lack a clear biological referent.

**Key features of lifelong learning.** Six features are defined: (1) **Transfer and adaptation** — apply old knowledge to new tasks and adapt rapidly without offline retraining; few-shot and meta-learning attack this. (2) **Overcoming catastrophic forgetting** — the stability–plasticity dilemma; old memory is *overwritten*, not lost from lack of capacity. (3) **Exploiting task similarity** — forward and backward transfer, achieved via compositionality / sub-task reuse. (4) **Task-agnostic learning** — no oracle tells the system when tasks switch or which one is active. (5) **Noise tolerance** — graceful performance on out-of-distribution sensors / inputs. (6) **Resource efficiency and sustainability** — replay buffers cannot grow forever; inference latency must not blow up with the number of tasks.

**Biological mechanisms.** Each of the nine mechanisms is described in one section:

- **Neurogenesis.** Adult production of new neurons in dentate gyrus and subventricular zone; correlated with rich experience in mice; offers extra capacity for new memories without overwriting old ones. Insect metamorphosis shows learned responses survive massive neural remodelling.
- **Episodic replay.** Hippocampal place-cell sequences from waking experience are replayed during sleep / rest, often time-compressed into sharp-wave ripples, and coordinated with neocortex. Supports the *complementary learning systems* (CLS) hypothesis: fast hippocampal encoding then slow neocortical consolidation. Some replay is "Brownian" / non-experienced — closer to *generative replay* than literal rehearsal. REM dreams are out-of-distribution elaborations, potentially aiding generalisation.
- **Metaplasticity.** "Plasticity of plasticity" — a synapse's biochemical state, modulated by its modification history and recent activity, sets how easily it can change next. Limited (4–5 bit) biological synapse precision creates catastrophic forgetting; cascade models with multiple timescales solve it (Fusi, Drew, Abbott 2005; Benna & Fusi 2016). Heterosynaptic modulation is crucial for synaptic consolidation.
- **Neuromodulation.** Subcortical nuclei release broadcast neurotransmitters with local + global effects on activity and plasticity. ACh: stimulus-driven vs. goal-driven attention, expected uncertainty; NA: novelty / surprise / unexpected uncertainty (Yu & Dayan 2005); 5-HT: patience / risk; dopamine: reward prediction error, STP-to-LTP gating. Phasic = decisive / exploitative; tonic = curious / exploratory. Insect mushroom body: valence encoding.
- **Context-dependent perception and gating.** Olfactory bulb receives more top-down than bottom-up input; gain modulation in insect vision enhances behaviourally relevant trajectories. Prefrontal cortex stores *schemas* that allow new similar memories to slot in without overwriting old ones.
- **Hierarchical distributed systems.** Many organisms have no central brain or a tiny one; computation is decentralised, with high intra-cluster and sparse inter-cluster connectivity. Central pattern generators autonomously handle perturbations during locomotion. Brain–body co-evolution makes possible robust control under noisy sensors and slow actuators.
- **Cognition outside the brain.** Bioelectric networks (BEN), gap junctions, transcriptional networks let single cells and tissues compute, learn, and respond to novelty. A simple BEN can be trained as a logic gate. Same machinery handles morphogenesis *and* decision-making.
- **Reconfigurable organisms.** Tadpoles with eyes on the tail still learn to see; planarians regenerate; xenobots self-assemble from skin cells. Bioelectric circuits hold global anatomical knowledge separable from genome.
- **Multisensory integration.** Vision + tactile + auditory + proprioception combined for balance and coordinated movement, e.g. superior colliculus orienting; this supports task-agnostic processing.

**Application of biologically inspired models in L2.** A reverse mapping — for each L2 feature, the paper lists which bio-inspired mechanisms have been *implemented in algorithms* and provides reference numbers in Figure 10. Highlights: neuromodulation has been used for transfer / adaptation (Doya 2002, Soltoggio et al., Velez & Clune), for overcoming forgetting (uncertainty-modulated plasticity), for task-agnostic learning, and for noise tolerance. Replay (Brain-Inspired Replay, van de Ven et al. 2020) and generative replay tackle catastrophic forgetting at scale. Cascade metaplasticity (Benna & Fusi 2016) and binary-weight metaplasticity (Laborieux et al. 2021) handle forgetting under bounded resources. Context-dependent gating (Masse et al. 2018) disentangles task representations.

**Conclusions.** The most consistent message is that **single mechanisms are insufficient** — building a real L2M requires *composing* several of these biological tricks. Open needs: realistic continual test environments (not pre-prepared datasets), compute-efficient L2 architectures (neuromorphic hardware), and integration of mechanisms still unrepresented in AI (active forgetting, extinction, memory reconsolidation, gene-regulation, intercellular signalling).

## Phase 1 — Undergraduate-level synthesis

The big idea is simple. Today's neural networks are great at one task but terrible at the *sequence* of tasks an animal faces in its life — when you train a network on Task B, it usually destroys what it knew about Task A. Biologists know a dog, a bee, or even a flatworm does not have this problem. Why?

The paper inventories what those creatures have that current AI lacks. It is not one trick; it is a **portfolio**:

- They **grow** new neurons when they need more capacity (neurogenesis).
- They **rehearse** past experiences during sleep, so the new pattern does not crowd the old one out (replay).
- They have synapses whose **willingness to change** is itself learnable: important memories become "stiff" and rare events stay "soft" (metaplasticity).
- They release **broadcast chemicals** (dopamine, noradrenaline, acetylcholine, serotonin) that re-tune the brain's learning rate and exploration on the fly when the world surprises them (neuromodulation).
- They **gate** their networks by context, so the same neurons are reused in different "modes" depending on the situation (context gating).
- They **distribute** computation across the body and across hierarchical levels (a spinal pattern generator handles walking; the cortex just chooses a gait).
- They compute outside the brain — even single cells use **bioelectric voltage patterns** to represent and update goal-directed structure.

For each of these biological tricks the survey points at concrete recent AI papers that have prototyped the trick — usually demonstrating *one* feature (e.g. "we mitigated catastrophic forgetting on Split-MNIST") on a small benchmark. The set-up: pick a feature (say, catastrophic forgetting), grab a mechanism (say, replay), implement an analogue in a deep net, measure on continual benchmarks (Split-MNIST, Permuted-MNIST, CIFAR continual splits, robot navigation).

The result: each individual mechanism works for the feature it was designed for, but no current AI system has integrated enough of them to look like an animal. The paper's recommendation is therefore not "do mechanism X" but "**compose** several mechanisms, and **test** them on realistic continual environments instead of toy benchmarks."

## Phase 2 — Graduate-level deep dive

This is a survey, not an empirical paper, so it does not introduce its own equations. Below is the **biological-mechanism → algorithm mapping table** the question brief asked for, condensed from the paper's Figs. 2 and 10 and section text. Where the paper points at a canonical equation in another work, that equation is included.

### Biological mechanism → algorithmic implementation mapping

| Biological mechanism | L2 features it supports (Fig. 2) | Canonical algorithmic implementation | Key idea / equation |
|---|---|---|---|
| **Neurogenesis** | Overcome forgetting; Resource efficiency | Progressive Neural Networks (Rusu et al.); growing networks (Yoon et al.); Self-Net | Allocate new units when an existing population fails to represent novel input; freeze old columns |
| **Episodic replay** | Overcome forgetting; Noise tolerance | Experience Replay (Lin 1992; Mnih DQN); Brain-Inspired Replay (van de Ven, Siegelmann, Tolias 2020); pseudo-rehearsal (Robins 1995); generative replay (Shin et al. 2017) | Sample minibatches mixing $\mathcal{D}_t \cup \mathcal{D}_{<t}$; or draw $\tilde{x} \sim p_\theta(x \mid z)$ from a learned generator and interleave |
| **Metaplasticity** | Overcome forgetting; Resource efficiency | Cascade synapses (Fusi, Drew, Abbott 2005); Benna–Fusi (2016); EWC (Kirkpatrick et al. 2017); SI (Zenke, Poole, Ganguli 2017); Variational Continual Learning; binarized metaplasticity (Laborieux et al. 2021) | A per-synapse "importance" $\Omega_i$ penalises change: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \tfrac{\lambda}{2}\sum_i \Omega_i (\theta_i - \theta_i^{*})^2$ |
| **Neuromodulation** | Transfer; Overcome forgetting; Task-agnostic; Noise tolerance | Doya 2002 metalearning ($\beta, \alpha, \gamma, \rho$ assignment); Soltoggio et al. evolved modulatory neurons; Velez & Clune; Ben-Iwhiwhu et al. 2022 context meta-RL; Wang et al. 2024 neuromodulated meta-learning; Vecoven et al. 2020 | Modulator $m_t$ gates plasticity / activity: $\Delta w_{ij} = m_t \, \eta \, x_i y_j$ or $h^{\text{out}} = m_t \odot h^{\text{in}}$ (FiLM-style gain) |
| **Context-dependent gating** | Forgetting; Task similarity; Task-agnostic; Noise tolerance | Masse, Grant & Freedman 2018 (XdG); Serra et al. HAT; superposition of contexts | A context-conditioned binary or continuous mask $\mathbf{g}_t$ selects active subnetwork: $h_t = \mathbf{g}_t \odot \phi(W x_t)$ |
| **Hierarchical distributed systems** | Adaptation; Forgetting; Task similarity; Resource | Central-pattern-generator robotic controllers; options framework; subsumption architectures | Low-level dynamics handled in fast loops; high-level cortex only selects / tunes; reduces effective I/O dimensionality |
| **Cognition outside the brain** | Adaptation; Forgetting; Noise tolerance | Bioelectric-network trained as AND gate (Manicka & Levin 2022); biomolecular perceptron (Sarpeshkar group) | Cell-cell gap-junction networks with voltage gating: state evolution by ion fluxes through gated channels |
| **Reconfigurable organisms** | Adaptation; Task similarity; Noise tolerance | Xenobots (Kriegman, Blackiston, Bongard, Levin 2020) | Self-assembly + regeneration → noise-robust functional structure |
| **Multisensory integration** | Adaptation; Task-agnostic; Noise tolerance | Bio-inspired spiking multisensory net | Cross-modal binding $z = f(W_v x_v + W_a x_a + W_t x_t)$ with shared latent |

### Salient equations cited

The paper itself is largely text, but two cascade-metaplasticity ideas anchor much of its forgetting discussion:

**Cascade model of synaptic plasticity (Fusi, Drew & Abbott 2005).** A synapse is a state machine with $K$ levels per polarity, and a slow / fast hierarchy. Effective memory lifetime under random uncorrelated patterns scales much better than $1 / N$ for naïve binary synapses; concretely the signal-to-noise ratio satisfies

$$
\text{SNR}(t) \;\propto\; \frac{1}{\sqrt{t \, \log t}}
$$

instead of the $1 / t$ scaling of a single-timescale Hebbian synapse, giving polynomial improvement in retention.

**Elastic Weight Consolidation (Kirkpatrick et al. 2017, referenced as metaplasticity analogue).** When moving from task A to task B, penalise deviation from the post-A weights $\theta^{*}_A$ proportional to the diagonal Fisher information:

$$
\mathcal{L}_B(\theta) \;=\; \mathcal{L}_{\text{task}}^{B}(\theta) \;+\; \frac{\lambda}{2} \sum_{i} F_{i}^{A} \bigl( \theta_i - \theta_{i, A}^{*} \bigr)^2
$$

where $F_{i}^{A} = \mathbb{E}_{x \sim \mathcal{D}_A}\!\left[\bigl(\partial \log p(y \mid x, \theta) / \partial \theta_i \bigr)^2\right]$ approximates synapse-level importance after task A.

**Neuromodulator-as-gain (multiple corpus papers).** The paper's discussion of neuromodulation as a multiplicative re-tuning of activity and plasticity corresponds to the family of equations developed across the corpus (Doya 2002; Vecoven et al. 2020; Ben-Iwhiwhu et al. 2022). The most compact form is multiplicative gating:

$$
h_{t}^{\text{out}} \;=\; m_t \,\odot\, h_{t}^{\text{in}}, \qquad m_t \;=\; \sigma\bigl( W_m c_t + b_m \bigr)
$$

with $c_t$ a context / uncertainty signal and $\sigma$ a bounded nonlinearity. This is the unifying primitive linking the survey's "neuromodulation" entry to the gain-modulation papers (Rodriguez-Garcia et al. 2026, Ferguson & Cardin 2020, Wainstein et al. 2025) elsewhere in this corpus.

### Computational mechanism summary

The survey's central architectural claim is that none of the above equations should be deployed in isolation. The proposed *composite* lifelong-learning machine has, at minimum, four interacting subsystems:

1. A **growing capacity** module (neurogenesis or dynamic-architecture expansion) that allocates new neurons on demand under a saturation criterion.
2. A **replay / consolidation** module that interleaves new tasks with generated samples from old tasks (via a generative model rather than a stored buffer).
3. A **metaplasticity** layer that protects important parameters via per-synapse $\Omega_i$ or cascade dynamics, without requiring task-boundary signals.
4. A **neuromodulatory** controller that converts uncertainty / novelty / surprise signals into gain and learning-rate modulation across the network.

Context gating, hierarchical control, and multisensory binding are described as scaffolding around these four cores. The paper's Figure 10 shows that *no current AI system combines more than two or three of these mechanisms simultaneously* — the compositional system has yet to be built. That gap is the survey's principal call to action.

## Connections to other papers in this corpus

This survey is the **scaffold for the entire `neuromodulatory_algorithms` corpus**. Every other paper in the corpus implements at least one of its nine mechanisms. Explicit pointers:

- **Neuromodulation row.** The survey's neuromodulation discussion directly references work that overlaps with this corpus: Doya 2002 (cited heavily — see `doya_2002_metalearning_neuromodulation.md` if present in another batch), Soltoggio & colleagues, Ben-Iwhiwhu et al. 2022 (`ben-iwhiwhu_2022_*` in another batch), and the broader neuromodulation-in-deep-nets line that includes Vecoven et al. 2020 (`vecoven_2020_neuromodulation_deep_nets`), Mei et al. 2022 (`mei_2022_multiscale_neuromodulation`), Wang et al. 2024 (`wang_2024_neuromodulated_meta_learning`), and Lee et al. 2024 (`lee_2024_lifelong_rl_neuromodulation`). For the specific *gain-modulation* sub-variant this corpus collects, see Rodriguez-Garcia et al. 2026 (`rodriguezgarcia_2026_ne_stability_gap` — in this batch), Ferguson & Cardin 2020 (`ferguson_cardin_2020_gain_modulation`, other batch), Wainstein et al. 2025 (`wainstein_2025_gain_perceptual_switches`, other batch), and Shine et al. 2021 (`shine_2021_cellular_to_dynamics`, other batch).
- **Replay row.** Cites van de Ven, Siegelmann & Tolias 2020 — Siegelmann and Tolias are co-authors on this very survey, so the connection is direct. Brain-Inspired Replay is the canonical algorithm referenced.
- **Metaplasticity row.** Cites Fusi, Drew & Abbott (Stefano Fusi is a co-author of this survey) and Benna–Fusi 2016 — the cascade-synapse line.
- **Context-gating row.** Connects to Driscoll et al. 2022 on shared dynamical motifs (`driscoll_2022_shared_dynamical_motifs`, other batch), Tsuda et al. 2021 on hypertube shifts under neuromodulators (`tsuda_2021_hypertube_shifts`, other batch), and Costacurta et al. 2024 on structured flexibility via neuromodulation (`costacurta_2024_structured_flexibility`, other batch).
- **Reconfigurable organisms row.** Co-authors Levin, Blackiston, Bongard, Kriegman — these are the xenobot papers cited.
- **Cognition outside the brain.** Co-authors Levin and Manicka — the BEN logic-gate work cited.
- **Krichmar agents.** Co-authors include Krichmar — directly links the survey to most Krichmar-lab papers in this corpus: Cox & Krichmar 2009 (`cox_krichmar_2009_neuromodulation_robot`), Krichmar 2013 (`krichmar_2013_anxious_curious`), Avery & Krichmar 2017 (`avery_krichmar_2017_models_neuromodulation`), Hwu & Krichmar 2020 (`hwu_krichmar_2020_schemas`), Chiba & Krichmar 2020 (`chiba_krichmar_2020_self_monitoring`), and Krichmar & Hwu 2022 (`krichmar_hwu_2022_neurorobotic_principles`).
- **SNN row.** The "neuromorphic accelerators" closing paragraph thematically links to the SNN-based continual-learning work in this batch — Espino et al. 2024 (`espino_2024_snn_path_planning`), AlKilany & Goodman 2025 (`alkilany_goodman_2025_snn_dynamic_sensory`), and Krotov-Hopfield-based RBM extensions like Tambaş et al. 2025 (`tambas_2025_krotov_hopfield_rbm`).
- **Stability gap.** The survey's "stability–plasticity dilemma" terminology is the conceptual frame for Rodriguez-Garcia et al. 2026 (`rodriguezgarcia_2026_ne_stability_gap`, in this batch), which sits squarely in the neuromodulation × overcome-forgetting cell of Figure 10.
- **Durstewitz et al. 2025** (`durstewitz_2025_neuroscience_continual_learning`, in this batch) is the natural *companion piece* — a parallel review from a more PFC- and dynamical-systems-centric angle, three years later.
