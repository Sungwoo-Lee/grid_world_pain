---
title: "What neuroscience can tell AI about learning in continuously changing environments"
authors:
  - Daniel Durstewitz
  - Bruno Averbeck
  - Georgia Koppe
year: 2025
venue: "Nature Machine Intelligence, 7, 1897–1912"
slug: durstewitz_2025_neuroscience_continual_learning
source_pdf: "sources/Durstewitz et al. 2025 - What neuroscience can tell AI about learning in continuously changing environments.pdf"
topic: neuromodulatory_algorithms
---

# Durstewitz, Averbeck & Koppe (2025) — What neuroscience can tell AI about learning in continuously changing environments

## Plain-English entry point

This *Nature Machine Intelligence* Perspective is the **2025 companion piece** to the 2022 Kudithipudi et al. lifelong-learning survey. It asks the same question — *why can a rat adapt in a few trials when GPT-class models need billion-scale retraining?* — but answers it from a more **dynamical-systems and prefrontal-cortex** angle. Three authors: Durstewitz (theoretical neuroscience), Averbeck (NIH learning & decision-making), Koppe (computational psychiatry).

The headline claim is that animals master *non-stationary* environments using two coupled toolkits that mainstream AI almost ignores: **(a) neuro-dynamical mechanisms** — the brain operates near *bifurcations* in phase space, with **manifold attractors**, **ghost attractors**, **heteroclinic channels**, and **chaotic itinerancy** providing a substrate for fast, parameter-free in-context computation, and **(b) multi-scale plasticity** — synaptic change spans milliseconds (short-term plasticity) through seconds (behavioural-timescale plasticity, BTSP) through hours to years (LTP, structural plasticity, metaplasticity), letting one-shot experiences be ingrained while older memories are protected.

The authors map these phenomena onto the two big AI paradigms for handling novelty — **continual / lifelong learning** (in-weights updates) and **in-context learning** (no weight change; LLM-style few-shot inference) — and argue both are pale shadows of what brains do. Key takeaway: animal behaviour on rule-shift and drifting-bandit tasks shows **sudden jumps**, not gradient descent, in both behaviour and neural population activity. These jumps look like bifurcations in a dynamical system, gated by dopamine and other neuromodulators. The paper closes with a concrete agenda (Table 1) for porting eight specific neuro-mechanisms into AI architectures, including BTSP-based associative memories that are one-shot and resistant to overwriting.

For this project the paper matters because it is **the most recent, post-LLM-era statement** of the case that flexible adaptation needs more than gradient descent on bigger corpora — it needs dynamical-attractor structure and biologically inspired plasticity timescales.

## Section-ordered backbone

**Introduction.** Animals adapt continuously; modern AI is "train once, deploy frozen". This gap matters for embodied robotics, autonomous vehicles, and *agentic AI* (LLMs interacting online with humans). The authors integrate three threads usually kept separate: continual / lifelong learning, in-context learning (ICL), and neuroscience of rule-shifting and drifting-bandit tasks. They organise adaptation into two families: **in-weights** (parameter updates — fine-tuning, continual learning) and **in-context** (no weight change — ICL, chain-of-thought).

**Learning in non-stationary environments in AI systems.** All current AI training is gradient descent (GD); LRs of $10^{-3}$ to $10^{-6}$, billions of repetitions, slow and costly. *Catastrophic forgetting* / *plasticity–stability dilemma* known since the 1980s. Four mitigation families: (1) **regularise** parameter shifts (EWC, Bayesian priors); (2) **architectural** modularity / freezing / adapters / LoRA / neural Turing machines / retrieval-augmented transformers; (3) **experience replay** (hippocampus-inspired); (4) **functional resets** like Dohare et al. 2024 *continual backpropagation*, which re-initialises dormant ("unit-debris") neurons. All four mostly inherit GD's slowness.

**Adapting through inference.** *In-context learning* (ICL): an LLM is shown a sequence of $\{x_i, y_i\}$ from a never-before-seen function $\tilde f$ and predicts $\tilde y_{N+1}$ without weight change. The "in-context GD" explanation (transformer layers implement GD steps) is mathematically possible but empirically unsupported. More likely: ICL is associative-memory recall, interpolation, or compositional recombination of training tasks. ICL stays limited by the training corpus.

**Learning in non-stationary environments in animals.** Animal behaviour on rule-shifting and drifting bandit tasks resembles class-incremental learning. Five observations are highlighted:

1. New rules need only a few trials, not thousands.
2. Animals keep exploring even when reward is stable (sensible prior for non-stationarity).
3. Behaviour is **compositional** — reuse of pre-learned schemata and biases like "win-stay, lose-shift".
4. Performance changes are **sudden jumps**, not gradual curves. Gradual learning curves in textbooks are averaging artefacts.
5. Extinction is *suppression*, not unlearning — old behaviour can be reinstated rapidly.

Sudden behavioural transitions co-occur with abrupt re-organisation of prefrontal-cortex / cingulate population activity (Karlsson, Tervo, Karpova 2012; Durstewitz et al. 2010). Change-point analysis of neural activity correlates with change-point analysis of behaviour at $r > 0.91$, $p < 10^{-4}$.

**Neuro-dynamical mechanisms.** Dynamical Systems Theory (DST) treats the brain as a system of ODEs evolving through state space, converging to attractors (fixed points, limit cycles, chaos). The paper highlights:

- **Manifold attractors** — continuous sets of marginally stable fixed points. Substrate for graded working memory without parameter updates. Empirically supported by oculomotor integrator, place / head-direction cells. RNN training with manifold-attractor regularisation (Schmidt et al. 2024 ICLR) outperforms LSTMs on long-range arena.
- **Ghost attractors / heteroclinic channels** — attractors that just lost stability give a slow-flow region; chains of these implement sequences with adjustable timescales and produce *ramping* firing-rate profiles in timing tasks.
- **Bifurcations.** Many bifurcation types produce abrupt restructuring of dynamics when parameters move. Brains operate **near multiple bifurcations**, so small modulatory shifts (dopamine, noradrenaline) can rapidly re-configure the attractor landscape — explaining sudden behavioural jumps. Bifurcations also occur during GD training of recurrent models.
- **Chaotic itinerancy** — chaotic wandering among ghost attractors as a flexible cognitive substrate.

The paper stresses that DST gives a *common formal language* for biological and artificial recurrent systems.

**Plasticity mechanisms.** Two distinguishing properties:

1. **Many timescales.** Short-term depression / facilitation / post-tetanic potentiation (ms–min); LTP / LTD (min–days); structural plasticity in dendritic spines (min–hr); developmental plasticity windows (lifetime). Synaptic scaling prevents runaway activity. Metaplasticity (Abraham & Bear 1996) is "plasticity of plasticity", modulated by dopamine. Developmental windows close early in sensory cortex, late in PFC — supporting compositional learning on stable sensory primitives.
2. **Unsupervised, continual, one-shot.** No train/test split. Latent / incidental learning — one-time experiences stored for years without explicit reward. Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995) — fast hippocampus + slow neocortex, hippocampal replay during sleep consolidating into cortical schemata. **Behavioural timescale synaptic plasticity (BTSP)** (Bittner, Milstein, Grienberger, Romani & Magee 2017): seconds-wide window, one-shot place-field formation in hippocampal CA1, requires only co-occurrence of synaptic input and dendritic plateau potential. Tyulmankov, Yang & Abbott 2022; Tyulmankov, Tabachnik et al. 2024 show BTSP can drive content-addressable associative memory in artificial networks robust to overwriting — a plausible ICL substrate.

**Conclusion and future directions.** Table 1 lists eight neuroscience phenomena and their AI translations: manifold attractors (Schmidt et al. 2024); ghost attractors (Schmidt, ghost-attractor chains for non-catastrophic replay); multi-timescale synaptic plasticity (Sodhani et al. on multiple eligibility-trace timescales); structural plasticity (Dohare et al. continual backprop); developmental plasticity stages (layer-freezing schedules); metaplasticity (meta-learning of learning-rate); CLS (artificial + spiking hybrids); BTSP (one-shot associative memories).

The authors propose four agenda items: (a) align AI benchmarks with rule-shift / drifting-bandit animal-task designs; (b) use AI as functional hypothesis-generation for neuroscience (train RNNs on the same tasks animals do, dissect dynamical motifs); (c) use **dynamical-systems reconstruction (DSR)** — train surrogate generative models on multi-modal spike + behaviour data — to inherit brain computation directly; (d) use ML to infer synaptic plasticity rules from data.

## Phase 1 — Undergraduate-level synthesis

Here is the question and the answer in plain English.

**Question.** Why can a rat in a maze learn a new rule in a handful of trials, while ChatGPT had to read a sizeable fraction of the internet to learn its weights, and cannot really update them on the fly?

**Answer (the paper's argument).** Today's AI has two adaptation tools: (1) **fine-tuning / continual learning** — change the weights with gradient descent on new data, slowly, and risk forgetting old tasks; (2) **in-context learning** — feed examples into the prompt and let the frozen model figure out the new pattern. The brain uses neither of these as its main mechanism. Instead it uses two complementary toolkits.

**Tool 1: Brain dynamics that sit near tipping points.** Imagine the brain's network activity as a ball rolling on a landscape of valleys (attractors). The brain keeps the landscape *almost* flat in many places — a ridge that is barely a ridge ("manifold attractor"), or a valley that just stopped being a valley ("ghost attractor"). On such a landscape, **a tiny push from a neuromodulator** (a chemical broadcast like dopamine, noradrenaline, acetylcholine) **can completely re-route the ball**: a new valley appears, an old one vanishes. This is a *bifurcation*. It explains why animals show abrupt behavioural jumps and abrupt neural-population reorganisation when a rule shifts — exactly the opposite of the smooth curves we see during gradient descent.

**Tool 2: Plasticity that runs on many timescales at once.** Synapses do not have one learning rate. They have a hierarchy: milliseconds of short-term facilitation, seconds of behavioural-timescale plasticity, minutes-to-days of LTP, lifetime structural plasticity. The seconds-scale BTSP is especially striking — a hippocampal place cell can write a brand-new field in one shot, when synaptic input happens to coincide with a dendritic plateau potential. And the brain has **metaplasticity** (plasticity of plasticity): how easily a synapse changes depends on its history and on neuromodulators, so important memories are protected while novel events stay malleable.

Putting these together: rapid adaptation in animals = (fast neuromodulator-gated bifurcation in pre-existing dynamics) + (BTSP-level one-shot plasticity to consolidate the change) + (slow CLS-style replay during sleep to integrate it into the cortical schema). The paper's Table 1 lists eight concrete AI translations of these ideas. The paper is not an empirical result; it is a research agenda for neuro-inspired AI.

## Phase 2 — Graduate-level deep dive

### Manifold attractors

A *manifold attractor* is a continuous submanifold $\mathcal{M} \subset \mathbb{R}^n$ of marginally stable equilibria of a dynamical system $\dot{\mathbf{z}} = f(\mathbf{z}; \theta)$:

$$
\mathcal{M} \;=\; \bigl\{ \mathbf{z}^{*} \in \mathbb{R}^{n} \;:\; f(\mathbf{z}^{*}; \theta) = \mathbf{0} \;\text{and}\; \exists\, \mathbf{v}\in T_{\mathbf{z}^{*}}\mathcal{M}\; \text{s.t.}\; \mathbf{v}^{\top}\!\bigl(\nabla f\bigr)\mathbf{v} = 0 \bigr\}.
$$

Off the manifold, the flow is contractive ($\text{Re}(\lambda_i) < 0$ for all transverse Jacobian eigenvalues); on the manifold, one or more eigenvalues are exactly zero, so the system holds its position indefinitely. This is the dynamical substrate for **graded working memory without parameter updates**: input pushes the state to a point on $\mathcal{M}$; in the absence of further input, the memory persists.

The Schmidt et al. (2024) ICLR construction encourages manifold attractors in an RNN via a regulariser that penalises the *minimum* transverse eigenvalue magnitude of the Jacobian in latent space:

$$
\mathcal{L}_{\text{total}} \;=\; \mathcal{L}_{\text{task}} + \lambda_{\text{ma}} \cdot \mathbb{E}_{\mathbf{z}}\Bigl[\,\bigl|\min_{i}\;\mathrm{Re}(\lambda_i(J(\mathbf{z})))\bigr|^{2}\Bigr],
$$

with $J(\mathbf{z}) = \partial f / \partial \mathbf{z}$. The result is RNNs that beat LSTMs on long-range arena tasks while staying interpretable.

### Ghost attractors and slow flows

A *ghost attractor* is a region of state space near a recently destroyed fixed point. After a saddle-node bifurcation that annihilates a stable fixed point $\mathbf{z}^{*}$, the local flow does not vanish but becomes arbitrarily slow:

$$
\dot{z} \;=\; \mu + z^{2}, \qquad \mu = 0^{-} \;\Rightarrow\; \text{two fixed points;}\quad \mu = 0^{+} \;\Rightarrow\; \text{no fixed point but } \dot z \approx \mu \to 0,
$$

with passage time scaling as $T_{\text{passage}} \sim 1/\sqrt{\mu}$. This gives **arbitrarily long, parameter-tunable timescales** without explicit recurrence. Ramping firing rates with adjustable slopes observed in motor and parietal cortex during timing tasks (Mauritz et al.; Wang et al.) match this signature.

A **chain of ghost attractors** $G_1 \to G_2 \to \cdots \to G_K$ connected by heteroclinic orbits implements robust-but-flexible sequences. Sequence elements can be exchanged or re-ordered by changing which ghost is reachable, providing a substrate for *replay of relevant memories without catastrophic interference* (Schmidt et al.) and rapid re-organisation of motor primitives.

### Bifurcations as the substrate for sudden jumps

A bifurcation is a qualitative change in the phase portrait of $\dot{\mathbf{z}} = f(\mathbf{z}; \theta)$ as $\theta$ crosses a critical surface in parameter space. For a saddle-node:

$$
\dot{z} \;=\; \mu - z^{2}, \qquad \mu > 0:\;\text{two fixed points} \;\big(z = \pm\sqrt{\mu}\big);\quad \mu < 0:\;\text{none}.
$$

When the brain operates **near a bifurcation surface**, a small change in a neuromodulator level $\theta$ — e.g. a phasic dopamine increase — can produce a *topological* re-organisation: an attractor that previously stored "rule A" vanishes and a new attractor encoding "rule B" appears. This formally accounts for the change-point coincidence ($r > 0.91$) between neural-population and behavioural transitions observed in Durstewitz et al. 2010 *Neuron* and Karlsson et al. 2012.

### Behavioural-timescale synaptic plasticity (BTSP)

BTSP (Bittner et al. 2017 *Nat. Neurosci.*) is a one-shot, seconds-wide plasticity rule. The Tyulmankov, Yang & Abbott 2022 formalisation can be written as

$$
\Delta w_{ij} \;=\; \eta \cdot K_{\text{BTSP}}(\Delta t) \cdot \phi(w_{ij}) \cdot e_{i} \cdot \mathbb{1}_{[\text{plateau at } j]},
$$

with

$$
K_{\text{BTSP}}(\Delta t) \;=\; \exp\!\Bigl(-\tfrac{|\Delta t|}{\tau_{\text{BTSP}}}\Bigr), \qquad \tau_{\text{BTSP}} \approx 2\!-\!4\text{ s},
$$

where $\Delta t$ is the temporal lag between pre-synaptic input and post-synaptic dendritic plateau, $e_i$ is a pre-synaptic eligibility trace, and $\phi(w_{ij})$ is a *weight-dependent* update factor: LTP if the synapse is initially weak, LTD if it is initially strong. Two time windows — input within seconds *before* or *after* a plateau — both produce updates. Result: one-shot, content-addressable associative memories with much greater capacity and overwrite-robustness than classical Hopfield networks.

### Metaplasticity and dopamine-gated plasticity

A compact metaplasticity model is

$$
\eta_{ij}^{\text{eff}}(t) \;=\; \eta_{0} \cdot g\!\bigl(\Omega_{ij}(t)\bigr) \cdot h\!\bigl(\text{DA}(t), \text{NE}(t), \dots\bigr),
$$

where $\Omega_{ij}(t)$ tracks per-synapse history (e.g. Fisher-information estimate as in EWC), and $h(\cdot)$ is a multiplicative gain set by neuromodulator concentrations. Dopamine, in particular, gates LTP induction in PFC and STP-to-LTP conversion, providing the biological bridge between dynamical bifurcations (Tool 1) and synaptic consolidation (Tool 2).

### Multi-timescale plasticity in continual learning

The Sodhani et al. / Najarro & Risi line implements multi-timescale eligibility traces:

$$
e_{ij}^{(k)}(t+1) \;=\; \alpha_k \, e_{ij}^{(k)}(t) \,+\, (1-\alpha_k) \, x_i(t)\, y_j(t), \qquad k = 1, \dots, K,
$$

with timescales $\tau_k = -1/\log\alpha_k$ spanning ms to hours, and weight update

$$
\Delta w_{ij} \;=\; \sum_{k=1}^{K} \gamma_k \cdot e_{ij}^{(k)} \cdot \delta(t),
$$

where $\delta(t)$ is a global error / reward signal. This permits learning across delays without standard BPTT's window limit.

### Computational mechanism summary

The paper's central architectural prescription, distilled from Table 1 and the body text, is a brain-inspired architecture with five interacting layers:

1. **Recurrent dynamical core** with regularised manifold attractors and ghost-attractor chains for graded memory and tunable timescales.
2. **Neuromodulatory broadcast** that adjusts the dynamical core's parameters and drives it across bifurcation surfaces in response to surprise, reward, novelty (dopamine, NE, ACh, 5-HT).
3. **Multi-timescale plasticity** including a BTSP-style fast one-shot rule for episodic encoding.
4. **Metaplasticity** that protects important weights and is itself modulated by (2).
5. **Complementary-systems wrapper** with a fast hippocampus-like memory and a slow cortex-like consolidator linked by replay (potentially generative).

No current AI system has all five. Building one is the agenda.

## Connections to other papers in this corpus

- **Companion piece to Kudithipudi et al. 2022.** Both are *Nature Machine Intelligence* perspectives, three years apart, with overlapping six-feature scaffolds. Where Kudithipudi (`kudithipudi_2022_lifelong_learning`, this batch) gives the *mechanism catalogue* with biology-to-algorithm mappings, Durstewitz gives the *dynamical-systems and prefrontal-cortex viewpoint* with a stronger emphasis on bifurcations and BTSP. Together they bracket the corpus.
- **Neuromodulation as bifurcation gate.** The paper's framing — dopamine and NE as parameters that move the dynamical system across bifurcation surfaces — connects directly to the gain-modulation papers in this corpus: Rodriguez-Garcia et al. 2026 (`rodriguezgarcia_2026_ne_stability_gap`, this batch), Ferguson & Cardin 2020 (`ferguson_cardin_2020_gain_modulation`, other batch), Wainstein et al. 2025 (`wainstein_2025_gain_perceptual_switches`, other batch), Shine et al. 2021 (`shine_2021_cellular_to_dynamics`, other batch), Doya 2002 (`doya_2002_metalearning_neuromodulation`, other batch).
- **Recurrent dynamics under modulation.** Tsuda et al. 2021 (`tsuda_2021_hypertube_shifts`, other batch) on "shifting activity hypertubes" via neuromodulators in RNNs and Costacurta et al. 2024 (`costacurta_2024_structured_flexibility`, other batch) on structured flexibility via neuromodulation are exactly the kind of "neuromodulator-driven dynamical re-organisation" the paper advocates.
- **Driscoll et al. 2022** (`driscoll_2022_shared_dynamical_motifs`, other batch) on multi-task RNN computation via shared dynamical motifs is the direct empirical instantiation of the "manifold-attractor / ghost-chain" claims here.
- **Hopfield extensions.** The associative-memory framing (BTSP → content-addressable memory) connects to Osman et al. 2024 (`osman_2024_hopfield_arousal`, other batch), Alonso & Krichmar 2023 (`alonso_krichmar_2023_sparse_quantized_hopfield`, other batch), and Tambaş et al. 2025 (`tambas_2025_krotov_hopfield_rbm`, this batch) — all of which extend Hopfield-style energy-based memories with neuromodulatory or sparsity features.
- **Continual-learning algorithm row.** The continual-backprop / partial-reset idea (Dohare et al. 2024) connects to Kolouri et al. 2019 (`kolouri_2019_attention_based_structural_plasticity`, other batch) and to the stability-gap focus of Rodriguez-Garcia et al. 2026.
- **Spiking implementations.** The paper's STDP and neuromorphic-hardware remarks link to the SNN papers in this batch — Espino et al. 2024 (`espino_2024_snn_path_planning`, this batch) and AlKilany & Goodman 2025 (`alkilany_goodman_2025_snn_dynamic_sensory`, this batch).
- **Meta-learning of plasticity.** The "meta-learn the plasticity rule itself" thread maps to Wang et al. 2024 (`wang_2024_neuromodulated_meta_learning`, other batch), Ben-Iwhiwhu et al. 2022 (`ben-iwhiwhu_2022_context_meta_rl`, other batch), Vecoven et al. 2020 (`vecoven_2020_neuromodulation_deep_nets`, other batch), and Mei et al. 2022 (`mei_2022_multiscale_neuromodulation`, other batch).
