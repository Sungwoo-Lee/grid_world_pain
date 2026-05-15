---
title: "Informing deep neural networks by multiscale principles of neuromodulatory systems"
authors: "Jie Mei, Eilif Muller, Srikanth Ramaswamy"
year: 2022
venue: "Trends in Neurosciences 45(3), 237–250"
slug: "mei_2022_multiscale_neuromod"
source_pdf: "sources/Mei et al. 2022 - Informing deep neural networks by multiscale principles of neuromodulatory systems.pdf"
topic: "neuromodulatory_algorithms"
---

## Plain-English entry point

This is a *review article*, not an empirical paper. It asks: biological brains have a system of chemical messengers — neuromodulators like acetylcholine (ACh), noradrenaline (NA), serotonin (5-HT), and dopamine (DA) — that diffusely tune how individual neurons and whole circuits behave depending on context. They operate across many spatial scales (a single ion channel up to whole brain regions) and many time scales (milliseconds for fast attention shifts up to hours for sleep transitions). Deep neural networks (DNNs), in contrast, are typically static at inference time: weights are frozen, hyperparameters like learning rate are set once. The paper makes a case that DNNs would learn more flexibly — generalising across tasks, adapting on the fly, avoiding catastrophic forgetting (the failure mode where learning task B erases what was learned in task A) — if they incorporated *multiscale* neuromodulation in a principled way.

Why it matters: the paper is a *taxonomy* and *roadmap*. It catalogues what each major neuromodulator does in the brain (DA as the canonical reward prediction error signal; NA as a gain knob trading off exploitation vs. exploration; ACh as a gate on plasticity and a regulator of attention; 5-HT as a temporal-discounting and patience modulator), how they cooperate and compete, and which DNN-side mechanisms have already implemented small slices of these ideas. The headline contribution is a *framework* (Figure 3) that proposes integrating neuromodulation into DNNs at four spatial scales simultaneously: (1) reconfiguration of network hyperparameters (learning rate, dropout); (2) cell-type-specific neuromodulation on subpopulations; (3) neuromodulated scaling and updating of synaptic weights; (4) computations using compartmental (multi-dendrite) neuron models. The authors call this class of systems "neuromodulation-aware DNNs" and frame them as a path toward continual, adaptive learning.

## Section-by-section backbone

### Highlights
Neuromodulators are central to biological learning and adaptive behaviour. They operate on a spectrum of spatio-temporal scales and can act in tandem or opposition. Their phenomenology is mostly absent from DNNs apart from reinforcement-learning analogues; the paper sketches organizing principles and proposes how to bring them in.

### Introduction — Biological neuromodulation and DNNs
Defines neuromodulators as diffuse chemical messengers released via paracrine signalling (en passant axon collaterals → varicosities → local volume diffusion). Key example: in central pattern generators (CPGs), blocking all modulation and reapplying DA or 5-HT alone produces different spiking patterns *in the same neurons*. This is the canonical evidence that neuromodulation *reconfigures* a network without changing its synaptic weights.

### Neuromodulatory systems and their functions
Long-range projections from midbrain/hindbrain/forebrain nuclei release ACh / NA / 5-HT / DA into target regions (hippocampus, neocortex, striatum). Each neuromodulator binds multiple receptor classes with cell-type-specific expression, leading to differential timing and effects. Cooperation: ACh+5-HT for working memory; DA+ACh for motor learning; DA+5-HT for reward processing. Competition: ACh vs 5-HT in prefrontal layer 6 ("attentional tug-of-war"); DA vs 5-HT in vigor-vs-quiescence trade-offs.

### Box 1 — Neuromodulation-inspired DNN models of learning and action selection
**DA / 5-HT and TD error.** In actor–critic RL, the temporal-difference error is

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \tag{I}$$

The dopaminergic system signals $\delta_t$; serotonergic levels regulate the discount factor $\gamma$ via cortico-basal-ganglia loops.

**NA and exploration–exploitation.** Softmax action selection,

$$P(a_i \mid s) = \frac{\exp(\beta \cdot Q(s, a_i))}{\sum_{j=1}^n \exp(\beta \cdot Q(s, a_j))} \tag{II}$$

with $\beta$ controlling exploration vs. exploitation. Tonic locus-coeruleus (LC) activity drives search (low $\beta$); phasic LC activity drives task exploitation (high $\beta$).

### Neuromodulation, adaptive learning, and behavioural flexibility
Detailed neuro-pharmacology: DA tonic firing ≈ motivation, phasic ≈ RPE; transient DA release induces dendritic spine enlargement in striatal neurons within 0.3–2 s, providing a cellular substrate for plasticity. 5-HT regulates *temporal discounting* and *patience* — inhibited 5-HT impairs waiting for long-delayed rewards; activated dorsal raphe 5-HT enhances waiting.

### Context-based state transitions and information processing
NA modulates signal-to-noise ratio in sensory cortices. ACh disables weak connections in neocortex, favouring feedforward over top-down inputs. Two NA theories: **adaptive gain** (phasic = exploit, tonic = explore) and **network reset** (NA triggers arousal/reset on unexpected cues). Both are partial; recent evidence shows NA neurons have *projection-specific* functions.

### Neuromodulation-inspired DNNs
The authors call out three existing DNN implementations:
1. **Vecoven et al. 2020 [125]** — modulate the slope and bias of activation functions via a separate neuromodulatory network conditioned on context.
2. **Miconi et al. (Backpropamine) [126]** — network-computed neuromodulatory signal that gates Hebbian plasticity per connection (DA-like eligibility traces).
3. **Beaulieu et al. 2020 (ANML) [107]** — meta-learning + neuromodulation to overcome catastrophic forgetting on sequential classification tasks.

Plus neurorobotics work (Sporns, Cox & Krichmar, Xing et al., Krichmar 2012/2013) that uses high-level abstractions of DA/5-HT/ACh/NA roles for action selection in physical agents.

### Box 2 — Biological vs. artificial attention
Artificial attention (Transformer-style) approximates biological attention as a learned weighting over input components. The authors caution that artificial attention is application-driven and lacks biological multimodality / context-dependent reweighting.

### A four-scale framework (Figure 3)
This is the paper's central proposal. A neuromodulation-aware DNN integrates four parallel scales:

1. **Network-level hyperparameter reconfiguration** — learning rate, dropout probability, even the activation function become *learnable functions* of internal latent state, environment, reward, novelty.
2. **Cell-type-specific neuromodulation** — separate modulation of excitatory vs. inhibitory subpopulations, enabling disinhibition.
3. **Neuromodulated scaling and update of weights** — local or global scaling of synaptic weights; gating of plasticity.
4. **Compartmental model neurons** — neurons with explicit dendritic compartments where downstream global neuromodulators dynamically gate local error computations.

Performance is evaluated both on *subtask-specific/local* loss (e.g., avoidance of an aversive cue) and *long-term/global* loss (e.g., maze navigation).

### Concluding remarks and outstanding questions
Five open questions: (1) how to implement DA / ACh interactions producing synergistic behaviour; (2) how multi-scale NA parameter updates affect DNN reconfiguration; (3) whether dendritic 5-HT receptor "hot spots" can solve credit assignment; (4) how to combine rewiring + weight updates for state transitions; (5) how to mimic sleep-replay for catastrophic forgetting prevention.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** The brain has a tiered control system. Below the level of "which weights connect to which neurons" there's a fast, diffuse chemical layer that can rapidly retune *how* the network operates: it can change a neuron's gain (the slope of its input–output curve), allow or block weight changes (gate plasticity), redirect attention, switch between exploration and exploitation, and shift global states like sleep vs. wake. Different chemicals do different jobs: dopamine carries the reward-prediction signal that lets the network update which actions paid off; noradrenaline cranks the gain up or down depending on whether you're focused or searching; acetylcholine gates which inputs the network listens to and when weights can change; serotonin governs how patient the agent is and how much it discounts delayed rewards.

The paper argues that today's DNNs ignore almost all of this. They have a fixed learning rate, fixed weights at inference, no concept of "current arousal", no per-cell-type modulation. So when a trained DNN faces a new task, it can either be fine-tuned (slowly, with risk of forgetting old tasks) or fail. The biological brain doesn't have these problems because it has the chemical layer.

**The proposal.** Build "neuromodulation-aware DNNs" that integrate biological neuromodulation at four scales at once: (1) make hyperparameters learnable functions of context; (2) treat excitatory and inhibitory cell populations differently; (3) let a modulator scale weights or gate plasticity on the fly; (4) use compartmental neurons whose dendrites can be independently modulated.

**Worked example.** Existing pieces of this framework already exist in the literature: Vecoven 2020 modulates activation functions (scale 3 in the framework — kind of); Backpropamine gates synaptic plasticity per connection (scale 3 squarely); ANML (Beaulieu et al. 2020) uses meta-learning + activation gating to prevent forgetting; cognitive-robot work uses abstracted DA/5-HT/ACh/NA roles. None of them integrates all four scales at once — that integration is the open challenge.

## Phase 2 — Graduate-level deep dive

### Equation I — DA-modulated TD learning

The temporal-difference error is the discrepancy between predicted and obtained reward:

$$\delta_t = r_{t+1} + \gamma\, V(s_{t+1}) - V(s_t)$$

where $r_{t+1}$ is the reward at time $t+1$, $V(s_t)$ is the value-function estimate at state $s_t$, and $\gamma \in [0,1]$ is the discount factor. The standard claim (Schultz, Dabney et al., Gershman & Uchida) is that midbrain DA neuron firing approximates $\delta_t$; the value function is implemented by direct/indirect inhibitory connections from the striatum to the substantia nigra pars compacta. The authors highlight (citing Doya 2002) that 5-HT *separately* regulates $\gamma$ via cortico-basal-ganglia loops, so DA and 5-HT carry complementary but distinct RL signals.

### Equation II — NA-modulated softmax exploration

$$P(a_i \mid s) = \frac{\exp(\beta\, Q(s, a_i))}{\sum_{j=1}^n \exp(\beta\, Q(s, a_j))}$$

Larger $\beta$ → action distribution concentrates on the highest-value action (exploitation). Smaller $\beta$ → distribution flattens (exploration). The Aston-Jones & Cohen (2005) "adaptive gain" theory identifies $\beta$ with the LC noradrenergic gain: tonic LC firing → low $\beta$ → search; phasic firing → high $\beta$ → exploit.

### The four-scale framework (Figure 3) — formal sketch

**Scale 1 — hyperparameters as functions of state.** Let $\eta$ be the learning rate, $d$ the dropout probability, $\sigma(\cdot)$ the activation function. The framework proposes

$$\eta = f_\eta(\mathbf{m}_t),\qquad d = f_d(\mathbf{m}_t),\qquad \sigma = \sigma_{f_\sigma(\mathbf{m}_t)}$$

where $\mathbf{m}_t$ is a vector of neuromodulatory variables (DA, NA, ACh, 5-HT-analogues) computed from environmental state, behavioural state, and DNN internals. The Vecoven 2020 paper instantiates a slice of this with $\sigma$ depending on $\mathbf{m}$.

**Scale 2 — cell-type-specific modulation.** Partition the network's neurons into populations $\mathcal{P}_1, \ldots, \mathcal{P}_K$ (e.g., excitatory vs. inhibitory). The modulator vector $\mathbf{m}_t$ acts on each population through a separate gating $g_k(\mathbf{m}_t)$, so the effective activation of a neuron $j \in \mathcal{P}_k$ is

$$h_j = g_k(\mathbf{m}_t) \cdot \sigma(W_j x + b_j)$$

Inhibition / disinhibition emerges from $g_{\text{inh}}(\mathbf{m}_t)$ vs. $g_{\text{exc}}(\mathbf{m}_t)$. Tsuda et al. 2021 demonstrate this in recurrent networks.

**Scale 3 — neuromodulated weight scaling and plasticity gating.** A weight update rule like

$$\Delta w_{ij} = \alpha \cdot \mu_{ij}(\mathbf{m}_t) \cdot \text{Hebb}(x_i, x_j) + \nu_{ij}(\mathbf{m}_t) \cdot e_{ij}$$

where $\mu_{ij}(\mathbf{m}_t)$ gates Hebbian plasticity per connection, $\nu_{ij}$ modulates an eligibility trace $e_{ij}$, and both modulators are computed by the neuromodulatory subnetwork. Miconi's Backpropamine implements this for the DA-eligibility-trace case. The biological inspiration: DA, 5-HT, and ACh gate STDP polarity and time windows differently across regions (e.g., ACh reverses LTP to LTD in PFC; NA flips depression to potentiation in V1).

**Scale 4 — compartmental neurons.** Replace point-neuron units with multi-compartment neurons. Local errors are computed at dendritic compartments and gated by region-specific neuromodulatory inputs (cf. Guerguiev et al. 2017 segregated-dendrite credit assignment). Distinct neuromodulators target distinct dendrite zones — e.g., 5-HT 'hot spots' on distal apical dendrites of layer-5 pyramidal cells.

### Multi-objective evaluation

The framework recommends evaluating neuromodulation-aware DNNs on a *composite* loss

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{long-term}} + \lambda_2 \mathcal{L}_{\text{subtask}}$$

with $\mathcal{L}_{\text{long-term}}$ measuring goal-attainment (total reward, navigation success) and $\mathcal{L}_{\text{subtask}}$ measuring local capabilities (avoidance, attention to salient cues). Trained agents are then probed to see *which* neuromodulatory scale drives which behavioural component.

### The "matchmaker" metaphor

The authors describe neuromodulators as "matchmakers between the environment and the brain": they infer behavioural context and broadly adapt multiscale properties (in DNN terms, hyperparameters) to bridge millisecond-scale neural events to slower behavioural signals like reward or aversion. This integration across timescales is what point-process plasticity (STDP) alone cannot achieve; neuromodulation provides the slow timescale.

### Critical gaps the authors flag

1. **Volume vs. synaptic transmission.** Biological neuromodulators act via both modes; their relative role is debated. DNNs typically have only synapse-like (weight-targeted) modulation. The Vecoven-style activation modulation is closer to *volume* signalling (the modulator $z$ broadcasts to all neurons).
2. **Cooperation vs. competition.** No DNN has yet implemented the *opponent* dynamics seen biologically (DA vs. 5-HT, ACh vs. 5-HT).
3. **Sleep / replay.** Biological replay during sleep is a possible solution to catastrophic forgetting. Brain-inspired replay (van de Ven et al. 2020) is a sister direction.

## Connections

This is a survey/review and therefore stitches together many papers in this corpus. It positions Vecoven 2020 and Backpropamine as the closest existing implementations of its four-scale framework, but argues none of them is comprehensive.

- **[vecoven_2020_neuromod_dnn](vecoven_2020_neuromod_dnn.md)** — cited as reference [125]. Vecoven's activation-slope/bias modulation is the principal worked DNN example of "modulate the activation function" (Scale 1, partial).
- **[ben-iwhiwhu_2022_context_meta_rl](beniwhiwhu_2022_context_meta_rl.md)** — published concurrently, not cited; covers the activity-gating slice of Scale 1/2.
- **Backpropamine (Miconi et al. 2019)** — cited as ref [126]; canonical example of Scale 3 (neuromodulated plasticity).
- **ANML (Beaulieu et al. 2020)** — cited as ref [107]; closest example of meta-learning + neuromodulation for continual learning.
- **Tsuda et al. 2021** — cited as ref [132]; exemplar of cell-type-specific modulation in recurrent networks (Scale 2).
- **Doya 2002** — cited as ref [22, 60]; the foundational paper mapping ACh / NA / 5-HT / DA to RL hyperparameters (learning rate, exploration, discount, TD error). Mei et al. extend Doya's framing to DNNs.
- **Shine et al. 2021** — cited as ref [76]; computational link between cellular neuromodulation and large-scale neural dynamics; closely related framework paper.
- **[ferguson_cardin_2020_gain_modulation](ferguson_cardin_2020_gain_modulation.md)** — cortical-gain-modulation review; biological substrate for Scale 1/2.
- **Krichmar (2012/2013), Cox & Krichmar 2009, Xing et al. 2020** — cited as refs [127–131]; neurorobotic implementations of high-level neuromodulatory roles for action selection.
- **Avery & Krichmar 2017** — cited as ref [61]; review of computational neuromodulation models.
- **Botvinick et al. 2020, Wang et al. 2018** — cited as refs [133, 137]; deep RL as neuroscience model; Wang's "prefrontal cortex as meta-RL" is foundational for the meta-RL framing of NMN-style papers.
- **[lee_2024_lifelong_rl](lee_2024_lifelong_rl.md)** and **[wang_2024_neuromod_meta](wang_2024_neuromod_meta.md)** — published after this review, but conceptually fit Scales 1–3.
