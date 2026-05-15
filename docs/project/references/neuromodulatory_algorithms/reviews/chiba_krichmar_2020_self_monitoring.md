---
title: "Neurobiologically Inspired Self-Monitoring Systems"
authors: ["Andrea A. Chiba", "Jeffrey L. Krichmar"]
year: 2020
venue: "Proceedings of the IEEE, advance online (early access, 2020)"
slug: chiba_krichmar_2020_self_monitoring
source_pdf: "sources/Chiba and Krichmar 2020 - Neurobiologically Inspired Self-Monitoring Systems.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This is a **survey-style perspective paper** in *Proceedings of the IEEE* that lays out a three-level architecture for **self-monitoring** in autonomous systems, drawing the design tightly from how the mammalian brain monitors itself. The authors — Chiba (an experimental neuroscientist who studies cholinergic attention circuits in rats) and Krichmar (a computational neurorobotics lead) — argue that autonomous systems (robots, edge devices, self-driving cars, IoT) should not just *act*; they should *monitor their own state* at three nested levels, like the nervous system does.

The three levels:

1. **Low level — sensory and motor primitives**. Reflexes, motor efference copy, smart sensors, central pattern generators. Innate values ("good taste" / "bad taste") wired in from the start so the agent never has zero data. Maps onto Braitenberg vehicles, subsumption architecture, the Darwin series of brain-based devices.
2. **Intermediate level — homeostasis and allostasis**. The autonomic nervous system maintains internal set-points (temperature, hunger, thirst, heart rate); **allostasis** is the *predictive* version — the set-points themselves shift based on context to avoid future deficit. Maps onto self-modelling and damage-recovery in robots (the Cully et al. 2015 hexapod-with-a-broken-leg story is the canonical example). **Neuromodulators** sit at this level as value systems that signal *context-relevant* significance and trigger learning. **Interoception** — the *sense of internal state* (heart rate, gut state, breathing) — is named explicitly as the entry-point for "feelings" and "self-awareness" via the anterior insula and amygdala.
3. **High level — cognitive control**. Predictive coding, attention systems, learning and memory, affective behaviour. Schemas (Hwu and Krichmar 2019) allow contextual reuse of memory. Hippocampal place / grid cells inspire SLAM. Affective signalling (pupil dilation, the Epi humanoid) and rat–robot interaction (PiRat) extend self-monitoring to *other*-monitoring.

The take-away the authors emphasise — and the most actionable claim for the present project — is that **interoception and homeostasis sit at the *intermediate* level, with neuromodulators as the bridge to cognition**. They explicitly cite anterior insula as the integrative interoceptive hub for emotion and self-awareness, link this to the Cañamero-school homeostatic-robot programme via citations 17 and 18 (Craig 2009; Damasio 2003), and argue that neuromodulators are the *information-carrying* signal that tells the cognitive level *which* internal state is currently most relevant. This is a unification statement: it places the Krichmar four-modulator architecture and the Cañamero homeostatic-architecture under one umbrella.

## Section-ordered backbone

### Abstract
The article explores neurobiological principles for systems requiring **self-preservation, adaptive control, and contextual awareness**. Three levels: (i) low-level control (sensor processing + motor reflexes), (ii) intermediate-level homeostasis / allostasis / set-point prediction, (iii) high-level cognitive planning and prediction. Information flows up and down between levels. Examples are drawn from neuroscience, AI, machine learning, and robotics.

### I. Introduction
Neurobiology has a long tradition of inspiring engineering — neural networks from synaptic anatomy, RL from animal learning, SLAM from rodent navigation circuits. The article uses a three-level roadmap (Fig. 1): blue = low-level sensory + motor primitives; green = homeostasis + allostasis + self-monitoring; orange = high-level planning + cognitive control + attention. Information moves both ways: status reports up, context signals down.

### II. Sensory and Motor Primitives

**Reflexive behaviour.** Innate behaviours / reflexes are minimum competence. Braitenberg (1986) showed how sensor-to-actuator crossover (contralateral wiring) plus excitatory / inhibitory choice generates a small zoo of reflexive behaviours (orient vs. avoid). The spinal cord and subcortical controllers execute pre-programmed motor primitives; *central pattern generators* (CPGs) arbitrate among them. This mirrors Brooks' subsumption architecture.

**Innate values and preferences.** Organisms have built-in good/bad signals (e.g., gustatory preference for sugar; nociceptive avoidance of pain). The Darwin series of brain-based devices (Edelman 2007) embedded such innate values via "metal-tasting" sensors; this seed-data lets later learning bootstrap. Field robots, edge devices, IoT all need analogous built-in priors to survive their first day.

### III. Homeostasis and Allostasis

Homeostasis = stability through error correction toward fixed set-points. Allostasis = *predictive* stability — set-points themselves are adjusted in anticipation of expected future load (Schulkin and Sterling 2019). When predictions fail, error signals are sent from intermediate to high-level controllers for re-evaluation.

**System health and self-monitoring.** The autonomic nervous system constantly tracks variables (thermoregulation, hunger, thirst, defence). Allostasis can adapt set-points to current conditions: a stressed human may down-regulate thyroid output, reducing metabolic rate at the cost of some short-term memory function. The benefit of staying functional under chronic load outweighs the short-term cost.

The article then makes its strongest "self-awareness" claim by way of citations to Craig (2009) and Damasio (2003): **internal-state monitoring is a step toward self-awareness**. The proposed neural substrate is the *amygdala* and *insular cortex*, which receive visceral-organ inputs and support **interoception** — the sense of one's internal state. Interoception is named as "fundamentally necessary for emotion regulation and for assessing the state of another being".

**Safety and damage control.** Analogous to SISSY (Self-Integrating and Self-improving Systems, used in aerospace; Bellman 2018): detect fault → self-protect → set minimal acceptable performance. The hypothalamus does exactly this in biology. The Cully et al. (2015) hexapod robot, shown in Fig. 2, is the engineering exemplar: when a leg is damaged, the robot recognises the kinematic anomaly, "imagines" alternative gaits from a stored repertoire, picks the best one, and recovers within minutes.

**Neuromodulation and value systems.** The four-modulator catalogue (DA, 5-HT, NA, ACh) is reproduced here, with one new emphasis: the Seo et al. (2019) finding that under *intense threat* the dorsal Raphe 5-HT neurons undergo a *paradoxical operational switch* — they normally drive freezing under low-threat conditions, but under high-threat conditions they switch to driving active escape. This means modulator function is *state-dependent*, not fixed.

The article also cross-references the Krichmar (2013) anxious/curious neurorobot (`krichmar_2013_neurorobotic_anxiety_curiosity.md`) as Fig. 3, and points at neuromodulation as a remedy for *catastrophic forgetting* in deep nets (citing Hwu and Krichmar 2019, schema networks). NA is highlighted for *one-shot learning* and *task switching*: phasic LC activity can "clear" a memory that is no longer valid and trigger rapid adaptation.

### IV. Cognitive Control

**Predictive control.** Cortical regions (frontal, parietal) implement predictive coding (Clark 2013; Friston 2010 free-energy principle). The brain maintains internal models from motor control to language. Robot exemplars: Carrillo et al. (2006) Segway with a cerebellum-inspired predictive model that converts collision-prone navigation into smooth obstacle-avoidance via optic flow. Hippocampus-inspired models (Krichmar et al. 2005) support SLAM-like navigation.

**Attention systems.** Bottom-up (stimulus-driven, fast, salience-based) vs. top-down (goal-driven, slow, feature-search). Cortical loci: PFC for feature attention, parietal cortex for spatial attention. The basal forebrain has *two* cholinergic pathways — an **incremental** pathway that increases attention to relevant stimuli, and a **decremental** pathway projecting to the hippocampus that *reduces* attention to irrelevant features. Locus coeruleus / NA provides global "scan-the-environment-for-threats" arousal and can rapidly switch attentional focus during emergencies (e.g., the autonomous-car-avoiding-an-accident example). Deep-learning attention mechanisms are noted as an artificial parallel (citation 61; the article was written in 2020 just as Transformer-style attention was becoming dominant).

**Learning and memory.** Brains learn over a lifetime; artificial neural networks suffer **catastrophic forgetting** when re-trained. The hippocampus learns rapidly; consolidation to neocortex happens during sleep / rest (interleaved learning theory). **Schemas** — high-level context structures — allow rapid learning of new items that fit existing structures (the Hwu / Krichmar 2019 schema network, implemented on the Toyota HSR robot for kitchen / breakroom / classroom object retrieval). **Hippocampal place / grid cells** inspired both deep-learning navigation models and competitive SLAM systems.

**Affective behaviour.** Emotions are part of cognition. Pupil dilation and eye colouration on the Lund Epi robot (Fig. 6) is used to convey emotional state — designed to overcome the uncanny valley. Rat–robot social-interaction experiments with PiRat (Fig. 7) show that rats can discriminate "helpful" from "non-helpful" robots and are more likely to free a previously-helpful trapped robot from a cage.

### V. Conclusion
The article ends by tying the three levels together. Autonomic / homeostatic functions can operate below awareness, but can be *brought into awareness* when high-stakes decisions are needed (e.g., feeling that another animal is in pain and needs help). This switching between self-monitoring and self-awareness is energetically costly but socially indispensable. The proposed architecture differs from Sloman / Chrisley H-CogAff in being grounded in *systems neuroscience* rather than philosophy of consciousness. The closing call: deep learning and deep RL would gain from incorporating more realistic neuroanatomy and dynamics, including a three-level self-monitoring architecture.

## Phase 1 — Undergraduate-level synthesis

**The three-tier story.**

*Tier 1 (low-level).* Reflexes and innate preferences. The robot can already do the basics — orient toward the light, avoid the wall, prefer "tasty" objects — without learning anything. This is the spinal cord and gustatory cortex of biology, the Braitenberg vehicle and the subsumption architecture of robotics. The point: you need *seed data* in the system from the start, or the agent has nothing to learn from.

*Tier 2 (intermediate).* Maintaining internal balance. The body has set-points (temperature, hunger, blood pressure) that drift from their ideals as the body lives, and the autonomic nervous system pulls them back. *Allostasis* is the upgraded version: the set-points themselves shift to anticipate future load. (A stressed person down-regulates their thyroid hormone, slowing metabolism, because the body is bracing for a long demanding period.) The four ascending neuromodulators (dopamine, serotonin, acetylcholine, noradrenaline) carry the *value-laden* signals that tell the rest of the brain *which* internal state is currently most urgent. **Interoception** — the brain's sense of the body's interior — happens here, in the amygdala and insular cortex, and the authors emphasise that this is the substrate of feelings and self-awareness.

*Tier 3 (high-level).* Cognition. Predict the future. Plan. Pay attention to what matters. Remember the past. Use those memories to interpret the present. Recognise emotion in yourself and others.

**The key insight for autonomous-systems design.** Most of what robots and deep RL agents do today happens at Tier 1 (reflex policies) and Tier 3 (the neural network). Tier 2 — the *self-monitoring*, *value-system*, *homeostatic* middle layer — is mostly missing. The authors argue this is the most important tier to add, because it provides:
- The reason to keep operating ("survive, stay healthy");
- The signal that tells Tier 3 *which* high-level computation is currently relevant;
- A graceful-degradation mechanism for when Tier 1 reflexes are broken (the Cully hexapod with a missing leg);
- A bridge to social cognition (the rats recognising the helpful PiRat).

**A worked example: anxious vs. curious behaviour.** The Krichmar 2013 neurorobot (cited as Fig. 3 here) is the cleanest demonstration. The same simulated brain, controlling the same iRobot Create, produces *anxious* behaviour (hugging walls, returning to dock) when simulated 5-HT is high, and *curious* behaviour (exploring the centre, investigating a novel object) when simulated DA is high. The behaviour switches when a stressor (a flash of light) is introduced. The point is that the modulator levels are the *Tier-2 self-monitoring variables* that tell the Tier-3 controller which behaviour-set to deploy.

## Phase 2 — Graduate-level deep dive

This is a *perspective / survey* paper rather than a model paper; it contains no original equations. What follows is therefore a graduate-level *consolidation* of the formal claims the article makes by reference to its citations.

### The three-tier architecture, formalised

For a self-monitoring agent with state $\mathbf{x}(t)$ partitioned into exteroceptive state $\mathbf{x}_e$, interoceptive state $\mathbf{x}_i$, and motor state $\mathbf{u}$, the article advocates three controllers:

$$
\mathbf{u}_{\text{reflex}}(t) = \pi_{1}\!\big(\mathbf{x}_e(t)\big) \quad \text{(Tier 1, fast, innate)},
$$

$$
\frac{d \mathbf{x}_i(t)}{d t} = f\!\big(\mathbf{x}_i(t), \mathbf{x}_e(t), \mathbf{u}(t)\big), \quad \boldsymbol{\nu}(t) = g\!\big(\mathbf{x}_i(t), \mathbf{x}_e(t)\big) \quad \text{(Tier 2, homeostasis + neuromodulator } \boldsymbol{\nu}\text{)},
$$

$$
\mathbf{u}_{\text{cognitive}}(t) = \pi_{3}\!\big(\mathbf{x}_e(t), \mathbf{x}_i(t), \boldsymbol{\nu}(t), \mathcal{M}(t)\big) \quad \text{(Tier 3, slow, learned, model-based)},
$$

with $\mathcal{M}(t)$ a learned internal model and $\boldsymbol{\nu}(t) \in \mathbb{R}^4$ the neuromodulator vector (DA, 5-HT, NA, ACh). The final motor command arbitrates among the tiers, with Tier 1 reflexes preempting when latency is critical and Tier 3 cognition gating when planning is possible.

### Allostasis vs. homeostasis, formalised

Homeostasis: a fixed set-point $\mathbf{x}_i^* $ and a negative-feedback controller

$$
\mathbf{u}(t) = -K \left(\mathbf{x}_i(t) - \mathbf{x}_i^{*}\right).
$$

Allostasis: the set-point is **state- and context-dependent**,

$$
\mathbf{x}_i^{*}(t) = h\!\big(\mathbf{x}_e(t), \mathcal{C}(t), \boldsymbol{\nu}(t)\big),
$$

where $\mathcal{C}(t)$ is the agent's context (e.g., "stressed", "fed", "predator nearby"), and the set-point may shift *predictively* before any error is observed (Sterling 2020; Schulkin and Sterling 2019). This is the same structural commitment that **Khan and Cañamero (2022)** ([`khan_canamero_2022_social_buffering.md`](khan_canamero_2022_social_buffering.md)) implements with the OT-modulated stress-tolerance threshold $\theta_{ST,t} = \theta_{ST,0} \cdot (0.5 + \mathrm{OT}_t)$.

### The four-modulator value-system catalogue (re-stated)

The article re-emphasises the catalogue laid out in `avery_krichmar_2017_models_neuromodulation.md` with two refinements:
1. **State-dependent paradoxical switching** of 5-HT: Seo et al. (2019) show that under intense threat, dorsal Raphe 5-HT neurons switch from a *freezing* mode (low-threat default) to an *active escape* mode. The implication for modelling: a modulator's effect is not a fixed parameter but a function of the modulator's own tonic baseline.
2. **NA as one-shot-learning trigger**: Grella et al. (2019) and Wagatsuma et al. (2018) show that phasic LC activity can drive single-trial global remapping in hippocampus, providing the substrate for rapid task switching. This is the empirical evidence for the "network reset" interpretation Bouret and Sara (2005) advocated.

### Interoception as the entry point to "self-awareness"

The article quotes Craig (2009) for the claim that the **anterior insula** integrates visceral / autonomic signals into a unified representation of the body's internal state, and Damasio (2003) for the claim that this representation is the substrate of "feelings". The technical content: **interoception is multi-modal sensory integration of internal-body signals plus their hedonic / affective tag, mediated by the insula-amygdala-cingulate network and gated by the four neuromodulators**.

For the present project, this is the empirical anchor for the claim that an artificial pain signal — to qualify as more than a scalar penalty — must be embedded in a multi-channel internal-state representation, tagged with hedonic valence, and modulated by an analogue of the four neuromodulator system.

### Damage detection and self-modelling

The Cully et al. (2015) hexapod algorithm:
1. Pre-train a Map-Elites-style behavioural repertoire offline, indexing a large set of gait policies $\pi_\theta$ by their behaviour descriptors.
2. After damage, observe a kinematic anomaly: current performance falls below the prior expectation under the current $\pi_\theta$.
3. Use Bayesian optimisation in the behaviour space to select a candidate alternative $\pi_{\theta'}$, test it in 1–2 trials, update the prior, iterate.
4. Convergence to a working compensatory gait typically within minutes.

This is offered by Chiba and Krichmar as the canonical Tier-2 self-monitoring algorithm: detect a system anomaly via *internal* state (kinematic mismatch), then *imagine* (model-based search) alternative actions before testing them. The same template applies to a pain-modulated RL agent in a grid world — if the modulator signals that the current policy is yielding too much pain, the Tier-3 cognitive level should switch the active policy from a pre-trained portfolio rather than re-train from scratch.

### Schemas, contextual reuse, and catastrophic forgetting

Hwu and Krichmar (2019) — `hwu_krichmar_2020_schemas_memory.md` in this corpus — provide the schema-network model that the article cites repeatedly. The structural claim: contextual reuse of memory through schemas is the brain's solution to catastrophic forgetting. A schema is a learned high-level context (a "kitchen", a "classroom") that gates which low-level memory items are retrieved. Adding schemas to a neural network preserves old knowledge across new training, by routing new examples to the schema-appropriate sub-network rather than overwriting shared parameters.

### Why neuromodulation is the bridge

The article's most coherent unifying claim is that **the neuromodulators are the bridge between Tier 2 (interoception / homeostasis) and Tier 3 (cognition)**. Each modulator carries a low-dimensional, value-laden summary of one aspect of the agent's interoceptive / contextual state, and broadcasts it widely to cortical and subcortical circuits. From Tier 3's perspective, the modulator vector $\boldsymbol{\nu}(t)$ is a context vector that gates which cognitive operation runs. From Tier 2's perspective, the modulator vector is the *output* of the homeostatic / allostatic controller — what it tells the rest of the brain about how the body is doing.

This double role is the structural reason that pain — which is unambiguously an interoceptive signal — must enter the cognitive controller *as a modulator-style scalar*, not as a pixel in the observation. The cognitive controller does not need to know the location and intensity of every nociceptor; it needs the *integrated, value-laden* summary that the Tier-2 controller has computed.

### Relevance flag for the present project

This paper is the **architectural manifesto** for the project's pain-modulated RL approach. Five transferable structural commitments:

1. **Three tiers, with neuromodulation as the bridge.** A pain-modulated RL agent in a grid world should have (a) a fast reflex / innate-value layer (already there in basic homeostatic-RL setups), (b) an interoceptive / homeostatic layer producing one or more neuromodulator scalars, (c) a cognitive policy that takes the modulator(s) as a context input.
2. **Interoception is multi-channel.** Pain alone is not enough — the project should plan from the start for a small modulator vector (e.g., pain × hunger × fatigue × novelty) rather than a single scalar, even if early experiments use only one channel.
3. **Allostasis means predictive set-points.** Khan and Cañamero (2022) implements this for the stress threshold. The present project can adopt the same pattern: a pain-tolerance threshold that shifts based on cumulative recent nociceptive history.
4. **The damage-recovery analogue.** A pain-modulated agent should, when pain is high, *not retrain* — it should *switch* policies from a pre-trained repertoire indexed by context. This maps onto the hypernet / FiLM conditional-architecture line of the present project's other reference corpora (`docs/project/references/Hypernetwork/`, `docs/project/references/FiLM/`).
5. **Construct validity for "pain-like" claims goes through interoception.** Any paper claiming a deep RL system has "pain-like" behaviour will face the construct-validity question: does the system have an interoceptive layer that integrates nociceptive signals into a value-laden context vector that gates the cognitive policy? The Chiba-Krichmar framework gives the answer template.

## Connections

- **`cox_krichmar_2009_neuromodulation_robot_controller.md`** — the engineering exemplar cited as the foundational neurorobot work this article surveys.
- **`krichmar_2013_neurorobotic_anxiety_curiosity.md`** — cited as Fig. 3 (the anxious/curious open-field robot); the canonical demonstration of modulator-driven context switching that this article uses as its central example.
- **`avery_krichmar_2017_models_neuromodulation.md`** — the lab's prior review of modulator computational models; this 2020 article is its self-monitoring-architecture successor.
- **`hwu_krichmar_2020_schemas_memory.md`** (in this corpus) — cited as the schema-network model for catastrophic-forgetting avoidance; the Toyota HSR breakroom / classroom example in Fig. 5.
- **`zou_krichmar_2020_neuromodulated_attention.md`** (in this corpus) — cited as the neuromodulated goal-driven attention work; relevant to the article's attention-systems section.
- **`alonso_krichmar_2023_sparse_hopfield.md`**, **`kolouri_2019_attention_structural_plasticity.md`** (in this corpus) — Krichmar-lab successors on continual learning and structural plasticity.
- **Cañamero school papers** (`khan_canamero_2022_social_buffering.md`, `lharidon_canamero_2023_stress_pain.md`, `scarinzi_canamero_2022_affective_interactions.md`) — the parallel European programme on homeostatic / hormonal self-monitoring in robots. Chiba and Krichmar's "interoception is the substrate of self-awareness" claim is in direct sympathy with the Cañamero-school construct-validity argument that affective state must be embodied via interoceptive signals.
- **Craig (2009)** ("How do you feel — now? The anterior insula and human awareness", *Nature Reviews Neuroscience*) — cited as [17]; the empirical anchor for the interoception-and-self-awareness claim.
- **Damasio (2003)** — cited as [18]; the philosophical companion.
- **Friston (2010)** ("The free-energy principle") — cited as [38]; the predictive-coding / active-inference framework the article points at for Tier 3.
- **Yu and Dayan (2005)** ("Uncertainty, neuromodulation, and attention") — cited as [27]; the expected / unexpected uncertainty framework. Shared with `avery_krichmar_2017_models_neuromodulation.md` and the project's separate uncertainty corpus.
- **Cully et al. (2015)** ("Robots that can adapt like animals", *Nature*) — cited as [11] and shown in Fig. 2; the canonical damage-recovery exemplar.
