---
title: "Design Principles for Neurorobotics"
authors: ["Jeffrey L. Krichmar", "Tiffany J. Hwu"]
year: 2022
venue: "Frontiers in Neurorobotics 16:882518"
slug: krichmar_hwu_2022_design_principles_neurorobotics
source_pdf: "sources/Krichmar and Hwu 2022 - Design Principles for Neurorobotics.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This is a *position paper / mini-review*, not an experimental study. Krichmar and Hwu lay out the design principles they argue every "neurorobot" — a robot whose controller is modeled on real nervous-system architecture — should follow. The starting point is Pfeifer and Bongard's 2006 book *How the Body Shapes the Way We Think*, which framed embodied cognition: clever body mechanics can do a surprising amount of "thinking" by simply offloading work onto physics. Krichmar and Hwu extend Pfeifer & Bongard's principles by grounding them explicitly in neuroscience and by **adding a set of principles specifically about neuromodulation and behavioral trade-offs**.

The paper organizes principles into three families:

1. **Embodiment and reactions — responding to the here and now.** Morphological computation (the body solves problems before the brain has to); cheap design (energy-efficient solutions like passive-dynamic walkers); sensory-motor integration (action and perception form one loop, not two stacks); degeneracy (multiple paths to the same outcome → fault tolerance); multitasking / event-driven processing (parallel, asynchronous brains, not serial sense-think-act).
2. **Adaptive behavior — learning and memory.** Complementary learning systems (fast hippocampus + slow cortex); schemas and contextual memory; value systems (intrinsic motivation, dopaminergic reward); prediction (active inference, free-energy minimization).
3. **Behavioral trade-offs — contextual decision-making regulated by chemicals.** Reward vs punishment (DA vs 5-HT); invigorated vs withdrawn (the curious/anxious axis); expected vs unexpected uncertainty (ACh vs NE, Yu & Dayan 2005); exploration vs exploitation (tonic vs phasic neuromodulation); foraging vs defending (hormones); stress vs calm (glucocorticoids, empathy); social vs solitary (oxytocin).

Each principle is illustrated with real robot experiments — DarwinX (hippocampal place cells), Hwu's HSR schema robot, Xing's patience-modulated navigation, Zou's ACh/NE attention robot, Krichmar's curious-vs-anxious iRobot Create, Cox's CARL robot, Cañamero's hormonal robots, the rat-rescue-robot empathy study. The paper is, in effect, the **citation hub** for the whole Krichmar-lab neuromodulation-for-robotics program: it cites Hwu & Krichmar 2020, Xing 2020, Xing 2022 (forthcoming-then), Zou 2020, Cox & Krichmar 2009, Krichmar 2013, Avery & Krichmar 2017, Chiba & Krichmar 2020, and so on — almost every other paper in this corpus. Doya 2002 is cited as the foundational meta-learning / neuromodulation framework. The paper's argument is that following these principles will make robots more naturalistic, more efficient, and more capable in dynamic real-world environments.

## Section-ordered backbone

### Abstract
Pfeifer & Bongard (2006) put forth an embodied approach to cognition. The authors propose neurorobot design principles inspired by Pfeifer & Bongard but grounded in neuroscience and extended with neuroscience-based principles. Three categories: (1) reactive responses, (2) learning + memory, (3) survival trade-offs.

### 1. Introduction
Neurorobotics is a tool for testing brain theories — the experimenter has full access to the artificial brain. Morphological computation, degeneracy, multi-process event-driven control are framed. Value systems, neuromodulators and hormones regulate the behavioral trade-offs that animals need to survive.

### 2. Embodiment and Reactions
**2.1 Embodiment.** Brains don't work in isolation; the body and peripheral nervous system handle much moment-to-moment behavior (Chiel & Beer 1997). Sensor distribution matches niche (fingertips have more touch receptors).

**2.2 Efficiency through cheap design.** Passive-dynamic walkers (Collins et al. 2005) achieve human-like 0.20 energy consumption (vs Asimo's 3.23). Brains also use cheap design: sparse coding (Olshausen & Field 2004; Beyeler et al. 2019), small-world wiring (Sporns 2010). Brains run on ~20 W (Krichmar et al. 2019).

**2.3 Sensory-motor integration.** Fitzpatrick & Metta (2003): a robot's hand bumping a toy block triggers visual motion detection → figure-ground emerges from action. Motor efference copies feed back to error-check expectations. Most cortex is associational, not purely sensory or motor (Fuster 2004).

**2.4 Degeneracy.** Edelman & Gally (2001). Different structural elements yielding same function. Genetic code, neuron-type diversity. Robot example: **DarwinX** (Krichmar et al. 2005; Fleischer et al. 2007) — modeled hippocampus solving dry Morris maze. Three levels of degeneracy emerged: at the neuronal level (different upstream activity → same place-cell output), at the systems level (vision, whiskers, compass, laser — any subset still localizes), at the behavioral level (9 robots with slightly different initial weights solved the same task in idiosyncratic ways).

**2.5 Multitasking and event-driven processing.** Brain has no central clock. Brooks (1991) / Arkin (1998) behavior-based robotics maps onto subsumption architecture. Prescott et al. (1999) — defense as layered subsumption. Chiba & Krichmar (2020) — self-monitoring systems. Neuromorphic chips (TrueNorth, Loihi) are asynchronous and event-driven (Merolla et al. 2014; Davies et al. 2018).

### 3. Adaptive Behavior — Learning and Memory
**3.1 Learning and memory.** Hippocampus rapidly encodes; long-term consolidation into cortex (CLS, McClelland et al. 1995; Kumaran et al. 2016). Hippocampal indexing theory (Teyler & DiScenna 1986). Schemas (van Kesteren et al. 2012). Tse et al. (2007) rats fast-learn within preexisting schemas. **Hwu & Krichmar 2020** modeled this with CHL + mPFC/HPC indexing + neuromodulators. Robot demo on Toyota HSR (Hwu et al. 2020): retrieve objects in classroom vs break room; novel-item-in-familiar-room learned in one trial; second schema learned without forgetting first; cued to retrieve a banana → robot goes to the break room (not the classroom).

**3.2 Value systems.** Krichmar (2008) — neuromodulators as the brain's value system. **CARL robot** (Cox & Krichmar 2009): green panel = reward, red panel = punishment, neutral colors. Phasic neuromodulation amplified attention to salient panels, learning emerged. **HSR + Zou et al. 2020**: ACh tracked expected uncertainty over four goal actions (eat / work-on-computer / read / say-hi); NE detected unexpected goal switches. Body-decoupled value systems are limited; tying values to internal state (battery, fatigue, hunger) is a frontier. Cully et al. (2015) hexapod learns a damage-recovery policy via internal "imagination" of gaits.

**3.3 Prediction.** Active inference / free-energy minimization (Friston 2010). Model-based RL builds internal models (Solway & Botvinick 2012); model-free is more robust under uncertainty (Renaudo et al. 2015). Murata et al. (2014) humanoid with predictive model gestures reactively and proactively. Tani's group (Tani 2016; Ahmadi & Tani 2019; Chame et al. 2020) — hierarchical CTRNNs with multi-timescale prediction, slow units capture sequence abstractions, fast units capture primitives; mirrors PFC/M1 hierarchy. Chame et al. (2020) — active-inference robots for human-robot interaction.

### 4. Behavioral Trade-offs — Contextual Decision-Making
**4.1 Reward vs punishment.** Schultz et al. (1997) DA = reward prediction error. Daw et al. (2002) — tonic 5-HT tracks punishment rate, tonic DA tracks reward rate; phasic 5-HT may signal future-punishment prediction error. Boureau & Dayan (2011) DA / 5-HT opponent dynamics.

**4.2 Invigorated vs withdrawn.** DA / 5-HT regulate curious novelty-seeking vs withdrawn risk-aversion. Open-field test — high 5-HT mice stay near borders; cocaine (high DA) increases exploration. **Krichmar 2013** iRobot Create in an open-field arena: high simulated 5-HT → wall-following / find-home; high DA → openfield / exploreobject; a flashing-light "stressor" event drove phasic 5-HT; phasic DA → behavior switch.

**4.3 Expected uncertainty vs unexpected uncertainty.** Yu & Dayan (2005) — ACh tracks expected, NE tracks unexpected. **Zou et al. 2020 on HSR**: ACh tracks goal-action probability, NE detects goal switch and resets priors. Naude et al. (2016) — ACh mediates uncertainty seeking; Belkaid & Krichmar (2020) — cholinergic-on-dopamine model of uncertainty seeking.

**4.4 Exploration vs exploitation.** Aston-Jones & Cohen (2005) — tonic NE → exploration; phasic NE → exploitation. **CARL** demonstrates this: tonic = random color exploration; salient color triggers phasic → exploit.

**4.5 Foraging vs defending.** Orexin regulates hunger (Padilla et al. 2016). **Cañamero's group** (Canamero 1997; Lones et al. 2018) — robots with hormonal regulation of battery level + internal temperature. Epigenetic-hormone-monitoring system improves adaptability.

**4.6 Stress vs calm.** Glucocorticoids; Why Zebras Don't Get Ulcers (Sapolsky 2004). Chronic stress damages HPC. **Quinn et al. 2018** — rats remember which robot helped/refused-to-help them and selectively rescue the helper. Implications for rescue / caregiver robotics.

**4.7 Social vs solitary.** Oxytocin and pair-bonding (Young & Wang 2004). Cañamero's group — robot–caregiver attachment with social hormone simulation; balance between asking-for-help and learning-on-its-own.

### 5. Discussion
**5.1 Importance of low-level processes and model organisms.** Insect-visual-system robots (Galluppi et al. 2014; Schoepe et al. 2021); OpenWorm (Sarma et al. 2018); drosophila connectome (Scheffer et al. 2020). Don't always go for the brain — go for the smallest organism whose data is rich enough. Foundation-first; cognition emerges from the embodied + neuromodulator-regulated base.

**5.2 Next steps.** Neurorobotics may address lifelong continual learning, efficient computing, scarce-knowledge regimes, HCI. Neuromorphic hardware + neuromodulatory algorithms → edge robotics. Long-term: blurring conventional/neurorobotic distinction.

### 6. Conclusion
Following these principles yields more naturalistic, capable robots, and provides better models for neuroscience.

## Phase 1 — Undergraduate-level synthesis

**Key idea.** This is a how-to checklist for building robots that act like animals. Pfeifer and Bongard already gave us five principles in 2006, mostly about the body. Krichmar and Hwu say: those are good, but they miss what makes brains *brains* — neuromodulators (the slow, broadcast chemical signals like dopamine, serotonin, acetylcholine, noradrenaline) and the trade-offs they regulate. Real intelligence isn't just "the body solves it" or "the algorithm solves it"; it's "the body + the algorithm + a small set of chemical knobs that let the system change strategy based on context".

**The three categories.**

1. **Embodiment + reactions.** Don't fight physics. A pliable plastic hoop around a soccer-playing robot can "trap" the ball faster than any vision pipeline can react. A passive-dynamic walker uses gravity to walk like a human while using 1/15th the energy of Asimo. Sparse neural codes are the brain's version of cheap design. Multiple, redundant pathways (called *degeneracy*) make the system robust to damage — DarwinX still localizes when half its sensors are lesioned.

2. **Adaptive behavior.** Animals learn fast and don't catastrophically forget. They organize new information into *schemas* — a banana fits into a "kitchen" schema instantly because we know what kitchens contain. Robots should do the same: the HSR robot in Hwu's experiments retrieves objects from a classroom or a break room based on which schema the object belongs to. The hippocampus + prefrontal cortex is one model for how schemas work, and neuromodulators decide *when* to encode quickly versus slowly.

3. **Behavioral trade-offs.** Every meaningful decision is a balance — invigorated vs. withdrawn, social vs. solitary, foraging vs. defending. In the brain, these balances are largely set by tonic levels of neuromodulators and hormones. A robot with simulated serotonin behaves anxiously near a wall; a robot with simulated dopamine eagerly explores the middle of the arena. Add a flashing-light "stressor", and watch serotonin spike and the robot retreat — just like a mouse in an open-field test.

**Concrete worked example: the HSR + ACh/NE attention robot (Zou 2020 referenced in §3.2 and §4.3).** A Toyota Human Support Robot watches a classroom. Four ACh neurons each represent a goal action: "eat", "work-on-computer", "read", "say-hi". An NE neuron is the surprise alarm. The user says "I want to eat". ACh-"eat" rises. Contrastive Excitation Backprop highlights bananas / apples in the scene. The robot grasps an apple. User says "yes". ACh-"eat" rises higher; NE falls. Now the user changes their mind to "work-on-computer" silently. The robot's next attempt hands them a sandwich. User says "no". NE rises. After enough misses, NE crosses threshold → reset → ACh re-equalizes → the robot tries each action with roughly equal probability until ACh-"work-on-computer" rises. The principle: the robot doesn't need to *retrain* anything; it just adjusts a small modulator layer on top of a frozen pre-trained perception backbone.

## Phase 2 — Graduate-level deep dive

This is a survey paper, so the mathematical content is mostly *imported* from the cited primary papers. Below I give the formal versions of the principles that have explicit equations elsewhere in this corpus, plus their conceptual derivations.

### Cheap design — bound on bipedal locomotion energy

Collins et al. (2005, cited as Table 1 in this paper) report per-unit-weight-per-unit-distance energy:

| Agent | Energy consumption |
|---|---|
| Asimo (active control) | 3.23 |
| Cornell biped (passive dynamic) | 0.20 |
| Humans | 0.20 |

The Cornell passive walker matches human efficiency by exploiting gravitational potential energy: each leg is a compound pendulum with natural frequency $\omega_n = \sqrt{g/L_{eff}}$. The walker's controller does **no** stabilization work during swing — gravity does. Walking speed $v \approx \omega_n \cdot a$ where $a$ is stride amplitude. Energy per step is dominated by heel-strike loss $\Delta E \approx \frac{1}{2} m v^2 (1 - \cos(\theta_{step}))$, minimized by small step angles. This is "morphological computation" formalized: a body whose dynamics naturally solve the locomotion ODE.

### Neuromodulators in the value-system framework

Krichmar (2008), the lab's foundational position, names neuromodulators as the brain's value system. The Doya (2002) mapping (covered in detail in [doya_2002_metalearning_neuromodulation](doya_2002_metalearning_neuromodulation.md)) gives:

| Modulator | RL meta-parameter |
|---|---|
| Dopamine (DA) | TD error / reward prediction error |
| Acetylcholine (ACh) | Memory time-constant / learning rate |
| Norepinephrine (NE) | Exploration / inverse softmax temperature |
| Serotonin (5-HT) | Temporal discount factor $\gamma$ |

The TD update rule under this framing:

$$
\delta_t = r_t + \gamma\, V(s_{t+1}) - V(s_t), \qquad V(s_t) \leftarrow V(s_t) + \alpha\,\delta_t.
$$

Each neuromodulator gates one parameter:
- DA encodes $\delta_t$.
- 5-HT sets $\gamma$.
- ACh sets $\alpha$.
- NE sets the softmax temperature in $\pi(a|s) \propto \exp(Q(s,a)/T)$.

This four-way decomposition is the conceptual scaffold for nearly every neuromodulator-as-meta-parameter robot in the lab.

### Expected vs unexpected uncertainty (Yu & Dayan 2005)

In a discrete-goal setting with goals $g \in \{1, \dots, K\}$ and observations $o_t$:

$$
H[g_t \mid o_{1:t}] = \underbrace{H[g_t \mid g_t = g, o_{1:t}]}_{\text{expected uncertainty (ACh)}} + \underbrace{\text{KL}(p(g_t \mid o_{1:t}) \,\|\, p(g_{t-1} \mid o_{1:t-1}))}_{\text{unexpected uncertainty (NE)}}.
$$

ACh tracks the *noise level within a known goal* (expected uncertainty); NE tracks *changes in which goal is active* (unexpected uncertainty). When NE exceeds threshold, a "network reset" (Bouret & Sara 2005) clears the goal prior. The Zou 2020 and Xing 2022 papers in this corpus operationalize this with multiplicative ACh/NE updates and a softmax-and-threshold reset rule.

### Active inference / free-energy minimization (Friston 2010)

The variational free energy:

$$
F = \mathbb{E}_{q(s)}[\log q(s) - \log p(o, s)] = D_{KL}(q(s) \,\|\, p(s)) - \log p(o),
$$

where $q(s)$ is the agent's recognition density over hidden state $s$, $p(o, s)$ is the generative model over observations and states. Minimizing $F$ equivalently maximizes evidence $\log p(o)$ — *active inference* extends this by including action selection that minimizes *expected* future free energy. The Tani group's hierarchical CTRNNs and the Chame et al. (2020) work cited in §3.3 implement this in robot controllers. Although not central to the Krichmar-lab program, the paper lists active inference as one of the predictive-modeling principles a neurorobot should respect.

### Contrastive Hebbian Learning + neuromodulator-gated replay (Hwu & Krichmar 2020)

For the schema-learning HSR robot demo in §3.1, the underlying network is [hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md). Recall:

$$
\Delta W_k = \eta_{CHL}\,\left(\hat x_k \hat x_{k-1}^T - \check x_k \check x_{k-1}^T\right),
$$

with neuromodulator-gated epoch count:

$$
\text{epochs} = e_{default} + \nu \cdot e_{boost}, \qquad \nu = x_{novelty} \cdot x_{familiarity}.
$$

The principle: replay rate is set by the *product* of novelty and familiarity — the network rehearses harder when something new appears inside a familiar schema.

### Krichmar 2013 invigorated-vs-withdrawn robot (§4.2)

A neural network with simulated DA and 5-HT pools drives an iRobot Create. The state-action policy depends on which neuromodulator pool dominates:

- Tonic 5-HT high → state $\in$ {WallFollow, FindHome} (anxious).
- Tonic DA high → state $\in$ {OpenField, ExploreObject} (curious).
- Phasic 5-HT (light event) → transient anxious switch.
- Phasic DA (object event) → transient curious switch.

Behavior time-locked to the light event shows ~60 s of post-stress anxious behavior before recovery, with variance across trials due to differing event onsets. The same architecture, varying tonic levels, lets the experimenter probe behavioral signatures of anxiety, depression, OCD-like states (chronically elevated 5-HT → indefinite wall-following).

### Hormonal regulation (§4.5)

Cañamero / Lones et al. (2018) hormones $h(t)$ obey simple secretion–decay dynamics:

$$
\frac{dh_i}{dt} = -\lambda_i\, h_i + g_i(\text{trigger}_i),
$$

with trigger functions $g_i$ from internal state (battery level, internal temperature) and environment. The robot's behavior is a function of the hormone vector $\mathbf{h}$ — high "hunger hormone" triggers foraging; high "thermoregulation hormone" triggers seeking shelter; the epigenetic layer adapts the secretion rates $\lambda_i, g_i$ over generations.

### Summary table — principle → which Krichmar-lab paper instantiates it

| Principle | Robot / model | Cite |
|---|---|---|
| Morphological computation | Segway soccer trap | Fleischer et al. 2006 |
| Cheap design | Passive-dynamic walker | Collins et al. 2005 |
| Sensory-motor integration | Robot hand bumping toy | Fitzpatrick & Metta 2003 |
| Degeneracy | DarwinX | Krichmar et al. 2005 |
| Multitasking / event-driven | Neuromorphic chips | Merolla 2014; Davies 2018 |
| Learning + memory (schemas) | HSR + CHL | Hwu et al. 2020; Hwu & Krichmar 2020 |
| Value systems (DA) | CARL | Cox & Krichmar 2009 |
| Prediction (multi-timescale) | Tani's CTRNN robots | Tani 2016 |
| Reward vs punishment | Opponent DA/5-HT | Daw et al. 2002 |
| Invigorated vs withdrawn | Krichmar 2013 iRobot Create | Krichmar 2013 |
| Expected vs unexpected uncertainty | HSR + ACh/NE | Zou et al. 2020 |
| Exploration vs exploitation | CARL phasic/tonic | Cox & Krichmar 2009 |
| Foraging vs defending | Cañamero hormonal robots | Lones et al. 2018 |
| Stress vs calm | Rat-rescue robot | Quinn et al. 2018 |
| Social vs solitary | Robot–caregiver attachment | Hiolle et al. 2012 |

## Connections

**Direct references inside this corpus (extensive — this paper is the citation hub):**

- **[hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md)** — §3.1 schema robot. The CHL + mPFC/HPC indexing + neuromodulator architecture is exactly this paper's robot model, embodied on the HSR.
- **[xing_2020_neuromodulated_patience](xing_2020_neuromodulated_patience.md)** — §4 cited implicitly; the 5-HT temporal-discounting principle.
- **[zou_2020_neuromodulated_attention](zou_2020_neuromodulated_attention.md)** — §3.2 and §4.3 cited extensively; the ACh+NE goal-driven attention is the worked example for value systems and expected/unexpected uncertainty.
- **[xing_2022_neuromodulation_rl_environment_changes](xing_2022_neuromodulation_rl_environment_changes.md)** — published the same year; the lifelong RL with ACh+NE extension. Not directly cited (the two papers are concurrent) but explicitly downstream.
- **[chiba_krichmar_2020_self_monitoring](chiba_krichmar_2020_self_monitoring.md)** — §2.5 cited for self-monitoring systems architecture (Figure 3 in this paper is adapted from Chiba & Krichmar 2020).
- **[avery_krichmar_2017_models_neuromodulation](avery_krichmar_2017_models_neuromodulation.md)** — §3.2 and §4 cited for ACh/NE/DA/5-HT interactions.
- **[krichmar_2013_neurorobotic_anxious_curious](krichmar_2013_neurorobotic_anxious_curious.md)** — §4.2 worked example; the iRobot Create open-field experiments.
- **[cox_krichmar_2009_neuromodulation_robot_controller](cox_krichmar_2009_neuromodulation_robot_controller.md)** — §3.2 and §4.4 CARL robot — the explicit value-system and exploration/exploitation example.
- **[canamero_1997_motivations_emotions](canamero_1997_motivations_emotions.md)** — §4.5 cited for hormonal regulation in robots.
- **[blanchard_canamero_2006_affect_modulated](blanchard_canamero_2006_affect_modulated.md)** — implicit; same Cañamero-group lineage.
- **[lones_canamero_2013_epigenetic](lones_canamero_2013_epigenetic.md)** and **lones_2018_hormone_epigenetic** — §4.5 worked example.
- **[doya_2002_metalearning_neuromodulation](doya_2002_metalearning_neuromodulation.md)** — foundational; cited for the four-modulator meta-parameter mapping.

**Forward references in the corpus.** Lee et al. 2024, Vecoven et al. 2020, Ben-Iwhiwhu et al. 2022, Wang et al. 2024, Mei et al. 2022, Tsuda et al. 2021, Wainstein et al. 2025, Costacurta et al. 2024 — later papers in deep-RL / DNN-modulation cluster all sit downstream of the design principles articulated here.

**External anchors.** Pfeifer & Bongard 2006 (embodied cognition framework); Friston 2010 (free energy); Yu & Dayan 2005 (uncertainty + neuromodulation); Krichmar 2008 (neuromodulation as survival framework); Schultz et al. 1997 (DA = TD error); Brooks 1991 (subsumption); Edelman & Gally 2001 (degeneracy); McClelland et al. 1995 (CLS); Tse et al. 2007 (schema rapid learning).
