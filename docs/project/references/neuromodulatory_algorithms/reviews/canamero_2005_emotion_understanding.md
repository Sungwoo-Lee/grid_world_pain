---
title: "Emotion understanding from the perspective of autonomous robots research"
authors: ["Lola Cañamero"]
year: 2005
venue: "Neural Networks 18 (2005) 445–455 (Special Issue), Elsevier"
slug: canamero_2005_emotion_understanding
source_pdf: "sources/Cañamero 2005 - Emotion understanding from the perspective of autonomous robots research.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This 2005 article is a **position paper / review** by Lola Cañamero in the Neural Networks special issue on emotion. It steps back from any single robot to ask the methodological question: *what can the field of autonomous-robot emotion modeling actually contribute to the science of emotion?* Cañamero argues that physical robots offer a way to *operationalize* emotion theories — to take fuzzy psychological constructs (fear, anger, pleasure, valence, appraisal) and force them into mechanisms that have to actually run inside a body interacting with a real environment. The paper distinguishes two design strategies. The **designed approach** wires emotion machinery — homeostatic motivations, basic-emotion modules, simulated hormones — directly into the architecture (her own 1997 Abbott creature is the running example). The **emergent approach** (Braitenberg vehicles, Pfeifer's fungus-eaters) implements no explicit "emotion components" and lets emotion-like behavior arise from the agent-environment interaction. Cañamero argues the two approaches answer different questions and should be used in combination. She then catalogues open *bottlenecks* — what emotion theories should drive what mechanisms, how to ground emotional value in the robot's own needs (not the designer's terms), how to "dissolve" the symbolic-vs-embodied split, and how to *measure* whether adding emotions actually improves a robot's task performance. The paper is important to the corpus because (i) it formalizes the **neuromodulatory / hormonal modulation view of emotion** as one of the two principled mechanism families (the other being neural-circuit models), and (ii) it explicitly enumerates how Cañamero-school **synthetic physiology + hormone-modulated perception + motivational priority shifts** are meant to operationalize emotion in autonomous robots.

## Section-ordered backbone

### Abstract
Discusses how modeling emotions in autonomous robots can advance the understanding of human emotions, both as "sited in the brain" and as used in interactions with the environment. Two contributions are claimed: (a) robots as tools and "virtual laboratories" to test theories of human emotion; (b) modeling that forces conceptual clarification and operationalization. The field is in its infancy but already shares conceptual problems with the other affective sciences; multidisciplinary effort is required.

### Section 1 — Introduction
Frames Affective Computing (citing Picard 1997) and the engineering case for affect in robots: emotion expression makes social robots more believable; emotion-like mechanisms can improve adaptation in dynamic, unpredictable environments. Restricts the paper to autonomous robots (rather than computer simulations), because embodiment, noisy/limited real-world sensing, and real-time environmental dynamics impose constraints that purely-software emotion models do not face. Lists the four contributions the area makes: human-perception studies, theory-testing tools, a synthetic approach, and forced operationalization.

### Section 2 — Bottlenecks
Lists the open methodological problems the field faces along four axes: about **models** (scope, definitions, combinability of competing emotion theories); about **emotion machinery** (mechanisms underlying emotion, cognition, and action; how to integrate computational mechanisms from different traditions); about **applications** (which aspects of emotions are meaningful to implement in robots); about **assessment** (how to quantify emotional states, both from the architecture and from behavior). These are not pre-conditions to be solved before modeling — Cañamero advocates a two-way interplay where modeling can even reformulate theoretical questions.

### Section 3 — Approaches to modeling emotion in autonomous robots
Splits the survey along the time-scale of adaptation: **action-selection** (short term, fast decisions) and **learning** (long term).

**3.1 Emotion in action selection.** Two sub-styles.

*3.1.1 Designed emotions for behavior control.* Emotions are explicit, integral parts of the architecture. The exemplar is Cañamero (1997, i.e. `canamero_1997_motivations_emotions.md`): a synthetic physiology of homeostatically-controlled survival variables (blood sugar, vascular volume, energy, …), simulated hormones that can alter those variables, a set of motivations (aggression, cold, curiosity, fatigue, hunger, self-protection, thirst, warmth) activated by deficit/excess errors, a behavior repertoire whose execution modifies the variables, and a set of basic emotions (anger, boredom, fear, happiness, interest, sadness) that release hormones when active. Under normal circumstances behavior is driven by motivations; emotions are a *second-order* control mechanism that runs in parallel, "monitoring" the environment and altering motivational priorities and behavior execution through hormone-mediated effects on physiology, arousal, attention, and perception. Velásquez 1998 (action selection + learning) and Breazeal 2002 (social robot) are flagged as closely related architectures. A footnote notes that Cañamero's simulated-robot architecture is being adapted to real robots (Avila-García & Cañamero 2004, 2005).

*3.1.2 Emergent emotions.* Typically adopted in artificial-life models. In the most extreme form (Braitenberg's *Vehicles* 1984), the architecture has no emotion components and the apparent affect ("love", "fear", "aggression", "cowardice", "curiosity") is a side effect of sensor-motor wiring + agent-environment interaction. Pfeifer's *Fungus Eaters* (1993) is the second classic example. A second strand within the emergent approach explores how emotion-typed behavior can emerge from simulated "hormones" or "neurohormones" that modulate an underlying control architecture (Avila-García & Cañamero 2004, 2005; French & Cañamero 2005; Neal & Timmis 2003), most often around the fight/flight axis. These architectures combine homeostatic control with neural networks at varying abstraction levels.

**3.2 Learning.** Reinforcement-learning architectures (Gadanho & Hallam 2001; Ventura et al. 2001) and ethologically-grounded learning architectures (Blumberg 1996 — Silas T. Dog; Velásquez 1998 — Yuppy with fear conditioning via an associative network) are reviewed. The key problem is making the reward signal *truly meaningful* to the robot rather than externally supplied. Solutions cited use pleasure / pain signals grounded in internal homeostatic well-being (Andry et al. 2001; Cos-Aguilera, Cañamero & Hayes 2003; Lahnstein 2005) — these are the seeds of the affordance- and pleasure-grounded reward formulations that later show up in `cos_2010_affordances_consummatory.md` and `lewis_canamero_2016_hedonic_pleasure.md`. More neuroscience-inspired emotion-learning models (Balkenius & Morén 2001, Morén 2002 — explicit amygdala / thalamus / sensory cortex / orbitofrontal networks for fear conditioning) exist but have not yet been implemented in robots.

**3.3 Memory.** Selective memory inspired by mood-congruent recall and autobiographic memory (Araujo 1994; LeDoux 1989, 1996) is flagged as one of the few ways to address cognitive-overload / recall-time problems in autonomous robots. Largely unimplemented.

### Section 4 — Shared conceptual problems

**4.1 Mechanisms underlying emotion-in-cognition.** Distinguishes "circuit" / "adaptational" models (Panksepp 1998; Rolls 1999; LeDoux 1996) that postulate specific *emotion centers* / *neural circuits* from "peripheral feedback" / *neuromodulation* models (Damasio 1999; Fellous 1999, 2004) that view emotions as *dynamical patterns of neuromodulations* affecting brain areas across all functional levels. The two frames give rise to two implementation styles: explicit emotion-circuit networks vs. simulated-hormone modulation of a control architecture. Cañamero positions her own work in the latter family.

**4.2 Emotion elicitors.** Cites Izard's (1993) four elicitors — *neural/neurochemical*, *sensorimotor*, *motivational*, and *cognitive* — and notes that very few robotic systems (Velásquez 1996 among the exceptions) attempt all four simultaneously. Appraisal theories (Scherer 2001; Smith 2004) are promising but suffer from the gap between theory abstraction and implementation choices.

**4.3 Emotions as cognitive modes.** Asks a long list of open questions about whether and how an architecture can support an *integrated* perceiving-assessing-prioritizing mode rather than a single scalar; emphasizes the synchronization problem across subsystems, the fast/slow-pathway integration, and the absence of a robotic answer to "what counts as high-impact information for a robot."

**4.4 Emotions, value systems, motivation, and action.** Enumerates four functional roles emotions play in autonomous robots: (i) being tied to general concerns rather than specific responses, hence enabling richer behavior; (ii) acting as a second-order control monitoring internal/external environment for threats; (iii) modifying/amplifying motivation by reshuffling priorities; (iv) constituting value systems that bias goal selection. The link must be grounded in an internal value system to distinguish emotion from mere cognition. Quantitative performance indicators (Avila-García & Cañamero 2002; Avila-García, Cañamero & te Boekhorst 2003 — drawing on viability theory and ethology) are flagged as needed.

### Section 5 — Challenges and goals for future research

**5.1 Origins and grounding problem.** Most synthetic emotion systems hardwire emotion building blocks labeled after psychology terms; this creates two dangers — over-attribution by users (e.g. attributing feelings, consciousness) and lack of grounding (the components exist because they mean something to the designer, not the robot). The two countermeasures are (a) the emergent approach (avoids over-attribution by removing explicit emotion components) and (b) developmental and evolutionary models in which emotional systems grow through agent-environment interaction over individual lifetime or species evolution.

**5.2 Dissolving the mind-body problem.** Symbolic AI models emotion via rule-based introspection-driven systems; embodied AI uses dynamical systems / neural networks / behavior-based robotics for lower-level aspects. Current "complete" architectures are hybrids (Gratch & Marsella 2004; Petta 2003) — Cañamero regards these as a "pineal-gland" placeholder rather than a true integration and proposes using emotion to *synchronize* multiple cognitive/behavioral subsystems as a path toward genuine integration.

**5.3 Untangling the knot of cognition — emotion and intelligence.** Warns against using the post-Damasio / post-LeDoux consensus ("emotion is essential to intelligence") as an unquestioned premise. Demands mechanism-by-mechanism investigation of how emotion modulates cognition and action, plus parallel modeling of self / bodily-self / autobiographic-self / social-motivation / simple empathy.

**5.4 Measuring progress.** Calls for systematic with-vs-without-emotion control experiments and the development of quantitative performance indicators / testbeds / scenarios for emotion-enabled robots.

### Section 6 — Conclusion
Reiterates the two contributions (virtual lab + forced operationalization), restates the early-stage status of the field, and invites neuroscientists, psychologists, and philosophers to collaborate.

## Phase 1 — undergraduate-level synthesis

**The plain question.** If you give a robot something you call an "emotion," is it actually doing emotion the way humans and animals do, or is it just a metaphor? And if it is doing something useful, how can the robot help us understand human emotion?

**The answer the paper offers.** Cañamero argues that there are two ways to put emotion into a robot. Either you *design it in* — wire up an explicit synthetic body with hunger, thirst, fatigue variables, attach simulated hormones that get released when the body is doing well or badly, and let those hormones bias what the robot pays attention to, what motor action it picks, and how strongly it acts. Or you *don't design it in* — you make a much simpler robot with no "emotion box" at all, and you discover that an outside observer will still read its behavior as fearful, curious, or aggressive depending on the situation. The two strategies answer different questions: the designed approach is good at testing what *mechanisms* implement emotion-in-cognition; the emergent approach is good at warning us not to over-attribute. Cañamero recommends combining them.

**The mechanism family she puts forward.** Her own architecture, originally from `canamero_1997_motivations_emotions.md`, is a worked example of the *neuromodulation* family of emotion theories (Fellous 1999, Damasio 1999): no dedicated emotion circuit, just simulated hormones (adrenaline, endorphine, dopamine) released by emotion agents and acting on perception (e.g. raising the ART vigilance threshold, so the world is categorized more or less finely), on motivational priorities (e.g. fear bumps self-protection above hunger), and on behavior execution (e.g. anger increases motor intensity). This is the prototype that the rest of the Cañamero corpus refines.

**Setup and verdict.** No new experiment is run — the paper is a methodological survey. The verdict is: the field is in its infancy, but the *operationalization discipline* it imposes is already useful, and a multidisciplinary effort with emotion theorists / neuroscientists / psychologists is the way to make sound progress.

## Phase 2 — graduate-level deep dive

There are no original equations in this paper. The technical content is the formalization of the *neuromodulation vs. circuit* dichotomy and the four-role decomposition of emotion in autonomous-robot architectures. I therefore reproduce, in compact mathematical form, the architecture Cañamero invokes as her running example and the mechanism statements she makes verbally — using the notation of the 1997 paper, since the 2005 paper does not redefine it.

### The neuromodulation view as a formal control architecture

Let the robot have a body state vector $\mathbf{x}(t) \in \mathbb{R}^n$ (homeostatic variables — blood sugar, vascular volume, energy, temperature, …), each variable with a *set point* $\bar{x}_i$ and tolerance $\Delta_i$. For each motivation $m$, the **drive** is

$$
d_m(t) = \big| x_{i(m)}(t) - \bar{x}_{i(m)} \big| \cdot \mathbb{1}\big\{ \text{sign of deviation matches } m \big\}.
$$

Let the **emotion state** be a vector $\mathbf{a}^{\mathcal{E}}(t) \in [0,1]^{|\mathcal{E}|}$ with entries $a_e(t)$ giving each emotion's activation. In the *neuromodulation* family that Cañamero positions herself in, emotions act on the rest of the architecture through a vector of simulated hormones

$$
\mathbf{h}(t) = \sum_{e \in \mathcal{E}} a_e(t) \cdot \boldsymbol{\alpha}_e, \qquad \boldsymbol{\alpha}_e \in \mathbb{R}^{|\mathcal{H}|},
$$

where $\boldsymbol{\alpha}_e$ is the *release profile* of emotion $e$ across the hormone alphabet $\mathcal{H} = \{\text{adrenaline, dopamine, endorphine, …}\}$. The role of $\mathbf{h}$ is to **modulate** four functional surfaces of the architecture — perception, attention, motivational priorities, and behavior execution. Schematically:

#### (i) Perceptual modulation
The vigilance threshold of an ART recognizer (or, equivalently, the gain of any perceptual classifier) is set as

$$
\rho(t) = \rho_0 + \boldsymbol{\beta}^\top \mathbf{h}(t),
$$

so high-arousal hormones raise $\rho$ (finer categorization, perceptual sharpening). This is the "emotion modulates perception" channel Cañamero highlights in §4.1 as one of the principal mechanisms in the neuromodulation family.

#### (ii) Motivational priority modulation
The activation of motivation $m$ becomes

$$
a_m(t) = w_m \cdot d_m(t) + \boldsymbol{\gamma}_m^\top \mathbf{h}(t),
$$

with $\boldsymbol{\gamma}_m$ encoding emotion-driven amplification. In particular, fear amplifies self-protection beyond what its raw deviation $d_{\text{self-protection}}$ would dictate. The winner-takes-all selection $m^* = \arg\max_m a_m(t)$ thus depends on emotion through $\mathbf{h}$.

#### (iii) Behavior execution intensity
The motor intensity of the selected behavior $b^*$ is

$$
I_{b^*}(t) = I_0 \cdot \big( 1 + \boldsymbol{\delta}_{b^*}^\top \mathbf{h}(t) \big),
$$

so anger-released glutamate / adrenaline boosts motor strength — Cañamero's specific example.

#### (iv) Perceived bodily state
The interoceptive readout of $\mathbf{x}$ is filtered through hormones:

$$
\tilde{x}_i(t) = x_i(t) - \kappa_i^\top \mathbf{h}(t),
$$

so endorphine reduces *perceived* pain $\tilde{x}_{\text{pain}}$ without changing the underlying nociceptive variable $x_{\text{pain}}$. This is the mechanism that lets a happy/euphoric state attenuate the fear/withdrawal loop.

The four surfaces (i)–(iv) together define what Cañamero calls "emotions as **cognitive modes**" in §4.3: a single hormone vector $\mathbf{h}$ simultaneously shifts perception, motivation, action, and self-perception.

### Why this is the *neuromodulation* family and not the *circuit* family

In the circuit family (Panksepp 1998; Rolls 1999; Balkenius & Morén 2001; Morén 2002), the architecture explicitly instantiates network modules labelled "amygdala", "thalamus", "sensory cortex", "orbitofrontal cortex", and emotion is computed as a function of the activity of those modules. Schematically:

$$
\mathbf{a}^{\mathcal{E}}(t) = f_{\text{circuit}}\big( \mathbf{z}_{\text{thal}}, \mathbf{z}_{\text{amyg}}, \mathbf{z}_{\text{cortex}}, \ldots \big),
$$

with task-relevant losses (e.g. fear conditioning) applied to specific modules. In the neuromodulation family, no such modules exist; emotion is a distributed property of the *modulation pattern* $\mathbf{h}(t)$ that biases the underlying control surfaces. Cañamero argues both are needed: circuit models are good for studying specific emotion subsystems in specific cognitive skills; neuromodulation models are good for integrated, global emotional modes across all subsystems.

### Performance indicators (gestured at, not formalized)

Cañamero alludes to *viability theory* and *ethology* as sources of performance indicators (Ashby 1952; Maes 1995; Tyrrell 1993; Avila-García & Cañamero 2002, 2003). The implicit performance functional is *time alive* — a *survival horizon*:

$$
T_{\text{survive}} = \inf \big\{ t : \mathbf{x}(t) \notin \mathcal{V} \big\},
$$

where $\mathcal{V}$ is the viability set (the box defined by $|x_i - \bar{x}_i| \le \Delta_i^{\max}$ for all critical variables $i$). This is the survival-step measure that the project at hand uses — the 2005 paper makes the lineage explicit by citing Ashby's *Design for a Brain* (1952) as the source of the viability framing.

## Connections

- **Direct precursor.** `canamero_1997_motivations_emotions.md` — the 1997 Abbott / Gridland architecture is the recurring case study in §3.1.1 and the implicit reference for every "neuromodulation-style architecture" mention.
- **Reward / pleasure-grounded learning thread.** Cited works Cos-Aguilera, Cañamero & Hayes 2003 and Lahnstein 2005 are the proximal ancestors of `cos_2010_affordances_consummatory.md` (affordance-learning under a motivation-driven reward signal) and `lewis_canamero_2016_hedonic_pleasure.md` (rethinking reward as hedonic quality decoupled from need).
- **Affect-modulated behavior thread.** Avila-García & Cañamero 2004/2005, cited as the real-robot adaptation of the 1997 architecture, lead directly into `blanchard_canamero_2006_affect_modulated.md` — which uses the same framework to explore stability / exploration / exploitation / imitation trade-offs.
- **Long-term adaptation / hormones thread.** The paper flags hormone modulation of underlying control architectures as a promising direction; this is the thread that `lones_canamero_2013_epigenetic_hormones.md` and `lones_2018_hormone_epigenetic.md` later operationalize as an *epigenetic* lifetime-adaptation mechanism.
- **Cross-corpus links (likely handled in other batches).** Reviews of the broader neuromodulation-as-control program — e.g. Cox & Krichmar 2009, Avery & Krichmar 2017, Krichmar 2013 — will naturally cite this 2005 paper as the canonical articulation of the neuromodulation-family stance vs. the circuit-family stance; the curator should cross-link.
