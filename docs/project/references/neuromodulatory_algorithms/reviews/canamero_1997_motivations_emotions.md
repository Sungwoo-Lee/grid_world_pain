---
title: "Modeling Motivations and Emotions as a Basis for Intelligent Behavior"
authors: ["Dolores Cañamero"]
year: 1997
venue: "Autonomous Agents '97 (Marina del Rey, CA, USA), ACM 0-89791-877-0/97/02"
slug: canamero_1997_motivations_emotions
source_pdf: "sources/Cañamero 1997 - Modeling motivations and emotions as a basis for intelligent behavior.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This 1997 paper is the foundational reference for the long Cañamero line of work on **emotion- and motivation-driven autonomous robots**, and it lays down the action-selection architecture that nearly every later Cañamero / Lones / Lewis / Blanchard paper builds on. Cañamero builds a simulated creature called *Abbott* that lives in a 2-D grid world called *Gridland* and has to stay alive without anyone telling it what to do. Abbott has a body with simulated *physiological variables* — blood sugar, vascular volume (a proxy for hydration), temperature, energy, pain, adrenaline, dopamine, endorphine, heart rate, blood pressure — each with a normal "set point" and a tolerance band. When a variable strays from its set point, an associated *motivation* (hunger, thirst, cold, fatigue, etc.) generates an error signal called a **drive**. Whichever motivation has the highest activation wins control and picks a *behavior* (eat, drink, walk, withdraw, etc.) that is expected to push the offending variable back to its set point. On top of motivations sits a second layer of *emotions* (fear, anger, happiness, sadness, boredom, interest) that **release "hormones"** which (i) modulate the *intensity* of the chosen behavior, (ii) change the **vigilance threshold of the perceptual recognizers** (an ART-1 network) so that the world is categorized more or less finely depending on mood, and (iii) alter the *perceived* bodily state — e.g. an "endorphine" release reduces felt pain. The paper claims this hormone-modulated, homeostatic, multi-motivation architecture, built inside Minsky's *Society of Mind* (SoM) frame, gives a *newborn-level* creature a principled basis for survival-driven decisions and for the later acquisition of more complex cognition.

## Section-ordered backbone

### Abstract
Reports an experiment implementing an autonomous creature in a 2-D Gridland environment, framed inside Minsky's *Society of Mind*. The creature is at a "newborn" developmental stage and is driven by **motivational states** (impulses to action based on bodily needs, expressed in terms of arousal/satiation under an *exploitation principle*) and **basic emotions** (peripheral and cognitive responses to significant events). Physiological parameters are modeled by analogy with control-system variables; emotions **release "hormones"** that modulate the intensity of the selected behavior, enable it, or prevent it, and they also influence attentional and perceptual mechanisms.

### Introduction
Motivates the goal of an autonomous creature that not only survives but also plays, is happy, solves problems, recognizes external events and the consequences of its actions, learns from experience, and "enjoys life" in a 2-D dynamic world. Adopts SoM as the framework — both as the general view that intelligence emerges from many simple agents and as a concrete set of agent types from SoM. Pairs SoM with an "infant developmental" approach: the creature is born with innate spatial / object structure, a motivational/emotional system for rapid decisions, and an affect system that Cañamero, following Tomkins, treats as "the primary blueprint for cognition, decision, and action." Aligns the incremental design ethos with Brooks's subsumption architecture: add agents, never modify existing connections.

### Experimental Setting — Gridland
A 2-D grid where each cell carries a six-element vector — gravity level, occupancy, hardness, brightness, surface, and amount-of-organic-matter. Inhabitants come in three categories: living beings (Abbotts and Enemies, both single-cell dots), food/water sources (single-cell dots), and inanimate blocks of varying shape (lines, triangles, squares, rectangles, circles). Enemies have simple fixed behavior: wander, eat Abbotts, deliver pain on bite. Food/water exhaust on consumption and regenerate at a new random location. Table 1 lists each entity's surface, hardness, brightness, and organic-matter amount, all of which feed into Abbott's sensors.

### Creatures as Societies of Agents
Adopts Minsky's broad definition of an *agent* (any part or process of the mind simple enough to understand on its own). Each agent is implemented as a Lisp object with five slots: `name`, `owner`, `stimulus` (a list of physical features or another agent), `activation` (a real number), and `state` (0 or 1). Enemies have nine agents; Abbott has many more, organized into sensors, recognizers, direction-nemes, maps, effectors, behaviors, managers, motivations, and emotions. The set of physiological variables (Table 2 — adrenaline, blood pressure, blood sugar, dopamine, endorphine, energy, heart rate, pain, respiration rate, temperature, vascular volume) define Abbott's bodily state with explicit initial values, set points, and variability bands.

### Sensing, Perceiving, and Acting
**Sensors** include somatic sensors that read each physiological variable and report deviation from its set point; *tactile sensors* output a 9-bit vector covering eight surrounding cells plus Abbott's own cell, across four tactile features (gravity, occupancy, hardness, organic-amount); a 5×5 visual sensor reports brightness and (squared) distance to obstacles. **Recognizers** are higher-level agents implementing an **ART-1 unsupervised, vector-clustering, competitive-learning network** (Carpenter & Grossberg 1988), output layer fixed at 20 units; chosen because it requires only one training pass per stimulus, solves stability/plasticity, lets the user set the number of categories *a priori* via the *vigilance threshold*, and supports the ability to *forget* infrequent categories. Critically, Cañamero introduces a mechanism that **resets ART forward/backward weights and changes the vigilance threshold under emotional control** — so the same input can be coarsely or finely categorized depending on mood. **Direction-nemes** label each of eight spatial directions plus center and assemble sensor information per spatial region. **Maps** (picture-frame-style) link direction-nemes for tactile and visual modalities (five tactile maps — occupancy, water, food, living-being, block; six visual maps — same five plus a visual-enemy and visual-abbott map).

### Effectors and Behaviors
Abbott has three effectors — a hand, a foot, and a mouth — and three behavior categories: walking, eating, withdrawing. Behaviors implement *goal-achieving systems* (McFarland 1995): each behavior agent carries a preprogrammed incentive stimulus, an effector, and a list of effects on physiological parameters. The behavior runs only if (a) the motivational/emotional state has selected it and (b) its incentive stimulus is being observed; otherwise the motivational system falls back to alternatives or invokes a **manager** (appetitive-style agent — Find, Get, Put, Grasp, Go-toward) that searches for the missing stimulus. Table 3 maps Abbott's behaviors to their stimulus and main physiological effect (Attack → –adrenaline; Drink → +vascular-volume; Eat → +blood-sugar; Play → +endorphine; Rest → +energy; Walk-avoiding-obstacles → +temperature; Withdraw → –pain). Managers run on Minsky's *exploitation principle*: one agency uses another's outputs without knowing how they work.

### Primitive Affects — Motivations and Emotions
Motivations are framed as **homeostatic processes**: a controlled variable, a set point, an error/drive signal, a satiation criterion. Drives serve three functions following Kandel et al. — *directing*, *activating*, *organizing*. Emotions are characterized by an incentive stimulus, an intensity proportional to activation, a list of hormones released, a list of physiological symptoms, and the list of physiological variables they affect. Emotions can be triggered by three orthogonal routes — external events (innate triggers listed in Table 5: fear ← enemy presence, anger ← achievement frustration, happiness ← goal achievement, etc.), general patterns of physiological change (e.g., sustained high level of any variable → anger), and specific patterns of physiological values (e.g., high heart rate + low temperature → fear vs. low heart rate + … → interest).

### Action Selection
The full per-time-step loop is: (1) reset all agent activation; (2) sense internal + external variables and "subliminally" recognize objects and build maps; (3) compute motivations and the effects of the emotional state; **select the motivation with the highest activation**; (4) the active motivation picks the behavior(s) that best satisfy its drive — consummatory if the incentive stimulus is present, appetitive (i.e., a manager) otherwise. Selected emotions affect the loop in two ways: (i) modulate intensity of the chosen behavior — e.g., anger releases glutamate, so angry attack hits harder; (ii) alter perceived bodily state — e.g., happiness releases endorphine, which reduces felt pain.

### Conclusions
The architecture gives a self-sufficient newborn creature able to make rapid choices that satisfy needs, to look for satisfiers when missing, to keep working on a long task while remaining responsive to urgent interrupts, and to suspend an activity to avoid an enemy bite if more important. Future work: let the creature "grow" by learning to control its proto-specialists toward more adult emotions, by acquiring new problem-solving skills, and by adding reward/punishment mechanisms grounded in the affect model; memory agents based on Minsky's K-lines (SoM 8.1) are flagged as the next step.

## Phase 1 — undergraduate-level synthesis

**Key idea.** Give a robot a *body* with internal variables that drift away from healthy values whenever the robot doesn't act. Wire each variable to a *motivation* (hunger ← low blood-sugar, thirst ← low vascular-volume, cold ← low temperature, etc.). At every tick, compute how far each variable is from its target ("the drive"). Whichever motivation is currently in the worst shape wins the right to drive behavior. That motivation picks a behavior that is known to fix it (eat → +blood-sugar, drink → +vascular-volume, walk → +temperature). On top of this layer, run a parallel set of *emotions* — fear, anger, happiness, etc. — that respond to events (enemy spotted → fear, goal achieved → happiness) and to body-wide patterns of physiology. Emotions release simulated "hormones" that change three things: the strength of the current motor action, the gain of the perceptual system (the ART-1 vigilance threshold — how finely the world is categorized), and the *perceived* body state (endorphine release lowers felt pain). The creature can be selfishly hungry, suddenly fearful, briefly happy, and back to thirsty — all without a central planner.

**Experimental setup.** The world is a 2-D grid called Gridland with food sources, water sources, blocks of various shapes, and *Enemies* that bite. Abbott has somatic sensors for every physiological variable, tactile sensors for the eight neighboring cells, and a small visual sensor for a 5×5 field. The author implements it in Harlequin Lucid Common Lisp on a Sparc station; the GUI (Figure 1 in the paper) shows Abbott's grid position, current physiological vector, drive values, and recently selected behavior, and exposes a stepwise run mode for inspecting which agent fires when.

**Result (qualitative).** Abbott displays both goal-oriented and opportunistic behavior. It keeps working on a long task without forgetting other needs — a hungry Abbott will stop to drink if it begins to also be thirsty, or to flee if an Enemy approaches. When an Enemy bites, pain rises, the withdraw behavior is selected, and adrenaline + heart rate respond. When happiness is active (goal achieved), endorphine is released, which both relaxes the creature and reduces its felt pain on subsequent bites. There are no quantitative survival-time tables — the contribution is the *architecture* and the demonstration that hormone-modulated motivations + emotions are a workable basis for behavior selection.

**Concrete instantiation (the table that nails the design choice).** Table 4 spells out the seven core motivations and their drive equations in plain text: Aggression → decrease adrenaline; Cold → increase temperature; Curiosity → increase endorphine; Fatigue → increase energy; Hunger → increase blood sugar; Self-protection → decrease pain; Thirst → increase vascular volume; Warmth → increase temperature. Table 5 maps external triggers to emotions: Fear ← presence of enemy; Anger ← accomplishment of a goal menaced or undone; Happiness ← achievement of a goal; Sadness ← inability to achieve a goal; Boredom ← repetitive activity; Interest ← presence of novel object/event.

## Phase 2 — graduate-level deep dive

### Physiological state and homeostatic drive

Let Abbott's bodily state at time $t$ be the vector

$$
\mathbf{x}(t) = \big[ x_1(t), x_2(t), \ldots, x_n(t) \big]^\top,
$$

with $n = 11$ in the paper (adrenaline, blood-pressure, blood-sugar, dopamine, endorphine, energy, heart-rate, pain, respiration-rate, temperature, vascular-volume). Each variable has a *set point* $\bar{x}_i$ and a tolerated variability $\pm \Delta_i$ (Table 2 in the paper, e.g. blood-sugar: init 30, set point 20, $\Delta = \pm 10$; energy: init 120, set point 100, $\Delta = \pm 50$; pain: init 0, set point 0, $\Delta = \pm 2$).

For each motivation $m \in \mathcal{M}$ controlling variable $i(m)$, the **drive (error signal)** is

$$
d_m(t) = f_m \big( x_{i(m)}(t) - \bar{x}_{i(m)} \big),
$$

where $f_m$ is monotone in the magnitude of the deviation and signed by whichever direction is "bad" for $m$ (hunger fires only when blood-sugar is *below* its set point, fatigue only when energy is below, cold only when temperature is below, etc.). The paper does not state $f_m$ in closed form; the only operational requirement is that $d_m$ rise with deviation and reach zero at the set point.

The **activation** of motivation $m$ is a function of its drive plus the incentive stimulus $s_m$ associated with that motivation, modulated by the current emotional state $\mathbf{e}(t)$. Schematically:

$$
a_m(t) = g_m \big( d_m(t), \, s_m(t), \, \mathbf{e}(t) \big).
$$

Cañamero specifies that emotions *cannot trigger* a motivation outright but can *amplify* its activation: this is the "incentive stimulus can increase the motivation's activation level, but cannot trigger it" clause in the paper.

### Winner-takes-all action selection

Let $\mathcal{B}_m$ be the set of behaviors whose main physiological effect coincides with reducing $d_m$ (consummatory behaviors for $m$, per Table 3 in the paper). The selected motivation is

$$
m^*(t) = \arg\max_{m \in \mathcal{M}} a_m(t),
$$

and the selected behavior is

$$
b^*(t) = \arg\max_{b \in \mathcal{B}_{m^*(t)}} \big[ \mathbb{1}\{ s_b(t) \text{ observed} \} \cdot \text{effect}_b \big],
$$

i.e., the consummatory behavior in $\mathcal{B}_{m^*}$ whose incentive stimulus $s_b$ is *currently observed*. If no such $b$ exists, a **manager** (appetitive agent — Find, Get, Go-toward) is invoked with $s_b$ as its goal. The manager exhibits goal-directed search rather than reactive consumption.

### Behavior intensity and physiological coupling

When behavior $b^*$ runs, every physiological variable $x_i$ is updated as

$$
x_i(t+1) = x_i(t) + \eta_b \cdot \text{effect}_{b,i} \cdot I_{b^*}(t) + \text{noise}_i(t),
$$

where $\text{effect}_{b,i}$ is the signed entry of behavior $b$'s effects vector on variable $i$ (e.g., for the `drinking` behavior, Figure 3 in the paper lists `(+ vascular-volume) (+ adrenaline) (+ blood-sugar) (+ endorphine) (+ energy) (− temperature)`), $\eta_b$ is a per-behavior step size, and $I_{b^*}(t)$ is the *intensity* of the selected behavior. The paper makes the key claim that **emotions set $I$** — i.e., the urge magnitude $d_{m^*}$ chooses *which* behavior, but the emotional state chooses *how strongly* it executes ("an angry creature executes motor actions with more strength, as glutamate is released").

### Hormonal modulation of perception (ART-1 vigilance)

Recognizers in Abbott are ART-1 networks (Carpenter & Grossberg 1988) with a vigilance threshold $\rho$ that controls how similar an incoming pattern must be to an existing prototype before that prototype is reused (rather than a new category opened). For an input vector $\mathbf{I}$ and bottom-up weights $\mathbf{b}_j$, top-down weights $\mathbf{t}_j$, the winning category $J$ satisfies

$$
J = \arg\max_j \frac{\mathbf{b}_j^\top \mathbf{I}}{\beta + \lVert \mathbf{b}_j \rVert_1},
$$

and the **vigilance test** (which decides commitment vs. reset) is

$$
\frac{\lVert \mathbf{t}_J \wedge \mathbf{I} \rVert_1}{\lVert \mathbf{I} \rVert_1} \ \geq \ \rho.
$$

Cañamero's modification is that **$\rho$ is set by the emotional state**:

$$
\rho(t) = \rho_0 + \delta\rho \big( \mathbf{e}(t) \big),
$$

so that an emotion such as fear or interest *raises* $\rho$ (finer categorization, perceptual sharpening), and a confused or low-arousal state *lowers* $\rho$ (coarser categorization — "a tactile block map can become active in the presence of an Enemy, when [Abbott] is confused"). The footnote (footnote 2 in the paper) clarifies that a second parameter — Grossberg's *self-scaling* of forward weights — is computed alongside $\rho$ for classification performance. This is the earliest version of the modulator-changes-perceptual-gain idea that propagates through the entire Cañamero school.

### Emotion-induced hormone release and perceived state

For each emotion $e$ with activation $a_e(t)$, the paper specifies that $e$ releases a list of "hormones" $H_e$ with intensity proportional to $a_e$:

$$
\Delta h_k(t) = \sum_{e \in \mathcal{E}} \alpha_{e,k} \cdot a_e(t), \qquad k \in H_e,
$$

where $h_k$ is the concentration of hormone $k$ (Abbott's hormones include adrenaline, dopamine, endorphine). The released hormones then re-enter the somatic-sensor loop and **change the perceived bodily state**:

$$
\tilde{x}_i(t) = x_i(t) - \kappa_{i,k} \cdot h_k(t),
$$

so for instance happiness $\to$ +endorphine $\to$ lower *perceived* pain $\tilde{x}_{\text{pain}}$, even with the underlying $x_{\text{pain}}$ unchanged. The drive equation $d_m(\tilde{x})$ then uses *perceived* state, not raw state — this is how "an euphoric emotional state reduces the perception of pain" closes the loop.

### Emotion activation: three independent triggers

Per the paper, emotion $e$ becomes active if any of the three following predicates holds:

1. **External-event trigger:** $\text{TRIG}_e^{\text{ext}}(t) = 1$ if a stimulus matching $e$'s innate triggers (Table 5) is observed. E.g., fear: presence of an enemy. Innate at the newborn stage; the paper anticipates memorized triggers later.
2. **General activation pattern:** a sustained abnormally high level of *any* physiological variable activates anger.
3. **Specific physiological signature:** e.g., $(x_{\text{heart-rate}} \text{ high}) \wedge (x_{\text{temperature}} \text{ low}) \Rightarrow \text{fear}$; $(x_{\text{heart-rate}} \text{ low}) \wedge \cdots \Rightarrow \text{interest}$.

A winner-takes-all is then applied to the active emotions in the priority order (1) → (2) → (3): the external-event trigger dominates; if absent, the general pattern decides; if it produces ties, the specific signature breaks them.

### Pseudocode of one tick

```
At each time step t:
  1. Reset all agent activations to 0.
  2. Update physiological state x(t) (driven by world dynamics + last behavior's effects).
  3. Update hormones h(t) from current emotion activations; compute perceived state x_tilde(t).
  4. For each motivation m:
        d_m(t)   = error(x_tilde_{i(m)}(t), set_point_{i(m)})
        a_m(t)   = g_m(d_m, s_m, e(t))
  5. m* = argmax_m a_m(t)                          # winner-takes-all on motivations
  6. Determine emotion(s) e(t) by the 3-trigger cascade above (winner-takes-all).
  7. Set vigilance ρ(t) = ρ0 + δρ(e(t)).
  8. Run recognizers/maps with current ρ.
  9. Pick behavior:
        if exists b in B_{m*} with stimulus observed:
            b* = b           # consummatory
        else:
            b* = manager(s)  # appetitive search for missing stimulus
  10. Execute b* with intensity I = I_b * f_emo(e(t)); update x(t+1).
```

This loop is the prototype that **every later Cañamero-school paper extends**, whether by adding hormone-controlled epigenetic plasticity (Lones & Cañamero 2013; Lones et al. 2018), by replacing the homeostatic-drive scalar with a more careful pleasure/well-being function (Lewis & Cañamero 2016), by overlaying social-buffering modulation (Khan & Cañamero 2022), or by injecting predation- and stress-driven pain modulation (L'Haridon & Cañamero 2023).

## Connections

This paper is the **root** of the corpus's Cañamero-lineage thread:

- **Directly extended by** `canamero_2005_emotion_understanding.md` — Cañamero's reflective review of what emotion-modeling in autonomous-robot research has and hasn't accomplished, with this 1997 architecture as her primary case study.
- **Operationalized as a "affect-modulated behavior" testbed by** `blanchard_canamero_2006_affect_modulated.md` — Blanchard & Cañamero 2006 take the same homeostatic, hormone-modulated motivational architecture and use it to explore the stability/exploration/exploitation/imitation trade-off.
- **Extended with epigenetic / lifetime-adaptation hormones by** `lones_canamero_2013_epigenetic_hormones.md` and `lones_2018_hormone_epigenetic.md` — Lones et al. keep the homeostatic-drive backbone and graft on a hormone-modulated, lifetime developmental layer.
- **Refined on the reward/pleasure side by** `lewis_canamero_2016_hedonic_pleasure.md` — Lewis & Cañamero 2016 argue that simple homeostatic-error reward of the Abbott form is not enough and propose a hedonic-quality term that decouples *pleasure* from *need*.
- **Affordance-learning extension by** `cos_2010_affordances_consummatory.md` — Cos et al. 2010 keep the consummatory/appetitive split from Abbott and add a learned affordance layer so that the manager-style search can be guided by experience.
- **Cited foundation for the broader neuromodulation-as-controller program** in `cox_krichmar_2009_neuromodulation_robot_controller.md` and `avery_krichmar_2017_models_of_neuromodulation.md` (in batches handled by other reviewers). Cañamero's 1997 hormone-modulated action selection is one of the canonical exemplars cited when those works distinguish *neuromodulation* from *standard reactive control*.
