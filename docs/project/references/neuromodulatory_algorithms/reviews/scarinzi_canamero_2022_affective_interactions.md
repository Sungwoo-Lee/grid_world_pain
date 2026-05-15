---
title: "Toward Affective Interactions: E-Motions and Embodied Artificial Cognitive Systems"
authors: ["Alfonsina Scarinzi", "Lola Cañamero"]
year: 2022
venue: "Frontiers in Psychology, 13:768416 (Opinion article)"
slug: scarinzi_canamero_2022_affective_interactions
source_pdf: "sources/Scarinzi and Cañamero 2022 - Toward affective interactions - E-motions and embodied artificial cognitive systems.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This is a short three-page **opinion article** in *Frontiers in Psychology*. It is **not an empirical study and contains no equations or experiments**. The authors are philosophers / roboticists working in the Cañamero school of affective autonomous robotics. Their question is: *what conditions must an artificial agent satisfy if it is to participate genuinely in an emotional interaction with a human, rather than merely recognising or simulating emotion?*

Their answer relies on a distinction they borrow from Fuchs and Koch (2014) between **"affection"** (the bodily resonance — the felt change in posture, breathing, kinaesthesia — that an emotion-laden situation evokes in the agent) and **"e-motion"** (the *action readiness* that this resonance produces — the disposition to move toward, away from, or against something in the world). Emotion, in their framing, is *not* a label to be classified from facial pixels; it is **the motivational source of movement**, and it exists in a **circular feedback loop** between the agent's body, the partner's body, and the environment's **"affective affordances"** (the likelihood that a feature of the world will elicit emotional states and matching behaviour).

The opinion is that **affective computing has over-invested in emotion *categorisation*** (classifying a human face as "joy" or "anger") and under-invested in giving robots their own bodily resonance and action readiness. Without **organismoid embodiment** (a Ziemke 2001 term: an artificial body with sensorimotor capacities similar enough to a living body that it can be moved emotionally), a robot cannot close the affective loop with a human.

The piece is positioning rather than methodology. For the present project — pain-modulated reinforcement learning in a grid world — its value is conceptual: it argues that *interoceptive / homeostatic state* and *motor readiness* should be co-constitutive, not separately bolted on, if one wants to defend an "emotion-like" interpretation of an agent's behaviour.

## Section-ordered backbone

### Introduction
Emotion is framed as a **"second-order" control mechanism** in behaviour-based robot architectures: it alters motivational priorities and biases action selection for fast adaptation, in line with Cañamero and Gaussier (2005) and Cañamero (2019). The authors then place emotion outside the head: emotion and affectivity are *embodied, situated, distributed in the environment, and subordinated to movement* (Fuchs and Koch, 2014). They introduce the concept of **affective affordances** — environmental properties that make the elicitation of particular emotional states and matching behaviours likely (a garden's path, bench, pool, plant arrangement). On this view, "the sense-maker's mode of being in interaction is joyful" is more accurate than "the sense-maker is joyful", because joy is extended over body, environment, and situation as a whole. The contribution promised is a reflection on what *movement-based* affective interaction between an artificial agent and a human requires, shifting attention away from emotion *categorisation and recognition*.

### Engaging in Embodied Affectivity with a Human Partner
Affective computing typically extracts emotion-labels from movement features (speed, acceleration, posture, facial expression). Spezialetti et al. (2020) is cited for the success of body-motion-based emotion classification. The authors then *invert the relation*: rather than treating movement as a signal *from which* emotion is read out, they treat movement as **constitutive** of emotion. Following de Rivera (1977), they emphasise that agents are "moved to move" — emotion is not the kick, the embrace, or the running-away, but the motivational-affective *source* of those actions. Fuchs and Koch (2014) call this action readiness "**e-motion**", and call the bodily resonance that an affective affordance triggers "**affection**". Critically, the proposed model is one of **circular causality**, not linear belief-desire-action causality (the lion-fear example is rejected as unable to capture the *changing intensity* of emotion within the same situation). In every social encounter, **two cycles of embodied affectivity** run in parallel, one in each partner, and each continuously modifies the other's affordances and resonance. The authors then state their normative requirement on artificial agents: they need **organismoid embodiment** (Ziemke, 2001) — body form and sensorimotor capacity similar enough to a living body that the agent can both have its own resonance and contribute to the partner's resonance. Without this, the agent cannot co-determine the affective interaction. They reiterate (Cañamero 2019) that the emotions implemented in such agents must have their own temporal dynamics and must interact with one another.

### Conclusion
Two design implications, restated in bullet form by the authors: (i) the artificial partner needs **its own "affection"** (bodily resonance) so that it can influence how the human partner *evaluates* the situation and thereby trigger the human's action readiness; (ii) the artificial partner needs **its own "e-motion"** (action readiness) so that it can in turn trigger the human's bodily resonance. Both partners must be able to read the environment's affective affordances. Emotion is then a co-determined property of a shared space.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** A robot that is supposed to be affectively engaging with a person cannot just *classify* the person's emotional expression — it must have its own body whose internal state is moved by the situation, and that movement must in turn shape the person.

**The mechanism.** The authors lift two terms from phenomenology of embodied cognition:
- *Affection* = the bodily resonance an affective situation evokes (heart-rate change, posture change, "I feel the dread tighten my shoulders"). For a robot, this would be a change in its internal homeostatic / motor state.
- *E-motion* = the *readiness to move* that follows from that resonance ("I am ready to flee", "I am ready to embrace"). For a robot, this would be a change in its action-selection bias.

In a human–human affective encounter, these two run in **two simultaneous circular loops** — the human's affordances modify the robot's resonance which modifies the robot's action readiness which modifies the robot's motion which modifies the human's affordances which modifies the human's resonance, etc. There is no clean cause and no clean effect.

**Worked example (from the paper).** A garden contains *affective affordances*: a path to walk, a bench to sit on, plants that crowd close to a pool. The visitor's stroll is **a mosaic of intermittent motion and stillness** that the garden's affordances structure. The emotion "joy" is not a label inside the visitor's head; it is the joint property of garden + body + walk.

**Implication for designing affectively competent robots.** A robot that lacks an internal resonance system that the situation can move, or that lacks the means to act in a way that moves the human, is excluded from the affective loop by construction. So the construct of "organismoid embodiment" (Ziemke 2001) — a robot body and sensors similar enough to a living body — is foundational, not optional.

## Phase 2 — Graduate-level deep dive

This is an opinion article; **it contains no equations and no formal model**. There is therefore nothing to derive. What can usefully be made formal for the present project is the structural commitment the authors lay out, so I render it below in pseudo-formal language without inventing claims the authors do not make.

Let the human partner be $H$ and the artificial partner be $A$, each at time $t$ in a shared environment $E$. The authors assert the existence of, for each partner $X \in \{H, A\}$:

- a state of **bodily resonance** $r_X(t)$ ("affection"), and
- a state of **action readiness** $m_X(t)$ ("e-motion"),

with $E$ carrying a field of **affective affordances** $\alpha(\cdot)$ that depends on the joint configuration of both partners and their behaviour. The structural claim is that these quantities are coupled by **circular causality**, schematically:

$$
\frac{d r_X(t)}{d t} = f_X\!\left(\alpha\!\big(E, m_H(t), m_A(t)\big), \; m_X(t)\right),
$$

$$
\frac{d m_X(t)}{d t} = g_X\!\left(r_X(t), \; m_{\neg X}(t), \; \alpha(\cdot)\right),
$$

i.e. each partner's *resonance* is driven by the environment's affordances (themselves a function of how *both* partners are currently moving), and each partner's *action readiness* is driven by its own resonance plus the partner's motion. The authors explicitly reject any reduction of this to a one-directional belief-desire-action chain of the form $\text{percept} \to \text{belief} \to \text{desire} \to \text{action}$, on the empirical ground (their citation of Fuchs and Koch 2014) that such a chain cannot account for the *graded intensity* of an emotion within an otherwise unchanged situation.

The design corollaries they extract are:

1. $A$ must possess a non-trivial $r_A$ — i.e. an internal physiological-like state that the situation can move. This is the **organismoid-embodiment requirement** (Ziemke, 2001).
2. $r_A$ and $m_A$ must have their own **temporal dynamics**, and the different "emotion" channels must **interact with one another** (Cañamero, 2019). They do not specify the form of that interaction.
3. The environment field $\alpha$ must be readable by both $H$ and $A$ — i.e. the robot's perception must register affordances, not just object identities.

No learning rule, reward signal, or update equation is proposed. The contribution is normative / definitional.

### Relevance flag for the present project

The Cañamero-school theoretical claim — that *internal homeostatic state and motor readiness must be co-modulated, not separately bolted on, for an emotion-like construct to obtain* — is directly relevant to a pain-modulated RL agent in a grid world. It is a construct-validity argument: if a project wants to claim that a modulated policy is "pain-like" or "affective" in any non-metaphorical sense, the same loop structure (resonance ↔ action readiness ↔ environment) must be visible in the architecture, not merely in the post-hoc interpretation of the policy.

## Connections

- **`khan_canamero_2022_social_buffering.md`** — same lab, same year, makes the affective-affordance / two-cycle commitment concrete with a robot experiment in which *context* gates affective perception.
- **`lharidon_canamero_2023_stress_pain.md`** — operationalises affective resonance as a stress / pain neuromodulator in an autonomous robot, the kind of mechanism this opinion piece is calling for.
- Earlier Cañamero work the authors cite as background: **Cañamero 1997** (motivations / emotions as basis for behaviour), **Cañamero 2005** (emotion as tool *and* model — with Gaussier), **Cañamero 2019** (embodied-robot models for interdisciplinary emotions research). The first two are slugs `canamero_1997_motivations_emotions.md` and `canamero_2005_emotion_understanding.md` in this corpus and should be cross-linked by the curator.
- **`avery_krichmar_2017_models_neuromodulation.md`** and **`cox_krichmar_2009_neuromodulation_robot_controller.md`** — the Krichmar-school complement: where Cañamero's school argues *why* affective state must be embodied, the Krichmar school argues *how* (which neuromodulator implements which switch). The opinion piece does not cite them, but they belong in the cross-paper synthesis.
- **`krichmar_2013_neurorobotic_anxiety_curiosity.md`** — operationalises an "anxious / curious" trade-off in a robot in a way that is exactly the kind of *temporal-dynamics-with-interactions* the authors call for in their final paragraph.
