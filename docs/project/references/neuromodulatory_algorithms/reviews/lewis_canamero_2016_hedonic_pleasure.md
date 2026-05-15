---
title: "Hedonic quality or reward? A study of basic pleasure in homeostasis and decision making of a motivated autonomous robot"
authors: ["Matthew Lewis", "Lola Cañamero"]
year: 2016
venue: "Adaptive Behavior 24(5), 267–291, SAGE (Special Issue on Grounding Emotions in Robots)"
slug: lewis_canamero_2016_hedonic_pleasure
source_pdf: "sources/Lewis and Cañamero 2016 - Hedonic quality or reward - A study of basic pleasure in homeostasis and decision making of a motivated autonomous robot.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

The earlier Cañamero-school architectures (in particular Cañamero 1997, Avila-García & Cañamero 2004, Cos et al. 2010, and Cañamero & Avila-García 2007) treat **pleasure** and **reward** as essentially the same thing — a positive signal released when the robot's homeostatic variables get *less* deficient. This paper asks: are they really the same? Lewis & Cañamero argue no. They run **three sets of experiments** on a Nao humanoid robot in a two-resource problem (food = red ball, drink = green ball), where the deficit on each resource grows at 0.1 every 250 ms and a "bite" reduces it by 10. The robot's motivations are computed as $\text{motivation}_i = d_i + (d_i \cdot \alpha \cdot \text{cue}_i)$, where $\alpha$ is a single shared *perceptual modulation* parameter — the **incentive-salience knob**. They test four ways to set $\alpha$: three fixed values (0.1, 0.35, 0.6) and a **modulated** condition where $\alpha$ tracks a simulated **pleasure hormone** $h$. The hormone is released proportionally to the *negative second derivative* of any deficit ($-d_i'' \cdot s$) — i.e., it fires when a deficit's downward trend *accelerates*, signaling "the interaction is going well." A third experiment then adds a **second** source of pleasure release — a constant 40-unit pulse on the successful execution of a consummatory behavior, *decoupled from nutritional value* — a *hedonic-quality* pleasure that fires from "tasting good," not from "becoming sated." The headline findings: (i) modulating $\alpha$ via the hormone gives the robot a way to **decouple persistence from opportunism**, which fixed-$\alpha$ values cannot do; (ii) the extra hedonic pleasure, even when attached to the *abundant* resource, decreases overall discomfort and produces a measurable shift in deficit-balance; (iii) attaching the extra pleasure to the *scarce* resource is the most adaptive setting — it partially corrects the asymmetry the environment imposes. The paper is significant in the corpus because it argues that **pleasure ≠ reward** even in a minimal homeostatic agent: a sensory / hedonic pleasure decoupled from need can have its own adaptive value via incentive-salience modulation.

## Section-ordered backbone

### Abstract
A robot architecture and experiments to investigate the roles of pleasure in action selection of an autonomous robot that must survive. Three sets of experiments compare different types of pleasure — related vs. unrelated to physiological-need satisfaction — under different environmental circumstances. Pleasure, including pleasure unrelated to need satisfaction, has value for homeostatic management in terms of improved viability and increased flexibility.

### Section 1 — Introduction
Frames the paper inside the larger Cañamero-school programme on emotion/motivation/cognition. Pleasure has multiple meanings (sensory, achievement, gain/relief, social, activity, aesthetic — Frijda 2010; Kringelbach & Berridge 2010) but shares a common core of "positive affect / liking". The paper adopts the broad common-sense definition of pleasure as *liking*. Focuses on two contexts in which liking happens: (a) linked to physiological-need satisfaction; (b) as pure hedonic quality unrelated to need satisfaction. The mechanism used to model pleasure is a **simulated hormone that modulates perception** — abstract, capturing gross dynamics rather than modeling specific chemicals (contrasted with Krichmar 2008, 2012, 2013; Sporns & Alexander 2002). Key claim: prior Cañamero work (Cañamero & Avila-García 2007; Cos et al. 2013) and prior work in general have collapsed pleasure into reward (Cos et al. 2013; Gadanho & Hallam 2001; Hiolle & Cañamero 2009; Kitano 1995); this paper instead asks what *pleasure-without-reward* contributes. The paper's pleasure hormone is released as a function of need satisfaction (signaling improvement) and fosters **openness** — increasing or continuing an interaction that is going well. Tested in a two-resource problem (Spier & McFarland 1997).

### Section 2 — Robot's Action Selection Architecture

**2.1 Physiology.** Two essential variables — energy (replenished by food / red ball) and hydration (replenished by drink / green ball) — each in $[0, 100]$ with the upper bound as the ideal. Deficits $d_i = 100 - V_i$. Both deficits grow by 0.1 every 250 ms; starting at $d = 20$, the robot dies in 3 min 20 s without intake.

**2.2 Perception and actuation.** Nao humanoid with on-board camera (color/size detection of plastic balls), sonars (obstacles), foot contact sensors (collisions), gyroscopes (falls). Resources consumed in discrete "bites" of 10 deficit units. Resources do not deplete on consumption.

**2.3 Motivations.** Each variable has one associated motivation, computed as

$$
\text{motivation}_i = d_i + \underbrace{(d_i \cdot \alpha \cdot \text{cue}_i)}_{\text{perceptual component}}, \qquad (1)
$$

where $\text{cue}_i$ is the size of the largest detected resource of type $i$ (or zero if none), and $\alpha$ is the **shared perceptual-modulation parameter**. The whole paper is about how $\alpha$ should be set — fixed vs. hormone-modulated.

**2.4 Behaviors.** Four top-level behaviors: a reflex `recover-from-fall`, a `return-to-neutral`, and two motivated behavioral subsystems (`hunger`, `thirst`), each composed of an appetitive search and a consummatory consumption sub-tree (`wander → visual-search → gaze → approach → eat/drink`). Behaviors inherit their parent's activation; multiple behaviors may run in parallel if they don't share an actuator.

**2.5 Modeling pleasure.** Three pleasure types:

1. **Modulated-α (Moa)**: hormone released as a function of need satisfaction; $\alpha$ is then a function of hormone level. Defined per:

   $$
   \Delta_+ h = -d_t'' \cdot s, \quad \text{if } d_t'' < 0, \qquad (2)
   $$

   where $d_t''$ is the discrete-time second derivative of any deficit, computed over a 500 ms cycle:

   $$
   d_t' = (d_t - d_{t-0.5}) / 0.5, \qquad d_t'' = (d_t' - d_{t-0.5}') / 0.5,
   $$

   and $s = 0.4$ (or 0.8 in Experiment 3). After a "bite", $d_t'' \approx -40$, so a single bite releases ~16 hormone units (or 21.6 in Experiment 3).

   Hormone decay to a background level:

   $$
   \Delta h = -(h - b) \cdot k, \qquad (3)
   $$

   with $b = 10$ and $k = 0.19$, giving a half-life of ~3.3 s relative to background. The hormone modulates $\alpha$ via

   $$
   \alpha = 0.01 \cdot h. \qquad (4)
   $$

2. **Fixed-α controls (LFa = 0.1, MFa = 0.35, HFa = 0.6)** — used in Experiments 1 and 2 as baselines. LFa $\alpha = 0.1$ was deliberately chosen to match the background-level hormone in Moa (where $\alpha = 0.01 \cdot 10 = 0.1$).

3. **Additional hedonic release** — used in Experiment 3. On every successful consummatory `eat` or `drink`, a *constant* 40 units of hormone are released, decoupled from any need satisfaction.

**2.6 Action selection process.** Asynchronous parallel loops. The behavior-selection cycle runs every 125 ms (8 Hz) and, for each top-level behavioral subsystem, sorts sub-behaviors by activation (highest first), checks whether each is active and whether its needed actuators are free, and selects.

### Section 3 — Method and Metrics

**Method.** 2×2 m arena with wooden walls (40 cm). Resources fixed to walls (sometimes a central box). Manipulated: easy vs. difficult access, symmetric vs. asymmetric distribution, pleasure-tied-to-nutrition vs. pleasure-tied-to-act-of-consuming.

**Metrics.**

- *Discomfort* (the converse of Avila-García & Cañamero 2004's comfort):
  - Arithmetic: $D_A(t) = \tfrac{1}{n} \sum_i d_i(t)$
  - Geometric: $C_G(t) = \big( \prod_i (100 - d_i(t)) \big)^{1/n}$, then $D_G(t) = 100 - C_G(t)$.

  Geometric discomfort is preferred because (a) it is maximal iff the robot is dead, (b) $D_G \ge D_A$, (c) it is closer to the largest deficit (the most pressing need).

- *Variance of deficits.*

- *Persistence and opportunism* — extending Avila-García & Cañamero 2004:
  - *Persistent consumption*: consumption that started when its deficit was the largest, continued past the *crossover point* (where the other deficit grew larger).
  - *Opportunistic consumption*: consumption that started when its deficit was *not* the largest.
  - *Attempted persistence/opportunism*: same predicates, but counting *attempts* rather than only successful consumption — capturing the cost of failed attempts.

### Section 4 — Experiment 1: Comfortable Environment

Setup: 4 food + 4 drink balls fixed to walls, symmetric distribution. 4 conditions: LFa, MFa, HFa, Moa. 10 runs each (40 total), 6 min cap. All 40 runs survive.

Findings:
- Discomfort decreases monotonically with $\alpha$: $\overline{D_G} = 49.8, 33.3, 27.1$ for LFa, MFa, HFa; $\overline{D_G} = 39.7$ for Moa (between LFa and MFa).
- Variance of deficits *rises* with $\alpha$: 28.0, 59.3, 71.2 for LFa, MFa, HFa; 47.0 for Moa.
- Persistence and opportunism rise *together* with $\alpha$ for fixed conditions — but for Moa, **opportunism stays near LFa level while persistence rises to MFa level**. The pleasure hormone has produced a decoupling that no fixed $\alpha$ value can produce.

Interpretation: in a comfortable environment, simply maximising $\alpha$ is best for discomfort. The Moa condition is not best on discomfort, but its decoupling of persistence from opportunism is a behavioural pattern unreachable by any fixed $\alpha$.

### Section 5 — Experiment 2: Difficult Access to Resources

Setup: central white cardboard box obstructs vision across the arena; food collected on one side, drink on the other; same 4 conditions. Robot must traverse to access each.

Findings:
- Death rates: LFa 9/10, MFa 2/10, HFa 4/10, Moa 5/10. **High $\alpha$ is no longer best — too much attempted opportunism wastes time.**
- $\overline{D_G}$: 65.7, 52.8, 50.6 for LFa/MFa/HFa; 61.9 for Moa. Variance: 44.9, 146.9, 192.4, 89.1.
- Persistence/opportunism: same decoupling pattern as Experiment 1 in the Moa condition.

Interpretation: in a challenging environment, the persistence-opportunism decoupling enabled by Moa is genuinely useful — the designer can independently tune *attentional commitment to ongoing consumption* (via hormone-release dynamics) and *attentional commitment to alternative resources* (via background hormone level). Moa was not optimized; the authors emphasise this. The point is that Moa offers a *mechanism* for a designer or evolutionary algorithm to adjust persistence and opportunism independently.

### Section 6 — Experiment 3: Introducing Asymmetry

Setup: 6 food + 2 drink (asymmetric environment) or 4+4 (symmetric, for comparison), plus a *second* hormone release tied to successful consummatory acts (not to nutrition). 5 conditions: symmetric env + symmetric pleasure (baseline); symmetric env + extra pleasure from drink; asymmetric env + symmetric pleasure; asymmetric env + extra pleasure from drink (scarce); asymmetric env + extra pleasure from food (abundant). Nutritional value per bite lowered from 10 to 7 to make asymmetries visible.

Findings (Table 2 in paper):
- Symmetric environment, symmetric pleasure: $\overline{D_A} = 46.7$, $\overline{D_G} = 47.4$, variance 61.8.
- Symmetric env, extra-drink pleasure: $\overline{D_G} = 43.4$, variance 81.5 — pleasure decreases discomfort but raises variance.
- Asymmetric env, symmetric pleasure: $\overline{D_G} = 57.6$, variance 76.2; 4 deaths.
- Asymmetric env, extra-drink pleasure: $\overline{D_G} = 50.1$, variance 98.0; 1 death. The scarce-resource pleasure partly corrects the environmental asymmetry.
- Asymmetric env, extra-food pleasure: $\overline{D_G} = 52.2$, variance 105.5; 3 deaths. Extra pleasure on the abundant resource still lowers discomfort, but the variance is high and the imbalance worsens.

Interpretation: pleasure can be a *mechanism for adapting to environmental asymmetry*, independently of the nutritional value. The most adaptive setting is when extra pleasure aligns with environment scarcity. But even mis-aligned extra pleasure (food, the abundant resource) reduces aggregate discomfort — because the *fixed* size of extra pleasure means it amplifies persistence on whichever resource is selected, raising consumption of *both* on average.

### Section 7 — Discussion
Reviews the three experiments. Key conceptual point: in Moa, **rates of persistence and opportunism are decoupled** because:
- The *background* hormone level (b = 10) controls opportunism (independent of any ongoing interaction).
- The *peak* hormone level during consumption (governed by $s$ and $k$) controls persistence (only relevant when interacting).

This decoupling is a true degree of freedom that fixed $\alpha$ does not provide. Three things distinguish the *additional sensory pleasure* mechanism from simply scaling up $s$ in the nutrition-tied release:
1. In a world with heterogeneous nutritional values, $s$-scaling would couple to nutritional value while a separate "act of consuming" hormone would not — enabling the agent to keep persistence on a low-nutrition resource.
2. The two-source mechanism can model more complex phenomena (e.g. resources that are more pleasurable when not eaten recently, cultural taste differences).
3. Temporal dynamics: in organisms with slower digestion, nutritional pleasure is delayed; the two-source mechanism can keep an immediate "tasting good" pleasure separate from a delayed "becoming sated" pleasure.

### Section 8 — Conclusion and Future Work
Pleasure can play an important role at the *basic*, perception-level — not only in high-level learning. Pleasure ≠ reward (extending Krichmar & Röhrbein 2013's argument that value ≠ reward). Modulated-$\alpha$ improved viability and behavioural flexibility, especially in challenging environments. Hedonic-quality pleasure decoupled from nutrition can correct environmental asymmetry. Future steps: heterogeneous resources, dynamic environments, integration into the Robin social robot.

## Phase 1 — undergraduate-level synthesis

**The plain idea.** A robot has two needs (food, drink) and one knob $\alpha$ that controls how strongly visible resources pull at it. If $\alpha$ is small, the robot more or less ignores food / drink it sees and decides purely on internal deficits. If $\alpha$ is big, the robot is strongly drawn to whatever resource it sees, sometimes at the cost of its more urgent need. Lewis & Cañamero ask: what if $\alpha$ is not a constant but is **set on the fly by a hormone** that fires when the robot is currently *doing well* — specifically when the deficit it's working on is *accelerating downwards*? Then the robot will be most easily pulled by visible resources *when it's already eating well*, and least pulled when it's far from food and exploring. This is a different kind of "rewarded" signal: not "I need this", but "this is going great, keep going."

**Why it matters.** With a fixed $\alpha$, persistence (sticking with what you're eating past the crossover point) and opportunism (grabbing a less-needed resource when you see it) move together — both go up as $\alpha$ goes up. With the **hormone-modulated** $\alpha$, persistence and opportunism move *independently*: the background hormone level (1) controls opportunism (during exploration), while the peak hormone level during a successful interaction (2) controls persistence (during consumption). This is a free design dimension that fixed $\alpha$ cannot give you.

**The setup.** A Nao humanoid in a 2 × 2 m arena with red ball (food) and green ball (drink) on the walls. Three experiments: (1) easy environment with 4 of each; (2) the same with a central obstacle so the robot must cross the arena to switch resources; (3) asymmetric distribution (6 food + 2 drink) and an *extra* pleasure pulse on consummatory acts, decoupled from nutritional value.

**The result.** Experiment 1 (easy) — both fixed-$\alpha$ and Moa work; high $\alpha$ wins on discomfort. Experiment 2 (hard) — high $\alpha$ stops winning; some agents die from over-eager opportunism; the Moa condition gives the most flexible behavior and the second-lowest discomfort. Experiment 3 (asymmetric) — adding hedonic pleasure to the scarce resource cuts deaths from 4/10 to 1/10 and corrects most of the environmental imbalance. Adding hedonic pleasure to the abundant resource still lowers discomfort but worsens the imbalance.

**Concrete worked example (Experiment 2).** The robot has food deficit 30 and drink deficit 40. It walks into the food side of the arena (drink is hidden behind the box). With fixed high $\alpha = 0.6$, the visible food balls dominate motivation; the robot eats, eats, eats — by the time it disengages and goes looking for drink, its drink deficit is 80 and it dies on the way. With Moa, the hormone $h$ briefly rises during eating and pulls $\alpha$ up to ~0.4; once the deficit-acceleration is over and the hormone decays, $\alpha$ drops back to ~0.1 and the robot freely disengages and searches for the more urgent drink.

## Phase 2 — graduate-level deep dive

### Motivation function and the role of $\alpha$

Per Eq. 1,

$$
m_i(t) = d_i(t) + d_i(t) \cdot \alpha(t) \cdot \text{cue}_i(t) = d_i(t) \big( 1 + \alpha(t) \cdot \text{cue}_i(t) \big),
$$

so $m_i$ is the product of the *internal urgency* $d_i$ and a *perceptual amplification* $(1 + \alpha \cdot \text{cue}_i)$. The two limits: when $\text{cue}_i = 0$ (no resource visible), $m_i = d_i$ and the robot acts purely on internal state. When $\text{cue}_i \gg 0$ and $\alpha$ is large, $m_i \approx d_i \cdot \alpha \cdot \text{cue}_i$ and the visible resource dominates. This is exactly the *incentive salience* formulation Berridge & Robinson (1998) propose for dopamine — and the paper makes the link explicit (§1, citing Berridge & Robinson 1998 and Pessoa 2013).

### Pleasure hormone dynamics

Hormone *release* is triggered by the negative second derivative of any deficit:

$$
\Delta_+ h(t) = \begin{cases} -d_t'' \cdot s & d_t'' < 0 \\ 0 & d_t'' \ge 0 \end{cases}, \qquad (2)
$$

with $d_t', d_t''$ computed on a 500 ms grid. Hormone *decay* to background $b$ at rate $k$:

$$
\Delta h(t) = -(h(t) - b) \cdot k, \qquad (3)
$$

so $h$ obeys, in continuous-time form,

$$
\dot h(t) = -k \big( h(t) - b \big) + \sum_n [-d''(t_n) \, s]_+ \delta(t - t_n),
$$

an exponential relaxation to $b$ punctuated by impulsive positive kicks at the time-steps $t_n$ where the deficit acceleration is negative. The half-life relative to background is $\log(2)/k = \log(2)/0.19 \approx 3.65$ s — close to the paper's stated 3.3 s when measured at the 500 ms grid.

### The decoupling of persistence and opportunism — derivation

Opportunism, by the paper's definition, happens when the robot starts a consummatory behaviour on a resource whose deficit is *not* the largest. By definition, no consumption has just happened on this resource, so $d''_i \ge 0$ and the hormone $h$ stays near background $b$. Therefore $\alpha = 0.01 \cdot h \approx 0.01 \cdot b = 0.1$ during opportunism — *equivalent to LFa*. The rate of opportunism is therefore set by $b$.

Persistence, by contrast, happens when the robot continues consuming a resource past the crossover. By definition, consumption is ongoing, so $d''_i < 0$ and $h$ is well above $b$. The peak $h_{\max}$ during steady consumption is approximately

$$
h_{\max} \approx b + \frac{r}{k} = b + \frac{|d_t''| \cdot s}{k},
$$

where $r = |d_t''| \cdot s$ is the per-step release. With $|d_t''| \approx 40$, $s = 0.4$, $k = 0.19$: $h_{\max} \approx 10 + 84 = 94$. The corresponding $\alpha_{\max} \approx 0.94$. So during persistence the effective $\alpha$ is much higher than LFa — *equivalent to or beyond HFa*. The rate of persistence is therefore set by the per-bite release $r$ and decay rate $k$.

Because the two states (no-recent-consumption vs. mid-consumption) tap separate parameters of the hormone dynamics, persistence and opportunism are mechanistically decoupled. With a fixed $\alpha$, the same $\alpha$ value sets both — explaining why fixed-$\alpha$ conditions show persistence and opportunism rising in lockstep, while Moa does not.

### Geometric vs. arithmetic discomfort

$$
D_A(t) = \frac{1}{n} \sum_i d_i(t), \qquad C_G(t) = \left( \prod_i (100 - d_i(t)) \right)^{1/n}, \qquad D_G(t) = 100 - C_G(t).
$$

For $n = 2$: $D_G = 100 - \sqrt{(100 - d_1)(100 - d_2)}$. Two properties make $D_G$ preferable:

1. *Death-sensitivity*: if any $d_i = 100$, the product collapses to 0 and $D_G = 100$. The arithmetic mean does not have this property — a robot with $d_1 = 100, d_2 = 0$ has $D_A = 50$, which underestimates the danger.
2. *AM-GM inequality*: $D_G \ge D_A$ with equality iff $d_1 = d_2$. So $D_G$ is "closer to the largest deficit" — exactly the most-pressing-need quantity.

### Two-source pleasure model (Experiment 3)

In Experiment 3, the hormone release rule becomes

$$
\Delta_+ h(t) = \underbrace{[-d_t'' \cdot s]_+}_{\text{nutritional}} + \underbrace{40 \cdot \mathbb{1}\{ b_k^* \text{ just executed} \}}_{\text{hedonic, conditional on resource type}},
$$

with $s = 0.8$ (vs. 0.4 in Experiments 1-2) and the hedonic 40-unit pulse fired only when the consummatory behaviour of the *target resource* completes. The hedonic kick is independent of nutritional yield — i.e., it would still fire even if the resource gave zero deficit reduction. This is the operational realization of the *liking vs. wanting* dissociation discussed in Berridge & Robinson 1998.

### Mathematical commentary

Three structural choices repay reflection:

1. **Second derivative as the release trigger.** Cañamero 1997 used the *deficit itself* to drive motivation; Blanchard & Cañamero 2006 used the *first derivative of well-being* (their pleasure $P_l$); Lewis & Cañamero 2016 use the *second derivative of deficit*. Going up an order each time. The reason: the second derivative is the *acceleration* of compensation — it fires only when the robot's interaction with the world is *getting better*, not when it is merely good. This is a sharper teaching signal than first-derivative reward.

2. **A single shared $\alpha$ as the perceptual gate.** All motivations share the same $\alpha$. This is a deliberate simplification — Cañamero 1997 had per-emotion hormones each acting on different downstream channels; here a single hormone drives a single perceptual scalar. The benefit is interpretability: every effect can be traced to $\alpha(t)$.

3. **Hedonic pleasure as an exogenous mechanism.** In Experiment 3, the 40-unit hedonic pulse is *fired by the robot's own action* (executing the consummatory behaviour), not by any state variable. This dissociates pleasure from need entirely — pleasure becomes an *act-of-consuming* property rather than a *being-fed* property. The connection to Berridge & Robinson's *wanting* (incentive salience, dopaminergic) vs. *liking* (hedonic, opioid) dichotomy is explicit in §1 and §6.

### Pseudocode

```
Initialize: V_E = 100, V_H = 100, h = b = 10
Parameters: s = 0.4 (Exp 1,2) or 0.8 (Exp 3), k = 0.19, b = 10

Every 250 ms:
  d_E(t) = 100 - V_E
  d_H(t) = 100 - V_H
  d_E += 0.1; d_H += 0.1                       # passive growth

Every 500 ms (hormone update):
  d_t'  = (d_t - d_{t-0.5}) / 0.5
  d_t'' = (d_t' - d_{t-0.5}') / 0.5
  if d_t'' < 0:
    delta_plus = -d_t'' * s
    h += delta_plus

Every 1 s (hormone decay):
  h -= (h - b) * k

Continuously:
  alpha = 0.01 * h
  for each resource i:
    cue_i = max blob size in camera for resource i (or 0)
    m_i = d_i * (1 + alpha * cue_i)
  active_motivation = argmax_i m_i

Every 125 ms (behavior selection):
  Sort sub-behaviors of active_motivation by activation
  Pick highest-activation sub-behavior with free actuators
  Execute

On successful consummatory bite of resource i:
  d_i -= 10  # or 7 in Exp 3
  V_i += 10  # or 7 in Exp 3
  # In Exp 3 additionally:
  if i == favoured_resource_for_extra_pleasure:
    h += 40
```

## Connections

- **Direct architectural ancestor.** `canamero_1997_motivations_emotions.md` — same homeostatic-physiology + motivation-with-incentive-stimulus formulation; the $\alpha$ parameter generalizes the perception-modulation mechanism Cañamero introduced via the ART-1 vigilance threshold in 1997.
- **Direct intellectual antecedent.** Cañamero & Avila-García 2007 (cited heavily as "previous work" in §1) — used hormonal modulation of motivation in a *threat-and-deficit-driven* way; Lewis & Cañamero 2016 invert the polarity: the hormone fires when things go *well*, not when things go badly.
- **PerAc cousin.** `blanchard_canamero_2006_affect_modulated.md` — the pleasure signal $P_l$ (variation of well-being) plays the same conceptual role as the second-derivative-of-deficit hormone here; both papers use a positive scalar to flip the sign of behavioural attention. The 2006 paper's per-time-scale architecture is more complex; this 2016 paper is simpler and tightly tied to a fixed $\alpha$ knob.
- **Affordance / value cousin.** `cos_2010_affordances_consummatory.md` — the prior paper used the hormonal signal $S$ as a Hebbian-update teaching signal for affordance synapses; this paper uses an analogous hormone $h$ as a *perceptual-gain* modulator rather than a learning teacher. The two architectures could be straightforwardly composed.
- **Long-term-plasticity cousin.** `lones_canamero_2013_epigenetic_hormones.md` — same lab, same hormone-modulation philosophy, but at the developmental time-scale. Lewis & Cañamero 2016 stays at the adult-action-selection time-scale.
- **Cited Cañamero-school RL companion.** Cos, Cañamero, Hayes & Gillies 2013 ("Pleasure and reward in reinforcement learning") — the conceptual companion that *does* equate pleasure with reward; the present paper is partly an answer to it.
- **Cross-corpus links** (likely in other batches). Berridge & Robinson 1998 (wanting vs. liking), Kringelbach & Berridge 2010 (pleasure in the brain), Damasio 1999, Panksepp 1998 — the affective-neuroscience scaffolding. Krichmar 2008, 2012, 2013; Cox & Krichmar 2009; Avery & Krichmar 2017 — the broader neuromodulation-as-control corpus. The curator should cross-link.
