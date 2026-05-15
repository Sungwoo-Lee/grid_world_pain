---
title: "The Long-Term Efficacy of 'Social Buffering' in Artificial Social Agents: Contextual Affective Perception Matters"
authors: ["Imran Khan", "Lola Cañamero"]
year: 2022
venue: "Frontiers in Robotics and AI, 9:699573"
slug: khan_canamero_2022_social_buffering
source_pdf: "sources/Khan and Cañamero 2022 - The Long-Term Efficacy of 'Social Buffering' in Artificial Social Agents - Contextual Affective Perception Matters.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper asks a biology-derived question with an artificial-society experiment: **does positive social contact between agents protect them from environmental stress in the long run, and if so, by what mechanism?** The biological hypothesis being tested is called **"social buffering"** — the observation in humans and other social mammals that individuals who maintain affectionate social bonds report better survival, lower stress hormone levels, and improved adaptation in difficult environments. The hormone most commonly implicated is **oxytocin (OT)**, sometimes called the "social hormone" because it is released during positive touch (e.g., grooming). The hormone most commonly implicated as the *stressor signal* is **cortisol (CT)**, released by perceived internal-physiological deficit and perceived loss of control over the external environment.

The authors build a NetLogo society of six artificial agents that must keep two internal needs ("Energy" via eating, "SocialNeed" via touch) above zero or they die. Each agent has a homeostatic action-selection architecture, a rank, a possible bonded partner, and an internal hormone state (OT, CT, and a **Stress Tolerance** $\theta_{ST}$ that represents how much cortisol the agent can carry before flipping into "stressed" mode and switching from grooming to aggression). They compare three agent designs across three environments: **Control** (no OT effects), **Type BV** (OT increases bond-partner valence — agents perceive bonded partners as more positive), and **Type BV+ST** (OT also raises the stress tolerance $\theta_{ST}$).

The headline result: **Type BV+ST agents do best**, but *not for the reason the authors had predicted.* The benefit is **contextual** and **temporally front-loaded** — bonded agents who interact a lot *early* (when food is abundant) accumulate OT, which raises their stress tolerance *before* the environment becomes harsh, so that when the food crisis hits they continue to groom rather than fight. This is an **anticipatory ("allostatic") adaptation**: a positive feedback loop where early grooming raises OT raises $\theta_{ST}$ keeps the agent in the prosocial regime keeps OT high. The authors call OT and CT **"embodied biomarkers"** — low-dimensional aggregate signals that summarise interoceptive deficit + environmental uncertainty, doing the job a more expensive memory or predictive-processing model might do.

## Section-ordered backbone

### Abstract
In a small society of artificial agents that must regulate two homeostatic needs (Energy, SocialNeed), the authors model **two hypothesised hormonal mechanisms of social buffering**: (i) OT increases the perceived valence of bonded partners, and (ii) OT modulates an internal stress tolerance threshold. Both mechanisms are tested across three social-bond combinations (high-rank, middle-rank, low-rank "close-kin" bonds) and three environments (static, seasonal, extreme food availability). Long-term wellbeing improvements depend on *contextual* affective perception, on the volume of early-stage positive interactions, and on the severity of physical / social challenge. Stress can be adaptive *or* maladaptive depending on context; the OT / CT signals can be read as embodied "biomarkers" of social support and environmental stress.

### Introduction
Social adaptation in dynamic environments depends on the agent's affective perception of context — its current affective state, environmental cues, and the kinds of interactions available. Stress can be adaptive (it drives behavioural change) or maladaptive (it dysregulates physiology). One persistent finding in animals and humans (Schülke et al. 2010; Holt-Lunstad et al. 2010): individuals embedded in supportive affective networks report improved wellbeing and survival. The "social buffering" hypothesis (Kikusui et al. 2006; Cohen and Wills 1985) attributes this to social support buffering both psychological and physiological stress responses. Oxytocin has been implicated in at least two ways: (i) increasing the affective valence of bond partners; (ii) "buffering" the autonomic stress response. The authors integrate these into the **social allostasis** framework (McEwen and Wingfield 2003; Sterling 2020) — homeostatic set-points are not fixed but are anticipatorily *adjusted* by hormones such as cortisol and oxytocin to minimise future error before it occurs. The paper's contribution is an *in silico* test of whether modelling these mechanisms in an artificial society reproduces the long-term wellbeing benefit observed in nature, and what its limits are.

### Related work
Existing affective-robotics work has examined affect-based behavioural adaptation (Hiolle et al. 2014; Tanevska et al. 2019), imitation (Breazeal et al. 2005), and social learning (Bartoli et al. 2020), often at a *dyadic* (human–robot) level. The authors' prior work has moved to group/society scale (Khan et al. 2019, 2020; Khan and Cañamero 2021). On the stress side, most HRI work examines effects on the *human*, not the *robot* (Aminuddin et al. 2016; Willemse and Van Erp 2019). Lewis and Cañamero (2019) is one of the few that examines an embodied stress model's effect on a robot's own wellbeing. The present paper extends that line by adding the social-buffering mechanism.

### Research questions and hypotheses
**H1**: Social buffering will improve long-term wellbeing of bonded agents compared to unbonded controls (with sub-questions RQ1.1, RQ1.2 separating the bond-valence and stress-tolerance mechanisms). **H2**: Social buffering will significantly affect the type of social behavioural dynamics (rates of grooming vs. aggression) (with sub-questions RQ2.1, RQ2.2). Their *prediction* — based on mixed human evidence — is that **OT's effect on social-salience (Type BV) will dominate** over its effect on stress tolerance (Type BV+ST). The empirical result inverts this prediction.

### Agent model
**Action-Selection Architecture (ASA)**. Following Cañamero (1997), agents regulate two internal variables $v_1$ (Energy) and $v_2$ (SocialNeed), each with a small per-time-step decay $\gamma_v$. Each tick, the deficit $d_v = D - v$ is combined with perception of the relevant stimulus to form a motivation $m$ (Hungry, Lonely); the higher motivation wins, and the matching behaviour (Eat, Touch) is selected.

**Social Assessment Component (SAC)**. Before executing a Touch, the agent computes an affective appraisal of each visible agent $B$, combining the rank difference $\Delta K_{AB}$ and the existence + strength of an affective bond (Dyadic Strength Index $\Upsilon$), modulated by the actor's own OT level. This gives a number $\chi_B$ (called AgentVal) that gates whether Touch becomes Groom (positive) or Aggression (negative).

**Affective hormonal system**. CT is released as a function of internal physiological deficit plus perceived external uncertainty (low food availability, low bond-partner availability). OT is released by positive tactile interaction with bond partners. **Stress Tolerance** $\theta_{ST}$ defaults to 0.5; in the BV+ST model it is modulated by current OT. CT modulates the agent's movement speed (and therefore Energy expenditure) and the *intensity* of any tactile interaction it executes.

### Experimental design
3 model types (Control / BV / BV+ST) × 3 bond combinations (high-rank A1-A2-A6 / middle A3-A4-A5 / low A4-A5-A6) × 3 environments (Static / Seasonal / Extreme). 20 runs of 15,000 time steps each.

### Results
**Viability** (Life Length, Comfort, Physiological Balance): BV+ST > BV > Control, with statistically significant gains in *all* environments. Life-length improvements were 20%–75% across conditions. **Hormones**: bonded agents had highest OT in static (easiest) environments; OT was inversely correlated with environmental difficulty. CT was *strongly inversely correlated with rank in Control* (lower-ranked = more stressed) but this rank-CT correlation disappeared once bond effects were modelled. **Stress Tolerance**: under BV+ST, $\theta_{ST}$ ended up *lower* than default for the highest-ranked bonded agents (0.29–0.33) and *higher* for the lowest-ranked bonded agents (0.40–0.61) — i.e., the buffering effect was strongest where it was least biologically expected. **Social interactions**: intra-bond grooming was higher and intra-bond aggression lower under BV+ST; under BV alone, grooming was front-loaded in early time steps and dropped off, under BV+ST it stayed elevated throughout. **Early-stage grooming** correlated strongly with long-term life length under BV+ST (r ≈ 0.6 – 0.8 across conditions) but only weakly under BV.

### Discussion
The key finding is that **the two OT mechanisms are not interchangeable**. Modelling only the bond-valence effect (BV) gives a modest improvement; adding the stress-tolerance effect (BV+ST) unlocks a *positive feedback loop*: early grooming → OT up → $\theta_{ST}$ up → CT less likely to exceed $\theta_{ST}$ when the environment turns harsh → continued grooming rather than aggression → more OT → loop closes. The authors interpret this as an **anticipatory allostatic adaptation**: the agent prepares its physiology *before* the stressor arrives. The behavioural footprint maps onto biology: BV-only agents look "fight-or-flight", BV+ST agents look "tend-and-befriend" (Taylor 2006). The authors propose that the simple OT and CT scalars can be read as **embodied biomarkers** that aggregate interoceptive deficit, external uncertainty, and social-bond history, providing a computationally cheap alternative to deep predictive-processing models for socially-adaptive embodied agents.

### Limitations / future work (Section 6)
Future work should extend to physical robots, larger societies, more hormones (testosterone, oestrogen, dopamine), and direct biological tests of the "early-stage OT regulates long-term $\theta_{ST}$" hypothesis.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** Two artificial hormones — **oxytocin (OT, released by good social touch)** and **cortisol (CT, released by hunger and uncertainty)** — combined with a simple **stress tolerance** dial that OT raises and CT triggers, are enough to reproduce the human-biology pattern in which agents with affectionate early-life relationships survive better in later-life crises.

**Setup.** Six little discs in a NetLogo world (99 × 99 grid). They need to eat food (yellow spheres) and touch each other (good touch = grooming, bad touch = aggression). They have ranks. Three of the six share an "affective bond" with each other; the others don't. Each has internal hormone levels that change over time.

**The three flavours of agent.**
- *Control:* no oxytocin effects. Cortisol still flips them between grooming and aggression.
- *Type BV* (Bond Valence): oxytocin makes bond partners *look* more attractive (raises $\chi_B$, the value the agent assigns to its partner), so the agent prefers grooming them and treats them as a "safer" feature of the environment.
- *Type BV+ST*: in addition, oxytocin raises the agent's *stress tolerance* — i.e., the amount of cortisol the agent can carry before it flips from grooming into aggression.

**The three environments.** Static (food never moves), Seasonal (food smoothly cycles 4 → 1 → 4 every 1000 steps), Extreme (food cliff-edges 4 → 1 instantly).

**The result, in plain English.** BV+ST agents outlive the others. The reason isn't that BV+ST agents are *less stressed by* the food crisis at the moment it arrives. It is that, **before the crisis**, when food is abundant, they groom each other a lot, build up oxytocin, and *raise their stress tolerance threshold in advance*. When the food crisis arrives and their cortisol spikes, the cortisol doesn't exceed the (now-raised) tolerance, so they don't flip into "fight" mode — they keep grooming each other, which keeps oxytocin up, which keeps the threshold up. It is **anticipation by buffer**, not anticipation by prediction. The take-away the authors emphasise: **early-life positive social interaction sets the long-term affective regime**. Take it away and you get a fight-or-flight agent; provide it and you get a tend-and-befriend agent.

## Phase 2 — Graduate-level deep dive

### Internal-need decay and motivation

Each agent has two internal variables $v_n \in [0,1]$ ($n=1$ Energy, $n=2$ SocialNeed), with set-point $D=1$ and per-tick decay:

$$
v_{n,t} = v_{n,t-1} - \gamma_v.
$$

SocialNeed has a constant loss rate $\gamma_2 = 0.003$. Energy's loss rate is *coupled to movement speed* (see speed equation below), so faster movement spends more energy:

$$
\gamma_1 = \gamma_{E0} \cdot 2 \cdot \mathrm{Speed}_t.
$$

The deficit is $d_{v,t} = D - v_{n,t}$. Using McFarland and Spier's (1997) **cue-deficit model**, the motivational intensity for need $n$ given the perceived relevant stimulus $S_i$ is

$$
m_t = d_t + d_t \cdot S_i,
$$

and the winning motivation $M_t = \max(m_H, m_L)$ (Hungry vs. Lonely). The winning behaviour $B_t$ comes from coupling $M_t$ to the satisfaction weight $\omega_{bv}$ of behaviour $b$ on the relevant variable $v$:

$$
b_t = M_t \cdot \omega_{bv}, \qquad B_t = \max(b_{\text{Eat}}, b_{\text{Touch}}).
$$

### Social appraisal

Rank difference $\Delta K(A,B) = K_A - K_B \in [-1, +1]$. Affective bond indicator $\Xi_{AB} \in \{0,1\}$. Dyadic Strength Index $\Upsilon_{AB} \in [0,2]$ with decay $\mu_\Upsilon = 0.9997$ per tick when no interaction occurs. The agent's appraisal of $B$ is

$$
\chi_B = \underbrace{\Delta K(A,B)}_{\text{rank difference}} + \underbrace{\Xi_{AB} \cdot \Upsilon_{AB} \cdot \mathrm{OT}_A}_{\text{bond status, gated by OT}}.
$$

Range: $\chi_B \in [-1, +3]$. **This is the Type-BV mechanism**: increasing $\mathrm{OT}_A$ multiplicatively scales how positively the actor sees its bond partner.

### Cortisol dynamics

$$
\mathrm{CT}_t = \mathrm{CT}_{t-1} + \gamma_{\mathrm{CT}},
$$

with secretion rate

$$
\gamma_{\mathrm{CT}} = \left( \underbrace{\bar{d}_v}_{\text{internal stress}} - \underbrace{\frac{\hat{S}_{\text{agents}} + \hat{S}_{\text{food}}}{2}}_{\text{external availability}} \right) \cdot w,
$$

where $\bar{d}_v$ is the mean of the two deficits, and $\hat{S}_{\text{agents}}, \hat{S}_{\text{food}}$ are perceived availabilities computed via

$$
\hat{S}_{\text{agents}} = S_{\text{agents}} \cdot (1 - \chi_B), \qquad \hat{S}_{\text{food}} = \mathbb{1}\{\chi_B \ge 0\}.
$$

$w = 0.005$ is the cortisol-sensitivity gain. CT modulates default speed $\mathrm{Speed}_0 = 0.5$ by

$$
\mathrm{Speed}_t = \mathrm{Speed}_0 \cdot (1 + \mathrm{CT}),
$$

closing a feedback loop: more CT → faster movement → faster Energy depletion → larger $d_v$ → more CT.

CT also scales tactile interaction intensity:

$$
\mathrm{TactInt} = b_{\text{Touch}} \cdot \mathrm{CT}.
$$

After a Touch event, the actor's SocialNeed is updated and its own CT is reduced:

$$
v_2 \leftarrow v_{2,t-1} + \mathrm{TactInt} \cdot c, \qquad \mathrm{CT}_t \leftarrow \mathrm{CT}_t - \mathrm{TactInt},
$$

with $c = 0.1$. For the recipient $R$ of the touch:

$$
\mathrm{CT}_R \leftarrow \begin{cases}
\mathrm{CT}_R - \mathrm{TactInt}_A \cdot o, & \text{TouchGroom},\\
\mathrm{CT}_R + \mathrm{TactInt}_A \cdot o, & \text{TouchAggression},
\end{cases}
$$

with $o = 0.3$. Bond strength evolves symmetrically:

$$
\Upsilon_{AR} \leftarrow \begin{cases}
\Upsilon_{AR} + \mathrm{TactInt} \cdot i, & \text{TouchGroom},\\
\Upsilon_{AR} - \mathrm{TactInt} \cdot i, & \text{TouchAggression},
\end{cases}
$$

with $i = 0.5$.

### Touch-type decision rule

The flip from grooming to aggression is gated by the **stress tolerance** $\theta_{ST}$ and the partner appraisal $\chi_B$:

$$
b_{\text{Touch}} = \begin{cases}
\text{TouchGroom}, & \mathrm{CT} < \theta_{ST},\\
\text{TouchGroom}, & \mathrm{CT} \ge \theta_{ST} \text{ and } \chi_B \ge 1,\\
\text{TouchAggression}, & \mathrm{CT} \ge \theta_{ST} \text{ and } \chi_B < 1.
\end{cases}
$$

This is the *behavioural switch* — the agent is calm (grooms) until CT exceeds tolerance; once tolerance is breached, it grooms only well-bonded partners ($\chi_B \ge 1$) and attacks everyone else.

### Oxytocin dynamics

OT is released bilaterally by a positive tactile interaction (intensity = $\mathrm{TactInt}_A$) and decays slowly with $\mu_{\mathrm{OT}} = 0.005$:

$$
\mathrm{OT}_t = (\mathrm{OT}_{t-1} - \mu_{\mathrm{OT}}) + \mathrm{TactInt}_A.
$$

### The Type-BV+ST coupling

Stress tolerance is *itself* modulated by OT around its default $\theta_{ST,0} = 0.5$:

$$
\theta_{ST,t} = \theta_{ST,0} \cdot (0.5 + \mathrm{OT}_t), \qquad \theta_{ST,t} \in [0.25,\, 0.75].
$$

This is the single equation that creates the positive feedback loop the authors call **affective anticipation**. Read it carefully: more OT → higher $\theta_{ST}$ → larger gap between current CT and the threshold → grooming continues (because the second branch of the Touch rule does not fire) → more grooming → more OT → loop. Crucially, $\theta_{ST}$ is modulated *bilaterally and continuously*, so OT accumulated during the easy early phase of an experiment is "banked" against later environmental stress.

### Viability metrics

Life Length is the fraction of run time the agent's Energy stayed above zero,

$$
\mathrm{LL}_A = \frac{t_{\text{life},A}}{t_{\max}}.
$$

Comfort is mean satisfaction across the two needs,

$$
\mathrm{CO}_A = \frac{1}{t_{\text{life},A}} \sum_{i=1}^{t_{\text{life},A}} \left( 1 - \bar{d}_i \right).
$$

Physiological Balance penalises asymmetric satisfaction between Energy and Social,

$$
\mathrm{PB}_A = \frac{1}{t_{\text{life},A}} \sum_{i=1}^{t_{\text{life},A}} \left( 1 - |d_1 - d_2| \right).
$$

Intra-bond rates are

$$
\mathrm{intraBondGroom} = \frac{\mathrm{Groom}_{A \to B}}{\mathrm{Groom}_{A \to B} + \mathrm{Groom}_{A \to UB}},
$$

and identically for Aggression.

### Why the prediction inverted

The authors had predicted (from mixed human evidence on cortisol buffering) that the *social-salience* effect would dominate. It did not — the *stress-tolerance* effect dominated. The mechanism is that the BV mechanism alone is a passive valence boost: it makes the partner *look* better but does not change *when* the agent flips out of grooming. The BV+ST mechanism is active because it changes the *threshold itself*. Once $\theta_{ST}$ is raised, the agent's behavioural regime stays in the "tend-and-befriend" branch for longer, which generates more OT, which raises $\theta_{ST}$ further. **The benefit is in the threshold dynamics, not in the perception dynamics.**

### Relevance flag for the present project

For a pain-modulated RL agent in a grid world, this paper is a clean working example of **a neuromodulatory scalar gating a behavioural-regime switch** through a thresholded comparison. The structural argument — that the *threshold* moves, not the underlying signal, and that the move is anticipatory — is directly transferable to a pain-tolerance threshold modulated by an interoceptive state. The paper also documents the *correct null comparison* (the same architecture *without* the threshold modulation) and the standard viability metrics (life length, comfort, balance) used in the Cañamero school.

## Connections

- **`scarinzi_canamero_2022_affective_interactions.md`** — same lab, same year, the conceptual companion piece. Where Scarinzi and Cañamero argue *why* affective state must be embodied and co-constituted, this paper is the *how* with measurements.
- **`lharidon_canamero_2023_stress_pain.md`** — extends the same architecture (homeostatic + hormonal modulation) to *pain* perception under stress, directly relevant to the present project.
- **`canamero_1997_motivations_emotions.md`** — the original homeostatic ASA used here.
- **`canamero_2005_emotion_understanding.md`** — frames the methodology (robots as tools to study emotion).
- **Lewis and Cañamero (2019)** ([`lewis_canamero_2019_hedonic_pleasure.md`](lewis_canamero_2019_hedonic_pleasure.md), in this corpus) — predecessor on embodied stress affecting compulsive behaviours in a robot.
- **Lones et al. (2018)** ([`lones_canamero_2018_hormone_epigenetic.md`](lones_canamero_2018_hormone_epigenetic.md), in this corpus) — same lab tradition; hormones drive epigenetic-style adaptive switches.
- **Krichmar school** (`avery_krichmar_2017_models_neuromodulation.md`, `cox_krichmar_2009_neuromodulation_robot_controller.md`, `krichmar_2013_neurorobotic_anxiety_curiosity.md`, `chiba_krichmar_2020_self_monitoring.md`) — complementary architectural treatment of neuromodulation; in particular, Krichmar's noradrenergic / serotonergic anxiety-vs-curiosity trade-off has the same *threshold-modulation-by-hormone* signature as the OT–$\theta_{ST}$ coupling here.
