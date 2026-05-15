---
title: "The effects of stress and predation on pain perception in robots"
authors: ["Louis L'Haridon", "Lola Cañamero"]
year: 2023
venue: "Proceedings of the 11th International Conference on Affective Computing and Intelligent Interaction (ACII 2023)"
slug: lharidon_canamero_2023_stress_pain
source_pdf: "sources/L'Haridon and Cañamero 2023 - The effects of stress and predation on pain perception in robots.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper is the closest analogue in the corpus to **the present project**: an embodied robot must seek and consume resources while a predator can damage it, and the **pain signal that the damage produces is modulated by an internal stress hormone (cortisol)**. The research question is: *does stress-modulated pain perception help or hurt a robot's survival, and does the answer depend on how dangerous the environment is?*

The agents are physical Khepera-IV wheeled robots in a wooden arena. They have three "motivations" — **hunger** (need to find food tiles), **cold** (need to find shade tiles), and **avoid danger** (need to escape predators) — running on a homeostatic action-selection architecture in the Cañamero school (Cañamero 1997). The predators are Thymio-II robots that *stalk* the Khepera. Damage is detected by an "artificial skin": the IR proximity sensors do double duty as **nociceptors**, computing **impact damage** (sudden change in distance to obstacle = collision) and **tearing damage** (spatial gradient across adjacent IR sensors = something is being scraped off). A **Gaussian pain-irradiation rule** spreads the nociceptive signal to neighbouring nociceptors so that a single hit is felt in a broad band, not a point.

**Cortisol** is a scalar released by *both* nociceptor activity *and* low comfort (high physiological deficit), with a decay constant — so it has a **hysteresis / memory** property. The key equation is one line: **pain = cortisol × damage**. Pain in turn slows the robot's wheels (motor inhibition).

The comparison is **N-models** (nociception only — pain is linearly correlated with damage) versus **C-models** (cortisol-modulated — pain is gated by accumulated stress). The headline result: **cortisol-modulated agents survive significantly better in the harsh environments (2 or 3 predators), but offer no benefit in the safe environment (0 or 1 predator)**. Emergent behaviour: in the 3-predator condition, the C-model agents start to *fight back* (turning loops that bump predators away to escape) because high cortisol *lowers their pain-aversion*. The take-away the authors propose: **cortisol modulation of pain perception is an adaptive mechanism specifically for high-stress regimes; under low stress, naive nociception is fine**.

## Section-ordered backbone

### Abstract
The paper investigates the impact of stress on pain perception in a survival-oriented robot. Cortisol-modulated pain perception is shown to be advantageous in high-predation environments (i.e., high stress-related danger) and less advantageous in low-predation environments.

### I. Introduction
Pain is a perceptual *and* affective experience associated with actual or potential tissue damage (Williams and Craig 2016), and its perception is modulated by multiple factors — nociceptor activity, cortisol concentration, attention, memory, prior history. **Hysteresis** in the cortisol signal means current behaviour depends on the agent's stress history, not just its present nociceptive input. The authors motivate the experimental robotic approach as a way to do controlled studies of the stress × pain interaction that are not ethically or methodologically feasible in animals.

### II. Robot Model

**Action-selection architecture.** Inspired by Cañamero (1997) and Avila-García and Cañamero (2004); uses a **Two-Resource Problem** with two homeostatic variables (Energy via food, Temperature via shade) plus a third "avoid danger" motivation. Winner-take-all selection.

**Perception.** Distance sensors (US, 25 cm – 1 m); proximity sensors (IR, 0–20 cm, normalised); ambient-light sensors (under the robot) for floor-tile resource discrimination.

**Nociception.** Each IR proximity sensor doubles as a *nociceptor* with two channels: **impact damage** = positive temporal gradient of sensor value (closing on an obstacle); **tearing damage** = spatial gradient across three adjacent IR sensors (the bigger the difference, the stronger the tear signal). The two channels are averaged. Damage **irradiates** to neighbouring nociceptors via a Gaussian-spread routine, simulating the *pain irradiation principle* (Avila-García and Cañamero 2004).

**Motivational system.** Physiological variables $v_i$ each have an ideal value and an error $\Delta_i$. External *cues* $c_i$ feed into a motivation intensity using the classical ethology formula adapted by Avila-García and Cañamero, $m_i = \Delta_i + \Delta_i \cdot c_i$. Three motivations: hunger, cold, avoid-danger. Winner-take-all.

**Behavioural systems.** Two: *Seek & Consume* (layered: consume → seek → wander) and *Reactive* (a Braitenberg "danger escape"). Each behaviour has a main and possibly secondary effect on physiological variables.

**Cortisol hormone.** Cortisol concentration $c_{\text{cortisol}}(t)$ is updated with a release rate $r_{\text{cortisol}}$ that depends on mean nociceptor activity and (1 − comfort). Cortisol decays multiplicatively per tick by $\psi_{\text{cortisol}}$ — this is what gives the system its *memory / hysteresis*.

**Pain perception.** Pain is defined as $\text{pain}(t) = c_{\text{cortisol}}(t) \cdot \text{damage}$. Pain throttles motor commands: left/right wheel speeds are multiplied by $1 + 0.5 \cdot \text{pain}(t)$ — and crucially the speed *boosts* with pain, meaning damage induces a flight-speed increase (the higher the cortisol, the more amplified the motor response).

### III. Experiments

**Hypothesis.** Cortisol modulation of pain perception will improve survival relative to nociception-alone, especially in predator-rich environments.

**Setup.** Khepera-IV robot in a 1.5 m × 1.5 m wooden arena with two resource types (food / shade) distributed symmetrically. Predators are Thymio-II robots running a *stalking* behaviour: slow movement, avoiding the arena border, attempting to follow and attack any robot they detect.

**Protocol.** 8 conditions = {0, 1, 2, 3 predators} × {nociception-only N, cortisol-modulated C}. 5 runs of 600 s each, total 40 runs.

**Results.**
- *Survival rate*: C-models match or beat N-models everywhere. In 2-predator and 3-predator conditions, N3 and N4 collapse to 60% and 30% survival, while C3 and C4 hold at 80%. The cortisol-modulated agent uses elevated cortisol as a "memory of previous stressful experiences" to avoid predators more efficiently.
- *Activity cycles* (trajectory in the Energy × Temperature error space): C-models maintain balanced consumption of both resources across all four conditions; N-models drift toward higher deficits at run-end, especially in N1, N2, N4 — they get pulled close to the "danger zone" of high combined deficits.
- *Cortisol intensity over time*: in low-predator C1 / C2 conditions, cortisol stays low. In C3 / C4, cortisol stays high for extended periods due to the hysteresis property — the robot "remembers" stress.
- *Hormone secretion dynamics graph* (concentration vs. release rate): clustering analysis (elbow method, x-mean) shows that C4 is uniform, C3 splits into a memory-dominated cluster (high concentration, low release) and a recovery cluster, C2 has two clusters, and C1 has only the high-release-low-concentration cluster — i.e., **more predators induce stronger hormonal memory**.
- *Motivations × nociception in C3*: a clear correlation — when nociceptor excitation spikes around the 110 s mark, the danger motivation dominates for an extended period; multiple nociceptor excitation pushes danger above hunger / cold.
- *Emerging behaviour*: in C3 and C4, the cortisol-modulated robot occasionally engages in a **"fight" response** — turning loops that bump predators, repelling them, allowing the robot to escape and reach the resource. The authors interpret this as elevated cortisol *lowering pain aversion*, enabling risk-taking.

### IV. Conclusion
Cortisol-modulated pain perception is adaptive specifically in high-predation environments — it produces both better survival rates and balanced physiological-deficit management. The Motivation–Emotion–Cognition Loop is incomplete without a stress signal feeding pain. Future work: study how alternative hormonal-decay or memory dynamics shape long-term sensitisation / chronic pain.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Build a wheeled robot that has to eat, stay warm, and avoid being eaten. Give it a body that can be "hurt" (IR sensors double as a pain skin). Then ask whether **letting an internal stress hormone amplify or dampen the pain signal** makes the robot survive better.

**The two designs.**
- *N-model (control):* the robot's pain is just a linear function of damage. Lots of damage = lots of pain = slow wheels = vulnerable.
- *C-model:* damage triggers cortisol, cortisol multiplies damage to give pain, and cortisol *decays slowly* over time. So a one-off scrape gives a short pain spike. Repeated scrapes in a high-predation arena raise cortisol steadily, and now even a *small* nociceptor activation produces a *large* pain response (because pain = cortisol × damage). But cortisol also accumulates as a *memory* of stress, biasing future motivation selection toward "avoid danger" before damage actually occurs.

**Setup.** Khepera robot in a 1.5 m × 1.5 m wooden arena. Food and shade tiles. 0, 1, 2, or 3 stalking Thymio-II predators.

**The result.**
- Without predators or with one predator, the two designs are roughly tied.
- With 2 predators, the no-cortisol robot drops to 60% survival; the cortisol robot holds at 100%.
- With 3 predators, the no-cortisol robot drops to 30%; the cortisol robot holds at 80%.

The reason the cortisol robot wins is **memory**: high cortisol after an early attack biases later behaviour toward danger-avoidance, even when the predator is not currently visible. It also produces an unexpected emergent behaviour in the hardest condition: when surrounded by three predators with no escape route, the cortisol robot's *lowered pain aversion* (because cortisol is so high that any pain feels routine) makes it execute aggressive turning loops that physically push predators away, opening a path to the resource. The robot has gone from "flight" to "fight" — purely as a side effect of cortisol's modulatory action on the pain channel.

## Phase 2 — Graduate-level deep dive

### Nociception with impact + tearing + Gaussian irradiation

Each of the $n$ IR sensors yields a normalised value $s_i(t) \in [0,1]$ where 1 = contact and 0 = nothing in range. Two damage channels are computed:

$$
\text{impact}_i(t) = \max\!\left(0, \; \frac{s_i(t) - s_i(t-1)}{(t) - (t-1)}\right),
$$

i.e., the positive part of the temporal gradient — only *closing on* an obstacle counts as impact; *receding from* does not. The denominator $\Delta t$ is one time step but is written explicitly in the paper.

$$
\text{tearing}_i(t) = f\!\left(s_{i-1}(t), s_i(t), s_{i+1}(t)\right),
$$

with $f$ proportional to the largest pairwise difference among the three adjacent IR readings — the bigger the gradient across the local strip of skin, the stronger the tearing signal. The two channels are averaged per nociceptor:

$$
\text{nociceptor}_i = 0.5 \cdot \left( \text{impact}_i + \text{tearing}_i \right).
$$

A **Gaussian-irradiation pass** then spreads each nociceptor's value to its neighbours: an $n \times n$ array is built whose $i$-th row is $\text{nociceptor}_i$ centred at index $i$ and Gaussian-decaying away; the final per-nociceptor value is the column-mean of that array, giving each nociceptor a contribution from all others weighted by spatial distance. This is the *pain-irradiation principle* — a local hit is felt as a broad band.

### Motivational intensity

For physiological variable $v_i$ with deficit $\Delta_i$ and relevant exteroceptive cue $c_i$:

$$
m_i = \Delta_i + \Delta_i \cdot c_i,
$$

i.e., the deficit acts both as a baseline and as a *gain* on the cue — a hungry robot detecting food gets more excited than a satiated robot detecting the same food. Winner-take-all selection $M_t = \max_i m_i$ over {hunger, cold, avoid-danger}.

### Comfort

Comfort is the mean of the inverse errors:

$$
\text{comfort} = \frac{1}{N} \sum_i (1 - \Delta_i) = \text{mean}(1 - \Delta_i),
$$

so $\text{comfort} \in [0,1]$ with 1 = perfect satisfaction of all needs.

### Cortisol release and concentration

Release rate is a weighted sum of nociception and discomfort:

$$
r_{\text{cortisol}} = \alpha \cdot \text{mean}(\text{nociceptors}) + \beta \cdot (1 - \text{comfort}),
$$

with $\alpha, \beta$ tunable constants setting the relative weight of *acute pain* vs. *chronic physiological deficit*. The concentration is integrated with multiplicative decay $\psi_{\text{cortisol}}$ and an upper saturation at 1:

$$
c_{\text{cortisol}}(t) = \min\!\left(1.0, \; c_{\text{cortisol}}(t-1) \cdot \psi_{\text{cortisol}} + r_{\text{cortisol}}\right).
$$

(The paper writes `max(1.0, ...)`, which is almost certainly a typo for `min(1.0, ...)` — cortisol is bounded above by 1, not below. The dynamics shown in Fig. 8 are consistent with the `min` reading.)

This is the **hysteresis** mechanism: $\psi_{\text{cortisol}} \in (0,1)$ controls how long an elevated cortisol level persists in the absence of new releases. Setting $\psi$ closer to 1 makes the robot a longer "rememberer" of stress; setting it closer to 0 collapses the system to instantaneous nociception.

### Pain — the bottleneck equation

The single equation that *modulates damage by stress* before damage drives behaviour is

$$
\text{pain}(t) = c_{\text{cortisol}}(t) \cdot \text{damage},
$$

where $\text{damage}$ is the mean nociceptor activity (post-irradiation). This is multiplicative gain, not additive — when cortisol is high, even modest damage feels major; when cortisol is low, the same damage feels minor. *This is the construct that the present project is most directly mapped to.*

### Motor inhibition by pain

Wheel speeds (per side) are multiplied by

$$
\text{speed}_{L/R} \leftarrow \text{speed}_{L/R} \cdot (1 + 0.5 \cdot \text{pain}(t)).
$$

Note the *positive* sign: pain *amplifies* speed, not attenuates it. The text states this is to "reduce the strength of motor actions to minimize further damage" but the equation as written increases motor magnitude with pain. Two plausible readings: (a) the robot uses higher speed to flee from the painful situation faster (consistent with the observed emergent fight/flight behaviour), or (b) this is an aggressive turning-loop gain that lets the robot ram predators in the C3 / C4 condition. Either way, the equation does not implement a classical analgesic-style slow-down.

### Why the survival result has the shape it does

In a low-predation arena (0 or 1 predator), nociceptor activity is sparse, so $r_{\text{cortisol}}$ stays small and $c_{\text{cortisol}}(t) \to 0$ under decay. The two models collapse to each other — the cortisol multiplier is approximately 1 (in the relevant operating regime), so pain ≈ damage in both cases.

In a high-predation arena (2 or 3 predators), nociceptor activity is sustained, $r_{\text{cortisol}}$ stays positive, and the integrator $c_{\text{cortisol}}(t) \cdot \psi_{\text{cortisol}} + r_{\text{cortisol}}$ converges to a fixed point set by the predator-encounter rate divided by $(1 - \psi_{\text{cortisol}})$. Sustained high $c_{\text{cortisol}}$ pumps up pain via the multiplicative rule, which (i) keeps the danger-motivation winning the winner-take-all for longer windows, and (ii) increases wheel speed during danger episodes (the speed equation), enabling the emergent fight-back loop the authors report in Fig. 12.

### Relevance flag for the present project

This is the **direct precedent** for stress-modulated pain in an embodied agent on a survival task. The structural commitments that transfer directly:

1. **Pain is gain × damage**, not pain = damage. The modulator (cortisol) is *multiplicative* on the nociceptive signal.
2. **Cortisol has memory** (one-pole IIR filter with decay $\psi$) — the modulator is not stateless.
3. **Cortisol is driven by *two* sources**: acute nociceptor activity ($\alpha$ term) *and* chronic homeostatic deficit ($\beta$ term). The deficit term is what generates *anticipatory* stress before damage occurs.
4. **The behavioural switch (flight → fight) is emergent**, not hard-coded — it falls out of the speed equation and the pain-gain becoming saturating in high-predator conditions.
5. **Construct validity is environment-conditional**: the modulation is adaptive in the harsh regime and neutral in the safe regime. Any pain-modulator design in the present project should be benchmarked against a no-modulator baseline across a *range* of difficulty levels, not just one.

## Connections

- **`canamero_1997_motivations_emotions.md`** — the original homeostatic motivational architecture that this work re-uses.
- **`khan_canamero_2022_social_buffering.md`** — the sibling paper. Both papers use the same Cañamero-school cortisol + threshold pattern; Khan & Cañamero modulate the *stress tolerance* with oxytocin, L'Haridon & Cañamero modulate the *pain gain* with cortisol. Together they cover the two halves of the threshold-vs-gain neuromodulation design space in the same lab.
- **`scarinzi_canamero_2022_affective_interactions.md`** — the conceptual companion: pain is not damage, it is an *affective* state shaped by an internal hormonal context. The L'Haridon paper is the empirical instantiation of that claim.
- **Avila-García and Cañamero (2004)** — cited as [1]. The hormonal-feedback-for-action-selection paper from the same lab; introduces the comfort metric and the irradiation routine.
- **Lewis and Cañamero (2019)** ([`lewis_canamero_2019_hedonic_pleasure.md`](lewis_canamero_2019_hedonic_pleasure.md), in this corpus) — robot model of stress-induced compulsive behaviour; cited as [11]. Same architecture family with a different stress endpoint.
- **Lewis, Fineberg, and Cañamero (2019)** — OC-spectrum disorders model; cited as [12]. Same architecture extended to psychopathology.
- **Krichmar school** (`krichmar_2013_neurorobotic_anxiety_curiosity.md`, `avery_krichmar_2017_models_neuromodulation.md`, `cox_krichmar_2009_neuromodulation_robot_controller.md`) — complementary architectural treatment of neuromodulation; especially Krichmar's anxiety / curiosity trade-off, which is the same "behavioural-regime switch under stress" pattern as the L'Haridon flight→fight emergence.
- **Lones et al. (2018)** ([`lones_canamero_2018_hormone_epigenetic.md`](lones_canamero_2018_hormone_epigenetic.md), in this corpus) — same lab tradition of hormone-driven slow adaptation in autonomous robots.
