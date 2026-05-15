---
title: "Epigenetic adaptation through hormone modulation in autonomous robots"
authors: ["John Lones", "Lola Cañamero"]
year: 2013
venue: "2013 The Third IEEE International Conference on Development and Learning and on Epigenetic Robotics (ICDL-EpiRob)"
slug: lones_canamero_2013_epigenetic_hormones
source_pdf: "sources/Lones and Cañamero 2013 - Epigenetic adaptation through hormone modulation in autonomous robots.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

Most of the Cañamero-school robot architectures fix the robot's "wiring" at birth — the hormone glands, the receptor sensitivities, the gain of each motivation are all hand-tuned by the designer for a particular environment. This paper asks: what if those parameters were *not* hand-tuned but instead **grew themselves** during the robot's first few minutes of life, shaped by what the robot bumps into? In biology this is called **epigenetic adaptation** — environmental cues during a critical developmental window leave durable marks on hormone glands and receptor sensitivities without rewriting DNA. The authors port that idea onto a Koala wheeled robot living in a 2×2 m arena with **three internal needs** (energy, condition, temperature) and **two coloured-ball resources** (energy and repair). The robot has four simulated hormones: $E_1, C_1, T_1$ (endocrine, one per homeostatic deficit) and $D_1$ (a neuro-hormone that responds to visual environmental cues and facilitates *aggressive*, contact-tolerating behaviour). Each hormone has a **gland activity** $\theta_h$ and **receptor sensitivity** $\text{sens}_h$. The novelty is that, only during the first 3 minutes ("early life"), $\theta_h$ and $\text{sens}_h$ themselves drift according to a slow integral update driven by current hormone concentration: glands grow stronger when their hormone is repeatedly needed, receptors become more sensitive when they keep firing. After 3 minutes the parameters freeze, but the robot is now *tuned to the environment it actually lives in*. Across three scenarios (open arena, obstacle-blocked resources, uneven-resource density), the epigenetic robot consistently outperforms a non-epigenetic baseline on Avila-García & Cañamero's *comfort* and *risk-of-death* viability indicators, and develops qualitatively new behaviours — including an *ambush-style* "pounce" gait that no designer programmed in. The paper is significant in the corpus because it adds a **slow developmental plasticity** dimension on top of the standard Cañamero homeostatic + hormone-modulation architecture.

## Section-ordered backbone

### Abstract
Epigenetic adaptation lets biological organisms adjust physiology/morphology in response to environment, and recent research suggests this is hormone-controlled. The paper presents a model that allows an autonomous robot to develop its systems in accordance with the environment it is situated in. Experiments span multiple environments with different challenges; results show emergence of unique behaviours tailored to exploiting the current environment.

### Section I — Introduction
Frames epigenetic adaptation as gene-expression change triggered by environmental cues during specific developmental periods, producing morphology / physiology changes that can occasionally be passed on without DNA changes (refs 7,8). Cites empirical examples: prenatal stress → risk-averse adult behaviour (refs 9–11); exposure to new predators → more defensive phenotype (ref 12); prenatal malnourishment → energy-conservation adulthood (ref 13). Cites recent evidence that **hormones provide the signal for epigenetic change** (refs 2, 15) and that this mechanism can be ported to epigenetic robotics (Bowes et al. 2009, ref 14). The contribution: a hormone-based epigenetic adaptation mechanism layered on the hormone-modulated action-selection architecture of Avila-García & Cañamero 2004 (ref 17), tested on a three-resource action-selection problem.

### Section II — Robotic Model

**A. Environment.** 2×2 m bordered arena. Two coloured balls = energy and repair resources. A climate variable sets ambient temperature.

**B. Morphology.** K-team Koala with 16 IR range-finders, a camera, an OpenCV-based vision stack. IR sensors are grouped into eight overlapping quartets to compute resistance/obstruction along eight cardinal/ordinal directions, with a slight forward bias. IR is also used as a *touch* surrogate via a 1 cm exclusion zone — encroachment is registered as contact with two measurements (initial impact force, continued pressure).

**C. Physiology.** Three homeostatic variables $H = \{ \text{Energy}, \text{Condition}, \text{Temperature} \}$, each with an ideal value and a viability bound (Table 1). Energy declines at constant basal metabolic rate $\text{BMR} = 0.1$. Condition is degraded by collisions. Temperature obeys

$$
\dot R_{\text{temp}} = T \cdot \frac{v_{\text{current}}}{v_{\text{max}}} - c_d, \qquad (1)
$$

with $T$ a climate-driven generation coefficient, $c_d$ a constant dissipation rate. Wellbeing is the combined viability-style measure (Ashby 1952, ref 18).

**D. Hormone System.** Four hormones split into two groups (Table 2). Three *endocrine* hormones $E_1, C_1, T_1$ each tied to one homeostatic deficit; secretion obeys

$$
H_{Eh}(t) = H_{Eh}(t-1) + \theta_g \cdot \sigma \cdot d_i, \qquad (2)
$$

where $\theta_g$ is the gland's activity level, $\sigma$ scales secretion size, and $d_i$ is the current homeostatic deficit. Each secretion persists in the "bloodstream" for a random number of action loops $du \in [\min, \max]$ before decay. One *neuro-hormone* $D_1$ facilitates aggressive behaviour; its secretion is triggered by mean visual environmental-cue size $\overline{ec}$:

$$
H_{Nh}(t) = H_{Nh}(t-1) + \theta_g \cdot \xi \cdot \overline{ec} - \mu \, H_{Nh}(t-1), \qquad (3)
$$

with $\mu$ a constant dispersion rate and $\xi$ a predetermined weighting. Unlike for endocrine hormones, $\theta_g$ for $D_1$ is itself driven by mean concentration of the endocrine hormones $\overline{Eh}$ through a Gompertz / sigmoid function:

$$
\theta_g(t) = \exp\!\big( -4 \cdot \exp(-0.5 \cdot \overline{Eh}) \big), \qquad (4)
$$

bounded between 0 (gland inactive) and 1 (fully active). Final hormone amount is filtered through receptor sensitivity:

$$
H_h(t) = \text{sens}_h \cdot H_h(t), \qquad (5)
$$

where $\text{sens}_h$ is the per-hormone receptor sensitivity.

**E. Hormones and the Action-Selection Mechanism (ASM).** Two-layer voting architecture (Avila-García, Cañamero, Boekhorst, ref 27). Layer 1 computes motivations from hormone levels (not directly from deficits):

$$
M_i = \begin{cases} M_E = E_1 + \text{cue}_E - T_1 \\ M_C = C_1 + \text{cue}_C - T_1 \\ M_T = 1 \text{ if } T_1 > m_t \, T_1^{\max}, \text{ else 0} \end{cases} \qquad (6)
$$

with $m_t = 0.75$ a constant. $M_T$ (hyperthermia) acts as a *suppressor* — it pushes the velocity down via temperature dissipation rather than competing for behaviour selection. $M_E$ is given a small restless bonus and a 10 % hysteresis bonus when active to prevent oscillation. Layer 2 selects behaviour given current motivational state + environment. Unlike earlier work (refs 17, 20), behaviours are not pre-modeled but emerge from dynamic combinations of sub-systems with no preset cost/gain. The key sub-system is the *personal space* (Ps) radius (Hall 1966, ref 21) — an area the robot treats as an extension of its body. The Ps radius is hormone-modulated:

$$
P_s = n_s + n_s \cdot \frac{D_1 - E_1 - C_1}{Eh_{\max}}, \qquad (7)
$$

so high $D_1$ shrinks $P_s$ (tolerates contact, enables aggression); high endocrine hormones grow $P_s$ (avoids contact when needs are pressing without aggressive drive).

**F. Hormone-Signalled Epigenetics.** The mechanism that distinguishes this paper from the 2004 base architecture. During the *early life window* (first 3 min = 2880 steps at 16 Hz), gland activity $\theta_g$ and receptor sensitivity $\text{sens}_h$ themselves drift:

$$
\theta_g(t+1) = \theta_g(t) + k \cdot \frac{H_h(t)}{H_h^{\max}}, \qquad (8)
$$

$$
\text{sens}_h(t+1) = \text{sens}_h(t) + k \cdot \frac{H_h(t)}{H_h^{\max}}, \qquad (9)
$$

with $k$ a predetermined epigenetic-rate constant (different between Eqs. 8 and 9). The intuition: glands that release more hormone (because the relevant deficit keeps appearing in this environment) grow stronger; receptors that keep seeing high hormone levels become more sensitive. After 3 minutes the updates stop and the parameters are frozen at their developed values.

**Preliminary case.** In an arena with fast-moving energy sources, the non-epigenetic robot is slower than its prey and only "takes bites" on near-passes, suffering homeostatic crashes until $D_1$ rises enough to tolerate dangerous overheating. The epigenetic robot, by contrast, develops an *ambush predator* strategy — stay sedentary until prey approaches, then pounce at full speed and pin to a wall (Fig. 4 in paper).

### Section III — Experiments

35 runs across three scenarios (10/10/15), each run 10,000 steps ≈ 10 min 40 s. Epigenetic window = first 3 min. Performance metrics from Avila-García & Cañamero 2004: *comfort* (1 − average normalized homeostatic deficit) and *risk of death* (closeness to lethal values). Baseline = same architecture without the epigenetic updates of Eqs. 8–9.

**Scenario 1 (open arena, low temperature, one resource per type).** Both robots perform adequately. The epigenetic model is slightly better in late-game comfort. Both develop a *tight-circling* behaviour around resources when $E_1/C_1$ lags behind real deficits.

**Scenario 2 (movable obstacles around resources, low temperature).** Non-epigenetic baseline fails — 3 of 10 runs die around mid-game (starvation or last-ditch desperation runs); high standard deviation. Epigenetic robot adapts: perceiving visible-but-inaccessible resources raises $D_1$, which raises receptor sensitivity to $D_1$ via Eq. 9, which lowers $P_s$ via Eq. 7, which enables aggressive pushing-through of obstacles. Pushing through creates condition deficits, which grow the $C_1$ gland via Eq. 8, which raises the priority of replenishing condition next time — closing a feedback loop.

**Scenario 3 (uneven resource density: 5 energy + 1 repair, or 1 energy + 5 repair, or hot temperature).** Non-epigenetic baseline distracted by the over-abundant resource; rare-resource deficit averages 32 % higher than the abundant resource's deficit before the rare one is approached. Epigenetic model: early sustained deficits on the rare resource grow the corresponding gland, raising sensitivity to that deficit; the threshold shifts so that the abundant resource deficit must be 42 % higher (or in 4 cases of resource-blocking, fall to 60–80 %) before attention is diverted. Hot-temperature subset shows no comfort difference but a qualitative change: the epigenetic robot develops a *stalking* gait — move toward resource, break, burst forward at full speed within range — that generates less overall heat than the constant-speed baseline.

### Section IV — Conclusion
Distinguishes two forms of epigenetic adaptation: (a) on/off switches for gene expression with predictable phenotypic outcomes (refs 5,6,7), (b) subtle changes in neural and endocrine systems whose phenotype varies with experience (refs 9–12). The paper introduces the latter form. The epigenetic mechanism significantly improves adaptability across the three scenarios and produces qualitatively new behaviours. Rebuts the criticism that the baseline was unfair (because it had no environment-specific tuning): the *point* is to demonstrate that the epigenetic robot can be dropped into an unknown environment and not just survive but develop behaviours that thrive in its niche.

## Phase 1 — undergraduate-level synthesis

**The plain idea.** Take the standard Cañamero-school robot (a Koala with three needs — energy, condition, temperature — and four simulated hormones). Don't pre-tune the hormone glands and receptors to the environment. Instead, run a brief "childhood" — say the first three minutes — during which the *activity of each hormone gland* and the *sensitivity of each receptor* both **integrate** the hormone concentration they actually experience. Glands that fire a lot in this environment grow stronger; receptors that see a lot of hormone become more sensitive. After three minutes, freeze the values. The robot is now tuned to where it lives.

**Why this matters.** Pre-tuning a hormone system to a *specific* environment works if the environment is known. But the moment you place the same robot in a different environment — say one full of obstacles that block the resources — the pre-tuned hormone weights are wrong: aggression is too low (the robot can't push through), receptors are too sluggish, glands are too quiet. The epigenetic mechanism replaces hand-tuning with a brief self-tuning phase.

**The setup.** Koala robot in a 2×2 m arena with two coloured-ball resources. Three scenarios: open arena, obstacle-blocked resources, uneven resource density. 35 runs of about 10 min each. Two arms: with vs. without the epigenetic updates of Eqs. 8 and 9. Metrics: comfort (1 − average homeostatic deficit) and risk of death (closeness to viability bounds).

**The result.** In scenario 1 (easy environment) both architectures cope; the epigenetic model is marginally better. In scenarios 2 (obstacles) and 3 (uneven resources) the baseline fails — in scenario 2 three robots die mid-game — while the epigenetic robot adapts. More interesting than the comfort numbers: the epigenetic robot develops *new behaviours* the designer never wrote. In the preliminary "fast prey" experiment it learns a pounce / ambush pattern; in scenario 3 it learns a *stalk and burst* gait that minimizes overheating. The architecture is exactly the same as the baseline — only the early-life parameter updates differ.

**Concrete instantiation.** Eqs. 8 and 9 are the heart of the paper:

$$
\theta_g(t+1) = \theta_g(t) + k \cdot \frac{H_h(t)}{H_h^{\max}}, \qquad \text{sens}_h(t+1) = \text{sens}_h(t) + k \cdot \frac{H_h(t)}{H_h^{\max}},
$$

run for the first 2880 time steps and then frozen. That is it. Every other equation in the architecture is inherited from Avila-García & Cañamero 2004 and from Cañamero 1997.

## Phase 2 — graduate-level deep dive

### Hormone dynamics — endocrine vs. neuro-hormone

Three endocrine hormones $E_1, C_1, T_1$ corresponding to energy / condition / temperature deficits. For each:

$$
H_{Eh}(t) = H_{Eh}(t-1) + \theta_g \cdot \sigma \cdot d_i, \qquad (2)
$$

with $d_i = \max(0, V_{\text{opt},i} - V_i(t))$ the current normalized deficit, $\sigma = 0.09$ (Table 2) the secretion scale. Each pulse persists for a random duration $du$ uniformly drawn from $[du_{\min}, du_{\max}]$ before deletion. The "bloodstream" is therefore a list of decaying pulses, summed at each tick. Note Eq. 2 is incremental, not differential — Lones uses a discrete-time additive formulation rather than a continuous ODE.

The neuro-hormone $D_1$ obeys

$$
H_{Nh}(t) = H_{Nh}(t-1) + \theta_g \cdot \xi \cdot \overline{ec} - \mu \, H_{Nh}(t-1), \qquad (3)
$$

with $\overline{ec}$ the mean visual environmental-cue magnitude (computed from the camera + OpenCV via the area-weighted blob average). $\mu$ governs steady dispersion, $\xi$ scales secretion. The Gompertz function

$$
\theta_g^{D_1}(\overline{Eh}) = \exp\!\big( -4 \cdot \exp(-0.5 \cdot \overline{Eh}) \big), \qquad (4)
$$

couples the neuro-hormone's *gland activity* to the mean concentration of the endocrine hormones $\overline{Eh}$. Properties: $\theta_g^{D_1}(0) = e^{-4} \approx 0.018$; $\theta_g^{D_1}(\infty) = 1$. The Gompertz is monotone-increasing with an inflection around $\overline{Eh} \approx \log 2 / 0.5 \approx 1.39$. So $D_1$ stays nearly silent until endocrine hormones are non-trivially elevated, then ramps up to full aggression. This is the tropic-hormone analogue Lones cites (Sherwood 2003, ref 23) — endocrine pressure summoning a meta-hormonal response.

Receptor filtering closes the loop:

$$
\tilde H_h(t) = \text{sens}_h \cdot H_h(t), \qquad (5)
$$

so the hormone seen by downstream consumers (motivations, personal space, etc.) is gain-modulated by $\text{sens}_h$.

### Personal-space modulation

Eq. 7 makes $P_s$ a *linear function of the hormone balance*:

$$
P_s = n_s + n_s \cdot \frac{D_1 - E_1 - C_1}{Eh_{\max}},
$$

with $n_s$ the *normal*, unadjusted personal-space radius. Two limits:

- $D_1 \to Eh_{\max}$, $E_1, C_1 \to 0$: $P_s \to 2 n_s$ (radius doubles → spacious posture). This is *not* the aggressive mode; it is the calm-curious mode.
- $D_1 \to 0$, $E_1 \to Eh_{\max}$, $C_1 \to Eh_{\max}$: $P_s \to -n_s$ (radius goes *below* zero — the robot tolerates encroachment, i.e., aggressive contact-seeking). The negative-$P_s$ regime is the operational realization of *aggression suppresses personal space*.

Re-reading the paper text against Eq. 7: the intended reading is the second limit. The signed-difference $D_1 - E_1 - C_1$ should be interpreted carefully — when *all three* are at maximum, the negative endocrine terms dominate, $P_s \to -n_s$, and the robot pushes through obstacles to reach resources.

### Action selection

Motivations are computed from hormones (not deficits — a departure from Cañamero 1997 and Avila-García & Cañamero 2004 in which deficits drove motivations directly):

$$
M_E = E_1 + \text{cue}_E - T_1, \qquad M_C = C_1 + \text{cue}_C - T_1, \qquad M_T = \mathbb{1}\{ T_1 > m_t \cdot T_1^{\max} \},
$$

with $\text{cue}_i$ the visual evidence for a resource of type $i$ in the field of view, $T_1$ acting as a suppressor on $M_E, M_C$ (i.e., overheating overrides hunger and damage drives, encouraging slowing). The 10 % hysteresis bonus on the currently active motivation prevents oscillation; the "restless" +1 bonus on $M_E$ ensures the robot moves even with no current cue. Behaviour selection then disinhibits the maximum-vote motivation and runs a behaviour-arbitration over sub-systems (movement direction, speed, aggression vs. retreat).

### Epigenetic update rule

The novelty of the paper (Eqs. 8–9):

$$
\theta_g(t+1) = \theta_g(t) + k_\theta \cdot \frac{H_h(t)}{H_h^{\max}}, \qquad t < T_{\text{early}}, \qquad (8)
$$

$$
\text{sens}_h(t+1) = \text{sens}_h(t) + k_s \cdot \frac{H_h(t)}{H_h^{\max}}, \qquad t < T_{\text{early}}, \qquad (9)
$$

with $T_{\text{early}} = 2880$ steps (3 min at 16 Hz), and $k_\theta, k_s > 0$ small per-step constants (different values for the two equations — the paper notes "the value of k is different for [Eqs.] 13, 14" which in the paper's numbering is Eqs. 8, 9). After $T_{\text{early}}$ the updates are switched off and $\theta_g, \text{sens}_h$ are held fixed for the remainder of the run.

#### Integration interpretation

Integrating Eq. 8 over the early-life window gives

$$
\theta_g(T_{\text{early}}) = \theta_g(0) + \frac{k_\theta}{H_h^{\max}} \cdot \int_0^{T_{\text{early}}} H_h(t) \, dt.
$$

So gland activity at the end of childhood is proportional to the *time integral of hormone exposure during childhood*. This is a discrete-time analogue of a biological "developmental imprint": a gland that has been worked hard during the critical window stays strong for adult life. Same shape for receptor sensitivity in Eq. 9 — a critical-window integration of input.

#### Stability

There is no explicit homeostasis on $\theta_g$ or $\text{sens}_h$ — both are monotonically non-decreasing within the critical window. Stability is enforced only by the time-limited integration: $T_{\text{early}}$ is small enough, and $k_\theta, k_s$ small enough, that runaway is unlikely. The choice not to bound $\theta_g \le 1$ explicitly (the paper does not state a clip) is a minor implementation concern — but Eqs. 4 and 5 saturate downstream, so any blow-up of $\theta_g$ would be capped by the Gompertz of Eq. 4 in the $D_1$ case and by $H_h^{\max}$ in the sensitivity case.

#### Feedback loops produced

The architecture closes two distinct feedback loops in scenarios 2 and 3:

**Loop A (obstacle scenario).** Visible-but-inaccessible resources → $\overline{ec}$ remains high → $D_1$ rises via Eq. 3 → $\text{sens}_{D_1}$ grows via Eq. 9 → effective $\tilde D_1$ amplifies → $P_s$ shrinks via Eq. 7 → robot pushes through obstacles → condition deficit appears → $E_1$-style $C_1$ rises via Eq. 2 → $\theta_g^{C_1}$ grows via Eq. 8 → future $C_1$ responses are larger → priority of replenishing condition is raised in $M_C$ via Eq. 6.

**Loop B (uneven-resource scenario).** Persistent under-supply of the rare resource → corresponding $E_1$ or $C_1$ stays high → $\theta_g$ for that hormone grows via Eq. 8 → secretion to subsequent deficits is amplified → the agent attends to the rare deficit at lower deficit magnitudes (the 42 % shift in attention threshold reported in §III.C).

Both loops are emergent — neither is hand-engineered.

### Pseudocode

```
Initialize: V_E = 100, V_C = 100, V_T = 0
           theta_g[E1] = theta_g[C1] = theta_g[T1] = theta_0
           sens[E1] = sens[C1] = sens[T1] = sens[D1] = sens_0
           hormone_levels = empty list of decaying pulses

At each tick t in {0, ..., T_run}:
  # 1. Physiology
  V_E -= BMR
  if contact: V_C -= contact_force
  V_T += T * v_current / v_max - c_d           # Eq. 1
  # 2. Compute deficits and secrete endocrine hormones
  for hormone h in {E1, C1, T1}:
    d_h = max(0, V_opt[h] - V[h])
    pulse_size = theta_g[h] * sigma * d_h       # Eq. 2
    hormone_levels[h].append(pulse_size, duration=random(du_min, du_max))
  # 3. Compute D1 (neuro-hormone)
  ec_bar = mean(visual_cues_from_OpenCV)
  Eh_bar = mean(hormone_levels[E1], hormone_levels[C1], hormone_levels[T1])
  theta_g[D1] = exp(-4 * exp(-0.5 * Eh_bar))    # Eq. 4
  D1 += theta_g[D1] * xi * ec_bar - mu * D1     # Eq. 3
  # 4. Receptor filtering
  for h in hormones:
    H_tilde[h] = sens[h] * H[h]                 # Eq. 5
  # 5. Motivations (Eq. 6)
  M_E = H_tilde[E1] + cue_E - H_tilde[T1] + restless + hysteresis_bonus
  M_C = H_tilde[C1] + cue_C - H_tilde[T1] + hysteresis_bonus
  M_T = 1 if H_tilde[T1] > m_t * T1_max else 0
  # 6. Personal space (Eq. 7)
  P_s = n_s + n_s * (H_tilde[D1] - H_tilde[E1] - H_tilde[C1]) / Eh_max
  # 7. Behaviour selection
  motivation = argmax(M_E, M_C, M_T)
  execute_behaviour(motivation, P_s, environment)
  # 8. EPIGENETIC UPDATES (only in critical window)
  if t < T_early:
    for h in {E1, C1, T1, D1}:
      theta_g[h] += k_theta * H_tilde[h] / H_max[h]   # Eq. 8
      sens[h]    += k_s     * H_tilde[h] / H_max[h]   # Eq. 9
  # else: theta_g and sens frozen.
```

### Mathematical commentary

Three structural choices repay reflection:

1. **Critical-window integration is a one-time blackboard update.** Unlike continuous adaptation, the rule has a sharp "child → adult" transition at $T_{\text{early}}$. This makes the adult parameter $\theta_g(T_{\text{early}})$ a *sufficient statistic* of childhood hormone exposure. Behaviorally this is exactly the **prenatal-stress-imprint** family of phenomena cited in the introduction (Maccaria et al. 2003; Oberlander et al. 2008; Karst et al. 2010).

2. **Same update rule for gland and receptor.** Eqs. 8 and 9 differ only in $k_\theta$ vs. $k_s$. Both push $\theta_g$ and $\text{sens}_h$ in the same direction (up, monotone). Yet downstream they act multiplicatively (Eq. 5: $\tilde H_h = \text{sens}_h \cdot H_h$, and Eq. 2: $H_{Eh} \propto \theta_g$). The combined effect on $\tilde H_h$ is roughly quadratic in childhood hormone exposure — a sharper-than-linear gain on the developmental signal.

3. **No designer-set environment-specific tuning.** This is the methodological core of the paper. The conclusion's rebuttal — that the comparison is "unfair" because the non-epigenetic baseline lacks environment-specific tuning — is *the contribution*. The architecture replaces hand-tuning with a brief automatic developmental phase that *implements* hand-tuning under environmental supervision. This connects to the broader meta-learning / developmental-RL literature (Doya 2002; Wang et al. 2024) — the curator should cross-link.

## Connections

- **Direct architectural ancestor.** `canamero_1997_motivations_emotions.md` — the homeostatic-physiology + hormone-modulation backbone.
- **Hormone-modulated action-selection ancestor.** Avila-García & Cañamero 2004 (ref 17 in this paper) — the two-layer voting ASM with hormones is inherited essentially unchanged; this paper adds Eqs. 8–9 and the critical window.
- **Affect-modulated PerAc cousin.** `blanchard_canamero_2006_affect_modulated.md` — the same lab's affect/well-being modulator architecture, addressed at the much-faster *behaviour-arbitration* time scale rather than the much-slower *developmental* time scale.
- **Affordance-learning cousin.** `cos_2010_affordances_consummatory.md` — also extends the Cañamero homeostatic backbone with a learning layer (Hebbian on affordances), but at the *adult* time scale rather than during a critical window. The two architectures are complementary: the present paper learns *hormone-system parameters*, Cos et al. 2010 learn *object-affordance synapses*.
- **Direct successor.** `lones_2018_hormone_epigenetic.md` — Lones et al. 2018 is the journal version that extends, refines, and re-runs the experiments of this 2013 conference paper. The two papers should be read together.
- **Methodological framing.** `canamero_2005_emotion_understanding.md` — Cañamero's 2005 review flagged developmental/evolutionary models of emotion grounding as the way to address the *origins problem*; this paper is one of the most direct realizations of that programme in the corpus.
- **Cross-corpus links** (likely in other batches). Krichmar 2012 (ref 20 — *A biologically inspired action selection algorithm based on principles of neuromodulation*), French & Cañamero 2005 (ref 24 — *Introducing neuromodulation to a Braitenberg vehicle*), and the broader Krichmar neuromodulation programme (Cox & Krichmar 2009; Avery & Krichmar 2017) are the natural cross-corpus neighbours.
