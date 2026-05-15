---
title: "A Hormone-Driven Epigenetic Mechanism for Adaptation in Autonomous Robots"
authors: ["John Lones", "Matthew Lewis", "Lola Cañamero"]
year: 2018
venue: "IEEE Transactions on Cognitive and Developmental Systems 10(2), 445–454 (June 2018)"
slug: lones_2018_hormone_epigenetic
source_pdf: "sources/Lones et al. 2018 - A Hormone-Driven Epigenetic Mechanism for Adaptation in Autonomous Robots.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This 2018 IEEE Transactions paper is the journal extension of `lones_canamero_2013_epigenetic_hormones.md`, with three new experiments and a sharper formal definition of the epigenetic update. The proposal: give a Koala robot a body with three needs (energy, health, temperature) and four simulated hormones — three *endocrine* hormones $E_1, H_1, T_1$ tied to each deficit, plus one *neurohormone* $D_1$ that suppresses negative stimuli (collisions, overheating). Each hormone has a **receptor with a sensitivity** $\text{Sens}_h$. The novelty is a single one-line update rule:

$$
\text{Sens}_i(t+1) = \text{Sens}_i(t)^{\,E_h^i / \sigma},
$$

where $\sigma$ is a small constant. The hormone level *exponentially* raises or lowers its own receptor's sensitivity — high hormone → upregulation, low hormone → downregulation. This is the closest the Cañamero-school architectures get to modeling DNA methylation's effect on hormone-receptor expression (Fowden & Forhead 2009; Crews 2010; Zhang & Ho 2011). Three robot tiers are compared in six environments — *basic*, *hormone-modulated*, *epigenetic* (with the new Eq. 12). Across the six environments (open arena, movable obstacles, moving "prey" resources, dynamic day/night climate cycle, uneven resource densities, brief temporal-window resources), the epigenetic robot consistently outperforms the two simpler baselines on Avila-García & Cañamero's *comfort* metric and develops a strikingly varied behavioural repertoire — none of which is programmed in. The emergent behaviours include **pushing-through-obstacles**, **ambush-style pouncing** on fast prey, **hibernation** during the hot phase of a day/night cycle, and **stalking** in temperature-constrained scenarios. The paper is significant in the corpus because (i) it cleanly demonstrates that a *single* receptor-update rule produces *qualitatively different phenotypes* matched to environment, and (ii) it argues that this kind of receptor-level adaptation — not just hormone-level modulation — is what gives biological organisms their robustness and what synthetic agents need too.

## Section-ordered backbone

### Abstract
Different epigenetic mechanisms allow biological organisms to adjust physiology / morphology and adapt. The paper investigates whether an epigenetic model in which hormone concentrations are linked to the regulation of hormone receptors can provide a robust and general adaptive mechanism for autonomous robots. Implementation on a Koala robot, six environments. Results show emergence of varied behaviours tailored to each environment.

### Section I — Introduction
Frames the gap left by Krichmar 2012: most controllers are task-specific and brittle under environmental change (refs 2–4). Argues that an epigenetic process in which environmental stimuli affect hormone receptors could provide a useful general mechanism. The mechanism: hormone level regulates the sensitivity of *its own receptor* via a positive feedback loop — high concentration → upregulation, low concentration → downregulation. Six environments will test allostatic adaptation. The paper explicitly extends Lones & Cañamero 2013 (ref 13, i.e. `lones_canamero_2013_epigenetic_hormones.md`) — adding new experimental conditions and a more rigorous formal model.

### Section II — Robot Model

Three layered sub-architectures (Fig. 1):

1. **Basic architecture** (Section II-A) — drives, motivational state, movement.
2. **Neuromodulatory system** (Section II-B) — four simulated hormones modulate the basic architecture.
3. **Epigenetic mechanism** (Section II-C) — receptor sensitivity is dynamically regulated by hormone levels.

This produces three robot variants for ablation: *basic*, *hormone-modulated* (basic + neuromodulatory), *epigenetic* (all three).

**A. Basic Architecture.** Three homeostatic variables — energy, health, temperature, each in $[0, 100]$. Energy: ideal 100, fatal 0, loses 1 unit/s, replenished by approaching pink ball within 2 cm. Health: ideal 100, fatal 0, decreased by contact via the 16 IR sensors:

$$
\text{Health} = \sum_i \begin{cases} -\text{Ir}_i / \text{Ir}_{\max} & \text{if } \text{Ir}_i > 0.95 \cdot \text{Ir}_{\max} \\ 0 & \text{otherwise} \end{cases}, \qquad (1)
$$

replenished by approaching blue ball. Temperature: ideal 0, fatal 100, error is excess only:

$$
\text{Temperature} = \frac{|sp|}{10} \cdot \text{Climate} - (\text{Temperature} \cdot 0.05), \qquad (2)
$$

with $sp$ wheel speed in rotations per loop, $\text{Climate}$ ambient. Two explicit drives:

$$
\text{Drive}_i = \frac{\text{Deficit}_i}{\text{Error}_t}, \qquad (3)
$$

for $i \in \{e, h\}$, divided by temperature error so that overheating throttles all other drives. The motivational state is *not* a discrete winner-takes-all (departing from the 1997, 2004, 2010, 2013 Cañamero-school architectures), but a per-direction vector across eight body-centric directions:

$$
\text{MotState}_j = \sum_{i \in \{e, h\}} \sum_c s_{i,c} \cdot (\text{Drive}_i \cdot \text{Cue}_{c,j}) - \text{Avoid}_j, \qquad (4)
$$

with $s_{i,c}$ the scaling factors of Table I (drive-to-cue type mapping), $\text{Cue}_{c,j}$ the magnitude of environmental cue type $c$ in direction $j$, and

$$
\text{Avoid}_j = \begin{cases} \text{Ir}_j & \text{if } \text{Ir}_{\max} - \text{Ir}_j < \text{Drive}_h \\ 0 & \text{otherwise} \end{cases}. \qquad (5)
$$

Wheel speeds:

$$
\text{WheelSpeed}_k = \sum_{j=0}^{7} \big( \text{MotState}_j \cdot \text{Set}_{k,j} \big), \qquad (6)
$$

with $\text{Set}_{k,j}$ vectors converting eight-direction motivational state into a left/right wheel command.

**B. Neuromodulatory System.** Four simulated hormones in two groups (Table II). Three endocrine hormones $E_1, H_1, T_1$ track the three deficits via

$$
\text{EhSecretion}_i = \psi_i \cdot \text{Error}_i, \qquad (7)
$$

each pulse persisting for a random number of loops before decay. One neurohormone $D_1$ secreted in proportion to total drive intensity:

$$
\text{NhSecretion} = \sum_i \text{Drive}_i. \qquad (8)
$$

Hormones modulate drives:

$$
\text{Drive}_i = \frac{\text{Sens}_i \cdot E_h^i}{\text{Sens}_t \cdot E_h^t}, \qquad (9)
$$

for $i \neq t$, with $\text{Sens}_h$ the per-hormone receptor sensitivity (in the hormone-modulated robot, $\text{Sens}_h = 1$ constant; in the epigenetic robot, $\text{Sens}_h$ is updated by Eq. 12). $D_1$ suppresses avoidance:

$$
\text{Avoid}_j = \begin{cases} \text{Ir}_j / N_h & \text{if } \text{Ir}_{\max} - \text{Ir}_j < \text{Drive}_h \\ 0 & \text{otherwise} \end{cases}, \qquad (10)
$$

and amplifies wheel speed:

$$
\text{WheelSpeed}_k = \sum_{j=0}^{7} \big( \text{MotState}_j \cdot \text{Set}_{k,j} \big) \cdot \text{Sens}_{N_h} \cdot N_h. \qquad (11)
$$

So high $D_1$ → lower effective avoidance and higher wheel speed → "aggressive / dominant" behaviour.

**C. Epigenetic System.** The key new equation:

$$
\text{Sens}_i(t+1) = \text{Sens}_i(t)^{\,E_h^i / \sigma}, \qquad (12)
$$

with $\sigma$ a small predetermined constant. The hormone exponent regulates its own receptor. The functional form gives a *positive feedback*: when $E_h^i > \sigma$, $\text{Sens}_i$ is raised above 1; when $E_h^i < \sigma$, $\text{Sens}_i$ falls toward 0. Receptor regulation values are tabulated in Table IV per experiment.

### Section III — Experiments

Six environments in a 2×2 m wooden arena. Each of the first four experiments: 10 runs of 10,000 steps (16 Hz, ≈ 10 min) per robot tier. Experiments 5–6: 15 runs (5 per sub-condition). Performance metric: *comfort* (normalized mean of energy, health, $100 - \text{temperature}$), from Cañamero & Avila-García 2007 (ref 9). Death sets future comfort = 0.

**A. Experiment 1 — Basic environment.** Simple, low temperature. All three robots survive (Fig. 4). Epigenetic robot stays close to an energy resource at all times — $H_1$ and $T_1$ receptors downregulate because the environment generates little health/temperature pressure; the epigenetic robot's stationary "guard" strategy almost eliminates the other drives.

**B. Experiment 2 — Movable obstacles** (around resources). Basic robot dies in all 10 runs; hormone-modulated has 3 deaths; epigenetic survives (Fig. 5). The epigenetic robot develops a *push-through-obstacles* behaviour via a chain reaction: persistent visible-but-blocked resources → high $D_1$ → upregulation of $D_1$ receptor → effective avoidance suppressed → robot pushes through → collision damage → $H_1$ rises → upregulation of $H_1$ receptor → motivation to replenish health rises → robot consistently maintains enough health to absorb future pushes. Average push-speed: 230 mm/s (epigenetic) vs. 150 mm/s (hormone-modulated) vs. 60 mm/s (basic).

**C. Experiment 3 — Moving "prey" resources** (resources move at the default robot speed). Pure chasing fails. The epigenetic robot develops an *ambush-pounce*: stay sedentary until a resource passes nearby, then burst at full speed and pin to a wall. Mechanism: early chases overheat → upregulation of $T_1$ receptor → strong suppression of movement → low baseline temperature → bursts can hit full speed without overheating. Success rates: 87% (epigenetic) vs. 72% (hormone-modulated) vs. 13% (basic). Average chase length: 4 s (epigenetic) vs. 14 s (hormone) vs. 12 s (basic).

**D. Experiment 4 — Dynamic climate (day/night cycle).** Basic robot dies before the second cycle. Hormone-modulated robot survives in 7/10 runs. Epigenetic robot survives all 10 with two distinct phenotypes: in 7 runs it develops *hibernation* — fully replenishes during cold periods, lies dormant during hot — via upregulation of $T_1$ receptor + $E_1$ + $H_1$ receptors; in 3 runs it develops *stay-near-energy* — minimal movement, only briefly leaves to repair when needed, via upregulation of $T_1$ + $E_1$ only. Which phenotype the robot adopts depends on early-life health loss: if the first cycles include collisions, hibernation; if not, stay-near-energy. Once adopted, the phenotype persists.

**E. Experiment 5 — Uneven resources** (3 sub-conditions: extra energy, extra repair, hot climate). Basic robot performs poorly. Hormone-modulated robot survives but with rare-resource deficit 32 % greater than abundant-resource deficit on average. Epigenetic robot adapts: rare-resource hormone upregulates rare-resource receptor → robot attends to rare-resource deficit at lower magnitudes. The 32 % gap becomes 42 % before attention is reallocated; when the rare resource is hidden behind another, the common-resource deficit must drop to 60–80 % before the rare-search is broken. In the hot-climate subset the epigenetic robot develops *stalking* — same upregulation pattern as the pounce but milder.

**F. Experiment 6 — Temporal-window resources** (energy resource appears only briefly, with window shrinking from 30 s in sub-condition 1 to a few seconds in sub-condition 3). Basic robot dies in all runs. Hormone-modulated reaches the temporal resource 62 % of the time; epigenetic reaches it 84 %. Mechanism: rapid upregulation of $E_1$ receptor → robot overrides health needs to take advantage of the brief energy window → only 1 epigenetic death vs. 7 hormone-modulated deaths.

### Section IV — Conclusion
The epigenetic robot consistently outperforms the basic and hormone-modulated baselines on comfort, with much lower run-to-run variance. The improvement comes from receptor regulation: changes to receptors create *tolerances* and *sensitivities* tuned to the environment, which produce qualitatively distinct behavioural phenotypes (guard, push, pounce, hibernation, stay-near, stalking, opportunistic-feeding) — none designed in. The mechanism is general: the same receptor-update rule (Eq. 12) handles six different environments. The paper notes external applications: human-robot interaction (ref 16, 24), group formation (ref 25), neural-network learning modulation (ref 16).

## Phase 1 — undergraduate-level synthesis

**The plain idea.** The 2013 conference precursor (`lones_canamero_2013_epigenetic_hormones.md`) showed that a robot can self-tune its hormone glands during a 3-minute "childhood" window and then survive in environments the designer didn't anticipate. This 2018 journal paper extends the same architecture in one critical way: instead of a brief critical window with two parallel updates for *gland activity* and *receptor sensitivity*, the new version uses a single *receptor-only* update that runs throughout life:

$$
\text{Sens}_i \leftarrow \text{Sens}_i^{\,E_h^i / \sigma}.
$$

If the hormone $E_h^i$ is currently high (above $\sigma$), the exponent is greater than 1 — the receptor sensitivity gets squared, cubed, etc. — and **upregulates**. If $E_h^i$ is low (below $\sigma$), the exponent is less than 1 — the receptor gets fractional — and **downregulates**. This single rule produces a positive feedback loop: high hormone → more sensitive receptor → bigger effective signal → stronger downstream drive.

**Why it matters.** Compared to a robot that pre-fixes its receptor sensitivities, the epigenetic robot can *grow* into the environment. Drop it in an obstacle-strewn world and within minutes its $D_1$ receptors upregulate, $H_1$ receptors upregulate, and it learns (without learning anything explicit) to push through obstacles to feed. Drop the same robot in a sun-bakedclimate-cycling world and it discovers *hibernation* — a behaviour the designer literally never wrote. Across the six environments tested, the epigenetic robot wins every time and shows the lowest run-to-run variance.

**The setup.** A Koala robot in a 2 × 2 m arena. Three resources: pink ball (energy), blue ball (health/repair), and the climate. Three robot tiers — basic, hormone-modulated, epigenetic — tested in six environments. Performance: comfort (normalized mean of energy + health + (100 − temperature)).

**The result.** In all six environments the epigenetic robot is the most adaptive — sometimes by a small margin (Experiment 1, easy), sometimes by a huge margin (Experiment 2: 0 deaths vs. 10 for basic). The qualitative emergent behaviours — pushing, pouncing, hibernation, stalking, stay-near-energy, opportunistic-feeding — are the headline. Crucially, two robots with *identical* initial parameters in the day/night-cycle environment can develop *different* phenotypes (hibernation vs. stay-near) based on whether their early-life cycles included collisions. Phenotype is path-dependent on experience — a feature of biological epigenetic development.

## Phase 2 — graduate-level deep dive

### The receptor-update rule (Eq. 12)

The core rule is

$$
\text{Sens}_i(t+1) = \text{Sens}_i(t)^{\,E_h^i(t) / \sigma}. \qquad (12)
$$

Taking $\log$ of both sides:

$$
\log \text{Sens}_i(t+1) = \frac{E_h^i(t)}{\sigma} \cdot \log \text{Sens}_i(t).
$$

So $\log \text{Sens}_i$ obeys a multiplicative-noise random walk where the multiplier is the current hormone over $\sigma$.

#### Equilibrium and bifurcation

A fixed point of Eq. 12 is any $\text{Sens}^* \in \{1\}$ (since $1^x = 1$ for any $x$), plus 0 and $\infty$ as degenerate fixed points. The stability of $\text{Sens}^* = 1$ depends on whether $E_h^i / \sigma$ averages to a value above or below 1:

- If $\langle E_h^i / \sigma \rangle < 1$ (low chronic hormone), $\text{Sens}_i$ decays toward 0 (downregulation).
- If $\langle E_h^i / \sigma \rangle > 1$ (high chronic hormone), $\text{Sens}_i$ grows toward $\infty$ (upregulation).
- If $\text{Sens}_i$ is currently $< 1$ and $E_h^i / \sigma > 1$, the exponent makes $\text{Sens}_i$ rise toward 1 (recovery from undersensitization).
- If $\text{Sens}_i$ is currently $> 1$ and $E_h^i / \sigma < 1$, the exponent makes $\text{Sens}_i$ fall back toward 1 (recovery from oversensitization).

So **$\sigma$ acts as a threshold-hormone-level above which the receptor upregulates and below which it downregulates**, with the relaxation always pointed at $\text{Sens}^* = 1$ unless the hormone level is held away from $\sigma$ chronically.

#### Drive equation under the rule

Plugging into Eq. 9:

$$
\text{Drive}_i = \frac{\text{Sens}_i \cdot E_h^i}{\text{Sens}_t \cdot E_h^t}.
$$

After receptor regulation, an environment that chronically elevates $E_h^i$ will *also* chronically raise $\text{Sens}_i$ — producing a **multiplicative amplification** of the drive on top of the linear effect of $E_h^i$. If $\text{Sens}_i$ rises from 1 to 4 over a few minutes of chronic energy deficit, the drive itself rises from $E_h^i / (\text{Sens}_t \cdot E_h^t)$ to $4 E_h^i / (\text{Sens}_t \cdot E_h^t)$ — a fourfold increase in attentional priority at fixed hormone level. This is the mechanistic explanation for the *rare-resource attention shift* of Experiment 5 (the abundant-resource deficit must be 42 % greater, not 32 %, to attract attention after epigenetic tuning).

#### Positive feedback and path dependence

The most striking property of Eq. 12 is positive feedback: a small initial elevation of $E_h^i$ raises $\text{Sens}_i$, which raises effective drive, which can raise behaviour-mediated $E_h^i$ further. Two consequences:

1. **Hysteresis.** Once $\text{Sens}_i$ has been driven well above 1, returning to baseline requires a sustained drop in $E_h^i$ that may not be available in the chosen environment.

2. **Path dependence.** Two robots with identical initial parameters can diverge into different phenotypes based on early-life experience — exactly the day/night-cycle outcome of Experiment 4 (hibernation vs. stay-near).

The path-dependent divergence is the most biologically plausible feature of the rule — it is the architectural analogue of how prenatal stress imprints adult phenotype (Maccaria et al. 2003 etc. cited in the 2013 predecessor).

### Difference from the 2013 update rule

The 2013 conference paper used the *additive* rule

$$
\theta_g(t+1) = \theta_g(t) + k \cdot \frac{H_h(t)}{H_h^{\max}}, \qquad \text{sens}_h(t+1) = \text{sens}_h(t) + k \cdot \frac{H_h(t)}{H_h^{\max}},
$$

run only during the first 3 minutes (T_early = 2880 steps). The 2018 paper replaces these with the *multiplicative-exponent* rule Eq. 12, run continuously for the entire life of the robot. Three consequences:

1. **No critical window.** The 2013 paper froze parameters after 3 min. The 2018 paper does not — adaptation continues. This explains Experiment 4's *phenotype lock-in*: although the rule keeps running, the positive feedback creates strong attractors that survive ongoing experience.
2. **Multiplicative rather than additive.** Linear addition of $k \cdot H/H_{\max}$ produces gentle growth. Exponentiation produces explosive growth or decay. This makes the 2018 rule more sensitive to hormone level and more capable of producing dramatic phenotypic differences.
3. **Single update vs. two.** The 2013 rule updated *both* gland activity $\theta_g$ and receptor sensitivity $\text{sens}_h$. The 2018 rule keeps only the receptor update. Gland-activity update is dropped. This simplifies the architecture and isolates the mechanism to a single biological hypothesis — *epigenetic regulation of hormone-receptor expression*.

### Motivational state as a vector field, not a winner-takes-all

The 2018 paper departs from the discrete-motivation winner-takes-all of Cañamero 1997 (and most subsequent Cañamero-school papers) by computing motivation as a *per-direction vector*:

$$
\text{MotState}_j = \sum_{i \in \{e, h\}} \sum_c s_{i,c} \cdot (\text{Drive}_i \cdot \text{Cue}_{c,j}) - \text{Avoid}_j.
$$

Each of the 8 body-relative directions $j$ gets its own motivation. Wheel speeds (Eq. 6) are linear combinations of all 8 direction motivations, so the robot's actual motion is a vector-field gradient ascent — closer in spirit to Khatib-style potential-field navigation than to Cañamero 1997's behavioural-arbitration. This is also the simplification that makes the "continuous sensorimotor mapping" of the paper possible — there are no discrete behaviour modules to switch between.

### Pseudocode

```
Initialize: V_e = V_h = 100, V_t = 0
           Sens_E1 = Sens_H1 = Sens_T1 = Sens_D1 = 1
           hormone_levels[h] = empty (decaying-pulse buffer)
Parameters: sigma, psi_i, climate, set vectors Set_{k,j}, scaling s_{i,c}

At each loop t (16 per second):
  # 1. Physiology updates
  V_e -= 1/16                                  # 1 unit per second
  Health_change = sum over IR sensors per Eq. 1
  V_h += Health_change                          # could be negative
  V_t = abs(wheel_speed)/10 * climate - V_t * 0.05  # Eq. 2
  # 2. Endocrine hormone secretion
  for h in {E1, H1, T1}:
    pulse = psi_h * Error_h
    hormone_levels[h].append(pulse, duration=random_range)
  # 3. Neurohormone secretion
  pulse_D1 = sum(Drive_e + Drive_h + Drive_t)
  hormone_levels[D1].append(pulse_D1, duration=random_range)
  # 4. Decay buffer
  for h: decay_oldest_pulse(hormone_levels[h])
  E_h^h = sum_of_active_pulses(hormone_levels[h])
  # 5. Drives (Eq. 9)
  Drive_e = (Sens_E1 * E_h^E1) / (Sens_T1 * E_h^T1)
  Drive_h = (Sens_H1 * E_h^H1) / (Sens_T1 * E_h^T1)
  Drive_t = E_h^T1                              # implicit
  # 6. Motivational state per direction (Eq. 4)
  for j in 0..7:
    cue_e_j = visual cue magnitude for pink ball in direction j
    cue_h_j = visual cue magnitude for blue ball in direction j
    avoid_j = IR_j if IR_max - IR_j < Drive_h else 0  (Eq. 5)
    avoid_j /= max(1, sens_D1 * E_h^D1)               (Eq. 10)
    MotState_j = sum over (i, c) of (s_{i,c} * Drive_i * Cue_{c,j}) - avoid_j
  # 7. Wheel speeds (Eq. 6, then Eq. 11 multiplier)
  for k in {0, 1}:
    base = sum_{j} (MotState_j * Set_{k,j})
    WheelSpeed_k = base * Sens_D1 * E_h^D1
  Apply WheelSpeed to motors
  # 8. EPIGENETIC UPDATE (the new rule)
  for h in {E1, H1, T1, D1}:
    Sens_h = Sens_h ** (E_h^h / sigma)         # Eq. 12
  # 9. Mortality
  if V_e <= 0 or V_h <= 0 or V_t >= 100: die; end episode
```

### Mathematical commentary

Three structural choices repay reflection:

1. **Exponentiation is the simplest closed-form positive-feedback receptor regulation.** Linear feedback (the 2013 paper's $\theta \leftarrow \theta + k \cdot H/H_{\max}$) produces gentle saturation. Exponential feedback produces phase transitions — robots with similar initial hormone exposure can land in qualitatively different attractors. The Experiment 4 hibernation-vs-stay-near phenotype split is the most direct demonstration of this.

2. **The fixed point at $\text{Sens}^* = 1$ corresponds to no regulation.** This is by construction — a robot that starts at $\text{Sens} = 1$ and experiences no hormones stays at $\text{Sens} = 1$ forever. Any deviation from 1 requires sustained hormonal pressure, and the deviation grows or shrinks geometrically. This makes the mechanism *self-stabilizing* in the absence of pressure and *self-amplifying* under pressure.

3. **Receptor regulation, not gland regulation.** The 2013 paper updated both ends of the hormone chain. The 2018 paper keeps only receptor regulation — matching the biology of glucocorticoid-receptor methylation more closely (Fowden & Forhead 2009; Crews 2010; Zhang & Ho 2011, refs 5–8). The methodological tightening is a deliberate isolation of a single biological hypothesis.

## Connections

- **Direct predecessor.** `lones_canamero_2013_epigenetic_hormones.md` — the conference paper that proposed the additive critical-window rule. The 2018 paper replaces the additive rule with the multiplicative-exponent Eq. 12, removes the critical-window constraint, and replaces the discrete-motivation arbitration with a vector-field motivational-state formulation.
- **Direct architectural ancestor.** `canamero_1997_motivations_emotions.md` (ref 15 in the paper) — the homeostatic + drive + behaviour backbone. The 2018 paper diverges most sharply from 1997 in replacing winner-takes-all with continuous sensorimotor mapping.
- **Pleasure-as-modulator companion.** `lewis_canamero_2016_hedonic_pleasure.md` (ref 18) — same lab, complementary modulation philosophy. Lewis & Cañamero modulate a *perceptual gain* $\alpha$ via a fast pleasure hormone; Lones et al. modulate *receptor sensitivities* via a slower epigenetic rule. Composition would be natural.
- **PerAc cousin.** `blanchard_canamero_2006_affect_modulated.md` (ref 19) — the same lab's affect-modulated PerAc, with similar continuous-modulation philosophy at a different time scale.
- **Affordance-learning companion.** `cos_2010_affordances_consummatory.md` (related work via ref 17 *Hedonic value: enhancing adaptation for motivated agents*, by Cos, Cañamero, Hayes, Gillies 2013) — adds an affordance-learning layer on top of the homeostatic backbone. The 2018 paper's receptor regulation is at a *higher* (parameter-of-the-modulator) level than Cos et al.'s synapse-weight learning.
- **Methodological framing.** `canamero_2005_emotion_understanding.md` — the 2005 paper explicitly identified developmental / evolutionary mechanisms as the way to ground emotion in the agent; this 2018 paper realises that programme.
- **Cross-corpus links** (likely in other batches). Krichmar 2012, 2013 (refs 1, 11), Cañamero & Avila-García 2007 (ref 9), French & Cañamero 2005 (ref 22), Parussel & Cañamero 2007 (ref 23). The broader Krichmar neuromodulation programme — Cox & Krichmar 2009, Avery & Krichmar 2017 — and the modern neuromodulated-meta-learning literature (Wang et al. 2024; Lee et al. 2024; Doya 2002) are the natural cross-corpus neighbours.
