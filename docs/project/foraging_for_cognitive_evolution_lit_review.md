---
title: "Literature Review — Foraging as a Driver of Cognitive Evolution (Economic Choice + Predator-Prey)"
topic: foraging_for_cognitive_evolution
status: DRAFT v1 — both papers populated (backbone + Phase 1 + Phase 2)
created: 2026-05-07
last_updated: 2026-05-07
skill_used: pdf (pdfplumber)
related:
  - perceptual_noise_lit_review.md
  - computational_pain_models_lit_review.md
  - concepts/active_inference_hypervigilance.md
  - concepts/pain_vs_nociception_construct.md
  - directions/architecture_for_pain_computation.md
  - critiques/biological_plausibility_of_three_site_modulation.md
corpus:
  - "Hayden, B. Y. (2018). Economic choice: the foraging perspective. Current Opinion in Behavioral Sciences, 24, 1–6."
  - "Wooster, E. I. F., Whiting, M. J., Nimmo, D. G., Sayol, F., Carthey, A., Stanton, L., & Ashton, B. J. (2026). Predator–prey interactions as drivers of cognitive evolution. (Preprint.)"
---

# Literature Review — Foraging as a Driver of Cognitive Evolution

> **Scope note.** Two-paper review of why and how foraging — broadly construed
> as encounter-driven accept/reject decisions in patchy environments populated by
> prey, conspecifics, and predators — selects for the inferential and modulatory
> machinery our project is trying to model in silico. Hayden (2018) supplies the
> *internal-decision* side (rate-maximization, accept-reject, dACC/vmPFC competition,
> stopping). Wooster et al. (2026) supplies the *ecological-driver* side (a
> cognitive arms race between predators and prey via the Predatory Intelligence
> Hypothesis). Together they motivate why a gridworld with patchy prey, hazard,
> and homeostatic state is a *natural* test bed for ascending-modulator analogs,
> precision-weighting, and exploration–exploitation trade-offs under bodily
> constraint.
>
> **Project framing tags used throughout.** The following inline markers flag
> threads the downstream professor / experiment-designer agents should pull on:
> `[PC]` predictive coding under uncertainty; `[NM-{ACh,NE,DA,5HT,OPI}]`
> ascending-neuromodulator analog; `[RS]` risk-sensitivity / CVaR /
> distributional / prospect-theoretic; `[EE]` exploration–exploitation under
> bodily-state constraint; `[MVT-H]` MVT extensions with internal state /
> homeostasis; `[HV]` predator-driven hypervigilance (directly relevant to our
> "injury → hypervigilance" hypothesis).

---

## Table of Contents

- [1. Hayden (2018) — *Economic choice: the foraging perspective*](#1-hayden-2018--economic-choice-the-foraging-perspective)
  - [1.1 Phase 1 — Foundational Overview](#11-phase-1--foundational-overview)
  - [1.2 Phase 2 — Graduate-Level Deep Dive](#12-phase-2--graduate-level-deep-dive)
  - [1.3 Project-Framing Hits](#13-project-framing-hits)
  - [1.4 Appendix: Section-by-Section Backbone](#14-appendix-section-by-section-backbone)
- [2. Wooster et al. (2026) — *Predator–prey interactions as drivers of cognitive evolution*](#2-wooster-et-al-2026--predator-prey-interactions-as-drivers-of-cognitive-evolution)
  - [2.1 Phase 1 — Foundational Overview](#21-phase-1--foundational-overview)
  - [2.2 Phase 2 — Graduate-Level Deep Dive](#22-phase-2--graduate-level-deep-dive)
  - [2.3 Project-Framing Hits](#23-project-framing-hits)
  - [2.4 Appendix: Section-by-Section Backbone](#24-appendix-section-by-section-backbone)
- [3. Cross-Paper Synthesis & Hand-offs](#3-cross-paper-synthesis--hand-offs)

---

## 1. Hayden (2018) — *Economic choice: the foraging perspective*

**Citation.** Hayden, B. Y. (2018). Economic choice: the foraging perspective.
*Current Opinion in Behavioral Sciences*, **24**, 1–6.
DOI: 10.1016/j.cobeha.2017.12.002. Themed issue: *Survival circuits* (eds.
Mobbs & LeDoux).

**One-line.** Re-frames neuroeconomics: economic choice should be modelled as
a sequence of **accept-reject** decisions against a *background-rate threshold*
(the foraging primitive), not as a binary side-by-side comparison; binary
choice is a paired race-to-threshold, and choice itself collapses onto
**stopping**.

### 1.1 Phase 1 — Foundational Overview

#### Introduction (undergrad-level)

Standard microeconomics frames choice as a cool side-by-side comparison of two
labelled options. Hayden argues that this is the wrong null hypothesis for
animal brains. Brains evolved to keep their owners *alive* in patchy
environments where prey and threats appear **one at a time**: a gorilla
walking through a forest does not see "Apple A vs. Apple B"; it sees one
foreground item and a remembered background richness. The natural primitive
is therefore **accept the foreground item** vs. **reject and keep
searching**. Apparent two-option choices in the lab (e.g. monkey picks left
target vs. right target) are, on this view, **two interleaved accept-reject
decisions** under a moving spotlight of attention.

#### Key Findings

1. **Accept-reject is the elemental decision.** It is asymmetric: accepting
   triggers consumption, monitoring, and learning; rejecting maintains the
   status quo and continues search. The two options are not symmetric labels.
2. **The decision variable is profitability vs. background.** The brain compares
   a (dynamic) foreground profitability to a (slow) background "average value
   of the environment." This is the **Charnov / MVT** core, repurposed as a
   neural mechanism.
3. **Self-control failures are mostly task-misreading.** Animals look impulsive
   in delay-discounting tasks because they apply an evolutionarily-favoured
   **expectation-of-ratios** (rate-maximizing) heuristic; in proper foraging
   tasks, the same animals are nearly perfectly patient.
4. **dACC encodes "alternative / reject value"; vmPFC encodes "accept / default
   value."** Their competition implements the comparison-to-threshold.
   vmPFC lesions disrupt the default-vs-alternative arbitration.
5. **Binary choice ≈ paired race-to-threshold.** Reaction-time and
   choice-probability data, plus OFC ensemble oscillations between two states,
   support paired-races over a single drift-diffusion between two bounds.
6. **Choice ≈ stopping.** Accepting performs a primed motor act; rejecting
   withholds it. This is mathematically and neurally close to a stop-signal
   decision, opening a "grand unified theory" of choice and inhibitory control.
7. **"Value" may not be explicitly represented.** A more parsimonious null is
   that the brain performs a gradient rotation from sensory input to motor
   output without a special amodal "value" layer; what we call value is a
   *tentative commitment to acceptance*.

#### Initial Takeaway

Foraging is not one option among many for modelling decision-making — it is
the **substrate** out of which decision-making evolved. For our project this
re-licenses two design choices: (i) gridworld decisions framed as accept-reject
against a slowly-updating background (rather than as side-by-side menu
comparisons); (ii) homeostatic / hazard signals interpreted as dynamic
*threshold modulators* on that background, not as separate reward terms.

### 1.2 Phase 2 — Graduate-Level Deep Dive

#### 1.2.1 The MVT primitive that Hayden imports

Although Hayden does not write the equations out, the entire paper sits on top
of Charnov's (1976) **Marginal Value Theorem**. Below we derive the rate-
maximizing optimum that Hayden treats as the brain's implicit cost function;
we will return to it again in §2 because Wooster et al. cite Charnov directly.

Let an animal cycle between **travel** (no intake) for time $\tau$ and
**patch exploitation** for time $t$ during which cumulative gain is $g(t)$,
with $g$ concave (diminishing returns). The long-run intake rate over many
patches is

$$
R(t) \;=\; \frac{g(t)}{\tau + t}.
$$

We maximize $R(t)$ in $t$:

$$
\frac{dR}{dt}
= \frac{g'(t)\,(\tau+t) - g(t)}{(\tau+t)^2} \;=\; 0
\;\;\Longleftrightarrow\;\;
g'(t^\star) \;=\; \frac{g(t^\star)}{\tau + t^\star} \;=\; R(t^\star).
$$

So the optimal patch-leaving time $t^\star$ is the moment at which the **local
gain rate** $g'(t)$ equals the **environmental average rate** $R^\star$. That
$R^\star$ is exactly the "background value" Hayden says vmPFC / dACC must
maintain. The accept-reject decision becomes:

$$
\text{Accept iff } \pi_{\text{fg}} \;\ge\; R^\star,
\qquad
\pi_{\text{fg}} \;=\; \frac{\text{expected gain}}{\text{expected handling+travel cost (incl. opportunity)}}.
$$

This is the *profitability vs. threshold* rule Hayden states verbally.
Two consequences he flags but does not derive:

1. **Opportunity cost is the threshold.** Rejecting means you pay $R^\star$
   per unit time forgone. The dACC encoding of the rejected option's value
   *plus* a delay term (Blanchard & Hayden 2014) is therefore exactly the
   opportunity-cost signal. **`[MVT-H]`** Adding homeostatic state $h$ (e.g.
   energy, hydration) replaces the scalar $R^\star$ with a state-dependent
   $R^\star(h)$ — the *same* MVT structure, but with a moving threshold.
2. **Apparent impulsivity from rate-of-ratios.** Under standard intertemporal
   choice with a post-reward "blank" interval $\delta$, the animal's
   *evolutionarily natural* estimator is

$$
\hat R_i \;=\; \mathbb{E}\!\left[\frac{r_i}{d_i}\right],\quad i \in \{\text{small-soon},\text{large-late}\},
$$

   *not* $\mathbb{E}[r_i]/\mathbb{E}[d_i]$. Jensen's inequality bites because
   $1/d$ is convex: the small-soon option is over-valued. Bateson & Kacelnik
   (1996) — Hayden's ref [8] — show this analytically. With cued post-reward
   delays, the bias collapses, exactly as Hayden's [9,10] report.

#### 1.2.2 Accept-reject as drift-diffusion against a moving threshold

Hayden's mechanistic claim is that the brain maintains a **dynamic** representation
of foreground profitability $\pi_{\text{fg}}(t)$ and a **stable** representation of
background $R^\star$ and compares them. Formalizing this as drift-diffusion:

$$
dX_t \;=\; \big(\pi_{\text{fg}}(t) - R^\star\big)\,dt \;+\; \sigma\,dW_t,
$$

with **Accept** when $X_t \ge +b$ and **Reject** when $X_t \le -b$ (or by timeout).
Two important departures from textbook 2-AFC drift-diffusion:

- The drift is **asymmetric in sign** (not a difference of two symmetric
  evidences); accept and reject have different post-decision consequences
  (consumption + monitoring vs. continued search).
- The threshold is itself learned: control systems "modulate these
  representations [and] regulate the threshold for accepting a presented
  option" — Hayden's [20, 21, 22]. Concretely, dACC integrates the
  alternative value across multiple trials in patch-leaving tasks, gradually
  raising the bar to reject. **`[NM-NE]` `[NM-ACh]`** Threshold modulation
  is exactly the territory of ascending-neuromodulator analogs in our model.

#### 1.2.3 Binary choice = paired race-to-threshold

The serial-attention argument compresses to: when two offers are
simultaneously on screen, attention spotlights one at a time, OFC ensembles
oscillate between two discrete states (Rich & Wallis 2016; Hayden's [37]),
and each state is an accept-reject race. Let $X^{(i)}_t$ be the evidence for
option $i \in \{1,2\}$, each obeying

$$
dX^{(i)}_t \;=\; \big(\pi_i(t) - R^\star\big)\,dt + \sigma\,dW^{(i)}_t,
$$

with attention $a_t \in \{1,2\}$ gating which $\pi_i(t)$ updates at any given
moment. Choice is whichever $X^{(i)}$ first crosses $+b$. This is a *paired
race* (Vickers / LCA family) rather than a single drift between two bounds.
Practical implications Hayden flags:

1. The **representation is relative, not absolute** (Lim et al. 2011; Strait
   et al. 2014): vmPFC fires for *value of attended − value of background*.
   No amodal value head is required.
2. The mechanism subsumes "menu effects" (range adaptation, normalization)
   because $R^\star$ rescales every $\pi_i$.
3. Under pre-cue priming, the **"primed" option's accept race already has a
   head start** — exactly the asymmetry that produces default biases in
   binary choice (vmPFC for default, dACC for alternative).

#### 1.2.4 Choice ≡ stopping (the unification claim)

Hayden's most speculative move: an accept-reject decision is structurally a
**stopping** decision. Rejection = withhold the primed motor plan; acceptance
= release it. Mathematically this maps onto the **horse-race model** of
stop-signal:

$$
T_{\text{accept}} \sim F_{\text{go}}(\cdot;\, \pi_{\text{fg}}),\quad
T_{\text{reject}} \sim F_{\text{stop}}(\cdot;\, R^\star),
\qquad
\text{decision} \;=\; \arg\min\big(T_{\text{accept}}, T_{\text{reject}}\big).
$$

If $T_{\text{stop}} < T_{\text{go}}$, the action is inhibited (= reject). The
go-rate is foreground profitability, the stop-rate is background opportunity.
This formalizes Hayden's "grand unified" picture and predicts (i) shared
neural substrate (motor / premotor cortex, broad PFC) for choice and stopping
— supported by his refs [55, 56, 57]; (ii) self-control deficits as failures
of the stop race, not of "value computation" per se.

#### 1.2.5 The "no amodal value" null hypothesis

Hayden closes with a strong null: the brain may not compute value as an
intermediate quantity. Formally, view the brain as a smooth map
$f_\theta : \mathcal{S} \to \mathcal{A}$ from sensorium to action, so that
the *gradient* — not a learned scalar — drives choice:

$$
a^\star \;=\; \arg\max_{a \in \mathcal{A}} \; f_\theta(s)\!\cdot\!\phi(a),
\qquad
\nabla_\theta \mathbb{E}\!\left[\sum_t r_t\right] \;=\; \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R_t\right]
$$

(a policy-gradient view, equivalent to "Cisek's distributed consensus" framing,
his refs [56, 57]). What we call "value" is then the magnitude of the
tentative commitment to release the primed action — a derived rather than
primary quantity. **Implication for our architecture.** A pure model-free
policy-gradient agent already has *no* explicit value head: the entire
debate over "where in the agent does value live?" is recast as "how does the
threshold for action release adapt to bodily state and risk?" — which is
precisely the question our precision / neuromodulator modules answer.

### 1.3 Project-Framing Hits

| Tag | Where in Hayden | Implication for our project |
|---|---|---|
| `[MVT-H]` | The "background value $R^\star$" is the MVT optimum; replacing $R^\star$ with $R^\star(h)$ for homeostatic state $h$ is a one-line extension | Our gridworld with satiation/hydration drives is *the* natural arena to implement state-dependent MVT thresholds; a clean experiment is *predict patch-leaving time as a function of internal state*. |
| `[EE]` | "Accept ↔ exploit; reject ↔ continue search ↔ explore" mapping is explicit | The accept-reject framing is a *cleaner* exploration–exploitation decomposition than ε-greedy; the threshold $R^\star$ *is* the explore/exploit knob, and it is bodily-state-modulated. |
| `[NM-NE]` `[NM-ACh]` | "Control systems in the brain then can modulate these representations [and] regulate the threshold for accepting a presented option" (refs 20, 21, 22) | Direct license for tonic-NE / ACh analogs to act as **threshold modulators** on the accept-race, not as direct reward modifiers. Hand to `professor-neuromodulation`. |
| `[RS]` | Steiner & Redish 2014 (regret in rats; ref [22]) is cited as a foraging-task result; risk-sensitivity is implicit in $\pi_{\text{fg}}$ vs. $R^\star$ comparison under uncertainty | The accept-race naturally accommodates CVaR / distributional drift — replace the scalar $\pi_{\text{fg}}$ with a quantile or risk-distorted value. |
| `[PC]` | "tentative commitment" + "gradient rotation from input to output without a special amodal value layer" | Equivalent to the *active-inference* / predictive-coding null: action is selected to minimize prediction error (or maximize evidence for the primed proposition). Hand to `professor-bayesian-brain`. |
| `[HV]` | *Not directly addressed* — Hayden is silent on hazard-driven threshold raising | Wooster et al. fills this gap; cross-link in §3. |

### 1.4 Appendix: Section-by-Section Backbone

Reproducing the paper's section order (six pages, no numbered sections beyond
heads):

#### B-1.1 Introduction

- "Bloodless" microeconomic framing of choice is misleading; choice circuits
  evolved under intense selection for survival.
- Lab decisions have a direct natural analog: **feeding decisions** by
  foragers. They must be quick (else lose the prey to flight or rivals) and
  accurate (life-or-death).
- Foraging therefore "puts economic choice under the rubric of *survival
  circuits*" (refs LeDoux [3, 4]; Mobbs et al. ecology of fear [3]).

#### B-1.2 Foraging decisions are accept-reject decisions

- Real environments are **patchy in space and ephemeral in time** (Stephens
  & Krebs 1986, ref [5]).
- Encounters are one-at-a-time; the elemental decision is **accept (pursue)**
  vs. **reject (continue search)**.
- Accept-reject is not symmetric:
  - Pursuing is *active* (state-change, consumption, monitoring, learning);
  - Rejecting is *passive* (status-quo, returns the agent to outwardly
    oriented search mode → exploratory).

#### B-1.3 Insights into self-control from the accept/reject framing

- In intertemporal-choice (delay-discounting) tasks, animals look impulsive,
  which is *evolutionarily maladaptive*.
- Resolution: animals apply an **expectation-of-ratios** (rate-maximizing)
  heuristic that is correct in the wild but mis-applied to the lab task
  structure with hidden post-reward intervals (Bateson & Kacelnik 1996, ref
  [8]; Blanchard et al. 2013, ref [9]).
- When post-reward delays are unambiguously cued, choice becomes
  rate-maximizing; in patch-leaving foraging tasks, animals are *almost
  perfectly patient* (refs [10, 11, 12, 13]).
- **Persistence** (continuing to pursue after the decision is made) is a
  separate computation: dynamic value updating (refs [14, 15, 16, 17, 18]);
  failures of this updating may underlie self-control failures.

#### B-1.4 How accept-reject decisions are implemented

- Decision variable: **profitability** of the foreground item vs. **average
  value of the environment**, $R^\star$.
- Implementation: dynamic foreground representation + stable background
  representation + comparator + threshold (refs [15, 19, 20]).
- Control systems can modulate either representation or the threshold (refs
  [20, 21, 22]).
- **vmPFC ↔ default (accept)**, **dACC ↔ alternative (reject)** (refs [19,
  23]). vmPFC encodes value of accepted offer; dACC encodes value of rejected
  offer + opportunity-cost delay (Blanchard & Hayden 2014, ref [26]). dACC
  responses *gradually rise* across trials in patch-leaving tasks (Hayden,
  Pearson, & Platt 2011, ref [20]). vmPFC lesions disrupt repeating after
  large rewards (ref [25]).
- The "competition" between these two systems implements the comparator.

#### B-1.5 Are ostensibly binary choices really paired accept-reject choices?

- Foragers' brains evolved for single encounters (Stephens et al. 2004, ref
  [7]; Kacelnik et al. 2011, ref [30]).
- Reaction-time and choice-probability evidence (refs [31–34]) supports
  **paired race-to-threshold** over single drift between two bounds.
- **Attention is the bottleneck.** Feature-binding requires a single
  spotlight (Treisman & Gelade 1980, ref [35]). Two simultaneously presented
  options are evaluated *serially*. Rich & Wallis 2016 (OFC ensemble
  oscillations between two discrete states, ref [37]) is the key
  electrophysiological evidence.
- vmPFC / OFC preferentially track value of *attended* offers (refs [38, 39,
  40, 41, 42]).

#### B-1.6 How can comparison occur in serial choice models?

- Possibility: brain computes **relative** (not absolute) value — value
  difference or quotient relative to background — so that a single thresholding
  of a normalized signal suffices (refs [38, 39, 43–47]).
- During each attention epoch, evidence accumulates stochastically (refs
  [36, 48–50]), at least partly via *sampling from memory* of past
  stimulus-action-value mappings (Shadlen & Shohamy 2016, ref [51]). Memory
  sampling is especially needed when options are multi-dimensional (ref [45]).

#### B-1.7 Foraging suggests a unification of economic and stopping decisions

- Accept = release primed motor plan; reject = withhold it.
- Therefore accept-reject ≈ **stopping**; binary choice ≈ paired interacting
  stopping decisions.
- Evidence: motor / premotor cortex roles in stopping (refs [56, 57]) and in
  economic choice; broad PFC distribution for both (refs [55, 57, 58]).
- Promised payoff: importing the well-developed neuroscience of stopping
  (refs [52–55]) directly into economic-choice neuroscience.

#### B-1.8 Value as tentative commitment to a decision

- Evidence that the brain explicitly computes amodal value is **equivocal**
  (refs [58, 61]).
- Null hypothesis: brain performs a gradual sensorimotor *rotation* from
  input to output, with no special amodal-value middle layer.
- Empirical support: sensorimotor information pervades reward areas (refs
  [40, 57, 62, 63, 64]).
- "Value" $\to$ "tentative commitment" to accept an offer / proposition
  (Shadlen et al. 2008 intentional-framework, ref [65]).
- Future work should be **ethologically embedded** (ref [1]).

---

## 2. Wooster et al. (2026) — *Predator–prey interactions as drivers of cognitive evolution*

**Citation.** Wooster, E. I. F., Whiting, M. J., Nimmo, D. G., Sayol, F.,
Carthey, A., Stanton, L., & Ashton, B. J. (2026). Predator–prey interactions
as drivers of cognitive evolution. (Preprint, 21 pp.)

**One-line.** Formalizes a third hypothesis for the evolution of cognition —
the **Predatory Intelligence Hypothesis (PIH)** — alongside the long-standing
Social (SIH) and Ecological (EIH) Intelligence Hypotheses. Argues that the
asymmetric, repeated, partially-cryptic information game between predators
and prey drives a cognitive arms race, that brain-size and cognitive-test
data already partially support this, and that recent network-ecology +
camera-trap + biologger advances make the PIH testable for the first time.

### 2.1 Phase 1 — Foundational Overview

#### Introduction (undergrad-level)

Why are some animals so much smarter than others? Two long-standing
explanations are:

- **SIH (Social Intelligence Hypothesis):** complex social life — keeping
  track of allies, rivals, and "Machiavellian" coalitions — selects for
  cognition.
- **EIH (Ecological Intelligence Hypothesis):** complex food and habitat —
  variable, patchy, hard-to-extract resources — selects for cognition.

Both have empirical support, but a meta-analysis of 103 studies still leaves
**>40% of cognitive variation unexplained**. Wooster et al. argue the missing
driver is **predation**: prey have to avoid being killed by predators that
hide their intent, while predators have to outwit prey that hide their
location. Both sides face problems (cue recognition, partial observability,
spatial-temporal memory, response timing) that look exactly like the kinds of
cognitive challenges that should select for bigger, more flexible brains.
They package this as the **PIH**.

#### Key Findings

1. **PIH formal claim.** Predator-prey interactions are a **co-evolutionary
   arms race** that produces *bidirectional* enhancements in cognition —
   smarter prey, then smarter predators, then smarter prey, etc.
2. **Predation is partially-observable and adversarial.** Prey rely on
   *cryptic and indirect* cues (predators are *trying* to hide). This is a
   different, harder problem than the EIH problem of foraging on plants.
3. **Cognition is already correlated with survival under predation.**
   - Pheasants with better spatial memory have smaller home ranges *and*
     lower predation risk (Heathcote et al. 2023, ref [21]).
   - African striped mice with faster reaction times and better spatial
     memory survive longer (Maille & Schradin 2016; Rochais et al. 2023, refs
     [22, 50]).
   - Female guppies with larger brains assess predation risk faster (van der
     Bijl et al. 2015, ref [25]); guppies under high predation evolve larger
     telencephala (Kotrschal et al. 2017, ref [24]).
   - Grey mouse lemur composite cognitive score positively correlates with
     wild survival (Fichtel et al. 2023, ref [49]).
4. **But cognition is not always beneficial.** Brain tissue is metabolically
   expensive; species with strong physical defences (armour, spines) show
   *reduced* brain size (Stankowich & Romero 2017, ref [26]); fast pure-speed
   escapers may invest in muscle instead.
5. **Diversity of predators / prey amplifies the pressure.** A prey facing
   *functionally distinct* predators (ambush + pursuit + aerial) must hold
   multiple recognition templates; a *generalist* predator hunting many
   functionally distinct prey types must hold multiple search images and
   strategies. Both should be more cognitively demanding than a one-predator
   / one-prey life.
6. **Cooperative hunting sits at the PIH ↔ SIH interface.** Group hunting
   requires both individual-tracking (SIH) and prey-strategy modeling (PIH).
7. **The PIH is now testable.** Camera traps, biologgers, animal-borne
   cameras, ancient predator-prey network reconstruction (Pleistocene-scale),
   functional-trait macroecology, and phylogenetic comparative methods make
   it possible to (i) measure predation pressure, (ii) quantify functional
   diversity of predator/prey guilds, and (iii) correlate these with brain
   size and cognitive test performance.

#### Initial Takeaway

For our project: predation is not just a hazard term in the reward function —
it is, on this hypothesis, the *primary evolutionary justification* for the
neural circuitry our agents are supposed to instantiate. A gridworld with a
predator (or several functionally distinct predators) is a deliberate PIH
test bed in miniature, and the natural dependent variables are *how richly
the agent's policy uses uncertainty estimates, memory, and prior risk maps*
— the very things our precision / neuromodulator modules are designed to
control.

### 2.2 Phase 2 — Graduate-Level Deep Dive

The paper is a synthesis-and-predictions article rather than a model paper,
so the formal content is mostly imported (Charnov MVT, Bayesian updating,
arms-race theory). We make those imports explicit and derive the missing
intermediate steps.

#### 2.2.1 Charnov MVT inside the PIH

Wooster et al. invoke MVT (their ref [79], Charnov 1976) when arguing that
predator selection on cognition intensifies as prey become scarce: each failed
hunt then has a higher relative fitness cost, so optimal hunt-decision
making — which prey to pursue, when to give up — pays off more. They do not
write the equations, but the argument is exactly the MVT derivation in §1.2.1
above, transposed: prey-encounters replace patch arrivals, handling time
replaces patch-residence, and "prey scarcity" raises $\tau$ (search time).
Recall

$$
R(t) = \frac{g(t)}{\tau + t},
\qquad
g'(t^\star) = \frac{g(t^\star)}{\tau + t^\star} = R^\star.
$$

Differentiate $R^\star$ with respect to $\tau$ implicitly to see how the
optimal threshold shifts as prey become rare:

$$
g''(t^\star)\,\frac{dt^\star}{d\tau}
= \frac{g'(t^\star)\,(\tau + t^\star)\,(1 + dt^\star/d\tau) - g(t^\star)\,(1 + dt^\star/d\tau)}{(\tau + t^\star)^2}.
$$

Using $g'(t^\star)(\tau + t^\star) = g(t^\star)$ at the optimum, the right-hand
side reduces to $0\cdot(1 + dt^\star/d\tau)/(\tau + t^\star)^2$ once we
expand carefully — a clearer route is to differentiate $g'(t^\star) = R^\star$:

$$
g''(t^\star)\,\frac{dt^\star}{d\tau}
= \frac{dR^\star}{d\tau}
= \frac{g'(t^\star)\,\frac{dt^\star}{d\tau}\,(\tau+t^\star) - g(t^\star)\big(1 + \frac{dt^\star}{d\tau}\big)}{(\tau + t^\star)^2}.
$$

Substituting $g'(t^\star) = g(t^\star)/(\tau+t^\star) = R^\star$:

$$
g''(t^\star)\,\frac{dt^\star}{d\tau}
= \frac{R^\star\,\frac{dt^\star}{d\tau}\,(\tau+t^\star) - R^\star(\tau+t^\star)\big(1 + \frac{dt^\star}{d\tau}\big)}{(\tau+t^\star)^2}
= -\,\frac{R^\star}{\tau + t^\star}.
$$

Therefore

$$
\boxed{\;
\frac{dt^\star}{d\tau} \;=\; -\,\frac{R^\star}{g''(t^\star)\,(\tau + t^\star)} \;>\; 0
\;}
$$

(since $g''(t^\star) < 0$ by concavity). When prey are scarce ($\tau$ large),
the predator should stay longer in each "patch" (read: persist longer with a
given hunt), and the optimal background rate $R^\star$ falls. Predators that
*estimate* $R^\star$ better — which requires memory and Bayesian integration
over prey-encounter history — therefore have a fitness advantage, exactly as
Wooster et al. claim verbally. **`[MVT-H]`** This is the MVT extension
Wooster et al. invoke; layering homeostasis is then a further step (§3).

#### 2.2.2 Bayesian updating as the formal substrate of "appropriate response"

The paper repeatedly says prey "must detect predator cues, integrate this
information, and then respond appropriately" via "a process known as Bayesian
updating" (Box 1 glossary; main text). They do not write the rule; we do.

Let $H \in \{0,1\}$ be the hidden hazard (predator absent / present), $C$
a cue (olfactory, visual, vibrational, auditory). The Bayesian posterior is

$$
p(H = 1 \mid C) \;=\; \frac{p(C \mid H=1)\,p(H=1)}{\sum_{h\in\{0,1\}} p(C \mid H = h)\,p(H = h)}.
$$

Sequentially across cues $C_1, \dots, C_T$ assumed conditionally independent
given $H$,

$$
p(H=1 \mid C_{1:T})
\;\propto\; p(H=1)\,\prod_{t=1}^{T} p(C_t \mid H=1),
$$

which in log-odds form is the *additive* rule

$$
\ell_T \;=\; \ell_0 \;+\; \sum_{t=1}^{T} \log\!\frac{p(C_t \mid H=1)}{p(C_t \mid H=0)},
\qquad \ell_t \;:=\; \log\frac{p(H=1 \mid C_{1:t})}{p(H=0 \mid C_{1:t})}.
$$

This is exactly the **drift-diffusion / sequential probability ratio test
(SPRT)**: a prey accumulates log-likelihood ratios and acts (flee, hide, or
continue foraging) when $\ell_t$ crosses a decision threshold. Two
project-relevant features:

1. **Per-cue precision** is the per-step drift magnitude. Cues with lower
   $|\log p(C_t \mid H=1) - \log p(C_t \mid H=0)|$ are *less informative* —
   they should be down-weighted. This is exactly **precision-weighted
   prediction error** in active-inference / predictive-coding terms. **`[PC]`**
2. **The flight threshold is metabolically and ecologically tunable.** Risk-
   averse animals lower the threshold for $H=1$; satiated / safe animals
   raise it. This is the *vigilance* knob — a very natural target for
   ascending neuromodulator analogs (NE for arousal-driven gain, ACh for
   sensory precision). **`[NM-NE]` `[NM-ACh]` `[HV]`**

#### 2.2.3 Dawkins-Krebs arms-race formalism

Wooster et al. cite Dawkins & Krebs (1979, ref [84]) for the arms-race
framing. We need to make their "bidirectional cognitive enhancement" claim
sharp, since it's the central PIH prediction.

Let $x \in \mathbb{R}_{\ge 0}$ be predator cognitive investment, $y$ prey
cognitive investment, with hunt success probability $\phi(x,y)$ increasing
in $x$, decreasing in $y$. Per-encounter fitness is

$$
W_p(x,y) \;=\; \phi(x,y)\,B_p \;-\; c_p(x), \qquad
W_v(x,y) \;=\; -\,\phi(x,y)\,B_v \;-\; c_v(y),
$$

with $B_p, B_v$ benefits (energy gain / mortality cost), $c_p, c_v$
metabolic costs of cognition (convex, increasing). Each side best-responds:

$$
\frac{\partial W_p}{\partial x} = \phi_x\,B_p \;-\; c_p'(x) = 0,
\qquad
\frac{\partial W_v}{\partial y} = -\,\phi_y\,B_v \;-\; c_v'(y) = 0.
$$

A **co-evolutionary equilibrium (ESS)** $(x^\star, y^\star)$ satisfies both
simultaneously:

$$
c_p'(x^\star) \;=\; \phi_x(x^\star, y^\star)\,B_p,\quad
c_v'(y^\star) \;=\; -\,\phi_y(x^\star, y^\star)\,B_v.
$$

If the cross-partials $\phi_{xy}$ are positive (improvements escalate the
arms race rather than damping it), the equilibrium $(x^\star, y^\star)$
moves *up and to the right* — precisely the bidirectional escalation Wooster
et al. predict. The cost of cognition $c_p, c_v$ is the metabolic-trade-off
brake that explains §2.2.4 (when armour or pure-speed cheaply produces low
$\phi(x,y)$, the prey's best-response $y^\star$ falls — the brain shrinks).

#### 2.2.4 Functional diversity → cognitive load (information-theoretic
take)

Wooster et al. predict cognition rises with *functional diversity* of the
opposing guild, not just number of species. This is captured cleanly with
information-theoretic load. Let $S$ be the set of predator types a prey
encounters, with frequencies $\{p_s\}$ and required-response sets $\{a_s\}$.
The cognitive load can be lower-bounded by the conditional entropy of
appropriate response given cue $C$:

$$
H(A \mid C) \;=\; -\sum_{c} p(c)\sum_{a} p(a \mid c)\,\log p(a \mid c),
$$

and the entropy is largest when (i) the number of distinct $a$ is large
(rich response repertoire) and (ii) cues $C$ are *non-discriminating* across
predators (high overlap → ambiguity). Multi-predator with functionally
similar styles → cues collapse, entropy stays low (the "predator archetype"
generalization, ref [100]). Multi-predator with functionally distinct styles
→ entropy is high, requiring storage of $|S|$ separate response policies and
a classifier mapping $C \to s$. The brain-size prediction follows from
representational-capacity arguments (Vapnik-Chervonenkis-flavoured: more
distinct policies → more parameters). **`[PC]` `[EE]`** The paper's
*multi-predator hypothesis* (refs [28, 100]) is in effect a discrete
approximation of this entropy account.

#### 2.2.5 Brain-size as a (noisy) cognitive proxy and the metabolic budget
constraint

The paper's empirical synthesis leans on brain (or telencephalon) volume as
a cognitive proxy. The energy-budget constraint that *limits* brain size is
formalizable:

$$
E_{\text{tot}} \;=\; E_{\text{brain}} + E_{\text{soma}} + E_{\text{repro}} + E_{\text{escape}} + \cdots
$$

with $E_{\text{brain}} \approx 20\%$ of resting metabolic rate in primates
(Parker 1990, ref [53]). Selection therefore optimizes net fitness over the
allocation:

$$
\max_{\{E_i\}} \; W\big(E_{\text{brain}}, E_{\text{soma}}, E_{\text{repro}}, E_{\text{escape}}\big)
\quad\text{s.t.}\quad \sum_i E_i \le E_{\text{tot}}.
$$

Lagrangian KKT conditions equate marginal fitness returns:

$$
\frac{\partial W}{\partial E_{\text{brain}}} = \frac{\partial W}{\partial E_{\text{escape}}} = \frac{\partial W}{\partial E_{\text{repro}}} = \cdots
$$

If physical defences cheaply produce a high marginal return on
$E_{\text{escape}}$ (armour, spines), the equilibrium $E_{\text{brain}}^\star$
is small — predicting Stankowich & Romero 2017 (ref [26]). The same logic
predicts mesopredators, who must *both* hunt and avoid being hunted, allocate
more to brain than apex predators do (one of the paper's testable
predictions in Table 1).

#### 2.2.6 Methodological pipeline (testability claim)

The paper's testability claim rests on a clean three-tier methodological
stack:

| Tier | Methods | What it measures |
|---|---|---|
| Macroevolutionary | Phylogenetic comparative methods (PGLMM, PMM); ancient predator-prey network reconstruction (refs [91–94, 97]); functional-trait macroecology (ref [95]) | Co-evolution of brain size with predator-guild richness / functional uniqueness across deep time |
| Population (intra-species) | Comparisons across islands / fenced reserves / conservation havens (refs [108, 109, 110]); split-clutch guppy experiments | Effect of *current* predation regime on cognitive phenotypes within a species |
| Individual | Cognitive test batteries × predation-risk experiments (giving-up densities, refs [105, 106]; playback experiments, ref [107]); biologgers + animal-borne cameras (refs [89, 90]) | Whether cognitive scores predict survival / hunt success in wild individuals |

Note Box 1 glossary explicitly includes **Marginal Value Theorem**,
**Bayesian updating**, **Inhibitory control**, **Behavioural flexibility** —
the conceptual scaffolding on which their predictions stand.

### 2.3 Project-Framing Hits

| Tag | Where in Wooster et al. | Implication for our project |
|---|---|---|
| `[HV]` (predator-driven hypervigilance) | Whole paper. "Individuals who can remember and learn when to be vigilant, when and where to flee or hide, and know when to not invest in these costly behaviours are likely to be selected for under predation risk." | Directly supports our "injury → hypervigilance" hypothesis: vigilance is a *learned, cue-conditioned, cost-aware* policy, not a fixed reflex. The *dynamic* threshold in the SPRT formulation (§2.2.2) is the formal handle. Hand to `professor-pain-modeling` and `professor-bayesian-brain`. |
| `[PC]` predictive coding under uncertainty | Box 1 glossary explicitly lists Bayesian updating; main-text repeatedly invokes it as the prey's cue-integration mechanism | The PIH operationalizes prey cognition as *Bayesian inference under partial observability* — exactly the substrate of active inference / predictive coding. Hand to `professor-bayesian-brain`. |
| `[NM-NE]` `[NM-ACh]` ascending-neuromodulator analogs | Not by name. The "rapid integration" + "filter information against background noise" + "reaction time under risk" trio (esp. African striped mice work, refs [22, 50]) is a textbook NE/ACh role | We can claim that NE-like tonic gain and ACh-like sensory precision are the *neural mechanism* by which the PIH's cognitive gain is realized. Hand to `professor-neuromodulation`. |
| `[NM-OPI]` opioidergic | Not addressed by Wooster et al. | Cross-link with `computational_pain_models_lit_review.md` and the `pain_vs_nociception_construct.md` concept doc — the *cost* side of antipredator behaviour (avoidance of injury) is exactly where opioidergic modulation should sit. |
| `[RS]` risk-sensitivity (CVaR / distributional / prospect theory) | The "asymmetry in selection — life-vs.-dinner" (Humphreys & Ruxton 2020, ref [38]) is the canonical motivation for risk-sensitive policy: prey need to optimize tail risk, not expectation | Direct license for distributional / CVaR variants of our agent in predator gridworlds. Hand to `professor-rl-bayesian-dl`. |
| `[EE]` exploration–exploitation under bodily-state constraint | "When and where to flee or hide, and know when to not invest in these costly behaviours" is exactly the bodily-cost / explore-exploit trade-off | Ties to Hayden's accept-reject threshold; the PIH adds the predator-risk dimension to that threshold. |
| `[MVT-H]` MVT with internal state | Charnov 1976 cited (ref [79]); used to argue selection on predators sharpens when prey are scarce | Adding homeostatic state $h$ to $R^\star(h)$ (§1.2.1) is the homeostatic extension; combined with predator-presence, $R^\star$ becomes $R^\star(h, \text{risk})$ — the master decision variable for a homeostatic agent in our gridworld. |

### 2.4 Appendix: Section-by-Section Backbone

Section / heading order is the paper's own. Italicized subheads are explicit
in the manuscript.

#### B-2.1 *Abstract*

- Cognitive variation among/within species is hotly debated.
- SIH and EIH are leading hypotheses; both fall short.
- Predator-prey interactions are an under-evaluated driver; recent methods
  make them testable.
- They formalize the **Predatory Intelligence Hypothesis (PIH)**: predator-
  prey cognitive challenges drive a co-evolutionary arms race promoting
  *bidirectional* enhancements in cognition.
- Provides predictions, methodologies, and future directions.

#### B-2.2 *Introduction*

- Defines cognition (Shettleworth 2009, ref [1]): how individuals acquire,
  process, store, and act on environmental information.
- Reviews SIH (Byrne & Whiten; Ashton et al., refs [2, 3]) — but >40% of
  cognitive variation unaccounted for in 103-study meta-analysis (Speechley
  et al. 2024, ref [4]).
- Reviews EIH (Allman; Sol; Rosati; Byrne; Melin; Gibson, refs [5–10]).
  Conflicting results across taxa.
- Introduces predator-prey as a third driver (Byrne & Bates 2007; Amodio et
  al. cephalopods; Dunbar & Shultz, refs [14–16]) — long-noted but
  unevaluated.
- Argues prey rely on **cryptic and indirect cues** (predators are *trying*
  to hide), making this distinct from EIH's vegetative-foraging cues.
- Empirical anchors:
  - Heathcote et al. 2023 (pheasant spatial memory ↔ home range ↔ predation
    risk).
  - Maille & Schradin 2016; Rochais et al. 2023 (African striped mice
    reaction time + spatial memory ↔ wild survival).
  - Kotrschal et al. 2017; Mitchell et al. 2020 (guppy brain / telencephalon
    ↔ predation regime).
  - van der Bijl et al. 2015 (large-brained female guppies assess risk
    faster).
  - Stankowich & Romero 2017 (mammals: physical antipredator defences
    correlate with *reduced* brain size).

#### B-2.3 Figure 1 — Cognitive challenges of predator-prey interactions

- Lists modalities: olfactory, visual, auditory cue learning, recognition,
  recall.
- Decisions: prey when to hide / run / forage; predator whether to attack
  aposematic / dangerous prey.
- Reaction times under selection for prey; likely also for predators.
- Spatial memory (locations of opposite member).
- Social learning (prey only documented; refs [31–33]).
- Group coordination for hunting and defence (refs [35–37]).

#### B-2.4 Box 1 — Glossary

Defines: antipredator behaviour, **Bayesian updating**, cognition, cognitive
performance, inhibitory control, innovation, **Marginal Value Theorem**,
spatial memory, predator-prey interactions, learning, behavioural
flexibility, SIH, EIH, **giving-up density**, **predation pressure**.
*Conceptual scaffolding*: MVT and Bayesian updating are explicitly the formal
machinery the PIH leans on.

#### B-2.5 *The Predatory Intelligence Hypothesis*

- Predation is the dominant evolutionary force; predator-prey selection is
  asymmetric (Humphreys & Ruxton 2020, ref [38]; "life-vs.-dinner principle").
- Examples: curly-tailed lizards × brown anoles (Lapiedra et al. 2018, ref
  [41]); rapid morphological evolution of predators with novel prey (Cattau
  et al. 2017, ref [39]).
- Predator-prey theory: outcomes depend on capacity to appraise the other's
  physical traits + intent, respond during encounters, learn across
  encounters (refs [27, 43]).
- Information processing under noisy / ambiguous cues is cognitively
  demanding (Leavell & Bernal 2019, ref [44]).
- Prey cognition correlates with survival across taxa (refs [4, 12, 21, 22,
  25, 48–50]).
- Predators with larger brains may be better hunters (ref [51]); contested
  for hominids (Faith et al. 2020, ref [52]).
- Predicts geographic variation in predatory regimes generates and maintains
  cognitive variation (Figure 2).

#### B-2.6 *When cognition might not improve survival*

- Brain tissue is metabolically costly (refs [53–55]).
- Trade-offs against reproduction (offspring number / size / frequency, ref
  [56]).
- Trade-offs against longevity, growth rate, age at first reproduction (refs
  [57–61]).
- Energy may be better spent on muscle for raw escape speed (refs [62, 63]).
- Physical defences (armour, spines) reduce need for large brains
  (Stankowich & Romero 2017, ref [26]).
- Crypsis or pursuit-deterrent signalling can substitute for cognitively
  demanding strategies (Cooper & Blumstein 2015, ref [64]).
- Some species' anti-predator strategies are largely cognition-independent.

#### B-2.7 Figure 2 — Predation as evolutionary force

- Schematic: low-predation populations vs. high-predation populations;
  individuals greyed out = predated.
- *Diverse, amplified* predation pressure selects for cognitive performance
  and drives divergence between populations.
- Predicts the same pattern in reverse for predators driven by prey
  functional complexity.

#### B-2.8 *Mechanisms of selection on prey cognition*

- Anti-predator behaviour repertoire (Cooper & Blumstein 2015, ref [65]):
  detect, evade, fight back, escape.
- Cognitive demands: cue memory across modalities, dangerous-area / time
  memory, decision-making, social learning of antipredator behaviour.
- **Bayesian updating** named explicitly as the lifelong process by which
  prey integrate new evidence (refs [34, 44, 66, 67]).
- Differentially advantageous abilities: pattern recognition, learning
  ability, causal understanding, spatial memory.
- Higher predation pressure → larger expected return on cognition.

#### B-2.9 *Mechanisms of selection on predator cognition*

- Predators evolve match-to-prey diel patterns, prey-cue recognition,
  vulnerability-exploiting hunting strategies (refs [68–70]).
- Pattern recognition + learning + spatial memory.
- **Search-image recognition** for cryptic prey (Ishii & Shimada 2010, ref
  [71]).
- Generalist predators (multi-prey-type) face higher cognitive demand than
  specialists (refs [28, 72, 81]).
- Mimicry complexes (e.g. 140-mimic Australian arthropod complex from Pekár
  et al. 2017, ref [73]) impose polymorphic-state cognitive load.
- **MVT (Charnov 1976, ref [79])** invoked: when prey scarce, optimal hunt
  decisions matter more, sharpening selection on predator cognition.
- Dangerous prey require knowledge of how to hunt, ability to plan, ability
  to abort (refs [80]).
- Cooperative hunting at PIH ↔ SIH interface (refs [82, 83]).
- Coevolutionary arms race example: garter snake × newt TTX-resistance
  (Geffeney et al., refs [85, 86]).
- Behavioural plasticity (cognitively underpinned, ref [87]) enables faster
  adjustment than morphological evolution.

#### B-2.10 *Predicting how predation shapes cognition*

- Methodological renaissance:
  - Camera traps (ref [89]); animal-borne cameras; biologgers (ref [90]).
  - Ancient predator-prey network reconstruction (Pleistocene scale, refs
    [91–94, 97]).
  - Functional macroecology (Fricke et al. 2022, ref [95]).
  - Phylogenetic comparative methods (Nakagawa & de Villemereuil 2019;
    Revell & Harmon 2022, refs [98, 99]).
- Network analysis quantifies #predator species × functional richness ×
  functional uniqueness; correlate with brain size as cognitive proxy.

#### B-2.11 *Variation in cognition among species*

- Macroevolutionary tests with phylogenetic comparative methods.
- **Predator-archetype hypothesis** (refs [28, 100]): functionally similar
  predators allow generalization; functionally distinct predators force
  per-predator policies → higher cognitive demand.
- Examples of prey developing per-predator anti-predator strategies (refs
  [101, 102]).
- Symmetric prediction for predators hunting many functionally diverse
  prey species (Henke-von der Malsburg et al. 2020, ref [78]).
- Possibility that one particularly difficult prey species (e.g. dangerous,
  large-bodied) is the dominant driver — testable.

#### B-2.12 *Variation in cognition: from individuals to populations*

- Developmental angle: juveniles most vulnerable; predation may shape
  cognitive development as social environment does.
- Cross-fostering avian / reptilian eggs across high-vs-low-predation
  regimes is a clean experimental design.
- Combine cognitive batteries with predation-risk experiments:
  - **Giving-up density** (Bedoya-Perez et al. 2013; Wooster et al. 2024,
    refs [105, 106]).
  - **Playback experiments** (Palmer et al. 2022 BoomBox, ref [107]).
- Islands, fenced conservation reserves, and conservation havens (Harrison
  et al. 2023; Moseby et al. 2018, refs [108, 109]) as natural-experiment
  systems with simplified or manipulable trophic networks.
- Mainland-vs-island prediction: lower dietary diversity on small islands
  reduces cognitive selection (Gavriilidi et al. 2022, ref [110]).
- Predator cognition × hunting performance is "an unexplored avenue."

#### B-2.13 Table 1 — Key predictions

Reproduced (paraphrased) for the record:

**Both predators and prey.** Behavioural flexibility, cooperation, spatial
memory, associative learning are positively associated with predatory and
anti-predatory success. Methods: cognitive test protocols + biologgers +
foraging / avoidance experiments; split-clutch guppy designs.

**Prey-specific.**

- Prey under multiple, functionally distinct predator threat → enhanced
  cognition vs. single-predator. Methods: network analysis + brain size;
  island / havened-population comparisons.
- Associative learning, spatial memory, causal understanding (and
  underpinning neural architecture / neurotransmitters / genes) positively
  associated with appropriate antipredator behaviour. Methods: cognitive
  testing + giving-up densities / playback.
- Cognitive evolution is linked to predation risk through deep time —
  bidirectional brain-size response to predator-diversity shifts. Methods:
  network analysis + brain-size shifts mapped against community assembly.

**Predator-specific.**

- Generalist predators (high prey functional diversity) → enhanced cognition
  vs. specialists. Methods: network analysis + brain size.
- **Mesopredators** > apex predators in cognition (because they both hunt
  and avoid). Methods: closely-related apex/mesopredator pairs (e.g.
  canids).
- Cooperatively-hunting predators > similar-sized solitary predators in
  cognition. Methods: closely-related cooperative vs. solitary pairs (e.g.
  group vs. solitary felids).

#### B-2.14 *Conclusions*

- Existing SIH and EIH explanatory power is real but incomplete.
- The PIH formalizes predation as a driver and provides a testable
  prediction set.
- Recent technological advances make the program tractable.
- Hope: stimulate new research to address unaccounted-for cognitive
  variation.

---

## 3. Cross-Paper Synthesis & Hand-offs

### 3.1 Where Hayden 2018 and Wooster 2026 meet

Both papers converge on the same primitive — *encounter-driven, partially-
observable, repeated decision-making against a slowly-updating background* —
but cover its complementary halves. Hayden specifies the **internal
implementation** (accept-reject vs. background, dACC/vmPFC competition,
race-to-threshold, choice-as-stopping). Wooster specifies the **selective
pressure** (predator-prey arms race, asymmetric life-vs.-dinner mortality,
cryptic-cue Bayesian updating) that *forced* that internal implementation to
exist. The shared explicit mathematical primitive is **Charnov MVT**
(Hayden ref [5]; Wooster ref [79]); the shared implicit primitive is
**Bayesian / SPRT cue integration** (Hayden's "evidence accumulation," refs
[36, 48–50]; Wooster's "Bayesian updating" in Box 1).

For our gridworld:

- The **decision surface** is Hayden's: profitability of foreground vs.
  background $R^\star$.
- The **threshold's modulators** are Wooster's: predator presence /
  diversity raises the cost of accept errors, raising the bar to release a
  primed approach action.
- The **homeostatic extension** $R^\star \to R^\star(h, \text{risk})$ is
  the unique synthesis that neither paper fully states; this is where our
  project sits.

### 3.2 Cross-Paper Threads to Hand Off

Two-paper corpora rarely yield clean cross-paper threads, but in this case
both papers point at the same handful of mechanisms from opposite
directions. The following threads are *jointly* supported and worth handing
to specific professor agents:

| # | Thread | Joint support | Owner agent |
|---|---|---|---|
| 1 | **State-dependent MVT threshold $R^\star(h, \text{risk})$ as the master decision variable.** Hayden's "background value" is Charnov's MVT optimum repurposed neurally; Wooster invokes Charnov for predator decisions. Neither paper writes the homeostatic extension. Concrete experiment: in a gridworld with satiation/hydration drives + a predator, fit $R^\star(h, \text{risk})$ from patch-leaving times. **`[MVT-H] [EE]`** | Hayden §1.2.1 + Wooster §2.2.1 | `professor-bayesian-brain` (theoretical formulation), then `experiment-designer` for the gridworld instantiation |
| 2 | **Vigilance / hypervigilance as a learned, cost-aware Bayesian threshold (formal "injury → hypervigilance" hypothesis).** Wooster's Bayesian-updating prey is an SPRT agent with a flight threshold; Hayden's accept-reject brain is the same SPRT under a different cost geometry. Injury / opioid drop should *lower* the flight threshold. Concrete prediction: post-injury agents in our gridworld should evidence faster freezing onset and longer freezing duration at the cost of foraging rate, *recovering* as the opioid analog returns to baseline. **`[HV] [PC] [NM-OPI] [NM-NE]`** | Hayden §1.2.2 + Wooster §2.2.2 | `professor-pain-modeling` (construct validity of "hypervigilance" claim), `professor-neuromodulation` (opioid/NE mechanism), `professor-bayesian-brain` (SPRT formalism) |
| 3 | **Threshold modulation as the natural niche for ascending neuromodulators.** Hayden explicitly cites threshold-modulation control systems (refs [20, 21, 22]); Wooster's "filter information against background noise + rapid integration under risk" is the same target. Concrete prediction: tonic-NE-analog gain on cue precision, tonic-DA-analog gain on $R^\star$, ACh-analog on cue-likelihood weighting, all collapse to *threshold parameters in the SPRT*. **`[NM-NE] [NM-ACh] [NM-DA] [PC]`** | Hayden §1.2.2 + Wooster §2.2.2, §2.2.4 | `professor-neuromodulation` (which knob maps to which ascending system), then `professor-rl-bayesian-dl` (FiLM / hypernet implementation) |
| 4 | **Risk-asymmetry → distributional / CVaR policy as biologically natural.** Wooster's "life-vs.-dinner" asymmetry (Humphreys & Ruxton 2020, ref [38]) makes prey *tail-risk* optimizers, not expected-value optimizers; Hayden's drift-diffusion against background can be CVaR-shifted by replacing the scalar $\pi_{\text{fg}}$ with a quantile. Concrete prediction: distributional / CVaR variants of our agent should outperform expectation-only variants under predator regimes but not under predator-free regimes. **`[RS] [PC]`** | Hayden §1.2.2 + Wooster §2.5 (life-vs.-dinner asymmetry) | `professor-rl-bayesian-dl` |
| 5 | **Functional diversity of predators → representational capacity demand → architecture-scaling prediction.** Wooster's multi-predator / functional-diversity prediction is information-theoretically tight (§2.2.4). For a fixed-capacity neural network agent, scaling the *number of functionally distinct hazards* should reveal a capacity threshold beyond which performance collapses unless we add modular / FiLM-conditional architecture. This is a clean experiment-designer brief for our existing FiLM stack. **`[EE]`** | Wooster §2.2.4, §2.11; Hayden's serial-attention bottleneck §1.2.3 | `experiment-designer` (paired with `professor-rl-bayesian-dl` for FiLM hypothesis); supersession candidate for current single-predator gridworld configs |

### 3.3 Single-paper standout findings

Beyond the cross-paper threads, two single-paper standouts deserve their own
hand-off because they are not really represented in the other paper:

- **(Hayden only)** *Choice ≡ stopping*. This is the unification claim that
  collapses economic choice into the stop-signal literature. For our project
  this licenses a cleaner action-release / motor-inhibition decomposition in
  the agent (release primed-action vs. inhibit), and it tracks clean
  neuroanatomy (motor / premotor cortex). Owner: `professor-rl-bayesian-dl`
  for the architectural proposal; `senior-developer` would be the eventual
  recipient if the team chooses to refactor the policy head.

- **(Wooster only)** *Mesopredators > apex predators in cognition.* This is
  a clean differential prediction of the PIH that has no analog in Hayden.
  In our gridworld it would translate to: an agent that *is both predator
  and prey* (e.g. a mid-level animal that hunts smaller agents while being
  hunted by a larger one) should show higher learned-vigilance and
  policy-flexibility than agents in the pure-prey or pure-predator role. A
  lab-scale digital test of the mesopredator prediction is unusually
  cheap. Owner: `experiment-designer` (paired with `professor-pain-modeling`
  for the construct-validity guardrails).

### 3.4 What this corpus does *not* cover

- No formal model of opioidergic modulation. Cross-link to
  `computational_pain_models_lit_review.md` is needed before the `[NM-OPI]`
  thread above can be made fully concrete.
- No FiLM / hypernet / Bayesian-DL implementation discussion. Cross-link to
  the (forthcoming) FiLM lit review is required for thread #5.
- No explicit treatment of *interoception* as a state on which $R^\star$
  depends. The homeostatic extension in §3.2 thread #1 is *our* synthesis;
  neither Hayden nor Wooster writes it down.
- No discussion of perceptual noise *injection* mechanisms (see
  `perceptual_noise_lit_review.md` for that complementary thread).

---

## Provenance

- Source PDFs: `docs/project/references/foraging_for_cognitive_evolution/` (2
  files, 27 pp total).
- Extraction tool: `pdfplumber` (v0.11.9), Python 3 system interpreter.
- Working backbone files: `tmp/20260507_173656_foraging/{hayden_2018,wooster_2026}_full.txt`.
- Progress snapshot: `tmp/20260507_173656_foraging_cognitive_evolution_progress.md`.
- Original NotebookLM context (auth stale, PDFs used directly):
  https://notebooklm.google.com/notebook/975c86b2-23b8-4362-b986-29c4cc1020e4

