# Perceptual Decision-Making: Reference Review

> Source: NotebookLM notebook `35231efb-751f-461c-b36c-aca4a403afc2`
> Scope: Neurobiology and computational modeling of perceptual decision-making, from foundational primate single-unit studies to cross-species comparisons and human neuroimaging / neurally-informed modeling.
> Last updated: 2026-04-16

## Table of Contents

1. [Gold & Shadlen (2007) — The Neural Basis of Decision Making](#1-gold--shadlen-2007--the-neural-basis-of-decision-making)
2. [Hanks & Summerfield (2017) — Perceptual Decision Making in Rodents, Monkeys, and Humans](#2-hanks--summerfield-2017--perceptual-decision-making-in-rodents-monkeys-and-humans)
3. [O'Connell & Kelly (2021) — Neurophysiology of Human Perceptual Decision-Making](#3-oconnell--kelly-2021--neurophysiology-of-human-perceptual-decision-making)
4. [Cisek (2021) — Evolution of Behavioural Control from Chordates to Primates](#4-cisek-2021--evolution-of-behavioural-control-from-chordates-to-primates)
5. [Fleming (2024) — Metacognition and Confidence: A Review and Synthesis](#5-fleming-2024--metacognition-and-confidence-a-review-and-synthesis)
6. [Gershman et al. (2024) — Explaining Dopamine Through Prediction Errors and Beyond](#6-gershman-et-al-2024--explaining-dopamine-through-prediction-errors-and-beyond)
7. [Smith, Friston & Whyte (2022) — A Step-by-Step Tutorial on Active Inference](#7-smith-friston--whyte-2022--a-step-by-step-tutorial-on-active-inference)
8. [LeDoux & Daw (2018) — Surviving Threats: Taxonomy of Defensive Behaviour](#8-ledoux--daw-2018--surviving-threats-taxonomy-of-defensive-behaviour)
9. [Tashjian, Zbozinek & Mobbs (2021) — A Decision Architecture for Safety Computations](#9-tashjian-zbozinek--mobbs-2021--a-decision-architecture-for-safety-computations)
10. [Levy & Schiller (2021) — Neural Computations of Threat](#10-levy--schiller-2021--neural-computations-of-threat)
11. [Wiech (2016) — Deconstructing the Sensation of Pain](#11-wiech-2016--deconstructing-the-sensation-of-pain)
12. [Seymour, Crook & Chen (2023) — Post-Injury Pain and Behaviour: A Control Theory Perspective](#12-seymour-crook--chen-2023--post-injury-pain-and-behaviour-a-control-theory-perspective)
13. [Mahajan & Seymour (2025) — Forward and Reverse Engineering the Pain System](#13-mahajan--seymour-2025--forward-and-reverse-engineering-the-pain-system)

---

## 1. Gold & Shadlen (2007) — The Neural Basis of Decision Making

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Gold and Shadlen ask how the brain implements the computational elements of a simple perceptual or motor decision — i.e. how noisy, fleeting sensory inputs are converted into a categorical choice. They frame deliberation as a problem of **statistical inference**: the brain must weigh evidence over time to choose between competing hypotheses.

**Key Findings.**
- The brain distinguishes between the *transient encoding of sensory evidence* and a continuously updating **decision variable (DV)**.
- In the random-dot motion (RDM) task, sensory neurons in visual area **MT** encode the momentary evidence about motion direction.
- Downstream sensorimotor-association neurons (notably in the **lateral intraparietal area, LIP**) *accumulate* this noisy evidence over time, producing a DV.
- A choice is committed when accumulated LIP activity reaches a stereotyped firing-rate **bound**; the corresponding saccade follows ~70 ms later, regardless of stimulus strength or reaction time.

**Main Methodology.** Single-unit recordings in behaving macaques performing RDM and vibrotactile-frequency-discrimination tasks, correlating trial-by-trial firing rates with choices and reaction times. Causality was established by **microstimulation** of MT (biasing perceived motion) and LIP (biasing choice). The neural dynamics were mapped onto **Signal Detection Theory (SDT)** and **Sequential Analysis** (Wald's SPRT / drift-diffusion).

**Initial Takeaway.** The paper is foundational because it provides a direct, mechanistic bridge from abstract cognitive psychology to neurobiology: specific neural circuits literally compute the mathematical integrals predicted by SPRT, giving a universal biological framework for how the brain trades off speed, accuracy, expectations, and value.

### Phase 2: Graduate-Level Deep Dive

> Note: core equations below are from Gold & Shadlen (2007). The paper cites Green & Swets (1966) and Wald (1947) for derivations; step-by-step derivations shown here are reconstructed from standard statistical-decision-theory material and should be independently verified.

#### 2.1 Signal Detection Theory (SDT)

For a binary decision between hypotheses $h_1, h_2$ given noisy evidence $e$, the likelihoods are $P(e \mid h_1)$ and $P(e \mid h_2)$. The **likelihood ratio** is

$$
l_{12}(e) \;\equiv\; \frac{P(e \mid h_2)}{P(e \mid h_1)} .
$$

The decision rule is: choose $H_1$ if $l_{12}(e) \ge \beta$, with the criterion $\beta$ set by the observer's goal:

- Maximize accuracy, equal priors: $\beta = 1$.
- Maximize accuracy, unequal priors: $\beta = \dfrac{P(h_1)}{P(h_2)}$.
- Maximize expected value: $\displaystyle \beta = \frac{(v_{11} + v_{12}) \, P(h_1)}{(v_{22} + v_{21}) \, P(h_2)}$ where $v_{ij}$ is the value of choosing $H_j$ when $h_i$ holds.

**Derivation of the value-maximizing criterion.** Expected values of each choice given posterior beliefs:

$$
\begin{aligned}
EV(H_1) &= P(h_1 \mid e)\, v_{11} + P(h_2 \mid e)\, v_{21}, \\
EV(H_2) &= P(h_1 \mid e)\, v_{12} + P(h_2 \mid e)\, v_{22}.
\end{aligned}
$$

At the optimal boundary $EV(H_1) = EV(H_2)$:

$$
P(h_1 \mid e)\,(v_{11} - v_{12}) \;=\; P(h_2 \mid e)\,(v_{22} - v_{21}).
$$

(The form with additive $v_{11} + v_{12}$ in Gold & Shadlen arises when $v_{12}, v_{21}$ are treated as non-negative *penalty magnitudes* rather than signed values.) Applying Bayes' theorem $P(h_i \mid e) = P(e \mid h_i) P(h_i) / P(e)$ to both sides and cancelling $P(e)$:

$$
\frac{P(e \mid h_1) P(h_1)}{1} \,(v_{11} + v_{12}) \;=\; \frac{P(e \mid h_2) P(h_2)}{1} \,(v_{22} + v_{21}),
$$

$$
\boxed{\;\frac{P(e \mid h_2)}{P(e \mid h_1)} \;=\; \frac{(v_{11}+v_{12})\,P(h_1)}{(v_{22}+v_{21})\,P(h_2)} \;=\; \beta.\;}
$$

#### 2.2 Sequential Analysis and the Sequential Probability Ratio Test (SPRT)

With a sequence of i.i.d. evidence samples $e_1, e_2, \dots, e_n$, the joint likelihood factorizes, so the log-likelihood ratio (the *weight of evidence*) is additive:

$$
\mathrm{logLR}_{12} \;\equiv\; \log \frac{P(e_1,\dots,e_n \mid h_2)}{P(e_1,\dots,e_n \mid h_1)} \;=\; \sum_{i=1}^{n} \log \frac{P(e_i \mid h_2)}{P(e_i \mid h_1)}.
$$

Define the per-sample weight and the running decision variable

$$
w_i \;=\; \log \frac{P(e_i \mid h_2)}{P(e_i \mid h_1)}, \qquad y_n \;=\; \sum_{i=1}^{n} w_i .
$$

Accumulation continues until $y_n$ crosses one of two bounds:

$$
\text{choose } H_1 \text{ if } y_n \ge \log\tfrac{1-\alpha}{\alpha}, \qquad
\text{choose } H_2 \text{ if } y_n \le \log\tfrac{\beta}{1-\beta},
$$

where $\alpha = P(\text{choose } H_1 \mid h_2)$ (Type I error) and $\beta = P(\text{choose } H_2 \mid h_1)$ (Type II error).

**Derivation of SPRT bounds (Wald's approximation).** At the upper bound $A$, the accumulated likelihood ratio satisfies $\mathrm{LR} \approx e^{A}$. Partitioning choices by ground truth:

$$
e^{A} \;\approx\; \frac{P(\text{choose } H_1 \mid h_2)}{P(\text{choose } H_1 \mid h_1)} \;=\; \frac{\alpha}{1-\beta},
$$

giving $A \approx \log\frac{1-\beta}{\alpha}$. Symmetrically, the lower bound is $B \approx \log\frac{\beta}{1-\alpha}$. Under symmetric error rates ($\alpha = \beta$) these reduce to $\pm \log\frac{1-\alpha}{\alpha}$; with $\alpha = \beta = 0.05$ this gives $|y_n| \ge \log 19$, matching the paper.

#### 2.3 Continuous-Time Limit: Drift-Diffusion and the Speed–Accuracy Tradeoff

In the continuous-time limit with Gaussian evidence, the SPRT becomes **bounded diffusion with drift**:

$$
dy \;=\; \mu\, dt + \sigma\, dW_t, \qquad y(0)=0,
$$

with absorbing bounds $\pm A$. Drift $\mu$ represents signal strength (e.g. motion coherence) and $\sigma$ the evidence noise.

**LATER model (simple RT).** A DV rises linearly from baseline $S_0$ with rate $r$ to a threshold $S_T$ at time $t_i$:

$$
r \;=\; \frac{S_T - S_0}{t_i}, \qquad r \sim \mathcal{N}(\mu, \sigma^2).
$$

Because $r$ is Gaussian, the *reciprocals* of the RTs ($1/t_i$) are Gaussian, producing the characteristic right-skewed RT distribution; the harmonic-mean RT is

$$
\mathrm{RT}_{\text{harmonic}} \;=\; \frac{S_T - S_0}{\mu}.
$$

**Speed–accuracy tradeoff.** The bound distance $S_T - S_0$ directly controls the tradeoff: raising the criterion reduces the false-alarm rate (higher accuracy) but requires more samples before crossing (longer RT). Formally, with symmetric bounds $\pm A$ and drift $\mu$ in the Wiener process, the error rate and mean decision time obey

$$
P(\text{error}) = \frac{1}{1+e^{2A\mu/\sigma^2}}, \qquad
\langle T_{\text{dec}}\rangle = \frac{A}{\mu}\tanh\!\frac{A\mu}{\sigma^2},
$$

both monotonic in $A$ — larger $A$ → lower error, longer $T$.

#### 2.4 Neural Implementation: from $w_i$ to LIP firing rates

1. **Momentary evidence $w(t)$ in MT.** For the RDM task with oppositely tuned MT pools, the instantaneous logLR is well-approximated by the difference of their firing rates:

   $$
   w(t) \;\approx\; \mathrm{Rate}_{\text{MT,Right}}(t) \;-\; \mathrm{Rate}_{\text{MT,Left}}(t).
   $$

2. **Accumulator in LIP.** LIP neurons whose response field includes the choice target integrate the MT difference:

   $$
   \mathrm{FiringRate}_{\text{LIP}}(t) \;\propto\; \int_{0}^{t} \bigl(\mathrm{Rate}_{\text{MT,Right}}(\tau) - \mathrm{Rate}_{\text{MT,Left}}(\tau)\bigr)\, d\tau,
   $$

   i.e. LIP activity tracks the continuous-time SPRT sum $y(t)$.

3. **Neural bound.** In free-response RDM, LIP firing rates ramp up and reach a **stereotyped threshold** roughly 70 ms before saccade onset, independent of coherence or RT. This fixed neural firing-rate ceiling physically instantiates the SPRT bound $A$.

**Bottom line.** The rate of LIP firing *is* a statistical decision variable: the brain achieves (near-)optimal inference by physically integrating differences in sensory spike rates to a hard firing-rate bound.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order, compact rewrite:

- **Introduction.** Decision = deliberative commitment to a categorical proposition; laboratory sensory-motor tasks chosen for precise control; goal is principles that scale to higher cognition.
- **Elements of a decision (Fig. 1).** Prior $P(h_i)$, evidence $e$ with likelihoods $P(e\mid h_i)$, and value $v$ accumulate into a **decision variable (DV)**; a decision rule applies a boundary; evaluation / learning closes the loop.
- **SDT framework.** $l_{12}(e) = P(e\mid h_2)/P(e\mid h_1)$; choose $H_1$ if $l_{12} \ge \beta$; three goal-dependent criteria (accuracy equal priors, accuracy unequal priors, expected value) as in existing §1 Phase 2.1.
- **Sequential analysis.** logLR as additive DV under independence, first-passage to bounds — bounded diffusion (existing §1 Phase 2.2–2.3).
- **SPRT (Box).** Wald's minimum-sample-count binary test (existing §1 Phase 2.2 derivation).
- **Experimental survey.** VTF → RDM → heading → disparity → face/object → olfaction → detection; across tasks, dissociation of *momentary evidence* (early sensory cortex) from a persistent *DV* (sensorimotor / association areas). Key RDM result: LIP ramps to a stereotyped pre-saccadic threshold ≈ integral of MT left/right difference (existing §1 Phase 2.4).
- **LATER / simple motor latencies.** $r \sim \mathcal N(\mu,\sigma^2)$; harmonic RT $= (S_T - S_0)/\mu$; prior-probability ⇒ *log* threshold shifts ⇒ DV in log-probability units; countermanding ⇒ race between independent go/stop LATER processes.
- **Value-based decisions.** Value enters DV as additive bias; apparent matching-law randomness may be deterministic DV + Poisson decision rule.
- **Conclusions & Future issues.** MSPRT for $>2$ alternatives, abstract motor-independent DVs, explicit-randomness regimes, experience-driven tuning — all flagged as open problems.

### Appendix: Section-by-Section Backbone

**Introduction.** Decision defined as deliberative commitment to a categorical proposition ("judge/jury weighing evidence"); review focuses on simple sensory-motor tasks for controllability; goal is scalable principles.

**Elements of a Decision (Fig. 1).** Prior $P(h_i)$; evidence $e$ with conditional likelihoods $P(e\mid h_i)$; value $v$ (subjective cost/benefit); DV accumulates support/opposition; decision rule applies boundary (criterion); concludes with behaviour + performance monitoring feeding learning.

**Conceptual Framework — Signal Detection Theory.**
- LR DV: $l_{12}(e) \equiv P(e\mid h_2)/P(e\mid h_1)$.
- Rule: choose $H_1$ if $l_{12}(e) \ge \beta$.
- Criterion settings: $\beta = 1$ (accuracy, equal priors); $\beta = P(h_1)/P(h_2)$ (accuracy, unequal priors); $\beta = (v_{11}+v_{12})P(h_1)/[(v_{22}+v_{21})P(h_2)]$ (expected value).
- Caveat: any neural quantity monotone in LR is observationally equivalent ⇒ cannot dissociate priors / evidence / value from SDT alone.

**Conceptual Framework — Sequential Analysis.**
- Time-extended DV as logLR: $\sum_i \log[P(e_i\mid h_2)/P(e_i\mid h_1)]$.
- Isomorphic to bounded diffusion / random walk (Fig. 2).
- Temporal evolution enables dissociating transient evidence from an evolving DV — the key methodological move.

**Box — Sequential Probability Ratio Test (SPRT).**
- Toy example: trick coin at 60% heads; per-toss weight of evidence $w_i$.
- Wald's theorem: SPRT minimizes expected sample count for a fixed error rate in binary hypotheses.

**Experiments — Perceptual Tasks.**
- *VTF discrimination (Fig. 3).* S1 firing rate $\propto$ stimulus frequency, weak trial-by-trial choice probability ⇒ momentary evidence, not DV. S2, dlPFC, MPC, VPC carry persistent firing through inter-stimulus delay and reflect the f2 vs f1 comparison ⇒ working-memory DV. Identification ambiguity (memory vs predictive DV).
- *RDM direction discrimination (Figs. 4–6).* MT/V5 = direction-selective sensory evidence (firing correlates behaviour; microstim biases choices). DV in LIP, SC, FEF. LIP ramp reaches stereotyped firing threshold ~70 ms pre-saccade regardless of motion strength; ramp ≈ integral of MT right-left difference. Microstim differentiates: MT stim = drift-rate offset; LIP stim = DV offset (shifts starting point, not slope).
- *Heading (MST, VIP).* MST carries evidence (optic flow), but microstim effect on heading choice is weak/mixed due to competing vestibular inputs in MST; DV must live downstream (e.g. LIP or premotor).
- *Disparity (MT).* MT jointly encodes direction + disparity; in direction tasks, microstim at strongly disparity-tuned MT sites has *weaker* behavioural effect ⇒ dynamic, context-dependent readout discounts irrelevant dimension. Ambiguous cylinder (transparent motion) resolution yields very high choice probability in MT ⇒ top-down feedback.
- *Face/object (IT, dlPFC).* IT microstim biases face vs non-face choice ⇒ IT = causal evidence. Human fMRI: dlPFC BOLD $\propto$ $(V_{\text{face area}} - V_{\text{house area}})$ ⇒ candidate DV. EEG: early ~170 ms evidence ERP; late ~300 ms difficulty-scaled DV ERP.
- *Olfaction (rodent).* Olfactory-bulb activation patterns = evidence. Single-sniff design hides integration; speed-accuracy pressure reveals ~400 ms integration window, comparable to primate visual timescale.
- *Detection (Figs. 7–8).* Motion-detection RDM: MT population rate rises with motion onset, correlates with RT (evidence). VIP aligns to behavioural lever-release (DV / decision outcome). Vibrotactile detection: S1 tracks stimulus intensity (evidence); MPC activity modulated by the report (DV / outcome).

**Simple Motor Latencies — LATER (Fig. 9).**
- DV rises linearly from baseline $S_0$ with rate $r$ to threshold $S_T$ at time $t_i$; $r \sim \mathcal N(\mu, \sigma^2)$ per-trial.
- Because $r$ is Gaussian, $1/t_i$ is Gaussian; harmonic mean RT $= (S_T - S_0)/\mu$.
- Manipulating target prior probability shifts $\log(P)$ *linearly* in RT distributions ⇒ DV operates in log-probability units.
- Neural: SC baseline activity scales with target probability; countermanding tasks ⇒ independent go / stop LATER races in FEF and SC.

**Value-Based Decisions.**
- Neural value code in OFC, ACC, basal ganglia; value signals also in LIP, dlPFC.
- Within SDT/SA, value modifies DV the same way a prior does (additive bias).
- Apparent exploration / matching-law randomness could be a deterministic DV with a Poisson read-out at the decision-rule stage ⇒ randomness located at rule, not at integration.

**Conclusions.**
- Decision neurobiology maps cleanly onto SDT + SA formalisms; even overtrained reflexive sensorimotor choices involve deliberative elements.
- Open: MSPRT for $>2$ alternatives; motor-independent / abstract decisions.

**Summary Points.**
1. Decision = prior + evidence + value → categorical commitment.
2. SDT + SA give the theoretical skeleton.
3. Monkey physiology has begun to biologically instantiate the skeleton.
4. Evidence (transient, early sensory) vs DV (persistent, sensorimotor) is the key neural dissociation.
5. Speed-accuracy trade-off explained by DV relative to a criterion.

**Future Issues.**
1. Where / how priors + evidence + value combine.
2. DV as abstraction vs explicit neural representation.
3. Temporal-integration mechanism.
4. Bound implementation.
5. $>2$-alternative scaling.
6. When explicit randomness is used by the rule.
7. Motor-uncoupled decisions.
8. Experience-driven tuning of the DV.

---

## 2. Hanks & Summerfield (2017) — Perceptual Decision Making in Rodents, Monkeys, and Humans

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Classical perceptual-decision neuroscience was built on a *primate-centric* canon in which noisy evidence is integrated to a bound, largely in parietal areas such as LIP. Hanks & Summerfield (Neuron, 2017) ask how the field changes when we add modern rodent toolkits (optogenetics, large-scale perturbation, Poisson-click tasks) and human non-invasive recordings (EEG, MEG, fMRI, TMS) alongside macaque physiology.

**Key Findings.**
- **Heterogeneous / mixed selectivity.** Putative "decision" neurons in monkey LIP and PFC are not pure integrators; they multiplex sensory, motor, and decision variables.
- **Parietal cortex is not an obligatory hub.** Reversible inactivation of LIP (monkey) and PPC (rat) barely affects perceptual choice in many tasks — suggesting these areas contribute but are not the causal bottleneck of accumulation.
- **Rodent frontal vs. parietal dissociation.** FOF (frontal orienting fields) shows *categorical, step-like* choice encoding; PPC shows more graded, linear ramps consistent with accumulated evidence.
- **Urgency vs. collapsing bounds.** Rather than an algorithmically collapsing bound, a stimulus-independent *urgency* signal may drive neural activity toward a fixed threshold, producing the same speed-accuracy behaviour.
- **Human cognitive architecture.** CPP, parietal alpha-band oscillations, and fMRI reveal rhythmic evidence weighting and central-bottleneck limits on decision formation.

**Main Methodology.** A review synthesizing species-specific methods: macaque single-unit / microstimulation; rodent optogenetics, cell-type-specific viral tagging, and Poisson-click accumulation tasks; human fMRI / TMS / EEG-MEG with CPP as a supramodal DV marker; all tied to DDM / SPRT / urgency-gating frameworks.

**Initial Takeaway.** A paradigm shift: from localizing an abstract "accumulator box" to mapping the distributed microcircuitry that implements urgency, confidence, and context-dependent value across species.

### Phase 2: Graduate-Level Deep Dive

> The source paper is a review; the equations below are the standard formulations from the foundational works it cites (Bogacz et al. 2006; Kiani & Shadlen 2009; Brunton et al. 2013; Cisek et al. 2009).

#### 2.1 Canonical Drift-Diffusion Model (DDM)

The cross-species DDM is the SDE

$$
dx \;=\; A\, dt \;+\; c\, dW_t, \qquad x(0) = 0,
$$

with absorbing bounds $\pm a$; $A$ is the drift rate (signal), $c$ the noise scale, $W_t$ standard Brownian motion.

**Forward (Fokker–Planck) equation** for the density $p(x,t)$:

$$
\frac{\partial p}{\partial t} \;=\; -A\, \frac{\partial p}{\partial x} \;+\; \frac{c^{2}}{2}\, \frac{\partial^{2} p}{\partial x^{2}}.
$$

**Derivation of $P(\text{correct})$ (backward equation).** Let $P(x)$ = probability of reaching $+a$ starting from $x \in (-a,a)$. In steady state,

$$
A\, \frac{dP}{dx} + \frac{c^{2}}{2}\, \frac{d^{2}P}{dx^{2}} \;=\; 0.
$$

Characteristic equation $A\lambda + \tfrac{c^2}{2}\lambda^2 = 0 \Rightarrow \lambda_1 = 0,\ \lambda_2 = -2A/c^2$, so

$$
P(x) \;=\; C_1 + C_2\, e^{-2Ax/c^{2}}.
$$

Boundary conditions $P(+a)=1,\ P(-a)=0$:

$$
C_1 + C_2 e^{-2Aa/c^{2}} = 1, \qquad C_1 + C_2 e^{+2Aa/c^{2}} = 0.
$$

Subtracting gives $C_2 (e^{-2Aa/c^{2}} - e^{+2Aa/c^{2}}) = 1$, and evaluating at $x_0 = 0$:

$$
\boxed{\;P(\text{correct}) \;=\; \frac{1}{1 + e^{-2 A a / c^{2}}}.\;}
$$

**Mean decision time.** The expected first-passage time $\bar T(x)$ satisfies

$$
A\, \frac{d\bar T}{dx} + \frac{c^{2}}{2}\, \frac{d^{2}\bar T}{dx^{2}} \;=\; -1,
$$

with $\bar T(\pm a)=0$. Solving and adding a non-decision time $t_{nd}$ gives

$$
\boxed{\;\langle \mathrm{RT}\rangle \;=\; \frac{a}{A}\,\tanh\!\left(\frac{A a}{c^{2}}\right) + t_{nd}.\;}
$$

#### 2.2 Drift Rate ↔ Stimulus Evidence (Psychometric Function)

With drift proportional to stimulus strength (e.g. motion coherence $C$, click ratio), $A = k\,C$. Substituting into $P(\text{correct})$:

$$
P(\text{correct}\mid C) \;=\; \frac{1}{1 + e^{-\beta C}}, \qquad \beta = \frac{2 k a}{c^{2}},
$$

the standard logistic psychometric function; $\beta$ aggregates sensitivity $k$, bound $a$, and noise $c$.

#### 2.3 Collapsing Bounds vs. Urgency-Gating

**Collapsing bound.** Replace $\pm a$ with time-varying $\pm b(t)$, e.g.

$$
b(t) \;=\; a\, e^{-t/\tau},
$$

so late-arriving weak evidence can still commit a choice. Decision at $|x(t)| \ge b(t)$.

**Urgency-gating (Cisek et al.).** Low-pass-filtered momentary evidence $E(\tau)$ is *multiplied* by a monotonically increasing, stimulus-independent urgency $u(t)$ and compared to a fixed bound:

$$
x(t) \;=\; u(t) \cdot \!\int_{0}^{t}\! E(\tau)\, d\tau, \qquad \text{decide at } |x(t)| \ge a.
$$

Because $u(t)$ multiplies both signal and noise, neural gain grows with elapsed time without any *physical* collapse of the firing-rate threshold — matching LIP/FOF data that show a fixed pre-saccadic rate.

#### 2.4 Rodent Poisson-Click Accumulator (Brunton, Brody et al. 2013)

For the auditory Poisson-click task used in rat PPC/FOF studies, the accumulator $a(t)$ is

$$
da \;=\; \bigl( \delta_{t,t_R}\, \eta_R \;-\; \delta_{t,t_L}\, \eta_L \bigr) \;+\; \lambda\, a\, dt \;+\; \sigma_{a}\, dW_t,
$$

with:

- $\delta_{t,t_{R/L}}$ — Dirac deltas at right/left click times.
- $\eta_{R/L} \sim \mathcal{N}(1, \sigma_s^{2})$ — per-click magnitude with **sensory noise** $\sigma_s^{2}$.
- $\lambda$ — memory parameter: $\lambda<0$ leaky, $\lambda>0$ unstable, $\lambda=0$ perfect integration.
- $\sigma_a^{2}$ — **accumulator noise** (diffusion per unit time).
- $a(0) \sim \mathcal{N}(0, \sigma_i^{2})$ — **initial-condition noise**.
- Sticky bounds $\pm B$: once reached, further input is ignored.

Behavioural fits (Brunton 2013) yield near-zero $\lambda$, small $\sigma_a^{2}$, and large $\sigma_s^{2}$ — most of the noise in rat decisions is sensory, not accumulation noise, a claim consistent with the review's argument that parietal "accumulators" may not be the bottleneck.

#### 2.5 Neural Confidence: Kiani–Shadlen Log-Odds Map

When drift $A$ is drawn from a prior $f(A)$ over difficulties, the observer must infer reliability from $(x,t)$. Let $p(x,t\mid A)$ = unabsorbed density; $P(x\to\text{bound}_{\text{correct}}\mid x, A)$ = prob. of eventually crossing the correct bound given drift $A$. Marginalizing:

$$
P(\text{correct}\mid x,t) \;=\; \frac{\displaystyle \int P(x\to \text{bound}_{\text{correct}}\mid x, A)\, p(x,t\mid A)\, f(A)\, dA}{\displaystyle \int p(x,t\mid A)\, f(A)\, dA}.
$$

Confidence is then the log-odds map

$$
L(x,t) \;=\; \log\frac{P(\text{correct}\mid x,t)}{P(\text{error}\mid x,t)}.
$$

Empirically, LIP firing rates at a given $(x,t)$ track this $L(x,t)$ surface, so decision areas *simultaneously* encode the choice (via bound-crossing) and its subjective certainty.

**Cross-species implication.** The same $(x,t)$→confidence map is expected to appear in rodent FOF/PPC and in the human CPP; dissociations between choice and confidence (e.g. opt-out tasks) provide the critical tests highlighted by the review.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Introduction.** Canonical primate-DDM view has diversified in the last 5 years with rodent tools and human non-invasive recordings; the review synthesizes convergences and open challenges.
- **Canonical perspectives.** DDM: linear integration of noisy evidence to a fixed bound; LIP/FEF ramp-to-threshold in RDK; MT vs LIP microstimulation dissociates evidence from integration (existing §2 Phase 2.1).
- **§1.1 Monkey challenges.** (i) LIP mixed selectivity — aggregate ramp may not be a single-neuron property; (ii) ramping vs stepping single-trial dynamics; (iii) unilateral LIP inactivation has minimal effect on choice — compensation or auxiliary role.
- **§1.2 Rodent advances.** Poisson-click task; PPC encodes accumulated evidence *linearly*, FOF encodes it *categorically/step-like*; PPC inactivation has little effect; projection-specific targeting (e.g., A1→striatum neurons drive biases) (existing §2 Phase 2.4 for Brunton equations).
- **§1.3 Human signals.** BOLD in parietal/insula tracks conflict, not accumulation; EEG/MEG CPP = supramodal build-to-threshold DV; parietal alpha-phase rhythmic gain control as evidence-weighting bottleneck; *selective integration* model for value-based choices (existing §2 Phase 2.1–2.5 carry the math).
- **§2.1 Speed-accuracy.** Under speed pressure, LIP/FEF/PMd show *higher baselines / faster ramps* rather than lower terminal bound ⇒ urgency signal (stimulus-independent, time-growing); effectively a collapsing bound (existing §2 Phase 2.3).
- **§2.2 Prior probability & value.** Modeled as starting-point bias and stimulus-independent drift offset; LIP baseline shifts with target prior; human fMRI/EEG replicates under predictive-coding framework (existing §2 Phase 2.2).
- **§2.3 Decision confidence.** Opt-out and waiting-time paradigms; predicted "X-pattern" of confidence vs signal strength (rising on correct, falling on error trials); rodent OFC and macaque pulvinar show this; confidence may reduce to $(x, t)$-readout on the same DV — no separate metacognitive circuit required (existing §2 Phase 2.5 Kiani–Shadlen map).
- **Summary & Perspective.** DDM must be elaborated (urgency, multi-site coordination, cell types); priority: cross-species cell-type-level microcircuit bridging.

### Appendix: Section-by-Section Backbone

**Introduction.** 25-year canonical primate DDM; last 5 years diversified (rodent optogenetics, human EEG/MEG, economic paradigms); review synthesizes convergences and open problems.

**Canonical Perspectives.** DDM: sequential sampling of noisy evidence integrated linearly to a fixed bound. RDK (Fig. 1A): LIP and FEF firing rates ramp, rate scales with evidence quality, terminal firing is stereotyped pre-response. Microstimulation: MT biases drift; LIP biases accumulator. Establishes parietal/frontal areas as linear integrators.

**§1 — Insights from New Approaches.** (Structural header.)

**§1.1 New Challenges in Monkeys.** (i) Mixed selectivity: LIP neurons multiplex sensory/motor/decision ⇒ ramp may be an aggregate statistic, not a per-neuron property. (ii) Ramping vs stepping (Fig. 1B): HMM-based fits debate whether DV grows smoothly or jumps stochastically. (iii) Unilateral pharmacological LIP inactivation (Fig. 1C) leaves perceptual choice near-intact ⇒ causal obligate role of LIP under question; compensatory or auxiliary role plausible.

**§1.2 Rodent Approaches.** High-throughput training + optogenetics + Ca²⁺ imaging + cell-type viral tagging. Poisson-click task (Fig. 2A) gives closed-form trial-by-trial accumulator state (Brunton/Brody 2013). Neural: PPC encodes accumulated evidence linearly; FOF encodes it step-like/categorical (Fig. 2B). Rodent-PPC inactivation (Fig. 2C) barely impairs accumulation. Projection-specific analysis (Fig. 2D) — e.g., A1→striatum subpopulation explains specific biases.

**§1.3 Human Paradigms and Signals.** fMRI BOLD in parietal/insula tracks decision *conflict/uncertainty*, not accumulation. EEG/MEG CPP (Fig. 3A) = supramodal, effector-independent build-to-threshold DV mirroring LIP. Rhythmic parietal alpha phase controls evidence-sample gain (Fig. 3B): discrete-sample paradigms reveal a central bottleneck for evidence weighting. Selective integration model (Fig. 3C): multi-attribute economic choice weights better option disproportionately; suboptimal on paper but optimal once late-integration noise is modeled.

**§2 — Opportunities to Bridge across Model Systems.** (Structural header.)

**§2.1 Speed-Accuracy Trade-off.** Classical DDM: adjust bound $a$. Data: under speed pressure, monkey LIP/FEF/PMd baseline rates start higher or ramp faster (Fig. 4) — *not* a lower terminal threshold. Proposal: stimulus-independent *urgency* signal $u(t)$ multiplicatively scales activity toward a fixed bound, equivalent to an effective collapsing bound. Fixed-vs-dynamic-bound debate unresolved across species.

**§2.2 Probability and Value Modulation.** Priors / value modeled as (i) starting-point offset $z$ and/or (ii) stimulus-independent drift bias. Monkey LIP baseline rises with high-probability targets. Human fMRI/EEG shows bias signals across sensory/parietal/prefrontal cortex, interpreted in a predictive-coding framework (reciprocal prior-evidence integration).

**§2.3 Decision Confidence.** Humans: explicit report. Animals: opt-out (Fig. 5A) or waiting-time (Fig. 5B). Optimal pattern: confidence rises with signal strength on correct trials, *falls* with signal strength on error trials (the "X pattern"). Rodent OFC (Fig. 5C) and macaque pulvinar show the X pattern. Alternative view: confidence is not a separate metacognitive code — $(x, t)$ alone (accumulator state + elapsed time as difficulty proxy) suffices, encoded in LIP (Fig. 5D; Kiani–Shadlen).

**Summary & Perspective.** DDM remains useful but must absorb urgency, selective integration, confidence readout; future priority: simultaneous multi-region recordings + cell-type dissection + cross-species harmonization.

---

## 3. O'Connell & Kelly (2021) — Neurophysiology of Human Perceptual Decision-Making

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Invasive single-neuron recordings have given a detailed picture of how monkeys and rodents accumulate sensory evidence, but humans have traditionally been studied with tools (fMRI, scalp EEG) that either lack temporal resolution (fMRI) or were hard to map onto algorithmic decision variables (EEG). O'Connell & Kelly review how recent noninvasive methodology has closed this gap and allows tracking of the fine temporal dynamics of human decision formation.

**Key Findings.**
- **Centroparietal positivity (CPP).** A slow, build-up EEG signal that (i) rises at a rate proportional to evidence strength, (ii) peaks at response, and (iii) is *supramodal* — invariant to sensory modality and motor effector. It is interpreted as a scalp-level read-out of the abstract decision variable $x(t)$.
- **Effector-selective motor preparation.** Desynchronization of **mu/beta** (~8–30 Hz) over contralateral motor cortex tracks preparation of a specific action; it reaches a stereotyped threshold just before movement — a race-to-bound signal separate from the CPP.
- **Urgency dissociation.** The CPP reflects *pure evidence*, while mu/beta carries an evidence-independent urgency component — so under time pressure the motor signal reaches threshold with less CPP amplitude.
- **Neurally-informed modeling.** Using single-trial EEG features as regressors / priors on DDM parameters breaks degeneracies in purely behavioural fits; e.g., age-related slowing is attributable to *reduced drift variability* $\eta$ rather than a higher bound $a$.

**Main Methodology.** Time-stretched psychophysical paradigms (gradual-onset contrast, continuous random-dot stimuli), multichannel EEG/MEG with spatial filtering (LDA, CSP) to extract single-trial decision variables, and joint-modeling frameworks where EEG-derived regressors constrain hierarchical Bayesian DDM fits.

**Initial Takeaway.** Human decision-making can now be decomposed non-invasively into specific, interpretable components — evidence accumulation (CPP), effector preparation (mu/beta), and urgency — and mapped onto DDM parameters. This reframes individual differences (aging, clinical populations) as differences in *specific* algorithmic variables, not in a monolithic "threshold."

### Phase 2: Graduate-Level Deep Dive

> The review discusses these ideas at a conceptual level; equations below are the standard formulations from the literature the authors rely on (Philiastides & Sajda; Kelly & O'Connell; Turner et al.; Ratcliff & McKoon).

#### 3.1 CPP and Mu/Beta as ERP Read-outs of DDM Variables

Baseline DDM: $dx = v\, dt + \sigma\, dW_t$ with bound $a$.

**CPP ↔ decision variable.** Let $A_{\text{CPP}}(t)$ be the trial-averaged (or single-trial) CPP amplitude. The linking hypothesis is

$$
A_{\text{CPP}}(t) \;\propto\; x(t) \;=\; \int_0^t v\, d\tau + \int_0^t \sigma\, dW_\tau .
$$

- **Slope ↔ drift.** The CPP buildup rate is an estimator of drift:

  $$
  \widehat v_i \;=\; \frac{dA_{\text{CPP},i}}{dt}\Big|_{\text{deliberation}} .
  $$

- **Peak ↔ bound.** At the response time $T_R$, $A_{\text{CPP}}(T_R) \propto a$. Under speed pressure, $A_{\text{CPP}}(T_R)$ drops — consistent with a lower effective bound at commitment.

**Mu/beta as race-to-threshold motor variable.** Let $P_{\mu\beta}(t)$ be contralateral-motor-cortex spectral power in the mu/beta band and $P_0$ a prestimulus baseline. The effector-selective preparation signal is

$$
M(t) \;=\; P_0 \;-\; P_{\mu\beta}(t), \qquad \text{response when } M(t) \ge \theta_{\text{motor}} .
$$

Crucially, $\theta_{\text{motor}}$ is approximately fixed across conditions — the signal has a *stable terminus* — while its *starting level* and *growth rate* are what vary.

#### 3.2 Urgency as a Dissociation Between CPP and Mu/Beta

Let $U(t)$ be an evidence-independent urgency signal (empirically, a roughly linear component in mu/beta). The motor preparation combines evidence and urgency:

$$
M(t) \;=\; x(t) \;+\; U(t).
$$

With a fixed motor threshold $\theta_{\text{motor}}$, the *effective* evidence bound is

$$
x(t) \;\ge\; \theta_{\text{motor}} - U(t) \;\equiv\; a(t).
$$

Modelling $U(t) = k\, t$ gives a linearly collapsing evidence bound

$$
\boxed{\; a(t) \;=\; a_0 \;-\; k\, t. \;}
$$

This predicts — and the review confirms empirically — that the terminal CPP amplitude is smaller at long RTs or under speed stress, because the motor threshold is crossed by urgency *before* evidence has fully accumulated.

#### 3.3 Neurally-Informed DDM (Joint Modeling)

Purely behavioural DDM fits suffer from trade-offs between parameters (e.g. $v$ vs. $a$ vs. $\eta$). Neurally-informed modeling constrains a parameter $\theta_i$ on trial $i$ by a single-trial neural regressor $N_i$:

$$
\theta_i \;=\; \beta_0 \;+\; \beta_1\, N_i \;+\; \epsilon_i .
$$

Examples emphasized in the review:
- $N_i$ = single-trial CPP slope → constrains trial-wise drift $v_i$.
- $N_i$ = prestimulus contralateral mu/beta power → constrains starting point $z_i$.
- $N_i$ = stimulus-evoked early visual activity (e.g., SSVEP) → constrains pre-accumulation encoding delay.

Imposing these linkages breaks identifiability and yields the review's key substantive finding: age-related RT slowing is mediated by a decrease in drift-rate variability $\eta$, not by elevated bound $a$.

#### 3.4 Hierarchical Bayesian DDM

For subject $j$ and trial $i$, the choice–RT pair $(c_{ij}, y_{ij})$ follows the **Wiener first-passage-time (WFPT)** distribution

$$
(y_{ij}, c_{ij}) \;\sim\; \mathrm{WFPT}(a_j,\, v_{ij},\, z_j,\, t_{0,j}) ,
$$

whose density for the upper bound is the Navarro–Fuss series

$$
f_{+}(t \mid a, v, z) \;=\; \frac{\pi}{a^{2}}\, e^{\,v\, a z - \tfrac{v^{2} t}{2}} \sum_{k=1}^{\infty} k\, \sin\!\left(\frac{k \pi z}{a}\right) \exp\!\left(-\frac{k^{2} \pi^{2} t}{2 a^{2}}\right) .
$$

Subject-level parameters are drawn from group hyper-distributions:

$$
v_j \sim \mathcal N(\mu_v, \sigma_v^{2}), \quad a_j \sim \mathcal N_+(\mu_a, \sigma_a^{2}), \quad t_{0,j} \sim \mathcal N_+(\mu_{t0}, \sigma_{t0}^{2}),
$$

with weakly informative hyper-priors $\mu_v \sim \mathcal N(m_0, s_0^2)$, $\sigma_v \sim \mathrm{Gamma}(\alpha,\beta)$, etc. Neural-linkage coefficients $(\beta_0, \beta_1)$ are themselves estimated hierarchically, so that across-subject pooling *and* within-subject neural shrinkage operate jointly. Posterior inference (typically MCMC via HDDM / JAGS / Stan) yields joint credible intervals for behaviour–brain mappings.

#### 3.5 Extracting Single-Trial Decision Variables from EEG

Multichannel EEG $X(t) \in \mathbb{R}^{C\times T}$ is projected onto a spatial filter $w$ to yield a 1-D neural DV

$$
y(t) \;=\; w^{\top} X(t) .
$$

Fisher's LDA chooses $w$ to maximize the Rayleigh quotient of between-class over within-class scatter

$$
w_{\text{opt}} \;=\; \arg\max_{w}\; \frac{w^{\top} S_B w}{w^{\top} S_W w} ,
$$

with solution

$$
w_{\text{opt}} \;=\; S_W^{-1}\, (\mu_1 - \mu_2) ,
$$

where $\mu_{1,2}$ are class-conditional mean spatial patterns and $S_W = \tfrac12 (\Sigma_1 + \Sigma_2)$. The projected $y(t)$ serves as the single-trial regressor feeding §3.3.

An analogous Common Spatial Pattern (CSP) objective maximizes the ratio of band-power variances,

$$
w_{\text{CSP}} \;=\; \arg\max_{w}\; \frac{w^{\top} \Sigma_1 w}{w^{\top} \Sigma_2 w} ,
$$

used to isolate mu/beta lateralization.

**Synthesis.** Taken together, §§3.1–3.5 formalize the review's programme: extract an interpretable 1-D neural DV per trial via spatial filtering (§3.5); identify its role (CPP = $x(t)$, mu/beta = $M(t)$) and its relationship to algorithmic pieces of the DDM (§§3.1–3.2); then use hierarchical Bayesian joint modeling (§§3.3–3.4) to obtain *identifiable* cognitive parameters that can be linked to individual differences, development, aging, and pathology.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Neural signatures of decision formation.** Invasive primate/rodent work established integration-to-bound in LIP / FEF / PMd; human noninvasive tools now catch up via BOLD, lateralized beta / readiness potentials, rhythmic gain, and ML decoders (existing §3 Phase 2.5).
- **Multiple processing levels.** Two dissociable human DV classes: (i) *effector-selective* mu/beta desynchronization over motor cortex carrying urgency; (ii) *supramodal* CPP (≈ P3b) tracking pure evidence independent of modality and effector (existing §3 Phase 2.1). fMRI gives an amodal network (dlPFC, IPS, insula) but needs EEG/MEG time resolution to be interpretable.
- **Neurally informed modeling.** Constrain DDM parameters with single-trial CPP and mu/beta regressors to break parameter trade-offs (existing §3 Phase 2.3). Two worked examples:
  - *Speed pressure* ⇒ mu/beta pre-stimulus urgency + steeper CPP buildup ⇒ *drift rate actually rises* (opposite of behaviour-only DDM inference).
  - *Aging* ⇒ older > younger performance; behavioural-only DDM says elevated bound; CPP/mu-beta say bound unchanged; model with fixed bound reveals older adults benefit from *reduced drift-rate variability* $\eta$, validated by reduced trial-to-trial CPP-slope variance.
- **Ancillary processes.** pMFC theta = conflict ⇒ transient bound elevation via STN; striatal connectivity ⇒ bound lowering under speed pressure (Fig. 4). Pupil-linked arousal ⇒ global gain / urgency that suppresses prior bias. Early visual N2 / N2pc target selection ⇒ predicts CPP onset and buildup; mapped onto DDM non-decision time.
- **Concluding comments.** Human noninvasive signals now support cross-species neurally informed modeling. Next: biophysical origin of extracranial signals; translational mapping to psychiatric/neurological deficits.

### Appendix: Section-by-Section Backbone

**Neural Signatures of Decision Formation in the Human Brain.** Recaps the canonical primate story (evidence accumulators in LIP / FEF / PMd) and surveys the human toolbox now available (Fig. 1): BOLD timing; lateralized beta/LRP; rhythmic gain control; ML classifiers. Frames the rest of the review around three uses: parsing processing levels, neurally informing models, and studying ancillary systems.

**Multiple Processing Levels for Perceptual Decision-Making.** Requires parsing signals into functional roles (evidence, DV, motor preparation, decisional gain). Humans now offer two isolable DV classes:
1. **Effector-selective mu/beta desynchronization** — builds toward an action-triggering threshold; carries evidence-independent urgency that shifts baseline under time pressure.
2. **Centroparietal positivity (CPP)** — supramodal, motor-independent evidence accumulator; buildup rate scales with evidence strength; baseline stays fixed; under speed stress, terminates at lower amplitude (reflecting lowered motor threshold). Sidebar links CPP with the classic P3b as the same algorithmic signal.
fMRI amodal decision network (dlPFC, IPS, insula) useful for localization but too slow alone ⇒ must be paired with EEG/MEG.

**Neurally Informed Modeling.** Traditional: fit DDM to behaviour, then regress neural data. Problem: model-misspecification and parameter trade-offs. Better: use single-trial neural regressors to *constrain* model parameters directly. Two flagship applications:
- **Speed-accuracy trade-offs (Fig. 2).** Pre-stimulus anticipatory mu/beta buildup = urgency starting point. With this neural constraint, speed pressure is fit as *increased drift rate*, contradicting behaviour-only DDM which had inferred *decreased* drift. Confirmed by steeper single-trial CPP slopes under speed pressure.
- **Aging.** Older adults outperform younger in continuous monitoring. Behaviour-only DDM: elevated bound, elevated drift. Neural data: CPP / mu-beta thresholds unchanged across groups. Forcing bound fixed (neural constraint) reveals: older adults have *reduced drift-rate variability* $\eta$ — independently confirmed by reduced CPP-slope variance. Shows how neural constraints resolve otherwise degenerate model fits, especially valuable for low-trial-count clinical/aging populations.

**Ancillary Processes.** Core decision machinery interacts with:
- **Conflict monitoring (pMFC theta)** — signals post-response conflict/uncertainty; predicts subsequent RT slowing; modeled as transient bound elevation via cortical-STN hyperdirect pathway.
- **Striatal connectivity (Fig. 4)** — stronger white-matter link to striatum predicts greater bound *lowering* under speed pressure.
- **Pupil-linked arousal (LC-NE system)** — pupil dilation carries time-dependent urgency; correlates with pMFC conflict; globally suppresses prior biases on a trial-by-trial basis.
- **Early target selection (N2 / N2pc)** — occipitotemporal detection of goal-relevant features; its latency and amplitude predict CPP onset and buildup rate and map onto DDM *non-decision time* $t_{0}$. Acts as a gate that initiates evidence integration.

**Concluding Comments.** Human noninvasive signals now support a unified cross-species computational framework. Priorities: biophysical origin of scalp signals; translation to psychiatric / neurological deficits (esp. aging, Parkinson's, schizophrenia, ADHD).

---

## 4. Cisek (2021) — Evolution of Behavioural Control from Chordates to Primates

*Philosophical Transactions of the Royal Society B, 2021.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Classical cognitive neuroscience treats the brain as a serial information processor: *perceive → decide → plan → act*. Cisek argues this schema is a historical artefact of psychological introspection, not a design principle of biological brains. Brains did not evolve for abstract cognition; they evolved to keep a body alive via closed-loop interaction with the environment. The paper asks: *what is the correct functional architecture of the brain once we take its evolutionary trajectory seriously?*

**Key Findings.**
- **Affordance competition.** Perception and action are not sequential stages. Multiple candidate actions ("\affordances"\) are specified in parallel in sensorimotor cortex and *compete* until one wins. What we call a "\perceptual decision"\ is implemented in the same circuits that execute the action.
- **Evolutionary layering.** The modern brain is a stack of parallel control loops added by successive ecological transitions:
  - *Early chordates* — midbrain/tectum: spatial approach/avoidance.
  - *Amphibians (land)* — expanded pallium: long-range navigation (MPall, proto-hippocampus) and key-stimulus learning (VLPall).
  - *Mammals (nocturnality)* — dorsal pallium → neocortex with parallel action maps arbitrated by basal-ganglia loops.
  - *Primates (arboreality)* — fronto-parietal expansion, fovea + FEF/LIP gaze executive, distinction between *pragmatic* and *epistemic* action.
- **Urgency-gating over pure integration.** In the real world perfect integration is impossible (the environment changes); the brain trades integration for a time-dependent urgency signal that multiplicatively scales sensorimotor activity.

**Main Methodology.** *Phylogenetic refinement*: comparative neuroanatomy + fossil/ethology evidence used to reconstruct, transition by transition, which circuit solved which adaptive problem.

**Initial Takeaway.** Perceptual decision-making should not be modeled as an abstract DV in a Platonic accumulator. It should be modeled as biased competition among embodied action plans, with DDM/SPRT-like dynamics as a *special case* that emerges when the task happens to require a single binary choice.

### Phase 2: Graduate-Level Deep Dive

> Cisek (2021) is a phylogenetic synthesis; it references without reprinting the formal models from Cisek 2006/2007, Thura & Cisek 2014, and Pezzulo & Cisek 2016. Equations below are from that canonical companion literature.

#### 4.1 Affordance Competition via Dynamic Neural Fields

Let $x_i(t)$ be the activity of a neural population tuned to movement parameter $i$ (e.g. reach direction). In the Amari–Wilson-Cowan neural-field formulation,

$$
\tau\, \frac{dx_i(t)}{dt} \;=\; -x_i(t) \;+\; \sum_{j} w_{ij}\, f\!\left(x_j(t)\right) \;+\; I_i(t) \;+\; h ,
$$

with $\tau$ the membrane time constant, $f(\cdot)$ a sigmoidal gain, $h$ tonic drive, $I_i(t)$ the bottom-up affordance drive for action $i$, and $w_{ij}$ the lateral coupling. A "Mexican-hat" kernel over the tuning distance $d_{ij}$ implements local excitation and global inhibition:

$$
w_{ij} \;=\; A\, \exp\!\left(-\frac{d_{ij}^{2}}{a^{2}}\right) \;-\; B\, \exp\!\left(-\frac{d_{ij}^{2}}{b^{2}}\right), \qquad A>B, \; a<b .
$$

**Winner-take-all limit.** For a two-option case with activities $x_1, x_2$, symmetric self-excitation $\alpha$, and cross-inhibition $\beta$:

$$
\tau\, \dot x_1 = -x_1 + \alpha f(x_1) - \beta f(x_2) + I_1 + h,
$$
$$
\tau\, \dot x_2 = -x_2 + \alpha f(x_2) - \beta f(x_1) + I_2 + h.
$$

Setting $\dot x_1 = \dot x_2 = 0$ gives a pitchfork bifurcation at $\beta - \alpha = 1/f'(x^{*})$: below threshold, both populations coexist (deliberation); above threshold the symmetric fixed point is unstable and the system snaps to a WTA attractor where one population saturates and the other is suppressed. Top-down biases shift $I_i$ and break the symmetry. This is the circuit-level instantiation of a decision.

#### 4.2 Urgency-Gating Model (Thura & Cisek) vs. DDM

In Ratcliff/Shadlen DDM the decision variable is the *perfect integral* $x(t) = \int_0^t E(\tau)\, d\tau + \text{noise}$. Cisek argues biology cannot afford perfect integration in a non-stationary world. Replace perfect integration with a *leaky* estimate of momentary evidence,

$$
y_i(t) \;=\; \int_{0}^{t} E_i(\tau)\, \exp\!\left(-\frac{t-\tau}{\tau_{\text{leak}}}\right) d\tau ,
$$

and multiply by a rising, stimulus-independent urgency signal $u(t)$ (putatively from the basal ganglia):

$$
\boxed{\; x_i(t) \;=\; y_i(t)\, \cdot\, u(t). \;}
$$

Commitment occurs when $x_i(t)$ crosses the fixed WTA threshold of §4.1. Because $u(t)$ grows, the amount of instantaneous evidence $y_i(t)$ needed to cross shrinks — a phenomenon that *looks* like a collapsing bound when projected back into the DDM parameterization (cf. §3.2 O'Connell–Kelly). Critically, the UGM predicts that neural firing rates continue to rise in time even when evidence is constant — a signature observed in PMd and not in pure integrator accounts.

**Mapping to DDM.** Under the UGM the effective drift is $v_{\text{eff}}(t) = u(t)\, E(t)/\tau_{\text{leak}}$ and the effective bound is the fixed WTA threshold $\theta$. The classic DDM is recovered in the limit $\tau_{\text{leak}} \to \infty$, $u(t) \equiv 1$.

#### 4.3 Evolutionary Sequence of Control Loops

Cisek organizes behavioural control as nested negative-feedback loops added over phylogeny:

| Stage | Ecological pressure | Key neural innovation | Control loop extended |
|---|---|---|---|
| Chordates / early vertebrates | Swim toward prey, away from threat | Optic tectum, dopaminergic midbrain | Spatial approach/avoidance in body frame |
| Amphibians | Land, longer visual horizon | Medial pallium (bearing/sketch maps); ventrolateral pallium (key stimuli) | Allocentric navigation; stimulus-specific learning |
| Mammals | Nocturnality, tactile/auditory load | Dorsal pallium → six-layered neocortex; cortico–BG–thalamo–cortical loops | Parallel action maps arbitrated by basal ganglia |
| Primates | Arboreality, foveal vision | Fronto-parietal reach/grasp circuits; FEF/LIP gaze executive | Epistemic (sensor-directed) action before pragmatic action |

Mathematically each new layer adds an outer feedback loop around the earlier ones; the modern primate brain is therefore a *heterarchy* rather than a clean hierarchy.

#### 4.4 Pragmatic vs. Epistemic Action

Let the world state be $s$ and the agent’s belief $b(s)$. Two classes of action:

- **Pragmatic action $a_p$** changes $s$ (and therefore expected reward).
- **Epistemic action $a_e$** changes $b(s)$ without (appreciably) changing $s$ — e.g. a saccade.

In primates the cost of a wrong pragmatic action (falling out of a tree) is enormous, making it profitable to first deploy epistemic actions. Formally, choose $a \in \{a_p, a_e\}$ to maximize

$$
Q(a) \;=\; \mathbb{E}_{b}\!\left[\,R(s,a)\,\right] \;-\; C(a) \;+\; \gamma\, \mathbb{E}\!\left[\max_{a'} Q(a', b')\right],
$$

where $b' = \text{update}(b, a, o)$ uses the observation $o$ induced by $a$. This is essentially the POMDP value of information, but Cisek emphasizes that biology solves it with *dedicated circuitry* — the FEF/LIP gaze system — rather than as an abstract computation. This is the evolutionary wedge that eventually lets the brain "decide about decisions" and prefigures the neurally-informed CPP/metacognition stories in §§3 and 5.

**Bottom line.** Perceptual decision-making in the Gold–Shadlen/Hanks–Summerfield tradition is a *limit case* of affordance competition plus urgency, running on a primate-specific gaze-executive overlay. This reframing is essential for connecting §§5–7 (confidence, dopamine/RL, active inference) back to grounded neurobiology.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **§1 Behavioural control systems.** Behaviour as negative-feedback control; *impetus* for action; *key stimuli* vs *affordances*; exploitation / exploration / escape as closed loops with the environment (Fig. 1). Perception / action / cognition are a single control problem, not separable modules.
- **§2 Hypothetical phylogeny.** Sets up an evolutionary roadmap (Fig. 2) of niche transitions structuring the rest of the paper.
- **§3 Vertebrate Bauplan.** Diffuse net → dorsal neural tube (Fig. 3a–b). Early chordates: dopamine gates exploitation/exploration; tectum precursor drives escape. Early vertebrates: paired lateral eyes → tectal spatial maps; *between-system* (approach vs escape thresholds) and *within-system* (target location) decisions; VLPall learns key stimuli, MPall supports long-range navigation; cerebellum as adaptive filter subtracting efference copy.
- **§4 Transition to land.** Devonian — expanded visual range, new homeostatic demands. MPall ⇒ allocentric bearing/sketch maps (Fig. 4a–b). Dorsal pallium (DPall) and lateral pallium (LPall) emerge at the MPall/VLPall border; LPall integrates internal state with external cues along appetitive-aversive axis. Key claim: DPall evolves to detect *affordances* (e.g., hideability), not to build retinotopic images (Fig. 5); turtle DPall data corroborate.
- **§5 Retreat into nocturnality.** Synapsid mammals become small/nocturnal; somatosensation (vibrissae) + audition + olfaction replace vision. DPall expands into six-layered neocortex (Fig. 6a–b). Parallel cortical action maps: searching (cingulate), handling (S1/M1), ingesting (insular). Basal-ganglia loops arbitrate *between* action maps; within each map, competitive WTA resolves *within-system* movement metrics (connects to existing §4 Phase 2.1 Amari/Wilson-Cowan equations).
- **§6 Primate innovations.** Arboreal → diurnal return. Fronto-parietal expansion (Fig. 7a–b): MIP/PMd reach, AIP/PMv grasp, LIP/FEF gaze. Fovea + OFC make gaze an *executive* epistemic controller — evaluate offer value *before* pragmatic commitment. PFC/cerebellum extend prediction over sub-goals (connects to existing §4 Phase 2.4 pragmatic/epistemic).
- **§7 Conclusions.** Brain evolution = incremental extension of feedback control; "cognitive" functions should be decomposed along phylogenetic niche transitions, not classical psychology labels.

### Appendix: Section-by-Section Backbone

**§1 Behavioural control systems.** Closed-loop negative feedback is the basic unit. *Impetus* = condition motivating action that reduces that condition (hunger + food ⇒ ingestion). *Key stimuli* (categorical cues of external state) vs *affordances* (metrics of available actions). Exploitation, exploration, escape (Fig. 1) are each feedback loops over the environment. Standard psychology splits (perception/action, cognition/behaviour) dissolve inside this framing.

**§2 Hypothetical phylogeny (Fig. 2).** Sequence of niche transitions up the human lineage used to structure the review.

**§3 Evolution of the vertebrate Bauplan (Figs. 3a-b).** From diffuse net to dorsal neural tube.
- Early chordates: DA gates exploitation vs exploration; dorsal tectum precursor triggers escape.
- Early jawed vertebrates: paired lateral eyes → tectal spatial maps. Two decision types: *between-system* (threshold/go–no-go: approach vs escape) and *within-system* (WTA for target location, signal averaging for escape direction).
- Telencephalon elaborated: VLPall learns key stimuli for local exploitation; MPall supports long-range navigation.
- Thalamic parcellation mirrors dorsal/ventral split.
- Cerebellum = adaptive filter subtracting motor efference copy and predicting sensory consequences.

**§4 The transition to land.** Devonian terrestrial transition — massive visual range increase, new homeostatic challenges.
- MPall (Fig. 4a-b) develops allocentric bearing + sketch maps for long-range navigation.
- DPall and LPall emerge at the MPall/VLPall border.
- Sauropsid vs mammal divergence in ventral pallium handling of key stimuli.
- LPall integrates internal state (appetite/aversion) with external cues.
- **Key claim:** DPall is an *affordance detector*, not a retinotopic map; e.g., specific optic-flow patterns (motion-contrast edges above expansion points) reliably indicate *hideability* (Fig. 5). Turtle DPall physiology matches this: non-retinotopic but motion-structure-sensitive.

**§5 Retreat into nocturnality.** Early mammals small, nocturnal, endothermic ⇒ vision deprioritized, vibrissae + audition + olfaction expanded.
- DPall expands radially into six-layered neocortex.
- New direct cortical–spinal projections and subpallial–cortical thalamic returns move control from visuomotor midbrain to somatomotor forebrain (Figs. 6a–b).
- Parallel cortical **action maps** for species-typical behaviours: search (medial/cingulate), handling (S1/M1), ingest (lateral/insular).
- **Hierarchical control:** basal-ganglia loops arbitrate *between* action maps; competitive dynamics within a map resolve *within-system* movement metrics. (This is the phylogenetic substrate of the affordance-competition / WTA formalism in the existing §4 Phase 2.1.)

**§6 Innovations of primates.** Arboreal origin, diurnal return; demands precise visual-motor coordination.
- Fronto-parietal expansion (Figs. 7a-b): specialized reach (MIP/PMd), grasp (AIP/PMv), gaze (LIP/FEF) circuits in idiosyncratic reference frames.
- Temporal-lobe expansion categorizes objects motivating affordances.
- Fovea + granular OFC make the gaze system an **executive**: visually forage, evaluate offer value of distant options *before* committing to costly pragmatic action (the pragmatic vs epistemic distinction elaborated in existing §4 Phase 2.4).
- PFC + cerebellum extend prediction into future sub-goals, enabling sequential and implicit/explicit planning.

**§7 Concluding remarks.** Brain evolution = continuous extension of feedback control into the environment. Primate brain = hierarchy of parallel, competing control systems shaped by specific niche transitions. Argument: decompose "cognition" along evolutionary history, not along classical psychology constructs. Human-specific adaptations require this non-human primate baseline before being built on top.

---

## 5. Fleming (2024) — Metacognition and Confidence: A Review and Synthesis

*Annual Review of Psychology, 2024.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Research on uncertainty has long been split. *Subpersonal* accounts (sensory/motor uncertainty, Bayesian coding, §§1–3) describe how individual circuits track noise. *Personal-level* metacognition (introspective confidence, feelings of knowing, explicit reports) describes subjective beliefs about one's own performance. Fleming's 2024 synthesis asks: **how is local sensory uncertainty transformed into conscious, reportable confidence — and which neural circuits implement that transformation?**

**Key Findings.**
- **Propositional confidence.** Metacognition is grounded in a *self-centered* belief $P(a = d \mid \hat s)$ — the probability that the agent's own action matches the true world state — distinct from raw sensory uncertainty.
- **A distributed confidence network.**
  - Early sensory cortex encodes *implicit* sensory uncertainty (likelihood shape).
  - **vmPFC / pgACC** computes propositional confidence in a decision-centered frame.
  - **pMFC (posterior medial frontal)** accumulates *post-decisional* evidence, producing the error-related negativity (ERN) and Pe.
  - **aPFC / rlPFC (frontopolar)** mediates the *global broadcast* of confidence into explicit reports; its disruption lowers meta-$d'$ without touching $d'$.
- **Model-based self-inference.** Confidence is shaped by an implicit *self-model* that incorporates response fluency, interoceptive arousal, and priors about one's ability — analogous to theory-of-mind for the self.
- **Hierarchical confidence.** Trial-level confidence is integrated over time into *global* self-performance beliefs, explaining systematic individual differences and metacognitive miscalibration in clinical and developmental populations.

**Main Methodology.** Cross-level synthesis: explicit confidence ratings, opt-out paradigms (monkeys, rats), SDT-based metrics ($d'$, meta-$d'$, M-ratio), Bayesian ideal-observer modeling, two-stage drift-diffusion confidence models, and human fMRI/EEG/TMS targeting vmPFC, pMFC, and frontopolar cortex.

**Initial Takeaway.** Confidence is not a single quantity. It is the output of a multi-stage computation: sensory likelihood → propositional confidence → post-decisional updating → global broadcast. Each stage has a distinct mathematical object and a distinct neural substrate. This reframes many "metacognitive deficits" (aging, schizophrenia, OCD) as disorders of *specific* stages, not of "insight" in general.

### Phase 2: Graduate-Level Deep Dive

> Fleming (2024) surveys these models; equations below follow the canonical references he cites (Maniscalco & Lau 2012; Kiani & Shadlen 2009; Pleskac & Busemeyer 2010; Rouault, Seow, Gillan, Fleming 2018).

#### 5.1 Bayesian Ideal Observer and Propositional Confidence

Given a noisy measurement $X$ of stimulus $s$, the posterior is

$$
p(s \mid X) \;\propto\; p(X \mid s)\, p(s).
$$

For a binary categorization at boundary $m$ (e.g., CW vs CCW):

$$
P(d_{\text{CW}} \mid \hat s) = \int_{-\infty}^{m} p(s\mid X)\, ds, \qquad
P(d_{\text{CCW}} \mid \hat s) = \int_{m}^{\infty} p(s\mid X)\, ds.
$$

After committing to action $a$, **propositional confidence** is

$$
\boxed{\; \mathrm{Conf}(a) \;=\; P(a = d \mid \hat s). \;}
$$

In a time-extended DDM this reduces to the Kiani–Shadlen log-odds-correct map (cf. §2.5):

$$
L(x,t) \;=\; \log \frac{P(\text{correct}\mid x, t)}{P(\text{error}\mid x, t)},
$$

so the same accumulator state simultaneously supports a choice and a confidence rating.

#### 5.2 Type-1 vs Type-2 SDT and meta-$d'$

**Type-1 SDT (task performance).** Two stimulus classes $S_1, S_2$; observer uses criterion $c$.

$$
d' \;=\; z(\mathrm{HR}) \;-\; z(\mathrm{FAR}),
$$

with $\mathrm{HR} = P(\text{say } S_2 \mid S_2)$, $\mathrm{FAR} = P(\text{say } S_2 \mid S_1)$, and $z = \Phi^{-1}$.

**Type-2 SDT (metacognitive performance).** Replace the stimulus labels with *correctness*:

$$
\mathrm{HR}_2 = P(\text{high conf} \mid \text{correct}), \quad \mathrm{FAR}_2 = P(\text{high conf} \mid \text{error}).
$$

A naive Type-2 $d'$ is confounded by $d'$ itself and by response bias.

**meta-$d'$ (Maniscalco & Lau 2012).** Define $\mathrm{meta}\text{-}d'$ as the Type-1 sensitivity an ideal observer would need *under the equal-variance SDT generative model* in order to produce the observed Type-2 ROC. Formally, with Type-1 choice criterion $c$ and a vector of confidence criteria $\{c^{\text{conf}}_k\}$ flanking $c$, predicted Type-2 cell counts are

$$
\hat n_{r,k} \;=\; N\, \Phi_{d', c, c^{\text{conf}}_k}(r, k),
$$

and meta-$d'$ is the $d^{*}$ that maximizes the likelihood of the *observed* counts when $c, \{c^{\text{conf}}_k\}$ are re-estimated under $d^{*}$:

$$
\mathrm{meta}\text{-}d' \;=\; \arg\max_{d^{*}} \; \max_{c,\{c^{\text{conf}}_k\}} \; \prod_{r,k} \hat n_{r,k}(d^{*}, c, \{c^{\text{conf}}_k\})^{\,n_{r,k}}.
$$

**M-ratio (metacognitive efficiency).**

$$
\boxed{\; M_{\text{ratio}} \;=\; \frac{\mathrm{meta}\text{-}d'}{d'} .\;}
$$

$M_{\text{ratio}} = 1$: observer uses all available sensory evidence for confidence. $M_{\text{ratio}} < 1$: metacognitive noise or suboptimal readout. $M_{\text{ratio}} > 1$ is diagnostic of violations of equal-variance SDT (e.g., hierarchical or second-order mechanisms).

**zROC slope.** Plotting Type-2 ROC in $z$-space,

$$
z(\mathrm{HR}_2) \;=\; s \cdot z(\mathrm{FAR}_2) \;+\; b,
$$

the slope $s$ equals the standard-deviation ratio $\sigma_{\text{error}}/\sigma_{\text{correct}}$ of the latent confidence-evidence distributions, diagnosing whether errors carry more variable evidence than correct trials.

#### 5.3 Second-Order Models: Two-Stage Dynamic Signal Detection

Pleskac & Busemeyer (2010) extend DDM with a post-decisional phase:

- **Stage 1 (decision).** $dx = v\, dt + \sigma\, dW_t$ until $x$ hits $\pm a$ at time $T_{\text{dec}}$.
- **Stage 2 (confidence).** Evidence continues to accumulate from the chosen bound for an additional window $\Delta$:

$$
x(T_{\text{conf}}) \;=\; \pm a \;+\; \int_{T_{\text{dec}}}^{T_{\text{dec}}+\Delta} v\, dt + \sigma\, dW_t .
$$

Confidence is a monotone function of $x(T_{\text{conf}})$; when post-decisional evidence reverses sign, the model predicts *changes of mind*. This two-stage architecture explains:

- Confidence–accuracy dissociations (high confidence on errors when Stage 2 reinforces the wrong bound).
- Continued build-up of the centroparietal Pe after an error — a Stage-2 signature (cf. §3.1).
- Post-decisional slowing and gradual confidence calibration.

Hierarchical gated extensions (e.g., Moran et al.) allow Stage 2 to *gate* a second-order accumulator driven only by the *congruence* between the sensory trace and the committed choice.

#### 5.4 Neural Mapping

| Stage | Quantity | Neural substrate |
|---|---|---|
| Sensory | Likelihood $p(X\mid s)$ | V1/MT — probabilistic population codes |
| Decision | $x(t)$, choice bound | LIP / PPC / CPP |
| Propositional confidence | $P(a = d \mid \hat s)$ | **vmPFC / pgACC** |
| Post-decisional update | Stage-2 accumulator; ERN / Pe | **pMFC** |
| Global broadcast | Explicit reportable confidence | **aPFC / rlPFC (frontopolar)** |
| Hierarchical self-model | $\mu_v$ (ability prior) | **Precuneus, vmPFC (long timescale)** |

TMS / lesion / fMRI evidence: frontopolar disruption selectively lowers meta-$d'$, leaving $d'$ intact — exactly what the Maniscalco–Lau decomposition predicts if these regions are metacognitive readouts rather than first-order sensors.

#### 5.5 Hierarchical Confidence and Calibration Metrics

**Hierarchical generative model.** Let $v_i$ be the drift rate on trial $i$ and $\mu_v$ the latent ability:

$$
v_i \sim \mathcal N(\mu_v, \sigma_v^{2}), \qquad \mu_v \sim p(\mu_v).
$$

Trial-wise propositional confidence $c_i = P(\text{correct} \mid x_i, t_i)$ updates the posterior over $\mu_v$:

$$
p(\mu_v \mid c_{1:t}) \;\propto\; p(\mu_v) \prod_{i=1}^{t} p(c_i \mid \mu_v).
$$

This captures why aggregated self-performance beliefs are *sluggish* relative to single-trial signals — a property tracked by vmPFC/precuneus on long timescales.

**Confidence bias.** With $c_t \in [0,1]$ and $o_t \in \{0,1\}$,

$$
\mathrm{Bias} \;=\; \frac{1}{N}\sum_{t=1}^{N} c_t \;-\; \frac{1}{N}\sum_{t=1}^{N} o_t .
$$

**Brier score** (strictly proper scoring rule for probabilistic forecasts):

$$
\mathrm{BS} \;=\; \frac{1}{N}\sum_{t=1}^{N} (c_t - o_t)^{2} .
$$

Murphy decomposition:

$$
\mathrm{BS} \;=\; \underbrace{\sum_{k} \tfrac{n_k}{N}(\bar c_k - \bar o_k)^{2}}_{\text{reliability}} \;-\; \underbrace{\sum_{k} \tfrac{n_k}{N}(\bar o_k - \bar o)^{2}}_{\text{resolution}} \;+\; \underbrace{\bar o (1 - \bar o)}_{\text{uncertainty}},
$$

where $k$ indexes confidence bins, $n_k$ the bin count, $\bar c_k, \bar o_k$ bin-averaged confidence and outcome, and $\bar o$ the base rate. *Reliability* is calibration error; *resolution* is discrimination ability (closely related to $M_{\text{ratio}}$); *uncertainty* is irreducible task difficulty.

**Synthesis.** §§5.1–5.5 formalize Fleming's argument: confidence is a *computation*, not a readout. $d'$, meta-$d'$/$M_{\text{ratio}}$, second-order DDM states, and hierarchical $\mu_v$ posteriors each index a different stage of that computation, with dissociable neural substrates. This equips the rest of the review (Gershman 2024, Smith–Friston–Whyte 2022) with the precise quantities that Bayesian RL / active inference must explain.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **§1 Introduction.** Metacognition = capacity to reflect on / evaluate / control mental functions; accuracy of self-evaluation ("metacognitive sensitivity") is critical for adaptive behaviour. Divide to bridge: subpersonal sensory-uncertainty coding (neuroscience) vs personal-level self-performance beliefs (psychology). Unifying object: **propositional confidence** $P(a = d \mid \hat s)$.
- **§2 Scope and definitions.** Metacognition = mechanisms forming beliefs about other mental operations for regulation and communication; distinct from executive function / fluid intelligence. Confidence = propositional (subjective surety), not objective world-probability. Excludes animal metacognition, development, interpersonal functions for scope.
- **§3 Paradigms and findings.** Explicit confidence ratings + opt-out paradigms. **Box — Measurement:** meta-$d' / d'$ decomposes sensitivity from bias (existing §5 Phase 2.2). Two research threads: (i) cues biasing confidence (fluency, accessibility); (ii) neural substrates — rlPFC (human/primate) and OFC (rodent) support metacognition independent of first-order task performance.
- **§4 Components of a metacognitive judgment.** Four-stage model:
  - **§4.1 Representing uncertainty** — world-centered distributional codes (probabilistic population codes, sampling, neuromodulator summary stats); multisensory integration shows optimal readout; fMRI decoding of V1 orientation uncertainty and LIP firing variance predict confidence.
  - **§4.2 Propositional confidence** — Bayesian reframing from world-centered to self-centered; **Box — Computing propositional confidence** (existing §5 Phase 2.1 equations); "folded X" pattern; neural substrates rodent OFC + human pgACC/vmPFC; post-decisional ERN / Pe from pMFC as Stage-2 evidence (existing §5 Phase 2.3).
  - **§4.3 Global broadcast and communication** — confidence must be broadcast for cross-modal comparison, learning, social report. Lateral frontopolar cortex engaged specifically for explicit metacognitive reports and private-to-public mapping.
  - **§4.4 Role of self-models** — inferential, theory-of-mind-like; fluency, brightness, interoceptive arousal falsely boost confidence; motor feedback updates confidence post-decisionally; correlates with mentalizing; impaired in ASD.
- **§5 Confidence formation and the psychology of metacognition.** Synthesis map: sensory cortex (world-centered) ⇒ vmPFC + pMFC (self-centered) ⇒ lateral frontopolar (global broadcast, public communication) ⇒ dmPFC / ToM (self-models as priors).
- **§6 Revisiting controversies.**
  - **§6.1 Biases.** Simple tasks ⇒ confidence is direct evidence readout. Complex tasks ⇒ post-decisional + self-model machinery ⇒ opportunities for sub-optimality. Positive Evidence Bias (PEB) is rational under high-dim evidence mapping, not a heuristic flaw.
  - **§6.2 Domain-generality.** Behavioural correlation across domains vs domain-specific lesion effects (precuneus ⇒ metamemory; frontopolar ⇒ metaperception). Reconciled: shared global-broadcast ⇒ positive manifold; local uncertainty propagation is domain-specific.
  - **Box — Individual and group differences:** meta-$d'$ is test-retest reliable; correlates with transdiagnostic symptoms, dogmatism, info-seeking.
- **§7 Where next?**
  - **§7.1 Common computational principles** across metaperception / metamemory.
  - **§7.2 Local vs global metacognition** — global confidence as hierarchical prior over local; vmPFC + precuneus integrate across long timescales (existing §5 Phase 2.5 hierarchical model).
  - **§7.3 Interventions.** Training / neurofeedback / stimulation promising but often modify private-public mapping rather than underlying confidence.
- **§8 Conclusions.** Metacognition research = study of propositional confidence in all forms; unifies previously separate literatures.

### Appendix: Section-by-Section Backbone

**§1 Introduction.** Metacognition across domains (metamemory, metaperception). Sensitivity matters for adaptive behaviour. Bridging move: subpersonal neuroscience of uncertainty ↔ personal-level confidence psychology. Unifying construct: propositional confidence.

**§2 Scope and definitions.** Explicit definitions. Metacognition ≠ executive function / fluid intelligence / first-order cognitive control. Confidence is strictly subjective surety. Scope excludes animal metacognition, development, interpersonal uses.

**§3 Paradigms and findings in metacognitive neuroscience.** Explicit confidence ratings and opt-out paradigms. Box (Measurement): meta-$d' / d'$ — performance-controlled metacognitive efficiency; separates sensitivity from bias. Two threads: psychological cues that bias confidence; neural substrates in rlPFC (human/primate) and OFC (rodent), preserving first-order task performance.

**§4 Components of a metacognitive judgment.** Structural header.
- **§4.1 Representing uncertainty.** World-centered codes: probabilistic population codes, sampling, neuromodulator summary stats. Optimal multisensory integration. Decoded V1 orientation uncertainty (fMRI ML) correlates negatively with reported confidence; LIP firing variance predicts opt-out (monkey).
- **§4.2 Propositional confidence.** **Box — Computing propositional confidence** provides the Bayes-rule computation: $p(s\mid X_i) \propto p(X_i\mid s)p(s)$; categorical $p(d_{CW}\mid\hat s) = \int_{-\infty}^{m} p(s\mid X_i) ds$; **Confidence** $= p(a = d \mid \hat s)$. Behavioural signature: "folded X" (confidence rising on correct, falling on error with signal strength). Neural: rodent OFC + human pgACC/vmPFC. Post-decisional accumulation: pMFC ERN + centroparietal Pe = self-centered evidence accumulation continuing after choice (ties Fleming's framework to the error-monitoring literature).
- **§4.3 Global broadcast and communication.** Propositional confidence must be broadcast for: cross-modal comparison, learning signal, social communication. Modality-independent confidence in PFC; lateral frontopolar cortex specifically recruited by explicit reports and by private-public mapping adjustments.
- **§4.4 Role of self-models.** Inferential ToM-like self-model incorporates fluency, accessibility, interoceptive arousal, motor feedback. Stimulus brightness / font size falsely raise confidence without raising accuracy ⇒ evidence for self-model. Computation: priors on cue-accuracy links or motor-feedback posterior correction. Mentalizing correlation; ASD impairment.

**§5 Confidence formation and the psychology of metacognition.** Synthesis: confidence emerges from interacting stages spanning distinct reference frames. Circuit map: world-centered sensory cortex → self-centered vmPFC + pMFC → frontopolar global broadcast → dmPFC ToM self-models.

**§6 Revisiting current controversies.** Structural header.
- **§6.1 Biases and suboptimalities.** Simple tasks: confidence = direct DV readout. Complex tasks: secondary machinery (post-decision + self-models) ⇒ opportunities for suboptimality. Positive Evidence Bias (PEB): confidence over-weights choice-congruent evidence. Recent NN modelling: PEB is rational when mapping high-dim evidence into scalar confidence, not a heuristic failure.
- **§6.2 Sources of domain-generality.** **Box — Individual and group differences:** meta-$d'$ shows test-retest reliability; correlates with transdiagnostic symptoms (including anxiety/compulsivity axes), dogmatism, info-seeking. Main text: behaviour suggests domain-general; lesions suggest domain-specific (precuneus → metamemory; frontopolar → metaperception). Resolution: shared global-broadcast gives behavioural positive manifold; localized uncertainty propagation gives domain-specific deficits.

**§7 Where next?** Structural header.
- **§7.1 Common computational principles.** Bridge metaperception (high-trial-count psychophysics + modelling) with metamemory (naturalistic stimuli + cue manipulations) to find shared metacognitive constraints.
- **§7.2 From local to global metacognition.** Local = per-trial confidence; global = self-efficacy, general-ability beliefs. Global acts as hierarchical prior over local. vmPFC + precuneus integrate local confidence across long timescales to form global self-beliefs used for strategy switching and task engagement.
- **§7.3 Opportunities for interventions.** Adaptive training, neurofeedback, brain stimulation. Underused because locus of action unknown. Recent analyses: behavioural training may only change the private-public scale-use mapping, leaving intrapersonal confidence formation untouched — improving report but not real-world cognitive control.

**§8 Conclusions.** Metacognition = study of propositional confidence. Unifies cognitive psychology + computational neuroscience. Confidence formation is a multi-stage pipeline (subpersonal uncertainty → propositional confidence → global broadcast → ToM-like self-model); intervention design should target specific stages.

---

## 6. Gershman et al. (2024) — Explaining Dopamine Through Prediction Errors and Beyond

*Samuel J. Gershman, John A. Assad, Sandeep R. Datta, Scott W. Linderman, Bernardo L. Sabatini, Naoshige Uchida, Linda Wilbrecht. Nature Neuroscience, 2024.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** The canonical account of dopamine (DA) — phasic bursts of midbrain DA neurons encode a scalar **reward prediction error** (RPE) $\delta = r + \gamma V(s') - V(s)$ — has been the backbone of reward learning for three decades. But recent data don't fit: DA signals *ramp* during goal approach, respond to *neutral sensory identity changes*, scale with **movement vigor**, and show strong **cell-type and projection-target heterogeneity**. Gershman et al. ask whether the RPE hypothesis must be discarded or whether a richer *generalized-prediction-error* framework can absorb these phenomena.

**Key Findings.**
- **Ramps as RPE under belief-state uncertainty.** Under noisy timekeeping, cumulative value estimates become biased; correcting via sensory feedback produces a monotonic ramp in $\delta_t$ — still an RPE, just over a *belief state* rather than a putative Markov state.
- **Generalized PE via Successor Representation.** DA tracks not only scalar reward but a vector of *feature predictions* $M(s,s')$; this explains responses to identity changes and sensory preconditioning.
- **Heterogeneity and labeled-line coding.** Distinct DA populations (VTA → NAc, SNc → DMS/DLS, tail-of-striatum) carry different functional variables: reward RPE, movement, novelty/threat, even sensory PEs.
- **Dual phasic/tonic architecture.** Phasic DA remains RPE-like (signed, fast); *tonic* DA tracks **average reward rate** $\rho$, which sets the **opportunity cost of time** and therefore controls vigor, response rates, and — crucially for perceptual decisions — the **decision bound**.
- **Links to perceptual decision-making.** Phasic DA at feedback is **confidence-weighted**: $\delta_{\text{out}} = r - P(\text{correct}\mid x, t) R$. Tonic DA adjusts the DDM bound $a$ via the opportunity-cost trade-off of Niv et al.

**Main Methodology.** A theoretical perspective that reinterprets recent mouse optogenetic, cell-type-specific calcium-imaging, and virtual-reality data under three extended RL formalisms: belief-state MDPs, distributional RL (Dabney et al. 2020), and average-reward RL with vigor (Niv). Focus on the mesostriatal / mesolimbic anatomy as the substrate.

**Initial Takeaway.** Dopamine is not one signal — it is a *family* of prediction errors over different predictive objects (scalar reward, feature vectors, return distributions, average reward). For perceptual decision-making this matters directly: DA modulates **drift rate** (via confidence-weighted learning signals), **decision bound** (via tonic-DA opportunity-cost signalling), and **action vigor** — linking §§2–3 DDM variables to the neuromodulatory system that tunes them.

### Phase 2: Graduate-Level Deep Dive

> The paper is a perspective; equations 6.1, 6.5, 6.7 are directly from the text. Derivations for distributional RL (§6.3), belief-state Bayes filter (§6.4), multi-timescale DA (§6.6), Niv vigor (§6.7), and policy gradient (§6.8) are reconstructed from the references the authors cite (Dabney et al. 2020; Starkweather et al. 2017; Niv et al. 2007; Sutton & Barto; Lak et al. 2017).

#### 6.1 Classical TD-Learning RPE

With policy $\pi$, expected return $V_t = \mathbb E_\pi[r_t + \gamma r_{t+1} + \dots]$ obeys the Bellman equation

$$
V_t \;=\; \mathbb E_\pi\!\left[r_t + \gamma\, V_{t+1}\right].
$$

The TD error is

$$
\boxed{\;\delta_t \;=\; r_t + \gamma\, \hat V_{t+1} - \hat V_t .\;}
$$

Value updates (semi-gradient): $\Delta w = \alpha\, \delta_t\, \nabla_w \hat V_t$. Classical RPE = scalar, signed, instantaneous.

#### 6.2 Why Pure TD-RPE Fails

- **Ramps.** Pure RPE should spike and settle; DA often ramps continuously during approach.
- **Identity / sensory PEs.** DA fires to identity-preserving reward swaps (grape ↔ banana) where scalar $\mathbb E[r]$ is unchanged.
- **Vigor.** DA modulates kinematics and response rate instantaneously — outside the credit-assignment role of $\delta_t$.
- **Heterogeneity.** Recordings show cell-type- and target-specific signals (threat in tail-of-striatum, movement in DLS).

These observations motivate the four extensions below.

#### 6.3 Distributional RL (Dabney et al. 2020)

Standard TD learns only $\mathbb E[Z]$ of the return $Z$. Distributional RL learns the full return distribution via a population of expectile/quantile estimators, each indexed by $\tau \in (0,1)$. The **asymmetric expectile loss** is

$$
L_\tau(\delta) \;=\; \bigl|\tau - \mathbb 1[\delta \le 0]\bigr|\, \delta^{2}.
$$

Differentiating w.r.t. the value estimate $V_\tau$ gives the asymmetric update

$$
\Delta V_\tau \;\propto\; \begin{cases} \alpha_{+}\, \delta, & \delta > 0 \\ \alpha_{-}\, \delta, & \delta \le 0 \end{cases}, \qquad \alpha_{+} = \alpha\, \tau,\; \alpha_{-} = \alpha\, (1 - \tau).
$$

The fixed point is the $\tau$-expectile of the return distribution. A **population** of DA neurons with heterogeneous $\tau$ (empirically: varying asymmetric slopes of responses to positive vs negative RPEs) jointly represent the full distribution of $Z$ — enabling risk-sensitive decisions that pure TD cannot make.

#### 6.4 Belief-State (POMDP) Extension

When the true state is hidden, the agent maintains a belief $b_t(s) = P(s \mid h_{1:t})$. The Bayes filter is

$$
b_t(s) \;\propto\; P(o_t \mid s)\, \sum_{s'} P(s \mid s', a_{t-1})\, b_{t-1}(s').
$$

Value over beliefs: $V(b_t) = \sum_s b_t(s)\, V(s)$. The belief-state RPE is

$$
\boxed{\;\delta_t \;=\; r_t + \gamma\, V(b_{t+1}) - V(b_t).\;}
$$

Two key consequences used by Gershman et al.:
1. **Ramps fall out.** Under noisy time (uncertainty kernel $p(t \mid \tau)$), the belief entropy and hence the bias in $V(b_t)$ grow with elapsed time; correcting via feedback yields a monotone ramp in $\delta_t$.
2. **Confidence-scaling.** Ambiguous sensory $o_t$ keeps $b_t$ broad, attenuates $V(b_t)$, and blunts the RPE — a direct mechanistic link between *perceptual uncertainty* and DA amplitude.

#### 6.5 Successor Representation (Generalized PE)

Decompose $V = M R$ where $R(s)$ is the reward vector and $M$ the discounted occupancy:

$$
M(s_i, s_j) \;=\; \mathbb E_\pi\!\left[\sum_{t=0}^{\infty} \gamma^{t}\, \mathbb 1[s_t = s_j] \,\Big|\, s_0 = s_i\right].
$$

Because $M$ obeys its own Bellman equation, it is learned via a *vector-valued* TD error

$$
\boxed{\;\delta_t(s) \;=\; \mathbb 1[s_t = s] + \gamma\, \hat M_{t+1}(s) - \hat M_t(s).\;}
$$

A DA ensemble broadcasting components of $\delta_t(s)$ explains firing to identity changes, preconditioning, and novel-cue responses that scalar RPE cannot capture.

#### 6.6 Multi-Timescale DA

Different DA subpopulations carry different discount factors $\gamma_k \in (0,1)$:

$$
\delta_t^{(k)} \;=\; r_t + \gamma_k\, \hat V^{(k)}(s_{t+1}) - \hat V^{(k)}(s_t).
$$

Averaging across $k$ implements a discrete Laplace transform of future rewards; this produces the observed spectrum of transient → sustained DA responses across striatal zones and gives a scale-invariant temporal memory.

#### 6.7 Average Reward, Tonic DA, and Vigor

In *average-reward* RL, value is differential:

$$
\delta_t \;=\; r_t - \hat\rho_t + \hat V(s_{t+1}) - \hat V(s_t), \qquad
\Delta \hat\rho_t = \eta\, \alpha\, \delta_t.
$$

Tonic DA encodes $\rho$ — the average reward rate / opportunity cost of time.

**Niv vigor.** Choose response rate (vigor) $v$ to maximize per-trial utility

$$
J(v) \;=\; R - C(v) - \rho\, \tau(v),
$$

with $C(v)$ convex (effort cost) and $\tau(v)$ the latency. FOC: $C'(v^{*}) = -\rho\, \tau'(v^{*})$. Higher $\rho$ ⇒ higher opportunity cost of slow movements ⇒ larger optimal $v^{*}$, explaining why tonic-DA manipulations change vigor without changing *which* action is chosen.

#### 6.8 Policy Gradient View of Phasic DA

Treat DA as the scalar that drives actor updates. For policy $\pi_\theta$,

$$
\nabla_\theta J(\theta) \;=\; \mathbb E_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, Q^{\pi_\theta}(s_t, a_t)\right],
$$

and with $V^\pi$ as baseline,

$$
\Delta \theta \;\propto\; \delta_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t).
$$

Phasic DA provides $\delta_t$; heterogeneous projections let different cortico-striatal loops run policy gradients on different sub-policies (motor kinematics, spatial strategy, economic choice) — a direct correlate of the *heterogeneity* emphasized in §6.1.

#### 6.9 Linking DA to Perceptual Decision Variables

**Confidence-weighted outcome RPE.** After a perceptual choice committed when the DDM accumulator reaches bound $B$, expected value equals confidence-weighted reward:

$$
\hat V_{\text{choice}} \;=\; P(\text{correct} \mid x(T) = B,\, T)\, R .
$$

Phasic DA at feedback is

$$
\boxed{\;\delta_{\text{out}} \;=\; r - P(\text{correct}\mid B, T)\, R .\;}
$$

High confidence ⇒ small positive RPE (reward expected); low confidence / guess ⇒ large positive RPE. This is the Lak et al. (2017) prediction the review endorses.

**Tonic DA → decision bound.** Under Niv's logic, the optimal bound minimizes deliberation cost $\rho\, T_{\text{dec}}(a) - A\, a / c^2 \cdot \mathrm{accuracy}(a)$. Differentiating w.r.t. $a$ gives

$$
a^{*} \;\propto\; f\!\left(\frac{1}{\rho}\right),
$$

i.e. tonic-DA elevation (high $\rho$) *lowers* the bound, trading accuracy for speed — consistent with §3.2 urgency and §4.2 urgency-gating. Thus dopamine is the neuromodulator that re-parameterizes the DDM online.

**Synthesis.** §§6.3–6.8 give dopamine multiple jobs — distributional PE, belief-state PE, feature PE, multi-timescale PE, average-reward signalling, and policy-gradient delivery. §6.9 shows each of these maps cleanly onto specific DDM parameters (drift via confidence-weighted $\delta_{\text{out}}$; bound via tonic $\rho$; policy via phasic actor updates). This is the bridge between RL and the perceptual-decision-making core of the first half of this review, and the natural stepping stone to the active-inference generalization in §7.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Introduction.** Classical RPE hypothesis (phasic midbrain DA = scalar $\delta = r + \gamma V' - V$) is too narrow to explain: (1) ramps, (2) sensory-feature responses, (3) motor/action modulation. Goal: show that generalized-PE extensions explain most data, but acknowledge that some phenomena (saliency, causal inference, adaptive $\alpha$) require genuinely new frameworks.
- **Box 1 — RL primer.** Bellman, TD error $\delta_t$, gradient descent $\Delta w = \alpha\delta_t \nabla_w\hat V_t$ (existing §6 Phase 2.1).
- **Why does DA ramp toward reward?** Under noisy timekeeping, convex value approximation + sensory-feedback-driven decay yields a non-zero asymptotic RPE ramp. VR teleport experiments: sudden distance jumps produce transient spikes ⇒ signal is RPE, not raw value (existing §6 Phase 2.4).
- **Why does DA respond to sensory features?** Novelty / identity changes. Generalized PE: vector-valued error over arbitrary features via successor representation.
- **Box 2 — Successor representation.** $V_t = \sum_s M_t(s) R(s)$, $M_t(s) = \mathbb E_\pi\bigl[\mathbb 1[S_t=s] + \gamma \mathbb 1[S_{t+1}=s] + \dots\bigr]$ (existing §6 Phase 2.5).
- **Box 3 — Origins of functional diversity.** Spatial topography (lateral = classical RPE, medial = sustained motivational); target-cell specificity (D1 Go / D2 NoGo); VTA→lateral septum D2-specific aggression circuit; D1/D2 opponency as distributional-RL substrate.
- **Why does DA respond to motor features?** Credit-assignment problem in distributed motor control. Off-policy continuous Q-learning with "action surprise" signal explains: decline with practice; tail-of-striatum action-PE driving choice bias without reward tracking.
- **DA in action selection & motivation.** Optogenetic DA at choice time instantly alters kinematics. Dual-channel (tonic action / phasic learning) reconciled via average-reward RL (Box 4). Opponent-actor learning: RPEs directly shape D1/D2 weighting ⇒ policy update is real-time.
- **Box 4 — Phasic/tonic via average-reward RL.** Differential value $V_t = \mathbb E_\pi(r_t - \rho + V_{t+1})$; $\delta_t = r_t - \hat\rho_t + \hat V(s_{t+1}) - \hat V(s_t)$; $\Delta \hat\rho_t = \eta\alpha\delta_t$ (existing §6 Phase 2.7).
- **Beyond RPEs.** Three residual-phenomena models: (i) perceived saliency (explains aversive-omission, novelty responses in NAc core); (ii) retrospective causal inference (local − background rate); (iii) adaptive learning rate (mesolimbic DA tunes $\alpha$).
- **Exploiting new methods.** FL-lifetime photometry of PKA reveals D1 vs D2 asynchronous biochemical plasticity; MoSeq pose-estimation + optogenetics reveals DA structures spontaneous behaviour at the syllable level.
- **Conclusions.** DA is a *mosaic* of mechanisms across timescales and circuits. Generalized PE explains most, but a full theory must also integrate saliency, causal inference, and adaptive $\alpha$.

### Appendix: Section-by-Section Backbone

**Introduction (untitled).** Classical RPE dominant for ~30 years. Three empirical challenges: ramps, sensory responses, motor modulation. Paper scope: show generalized-PE extensions resolve most; flag what remains.

**Box 1 — Brief review of RL concepts.**
- Value $V_t = \mathbb E_\pi(r_t + \gamma r_{t+1} + \dots)$; policy $\pi(a\mid s)$.
- Bellman: $V_t = \mathbb E_\pi(r_t + \gamma V_{t+1})$.
- TD error / RPE: $\delta_t = r_t + \gamma \hat V_{t+1} - \hat V_t$.
- Parameterized approximation $\hat V_t(w) \approx V(s_t)$; weight update $\Delta w = \alpha \delta_t \nabla_w \hat V_t$.
- Linear features $\hat V_t(w) = w^{\top} f(s_t)$ ⇒ gradient = feature vector.

**Why does DA ramp up as animals approach reward?** Ramps look like value signals but are actually RPEs under convex $\hat V$: $\delta_t \approx \hat V_{t+1} - \hat V_t$. VR teleport manipulations: discrete distance jumps produce transient RPE spikes. Ramps occur under spatial-navigation sensory feedback (timekeeping uncertainty ⇒ value-decay correction ⇒ non-zero asymptotic $\delta$) but not in classical delay conditioning.

**Why does DA respond to sensory features?** Novelty / surprise / identity changes produce DA firing independent of reward. Exploration-bonus models fit tail-of-striatum but not sensory preconditioning or identity unblocking. Resolution: generalized PE — DA trains a predictive map over sensory features, not just scalar reward.

**Box 2 — Successor representation.**
- $V_t = \sum_s M_t(s) R(s)$.
- $M_t(s) = \mathbb E_\pi\!\left[I_t(s) + \gamma I_{t+1}(s) + \gamma^2 I_{t+2}(s) + \dots\right]$ with $I_t(s) = \mathbb 1[S_t = s]$.
- Permits vector-valued feature PEs; generalizes scalar RPE.

**Box 3 — Origins of functional diversity.**
- Spatial topography in VTA (lateral RPE, medial sustained motivational).
- D1 direct pathway = Go; D2 indirect = NoGo.
- Combined projection + cell-type specificity (e.g. VTA→lateral septum D2 pathway modulating aggression).
- D1/D2 opponency plausibly encodes full distributional return statistics.

**Why does DA respond to motor features?** Dopamine encodes initiation, vigor, kinematics. Classical RPE can't do credit assignment in distributed motor control. Proposed solution: off-policy continuous Q-learning with a *teaching signal that combines classical RPE + action surprise* (deviation of sampled action from the argmax). Explains: lever-press DA declines with practice; tail-of-striatum action-PE biases choice without tracking reward.

**What is the role of DA in action selection and motivation?** Optogenetic DA at choice alters kinematics and motivation within milliseconds. Dual-channel hypothesis (tonic action, phasic learning) reconciled via average-reward RL (Box 4). Opponent-actor learning: phasic DA differentially modulates D1 (Go) / D2 (NoGo) ⇒ positive RPE increases intensity/vigor; links policy update and motor output in real time.

**Box 4 — Linking phasic and tonic DA via average-reward RL.**
- Differential value: $V_t = \mathbb E_\pi(r_t - \rho + V_{t+1})$, with $\rho = \lim_{T\to\infty} \mathbb E_\pi(\frac 1T \sum_{t=1}^T r_t)$.
- Updates: $\Delta \hat V_t(s_t) = \alpha \delta_t$, $\Delta \hat \rho_t = \eta \alpha \delta_t$.
- Average-reward RPE: $\delta_t = r_t - \hat \rho_t + \hat V_{t+1}(s_{t+1}) - \hat V_t(s_t)$.
- One $\delta_t$ updates *both* channels ⇒ mathematical unification of phasic and tonic DA; explains antagonism between tonic $\rho$ and phasic amplitude as reward rates rise.

**Beyond RPEs.** Three residual challenges:
1. **Perceived saliency** (salience × attentional value) — explains NAc-core DA to aversive omissions and novel neutral stimuli independent of valence.
2. **Retrospective causal inference** — DA tracks local − background rate to identify meaningful causes; explains *increasing* DA to uncued rewards across trials.
3. **Adaptive learning rate** — mesolimbic DA tunes $\alpha$ as a function of sensory-weight strength and policy change; resolves cue/reward DA dissociation in trace conditioning that pure TD cannot.

**Exploiting new methods to test theories of DA.**
- Fluorescent-lifetime photometry of PKA ⇒ D1 / D2 MSNs show asynchronous biochemical plasticity, matching dichotomous RL update rules.
- MoSeq (markerless pose estimation, behavioural syllables) + optogenetic DA ⇒ DA actively structures spontaneous behaviour at the syllable level (stochasticity, expression, speed).

**Conclusions.** DA is a *mosaic* across timescales and circuits. Generalized PE (value decay, SR feature coding, off-policy action surprise, average-reward unification) explains most. A full theory must additionally absorb saliency detection, causal inference, and adaptive $\alpha$.

---

## 7. Smith, Friston & Whyte (2022) — A Step-by-Step Tutorial on Active Inference

*Ryan Smith, Karl J. Friston, Christopher J. Whyte. Journal of Mathematical Psychology, 2022.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Traditional models treat perception, learning, and decision-making as separate processes with separate objective functions (e.g., likelihoods for perception, reward for action). The *Free Energy Principle* (FEP) claims a single imperative unifies them: an organism must **minimize variational free energy**, an upper bound on surprise. Until this tutorial, the FEP's dense mathematics kept it out of mainstream empirical use. Smith, Friston & Whyte aim to make the formalism *runnable*: explicit POMDP matrices, message-passing updates, and ready MATLAB code, so that empirical psychologists can fit active-inference models to real behavioural and neural data.

**Key Findings / Contributions.**
- **Unification via variational inference.** Perception = minimizing VFE over states $q(s)$. Action = minimizing *Expected* FE (EFE) over policies $q(\pi)$. The same objective subsumes the explore–exploit trade-off: EFE decomposes into *pragmatic value* (preference satisfaction) and *epistemic value* (information gain).
- **POMDPs with A/B/C/D/E matrices.** The generative model is fully specified by likelihoods $A$, transitions $B_\pi$, preferences $C$, priors $D$, and habits $E$ — a recipe transparent enough to fit empirically.
- **Neurobiologically grounded message passing.** Marginal message passing yields *neuronal* updates: state prediction errors drive membrane voltages $v_{\pi,\tau}$, softmax gives firing rates $s_{\pi,\tau}$; dopamine = precision $\gamma$ over policy EFE.
- **Empirical ERPs reproduced.** Hierarchical active-inference models reproduce the *mismatch negativity* (MMN) and *P300* as prediction errors at distinct temporal scales.
- **Connection to perceptual decision-making.** Perceptual choice becomes *self-evidencing*: the agent jointly infers hidden states and selects saccades/reports that most reduce EFE. In the DDM limit this recovers SPRT-style log-likelihood accumulation.

**Main Methodology.** Tutorial-style: (i) derive VFE as a tractable upper bound on surprise; (ii) instantiate discrete POMDPs with $A,B,C,D,E$; (iii) run gradient descent on VFE to infer states; (iv) compute EFE and softmax to select policies; (v) demonstrate fits to EEG/behavioural data (MMN, P300, foraging, decision-making).

**Initial Takeaway.** Active inference is not an alternative to DDM / RL / Bayes — it is a *unifying generative framework* that *contains* them. For perceptual decision-making this gives us: (a) explicit Bayesian belief updates that reduce to SPRT; (b) principled treatment of information-seeking actions (saccades, opt-outs) that pure DDM cannot express; (c) a mechanistic role for dopamine (precision $\gamma$) and norepinephrine (action precision $\alpha$) that dovetails with §6.

### Phase 2: Graduate-Level Deep Dive

#### 7.1 Variational Free Energy as a Tractable Bound on Surprise

Exact Bayesian inference requires $p(o) = \sum_s p(o,s)$, which is intractable for realistic state spaces. Introduce an approximate posterior $q(s)$ and measure its divergence from the true posterior:

$$
D_{\mathrm{KL}}\!\left[q(s)\,\|\,p(s\mid o)\right] \;=\; \sum_s q(s)\, \ln \frac{q(s)}{p(s\mid o)}.
$$

Using $p(s\mid o) = p(o,s)/p(o)$:

$$
D_{\mathrm{KL}}\!\left[q(s)\,\|\,p(s\mid o)\right] \;=\; \underbrace{\sum_s q(s)\, \ln \frac{q(s)}{p(o,s)}}_{\equiv \, F} \;+\; \ln p(o).
$$

Define **variational free energy**

$$
\boxed{\; F \;=\; \mathbb E_{q(s)}\!\left[\ln q(s) - \ln p(o,s)\right] \;=\; D_{\mathrm{KL}}\!\left[q(s)\,\|\,p(s\mid o)\right] - \ln p(o). \;}
$$

Since $D_{\mathrm{KL}} \ge 0$, we have $F \ge -\ln p(o)$ — $F$ upper-bounds surprise. Minimizing $F$ over $q$ (with $p(o)$ fixed) simultaneously (i) drives $q(s) \to p(s\mid o)$ (perception) and (ii) maximizes model evidence $\ln p(o)$.

Useful alternative decomposition:

$$
F \;=\; \underbrace{D_{\mathrm{KL}}[q(s)\,\|\,p(s)]}_{\text{complexity}} \;-\; \underbrace{\mathbb E_q[\ln p(o\mid s)]}_{\text{accuracy}}.
$$

#### 7.2 Discrete POMDP Generative Model

Observations $o_{1:T}$, hidden states $s_{1:T}$, policies $\pi \in \Pi$. The factorized generative model is

$$
p(o_{1:T}, s_{1:T}, \pi) \;=\; p(s_1)\, p(\pi)\, \prod_{\tau=1}^{T} p(o_\tau \mid s_\tau)\, \prod_{\tau=2}^{T} p(s_\tau \mid s_{\tau-1}, \pi),
$$

with categorical parameters:

| Matrix | Distribution | Role |
|---|---|---|
| $A$ | $p(o_\tau \mid s_\tau) = \mathrm{Cat}(A)$ | Likelihood |
| $B_{\pi,\tau}$ | $p(s_{\tau+1}\mid s_\tau, \pi) = \mathrm{Cat}(B_{\pi,\tau})$ | Policy-dependent transitions |
| $C$ | $p(o_\tau) = \sigma(C)$ | Log-preferences over outcomes |
| $D$ | $p(s_1) = \mathrm{Cat}(D)$ | State prior |
| $E$ | $p(\pi) = \mathrm{Cat}(E)$ | Habit prior over policies |

Factorized variational posterior: $q(s_{1:T}, \pi) = q(\pi)\, \prod_\tau q(s_\tau \mid \pi)$.

#### 7.3 State Estimation via Gradient Descent on $F$

Parameterize $q(s_\tau \mid \pi) = \sigma(v_{\pi,\tau})$ where $v_{\pi,\tau}$ is a log-belief / depolarization variable. The marginal message-passing gradient of $F$ w.r.t. $s_{\pi,\tau}$ gives the **state prediction error**

$$
\varepsilon_{\pi,\tau} \;=\; \tfrac{1}{2}\!\left(\ln(B_{\pi,\tau-1}\, s_{\pi,\tau-1}) + \ln(B_{\pi,\tau}^{\dagger}\, s_{\pi,\tau+1})\right) + \ln A^{\top} o_\tau - \ln s_{\pi,\tau} ,
$$

with $B^{\dagger}$ the column-normalized transpose. Iteratively update

$$
v_{\pi,\tau} \leftarrow v_{\pi,\tau} + \varepsilon_{\pi,\tau}, \qquad s_{\pi,\tau} \leftarrow \sigma(v_{\pi,\tau}),
$$

until $\varepsilon_{\pi,\tau} = 0$ — the fixed point at which $F$ is minimized. Identifying $v$ with membrane voltage, $s$ with firing rate, and $\varepsilon$ with prediction error gives a direct neural-circuit interpretation.

#### 7.4 Expected Free Energy $G$ Over Policies

Action requires projecting free energy *into the future*. For future time $\tau$ under policy $\pi$:

$$
G(\pi, \tau) \;=\; \mathbb E_{q(o_\tau, s_\tau \mid \pi)}\!\left[\ln q(s_\tau \mid \pi) - \ln p(o_\tau, s_\tau \mid \pi)\right].
$$

Total EFE over the policy: $G_\pi = \sum_\tau G(\pi, \tau)$.

**Decomposition 1 — epistemic + pragmatic.**

$$
\boxed{\; G_\pi \;=\; \underbrace{-\,\mathbb E_{q(o,s\mid\pi)}\!\left[\ln q(s\mid o, \pi) - \ln q(s\mid\pi)\right]}_{\text{epistemic: } -\text{info gain}} \;-\; \underbrace{\mathbb E_{q(o\mid\pi)}\!\left[\ln p(o\mid C)\right]}_{\text{pragmatic: } -\text{expected log preference}}. \;}
$$

Minimizing $G_\pi$ therefore simultaneously maximizes information gain (mutual information between observations and states) and expected log preference — a formal unification of exploration and exploitation.

**Decomposition 2 — risk + ambiguity.**

$$
G_\pi \;=\; \underbrace{D_{\mathrm{KL}}\!\left[q(o\mid\pi)\,\|\,p(o\mid C)\right]}_{\text{risk}} \;+\; \underbrace{\mathbb E_{q(s\mid\pi)}\!\left[H[p(o\mid s)]\right]}_{\text{ambiguity}} .
$$

*Risk* penalizes policies whose predicted outcomes diverge from preferences; *ambiguity* penalizes policies that lead to states with high sensory entropy (uninformative likelihood). The two decompositions are equivalent but highlight different objectives.

#### 7.5 Softmax Policy Selection and Precisions

$$
q(\pi) \;=\; \sigma\!\left(\ln E - F_\pi - \gamma\, G_\pi\right),
$$

combining (i) habit prior $\ln E$, (ii) retrospective fit $F_\pi$ (evidence already observed), (iii) prospective EFE $G_\pi$ weighted by precision $\gamma$.

**Dopamine = precision $\gamma$.** Tonic $\gamma$ sets the sharpness of the policy posterior. It is updated by a scalar *precision prediction error*

$$
G_{\text{error}} \;\leftarrow\; (\pi - \pi_0) \cdot (-G), \qquad \gamma \leftarrow \gamma + \eta\, G_{\text{error}},
$$

with $\pi_0$ the prior and $\pi$ the posterior over policies. When data confirm initial EFE evaluations, $G_{\text{error}} > 0$, $\gamma$ rises, confidence sharpens — mirroring the phasic/tonic DA dissociation of §6.

**Action precision $\alpha$.** Given inferred $\pi$, motor output $u_t$ is sampled with

$$
p(u_t) \;=\; \sigma\!\left(\alpha\, \ln p(u_t \mid \pi)\right).
$$

Smith, Friston & Whyte tie $\gamma$ to dopamine; broader literature ties $\alpha$ (action precision) and learning-rate $\eta$ (volatility) to **norepinephrine**, mapping unexpected uncertainty onto arousal.

#### 7.6 Hierarchical / Deep Temporal Models

For nested timescales, stack generative models. Higher-level state $s_\tau^{(2)}$ parameterizes the lower-level prior via

$$
D^{(1)} \;=\; A^{(2)\top}\, s_\tau^{(2)},
$$

and the terminal lower-level posterior $s_{t=T}^{(1)}$ serves as the higher-level observation $o_\tau^{(2)}$.

This produces two distinct prediction-error waveforms:

- *Fast / local* violations → early state PEs at level 1 → **mismatch negativity (MMN)**.
- *Slow / global* violations → delayed PEs at level 2 → **P300**.

A single generative model thus reproduces two canonical ERPs without ad-hoc assumptions — a strong empirical bridge from FEP to electrophysiology.

#### 7.7 Reduction to SPRT / DDM (Limit Case)

For a binary categorical state $\{s_1, s_2\}$ with i.i.d. observations $o_i$, the log-posterior ratio evolves by

$$
\ln\frac{q_n(s_2)}{q_n(s_1)} \;=\; \ln\frac{q_{n-1}(s_2)}{q_{n-1}(s_1)} \;+\; \ln\frac{p(o_n\mid s_2)}{p(o_n\mid s_1)}.
$$

Expanding from the $D$-prior:

$$
\mathrm{DV}_n \;=\; \ln\frac{q_n(s_2)}{q_n(s_1)} \;=\; \ln\frac{D_2}{D_1} \;+\; \sum_{i=1}^{n} \ln\frac{p(o_i\mid s_2)}{p(o_i\mid s_1)},
$$

which is *exactly* the SPRT weight-of-evidence of §2.2. In the continuous-time Gaussian limit this is the DDM. Commitment is triggered when $q(\pi)$ concentrates — i.e., when accumulated log-likelihood plus pragmatic value crosses the softmax threshold set by $\gamma, \alpha$. Thus active inference **contains** SPRT/DDM as a limit case, while additionally permitting epistemic (information-seeking) actions that DDM cannot represent.

#### 7.8 Synthesis Across the Review

- **§§1–3 (SDT, DDM, CPP):** recover as the perception-only limit ($G$ reduces to $F$; bounded accumulation).
- **§4 (Cisek):** affordance competition and urgency map onto precision $\gamma$ dynamics and action-precision $\alpha$ over competing policies.
- **§5 (Fleming):** propositional confidence $P(a=d\mid\hat s)$ is the posterior over $\pi$ given observations; meta-$d'$ indexes the fidelity of the $\gamma$-weighted readout.
- **§6 (Gershman):** phasic DA = $G_{\text{error}}$ on precision; tonic DA = average $\gamma$, controlling speed–accuracy as in the DDM bound.

**Bottom line.** Active inference does not replace the DDM; it places it inside a generative framework that also explains information-seeking, multi-timescale inference, precision control, and the neuromodulators that tune all three. This is the 2022+ synthesis that O'Connell & Kelly's "neurally-informed modelling" programme naturally points towards.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Introduction.** POMDP-based active inference unifies perception, learning, decision-making; actions chosen to trade off reward (pragmatic) vs information (epistemic); tutorial supplies MATLAB-ready recipes.
- **§1 Basic terminology, concepts, and mathematics.**
  - **§1.1** Bayes' rule; generative-model factorization $p(o, s, \pi) = p(o\mid s, \pi) p(s\mid\pi) p(\pi)$; goals encoded as prior preferences $p(o\mid C)$, not reward (existing §7 Phase 2.1).
  - **§1.2** VFE as a tractable bound on surprise; EFE as policy score balancing pragmatic and epistemic value (existing §7 Phase 2.1, 2.4).
  - **§1.3 (optional technical)** Derives $F_\pi \ge -\ln p(o)$ via Jensen; decomposes $F_\pi = D_{\mathrm{KL}}[q(s\mid\pi)\|p(s\mid\pi)] - \mathbb E_{q(s\mid\pi)}[\ln p(o\mid s)]$; $G_\pi$ splits into epistemic + pragmatic OR risk + ambiguity — unifies explore/exploit.
- **§2 Building and solving POMDPs.**
  - **§2.1** Temporal bookkeeping $t$ vs $\tau$; matrices $A, B_{\pi,\tau}, C, D, E$; $U$ (shallow) vs $V$ (deep) policies; precision $\gamma$ (existing §7 Phase 2.2).
  - **§2.2** Bayesian networks + Forney factor graphs; policy softmax $\pi = \sigma(\ln E - F - \gamma G)$ (existing §7 Phase 2.5).
  - **§2.3 (optional technical)** Variational / marginal message passing; mean-field factorization; state update $s_{\pi,\tau} = \sigma(\ln B_{\pi,\tau-1} s_{\pi,\tau-1} + \ln B^{\dagger}_{\pi,\tau} s_{\pi,\tau+1} + \ln A^\top o_\tau)$, with $\tfrac 12$ scaling under marginal MP (existing §7 Phase 2.3).
  - **§2.4** Prediction-error reformulation: state PE $\varepsilon_{\pi,\tau}$, depolarization $v \leftarrow v + \varepsilon$, firing rate $s = \sigma(v)$; outcome-level PE $\zeta_{\pi,\tau}$ combining risk + ambiguity gradients.
- **§3 Building specific task models.**
  - **§3.1** Explore-exploit multi-armed bandit, MATLAB: two hidden state factors (context, choice); matrices spelled out.
  - **§3.2** `spm_MDP_VB_X_tutorial` runs single trials; varying the Risk-Seeking magnitude in $C$ flips the agent between "pay for hint" and "guess now."
- **§4 Modeling learning.**
  - **§4.1 (optional)** Dirichlet conjugate priors parameterize matrices; concentration counts accumulate posteriors.
  - **§4.2** "Learning by counting" / Hebbian update $d_{\text{trial}+1} = \omega d_{\text{trial}} + \eta s_{\tau=1}$; forgetting rate $\omega$ handles volatility; EFE gains a *novelty* term $-A s_{\pi,\tau} \cdot W s_{\pi,\tau}$ driving parameter exploration.
  - **§4.3** 30-trial bandit simulation; risk-averse agent stops asking for hints as it learns; reversal at trial 5 triggers context belief switch and brief hint-seeking rebound.
- **§5 Neural process theory.** Maps MP equations onto cortical columns: superficial pyramidal = posterior/state-PE; deep = expected policies; synaptic weights = $A, B$; subcortical DA = precision $\gamma$ updated by $G_{\text{error}}$ (existing §7 Phase 2.5). Simulates raster plots, firing-rate traces, LFPs as first-derivative ERPs.
- **§6 Building hierarchical models.**
  - **§6.1** Two-level semi-Markov POMDP; $D^{(1)} = A^{(2)\top} s^{(2)}$; $o^{(2)} = s^{(1)}_{t=T}$ (existing §7 Phase 2.6).
  - **§6.2** Local-global auditory mismatch task; level-1 tone inference; level-2 four-tone sequence inference.
  - **§6.3** Simulation reproduces MMN (local violations) and P300 (global sequence violations).
- **§7 Fitting models to behavior.** DCM structure; variational-Laplace MLE; gradient descent over generative-model priors; parameter recoverability checks; Bayesian Model Selection (BMS); Parametric Empirical Bayes (PEB) for group-level inference.
- **§8 Concluding remarks.** A launching point; the field is advancing to multi-agent, deep parametric, and mixed discrete-continuous models.
- **Appendices.** (A) probability primer, VI derivation, EFE risk-ambiguity derivation, softmax, gamma function. (B) pencil-and-paper static + dynamic perception exercises with worked answers. (C) supplementary data.

### Appendix: Section-by-Section Backbone

**Introduction.** Active-inference POMDP unifies perception / learning / decision-making; goals = prior preferences $p(o\mid C)$; tutorial targets empirical researchers lacking heavy ML background.

**§1 Basic terminology, concepts, and mathematics.** Structural header.
- **§1.1 Mathematical foundations.** Bayes: $p(s\mid o, m) = p(o\mid s, m) p(s\mid m) / p(o\mid m)$. Generative model factored as $p(o, s, \pi) = p(o\mid s, \pi) p(s\mid\pi) p(\pi)$. Categorical inference over policies $\pi$; preferences $p(o\mid C)$ replace reward.
- **§1.2 Non-technical VFE / EFE.** $F = \sum_s q(s) \ln[q(s)/p(o,s)]$; $F = \mathbb E_{q(s)}[\ln q(s)/p(s\mid o)] - \ln p(o)$ ⇒ $F$ upper-bounds surprise. Minimizing $F$ pulls $q \to p(s\mid o)$ and maximizes model evidence while penalizing complexity. Policy scoring uses EFE (pragmatic + epistemic).
- **§1.3 Technical (optional).** Jensen's-inequality derivation of $F_\pi \ge -\ln p(o)$; $F_\pi = D_{\mathrm{KL}}[q(s\mid\pi)\|p(s\mid\pi)] - \mathbb E_{q(s\mid\pi)}[\ln p(o\mid s)]$ (complexity minus accuracy). $G_\pi = \mathbb E_{q(o,s\mid\pi)}[\ln q(s\mid\pi) - \ln p(o,s\mid\pi)]$ decomposed either as epistemic + pragmatic or as $G_\pi = D_{\mathrm{KL}}[q(o\mid\pi)\|p(o\mid C)] + \mathbb E_{q(s\mid\pi)}[H[p(o\mid s)]]$ (risk + ambiguity).

**§2 Building and solving POMDPs.** Structural header.
- **§2.1 Formal POMDP structure.** $t$ = now, $\tau$ = time-index in agent's belief. Matrices: $A$ likelihood; $B_{\pi,\tau}$ transitions; $C$ preferences; $D$ initial-state prior; $E$ habits. Policies: shallow $U$ or deep $V$. Precision $\gamma$ modulates EFE weighting.
- **§2.2 Graphical models.** Bayes-net and Forney-style factor graphs. Policy: $\pi = \sigma(\ln E - F - \gamma G)$.
- **§2.3 Variational / marginal message passing (optional technical).** Mean-field factorization. State update: $s_{\pi,\tau} = \sigma(\ln B_{\pi,\tau-1} s_{\pi,\tau-1} + \ln B^{\dagger}_{\pi,\tau} s_{\pi,\tau+1} + \ln A^\top o_\tau)$. Marginal MP adds $\tfrac 12$ scaling to prevent overconfidence.
- **§2.4 Prediction-error formulation.** State PE $\varepsilon_{\pi,\tau} = \tfrac 12 (\ln B_{\pi,\tau-1} s_{\pi,\tau-1} + \ln B^{\dagger}_{\pi,\tau} s_{\pi,\tau+1}) + \ln A^\top o_\tau - \ln s_{\pi,\tau}$; $v \leftarrow v + \varepsilon$; $s = \sigma(v)$. Outcome-level PE $\zeta_{\pi,\tau} = A s_{\pi,\tau} \cdot (\ln A s_{\pi,\tau} - \ln C_\tau) - \mathrm{diag}(A^\top \ln A) \cdot s_{\pi,\tau}$, combining risk + ambiguity.

**§3 Building specific task models.** Structural header.
- **§3.1 Explore-exploit task.** Multi-armed bandit with two hidden state factors (context, choice). Concrete $D, A, B, C, V$ matrices walked through.
- **§3.2 Running and plotting simulations.** `spm_MDP_VB_X_tutorial` MATLAB; output plots = state posterior, action probability, chosen policy, outcomes, expected-precision (DA) updates. Manipulating Risk-Seeking weight in $C$ flips hint-vs-guess behaviour.

**§4 Modeling learning.** Structural header.
- **§4.1 Dirichlet priors (optional technical).** Conjugate priors for categorical updates; concentration counts accumulate from posterior.
- **§4.2 Non-technical continuation.** Hebbian "count" update: $d_{\text{trial}+1} = \omega d_{\text{trial}} + \eta s_{\tau=1}$; $\omega$ controls forgetting/volatility. EFE gains a *novelty* term $-A s_{\pi,\tau} \cdot W s_{\pi,\tau}$ ⇒ parameter-exploration drive.
- **§4.3 Simulating learning.** 30-trial simulation; risk-averse agent halts hint-seeking after learning; reversal at trial 5 triggers rapid belief switch + brief hint-seeking rebound.

**§5 Neural process theory.** Cortical-column mapping: superficial pyramidal = state posterior/PE; deep = expected policies; synapses = $A, B$; subcortical DA = $\gamma$; $G_{\text{error}}$ updates $\gamma$. Simulates rasters, firing rates, and LFP/ERP from first-derivative of firing rates.

**§6 Building hierarchical models.** Structural header.
- **§6.1 Structure.** Deep-temporal POMDP: $D^{(1)} = A^{(2)\top} s^{(2)}$; $o^{(2)} = s^{(1)}_{t=T}$. Higher level slower timescale ⇒ models working memory, narrative comprehension.
- **§6.2 Building.** Auditory local-global mismatch task; level-1 tone (high/low), level-2 four-tone sequence; full A/B/C/D matrices listed.
- **§6.3 Plotting.** Level-1 tone violations ⇒ fast ERP = MMN; level-2 sequence violations ⇒ slower ERP = P300.

**§7 Fitting models to behavior.** DCM struct (generative model + participant actions + observations); variational-Laplace MLE by gradient descent; priors trade off complexity vs fit; parameter *recoverability* checks (simulate-then-recover); Bayesian Model Selection via free energy; Parametric Empirical Bayes for group-level analyses.

**§8 Concluding remarks.** Tutorial is a launching point; field is moving toward multi-agent, deep parametric, mixed discrete-continuous extensions.

**Appendix A — Additional mathematical details.** Probability primer (sum/product/Bayes); variational-inference derivation of $F$; EFE derivation into epistemic+pragmatic and risk+ambiguity (Eqs A.12–A.13); extension with parameter uncertainty ⇒ salience + novelty (Eq A.14); softmax definition with precision; gamma function for non-integer Dirichlet counts.

**Appendix B — Pencil-and-paper exercises.** Static perception: $s = \sigma(\ln D + \ln A^\top o)$. Dynamic perception (HMM): $s_{\tau=1} = \sigma(\tfrac 12 (\ln D + \ln B^{\dagger}_\tau s_{\tau+1}) + \ln A^\top o_\tau)$; $s_{\tau=2} = \sigma(\tfrac 12 (\ln B_{\tau-1} s_{\tau-1}) + \ln A^\top o_\tau)$. Worked answers provided.

**Appendix C — Supplementary data.** DOI link to online materials.

---

## Cross-Paper Synthesis

The seven reviews trace a single conceptual lineage:

1. **Inference primitives.** Gold & Shadlen (2007) cast decisions as SPRT/DDM with LIP as integrator. Hanks & Summerfield (2017) generalize this across species, raising doubts about parietal as the unique bottleneck and surfacing urgency / confidence. O'Connell & Kelly (2021) bring the framework to humans via CPP and neurally-informed hierarchical Bayesian fits.
2. **Beyond the 2AFC.** Cisek (2021) argues the whole 2AFC canon is a special case of embodied affordance competition and urgency-gating; decision variables live in the sensorimotor circuits that execute action.
3. **Beyond the choice.** Fleming (2024) formalizes confidence (meta-$d'$, two-stage DDM) and maps it onto vmPFC / pMFC / frontopolar.
4. **Beyond single-trial inference.** Gershman et al. (2024) frame dopamine as a *family* of prediction errors tuning drift, bound, and vigor — the neuromodulatory substrate of everything above.
5. **Beyond RPE.** Smith, Friston & Whyte (2022) supply an active-inference framework in which all of the above — SPRT, urgency, confidence, RPE, opportunity cost — are limit cases of minimizing (expected) free energy over a generative POMDP.

Put differently, each paper *adds a factor* to the generative model of the previous: evidence $\to$ action plan $\to$ confidence $\to$ learning / vigor $\to$ policy-level free energy. A modern treatment of perceptual decision-making now requires engaging with all five layers simultaneously.

---

## 8. LeDoux & Daw (2018) — Surviving Threats: Taxonomy of Defensive Behaviour

*Joseph LeDoux & Nathaniel D. Daw. Nature Reviews Neuroscience, 2018.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Classical "fear conditioning" treats all defensive behaviour as a single phenomenon — a reflexive response elicited by aversive cues, mediated by amygdala circuits. LeDoux & Daw argue this conflates *defensive survival circuits* (unconscious, reactive) with *subjective fear* (the feeling) and with *goal-directed avoidance* (flexible, cognitive). The paper's goal is to replace the monolithic "fear" concept with a **6-level taxonomy** whose levels map cleanly onto distinct computational algorithms and distinct neural circuits.

**Key Findings.**
- **Six-level taxonomy** from reflex → deliberation:
  1. **Reflexes** (startle) — hardwired brainstem.
  2. **Fixed reaction patterns** (freezing, flight) — species-specific defense reactions (SSDRs), LA→CeA→PAG.
  3. **Pavlovian habits** — cached Stimulus→Response associations, dorsolateral striatum after overtraining.
  4. **Instrumental habits** — model-free Q-learning over actions; BA→NAcc.
  5. **Action–outcome (goal-directed) avoidance** — model-based RL over a learned transition model; prefrontal / hippocampal.
  6. **Deliberative / conscious planning** — explicit tree search, possibly with conscious "fear" as a metacognitive label.
- **Subjective fear is decoupled from defensive circuits.** Levels 1–4 proceed without conscious fear; conscious fear belongs to level 6 and is *not* causal for freezing or habitual avoidance.
- **Circuit mapping.** LA→CeA→PAG for reactions; LA→BA→NAcc for instrumental avoidance; infralimbic PFC switches between them; BNST handles *sustained, uncertain* threat (anxiety) vs. phasic amygdala for identifiable threat.
- **Pavlovian–instrumental interaction.** Pavlovian aversive value biases toward behavioural inhibition (freezing), producing the characteristic approach–avoidance asymmetry that must be overcome for active avoidance.

**Main Methodology.** A conceptual / theoretical synthesis bridging ethology, associative-learning theory, rodent circuit-tracing and lesion work, and computational RL (Daw et al., Dayan, Sutton & Barto).

**Initial Takeaway.** Defensive behaviour is not one thing. It is a hierarchy of decision systems — reflex, Pavlovian habit, model-free instrumental, model-based goal-directed — each with a matching RL algorithm and a matching circuit. Clinical implication: PTSD / OCD / phobia are best re-cast as *imbalances between levels* (e.g. habit dominating goal-direction) rather than "too much fear."

### Phase 2: Graduate-Level Deep Dive

> The paper is a theoretical review and cites formalisms without reprinting them; equations below are standard RL (Sutton & Barto; Daw et al. 2005; Rescorla–Wagner).

#### 8.1 Level-by-Level Computational Mapping

| Level | Algorithmic class | Canonical update | Neural substrate |
|---|---|---|---|
| Reflex / SSDR | Innate stimulus→response | none (genetic prior) | Brainstem, LA→CeA→PAG |
| Pavlovian habit | TD-learned cached value | $\Delta V = \alpha\, \delta$ | Amygdala, dorsolateral striatum |
| Instrumental habit | Model-free Q-learning | SARSA / Q-learning | BA→NAcc, dorsolateral striatum |
| Action–outcome | Model-based RL (tree search) | Bellman backup over $T, R$ | mPFC, hippocampus |
| Deliberation | Model-based + meta | planning + self-model | lateral PFC, frontopolar |

#### 8.2 Pavlovian Reaction: Rescorla–Wagner and TD

For a CS–US pairing, associative strength $V(s)$ of the CS updates with prediction error

$$
\Delta V(s) \;=\; \alpha\, \beta\, \bigl(\lambda - V(s)\bigr), \qquad \delta = \lambda - V(s),
$$

with $\lambda$ the US magnitude. In time-extended settings (serial compound stimuli), the TD generalisation is

$$
\delta_t \;=\; r_t + \gamma\, V(s_{t+1}) - V(s_t), \qquad V(s_t) \leftarrow V(s_t) + \alpha\, \delta_t .
$$

Lateral-amygdala plasticity tracks this $\delta$ in single-unit and optogenetic studies.

#### 8.3 Model-Free Instrumental Avoidance: Q-Learning / Actor-Critic

**Q-learning** update for state–action value:

$$
Q(s,a) \;\leftarrow\; Q(s,a) + \alpha\!\left[r + \gamma\, \max_{a'} Q(s',a') - Q(s,a)\right].
$$

**Actor–critic** partitioning (BLA = critic, striatum = actor):

$$
\text{critic: } V(s) \leftarrow V(s) + \alpha\, \delta, \qquad
\text{actor: } \pi(a\mid s) \leftarrow \pi(a\mid s) + \alpha\, \delta .
$$

In avoidance, $r$ is the *termination* of an aversive stimulus (safety signal) — negative RPE during threat, positive RPE at offset — which is the canonical two-factor account LeDoux & Daw formalize as RL.

#### 8.4 Model-Based (Goal-Directed) Avoidance

Learn transitions $T(s, a, s') = P(s' \mid s, a)$ and reward/cost $R(s, a)$ and solve Bellman online:

$$
\boxed{\; Q(s,a) \;=\; R(s,a) + \gamma\, \sum_{s'} T(s,a,s')\, \max_{a'} Q(s',a'). \;}
$$

Critical prediction: if the environment changes (escape route blocked), model-based control re-plans immediately, whereas model-free Q-values perseverate with the old habit — the operational definition of "goal-directedness" in devaluation and contingency-degradation tests.

#### 8.5 Arbitration Between Model-Free and Model-Based

Daw et al. (2005) propose arbitration by posterior uncertainty. With variances $\sigma^2_{\text{MF}}, \sigma^2_{\text{MB}}$ of each system's $Q$ estimate:

$$
w_{\text{MB}} \;=\; \frac{1/\sigma^2_{\text{MB}}}{1/\sigma^2_{\text{MB}} + 1/\sigma^2_{\text{MF}}},
\qquad Q_{\text{net}} \;=\; w_{\text{MB}}\, Q_{\text{MB}} + (1 - w_{\text{MB}})\, Q_{\text{MF}}.
$$

Under *threat imminence*, MB search becomes too slow or its $\sigma^2_{\text{MB}}$ explodes, so $w_{\text{MB}} \to 0$ and control collapses onto MF habits and then onto innate reflexes — exactly the behavioural transition from flight planning to freezing to startle as a predator closes in.

#### 8.6 Pavlovian Bias and Approach–Avoid Asymmetry

Pavlovian aversive value biases policy toward behavioural inhibition regardless of instrumental $Q$:

$$
Q_{\text{net}}(s,a) \;=\; w_{\text{inst}}\, Q_{\text{inst}}(s,a) \;+\; w_{\text{pav}}\, V_{\text{pav}}(s)\, \kappa(a),
$$

with $\kappa(a) > 0$ for "withhold / inhibit" actions and $\kappa(a) < 0$ for "approach / act." When $V_{\text{pav}} \ll 0$, the Pavlovian term rewards inhibition — producing freezing even when the instrumental optimum is to flee or press a lever. Successful active avoidance requires infralimbic PFC (PL/IL) to suppress this bias.

#### 8.7 Extinction as Inhibitory Learning

Extinction is *not* erasure of $V_{\text{excitatory}}$; it is acquisition of a competing inhibitory trace $V_{\text{inhibitory}}$:

$$
V_{\text{net}} \;=\; V_{\text{excitatory}} \;-\; V_{\text{inhibitory}}.
$$

Return of fear (spontaneous recovery, renewal, reinstatement) is explained by context-dependent gating of the two traces. Infralimbic mPFC drives $V_{\text{inhibitory}}$; BLA stores $V_{\text{excitatory}}$; hippocampus supplies context for gating.

#### 8.8 Uncertain Threat → POMDP and the BNST

When threat is uncertain (dark alley, ambiguous cue), the agent does not know the true state $s$ and maintains belief $b(s)$. Bayes filter update:

$$
b'(s') \;=\; \frac{P(o \mid s')\, \sum_s P(s' \mid s, a)\, b(s)}{\sum_{s''} P(o \mid s'')\, \sum_s P(s'' \mid s, a)\, b(s)} .
$$

Value over beliefs: $V(b) = \sum_s b(s)\, V(s)$. Policy:

$$
\pi^{*}(b) \;=\; \arg\max_{a}\, \mathbb E_b\!\left[R(s,a) + \gamma\, V(b')\right].
$$

LeDoux & Daw map phasic, low-entropy threat processing onto the **central amygdala (CeA)** and sustained, high-entropy (anxious) belief-state processing onto the **bed nucleus of the stria terminalis (BNST)** — a cleanly circuit-testable split between MDP and POMDP regimes.

#### 8.9 Circuit Summary

- **Reactions (freezing/SSDR):** sensory → LA → CeA → PAG.
- **Instrumental active avoidance:** sensory → LA → BA → NAcc / dorsomedial striatum.
- **Habit avoidance (overtrained):** dorsolateral striatum; amygdala-independent.
- **Switching reaction ↔ action:** infralimbic PFC inhibits CeA-PAG freezing and disinhibits BA-NAcc avoidance.
- **Uncertain / sustained threat:** BNST; drives HPA axis and generalized anxiety.

#### 8.10 Bridge to the Perceptual-Decision Framework of §§1–7

Defensive decisions are *decisions* in the same formal sense as §§1–7, but with a different value landscape (avoiding cost $R < 0$) and an additional Pavlovian bias term. The same machinery appears:
- **DDM / SPRT (§1–2)** → sequential evidence accumulation for "is there a threat?" under a POMDP belief filter (§8.8).
- **Urgency / collapsing bound (§4)** → threat imminence lowering the effective bound and shifting arbitration from MB to MF to reflex (§8.5).
- **Confidence (§5)** → posterior on threat state $P(\text{threat} \mid o)$ driving BNST vs CeA engagement.
- **Dopamine RPE (§6)** → Rescorla–Wagner / TD in lateral amygdala; aversive RPE in the same framework.
- **Active inference (§7)** → the POMDP belief filter of §8.8 *is* the variational posterior; defensive action selection minimizes expected free energy with preferences $C$ heavily penalizing high-$V_{\text{excitatory}}$ states.

**Bottom line.** LeDoux & Daw translate the psychological hierarchy of defensive behaviour into a *computational* hierarchy — reflex < Pavlovian habit < MF instrumental < MB goal-directed < deliberative — that sits cleanly inside the perceptual-decision / RL / active-inference formalisms reviewed above. The rest of the threat block (§§9–10) now fills in (i) what it means to compute *safety* rather than threat, and (ii) the detailed neural computations of threat itself.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Introduction.** Monolithic "fear" obscures distinct defensive computations; the paper advances a hierarchical taxonomy tying psychological classes to algorithms and circuits.
- **A defensive taxonomy.** Six-level hierarchy extending Dickinson & Balleine: (1) reflexes, (2) fixed reaction patterns, (3) habits, (4) action–outcome behaviours, (5) implicit deliberative actions, (6) explicit deliberative actions. Algorithmic mapping: preprogrammed → model-free → simple model-based → complex cognitive forecasting (existing §8 Phase 2.1).
- **Innate responses.** Reflexes (fast, restricted-muscle, e.g. acoustic startle) vs fixed reaction patterns (freeze / flight, context- and imminence-dependent). Both can be recruited by CS via Pavlovian conditioning.
- **Defensive instrumental behaviours.** Action-outcome (goal-directed with simple world-model) vs habits (outcome-independent after overtraining) vs deliberative forecasting (implicit non-conscious working memory; explicit conscious planning, likely human-unique).
- **Box 1 — Actions vs habits in avoidance.** Historical rejection of instrumental-avoidance view (Bolles) now overturned: devaluation tests confirm active avoidance *is* goal-directed early on, then transitions into habit with overtraining.
- **Neural circuits for defence.** Structural header — reactions → actions → habits → deliberation maps onto shifting neural substrates.
- **Circuits for innate responses.**
  - Reflexes: short brainstem pathways (cochlear nucleus → pontine RN → motor neurons); CeA potentiates during Pavlovian conditioning.
  - Fixed reactions (freezing): innate threat (predator odour) via LA / medial amygdala → VMH + PAG; learned Pavlovian via LA plasticity → CeA → PAG + autonomic modulators (existing §8 Phase 2.8).
  - Sustained / uncertain threat shifts control from CeA to **BNST** (the phasic-vs-sustained distinction; existing §8 Phase 2.8 POMDP).
  - Human fMRI confirms implicit amygdala processing.
- **Box 2 — Conscious emotional experiences.** Defensive responses can occur without subjective fear; lesions dissociating fear feeling from freezing exist. Designate the amygdala–PAG network as *defensive survival circuit*, not "fear circuit". Subjective fear is a higher-order cognitive state assembled in working memory, causally relevant only for level 6 (explicit deliberation).
- **Circuits for instrumental responses.** Three phases:
  1. Pavlovian freezing via LA → CeA → PAG.
  2. Goal-directed active avoidance via LA + BA → NAcc.
  3. **Arbitration by infralimbic PFC (PFCIL):** suppresses CeA-freezing, facilitates BA–NAcc avoidance (existing §8 Phase 2.9).
  - Overtrained avoidance shifts to **dorsolateral striatum**, becoming amygdala-independent habit.
  - Novel-route deliberative avoidance recruits hippocampus (cognitive map) + PFC (working memory).
- **Box 3 — Appetitive ↔ aversive circuit parallels.** LA + BA required for both goal-directed appetitive and goal-directed aversive actions; dispensable for the respective habits. Amygdala lesions: appetitive habits still acquirable, *active avoidance not acquirable at all* — amygdala indispensable for initial aversive action-outcome contingency.
- **Conclusions.** "Fear" alone is inadequate; survival = multiplicity of computational systems. Clinical implication: PTSD, OCD, anxiety / depression arise from specific *imbalances* in deployment (e.g. OCD = habit dominance; anxiety = excessive rumination / deliberation; depression = failed arbitration between active and passive coping). Explicit deliberation as a distinct computational level means subjective experience must enter clinical treatment design.

### Appendix: Section-by-Section Backbone

**Introduction (untitled).** Historical imbalance: innate-reaction focus obscures goal-directed defensive choice. Claim: hierarchical taxonomy of psychological/computational classes maps to distinct circuits.

**A defensive taxonomy.** Six categories, extending Dickinson & Balleine:
- Elicited: (1) reflexes; (2) fixed reaction patterns.
- Emitted instrumental: (3) habits; (4) action–outcome; (5) implicit deliberative; (6) explicit deliberative.
- Algorithmic mapping: preprogrammed → MF → MB (associative or cognitive).

**Innate responses.**
- Reflexes — rapid, restricted-muscle (startle, eye-blink); unconditioned US.
- Fixed reaction patterns — whole-body coordinated (freeze, flight); context + imminence-dependent.
- Both can come under CS control via Pavlovian threat conditioning.

**Defensive instrumental behaviours.**
- Action–outcome: learned contingency action ↔ safety (e.g. lever-press stops shock); simple world-model.
- Habits: model-free, outcome-independent after overtraining.
- Deliberative actions: cognitive forecasting / spatial maps / episodic memory; implicit (most animals) or explicit (human conscious planning).

**Box 1 — Actions vs habits in avoidance.** Appetitive devaluation test is the gold standard. Historical claim (Bolles): avoidance isn't instrumental, just Pavlovian flight. Modern consensus (and this paper): avoidance is initially goal-directed, becomes habit with overtraining.

**Neural circuits for defence.** (Structural header.) Circuit substrate shifts as behaviour class shifts.

**Circuits for innate responses.**
- Reflex: cochlear nucleus → pontine RN → motor neurons; CeA potentiates.
- Freezing (innate): LA / medial amygdala → VMH + PAG.
- Freezing (learned Pavlovian): LA plasticity → CeA → PAG + autonomic modulators.
- Sustained / uncertain threat: BNST takes over from CeA.
- Human fMRI + lesions confirm implicit amygdala processing.

**Box 2 — Conscious emotional experiences and explicit action forecasting.** Defensive behaviour can occur without conscious fear; lesions can abolish freezing without abolishing fear feeling. Rename: defensive *survival* circuit, not fear circuit. Conscious fear = higher-order working-memory construct; relevant only for explicit deliberation.

**Circuits for instrumental responses.**
- Avoidance phase 1: freezing via LA → CeA → PAG.
- Avoidance phase 2: goal-directed instrumental via LA + BA → NAcc ventral striatum.
- Arbitration: **infralimbic PFC (PFCIL)** suppresses freezing, enables avoidance.
- Avoidance phase 3: habit via dorsolateral striatum; amygdala-independent.
- Deliberative spatial avoidance: hippocampus (maps) + PFC (WM).

**Box 3 — Appetitive/aversive parallels.** LA + BA required for goal-directed in both domains; dispensable once habitual. Amygdala lesions abolish active avoidance entirely but leave appetitive habits acquirable ⇒ LA/BA is indispensable for aversive action-outcome acquisition.

**Conclusions.** Survival = multiple computational systems. Clinical reframing:
- OCD = dominance of habitual avoidance.
- Anxiety = excessive implicit/explicit deliberation / rumination.
- Depression = failed PFC arbitration between active and passive coping.
- Explicit deliberation entails subjective experience, which must enter clinical care.

---

## 9. Tashjian, Zbozinek & Mobbs (2021) — A Decision Architecture for Safety Computations

*Sarah M. Tashjian, Tomislav D. Zbozinek, Dean Mobbs. Trends in Cognitive Sciences, 2021.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Classical affective neuroscience treats *safety* as the mere absence of threat ($\text{safety} = 1 - \text{threat}$). Tashjian, Zbozinek, and Mobbs argue this is wrong: safety is a **distinct, active computation**. Without it, an organism stays locked in defensive circuits and cannot forage, mate, or explore. They ask: *how is safety computed, and how does it gate the transition from defensive to appetitive behaviour?*

**Key Findings.**
- **Dual-evaluation architecture.** Safety $S(t)$ integrates two orthogonal vectors:
  - **Threat-oriented:** imminence, value (magnitude of harm), uncertainty.
  - **Self-oriented:** policies (available protective actions), experience, **perceived control**.
- **Safety ≠ inverted threat.** Because self-oriented variables (e.g. being inside a shark cage) can *decouple* $S(t)$ from objective threat, an agent can be both under high objective threat and compute high safety.
- **vmPFC dichotomy.** *Anterior* vmPFC is the **safety hub** (co-activates with hippocampus and ventral striatum; drives non-defensive behaviour). *Posterior* vmPFC is the **anticipated-threat hub** (co-activates with amygdala and PAG).
- **Clinical implication.** PTSD / anxiety are recast as *deficits in safety computation* — failures of safety generalization, underweighted perceived control, and excessive ambiguity aversion — not simply hyperactive threat processing.

**Main Methodology.** Conceptual synthesis plus Neurosynth meta-analytic fMRI decoding to establish the anterior-vs-posterior vmPFC functional dissociation. Integrates ethology (Threat Imminence Continuum) with RL / Bayesian control frameworks.

**Initial Takeaway.** Safety is a *positive, self-model-dependent* decision variable, computed in anterior vmPFC, that gates the release of defensive policies and enables exploration. Formally, it slots into active inference as a prior preference $C$, into DDM as a drift and bound modulator, and into Pavlovian RL as a distinct inhibitory term.

### Phase 2: Graduate-Level Deep Dive

> The paper states the architecture conceptually without printing formal equations; equations below are reconstructed from the literature it cites (Rescorla & Wagner; Shepard 1987; Ratcliff DDM; Smith, Friston & Whyte 2022).

#### 9.1 Safety as a Distinct Decision Variable

Let the threat vector be $\mathbf T = (I, V_T, U_T)$ (imminence, threat value, threat uncertainty) and the self-oriented vector be $\mathbf O = (\Pi, E, K)$ (available policies, prior experience, perceived control). Safety is

$$
\boxed{\; S(t) \;=\; f\bigl(\mathbf T(t),\, \mathbf O(t)\bigr), \;}
$$

with the crucial property $\partial S/\partial K > 0$ even at fixed $\mathbf T$ — *control decouples safety from threat*.

The full utility of an action embeds both threat and safety as separate terms (common-currency extension):

$$
U(a) \;=\; w_1\, V_{\text{reward}}(a) \;+\; w_2\, V_S(a) \;-\; w_3\, V_{\text{threat}}(a).
$$

Safety is thus a *positive* utility, not a negated threat.

#### 9.2 Safety Learning: Conditioned Inhibition

In Rescorla–Wagner for a feature-positive/feature-negative paradigm (A+, AX$-$):

$$
\delta \;=\; r - \bigl(V_A + V_X\bigr), \qquad \Delta V_X \;=\; \alpha_X\, \beta\, \delta .
$$

When A predicts shock ($V_A > 0$) but the AX compound yields no shock ($r = 0$), $\delta < 0$ and $V_X \to$ negative, giving X conditioned-inhibitor status. Under the *distinct-safety* account Tashjian et al. advocate, the brain additionally maintains a positive $V_S$ scalar,

$$
\Delta V_S \;=\; \alpha_S\, \bigl(\lambda_S - V_S\bigr), \qquad \lambda_S = \mathbb 1[\text{expected threat omitted}] ,
$$

read out by anterior vmPFC/ventral striatum and used directly in $U(a)$ of §9.1.

#### 9.3 Safety Generalization (Shepard)

Let $x$ index a context/stimulus in a psychological feature space with metric $d$. Shepard's universal law of generalization gives

$$
S(x_{\text{new}}, x_i) \;=\; \exp\!\bigl(-c\, \| x_{\text{new}} - x_i \|^{p}\bigr),
$$

with $c$ a sensitivity parameter and $p \in \{1, 2\}$. Generalised safety value:

$$
V_S(x_{\text{new}}) \;=\; \sum_i S(x_{\text{new}}, x_i)\, V_S(x_i).
$$

For structured multidimensional contexts, represent each context as a tensor outer product $\mathbf f = \mathbf c \otimes \mathbf s$ and use cosine similarity in the resulting product space. Clinical overgeneralization of threat / undergeneralization of safety corresponds to a *threat-specific* $c_T > c_S$ asymmetry.

#### 9.4 vmPFC — Amygdala — Hippocampus Circuit

- **Threat vector $\mathbf T$:** sensory cortex → BLA → **posterior vmPFC** → CeA / PAG (defensive output).
- **Self-oriented vector $\mathbf O$:** hippocampus (experience, context) + lateral PFC (policies) → **anterior vmPFC**.
- **Comparator:** anterior vmPFC integrates $\mathbf T$ and $\mathbf O$, produces $S(t)$, and exerts top-down inhibition on CeA:

$$
r_{\text{CeA}}(t) \;=\; \max\!\bigl(0,\; W_{\text{threat}}\, \mathbf T(t) \;-\; W_{\text{safety}}\, r_{\text{ant-vmPFC}}(t)\bigr).
$$

- **Behavioural switch:** when $S(t)$ exceeds threshold, BLA output is redirected from CeA (freezing) to **NAcc** (appetitive/exploratory actions) — aligning with LeDoux & Daw §8.9.

#### 9.5 Safety in DDM: Drift and Bound Modulation

For an approach-vs-avoid DDM, safety modulates two parameters:

**Drift rate.** Set drift proportional to the relative utility,

$$
v \;=\; k\, \bigl(V_S - V_{\text{threat}}\bigr).
$$

Elevated $V_S$ (high perceived control) produces positive drift toward approach.

**Effective bound via urgency suppression.** Under imminence, tonic urgency $u(t)$ collapses the bound (cf. §§3.2, 4.2):

$$
a(t) \;=\; a_0 - u(t).
$$

Anterior-vmPFC-mediated safety **suppresses** $u(t)$, restoring $a(t) \to a_0$ and allowing slower, accuracy-preserving model-based prospection. This is the formal explanation for why safe contexts enable deliberation while unsafe contexts force reflexive reactions.

#### 9.6 Safety in Active Inference: Prior Preferences $C$

In the §7 active-inference formalism, safety is encoded as a high-precision prior preference over observations,

$$
p(o \mid C) \;\propto\; \exp(C(o)),
$$

with $C(o_{\text{safe}}) \gg C(o_{\text{threat}})$. EFE decomposes as

$$
G_\pi \;=\; \underbrace{D_{\mathrm{KL}}\bigl[q(o\mid\pi) \,\|\, p(o\mid C)\bigr]}_{\text{risk}} \;+\; \underbrace{\mathbb E_{q(s\mid\pi)}\!\left[H[p(o\mid s)]\right]}_{\text{ambiguity}} .
$$

- **High threat.** Predicted $q(o\mid\pi)$ diverges from safe $p(o\mid C)$ → risk explodes → agent locks into reactive policies that minimize risk (fleeing, freezing).
- **High safety computed by ant-vmPFC.** $q(o\mid\pi) \approx p(o\mid C)$ → risk small → policy selection now driven by *ambiguity* → agent is freed to explore for epistemic value.

This gives a clean formal meaning to the paper's clinical claim: **safety releases epistemic action.**

#### 9.7 Safety Under Ambiguity

Distinguish *risk* (known $P$) from *ambiguity* (uncertainty over $P$). A mean–variance–ambiguity decomposition of subjective safety:

$$
\boxed{\; V_{S,\text{subj}} \;=\; \mathbb E[V_S] \;-\; \alpha\, \mathrm{Var}(V_S) \;-\; \beta\, A(V_S), \;}
$$

with $\alpha$ risk aversion, $\beta$ ambiguity aversion, and $A(\cdot)$ a second-order dispersion (e.g. entropy of the belief over parameters). In anxiety, $\beta \gg 0$ penalizes $V_{S,\text{subj}}$ even when objective $\mathbb E[V_S]$ is high, producing chronic pre-encounter anxiety and failure to disengage defensive policies.

#### 9.8 Bridge to the Surrounding Papers

- **§8 (LeDoux & Daw).** Tashjian et al. supply the *positive control signal* that LeDoux & Daw's infralimbic-driven "switch" from CeA freezing to BA-NAcc avoidance uses. The anterior vmPFC is the comparator whose output is the switch input.
- **§7 (active inference).** Safety is a prior preference $C$; its computation gates whether EFE is dominated by risk (threat) or ambiguity (exploration).
- **§5 (Fleming).** Perceived control $K$ is a metacognitive self-model quantity; its individual differences are exactly the kind of meta-$d'$-like signal Fleming describes.
- **§6 (Gershman).** Safety-induced release of urgency corresponds to lowered tonic $\rho$, raising the DDM bound and lengthening deliberation.

**Bottom line.** Safety is a computed decision variable, not a shadow of threat. Anterior vmPFC computes it by integrating threat with self-model variables (especially control), and its output tunes DDM drift / bound, gates Pavlovian output, and sets the prior preference $C$ in an active-inference controller. Paper 10 now drills into the complementary system: how the brain *computes the threat itself*.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Introduction (untitled preamble).** Safety estimation is understudied despite being essential for non-defensive behaviour (foraging, mating). Thesis: safety is *not* the linear inverse of danger; it is an independent computation mediating the canonical defensive circuits.
- **Safety as a distinct computation.** Anterior vmPFC as safety hub; posterior vmPFC as anticipated-threat hub; the dichotomy parallels rodent infralimbic/prelimbic opposing control of the amygdala (existing §9 Phase 2.1, 2.4).
- **Box 1 — Psychopathology and development.** PTSD: chronic underestimation of safety, dual (threat + self) distortions, vmPFC hypoactivation + amygdala hyperactivation. Adolescence is a vulnerable window: protracted PFC development ⇒ amygdala dominates immature metacognitive safety computation.
- **Defining safety.** Continuum: complete/contextual safety; extinction-based safety; safety signals (mitigators); prospective safety (future-oriented).
- **Linking safety to the TIC.** Fanselow–Lester Threat Imminence Continuum (safe → pre → post → circa-strike). Safety can occur at *any* stage — coping ability is orthogonal to distance/time.
- **Protection taxonomy.** Phenotypic / niche; social ("many eyes", "Chuck Norris"); manufactured (weapons); tactical (symbolic, predictive). Highly prospective protections deploy earlier along the TIC.
- **Safety is not inverted threat.** Subjective safety drives behaviour; misestimation in either direction is costly (risk compensation vs chronic fear).
- **Safety Decision Model.** Core architecture = interaction of **threat-oriented evaluation** (imminence, value, uncertainty) and **self-oriented evaluation** (policies, experience, control); anterior vmPFC is the integrator (existing §9 Phase 2.1).
- **Threat-oriented evaluation.** (Header.)
  - *Imminence.* Spatiotemporal proximity biased by needs/intensity/protections; increasing imminence shifts control from ant-vmPFC to hardwired PAG / hypothalamus / midcingulate.
  - *Value.* Weight of danger computed across amygdala / striatum / insula / hippocampus / vmPFC; vmPFC down-regulates BLA and couples with striatum under safety.
  - *Uncertainty.* Reducible (estimation) vs irreducible; unexpected uncertainty sharply lowers safety; aMCC guides learning under uncertainty.
- **Self-oriented evaluation.** (Header.)
  - *Policies.* Safe ⇒ long horizons ⇒ model-based; danger ⇒ model-free habits; hippocampus + vmPFC support MB prospection.
  - *Experience.* Bayesian integration of prior + likelihood; hippocampus/vmPFC builds environment model; overgeneralizing past threat = failure of ant-vmPFC threat-safety discrimination.
  - *Control.* Perceived control reverses behavioural inhibition, boosts MB action; underestimated control ⇒ reduced error-related learning (transdiagnostic anxiety marker); ant-vmPFC → limbic inhibition of stress response (existing §9 Phase 2.7 via $K$).
- **Observable safety decision responses.** Continuous integration of bottom-up + top-down signals; outputs modulated by comparison, memory, self-projection, conflict monitoring; PE updates refine the model.
- **Safety neural circuitry.** Adaptive coding by mixed-selectivity neurons in BLA / aMCC / hippocampus / striatum / insula under vmPFC regulation. Under safety, BLA routes to striatum (approach), not CeA (freezing/fleeing) — Tashjian's contribution to the §8 PFCIL arbitration story.
- **Anterior/posterior vmPFC identification.** Anterior = BA 10r = safety hub, part of DMN (metacognition / mentalizing). Posterior = BA 25 / 32PL = threat hub.
- **Decoding meta-analysis.** Neurosynth term decoding: ant-vmPFC loads on Safety Decision Model components; post-vmPFC specifically on "fear".
- **Coactivation meta-analysis.** Ant-vmPFC coactivates with hippocampus, ventral striatum, thalamus, hypothalamus; post-vmPFC with PAG, aMCC, dorsal anterior insula.
- **Concluding remarks.** Safety computation = baseline human cognition (DMN "default"). DMN-related psychopathologies (anxiety) as metacognitive safety deficits.
- **Outstanding questions.** (i) Is ant-vmPFC *necessary* for safety (primate causality)? (ii) Developmental sources of safety learning (peers vs parents). (iii) How does self-oriented evaluation block over-generalization? (iv) How is protection evaluated across types?

### Appendix: Section-by-Section Backbone

**Introduction (untitled).** Safety is understudied; claim: independent computation, not inverse of threat; ant-vmPFC is the hub.

**Safety as a distinct computation.** Brain integrates stimulus-value + perceived-control to estimate survival capability. Dichotomy within vmPFC: anterior = safety, posterior = anticipated threat; parallels rodent IL/PL opposition over amygdala.

**Box 1 — Psychopathology and development.** PTSD: chronic safety under-estimation; dual distortions (threat-side + self-side); circuitry: vmPFC hypo + amygdala hyper + altered hippocampus. Developmentally: child safety learning is amygdala-heavy → adult distributed vmPFC-hippocampus; adolescence vulnerable due to immature PFC.

**Defining safety.** Four constructs: complete/contextual, extinction-based, safety signals (e.g., enclosure), prospective (expectation of future safety + active safety-seeking).

**Linking safety to the threat imminence continuum.** Fanselow–Lester stages (safe → pre → post → circa-strike). Claim: safety can appear at any stage (e.g., weapon during circa-strike); safety perception overrides TIC default behaviours.

**Protection taxonomy.** Deployment depends on how prospective the protection is.
- Phenotypic / niche (post-encounter, circa-strike).
- Social ("many eyes", "Chuck Norris") + manufactured (weapons) — pre- and post-encounter.
- Tactical + symbolic predictions (facial expressions) — safe + pre-encounter.

**Safety is not inverted threat.** Subjective perception drives behaviour. Non-defensive motivation persists under high threat if perceived safety is high. Risks: over-estimation ⇒ risk compensation; under-estimation ⇒ maintained fear.

**Safety Decision Model.** Two interacting evaluative components:
- **Threat-oriented:** imminence, value, uncertainty.
- **Self-oriented:** policies, experience, control.
Integration in anterior vmPFC. Roller-coaster example: self-oriented evaluation dampens threat-oriented defensive response ⇒ enjoyment instead of terror.

**Threat-oriented evaluation.** Header.
- *Imminence.* Subjective, biased by needs/intensity/protection; growing imminence ⇒ ant-vmPFC ceded to PAG, hypothalamus, MCC; circa-strike bypasses vmPFC entirely.
- *Value.* Encoded across amygdala/striatum/insula/hippocampus/vmPFC; vmPFC → BLA down-regulation, striatum approach coupling.
- *Uncertainty.* Reducible (solvable by learning) vs irreducible; unexpected uncertainty sharply lowers safety; aMCC integrates affect + sensation to guide learning; outputs to amygdala/insula/striatum/PAG.

**Self-oriented evaluation.** Header.
- *Policies.* Safe contexts enable long horizons ⇒ MB prospection; dangerous contexts revert to MF habits; hippocampus+vmPFC supply memory / prospection.
- *Experience.* Bayesian priors over survival history; hippocampus+vmPFC build the environment model; overgeneralization of threat to safe contexts = ant-vmPFC discrimination failure.
- *Control.* Perceived control reverses stress-induced behavioural inhibition; underestimated control ⇒ reduced error learning (anxiety); ant-vmPFC → limbic inhibition for resilience / coping calibration.

**Observable safety decision responses.** Bottom-up threat-oriented + top-down self-oriented signals combined; outputs shaped by comparison, memory, self-projection, conflict monitoring; PE/reinforcement update the environment model.

**Safety neural circuitry.** Mixed-selectivity adaptive coding across BLA, aMCC, hippocampus, striatum, insula under vmPFC regulation. Under safety, BLA → ventral striatum (approach), not CeA (freeze/flee).

**Anterior/posterior vmPFC identification.** Ant vmPFC = BA 10r = safety hub and DMN node; post vmPFC = BA 25 / 32PL = threat hub. Anterior specialization aligns with metacognition/mentalizing.

**Decoding meta-analysis.** Neurosynth term decoding: ant vmPFC loads on Safety Decision Model components (metacognition, mentalizing, uncertainty); post vmPFC specifically on "fear".

**Coactivation meta-analysis.** Neurosynth coactivation: ant vmPFC ↔ hippocampus, ventral striatum, thalamus, hypothalamus; post vmPFC ↔ PAG, aMCC, dorsal anterior insula.

**Concluding remarks.** Safety = baseline human cognition (DMN default); DMN-related psychopathologies may stem from metacognitive safety deficits.

**Outstanding questions.** Causality of ant-vmPFC (primate lesions/inactivation); developmental sources of safety learning (peers vs parents); role of self-oriented evaluation in preventing overgeneralization; evaluation mechanisms for different protection types.

---

## 10. Levy & Schiller (2021) — Neural Computations of Threat

*Ifat Levy & Daniela Schiller. Trends in Cognitive Sciences, 2021.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Research on threat has historically been Balkanized — separate literatures on defensive reactions, fear learning, fear memory, and decisions under stress. Levy & Schiller argue these are facets of a single computational problem: *decision-making under uncertainty with survival stakes*. The paper's goal is to provide a unified taxonomy of the **computations** the brain performs across the full threat lifecycle, from first encounter to reconsolidated memory to strategic avoidance.

**Key Findings.**
- **Five stages of threat processing.** (i) Encounter / imminent defense; (ii) threat-association learning (Pavlovian); (iii) post-association learning (extinction, reversal); (iv) memory storage, retrieval, reconsolidation; (v) value-based decision-making under threat.
- **Shared computational primitives across stages.** Expected value, prediction error, volatility-driven learning-rate control, risk, ambiguity, and controllability recur at every stage.
- **Uncertainty is multidimensional and dissociable.** Risk (known probabilities) and ambiguity (unknown probabilities) have independent behavioural attitudes and distinct neural substrates.
- **Gains ≠ losses.** Risk / ambiguity attitudes in the threat (loss) domain are uncorrelated with the gain domain.
- **Psychopathology as dysfunctional computation.** PTSD / anxiety reflect specific failures: inflexible learning rate under volatility, extreme ambiguity aversion, flattened generalization gradients, and salience-instead-of-value coding under chronic stress.

**Main Methodology.** Integrative review synthesizing human behavioural economics, human fMRI, computational modelling (Rescorla–Wagner, Bayesian volatility, latent-cause models, SR), and rodent circuit-level evidence (optogenetics, in-vivo ephys).

**Initial Takeaway.** Threat processing is not a single reflex but a *stack of decisions under uncertainty*. The right level of description is computational — value, risk, ambiguity, imminence, controllability — with each component mapping onto dissociable neural substrates. This establishes an interface with §§1–7 (DDM, active inference, RL) and complements §§8–9 (defensive taxonomy, safety).

### Phase 2: Graduate-Level Deep Dive

> Paper is a conceptual review; equations below are the canonical formalisms it cites (von Neumann–Morgenstern; Rescorla–Wagner; Pearce–Hall; Dayan SR; Gershman latent-cause; Shepard 1987).

#### 10.1 Taxonomy of Threat Variables

| Variable | Meaning | Key parameter |
|---|---|---|
| Value $V_{\text{harm}}$ | Magnitude of potential harm | units of utility |
| Probability / risk | Known $P(\text{harm})$ | variance $\sigma^2$ |
| Ambiguity | Second-order uncertainty over $P$ | dispersion $A$ |
| Imminence | Spatial / temporal proximity | distance $d$ |
| Controllability | Action-vs-omission divergence | $C$ (§10.1.5) |

**10.1.1 Expected threat utility.**

$$
EU(a) \;=\; -\, P(\text{harm}\mid a)\, V_{\text{harm}}.
$$

**10.1.2 Risk-adjusted utility (mean–variance).**

$$
EU_{\text{risk}} \;=\; EU(a) \;-\; \alpha\, \sigma^{2},
$$

with risk aversion $\alpha > 0$.

**10.1.3 Ambiguity-adjusted utility.** Let $A$ index second-order dispersion (e.g. entropy of the prior over $P$):

$$
EU_{\text{amb}} \;=\; EU_{\text{risk}} \;-\; \beta\, A .
$$

$\alpha$ and $\beta$ are empirically uncorrelated across subjects.

**10.1.4 Imminence (hyperbolic discounting).**

$$
V_{\text{subj}}(d) \;=\; \frac{V_0}{1 + k\, d}.
$$

As $d \to 0$, $V_{\text{subj}} \to V_0$, crossing the threshold that switches control from prefrontal evaluation to ventral→dorsal PAG panic circuits (cf. §8.9).

**10.1.5 Controllability.** Instrumental divergence between action and omission:

$$
C \;=\; P(\text{safe}\mid \text{action}) \;-\; P(\text{safe}\mid \text{no action}).
$$

High $C$ enables active avoidance; $C \approx 0$ collapses policy to passive freezing (learned helplessness).

#### 10.2 Pavlovian Threat Learning

Rescorla–Wagner with threat PE:

$$
\delta \;=\; \lambda - V_{\text{CS}}, \qquad \Delta V_{\text{CS}} \;=\; \alpha\, \beta\, \delta .
$$

**Second-order conditioning.** With CS$_1$ already paired with shock, pairing a new CS$_2$ with CS$_1$ uses $V_{\text{CS}_1}$ as teaching signal:

$$
\lambda^{(2)} \;=\; V_{\text{CS}_1}, \qquad \Delta V_{\text{CS}_2} \;=\; \alpha\, \beta\, (V_{\text{CS}_1} - V_{\text{CS}_2}).
$$

This propagates threat values through associative structure without further exposure to the US.

#### 10.3 Adaptive Learning Rates (Pearce–Hall; Bayesian Volatility)

When contingencies are volatile, a fixed $\alpha$ is suboptimal. Pearce–Hall:

$$
\alpha_{t+1} \;=\; \eta\, |\delta_t| \;+\; (1 - \eta)\, \alpha_t,
$$

i.e. associability tracks recent surprise. Bayesian volatility models (Behrens et al.) make this optimal by tracking a hierarchical posterior over the *rate of change* of $P$. Anxiety patients show blunted volatility tracking — $\alpha$ fails to rise in volatile environments and fails to fall in stable ones, producing perseverative threat predictions.

#### 10.4 Model-Based Threat Prediction: Successor Representation

Model-free R-W values cache one-step associations; model-based control requires the *structure* of the threat environment. The successor representation

$$
M(s, s') \;=\; \mathbb E_\pi\!\left[\sum_{t=0}^{\infty} \gamma^{t}\, \mathbb 1[s_t = s'] \,\Big|\, s_0 = s\right]
$$

enables rapid recomputation of threat value whenever the reward vector changes:

$$
V(s) \;=\; \sum_{s'} M(s, s')\, R_{\text{threat}}(s').
$$

A novel aversive outcome at a known state immediately propagates to all states that lead there — one-shot avoidance without relearning. The hippocampus supports $M$; the striatum caches $R$.

#### 10.5 Threat Generalization (Shepard), Reversal, and Latent Causes

**Shepard exponential gradient.** Transfer of threat value from $x_{\text{old}}$ to novel $x_{\text{new}}$:

$$
V(x_{\text{new}}) \;=\; V(x_{\text{old}})\, \exp\!\bigl(-c\, \| x_{\text{new}} - x_{\text{old}} \|^{p}\bigr).
$$

Anxiety → flattened $c$ → overgeneralization.

**Latent-cause extinction.** Extinction is not erasure but Bayesian inference of a new latent cause $Z_{\text{safe}}$:

$$
P(Z \mid \text{data}) \;\propto\; P(\text{data}\mid Z)\, P(Z), \qquad Z \in \{Z_{\text{threat}}, Z_{\text{safe}}, \dots\}.
$$

If extinction-phase data yield a low likelihood under $Z_{\text{threat}}$, the posterior shifts to $Z_{\text{safe}}$ — but $Z_{\text{threat}}$ persists, so context-dependent gating re-selects it (spontaneous recovery, renewal, reinstatement).

#### 10.6 Decisions Under Threat: Combined Utility

Integrating §§10.1–10.5 into a single subjective utility for defensive action $a$:

$$
\boxed{\;
U(a) \;=\; -\, P(\text{harm}\mid a)\, V_{\text{harm}}
\;-\; \alpha\, \sigma^{2}(a)
\;-\; \beta\, A(a)
\;-\; \gamma\, D_{\text{imm}}(d)
\;+\; \eta\, C(a).
\;}
$$

Each term is dissociable behaviourally and neurally; individual-difference vectors $(\alpha, \beta, k, \eta)$ constitute a *computational fingerprint* of anxiety/PTSD.

#### 10.7 Neural Substrates (Dimensional Map)

| Variable | Neural substrate |
|---|---|
| $V_{\text{harm}}$ | **BLA**; converges on **vmPFC** for subjective value |
| Reactive output | **CeA → PAG** (ventral / dorsal) |
| Imminence | **PAG gradient**; circa-strike → dorsal PAG / dorsal raphe |
| Risk ($\sigma^2$) | Posterior parietal cortex; anterior insula |
| Ambiguity ($A$) | **vlPFC**, anterior insula |
| Uncertain / sustained threat | **BNST** |
| Interoceptive threat | **Insula** |
| Controllability | **mPFC (PL/IL), dorsal striatum** |
| Volatility-driven learning rate | **dACC**, pulvinar |

This is the precise neural correspondence Tashjian et al. (§9) rely on for their anterior-vs-posterior vmPFC dichotomy and that LeDoux & Daw (§8) rely on for the CeA/BNST split.

#### 10.8 Salience-vs-Value Coding Under Chronic Threat

Normally, ventral striatum and lateral habenula encode monotonic value — positive for good, negative for bad. Under chronic stress, recordings show a shift to a **U-shaped salience code**, tracking $|V|$ instead of $V$. Formally,

$$
r_{\text{neuron}} \;\propto\; |V(s)|, \qquad \text{not } r_{\text{neuron}} \propto V(s),
$$

which explains how trauma degrades precise value-based policies into generalized hyper-reactivity. This bears directly on distributional-RL (§6.3): chronic threat may *bias the expectile distribution* $\tau$ of dopaminergic neurons rather than merely shift the scalar mean.

#### 10.9 Integration With the Rest of the Review

- **§§1–3 (SDT, DDM, CPP).** Threat detection is a perceptual decision: the accumulator drift is driven by $-\partial U/\partial a$ from §10.6; imminence controls bound collapse (§§3.2, 4.2).
- **§4 (Cisek).** Predatory imminence continuum *is* the urgency signal; affordance competition among flee / freeze / fight resolves in the sensorimotor circuits Cisek describes.
- **§5 (Fleming).** Confidence in threat ($P(\text{threat}\mid \hat s)$) is the metacognitive gate that determines whether CeA (committed) or BNST (uncertain) is engaged.
- **§6 (Gershman).** All learning-rate, volatility, and distributional phenomena are instantiated by dopamine; chronic threat shifts the expectile distribution $\tau$.
- **§7 (Smith–Friston–Whyte).** The integrated utility of §10.6 *is* a negative EFE with risk + ambiguity already split; Levy & Schiller's five stages are nested timescales of the same variational inference.
- **§8 (LeDoux & Daw).** The circuit hierarchy is identical; Levy & Schiller add the computational variables that *ride* on it.
- **§9 (Tashjian et al.).** Safety is the complement — same variables, sign-flipped — computed in anterior vmPFC and used as a gate on the threat-system outputs described here.

**Bottom line.** Levy & Schiller hand the field a *dimensional* computational framework: threat is a vector $(V_{\text{harm}}, \sigma^2, A, d, C)$ plus adaptive learning-rate dynamics, mapped onto dissociable neural substrates. The papers above (§§1–9) now compose cleanly: perceptual evidence accumulation (§§1–3) feeds urgency and affordance competition (§§4), metacognitive confidence (§5) gates amygdala vs BNST, dopamine (§6) tunes learning rates and distributional codes, active inference (§7) provides the variational umbrella, defensive RL (§8) describes the algorithmic levels, and safety computation (§9) supplies the positive signal that releases the whole system into non-defensive behaviour.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Neural computations of threat (Intro).** Threat = organism/object/situation likely to cause harm. Five-stage framework: imminent experience → threat association → post-association learning → memory storage/update → decision under threat. Common computational primitives across stages: value, PE, adaptive learning rate, uncertainty. PTSD/anxiety = disorders of prediction (existing §10 Phase 2).
- **The stages of threat experience.** (Header.)
- **Experience of imminent threat.** Fanselow threat-imminence continuum. Distal: PFC, model-based planning. Circa-strike: PAG, model-free reflexes. PTSD/anxiety: pre-encounter behaviours (hypervigilance, rumination) in inappropriate contexts.
- **Box 1 — Neural mechanisms of threat detection and modification.** Three functional units: *detection* (sensory), *integration* (LA + hypothalamus), *output* (PAG). LA plasticity encodes aversive PE. CeA = output. Prelimbic PFC (~human dACC) prolongs; infralimbic PFC (~human vmPFC) and intercalated cells mediate extinction.
- **Box 2 — PTSD and anxiety.** DSM-5 Trauma and Stressor-related Disorders. Neural: amygdala hyperreactivity, vmPFC hypoactivation, impaired hippocampal context. Uncertain-threat = BNST modulates information between amygdala and ventral striatum.
- **Backpropagation of threat.** Rescorla–Wagner PE; hybrid RW + Pearce–Hall for adaptive associability (existing §10 Phase 2.2–2.3). Combat-veteran data: higher subject-specific PE weighting ↔ worse PTSD; better striatal associability tracking ↔ better symptoms. Instrumental-avoidance circuit competition: LA-BA-NAcc (active avoidance) vs LA-CeA-PAG (freezing); prelimbic + infralimbic PFC arbitrate. Bayesian volatility: subjective volatility tracks pupil, ACC, stress response.
- **Box 3 — Types of uncertainty.** *Expected* (risk, irreducible variance in known distributions), *estimation* (ambiguity, reducible by experience), *unexpected* (volatility, triggers learning-rate increase).
- **Flexible threat associations.** Extinction: RW sees it as unlearning; Pearce-Hall as new learning. **Latent-cause model** (Gershman-style) reconciles them: small PE → update old weight; large PE → infer new latent cause. OFC + dentate gyrus represent latent causes; VTA carries PE; dentate neurogenesis supports new-cause inference (existing §10 Phase 2.5).
- **Life cycle of threat memory.** Reactivation destabilizes engram → reconsolidation restabilizes (needs protein synthesis). Labile window enables reconsolidation-update (post-retrieval extinction, counterconditioning). Latent-cause model: brief reminder → reassign info to original cause (true update); prolonged reminder → new latent cause (standard temporary extinction). Basis for clinical coherence therapy.
- **Decision making under threat.** Subjective value encoded across vmPFC / ventral striatum / OFC / PCC / PPC (monotonic). dACC, anterior insula, amygdala encode *salience* (U-shape). BLA integrates reward + punishment. Post-trauma: shift from value to salience coding in ventral striatum and lateral habenula. Risk/ambiguity/loss-aversion attitudes dissociable; PTSD veterans = elevated ambiguity aversion on losses; intolerance of uncertainty = transdiagnostic factor across PTSD/OCD/GAD/social-anxiety.
- **Ambiguity vs risk attitudes.** Four independent characteristics (risk/ambiguity × gain/loss).
- **Concluding remarks.** Learning / memory / decision-making interconnected; two core computational problems across stages: long-term accumulated-outcome prediction + uncertainty tracking. Track specific computations across tasks to decide domain-general vs domain-specific psychiatric disruptions.
- **Outstanding questions.** (i) Longevity of PTSD; (ii) MB vs MF balance in post-trauma symptoms; (iii) how subjective-value comparison generates choice; (iv) value vs salience relationship; (v) shared uncertainty mechanisms across domains; (vi) preexisting vulnerability vs acquired consequence.
- **Box 5 — From algorithms to feelings.** First-order threat computations (value, uncertainty, volatility in survival circuits) vs higher-order conscious fear (integrating memory, schemas, mental models). Separating the two enables targeted treatments. Mood modeled as running average of recent PEs — biases future prediction and choice recursively.

### Appendix: Section-by-Section Backbone

**Neural Computations of Threat: Learning, Memory, and Decision Making (Intro).** Threat = harm-likely entity. Five-stage framework (imminent, associative, post-association, storage/update, decision). Shared primitives: value, PE, adaptive learning rate, uncertainty. PTSD/anxiety reframed as prediction-algorithm disorders.

**The stages of threat experience.** (Header.)

**The experience of imminent threat.** TIC: pre-encounter = PFC model-based planning; circa-strike = PAG model-free reflex. PTSD/anxiety: pre-encounter behaviours (hypervigilance, rumination) in inappropriate contexts ⇒ spatiotemporal-threat computation + MB planning are stress-vulnerable.

**Box 1 — Neural mechanisms of threat detection and modification.** Circuit: sensory detection → LA integration (+ hypothalamus) → CeA + PAG output. LA long-term potentiation after CS-US pairing. LA encodes aversive PE (strong for unexpected threats, reduced when predicted; amygdala-PAG feedback). Prelimbic PFC (human dACC) prolongs response; infralimbic PFC (human vmPFC) + intercalated cells drive extinction-inhibition.

**Box 2 — PTSD and anxiety.** DSM-5 places PTSD under Trauma/Stressor disorders. Neural: amygdala hyperreactivity, vmPFC hypoactivation, impaired hippocampal context. Uncertain threat engages BNST linking amygdala ↔ ventral striatum.

**Backpropagation of threat.** Associative learning via PE (expected − received). RW basic; hybrid RW + Pearce-Hall adjusts associability dynamically. In combat veterans: higher PE weighting ↔ worse PTSD symptoms; better striatal associability tracking ↔ better symptoms. Instrumental avoidance: LA-BA-NAcc (active) vs LA-CeA-PAG (freezing); prelimbic + infralimbic PFC arbitrate. Bayesian volatility models: subjective unexpected-uncertainty tracks pupil, ACC, and stress.

**Box 3 — Types of uncertainty.** Expected = risk = irreducible variance under known distribution, should not drive new learning. Estimation = ambiguity = reducible by experience, signals how much learning is required. Unexpected = volatility = change in distribution ⇒ raise learning rate.

**Flexible threat associations (extinction).** RW: unlearning. Pearce-Hall: new learning. Latent-cause model reconciles: small PE → update old weight; large PE → infer new latent cause. OFC + dentate gyrus represent latent causes; VTA signals PE; dentate gyrus neurogenesis supports new-cause generation.

**Life cycle of threat memory.** Engram oscillates between stable and labile. Reactivation (memory reminder) destabilizes; reconsolidation (protein-synthesis-dependent) restabilizes. Labile window allows update via post-retrieval extinction or counterconditioning. Latent-cause model: brief reminder → same cause, true update; prolonged → new latent cause, standard temporary extinction. Clinical basis for coherence therapy.

**Decision making under threat.** Monotonic subjective-value code: vmPFC, ventral striatum, OFC, PCC, PPC. Salience code (U-shape, |V|): dACC, anterior insula, amygdala. BLA integrates reward + punishment. Post-trauma: value-code → salience-code shift in ventral striatum and lateral habenula (explains generalized hyper-reactivity). Risk / ambiguity / loss-aversion dissociable. Combat-veteran PTSD: enhanced ambiguity aversion on losses. Intolerance of uncertainty = transdiagnostic factor for PTSD/OCD/GAD/social anxiety.

**Ambiguity vs risk attitudes.** Four independent decision characteristics: (risk vs ambiguity) × (gain vs loss).

**Concluding remarks.** Learning / memory / decision-making are a single interleaved process. Two core computational problems across stages: long-term predictive accumulation + uncertainty tracking. Strategy: track computations across tasks + domains to decide algorithmic-general vs domain-specific psychiatric disruption.

**Outstanding questions.** Longevity of PTSD; MB/MF contributions to post-trauma symptoms; mechanism of value comparison for choice; value–salience relationship (parallel vs derived); whether uncertainty mechanisms are shared across learning/memory/decision and across threats/rewards; vulnerability vs consequence of stress.

**Box 5 — From algorithms to feelings.** First-order representation = threat computations in survival circuits (value, uncertainty, volatility). Higher-order representation = conscious fear (integrates memory, schemas, mental models). Dissociation enables targeted treatments. Mood = running average of recent PEs, biases future predictions recursively.

---

## Cross-Paper Synthesis (updated)

The ten papers together span a full computational theory of decision-making from perceptual inference to survival control:

1. **Inference core (§§1–3).** SDT, SPRT, DDM; LIP accumulator; CPP / mu-beta; neurally-informed hierarchical-Bayesian DDM.
2. **Embodiment (§4).** Affordance competition + urgency; the DDM is a limit case of parallel sensorimotor competition in evolutionarily layered control loops.
3. **Metacognition (§5).** Propositional confidence, meta-$d'$, two-stage DDM; dedicated vmPFC / pMFC / frontopolar substrates.
4. **Neuromodulation (§6).** Dopamine as a family of prediction errors — scalar, distributional, belief-state, successor, multi-timescale, average-reward — tuning drift, bound, and vigor.
5. **Variational umbrella (§7).** Active inference under a POMDP generative model; VFE for inference, EFE for policy; precision $\gamma$ (dopamine), $\alpha$ (NE). Contains §§1–6 as limits.
6. **Defensive taxonomy (§8).** Reflex → Pavlovian habit → MF instrumental → MB goal-directed → deliberation; arbitration via uncertainty; POMDP for sustained/uncertain threat (BNST vs CeA).
7. **Safety (§9).** Distinct positive computation in anterior vmPFC, integrating threat with perceived control; gates DDM drift/bound, sets active-inference prior $C$.
8. **Threat computations (§10).** Multidimensional threat vector $(V_{\text{harm}}, \sigma^2, A, d, C)$; five-stage lifecycle; dimensional neural map; psychopathology as specific computational deficits.

A single formal object unifies them all: an agent performing **variational inference over a POMDP** with a generative model that contains (i) a DDM-like perceptual-evidence sub-module, (ii) Pavlovian and instrumental RL valuation systems with adaptive learning rates, (iii) safety and threat variables with distinct anterior/posterior vmPFC readouts, and (iv) a precision-weighted policy selector driven by dopaminergic/noradrenergic control. This is the contemporary integrated picture of perceptual decision-making and survival decision-making as a single computational framework.

---

## 11. Wiech (2016) — Deconstructing the Sensation of Pain

*Katja Wiech. Science, 2016. "Deconstructing the sensation of pain: The influence of cognitive processes on pain perception."*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** The classical specificity view treats pain as a straight read-out of nociceptive input ascending from the periphery. This is empirically untenable: placebo analgesia, distraction, catastrophizing, and chronic pain without tissue damage all show that pain is *constructed*. Wiech asks: what are the cognitive computations that produce the subjective experience of pain, and how do they interact with ascending nociceptive signals?

**Key Findings.**
- **Pain is perceptual inference.** The experience is the posterior of a Bayesian combination of bottom-up nociceptive evidence with top-down expectations and context.
- **Precision-weighted predictive coding.** Ascending prediction errors are multiplied by sensory precision (attentional gain); descending predictions are weighted by prior precision (expectation certainty). Attention, expectation, and mood all act by shifting these *precisions*, not by changing the stimulus.
- **Valuation > sensation for self-regulation.** Cognitive self-regulation does not primarily modulate S1/S2 activity (the Neurologic Pain Signature is preserved); it acts through mesolimbic valuation circuits — **vmPFC** and **NAcc**.
- **Expectation biases the decision, not just the signal.** DDM fits to pain psychophysics show expectation shifts the accumulator **starting point $z$** (a bias) more than the drift rate $v$ (sensitivity).
- **Chronic pain = aberrant priors.** Excessively precise negative priors ($\sigma^2_{\text{prior}} \to 0$), coupled with failure to update, produce pain experience that is *decoupled from peripheral input*.

**Main Methodology.** Review integrating human fMRI, psychophysics, multivariate decoding (Neurologic Pain Signature), and computational modeling (Bayesian integration, predictive coding, DDM).

**Initial Takeaway.** Pain is an *inferential construct* — the output of the same Bayesian/predictive-coding machinery that §§1–7 describe for perception and decision, operating on interoceptive channels. Placebo, distraction, reappraisal, and chronic pain are all formal manipulations of priors, likelihoods, and precisions — giving a unified framework for cognitive pain modulation.

### Phase 2: Graduate-Level Deep Dive

> The review is conceptual; equations below follow the standard Bayesian / predictive-coding literature Wiech cites (Büchel, Geuter, Friston, Seymour).

#### 11.1 Pain as Bayesian Perceptual Inference

Let $x$ be the latent bodily/pain state and $o$ the noisy nociceptive observation. Bayes' rule:

$$
P(x \mid o) \;=\; \frac{P(o \mid x)\, P(x)}{P(o)}.
$$

**Gaussian closed form.** With $P(x) \sim \mathcal N(\mu_{\text{prior}}, \sigma_{\text{prior}}^{2})$ and $P(o\mid x) \sim \mathcal N(x, \sigma_{\text{obs}}^{2})$, the posterior mean (subjective pain intensity) is the *precision-weighted* combination

$$
\boxed{\;
\mu_{\text{post}} \;=\; \frac{\sigma_{\text{obs}}^{2}}{\sigma_{\text{obs}}^{2} + \sigma_{\text{prior}}^{2}}\, \mu_{\text{prior}} \;+\; \frac{\sigma_{\text{prior}}^{2}}{\sigma_{\text{obs}}^{2} + \sigma_{\text{prior}}^{2}}\, o,
\;}
$$

with posterior precision $\Pi_{\text{post}} = \Pi_{\text{prior}} + \Pi_{\text{obs}}$ (where $\Pi = 1/\sigma^2$).

#### 11.2 Placebo and Nocebo as Prior-Manipulation

- **Placebo.** Belief in an analgesic shifts $\mu_{\text{prior}}$ downward → $\mu_{\text{post}} < o$ → analgesia.
- **Nocebo.** Anxiety / negative expectation shifts $\mu_{\text{prior}}$ upward → hyperalgesia.

**Infinitely precise prior (limit case).** If $\sigma^2_{\text{prior}} \to 0$ (absolute certainty):

$$
\mu_{\text{post}} \;\to\; \mu_{\text{prior}} ,
$$

and the brain *ignores* the bottom-up signal. This captures hallucinated pain from expectation alone, and — in chronic pain — the decoupling of experience from periphery.

#### 11.3 Predictive Coding and Precision-Weighted Prediction Errors

Define the nociceptive prediction error

$$
\delta \;=\; o - \mu_{\text{prior}}.
$$

Equivalent posterior update (Kalman form):

$$
\mu_{\text{post}} \;=\; \mu_{\text{prior}} \;+\; \underbrace{\frac{\Pi_{\text{obs}}}{\Pi_{\text{obs}} + \Pi_{\text{prior}}}}_{\text{Kalman gain } K}\, \delta.
$$

**Attention = gain on $\Pi_{\text{obs}}$.** Attending *to* pain raises $\Pi_{\text{obs}}$, $K \to 1$, $\mu_{\text{post}} \to o$ — amplification. Distraction lowers $\Pi_{\text{obs}}$, $K \to 0$, $\mu_{\text{post}} \to \mu_{\text{prior}}$ — descending inhibition dominates. This is the formal version of the PAG-mediated analgesia gate. It is the interoceptive analogue of the meta-$d'$ precision read-out in §5.

#### 11.4 Signal-Detection & DDM Mapping

For binary pain vs. no-pain discrimination, the DDM accumulator

$$
dZ \;=\; v\, dt + \sigma\, dW_t, \qquad Z(0) = z.
$$

Wiech highlights two dissociable routes by which expectation alters behaviour:

| Mechanism | Formal change | Empirical signature |
|---|---|---|
| **Sensory gain** | $v \to v + \Delta v$ (drift) / $d' \to d' + \Delta d'$ | Symmetric RT changes; sensitivity shift |
| **Decisional bias** | $z \to z + \Delta z$ (starting point) | *Faster* expectation-congruent errors |

Human pain-psychophysics studies fitted with HDDM predominantly find *$z$-shifts*, not $v$-shifts — expectation biases the *decision*, not the early sensory code. This is exactly the O'Connell–Kelly neurally-informed DDM logic of §3 applied to nociception.

#### 11.5 Cognitive Reappraisal as Generative-Model Change

Reappraisal switches the conditioning context of the generative model: $P(x) \to P(x \mid \text{context} = \text{"healing procedure"})$. The vmPFC–NAcc system evaluates the new context and rewrites the prior over $x$. This is *not* sensory gating — it is model-structural change, exactly the substitution of a new generative model that §7 active inference formalizes. This explains why self-regulation leaves S1/S2 and NPS intact but recruits mesolimbic valuation circuits.

#### 11.6 Neural Circuit Mapping

| Computational variable | Neural substrate |
|---|---|
| Likelihood $P(o\mid x)$ | S1, S2, posterior insula, lateral thalamus |
| Top-down prior $P(x)$ | DLPFC, rostral ACC |
| Descending precision / analgesia gate | ACC → PAG → spinal dorsal horn |
| Prediction error $\delta$ and valuation | vmPFC, NAcc (mesolimbic) |
| Attentional precision $\Pi_{\text{obs}}$ | Salience network: ant. insula, mid-cingulate |
| DDM starting point $z$ (expectation bias) | Prefrontal → striatal pre-stimulus baseline |

#### 11.7 Pain Catastrophizing and Active-Inference Trap

In §7 active-inference terms, catastrophizing = **excessive prior precision on tissue-damage states** plus an inability to update the generative model. EFE:

$$
G_\pi \;=\; D_{\mathrm{KL}}\!\left[q(o\mid\pi) \,\|\, p(o\mid C)\right] \;+\; \mathbb E_{q(s\mid\pi)}\!\left[H[p(o\mid s)]\right].
$$

If $p(o\mid C)$ heavily penalizes any nociceptive observation, all policies look catastrophically *risky*. The agent adopts hyper-avoidance policies that *restrict sensory sampling*, preventing corrective prediction errors. The belief "I am in pain and in danger" becomes a self-confirming attractor — a **dark-room problem** on interoception. This gives a precise formalization of the chronic-pain trap: a local minimum of EFE in which avoidance suppresses the only signal that could update the prior.

#### 11.8 Bridge to the Surrounding Review

- **§§1–3 (DDM/CPP).** Pain is an interoceptive perceptual decision; expectation acts on $z$, attention on $v$.
- **§5 (Fleming).** Meta-$d'$ of *interoceptive* pain discrimination is the same precision read-out; catastrophizing maps onto miscalibration of confidence.
- **§7 (Smith–Friston–Whyte).** Placebo, reappraisal, catastrophizing all formalize as generative-model / precision manipulations in the same POMDP.
- **§§8–10 (threat / safety).** Nociception is the prototypical aversive signal feeding the threat-value $V_{\text{harm}}$ and the anterior-vmPFC safety comparator; pain and threat share the vmPFC–amygdala–PAG control loop.

**Bottom line.** Pain is not a sensation that happens *to* the perceiver; it is an inference *performed by* the perceiver. All "psychological" modulations of pain (placebo, attention, reappraisal, catastrophizing) reduce to precise manipulations of priors, likelihoods, and precisions in the same Bayesian/predictive-coding machinery that drives perceptual decision-making, and they run on the same vmPFC / ACC / insula / PAG / mesolimbic circuits that §§8–10 describe for threat and safety. Papers 12 and 13 now extend this static inferential picture into *dynamic control theory* (Seymour et al.) and *forward/reverse engineering* for neuro-engineering (Mahajan & Seymour).

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Introduction (untitled).** Distraction alters dental-drill pain ⇒ pain is *not* a hardwired readout. Goal: synthesize cognitive-modulation findings under modern computational (predictive-coding / perception-as-inference) concepts, drawing from healthy-participant experiments.
- **The search for a "signature of pain in the brain."** NPS (Neurologic Pain Signature) from machine-learning fMRI decoding: thalamus + S1/S2 + anterior insula + ACC; ~90% specificity for physical vs social pain. Cautions about clinical translation. The traditional sensory-discriminative / cognitive-affective dichotomy is an oversimplification.
- **Descending pain control system.** DLPFC + rostral ACC + PAG drive top-down modulation down to the spinal dorsal horn; DLPFC is pivotal (TMS abolishes placebo); opioids primary, cannabinoids + dopamine contributory (existing §11 Phase 2.6).
- **Frontostriatal system (vmPFC–NAcc).** Cognitive self-regulation *bypasses* the NPS; modulates mesolimbic valuation connectivity instead ⇒ NPS isn't a universal signature. Frontostriatal = "common currency" integrating sensory / cognitive / affective pain; longitudinal predictor of acute → chronic transition (existing §11 Phase 2.5).
- **Attention and spontaneous brain activity.** Three interacting networks: **salience** (aMCC, ant insula, TPJ, DLPFC) engages when attention moves toward pain; **DMN** (mPFC, PCC, precuneus) engages when attention drifts away; **descending pain control** (DLPFC + rACC + PAG). Intrinsic attention to pain depends on DMN↔PAG structural/functional connectivity; altered in chronic pain.
- **The need for a comprehensive unified framework.** Existing studies don't explain initiation / maintenance / integration of cognitive pain modulation or its balance with survival goals.
- **The construction of a pain experience: "Perception as inference."** Predictive coding: perception biases in favor of expectation and down-weights incongruent input. *Fig. 1* uses the **Diffusion Decision Model**: prior info can (i) change sensory processing via drift rate $\Delta v$ or (ii) bias decision via starting point $\Delta Z$. Experimentally (80/20, 20/80, 50/50 cueing), expectation predominantly shifts **starting point $Z$**, not drift; $v$ changes only when strong high-pain expectation is violated by a low-intensity stimulus (existing §11 Phase 2.4).
- **Learning and updating internal models.** When expectation and input diverge, a **prediction error** message updates the model (Fig. 2). Trade-off: premature updating ⇒ volatile; delayed updating ⇒ costly perseveration. Chronic pain = delayed updating / rigid priors. Clinical caveat: premature *downward* correction of placebo expectations on initial insufficient relief causes patient dropout.
- **Conclusions & outlook.** Predictive coding unifies cognitive-modulation phenomena; the descending system is part of a recurrent multi-level PE-exchanging network, not strictly top-down. DLPFC orchestrates evidence accumulation + perceptual decision. Clinical implication: expectations shaped by practitioner communication directly determine outcome — pain really *is* "in the head" because the brain constructs the experience; that is what makes it treatable.

### Appendix: Section-by-Section Backbone

**Introduction (untitled).** Cognitive processes (distraction, anticipation, reappraisal, perceived control, placebo) modulate pain perception. Review aim: integrate cognitive-modulation findings with predictive-coding / perception-as-inference.

**The search for a "signature of pain in the brain."** Cognitive modulators: attention, anticipation, catastrophizing, reappraisal, control; placebo most studied. Traditional sensory-discriminative (lateral thalamus, S1/S2) / cognitive-affective (ant insula, ACC) split too simple. ML-based Neurologic Pain Signature (NPS) comprising thalamus, S1/S2, ant insula, ACC — ~90% specificity for physical vs social pain; clinical translation cautionary.

**The descending pain control system.** Network = DLPFC + rostral ACC + PAG; modulates down to spinal dorsal horn. DLPFC pivotal — TMS of DLPFC abolishes placebo analgesia. Opioids primary mediator; cannabinoids + dopamine contributing.

**The frontostriatal system.** Heat-intensity modulation is captured by NPS; cognitive self-regulation is *not* — it modulates vmPFC–NAcc connectivity instead. Therefore NPS is not a universal pain code. Frontostriatal system = common-currency integrator of sensory/cognitive/affective pain aspects; longitudinal predictor of acute → chronic transition.

**Attention and the influence of spontaneous brain activity.** Three networks:
- Salience (aMCC, ant insula, TPJ, DLPFC) — engaged when attention focuses on pain.
- DMN (mPFC, PCC, precuneus, lateral posterior lobe, medial temporal lobe) — engaged when attention drifts away.
- Descending pain control (DLPFC + rACC + PAG).
Individual intrinsic attention to pain = structural/functional DMN↔PAG coupling; altered in chronic pain.

**The need for a comprehensive unified framework.** Existing literature lacks: what triggers/stops cognitive modulation; how cognitive factors integrate into the construction of pain; how short-term modulation balances long-term survival goals.

**The construction of a pain experience: "Perception as inference."** Predictive coding / perception as inference: expectations bias perception toward prior; incongruent input is down-weighted. **Fig. 1 — Influence of expectations on pain:** DDM framework. Prior can change (i) drift rate $\Delta v$ (altered sensory processing) or (ii) starting point $\Delta Z$ (altered perceptual decision). Empirically with 80/20, 20/80, 50/50 cued-probability paradigm: expectation robustly shifts **$Z$**; $v$ changes only with high-prior + low-actual mismatches. Take-away: cognition influences *higher-order decision translation*, not early sensory gating.

**Learning and updating internal models about pain.** **Fig. 2 — PE processing and learning in the context of pain:** cue/input → expectation → compared to nociceptive input → PE if divergent → learning rule updates expectation. Trade-off: premature update ⇒ volatile model consuming attentional resources; delayed update ⇒ costly perseveration of outdated model. Chronic pain shows the delayed-update failure mode. Clinical caveat on placebo: premature downward correction of positive expectation after insufficient initial relief causes treatment dropout and poor long-term outcomes.

**Conclusions and outlook.** Predictive coding provides a unified framework for cognitive pain modulation. Descending system is part of a larger recurrent multi-level PE network, not strictly top-down. DLPFC orchestrates evidence accumulation. Clinical implication: patients' pain is "all in their head" in the precise sense that the brain *constructs* it via cognitive models — so modifying those models (through communication, training, and context) is the lever for treatment.

---

## 12. Seymour, Crook & Chen (2023) — Post-Injury Pain and Behaviour: A Control Theory Perspective

*Ben Seymour, Robyn J. Crook, Zhe Sage Chen. Nature Reviews Neuroscience, 2023.*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Classic pain neuroscience treats pain as an *alarm* — a sensation triggered by nociceptive input, useful for detecting damage and driving withdrawal. That view explains acute nociception but not *post-injury pain*: the prolonged aching, hypersensitivity, guarding, and fatigue that follow actual tissue injury. Seymour, Crook & Chen ask: what is the *biological function* of this persistent state, and why does it sometimes fail to resolve into chronic pain?

**Key Findings.**
- **Post-injury pain is a recuperation controller.** Its purpose is not to signal damage but to *regulate protective behaviour* during healing — suppressing movement, promoting rest, reallocating energy — trading short-term utility for long-term tissue repair.
- **Pain as closed-loop feedback control.** Body tissue is the *plant*, nociceptors are the *sensor*, the CNS is the *controller*. The brain infers a latent injury state $\hat x(t)$ and issues optimal control signals $u^*(t) = -K\hat x(t)$ (guarding, descending modulation, autonomic shifts).
- **Information restriction is the chronic-pain vulnerability.** Optimal guarding suppresses movement → no movement-evoked sensory feedback → prediction error vanishes *not because tissue is healed* but because the system is starved of data → $\hat x(t)$ is stuck at an injured value → chronic pain.
- **Comparative evidence.** Cephalopods (octopus, cuttlefish, squid) show persistent wound-directed guarding for days, identical in logic to mammalian post-injury pain — this is an **ancient, conserved control strategy**, not a mammalian peculiarity.
- **Acute vs injury-state pain are dissociable.** Acute nociception = fast feedforward reflex to minimise damage. Post-injury pain = slow, hierarchical meta-control that rewrites cost weights, reward sensitivity, and risk tolerance to protect long-term plant integrity.

**Main Methodology.** Theoretical integration of (i) evolutionary ethology and cross-species comparative biology, (ii) optimal control theory (LQR/LQG), (iii) Bayesian state estimation (Kalman filtering), and (iv) POMDP / active inference formulations.

**Initial Takeaway.** Pain is a *control problem*, not a sensation problem. Seen through this lens, chronic pain is a **control failure** — specifically, a failure of state estimation due to self-inflicted information restriction. This reframes treatment: the CNS must be *given information* (via movement, therapy, neuro-stimulation) that the tissue has healed, so the controller can release its guarding policy.

### Phase 2: Graduate-Level Deep Dive

> The review states the LTI state-space formulation and POMDP / active-inference framing. Derivations of LQG cost, Kalman filter, and Riccati feedback gain below are reconstructed from standard references the authors rely on.

#### 12.1 Plant–Sensor–Controller LTI Model

Let $x(t) \in \mathbb R^{n}$ be the latent injury/healing state vector (e.g. inflammation, mechanical integrity, neural sensitization), $u(t) \in \mathbb R^{m}$ the control input (guard, rest, descending analgesia), $y(t) \in \mathbb R^{N}$ the nociceptive/afferent observation.

$$
\dot x(t) \;=\; A\, x(t) \;+\; B\, u(t) \;+\; w(t), \qquad w \sim \mathcal N(0, W) ,
$$

$$
y(t) \;=\; C\, x(t) \;+\; D\, u(t) \;+\; v(t), \qquad v \sim \mathcal N(0, V) .
$$

$A$ captures natural healing dynamics (stable matrix $\Re(\lambda_i(A)) < 0$ ⇒ spontaneous recovery); $B$ captures how actions modulate healing (rest accelerates, movement decelerates); $C$ the nociceptor transduction; $D$ efference-copy contribution.

#### 12.2 Kalman–Bucy State Estimation Under Partial Observability

Because $x$ is hidden, the brain maintains posterior $\hat x(t)$:

$$
\dot{\hat x}(t) \;=\; A\hat x(t) \;+\; B u(t) \;+\; L\bigl(y(t) - \hat y(t)\bigr), \qquad \hat y(t) = C\hat x(t) + D u(t).
$$

The Kalman gain

$$
L \;=\; \Sigma\, C^{\top} V^{-1}
$$

is the precision-weighted responsiveness of the estimator to the *innovation* $y - \hat y$. $\Sigma$ is the steady-state error covariance, solving the filtering Riccati equation

$$
A\Sigma + \Sigma A^{\top} - \Sigma C^{\top} V^{-1} C \Sigma + W \;=\; 0.
$$

**Physiological correspondence.** $L$ increases with nociceptor sensitivity (central sensitization) → over-weights incoming signals → prediction errors are amplified. This is the formal substrate of hyperalgesia / allodynia.

#### 12.3 LQG Optimal Control for Recuperation

Objective: drive $x \to 0$ (healed) with minimum behavioural cost:

$$
\boxed{\; J \;=\; \mathbb E\!\left[\int_{0}^{\infty}\!\bigl(x^{\top}\! Q\, x \;+\; u^{\top}\! R\, u\bigr)\, dt\right]. \;}
$$

- $Q \succeq 0$: biological cost of being injured (metabolic drain, immune cost, fitness loss).
- $R \succ 0$: opportunity cost of protection (foraging foregone, mating foregone, energy expenditure of guarding posture).

**Separation principle.** Optimal control uses the Kalman estimate:

$$
u^{*}(t) \;=\; -K\, \hat x(t),
$$

with $K = R^{-1} B^{\top} P$ and $P$ solving the control Riccati equation

$$
A^{\top} P + P A - P B R^{-1} B^{\top} P + Q \;=\; 0.
$$

Interpretation: when $\|\hat x\|$ is large, severe guarding; as $\hat x \to 0$, $u^{*} \to 0$ and normal foraging resumes.

#### 12.4 Discrete POMDP Formulation

For longer-horizon behavioural choice:

- **States $S$:** healing stages — *acute damage, inflammation, proliferation, remodeling, healed*.
- **Actions $A$:** *rest, guard, move, forage*.
- **Reward:** $R(s, a) \;=\; -\,\mathrm{Pain}(s) \;-\; \mathrm{Energy}(a) \;+\; \mathrm{HealingRate}(s, a)$.
- **Observations $o$:** nociceptive afferents, proprioception, autonomic signals.
- **Belief $b_t(s)$:** Bayes filter over healing stages.

Optimal Bellman equation over beliefs:

$$
V^{*}(b) \;=\; \max_{a}\!\left[\sum_s b(s)\, R(s,a) + \gamma \sum_o P(o\mid b, a)\, V^{*}(b')\right].
$$

#### 12.5 Chronic Pain as Control Failure

Three failure modes in this framework, all observed clinically:

1. **Information restriction.** Optimal guarding zeroes out movement-evoked $y(t)$; innovation $y - \hat y \to 0$ not because $x \to 0$ but because $y$ carries no new information. $\hat x$ stays stuck.
2. **Over-precise prior on injury.** In Bayesian terms, $\Sigma$ (posterior uncertainty) collapses onto an injured mean: the controller is unwilling to entertain the "healed" hypothesis.
3. **Sensitization inflating $C$.** Central sensitization multiplies nociceptor gain; innovations are amplified well beyond actual damage, and $\hat x$ is driven upward by noise.

These three feed each other: guarding → less data → posterior narrows → more guarding. A self-sustaining positive-feedback trap — the **dark-room problem** on interoception (cf. §11.7).

#### 12.6 Acute vs. Injury-State Pain

| Mode | Timescale | Computational role | Neural signature |
|---|---|---|---|
| Acute nociception | ms–s | Feedforward reflex; minimise $x$ damage rate | Spinal withdrawal, PAG, lateral pain system (S1/S2) |
| Injury-state pain | hours–weeks | Feedback LQG recuperation; re-weight $Q/R$; modulate reward sensitivity | Insula, ACC, vmPFC, BNST; mesolimbic downshift |

Separation is not just phenomenological — their cost matrices $(Q, R)$ and timescales differ, and they recruit different control substrates.

#### 12.7 Cross-Species Evidence

Cephalopod data (Crook and colleagues) show:

- Post-injury primary-nociceptor plasticity (threshold reduction, after-discharge) lasting tens of minutes.
- **Centralised wound-directed grooming and guarding for days** — behaviour expected under an LQG controller with large $Q$ and finite $R$.
- Analgesia from morphine-class agonists modulates these behaviours — preserving the descending-gain role across ~500 million years of divergence.

The convergence is strong evidence that persistent post-injury behaviour is an ancient, optimal-control solution, independently reinvented across phyla because the same LQG problem recurs.

#### 12.8 Meta-Control and Neuromodulatory Precision

Above the primary LQG loop sits a **meta-controller** (insula / ACC / vmPFC) that rewrites $Q$ and $R$ based on context (safe shelter vs. open predator-rich environment, social support, food availability). Meta-control does not solve the LQG itself — it *tunes* it.

- **Opioids** modulate descending gain on $C$ and $L$: under acute threat, stress-induced analgesia transiently *lowers* $L$, letting threat-escape action override protection. Clinically, opioid analgesia collapses $L$ (the innovation becomes invisible).
- **Dopamine** sets effective $Q/R$ ratio and controls action vigor (cf. §6.7 Niv). Post-injury downshift of mesolimbic DA raises the effective $R$ on foraging — enforcing rest.

#### 12.9 Active-Inference Equivalence

The LQG / POMDP formulation maps cleanly onto §7 active inference:

$$
G_\pi \;=\; \underbrace{D_{\mathrm{KL}}\!\left[q(o\mid\pi) \,\|\, p(o\mid C)\right]}_{\text{risk}} \;+\; \underbrace{\mathbb E_{q(s\mid\pi)}\!\left[H[p(o\mid s)]\right]}_{\text{ambiguity}} .
$$

- $p(o\mid C)$: preference for healed-body observations — the analog of $x^{*} = 0$ in LQG.
- Guarding minimises *risk* (predicted observations match the "no further damage" preference) but maximises *ambiguity* (suppresses the very observations that could update belief on healing).
- Chronic pain = prior on injury with precision $\to \infty$; the EFE trade-off collapses onto risk-only, and epistemic sampling ceases.

This gives a formally equivalent statement of §12.5: chronic pain is an **active-inference trap** where policies that minimise immediate risk starve the inference that would permit releasing them.

#### 12.10 Bridge to the Surrounding Review

- **§7 (active inference), §11 (Wiech):** provide the inferential machinery. Seymour et al. add the *control-theoretic* loop around it and the dynamic extension to recuperation.
- **§§8–10 (threat / safety / threat computations):** recuperation is a *distinct* defensive phase — post- rather than pre- / peri-encounter — with its own cost structure ($Q$ emphasizing $x$, $R$ penalizing effort).
- **§6 (Gershman):** opioids and dopamine manifest as precision/gain on $C$, $L$, and effective $R$.
- **§5 (Fleming):** individual differences in meta-$d'$ of interoception predict how well the controller can detect and trust recovery signals — a testable link between metacognition and chronicity risk.

**Bottom line.** Post-injury pain is the *control law* the CNS applies to an injured body. Given the state-space LTI model, Kalman estimation of tissue state, and an LQG cost over recuperation-vs-opportunity-cost, the optimal behaviour is exactly the observed guarding / rest / hypersensitivity profile — *and* the observed failure mode (information restriction producing a stuck $\hat x$) is the formal origin of chronic pain. Paper 13 completes the programme by asking how this framework is to be *forward- and reverse-engineered* into neural interventions.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Introduction.** Acute phasic pain = protective alarm. Post-injury pain is distinct: sensitization, spontaneous pain, fatigue, anxiety. Thesis: brain acts as an *optimal inference engine* driving recuperation; vulnerability is **information restriction** — protective behaviour itself starves the brain of the feedback needed to infer recovery.
- **Ecological and evolutionary perspectives.** Nociception conserved across phyla. Selection pressure switches after injury: need rest, guarding, hypervigilance. Sociality buffers recovery cost. Persistent pain is normal and adaptive, not pathological per se.
- **Box 1 — Comparative studies.** Cephalopod evidence (octopus, cuttlefish, squid): peripheral nociceptors show threshold reduction, increased spontaneous firing; mammal-like wound-directed behaviour + day-long spontaneous firing ⇒ tonic pain shapes long-term behavioural choice. Early-life injury yields permanent threshold changes (tolerance-like adaptation).
- **Formalizing models of persistent pain.** Treat pain as a *signal optimizing a cost-to-go function*, not a subjective phenomenon. Hierarchical loop: saliency → action selection → execution, overseen by meta-learning.
- **A control systems approach to persistent pain and post-injury behaviour.**
  - **Injury state representation.** Bayesian integration of tonic nociception, mechanical haptics, autonomic / immune signals, action-feedback → posterior on injury site and severity.
  - **Control output.** Three effector domains: (i) neuromodulatory (DA/5-HT altering exploration, reward/punishment sensitivity, producing fatigue/anxiety); (ii) pain-sensitivity (stress-induced analgesia, post-threat hyperalgesia/allodynia); (iii) autonomic/endocrine/immune.
  - **Closed-loop control.** Exploration-exploitation dilemma: protective avoidance restricts the information needed to detect recovery.
  - **Meta-control.** Higher layer optimizes meta-parameters (explore vs risk) under context (conspecific presence, resource availability).
- **Box 2 — Control theory and inference.** LTI state-space (existing §12 Phase 2.1): $\dot x = Ax + Bu$, $y = Cx + Du$. Feedforward vs feedback control distinction. Optimal cost $\int_0^T \rho(s,a) dt + \rho_T$ with terminal boundary conditions; stochastic extension = inference problem (Welch–Baum / HMM).
- **Neural implementation.**
  - *Hierarchical inference.* **Insular cortex** = primary hub for multimodal injury-state inference; integrates nociceptive + somatosensory + interoceptive + exteroceptive streams. VMPFC + ACC also participate. Higher-level injury beliefs act as Bayesian priors. Recurrent circuits + STDP implement the probabilistic inference.
  - *Efferent control.* Descending facilitation + stress-hypoalgesia via PAG, hypothalamus, ACC, VMPFC, insula. Hypothalamus regulates sleep, appetite, endocrine tone. Midbrain, frontostriatal, amygdala networks modulate motivational behaviour (revalue rewards, enhance punishment sensitivity, shift risk preferences).
  - *Oscillatory neurodynamics.* Theta (4–9 Hz) in insula, thalamus, S1 tracks saliency; pre-stimulus insular theta modulates pain perception; theta peak-frequency shifts with noxious input. Hypothesis: low-frequency traveling waves coordinate distributed pain network.
- **Chronic pain transition: information restriction model.** Four failure modes:
  1. **Maladaptive learning.** Avoidance prevents acquiring "it no longer hurts" evidence; compounded by increased punishment sensitivity.
  2. **Maladaptive model.** Sensitization inflates $C$ without accompanying efference-copy correction ⇒ noise + uncertainty bias the posterior toward "still injured" because under-estimation has high fitness cost.
  3. **Maladaptive integration.** Nerve lesions / amputations create persistent multisensory incongruence the brain cannot absorb.
  4. **Maladaptive priors.** Pessimistic cognitive beliefs and low perceived controllability override incoming afferent recovery evidence.
- **Conclusion.** Brain has evolved hierarchical optimal-control strategies for injury; distinguishes phasic-defensive from persistent-recuperative pain. Insula-centered network is the substrate. Systemic vulnerability = protective behaviour suppresses its own corrective feedback ⇒ chronic-pain transition.

### Appendix: Section-by-Section Backbone

**Introduction.** Injury universal; healing requires recuperative behaviour. Acute phasic pain = alarm; post-injury pain = persistent, sensitized, affectively loaded. Brain = continuous optimal-inference engine driving recuperation; vulnerability is information-restriction driving chronic pain.

**Ecological and evolutionary perspectives.** Nociception highly conserved. Post-injury selection shifts toward rest/guarding/hypervigilance. Social species can afford longer recovery. Persistent pain = adaptive, not inherently pathological.

**Box 1 — Comparative studies (acute vs prolonged pain function).** Cephalopod nociceptors after injury: threshold reduction + spontaneous firing ⇒ hypervigilance. Short-term plasticity supports wound-directed behaviour; permanent plasticity tolerates persistent danger after early-life injury. Multi-day spontaneous firing = tonic pain analog shaping long-term contextual memory.

**Formalizing models of persistent pain.** Control-theoretic framing: pain is the signal that optimizes a cost-to-go. Hierarchical loop: saliency → action selection → action execution, overseen by meta-learning.

**A control systems approach to persistent pain and post-injury behaviour.**
- **Injury state representation.** Bayesian integration of: tonic nociceptor firing, haptic/mechanical, autonomic/immune, action-driven sensory feedback ⇒ posterior on injury site + severity.
- **Control output (three domains).**
  1. Neuromodulatory (DA, 5-HT): change reward/punishment sensitivity, reduce exploration, produce fatigue/anxiety.
  2. Pain-modulatory: stress-induced analgesia, post-threat hyperalgesia/allodynia.
  3. Autonomic/endocrine/immune responses.
- **Closed-loop control.** Explore-exploit dilemma: protective avoidance restricts the signal needed to detect recovery; trade-off between cost of false "recovered" and cost of false "still injured".
- **Meta-control.** Higher-level regulation tunes meta-parameters (exploration, risk appetite), modulates value functions based on context (conspecifics, resources).

**Box 2 — Control theory and inference.**
- LTI state space: $\dot x(t) = A x(t) + B u(t)$; $y(t) = C x(t) + D u(t)$.
- Feedforward vs feedback control trade-off (speed/efficiency vs robustness to uncertainty).
- Optimal-control cost: $\text{Cost}[a] = \int_0^T \rho(s(t), a(t)) dt + \rho_T$, given dynamics $\dot s = f(s, a)$ with fixed boundary conditions.
- Stochastic extension: optimal policy search ⇔ inference problem, e.g., Welch–Baum in HMMs.

**Neural implementation.**
- **Hierarchical inference.** Insular cortex = primary hub (multimodal integration of nociceptive, somatosensory, interoceptive, and exteroceptive afferents) + VMPFC + ACC supporting a distributed network. Higher-level injury beliefs act as Bayesian priors shaping lower-level processing. Implementation: recurrent local circuits + spike-timing-dependent plasticity.
- **Efferent control.** Descending pain modulation (PAG, hypothalamus, ACC, VMPFC, insula). Hypothalamic regulation of sleep, appetite, endocrine tone. Midbrain / frontostriatal / amygdala circuits re-value rewards, raise punishment sensitivity, modify risk preferences.
- **Oscillatory neurodynamics.** Theta (4–9 Hz) elevated in insula, thalamus, S1 during pain; pre-stimulus insular theta modulates perception; LFP theta peak frequency shifts with noxious input; low-frequency traveling waves posited as inter-area coordination mechanism.

**Chronic pain transition: information restriction model.** Four failure modes:
1. *Maladaptive learning* — avoidance starves the learner; altered motivational valuation (punishment hypersensitivity) amplifies the bias.
2. *Maladaptive model* — sensitization inflates nociceptive $C$ without accompanying efference-copy; the posterior is biased toward persistent-injury because under-estimation is evolutionarily costly.
3. *Maladaptive integration* — nerve lesions / amputations create incongruent multisensory input the brain cannot absorb into its model.
4. *Maladaptive priors* — pessimism / low controllability discount recovery evidence.

**Conclusion.** Brain evolved hierarchical optimal-control strategies for injury; distinguishes acute phasic pain (alarm) from persistent pain (recuperative control). Insular network is the substrate. Protective behaviour restricts its own corrective feedback ⇒ this is the systemic vulnerability that produces chronic pain.

---

## 13. Mahajan & Seymour (2025) — Forward and Reverse Engineering the Pain System

*Pranav Mahajan & Ben Seymour. PAIN, 2025. "Forward and reverse engineering the pain system: from computational neuroscience to neuro-engineering."*

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — Core Problem.** Pain research has three silos: (i) molecular / physiological neuroscience, (ii) psychological and computational theory, and (iii) clinical neurotechnology (DBS, SCS). Theories lack mathematical rigor; pure ML lacks mechanism. Mahajan & Seymour ask: *how can we build a principled engineering pipeline — forward models that predict, and reverse models that decode — that translates computational neuroscience into working clinical devices?*

**Key Findings.**
- **Pain is a POMDP.** The brain must *infer* a hidden body-state from noisy observations and *select control policies* to optimize recuperation.
- **Forward vs reverse engineering as a tight loop.**
  - **Forward:** build mechanistic generative models (Bayesian pain inference, predictive coding, RL with aversive RPE, LQG controller, MPC).
  - **Reverse:** fit those models to data (HDDM, hierarchical Bayesian, EM/Kalman system identification, VAEs for latent axes, NPS-style decoding).
  - Only a *closed loop* of the two yields mechanistically valid, clinically actionable models.
- **Computational phenotyping.** Individual patients have dissociable failures in specific computations (prior precision, drift rate, learning rate, controller gain $K$). "Pain" is not one thing; it is a fingerprint of parameter-level deviations.
- **Digital twins.** A patient-specific POMDP — parameters fit to their data — simulates therapeutic trials in silico before any intervention is performed.
- **Closed-loop neuromodulation.** Smart SCS/DBS devices read a neural biomarker $y_t$, estimate $\hat x_t$, and apply MPC or RL-derived stimulation $u_t$ that drives $\hat x_t$ toward a target while penalizing energy.

**Main Methodology.** Perspective piece synthesizing (i) computational psychiatry / neuroscience (HDDM, active inference), (ii) machine-learning tools (VAEs, latent dynamical systems, system identification), and (iii) control engineering (LQG, MPC, RL-based policy optimization) under a unified POMDP formulation; builds directly on Seymour et al. (§12) and Wiech (§11).

**Initial Takeaway.** Pain neuroscience becomes *pain engineering*. Once a patient's generative model is reverse-engineered, forward-engineered simulation + closed-loop control become tractable. The path from §§1–12 theory to clinical impact passes through this forward/reverse pipeline.

### Phase 2: Graduate-Level Deep Dive

> The paper is a perspective and does not reprint all the algorithmic detail below; equations follow the standard literature it cites (Ratcliff HDDM; Gelman hierarchical Bayes; Macke LDS-EM; Kingma VAE; Åström MPC; Silver DDPG).

#### 13.1 Forward Engineering: The Generative Stack

**Bayesian pain inference (from §11).**

$$
\mu_{\text{post}} \;=\; \frac{\Pi_{\text{obs}}\, o + \Pi_{\text{prior}}\, \mu_{\text{prior}}}{\Pi_{\text{obs}} + \Pi_{\text{prior}}}, \qquad \delta = o - \mu_{\text{prior}}.
$$

**RL with aversive RPE.** With $r_t$ signed (pain magnitude negative, relief positive),

$$
\delta^{\text{RPE}}_t \;=\; r_t + \gamma\, \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t), \qquad
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\, \delta^{\text{RPE}}_t .
$$

**LQG controller (from §12).**

$$
\dot x = A x + B u + w, \quad y = C x + D u + v, \quad u^{*} = -K\hat x,
$$

with $K = R^{-1} B^{\top} P$, $P$ solving the control Riccati equation.

**Model Predictive Control (MPC).** Finite-horizon optimization with constraints:

$$
\boxed{\; \min_{u_{t:t+H}}\; \sum_{k=t}^{t+H}\!\left(\hat x_{k|t}^{\top} Q\, \hat x_{k|t} + u_k^{\top} R\, u_k\right), \;}
$$

subject to $\hat x_{k+1|t} = A\hat x_{k|t} + B u_k$ and $x_{\min} \le \hat x_k \le x_{\max}$, $u_{\min} \le u_k \le u_{\max}$. Implement first $u_t^{*}$, re-plan at $t+1$ (receding horizon). This is the algorithmic backbone of closed-loop neuromodulation.

#### 13.2 Reverse Engineering: Fitting Latent Computations

**13.2.1 Hierarchical Bayesian (HDDM-style).** Subject $j$, trial $i$:

$$
\theta_j \sim \mathcal N(\mu_\theta, \sigma_\theta^{2}), \qquad \mu_\theta \sim p(\mu_\theta), \; \sigma_\theta \sim p(\sigma_\theta),
$$

and choice–RT $(c_{ij}, y_{ij}) \sim \mathrm{WFPT}(a_j, v_{ij}, z_j, t_{0,j})$. Posterior

$$
p(\mu_\theta, \sigma_\theta, \{\theta_j\}\mid \mathcal D) \;\propto\; p(\mu_\theta)\, p(\sigma_\theta)\, \prod_j p(\theta_j \mid \mu_\theta, \sigma_\theta)\, \prod_i p(c_{ij}, y_{ij}\mid \theta_j),
$$

estimated by MCMC (NUTS) or variational inference. Yields subject-level drift / bound / starting-point estimates identifiable even at low trial counts.

**13.2.2 Linear Dynamical System identification via EM.** Latent $x_t$, observation $y_t$:

$$
x_{t+1} = A x_t + B u_t + w_t, \qquad y_t = C x_t + D u_t + v_t,
$$

with $w_t \sim \mathcal N(0, W)$, $v_t \sim \mathcal N(0, V)$. EM:

- **E-step:** run Kalman smoother with current $(A, B, C, D, W, V)$ to obtain posterior $p(x_{1:T} \mid y_{1:T}, u_{1:T})$ — in particular the posterior means $\hat x_t$ and cross-covariances.
- **M-step:** closed-form updates, e.g.

$$
C^{\text{new}} \;=\; \left(\sum_{t} y_t\, \hat x_t^{\top}\right) \left(\sum_{t} (\hat \Sigma_t + \hat x_t \hat x_t^{\top})\right)^{-1} .
$$

Iterate until ELBO converges. This reverse-engineers the patient's $A, B, C$ from multimodal time-series (neural + behavioural + autonomic).

**13.2.3 Variational autoencoders for nonlinear latent axes.** With encoder $q_\phi(z \mid x)$ and decoder $p_\theta(x \mid z)$, maximize the ELBO

$$
\mathcal L(\theta, \phi; x) \;=\; \mathbb E_{q_\phi(z\mid x)}\!\left[\log p_\theta(x\mid z)\right] \;-\; D_{\mathrm{KL}}\!\left[q_\phi(z\mid x) \,\|\, p(z)\right] .
$$

Trained on multi-subject fMRI/EEG, VAEs discover low-dimensional manifolds whose coordinates track acute→chronic pain trajectories, providing *continuous computational biomarkers*.

**13.2.4 Linear decoders (NPS / LDA).** Multivariate pattern:

$$
\hat P \;=\; w^{\top} X + b,
$$

with $w$ learned by LASSO / SVR (Neurologic Pain Signature) or Fisher LDA $w = S_W^{-1}(\mu_1 - \mu_2)$ for spatial filtering of EEG mu/beta (cf. §3.5).

#### 13.3 Closed-Loop Neuromodulation

**MPC-based DBS/SCS.** At each time $t$:

$$
\min_{u_{t:t+H}}\; \sum_{k=t}^{t+H}\!\left((\hat x_k - x^{\text{target}})^{\top} Q (\hat x_k - x^{\text{target}}) + u_k^{\top} R\, u_k\right),
$$

subject to safety constraints $u_k \in \mathcal U_{\text{safe}}$ (current density, charge-per-phase, impedance). $\hat x_t$ is the estimated latent pain state from the Kalman/VAE pipeline of §13.2. Yields stimulation parameters that minimise pain under hard safety limits and energy budget — the formal recipe for the next generation of SCS.

**RL-based SCS parameter optimization.** Alternatively frame the stimulator as an RL agent:

- State $s_t$ = biomarker features (LFP band power, autonomic).
- Action $a_t$ = adjustments to $(\text{freq}, \text{amp}, \text{pulse width})$.
- Reward $r_t = -(w_1\, \widehat{\text{Pain}}_t + w_2\, \text{Energy}(a_t))$.

Deep deterministic policy gradient (DDPG):

$$
\nabla_{\theta^\mu} J \;\approx\; \mathbb E_{s\sim\rho}\!\left[\nabla_a Q(s, a\mid\theta^Q)\big|_{a=\mu(s)}\, \nabla_{\theta^\mu} \mu(s\mid\theta^\mu)\right],
$$

with target-network stabilization. Produces patient-specific stimulation policies that adapt on a per-session timescale without clinician re-tuning.

#### 13.4 Computational Phenotyping and Digital Twins

Once parameters $(\Pi_{\text{prior}}, \Pi_{\text{obs}}, \alpha, \gamma, A, B, C, K)$ are fit per patient, the set $\Theta_j = \{\cdots\}_j$ is the **computational phenotype** — the low-dimensional clinical vector that replaces vague labels like "catastrophizer" with machine-readable parameter deviations.

**Digital twin.** Simulate forward under $\Theta_j$ any candidate intervention $\mathcal I$ (pharmacological, behavioural, stimulation-based):

$$
\widehat{\Delta}_{\mathcal I} \;=\; \mathbb E_{\Theta_j}\!\left[\;\widehat{\text{Pain}}_{t+\Delta t}(\mathcal I) - \widehat{\text{Pain}}_{t+\Delta t}(\text{no intervention})\;\right].
$$

Run in silico trials, select interventions with maximal predicted benefit at minimal predicted risk, then deploy and retune from new data.

#### 13.5 Integration With the Rest of the Review

- **§§1–3 (DDM/CPP)** are *forward-engineered* models. Mahajan & Seymour's contribution is the *reverse* side: HDDM fitting pipelines to extract $(v, a, z, t_0)$ from real pain-discrimination data and use them as clinical phenotypic axes.
- **§4 (Cisek)** urgency-gating becomes a specific MPC predictor inside the forward stack.
- **§5 (Fleming)** meta-$d'$ estimates are reverse-engineered from confidence data and enter $\Theta_j$ as metacognitive-precision biomarkers.
- **§6 (Gershman)** dopamine/opioid modulation of $K$, $L$, $R/Q$ gives the forward model of neuromodulatory tuning that closed-loop devices exploit.
- **§7 (Smith–Friston–Whyte)** the POMDP + VFE framework *is* the forward model; parameter fitting is model inversion in the same formalism.
- **§§8–10 (threat / safety / threat computations)** supply the cost structure $Q$ and the safety-gate releases on the controller.
- **§11 (Wiech)** gives the Bayesian perceptual stack being reverse-engineered.
- **§12 (Seymour et al.)** supplies the control-theoretic formulation ($A, B, C, K$) that Mahajan & Seymour now propose to fit and close-the-loop on.

#### 13.6 Conceptual Closure

The ten-paper arc from Gold & Shadlen (2007) to Levy & Schiller (2021) gave us the *computational theory* of decision-making. Papers 11 and 12 specialised it to pain as perceptual inference and as an LQG controller. Paper 13 closes the loop by stating the engineering programme: **fit the forward models (§13.2), deploy them as digital twins and closed-loop controllers (§§13.3–13.4), and use the resulting parameter vectors as clinical phenotypes.** The modern integrated framework for perceptual, threat, and pain decision-making is therefore not just a theoretical construct — it is the specification of a *device stack* for computational pain medicine.

### Phase 1 / Phase 2 (Backbone-Aligned Synthesis)

Following the paper's own section order:

- **Abstract.** Pain is multi-level (sensory + motivational + cognitive). Thesis: forward + reverse engineering synergistically refines computational models — across RL, control theory, Bayesian inference, active inference — to enable clinical applications (computational phenotyping, personalized therapy, adaptive neuro-engineering).
- **§1 Why do we need a computational approach to understanding pain?** Theory constrains experimental search; models bridge theory ↔ data. Marr-like three-level rationale: descriptive (what), mechanistic (how), normative (why). Contrast with purely data-driven ML.
- **§2 Advent of computational frameworks for pain.** Roots in Pavlovian + instrumental conditioning. Late-1990s RL extended from reward/dopamine to pain prediction, fear conditioning, affective-motivational theories; prediction error as central teaching signal. Computational neuroimaging: striatum and vmPFC implicated in pain as in reward. Bayesian perceptual inference models explain placebo/nocebo. Motivational models frame analgesia as value-based decision balancing pain-avoidance vs reward-seeking (existing §13 Phase 2.1).
- **§3 Theory considerations.** Pain = control under uncertainty ⇒ **POMDP** solved by Bayesian Decision Theory (Fig. 1: environment POMDP = body; agent = mind). Model-class taxonomy:
  1. *State inference without control* — Bayesian perception + DDM quantify suboptimal learning / confirmation bias in chronic pain.
  2. *Control with full observability (MDP)* — RL for safe exploration, self-preservation, punishment sensitivity elevation in chronic pain.
  3. *Inference + control (POMDP)* — belief-MDP planning or RNN-learned representations; formalizes **information-restriction hypothesis** (existing §12 backbone).
  4. *Active inference* — utilities replaced by preference priors; planning = inference; algebraically equivalent to Bayesian Decision Theory via KL-control / soft Bellman (existing §7).
- **§4 Forward and reverse engineering the pain system.** Synergistic loop: forward = generative simulations → testable predictions; reverse = Bayesian model comparison / parameter fitting → mechanistic inference. Example — Pavlovian-instrumental interaction: forward simulations showed Pavlovian fear biases enhance safe learning but reduce efficiency ⇒ hypothesis that **uncertainty-gated** Pavlovian responses optimize safety-efficiency trade-off ⇒ confirmed behaviorally (existing §13 Phase 2.2).
- **§5 Clinical applications and neuro-engineering.** Four avenues:
  - **Computational phenotyping ("computomics")** — principled model-based biomarkers with lower data requirements than brute-force ML (existing §13 Phase 2.4).
  - **Personalised task controllers** — CBT as a two-player game; controller uses forward model to shape patient's generative model toward reduced avoidance / greater resilience.
  - **Activity pacing** — personalised models of activity-pain temporal dynamics to break boom-bust cycles.
  - **Closed-loop neuro-technologies** — model-informed SCS/DBS dynamically tuned to biomarker outcome; systems engineering integrates sensing + stimulation + drugs (existing §13 Phase 2.3).
- **Conflict of interest / Acknowledgements.** Funding: Wellcome Trust, IITP, JSPS, NIHR Oxford Health BRC. Noted ChatGPT use in abstract drafting.

### Appendix: Section-by-Section Backbone

**Abstract.** Pain is multi-level (sensory/motivational/cognitive). Computational approaches bridge theory + data across RL, control theory, Bayesian inference, active inference. Forward + reverse engineering synergize to deliver computational phenotyping, personalised therapy, adaptive neuro-engineering.

**§1 Why do we need a computational approach to understanding pain?** Theories constrain hypothesis generation; computational models convert theories into mathematical structures testable against observed data. Contrast with purely data-driven ML. Marr-like rationale: descriptive (what problem), mechanistic (how solved), normative (why solved this way). Multilevel unification across phenotype + methodology is essential for pain.

**§2 Advent of computational frameworks for pain.** Roots: Pavlovian + instrumental conditioning. Late-1990s RL extended from reward/dopamine to fear conditioning, pain prediction, affective-motivational pain theory; prediction error as teaching signal. Neuroimaging: striatum + vmPFC involved in pain as in reward. Bayesian perceptual inference: priors shape pain (placebo, nocebo). Motivational-analgesia models: pain modulation as value-based decision balancing avoidance and reward, echoing behavioural economics.

**§3 Theory considerations.** Brain faces control under uncertainty ⇒ POMDP solved via Bayesian Decision Theory.
- *Figure 1 schematic.* Environment POMDP = body; agent = mind. Sensory processing = Bayesian filtering of nociceptive observations; motivational signals are punishment-based temporal credit assignment.
- Model taxonomy:
  - State inference without control (Bayesian perception + DDM) — quantify suboptimal learning, confirmation bias.
  - Control with full observability (MDP) — RL for safe exploration, self-preservation, punishment sensitivity.
  - Inference + control (POMDP / belief-MDP or RNN) — formalizes information-restriction hypothesis for chronic pain.
  - Active inference — preference priors replace utility; Bayes-optimal behaviour via KL-control / soft-Bellman equivalent to BDT.
- Mapping: belief states, value functions, expected utilities, prediction errors, Bayesian surprise all become explicit POMDP components.

**§4 Forward and reverse engineering the pain system.** Loop: forward (build from first principles, simulate, predict) + reverse (fit models to data via Bayesian model comparison etc.) ⇒ synergistic refinement. Example: Pavlovian + instrumental withdrawal interaction. Forward simulations: Pavlovian fear biases improve safe learning but reduce efficiency. Normative hypothesis: uncertainty-gated Pavlovian bias best balances safety and efficiency. Behavioural test confirmed the hypothesis ⇒ illustrates the constructivist-plus-reductionist cycle.

**§5 Clinical applications and neuro-engineering.** Four avenues:
1. **Computational phenotyping ("computomics")** — model-based biomarkers requiring less data than brute-force ML.
2. **Personalised task controllers** — CBT as two-player game: controller shapes patient's generative model toward reduced avoidance / greater resilience using a forward-engineered model.
3. **Activity pacing** — personalised activity-pain temporal models guide routine to break boom-bust cycles in chronic pain.
4. **Closed-loop neuro-technologies** — model-informed understanding of biomarker generative processes enables dynamically-tuned SCS/DBS; systems engineering toward holistic sensing + intervention + pharmacological integration.

**Conflict of interest statement.** None declared.

**Acknowledgements.** Funding: Wellcome Trust, IITP, JSPS, NIHR Oxford Health BRC. Authors disclose use of ChatGPT for abstract wording / length compliance.

---

## Cross-Paper Synthesis (final)

Thirteen papers now constitute a single coherent framework spanning perception → action → survival → pain → clinic:

1. **Inference core (§§1–3).** SDT / SPRT / DDM; LIP and CPP; neurally-informed hierarchical-Bayesian DDM.
2. **Embodied action (§4).** Affordance competition and urgency-gating; the DDM is a limit case of parallel sensorimotor competition.
3. **Metacognition (§5).** Propositional confidence, meta-$d'$, two-stage DDM; vmPFC / pMFC / frontopolar.
4. **Neuromodulation (§6).** Dopamine as a family of PEs tuning drift, bound, vigor, and average reward.
5. **Variational umbrella (§7).** Active inference: POMDP + VFE / EFE; contains §§1–6 as limits.
6. **Defensive taxonomy (§8).** Reflex → Pavlovian → MF → MB → deliberation; POMDP for uncertain threat.
7. **Safety computation (§9).** Anterior-vmPFC positive safety signal integrating threat and perceived control; gates DDM and EFE.
8. **Threat computations (§10).** Dimensional threat vector $(V_{\text{harm}}, \sigma^2, A, d, C)$; volatility, latent causes, dimensional neural map.
9. **Pain as inference (§11).** Bayesian posterior with precision-weighted PE; expectation shifts DDM starting point; placebo/nocebo as prior manipulation; chronic pain as active-inference trap.
10. **Pain as control (§12).** LTI plant–sensor–controller; LQG recuperation cost; Kalman-filtered $\hat x$; chronic pain = information-restriction failure.
11. **Pain engineering (§13).** Forward generative stack + reverse fitting (HDDM, LDS-EM, VAE) + closed-loop MPC/RL control → computational phenotypes and digital twins.

The unifying object is an **agent performing variational inference over a POMDP** whose generative model contains: (i) a DDM-like perceptual module, (ii) Pavlovian + instrumental RL systems with adaptive learning rates, (iii) safety and threat variables with anterior/posterior vmPFC readouts, (iv) an LQG/MPC body-recuperation controller with Kalman tissue-state estimation, and (v) precision/gain parameters tied to dopamine and opioids. Fitting that agent to individual data is the translational bridge from perceptual decision-making theory to computational pain medicine.
