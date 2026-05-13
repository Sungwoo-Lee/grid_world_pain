---
title: "Literature Review — Computational Models of Pain (local PDF corpus, 2026)"
status: DRAFT v2 — all 6 papers populated from local PDFs (NotebookLM no longer required)
last_updated: 2026-05-07
source_corpus: "Local PDFs at docs/project/references/computational_models_of_pain/ (NotebookLM session retired due to auth instability)"
related:
  - project_plan.md
  - perceptual_noise_lit_review.md
  - ../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md
  - ../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md
---

> **Status note (2026-05-07, resumed run).** All six papers are now populated. Paper 1 (Crook
> 2014) was previously populated via a successful NotebookLM source-grounded query and is
> retained as-is (cross-checked against the local PDF; no factual discrepancies found).
> Papers 2–6 were populated by reading the local PDFs directly with the `Read` tool, as
> NotebookLM auth was unstable across batches. Section ordering still reflects per-paper
> processing order; thematic regrouping is left to a `literature-curator` pass — see
> *Cross-paper synthesis hooks* at the bottom of this document for the seed clusters.

# Literature Review — Computational Models of Pain

> **Scope.** This review covers the *Computational Models of Pain* NotebookLM corpus:
> behavioral-ecology accounts of nociceptive sensitization under predation, persistent nociceptor
> hyperactivity as an evolutionary adaptation, hypervigilance signatures in chronic-pain models,
> safe-RL formulations of punishment-driven control, and control-theoretic / inference-and-control
> frameworks for post-injury behavior. Each paper is processed with a four-step section-ordered
> backbone followed by a Phase 1 (foundational) + Phase 2 (graduate) synthesis with full LaTeX
> derivations, and ends with an explicit *Relevance to project* line linking the paper's
> computational primitives to the GridWorld Pain agent (FiLM precision head, predictive-coding
> gating, interoceptive uncertainty).

## Project hooks this review explicitly tracks

- **Nociception vs pain.** Where does each paper draw the line, and does the formalism distinguish
  the two (transduction vs valuation/inference)?
- **Predictive-coding accounts of chronic pain.** Büchel/Wiech-style precision-on-prediction-error
  formulations and any analogue.
- **Bayesian placebo / nocebo.** Prior-driven modulation of perceived pain.
- **Hypervigilance & attentional bias.** Observable computational signatures we could replicate
  in-grid (e.g., prior shift toward predator detection after injury).
- **Fear-avoidance, interoceptive inference, allostasis.** Control-as-inference frames.
- **Opioidergic / descending modulation as a computational variable.** Gain modulation, gate
  control.
- **FiLM / precision-head hooks.** Whenever a paper's mechanism could plug into our precision
  modulation head.

## Table of Contents

- [1. Crook (2014) — Nociceptive Sensitization Reduces Predation Risk](#1-crook-2014--nociceptive-sensitization-reduces-predation-risk) — **POPULATED**
- [2. Elfwing & Seymour (2017) — Parallel reward & punishment control: MaxPain](#2-elfwing--seymour-2017--parallel-reward--punishment-control-maxpain) — **POPULATED**
- [3. Lister et al. (2020) — Chronic pain produces hypervigilance to predator odor in mice](#3-lister-et-al-2020--chronic-pain-produces-hypervigilance-to-predator-odor-in-mice) — **POPULATED**
- [4. Mahajan, Dayan & Seymour (2026) — Homeostasis after injury: intertwined inference and control](#4-mahajan-dayan--seymour-2026--homeostasis-after-injury-intertwined-inference-and-control) — **POPULATED**
- [5. Seymour et al. (2023) — Post-injury pain and behaviour: a control theory perspective](#5-seymour-et-al-2023--post-injury-pain-and-behaviour-a-control-theory-perspective) — **POPULATED**
- [6. Walters et al. (2023) — Persistent nociceptor hyperactivity as a painful evolutionary adaptation](#6-walters-et-al-2023--persistent-nociceptor-hyperactivity-as-a-painful-evolutionary-adaptation) — **POPULATED**
- [Cross-paper synthesis hooks](#cross-paper-synthesis-hooks)

> Order reflects the order in which papers were processed; thematic regrouping (predator-driven
> nociception cluster ↔ control-theoretic cluster) is left to a follow-on `literature-curator`
> pass once all six entries are populated.

---

<!-- Paper sections will be appended below this marker. -->

## 1. Crook (2014) — Nociceptive Sensitization Reduces Predation Risk

**Citation key.** Crook, R. J., Dickson, K., Hanlon, R. T., & Walters, E. T. (2014). Nociceptive
sensitization reduces predation risk. *Current Biology*. (Squid *Doryteuthis pealeii* vs black
sea bass *Centropristis striata*.) Source-grounded via NotebookLM corpus.

### Phase 1 — Foundational overview

**Introduction.** Animals injured non-lethally develop *nociceptive sensitization*: the
nociceptors and downstream circuits become more easily excited for hours to days after the
event, producing what looks behaviorally like ongoing tenderness, guarding, and elevated
startle. The textbook story has long been that this is *adaptive*: it should make the animal
more cautious and therefore more likely to survive while it heals. Crook et al. provide the
first direct experimental test of that adaptive-fitness claim, by comparing injured squid that
*did* develop nociceptive sensitization with injured squid in which sensitization was
selectively blocked.

**Key findings.**

1. **Injury attracts predators.** Black sea bass selectively oriented toward and pursued
   injured squid from significantly longer distances than uninjured squid, even though the
   minor arm-tip amputation produced no detectable swimming impairment (start distance
   $p=0.02$; pursuit-initiation distance $p=0.001$). Mixed-treatment trials confirmed selective
   capture (odds ratio $11.7$, $p=0.05$).
2. **Sensitization is a hypervigilance signature.** Injured-but-unanesthetized squid (sensitization
   intact) initiated secondary defensive behaviors and flight at *greater* distances than
   uninjured squid ($p=0.03$ and $p=0.008$ respectively, Mann–Whitney U), and engaged defenses
   at significantly earlier stages of the predator's escalation cascade ($p=0.004$).
3. **Net survival benefit.** Across 30-minute free-interaction trials:
   - Uninjured (U): $16/20 = 80\%$ survival
   - Uninjured + anesthetic (UA): $12/16 = 75\%$
   - Injured (I): $9/20 = 45\%$
   - Injured + anesthetic (IA, sensitization blocked): $3/16 \approx 18.75\%$
   The cost of injury (I vs U) is an odds ratio of $4.89$ ($p=0.026$); blocking sensitization
   on top of injury (IA vs U) raises that to $17.33$ ($p=0.008$); and the *benefit* of
   sensitization (I vs IA) is an odds ratio of $3.54$ ($p=0.05$).

**Initial takeaway.** Nociceptive sensitization is not just a side-effect of tissue damage —
it is a behaviourally measurable, evolutionarily conserved *threat-gain modulator* whose
operational signature is a longer alert distance and earlier engagement of escape responses.
Pain-like persistence makes mechanistic sense as a survival circuit, not as a bug.

### Phase 2 — Graduate-level deep dive

**Experimental machinery.** Adult *D. pealeii* (mantle length $14$–$22\;\mathrm{cm}$, $n=72$)
were assigned 6 h before trials to one of four treatments — Uninjured (U, $n=20$), Injured
(I, $n=20$), Uninjured + Anesthetic (UA, $n=16$), Injured + Anesthetic (IA, $n=16$). Injury
was a $5$–$10\;\mathrm{mm}$ amputation of one third arm with sterile scalpel. The anesthetic
regimen combined general anesthesia ($1\%$ ethanol in seawater for $2\;\mathrm{min}$) with a
focal local block ($0.5$–$1\;\mathrm{ml}$ isotonic $\mathrm{MgCl}_2$ injected into arm muscle
$1\;\mathrm{min}$ before injury). The local $\mathrm{MgCl}_2$ block stays at the injection
site (verified by absence of chromatophore relaxation outside that region) and prevents the
development of both local and remote nociceptor sensitization without paralyzing the animal.
This pharmacological double dissociation is what isolates "nociceptive sensitization" as a
controllable variable.

**Two stages of the predator–prey encounter.** Each trial decomposes into a Markov-like
escalation cascade

$$\text{orient} \rightarrow \text{pursuit} \rightarrow \text{attack} \rightarrow \text{capture}$$

with five distance measurements: predator *start distance* (orientation onset), *pursuit
initiation distance*, prey *alert distance*, *flight-initiation distance*, and *predation
stage at alert*. All distances were normalized to squid body length (the figures show the
y-axis range $0$–$20$ body lengths) since mantle length varied across animals.

**Statistical model.** Crook et al. used:

- two-way ANOVA + post-hoc two-tailed $t$-tests on continuous distance measures (start
  distance, pursuit-initiation distance);
- Kruskal–Wallis + Bonferroni-corrected Mann–Whitney $U$ on rank/ordinal measures (alert
  distance, flight-initiation distance, predation stage);
- Fisher's exact tests on categorical transition probabilities (orient→pursuit, pursuit→attack,
  attack→capture);
- one-tailed odds ratios with $z$-scores on the *a priori* survival comparison IA vs I; all
  other tests two-tailed at $\alpha=0.05$.

This is a clean factorial design: with the four cells (U, I, UA, IA) we can identify the main
effect of injury (I − U), the main effect of anesthesia (UA − U), and — critically — the
interaction (IA − I) − (UA − U), which is the contribution of nociceptive sensitization
*conditional on injury*.

**A back-of-envelope decomposition of the survival data.** Let $p_T$ be survival probability
under treatment $T$. We have

$$p_U = 0.80,\quad p_{UA} = 0.75,\quad p_I = 0.45,\quad p_{IA} = 0.1875.$$

The log-odds for each cell are $\ell_T = \log\!\bigl(p_T/(1-p_T)\bigr)$:

$$\ell_U = \log(0.80/0.20) = \log 4 \approx 1.386,$$

$$\ell_{UA} = \log(0.75/0.25) = \log 3 \approx 1.099,$$

$$\ell_I = \log(0.45/0.55) \approx -0.201,$$

$$\ell_{IA} = \log(0.1875/0.8125) \approx -1.466.$$

The cost of injury in log-odds:

$$\Delta_{\text{injury}} = \ell_U - \ell_I \approx 1.386 - (-0.201) = 1.587,$$

which exponentiates to an odds ratio $e^{1.587} \approx 4.89$ — exactly the value reported by
Crook et al. The *additional* cost of removing sensitization on top of injury:

$$\Delta_{\text{sensitization}} = \ell_I - \ell_{IA} \approx -0.201 - (-1.466) = 1.265,$$

so the protective odds ratio of sensitization is $e^{1.265} \approx 3.54$, again matching the
paper. The total IA-vs-U gap is $\Delta_{\text{injury}} + \Delta_{\text{sensitization}} \approx
1.587 + 1.265 = 2.852$, giving $e^{2.852} \approx 17.3$ (paper: $17.33$). The arithmetic is
consistent and reveals an additive log-odds structure: injury and sensitization-blockade
contribute *separable* hazards.

The anesthetic alone (UA − U) is small ($\Delta_{\text{anesth, baseline}} = \ell_U - \ell_{UA}
\approx 0.287$), confirming that the $\mathrm{MgCl}_2$ block by itself is not a mortality
confound.

**A simple precision/gain reading of the alert-distance data.** The behavioural variables most
useful for our project are *alert distance* and *flight-initiation distance*. In a minimal
threat-detection model the prey integrates noisy sensory evidence $x_t$ about predator
proximity and triggers defense when a posterior threat probability $P(\text{threat} \mid
x_{1:t})$ crosses a threshold $\theta$. Hypervigilance can show up either as a lower
threshold $\theta$, a higher prior $P(\text{threat})$, or a higher likelihood gain (precision)
$\pi$. Under any of these three modulations, the *expected hit time* — the predator distance
at which the threshold is first crossed — increases. Crook et al. observe exactly that
distance increase ($p=0.03$ for alert distance, $p=0.008$ for flight-initiation distance),
and the increase is gated by injury *and* by an intact nociceptive pathway. This is
behaviourally consistent with state-dependent gain modulation on threat detection — the
kind of modulation our FiLM-based precision head is built to express.

**Why the orientation-to-pursuit drop matters.** A second-order finding: among encounters
that began (predator orients), sensitized injured squid were *less likely* to escalate to
active pursuit than either uninjured ($p=0.046$) or anesthetized-injured ($p=0.017$) squid.
This is consistent with the sensitized squid producing earlier or more conspicuous *secondary
defense* — postural changes, ink, jetting — that aborts the chase. The implication is that
the survival benefit of nociceptive sensitization arises at *two* points in the cascade
(longer alert distance + lower escalation probability), not from a single reflex change.
Computationally, the system has both a detection-gain effect and a policy effect.

### Relevance to project

- Direct hook for our FiLM precision head: the alert-distance and flight-distance shifts are
  exactly the kind of *behavioural* signature we want to exhibit when an injury / nociceptive
  signal up-modulates the precision applied to predator-relevant features. The factorial
  (injury × sensitization) design is also a template for an in-grid ablation: nociceptive
  signal on/off × precision-head on/off.
- Supports the project framing of nociception as adaptive *gain-control on threat-relevant
  prediction error*, not merely as a punishment scalar.
- The two-stage benefit (detection + escape) suggests our agent's hypervigilance hook should
  affect *both* the perception module (precision on predator features) and the policy
  (action selection threshold), not just one — see also Mahajan & Seymour (2024) and Seymour
  (2019) below for control-theoretic versions of the same idea.
- The clean log-odds decomposition above means we have a quantitative target shape for any
  in-grid replication: a behavioural manipulation that adds an injury-dependent, precision-head-dependent
  log-odds boost to survival of order $\Delta \approx 1.27$ would qualitatively reproduce
  Crook et al.'s "benefit of sensitization" effect size.

---

## 2. Elfwing & Seymour (2017) — Parallel reward & punishment control: MaxPain

**Citation key.** Elfwing, S., & Seymour, B. (2017). Parallel reward and punishment control in
humans and robots: safe reinforcement learning using the MaxPain algorithm. *IEEE ICDL-EPIROB*
2017. Source: local PDF (8 pp.) at
`docs/project/references/computational_models_of_pain/sources/Elfwing and Seymour 2017 - Parallel reward and punishment control in humans and robots - Safe reinforcement learning using the MaxPain algorithm.pdf`.

### Phase 1 — Foundational overview

**Introduction.** Standard RL collapses positive and negative outcomes onto a single scalar
reward and learns one $Q$-function. Biology is more nuanced: rodents and humans appear to use
*separate* circuits for reward (dopaminergic) and punishment (insular / striatal aversive)
prediction, and these circuits can be independently lesioned, drug-modulated, or behaviorally
biased (Eldar et al. 2016 show that *learning rates* for "punishment" and "punishment omission"
dissociate at the single-subject level). Elfwing & Seymour translate that into an algorithmic
proposal: train **two parallel Q-functions** — one for accumulated *positive* reward $Q_r$,
one for accumulated *pain* $Q_p$ — and combine them at policy time with a single scaling knob
$w$. The hypothesis: this *parallel* scheme should produce dramatically *safer* exploration
without sacrificing long-run performance, especially under sparse / delayed reward in
dangerous environments.

**Key findings.**

1. **Painful grid-world (10×10-ish maze with inner walls; +1 at goal, $-0.1$ per wall hit).**
   Across 100 runs of 1000 episodes:
   - MaxPain with $w=0.5$ cut the cumulative steps-to-goal by **75%** (from $4{,}928{,}982$ for
     vanilla Q-learning to $1{,}236{,}884$).
   - Cumulative wall hits dropped by **62%** ($191{,}483 \to 72{,}141$).
   - Final-policy quality was *not* sacrificed: $64.75$ steps-to-goal vs Q-learning's $64.02$
     (the optimal path is $64$ steps).
   - $w=0.1$ is the safest in early exploration but converges to a worse final policy
     ($92.30$ steps); $w=0.9$ matches the final but doesn't gain on safety.
2. **Delayed-reward mountain car** (sparse +1 at the top, $0.1$ pain when stuck near valley
   bottom at low velocity). Same pattern: $w=0.5$ cut cumulative steps by **62%**
   ($1{,}652{,}862 \to 632{,}343$) and gave the best final performance ($116.8$ vs $135.9$ for
   SARSA($\lambda$)).
3. **Mechanism is a "potential field" of pain.** The learned $Q_p$ surface looks like a
   repulsive potential rising away from the goal — it actively *steers* exploration toward
   safe corridors and toward the goal. With a single $Q$-stream this signal is buried in
   noise; with parallel streams it survives as an independent gradient.

**Initial takeaway.** A pain stream is not just bookkeeping — it acts as a learned safety
gradient that biases exploration. The whole effect is parameterised by a *single scalar* $w$
that interpolates between "safety-first" and "reward-first" policies. This single scalar is a
candidate parametric model of clinically observed extremes: low-$w$-like behavior in
**fear-avoidance chronic pain** and OCD, high-$w$-like behavior in disorders that
under-weight harm.

### Phase 2 — Graduate-level deep dive

**Setup.** Standard discounted MDP. The agent observes state $s$, chooses $a$ from a
stochastic policy $\pi_t(s,a)$, gets scalar reward $R$, transitions $s \to s'$. The
action-value function is

$$Q^\pi(s,a) = \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k R_{t+k} \,\Big|\, s_t=s,\; a_t=a\right], \qquad Q^*(s,a)=\max_\pi Q^\pi(s,a). \tag{1}$$

**Sign-split of the reward.** MaxPain decomposes $R$ into two non-negative streams:

$$r = \max(R, 0) \;\geq\; 0, \tag{2} \qquad p = -\min(R, 0) \;\geq\; 0. \tag{3}$$

This is a *lossless* re-parameterisation when $R\in\mathbb{R}$ — note $R = r - p$ — so no
information is thrown away. Two action-value functions $Q_r, Q_p$ are then learned
*independently*, each estimating the discounted accumulation of its own non-negative stream.

**Policy combiner.** The combined Q used for action selection is the linear pencil

$$Q_w(s,a) \;=\; w\, Q_r(s,a) \;-\; (1-w)\, Q_p(s,a), \qquad w \in [0,1]. \tag{4}$$

The minus sign is the key: $Q_p$ is a *positive* prediction of accumulated pain, so it must
be *subtracted* in the combiner so that "high pain" repels the policy. $w$ trades safety
($w\to 0$) against reward-seeking ($w\to 1$).

**Updates.** Both streams use TD with their own learning rates and discount factors:

$$Q_r(s,a) \leftarrow Q_r(s,a) + \alpha_r \,\delta_r, \tag{5} \qquad Q_p(s,a) \leftarrow Q_p(s,a) + \alpha_p \,\delta_p. \tag{6}$$

The *reward* TD-error has the standard two flavours, off-policy Q-learning

$$\delta_r \;=\; r + \gamma_r\, Q_r\!\left(s',\; \arg\max_{a'} Q_w(s',a')\right) - Q_r(s,a), \tag{7}$$

or on-policy SARSA

$$\delta_r \;=\; r + \gamma_r\, Q_r(s', a') - Q_r(s,a). \tag{8}$$

The *pain* TD-error is **always off-policy** with respect to the combined behaviour policy and
uses an `argmin` rather than `argmax`:

$$\boxed{\;\delta_p \;=\; p + \gamma_p\, Q_p\!\left(s',\; \arg\min_{a'} Q_w(s',a')\right) - Q_p(s,a).\;} \tag{9}$$

**Why the `argmin Q_w` and not `argmax Q_p` matters (derivation).** A naive parallel scheme
would update $Q_p$ towards the action that *maximises future pain* under $Q_p$ alone:

$$\delta_p^{\text{naive}} = p + \gamma_p \max_{a'} Q_p(s',a') - Q_p(s,a). \quad(\text{not used})$$

That target is the *worst-case-self* pain — what a maximally self-destructive policy would
suffer — and is mostly irrelevant once the agent is acting under $Q_w$. Eq. (9) instead asks:
*"if I follow my real (combined) policy from $s'$, which action will it most likely take, and
how much pain do I predict from that action?"* Formally, let $\pi_w$ be the (deterministic
limit of the) softmax policy over $Q_w$. Then the right-hand TD target in (9) is an off-policy
estimate of

$$Q_p^{\pi_w}(s,a) = \mathbb{E}_{s'\sim P(\cdot\mid s,a)}\!\left[ p + \gamma_p\, Q_p^{\pi_w}(s', \pi_w(s')) \right].$$

But note Elfwing & Seymour use $\arg\min_{a'} Q_w(s',a')$ — the action $\pi_w$ is *least* likely
to take. The cleanest reading is that they are estimating the *escape value* $Q_p$ along the
combined policy's *avoidance* direction: "how much pain would I incur if I deviated toward the
action my policy is repelling away from?" This is what makes $Q_p$ usable as a *negative
potential* in (4) without it collapsing to $Q_r$. The expectation taken on each step is

$$\mathbb{E}\bigl[\delta_p\bigr] = \mathbb{E}\!\left[p + \gamma_p Q_p\!\left(s',\,\arg\min_{a'}Q_w(s',a')\right)\right] - Q_p(s,a),$$

and the fixed point (when $\delta_p \to 0$ in expectation) gives a $Q_p$ that is consistent
with bootstrapping along the *anti-policy* of $Q_w$ — i.e. it *anchors* what avoidance is
avoiding.

**Action selection.** Boltzmann softmax over $Q_w$ with hyperbolic temperature schedule:

$$\pi(a\mid s) \;=\; \frac{\exp\bigl(Q_w(s,a)/\tau\bigr)}{\sum_b \exp\bigl(Q_w(s,b)/\tau\bigr)}, \tag{10} \qquad \tau(i) \;=\; \frac{\tau_0}{1 + \tau_k\, i}. \tag{11}$$

For the grid-world: $\alpha=0.1$, $\gamma=0.99$, $\tau_0=0.5$, $\tau_k=0.05$. For mountain car
with SARSA($\lambda$) plus 16×16 RBF features, $\alpha=0.1$, $\gamma=0.995$, $\lambda=0.8$,
$\tau_0=1$, $\tau_k=1$. Function approximation uses

$$Q(s,a\mid\boldsymbol{\theta}) \;=\; \sum_i \theta_{ai}\, \phi_i(s), \quad \phi_i(s)\;=\;\exp\!\left(-\frac{\|s-c_i\|^2}{2\sigma_i^2}\right), \tag{16,17}$$

with eligibility-trace SARSA($\lambda$):

$$\boldsymbol{e} \leftarrow \gamma\lambda \boldsymbol{e} + \nabla_{\boldsymbol\theta} Q(s,a\mid\boldsymbol\theta), \qquad \boldsymbol\theta \leftarrow \boldsymbol\theta + \alpha\,\delta\,\boldsymbol{e}. \tag{18,19}$$

**Why MaxPain wins early — a back-of-envelope reading.** Consider a single trajectory of
length $T$ in which the agent hits $k$ walls before reaching the goal. Under vanilla
Q-learning with $R\in\{+1, -0.1\}$, every wall-hit injects a $-0.1$ TD target *competing* with
the same $Q$ that is also trying to predict the eventual $+1$. The signal-to-noise on the
"avoid wall" gradient is low because both targets share an estimator. Under MaxPain those
$-0.1$'s are diverted to $Q_p$ as $+0.1$ pain, so $Q_p$ is a *purely positive* function of
"discounted distance to wall hits", and (4) treats it as a clean repulsive potential. This is
why Elfwing & Seymour's heat maps of $V_w(s) = \max_a Q_w(s,a)$ (their Fig. 2 middle row) show
a smooth potential field that visibly tilts away from walls and toward the goal — a structure
that is essentially *invisible* in the vanilla Q-learning $V(s)$ heat-map (their Fig. 2
bottom row) until late in training.

**Where $w$ matters most.** The sensitivity of the final policy and of the cumulative
exploration cost to $w$ is non-trivial:

| $w$ | grid-world final steps | grid-world cum. steps (×$10^6$) | grid-world cum. wall hits | mountain-car final steps |
|---|---|---|---|---|
| $0.1$ | $92.30$ (worst) | smallest | smallest | $160.2$ (worst) |
| $0.5$ | $\mathbf{64.75}$ | $\mathbf{1.24}$ | $\sim$72k | $\mathbf{116.8}$ |
| $0.9$ | $64.70$ | almost as large as Q-learning | $\sim$Q-learning | $118.9$ |
| Q-learning ref | $64.02$ | $4.93$ | $191{,}483$ | $135.9$ (SARSA$\lambda$) |

The U-shape in the *cumulative-cost* column says: $w$ is genuinely two-sided — too small and
the agent never commits to the goal; too large and it stops exploiting the safety stream.
$w=0.5$ is the equal-weight point and (somewhat surprisingly given the asymmetric reward
magnitudes $+1$ vs $-0.1$) is empirically the best operating point in both benchmarks. This
is partly a feature of the small action space and small punishment magnitude; in higher-pain
regimes the optimal $w$ would shift toward $0$.

### Relevance to project

- **Direct architectural map onto our agent.** Our predictive-coding setup already separates
  perceptual prediction error from value; MaxPain's *parallel-Q* skeleton is the natural
  control-side analogue — one head for $Q_r$ (food, survival reward), one for $Q_p$ driven by
  the nociceptive / interoceptive stream. The combiner weight $w$ is the most natural place
  to plug **our FiLM precision head**: instead of a global scalar, FiLM produces a
  state-dependent scaling that effectively makes $w \to w(s, \text{interoceptive state})$.
  This matches the project's working hypothesis that injury / nociception should up-weight
  the avoidance stream at exactly the contexts where prediction-error is interoceptively
  certain.
- **Construct-validity of "pain-like" claims.** Section IV explicitly proposes that
  pathological values of $w$ correspond to **fear-avoidance chronic pain** (Crombez et al.
  2012) and **compulsive avoidance in OCD** (Hauser et al. 2016). This is the most concrete,
  algorithmic version of "chronic pain as a controller miscalibration" we have in this
  corpus. It is a strong constraint on us: any "pain-like" claim we make for our agent
  should be expressible as a shift in an explicit, identifiable parameter (here, $w$, or
  in our case the FiLM gain on the $Q_p$ pathway), not as a mysterious behavioural change.
- **Argmin-under-Q_w trick is reusable.** Equation (9) is a small, exact, off-policy
  modification we can replicate in our codebase if/when we want a parallel-Q variant — note
  that it requires the *combined* $Q_w$ to drive bootstrapping in the pain stream, so the
  two streams cannot be trained truly independently in any naive sense.
- **Hypervigilance signature is grid-visible.** The heat-maps (their Fig. 2) show that
  MaxPain agents *concentrate visits in the centre of corridors* (≥2 steps from any wall),
  i.e. produce a behavioural signature of "extra wall-avoidance margin" that is exactly what
  Crook et al. 2014 (Section 1 of this review) would predict as an alert-distance increase
  for an injured / sensitised animal. That is, MaxPain provides a *computational mechanism*
  whose behavioural signature matches Crook's *biological* hypervigilance signature — a
  fact the authors do not make explicitly but that is critical for our story.
- **Limit cases for our config matrix.** A useful set of in-grid sanity runs is the same
  $w$-sweep ($\{0.1, 0.5, 0.9\}$) Elfwing & Seymour use, with our nociceptive signal in the
  $p$ slot. If FiLM-gated $w(s)$ outperforms any single $w$ in their sweep on cumulative
  steps + wall-hit cost, that's a clean architecture-effect demonstration; if it doesn't,
  the FiLM head is not buying us anything beyond a tuned scalar.

> **Recommended follow-up referrals.** `professor-rl-bayesian-dl` to assess MaxPain ↔ FiLM
> compatibility (whether replacing scalar $w$ by a FiLM-conditioned $w(s)$ preserves the
> argmin-under-$Q_w$ fixed-point argument). `professor-pain-modeling` to evaluate whether
> $w$-as-pathology is a defensible "pain-like" construct or merely a control-tuning knob.

---

## 3. Lister et al. (2020) — Chronic pain produces hypervigilance to predator odor in mice

**Citation key.** Lister, K. C., Maldonado Bouchard, S., Markova, T., Aternali, A., Denecli, P.,
Donayre Pimentel, S., Majeed, M., Austin, J.-S., Williams, A. C. de C., & Mogil, J. S. (2020).
Chronic pain produces hypervigilance to predator odor in mice. *Current Biology*, **30**(15),
R866–R867 (Correspondence; pp. R866–R867). Source: local PDF (2 pp.) at
`docs/project/references/computational_models_of_pain/sources/Lister et al. 2020 - Chronic pain produces hypervigilance to predator odor in mice.pdf`.

### Phase 1 — Foundational overview

**Introduction.** Pain researchers usually treat *acute* and *tonic* pain as adaptive
(withdraw-from-damage; enforce-recuperation) and *chronic* pain as a pathophysiological
mistake — the textbook framing is the **"smoke detector" principle** (Nesse & Schulkin 2019):
chronic pain is a stuck-on alarm. Crook et al. (2014; Section 1 of this review) challenged
that view in *invertebrates* (squid + black sea bass). Lister et al. ask the obvious follow-up
in *mammals*: does chronic pain in a mouse increase **predator-relevant vigilance** in a way
that is consistent with chronic pain being an adaptive *hypervigilance* mechanism rather than
a malfunction?

**Key findings.**

1. **Custom octagonal O-maze with spatially restricted predator odor.** Mice are food-deprived
   and trained to fetch a food reward via a *short route* (~$30\;\mathrm{cm}$); the *long
   route* around the octagon is ~$200\;\mathrm{cm}$. A pump-and-vacuum system pushes
   volatilised fox urine into *only the short-route octant*. Mice are pre-trained until they
   take the short route on $>80\%$ of $10$ consecutive trials.
2. **Fox urine alone produces dose-dependent route avoidance.** With sample sizes
   $n=5$–$11$ across three concentrations, a one-way ANOVA on the percent-short-route metric
   gives $F_{2,19}=7.4$, $p=0.004$; mice avoid the fox-odor short route in a dose-dependent
   fashion. This is the *baseline vigilance* curve.
3. **Spared nerve injury (SNI) produces hypervigilance — the headline result.** Mice
   received SNI or sham surgery $1$–$2$ weeks before training, $2$–$3$ weeks before testing,
   under blinded surgical status. SNI is a preclinical complex-regional-pain-syndrome
   type-2 model: transection of two of the three distal sciatic-nerve branches innervating
   the hind paw, producing robust long-lasting nociceptive sensitization and neuropathic
   pain-related behaviors (Decosterd & Woolf 2000; King et al. 2009). Two-way ANOVA on
   $n=11$–$13$ per cell gives a significant **odor × surgery interaction**, $F_{1,44}=6.6$,
   $p=0.01$. Post-hoc: under fox urine, SNI mice avoid the short route significantly more
   than sham mice ($p=0.037$, Student's $t$-test).
4. **Interpretation — Crook's hypothesis travels to mammals.** Ethical constraints prevented
   a real predation assay (cf. Crook's free-interaction tank), so the authors propose
   route-choice avoidance as a *fitness proxy*: chronic pain may continually remind the
   organism of elevated predation risk, increase vigilance, and (by extrapolation) increase
   Darwinian fitness.

**Initial takeaway.** Lister et al. is the cleanest single-figure proof that the
**chronic pain → hypervigilance** chain reported in invertebrates extends to mammals:
neuropathic injury shifts the cost–benefit trade-off of an *unrelated* predator-relevant
decision in the direction predicted by an "adaptive smoke-detector" account. The effect is
quantitatively small but statistically reliable, and it is *behavioural* — not just
neurophysiological — which is what makes it a clean target for in-grid replication.

### Phase 2 — Graduate-level deep dive

**Why this design is well-controlled.** The maze pumps odor into *one octant only*, so the
short-route choice is a clean binary readout of route-level fox-urine cost weighting,
unconfounded by global anxiety. A short-route choice with fox urine is a *deliberate
willingness to traverse a predator cue* in exchange for a metabolic reward (food). The pre-
training criterion ($>80\%$ short-route in baseline) ensures every animal has the same prior
on the short route. The four cells of the design — $\{$sham, SNI$\} \times \{$room air,
fox urine$\}$ — let the authors decompose the signal:

$$\Delta_{\text{odor, sham}} = (\%\text{short}_{\text{air, sham}}) - (\%\text{short}_{\text{urine, sham}})$$

$$\Delta_{\text{odor, SNI}} = (\%\text{short}_{\text{air, SNI}}) - (\%\text{short}_{\text{urine, SNI}})$$

Hypervigilance in SNI is the contrast $\Delta_{\text{odor, SNI}} - \Delta_{\text{odor, sham}}$,
i.e. *"how much more does fox urine cost when you also have neuropathic pain?"* — and that
contrast is exactly what the **odor × surgery interaction** ($F_{1,44}=6.6$, $p=0.01$)
measures. The reported *between-cells* $t$-tests on Figure 1C are post-hocs that confirm
the interaction is in the predicted direction.

**A minimal Bayesian threat-detection reading.** Let $z\in\{0,1\}$ be a latent "predator
present" state and let $x$ be the chemosensory evidence emitted by the fox urine. The mouse's
posterior is

$$P(z=1\mid x) \;=\; \frac{P(x\mid z=1)\,P(z=1)}{P(x\mid z=1)\,P(z=1) + P(x\mid z=0)\,P(z=0)}.$$

Route choice is a utility comparison: take the short route iff

$$U_{\text{short}}(x) \;>\; U_{\text{long}}(x), \quad\text{with}\quad U_{\text{short}}(x) \;=\; r_{\text{food}} - C\cdot P(z=1\mid x),$$

where $r_{\text{food}}$ is the reward and $C>0$ is the *subjective cost of being predated*.
Three computational variables can produce the SNI × odor interaction:

1. **Prior shift.** SNI elevates $P(z=1)$ — the mouse acts as if predators are more probable.
2. **Likelihood-precision shift.** SNI multiplies $\log P(x\mid z=1) - \log P(x\mid z=0)$ by
   a precision $\pi>1$ — the mouse weights the same chemosensory log-likelihood ratio more
   strongly. In predictive-coding language this is a **gain on threat-relevant prediction
   error**.
3. **Cost shift.** SNI raises $C$ — the subjective cost of being predated is larger when
   already injured.

Lister's dose-dependent baseline curve (Figure 1B) constrains us: even sham mice avoid the
short route more under high than low concentration, so the *likelihood* itself is graded.
SNI then shifts the dose-response curve up. Without titration of the dose × SNI interaction
(only one fox-urine concentration was tested in the SNI experiment), prior, precision, and
cost cannot be uniquely identified from this dataset — but the existence of the interaction
is consistent with *any* of the three. This is the most useful framing for our purposes:
Lister et al. operationalise *"hypervigilance"* as a *behavioural* shift in route choice that
is mechanistically agnostic between prior, precision, and cost mechanisms.

**Effect-size estimate from the published statistics.** Two-way ANOVA on a binary-percentage
outcome with $n=11$–$13$ per cell ($\sim 48$ total) and an interaction $F_{1,44}=6.6$
corresponds (using $\eta_p^2 = F\cdot \mathrm{df}_1 / (F\cdot \mathrm{df}_1 + \mathrm{df}_2)$)
to

$$\eta_p^2 \;=\; \frac{6.6 \cdot 1}{6.6 \cdot 1 + 44} \;\approx\; 0.130, \qquad \text{Cohen's } f \;=\; \sqrt{\frac{\eta_p^2}{1 - \eta_p^2}} \;\approx\; 0.387.$$

This is a *medium* interaction effect — substantial but not enormous — consistent with the
small absolute group sizes and the noisy nature of route-choice readouts. For us, this is the
right reference point: any in-grid replication of "chronic-pain hypervigilance" should be
expected to produce an interaction effect of similar magnitude, not a dramatic flip.

**What the paper does *not* claim.** Lister et al. are explicit about the limits:

- *Fitness was not measured*; route avoidance is treated as a "tractable proxy". The
  step from "vigilance" to "fitness" is deferred to Crook et al. (2014).
- *No neural recordings* — this is a behavioural Correspondence. Mechanism (amygdala vs PAG
  vs cortical priors) is not addressed. Any inference about whether the SNI effect reflects
  a *prior shift* vs *precision modulation* vs *value-of-life shift* is unresolved.
- *Sex was not reported* in the main text — only the SNI procedure is described, which is a
  known caveat for mouse pain research where sex × pain-mechanism interactions are large.

### Relevance to project

- **Cleanest mammalian behavioural target our agent can mimic.** Route choice between a
  short-and-risky and a long-and-safe path under a state-dependent threat cue is *almost
  literally* a grid-world layout. The $\Delta_{\text{odor, SNI}} - \Delta_{\text{odor, sham}}$
  interaction is a target shape for the behavioural metric an injured / nociceptive-
  signal-active agent should produce relative to the same agent without injury, when a
  predator-relevant cue is in the short route.
- **Hypervigilance is mechanistically agnostic — a feature, not a bug.** The Bayesian
  decomposition above shows that the same behavioural pattern can be produced by (i) a
  prior shift on $P(\text{predator})$, (ii) a precision shift on the likelihood, or (iii) a
  cost shift on subjective predation cost. **This maps directly onto our FiLM precision
  head's freedom of action**: the head can implement (ii) by gaining up predator-feature
  prediction errors, but the same observable signature could be implemented as a prior shift
  inside the world model. Lister et al. thus give us *cover* to implement hypervigilance as
  precision-modulation in our agent, with the understanding that we cannot claim the mouse
  is doing the same thing — only that the *behavioural construct* is consistent.
- **Construct-validity caveat for our "pain-like" claim.** SNI is a peripheral neuropathic
  injury — the nociceptive sensitization is *real and chronic*, not a paper construct.
  Lister et al. are careful to call the readout "hypervigilance" and to defer the fitness
  claim. We should mirror that discipline: any "pain-like" behavioural claim we make for our
  agent should be stated as *"a hypervigilance signature consistent with the SNI literature"*,
  not as *"the agent is in pain"*.
- **Direct dose-response handle.** Lister's Figure 1B (concentration × short-route)
  suggests an obvious in-grid extension we have not run: a **predator-cue intensity sweep**
  $\times$ **injury-signal on/off**. If our FiLM-modulated agent shows a multiplicative
  interaction (steeper dose-response under injury), that's a clean replication of Figure 1
  in silico. If it only shifts the intercept, the mechanism is closer to a prior shift than
  to a precision/likelihood shift.
- **Cross-paper hook.** Lister 2020 cites Crook 2014 explicitly and frames itself as the
  mammalian extension. With Walters (2023, Section 6), this completes a "predator-driven
  nociception cluster" — see *Cross-paper synthesis hooks* below.

> **Recommended follow-up referrals.** `professor-pain-modeling` to settle whether
> "hypervigilance via route avoidance" is acceptable as a pain-like behavioural construct
> for our agent. `professor-bayesian-brain` to formalise the prior-vs-precision-vs-cost
> identifiability question — the right experimental design (concentration × SNI factorial)
> would let us identify which mechanism we have implemented.

---

## 4. Mahajan, Dayan & Seymour (2026) — Homeostasis after injury: intertwined inference and control

**Citation key.** Mahajan, P., Dayan, P., & Seymour, B. (2026). Homeostasis after injury: How
intertwined inference and control underpin post-injury pain and behaviour. *PLOS Computational
Biology*, **22**(1): e1013538. https://doi.org/10.1371/journal.pcbi.1013538. Code:
https://github.com/PranavMahajan25/InjuryPOMDP. Source: local PDF (17 pp.) at
`docs/project/references/computational_models_of_pain/sources/Mahajan et al. 2026 - Homeostasis after injury - How intertwined inference and control underpin post-injury pain and behaviour.pdf`.

> Note: prior placeholder labeled this paper "Mahajan & Seymour 2024". Correct citation is
> Mahajan, Dayan & Seymour (2026), PLoS Comp Biol; preprint dates from May 2025. Updated
> throughout. The paper is also the **first concrete computational realisation of the
> control-theoretic framing in Seymour, Crook & Chen (2023)** — Section 5 of this review.

### Phase 1 — Foundational overview

**Introduction.** Seymour, Crook & Chen (2023; Section 5) had argued that the brain
represents the *state* of an injury and uses that representation to organise post-injury
behaviour — but the 2023 paper is conceptual. Mahajan, Dayan & Seymour (2026) deliver the
formal model: a **partially observable Markov decision process (POMDP)** in which the true
injury state $s$ is a hidden binary variable (healthy / injured), and the brain only has
access to noisy observations $o$ derived from sustained nociceptor firing, autonomic
signals, exteroception, and prior expectations. Tonic pain is *operationalised* as the
posterior belief $b_t(s=1)$ that the agent is currently injured. Phasic pain is the
belief-weighted negative reinforcement signal accrued by injury-investigation actions.
Chronic pain is what happens when this Bayes-optimal inference loop *gets stuck*.

**Key findings.**

1. **A normative explanation for *why we probe injuries despite it hurting*.** With a small
   action set $\mathcal{A} = \{a_{\text{act}}, a_{\text{r\&r}}, a_{\text{que}}, a_{\text{nul}}\}$
   — commit to demanding activity, rest & recuperate, *investigate the injury*, do nothing —
   and binary states $s\in\{0,1\}$, value iteration on the belief MDP makes
   $a_{\text{que}}$ the optimal action over a wide *uncertain* belief band $b_t \in
   [b_{\text{lo}}, b_{\text{hi}}]$. The agent voluntarily incurs phasic pain
   ($r(s=1, a_{\text{que}}) = -4$) because the **value of information** about $s$ outweighs
   the immediate cost when $b_t$ is uncertain. This is a single, principled account for
   apparently paradoxical clinical behaviours (rubbing the injured area, bending a sore back,
   probing a joint) that Gate Control Theory only handles for *contact* exploration.
2. **Two computationally identifiable failure modes for chronic pain.**
   - **(a) Information restriction.** When phasic pain $|r(s=1, a_{\text{que}})|$ is raised
     from $4$ to $16$, the value of $a_{\text{que}}$ falls below the value of the
     low-information null action $a_{\text{nul}}$. The agent *prefers* the cheap-but-
     uninformative action; beliefs no longer update; even though the *true* state has gone
     to $s=0$ (healed), the belief stays elevated and the policy keeps choosing
     $a_{\text{r\&r}}$. This is a fully formal version of the **Fear-Avoidance** account of
     pain chronification (Vlaeyen & Linton 2000; Crombez et al. 2012).
   - **(b) Aberrant priors.** A wrong starting belief $b_0$ produces two pathologies:
     *underestimating* injury when truly injured ($s=1$ but $b_0$ near $0$) leads to
     excessive cumulative phasic pain from too much investigation; *overestimating* when
     recovered ($s=0$ but $b_0$ near $1$) leads to a stable wrong attractor — persistent
     incorrect tonic pain even though the periphery is no longer signalling anything. This
     is the formal version of *catastrophising* / self-fulfilling-prophecy accounts (Jepma
     et al. 2018).
3. **Chronic pain as "wrong problem / wrong solution / wrong environment".** The paper maps
   its two failure modes onto a tripartite diagnostic taxonomy from Huys et al. (2015):
   aberrant priors = *solving the wrong problem*; excessive avoidance / habit = *solving the
   right problem with the wrong solution*; once-adaptive priors-now-wrong = *right problem
   right solution wrong environment*. This ladder maps onto **computational nosology**.

**Initial takeaway.** Mahajan, Dayan & Seymour give us the cleanest, simplest computational
account in the corpus of how *the same Bayesian inference machinery that adaptively probes
an injury* can, with a single change in the cost-of-investigation parameter, *lock the
agent into a chronic-pain attractor* even when the periphery has healed. **It is the only
paper in the corpus that formally distinguishes adaptive tonic pain from maladaptive chronic
pain by a difference in a model parameter, not by a difference in narrative framing.**

### Phase 2 — Graduate-level deep dive

**POMDP scaffolding.** The internal environment is the standard tuple
$\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{O}, \mathcal{R} \rangle$ with
$\mathcal{T}(s, a, s') = P(s'\mid s, a)$, $\mathcal{O}(s', a, o) = P(o\mid a, s')$, and
internal feedback $r(s, a)$. The agent maintains a **belief state**

$$b_t(s) \;=\; P(s_t = s \mid h_t, b_0), \qquad h_t = \{a_0, o_1, \ldots, o_{t-1}, a_{t-1}, o_t\}, \tag{1}$$

which is a sufficient statistic for the action-observation history (Smallwood & Sondik
1973). The **belief update** follows Bayes' rule applied to the joint $(s, o)$:

$$b_t(s') \;=\; \tau(b_{t-1}, a_{t-1}, o_t)(s') \;=\; \frac{\mathcal{O}(s', a_{t-1}, o_t)\;\sum_{s\in\mathcal{S}} \mathcal{T}(s, a_{t-1}, s')\, b_{t-1}(s)}{P(o_t \mid b_{t-1}, a_{t-1})}, \tag{2}$$

where the denominator is the marginal $P(o_t\mid b_{t-1}, a_{t-1}) = \sum_{s'} \mathcal{O}(s', a_{t-1}, o_t)\sum_s \mathcal{T}(s, a_{t-1}, s')b_{t-1}(s)$
acting as the normaliser.

**No-transition simplification.** In their simulations the true state is held fixed during
an episode (no recovery / worsening within the episode), so $\mathcal{T}(s,a,s') =
\mathbb{1}[s=s']$. Eq. (2) collapses to

$$b_t(s) \;=\; \frac{\mathcal{O}(s, a_{t-1}, o_t)\, b_{t-1}(s)}{P(o_t\mid b_{t-1}, a_{t-1})}. \tag{9}$$

This is the **Tiger POMDP** (Kaelbling, Littman & Cassandra 1998) generalised so that the
information-gathering action $a_{\text{que}}$ is itself belief-state-dependent in cost.

**Belief-feedback function.** The expected immediate reinforcement on belief states is

$$\rho(b_t, a_t) \;=\; \sum_{s\in\mathcal{S}} b_t(s)\, r(s, a_t). \tag{4}$$

For a binary state, with $b = b_t(s=1)$ and $1-b = b_t(s=0)$,

$$\rho(b, a) \;=\; b\,r(s=1, a) + (1-b)\,r(s=0, a),$$

which is a *line* in $b$ for each fixed $a$ (the curves in Fig. 2A: $a_{\text{act}}$ slopes
steeply down with $b$, $a_{\text{r\&r}}$ slopes up). The intersection of those two lines
defines the belief threshold at which the agent indifferent between activity and rest:

$$b^*_{\text{act,r\&r}} \;=\; \frac{r(0, a_{\text{act}}) - r(0, a_{\text{r\&r}})}{[r(0, a_{\text{act}}) - r(0, a_{\text{r\&r}})] + [r(1, a_{\text{r\&r}}) - r(1, a_{\text{act}})]}.$$

Plugging in their utilities ($r(0, a_{\text{act}}) = +100$, $r(0, a_{\text{r\&r}}) = -100$,
$r(1, a_{\text{act}}) = -400$, $r(1, a_{\text{r\&r}}) = +100$):

$$b^*_{\text{act,r\&r}} \;=\; \frac{100-(-100)}{[100-(-100)] + [100-(-400)]} \;=\; \frac{200}{200 + 500} \;=\; \frac{200}{700} \;\approx\; 0.286.$$

So the *one-step* indifference threshold sits around $b\approx 0.29$. Below that, doing the
risky activity is one-step optimal; above, resting is. The information action $a_{\text{que}}$
matters *only in the band* where the difference between these one-step values is smaller
than the expected information gain — exactly the basin where the optimal policy in their
Fig. 3 selects $a_{\text{que}}$.

**Bellman recursion on the belief MDP.** A POMDP is reducible to a fully observable belief
MDP whose state is $b\in\Delta(\mathcal{S})$. The value function obeys

$$V^\pi(b) \;=\; \sum_{a\in\mathcal{A}} \pi(b, a)\!\left[\rho(b, a) + \gamma\sum_{o\in\mathcal{O}} P(o\mid b, a)\,V^\pi(\tau(b, a, o))\right], \tag{5}$$

with optimal policy $\pi^*(b) = \arg\max_\pi V^\pi(b)$ and Bellman-optimal value

$$V^*(b) \;=\; \max_{a\in\mathcal{A}}\!\left[\rho(b, a) + \gamma\sum_{o\in\mathcal{O}} P(o\mid b, a)\,V^*(\tau(b, a, o))\right]. \tag{7}$$

The optimal Q-function is

$$Q^*(b, a) \;=\; \rho(b, a) + \gamma\sum_{o\in\mathcal{O}} P(o\mid b, a)\,V^*(\tau(b, a, o)). \tag{8}$$

A classic result (Sondik 1971; Smallwood & Sondik 1973): $V^*$ over the belief simplex is
**piecewise-linear and convex (PWLC)** for finite-horizon problems and approached by PWLC
for infinite-horizon discounted problems. The convexity is what makes the *value of
information* land between two terminating-action lines: the parabolic-looking $a_{\text{que}}$
curve in their Fig. 3 is the upper PWLC envelope of value functions across observation
realisations, sitting *above* the lower envelope of the two terminal lines exactly in the
band of belief uncertainty.

**Algorithm.** They solve (7) by **belief-grid value iteration** (Lovejoy 1991): discretise
$b\in[0,1]$ in steps of $0.01$ and the observation space $o\in(0, 100]$ in steps of $1$,
take the observation function as known, and iterate (7) to fixed point. Maximum trial-
horizon per episode = 100; results averaged over 1000 episodes; $\gamma = 1$ (no temporal
discount within an episode).

**Pain operationalisation — equations 10 and 11.** This is the critical formalisation for
our purposes:

$$\boxed{\;\text{Phasic pain} \;\propto\; \sum_{s\in\mathcal{S}} b(s)\,\zeta(s, a),\;} \tag{10}$$

where $\zeta(s, a)$ is *only the pain-relevant* component of $r(s, a)$ (excludes opportunity
costs, resource loss, etc.). In this paper $\zeta(s, a_{\text{que}}) = r(s, a_{\text{que}})$
since $a_{\text{que}}$'s only cost is nociceptive.

$$\boxed{\;\text{Tonic pain} \;\propto\; b(s = 1).\;} \tag{11}$$

Tonic pain is **the posterior belief that you are injured**. Phasic pain is the
**belief-weighted expected nociceptive cost of an action**. Chronic pain is what (10)–(11)
look like when (2) is starved of informative observations.

**Information restriction — quantitative reading.** Define information gain of an action $a$
in belief $b$ as the expected KL divergence from prior $b$ to posterior $b'$:

$$\mathrm{IG}(b, a) \;=\; \mathbb{E}_{o\sim P(\cdot\mid b, a)}\!\bigl[\,\mathrm{KL}\bigl(\tau(b, a, o)\,\|\,b\bigr)\,\bigr].$$

For their high-information observation channel ($a_{\text{que}}$ in injured state: Gaussian
with mean offset $15$, $\sigma = 15$), $\mathrm{IG}$ is sharp and concentrated around $b
\approx 0.5$. For their low-information channel ($a_{\text{nul}}$: mean offset $5$, $\sigma=
30$), $\mathrm{IG}$ is small almost everywhere. The agent's net benefit of $a_{\text{que}}$
over $a_{\text{nul}}$ in any belief is

$$\Delta Q(b) \;=\; Q^*(b, a_{\text{que}}) - Q^*(b, a_{\text{nul}}) \;\approx\; \underbrace{\gamma\,[\,V^*(\tau_{\text{HI}}(b)) - V^*(\tau_{\text{LI}}(b))\,]}_{\text{info-gain bonus}} \;-\; \underbrace{|\,b\cdot r(1, a_{\text{que}}) - b\cdot r(1, a_{\text{nul}})\,|}_{\text{phasic-pain cost}}.$$

Increasing $|r(1, a_{\text{que}})|$ from $4$ to $16$ flips the sign of $\Delta Q$ in the
middle belief band — empirically (their Fig. 5A→5B) the $a_{\text{que}}$ curve is pushed
below the $a_{\text{nul}}$ curve, and the policy switches. This is the **chronicity bifurcation**.

**Aberrant-prior dynamics — a Markov-attractor reading.** Treat the belief sequence under a
fixed policy $\pi$ as a stochastic process on $[0,1]$. With non-zero $\mathrm{IG}$, this
process is ergodic and its stationary distribution concentrates at the true state ($b\to 0$
when $s=0$ and $b\to 1$ when $s=1$). With degenerate $\mathrm{IG}$ (after the chronicity
bifurcation), the process is *not* ergodic in finite time and the stationary distribution
*reflects* the prior $b_0$. Their Fig. 6B shows this directly: the stationary trajectory of
$b_t$ tracks the starting $b_0$ band rather than collapsing to $0$. **In our language: the
agent has a *posterior collapse* to the prior because the likelihood ratio is too weak to
override it.**

**Mapping the failure modes to clinical labels.**

| Mahajan et al. failure | Corresponding clinical narrative | Identifying parameter |
|---|---|---|
| Information restriction | Fear-avoidance chronic pain (Vlaeyen 2000) | high $|r(1, a_{\text{que}})|$ relative to $\mathrm{IG}$ |
| Underestimating prior at $s=1$ | Pavlovian over-confident exploration (delayed help-seeking) | low $b_0$ |
| Overestimating prior at $s=0$ | Catastrophising / nociplastic chronicity (Fitzcharles 2021) | high $b_0$ |
| (Alluded) Pavlovian bias on $a_{\text{r\&r}}$ | Habitual/excessive avoidance | structural |

**Limitations stated by the authors.** No within-episode state transitions (so no actual
recovery dynamics modelled); utilities are hand-set rather than derived from physiology;
single observation channel (no multisensory integration); no explicit precision parameter
on the likelihood; no neural model. The authors flag these as next-step targets and
explicitly mention **ACL tear** as their planned empirical test case.

### Relevance to project

- **This is the formal backbone we should be aiming our agent at.** Among all six papers
  in this corpus, Mahajan, Dayan & Seymour 2026 is the only one with (i) a fully specified
  POMDP, (ii) explicit equations for "tonic pain" and "phasic pain" as functions of belief,
  and (iii) two well-identified parametric pathways from acute to chronic pain. **Our
  GridWorld-Pain agent should reproduce, in some form, the chronicity bifurcation: a single
  parameter (cost of information-gathering, or precision on nociceptive likelihood) whose
  variation flips the agent's behaviour from adaptive probing to information-restricted
  avoidance.**
- **Our FiLM precision head plugs in *exactly* where their model is silent.** Mahajan et al.
  use a *fixed* observation likelihood $\mathcal{O}$. The natural project-direction
  upgrade is to make the observation likelihood **precision-modulated** — i.e. multiply the
  log-likelihood by a state-dependent precision $\pi(b, \text{interoceptive state})$ that
  the FiLM head outputs. This is a one-line change to (2)/(9): replace
  $\mathcal{O}(s, a, o)$ by $\mathcal{O}(s, a, o)^{\pi}$ (renormalised). This unifies their
  model with Büchel's predictive-coding account of placebo analgesia (cited as their ref
  [7]) and gives our FiLM head a principled job: **it sets the precision on the injury-
  belief likelihood**.
- **Tonic = belief, phasic = belief-weighted negative reinforcement.** This is the
  **operational definition** we should adopt across the project. Anywhere we say "tonic pain
  signal" in code or in `docs/develop/`, we should be referring to a quantity proportional
  to $b_t(s=1)$. Anywhere we say "nociceptive feedback" or "phasic" we should be referring
  to $\sum_s b_t(s)\zeta(s, a)$. This is a strict construct-validity discipline: it ties
  every "pain-like" claim back to a quantity that can be read out from the agent's belief
  module.
- **Two clean target experiments.** (a) **Cost-of-investigation sweep**: vary the cost of an
  injury-probing action and watch for a phase transition in average cumulative tonic pain
  $b_t$ when (true state) = $s=0$. (b) **Aberrant-prior sweep**: initialise $b_0$ with a
  bias and trace the trajectory of $b_t$ vs episode time. If our FiLM-modulated agent shows
  these two qualitative shapes (Mahajan et al. Fig. 5C/D and Fig. 6A/B), we have a
  reproducible chronic-pain analogue in-grid.
- **Construct-validity guardrail.** The paper is careful that *belief over injury state* is
  one operational stand-in for tonic pain among at least three (also: $V^\pi(b)$, or a
  predictive state representation). Our project should pick *one* of these and stick with
  it — almost certainly Eq. (11) since it is the simplest and the only one for which the
  Mahajan et al. simulations are directly diagnostic.
- **Connects forward to Seymour 2023 and backward to Crook / Lister.** The "wrong problem /
  wrong solution / wrong environment" taxonomy explicitly uses Seymour's control-theoretic
  language. Their information-restriction account is what *would happen to* a Lister-style
  hypervigilant SNI mouse if it kept avoiding the predator-cue corridor long after the
  injury healed: protective behaviour starves the brain of evidence and chronic
  hypervigilance ensues. We should test this in-grid.

> **Recommended follow-up referrals.** `professor-bayesian-brain` for an active-inference
> deep dive — the paper is decision-theoretic-Bayesian but never invokes free energy, and
> a complementary active-inference critique would sharpen our understanding of what
> "precision" should mean in our FiLM head. `senior-developer` to draft an `issue_plan` for
> the Mahajan-style cost-of-investigation sweep on the existing grid (read-only from a
> reviewer perspective; we cannot write that plan ourselves). `professor-rl-bayesian-dl` on
> the question of whether a recurrent actor-critic with sufficient capacity (their footnote
> on Hennig et al. 2023) can recover the same phase transition without an explicit POMDP
> solver — this directly affects whether our Dreamer-V3-based agent can in principle host
> the same dynamics.

---

## 5. Seymour et al. (2023) — Post-injury pain and behaviour: a control theory perspective

**Citation key.** Seymour, B., Crook, R. J., & Chen, Z. S. (2023). Post-injury pain and
behaviour: a control theory perspective. *Nature Reviews Neuroscience*, **24**(6), 378–392.
https://doi.org/10.1038/s41583-023-00699-5. Source: local PDF (15 pp.) at
`docs/project/references/computational_models_of_pain/sources/Seymour et al. 2023 - Post-injury pain and behaviour - a control theory perspective.pdf`.

> Note: prior placeholder labeled this paper "Seymour 2019". The correct citation is
> **Seymour, Crook & Chen (2023)**, *Nat Rev Neurosci*. The 2019 paper of similar topic is
> Seymour's *Neuron* paper "Pain: a precision signal for reinforcement learning and control"
> (cited as [12] of Mahajan et al. 2026), which is **not** in this corpus. Updated throughout.

### Phase 1 — Foundational overview

**Introduction.** Seymour, Crook & Chen (2023) is the **conceptual parent** of Mahajan,
Dayan & Seymour (2026) (Section 4 of this review). Where Mahajan et al. give a concrete
POMDP, Seymour et al. give the *case* — across 15 pages of *Nature Reviews Neuroscience* —
that post-injury behaviour and persistent pain should be understood as the output of a
**hierarchical optimal-control system with a Bayesian inference module**, in which:

- the **plant** is a continuous estimate of the latent injury state $x(t)$;
- the **observation** $s(t)$ is the multimodal afferent stream (nociceptive, autonomic,
  endocrine, exteroceptive);
- the **control signal** $u(t)$ is generated by descending modulation, neuromodulator
  dynamics, and explicit motor output;
- the **system output** $y(t)$ is post-injury behaviour (rest, hypervigilance, guarding);
- and **chronic pain is what happens when this control loop's information channel collapses**
  — a phenomenon the authors call **information restriction**.

The paper is a *Perspective* rather than an empirical study, but it is the canonical
reference for the entire control-theoretic framing of pain. It is also where the
**insula-hub neural implementation** of injury-state inference is laid out, and where the
explicit *information-restriction model of chronic pain* (their Fig. 4) is first published.

**Key claims (sections, in order).**

1. **Ecological & evolutionary perspectives.** Nociception is conserved across nearly all
   animal phyla. Two functional roles must be distinguished: (a) acute phasic pain — fast
   defensive avoidance, well captured by RL value-learning; (b) **persistent pain** — drives
   recuperative behaviour during healing, which the authors argue is *just as important*
   for fitness. They cite cephalopod (squid, octopus) work (Box 1) — including Crook 2014
   from this corpus — as evidence that persistent pain has adaptive function.
2. **Sociality and resource allocation as constraints.** Whether prolonged recuperation is
   adaptive depends on whether the social group / metabolic budget can buffer reduced
   foraging. McNamara & Buchanan's (2005) optimality model: animals should trade off
   recuperation against immediate-survival actions until the marginal cost of accumulated
   damage outweighs the foraging deficit. **Pain-signalling strategies** (vocalisation, facial
   expression) emerged as a *social* control variable.
3. **Formalising persistent pain as an optimal control problem.** A plant + controller +
   meta-control hierarchy (their Fig. 1a). Two distinguishable failure modes follow from
   *control–inference duality*: failures of the controller (wrong actions) and failures of
   the estimator (wrong beliefs). The authors lean toward **estimation failure** as the
   dominant pathway to chronicity.
4. **Box 2 — control theory primer.** State-space LTI form $\dot{x}=Ax+Bu$, $y=Cx+Du$;
   controllability ↔ predictability duality; cost-rate function $L(s, a)$ to be minimised
   over a horizon; **control-as-inference** identity (Levine 2018 / Attias 2003). This is
   the formal scaffolding Mahajan et al. 2026 then specialise.
5. **Injury state representation.** *Multimodal* (nociception, somatosensory non-noxious,
   autonomic, immune, exteroceptive, sensorimotor) with the **insula** proposed as the
   integrative hub, organised in a posterior → mid → anterior gradient that mirrors a
   primary-sensory → injury-state-representation → meta-cognitive hierarchy.
6. **Control output — three effector domains.** (a) Neuromodulatory bias on motivated
   behaviour (DA, 5-HT) — relief seeking, increased effort cost, risk preference shifts.
   (b) Endogenous pain-sensitivity tuning — *stress-induced analgesia* during the acute
   phase, *peripheral and central sensitization* (allodynia, hyperalgesia) during the
   recovery phase. (c) Autonomic / endocrine / immune coordination.
7. **Closed-loop control & the information-restriction problem.** The same protective
   behaviours that adaptively reduce further damage *paradoxically restrict* the information
   channel about injury resolution — a fundamental exploration–exploitation dilemma whose
   asymmetric costs (under-estimating injury is more dangerous than over-estimating) bias
   the steady-state inference toward *persistent* injury.
8. **Meta-control.** A higher layer monitoring controllability, modulating the cost-rate
   function, and arbitrating between exploration and risk-seeking.
9. **Neural implementation — Fig. 2.** Insula at the centre of an afferent–efferent network
   with brainstem (PAG, parabrachial, RVM), thalamus, ACC, VMPFC, amygdala, hypothalamus,
   sensorimotor cortex (S1, SMA). Fig. 2b shows the posterior → mid → anterior insula
   gradient as the algorithmic-anatomical correlate of the inference hierarchy.
10. **Oscillatory neurodynamics — Fig. 3.** **Theta (4–9 Hz)** is the proposed long-range
    integrative carrier — augmented in chronic pain across thalamus, S1, posterior insula
    (Fig. 3a, neuropathic-pain MEG); pre-stimulus theta in insula gates pain perception in
    healthy individuals (Fig. 3c); travelling theta waves along the insula's anterior–
    posterior axis (Das et al. 2022); brain-computer-interface up-/down-training of insular
    theta bidirectionally modulates pain discrimination. **Gamma** carries phasic
    nociceptive responses locally.
11. **Information-restriction model of chronic pain — Fig. 4.** Four parallel routes to
    chronicity: **(i) maladaptive learning** (avoidance starves the learner), **(ii)
    maladaptive model** (descending facilitation + central sensitization noise the
    likelihood), **(iii) maladaptive integration** (lesions / amputation block sensory
    channels), **(iv) maladaptive priors** (pessimistic expectancies bias the posterior).
    Each is a **specific failure mode** of an otherwise normative Bayesian controller.

**Initial takeaway.** This is the *blueprint*: persistent pain is the **output of an
optimal-controller-with-Bayesian-state-estimator** that is **inherently susceptible to
information restriction**. The paper does not give equations beyond Box 2's primer; that
gap is filled by Mahajan et al. 2026.

### Phase 2 — Graduate-level deep dive

**The control-theoretic primer (Box 2 — verbatim formal content).** Continuous-time linear
time-invariant system:

$$\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t), \qquad \mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t),$$

with latent state $\mathbf{x}(t)\in\mathbb{R}^n$, observable output $\mathbf{y}(t)\in\mathbb{R}^N$,
control input $\mathbf{u}(t)\in\mathbb{R}^m$. The eigenvalues of $\mathbf{A}$ govern the
mode-time-constants. **Kalman's controllability criterion**: the LTI system is controllable
iff the controllability matrix $[\mathbf{B}\;\mathbf{AB}\;\mathbf{A^2 B}\;\cdots\;\mathbf{A^{n-1}B}]$
has rank $n$. By the well-known **control–observability duality**, the same algebraic
condition (rank of an analogous matrix) defines observability. This is what Seymour et al.
mean by *"uncontrollability also implies unpredictability"* — a system you cannot drive to
a desired state is also one you cannot uniquely identify from output measurements. This is
the formal hook for their later argument that **information restriction is a controllability
failure that simultaneously prevents the brain from inferring injury resolution**.

**Optimal-control objective.** They write the standard cost-functional

$$\text{Cost}[\mathbf{a}] \;=\; \int_{0}^{T} L(\mathbf{s}(t), \mathbf{a}(t))\, dt \;+\; \rho T, \qquad \dot{\mathbf{s}}(t) = f(\mathbf{s}(t), \mathbf{a}(t)),$$

with $\mathbf{s}(0), \mathbf{s}(T)$ fixed and $\rho > 0$ penalising horizon. The paper notes
two extensions: (i) **risk-aversion penalisation** (e.g. add a variance term) which can
shift the agent toward exploration, and (ii) **finite-horizon vs infinite-horizon
trade-off** which mirrors short-vs-long-term cost.

**Control-as-inference identity.** This is the formal pivot of the paper. The authors invoke
the result (Levine 2018) that the search for an optimal policy in a stochastic dynamical
system is equivalent to a probabilistic inference problem on a graphical model. With a
binary optimality variable $\mathcal{O}_t\in\{0,1\}$ and likelihood
$P(\mathcal{O}_t = 1\mid \mathbf{s}_t, \mathbf{a}_t) \propto \exp\!\bigl(-L(\mathbf{s}_t, \mathbf{a}_t)\bigr)$,
the optimal policy satisfies

$$\pi^*(\mathbf{a}_t \mid \mathbf{s}_t) \;\propto\; \int P(\mathcal{O}_{t:T}=1 \mid \mathbf{s}_t, \mathbf{a}_t, \cdots)\, d\mathbf{s}_{t+1:T},$$

which formalises "**which action is taken given the future is optimal?**" — the inferential
restatement of optimal control. This is exactly the dual that Mahajan et al. 2026 then
implement explicitly via belief-MDP value iteration.

**Pain plant equation (their Fig. 1b in equation form).** Although the paper does not write
this equation, the closed-loop diagram unambiguously specifies it. With a top-down
reference $\mathbf{r}(t)$ (expectation / prior / emotion), bottom-up multimodal afferents
$\mathbf{s}(t)$, controller $\mathbf{u}(t)$, plant state $\mathbf{x}(t)$ (injury, pain),
behavioural output $\mathbf{y}(t)$, and feedback $\mathbf{f}(t)$ (efference copy),

$$\begin{aligned}
\hat{\mathbf{x}}(t) &= \mathbb{E}\!\bigl[\mathbf{x}(t)\mid \mathbf{s}_{0:t},\, \mathbf{u}_{0:t-1},\, \mathbf{r}\bigr] \quad\text{(estimator)} \\
\mathbf{u}(t) &= K\!\bigl(\mathbf{r}(t) - \hat{\mathbf{x}}(t)\bigr) \quad\text{(controller)} \\
\mathbf{y}(t) &= \mathbf{x}(t) + \mathbf{n}(t) \quad\text{(action with sensory noise)} \\
\mathbf{f}(t) &= h(\mathbf{u}(t), \mathbf{y}(t)) \quad\text{(efference copy / feedback)}
\end{aligned}$$

Under linear-Gaussian assumptions $\hat{\mathbf{x}}(t)$ is a **Kalman filter**; the
controller $K$ is a **linear feedback law** producing a Linear-Quadratic-Gaussian (LQG)
controller. Seymour et al. do not commit to this specific form — they want the framework
to cover non-linear, non-Gaussian instances (active inference, POMDPs) — but LQG is the
canonical reading of Fig. 1b.

**Bayesian likelihood with precision modulation — Fig. 1c.** The paper makes one explicit
mathematical commitment: the lower-level multimodal evidence is integrated with prior beliefs
through a generative model $F$ implemented in a recurrent neural circuit, with a
**modulatory / precision-weighting input** that controls how much weight the brain puts on
incoming afferent information vs prior. Schematically,

$$P(\mathbf{x}\mid \mathbf{s}; \pi) \;\propto\; P(\mathbf{x}) \cdot \prod_i P(s_i \mid \mathbf{x})^{\pi_i},$$

where $\pi_i$ is the **precision** assigned to channel $i$. **This is the predictive-coding
equation that our project's FiLM precision head is designed to instantiate.** When $\pi_i$
goes up, channel $i$ dominates the posterior; when it goes down, the prior dominates. The
"maladaptive priors" failure mode of Fig. 4b is exactly the case $\pi_i \to 0$.

**Information-restriction model — Fig. 4 in inference language.** Let $b_t = P(\text{injured}_t \mid h_t)$
be the running posterior over injury, and let $a_t \in \mathcal{A}$ be the action taken.
Information gain of action $a_t$ is

$$\mathrm{IG}(a_t \mid b_t) \;=\; \mathbb{E}_{o\sim P(\cdot\mid b_t, a_t)}\!\bigl[\,H(b_t) - H(b_{t+1})\,\bigr],$$

with $H$ the entropy of the posterior. The four failure modes of Fig. 4b correspond to:

| Failure | Mathematical signature |
|---|---|
| Maladaptive learning | $\mathrm{IG}(a_t)$ collapses because $a_t$ is restricted to a small subset (avoidance) |
| Maladaptive model | $P(o\mid x, a)$ is inflated by central sensitization → likelihood is mis-specified |
| Maladaptive integration | One or more channels $s_i$ are absent (lesion, amputation) → multisensory $P(\mathbf{s}\mid x)$ is mis-specified |
| Maladaptive priors | $P(x_0)$ is biased away from the truth and the likelihood is too weak to override it |

The unifying observation: **all four failure modes degrade the *effective* signal-to-noise
of the likelihood ratio that the agent uses to update $b_t$**. Once that ratio is small, the
posterior tracks the prior. Combined with the *asymmetric cost structure*
$C(\text{miss-injury}) > C(\text{false-alarm-injury})$, the optimal posterior policy
defaults to "remain alarmed" — chronic pain.

**Asymmetric-cost steady state (back-of-envelope).** Suppose the agent classifies $x=1$
("injured") iff $b_t > b^*$ (some threshold). Bayes-optimal $b^*$ given costs
$C_{01} = $ cost of saying "healed" when injured, $C_{10} = $ cost of saying "injured" when
healed satisfies

$$b^* \;=\; \frac{C_{10}}{C_{10} + C_{01}}.$$

If $C_{01} \gg C_{10}$ (the cost of missing a real injury is much greater than the cost of
treating a phantom one), $b^*$ approaches $0$ — the agent declares "injured" on weak
evidence. This is the Bayesian analogue of the *smoke detector principle* (Nesse & Schulkin
2019, cited by Lister et al. 2020 in this corpus): persistent over-detection is the optimal
response to an asymmetric cost structure, but it becomes **chronic pain when the likelihood
gradient is too weak to ever push $b_t$ below $b^*$**.

**Neural implementation — what the paper commits to.** (Fig. 2)

- **Insula = injury-state hub.** Posterior insula = primary multimodal afferent receptor;
  mid insula = injury-state representation + cortico-cortical priors + hypothalamic
  endocrine feed; anterior insula = meta-cognitive layer.
- **Brainstem efferent network.** PAG, parabrachial nucleus, rostral ventromedial medulla
  (RVM) — descending modulation of dorsal-horn nociception; opioidergic / monoaminergic.
- **Frontostriatal-amygdala circuit** for motivational re-valuation (relief seeking,
  punishment-sensitivity gain, risk-preference shift).
- **Hypothalamic** axis for sleep / appetite / immune / endocrine coupling. Specifically
  the **dynorphin–κ-opioid receptor** circuit (Ito et al. 2022) modulates wakefulness and
  vigilance under neuropathic injury — a direct biological substrate for *hypervigilance*.

**Oscillatory neurodynamics — falsifiable predictions.** The theta (4–9 Hz) hypothesis is
the most concrete bet: theta is the carrier frequency for long-range integration in the
insula-centred network during persistent-pain states. If true, three predictions follow:
(i) chronic-pain populations show elevated theta in insula/thalamus/S1 (supported); (ii)
pre-stimulus insula theta predicts pain perception trial-by-trial in healthy participants
(supported, Taesler & Rose 2016); (iii) closed-loop modulation of insular theta should
bidirectionally modulate pain discrimination (supported in BCI work, Taesler & Rose 2021).

**Novel testable predictions stated by the paper.**
(P1) Nociceptive input is **not necessary** for persistent pain — multisensory illusions
should be sufficient.
(P2) Generalised punishment-sensitivity increases after acute injury, and its *magnitude*
predicts longitudinal transition to chronicity.
(P3) Transcranial ultrasound stimulation (TUS) of insula reduces generalised post-injury
behaviours.

### Relevance to project

- **The conceptual scaffolding for our entire project.** Section 5 + Section 4 (Mahajan,
  Dayan & Seymour 2026) jointly define what we are trying to model. Seymour et al. 2023
  gives the *story*; Mahajan et al. 2026 gives the *math*. Our agent's
  injury → hypervigilance → policy-shift chain should be readable as **a special case of
  Fig. 1b** with the FiLM precision head implementing the modulatory input on Fig. 1c, and
  with information-restriction-style failures available as a deliberate ablation.
- **Direct map of FiLM precision head onto the framework.** The FiLM head's job in our
  agent is to set the precision $\pi$ on injury-relevant likelihood channels. This *is* the
  variable that distinguishes "information restriction" from "adaptive vigilance" in the
  Seymour 2023 ontology — a clear functional role. We can now defend the FiLM head as the
  *computational substrate of meta-control* in their hierarchy.
- **Construct-validity discipline.** The paper insists on the **nociception ≠ pain**
  distinction (their ref [2] = Seymour 2019 *Neuron*). Nociception is the afferent signal;
  pain is the *control signal* that governs action selection and learning. Our project
  must mirror this: the input to the agent is nociceptive ("there is damage at site X");
  the output relevant to pain-like behaviour is the agent's *action policy shift* under
  that input, mediated by the injury-state belief.
- **Information-restriction is the in-grid experiment.** Of the four failure modes, the
  one most directly testable in our grid-world is **maladaptive learning** (avoidance
  starves the learner). Concretely: train an agent that has learned to forage, then
  introduce a "predator-relevant cue" that the agent can avoid by taking a long route
  (cf. Lister et al. 2020 in this review). Inject an "injury" signal that raises FiLM
  precision on predator-feature channels. **Predict: the agent persists in avoidance long
  after the predator-cue distribution shifts away — chronic hypervigilance with no
  ongoing peripheral driver.** This would be a clean replication of Seymour's Fig. 4a in
  silico.
- **Theta hypothesis is *not* directly applicable to our agent** — we don't model neural
  oscillations — but it is a useful disciplining constraint on what we *don't* claim. We
  should not claim our agent recapitulates the *neural* account; we should claim only the
  *computational-level* account (Marr level 1). This is exactly the discipline Mahajan et
  al. 2026 explicitly take (their footnote on Marr levels).
- **The four failure modes give us four ablation directions.** Each maladaptive type maps
  to a clean code-change recipe — *for delegation to senior-developer, not for us to write*:
  - Maladaptive learning: restrict the action space when injury signal is on.
  - Maladaptive model: inject correlated noise in the belief update (likelihood mis-spec).
  - Maladaptive integration: drop one observation channel (vision-only, no nociception).
  - Maladaptive priors: bias the prior $b_0$ toward "injured".

> **Recommended follow-up referrals.** `professor-bayesian-brain` for an active-inference
> re-derivation of Fig. 1c — specifically whether their schematic precision-weighting is
> the same object as Friston-style precision on free-energy gradients (the paper cites
> Smith et al. 2022's active-inference tutorial but doesn't commit). `professor-pain-modeling`
> for the construct-validity arbitration between Seymour's "control plant" framing and
> traditional sensory-physiological pain definitions — does the project gain or lose by
> staking out the control-theoretic position? `professor-neuromodulation` for the descending
> modulatory pathway claims (PAG, RVM, opioid / monoaminergic) — can our agent's "modulatory
> input" channel be defended as biologically plausible if it is implemented as a single
> FiLM scalar rather than a multi-pathway signal?

---

## 6. Walters et al. (2023) — Persistent nociceptor hyperactivity as a painful evolutionary adaptation

**Citation key.** Walters, E. T., Crook, R. J., Neely, G. G., Price, T. J., & Smith, E. St J.
(2023). Persistent nociceptor hyperactivity as a painful evolutionary adaptation. *Trends
in Neurosciences*, **46**(3), 211–227.
https://doi.org/10.1016/j.tins.2022.12.007. Source: local PDF (17 pp.) at
`docs/project/references/computational_models_of_pain/sources/Walters et al. 2023 - Persistent nociceptor hyperactivity as a painful evolutionary adaptation.pdf`.

> Note: prior placeholder labeled this "Walters 2019" / *Frontiers*. The correct citation is
> the *Trends in Neurosciences* 2023 review. Updated throughout. Note that Crook (2014;
> Section 1 of this review) is the primary empirical evidence cited, and Walters is a
> co-author on Crook 2014 — Section 1 and Section 6 are conceptually paired.

### Phase 1 — Foundational overview

**Introduction.** The default clinical assumption is that **chronic neuropathic pain** and
the **persistent nociceptor hyperactivity** that drives it are *maladaptive* — a stuck-on
alarm with no biological function. Walters and colleagues challenge this assumption with a
**comparative-physiology argument**: persistent nociceptor hyperactivity is conserved across
multiple distantly-related phyla (gastropod molluscs, cephalopod molluscs, mammals
including humans), uses **shared molecular machinery** (cAMP/PKA/CREB, ERK/MNK/eIF4E)
across $\sim 600$ million years of divergence, and produces a **survival benefit** that
has been directly measured (Crook 2014, Section 1) or strongly implied (Lister 2020,
Section 3) in predator–prey interactions. They propose a unifying functional account:
**persistent nociceptor hyperactivity drives a state of behavioural hypervigilance that
protects injured animals during a window of elevated predation risk**.

**Key claims.**

1. **Hyperexcitability ≠ hyperactivity (Box 1).** Hyperexcitability is *electrical*
   (lower rheobase, more negative threshold). Hyperactivity is the *physiological state of
   increased AP discharge* — what actually feeds the CNS. Many things can drive
   hyperactivity: hyperexcitability, but also hypersensitivity to extrinsic excitatory
   signals, disinhibition (loss of inhibitory inputs), or hypersensitivity to normal body
   temperature. **Pain-relevant evolution acts on hyperactivity, not necessarily on
   hyperexcitability.**
2. **Two complementary sources of post-injury hyperactivity (Fig. 2).** **Extrinsic**:
   DAMPs ($K^+$, $H^+$, ATP, glutamate, heat-shock proteins from dying cells), PAMPs
   (LPS, hemolysin from microbes), pro-inflammatory cytokines (TNF-α, IL-1β, CXCL1, CCL2,
   MIF), biogenic amines (5-HT, histamine, NE), eicosanoids, peptides (bradykinin), growth
   factors (NGF, GDNF) — *and* **disinhibition** by reduced inhibitory cytokines (IL-10),
   endogenous opioids, somatostatin, endocannabinoids, GABA. **Intrinsic** (cell-autonomous):
   six biophysical loci can promote hyperexcitability — increased sensory generator
   potentials (more TRP channels), increased membrane resistance, depolarised resting
   membrane potential (RMP), hyperpolarised AP threshold (NaV channel-shift), increased AP
   discharge rate (more NaV, fewer KV), increased depolarising spontaneous fluctuations
   (DSFs); plus three more for sensitivity tuning.
3. **Comparative survey (Table 1).** Persistent nociceptor hyperactivity is **functionally
   important** in *Aplysia* (snail), *Doryteuthis* (squid), and rodents/humans, but **absent
   or weak** in *C. elegans* (short lifespan, tiny body) and *Heterocephalus glaber* (naked
   mole-rat: subterranean, low predation, fewer C-fibers, NaV1.7 acid-block mutation, TrkA
   hypofunction). The pattern *predicts*: hyperactivity is selected when (a) lifespan is
   long enough for chronic injury, (b) predation risk is real, (c) impairment is detectable
   by predators. *Drosophila* is intermediate — its nociceptor cell bodies sit
   subepidermally so they cannot survive serious peripheral injury; central sensitisation
   substitutes.
4. **Direct-evidence link to Crook 2014 (their Fig. 3A).** Walters et al. cite Crook's
   squid–sea-bass survival data as the **only direct fitness measurement** of nociceptive
   sensitization in any species: $\sim 75\%$ uninjured survival vs $\sim 45\%$ injured
   (sensitised) vs $\sim 20\%$ injured-with-MgCl₂-block (sensitisation prevented).
5. **Indirect-evidence link to Lister 2020 (their Fig. 3B).** SNI mice avoid a fox-urine
   short-route more than sham mice (Lister et al. 2020, Section 3 of this review) — the
   mammalian **behavioural** (not fitness) extension of the squid result.
6. **Speculative human extension (Fig. 3C).** Hominin lifespan, slow-healing injuries, and
   intense social cooperation produce a unique selective regime: persistent nociceptor
   hyperactivity may be **even more strongly selected** in humans because it (i) drives
   help-seeking communication, (ii) elicits aid from kin, (iii) protects the injured
   individual during a long recovery. The flip side: humans may be **more susceptible to
   chronic pain** than rodents because their nociceptors are *more polymodal* and *more
   intrinsically excitable* (every adult human nociceptor expresses TrkA; nearly all
   express TRPV1; higher density of TTX-sensitive Na currents; minimal use-dependent
   inactivation).
7. **Shared molecular machinery — a deep-time argument (Fig. 4).** Two pathways
   conserved across $\sim 600\;\mathrm{Myr}$:
   (a) **Ras → Raf → MEK → ERK → MNK → eIF4E** axis controlling activity-dependent mRNA
   translation in axons and cell bodies.
   (b) **G$_s$-coupled GPCR → adenylyl cyclase → cAMP → PKA → CREB** axis controlling ion
   channel function and gene expression.
   Both pathways have *direct pharmacological evidence* in *Aplysia* and in mouse/rat
   nociceptors. This is the molecular signature of a deeply conserved adaptation.
8. **The cell-body location of nociceptor hyperactivity is functionally important.**
   Mammalian nociceptor cell bodies (DRG) are protected by bone and dura, and they integrate
   intrinsic + extrinsic + axonal-transport signals from the periphery. Crucially: human
   amputees with phantom-limb pain show **complete pain relief from lidocaine delivered to
   the DRG** at concentrations that block local AP generation but spare conduction (Vaso et
   al. 2014). This is direct human evidence that *cell-body-generated spontaneous activity*
   is the substrate of chronic pain after amputation.

**Initial takeaway.** Walters et al. 2023 is the **integrative biological review** of the
corpus. Where Mahajan, Dayan & Seymour 2026 (Section 4) gives the formal computational
model, Walters et al. give the **multi-species, multi-millennium evidence base** that the
phenomenon being modelled is *real, conserved, adaptive, and mechanistically specific*.
**Sensitised nociceptor activity is not a bug; it is a hypervigilance signal whose
parameters have been tuned by predation selection.**

### Phase 2 — Graduate-level deep dive

**The hyperexcitability-vs-hyperactivity distinction — a quantitative framing.** Define a
nociceptor's instantaneous firing probability per unit time as $\lambda(t)$. The
electrophysiological state is captured by three quantities:

- $V_{\text{th}}$ — voltage threshold for AP generation
- $V_{\text{rest}}$ — resting membrane potential
- $\sigma_V(t)$ — root-mean-square amplitude of depolarising spontaneous fluctuations (DSFs)

Under a Gaussian DSF model, the spontaneous firing rate is approximately

$$\lambda(t) \;\approx\; \nu_0 \cdot \mathrm{erfc}\!\left(\frac{V_{\text{th}}(t) - V_{\text{rest}}(t)}{\sqrt{2}\,\sigma_V(t)}\right),$$

where $\nu_0$ is the channel-opening attempt rate and $\mathrm{erfc}$ is the complementary
error function. **Hyperexcitability** is a reduction in the *gap* $V_{\text{th}} - V_{\text{rest}}$
(channel-shift), e.g. by NaV1.7 hyperpolarisation. **Increased DSFs** is an increase in
$\sigma_V$. **Hyperactivity** is an increase in $\lambda(t)$, which (per the equation above)
can result from any of the three channels. Walters et al.'s Fig. 1D and Box 1 thus correspond
directly to the three independent axes that determine spontaneous firing — **the brain's
"injury signal" comes from a sum/AND of these axes, not from any one of them alone**.

**Walters et al.'s comparative phylogenetic table — formalised.** Let $S$ index species,
$h_S \in \{0, 1\}$ denote presence of injury-induced persistent nociceptor hyperactivity,
$f_S$ denote the existence of a measured fitness benefit, $m_S$ denote conserved molecular
machinery (Ras-MNK and/or cAMP-PKA-CREB). From Table 1:

| $S$ | $h_S$ | $f_S$ (direct/indirect) | $m_S$ | Predation pressure |
|---|---|---|---|---|
| *C. elegans* | $0$ | $-$ | unknown | low (short lifespan) |
| *Drosophila* | unclear (central) | unknown | unclear | central pathway |
| *Aplysia* | $1$ | plausible | yes (Ras-MNK + cAMP-PKA) | yes (Navanax) |
| *Doryteuthis* (squid) | $1$ | **direct** | unknown | yes (sea bass) |
| *Mus/Rattus* | $1$ | indirect | yes (Ras-MNK + cAMP-PKA) | yes |
| *Heterocephalus* (naked mole-rat) | $0$ | $-$ | likely absent | very low (subterranean) |
| *Homo sapiens* | $1$ | plausible | likely yes | low extant; high ancestral |

The phylogenetic distribution of $h_S = 1$ does *not* track simple homology (the trait is
absent in *C. elegans* and naked mole-rat despite both being deep within nociceptor-bearing
clades) but *does* track ecological need (predation × lifespan × healing-time). Walters et
al.'s argument is that this is the signature of *parallel adaptive evolution* of a complex
trait, not a single ancestral character — which is why the same molecular pathways
(Ras-MNK, cAMP-PKA-CREB) are recruited in *Aplysia* and rodents despite their extreme
phylogenetic divergence.

**Cell-body-as-integrator argument — Bayesian reading.** Mammalian nociceptors have central
cell bodies in DRG that receive (i) action potentials from peripheral terminals, (ii)
axonal-transport-mediated chemical signals from injury sites, (iii) systemic circulating
cytokines/DAMPs/PAMPs (DRG lacks an effective vascular permeability barrier), (iv)
satellite-glial-cell signalling, (v) infiltrating immune cell signalling. Treat each as an
observation channel $o_i$ informing a latent injury-state $x \in \{0, 1\}$ with likelihood
$P(o_i \mid x)$. Under conditional independence of channels given $x$, the log posterior is

$$\log \frac{P(x=1\mid o_{1:N})}{P(x=0\mid o_{1:N})} \;=\; \log \frac{P(x=1)}{P(x=0)} + \sum_{i=1}^{N} \log\frac{P(o_i\mid x=1)}{P(o_i\mid x=0)}.$$

The DRG is **anatomically positioned to be the physical substrate of this sum**: it can
spike (hence transmit "injury here, intensity $\lambda$") only when the integrated likelihood
ratio is high enough. This is a clean computational reason why evolution would put the
spontaneous-activity-generator in the cell body, not (only) in the peripheral terminal —
and it is consistent with Mahajan et al.'s POMDP belief-update framework (Section 4) where
the agent integrates multimodal observations into a single belief.

**Spontaneous activity ≠ noise — three biophysical alterations co-vary in injured
nociceptors.** Walters et al. note (their § "Indirect evidence consistent with persistent
nociceptor hyperactivity being adaptive in mammals") that, in rodent nociceptors after
spinal cord injury or peripheral nerve injury, **all three** mechanisms by which a cell can
spontaneously cross AP threshold are simultaneously altered: $V_{\text{rest}}$ depolarises,
$V_{\text{th}}$ hyperpolarises, and $\sigma_V$ (DSFs) increases. The probability of three
independent random alterations conspiring to push a quiescent cell into spontaneous firing
is small; the conjunction is therefore strong evidence for a *coordinated, selected* state.
The same triple alteration is observed in DRG neurons isolated from neuropathic-pain
*human cancer patients* (North et al. 2019; their refs [84, 104]) — an empirical
human-rodent parallel that is among the most direct in the corpus.

**Cross-phyla molecular signatures (Fig. 4) — quantitative claim.** The Ras-MNK-eIF4E axis
controls *selective* translation of mRNAs with structured 5'-UTRs; in mouse nociceptors,
genetic or pharmacological MNK inhibition abolishes inflammation-induced hyperexcitability
(Moy et al. 2017, ref [107]). In *Aplysia*, MNK1, eIF4E, and the critical phospho-site
(serine on eIF4E) are all conserved, and a selective MNK inhibitor abolishes nerve-
depolarisation-induced axonal hyperexcitability (Mihail et al. 2019, ref [109]). The
cAMP-PKA-CREB axis: cAMP increases excitability in mammalian nociceptors (Gold 1998),
PKA + scaffolded adenylyl cyclase maintain rat nociceptor hyperactivity months after spinal
cord injury (Bavencoffe et al. 2016), and CREB is required for persistent sensitisation in
both species. The *quantitative* claim: pharmacological perturbation of these two pathways
produces *qualitatively similar* effects on nociceptor hyperactivity in species separated
by $\sim 600\;\mathrm{Myr}$ — consistent with these pathways being **the conserved core
machinery** of persistent nociceptive sensitisation.

**Why human nociceptors may be the most pain-prone — a parameter-shift hypothesis.** Walters
et al. compile six species-specific findings about human nociceptors:

1. **All adult human nociceptors are peptidergic** (express TrkA + CGRP), unlike rodents
   (~50% peptidergic).
2. **All express TRPV1** (rodents: ~50%) — and TRPV1's activation threshold falls into the
   range of normal core body temperature under inflammation, producing tonic discharge.
3. Human nociceptors express **more acid-sensing ion channels** (ASIC1, 2, 3) than rodents.
4. Human DRG neurons have **higher density** of both TTX-sensitive and TTX-resistant Na+
   currents than rodent nociceptors.
5. Nav1.7 activation is at **more hyperpolarised potentials** in human than rodent
   nociceptors — i.e. it opens earlier on the depolarisation curve.
6. **Little or no use-dependent inactivation** of TTX-resistant Na currents — i.e. the
   channel doesn't fatigue under sustained discharge.

Each of these shifts the same firing-rate equation $\lambda(t) \approx \nu_0\,
\mathrm{erfc}\!\bigl((V_{\text{th}}-V_{\text{rest}})/(\sqrt{2}\sigma_V)\bigr)$ in the
*hyperactivity-promoting* direction. The conjunction is interpreted by Walters et al. as
evidence that **hominin evolution selected for a more hair-trigger, more sustained
nociceptor**, possibly to support (i) more reliable communication of injury to social
group members and (ii) longer protective hypervigilance during humans' uniquely slow
healing trajectories. The flip side: this same parameter shift also explains why humans
are clinically prone to **transitions to maladaptive chronic pain** that other mammals
rarely show — the system is closer to its bifurcation point.

### Relevance to project

- **The "is the agent in pain?" guard rail.** Walters et al. (and Box 1 in particular) make
  it explicit that "nociceptor hyperactivity" is a *neurophysiological state*, distinct
  from *pain* (the perceived/computed signal that drives behaviour). Our project should
  preserve this distinction: the **input** to our agent is a signal we can defensibly call
  "**nociceptive** / injury-state evidence" (a noisy observation channel); the **outputs**
  we can claim from our agent are *behavioural correlates of hypervigilance*. Calling
  anything in our agent "pain" requires the additional step Mahajan et al. 2026 take —
  proportional-to-belief operationalisation (Eq. 11 in Section 4).
- **The adaptive-vs-maladaptive bifurcation gives us our success criterion.** A *good*
  in-grid demonstration in our project would be one that produces both regimes: (a)
  short-window injury → adaptive hypervigilance → improved survival under threat; (b)
  prolonged or aberrant injury signal → information-restriction-style chronicity →
  *worse* survival under no real threat. The first regime is the Walters/Crook story; the
  second is the Mahajan/Seymour story; together, they bracket what a complete project
  demonstration should look like.
- **Construct-validity hard constraint.** Walters et al. lean on **species comparison**
  as the methodological gold standard for "this trait is adaptive". We do not have
  species — we have a single agent. The honest equivalent in our project is the
  **architecture-comparison** ablation: with FiLM precision head vs without, with a
  parallel-Q (MaxPain-style) vs single-Q, with a POMDP-style belief-update vs none. Any
  "this is pain-like" claim must be *paired* with an architecture-comparison demonstrating
  that the absence of the candidate substrate degrades the relevant behavioural shape.
- **Crook + Lister + Walters = a closed empirical loop the project can hit.** The three
  papers in the predator-driven cluster jointly specify a behavioural target shape: an
  injured agent should (i) detect a threat-relevant cue at greater distance / earlier in
  the cascade (Crook 2014 — Section 1), (ii) switch to a longer / safer route under that
  cue (Lister 2020 — Section 3), and (iii) *not* show this switch when the injury signal
  is removed or never instantiated (Walters et al. 2023 — Section 6, by inference from
  the squid-sensitisation-block experiment). This three-way replication target is more
  rigorous than any individual paper and is what we should design our `experiment-designer`
  task around.
- **The DRG-as-Bayesian-integrator reading is reusable.** The Bayesian summing-of-log-
  likelihood-ratios argument we wrote above for DRG cell bodies is exactly the
  operation our world model should perform across modalities. **The FiLM precision head
  can be defended as the implementation of channel-specific precision $\pi_i$ in
  $\sum_i \pi_i \log[P(o_i\mid x=1)/P(o_i\mid x=0)]$** — i.e. it modulates the relative
  weight each observation channel contributes to the injury-state belief.
- **Conserved-molecular-machinery is *not* a project hook** — we don't model molecules.
  But the deep-time conservation argument *does* tell us something about what counts as a
  defensible computational claim: a behavioural signature that requires a cAMP-style
  mechanism is *less* defensible in our agent than a signature that follows from a
  generic Bayesian-precision argument, because the latter is closer to Marr level 1
  (computational) and is mechanism-agnostic. We should keep our claims at level 1.

> **Recommended follow-up referrals.** `professor-pain-modeling` for the construct-validity
> arbitration: the Walters et al. comparative-physiology argument is the strongest
> available defence of "persistent nociceptor activity is adaptive hypervigilance, not
> stuck-on alarm" — but does it also *constrain* what our agent must demonstrate before
> we can claim "pain-like adaptiveness"? `professor-neuromodulation` for the descending-
> modulation question: Walters et al. note opioid-mediated stress-induced analgesia and
> opioidergic disinhibition pathways — should our agent's modulation channel be
> implemented as a single FiLM scalar, or as a pair of opposing channels (excitatory vs
> inhibitory)? `senior-developer` to draft an `issue_plan` for the three-way
> Crook+Lister+Walters in-grid replication target sketched above (we cannot write that
> plan ourselves; it requires touching configs and code).

---

## Cross-paper synthesis hooks

> Two thematic clusters are visible across the six papers. These are *seed* hooks for a
> follow-on `literature-curator` synthesis pass — they intentionally do not exhaust the
> connections, but flag the load-bearing arguments and disagreements.

### Cluster A — Predator-driven nociception
**Papers:** Crook (2014, Section 1), Lister et al. (2020, Section 3), Walters et al. (2023,
Section 6).

- **Joint argument.** Persistent nociceptor sensitisation is an *adaptive hypervigilance
  signal* selected by predation pressure. Crook gives the only direct fitness measurement
  (squid + sea bass; OR $\approx 3.5$ for sensitisation benefit). Lister demonstrates the
  mammalian behavioural extension (SNI + fox urine; medium-effect interaction $\eta_p^2
  \approx 0.13$). Walters et al. give the comparative-physiology + molecular-conservation
  scaffolding ($\sim 600\;\mathrm{Myr}$ shared MNK / cAMP-PKA-CREB machinery) that ties
  the two together and predicts where the trait will *not* appear (naked mole-rat,
  *C. elegans*).
- **Where they disagree.** *Crook & Walters* treat sensitisation as primarily peripheral
  (nociceptor-driven); *Lister* uses spared-nerve-injury (peripheral neuropathic) but the
  hypervigilance manifests as a *centrally* processed predator-cue avoidance — leaving
  open whether the SNI effect is strictly downstream of nociceptor hyperactivity, or
  whether it requires central sensitisation as well. Walters et al. concede this is
  underdetermined for mammals and explicitly flags it as an outstanding question.
- **What the cluster gives us.** A behavioural signature (longer alert distance, earlier
  flight initiation, route-switch under threat cue) and an effect-size range (OR 3–5;
  $\eta_p^2 \sim 0.1$–$0.15$) that an in-grid replication should produce.

### Cluster B — Control-theoretic / inference-and-control
**Papers:** Elfwing & Seymour (2017, Section 2), Mahajan, Dayan & Seymour (2026, Section 4),
Seymour, Crook & Chen (2023, Section 5).

- **Joint argument.** Pain and post-injury behaviour should be understood as the output of
  a hierarchical optimal-control system with a Bayesian state estimator. Seymour et al.
  2023 gives the conceptual frame; Mahajan et al. 2026 gives a concrete POMDP realisation;
  Elfwing & Seymour 2017 gives a parallel-Q algorithmic instantiation that pre-dates the
  POMDP work but anticipates its single-scalar weighting parameter (our $w$ in Section 2;
  their cost-of-investigation in Section 4).
- **Where they disagree (or only partially overlap).** *Elfwing & Seymour* treat reward and
  punishment as **two parallel value functions** combined with a single scalar $w$ — a
  clean RL-engineering proposal that maps onto biology only loosely. *Mahajan et al.* and
  *Seymour et al.* treat the inference-and-control loop as the unit of analysis — the
  belief over injury state $b_t$ is the central variable, and chronic pain is a fixed-point
  failure of that loop. The two pictures are reconcilable (the parallel-Q can be derived
  as the action-value half of a belief-MDP) but no paper in this corpus does the
  reconciliation explicitly. *Seymour 2023* is also more concrete about the **neural
  substrate** (insula hub, descending modulation, theta oscillations) than Mahajan et al.
  2026 (which deliberately stays at Marr level 1).
- **What the cluster gives us.** A formal definition of tonic pain ($\propto b_t(s=1)$,
  Mahajan Eq. 11), phasic pain ($\propto \sum_s b(s)\zeta(s,a)$, Mahajan Eq. 10), and a
  parametric handle on chronic pain (cost-of-investigation, $w$, FiLM precision). Plus
  three named failure modes — information restriction, aberrant priors, maladaptive model
  — each of which is an in-grid ablation we can run.

> **Note for `literature-curator`.** A natural cross-cluster synthesis would be to ask
> whether Walters et al.'s *cell-body-as-integrator* argument (Section 6) is the
> biological substrate of Mahajan et al.'s *POMDP belief-update* (Section 4) — i.e.
> whether the DRG is implementing $\sum_i \pi_i \log[P(o_i\mid x=1)/P(o_i\mid x=0)]$
> at the cellular level. Neither paper makes that bridge; it is exactly the kind of
> connection a curator pass should formalise.

---

## Appendix A — Source corpus

All six papers are local PDFs at
`docs/project/references/computational_models_of_pain/`. The previous NotebookLM-based
workflow (notebook `184fc522-49d0-4af1-bcb9-57616360bfe3`) was retired due to auth
instability on rapid headless retries; the local-PDF workflow used by this resumed run is
the canonical one going forward.

## Appendix B — Working-files trail

| File | Purpose |
|---|---|
| `tmp/20260507_pain_lit_review_workspace/00_inventory.md` | Original notebook source inventory (6 primary + 1 secondary) |
| `tmp/20260507_pain_lit_review_workspace/01_crook_backbone.md` | Crook step-1+2 raw NotebookLM section-by-section capture (paper 1) |
| `tmp/20260507_pain_lit_review_workspace/01_crook_quantitative.md` | Crook quantitative details (paper 1) |
| `tmp/20260507_pain_lit_review_workspace/nblmask.sh` | Cleanup-aware wrapper around the deprecated NotebookLM workflow |
| `tmp/20260507_175453_pain_lit_review.md` | Working notes for resumed (PDF-based) run, papers 2–6 |
