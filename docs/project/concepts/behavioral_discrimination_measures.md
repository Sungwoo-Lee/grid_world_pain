---
title: "General-purpose measures of behavioural discrimination between two threat classes"
topic: hypervigilance
status: active
created: 2026-06-12
last_updated: 2026-06-12
aliases: [behavioral-discrimination-measures]
---

# General-purpose measures of behavioural discrimination between two threat classes

> One-line summary: a toolkit of **conditional, matched, and distributional** behaviour measures that detect whether a policy treats stimulus class A differently from class B *without averaging the effect to zero* — the exact failure that hid predator-vs-rabbit discrimination behind a flat episode-mean.

## Purpose — what this memo is for, in plain English

The project's "predator vs rabbit" study asks a simple question: when a harmful animal (a **predator**, whose touch injures the agent) and a harmless one (a **rabbit**, whose touch does nothing) are made identical on everything the agent can sense at a distance — same smell, same chase, same speed — does the trained agent *behave* differently toward them? The honest answer matters for the whole hypervigilance line: an agent that steers clear of the predator *before* contact is recognising danger; one that only reacts after being hit is just a pain reflex.

The measurement went wrong in a specific, instructive way. Every metric we used was an **average over an entire episode or over all states** — mean distance to the predator vs the rabbit (gap ≈ 0), a flee rate of 76% vs 76%, an interrupted-feeding rate of exactly 0.000. All of them said "the agent is blind to the difference." But the difference was real and **conditional**: it only switched on in certain states (when the agent's vision count showed both rabbits sitting on its own cell, so the only *other* approaching smell had to be the predator). Mixing the "rabbits accounted for" states with the "rabbits scattered" states, and averaging, cancelled the effect to zero. The mechanism only surfaced when someone read individual trajectories step by step.

This memo turns that lesson into reusable measurement. It proposes **general-purpose** behavioural-discrimination measures — usable in any two-context study, not tied to predators, rabbits, or this gridworld — that are built so a conditional effect cannot hide inside a mean. For each I give the plain question it answers, the formula, the failure-of-the-mean it fixes, its confounds, and roughly how hard it is to compute from our existing `.rec.gz` rollout recordings. The memo closes by dissecting the naive seed idea ("just measure avoidance of the predator alone") and naming the minimal fix.

The connective principle, stated once: **a mean is the wrong summary when the behaviour you are looking for is gated on a latent state. Measure the policy's response conditional on that state, on matched states, and as a full distribution — never as a single scalar averaged over the gate.**

---

## §1 Setup and notation

Let the agent follow a (recurrent) policy $\pi_\theta(a \mid h_t)$, where $h_t$ is the recurrent hidden state summarising history up to step $t$. Recordings give us, per step: agent position, every animal's position and class $c \in \{A, B\}$ (here $A$ = predator, $B$ = rabbit), the chosen action $a_t$, the 27-d observation $o_t$, and event flags (eat / rest / damage). We do **not** have $\pi_\theta$'s full action *distribution* logged in the deterministic eval — only the argmax action. Two of the measures below therefore come in a cheap "hard" form (from recorded argmax actions) and a stronger "soft" form (if a future eval logs the policy logits; flagged where relevant, hand-off to `experiment-designer` / `senior-developer`).

Define a **context label** $z_t$ = the discrete situation we want to condition on. The canonical choice here is the *nearest approaching animal's class*, but $z_t$ can be any state predicate (e.g. "predator within radius $r$ AND rabbits accounted-for"). The whole memo is about not marginalising over $z_t$.

A **matched-state set** $\mathcal{M}$ is a collection of step-pairs $(t_A, t_B)$ — one with an approaching $A$, one with an approaching $B$ — equated on every nuisance covariate $x$ (distance to the animal, agent energy/hunger, time-in-episode, local food availability, cover availability). Matching is what lets a class contrast mean "same situation, different class," rather than "different situations that happen to differ in class."

---

## §2 The measures

### M-A. Context-conditioned action-distribution divergence (the headline measure)

**Plain question.** In matched situations, does the agent's *choice of action* differ between "an A is approaching" and "a B is approaching"?

**Computation.** Bin steps by context $z \in \{A\text{-near}, B\text{-near}\}$ within a matched-state set $\mathcal{M}$ (match on distance, energy, time). Form the empirical action distributions $\hat p_A(a) = \Pr(a \mid z = A\text{-near})$ and $\hat p_B(a) = \Pr(a \mid z = B\text{-near})$ over the action set $a \in \{\text{N,S,E,W,Eat,Rest,...}\}$. Report a symmetric divergence — **Jensen–Shannon** is the right default (bounded in $[0, \log 2]$, defined even when a support is missing):

$$
\mathrm{JSD}(\hat p_A \,\|\, \hat p_B) = \tfrac{1}{2}\mathrm{KL}(\hat p_A \| m) + \tfrac{1}{2}\mathrm{KL}(\hat p_B \| m), \qquad m = \tfrac{1}{2}(\hat p_A + \hat p_B).
$$

Total variation $\mathrm{TV} = \tfrac12\sum_a |\hat p_A(a) - \hat p_B(a)|$ is an interpretable companion ("fraction of action mass that moves"). Test significance by a **permutation test**: shuffle the class labels across the matched pairs, recompute JSD, and read the observed JSD against the null distribution.

**Soft form (stronger, needs logits).** If the eval logs $\pi_\theta(\cdot\mid h_t)$, compute a *per-step* divergence between the policy distributions in matched pairs and average: $\frac{1}{|\mathcal{M}|}\sum_{(t_A,t_B)} \mathrm{JSD}\big(\pi_\theta(\cdot\mid h_{t_A}) \,\|\, \pi_\theta(\cdot\mid h_{t_B})\big)$. This is far more sensitive than the argmax form — it sees a shifted-but-not-flipped preference (e.g. flee probability $0.45 \to 0.55$) that the argmax throws away. **This is the single most informative upgrade an eval change could buy.**

**What failure of the mean it fixes.** A mean distance / mean flee-rate collapses the *entire action vector* to one scalar and then averages it over context. JSD keeps the full action distribution and refuses to marginalise over $z$. Two policies with identical mean flee-rate but opposite *which-direction-they-flee* have JSD $> 0$ and mean-gap $= 0$.

**Pros.** Directly operationalises "treats A differently from B"; bounded and comparable across studies; the permutation null needs no parametric assumptions. **Cons / confounds.** (1) Argmax form is blind to sub-threshold preference shifts — hence the soft form. (2) Matched-pair construction is where bias enters: if $A$-near and $B$-near states differ in distance and you do not match on it, JSD will report a *distance* effect mislabelled as a *class* effect. (3) Small per-bin counts inflate JSD upward (finite-sample bias toward apparent divergence) — the permutation test corrects the threshold, but report effective $n$.

**Compute cost.** Low for the argmax form (group-by + histogram over existing recordings). Medium for matching. The soft form needs a logit-logging eval change.

---

### M-B. Behavioural decodability (can class be read off behaviour?)

**Plain question.** If I hide the animal's class and show you only what the agent *did* in the seconds around an encounter, can you recover the class? If yes, the behaviour carries class information.

**Computation.** For each pre-contact encounter, build a feature vector from the agent's behaviour in a window $[t_0 - w, t_0]$ before first contact $t_0$ — action histogram, net displacement away from the animal, eat/rest counts, min distance reached, cover-entries. Train a simple classifier (logistic regression / shallow tree) to predict class $c$ from these features. Report **cross-validated balanced accuracy** or **AUC**, with the chance baseline from a label-permuted null. The mutual information $I(c; \text{behaviour})$ is the population-level quantity this estimates.

**What failure of the mean it fixes.** A mean is one projection of behaviour; decodability searches *all* linear (or simple nonlinear) projections for any that separate the classes. It catches discrimination that lives in a feature you did not think to average — e.g. *direction* of flight, *timing* of the first defensive action, or an interaction (flee only when also hungry). It converts "is there ANY behavioural signature?" into a single number with a calibrated chance level.

**Pros.** Omnibus — one test for "any discrimination, anywhere in behaviour space," which is exactly the safety net against the means-hid-it failure. Naturally handles conditional effects because the classifier can use interaction features. **Cons / confounds.** (1) **Leakage is the cardinal sin**: any feature that encodes the animal's *position/identity directly* (rather than the agent's response) makes decoding trivial and meaningless — the window must contain only agent-controlled quantities, and must end *at or before* first contact (post-contact damage trivially reveals the predator). (2) Accuracy alone does not localise *which* behaviour discriminates — pair it with M-A or feature-importance. (3) Needs enough encounters; with $N=20$ episodes, cross-validation variance is high.

**Compute cost.** Medium — window extraction + a scikit-learn classifier over recordings. No code change to the env or eval.

---

### M-C. Distributional avoidance (full distance distribution, not its mean)

**Plain question.** Forget the average distance — across the *whole distribution* of closest approaches, does the agent let the predator get as close as the rabbit? Especially in the dangerous tail?

**Computation.** Per encounter, take the minimum agent–animal distance reached pre-contact, $d_{\min}$. Form the class-conditional distributions $F_A(d), F_B(d)$ of $d_{\min}$. Compare them with measures the mean cannot see:
- **Two-sample Kolmogorov–Smirnov / Cramér–von Mises** statistic on $F_A$ vs $F_B$ — "do the distributions differ anywhere?"
- **1-D Wasserstein (earth-mover) distance** $W_1(F_A, F_B)$ — magnitude of the shift in the natural units (cells).
- **Tail / quantile contrast**: $q_{0.1}^A - q_{0.1}^B$, the gap at the 10th percentile of closest approach. Danger-recognition should show up as the agent refusing to let $A$ into the close tail it tolerates for $B$ — a *quantile* effect that a mean dilutes.

**What failure of the mean it fixes.** Mean distance is a single moment. A risk-sensitive agent can hold identical *mean* distance to both classes while clamping the predator's *lower tail* (never let it get to distance 1) and tolerating closer-but-rarer rabbit approaches — the mean cancels, the tail does not. This is the distance-domain analogue of a risk-sensitive (CVaR-like) objective: the tail of the approach distribution is where injury lives.

**Pros.** Cheap, assumption-light, and directly interpretable in cells; the quantile contrast is the natural target if the hypothesis is "anticipatory tail-avoidance." **Cons / confounds.** (1) $d_{\min}$ confounds *agent avoidance* with *animal-driven approach geometry* — if the predator simply gets interrupted (strike-and-retreat) at a different phase than the rabbit, $F$ shifts for reasons other than the agent's policy. Condition on matched approach geometry. (2) Episode count drives tail-estimate variance; the 10th-percentile contrast needs $\gtrsim$ tens of encounters per class.

**Compute cost.** Low — one scalar per encounter, then standard two-sample distribution tests.

---

### M-D. Context-conditioned response curves (replace every scalar mean with a curve)

**Plain question.** As the threat gets closer (or as the latent gate flips), how does the agent's defensive response *change* — and does that response-vs-state curve differ by class?

**Computation.** Pick a continuous or ordinal state variable $s$ (distance to animal; or the binary gate "rabbits accounted-for"). For each class, estimate the response curve $g_c(s) = \mathbb{E}[\text{response} \mid s, c]$ where *response* is flee probability, eat-suppression, or cover-entry. Plot $g_A(s)$ and $g_B(s)$ together; the discrimination is the **area between the curves**, $\int |g_A(s) - g_B(s)|\, dw(s)$, weighted by the state-visitation density $w(s)$. For the binary gate, this is just a $2\times2$ conditional table: response rate in (gate-on, A) vs (gate-on, B) vs (gate-off, ...).

**What failure of the mean it fixes.** This is the *direct* antidote to the project's failure: the flat 76%-vs-76% flee rate was $g_A$ and $g_B$ already collapsed across $s$. Binning by the gate state $s$ un-collapses them. A curve cannot hide a crossing or a gated effect that a single number averages away.

**Pros.** The most *legible* output — a plot a reader instantly understands, and the one that would have caught the original failure on sight. Generalises to any gate. **Cons / confounds.** (1) Requires correctly *identifying the gate* $s$; if you condition on the wrong latent, the curves re-overlap and you wrongly conclude no effect (the gate here was discovered only by trajectory reading — so M-D pairs naturally with M-B, which finds gates automatically). (2) Sparse bins at extreme $s$. (3) Visitation-weighting $w(s)$ matters — an effect concentrated in rarely-visited states is real but low-impact; report both weighted and unweighted.

**Compute cost.** Low-medium — binning + conditional rates over recordings.

---

### M-E. Sequence/motif-level signatures (does the *order* of actions differ?)

**Plain question.** Beyond *which* actions, does the agent's defensive *routine* — the temporal pattern, e.g. "stop-eating → turn → flee → dive into cover" — fire differently for A than for B?

**Computation.** Encode each pre-contact window as a short action/event string. Compare class-conditional **$n$-gram (motif) frequency distributions** (e.g. bigrams/trigrams of {flee, eat, rest, cover}) with the same JSD/permutation machinery as M-A. Or, for a model-light version, count the rate of a hand-named defensive motif (e.g. "eat→flee→cover within 3 steps") per class. A discrete world-model / sequence-model agent (DreamerV3, IRIS) makes this especially natural, but it needs no model — just the recorded action sequence.

**What failure of the mean it fixes.** Per-step action frequencies (M-A) are order-blind: "eat then flee" and "flee then eat" have identical action histograms. If discrimination lives in *sequencing* (the agent aborts feeding sooner for the predator), only a motif measure sees it. This is where anticipatory danger-recognition would most plausibly hide — anticipation is fundamentally about *timing*.

**Pros.** Captures the temporal structure that scalar and even per-step-distribution measures miss; the most likely home for a genuine "anticipation" signature. **Cons / confounds.** (1) Combinatorial sparsity — trigram counts get thin fast; restrict to a small motif vocabulary. (2) Motif definitions are researcher-chosen — pre-register them or use the omnibus $n$-gram-JSD to avoid cherry-picking. (3) Alignment: windows must be aligned to a common event (first-approach or first-contact) or the motifs smear.

**Compute cost.** Medium — sequence encoding + $n$-gram histograms. No code change.

---

### M-F. State-occupancy / visitation divergence (whole-trajectory, class-driven)

**Plain question.** Over a whole episode, does the *set of states the agent chooses to occupy* differ when the dangerous class is present vs the harmless one — e.g. does it spend more time near cover, or away from the predator's patrol region?

**Computation.** Estimate the agent's state-occupancy distribution $\rho_A(s)$ (episodes with $A$ active) vs $\rho_B(s)$ over a coarse state abstraction (grid region $\times$ near-cover flag $\times$ hunger bin). Compare with JSD or Wasserstein over the occupancy. This is the trajectory-level, integrated cousin of M-A: M-A asks "different action in the moment," M-F asks "different stationary footprint over the episode."

**What failure of the mean it fixes.** Some discrimination is not visible in any single matched moment but accumulates as a *different place to live* — e.g. the agent forages the far quadrant whenever the predator is out. Mean distance averages position into one number; occupancy keeps the spatial structure.

**Pros.** Captures slow, strategic, whole-episode discrimination invisible to step-local measures; ties cleanly to RL theory (occupancy measures are the dual object of the policy). **Cons / confounds.** (1) Strongly confounded by the *animals' own* movement — if the predator simply occupies different cells, $\rho$ differs without any agent discrimination. Must compare occupancy *controlling for* where the animals were, or restrict to the agent's controllable footprint. (2) Needs a good state abstraction; too fine → everything looks different, too coarse → everything looks same. (3) Most useful when class is an *episode-level* manipulation (A-world vs B-world) rather than both-present.

**Compute cost.** Medium — occupancy histograms over a chosen abstraction.

---

## §3 Critique of the naive seed: "just measure avoidance of the predator alone, not the mean"

The user floated this knowing it is naive and wants the failure modes named. There are three, in increasing severity:

1. **No contrast → no baseline for "avoidance."** Avoidance of the predator *alone* has no scale. Is a min-distance of 2.3 cells "avoidance" or just "where a forager incidentally ends up"? Without the harmless-class contrast you cannot separate **class-specific avoidance** from the agent's **baseline movement statistics** (it keeps *some* distance from *everything* simply because animals occupy cells and the agent is busy foraging). The rabbit is the control that converts an absolute number into a *difference*. Drop it and you are measuring foraging geometry, not danger recognition.

2. **"Avoidance of the predator" is still a mean unless you say otherwise.** The phrase, taken literally, is an episode-or-encounter average of some avoidance scalar — exactly the object that just failed. Removing the rabbit does not remove the *averaging*; it removes the *contrast*. You inherit the original pathology (conditional effect cancels) and lose the only thing that anchored it. Worst of both.

3. **It cannot distinguish anticipatory from reactive avoidance** — the actual research question. "How much it avoided the predator" mixes pre-contact steering (recognition) with post-contact fleeing (pain reflex). The settled project verdict is precisely that the avoidance is *post-contact*; an undifferentiated avoidance number would happily report "lots of avoidance" and obscure that it all happens *after* the first hit.

**Minimal fix.** Keep the predator focus if you want, but make it (a) **contrastive** — always paired against the matched harmless class (the rabbit is the within-subject control); (b) **conditional, not averaged** — reported as a response *curve* over the gate state (M-D) or a *distribution* of closest approach (M-C), never a single scalar; and (c) **pre-contact-windowed** — every avoidance quantity computed strictly in the window *before* first contact, so reaction cannot masquerade as anticipation. The smallest defensible upgrade of the seed idea is therefore: *the pre-contact, distance-matched, predator-vs-rabbit quantile gap in closest approach* (M-C tail contrast) — one line of additional structure on the naive measure, and it answers the real question.

---

## §4 Closest published precedents

No measure here is novel; each is a standard tool imported into behavioural RL analysis. Naming the precedents (some are missing from `docs/project/references/` and should be added):

- **Policy/action-distribution divergence (M-A, M-E, M-F).** The KL/JSD-between-policies machinery is the trust-region core of **TRPO (Schulman et al., 2015)** and **PPO (Schulman et al., 2017)** — there it constrains successive policies; here we repurpose it to compare *context-conditioned* slices of one fixed policy. The closest thing to what we propose is the *policy-similarity* and *behavioural-distance* literature (e.g. **Policy Similarity Metrics / bisimulation-based behavioural distances, Castro 2020; Agarwal et al. 2021**); we differ in conditioning the comparison on a *latent state gate within a single policy* rather than comparing two policies or two MDP states.
- **Behavioural decodability (M-B).** This is **representation-probing / linear-decodability** (the neuroscience "can you decode stimulus X from population activity" paradigm, and the RL probing-classifier literature) applied to *actions* instead of activations. Closest precedent: decoding analyses in systems neuroscience and the SPR/representation-probing RL work; we differ by decoding the *external behaviour stream*, not the network's internal representation (no `src/` access needed).
- **Distributional / tail avoidance (M-C).** Two-sample distribution tests (KS, Wasserstein) are standard; the *risk-sensitive* framing — care about the lower tail of approach distance, not the mean — is the behavioural shadow of **distributional RL (C51, Bellemare et al. 2017; QR-DQN, Dabney et al. 2018; IQN)** and **CVaR objectives**. We differ in that the distribution is over an *observed behavioural quantity* (closest approach), not the return.
- **Occupancy divergence (M-F).** Occupancy-measure divergence is the object minimised in **distribution-matching imitation/IRL (GAIL, Ho & Ermon 2016; state-marginal matching, Lee et al. 2019)**. We differ by using it descriptively to *contrast two class-conditional occupancies* of one trained agent.

**References to add to `docs/project/references/`:** a `behavioral_analysis/` or `policy_distance/` topic holding Castro (2020, bisimulation metrics), Agarwal et al. (2021, PSM/contrastive behavioural similarity), Ho & Ermon (2016, GAIL/occupancy), and the distributional-RL anchors (Bellemare 2017, Dabney 2018) — several already live under `references/TD/` or could.

---

## §5 Empirical signatures — what would confirm each measure is doing real work

- **M-A**: matched-state JSD significantly above the permutation null, *and* the moving action mass is interpretable (e.g. flee-direction mass shifts toward "away from predator"). A null result here with a positive M-B means the discrimination is in timing/sequence, not per-step choice.
- **M-B**: pre-contact balanced accuracy clearly above the label-permuted chance band, with feature importances pointing at *agent-controlled* quantities (not leaked position).
- **M-C**: a negative 10th-percentile gap (predator held farther in the dangerous tail) even when mean gap ≈ 0 — the canonical "mean hid a tail effect" signature.
- **M-D**: separation of $g_A(s)$ and $g_B(s)$ that appears *only* in the gated bins (rabbits accounted-for) and vanishes in the ungated bins — reproducing, as a measure, the exact conditional structure trajectory-reading found by hand.
- **M-E**: class-separated motif distributions concentrated on a *timing* motif (e.g. "abort-eat earlier for predator").
- **M-F**: class-conditional occupancy divergence that survives controlling for the animals' own positions.

The diagnostic combination: **M-D + M-B together**. M-B finds *whether* any gate exists (omnibus); M-D *displays* the effect across that gate (legible). Running the headline scalar-mean *alongside* them is the control that demonstrates the mean would have missed it — that contrast is itself the publishable methodological point.

---

## §6 Open questions / hand-offs

- **Gate discovery.** M-D needs the gate $s$ named; M-B can find it automatically but only as a black-box accuracy. An open methodological question: a principled gate-discovery step (e.g. fit a shallow decision tree on M-B features and read its top split as the candidate gate, then validate with M-D). Worth a short methods contribution.
- **Soft (logit-logging) eval.** The single highest-value code change is logging $\pi_\theta(\cdot \mid h_t)$ in the deterministic eval so M-A/M-E can use the policy distribution, not just argmax. Contained change to the eval-rollout writer; **hand-off to `senior-developer`** (eval recording schema) and **`experiment-designer`** (which encounters/episodes to log at what density). Does **not** touch `src/` training.
- **Sample size.** Tail (M-C) and motif (M-E) measures need encounter counts the current $N=20$–200 evals may not supply per class; **hand-off to `experiment-designer`** for an encounter-power estimate.
- **Construct validity.** Whether "behavioural decodability > chance" should count as *danger recognition* (vs. some non-anticipatory tell) is a construct question — **route to `professor-pain-modeling`**. Whether the gate framing maps onto a precision-weighted / belief-state account — **route to `professor-bayesian-brain`**.

---

## Next steps

- **experiment-analyzer** — apply M-D (gate-conditioned response curve) and M-C (closest-approach quantile contrast) to the existing matched-aggression recordings (`results/eval/matchedAggression_traj/.../10000024/`) and the cell-08 lethal-contact recordings; report each *alongside* the flat scalar mean as the control. These two need no code change.
- **experiment-designer** — power/encounter-count estimate for M-C tail and M-E motif measures; decide logging density for a logit-logging eval.
- **senior-developer** — scope the contained eval-rollout schema change to log per-step policy logits (enables the soft form of M-A/M-E). No `src/` training changes.
- **professor-pain-modeling** — construct-validity ruling: does "class is decodable from pre-contact behaviour" license a "danger-recognition" claim?
- **professor-bayesian-brain** — does the gate-conditioned framing correspond to a belief-state / precision-weighted account of the agent's behaviour?
