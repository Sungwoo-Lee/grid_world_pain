---
title: "General-purpose behavioural-discrimination measures for two-context RL studies"
topic: rl_methodology
status: active
created: 2026-06-12
last_updated: 2026-06-12
author: professor-rl (Fable)
---

# General-purpose behavioural-discrimination measures for two-context RL studies

> One-line summary: a menu of reusable measures for "does the policy treat context A differently from context B", each built so that a behaviour which only appears in *some* states cannot be averaged away — the failure that sank the predator-vs-rabbit aggregates.

## Purpose (plain language)

We repeatedly want to ask one question of a trained agent: **does its behaviour change depending on a hidden property of the world?** In the current study the hidden property is "is this approaching animal harmful (predator) or harmless (rabbit)?", and the two are made identical on everything sensable at a distance. But the question is generic — it applies to any study with two otherwise-matched contexts (safe vs. risky arm, cued vs. uncued trial, drug vs. placebo, two reward regimes).

The measures we *had* — episode-mean distance, an overall flee rate, an interrupted-feeding rate — all gave a "no difference" answer, and all were **wrong for the same reason**: they are marginal means. They collapse the behaviour across every state into one number. If the agent only discriminates in *some* states (e.g. only once it has "accounted for" the rabbits, or only within two cells, or only before it has been hit), a marginal mean mixes the discriminating states with the non-discriminating ones and averages the effect toward zero. The verdict "76% flee vs 76% flee → class-blind" is the signature pathology: two marginals matched, while the underlying *conditional* response could differ sharply.

This memo gives measures that compare the policy's **conditional** response in the two contexts and only then aggregate — the opposite order of operations. Each entry states the plain-English question it answers, the computation, which failure of the means it repairs, its confounds, and the compute cost from the `.rec.gz` recordings (per-step agent position, all animal positions + class, action, the 27-number observation, eat/rest/damage events). None of them is specific to predators, rabbits, or this grid.

## The one principle

> **Discrimination is a property of the policy's *conditional* response. Measure it as a divergence between two context-conditioned distributions evaluated at *matched* states, then aggregate the divergences — never aggregate the behaviours first and subtract two marginal means.**

Symbolically, the means computed
$$
\Delta_{\text{mean}} \;=\; \mathbb{E}_{s\sim\rho_A}[\,g(s)\,] \;-\; \mathbb{E}_{s\sim\rho_B}[\,g(s)\,],
$$
which is zero whenever the two marginals coincide even if the per-state behaviour differs everywhere. The correct object is a *within-state* contrast aggregated afterwards,
$$
D \;=\; \mathbb{E}_{s\sim\rho_{\text{match}}}\!\big[\, d\big(\pi(\cdot\mid s, A),\, \pi(\cdot\mid s, B)\big)\,\big],
$$
where $d$ is a non-negative divergence. By Jensen / non-negativity, $D$ does not cancel: distinct conditional responses contribute positively at every state, so they cannot average to zero. Every measure below is a concrete instance of this object or a test statistic for it.

---

## Measure catalogue

Notation: $o_t$ is the 27-d observation, $a_t$ the action, $h_t$ the recurrent hidden state, $c\in\{A,B\}$ the latent context (class), $\pi$ the policy. "Matched state" means a pair $(s_A, s_B)$ equal on everything the policy can condition on *except* $c$.

### M-DIV — Matched-state policy divergence (the central measure)

**Question.** At states that are identical except for which class is present, does the agent *choose differently*?

**Computation.** Stratify recorded steps into bins $b$ keyed on the policy-relevant state (agent body-state, distance-to-nearest-animal, approach geometry, and — see confound — recent-history features). Within each bin, form the empirical action distributions for the two contexts, $\hat\pi(\cdot\mid b, A)$ and $\hat\pi(\cdot\mid b, B)$, and take the Jensen–Shannon divergence
$$
\mathrm{JSD}(b) = \tfrac12 \mathrm{KL}\!\big(\hat\pi_A \,\|\, \bar\pi\big) + \tfrac12 \mathrm{KL}\!\big(\hat\pi_B \,\|\, \bar\pi\big),\qquad \bar\pi = \tfrac12(\hat\pi_A+\hat\pi_B),
$$
then report the bin-weighted mean $\bar D = \sum_b w_b\,\mathrm{JSD}(b)$ **and the per-bin profile** $\{\mathrm{JSD}(b)\}$. The profile localises *where* in state space discrimination lives; the scalar is the omnibus.

**What it fixes.** This is the order-of-operations fix itself: divergence-of-conditionals, averaged after. JS is symmetric and bounded in $[0,\log 2]$ (well-behaved when a bin is sparse, unlike raw KL). Total variation $\tfrac12\sum_a|\hat\pi_A(a)-\hat\pi_B(a)|$ is an equally valid, even cheaper drop-in.

**Confounds / cautions (RL-specific, load-bearing).**
- *Deterministic eval is degenerate.* Best-action eval makes $\pi(\cdot\mid s)$ a point mass, so a per-step JSD is 0/1 noise. Two routes: (i) **log the pre-argmax action probabilities / logits** during eval and use the full categorical per step — strongly recommended, one line in the eval head, no architectural change; or (ii) keep argmax but estimate $\hat\pi$ as the empirical action *frequency within a bin* (needs enough matched steps per bin).
- *The policy is recurrent.* $\pi$ conditions on $h_t$, not $o_t$. Two observationally-identical states can carry different beliefs (e.g. "I was hit 3 steps ago"). Matching on $o_t$ alone re-creates the averaging problem one level down (mixing belief regimes inside a bin). **Bin on history features too** — steps-since-last-damage, the accounted-for count (visual ch7), cumulative contacts this episode. This is the same anticipatory-vs-reactive issue the study already hit, restated as a belief-identifiability problem.
- *Discrimination ≠ anticipation.* M-DIV detects "treats A≠B". To make it *anticipatory*, restrict the bins to **pre-contact windows only** (no damage event yet this encounter). M-DIV over pre-contact states is the precise operationalisation of "danger recognised at a distance".

**Compute cost.** Medium. Binning + empirical distributions is cheap; the logit-logging variant needs an eval-side change (hand to `senior-developer`, contained).

### M-CURVE — Context-conditioned response curves

**Question.** Plotted *against* a state variable (distance, hunger), where do the two contexts' responses separate?

**Computation.** For a chosen response $g$ (P(flee), P(eat), P(bush-dive)) and a conditioning variable $x$ (distance-to-animal), estimate two curves $\hat g(x\mid A)$, $\hat g(x\mid B)$ with bootstrap CIs, and the area between them $\int |\hat g(x\mid A)-\hat g(x\mid B)|\,dx$ as a scalar summary.

**What it fixes.** Directly exposes the averaging: if the curves overlap at far range and separate at close range, the marginal (which integrates over $x$ with whatever distance distribution each context happened to produce) hides it. This is the cheapest, most legible repair of the "distance-matched flee 76% vs 76%" result — that number is a single point on a curve that may diverge elsewhere.

**Confounds.** The two contexts may *occupy different $x$-distributions* (predator chases harder → more close-range steps). Compare curves only on overlapping support, and weight the area integral by the shared support, or it conflates "responds differently" with "ends up at different distances".

**Compute cost.** Low. Pure post-processing of recordings.

### M-DECODE — Behavioural decodability (class-from-actions)

**Question.** Can an outside observer recover the hidden context *from the agent's behaviour alone*?

**Computation.** Train a classifier $f:\;(\text{window of } a_{t-k:t},\ \text{agent kinematics}) \mapsto \hat c$, using **only agent-controlled variables** (never the animal's own position/class, which leak $c$ trivially). Report balanced accuracy / AUC against a **label-shuffle null** (permute $c$, re-fit, repeat → null distribution; the gap is the effect). Above-chance decodability ⇒ the policy's behaviour carries class information.

**What it fixes.** It is an *omnibus, model-free* discrimination test: the classifier will exploit any state-gated or sequence-level signal automatically, so it catches conditional effects you didn't think to bin for. It is the natural "is there *any* signal?" screen to run before hand-designing M-DIV bins.

**Confounds.** Leakage is the killer — any feature that encodes the animal's identity/dynamics rather than the agent's choice inflates AUC. Restrict the feature set to agent action + agent position/velocity. Decodability shows *that* class is recoverable, not *what* the discriminating behaviour is — pair it with M-CURVE/M-DIV for interpretation.

**Compute cost.** Medium. A small logistic-regression / gradient-boost on windowed features + a permutation loop.

### M-DIST — Distributional / quantile avoidance

**Question.** Forget the mean distance — does the *shape*, especially the dangerous tail, differ?

**Computation.** Take the full distribution of agent–animal distance (or, better, **closest-approach per encounter**) for each context and compare with a two-sample distributional test: Wasserstein-1 (earth-mover) $W_1(P_A,P_B)$ and/or Kolmogorov–Smirnov, plus an explicit **lower-tail quantile contrast** $Q_{0.05}^A - Q_{0.05}^B$ (how close does the agent *ever* let each class get?).

**What it fixes.** A zero mean-gap is fully consistent with "never lets the predator inside 2 cells, lets the rabbit touch" — a tail-only effect the mean erases. Risk-sensitive avoidance lives in the tail, not the centre; the mean is the wrong functional.

**Confounds.** Same occupancy caveat as M-CURVE — different chase dynamics shift the distance distribution independent of policy. Closest-approach-per-encounter is more policy-attributable than per-step distance.

**Compute cost.** Low. `scipy.stats` two-sample + empirical quantiles.

### M-EVENT — Event-aligned encounter motif (peri-contact response)

**Question.** *When* does the defensive response fire — before the touch (anticipatory) or after (reactive)?

**Computation.** Segment encounters (approach → min-distance → separation). Align each to an event onset — the step the animal crosses "$r$ cells away", or the contact step — and average the defensive-action probability across encounters time-locked to that onset (a PSTH/peri-event time histogram), separately per context. Compare the **pre-onset** portions and the **response latency** distributions.

**What it fixes.** Episode-level means destroy timing. The whole project hinges on pre- vs. post-contact, and an event-aligned average is the standard tool that preserves it. A positive pre-onset divergence between contexts is the cleanest possible "anticipatory discrimination" evidence; a flat pre-onset with divergence only after contact reproduces the "pain-reactor" verdict quantitatively.

**Confounds.** Needs a principled encounter definition and enough encounters per context for a stable PSTH; ragged encounter lengths require care in alignment (align to onset, not episode start).

**Compute cost.** Medium. Encounter segmentation + alignment + averaging.

### M-OCC — State-occupancy divergence (secondary)

**Question.** Does the agent steer itself into *different regions of state space* depending on context, even if instantaneous actions look alike?

**Computation.** Estimate occupancy measures $\rho_A(s),\rho_B(s)$ over a coarse state featurisation (distance bin × body-state × region) and take $\mathrm{JSD}(\rho_A\|\rho_B)$ or $W_1$.

**What it fixes.** Captures cumulative, trajectory-level steering that per-step action divergence can miss (small per-step biases compounding into different visited regions).

**Confounds.** Occupancy is jointly driven by policy *and* by the animal's (here matched) dynamics; with matched dynamics this is interpretable, otherwise it conflates the two. Weaker attribution than M-DIV — keep it secondary.

**Compute cost.** Low–medium.

---

## Critique of the floated seed idea

**Idea:** "just measure how much the agent avoided the predator only, not the mean."

The instinct (stop averaging) is right; the proposal as stated still fails, three ways:

1. **It is usually still a mean.** "How much it avoided the predator" computed as mean closest distance — or mean flee rate near the predator — is the *same marginal* over states, now restricted to one class. It still mixes the accounted/not-accounted (and pre-/post-contact) regimes and averages the conditional effect toward zero. Dropping the second class does not change the order of operations that caused the washout.

2. **No contrast ⇒ no isolation of the *class* effect.** The agent flees *all* approaching animals (it fled rabbits at 76% too). Predator-avoidance-in-absolute is "avoid moving things near me", not "recognise danger". The matched rabbit is not optional decoration — it is the **control that holds the animal's dynamics fixed and isolates `is_damaging` as the only varying factor**. Without it, any avoidance you measure is unattributable: it could be general caution, the predator's distinct kinematics, or class recognition, with no way to separate them.

3. **Predator-only is off-distribution.** Removing the rabbit to "look at the predator alone" (the predator-only world) is exactly the OOD confound already flagged in the settled verdict — the agent never trained with zero rabbits, so its behaviour there is not evidence about its trained policy.

**Minimal fix.** Keep the matched rabbit as a **within-episode contrast**; restrict to **pre-contact windows**; bin by **belief/history** (accounted-for count, steps-since-damage); and report the **distribution of the matched-state difference**, not its scalar mean. That is precisely M-DIV (or its paired-difference form) restricted to pre-contact states — contrast + condition + distribution-not-mean, the three things the seed idea drops.

---

## Recommended program (ranked)

1. **M-CURVE** first — cheapest, most legible, turns the misleading "76% vs 76%" into a curve that either does or does not separate somewhere.
2. **M-DECODE** as the omnibus screen — "is there *any* class signal in behaviour?" before hand-designing bins.
3. **M-DIV** (pre-contact, history-binned) as the principled core measure — with the eval-side logit logging so action distributions are non-degenerate.
4. **M-DIST / M-EVENT** to characterise the *kind* of discrimination — tail-only avoidance (M-DIST) and anticipatory-vs-reactive timing (M-EVENT).

## Open questions

- **Logit logging.** M-DIV is much stronger with the policy's pre-argmax probabilities recorded. Is a contained eval-head change acceptable (route to `senior-developer`), or do we stay argmax-only and pay the binning cost?
- **Belief identifiability.** Even a clean M-DIV signal cannot, on its own, separate "recognises the class" from "remembers recent pain". Settling that needs the in-distribution accounted-for conditioning already named in the predator-only memory note — these measures sharpen it, they don't replace it.
- **Multiple comparisons.** Per-bin M-DIV profiles and PSTH pre-onset windows invite many tests; pre-register the bins / use a permutation null at the profile level.

## Next steps

- **`experiment-designer`** — fold M-CURVE / M-DIST / M-EVENT into the pending cell-08 final-checkpoint analysis and the in-distribution confirmation; they need only the existing `.rec.gz`.
- **`senior-developer`** — scope the contained eval-head change to record per-step action probabilities (enables non-degenerate M-DIV / M-DECODE).
- **`professor-bayesian-brain`** — M-EVENT pre-onset divergence is the bridge to the precision-weighting / anticipation framing; worth a cross-read.
