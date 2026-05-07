---
title: Nature MI Paper Framing — Pain Computation vs. Nociception, and the Simulation-Methodology Contribution
topic: paper-framing
status: draft
mode: first-pass synthesis (with embedded triage)
created: 2026-05-07
last_updated: 2026-05-07
---

# Nature MI Paper Framing — Pain Computation vs. Nociception, and the Simulation-Methodology Contribution

> Synthesis memo. Written by the research postdoc in response to the user's
> request to make `project_plan.md` serve as the spine of a Nature Machine
> Intelligence submission. **This memo does not modify the plan**; it audits
> what the plan would need to defend, what is currently missing, and which
> professors should be consulted to close the gap.
>
> Anchors: [project_plan.md](../project_plan.md) §§1–4, the four v8 null causes,
> hypotheses H1–H5 in
> [NEUROMODULATION_ALGORITHM.md §1.4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md),
> and the noise-landscape draft in
> [phase_1_noise_landscape.md](../phase_1_noise_landscape.md).

---

## 1. The Paper Thesis, Restated for an Editor

### 1.1 Robotics angle (the Nature MI hook)

> Embodied agents that survive in the open world need *more* than nociception.
> A nociceptor — a thresholded damage signal driving an avoidance reflex — is
> sufficient to keep a manipulator from grinding its end-effector into a
> table, but it does not reproduce the qualitatively different behavior we
> see in animals after injury: persistent attentional bias toward threat,
> longer integration of past harm, and conflict between protective withdrawal
> and competing drives such as feeding. We argue these behaviors are the
> signature of a *second* computation, distinct from nociception, that we will
> call **pain computation**: a slowly-evolving interoceptive state that
> reweights perception, memory, and policy *together* via a shared modulatory
> signal. We show in a controlled grid-world environment that adding
> pain computation on top of nociception (i) is necessary to produce
> hypervigilance-like behavior under conflicting drives and noisy sensing,
> and (ii) yields measurable survival benefit in regimes where pure
> nociception breaks down. The contribution to robotics is a concrete
> minimal architecture — a shared recurrent neuromodulatory core driving
> three injection sites (perception A, memory B, policy/reward C) — for
> giving artificial agents pain rather than just damage detection.

### 1.2 Computational neuroscience angle (the simulation-methodology hook)

> The computational pain literature is rich in *theory* (predictive coding
> accounts of chronic pain, Bayesian placebo, fear-avoidance) but poor in
> *runnable systems* in which those theories can be tested under closed-loop
> action. We release a simulation methodology that asks, for each candidate
> ingredient — heterogeneous sensory noise, conflicting homeostatic drives,
> shared neuromodulation, learned precision — *whether removing it abolishes
> the pain-like phenotype*. The methodology produces a necessity/sufficiency
> ladder: a sequence of progressively richer environments and agents in
> which we can locate the precise rung at which each pain-like signature
> first appears. The contribution to computational neuroscience is the
> ladder itself, and the platform that lets others extend it.

These are two different audiences and two different figures. A Nature MI
submission has to land both, and to do so within the same evidence base.

---

## 2. Minimum Claims the Paper Has To Defend

For a publishable Nature MI paper anchored in this project, here is the
minimal claim set, with the experimental evidence each claim needs. If any
of these is not credibly defended by the proposed phases, the paper is not
yet there.

### Claim 1. Nociception ≠ Pain.
A *nociception-only* baseline (damage signal in observation, no whole-network
modulation, no shared interoceptive state) does not produce the
hypervigilance signature: no time-locked, cross-domain post-injury
reweighting; no chronic-style persistence; no asymmetric drive-conflict
resolution.
- **Evidence required:** an explicit nociception-only control run alongside
  the modulated agent on the canonical task. Same observation, same reward,
  same noise. The nociception-only agent must perform survival fine on the
  base task but *fail* the hypervigilance fingerprint that the modulated
  agent passes.
- **Status in plan:** *not present*. G2 contrasts modulated vs. unmodulated,
  but "unmodulated" in the current diagnosis series is LayerNorm-only — the
  damage signal is still in the obs vector. There is no clean "nociception
  channel + reflex policy, no neuromodulation" condition.

### Claim 2. Pain computation buys something measurable.
Adding pain computation either (a) improves survival in regimes where
nociception alone fails — multi-drive conflict, heterogeneous sensory noise,
chronic-injury contexts — or (b) produces a behavioral phenotype that
matches a known clinical signature (hypervigilance, fear-avoidance,
chronic-pain persistence) that the nociception-only agent cannot produce.
The paper does not need both, but it needs (a) *or* a strong (b) with
explicit clinical mapping.
- **Evidence required:** ΔSurvival between modulated and nociception-only,
  in at least two non-trivial regimes, *or* a quantitative match between
  modulated-agent behavior and a published behavioral signature
  (e.g. attentional-bias paradigm, Stroop-analog, return-to-noxious-context
  task).
- **Status in plan:** G2 specifies the cross-domain time-locked fingerprint;
  it does not specify the survival benefit *over a nociception-only control*
  or the clinical signature mapping. Phase 4 is described as the figure but
  is not currently linked to a published behavioral target.

### Claim 3. The benefit is in the whole-network modulation, not any single site.
Lesioning any one of the three injection sites (A perception, B memory,
C policy/reward) collapses the phenotype; lesioning the shared modulator
collapses it more. This is the H4 (coordinated multi-domain) prediction
operationalized.
- **Evidence required:** factorial ablation across the three injection sites
  ({A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C}) plus the "shared core
  off / per-site cores on" control. This is the load-bearing comp-neuro
  ablation table.
- **Status in plan:** the architecture supports this (the
  `modulation.perceptual_only`-style flags described in
  `NEUROMODULATION_ALGORITHM.md`), but the plan does not list this factorial
  as a Phase-4 deliverable. It is a design implication, not a planned
  experiment.

### Claim 4. The phenomenon is time-locked, persists, and survives an environment richer than the smallest version that produces it.
This is the H1+H2+H5 test — post-injury γ↑, memory-bias↑, policy temperature↓
within a tight window of the injury event, with H5 chronicity
(persistence past recovery) controlled by modulator timescale.
- **Evidence required:** event-locked traces with bootstrap CIs across
  seeds; modulator-GRU-timescale sweep showing chronic-vs-acute regime
  controllable by `mod_hidden_size`; replication on at least one
  environmentally-richer rung (e.g. larger map, different predator
  dynamics, different noise profile).
- **Status in plan:** G2 specifies the time-locked fingerprint and
  Phase 4 names the timescale sweep. The replication-on-richer-environment
  step is not present — see Claim 5.

### Claim 5. The methodology is a contribution, not just the result.
The paper releases a *ladder* of environments (simple → +injury → +predators
→ +conflicting drives → +heterogeneous noise → +chronic dynamics) and
shows for each pain-like signature on which rung it first appears.
- **Evidence required:** at minimum a 4-rung ladder, with the same agent
  trained on each rung and the hypervigilance fingerprint computed on each.
  The figure shows which rung each signature emerges at — this is the
  necessity/sufficiency map.
- **Status in plan:** *the plan does not have an environment ladder*. It
  jumps directly to the full GridWorld and tunes the noise profile.
  Phase 1 is a tuning sweep on one environment, not a sweep across
  environment-complexity rungs.

### Claim 6. The platform is positioned for use by the field.
For a Nature MI simulation-methodology paper, the platform release is part
of the contribution. This is documentation, reproducibility, and a clear
extensibility story, not a separate experiment.
- **Evidence required:** code release, baseline results that someone else
  can reproduce on a single GPU, and a documented extension path (how do
  I add a new modality, a new modulator, a new rung).
- **Status in plan:** not present as a contribution.

If Claims 1–4 hold, the paper is publishable as a robotics-flavored
neuromodulation paper. If Claim 5 also holds, it becomes a
simulation-methodology contribution as well, which is the second hook the
user wants. Claim 6 is supporting infrastructure but is what makes the
methodology paper credible.

---

## 3. Gaps Between the Claim Set and `project_plan.md`

The user's six-bullet starting list is essentially correct. I have merged a
few items, sharpened wording, and added one item (the replication
constraint behind Claim 4 and the platform-release framing).

### G-A. Pain-vs-nociception is in the project's DNA but not formalized as a success criterion.
- **Problem:** §1 of the plan distinguishes "pain-like behavior" from a
  "negative reward signal," but G2 operationalizes the test as
  modulated-vs-unmodulated — and the unmodulated condition retains the
  damage signal in observation. There is no clean nociception-only control.
- **Plan-level fix:** add an explicit *nociception-only baseline* to the
  exit gate of every phase from Phase 2 onward. Define it concretely:
  damage signal preserved in observation, modulator architecture present
  but `modulation.type=null` *and* the interoceptive channels (injury,
  satiation, nutrition) optionally suppressed in the modulator's input —
  the second variant tests whether the *architecture* helps without the
  *interoceptive read*. The hypervigilance fingerprint must qualitatively
  separate this control from the modulated agent.
- **Maps to Claim:** 1, 3.

### G-B. The "simple → complex" methodological arc is asserted but not present.
- **Problem:** the plan's phases are *algorithmic* phases (noise → FiLM
  variants → precision head → analysis). It does not have *environmental*
  phases (a ladder of rungs). The user's stated comp-neuro contribution
  presumes a ladder.
- **Plan-level fix:** introduce a Phase 0 (or restructure Phase 1) into a
  rung-by-rung environment ladder. Minimal proposal:
  - Rung 1: empty grid + injury (no predators, no food). Tests pure
    pain-following-injury reflex.
  - Rung 2: + food and starvation. Tests drive conflict (eat vs. avoid).
  - Rung 3: + predators with heterogeneous threat profile. Tests
    threat-driven hypervigilance.
  - Rung 4: + heterogeneous sensory noise. Tests precision-weighting.
  - Rung 5: + slow injury dynamics (long recovery). Tests chronic-style
    persistence.
  Each rung trains the same agent class. The hypervigilance fingerprint is
  computed on each. The figure is the rung × signature matrix.
- **Maps to Claim:** 5; supports Claim 4 replication.
- **Cost:** non-trivial. This is multiple training runs and almost certainly
  expands Phase 1 by ≥4×. The user has to decide whether the comp-neuro
  hook is worth that cost vs. shipping the robotics-only version.

### G-C. Phase 4 is the paper, but currently sits in the appendix.
- **Problem:** time-locked event analysis, modulator-timescale sweep, and
  the H4 cross-correlation test are the figures of any submission. Right
  now they are gated behind G1+G2 passing and treated as "what we look at
  after the modulator works." For a paper they need to be designed *up
  front* so that Phases 1–3 produce the data that Phase 4 needs without
  additional re-runs.
- **Plan-level fix:** in each of Phases 1–3, list the Phase-4 metrics that
  must already be logged from those runs. Concretely: γ trajectories,
  memory-gate bias, action entropy / reward scale, all on a per-step basis
  with injury-event annotations, even when survival is the headline
  metric. Today these are mentioned as "diagnostic" probes; for the paper
  they are primary observables and should be specified at the run level
  (sampling rate, episode budget, seed count).
- **Maps to Claim:** 4.

### G-D. No robotics-motivation framing.
- **Problem:** the plan reads as a lab-internal engineering roadmap. There
  is no "what does a roboticist gain over nociception" framing. Without
  it, the Nature MI angle is not legible.
- **Plan-level fix:** add §0 or §6 of the plan: "Robotics motivation —
  why a roboticist should care." One page. Concretely contrasts: the
  thresholded force-kill in industrial manipulators (nociception) vs.
  whole-policy reweighting after damage-history (pain), with the closed
  hypothesis that the latter is the cheap way to get the former's
  protective behavior to *generalize* across drive contexts. This is also
  the framing for the cover letter.
- **Maps to Claim:** the editor-level framing of all claims; mostly
  Claims 1–2.

### G-E. No engagement with the pain-computation theoretical lineage.
- **Problem:** the neuromodulation lineage (Doya, Friston, Shine, Wainstein)
  is well-cited inside `NEUROMODULATION_ALGORITHM.md`. The pain-computation
  lineage — predictive-coding accounts of chronic pain (Büchel, Wiech,
  Tabor, Den Bossche), interoceptive inference (Seth, Barrett),
  Bayesian placebo/nocebo (Büchel et al. 2014), fear-avoidance
  (Vlaeyen, Crombez), Bayesian models of chronic pain (Geuter et al.,
  Mancini et al.) — is absent. A pain-modeling reviewer at Nature MI will
  ask why.
- **Plan-level fix:** an explicit "pain-computation lineage" subsection in
  the plan that locates *which* of those theoretical accounts the paper is
  operationalizing. It's not enough to cite them; the paper needs to say
  "we are testing the predictive-coding-based hypervigilance prediction
  in a closed-loop setting where it has not previously been tested." Or
  whichever subset the team agrees on.
- **Maps to Claim:** 2 (the clinical-signature variant) and the editor's
  domain-credibility check.
- **Triage note:** this gap is also the strongest reason to consult
  `professor-pain-modeling` *first* — see §4.

### G-F. No platform-release framing.
- **Problem:** the simulation-methodology contribution requires a release
  story. None is in the plan.
- **Plan-level fix:** a one-page §7 covering: code release, license,
  baseline reproducibility on a single GPU, and how to add a rung. This
  is a necessary part of the methodology contribution; it is not a
  separate experiment.
- **Maps to Claim:** 6.

### G-G. No specification of what the agent itself learns about pain.
- **Problem (added to user's list):** all the gaps above are about
  *external* claims. There is also an internal claim implicit in the
  thesis: the agent's modulator GRU acquires a representation that
  *generalizes* across contexts (e.g., the same internal state predicts
  hypervigilance in environment-rung 4 and rung 5). Without this, the
  paper risks looking like "we tuned a noise profile until the modulator
  helped." Generalization across rungs is what makes it pain *computation*
  rather than environment-specific gating.
- **Plan-level fix:** a representational analysis (modulator-state
  decoding or transfer-learning probe) added to Phase 4. Train on
  rung 3, freeze the modulator, deploy on rung 4 — does the pain-like
  fingerprint transfer?
- **Maps to Claim:** 5 (necessity/sufficiency *plus* generalization).

---

## 4. Triage to Professors

This question genuinely spans four professor domains. Here is the recommended
sequence, with the *exact* prompt for each. The user can spawn directly from
these or merge them.

### 4.1 Order of consultation (recommended)

1. **`professor-pain-modeling`** — first, because Claims 1, 2, and 5 hinge
   on the pain-vs-nociception distinction being construct-valid, and on the
   clinical signature(s) the paper claims to reproduce being well-defined.
   Without a pain-modeling sign-off, the rest of the framing is
   unsupported.
2. **`professor-bayesian-brain`** — second, because once the
   pain-modeling construct is fixed, the Bayesian/predictive-coding
   formalization of "hypervigilance = increased precision on threat
   channels" is what gives the paper a theoretical spine. This also
   tightens Claim 2 (option b — clinical signature match).
3. **`professor-rl-bayesian-dl`** and **`professor-neuromodulation`** —
   third, *in parallel*. RL/BDL covers the architectural side
   (FiLM/precision-head/Dreamer), and neuromodulation covers the
   biological-plausibility side (whether the three-injection-site,
   shared-core architecture maps cleanly onto known modulatory systems
   and what the paper can or cannot claim about that mapping). They are
   independent and can run concurrently after the first two memos
   land.

The first two memos are sequential because the pain-modeling memo
constrains what the Bayesian-brain memo formalizes. The last two are
parallel because they live downstream of the conceptual frame.

### 4.2 Ready-to-paste prompts

#### (a) `professor-pain-modeling`

> Write a 2–3 page concept memo at
> `docs/project/concepts/pain_vs_nociception_construct.md` that:
>
> 1. States the operational distinction between *nociception* (a thresholded
>    damage signal driving a reflex) and *pain computation* (a
>    slowly-evolving interoceptive state that reweights perception, memory,
>    and policy via a shared modulatory signal), in terms a computational
>    pain reviewer at Nature MI would accept. Locate this in the published
>    pain-computation literature (Büchel, Wiech, Tabor, Den Bossche,
>    Seth/Barrett interoceptive inference, Vlaeyen fear-avoidance, Geuter
>    Bayesian placebo, etc.) — not exhaustively, but enough that the paper
>    has a defensible theoretical anchor.
> 2. Defines, concretely and quantitatively, the **hypervigilance
>    fingerprint** that the paper would be claiming the modulated agent
>    reproduces. What is the measurable signature in animal / human pain
>    behavior? What is the simulation analog? Where can it go wrong (i.e.
>    what would *false-positive* hypervigilance look like in our system —
>    something that looks like H1+H2+H5 but is actually a degenerate
>    modulator collapse)?
> 3. Adjudicates whether the proposed **nociception-only control** (damage
>    signal in obs, no neuromodulation, optional masking of interoceptive
>    channels in the modulator input) is the *right* control for the
>    pain-vs-nociception claim, or whether construct validity demands a
>    stronger or different control.
> 4. Flags any place where the project-internal phrase "pain-like
>    behavior" is doing more rhetorical work than the construct will
>    bear.
>
> Project anchors: [project_plan.md](../project_plan.md) §§1–2, G2;
> [NEUROMODULATION_ALGORITHM.md §1.4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)
> H1, H2, H5; this memo §§1–2 above.
>
> Closes gap **G-A**, **G-E**, and the construct-validity side of **G-B**.

#### (b) `professor-bayesian-brain`

> Write a 2–3 page concept memo at
> `docs/project/concepts/active_inference_hypervigilance.md` that:
>
> 1. Derives, sketch-level, the active-inference / predictive-coding
>    formulation of post-injury hypervigilance: what is the precision
>    matrix Πₛ doing during injury, why does increased gain on
>    threat-relevant channels minimize variational free energy under a
>    prior that predicts danger, and how does this map onto Injection A
>    (sensory gain), Injection B (memory persistence as prior dominance),
>    and Injection C (policy/reward as expected free energy weighting).
> 2. Contrasts this with a non-active-inference precision-weighting account
>    (e.g., heteroscedastic Bayesian DL: Kendall & Gal, FiLM-Ensemble) and
>    states what would empirically distinguish the two in the GridWorld
>    Pain setting. The paper has to choose how strongly to commit to
>    active inference; this memo gives the user the basis for that
>    choice.
> 3. Identifies the **degenerate solution** that an active-inference
>    framing risks: the modulator can satisfy free-energy minimization by
>    flattening Πₛ uniformly (the v8 null result, in this language),
>    rather than by selectively elevating threat-channel precision. What
>    in the architecture or environment forces selectivity?
>
> Project anchors: [project_plan.md](../project_plan.md) §3, G1, G2,
> Phase 3 (precision head); this memo §§1–2.
>
> Closes gap **G-E** (Bayesian-brain side) and gives Claim 2(b) its
> theoretical spine.

#### (c) `professor-rl-bayesian-dl` (parallel with (d))

> Write a 2–3 page direction memo at
> `docs/project/directions/architecture_for_pain_computation.md` that:
>
> 1. Reviews the architectural choices on the table — FiLM variants
>    catalogued in
>    [FILM_MODULATION_PLAN.md](../../develop/active/filim/FILM_MODULATION_PLAN.md),
>    the precision head from
>    [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md),
>    the DreamerV3 decoder route — and recommends the **smallest
>    architectural commitment** that defends Claims 1, 2, 3 of this memo.
>    The four v8 null causes
>    ([project_plan.md §4](../project_plan.md#4-main-issue-neuromodulation-does-not-outperform-the-baseline))
>    are constraints, not options.
> 2. Specifies the **factorial ablation** for Claim 3 (single, double,
>    triple injection-site lesions; shared-core lesion). What is the
>    minimum run count and seed count for this to be statistically
>    legible? Where is the budget likely to bite?
> 3. Adjudicates whether the rung-by-rung environment ladder
>    (gap **G-B**) is feasible *with the same architecture across rungs*,
>    or whether the architecture has to be re-tuned per rung — which
>    would compromise the "platform" framing of Claim 5.
>
> Project anchors: [project_plan.md §4](../project_plan.md#4-main-issue-neuromodulation-does-not-outperform-the-baseline)
> (the four causes); G1, G2, Phases 1–3.
>
> Closes the architectural side of **G-B**, **G-C**, and supplies the
> ablation table for Claim 3.

#### (d) `professor-neuromodulation` (parallel with (c))

> Write a 2–3 page critique memo at
> `docs/project/critiques/biological_plausibility_of_three_site_modulation.md`
> that:
>
> 1. Audits the project's three-injection-site, shared-recurrent-core
>    architecture (Injections A perception, B memory, C-PPO action / C-Dreamer
>    reward, all driven by one GRU) against the modulatory-systems
>    literature (Doya 2002; ACh / NE / DA / 5-HT / opioid). Where does the
>    mapping hold? Where is it loose or wrong?
> 2. Flags any claim the paper *cannot* make about biological plausibility
>    given that all four functional roles are emitted by a single GRU.
>    Specifically: is "shared core → coordinated modulation" a fair
>    operationalization of H4, or does the claim require multiple modulators
>    that *learn to coordinate*?
> 3. Identifies whether the modulator-timescale sweep
>    ([NEUROMODULATION_ALGORITHM.md H5](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md))
>    is the right operationalization of chronic-pain dynamics, or whether
>    a more biologically-defensible variant (multi-timescale modulator,
>    explicit tonic/phasic split) is needed for the chronic-pain claim
>    to be defensible to a comp-neuro reviewer.
>
> Project anchors: [project_plan.md §3](../project_plan.md#3-approach-whole-network-neuromodulation),
> H4, H5 in
> [NEUROMODULATION_ALGORITHM.md §1.4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md).
>
> Closes the biological-plausibility side of Claims 3 and 4.

After these four memos are in, a follow-up cross-professor synthesis
(by the postdoc) merges them and the user decides whether to commission a
plan rewrite.

---

## 5. Where the Parent's Framing May Be Oversimplified

In the spirit of pushing back where useful — three flags.

### 5.1 "Pain in robotics" is more contested than the framing suggests.
Not every roboticist will accept that there is a meaningful distinction
between "richer nociception" and "pain." A reasonable reviewer position is
that what we are calling pain is just *learned-precision-weighted
nociception with a slow internal state* — and that calling it pain
introduces unnecessary clinical baggage. This is fine; the paper just has
to engage with it. **Implication:** the pain-modeling concept memo (§4.2a)
should be read partly as a *defense* of the language choice, not just a
literature anchoring. A sober alternative framing is "interoceptive
neuromodulation as a substrate for pain-like behavior" rather than "pain
computation"; the user should hear what `professor-pain-modeling` says
before locking the headline term.

### 5.2 "Simple → complex emergence" has a known overclaiming pitfall.
The history of artificial-life and emergent-behavior papers includes a
well-known failure mode: the system produces *some* behavior at the
complex rung that wasn't visible at the simple rung, and this is reported
as emergence, when in fact the simple rung also produces it under closer
inspection or the complex rung's "emergent" behavior is just the
finer-grained noise of an under-trained system. **Implication:** the
ladder figure (Claim 5) needs to be a *necessity/sufficiency* matrix,
not a "look how rich the agent gets at rung 5" story. Each cell of the
matrix has to report the signature *and* the seed-stable absence of the
signature on lower rungs. Without the latter, "emergence" is a label, not
a finding.

### 5.3 G2's "cross-domain time-locked" criterion may already be at risk
of post-hoc fitting.
H4 (coordinated multi-domain) is operationalized as cross-correlation of
γ, memory-gate bias, and action/reward modulation around injury events.
With three signals and many possible lag windows, the chance of finding
*some* significant cross-correlation pattern under any modulator is
non-trivially above zero. **Implication:** the paper has to pre-register
(at minimum, internally) the cross-correlation window, the
multiple-comparison correction, and the null distribution this is being
compared against, *before* Phase 4 reads. The current G2 wording is too
loose for the cross-correlation result to be a publishable claim. This is
an analysis-design issue best raised with `experiment-designer` once the
professor memos are in.

---

## 6. Recommended Next Step

1. Spawn `professor-pain-modeling` with prompt §4.2(a). This is the highest
   leverage memo in the chain — it constrains the rest.
2. After (1) lands, spawn `professor-bayesian-brain` with prompt §4.2(b).
3. Then spawn (c) and (d) in parallel.
4. The postdoc (this agent) writes a cross-professor synthesis at
   `docs/project/ideas/nature_mi_paper_framing_synthesis.md`.
5. *Then* the user decides whether to commission a rewrite of
   `project_plan.md`. The rewrite is an `agent-manager` job
   (senior-developer to plan the doc edits, possibly experiment-designer
   to draft the ladder configs).

The user does not need to commit to the full ladder (gap G-B) before
hearing from the professors. If `professor-pain-modeling` returns a memo
that says the construct only requires Rungs 3–4, the rewrite is much
cheaper than if all five rungs are construct-relevant.

---

## 7. Cross-references

- [project_plan.md](../project_plan.md) — the document this memo will eventually drive a rewrite of.
- [phase_1_noise_landscape.md](../phase_1_noise_landscape.md) — the existing draft for Phase 1; relevant to gap G-B.
- [perceptual_noise_lit_review.md](../perceptual_noise_lit_review.md) — relevant background for the precision-weighting framing.
- [NEUROMODULATION_ALGORITHM.md §1.4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) — H1–H5.
- [NMN_PERFORMANCE_DIAGNOSIS_v8.md](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — the v8 null result that Claims 1–4 must explain.
- [PRECISION_MODULATION_ARCHITECTURE.md](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md), [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md) — Phase 3 architectural anchors.
- [computational_pain_hypervigilance_tutorial.md](../tutorials/computational_pain_hypervigilance_tutorial.md), [multi_target_bayesian_and_rl_comparison.md](../tutorials/multi_target_bayesian_and_rl_comparison.md) — existing tutorial-grade background that the pain-modeling concept memo can build on.
