---
title: "Symposium contribution — pain-construct-validity position on impactful-vs-reasonable framing"
session: 2026-05-16_impact_vs_reasonable
contributor: professor-pain-modeling
date: 2026-05-16
inputs:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md
  - docs/project/concepts/film_neuromod_integration.md
  - docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md
  - docs/project/concepts/pain_vs_nociception_construct.md
  - .claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md
prior_contributions:
  - docs/project/references/neuromodulatory_algorithms/feedback/20260516_pain_modeling_predictions_for_v3.md (v3 round)
---

# Symposium contribution — pain-construct-validity position on impactful-vs-reasonable framing

## Plain-English entry point

The symposium asks a portfolio-level question: *given the user's two new constraints — "the paper should suggest the possibility of integrating neural-gain-like modulation with behavioural-hyperparameter modulation, not claim full coverage" and "no one-to-one neuromodulator-function mapping" — is the pain-relevant story still publishable as a single paper at the current empirical stage, or does it need to be deferred?*

My role in the room is unique: I am the **construct-validity guardian**. None of the other contributors will write what does *not* count as pain-relevant. That gate-keeping is my deliverable.

Plain-language summary of the question, before any pain-science jargon:
- The project has shown that adding a small side-network (the "modulator", which emits scale + shift signals into the main policy via a FiLM layer) lets the agent recover better when it re-encounters a previously-seen environment stage. That is the R2 anchor — a single-seed, single-run finding that the modulator beat the unmodulated baseline by roughly 25× the seed-noise floor on returns to a previously-active task stage.
- The team is debating whether this result is *strong enough on its own* to support a paper that uses pain-science language ("hypervigilance", "pain-like behaviour") in the headline, or whether the paper should keep pain in the introduction and the discussion only, with the result framed in pure RL-modulation terms.
- The user's new constraints relax one half of the bar (possibility, not coverage) and tighten the other half (no one-to-one mapping). My read: the **possibility** relaxation buys the project enough room to use pain-construct language in the *motivation*, but the **no-one-to-one** constraint does *not* buy enough room to relax the four-dissociation gate that decides whether the headline can claim *pain*-relevance specifically rather than *aversion*-relevance generally. The two constraints push in opposite directions and the gate has to be re-stated, not removed.

**My recommendation, stated up front so the rest of this memo can defend it.** The project should adopt a hybrid of **Position B (pain-relevant motivation, not headline) for v4-as-paper** plus a **soft Position A reach** (pain-relevant headline) that is *contingent on* the four-dissociation gate landing, re-stated in behavioural-effect language consistent with the user's no-one-to-one constraint. Track A then publishes the FiLM-substrate-with-pain-motivation paper now; Track B's probe-trial extension is the *upgrade path* that lifts the same paper from "pain-motivated" to "pain-claiming" without needing a second submission. The minimum publishable pain claim at the current empirical stage is **"interoceptive neuromodulation produces a hypervigilance-*shaped* behavioural signature — the four-dissociation gate has not yet been tested, so the construct-validity status is pre-registered as 'pending' rather than 'validated'"**.

The rest of this memo builds the case for that recommendation and re-states the construct-validity gate so it survives the user's two constraints.

---

## §A. Pain-relevance under the user's two constraints

### A.1 What "possibility framing" buys

The user's first constraint — "suggest the *possibility* of integrating neural-gain-like modulation can act like hyperparameter and behavior modulation, not full coverage" — relaxes the burden of proof in one specific direction. Under a "full-coverage" framing, the paper would have to demonstrate that the FiLM substrate accounts for *all* of pain phenomenology that maps onto its claimed channels. That bar is unreachable on the current testbed and was always going to require multi-paper development. Under a "possibility" framing, the paper instead has to demonstrate that *one* defensible pain-relevant computation runs on the substrate, with the rest pre-registered as future work.

This relaxation is *real* and *load-bearing*. A computational-pain reviewer at *PAIN* / *Nature Mental Health* will accept a "this substrate can compute X, and X is one component of pain phenomenology, leaving the rest open" framing — that is the standard structure of every published pain-computation paper since Büchel et al. 2014. Wiech 2016 (*Science*) explicitly proposes the predictive-coding-of-pain framework *as a research programme*, not as a closed account. The project does not have to close pain.

But the relaxation is *not free*. The "one defensible computation" still has to clear the construct-validity bar — otherwise the paper claims that *any* aversion-shaped signal demonstrates the possibility of pain computation, and the constructs collapse. Eccleston & Crombez 1999's "interruption hypothesis" of pain is precisely the position that pain is *not* general aversion — pain has a *specific* attentional signature, distinct from threat in general. Possibility-framing relaxes *coverage*; it does not relax *specificity*.

### A.2 What "no one-to-one mapping" tightens

The user's second constraint — "this mapping cannot be done by one-to-one as our brain's neuromodulation is not affect to single functionality" — is biologically correct and methodologically tightening. A single biological neuromodulator (NA, ACh, DA, 5-HT) affects multiple downstream functions: NA shifts inverse-temperature *and* gain *and* memory consolidation *and* arousal *and* network metastability (Aston-Jones & Cohen 2005; Shine 2021; Wainstein 2025). Forcing a one-to-one map is biologically false and inflates the unification claim past what the literature supports.

The constraint cleans this up at the *biological* side. But — and this is the part the symposium needs to be honest about — it makes the *construct-validity* side *harder*, not easier. Here's why. My v3-round four-dissociation gate (channel-selective γ at site A perception, time-locked to injury, within-trial cross-domain coupling, C1 EMA baseline) was implicitly a *one-to-one channel-to-construct commitment*. It required "γ at site A" to mean "perception channel" specifically, so that the dissociation tested *that* channel's pain content. If the project re-frames the substrate as one-to-many (the FiLM substrate carries *several* coupled channels, not three dissociated ones), then the dissociation tests have to be re-stated in language that does not commit to *which channel* is doing the work — only that *some* behavioural signature consistent with hypervigilance emerges.

This is a non-trivial rewrite of the gate. It is the load-bearing technical contribution of this memo.

### A.3 Joint effect of both constraints on the pain-relevant story

Possibility-framing buys the project room to claim *one* pain-relevant computation without full coverage. No-one-to-one removes the project's ability to *name* that computation at the channel level. The intersection — what survives both constraints — is a class of claims of the following shape:

> *"The modulator produces a behavioural signature that, on the dissociations the literature uses to identify hypervigilance, qualitatively matches the construct, regardless of which Doya channel is doing the work."*

This is a strictly *behavioural* claim that uses pain-construct language to describe a *pattern of behavioural dissociations*, not to localise a computation. It is consistent with both constraints. It is also weaker than v4 §5.7 currently claims — v4's gate dissociation 1 is "channel-selective γ at site A, not volume control", which is channel-language. Under the user's new constraints that dissociation should be re-phrased as "channel-selective gain shifts on threat-relevant inputs, regardless of which injection site carries them".

This re-phrasing is the §C deliverable below.

### A.4 Re-evaluating my v3-round verdict

My v3-round verdict was: *"current testbed + T/P split is enough for 'interoceptive neuromodulation produces hypervigilance-like behaviour' but NOT for full pain-computation".* Under the user's possibility-framing, "hypervigilance-like behaviour" is *exactly* the right register. The verdict holds. What changes is the **threshold for the headline term**: in the v3 round I said "if all four dissociations hold, 'pain-like hypervigilance signature' is defensible". Under possibility-framing, the looser variant "*pattern* of hypervigilance-shape dissociations" is defensible at a slightly lower bar — but only if the four-dissociation gate is re-stated to drop the channel-level commitment and replaced by behaviourally-defined criteria.

---

## §B. Three positions on the impactful-vs-reasonable trade-off

### B.1 Position A — Pain-relevant headline

The paper claims the FiLM substrate demonstrates *interoceptive hypervigilance-like behaviour* in an RL agent. The four-dissociation gate (re-stated per §C below) is the construct-validity bar; if it lands, the headline survives a *PAIN* / *Nature Mental Health* reviewer. Possibility-framing satisfied: "*potential* substrate for pain-like modulation".

- **Impact ceiling.** High. The project's downstream identity is pain modelling; a pain-relevant headline at the FiLM-substrate level is what makes the work portable to clinical-translational venues later (Apkarian's chronic-pain-as-controller-failure programme is *PAIN*-natural).
- **Risk.** High. Any one of the four re-stated dissociations failing on R2-anchor + T/P-split data forces a mid-review retreat to Position B-language under reviewer pressure. The R2 anchor alone provides *zero* information about three of the four dissociations (only the within-trial cross-domain coupling becomes informative under the T/P split, and only if the split lands).
- **Empirical prerequisite.** T/P split has to land (Phase 0 prerequisite in v4 §3 already); the C1 EMA-of-nociception baseline has to be implemented (small senior-developer add per v4 §5.7 footnote); the within-trial cross-domain coupling has to be measurable (this depends on whether the modulator actually dissociates across sites — see Vlaeyen & Linton 2000 on whether perceptual / mnemonic / behavioural hypervigilance dissociate in clinical samples; the answer is "partially", which is *the* finding the project could deliver).
- **Current-data verdict.** Cannot be carried by R2-anchor alone — the anchor is *consistent with* all four dissociations and *consistent with* all four being false (per v4 §5.7 final paragraph, retained from my v3-round submission).

### B.2 Position B — Pain-relevant motivation, not headline

The paper's contribution is the FiLM-substrate-as-possibility-bridge claim (the unification thesis of v4 §3); pain is the *motivating context* in §1 and a *future-work* item in §7. The professor on the RL/BDL side will likely recommend exactly this.

- **Impact ceiling.** Moderate. The substrate-as-bridge claim is publishable at NeurIPS / ICLR / *Nature Machine Intelligence* on its own merits — Lee 2024's two-channel ACh+NA paper went to a tier-1 RL venue without pain-construct content. The project's contribution-over-Lee is the multi-site FiLM substrate, the FiLM ⊂ Hypernet ⊂ point-mass-BHN-limit lineage, and the R2 continual-learning anchor — sufficient on its own.
- **Risk.** Low. The construct-validity gate becomes a *future-work commitment* rather than a *claim-validity test*. The paper is reviewable on the RL/BDL content alone.
- **Cost.** The project's downstream identity is pain modelling. Position B publishes the substrate result without the pain identity attached to it. The follow-up paper (Track B) then has to *re-establish* the pain framing — a harder pitch than building it in from the start, because reviewers will read the first paper as "this is an RL paper" and expect the second to *also* be RL with pain as motivation.
- **Construct-validity-side verdict.** Defensibly publishable on the R2 anchor. Pain content lives in the introduction's framing of *why* this substrate matters (because the project sits at the interoceptive-modulation interface and that interface is the substrate where pain-computation theory predicts hypervigilance signatures should emerge — Büchel 2014; Wiech 2016; Tabor & Burr 2019), and in the discussion's roadmap to the four-dissociation gate.

### B.3 Position C — Two-paper strategy

Paper 1 (now): FiLM-substrate result with no pain claims, possibly even pain-free in motivation. Paper 2 (after probe-trial extension): the pain-relevant four-dissociation gate result.

- **Impact ceiling.** Highest, on a sufficient timeline. Two clean papers each landing the strongest claim their data supports.
- **Risk on Paper 2.** The probe-trial extension still has to deliver. If the four-dissociation gate fails on the post-recovery probe, Paper 2 retreats to Position B-language under harsher conditions (a reviewer who has already seen Paper 1 will be less forgiving of the gap between Paper 2's framing and its results).
- **Cost.** Slowest. The probe-trial harness has to be implemented (senior-developer); the T/P split has to land; the C1 baseline has to be run; ≥ 5 seeds for the within-trial coupling threshold. Realistically 3–6 months of additional work between papers.
- **Construct-validity-side verdict.** This is the *cleanest* publishable trajectory. The cost is wall-clock time.

### B.4 Recommendation

**Hybrid of B + soft A — *not* C.** Two reasons.

First, **the project's identity is at risk under Position B alone**. Position B reads the v4 result as an RL contribution and defers pain to future-work. But the project's reference corpus, its prior framing memos (`pain_vs_nociception_construct.md`, `active_inference_hypervigilance.md`), its testbed design (predator-induced injury with recovery dynamics), and the user's stated downstream identity all sit in pain modelling. Publishing the substrate result with pain only in motivation lets the *next* paper own the pain identity, but it ships the substrate-claim into the literature with no pain framing attached — a reviewer reading Paper 2 will not automatically treat Paper 1 as "the precursor pain-substrate paper". This is the *position-cost* of Position B.

Second, **Position C is the cleanest but Track A and Track B are not architecturally separate**. The probe-trial extension that lifts the paper from Position B to Position A is small enough that the *same paper* could run it as a final-results section before submission. v4 §5.8 already names this as a "small senior-developer add" — a post-recovery predator-cue probe trial with no injury risk. That add is the **pivot point** between the two positions: with it, the paper can pre-register the four-dissociation gate and report the result (Position A); without it, the paper reports the substrate result with the gate as future-work (Position B).

The hybrid recommendation is: write the paper in **Position-B-default-Position-A-conditional** register. Frame the substrate result in §1 with pain as the load-bearing motivation. Defer the four-dissociation gate to §6 "Future work". *Conditional on the probe-trial extension landing before submission*, promote the gate result into §4 or §5 of the paper, and re-write §1's possibility-framing as a *claim* rather than a *motivation*. The paper then sits at Position A or Position B depending on what the data delivered, but the writing strategy doesn't lock the choice in advance.

This is consistent with the user's possibility-framing ("we suggest the *possibility* — the four-dissociation gate is the strongest test, and we run as much of it as the testbed supports") and with the no-one-to-one constraint (Position B's substrate-as-bridge claim does not commit to channel-level pain attribution at all).

---

## §C. The construct-validity gate, re-stated under no-one-to-one

This is the load-bearing technical contribution of the memo. The v3-round / v4 §5.7 gate was channel-coded; under the user's no-one-to-one constraint, it has to be re-stated in *behavioural-effect language* without committing to which Doya channel implements which dissociation.

### C.1 The four dissociations, re-stated

**Re-stated dissociation 1 — Channel-selective gain shifts on threat-relevant inputs, regardless of injection site.**
The original gate required γ at site A (encoder) to rise more on threat-coupled input channels (olfaction, vision when carrying predator cues, exteroceptive nociception) than on threat-irrelevant ones (proprioception, satiation). Under no-one-to-one, the dissociation is restated as: *somewhere in the modulator's read-out — at one or more injection sites jointly — the post-injury gain pattern on the encoder's input modalities is non-uniform, with threat-relevant modalities receiving systematically larger upward shifts than threat-irrelevant ones, by a margin of $\Delta \gamma_{\mathcal{T} \setminus \mathcal{N}} \ge 0.2$ across $\ge 5$ seeds*. Whether the read-out is implemented purely by α at A, partly by γ at C feeding back into the encoder via the recurrent loop, or by a mixture, is unspecified. The dissociation is on the *behavioural-readout* gain pattern, not on the architectural site.

This is consistent with the clinical literature. Eccleston & Crombez 1999, Crombez et al. 2005, and Asmundson et al. 1997 all operationalise hypervigilance at the *attentional* level (reaction-time advantages on threat-relevant cues) — none localises it to a single cellular gain mechanism. Wiech 2016 frames the precision-weighting as a process distributed across hierarchical levels of cortical inference, not pinned to one. The re-stated dissociation tracks the clinical operationalisation rather than the channel-attribution.

**Re-stated dissociation 2 — Time-locked to interoceptive injury inference, not to the threat input.**
Unchanged from v3 / v4 §5.7. The gain pattern in dissociation 1 must lag injury-channel onset by $\le \min(5, 2\tau_{\text{mod}})$ steps and remain insensitive to predator-presence-change without injury. This dissociates pain-computation from task-relevance-learning at the temporal level, and does not depend on which Doya channel is doing the work — it depends on what the modulator's input is conditioned on.

**Re-stated dissociation 3 — Joint perceptual–mnemonic–policy reweighting through a common modulator state.**
The v3 dissociation required within-trial cross-domain coupling $r \ge 0.3$ at lag $\le 5$ between the site-A gain shift, the site-B memory shift, and the site-C policy shift on the same trial, same seed. Under no-one-to-one this remains testable but is *re-interpreted*: the dissociation does not require that these three shifts correspond to three separable neuromodulators (ACh, plasticity-gating, NA — which is the v3 / v4 §3 mapping). It requires only that *behavioural effects at perception, memory, and policy are co-modulated within a trial* — that they ride on a *common* slow state, whatever Doya channels that state instantiates. This is the structural core of Vlaeyen & Linton 2000's fear-avoidance model and of the Crombez et al. 2005 "concern-dependent" framing: the three are jointly regulated through a common pain-related concern, not independently. The clinical literature *itself* does not commit to channel-attribution for this coupling.

This re-statement is actually *more biologically defensible* than the v3 version, because real neuromodulators *do* affect multiple functions jointly — the within-trial coupling is the *expected* signature of any biologically-realistic modulator, not a one-to-one channel commitment.

**Re-stated dissociation 4 — Dissociation from a slow-input baseline (C1 EMA contrast).**
Unchanged in structure from v3 / v4 §5.7. The modulated agent's signature must differ from a trainable exponential-moving-average-of-nociception baseline wired through the same A/B/C heads. Under no-one-to-one, the contrast metric becomes: *the modulated agent produces a non-uniform across-channel gain pattern (dissociation 1) that the C1 EMA baseline cannot reproduce because the EMA has only one scalar to assign*. This dissociation is *channel-agnostic on the architectural side* — it works because the C1 baseline is a single scalar by construction, so any across-channel non-uniformity in the modulated agent rules it out. The contrast does not depend on attributing the modulated agent's non-uniformity to a specific Doya channel.

### C.2 What the re-stated gate gains

- **Consistency with no-one-to-one.** The four dissociations are now stated in behavioural-effect language, with no commitment to which Doya channel implements which dissociation. The modulator can be doing α-work at one site, β-work at another, plasticity-gating at a third, with all three driven by a single `mod_h` whose "channels" cross-talk — exactly what biology does.
- **Robustness to v4 §5.6.5's cross-seed channel-attribution warning.** If different seeds produce different channel-attributions (site A is α in one seed, β in another), the original gate would fail to be well-defined. The re-stated gate is seed-invariant — it measures behavioural-effect patterns, not site assignments.
- **Higher clinical fidelity.** Tracks the clinical operationalisation of hypervigilance (attentional bias, joint perceptual-mnemonic-policy reweighting, prior-shift independent of input) rather than a one-to-one neuromodulator mapping that the clinical literature itself does not commit to.

### C.3 What the re-stated gate loses

- **Some architectural discrimination power.** The original v3 / v4 §5.7 gate, if it held *with* the channel-attribution intact, would have been *stronger* evidence — it would have shown not just that the modulator produces a hypervigilance-shaped behavioural pattern, but that the specific Doya channels predicted by Lee 2024 / Doya 2002 are recoverable from the architecture. The re-stated gate is silent on channel-attribution; recoverability is no longer part of the test. This is the cost of the user's no-one-to-one constraint and it is *worth paying* — over-claiming channel-attribution from a single `mod_h` is the larger risk.
- **Cross-link to v4 §5.6.5 (cross-seed consistency).** v4 §5.6.5 still has value as an *exploratory* analysis, but the gate no longer depends on its outcome. If channel-attribution is inconsistent across seeds (Spearman correlation < 0.3), this is a finding *in its own right* (the substrate's channels do not have stable Doya-side identities) but does not refute the gate.

---

## §D. Minimum publishable pain claim at the current empirical stage

The current empirical state:
- R2 continual anchor: NMN (the modulator-augmented agent — using configs `recurrent_ppo_nmn_film_g1_tempceil5.yaml` and `recurrent_ppo_nmn_het_unmod.yaml` for the unmodulated baseline) beat baseline by approximately 25× the seed-noise floor on returns-to-active-predator stages.
- Single seed, single run on n106 RTX 3090, 5.1M episodes.
- No probe-trial extension yet.
- No placebo or expectation manipulation.
- No T/P split landed yet (Phase 0 prerequisite per v4 §3).
- No C1 EMA baseline run yet.

**Minimum publishable pain claim, one sentence:**

> *Interoceptive neuromodulation produces a behavioural pattern qualitatively consistent with the recovery-speed and post-regime-change-protection signatures predicted by the predictive-coding-of-pain framework (Büchel 2014; Wiech 2016; Tabor & Burr 2019) when interoception is treated as a slow latent driving policy adaptation; the four-dissociation gate that would distinguish this pattern from generic continual-learning consolidation is pre-registered and pending the probe-trial extension.*

What the R2 data *can* say:
- The modulator's behavioural signature is "recover faster on return to a previously-encountered active-predator stage" and "second return outperforms first return by approximately 6 survival steps, while baseline shows monotone decay of approximately 20 steps". This pattern is *consistent with* (but not exclusive evidence of):
  - **A continual-learning-resistance reading** (the baseline framing the R2 anchor was originally written under).
  - **An interoceptive-modulation reading** (the v4 framing — a slow state-dependent hyperparameter shift produces faster recovery).
  - **A weak pain-relevant reading** under predictive-coding-of-pain — the modulator's persistent state acts as a prior over future predator encounters, raising precision on threat-relevant evidence on re-encounter, which is *one* mechanism the pain literature proposes for post-injury hypervigilance.

What the R2 data *cannot* say:
- Whether the gain pattern is channel-selective (dissociation 1) — not measured.
- Whether the gain is time-locked to injury rather than to predator presence (dissociation 2) — not measured at the resolution needed; the per-stage-mean numbers in the memory insight collapse across the within-stage timecourse.
- Whether perceptual / mnemonic / policy effects are jointly modulated within-trial (dissociation 3) — single `mod_h` makes this by construction; testable only under T/P split.
- Whether the effect is dissociable from a slow-EMA baseline (dissociation 4) — C1 not implemented.
- Whether the persistence is a chronic-pain analog (Vlaeyen-Linton; Apkarian 2009) versus recurrent inertia — requires the post-recovery probe-trial extension that is not yet in scope.

**Honest framing.** The R2 anchor is *motivating evidence* that the substrate produces behaviour worth investigating under the pain-construct framework. It is *not yet* evidence that the substrate computes pain. The minimum publishable claim has to be honest about this — possibility-framing makes the honesty cheap (the paper isn't claiming full coverage anyway), but it doesn't make the honesty optional.

---

## §E. New ideas for the symposium

### E.1 *Behaviourally-defined construct gate* — published as the paper's pre-registered claim-validity protocol

**Idea.** The paper publishes the §C re-stated four-dissociation gate *as the construct-validity protocol* — explicitly, as a methodological contribution in its own right. Framing: "the field has never had a behaviourally-defined gate for distinguishing pain-construct content from generic aversion in an RL agent; here is ours; the FiLM-substrate result satisfies $k$ of $n$ of these dissociations, and we pre-register $n - k$ for the next paper". Even if only 1–2 of the 4 dissociations land on R2 + T/P-split data, *publishing the gate as a methodological contribution* generates citation traffic from anyone else doing pain-RL work later. This is the same move Vlaeyen & Linton 2000 pulled with the fear-avoidance model itself — published as a framework, not as a single empirical result, and now is the most-cited paper in the field.

**Fit to user's constraints.** Possibility-framing satisfied (the gate is the *test* of possibility); no-one-to-one satisfied (the §C re-statement is channel-agnostic).

### E.2 *R2 reframe — "target-flexible interoceptive modulation"* as the project's specific contribution

**Idea.** The current framing reads the R2 anchor as "continual-learning recovery". The interoceptive-modulation reframe reads it as "target-flexible response of a single slow state to a regime change". Pain-construct language without channel commitment. The reframe matters because **the project's testbed-specific contribution** — that the regime change is *interoceptive* (injury) rather than *exteroceptive* (predator presence) — is what makes this a pain-relevant result rather than a generic continual-learning result. Lee 2024 and Wang 2024 do continual learning under exteroceptive non-stationarity; the project is the first (in the v4 corpus) to do continual-learning-style recovery under *interoceptive* non-stationarity. The pain-construct language for that is "the agent's interoceptive prior adapts on recovery from injury, producing faster re-engagement with a previously-aversive context" — clinically, this is the Vlaeyen-Linton "concern-dependent" learning signature.

**Fit to user's constraints.** Possibility-framing satisfied (the reframe is one possible reading, not the only one). No-one-to-one satisfied (the claim is at the behavioural-pattern level, not at any specific channel).

### E.3 *Behavioural placebo-shadow probe* — a sub-experiment that adds pain content without architectural extension

**Idea.** The full placebo / cue-injury contingency-learning probe is out-of-scope for the current paper (v4 §5.10; my v3-round P-Pain-6). But there is a *smaller* sub-experiment that adds pain-relevance without requiring a new sub-task: **the post-recovery predator-cue probe trial with no injury risk** (already named in v4 §5.8 as a "small senior-developer add"). My re-read: this probe is a *behavioural placebo shadow*. It tests whether the modulator's persistent state produces *avoidance behaviour in the absence of injury risk* — the prior moves without the input changing. This is the cleanest current-testbed-feasible operationalisation of the predictive-coding-of-pain decomposition $p(\text{pain}|\text{stim}) \propto p(\text{stim}|\text{pain}) \, p(\text{pain}|\text{expect})$: the second term moves on the probe trial without the first term changing.

If the probe lands (the modulator-augmented agent avoids near-injury-site regions more than the baseline does, even when injury risk is demonstrably zero), this is **single-paper sufficient evidence** for one of the four dissociations under §C: the third (joint perceptual–mnemonic–policy reweighting tied to a prior, dissociable from the input). The probe is a small-to-medium senior-developer add. If the user is willing to scope it in before submission, the paper moves from Position B to Position A on this dimension alone.

**Fit to user's constraints.** Possibility-framing satisfied (the probe demonstrates the *possibility* of input-dissociable behaviour, not the full placebo decomposition). No-one-to-one satisfied (the probe measures behavioural avoidance, not channel-level computation).

---

## §F. Construct-validity gate-keeping summary — what does NOT count as pain-relevant under possibility-framing

This is the deliverable only I write. Possibility-framing relaxes the bar but does not abolish it. The following behavioural patterns will be tempting to read as pain-relevant when the paper drafts; under the §C re-stated gate, they do *not* count.

**The pattern of behavioural signatures that does *not* count as pain-relevant under possibility-framing — even when the framing is loose.**

A survival-step advantage of the modulated agent over the unmodulated baseline on continual-learning recovery (the R2 anchor itself) is *consistent with* a hypervigilance reading but *also consistent with* generic continual-learning consolidation under Durstewitz-2025 mechanisms, with NA-as-context-detector under Wainstein-2025, with target-flexible inverse-temperature modulation under Doya-NA, and with simple modulator-state inertia under any slow-recurrent-architecture; it is therefore not evidence on pain-construct content (P-Pain-5a). A modulator burst at injury onset is the canonical unexpected-uncertainty signature (Aston-Jones & Cohen 2005; Wainstein 2025) and carries no pain content on its own (P-Pain-5b). Slow recovery of survival after injury is consistent with any slow-state architecture and is not chronic-pain-analog without the cross-context transfer probe (P-Pain-5c). Increased aversion to nociceptive input is what reward shaping on the nociception channel produces (Brand & Yancey 1997; Cox et al. 2006); pain content requires dissociability from the input — the posterior moves when the prior moves without the input changing — which is the §E.3 probe-trial direction (P-Pain-5d). Continual-learning resistance per se is *one* readout of state-dependent hyperparameter modulation, not pain content; the v4 redirect away from "continual learning" as the paper's headline is correct precisely because of this (P-Pain-5e). And — new under possibility-framing — *qualitative narrative similarity* between the modulated agent's behaviour and clinical pain phenomenology, *unsupported* by any of the four §C re-stated dissociations, does *not* clear the possibility bar; "possibility" means "one defensible computation runs on the substrate", not "the substrate's behaviour can be re-described in pain-language". The reviewer's question will be "what is the *test* of the possibility claim, and what would refute it?" and the answer has to be one or more of the §C dissociations, not the narrative.

The possibility-framing relaxes coverage. It does not relax specificity, falsifiability, or dissociability from generic-aversion alternatives. The §C re-stated gate is the bar. Anything below it is not pain-relevant under the user's constraints, however suggestive the behavioural pattern looks.

---

## Cross-references

- v4 direction memo: [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md`](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md) — §5.7 four-dissociation gate; §5.8 testbed-adequacy boundary; §5.10 future-work register.
- Construct-validity foundation: [`docs/project/concepts/pain_vs_nociception_construct.md`](../../concepts/pain_vs_nociception_construct.md) — §2.2 four-property hypervigilance fingerprint; §3 three-control ladder C0/C1/C2; §4.4 sober vs strong register.
- Bayesian-brain memo (sister concept): [`docs/project/concepts/active_inference_hypervigilance.md`](../../concepts/active_inference_hypervigilance.md) — §1.4 mapping to injection sites.
- FiLM substrate lineage: [`docs/project/concepts/film_neuromod_integration.md`](../../concepts/film_neuromod_integration.md); [`docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md`](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md).
- R2 empirical anchor: [`.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md`](../../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md).
- Study report: [`docs/experiments/summaries/20260513_0321_nmn_comparison_study.md`](../../../experiments/summaries/20260513_0321_nmn_comparison_study.md).
- My v3-round contribution: [`docs/project/references/neuromodulatory_algorithms/feedback/20260516_pain_modeling_predictions_for_v3.md`](../../references/neuromodulatory_algorithms/feedback/20260516_pain_modeling_predictions_for_v3.md).

**Canonical pain literature anchors:**
- Eccleston & Crombez 1999 — interruption hypothesis (attentional specificity of pain, not generic aversion).
- Crombez, Van Damme & Eccleston 2005 — hypervigilance as concern-dependent attentional bias.
- Vlaeyen & Linton 2000; Vlaeyen, Crombez & Linton 2016 — fear-avoidance model (joint perceptual-mnemonic-policy reweighting).
- Büchel et al. 2014 — placebo analgesia as predictive coding (the prior-shift / posterior-shift dissociation underwriting §E.3).
- Wiech 2016 — pain as constructed perception; the predictive-coding-of-pain programme.
- Tabor & Burr 2019 — Bayesian learning models of pain (prior × likelihood decomposition).
- Apkarian, Baliki & Geha 2009 — chronic pain as failure mode of an adaptive controller (§B.4 future-work framing).
- Asmundson et al. 1997 — attentional bias to threat in chronic pain.

---

*Contribution by `professor-pain-modeling`, 2026-05-16.*
