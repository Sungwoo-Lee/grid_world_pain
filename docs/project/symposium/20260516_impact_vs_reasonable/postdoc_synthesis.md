---
title: "Postdoc synthesis of the impactful-vs-reasonable symposium (20260516)"
session: 2026-05-16_impact_vs_reasonable
synthesiser: research-postdoc
date: 2026-05-16
inputs:
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_neuromodulation_contribution.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_bayesian_brain_contribution.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_rl_bayesian_dl_contribution.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_pain_modeling_contribution.md
user_refined_constraint: "Targeting a specific neuromodulatory system is valid for a single paper (without a target the paper is uninterpretable); full coverage of all neuromodulator-function mappings by one-to-one is not (mathematically impossible because biological function is overlapped and distributed). Middle path: target one (e.g., NA / LC), acknowledge one-to-many, defer full coverage to future work."
output_format: 2-4 candidate paper framings with impactful-vs-reasonable trade-off explicit per framing
downstream: docs/pi/calls/2026-05-16_impact_vs_reasonable_call.md (PI will write next)
---

# Postdoc synthesis of the impactful-vs-reasonable symposium (20260516)

## §1 — Plain-English entry point

The user called a four-professor symposium today to decide one portfolio-level question: where on the *impactful-vs-reasonable* spectrum should the next paper sit? After the four contributions landed, the user clarified one of their two original constraints, and this synthesis applies that refinement consistently — the professors did not have it.

What the four professors agreed on, in one sentence: **the project's architecture (a small "modulator" network that emits multiplicative-scale and additive-shift signals into three places in the policy: the perceptual encoder, the memory-cell gate, and the action-selection logits) is fine and does not need redesign; the way the v4 direction memo *describes* that architecture is the problem and is fixable by a rewrite alone, with no code, no experiments, and no equation changes**. Each professor said this in their own vocabulary: the neuromodulation lens calls it *target-as-basket-of-effects*; the Bayesian-brain lens calls it *one-signal-many-precisions*; the RL / Bayesian-DL lens calls it *observation-point reframing* (with the Galanti & Wolf 2020 modularity theorem as the formal anchor); the pain-modeling lens calls it *target-flexible interoceptive modulation*. These are four names for the same recommendation.

The user's refined constraint then resolves a question the professors left half-open: a paper that names no biological target is uninterpretable; a paper that claims full coverage of all biological-neuromodulator-to-RL-knob mappings is mathematically impossible. The right move is to **target one biological system explicitly** and **acknowledge that the targeted system produces multiple downstream effects, with full coverage deferred**. Given the project's empirical anchor — the modulator-augmented agent recovered roughly 25× the seed-noise floor faster than the unmodulated baseline on the second return to a previously-seen task stage — the natural target is the **noradrenergic locus coeruleus (NA / LC) system**, because phasic burst at regime change is the canonical LC signature (Aston-Jones & Cohen 2005).

The recommended candidate framing is **NA-targeted multi-effect demonstration with optional pain-construct upgrade**. The rest of this memo derives it, lays out three alternatives with explicit trade-offs, and lists the rhetorical-only edits a v5 of the direction memo would absorb.

---

## §2 — The symposium's load-bearing convergence

Four lenses, one recommendation. Reading the four professor memos side-by-side, the substantive claims line up almost too cleanly: each professor approached the user's constraints from a different starting framework, and each arrived at the same architectural verdict and the same rhetorical correction. The convergence is the symposium's strongest output — when four independently-trained domain experts arrive at one position from four different paths, the position is load-bearing.

| Lens | Formulation of the recommendation | Convergent claim |
|---|---|---|
| Neuromodulation (substrate-side) | **Target-as-basket-of-effects** — one upstream signal acting on multiple targets, each target carrying several effect-flavours via target-specific receptor-density fan-out | Substrate right; channel-attribution rhetoric wrong |
| Bayesian-brain (theory-side) | **One-signal-many-precisions** — one precision-control signal acts across hierarchical levels, each level producing a different downstream effect | Substrate right; channel-attribution rhetoric wrong |
| RL / Bayesian-DL (architecture-side) | **Observation-point reframing** — sites A/B/C are not channels but observation points on the effect of one modulator, with dissociation living in per-site read-out matrices (Galanti & Wolf 2020 Theorem 4) | Substrate right; channel-attribution rhetoric wrong |
| Pain-modeling (construct-side) | **Target-flexible interoceptive modulation** — the dissociation gate is re-stated in behavioural-effect language, channel-agnostic | Substrate right; channel-attribution rhetoric wrong |

### 2.1 The neuromodulation reading in one paragraph

The brain's neuromodulators (acetylcholine / ACh, noradrenaline / NA, dopamine / DA, serotonin / 5-HT) are not single-function. A single locus-coeruleus burst affects arousal, signal-to-noise gain, attentional bias, exploration-vs-exploitation balance, memory consolidation, and even descending pain modulation — all from one nucleus, with the downstream effect at each cortical site depending on which adrenergic-receptor subtypes that site expresses (Aston-Jones & Cohen 2005; Sara 2009; Ferguson & Cardin 2020). The v4 direction memo's per-site labelling — *"Site A = ACh-as-α (learning rate); Site C = NA-as-β (inverse temperature)"* — is itself the same one-to-one error as the older Doya 2002 four-channel framing, just at a finer granularity. The correct framing under one-to-many: each FiLM site is *one target* of a single modulator, and at each target the modulator produces a *basket* of effects (the encoder receives learning-rate-like, precision-like, and divisive-normalisation-like effects simultaneously; the policy receives temperature-like, action-prior-shift, and attentional-bias effects simultaneously; and so on). The FiLM read-out matrix is the in-silico analogue of receptor-density fan-out.

### 2.2 The Bayesian-brain reading in one paragraph

In predictive-coding generative models (Friston 2005, 2010; Rao & Ballard 1999; Bastos et al. 2012), the agent maintains a hierarchical posterior over hidden states, and at each level prediction errors are weighted by a *precision* matrix. Precision is the theoretical home of neuromodulatory gain — and the theory does not commit a single biological neuromodulator to a single precision level. Yu & Dayan 2005 maps ACh to expected uncertainty *and* NA to unexpected uncertainty, with the two systems jointly controlling different combinations of precisions over state and action — one signal, many precisions, no one-to-one. The project's three FiLM injection sites can be read directly as three precision read-outs from one upstream modulator state $h_{\text{mod}}$: sensory precision $\Pi_s$ at the encoder, state-transition precision $\Pi_z$ at the GRU update gate, policy precision $\Pi_\pi$ at the policy logits. The architecture *is* the one-signal-many-precisions abstraction by construction; biological-neuromodulator labels become interpretive overlays, not the architectural skeleton.

### 2.3 The RL / Bayesian-DL reading in one paragraph

The Galanti & Wolf 2020 modularity theorem (Theorem 4) is the formal grounding for why a hypernetwork — and a FiLM substrate is a restricted hypernetwork — produces dissociable per-target effects without the conditioning signal needing to decompose into separable channels. The theorem's standard reading is *parameter efficiency*: $N_g = O(\epsilon^{-m_1/r})$ for a hypernet vs $\Omega(\epsilon^{-(m_1 + m_2)/r})$ for embedding-concatenation. The under-used second reading is *identifiability*: the parameter efficiency comes from the read-out matrices carrying the per-target specificity while the conditioner is shared, which means **the dissociation lives in the read-out, not in the signal**. A sceptical reviewer asking "if one signal produces three dissociable effects, what *are* the three effects in the signal itself?" has a formal answer: nothing — the signal is low-dimensional; the effects live in the read-outs; this is exactly what hypernetworks are for. Row EE-6 in the concept memo (FiLM-Ensemble plus heteroscedastic-precision compound) is the project's strongest architectural artefact and reads cleanly as the first approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy.

### 2.4 The pain-modeling reading in one paragraph

The four-dissociation gate that decides whether the project can use pain-construct language in the headline was originally stated in channel-attribution terms (*"channel-selective γ at site A"*, *"site-B memory shift"*, *"site-C policy shift"*). Under one-to-many, that gate cannot survive — it implicitly commits to which Doya channel implements which clinical operationalisation of hypervigilance. The pain-modeling memo re-states the gate in *behavioural-effect language* that is channel-agnostic: dissociation 1 becomes "the post-injury gain pattern on threat-relevant input modalities is non-uniformly elevated, regardless of which injection site carries the elevation"; dissociation 3 becomes "perceptual, mnemonic, and policy effects are co-modulated within a trial through a common slow state, whatever Doya channels that state instantiates". The re-stated gate is actually *more biologically defensible*, because real neuromodulators *do* jointly affect multiple functions — joint within-trial coupling is the expected signature of any biologically-realistic modulator, not a one-to-one channel commitment.

### 2.5 What converges and what does not

**Strong convergence (the load-bearing finding).** All four professors agree on: (a) v4's substrate is architecturally fine; (b) v4's experimental program (per-site clamps, the 3×3 dissociation matrix, the T/P split, the Hessian probe) is unchanged under the rewrite; (c) the rhetorical commitment to a channel-to-site mapping (*"α at A, β at C"*) is the one thing that needs to be softened; (d) re-framing is a zero-code, zero-experiment, zero-equation change — pure rewrite. v5 of the direction memo absorbs the convergence.

**Soft divergence (easy to merge).** The lenses differ on *which lens leads the paper*: neuromodulation wants the substrate-side biological framing (B.3 receptor-density-as-readout + B.2 single-demonstration-of-multifunctional-gain); Bayesian-brain wants precision-coding (B1 light + B3 architectural realisation; B2 active inference deferred to discussion); RL / Bayesian-DL wants Row EE-6 as the architectural headline (Framing C2); pain-modeling wants a Position-B-default-Position-A-conditional hybrid. These are not contradictory — each is a different *front-end* on the same underlying paper. The candidate framings in §4 below correspond directly to the lens that leads.

**No hard contradictions.** I did not find a single load-bearing claim in one professor's memo that another professor's memo refuted. The closest was the neuromodulation memo's pushback against the RL / Bayesian-DL memo's Framing C2 (Row EE-6 as architectural headline), which the neuromod memo called "post-hoc rationalisation of a choice motivated by target-specific gain on a multifunctional modulator". This is a positioning disagreement, not a substantive one — both agree the substrate is correct; they disagree on which audience the paper should lead with.

---

## §3 — Applying the user's refined constraint

The professors had only the first version of the user's second constraint: *"no one-to-one neuromodulator-function mapping"*. Three of the four memos read this maximally — *"no channel attribution at all"* — and recommended dropping all biological-system-specific framing in favour of either abstract precision-coding (Bayesian-brain), abstract observation-points (RL / Bayesian-DL), or abstract receptor-density-fan-out (neuromodulation). The pain-modeling memo was more cautious, re-stating its dissociation gate in channel-agnostic behavioural language while keeping pain as the load-bearing motivation.

The user's refined constraint, surfaced after the four memos landed, says something subtler than the maximal reading: **targeting one biological system is valid and necessary** (without a target, the paper is uninterpretable to a biological reviewer and the project's downstream identity in pain modelling does not survive); **full coverage of all neuromodulator functions by one-to-one mapping is invalid** (because biological functions are overlapped and distributed). Targeting a single system → fine. Claiming the substrate covers all four Doya channels by one-to-one mapping → not fine.

### 3.1 Why a target is necessary

A paper that says only *"a learned modulator produces dissociable behavioural effects at three FiLM sites"* — with no biological-system-specific framing — is interpretable to an RL / Bayesian-DL audience (the audience the C2 framing targets), but is genuinely uninterpretable to the biological reviewers whose buy-in the project's downstream identity in computational pain modelling depends on. *Which* biology does this substrate computationally instantiate? *Why* this architecture rather than another? The answer cannot be *"some unspecified neuromodulator"* — that is precisely the rhetorical hole that lets a sceptical reviewer dismiss the paper as biology-flavoured RL. The Doya 2002 framework was a 23-year-old normative simplification, but its enduring contribution was the recognition that biologically-grounded RL needs to *name* the biological system it claims as inspiration; otherwise the cellular-to-behavioural bridge is unmotivated.

### 3.2 Why full coverage is impossible

A paper that claims *"the substrate carries ACh-as-α at site A, plasticity-gating at site B, NA-as-β at site C, and acknowledges 5-HT-as-γ is out-of-substrate"* — which is the v4 framing — is making four separate one-to-one channel commitments. Each one is biologically wrong: ACh is multifunctional (learning rate, expected-uncertainty precision, attentional gain, divisive normalisation, memory consolidation), NA is multifunctional (inverse temperature, gain, working-memory stability, large-scale brain-state switching, descending pain modulation), and so on. v4's narrowing from v3's four-channel to a two-channel claim was a real improvement, but *each of those two channels is itself a one-to-one commitment at the per-site level*. The neuromodulation memo flagged this clearly: v4 walked back Doya's four-channel mapping but did not walk back the same error one level down.

### 3.3 The right middle path: target one, acknowledge many, defer the rest

Given the project's empirical anchor — the modulator-augmented agent recovered roughly 25× the seed-noise floor faster than the unmodulated baseline on the second return to a previously-seen task stage, with the modulator's `temperature_mean` swinging ≥3σ at three of four stage transitions — the natural targeted system is **noradrenaline / locus coeruleus (NA / LC)**. The Aston-Jones & Cohen 2005 adaptive-gain theory's canonical phasic-burst signature is exactly what the R2 anchor shows: a one-nucleus signal bursting at regime change, driving a recovery-speed advantage on return-to-active stages, with the modulator emitting sign-consistent phasic shifts at exactly the stage boundaries the theory predicts.

What this targeting buys the paper:
- **Interpretability.** The paper is framed as a computational demonstration of NA-LC-inspired multi-effect gain modulation. A biological reviewer can place the contribution within the field's existing literature on LC's role in arousal, gain, and regime detection.
- **Empirical anchoring.** The R2 anchor is exactly the kind of result the Aston-Jones theory predicts. The narrative writes itself.
- **One-to-many honesty.** The targeted system (NA / LC) is acknowledged as multifunctional: the targeted modulator produces multiple downstream effects on the substrate, measured at three injection sites, and we do not claim these are *the* canonical NA effects, only that they are *consistent with* the kinds of effects NA-LC is known to produce.

What this targeting defers:
- **ACh, DA, 5-HT modulation.** Each named as out-of-scope-for-this-paper future work. The project's downstream identity has room to extend in any of these directions later — but trying to cover all four in one paper is what creates the one-to-one trap.
- **Receptor-subtype dissociation** ($\alpha_1$, $\alpha_2$, $\beta_1$, $\beta_2$). The FiLM read-out matrices play the receptor-density-fan-out role at an abstraction level above subtype-specificity. A future paper could lift the architecture to predict subtype-specific dissociation; this one does not.

### 3.4 The refined constraint applied to each professor's framing

- **Neuromodulation memo's B.3 + B.2 recommendation** (receptor-density-as-readout + single-demonstration-of-multifunctional-gain) survives the refined constraint cleanly, *with one anchoring move*: name the targeted modulator as NA-LC explicitly in §1 and §3 of v5. The basket-of-effects framing then applies to the NA-LC-analog: one upstream NA-analog signal, three target sites, each target producing a basket of effects.
- **Bayesian-brain memo's B1 + B3 recommendation** (precision-weighted FiLM as headline + Row EE-6 as architectural realisation; B2 active inference in discussion) survives the refined constraint cleanly, *with one anchoring move*: explain in §1 that the precision-weighting language is the *theoretical home* of the NA-LC gain signal the architecture is targeting. Precision-coding is then read as the formal framework for "what NA-LC does computationally", not as a competing biological story.
- **RL / Bayesian-DL memo's Framing C2** (Row EE-6 as architectural headline; neuromodulation as motivation) is the framing that most cleanly satisfies the *no one-to-one* part of the refined constraint, but as written it satisfies the *target* part only weakly — neuromodulation is "motivation only". To satisfy both: keep Row EE-6 as the architectural contribution, but commit to NA-LC as the biological target in §1's motivation and in §6's discussion-of-implications. The substrate then has both an architectural identity (BHN-with-heteroscedastic-likelihood) and a biological identity (NA-LC-analog gain modulation), and the paper can be read by both audiences.
- **Pain-modeling memo's Hybrid B + soft A** survives the refined constraint cleanly. The behaviourally-defined gate is channel-agnostic by construction; the targeted-system commitment to NA-LC is upstream of the gate and does not interfere with it. The pain-modeling memo's Position A reach is conditional on the probe-trial extension landing — this is unchanged.

### 3.5 The reframing's net effect on impact

Targeting NA-LC and acknowledging one-to-many is actually a *stronger* impact claim than v4's two-cleanly-carried-channels framing, not a weaker one. v4 says: *"the substrate carries two specific Doya channels"*. The reframed claim says: *"the substrate computationally instantiates the principle — found across biology's neuromodulator systems but exemplified by NA-LC — that one signal produces multiple effects via target-specific read-outs"*. The first is a narrow claim about which two of four Doya channels are present. The second is a broader claim about a structural feature of biological neuromodulation, with NA-LC as the specific exemplar and the substrate as the computational demonstration. **Breadth comes from committing less to channel-assignment**, paradoxically.

---

## §4 — Candidate paper framings

Each framing below absorbs different combinations of the four professors' recommendations and applies the user's refined constraint consistently. They are ordered by increasing distance from v4 of the direction memo. Each framing names: the headline claim, the structural commitment, the evidence the project's current state supports, the evidence still needed, the position on the impactful-vs-reasonable spectrum, and the downstream cost.

### Framing 1 — NA-targeted multi-effect demonstration (Reasonable, with one anchoring move)

**Headline.** "A FiLM-modulated actor-critic agent demonstrates the *possibility* that a single NA-LC-analog gain signal produces multiple behavioural-hyperparameter-like effects on an interoceptive RL substrate, with the three injection sites instantiating receptor-density-style fan-out of one signal to three targets."

**Structural commitment.** Targets noradrenaline / locus coeruleus explicitly in §1 and §3; absorbs the neuromodulation memo's B.3 (receptor-density-as-readout) plus B.2 (single demonstration of multifunctional gain); keeps the Bayesian-brain precision-coding reading as one of three interpretive lenses in §3.5 (alongside Doya-hyperparameter and observation-point readings); acknowledges in §5.9 that ACh, DA, and 5-HT modulation are explicitly out of scope.

**Evidence the project's current state supports.** R2 anchor — the modulator-augmented agent recovered ~25× the seed-noise floor faster than the unmodulated baseline on return-to-active stages, with the modulator's `temperature_mean` swinging ≥3σ at three of four stage transitions. This is the canonical Aston-Jones & Cohen 2005 phasic-LC-burst signature instantiated in an end-to-end-trained agent. The mechanism check (modulator mechanistically engaged at stage boundaries, sign-consistent) is a direct empirical fit.

**Evidence still needed.** v4 §5.1 four-arm clamp at site C (γ-clamp / β-clamp / both-clamp). v4 §5.4 per-site freeze dissociation matrix (the 3×3). v4 §5.5 T/P split as Phase 0 prerequisite (a single `mod_h` is non-identifiable for the multi-effect claim). Three seeds on R2 to seed-lock.

**Position.** Reasonable end of the spectrum. The headline is "demonstration of possibility", not "verification of bridge". Compatible with a NeurIPS / TMLR-shaped 8–10 page paper or a *Neural Computation* / *PLOS Computational Biology* submission, depending on which lens the paper leads with.

**Downstream cost.** Zero code, zero new experiments beyond the v4 §5 program. Pure v5 rewrite of the direction memo. The probe-trial extension is *not* required for the primary claim.

### Framing 2 — Approximate Bayesian Hypernetwork as one-signal-many-precisions substrate (Middle / architectural-lead)

**Headline.** "First instantiation of an approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy, demonstrating that posterior variance over a neuromodulator-inspired side-signal produces dissociable behavioural-effect readouts at three injection sites of an interoceptive agent — consistent with the noradrenergic-gain-modulation literature."

**Structural commitment.** Absorbs the RL / Bayesian-DL memo's Framing C2 (Row EE-6 as architectural headline); keeps NA-LC as the biological target in §1's motivation; uses Galanti & Wolf 2020 Theorem 4 as the formal identifiability anchor for one-to-many readouts (RL / Bayesian-DL memo's E2); deploys the FiLM ⊂ Hypernet ⊂ point-mass-BHN-limit lineage from the investigation memo as the architectural ancestry; defers pain-construct content to discussion / future work; keeps the Bayesian-brain precision-coding language for the discussion of "what the architecture computes".

**Evidence the project's current state supports.** Row EE-6's lineage is fully derived in the investigation memo's five formal claims; the math-reviewer audit has cleared the substrate's coherence; the R2 anchor supports the "single side-signal produces dissociable behavioural effects" claim at the headline level.

**Evidence still needed.** Row EE-6 itself has not been implemented (FiLM-Ensemble at the actor head + heteroscedastic precision head + the $\rho$-init choice + the variance-space disambiguation). The four-arm clamp at site C, the 3×3 dissociation matrix, the T/P split (as Phase 0 prerequisite). Behavioural-effect heterogeneity statistics (effective rank of γ, cross-site cross-correlation structure).

**Position.** Middle. Architecturally the strongest claim the project can defend; biologically lighter than Framing 1 (NA-LC is in the motivation, not the headline). NeurIPS / ICLR architecture track; AABI; *TMLR*.

**Downstream cost.** Row EE-6 implementation (non-trivial — a new head plus ensemble harness); same experimental program as Framing 1 plus the Row EE-6 ablations. The neuromodulation memo's pushback against C2 — "post-hoc rationalisation" — applies here at the rhetorical level and would have to be answered in the paper's discussion.

### Framing 3 — NA-targeted with interoceptive-pain motivation and conditional hybrid pain gate (Reasonable + soft impactful)

**Headline.** "A FiLM-modulated actor-critic agent demonstrates the *possibility* of a single NA-LC-analog gain signal producing behavioural-hyperparameter-like effects under interoceptive non-stationarity, with the dissociation profile across three injection sites consistent with the kinds of multi-effect modulation the predictive-coding-of-pain framework predicts for post-injury hypervigilance."

**Structural commitment.** Framing 1 plus the pain-modeling memo's hybrid Position-B-default-Position-A-conditional. Writes the paper in Position-B register (NA-LC as the load-bearing biological motivation; pain in the introduction and the discussion only); conditional on the post-recovery probe-trial extension landing before submission, promotes the pain-construct gate into the body (Position A). Absorbs the §C re-stated four-dissociation gate verbatim — the behavioural-effect language is channel-agnostic and survives the refined constraint cleanly.

**Evidence the project's current state supports.** Same as Framing 1, plus the pain-modeling memo's "minimum publishable pain claim" — "interoceptive neuromodulation produces a behavioural pattern qualitatively consistent with the recovery-speed and post-regime-change-protection signatures predicted by the predictive-coding-of-pain framework". The R2 anchor is *consistent with* but does not on its own *demonstrate* pain-construct content.

**Evidence still needed.** Framing 1's requirements, plus the post-recovery predator-cue probe-trial extension (v4 §5.8 — flagged as a "small senior-developer add"). The C1 EMA-of-nociception baseline implementation. ≥ 5 seeds for the within-trial coupling threshold of dissociation 3.

**Position.** Reasonable on the substrate claim; soft-impactful on the pain claim, conditional on the probe-trial landing. The paper is reviewable at *PAIN* / *Nature Mental Health* if the gate lands, and at NeurIPS / *Neural Computation* if it does not — the writing strategy does not lock the venue choice in advance.

**Downstream cost.** Framing 1's cost plus the probe-trial extension plus the C1 baseline plus the 5-seed requirement. Realistically 1–3 months of additional empirical work before submission, with the upside that the paper carries the project's downstream pain identity from the start rather than deferring it.

### Framing 4 — Pure hypernet-lineage paper, neuromodulation as motivation only (Impactful pole, architectural)

**Headline.** "FiLM ⊂ Hypernetwork ⊂ Bayesian-Hypernetwork-limit lineage made formal; Row EE-6 as an approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy, demonstrating behavioural-effect heterogeneity from a single context-conditional generator."

**Structural commitment.** Absorbs the RL / Bayesian-DL memo's E3 (the hypernet lineage as the paper itself). The five formal claims from the lineage investigation memo become the paper's contribution. NA-LC and pain content are motivation paragraphs only; the paper's bulk is architecture, lineage, and empirical demonstration on the RL substrate.

**Evidence the project's current state supports.** The five formal lineage claims are derived; the math-reviewer audit has cleared the substrate; the R2 anchor demonstrates behavioural-effect heterogeneity from one signal.

**Evidence still needed.** Row EE-6 implementation; the four-arm clamp and 3×3 dissociation matrix; the T/P split.

**Position.** Impactful pole on the architectural axis; reasonable pole on the biological / clinical axis (because biology is motivation only). NeurIPS BDL workshop, AABI, possibly ICLR; partially abandons the cellular-and-behavioural bridge that motivated the project's downstream identity.

**Downstream cost.** Same as Framing 2 minus the biological-anchoring rhetorical work — but with the substantive cost that the project's pain-modelling identity is silently deferred to a follow-up paper. The pain-modeling memo's §B.4 cautions explicitly against this: publishing the substrate result with no pain framing attached makes the *next* paper's pain claim harder to land, because reviewers will have read Framing-4-the-paper as "an RL architecture paper" and expect any follow-up to also be RL.

### Framing comparison

| Framing | Headline-leading lens | Target named? | Pain-construct content | Code/experiments needed | Downstream identity preserved? |
|---|---|---|---|---|---|
| 1 — NA-targeted multi-effect | Neuromodulation | NA-LC explicit | Motivation only | v4 §5 program + 3-seed R2 | Partially (NA-LC framing supports pain extension) |
| 2 — Approximate BHN substrate | RL / Bayesian-DL | NA-LC in motivation | Discussion only | v4 §5 + Row EE-6 impl | Weakly (BHN headline is RL-shaped) |
| 3 — NA-targeted + pain hybrid | Neuromodulation + pain | NA-LC explicit | Conditional headline | Framing 1 + probe-trial + C1 EMA + 5 seeds | Strongly (pain identity in paper) |
| 4 — Pure hypernet lineage | RL / Bayesian-DL | NA-LC mentioned only | Motivation only | v4 §5 + Row EE-6 impl | Weakly (architectural identity dominant) |

---

## §5 — The synthesis's recommended candidate

**Top recommendation: Framing 1 (NA-targeted multi-effect demonstration) as the baseline, with Framing 3 (pain-construct upgrade) as a conditional add if the user wants the probe-trial extension in scope before submission.**

The reasoning, drawing on all four professor contributions plus the user's refined constraint:

1. **Framing 1 is the unique point on the trade-off curve that absorbs the symposium's load-bearing convergence in its strongest form.** All four professors agreed that the substrate is right and the rhetoric needs softening; Framing 1 implements exactly that softening, with NA-LC as the targeted biological system answering the user's "without a target the paper is uninterpretable" condition, and with the basket-of-effects rhetoric satisfying the "no full-coverage by one-to-one" condition.

2. **Framing 1's evidence-needed list is the project's existing v4 §5 program plus 3-seed replication of R2.** No new architectural commitment, no new experiments beyond what v4 already names. The only additional move is the v5 rewrite of the direction memo (§6 below). This is the lowest-cost framing that preserves the project's downstream identity.

3. **The Aston-Jones & Cohen 2005 anchoring is non-trivial.** The R2 anchor reads cleanly as a computational instance of phasic LC bursting at regime change. The mechanism check (modulator `temperature_mean` swinging ≥3σ at three of four stage transitions, sign-consistent) is exactly the empirical pattern the adaptive-gain theory predicts. The paper writes itself around this anchor in a way it does not around any other biological system the project could plausibly target.

4. **Framing 3 is the upgrade path, not a competitor.** The post-recovery probe-trial extension (v4 §5.8) is the *single move* that lifts Framing 1 to Framing 3 — it tests dissociation 3 of the re-stated gate (joint perceptual-mnemonic-policy reweighting tied to a prior, dissociable from the input). If the user is willing to scope the probe-trial extension into pre-submission work, Framing 3 captures the project's pain-modelling identity from the start; if not, Framing 1 preserves the option of building the pain identity into the follow-up paper. The pain-modeling memo's hybrid recommendation is exactly this conditional structure.

5. **Framing 2 is a credible alternative, but the substantive cost is the project's biological identity.** Row EE-6 as architectural headline is the project's strongest pure-architecture move; it is also the move that pushes neuromodulation into "motivation only". For a project whose downstream identity is computational pain modelling, this is a strategic cost the user should weigh, not a default to accept.

6. **Framing 4 is the riskiest.** It maximises architectural impact at the price of silently deferring pain to a future paper that will have to re-establish the framing from scratch. The pain-modeling memo's §B.4 lays out exactly why this is a worse trajectory than Framing 1 or Framing 3.

**Recommended ordering of options for the PI to surface to the user**: Framing 1 as the baseline; Framing 3 as the probe-trial-conditional upgrade; Framing 2 as the architectural-track alternative; Framing 4 as a deliberate two-paper strategy if the user is willing to write two papers rather than one.

---

## §6 — Reframing actions for v5 of the direction memo

The convergence is rhetorical-only. v5 of the direction memo absorbs the symposium's recommendations as a sequence of edits to v4, no code or experiment changes. The compact list:

1. **§1 — Name the targeted biological system explicitly.** Change v4's *"the NMN's algorithm — multiplicatively scale and additively shift the policy's internal activations — is simultaneously the cellular-level account of biological neuromodulation (gain control on target neurons) and the behaviour-level account of how two of Doya 2002's four RL-hyperparameter channels get modulated"* to a formulation that targets NA-LC: *"the NMN's algorithm computationally instantiates the noradrenergic-gain-modulation principle (Aston-Jones & Cohen 2005) — one upstream signal producing multiple behavioural-hyperparameter-like effects on the substrate via target-specific FiLM read-out matrices, consistent with the receptor-density fan-out by which a single nucleus produces heterogeneous downstream effects in the brain"*.

2. **§3 — Drop the per-site channel-attribution rhetoric.** Replace v4's *"Site A (encoder pre-fusion) — α channel. ACh-as-learning-rate"* and *"Site C (policy logits) — β channel. NA-as-inverse-temperature"* with target-flavoured language: *"Site A is one target of the modulator; the dissociation profile at A is consistent with α-like (learning-rate-like) and precision-like effects; alternative readings (divisive-normalisation, expected-uncertainty) remain admissible at this site"*. Same pattern at sites B and C. The neuromodulation memo's flag — that v4's per-site labelling is itself a one-to-one error at finer granularity — is the load-bearing motivation for this edit.

3. **§3.5 — Add an "Interpretive lenses" subsection.** List four lenses on the same substrate without privileging any one as load-bearing:
   - *Doya-hyperparameter lens* — the substrate's behavioural effects map onto multiple of Doya 2002's named channels, including learning-rate-like effects at A, plasticity-gating at B, and inverse-temperature-like effects at C; full coverage is not claimed.
   - *Precision-coding lens (Bayesian-brain)* — the three injection sites read as sensory $\Pi_s$ at A, state-transition $\Pi_z$ at B, policy $\Pi_\pi$ at C; one upstream modulator state $h_{\text{mod}}$, three precision read-outs, three different effects.
   - *Observation-point lens (RL / Bayesian-DL)* — sites A/B/C are three observation points on the effect of one side-signal acting through three different read-out matrices; the dissociation profile is a behavioural fingerprint, not a channel decomposition. Galanti & Wolf 2020 Theorem 4 is the formal identifiability anchor.
   - *Target-flexibility lens (pain-modeling)* — the modulator's effects are jointly distributed over perceptual, mnemonic, and policy substrates via a common slow state; the dissociation gate is stated in behavioural-effect language without channel commitment.

4. **§3.0 (lineage callout) — Cite Galanti & Wolf 2020 Theorem 4 for identifiability**, not only for capacity. Add a paragraph titled *"Architectural grounding for one-to-many readouts"* that gives the identifiability reading of Theorem 4 explicitly. (RL / Bayesian-DL memo's E2 — a one-paragraph addition that works under any framing.)

5. **§5.7 — Update the construct-validity gate to the pain-modeling memo's §C re-stated form.** The four dissociations are restated in behavioural-effect language:
   - Re-stated dissociation 1: *channel-selective gain shifts on threat-relevant inputs, regardless of injection site* — the post-injury gain pattern on encoder input modalities is non-uniform, with threat-relevant modalities receiving systematically larger upward shifts than threat-irrelevant ones, by a margin of $\Delta \gamma_{\mathcal{T} \setminus \mathcal{N}} \ge 0.2$ across ≥ 5 seeds.
   - Re-stated dissociation 2: *time-locked to interoceptive injury inference, not to threat input* — the gain pattern lags injury-channel onset by $\le \min(5, 2\tau_{\text{mod}})$ steps and is insensitive to predator-presence-change without injury.
   - Re-stated dissociation 3: *joint perceptual–mnemonic–policy reweighting through a common modulator state* — within-trial cross-domain coupling $r \ge 0.3$ at lag $\le 5$ between gain shifts at A, B, and C on the same trial.
   - Re-stated dissociation 4: *dissociation from a slow-input baseline (C1 EMA contrast)* — the modulated agent produces an across-channel gain pattern that the C1 EMA-of-nociception baseline cannot reproduce because the EMA has only one scalar to assign.

6. **§5.9 — Promote the no-map boundary to the load-bearing honesty section** under the refined constraint. The honest framing is no longer *"the substrate carries two of four Doya channels; γ_Bellman is out-of-substrate"* — it is *"the substrate targets NA-LC and instantiates the multifunctional-gain principle; ACh, DA, and 5-HT modulation are explicitly out of scope of this paper and named as future work; full coverage by one-to-one channel mapping is mathematically impossible and is not claimed"*.

7. **§8 contributors — Add the four symposium professors** (`professor-neuromodulation`, `professor-bayesian-brain`, `professor-rl-bayesian-dl`, `professor-pain-modeling`) as v5 contributors, alongside the v4 reviewers.

8. **New §X — "Multifunctionality readouts" (optional, from neuromodulation memo's D.1 and D.2).** Define the multifunctionality index $M$ (number of partial correlations above a pre-registered threshold between `mod_h` and reading-specific empirical signatures) and the target-specificity ratio $T$ (diagonal-dominance of the $3 \times 3$ per-site freeze deficit matrix). These are *new readouts on already-planned experiments*, not new experiments. $M \ge 4$ on three sites driven by one `mod_h` is the strongest biological-plausibility claim without committing to which four functions are the four. Failure mode $M = 1$ is still publishable.

The v5 rewrite is bounded: it touches §1, §3, §3.5 (new), §3.0, §5.7, §5.9, §8, and optionally adds a multifunctionality readout section. No equation changes. No experiment changes. No code changes.

---

## §7 — Open questions for the PI

The synthesis cannot resolve the following decisions; the PI should surface them to the user via the focus-vs-explore call.

### 7.1 Scope — single paper vs two-paper strategy

The recommended baseline (Framing 1) and its conditional upgrade (Framing 3) are both single-paper trajectories. Framing 4 (pure hypernet lineage) implies a two-paper strategy where the lineage paper publishes first and a pain-substrate paper follows. The pain-modeling memo's §B.3 lays out the cost of two-paper strategies — the pain paper has to re-establish framing reviewers have already seen, against a paper they read as RL — and recommends against it. The RL / Bayesian-DL memo's §E3 supports two-paper as the high-impact option but explicitly recommends Framing C2 (single paper) instead. **The PI should ask the user whether two papers is acceptable strategic cost for the architectural-impact gain.**

### 7.2 Pain-construct gate inclusion — Framing 1 vs Framing 3

The pivot point between Framing 1 and Framing 3 is the post-recovery probe-trial extension (v4 §5.8). It is a "small senior-developer add" but it is not trivial — it requires a new task variant (predator-cue probe with no injury risk), a new logged quantity (avoidance behaviour in absence of injury), and the C1 EMA-of-nociception baseline as the comparator. **The PI should ask the user whether the probe-trial extension is in scope for the paper or deferred to follow-up work.** Framing 1 is publishable without it; Framing 3 captures the project's pain identity from the start.

### 7.3 Venue family — biological vs architectural lead

Framing 1 reads naturally at *Neural Computation*, *PLOS Computational Biology*, *eLife computational neuroscience*, COSYNE, CCN. Framing 2 reads naturally at NeurIPS, ICLR, *TMLR*, AABI, NeurIPS BDL workshop. Framing 3 reads at *PAIN*, *Nature Mental Health*, *Computational Psychiatry* if the probe-trial lands — or at the Framing 1 venues if it does not. Framing 4 reads at the architectural venues only. **The PI should ask the user which audience the paper should lead with**, recognising that the audience determines the framing and the framing determines the rewrite.

### 7.4 Row EE-6 implementation timing

Framing 2 and Framing 4 both require Row EE-6 implementation before submission. Framing 1 and Framing 3 do not — Row EE-6 stays as a future-work item under those framings. The RL / Bayesian-DL memo recommends *not* implementing Row EE-6 until the paper scope is decided. **The PI should confirm with the user that Row EE-6 implementation is off the critical path unless the user chooses Framing 2 or Framing 4.**

### 7.5 The T/P split as Phase 0 prerequisite

All four framings require the T/P split (two recurrent modulator states with separable phasic and tonic timescales) to land before the multi-effect claim is identifiable — a single `mod_h` forces correlated movement on knobs biology dissociates. v4 §3 already names this as the Phase 0 prerequisite. **The PI should confirm with the user that Phase 0 T/P split is on the critical path regardless of which framing is chosen**, and that the wall-clock cost of landing the T/P split is acceptable for the chosen submission timeline.

---

## Cross-references

- v4 of the direction memo (what v5 will replace): [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md`](../../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md)
- Concept memo (the substrate's home document): [`docs/project/concepts/film_neuromod_integration.md`](../../concepts/film_neuromod_integration.md) — especially §3.0 lineage callout and §5.5 no-map boundaries.
- Lineage investigation: [`docs/project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md`](../../concepts/film_hypernet_bnn_lineage_and_critic_modulation.md) — five formal lineage claims.
- Active inference concept: [`docs/project/concepts/active_inference_hypervigilance.md`](../../concepts/active_inference_hypervigilance.md) — precision-update equation, expected-free-energy decomposition, site-mapping table.
- Pain construct: [`docs/project/concepts/pain_vs_nociception_construct.md`](../../concepts/pain_vs_nociception_construct.md) — four-property hypervigilance fingerprint, three-control ladder.
- Empirical anchor: [`.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md`](../../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md)
- Four professor contributions (symposium inputs): [`professor_neuromodulation_contribution.md`](professor_neuromodulation_contribution.md); [`professor_bayesian_brain_contribution.md`](professor_bayesian_brain_contribution.md); [`professor_rl_bayesian_dl_contribution.md`](professor_rl_bayesian_dl_contribution.md); [`professor_pain_modeling_contribution.md`](professor_pain_modeling_contribution.md).

## Hand-offs

- **`pi`** — owns the focus-vs-explore call. This synthesis recommends Framing 1 as the baseline with Framing 3 as the conditional upgrade; the PI surfaces the framing choice (and §7's open questions) to the user. The PI's call doc lives at `docs/pi/calls/2026-05-16_impact_vs_reasonable_call.md` (next).
- **`senior-developer`** — *not* invoked by this synthesis. The recommended baseline (Framing 1) requires no code changes. The §6 v5 rewrite of the direction memo is a research-postdoc / professor task, not a developer task. If the user picks Framing 2 or Framing 4, senior-developer is invoked for Row EE-6 implementation planning at that point. If the user picks Framing 3, senior-developer is invoked for the probe-trial extension scoping at that point.
- **`experiment-designer`** — *not* invoked by this synthesis. The v4 §5 experimental program is unchanged under the recommended framing. If Framing 3 is chosen, the probe-trial extension hands off to experiment-designer for design-doc authoring.
- **Four symposium professors** — invited to review the v5 rewrite of the direction memo once authored (v5 §8 contributors list). The rhetorical-only edits should not require new substantive contributions from them, but the four-lens §3.5 subsection is the kind of structural move that benefits from a professor pass before submission.

---

*Synthesis by `research-postdoc`, 2026-05-16.*
