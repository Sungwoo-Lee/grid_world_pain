---
title: "PI call — impactful vs reasonable framing for the first NMN paper"
date: 2026-05-16
session: 2026-05-16_impact_vs_reasonable
caller: pi
inputs:
  - docs/project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_neuromodulation_contribution.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_rl_bayesian_dl_contribution.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_bayesian_brain_contribution.md
  - docs/project/symposium/20260516_impact_vs_reasonable/professor_pain_modeling_contribution.md
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md
trigger: "User opened a fresh symposium to decide which direction is 1) impactful for integrating neural-gain-like algorithms with hyperparameter modulation and 2) reasonable for a single paper at the current empirical stage; explicit pace constraint: still in planning, no experiments yet."
status: pending-user-decision
---

# PI call — impactful vs reasonable framing for the first NMN paper

## §1 Plain-English entry point

The question on the table is **which shape the project's first paper should take** — concretely, *how it should describe what the agent does*, not what experiments to run, not what code to change. The team has built a small "modulator" — a side network that emits a scale-and-shift signal into three places inside a recurrent policy (one near the senses, one in memory, one at the action head) — and showed last week that this modulator helps the agent recover faster when it revisits a previously-seen task stage, by roughly 25× the noise floor between random seeds (the R2 anchor; details in [the NMN comparison study summary](../../experiments/summaries/20260513_0321_nmn_comparison_study.md)).

A four-professor symposium met today to ask: given that one empirical anchor, how should the paper be *framed*? The professors agreed unanimously — and from four different disciplinary starting points — on one thing: **the architecture and the planned experiments are fine; the way the current direction memo describes them (the v4 memo, [`20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md`](../../project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md)) commits to a one-modulator-to-one-RL-knob mapping at the per-site level that biology does not support and that the user's own constraint rejects**. The fix is rhetorical — a rewrite — not technical.

That convergence narrows the question to: *which target audience does the paper face, and how much pain-construct content do we put in the body vs. defer to follow-up?* The postdoc synthesis ([`postdoc_synthesis.md`](../../project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md)) proposed four candidate framings; this call surfaces them to you with my own ranking and the trade-offs called out.

**My recommendation, one line:** **Framing 1 — NA-targeted multi-effect demonstration** as the baseline, *without* the conditional pain-construct upgrade (Framing 3) at this point, because the pain upgrade adds 1–3 months of empirical work to a project that has already cycled through four versions of the direction memo in four days.

## §2 Where the project is

The project's first clearly-positive architectural finding landed three days ago. The modulator-augmented agent beat the unmodulated baseline on returns to a previously-seen task stage by roughly 25× the seed-noise floor, with the modulator's temperature output swinging sign-consistently at three of four stage boundaries — the canonical phasic-burst signature predicted by the locus-coeruleus / noradrenaline (LC-NA) adaptive-gain theory (Aston-Jones & Cohen 2005). Single seed each. Details: [memory insight `20260513_0014`](../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md); reader-facing summary in [the comparison study](../../experiments/summaries/20260513_0321_nmn_comparison_study.md).

Since that finding, the direction memo has gone through four versions in four days (v1 → v2 → v3 → v4), with a math-reviewer audit, a FiLM-Hypernet-Bayesian-Hypernet lineage investigation, an active-inference concept memo, and now a four-professor symposium with full postdoc synthesis ([`postdoc_synthesis.md`](../../project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md)). v4 says the modulator's algorithm — multiplicative scale and additive shift applied at three injection sites of the policy — cleanly carries two of Doya 2002's four RL-hyperparameter channels: acetylcholine-as-learning-rate (α) at the encoder, noradrenaline-as-inverse-temperature (β) at the action head, with plasticity-gating at the GRU update gate as a re-named third, and 5-HT/discount honestly named out-of-substrate. The unification narrowed substantially from v3's four-channel claim.

The symposium's load-bearing finding is that v4 still does the one-to-one mistake one level down. Calling site A "the ACh-α site" is the same kind of single-function commitment that the v4 §3.0 lineage callout just criticised in Doya's framework — it just calls one effect "the channel" at finer granularity. Real neuromodulators produce baskets of effects via target-specific receptor density; the FiLM read-out matrices are the in-silico analogue of receptor density, so the substrate naturally produces baskets of effects too — but v4's prose only mentions one effect per site.

The postdoc's four candidate framings (full text in [§4 of the synthesis](../../project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md)):

- **Framing 1.** Target the noradrenaline / locus-coeruleus system explicitly. Say the substrate computationally instantiates the principle "one ascending signal produces multiple effects via target-specific read-outs", with LC-NA as the named exemplar. ACh, DA, 5-HT named as out-of-paper-scope future work.
- **Framing 2.** Lead with the architecture. Position Row EE-6 (FiLM-Ensemble plus a heteroscedastic-precision head — see [the concept memo §5.4](../../project/concepts/film_neuromod_integration.md)) as "the first approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy". LC-NA in motivation only.
- **Framing 3.** Framing 1 plus a soft pain-construct headline conditional on a probe-trial extension landing before submission. The probe-trial adds avoidance behaviour in the absence of injury risk; if it shows dissociation, the paper claims hypervigilance-shape; if not, the paper retreats to Framing 1.
- **Framing 4.** Pure architectural lineage paper (FiLM ⊂ Hypernet ⊂ Bayesian-Hypernet-limit). Neuromodulation in motivation only; pain entirely deferred to follow-up.

## §3 Candidate paths

I lay out the four postdoc framings as paths, plus one PI addition (Framing 1-narrow) flagged clearly as my own. For each: headline claim in plain English, impactful-vs-reasonable position, venue family, compute / planning cost, submission timeline, and risk profile.

### Path A — Framing 1: NA-targeted multi-effect demonstration

- **Headline (plain English).** A small modulator network, trained end-to-end on an interoceptive RL agent, suggests the *possibility* that a single noradrenaline-like ascending signal can produce multiple behavioural-knob-like effects (faster recovery on returning to a known stage; sign-consistent temperature swings at stage boundaries; dissociable per-site clamps), via target-specific read-out matrices that are the in-silico analogue of receptor density. We do not claim to recover Doya's four channels; we claim one demonstration of the multifunctional-gain principle, with LC-NA as the named biological exemplar.
- **Impactful-vs-reasonable.** Reasonable end of the spectrum. Strong on the biological-substrate side (LC-NA is the canonical regime-change burst system; the R2 anchor's sign-consistent swings at stage boundaries fit the Aston-Jones & Cohen 2005 phasic pattern); softer on the architectural-novelty side (Row EE-6 stays as future work).
- **Venue family.** *Neural Computation* / *PLOS Computational Biology* / *eLife computational neuroscience* / CCN / COSYNE. NeurIPS bio-inspired track is plausible but not the natural home.
- **Compute / planning cost.** Zero new compute. The v4 §5 experimental program (per-site clamps, 3×3 dissociation matrix, T/P split as Phase 0 prerequisite, three-seed replication of R2) is unchanged. The only deliverable beyond v5 of the direction memo is the experimental program itself, which the user has not yet authorised to launch.
- **Submission timeline.** Earliest of the four paths. Pre-registered program plus three-seed replication plus T/P split implementation. Realistic 2–4 months wall-clock if launches start within ~2 weeks of v5 landing.
- **Risk profile.** Lowest. Backward-compatible with everything the project has built. The professor convergence is unanimous that this framing is defensible.

### Path B — Framing 2: Approximate Bayesian Hypernetwork (architectural lead)

- **Headline (plain English).** We introduce the first approximate Bayesian Hypernetwork (BHN — a network that produces a *distribution* over another network's weights as a function of input context) for an RL policy, with heteroscedastic likelihood (the BHN also predicts its own uncertainty). The substrate is biologically motivated by noradrenergic gain modulation. We demonstrate that posterior variance over the modulator's side-signal produces dissociable behavioural read-outs at three injection sites, with the substrate's behaviour consistent with the noradrenergic-gain-modulation literature.
- **Impactful-vs-reasonable.** Middle / architectural-impactful. The strongest pure-architecture claim the project can defend. Galanti & Wolf 2020 Theorem 4 (the parameter-efficiency / one-conditioner-many-readouts theorem for hypernetworks) is the formal identifiability anchor.
- **Venue family.** NeurIPS / ICLR / *TMLR* / AABI / NeurIPS BDL workshop.
- **Compute / planning cost.** Row EE-6 has to be implemented (non-trivial — a new ensemble head, a heteroscedastic precision head, the ρ-initialisation choice, the variance-space disambiguation; see [investigation memo §3](../../project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md)) on top of v4's §5 experimental program. Order of magnitude: 4–6 weeks of senior-developer + developer time for the implementation, plus the experiments themselves.
- **Submission timeline.** 3–5 months wall-clock. Behind Path A by ~6 weeks because of Row EE-6 implementation.
- **Risk profile.** Architectural-implementation risk lives on Row EE-6. The neuromodulation professor flagged that this framing risks reading as "post-hoc rationalisation of a choice motivated by target-specific gain on a multifunctional modulator" — the paper would have to answer that in the discussion. The project's downstream pain identity sits in the motivation only, weakening the bridge to follow-up clinical work.

### Path C — Framing 3: Framing 1 + conditional pain-construct upgrade

- **Headline (plain English).** Framing 1's headline, plus a soft second claim: the post-injury behavioural signature of the modulated agent is *consistent with* the kinds of multi-effect modulation the predictive-coding-of-pain framework predicts for hypervigilance (Wiech 2016; Tabor & Burr 2019). Specifically, the modulator's persistent state can produce avoidance behaviour in the absence of injury risk — a dissociation the predictive-coding-of-pain literature identifies as the signature of a prior-shift separable from the input.
- **Impactful-vs-reasonable.** Reasonable on the substrate claim; soft-impactful on the pain claim, conditional on the probe-trial extension landing.
- **Venue family.** If the probe-trial lands: *PAIN* / *Nature Mental Health* / *Computational Psychiatry*. If not: the Path A venue list.
- **Compute / planning cost.** Path A's full cost, plus: the post-recovery predator-cue probe-trial extension (v4 §5.8 — a "small senior-developer add" per the pain-modeling memo, but it requires a new task variant, a new logged quantity, and the C1 EMA-of-nociception baseline as a comparator); five-seed minimum for the within-trial cross-domain coupling threshold; the C1 baseline run; and the four-dissociation gate's re-stated form (in [pain-modeling memo §C](../../project/symposium/20260516_impact_vs_reasonable/professor_pain_modeling_contribution.md)) pre-registered. Order of magnitude: Path A's wall-clock plus 1–3 months. Realistically the project crosses the 4–6 month line for submission.
- **Submission timeline.** 4–7 months wall-clock. The probe-trial scoping itself needs user authorisation as a separate planning step (the user has explicitly said "no experiments before I say okay"), then `senior-developer` plans (no code), then user authorises the implementation, then implementation lands, then data collection. That chain is 5–8 weeks before the probe even produces data; then five seeds; then re-analysis.
- **Risk profile.** Double-conditional. Path A is the floor (the paper publishes regardless). Path C upgrades to a clinical-translational venue *if* the probe-trial lands cleanly *and* the four-dissociation gate (channel-selective gain on threat-relevant inputs; time-locked to injury inference; joint perceptual-mnemonic-policy coupling; dissociation from a slow-input baseline) passes on five seeds. The pain-modeling memo's §F is explicit that mere "narrative similarity" to clinical pain phenomenology does not clear the bar — the gate is the bar.

### Path D — Framing 4: Pure hypernet-lineage paper

- **Headline (plain English).** Architectural-only paper: the FiLM-modulator-on-RL-policy family is a sub-family of Hypernetworks (rigorous), which are themselves a point-mass limit of Bayesian Hypernetworks (rigorous), which are orthogonal to standard Bayesian Neural Networks (rigorous). The five formal claims from [the lineage investigation memo](../../project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md) become the contribution. Neuromodulation and pain are motivation paragraphs only.
- **Impactful-vs-reasonable.** Impactful pole on the architectural axis; effectively abandons the cellular-and-behavioural bridge.
- **Venue family.** NeurIPS / ICLR / AABI / NeurIPS BDL workshop only.
- **Compute / planning cost.** Same as Path B; the architecture has to be implemented.
- **Submission timeline.** Similar to Path B.
- **Risk profile.** The cost is *strategic*, not technical. The pain-modeling memo's §B.4 is explicit: publishing the substrate result with no pain framing attached makes the *next* paper's pain claim harder to land, because reviewers will have read Path-D-the-paper as "an RL architecture paper" and expect any follow-up to also be RL with pain bolted on. This is the silent two-paper-strategy trap.

### Path E — PI addition: Framing 1-narrow (single-claim version)

I flag this as my own addition; the postdoc did not surface it. Framing 1 as written includes a non-trivial pre-registered experimental program (per-site clamps, 3×3 dissociation matrix, T/P split as Phase 0 prerequisite, three-seed replication). A *narrower* Framing 1 variant would publish on the R2 anchor plus T/P split plus three-seed replication only — without the 3×3 dissociation matrix or the four-arm clamp at site C. The headline retreats from "we demonstrate dissociable per-site effects of an NA-LC-analog signal" to "we demonstrate that a single NA-LC-analog signal can drive recovery-speed advantages and sign-consistent phasic burst at regime change in an end-to-end-trained interoceptive RL agent".

- **Impactful-vs-reasonable.** More reasonable than Path A. The dissociation evidence is weaker, but the empirical anchor is fully replicated rather than newly collected.
- **Venue family.** Same as Path A, with a workshop fallback (NeurIPS bio-inspired-RL workshop, *Neural Computation* short-paper track) more accessible.
- **Compute / planning cost.** Lowest of the five paths. Three-seed R2 replication plus the T/P split (still required for identifiability of the multi-effect claim per v4 §3) plus v5 rewrite. No new experimental program beyond what is already in v4.
- **Submission timeline.** 1.5–3 months. Earliest of all paths.
- **Risk profile.** The biological-substrate professor would likely flag this as under-claiming. The architectural professor would push back that the dissociation matrix is the substrate's strongest test. If the user is **time-constrained or pace-fatigued**, Path E is the option that ships earliest with the lowest risk.

### Path comparison

| Path | Headline lens | Target named? | Pain content | Code work | Wall-clock to submission | Venue family |
|---|---|---|---|---|---|---|
| A — Framing 1 | Neuromodulation (LC-NA) | yes | motivation only | none (v4 §5 program runs as-is) | 2–4 months | comp-bio / comp-neuro |
| B — Framing 2 | RL / Bayesian-DL (Row EE-6) | in motivation | discussion only | Row EE-6 implementation | 3–5 months | NeurIPS / ICLR / TMLR |
| C — Framing 3 | Path A + soft pain-construct | yes | conditional headline | Path A + probe-trial harness + C1 baseline | 4–7 months | comp-bio default; *PAIN* if probe lands |
| D — Framing 4 | Pure architecture | mention only | motivation paragraph | Row EE-6 implementation | 3–5 months | NeurIPS / ICLR only |
| E — PI: Framing 1-narrow | LC-NA, single-claim | yes | motivation only | none | 1.5–3 months | comp-bio / comp-neuro + workshop fallback |

## §4 My independent assessment

I work through the five sub-questions the brief asked me to weigh.

### 4.1 The pace risk

The project has cycled through four versions of the direction memo (v1 → v4), one concept memo, one lineage investigation, one active-inference concept memo, and now a four-professor symposium with full postdoc synthesis, in four days. The symposium is the third planning-redesign round in that window. **This is the dominant strategic risk on my read** — not a technical risk, a portfolio-pace risk. Each round absorbs a real chunk of compute (in the loose sense: a chunk of attention, planning capacity, and the user's own time). Each round has delivered a real improvement in framing precision. But the *marginal* improvement is shrinking: v3 → v4 absorbed the math-reviewer audit (substantive); v4 → v5 absorbs a rhetorical reframing (substantive but rhetorical-only); v5 → v6 risks being a polishing pass, not a structural one.

The empirical floor — the R2 anchor and the canonical-phasic-burst signature — has not changed since 2026-05-13. The project's binding constraint is *no longer evidence; it is the gap between framing and experiments*. v4's experimental program (per-site clamps, 3×3 dissociation matrix, T/P split as Phase 0 prerequisite, three-seed replication) was first surfaced in v3 and has not been touched by the symposium. None of it has run. **Each additional round of framing without an experiment to ground it is a round that delays the first new datum.** The right time to stop redesigning is when (a) the next redesign would not change the experimental program and (b) the writing is precise enough to survive a careful reviewer. We are approximately at (a) now; (b) is what v5 is for.

My recommendation on pace: **v5 of the direction memo absorbs the symposium and is the last redesign round before experimental execution.** If, after v5, the project still has a major framing issue, that issue is a v6 issue *after* the first batch of new experiments — not before. The user should explicitly call this in v5's frontmatter ("v5 is the last redesign round before execution; subsequent rewrites are conditional on new data").

### 4.2 The single-paper-vs-two-paper question

The user has been explicit about a single paper. The postdoc synthesis respects this. Path D (pure hypernet-lineage paper) implicitly proposes a two-paper strategy: the architectural paper publishes first, the pain-substrate paper follows. The pain-modeling memo's §B.4 lays out the cost of that trajectory — the second paper has to re-establish framing against reviewers who have already read the first as an RL paper. That cost is real and asymmetric (the first paper is fine; the *second* paper bears the cost). Two papers are also not closer to the project's downstream identity (computational pain modelling); they are *further* from it, because the first paper anchors the project in the literature as RL-architecture rather than as biology-and-pain.

My read: **the single-paper constraint binds correctly here**. The user's identity-preservation instinct is the right portfolio call. Paths A, B, C, and E are all single-paper trajectories; Path D is the silent two-paper trap.

### 4.3 The venue family question

Each framing partitions to a different venue family with different reviewer expectations, page budgets, and submission cadences:

- Path A (Framing 1) → biology-lead → *Neural Computation* (paper-shaped 30+ pages, slow review, comp-neuro reviewers) / *PLOS Computational Biology* / *eLife* / CCN-CCNB / COSYNE (abstract + poster, fast iteration).
- Path B (Framing 2) → architecture-lead → NeurIPS / ICLR (8–10 pages, fast review, ML reviewers) / *TMLR* (rolling review, ML reviewers).
- Path C (Framing 3) → interoceptive-pain-lead, conditional → *PAIN* / *Nature Mental Health* / *Computational Psychiatry* (8–12 pages, clinical reviewers) if probe-trial lands.
- Path D (Framing 4) → pure-architecture → NeurIPS / ICLR / NeurIPS BDL workshop only.
- Path E (Framing 1-narrow) → biology-lead lite → same as Path A, with NeurIPS bio-inspired-RL workshop or *Neural Computation* short-paper track as faster fallbacks.

The venue choice is downstream of the framing, but it has its own implications: NeurIPS / ICLR cycles run on specific deadlines; *Neural Computation* and biology venues are rolling but slow; clinical venues (Path C) have the longest review cycles and the most demanding reviewer expectations on construct-validity. **I do not have a recommendation on venue — that depends on the user's submission-timeline preference, which is exactly the trade-off the path choice surfaces.** A user who wants a fast first paper picks Path E or Path A targeting a workshop; a user who wants the highest-impact first paper picks Path B or C and accepts the longer timeline.

### 4.4 The conditional-upgrade viability of Path C

Path C is Framing 1 plus a *conditional* pain-construct upgrade. The upgrade requires:

1. User authorising the probe-trial extension as a planning item.
2. `senior-developer` planning the extension (no code).
3. User authorising implementation.
4. Implementation landing (medium senior-developer + developer effort — new task variant, new logged quantity, C1 baseline harness).
5. Probe-trial data collected on at least five seeds.
6. Four-dissociation gate evaluated; conditional headline promoted or retreated.

Each of (1)–(6) is a real step. Steps (1)–(4) take wall-clock; step (5) requires GPU-weeks. **The honest read is that Path C is a two-stage paper compressed into one submission cycle.** If the probe-trial fails or the gate doesn't pass, the paper retreats to Path A's headline mid-write and the team has spent 1–3 months of compute on a section that becomes "future work". If it succeeds, the paper goes to *PAIN* / *Nature Mental Health* and the project's downstream identity is captured in the first submission. The bet is on the probe-trial's empirical success.

The pain-modeling professor's view, as I read it, is that the dissociation evidence is *consistent with* the gate signatures but the gate's four dissociations have not been tested at the resolution needed. R2 alone can say nothing about three of the four. So Path C is a *project-extension* path, not a *current-data* path.

My read: **Path C is the right paper for the project's downstream identity if the user is willing to spend 1–3 additional months on it.** It is *not* the right paper if the user is time-constrained or pace-fatigued. The trade-off is honest; the user owns it.

### 4.5 The "impactful" question

The user named "impactful for integrating neural-gain-like algorithms with hyperparameter modulation" as the first criterion. The two competing readings:

- **Impactful within the project's downstream pain identity.** Path C is the highest score; Path A second; Paths B, D, E lower. Pain reviewers care about construct-validity dissociation; architectural reviewers don't.
- **Impactful within the broader AI / computational-neuroscience field.** Path B is highest (BHN-with-heteroscedastic-likelihood for an RL policy is a real novelty); Path D second (the lineage paper); Paths A, C, E lower. NeurIPS / ICLR reviewers care about architectural novelty; comp-bio reviewers don't.

These two readings compete. The user has not stated which one matters more. My instinct, from the project's stated downstream identity in pain modelling, is the *first* reading (impactful-within-pain), which favours Paths A and C. But the user has the right to choose the second reading, which favours Paths B and D. **This is one of the two key trade-offs the user-facing question should surface explicitly.**

## §5 My recommendation

**Path A (Framing 1) as the baseline. Not Path C (the pain-construct upgrade) at this round.**

The reasoning, ranked:

1. **Path A captures the symposium's load-bearing convergence in its strongest form** — all four professors agreed that the substrate is right and the rhetoric needs softening; Path A implements exactly that softening with LC-NA as the named target. The postdoc's top recommendation is Path A + Path C, and I agree on Path A but step back from Path C below.

2. **Path A is the unique zero-additional-cost option that preserves the project's pain identity.** Path E is cheaper but under-claims; Path B trades pain identity for architectural impact; Path D abandons pain identity entirely; Path C is Path A plus a substantial scope-add that the user has explicitly said is still in planning. Path A is the unique point where biological identity is preserved at zero scope-add.

3. **Path C's bet is on data the project does not yet have.** The pain-modeling memo is unambiguous that R2 alone says *nothing* about three of the four dissociations. Path C's upgrade depends on the probe-trial landing cleanly *and* the four-dissociation gate passing on five seeds. That's two contingent gates, each non-trivial. If either fails, the paper retreats to Path A's headline mid-write — i.e., Path A *is* the floor even if the user chooses Path C. **Picking Path C does not avoid Path A; it adds 1–3 months and two contingent gates on top.**

4. **The pace risk argues against scope-add right now.** The project has not made the strategic mistake of redesigning past the point of diminishing returns *yet*, but it is in the pace regime where each additional planning round delays the first new datum. Path C's scope-add starts another planning round (probe-trial design, C1 baseline scoping, gate pre-registration). Path A's scope is locked at v4 §5; v5 of the direction memo is the last rhetorical round before launch. **The right move is to stop redesigning and start running the v4 §5 experiments, with v5 as the rhetorical anchor.**

5. **Path C's pain identity can be captured by a faster follow-up paper, not by stretching the first.** If Path A's first paper has LC-NA as the explicit biological target and the four-dissociation gate pre-registered as future work (pain-modeling memo's §E.1 — "publish the gate as a methodological contribution in its own right"), then the *next* paper has the gate already pre-registered, the probe-trial as the test, and the project's pain identity intact. This is the *fast two-paper trajectory*: Path A now (3 months), Path A→C as paper 2 (6 months after paper 1). The user has not authorised this two-paper trajectory; my recommendation is to surface it as the natural alternative to Path C's one-paper-but-longer trajectory.

6. **If the user is more architectural-impact-leaning than I read, Path B is the second-best option.** Path B trades pain identity for architectural novelty (Row EE-6 as the BHN-with-heteroscedastic-likelihood headline). The implementation cost is real (4–6 weeks) but bounded; the venue family is closer to the project's velocity-friendly NeurIPS / ICLR cadence; and the lineage investigation has already done the hard formal work. If the user reads the "impactful" criterion as field-level rather than identity-level, Path B is the right answer.

7. **Path E (Framing 1-narrow) is the right answer if pace-fatigue dominates.** If the user is feeling the pace strain from four direction-memo versions in four days and wants the *fastest* publishable paper, Path E ships in 1.5–3 months on already-anchored evidence plus three-seed replication plus T/P split. The headline retreats from "we demonstrate dissociable effects" to "we demonstrate a single LC-NA-analog signal can drive recovery-speed advantage and phasic burst at regime change", which is a reasonable and defensible claim on the project's current data. The cost is under-claiming relative to what v4 §5 could deliver.

**My ranking, with caveats:** Path A > Path E > Path B > Path C > Path D, *if* the user reads "impactful" as identity-level. If the user reads "impactful" as field-level: Path B > Path A > Path D > Path E > Path C.

## §6 The user-facing question

Top-level Claude should surface this via `AskUserQuestion`. I propose four options (Paths A, B, C, E) for the prompt — Path D is dominated by Path B on its own terms (both are architectural-lead but Path D abandons pain identity entirely, which the user's stated downstream identity argues against), so I do not surface it. Optional fifth: a "pick-PI-default-and-let-postdoc-write-v5" option for users who prefer not to choose between the framings.

**Question prompt (one sentence summary):** *Which framing should the first NMN paper take? Each is a complete single-paper trajectory; the choice fixes the venue family and the experimental scope.*

**Options:**

- **(Recommended) Path A — NA-targeted multi-effect demonstration.** Frame the modulator as a computational instance of the noradrenaline / locus-coeruleus "one signal, many effects" principle, with pain as motivation only and the four-dissociation gate published as a pre-registered protocol for the follow-up paper. Trade-off: lowest cost (zero scope-add beyond v4); highest preservation of pain identity at zero risk; venue family is *Neural Computation* / *PLOS Comp Bio* / *eLife*; submission target 2–4 months.
- **Path B — Architectural-lead (Approximate Bayesian Hypernetwork).** Lead with Row EE-6 as the first BHN-with-heteroscedastic-likelihood for an RL policy; LC-NA in motivation; pain in discussion only. Trade-off: highest pure-architecture impact; requires 4–6 weeks of Row EE-6 implementation; venue NeurIPS / ICLR / TMLR; submission target 3–5 months; pain identity weakly preserved.
- **Path C — NA-targeted plus conditional pain-construct upgrade.** Path A's body plus a probe-trial extension that lifts the paper to a pain-construct headline if it lands. Trade-off: highest reach into the project's downstream identity; conditional on probe-trial success and four-dissociation gate passing; adds 1–3 months wall-clock; venue *PAIN* / *Nature Mental Health* if it lands, else falls back to Path A's venues.
- **Path E — Framing 1-narrow (single-claim version, PI addition).** Path A's substrate-level claim restricted to "we demonstrate that an LC-NA-analog signal can drive recovery-speed advantage and phasic burst at regime change", dropping the 3×3 dissociation matrix and the four-arm clamp. Trade-off: fastest submission (1.5–3 months) on already-anchored evidence plus three-seed replication; under-claims relative to what v4 §5 could deliver; useful if pace-fatigue dominates.

## §7 What happens next given the user's choice

For each path, who picks up next. All paths share two upstream steps: (i) `research-postdoc` writes v5 of the direction memo absorbing the chosen framing (rhetorical-only edits per [postdoc synthesis §6](../../project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md)); (ii) v5 is reviewed by the four symposium professors. All paths share one downstream commitment: **no experimental-design / training-runner work until the user explicitly authorises**, per the user's "no experiments before I say okay" pace constraint.

| Path | Immediate next agent | Concrete next step |
|---|---|---|
| A | `research-postdoc` | Write v5 of the direction memo with Framing 1 (NA-LC as named target; per-site language softened to target-flavour basket-of-effects; pre-registered four-dissociation gate as methodological contribution). User then decides whether to authorise the v4 §5 experimental program. |
| B | `research-postdoc` | Write v5 of the direction memo with Framing 2 (Row EE-6 as architectural headline; LC-NA in motivation). User then decides whether to authorise (i) Row EE-6 implementation planning by `senior-developer` (no code yet) and (ii) the v4 §5 experimental program. |
| C | `research-postdoc` + (later) `senior-developer` | Write v5 of the direction memo with Framing 3 (Path A body + probe-trial conditional upgrade pre-registered). User then decides whether to authorise (i) probe-trial scoping by `senior-developer` (planning only, no code) — *separately from* implementation authorisation; (ii) the v4 §5 experimental program; (iii) the C1 EMA-of-nociception baseline. |
| E | `research-postdoc` | Write v5 of the direction memo with Framing 1-narrow (single-claim version, dropping the dissociation matrix and four-arm clamp from the body). User then decides whether to authorise (i) three-seed R2 replication and (ii) T/P split implementation as the minimum experimental program. |

In all cases the chain is: **v5 → professor review → user authorises experimental program (separate decision) → `experiment-designer` design docs → user authorises launch (separate decision) → `training-runner` launches.** The user owns each of the three authorisations independently.

## §8 Pace flag for the user

I am surfacing this as a flag, not a blocking concern.

**The project has cycled through four versions of the direction memo, one concept memo, one lineage investigation, one active-inference concept memo, and a four-professor symposium with full postdoc synthesis — in four days.** Each round has produced a real improvement in framing precision: v3 → v4 absorbed a substantive math-reviewer audit; v4 → v5 absorbs a substantive rhetorical reframing. But the *marginal* improvement is shrinking, and the empirical floor (the R2 anchor and the canonical phasic-burst signature) has not changed since 2026-05-13. The project's binding constraint is no longer evidence; it is the gap between framing and experiments. Each additional round of framing without an experiment to ground it delays the first new datum.

My recommendation, restated here for visibility: **v5 of the direction memo is the last rhetorical round before experimental execution.** If the user picks Path A, B, C, or E, the next decision after v5 should be whether to authorise the experimental program — not whether to redesign the framing again. If a v6 becomes necessary, it should be triggered by new data, not by new theoretical reading.

The user's instinct to call a symposium *before* committing to a paper shape is correct; the cost of redesigning after launching is much higher than the cost of one more planning round now. I am not asking the user to stop being careful. I am asking the user to recognise that we are approximately at the point of diminishing returns on planning-without-data, and to use the symposium's output as the *resolution* of the pace cycle, not as the input to another one.

---

## Cross-references

- Postdoc symposium synthesis: [`docs/project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md`](../../project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md)
- Professor contributions: [neuromodulation](../../project/symposium/20260516_impact_vs_reasonable/professor_neuromodulation_contribution.md); [RL / Bayesian-DL](../../project/symposium/20260516_impact_vs_reasonable/professor_rl_bayesian_dl_contribution.md); [Bayesian-brain](../../project/symposium/20260516_impact_vs_reasonable/professor_bayesian_brain_contribution.md); [pain-modeling](../../project/symposium/20260516_impact_vs_reasonable/professor_pain_modeling_contribution.md)
- v4 of the direction memo (the artefact the symposium is reviewing): [`docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md`](../../project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v4.md)
- Empirical anchor (R2 anchor): [memory insight `20260513_0014`](../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md)
- Reader-facing study summary: [`docs/experiments/summaries/20260513_0321_nmn_comparison_study.md`](../../experiments/summaries/20260513_0321_nmn_comparison_study.md)
- Concept memos: [FiLM ↔ neuromodulation integration](../../project/concepts/film_neuromod_integration.md); [FiLM-Hypernet-BHN-BNN lineage investigation](../../project/concepts/film_hypernet_bnn_lineage_and_critic_modulation.md); [active-inference hypervigilance](../../project/concepts/active_inference_hypervigilance.md)
- Portfolio: [`docs/pi/PORTFOLIO.md`](../PORTFOLIO.md) — Track A / Track B not yet ratified; this call is the natural precursor to a first-track ratification call.
- Prior PI calls for tone and pace continuity: [2026-05-12 Dreamer backend](2026-05-12_dreamer_backend.md); [2026-05-14 D-013 parity-launch disposition](2026-05-14_d013_parity_launch_disposition.md)

## Hand-offs

- **User decision next** — surfaced via `AskUserQuestion` by top-level Claude (PI does not surface directly in background-agent mode).
- **`research-postdoc`** — writes v5 of the direction memo with the chosen framing, after the user picks. The rewrite is rhetorical-only per [postdoc synthesis §6](../../project/symposium/20260516_impact_vs_reasonable/postdoc_synthesis.md); no code, no experiments, no equation changes.
- **Four symposium professors** — invited to review v5 once it is written. Their convergence is the load-bearing finding for the rewrite; their pass on v5 is the rhetorical closure of the symposium.
- **`senior-developer`** — *not invoked yet*. Path B and Path C each require senior-developer planning steps (Row EE-6 for B; probe-trial scoping for C), but those are post-v5 and gated by separate user authorisation.
- **`experiment-designer` / `training-runner`** — *not invoked yet*. The v4 §5 experimental program is unchanged under any framing, but its launch is gated by separate user authorisation. The user has explicitly said no experiments before that authorisation.

---

*Call logged by `pi`, 2026-05-16. Awaiting user decision via `AskUserQuestion`.*
