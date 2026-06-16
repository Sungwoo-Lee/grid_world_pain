# Behavioural assays for threat discrimination: translating the fear/pain-learning paradigms into held-out gridworld testbeds

> **One-line summary.** The animal-learning literature already solved "does the animal treat a danger cue differently from a safe cue" with a dozen well-validated behavioural assays. This memo translates six of them into held-out gridworld testbeds (harmful predator vs. matched harmless rabbit), and for each names the construct it measures, the read-out that counts as genuine discrimination, and — most important — the confound it guards against. The throughline is one distinction the project keeps tripping over: **anticipatory threat-recognition** (the agent infers danger *before* contact) vs. **post-contact pain-reaction** (the agent only responds *after* being hurt). The assays are ranked by how cleanly they separate those two.

## Purpose — what this memo is for and why it exists

Our June study reached a sharp, well-supported negative verdict: when a harmful "predator" (its touch injures the agent) and a harmless "rabbit" (its touch does nothing) are matched on everything the agent can sense from a distance — same smell, same chasing motion, same approach speed, same strike-and-retreat rhythm — the trained agent treats them **alike before contact**. Its avoidance is driven by *the consequence of being touched*, not by *recognising the danger in advance*. ([study summary, 2026-06-12](../../experiments/summaries/20260612_1625_predator_rabbit_discrimination.md).)

But that verdict was read off **naturalistic episodes** — open-grid foraging where motion, aggression-speed, animal count, out-of-distribution states, and post-contact reaction all bled into the measurement and (per the study's own post-mortem) repeatedly produced confounded or hidden signals. The user's new direction is the right fix and it is exactly the move the project's own platform paper already endorses: the EVAAA design (Lee et al. 2025) pairs a **naturalistic training curriculum** with a **suite of controlled testbeds** whose stated purpose is "to isolate specific decision-making processes ... by minimizing potentially irrelevant extrinsic cues." We need that second tier for threat-discrimination — held-out environments where **the environment itself is the measurement instrument**, designed so that a positive read can *only* mean the agent discriminated danger.

The animal-learning literature is the right source for that design, because the question "does this animal behave differently toward a danger cue than a matched safe cue, and is the difference anticipatory or reactive" has a century of validated paradigms behind it — conditioned place avoidance, approach–avoidance conflict, conditioned suppression, discrimination training, the predatory-imminence continuum, and risk-assessment ethograms. Each paradigm was *designed* to rule out a specific alternative explanation (general caution, neophobia, a nociceptive reflex). That is precisely the construct-validity work this project needs. This memo is the bridge from those paradigms to our gridworld.

**What this memo is and is not.** This is a **concept / design-principle memo** — it specifies *which paradigms* to instantiate and *what makes each a valid discrimination test*. It does not write configs (that is `experiment-designer`) or touch `src/` (the predator/rabbit FSM, place-conditioned damage, and CS channels are env features `senior-developer` would scope). It is the upstream companion to the [behavioural-detectability memo](behavioural_discrimination_detectability.md): that memo says *how to detect* a discrimination signal in a behavioural read-out (d′, AUC, mutual information, ideal-observer ceiling); **this memo says how to build an environment in which a detected signal is interpretable as threat-recognition rather than something else.** You need both — a perfect detector on a confounded environment still yields a confounded conclusion.

A standing terminology guard for the whole memo: in animal learning a **CS** ("conditioned stimulus") is a *cue that predicts* an outcome, and the **US** ("unconditioned stimulus") is the outcome itself — here, the injury at contact. The project's recurring error is to let the agent only ever experience the US (the bite) and then call the resulting withdrawal "threat recognition." Recognition is a claim about behaviour controlled by the **CS in advance of the US**. Every assay below is engineered to make that distinction measurable.

---

## §1 The core distinction the assays must respect: defensive distance / predatory imminence

Before the individual assays, one organising framework from the predator-defence literature subsumes the project's central question and should structure the whole testbed suite.

**Fanselow & Lester's predatory-imminence continuum** (and Blanchard's defensive-distance account) says defensive behaviour is not one thing — it is a *sequence of modes* selected by **psychological distance to the threat**, and the modes are qualitatively different:

| Imminence mode | Trigger (defensive distance) | Behaviour | What it is in our terms |
|---|---|---|---|
| **Pre-encounter** | Threat possible but not present | Altered foraging schedule, meal pattern shift, thigmotaxis, avoiding risky patches | *Anticipatory* — danger inferred from context, no threat in view |
| **Post-encounter** | Threat detected, not yet striking | Freezing, flight, risk assessment (stretch-attend) | *Anticipatory* — danger recognised at a distance |
| **Circa-strike** | Threat attacking / contact imminent | Explosive flight, defensive attack, withdrawal | *Reactive* — overlaps the moment of contact |

**Why this is the master frame.** The project's "anticipatory vs. post-contact" axis *is* the imminence continuum: pre-encounter and post-encounter modes are anticipatory threat-recognition; circa-strike is the reflexive pain-reaction the June agent showed. The study found the agent operating **only in circa-strike** — it "forages and rests in place while the predator beelines into it, hiding only after being hit." A valid discrimination assay must create a regime in which a *pre-encounter or post-encounter* mode can be expressed and measured **separately from** the circa-strike reflex. Assays that put the agent into contact (or near-contact) cannot do this — they read circa-strike and over-claim it as recognition. Assays that force a *spatial or temporal gap* between the cue and the contact can.

**This gives the single sharpest design principle in the memo:** *put a measurable gap — in space or in time — between "the danger cue is present" and "contact could occur," and read the agent's behaviour in that gap.* Behaviour in the gap is anticipatory by construction; behaviour at contact is not. Every ranking below is essentially "how clean is the gap this assay creates."

---

## §2 The assays

For each: the paradigm and its source, the plain-English construct, the gridworld instantiation, the read-out that counts as *genuine* discrimination, the construct-confound it guards against, and an explicit **anticipatory vs. reactive** tag.

### Assay A — Conditioned Place Avoidance / Preference (CPA/CPP)

**Source.** The conditioned place preference/aversion paradigm (Tzschentke's reviews; the workhorse assay of motivational learning). An animal is exposed to a distinctive *place* paired with an outcome; later, with the outcome absent, its dwell-time across places reveals what it learned the place predicts.

**Construct.** *Has the agent learned that a place predicts injury, such that it avoids the place even when no threat is currently present?* This isolates **context-conditioned anticipation** — the pre-encounter mode — from any in-the-moment threat stimulus.

**Gridworld instantiation.** A held-out two-zone arena. In a *conditioning* phase the predator (damage on contact) is confined to Zone P and the rabbit (harmless) to Zone R; the two zones are otherwise identical (same food density, same smell field, same visual texture). In the **test phase the animals are removed entirely** and the agent free-forages across both zones. The CPA score is dwell-time and entry-rate in Zone P vs. Zone R with both animals absent.

**Read-out = genuine discrimination.** Agent spends *reliably less* time in (and is slower to enter / quicker to leave) the predator-zone *with no animal present*. Because nothing is there to react to, a place difference can only come from a learned place→injury association — the cleanest possible operationalisation of "context-conditioned prediction of damage," which is exactly the project's definition of pain-like avoidance (project_plan §3.1, Avoidance row: "generalises ... to context-conditioned predictions of damage — not reflex withdrawal at the moment of contact").

**Confound it guards against.** **The nociceptive reflex / post-contact reaction.** With the animals absent at test, there is no contact to react to and no threat stimulus to flee — so a circa-strike reflex *cannot* produce the signal. It also guards against motion and aggression confounds (no moving animal at test). Residual confound to control: the zones must be matched on *everything else the place affords* (food, temperature, smell), or the agent could be avoiding a poor-food zone, not a dangerous one — so the conditioning must pair injury with an *arbitrary* place feature, not a resource-correlated one.

**Anticipatory vs. reactive.** **Purely anticipatory.** This is the gold-standard separation: the read-out is *defined* on states with no threat present, so it cannot be contaminated by reaction. Strongly licenses an anticipatory-recognition claim if positive; a clean, interpretable null if not.

### Assay B — Approach–Avoidance Conflict (reward guarded by threat)

**Source.** The Geller–Seifter / Vogel conflict tests and the modern elevated-plus-maze / open-field conflict literature; the canonical assay of *how much an animal will pay to avoid danger*. Anxiolytics are *defined* by their effect here.

**Construct.** *How much reward will the agent forgo to avoid the predator — and does it forgo more for the predator than for the matched rabbit?* This measures the **subjective cost the agent assigns to the threat**, graded.

**Gridworld instantiation.** A single high-value food cache placed adjacent to, or reachable only by passing, an animal. Two matched conditions, run as held-out episodes: food guarded by the **predator** vs. food guarded by the **matched rabbit**. Grade the conflict by varying the food value (hunger level at episode start) or the geometry (how close the path passes the animal). The read-out is the **indifference point**: the food value / proximity at which the agent switches from "approach and forage" to "stay away."

**Read-out = genuine discrimination.** The indifference curve is **shifted** for the predator vs. the rabbit — the agent demands a higher food value, or a wider berth, before it will approach the predator-guarded cache. A graded shift (not just a binary avoid/approach) is the strong signal: it shows the agent has assigned a *quantitatively larger threat-cost* to the predator. This directly instruments the project's "managing conflict needs" category (project_plan §3.1) — the trade-off of avoidance against hunger — and turns it into a psychophysical curve.

**Confound it guards against.** **General caution / generic risk-aversion.** A merely jumpy agent avoids *both* caches equally; the predator-vs-rabbit *shift* in the indifference point is what isolates threat-class-specific cost from a blanket "don't go near animals" policy. It also guards against the floor/ceiling problem of binary avoidance metrics (the June study's avoid-rate that read identically for both). Caveat: the conflict must be staged so the agent *can* in principle reach the food without contact (a gap exists, per §1) — otherwise it collapses into a contact test and reads circa-strike.

**Anticipatory vs. reactive.** **Largely anticipatory, if staged with a spatial gap.** The decision to approach-or-not is made *before* the agent is in contact range, so the indifference point is an anticipatory read. The risk is leakage: if the only way to learn the cost is by getting bitten during the episode, late-episode behaviour becomes reactive. Mitigate by reading the **first-approach decision** before any contact in the episode.

### Assay C — Conditioned Suppression of foraging under a danger cue (Estes–Skinner)

**Source.** The Estes–Skinner conditioned-suppression / conditioned-emotional-response (CER) paradigm — the single most-validated assay in the fear-conditioning literature. An animal lever-presses for food; a CS that predicts shock *suppresses* the ongoing appetitive behaviour. The **suppression ratio** is the classic read-out of learned threat.

**Construct.** *Does a danger-predicting cue suppress the agent's ongoing foraging — and does the predator-cue suppress it while the matched rabbit-cue does not?* This measures **anticipatory threat-driven interruption of appetitive behaviour**, the behavioural core of hypervigilance's "interruption" function (Eccleston & Crombez 1999).

**Gridworld instantiation.** The agent is steady-state foraging (eating respawning food). A discrete **cue** is then presented — a distal smell plume, a visual marker, or a light/phase change — that during conditioning reliably preceded a predator appearance by a fixed delay; a *different* cue preceded the harmless rabbit. At test, present each cue **with no animal yet present** and measure foraging in the cue window. The EVAAA "Predators with day/night" testbed is a primitive version of this (suppress foraging by day when predators hunt) — this assay sharpens it into a matched predator-cue vs. rabbit-cue contrast.

**Read-out = genuine discrimination.** A **suppression ratio** $SR = \frac{r_{\text{cue}}}{r_{\text{cue}} + r_{\text{pre-cue}}}$ on eat-rate (the project's M5/M1 family formalised): $SR < 0.5$ under the predator-cue (foraging suppressed) but $SR \approx 0.5$ under the rabbit-cue (no suppression). The *difference* in suppression between the two cues, measured *before any animal arrives*, is the discrimination signal.

**Confound it guards against.** **Post-contact reaction and motion/aggression confounds, simultaneously** — because the suppression is read in the **cue window before the animal is present**, there is no animal to react to, flee, or be chased by. It is the temporal analogue of Assay A's spatial gap. It also cleanly separates a *cue-specific* response (discrimination) from a *generalised arousal* response (suppress to any cue). The confound it does *not* itself rule out is latent-inhibition / novelty: the predator-cue must not simply be more salient than the rabbit-cue — handled by counterbalancing which cue maps to which animal across episodes.

**Anticipatory vs. reactive.** **Purely anticipatory.** The suppression-in-the-cue-window read-out is defined on pre-animal states. Together with Assay A (spatial gap) this is the second gold-standard separation — A isolates the anticipatory signal in *space*, C isolates it in *time*.

### Assay D — Discrimination training & generalisation gradient (CS+ / CS−, with safety learning)

**Source.** Pavlovian discrimination training (CS+ reinforced, CS− not) and the fear-generalisation-gradient literature (Lissek; Dunsmoor); plus safety-signal learning (Rogan; the conditioned-inhibitor literature). The paradigm that asks not just "does it discriminate" but "*how precisely*, and does it learn an explicit safe cue."

**Construct.** *Can the agent learn a discriminated threat — responding to a danger cue (CS+) and withholding response to a safe cue (CS−) — and how sharply tuned is that discrimination across a stimulus dimension?* This measures **precision of threat-tuning** and **safety learning**, the dissociation between threat and explicitly-safe.

**Gridworld instantiation.** Give the predator and rabbit a *graded* shared sensory dimension — e.g. a smell channel with a continuous intensity, where predator = high end (CS+), rabbit = low end (CS−). After training, probe with **intermediate, never-seen smell values** (generalisation-gradient test) and measure the defensive-response gradient across the dimension. A **safety variant**: add a *compound* cue where the predator-smell is sometimes accompanied by an explicit safe-marker that signals "this time it's harmless," and test whether the safe-marker suppresses defence (conditioned inhibition).

**Read-out = genuine discrimination.** (i) A **monotonic generalisation gradient** — defence highest at the CS+ end, lowest at the CS− end, graded in between — is positive evidence of a tuned threat-representation (flat = no discrimination; a peak shifted off the CS+ = overgeneralisation, itself an interesting *maladaptive* signature). (ii) The safe-marker reliably **reduces** defence to the predator-smell = genuine safety learning, not just threat learning. *This is the one assay that can directly support a "pain-like hypervigilance" claim with construct validity*, because **overgeneralisation of the gradient** (broad, flat tuning; defence to safe-leaning cues) is the operational signature of clinical pathological hypervigilance (Lissek's overgeneralisation account) — it distinguishes adaptive sharp discrimination from maladaptive broad threat-tuning.

**Confound it guards against.** **Conflating "responds to threat" with "fails to recognise safety."** Threat-only assays (A, B, C) can't tell a sharply-tuned discriminator from a broadly-fearful agent that happens to respond more at the threat end; the gradient + safety-marker design separates these. It is also the assay that guards against **over-claiming hypervigilance**: a project that wants to say "hypervigilant" must show *selective, persistent, threat-tuned* sensitivity, and the gradient's *shape* is what licenses (sharp = healthy discrimination) or licenses-differently (broad/overgeneralised = pathological hypervigilance) that claim.

**Anticipatory vs. reactive.** **Anticipatory** *if* the cue dimension is sensable at a distance (the gradient probe is read on the distal cue before contact). This assay's validity depends entirely on the CS being a *genuine distal predictor* — which is exactly what the June environment lacked (the matched smell meant there was no CS+ to tune to). Building a learnable distal CS+ is the prerequisite, and is itself the project's "give the agent a learnable route to recognition" follow-up.

### Assay E — Risk-assessment ethogram (pre-contact defensive read-outs)

**Source.** The ethological defence literature (Blanchard & Blanchard; the rodent risk-assessment repertoire — stretch-attend postures, vigilant scanning, thigmotaxis, cautious approach). These are the *pre-encounter and post-encounter* behaviours of the imminence continuum, and they are read-outs, not a separate environment.

**Construct.** *Before any contact, does the agent emit threat-assessment behaviours — cautious/interrupted approach, scanning, edge-hugging — selectively more toward the predator than the matched rabbit?* This measures **pre-contact vigilance** directly, as a behavioural ethogram rather than an outcome.

**Gridworld instantiation.** Not a new environment but a **read-out layer** applied across Assays A–D and the open-grid encounters: define gridworld analogues of risk-assessment acts — *interrupted/hesitant approach* (advance-then-pause oscillation toward the animal), *thigmotaxis* (hugging walls/bushes when the animal is distal), *vigilant orienting* (turning toward the animal without approaching), *stretch-attend analogue* (approach-to-threshold-then-retreat). Score their rate and timing in the **pre-contact window**, predator vs. rabbit.

**Read-out = genuine discrimination.** Risk-assessment acts are **selectively elevated toward the predator before first contact**. Crucially these are *graded, information-seeking* behaviours — their presence indicates the agent is *uncertain and assessing*, which is itself evidence of an anticipatory threat-evaluation process rather than a binary reflex.

**Confound it guards against.** **The binary avoid/approach trap that hid the June signal.** Mean-distance and flee-rate are coarse; risk-assessment ethograms catch the *micro-structure* of cautious behaviour that a binary metric averages away — directly addressing the study's "aggregate stats hid a conditional behaviour" post-mortem. It guards against under-detecting genuine pre-contact discrimination. What it does *not* guard against on its own: these read-outs still need a matched-rabbit contrast (else "cautious approach" could be generic), so E is a read-out layer *for* B/D, not a standalone claim.

**Anticipatory vs. reactive.** **Anticipatory by definition** (pre-contact window), but **weaker evidential status than A/C** because the behaviours are subtle and can be confounded with non-defensive exploration. Best used to *enrich* the gold-standard assays (A, C) with mechanism, not to carry the verdict alone.

### Assay F — Latent inhibition / pre-exposure (a confound-controller, not a primary claim)

**Source.** The latent-inhibition literature (Lubow) — pre-exposure to a cue *without* consequence retards later learning that the cue is dangerous. Included here as a **control assay** that hardens the others.

**Construct.** *Is the agent's (non-)discrimination due to genuine inability, or to having learned the cue is safe through unconsequenced exposure?* A diagnostic, not a pain-construct in itself.

**Gridworld instantiation.** Manipulate **pre-exposure**: one group's predator-cue is pre-exposed harmlessly before it becomes damaging; a control group's is not. Compare discrimination learning rate.

**Read-out / use.** Slower threat-learning in the pre-exposed group confirms the agent's learning machinery is *engaging the cue* (rules out "the cue is simply unusable"). Mainly this **guards against a false-negative**: if the June agent "can't discriminate," LI tells you whether that's a representational limit or an over-learned-safe artefact.

**Anticipatory vs. reactive.** N/A — a control on the *learning process*, run alongside D.

---

## §3 Construct-validity verdict — which assays license which claims

This is the guardian section. Mapping the assays onto what the project is allowed to claim:

| Claim the project might want to make | Which assay licenses it | Strength |
|---|---|---|
| "The agent recognises danger *before contact*" (anticipatory) | **A (place avoidance, animals absent)** and **C (suppression in cue window)** | **Strong** — read-outs defined on no-threat-present states; reflex cannot produce them |
| "The agent assigns a graded *cost* to threat, weighs it against hunger" | **B (approach–avoidance conflict, indifference shift)** | **Strong** for the cost claim; anticipatory only if a spatial gap is staged |
| "The agent has a *tuned, threat-specific* representation (not generic fear)" | **D (CS+/CS− gradient + safety marker)** | **Strong** for tuning; the *only* assay that separates threat-recognition from safety-failure |
| **"The agent is pain-like *hypervigilant*"** | **D specifically** — broad/overgeneralised gradient + persistence (run D's gradient at multiple post-injury delays, tying to H5) | **The only defensible route.** Hypervigilance = selective + persistent + threat-tuned. A, B, C show selectivity; D's gradient shape shows tuning; repeating across recovery shows persistence. **Claiming hypervigilance from A/B/C/E alone is over-claiming** — they show threat-sensitivity, not the tuned-and-persistent sensitivity the term requires. |
| "The agent's avoidance is a *reflex*, not recognition" (the current negative verdict) | Confirmed cleanly by a **null on A and C** | A null on the gold-standard anticipatory assays is the *strong* form of the project's negative result — far cleaner than the OOD predator-only world that carried only "medium confidence" |

**Three over-claiming traps to name explicitly:**

1. **Circa-strike ≠ recognition.** Any assay that reads behaviour at or near contact (the open-grid encounters, a conflict with no spatial gap) measures the reactive reflex. The June verdict is robust precisely because it caught the agent in circa-strike-only mode. New testbeds must create the §1 gap or they repeat the error.

2. **Threat-sensitivity ≠ hypervigilance.** "Avoids the predator more" (A, B) is necessary but not sufficient for hypervigilance. Hypervigilance is a *tuning + persistence* property — it needs D's gradient and a recovery-time axis (H5). The project's §3.1 hypervigilance row already says this ("channel-selective, context-modulated, outlasting the immediate input — not a uniform fear response") — D is the assay that operationalises every word of that definition.

3. **General caution ≠ class discrimination.** A jumpy agent passes any single-class avoidance test. *Every* assay here is built on a **matched-rabbit contrast** for this reason — the rabbit is the CS−/safe baseline without which "avoidance" is uninterpretable. This is the same point the [detectability memo §3](behavioural_discrimination_detectability.md) makes about needing a contrast; here it is enforced at the *environment* level, not just the metric level.

**Project mapping to the EVAAA two-tier frame.** These six assays are the threat-discrimination wing of the platform's "controlled testbed suite." A and C are new held-out environments; B generalises the existing EVAAA "risk-taking" / conflict tasks to a predator-guarded-reward form; the "Predators with day/night" testbed is already a coarse conditioned-suppression task (C) that this memo would sharpen into a matched-cue contrast; D requires a *new env feature* (a learnable distal CS+ channel) and is the natural home of the "give the agent a learnable route to recognition" research move.

---

## §4 Mapping to the project's hypotheses

| Hypothesis (NEUROMODULATION §1.4) | Assay that tests it | Note |
|---|---|---|
| H1 perceptual amplification (threat-gain after injury) | **E** (risk-assessment rate ↑ post-injury) + **D** (gradient sharpens/broadens post-injury) | H1's "attentional bias toward threat" *is* an anticipatory read — needs A/C/D, not contact data |
| H2 memory persistence (hold threat info longer) | **A** (place avoidance persists across a delay between conditioning and test) | A's test-phase delay directly probes "I remember the danger zone after leaving it" |
| H3 exploration suppression / risk-aversion | **B** (indifference-point shift = behavioural risk-aversion) + **C** (suppression ratio) | B turns H3 into a graded psychophysical curve |
| H4 coordinated multi-domain response | run **A+B+C+D on one agent**; coherence = consistent threat-class ranking across all four | Cross-assay consistency is the behavioural face of H4's "coherence" |
| **H5 chronic-pain analog (vigilance outlasts injury)** | **D's gradient + A's place avoidance, measured at increasing delays post-recovery** | **This is the assay design for the project's flagship pain claim.** Persistence of a *threat-tuned* gradient after the injury source is removed = maladaptive chronic-pain-like hypervigilance. A flat-then-decaying gradient = adaptive caution. The *time-course of the gradient after recovery* is the single most publishable read-out in the suite. |

---

## Next steps (hand-offs)

- **`experiment-designer`** — convert the ranked shortlist (§5) into concrete held-out testbed configs. Start with **A (conditioned place avoidance)** and **C (conditioned suppression)** — they are the cleanest anticipatory separations and reuse the existing predator/rabbit + damage machinery with only layout/cue changes (no new `src/` feature needed). Specify per-assay: the matched-rabbit contrast, the no-threat-present test window, the suppression-ratio / dwell-time / indifference-point read-out, and the pre-registered null. Pair every assay with the [detectability memo](behavioural_discrimination_detectability.md)'s AUC/MI/ideal-observer instrument so a detected signal is both *valid* (right environment) and *detectable* (right measure).
- **`senior-developer`** — scope two env features that gate the higher-value assays: (i) **place-conditioned damage** (predator/rabbit confined to zones during conditioning, removable at test) for Assay A; (ii) a **learnable distal CS+ channel** (a graded smell/visual cue that genuinely predicts the predator at a distance) for Assay D — this is the feature without which any "the agent learned to recognise danger" claim is impossible, and it is the concrete form of the "give the agent a learnable route to recognition" direction. Flagging as hand-offs; this memo writes no code.
- **`professor-bayesian-brain`** — Assay D's generalisation gradient and Assay C's suppression are precision-weighting phenomena in the predictive-coding frame; the overgeneralisation-as-hypervigilance reading (Lissek) connects to the precision/active-inference line in [[active_inference_hypervigilance]]. Pull in for the *why* if the project pursues the hypervigilance gradient.
- **`pi`** — the choice between (a) hardening the negative verdict with A+C (cheap, high-confidence, publishable construct-validity result) and (b) building the Assay-D distal-CS+ feature to chase a *positive* recognition result (more work, higher upside) is a portfolio-level focus-vs-explore call.

**References to add** (none currently in `docs/project/references/computational_models_of_pain/`): Fanselow & Lester (predatory imminence); Blanchard & Blanchard (defensive ethogram / risk assessment); Estes & Skinner 1941 (conditioned suppression); Tzschentke (CPP/CPA review); Lissek et al. (fear generalisation / overgeneralisation); Lubow (latent inhibition); Eccleston & Crombez 1999 (the interruption / hypervigilance account — likely already cited elsewhere in the project).

**Cross-links:** [[behavioural_discrimination_detectability]] (the measurement companion — how to *detect* a signal once the environment makes it *valid*) · [[pain_vs_nociception_construct]] (the foundational distinction these assays operationalise) · [[active_inference_hypervigilance]] (the precision framing of the gradient) · the predator-rabbit study summary (2026-06-12) · EVAAA testbed design (Lee et al. 2025, §6 — the two-tier frame this memo extends).
