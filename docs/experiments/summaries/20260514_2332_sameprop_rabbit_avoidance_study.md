---
title: "SameProp Rabbit-Avoidance Study — Reader-friendly re-summary with take-home messages (2026-05-07 → 2026-05-14)"
study: sameprop_rabbit_avoidance_study
generated: 2026-05-14T23:32
window: "2026-05-07 → 2026-05-14"
status: snapshot
---

# SameProp Rabbit-Avoidance Study — Reader-friendly re-summary as of 2026-05-14 23:32 KST

> **This is a re-summary, not a replacement.** The 2026-05-13 14:20 snapshot ([`20260513_1420_sameprop_rabbit_avoidance_study.md`](20260513_1420_sameprop_rabbit_avoidance_study.md)) was correct on the science but **read too densely** and **mis-reported Round 2.6 as in-flight**. This version is rewritten to be skimmable end-to-end and updates the Round 2.6 status: **both runs crashed at ~6.6 h on n106; no seed-lock yet.** Prior summaries are retained as historical snapshots; nothing is edited in place.

---

## Take-home messages — the whole study in 6 bullets

1. **The setup.** The agent normally tells predators from rabbits by **smell** (a per-class olfactory signature). This study deliberately makes the predator's smell and the rabbits' smell **identical** ("sameProp") and asks: does the agent still behave differently around them?

2. **Answer #1 — where the agent walks: NO.** When you measure "average distance the agent keeps from a predator vs from a rabbit," the difference **disappears or even flips the wrong way** under matched smells. So in pure spatial terms, the agent looks **class-blind**.

3. **Answer #2 — what the agent does when one gets close: YES, strongly.** When you instead watch what the agent *does* in the moments a creature enters its danger radius, the difference is huge: it dives into a bush **+37 percentage points more often** for a predator than a rabbit (88% vs 51%) and **stops eating** near a predator (0.75× normal) while **eating slightly more** near a rabbit (1.19× normal). Same agent, same checkpoint — opposite verdict, because we measured a different thing.

4. **The big methodological lesson.** **Averages hide the signal.** Mean-distance metrics average over many calm steps and dissolve the moments that actually matter. Whenever a study concludes "the agent doesn't discriminate," it should be cross-checked with event-rate measures (bush-dive, eat-suppression, behavioural motif) before that verdict is trusted. The project now has the toolkit for this.

5. **What "hypervigilance" should mean here.** The project's bigger goal — modelling pain-driven hypervigilance — is now better anchored. The event-level defensive repertoire (bush-diving, eat-suppression) is the layer where the agent already *does* discriminate; this is the natural place to look for "over-amplification under pain," not the spatial-trajectory layer (which is class-blind anyway under matched smells, so there's no signal to amplify).

6. **Where the work stands today.** The two-level finding is real but rests on **one seed per cell**. The seed-lock run (Round 2.6) was launched on 2026-05-12 — and **crashed** at ~6.6 h on the lab node, well short of the 22 h target. So the headline finding is **strongly suggestive but not yet seed-locked**; a clean re-launch is the immediate next step.

---

## 1. Study question

The grid world has three kinds of moving creatures:

- a **patrolling predator** (deals damage if it touches the agent),
- **four hiding-predators perched on rocks** (deal damage at a distance), and
- **harmless rabbits** (do nothing).

Normally each class carries its own **smell signature**, so the agent can tell them apart from far away by using its smell sensor. This study removes that cue: the **patrolling predator** and the **rabbits** are given the **same smell signature** ("sameProp"). The smell sensor can no longer separate them.

**The puzzle:** does the agent still behave differently around the dangerous one? If yes, *how* — by sight at close range, by post-contact pain memory, by movement pattern, by something else? And — the question that emerged halfway through — does the answer depend on **what kind of "differently" you measure**?

**Why this matters for the broader project.** The wider research goal is to model **hypervigilance** — over-cautious avoidance of safe things, driven by pain or fear. To make that claim cleanly, we need to know what *baseline* discrimination looks like under matched smells. This study mapped that baseline and, in doing so, surfaced a methodological lesson big enough to affect every future "the agent doesn't discriminate" verdict in this project.

---

## 2. Experiments completed this study

| # | Experiment | Plain-English question | What was changed | High-level finding (plain English) |
|---|---|---|---|---|
| **0** | **Existing-run survey** *(2026-05-07)* | Look at an already-running matched-smell training — does anything in the logs still distinguish predator from rabbit? | Nothing changed; just re-read the existing logs. | The agent kept the predator about **half a cell farther** than rabbits on average. Suggestive but one seed, and food + rabbits share the same quadrants — could be food-seeking, not class avoidance. |
| **1** | **Round 1 — clean re-run with proper logging** *(2026-05-07 → 2026-05-08)* | Re-run cleanly with two seeds and per-creature logging. Does the half-cell gap survive? | Per-creature distance logging implemented; two seeds (42, 43). | **Pattern replicates at both seeds** (gap ≈ +0.63 cells). Upgraded the survey from "anecdote" to "real signal." The food/rabbit-share-quadrants confound now needs a controlled test. |
| **2** | **Round 2 — kill the confounds** *(2026-05-08, stopped early)* | Cell C: move food *away* from rabbit quadrants. Cell A1: turn off the predator's hunt-mode, so only post-contact teaching can drive avoidance. | Cell C: food spawn-zones moved. Cell A1: predator restricted to one quadrant and made passive. | Both runs SIGINT'd at ~4–5% of budget. Cell C's gap **flipped sign** (rabbits farther than predator). Cell A1's gap *exploded* — but the agent never actually visited the predator's quadrant. We were measuring **corner-camping**, not class avoidance. Need a sharper metric. |
| **3** | **Round 2.5 — full budget + per-tag distances** *(2026-05-09 → 2026-05-10, 10M episodes per cell)* | Same two cells, but now measure distance to the *specific* same-corner predator vs the *specific* same-corner rabbit, so corner-camping doesn't fake a result. | No experimental knob change. Full 10M-episode budget; one seed per cell. | **Spatial verdict: NO class discrimination.** Cell A1 corner-camps (per-tag gap essentially zero, 0.004 cells). Cell C's gap inverts (rabbits farther than predator). The "agent treats predators specially in space" claim does not survive matched smells. |
| **3b** | **Round 2.5 appendix — same checkpoints, new "what does it do" measures** *(2026-05-11)* | Read the *same* converged agents through a new toolkit: bush-dive rate, eating-while-threatened ratio, behavioural-motif clustering. Does the agent's defensive *behaviour* discriminate, even though its *route* doesn't? | No new training — re-evaluated the Round 2.5 checkpoints with new metrics. 200 deterministic episodes per cell. | **Event-level verdict: YES, strongly, for Cell C.** Bush-dive rate when threat enters R=3 cells: predator **88%** vs rabbit **51%** → **+37 pp** gap. Eat-under-threat ratio: predator **0.75×** (eating suppressed) vs rabbit **1.19×** (eating elevated — agent treats nearby rabbit as a non-threat). Per-tag fan-out across the two rabbits is within noise → not a single-rabbit artifact. Cell A1 stays class-blind at every layer (corner-camping policy almost never has the agent feeding near a predator anyway). |
| **4** | **Round 2.6 — seed-lock attempt** *(2026-05-12 17:01 launch, crashed 23:38)* | Repeat Cell C (new seed 44) and Cell A1 (new seed 45) with the event-level toolkit logging live. Confirm the +37 pp gap holds at a fresh seed, and confirm Cell A1's class-blindness is robust to seeds too. | Seed 44 (Cell C) and seed 45 (Cell A1) on the same node (n106). Behavior-measure logging backfilled into the configs. | **Both runs CRASHED at ~6.6 h on n106**, reaching 1.18 M episodes (C) / 0.87 M episodes (A1) out of the 10 M-episode target. Crashed within 3 minutes of each other → likely a node-level event, not a training bug. **Partial data is encouraging:** Cell C's bush-dive gap was tracking at **+28.5 pp and still rising** when the run died (same direction as Round 2.5's +37 pp). But the seed-lock is not yet earned; a clean re-launch is needed. |

---

## 3. Where this leaves the study

- **The headline is a two-part answer, not one.** Under matched smells the agent is **class-blind in where it walks** but **class-discriminating in how it defends itself**. Both statements are simultaneously true; they're talking about different layers.

- **The Cell C event-level finding (+37 pp bush-dive gap, opposite-direction eat-suppression) is the paper-grade signal of this study.** It is large, internally consistent (per-tag fan-out within noise), backed by a third independent measure (motif clustering, 83% predator-triggered for the `predator_pursuit_with_bush` cluster) — and **directionally replicating in the Round 2.6 partial data before it crashed** (+28.5 pp and rising at 1.18 M episodes).

- **Round 2.6's crash blocks the formal seed-lock.** The current study verdict is "very strongly suggestive at one seed, partially replicating at a second seed before failure." That's enough to keep building on it (R3 design, NMN sequel) but not enough to publish as a closed result yet.

- **Cell A1's "corner-camping" was a seed-specific policy.** At seed 43 (Round 2.5) the agent corner-camped; at seed 45 (Round 2.6 partial) the agent picked a different "stay-and-eat" policy with the same high survival. Both are **class-blind at the event level**, so the *class-blindness conclusion* survives — but the *specific policy* underneath it doesn't. This vindicates the user's pre-launch call that Cell A1 also needed a second seed.

- **Big methodological win for the whole project.** Mean-distance metrics in this project are now known to be insensitive to event-level discrimination whenever the spatial layout is geometrically constrained (food in some quadrants, threats in others). Any future "no class discrimination" verdict should be cross-checked with the event-level toolkit (bush-dive rate, eat-suppression, motif clustering) before being trusted. Round 2.5 is the worked example of why.

- **Wider arc:** hypervigilance work should anchor on the event-level baseline (+37 pp bush-dive gap), not the spatial baseline (no gap). The latter has no signal under matched smells, so over-amplification by pain has nothing to perturb. The former does — and is the natural place to look for over-cautious bush-diving / over-suppressed eating in pain-treated agents.

---

## 4. What's next (still pending decision)

1. **Diagnose n106's crash before re-launching Round 2.6.** Both runs died within 3 min on the same node — almost certainly a node-level event (OOM from another tenant, GPU driver event, shared parent process killed). A blind re-launch risks burning another 22 h. *(`senior-developer` or user — quick log inspection on n106.)*

2. **Re-launch Round 2.6 on a healthy node, identical seeds.** Same configs (seed 44 Cell C, seed 45 Cell A1) on a different GPU pair, with the event-level toolkit logging live during training so we don't depend on post-hoc eval rollouts. ETA ~22 h. *(`training-runner`; blocked on item 1.)*

3. **Analyze Round 2.6 when it lands.** Apply the locked confirmation criteria from the Round 2.5 §12 appendix: bush-dive rate near predator ≥ 80%, gap vs rabbit ≥ +30 pp, eat-under-threat ratio near predator < 0.80. Confirm seed 45's "stay-and-eat" policy stays class-blind at the event level. *(`experiment-analyzer`; blocked on item 2.)*

4. **Re-summary after Round 2.6 closes.** Write the closing snapshot of the study. *(top-level Claude / `summarize-study`; blocked on item 3.)*

5. **Round 3 — close the spatial-camping loophole.** Spawn food in all four quadrants so no corner is uniformly safe and the agent can't trivially camp. Pre-register the event-level toolkit (M2 / M5 / M7) as the primary verdict. If the +37 pp gap survives, class discrimination is robust to spatial geometry; if it collapses, the Cell C event-level discrimination was itself spatially mediated. *(`experiment-designer` to author; not blocking R2.6.)*

6. **NMN / FiLM agent sequel on Cell C.** Does the modulated agent show a *larger* event-level class discrimination than plain RPPO under the same setup? Or is the +37 pp gap a baseline capability that the modulator doesn't add to? This is the natural bridge back to the NMN comparison thread. *(`experiment-designer` to flag in the R3 design.)*

7. **Promote the event-level toolkit to "default analysis layer."** Any future "no class discrimination" verdict gets cross-checked with M2 / M5 / M7 before being trusted. *(`experiment-analyzer` profile update; small scope.)*

---

## 5. Links

### Design docs

- [`docs/experiments/active/hypervigilance/sameprop_existing_run_survey.md`](../active/hypervigilance/sameprop_existing_run_survey.md) — the post-hoc survey that opened the study.
- [`docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md) — Round 1 analysis at two seeds (~7.2 M episodes).
- [`docs/experiments/active/hypervigilance/sameprop_round2_design.md`](../active/hypervigilance/sameprop_round2_design.md) — Round 2 design (Cells C + A1) with §9 truncated-data partial analysis.
- [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md) — Round 2.5 design with the §§9–11 spatial verdict and the §12 event-level appendix. **The single document that carries both halves of the verdict.**
- [`docs/experiments/active/hypervigilance/sameprop_round26_design.md`](../active/hypervigilance/sameprop_round26_design.md) — Round 2.6 seed-lock design (the crashed run).
- [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) — the event-level toolkit's experimental design.

### Supporting plans (under `docs/develop/active/`)

- [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../develop/active/hypervigilance/sameprop_discriminating_channels.md) — Phase-1 memo: which cues *can* still separate predator from rabbit under matched smells?
- [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../develop/active/hypervigilance/per_entity_avoidance_logging.md) — aggregated per-entity distance logging (Round 1's enabler).
- [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) — per-tag distance logging (the metric that resolved Round 2.5 at the spatial level).
- [`docs/develop/active/behavior/behavior_measure_toolkit_v1_plan.md`](../../develop/active/behavior/behavior_measure_toolkit_v1_plan.md) — the event-level toolkit implementation plan (`verification_status: pass`).
- [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../project/ideas/20260510_behavior_measure_toolkit.md) — biological grounding for the toolkit (predictive-coding / active-inference framing).

### Memory insights

- [`docs/llm_wiki/entries/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../../docs/llm_wiki/entries/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md) — **the event-level verdict** (+37 pp bush-dive gap, 0.75× / 1.19× eat-suppression, per-tag fan-out within noise).
- [`docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md`](../../../docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md) — **the spatial-level verdict** (Round 2.5 mean-distance verdict; complementary, not superseded).
- [`docs/llm_wiki/entries/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`](../../../docs/llm_wiki/entries/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md) — Round 2 partial verdict at ~5% of budget.
- [`docs/llm_wiki/entries/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../../docs/llm_wiki/entries/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md) — per-tag metric design rationale (methodological precursor to the §12 toolkit).
- [`docs/llm_wiki/entries/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md`](../../../docs/llm_wiki/entries/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md) — Round 1 verdict + confound flag.
- [`docs/llm_wiki/entries/hypervigilance/20260508_1445_sameprop_discriminating_channels.md`](../../../docs/llm_wiki/entries/hypervigilance/20260508_1445_sameprop_discriminating_channels.md) — Phase-1 channels memo.
- [`docs/llm_wiki/entries/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`](../../../docs/llm_wiki/entries/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md) — engineering lesson from the per-tag implementation week.

### Working files

- `tmp/20260510_round25_cellC_last10.md` — Cell C last-10% spatial analyzer worksheet.
- `tmp/20260510_round25_cellA1_last10.md` — Cell A1 last-10% spatial analyzer worksheet.
- `tmp/20260510_round25_combined.md` — combined cross-cell / cross-round contrast notes.
- `tmp/20260511_r25_appendix_*` — event-level appendix intermediate scripts + CSVs.

### Diary days

- [`docs/diary/2026-05-07.md`](../../diary/2026-05-07.md) — Round 1 relog launches, Round 2 design, Phase-1 channels memo.
- [`docs/diary/2026-05-08.md`](../../diary/2026-05-08.md) — Round 1 relog completes, Round 2 launches + SIGINT'd, partial-verdict analysis, three insights.
- [`docs/diary/2026-05-09.md`](../../diary/2026-05-09.md) — per-tag metrics revised plan + implementation + verification + bug-fix; three insights; Round 2.5 launch.
- [`docs/diary/2026-05-10.md`](../../diary/2026-05-10.md) — Round 2.5 completes; §§9–11 spatial verdict; insight captured.
- [`docs/diary/2026-05-11.md`](../../diary/2026-05-11.md) — behavior-measure toolkit shipped (29 tests, two bugs caught + fixed, `verification_status: pass`); §12 appendix.
- [`docs/diary/2026-05-12.md`](../../diary/2026-05-12.md) — event-level verdict insight captured; Round 2.6 designed + launched.
- [`docs/diary/2026-05-13.md`](../../diary/2026-05-13.md) — Round 2.6 progress check (mis-read as in-flight); prior re-summary generated.
- [`docs/diary/2026-05-14.md`](../../diary/2026-05-14.md) — Round 2.6 crash discovered (`state=failed` in WandB, ~6.6 h, n106-level event); this re-summary generated.

### Prior summaries (this study)

- [`docs/experiments/summaries/20260513_1420_sameprop_rabbit_avoidance_study.md`](20260513_1420_sameprop_rabbit_avoidance_study.md) — 2026-05-13 14:20 snapshot. **Caveat:** mis-reported Round 2.6 as in-flight (WandB query landed on the post-crash frozen snapshot). Content is otherwise correct; this re-summary is the reader-friendly rewrite + R2.6 status fix.
- [`docs/experiments/summaries/20260510_2253_sameprop_rabbit_avoidance_study.md`](20260510_2253_sameprop_rabbit_avoidance_study.md) — 2026-05-10 22:53 snapshot; the spatial-level-only verdict.
- [`docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md`](20260509_1552_sameprop_rabbit_avoidance_study.md) — 2026-05-09 15:52 snapshot; pre-Round-2.5.

### Implementation commits

- `4b55fc6` — `feat(hypervigilance): per-entity avoidance logging` — Round 1 relog enabler.
- `42cc049` — `docs(hypervigilance): Round 2 truncated-data partial analysis` — §9 of Round 2 design.
- `0a73613` — `feat(hypervigilance): per-tag per-instance distance logging (Round-2 §7)` — spatial-level resolver.
- `82ae039` — `docs(hypervigilance): Round 2.5 analysis — H₀(A1) confirmed, Cell C inverted (provisional)` — spatial-level verdict.
- `ba0d766` — `docs(memory): capture 1 insight — R2.5 sameProp class avoidance refuted` — spatial-level verdict memory.
- `e5e1155`, `2d6d288` — behavior-measure toolkit bug fixes after independent verification.
- `dbd0e64` — toolkit `verification_status: pass`.
- `ed5cff3` — `docs(hypervigilance): §12 toolkit appendix to R2.5 design` — **event-level verdict commit**.
- `c110a2c` — `docs(memory): capture 1 insight — R2.5 event-level class discrimination` — event-level verdict memory.
- `83ab27c` — `docs(hypervigilance): Round 2.6 design` — seed-lock design.
- `c81b48d` — `docs(experiments): behavior_measures: block backfill into R2.6 configs` — toolkit logging in R2.6 configs.

---

## 6. Reading order if you have 10 minutes

1. **This summary, sections 1–4** (5 min) — gives you the take-home messages, the experiment table, the two-level verdict, and the open follow-ups.
2. **§12 of [`sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md)** (3 min) — the event-level toolkit appendix with the actual per-class numbers and the motif-clustering breakdown.
3. **The event-level verdict insight, [`20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../../docs/llm_wiki/entries/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md)** (2 min) — the same verdict in 5-section memory form with the methodological generalization.

If you have 30 minutes, also read:

4. **§§9–11 of [`sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md)** — the spatial-level verdict that the §12 appendix complements.
5. **The spatial-level verdict insight, [`20260510_2237_sameprop_round25_no_class_avoidance.md`](../../../docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md)** — two-confound decomposition.
6. **The prior re-summary at [`20260513_1420_sameprop_rabbit_avoidance_study.md`](20260513_1420_sameprop_rabbit_avoidance_study.md)** — useful for seeing how the framing of the two-level verdict was developed (skip its "R2.6 in-flight" framing — it crashed).
7. **The toolkit's biological grounding, [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../project/ideas/20260510_behavior_measure_toolkit.md)** — why M1 / M2 / M5 map to the predictive-coding / active-inference defensive repertoire.
