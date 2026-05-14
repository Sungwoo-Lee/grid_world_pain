---
title: "SameProp Rabbit-Avoidance Study — Round 2.5 verdict refined to two-level interpretation (re-summary, week of 2026-05-07 → 2026-05-13)"
study: sameprop_rabbit_avoidance_study
generated: 2026-05-13T14:20
window: "2026-05-07 → 2026-05-13"
status: snapshot
---

# SameProp Rabbit-Avoidance Study — Re-summary as of 2026-05-13 14:20 KST

> **One-paragraph summary.** This study tested whether the agent **learns to treat predators and neutral rabbits differently when their olfactory signatures are made identical** ("sameProp" — both classes carry the same smell-sensor `properties` vector). A post-hoc survey, a Round-1 re-run with per-entity logging at two seeds, a partial Round 2 confound-control attempt, and a full-budget Round 2.5 with a new per-tag distance metric all converged, by 2026-05-10, on the conclusion that **at the spatial-trajectory level** — mean distance to predator vs. mean distance to rabbit — the agent does *not* show class-conditional avoidance under matched smells (Cell A1 corner-camps; Cell C's gap inverts and becomes bilateral rabbit avoidance). That verdict is the prior summary's headline and it still stands. **What's new since the 2026-05-10 snapshot**: a fine-grained behavior-measure toolkit (M1 interrupted-feeding, M2 bush-dive, M5 eat-under-threat, M7 motif clustering) shipped on 2026-05-11 and was applied to the same Round 2.5 checkpoints as a §12 appendix. The appendix **inverts the framing** at a different level: in Cell C, the agent's bush-dive rate is **+37 percentage points higher** when a predator (vs a rabbit) enters R=3 cells (87.6% vs 50.8%); its feeding is **suppressed to 0.75×** baseline near a predator and **elevated to 1.19×** baseline near a rabbit; a within-class fan-out across the two rabbit tags is within noise, ruling out single-instance artifacts. **Under matched smells, the agent IS class-discriminating — but in its defensive event repertoire, not in its foraging route.** The mean-distance metric averages over many threat-far steps and dissolves the signal; bush-diving, eat-suppression, and motif selection don't. Round 2.6 (seed 44 Cell C + seed 45 Cell A1) is in flight to seed-lock both findings before publication; at 29% elapsed time the Cell C event-level discrimination is already replicating directionally (window-6 bush-dive gap +28.5 pp and still climbing).

> **This is a re-summary, not a replacement.** The prior summary at [`20260510_2253_sameprop_rabbit_avoidance_study.md`](20260510_2253_sameprop_rabbit_avoidance_study.md) reports the spatial-level verdict ("no class-conditional avoidance under sameProp") and is **correct at its level**. This re-summary adds the event-level finding the toolkit surfaced on 2026-05-11–12, which complements rather than supersedes it. Read the prior summary for the spatial verdict; read this one for the two-level synthesis.

> **This is a snapshot.** Re-summaries are written as new dated files in this folder; the older summary stays as the historical snapshot of what was known on 2026-05-10.

---

## 1. Study question

The environment has three classes of moving entities the agent encounters: a **patrolling predator** (damages on contact), four **hiding-predators on rocks** (damage at distance), and **neutral rabbits** (harmless). Normally the agent can tell predators from rabbits by their **olfactory `properties` vector** — a per-class one-hot the smell sensor returns when an entity is nearby. The "sameProp" condition deliberately collides those vectors: the patrolling predator and the neutral rabbits are given **the same olfactory `properties`**, so the agent cannot use smell alone to distinguish them.

**The puzzle**: does the agent still avoid predators preferentially? If yes, what cue is it using — visual class identity at contact, post-contact pain teaching, the predator's distinctive movement pattern, or just "the predator's quadrant happens to be unfriendly"? And — newly added with the 2026-05-11 toolkit — does the answer depend on **what kind of "preferential treatment" you measure**? Mean distance is one answer; bush-diving when threatened is another.

**The dependent variables, by level**:

- **Spatial-trajectory level** (Round 2.5 §§9–11): per-class mean distance — `MeanDistRabbit` vs `MeanDistPredator` (cells, L2), per-tag distance to same-corner predator vs same-corner rabbit, plus contact counts (`RabbitHits`, `PredatorHits`).
- **Event-response level** (Round 2.5 §12, added 2026-05-11): per-class event rates — interrupted-feeding rate (M1), bush-dive rate when a class enters R=3 (M2), eat-under-threat ratio (M5, eating-rate-near-class / eating-rate-far-from-class), and trajectory-motif clustering of threat-window snippets (M7).

A genuinely class-avoiding agent at the *spatial* level should keep predators systematically farther than rabbits in mean L2 distance. A genuinely class-discriminating agent at the *event* level should fire defensive actions (bush-dive, eat-suppress) more often near a predator than near a rabbit, even when its spatial route looks similar near both. These can come apart, and under sameProp they do.

**Why this matters for the wider project**: the project's larger thesis ties rabbit-avoidance behaviour to a "hypervigilance" signature — over-avoidance of safe entities under uncertainty. To talk about hypervigilance honestly, we first need to know what *baseline* discrimination looks like under matched olfactory channels, and the two-level finding sharpens the question: hypervigilance might over-amplify *event-level* defences (the part that survives sameProp) rather than spatial avoidance (the part that doesn't).

---

## 2. Experiments completed this study

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |
|---|---|---|---|---|---|
| **0** *(post-hoc, single seed)* | **SameProp existing-run survey** *(2026-05-07)* | Does the agent's behaviour, under matched olfactory smells, still distinguish predators from rabbits in any measurable way? | Nothing — analyzed an already-running training (n=1) using survival + per-class distance metrics that were already in the WandB log. | The agent kept the patrolling predator about **0.6 cells farther** than rabbits at convergence, with **3× more rabbit-contacts than predator-contacts** per episode. | Established the central observation: matched smells alone do not erase class-conditional behaviour. Motivated a clean re-run with two seeds and proper per-entity logging. The analyzer also flagged a confound: food and rabbits share the same two quadrants, while the predator roams the full grid — so the gap might be food-seeking spillover, not class avoidance. |
| **1** | **Round 1 relog** *(2026-05-07 evening → 2026-05-08 noon)* | Does the same pattern hold when we re-run cleanly with two seeds and **per-entity logging** added to the codebase? | Per-entity logging implementation merged. Two seeds (42, 43) of the same sameProp config relaunched. | **Pattern confirmed at both seeds.** Gap ≈ +0.63 cells; rabbit-contacts ≈ 6.5/ep, predator-contacts ≈ 3.4/ep, gap ≈ +3.1/ep. The two seeds agreed within ±0.03 cells across all distance metrics. | Promoted the survey from "n=1 motivation" to a replicated finding. The food/quadrant confound was now formally on the docket. |
| **2** | **Round 2 — confound control + movement-signature ablation** *(2026-05-08, n=1 each cell, ~4–5% of budget)* | **Cell C (decoupleFood)**: does the gap survive when food and rabbits no longer share quadrants? **Cell A1 (passivePredator)**: does the gap survive on post-contact teaching alone, with the predator's hunt mode disabled? | Cell C: food spawn-zones moved to the two quadrants without rabbits. Cell A1: predator restricted to top-left quadrant, hunt-stamina threshold raised so the predator never engages hunt mode. | Both cells stopped at ~3.5 h / ~0.4 M episodes. Cell C: gap **flipped sign** (rabbits ~0.4 cells farther than predators). Cell A1: aggregated gap grew to ~3.9 cells, but the agent never visited the predator's quadrant — measuring corner-camping not class avoidance. | Pushed the next chain step to be **a tooling change, not another experiment**: ship a per-tag distance metric so same-corner predator and same-corner rabbit get separate, comparable distance numbers. |
| **3** | **Round 2.5 — full-budget re-launch with per-tag metrics, §§9–11 spatial verdict** *(2026-05-09 evening → 2026-05-10 evening, single seed per cell, 10 M ep each)* | Same two cells, same configs, but now read off per-tag distance keys so corner-camping (Cell A1's contamination) and food-spillover (Cell C's confound) become independently testable. | No experimental knob change. Single seed per cell. 10 M episodes each. Pre-registered analysis window 9–10 M ep. | **Both pre-registered spatial hypotheses refuted.** Cell A1 per-tag predator-vs-rabbit gap = +0.004 cells at 486/500 survival (corner-camping signature). Cell C aggregated gap = −0.53 cells (rabbits farther than predator), with both rabbit corners avoided. | **At the spatial-trajectory level, the study resolves to "no genuine class-conditional avoidance under sameProp."** The per-tag metric earned its keep — without it, Cell A1's aggregated +3.86-cell gap would have been published as "class avoidance dramatically confirmed." |
| **3b** | **Round 2.5 §12 toolkit appendix — event-response re-read** *(2026-05-11)* | Reads the *same* two Round-2.5 checkpoints with a new behavior-measure toolkit (M1 interrupted-feeding, M2 bush-dive, M5 eat-under-threat, M7 motif clustering). Does the event-level repertoire discriminate by class even when the mean distance does not? | No new training. 200 deterministic eval episodes per cell-checkpoint at R=3.0 cells threat-window. | **Inverts the framing at the event level for Cell C.** Cell C bush-dive rate predator vs rabbit: **87.6% vs 50.8% (+36.8 pp gap)**. Eat-under-threat ratio: **0.75× near predator** (eating suppressed), **1.19× near rabbit** (eating slightly elevated — agent treats rabbit-proximity as a non-threat marker). Per-tag rabbit fan-out (rabbit_TL vs rabbit_BR) within 2 pp on M2, 0.08 on M5 — rules out single-instance artifacts. Motif clustering: ~78% of threat-window activity in bush-involved clusters, one cluster (`predator_pursuit_with_bush`) 83% predator-triggered. Cell A1 remains class-blind at every layer (corner-camping policy almost never has the agent encountering a same-corner predator while feeding). | **Cell C is spatially class-blind but behaviourally class-discriminating.** The spatial-level verdict (§§9–11) is correct but incomplete: mean-distance metrics measure foraging routes, not defensive responses. Under sameProp the class signal lives in the defensive event repertoire, and the mean dissolves it. |
| **4** *(IN FLIGHT — not yet in scope as a completed experiment; see §4)* | **Round 2.6 — seed-lock of Round 2.5 with behavior-toolkit measures as primary** *(launched 2026-05-12 17:02–17:05, ETA ~22 h)* | Does the §12 event-level discrimination replicate at a fresh seed? Does Cell A1's corner-camping policy also replicate, or does the agent find a different "safe" policy at a different seed? | Seed 44 Cell C + Seed 45 Cell A1, both n106 RTX 3090, same configs as Round 2.5. `behavior_measures:` block backfilled so M1/M2/M5 log live during training. | **Provisional snapshot at ~29% elapsed time** (checked 2026-05-13 14:20). Cell C-s44: bush-dive gap predator vs rabbit = +28.5 pp and still climbing window-by-window (R2.5 was +36.8 pp; same direction, similar trajectory); eat-under-threat ratio 0.99 vs 1.40, predator suppression just crossing into < 1.0. Cell A1-s45: ep_len ≈ 491 (max-step cap), but reward −120 vs Cell C's −205, FoodEaten 174 vs 56, RabbitHits 17.9 vs 0.7 — **a different policy regime**: "stay-and-eat-and-absorb-hits" rather than corner-camping. Class-blind at the event level (bush-dive ~0.04/0.01; EUT ratio ~1.05 both). | **Cell C event-level discrimination is replicating at seed 44 directionally**; final magnitudes likely closer to R2.5 by 22 h. **Cell A1 corner-camping is seed-specific** — at seed 45 the same config converges to a different high-survival policy, supporting the user's call (before launch) that Cell A1 needed a second seed too. Verdict still incomplete; full results expected ~09:00 KST 2026-05-14. |

---

## 3. Where this leaves the study

- **The study now has a two-level verdict, not one.** At the **spatial-trajectory level** (mean L2 distance per class), under sameProp the agent does *not* learn class-conditional avoidance — Cell A1 corner-camps, Cell C inverts to bilateral rabbit avoidance, neither is class recognition. At the **event-response level** (bush-dive rate, eat-under-threat suppression, motif selection), Cell C *is* class-discriminating by paper-grade margins (+37 pp bush-dive gap, opposite-direction eat-suppression). The two levels are complementary, not contradictory: mean distance averages over many threat-far steps and dissolves the event-level signal.
- **The Cell A1 spatial verdict is rock-solid; the Cell C spatial verdict needs Round 2.6 for seed-lock.** Cell A1's per-tag predator-vs-rabbit gap is 75× past the corner-camping threshold and stable across the last three of ten temporal sub-windows; one seed is enough. Cell C's sign-flip is at 1.6–1.8× threshold, and Round 2.6 seed 44 (in flight) provides the second seed.
- **The Cell C event-level verdict needs Round 2.6 for seed-lock too**, but the partial replication at 29% time is encouraging: window-6 bush-dive gap is +28.5 pp and climbing, with the predator EUT ratio just crossing 1.0 into suppression territory. Same direction as R2.5; magnitudes likely to keep growing.
- **Cell A1's "corner-camping" finding was seed-specific.** The user pushed back before launch ("single-seed nulls are weaker than single-seed positives"), which Round 2.6 is now validating: at seed 45, Cell A1 converges to a *different* high-survival policy ("stay-and-eat-and-take-rabbit-hits") rather than corner-camping. Both are class-blind at the event level — so the *class-blindness* of Cell A1 is robust, but the *specific policy* under it is not.
- **Methodology lesson, generalised**: aggregated mean-distance metrics in this project may be insensitive to event-level class discrimination whenever the agent's spatial trajectory is geometrically constrained (food in one set of quadrants, threats in another). Whenever a mean-distance verdict says "no class-conditional avoidance", the toolkit's M2 + M5 + M7 should be applied before concluding the agent is class-blind. The Round 2.5 case is a worked example; future studies should adopt this as standard practice.
- **The methodological win compounds.** Round 2.5's earlier methodology win was the per-tag distance metric (which flipped Cell A1's aggregated 10× confirmation into a 75× null). The §12 appendix's win is the same shape one level up: a new measurement layer (event-rate counters + motif clustering) recovered a signal the prior layer dissolved. The toolkit shipped with `verification_status: pass` after 29 tests and two bugs caught by independent verification — both findings rest on the same engineering investment.
- **The wider hypervigilance arc has a much richer baseline now.** Not just "does the agent avoid the dangerous class in space?" (answer: no) but "does its event-level defensive repertoire treat the dangerous class differently?" (answer: yes, at +37 pp). Future hypervigilance work should anchor on the event-level baseline; the spatial baseline is class-blind under sameProp and offers no signal to perturb.

---

## 4. What's next (still pending decision)

1. **Wait for Round 2.6 to finish** — ETA ~09:00 KST 2026-05-14, ~17 h from this re-summary. Both runs healthy at 29% elapsed time, no NaN / no collapse. *(no action; passive wait.)*
2. **Spawn `experiment-analyzer` when Round 2.6 lands.** Fills in the R2.6 verdict using the event-level toolkit measures (M1/M2/M5) as primary criteria per §12.7 of the R2.5 design — locked confirmation: `M2_predator ≥ 0.80 AND Δ_M2_class ≥ +30 pp AND M5_predator < 0.80`. Cross-check the +37 pp bush-dive gap at seed 44 (Cell C) and confirm seed 45's "stay-and-eat" policy is still class-blind at the event level (Cell A1). *(`experiment-analyzer`; blocked on item 1.)*
3. **Re-summary after Round 2.6.** When the seed-lock lands, write a fourth snapshot in `summaries/` that closes the study. The current re-summary is the two-level verdict snapshot; the post-2.6 summary is the closing snapshot. *(top-level Claude / `summarize-study` skill; blocked on item 2.)*
4. **Round 3 — close the spatial-camping loophole.** The cleanest follow-up is a sameProp configuration with food spawned in *all four* quadrants, so no corner is uniformly safe and the agent cannot trivially camp. Pre-register M2 / M5 *widening* (or maintaining the +37 pp gap) as the confirmation signal for class-conditional defence under a closed spatial loophole. If M2_predator / M5_predator under all-4-quadrant food still show the +37 pp / 0.75× pattern, class discrimination is robust to spatial geometry; if they collapse, the Cell C event-level discrimination was itself spatially mediated. *(`experiment-designer` to author; not blocking 2.6.)*
5. **NMN / FiLM agent sequel.** Does a modulated agent under the same Cell C setup show a *larger* event-level class discrimination (higher Δ_M2)? Or is the +37 pp gap a baseline plain-RPPO capability and the modulator doesn't help here? Natural sequel that ties this study back to the project's NMN comparison thread. *(`experiment-designer` to flag in the Round-3 design.)*
6. **Promote behavior-measure toolkit to "default analysis layer".** Whenever a mean-distance verdict says "no class-conditional avoidance", M2 + M5 + M7 are applied before concluding the agent is class-blind. *(`experiment-analyzer` profile update; small scope.)*

---

## 5. Links

### Design docs

- [`docs/experiments/active/hypervigilance/sameprop_existing_run_survey.md`](../active/hypervigilance/sameprop_existing_run_survey.md) — the post-hoc survey that opened the study.
- [`docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md) — formal Round 1 analysis (n=2 seeds at ≈ 7.2 M episodes).
- [`docs/experiments/active/hypervigilance/sameprop_round2_design.md`](../active/hypervigilance/sameprop_round2_design.md) — Round 2 design (Cells C + A1) with the §9 truncated-data partial analysis appended.
- [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md) — Round 2.5 design with §§9–11 spatial verdict and §12 event-level toolkit appendix. **The doc that carries both levels of the verdict.**
- [`docs/experiments/active/hypervigilance/sameprop_round26_design.md`](../active/hypervigilance/sameprop_round26_design.md) — Round 2.6 seed-lock design (in flight as of this re-summary).
- [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) — the behavior-measure toolkit's experimental design.

### Supporting plans (under `docs/develop/active/`)

- [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../develop/active/hypervigilance/sameprop_discriminating_channels.md) — Phase-1 channel-ranking memo: under matched smells, what cues *can* still distinguish predator from rabbit? Feeds Round 2's design choices.
- [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../develop/active/hypervigilance/per_entity_avoidance_logging.md) — prior aggregated per-entity logging plan; the precursor that Round 1 relog uses.
- [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) — the tag-based per-instance distance plan. The metric that resolved Round 2.5 Cell A1 at the spatial level.
- [`docs/develop/active/behavior/behavior_measure_toolkit_v1_plan.md`](../../develop/active/behavior/behavior_measure_toolkit_v1_plan.md) — the toolkit implementation plan (`verification_status: pass`, commit `dbd0e64`). The measures that resolved Round 2.5 Cell C at the event level.
- [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../project/ideas/20260510_behavior_measure_toolkit.md) — biological grounding for the toolkit (predictive-coding + active-inference framing).

### Memory insights

- [`.claude-memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../../.claude-memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md) — **the new event-level verdict insight** (complements, does not supersede, the spatial verdict). +37 pp bush-dive gap, 0.75× / 1.19× eat-suppression, per-tag fan-out within noise.
- [`.claude-memory/memories/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md`](../../../.claude-memory/memories/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md) — **the prior spatial verdict insight**. Round 2.5 two-cell decomposition at the mean-distance level; correct at its level, complemented (not superseded) by the 2026-05-12 insight.
- [`.claude-memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`](../../../.claude-memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md) — Round 2 partial verdict (the truncated run that pointed at the spatial verdict at ~5% of the budget).
- [`.claude-memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../../.claude-memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md) — the per-tag metric design rationale; the methodological precursor to the §12 toolkit (same shape of lesson: new measurement layer recovers a signal the prior layer dissolved).
- [`.claude-memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md`](../../../.claude-memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md) — Round 1 verdict + confound flag.
- [`.claude-memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md`](../../../.claude-memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md) — channels memo (movement signature dominant, visual ch.5/ch.7 at contact, extero-noc contact-only) and the structural-disable trick used in Cell A1.
- [`.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`](../../../.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md) — wiring-bug + verification-practice lesson from the per-tag implementation week.

### Working files

- `tmp/20260510_round25_cellC_last10.md` — Cell C last-10% window spatial analyzer worksheet.
- `tmp/20260510_round25_cellA1_last10.md` — Cell A1 last-10% window spatial analyzer worksheet.
- `tmp/20260510_round25_combined.md` — combined cross-cell + cross-round contrast notes.
- `tmp/20260511_r25_appendix_*` — toolkit appendix intermediate scripts + CSVs (Cell A1 + Cell C event-level eval-rollout data).

### Diary days

- [`docs/diary/2026-05-07.md`](../../diary/2026-05-07.md) — Round 1 relog launches, Round 2 design, Phase-1 channels memo.
- [`docs/diary/2026-05-08.md`](../../diary/2026-05-08.md) — Round 1 relog completes, Round 2 launches + SIGINT'd, partial-verdict analysis written, three insights captured.
- [`docs/diary/2026-05-09.md`](../../diary/2026-05-09.md) — per-tag metrics revised plan + implementation + verification + bug-fix; three insights captured; Round 2.5 design + launch.
- [`docs/diary/2026-05-10.md`](../../diary/2026-05-10.md) — Round 2.5 completes; analyzer fills in §§9–11; spatial verdict insight captured.
- [`docs/diary/2026-05-11.md`](../../diary/2026-05-11.md) — behavior-measure toolkit shipped (29 tests, two bugs caught + fixed, `verification_status: pass`); §12 appendix to R2.5 design.
- [`docs/diary/2026-05-12.md`](../../diary/2026-05-12.md) — event-level verdict-refinement insight captured; Round 2.6 designed + launched (Cell C seed 44 n106 cuda:0; Cell A1 seed 45 n106 cuda:1).
- [`docs/diary/2026-05-13.md`](../../diary/2026-05-13.md) — Round 2.6 progress check at 29% elapsed time; this re-summary generated.

### Prior summaries (this study)

- [`docs/experiments/summaries/20260510_2253_sameprop_rabbit_avoidance_study.md`](20260510_2253_sameprop_rabbit_avoidance_study.md) — the snapshot at 2026-05-10 22:53, the spatial-level verdict. Retained as a historical snapshot; complemented (not superseded) by this re-summary at the event level.
- [`docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md`](20260509_1552_sameprop_rabbit_avoidance_study.md) — the snapshot at 2026-05-09 15:52, written before Round 2.5 finished. Retained as a historical snapshot.

### Implementation commits

- `4b55fc6` — `feat(hypervigilance): per-entity avoidance logging` — the Round-1-relog enabler.
- `42cc049` — `docs(hypervigilance): Round 2 truncated-data partial analysis` — §9 of the Round-2 design doc.
- `0a73613` — `feat(hypervigilance): per-tag per-instance distance logging (Round-2 §7)` — the spatial-level resolver and the toolkit foundation.
- `82ae039` — `docs(hypervigilance): Round 2.5 analysis — H₀(A1) confirmed, Cell C inverted (provisional)` — the spatial-level verdict commit.
- `ba0d766` — `docs(memory): capture 1 insight — R2.5 sameProp class avoidance refuted` — the spatial-level verdict memory commit.
- `e5e1155`, `2d6d288` — behavior-measure toolkit bug fixes after independent verification.
- `dbd0e64` — toolkit re-verify pass (`verification_status: pass`).
- `ed5cff3` — `docs(hypervigilance): §12 toolkit appendix to R2.5 design` — the **event-level verdict commit**.
- `c110a2c` — `docs(memory): capture 1 insight — R2.5 event-level class discrimination` — the **event-level verdict memory commit**.
- `83ab27c` — `docs(hypervigilance): Round 2.6 design` — the seed-lock design.
- `c81b48d` — `docs(experiments): behavior_measures: block backfill into R2.6 configs` — live toolkit logging during R2.6 training.

---

## 6. Reading order if you have 10 minutes

1. **This summary** — start here (5 minutes).
2. **§12 of [`sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md)** — the toolkit appendix with the per-cell × per-class × per-tag M1/M2/M5 table, the M7 motif clustering, the cross-cell comparison, and the explicit "what §§9–11 missed, and what they confirmed" section (3 minutes).
3. **The event-level verdict insight, [`20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../../.claude-memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md)** — the same verdict in 5-section form with the methodological generalization (mean-distance metrics are insensitive to event-level discrimination when spatial trajectory is geometrically constrained) and the implications for R2.6 + R3 + the NMN sequel (2 minutes).

If you have 30 minutes, also read:

4. **§§9–11 of [`sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md)** — the spatial-level verdict: per-tag numbers, temporal-evolution table, cross-round contrast against Round 1, the §5 failure-mode catalog mapping, and the R2.6 escalation policy. The level the §12 appendix complements.
5. **The spatial-level verdict insight, [`20260510_2237_sameprop_round25_no_class_avoidance.md`](../../../.claude-memory/memories/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md)** — the two-confound decomposition argument; the methodology takeaway about pre-registered failure-mode catalogs.
6. **The prior summary, [`20260510_2253_sameprop_rabbit_avoidance_study.md`](20260510_2253_sameprop_rabbit_avoidance_study.md)** — the spatial-only snapshot at 2026-05-10, useful for seeing how the verdict has refined.
7. **The behavior-measure toolkit's biological grounding, [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../project/ideas/20260510_behavior_measure_toolkit.md)** — why the M1/M2/M5 measures map to the predictive-coding / active-inference defensive repertoire, and why a paper-grade behavioural decomposition needs them.
