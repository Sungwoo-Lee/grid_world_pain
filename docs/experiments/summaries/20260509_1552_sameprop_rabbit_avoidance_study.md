---
title: "SameProp Rabbit-Avoidance Study — Week of 2026-05-07 → 2026-05-09 (R1 relog + R2 partial + per-tag metrics ship)"
study: sameprop_rabbit_avoidance_study
generated: 2026-05-09T15:52
window: "2026-05-07 → 2026-05-09"
status: snapshot
---

# SameProp Rabbit-Avoidance Study — Summary as of 2026-05-09 15:52 KST

> **One-paragraph summary.** This week's work tested whether the agent **learns to treat predators and neutral rabbits differently when their olfactory signatures are made identical** ("sameProp"). A post-hoc survey at the start of the week noticed the agent appears to keep predators ~0.6 cells *farther* from itself than rabbits — but the experimental setup also placed food and rabbits in the same two quadrants while the predator roamed the full grid, so the distance gap might be food-seeking spillover, not class-level avoidance. A **pre-registered re-run with per-entity logging confirmed the same gap at two seeds**. A **follow-up Round 2** then attempted to test the food/quadrant confound directly (Cell C: move the food away from the rabbits) and to ablate the predator's pre-contact movement signature (Cell A1: disable the predator's hunt mode and shrink its patrol zone). **Both Round-2 cells were stopped early at ~4-5% of their planned training budget**, so the partial verdict is provisional, but the early signal points two ways at once: Cell C's distance gap *flipped sign* (rabbits now further than predators), suggesting the food/quadrant overlap was load-bearing for the original effect; Cell A1's gap got *six times bigger*, but with a hidden contamination — the agent survives most of every episode in the safe corner and never visits the predator's quadrant, so the gap is measuring location-avoidance, not class-avoidance. **The headline lesson** is that the original survey's finding was probably overstated, and the way to settle the cell-A1 ambiguity isn't more training but **better metrics**. As a final deliverable for the week, geometry-agnostic **per-tag distance metrics** were designed and merged so Round 2.5 can be re-launched with class-vs-quadrant disambiguation built in.

> **This is a snapshot.** Re-summaries should be written as new dated files in this folder, not by editing this one.

---

## 1. Study question

The environment has three classes of moving entities the agent encounters: a **patrolling predator** (damages on contact), four **hiding-predators on rocks** (damage at distance), and **neutral rabbits** (harmless). Normally the agent can tell predators from rabbits by their **olfactory `properties` vector** — a per-class one-hot the smell sensor returns when an entity is nearby. The "sameProp" condition deliberately collides those vectors: the patrolling predator and the neutral rabbits are given **the same olfactory `properties`**, so the agent cannot use smell alone to distinguish them.

**The puzzle**: does the agent still avoid predators preferentially? If yes, what cue is it using — visual class identity at contact, post-contact pain teaching, the predator's distinctive movement pattern, or just "the predator's quadrant happens to be unfriendly"?

**The dependent variables** are per-class distances and contact counts: `MeanDistRabbit` vs `MeanDistPredator` (cells, L2), `RabbitHits` vs `PredatorHits` (count per episode), and survival steps. A genuinely class-avoiding agent should keep predators *systematically* farther than rabbits — not just because of where they spawn, but because it has learned to recognise them.

**Why this matters for the wider project**: the project's larger thesis ties rabbit avoidance behaviour to a "hypervigilance" signature — over-avoidance of safe entities under uncertainty. To talk about hypervigilance honestly, we first need to know what *baseline* avoidance looks like under matched olfactory channels. This week's work is the baseline measurement.

---

## 2. Experiments completed this week

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |
|---|---|---|---|---|---|
| **0** *(post-hoc, single seed)* | **SameProp existing-run survey** *(2026-05-07)* | Does the agent's behaviour, under matched olfactory smells, still distinguish predators from rabbits in any measurable way? | Nothing — analyzed an already-running training (`interoNocicept_smaProp` run, n=1) using survival + per-class distance metrics that were already in the WandB log. | The agent kept the patrolling predator about **0.6 cells farther** than rabbits at convergence, with **3× more rabbit-contacts than predator-contacts** per episode. Survival was unimpressive (≈340 steps), but the *direction* of the asymmetry was clear and consistent. | Established the central observation: matched smells alone do not erase class-conditional behaviour. Motivated a clean re-run with two seeds and proper per-entity logging. **Also surfaced a confound** the analyzer flagged in parallel: food and rabbits share the same two quadrants, while the predator roams the full grid — so "distance to predator > distance to rabbit" might be "agent goes where food is, rabbits happen to be there." |
| **1** | **Round 1 relog** *(2026-05-07 evening → 2026-05-08 noon)* | Does the same pattern hold when we re-run cleanly with two seeds and **per-entity logging** added to the codebase? | Per-entity logging implementation merged (`Episode/MeanDistRabbit`, `Episode/RabbitHits`, `Episode/MeanDistHidingPredator`, `Episode/HidingPredatorHits`). Two seeds (42, 43) of the same sameProp config relaunched on node 112. | **Pattern confirmed at both seeds.** At ~7.2 million episodes (training stopped early due to time, not by design): MeanDistRabbit ≈ 3.77, MeanDistPredator ≈ 4.40, gap ≈ +0.63 cells; RabbitHits ≈ 6.5/ep, PredatorHits ≈ 3.4/ep, gap ≈ +3.1/ep. The two seeds agreed within ±0.03 cells across all distance metrics. Random-policy floor on this 10×10 grid is ≈ 4.68 cells, so the rabbit-distance number is meaningfully below the random-walk baseline; the predator-distance number is roughly *at* random-walk. | Promoted the survey from "n=1 motivation" to a replicated finding. The **food/quadrant confound** flagged at the survey stage was now formally on the docket: nothing in this experiment *resolved* it, so the next round had to attack it directly. |
| **2** | **Round 2 — confound control + movement-signature ablation** *(2026-05-08 afternoon, n=1 seed each cell, partial budget)* | Two cells in parallel. **Cell C (decoupleFood)**: with the food and rabbits no longer sharing quadrants, does the distance gap survive — i.e. is it really class-conditional avoidance, or was the survey just measuring food-seeking? **Cell A1 (passivePredator)**: with the predator's hunt mode structurally disabled and its patrol zone shrunk to one quadrant — eliminating its distinctive movement pattern — does the gap survive on post-contact teaching alone? | Cell C: food spawn-zones moved to the two quadrants without rabbits; rabbits left where they were. Cell A1: predator restricted to top-left quadrant only, hunt-stamina threshold raised so the predator never engages hunt mode (kinematically equivalent to a rabbit). | **Both cells were SIGINT'd at ~3.5 hours / ~0.4 million episodes** — about 4–5% of the 10-million-episode plan. Verdicts are provisional, not final, but the early signal is striking on both. **Cell C**: distance gap **flipped sign** (rabbits now ~0.4 cells *farther* than predators). The provisional reading is that the survey's apparent "class avoidance" was substantially driven by food-quadrant overlap — when food moves away from rabbits, the agent's relationship to those quadrants inverts. **Cell A1**: distance gap **grew to ~3.9 cells** — six times the original effect — which on its face would confirm class avoidance even *without* the predator's movement signature. **However**, the agent in Cell A1 survives ~482 of 500 steps and never visits the predator's quadrant, which means the metric is measuring "agent stays in the safe corner" not "agent recognises predators". The verdict is **uninterpretable without per-quadrant or per-class evidence the current logging cannot provide**. | Two separate updates. **For Cell C**: the original survey finding was probably overstated; the food/quadrant overlap was a real confound and is doing meaningful work in the original signal. **For Cell A1**: re-launching at full budget would not resolve the ambiguity — the metric itself is the bottleneck, not the training horizon. This finding pushed the next item on the chain to **be a tooling change, not another experiment**. |

---

## 3. Where this leaves the study

- **The original "rabbits closer than predators under sameProp" finding (≈ 0.6 cells) is partially confounded.** Round 2 Cell C's early reading inverts the sign, which strongly suggests food-quadrant overlap was contributing — possibly the dominant contributor — to the survey and the Round-1 relog. We do not yet know how much; that requires Round 2.5 at full training budget.
- **Whether class-conditional avoidance exists at all under matched smells is not yet decided.** The most promising place to find it (Cell A1, predator passive in one quadrant) gave a striking number that is structurally uninterpretable due to corner-camping. Re-running at longer budget will not fix this — the metric needs to change.
- **The week's main deliverable for the next round is instrumentation, not data.** A geometry-agnostic per-tag distance metric was designed and merged: each entity instance in YAML now carries an optional `tag` string; the source code stays geometry-agnostic; WandB receives `MeanDistRabbit_<tag>` and `MeanDistPredator_<tag>` per instance. The Cell A1 ambiguity directly resolves once both predator and rabbit at the *same* location have separate metrics — if the agent treats them identically, those metrics agree (location-avoidance); if it treats them differently, they diverge (class-avoidance).
- **A code-quality lesson surfaced in passing.** The implementation tripped on a five-site wiring task: four sites used `dict.get('key')` direct extraction (safe), one site used a fixed-key-list pattern that silently drops new keys (unsafe). All unit tests passed even with the bug present; only an end-to-end run on the actual training entrypoint caught it. The takeaway, captured as a memory insight, is that future verifications must demand the literal command run, not a synthetic test script — and code should prefer direct-extraction patterns when adding new optional keys.
- **The original Cell A1 §5 row-2 failure-mode warning was load-bearing.** It was written speculatively at design time as something to watch out for; it is exactly what happened. Pre-registered failure-mode catalogs continue to earn their keep.

---

## 4. What's next (still pending decision)

1. **Round 2.5 — re-launch Cell C + Cell A1 at full 10-million-episode budget with per-tag metrics enabled.** This is the actual confirmatory round; the partial Round 2 was reconnaissance. The new per-tag metrics let us read class-vs-quadrant disambiguation directly off WandB — `MeanDistRabbit_TL` next to `MeanDistPredator_TL` is the smoking gun for Cell A1; Cell C just needs the longer horizon to know whether the sign-flip is a transient or a real reversal. *(experiment-designer to author the design doc; training-runner to launch once a node:GPU is identified.)*
2. **(Optional) Backfill Round 1 with per-tag metrics for cross-round comparability.** Adding `tag: TL`/`tag: BR` to the Round-1 baseline configs and re-running seeds 42/43 would give Round 2.5 an apples-to-apples comparator at the same metric resolution. Adds ~15-30 hours of training. *(experiment-designer to author; user decision pending — not blocking 2.5.)*
3. **(Optional) Refactor the `train.py` Site-1 fixed-key-list pattern to direct-extraction.** A small follow-on cleanup; eliminates the structural bug class behind the per-tag wiring incident. *(senior-developer to plan; low priority — the 2-line surgical fix is in place.)*
4. **(Optional) Tag the four `hiding_predators` instances if Cell A1 re-launch flags hiding-predator-class avoidance as a separate confound.** Currently they are unflagged; the per-tag plan deferred them because their YAML surface is different from `predators` and `neutrals`. If post-Round-2.5 analysis shows they are part of the puzzle, this becomes an issue. *(experiment-designer to flag if needed.)*

---

## 5. Links

### Design docs

- [`docs/experiments/active/hypervigilance/sameprop_existing_run_survey.md`](../active/hypervigilance/sameprop_existing_run_survey.md) — the post-hoc survey that opened the study.
- [`docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md) — formal Round 1 analysis (n=2 seeds at ≈ 7.2M episodes).
- [`docs/experiments/active/hypervigilance/sameprop_round2_design.md`](../active/hypervigilance/sameprop_round2_design.md) — Round 2 design (Cells C + A1) with the §9 truncated-data partial analysis appended.

### Anchor / supporting plans (under `docs/develop/active/hypervigilance/`)

- [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../develop/active/hypervigilance/sameprop_discriminating_channels.md) — Phase-1 channel-ranking memo: under matched smells, what cues *can* still distinguish predator from rabbit? Feeds Round 2's design choices.
- [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../develop/active/hypervigilance/per_entity_avoidance_logging.md) — the prior aggregated per-entity logging plan (commit `4b55fc6`); the precursor that Round 1 relog uses.
- [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) — the new tag-based per-instance distance plan (revised after the user rejected the originally-planned hard-coded quadrant approach). Status: implemented + verified; gates Round 2.5.

### Memory insights

- [`.claude-memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md`](../../../.claude-memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md) — Round 1 verdict + confound flag, captured at end of the analyzer chain.
- [`.claude-memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md`](../../../.claude-memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md) — channels memo's findings (movement signature dominant, visual ch.5/ch.7 at contact, extero-noc contact-only) and the `hunt_stamina_threshold > 1.0` structural-disable trick used in Cell A1.
- [`.claude-memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`](../../../.claude-memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md) — Round 2 partial verdict (this week): Cell C sign-flip, Cell A1 contamination by §5 row-2 failure mode.
- [`.claude-memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../../.claude-memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md) — design rationale for the per-tag metrics that supersede the originally-planned quadrant-hardcoded approach.
- [`.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`](../../../.claude-memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md) — the wiring-bug + verification-practice lesson encountered during the per-tag implementation.

### Working files

- (none for this study — the analyzer chain wrote directly into the design doc rather than via `tmp/` working files this week.)

### Diary days

- [`docs/diary/2026-05-07.md`](../../diary/2026-05-07.md) — Round 1 relog launches (seeds 42/43 on node 112), Round 2 design, Phase-1 channels memo. Three sessions in one day; the rabbit/predator thread is in `b12b1fea` and `35f1841`'s precursor work.
- [`docs/diary/2026-05-08.md`](../../diary/2026-05-08.md) — Round 1 relog completes, Round 2 launches, Round 2 SIGINT'd, partial-verdict analysis written, three insights captured (`b12b1fea`).
- [`docs/diary/2026-05-09.md`](../../diary/2026-05-09.md) — per-tag metrics revised plan + implementation + verification + bug-fix; three more insights captured at end-of-session.

### Implementation commits

- `4b55fc6` — `feat(hypervigilance): per-entity avoidance logging` — the Round-1-relog enabler.
- `42cc049` — `docs(hypervigilance): 📚 Round 2 truncated-data partial analysis` — §9 of the design doc.
- `40bcc1d` — `docs(hypervigilance): 📚 plan §7 per-quadrant + per-rabbit-instance logging` — the *original* (rejected) quadrant-hardcoded plan.
- `18bde6f` — `docs(hypervigilance): 📚 revise §7 plan to tag-based geometry-agnostic design` — the revised plan.
- `0a73613` — `feat(hypervigilance): ✨ per-tag per-instance distance logging (Round-2 §7)` — the implementation.
- `6d3d382` — `fix(hypervigilance): 🐛 wire dist_per_{neutral,predator} into RPPO info_np` — the Site 1 wiring bug fix.
- `e9d5745`, `8d75379` — verification appendices to the plan doc (RPPO and Dreamer T5 smokes).
- `76c5328` — `docs(memory): 📚 capture 3 insights` — this week's memory-capture commit.

---

## 6. Reading order if you have 10 minutes

1. **This summary** — start here (5 minutes).
2. **§9 of [`sameprop_round2_design.md`](../active/hypervigilance/sameprop_round2_design.md)** — the truncated-data partial analysis with the actual numbers per cell, the §5 row-2 failure-mode mapping that explains the Cell A1 caveat, and the explicit "what's needed to complete the round" recommendation (3 minutes).
3. **The Round 2 truncated-verdict insight, [`20260509_1532_sameprop_round2_truncated_verdict.md`](../../../.claude-memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md)** — the same verdict in 5-section form with rationale (2 minutes).

If you have 30 minutes, also read:

4. **The tag-based-design insight, [`20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../../.claude-memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md)** — why the new per-tag metric is the right disambiguator and what it costs.
5. **[`round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md)** — the full Round 1 numbers + per-seed agreement check + the confound the analyzer flagged in parallel.
