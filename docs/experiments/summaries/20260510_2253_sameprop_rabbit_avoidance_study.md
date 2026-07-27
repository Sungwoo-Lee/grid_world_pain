---
title: "SameProp Rabbit-Avoidance Study — Round 2.5 verdict (re-summary, week of 2026-05-07 → 2026-05-10)"
study: sameprop_rabbit_avoidance_study
generated: 2026-05-10T22:53
window: "2026-05-07 → 2026-05-10"
status: snapshot
---

# SameProp Rabbit-Avoidance Study — Re-summary as of 2026-05-10 22:53 KST

> **One-paragraph summary.** This study tested whether the agent **learns to treat predators and neutral rabbits differently when their olfactory signatures are made identical** ("sameProp"). It started from a post-hoc survey that noticed the agent appeared to keep predators about 0.6 cells farther than rabbits at convergence, and a Round-1 re-run with proper per-entity logging at two seeds confirmed that gap. Round 2 attempted to control the two confounds suggested by the survey-stage analyzer — food/quadrant overlap (Cell C) and the predator's distinctive movement pattern (Cell A1) — but was stopped early at ~4-5% of its planned training budget; its partial results pointed two ways at once and the Cell A1 signal looked structurally contaminated. The week's main deliverable was a new geometry-agnostic per-tag distance metric so a re-launched Round 2.5 could resolve the contamination directly. **Round 2.5 has now run cleanly to the full 10-million-episode budget on n106 (single seed per cell) and has resolved the study.** The headline finding inverts the original survey: under matched olfactory smells, the agent does **not** learn to recognise predator class. Cell A1's per-tag distance to the same-corner predator and the same-corner rabbit is statistically indistinguishable (the agent simply camps the safe corner, surviving 486 of 500 steps); Cell C's gap reverses sign relative to Round 1 once food and rabbits are decoupled. The original survey's 0.6-cell gap decomposes into two confounds working in concert — food-quadrant overlap and spatial-avoidance camouflage — neither of which is class-conditional avoidance. One follow-up remains pending: a single second seed of Cell C (Round 2.6) to confirm the inverted finding is seed-stable.

> **This is a re-summary.** The prior snapshot at `20260509_1552_sameprop_rabbit_avoidance_study.md` was written before Round 2.5 finished and reports it as "running"; that file stays as a historical snapshot of what was known on 2026-05-09 15:52. Read this file for the verdict; read the older one for the mid-study state.

> **This is a snapshot.** Re-summaries should be written as new dated files in this folder, not by editing this one.

---

## 1. Study question

The environment has three classes of moving entities the agent encounters: a **patrolling predator** (damages on contact), four **hiding-predators on rocks** (damage at distance), and **neutral rabbits** (harmless). Normally the agent can tell predators from rabbits by their **olfactory `properties` vector** — a per-class one-hot the smell sensor returns when an entity is nearby. The "sameProp" condition deliberately collides those vectors: the patrolling predator and the neutral rabbits are given **the same olfactory `properties`**, so the agent cannot use smell alone to distinguish them.

**The puzzle**: does the agent still avoid predators preferentially? If yes, what cue is it using — visual class identity at contact, post-contact pain teaching, the predator's distinctive movement pattern, or just "the predator's quadrant happens to be unfriendly"?

**The dependent variables** are per-class distances and contact counts — `MeanDistRabbit` vs `MeanDistPredator` (cells, L2), `RabbitHits` vs `PredatorHits` (count per episode), and survival steps. A genuinely class-avoiding agent should keep predators *systematically* farther than rabbits — not just because of where they spawn, but because it has learned to recognise them.

**Why this matters for the wider project**: the project's larger thesis ties rabbit-avoidance behaviour to a "hypervigilance" signature — over-avoidance of safe entities under uncertainty. To talk about hypervigilance honestly, we first need to know what *baseline* avoidance looks like under matched olfactory channels. The arc of this study is that baseline measurement.

---

## 2. Experiments completed this week

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |
|---|---|---|---|---|---|
| **0** *(post-hoc, single seed)* | **SameProp existing-run survey** *(2026-05-07)* | Does the agent's behaviour, under matched olfactory smells, still distinguish predators from rabbits in any measurable way? | Nothing — analyzed an already-running training (n=1) using survival + per-class distance metrics that were already in the WandB log. | The agent kept the patrolling predator about **0.6 cells farther** than rabbits at convergence, with **3× more rabbit-contacts than predator-contacts** per episode. The *direction* of the asymmetry was clear and consistent. | Established the central observation: matched smells alone do not erase class-conditional behaviour. Motivated a clean re-run with two seeds and proper per-entity logging. The analyzer also flagged a confound: food and rabbits share the same two quadrants, while the predator roams the full grid — so the gap might be food-seeking spillover, not class avoidance. |
| **1** | **Round 1 relog** *(2026-05-07 evening → 2026-05-08 noon)* | Does the same pattern hold when we re-run cleanly with two seeds and **per-entity logging** added to the codebase? | Per-entity logging implementation merged. Two seeds (42, 43) of the same sameProp config relaunched on node 112. | **Pattern confirmed at both seeds.** At ~7.2 million episodes: gap ≈ +0.63 cells; rabbit-contacts ≈ 6.5/ep, predator-contacts ≈ 3.4/ep, gap ≈ +3.1/ep. The two seeds agreed within ±0.03 cells across all distance metrics. | Promoted the survey from "n=1 motivation" to a replicated finding. The **food/quadrant confound** flagged at the survey stage was now formally on the docket: nothing in this experiment *resolved* it, so the next round had to attack it directly. |
| **2** | **Round 2 — confound control + movement-signature ablation** *(2026-05-08 afternoon, n=1 each cell, partial budget)* | Two cells in parallel. **Cell C (decoupleFood)**: with the food and rabbits no longer sharing quadrants, does the gap survive? **Cell A1 (passivePredator)**: with the predator's hunt mode disabled and its patrol zone shrunk to one corner, does the gap survive on post-contact teaching alone? | Cell C: food spawn-zones moved to the two quadrants without rabbits. Cell A1: predator restricted to top-left quadrant, hunt-stamina threshold raised so the predator never engages hunt mode. | **Both cells were stopped at ~3.5 hours / ~0.4 million episodes** — about 4–5% of the 10-million-episode plan. Verdicts provisional. **Cell C**: gap **flipped sign** (rabbits ~0.4 cells farther than predators) — suggested food/quadrant overlap was load-bearing. **Cell A1**: gap **grew to ~3.9 cells** — six times the original effect — but the agent never visited the predator's quadrant, so the metric was measuring corner-camping not class avoidance. | Two updates. **Cell C**: original survey finding probably overstated. **Cell A1**: re-launching at full budget would not resolve the ambiguity — the metric itself was the bottleneck. Pushed the next chain step to be **a tooling change, not another experiment**: ship a per-tag distance metric so the same-corner predator and same-corner rabbit get separate, comparable distance numbers. |
| **3** | **Round 2.5 — full-budget re-launch with per-tag metrics** *(2026-05-09 evening → 2026-05-10 evening, single seed per cell, 10 M episodes each)* | Same two cells, same configs, but now read off the new per-tag distance keys: same-corner predator vs same-corner rabbit get separate metrics, so corner-camping (Cell A1's contamination) and food-spillover (Cell C's confound) become independently testable. | No experimental knob change. Single seed per cell. Same two configs Round 2 used (now carrying YAML `tag` fields per the per-tag metric work). 10-million-episode budget reached cleanly on both runs; pre-registered analysis window 9-10 M episodes. | **Both pre-registered hypotheses refuted.** **Cell A1**: distance to the same-corner predator vs same-corner rabbit is *indistinguishable* — gap = +0.004 cells at 486/500 survival, exactly the corner-camping signature the failure-mode catalog anticipated. The agent does not recognise predators; it just avoids the dangerous corner. **Cell C**: gap = −0.53 cells (rabbits *farther* than predator), with both rabbit corners avoided — Round 1's apparent class avoidance was substantially food-coupling, and the agent under decoupling went past neutral into bilateral rabbit avoidance. | **The study resolves to "no genuine class-conditional avoidance under sameProp."** The original survey's 0.6-cell gap decomposes into two confounds in concert: food-quadrant overlap and spatial-avoidance camouflage. Neither one is class recognition. The per-tag metric earned its keep — without it, Cell A1's *aggregated* gap of +3.86 cells (~6× R1, ~10× threshold) would have been published as "class avoidance dramatically confirmed." |

---

## 3. Where this leaves the study

- **The original "rabbits closer than predators under sameProp" finding does not survive.** Round 2.5's two cells, read together, decompose the survey's 0.6-cell gap into two confounds. Neither cell shows a residual class-recognition signal once its targeted confound is controlled. The headline answer to the study question is therefore **no — under matched olfactory smells, the agent does not learn to recognise the predator class**.
- **Cell A1's verdict is rock-solid.** The same-corner-predator-vs-rabbit comparison is 75× past the corner-camping threshold, three orders of magnitude tighter than the band that would have been called ambiguous, and stable across the last three of ten temporal sub-windows. There is no residual ambiguity to re-test.
- **Cell C's verdict is provisional pending one more seed.** The sign-flip is at 1.6–1.8× the threshold for confirmation — meaningful, but just shy of the 2× single-seed elevation bar the study uses. One Round-2.6 seed will firm it up; the configuration is already known to work and the sign is already inverted past threshold, so the second seed is a confirmation, not a discovery.
- **Methodological win**: the per-tag per-instance distance metric the project shipped this week was directly and decisively load-bearing. Cell A1's *aggregated* numbers looked like a 10× confirmation of class avoidance; the per-tag pair flipped it. Pre-registered failure-mode catalogs continue to earn their keep — the §5 row that warned "agent never visits TL → MeanDist looks high purely by spatial separation" was written speculatively at design time and fired exactly as written.
- **A code-quality side lesson surfaced.** The metric implementation tripped on a five-site wiring task: four sites used direct-extraction (safe), one used a fixed-key-list pattern that silently dropped new keys (unsafe). All unit tests passed even with the bug present; only an end-to-end run on the actual training entrypoint caught it. Captured separately as a memory insight; the takeaway is that future verifications must demand the literal command run, not a synthetic test script.
- **The wider hypervigilance arc has its baseline now**, and the baseline is sobering: the simple sameProp setup does not produce class-conditional avoidance. Future experiments that want to study "over-avoidance under uncertainty" cannot start from "the agent already avoids the dangerous class and we will perturb it" — that premise is false at this scale. They will have to either pick a setup where class avoidance does emerge, or study spatial-avoidance dynamics directly.

---

## 4. What's next (still pending decision)

1. **Round 2.6 — Cell C with seed 44 only.** The Cell-C sign-flip verdict is at 1.6–1.8× threshold; one more seed locks it in. Same configuration, same node 106, ~22 hours. Cell A1 does not need a second seed — its verdict is 75× past threshold. *(experiment-designer to author a one-cell design doc; training-runner to launch on next n106 rotation.)*
2. **Re-summary after Round 2.6.** When seed 44 lands, write a third snapshot in `summaries/` that closes the study. The current re-summary is the verdict snapshot; the post-2.6 summary is the closing snapshot. *(top-level Claude / `summarize-study` skill.)*
3. **Round 3 — close the spatial-avoidance loophole.** The cleanest follow-up is a sameProp configuration with food spawned in *all four* quadrants, so no quadrant is uniformly safe and the agent cannot trivially camp. Any future channel-attribution work for the wider hypervigilance arc should run on this baseline, not on Round-1's geometry. *(experiment-designer to author; not blocking 2.6.)*
4. **Generalise to NMN / FiLM agents.** Round 2.5 used plain RPPO. Whether a modulated agent under the same Cell A1 setup also corner-camps, or whether the modulator changes the policy's relationship to the safe corner, is a natural sequel that ties this study back to the project's NMN comparison thread. *(experiment-designer to flag in the Round-3 design.)*
5. **Refactor the `train.py` Site-1 fixed-key-list pattern.** Eliminates the structural bug class behind the per-tag wiring incident. Optional / low priority — the surgical fix is in place. *(senior-developer to plan if pursued.)*

---

## 5. Links

### Design docs

- [`docs/experiments/active/hypervigilance/sameprop_existing_run_survey.md`](../active/hypervigilance/sameprop_existing_run_survey.md) — the post-hoc survey that opened the study.
- [`docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md) — formal Round 1 analysis (n=2 seeds at ≈ 7.2M episodes).
- [`docs/experiments/active/hypervigilance/sameprop_round2_design.md`](../active/hypervigilance/sameprop_round2_design.md) — Round 2 design (Cells C + A1) with the §9 truncated-data partial analysis appended.
- [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md) — Round 2.5 design with §§9–11 results / analysis / conclusions filled in. **The verdict doc.**

### Anchor / supporting plans (under `docs/develop/active/hypervigilance/`)

- [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../develop/active/hypervigilance/sameprop_discriminating_channels.md) — Phase-1 channel-ranking memo: under matched smells, what cues *can* still distinguish predator from rabbit? Feeds Round 2's design choices.
- [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../develop/active/hypervigilance/per_entity_avoidance_logging.md) — the prior aggregated per-entity logging plan; the precursor that Round 1 relog uses.
- [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) — the new tag-based per-instance distance plan. Status: implemented + verified; the metric that resolved Round 2.5 Cell A1.

### Memory insights

- [`docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md`](../../../docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md) — **the verdict insight** for this re-summary. Round 2.5's two-cell decomposition; provisional pending Round 2.6 seed 44.
- [`docs/llm_wiki/entries/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`](../../../docs/llm_wiki/entries/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md) — Round 2 partial verdict (the truncated run that pointed at this verdict at ~5% of the budget).
- [`docs/llm_wiki/entries/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../../docs/llm_wiki/entries/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md) — the design rationale for the per-tag metric that Round 2.5 empirically validated.
- [`docs/llm_wiki/entries/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md`](../../../docs/llm_wiki/entries/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md) — Round 1 verdict + confound flag.
- [`docs/llm_wiki/entries/hypervigilance/20260508_1445_sameprop_discriminating_channels.md`](../../../docs/llm_wiki/entries/hypervigilance/20260508_1445_sameprop_discriminating_channels.md) — channels memo (movement signature dominant, visual ch.5/ch.7 at contact, extero-noc contact-only) and the structural-disable trick used in Cell A1.
- [`docs/llm_wiki/entries/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`](../../../docs/llm_wiki/entries/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md) — the wiring-bug + verification-practice lesson encountered during the per-tag implementation.

### Working files

- `tmp/20260510_round25_cellC_last10.md` — Cell C last-10% window analyzer worksheet.
- `tmp/20260510_round25_cellA1_last10.md` — Cell A1 last-10% window analyzer worksheet.
- `tmp/20260510_round25_combined.md` — combined cross-cell + cross-round contrast notes.

### Diary days

- [`docs/diary/2026-05-07.md`](../../diary/2026-05-07.md) — Round 1 relog launches, Round 2 design, Phase-1 channels memo.
- [`docs/diary/2026-05-08.md`](../../diary/2026-05-08.md) — Round 1 relog completes, Round 2 launches + SIGINT'd, partial-verdict analysis written, three insights captured.
- [`docs/diary/2026-05-09.md`](../../diary/2026-05-09.md) — per-tag metrics revised plan + implementation + verification + bug-fix; three insights captured; Round 2.5 design + launch.
- [`docs/diary/2026-05-10.md`](../../diary/2026-05-10.md) — Round 2.5 completes; analyzer fills in §§9–11; verdict insight captured.

### Prior summary (this study)

- [`docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md`](20260509_1552_sameprop_rabbit_avoidance_study.md) — the snapshot at 2026-05-09 15:52, written before Round 2.5 finished. Reports R2.5 as "running"; superseded by this re-summary for current state, retained as a historical snapshot.

### Implementation commits

- `4b55fc6` — `feat(hypervigilance): per-entity avoidance logging` — the Round-1-relog enabler.
- `42cc049` — `docs(hypervigilance): Round 2 truncated-data partial analysis` — §9 of the Round-2 design doc.
- `40bcc1d` — `docs(hypervigilance): plan §7 per-quadrant + per-rabbit-instance logging` — the *original* (rejected) quadrant-hardcoded plan.
- `18bde6f` — `docs(hypervigilance): revise §7 plan to tag-based geometry-agnostic design` — the revised plan.
- `0a73613` — `feat(hypervigilance): per-tag per-instance distance logging (Round-2 §7)` — the implementation.
- `6d3d382` — `fix(hypervigilance): wire dist_per_{neutral,predator} into RPPO info_np` — the Site 1 wiring bug fix.
- `e9d5745`, `8d75379` — verification appendices to the per-tag plan doc (RPPO and Dreamer T5 smokes).
- `76c5328` — `docs(memory): capture 3 insights` — the per-tag metrics-week memory commit.
- `24dd4a3` — `docs(hypervigilance): rewrite per-tag plan Context to plain-English entry` — doc-framing rule applied retroactively.
- `ac479e4` — `docs(hypervigilance): Round 2.5 design — 10M-ep relaunch with per-tag metrics`.
- `82ae039` — `docs(hypervigilance): Round 2.5 analysis — H₀(A1) confirmed, Cell C inverted (provisional)` — the verdict commit.
- `ba0d766` — `docs(memory): capture 1 insight — R2.5 sameProp class avoidance refuted` — the verdict-insight memory commit.

---

## 6. Reading order if you have 10 minutes

1. **This summary** — start here (5 minutes).
2. **§§9–11 of [`sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md)** — the actual numbers per cell, the temporal-evolution table, the cross-round contrast against Round 1, the §5 failure-mode catalog mapping, and the Round-2.6 escalation policy (3 minutes).
3. **The Round 2.5 verdict insight, [`20260510_2237_sameprop_round25_no_class_avoidance.md`](../../../docs/llm_wiki/entries/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md)** — the same verdict in 5-section form with rationale, the two-confound decomposition argument, and the methodology takeaway about pre-registered failure-mode catalogs (2 minutes).

If you have 30 minutes, also read:

4. **The per-tag metric design rationale, [`20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../../docs/llm_wiki/entries/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md)** — why the metric was the right disambiguator and what it cost. Useful context for understanding why Round 2.5's Cell A1 is interpretable when Round 2's Cell A1 was not.
5. **[`round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md)** — the full Round 1 numbers + per-seed agreement check + the food/quadrant confound the analyzer flagged at the survey stage, which Round 2 + 2.5 then attacked directly.
6. **The prior summary, [`20260509_1552_sameprop_rabbit_avoidance_study.md`](20260509_1552_sameprop_rabbit_avoidance_study.md)** — the snapshot before Round 2.5 finished. Useful for seeing how the predicted outcomes (designer's pre-registered priors) compared to what actually landed.
