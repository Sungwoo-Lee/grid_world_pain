---
title: "SameProp Rabbit-Avoidance Study — Closing re-summary (Rounds 1–2.6, 2026-05-07 → 2026-05-21)"
study: sameprop_rabbit_avoidance_study
generated: 2026-05-21T15:46
window: "2026-05-07 → 2026-05-21"
status: snapshot
---

# SameProp Rabbit-Avoidance Study — Closing re-summary as of 2026-05-21 15:46 KST

> **This is the closing re-summary for the study.** The 2026-05-14 23:32 snapshot ([`20260514_2332_sameprop_rabbit_avoidance_study.md`](20260514_2332_sameprop_rabbit_avoidance_study.md)) carried the two-level verdict (spatial: class-blind; event: +37 pp bush-dive gap) but left **Round 2.6 (the seed-lock attempt) crashed and not yet replicated**. This version updates the headline: **Round 2.6 was successfully re-launched, completed cleanly, and replicated the event-level finding at a second seed.** The class-conditional active defence under matched smells is now seed-stable across two independent seeds. The corner-camping policy in the sister cell is NOT — it failed to replicate, so corner-camping is one basin out of at least two. Prior summaries remain as historical snapshots.

---

## Take-home messages — the whole study in 6 bullets

1. **The setup.** Normally the agent tells a dangerous patrolling predator from a harmless rabbit by **smell**. This study deliberately gives them the **same smell** ("sameProp") and asks: does the agent still behave differently around them, and how?

2. **Two answers, two layers — both still hold.** Where the agent *walks* on average is class-blind under matched smells (it doesn't keep predators farther away than rabbits). What the agent *does in the moments a creature enters its danger radius* is strongly class-discriminating: it dives into a bush **+37 percentage points more often** for a predator than a rabbit, and **stops eating** near a predator while eating normally near a rabbit. Same agent, opposite verdict — different layer of measurement.

3. **Now seed-locked.** The event-level finding (+37 pp bush-dive gap, opposite-direction eat suppression) has now been replicated at a second seed. Cross-seed agreement is exact-to-noise: 88.8% vs 87.6% bush-dive rate, +37.3 pp vs +36.8 pp class gap, 0.769 vs 0.748 eat-suppression. **Two independent seeds, same numbers.** This was Round 2.6's job and it succeeded.

4. **The sister-cell story is more nuanced.** The companion experiment that turned the predator into a harmless camp-target (Cell A1) **did not replicate its first-seed finding**. The first seed (Round 2.5, seed 43) learned to camp in a safe corner and live to the timer; the second seed (Round 2.6, seed 45) learned a different "avoid the dangerous corner but starve" policy. **Corner-camping is one solution out of at least two**, not a generic basin. The class-blindness conclusion still holds — both seeds are class-blind — but the headline "agent corner-camps" needs more seeds to characterise the policy distribution.

5. **The methodological lesson is now project canon.** Averages dissolve event-level signal. Whenever a study concludes "the agent doesn't discriminate," cross-check it with the behavior-measure toolkit (bush-dive rate, eat-under-threat ratio, defensive-motif clustering) before trusting the verdict. Round 2.5's spatial-only "no class discrimination" verdict was right as a spatial statement and wrong as a behavioural one — the toolkit recovered the signal mean-distance metrics had dissolved.

6. **Where the work goes next.** The study has produced a paper-grade two-seed result on the event-level class-conditional defence. Open decisions: (a) does the corner-camping basin distribution merit a 4-seed Round 2.7 follow-up?, (b) does **Round 3** (food in all four quadrants — closing the spatial-camping loophole) launch next with the behavior toolkit pre-registered as the primary verdict?, (c) does the project pivot from this thread to the modulated-agent comparison (NMN / FiLM sequel on Cell C) or to a different research line entirely? The PI-level call has not yet been made.

---

## 1. Study question

The grid world has three kinds of moving creatures:

- a **patrolling predator** (deals damage if it touches the agent),
- **four hiding-predators perched on rocks** (deal damage at a distance), and
- **harmless rabbits** (do nothing).

Normally each class carries its own **smell signature**, so the agent can tell them apart from far away by using its smell sensor. This study removes that cue: the **patrolling predator** and the **rabbits** are given the **same smell signature** ("sameProp"). The smell sensor can no longer separate them.

**The puzzle.** Does the agent still behave differently around the dangerous one? If yes, *how* — by sight at close range, by post-contact pain memory, by movement pattern, by something else? And — the question that emerged halfway through — does the answer depend on **what kind of "differently" you measure**?

**Why it matters for the broader project.** The wider research goal is to model **hypervigilance** — over-cautious avoidance of safe things, driven by pain or fear. To make that claim cleanly we needed to know what *baseline* discrimination looks like under matched smells. This study mapped that baseline and, in doing so, surfaced a methodological lesson that affects every future "the agent doesn't discriminate" verdict in this project.

---

## 2. Experiments completed this study

| # | Experiment | Plain-English question | What was changed | High-level finding (plain English) |
|---|---|---|---|---|
| **0** | **Existing-run survey** *(2026-05-07)* | Look at an already-running matched-smell training — does anything in the logs distinguish predator from rabbit? | Nothing changed; re-read existing logs. | The agent kept the predator about **half a cell farther** than rabbits on average. Suggestive but one seed; food and rabbits share quadrants — could be food-seeking, not class avoidance. |
| **1** | **Round 1 — clean re-run** *(2026-05-07 → 2026-05-08)* | Re-run cleanly with two seeds and per-creature logging. Does the half-cell gap survive? | Per-creature distance logging implemented; two seeds (42, 43). | **Pattern replicates at both seeds** (gap ≈ +0.63 cells). Upgraded the survey from "anecdote" to "real signal." Food/rabbit-share-quadrants confound now needs a controlled test. |
| **2** | **Round 2 — kill the confounds** *(2026-05-08, stopped early)* | Cell C: move food *away* from rabbit quadrants. Cell A1: turn off the predator's hunt-mode so only post-contact teaching can drive avoidance. | Cell C: food spawn-zones moved (decoupleFood). Cell A1: predator restricted to one quadrant and made passive (passivePredator). | Both runs SIGINT'd at ~4–5% of budget. Cell C's gap flipped sign (rabbits farther than predator). Cell A1's gap exploded — but the agent never visited the predator's quadrant. We were measuring **corner-camping**, not class avoidance. Need a sharper metric. |
| **3** | **Round 2.5 — full budget + per-tag distances** *(2026-05-09 → 2026-05-10)* | Same two cells, but now measure distance to the *specific* same-corner predator vs the *specific* same-corner rabbit, so corner-camping doesn't fake a result. | No experimental knob change. Full 10M-episode budget; one seed per cell (42 / 43). | **Spatial verdict: NO class discrimination.** Cell A1 corner-camps (per-tag gap essentially zero, 0.004 cells, agent survives to the timer). Cell C's gap inverts (rabbits farther than predator). The "agent treats predators specially in space" claim does not survive matched smells. |
| **3b** | **Round 2.5 appendix — new "what does it do" measures** *(2026-05-11)* | Read the *same* converged agents through a new toolkit: bush-dive rate, eating-while-threatened ratio, behavioural-motif clustering. Does the agent's defensive *behaviour* discriminate even though its *route* doesn't? | No new training — re-evaluated the Round 2.5 checkpoints with new metrics. 200 deterministic episodes per cell. | **Event-level verdict: YES, strongly, for Cell C.** Bush-dive rate when threat enters R=3 cells: predator **88%** vs rabbit **51%** → **+37 pp** gap. Eat-under-threat ratio: predator **0.75×** (eating suppressed) vs rabbit **1.19×** (eating elevated). Per-tag fan-out across the two rabbits is within noise → not a single-rabbit artifact. Cell A1 stays class-blind at every layer. |
| **4** | **Round 2.6 — seed-lock attempt (first try)** *(2026-05-12 17:01 launch, crashed 23:38)* | Repeat Cell C (new seed 44) and Cell A1 (new seed 45) with event-level toolkit logging live. | Seed 44 (Cell C) and seed 45 (Cell A1) on the same node (n106). Behavior-measure logging backfilled into the configs. | **Both runs CRASHED at ~6.6 h on n106**, reaching 1.18 M / 0.87 M episodes out of 10 M. Crashed within 3 minutes of each other → node-level event, not a training bug. Partial data was tracking the +37 pp gap (+28.5 pp and rising at 1.18 M ep). Seed-lock not yet earned. |
| **5** | **Round 2.6 — seed-lock re-launch** *(2026-05-16 → 2026-05-19, finished cleanly)* | Re-launch the two crashed runs on healthy nodes. Same configs, same seeds. | Cell C seed 44 on n101 cuda:0 (81.8 h to complete on the slower 2080 Ti); Cell A1 seed 45 on n102. | **Both runs finished cleanly to 10 M episodes.** Final eval-rollout numbers below. |
| **6** | **Round 2.6 closing analysis — primary verdict** *(2026-05-21)* | Apply the pre-registered confirmation criteria (bush-dive rate ≥ 80% near predator, gap ≥ +30 pp, eat-suppression < 0.80 near predator) to the 10 M checkpoint of the Cell C re-launch. | No new training — same offline eval-rollout pipeline (200 deterministic episodes, fresh seeds 1000–1199) used on the Round 2.5 §12 appendix. | **Cell C: confirmation criteria met by a wide margin.** Bush-dive rate near predator **0.888** (threshold ≥ 0.80, +11% over), class gap **+0.373** (threshold ≥ +0.30, +24% over), eat-suppression near predator **0.769** (threshold < 0.80). Seed-paired vs Round 2.5: all numbers agree to within within-seed noise (cross-seed M2 1.2 pp, Δ_M2 0.5 pp, M5 0.02). **Two-seed replication achieved.** Cell A1: corner-camping basin **refuted** — seed 45 survived only 98/500 steps (threshold ≥ 470), learning a starve-while-avoiding-the-dangerous-corner policy instead. The class-blindness conclusion holds; the corner-camping policy itself is one basin of at least two. |

---

## 3. Where this leaves the study

- **The headline is now seed-locked.** The event-level class-conditional active defence (+37 pp bush-dive gap, +0.44 eat-suppression delta) replicates at a second independent seed under the same matched-smells configuration. This is the paper-grade result of the study.

- **The two-level reading is unchanged.** Under matched smells the agent is **class-blind in where it walks** and **class-discriminating in how it defends itself**. Both statements are simultaneously true; the spatial and event-level metrics are talking about different things, and the spatial-only verdict missed the signal.

- **Corner-camping is not the only basin.** The companion experiment (Cell A1, predator made passive) had two seeds picked two different policies: seed 43 corner-camped at 486/500 survival, seed 45 starved at 98/500 survival while still avoiding the dangerous corner. **Both are class-blind at the event level** — neither distinguishes predator from rabbit when forced to engage them — so the *class-blindness* conclusion is robust. But the *specific policy* underneath it is seed-sensitive: there are at least two basins. Characterising the basin distribution would need 4+ more seeds (a hypothetical Round 2.7).

- **The methodological win is now load-bearing for the rest of the project.** Mean-distance metrics are insensitive to event-level discrimination whenever the spatial layout is geometrically constrained. Any future "no class discrimination" verdict gets cross-checked with the behavior-measure toolkit before being trusted. Round 2.5 is the worked example of why; Round 2.6 is the seed-stability proof that the toolkit's signal is real, not a Round 2.5 artifact.

- **One soft caveat to log.** The per-tag rabbit fan-out check on the eat-suppression measure (0.139) marginally exceeded the ±0.10 secondary-check band at seed 44. The primary verdict (eat-suppression near predator < 0.80) was not affected. This is consistent with the food-density asymmetry across quadrants and is flagged as a toolkit-v2 candidate (either widen the band to ±0.15 or split the measure by quadrant). It does not change the verdict.

- **Wider arc.** Hypervigilance work should anchor on the **event-level baseline** (+37 pp bush-dive gap, eat-suppression near predator), not the spatial baseline (no gap under matched smells). The latter has no signal to amplify; the former does. The natural next pain-modulation experiments are: does a pain-driven modulator widen the gap further (over-amplified defence) or narrow it (suppressed)? — and on Round 3, where the spatial-camping loophole is closed, does the gap survive?

---

## 4. What's next (still pending decision)

1. **PI-level call on the next research thread.** The study has produced a publishable two-seed event-level finding. The three candidate next moves are not mutually exclusive but compete for compute: (a) characterise the corner-camping basin distribution with a 4-seed Round 2.7 on Cell A1; (b) launch Round 3 (food in all four quadrants, behavior toolkit pre-registered as primary) to close the spatial-camping loophole and test whether the +37 pp gap survives; (c) pivot to the modulated-agent (NMN / FiLM) sequel on Cell C — does the modulated agent show a *larger* event-level class discrimination than plain RPPO? *(`pi` to surface; user decides.)*

2. **Round 3 — close the spatial-camping loophole.** Spawn food in all four quadrants so no corner is uniformly safe. Pre-register the event-level toolkit (M2 bush-dive rate / M5 eat-under-threat / M7 defensive-motif clustering) as the *primary* verdict, not a secondary check. Designer's prior: if the agent's class discrimination is genuine, the bush-dive gap widens further (88% → ~95%) and eat-suppression deepens (0.75 → ~0.55); if the gap was spatially mediated, it collapses. *(`experiment-designer` to author, blocked on item 1.)*

3. **(Optional) Round 2.7 — characterise the A1 basin distribution.** 4 seeds on Cell A1 (46, 47, 48, 49) to map how the agent's policy distribution splits across the corner-camping basin vs the starve-while-avoiding basin vs anything else that emerges. *(`experiment-designer`, optional; only launch if the basin distribution is research-relevant downstream.)*

4. **(Optional) NMN / FiLM agent sequel on Cell C.** Does the modulated agent show a larger event-level class discrimination than plain RPPO under the same matched-smells configuration? This is the natural bridge back to the NMN comparison thread. *(`experiment-designer` to flag in the next round design.)*

5. **Toolkit-v2 candidate — refine the M5 per-tag band.** The 0.139 marginal miss on the rabbit-tag fan-out check is a soft toolkit-design issue, not a verdict issue. Options: widen the band to ±0.15, or compute M5 per-quadrant against a quadrant-matched safe baseline. Not blocking. *(`experiment-designer` or whoever owns toolkit v2.)*

6. **Diary backfill — small.** The 2026-05-16 Round 2.6 re-launch never wrote a `training-start` row to the diary (the original n106 launch on 2026-05-12 used a different tag, and the relaunch tag was not separately registered), so the closing analysis couldn't update a row in place. The closing analysis was posted as a `note` instead. Not actionable for the study; flagged here for future re-launch hygiene. *(diary skill or top-level Claude on the next R2.6-like re-launch.)*

---

## 5. Links

### Design docs

- [`docs/experiments/active/hypervigilance/sameprop_existing_run_survey.md`](../active/hypervigilance/sameprop_existing_run_survey.md) — the post-hoc survey that opened the study.
- [`docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md`](../active/hypervigilance/round1_relog_baseline_analysis.md) — Round 1 analysis at two seeds.
- [`docs/experiments/active/hypervigilance/sameprop_round2_design.md`](../active/hypervigilance/sameprop_round2_design.md) — Round 2 design (Cells C + A1) with §9 truncated-data analysis.
- [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md) — Round 2.5 design with §§9–11 spatial verdict and the §12 event-level appendix. **The single document that carries the original two-level verdict.**
- [`docs/experiments/active/hypervigilance/sameprop_round26_design.md`](../active/hypervigilance/sameprop_round26_design.md) — Round 2.6 seed-lock design, now with §§9–12 filled in (the 2026-05-21 closing analysis). **The document that carries the seed-stability verdict.**
- [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) — the event-level toolkit's experimental design.

### Supporting plans (under `docs/develop/active/`)

- [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../develop/active/hypervigilance/sameprop_discriminating_channels.md) — Phase-1 memo: which cues *can* still separate predator from rabbit under matched smells?
- [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../develop/active/hypervigilance/per_entity_avoidance_logging.md) — aggregated per-entity distance logging (Round 1's enabler).
- [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) — per-tag distance logging (the metric that resolved Round 2.5 at the spatial level).
- [`docs/develop/active/behavior/behavior_measure_toolkit_v1_plan.md`](../../develop/active/behavior/behavior_measure_toolkit_v1_plan.md) — the event-level toolkit implementation plan.
- [`docs/project/ideas/20260510_behavior_measure_toolkit.md`](../../project/ideas/20260510_behavior_measure_toolkit.md) — biological grounding for the toolkit (predictive-coding / active-inference framing).

### Memory insights (hypervigilance folder unless noted)

- [`docs/memory/memories/hypervigilance/20260518_1736_sameprop_c_seed44_directional_replication.md`](../../memory/memories/hypervigilance/20260518_1736_sameprop_c_seed44_directional_replication.md) — **the Round 2.6 closing insight (now settled)**: H₁(C-event) confirmed at eval-time, two-seed-elevated.
- [`docs/memory/memories/hypervigilance/20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md`](../../memory/memories/hypervigilance/20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md) — **the Cell A1 seed-45 refutation** (corner-camping basin is NOT generic; survival 98/500).
- [`docs/memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md) — **the original event-level verdict** (+37 pp bush-dive gap, 0.75× / 1.19× eat-suppression, per-tag fan-out within noise) at seed 42.
- [`docs/memory/memories/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md`](../../memory/memories/hypervigilance/20260510_2237_sameprop_round25_no_class_avoidance.md) — **the spatial-level verdict** (Round 2.5 mean-distance verdict; complementary, not superseded).
- [`docs/memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`](../../memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md) — Round 2 partial verdict.
- [`docs/memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md`](../../memory/memories/hypervigilance/20260509_1533_tag_based_distance_supersedes_quadrant.md) — per-tag metric design rationale.
- [`docs/memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md`](../../memory/memories/hypervigilance/20260508_1444_sameprop_round1_finding_and_confound.md) — Round 1 verdict + confound flag.
- [`docs/memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md`](../../memory/memories/hypervigilance/20260508_1445_sameprop_discriminating_channels.md) — Phase-1 channels memo.
- [`docs/memory/memories/cluster_ops/20260518_1737_wandb_post_crash_frozen_state_misread.md`](../../memory/memories/cluster_ops/20260518_1737_wandb_post_crash_frozen_state_misread.md) — methodological lesson from the Round 2.6 crash: `run.summary` keys stay frozen post-crash; always check `run.state` first.
- [`docs/memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md`](../../memory/memories/subagent_engineering/20260509_1534_synthetic_smoke_masks_dict_assembly_bugs.md) — engineering lesson from the per-tag implementation week.

### Working files (representative — not exhaustive)

- `tmp/20260510_round25_cellC_last10.md`, `tmp/20260510_round25_cellA1_last10.md`, `tmp/20260510_round25_combined.md` — Round 2.5 spatial-analysis worksheets.
- `tmp/20260511_r25_appendix_*` — Round 2.5 §12 event-level appendix intermediates (scripts, JSON, CSVs, writeup notes).
- `tmp/20260521_r26_c_seed44_eval_analysis.py`, `tmp/20260521_r26_c_seed44_aggregates.json`, `tmp/20260521_r26_c_seed44_writeup.md`, plus the matching `_eval_rollout.log` / `_motif_cluster.log` / `_last10pct.md` / `_motifs.csv` / `_motif_by_class.csv` / `_motif_by_tag.csv` — Round 2.6 closing-analysis intermediates.

### Eval-rollout outputs

- `results/eval/models/10000022/` — Round 2.5 Cell C eval rollout (seed 42, source run `bdnfc0lu`).
- `results/eval/models/10000003/` — Round 2.5 Cell A1 eval rollout (seed 43, source run `nm8gn7y2`).
- `results/eval/models/10000021/` — **Round 2.6 Cell C eval rollout (seed 44, source run `ja5fu5k3`)** — the seed-lock evidence base.

### Diary days

- [`docs/diary/2026-05-07.md`](../../diary/2026-05-07.md) — Round 1 relog launches, Round 2 design, Phase-1 channels memo.
- [`docs/diary/2026-05-08.md`](../../diary/2026-05-08.md) — Round 1 relog completes, Round 2 launches + SIGINT'd, partial-verdict analysis.
- [`docs/diary/2026-05-09.md`](../../diary/2026-05-09.md) — per-tag metrics revised plan + implementation + verification; Round 2.5 launch.
- [`docs/diary/2026-05-10.md`](../../diary/2026-05-10.md) — Round 2.5 completes; §§9–11 spatial verdict.
- [`docs/diary/2026-05-11.md`](../../diary/2026-05-11.md) — behavior-measure toolkit shipped; §12 appendix written.
- [`docs/diary/2026-05-12.md`](../../diary/2026-05-12.md) — event-level verdict insight captured; Round 2.6 designed + launched (which then crashed).
- [`docs/diary/2026-05-13.md`](../../diary/2026-05-13.md) — Round 2.6 progress check (mis-read as in-flight); prior re-summary.
- [`docs/diary/2026-05-14.md`](../../diary/2026-05-14.md) — Round 2.6 crash discovered; reader-friendly re-summary.
- [`docs/diary/2026-05-16.md`](../../diary/2026-05-16.md) — Round 2.6 re-launch on n101 / n102.
- [`docs/diary/2026-05-18.md`](../../diary/2026-05-18.md) — Cell A1 seed-45 finished (refuted at 98/500); Cell C seed-44 directional-replication insight at 62.5%.
- [`docs/diary/2026-05-19.md`](../../diary/2026-05-19.md) — Cell C seed-44 finished cleanly on n101 (10 M ep, 81.8 h).
- [`docs/diary/2026-05-21.md`](../../diary/2026-05-21.md) — Cell C closing analysis (H₁(C-event) confirmed two-seed-elevated); this re-summary.

### Prior summaries (this study)

- [`docs/experiments/summaries/20260514_2332_sameprop_rabbit_avoidance_study.md`](20260514_2332_sameprop_rabbit_avoidance_study.md) — 2026-05-14 snapshot: reader-friendly version of the two-level verdict with Round 2.6 crash correction. Predecessor of this re-summary.
- [`docs/experiments/summaries/20260513_1420_sameprop_rabbit_avoidance_study.md`](20260513_1420_sameprop_rabbit_avoidance_study.md) — 2026-05-13 snapshot: first two-level re-summary (mis-reported Round 2.6 as in-flight; corrected by the 2026-05-14 version).
- [`docs/experiments/summaries/20260510_2253_sameprop_rabbit_avoidance_study.md`](20260510_2253_sameprop_rabbit_avoidance_study.md) — 2026-05-10 snapshot: spatial-level-only verdict.
- [`docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md`](20260509_1552_sameprop_rabbit_avoidance_study.md) — 2026-05-09 snapshot: pre-Round-2.5.

### Implementation commits (load-bearing)

- `4b55fc6` — per-entity avoidance logging (Round 1 relog enabler).
- `0a73613` — per-tag per-instance distance logging (the spatial-level resolver).
- `82ae039` — Round 2.5 analysis (spatial-level verdict commit).
- `ed5cff3` — Round 2.5 §12 toolkit appendix (event-level verdict commit).
- `c110a2c` — event-level verdict captured as memory insight.
- `83ab27c` — Round 2.6 design.
- `c81b48d` — behavior_measures: block backfill into R2.6 configs.
- `42b4449` — Round 2.6 re-launch + results-check memory captures (Cell A1 refuted, Cell C directional replication, WandB-state lesson).
- `074ed57` — **Round 2.6 Cell C closing analysis** (H₁(C-event) confirmed two-seed-elevated; design doc §§9-12 filled in; closing memory update).

---

## 6. Reading order if you have 10 minutes

1. **This summary, sections 1–4** (5 min) — the take-home messages, the experiment table including the closing Round 2.6 row, the two-level verdict, and the open decision points.
2. **§§9–12 of [`sameprop_round26_design.md`](../active/hypervigilance/sameprop_round26_design.md)** (4 min) — the closing-analysis fill-in: the four key numbers vs their thresholds, the seed-paired cross-round agreement table, the toolkit appendix in the Round 2.5 §12 format.
3. **The Cell C closing insight, [`20260518_1736_sameprop_c_seed44_directional_replication.md`](../../memory/memories/hypervigilance/20260518_1736_sameprop_c_seed44_directional_replication.md)** (1 min) — the verdict in 5-section memory form with the original directional-replication record and the 2026-05-21 closing update.

If you have 30 minutes, also read:

4. **§12 of [`sameprop_round25_design.md`](../active/hypervigilance/sameprop_round25_design.md)** — the original event-level appendix that the Round 2.6 work replicated.
5. **The Round 2.5 event-level verdict insight, [`20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md)** — the seed-42 paper-grade memory form.
6. **The Cell A1 refutation insight, [`20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md`](../../memory/memories/hypervigilance/20260518_1735_sameprop_a1_seed45_corner_camping_refuted.md)** — the corner-camping-is-not-generic finding.
7. **The methodological lesson, [`20260518_1737_wandb_post_crash_frozen_state_misread.md`](../../memory/memories/cluster_ops/20260518_1737_wandb_post_crash_frozen_state_misread.md)** — why post-crash WandB summary reads are dangerous, and the corrected check pattern.
8. **The prior re-summary at [`20260514_2332_sameprop_rabbit_avoidance_study.md`](20260514_2332_sameprop_rabbit_avoidance_study.md)** — useful for seeing the two-level verdict before the seed-lock landed.
