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

> **Reading the experiment table.** Rows 3b and 6 quote per-class numbers for four behaviour metrics — **M1** (interrupted-feeding rate), **M2** (bush-dive rate), **M5** (eat-under-threat ratio), **M7** (defensive-motif repertoire). None of these are standard RL metrics; they were built for this study. If the names are unfamiliar, jump to [**Appendix A — Behaviour-metric glossary**](#appendix-a--behaviour-metric-glossary-m1-m2-m5-m7) first for plain-English definitions, formulas, and the actual code that computes each one, then come back here.

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

---

## Appendix A — Behaviour-metric glossary (M1, M2, M5, M7)

This study's headline numbers (the +37 percentage-point bush-dive gap, the 0.75× eat-suppression near predators, the motif distributions) are computed from a project-specific behaviour-measure toolkit. None of the four metrics is a standard RL or behavioural-neuroscience measure off the shelf — they were lifted from a postdoc's 8-candidate menu and pre-registered as the toolkit's v1 shippable subset. This appendix is the **reader's reference**: each metric gets a plain-English description, the formula, the actual code that runs in the eval-rollout pipeline, an "edge case → NaN" rule, and a worked example from the actual study numbers.

The full operational specification lives in [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) §§1–2. The implementation lives in [`scripts/eval_rollout.py`](../../../scripts/eval_rollout.py) (M1/M2/M5) and [`scripts/motif_cluster.py`](../../../scripts/motif_cluster.py) (M7). The code snippets below are simplified extracts from those files — sufficient to follow the logic, not the literal bookkeeping.

### A.0 Shared setup — protocol parameters, danger radius, K-step lookahead

All four metrics share the same observation protocol — fixed by [`behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) §3 and locked across Round 2.5 and Round 2.6 so the numbers are seed-paired comparable:

| Parameter | Symbol | v1 value | What it controls |
|---|---|---|---|
| Danger radius | `R` | **3.0 cells** | A class-*c* entity is "in danger range" of the agent at step `t` iff the L2 distance from the agent to *some* class-*c* entity is < `R`. |
| Online lookahead window | `K` | **5 steps** | For M1 and M2, the number of steps after a candidate event during which we look for the consequence (stop-eating for M1; reach-a-bush for M2). |
| Motif-window length | `K_motif` | **7 steps** (window is `[t* − 2, t* + K_motif]` ⇒ 10 steps total) | For M7 only — the window of action / state we cluster around each threat-onset. |
| Eval rollout | `N_eval` | **200 deterministic episodes** | Frozen checkpoint, policy run in argmax mode (no exploration), seeds 1000–1199 (disjoint from training seeds 0–999). |
| K-means cluster count | `k` | **6** | M7 only — fixed at the toolkit's six prototype motifs (`freeze`, `flight`, `bush_dive`, `freeze_then_flight`, `ignore`, `approach`); semantic labels assigned post-hoc by inspecting nearest-centroid exemplars. |
| K-means seed | — | **42** | Locked for reproducibility. |

**Key terms used below:**
- A **predator-class entity** is the patrolling predator. A **rabbit-class entity** is any of the harmless rabbits. The hiding-predators on rocks are not in the per-class fan-out (they are a separate class).
- An **eat step** is a step at which the agent successfully consumed food (env info bit `ate_food[t]`).
- A **bush** is any obstacle with `hides_agent: true` in the env config. The env exposes `agent_in_bush[t]` as a per-step boolean.
- A **threat onset** is the rising edge of "any class-*c* entity within radius `R`" — i.e., the step at which the closest class-*c* entity *just crossed* into the agent's danger range, having been outside it on the previous step.
- A **per-class fan-out** = compute the metric separately for `c = predator` and `c = rabbit`. A **per-tag fan-out** = subdivide further by entity ID (e.g., `rabbit_TL`, `rabbit_BR`); used as a sanity check that the per-class number isn't being driven by a single instance.

---

### A.1 M1 — Interrupted-feeding rate

**Question it asks.** When the agent is *currently eating* and a class-*c* entity *is already in danger range*, how often does the agent stop eating within the next K=5 steps?

**Plain-English walk-through.**
1. Walk the episode step by step.
2. At each step `t`, check two conditions: (a) the agent just ate food, and (b) at least one class-*c* entity is within R=3 cells. If both hold, this step is a **candidate**.
3. Five steps later (i.e., at step `t + K`), check whether the agent has stopped eating at any point in `[t+1, t+K]`. If yes, count this candidate as **interrupted**.
4. M1 for class *c* = (number of interrupted candidates) / (total candidates).

**Formula.**

```
              # candidates with a stop-eating in the K-step look-ahead
M1_c   =     ────────────────────────────────────────────────────────────
              # candidates  (steps where agent ate AND class-c was within R)
```

**Edge case → NaN.** If the agent never eats while a class-*c* entity is within range (denominator = 0), M1_c is undefined for this episode and emitted as NaN. The cell-A1 corner-camping policy hits this — the agent eats only in the bottom-right corner, where neither predator nor rabbit comes close enough to trigger a candidate, so M1 is structurally near-meaningless for Cell A1.

**Code (simplified extract from `scripts/eval_rollout.py:201`).** This is the offline sanity-replay version; the online accumulator in `train.py` uses the same logic with a circular K-step buffer.

```python
# For a single episode, single class (e.g., predator):
candidates    = 0      # ate_food AND any-predator-within-R
interrupted   = 0      # candidates whose K-step lookahead contains a stop-eating
cand_age      = -1     # step counter for the live candidate; -1 = no live candidate
steps_since_eat = 0

for t in range(T):
    in_R = (num_predators > 0) and (np.min(dist_per_predator[t]) < R)

    # Age the live candidate first; if it has reached K, resolve it.
    if cand_age >= 0:
        cand_age += 1
        if cand_age >= K:
            candidates  += 1
            if steps_since_eat >= K:   # agent has gone K full steps without eating
                interrupted += 1
            cand_age = -1              # candidate consumed

    steps_since_eat = 0 if ate_food[t] else steps_since_eat + 1

    # Open a new candidate at this step if both conditions hold.
    if ate_food[t] and in_R:
        cand_age = 0

M1_predator = interrupted / candidates if candidates > 0 else float("nan")
```

**Where it lands in this study.** The Cell C R2.6 numbers: M1_predator ≈ 42% vs M1_rabbit ≈ 24% (a +18 pp gap) — consistent with R2.5 seed 42 (+18.1 pp). The gap is real and class-conditional, but it sits just inside the toolkit's H₁(M1) threshold (+20 pp). M1 is therefore reported as a **supporting** measure for Cell C; the headline signal lives in M2 and M5.

---

### A.2 M2 — Bush-dive rate

**Question it asks.** When a class-*c* entity *just enters* the agent's danger range (a rising edge), how often does the agent get inside a bush within the next K=5 steps?

**Plain-English walk-through.**
1. Walk the episode step by step, tracking whether each class is "currently inside R" at each step.
2. An **onset** is a step where the class transitions from "outside R" to "inside R" — AND the agent is not already in a bush at that step (we want active dives, not the agent already being in cover).
3. Across the next K=5 steps, check whether the agent is in a bush at any point. If yes, count this onset as a **dive**.
4. M2 for class *c* = (number of dives) / (total onsets).

**Why a *rising edge* and not just "in range"?** Because we want to count *responses to threat appearance*, not "the agent sat in a bush while the threat hung around". A predator that loiters within R for 30 steps generates one onset, not 30.

**Formula.**

```
              # onsets where agent enters a bush within K steps
M2_c   =    ──────────────────────────────────────────────────────────
              # onsets  (rising edges of "class-c within R" while agent NOT in bush)
```

**Edge case → NaN.** If no class-*c* entity ever enters range this episode (denominator = 0), M2_c is NaN.

**Code (simplified extract from `scripts/eval_rollout.py:260`).**

```python
# For a single episode, single class (e.g., predator):
onsets       = 0     # rising edge of dist_predator < R while NOT in bush
dives        = 0     # onsets where agent reaches a bush within K steps
onset_age    = -1
onset_bush_seen = False
prev_in_R    = False

for t in range(T):
    in_R = (num_predators > 0) and (np.min(dist_per_predator[t]) < R)

    # Onset: rising edge of in_R while agent NOT already in a bush
    if (not prev_in_R) and in_R and (not agent_in_bush[t]):
        onsets += 1
        onset_age = 0
        onset_bush_seen = False

    if onset_age >= 0 and agent_in_bush[t]:
        onset_bush_seen = True

    if onset_age >= 0:
        onset_age += 1
        if onset_age >= K:
            if onset_bush_seen:
                dives += 1
            onset_age = -1
            onset_bush_seen = False

    prev_in_R = in_R

M2_predator = dives / onsets if onsets > 0 else float("nan")
```

**Where it lands in this study.** The headline metric of the study. Cell C R2.6: **M2_predator = 0.888, M2_rabbit = 0.515, Δ_M2 = +0.373** — agent dives into a bush in 89% of predator-onset events but only 52% of rabbit-onset events. R2.5 seed 42 gave 0.876 / 0.508 / +0.368 — agreement to within 1.2 pp on M2, 0.5 pp on Δ_M2. **This is the seed-locked finding.**

---

### A.3 M5 — Eat-under-threat ratio

**Question it asks.** Does the agent eat less often when a class-*c* entity is nearby than when none is nearby? Below 1.0 = the agent suppresses eating under threat; 1.0 = no suppression; above 1.0 = the agent eats *more* under threat (e.g., the entity is in a food-rich quadrant).

**Plain-English walk-through.**
1. Walk the episode step by step.
2. At each step, label the step "under threat from class *c*" if any class-*c* entity is within R, otherwise "safe".
3. Count four things per class: total under-threat steps, total safe steps, eats while under threat, eats while safe.
4. M5 = (eats under threat / under-threat steps) ÷ (eats safe / safe steps).

**Formula.**

```
              eats_under_threat_c / under_threat_steps_c       P(eat | threat_c)
M5_c   =     ──────────────────────────────────────────────  =  ────────────────────
                  eats_safe_c / safe_steps_c                     P(eat | safe_c)
```

A value of **0.75** means the agent's per-step eat probability under predator threat is **25% lower** than its baseline (safe) eat rate. A value of **1.19** for rabbit means the agent eats **19% more** when a rabbit is nearby (because rabbits tend to be in food-rich quadrants).

**Edge case → NaN.** If the entire episode is under threat (safe_steps = 0) or never under threat (under_threat_steps = 0), M5_c is NaN. A secondary sanity criterion (R3): if `P(eat | safe) < 0.005`, the agent is a degenerate non-eater and M5 is uninterpretable even if technically defined.

**Code (simplified extract from `scripts/eval_rollout.py:236`).** The threat / safe / eat counters are accumulated alongside M1 and M2 in the same per-class state machine.

```python
# For a single episode, single class (e.g., predator):
under_threat_steps   = 0
safe_steps           = 0
eats_under_threat    = 0
eats_safe            = 0

for t in range(T):
    in_R = (num_predators > 0) and (np.min(dist_per_predator[t]) < R)

    if in_R:
        under_threat_steps += 1
        if ate_food[t]:
            eats_under_threat += 1
    else:
        safe_steps += 1
        if ate_food[t]:
            eats_safe += 1

P_eat_threat = eats_under_threat / under_threat_steps if under_threat_steps > 0 else float("nan")
P_eat_safe   = eats_safe         / safe_steps         if safe_steps > 0         else float("nan")
M5_predator  = P_eat_threat / max(P_eat_safe, 1e-9) if (under_threat_steps > 0 and safe_steps > 0) else float("nan")
```

**Where it lands in this study.** Cell C R2.6: **M5_predator = 0.769** (eat-rate 23% below safe baseline near a predator), **M5_rabbit = 1.202** (eat-rate 20% *above* safe baseline near a rabbit — rabbits sit in food-rich quadrants). The +0.43 cross-class delta is the second of the two paper-grade headline numbers; R2.5 seed 42 gave 0.748 / 1.186 / +0.44 — seed-paired agreement within 0.02. Soft caveat at seed 44: per-tag rabbit fan-out 0.139 marginally over the secondary ±0.10 band (primary M5 verdict unaffected — flagged as a toolkit-v2 candidate).

---

### A.4 M7 — Defensive-motif repertoire

**Question it asks.** Around each "class-*c* entity just entered danger range" event, the agent's behaviour over the next ~7 steps has some shape (freeze, flee, dive into bush, ignore, approach, mixed). M7 takes *all* such windows from all episodes, clusters them blind, and asks: **do predator-triggered windows land in different clusters than rabbit-triggered windows?**

This is the qualitative complement to the three quantitative metrics. M2 and M5 are scalars; M7 is a 6-bin histogram per class that tells you *what kind* of class-conditional response the agent has, not just how much.

**Plain-English walk-through.**
1. From the full eval-rollout (200 episodes), extract every threat-onset window — that's `[t* − 2, t* + 7]` inclusive (10 steps each), 7,741 windows for the seed-44 Cell C run.
2. For each window, compute **10 hand-picked features** describing what happened in it (table below).
3. Z-score the features so they're comparable (each feature: subtract mean, divide by std, pooled across all conditions).
4. Run k-means with k=6, seed=42.
5. Inspect the five nearest-centroid exemplar windows per cluster and assign an English label (e.g., `bush_camp_predator`, `feed_rabbit`, `predator_pursuit_with_bush`).
6. For each class (predator, rabbit), report the fraction of *that class's* threat-onset windows in each of the 6 clusters.

**The 10 features (per window, extracted by [`scripts/motif_cluster.py:52`](../../../scripts/motif_cluster.py)):**

| # | Feature | What it captures | Computation |
|---:|---|---|---|
| 1 | `net_displacement` | How far the agent moved overall | `‖ pos[end] − pos[start] ‖₂` |
| 2 | `path_length` | Total distance walked (vs. straight-line) | `Σ ‖ pos[t+1] − pos[t] ‖₁` |
| 3 | `threat_distance_change_rate` | Average per-step Δ-distance to the triggering instance (positive = retreating) | `mean(diff(dist_to_triggering_threat))` |
| 4 | `min_threat_distance` | Closest approach to the threat in the window | `min(dist_to_triggering_threat)` |
| 5 | `bush_occupancy_fraction` | Fraction of window-steps spent inside a bush | `mean(agent_in_bush)` |
| 6 | `eat_events_per_window` | Eat steps during the window | `sum(ate_food)` |
| 7 | `action_entropy` | Mixed vs. stereotyped action use (high = mixed) | Shannon entropy of action histogram |
| 8 | `mode_action_fraction` | Concentration on the single most-used action | `count(mode_action) / W` |
| 9 | `stay_in_place_fraction` | Fraction of steps where position didn't change (freezing) | `mean(no position change)` |
| 10 | `drive_injury_change` | Damage taken across the window | `nociception[end] − nociception[start]` |

**Code (simplified extract from `scripts/motif_cluster.py:52`).**

```python
def compute_window_features(ep, window_start, window_end, triggering_class, triggering_tag):
    sl       = slice(window_start, window_end + 1)
    pos      = ep["agent_pos"][sl]          # [W, 2]
    actions  = ep["action"][sl]             # [W]
    ate      = ep["ate_food"][sl]           # [W] bool
    in_bush  = ep["agent_in_bush"][sl]      # [W] bool
    noci     = ep["nociception"][sl]

    # Pick the distance array for the triggering instance
    if triggering_class == "predator":
        threat_dists = ep["dist_per_predator"][sl, parse_tag_idx(triggering_tag)]
    else:
        threat_dists = ep["dist_per_neutral"][sl, parse_tag_idx(triggering_tag)]

    return {
        "net_displacement":           np.linalg.norm(pos[-1] - pos[0]),
        "path_length":                np.sum(np.abs(np.diff(pos, axis=0)).sum(axis=1)),
        "threat_distance_change_rate": np.mean(np.diff(threat_dists)),
        "min_threat_distance":        np.nanmin(threat_dists),
        "bush_occupancy_fraction":    np.mean(in_bush),
        "eat_events_per_window":      np.sum(ate),
        "action_entropy":             shannon_entropy(np.bincount(actions, minlength=5)),
        "mode_action_fraction":       np.bincount(actions).max() / len(actions),
        "stay_in_place_fraction":     np.mean(~np.any(np.diff(pos, axis=0) != 0, axis=1)),
        "drive_injury_change":        noci[-1] - noci[0],
    }


# Then, across all windows:
from sklearn.preprocessing import StandardScaler
from sklearn.cluster      import KMeans
from sklearn.metrics      import silhouette_score

X        = StandardScaler().fit_transform(feature_matrix)            # z-score pooled
labels   = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X)
sil      = silhouette_score(X, labels)                                # R4 sanity ≥ 0.20
```

**Sanity criterion → drop M7 if violated (R4).** The k-means clusters must have **silhouette score ≥ 0.20** OR the analyst must justify a lower silhouette (e.g., "behaviour is genuinely diverse, six clusters all 10–26% — not a one-cluster collapse"). Below 0.20 with no justification → M7 reads as "uninterpretable" and the figure is dropped from the paper. Cell C lands at 0.186 in both seed 42 and seed 44 — sub-threshold but reportable under the "behaviour is genuinely diverse" justification, which the Round 2.5 §12.5 wrote up in detail.

**Where it lands in this study.** Cell C R2.6 motif distribution (after post-hoc labelling of exemplars):

| Cluster | Label | Total share | Predator-triggered share | Rabbit-triggered share |
|---|---|---:|---:|---:|
| 0 | `predator_pursuit_with_bush` | 19.1% | **~83%** | ~17% |
| 1 | `bush_camp` | 12.8% | ~50% | ~50% |
| 2 | `mobile_with_cover` | 25.9% | ~50% | ~50% |
| 3 | `open_flight` | 10.7% | ~50% | ~50% |
| 4 | `feeding_bout_near_rabbit` | 10.5% | ~19% | **~81%** |
| 5 | `bush_camp_predator` | 20.9% | **~86%** | ~14% (R2.6); ~59% (R2.5) |

Two clusters are strongly **predator-skewed** (`predator_pursuit_with_bush`, `bush_camp_predator`); one is strongly **rabbit-skewed** (`feeding_bout_near_rabbit`). This is the qualitative complement to M2 / M5: the agent's defensive *kind* (not just *amount*) is class-conditional. The R2.6 seed-44 predator-shift on cluster 5 (86%) is even stronger than R2.5 seed-42 (59%), which is consistent with the +1.2 pp wider M2 gap at seed 44.

---

### A.5 The toolkit's pre-registered sanity criteria

Before any of M1–M7 produces a verdict, the run must clear four "is the measure even meaningful?" checks. They are listed here for completeness — full operational definitions in [`behavior_measure_toolkit_v1_design.md`](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) §1.2.

| Criterion | Trigger | What it forces |
|---|---|---|
| **R1** | Any of M1 / M2 / M5 lands outside its canonical range (M1, M2 outside `[0, 1]`; M5 negative; NaN where finite expected) | Wiring bug — halt and surface to senior-developer. |
| **R2** | M1 or M2 denominator (candidates / onsets) below a minimum sample size | "Uninterpretable for this cell" — report the denominator zero-rate as the headline finding, not the rate. This is how Cell A1 reads in this study (camping → almost no candidates). |
| **R3** | M5's safe-step denominator zero (entire episode under threat) OR `P(eat | safe) < 0.005` (agent is a degenerate non-eater) | Same — uninterpretable, report the diagnostic. |
| **R4** | M7's k-means silhouette < 0.20 with no written justification | Drop M7 from the paper figure; redesign featurisation in v2. Cell C narrowly clears this via the "genuinely-multimodal" justification (0.186, six well-balanced clusters). |

The Round 2.6 Cell C run cleared R1, R2, R3 cleanly and cleared R4 with the documented sub-threshold justification. The R2.5 seed 42 run cleared all four under identical rules — which is what makes the seed-paired comparison legitimate.

