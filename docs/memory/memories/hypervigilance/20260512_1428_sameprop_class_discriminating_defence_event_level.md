---
id: 20260512_1428_sameprop_class_discriminating_defence_event_level
date: 2026-05-12
time: "14:28"
folder: hypervigilance
tags: [hypervigilance, learned_lesson, decision]
summary: "Round 2.5 toolkit-v1 appendix refines the verdict — under sameProp matched olfactory smells, the agent IS class-discriminating, but at the EVENT level (bush-dive rate, eat-under-threat suppression) not the spatial-trajectory level (mean distance). In Cell C the agent's M2 bush-dive rate is +37 pp higher for predator vs rabbit (87.6% vs 50.8%); M5 eat-under-threat is 0.75× near predator and 1.19× near rabbit; per-tag fan-out across rabbit_TL / rabbit_BR is within noise, ruling out single-instance artifacts. The §§9-11 mean-distance verdict ('no class-conditional avoidance') is correct at its level but incomplete — the trajectory-level dynamics that mean distances dissolved away encode genuine class discrimination. Complements `20260510_2237_sameprop_round25_no_class_avoidance`, does not supersede it."
related: [20260510_2237_sameprop_round25_no_class_avoidance, 20260509_1533_tag_based_distance_supersedes_quadrant, 20260508_1444_sameprop_round1_finding_and_confound]
session_origin: claude_code
session_label: "hypervigilance Round 2.5 + behavior-measure toolkit ship"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl
raw_completeness: full
---

# SameProp class-conditional defence is at the event level, not the spatial-trajectory level

## Key conclusion

The Round 2.5 mean-distance verdict (`20260510_2237_sameprop_round25_no_class_avoidance`) concluded that under matched olfactory smells, the agent does not learn class-conditional avoidance — Cell A1 collapsed to corner-camping (per-tag Δ_TL = +0.004 cells), Cell C inverted with bilateral rabbit avoidance (aggregated Δ = −0.53 cells). The behavior-measure toolkit v1 appendix to that design doc (§12, applied 2026-05-11) reads the same two checkpoints with M1 (interrupted-feeding rate), M2 (bush-dive rate), M5 (eat-under-threat ratio), and M7 (motif clustering on threat-window trajectories). The toolkit recovers a sub-pattern the mean dissolved away: **Cell C is spatially class-blind but behaviourally class-discriminating**. The agent's bush-dive rate is +37 pp higher when a predator (vs a rabbit) enters R=3 cells (87.6% vs 50.8%); its eat-under-threat ratio is suppressed to 0.75× near predator and elevated to 1.19× near rabbit; motif distribution is dominated (~78%) by bush-involved clusters with one cluster (`predator_pursuit_with_bush`) 83% predator-triggered. Cell A1 remains class-blind at every layer — both spatially AND behaviourally — because the corner-camping policy almost never has the agent encountering a same-corner predator while feeding. The headline reframing: **mean-distance metrics measure foraging routes, not defensive responses; under sameProp the class signal lives in the defensive event repertoire, and the mean dissolves it**.

## Evidence, measurements, facts

- **Eval-rollout setup**: 200 deterministic episodes per checkpoint, R=3.0 cells, K=5 steps, K_motif=7 steps. Dumps under `results/eval/models/10000003/` (Cell A1) and `results/eval/models/10000022/` (Cell C). Eval-time numbers (not training-time WandB scalars) so the verdict reflects the converged deterministic policy.

- **Cell C (decouple food, seed 42, WandB `bdnfc0lu`, eval-time)**:
  - M1 interrupted-feeding rate per class: predator_full ≈ **42.2%**, rabbit (mean over BR + TL) ≈ **24.1%**. Predator triggers interruption nearly 2× as often as a rabbit when both are within R=3 of the eating agent.
  - M2 bush-dive rate per class: predator_full ≈ **87.6%**, rabbit (mean) ≈ **50.8%**. **Δ_M2 = +36.8 pp** in favour of predator — the smoking-gun number for event-level class discrimination.
  - M5 eat-under-threat ratio per class: predator_full ≈ **0.748** (eating suppressed when predator near), rabbit (mean) ≈ **1.186** (eating slightly elevated when rabbit near — the agent treats rabbit-proximity as a non-threat marker).
  - Per-tag rabbit fan-out: rabbit_TL M2 = 50.3%, rabbit_BR M2 = 52.3% (Δ = 2 pp, within noise); rabbit_TL M5 = 1.21, rabbit_BR M5 = 1.13. **Class-conditional defence is consistent across rabbit instances** — not a "one rabbit triggers the response, the other doesn't" artifact.
  - M7 motif distribution (k=6, silhouette 0.186): C2 mobile_with_cover 25.9%, C5 bush_camp_predator 20.9% (59% predator-triggered), C0 predator_pursuit_with_bush 19.1% (**83% predator-triggered**), C1 bush_camp 12.8%, C3 open_flight 10.7%, C4 feeding_bout_near_rabbit 10.5% (81% rabbit-triggered). Bush-involved motifs total ≈ **78%** of all threat-window activity.

- **Cell A1 (passive predator, seed 43, WandB `nm8gn7y2`, eval-time)**:
  - M1: predator_TL = 11.8%, rabbit (mean) = 12.8% (within window-noise, |Δ| = 1 pp).
  - M2: predator_TL = 3.2%, rabbit (mean) = 0.5% (both low; agent rarely uses bushes because it rarely encounters threats due to corner-camping).
  - M5: predator_TL ≈ 0.999, rabbit ≈ 1.141 (no class-conditional suppression).
  - Per-tag Δ_M1_TL = −2.8 pp, Δ_M2_TL = +0.9 pp, Δ_M5_TL = −0.06 — class-blind at the event level too, mirroring the mean-distance verdict.
  - M7 motif distribution (k=6, silhouette 0.237): C0 freeze_near_BR 33.5%, C5 stationary_rabbit_approaches 26.3%, C4 engage_BR 17.4%, C2 approach_transit 11.4%, C3 feeding_bout 10.8%, C1 anomaly 0.5%. Stationary/freeze motifs total ≈ **60%**; bush-involved ≈ 22% (much lower than Cell C).

- **Cross-cell comparison**: the two cells are qualitatively DIFFERENT at the event level. Cell A1's class-blindness is genuine at every layer (spatial + behavioural). Cell C looks class-blind in the mean but is class-DISCRIMINATING at the event level — bush-diving rate, eat-suppression, and predator-triggered motif fraction all separate predator from rabbit by paper-grade margins.

- **Methodological implication**: aggregated `Episode/MeanDistPredator − Episode/MeanDistRabbit` (the §§9-11 measure) is **insensitive to event-level class discrimination** when the agent's spatial trajectory is dominated by foraging-route geometry. Under sameProp, the agent walks similar routes near both classes (the geometry forces this — food is in TR+BL, rabbits in TL+BR, predator full-grid), but it bush-dives selectively when the *predator* enters R. The mean averages over both threat-near and threat-far steps, where the latter dominate.

## Decisions and actions

- §12 appendix added to `docs/experiments/active/hypervigilance/sameprop_round25_design.md` (commit `ed5cff3`); §9 carries a one-line pointer to §12.
- Verdict refinement: Round 2.5 closes with **two-level interpretation**, not one. (a) Spatial-trajectory level (§§9-11): no class-conditional avoidance — agent doesn't keep predators farther than rabbits in mean L2 distance. (b) Event-response level (§12): genuine class-conditional defence — agent bush-dives, suppresses feeding, and selects different motifs when the predator approaches.
- **Round 2.6 design (seed 44, Cell C)** should pre-register the toolkit measures as primary verdicts, not just the aggregated Δ. Proposed locked confirmation criteria from §12.7: `M2_predator ≥ 0.80 AND Δ_M2_class ≥ +30 pp AND M5_predator < 0.80`. The mean-distance Δ stays as a secondary metric for cross-round comparability.
- **Round 3 design** (food in all four quadrants — closing the spatial-avoidance loophole) should pre-register M2/M5 *widening* (or maintaining the +37 pp gap) as the confirmation signal for class-conditional defence under a closed spatial loophole. If M2_predator / M5_predator under all-4-quadrant food still show the +37 pp / 0.75× pattern, class discrimination is robust to spatial geometry; if they collapse, the Cell C event-level discrimination was itself spatially mediated.
- The wider hypervigilance arc now has a richer baseline question: not just "does the agent avoid the dangerous class in space?" but "does its event-level defensive repertoire treat the dangerous class differently?" The latter survives sameProp; the former does not.

## Open questions and follow-ups

- **NMN / FiLM agent comparison**: does a modulated agent under the same Cell C setup show a *larger* event-level class discrimination (higher Δ_M2)? Or is the +37 pp gap a baseline plain-RPPO capability and the modulator doesn't help here? This is the natural sequel that ties this finding to the NMN comparison thread.
- **Habituation traces (M6 from the postdoc memo, currently deferred)**: do repeated within-episode predator encounters weaken the bush-dive response (habituation) or strengthen it (sensitisation)? The current M2 is a per-episode rate; M6 would expose dynamics across encounters. Worth promoting to the v1.1 toolkit if Round 2.6 / Round 3 needs it.
- **The Cell A1 "C1 anomaly" cluster (0.5%)**: tiny, but its existence in the corner-camping policy suggests rare excursions from the BR corner are not just noise — they may be informative about transition dynamics. Not load-bearing; flagged for hand-inspection if a future round wants it.
- **Why does Cell C use bushes so much (78% of motifs)?** Possible explanations: the predator's full-grid roaming makes pre-emptive bush-camping a viable policy; the food in TR + BL is geometrically near bush patches so the agent learns to "graze near refuge." The motif clusters distinguish bush_camp_predator (20.9%, 59% predator-triggered) from bush_camp generic (12.8%) — suggesting both pre-emptive and reactive bush use exist. A behavioural decomposition of "pre-emptive vs reactive" bush-diving would refine this further.
- **Generalisation of the methodology lesson**: every aggregated mean-distance metric in this project may be insensitive to event-level discrimination. Whenever a mean-distance verdict says "no class-conditional avoidance", the toolkit's M2 + M5 + M7 should be applied before concluding the agent is class-blind. The Round 2.5 case is a worked example; future studies should adopt this as standard practice.

## References

- Design doc + appendix: `docs/experiments/active/hypervigilance/sameprop_round25_design.md` §12 (commit `ed5cff3`).
- Mean-distance verdict that this insight refines: `20260510_2237_sameprop_round25_no_class_avoidance` (NOT superseded — complementary; the prior insight is correct at its level).
- Toolkit design + plan + verification:
  - `docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md` (design)
  - `docs/develop/active/behavior/behavior_measure_toolkit_v1_plan.md` (`verification_status: pass`, commit `dbd0e64`)
  - `docs/project/ideas/20260510_behavior_measure_toolkit.md` (biological grounding)
- Working files: `tmp/20260511_r25_appendix_*` (analyzer's intermediate scripts + CSVs).
- Eval dumps: `results/eval/models/10000003/` (Cell A1), `results/eval/models/10000022/` (Cell C). 200 episodes each.
- WandB runs: `nm8gn7y2` (Cell A1), `bdnfc0lu` (Cell C).
- Re-summary candidate: a third snapshot of the sameProp study under `docs/experiments/summaries/` is appropriate after this finding lands, but deferred unless the user requests it (the §12 appendix + this insight already cover the verdict refinement).
- Toolkit fix history: developer's bug-fix commits `e5e1155` (ratio NaN) + `2d6d288` (eval_rollout RPPO restore) shipped after the verification protocol caught them.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 89124f20-6e84-466c-ab10-50a9651c68c8` or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl /tmp/20260512_1428.md`.
