# _topic_index.md — `hypervigilance` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `hypervigilance` topic.

**Folder definition**: Hypervigilance experiments
**Insights**: 7
**Last updated**: 2026-05-13

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-13 | 00:15 | `20260513_0015_active_swapped_geq_matched_reframes_meta` | Specialist ceiling table from the 6-world unmodulated probe (single-seed, 10M ep each) shows active_swapped (402 steps) > active_matched (338 steps). Swap is NOT harder than matched in isolation, which refutes the original 'swap is load-bearing' framing for the upcoming meta head-to-head and reframes it as a CKA-factorisation test instead. |
| 2026-05-12 | 14:28 | `20260512_1428_sameprop_class_discriminating_defence_event_level` | Round 2.5 toolkit-v1 appendix refines the verdict — under sameProp matched smells, the agent IS class-discriminating, but at the EVENT level (bush-dive rate +37 pp predator vs rabbit; eat-under-threat 0.75× near predator vs 1.19× near rabbit), not the spatial-trajectory level (mean distance). Per-tag fan-out across rabbit_TL / rabbit_BR within noise — rules out single-instance artifacts. Complements `20260510_2237_sameprop_round25_no_class_avoidance`, does not supersede. |
| 2026-05-10 | 22:37 | `20260510_2237_sameprop_round25_no_class_avoidance` | Round 2.5 (10M ep, n=1 each cell) refutes both pre-registered hypotheses: per-tag Δ_TL = +0.004 cells in Cell A1 (75× inside the H₀ band) confirms location-conditional corner-camping, not class recognition; Cell C aggregated Δ = −0.53 cells (sign-flipped vs R1's +0.63) with bilateral rabbit avoidance lands in §4.3 Inverted. The original sameProp survey effect (+0.6 cells) decomposes into two confounds (food/quadrant overlap + spatial-avoidance camouflage) — no genuine class-conditional avoidance under matched olfactory smells. Provisional pending Round 2.6 seed 44 for Cell C. |
| 2026-05-09 | 15:33 | `20260509_1533_tag_based_distance_supersedes_quadrant` | Tag-based per-instance distance design (optional `tag: <string>` per entity in YAML, source code geometry-agnostic) supersedes the originally-planned quadrant-hardcoded approach. `MeanDistRabbit_<tag>` vs `MeanDistPredator_<tag>` (same tag) is a more direct disambiguator for quadrant-vs-class avoidance than `QuadrantOccupancy_*` would have been. |
| 2026-05-09 | 15:32 | `20260509_1532_sameprop_round2_truncated_verdict` | Round 2 SIGINT'd at 0.4M ep / 3.5h: H1(C) provisionally refuted (Δ=−0.43 sign-flipped vs R1's +0.63); Cell A1's striking Δ=+3.86 is structurally uninterpretable because the agent survives 482/500 steps in the BR corner — §5 row-2 failure mode is active and cannot be discounted without per-tag distance metrics. |
| 2026-05-08 | 14:45 | `20260508_1445_sameprop_discriminating_channels` | Phase 1 ranked channels under sameProp olfactory matching: movement signature dominates; visual ch.5/ch.7 teaches at contact; extero_nociception is contact-only (not at-distance). Plus the `hunt_stamina_threshold > 1.0` structural-disable trick used in Round 2 Cell A1. |
| 2026-05-08 | 14:44 | `20260508_1444_sameprop_round1_finding_and_confound` | RPPO Round 1 confirmed MeanDistRabbit 3.77 < MeanDistPredator 4.40 under matched olfactory properties (n=2 at 7.2M ep), but food/rabbit-quadrant overlap in 01-interoNocicept_sameProp.yaml is a fatal confound — Round 2 Cell C decouples to test. |

---

## Change history

- 2026-05-13: Added insight `20260513_0015_active_swapped_geq_matched_reframes_meta` from the NMN R2 continual + 6-specialist analyzer verdict session — specialist ceiling table inverts design-time intuition (active_swapped > active_matched by +64 steps), refutes 'swap is load-bearing' framing for the upcoming meta head-to-head and reframes it as a CKA-factorisation test. No new tags promoted (all reused: hypervigilance, learned_lesson, refutation, design).
- 2026-05-12: Added insight `20260512_1428_sameprop_class_discriminating_defence_event_level` from the behavior-measure toolkit v1 application to the R2.5 checkpoints — refines (not supersedes) the prior `_no_class_avoidance` verdict. Under sameProp the agent IS class-discriminating at the event level (bush-dive rate, eat-suppression) even though it is class-blind in mean distances. No new tags promoted (all reused: hypervigilance, learned_lesson, decision).
- 2026-05-10: Added insight `20260510_2237_sameprop_round25_no_class_avoidance` from the Round 2.5 launch + analysis session — both pre-registered hypotheses refuted; sameProp survey effect decomposed into two confounds, no genuine class-conditional avoidance. Provisional pending Round 2.6 seed 44 (Cell C only). No new tags promoted (all reused: hypervigilance, refutation, learned_lesson, decision).
- 2026-05-09: Added 2 insights from the hypervigilance Round 2 partial-verdict + per-tag metrics ship session: `20260509_1532_sameprop_round2_truncated_verdict` (Round 2 SIGINT'd at 0.4M ep; H1(C) refuted, Cell A1 contaminated), `20260509_1533_tag_based_distance_supersedes_quadrant` (tag-based design replaces quadrant-hardcoded). No new tags promoted (all reused: hypervigilance, refutation, learned_lesson, design, decision, meta).
- 2026-05-08: Folder created. Added 2 insights from the hypervigilance/sameProp Round 1 + Round 2 design session: `20260508_1444_sameprop_round1_finding_and_confound`, `20260508_1445_sameprop_discriminating_channels`.
