# _topic_index.md — `hypervigilance` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `hypervigilance` topic.

**Folder definition**: Hypervigilance experiments
**Insights**: 4
**Last updated**: 2026-05-09

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-09 | 15:33 | `20260509_1533_tag_based_distance_supersedes_quadrant` | Tag-based per-instance distance design (optional `tag: <string>` per entity in YAML, source code geometry-agnostic) supersedes the originally-planned quadrant-hardcoded approach. `MeanDistRabbit_<tag>` vs `MeanDistPredator_<tag>` (same tag) is a more direct disambiguator for quadrant-vs-class avoidance than `QuadrantOccupancy_*` would have been. |
| 2026-05-09 | 15:32 | `20260509_1532_sameprop_round2_truncated_verdict` | Round 2 SIGINT'd at 0.4M ep / 3.5h: H1(C) provisionally refuted (Δ=−0.43 sign-flipped vs R1's +0.63); Cell A1's striking Δ=+3.86 is structurally uninterpretable because the agent survives 482/500 steps in the BR corner — §5 row-2 failure mode is active and cannot be discounted without per-tag distance metrics. |
| 2026-05-08 | 14:45 | `20260508_1445_sameprop_discriminating_channels` | Phase 1 ranked channels under sameProp olfactory matching: movement signature dominates; visual ch.5/ch.7 teaches at contact; extero_nociception is contact-only (not at-distance). Plus the `hunt_stamina_threshold > 1.0` structural-disable trick used in Round 2 Cell A1. |
| 2026-05-08 | 14:44 | `20260508_1444_sameprop_round1_finding_and_confound` | RPPO Round 1 confirmed MeanDistRabbit 3.77 < MeanDistPredator 4.40 under matched olfactory properties (n=2 at 7.2M ep), but food/rabbit-quadrant overlap in 01-interoNocicept_sameProp.yaml is a fatal confound — Round 2 Cell C decouples to test. |

---

## Change history

- 2026-05-09: Added 2 insights from the hypervigilance Round 2 partial-verdict + per-tag metrics ship session: `20260509_1532_sameprop_round2_truncated_verdict` (Round 2 SIGINT'd at 0.4M ep; H1(C) refuted, Cell A1 contaminated), `20260509_1533_tag_based_distance_supersedes_quadrant` (tag-based design replaces quadrant-hardcoded). No new tags promoted (all reused: hypervigilance, refutation, learned_lesson, design, decision, meta).
- 2026-05-08: Folder created. Added 2 insights from the hypervigilance/sameProp Round 1 + Round 2 design session: `20260508_1444_sameprop_round1_finding_and_confound`, `20260508_1445_sameprop_discriminating_channels`.
