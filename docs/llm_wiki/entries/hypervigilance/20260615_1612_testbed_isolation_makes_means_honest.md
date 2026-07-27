---
id: 20260615_1612_testbed_isolation_makes_means_honest
date: 2026-06-15
time: "16:12"
folder: hypervigilance
tags: [hypervigilance, design, decision, learned_lesson]
summary: "Methodology decision: rather than abandon mean-level measures (we need scalable quantitative numbers), build dedicated isolated TEST environments (frozen-checkpoint probes, no retraining) that strip the confounds which made means misleading in the complex training world — so the mean becomes a trustworthy proxy for the trajectory-level truth. Every new testbed must first be calibrated against a known discriminator (Cell C agent) and a known class-blind agent (cell-08) before its numbers are trusted."
related: ["20260512_1428_sameprop_class_discriminating_defence_event_level", "20260609_1721_aggregate_stats_hide_conditional_behavior", "20260609_1747_avoidance_is_post_contact_not_preemptive", "20260612_1625_predator_rabbit_discrimination"]
session_origin: claude_code
session_label: "predator-rabbit discrimination — testbed charter + measurement-direction rethink"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/144f126e-4b23-4e3f-bc6f-cc2491b3f2a4.jsonl
raw_completeness: full
---

# Isolated test environments make mean-level measures honest (testbed charter rationale)

## Key conclusion

The fix for "mean-level measures hid a real discrimination" is **not** to abandon means — we
need scalable quantitative numbers, and hand-reading trajectories does not scale. The fix is
to change the *situation*, not the *statistic*: build dedicated **TEST environments** kept
entirely separate from training (frozen-checkpoint probes, **no retraining**), each
**isolating one factor and removing the confounds**, so that a simple mean-level measure is no
longer misled and again reflects what a step-by-step trajectory read would show. The test
environment's whole job is **confound removal in service of making the mean honest**. Because
the complex training world mixes many situations together, its aggregates average a
*conditional* behaviour to ~zero; an isolated world has nothing left to mislead the average.

## Evidence, measurements, facts

- Origin of the problem: in the complex world the aggregates declared the agent "class-blind"
  (mean predator-vs-rabbit distance ~0; distance-matched flee rate 76% vs 76%;
  interrupted-feeding 0.000), but a full step-by-step trajectory read showed it WAS
  discriminating (kept eating when a rabbit approached; stopped eating + dived into a bush when
  the predator approached). See [[20260609_1721_aggregate_stats_hide_conditional_behavior]].
- **Calibration requirement (hard rule):** a new testbed's numbers are not trusted until the
  testbed reproduces a known result. Run two control agents through every new test world:
  - known **discriminator** = the Cell C matched-smell agent (checkpoint `…20260516-132103_hypervigilance-round26-C-seed44…/models/10000021`): bush-dive 0.76 predator vs 0.44 rabbit; eat-under-threat 0.73 vs 1.26.
  - known **class-blind** = the cell-08 single-pred-rabbit agent (checkpoint `…20260609-191226_recurrent_ppo_08-singlePredRabbit_disengage…/models/7140046`): bush-dive 0.576 vs 0.562; eat-under-threat ~1.0 vs ~1.0.
  The testbed PASSES only if its mean-level measure separates these two the way the trajectory
  read does. If not, the isolation is insufficient and the testbed is reworked.
- Design rules for any testbed: frozen-checkpoint probe (no retraining); change one factor,
  match everything else; readable before contact (or contact impossible) to split anticipatory
  from post-contact reaction; avoid the known confounds (motion/chase, approach-speed/aggression,
  animal number, out-of-distribution asymmetry, pre/post-contact mixing); same measurement panel
  across all worlds reporting distributions not just means; always a matched contrast (never a
  single class); cross-check the mean against a `trajectory-story` read.
- Two-tier framing comes from the project's own EVAAA benchmark paper (Lee et al. 2025): a
  naturalistic training curriculum plus controlled testbeds that isolate one decision process by
  minimising irrelevant cues (`docs/project/references/InteroceptiveAI/sources/`).
- First two testbeds: **predator-solo** and **rabbit-solo** (renamed from the earlier E3/E2),
  one-animal ablations of the single-pred-rabbit world; the read-out is the predator-solo vs
  rabbit-solo contrast on the shared panel.

## Decisions and actions

- Adopted: isolated test environments as the measurement instrument; means retained but measured
  where confounds can't fool them; trajectory read demoted from primary verdict to validation
  cross-check.
- Wrote the anchor charter doc `docs/experiments/active/hypervigilance/behavior_measurement_charter.md`
  (governs all future behavior-measurement testbeds).
- Renamed the first two testbeds E3 -> predator-solo, E2 -> rabbit-solo.
- Next action: experiment-designer to author predator-solo / rabbit-solo configs (one-factor
  edits of cell-08), env-config-auditor pre-flight, then run the Cell C + cell-08 control pair
  through both worlds to calibrate before trusting the assay.

## Open questions and follow-ups

- Does the mean-level panel in predator-solo / rabbit-solo actually separate Cell C from cell-08
  (calibration pass)? If the isolated world's mean still cannot separate the known discriminator
  from the known class-blind agent, the single-animal ablation is insufficient and a stronger
  isolation (forced-choice / barrier-anticipation) is needed.
- Both control agents trained in different worlds and go mildly out-of-distribution in the solo
  worlds; the OOD-ness is matched across the two worlds and the two agents, but flag it in any
  write-up.

## References

- Anchor doc: `docs/experiments/active/hypervigilance/behavior_measurement_charter.md`.
- Study summary (with the trajectory-vs-mean "most important finding" callout): [[20260612_1625_predator_rabbit_discrimination]] is the study-summary slug under `docs/experiments/summaries/`.
- Builds on the methodology post-mortem [[20260609_1721_aggregate_stats_hide_conditional_behavior]] and the event-level discrimination finding [[20260512_1428_sameprop_class_discriminating_defence_event_level]]; the class-blind cell-08 reading sits under the June discrimination thread alongside [[20260609_1747_avoidance_is_post_contact_not_preemptive]].
- Candidate testbed menu: `docs/experiments/active/hypervigilance/predator_rabbit_testbeds.md`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 144f126e-4b23-4e3f-bc6f-cc2491b3f2a4` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260616_0142_discrimination_is_spatial_encounter_artifact]] (hypervigilance, 2026-06-16) — Autonomous testbed search (4 designs, frozen-checkpoint evals of a Cell C 'discr
- [[20260616_1514_experimental_env_as_behavior_platform]] (hypervigilance, 2026-06-16) — Reframe: the testbed-search result that the predator-vs-rabbit 'discrimination' 
<!-- END BACKLINKS -->
