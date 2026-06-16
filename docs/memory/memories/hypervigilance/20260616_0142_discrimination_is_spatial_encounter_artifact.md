---
id: 20260616_0142_discrimination_is_spatial_encounter_artifact
date: 2026-06-16
time: "01:42"
folder: hypervigilance
tags: [hypervigilance, refutation, learned_lesson, decision]
summary: "Autonomous testbed search (4 designs, frozen-checkpoint evals of a Cell C 'discriminator' vs a cell-08 'class-blind' control) shows the predator-vs-rabbit bush-dive 'discrimination' is a SPATIAL-ENCOUNTER ARTIFACT, not class-recognition: a KNOWN class-blind agent reproduces the full +0.31 bush-dive gap when dropped into Cell C's asymmetric decoupled-food/confined-rabbit world, and BOTH agents go flat under geometry control (symmetric common world; mirror-counterbalanced arena). No isolated test world calibrated as a discrimination metric. Resolves insight 1428's open question: the +37pp event-level gap was itself spatially mediated."
related: ["20260512_1428_sameprop_class_discriminating_defence_event_level", "20260609_1720_chasing_rabbit_avoidance_damage_driven", "20260609_1721_aggregate_stats_hide_conditional_behavior", "20260609_1747_avoidance_is_post_contact_not_preemptive", "20260615_1612_testbed_isolation_makes_means_honest"]
session_origin: claude_code
session_label: "autonomous overnight search for a validated discrimination-measurement testbed"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/144f126e-4b23-4e3f-bc6f-cc2491b3f2a4.jsonl
raw_completeness: full
---

# Predator-vs-rabbit "discrimination" is a spatial-encounter artifact, not class-recognition

## Key conclusion

An autonomous 4-iteration testbed search — built to find an isolated test environment whose
mean-level measure reproduces the known verdict (Cell C discriminates, cell-08 class-blind) —
found NO valid metric, because the apparent discrimination is a **spatial-encounter artifact**.
The decisive evidence: the **known class-blind agent (cell-08) reproduces the full bush-dive gap
(+0.307) when evaluated in Cell C's asymmetric world** (decoupled food in two quadrants, rabbits
confined to the other two, predator full-grid) — essentially identical to the "discriminator"
Cell C's +0.295 in the same world. Under controlled geometry (a symmetric common world, and a
mirror-counterbalanced static arena) **neither** agent shows any class difference on bush-dive,
eat-suppression, or distance. The gap tracks the WORLD, not the AGENT: in that geometry the
full-grid predator simply encounters the foraging agent in threatening contexts more than the
spatially-confined rabbits do, so any agent looks predator-avoidant.

## Evidence, measurements, facts

- **The 2×2 (within-world Δ = predator − rabbit, bush-dive rate):**
  - symmetric world (cell-08 config): Cell C +0.016, cell-08 −0.002.
  - asymmetric world (Cell C native, 1 pred + 2 confined rabbits): Cell C +0.295, **cell-08 +0.307**.
- **Eat-suppression** told the same story: Cell C's selective predator-suppression (Δ −0.370) exists
  ONLY in its native asymmetric world; in the symmetric mirror arena it collapses to −0.012
  (cell-08 −0.016) — i.e. also geometry-mediated.
- **Trajectory-level cross-check (charter rule #7, done post-hoc):** confirms the means. Step-by-step dumps show the rabbits locked in their TL/BR corners (distance 5-16 the whole episode, never approaching the agent) while the predator roams the full grid — so a class-blind agent 'avoids the predator more' only because the predator is the sole animal that gets near. Flee-decomposition shows NO predator-specific avoidance (both agents flee the close-range rabbit as much/more); symmetric world flat at trajectory level too (cell-08 dist-1 flee predator +0.592 vs rabbit +0.564).
- **Solo worlds (predator-only / rabbit-only, cross-world contrast):** FAIL and INVERT — cell-08 showed
  the only suppression (−0.119, a reactive response to a lone lethal predator); Cell C looked flat
  (+0.069 bush-dive). Single-animal isolation destroys the contrast discrimination is made of and adds
  a lethality confound (Cell C dies early in predator-only: 160 vs 443 steps).
- **Symmetric mirror arena** (static predator vs rabbit at mirror halves, predator-left and -right runs
  averaged to cancel side bias): Cell C Δdist +0.22 / Δeat −0.012; cell-08 Δdist 0.00 / Δeat −0.016.
- Controls: Cell C `…20260516-132103_hypervigilance-round26-C-seed44…/models/9990005`; cell-08 final
  `…20260609-191226_recurrent_ppo_08-singlePredRabbit_disengage…/models/9990029`. 200 deterministic
  episodes per run; frozen checkpoints, no retraining.
- Loadability note: Cell C's May config needed one deprecated v2.0 key stripped (`environment.predator_enabled`)
  to load in the current pipeline; geometry otherwise intact (1 predator + 2 rabbits, decoupled food).

## Decisions and actions

- Adopted: the predator-vs-rabbit discrimination signal is a spatial-encounter artifact; this
  mechanistically explains the June "class-blind / pain-reactive, not anticipatory" verdict
  ([[20260609_1747_avoidance_is_post_contact_not_preemptive]]) and resolves the open question in
  [[20260512_1428_sameprop_class_discriminating_defence_event_level]] (its +37pp event-level gap was
  itself spatially mediated — that insight's within-world numbers stand; its class-recognition
  interpretation does not survive the geometry controls).
- The metric-validation program has **no confirmed genuine discriminator** to calibrate against.
- Next moves (deferred to user): (1) establish a real discriminator by training with a learnable
  distal class cue, then re-run this calibration; or (2) accept the class-blind/artifact conclusion.
- Reusable lesson for the charter: calibrate-before-trust caught this — every candidate world would
  have shipped as a "discrimination metric" while measuring geometry; and single-animal isolation is
  the wrong isolation for a contrast phenomenon. See [[20260615_1612_testbed_isolation_makes_means_honest]].

## Open questions and follow-ups

- Single seed per agent; the symmetric tests put the Cell C agent out-of-distribution, so its OWN
  flatness there could be OOD confusion — but the airtight leg (a class-blind agent reproducing the
  gap in Cell C's world) does not depend on that.
- Would an agent trained with a genuine distal class cue show a geometry-ROBUST gap? That is the test
  that would confirm the phenomenon is real rather than artifactual.

## References

- Results write-up: `docs/experiments/active/hypervigilance/testbed_solo_validation_results.md`.
- Charter: [[behavior_measurement_charter]] is at `docs/experiments/active/hypervigilance/behavior_measurement_charter.md`.
- Refines [[20260512_1428_sameprop_class_discriminating_defence_event_level]]; reinforces [[20260609_1747_avoidance_is_post_contact_not_preemptive]] and [[20260609_1720_chasing_rabbit_avoidance_damage_driven]]; method [[20260615_1612_testbed_isolation_makes_means_honest]] and [[20260609_1721_aggregate_stats_hide_conditional_behavior]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 144f126e-4b23-4e3f-bc6f-cc2491b3f2a4` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260616_1514_experimental_env_as_behavior_platform]] (hypervigilance, 2026-06-16) — Reframe: the testbed-search result that the predator-vs-rabbit 'discrimination' 
<!-- END BACKLINKS -->
