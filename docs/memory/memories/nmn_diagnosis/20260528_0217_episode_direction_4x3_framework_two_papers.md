---
id: 20260528_0217_episode_direction_4x3_framework_two_papers
date: 2026-05-28
time: "02:17"
folder: nmn_diagnosis
tags: [nmn, decision, design, meta]
summary: "project_plan.md rewritten from 760-line Nature MI staged plan to 138-line stable direction context. Project rebranded EPISODE (Emergence of Pain In Simulated Organismic & Dynamic Environments). 4 behavioral categories × 3 levels of analysis. Two papers (NMI + NeurIPS) at perspective-plus-pilot calibration."
related: ["20260516_1504_symposium_substrate_right_rhetoric_wrong", "20260516_1505_target_one_acknowledge_many_defer_full_coverage", "20260516_1508_v5_last_rhetorical_round_before_experiments", "20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases"]
session_origin: claude_code
session_label: "EPISODE project_plan rewrite + Foam wikilink convention + VSCode startup fix"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/e386e7cb-3b2e-4604-abc2-921d1b6c973a.jsonl
raw_completeness: full
---

# EPISODE — project_plan as stable direction context, 4×3 framework, two papers at perspective-plus-pilot

## Key conclusion

`docs/project/project_plan.md` was rewritten end-to-end from the prior 760-line Nature MI–specific staged plan (G1′/G2′ gates, T/P modulator split, ~138–168 run budget, four-professor consultation chain) to a 138-line stable direction-context document. The project gained an acronym — EPISODE: Emergence of Pain In Simulated Organismic & Dynamic Environments. The new frame organises work in a 4 × 3 matrix: four behavioral categories of pain-like behavior (avoidance / recovery / managing conflict needs / hypervigilance) × three levels of analysis (behavioral / representational / algorithmic). Two papers are committed: Nature Machine Intelligence (neuro + robotics audience — "pain is more than nociception") and NeurIPS (ML audience — "neuromodulation-inspired modulators unify perceptual + hyperparameter + continual-learning modulation"). Both papers calibrated as "perspective-plus-pilot" — demonstrate the framework is plausible and worth pursuing, NOT exhaustively solve any category. The strict falsification framing (G2′ five-quantity fingerprint) is dropped; v8 null becomes evidence for the framework rather than a failure.

## Evidence, measurements, facts

- Prior plan archived to `docs/project/archive/project_plan_2026-05-26.md` (commit `82a4336`, replaced with 138-line direction context). EPISODE title applied (`8ab278c`). Aliases added (`bb522d4`). §3.2 "Perceptual" level renamed to "Representational" (`6f0d01a`) to capture memory-gate activity, value/policy representations, modulator-state trajectories — not just sensory features.
- The four behavioral categories were the user's framing (not Claude's): avoidance (context-conditioned predictions beyond reflex withdrawal), recovery (graded post-damage trajectory), conflict management (foraging-vs-avoidance trade-off — drive conflict as a FEATURE, not a confound to rule out), hypervigilance (channel-selective + context-modulated + outlasting input).
- The three levels: behavioral (action distributions, survival, time-to-recover), representational (internal-representation changes — encoder gain, memory-gate activity, value/policy, modulator state, damage-history latents), algorithmic (modulator's place in FiLM/hypernet family, Doya hyperparameter modulation, continual-learning modulator gating).
- Paper 1 (NMI) algorithmic commitment is minimal and shared: one FiLM-variant modulator at the perceptual modulation injection site, applied uniformly across all four categories. Hyperparameter and continual-learning modulation are deferred to Paper 2.
- Paper 2 (NeurIPS) unifies three ML subfields treating parameter-modulation as separate problems: perceptual modulation (FiLM, AdaIN, conditional batch/layer norm, hypernets), hyperparameter modulation (Doya 2002 lineage — modulator outputs temperature/exploration/discount/learning-rate as context-conditioned functions), continual learning (HyperNet, PathNet-family, modulator-gated experts for catastrophic-forgetting prevention).
- Documentation framing of new doc: opens with plain-English entry-point, "What this document is" section explicitly states what it is NOT (no phases, no gates, no run budgets, no current state, no pre-registered hypotheses). Subtask documents in `docs/develop/`, `docs/experiments/`, `docs/pi/` carry the concrete artifacts. This document does NOT change as subtasks land.
- The user explicitly extends the previous "target one, acknowledge many, defer full coverage" rule ([[20260516_1505_target_one_acknowledge_many_defer_full_coverage]]): EPISODE keeps full coverage of all 4 categories AND all 3 subfields, but at perspective-plus-pilot calibration — make the framework concrete, runnable, credible enough to move the field; do NOT saturation-solve any single cell.

## Decisions and actions

- All future research subtasks (plans, experiments, analyses) should locate themselves in the 4 × 3 matrix and declare which paper (NMI Paper 1 or NeurIPS Paper 2) they contribute to. Subtask docs reference the project_plan via the `[[project_direction]]` alias.
- The v8 null result is no longer a "failure to crack hypervigilance" — under perspective-plus-pilot, it's a useful negative finding for the discussion section ("hypervigilance is the hardest of the four; here's what works, what doesn't, what the field should pursue").
- "Subjective pain / qualia", "clinical translation", "affective-vs-sensory dissociation", "sim-to-real transfer", and "exhaustive cracking of any one category or subfield" are explicitly out of scope. Reviewer questions on these get acknowledged-as-limitation responses, not over-reaching claims.
- The previous strict G2′ five-quantity pre-registered fingerprint is dropped. Each category gets at least a behavioral-level demonstration; one or two carry representational-level mechanism; the framework's joint demonstration is the contribution.

## Open questions and follow-ups

- Concrete experimental designs for each of the 4 × 3 = 12 cells are not in this document — they belong in `docs/experiments/` plans that reference back to this frame.
- The order of the four categories' empirical work (which to do first) is not committed in this doc. Likely "avoidance" first (foundational, most existing infrastructure) and "conflict management" second (clearest pain-vs-nociception differentiator), but this is operational sequencing, not direction.
- Paper 2's empirical bar — whether to require a tighter benchmark comparison on one of the three subfields — is still TBD. The convention says "at least one of the three" gets a tighter empirical anchor; which one is unresolved.

## References

- New plan: `docs/project/project_plan.md` (referenceable via `[[project_direction]]` alias).
- Archived prior plan: `docs/project/archive/project_plan_2026-05-26.md` (760 lines; full G1′/G2′/T-P-split history preserved).
- Commits: `82a4336` (rewrite + archive), `8ab278c` (EPISODE rename), `bb522d4` (aliases on 6 god-nodes), `6f0d01a` (Perceptual → Representational rename).
- Prior NMN-diagnosis insights this extends:
  - [[20260516_1504_symposium_substrate_right_rhetoric_wrong]] (sober rhetoric authorising the calibration drop)
  - [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] (extended to perspective-plus-pilot on all 4 cells)
  - [[20260516_1508_v5_last_rhetorical_round_before_experiments]] (PI pace flag → motivated the move from staged plan to direction context)
- Companion convention insight (how docs reference this plan): [[20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases]]
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume e386e7cb-3b2e-4604-abc2-921d1b6c973a` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
