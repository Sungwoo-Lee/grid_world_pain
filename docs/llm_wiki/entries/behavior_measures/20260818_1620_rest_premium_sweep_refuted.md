---
id: 20260818_1620_rest_premium_sweep_refuted
date: 2026-08-18
time: "16:20"
folder: behavior_measures
tags: [refutation, design, decision, learned_lesson, hypervigilance]
summary: "REFUTED: making uninterrupted rest valuable does NOT drive an injured agent to the refuge bush. 10 arms spanning a 1x-129962x rest-streak premium (injured window held matched at 13-15 steps) show injured bush use FLAT vs premium at both ~20M (rho=+0.10, p=0.78) and ~36-40M steps (rho=-0.19, p=0.60). The anti-hypervigilant gap survives everywhere: uninjured minus injured cover use is POSITIVE in all 10 arms at both depths (+1.9 to +12.2 pp). Cover use does rise with training (13.5->28.0%), just never more when injured."
related: ["20260710_1635_bush_hiding_metastable_dwell_measure", "20260805_0118_regime_matched_probe_and_run_sweep_probe_dir", "20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest", "20260818_1622_yaml_list_replace_and_runtime_env_test"]
session_origin: claude_code
session_label: "rest-premium sweep + no-hiding-predator replicate + log_code fix"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# Re-pricing rest continuity does not convert healing pressure into cover-seeking

## Key conclusion
Injury heals only via the Rest action, so an injured agent freezes-and-heals in the open instead of travelling to the refuge bush ([[20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest]]). The intervention tested here: make an INTERRUPTED rest expensive (raise `recovery_accel_rate`, so a long unbroken streak is worth far more than a restarted one), which should make a SAFE place to complete the heal — the refuge bush, which predators cannot enter — worth travelling to. It does not work. Across five orders of magnitude of interruption cost, injured cover use is flat, and the injury-suppression effect is untouched.

## Evidence, measurements, facts
- Design: 10 arms on bush-refuge basic04 (predator pounce; the bush blocks the pounce). `recovery_base_rate` was SOLVED per arm as `recovery_accel_rate` rose so every arm still sheds injury 70 in ~13-15 rest steps — the injured window is MATCHED and premium is the only variable (1x, 11x, 38x=default regime, 119x, 323x, 512x, 1207x, 3815x, 19683x, 129962x). Configs `5e170a1`; probe data `results/eval/avoidance/restprem/`.
- KEY TEST (converged = last 25% of checkpoints, bush-refuge-matched probe), injured bush use vs log premium, Spearman:
  - at ~20M steps: rho=+0.10, p=0.78 (range 5.7-11.0%)
  - at ~36-40M steps: rho=-0.19, p=0.60 — if anything slightly NEGATIVE
- Anti-hypervigilant gap (uninjured minus injured, wandering-rabbit): POSITIVE in **all 10 arms at both depths**; at depth +1.9 to +12.2 pp. The suppression is robust to the intervention.
- Not a floor effect: cover use grew strongly with training (a01 uninjured wander 13.5% -> 28.0%; predator conditions 41-62%). The agents learn to hide; they just never hide MORE when injured.
- Secondary hint (not robust): uninjured cover use trends DOWN with premium (rho=-0.62, p=0.054), i.e. a very high rest premium may slightly reduce cover use overall.
- Noise caveat: n=1 seed per arm and bush-hiding is metastable. Arm a03 (38x) read ~1/3 of its premium-neighbours a02/a04 — between-arm scatter exceeds the effect being sought.

## Decisions and actions
- Verdict: the heal-by-rest confound is NOT fixable by re-pricing rest continuity. The agent still rests where it stands.
- Next hypothesis (user + collaborator): MOVING IS DANGEROUS. The scene spawns 2-12 ambush `hiding_predator` resources (damage 15-45), so an injured agent one hit from death may rationally freeze rather than cross open ground. A 10-arm no-hiding-predator replicate was launched 2026-08-16 (configs `cfc0293`, see [[20260818_1622_yaml_list_replace_and_runtime_env_test]]); probe sweep spec `d0da3ce`.
- A frozen-injury PROBE remains rejected as off-distribution (agents never trained under non-healing injury) — any decoupling must happen in the TRAINING env.

## Open questions and follow-ups
- Does removing ambush risk free the injured agent to travel to cover? (running)
- If that also fails, the remaining levers change WHERE healing can happen (e.g. heal only in the refuge) rather than what it costs — a more invasive design change worth a proper design pass.
- Single-seed arms limit sensitivity; a seed-replicated subset around the informative middle (119x-3815x) would be needed before any positive claim.

## References
- Parent finding: [[20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest]]; matched-probe method [[20260805_0118_regime_matched_probe_and_run_sweep_probe_dir]]; metastability caveat [[20260710_1635_bush_hiding_metastable_dwell_measure]].
- Data + figures: `results/eval/avoidance/restprem/` (per-arm FIG_bush_dwell.png + SUMMARY_premium_vs_bushdwell.png).
- Raw conversation: synced via `./sync-agent-data.sh claude push`; `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260818_1622_yaml_list_replace_and_runtime_env_test]] (config_system, 2026-08-18) — To REMOVE an item from a YAML list (resources/entities/obstacles) via extends:, 
<!-- END BACKLINKS -->
