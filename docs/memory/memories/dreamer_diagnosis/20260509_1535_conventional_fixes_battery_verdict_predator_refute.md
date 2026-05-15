---
id: 20260509_1535_conventional_fixes_battery_verdict_predator_refute
date: 2026-05-09
time: "15:35"
folder: dreamer_diagnosis
tags: [dreamer, hypervigilance, refutation, decision]
summary: "2-cell DreamerV3 conventional-fixes mini-battery (n113, 700k env-steps each): top-2 ranked conventional causes (reward-scale + replay-ratio) refuted on predator (Cell A2 H₀, survival 27 ≈ unmodified anchor). NoPred (Cell A1) collapse prevented by replay-ratio fix but stuck at survival 106 starvation equilibrium (H₂-with-caveat). Predator blockage is downstream of the reward head."
related: ["20260508_1431_diagnostic_battery_refutes_four_fixes", "20260508_1432_probe_refutes_imagined_death_absence", "20260509_1534_wm_reward_head_localized_failure_a1"]
session_origin: claude_code
session_label: "dreamer_conventional_fixes_battery_2026-05-09"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# DreamerV3 conventional-fixes battery: top-2 causes refuted on predator, partial NoPred win

## Key conclusion
The 2-cell mini-battery designed from professor-rl-bayesian-dl's top-3 ranked conventional causes refutes the joint conventional-fix hypothesis on the predator (hypervigilance) task. Cell A2 (Predator + `replay_ratio: 0.0625` + `death_penalty: 1`) lands at survival 27, mean_advantage −0.19 — pixel-identical to the unmodified `owz29n3t` anchor — so neither reward-scale mismatch nor replay-ratio over-training was actually the bottleneck on predator. Cell A1 (NoPred + `replay_ratio: 0.0625`) repaired the published modal-trace `3zjhap9w` collapse pattern (mae asymmetry closes rather than widening; mean_value drifts less negative over training) but saturates at survival 106 with T_starv 0.60 — a stable starvation equilibrium that the §6.3 verdict matrix did not anticipate. Combined reading: predator has a structural component beyond conventional knobs; the failure mode is downstream of the reward head.

## Evidence, measurements, facts
- Design: `docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md` (Framing B chosen: A1 single-knob test on the published failure-template task, A2 kitchen-sink on the project-relevant task; deferred Cell B = Predator + rr=0.0625 only with dp=100 unchanged).
- Configs (1-line deltas each): `configs/models/dreamer_v3_rr06.yaml` (`replay_ratio: 0.5 → 0.0625`), `configs/experiment/dreamer_diagnostic/01-PredInterval3_NutGain18_DeathPenalty1.yaml` (`body.death_penalty: 100 → 1`).
- Cell A1 (`czfnljf0`, NoPred + rr=0.0625, n113:0, 700k env-steps, 1h38m wall-clock): survival 106, T_starv 0.60. No `3zjhap9w` collapse signature: mae asymmetry gap *closed* (final 0.06, vs `3zjhap9w`'s widening to 0.6); mean_value drifts *less* negative over training. Headline regime: stable starvation equilibrium. Verdict row: §6.3 H₂(A1)-with-caveat (collapse-prevention confirmed, but starvation-equilibrium ≠ competence; threshold T_STARV_F ≤ 0.30 broken at 0.60).
- Cell A2 (`houwr6js`, Predator + rr=0.0625 + dp=1, n113:1, 700k env-steps, 28m wall-clock): survival 27 (vs unmodified anchor `owz29n3t` 30.4 — within noise), mean_advantage = −0.19 (pixel-identical, ±0.005), mae gap 2.18. Critic well-calibrated to the new dp=1 reward scale (mean_value ≈ achievable return) — so the dp=1 fix did NOT break value learning, the failure is elsewhere. Verdict row: §6.3 H₀(A2) clean refutation; R-A2a fires on every clause by 3–7× margin.
- Pre-flight audit pass: `docs/reviews/env_config_audit_dreamer_conventional_fixes_battery.md` (4 items PASS: get_mandatory discipline, single-line delta verification, tag uniqueness, modality fingerprint identical across cells).
- Diary `training-done` rows logged at 13:36 by `4ae5401e/training-runner` then updated by `4ae5401e/experiment-analyzer`.
- Combined verdict §6.3 row hybrid: "H₂(A1), any A2" (extend A1 to 1.5M to disambiguate slow-learning from asymptotic-equilibrium) ∩ "H₁(A1), H₀(A2)" (predator has structural component). The A2 saturation signal (window-by-window std *falling*, mean_value plateau, advantage saturated at −0.19) is robust — extending A2 would not change the verdict.
- Hand-off triggered the offline WM diagnostic (sibling insight `20260509_1534`) which then localized the predator-side failure to the reward head specifically.

## Decisions and actions
- Promote `replay_ratio: 0.0625` to default (refutes published modal-trace collapse on NoPred at single-seed; multi-seed confirmation deferred to a separate experiment).
- Demote `death_penalty: 1` from "candidate fix" to "calibrationally safe" — Crafter recipe transfers without breaking value learning, but it does NOT improve predator survival.
- Park the deferred Cell B (Predator + rr=0.0625 only, dp=100 unchanged) — the offline WM diagnostic on A1 produced a stronger localization than Cell B would have.
- Pivot to component-level diagnostics on the predator pathology: offline WM-imagination test (now done — see sibling insight `20260509_1534`).
- For future hypervigilance experiments on Dreamer: reward-head class-balanced loss (originally priority 4 in `dreamer_hypervigilance_learning_failure.md`) is the next priority intervention, supported by the offline-test reward-MAE finding.

## Open questions and follow-ups
- Will an extended A1 run (1.5M env-steps) escape the survival-106 starvation equilibrium, or is this asymptotic on `replay_ratio: 0.0625`?
- Does multi-seed A1 confirm the no-collapse claim, or does `3zjhap9w` collapse re-appear at different seeds?
- Does the reward-head class-balanced loss intervention (next planned step) actually move predator survival, or is the structural component on predator something else entirely?

## References
- Design + Results + Analysis + Conclusions: `docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md`.
- Professor's analysis ranking the top-3 conventional causes: `docs/project/critiques/dreamer_conventional_failure_modes_for_our_setup.md`.
- Strip-down direction memo: `docs/project/directions/dreamer_minimum_viable_strip_down.md`.
- Sibling insight (same session): `20260509_1534_wm_reward_head_localized_failure_a1` — the offline WM diagnostic that localized the failure motivated by this verdict.
- Parent insights (refuted at the start of the diagnostic chain): `20260508_1431_diagnostic_battery_refutes_four_fixes`, `20260508_1432_probe_refutes_imagined_death_absence`.
- WandB runs: `czfnljf0` (Cell A1), `houwr6js` (Cell A2). Group `dreamer_conventional_fixes`. Reference baseline: `owz29n3t` (E1 anchor, prior probe battery).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
