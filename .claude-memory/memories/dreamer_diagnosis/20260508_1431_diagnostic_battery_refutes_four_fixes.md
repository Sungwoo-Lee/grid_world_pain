---
id: 20260508_1431_diagnostic_battery_refutes_four_fixes
date: 2026-05-08
time: "14:31"
folder: dreamer_diagnosis
tags: [dreamer, hypervigilance, decision, learned_lesson]
summary: "DreamerV3 4-cell diagnostic battery (2026-05-07) refuted all four candidate fixes (lever-only, curriculum-only, no-homeostatic-reward, smoother-ramp) for the 5×5 hypervigilance failure. Each cell's pre-registered refutation criterion fired."
related: []
session_origin: claude_code
session_label: "dreamer_hypervigilance_investigation_2026-05-07"
importance: high
status: settled
supersedes: []
raw_source: _archive/raw_conversations/20260508_1431_diagnostic_battery_refutes_four_fixes.md
raw_completeness: full
---

# Dreamer 4-cell diagnostic battery refutes curriculum, levers, no-homeostatic, smoother-ramp

## Key conclusion
Curriculum, lever fixes (entropy_scale + cont_loss_weight), no-homeostatic reward regime, and a smoother difficulty ramp are all NOT the structural cause of DreamerV3's failure on `01-5X5_PredInterval3_NutGain18`. Every cell's pre-registered refutation criterion fired. The surviving inferred mechanism after this battery: WM continuation head fails to predict death events in imagined rollouts → imagined trajectories never terminate → actor optimizes against returns missing the −100 death penalty → mean_advantage is degenerate → policy stays at uniform (this hypothesis was REFUTED in turn by the next battery — see `20260508_1432_probe_refutes_imagined_death_absence`).

## Evidence, measurements, facts
- Battery design: `docs/experiments/active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md`. WandB group `dreamer_diagnostic_battery`. All 4 cells: seed 0, num_envs 16, 700k episodes (cells A/C/D) or 766k (cell B), node 114 cuda:0–3.
- Cell A — Lever-only (`2yti69a5`): R-A1 fired. Final survival 30.10±0.34, mean_entropy 1.66, mean_advantage −0.190 std 0.009, Term_Injury 99% — bit-for-bit qont5dac signature.
- Cell B — Curriculum-only (`d8fyau0c`): R-B2 fired. Stage 0 unfroze cleanly without levers (Q4 Steps 144.7, advantage std 0.031); Stage 1 reached 79 steps / 70% predator-deaths; Stage 2 collapsed to 31.11±0.57 (≡ prior 95mpudwp Stage 2 collapse).
- Cell C — No-homeostatic reward (`4fctrt42`): C-C1/C-C2 partially fail. Survival 43.70±1.64 — only cell where actor durably commits (entropy 0.55, advantage **+0.014**, FoodEaten 7.33), but predator-deaths still 99%. positive_buffer saturates by iter 150, REFUTING the user's empty-buffer prediction.
- Cell D — Smoother ramp (4-stage curriculum: food → very_slow → medium → full) (`q3ph89rq`): R-D1 fired. Stage 2 (predator_medium with `move_interval=5, damage=[10,30]`) clearly worked (Steps 58→83, Term_Injury 0.86→0.76); Stage 3 (full predator) collapsed within ~3,200 episodes to 32.25±1.01.
- 2×2 contrast: A vs B isolates curriculum vs levers; both fail. Combined (95mpudwp from prior session) also fails. C tests reward regime; survival doubles but still mostly dies. D tests ramp gradient; intermediate stage works, full task does not transfer.
- The collapse signature is identical across all "failing" cells: mean_advantage clusters at −0.18 to −0.19, mean_entropy at 1.65–1.67, Term_Injury at 99%.

## Decisions and actions
- Pre-registered refutation criteria fired in 4/4 cells → conclusion: NONE of these levers/regimes is the structural fix.
- Surviving inferred mechanism flagged for direct probing: "WM cannot imagine death events under full-predator dynamics" → priority-1 next experiment is the imagined-rollout termination probe.
- Design doc §10–§12 filled in with FLAGGED-FOR-FOLLOW-UP marker (commit `9a614c8`).
- Configs landed at `configs/experiment/dreamer_diagnostic/{03_predator_full_no_homeostatic.yaml, curriculum_smooth/{01..04}_*.yaml}`, schedule at `configs/continual/dreamer_curriculum_smooth_4stage.yaml`, design doc at `docs/experiments/active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md`.

## Open questions and follow-ups
- Cell C revealed an unexpected actor-commit pattern (positive advantage, low entropy, but high death rate) — what value function did the actor commit to under all-negative reward? Worth probing in a future ablation.
- Does the `cont_loss_weight=5` lever in cells A/D have any non-zero effect, or does any value of the weight (1, 5, 10) leave imagination indistinguishably broken? See `20260508_1432_probe_refutes_imagined_death_absence` for the answer (10 also fails).

## References
- Design doc + Launch Manifest: `docs/experiments/active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md` (§3 manifest, §10 plumbing, §11 results, §12 conclusions FLAGGED).
- Related insight: `20260508_1432_probe_refutes_imagined_death_absence` (the probe-battery follow-up that refuted the surviving hypothesis from this battery).
- Original failure-mode diagnosis: `docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md` (the qont5dac analysis that motivated the battery).
- Working file with raw extractions: `tmp/20260508_dreamer_battery_analysis.md`.
- WandB runs: `2yti69a5`, `d8fyau0c`, `4fctrt42`, `q3ph89rq` (all in group `dreamer_diagnostic_battery`).
- Why a new folder: closest existing folder is `subagent_engineering` (about agent infrastructure). The other parallel session created `nmn_diagnosis` for a sibling investigation; this insight is the symmetric Dreamer-architecture-failure-investigation entry. `dreamer_diagnosis` is needed for this and future Dreamer failure-mode investigations and parallels the `nmn_diagnosis` sibling cleanly.
- raw_source link is local-only (archives are gitignored — broken link on a fresh clone). The same archive file backs all 4 insights from this capture session.
