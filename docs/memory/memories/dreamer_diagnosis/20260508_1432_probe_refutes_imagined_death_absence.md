---
id: 20260508_1432_probe_refutes_imagined_death_absence
date: 2026-05-08
time: "14:32"
folder: dreamer_diagnosis
tags: [dreamer, hypervigilance, decision, learned_lesson]
summary: "DreamerV3 imagined-rollout probe (2026-05-08) refuted the structural hypothesis: imagination predicts ~27% terminations within h=15 (not zero), but at step ~12 vs real ~23. Cont_loss_weight=10 raised the metric without raising survival. Working hypothesis flips: imagined deaths are miscalibrated in time and per-action, not absent."
related: ["20260508_1431_diagnostic_battery_refutes_four_fixes"]
session_origin: claude_code
session_label: "dreamer_hypervigilance_investigation_2026-05-08"
importance: high
status: settled
supersedes: []
raw_source: _archive/raw_conversations/20260508_1431_diagnostic_battery_refutes_four_fixes.md
raw_completeness: full
---

# Dreamer probe battery refutes "WM can't imagine death" structural hypothesis + priority-2 fix

## Key conclusion
The surviving hypothesis from the prior diagnostic battery — "WM continuation head fails to predict death events in imagined rollouts" — is **wrong**. The newly-implemented imagined-rollout probe shows that DreamerV3's WM does predict terminations in imagination (~27% within h=15 in failing cells), but at step ~12.6 vs real ~23, and uniformly across actions. The priority-2 fix `cont_loss_weight=10` (vs prior 5) successfully raised the imagined-termination fraction (peak 0.35 vs 0.30) but had **zero behavioral effect**: survival 30.4, entropy 1.66, advantage −0.19 — bit-identical to the unboosted anchor. New working hypothesis: imagined deaths are miscalibrated in time (predicted ~10 steps too early) and not differentiated per-action, leaving the value gradient anaemic but not zero.

## Evidence, measurements, facts
- Probe instrumentation: new `agent.imagined_rollout_probe` flag (default false → bit-identical), logs `WorldModel/imagined_termination_fraction_h{8,15}`, `imagined_first_term_step_mean`, `imagined_term_step_p{50,10,90}`, `imagined_real_term_step_mean`. Plan + impl: `docs/develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md`. Implementation commit `82a31ff`.
- Battery design: `docs/experiments/active/continual_learning/DREAMER_PROBE_BATTERY.md`. 4 cells, seed 0, num_envs 16, 700k episodes, node 114 cuda:0–3. WandB group `dreamer_probe_battery`.
- Cell E1 — Anchor + probe (`owz29n3t`, mechanism confirmation): final survival 30.4, mean_entropy 1.67, mean_adv −0.19, **img_h15 = 0.27** (NOT ≈ 0), img_h8 = 0.16, **img_first = 12.6, real_term = 23.1**. Mismatch: imagination predicts deaths ~10 steps earlier than reality.
- Cell E2 — Anchor + cont_loss_weight=10 + probe (`mruhypnu`, priority-2 test): img_h15 = 0.33 (peak Q50 in temporal: 0.35 vs E1's 0.30), img_h8 = 0.16. Survival 30.4, entropy 1.66, advantage −0.19 — **identical to E1 in behavior despite higher imagination-termination rate**.
- Cell E3 — 00-NoPred + probe (`xekunmbw`, positive control, in progress at ~500k env steps as of 2026-05-08 14:30): img_h15 = 0.02 (low — episodes really long, MaxSteps=500), real_term = 105 steps. Survival 316, entropy 1.02, advantage **+0.005**. Probe tracks reality: low termination predictions track long real episodes. Confirms probe instrumentation works; isolates mechanism to predator-task dynamics. **Caveat (added post-hoc): a near-identical config replication on the same NoPred task — `3zjhap9w` (`replay_ratio=0.5, entropy_scale=3e-4, cont_loss_weight=1`, agent-side seed=42; differs from E3 only by the `imagined_rollout_probe` flag) — reached the same commitment at 400–600k env steps and then COLLAPSED by 800k env steps to survival ~140 with 80% starvation. The "durably commits" claim about E3 is therefore at risk: NoPred competence may be transient, not asymptotic, on this configuration. E3 has not yet been trained past the `3zjhap9w` collapse boundary.**
- Cell E4 — 3-stage curriculum + levers + probe (`udqry11l`, replicate 95mpudwp): final = qont5dac signature (Steps 30.5, advantage −0.18, img_h15 = 0.28). Stage 0/1 had healthy progress (early advantage −0.56 with real_term=60 — long food-only episodes), Stage 2 collapsed identically to prior 95mpudwp. Probe shows imagined termination drops back to qont5dac level after Stage 2 transition.
- E1 temporal: img_h15 climbs 0.18 → 0.30 → 0.27 during training. The metric DOES move; it does not stay at zero.
- E2 temporal: img_h15 climbs 0.14 → 0.35 → 0.33. Confirms the cont_weight lever works on the metric; behavior is unchanged.

## Decisions and actions
- The "WM cannot imagine death" hypothesis is **refuted** by E1's img_h15 = 0.27 (the prior battery's surviving hypothesis was wrong).
- The "boost cont_loss_weight" priority-2 fix is **refuted** by E2 (lever moves metric, not behavior).
- E3 (positive control) confirms the probe instrumentation is valid — it tracks reality on a learnable task.
- Working hypothesis updated to: "imagined deaths are miscalibrated in time and per-action, not absent." Future probes need (a) per-action conditional `cont` predictions, and (b) to show where in the imagined trajectory the predicted death lands relative to the action that "caused" it.
- Priority-1 next experiment is no longer this probe — it has been completed. The new top-3 next experiments: (1) per-action `cont` instrumentation; (2) reward-head class-balanced loss (priority 4 from original diagnosis, now elevated); (3) checkpoint-based imagination probe — load rPPO's policy into Dreamer's imagine and see if termination calibration is policy-dependent or WM-dependent.
- 4 cells launched and committed (`7639c2b`). Audit-trail commit for E4 launch: `a9c6f34`.

## Open questions and follow-ups
- Why does cont_loss_weight=10 raise imagined-termination rate but not improve survival? If the WM is "more willing to predict termination," why doesn't the actor see more value-gradient signal? Hypothesis: terminations happen at the same imagined step regardless of action choice — i.e., the WM predicts "some imagined trajectory will die at step 12 no matter what action you took." Per-action conditional probe will answer.
- Is the "10 steps too early" miscalibration coming from the WM's autoregressive drift (states becoming OOD by step ~12)? Could be tested by comparing imagination-from-replay-state vs imagination-from-policy-rollout-state.
- Cell E3 hit positive advantage and 316-step survival — is the probe's healthy `img_h15 ≈ 0.02` causally linked to the learning success, or is it a side effect of long real episodes alone? Run a hypervigilance-task variant with reduced predator strength to disambiguate.
- **Does E3 follow `3zjhap9w` into collapse past 800k env steps?** Strong test of the "imagined-deaths-miscalibrated-in-time" hypothesis: rr05 shows the predicted self-confirming-pessimism signature without the probe (model_reward_mae_pos collapsing 0.15→0.05 while mae_neg deteriorates 0.37→0.60; mean_value drifts more negative while loss_critic falls — critic gets confident in pessimistic values; food intake AND hiding-predator hits both fall — the agent avoids everything). If E3 collapses with the same WM signature, the indirect-evidence story becomes very strong even without per-action conditional probe data.
- Is the rr05 collapse permanent at the 10M-step budget, or does it recover after re-learning? Decides whether NoPred Dreamer-competence is fundamentally transient or eventually stable.

## References
- Design doc + Launch Manifest: `docs/experiments/active/continual_learning/DREAMER_PROBE_BATTERY.md`.
- Probe plan + Implementation Report: `docs/develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md`.
- Prior diagnostic battery (refuted 4 fixes): `20260508_1431_diagnostic_battery_refutes_four_fixes`.
- Original failure-mode diagnosis: `docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`.
- WandB runs: `owz29n3t` (E1 anchor), `mruhypnu` (E2 cont10), `xekunmbw` (E3 NoPred), `udqry11l` (E4 curriculum) — all in group `dreamer_probe_battery`.
- Implementation commit: `82a31ff`. Battery configs commit: `7639c2b`. Audit-trail commit: `a9c6f34`.
- Post-hoc analysis of the older `3zjhap9w` NoPred + replay_ratio=0.5 run (started 2026-05-07 15:12:59, 24h+ wall-clock, 1.87M of 10M env steps as of 2026-05-08 14:30): `tmp/20260508_replayRatio05_NoPred_analysis.md`. Run is essentially a two-seed replication of E3 with no probe instrumentation. Shows the predicted self-confirming-pessimism signature (model_reward_mae_pos ↓ while mae_neg ↑, mean_value drifting more negative, all action types declining together) and refutes the asymptotic-stability claim about Dreamer on NoPred.
- raw_source link is local-only (archives are gitignored — broken link on a fresh clone). The archive is shared with the 3 sibling insights captured in the same session.
