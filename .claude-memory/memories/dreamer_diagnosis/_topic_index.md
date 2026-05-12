# _topic_index.md — `dreamer_diagnosis` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `dreamer_diagnosis` topic.

**Folder definition**: DreamerV3 failure investigation
**Insights**: 7
**Last updated**: 2026-05-12

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-12 | 17:54 | `20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned` | After 3-reviewer ✅ PASS on a 1033-line JAX re-implementation plan, user pivoted via PI call to sheeprl PyTorch direct; static review of plan-against-paper does not predict integration-layer execution success. |
| 2026-05-11 | 15:34 | `20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt` | Cell Z2 (paper-canonical twohot bins on top of Z1's zero-init) re-ran the no-predator task and fired H2 — partial fix. Reward MAE @ h=5: 0.386 → 0.277 → 0.177 (cumulative cascade −54% from A1; H1 < 0.15 narrowly missed by 0.027). Mechanistic prediction validated: training-time `model_reward_mae_neg` dropped 45% Z1→Z2. New residual pattern: long-horizon reward MAE compounds 0.18 → 3.05 across h=5 → h=50 — mechanistically points at GRU reset gate (§6 item 28) as the next candidate. |
| 2026-05-10 | 22:39 | `20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry` | Cell Z1 (zero-init reward+critic output layers, sheeprl candidate #4) re-run on the simplest food-only task fired H2 — partial fix. Reward MAE @ h=5 dropped 0.386 → 0.277 (28%). Mechanistic headline: training-time positive-reward MAE improved 49% but negative-reward MAE only 14%. The pos/neg asymmetry directed the next fix candidate (twohot bin range, §6 item 2) since negative rewards of magnitude > 20 sit outside our head's representable bin support. |
| 2026-05-09 | 15:35 | `20260509_1535_conventional_fixes_battery_verdict_predator_refute` | 2-cell DreamerV3 conventional-fixes mini-battery (n113, 700k env-steps each): top-2 ranked conventional causes (reward-scale + replay-ratio) refuted on predator (Cell A2 H₀, survival 27 ≈ unmodified anchor). NoPred (Cell A1) collapse prevented by replay-ratio fix but stuck at survival 106 starvation equilibrium (H₂-with-caveat). Predator blockage is downstream of the reward head. |
| 2026-05-09 | 15:34 | `20260509_1534_wm_reward_head_localized_failure_a1` | Offline WM-imagination diagnostic on Cell A1's NoPred checkpoint pinpoints the failure: encoder/decoder + continuation head are sound, but reward MAE 0.386 (2.6× threshold) at h=5 even on the simplest food-only task. Localizes the bottleneck to the reward head specifically. |
| 2026-05-08 | 14:32 | `20260508_1432_probe_refutes_imagined_death_absence` | Imagined-rollout probe refuted the structural "WM can't imagine death" hypothesis (img_h15 = 0.27, NOT zero) and the priority-2 fix (cont_loss_weight=10 moved metric, not behavior). New hypothesis: imagined deaths are miscalibrated in time and per-action. |
| 2026-05-08 | 14:31 | `20260508_1431_diagnostic_battery_refutes_four_fixes` | DreamerV3 4-cell diagnostic battery (2026-05-07) refuted lever-only, curriculum-only, no-homeostatic-reward, and smoother-ramp as fixes for the 5×5 hypervigilance failure. |

---

## Change history

- 2026-05-11: Added 1 insight from the Z2 verdict session: `20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt` (paper-canonical bins H2 partial fix; cumulative cascade −54%; mae_neg 45% drop validates mechanism; new long-horizon-compounding residual selects GRU reset gate as next candidate). No new tags.
- 2026-05-10: Added 1 insight from the dreamer sheeprl-comparison + zero-init session: `20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry` (Cell Z1 H2 partial fix; pos/neg reward-MAE asymmetry directs next candidate toward bin-range fix). No new tags (all reused: dreamer, learned_lesson, decision, refutation).
- 2026-05-09: Added 2 insights from the conventional-fixes battery investigation: `20260509_1535_conventional_fixes_battery_verdict_predator_refute` (experimental verdict — top-2 conventional causes refuted on predator) and `20260509_1534_wm_reward_head_localized_failure_a1` (offline WM diagnostic localizes the failure to the reward head specifically).
- 2026-05-08: Folder created. Added 2 insights from the dreamer hypervigilance investigation.
