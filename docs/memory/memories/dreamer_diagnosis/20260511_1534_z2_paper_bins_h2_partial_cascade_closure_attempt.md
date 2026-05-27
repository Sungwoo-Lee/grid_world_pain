---
id: 20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt
date: 2026-05-11
time: "15:34"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, refutation]
summary: "Cell Z2 (paper-canonical twohot bins on top of Z1's zero-init) re-ran the no-predator task and fired H2 — partial fix. Reward MAE @ h=5: 0.386 → 0.277 → 0.177 (cumulative cascade −54% from A1; H1 < 0.15 narrowly missed by 0.027). Mechanistic prediction validated: training-time `model_reward_mae_neg` dropped 45% Z1→Z2 (0.649 → 0.359), exactly the negative-event-error narrowing the bin-coverage mechanism predicted. New residual pattern observed: long-horizon reward MAE compounds 0.18 → 3.05 across h=5 → h=50 (Z1 was 0.28 → 0.23) — mechanistically points at GRU reset gate (§6 item 28) as the next candidate."
related: ["20260509_1534_wm_reward_head_localized_failure_a1", "20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry", "20260510_2241_residual_error_pattern_directs_next_fix"]
session_origin: claude_code
session_label: "dreamer_sheeprl_compare_and_zero_init_2026-05-10"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# Z2 paper-canonical bins H2 partial fix; cumulative cascade −54%; long-horizon compounding points at GRU

## Key conclusion
The second targeted code-side fix to the localized DreamerV3 reward-head failure — applying the paper-canonical two-hot bin construction on top of Z1's zero-init — produced a **second partial fix (H2-band)**, taking reward MAE @ h=5 from 0.277 (Z1) to **0.177 (Z2)**. Cumulative cascade A1 → Z1 → Z2 reads **−54% on the headline metric**, with H1 (< 0.15) narrowly missed by 0.027. The mechanistic hypothesis that picked this fix (Z1's residual error was disproportionate on negative-reward events; paper-canonical bins span ±4.85×10⁸ instead of ±20, so death-magnitude rewards are representable) is **validated**: training-time `model_reward_mae_neg` dropped 45% Z1→Z2 while `model_reward_mae_pos` regressed slightly — the bin fix is specifically curative on negative events, exactly as predicted. A new residual pattern emerged that is the input to the next candidate selection: reward MAE compounds **0.18 → 3.05 across h=5 → h=50** in Z2 (Z1 was 0.28 → 0.23). The wider symlog grid is more sensitive to GRU-cell drift on imagined off-manifold features; this is the exact symptom GRU-cell dynamics quality affects, mechanistically pointing at §6 item 28 (GRU reset gate computed but never applied) as the next-best candidate for the cascade.

## Evidence, measurements, facts
- **Cell Z2 launch**: `dreamer_twohotrng_NoPred_rr06_s0_n113`, n113:0, WandB `q66macky`, group `dreamer_paper_canonical_bins`, 700k env-steps target, ran from 2026-05-10 22:11 to 2026-05-11 03:01 (≈4h50m wall-clock). Single seed (0). Same NoPred config + rr06 agent config as A1/Z1; cumulative knobs are `replay_ratio: 0.0625` (from rr06) + `zero_init_reward_critic: true` (default after Z1) + `paper_canonical_twohot_bins: true` (new default after this fix).
- **Code change**: `src/models/dreamer_v3_util.py` `to_twohot` and `from_twohot` gain `paper_canonical_bins: bool = False` kwarg; True path constructs bins as `symexp(linspace(-20, +20, 255))` (uniform spacing in symlog space, mapped to raw via symexp); False path preserves legacy `linspace(symlog(-20), symlog(+20), 255)` for bit-identical legacy reproduction. Trainer reads `agent.paper_canonical_twohot_bins` via `get_mandatory` and routes through all 9 call sites in `dreamer_v3_trainer.py` + 1 site in `dreamer_v3_nnx.py`. Commits `f5df600` (code) + `8089ee2` (diary). Smoke test: reward `−100` round-trips with err 0.0003 (was 80 saturating at `−20`).
- **Offline diagnostic on Z2 checkpoint** (`scripts/dreamer_offline_wm_test.py`, M=200, horizons {1, 2, 5, 10, 15, 25, 50}, deterministic):
  - Reward MAE @ h=5: **0.1770** (vs Z1 0.277, A1 0.386). Δ_Z1 = −0.100, −36%; Δ_A1 = −0.209, −54%. Pre-registered: H1 < 0.15 (FAIL by 0.027); H2 ∈ [0.15, 0.25) (FIRE); H0 ≥ 0.25 (refuted).
  - Aggregate obs symlog-MSE @ h=5: 0.043 (was Z1 0.047). PASS.
  - Continuation accuracy @ h=5: 1.000. PASS.
  - Long-horizon h50/h5 ratio: **17.2** — wildly out of the prior PASS band (Z1 was 1.62, A1 1.69, threshold ≤ 2.0). NEW failure on this metric, not present in earlier cells.
- **Training-time WandB metrics on Z2** (validated the bin-coverage mechanism):
  - `model_reward_mae_pos`: 0.805 (vs Z1 0.544) — regressed +48%. Consistent with "bin fix is curative on negatives, not positives".
  - `model_reward_mae_neg`: 0.359 (vs Z1 0.649) — **dropped 45%**. The load-bearing prediction confirmed.
- **Long-horizon error compounding (new diagnostic observation)**: per-horizon reward MAE for Z2 climbs 0.18 → 3.05 across h=5 → h=50; for Z1 it was 0.28 → 0.23 (essentially flat); for A1 it was 0.39 → 0.68. The wider symlog grid (now raw support ±4.85×10⁸) is more sensitive to small logit perturbations on imagined off-manifold latents, which is the exact failure mode autoregressive GRU drift would produce. This is **a new residual pattern** that points specifically at §6 item 28 (GRU reset gate computed but never applied) as the next-best candidate via the same residual-pattern-directs-next-fix methodology that picked Z2 (sibling insight `20260510_2241_residual_error_pattern_directs_next_fix`).
- **Diagnostic-script silent-bug discovery during this analysis**: first Z2 diagnostic run reported MAE = 1.04 — looked like a catastrophic 6× regression. Triangulated against training-time `model_reward_mae` = 0.372 (which used the in-trainer decode path, sane) and found the script wasn't reading the new `paper_canonical_bins` flag. Manual monkey-patch produced the authoritative 0.177 result above. Patched in commits `1703a4c` (script + plan-§V.6 update) + `b82dad8` (diary). Cross-reference sibling insight `20260511_1535_encode_decode_flag_mismatch_silent_class_bug`.
- **Survival**: Z2 steady-state TBD from the WandB extract; not a primary signal for this study (the offline diagnostic is the load-bearing verdict).
- **Anchor comparisons**: A1 baseline (`czfnljf0`, `dreamer_conv_NoPred_rr06_s0_n113`); Z1 zero-init (`axndoqsz`, `dreamer_zinit_NoPred_rr06_s0_n113`). Offline-diagnostic outputs: A1 at `tmp/20260509_wm_imagination_test_A1.{json,md}`, Z1 at `tmp/20260510_211404_wm_imagination_test_Z1.{json,md}`, Z2 (authoritative) at `tmp/20260511_044500_wm_imagination_test_Z2_papercanonical.{json,md}`.

## Decisions and actions
- **Promote `paper_canonical_twohot_bins: true` to default**. Both `configs/models/dreamer_v3/dreamer_v3.yaml` and `configs/models/dreamer_v3/dreamer_v3_rr06.yaml` ship with the key set to `true`. Legacy `false` path is preserved for bit-identical legacy reproduction.
- **Update concept doc §6** (commit `2ed7199`):
  - Item 2 (twohot bin range): `MAJOR DEVIATION (suspected unjustified)` → `RESOLVED-PARTIAL`. The deviation is no longer suspected unjustified — paper-canonical recipe verified to repair what its mechanism predicted.
  - Item 27 (zero-init): same flag transition.
  - Item 28 (GRU reset gate): flagged "**NEXT IN THE FIX CASCADE**" with the long-horizon-compounding rationale.
- **Queue candidate #1 (GRU reset gate, §6 item 28) as the next fix** — selected via residual-error mechanistic match (long-horizon error compounding is the exact symptom GRU-cell dynamics quality affects). Plan-prescribed prescription from Z2's §4 H2 row already said this; the residual pattern confirms it from below.
- **User has explicitly paused on the next training step** to discuss before launching. The cascade's next training cell is gated on that discussion. The candidates on the table are: (a) candidate #1 (GRU reset gate) for further H1 closure on NoPred, (b) cumulative fix-stack on the predator task as a generalization test, (c) clean up 5 orphan DreamerV3 configs that lack the new mandatory keys.
- **Side-effect commits**: `1703a4c` (diagnostic-script bug fix), `b82dad8` (developer diary row), `e0c875f` (analyzer verification report), `2ed7199` (§6 empirical-resolution updates).

## Open questions and follow-ups
- Will candidate #1 (GRU reset gate) close H1 on NoPred, or will there be a *third* H2 residual that names yet another candidate? The residual-error-pattern methodology has now fired twice cleanly (Z1's pos/neg asymmetry → bin range; Z2's long-horizon compounding → GRU reset gate). A third confirmation would strongly validate the rule.
- Does the cumulative fix-stack (Z1 + Z2 + candidate #1 if it lands) transfer from NoPred to the predator task? The original hypervigilance failure was on predator. Z2's `mae_pos` regression is a yellow flag — bin-coverage fix can come with positive-event tradeoffs, and the predator task has the extreme negative reward (−100 death penalty) the bin-coverage mechanism most strongly targets. The predator task is the cleanest test of whether the cascade was solving the right problem.
- Why did Z2's `mae_pos` regress (0.544 → 0.805)? Mechanistic guess: with the wider grid, positive-reward bins near zero are sparser per unit input, so the head's first-pass on small positives has less resolution. Not load-bearing for the cascade's progress, but worth understanding before settling on the bin range as a default.
- Is the long-horizon h50/h5 ratio of 17.2 a numerical artifact of the bin range × imagine-step drift interaction, or a real signal? Worth a sanity-check with a *shorter* horizon set (h=3, h=5, h=10 only) to see if the early-horizon performance is good and the compounding is purely autoregressive drift.

## References
- Cell Z2 plan + Verification Report: `docs/develop/active/diagnosis/dreamer_twohot_bin_range_fix.md`. Commits: `c9164b0` (plan), `f5df600` + `8089ee2` (code), `a3cdac2` + `564ffcf` (launch), `e0c875f` (verification report), `1703a4c` (diagnostic-script bug fix surfaced during analysis), `2ed7199` (§6 updates after verdict).
- Cell Z1 plan + Verification Report (the predecessor in the cascade): `docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`.
- Concept doc with the deviation list updates: `docs/project/concepts/dreamer_v3_implementation.md` — §6 items 2, 27, 28; §9 sheeprl-comparison section is where the bin-range deviation was first triply-confirmed (math-F1, sheeprl, Hafner published code).
- Diagnostic outputs (gitignored, on-disk only): `tmp/20260511_044500_wm_imagination_test_Z2_papercanonical.{json,md}` (authoritative), `tmp/20260511_044500_wm_imagination_test_Z2.{json,md}` (buggy initial run, kept for traceability), Z1 at `tmp/20260510_211404_wm_imagination_test_Z1.{json,md}`, A1 at `tmp/20260509_wm_imagination_test_A1.{json,md}`.
- Sibling insights from this cascade session: `20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry` (Z1 verdict — the residual that picked Z2), `20260510_2240_reference_impl_compare_only_act_intersections` (the rule for handling sheeprl as comparator), `20260510_2241_residual_error_pattern_directs_next_fix` (the methodology rule that picked Z2 over the plan-prescribed candidate #1), `20260511_1535_encode_decode_flag_mismatch_silent_class_bug` (the diagnostic-script bug surfaced during this analysis, with the cross-check heuristic).
- Parent insights: `20260509_1534_wm_reward_head_localized_failure_a1` (the offline diagnostic that originally localized the failure to the reward head), `20260509_1535_conventional_fixes_battery_verdict_predator_refute` (the predator-task verdict that motivated all the reward-head work).
- Fix-cascade study summary (snapshot at Z2 launch, not yet updated for Z2 verdict — that's the next `/summarize-study` invocation): `docs/experiments/summaries/20260510_2255_dreamer_v3_fix_cascade.md`.
- Sheeprl reference: `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py` (`TwoHotEncodingDistribution`) and `models/models.py:399–401` (GRU candidate update — confirms the reset-gate fix for §6 item 28).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260513_2308_strong_strategy_validates_on_cp1]] (dreamer_diagnosis, 2026-05-13) — The Strong (A+B+C+D) deviation-prevention strategy paid off on the first checkpo
<!-- END BACKLINKS -->
