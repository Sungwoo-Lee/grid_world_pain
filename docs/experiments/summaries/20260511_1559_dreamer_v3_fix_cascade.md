---
title: "DreamerV3 Fix Cascade — Z2 verdict landed (2026-05-07 → 2026-05-11)"
study: dreamer_v3_fix_cascade
generated: 2026-05-11T15:59
window: "2026-05-07 → 2026-05-11"
status: snapshot
---

# DreamerV3 Fix Cascade — Summary as of 2026-05-11 15:59 KST

> **Re-summary of the fix-cascade study.** The previous snapshot ([20260510_2255_dreamer_v3_fix_cascade.md](20260510_2255_dreamer_v3_fix_cascade.md)) captured the state with the second fix candidate (paper-canonical two-hot bins) still in flight. This snapshot updates with the verdict — partial fix, just short of the correctness threshold — and the new mechanistic finding that selects the next candidate. The prior summary is preserved as a historical snapshot; do not edit it.

> **One-paragraph summary.** The world model's reward head, which the prior diagnosis pinpointed as the single broken component on the simplest possible task, has now been repaired by **54%** through two paper-canonical code-side fixes applied in sequence. The first fix (initialise the reward and critic output layers to zero, like the published code does) gave a 28% reduction; the second fix (fix the two-hot bin layout so the head can represent rewards outside ±20 in raw space) gave another 36% reduction on top of that. Both fixes were *predicted* in advance by reading the residual error pattern from the previous cell and matching it against one specific candidate's mechanism. Both predictions held — positive-event prediction improved after fix one, negative-event prediction after fix two, exactly as the mechanisms said. We did **not** clear our correctness threshold; residual reward MAE is 0.18 against a target of 0.15. The new residual signature (the head's predictions diverge dramatically at long imagination horizons but are fine at short ones) is mechanistically explained by a *third* sheeprl-comparison candidate — a GRU reset gate that is computed but never applied to the candidate hidden-state update — which is now queued as the next rung of the cascade. Along the way, a critical silent-failure bug in the diagnostic script was discovered and patched: the script had been decoding trained-model outputs with the wrong bin layout, producing a fake catastrophic-regression reading that was caught only by cross-checking against the training-time logger.

> **This is a snapshot.** Future re-summaries should write a new dated file, not edit this one.

---

## 1. Study question

The previous study localised the DreamerV3 reward-head failure to one specific component. **This continuation study asks: can we repair that component by walking a sequence of paper-canonical code-side fixes, picking each next rung mechanistically from the residual error pattern left by the previous?** The constraints from the user, stated in earlier sessions:

- **Be paper-canonical**: each candidate fix is a known difference between our implementation and a reference (the paper, the published Hafner code, or the actively-maintained sheeprl third-party implementation). Don't invent project-specific repairs.
- **Be mechanistically motivated**: don't go down the list of candidates in order. After each fix, look at *what the diagnostic now shows is still wrong* and pick the candidate whose mechanism most directly explains that residual.
- **Stay reversible**: each fix is gated on a config knob with the new behaviour as default but a legacy bit-identical fallback.

The success criterion is the offline reward-prediction diagnostic, run on each trained checkpoint: predicted reward at imagination horizon 5 should be within ±0.15 of the true reward on average. The previous diagnosis showed this metric at 0.39 — 2.6× the tolerance — on the simplest possible task (food-only foraging, no predator).

---

## 2. Experiments completed this window

Rows 0a–0d cover the original four-experiment refutation chain that localised the failure (covered in the earliest summary, [20260509_1555_dreamer_v3_diagnosis.md](20260509_1555_dreamer_v3_diagnosis.md)). Row 1 covers the sheeprl-comparison doc-side event from the predecessor fix-cascade summary. Rows 2 and 3 are the in-window fix attempts; row 4 is the bug-fix side event that landed during row 3's verdict.

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed |
|---|---|---|---|---|---|
| **0a-d** *(prior anchors — see earliest summary)* | Diagnostic battery → probe battery → conventional-fixes battery → offline world-model imagination test | Why does DreamerV3 fail on our hypervigilance task? | Four obvious task-shaping fixes; an imagined-rollout instrumentation probe; the top-2 textbook conventional knobs; an offline frozen-checkpoint world-model audit. | First three refutation classes closed; offline diagnostic localised the failure to **the reward head specifically** — encoder, decoder, continuation head, and long-horizon dynamics all passed on the simplest task, only reward-prediction failed (MAE 0.39 vs tolerance 0.15). | Defined the load-bearing target for this continuation study. |
| **1** *(prior anchor — see predecessor 2255 summary)* | **Sheeprl reference-implementation comparison** *(doc-side event)* | Of the differences between our codebase and a high-quality independent DreamerV3 implementation (sheeprl), which are likely real bugs vs deliberate research choices vs framework idiom? | A senior-developer agent walked sheeprl side-by-side with our trainer, filtering framework idiom from algorithmic substance. | **Four new candidate deviations surfaced** beyond the existing 26-item deviation list: zero-init on reward+critic output layers, GRU reset gate not applied, missing critic self-EMA regularization term, prior+posterior heads with no hidden layer. User stated rule: document by default, act on those that intersect the live failure investigation. | Added candidates 27-30 to the deviation list; established the routing rule for any future reference-impl comparison. |
| **2** *(prior anchor — see predecessor 2255 summary)* | **Cell Z1 — zero-init reward + critic output layers** | Does removing the startup noise from the reward head's output (so it starts predicting nothing-yet instead of random noise) repair the reward head? | A single config knob added (default true) that initialises only the reward and critic *output* linear layers to zero; the rest of the network keeps the existing scheme. | **Partial fix.** Reward MAE fell from 0.39 to 0.28 (28% reduction); did not clear the 0.15 threshold. The agent's positive-reward predictions improved 49% in training; negative-reward predictions only 14%. | Promoted zero-init to default. The pos/neg asymmetry in the residual mechanistically named the next candidate. |
| **3** | **Cell Z2 — paper-canonical two-hot bins** *(2026-05-10 22:11 → 2026-05-11 03:01, no-predator task, 700,000 episodes, ~4 h 50 m)* | The reward head encodes its prediction over 255 discrete bins along the reward axis. Our bins spanned only `±20` in raw reward space (the published code's span `±4.85×10⁸`). The Z1 residual was concentrated on negative-reward events — exactly the regime where our narrow bin range made values like a death penalty of `−100` unrepresentable. Does fixing the bin construction close the remaining gap? | A second config knob (default true) re-built the bins as `symexp(linspace(−20, +20, 255))` — uniform spacing in symlog space, then mapped to raw — instead of the legacy `linspace(symlog(−20), symlog(+20), 255)`. Cumulative knobs at training time: replay-ratio 0.0625 + zero-init + paper-canonical bins. | **Partial fix again.** Reward MAE fell from 0.28 to **0.18**; cumulative reduction A1 → Z1 → Z2 is **−54%**. Clear threshold of 0.15 was narrowly missed by 0.027. Mechanistic prediction validated: training-time negative-event MAE dropped **45%** Z1→Z2 while positive-event MAE regressed slightly — exactly the curative-on-negatives signature the bin-coverage mechanism predicted. **New residual** observed: reward MAE at horizon 5 is 0.18, but at horizon 50 it climbs to 3.05. The wider symlog grid is more sensitive to recurrent-state drift on imagined off-manifold latents. | Promoted paper-canonical bins to default. Updated the deviation list — the bin-range and zero-init deviations are now marked partially-resolved with empirical results attached. The long-horizon error-compounding signature **directly fingers candidate #1 (GRU reset gate computed but never applied) as the next rung** — same mechanistic-residual-match heuristic that picked Z2. |
| **4** | **Diagnostic-script silent-bug discovery and patch** *(2026-05-11, code-side side event)* | During Z2's verdict analysis, the first diagnostic run reported reward MAE = 1.04 — looking like a 6× catastrophic regression. The number was caught and patched within the same analysis pass. | The diagnostic script had a single-line bug: it decoded the trained model's outputs using the legacy bin layout regardless of what the saved checkpoint's config said. So the trained model spoke the new bin layout, the script listened in the old layout, output was garbage. The patch: read the saved-config flag for the bin layout, pass it through to every decode call, fall back to legacy default for pre-fix checkpoints (Z1, A1) that didn't have the key. | Patched in commits `1703a4c` (script + plan note) + `b82dad8` (diary). Backward compatibility verified — re-running against Z1's checkpoint reproduces the prior authoritative number 0.277; re-running against Z2 reproduces the authoritative 0.177. | This is the **second instance** of the same bug class in this codebase (the first was the train.py orbax-vs-NNX checkpoint-restore skew, surfaced during the offline diagnostic build). The pattern is "auxiliary tooling falls out of sync with a knob-gated production change; failure is silent." The discriminating signal that catches it: cross-check at least one diagnostic metric against the training-time logger on the same checkpoint. Codified as a methodology rule for any future fix-cascade. |

---

## 3. Where this leaves the study

- **Two paper-canonical fixes shipped, both partial wins, cumulative −54% on the headline metric.** The cascade is converging by mechanism. Each fix's prediction held; each next-candidate selection was right.
- **Threshold not yet cleared.** Residual reward MAE on the simplest task is 0.18 against a 0.15 target. Close, but not done.
- **The mechanistic-residual-match heuristic has fired twice cleanly.** Z1's residual (positive-vs-negative asymmetry) pointed at the bin-range fix; Z2 confirmed it (45% negative-event improvement). Z2's residual (long-horizon error compounding) now points at the GRU reset gate fix. If a third confirmation lands, the rule is well-validated and can be carried forward to other algorithm investigations.
- **A working defensive practice emerged**: whenever a knob-gated change ships in production training, every auxiliary tool that decodes trained-model outputs must be updated in lockstep, AND every diagnostic should cross-check against the training-time logger on the same checkpoint. This is the second time we've hit a silent failure in tooling that fell out of sync; codifying the trip-wire prevents the third.
- **The cumulative-fix configuration on NoPred** sits at survival ~115 (vs A1's ~106 and Z2's TBD-but-on-trajectory). The original project goal — predator-task survival — has not yet been tested with the cumulative stack; that's the natural follow-up after the cascade closes on NoPred.

---

## 4. What's next (still pending decision)

1. **Queue candidate #1 (GRU reset gate) as the next training cell** *(`senior-developer` to plan, `developer` to implement, `training-runner` to launch)*. The mechanistic match is clean: long-horizon compounding is exactly the symptom GRU-cell dynamics quality affects. Fix scope: a one-line correction in `LayerNormGRUCell` (the `reset` gate is computed via sigmoid but the variable is then never multiplied into anything; paper formula is `cand = tanh(reset * cand)`). Same plan-then-implement-then-launch cycle as Z1 and Z2.
2. **Test the cumulative fix-stack on the predator task** *(`experiment-designer` to author, blocked on item 1's outcome)*. The reward-head investigation has been on the no-predator task throughout because that's where the failure was first localised. The original hypervigilance goal — surviving while a predator is in the world — has the strongest test of the bin-coverage mechanism specifically (death penalty `−100` is the cleanest case of "this reward is outside the legacy bin range and the head couldn't represent it before").
3. **Clean up the five orphan DreamerV3 agent configs** *(`senior-developer` to plan, no blocking dependency)*. Five legacy configs (`dreamer_v3_probe.yaml`, `dreamer_v3_curriculum*.yaml`, `dreamer_v3_probe_cont10.yaml`, `neuromodulated_dreamer_v3.yaml`) lack the two new mandatory keys and would error if used. Each needs a one-line addition.
4. **Decide on candidates #29 and #30** *(deferred until #28 lands)*. Candidate #29 (missing critic self-EMA regularization) and #30 (prior+posterior heads have no hidden layer) are real paper-canonical deviations but neither is the most-targeted match for any current residual. Queue order depends on what #28's outcome reveals.
5. **Write up the deviation closure log when the cascade settles** *(later, after #28 / predator transfer / orphan cleanup)*. Each `RESOLVED-PARTIAL` entry in the §6 deviation list should eventually carry the experiment IDs and metric deltas that closed it.
6. **Optional: 3-seed sweep at the cumulative-fix configuration** *(`experiment-designer`, lower priority)*. All cascade cells so far are single-seed; the survival improvements (106 → 115 from A1 → Z1) carry seed-noise we have not characterised in this regime.

---

## 5. Links to authoritative documents

### Predecessor summaries (read these for the pre-window arc)

- [20260510_2255_dreamer_v3_fix_cascade](20260510_2255_dreamer_v3_fix_cascade.md) — fix-cascade snapshot with Z2 in flight (immediate predecessor).
- [20260509_1555_dreamer_v3_diagnosis](20260509_1555_dreamer_v3_diagnosis.md) — original four-experiment refutation chain (anchors 0a–0d).

### Design docs (the experiments themselves)

- [DREAMER_DIAGNOSTIC_BATTERY](../active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md) — Experiment 0a.
- [DREAMER_PROBE_BATTERY](../active/continual_learning/DREAMER_PROBE_BATTERY.md) — Experiment 0b.
- [DREAMER_CONVENTIONAL_FIXES_BATTERY](../active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) — Experiment 0c.
- [dreamer_offline_wm_imagination_test](../../develop/active/diagnosis/dreamer_offline_wm_imagination_test.md) — Experiment 0d.
- [dreamer_zero_init_reward_critic_fix](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md) — Experiment 2 (Z1).
- [dreamer_twohot_bin_range_fix](../../develop/active/diagnosis/dreamer_twohot_bin_range_fix.md) — Experiment 3 (Z2). Now carries the completed Verification Report.

### Algorithm reference (the place where deviations are tracked)

- [dreamer_v3_implementation](../../project/concepts/dreamer_v3_implementation.md) — full implementation reference doc; §6 has the 30-item deviation list. Items 2 and 27 are now `RESOLVED-PARTIAL` with empirical evidence attached. Item 28 (GRU reset gate) is flagged "NEXT IN THE FIX CASCADE". §9 is the sheeprl-comparison section that surfaced items 27–30.

### Anchor diagnosis (the prior work the study extends)

- [dreamer_hypervigilance_learning_failure](../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md).

### Supporting domain-expert memos

- [dreamer_conventional_failure_modes_for_our_setup](../../project/critiques/dreamer_conventional_failure_modes_for_our_setup.md).
- [dreamer_minimum_viable_strip_down](../../project/directions/dreamer_minimum_viable_strip_down.md).
- [dreamer_v3_implementation_critique](../../project/critiques/dreamer_v3_implementation_critique.md) — architectural-soundness review (professor-rl-bayesian-dl).
- [dreamer_v3_implementation_code_review](../../reviews/dreamer_v3_implementation_code_review.md).
- [dreamer_v3_implementation_math_review](../../reviews/dreamer_v3_implementation_math_review.md) — where the bin-range deviation was first surfaced as a math-side audit finding.

### Memory insights (per-finding rationale)

NEW this window:

- [20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt](../../../.claude-memory/memories/dreamer_diagnosis/20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt.md) — Experiment 3 verdict + the long-horizon-compounding residual that names candidate #1 as next.
- [20260511_1535_encode_decode_flag_mismatch_silent_class_bug](../../../.claude-memory/memories/cluster_ops/20260511_1535_encode_decode_flag_mismatch_silent_class_bug.md) — the second instance of the auxiliary-tooling-falls-out-of-sync bug class + the cross-check anti-silent-failure-mode trip-wire.

From prior window (the immediate predecessor):

- [20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry](../../../.claude-memory/memories/dreamer_diagnosis/20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry.md) — Experiment 2 verdict + the pos/neg asymmetry that picked Experiment 3.
- [20260510_2240_reference_impl_compare_only_act_intersections](../../../.claude-memory/memories/subagent_engineering/20260510_2240_reference_impl_compare_only_act_intersections.md) — the rule for handling reference-impl comparisons.
- [20260510_2241_residual_error_pattern_directs_next_fix](../../../.claude-memory/memories/subagent_engineering/20260510_2241_residual_error_pattern_directs_next_fix.md) — the methodology rule applied twice so far in the cascade.

Older anchors:

- [20260509_1534_wm_reward_head_localized_failure_a1](../../../.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md), [20260509_1535_conventional_fixes_battery_verdict_predator_refute](../../../.claude-memory/memories/dreamer_diagnosis/20260509_1535_conventional_fixes_battery_verdict_predator_refute.md), [20260509_1536_train_py_checkpoint_restore_nnx_skew](../../../.claude-memory/memories/cluster_ops/20260509_1536_train_py_checkpoint_restore_nnx_skew.md) — the first instance of the silent class-bug pattern that returned this window.

### Working files (raw analyzer extractions; gitignored)

- `tmp/20260511_044500_wm_imagination_test_Z2_papercanonical.{json,md}` — authoritative Z2 diagnostic output (matched-flag run).
- `tmp/20260511_044500_wm_imagination_test_Z2.{json,md}` — buggy initial Z2 run (mismatched flag, kept for traceability).
- `tmp/20260511_044500_z2_wandb_extract.md` — Z2 WandB metric extract used for the long-horizon-compounding finding.
- `tmp/20260510_211404_wm_imagination_test_Z1.{json,md}` — Z1 baseline for the cascade comparison.
- `tmp/20260509_wm_imagination_test_A1.{json,md}` — A1 baseline.

### Diary days covering this window

- [2026-05-07](../../diary/2026-05-07.md) → [2026-05-11](../../diary/2026-05-11.md) — five consecutive days of cascade work, including this summary's diary note row.

### Implementation commits relevant to this window (newest first)

- `b71d72c` — memory captures for Z2 verdict + silent-class-bug insight.
- `2ed7199` — §6 update with Z1/Z2 empirical resolutions.
- `e0c875f` — Z2 verification report filled by analyzer.
- `b82dad8`, `1703a4c` — diagnostic-script silent-bug patch.
- `a3cdac2`, `564ffcf` — Z2 launch (training-runner diary + train_command audit trail).
- `f5df600`, `8089ee2` — Z2 code change (paper-canonical bin construction).
- `c9164b0`, `6a66c7a` — Z2 plan written, diary row.
- Earlier commits (cascade predecessors) listed in the predecessor summary [20260510_2255_dreamer_v3_fix_cascade.md](20260510_2255_dreamer_v3_fix_cascade.md).

---

## 6. Reading order if you have 10 minutes

1. **This document** (5 min) — the cumulative state at the close of Z2, with the next candidate selected and rationale.
2. Memory insight [20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt](../../../.claude-memory/memories/dreamer_diagnosis/20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt.md) — Z2 verdict with the load-bearing residual-pattern reasoning that named candidate #1 (3 min).
3. The §Verification Report section of [dreamer_twohot_bin_range_fix](../../develop/active/diagnosis/dreamer_twohot_bin_range_fix.md) — concrete A1 vs Z1 vs Z2 metric tables (2 min).

If you have 30 minutes, also read:

- The §6 deviation list and the §9 sheeprl-comparison section in [dreamer_v3_implementation](../../project/concepts/dreamer_v3_implementation.md) — items 2, 27, 28 are the active candidates; items 29 and 30 are the deferred siblings; the rest is context.
- Memory insight [20260511_1535_encode_decode_flag_mismatch_silent_class_bug](../../../.claude-memory/memories/cluster_ops/20260511_1535_encode_decode_flag_mismatch_silent_class_bug.md) — the defensive practice this window codified (cross-check diagnostic against training-time logger).
- The two prior summaries in this study lineage: [20260509_1555_dreamer_v3_diagnosis](20260509_1555_dreamer_v3_diagnosis.md) and [20260510_2255_dreamer_v3_fix_cascade](20260510_2255_dreamer_v3_fix_cascade.md) — full backstory of how we got here.
