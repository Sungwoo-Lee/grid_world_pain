---
id: 20260509_1534_wm_reward_head_localized_failure_a1
date: 2026-05-09
time: "15:34"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, refutation]
summary: "Offline WM-imagination diagnostic on Cell A1's NoPred checkpoint pinpoints the failure: encoder/decoder + continuation head are sound (aggregate obs symlog-MSE 0.059 PASS, cont accuracy 0.995 PASS, autoregressive drift healthy at h50/h5=1.69), but reward MAE 0.386 (2.6× threshold) at h=5 even on the simplest food-only task. Localizes the bottleneck to the reward head specifically."
related: ["20260508_1431_diagnostic_battery_refutes_four_fixes", "20260508_1432_probe_refutes_imagined_death_absence"]
session_origin: claude_code
session_label: "dreamer_conventional_fixes_battery_2026-05-09"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# Offline WM diagnostic localizes DreamerV3 failure to the reward head

## Key conclusion
Loading Cell A1's trained NoPred checkpoint (700k env-steps, `replay_ratio: 0.0625`) and running it in offline imagination from M=200 real replay states refutes "WM is broken" as a global statement and replaces it with a precise localization: the encoder/decoder + continuation head are sound, but the reward head fails to predict per-step reward at h=5 by 2.6× the pre-registered threshold even on the simplest possible task (food only, no predator). This explains both Cell A1's stuck-at-survival-106 starvation equilibrium and Cell A2's H₀ verdict on predator: the actor cannot learn which actions yield food because the imagined reward signal is miscalibrated.

## Evidence, measurements, facts
- Diagnostic script: `scripts/dreamer_offline_wm_test.py` (commit `3d93b0a`, verified `e6d2bbe`).
- Plan: `docs/develop/active/diagnosis/dreamer_offline_wm_imagination_test.md` (incl. Implementation Report + Verification Report).
- Method: replay-state-conditional imagination (D1), deterministic (RSSM categorical-mode + actor argmax, D5), M=200 starts, horizons `{1, 2, 5, 10, 15, 25, 50}`. JSON + Markdown output under `tmp/`.
- Verdict at h=5 (pre-registered):
  - Reward MAE = **0.386** vs threshold < 0.15 → **FAIL (2.6×)**.
  - Aggregate obs symlog-MSE = 0.059 vs threshold < 0.10 → PASS.
  - Continuation accuracy = 0.995 vs threshold > 0.95 → PASS.
  - Long-horizon h50/h5 ratio = 1.69 vs threshold ≤ 2.0 → PASS (autoregressive drift healthy out to 3× training horizon).
  - Per-channel: Satiation/Interoceptive Nociception PASS; Olfaction PASS; Collision PASS; Proprioception MSE 0.107 vs threshold 0.05 → FAIL (corroborating).
- Proprioception clarification (verification correction): proprio is `jax.nn.one_hot(state.last_action, action_dim)` per `src/environment/sensor.py:300–302`, NOT (x, y) + orientation. The MSE 0.107 is above the uniform-prediction baseline ~0.068, so the decoder is putting density on the wrong action bin — interpretation is "decoder fails on a deterministic action-history channel," not "WM can't predict its own position." `location_sensor` is disabled in A1's saved config; the diagnostic does not measure positional prediction at all.
- Test outputs: `tmp/20260509_wm_imagination_test_A1.json`, `tmp/20260509_wm_imagination_test_A1.md`.
- A1 checkpoint: `results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113/` (WandB `czfnljf0`).

## Decisions and actions
- WM bottleneck localized to the reward head. Encoder/decoder architecture is sound — no need to redesign the perceptual pipeline.
- Reward-head class-balanced loss (was priority 4 in `dreamer_hypervigilance_learning_failure.md`) elevated to top priority for the next intervention plan.
- Two-hot bin-spacing audit added as a candidate root cause: DreamerV3's two-hot reward encoding uses fixed bin spacing; if it doesn't match our reward distribution (sparse positives + −100 terminal in predator, ±0.5 dense in NoPred), the head can't represent per-step reward accurately.
- Next cheap diagnostic queued: run the same script against Cell A2's predator checkpoint to confirm the reward-head failure is task-independent (single root cause) vs. task-specific (predator has additional pathology beyond what NoPred shows).

## Open questions and follow-ups
- Does Cell A2's predator checkpoint show the same reward-head MAE pattern? Single root cause vs. task-specific.
- Is the reward-head failure due to two-hot bin spacing (architectural), class imbalance (loss-shaping), or something else (e.g., reward symlog calibration)?
- Why does the proprioception (= previous-action) decoder fail at h=1? It's a fully-determined channel from `(deter, stoch)` — should be near-zero error. Possibly a clue about how much information about "the action just taken" survives the RSSM compression.
- Does the WM's reward-head failure on the simplest task (NoPred, food-only) imply this same architecture has never worked on any of our prior tasks, or only that the homeostatic reward signal is harder than expected?

## References
- Plan + Implementation + Verification: `docs/develop/active/diagnosis/dreamer_offline_wm_imagination_test.md`.
- Sibling insight (same session): `20260509_1535_conventional_fixes_battery_verdict_predator_refute` — the experimental verdict that motivated this offline diagnostic.
- Parent investigation: `20260508_1432_probe_refutes_imagined_death_absence` (during-training imagined-rollout probe; refuted "WM cannot imagine death") and `20260508_1431_diagnostic_battery_refutes_four_fixes`.
- Professor analysis that ranked the conventional causes: `docs/project/critiques/dreamer_conventional_failure_modes_for_our_setup.md`.
- Original failure-mode diagnosis (priority 4 reward-head class-balanced loss now elevated): `docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`.
- WandB run: `czfnljf0` (Cell A1 NoPred); commits `3d93b0a` (script + plan-doc), `e6d2bbe` (verification).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).
