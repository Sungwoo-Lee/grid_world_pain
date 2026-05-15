---
title: "DreamerV3 Diagnosis Study — Week of 2026-05-07 → 2026-05-09 (refutation chain to component-level failure localization)"
study: dreamer_v3_diagnosis
generated: 2026-05-09T15:55
window: "2026-05-07 → 2026-05-09"
status: snapshot
---

# DreamerV3 Diagnosis Study — Summary as of 2026-05-09 15:55 KST

> **One-paragraph summary.** This week's work tried to figure out **why DreamerV3 fails on our hypervigilance task** (predator + foraging) where it gets stuck around 27–30 survival steps regardless of intervention. Four investigations ran in sequence, each refuting the prior week's hypothesis and pointing further in. By the end of the week, the failure is **localized to a single component**: the reward head of the world model is severely miscalibrated even on the simplest possible task (food only, no predator). Reward predictions are off by 2.6× the pre-registered tolerance — the actor cannot learn which actions yield food because the world model's imagined reward signal is wrong. The encoder, decoder, and continuation head are all sound. The previously-suspected fixes (curriculum, no-homeostatic-reward, lever-only training, conventional reward-scale and replay-ratio knobs) and structural hypotheses ("the world model cannot imagine death events") are all **refuted** — leaving the reward head as the next target for redesign.

> **This is a snapshot.** Re-summaries should be written as new dated files in this folder, not by editing this one.

---

## 1. Study question

DreamerV3 — a learned-world-model RL agent — was integrated into our project to compare its sample efficiency against our recurrent PPO baseline on the **hypervigilance task** (a small grid-world where the agent must forage food while avoiding a moving predator, with energy-depletion death as a backup termination). On every variant we have tried, DreamerV3 plateaus at survival ~27–30 steps with constant negative advantage, near-uniform actor entropy, and food intake collapsing — a clean *no-learning* signature, not a *partial-learning* one.

**The puzzle**: standard DreamerV3 hyperparameters were calibrated for Atari, DMControl, and Crafter. Our task differs in several ways at once — vector-only observations, sparse-mixed-magnitude rewards (small dense ±0.5 + a large terminal −100), homeostatic state (satiation), small grid (5×5), short episodes. Any one of those could be the bottleneck. The week's job was to identify *which*.

The investigation followed a refutation chain: each experiment ruled out a class of candidate causes and pointed to the next class to test. Performance is measured in **survival steps** (project rule), never cumulative reward.

---

## 2. Experiments completed this week

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |
|---|---|---|---|---|---|
| **1** | **Diagnostic battery** *(2026-05-07, n114, 4 parallel cells)* | Among four "obvious" candidate fixes — give the agent only the predator-avoidance subtask (lever-only); give it only the foraging subtask (curriculum-only); remove the homeostatic energy/satiation reward; soften the predator-introduction ramp — does any of them recover learning? | Four different config variants, one per cell; same agent, same seed, 700k episodes each. | **All four fixes refuted.** Every cell hit its pre-registered refutation criterion. Survival stayed around the unmodified-baseline anchor; no cell recovered learning. | Closed the "obvious task-shaping fix" hypothesis class. The remaining surviving hypothesis at this point was a **structural** one: the world model itself is failing to imagine the death event, so the value gradient never flows through "this action causes death later." |
| **2** | **Imagined-rollout probe battery** *(2026-05-08, n114, 4 cells E1–E4)* | Does the world model's continuation head actually predict death events when the agent is rolled out in imagination? If not, that explains the value-gradient failure. **Anchor + boost + positive-control + curriculum.** | Added an instrumentation flag that logs, during imagination, the fraction of trajectories that predict termination at horizons 8 and 15. One cell ran the unmodified baseline; one boosted the continuation-head loss weight 5× to test the obvious "if the head is undertrained, train it harder" fix; one ran the no-predator task as a positive control; one ran a 3-stage food-then-predator curriculum. | **Structural hypothesis refuted.** The world model **does** predict ~27% terminations within horizon 15 — not zero. But it predicts them at imagined step ~12 vs real step ~23 (i.e., **10 steps too early**) and **uniformly across all 6 actions** (no per-action discrimination). The "boost the continuation-head loss" fix raised the imagined-termination rate from 0.30 to 0.35 but had **zero effect on survival, entropy, or advantage** — they were bit-identical to the unboosted anchor. | Closed "world model cannot imagine death" as the explanation. New (preliminary) hypothesis: imagined deaths are **miscalibrated in time and per-action**, not absent. The continuation head's metric moves but the value gradient stays anaemic because every action looks equally fatal. |
| **3** | **Conventional-fixes mini-battery** *(2026-05-09, n113, 2 cells A1 + A2)* | After the structural hypothesis collapsed, a domain-expert review (the project's "professor-rl" agent) walked the textbook DreamerV3 failure-mode checklist and ranked the top three conventional causes our config was likely tripping: (1) reward-scale mismatch (the −100 death penalty dwarfs ±0.5 dense rewards), (2) replay-ratio over-training (we ran 0.5; the paper default is 0.0625; the published "modal-trace collapse" failure mode matches our observed signature), (3) imagine-horizon shorter than the death-event horizon. **Do the top two one-line config fixes work?** | Cell A1: simplest task (no predator) + replay ratio dropped to the published default. Cell A2: full predator task + both fixes applied jointly (replay-ratio drop + death-penalty rescaled −100 → −1). 700k episodes each. | **Top-2 conventional causes refuted on predator.** Cell A2 lands at survival 27 with mean-advantage −0.19 — pixel-identical to the unmodified baseline. The critic is well-calibrated to the new reward scale, so the death-penalty rescaling did not break value learning; the failure is genuinely *elsewhere*. **Cell A1 partial result.** No collapse (the published modal-trace failure mode IS prevented by the replay-ratio fix), but survival saturates at 106 in a stable starvation equilibrium — well below the unfixed run's pre-collapse peak of 332. | Closed "this is a textbook conventional-knob problem on predator." The blockage on predator is *downstream of the reward head* and *downstream of the critic*. The replay-ratio fix is, however, a real win on the simplest task — it should be promoted to a default. The next test had to look at the trained world model itself, not at the training loop. |
| **4** | **Offline world-model imagination test** *(2026-05-09, single A1 checkpoint)* | Now that we have a Cell-A1 trained checkpoint that does *something* (survival 106, no collapse) on the simplest possible task — load it offline, freeze the weights, and ask: can the world model actually predict the next several steps of observation, reward, and continuation accurately, or not? If yes, the bottleneck has to be in the actor / value / training loop. If no, the bottleneck is *inside* the world model and we can localize *which head*. | Built a one-off diagnostic script that loads the checkpoint, replays real episodes to get ground-truth observation-action-reward sequences, and from 200 starting states runs the world model in imagination for up to 50 steps. At each horizon, compares predicted observation, predicted reward, and predicted continuation against the recorded reality. Pre-registered tolerances per channel. | **The world model is broken at the reward head, *only*.** Continuation prediction at horizon 5 is 99.5% accurate; observation-channel reconstruction passes most channels; long-horizon drift out to 50 steps is healthy (only 1.7× the horizon-5 error). **But reward MAE is 0.386 vs the 0.15 tolerance — 2.6× over** — even on the simplest food-only task. Per-action proprioception (= a deterministic encoding of "what action did I just take?") also fails its tolerance — the decoder cannot reconstruct the most determined channel even at horizon 1, suggesting the latent representation is losing action-identity information. | **Cleanly localized the failure to the reward head.** The encoder/decoder architecture is sound; the continuation head is sound; the latent-dynamics autoregression is sound. The single component the actor depends on for "which action gets food?" is the one component that's miscalibrated. This is the headline finding of the week — every prior negative result becomes interpretable through this lens. |

---

## 3. Where this leaves the study

- **The failure mode is now localized to one component.** Three weeks of "the agent doesn't learn" have collapsed onto a single specific claim: the world model's reward head cannot accurately predict per-step rewards, even on the simplest possible task we can construct (food only, no predator, no homeostatic pressure beyond satiation). This is the next intervention target, not yet another task-shaping or hyperparameter sweep.
- **The "structural" and "conventional-knob" hypotheses are both closed.** Curriculum training, lever-only training, removing homeostatic reward, softening the predator ramp, boosting the continuation-head loss, rescaling the death penalty, dropping the replay ratio to the paper default, and giving the agent more reward-scale headroom were all tried. None of them recovered learning on the predator task. This negative result is informative — it tells us not to spend further GPU-time on these classes of fix.
- **One real win.** The replay-ratio fix (`replay_ratio: 0.5 → 0.0625`) prevents the previously-published "modal-trace collapse" failure mode on the no-predator task. It does not, on its own, get the agent to competence — but it does prevent the runaway pessimism collapse seen in the older `replay_ratio=0.5` run on the same task. It should be promoted to a default for any future DreamerV3 work in this codebase.
- **A methodological surprise**: a previously-deferred component-level diagnostic (loading a trained checkpoint and asking "can the world model predict the world?") turned out to be the highest-leverage tool of the week. It cost ~30 seconds of wall-clock and one one-off script (~490 lines) and pinpointed the failure where three multi-cell training batteries had only narrowed the search. The lesson: when training-time diagnostics keep refuting candidate causes, switch to inference-time diagnostics on the trained checkpoint. The training loop is the slow channel; the trained model is the fast channel.
- **A latent infrastructure bug surfaced** as a side effect: the trainer's checkpoint-restore path (`train.py:981–1000`) would currently fail on a fresh `--resume` because the orbax-saved format does not round-trip through current NNX without the workaround the new diagnostic script applied. This is non-blocking (we don't currently `--resume`) but should be fixed before any long-run extension or warm-restart workflow.
- **A reader-facing cleanup**: a previous write-up of these experiments referred to "proprioception" as the agent's position and orientation. It is not — it is a one-hot encoding of the previous action. The "proprioception is broken" finding from Experiment 4 is still real, but its interpretation is "the decoder cannot reconstruct the most determined channel," not "the world model doesn't know its own location." The diagnostic does not measure positional prediction at all (the location sensor is disabled in the trained config).

---

## 4. What's next (still pending decision)

1. **Run the same offline diagnostic on the predator-task checkpoint** (`experiment-analyzer` to author a one-off, ~30 s wall-clock). Cell A2's checkpoint exists; the diagnostic script is already written. This tells us whether the reward-head failure is task-independent (single root cause across NoPred + predator) or task-specific (predator has additional pathology). If single-root-cause, a single intervention fixes both tasks. If task-specific, predator needs separate work.
2. **Plan a reward-head intervention** (`senior-developer` to draft a plan, blocked on item 1). Two candidate targets surfaced this week: (a) a class-balanced loss on the reward head (originally priority 4 in the failure-mode diagnosis; now elevated by the offline-diagnostic finding); (b) an audit of DreamerV3's two-hot reward encoding bin-spacing — its fixed bin layout was calibrated for the paper's reward distributions, which look nothing like ours (sparse positives + −100 terminal). Either of these is a single-file change to the trainer; the plan should pick one to test first.
3. **Promote `replay_ratio: 0.0625` to a default** (`experiment-designer` to update the canonical DreamerV3 agent config, no blocking dependency). This is a clean win from Experiment 3 and should not regress on any future run.
4. **Fix the `train.py` checkpoint-restore bug** (`senior-developer` to plan, no blocking dependency on the science work). The diagnostic script worked around it locally; the trainer itself still has the bug. Any future warm-restart or resume workflow will hit it.
5. **Optional: extend the no-predator run to 1.5M episodes** (`experiment-designer`, blocked on item 2). The week-3 verdict on Cell A1 was "stable starvation equilibrium at 106 survival, not collapse" — but that's at 700k episodes. Whether the agent eventually escapes the equilibrium or genuinely asymptotes there is unknown. Cheap test, but lower priority than the reward-head work.

---

## 5. Links to authoritative documents

### Design docs (the experiments themselves)

- [DREAMER_DIAGNOSTIC_BATTERY](../active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md) — Experiment 1, full design + Results.
- [DREAMER_PROBE_BATTERY](../active/continual_learning/DREAMER_PROBE_BATTERY.md) — Experiment 2, full design + Results / Analysis.
- [DREAMER_CONVENTIONAL_FIXES_BATTERY](../active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) — Experiment 3, full design + Results / Analysis / Conclusions.
- [dreamer_offline_wm_imagination_test](../../develop/active/diagnosis/dreamer_offline_wm_imagination_test.md) — Experiment 4, plan + Implementation Report + Verification Report (this is a develop-doc, not an experiment-doc, because it built a new diagnostic tool rather than running a training cell).

### Anchor diagnosis (the prior work this study extends)

- [dreamer_hypervigilance_learning_failure](../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md) — the original failure-mode diagnosis that defined the central puzzle and ranked the candidate-fix priorities Experiments 1 + 2 tested.

### Supporting domain-expert memos

- [dreamer_conventional_failure_modes_for_our_setup](../../project/critiques/dreamer_conventional_failure_modes_for_our_setup.md) — the professor-rl-bayesian-dl analysis that ranked the top-3 conventional DreamerV3 failure modes our config was likely tripping. Experiment 3 was designed against this checklist.
- [dreamer_minimum_viable_strip_down](../../project/directions/dreamer_minimum_viable_strip_down.md) — direction memo on the minimum-viable strip-down test (food only, no predator); informed Cell A1's design.

### Memory insights (per-finding rationale, refutations, methodological notes)

Each insight carries the full reasoning behind its conclusion and explicitly names what it supersedes / refines:

- [20260508_1431_diagnostic_battery_refutes_four_fixes](../../../docs/memory/memories/dreamer_diagnosis/20260508_1431_diagnostic_battery_refutes_four_fixes.md) — Experiment 1 verdict: all four candidate fixes refuted.
- [20260508_1432_probe_refutes_imagined_death_absence](../../../docs/memory/memories/dreamer_diagnosis/20260508_1432_probe_refutes_imagined_death_absence.md) — Experiment 2 verdict: structural hypothesis refuted; new working hypothesis (since superseded by Experiment 4's finding) was "imagined deaths miscalibrated in time and per-action."
- [20260509_1535_conventional_fixes_battery_verdict_predator_refute](../../../docs/memory/memories/dreamer_diagnosis/20260509_1535_conventional_fixes_battery_verdict_predator_refute.md) — Experiment 3 verdict: top-2 conventional causes refuted on predator; partial NoPred win.
- [20260509_1534_wm_reward_head_localized_failure_a1](../../../docs/memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md) — Experiment 4 verdict (the week's headline): failure localized to the reward head; encoder + decoder + continuation head are sound.

Two cross-cutting insights from the same window:

- [20260509_1536_train_py_checkpoint_restore_nnx_skew](../../../docs/memory/memories/cluster_ops/20260509_1536_train_py_checkpoint_restore_nnx_skew.md) — the latent infrastructure bug surfaced while building Experiment 4's tooling.
- [20260509_1537_professor_analysis_resets_exotic_investigation](../../../docs/memory/memories/subagent_engineering/20260509_1537_professor_analysis_resets_exotic_investigation.md) — the methodological pattern that produced Experiment 3's design (pull domain-expert analysis BEFORE writing more bespoke probes).

### Working files (raw analyzer extractions, intermediate notes)

- [20260508_replayRatio05_NoPred_analysis](../../../tmp/20260508_replayRatio05_NoPred_analysis.md) — post-hoc analysis of the older `replay_ratio=0.5` no-predator collapse run; established the failure-template the conventional-fixes battery was designed to refute.
- [20260509_dreamer_conv_A1_timeseries](../../../tmp/20260509_dreamer_conv_A1_timeseries.md), [20260509_dreamer_conv_A2_timeseries](../../../tmp/20260509_dreamer_conv_A2_timeseries.md), [20260509_dreamer_conv_battery_verdict](../../../tmp/20260509_dreamer_conv_battery_verdict.md) — Experiment 3 raw analyzer output.
- [20260509_wm_imagination_test_A1](../../../tmp/20260509_wm_imagination_test_A1.md) (+ matching `.json`) — Experiment 4 raw output.

### Diary days covering this study

- [2026-05-07](../../diary/2026-05-07.md) — Experiment 1 launches; the diagnostic-battery refutation chain begins.
- [2026-05-08](../../diary/2026-05-08.md) — Experiment 2 launches + analysis + the probe-battery insights captured; CIFS / runner ops insights also from this day.
- [2026-05-09](../../diary/2026-05-09.md) — Experiment 3 launches (05:09) + analysis (13:36); Experiment 4 plan (14:13), implementation (14:55), verification (15:18); week's insights captured (15:41).

### Implementation commits relevant to this study

- `1d87ab8` — `agent.cont_loss_weight` knob added to the trainer (used as Experiment 2's lever).
- `82a31ff` — imagined-rollout probe instrumentation (the during-training probe Experiment 2 ran).
- `7639c2b` — Experiment 2 configs.
- `3d93b0a` — Experiment 4 implementation (`scripts/dreamer_offline_wm_test.py` + plan-doc Implementation Report).
- `e6d2bbe` — Experiment 4 verification report.
- `f08d18b` — week's memory insights captured (4 insights + indexes + diary).

---

## 6. Reading order if you have 10 minutes

1. **This document** (5 min) — gets you the headline finding (failure is localized to the reward head) and the open follow-ups.
2. Memory insight [20260509_1534_wm_reward_head_localized_failure_a1](../../../docs/memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md) — the week's load-bearing finding with full evidence and verdict thresholds (3 min).
3. The Verification Report section of [dreamer_offline_wm_imagination_test](../../develop/active/diagnosis/dreamer_offline_wm_imagination_test.md) — confirms the headline finding is methodologically sound and surfaces the proprioception clarification (2 min).

If you have 30 minutes, also read:

- The professor-rl analysis at [dreamer_conventional_failure_modes_for_our_setup](../../project/critiques/dreamer_conventional_failure_modes_for_our_setup.md) — the conventional-cause checklist that motivated Experiment 3.
- Memory insight [20260508_1432_probe_refutes_imagined_death_absence](../../../docs/memory/memories/dreamer_diagnosis/20260508_1432_probe_refutes_imagined_death_absence.md) — the structural-hypothesis refutation that triggered the pivot to conventional causes.
- The §6.3 verdict matrix in [DREAMER_CONVENTIONAL_FIXES_BATTERY](../active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) — the pre-registered confirmation/refutation criteria and which row Experiment 3 actually hit.
