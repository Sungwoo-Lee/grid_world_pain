---
title: "DreamerV3 Fix Cascade — Sheeprl-comparison + first targeted code-side fixes (2026-05-07 → 2026-05-10)"
study: dreamer_v3_fix_cascade
generated: 2026-05-10T22:55
window: "2026-05-07 → 2026-05-10"
status: snapshot
---

# DreamerV3 Fix Cascade — Summary as of 2026-05-10 22:55 KST

> **Continuation of the prior dreamer_v3_diagnosis summary** ([20260509_1555_dreamer_v3_diagnosis.md](20260509_1555_dreamer_v3_diagnosis.md)) which ended at "reward head localized as the failure component." This summary covers what happened next: a third-party reference implementation (sheeprl) was used as a comparator to surface a deeper deviation list, then the first targeted code-side fix was applied and tested, then the next fix was queued mechanistically. The prior summary remains a historical snapshot of what we knew on 2026-05-09; this one is the current state.

> **One-paragraph summary.** We localised the DreamerV3 failure to the reward head on the simplest possible task (food-only foraging, no predator). To pick the first repair, we compared our codebase against sheeprl, an actively-maintained third-party reference implementation of DreamerV3, and surfaced four candidate deviations. We applied the most-targeted candidate first — initialising the reward and critic output layers to zero, the way the published DreamerV3 code does — and re-ran the same task. The fix produced a **clear partial improvement** (reward-prediction error fell 28%) but did **not** clear our pre-registered correctness threshold. The residual error has a specific shape: the reward head's positive-event predictions improved 49% but its negative-event predictions improved only 14%. That asymmetry directly fingered a *second* deviation as the next candidate — the head's representable value range is far narrower than the paper's, so any reward outside ±20 saturates at the boundary. A second fix targeting that range is now in flight on the same task. We are running a paper-faithfulness ladder, picking the next rung mechanistically rather than from a fixed list.

> **This is a snapshot.** The prior summary is preserved as `20260509_1555_dreamer_v3_diagnosis.md`; do not edit it. The next summary, written after Z2 lands and any further fixes, should be a new dated file.

---

## 1. Study question

The previous diagnosis (the four-experiment refutation chain summarised on 2026-05-09) localised the DreamerV3 failure to **one component** — the reward head of the world model. On the simplest task we can build (a 5×5 grid-world with food and no predator), the trained world model predicts observation channels, continuation, and long-horizon dynamics correctly, but its predicted per-step reward is off by 2.6× the pre-registered correctness threshold. Without an accurate predicted reward, the actor cannot tell which actions yield food.

This study asks the practical follow-up: **which code-side change, applied first, repairs the reward head most cleanly?** The constraints are:

- **Be paper-canonical.** The user's stated rule is to converge toward the published DreamerV3 algorithm rather than invent project-specific repairs. Each candidate fix is a known difference between our implementation and a reference: the DreamerV3 paper, the published Hafner code, or the sheeprl third-party implementation.
- **Be mechanistically motivated.** The candidate list contains many differences. Pick the one whose mechanism most directly explains the observed failure pattern. After each fix, look at the residual error pattern and use it to pick the next.
- **Stay reversible.** Each fix is gated on a config knob that defaults to the new behaviour but allows reverting to the legacy behaviour bit-identically.

Performance is measured in two ways here: (a) the world model's reward-prediction error on a frozen-checkpoint diagnostic, which is the *primary* signal because it isolates the localised failure component; (b) survival steps in the live environment, which is the project-wide rule but is a secondary signal in this sub-study.

---

## 2. Experiments completed this window

Table covers experiments from the predecessor summary (rows 0a–0d, marked *(prior anchor)*) plus the three new events of this window (rows 1, 2, 3).

| # | Experiment | Question (plain English) | What was varied | High-level finding | What it changed about our understanding |
|---|---|---|---|---|---|
| **0a** *(prior anchor)* | **Diagnostic battery** | Among four obvious task-shaping fixes, does any recover learning? | 4 cells: lever-only, curriculum-only, no-homeostatic-reward, smoother-ramp. | All four refuted. | Closed the obvious task-shaping fix class. |
| **0b** *(prior anchor)* | **Probe battery** | Does the world model imagine the death event during rollouts? If not, that explains the value-gradient failure. | 4 cells with a new instrumentation flag logging imagined-termination metrics. | The world model does predict terminations (about a quarter of imagined trajectories within 15 steps), but at the wrong time and uniformly across actions. Boosting the continuation-head loss raised the metric without raising survival. | Closed the "world model can't imagine death" hypothesis. |
| **0c** *(prior anchor)* | **Conventional-fixes battery** | Do the top two textbook DreamerV3 fixes (replay-ratio, reward-scale rescale) recover learning? | 2 cells: NoPred + low replay-ratio; predator + both fixes jointly. | Both fixes refuted on the predator task; the no-predator cell prevented the published modal-trace collapse but plateaued in a stable starvation equilibrium. | Closed the textbook conventional-knob hypothesis class on the predator task. |
| **0d** *(prior anchor)* | **Offline world-model imagination test** | With the no-predator agent's trained checkpoint frozen, can the world model itself accurately predict the next several steps? | A new offline diagnostic script that runs the trained world model in imagination from real replay states. | Encoder, decoder, continuation head, and long-horizon dynamics all pass their thresholds. **Only the reward head fails** — predicted reward 2.6× the tolerance. | **Localised the failure to one component.** Headline finding of the prior summary. |
| **1** | **Sheeprl reference-implementation comparison** *(2026-05-09 → 2026-05-10, doc-side event, not a training run)* | We have one paper-canonical reference (Hafner 2023) and one published-code reference (Hafner's released code). Sheeprl is an actively-maintained third-party DreamerV3 implementation; a side-by-side walk surfaces *which* of our deviations from paper are also deviations from a high-quality independent implementation. Pulling the third reference identifies which differences are likely-real-bugs vs framework idiom vs deliberate-research-choices. | A senior-developer agent walked sheeprl's DreamerV3 codebase (`tmp/sheeprl/`), filtered framework idiom (PyTorch vs JAX/NNX) from algorithmic substance, and added a §9 "Comparison with sheeprl reference implementation" section to our concept doc. | **Four new candidate deviations surfaced** beyond the existing 26-item deviation list: (i) the GRU's reset gate is computed but never applied to the candidate hidden-state update, (ii) the prior and posterior categorical-distribution heads are single linear layers instead of one-hidden-layer MLPs, (iii) the critic loss is missing a term that regularises the critic toward an exponentially-moving-average copy of itself, (iv) the reward and critic output layers are not initialised to zero. The user's stated rule for handling these: **document by default; act on a candidate only when it directly intersects the live failure investigation**. Of the four, only candidate (iv) directly targets the localised reward-head failure. | Surfaced new candidates AND established a routing rule (any reference-impl comparison surfaces N differences; only act on those whose mechanism intersects the live issue). |
| **2** | **Cell Z1 — zero-init reward + critic output layers** *(2026-05-10, on Node 113, no-predator task, 700,000 episodes, ~2.5 h wall-clock)* | The reward head's output layer is initialised with the project's default scheme (non-zero noise). Sheeprl initialises it to all zeros so the head starts predicting zero reward and learns from there. Does removing the startup-noise contamination repair the reward head? | A single config knob `agent.zero_init_reward_critic` was added (default true) that initialises only the reward and critic *output* linear layers to zero (kernel = 0, bias = 0); the rest of the network keeps the existing scheme. | **Partial fix — the reward-prediction error fell from 0.39 to 0.28 (a 28% improvement)** on the offline diagnostic, but did not clear the H1 threshold of 0.15. The agent's training-time reward-prediction error on positive events improved 49%; on negative events it improved only 14%. Survival lifted soft (106 → 115 steps, single seed, not a primary signal). | **Confirmed candidate (iv) was a real and high-leverage intersection** — sheeprl's pattern matters. AND surfaced a residual-error pattern (positive vs negative asymmetry) that points at a *specific* next candidate, not just "try the next fix on the list." |
| **3** *(in flight, verdict pending)* | **Cell Z2 — paper-canonical two-hot reward bins** *(2026-05-10 ~22:11, on Node 113, no-predator task, 700,000 episodes, expected ~01:00 KST)* | The reward head encodes its prediction as a "two-hot" categorical distribution over a fixed set of bins. Our bins span only `±20` in raw reward space; the published DreamerV3 code's bins span `±4.85×10⁸`. The death-penalty event of `−100` therefore lives outside our head's representable bin support entirely. The Z1 partial-fix's residual error was concentrated on negative events — exactly the regime that the bin-range mismatch makes unrepresentable by construction. Does fixing the bin range close the remaining gap? | A second config knob `agent.paper_canonical_twohot_bins` was added (default true) that constructs bins as `symexp(linspace(−20, +20, 255))` — uniform spacing in symlog space, then mapped back to raw — instead of our previous `linspace(symlog(−20), symlog(+20), 255)`. The numerical effect: the smoke test confirmed that a reward of `−100` now round-trips with error 0.0003 (was 80, due to clipping at `−20`). | **Pending — training run started 2026-05-10 22:11 KST, expected to finish ~01:00 KST 2026-05-11.** Verdict comes from re-running the offline diagnostic on the resulting checkpoint. The pre-registered thresholds for this cell are tighter than Z1's because Z1 already moved the metric to 0.28: the H1 (full repair) threshold is < 0.15; partial-improvement < 0.25; refuted ≥ 0.25. | TBD. The mechanistic prediction is "the residual is on negative events; bin-range fix targets exactly that." A clean H1 outcome would close the reward-head investigation. A partial-improvement outcome would queue candidate (i) (GRU reset gate) as the next rung. A refutation outcome would imply the residual is structural in some other way. |

---

## 3. Where this leaves the study

- **The fix cascade is working so far.** The first targeted code-side fix produced a clean partial improvement (28% reduction in the load-bearing metric) and pointed unambiguously at the next candidate. We have not yet had a fix that did nothing, and we have not yet had to revisit a refuted hypothesis class. The arc is converging.
- **A new comparator is now in our toolkit.** Adding sheeprl as a third reference (alongside the paper text and the published Hafner code) was high-leverage; it surfaced four new candidate deviations the paper-vs-our-code audit had not. **Side-by-side independent implementations catch things text alone does not.** The same pattern likely transfers to other algorithms in this codebase — when investigation surfaces a localised failure, an independent implementation comparison is a cheap next step.
- **A working rule for handling reference-impl comparisons emerged**: "document differences by default; act on them only when the mechanism intersects the live failure." This applies to any external comparator (paper, library, third-party impl). Reflexively matching every difference is a category error — most differences are deliberate research choices, framework idiom, or bugged-in-them-not-us.
- **A working rule for sequencing fixes emerged**: when a partial fix lands, **look at the residual error pattern in the diagnostic output and let it pick the next candidate**. Plan-prescribed lists of "what to try next" are useful as defaults, but the residual-error pattern carries information the list does not. Z1's positive-vs-negative asymmetry directly fingered the bin-range deviation as the mechanistic match — a candidate that a plan-prescribed list would have queued lower.
- **Practical config-default change** that has fallen out so far: zero-initialise the reward and critic output layers (now default in our two main DreamerV3 agent configs). The next default change is conditional on Z2's outcome.
- **A small infrastructure-hygiene debt** has accumulated as a side effect: five other DreamerV3 agent configs in the codebase (used for prior probe / curriculum / neuromodulated experiments) lack the new mandatory config keys and would now error on use. Pre-existing condition; flagged for cleanup but not blocking.

---

## 4. What's next (still pending decision)

1. **Read Z2's verdict and act on it accordingly** *(`experiment-analyzer` runs the offline diagnostic on the trained Z2 checkpoint; the user reads the result and routes)*. Three branches:
   - If Z2 clears the H1 threshold (full repair on the simplest task), promote both fixes to defaults, then test the same fix-stack on the predator task (the original hypervigilance failure). The fix cascade closes for the no-predator regime.
   - If Z2 partially improves but does not clear H1, **read the new residual pattern** and use it to pick the next candidate. Most likely next rung is the GRU reset-gate fix (sheeprl candidate (i), a likely-real bug in our recurrent dynamics that affects every imagined step uniformly).
   - If Z2 does not improve materially, the bin-range deviation was not a real intersection despite the mechanistic match. Revisit the residual-error reasoning and consider the missing-critic-EMA-regularisation candidate (sheeprl (iii)) or the prior/posterior MLP undersizing (sheeprl (ii)).
2. **Test the cumulative fix-stack on the predator task** *(`experiment-designer` to author, blocked on item 1)*. The whole reward-head investigation has been on the no-predator task because that's where the failure was first localised. The original project goal — survival under predator pressure — is one task harder. Before declaring victory, the same fixes must transfer.
3. **Clean up the five orphan DreamerV3 agent configs** *(`senior-developer` to plan, no blocking dependency)*. They lack the two new mandatory keys (`zero_init_reward_critic`, `paper_canonical_twohot_bins`) and would error on use. Pre-existing condition that grew with each new knob; one-line additions per file plus a regenerated develop-docs index.
4. **Write up the updated deviation list as a stand-alone reference** *(later, after the fix cascade resolves)*. The DreamerV3 implementation reference doc now has 30 entries in its deviation summary section; once the cascade settles, a separate "deviation closure log" — what was real, what was deliberate, what was framework — would be a useful artefact for future re-implementation work.
5. **Optional: 3-seed sweep at the cumulative-fix configuration** *(`experiment-designer`, lower priority than item 2)*. Z1 and Z2 are single-seed; the survival lift (106 → 115 between A1 and Z1) carries seed-noise we have not characterised in this regime. Tightening the survival estimate is useful for paper-figures eventually but does not block any decision.
6. **Extend the offline imagination diagnostic with a per-event-type breakdown** *(later; cheap; `senior-developer` to plan)*. The current diagnostic reports aggregate reward MAE; the residual analysis required pulling per-event-type metrics from the WandB training logs. A direct per-event-type column in the offline diagnostic would have made Z1's residual visible without that extra step.

---

## 5. Links to authoritative documents

### Design docs (the experiments themselves)

- *(prior anchors)* Predecessor summary: [20260509_1555_dreamer_v3_diagnosis](20260509_1555_dreamer_v3_diagnosis.md) — covers experiments 0a–0d.
- [DREAMER_DIAGNOSTIC_BATTERY](../active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md) — Experiment 0a.
- [DREAMER_PROBE_BATTERY](../active/continual_learning/DREAMER_PROBE_BATTERY.md) — Experiment 0b.
- [DREAMER_CONVENTIONAL_FIXES_BATTERY](../active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) — Experiment 0c.
- [dreamer_offline_wm_imagination_test](../../develop/active/diagnosis/dreamer_offline_wm_imagination_test.md) — Experiment 0d.
- [dreamer_zero_init_reward_critic_fix](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md) — Experiment 2 (Z1).
- [dreamer_twohot_bin_range_fix](../../develop/active/diagnosis/dreamer_twohot_bin_range_fix.md) — Experiment 3 (Z2, in flight).

### Algorithm reference (the place where the deviations are documented)

- [dreamer_v3_implementation](../../project/concepts/dreamer_v3_implementation.md) — full implementation reference doc; §6 has the 30-item deviation list (item 2 = two-hot bin range; item 4 = GRU reset gate; items 27–30 = the four sheeprl-comparison additions); §9 is the side-by-side comparison with sheeprl that surfaced experiment 1 of this summary.

### Anchor diagnosis (the prior work this study extends)

- [dreamer_hypervigilance_learning_failure](../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md) — the original failure-mode diagnosis that ranked candidate fixes for the four-experiment refutation chain.

### Supporting domain-expert memos (from the conventional-fixes pivot)

- [dreamer_conventional_failure_modes_for_our_setup](../../project/critiques/dreamer_conventional_failure_modes_for_our_setup.md) — the professor-rl-bayesian-dl analysis that ranked the textbook DreamerV3 failure modes (used by Experiment 0c).
- [dreamer_v3_implementation_critique](../../project/critiques/dreamer_v3_implementation_critique.md) — the professor-rl-bayesian-dl review of the implementation reference doc (architectural soundness lens).
- [dreamer_v3_implementation_code_review](../../reviews/dreamer_v3_implementation_code_review.md) — the code-reviewer's reverse-pass check of the implementation reference doc.
- [dreamer_v3_implementation_math_review](../../reviews/dreamer_v3_implementation_math_review.md) — the math-reviewer's paper-equation fidelity check, where the bin-range deviation was first surfaced (math-F1).

### Memory insights (per-finding rationale, methodology rules)

- [20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry](../../../docs/llm_wiki/entries/dreamer_diagnosis/20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry.md) — Experiment 2 verdict + the residual-asymmetry mechanism that picked Experiment 3.
- [20260510_2240_reference_impl_compare_only_act_intersections](../../../docs/llm_wiki/entries/subagent_engineering/20260510_2240_reference_impl_compare_only_act_intersections.md) — the rule for handling reference-impl comparisons.
- [20260510_2241_residual_error_pattern_directs_next_fix](../../../docs/llm_wiki/entries/subagent_engineering/20260510_2241_residual_error_pattern_directs_next_fix.md) — the methodology rule for sequencing fixes via residual-error pattern matching.
- *(prior — predecessor summary)* [20260509_1534_wm_reward_head_localized_failure_a1](../../../docs/llm_wiki/entries/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md), [20260509_1535_conventional_fixes_battery_verdict_predator_refute](../../../docs/llm_wiki/entries/dreamer_diagnosis/20260509_1535_conventional_fixes_battery_verdict_predator_refute.md), [20260508_1431_diagnostic_battery_refutes_four_fixes](../../../docs/llm_wiki/entries/dreamer_diagnosis/20260508_1431_diagnostic_battery_refutes_four_fixes.md), [20260508_1432_probe_refutes_imagined_death_absence](../../../docs/llm_wiki/entries/dreamer_diagnosis/20260508_1432_probe_refutes_imagined_death_absence.md).

### Working files (raw analyzer extractions)

- [20260510_211404_wm_imagination_test_Z1](../../../tmp/20260510_211404_wm_imagination_test_Z1.md) (+ matching `.json`) — Experiment 2 raw output.
- [20260510_211404_z1_wandb_extract](../../../tmp/20260510_211404_z1_wandb_extract.md) — Experiment 2 WandB metric extract used for the residual-asymmetry analysis.
- *(prior anchor)* [20260509_wm_imagination_test_A1](../../../tmp/20260509_wm_imagination_test_A1.md) (+ matching `.json`) — Experiment 0d output, used as the comparison baseline for Experiment 2.
- *(in-flight)* Experiment 3 outputs will be at `tmp/20260511_<HHMMSS>_wm_imagination_test_Z2.{json,md}` once the run finishes and the analyzer runs.

### Diary days covering this window

- *(prior — extends the predecessor summary's coverage)* [2026-05-07](../../diary/2026-05-07.md), [2026-05-08](../../diary/2026-05-08.md), [2026-05-09](../../diary/2026-05-09.md).
- [2026-05-10](../../diary/2026-05-10.md) — Experiment 1 (sheeprl-comparison) docs landed; Experiment 2 (Z1) launch + analysis; Experiment 3 (Z2) launch; this summary's diary note row.

### Implementation commits relevant to this window

- `7186606` — Experiment 1 (sheeprl-comparison §9 added to the implementation reference doc).
- `cff1faf` — fold of the four sheeprl-comparison candidates into the §6 deviation list + Experiment 2 plan written.
- `66665cd` — frontmatter type fix + develop-docs INDEX regen (made by parent Claude as a side-cleanup).
- `a88002f` — Experiment 2 code change (zero-init wired into the network construction).
- `dad09e1` — Experiment 2 diary row.
- `b55dee4` + `23d10cb` — Experiment 2 launch (training-runner diary row + train_command audit trail).
- `c4b7633` — Experiment 2 verification report (analyzer fill-in with the H2 verdict).
- `c9164b0` — Experiment 3 plan written.
- `6a66c7a` — Experiment 3 diary row (plan-write).
- `f5df600` + `8089ee2` — Experiment 3 code change (paper-canonical bin construction).
- `a3cdac2` + `564ffcf` — Experiment 3 launch (training-runner diary row + train_command audit trail).
- `69f10ae` — three memory insights captured from this window (Z1 verdict + the two methodology rules).
- *(prior anchors)* `d66cb9c` (predecessor summary's reconciliation), `f08d18b` (predecessor summary's memory captures).

---

## 6. Reading order if you have 10 minutes

1. **This document** (5 min) — gets you the headline finding (zero-init helps; paper-canonical bins is the next test, in flight) and what each Z2 outcome would mean.
2. Memory insight [20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry](../../../docs/llm_wiki/entries/dreamer_diagnosis/20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry.md) — Experiment 2 verdict with the load-bearing pos/neg asymmetry numbers (3 min).
3. The Verification Report section of [dreamer_zero_init_reward_critic_fix](../../develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md) — concrete A1-vs-Z1 metric tables and the per-channel breakdown (2 min).

If you have 30 minutes, also read:

- The §6 deviation list in [dreamer_v3_implementation](../../project/concepts/dreamer_v3_implementation.md) — items 2, 27–30 are the active candidates; the rest is context for what the cascade is working through.
- The §9 sheeprl-comparison section in the same document — Experiment 1's full output, including the rows of the differences table that the cascade has not yet acted on.
- The two methodology insights from this window — [20260510_2240_reference_impl_compare_only_act_intersections](../../../docs/llm_wiki/entries/subagent_engineering/20260510_2240_reference_impl_compare_only_act_intersections.md) and [20260510_2241_residual_error_pattern_directs_next_fix](../../../docs/llm_wiki/entries/subagent_engineering/20260510_2241_residual_error_pattern_directs_next_fix.md) — the rules for using external reference impls as comparators and for sequencing fixes by residual error.
