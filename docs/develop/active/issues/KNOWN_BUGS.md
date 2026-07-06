---
title: "Known Bugs"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-06
---

# Known Bugs

## What this is (plain-language entry point)

This is a **single-glance registry of the bugs this project has found**, so that future
planning and development does not re-discover the same problem from scratch. Each row is a
one-line summary; the real detail lives behind the links (a diagnosis doc, a fix plan, a
memory insight, or a fix commit). It is **not** a place to reproduce the analysis — open the
linked source for that. **This doc is intentionally a compact index, not an encyclopedia**:
it covers the project's whole git history but keeps only bugs a future planner/developer
benefits from remembering. For on-demand depth on any row (or bugs deliberately left out),
ask the `bug-curator` agent, which reconstructs the full story from git and memory.

**How to read it.** Rows are grouped by status. **Open / undecided** items (things that still
need a decision) come first, because those are the ones that can bite a new plan. **Fixed**
items follow — split into the *Fable 5 re-diagnosis cluster* (a second whole-pipeline sweep,
2026-07-04, see [[00_combined_diagnosis]]), the earlier *v3.0 pipeline-audit cluster*
(2026-07-04, see [[v3_pipeline_correctness_diagnosis]]), and *older historical fixes*.
A short **latent / needs-verification** section lists findings that were recorded but never
confirmed closed. Each row leads with a short plain-English name; where it helps cross-reference,
the diagnosis doc's finding label (Finding A, B, E, F, …) is kept in parentheses at the end.

**Severity** is the practical blast radius: *High* = distorts what a live training run learns, or
silently trains on the wrong environment; *Med* = affects analysis numbers or a narrower path;
*Low* = deprecated/unused path, cosmetic, or robustness-only.

**The headline bug** (now fixed) was that the environment punished an agent for *surviving to the
time limit* exactly as if it had died (a large −100 penalty), and the learner threw away its
future-value estimate on those endings — this predated the project's `main` branch, so every
historical run shares the distortion (the "survival punished like death" family of rows). The
other recurring theme is **silent config/environment drift**: a config layer, a checkpoint's
training stage, or an evaluation environment quietly not being what it looked like (config
inheritance ignored, eval using the wrong stage, ghost predators).

> **Process hazard (not a code bug, but read it).** This project's working copy lives on a shared
> NAS with **no symlink support**, and multiple Claude sessions run against the same checkout. During
> the v3.0 audit, a concurrent session's `git reset --hard` **wiped an uncommitted, verified fix**
> (the eval-video noise-visibility change — the "noise invisible in eval video" row below; it was
> later recovered from a git stash and committed), and a stray `rm -f` deleted another session's
> scratch. Commit early; snapshot untracked data before any reset/merge/branch switch. See memory
> insight `20260528_0218_git_lock_parallel_session_contamination` and the git-safety rules in the
> project `CLAUDE.md`.

---

## Open / undecided (needs a decision)

| Bug | What happened | Status | Severity | Area | Detail links |
|-----|---------------|--------|----------|------|--------------|
| **DreamerV3 never tells the model an episode started during collection** | The action-selection path hard-codes the "episode just started" flag to false, so during experience collection the recurrent world-model state and the previous action are **never reset at episode boundaries** — and the first observation after a reset never enters the replay buffer. Rated Med in the original diagnosis; the H6/H7 fix review re-flagged it as **the remaining gap** in making training and inference see the same data. | **OPEN** | Med | DreamerV3-NNX collection (`get_action`) | [[05_dreamer_v3_nnx]] · `docs/reviews/review_h6h7_dreamer_v3_world_model.md` |
| **Dreamer-srl world-model loss can spike to astronomical values (no gradient clipping)** | During live smoke runs for the H5 fix verification, the world-model loss transiently jumped to ~1e29–1e31 before recovering. Consistent with the already-recorded deviation that dreamer_srl **omits the reference implementation's gradient clipping** (diagnosis Finding 3) — previously theoretical, now with empirical evidence it bites in practice, so its priority is raised. | **OPEN — priority raised** | Med | dreamer_srl world-model training | [[04_dreamer_srl]] Finding 3 · [[fix_plan_h5_dreamer_srl_buffer_reset]] Verification Report §5 · `tmp/20260706_h5_speed_verify.log` |
| **Plain-PPO MC returns also lack the window-edge bootstrap** | The plain-PPO trainer has the **identical** Monte-Carlo window-edge defect that H4 fixed for rPPO (returns cut to zero at each rollout-window boundary instead of bootstrapping); deliberately left unfixed as out of scope. | **OPEN** | Low (no live config uses plain PPO) | ppo trainer (`src/models/ppo_trainer.py:80`) | [[fix_plan_h4_mc_window_bootstrap]] |
| **Three stale env tests reference retired configs** | Two cases in the truncation-not-death test (`tests/env/test_truncation_not_death.py`) and one in the inactive-animal-offgrid test (`tests/env/test_inactive_animal_offgrid.py`) still point at basic-ladder configs retired by the 2026-07-02 re-leveling (`b093023`). Red since then; formerly masked by the config soft-fail that the H3 fix removed. Need their config paths updated to the re-leveled ladder. | **OPEN** | Low (test hygiene) | env tests | [[fix_plan_h1h2h3_resume_config]] (Verification Report, known-red baseline) |
| **Dreamer-srl offline world-model smoke test red** | The offline world-model smoke test fails deterministically with "Only 36 valid starting states (< 50)". Empirically verified pre-existing and independent of the H1–H4 fixes; root cause not yet triaged. | **OPEN — needs triage** | Low–Med | dreamer_srl tests (`tests/scripts/test_dreamer_srl_offline_wm_test.py`) | [[fix_plan_h1h2h3_resume_config]] (Verification Report) |
| **Behavior-metric episode-end rule undecided** (M1/M2) | For the interrupted-feeding (M1) and bush-dive (M2) behavior metrics, there is **no agreed rule for what to do with events still in progress when an episode ends** — three conflicting write-ups exist (finish them / drop them / leave them out of the denominator). Needs one decision. | **UNDECIDED — needs user call** | Med (analysis numbers, not training) | behavior measures (`src/behavior/accumulators.py`) | [[v3_pipeline_correctness_diagnosis]] Finding M1/M2 · `docs/reviews/diag_v3_pipeline_math.md` |
| **CLI overrides not saved to config** (Finding L4) | When you pass model-size flags on the command line (`--hidden_size`, `--num_steps`, `--lr`), the values are **not written into the saved config**, so a later evaluation rebuilds the model at the wrong size and fails to restore. A sibling of the same kind: a dead config key means `--no-satiation` isn't reflected in the saved config, so re-evaluation runs with satiation back *on*. | **OPEN — follow-up** | Low (rarely-used flags; sizes usually set in YAML) | `train.py` config-save | [[v3_pipeline_correctness_diagnosis]] §4.4 (L4) + §4.5 |
| **Dreamer-srl checkpoints drop the optimizer's momentum on save** | Saved checkpoints keep only the world-model/actor/critic/target-critic weights — the Adam optimizer's momentum state is never saved — so resuming or continuing a run silently restarts optimizer momentum from zero. | **OPEN** | Med (resume / continual-learning correctness) | dreamer_srl checkpoint (`src/algorithms/dreamer_srl/checkpoint.py:85-88`) | memory `20260609_1726_doc_audit_surfaces_latent_bugs` |
| **"Chasing rabbit" stays glued to the agent after contact** | The post-contact pause that normally separates agent and predator only fires for *damaging* animals; a non-damaging chased animal (e.g. a rabbit) gets no pause and keeps riding the agent's cell at zero distance — skews the chasing-rabbit / hypervigilance behavior read. | **OPEN** | Med (behavior-experiment interpretation, not training correctness) | env core (`core.py:629`) | memory `20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride` |

## Confirmed NOT a bug

| Item | What was investigated | Verdict | Area | Detail links |
|------|-----------------------|---------|------|--------------|
| **FiLM gain shared across senses (intended)** | Whether the FiLM modulator applying one shared gain across sensory channels was a mistake. | **INTENDED — matches the design, not a bug** | modulation / FiLM | `docs/reviews/diag_v3_pipeline_math.md` |

---

## Fixed — Fable 5 re-diagnosis cluster (2026-07-05)

Source of record: [[00_combined_diagnosis]] (roll-up of the 2026-07-04 re-diagnosis; per-area
reports 01–07 in the same folder). Implementation + Verification Reports live in the fix plans
[[fix_plan_h1h2h3_resume_config]], [[fix_plan_h4_mc_window_bootstrap]],
[[fix_plan_h5_dreamer_srl_buffer_reset]], [[fix_plan_h6h7_dreamer_v3_world_model]],
[[fix_plan_h8h9_eval_output_correctness]], and [[fix_plan_h10_dreamer_batch_bm]].

| Bug | What happened | Status | Severity | Area | Fix commit + detail |
|-----|---------------|--------|----------|------|---------------------|
| **Resume silently restored nothing — trained from random weights** (H1) | rPPO `--load-checkpoint` never actually restored: the saved parameter paths didn't match the live model's (a DictKey/GetAttrKey naming mismatch), the resulting error was **swallowed**, and training continued from fresh random weights while looking like a resume. Closes the long-latent "checkpoint restore may not map onto model" risk (memory `20260509_1536`) — confirmed real, now fixed; restore fails loudly on any mismatch. | FIXED | **High** (silent fake resume) | `train.py` checkpoint restore | `9ee2a30` · [[fix_plan_h1h2h3_resume_config]] · [[00_combined_diagnosis]] H1 |
| **Continual resume ran the wrong stage's world** (H2) | Resuming a continual run restored the stage counter, which **suppressed the environment rebuild** for that stage — so a stage-N resume kept training on the stage-0 world with stage-N bookkeeping. | FIXED | **High** (wrong-environment training) | `train.py` continual resume | `9ee2a30` · [[fix_plan_h1h2h3_resume_config]] · [[00_combined_diagnosis]] H2 |
| **Typo'd config path silently trained on the default env** (H3) | A missing or misspelled `--config` path produced only a warning; the empty config fell through to the built-in default environment and the run proceeded with **no error**. Now a bad path fails loudly. | FIXED | **High** (silent wrong-environment training) | config loader (`src/utils/config.py`) | `9ee2a30` · [[fix_plan_h1h2h3_resume_config]] · [[00_combined_diagnosis]] H3 |
| **MC returns had no bootstrap at the rollout-window edge** (H4) | rPPO's Monte-Carlo return mode — the live mode in **all 13 live configs** — cut returns to zero at every 128-step rollout-window boundary instead of bootstrapping from the value estimate, position-dependently biasing every value target (episodes run to 500 steps). **Comparability caveat: runs trained before this fix are not comparable to post-fix runs.** | FIXED | **High** (biased value targets in every live rPPO run) | rPPO trainer (`recurrent_ppo_trainer.py`) | `5488c98` · [[fix_plan_h4_mc_window_bootstrap]] · [[00_combined_diagnosis]] H4 |
| **Dreamer-srl leaked the previous episode's death into each new episode's first step** (H5) | The replay buffer's staged row was not cleared at episode boundaries, so each new episode's **first row carried the previous episode's terminal reward and death flag** — the world model trained on episodes that appear to start with a death. **Comparability caveat: dreamer_srl runs trained before this fix are not comparable to post-fix runs.** | FIXED | **High** (corrupts world-model training data) | dreamer_srl replay buffer | `b1dd90a` · [[fix_plan_h5_dreamer_srl_buffer_reset]] · [[00_combined_diagnosis]] H5 |
| **DreamerV3 paired each action with the wrong step's observation** (H6) | The NNX replay stored the **pre-step** observation with each action, so every (observation, action) pair the world model trained on was one step out of line with what the agent sees at inference. Fixed by storing the **arrival** observation at collection time (not by shifting at train time). **Comparability caveat: DreamerV3-NNX runs trained before this fix are not comparable to post-fix runs.** | FIXED | **High** (train-vs-inference misalignment) | DreamerV3-NNX replay | `bfb3780` · [[fix_plan_h6h7_dreamer_v3_world_model]] · [[00_combined_diagnosis]] H6 |
| **DreamerV3 replay spliced two environments into one training sequence after wrap** (H7) | Once the replay buffer wrapped around, a sampled training sequence could straddle the wrap point and **mix two different parallel environments' data mid-sequence**. Fixed by flooring buffer capacity to a multiple of the sequence length (the logged capacity drops 1,000,000 → 999,936 — expected). | FIXED | **High** (corrupt training sequences after wrap) | DreamerV3-NNX replay | `bfb3780` · [[fix_plan_h6h7_dreamer_v3_world_model]] · [[00_combined_diagnosis]] H7 |
| **Eval stats CSV columns mislabeled** (H8) | The per-step evaluation CSV omitted one sensor header while values stayed positional, so sensor columns held their **neighbour's** numbers — and fix-round planning found a **worse variant in obstacle envs**, where each data row is one field longer than the header so pandas misassigns *every* column. Old CSVs are re-derivable: the remap recipe is in the fix plan's Consumer analysis. | FIXED | **High** (wrong numbers in study analyses; training unaffected) | eval CSV (`evaluation_core.py`) | `332ce9f` · [[fix_plan_h8h9_eval_output_correctness]] · [[00_combined_diagnosis]] H8 · [[06_evaluation_path]] |
| **Offline interrupted-feeding rate was structurally always zero** (H9) | The offline replay path that current studies consume (`online_replay.json`) checked the interrupted-feeding measure **before** updating the eating timer it depends on, so the rate was 0.0 by construction. Old `online_replay.json` files are recomputable from the saved `episodes/*.npz`. | FIXED | **High** (study numbers; training unaffected) | eval replay (`eval_rollout.py`) | `332ce9f` · [[fix_plan_h8h9_eval_output_correctness]] · [[00_combined_diagnosis]] H9 · [[07_behavior_measures]] |
| **DreamerV3 batch loop corrupted behavior-metric episode boundaries** (H10) | The DreamerV3 batch training path finalised and reset the behavior measures in the wrong order at episode boundaries, corrupting the per-episode metrics. Dormant in practice — the measures are **off by default** — but would bite anyone switching them on. | FIXED | **High** (dormant; measures off by default) | DreamerV3 batch path / behavior measures | `bb47f94` · [[fix_plan_h10_dreamer_batch_bm]] · [[00_combined_diagnosis]] H10 |

## Fixed — v3.0 pipeline-audit cluster (2026-07-04)

Source of record: [[v3_pipeline_correctness_diagnosis]] (final findings table) and, for the
reward bug, the fix plan [[FIX_TRUNCATION_TREATED_AS_DEATH]].

| Bug | What happened | Status | Severity | Area | Fix commit + detail |
|-----|---------------|--------|----------|------|---------------------|
| **Config inheritance ignored** | `train.py` loaded a config **without applying its `extends:` inheritance**, so every inherited layer (perceptual noise, random start ranges, combined-predator scene) was **silently dropped** — training ran on a stripped-down environment that *looked* correct. | FIXED | **High** (silent wrong-environment training) | config loader | `22c73ba` · memory `20260703_1507_train_py_ignores_extends_drops_layers` · [[v3_pipeline_correctness_diagnosis]] Purpose + E2E-3 |
| **Survival punished like death** (Finding B, Part 1) | Surviving to the step limit (the *success* outcome) was hit with the full **−100 death penalty**, the same as dying — swamping the learning signal by roughly 500×. Pre-existing (identical to `main`). | FIXED | **High** (corrupts the survival objective) | env reward (`core.py`) | `ef0fd25` · [[FIX_TRUNCATION_TREATED_AS_DEATH]] Part 1 |
| **Value estimate dropped on timeout** (Finding B, Part 2) | On a time-limit ending the trainer **threw away its estimate of future reward** instead of keeping it, teaching the critic to expect a cut-off single-step reward rather than the real continuation. **Scope caveat found later:** `3c60f6f` fixed the **GAE branch only, which no live config uses** — live runs (MC mode) were only actually corrected by the H4 window-edge fix (`5488c98`, Fable 5 cluster above). | FIXED | **High** (corrupts value targets) | rPPO trainer | `3c60f6f` · [[FIX_TRUNCATION_TREATED_AS_DEATH]] Part 2 · [[02_rppo_stack]] |
| **Plain-PPO timeout bootstrap bug** (Finding B sibling) | Same "value dropped on timeout" defect in the plain-PPO trainer, plus a related bug where the next-state value was read *after* the auto-reset every step. | FIXED | Low (no live config uses plain PPO) | ppo trainer | `926c2c3` |
| **Dreamer treats timeout as death** (Finding B sibling) | DreamerV3's "will the episode continue?" head was trained to treat a time-limit ending as a real death; now it only counts genuine death. | FIXED | Med | dreamer trainer | `5b093bf` |
| **Plain-PPO advantage baseline bug** (Finding B sibling) | Plain-PPO computed its advantage against a **value shifted from the neighbouring step** instead of the value of the current state. | FIXED | Low (no live config uses plain PPO) | ppo trainer | `8c1ad2f` |
| **Eval uses wrong stage's environment — live path** (Finding E) | For multi-stage (continual) runs, evaluation rebuilt the environment from the **first stage's config**, so a checkpoint trained in a later stage was tested against the wrong (early-stage) world. Fixed on the live eval path (it now detects the checkpoint's own stage). | FIXED | Med (continual runs only) | eval | `a3ab4cc` · [[v3_pipeline_correctness_diagnosis]] Finding E |
| **Eval uses wrong stage's environment — old script** (Finding L2) | The same wrong-stage defect in the deprecated `evaluation.py --all` batch loop; it now evaluates each checkpoint against its own stage's config. | FIXED | Med (deprecated/demo path) | eval | `863052f` |
| **Old eval script crashes on modulated/Dreamer checkpoints** (Finding A) | The deprecated `evaluation.py` rebuilt modulated checkpoints from a stale key list (crashed on a missing `memory_clip` key) and passed the wrong object type for DreamerV3. Demo-only path; the live eval path was unaffected. | FIXED | Med (deprecated path) | eval model-rebuild | `2ad9104` · [[v3_pipeline_correctness_diagnosis]] Finding A |
| **Eval silently keeps random weights** (Finding L3) | The deprecated `evaluation.py` restore would **keep freshly-initialised random weights** for any layer missing from the checkpoint, with no error — so evaluation could run on partly-random weights. Now it asserts the restore is complete. | FIXED | Low (deprecated eval only; live path already immune) | eval restore | `2ad9104` |
| **Parity test silently skipped** (Finding F) | The environment "parity" golden-snapshot test still read old key names after the fixtures were regenerated, so its position checks **silently turned into no-ops** — committing the new fixtures alone would have switched the gate off. Test migrated to the new keys. | FIXED | Low (test hygiene, must-fix-before-commit) | parity test | `0bebe06` · [[v3_pipeline_correctness_diagnosis]] Part C |
| **Behavior-metric counts mismatched** (Finding C-math #2) | The M1/M2 behavior metrics **counted the numerator and denominator at different moments**, biasing the rates (preferentially dropping death-by-predator interruptions). (The still-open *episode-end rule* is the separate undecided row above.) | FIXED | Med (analysis numbers) | behavior measures | `3e1e53e` |
| **Observability gate fires at step 0** (Finding G2) | An observability gate could trigger on the very first step because the agent could start already in contact; the start position is now fixed so there is no step-0 contact. | FIXED | Med | config / env | `84014e4` |
| **CLI flags not persisted to saved config** (Finding G3) | Command-line overrides like `--no-satiation` were not written into the saved config; now persisted. (Distinct from the still-open L4 row, which covers model-size scalars.) | FIXED | Med | `train.py` config-save | `75976e2` |
| **Online behavior measures on by default** | The online behavior measures ran during every training by default, redundant with eval-environment testing; now off by default. | FIXED | Low (perf/hygiene) | config | `9eacf82` |
| **Noise invisible in eval video** | Evaluation videos showed **no visible sensory noise** because the recorder fell back to the noisy observation instead of also recording the clean "true" observation for the side-by-side panel. Recovered from a git stash (after a concurrent reset wiped it — see the process-hazard note above) and committed. | FIXED | Low–Med (diagnostics only) | eval / video (`src/utils/evaluation_core.py`) | `80d3b70` · [[v3_pipeline_correctness_diagnosis]] §4.3 · memory `20260703_1508_eval_video_drops_true_obs_no_noise_contrast` |
| **Comment / cosmetic nits** | Cosmetics: noted that the temperature-clip lower bound is effectively 0.5 (the 0.1 bound was dead), and corrected a value-loss code comment (plain MSE, not clipped). | FIXED | Low (nits) | neuromod / trainer comments | `5caa0df`, `3d34dd8` |

## Fixed — historical (earlier than the v3.0 audit)

| Bug | What happened | Status | Severity | Area | Fix commit + detail |
|-----|---------------|--------|----------|------|---------------------|
| **Ghost predators stuck to agent** | Inactive (this-episode-unused) predators were hidden from sensing and damage but **still moved and rendered**, so they drifted onto the agent as invisible, harmless "ghosts". Now parked off-grid each step. | FIXED | Med | env entities | `3634887` · memory `20260629_1723_ghost_predator_inactive_slots_render` |
| **Dead resource slots revived** | Inactive food slots came **back to life on step 1** because the code treated "inactive" and "eaten" as the same flag; fixed with a separate allocation mask. | FIXED | Med | env resources | `db8bd03` · memory `20260623_0144_inactive_resource_slots_revive_respawn` |
| **Continual stage-transition crash** | A stage change in continual training **crashed** (a missing-name error) because a large refactor missed one of six reset sites; latent for ~40 days. Fixed with the shared reset loop. | FIXED | Med (continual crash) | behavior measures / continual | memory `20260622_1704_continual_bm_transition_nameerror_refactor_drift` |
| **Dreamer recompiles every iteration (hang)** | The dreamer_srl gradient loop had no persistent compilation, so it **recompiled a huge graph every iteration** — the multi-hour training hang. Fixed with a persistent compile + constant step count. | FIXED | High (training unusable) | dreamer_srl | `0119e87` · memory `20260630_1720_dreamer_srl_train_step_jit_compile_once` |
| **Dreamer per-step CPU→GPU re-upload** | The dreamer_srl gradient loop re-uploaded data from CPU to GPU on **every step** instead of once per iteration ("Option S": a single upload now covers the whole iteration). | FIXED | Med–High (training throughput) | dreamer_srl (`dreamer_srl_main.py`) | supersedes memory `20260519_1507_dreamer_srl_v2_cpu_buffer_regression` (its `valid_until` passed; verified resolved in live code 2026-07-04) |
| **Dreamer reset-storm recompile (out-of-memory)** | The per-step reset sized arrays to the **variable number of finished environments**, forcing a fresh compile per size and eventually running out of GPU memory / stalling all runs. Fixed with a fixed-width masked reset. | FIXED | High (OOM / stall) | dreamer_srl | memory `20260622_1746_dreamer_srl_recompile_storm_done_count` |
| **Dreamer REINFORCE resamples actions** | The dreamer_srl v1 policy loss **re-drew actions at loss time**, pairing the probability of a *new* action with the advantage of the *old* one — the v1 root-cause failure. Fixed by carrying the taken action through the loss. | FIXED | High (was the v1 root cause) | dreamer_srl | memory `20260518_1512_reinforce_resampling_bug_imag_action_threading` |
| **Lazy import crashes hours into training** | Four runs crashed ~7 hours in because a helper was **imported only on first use**, after a parallel session had changed a data schema; the mismatch surfaced only at that late first call. | FIXED | Med (delayed crash) | dreamer_srl / tooling | memory `20260529_1826_lazy_import_schema_drift_first_call_crash` |
| **Silent encode/decode layout drift** | A knob-gated change to Dreamer's reward two-hot encoding shipped in the trainer but **not** in an offline diagnostic script, so the tool decoded the model's outputs with the *old* layout and reported wrong numbers with **no error** — a fake 5.7× regression. General class: any auxiliary tool falling out of sync with production's evolving data layout. | FIXED | Med (silent wrong analysis; recurrence-prone) | dreamer encode / diagnostics | `f5df600` + `1703a4c` · memory `20260511_1535_encode_decode_flag_mismatch_silent_class_bug` |
| **Noise painted on wrong sensory channel** | The evaluation sensory visualiser assumed a different modality order than the observation vector actually used, so perceptual noise showed up on the **wrong channel**; fixed by making channel order YAML-driven and unifying the true-observation computation. | FIXED | Med (diagnostics mislead; shaped current ordering design) | eval / sensory viz | `eee4f08` (diagnosis) → `2e4ac34` |
| **JAX arrays returned read-only → crash** | The JAX vector-env handed back **non-writable** arrays; a downstream numpy in-place write crashed the run. Now returns writable numpy copies. | FIXED | Med (training crash) | jax_vector_env | `d729e2c` |
| **Behavior-metric NaN on zero denominator** | A behavior ratio (eat-under-threat) produced **NaN** whenever its denominator was zero (no safe-eat events), poisoning logged metrics; guarded. Same zero-denominator class recurs across ratio metrics. | FIXED | Low–Med (analysis numbers; recurrence class) | behavior measures | `e5e1155` |
| **Dead config key silently ignored** | The `memory_clip` clamp config key existed but was **never enforced** — a no-op that looked active — so the neuromodulator's memory value went unclamped; later actually implemented. (This same key later crashed the old eval rebuild — Finding A above.) | FIXED | Low–Med (silent no-op; dead-key class) | neuromod (NeuromodulatorRNN) | `014a195` → `ea4bb6e` |
| **Fractional `attack_range`/`detection_range` bound silently had no effect** | A range written as `[2,3]` sampled as a float and, compared against an integer grid distance, behaved identically to `[2,2]` — the top of the range was dead with no warning. Now both fields sample as **inclusive integers** (`[2,3]` means `{2,3}`), with a guard rejecting fractional bounds. | FIXED | Low–Med (silent config drift) | env config (attack_range/detection_range) | `7ff8d1f` · [[INCLUSIVE_INTEGER_RANGE_SAMPLING]] · memory `20260703_0344_attack_range_float_threshold_gotcha` |

## Latent / needs-verification (recorded, not confirmed closed)

| Finding | What was recorded | Status | Severity | Area | Detail links |
|---------|-------------------|--------|----------|------|--------------|
| **Env doc-audit latent findings** | An environment documentation audit surfaced ~13 latent findings; the top two: **over-eating never actually ends the episode**, and the **termination-reason field is unreliable when a body system is switched off**. Status of each not individually tracked. Also from this audit: `auto_reset_step()` in `src/environment/wrapper.py:35` ignores its own `key` argument — **verified dead code, no callers anywhere in `src`/`scripts`** — not an active bug, just a remove-on-touch candidate. | LATENT — verify | Med | env body / termination | memory `20260609_1726_doc_audit_surfaces_latent_bugs` |
