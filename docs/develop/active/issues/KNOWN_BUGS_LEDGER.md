---
title: "Known-Bugs Ledger"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Known-Bugs Ledger

## What this is (plain-language entry point)

This is a **single-glance registry of the bugs this project has found**, so that future
planning and development does not re-discover the same problem from scratch. Each row is a
one-line summary; the real detail lives behind the links (a diagnosis doc, a fix plan, a
memory insight, or a fix commit). It is **not** a place to reproduce the analysis — open the
linked source for that.

**How to read it.** Rows are grouped by status. **Open / undecided** items (things that still
need a decision) come first, because those are the ones that can bite a new plan. **Fixed**
items follow — split into the recent *v3.0 pipeline-audit cluster* (a whole-pipeline correctness
sweep run on 2026-07-04, see [[v3_pipeline_correctness_diagnosis]]) and *older historical fixes*.
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
| **Behavior-metric episode-end rule undecided** (M1/M2) | For the interrupted-feeding (M1) and bush-dive (M2) behavior metrics, there is **no agreed rule for what to do with events still in progress when an episode ends** — three conflicting write-ups exist (finish them / drop them / leave them out of the denominator). Needs one decision. | **UNDECIDED — needs user call** | Med (analysis numbers, not training) | behavior measures (`src/behavior/accumulators.py`) | [[v3_pipeline_correctness_diagnosis]] Finding M1/M2 · `docs/reviews/diag_v3_pipeline_math.md` |
| **CLI overrides not saved to config** (Finding L4) | When you pass model-size flags on the command line (`--hidden_size`, `--num_steps`, `--lr`), the values are **not written into the saved config**, so a later evaluation rebuilds the model at the wrong size and fails to restore. A sibling of the same kind: a dead config key means `--no-satiation` isn't reflected in the saved config, so re-evaluation runs with satiation back *on*. | **OPEN — follow-up** | Low (rarely-used flags; sizes usually set in YAML) | `train.py` config-save | [[v3_pipeline_correctness_diagnosis]] §4.4 (L4) + §4.5 |

## Confirmed NOT a bug

| Item | What was investigated | Verdict | Area | Detail links |
|------|-----------------------|---------|------|--------------|
| **FiLM gain shared across senses (intended)** | Whether the FiLM modulator applying one shared gain across sensory channels was a mistake. | **INTENDED — matches the design, not a bug** | modulation / FiLM | `docs/reviews/diag_v3_pipeline_math.md` |

---

## Fixed — v3.0 pipeline-audit cluster (2026-07-04)

Source of record: [[v3_pipeline_correctness_diagnosis]] (final findings table) and, for the
reward bug, the fix plan [[FIX_TRUNCATION_TREATED_AS_DEATH]].

| Bug | What happened | Status | Severity | Area | Fix commit + detail |
|-----|---------------|--------|----------|------|---------------------|
| **Config inheritance ignored** | `train.py` loaded a config **without applying its `extends:` inheritance**, so every inherited layer (perceptual noise, random start ranges, combined-predator scene) was **silently dropped** — training ran on a stripped-down environment that *looked* correct. | FIXED | **High** (silent wrong-environment training) | config loader | `22c73ba` · memory `20260703_1507_train_py_ignores_extends_drops_layers` · [[v3_pipeline_correctness_diagnosis]] Purpose + E2E-3 |
| **Survival punished like death** (Finding B, Part 1) | Surviving to the step limit (the *success* outcome) was hit with the full **−100 death penalty**, the same as dying — swamping the learning signal by roughly 500×. Pre-existing (identical to `main`). | FIXED | **High** (corrupts the survival objective) | env reward (`core.py`) | `ef0fd25` · [[FIX_TRUNCATION_TREATED_AS_DEATH]] Part 1 |
| **Value estimate dropped on timeout** (Finding B, Part 2) | On a time-limit ending the trainer **threw away its estimate of future reward** instead of keeping it, teaching the critic to expect a cut-off single-step reward rather than the real continuation. | FIXED | **High** (corrupts value targets) | rPPO trainer | `3c60f6f` · [[FIX_TRUNCATION_TREATED_AS_DEATH]] Part 2 |
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
| **Dreamer reset-storm recompile (out-of-memory)** | The per-step reset sized arrays to the **variable number of finished environments**, forcing a fresh compile per size and eventually running out of GPU memory / stalling all runs. Fixed with a fixed-width masked reset. | FIXED | High (OOM / stall) | dreamer_srl | memory `20260622_1746_dreamer_srl_recompile_storm_done_count` |
| **Dreamer REINFORCE resamples actions** | The dreamer_srl v1 policy loss **re-drew actions at loss time**, pairing the probability of a *new* action with the advantage of the *old* one — the v1 root-cause failure. Fixed by carrying the taken action through the loss. | FIXED | High (was the v1 root cause) | dreamer_srl | memory `20260518_1512_reinforce_resampling_bug_imag_action_threading` |
| **Lazy import crashes hours into training** | Four runs crashed ~7 hours in because a helper was **imported only on first use**, after a parallel session had changed a data schema; the mismatch surfaced only at that late first call. | FIXED | Med (delayed crash) | dreamer_srl / tooling | memory `20260529_1826_lazy_import_schema_drift_first_call_crash` |

## Latent / needs-verification (recorded, not confirmed closed)

| Finding | What was recorded | Status | Severity | Area | Detail links |
|---------|-------------------|--------|----------|------|--------------|
| **Checkpoint restore may not map onto model** | A recorded risk that saved checkpoints **don't map cleanly onto the model's parameter tree** on restore (an Orbax-vs-NNX structure skew). Never confirmed fixed. | LATENT — verify | Med (silent wrong-weights risk) | checkpoint restore | memory `20260509_1536_train_py_checkpoint_restore_nnx_skew` |
| **Env doc-audit latent findings** | An environment documentation audit surfaced ~13 latent findings; the top two: **over-eating never actually ends the episode**, and the **termination-reason field is unreliable when a body system is switched off**. Status of each not individually tracked. | LATENT — verify | Med | env body / termination | memory `20260609_1726_doc_audit_surfaces_latent_bugs` |
