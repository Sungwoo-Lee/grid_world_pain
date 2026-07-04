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

**How to read it.** Rows are grouped by status. **Open / parked** items (things that still need
a decision or a re-land) come first, because those are the ones that can bite a new plan.
**Fixed** items follow — split into the recent *v3.0 pipeline-audit cluster* (a whole-pipeline
correctness sweep run on 2026-07-04, see [[v3_pipeline_correctness_diagnosis]]) and *older
historical fixes*. A short **latent / needs-verification** section lists findings that were
recorded but never confirmed closed. Severity is the practical blast radius: *High* = distorts
what a live training run learns or silently trains on the wrong environment; *Med* = affects
analysis numbers or a narrower path; *Low* = deprecated/unused path, cosmetic, or robustness-only.

**The headline bug** (now fixed) was that the environment punished an agent for *surviving to the
time limit* exactly as if it had died (a large −100 penalty), and the learner threw away its
future-value estimate on those endings — this predated the project's `main` branch, so every
historical run shares the distortion (see the "truncation treated as death" rows). The other
recurring theme is **silent config/environment drift**: a config layer, a checkpoint's stage, or
an evaluation environment quietly not being what it looked like (the `extends:` bug, continual-stage
eval, ghost predators).

> **Process hazard (not a code bug, but read it).** This project's working copy lives on a shared
> NAS with **no symlink support**, and multiple Claude sessions run against the same checkout. During
> the v3.0 audit, a concurrent session's `git reset --hard` **wiped an uncommitted, verified fix**
> (the eval-video true-obs change — see row **EVID** below), and a stray `rm -f` deleted another
> session's scratch. Commit early; snapshot untracked data before any reset/merge/branch switch. See
> memory insight `20260528_0218_git_lock_parallel_session_contamination` and the git-safety rules in
> the project `CLAUDE.md`.

---

## Open / parked / needs-decision

| ID | Bug (plain English) | Status | Severity | Area | Detail links |
|----|---------------------|--------|----------|------|--------------|
| **EVID** | Eval-video render-gate fix that makes injected perceptual noise **visible in eval videos** (records noise-free "true obs" alongside the noisy obs) was VERIFIED WORKING but **lost to a concurrent-session `git reset --hard`** before it was committed. | **OPEN — reland** (pending reconstruction + commit) | Low–Med (diagnostics only) | eval / video (`src/utils/evaluation_core.py`) | [[v3_pipeline_correctness_diagnosis]] §4.3 · memory `20260703_1508_eval_video_drops_true_obs_no_noise_contrast` |
| **M12-sem** | Behavior measures M1 (interrupted-feeding) / M2 (bush-dive): what to do with **still-pending candidate events at episode end** has **three conflicting documented semantics** (resolve / drop / exclude-from-denominator). Needs a single adjudicated rule. | **PARKED — needs user decision** | Med (analysis numbers, not training) | behavior measures (`src/behavior/accumulators.py`) | [[v3_pipeline_correctness_diagnosis]] Finding M1/M2 · `docs/reviews/diag_v3_pipeline_math.md` |
| **L4** | Architecture-affecting CLI overrides (`--hidden_size`, `--num_steps`, `--lr`) are **not written back into the saved config**, so eval can rebuild the model with the wrong size (shape-mismatch on restore). Sibling: the dead `environment.with_satiation` key means `--no-satiation` isn't reflected in the saved config → re-eval with satiation *on*. | **OPEN — follow-up** | Low (rarely-used flags; sizes usually in YAML) | `train.py` config-save | [[v3_pipeline_correctness_diagnosis]] §4.4 (L4) + §4.5 |

## Confirmed NOT a bug

| ID | Claim investigated | Verdict | Area | Detail links |
|----|--------------------|---------|------|--------------|
| **FiLM-uni** | FiLM modulator applies a unimodal gain — suspected mis-implementation. | **INTENDED — not a bug** (matches the cited design) | modulation / FiLM | `docs/reviews/diag_v3_pipeline_math.md` |

---

## Fixed — v3.0 pipeline-audit cluster (2026-07-04)

Source of record: [[v3_pipeline_correctness_diagnosis]] (final findings table) and, for the reward
bug, the fix plan [[FIX_TRUNCATION_TREATED_AS_DEATH]].

| ID | Bug (plain English) | Status | Severity | Area | Fix commit + detail |
|----|---------------------|--------|----------|------|---------------------|
| **EXT** | `train.py` loaded a `--config` **without resolving `extends:`**, so every inherited layer (perceptual noise, random start ranges, combined-predator scene) was **silently dropped** — training ran on a stripped-down env that *looked* correct. | FIXED | **High** (silent wrong-env training) | config loader | `22c73ba` · memory `20260703_1507_train_py_ignores_extends_drops_layers` · [[v3_pipeline_correctness_diagnosis]] Purpose + E2E-3 |
| **B-1** | Surviving to `max_steps` (the *success* outcome) was punished with the full **−100 death penalty**, same as dying — dominates the learning signal ~500×. Pre-existing (byte-identical to `main`). | FIXED | **High** (corrupts survival objective) | env reward (`core.py`) | `ef0fd25` · [[FIX_TRUNCATION_TREATED_AS_DEATH]] Part 1 |
| **B-2** | rPPO GAE **zeroed the value bootstrap on timeout** too, training the critic to regress a cut-off single-step reward instead of the discounted continuation. | FIXED | **High** (corrupts value targets) | rPPO trainer | `3c60f6f` · [[FIX_TRUNCATION_TREATED_AS_DEATH]] Part 2 |
| **B-ppo** | Same truncation-bootstrap defect in plain-PPO trainer, plus a `next_value` read-after-auto-reset bug on every step. | FIXED | Low (no live config uses plain PPO) | ppo trainer | `926c2c3` |
| **B-drm** | DreamerV3 continue-head target treated `max_steps` timeout as termination; now built from real death only (`termination_reason >= 2`). | FIXED | Med | dreamer trainer | `5b093bf` |
| **B-base** | Plain-PPO `compute_gae` used a **shifted-neighbour value** as the advantage baseline instead of `V(s_t)`. | FIXED | Low (no live config uses plain PPO) | ppo trainer | `8c1ad2f` |
| **E / L2-live** | Continual-run eval reconstructed the env from **stage-0 config**, so later-stage checkpoints were evaluated against the wrong (first-stage) environment. Fixed on the live `eval_rollout.py` path (auto-detects the checkpoint's own stage). | FIXED | Med (continual runs only) | eval | `a3ab4cc` · [[v3_pipeline_correctness_diagnosis]] Finding E |
| **L2-root** | Same stage-0 defect in the **root `evaluation.py --all`** batch loop — now evaluates each `--all` checkpoint against its own stage config. | FIXED | Med (deprecated/demo path) | eval | `863052f` |
| **A** | Root `evaluation.py` rebuilt modulated checkpoints from a stale 6-key whitelist → `KeyError('memory_clip')`; DreamerV3 passed a plain dict where a `Config` was expected → `AttributeError`. Deprecated demo-only path (live path unaffected). | FIXED | Med (deprecated path) | eval model-rebuild | `2ad9104` · [[v3_pipeline_correctness_diagnosis]] Finding A |
| **L3** | Root `evaluation.py` **best-effort restore kept randomly-initialised weights** for any leaf absent from the checkpoint, with no error — could silently run eval on partly-random weights. Now asserts full restore. | FIXED | Low (root eval only; live path already immune) | eval restore | `2ad9104` |
| **F** | Parity golden-snapshot fixtures were regenerated to the unified `animal_*` schema, but the test still read old `pred_*/neutral_*` keys and **silently skipped** the position asserts — committing fixtures alone would no-op the parity gate. | FIXED | Low (test hygiene, must-fix-before-commit) | parity test | `0bebe06` · [[v3_pipeline_correctness_diagnosis]] Part C |
| **M12-den** | M1/M2 **per-class vs per-tag denominators used different timing instants**, biasing interrupted-feeding / bush-dive rates (preferentially dropping death-by-predator interruptions). (The pending-event *semantics* remain parked — row **M12-sem** above.) | FIXED | Med (analysis numbers) | behavior measures | `3e1e53e` |
| **G2** | Observability gates could fire at step 0 because the agent could start in contact; now use a fixed start position (no step-0 contact). | FIXED | Med | config / env | `84014e4` |
| **G3** | CLI overrides (`--no-satiation` etc.) were not persisted into the saved config; now written back. (Distinct from L4, which covers architecture scalars still unpersisted.) | FIXED | Med | `train.py` config-save | `75976e2` |
| **BM-off** | Online behavior measures ran by default, redundant with eval-env testing; disabled by default. | FIXED | Low (perf/hygiene) | config | `9eacf82` |
| **cos** | Cosmetics: `temp_clip` dead 0.1 lower bound (effective floor 0.5) noted; value-loss comment corrected (plain MSE, not clipped). | FIXED | Low (nits) | neuromod / trainer comments | `5caa0df`, `3d34dd8` |

## Fixed — historical (earlier than the v3.0 audit)

| ID | Bug (plain English) | Status | Severity | Area | Fix commit + detail |
|----|---------------------|--------|----------|------|---------------------|
| **GHOST** | Per-episode entity-count masking gated damage/sensing/obs but **not per-step movement/render**, so inactive "ghost" predators un-parked and stuck to the agent (0 damage, invisible). Now re-parks inactive slots off-grid each step. | FIXED | Med | env entities | `3634887` · memory `20260629_1723_ghost_predator_inactive_slots_render` |
| **RESREV** | Inactive resource slots were **revived on step 1** because `update_resources` overloaded `res_active=False` as "eaten"; fixed with an immutable allocation mask. | FIXED | Med | env resources | `db8bd03` · memory `20260623_0144_inactive_resource_slots_revive_respawn` |
| **BMNAME** | Continual stage-transition `NameError` (`m1_candidates`) introduced by a delete-heavy BMState refactor missing 1 of 6 reset sites; latent ~40 days in the continual + behavior-measures path. Fixed via the canonical `_bm_reset_env` loop. | FIXED | Med (continual crash) | behavior measures / continual | memory `20260622_1704_continual_bm_transition_nameerror_refactor_drift` |
| **DRMHANG** | dreamer_srl gradient `lax.scan` had no outer `@jax.jit`, so the 76k-HLO `train_step` **recompiled every iteration** — the multi-hour training hang. Fixed with a persistent JIT + constant `n_grad_steps`. | FIXED | High (training unusable) | dreamer_srl | `0119e87` · memory `20260630_1720_dreamer_srl_train_step_jit_compile_once` |
| **DRMSTORM** | dreamer_srl per-step env-reset sized arrays to the **variable done-env count** (a PyTorch→JAX port trap), compiling a new XLA executable per width → OOM/stall on all 5 runs. Fixed with a masked fixed-width reset + fixed scan bucket. | FIXED | High (OOM/stall) | dreamer_srl | memory `20260622_1746_dreamer_srl_recompile_storm_done_count` |
| **REINF** | dreamer_srl v1 REINFORCE **re-sampled actions at loss time**, pairing `log_prob(action_NEW)` with `advantage(action_OLD)` — the v1 root-cause failure. Fixed by threading the collected action through the imagined-rollout loss. | FIXED | High (was v1 root cause) | dreamer_srl | memory `20260518_1512_reinforce_resampling_bug_imag_action_threading` |
| **LAZYIMP** | 4 cells crashed ~7h into training at ep 10000: `eval_recording.py` was **lazy-imported after** a parallel session pulled a schema change; the mismatch surfaced only at first lazy-call site. | FIXED | Med (delayed crash) | dreamer_srl / tooling | memory `20260529_1826_lazy_import_schema_drift_first_call_crash` |

## Latent / needs-verification (recorded, not confirmed closed)

| ID | Finding (plain English) | Status | Severity | Area | Detail links |
|----|-------------------------|--------|----------|------|--------------|
| **CKPT-NNX** | A latent `train.py` Orbax-vs-NNX checkpoint-restore **structure skew** — restored leaves may not map cleanly onto the NNX module tree. Recorded during dreamer diagnosis; not confirmed fixed. | LATENT — verify | Med (silent wrong-weights risk) | checkpoint restore | memory `20260509_1536_train_py_checkpoint_restore_nnx_skew` |
| **DOCAUDIT** | Env doc audit surfaced ~13 latent findings; top two: **`overeating_death` never ends the episode**, and **`info['termination_reason']` unreliable when a body system is disabled**. Status of each not individually tracked. | LATENT — verify | Med | env body / termination | memory `20260609_1726_doc_audit_surfaces_latent_bugs` |
