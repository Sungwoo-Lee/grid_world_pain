---
title: "v3.0 Training + Evaluation Pipeline Correctness Diagnosis"
topic: diagnosis
status: active
created: 2026-07-04
last_updated: 2026-07-04
phase: complete
---

# v3.0 Training + Evaluation Pipeline Correctness Diagnosis

## Purpose (plain-language entry point)

**What this is.** A whole-pipeline correctness audit of the v3.0 training and evaluation
code, hunting for the same *class* of bug that recently bit us: `train.py` was loading
environment config files **without resolving the `extends:` inheritance chain**, so every
layer a config inherited from a parent (perceptual noise, random start-state ranges, the
combined-predator scene, etc.) was **silently dropped** — training ran on a stripped-down
environment while the config *looked* correct. That bug is now fixed, but it slipped through
a review that only read source code, so the user wants the same scrutiny applied across the
entire train→eval pipeline for **any** correctness regression introduced by the large v3.0
update wave.

**Why it exists.** v3.0 was a big change: since the branch split off from the shared `main`
line on 2026-04-21, roughly **148 commits touched `train.py`** and **78 touched the
evaluation code**. A change that large, verified only by reading the source, is exactly how
the `extends:` bug survived. The lesson: reading the code is not enough — we also run the
real thing and inspect what it *actually produces*.

**How we check it (method).** Two layers.
1. **Static multi-agent review** — four independent "surfaces", each a different lens:
   - **Surface 1 — JAX correctness** (owned by the `code-reviewer` agent): pytree/vmap/PRNG,
     JIT-recompile traps, Flax state discipline.
   - **Surface 2 — Math** (owned by the `math-reviewer` agent): do the equations still match
     the papers they cite (returns, losses, modulation).
   - **Surface 3 — Config / environment soundness** (owned by the `env-config-auditor` agent):
     YAML↔env consistency, observation-vs-noise sync, mandatory-key discipline.
   - **Surface 4 — Architecture / plan-adherence / train↔eval interface** (this document):
     does each code path do what the plan said, and can training and evaluation ever *silently
     disagree* about the model or the environment.
2. **Ground-truth end-to-end run** — a short real train→checkpoint→eval cycle whose saved
   config, logs, and outputs are inspected directly (not re-derived from source). This is the
   step that would have caught the `extends:` bug. **It is a separate, later step — not run here.**

**Scope.** The nine files at the heart of the pipeline, compared against the pre-v3.0 baseline
(`main`, merge-base commit `ba3fa5c`, 2026-04-21). One uncommitted working-tree change to the
evaluation helper (`src/utils/evaluation_core.py`) is included because it is live in the tree.
The audit also investigates an uncommitted churn in the environment "parity" golden-snapshot
test fixtures (see Part C / the Surface-4 section).

**Reader's shortcut.** If you only read one thing: Surface 4's verdict is that the *primary*
train↔eval path is consistent, with **no v3.0-introduced silent disagreement found**, but there
are a handful of **pre-existing latent interface gaps** (a stale model-reconstruction whitelist
in the deprecated root `evaluation.py`, continual-run eval always using the first-stage
environment, and a best-effort checkpoint restore that can keep randomly-initialised weights
without erroring). The parity-fixture churn is a **schema rename, not an environment regression**
— but committing it as-is would silently weaken the parity test, which needs follow-up.

## Sibling review documents

Each static surface writes its detailed findings into its own review doc; this document is the
shared home that the final reconciliation step pulls them together into.

- [[diag_v3_pipeline_jax]] — Surface 1, JAX correctness (`code-reviewer`)
- [[diag_v3_pipeline_math]] — Surface 2, mathematical faithfulness (`math-reviewer`)
- [[diag_v3_pipeline_config]] — Surface 3, config / environment soundness (`env-config-auditor`)

(Paths: `docs/reviews/diag_v3_pipeline_jax.md`, `docs/reviews/diag_v3_pipeline_math.md`,
`docs/reviews/diag_v3_pipeline_config.md`.)

## Scope manifest

| # | File | Baseline diff | Surface-4 focus |
|---|------|---------------|-----------------|
| 1 | `train.py` | `git diff main...v3.0 -- train.py` (+824/−301) | continual curriculum, budget/loop, model build, config save |
| 2 | `evaluation.py` | `git diff main...v3.0 -- evaluation.py` | true-obs, Orbax restore, model reconstruction |
| 3 | `src/utils/evaluation_core.py` | working-tree change (uncommitted) | true-obs recording gate |
| 4 | `src/environment/config_loader.py` | (extends chokepoint) | EnvParams reconstruction key reads |
| 5 | `scripts/eval/eval_rollout.py` | (post Phase-2 migration) | the *live* eval reconstruction path |
| 6 | `src/models/recurrent_ppo_network.py` | modulation key reads | modulation-config contract |
| 7–9 | continual configs / schedule, parity test + fixtures | — | Part C |

---

## Surface 1 — JAX correctness (code-reviewer)

_To be filled by the `code-reviewer` agent. See [[diag_v3_pipeline_jax]]._

## Surface 2 — Math (math-reviewer)

_To be filled by the `math-reviewer` agent. See [[diag_v3_pipeline_math]]._

## Surface 3 — Config / environment soundness (env-config-auditor)

_To be filled by the `env-config-auditor` agent. See [[diag_v3_pipeline_config]]._

## Surface 4 — Architecture / plan-adherence / train↔eval interface (this doc)

**Verdict: PASS with caveats.** No v3.0-introduced *silent* train↔eval disagreement was found
on the primary pipeline. The continual-learning curriculum and the training budget/loop are
correct and well-guarded. Three **pre-existing** latent interface gaps and one **cosmetic
dead-code** item are flagged below; none is a v3.0 regression, but all deserve a follow-up
plan because v3.0's feature growth has made two of them more likely to bite.

### 4.1 Continual-learning curriculum — PASS

The multi-stage schedule path (`train.py` `_build_continual_schedule`, lines ~154–204, and the
in-loop stage transition, lines ~1199–1285) is correct:

- **Each stage resolves `extends:`.** Stage configs are built by deep-copying the base config
  and merging `load_env_config(p)` (line 195) — the *same* extends-resolving loader that fixes
  the single-config path. A stage file with no `extends:` key loads byte-identically, so this
  is backward-compatible. This is the exact remediation of the original bug, applied to the
  curriculum path too. Good.
- **Run-level CLI flags propagate to every stage.** `--no-satiation` / `--no-overeating-death`
  are written to `body.with_satiation` / `body.overeating_death` on *all* stage configs
  (lines 357–362) — the correct key path that `load_env_params` actually reads
  (`config_loader.py` lines 1409/1411). See the note in §4.5 on the single-config path using a
  different, dead key.
- **Stage transition is complete.** On crossing an episode boundary the loop reloads params
  from the (extends-resolved) stage config, rebuilds the env, resets env state, wipes episode
  and behavior-measure accumulators, clears the DreamerV3 replay + positive buffers to prevent
  cross-stage world-model contamination, and resets the agent recurrent state (RecurrentPPO
  `h_state`; Dreamer RSSM/mod state). Transition granularity is per-iteration with documented,
  accepted drift.
- **Guards.** Boundaries must be strictly increasing and positive; `checkpoint_frequencies`
  positive; lengths must match the number of stage files; continual mode is restricted to
  RecurrentPPO/DreamerV3; `--episodes` is rejected as incompatible. A stage transition also
  runs a **modality-fingerprint + obs/action-dim equality check** across all stages before
  training (lines ~509–545), so a stage that would silently change the input layout is caught
  at startup rather than corrupting the shared model.

### 4.2 Budget / loop logic — PASS (well-guarded)

The termination expression is
`while (total_episodes_completed < episodes) if episodes > 0 else (global_step < total_timesteps)`
(line 1192).

- `episodes` is resolved from `schedule.episode_boundaries[-1]` (continual), else
  `args.episodes`, else **`config.get_mandatory('episodes')`** (line 467). Because it is
  `get_mandatory`, a missing key raises `ValueError` — there is **no silent `episodes=100`
  fallback in `train.py`**. The "default-100 trap" from session memory belongs to the *separate*
  `dreamer_srl` single-config trainer (which reads its budget from `env_cfg.training.*`), not to
  this entry point. In `train.py` the episodes-based branch is always taken (episodes is always
  a positive int), and the timesteps branch is effectively unreachable — correct and safe.
- `total_timesteps = args.total_timesteps or (episodes * env_max_steps * num_envs)` is used only
  for the profiler and display; the real stopping condition is the episode counter. No issue.

### 4.3 Evaluation — true-observation recording (evaluation_core.py working-tree change) — PASS

The uncommitted change to `src/utils/evaluation_core.py` (around line 180) widens the condition
under which noise-free "true" observations are recorded, from *stats-pass only* to *also the
video pass when the env has perceptual noise enabled*:

```
record_true_obs = config.get_mandatory('testing.record_true_observations') and (
    record_stats or (render_video and params.perceptual_noise_enabled)
)
```

This is a **correct, additive fix**: without it, the video's sensory panel falls back to
`true_obs = obs`, making injected perceptual noise invisible in recordings. The mandatory-key
discipline is preserved (`get_mandatory`). No downside — it only records *more* diagnostic data
when noise is active. Recommend committing it as part of the noise-diagnostics work with a
one-line rationale.

### 4.4 Train↔eval interface — enumerated disagreement points

The audit enumerated every place training and evaluation could silently disagree about the
model or the environment. The **primary path is consistent**; the flagged items are latent and
pre-existing.

**Consistent by construction (good):**

- **Saved config is fully resolved.** `train.py` saves `models/config.yaml` at line 585 —
  *after* all merges including the extends-resolving `load_env_config` (line 381) and the agent
  config (line 390). Eval reads that fully-resolved config, so **the `extends:` bug does not
  propagate into evaluation**. This is the correct design and the reason the eval side is not
  independently vulnerable to the original bug.
- **Live eval path reconstructs the model correctly.** `scripts/eval/eval_rollout.py` (the
  current eval pipeline after the Phase-2 migration, commit `10d9acd`) builds the model with
  `modulation_config = agent_config.get("agent.modulation")` (line 540) — the **full** modulation
  dict, identical to how `train.py` builds it (line 745). No whitelist, no mismatch. In-training
  periodic eval uses the *live* model object directly, so it also cannot disagree.

**Latent gap L1 / Finding A — stale modulation whitelist in the root `evaluation.py`
(pre-existing, low). STATUS: FIXED, commit `2ad9104`.**
The root `evaluation.py` (lines 252–259) reconstructs the model from a **hard-coded 6-key
whitelist**: `type, mod_hidden_size, grouping_size, percept_bias_init, memory_bias_init,
temp_clip`. But `ActorCriticRNN.__init__` reads **`memory_clip` unconditionally via bracket
access** for any modulation-enabled model (`recurrent_ppo_network.py` line 265:
`memory_clip = tuple(modulation_config['memory_clip'])`), plus `percept_add_bias_init` (safe
`.get`, line 262). v3.0 FiLM configs *set* `memory_clip` (e.g. `[-2.0, 2.0]`). Consequence:
reconstructing **any modulation-enabled RecurrentPPO checkpoint through the root `evaluation.py`
raises `KeyError`**. This is **not a v3.0 regression** — `main`'s `evaluation.py` had the same
6-key whitelist and `main`'s network already read `memory_clip` via bracket — but it is now
squarely load-bearing. Mitigating factor: the root `evaluation.py` is **only wired to
`generate_demo.sh`** (a demo-video helper) and is superseded by `scripts/eval/eval_rollout.py`,
which does it correctly.

**Fix (root `evaluation.py`, both halves of Finding A).** The RecurrentPPO branch now builds
`modulation_config` the same way `train.py:745-747` does — `config.get('agent.modulation')`,
null-normalised when `type` is unset — instead of the 6-key whitelist, so `memory_clip` (and any
future modulation key `ActorCriticRNN` reads) can no longer drift out of sync. Separately, the
DreamerV3 branch (previously a plain 5-key `dict` that crashed `DreamerTrainer`'s
`config.get_mandatory(...)` calls with `AttributeError`, and silently omitted `obs_breakdown`/
`modulation_config`) now passes the full `Config` object plus `obs_breakdown` and
`modulation_config`, mirroring `train.py:815-817`. Regression test:
`tests/scripts/test_evaluation_model_rebuild.py` — `test_modulated_rppo_eval_rebuild_succeeds`
builds a real FiLM-modulated (`memory_clip`-bearing) checkpoint and round-trips it through
`evaluation.py`'s actual `main()` (fails with `KeyError('memory_clip')` pre-fix, passes post-fix);
`test_dreamer_eval_rebuild_signature` exercises `main()`'s real DreamerV3 construction call-site
via a spy (fails an `isinstance(..., Config)` assertion pre-fix, passes post-fix — no on-disk
DreamerV3 checkpoint fixture exists to round-trip the full network end-to-end, so the
network-internal build itself is verified at the call-site/signature level, per the fix plan's own
allowance); `test_plain_rppo_eval_rebuild_still_succeeds` confirms the common, non-modulated case
is unaffected. Implemented by: developer.

**Latent gap L2 / Finding E — continual-run eval uses the first-stage environment (pre-existing, medium
for continual only). STATUS: FIXED, commit `a3ab4cc`.** For continual runs, `models/config.yaml` is
**stage 0** (`config` is set to `schedule.stage_configs[0]` at line 365 before the save). Per-stage
configs *are* dumped for auditability (`stage_XX_<name>.yaml`, lines 593–605), but neither
`evaluation.py` nor `eval_rollout.py` reads them — both reconstruct the env from `models/config.yaml`.
So a checkpoint trained in a *later* stage is evaluated against the **stage-0 environment**, which can
differ (predators, scene, noise). This does not affect single-config runs (the overwhelming majority).

**Fix (live path only, `scripts/eval/eval_rollout.py`).** Added `_resolve_continual_stage_config()`:
when `--config` is the run's own stage-0 `config.yaml` and a `schedule.yaml` is present next to the
checkpoint (the continual-run signature), it reads the checkpoint's **own saved `stage` field**
(`ckpt_data['stage']`, train.py ~line 2428) and loads the matching `stage_XX_<name>.yaml` instead. An
explicitly different `--config` (e.g. an out-of-distribution eval config) is left untouched, and
non-continual runs (no `schedule.yaml`) are unaffected — single-config eval behavior is unchanged.
The checkpoint→stage mapping was recoverable (not a blocker): the checkpoint's own `stage` field is
ground truth, **not** a recompute from `episode_boundaries` — confirmed empirically on a real
checkpoint from `results/JAX_RecurrentPPO/20260507-163055_continual_5x5_NoPred-to-PredInt3_rppo_s0/`,
where episode 811 (past boundary 800) was still recorded as stage 0 due to per-iteration
transition-check timing; a naive boundary-recompute would have silently picked the wrong stage.
Verified end-to-end against real continual-run checkpoints on disk (`results/JAX_RecurrentPPO/
20260623-220259_rppo_basic_curriculum_longL4/`): per-checkpoint stage resolution matches the
checkpoint's actual stage across all 5 curriculum stages, and a full `eval_rollout.py` run succeeds
with stage-appropriate behavior metrics for both an early and a late checkpoint. Regression test:
`tests/scripts/test_eval_rollout_stage_config.py` (5 cases, all passing; fail with `AttributeError`
pre-fix). Root `evaluation.py`'s `--all` multi-checkpoint loop was judged **not straightforward** to
fix the same way (it loads one config for a whole batch of checkpoints spanning possibly several
stages, needing a larger per-checkpoint refactor) and was left untouched — it is deprecated and only
wired to `generate_demo.sh`. Implemented by: developer.

**Latent gap L3 — best-effort checkpoint restore keeps init weights silently (pre-existing,
low/robustness). STATUS: FIXED, commit `2ad9104`.** `_merge_restored_into_module_state`
(`evaluation.py` lines 77–94) copies restored leaves into the module structure but, when a module
key is **absent from the restored checkpoint**, **keeps the freshly-initialised value** (line 91)
with no error and no completeness assertion. If train's saved structure ever diverged from eval's
reconstruction (e.g. a modulation-config mismatch, or a param added in v3.0), some layers would
silently run on **random init weights** and evaluation would report degraded-but-not-crashing
numbers. There is no "all restored leaves consumed / all module leaves filled" check.

**Fix.** `_merge_restored_into_module_state` now threads an accumulator (`_missing`) that records
the dotted path of every param leaf/subtree left at its randomly-initialised value (i.e. absent
from the restored checkpoint), and a new `_assert_full_restore(missing, label)` raises a `ValueError`
listing every such leaf if the list is non-empty — called right after the merge and before
`nnx.update(...)`, for both the RecurrentPPO and DreamerV3 (`wm`/`actor`/`critic`) restore paths.
This mirrors the strict completeness check the live path already has
(`scripts/eval/eval_rollout.py:713-737`). Regression tests (`tests/scripts/
test_evaluation_model_rebuild.py`): `test_merge_reports_missing_leaves` confirms the accumulator
correctly records absent leaves without altering the merge's existing behavior;
`test_assert_full_restore_raises_on_incomplete_checkpoint` constructs a deliberate structural
mismatch (a whole missing param subtree) and confirms it now raises `ValueError` naming the missing
leaf, instead of silently proceeding; `test_assert_full_restore_passes_on_complete_checkpoint`
confirms a fully-covering checkpoint is unaffected. Implemented by: developer.

**Latent gap L4 — architecture-affecting CLI overrides not written back to saved config
(pre-existing, low).** `--hidden_size` (and `--num_steps`, `--lr`) are resolved into local
variables (lines 481–494) and used to build the model, but are **not written back into
`config`** before it is saved. `--seed`, `--no-satiation`, wandb/tag flags *are* written back
(lines 393–401). Consequence: a run launched with `--hidden_size` different from the YAML value
saves the *YAML* value, and eval rebuilds the model with the wrong hidden size → shape-mismatch
on restore. Bounded because `--hidden_size` is rarely used (sizes normally live in the agent
YAML). **Recommendation:** follow-up plan to mirror the resolved architecture scalars back into
`config` before the config-save at line 585.

### 4.5 Cosmetic — dead config key on the single-config satiation override (not a bug)

`train.py` lines 400–401 set `environment.with_satiation` / `environment.overeating_death`, but
`load_env_params` reads `body.with_satiation` / `body.overeating_death`. On the single-config
path the flag is nonetheless enforced by `params = params.replace(with_satiation=False)`
(lines 503–504, commented "redundant now but safe"), so `--no-satiation` **does work** for
training. The `config.set('environment.*')` calls are dead/cosmetic. One downstream nuance worth
noting for the interface: because the *saved* `config.yaml` keeps `body.with_satiation=True`
(only the dead `environment.*` key is flipped), a run trained with `--no-satiation` would be
**re-evaluated with satiation on** (eval's `load_env_params` reads the unchanged `body.*` key).
This is pre-existing and only bites if the CLI flag is used instead of a YAML setting — most
runs encode satiation in YAML. Note it in the follow-up plan alongside L4 (both are "resolved
override not reflected in saved config" issues).

---

## Part C — Uncommitted parity-fixture churn (Surface-4 investigation)

**Question for the audit:** the working tree has 5 **modified** and 3 **untracked**
`tests/env/fixtures/parity/*.npz` golden snapshots. Were they regenerated to match a *deliberate*
environment change, or could a regenerated baseline be **masking an unintended env regression**?

**What these fixtures are.** `tests/env/test_unified_parity.py` runs 100 fixed steps from seed 0
for each config and asserts the live env matches a saved `.npz` "golden" snapshot. Regenerating a
snapshot moves the goalpost — so a regen that silently follows a behavior change is exactly the
"masked regression" risk.

**Finding — this is a schema rename, not an environment regression.** Diffing the committed
`.npz` against the working-tree `.npz` (array-by-array) for
`...hypervigilance__01-interoNocicept`:

- **Removed keys:** `stepNNN_pred_pos`, `stepNNN_neutral_pos`, `pred_property_sampled`,
  `neutral_property_sampled`, `pred_move_timer`, ... (the old **per-class** `pred_*`/`neutral_*`
  naming).
- **Added keys:** `stepNNN_animal_pos`, `animal_property_sampled`, `animal_move_timer`,
  `animal_state`, ... (the **unified** `animal_*` naming from the CP1 animal-entity refactor).
- **Of the 2012 keys present in both, ZERO differ** (agent position, satiation, nutrition,
  injury, terminated, and every info-dict field — hits, damage, distances — are **byte-identical**).

So the environment's actual behavior for these 5 configs is **unchanged**; only the *storage key
names* for animal state changed, because the fixture **generator** (`scripts/fixtures/
generate_parity_fixtures.py`) was updated to emit the unified `animal_*` schema (it no longer
emits `pred_pos`/`neutral_pos`). The 3 **untracked** fixtures are the `dreamer_srl_curriculum`
configs, which were previously non-loadable (skipped) and became loadable after the config
`extends:` chokepoint + unified refactor — so these are **new coverage**, benign.

**But there is a real follow-up flag — committing these as-is would silently weaken the test.**
The committed `test_unified_parity.py` still reads the **old** keys via `fixture.get("step000_pred_pos")`
(lines 146, 156, 185, 195) and *skips the assertion when the key is absent*
(`if old_val is not None:`). The new fixtures **do not contain** `pred_pos`/`neutral_pos`, so if
they are committed **without updating the test to read `animal_pos`**, the N1 (reset placement)
and B1 (per-step predator/neutral position) parity assertions **silently become no-ops** — the
test stays green while checking strictly less (only agent position, body scalars, and info-dict
fields, all of which happen to be byte-identical anyway). That is precisely the "regenerated
baseline masks a regression" failure mode — not for *these* configs (behavior is byte-identical
today), but structurally: the animal-position parity gate would be disabled for all future
changes.

**Read / recommendation (a note, needs follow-up — not a blocker for this audit):**
1. The churn is **safe today** — no env regression is hidden; the shared data is byte-identical.
2. **Do not commit the regenerated fixtures alone.** Either (a) commit them together with a
   `test_unified_parity.py` update that reads the unified `animal_*` keys (restoring the position
   assertions), or (b) leave fixtures on the old schema until the test is migrated. This should
   be a small `senior-developer` → `developer` follow-up plan (topic `issues` or `refactors`).
3. Whoever ran the generator should confirm it was a deliberate schema migration and not an
   accidental local regen; the working-tree state at audit start suggests an uncommitted generator
   run that has not yet been paired with a test update.

**Status: FIXED, commit `0bebe06`.** `test_unified_parity.py` now reads the unified `animal_*`
keys (sliced by `predator_indices`/`neutral_indices`) with a fallback to the legacy `pred_*`/
`neutral_*` keys for fixtures not yet regenerated, so the N1/N2/B1 predator/neutral position
assertions hard-execute against both fixture generations instead of silently skipping. The
regenerated fixtures were committed together with the test fix. See Implementation Report in
this repo's commit `0bebe06` for verification detail (pass/skip counts, perturbation sanity
check).

---

## Ground-truth end-to-end results

**Plain-language entry point.** We stopped reading source and ran the real thing. A short live
`train → checkpoint → evaluate` cycle was executed locally in the `grid_world_pain` conda env on
the "sensory-noise" level (`basic/06`, which inherits four config layers and turns on perceptual
noise + the behaviour metrics), plus two direct environment drives to probe the one finding that
could affect runs training **right now**. Three things came out of it:

1. **The "surviving to the time-limit is punished exactly like dying" bug is REAL** and I have the
   actual reward numbers to prove it. An agent that survives to the step cap gets **−100** on its
   final step — the same "death penalty" a starved agent gets — even though it did the right thing.
   And the trainer's value-learning throws away the future-value estimate on those time-limit
   endings. **This affects the six runs training right now** (all are survival tasks on this reward
   scheme). It is *not* a new v3.0 bug — it predates the branch — so every past run shares it too.
2. **The evaluation path the six live runs will actually use works correctly and completely.** I
   trained a real checkpoint and restored it through the live evaluator (`scripts/eval/eval_rollout.py`);
   the restore is byte-complete (proven by a two-different-seeds test) and the reloaded config is the
   fully-inherited one written by training itself. The crashes the JAX reviewer found are in a
   *different, deprecated* evaluator only wired to the demo-video script.
3. **The config-inheritance fix holds end-to-end** — the config the training process wrote to disk
   contains every inherited layer (random start, harder predator, perceptual noise), confirmed from
   the file the system produced, not a reload.

### E2E-1 — Finding B (PRIORITY): truncation is treated as death — CONFIRMED with real numbers

> **Fix plan:** [[FIX_TRUNCATION_TREATED_AS_DEATH]] (`docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md`) — Part 1 (death penalty on real death only, in `core.py`) + Part 2 (GAE truncation bootstrap). **STATUS: both parts implemented and verified.** Part 1 landed in commit `ef0fd25`; Part 2 (the `recurrent_ppo_trainer.py` GAE truncation-bootstrap fix — `terminated`-only mask for the value bootstrap, `V(true next state)` stored pre-auto-reset, MC/GAE static gating so the live MC-mode runs pay zero cost) landed in commit `3c60f6f`, verified by `senior-developer` 2026-07-04. The identical `ppo_trainer.py` sibling (plain, non-recurrent PPO — no live config uses this file) has since also been fixed, commit `926c2c3` (2026-07-04) — same `terminated`-only bootstrap mask, plus a `Transition.next_value` pre-auto-reset fix, since this file also had the auto-reset-before-value-read issue for every step, not just the recurrent trainer's final-step case. **Still open (deferred, comment-only follow-up, NOT fixed):** the same `done`-conflates-timeout pattern remains in `src/models/dreamer_v3_trainer.py`'s continue-head loss (`cont_target = 1.0 - terminal[..., None]`) — carries a durable follow-up comment linking this doc, out of scope for the current fix.
>
> **Reward-regime break point:** Part 1 landed in commit `ef0fd25` (2026-07-04). Runs trained on code **before** this commit reward surviving-to-`max_steps` with the full `death_penalty` (the Finding-B bug, below); runs trained **after** this commit do not — the terminal reward on a timeout is the ordinary homeostatic step value instead. Absolute reward curves are not comparable across this boundary (survival-step counts, the project's headline metric, are unaffected).

Method: drove the real `jax_step`/`jax_reset` on `basic/06`'s resolved `EnvParams` to (a) an episode
that ends by **max-step timeout** (truncation) and (b) one that ends by **starvation** (death), and
printed the ground-truth reward decomposition on the final step. Driver:
`tmp/20260704_finding_b_driver.py`. Resolved reward params (from the loader):
`use_homeostatic_reward=True`, `death_penalty=100`, `metabolic_cost=1.0`, `max_steps=500`.

| Episode | `termination_reason` | `terminated` (body death)? | truncated? | final `reward` | `reward_homeostatic` | death-penalty applied? |
|---|---|---|---|---|---|---|
| **Timeout / survive** | `1` (max_steps) | **False** | True | **−100.186** | −100.186 | **YES — −100** (homeostatic w/o penalty would be −0.186) |
| **Starvation / death** | `2` (starvation) | True | False | −100.660 | −100.660 | YES — −100 |

**Verdict: CONFIRMED.** Surviving to `max_steps` incurs the full `death_penalty` (−100), identical to
actually dying. The spurious −100 **dwarfs the ~0.2-magnitude per-step homeostatic reward by ~500×**,
so it dominates the terminal signal. Root cause is exactly as the math review traced:
`core.py:706` sets `done = done OR truncated`, and `core.py:722` subtracts `death_penalty` on `done`
(the homeostatic branch these runs use), so `done` being true on a timeout carries the penalty.

**Bootstrap side (value-target corruption), also confirmed.** The trainer's collected `done` array
element on a truncation step is the *same* `done` — the driver's third return value from `jax_step`
was `True` on the timeout step (that is the exact value `collect_trajectories` stores,
`recurrent_ppo_trainer.py:177,218`). `compute_gae` (`:63`) computes `delta = r + γ·V(s')·(1−done) − V`,
so on truncation the `γ·V(s')` bootstrap is **zeroed** — the value head is trained to regress the
truncated single-step reward (here `r − 100`) instead of the discounted continuation. `termination_reason`
distinguishes code 1 (timeout) from 2/3/4 (death) and is carried in the trajectory (`:212`) but is
**not** consulted by the bootstrap — the fix signal exists but is unused.

**Does it corrupt the survival reward signal, and are the 6 live runs affected?** Yes to both, with a
nuance the user should own:
- **Corruption:** the reward the agent optimises and the value targets it regresses are both distorted
  at every timeout, and in a survival task most episodes end by timeout, so the distortion is systematic,
  not rare. The non-terminal homeostatic reward telescopes to a small bounded quantity per episode, so
  the −100 terminal is the dominant term.
- **Live runs:** the six currently-training runs (basic/05, 06, 07 + jump-reach + variant-02, 03 — all
  non-modulated RecurrentPPO, homeostatic reward, `death_penalty=100`) are **all affected**.
- **Not a v3.0 regression:** `core.py:722` and the trainer's `done` handling are **byte-identical to
  `main`** (per the math review), so this is pre-existing and **every historical run shares it** —
  cross-run comparability is preserved. It is a correctness problem for the *absolute* survival objective,
  not a v3.0-introduced divergence. Whether to restart the six runs vs. let them finish and fix before the
  next wave is a **user call** (see recommendation).

### E2E-2 — LIVE checkpoint round-trip (the path the 6 runs use) — PASS

Real cycle: `train.py --config basic/06 --agent_config recurrent_ppo.yaml --episodes 8` produced
`results/JAX_RecurrentPPO/20260704-003724_default/` with a checkpoint at `models/819` and a saved
`models/config.yaml`. Restored through the **live** evaluator
(`scripts/eval/eval_rollout.py --checkpoint models/819`): restore succeeded with no error, 5 eval
episodes ran, behaviour metrics were produced.

**Restore fidelity — ground-truth two-seed test** (`tmp/20260704_restore_fidelity.py`): built the model
from two *different* init seeds, restored the same checkpoint into both.
- All **27/27** float param leaves **differ from random init** ⇒ restore actually loaded the trained weights.
- All **27/27** leaves are **byte-identical across the two different-init restores** ⇒ restore is
  **complete** — no leaf silently left at init.

**Key reconciliation fact:** the live `eval_rollout.py` restore has a **strict architecture check**
(`:610–629`) that raises `ValueError` on any missing or shape-mismatched leaf — i.e. it does **not**
have the silent-partial-restore gap (L3) that the deprecated root `evaluation.py` has. The live path the
six runs will use is guarded. Non-modulated RecurrentPPO round-trips correctly and completely — proven,
not asserted.

### E2E-3 — Extends-resolution end-to-end — PASS

The `models/config.yaml` the training process wrote (ground truth, not a reload) is fully flattened with
every inherited layer present: `random_start_pos: true`, `random_start_injury: true` (from basic/05),
predator `move_interval: [1,1]`, `max_stamina: [30,150]` (basic/05's all-predator-pressure update),
`perceptual_noise.enabled: true` with `olfaction.injury_noise_scale: 4.0` (basic/06's lever). No residual
`extends:` key. Confirms the config-loader fix holds through a real train run into the saved config eval reads.

### E2E-4 — Finding A reproduction (deprecated root `evaluation.py`) — REPRODUCED, then FIXED

Built a FiLM-modulated model exactly as root `evaluation.py:252–259` does (its hard-coded 6-key
modulation whitelist) against `recurrent_ppo_nmn_film_g1_screen.yaml` (which sets `memory_clip: [-2.0, 2.0]`).
Result: **`KeyError('memory_clip')`** at model construction — confirmed. Any modulation-enabled
RecurrentPPO checkpoint fails to build through the root `evaluation.py`. This path is only wired to
`generate_demo.sh`; the live `eval_rollout.py` uses the full modulation dict and is unaffected.

**Post-fix (commit `2ad9104`):** re-ran the same repro end-to-end through `evaluation.py`'s actual
`main()` — a real FiLM-modulated (`memory_clip: [-2.0, 2.0]`) checkpoint now builds and restores
without error. See Latent gap L1 / Finding A above for the fix detail and regression tests.

---

## Reconciliation + verdict

### Plain-language verdict (read this first)

**The pipeline is safe to keep running, with one real correctness bug worth fixing and a short list of
narrower follow-ups.** Across four independent code reviews and a live end-to-end run, we found **no new
bug introduced by the big v3.0 change wave that silently corrupts a current run** — the config-inheritance
fix holds, the environment's math and randomness are faithful, and the evaluator the live runs will use
restores checkpoints correctly and completely (proven by actually doing it). The one bug that matters is
**old, not new**: the environment punishes an agent for *surviving to the time limit* exactly as if it had
died (−100), and the learner throws away its future-value estimate on those endings. This distorts the
survival signal the six live runs are optimising — but because it predates the project's `main` branch,
every past run is distorted the same way, so run-to-run comparisons stay valid. Everything else is either a
crash on a **deprecated demo-only evaluator** (not the live path), a **metric-counting bias** that affects
analysis numbers but not training, or a **test-hygiene item** (don't commit the regenerated golden-snapshot
fixtures until the parity test is migrated, or a position check silently switches off). None of these block
training. Recommended order below.

### Severity reconciliation — the `evaluation.py` crashes (code-reviewer 🔴 vs Surface-4 "low")

Both are right about different things; reconciled severity is **Medium (confirmed crash, deprecated/narrow
path)**:
- **The crash is real** — I reproduced `KeyError('memory_clip')` for a modulated config (E2E-4); the
  DreamerV3 dict-vs-Config crash is signature-confirmed. The code-reviewer's 🔴 is correct *as a code defect*.
- **The blast radius is narrow** — the crashes are in the root `evaluation.py`, which is only wired to
  `generate_demo.sh` (demo videos) and is **superseded** by `scripts/eval/eval_rollout.py`. The live path is
  proven correct (E2E-2), builds the model with the full modulation dict, and has a strict-restore guard.
  Surface-4's "low real-world impact" is correct *for the live pipeline*.
- **Reconciled:** neither a launch blocker nor a live-run risk, but a real confirmed crash that blocks
  demo-video eval of every modulated / DreamerV3 checkpoint. Fix is a one-liner (reuse train's full-dict
  expression) or retire the root file.

### Final findings table (ranked by REAL impact)

| # | Finding | Ground-truth result | v3.0-new? | Real severity | Fix-flow |
|---|---------|---------------------|-----------|---------------|----------|
| **B** | Truncation treated as death: `death_penalty` (−100) applied on timeout (`core.py:706,722`) **+** GAE bootstrap zeroed on truncation (`recurrent_ppo_trainer.py:63`) | **CONFIRMED** — survivor reward −100.186 (−100 is the penalty); trainer `done`=True on timeout ⇒ bootstrap dropped. **Affects all 6 live runs.** | **Pre-existing** (byte-identical to `main`) | **HIGH** (corrupts the optimised survival objective + value targets; systematic in a survival task) — but shared by all historical runs, so comparability preserved | **FIXED — Part 1 `ef0fd25` (reward gate on real death, `core.py`), Part 2 `3c60f6f` (GAE truncation bootstrap, `recurrent_ppo_trainer.py`); both verified by `senior-developer` 2026-07-04.** `terminated`-only mask (true death, `reason∈{2,3,4}`) gates the value bootstrap; `V(true next state)` stored pre-auto-reset; MC/GAE branch is JIT-static so live MC-mode runs pay 0.00% cost. **`ppo_trainer.py` sibling FIXED, commit `926c2c3`** (2026-07-04) — same mask + a `next_value` pre-auto-reset fix (this file had the auto-reset-before-value-read issue on every step, not just the last); no live config uses plain PPO, so zero production effect. **Open follow-up (deferred, comment-only):** same pattern still in `dreamer_v3_trainer.py` continue-head. |
| **A** | Root `evaluation.py` modulation whitelist omits `memory_clip` (`:252–259`) → `KeyError`; DreamerV3 passes plain dict where `Config` expected (`:293–300`) → `AttributeError` | **FIXED, commit `2ad9104`** — both branches rebuilt to mirror train.py exactly; regression tests reproduce the pre-fix crash then confirm post-fix success | Pre-existing | **Medium** (confirmed crash, but deprecated demo-only path; live path proven correct) | **DONE.** RecurrentPPO branch now uses `config.get('agent.modulation')` (null-normalised), mirroring `train.py:745` / `eval_rollout.py:540`; DreamerV3 branch now passes `Config`+`obs_breakdown`+`modulation_config` to `DreamerTrainer`, mirroring `train.py:815-817`. |
| **M1/M2** | Interrupted-feeding & bush-dive rates: denominator-timing mismatch + pending-at-episode-end events dropped (`accumulators.py`), preferentially dropping death-by-predator interruptions | Consistent with e2e eval showing `interrupted_feeding_rate: NaN` on a 5-episode run | **v3.0-new** (behaviour measures are new code; bias is in the new logic, not a regression of old behaviour) | **Medium** (distorts *analysis* numbers, not training) | **senior-developer metric-fix plan → developer.** Resolve pending events at episode boundary or exclude from denominator; align per-class and per-tag denominators to the same instant. Cross-link `experiment-analyzer`. |
| **L2 / E** | Continual-run eval reconstructs env from stage-0 `config.yaml`; later-stage checkpoints eval'd against stage-0 environment | **FIXED, commit `a3ab4cc`** — verified against real continual checkpoints on disk across all 5 curriculum stages + regression test | Pre-existing | **Medium for continual only** (single-config runs unaffected) | **DONE (live `eval_rollout.py` path).** Auto-detects the checkpoint's own stage from its saved `stage` field and loads the matching `stage_XX_<name>.yaml`. Root `evaluation.py`'s `--all` batch loop not fixed (not straightforward — deprecated, demo-only). |
| **L3** | Silent partial restore in root `evaluation.py` (`_merge_restored_into_module_state`) keeps init weights for absent leaves | **FIXED, commit `2ad9104`** — live path was already IMMUNE (`eval_rollout.py:610–629` raises on missing/mismatched leaf; two-seed test proved complete restore); root `evaluation.py` now has an equivalent strict-completeness assertion | Pre-existing | **Low** (root `evaluation.py` only) | **DONE.** Folded into the **A** fix — `_assert_full_restore` raises `ValueError` listing every param leaf left at its randomly-initialised value, for both RecurrentPPO and DreamerV3 restore paths. |
| **C** | Uncommitted parity-fixture churn: `pred_*/neutral_*` → unified `animal_*` schema rename in golden snapshots | Benign — shared data byte-identical; only storage-key names changed | **v3.0-new** (CP1 animal-entity refactor) | **Low but MUST-FIX-BEFORE-COMMIT** (committing fixtures alone silently no-ops the animal-position parity asserts) | **FIXED, commit `0bebe06`** — `test_unified_parity.py` migrated to read the unified `animal_*` keys (sliced by `predator_indices`/`neutral_indices`) with a legacy-key fallback for still-unregenerated fixtures; regenerated fixtures committed together with the test fix. Verified: 34 passed/244 skipped before and after (unchanged — skip count is fixture-presence only); 1545 previously-silent array assertions across the 8 regenerated configs now execute for real; perturbation sanity check confirmed the migrated assertion fails on injected divergence. |
| minor | basic/00 M2-NaN (no bushes); observability-gate `random_start_pos`; `temp_clip` dead lower bound; value-loss comment mismatch | per surface docs | mixed | **Low / nits** | Batch into a housekeeping plan or address opportunistically. |

### Top recommendation — what to fix first

**Fix B first.** It is the only finding that touches the actual thing being optimised (the survival reward
and its value targets), it is confirmed with real numbers, and it affects both the six live runs and all
historical runs. The fix is well-scoped (separate a `terminated`-only mask from the timeout-inclusive `done`
for the bootstrap, and gate `death_penalty` on true death) and must ship with a regression test that asserts,
on a truncation step, (i) no `death_penalty` in the terminal reward and (ii) the GAE bootstrap retains
`γ·V(s')`. **Decision for the user:** the six live runs are already distorted but comparably so — restart-now
vs. finish-and-fix-before-next-wave is a portfolio call (candidate for a `pi` consult before the next launch).

**Then, before anything is committed:** the parity-test migration (**C**) — otherwise the animal-position
parity gate silently switches off.

**Then bundle A + L3** (one deprecated-evaluator cleanup), and schedule **M1/M2** and **L2** as metric/eval
correctness follow-ups.

All fixes fork to a **senior-developer fix plan → developer** flow after user approval. No `src/` code was
changed by this diagnosis.

Verified by: senior-developer
