---
title: "Code review — Track B eval/logging fixes (uncommitted working tree, 2026-07-23)"
topic: reviews
status: active
created: 2026-07-23
last_updated: 2026-07-23
---

# Code review — Track B eval/logging fixes (2026-07-23)

## Verdict in plain language

This reviews the uncommitted working-tree changes that fix five evaluation/logging
bugs from the 2026-07-23 diagnosis (plan: [[FIX_EVAL_LOGGING_TRACK_B_20260723]]).
In plain terms the five fixes are: (1) stop different training runs' eval outputs
from colliding on disk when you evaluate an rPPO step folder; (2) stop the
multi-environment evaluator from undercounting the first episodes' survival by one
step; (3) actually send the WandB "job type" tag (and fix its misspelled default);
(4) give stochastic evaluation a fresh random key every step so it is really
stochastic; (5) make the "evaluate with noise-free observations" knob fail loudly
instead of silently lying.

**All five fixes are correct.** The PRNG threading preserves reset-seed
reproducibility and leaves the deterministic (default) path byte-identical; the
step-0 seeding now matches the refill path exactly with no double-seed and the
off-by-one gone; the run-tag helper resolves every real invocation form; the
job-type wiring and typo fix are correct; the noise guard fails loud as designed.
The five regression tests are sound.

**One thing the reviewer must not miss:** the `scripts/eval/eval_rollout.py`
working-tree diff is **not just the five surgical edits the plan describes**. It
also contains a large, *unplanned* new feature — a `--config-list` multi-condition
"dwell-sweep" evaluation path — that refactors `main()` into several new helpers
and restructures both the rPPO and Dreamer branches into per-condition loops. This
feature is real work and appears self-consistent, but it is **outside the Track B
plan**, has **no test in this change set**, and if committed under a "Track B"
message would mislabel ~450 lines of feature work as a bug-fix. This is the primary
finding; the bug-fixes themselves are clean. See Finding S1.

---

## Scope reviewed

`git diff HEAD` on: `scripts/eval/eval_rollout.py`, `src/utils/evaluation_core.py`,
`configs/logger/wandb.yaml`, `train.py`, plus the five new tests
(`tests/scripts/test_eval_rollout_run_tag.py`,
`tests/scripts/test_parallel_eval_step0_seeding.py`,
`tests/training/test_wandb_job_type_wiring.py`,
`tests/scripts/test_eval_stochastic_key_distinct.py`,
`tests/scripts/test_eval_obs_noise_guard.py`).

Explicitly **excluded** (parallel track, per task): `src/environment/config_loader.py`,
`configs/environment/`. Not read, not flagged.

---

## Per-target verdict

| # | Target | Verdict |
|---|--------|:-------:|
| 1 | Per-step PRNG split in `_run_episode` / `_run_episode_with_recording` | OK |
| 2 | Step-0 seeding un-gating in `_run_parallel_env_eval` | OK |
| 3 | `_derive_run_tag` edge cases | OK (one nit) |
| 4 | Deterministic default path unchanged | OK (with S1 caveat) |
| 5 | Test quality | OK |

### Target 1 — per-step PRNG split (OK)

`eval_rollout.py:79` calls `state = jax_reset(params, rng_key)` **before** the step
loop; `rng_key` is never reassigned. The new `step_key = rng_key` (line 96 / 243)
is a pure alias, and `jax.random.split(step_key)` (line 99 / 245) is side-effect
free, so the reset key is untouched — reset-world reproducibility is preserved. Each
step consumes a freshly-split `act_key` and folds `step_key` forward; the folded key
is never reused. In deterministic mode `policy_fn` passes `key=None` to
`get_action_and_value_nnx`, so `act_key` is discarded and output is byte-identical to
pre-fix. Both loop bodies (`_run_episode`, `_run_episode_with_recording`) got the
identical, correct treatment.

### Target 2 — step-0 seeding un-gating (OK)

Removing the `if record_stats:` wrapper (`evaluation_core.py:565→`) makes the initial
per-slot seeding unconditional. I diffed it field-by-field against the refill block
(lines 695–705):

- Fields match: `agent_pos, satiation, nutrition, injury_level, rest_streak, res_pos,
  res_active, animal_pos, obs_pos` — same set, same single-env slice semantics
  (`states.<f>[i]` at init vs `new_state.<f>` at refill; both are single-env arrays).
- Sentinels match: `slot_actions ← -1`, `slot_rewards ← 0.0`, `slot_infos ← {}`,
  `slot_obs ← obs[i]` / `new_obs`, `slot_true_obs` gated on `record_true_obs` in both.
- **No double-seeding:** init runs exactly once per slot before the loop; refill first
  clears the slot buffers to `[]` (lines 664–670) then rebuilds a fresh one-element
  buffer — the two never both seed the same generation.
- **Off-by-one resolved:** `episode_lengths.append(len(slot_rewards[i]) - 1)` (line 636)
  now subtracts a step-0 sentinel that is present for *every* generation, not just
  refilled ones. `record_stats=True` behavior is unchanged (that path already ran this
  block once). Dtypes are consistent Python `int`/`float` between init and loop appends.

Signature check: the Bug 2 test's 24 kwargs align exactly with
`_run_parallel_env_eval`'s definition (`evaluation_core.py:538–542`), so the test
actually exercises the function.

### Target 3 — `_derive_run_tag` edge cases (OK, one nit)

`_CKPT_CONTAINER_DIRNAMES = {"checkpoints", "models"}`; grandparent if the parent is a
container name, else parent. Confirmed against every real form:

- rPPO step-dir `<run>/models/<step>` → `<run>` (the fix; was `"models"`).
- Dreamer step-dir `<run>/checkpoints/<step>` → `<run>` (unchanged).
- CheckpointManager-root `<run>/models` → `<run>` (unchanged — parent is `<run>`,
  not a container name).
- Bare `<run>/some_other_dir` → `<run>` (fallback).

Trailing-slash and relative paths are neutralized upstream by
`Path(args.checkpoint).resolve()` before the helper is called, so `.name` is always
the normalized final component. Windows-style backslash paths are not a concern on the
Linux target (POSIX path semantics).

🟢 **Nit (not blocking):** a run directory *literally named* `models` or `checkpoints`
evaluated in the **root form** (`.../models/models`) would resolve to the grandparent
rather than the run dir — a behavior *regression* vs. the old one-liner for that one
pathological case. The step-dir form of the same pathological run is handled correctly.
No real run is named `models`/`checkpoints`, so this is a documentation-worthy nit, not
a fix request.

### Target 4 — deterministic default path unchanged (OK, see S1)

For a single `--config` invocation the refactor is behavior-preserving: `raw_entries`
is a one-element list, `_resolve_entry` reproduces the old top-level sequence
(stage-config resolution → `load_env_config` → `load_behavior_measure_cfg` → n_eps/seeds
extension → `--batched` guard → `load_env_params` → `max_steps` → out_dir/rec_dir),
the model is built from `entries[0]["params"]` (== the single loaded params), the
policy closure closes over that same params object (so `jax_reset` and
`get_observation` see identical params, as before), the PRNGKey-per-episode
(`jax.random.PRNGKey(seeds[ep_idx])`) is unchanged, and `metadata.json` fields map
1:1 (`config→config_arg`, `config_resolved→config_path`, both equal to the old values
for one entry). The new `_assert_eval_obs_noise_supported` call only raises for
non-`training` modes, which no default config sets. I found no behavioral divergence on
the single-config deterministic path. **Caveat:** this confidence is a code-reading
argument over a ~450-line refactor, not an execution diff — see S1 for why that
refactor should not be in this change set at all.

### Target 5 — test quality (OK)

- **Bug 1 test**: pure unit test of `_derive_run_tag`, covers all four real forms. Good.
- **Bug 2 test**: integration test with a death-disabled tiny env so every episode
  truncates at `max_steps`; asserts all four lengths equal `max_steps`. Correctly
  targets the undercount. Good.
- **Bug 3 test**: uses the plan's pre-authorized lighter textual assertion (config typo
  gone + `wandb_kwargs` literal wires `job_type` from `wandb.job_type`). Acceptable, and
  matches the fix. It does not execute `wandb.init`, so it cannot catch a future
  refactor that renames the literal — acceptable given the constraint.
- **Bug 4 test**: runs `_run_episode` with `deterministic=False` and a stub policy that
  records keys; asserts pairwise distinctness. Correctly exercises the split. Note it
  verifies *distinctness* but not the *deterministic-mode byte-identity* claim — that is
  covered only indirectly by the pre-existing deterministic eval tests. Fine.
- **Bug 5 test**: asserts `NotImplementedError` for `zero`/`custom`, no raise for
  `training`. Direct and correct.

---

## Findings

| Sev | Location | Issue | Suggested action |
|-----|----------|-------|------------------|
| 🟡 concern | `scripts/eval/eval_rollout.py` (whole diff) | **S1 — unplanned feature bundled with the bug-fixes.** The diff adds a `--config-list` multi-condition "dwell-sweep" eval path (`_parse_config_list`, `--config-list` arg, `_resolve_entry`, `_run_rppo_entry`, `_make_policy_fn`, per-condition loops in both the rPPO and Dreamer branches) that the Track B plan does not mention — the plan promises "five independent, surgical edits, no shared helper." The plan's own Implementation Report describes only the five edits and says "Deviations: None," so it is inconsistent with the actual working tree. Two risks: committing this under a Track B message mislabels a large feature as a bug-fix; and the feature ships with **no test** in this set (there is no `--config-list` test). | Split into two commits: (a) the five surgical Track B edits + their five tests; (b) the `--config-list` feature + a dedicated multi-condition test. If the feature belongs to the separate dwell-sweep perf track (commit `d241b53` lineage), coordinate so it lands under that track's plan, not Track B's. |
| 🟢 nit | `scripts/eval/eval_rollout.py` `_derive_run_tag` | A run directory literally named `models`/`checkpoints` in the CheckpointManager-**root** form resolves to the grandparent instead of the run dir (regression vs. old one-liner for that pathological case only; step-dir form is correct). | Leave as-is; optionally note the assumption ("no run dir is named `models`/`checkpoints`") in the helper docstring. |
| 🟢 nit | `tests/scripts/test_eval_stochastic_key_distinct.py` | Verifies per-step key distinctness but not the deterministic-mode byte-identity guarantee (covered only indirectly by existing tests). | Optional: add a deterministic-mode assertion that a fixed seed reproduces identical actions with/without the split. Not required. |

## Conventions audit

- Pytree / immutability: ✅ (no in-place state mutation introduced; `jax.tree_util.tree_map`
  used correctly for per-slot slicing).
- JIT recompilation triggers: ✅ (no traced/static field reclassification; the extra
  `jax.random.split` is eval-only Python-level control flow, not inside a jitted step).
- vmap / batch conventions: ✅ (`ParallelEnv` axis-0 semantics untouched; per-slot
  indexing `x[i]` on batched state preserved; `EnvParams` still broadcast, not batched).
- PRNG threading: ✅ (per-step split correct; reset key preserved; refill uses
  `key, reset_key = jax.random.split(key)` advancing the master key — unchanged).
- Sensor / observation-breakdown sync: ✅ (no sensor added/renamed; the `--config-list`
  shape-check actually *strengthens* obs-breakdown consistency across conditions).
- Config protocol: ✅ (`config.get_mandatory('wandb.job_type')` used for the new key,
  matching sibling `project`/`entity`/`group`; no `.get(key, default)` for a required
  param).

## Conclusion

All five Track B bug-fixes are correct and their tests are sound; the deterministic
default path is behavior-preserving. The blocking-worthy issue is process, not
correctness: the `eval_rollout.py` working tree entangles an unplanned `--config-list`
feature (untested here) with the five surgical fixes — recommend splitting the commit
before landing. No 🔴 blockers.

Reviewed by: code-reviewer
