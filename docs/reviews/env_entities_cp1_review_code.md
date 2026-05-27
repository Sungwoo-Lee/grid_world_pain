---
title: "Code Review — env_entities CP1 (post-implementation)"
topic: env_entities
status: active
created: 2026-05-28
last_updated: 2026-05-28
aliases: [env_entities_cp1_code_review]
---

# Code Review — CP1 Unified Animal Entity Refactor (post-implementation)

**Branch**: `v2.0` (commits `c3892cb` impl + `78faf37` tests + `b854eb9` docs)
**Plan v0.3**: `docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md`
**Prior review**: `docs/reviews/env_entities_plan_review_code.md` (pre-impl)
**Reviewer**: `code-reviewer`
**Run date**: 2026-05-28

## Verdict: ACCEPT-WITH-MINOR-REVISIONS

CP1 PRNG/JAX correctness is verified end-to-end for 31 of 86 configs. The N1/N2/N3 work is clean: 6-way `placement_key` split, 4-way `prop_key` split, `[res, pred, obs, neutral]` resolve-scan order all preserved. The `@property` legacy aliases for `predator_tags` / `neutral_tags` work (verified end-to-end through `dreamer_srl_main.py:522-523`).

One test-hygiene blocker (9 pre-existing tests break under CP1) must be patched before CP2 is fully clean.

## Blocking issue

**B-CP1-1 — Test suite has 9 failures + 4 errors after CP1.** Two gaps:

- **G1 (CP1 sweep miss)** — 2 test files have embedded `predator_enabled: false` YAML literals (`tests/environment/test_per_tag_distance_logging.py:66`, `tests/environment/test_behavior_measures.py:68`). Trivial: strip both lines.
- **G2 (CP6 deferral leaks into CP1 test surface)** — Plan defers renderer + `eval_recording` to CP6, but existing tests exercise those paths. The hot training loop does NOT use eval-recording or the renderer, so training is unaffected, but the test red bar hides future regressions. Required action: either pull CP6 cleanup forward (~50 lines across 6 files) or `xfail` the affected tests with CP6 cross-reference. Recommend `xfail` — minimum-diff, plan-consistent.

Affected test files for G2:
- `tests/algorithms/dreamer_srl/test_eval_recording.py` (4 tests)
- `tests/algorithms/dreamer_srl/test_eval_rollout.py::test_eval_rollout_recordings_exist`
- `tests/algorithms/dreamer_srl/test_render_upload.py::test_render_and_upload_produces_mp4`
- `tests/algorithms/dreamer_srl/test_eval_video_smoke.py::test_e2e_smoke_checkpoints_and_recordings`

Stale references (CP6 cleanup):
- `src/utils/eval_recording.py:34-35`
- `src/utils/evaluation_core.py:387-388, 465-466, 662`
- `src/environment/renderer.py:461, 482`
- `src/environment/renderer_v2.py:374, 380`
- `src/environment/grid_world.py:445, 466`
- `scripts/benchmark_render.py:52-53`

## Non-blocking concerns

- **NC-1**: Parity-test coverage is 31 / 86 configs, not 86 / 86 as the plan promised. The 55 skipped configs pre-date the mandatory `sensory.injury_observable` key (added by an unrelated earlier PR) so they were failing pre-refactor too. **Not a CP1 regression** — but the plan's "86 configs" claim is misleading. Recommend updating to "31 loadable / 55 stale" and opening a separate triage ticket for the stale configs.
- **NC-2**: Plan deviation in parity-preserving direction. Plan said `jax.random.split(key, 5)` → `split(key, 6)` (extend by 1). Developer kept the 5-way split and derived `animal_episode_key = jax.random.fold_in(property_key, 0xAE1)` (`core.py:781-783`). Verified: `fold_in(property_key, 0xAE1)` does not collide with the four `split(property_key, 4)` sub-keys. **The dev's choice is more correct than the plan** — the plan-as-written would have shifted `agent_key`/`placement_key`/`body_key`/`property_key`, breaking all 31 fixtures. Plan should be retroactively updated.
- **NC-3**: Damage sampling shape changed from `(N_pred,)` to `(N_animals,)`. Threefry shape-stability says element 0 of either produces the same value, so byte-parity holds at the predator's index. Tests confirm. Plan should call this out explicitly.
- **NC-4**: `chem_dim = 5` hardcoded in N=0 branch of `_load_animals` (`config_loader.py:392, 396`). Latent bug if future zero-animal config uses obstacles with non-5-dim olfactory vectors. Defer to CP3.
- **NC-5**: Redundant ternary at `core.py:955` — both branches return the same value. Cosmetic.
- **NC-6**: `dist_per_animal` info-dict key has zero-shape when N=0. Downstream analysis scripts in CP6 must handle empty arrays.
- **NC-7**: `at_neutral_pre` Python-level shape-stability branch is config-static under JIT. Safe.

## Conventions audit checklist

| Convention | Status | Notes |
|---|:---:|---|
| Pytree immutability (`._replace`) | ✅ | All state updates use `state._replace(...)` in `jax_step`. |
| JIT shape-stability across configs with same animal counts but different bounds | ✅ | `animal_*_low/high` are pytree-node JAX arrays. |
| vmap safety | ✅ | `update_animals` operates on full batch; per-subset slicing is host-side via `pytree_node=False` index tuples. |
| PRNG threading | ✅ | 6-way step split preserved; 5-way reset outer split + `fold_in` (NC-2); 6-way `placement_key` inner split preserved; 4-way `prop_key` split preserved. |
| Sensor / observation breakdown sync | ✅ | `get_observation_breakdown` unchanged; noise modality map unchanged. |
| Configuration Protocol (`get_mandatory`) | ✅ | All mandatory fields preserved; new `predator_enabled` guard raises with clear migration message; behaviour-string validation present. |

## Parity-test coverage assessment

| Layer | Coverage | Detail |
|---|---|---|
| Fixture-backed byte-parity | 31 of 86 | All hypervigilance, continual nmn_double_return, nmn_meta_2x3_mixture, nmn_noise_heterogeneity, observability_gates, olfaction_parity, behavior_measures smoke, environment/default, death_penalty_ablation. |
| Backward-compat load | 31 of 91 | Same set as fixtures; 60 configs raise the pre-existing `sensory.injury_observable` ValueError and are skipped. |
| Configs not covered | 55 of 86 | The entire `configs/experiment/labmeeting/` family, `dreamer_curriculum/`, `dreamer_diagnostic/`, `2X2_area.yaml`. All stale pre-CP1. |

The plan's 86-config coverage claim is **not literally satisfied**, but the regression risk it was guarding against is intact: any config that loads is parity-tested.

## Conclusion

CP1 is JAX/Flax-correct. The three bugs the developer caught during implementation (PRNG outer-split, `obs_blocking` vs `obs_hides_agent`, zero-entity scan guard) are all valid and correctly resolved. CP2 is technically ready to start, but **before declaring CP1 complete, the user should resolve the 9-failing-existing-tests issue** — recommend `xfail` with CP6 cross-reference (~10 lines of test markers).

Reviewed by: code-reviewer

---

# Re-review of CP5 (2026-05-28)

**Question**: Did CP5 — the checkpoint that lights up *non-degenerate* per-episode sampling, adds the WandB per-episode log helpers, and asserts the JIT-no-recompile guarantee — land as a JAX/Flax-correct piece of code?

**Context**: CP1-CP4 already shipped the plumbing: a unified `entities:` YAML schema, per-episode draws at `jax_reset`, and parity-preserving step paths. But until CP5 the per-episode draws were always degenerate (every legacy config carried scalars or auto-filled `[0, 0]` bounds). CP5 is the first checkpoint where the random draws actually vary across episodes, and the first that explicitly tests the headline plan claim — "same animal counts and same class ordering, different distributional bounds, does NOT trigger a JIT recompile".

**Headline finding**: CP5 is JAX/Flax-correct. The JIT-no-recompile guarantee is verified end-to-end (exactly 1 compile across configs A+B with different bounds; exactly 2 compiles when class ordering swaps — both numbers confirmed by re-running the tests). The `build_episode_log_dict` helper returns Python floats, handles `N=0` cleanly, varies across episodes for non-degenerate ranges, and stays constant for degenerate ones. The `print → logging.debug` placement-diagnostic gate is real (zero `print` statements remain in `config_loader.py`).

**Verdict**: **ACCEPT-WITH-MINOR-REVISIONS**. One concern (silent acceptance of `[high, low]` bound inversion) and three nits. CP6 may proceed in parallel with the fix — the issues are non-blocking for the next checkpoint's plumbing, but `C-CP5-1` should be addressed before any training run uses a non-degenerate range.

## CP5 scope verified

| Surface | What was checked | Result |
|---|---|---|
| `tests/env/test_no_recompile.py` | Both tests pass; ran in 31.45 s | 2 / 2 pass |
| Negative control compile count | Re-ran with manual log capture (config X + Y, same classes) | exactly 1 compile after X+Y → claim holds |
| Positive control compile count | Confirmed via `assert count_after_D == 2` and re-derived independently | exactly 2 compiles → claim holds |
| `tests/env/test_distributional_yaml.py` | 11 / 11 pass; ran in 10.99 s | OK |
| `tests/env/test_per_episode_logging.py` | 9 / 9 pass | OK |
| Full `tests/env/` suite | 121 passed, 122 skipped, 0 failed in 292 s — matches developer's claim | OK |
| `print` removal in `config_loader.py` | `grep "print(" src/environment/config_loader.py` → no matches | OK |
| `02-entities-distributional.yaml` smoke load | Loads cleanly, N=3, hunt_idx=(0,), wander_idx=(1, 2); 5 seeds give 5 distinct sampled values for the 4 ranged fields and 1.500 for the scalar one | OK |

## Findings

| Severity | File:Line | Issue | Suggested fix |
|---|---|---|---|
| 🟡 **C-CP5-1** | `src/environment/config_loader.py:263-272` | `_parse_distributional` silently accepts `[high, low]` bound inversion. `detection_range: [5, 3]` loads, stores `low=5.0, high=3.0`, and `jax.random.uniform(minval=5, maxval=3)` returns `5.0` deterministically every episode. The user thinks they have randomness over `[3, 5]`; they get a constant. Verified by direct probe (see below). | Add a check after the unpack: `if hi < lo: raise ValueError(f"Animal entity {entity_label!r} (index {idx}): '{field}' bounds [{lo}, {hi}] are inverted (high < low). Use [low, high] order.")`. One line. |
| 🟢 **N-CP5-1** | `src/environment/config_loader.py:774, 235` | Two function-local `import logging` statements (one at line 235 inside `_load_animals`, one at line 774 inside `load_env_params` as `_logging_local`). Module has no top-level `import logging`. Style nit — works correctly, but hoisting to module top + dropping the alias would be simpler. | Move `import logging` to the file's import block and replace `_logging_local.getLogger` with `logging.getLogger`. |
| 🟢 **N-CP5-2** | `src/behavior/accumulators.py:521` | Local `import numpy as _np` inside `build_episode_log_dict`, but the module already imports `numpy as np` at line 21. Redundant. | Drop the local import and use `np.asarray` directly. |
| 🟢 **N-CP5-3** | `tests/env/test_no_recompile.py` (no test) | Plan-claim edge case not covered: configs with the same `animal_classes` tuple but different `animal_behaviours` (e.g., `[predator hunt, neutral wander]` vs `[predator hunt, neutral static]`). I verified manually that this DOES recompile (1 → 2) because `wander_idx`/`static_idx` are `pytree_node=False`. JIT behaviour is correct; coverage gap is minor. | Optional: add a Part 3 test mirroring Part 2's class-ordering swap. |

### C-CP5-1 reproduction

```yaml
detection_range: [5, 3]
```

```
low=5.0, high=3.0
sampled detect (seeds 0-4): 5.0, 5.0, 5.0, 5.0, 5.0
```

`jax.random.uniform(minval=5.0, maxval=3.0)` returns `minval` deterministically — verified across 20 keys, all return `5.0`. This is a silent foot-gun: the user expected a uniform draw over `[3, 5]` and gets a constant. One-line fix in the loader; no JIT or pytree implications.

## Item-by-item against the assignment

**1. JIT no-recompile guarantee** — VERIFIED.
- Log capture: `_CompileCounter` attaches a `StreamHandler` to `jax._src.interpreters.pxla` at WARNING and counts `"Compiling jit(jax_step)"` substrings. Mechanism is sound; I confirmed the same string appears in `jax_log_compiles=True` output by independent probe.
- Re-ran `tests/env/test_no_recompile.py`: both pass, 31.45 s. Manual probe (config X = predator+wander, config Y = predator+static, same N=2 and same classes) → 1 compile after X, 2 after Y. Behaviours-tuple change DOES retrigger as expected because `wander_idx`/`static_idx` are `pytree_node=False`. No silent leakage of traced values.
- The 1-compile-for-A+B claim is exact. The positive control (`test_class_ordering_swap_triggers_recompile`) asserts `count_after_D == 2`, which both the developer's run and mine confirm.
- Edge case: same `animal_classes` tuple, different `animal_behaviours` (e.g., neutral wander → neutral static) — DOES recompile. Verified manually (N-CP5-3 above).

**2. `build_episode_log_dict()` correctness** — VERIFIED.
- Reads all five `state.animal_*_sampled` `[N]` arrays once via `np.asarray()`, then iterates `params.animal_tags`. Per-tag fan-out is loop-based, not vectorised.
- N=0 case: `animal_tags == ()` → returns `{}`. Confirmed via probe with `entities: []` config.
- Contract: returns `dict[str, float]` (host-side Python floats via `float(arr[i])`). Safer than 0-d jax arrays for WandB. `test_logged_values_are_python_floats` asserts this. Good.
- No caller exists yet in `src/` or `scripts/` — CP5 ships the library + tests; CP6 will wire it into `dreamer_srl_main.py`. Plan explicitly defers integration. Not a blocker.
- Pytree immutability: function reads state, never mutates. vmap-safety n/a (call-site is outside the JIT step path).

**3. `_parse_distributional()` malformed-input handling** — 3 of 4 cases handled; 1 silent foot-gun.
- `[5]` (one-element list) → raises `ValueError`. PASS.
- `"five"` (non-numeric) → raises `ValueError` (via `float("five")`). PASS.
- `[1, 2, 3]` (three-element list) → raises `ValueError("must be a scalar or a 2-element list")`. PASS. (Not in the test suite, but the `len(val) != 2` guard catches it — verified by probe.)
- `[5, 3]` (high < low) → **silently accepted**, becomes a degenerate constant via `jax.random.uniform`. See C-CP5-1.

**4. Smoke config soundness** — `02-entities-distributional.yaml` is well-formed.
- Uses the new unified `entities:` schema; no legacy `predators:`/`neutral_animals:` keys.
- One non-degenerate field per range type: `detection_range: [0, 5]`, `max_stamina: [20, 40]`, `stamina_recovery_rate: [0.5, 1.5]`, `hunt_stamina_threshold: [0.5, 0.9]`, and one degenerate scalar (`lose_interest_multiplier: 1.5`).
- Loads without error; N=3 (1 predator + 2 wander rabbits), `hunt_idx=(0,)`, `wander_idx=(1, 2)`.
- Bounds load correctly: `animal_detect_low[0]=0.0`, `animal_detect_high[0]=5.0`, etc.

**5. Per-episode sampling actually varies across episodes** — VERIFIED.
- Probed 5 seeds (0, 1, 2, 17, 42): `detect ∈ {1.58, 0.83, 1.11, 3.42, 3.09}`, `stamina ∈ {37.5, 27.9, 35.1, 37.7, 36.4}`, etc. All within bounds; all distinct.
- `hunt_thresh_sampled` and `animal_detect_sampled` differ on different seeds → agent's hunt-onset distance differs episode-to-episode. The previously-untested non-degenerate path works.

**6. Carry-forward from CP1 review** — `print` diagnostic is genuinely replaced.
- `grep "print(" src/environment/config_loader.py` returns zero hits. The `_loader_log.debug(...)` calls are the actual mechanism (not a comment). Output suppressed at default WARNING level; reappears under `PYTHONLOG=DEBUG` or programmatic `logging.basicConfig(level=DEBUG)`. Acceptable.

**7. Conventions audit checklist**

| Convention | Status | Notes |
|---|:---:|---|
| Pytree immutability | ✅ | `build_episode_log_dict` reads state, never mutates. CP5 source files have zero in-place `state.foo = ...` patterns. |
| JIT shape-stability across same N + same class ordering + different bounds | ✅ | `tests/env/test_no_recompile.py::TestNegativeControl` proves 1 compile across configs A+B. Independently re-verified. |
| JIT shape-stability across same N + same class ordering + same behaviours | n/a — implied | The negative-control test covers this implicitly: behaviour tuples are equal between A and B since only numeric bounds differ. |
| JIT recompile on class-ordering swap (positive control) | ✅ | `TestPositiveControl` asserts `count_after_D == 2`. Independently re-verified. |
| JIT recompile on behaviours-tuple swap (gap) | 🟢 n/a — verified manually | Not in the test suite (N-CP5-3). Manual probe (wander → static, same classes) → 2 compiles. Coverage gap is cosmetic. |
| vmap safety | ✅ — n/a | `build_episode_log_dict` is host-side, called outside the JIT step path; no vmap interaction. |
| PRNG threading | ✅ — unchanged in CP5 | CP5 does not touch `core.py` PRNG logic; CP2's 6-way split + `fold_in(property_key, 0xAE1)` remains the reset path. |
| Sensor / observation breakdown sync | ✅ — unchanged in CP5 | CP5 adds no sensor or noise channels. |
| Configuration Protocol (`get_mandatory`) | ✅ — for new keys | The 5 distributional fields use `_parse_distributional` with `mandatory=True` for `hunt` and `mandatory=False` (auto-fill `[0, 0]`) for `wander`/`static`. Matches plan §"Mandatory-by-behaviour". |
| Static-vs-traced field boundary | ✅ | `animal_*_low/high/sampled` are traced `[N]` arrays. `animal_classes`/`animal_behaviours`/`hunt_idx`/`wander_idx`/`static_idx`/`animal_tags` are `pytree_node=False`. Test `TestPositiveControl` confirms the static side actually does trigger recompiles. |

## Acceptance gate

**Verdict: ACCEPT-WITH-MINOR-REVISIONS**. CP6 may proceed in parallel; C-CP5-1 (the `[5, 3]` silent-foot-gun) is a one-line validation fix and should land before any training run uses a configured non-degenerate range — but the fix lives entirely in the loader and does not touch CP6's surfaces. N-CP5-1, N-CP5-2, N-CP5-3 are non-blocking nits.

CP6's plumbing (wiring `build_episode_log_dict` into `dreamer_srl_main.py` and `train.py`, plus the legacy `dist_per_predator` / `dist_per_neutral` cleanup) is unblocked. The env-config-auditor's verdict (`NC-1` from `env_entities_cp1_audit_config.md`: `perceptual_noise.enabled` should be promoted to `get_mandatory`) was not picked up by CP5 and that is fine — it sits in a different surface and was always slated for a later checkpoint.

Re-reviewed by: code-reviewer
