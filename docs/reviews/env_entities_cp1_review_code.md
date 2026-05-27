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
