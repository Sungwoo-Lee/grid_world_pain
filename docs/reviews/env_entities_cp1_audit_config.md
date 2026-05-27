---
title: "Config Audit — env_entities CP1 (post-implementation)"
topic: env_entities
status: active
created: 2026-05-28
last_updated: 2026-05-28
aliases: [env_entities_cp1_config_audit]
---

# Config Audit — CP1 Unified Animal Entity Refactor (post-implementation)

**Branch**: `v2.0` (commits `c3892cb` impl + `78faf37` tests + `b854eb9` docs)
**Plan v0.3**: `docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md`
**Prior audit**: `docs/reviews/env_entities_plan_audit_config.md` (pre-impl)
**Auditor**: `env-config-auditor`
**Run date**: 2026-05-28

## Verdict: ACCEPT-WITH-MINOR-REVISIONS

The CP1 implementation is sound and CP2 may proceed. All blocking invariants pass. One concern requires documentation (the 55 stale configs are intentionally unverified by parity tests, with justification in commit `3d20aab` but not in the plan doc). One nit on mandatory-key discipline for the `perceptual_noise.enabled` key. No blocking issues.

## Findings

| Severity | Location | Issue | Suggested fix |
|---|---|---|---|
| Non-blocking | `tests/env/fixtures/parity/` vs 86 migrated configs | 31 of 86 covered. The 55 not covered pre-date the mandatory `sensory.injury_observable` key and could not load pre-refactor. Plan doc doesn't acknowledge this 31/86 split. | Add note to plan: "31 of 86 migrated configs have parity fixtures; the remaining 55 pre-date `sensory.injury_observable` and could not load under the pre-refactor code either." |
| Non-blocking | `src/environment/config_loader.py:930` — `perceptual_noise_enabled=config.get('perceptual_noise.enabled', False)` | Pre-existing soft default, asymmetric with other boolean flags loaded via `get_mandatory`. Every loadable config carries this key explicitly. | Change to `config.get_mandatory('perceptual_noise.enabled')` after verifying all loadable configs carry it. |
| Nit | `src/environment/config_loader.py:770–791` — `print("="*60)` placement diagnostics | Unconditional stdout noise on every `load_env_params` call, including during test runs. | Gate behind `logging.debug` or a verbosity flag. |

## Checklist

### Observation / Noise Modality Consistency — PASS
- `get_observation_breakdown` emits 10 canonical modalities; padding stays at 3 (total 13).
- `_YAML_KEY_TO_SENSOR_NAME` maps all 10 names.
- No 14th sensor introduced. New `animal_*` arrays feed into existing channels (visual 5/7, olfaction aggregation).
- `noise_modality_order` remains `pytree_node=False`. `[13]` padding annotations accurate.

### Mandatory-Key Discipline — PASS-WITH-MINOR-CONCERN
- `environment.predators` / `environment.neutral_animals` use `config.get(...)` — **intentional and correct** (optional list sections).
- All animal distributional fields for `behaviour: hunt` go through `_parse_distributional(..., mandatory=True, ...)` raising `ValueError` if absent. Satisfies "no fallback defaults" for critical fields.
- `perceptual_noise.enabled` soft default flagged above.
- Legacy dead keys `body.start_satiation` / `body.random_start_satiation` still loaded via `get_mandatory` (unchanged, schema-required but runtime-ignored).

### Static-Field and JIT Recompile Risk — PASS
All 12 new CP1 fields verified:

| Field | Type | pytree_node=False? |
|---|---|---|
| `animal_classes`, `animal_behaviours`, `animal_tags` (string tuples) | tuple | ✓ |
| `hunt_idx`, `wander_idx`, `static_idx` (int tuples) | tuple | ✓ |
| `predator_indices`, `neutral_indices` (int tuples) | tuple | ✓ |
| `animal_classes_int`, `animal_behaviours_int` | jnp.ndarray | (traced — correct) |
| `animal_is_damaging`, `animal_visual_channel` | jnp.ndarray | (traced — correct) |

### Known Latent-Bug Recurrences — PASS (no triggers found)
- `body.overeating_death`: zero configs set true.
- Legacy `property:` (singular): zero configs use it. `_read_properties` handles + warns.
- Per-entity olfactory missing: raises ValueError. No config triggers.
- `terminated` vs `done`: unaffected.

### Schema Padding and Modality-Count — PASS
- Default config: 10 modalities, padding 3, total 13. Unchanged.
- No new sensor channel. CP1 unified channels 5 (predator) + 7 (neutral) under `animal_visual_channel` — refactor, not addition.

## Migration Sweep Verification (86-Config Atomic Sweep)

- `git show c3892cb --name-only | grep "\.yaml"` returns exactly 86 files. **Verified.**
- `predator_enabled` removal: zero configs in `configs/` carry the key. The one occurrence in `src/environment/config_loader.py` is the migration guard that raises `ValueError` if detected at load time.
- `configs/verification/olfaction_parity_neutral.yaml` correctly hand-migrated (line 10 stripped; `predators: []` already on line 11).
- `lose_interest_multiplier`: all 85 configs with non-empty predator entries carry it explicitly. Soft-default-to-mandatory promotion is safe.

**Parity fixture coverage:**

| Category | Configs migrated | Fixtures | Pass |
|---|---|---|---|
| `configs/experiment/hypervigilance/` + sub-dirs | 10 | 10 | 10 |
| `configs/experiment/nmn_noise_heterogeneity/` | 5 | 5 | 5 |
| `configs/experiment/nmn_meta_2x3_mixture/` | 4 | 4 | 4 |
| `configs/experiment/behavior_measures/` | 1 | 1 | 1 |
| `configs/continual/nmn_double_return_stages/` | 5 | 5 | 5 |
| `configs/verification/` | 6 | 6 | 6 |
| `configs/environment/default.yaml` | 1 | 1 | 1 |
| `configs/experiment/labmeeting/` (stale) | 55 | 0 | 60 skipped |
| **Total** | **86** | **31** | **31 / 60 skipped** |

The 60 skipped tests are intentional and documented in `3d20aab`'s commit message (stale configs pre-dating `sensory.injury_observable`).

## Schema Invariants Checklist

| Invariant | Status |
|---|:---:|
| `predator_enabled` absent from all YAML | ✅ |
| `predator_enabled` guard raises ValueError | ✅ |
| Visual channel predator=5, neutral=7 preserved | ✅ |
| `noise_modality_order` unchanged | ✅ |
| Noise array padding = 13 | ✅ |
| New static tuple fields `pytree_node=False` | ✅ |
| Int-coded animal arrays are traced JAX | ✅ |
| Behaviour string validation raises ValueError | ✅ |
| Neutral legacy auto-fill (damage / attack_delay) | ✅ |
| Hunt distributional fields mandatory | ✅ |
| Wander/static distributional auto-fill `[0, 0]` | ✅ |
| Legacy `predator_tags`/`neutral_tags` as @property | ✅ |
| 31 fixture-backed parity tests pass | ✅ |

## Conclusion

**ACCEPT-WITH-MINOR-REVISIONS.** CP2 green-lit from config-soundness side. Three non-blocking items for the developer to address at CP2 or in a cleanup commit (not before launch): parity gap doc, `perceptual_noise.enabled` mandatory promotion, `print` placement diagnostics.

Audited by: env-config-auditor
