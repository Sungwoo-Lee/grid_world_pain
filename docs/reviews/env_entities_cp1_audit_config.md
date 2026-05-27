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

---

## Re-audit of CP2–CP4 (2026-05-28)

**Scope:** Multi-checkpoint — CP2 (per-episode sampling), CP3 (entities: schema in config loader), CP4 (sensor + damage + step path parity). Covers mandatory post-CP3 audit plus concurrent CP2 and CP4 work. Triggered by plan v0.3 §"Reviews needed".

**Branch:** `v2.0`

**Commits audited:** `4781105` (CP2), `a85a951` (CP3), `6685d0e` (CP4)

**Files audited:**
- `src/environment/config_loader.py` (lines 199–603, 743–791, 930, 953–992)
- `src/environment/core.py` (lines 770–1009, per-episode sampling block)
- `src/environment/sensor.py` (lines 59–88, `sense_extero_nociception`)
- `configs/experiment/v2_smoke/01-entities-smoke.yaml` (CP3 smoke config)
- `tests/env/test_per_episode_sampling.py` (CP2 test suite)
- `tests/env/test_entities_schema.py` (CP3 test suite)
- `tests/env/test_visual_parity.py` + `test_extero_noc_parity.py` (CP4 test suite)
- `tests/env/fixtures/visual_parity_ref.npz` + `tests/env/fixtures/extero_noc_parity_ref.npz` (CP4 fixtures)

**Audited by:** env-config-auditor

**Run date:** 2026-05-28

---

### Purpose

This is the mandatory post-CP3 re-audit required by plan v0.3 §"Reviews needed". It also covers CP2 (the per-episode sampling implementation, which landed alongside CP3) and CP4 (sensor and damage path parity). The question is: are the three delivered checkpoints sound enough to green-light CP5 (distributional schema and per-episode logging)?

---

### Summary

CP2, CP3, and CP4 are collectively sound. Every blocking invariant holds on live-load: the new `environment.entities:` schema loads correctly, all mandatory-field validation fires as specified, backward-compat paths are intact, observation-to-noise modality pairing is unchanged, and the noise array padding remains at 13. Two carry-over non-blocking items from CP1 remain open (the `perceptual_noise.enabled` soft default at `config_loader.py:930`, and the unconditional `print("="*60)` placement diagnostics). One new non-blocking item is identified: the implementation deviated from the plan's pseudocode by using `jax.random.fold_in` rather than a 6-way outer split for `animal_episode_key` — the deviation is technically correct and preserves parity (the CP1 parity fixtures still pass), but the plan's pseudocode is now misleading for future readers. No blockers.

---

### Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| Non-blocking (carry-over) | `src/environment/config_loader.py:930` | `perceptual_noise_enabled=config.get('perceptual_noise.enabled', False)` — soft default instead of `get_mandatory`. Flagged in CP1 audit; not addressed in CP2–CP4. | Promote to `config.get_mandatory('perceptual_noise.enabled')` — all loadable configs carry the key explicitly. |
| Non-blocking (carry-over) | `src/environment/config_loader.py:775–791` | Unconditional `print("="*60)` placement diagnostics on every `load_env_params` call. Flagged in CP1 audit; not addressed in CP2–CP4. | Gate behind `logging.debug` or a `VERBOSE` flag. |
| Non-blocking (new) | `src/environment/core.py:778–783` | Implementation uses `jax.random.fold_in(property_key, 0xAE1)` to derive `animal_episode_key`. Plan §File Changes says "extend the outer 5-way key split to 6-way (`key, agent_key, placement_key, body_key, property_key, animal_episode_key`)." The fold_in approach is correct (preserves all existing sub-key streams byte-for-byte) and is documented in the CP1 implementation report (§Debugging Notes item 1). However, the plan's pseudocode in §Per-episode sampling remains inconsistent with the actual implementation. | Update plan §"Per-episode sampling inside `jax_reset`" pseudocode to show `animal_episode_key = jax.random.fold_in(property_key, 0xAE1)` instead of the 6-way split, with a note explaining why `fold_in` was chosen to avoid disturbing existing sub-key streams. |
| Non-blocking (carry-over) | Plan doc §Test Plan / §CP1 checkpoint | The 31/86 parity-fixture coverage gap is documented in the CP1 Implementation Report (§Test Results: "60 configs without fixtures — skipped (were stale before refactor)") but NOT in the plan doc's Test Plan §(a) or in the Coverage Breakdown table. The CP1 audit flagged this as a required plan-doc addition. Still absent in the plan body (only in the implementation report subsection). | Add a sentence to Plan §Test Plan §(a) or the Coverage Breakdown table: "55 of 86 configs are skipped because they pre-date the mandatory `sensory.injury_observable` key and could not load under the pre-refactor code either; this is not a regression introduced by the refactor." |

---

### Checklist

**1. Observation / Noise Modality Consistency — PASS**

Live-loaded `01-entities-smoke.yaml` and the legacy parity-reference config. Both produce 7 active observation sensors (`Satiation`, `Interoceptive Nociception`, `Extero Nociception`, `Olfaction`, `Collision`, `Proprioception`, `Visual`). The `noise_modality_order` tuple covers all 7 active sensors plus 3 inactive ones (`Injury`, `Nutrition`, `Location`) = 10 entries. No active sensor is missing from the noise modality tuple. `noise_modes` array is padded to shape `(13,)` as required. No 14th sensor introduced by CP2–CP4.

**1.5. Behavior-measures Bush Presence — N/A**

`behavior_measures.enabled` is not set in either the smoke config or the parity-reference config. No `behavior_measures` section audited in this scope.

**2. Mandatory-Key Discipline — PASS**

Live-verified all NC-1 cases:
- `behaviour: hunt` + missing `detection_range` → `ValueError` raised with message "missing mandatory distributional field 'detection_range'". PASS.
- `behaviour: hunt` + missing `damage` → `ValueError` raised with message "missing mandatory field 'damage'". PASS.
- `behaviour: hunt` + missing `attack_delay` → `ValueError` raised with message "missing mandatory field 'attack_delay'". PASS.
- `behaviour: wander` + missing distributional fields → auto-fills `[0.0, 0.0]` for all 5 fields, logs at debug level. PASS (confirmed `animal_detect_low=0.0`, `animal_detect_high=0.0`).
- `behaviour: wander` + missing `damage` in the `entities:` path → `ValueError` raised (damage is mandatory for all entries in the `entities:` schema, consistent with the plan table which marks damage as mandatory for wander in the unified schema).

The `perceptual_noise.enabled` soft default (carry-over from CP1) remains at `config_loader.py:930`. Non-blocking; every loadable config carries the key.

The plan's `lose_interest_multiplier` mandatory-promotion note (from soft default `2.0` in legacy loader to mandatory in v2.0) is documented in the loader docstring (line 230–233). PASS.

**3. Static-Field and JIT Recompile Risk — PASS**

All static fields introduced by CP1–CP4 verified unchanged across the three checkpoints:
- `animal_classes`, `animal_behaviours`, `animal_tags`, `hunt_idx`, `wander_idx`, `static_idx`, `predator_indices`, `neutral_indices`: all `struct.field(pytree_node=False)` on `EnvParams`. Unchanged.
- `noise_modality_order`: `struct.field(pytree_node=False)`. Unchanged.
- Per-episode distributional bounds (`animal_detect_low/high` etc.): traced JAX arrays `[N]` float — correct, bounds changes do not force recompile. PASS (the CP5 JIT-recompile test will explicitly verify the no-recompile boundary).

No new static field introduced in CP2–CP4.

**4. Known Latent-Bug Recurrences — PASS (no triggers found)**

- `body.overeating_death`: false in the smoke config. No change from CP1.
- `random_start_pos`: true in `01-entities-smoke.yaml` (matched parity-reference config). The latent spawning-on-entity bug is pre-existing; not introduced by CP2–CP4.
- Legacy `property:` (singular) key: zero configs use it; `_read_properties` warns on the old singular key. Unchanged.
- `terminated` vs `done`: unaffected by CP2–CP4.
- Resource respawn occupancy gap: unaffected by CP2–CP4.

**5. Schema Padding and Modality-Count — PASS**

`noise_modes.shape == (13,)` confirmed by live-load. `len(noise_modality_order) == 10` with 3 padding slots. No new sensor channel introduced. CP2–CP4 do not touch the modality order tuple or the padding constant.

**6. Cross-Config Coherence (Sweep Audits) — N/A**

No sweep across multiple configs in this scope. Single smoke config plus single legacy parity-reference config.

---

### CP-specific Findings

#### CP2 — Per-episode sampling at reset

All 8 test plan requirements verified by reading `test_per_episode_sampling.py`:
1. Reproducibility (same key → same samples): test 1. PRESENT.
2. Divergence (different key → different samples): test 2. PRESENT.
3. Per-instance independence (N entities → N independent draws): test 3. PRESENT.
4. Sampled fields on `EnvState` not `EnvParams` (code-reviewer MF#1): test 4. PRESENT and explicit `hasattr` / `not hasattr` checks.
5. Cross-field independence (pairwise Pearson |r| < 0.5 over 100 resets): test 5. PRESENT.
6. Degenerate-range guard (scalar=5.0 → sampled=5.0): test 6. PRESENT, uses legacy path.
7. Wander/static entities keep `animal_state==0` throughout (intentional shape uniformity, MF#3/4): test 7. PRESENT.
8. Zero-animal smoke — reset without error, correct empty shapes, `hit_*=False`, `dist_to_*=99.0` (M3 guard, M6): test 8. PRESENT.

Implementation uses `jax.random.fold_in(property_key, 0xAE1)` to derive `animal_episode_key` (deviation from plan pseudocode but documented in implementation report). The `jax.random.uniform` calls on lines 963–972 use `jnp.maximum(high, low)` as the `maxval` argument to guard against degenerate ranges where `low == high`. This is functionally correct (`uniform([5, 5])` returns `5.0`) and avoids JAX's `maxval > minval` assertion at degenerate ranges. PASS.

#### CP3 — Entities schema in config loader

Schema fields: the plan defines 12 non-distributional fields (`class`, `behaviour`, `tag`, `count`, `properties`, `properties_std`, `nociception_intensity`, `move_interval`, `damage`, `spawn_area`, `patrol_area`, `attack_delay`) plus 5 distributional fields = 17 total YAML fields. The smoke config `01-entities-smoke.yaml` provides all 17 for the predator entry and the appropriate subset (12 non-distributional only) for the wander rabbit entries. The audit brief's reference to "14 schema fields" appears to be a counting artifact; the actual plan and implementation use 17 fields. All 17 are correctly handled.

Byte-parity claim: `test_entities_schema.py::test_entities_smoke_byte_parity` asserts 100-step obs equality between the smoke (unified schema) and legacy configs. The smoke config's predator-first ordering, degenerate distributional bounds, and matching olfactory properties confirm byte-parity is achievable by construction.

DeprecationWarning: live-verified that loading a config with both `entities:` and `predators:` sections emits a `DeprecationWarning` with the message "The unified 'entities:' schema takes precedence; legacy sections are ignored." The unified tags win (`wolf_unified` in animal_tags, not `legacy_wolf`). PASS.

Behaviour-string validation: validated at two places — inside the `entities:` path at `config_loader.py:303–307` (before index-tuple construction) and again in the class/behaviour loop at `config_loader.py:511–515`. The redundant second check is harmless.

Per-type placement mode: `grep -r "mode: per_type" configs/` returns zero results. The `per_type` code path in `core.py` and the `type_entity_map` builder are present but never exercised by any current config. CP3 implementation report notes this explicitly. The `type_entity_map` builder at `config_loader.py:743–768` uses the `[res, pred, obs, neutral]` entity ordering as specified by the N1 fix. No concern.

#### CP4 — Visual + exteroceptive parity fixtures

Both fixtures exist and are committed:
- `tests/env/fixtures/visual_parity_ref.npz`: present.
- `tests/env/fixtures/extero_noc_parity_ref.npz`: present.

The B2 fix (`sense_extero_nociception` uses `animal_is_damaging` mask): verified live. `params.animal_is_damaging` is `[True, False, False]` for the parity-reference config (1 predator + 2 neutral rabbits). Only the predator contributes to the extero-noc sum; neutrals are masked out. `test_extero_noc_parity.py::test_only_damaging_animals_contribute` asserts this explicitly. PASS.

Visual channel layout: `params.animal_visual_channel == [5, 7, 7]` (predator=5, neutral=7) confirmed live. `test_visual_parity.py::test_visual_channel_layout` asserts this. PASS.

Trainer grep: plan reports "0 surviving `state.pred_` / `state.neutral_` reads in `src/algorithms/` or `src/models/`". This was verified in the CP4 implementation report (commit `6685d0e`). Renderer and eval-recording paths are intentionally xfailed for CP6.

---

### Open Items from CP1 Audit (Status)

| CP1 Finding | Status at CP2–CP4 |
|---|---|
| 31/86 parity coverage gap not documented in plan body | Still open. The implementation report documents it. The plan's Test Plan §(a) and Coverage Breakdown table do not. Non-blocking — the information is findable. |
| `perceptual_noise.enabled` soft default at `config_loader.py:930` | Still open. Not addressed in CP2–CP4. Non-blocking. |
| `print("="*60)` placement diagnostics | Still open. Not addressed in CP2–CP4. Non-blocking (nit). |

---

### Conclusion

**ACCEPT — CP5 green-lit from config-soundness side.**

All three checkpoints (CP2, CP3, CP4) pass the audit. No blocking issues. Three non-blocking items carry over from CP1 (none new); one new non-blocking item (plan pseudocode now inconsistent with the `fold_in` implementation). These four items should be addressed in CP5 or a dedicated cleanup commit, not before launch. CP5 (distributional schema + per-episode logging) may proceed.

Audited by: env-config-auditor

---

## Re-audit of CP5 (2026-05-28)

**Scope:** Single checkpoint — CP5 (distributional YAML schema + per-episode logging + JIT no-recompile tests). Mandatory post-CP5 audit per plan v0.4 §"Reviews needed". Gating CP6 implementation (jointly with parallel code-reviewer verdict).

**Branch:** `v2.0`

**Commits audited:** `b783bca` (feat: CP5 implementation) + `0d0ded5` (docs: CP5 implementation report + checkpoint mark)

**Files audited:**
- `configs/experiment/v2_smoke/02-entities-distributional.yaml` (new smoke config)
- `tests/env/fixtures/dist_scalar.yaml` / `dist_range.yaml` / `dist_malformed_single.yaml` / `dist_malformed_str.yaml` (new test fixtures)
- `src/behavior/accumulators.py` (new `build_episode_log_dict` + `sampled_wandb_keys`)
- `src/environment/config_loader.py:770-795` (print-to-debug gating)
- `tests/env/test_distributional_yaml.py` / `test_per_episode_logging.py` / `test_no_recompile.py` (new test files)
- `docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md` (plan v0.4, CP5 report)

**Audited by:** env-config-auditor

**Run date:** 2026-05-28

---

### Purpose

CP5 is the first checkpoint to deliver a config that exercises the full distributional path with genuinely non-degenerate bounds. It also delivers the per-episode WandB logging helper and the JIT recompile test suite. This audit checks whether the new config is schema-complete and safe to use as a training config, whether the distributional YAML parsing covers all edge cases, and whether the WandB logging helper is correct. It also verifies the carry-forward non-blocking items from CP1 and the CP2-CP4 re-audit.

---

### Verdict: ACCEPT — CP6 green-lit from config-soundness side

All mandatory checks pass. No blocking issues. One carry-forward non-blocking item is now closed (`print` gating resolved). The `perceptual_noise.enabled` soft default remains open with an expanded scope explanation (model configs prevent the simple mandatory promotion). Two new informational findings are documented below, neither blocking: the `dist_*` fixture files are currently unreferenced by any test code, and inverted distributional bounds (`[high, low]`) are accepted silently but clipped at runtime by the `jnp.maximum` guard in `jax_reset`. Additionally, a new static-field observation: same-N + same-class-order but different `animal_tags` triggers a JIT recompile (confirmed live), which is expected but was not previously documented in the plan.

---

### Summary

CP5 is sound. The distributional smoke config `02-entities-distributional.yaml` loads cleanly under live inspection, stores correct `[low, high]` bounds for all 5 fields (predator index 0) and correct `[0, 0]` auto-fills for the two wander rabbit entries (indices 1 and 2). `build_episode_log_dict` produces 15 keys (5 fields times 3 animals), all Python floats, with predator values within the configured bounds and wander animals correctly zeroed. All 22 CP5 tests pass, and the 5 pre-existing parity tests (`test_visual_parity.py` + `test_extero_noc_parity.py`) remain green. The `print("="*60)` carry-forward item is closed. The `perceptual_noise.enabled` carry-forward item remains open but with a scope clarification: the developer's CP5 report confirms that ~46 configs in `configs/models/` do not carry the key, making a simple mandatory promotion a breaking change; this is now a future-sweep item, not a CP6 blocker.

---

### Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| Non-blocking (carry-over, open) | `src/environment/config_loader.py:930` | `perceptual_noise_enabled=config.get('perceptual_noise.enabled', False)` — soft default still present. Developer's CP5 report clarifies: ~46 model configs under `configs/models/` do not carry the key; promoting to mandatory would break them. | Scope the fix: audit `configs/models/` for the missing key, add it via a sweep commit, then promote to `get_mandatory`. Until the sweep lands, the soft default is the correct behaviour. Not a CP6 blocker. |
| Non-blocking (new) | `tests/env/fixtures/dist_*.yaml` (all four) | The four `dist_*.yaml` fixture files exist and are correctly authored, but no Python test code loads them by path. `grep -rn "dist_scalar\|dist_range\|dist_malformed" tests/` returns zero Python hits. The test cases they document are covered via inline YAML strings in `test_distributional_yaml.py` — the fixtures are documentation artifacts only. | Document in each fixture file's comment that it is a reference example, not a loadable test fixture (the actual test uses inline YAML). Alternatively, add a test that loads each fixture path explicitly. Low priority — the inline tests give equivalent coverage. |
| Non-blocking (new informational) | `src/environment/config_loader.py:263-272` (`_parse_distributional`) and `src/environment/core.py:964-972` | Inverted bounds (`detection_range: [5, 3]` where high < low) are accepted silently by the loader — no `ValueError`. The `jnp.maximum(high, low)` guard in `jax_reset` at lines 964-972 clips the effective range to the degenerate case (`uniform([5, 5]) = 5`), so runtime behaviour is deterministic and not dangerous. However, a user who writes `[5, 3]` by mistake gets silent clipping rather than a helpful error. | Add a `lo <= hi` check in `_parse_distributional` with a clear `ValueError`: `"distributional field '{field}': low ({lo}) must be <= high ({hi})"`. The `jnp.maximum` guard then becomes a defensive backstop rather than the primary handler. Non-blocking — no existing config uses inverted bounds. |
| Non-blocking (new informational — JIT static-field) | `src/environment/state.py` — `animal_tags: tuple = struct.field(pytree_node=False)` | Live-verified: a tag-only swap (same N, same class order, same bounds, different `animal_tags` string tuple) triggers a full JIT recompile of `jax_step`. This is correct behaviour — `animal_tags` is `pytree_node=False` — but it is not documented in the plan's §"JIT recompile" section or in `test_no_recompile.py`. A user who runs a sweep varying only the tag strings for labelling purposes will encounter unexpected recompiles with no guidance. | Add a sentence to plan §"JIT recompile" (§"Risks §2") and to `test_no_recompile.py`'s module docstring: "Tag-only swaps (same N + same class order + same bounds but different `animal_tags` strings) also trigger recompile because `animal_tags` is `pytree_node=False`. Tags are metric labels, not expected to vary within a run, so this is not a practical concern — but it should be documented for sweep authors." Not a CP5 bug; informational only. |
| CLOSED (carry-over from CP1/CP2-CP4) | `src/environment/config_loader.py:770-795` | `print("="*60)` placement diagnostics unconditional stdout. | Resolved in CP5. All print calls now `_loader_log.debug(...)`. Confirmed live at `config_loader.py:776-795`. |

---

### Checklist

**1. Observation / Noise Modality Consistency — PASS**

Live-loaded `02-entities-distributional.yaml`. All sensors disabled (`visual_sensor_enabled: false`, `olfactory_enabled: false`, `nociception_enabled: false`, `proprioception_enabled: false`, `interoceptive_nociception_enabled: false`, `injury_observable: false`, `nutrition_observable: false`). Active observation breakdown: `{'Satiation': 1, 'Collision': 5}`. `perceptual_noise.enabled: false`. `noise_modality_order: ()`. `noise_modes.shape == (13,)` (all zeros).

With noise disabled, the modality-order tuple is empty and the noise array is all-zero padding — no KeyError risk at runtime. No active sensor is missing from the (empty) modality order. The 13-element zero-padded `noise_modes` shape is preserved from CP1. The five parity tests (`test_visual_parity.py` × 2 + `test_extero_noc_parity.py` × 3) all pass with no change to the noise or observation pipeline. No 14th sensor introduced.

**1.5. Behavior-measures Bush Presence — N/A**

`behavior_measures` is not present in `02-entities-distributional.yaml`. No behavior-measures section in any CP5 scope config.

**2. Mandatory-Key Discipline — PASS-WITH-MINOR-CONCERN (carry-over)**

Live-verified: `02-entities-distributional.yaml` provides all 12 non-distributional fields explicitly for the predator entry (`class`, `behaviour`, `tag`, `count`, `properties`, `properties_std`, `move_interval`, `nociception_intensity`, `damage`, `spawn_area`, `patrol_area`, `attack_delay`) plus all 5 distributional fields (4 as `[low, high]` ranges, 1 scalar). The two wander rabbit entries omit the 5 distributional fields — confirmed auto-filled to `[0, 0]` as expected. `perceptual_noise.enabled: false` is explicit in the config (no reliance on the soft default). The `perceptual_noise.enabled` soft default at `config_loader.py:930` remains a non-blocking concern with expanded scope (see Findings).

**3. Static-Field and JIT Recompile Risk — PASS (with new informational note)**

Confirmed via live test: bounds-only swap (config A `[0,5]` → config B `[2,7]`, same N=3, same class order) produces exactly 1 compile — no recompile. Class-ordering swap (config C `[pred, neutral, pred]` → config D `[pred, pred, neutral]`) produces 2 compiles — recompile fires. Both match `test_no_recompile.py` assertions and developer's CP5 report.

New informational finding: tag-only swap (same N, same class order, different `animal_tags` string values) also triggers recompile because `animal_tags` is `pytree_node=False`. Live-verified. This is correct behaviour (tags are static dispatch keys for metric routing) but is undocumented. No practical sweep concern since tags do not vary within a training run.

No sweep config in CP5 varies any static field. `02-entities-distributional.yaml` is a standalone smoke config, not part of a sweep. No recompile risk at config-level.

**4. Known Latent-Bug Recurrences — PASS (no triggers found)**

- `body.overeating_death: false` in `02-entities-distributional.yaml`. No trigger.
- `random_start_pos: false` — agent spawns at fixed `start_pos: [5, 5]`. No spawning-on-entity risk.
- Legacy `property:` (singular) key: not present in any CP5 file.
- `terminated` vs `done`: unaffected.
- Resource respawn occupancy gap: `resources: []` — not applicable.
- `body.start_satiation` / `body.random_start_satiation`: both present (`start_satiation: 70.0`, `random_start_satiation: false`). Schema-required, runtime-ignored, no new caller reliance.

**5. Schema Padding and Modality-Count — PASS**

`noise_modes.shape == (13,)` confirmed live for `02-entities-distributional.yaml`. Padding unchanged. No 14th sensor introduced by CP5. `noise_modality_order` is an empty tuple (noise disabled) — consistent with 0 active modalities when `perceptual_noise.enabled: false`. The CP4 baseline parity tests confirm the canonical 10-modality + 3-padding layout for training configs.

**6. Cross-Config Coherence — N/A**

CP5 adds a single standalone smoke config, not a sweep. No cross-config comparison needed.

---

### CP5-specific Findings

#### 1. `02-entities-distributional.yaml` — Schema completeness

All fields present and correct:

- `class`, `behaviour`, `tag` explicit for all 3 entries. PASS.
- All 12 non-distributional mandatory fields explicit for the predator entry. PASS.
- 4 of 5 distributional fields use `[low, high]` syntax with `low != high` — genuinely non-degenerate. PASS.
- 5th distributional field (`lose_interest_multiplier: 1.5`) is scalar — degenerate `[1.5, 1.5]`. Covers the scalar backward-compat path. PASS.
- Wander rabbits omit distributional fields — auto-filled `[0, 0]`. Confirmed live. PASS.
- Scene topology: 1 predator + 2 wander rabbits = parity-reference topology. PASS.
- `placement.mode: per_entity` — matches smoke reference config. PASS.
- `resources: []`, `obstacles: []` — minimal scene. PASS.
- `perceptual_noise.enabled: false` explicit. PASS.
- Live-load: config loads without error. Verified `animal_detect_low = [0, 0, 0]`, `animal_detect_high = [5, 0, 0]`, etc. PASS.

#### 2. Distributional YAML edge cases

| Case | Fixture | Test | Result |
|---|---|---|---|
| Scalar → `[s, s]` | `dist_scalar.yaml` | `test_scalar_5_gives_degenerate_range` | PASS |
| `[low, high]` range | `dist_range.yaml` | `test_range_0_5_gives_correct_bounds` | PASS |
| `[5]` one-element list → ValueError | `dist_malformed_single.yaml` | `test_single_element_list_raises_valueerror` | PASS |
| `"five"` string → ValueError | `dist_malformed_str.yaml` | `test_non_numeric_string_raises_valueerror` | PASS |
| `[5, 3]` inverted range | (no fixture) | (no test) | SILENTLY ACCEPTED — `jnp.maximum` guard clips to degenerate. Non-blocking. |
| `[1, 3, 5]` three-element list → ValueError | (no fixture) | (no test) | ValueError RAISED — live-verified; `len(val) != 2` check at `config_loader.py:264` handles this correctly. |

The `dist_*.yaml` fixture files are documentation artifacts — no Python test code loads them by file path. All four covered cases are exercised by inline YAML strings in `test_distributional_yaml.py`.

#### 3. `build_episode_log_dict` schema correctness

Live-verified via `jax_reset(params, PRNGKey(0))` on `02-entities-distributional.yaml`:

- Key namespace: `Episode/sampled_<field>_<tag>`. Consistent with plan §"Per-episode logging". PASS.
- 5 keys per tag: `sampled_detect`, `sampled_max_stamina`, `sampled_recovery`, `sampled_hunt_thresh`, `sampled_lose_interest`. PASS.
- All 15 values (5 fields times 3 animals) are Python `float`. Confirmed by `test_logged_values_are_python_floats`. PASS.
- Predator sampled values within bounds: `detect_val ∈ [0, 5]`, `stam_val ∈ [20, 40]`, `lose_interest = 1.5 (constant)`. Confirmed live. PASS.
- Wander animals (rabbit0, rabbit1): all 5 sampled values = `0.0`. PASS.
- Zero-N config: covered by CP2's `test_per_episode_sampling.py` zero-animal smoke (test 8). CP2 parity fixture still passes. PASS.
- `sampled_wandb_keys(tags)` output set exactly matches `build_episode_log_dict(state, params).keys()`. Asserted by `test_keys_match_build_dict_keys`. PASS.

#### 4. Carry-forward NB items status

| Item | Prior status | CP5 status |
|---|---|---|
| `perceptual_noise.enabled` soft default at `config_loader.py:930` | Open (NB-2 from CP1) | Still open. Developer's CP5 report clarifies: ~46 `configs/models/` configs lack the key; promoting to mandatory requires a separate sweep. Not a CP6 blocker. |
| `print("="*60)` placement diagnostics | Open (NB-3 from CP1) | CLOSED. `config_loader.py:776-795` now uses `_loader_log.debug(...)`. Confirmed live. |
| 31/86 parity coverage gap not in plan body | Open (flagged CP1) | CLOSED. Plan v0.4 §"Test Plan §(a)" and Coverage Breakdown table now document the 31/86 split with the rationale. Fold-back at commit `cddf6f2`. |
| Plan pseudocode inconsistent with `fold_in` implementation | Open (new in CP2-CP4) | CLOSED. Plan v0.4 §"Per-episode sampling inside `jax_reset`" and §"Risks §7 Sub-case N0" now show `fold_in` with full rationale. Fold-back at commit `cddf6f2`. |

#### 5. JIT-recompile invariants

| Scenario | Expected | Test | Result |
|---|---|---|---|
| Bounds-only swap, same N + same class order | No recompile | `TestNegativeControl::test_bounds_change_no_recompile` | PASS — count=1 after both configs |
| Class-ordering swap, same N | Recompile | `TestPositiveControl::test_class_ordering_swap_triggers_recompile` | PASS — count=2 after second config |
| Tag-only swap, same N + same class order + same bounds | Recompile | (no test — live-verified in this audit) | Recompile fires. `animal_tags` is `pytree_node=False`. Correct but undocumented. |

The audit brief question "should swapping tags trigger recompile?" is answered yes. `animal_tags` is `pytree_node=False`, so any change to the tag tuple changes JAX's trace key. Tags are metric-label strings not expected to vary within a training run; sweeps that vary only tags for labelling purposes would see unexpected recompiles. This should be documented but is not a CP5 bug.

#### 6. `test_per_episode_logging.py` schema-level checks

- Exact 5-per-tag key set asserted by `test_all_five_keys_present_for_each_tag` and `test_key_count` (5 × N). PASS.
- Zero-N coverage: inherited from CP2's `test_per_episode_sampling.py` zero-animal smoke (test 8). PASS.
- Mixed degenerate / non-degenerate within one config: covered by `test_distributional_yaml.py::TestDistributionalConfig::test_lose_interest_degenerate` (scalar field) alongside the range assertions. PASS.

#### 7. Obs/noise modality invariants (carry-forward)

Five parity tests run live and all pass:

```
tests/env/test_visual_parity.py::test_visual_parity_byte_equal               PASSED
tests/env/test_visual_parity.py::test_visual_channel_layout                   PASSED
tests/env/test_extero_noc_parity.py::test_extero_noc_parity_byte_equal       PASSED
tests/env/test_extero_noc_parity.py::test_only_damaging_animals_contribute   PASSED
tests/env/test_extero_noc_parity.py::test_nociception_enabled_in_reference_config PASSED
```

Visual channel 5 (predator) and 7 (neutral) preserved. Noise array padding = 13. `noise_modality_order` 10-entry tuple unchanged for training configs. CP5 touches no sensor or noise code.

---

### Open Items (Status after CP5)

| Item | Status |
|---|---|
| `perceptual_noise.enabled` soft default | Open. Scope clarified: requires a `configs/models/` sweep before mandatory promotion. Not a CP6 blocker. |
| Inverted bounds `[hi, low]` accepted silently | New informational finding. `jnp.maximum` guard clips it safely; no crash. Consider adding `lo <= hi` check in `_parse_distributional`. CP6 or cleanup commit. |
| Tag-only swap triggers undocumented recompile | New informational finding. Correct JAX behaviour; undocumented in plan and tests. Add a note to plan §"JIT recompile" and `test_no_recompile.py` docstring. CP6 or cleanup commit. |
| `dist_*.yaml` fixtures unreferenced by tests | New informational finding. Documentation artifacts only. Consider adding path-loading tests or amending comments. Low priority. |

---

### Conclusion

**ACCEPT — CP6 green-lit from config-soundness side.**

CP5 is clean. All blocking invariants hold under live load and live test execution (22 CP5 tests pass, 5 carry-forward parity tests pass). The `print` gating carry-forward item is now closed. The `perceptual_noise.enabled` item remains open with a scope clarification that prevents a simple fix. Three new informational findings are documented — inverted-bounds silent acceptance, tag-swap recompile behaviour, and unreferenced fixture files — none of which block training or compromise correctness. CP6 may proceed.

Audited by: env-config-auditor
