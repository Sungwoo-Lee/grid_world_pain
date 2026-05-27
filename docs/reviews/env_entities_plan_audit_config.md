---
title: "Config Audit — env_entities plan (pre-implementation)"
topic: env_entities
status: active
created: 2026-05-28
last_updated: 2026-05-28
aliases: [env_entities_config_audit, plan_audit_config]
---

# Config Audit — Unified Animal Entity Plan (pre-implementation)

**Plan**: `docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md`
**Branch**: `v2.0` (most recent commit `7110323`)
**Auditor**: `env-config-auditor` (schema soundness + config migration safety)
**Run date**: 2026-05-28

## Verdict: ACCEPT-WITH-REVISIONS

The plan is structurally sound. The 86-config migration count is verified correct, the visual sensor parity invariant is well-designed, and the backward-compat loader path is logical. However, three issues require resolution before CP1 lands: one **blocker** (DISTRIBUTIONAL_FIELDS vs. neutral-animal omission contradiction), one **blocker** (`attack_delay` missing from unified schema field list), and one **hard concern** (parity test scope covers 74 of 86 migrated configs).

## Findings

| Severity | Location | Issue | Suggested fix |
|---|---|---|---|
| BLOCKER | Plan L90 vs. L128-131 and L446 | DISTRIBUTIONAL_FIELDS vs. neutral-animal field-omission contradiction. Plan says both "Missing → raise ValueError" (L446) and "loader fills these with zeros; the wander branch ignores them" (L131). | Pick one rule. Either (a) require DISTRIBUTIONAL_FIELDS only when `behaviour == "hunt"`, or (b) silently default missing fields to `[0, 0]` for non-hunt entries. |
| BLOCKER | Plan L90 | `attack_delay` omitted from the schema description text (present in example YAML L115, in `EnvParams` table L160, and read with `p_get()` mandatory in current loader at `config_loader.py:262`). A developer reading only the schema description would miss it. | Add `attack_delay` to schema description list at L90. |
| HARD CONCERN | Plan CP1 test L581 | Parity test globs only `configs/experiment/**/*.yaml` (74 files). The CP1 migration sweep atomically touches 86 configs (74 experiment + 5 continual + 6 verification + 1 environment/default.yaml). 12 configs migrated without automated parity verification. | Extend the parity test glob to all directories with migrated configs. |
| CONCERN | Plan L446 vs. "no fallback defaults" rule | `lose_interest_multiplier` currently loads with soft default `2.0` via `p.get('lose_interest_multiplier', 2.0)` (`config_loader.py:263`). The plan promotes it to mandatory. All 86 existing predator entries carry it, so safe in practice, but it's a semantic change. | Document in `_load_animals()` docstring that `lose_interest_multiplier` changed from soft-default to mandatory between v1.x and v2.0. |
| CONCERN | Plan L581 parity test | `glob('configs/experiment/**/*.yaml')` without `recursive=True` does NOT match `configs/experiment/2X2_area.yaml` (depth 0). That config carries `predator_enabled: true`. | Explicitly specify `recursive=True` in test + migration script. |
| CONCERN | Plan L81 — olfaction_parity_neutral hand-migration | Plan describes "hand-migrate to empty `predators: []`". Actual file already has `predators: []` on L11 alongside `predator_enabled: false` on L10. Migration is a one-line strip, not restructuring. | Reword: "Remove `predator_enabled: false` from `olfaction_parity_neutral.yaml`; `predators: []` is already present." |
| CONCERN | Plan L81 + L568 | Sequencing gap: if CP1 loader merges before migration sweep completes (partial commit), 86 existing configs would fail to load. | Add atomic-commit note: loader change and migration sweep land in single atomic commit. |
| CONCERN | Plan L189 JIT shape-stability | Plan documents that different N triggers recompile. Doesn't document that same N + different class ORDERING triggers recompile too (because `animal_classes` is `pytree_node=False`). | Add documentation; flag for CP5 JIT-recompile test (positive control). |
| CONCERN | Plan L102-115 vs. L446 | `damage: [15.0, 45.0]` (per-event) and `detection_range: [0, 5]` (per-episode) use IDENTICAL `[lo, hi]` syntax. Plan relies on DISTRIBUTIONAL_FIELDS membership; YAML is visually identical. Documentation-only risk for future authors. | Add inline YAML comment convention and loader-docstring annotation distinguishing per-event vs per-episode `[lo, hi]` fields. |
| NIT | `config_loader.py:410` | `config.get('environment.placement.mode', 'per_entity')` uses soft default, violating no-fallback-defaults. Pre-existing; not introduced by plan. | Separate cleanup; add `placement.mode: per_entity` to any configs missing it (most have it). |
| NIT | Plan L90 | Schema description omits `name` (which current loader reads via `p.get('name')` for tagging). | Clarify whether `name` is retained, optional, or replaced by `tag` in unified form. |

## Checklist

1. **Observation ↔ Noise Modality Consistency**: PASS. Visual channel reorder (`[res, pred, obs, neutral]` → `[res, animal, obs]`) is internal; `get_observation_breakdown` emits unified `"Visual"` modality. Predator channel 5 / neutral channel 7 preserved in `animal_visual_channel` derived array.
2. **Mandatory-Key Discipline**: PARTIALLY PASS — 2 blockers above. Backward-compat loader correctly uses `get_mandatory` for legacy keys.
3. **Static-Field / JIT Recompile Risk**: PASS with documentation gap. Same-N-different-bounds verified shareable; same-N-different-class-ordering needs documenting.
4. **Known Latent-Bug Recurrences**: N/A (plan doesn't touch overeating_death, body.start_satiation, property/properties typo).
5. **Schema Padding / Modality-Count**: PASS. No new observation modalities; `[13]` padding unaffected.
6. **Cross-Config Coherence**: PASS — parity gate is the appropriate coherence check for a structural refactor.

## Migration Sweep Audit

**Total `predator_enabled`-touching configs:** 86 (verified by `grep -rln`).

**Value breakdown:**
- `predator_enabled: false` — 1 config: `configs/verification/olfaction_parity_neutral.yaml`
- `predator_enabled: true` — 85 configs

**Directory breakdown:**

| Directory | Count | Migration action |
|---|---|---|
| `configs/experiment/**/*.yaml` | 74 | sed-strip `predator_enabled: true` |
| `configs/continual/nmn_double_return_stages/*.yaml` | 5 | sed-strip `predator_enabled: true` |
| `configs/verification/*.yaml` (except olfaction_parity_neutral) | 5 | sed-strip `predator_enabled: true` |
| `configs/verification/olfaction_parity_neutral.yaml` | 1 | Remove `predator_enabled: false` (L10); `predators: []` already on L11 |
| `configs/environment/default.yaml` | 1 | sed-strip `predator_enabled: true` (L126) |
| **Total** | **86** | |

**Plan's count (85 + 1 = 86): VERIFIED CORRECT.**

**Parity test coverage gap:** 12 of 86 migrated configs not in CP1 parity test:
- 5 continual stages (stated byte-identical copies of experiment configs — identity is manual, not machine-checked)
- 1 environment/default.yaml (base config used by many tests)
- 6 verification configs (4 `observability_gates_S1..S4` + 2 `olfaction_parity_*`)

The 4 observability gates configs use standard schema and would be caught by a full-scope backward-compat load test.

**`2X2_area.yaml` glob issue:** sits at depth 0 in `experiment/`. `glob('configs/experiment/**/*.yaml')` without `recursive=True` would miss it. Use `recursive=True`.

## Observation/Noise Channel Preservation

Existing encoding in `sensor.py:191-193`:
- Channel 5 = Predator (`one_hot(full((num_pred,), 5), 8)`)
- Channel 7 = Neutral animal (`one_hot(full((num_neutral,), 7), 8)`)

New encoding uses `params.animal_visual_channel` (a `[N] int32` built from `animal_classes` via `{'predator': 5, 'neutral': 7}`). Channel values 5 and 7 preserved. Concat order change doesn't alter channel values — only reorders entity positions in `all_pos`. Visual vector byte-identical for any config whose predator + neutral positions are unchanged.

## Backward-Compat Loader Coverage

All 86 configs carrying `predator_enabled` also carry both `predators:` and `neutral_animals:` keys (verified by automated scan). The backward-compat loader's `get_mandatory('environment.predators')` and `get_mandatory('environment.neutral_animals')` calls will succeed for all 86.

`configs/models/`, `configs/train/`, `configs/evaluation/`, `configs/logger/` do not contain environment entity sections — not affected.

## Conclusion

Fix the two blockers (DISTRIBUTIONAL_FIELDS contradiction + `attack_delay` missing from schema description) and one hard concern (parity test scope) before the developer begins CP1. The migration sweep count is verified correct. Obs/noise channel invariant is correctly preserved. Three concerns need documentation fixes before CP3/CP5: parity test scope expansion to all 86 configs, JIT recompile documentation for same-count different-class-assignment, and `[lo, hi]` per-event vs per-episode cadence annotation in YAML comments and loader docstring.

Audited by: env-config-auditor
