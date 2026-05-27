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

---

## Re-audit of v0.2 (2026-05-28, focused on blocker resolution)

**Plan version**: v0.2 (commit `6e5e03c`, branch `v2.0`)
**Scope**: Focused re-audit — verifying resolution of B-CFG-1, B-CFG-2, HC-1 from prior verdict; checking incorporated non-blocking suggestions; new surface check on `hunt_idx` / `wander_idx` / `static_idx` construction.
**Auditor**: env-config-auditor
**Date**: 2026-05-28

### Purpose

This re-audit gates CP1. The prior verdict (ACCEPT-WITH-REVISIONS, same date) identified two blockers and one hard concern. The `senior-developer` revised the plan to v0.2 and documented the fixes in the plan's Revision Log. This section verifies each fix held, checks incorporated suggestions, and surfaces one new concern introduced by the v0.2 revision itself.

### Per-finding verification table

| Finding | Status | Evidence | Notes |
|---|---|---|---|
| **B-CFG-1** — DISTRIBUTIONAL_FIELDS contradiction | PASS | Plan L93-101, L574-577 | Reconciliation is consistent — see detailed analysis below. |
| **B-CFG-2** — `attack_delay` missing from schema description | PARTIAL-PASS | Plan L91, L98, L126, L178 | Added to schema description text (L91) and Mandatory-key table (L98); present in example YAML (L126) and EnvParams table (L178). One gap: loader numbered-steps (L574-578) describe only DISTRIBUTIONAL_FIELDS handling; `attack_delay` is not named in the step-by-step recipe. Not a blocker — it delegates to existing `p_get` at `config_loader.py:262` — but the plan's Mandatory-key table at L98 contains an error in the legacy-column for wander entries that is a new concern (see NC-1 below). |
| **HC-1** — parity test scope covers only 74 of 86 configs | PASS | Plan L762-798, Test Plan §(a) | Glob widened to four-glob pattern with `recursive=True`; coverage breakdown table added. Machine-verified: the four-glob pattern captures all 86 `predator_enabled`-carrying configs (74 experiment + 5 continual-stages + 6 verification + 1 environment/default). `2X2_area.yaml` confirmed present in `experiment/**/*.yaml` with `recursive=True`. See §"HC-1 glob verification" below. |
| **C-CFG-3** — olfaction_parity_neutral wording | PASS | Plan L750 (CP1 desc), Revision Log L927 | Correctly reworded: "strip the single line `predator_enabled: false` (L10); `predators: []` is already present on L11." Confirmed against actual file: `predator_enabled: false` is on L10, `predators: []` on L11. |
| **C-CFG-4** — atomic-commit note for CP1 | PASS | Plan L750 | Explicit atomic-commit boundary stated: "loader change, the `verify_noise.py` rewrite, and the 86-config migration sweep all land in a single atomic commit." |
| **C-CFG-5** — positive control for JIT recompile test | PASS | Plan L754, Test Plan §(e) Part 2 | Positive control added: same-N different-class-ordering asserts `log.count("Compiling jax_step") == 2`. |
| **C-CFG-6** — YAML comment convention for per-event vs per-episode | PASS | Plan L586-596 | YAML cadence-distinction convention section added. `damage: [lo, hi]` documented as per-event; distributional fields documented as per-episode. Canonical example with inline comments provided. |
| `lose_interest_multiplier` mandatory-promotion semantic-change note | PASS | Plan L582 | Inline note added to loader section referencing the soft-default-to-mandatory semantic change; directs developer to add a v1.x → v2.0 note in the loader docstring. |

### B-CFG-1 detailed analysis

The reconciliation at plan L93-101 and L574-577 is internally consistent. The behaviour-conditional rule correctly resolves the original contradiction:

- `behaviour: hunt` entries: DISTRIBUTIONAL_FIELDS are **mandatory** (`ValueError` on missing).
- `behaviour: wander` / `static` entries: DISTRIBUTIONAL_FIELDS are **optional**; loader auto-fills `[0, 0]` and debug-logs. Correctly classified as an internal projection detail, not a user-facing fallback default. The wander/static code paths (`_wander_step`, `_hunt_step[wander_idx]` is never called) do not read these arrays.
- Legacy `neutral_animals:` re-projection: treated as wander-equivalent; auto-fill `[0, 0]` is purely internal.

The worked-example YAML comment (plan L140-143) and the loader numbered-steps (plan L574-577) are consistent with the table.

**Edge case: `behaviour: hunt` entity with `properties_std: 0` (a scalar, not a list).** The plan (L91, schema description) says `properties_std` is mandatory. A scalar `0` would be read as a zero-vector `[0, 0, 0, 0, 0]` (the `_read_properties_std` helper handles this). This is not ambiguous; `properties_std` is not a DISTRIBUTIONAL_FIELD and has no `[low, high]` interpretation.

**Edge case: `behaviour: hunt` entity where all five DISTRIBUTIONAL_FIELDS are present but one has value `0`.** The plan correctly handles this: `detection_range: 0` → degenerate range `[0, 0]` → sampled value `0.0` every episode (plan L349). No ambiguity.

The B-CFG-1 reconciliation PASSES.

### B-CFG-2 detailed analysis

`attack_delay` now appears in four locations:

1. **Schema description text** (plan L91): listed among the entity fields. PRESENT.
2. **Mandatory-key table** (plan L98): explicitly stated as mandatory for all behaviour modes. PRESENT.
3. **Example YAML** (plan L126): `attack_delay: 3` in the predator entry. PRESENT. (Not present in the wander entry, which is correct per the plan's choice to omit it from the example.)
4. **EnvParams table** (plan L178): `animal_attack_delay | [N] int32 | — | post-attack cooldown`. PRESENT.

The loader numbered-steps (plan L574-578) do not enumerate `attack_delay` explicitly but defer to the existing `p_get` at `config_loader.py:262`. This is acceptable as a documentation choice since `attack_delay` is not a DISTRIBUTIONAL_FIELD — the loader recipe's numbered-steps specifically describe how DISTRIBUTIONAL_FIELDS are processed.

However, a new concern (NC-1) arises from row L98 of the Mandatory-key table, which is addressed below.

B-CFG-2 is PARTIAL-PASS: the field is present in all four locations, but row L98 carries a factual error about the legacy column that creates a developer trap.

### HC-1 glob verification

Live verification with the project's Python environment:

```
glob('configs/experiment/**/*.yaml', recursive=True)  +
glob('configs/continual/**/*.yaml', recursive=True)   +
glob('configs/verification/**/*.yaml', recursive=True) +
['configs/environment/default.yaml']
= 91 total files, of which 86 carry `predator_enabled`
```

The 5 extra files (top-level `configs/continual/*.yaml` without `predator_enabled`) are correctly included in the glob but have no `predator_enabled` key, so the parity test's `predator_enabled`-migration assertion will simply find nothing to strip — no harm.

Spot-check results:

| Check | Result |
|---|---|
| `configs/environment/default.yaml` in glob | PASS — added as explicit entry |
| `configs/experiment/2X2_area.yaml` in glob | PASS — `recursive=True` captures depth-0 files |
| All 6 `configs/verification/*.yaml` in glob | PASS — confirmed by `find` |
| All 5 `configs/continual/nmn_double_return_stages/*.yaml` in glob | PASS — confirmed by `find` + `grep` |

HC-1 PASSES.

### Incorporated suggestions verification

| ID | Verification |
|---|---|
| C-CFG-3 | PASS — wording is accurate to the actual file on disk (L10/L11 confirmed). |
| C-CFG-4 | PASS — atomic-commit note present in CP1 description. |
| C-CFG-5 | PASS — positive control in Test Plan §(e) Part 2. |
| C-CFG-6 | PASS — YAML cadence-distinction section at plan L586-596 with canonical comment examples. |

### New surface check: `hunt_idx` / `wander_idx` / `static_idx` construction

The plan specifies (plan L234, L544-547) that `hunt_idx`, `wander_idx`, `static_idx` are `pytree_node=False` Python int tuples on `EnvParams`, built at param-load time as:

```python
hunt_idx   = tuple(i for i, b in enumerate(params.animal_behaviours) if b == 'hunt')
wander_idx = tuple(i for i, b in enumerate(params.animal_behaviours) if b == 'wander')
static_idx = tuple(i for i, b in enumerate(params.animal_behaviours) if b == 'static')
```

**Are these computed deterministically from `animal_behaviours` at param-build time?** Yes. `animal_behaviours` is itself `pytree_node=False` (plan L164, L541), so it is fixed at load time. The index tuples cannot drift. The plan's revision note at L944 confirms this: "the constructor is a one-liner. No new design decision required."

**Failure mode for a bad behaviour value (e.g., `behaviour: chase`).** The plan does not explicitly specify `ValueError` for an unrecognized behaviour string. The `ANIMAL_BEHAVIOUR_TO_INT` dict at plan L560 maps only `{"wander": 0, "hunt": 1, "static": 2}`. If a config author writes `behaviour: chase`, the integer-coding step (`ANIMAL_BEHAVIOUR_TO_INT["chase"]`) will raise a `KeyError` at load time — this is acceptable early-fail behaviour, though it is not explicitly documented as a `ValueError` with a clear message in the plan. The index-tuple construction would silently produce an empty tuple for all three behaviour categories (`hunt_idx = ()`, `wander_idx = ()`, `static_idx = ()`), which means the entity would be treated as `static` (no update, no draws) with no error. This is a silent misclassification risk if the integer-coding step is not reached before the tuple construction.

**Assessment:** CONCERN-LEVEL (not a blocker). The risk is that a future config author who introduces a typo (`behaviour: "Hunt"` vs `behaviour: "hunt"`) would see the entity silently treated as static. The fix is simple: add an explicit validation step in `_load_animals()` that raises `ValueError(f"Unknown behaviour: {b!r}. Must be one of {list(ANIMAL_BEHAVIOUR_TO_INT)}")` before building the index tuples. This is a documentation and implementation note for the developer, not a plan-level blocker.

### New concern surfaced by v0.2: NC-1 — Mandatory-key table L98 incorrectly states `attack_delay` is mandatory in the legacy `neutral_animals:` re-projection path

**Finding:** Plan L98, Mandatory-key table, legacy column for `attack_delay` states: "mandatory (`p_get`)". This is incorrect. The current `neutral_animals:` loader path at `config_loader.py:330-366` does NOT read `attack_delay` from neutral entries. The actual legacy neutral loader reads only: `properties`, `properties_std`, `nociception_intensity` (soft-defaulted), `move_interval`, `patrol_area` (soft-defaulted), `spawn_area` (soft-defaulted), and `tag` (soft-defaulted). There is no `n_get(n, 'attack_delay')` call.

**Consequence:** If a developer reads the plan's Mandatory-key table at L98 and implements `attack_delay` as mandatory during legacy re-projection for `neutral_animals:` entries, the following configs will fail to load (they have `neutral_animals:` entries without `attack_delay`):
- `configs/verification/olfaction_parity_neutral.yaml` (the `neutral_animals:` entry at L12-20 has no `attack_delay`)
- `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` (parity-reference config — its two `neutral_animals:` entries at L108-126 have no `attack_delay`)
- Every other config whose `neutral_animals:` entries lack `attack_delay` (confirmed: `olfaction_parity_neutral.yaml` has no `attack_delay` in its neutral section)

This would cause CP1's parity test to fail at load time for these configs, not at the step-comparison level.

**Correct behaviour for the legacy re-projection:** during `neutral_animals:` → `behaviour='wander'` re-projection, `animal_attack_delay` for those entries should be auto-filled to `0` (zero-int), not read via mandatory `p_get`. Wander entities never reach the attack-timer update path, so `0` is semantically correct and does not require a user-facing key.

**Severity:** CONCERN (not a blocker, because the CP1 parity test itself will catch this if the developer implements it incorrectly — the test would fail at load time and the error message from `p_get` would point directly to the missing key). However, the Mandatory-key table as written will mislead the developer. The table's L98 legacy column should be corrected from "mandatory (`p_get`)" to "auto-fill `0` (wander entities never attack; no user-facing key required)".

**Note:** Similarly, the Mandatory-key table L97 claims `damage` is mandatory for the legacy `neutral_animals:` re-projection path. The actual legacy neutral loader does not read `damage` at all (confirmed by reading `config_loader.py:330-366`). The `damage` field was not in scope as a blocker in the prior audit because the legacy neutrals simply have no `pred_damage`-equivalent in `EnvParams` today. Under the unified schema, wander entities will have `animal_damage` — the plan should specify that the legacy re-projection auto-fills `[0.0, 0.0]` for this field too, consistent with the DISTRIBUTIONAL_FIELDS auto-fill pattern. This is the same class of error as NC-1 for `attack_delay`.

### Findings table

| Severity | Location | Issue | Suggested fix |
|---|---|---|---|
| CONCERN | Plan L98 (Mandatory-key table, legacy column) | `attack_delay` stated as "mandatory (`p_get`)" for legacy `neutral_animals:` re-projection. Actual current loader does not read `attack_delay` from neutrals. Will cause CP1 parity test to fail at load time for configs like `olfaction_parity_neutral.yaml` and the parity-reference config if developer implements as written. | Correct the legacy column for `attack_delay` to: "auto-fill `0` (wander entities never attack; no user-facing key)". Same correction needed for `damage` in the legacy neutral column of L97. |
| CONCERN | Plan L560, loader step 3 | No explicit validation of the `behaviour` string value against `ANIMAL_BEHAVIOUR_TO_INT`. A typo (`behaviour: "Hunt"`) silently produces empty index tuples and treats the entity as static. | Add explicit `ValueError` with message "Unknown behaviour: {b!r}. Must be one of {list(ANIMAL_BEHAVIOUR_TO_INT)}" in `_load_animals()` before building index tuples. Document this in the loader spec. |
| NIT | Plan L574-578 (loader numbered-steps) | `attack_delay` not mentioned in the step-by-step loader recipe; only DISTRIBUTIONAL_FIELDS handling is described. A developer reading only step 3 will not know that `attack_delay` must also be read. | Add a step 3a or note in the loader recipe: "All non-distributional mandatory fields (`attack_delay`, `move_interval`, `damage`, etc.) are read via `p_get` for all behaviour modes (hunt and wander/static). For wander/static in the legacy path, `attack_delay` is auto-filled to `0` since the legacy schema never carried it." |

### Checklist (re-audit scope)

- [x] (1) Observation / Noise Modality Consistency — N/A (no change in v0.2, prior PASS stands)
- [x] (2) Mandatory-Key Discipline — PARTIAL-PASS. B-CFG-1 fully resolved. B-CFG-2 present in all four required locations. New concern NC-1: Mandatory-key table L98 misstates the legacy neutral re-projection rule for `attack_delay` (and `damage`). Not a plan-level blocker but requires correction before CP1 to avoid developer error.
- [x] (3) Static-Field / JIT Recompile Risk — PASS. `hunt_idx` / `wander_idx` / `static_idx` are `pytree_node=False`, computed from `animal_behaviours` (also static). C-CFG-5 positive control now in Test Plan §(e).
- [x] (4) Known Latent-Bug Recurrences — N/A (no change in scope)
- [x] (5) Schema Padding / Modality-Count — N/A (no change)
- [x] (6) Cross-Config Coherence — PASS. HC-1 glob machine-verified; all 86 configs confirmed captured.

### HC-1 Coverage Breakdown (verified)

| Directory | Count in glob | `predator_enabled` count |
|---|---:|---:|
| `configs/experiment/**/*.yaml` (recursive) | 74 | 74 |
| `configs/continual/**/*.yaml` (recursive) | 10 | 5 |
| `configs/verification/**/*.yaml` (recursive) | 6 | 6 |
| `configs/environment/default.yaml` (explicit) | 1 | 1 |
| **Total** | **91** | **86** |

The 5 continual configs without `predator_enabled` (top-level `configs/continual/*.yaml`) are harmlessly included — the migration sweep will simply find nothing to strip in those files.

### Final verdict

**ACCEPT-WITH-MINOR-REVISIONS.**

All three blockers (B-CFG-1, B-CFG-2, HC-1) are resolved. The plan is ready for CP1 implementation with two non-blocking corrections that should be made to the Mandatory-key table before the developer reads it:

1. **NC-1 (Concern):** Correct the Mandatory-key table at plan L97-98 to specify that `attack_delay` and `damage` for the legacy `neutral_animals:` re-projection path are **auto-filled** (to `0` and `[0.0, 0.0]` respectively), not read via mandatory `p_get`. This prevents a developer misreading the table from breaking CP1's parity test for configs like `olfaction_parity_neutral.yaml` and the parity-reference config (`01-interoNocicept_sameProp.yaml`).

2. **Behaviour-string validation (Concern):** Add explicit `ValueError` for unrecognized `behaviour:` values in `_load_animals()`. Document in the loader spec.

Neither concern requires a re-audit. The developer can apply these corrections inline as part of CP1 implementation and note them in the implementation report. If the CP1 parity test passes for all 86 configs, NC-1 was handled correctly.

Audited by: env-config-auditor
