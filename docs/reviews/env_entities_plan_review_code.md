---
title: "Code Review — env_entities plan (pre-implementation)"
topic: env_entities
status: active
created: 2026-05-28
last_updated: 2026-05-28
aliases: [env_entities_code_review, plan_review_code]
---

# Code Review — Unified Animal Entity Plan (pre-implementation)

**Plan**: `docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md`
**Branch**: `v2.0` (most recent commit `7110323`)
**Reviewer**: `code-reviewer` (JAX/Flax/vmap/PRNG correctness)
**Run date**: 2026-05-28

## Verdict: REJECT — masked-combine breaks PRNG parity for neutrals

The plan has one **blocking correctness flaw** and several **gaps in file coverage** that together prevent the CP1 parity gate from passing as written. The blocker is structural: the masked-combine approach in `update_animals` cannot achieve byte-parity for neutral entities because today's `update_neutral_animals` draws randoms over `(N_neutral,)` whereas the refactor will draw over `(N_animals,)` and slice. JAX's threefry is prefix-stable, so predators (front of array) parity holds but neutrals (positions ≥ N_pred) get different draws.

Recommend (a) per-subset call pattern that preserves draw shape.

## Blocking issues (must fix before CP1 lands)

### B1. Masked-combine breaks PRNG parity for neutrals (Plan L193–240, L568)

Today:
- `update_predators(predator_key)`: `randint(subkey, (num_pred,), -1, 2)` and `uniform(subkey3, (num_pred,))` (`core.py:192-193, 218`).
- `update_neutral_animals(neutral_key)`: `randint(subkey, (num_neutral,), -1, 2)` (`core.py:261-262`).

After refactor with masked-combine, both `_hunt_step` and `_wander_step` are called with the full unified array of length `num_animals`. The draws become `randint(subkey, (num_animals,), -1, 2)`.

Empirical confirmation (3-rabbit example):
```
today's neutral_key jitter_r at (3,): [1 1 0]
refactor wander_key jitter_r at (4,) sliced [1:]: [1 0 1]   ← DIFFERENT
```

Predators (indices 0..num_pred-1) retain byte-parity. Neutrals (indices num_pred..num_animals-1) do NOT.

**Failure surface.** Every config with both predators and ≥1 neutral fails the CP1 byte-parity gate.

**Fix options**:
1. **Per-subset call pattern (RECOMMENDED).** Statically index out the hunt-subset and wander-subset (host-side computed from `pytree_node=False` tags) and call `_hunt_step` with length `num_pred`, `_wander_step` with length `num_neutral`, then scatter back. Preserves `(num_pred,)`/`(num_neutral,)` draw shapes.
2. Reverse entity ordering. Fragile; not recommended.
3. Drop byte-parity, use statistical-equivalence gate. Weakens the load-bearing constraint.

### B2. `sense_extero_nociception` not in File Changes

`src/environment/sensor.py:59-87` reads `state.pred_pos` and `params.pred_nociception`. Plan §"File Changes" → `src/environment/sensor.py` only mentions visual (168-198) and olfactory (290-295). After CP1's `pred_*` removal, this function raises AttributeError.

**Fix**: Add to File Changes. Use `dist_animal = jnp.linalg.norm(state.animal_pos - agent_pos, axis=-1)` and mask via `params.animal_is_damaging` to preserve "only predators contribute to extero-noc". CP4 parity gate must cover extero-noc.

### B3. `dreamer_srl_main.py` reads `env_params.predator_tags` / `neutral_tags`

`src/algorithms/dreamer_srl/dreamer_srl_main.py:522-523` reads both. Plan L552-554 claims no trainer file touches these — wrong. After CP1 removes them from EnvParams, this raises AttributeError.

**Fix**: Either (a) keep `predator_tags`/`neutral_tags` as `@property` derived from `animal_tags`+`animal_classes`, or (b) update the script to read `animal_tags` via `class_indices`. Option (a) is minimum-diff and matches the legacy-alias strategy used elsewhere in the plan.

### B4. `scripts/verify_noise.py` constructs EnvState with removed fields

Plan L556-558 claims parity scripts need no edits. Actual: `scripts/verify_noise.py:37-43` constructs `EnvState(pred_pos=..., pred_state=..., pred_stamina=..., pred_move_timer=..., pred_attack_timer=..., neutral_pos=..., neutral_move_timer=...)`. After CP1 these field names don't exist.

**Fix**: Add to File Changes; rewrite using `animal_*` field names.

### B5. `hit_neutral` info-dict semantic flip (pre → post step)

Today (`core.py:440-442`):
```python
'hit_neutral': any(all(state.neutral_pos == new_agent_pos, axis=-1))   # PRE-step
'hit_predator': any(at_predator)  # at_predator uses new_pred_pos — POST-step
```

Pre-existing asymmetry. Plan at L467 says `info['hit_neutral']` becomes `any(at_animal & ~animal_is_damaging)` where `at_animal` uses `new_animal_pos` — POST-step. Changes behaviour: today, agent moving onto rabbit's old square (before rabbit moves) registers `hit_neutral=True`; after refactor, agent must end up on rabbit's NEW square.

**Fix**: Preserve the asymmetry — compute `at_neutral_pre = jnp.all(state.animal_pos == new_agent_pos, axis=-1) & ~animal_is_damaging` separately.

## Non-blocking concerns

- **C1**: Parity test glob only covers `configs/experiment/**/*.yaml` (74 files); 12 more configs migrated by CP1 sweep go untested. Widen to all directories where `predator_enabled` was stripped.
- **C2**: CP3's `placement.types:` re-projection under-specified. Enumerate `per_type` configs in CP1 pre-flight (`grep -l "mode: per_type" configs/`).
- **C3**: JIT-recompile test methodology — add `assert log.count("Compiling jax_step") == 1` regex on captured `jax_log_compiles` output.
- **C4**: Per-episode sampling test will fail spuriously for degenerate ranges; needs `[0, 5]` config explicitly. Missing test for cross-field independence between the 5 sampled fields.
- **C5**: Info-dict legacy aliases — plan addresses `hit_neutral` but not `hit_predator`/`damage_predator`/`damage_obstacle`/`damage_hiding_predator`. Quick grep needed.

## Conventions audit checklist

| Convention | Status | Notes |
|---|:---:|---|
| Pytree immutability (`._replace`) | ✅ | Plan uses `state._replace(**kwargs)`; no in-place mutation. |
| JIT shape-stability (static fields) | ✅ | `animal_classes`/`behaviours`/`tags` correctly `pytree_node=False`. Int-coded arrays correctly traced. |
| JIT shape-stability across configs (same N, different bounds) | ✅ | Bounds stored in traced `[N]` arrays, not static. |
| vmap safety | ✅ | Entity-axis vmap pattern preserved. |
| PRNG threading — `damage_key` reuse preserved | ✅ | All 3 uses preserved. |
| PRNG threading — 6-key step split preserved | ✅ | Verified byte-identical at split level. |
| PRNG threading — reset 5→6 split | ✅ | Prefix-stable; first 5 indices match. |
| PRNG — masked-combine parity | ❌ | **B1 above.** |
| Sensor — `sense_extero_nociception` | ❌ | **B2 above** — not in File Changes. |
| Configuration Protocol (`get_mandatory`) | ✅ | Plan reiterates "Missing → raise ValueError". |
| Backward-compat loader covers all 86 configs | ⚠ | C1 — only 74 in parity test. |
| `predator_enabled` semantic safety | ✅ | Verified `predator_enabled` is dead code in `src/`. Strips are true no-ops. |

## Test plan assessment

### (a) Per-step parity test
Would catch B5 (info-dict diff). Would catch B1 if asserted state includes `state.animal_pos[neutral_slice]` (plan says "every common field" — should). Recommend committing pre-refactor `(obs, state)` traces as compressed npz fixtures so the test is self-contained.

### (b) Per-episode sampling test
Reproducibility + divergence + per-instance independence covered. Missing: degenerate-range guard, cross-field independence test.

### (d) Visual parity test
Adequate. Sharpen by slicing visual block via `get_observation_breakdown(params)["Visual"]`.

### (e) JIT-recompile test
Add positive control: swap `animal_classes` tuples at same N → SHOULD trigger recompile.

### Missed failure modes
1. Per-episode sampled value leaking into `EnvParams` — add `assert hasattr(state, 'animal_detect_sampled') and not hasattr(params, ...)`.
2. Behaviour-mask wrong axis — add 4-animal config test with `[hunt, wander, hunt, static]`.
3. `animal_state` collapsing for non-hunt — add 1000-step assertion `state.animal_state[wander_mask] == 0`.
4. Document that wander/static entities get unused per-episode draws (intentional shape uniformity).
5. `per_type` placement enumeration in CP3 pre-flight.

## Summary

Update the plan to address blockers B1–B5 before any code lands. Non-blockers can be deferred to implementation. The plan is otherwise carefully thought-through; blockers are missed code surfaces (except B1, which requires localised redesign of `update_animals`).

Reviewed by: code-reviewer

---

## Re-review of v0.2 (2026-05-28, focused on blocker resolution)

### Verdict: REJECT — two new blockers surfaced during the re-review of `jax_reset` parity surfaces

The v0.2 plan resolves every blocker I originally raised (B1–B5 all PASS — see table below). The per-subset call pattern for `update_animals` is solid, the `@property` legacy aliases close the trainer-read gap, `sense_extero_nociception` and `scripts/verify_noise.py` are now in File Changes, and the pre/post `hit_*` asymmetry is preserved correctly.

However, re-reading `jax_reset` end-to-end with the v0.2 design surfaced **two byte-parity breaks the plan does not address** — both in code paths that were OUT of my original five blockers' scope. Both fire during the CP1 parity gate, so they will be detected, but they are not solved.

**New blockers:**
- **N1 — `jax_reset` placement: concat order `[res, animal, obs]` differs from today's `[res, pred, obs, neutral]`.** Changes `resolve_overlaps_global` processing order — obstacles now resolved BEFORE neutrals (today: AFTER). Breaks byte-parity for any config where an obstacle and a neutral could collide in spawn.
- **N2 — `jax_reset` property sampling: collapsing 4-way `prop_key` split to a single `prop_key_animal` changes the threefry stream.** A `jax.random.split(key, 3)` produces different sub-keys than a `jax.random.split(key, 4)`; furthermore the neutral half of `animal_property_sampled` cannot byte-match today's `neutral_property_sampled` because it would come from the predator-key prefix of a `(N_animals, V)` draw rather than from an independent `prop_key_neutral` draw.

Both are fixable without restructuring CP1–CP6 — the fix in each case is **preserve today's per-type key splits internally**, even though the final stored arrays are unified. Recommendation in §"New blockers — what to fix" below.

### Per-blocker verification table

| Blocker | Status | Plan location | Resolves? | Notes |
|---|:---:|---|:---:|---|
| **B1** — masked-combine breaks neutral PRNG parity | **PASS** | §"`update_animals()` dispatch" L230–321; Risks item 1 L78; CP1 spec L750; pseudocode L246–321 with per-subset slicing + `.at[idx].set` scatter | ✅ | Per-subset call pattern is correctly designed. `hunt_idx`/`wander_idx`/`static_idx` declared `pytree_node=False` (L545–547), so `jnp.asarray(params.hunt_idx)` is a JIT-static constant — slicing is free. `_hunt_step(hunt_key)` consumes the key inside (verbatim transcription of today's `update_predators`), so the 2-then-1 internal sub-split chain stays byte-identical. Edge case `N_pred == 0` correctly handled by host-side `if len(params.hunt_idx) > 0` (passive-predator and olfaction_parity_neutral.yaml both safe). The "different N triggers recompile" property is preserved and positively asserted in CP5 Part 2. |
| **B2** — `sense_extero_nociception` missing from File Changes | **PASS** | §"Sensor refactor → `sense_extero_nociception`" L399–415; File Changes L627; CP4 parity gate L753 ("Extero-noc parity gate") | ✅ | Diff is correct: `state.animal_pos` minus `agent_pos`, masked by `params.animal_is_damaging`. Sum yields byte-identical scalar to today because `animal_is_damaging` is True exactly where `animal_classes == 'predator'`. CP4 explicitly pins the extero-noc channel in `visual_parity_ref.npz`. |
| **B3** — `dreamer_srl_main.py:522-523` reads `predator_tags`/`neutral_tags` | **PASS** | §"Legacy aliases" L189–208; EnvParams table L171–172; File Changes L700–704 | ✅ | `@property` accessors return host-side Python tuples filtered by class. `tuple(env_params.neutral_tags)` (today's call) works transparently because the property returns a `tuple[str, ...]`. **Caveat (see §"Minor concerns"):** the plan does not explicitly say "remove the existing `predator_tags: tuple = struct.field(pytree_node=False)` field declaration on lines 89, 110 of `state.py`" — this MUST happen, otherwise the `@property` decorator collides with the dataclass field. Similarly the `config_loader.py:484, 500` constructor arguments `predator_tags=...`/`neutral_tags=...` must be dropped. Flag for developer; not a blocker because both are obvious consequences of the alias-replacement design. |
| **B4** — `scripts/verify_noise.py` constructs EnvState with removed fields | **PASS** | File Changes L710–740; CP1 atomic-commit L750 | ✅ | Rewrite snippet is correct (`animal_pos`/`animal_state`/`animal_stamina`/`animal_move_timer`/`animal_attack_timer`/`animal_property_sampled` + the five `*_sampled` fields). Atomic commit covers loader change + script rewrite + 86-config migration in one boundary so no intermediate state has broken loads. |
| **B5** — `hit_neutral` pre→post-step semantic flip | **PASS** | §"Damage logic" L417–456; File Changes L612 | ✅ | Two separate masks: `at_animal`/`at_damaging` (from `new_animal_pos`, post-step → `damage_pred`, `hit_predator`, `attack_timer`) and `at_neutral_pre` (from `state.animal_pos`, pre-step → `hit_neutral` only). Byte-parity holds because the empty-`animal_pos` Python-level guard (today's `if state.neutral_pos.shape[0] > 0 else jnp.array(False)`) is not strictly needed: `jnp.any` over an empty axis returns `False` natively. Confirmed via a sanity check on `jnp.any(jnp.zeros((0,), dtype=bool))`. |

### New blockers — what to fix

#### N1 — `jax_reset` per_entity placement concat order changes resolution order

**Where**: plan L620 says "in `per_entity` mode, concatenate `res / animal / obs` spawn areas (was `res / pred / obs / neutral`)".

**The issue**: today's `core.py:701-708`:
```python
all_positions  = jnp.concatenate([res_pos, pred_pos, obs_pos, neutral_pos], axis=0)
all_spawn_areas = jnp.concatenate([res, pred, obs, neutral], axis=0)
all_positions  = resolve_overlaps_global(all_positions, all_spawn_areas, ...)
```
`resolve_overlaps_global` (`core.py:558-604`) processes entities sequentially via `lax.scan`, building an occupancy bitmap entity-by-entity. The processing order determines tie-breaking when two entities collide in the same cell. Today's order: `res → pred → obs → neutral`. The plan's new order: `res → animal(pred+neutral) → obs`. Obstacle resolution now happens AFTER neutrals; today it happens BEFORE neutrals (and AFTER predators). For any of the 86 configs where an `obs_spawn_area` and a `neutral_spawn_area` overlap and the unresolved positions land in the same cell, the byte-parity gate fails because `resolve_overlaps_global` returns a different position.

**Fix (minimum-diff, preserves byte-parity)**: keep the concat order `[res, pred, obs, neutral]` even though the final stored array is `[res, animal=pred+neutral, obs]`. Concretely:
1. After per-type position sampling, build `pred_pos = positions_for[hunt_idx + static_idx_if_class_predator]` and `neutral_pos = positions_for[wander_idx + static_idx_if_class_neutral]` via host-side splits.
2. Concatenate `[res_pos, pred_pos, obs_pos, neutral_pos]` and resolve.
3. After resolution, slice back: `pred_pos_resolved = all_positions[num_res : num_res+N_pred]`, `neutral_pos_resolved = all_positions[num_res+N_pred+num_obs:]`, `obs_pos_resolved = all_positions[num_res+N_pred : num_res+N_pred+num_obs]`.
4. Then assemble `animal_pos = jnp.concatenate([pred_pos_resolved, neutral_pos_resolved])` for storage.

This needs the `pred_idx_in_animal` / `neutral_idx_in_animal` host-side index tuples (which already exist via `class_indices(params, 'predator')` / `class_indices(params, 'neutral')` in the v0.2 plan). Add them next to `hunt_idx`/`wander_idx`/`static_idx` on `EnvParams`.

#### N2 — `jax_reset` property-sampling key split goes from 4-way to N-way, breaking threefry stream

**Where**: plan L622 — "**Lines 776–784 (property sampling): collapse `pred_property_sampled` + `neutral_property_sampled` into one `animal_property_sampled`. Use a single `prop_key_animal`.**"

**The issue**: today's `core.py:775`:
```python
prop_key_res, prop_key_pred, prop_key_obs, prop_key_neutral = jax.random.split(property_key, 4)
pred_property_sampled    = normal(prop_key_pred,    params.pred_property.shape)
neutral_property_sampled = normal(prop_key_neutral, params.neutral_property.shape)
```
Each subkey is a 4-way split derivative. The plan implies a 3-way split: `prop_key_res, prop_key_animal, prop_key_obs = jax.random.split(property_key, 3)` and `normal(prop_key_animal, (N_animals, V))`.

Two breakages compound:
1. **Split arity matters.** `jax.random.split(key, 3)[1]` and `jax.random.split(key, 4)[1]` are different sub-keys. So `prop_key_animal ≠ prop_key_pred`, meaning **even the predator slice of `animal_property_sampled` does not byte-match today's `pred_property_sampled`**.
2. **Threefry prefix-stability is only on the *leading* axis of a single draw.** `normal(prop_key_animal, (N_animals, V))[:N_pred]` would byte-match `normal(prop_key_animal, (N_pred, V))` — but ONLY because they share the leading-axis prefix property of threefry. It does NOT match `normal(prop_key_pred, (N_pred, V))` because the keys differ.

**Fix (minimum-diff)**: keep the 4-way split exactly as today:
```python
prop_key_res, prop_key_pred, prop_key_obs, prop_key_neutral = jax.random.split(property_key, 4)
pred_prop    = _sample_property(prop_key_pred,    params.animal_property[predator_indices],    params.animal_property_std[predator_indices])
neutral_prop = _sample_property(prop_key_neutral, params.animal_property[neutral_indices],     params.animal_property_std[neutral_indices])
animal_property_sampled = jnp.zeros_like(params.animal_property)
animal_property_sampled = animal_property_sampled.at[jnp.asarray(predator_indices)].set(pred_prop)
animal_property_sampled = animal_property_sampled.at[jnp.asarray(neutral_indices)].set(neutral_prop)
```
This guarantees byte-parity for both the predator and neutral slices.

A simpler alternative — **only valid if predators are always at indices `0..N_pred-1` and neutrals at `N_pred..N`** (which is what the loader produces in v0.2):
```python
prop_key_res, prop_key_pred, prop_key_obs, prop_key_neutral = jax.random.split(property_key, 4)
pred_prop    = normal(prop_key_pred,    (N_pred, V))    # byte-matches today
neutral_prop = normal(prop_key_neutral, (N_neutral, V)) # byte-matches today
animal_property_sampled = jnp.concatenate([pred_prop, neutral_prop], axis=0)
```

**N1 + N2 together** strongly suggest revisiting the `jax_reset` patch sections of the plan and adding a Risks-item bullet titled "PRNG byte-parity in `jax_reset` requires per-type key splits to be preserved internally even when the stored arrays are unified" — symmetric to Risks item 1 for `jax_step`.

#### N3 — `placement_key` 6-way split likely needs preservation too (related to N1)

The same principle applies to the placement_key split at today's `core.py:680-681`:
```python
placement_key, res_key, pred_key, obs_key, neutral_key, resolve_key = jax.random.split(placement_key, 6)
```
If the plan changes this to a 5-way split (`[placement_key, res_key, animal_key, obs_key, resolve_key]`), `res_key` AND `obs_key` AND `resolve_key` all change because `jax.random.split(key, 5)` ≠ first 5 of `jax.random.split(key, 6)` — they're independent split arities, not prefix-related. This is **probably** the same root cause as N1.

**Fix**: keep the 6-way split, slice the animal initial-positions back out as in N1's fix.

The plan does NOT explicitly say how it intends to split `placement_key`. The pseudocode and File Changes section gloss over this. If the developer implements the obvious 5-way split, every legacy config will fail the CP1 parity gate (because `res_pos`, `obs_pos`, and the post-overlap positions all shift).

### Minor concerns (not blockers)

| ID | Concern | Where | Action |
|---|---|---|---|
| **M1** | Plan does not explicitly say "remove the existing `predator_tags: tuple = struct.field(pytree_node=False)` field at `state.py:89` and `neutral_tags` at `state.py:110`" — required for the `@property` (B3 fix) to take effect. | `state.py:89, 110` | Add to the "To be removed" list under §"`EnvState` / `EnvParams` field layout" (L187). |
| **M2** | Plan does not explicitly say "remove `predator_tags=predator_tags` and `neutral_tags=neutral_tags` from the `EnvParams(...)` constructor call at `config_loader.py:484, 500`". Implicit in "replace lines 228–366 with `_load_animals()`" but worth flagging. | `config_loader.py:484, 500` | Mention in File Changes → `config_loader.py`. |
| **M3** | Plan's `dist_to_pred` / `dist_to_neutral` refactor (File Changes L613) doesn't show the exact diff for the empty-N Python-level fallback (`if state.pred_pos.shape[0] > 0 else 99.0`). With unified `animal_pos`, the new guard is `if state.animal_pos.shape[0] > 0 else 99.0` plus a `jnp.where(animal_is_damaging, ...)` mask. If predators are zero but animals are nonzero, the masked `jnp.min` over an all-`99.0` array returns `99.0` — byte-parity holds. But the developer needs to be careful not to drop the fallback for the rare zero-animal config. | `core.py:495-498` | Add the exact diff to plan §"Damage logic" or as a callout in File Changes. |
| **M4** | The grep `grep -n "info\[" src/environment/core.py` for the §(h) test (L843) won't find `info['key'] = value` assignments outside `info = {...}` dict literal — there are several at L461, L487-492, L510-515, L525. Test (h) should grep `info\[` AND `info\s*=\s*{`. | Test Plan §(h) | Tighten the grep pattern in §(h). |
| **M5** | `agent_in_bush` info-dict key (`core.py:525`) wasn't in the original B5/C5 enumeration. It uses `state.obs_pos` — no change needed for animal-refactor — but the C5 sweep should cover it for completeness. | `core.py:525` | Add to test §(h). |
| **M6** | The plan documents that wander/static get unused per-episode draws via the 5-way uniform call (CP2 design at L335). This is fine, but for **zero-animal configs** the call `uniform(ek1, (0,), low=..., high=...)` produces a `(0,)` array — guard against empty-bound arrays (`params.animal_detect_low` of shape `(0,)`). Threefry handles zero-shape draws cleanly, so this should work; flagging because the plan doesn't explicitly test it. | `core.py:jax_reset` after placement | Add zero-animal smoke to CP2 test. |

### C-tier suggestions incorporation audit

| ID | Original concern | Incorporated? | Where |
|---|---|:---:|---|
| **C1** | Parity scope widened to 86 configs | ✅ | L750, L789–798 (coverage breakdown table) |
| **C2** | per_type pre-flight via `grep -rln "mode: per_type"` | ✅ | L750 (CP1 pre-flight), L752 (CP3 sweep + report) |
| **C3** | Regex recompile assertion `log.count("Compiling jax_step") == 1` | ✅ | L754 (CP5), L829 (Test Plan §(e) Part 1) |
| **C4** | Non-degenerate-range config + cross-field independence + state-vs-params + wander/static-zero-state | ✅ | L751, L807–810 (Pearson `|r| < 0.5`, hasattr assertion, 1000-step wander/static check) |
| **C5** | Info-dict legacy alias sweep | ✅ | L841–852 (§(h) `test_info_dict_aliases.py`); enumerates `hit_predator`, `hit_neutral`, `damage_predator`, `damage_obstacle`, `damage_hiding_predator`, `dist_per_predator`, `dist_per_neutral`. M5 above flags `agent_in_bush` as a still-missing legacy key. |

All C-tier suggestions are visibly incorporated and would survive code review. C5's sweep table is solid but should be widened by one entry (M5).

### Conventions audit checklist (v0.2)

| Convention | Status | Notes |
|---|:---:|---|
| Pytree immutability (`._replace`) | ✅ | v0.2 preserves. |
| JIT shape-stability — `hunt_idx`/`wander_idx`/`static_idx` are `pytree_node=False` | ✅ | L545–547 confirmed. |
| JIT shape-stability — same N + same class ordering = no recompile | ✅ | Positively asserted in CP5 Part 1 (C3). |
| JIT recompile boundary — same N + different class ordering = recompile | ✅ | Positively asserted in CP5 Part 2 (C-CFG-5). |
| vmap safety | ✅ | No vmap changes; entity-axis vmap pattern preserved. |
| PRNG threading — `jax_step` 6-key split byte-identical | ✅ | L80, L609 confirmed. |
| PRNG threading — `update_animals` per-subset draw shapes `(N_pred,)` / `(N_neutral,)` | ✅ | B1 fix; L78 + L246–321. |
| PRNG threading — `jax_reset` outer 5→6 split | ✅ | L618; prefix-stable, first 5 sub-keys byte-match. |
| **PRNG threading — `jax_reset` per_entity 6-way `placement_key` split preserved** | ❌ | **N3 blocker.** Plan doesn't address. |
| **PRNG threading — `jax_reset` 4-way `property_key` split preserved** | ❌ | **N2 blocker.** Plan explicitly collapses to "a single `prop_key_animal`" (L622). |
| **`resolve_overlaps_global` processing order preserved as `[res, pred, obs, neutral]`** | ❌ | **N1 blocker.** Plan reorders to `[res, animal, obs]` (L620). |
| Sensor — `sense_extero_nociception` covered | ✅ | B2 fix. |
| Configuration Protocol (`get_mandatory`) | ✅ | L584, L575. |
| Backward-compat loader covers 86 configs | ✅ | C1 fix; CP1 spec L750. |
| `predator_enabled` semantic safety | ✅ | Fully removed in CP1 atomic commit; L82. |

### Final verdict: REJECT

B1–B5 (the five blockers I originally raised) are all PASS — the v0.2 plan resolves them cleanly. However, while validating that resolution, I traced through `jax_reset` end-to-end and surfaced **three new blockers (N1, N2, N3) all in placement/property-sampling PRNG threading**. The CP1 byte-parity gate will detect them, but the plan as written does not give the developer the fix.

Recommendation: **fix N1+N2+N3 as a v0.3 micro-revision before the developer starts CP1.** The fix is mechanical (preserve today's per-type key splits internally, slice the unified array back into per-type buffers for the placement scan, then re-concat for storage). It is the same conceptual fix as B1 (the per-subset call pattern) but applied to `jax_reset`'s placement and property-sampling sections instead of `jax_step`'s `update_animals`. Add a Risks item titled "PRNG byte-parity in `jax_reset` requires per-type key splits to be preserved internally" symmetric to Risks item 1.

No re-review is required after the v0.3 fix — N1/N2/N3 are all variants of the same principle, and once the principle is documented in Risks the developer can apply it uniformly. Senior-developer can sign off on v0.3 directly.

Reviewed by: code-reviewer
