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
