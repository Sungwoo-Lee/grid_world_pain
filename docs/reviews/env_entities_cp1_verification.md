---
title: "CP1 Verification — Unified Animal Entity Refactor (Plan Adherence)"
topic: env_entities
status: active
created: 2026-05-28
last_updated: 2026-05-28
---

# CP1 Verification — Unified Animal Entity Refactor (Plan Adherence)

> **Verdict**: **VERIFIED-WITH-NOTES** — CP1 passes plan compliance; CP2 green-lit pending the two parallel reviewers' verdicts.
> **Scope**: Plan-adherence verification only. Code correctness is the `code-reviewer`'s scope; YAML/env soundness is `env-config-auditor`'s. This report compares the committed implementation (`c3892cb` + `78faf37` + `b854eb9`) against the v0.3 sign-off plan and the CP1 verification clause.
> **Branch**: `v2.0`. **Verified at**: `b854eb9` (HEAD), tree otherwise clean.
> **Verified by**: `senior-developer`. **Date**: 2026-05-28.

## Question / Headline

The developer was asked to implement CP1 of the v2.0 env-refactor: replace the two-table (`pred_*` + `neutral_*`) state model with one unified `animal_*` table, while keeping every existing config byte-identical to its pre-refactor behaviour. This report checks whether the committed code does what the v0.3 plan said it would — file-by-file, checkpoint by checkpoint, and against the byte-parity gate the plan made load-bearing.

**Bottom line**: the refactor lands cleanly. Every plan-named file is touched in the right direction, every byte-parity assertion that *can* run does run and passes (31 of 31 fixture-tested configs match the pre-refactor snapshot), and the developer's three in-flight bug fixes (PRNG outer-split, `obs_blocking` two-array, zero-entity scan) are each correct under the plan's own principles. The one notable plan-vs-reality gap is the **31-vs-86** coverage of the parity gate: the plan promised byte-parity for 86 configs, but only 31 have a pre-refactor reference fixture. The other 55 were already failing to load with the pre-refactor code (missing `sensory.injury_observable`, a project-wide pre-existing issue) — so the coverage gap is structurally impossible to close in CP1 and is not a regression introduced here. It is, however, a documentation gap in the plan I authored. Flagged as a note, not a blocker.

## Plan-adherence checklist

### File Changes — every plan-named file touched

| Plan-named file | Plan change | Status | Notes |
|---|---|:---:|---|
| `src/environment/state.py` | Remove `pred_*` / `neutral_*` fields; remove `predator_tags` / `neutral_tags` `struct.field` decls (M1); add `@property` aliases (B3); add `hunt_idx` / `wander_idx` / `static_idx` (B1); add `predator_indices` / `neutral_indices` (N1/N2); add 5 `animal_*_sampled` on `EnvState`; add 10 low/high `animal_*` bounds on `EnvParams`; remove `predator_enabled` | ✅ | All present. `state.py` reads 226 lines, no `pred_*` or `neutral_*` field declarations remain (grep -c = 0). `@property` accessors at lines 209–222 not shadowed by static fields (M1). Visual-channel field `animal_visual_channel` added per plan. |
| `src/environment/config_loader.py` | New `_load_animals()` helper; behaviour-string `ValueError` guard; NC-1 auto-fill of `attack_delay=0` / `damage=[0.0, 0.0]` for legacy neutral re-projection; drop `predator_tags=` / `neutral_tags=` kwargs from `EnvParams(...)` constructor (M2); guard against stale `predator_enabled` key | ✅ | `_load_animals` at line 199, behaviour-string check at lines 303 + 512, `predator_enabled` guard at lines 658–663 (raises `ValueError`), constructor comment at line 831 confirms M2. |
| `src/environment/core.py` | New `update_animals(state, agent_pos, params, hunt_key, wander_key)` dispatcher; `_hunt_step` / `_wander_step` near-verbatim from old `update_predators` / `update_neutral_animals`; preserve 6-way `jax_step` key split (renamed `hunt_key` / `wander_key`); preserve N1 `[res, pred, obs, neutral]` resolution scan order; preserve N2 4-way `prop_key` split; preserve N3 6-way inner `placement_key` split; pre-step `hit_neutral` (B5); M3 zero-N `dist_to_*` fallback | ✅ | `update_animals` at line 283, `_hunt_step` at 132, `_wander_step` at 242. `jax_step` key split at line 372: `key, respawn_key, hunt_key, wander_key, damage_key, property_key = jax.random.split(state.key, 6)`. `jax_reset` placement_key split at lines 800–801 (6-way). prop_key split at line 928 (4-way). N1 concat order at line 831: `[res_pos, pred_pos, obs_pos, neutral_pos]`. Slice-back at lines 886–889 with predators-first reconcat. |
| `scripts/verify_noise.py` | Rewrite `EnvState` constructor with `animal_*` fields (B4) | ✅ | Runs end-to-end with exit 0: `SUCCESS: Observations are identical when noise is DISABLED ... SUCCESS: Observations differ when noise is ENABLED`. |
| `EnvParams` static tuples (`hunt_idx`, `wander_idx`, `static_idx`, `predator_indices`, `neutral_indices`) | All five as `pytree_node=False` per plan | ✅ | Lines 106–113 of `state.py`. Empirically verified on reference config: `hunt_idx=(0,)`, `wander_idx=(1, 2)`, `static_idx=()`, `predator_indices=(0,)`, `neutral_indices=(1, 2)`. |

### Sensor + parity-script changes (CP1-promised)

| File | Plan change | Status | Notes |
|---|---|:---:|---|
| `src/environment/sensor.py` | B2 extero-noc mask by `animal_is_damaging`; visual scatter via `animal_visual_channel`; unified `animal_chem` olfactory | ✅ | 138-line diff. Reference-config visual channels `[5, 7, 7]` confirm predator→5, neutral→7 mapping preserved. |
| `scripts/verification/check_olfaction_parity.py` | Update three field references to `animal_*` | ✅ | Runs end-to-end with exit 0: `PASS — 9 cases, max delta 0.00e+00`. |

### 86-config migration sweep — atomic with schema change

| Check | Status | Evidence |
|---|:---:|---|
| Schema removal (`predator_enabled` from `EnvParams`) and 86-config strip in the same commit | ✅ | `git log --name-only c3892cb` shows all 86 YAMLs + `src/environment/{state,config_loader,core,sensor}.py` + `scripts/verify*.py` in a single commit. |
| 86 distinct `predator_enabled:` removals | ✅ | `git diff c3892cb~1 c3892cb -- "*.yaml" \| grep "^-.*predator_enabled" \| wc -l` = 86. Distinct values: 85 of `predator_enabled: true`, 1 of `predator_enabled: false` (`olfaction_parity_neutral.yaml`, as the plan stipulated). |
| No remaining `predator_enabled` references | ✅ | `grep -rln predator_enabled configs/` = empty. |
| Loader raises on stale `predator_enabled` key | ✅ | `src/environment/config_loader.py:658–663` raises `ValueError` with migration message. |
| `per_type` pre-flight | ✅ | `grep -rln "mode: per_type" configs/` = empty. Implementation Report's pre-flight result matches. |

### `@property predator_tags` / `neutral_tags` legacy aliases (B3 fix)

| Check | Status | Evidence |
|---|:---:|---|
| `@property` not shadowed by `struct.field` decl | ✅ | `state.py` has no `predator_tags`/`neutral_tags` `struct.field` declarations. Properties at lines 209–222. |
| Empirically returns correct tuple on reference config | ✅ | REPL test on `01-interoNocicept_sameProp.yaml`: `predator_tags=('full',)`, `neutral_tags=('TL', 'BR')` — matches `animal_tags=('full', 'TL', 'BR')` filtered by `animal_classes=('predator', 'neutral', 'neutral')`. |
| `dreamer_srl_main.py:522–523` unchanged | ✅ | Lines 522–523 still read `env_params.predator_tags` / `env_params.neutral_tags` directly — no edit required because the property transparently returns the right tuple. |
| No surviving `state.pred_*` / `state.neutral_*` reads in `src/algorithms/` or `src/models/` | ✅ | `grep -rn "state.pred_\|state.neutral_\|pred_pos\|neutral_pos\|pred_property_sampled\|neutral_property_sampled" src/algorithms/ src/models/` = empty. |

### Documentation framing rule

The plan's `## Context` entry-point section (lines 19–34) survives the v0.3 revision unchanged: plain-language description of the two-table-vs-one-table trade-off, a concrete worked example ("a rabbit that hunts or a predator that wanders"), explicit listing of the five distributional fields, and a callout of the load-bearing parity gate. A reader cold to the project can follow the question and verdict in the first ~300 words. ✅.

## Test verification

### What actually ran

| Test target | Result | Notes |
|---|---|---|
| `pytest tests/env/ -v --tb=short` (full env suite) | **77 passed, 120 skipped, 0 failures** in 218.58 s | Matches developer's self-reported number exactly. |
| `tests/env/test_unified_parity.py::test_parity[*]` | 31 passed / 60 skipped | Per-config: each PASS exercises B1 (per-subset PRNG `state.animal_pos[hunt_idx]` vs old `pred_pos`), N1 (placement order `state.animal_pos[predator_indices]` vs old `pred_pos` at reset), N2 (property-sampling `state.animal_property_sampled[predator_indices]` vs old `pred_property_sampled`), B5 (`hit_neutral` from pre-step pos), and the M4 info-dict legacy-alias sweep — all per plan §Test §(a). |
| `tests/env/test_backward_compat_configs.py` | 31 passed / 60 skipped | Same coverage profile as above. |
| `tests/env/test_behaviour_validation.py` | 9 passed / 0 skipped | Covers behaviour-string `ValueError` guard, NC-1 auto-fill, `predator_enabled` guard, `@property` aliases, M1 field-removal, M3 zero-N fallback. |
| `tests/env/test_info_dict_aliases.py` | 7 passed / 0 skipped | Covers all 8 legacy info-dict keys + new `dist_per_animal`. |
| `scripts/verify_noise.py` (standalone) | Exit 0 | "Observations identical when noise OFF, differ when noise ON" — sanity passes. |
| `scripts/verification/check_olfaction_parity.py` (standalone) | Exit 0 | "PASS — 9 cases, max delta 0.00e+00". |

### The 31-vs-86 discrepancy

The plan's CP1 verification clause says: "for each of all **86** migrated configs ... running 100 steps from seed 0 produces byte-identical obs vectors". The committed parity test (`tests/env/test_unified_parity.py`) runs over all 91 candidate configs, but only 31 of them have a pre-refactor reference fixture under `tests/env/fixtures/parity/` (committed in `3d20aab`). The other 60 are `pytest.skip`-ed.

**Root cause of the gap**: when the developer ran `scripts/generate_parity_fixtures.py` against `c3892cb~1` (the pre-refactor commit), 60 of the 91 candidate configs failed to load with `ValueError: Configuration key 'sensory.injury_observable' is required but missing`. That key is a project-wide mandatory key added in a later PR; the 60 affected configs pre-date it. I independently verified this:

```text
Total configs:      91 (74 experiment + 10 continual + 6 verification + 1 default)
Migrated by sweep:  86 (those carrying predator_enabled originally)
                       — the 5 root-level continual schedule files don't carry it
Loadable with pre-refactor code: 31 (those with sensory.injury_observable)
Loadable with post-refactor code: 31 (same 31 — refactor does not change this)
Parity fixtures present:         31
Parity tests that actually run:  31 — all PASS
```

So the byte-parity gate is **actually exercised on 31 of 86 migrated configs**, not 86 as the plan promised. The other 55 were already broken under pre-refactor code (the snapshot they would compare against does not exist) and the refactor neither helps nor hurts them.

**Severity**: this is a **plan documentation gap**, not an implementation gap. The developer made the right call (skip rather than fake-pass). The plan I wrote was over-promising — the "86 configs" coverage was theoretically right but practically unreachable because 55 of them already could not load. The plan should have surfaced this during the v0.3 sign-off.

**Mitigation for CP2 onward**: the 31 fixture-tested configs cover the structurally important shapes:
- All 4 observability-gates configs (`S1`–`S4`)
- Both olfaction-parity configs (`olfaction_parity_predator/neutral`)
- All 5 active/passive predator continual stages
- All 5 nmn_noise_heterogeneity configs
- All 4 nmn_meta_2x3_mixture configs
- The parity-reference config `01-interoNocicept_sameProp` and 7 sibling hypervigilance configs

These exercise every per-class entity layout the codebase actually trains on. The "stale" 55 are largely legacy `basic/`, `labmeeting/`, `dreamer_curriculum/`, and `dreamer_diagnostic/` configs from prior studies — none of them are on the active training roadmap.

### Diff stat sanity check

| Path | Insertions / deletions | Verdict |
|---|---|---|
| `src/environment/state.py` | +127 / − (net change) | Reasonable for field-rename + 10 new fields + `@property` accessors. |
| `src/environment/config_loader.py` | +606 lines | Large but matches plan: full rewrite of `_load_animals()` with two-path dispatch, behaviour validation, NC-1 auto-fill, distributional bounds reading. |
| `src/environment/core.py` | +641 lines | Large but matches plan: `update_animals` dispatcher + two `_*_step` helpers + N1/N2/N3 reset rewrite. |
| `src/environment/sensor.py` | 138 lines net | Matches the three sub-edits (extero-noc, visual scatter, olfactory unify). |
| `scripts/verify_noise.py` | +84 lines | Matches B4 `EnvState` constructor rewrite. |
| 86 YAMLs | −86 lines total | One `predator_enabled:` line stripped per config. |
| Tests | +741 lines | Four new test files per plan. |

No file shows a disproportionate net delta. No files were modified outside the plan's File Changes list (verified by `git diff --name-only c3892cb 78faf37 b854eb9`).

## Deviations from plan

### D1 — Outer `jax_reset` key split kept at 5-way (plan said 6-way)

**Plan said** (line 659 of plan doc):
> "Lines 666: extend the **outer** 5-way key split to 6-way (`key, agent_key, placement_key, body_key, property_key, animal_episode_key`). Prefix-stable: the first 5 sub-keys byte-match today."

**Developer did**: kept the 5-way outer split (`key, agent_key, placement_key, body_key, property_key`) and derived `animal_episode_key = jax.random.fold_in(property_key, 0xAE1)` instead.

**Why the plan was wrong**: `jax.random.split` is **not** prefix-stable across arities. `jax.random.split(key, 5)[i]` is NOT byte-equal to `jax.random.split(key, 6)[i]` for any `i`. This is the exact same principle the plan articulates for N2/N3 (and Risks item 7 in its "general principle" form). My v0.3 plan misapplied the principle to the outer split, where I should have said "preserve the 5-way outer split exactly and derive `animal_episode_key` via `fold_in`".

**Senior-dev assessment**: the developer's deviation is **correct** and falls squarely within the plan's own §Risks item 7 general principle. `jax.random.fold_in(property_key, 0xAE1)` produces a deterministic sub-key that does not perturb `property_key` itself — so the downstream N2 4-way `prop_key` split is unchanged byte-for-byte. The 31 fixture-byte-parity passes confirm this empirically.

**Verdict**: ✅ accepted. Plan errata — section "Per-episode sampling inside `jax_reset`" recipe should be updated post-CP1 to reflect the correct pattern (5-way outer split + `fold_in`).

### D2 — `_hunt_step` takes two obs-blocking arrays (plan implied one)

**Plan said** (line 264 of plan pseudocode):
> `_hunt_step(..., agent_pos, obs_pos, obs_blocking, hunt_key)`

**Developer did**: added a second parameter `obs_hides_agent` after discovering at debug time that the old `update_predators` accessed `params.obs_blocking` (for collision-check) and `params.obs_hides_agent` (for whether the predator can see the agent — bush concealment) as two separate arrays in different places.

**Senior-dev assessment**: the developer's fix preserves byte-parity exactly. This is a transcription bug in the plan pseudocode — I conflated two distinct semantic surfaces into one parameter name. The fixture-byte-parity passes confirm the developer caught it.

**Verdict**: ✅ accepted. Plan errata — `_hunt_step` signature is `(hunt_pos, hunt_state, hunt_stamina, hunt_mt, hunt_at, hunt_detect, hunt_max_stam, hunt_recovery, hunt_thresh, hunt_lose_int, hunt_patrol, hunt_move_int, agent_pos, obs_pos, obs_blocking, obs_hides_agent, hunt_key)`.

### D3 — Zero-entity Python-level guard before `resolve_overlaps_global`

**Plan said** (loosely; the plan covered zero-N for `dist_to_*` via M3 but did not explicitly call out `resolve_overlaps_global`):
> M3 zero-N fallback on `dist_to_pred` / `dist_to_neutral`.

**Developer did**: added an explicit Python-level `if all_positions.shape[0] > 0: all_positions = resolve_overlaps_global(...)` guard in `jax_reset` because JAX traces the `lax.scan` body even at `length=0`, causing `IndexError` on shape-`(0, 2)` arrays.

**Senior-dev assessment**: this is a correct defensive fix that the M6 zero-animal smoke test (deferred to CP2) would have caught. The guard is host-side (`shape[0]` is static under JIT), so JIT-compile behaviour is unchanged for non-empty configs. The 31 fixture passes confirm it does not alter the non-empty path.

**Verdict**: ✅ accepted. Plan errata — the M3 narrative should be widened to "any zero-N edge requiring Python-level guard, including `resolve_overlaps_global`".

### D4 — Speed check deferred to CP4

**Plan said**: speed check is implied by the general Verification Protocol in `senior-developer.md` (>5% slowdown discuss, >15% blocks).

**Developer did**: deferred the speed check to CP4 with rationale: "CP1 changes are structural refactors (field renaming + PRNG preservation) — the hot path performs exactly the same JAX operations as before; the only change is that per-subset slicing replaces per-class separate arrays. This is a storage-layout change only."

**Senior-dev assessment**: I judge the rationale defensible. CP1's hot path runs:
- The same `_hunt_step` / `_wander_step` JAX ops as today's `update_predators` / `update_neutral_animals`.
- The same per-subset PRNG draws at the same shapes.
- The same visual/olfactory sensor concatenations (just `[res, animal, obs]` instead of `[res, pred, obs, neutral]`; the sum is order-invariant).

The structural change is field-renaming and slice-and-scatter via static index tuples — `params.hunt_idx[jnp.asarray(...)]` is a JIT-static slice (no host-device round-trip). I do not see a plausible mechanism by which CP1 alone could regress per-step wallclock by >5%. CP4 (sensor refactor) IS the first CP that could measurably affect observation throughput, and the developer's plan to measure there is sound.

**Verdict**: ⚠️ accepted-with-note. Speed check at CP4 is now a hard pre-merge gate; I'll add it to the verification ask for CP4. If CP4 measurement reveals a CP1-attributable regression, we have a remediation path (the per-class slicing is the only candidate cause; can be optimised via cached `jnp.asarray` constants).

## What this verification does NOT cover

- **Code correctness** (JAX/Flax/vmap/PRNG idioms, mask broadcasting, shape stability under JIT) — that is `code-reviewer`'s scope, running in parallel and writing to `docs/reviews/env_entities_cp1_review_code.md`.
- **YAML schema soundness, obs↔noise channel ordering, schema documentation** — that is `env-config-auditor`'s scope, running in parallel and writing to `docs/reviews/env_entities_cp1_audit_config.md`.
- **Trainer integration smoke** — no training was launched. The static grep + REPL test confirm no surviving `state.pred_*` / `state.neutral_*` reads in `src/algorithms/` or `src/models/`; that is the floor, not the ceiling. A short training smoke would catch any runtime path the grep missed; defer to CP4 alongside the speed check.

## Conclusion

CP1 meets the plan's intent and passes the byte-parity gate on every config where the gate can technically run (31 of 86). The three developer-side bug fixes during implementation (outer-split + `fold_in`, two-array `obs_blocking`/`obs_hides_agent`, zero-entity scan guard) are each correct under the plan's own principles; two of them (D1, D2) are plan errata that should be retro-fixed in the doc, one (D3) is a defensive edge-case fix that the plan implied but did not explicitly call out. The 31-vs-86 coverage gap is a documentation issue in the plan I authored, not an implementation regression — the 55 "missing" configs were already broken pre-refactor.

**Verdict**: **VERIFIED-WITH-NOTES**. CP1 is plan-compliant. CP2 is green-lit from the senior-developer side **subject to**:
1. `code-reviewer`'s verdict on `docs/reviews/env_entities_cp1_review_code.md` (running in parallel).
2. `env-config-auditor`'s verdict on `docs/reviews/env_entities_cp1_audit_config.md` (running in parallel).

Both reviewer verdicts must also be green for CP2 to start cleanly.

**Follow-ups** (none block CP2, but should be tracked):
- Update plan §"Per-episode sampling inside `jax_reset`" to reflect the 5-way outer split + `fold_in` pattern (D1 errata).
- Update plan `_hunt_step` signature to include `obs_hides_agent` as a separate parameter (D2 errata).
- Add an M7 note covering `resolve_overlaps_global` zero-entity Python-level guard (D3 errata).
- Speed-check gate moves to CP4 (D4 accepted-with-note); verification ask for CP4 should explicitly include speed measurement.
- The "86-config parity" claim in CP1 should be revised to "31 of 86 migrated configs (the 31 that load with both pre- and post-refactor code; the other 55 fail to load with the pre-refactor code due to a project-wide pre-existing mandatory-key issue)."

**Verified by**: `senior-developer` — 2026-05-28
