---
title: "Config audit — configurable per-entity visual properties (v3.0 post-impl)"
topic: sensors
status: complete
created: 2026-06-18
last_updated: 2026-06-18
aliases: [config-visual-properties-audit]
---

# Config Audit — Configurable Per-Entity Visual Properties (v3.0 post-impl)

**Scope:** post-implementation pre-flight — single-feature config soundness check
**Commit audited:** `ddff125` (feat: configurable per-entity visual properties)
**Plan audited:** `docs/develop/active/sensors/CONFIGURABLE_VISUAL_PROPERTIES_PLAN.md`
**Files audited:** `configs/environment/default.yaml`, `configs/environment/experiment/basic/*.yaml` (5 configs), `configs/environment/experiment/archive/**` (sample, 18 loadable), `src/environment/config_loader.py`, `src/environment/sensor.py`, `src/environment/state.py`, `tests/env/test_visual_parity.py`, `tests/env/test_visual_properties.py`
**Audited by:** env-config-auditor
**Date:** 2026-06-18

---

## Summary

This refactor lets each entity in the environment carry a configurable "appearance vector" for
the visual sensor — the same way olfactory smell already works. Previously, every entity's
appearance was hard-wired as a fixed channel number in code (predator always occupied slot 5,
food slot 3, rock slot 6, etc.), making it impossible to change an entity's look without editing
code and breaking all saved checkpoints. The refactor replaces those four hard-coded constructions
with per-entity vectors stored on `EnvParams`, with defaults that exactly reproduce the old
hard-coded values at the default vector width of 8. A custom width (for example, 5 or 4 slots)
is opt-in and requires explicit appearance vectors on every entity.

The audit finds the implementation **sound on all six audit dimensions**. The observation-to-noise
sync is correct and auto-resizing — there is no second hardcoded width that needs to be kept in
sync. Zero-edit parity for all 6 modern live configs and all 18 loadable archive configs is
confirmed by both live Python instantiation and the byte-parity test suite (15 tests, 0 failures).
The one permitted config-key default (`sensory.visual_vector_size` falls back to 8 when absent) is
the correct and documented exception, enabling backward compatibility for ~86 archived configs.

**Verdict: PASS. Safe to launch.**

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| nit | `src/environment/sensor.py:409` | The `Olfaction` spectrum overlay in `build_sensory_viz` emits a hardcoded 8-label list (`['GRS','SND','PLN','FOD','DNG','PRD','RCK','NEU']`) for a sensor whose actual width is 5. Labels are display-only and do not affect observation bytes or training, but the mismatch is pre-existing and orthogonal to this refactor. | Mirror the visual-sensor pattern at L444-446: use the 8-label list when `params.olfactory_vector_size == 8` (which is never, in practice), else emit index strings `[str(i) for i in range(dim)]`. Out of scope for this refactor; flag for future cleanup. |
| note | `src/environment/config_loader.py:322` | `_load_animals(config, visual_vector_size: int = 8)` has a Python-level default `= 8`. This is sound because the only call site (`config_loader.py:856`) always passes the resolved value explicitly. The default exists as a safety net for direct calls in tests. Not a blocker. | No change needed. Document the pattern in the docstring if desired. |

No red or yellow findings.

---

## Checklist

### 1. Observation-to-Noise Modality Consistency

**PASS.**

The noise-application function (`apply_perceptual_noise`, `sensor.py:241`) builds the noise
vector by iterating `get_observation_breakdown(params)` and looking up each sensor name in
`noise_modality_order`. For the `Visual` modality it does `jnp.full((dim,), sigma)` where `dim =
breakdown["Visual"] = num_vis_cells * params.visual_vector_size`. There is no second place that
encodes the visual block width — the noise block follows `breakdown["Visual"]` exactly,
regardless of V.

The `_YAML_KEY_TO_SENSOR_NAME` map (`config_loader.py:1204`) maps YAML key `"visual"` to sensor
name `"Visual"`, which matches the key emitted by `get_observation_breakdown`. The `default.yaml`
noise-modalities block includes `visual:` at YAML-declaration index 8.

Live verification at V=8 (default, 6 configs) and V=5 (synthetic config): both produce
`breakdown["Visual"]` = `num_vis_cells * V`, and `clean_obs.shape == noised_obs.shape`. The
test `test_obs_noise_width_sync_v4` (CP5) explicitly verifies this at V=4.

**No index desync. Noise block auto-resizes with visual_vector_size.**

### 1.5. Behavior-Measures Bush Presence

**N/A.** This audit covers visual sensor configuration, not behavior-measure scope expansion.
`behavior_measures.enabled: true` is set in `default.yaml` and M2 is in scope in the
hypervigilance experiment track, but no config change in this refactor affects `obstacles:` or
`hides_agent:`. Existing bush presence in the relevant configs is unchanged.

### 2. Mandatory-Key Discipline

**PASS.**

The one new config-level key, `sensory.visual_vector_size`, is read with a documented read-site
fallback of 8 (`config_loader.py:765-766`):

```python
_vis_v = config.get('sensory.visual_vector_size')
visual_vector_size: int = int(_vis_v) if _vis_v is not None else 8
```

This is the single permitted fallback in this plan (documented in the plan §D1) and it exists
specifically to preserve byte-identical behaviour for ~86 archived configs that do not declare the
key. It is not a "critical param" in the sense that its absence causes a silent wrong-value
failure — at V=8 the behaviour is identical to the pre-refactor hard-coded path. The plan
explicitly documents this as the only exception to the no-defaults rule.

The `default.yaml` explicitly declares `sensory.visual_vector_size: 8`, so the modern config
surface carries the key explicitly. Only pre-modern archived configs rely on the fallback.

The new optional YAML key `sensory.visual_background_properties` (the 3×V background table) is
handled correctly: required when V≠8 (raises `ValueError` if missing), ignored when V=8 (defaults
to `eye(8)[:3]`). This is not a silent fallback for a critical computation — at V=8 the default
background vectors are byte-identical to the pre-refactor `one_hot(0/1/2, 8)` constructions.

All other new `EnvParams` fields (`res_visual_property`, `animal_visual_property`,
`obs_visual_property`, `visual_vector_size`, `visual_background_property`) are threaded to
the `EnvParams(...)` constructor at `config_loader.py:1086, 1116, 1130, 1184, 1185` — all
five confirmed present.

The internal helper `_load_animals` has `visual_vector_size: int = 8` as a Python default
parameter, but is always called explicitly with the resolved value (`config_loader.py:856`).
This is not a mandatory-key violation — it is a Python function default for a private helper.

### 3. Static-Field and JIT Recompile Risk

**PASS — recompile risk correctly managed.**

`visual_vector_size` is declared `struct.field(pytree_node=False)` (`state.py:223`), making it
static and shape-determining. This means any config that changes `visual_vector_size` from 8 to
another value will trigger an XLA recompile — which is the correct and intended behaviour, since
V determines the total observation width.

No sweep config currently varies `visual_vector_size` across runs. A grep of
`configs/environment/` confirms `visual_vector_size` appears in exactly one file (`default.yaml`,
fixed at 8). No recompile risk for any existing experiment.

**Future risk (flag for operators):** if an experiment sweep varies `visual_vector_size` across
parallel runs, each distinct V will force a separate JIT compilation and the configs will not be
checkpoint-compatible with each other. This is documented in the plan §Analysis and should be
called out in any future sweep design that uses V≠8.

### 4. Known Latent-Bug Recurrences

**PASS — no new trigger.**

- `body.overeating_death: false` in default.yaml. Not triggered.
- `random_start_pos: false` in default.yaml. No deterministic-evaluation risk.
- `property` (singular) YAML key: the new feature uses `visual_properties` (plural) consistently
  throughout `config_loader.py` (L296, L668, L810, L895). No legacy singular key exists for
  the visual channel. The olfactory `_read_properties` deprecation warning path is untouched.
- Per-entity olfactory `properties` key: the new `_read_visual_properties` helper is separate from
  `_read_properties` — it does not modify olfactory loading. All entity definitions still require
  `properties` for olfaction; this audit found no entity in any live config missing that key.
- NEU/RCK label swap (the bug being fixed here): both label occurrences at `sensor.py:409` and
  `sensor.py:445` now correctly emit `['GRS','SND','PLN','FOD','DNG','PRD','RCK','NEU']`
  (rock=6, neutral=7). Confirmed by grep. Label fix is display-only; numeric encoding unchanged.

### 5. Schema Padding and Modality-Count

**PASS.**

`_parse_noise_config` pads noise arrays to 13 (`config_loader.py:1228`). The default YAML has
10 modalities (indices 0-9). This refactor does not introduce a new noise modality — `visual`
was already present at index 8 in the pre-existing 10-modality schema. The V-dependent block
width is handled inside `apply_perceptual_noise` dynamically, not by altering the noise array
count. The 13-slot static padding is not affected.

### 6. Cross-Config Coherence

**N/A — this is a single-feature audit, not a sweep audit.** No multi-config sweep over
visual_vector_size or visual_properties exists.

---

## Live Load Verification

All 6 modern configs loaded cleanly via live Python instantiation:

```
configs/environment/default.yaml:                  V=8, visual_dim=8, total_obs=27
configs/environment/experiment/basic/00-forage_5x5.yaml:  V=8, visual_dim=8, total_obs=27
configs/environment/experiment/basic/01-slowPred_5x5.yaml: V=8, visual_dim=8, total_obs=27
configs/environment/experiment/basic/02-fastPred_8x8.yaml: V=8, visual_dim=8, total_obs=27
configs/environment/experiment/basic/03-multiPred_10x10.yaml: V=8, visual_dim=8, total_obs=27
configs/environment/experiment/basic/04-keenPred_10x10.yaml: V=8, visual_dim=8, total_obs=27
```

All produce `noise_modality_order` = `('Injury', 'Nutrition', 'Satiation', 'Interoceptive
Nociception', 'Extero Nociception', 'Olfaction', 'Collision', 'Proprioception', 'Visual',
'Location')` and `visual_background_property.shape = (3, 8)`.

Archive spot-check: 18 archive configs loaded (all returning V=8, visual_dim=8). The 12 that
failed did so with `ValueError: ... 'sensory.injury_observable' is required but missing` —
a pre-existing failure unrelated to this refactor (those configs predate the mandatory
`injury_observable` key added in an earlier release). No new failures introduced by this refactor.

V=5 synthetic config: loaded, `breakdown["Visual"]=5`, noise sync confirmed.

## Test Gate

```
pytest tests/env/test_visual_parity.py tests/env/test_visual_properties.py -q
15 passed in 114s
```

Full suite: `pytest tests/env/ -q` → 157 passed, 167 skipped, 0 failed.

---

## Conclusion

**Safe to launch.** All 6 audit checks pass. The refactor is additive and gated behind a
read-site default that maintains byte-identical observations for all ~86 existing configs. The
observation-to-noise sync is architecturally sound (single derivation point, no hardcoded widths
in the noise path). The one nit (olfactory spectrum overlay label list) is pre-existing and
outside this refactor's scope.

Audited by: env-config-auditor
