# Config Audit — Hunger-Gated Step 1 Discrimination-Onset Sweep (Pre-Flight)

**Scope:** Multi-config sweep (10 v3.0 sparse configs)
**Files audited:** `configs/environment/experiment/hunger_gated/01-s0_sig0.yaml` … `10-s0.5_sig0.4.yaml`
**Audited by:** env-config-auditor
**Date:** 2026-06-19

---

## Purpose

This audit guards a 10-run sweep that asks: "How far apart do a predator's and rabbit's
smell signals have to be before an agent starts treating them differently — and does its
hunger level shift that threshold?" Each config places one lethal predator and one harmless
rabbit in an identical 10-by-10 grid world and varies only the olfactory separation between
the two animals (wider separation = easier to tell apart) and the per-episode smell wobble
(larger wobble = harder to tell apart regardless of separation). The sweep covers 10 points
on a 2-D grid of (separation, wobble). Configs 01–05 have no wobble (σ = 0); configs 06–10
add wobble (σ = 0.2 or 0.4). At separation = 0 both animals smell identical; at separation
= 0.5 their smells are completely complementary (one all on channel 2, the other all on
channel 3). The design document lives at
`docs/experiments/active/hypervigilance/20260619_hunger_gated_step1_discrimination_onset.md`.
The base scene being reproduced is the cell-08 single-predator-vs-rabbit contrast (archived
at `configs/environment/experiment/archive/hypervigilance/08-singlePredRabbit_disengage.yaml`).

---

## Summary

**PASS.** All 10 configs load cleanly through the full `load_env_config` + `load_env_params`
pipeline with no errors. The merged `EnvParams` objects produce a consistent 27-dimension
observation vector, a correct 10-modality noise order (padded to 13 for JIT stability), and
the intended predator-vs-rabbit scene (one lethal predator `damage=[5,120]`, one harmless
rabbit `damage=[0,0]`). Olfactory arithmetic is verified numerically for all 10 (s, σ) grid
points. Sweep coherence is confirmed: the only non-comment differences across the 10 files
are the animal `properties` and `properties_std` lines — every other field (body, sensors,
noise, behavior_measures, chase params, lethality, obstacle layout) is byte-identical. Two
informational notes are recorded below, neither of which blocks launch.

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| NOTE | `08-s0.5_sig0.2.yaml` and `10-s0.5_sig0.4.yaml` — predator/rabbit `properties_std` | At `s=0.5` the mean channel values are `[0, 1, 0, 0, 0]` (predator) and `[0, 0, 1, 0, 0]` (rabbit). With `σ=0.2` or `σ=0.4`, a Gaussian draw can produce negative channel values (e.g. predator ch3 = 0.0 ± 0.4). Whether the olfaction sensor clips or allows negative values determines whether the ambiguity is actually symmetric. No crash is expected; behaviour depends on the sensor's clipping logic. Auditor cannot confirm clipping without reading `sensor.py` olfaction pooling — recommend verifying the clipping path covers negative draws. | Verify that `apply_olfaction` or the caller clips sampled `properties` at 0.0 before using them, or constrain the grid to `s + σ < 0.5` going forward. This is a scientific concern, not a config error. |
| NOTE | `random_start_pos: true` — latent-bug awareness | All 10 configs set `random_start_pos: true`, which is the intended design for hunger-range coverage. However, the documented latent bug means the agent may spawn on top of a predator or resource at reset (placement does not participate in the occupancy mask), so contact effects can fire on step 0. This is already present in the cell-08 baseline and the experiment design accounts for it. No action needed; recorded for reproducibility. | No fix required. Consistent with cell-08 baseline. |

---

## Checklist

### 1. Observation / Noise Modality Consistency

All 10 configs resolve to the same observation breakdown as the cell-08 baseline:
`{Satiation: 1, Interoceptive Nociception: 1, Extero Nociception: 1, Olfaction: 5, Collision: 5, Proprioception: 6, Visual: 8}` — total 27 dimensions.

The `noise_modality_order` is:
`('Injury', 'Nutrition', 'Satiation', 'Interoceptive Nociception', 'Extero Nociception', 'Olfaction', 'Collision', 'Proprioception', 'Visual', 'Location')` — 10 modalities, zero-padded to 13 slots for JIT stability.

No new sensors are introduced. No modality is omitted from the noise config.

The olfactory modality (index 5 in the noise array) is configured as `state_dependent`, `sigma=0.2` — this is the perceptual noise applied to the olfaction observation slice after aggregation. It is separate from and independent of the per-entity `properties_std` (which governs episode-to-episode sampling of each animal's mean smell vector). No confusion between these two noise mechanisms is present in the configs.

Status: PASS

### 1.5 Behavior-Measures Bush Presence

`behavior_measures.enabled: true` (inherited from `default.yaml` via `extends:`). All 10 configs carry 12 bush obstacles (4 quadrants of 3 each, all `hides_agent: true`, all `count: 3`). The M2 bush-dive measure is operable.

Status: PASS

### 2. Mandatory-Key Discipline

All 10 configs are sparse v3.0 `extends: environment/default` files. No sparse key bypasses `get_mandatory` — all critical fields are either inherited from `default.yaml` (which carries them as mandatory) or explicitly overridden in the sweep files.

`body.random_start_nutrition: true` is set → conditional-mandatory range keys `start_nutrition_low: 10`, `start_nutrition_high: 100` are present and valid (10 ≤ 100; minimum 10 is above the metabolic-cost-per-step starvation threshold).

`body.random_start_injury: true` is set → `start_injury_low: 0`, `start_injury_high: 80` are present and valid (0 ≤ 80 < 100 death line). The bound of 80 is strictly below the injury-death threshold.

`body.random_start_satiation: false` → satiation range keys are not required and are intentionally not set. The comment in each config correctly notes satiation is derived from nutrition at reset and the key is a documented no-op.

Legacy `property` (singular) key: not used in any of the 10 configs. All entities use `properties` (plural). No deprecation warning will fire.

Per-entity olfactory `properties` key: present and complete on all resource, entity, and obstacle entries. No missing-key hard failures.

Status: PASS

### 3. Static-Field and JIT Recompile Risk

All 10 configs share the same static fields:
- `height=10`, `width=10`
- `placement_mode=per_entity`
- `use_homeostatic_reward=True`
- `visual_sensor_enabled=True`
- `noise_modality_order` (same 10-tuple on all 10)
- `interoceptive_kernel_length` (inherited from default, unchanged)

The swept values — `properties` and `properties_std` on entity entries — are traced leaves in `EnvParams` (not static fields). Changing them across the 10 runs does NOT force XLA recompilation. The 10 runs are fully independent training launches, each with its own compiled graph; no cross-run recompile risk.

Status: PASS

### 4. Known Latent-Bug Recurrences

- `overeating_death: false` — confirmed on all 10 (inherited from cell-08 body block). The latent bug (sets `termination_reason=3` without triggering `done=True`) is not triggered.
- `random_start_pos: true` — present on all 10. The spawn-on-predator latent behaviour is active but is the intended design (see NOTE in Findings). Consistent with cell-08.
- `body.start_satiation` / `body.random_start_satiation` — `random_start_satiation: false` is explicitly set (a no-op, correctly documented in each config's inline comment). No reliance on the dead key.
- Legacy `property` (singular): absent from all 10 configs.
- Per-entity missing `properties`: absent — all entities carry both `properties` and `properties_std`.
- Resource respawn occupancy: not a concern at this density (4 × 2 = 8 food, regeneration_delay = 0, no high regen rate configured).
- `terminated` vs `done`: no config-level assumption about divergence between the two.

Status: PASS

### 5. Schema Padding and Modality-Count

10 modalities configured; noise arrays padded to 13. No new sensor is added in the sweep. The 13-slot static padding is unchanged from the base and cell-08. No 14th sensor is introduced.

Status: PASS

### 6. Cross-Config Coherence (Sweep Audit)

Full `diff` of all 10 configs against config 01 (`01-s0_sig0.yaml`) confirms:

Only non-comment lines that differ are:
- `entities[0].properties` — predator mean smell vector (varies with `s`)
- `entities[0].properties_std` — predator per-episode jitter (varies with `σ`)
- `entities[1].properties` — rabbit mean smell vector (mirrored)
- `entities[1].properties_std` — rabbit per-episode jitter (same σ)
- `entities[2].properties` — inert count:0 slot (matches predator for template consistency)
- `entities[2].properties_std` — inert count:0 slot std

Every other field is byte-identical across the 10: scene layout, food resources, obstacle positions, body parameters, sensor config, perceptual noise settings, `behavior_measures` config (including `eval_seeds` — all 10 inherit the `rng:42` generator from `default.yaml`, which produces the same 200 seeds as cell-08's explicit list and is therefore cross-cell comparable), and chase parameters.

Arithmetic verification (via live `load_env_params` inspection):

| Run | s | σ | predator ch2 | predator ch3 | rabbit ch2 | rabbit ch3 | Result |
|-----|---|---|---|---|---|---|---|
| 01 | 0.0 | 0.0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | CORRECT |
| 02 | 0.05 | 0.0 | 0.5500 | 0.4500 | 0.4500 | 0.5500 | CORRECT |
| 03 | 0.1 | 0.0 | 0.6000 | 0.4000 | 0.4000 | 0.6000 | CORRECT |
| 04 | 0.25 | 0.0 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | CORRECT |
| 05 | 0.5 | 0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | CORRECT |
| 06 | 0.1 | 0.2 | 0.6000 | 0.4000 | 0.4000 | 0.6000 | CORRECT (std=0.2) |
| 07 | 0.25 | 0.2 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | CORRECT (std=0.2) |
| 08 | 0.5 | 0.2 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | CORRECT (std=0.2) |
| 09 | 0.1 | 0.4 | 0.6000 | 0.4000 | 0.4000 | 0.6000 | CORRECT (std=0.4) |
| 10 | 0.5 | 0.4 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | CORRECT (std=0.4) |

The formula `predator = [0, 0.5+s, 0.5-s, 0, 0]`, `rabbit = [0, 0.5-s, 0.5+s, 0, 0]`, `std = [0, σ, σ, 0, 0]` is implemented correctly on all 10 grid points.

Status: PASS

---

## Scene Fidelity vs. Cell-08 Baseline

The configs reproduce the cell-08 contrast accurately:
- 1 predator (`class: predator`, `behaviour: hunt`, `count: 1`, `damage: [5.0, 120.0]`, `disengage_on_contact: true`, full-grid spawn/patrol)
- 1 rabbit (`class: neutral`, `behaviour: hunt`, `count: 1`, `damage: [0.0, 0.0]`, same chase profile)
- 1 inert hiding-predator slot (`count: 0`) — schema template only, does not instantiate
- 4 food quadrants × 2 = 8 food sources (`type: food`, `count: 2` each)
- 4 rock groups × 3 = 12 rocks (non-blocking, `damage: [1, 5]`)
- 4 bush groups × 3 = 12 bushes (`hides_agent: true`, no damage)
- 1 inert tree (`count: 0`)

Visual channels confirmed: predator ch5 (`[0,0,0,0,0,1,0,0]`), rabbit ch7 (`[0,0,0,0,0,0,0,1]`), food ch3 (`[0,0,0,1,0,0,0,0]`), rocks/bushes ch6 (`[0,0,0,0,0,0,1,0]`).

The key difference from cell-08 is that the configs add `visual_properties` and `visual_properties_std` to all entities (v3.0 feature, all std = zeros → deterministic appearance) and add `body.random_start_nutrition/injury` with ranges (not in cell-08, which started all episodes at max nutrition/zero injury). This is the intended experimental delta.

---

## Live-Load Verification

Both `load_env_config(path)` and `load_env_params(config)` complete without error on all 10 configs when invoked from the project root using the project conda interpreter (`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`). The `get_observation_breakdown(params)` call also completes without error on all 10.

---

## Conclusion

Safe to launch. No blockers. Two informational notes recorded (potential negative Gaussian draws at extreme `s=0.5, σ>0` grid points; `random_start_pos` spawn-on-entity latency, which is expected and matches cell-08).

Audited by: env-config-auditor
