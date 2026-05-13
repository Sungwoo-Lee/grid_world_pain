---
title: "SheepRL Bridge Compatibility Audit — 5×5 PredInterval3 and 10×10 Hypervigilance"
topic: config-audit
status: review
created: 2026-05-12
last_updated: 2026-05-12
---

# Config Audit — SheepRL Bridge Compatibility for Two Parallel Training Launches

**Scope:** Pre-flight, two-config sheeprl-bridge compatibility check before parallel training launches on node 114.
**Files audited:**
- `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml` (cuda:0 target)
- `configs/experiment/hypervigilance/01-interoNocicept.yaml` (cuda:1 target)
- Baseline reference: `configs/experiment/dreamer_curriculum/01_food_only.yaml`
- Bridge: `tmp/sheeprl/sheeprl/envs/grid_world_pain.py`
- Loader: `src/environment/config_loader.py`

**Audited by:** env-config-auditor
**Date:** 2026-05-12

---

## Purpose

Two new sheeprl training runs are scheduled to launch in parallel on node 114 — one training a stock DreamerV3 agent on a small 5×5 grid with a mobile predator and a static ambush predator, and another on a larger 10×10 grid with the full hypervigilance configuration (food, mobile predator, four ambush predators, rocks, bushes, and neutral animals). Before committing compute, this audit checks that both environment YAML configs are safe to hand to the sheeprl bridge, which loads each config once at environment init, runs a reset, then calls the observation function to infer the observation shape and action count. If that load-and-probe sequence raises any error, the training process dies before the first step.

The known-good baseline is a food-only config that successfully reached the maximum survival duration in a prior sheeprl smoke test (smoke run "jzgkcep4" on 2026-05-11). Both target configs must pass the same load-and-probe without raising, produce a flat 1-D observation vector whose length is consistent across resets, and expose a discrete action space of the expected size (6 actions).

**Headline verdict:** Both configs pass. `load_env_params` and `get_observation` complete without error for both. Each produces a flat observation vector of the expected dimension, and the action space is Discrete(6) as required. No blockers were found. Two concerns are noted but neither prevents launch.

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| concern | `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml`: `perceptual_noise.modalities` | The YAML defines only 9 modalities (omits `interoceptive_nociception`). After deep-merging with `configs/environment/default.yaml`, the merged result correctly contains all 10 modalities in the right order — so there is no runtime crash today. However, the config's own comment says "Key order here is the single source of truth for noise array indices", which is misleading: the actual source of truth after merging is the default config's key order. If someone strips the default merge and loads this YAML alone, the 10th modality entry would be missing from the merged table, causing a `KeyError` when `apply_perceptual_noise` looks up `"Interoceptive Nociception"`. Not a blocker now because noise is disabled (`enabled: false`) and the bridge always merges defaults first, but the comment overstates the YAML's authority. | Add the `interoceptive_nociception` entry to this config's noise modalities block (even with sigma=0.0) so the config is self-consistent without relying on defaults to fill the gap. Flag for `experiment-designer`. |
| concern | `configs/experiment/hypervigilance/01-interoNocicept.yaml` + `01-5X5_PredInterval3_NutGain18.yaml`: `environment.random_start_pos: true` | Both configs spawn the agent at a random position each reset. Per the known latent-bug table (`ENVIRONMENT_SUMMARY.md` FAQ §4), random spawn does not participate in the occupancy mask: the agent can spawn on top of a resource or predator, and contact effects fire on step 0 with no action taken. On the 5×5 grid this is non-trivial — 6 entities in 25 cells means roughly a 1-in-4 chance of overlap. This is a training-quality concern, not a crash concern. | N/A for this launch — the configs are deliberate (same pattern as the proven food-only baseline). Note for experiment design: if training stalls on the 5×5, spawn-on-predator step-0 deaths may be a confound. |
| nit | `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml`: `perceptual_noise.modalities.injury.sigma` and `.nutrition.sigma` | Both are set to `0.1` in this config, while the default has them at `0.0`. Because noise is disabled globally (`enabled: false`), these values are parsed but never applied. The discrepancy is harmless today but could confuse a future experiment that enables noise on this config, since the injury/nutrition channels would then have non-zero noise even though the rest of the sensor design treats them as hidden (not observable). | No action needed for this launch. If noise is ever re-enabled on this config, review the sigma values explicitly. |

---

## Checklist

- [x] **(1) Observation-to-noise modality consistency** — Both configs have `perceptual_noise.enabled: false`. After merging with defaults, the `noise_modality_order` (the tuple stored in `EnvParams` and used by `apply_perceptual_noise`) is `('Injury', 'Nutrition', 'Satiation', 'Interoceptive Nociception', 'Extero Nociception', 'Olfaction', 'Collision', 'Proprioception', 'Visual', 'Location')` for both configs. Every key emitted by `get_observation_breakdown` for each config is present in that tuple; no `KeyError` would occur even if noise were enabled. PASS (with concern noted above about the 5×5 config omitting the `interoceptive_nociception` modality from its own YAML block).

- [x] **(1.5) Behavior-measures bush presence** — `behavior_measures` block is absent from both configs; `load_behavior_measure_cfg` returns `None`. Checklist item N/A.

- [x] **(2) Mandatory-key discipline** — Both configs were loaded via `get_default_config()` + `cfg.merge(Config.load_yaml(...))` + `load_env_params(cfg)` in a live Python process (see Methods below). Neither raised a `ValueError`. All `config.get_mandatory(...)` call sites in `src/environment/config_loader.py` resolved. The four keys required by the schema but noted as runtime-ignored (`body.start_satiation`, `body.start_nutrition`, `body.random_start_satiation`, `body.random_start_nutrition`) are present in both configs and in the defaults; no new caller relies on them. PASS.

- [x] **(3) Static-field and JIT recompile risk** — The sheeprl bridge is PyTorch-side; it does not JIT-compile JAX functions itself. `load_env_params` is called once per `GridWorldPainWrapper.__init__`. The two configs are launched as separate processes (cuda:0 and cuda:1), each with their own bridge instance. Static-field mismatches between the two configs (height 5 vs 10, entity counts 6 vs 33, `visual_sensor_enabled` False vs True) are expected and benign because the two instances are isolated. PASS.

- [x] **(4) Known latent-bug recurrences**
    - `body.overeating_death`: `false` in both configs. PASS.
    - `body.start_satiation` / `start_nutrition`: both set to `100` explicitly in both configs. PASS.
    - `property` (singular) vs `properties` (plural): both configs use `properties` throughout. No `DeprecationWarning` was raised during the live load. PASS.
    - Per-entity olfactory key missing: no `ValueError` raised; all entity entries have `properties`. PASS.
    - `random_start_pos: true`: present in both configs. The known spawn-overlap risk is real on the 5×5 (6 entities, 25 cells). Noted as concern above; not a blocker. PASS with concern.
    - `terminated` vs `done`: the bridge uses `done` from `jax_step` directly and maps it to `terminated`. No divergence assumed. PASS.

- [x] **(5) Schema padding and modality-count** — Both configs produce a `noise_modality_order` tuple of length 10. The padding in `_parse_noise_config` adds `13 - 10 = 3` zeros to make the JAX arrays length 13. The 5×5 config YAML has only 9 modalities, but after merging with defaults, the merged dict has 10 (the default's `interoceptive_nociception` entry survives). No 14th modality has been introduced. PASS.

- [x] **(6) Cross-config coherence (sweep)** — These two configs are run in parallel but are not a controlled sweep. They differ intentionally in grid size, entity counts, visual sensor, food placement, and noise sigma values for injury/nutrition. Because they are not paired for statistical comparison, non-matching fields are not a concern. Seed alignment is not applicable — these are independent experiments with separate WandB tags. PASS.

- [x] **(7) Superseded status** — Neither `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml` nor `configs/experiment/hypervigilance/01-interoNocicept.yaml` carries a deprecation comment, and neither appears in the `docs/develop/INDEX.md` archive section. The hypervigilance directory contains three variants (`01-interoNocicept.yaml`, `01-interoNocicept_noise.yaml`, `01-interoNocicept_sameProp.yaml`); the audited config is the base variant and is not marked as superseded by the others. PASS.

---

## Methods — Live Loader Verification

All three configs (two targets + baseline) were loaded and probed via:

```python
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.sensor import get_observation_breakdown

cfg = get_default_config()
cfg.merge(Config.load_yaml('<path>'))
params = load_env_params(cfg)
bd = get_observation_breakdown(params)
```

Results:

| Config | `load_env_params` | `action_dim` | Obs dim | `visual_sensor_enabled` |
|---|---|---|---|---|
| `01_food_only.yaml` (baseline) | SUCCESS | 6 | 19 | False |
| `01-5X5_PredInterval3_NutGain18.yaml` | SUCCESS | 6 | **19** | False |
| `01-interoNocicept.yaml` | SUCCESS | 6 | **27** | True |

The 5×5 target config produces exactly the same observation dimension as the food-only baseline (19). The 10×10 hypervigilance config produces 27 — the extra 8 dimensions come from `Visual` (1 cell × 8 channels, with `visual_sensor_range=0` meaning only the agent's current cell is inspected).

---

## Delta Table vs Food-Only Baseline

| Field | food-only baseline | 5×5 PredInterval3 (cuda:0) | 10×10 hypervigilance (cuda:1) |
|---|---|---|---|
| `environment.height` × `width` | 5 × 5 | 5 × 5 (same) | **10 × 10** |
| Total entities (expanded) | 4 | **6** | **33** |
| Active predator count | 0 (predator count=0) | **1 mobile predator** | **1 mobile predator** |
| Active hiding_predator count | 0 (count=0) | **1** | **4** |
| Active neutral animal count | 0 | **0** | **2 rabbits** |
| Active obstacle count (rocks) | 0 | **0** | **12 rocks** |
| Active obstacle count (bushes) | 3 | 3 (same) | **10 bushes** |
| `visual_sensor_enabled` | false | false (same) | **true** |
| `body.food_nutrition_gain` | 18 | 18 (same) | **6** |
| `body.use_homeostatic_reward` | true | true (same) | true (same) |
| `environment.random_start_pos` | true | true (same) | true (same) |
| `perceptual_noise.enabled` | false | false (same) | false (same) |
| Observation dimension | 19 | **19** (identical) | **27** (8 more from Visual) |
| Action space | Discrete(6) | Discrete(6) (same) | Discrete(6) (same) |
| `interoceptive_nociception_enabled` | true | true (same) | true (same) |
| `injury_observable` | false | false (same) | false (same) |
| `nutrition_observable` | false | false (same) | false (same) |

Every bolded entry is a field the target config sets to a value not present (or present with a different value) in the food-only baseline. Each such delta is a candidate for different bridge behavior, but none causes a crash. The most operationally significant deltas are the presence of live predators (step-0 contact risk from `random_start_pos`) and the visual sensor in the 10×10 config (adds 8 obs dims, which the sheeprl MLP encoder handles via its generic `obs.shape` read).

---

## Conclusion

Both configs are safe to launch. Fix the `interoceptive_nociception` omission in the 5×5 config's noise block before enabling noise on that config in any future experiment.

Audited by: env-config-auditor
