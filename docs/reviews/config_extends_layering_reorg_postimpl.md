# Config Audit — Post-Reorg Soundness (extends-layering + experiment-path migration)

**Scope:** Post-implementation audit of the config-layering and experiment-path reorganisation that landed on branch `v3.0` (commits `c13a3ac..4997ece`). The reorg (a) moved 91 experiment config files from `configs/experiment/` to `configs/environment/experiment/archive/`, (b) added a new `load_env_config()` function that enables opt-in `extends:` layering, and (c) swept all code/script/test references to the new path. This audit verifies config soundness, parity coverage, obs-noise alignment, and latent-bug status.
**Files audited:** `configs/environment/default.yaml`, representative archived configs (hypervigilance, nmn_noise_heterogeneity, behavior_measures, basic, dreamer_curriculum groups), `tests/env/test_unified_parity.py`, `tests/env/test_backward_compat_configs.py`, `tests/env/test_extends_layering.py`, `src/environment/config_loader.py` (loader inspection only), `scripts/generate_parity_fixtures.py`, `.claude/agents/experiment-designer.md`, `.claude/agents/training-runner.md`
**Audited by:** env-config-auditor
**Date:** 2026-06-17

---

## Summary

The reorganisation is **structurally sound for the configs it covers**. The three core claims hold: (1) the 36 archived configs that have mandatory keys complete load cleanly with zero unexpected errors; (2) the 19 fixture-backed configs all pass parity (31 passed / 82 skipped — identical to the pre-reorg baseline of 31 / 82); (3) all observation-to-noise modality linkages in `default.yaml` and the representative archived configs are correctly aligned. Two items require attention: a **coverage gap** (55 of 91 archived configs are loadable only through the Camp-A seeded path, not standalone, meaning the parity suite has no byte-comparison coverage over them — this is a pre-existing condition, not a regression, but it is now more visible); and **stale-path text in two agent instruction files** (`experiment-designer.md` and `training-runner.md`) that will cause newly authored experiment configs to be placed at the wrong path. One informational note on `random_start_pos: true` in `default.yaml`.

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| CONCERN | `.claude/agents/experiment-designer.md:17,65,97,121,138` | Five lines still say "Place experimental configs at `configs/experiment/<topic>/`". Any future session that reads this agent doc will author new configs at the old (now-nonexistent) path instead of the current `configs/environment/experiment/<topic>/`. | Developer updates these five lines to `configs/environment/experiment/<topic>/`. |
| CONCERN | `.claude/agents/training-runner.md:168` | `--wandb-group` description says "Top dir under `configs/experiment/`". Misleads future launches. | Developer updates to `configs/environment/experiment/`. |
| INFORMATIONAL | `configs/environment/experiment/archive/` — 55 of 91 configs | 55 archived configs fail to load standalone (`sensory.injury_observable` missing — they pre-date that mandatory key). The backward-compat test and parity suite correctly skip them. This means **only 19 of 91 archived configs** have byte-comparison parity coverage. The 72 without fixtures cannot be verified to be byte-identical post-move by the test suite; they can only be verified by the `git mv` rename-only guarantee (which holds — `git status` at move time showed only R entries). This is a pre-existing gap, not introduced by the reorg, but it becomes more visible now that these configs are the *only* home for 55 files that cannot be loaded standalone. | No action required for the reorg itself. If any of the 55 configs are ever used again they must be loaded via Camp-A (the seeded-base path in `dreamer_srl_main`). Document this gap in the Implementation Report note. |
| INFORMATIONAL | `configs/environment/default.yaml:10` — `random_start_pos: true` | The base config has random start position enabled. This is a known latent behaviour: the agent may spawn on top of a predator or resource at reset, and contact effects fire on step 0. Any `extends:`-based sparse config that inherits this without an explicit override will silently acquire random-spawn semantics. This is correct behaviour and matches the pre-reorg default, but new sparse-config authors must be aware. | Add a comment in `default.yaml` near `random_start_pos` (or in the authoring guide) noting the step-0 contact implication. |
| NIT | `configs/continual/*.yaml` — comments only (e.g. `dreamer_srl_3stage_curric_T1.yaml:6,23`) | Several continual-learning schedule files have inline comments mentioning `configs/experiment/dreamer_srl_curriculum/` as the stage source. These are comments, not load-bearing paths (the actual stage YAMLs are embedded in the files themselves). No load or launch will break. | These can be updated in a future pass for consistency; not blocking. |

---

## Checklist

### 1. Observation ↔ Noise Modality Consistency

PASS. `default.yaml` carries 10 noise modalities in `perceptual_noise.modalities` (keys: `injury`, `nutrition`, `satiation`, `interoceptive_nociception`, `extero_nociception`, `olfaction`, `collision`, `proprioception`, `visual`, `location`). These map to the 10 display names the loader places in `noise_modality_order` via `_YAML_KEY_TO_SENSOR_NAME`. Tested live: `get_observation_breakdown(params)` on `default.yaml` returns `{Satiation, Interoceptive Nociception, Extero Nociception, Olfaction, Collision, Proprioception, Visual}` (7 active sensors, with `injury_observable: false` and `nutrition_observable: false`). Every active sensor key is present in the 10-entry noise modality tuple. The 3 inactive sensors (`Injury`, `Nutrition`, `Location`) are in the noise tuple but not emitted at runtime — this is correct (noise runs on the emitted observation slice only; the modality order just establishes index positions). The same pattern holds on all three representative archived configs tested (`01-interoNocicept_sameProp.yaml`, `p1_flat.yaml`, `smoke_test.yaml`). No `KeyError` risk.

Noise array padding: 10 configured modalities, padded to 13 (the fixed static shape). No 14th sensor introduced by the reorg.

### 1.5 Behavior-measures Bush Presence

PASS. `behavior_measures.enabled: true` in `default.yaml`. M2 (bush-dive rate) requires at least one obstacle with `hides_agent: true`. `default.yaml` declares 4 bush obstacle entries; two of them have `count: 5` each (10 bushes total placed at runtime). M2 is structurally interpretable for any `extends:`-based config that inherits the default and does not suppress obstacles.

Caveat: an `extends:`-based sparse config that overrides `obstacles: []` to suppress all obstacles will silently zero out M2 (always false, numerically 0.0 rather than NaN). This is an authoring pitfall, not a default-config defect. Should be noted in the sparse-config authoring guide.

### 2. Mandatory-Key Discipline

PASS for configs that load; flagged coverage gap for those that do not.

- **Camp-B standalone (archived) load of a representative newer config** (`01-interoNocicept_sameProp.yaml`): loads cleanly through `Config(yaml.safe_load(...))` → `load_env_params(...)` with no fallback — all mandatory keys satisfied.
- **Camp-A (extends:) notional path**: tested via `test_c2_extends_satisfies_mandatory_keys` (PASSED per Implementation Report). The merged config satisfies all mandatory keys through the base.
- **Skipped archived configs (55 of 91)**: fail with `ValueError: Configuration key 'sensory.injury_observable' is required but missing.` This is the correct and expected behaviour — these configs pre-date the `injury_observable` mandatory key and were never able to load standalone after that key became mandatory. The reorg does not change this; they were already in this state before the move.
- **`body.start_satiation` / `body.random_start_satiation`** dead keys: present in `default.yaml` (lines 269, 254) as required by the schema, never read by `core.py`. No caller newly relies on them. Status: schema-present, runtime-ignored, same as before. No regression.

### 3. Static-Field & JIT Recompile Risk

N/A for this reorg. No static fields are changed or swept. The reorg is a file rename + loader chokepoint addition. `load_env_config` runs once at startup, not in the hot loop. `load_env_params`, `jax_reset`, and `jax_step` are unchanged.

### 4. Known Latent-Bug Recurrences

PASS (no new triggers introduced).

- **`overeating_death`**: set to `false` in `default.yaml`. No archived config sampled sets it to `true`. Latent bug is not active.
- **`random_start_pos: true`**: set to `true` in `default.yaml`. This is a known step-0 spawn risk (agent may overlap predator/resource). This was `true` before the reorg and is unchanged. Flagged as INFORMATIONAL above because sparse `extends:` configs will now silently inherit this value. Not a regression.
- **Legacy `property` (singular) key**: zero occurrences in `configs/` (grep confirmed no `^\s*property:` hits).
- **`terminated` vs `done`**: these carry identical information; no config assumes they diverge.
- **Resource respawn occupancy**: no config sets abnormally high `regen` rates in the sample reviewed. Not introduced by the reorg.

### 5. Schema Padding & Modality-Count

PASS. 10 configured modalities, padded to 13. No new sensors added by this reorg. Padding count is unchanged and matches the static shape contract.

### 6. Cross-Config Coherence

N/A. This is not a sweep audit. The reorg is a structural reorganisation, not a multi-config sweep.

---

## Stale-Reference Verdict (Audit Item 2)

The migration claims C10 (zero stale references in `src/`, `scripts/`, `tests/`). **This holds for all runtime-executable files.** Confirmed by `grep -rn "configs/experiment" src/ scripts/ tests/` returning zero hits that are not already `configs/environment/experiment`.

**In `configs/` YAML files**: stale path text exists only inside YAML comments in `configs/continual/*.yaml` (not load-bearing). These are cosmetic.

**In `.claude/agents/` files**: two agent instruction documents (`experiment-designer.md` and `training-runner.md`) carry stale `configs/experiment/` path guidance. These are not executed code but ARE read by every session that invokes those agents, meaning new experiment configs will be placed at the wrong path. This is the most practically consequential finding in the audit and is flagged as CONCERN.

---

## Parity Suite Coverage

| Category | Total | With Parity Fixture | Without Fixture (skipped) |
|---|---|---|---|
| `configs/environment/experiment/archive/**` | 91 | 19 | 72 |
| `configs/continual/**` | various | 5 (nmn_double_return_stages) | remaining |
| `configs/verification/**` | several | 6 | remaining |
| `configs/environment/default.yaml` | 1 | 1 | — |

The parity gate reported **31 passed, 82 skipped** — exactly matching the pre-reorg baseline. The fixture slug rename (`configs__experiment__*` → `configs__environment__experiment__archive__*`) is correctly applied: 19 fixtures were renamed and the test glob finds them at the new paths.

Configs without fixtures are skipped (not failed). The skip reason for the 55 loadable-only-via-Camp-A configs is "No pre-refactor fixture for ... (config was stale before refactor)". This is accurate — these 55 configs failed to produce parity fixtures before the reorg because they required the Camp-A base seed. The reorg does not close this gap; the byte-for-byte guarantee for these 55 rests entirely on the `git mv` rename-only semantics (confirmed by `git status` at move time).

---

## Conclusion

Pass-with-notes. No blockers. Two concerns that require developer action before the next experiment session: update `experiment-designer.md` and `training-runner.md` to use the new `configs/environment/experiment/` path, or any future experiment config will be authored at the non-existent old location.

Audited by: env-config-auditor
