---
title: "Config audit — v3.0 config-system overhaul (independent, full-scope)"
topic: config-system
status: complete
created: 2026-06-20
last_updated: 2026-06-20
aliases: [config-v3-overhaul-audit]
---

# Config Audit — v3.0 Config-System Overhaul (Independent Full-Scope)

## What this audit is about

This is an independent, ground-up soundness audit of the entire v3.0 config-system overhaul
that landed on the `v3.0` branch across 49 commits and 107 config files changed since commit
`871f146`. The v3.0 overhaul introduced four main capabilities: (1) `extends:` layering so
new experiment configs can inherit from a single base and only override what differs, (2)
per-entity configurable visual appearance vectors replacing hard-coded channel numbers, (3)
per-episode visual-property sampling via a standard-deviation key, and (4) configurable
initial-state randomization ranges. An experiment-path reorganization moved ~91 older configs
into an `archive/` subdirectory and introduced two new experiment families: a five-level
difficulty curriculum (`configs/environment/experiment/basic/`) and a ten-config olfactory
discrimination sweep (`configs/environment/experiment/hunger_gated/`).

**Verdict: PASS WITH TWO CONCERNS.** All 16 changed/new configs load cleanly, all core
checks pass, and the config system operates correctly. However, a post-audit config rename
(commit `52c89c0`, 2026-06-19) broke the parity test guard for the five basic curriculum
configs, leaving those configs without active byte-equality regression coverage. This is a
P1 concern, not a P0 blocker — the configs themselves are sound, but the safety net is
disabled. One additional concern: the `configs/dreamer_srl/` directory does not exist,
making `tests/algorithms/dreamer_srl/test_lax_scan_train.py` silently fall back to a
different config path. This is pre-existing and orthogonal to the v3.0 overhaul.

---

**Scope:** v3.0 config-system overhaul (49 commits, 107 files) — `extends:` layering,
visual properties, visual sampling, initial-state ranges, experiment reorg, basic curriculum,
hunger-gated sweep  
**Base commit:** `871f146`  
**Files audited:** `configs/environment/default.yaml`, all 5 `configs/environment/experiment/basic/*.yaml`,
all 10 `configs/environment/experiment/hunger_gated/*.yaml`, archived configs (spot-checked),
`src/environment/config_loader.py`, `src/environment/sensor.py`, `src/environment/state.py`,
`src/behavior/accumulators.py`, all `tests/env/` test files  
**Audited by:** env-config-auditor  
**Date:** 2026-06-20

---

## Summary

The v3.0 config system works correctly. The `extends:` merge resolves cleanly (7/7
`test_extends_layering.py` tests pass), all 16 new/changed active configs load and produce
valid `EnvParams`, visual property channels are byte-identical to the intended class defaults
(food→ch3, hiding predator→ch4, hunt predator→ch5, rock/bush→ch6, neutral→ch7,
grass/sand/plain background→ch0/1/2), the hunger-gated sweep is internally coherent (static
fields identical across all 10 configs, only olfactory properties vary), and the initial-state
ranges propagate correctly through the `extends:` merge.

The full `tests/env/` suite passes (156 passed, 197 skipped, 0 failed in 508s). The skips
fall into two categories: stale archived configs that pre-date the mandatory
`sensory.injury_observable` key (expected, 55 configs, documented in
`test_backward_compat_configs.py`), and the parity/sampling tests that silently skip the
five renamed basic curriculum configs (not expected, see Finding #1).

Two concerns require developer attention. Neither blocks the already-running training experiments
(which use the hunger-gated configs, which are sound), but Finding #1 should be fixed before
the next code change that could silently alter observations in the basic curriculum.

---

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| P1 concern | `tests/env/test_visual_parity.py` lines 52-58; `tests/env/test_visual_sampling.py` lines 332, 378-382 | Config rename in commit `52c89c0` removed files `00-forage_5x5.yaml`, `01-slowPred_5x5.yaml`, `02-fastPred_8x8.yaml`, `03-multiPred_10x10.yaml`, `04-keenPred_10x10.yaml` and replaced them with new names. Three test files were NOT updated: `test_visual_parity.py` has fixture paths and parametrize entries referencing old names, causing 5/7 parity test parametrizations to skip silently on "config not found". `test_visual_sampling.py` Gate 4 (single-config) and `test_visual_obs_unchanged_all_configs` both use old names and skip silently. The visual parity fixtures at `tests/env/fixtures/visual_parity/` also use old slugs. | (a) Update the five config name entries in `test_visual_parity.py` (lines 53-57) to the new names; (b) generate new fixture files for the five renamed configs; (c) update `test_visual_sampling.py` lines 332 and 378-382 to use new names. The new configs have all-zero `visual_properties_std`, so the byte-parity test can run directly (no pre-change fixture is needed — run once with `--gen-fixtures` to capture). |
| nit | `tests/algorithms/dreamer_srl/test_lax_scan_train.py` line 58-61 | The test looks for `configs/dreamer_srl/01_food_only_smoke.yaml` (primary) and `configs/dreamer_srl/01_food_only.yaml` (fallback). Neither path exists — the directory `configs/dreamer_srl/` does not exist; the DreamerSRL agent configs live at `configs/models/dreamer_srl/`. The fallback logic silently loads from an unexpected path if the variable resolves elsewhere. This is pre-existing and orthogonal to the v3.0 overhaul. | Update the test to reference `configs/models/dreamer_srl/` or use a minimal inline config stub. Flag for `developer`. |

---

## Checklist

### 1. Observation vs Noise Modality Consistency

**PASS.**

`get_observation_breakdown` emits 7 sensors for the default config: Satiation, Interoceptive
Nociception, Extero Nociception, Olfaction, Collision, Proprioception, Visual (total=27). The
`noise_modality_order` tuple holds 10 names: Injury, Nutrition, Satiation, Interoceptive
Nociception, Extero Nociception, Olfaction, Collision, Proprioception, Visual, Location.
The 3 modalities present in the noise list but absent from the observation breakdown (Injury,
Nutrition, Location) correspond to sensors gated off by `injury_observable: false`,
`nutrition_observable: false`, and `location_sensor: false` in `default.yaml`. This
discrepancy is intentional and correct — `apply_perceptual_noise` iterates the observation
breakdown, not the noise list, so absent sensors receive no noise application.

No new sensors were added in this overhaul. The visual modality noise block auto-resizes
with `visual_vector_size` dynamically, with no second hardcoded width in the noise path
(confirmed from prior audit at commit `6554546`). All 16 new configs produce `total_obs=27`
and `noise_sigmas.shape=(13,)` (10 active modalities + 3 zero-padding slots).

### 1.5. Behavior-Measures Bush Presence

**PASS with design note.**

`behavior_measures.enabled: true` is inherited from `default.yaml` by all 15 sparse configs.
Configs `01-slow_predator_5x5` through `04-far_sight_predator_10x10` each explicitly declare
`obstacles:` lists containing bushes with `hides_agent: true` (3, 6, 10, 10 bushes
respectively). Config `00-static_predator_5x5` declares `obstacles: []` and `entities: []`,
giving it 0 bushes and 0 hunt-type predators. The M2 bush-dive measure computes onset
from `dist_per_predator` (hunt-type predators only, not hiding_predator resources), so
`00-static_predator_5x5` produces `threat_in_R=False` for all steps — M2 denominator = 0,
M2 rate = NaN. This is correct by design: the anchor level has no chase threat to dive from.
The hunger-gated configs have 12 `hides_agent: true` obstacles and 2 hunt-type animals
(predator + rabbit), so M2 is structurally well-defined.

### 2. Mandatory-Key Discipline

**PASS.**

No new mandatory keys were added in the overhaul beyond what the prior audits (commits
`6554546` and `1de20b7`) already cleared. The five new conditional-mandatory keys
(`start_nutrition_low`, `start_nutrition_high`, `start_injury_low`, `start_injury_high`)
are gated behind their `random_start_*` flags and read via `config.get_mandatory` only
when the flag is true. `default.yaml` carries all keys explicitly with the flags set false.
The hunger-gated configs override the flags to `true` and supply the range values.
Live instantiation confirms correct propagation: `start_nutrition_low=10.0`,
`start_nutrition_high=100.0`, `start_injury_high=80.0` in `01-s0_sig0.yaml`.

The `body.start_satiation` and `body.random_start_satiation` keys are declared in
`default.yaml` (as required by schema) and are read-but-ignored at runtime — no config in
this overhaul changes this behavior. No new caller relies on these keys.

No `property` (singular) legacy key appears in any new config. All entities use
`properties` and `visual_properties` (plural) consistently.

### 3. Static-Field and JIT Recompile Risk

**PASS.**

The basic curriculum varies `height` and `width` across its five levels (5×5, 5×5, 8×8,
10×10, 10×10). Both are declared `struct.field(pytree_node=False)` in `EnvParams`
(`state.py:88-89`), making them static. Each curriculum level is run as a **separate**
training process with its own JIT context, so cross-run recompilation is expected and not
a concern.

The hunger-gated sweep holds all static fields identical across its 10 configs (height=10,
width=10, visual_vector_size=8, interoceptive_kernel_length=12, placement_mode=per_entity,
visual_sensor_enabled=True). No JIT recompile hazard within the sweep.

`visual_vector_size` is fixed at 8 across all new configs and the default. No sweep varies
this field.

### 4. Known Latent-Bug Recurrences

**No new triggers introduced. Known issues flagged as informational below.**

- `body.overeating_death: false` in default.yaml and inherited by all sparse configs.
  Not triggered.
- `random_start_pos: true` is inherited from default.yaml by all sparse configs including
  both curriculum families. The agent may therefore spawn on a predator or resource at
  episode start, with contact effects firing on step 0 (occupancy mask does not participate
  in placement). This is the pre-existing behavior for all hypervigilance experiments and
  is accepted as known. Not a new regression from the v3.0 overhaul.
- No `property` (singular) key present in any new config.
- No new entity definition is missing the `properties` (olfactory) key.
- `terminated` vs `done` duality: no new config relies on these diverging.

### 5. Schema Padding and Modality-Count

**PASS.**

10 named modalities in `default.yaml` (indices 0-9), padded to 13 slots in the noise
arrays (`noise_sigmas.shape=(13,)`, confirmed by live load). No new sensor or noise
modality was added in this overhaul. The count remains well below the 13-slot ceiling.

### 6. Cross-Config Coherence (Sweep Audits)

**PASS for hunger-gated sweep (primary active sweep).**

All 10 hunger-gated configs share identical static fields (height, width, visual_vector_size,
interoceptive_kernel_length, placement_mode), identical body parameters (metabolic_cost,
food_nutrition_gain, max_nutrition, death_penalty, recovery_base_rate, init ranges),
identical visual properties for all entity classes, and identical obstacle layouts. The
only variation is `animal_property` (olfactory mean, `s` parameter) and `animal_property_std`
(`sigma` parameter) — exactly the two swept dimensions. The sweep matrix is correct:
`s ∈ {0, 0.05, 0.1, 0.25, 0.5}` with `sigma ∈ {0, 0.2, 0.4}` (10 points covering the
planned discrimination-onset map).

**N/A for basic curriculum** — the five levels are separate training targets, not a sweep;
they intentionally vary scene composition and grid size. Cross-config coherence analysis
does not apply.

---

## Config Load Results

| Config family | Load status | Count |
|---|---|---|
| `configs/environment/default.yaml` | PASS | 1/1 |
| `configs/environment/experiment/basic/` | PASS | 5/5 |
| `configs/environment/experiment/hunger_gated/` | PASS | 10/10 |
| `configs/environment/experiment/archive/**` (loadable) | PASS | 52/107 (55 skip as pre-v3.0 stale configs missing `sensory.injury_observable`) |

All 16 active (non-archived) configs produce `total_obs=27` and `noise_modality_order`
= `('Injury', 'Nutrition', 'Satiation', 'Interoceptive Nociception', 'Extero Nociception',
'Olfaction', 'Collision', 'Proprioception', 'Visual', 'Location')`.

---

## Visual Channel Parity Verification

All channels match the class defaults documented in `CONFIG_GUIDE.md`:

| Entity type | Expected channel | Verified |
|---|---|---|
| Food resource | 3 | Yes (all basic + hunger-gated) |
| Hiding predator resource | 4 | Yes (all basic) |
| Hunt predator animal | 5 | Yes (all basic + hunger-gated) |
| Rock obstacle | 6 | Yes (all basic + hunger-gated) |
| Bush obstacle | 6 | Yes (all basic + hunger-gated) |
| Neutral animal (rabbit) | 7 | Yes (03-rabbit, 04-farsight, hunger-gated) |
| Background grass | 0 | Yes (default + hunger-gated) |
| Background sand | 1 | Yes (default + hunger-gated) |
| Background plain | 2 | Yes (default + hunger-gated) |

All `visual_properties_std` vectors are all-zeros across all new configs (deterministic
visual appearance, no per-episode sampling noise).

---

## Reorg Integrity: Dangling Path References

The experiment-path reorganization moved configs from flat `configs/experiment/*/` paths to
`configs/environment/experiment/archive/*/`. The following dangling references were found but
are all **inert** — they appear only in documentation comments or test-skip guards:

- `configs/experiment/hypervigilance/...` — appears in archived YAML comment lines
  (provenance notes) and in doc files under `docs/develop/archive/` and `docs/develop/active/`.
  None are runtime-loaded paths.
- `configs/experiment/basic/...` — appears in archived YAML comment lines only.
- `configs/experiment/dreamer_srl_curriculum` — `test_continual_schedule.py` line 52
  references this path; the test has a `skipif not os.path.isdir(...)` guard that correctly
  skips when the path is missing. The real curriculum configs now live at
  `configs/environment/experiment/archive/dreamer_srl_curriculum/`.
- `configs/dreamer_srl/01_food_only.yaml` — `test_lax_scan_train.py` fallback path.
  Logged as nit finding above.

---

## Test Gate Summary

```
pytest tests/env/test_extends_layering.py    — 7 passed, 0 skipped, 0 failed
pytest tests/env/test_backward_compat_configs.py — 55 passed, 80 skipped, 0 failed
pytest tests/env/test_visual_parity.py       — 3 passed, 5 skipped (STALE PATHS), 0 failed
pytest tests/env/test_visual_sampling.py     — 6 passed, 1 skipped, 0 failed
pytest tests/env/test_initial_state_ranges.py — 6 passed, 0 skipped, 0 failed
pytest tests/env/ (full suite)               — 156 passed, 197 skipped, 0 failed (508s)
```

The 5 skipped visual parity entries are **not** healthy skips — they are silent coverage
gaps caused by the config rename (Finding #1). The test infrastructure treats a missing
config file as a skip, not a failure, so the breakage is invisible in CI output.

---

## Conclusion

**Pass with two concerns. Safe to launch existing and queued training runs.** The core config
machinery is sound, all 16 active configs load and produce correct observations, the
hunger-gated sweep is coherent, and the `extends:` layering works correctly. The P1 concern
(parity gate disabled for the five renamed basic configs) should be resolved before the
next code change that could silently perturb observations in that family. The nit (stale
dreamer_srl test path) is pre-existing and low-priority.

**Action items for `developer`:**
1. Update `tests/env/test_visual_parity.py` lines 53-57 and `tests/env/test_visual_sampling.py`
   lines 332 and 378-382 to use the new basic config file names (`00-static_predator_5x5.yaml`,
   `01-slow_predator_5x5.yaml`, `02-fast_predator_8x8.yaml`, `03-predator_and_rabbit_10x10.yaml`,
   `04-far_sight_predator_10x10.yaml`).
2. Generate new visual parity fixtures for the renamed configs (run `pytest tests/env/test_visual_parity.py --gen-fixtures` once with the new names in place).
3. (Optional / low priority) Fix `tests/algorithms/dreamer_srl/test_lax_scan_train.py` path
   to point at `configs/models/dreamer_srl/`.

Audited by: env-config-auditor
