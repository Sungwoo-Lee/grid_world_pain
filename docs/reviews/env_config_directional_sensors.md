# Config Audit — directional-sensors config surface (sweep, arms, obs↔noise sync)

**Scope:** new-sensor addition + 135-config sweep + 11-arm live experiment (pre-flight-style audit of already-launched runs)
**Files audited:** every YAML under `configs/` (415 files; 325 env-shaped configs live-loaded), the 11 arm configs + generator in `configs/environment/experiment/sensory_directional/`, `configs/environment/default.yaml`, `src/environment/config_loader.py`, `src/environment/sensor.py`, `src/environment/state.py`, the live runs' saved WandB metadata
**Audited by:** env-config-reviewer
**Date:** 2026-08-26
**Companion reviews:** [[math_directional_sensors]] (equations vs studies), [[plan_directional_sensors]] (the plan itself)

## Verdict (plain language)

A large sensor change recently landed: smell can now be sampled on a small grid of cells around the agent (so it carries a direction), and vision can be blurred, coarsened to "something is there", or blocked by line-of-sight. To keep every old experiment loading, roughly 135 config files — including frozen archived ones — were bulk-edited to carry the new settings with values that change nothing. Eleven training runs comparing these sensor variants are live on the lab cluster right now. This audit asked four questions: did the bulk edit break anything, do the new settings fail loudly when misused, does the noise system still line up with the (now variable-width) observation vector, and is each live experiment arm really the single-variable comparison its file header claims?

**The answer to all four is yes — this config surface is sound, and the live runs are correctly configured.** Every config that loaded before still loads; the 61 that fail are the same 61 that failed before the change (old archived files with a pre-existing missing key or a broken inheritance pointer). Enabling occlusion without its two sub-settings dies immediately with an error naming the missing key. The noise system was tested empirically at twelve combinations of smell-grid and vision ranges: the noise always lands on exactly the intended slice of the observation. All eleven arms were diffed against their named controls at the fully-resolved level: each differs by exactly its stated manipulation, the six generated files match a fresh regeneration byte-for-byte, and the live runs' own logged observation widths match prediction. Two moderate items (neither threatens the live runs): a negative smell-grid radius is accepted at load and only crashes later with an opaque error, and the visual-parity safety net silently skips 5 of its 8 configs because it still points at renamed file paths.

**Severity legend — reproduce it verbatim in every report so the labels never need looking up:** 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run · 🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| 🟡 Moderate | `src/environment/config_loader.py:1559` — `sensory.olfactory_grid_range` | No range validation at the read site: `olfactory_grid_range: -1` loads without complaint and only dies at JIT trace with an opaque `ValueError: axis 1 is out of bounds for array of dimension 1`. Contrast the occlusion keys (`config_loader.py:1016-1024`), which validate their ranges with named errors. | Add `>= 0` validation at the read site, mirroring the occlusion-key pattern. Owner: `developer`. |
| 🟡 Moderate (pre-existing) | `tests/env/test_visual_parity.py:148` | The visual-parity gate — the safety net CONFIG_GUIDE §6 names for exactly this feature area — silently skips 5 of its 8 configs: it still references `configs/environment/experiment/basic/00-forage_5x5.yaml` (and 4 more) which were renamed in the basic-curriculum re-level. Identical at the pre-change baseline, so not caused by this change — but the gate guarding the visual slice currently covers only `default.yaml` + 2 configs. | Refresh the config list (and fixtures, deliberately, from a pre-change commit) to the current `basic/` filenames. Owner: `developer`. |
| 🟢 Low | `docs/environment/ENVIRONMENT_SUMMARY.md:137-139` | The canonical observation table is stale for this change: the Olfaction row still says `V=olfactory_vector_size` (now `(2r²+2r+1)×V` with `r = olfactory_grid_range`), and the Visual row hard-codes `×8` (configurable `visual_vector_size` since v3.0). `olfactory_grid_range` appears nowhere in the file. The formal Maintenance Contract binds only CONFIG_GUIDE + 02_config_schema (both were updated), but the summary calls its table "authoritative". `10_perceptual_noise.md` also never mentions the variable Olfaction width, though its width-agnostic mechanism description remains correct. | Update the two table rows; one sentence in doc 10 noting Olfaction width is now range-dependent. Owner: `developer`/docs. |
| 🟢 Low | `docs/environment/CONFIG_CRITICAL_SETTINGS.md:26` (change-log entry) | The on-source blast-radius list ("the **eleven** configs shipping `decay_power: 2.0`") is complete **for active configs** but silent about ~60 *archived* configs (`archive/labmeeting/`, `archive/hypervigilance/`, `archive/dreamer_*`, `archive/nmn_*`, `archive/v2_smoke/`) plus 5 test files embedding `decay_power: 2.0` and `tests/env/test_visual_properties.py` at `1.5` (on-source now 2.83, was 2.0). None of these is gated on olfactory observation values (the unified parity gate checks state/info, not obs), so nothing breaks today — but a future re-run of an archived 2.0 config silently reads 4.0 on-source vs its historical runs. | Append one sentence to the change-log entry scoping the archive + test-file exposure. Owner: docs. |
| 🟢 Low | `configs/environment/experiment/sensory_directional/{H,I,J}*.yaml` — `obstacles[tree].blocks_sight: true` | The tree is flagged as a sight-blocker but ships `count: 0`, so in practice only rocks (6–12 per episode) occlude. Consistent with the "obstacles-only blocking" calibration in CONFIG_GUIDE §3.8; noted so nobody reads "rock+tree" in the generator and assumes trees participate. | None needed. |
| 🟢 Low | `src/environment/config_loader.py:321,333` — per-entity `visual_mask` / `blocks_sight` | Read with read-site defaults (`'none'` / `False`) rather than `get_mandatory`. This follows the established per-entity optional-attribute pattern (`hides_agent`, `config_loader.py:1209`), is documented as opt-in in 02_config_schema, and invalid values fail loudly (`visual_mask: sometimes` → `ValueError` naming the valid set). Not a no-fallback violation; recorded so the next auditor doesn't re-litigate it. | None needed. |

No 🔴 Critical findings. Nothing here threatens the eleven live runs.

## Evidence by task

### 1. The sweep (complete and meaning-preserving) — ✅

- **Coverage:** 135 files under `configs/` carry a top-level `sensory:` block; all 135 carry `olfactory_grid_range` and the six `visual_*` keys, all inside the `sensory:` block (checked mechanically per file). No file anywhere in `configs/`, `src/`, `tests/`, `scripts/` still uses the pre-rename key `olfactory_sensor_range`. All 23 non-config files embedding a sensory block (test fixtures, inline YAML in tests) were swept too.
- **Load test:** all 415 YAMLs parsed with a duplicate-key-rejecting loader — **zero duplicate keys, zero YAML errors**. All 325 env-shaped configs were live-loaded through `load_env_config` → `load_env_params`: **264 load, 61 fail**.
- **All 61 failures are pre-existing.** The identical sweep run in a worktree at the pre-change commit (`0e8a4ef^` = `4b0b4af`) produces the **byte-identical failure set**: 55 archived configs missing `sensory.injury_observable` (labmeeting / old-dreamer era, e.g. `configs/environment/experiment/archive/labmeeting/basic-03.yaml`), and 6 archived `basic_releveled_20260704/basic05_variants/*` whose `extends:` targets point at since-renamed `basic/` files. Zero failures caused by this change; zero fixed-by-accident (which would also have signalled a semantic edit).
- **Meaning-preserving:** the archive sweep is 91 files, **637 insertions, 0 deletions** — pure additions of the parity defaults at the top of each `sensory:` block. Value tally across all 135 files: every occurrence of every new key carries the parity default (`0` / `false` / `0.5` / `3.0` / `0.5` / `sum` / `false`) **except** in the `sensory_directional/` arms, where the non-default values are the experiment.
- **Parity gates:** `tests/env/test_visual_parity.py` + `tests/env/test_directional_sensors.py`: 26 passed, 5 skipped (the stale-path skips flagged above). Full `tests/env/test_unified_parity.py` result recorded at the end of this doc.

### 2. Conditional-mandatory occlusion keys — ✅

Implemented at `config_loader.py:1014-1025` exactly per CONFIG_GUIDE §5's conditional pattern, and verified behaviourally:

| Probe (layered on `environment/default`) | Result |
|---|---|
| `visual_occlusion_enabled: true`, both sub-keys absent | `ValueError: … 'sensory.visual_occlusion_cone_deg' is required but missing` |
| enabled, only `cone_deg` given | `ValueError` naming `visual_occlusion_strength` |
| enabled, only `strength` given | `ValueError` naming `visual_occlusion_cone_deg` |
| `cone_deg: 95.0` | `ValueError: must be in (0, 90)` |
| `strength: 1.5` | `ValueError: must be in [0, 1]` |
| disabled but sub-keys present | loads; keys inert (`_occ_cos=1.0, _occ_strength=0.0`, never read) — documented behaviour |
| `visual_value_mode: presence` | `ValueError: must be 'sum' or 'clamp'` |

No silent defaulting anywhere on this surface. The base `default.yaml` ships the two sub-keys commented out (`default.yaml:246-247`), so a config enabling occlusion **cannot** inherit them silently — it must declare them (H, I, J do).

### 3. obs ↔ noise width sync (this profile's specific charge) — ✅ verified empirically

Mechanism: `get_observation_breakdown` (`sensor.py:482`) computes Olfaction as `(2r²+2r+1) × V` with `r = olfactory_grid_range` (`sensor.py:504-507`); `apply_perceptual_noise` (`sensor.py:377`) re-derives per-element sigma vectors from that same breakdown by **name-keyed** lookup into the padded per-modality arrays (`sensor.py:386-402`) — so widths track automatically *if* breakdown and `get_observation` emit identically. That was not taken on faith:

- **Width:** all 12 combinations of `olfactory_grid_range ∈ {0,1,2,3}` × `visual_sensor_range ∈ {0,1,2}` instantiated and reset: `sum(breakdown.values()) == obs.shape[-1]` in every case (27 → 243 dims; Olfaction 5/25/65/125, Visual 8/40/104).
- **Slice targeting:** at 4 combos × 5 modalities each, exactly one modality was given a large constant sigma (all others `none`) and the noisy observation was compared element-wise against the clean one from the same state: **the perturbed indices equal the intended slice exactly, 20/20** — e.g. at `r_olf=3, r_vis=2`, Olfaction noise lands on `[3:128]` and Visual on `[139:243]`, nothing else. No mis-slice at any combination.
- **Omission fails loudly:** a *standalone* config whose noise block omits a modality present in the breakdown crashes with `KeyError: 'Interoceptive Nociception'` at the first noisy observation — the loud failure ENVIRONMENT_SUMMARY FAQ §10 promises. A *layered* config cannot even reach that state: the base's full 10-modality block survives the deep-merge underneath any partial override.
- **Order:** `default.yaml`'s modality order (injury → … → location) matches the breakdown emission order; the lookup being name-keyed means order is not load-bearing for noise application.
- **Padding:** 10 modalities against the fixed 13-slot pad (`config_loader.py:1632`) — no new modality was added by this change, so no padding risk.
- The clip-min/max concatenation guards with `if name in modality_map` (`sensor.py:418-419`) while the sigma loop uses a bare lookup (`sensor.py:395`) — an asymmetry, but pre-existing and already recorded in `10_perceptual_noise.md` §Suspected Bugs; the bare lookup fails first, so no silent path exists.

### 4. The eleven live arm configs — ✅ each arm is exactly its header

Every arm was resolved through the real loader and flattened-diffed against its **named control** (per-entity lists keyed by name/tag, so an entity-order change would surface):

| Arm vs control | Differing keys (complete list) | obs (header / measured) |
|---|---|---|
| B vs A | `sensory.olfactory_grid_range: 0→1` | 47 / 47 |
| C vs A | `sensory.visual_sensor_range: 0→2` | 123 / 123 |
| D vs C | `sensory.visual_blur_enabled: false→true` | 123 / 123 |
| D′ vs D | `sensory.visual_blur_anisotropy: 3.0→1.0` | 123 / 123 |
| E vs C | `sensory.visual_value_mode: sum→clamp` | 123 / 123 |
| F vs C | `visual_vector_size 8→1` + all 7 entities' `visual_properties`/`_std` reshaped to `[1.0]`/`[0.0]` + background zeroed — the single manipulation "identity removed", carried across the keys that encode it | 32 / 32 |
| G vs C | = F's keys + `visual_value_mode: clamp` (stated: identity **and** count removed) | 32 / 32 |
| H vs C | `visual_occlusion_enabled` + `cone_deg 5.0` + `strength 1.0` + `blocks_sight` stamped on all 7 entities (`true` only for rock, tree) | 123 / 123 |
| I vs C | same as H with `cone_deg 15.0` | 123 / 123 |
| J vs C | = G's keys + H's keys, nothing more (stated: "every weakening stacked") | 32 / 32 |

No arm carries any unstated difference — body, scene, noise profile, behaviour-measure blocks are identical to control in every pair.

- **Generator fidelity:** `generate_weakened_vision_arms.py`'s `build()` was re-executed in memory against **today's** `default.yaml` and dict-compared to the six on-disk generated files: **all six MATCH** — no drift between generation time and now, and no hand-edits.
- **Live-run ground truth (not a re-derivation):** the 11 WandB run dirs from 2026-08-26 (`wandb/run-20260826_*`, hosts docker-106/108/110/112/114) map one-to-one onto the 11 arm configs via their saved launch args; all use the same agent config and the config-owned seed 42 (`configs/train/default.yaml:83`, no `--seed` override in any launch line). The trainers' own logged observation breakdowns match prediction where checked (A=27, B=47 with Olfaction=25, E=123 with Visual=104, F=32 with Visual=13).
- Cross-config coherence (checklist 6): single swept variable per pair ✅, matched seed ✅, arm identity encoded in the config filename / WandB group ✅.

### 5. `configs/environment/default.yaml` — ✅

- All seven new `sensory.*` keys sit in the `sensory:` block (`default.yaml:206,222-228,238,245`) with correct types and explanatory comments; the occlusion sub-keys are present but commented out — the documented conditional pattern.
- **The `sensor_radius` vs `olfactory_grid_range` note (`default.yaml:193-205`) matches the code:** `sensor_radius` is the field's reach — the `dist <= radius` mask inside `_sense_olfaction_at` (`sensor.py` via `params.sensor_radius`, and at 20 it exceeds any distance on a 10×10 grid, so it never binds); `olfactory_grid_range` is the sampling diamond in `sense_olfaction_cells` (`sensor.py:44-63`). The dimension formula in the comment matches the breakdown (`sensor.py:504-507`).
- Read discipline: all eight keys via `get_mandatory` (`config_loader.py:1010-1024,1559-1563`) ✅.
- Registry: rows for the four high-impact new keys + two dated 2026-08-21 change-log entries exist in `CONFIG_CRITICAL_SETTINGS.md:15-18,26-27` — the same-commit logging protocol was followed ✅.
- Statics (recompile risk, checklist 3): `olfactory_grid_range`, `visual_blur_enabled`, `visual_value_mode`, `visual_occlusion_enabled` are `pytree_node=False` (`state.py:268,270,282,285`); the continuous blur knobs and `visual_occlusion_cos`/`strength` are traced leaves — matching the "sweep the angle freely" claim in CONFIG_GUIDE §3.8. The arms vary static fields **across separate runs**, each compiling once — expected, not a hazard. The curriculum modality fingerprint carries the new statics in both trainers (`train.py:810-814`, `dreamer_srl_main.py:802-806`).

### 6. `decay_power` on-source blast radius — ✅ list complete for active configs (🟢 scoping note above)

The rule change (`sensor.py:19-20`: literal `2.0` → `1/(0.5^γ)`) is bit-identical at the shipped `decay_power: 1.0` — every config inheriting `default.yaml` is untouched. Grep of every `decay_power:` in the repo confirms the registry's eleven:

- `configs/continual/nmn_double_return_stages/{01,03,05}_active_predator.yaml:281`, `{02,04}_passive_predator.yaml:307` (5)
- `configs/verification/observability_gates_S{1,2,3,4}.yaml:269` (4)
- `configs/verification/olfaction_parity_{neutral,predator}.yaml:57,64` (2)

— exactly the non-archived set carrying `2.0`; these read **4.0 instead of 2.0 on-source**. No other value ≠ 1.0 exists outside `archive/` and tests. The additional archive/test exposure is enumerated in the 🟢 finding; nothing in the test suite asserts on-source olfactory observation values, so all gates stay green (the unified parity fixtures compare state and info fields, not observations — `tests/env/test_unified_parity.py:1-12`).

## Checklist

- [x] (1) Observation ↔ Noise Modality Consistency — ✅ verified empirically (12 width combos, 20 slice probes, loud KeyError on standalone omission)
- [x] (1.5) Behavior-measures bush presence — N/A (`behavior_measures.enabled: false` in every arm)
- [x] (2) Mandatory-Key Discipline — ✅ all 8 keys `get_mandatory`; conditional pattern correct; per-entity keys follow the established optional pattern (🟢 noted)
- [x] (3) Static-Field & JIT Recompile Risk — ✅ statics as documented; cross-arm static variation is separate-runs, expected
- [x] (4) Known Latent-Bug Recurrences — ✅ registry grepped (`KNOWN_BUGS.md:188` typo'd-modality row is FIXED and its fix verified live; `overeating_death: false` and all body systems enabled in every arm, so neither latent env quirk at `KNOWN_BUGS.md:206,312` is triggered)
- [x] (5) Schema Padding & Modality-Count — ✅ 10 modalities vs 13-slot pad; no new modality
- [x] (6) Cross-Config Coherence (sweep) — ✅ single-variable diffs, matched config-owned seed 42, arm identity in filename/group

## Reproduction

Probe scripts (temp, not committed): `/tmp/cfg_audit/{load_all,diff_arms,regen_check,noise_sync,cond_mandatory,followup,followup2}.py`, run with the project interpreter under `JAX_PLATFORMS=cpu`; baseline comparison in a throwaway worktree at `4b0b4af` (removed after the audit).

**Full unified-parity gate** (`tests/env/test_unified_parity.py`, all experiment + continual + verification configs vs pre-refactor fixtures): **34 passed, 304 skipped, 0 failed** (2026-08-26, CPU backend). The 304 skips are configs without a pre-refactor fixture — by that test's own design ("stale before the refactor"), not caused by this change.

## Conclusion

**Safe to leave the eleven live runs running — no Critical findings; 2 Moderate items (a missing load-time range check, and a pre-existing visual-parity gate covering only 3 of its 8 configs) are fix-when-convenient, plus 4 Low notes.**

Audited by: env-config-reviewer
