---
title: "v3.0 config-system overhaul — source-code correctness review"
topic: environment
status: active
reviewer: code-reviewer
created: 2026-06-20
last_updated: 2026-06-20
audited_doc: "git diff 871f146..HEAD -- src/ (sensor.py, config_loader.py, state.py, core.py, dreamer_srl_main.py)"
---

# v3.0 Config-System Overhaul — Source-Code Review

## Verdict (plain language)

**PASS WITH CONCERNS.** The make-or-break property of this change set — that every
config written before v3.0 produces a *byte-for-byte identical observation* after the
refactor — holds, and I verified it three independent ways (reading the code, running the
project's parity tests, and running my own from-scratch reconstruction of the random-number
streams). The new feature is "give every entity a configurable appearance vector, and
optionally let that appearance vary a little from episode to episode"; the default appearance
is the same hard-coded one-hot pattern the old code used, and the optional jitter is switched
off by default. With jitter off, the new per-episode random draw is wired onto its **own
independent random-number stream** (a `fold_in` with a fresh constant), so it does not
disturb any of the existing draws that seed-locked baselines depend on. I confirmed the
existing olfactory/food-smell draws come out bit-identical with the new code in place.

The one material problem is **not in `src/` but in the test that is supposed to guard the
src parity claim**: commit `52c89c0` renamed the five live curriculum config files (e.g.
`00-forage_5x5.yaml` → `00-static_predator_5x5.yaml`) but did **not** update
`tests/env/test_visual_parity.py`, which still lists the old filenames. Because that test
"skips" any config file it can't find, the parity gate for **exactly the five configs about
to be trained** silently passes without running. The src code is correct; the safety net
over it is currently disconnected. I re-ran the parity check manually against the renamed
files and it passes — so this is a test-wiring gap, not a parity break — but it must be
fixed before relying on the gate.

## Test results (conda interpreter)

| Suite | Result |
|---|---|
| `tests/env/test_visual_parity.py` + `test_unified_parity.py` + `test_visual_sampling.py` + `test_per_episode_sampling.py` | 48 passed, 103 skipped, **0 failed** |
| Full `tests/env/` + `tests/environment/` | 195 passed, 197 skipped, **0 failed** |
| `tests/algorithms/dreamer_srl/test_lax_scan_train.py` | **collection ERROR** (stale config path — see F3) |

Manual parity verifications (my own, outside the test suite):
- Default-when-absent visual_properties returns exactly `one_hot(channel, 8)` for all 8 channels; default std returns exactly `[0]*8`. ✅
- All 5 live (renamed) `basic/*` configs: `*_visual_property_sampled == *_visual_property` bit-for-bit at std=0, V=8. ✅
- Olfactory draws byte-identical to an independent reconstruction across seeds {0,1,42,12345} → the new visual stream does not perturb existing draws. ✅
- `test_initial_state_ranges.py::test_legacy_parity_range_reproduces_old_behaviour` reproduces the old `[max/2,max]` / `[0,max/2]` body draws bit-for-bit with the same body-key split. ✅

## Parity verdict (the make-or-break)

**PARITY HOLDS.** Three checks pass:
1. **Default path** — absent `visual_properties` → exact one-hot of the class channel; absent
   `visual_properties_std` → exact zeros. (`config_loader.py:_read_visual_properties`,
   `_read_visual_properties_std`.)
2. **Sampling at std=0** — `mean + std*noise` with `std=0` is exactly `mean`; `jnp.clip(mean, 0.0, None)`
   does not alter a non-negative one-hot. Sampled fields equal mean fields bit-for-bit. (`core.py:988-1018`.)
3. **PRNG draw-order** — the new draw uses `jax.random.fold_in(property_key, 0x7150A1)`, a
   derived key that does **not** consume or advance `property_key`. The existing olfactory
   `jax.random.split(property_key, 4)` (core.py:953) and `animal_episode_key =
   fold_in(property_key, 0xAE1)` (core.py:804) are untouched. The two fold-in constants
   (`0x7150A1` vs `0xAE1`) differ, so no stream collision. Verified empirically.

## Findings

| Severity | File:line | Issue | Suggested fix |
|---|---|---|---|
| 🟡 P1 | `tests/env/test_visual_parity.py:53-57` | `_PARITY_CONFIGS` lists the **pre-rename** filenames (`00-forage_5x5`, `01-slowPred_5x5`, `02-fastPred_8x8`, `03-multiPred_10x10`, `04-keenPred_10x10`). Commit `52c89c0` (in range) renamed all five to `00-static_predator_5x5` … `04-far_sight_predator_10x10`. The test's `if not os.path.exists(cfg_path): pytest.skip(...)` then **silently skips** all 5 → the byte-parity gate over the live curriculum configs is a no-op (shows green). | Update the 5 paths/labels in `_PARITY_CONFIGS` to the renamed files; **regenerate the pinned fixtures** for them (`--gen-fixtures` on the pre-change commit, or accept the auto-generate-on-missing path). Outside `src/` so `developer` applies; flagged here because it directly undermines the src parity claim. |
| 🟡 P1 | `tests/algorithms/dreamer_srl/test_lax_scan_train.py:58-59` | References `configs/dreamer_srl/01_food_only{,_smoke}.yaml`; the dir moved to `configs/models/dreamer_srl/` in `f0fe297` (pre-range). The bad path is read at module scope (line ~74 `get_mandatory('algo.horizon')`), so the file **fails to COLLECT** (`ValueError`), interrupting any pytest run that includes it. Confirmed. Same stale path also in `test_buffers.py:333` and `bench_sps.py:54-55`, but those are inside functions / not collection-time so they don't error today. | Repoint to `configs/models/dreamer_srl/...`. The root-move predates this range, so this is pre-existing breakage surfaced by the reorg, not a regression introduced here — but it blocks the dreamer test collection now and should be fixed. |
| 🟢 P2 | `src/environment/sensor.py:411` | Olfactory viz label list carries **8 labels** (`['GRS','SND','PLN','FOD','DNG','PRD','RCK','NEU']`) but the olfaction vector is **5-dim** (`sensory.vector_size: 5`). Labels 5-7 are never rendered. **Pre-existing** (present at base commit `871f146`). Also: this diff fixed the NEU/RCK order on the **Visual** label list but left the **Olfactory** list with the old `...,'NEU','RCK'` order (harmless because indices 6/7 aren't shown, but inconsistent). | Viz-only, no observation impact. Trim Olfactory labels to the first 5 (`['GRS','SND','PLN','FOD','DNG']`) for clarity; optionally align the trailing order with the Visual fix. Not a blocker. |
| 🟢 nit | `src/environment/config_loader.py:33-98` (extends) | Two clarity nits already documented in `config_extends_layering.md` (missing-top-level-file silent-miss; no type-guard on `extends:` values). Unchanged since that review. | Carry over — no new action. |

## NEU/RCK label fix — CONFIRMED CORRECT

Encoding is `predator=5, rock=6, neutral=7` (`config_loader.py:108
ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator": 5, "neutral": 7}`; obstacle/rock → channel 6 in
`load_env_params`). The Visual viz labels at index 5/6/7 are therefore `PRD/RCK/NEU`. The
change `['…PRD','NEU','RCK'] → ['…PRD','RCK','NEU']` (`sensor.py:447`) **matches the true
encoding** — the old order was the bug, the new order is right.

## `extends:` layering — CONFIRMED CORRECT (independent re-check)

Deep-merge precedence (later base > earlier base > … child wins via final
`merged.merge(Config(raw))`), cycle detection (`frozenset` of abspaths threaded by value,
checked at entry), `extends` key stripped via `raw.pop("extends", None)` before reaching env
params, and the standalone (no-`extends:`) path returns `Config(raw)` directly → byte-identical
to the old loader. The `dreamer_srl_main.py` swap from `Config.load_yaml` → `load_env_config`
is correct and consistent (honours `extends:` for stage/env configs; standalone configs load
as before). No obs-dim assumptions were changed in the dreamer driver.

## Conventions audit

| Convention | Status | Note |
|---|---|---|
| Pytree / immutability | ✅ | New state fields set via `EnvState(...)` constructor in reset and `_replace`-style returns in step; no in-place mutation. `_sample_visual_property` uses `.at[idx].set(...)` (functional scatter). |
| JIT recompilation | ✅ | `visual_vector_size` is `pytree_node=False` (static, shape-determining) — verified at runtime. Visual property arrays (`[N,V]`) are `pytree_node=True` (traced data). No new Python branch on a traced value. `test_no_recompile.py` in the green suite. |
| vmap / batch | ✅ | New `[N,V]` arrays carry the same leading-N convention as existing per-entity arrays; renderer label code is host-side (`build_sensory_viz` imports `numpy`), not vmap'd. |
| PRNG threading | ✅ | New draw on an independent `fold_in(property_key, 0x7150A1)` stream; main key advance unchanged; constants distinct from `0xAE1`. Reset and step both use the same constant for the visual stream (consistent). Verified bit-parity of existing streams. |
| Sensor / obs-breakdown sync | ✅ | `get_observation_breakdown` Visual dim = `num_vis_cells * visual_vector_size` (sensor.py:382) matches `sense_visual` output width `[num_cells, V].flatten()`. Olfaction breakdown still `res_property.shape[-1]` (5) — unchanged. Noise modality_map path unchanged. |
| Config protocol (no-fallback) | ✅ with 1 sanctioned default | `visual_vector_size` uses a documented read-site default of 8 (`config_loader.py`, the one permitted default per the plan, to preserve parity for ~86 configs that lack the key). At V≠8 the loader *raises* if `visual_properties` / `visual_background_properties` are absent — strict. Init-range keys are conditional-mandatory (required only when the matching `random_start_*` flag is true). |
| `property` vs `properties` | ✅ | New code reads canonical plural `visual_properties` / `visual_properties_std`; from the correct raw dict (`dist_source` for animals — verified). |
| `terminated`/`done` desync | ✅ N/A | Untouched. |

## One-line conclusion

The v3.0 source changes are byte-parity-correct and PRNG-safe — verified three ways — but
the parity test that guards them silently skips the five live curriculum configs (renamed
out from under it), and a reorg orphaned a dreamer test's config path; fix both before
leaning on the test gate.

Reviewed by: code-reviewer
