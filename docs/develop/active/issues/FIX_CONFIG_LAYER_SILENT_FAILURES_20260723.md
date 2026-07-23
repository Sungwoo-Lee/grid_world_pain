---
title: "Fix config-layer silent failures (Track A — 4 P1 bugs from the 2026-07-23 diagnosis)"
topic: issues
status: active
created: 2026-07-23
last_updated: 2026-07-23
---

# Fix config-layer silent failures — Track A (config layer)

> **Status**: PLANNED
> **Opened**: 2026-07-23
> **Related**: [[review_full_diagnosis_20260723]] (P1 table rows 1–4) · [[findings_config_loader]] (Findings 1–3) · [[findings_configs]] (P1-1) · [[KNOWN_BUGS]] (rows P1 #1, #2, #3, #8)

---

## Context

The 2026-07-23 full-codebase diagnosis found that **the config layer fails silently in exactly the places this project's experiments depend on**: perceptual noise (the central experimental manipulation) and the animal scene (what predators/rabbits the agent faces). Four separate config bugs let a run train on a *different world or a different noise condition than the config claims*, with no error — only, at best, a warning nobody reads. This plan fixes all four surgically, each with a regression test that fails on today's code and passes after the fix.

The four bugs in plain language:

1. **Train and eval can silently run two different worlds.** The shared base config (`default.yaml`) now describes its animals in the new "unified entities" format. When an *old-format* config (one that lists its predators/rabbits the pre-v2.0 way) is loaded for **training**, the base's new-format animal list silently overrides the config's own animals — the run trains on the base scene, not the authored scene. The **evaluation** script loads the same file without the base underlay, so it sees the true old-format scene. Result: training and evaluation disagree about what world the agent is in. Only a misleading deprecation warning fires. 79 archived old-format configs are affected on rerun; **zero currently-active configs** use the old format (verified by grep), so nothing running right now is corrupted — the risk is on any rerun of an archived config.

2. **A typo in a noise "mode" silently disables that channel's noise.** Writing `mode: "state-dependent"` (hyphen) or any unrecognised string instead of `state_dependent` maps to "noise off" with no error — the experiment cell trains as if the manipulation ran when it never did.

3. **A typo in a noise modality name silently drops that channel's noise.** Writing `olfactory:` instead of `olfaction:` makes the whole entry vanish from the noise config; the channel falls back to the base's noise-free default. Silent again.

4. **A live noise config's header lies about what it does.** The hardest curriculum level, `05-sensory_noise_10x10.yaml`, says in its header that interoception (the "am I injured?" signals — satiation and both pain channels) is "kept CLEAN" so the injury-gating stays reliable. But the file only overrides two sub-keys per channel; the actual noise amount (`sigma = 0.1`, ten times the config's own "tiny" benchmark) deep-merges through from the base on all three interoceptive channels. Past runs that used this config (its lineage trained a 10M-episode run on 2026-07-03) therefore measured a *noisier* interoceptive condition than the header documents. This plan makes the config match its documented intent; **that changes the config's meaning going forward**, so it is flagged for the user's post-hoc review below.

### Decision on bug 1 semantics (reversible default — flagged for post-hoc review)

**The user pre-approved this fix family but is unavailable mid-flow.** Bug 1 has three candidate fix semantics; I chose one and record it here as a **reversible default** for the user to confirm later:

- **CHOSEN — legacy-scene precedence at all load sites.** When a config presents old-format `predators:`/`neutral_animals:` sections, those *define the scene*, even if a new-format `entities:` list arrived only via the base-config underlay. Concretely: flip the precedence in `_load_animals` so old-format sections win when present; keep a (now-accurate) deprecation warning telling the author to migrate. **Rationale:** this makes training agree with evaluation (which already uses the legacy scene) and lets the 79 archived configs rerun on the world they actually describe — a strictly-better outcome than today, achieved with a one-condition change and **no config edits**.
- **Rejected — hard-error on "both schemas present."** The base *always* injects `entities:`, so every legacy rerun would raise — this blocks the reversible-review the user wants and is maximally disruptive.
- **Rejected — migrate all 79 archived configs to `entities:`.** This is the escalation trigger (editing 79 configs / config-system redesign); out of scope for a bug-fix plan.

**Why reversible:** the chosen fix deletes nothing and adds no config. If the user later prefers strict-error or a migration, the precedence flip is trivially revertible, and once a config is migrated to `entities:` (legacy sections removed) the behaviour flips back to the entities path naturally.

**Escalation check:** the chosen fix is a single-condition change in one function — it does **not** require a config-system redesign or editing the 79 configs, so it stays inside bug-fix scope. No escalation needed.

## Analysis

### Bug 1 — root cause

`src/environment/config_loader.py:429-443` (`_load_animals`) picks the scene schema by:
```python
has_entities = config.get('environment.entities') is not None
if has_entities:
    ...entities path...   # ignores predators:/neutral_animals:
else:
    ...legacy path...
```
Under `train.py`, the load order is `get_default_config()` (which merges in `default.yaml`, carrying `environment.entities:` at line 58) **then** `config.merge(user_config)`. A legacy user file has no `entities:` key, so the base's `entities:` **survives the deep-merge**; the merged config now holds *both* `entities:` (from base) and `predators:` (from user). `has_entities` is `True` → entities path → the user's animals are ignored. The `DeprecationWarning` text ("Config has both…") is misleading because the *user's file* did not have both — the merge created the conflict.

`eval_rollout.py:849` loads via bare `load_env_config(config_path)` with **no** base underlay → the merged config has no `entities:` → legacy path → true legacy scene. Hence train/eval divergence.

Verified in the diagnosis (executed): `get_default_config(); cfg.merge(load_env_config('.../archive/2X2_area.yaml')); _load_animals(cfg)` yields the *base* scene tags, while the merged config holds both `entities` (base) and `predators` (user). `default.yaml` itself carries **only** `entities:` — no top-level `predators:`/`neutral_animals:` (verified: the `hiding_predator` at `default.yaml:40` is a `resources:` entry, not a legacy predator). This is what makes "legacy wins when present" safe: the base never supplies legacy sections, so they are always an unambiguous signal of user intent.

### Bug 2 — root cause

`config_loader.py:1509-1510`:
```python
def _parse_mode(s):
    return 2 if s == 'state_dependent' else 1 if s == 'constant' else 0
```
Any string that is neither exact match → `0` (= noise off). No validation. Violates the project no-fallback-defaults rule.

### Bug 3 — root cause

`config_loader.py:1512-1542`: every comprehension filters `if k in _YAML_KEY_TO_SENSOR_NAME`. A modality key not in that dict (`olfactory`, `visual_noise`, …) is silently dropped from all five noise arrays. No validation.

### Bug 4 — root cause

`configs/environment/experiment/basic/05-sensory_noise_10x10.yaml` overrides only `mode` + `injury_noise_scale` on `satiation`, `interoceptive_nociception`, `extero_nociception`. `sigma` is **not** overridden, so it deep-merges from `default.yaml` (`sigma: 0.1` on all three — `default.yaml:283,289,295`). The header claims these channels are "kept CLEAN". Loader-proven in the diagnosis: all three merge to `{'mode':'constant', 'sigma':0.1, ...}`.

## Implementation Plan

### Design

- Bugs 1–3: change validation/precedence in `_load_animals` and `_parse_noise_config`. No new config keys; no new mandatory keys.
- Bug 4: edit the YAML to add explicit `sigma: 0.0` to the three interoceptive overrides (match the documented "CLEAN" intent), and tighten the header wording so it is unambiguous.
- Bugs 2 & 3 raise `ValueError` per the no-fallback rule; the messages name the valid options so an author can self-correct.

### File Changes

#### `src/environment/config_loader.py` — Bug 1 (lines 429–443, minimal precedence flip)

The change is a **single guard condition** plus moving the (now-accurate) warning into the legacy branch. Both large branch bodies stay byte-identical in place.

```python
# BEFORE (429–443):
    has_entities = config.get('environment.entities') is not None
    has_legacy_predators = config.get('environment.predators') is not None
    has_legacy_neutrals = config.get('environment.neutral_animals') is not None

    if has_entities:
        # CP3 path — parse unified entities: list directly
        if has_legacy_predators or has_legacy_neutrals:
            warnings.warn(
                "Config has both 'environment.entities:' and legacy "
                "'environment.predators:'/'environment.neutral_animals:'. "
                "The unified 'entities:' schema takes precedence; legacy sections are ignored.",
                DeprecationWarning,
                stacklevel=3,
            )
        raw_entities = config.get('environment.entities') or []
        ...entities body...
    else:
        # Legacy path: re-project predators (hunt) + neutrals (wander)
        raw_predators = config.get('environment.predators') or []
        ...legacy body...

# AFTER:
    has_entities = config.get('environment.entities') is not None
    # bool(...) → present AND non-empty; the base default.yaml never supplies legacy
    # sections, so their presence is an unambiguous signal of user intent.
    has_legacy = bool(config.get('environment.predators')) or \
                 bool(config.get('environment.neutral_animals'))

    if has_entities and not has_legacy:
        # CP3 path — parse unified entities: list directly
        raw_entities = config.get('environment.entities') or []
        ...entities body (unchanged)...
    else:
        # Legacy path (also runs when NEITHER schema is present → empty scene, unchanged).
        # Legacy-scene precedence: when the user file authored legacy sections, they define
        # the scene even if an `entities:` list arrived via the default.yaml underlay under
        # train.py. This keeps train.py and eval_rollout.py agreeing on legacy configs.
        if has_entities:
            warnings.warn(
                "Config presents legacy 'environment.predators:'/'neutral_animals:' sections "
                "alongside an 'environment.entities:' list (typically inherited from "
                "default.yaml under train.py). The legacy scene takes precedence. Migrate this "
                "config to the unified 'entities:' schema.",
                DeprecationWarning,
                stacklevel=3,
            )
        raw_predators = config.get('environment.predators') or []
        ...legacy body (unchanged)...
```
Notes for the implementer: the old warning block inside the entities branch becomes unreachable (that branch now runs only when `not has_legacy`) — delete it. The `has_legacy_predators`/`has_legacy_neutrals` locals are replaced by `has_legacy`; check for other uses before removing (grep confirms they're only used here).

#### `src/environment/config_loader.py` — Bug 2 (lines 1509–1510)

```python
# BEFORE:
    def _parse_mode(s):
        return 2 if s == 'state_dependent' else 1 if s == 'constant' else 0

# AFTER:
    _VALID_NOISE_MODES = {'none': 0, 'constant': 1, 'state_dependent': 2}
    def _parse_mode(s):
        if s not in _VALID_NOISE_MODES:
            raise ValueError(
                f"Strict Config: unknown perceptual-noise mode {s!r}. "
                f"Must be one of {sorted(_VALID_NOISE_MODES)}."
            )
        return _VALID_NOISE_MODES[s]
```
Note: keep `'none'` valid — `default.yaml` and child configs rely on the read-site `.get('mode', 'none')` for un-overridden leaves (Bug 3's fix does not change that). This preserves current valid-config behaviour while rejecting typos.

#### `src/environment/config_loader.py` — Bug 3 (top of `_parse_noise_config`, ~line 1507)

Add one validation block right after `modalities_cfg` is read, before the comprehensions:

```python
# AFTER (insert after `modalities_cfg = config.get('perceptual_noise.modalities') or {}`):
    unknown = set(modalities_cfg) - set(_YAML_KEY_TO_SENSOR_NAME)
    if unknown:
        raise ValueError(
            f"Strict Config: unknown perceptual-noise modality key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(_YAML_KEY_TO_SENSOR_NAME)}."
        )
```
The existing `if k in _YAML_KEY_TO_SENSOR_NAME` filters may stay (now always-true, harmless) or be removed for clarity — implementer's choice; do not otherwise touch the comprehensions.

#### `configs/environment/experiment/basic/05-sensory_noise_10x10.yaml` — Bug 4

Add explicit `sigma: 0.0` to the three interoceptive overrides so the merged noise matches the documented "CLEAN" intent:

```yaml
# BEFORE:
    satiation:
      mode: "constant"
      injury_noise_scale: 0.0
    interoceptive_nociception:
      mode: "constant"
      injury_noise_scale: 0.0
    extero_nociception:
      mode: "constant"
      injury_noise_scale: 0.0

# AFTER:
    satiation:
      mode: "constant"
      sigma: 0.0          # explicit — was inheriting default.yaml sigma 0.1 (see FIX_CONFIG_LAYER_SILENT_FAILURES_20260723)
      injury_noise_scale: 0.0
    interoceptive_nociception:
      mode: "constant"
      sigma: 0.0
      injury_noise_scale: 0.0
    extero_nociception:
      mode: "constant"
      sigma: 0.0
      injury_noise_scale: 0.0
```
Also tighten the header (lines 10–11 / 38): keep "interoception kept CLEAN" but it is now *true*. No wording change strictly required once sigma is 0.0; optionally add a one-line footnote: `# (sigma explicitly 0.0 — do not rely on inherited defaults for the clean channels).`

**Meaning-change note (for the record and the user's review):** past runs using this config (or its `06-…` lineage, e.g. the 2026-07-03 10M-episode noise run) trained with `sigma = 0.1` on satiation + both nociception channels — a *different, noisier* condition than the header documented. This edit changes the config's behaviour **going forward** to the documented clean-interoception condition. It does not retroactively change past results; any comparison that mixes pre-fix and post-fix runs of this level is comparing two different conditions.

#### `docs/environment/02_config_schema.md` — doc sync (Maintenance Contract)

The schema-of-record documents the entities/legacy precedence and the noise block; the fix changes both semantics, so update in the same change:
- Line ~46: `entities: [ list — NEW unified schema (CP3); takes precedence when present ]` → note that **legacy `predators:`/`neutral_animals:` sections take precedence when the user file authored them** (they win over an `entities:` list inherited via the base underlay).
- §Unified Animal Entity (line ~25) + `perceptual_noise` (line ~75): add a one-line note that unknown noise `mode` strings and unknown modality keys now raise `ValueError` (no silent fallback).

(No change needed to `docs/environment/CONFIG_GUIDE.md` — it documents the list-replace footgun, not the animal-schema precedence or noise-mode validation. Mention only if the implementer finds a now-inaccurate sentence there.)

#### New mandatory config keys

**None.** All four fixes are validation/precedence/value changes. No `get_mandatory` call sites added; no new YAML keys introduced. (Bug 4 makes an existing `sigma` leaf explicit; it is already a documented noise leaf.)

#### Scripts dependency map

No files under `scripts/` are added, moved, renamed, or deleted → `SCRIPTS_DEPENDENCY_MAP.md` needs no update.

### Regression tests — one per bug (all must FAIL on pre-fix code)

New file: **`tests/env/test_config_layer_silent_failures_20260723.py`**. Follow the style of `tests/env/test_extends_layering.py` (repo root on `sys.path`; scratch YAML → `tmp/`).

- **`test_bug1_train_eval_scene_agree`** — the train-path load (`cfg = get_default_config(); cfg.merge(load_env_config('configs/environment/experiment/archive/2X2_area.yaml'))`) and the eval-path load (bare `load_env_config('.../2X2_area.yaml')`) must produce the **same animal scene**. Assert equality of a scene-identifying quantity from `_load_animals(cfg)` / `load_env_params(cfg)` — the animal-tag tuple or animal-slot count. `2X2_area.yaml` has 5 legacy rabbits + 2 legacy predators, whereas `default.yaml`'s `entities:` yields ≤2 rabbits + ≤2 predators, so the two paths differ **pre-fix** (train sees the base scene) and are **equal post-fix** (both see the legacy scene). Use the exact reproducer from `findings_config_loader.md` Finding 1.
- **`test_bug2_unknown_noise_mode_raises`** — build a `Config` with `perceptual_noise.enabled: true` and one modality (`injury`) set to `mode: "state-dependent"` (hyphen typo); assert `_parse_noise_config(config)` (or `load_env_params`) raises `ValueError`. Pre-fix: no error, that modality's `noise_modes` entry is `0`.
- **`test_bug3_unknown_modality_key_raises`** — same setup but with a typo'd modality key `olfactory:` (valid key is `olfaction:`); assert `ValueError` whose message lists the valid keys. Pre-fix: the key is silently dropped (`noise_modality_order` excludes it, no error).
- **`test_bug4_interoception_clean_in_05_config`** — `load_env_params(load_env_config('configs/environment/experiment/basic/05-sensory_noise_10x10.yaml'))`; locate `satiation`, `interoceptive_nociception`, `extero_nociception` via `noise_modality_order` and assert their `noise_sigmas` entries are `0.0`. Pre-fix these are `0.1` (fails); post-fix `0.0` (passes).

Add a valid-config guard so the new strictness does not regress real configs:
- **`test_valid_noise_configs_still_load`** — assert `load_env_params(load_env_config('configs/environment/experiment/basic/05-sensory_noise_10x10.yaml'))` succeeds (exercises the `state_dependent`/`constant`/`none` modes and all valid modality keys through the new validators). Optionally loop a couple more noise configs.

## Blast radius (Bug 1)

Which load sites change behaviour, and for which configs:

| Load site | Underlays `default.yaml`? | Behaviour change for **legacy** configs | Behaviour for **modern** (`entities:`) configs |
|---|---|---|---|
| `train.py:389` (single `--config`) | yes (`get_default_config()`) | **changes** → now uses the legacy scene (was: base `entities:`) | unchanged |
| `train.py:198` (continual stage) | yes | **changes** (same) | unchanged |
| `dreamer_srl_main.py:539` (single) | yes | **changes** (same) | unchanged |
| `dreamer_srl_main.py:133` (continual stage) | yes | **changes** (same) | unchanged |
| `eval_rollout.py:849` (+ probe `:753`) | no (bare `load_env_config`) | **unchanged** (already used the legacy scene) | unchanged |

**Net effect:** the fix makes the four training/underlay sites *agree with* the two eval sites on legacy configs. Modern configs (which have `entities:` and no legacy sections) are untouched everywhere — `has_legacy` is `False`, so they still take the entities path byte-identically.

**Active-config check (CORRECTED in verification — the original claim below was wrong):**

> ~~`grep -rlE "^\s*predators:" configs/ | grep -v archive` → **0 files** … no currently-active or currently-running config changes behaviour — the fix only affects reruns of archived configs (making them correct).~~

**Correction (found-in-verification, 2026-07-23, senior-developer):** the grep above actually returns **11 files**, not 0 — all tracked and committed since May 2026, so present when the plan/report ran the grep. Legacy-format configs are **not** confined to `archive/`. The true active-config breakdown:

- **6 verification configs** — `configs/verification/observability_gates_S{1,2,3,4}.yaml`, `olfaction_parity_{predator,neutral}.yaml`. Loaded **bare** (no `default.yaml` underlay) by `test_unified_parity.py`, so `has_entities` is `False` both pre- and post-fix → **behaviour UNCHANGED** in that path. (These are also the 4 pre-existing `test_unified_parity.py` S1–S4 failures, which are unrelated to this fix.)
- **5 continual training-stage configs** — `configs/continual/nmn_double_return_stages/0{1..5}_*_predator.yaml`. Loaded by **`train.py`'s continual path** (`train.py:197-198` deep-copies `base_config` from `get_default_config()`, which carries `default.yaml`'s `entities:`, then merges the stage YAML) → each stage ends up with **both** the inherited `entities:` and its own legacy sections → the precedence flip applies. **Behaviour CHANGES**: verified empirically that pre-fix these train on the base scene (**4 slots: 2 predators + 2 rabbits**) and post-fix on their authored legacy scene (e.g. `01_active_predator` → **3 slots: `full`, `TL`, `BR`**).

So **the fix DOES change training behaviour for an active config family** (the 5 `nmn_double_return_stages` stages) — it corrects a previously-silent mis-training rather than only touching archived reruns. `default.yaml` itself carries no legacy sections and is unaffected. The `test_extends_layering.py` C4 case (`entities: []` suppresses base animals) still holds: `entities: []` → `has_entities=True`, `has_legacy=False` → entities path with an empty list → empty scene, unchanged.

**Meaning-change note for the `nmn_double_return_stages` continual family (mirrors the Bug 4 `05`-config note above):** past continual runs of this family were trained on the shared `default.yaml` base scene (2 predators + 2 rabbits), **not** the authored per-stage scene each config describes. This edit changes their behaviour **going forward** to the authored legacy scene. It does not retroactively change past results, but any comparison that mixes pre-fix and post-fix runs of these stages is comparing **two different worlds** — results are **not comparable across the fix boundary**. Flagged for the user's post-hoc review alongside the `05`-config meaning-change.

## Checkpoints

- [x] After Bug 1: run `tests/env/test_extends_layering.py` — all C1–C9 still green (precedence change must not break the suppress-semantics or worked-example parity).
- [x] After Bug 1: the reproducer from Finding 1 (`get_default_config(); cfg.merge(load_env_config('.../archive/2X2_area.yaml')); _load_animals(cfg)`) now returns the **legacy** scene (5 rabbits + 2 predators), matching the bare-eval load.
- [x] After Bugs 2–3: `load_env_params(load_env_config('.../basic/05-sensory_noise_10x10.yaml'))` still succeeds (valid modes/keys not rejected).
- [x] After Bug 4: the three interoceptive `noise_sigmas` read back as `0.0`; `olfaction`/`visual` still `state_dependent` with their intended sigmas.
- [x] All 5 new tests fail on a clean checkout (pre-fix) and pass after the fix — capture the pre-fix failure output in the Implementation Report.
- [x] Speed: these are load-time/validation changes on a non-hot path (config parse happens once per run). No measurable step-time impact expected; a speed benchmark is **not required** — state this explicitly in the report.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-23

### Plain-language summary

All four config-layer silent failures are fixed and verified. In plain terms: (1) a legacy-format experiment config (the kind that lists its predators/rabbits the pre-v2.0 way) now trains on the *same world it describes*, instead of silently swapping in the shared default scene — the fix makes `train.py`/`dreamer_srl_main.py` agree with `eval_rollout.py` about what animals are in the grid; (2) a typo'd noise `mode` string (e.g. `"state-dependent"` with a hyphen) now raises a clear error naming the valid modes, instead of silently turning that channel's noise off; (3) a typo'd noise-modality key (e.g. `olfactory:` instead of `olfaction:`) now raises a clear error naming the valid keys, instead of silently vanishing from the noise arrays; (4) the hardest curriculum level's config (`05-sensory_noise_10x10.yaml`) now actually keeps the "am I injured?" interoceptive channels noise-free as its header claims — they were quietly inheriting a `sigma=0.1` from the shared base config despite the header saying "kept CLEAN".

### File-by-file summary

1. **`src/environment/config_loader.py`** — Bugs 1–3, plus one **necessary follow-on fix** discovered during testing (see Deviations below):
   - `_load_animals` (~line 429): precedence guard flipped from `if has_entities:` to `if has_entities and not has_legacy:` where `has_legacy = bool(predators) or bool(neutral_animals)`. The old warning (fired inside the entities branch) is now unreachable and was removed; a new, accurate warning fires in the legacy branch only when `has_entities` is also true (i.e. an `entities:` list arrived via the base underlay but the user's own legacy sections win).
   - `_parse_noise_config` (~line 1506): added an `unknown = set(modalities_cfg) - set(_YAML_KEY_TO_SENSOR_NAME)` check that raises `ValueError` naming the valid keys (Bug 3); replaced the silent `_parse_mode` fallback with a `_VALID_NOISE_MODES` dict lookup that raises `ValueError` naming the valid modes for anything not in `{'none', 'constant', 'state_dependent'}` (Bug 2).
   - **Deviation (necessary follow-on, same file, same root cause)**: `load_env_params` (~line 1087) contains a **second, independent copy** of the has-entities-vs-legacy dispatch decision, used only to build the `animal_count_low`/`animal_count_high`/`animal_entry_id` per-episode-count-variance metadata. This second copy was not in the plan's File Changes section (the plan only specified lines 429–443) and was not updated by the minimal guard flip there — leaving it in the OLD (`entities`-always-wins) precedence while `_load_animals()` now uses the NEW (legacy-wins) precedence. On any legacy config where `entities:` also survives the base-underlay merge, this mismatch produces animal-count metadata for a *different* schema than the one `_load_animals()` actually built, which breaks `jax_reset`'s activation-mask broadcast with a shape error (see Deviations section below for the concrete repro). Fixed by mirroring the exact same `has_legacy` check at this second site.
2. **`configs/environment/experiment/basic/05-sensory_noise_10x10.yaml`** — Bug 4: added explicit `sigma: 0.0` to the `satiation`, `interoceptive_nociception`, and `extero_nociception` modality blocks (previously only `mode` and `injury_noise_scale` were overridden, so `sigma` deep-merged from `default.yaml`'s `0.1`). Added an inline comment pointing at this plan doc.
3. **`docs/environment/02_config_schema.md`** — doc sync per the Maintenance Contract: updated the `entities:` top-level line, the "Unified Animal Entity" precedence paragraph, and the noise-mode/noise-modality strictness notes to describe the new legacy-wins precedence and the new `ValueError`-on-unknown behaviour (previously documented as "silently dropped" / unconditional "unified takes precedence").
4. **`tests/env/test_config_layer_silent_failures_20260723.py`** (new) — 5 tests: `test_bug1_train_eval_scene_agree`, `test_bug2_unknown_noise_mode_raises`, `test_bug3_unknown_modality_key_raises`, `test_bug4_interoception_clean_in_05_config` (all confirmed red pre-fix, green post-fix — see below), plus the guard `test_valid_noise_configs_still_load` (green both before and after — it is not a regression test, it just confirms the new strictness doesn't reject the currently-valid `05-sensory_noise_10x10.yaml`).
   - Deviation from the plan's suggested reproducer: for Bug 1 I used `_load_animals(cfg)` directly for both the train-path and eval-path loads (comparing `animal_property.shape[0]` slot count + `animal_tags` tuple), rather than the full `load_env_params(cfg)`. The archived `2X2_area.yaml` fixture is a genuinely pre-v2.0 config missing several unrelated mandatory keys (e.g. `sensory.injury_observable`) that only the `default.yaml` base underlay supplies — calling the full `load_env_params` on the bare eval-path load (no underlay) raises `ValueError` on those unrelated keys before ever reaching the animal-scene comparison. `_load_animals()` only needs `environment.height`/`width` plus the animal sections, so it isolates the precedence logic under test cleanly. This matches the plan's own Finding-1 reproducer, which also calls `_load_animals(cfg)` directly rather than `load_env_params`.
5. **`tests/env/test_entities_schema.py`** (existing file, **not in the plan's File Changes section** — flagged as a deviation): `test_both_schemas_warns_and_prefers_unified` directly encoded the OLD precedence semantics (asserting that `entities:` wins over legacy `predators:`/`neutral_animals:` when both are present in the same file) — exactly the behaviour Bug 1 intentionally inverts. Left unfixed, this existing test would go permanently red for a config-writing pattern (both schemas in one file) that is legal but rare. Renamed to `test_both_schemas_warns_and_prefers_legacy`, updated the docstring and assertions to expect `legacy_wolf` (not `wolf`) in `animal_tags`, and updated the module docstring's item 3. **Not fixed**: two docs (`docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md` and `docs/reviews/env_entities_cp1_verification.md`) still describe the old "unified wins" behaviour by name-checking this test — out of scope for `developer` (docs are `senior-developer`'s to edit); flagging here for follow-up.

### Per-bug red→green confirmation

All 4 regression tests were run and confirmed **FAILED** on the pre-fix code, then confirmed **PASSED** after each corresponding fix landed (and again in the final full run below).

- **Bug 1** (`test_bug1_train_eval_scene_agree`): pre-fix, train-path (`get_default_config()` + merge) saw **4** animal slots (base `entities:` scene) vs eval-path's **7** (true legacy scene) — `AssertionError: Train-path scene (4 slots, tags=('pred','pred','rabbit',...)) disagrees with eval-path scene (7 slots, ...)`. Post-fix: both paths see **7** slots, tags equal.
- **Bug 2** (`test_bug2_unknown_noise_mode_raises`): pre-fix, `mode: "state-dependent"` (hyphen typo) silently parsed with `Failed: DID NOT RAISE <class 'ValueError'>`. Post-fix: raises `ValueError` naming `['constant', 'none', 'state_dependent']`.
- **Bug 3** (`test_bug3_unknown_modality_key_raises`): pre-fix, `olfactory:` (typo for `olfaction:`) silently dropped — `Failed: DID NOT RAISE <class 'ValueError'>`. Post-fix: raises `ValueError` naming the 10 valid modality keys.
- **Bug 4** (`test_bug4_interoception_clean_in_05_config`): pre-fix, `Satiation sigma should be 0.0 ... got 0.10000000149011612` (deep-merged from `default.yaml`). Post-fix: all three interoceptive `noise_sigmas` read `0.0`.

### Test-suite results

1. **New regression file** (`tests/env/test_config_layer_silent_failures_20260723.py`) — `5 passed` (all 4 bug tests + the guard test), `1 warning` (the now-accurate `DeprecationWarning`).
2. **`tests/env/test_extends_layering.py`** (C1–C9) — `7 passed`. No precedence regression on the `entities: []` / omit-`entities:` suppress-semantics.
3. **`tests/env/test_entities_schema.py`** — `5 passed, 2 skipped` (the 2 skips are pre-existing, unrelated to this change — byte-parity fixture tests skipped for an unrelated reason predating this diff).
4. **`tests/environment/`** (full directory) — `47 passed`.
5. **T8 real-`train.py` smoke test** (`tests/environment/test_behavior_measures.py::test_t8_real_train_py_smoke`) — reported failing by the Track B developer, who attributed it to Track A's in-flight edits. **Confirmed**: my Bug 1 diff was the direct cause, via the second dispatch-copy gap described above (this test's smoke config is a legacy config trained through `train.py`'s `get_default_config()` + merge path, so it exercises exactly the mismatch). **Fixed within plan scope** (same file, same root cause, see Deviations above) — re-ran after the fix: `1 passed` (returncode 0, `env.reset` no longer raises a shape-broadcast error).
6. **`tests/env/` full directory** (all ~209 non-skipped tests, run once before the T8/second-dispatch-copy fix was applied, for a complete blast-radius check) — `204 passed, 5 failed, 508 skipped`. Of the 5 failures:
   - 1 was `test_entities_schema.py::test_both_schemas_warns_and_prefers_unified` — the expected, intentional consequence of the Bug 1 precedence flip (see Deviations above); fixed by updating the test, now passes as `test_both_schemas_warns_and_prefers_legacy`.
   - 4 were `test_unified_parity.py::test_parity[configs__verification__observability_gates_{S1,S2,S3,S4}]`, all failing with an identical signature (`agent_pos mismatch at step 0`, actual `[4,4]` vs fixture-desired `[2,2]`). **Verified pre-existing and unrelated to this diff**: I `git stash`-reverted only `src/environment/config_loader.py` to the pre-Track-A baseline and re-ran `test_parity[...observability_gates_S1]` — it failed with the **exact same** `agent_pos` mismatch, confirming this is a stale/broken parity fixture issue that predates Track A entirely (these configs are loaded via a bare `Config(yaml.safe_load(...))` with no `default.yaml` underlay at all, so Bug 1's precedence flip cannot affect them — `has_entities` is `False` in both the old and new code for these 4 configs since they carry no `entities:` key). Restored my fix afterward (`git stash pop`) and confirmed `config_loader.py`'s diff was intact. Not fixed (pre-existing, out of Track A's scope) — flagging for `senior-developer`/`bug-curator` follow-up.

### Speed check

Skipped per the plan (Checkpoints item, pre-approved): all four fixes are load-time/validation-only changes inside `_load_animals()` and `_parse_noise_config()`, which run once per training run at config-parse time, not on the JIT-compiled hot path (`jax_step`/`jax_reset` internals are untouched — the fixes only change which YAML branch is taken and add pure-Python validation before array construction). No step-time/SPS impact is possible from this diff.

### Deviations from the plan (summary)

1. **Second dispatch-copy fix** (`config_loader.py` ~line 1087, inside `load_env_params`) — not listed in the plan's File Changes (which specified only lines 429–443), but required to make Bug 1's precedence flip internally consistent; without it, `train.py` crashes with a shape-broadcast `ValueError` in `jax_reset` on any legacy config where `entities:` also survives the base-underlay merge (confirmed via the T8 smoke test). Same file, same root cause, same fix pattern (mirror `has_legacy`).
2. **`tests/env/test_entities_schema.py` test update** — not listed in the plan's File Changes; updated because it directly encoded the old (pre-fix) precedence semantics that Bug 1 intentionally inverts. Two docs (`UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md`, `env_entities_cp1_verification.md`) still describe the old behaviour and were **not** touched (out of `developer` scope) — flagged for `senior-developer`.
3. **Bug 1 regression test uses `_load_animals()` directly** instead of `load_env_params()`, as explained above — matches the plan's own Finding-1 reproducer.
4. **Pre-existing unrelated test failures** (`test_unified_parity.py` S1–S4) — investigated, confirmed unrelated (reproduces on baseline), left untouched (out of Track A's scope; flagging for follow-up, not fixing silently).

No other deviations. No new mandatory config keys / `get_mandatory` call sites were added, per the plan.

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-23
> **Verdict**: **PASS-WITH-NOTES** — all four fixes are implemented as specified and the code is correct; both developer deviations are sound and in-scope; regression tests are meaningful (re-run green here, red pre-fix as recorded). **One material documentation error**: the plan's blast-radius claim that *zero currently-active configs use the legacy format* is **false** — 11 active (non-archive) configs do, and at least the 5 continual training-stage configs among them **change training behaviour** under this fix (see Note A). That is the fix working correctly, but it means the "nothing running now is affected" reassurance is wrong and a meaning-change flag is owed to the user for the continual/verification config families. Not a code blocker — the correction is to the plan's analysis, not the diff.

### Plain-language verdict

The four config-layer fixes all do what the plan said, and I re-ran the tests to confirm. The one thing the plan got wrong is *scope*: it claimed no config that anyone is currently using would change, and that only old archived configs would be affected on rerun. In fact eleven live configs still use the old animal-list format, and the five "continual learning" training-stage configs among them (the `nmn_double_return_stages` set) were **silently training on the wrong world** before this fix — they inherited the shared default scene (2 predators + 2 rabbits) instead of the world they actually describe (e.g. 3 authored slots). The fix corrects that, which is good, but it means: (a) any past training runs from those five stage configs used a different world than their file describes, so past-vs-future results from that family are not directly comparable — the same kind of caveat the plan already flagged for the `05` noise config; and (b) the plan's blast-radius table and the developer's "0 active files" grep result need correcting.

### Verifier checklist

- [x] Track A files only — Track A's 5 files (`config_loader.py`, `05-sensory_noise_10x10.yaml`, `02_config_schema.md`, `test_entities_schema.py`, new test file) are the expected ones. Other dirty files in the tree (`eval_rollout.py`, `dwell_sweep/*`, `SCRIPTS_DEPENDENCY_MAP.md`, `train_command-agent.sh`, diaries) belong to other in-flight sessions and are **out of Track A scope** — not evaluated here.
- [x] Bug 1 diff is the minimal guard flip (`if has_entities and not has_legacy`) + warning relocation; both branch bodies unchanged; dead warning block removed; **no orphaned `has_legacy_predators`/`has_legacy_neutrals` references** (grep clean).
- [x] Bug 2/3 raise `ValueError` with the valid-options list in the message; `'none'` still accepted (guard test `test_valid_noise_configs_still_load` green).
- [x] Bug 4: three `sigma: 0.0` added; header footnote makes the "CLEAN" claim true; meaning-change note preserved in this doc.
- [x] All 5 new tests present; developer recorded each bug test FAILED pre-fix; re-run here → `17 passed, 2 skipped` across the new file + `test_extends_layering.py` + `test_entities_schema.py`.
- [x] `test_extends_layering.py` still green (7 passed — no precedence regression).
- [x] No new mandatory keys introduced (no new `get_mandatory` calls in the diff).
- [x] `02_config_schema.md` precedence + noise-strictness notes updated and accurate.
- [x] **Deviation A (second dispatch site, `load_env_params` ~L1087)** verified: same root cause, mirrors `has_legacy`; `load_env_config` and `load_env_params` now agree on precedence. **`test_t8_real_train_py_smoke` re-run by verifier → `1 passed`** (28s), confirming the `jax_reset` shape-broadcast crash is resolved.
- [x] **Deviation B (`test_entities_schema.py` prefers-legacy)** verified: matches the plan's decided legacy-wins semantics; assertions flipped correctly (`legacy_wolf` present, `wolf` absent); not masking any behaviour beyond the plan.
- [x] Parity failures (`test_unified_parity.py` S1–S4) confirmed **genuinely pre-existing**: `git stash`-reverted `config_loader.py` and re-ran S1 → identical failure (`agent_pos [4,4]` vs desired `[2,2]`) both with and without the fix. Unrelated to Track A. Fix restored intact after.
- [x] Speed: N/A — config-parse-time / validation-only path, not on the JIT hot path. Justified skip accepted (✅ no regression possible).

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/config_loader.py` | Bugs 1–3 + 2nd dispatch site (Dev. A) | ✅ | Minimal guard flip; both dispatch sites now consistent; t8 smoke green |
| `configs/environment/experiment/basic/05-sensory_noise_10x10.yaml` | Bug 4 | ✅ | Three `sigma: 0.0` added; interoception now genuinely clean |
| `docs/environment/02_config_schema.md` | doc sync | ✅ | Precedence + noise-strictness notes accurate |
| `tests/env/test_config_layer_silent_failures_20260723.py` | new tests | ✅ | 5 tests, re-run green; assertions spot-checked meaningful |
| `tests/env/test_entities_schema.py` | prefers-legacy update (Dev. B) | ✅ | Matches decided semantics; not masking a behaviour change |

### Note A — blast-radius claim is WRONG (11 active legacy configs; 5 change behaviour)

The plan's Blast-radius section and the Implementation Report both state `grep -rlE "^\s*predators:" configs/ | grep -v archive` → **0 files**. **Re-run: 11 files.** They are tracked and committed since May 2026, so they were present when the developer ran the grep — the "0 files" result is simply incorrect (likely a mistyped/mis-scoped grep on the developer's side):

- 6 verification configs — `configs/verification/observability_gates_S{1,2,3,4}.yaml`, `olfaction_parity_{predator,neutral}.yaml`. Loaded **bare** (no `default.yaml` underlay) by `test_unified_parity.py`, so `has_entities` is `False` both pre- and post-fix → **unaffected** in that path. (These are also the 4 pre-existing parity failures.)
- 5 continual training-stage configs — `configs/continual/nmn_double_return_stages/0{1..5}_*_predator.yaml`. Loaded by **`train.py`'s continual path** (`train.py:197-198`: deep-copies `base_config` from `get_default_config()`, which carries `default.yaml`'s `entities:`, then merges the stage YAML). So each stage config ends up with **both** the inherited `entities:` and its own legacy `predators:`/`neutral_animals:` → precedence flip applies. **Verified empirically**: pre-fix these train on the base scene (4 slots: 2 pred + 2 rabbit); post-fix on the authored legacy scene (e.g. `01_active_predator` → 3 slots: `full`, `TL`, `BR`). **This is a real training-behaviour change for an active config family.**

**Implication (owed to the user, same class as the Bug 4 meaning-change note):** the fix *corrects* a previously-silent mis-training — good — but any past `nmn_double_return_stages` continual runs were trained on the shared default scene, not their authored per-stage scene. Past-vs-future results from that family are therefore **not directly comparable**. The plan's reassurance that "no currently-active or currently-running config changes behaviour — the fix only affects reruns of archived configs" is false and should be corrected in the Blast-radius section. **This is an analysis/documentation correction, not a code defect** — the code behaves correctly.

### Note B — two stale docs: follow-up, not same-change blocker

`02_config_schema.md` (the authoritative live schema-of-record, covered by the Maintenance Contract) is updated correctly in this change. The two docs the developer flagged — `docs/develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md` and `docs/reviews/env_entities_cp1_verification.md` — still name-check the old test `test_both_schemas_warns_and_prefers_unified` and describe "unified takes precedence". These are **historical CP1/CP3 verification & design records** describing the state as-implemented at that time, not the live schema, so they do **not** require a same-change edit. Recommended follow-up: append a one-line "precedence later flipped by FIX_CONFIG_LAYER_SILENT_FAILURES_20260723" cross-reference to each so a future reader isn't misled. Non-blocking.

### Follow-ups (for the user / next session)

1. **Correct the plan's Blast-radius section** to reflect 11 active legacy configs and the 5 continual configs that change behaviour (Note A). — analysis fix, this doc.
2. **Flag the `nmn_double_return_stages` continual family** for the same past-vs-future comparability caveat as the Bug 4 `05`-config meaning-change note. Any prior continual runs from that set used the default scene, not the authored per-stage scene.
3. **Two stale env_entities docs** (Note B) — append cross-reference notes; non-blocking.
4. **Pre-existing `test_unified_parity.py` S1–S4 failures** — unrelated to Track A; hand to `bug-curator` for a registry row if not already tracked.
5. Ask `bug-curator` to flip KNOWN_BUGS rows P1 #1, #2, #3, #8 to FIXED with this doc as the fix link.

**Conclusion**: PASS-WITH-NOTES — code is correct and all four bugs fixed with meaningful regression coverage and two sound in-scope deviations; the plan's "0 active configs" blast-radius claim is materially wrong (11 active, 5 change behaviour) and needs correcting, plus a comparability caveat is owed for the `nmn_double_return_stages` continual family.

---

<!-- After the fix lands, ask bug-curator to flip KNOWN_BUGS rows P1 #1, #2, #3, #8 to FIXED with this doc as the fix link. -->
