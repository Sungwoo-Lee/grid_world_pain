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

**Active-config check (hasty):** `grep -rlE "^\s*predators:" configs/ | grep -v archive` → **0 files**. Every legacy-format config lives under `configs/environment/experiment/archive/` (79 files with `predators:`). So **no currently-active or currently-running config changes behaviour** — the fix only affects reruns of archived configs (making them correct). `default.yaml` carries no legacy sections, so it is unaffected. The existing `test_extends_layering.py` C4 case (`entities: []` suppresses base animals) still holds: `entities: []` → `has_entities=True`, `has_legacy=False` → entities path with an empty list → empty scene, unchanged.

## Checkpoints

- [ ] After Bug 1: run `tests/env/test_extends_layering.py` — all C1–C9 still green (precedence change must not break the suppress-semantics or worked-example parity).
- [ ] After Bug 1: the reproducer from Finding 1 (`get_default_config(); cfg.merge(load_env_config('.../archive/2X2_area.yaml')); _load_animals(cfg)`) now returns the **legacy** scene (5 rabbits + 2 predators), matching the bare-eval load.
- [ ] After Bugs 2–3: `load_env_params(load_env_config('.../basic/05-sensory_noise_10x10.yaml'))` still succeeds (valid modes/keys not rejected).
- [ ] After Bug 4: the three interoceptive `noise_sigmas` read back as `0.0`; `olfaction`/`visual` still `state_dependent` with their intended sigmas.
- [ ] All 5 new tests fail on a clean checkout (pre-fix) and pass after the fix — capture the pre-fix failure output in the Implementation Report.
- [ ] Speed: these are load-time/validation changes on a non-hot path (config parse happens once per run). No measurable step-time impact expected; a speed benchmark is **not required** — state this explicitly in the report.

## Implementation Report

> **Implemented by**: [developer]
> **Date**: [date]

<!-- developer fills: what changed, pre-fix test failure output, any deviations. -->

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

Verifier checklist:
- [ ] `git diff --stat HEAD` — only the 6 expected files touched (`config_loader.py`, `05-sensory_noise_10x10.yaml`, `02_config_schema.md`, the new test file; plus this doc + `INDEX.md`). Flag any others.
- [ ] Bug 1 diff is the minimal guard flip (`if has_entities and not has_legacy`) + warning relocation; both branch bodies unchanged; dead warning block removed; no orphaned `has_legacy_predators`/`has_legacy_neutrals` references.
- [ ] Bug 2/3 raise `ValueError` with the valid-options list in the message; `'none'` still accepted.
- [ ] Bug 4: three `sigma: 0.0` added; header no longer misleading; meaning-change note preserved in this doc.
- [ ] All 5 new tests present, and the developer recorded that each FAILED pre-fix.
- [ ] `test_extends_layering.py` still green (no precedence regression).
- [ ] No new mandatory keys introduced (confirm no new `get_mandatory` calls).
- [ ] `02_config_schema.md` precedence + noise-strictness notes updated.
- [ ] Speed: N/A (config-parse path) — confirm developer justified skipping the benchmark.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/config_loader.py` | Bugs 1–3 | | |
| `configs/environment/experiment/basic/05-sensory_noise_10x10.yaml` | Bug 4 | | |
| `docs/environment/02_config_schema.md` | doc sync | | |
| `tests/env/test_config_layer_silent_failures_20260723.py` | new tests | | |

**Conclusion**: [one-line summary]

---

<!-- After the fix lands, ask bug-curator to flip KNOWN_BUGS rows P1 #1, #2, #3, #8 to FIXED with this doc as the fix link. -->
