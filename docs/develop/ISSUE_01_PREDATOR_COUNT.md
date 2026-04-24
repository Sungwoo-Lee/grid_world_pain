# Predator `count` Field Silently Ignored

> **Status**: COMPLETED
> **Opened**: 2026-04-24
> **Related**: [labmeeting task list #1](../../CLAUDE.md) · sibling plans: [ISSUE_02](ISSUE_02_PROPERTY_KEY_UNIFY.md), [ISSUE_03](ISSUE_03_DANGER_TO_HIDING_PREDATOR.md), [ISSUE_04](ISSUE_04_CHECKPOINT_RETENTION.md)

---

## Context

YAML configs for predators include a `count:` field (e.g. `count: 3`), yet only one predator ever spawns per list entry. The user wants `count: N` to expand into `N` predators the same way it does for resources, obstacles, and neutral animals. Today, writing `count: 3` on a single predator entry silently produces exactly one predator — a silent config bug.

## Analysis

**Root cause.** `config_loader.py` expands `count` for resources/obstacles/neutrals but **not** for predators. The predator block iterates the YAML list directly — each list entry becomes exactly one predator, and `count` is never read.

**Evidence.** Resource expansion (works):

```python
# src/environment/config_loader.py:17-23
raw_resources = config.get_mandatory('environment.resources')
expanded_resources = []
if raw_resources:
    for r in raw_resources:
        count = r.get('count', 1)
        for _ in range(count):
            expanded_resources.append(r)
```

Predator block (no expansion):

```python
# src/environment/config_loader.py:56-66
predators = config.get_mandatory('environment.predators')
if predators:
    def p_get(p, key):
        val = p.get(key)
        if val is None:
            raise ValueError(f"Strict Config: Predator field '{key}' is required.")
        return val

    pred_property = jnp.array([p_get(p, 'property') for p in predators])
    # ... all subsequent arrays use len(predators) as the leading dim
```

Downstream shape assumption:

```python
# src/environment/core.py:638
num_pred = params.pred_damage.shape[0]
```

Because every predator array is built with shape `[len(predators), …]`, giving a single YAML entry a `count: 3` yields shape `[1, …]`. The reset loop spawns exactly one predator at one spawn area and the other two "copies" never exist.

**Example config that exposes the bug:** [configs/experiment/labmeeting/basic-00-predator.yaml:45-57](../../configs/experiment/labmeeting/basic-00-predator.yaml#L45-L57) declares `count: 1` on the predator entry — the key is loaded, ignored, and never read by the code.

## Implementation Plan

### Design

Add the same count-expansion idiom predators are missing, **before** any `pred_*` array is built. This keeps the fix local to `config_loader.py` and preserves every existing downstream invariant: `len(expanded_predators)` becomes the new leading axis, so `core.py:638` and every vmap over predators continues to work unchanged.

Rationale for matching the resource pattern exactly:

- Zero-impact on downstream shape handling (same static-axis semantics).
- Keeps the YAML schema uniform across entity types — fixes a latent inconsistency rather than introducing a new mechanism.
- Backwards-compatible: entries without `count` default to `1`, so every existing config keeps its current behavior.

### File Changes

#### `src/environment/config_loader.py` (lines 56-99)

```python
# BEFORE:
predators = config.get_mandatory('environment.predators')
if predators:
    def p_get(p, key):
        val = p.get(key)
        if val is None:
            raise ValueError(f"Strict Config: Predator field '{key}' is required.")
        return val

    pred_property = jnp.array([p_get(p, 'property') for p in predators])
    # ... remaining pred_* arrays built by iterating `predators` ...

# AFTER:
raw_predators = config.get_mandatory('environment.predators')
expanded_predators = []
if raw_predators:
    for p in raw_predators:
        count = p.get('count', 1)
        for _ in range(count):
            expanded_predators.append(p)

if expanded_predators:
    def p_get(p, key):
        val = p.get(key)
        if val is None:
            raise ValueError(f"Strict Config: Predator field '{key}' is required.")
        return val

    pred_property = jnp.array([p_get(p, 'property') for p in expanded_predators])
    # ... every subsequent `for p in predators` → `for p in expanded_predators` ...
```

Every list comprehension inside the predator block must switch from iterating `predators` to iterating `expanded_predators`. There are approximately a dozen such comprehensions between lines 63 and 99 — the implementing agent must update them all.

#### `docs/environment/02_config_schema.md` — update the predator YAML table

Add a row documenting `count` (default `1`) under the predator block, matching the existing resource/obstacle/neutral rows.

### Alternatives Considered

- **Reject `count` on predators loudly.** Cleaner from a "one obvious way" standpoint, but the user has explicitly asked for the field to work — so align with resources instead.
- **Move the expansion loop into a shared helper.** Attractive refactor, but out of scope for a single-issue fix. Leave for a separate cleanup.

## Checkpoints

- [x] **C1** — After editing, load [basic-00-predator.yaml](../../configs/experiment/labmeeting/basic-00-predator.yaml) and print `params.pred_damage.shape[0]` — must equal the sum of `count` across all predator entries. [16:09:28]
- [x] **C2** — Write a minimal config with two predator entries (`count: 2` and `count: 3`) and confirm `num_pred == 5`. [16:09:28]
- [x] **C3** — Run a single `jax_reset` → `jax_step` loop under `random_start_pos=True` and assert distinct predator positions (i.e. placement actually produced N entities, not one entity duplicated). [16:11:02]
- [x] **C4** — Render a frame via `src/environment/renderer_v2.py` with `count: 3` and visually confirm three predators on screen. [16:12:24]
- [x] **C5** — Run the smallest existing training smoke script (`scripts/` or `train.py` for a handful of steps) to confirm JIT tracing still compiles with the new predator leading-axis size. [16:13:01]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-04-24 16:13:00

- Implemented `count` expansion for predators in `config_loader.py`.
- Updated `02_config_schema.md` to document the `count` field.

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-04-24

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/config_loader.py` | Add count expansion for predators | ✅ | Lines 56-62 add the expansion loop mirroring the resource pattern. All 13 list comprehensions (lines 70-90) switched from `predators` → `expanded_predators`. Guard changed to `if expanded_predators:` so a config with all `count: 0` falls through safely. |
| `docs/environment/02_config_schema.md` | Document `count` field under predators | ✅ | New row at line 188: `Per-predator count | 1 | config_loader.py:91`. Row anchor line number is slightly off (the `.get('count', 1)` fallback actually lives at line 60, not 91) — cosmetic only. |

**Scope check.** `git diff --stat HEAD` shows only the two planned files changed (+23/-15). No out-of-scope edits.

**Edge-case check.** If a user writes `count: 0` for every predator entry, `expanded_predators` is empty and the code flows into the else-branch at line 92 that sets all zero-shaped arrays and still initializes `predator_enabled` (line 106). Safe.

**Minor nits (non-blocking).**

- Trailing whitespace on line 63 after the inner `for` loop.
- The doc-table reference to `config_loader.py:91` points at the `predator_enabled` line rather than the actual `count` fallback at line 60. Worth correcting next time the doc is touched.

**Conclusion**: ✅ Verified. Implementation matches the plan; all 5 checkpoints passed by Gemini; edge cases are safe.
