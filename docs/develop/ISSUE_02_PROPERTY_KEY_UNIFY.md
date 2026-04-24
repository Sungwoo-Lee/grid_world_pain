# Unify Olfactory YAML Key: `property` vs `properties`

> **Status**: PLANNED
> **Opened**: 2026-04-24
> **Related**: [labmeeting task list #2](../../CLAUDE.md) · [OLFACTORY_PROPERTY_VARIANCE.md](OLFACTORY_PROPERTY_VARIANCE.md) · [ENVIRONMENT_SUMMARY.md §FAQ](../environment/ENVIRONMENT_SUMMARY.md#cross-doc-clarifications--faq) · sibling plans: [ISSUE_01](ISSUE_01_PREDATOR_COUNT.md), [ISSUE_03](ISSUE_03_DANGER_TO_HIDING_PREDATOR.md), [ISSUE_04](ISSUE_04_CHECKPOINT_RETENTION.md)

---

## Context

Entities with olfactory signatures use **inconsistent YAML key names**:

| Entity | Key used today |
|--------|----------------|
| Resource | `properties` (plural) |
| Obstacle | `properties` (plural) |
| Predator | `property` (singular) |
| Neutral animal | `property` (singular) |

Using the wrong spelling yields a silent zero signature (or a `ValueError` deep inside array construction, depending on the entity). This is a documented gotcha — but the task is to **eliminate it**, not just document it.

## Analysis

**The asymmetric reads.**

```python
# src/environment/config_loader.py
res_property    = jnp.array([r_get(r, 'properties') for r in expanded_resources])   # :32   (plural)
pred_property   = jnp.array([p_get(p, 'property')  for p in predators])             # :63   (singular)
obs_property    = jnp.array([o.get('properties', [0.0]*chem_dim) for o in …])       # :124  (plural)
neutral_property= jnp.array([n_get(n, 'property') for n in expanded_neutral])       # :160  (singular)
```

The corresponding `_std` suffix keys (`properties_std` vs `property_std`) follow the same split.

**Failure mode.** When a user guesses the wrong spelling:

- Resources/obstacles have `.get('properties', [0.0]*chem_dim)` fallbacks in some paths → all-zero signature, no error. Agent perceives the entity as olfactorily neutral.
- Predators/neutrals use strict `p_get` / `n_get` → `ValueError`, but only when the canonical singular key is missing. Typing `properties:` on a predator triggers this error; typing `property:` on a resource silently falls back to zeros.

**Decision.** Pick `properties` (plural) as the canonical name everywhere. Justifications:

1. It already wins 2-to-2 on entity types, tied in raw count but higher in user-facing surface area (resources are the most frequently edited entity in experiment configs).
2. The field always holds a vector (list of `olfactory_vector_size` floats), so "properties" is linguistically more accurate.
3. Per-entity-type `std` keys also match (`properties_std` already exists for resources/obstacles).

## Implementation Plan

### Design

Two-stage migration to avoid breaking existing configs during the lab-meeting window:

**Stage A — accept both, prefer plural.** `config_loader.py` reads `properties` first, falls back to `property` with a `warnings.warn(...)` message naming the entity and key. No YAML changes required; existing configs keep working.

**Stage B — update bundled configs and docs.** Migrate every in-repo YAML + doc reference to the plural form. Leave the fallback in place for now so external users of the repo aren't broken.

**Stage C (future, out of scope for this plan)** — delete the fallback once enough grace time has passed.

Only stages A and B are in scope for this plan.

### File Changes

#### `src/environment/config_loader.py` (lines 56-66, 143-172)

Introduce a small helper near the top of the file:

```python
# AFTER (new helper near the top of config_loader.py):
import warnings

def _read_properties(entry, entity_label):
    """Read olfactory signature, preferring `properties` (plural)."""
    if 'properties' in entry:
        return entry['properties']
    if 'property' in entry:
        warnings.warn(
            f"{entity_label}: YAML key 'property' is deprecated — rename to 'properties'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return entry['property']
    raise ValueError(f"{entity_label}: missing required key 'properties'.")

def _read_properties_std(entry, entity_label):
    """Same, for the `*_std` variant."""
    if 'properties_std' in entry:
        return entry['properties_std']
    if 'property_std' in entry:
        warnings.warn(
            f"{entity_label}: YAML key 'property_std' is deprecated — rename to 'properties_std'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return entry['property_std']
    raise ValueError(f"{entity_label}: missing required key 'properties_std'.")
```

Replace the four read sites:

```python
# BEFORE (line 63):
pred_property = jnp.array([p_get(p, 'property') for p in predators])

# AFTER:
pred_property = jnp.array([_read_properties(p, 'Predator') for p in predators])
```

Do the same substitution at lines 32 (resources), 124 (obstacles), 160 (neutrals), and at every `*_std` counterpart. After the change every entity type is read through the helpers and there is exactly one spelling rule.

#### Configs under `configs/`

Rename `property:` → `properties:` and `property_std:` → `properties_std:` in every predator and neutral-animal YAML block. Pay particular attention to:

- `configs/experiment/labmeeting/basic-00-predator*.yaml`
- `configs/experiment/labmeeting/basic-04-predRange*.yaml`
- Any other `configs/experiment/**/*.yaml` that declares predators or neutrals.

Use `git grep -n "  property:\|  property_std:"` after the rename to confirm no hits remain.

#### `docs/environment/02_config_schema.md`

Remove the "inconsistent key names" warning (lines 282-293). Replace with a single table row noting that every entity uses `properties` / `properties_std`. Mention the deprecation warning in the migration notes.

#### `docs/environment/ENVIRONMENT_SUMMARY.md` FAQ (lines 181-182)

Update the "Why does my YAML key `property` do nothing for a resource?" entry — replace the gotcha with a migration note that both keys still work but the plural form is canonical and the singular form emits a `DeprecationWarning`.

### Alternatives Considered

- **Standardize on singular (`property`).** Rejected — requires more YAML edits (resources appear more often than predators in experiment configs) and the std-suffix form is already plural for resources.
- **Hard-fail on the old spelling immediately.** Rejected — breaks every experiment config overnight. Deprecation warning plus Stage B migration is safer.

## Checkpoints

- [ ] **C1** — After the helper is added, confirm an unchanged YAML (still using `property:` on a predator) loads **and** prints a `DeprecationWarning` once per predator entry.
- [ ] **C2** — Hand-edit one lab-meeting config to use `properties:` on every predator, rerun the loader, confirm no warning is printed and `pred_property` has the expected shape/values.
- [ ] **C3** — Run `python -c "from src.environment.config_loader import load_env_params; load_env_params('configs/experiment/labmeeting/basic-00-predator.yaml')"` before and after migration — byte-compare the resulting `EnvParams.pred_property` and `EnvParams.neutral_property` arrays to confirm numerical equivalence.
- [ ] **C4** — Run the perceptual-noise debug harness (`src/environment/debug_noise.py` or equivalent script — see `docs/develop/NOISE_RENDERING_RENDERER_FIX.md`) to confirm olfactory signals still reach the agent sensor correctly.
- [ ] **C5** — `git grep` for `'property'` and `'property_std'` string literals in `src/` — only the helper fallback branch should remain.

## Implementation Report

> **Implemented by**: _(pending)_
> **Date**: _(pending)_

## Verification Report

> **Verified by**: _(pending)_
> **Date**: _(pending)_

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/config_loader.py` | Add helpers, unify reads on `properties` | | |
| `configs/experiment/**/*.yaml` | Rename `property[_std]` → `properties[_std]` | | |
| `docs/environment/02_config_schema.md` | Remove inconsistency warning | | |
| `docs/environment/ENVIRONMENT_SUMMARY.md` | Update FAQ entry | | |

**Conclusion**: _(pending)_
