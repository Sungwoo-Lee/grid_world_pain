# Rename `danger` Resource → `hiding_predator`

> **Status**: PLANNED — **requires user decision on scope (A vs B)**
> **Opened**: 2026-04-24
> **Related**: [labmeeting task list #3](../../CLAUDE.md) · [docs/environment/07_predator_ai.md](../environment/07_predator_ai.md) · [docs/environment/08_resources_and_obstacles.md](../environment/08_resources_and_obstacles.md) · sibling plans: [ISSUE_01](ISSUE_01_PREDATOR_COUNT.md), [ISSUE_02](ISSUE_02_PROPERTY_KEY_UNIFY.md), [ISSUE_04](ISSUE_04_CHECKPOINT_RETENTION.md)

---

## Context

The environment has a "danger" entity today that is implemented as a **resource with `type: "danger"`**. It sits on the grid, deals damage on contact, does not move, and emits an olfactory signature. Conceptually the user wants this re-framed as a **hiding predator** — a stationary predator entity, not a resource — because:

- It is *functionally* predatory (deals damage, emits alarm olfaction).
- Treating it as a resource conflates "thing I want to consume" with "thing I want to avoid."
- A stationary-predator abstraction composes more cleanly with future predator behaviors (e.g. ambush predators with variable reveal distance).

The task is both a **rename** and a **system change** (per the Korean task label: *명칭 및 시스템 변경*).

## Analysis

**Where "danger" lives today.**

| Concern | File | Notes |
|---------|------|-------|
| Type discriminator | [config_loader.py:31](../../src/environment/config_loader.py#L31) | `res_type == 1` means danger (0 = food) |
| Default nociception | [config_loader.py:44](../../src/environment/config_loader.py#L44) | `0.9` baseline if type=danger |
| Damage accumulation | [core.py:370-380](../../src/environment/core.py#L370-L380) | `is_danger = params.res_type == 1`; info-dict key `damage_danger` |
| Olfactory pickup | shared with resources via `res_property` | No separate code path |
| Nociception sensor | [sensor.py:64-68](../../src/environment/sensor.py#L64-L68) | Comment reads "Danger Resource Contact" |
| Renderer | [grid_world.py](../../src/environment/grid_world.py) (~10 lines) | Color `#DC2626`, icon string `'danger'` |
| Example configs | [configs/experiment/labmeeting/basic-00-predator.yaml](../../configs/experiment/labmeeting/basic-00-predator.yaml) and siblings | `type: "danger"` under `resources:` |

**Two valid implementation scopes.** The user should pick before any code is touched:

### Option A — rename only (cosmetic, ~20 LoC)

Keep the current "danger is a resource with type=1" architecture intact. Only rename user-facing strings:

- YAML `type: "danger"` → `type: "hiding_predator"`.
- Info-dict key `damage_danger` → `damage_hiding_predator`.
- Color key, icon string, comments, docs.

Pros: Small diff, reversible, no retraining impact (state shape unchanged). Cons: The conceptual mismatch persists — "hiding predator" is still stored in `res_*` arrays, which is confusing for new contributors.

### Option B — architectural split (new entity type, ~200+ LoC)

Introduce a new entity category `hiding_predator` with its own arrays (`hpred_pos`, `hpred_damage`, `hpred_property`, …) parallel to `pred_*`. Move the contact-damage logic out of `update_resources` into a new `update_hiding_predators` that handles the stationary-predator case. The active `pred_*` code path handles moving predators; the new `hpred_*` code path handles stationary ones.

Pros: Correct abstraction; opens the door to future "ambush predator" variants; cleaner telemetry. Cons: Large diff; changes `EnvState` pytree structure → **breaks every existing checkpoint**; touches placement, renderer, sensor, sampling, and info-dict.

**Recommendation.** Start with **Option A** (rename-only) for this lab-meeting cycle so the external naming matches intent immediately, and open a follow-up plan for Option B if the architecture refactor is later desired. The rest of this document plans **Option A**. If the user prefers Option B, this plan needs to be reopened and expanded.

## Implementation Plan (Option A — rename only)

### Design

Change only the string literals and the one info-dict key. Preserve array layouts, state pytrees, and all downstream math. This guarantees checkpoint compatibility with pre-rename runs and keeps the diff surgical.

### File Changes

#### `src/environment/config_loader.py` (lines 31, 44)

```python
# BEFORE:
res_type = jnp.array([0 if r_get(r, 'type') == 'food' else 1 for r in expanded_resources], dtype=jnp.int32)
# ...
res_nociception = jnp.array([r.get('nociception_intensity', 0.9 if r_get(r, 'type') == 'danger' else 0.0) for r in expanded_resources])

# AFTER:
res_type = jnp.array([0 if r_get(r, 'type') == 'food' else 1 for r in expanded_resources], dtype=jnp.int32)
# (No logic change — still 0/1 — but document in a one-line comment that 1 means hiding_predator.)
res_nociception = jnp.array(
    [r.get('nociception_intensity', 0.9 if r_get(r, 'type') == 'hiding_predator' else 0.0)
     for r in expanded_resources]
)
```

Also add a one-line YAML value validator that accepts `'food'` and `'hiding_predator'` and emits a `DeprecationWarning` if `'danger'` is seen — this lets old configs keep working for one cycle before being removed.

#### `src/environment/core.py` (lines 370-380)

```python
# BEFORE:
is_danger = params.res_type == 1
damage_res = jnp.sum(jnp.where(jnp.logical_and(interact_resource, is_danger), sampled_res_damage, 0.0))
info = {
    # ...
    'damage_danger': damage_res,
}

# AFTER:
is_hiding_predator = params.res_type == 1
damage_res = jnp.sum(jnp.where(jnp.logical_and(interact_resource, is_hiding_predator), sampled_res_damage, 0.0))
info = {
    # ...
    'damage_hiding_predator': damage_res,
}
```

**Important:** the training code and any WandB logger that reads `info['damage_danger']` needs to be updated in lock-step. The implementing agent must `git grep -n "damage_danger"` and update every reference.

#### `src/environment/sensor.py` (lines 64-68)

Comment rename only — no logic change:

```python
# BEFORE:
# Danger Resource Contact
# AFTER:
# Hiding Predator Contact
```

#### `src/environment/grid_world.py`

Update the color key, the icon string in the legend list, and any `draw_icon(..., 'danger', ...)` call site. Keep the same RGB value (`#DC2626`) so rendered frames are visually identical.

#### Configs

Every YAML under `configs/` that contains `type: "danger"` must become `type: "hiding_predator"`. At minimum:

- `configs/experiment/labmeeting/basic-00-predator*.yaml`
- `configs/experiment/labmeeting/basic-04-predRange*.yaml`
- `configs/environment/default.yaml` if it references danger resources.

Run `git grep -n '"danger"' configs/` after the edit to confirm zero hits.

#### Docs

- `docs/environment/08_resources_and_obstacles.md` — rewrite the "danger" subsection header and prose.
- `docs/environment/ENVIRONMENT_SUMMARY.md` — update any references.
- `docs/environment/07_predator_ai.md` — add a one-line cross-reference ("see also: hiding-predator resource") noting the difference between active predators and hiding-predator resources until Option B lands.

### Alternatives Considered

- **Option B (architectural split).** Proper long-term answer; deferred per recommendation above. A separate plan will be drafted if the user approves.
- **Keep the name "danger" but add a `hides: true` flag to predators.** Rejected — this reverses the user's explicit rename directive.

## Checkpoints

- [ ] **C1** — After code edits, load [basic-00-predator.yaml](../../configs/experiment/labmeeting/basic-00-predator.yaml) (renamed to `type: "hiding_predator"`) and confirm no errors.
- [ ] **C2** — Run one `jax_reset` → step loop that deliberately walks the agent onto a hiding-predator tile. Confirm `info['damage_hiding_predator']` is non-zero and `info['damage_danger']` does **not** exist.
- [ ] **C3** — Render a frame via `renderer_v2.py`; confirm the visual output is identical to pre-rename baseline (same color, same position).
- [ ] **C4** — Load an old YAML still using `type: "danger"`; confirm a `DeprecationWarning` is emitted and the environment still works (temporary backwards-compat, to be removed next cycle).
- [ ] **C5** — `git grep -n "danger"` across `src/`, `configs/`, and `docs/` — only the deprecation-warning string literal and this plan document should match.

## Implementation Report

> **Implemented by**: _(pending)_
> **Date**: _(pending)_

## Verification Report

> **Verified by**: _(pending)_
> **Date**: _(pending)_

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/config_loader.py` | Accept `hiding_predator`, deprecate `danger` | | |
| `src/environment/core.py` | Rename `is_danger`, `damage_danger` key | | |
| `src/environment/sensor.py` | Rename comment | | |
| `src/environment/grid_world.py` | Rename color/icon keys | | |
| `configs/experiment/**/*.yaml` | Rename `type: "danger"` → `type: "hiding_predator"` | | |
| WandB logger / training code | Rename `damage_danger` references | | |
| `docs/environment/{07,08,summary}.md` | Text and cross-refs | | |

**Conclusion**: _(pending)_
