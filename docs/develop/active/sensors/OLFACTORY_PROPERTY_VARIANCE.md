---
title: Olfactory Property Variance
topic: sensors
status: active
created: 2026-04-21
last_updated: 2026-04-21
---

# Olfactory Property Variance

> **Status**: COMPLETED
> **Opened**: 2026-04-21
> **Related**: [`docs/environment/09_sensors_and_observation.md`](../environment/09_sensors_and_observation.md), [`docs/environment/02_config_schema.md`](../environment/02_config_schema.md)

---

## Context

Current olfactory observation uses a single fixed 5-dim chemical property vector per entity (e.g. food = `[1, 0, 0, 0, 0]`). This means every food source in every episode emits an identical chemical signature. For interoceptive research on olfactory ambiguity, perceptual decision making, and prey/predator discrimination under sensory noise, we want each entity's signature to be a **random draw** around a species-level mean, so the agent has to generalise over a chemical cloud rather than memorise a fixed vector.

This proposal adds a per-entity Gaussian draw at episode reset (and at resource respawn) whose mean comes from the existing YAML `properties`/`property` key and whose standard deviation comes from a new `properties_std`/`property_std` key. Sampled values are clipped to `[0, 1]`.

## Analysis

### User-selected design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sampling frequency | **Per-respawn** for resources; **per-reset** for predators/obstacles/neutrals (they don't respawn) | Makes each regenerated food instance a fresh draw; predators/rocks/bushes/rabbits persist through an episode so only reset is meaningful |
| Distribution | **Clipped Gaussian** — `clip(mean + std·N(0,1), 0, 1)` | Simplest JAX implementation; one `jax.random.normal` + `jnp.clip` call; acceptable mass-on-boundary behaviour given expected std ≤ 0.3 |
| Scope | **All four entity types**: resources, predators, neutral animals, obstacles | Uniform YAML surface; backward-compatible (std defaults to zero) |
| Bounds | `[0, 1]` via clip | No negative or >1 chemical concentrations |

### Current data flow (before change)

```
YAML                              EnvParams (immutable)           Runtime
─────────────────                 ───────────────────────         ────────────────────
properties: [1,0,0,0,0]   ───►    res_property [N_res, 5]  ───►   sense_resource reads
property:   [0,1,0,0,0]   ───►    pred_property            ───►   from params directly
properties: [0,0,0,1,0]   ───►    obs_property             ───►
property:   [0,0.3,0,0,0] ───►    neutral_property         ───►
```

### Target data flow (after change)

```
YAML                              EnvParams (means+stds)          EnvState (sampled)        Runtime
──────────────────                ───────────────────────         ──────────────────        ─────────
properties:     [1,0,0,0,0]  ──►  res_property      (mean)  ──┐   res_property_sampled ──►  sense_resource
properties_std: [.2,0,0,0,0] ──►  res_property_std  (std)  ──┤──► (re-drawn on                reads STATE
                                                             │    jax_reset AND                (not params)
                                                             │    per-resource respawn)
                                                             │
property:     [0,1,0,0,0]  ──►    pred_property    (mean) ──┤──►  pred_property_sampled
property_std: [0,.1,0,0,0] ──►    pred_property_std (std) ──┘     (drawn once at reset)

… same pattern for obs_property_sampled and neutral_property_sampled …

Sampling: sampled = clip(mean + std * N(0, 1), 0.0, 1.0)
```

### Key architectural observations

1. **`EnvParams` stays immutable** — means and stds are compile-time config (shape-stable). Both go in `EnvParams`.
2. **Sampled values must live in `EnvState`** because they change per-episode (and per-respawn for resources). This is the standard JAX pattern: anything the reset/step functions write must be in state, not params.
3. **Sensor reads shift from `params.X_property` → `state.X_property_sampled`** at the four call sites in `sensor.py:266–269`.
4. **Respawn hook exists already**: `update_resources` at `core.py:115` already returns a `respawn_mask` that is used at `core.py:305` to conditionally update `res_pos` for respawned resources. We reuse the exact same mask to conditionally update `res_property_sampled`.
5. **Backward compatibility**: if `properties_std`/`property_std` are absent from YAML, default to a zero vector of the same shape → `sampled ≡ mean` → identical behaviour to today.
6. **PRNG threading**: `jax_step` already splits a `respawn_key`. We allocate a new key stream (`property_key`) in the existing split so the number of sub-keys grows by one.

## Implementation Plan

### Design

**YAML schema addition** (optional, default zeros for backward compat):

```yaml
environment:
  resources:
    - name: "food"
      properties:     [1.0, 0.0, 0.0, 0.0, 0.0]   # existing — becomes the mean
      properties_std: [0.2, 0.05, 0.0, 0.0, 0.0]  # NEW — per-dim std (optional)

  predators:
    - name: "predator"
      property:     [0.0, 1.0, 0.0, 0.0, 0.0]
      property_std: [0.0, 0.15, 0.0, 0.0, 0.0]    # NEW

  neutral_animals:
    - name: "rabbit"
      property:     [0.0, 0.3, 0.0, 0.0, 0.0]
      property_std: [0.0, 0.1, 0.0, 0.0, 0.0]     # NEW

  obstacles:
    - name: "bush"
      properties:     [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.2, 0.0]   # NEW
```

**Sampling formula** (used everywhere):

```python
# Shape-preserving per-dim clipped Gaussian
noise   = jax.random.normal(key, shape=mean.shape)  # [N, 5]
sampled = jnp.clip(mean + std * noise, 0.0, 1.0)
```

**Where sampling happens:**

| Where | What samples | Key source |
|---|---|---|
| `jax_reset` (core.py) | All four: `res`, `pred`, `obs`, `neutral` | Fresh split from the reset PRNG |
| `jax_step` (core.py) inside the respawn branch | **Only** `res_property_sampled` at indices where `respawn_mask` is True | New sub-key from `state.key` split |

### File Changes

#### `src/environment/state.py` (lines 55–100)

Add `*_std` fields to `EnvParams` and `*_property_sampled` fields to `EnvState`.

```python
# BEFORE (EnvParams resource block, state.py:59-66):
res_type: jnp.ndarray       # [num_res] int (0:food, 1:danger)
res_property: jnp.ndarray   # [num_res, vector_size]
res_nociception: jnp.ndarray # [num_res]
...

# AFTER:
res_type: jnp.ndarray         # [num_res] int (0:food, 1:danger)
res_property: jnp.ndarray     # [num_res, vector_size]  MEAN
res_property_std: jnp.ndarray # [num_res, vector_size]  STD (zeros if absent)
res_nociception: jnp.ndarray
...
```

Repeat for `pred_property_std`, `obs_property_std`, `neutral_property_std` in the corresponding EnvParams blocks at `state.py:69`, `state.py:89`, `state.py:95`.

In `EnvState`, add four new sampled arrays (place them near `res_pos`, `pred_pos` etc.):

```python
# NEW in EnvState:
res_property_sampled: jnp.ndarray      # [num_res, vector_size]
pred_property_sampled: jnp.ndarray     # [num_pred, vector_size]
obs_property_sampled: jnp.ndarray      # [num_obs, vector_size]
neutral_property_sampled: jnp.ndarray  # [num_neutral, vector_size]
```

#### `src/environment/config_loader.py` (lines 32, 45, 60, 81, 118, 132, 152, 161)

Parse the new optional `*_std` keys; default to a zero vector matching the mean's shape.

```python
# Resource block — line 32 area
# BEFORE:
res_property = jnp.array([r_get(r, 'properties') for r in expanded_resources])

# AFTER:
res_property = jnp.array([r_get(r, 'properties') for r in expanded_resources])
chem_dim = res_property.shape[-1]
res_property_std = jnp.array([
    r.get('properties_std', [0.0] * chem_dim) for r in expanded_resources
])

# Empty-list fallback at line 45 area — add:
res_property_std = jnp.zeros((0, 5))
```

Apply the same pattern:
- Predators (line 60): `pred_property_std` ← `p.get('property_std', [0.0]*chem_dim)`; empty fallback at line 81 → `jnp.zeros((0, 5))`
- Obstacles (line 118): `obs_property_std` ← `o.get('properties_std', [0.0]*chem_dim)`; empty fallback at line 132 → `jnp.zeros((0, chem_dim))`
- Neutral animals (line 152): `neutral_property_std` ← `n.get('property_std', [0.0]*chem_dim)`; empty fallback at line 161 → `jnp.zeros((0, 5))`

Wire the new fields into the `EnvParams(...)` constructor at the end of `load_env_params` (around `config_loader.py:237-267`):

```python
res_property=res_property,
res_property_std=res_property_std,   # NEW
...
pred_property=pred_property,
pred_property_std=pred_property_std, # NEW
...
obs_property=obs_property,
obs_property_std=obs_property_std,   # NEW
...
neutral_property=neutral_property,
neutral_property_std=neutral_property_std,  # NEW
```

#### `src/environment/core.py` — `jax_reset` (line ~734 area)

Add initial sampling for all four entity types using fresh PRNG sub-keys.

```python
# BEFORE (inside jax_reset, around line 734):
return EnvState(
    ...
    res_active=jnp.ones(num_res, dtype=jnp.bool_),
    res_cons_count=jnp.zeros(num_res, dtype=jnp.int32),
    res_reg_timer=jnp.zeros(num_res, dtype=jnp.int32),
    ...
)

# AFTER (before the return, split one PRNG into 4 sub-keys):
prop_key_res, prop_key_pred, prop_key_obs, prop_key_neutral = jax.random.split(
    property_key, 4
)

def _sample_property(key, mean, std):
    noise = jax.random.normal(key, shape=mean.shape)
    return jnp.clip(mean + std * noise, 0.0, 1.0)

res_property_sampled     = _sample_property(prop_key_res,     params.res_property,     params.res_property_std)
pred_property_sampled    = _sample_property(prop_key_pred,    params.pred_property,    params.pred_property_std)
obs_property_sampled     = _sample_property(prop_key_obs,     params.obs_property,     params.obs_property_std)
neutral_property_sampled = _sample_property(prop_key_neutral, params.neutral_property, params.neutral_property_std)

return EnvState(
    ...
    res_property_sampled=res_property_sampled,
    pred_property_sampled=pred_property_sampled,
    obs_property_sampled=obs_property_sampled,
    neutral_property_sampled=neutral_property_sampled,
    ...
)
```

`property_key` must be added to the existing `jax.random.split` call at the top of `jax_reset`. Find the split and increase the number of sub-keys by 1.

#### `src/environment/core.py` — `jax_step` respawn branch (lines 289–305)

Re-sample `res_property_sampled` only at indices where `respawn_mask` is True. Reuse the exact same mask used for position respawn.

```python
# BEFORE (lines 289-305):
key, respawn_key, predator_key, neutral_key, damage_key = jax.random.split(state.key, 5)

new_active, new_reg_timer, new_cons_count, respawn_mask = update_resources(
    state.res_active, state.res_reg_timer, state.res_cons_count, params
)

# Displace resources that just respawned
res_keys = jax.random.split(respawn_key, num_res)
new_potential_pos = ...
res_pos_after_reg = jnp.where(respawn_mask[:, None], new_potential_pos, state.res_pos)

# AFTER — add one extra sub-key and a parallel where-update for sampled property:
key, respawn_key, predator_key, neutral_key, damage_key, property_key = jax.random.split(state.key, 6)

new_active, new_reg_timer, new_cons_count, respawn_mask = update_resources(
    state.res_active, state.res_reg_timer, state.res_cons_count, params
)

# Displace resources that just respawned
res_keys = jax.random.split(respawn_key, num_res)
new_potential_pos = ...
res_pos_after_reg = jnp.where(respawn_mask[:, None], new_potential_pos, state.res_pos)

# Re-sample chemical property for respawned resources (NEW)
noise = jax.random.normal(property_key, shape=params.res_property.shape)
new_sampled_prop = jnp.clip(params.res_property + params.res_property_std * noise, 0.0, 1.0)
res_property_sampled_after_reg = jnp.where(
    respawn_mask[:, None], new_sampled_prop, state.res_property_sampled
)
```

Thread `res_property_sampled_after_reg` through the rest of `jax_step` so it ends up in the final `state._replace(...)` call (around `core.py:490`):

```python
state._replace(
    ...
    res_active=final_active,
    res_reg_timer=next_reg_timer,
    res_cons_count=next_cons_count,
    res_property_sampled=res_property_sampled_after_reg,  # NEW
    ...
)
```

Predator / obstacle / neutral sampled properties are carried through unchanged (no respawn for them).

#### `src/environment/sensor.py` (lines 266–269, 314)

Switch the four `sense_resource` call sites from reading `params.X_property` to `state.X_property_sampled`.

```python
# BEFORE (sensor.py:266-269):
if params.olfactory_enabled:
    res_chem     = sense_resource(state.agent_pos, state.res_pos,     state.res_active,                                      params.res_property,     params.sensor_radius, params.sensor_decay)
    pred_chem    = sense_resource(state.agent_pos, state.pred_pos,    jnp.ones(state.pred_pos.shape[0], dtype=jnp.bool_),    params.pred_property,    params.sensor_radius, params.sensor_decay)
    obs_chem     = sense_resource(state.agent_pos, state.obs_pos,     jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_),     params.obs_property,     params.sensor_radius, params.sensor_decay)
    neutral_chem = sense_resource(state.agent_pos, state.neutral_pos, jnp.ones(state.neutral_pos.shape[0], dtype=jnp.bool_), params.neutral_property, params.sensor_radius, params.sensor_decay)
    obs_parts.append(res_chem + pred_chem + obs_chem + neutral_chem)

# AFTER:
if params.olfactory_enabled:
    res_chem     = sense_resource(state.agent_pos, state.res_pos,     state.res_active,                                      state.res_property_sampled,     params.sensor_radius, params.sensor_decay)
    pred_chem    = sense_resource(state.agent_pos, state.pred_pos,    jnp.ones(state.pred_pos.shape[0], dtype=jnp.bool_),    state.pred_property_sampled,    params.sensor_radius, params.sensor_decay)
    obs_chem     = sense_resource(state.agent_pos, state.obs_pos,     jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_),     state.obs_property_sampled,     params.sensor_radius, params.sensor_decay)
    neutral_chem = sense_resource(state.agent_pos, state.neutral_pos, jnp.ones(state.neutral_pos.shape[0], dtype=jnp.bool_), state.neutral_property_sampled, params.sensor_radius, params.sensor_decay)
    obs_parts.append(res_chem + pred_chem + obs_chem + neutral_chem)
```

`get_observation_breakdown` at `sensor.py:314` still reads the dim from `params.res_property.shape[-1]` — no change needed because `res_property` (mean) and `res_property_sampled` have identical last-axis size.

#### `configs/environment/default.yaml`

No change required for backward compatibility, but add commented examples next to each entity type so users see the new key:

```yaml
resources:
  - name: "food"
    type: "food"
    ...
    properties: [1.0, 0.0, 0.0, 0.0, 0.0]
    # properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]   # optional per-dim std, default zeros
```

#### `docs/environment/09_sensors_and_observation.md`

Extend the Olfaction section with a subsection documenting the new sampling behaviour, the `*_std` YAML keys, and the `X_property_sampled` state fields.

#### `docs/environment/02_config_schema.md`

Add `properties_std`/`property_std` to the optional-keys table with default `[0.0]*vector_size`.

#### `docs/environment/01_state_and_params.md`

Update the `EnvParams` and `EnvState` pytree listings to include the new fields.

## Checkpoints

- [x] **Backward compat**: running any existing config **without** `properties_std` keys produces identical observations to pre-change (std=0 → sampled=mean exactly). Diff a single episode's observation tensor against a pre-change baseline. [16:19:30]
- [x] **Shape consistency**: `state.res_property_sampled.shape == params.res_property.shape` at every step. Same for pred/obs/neutral. [16:19:30]
- [x] **Clip bounds**: `jnp.all(state.res_property_sampled >= 0.0)` and `<= 1.0` hold across a test episode with large std (e.g. 0.5). [16:19:30]
- [x] **Respawn re-sampling**: in an episode with food std=0.3, confirm that `res_property_sampled` for a given resource changes value exactly at the step when `respawn_mask[i] == True` and is constant otherwise. [16:19:30]
- [x] **PRNG reproducibility**: same initial `key` + same actions → identical `res_property_sampled` trajectory. No stray host-side RNG. [16:19:30]
- [x] **vmap compatibility**: `ParallelEnv.auto_reset_step` with N=16 envs produces 16 **independent** samples (not broadcast-identical). Check by printing `state.res_property_sampled[:, 0, 0]` across envs. [16:19:30]
- [x] **No sensor regression**: `get_observation_breakdown(params)` reports the same `"Olfaction"` dim as before (read from `res_property.shape[-1]`, unchanged). [16:19:30]
- [x] **Zero-entity edge case**: a config with `neutral_animals: []` still loads (zero-shape `neutral_property_std` array) and `jax_reset` / `jax_step` run without errors. [16:19:30]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-04-21 16:07:09

- Implementation complete across `state.py`, `config_loader.py`, `core.py`, and `sensor.py`.
- Documentation updated to reflect changes across all schemas and state documentation.
- The `default.yaml` has been injected with commented optional examples for `properties_std`.

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-04-21

Diff-stat: 9 files changed, +111 / −42. All changes within planned scope; no out-of-scope files touched.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/state.py` | Add 4 `*_std` (EnvParams) + 4 `*_property_sampled` (EnvState) fields | ✅ | All 8 fields present with correct shape comments (`state.py:16, 24, 29, 33, 66, 75, 96, 103`). |
| `src/environment/config_loader.py` | Parse 4 `*_std` optional YAML keys; wire into EnvParams | ✅ | `r.get('properties_std', [0.0]*chem_dim)` pattern applied consistently for all 4 types; empty-list fallbacks present (`config_loader.py:48, 87, 140, 172`); all 4 fields wired into EnvParams constructor (`:249, 256, 273, 279`). Minor: empty-list fallbacks for res/pred/neutral hardcode `(0,5)` instead of `(0, chem_dim)`, matching the pre-existing convention for `res_property` / `pred_property` / `neutral_property` — functionally fine since `chem_dim=5` is the only path exercised today. |
| `src/environment/core.py` (`jax_reset`) | Sample initial properties; extend PRNG split by 1 | ✅ | Top-level split bumped `4 → 5` adding `property_key` (`core.py:630`); inner split into 4 sub-keys at `:738`; clipped-Gaussian helper `_sample_property` defined locally at `:740-742`; all 4 entities sampled at `:744-747`; all 4 sampled arrays wired into `EnvState(...)` at `:756, 762, 764, 776`. |
| `src/environment/core.py` (`jax_step`) | Re-sample `res_property_sampled` on respawn; extend PRNG split by 1 | ✅ | Top-level split bumped `5 → 6` adding `property_key` (`core.py:289`); re-sampling uses the **same** `respawn_mask` already used for position respawn (`:310-312`); `state._replace(res_property_sampled=res_property_sampled_after_reg)` at `:500`. Pred/obs/neutral sampled properties carried through unchanged (correct — they don't respawn). |
| `src/environment/sensor.py` | Swap 4 `params.X_property` reads → `state.X_property_sampled` | ✅ | All 4 call sites updated (`sensor.py:266-269`); `get_observation_breakdown` at `:314` still reads `params.res_property.shape[-1]` — correct since shape is unchanged, olfaction dim stable. |
| `configs/environment/default.yaml` | Add commented examples of `*_std` keys | ✅ | Commented `*_std` examples added at representative entries (food, danger, rabbit, predator, rock, tree, bush). Active behaviour unchanged → full backward compatibility. |
| `docs/environment/09_sensors_and_observation.md` | Document sampling subsection | ✅ | New config-keys table + per-entity (mean/std/sampled) table + sampling-formula subsection present. |
| `docs/environment/02_config_schema.md` | Optional-keys table update | ✅ | `*_std` entries added to optional-keys table per plan. |
| `docs/environment/01_state_and_params.md` | Pytree listing update | ✅ | EnvState and EnvParams pytree listings updated with new fields. |

### Cross-reference sanity checks

- **No stale `params.*_property` reads as signal values**: verified via grep. Remaining references (`sensor.py:314`, `core.py:640`, `evaluation_core.py:235`) all read `.shape[-1]` or `.shape[0]` only — metadata, not chemical values.
- **`respawn_mask` reuse**: `core.py:310-312` uses the exact same mask produced by `update_resources` at `:292` that is already used for position respawn at `:305`. Re-sampling timing is therefore identical to position respawn timing.
- **Backward compatibility**: with `*_std` absent from YAML, `config_loader.py` fills zeros → `_sample_property` returns `clip(mean + 0·noise, 0, 1) = mean` exactly → sensor observations are bit-identical to pre-change behaviour. (Checkpoint 1 attested by Gemini.)
- **Zero-entity safety**: `(0, 5)`-shape arrays propagate cleanly through `_sample_property` and through `jnp.where(respawn_mask[:, None], …)` (both sides broadcast to `(0, 5)`). (Checkpoint 8 attested by Gemini.)
- **PRNG hygiene**: each sampling site consumes a dedicated sub-key (`prop_key_res/pred/obs/neutral` at reset; `property_key` at step). No key reuse. Determinism preserved.

**Conclusion**: Implementation is correct, scope-aligned, and matches the plan file-for-file. The new olfactory property variance mechanism (clipped Gaussian per-entity, re-sampled at reset and on resource respawn) is fully wired end-to-end from YAML → `EnvParams` → `EnvState` → sensor. Ready for training-time validation.
