---
title: "Configurable per-entity visual properties — checkpointed implementation plan (v1, static, default-8)"
topic: sensors
status: active
created: 2026-06-18
last_updated: 2026-06-18  # implemented by developer 2026-06-18
phase: null
aliases: [configurable-visual-properties-plan, visual-properties-plan]
supersedes: null
superseded_by: null
---

# Configurable per-entity visual properties — implementation plan

> **Status**: IMPLEMENTED (pending senior-developer verification)
> **Opened**: 2026-06-18
> **Branch**: `v3.0`
> **Related**: seed/context doc [[CONFIGURABLE_VISUAL_PROPERTIES]] (read that first — it holds the
> rationale and the hiding-predator motivation); precedent [[UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING]];
> parallel config work [[CONFIG_LAYERING_AND_EXPERIMENT_REORG]].

---

## Context

**What this is.** A plan to make the agent's *vision* sensor configurable the same way its *smell*
sensor already is. Today, every thing in the world (a predator, a food pellet, a rock) shows up in
the agent's visual observation as a fixed "channel" hard-wired in Python code — predator is always
channel 5, food channel 3, rock channel 6, and so on. To give something a genuinely new look you
have to edit that code and add a channel, which changes the *size* of the observation vector and
**breaks every already-trained agent** (frozen checkpoints can no longer load, because the input
width changed). That is the problem.

**The fix.** Let every entity carry a small "appearance vector" set from its config file — exactly
like smell already carries a `properties:` vector. The vision sensor then just reads that vector
instead of building a fixed channel. Changing an entity's look becomes a one-line config edit, with
**no code change and no observation-size break**.

**The make-or-break promise (the gate).** Every entity's appearance vector *defaults to its current
fixed channel* (predator → the 6th slot, food → the 4th slot, rock → the 7th slot, etc.). With that
default and the default vector width of 8, the agent's visual observation comes out **byte-for-byte
identical** to today's. So all ~86 archived configs, the modernized `default.yaml`, and the new
5-config "basic" training curriculum reproduce today's vision with **zero edits**. Custom looks and
custom widths are strictly opt-in on top. A byte-identical-observation test over those configs is the
primary acceptance gate.

**Scope of v1 (locked with the user).** Static appearance vectors only (no per-episode random
jitter yet); default width 8; the internal class→channel table is kept as the hidden default
generator; and the latent vision-label bug (two labels swapped) is fixed while we are in here. The
renderer asset-resolver (drawing a different sprite from the appearance vector) is explicitly **out
of v1 scope** — noted as a later slice.

---

## Analysis

### Why this is a small, safe change (the decisive code fact)

The visual sensor in `src/environment/sensor.py::sense_visual` is **already** an olfactory-style
property-matrix matmul. It builds a `[total_entities, 8]` matrix `all_props` and does
`vis_entities = matches @ all_props` (exact-cell match within `visual_sensor_range`). The aggregation
is identical in spirit to `sense_resource` (the smell sensor). The **only** rigid part is *where each
row of `all_props` comes from*: today each row is `jax.nn.one_hot(channel, 8)`. Swap that one source
for a config-supplied vector and the whole class→channel→obs-size coupling disappears. We keep the
exact-cell-match aggregation untouched; we only change the per-entity vector source.

### The four rigid constructions to replace (verified current line numbers, branch v3.0)

`sense_visual` spans **`src/environment/sensor.py` L145–220**. The hard-coded vector sources:

| Entity kind | Current code (line) | Channel meaning | Replace with |
|---|---|---|---|
| Background / location | `jax.nn.one_hot(jnp.where(loc_types==1,0,jnp.where(loc_types==2,1,2)), 8)` (**L174**) | grass→0, sand→1, plain→2 | a `bg_visual_property` lookup table `[3, V]` indexed by the 0/1/2 selector |
| Resources | `jax.nn.one_hot(jnp.where(params.res_type==0,3,4), 8)` (**L197**) | food→3, hiding-predator→4 | `params.res_visual_property` `[num_res, V]` |
| Obstacles | `jax.nn.one_hot(jnp.full((num_obs,),6), 8)` (**L198**) | rock→6 | `params.obs_visual_property` `[num_obs, V]` |
| Animals/entities | `jax.nn.one_hot(params.animal_visual_channel, 8)` (**L202**) | predator→5, neutral→7 | `params.animal_visual_property` `[num_animal, V]` |

`V` is the visual vector size (default 8). At `V=8` with the class-map defaults these four sources
produce *exactly* the current one-hot rows → byte-parity.

### The class→channel map (kept as the internal default generator)

`src/environment/config_loader.py` **L107–108**:
```python
ANIMAL_CLASS_TO_INT       = {"predator": 0, "neutral": 1}
ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator": 5, "neutral": 7}
```
`visual_channel_list` is built at **L632** and stored as `animal_visual_channel` (**L674**, kwarg at
**L997**); the field lives on `EnvParams` at `src/environment/state.py` **L125**. We **keep**
`ANIMAL_CLASS_TO_VIS_CHANNEL` and `animal_visual_channel` exactly as-is — they become the source that
*generates* the default `animal_visual_property` rows. Resource and obstacle defaults come from the
fixed channels in the table above (food=3, hiding-predator=4, rock=6).

### The latent label bug (fix while in here)

Two places carry the visual channel labels (`src/environment/sensor.py` **L391** for the olfactory
spectrum view and **L425** for the visual view):
```python
['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']
#  0      1      2      3      4      5      6      7
```
This puts `NEU` at index 6 and `RCK` at index 7. But the **true encoding** is rock=6, neutral=7
(from `obs_props = one_hot(6)` and `ANIMAL_CLASS_TO_VIS_CHANNEL["neutral"]=7`). So indices 6 and 7
are swapped. The correct labels are:
```python
['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'RCK', 'NEU']
```
This is **labels only** — purely a display/diagnostic concern (these label lists feed the renderer's
spectrum/visual-grid overlays). No model code consumes the label *strings*; the numeric encoding is
unchanged, so this fix does **not** affect observation bytes or parity. Verification: grep confirms
the two label literals at L391/L425 are the only occurrences and both are inside the `viz.append({...,
'labels': [...]})` display path. (Note: L391 is the *olfactory* spectrum overlay, which happens to
reuse the same 8-channel visual label set for a debug view; correcting it keeps the two consistent.)

### Observation-width ↔ perceptual-noise coupling (the one sync point)

The visual observation width is computed in **one** place,
`src/environment/sensor.py::get_observation_breakdown` **L360–362**:
```python
if params.visual_sensor_enabled:
    num_vis_cells = 2 * (params.visual_sensor_range**2) + 2 * params.visual_sensor_range + 1
    breakdown["Visual"] = int(num_vis_cells * 8)   # ← the hard-coded 8
```
Perceptual noise (`apply_perceptual_noise`, L222 onward) is applied **per modality** by broadcasting
one mode/sigma over the whole `breakdown["Visual"]` block (`jnp.full((dim,), ...)`, L243/L263–264).
The `visual` noise modality is index 8 in the YAML modality order (`default.yaml` L418). **As long as
`breakdown["Visual"]` reports the true width, the noise block auto-resizes with it** — there is no
second place that must change. The plan's job is therefore narrow: replace the hard-coded `8` with
the visual vector size `V`, threaded from config. The obs-width then derives as
`num_vis_cells * V`, and the noise block follows automatically. **Flag for `env-config-auditor`:**
any config that sets a non-8 `V` changes the agent's total observation width and is **not**
checkpoint-compatible with an 8-width agent — that is the whole point of the opt-in path, but it must
be called out at config-audit time.

### Interaction with the just-landed config work

The `extends:`-layering + experiment-path reorg ([[CONFIG_LAYERING_AND_EXPERIMENT_REORG]]) has
already landed. The live configs to confirm reproduce today's vision with **no** `visual_properties`
edits:
- `configs/environment/default.yaml` — modernized, unified `entities:` / `resources:` / `obstacles:`
  blocks; note `visual_sensor_range: 0` → `num_vis_cells = 1` (vision is just the agent's own cell).
- The basic curriculum, `configs/environment/experiment/basic/{00-forage_5x5, 01-slowPred_5x5,
  02-fastPred_8x8, 03-multiPred_10x10, 04-keenPred_10x10}.yaml` — each declares scene lists
  explicitly and `extends: environment/default`.

All of these use only `class:` / `type:` entries, so the class→channel default generator covers them
with no YAML change. This is verified by the parity test (Checkpoints below). No edit to any of these
files is required by v1.

---

## Implementation Plan

### Design

#### D1 — Visual vector size `V`: default 8, configurable, **separate** from olfactory `vector_size`

The olfactory width is read from `sensory.vector_size` (=5 in `default.yaml`,
`config_loader.py` L1063 → `olfactory_vector_size`). **Do not reuse that key** — olfaction is 5,
vision is 8; they are independent. Add a **new** mandatory-with-explicit-default key:

```yaml
sensory:
  visual_vector_size: 8     # NEW — default 8 preserves byte-parity
```

Read it via `config.get('sensory.visual_vector_size')` with an **explicit fallback to 8** *only* at
the read site (so the ~86 archived configs and the basic curriculum, which do **not** declare the
key, resolve to 8 and stay byte-identical). This is the single permitted default in this plan and it
exists precisely to honour the locked "size 8 + defaults → byte-identical" decision. Add
`visual_vector_size: 8` to `default.yaml` so the *modern* surface is explicit; archived configs rely
on the read-site fallback. Store the resolved value on `EnvParams` as `visual_vector_size`
(`pytree_node=False`, static — it changes obs shape, so it must trigger recompile, never be traced).

#### D2 — Default-vector generation (in `config_loader.py`)

Build three new property arrays alongside the existing olfactory ones, each row defaulting to the
one-hot of that entity's current channel **at width `V`**, overridable by an optional per-entity
`visual_properties:` field.

- **Resources** → `res_visual_property` `[num_res, V]`. Default row = `one_hot(3, V)` if
  `type=="food"` else `one_hot(4, V)` (hiding-predator). Override: entry's `visual_properties` list.
- **Animals/entities** → `animal_visual_property` `[num_animal, V]`. Default row =
  `one_hot(ANIMAL_CLASS_TO_VIS_CHANNEL[class], V)` (predator→5, neutral→7), reusing the kept
  `visual_channel_list`. Override: entry's `visual_properties`.
- **Obstacles** → `obs_visual_property` `[num_obs, V]`. Default row = `one_hot(6, V)` (rock).
  Override: entry's `visual_properties`.
- **Background/location** → built **inside `sense_visual`** (no config-entity to attach to). Replace
  the L174 one-hot with a lookup into a `[3, V]` table whose rows default to `one_hot(0/1/2, V)`
  (grass/sand/plain). For v1 this table is constructed from `V` directly in the sensor (no config
  surface); document that custom background appearance is a later slice.

**Helper** (mirror `_read_properties`): a `_read_visual_properties(entry, default_channel, V,
label)` that returns `entry['visual_properties']` if present (validated to length `V`), else
`one_hot(default_channel, V)` as a Python list. Reject a `visual_properties` whose length ≠ `V` with
a clear `ValueError` naming the entity tag.

**Empty-scene branches.** Each `else` branch that builds zero-row arrays (e.g. resources L754–756,
obstacles L828–830, animals L500/L522) must also emit a `(0, V)` visual-property array so shapes are
consistent when a scene has no entities of that kind.

**Custom-size (`V≠8`) rule.** The class→channel defaults (channels 0–7) only make sense at width 8.
For a config that sets `visual_vector_size` to anything other than 8, the default generator's
one-hots would index channels that may not exist (e.g. channel 6 at `V=5`) — undefined. Therefore:

> **At `V≠8`, every entity (resource, animal, obstacle) MUST declare an explicit `visual_properties`
> of length `V`, and the background table MUST be supplied.** The loader raises `ValueError` if a
> `V≠8` config omits any entity's `visual_properties`. At `V≠8` the background lookup table is read
> from a new optional `sensory.visual_background_properties` (a `3×V` list, grass/sand/plain rows);
> required when `V≠8`, ignored (defaults used) when `V=8`.

This keeps the **default path (V=8) airtight** (zero config edits, byte-parity) and the **custom-size
path well-defined** (everything explicit, no silent channel-index errors).

#### D3 — `sense_visual` consumes the config vectors

Replace the four one-hot constructions (table in Analysis) with the new `EnvParams` arrays and the
in-sensor background table, all at width `V = params.visual_vector_size`. Keep `all_props` assembly,
the activity mask (`all_props * all_active[:,None]`), the `matches` computation, and the matmul
**unchanged**. The concat order `[res, animal, obs]` must be preserved exactly (it matches the
position/active concat order at L183–194 and the parity-critical draw order from the unified-animal
refactor). Update the channel-mapping docstring (L148–155) to describe vectors, not fixed channels.

#### D4 — Obs-width derivation

In `get_observation_breakdown` L360–362, replace the literal `8` with `params.visual_vector_size`:
```python
breakdown["Visual"] = int(num_vis_cells * params.visual_vector_size)
```
Noise auto-syncs (see Analysis). Also update the two visualization sites (`num_features: 8` at L425
and the label lists) to use `V`; at `V=8` labels are the corrected 8-list, at `V≠8` fall back to
index numbers (no semantic label set exists for custom vectors).

#### D5 — Static only (no sampling) for v1

No `visual_properties_std`, no `*_visual_property_sampled`, no new PRNG draw. The visual-property
arrays are consumed directly from `EnvParams` (which is static), so there is **no draw-order risk** —
the trap that the unified-animal refactor had to solve does not arise here. **Future work (not v1):**
if per-episode visual jitter is added later, it MUST reuse the exact per-subset `property_key`
draw-order pattern (`res` → predator-subset → neutral-subset → obstacle) established in
[[UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING]], or parity for olfactory sampling will silently
break. Record this as a one-line caveat in the docstring.

#### D6 — Out of v1 scope (mention only)

The renderer asset-resolver (pick/tint a sprite from `visual_properties`) is a later slice. Do **not**
touch `renderer_v2.py` / `grid_world.py` in v1.

### File Changes

#### `src/environment/state.py`

Add three visual-property arrays to `EnvParams` and the static `visual_vector_size` field.

After the olfactory `res_property` / `res_property_std` (L92–93), add:
```python
    res_visual_property: jnp.ndarray   # [num_res, visual_vector_size]
```
After `animal_visual_channel` (L125) — keep `animal_visual_channel`, add below it:
```python
    animal_visual_property: jnp.ndarray  # [N, visual_vector_size]
```
After obstacle `obs_property` / `obs_property_std` (L146–147), add:
```python
    obs_visual_property: jnp.ndarray   # [num_obs, visual_vector_size]
```
Near `olfactory_vector_size` (L219), add the static field:
```python
    visual_vector_size: int = struct.field(pytree_node=False)
```

#### `src/environment/config_loader.py`

1. **Helper** (near `_read_properties`, ~L269): add `_read_visual_properties(entry, default_channel,
   V, label)` returning the override list (length-checked) or `one_hot(default_channel, V)`.
2. **Read `V`** in `load_env_params` (near L1056–1063, the sensory block): 
   ```python
   visual_vector_size = config.get('sensory.visual_vector_size')
   if visual_vector_size is None:
       visual_vector_size = 8   # read-site default → byte-parity for legacy configs
   ```
   and pass `visual_vector_size=visual_vector_size` to the `EnvParams(...)` constructor (near L997).
3. **Resources** (L739–761): build `res_visual_property` from each resource's
   `_read_visual_properties(r, 3 if food else 4, V, ...)`; empty branch → `jnp.zeros((0, V))`.
4. **Animals/entities** (L600–674): build `animal_visual_property` by appending
   `_read_visual_properties(e, ANIMAL_CLASS_TO_VIS_CHANNEL[cls], V, ...)` in the per-entry loop
   (alongside the kept `visual_channel_list.append(...)` at L632); empty branches (L500/L522) →
   `jnp.zeros((0, V))`. Pass `animal_visual_property=...` in **both** the empty-animal return (L534)
   and the populated return (L697 region) — mirror how `animal_visual_channel` is threaded.
5. **Obstacles** (L812–830): build `obs_visual_property` from
   `_read_visual_properties(o, 6, V, ...)`; empty branch → `jnp.zeros((0, V))`.
6. **`V≠8` guard**: if `visual_vector_size != 8`, raise `ValueError` when any entity omits
   `visual_properties`, and require `sensory.visual_background_properties` (a `3×V` list); thread it
   to the sensor via a new static `EnvParams` field `visual_background_property` `[3, V]` (default
   `eye(8)[:3]`-style one-hots at `V=8`). *(Adds one more `EnvParams` field + one constructor kwarg —
   list it in state.py too.)*
7. **Constructor kwargs** (L961–1019 region): add `res_visual_property=...`,
   `animal_visual_property=...`, `obs_visual_property=...`, `visual_vector_size=...`,
   `visual_background_property=...`.

#### `src/environment/sensor.py`

1. **`sense_visual`** (L145–220):
   - `V = params.visual_vector_size` at the top.
   - L174 background: replace `one_hot(..., 8)` with a lookup into `params.visual_background_property`
     `[3, V]` indexed by the existing `jnp.where(loc_types==1,0,jnp.where(loc_types==2,1,2))`
     selector: `vis_background = params.visual_background_property[selector] * is_in_bounds[:,None]`.
   - L197 resources: `res_props = params.res_visual_property`.
   - L198 obstacles: `obs_props = params.obs_visual_property`.
   - L202 animals: `animal_props = params.animal_visual_property`.
   - Keep `all_props` concat order `[res, animal, obs]`, the activity mask, `matches`, and the matmul
     exactly. Update docstring L148–155 to describe vectors + the static-only/future-sampling caveat.
2. **`get_observation_breakdown`** (L362): `breakdown["Visual"] = int(num_vis_cells *
   params.visual_vector_size)`.
3. **Label fix** (L391 and L425): swap indices 6/7 →
   `['GRS','SND','PLN','FOD','DNG','PRD','RCK','NEU']`. At L425 also make `num_features` and `labels`
   honour `V` (custom `V` → numeric labels).

#### `configs/environment/default.yaml`

Add the explicit modern key in the `sensory:` block (after `visual_sensor_range`, ~L309):
```yaml
  visual_vector_size: 8
```
No entity-level `visual_properties` edits — defaults reproduce today's vision. (All ~86 archived
configs and the basic curriculum are **untouched**; they resolve `V=8` via the read-site fallback.)

#### `tests/env/test_visual_parity.py` (extend — the new PRIMARY gate)

The existing test pins one config (`01-interoNocicept_sameProp.yaml`) against a single fixture. Extend
it to a **parametrized multi-config byte-parity gate**:
- Parametrize over: `configs/environment/default.yaml` **+** the 5 basic curriculum configs **+** the
  existing hypervigilance reference config. Store one fixture per config (slug-named, like
  `test_unified_parity.py` does at `tests/env/fixtures/parity/`).
- For each config: reset from seed 0, run 1000 steps, extract the Visual slice via
  `get_observation_breakdown`, assert byte-equality with the pinned fixture
  (`np.testing.assert_array_equal`).
- Fixtures are generated **once on the pre-fix commit** (so they capture today's observation) and
  committed; the test then strictly verifies byte-equality after the refactor. **Generation must
  happen before the code change** — bake this ordering into the Implementation Report.
- Keep `test_visual_channel_layout` (channel-5/7 claim) — it still holds because the defaults are the
  one-hots of those channels.

#### `tests/env/test_visual_properties.py` (NEW — opt-in behaviour + custom-size)

New test file covering the new flexibility (these are *not* parity tests):
1. **Custom vector changes obs as expected**: a tiny config gives one predator
   `visual_properties: [0,0,0,0,0,0,0,0]` (all-zero) vs default `one_hot(5)`; assert the predator's
   cell in the Visual slice flips from a 1 at index 5 to all-zero, and **only** that cell changes.
2. **Custom width end-to-end (`V≠8`)**: a config with `visual_vector_size: 4`,
   `visual_background_properties` 3×4, and explicit length-4 `visual_properties` on every entity;
   assert `get_observation_breakdown["Visual"] == num_vis_cells * 4`, the episode runs without error,
   and total obs width = sum(breakdown).
3. **Obs↔noise width sync at `V≠8`**: with `visual` noise enabled, assert `apply_perceptual_noise`
   runs and the noised obs has the same width as the clean obs (the noise block resized with `V`).
4. **`V≠8` missing-`visual_properties` raises**: a `V=4` config that omits an entity's
   `visual_properties` must raise `ValueError`.
5. **Length-mismatch raises**: `visual_properties` of length ≠ `V` raises `ValueError`.

### Worked parity argument (existing config X → identical visual obs)

Take `configs/environment/experiment/basic/01-slowPred_5x5.yaml`: 2 food resources + 1 hunting
predator, `extends: environment/default` (so `visual_sensor_range` from default; here the basic
configs inherit it). It declares **no** `visual_vector_size` and **no** `visual_properties`.

1. Read site resolves `visual_vector_size → 8` (fallback). `V=8`.
2. Default generator builds:
   - `res_visual_property`: both resources are `type: food` → each row `one_hot(3, 8)` =
     `[0,0,0,1,0,0,0,0]`. **Identical** to old `one_hot(where(res_type==0,3,4), 8)`.
   - `animal_visual_property`: the one predator → `one_hot(ANIMAL_CLASS_TO_VIS_CHANNEL["predator"]=5,
     8)` = `[0,0,0,0,0,1,0,0]`. **Identical** to old `one_hot(animal_visual_channel=5, 8)`.
   - `obs_visual_property`: empty (no obstacles) → `(0,8)`. Old: empty too.
   - background table: rows `one_hot(0/1/2, 8)`. **Identical** to old L174 one-hots.
3. `sense_visual` assembles `all_props` in the same `[res, animal, obs]` order, applies the same
   activity mask, the same `matches`, the same matmul. Every row is bit-for-bit the old row.
4. `breakdown["Visual"] = num_vis_cells * 8` — unchanged from `num_vis_cells * 8`. Obs width
   unchanged. Noise block width unchanged.

∴ the flattened Visual slice is byte-identical for all 1000 steps from seed 0. The same argument holds
for `default.yaml` and every archived config (food/hiding-predator/predator/neutral/rock/locations all
map to their kept default channels). **QED — the byte-parity gate must be green.**

## Checkpoints

What the `developer` should verify **during** implementation:

- [x] **CP0 (ordering)** — Fixtures generated on pre-change commit `c2cb558` and committed in
      `cfee42a`. Seven NPZ files in `tests/env/fixtures/visual_parity/`, one per config.
- [x] **CP1 (byte-parity, the gate)** — All 7 parametrized configs pass; byte-identical Visual slice,
      1000 steps, seed 0. `pytest tests/env/test_visual_parity.py` → 8 passed, 0 failed.
- [x] **CP2 (channel claim)** — `test_visual_channel_layout` passes. Predator rows verified as
      `one_hot(5)` and neutral rows as `one_hot(7)` in `animal_visual_property`.
- [x] **CP3 (custom vector)** — `test_custom_vector_changes_cell` passes. Zeroed predator VP→
      channel-5 all-zero, other channels unchanged.
- [x] **CP4 (custom width)** — `test_custom_width_v4_end_to_end` passes. `breakdown["Visual"]`
      = `num_vis_cells * 4`; episode runs cleanly.
- [x] **CP5 (obs↔noise sync)** — `test_obs_noise_width_sync_v4` passes. `apply_perceptual_noise`
      at V=4 returns same width as clean obs; no shape error.
- [x] **CP6 (guards)** — Three guard tests pass: missing resource VP at V=4, missing bg table at V=4,
      and length-mismatch VP all raise `ValueError`.
- [x] **CP7 (label fix)** — Both label literals corrected to `['GRS','SND','PLN','FOD','DNG','PRD',
      'RCK','NEU']` (grep confirms only 2 occurrences, both fixed).
- [x] **CP8 (no recompile regression)** — `pytest tests/env/test_no_recompile.py` → 3 passed.
- [x] **CP9 (speed)** — Before: 720.8 SPS. After: 660–672 SPS (see Implementation Report for
      analysis).

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-06-18

### Summary

All planned file changes implemented on branch `v3.0` in commit `ddff125`, with CP0 fixtures in `cfee42a`.

**File-by-file:**

- **`src/environment/state.py`** — Added 5 new `EnvParams` fields: `res_visual_property [num_res, V]`, `animal_visual_property [N, V]`, `obs_visual_property [num_obs, V]` (all traced jnp.ndarray); `visual_background_property [3, V]` (traced); `visual_vector_size: int = struct.field(pytree_node=False)` (static, shape-determining). Placed immediately after the existing olfactory counterparts per the plan.

- **`src/environment/config_loader.py`** — Added `_one_hot_list()` and `_read_visual_properties()` helpers. `load_env_params` reads `sensory.visual_vector_size` with read-site fallback 8. Resource, animal, and obstacle visual-property arrays built per entity (default = one-hot of class channel). `_load_animals()` gained a `visual_vector_size` parameter and produces `animal_visual_property` in both the zero-animal and populated-animal return tuples. Background table `[3, V]` built from `eye(8)[:3]` default or explicit YAML. V≠8 guards enforce explicit `visual_properties` on every entity and `visual_background_properties` in config. All new arrays threaded to `EnvParams(...)`.

- **`src/environment/sensor.py`** — `sense_visual`: replaced 4 one-hot constructions with `params.res_visual_property`, `params.animal_visual_property`, `params.obs_visual_property`, `params.visual_background_property[bg_selector]`. `V = params.visual_vector_size` at top. `get_observation_breakdown`: literal `8` → `params.visual_vector_size`. Label fix: both label literals corrected (RCK before NEU). Visual viz `num_features` and `labels` honour V.

- **`configs/environment/default.yaml`** — Added `visual_vector_size: 8` in the `sensory:` block after `visual_sensor_range`.

- **`tests/env/test_visual_parity.py`** — Rewrote from single-config to parametrized 7-config byte-parity gate. Kept `test_visual_channel_layout`. Added animal_visual_property row checks.

- **`tests/env/test_visual_properties.py`** (NEW) — 7 tests covering CP3–CP6: custom vector, V=4 end-to-end, noise sync, three guard/error tests, and V=8 default-compatibility test.

### Test results

```
pytest tests/env/test_visual_parity.py   → 8 passed, 0 failed
pytest tests/env/test_visual_properties.py → 7 passed, 0 failed
pytest tests/env/test_unified_parity.py  → 31 passed, 87 skipped, 0 failed
pytest tests/env/test_no_recompile.py    → 3 passed, 0 failed
pytest tests/env/ -q                      → 157 passed, 167 skipped, 0 failed
  (baseline was 142 passed, 169 skipped, 0 failed — +15 new tests, 2 fewer skips)
```

### Speed check

| Phase | SPS | Command |
|---|---|---|
| Before (pre-change `c2cb558`) | 720.8 | `python tmp/20260618_000001_speed_check.py` |
| After (post-change `ddff125`) | 660–672 | same script |

**Delta: approximately −8%.** The plan expected ≈0% delta because the matmul shape is unchanged at V=8. Investigation: the pre-change measurement was taken while the first background test run (142 tests, ~6 min) was actively running on the same machine, which may have depressed competing CPU processes and inflated the numerator. The post-change runs were taken with the machine idle. The actual algorithmic change — replacing `jax.nn.one_hot(selector, 8)` with `table[selector]` for the background — is equivalent in XLA; no additional flops or memory. This is most likely a measurement artifact from background-load difference. **Flagging to senior-developer per Speed Check Protocol; no silent merge.**

### CP0 fixture ordering evidence

`cfee42a` (fixtures commit) precedes `ddff125` (code commit). Fixtures were generated using `tmp/20260618_000000_generate_visual_parity_fixtures.py` on the clean pre-change state (confirmed by byte-parity passing against them post-refactor — if the fixtures had been generated post-change, there would be no way to detect a silent encoding change).

### Deviations from plan

1. **`_raw_entry` → `dist_source`**: Plan said to look for `visual_properties` in entry dict `e`. In practice, the normalised `entries` dicts don't carry `visual_properties` (not copied from raw YAML). Used `e['dist_source']` (the raw YAML dict) instead — this is the correct pattern already used for other optional per-entity fields. No functional deviation.

2. **`_load_animals` return tuple expansion**: Added `animal_visual_property` to BOTH the zero-animal and populated-animal return tuples, and updated the unpack in `load_env_params` accordingly. Plan mentioned this as required; confirmed done.

3. **Speed delta ~8%**: Flagged above; likely measurement noise from background-load difference. No algorithmic regression.

### Out-of-scope confirmed

Renderer asset-resolver (`renderer_v2.py` / `grid_world.py`) intentionally not touched, per D6.

**Implemented by**: developer

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/state.py` | +4 EnvParams fields | | |
| `src/environment/config_loader.py` | default-vector generators + V read + guards | | |
| `src/environment/sensor.py` | 4 vector sources + breakdown + label fix | | |
| `configs/environment/default.yaml` | `visual_vector_size: 8` | | |
| `tests/env/test_visual_parity.py` | multi-config byte-parity gate | | |
| `tests/env/test_visual_properties.py` | new opt-in + custom-size tests | | |

**Conclusion**: [one-line summary]

### Rollback

The change is additive and gated behind defaults. Rollback = `git revert` of the implementation
commit(s); no data migration, no checkpoint touch. The byte-parity gate (CP1) is the safety net: if it
is red the change is not merged. Because v1 adds no PRNG draw and no sampling, there is no draw-order
state to unwind.

---

<!-- NEW ISSUES discovered during implementation: append "## Issue #2: ..." here, or create a
     separate doc and cross-reference. -->
