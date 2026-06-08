# 03 — Entity Placement

> **Source**: `src/environment/core.py` (`jax_reset` line 762, `resolve_overlaps_global` line 662, `place_in_area` line 711), `src/environment/state.py`, `src/environment/config_loader.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

At the start of every episode, all entities — resources, animals (predators and neutral animals unified into a single array), obstacles — must be placed on the grid without overlapping each other. This is done inside `jax_reset` (`core.py:762`), which is decorated with `@jax.jit` and is entirely pure-functional.

**v2.0 unified animal entity**: the old separate `pred_*` / `neutral_*` arrays are gone. All animals now live in a single `animal_*` block in `EnvState` / `EnvParams`, ordered **predators first, then neutral animals**. Two static tuples, `params.predator_indices` and `params.neutral_indices`, hold the integer positions (within that unified array) that belong to each class — these are the key to understanding how placement, property sampling, and later step logic split the array by class.

Two placement modes are available, selected by `params.placement_mode` (YAML key `environment.placement.mode`, default `per_entity`):

- **`per_entity`**: each entity's position is sampled individually, then a single sequential scan resolves all collisions.
- **`per_type`**: entities are pre-grouped by their spawn-area bounding box; a `lax.scan` places one group at a time, updating a shared occupancy mask so later groups avoid already-taken cells.

After placement, `jax_reset` initialises body state and per-episode animal behavioural parameters (randomised from low/high ranges stored in `params`), then returns an `EnvState`. It does **not** return an initial observation — the caller must call `get_observation(state, params)` explicitly.

---

## `jax_reset`

**Signature**: `jax_reset(params: EnvParams, key: PRNGKey) → EnvState` (`core.py:762`)

### PRNG key splitting

```
key, agent_key, placement_key, body_key, property_key = jax.random.split(key, 5)
animal_episode_key = jax.random.fold_in(property_key, 0xAE1)
```

Five independent subkeys are derived from the master seed:
- `agent_key` — agent position sampling.
- `placement_key` — entity placement (further split 6-way inside the `per_entity` branch, or 2-way inside `per_type`).
- `body_key` — nutrition / injury initialisation.
- `property_key` — per-episode olfactory-property vector noise (split 4-way: res / pred / obs / neutral).
- `animal_episode_key` — per-episode behavioural parameter sampling (detect range, stamina, etc.); derived via `fold_in` so the existing four key streams remain byte-identical to pre-refactor code.

### Call sequence

1. **Agent position** (`core.py:786–787`): if `params.random_start_pos`, sample uniformly in `[0, H) × [0, W)` using `agent_key`; else use `params.start_pos` (0-based, converted from 1-based YAML by `config_loader`).

2. **Entity counts** (`core.py:790–795`): read static sizes from params —
   - `num_res = params.res_type.shape[0]`
   - `num_pred_class = len(params.predator_indices)` (animals whose class is `'predator'`)
   - `num_neutral_class = len(params.neutral_indices)` (animals whose class is `'neutral'`)
   - `num_obs = params.obs_blocking.shape[0]`

3. **Entity placement** (`core.py:797–883`): dispatch to `per_entity` or `per_type` branch — see sections below.

4. **Split positions back** (`core.py:885–889`): `all_positions` is a flat `[num_total, 2]` array in the canonical `[res | pred_class | obs | neutral_class]` ordering used during placement. Slice back by index arithmetic:
   - `res_pos = all_positions[:num_res]`
   - `pred_pos = all_positions[num_res : num_res + num_pred_class]`
   - `obs_pos = all_positions[num_res + num_pred_class : num_res + num_pred_class + num_obs]`
   - `neutral_pos = all_positions[num_res + num_pred_class + num_obs :]`

5. **Assemble unified `animal_pos`** (`core.py:891–904`): allocate `animal_pos_init = zeros([N, 2])` then scatter using the static index tuples:
   - `animal_pos_init[predator_indices] = pred_pos` (if any predators)
   - `animal_pos_init[neutral_indices] = neutral_pos` (if any neutrals)

   This produces a unified `[N, 2]` array with predators and neutrals at their correct positions, in predators-first order.

6. **Body initialisation** (`core.py:906–925`):
   - **Nutrition**: if `random_start_nutrition`, sample uniform in `[max_nutrition/2, max_nutrition]` using `body_key2`; else use `params.start_nutrition`.
   - **Satiation**: always derived from nutrition via the power-law `S = max_S × (N/max_N)^k` where `k = params.nutrition_to_satiation_scaling_factor`. Never set independently.
   - **Injury**: if `random_start_injury`, sample uniform in `[0, max_injury/2]` using `body_key3`; else `0.0`.
   - `injury_buffer`: zero-initialised, length `params.smoothing_duration`.
   - `nociception_history_buffer`: zero-initialised, length `params.interoceptive_kernel_length`.

7. **Property sampling** (`core.py:927–953`): `property_key` is split 4-way into `prop_key_res`, `prop_key_pred`, `prop_key_obs`, `prop_key_neutral`. For each entity group, a Gaussian noise vector is added to the mean property: `clip(mean + std × N(0,1), 0, 1)`. Predator-class and neutral-class subsets are sampled separately using their respective subkeys, then scattered back into the unified `[N, V]` array using `predator_indices` / `neutral_indices`.

8. **Per-episode behavioural parameter sampling** (`core.py:957–978`): `animal_episode_key` is split into 5 subkeys, one per behavioural field. Each field is sampled per-animal from a uniform distribution:
   - `animal_detect_sampled[i]` ~ Uniform(`animal_detect_low[i]`, `animal_detect_high[i]`)
   - `animal_max_stamina_sampled[i]` ~ Uniform(`animal_max_stamina_low[i]`, `animal_max_stamina_high[i]`)
   - `animal_recovery_sampled[i]` ~ Uniform(`animal_recovery_low[i]`, `animal_recovery_high[i]`)
   - `animal_hunt_thresh_sampled[i]` ~ Uniform(`animal_hunt_thresh_low[i]`, `animal_hunt_thresh_high[i]`)
   - `animal_lose_interest_sampled[i]` ~ Uniform(`animal_lose_interest_low[i]`, `animal_lose_interest_high[i]`)

   For wander / static animals the YAML stores `[0, 0]` for all five ranges, so the draws are exactly `0.0` — harmless and shape-stable.

9. **Construct `EnvState`** (`core.py:980–1012`): all animals start in state `0` (Patrol / idle), stamina = `animal_max_stamina_sampled` (i.e. full stamina at episode start), all timers zero, all resources active, `last_action` set to the rest-action index (if enabled) or the eat-action index.

---

## `per_entity` Mode

**Code**: `core.py:797–848`

### PRNG key splitting

```python
placement_key, res_key, pred_key, obs_key, neutral_key, resolve_key = \
    jax.random.split(placement_key, 6)
```

### Steps

1. For each entity group (resources, predator-class animals, obstacles, neutral-class animals), split the group key into per-entity subkeys and `vmap` `jax.random.randint` over that group's spawn-area rows to sample an initial `(row, col)` per entity. Edge cases: if a group has zero entities, an empty `[0, 2]` array is produced directly.

2. **Predator-class animals**: spawn areas are gathered on-the-fly from `params.animal_spawn_area` using `predator_indices`:
   ```python
   pred_sa = params.animal_spawn_area[jnp.array(list(params.predator_indices))]
   ```
   Neutral-class animals are handled identically with `neutral_indices`.

3. Concatenate all positions and spawn areas in the canonical order:
   ```python
   all_positions   = concat([res_pos, pred_pos, obs_pos, neutral_pos])
   all_spawn_areas = concat([res_spawn_area, pred_sa_all, obs_spawn_area, neutral_sa_all])
   ```

4. Call `resolve_overlaps_global(all_positions, all_spawn_areas, H, W, resolve_key)` to eliminate same-cell collisions (see below).

**Best for**: small grids or low entity counts. The initial vmap step is O(1) XLA instructions regardless of N.

---

## `per_type` Mode

**Code**: `core.py:850–883`

### Group construction (config-load time)

`load_env_params` in `config_loader.py:751–800` groups entities by their spawn-area bounding box at config-load time (Python/NumPy, not JAX). All entities sharing the exact same `(min_r, min_c, max_r, max_c)` tuple form one type group. The result is stored as static params fields:

| Field | Shape | Meaning |
|---|---|---|
| `params.type_areas` | `[T, 4]` | Spawn area per group |
| `params.type_counts` | `[T]` | Number of entities in each group |
| `params.type_entity_map` | `[T, max_per_type]` | Global entity indices for each group (padded to `max_per_type`) |
| `params.max_per_type` | scalar (static) | Maximum entities in any single group; XLA loop bound |
| `params.num_types` | scalar (static) | Number of groups; `lax.scan` iteration count |

Entity global indices follow the canonical `[res | pred_class | obs | neutral_class]` ordering (matching `per_entity`), so position arrays can be sliced back consistently in step 4.

### Steps at reset time

1. Initialise `all_positions = zeros([num_entities, 2])` and `occupancy = zeros([H*W], bool)`.
2. Run `jax.lax.scan` over `jnp.arange(params.num_types)`. Each scan step:
   a. Split a fresh subkey from `scan_key`.
   b. Read `area = params.type_areas[type_idx]` and `count = params.type_counts[type_idx]`.
   c. Call `place_in_area(subkey, area, occ, count, max_per_type, H, W)` — returns `(positions [max_per_type, 2], flat_indices [max_per_type])`.
   d. For each slot `j < max_per_type`: if `j < count`, mark `flat_indices[j]` as occupied in `occ` and write `positions[j]` to `all_positions[entity_indices[j]]`.
3. The final `all_positions` is returned.

**Best for**: large grids or dense configs where many entities share a spawn area. The `lax.scan` compiles to a fixed `num_types`-iteration loop instead of a `num_entities`-iteration loop, which is cheaper when many entities map to few groups.

---

## `resolve_overlaps_global`

**Signature**: `resolve_overlaps_global(all_positions, all_spawn_areas, grid_height, grid_width, key) → all_positions` (`core.py:662`)

Used by the `per_entity` branch to resolve collisions after the vmap initial-position sampling.

### Algorithm

1. Allocate a flat `occupancy [H*W]` boolean mask (all False).
2. Generate a single random permutation of all `H*W` cell flat-indices (`global_perm`).
3. Run `jax.lax.scan` over entities in order (index `i` counting up from 0). For entity `i`:
   a. Compute `flat_idx = positions[i, 0] * W + positions[i, 1]`.
   b. Check `is_taken = occupancy[flat_idx]`.
   c. Build `in_area` mask: cells where `min_r ≤ row < max_r` and `min_c ≤ col < max_c` (exclusive upper bound on both axes).
   d. `valid = in_area & ~occupancy`.
   e. Identify the first valid cell in the shuffled permutation: `first_valid_mask = valid[global_perm] & (cumsum(valid[global_perm]) == 1)`.
   f. `replacement_flat = where(first_valid_mask, global_perm, 0).sum()` — gives the flat index of the first valid cell, or `0` if none exists.
   g. `new_flat = where(is_taken, replacement_flat, flat_idx)` — only relocate if the original cell was taken.
   h. Mark `new_flat` as occupied; update `positions[i]`.
4. Return resolved `all_positions`.

### Properties

- **Entity order** (implicit priority): resources first, then predator-class animals, then obstacles, then neutral-class animals (the concat order from `per_entity` mode). Earlier entities keep their sampled cell if unoccupied; later entities are relocated if there is a collision.
- **Single global permutation**: all entities share the same `global_perm`, generated once before the scan. This means the "next available cell" search is reproducible given the same `resolve_key`, but is not independent per-entity.
- **Fallback on area exhaustion**: if no valid unoccupied cell exists within an entity's spawn area, `replacement_flat = 0` (grid origin `(0, 0)`). Multiple entities can silently stack on cell `(0, 0)`. This is a known limitation — avoid by ensuring every spawn area has strictly more cells than assigned entities.
- **Spawn-area bounds**: `in_area` checks `min_r ≤ row < max_r` and `min_c ≤ col < max_c`. The upper bounds are exclusive (consistent with the `_parse_area` conversion in `config_loader.py:243`).

---

## `place_in_area`

**Signature**: `place_in_area(subkey, area, occupancy, num_entities, max_entities, grid_height, grid_width) → (positions [max_entities, 2], flat_indices [max_entities])` (`core.py:711`)

Used by the `per_type` branch. Places up to `num_entities` non-overlapping entities within a rectangular area, respecting cells already marked in `occupancy`.

### Algorithm

1. Build `in_area` mask: all cells where `min_r ≤ row < max_r` and `min_c ≤ col < max_c`.
2. `valid = in_area & ~occupancy`.
3. Generate a random permutation of all `H*W` cells (`perm = jax.random.permutation(subkey, H*W)`).
4. Filter `perm` to valid cells: `valid_in_perm = valid[perm]`.
5. Take the first `num_entities` valid cells via cumsum: `selected = valid_in_perm & (cumsum(valid_in_perm) <= num_entities)`.
6. Map selected cells to flat indices with a sentinel for unselected: `selected_flat = where(selected, perm, H*W)`, sort ascending, take `[:max_entities]`.
7. Pad to `max_entities`: slots `j ≥ num_entities` are set to flat index `0`.
8. Extract `(row, col)` from flat indices and stack into `positions [max_entities, 2]`.

The caller (`per_type` scan body, `core.py:865`) guards unused slots with `valid = arange(max_per_type) < count` before writing to `all_positions` and `occupancy`.

### Spawn-area coordinate convention

`area` is a 1-D array `[min_r, min_c, max_r, max_c]` (0-based, upper bounds exclusive). This is produced by `_parse_area` in `config_loader.py:241–243`:

```python
def _parse_area(area, default_h, default_w):
    a = area if area is not None else [[1, 1], [default_h, default_w]]
    return [a[0][0]-1, a[0][1]-1, a[1][0], a[1][1]]
```

YAML `spawn_area: [[r1, c1], [r2, c2]]` (1-based, inclusive both ends) is converted to 0-based `[r1-1, c1-1, r2, c2]` where the upper bound is left exclusive. The number of cells in the area is therefore `(r2 - r1 + 1) × (c2 - c1 + 1)`.

---

## Unified Animal Entity and the Index Tuples

The v2.0 refactor unified all animals into a single `animal_*` block. Four static tuples in `EnvParams` (stored as `pytree_node=False`, not JAX arrays) tell the code which rows belong to which class/behaviour:

| Tuple | Length | Meaning |
|---|---|---|
| `predator_indices` | N_pred_class | Rows in `animal_*` arrays that are predator-class |
| `neutral_indices` | N_neutral_class | Rows in `animal_*` arrays that are neutral-class |
| `hunt_idx` | N_hunt | Rows whose behaviour is `'hunt'` |
| `wander_idx` | N_wander | Rows whose behaviour is `'wander'` |
| `static_idx` | N_static | Rows whose behaviour is `'static'` |

`predator_indices` / `neutral_indices` are used in `jax_reset` for placement and property sampling. `hunt_idx` / `wander_idx` / `static_idx` are used in `update_animals` (the step function) to dispatch to the correct movement/attack logic.

In the **legacy YAML schema** (`environment.predators:` + `environment.neutral_animals:`), all predators are loaded first with `behaviour='hunt'`, followed by all neutrals with `behaviour='wander'`. So `predator_indices == hunt_idx` and `neutral_indices == wander_idx` for all 86 pre-v2.0 configs.

In the **new unified schema** (`environment.entities:`), class and behaviour are set independently per entry, so a predator-class entity could in principle have `behaviour='wander'` and vice versa.

A host-side helper `select_by_class(params, class_name)` (`state.py:7`) returns a NumPy boolean mask for analysis scripts and renderers — it is not a JAX-traced function.

Legacy accessors `params.predator_tags` / `params.neutral_tags` are still available as `@property` aliases on `EnvParams` (`state.py:234–247`), derived from `animal_tags` filtered by class. These will be removed in a future release; direct consumers should migrate to `animal_tags + predator_indices / neutral_indices`.

---

## Config-Loader: Spawn-Area Coordinate Conversion

All spawn-area coordinates in YAML are **1-based, inclusive** at both ends. `config_loader.py` converts them to **0-based, exclusive-upper-bound** before storing in `EnvParams`:

```
YAML: spawn_area: [[r1, c1], [r2, c2]]   (1-indexed rows/columns, both bounds inclusive)
                        ↓  _parse_area()
params: [r1-1, c1-1, r2, c2]             (0-indexed, upper bound exclusive)
```

The same conversion is applied for `patrol_area`, `res_spawn_area`, `obs_spawn_area`, and `animal_spawn_area`. The `start_pos` conversion follows the same rule: `start_pos = array(yaml_start_pos) - 1`.

`config_loader.py` does **not** validate that spawn areas lie within the grid. Out-of-bounds areas (e.g. `max_r > H`) result in sampled positions that are off-grid; this is a silent failure that should be caught by config review.

`placement_mode` is read at `config_loader.py:779`:
```python
placement_mode = config.get('environment.placement.mode', 'per_entity')
assert placement_mode in ('per_entity', 'per_type'), f"Unknown placement mode: {placement_mode}"
```
The default is `'per_entity'` when the key is absent. An invalid value raises `AssertionError` immediately at config load.

---

## Random Start Conditions

Controlled by static boolean flags in `EnvParams`:

| Flag | Effect when True | Range |
|------|-----------------|-------|
| `random_start_pos` | Agent spawns at uniformly random grid cell | `[0, H) × [0, W)` |
| `random_start_nutrition` | Initial nutrition sampled uniformly | `[max_nutrition/2, max_nutrition]` |
| `random_start_injury` | Initial injury sampled uniformly | `[0, max_injury/2]` |
| `random_start_satiation` | Loaded but never used independently | — |

When disabled, the deterministic starting values are `start_nutrition`, `start_pos` (0-based from config loader), and injury `0.0`. Satiation is always derived from nutrition via the power-law scaling factor — `start_satiation` is loaded but ignored at runtime.

> **Note**: `start_satiation` is **loaded but never used** — satiation at reset is always derived from nutrition via `S = max_S * (N/max_N)^k`. See doc `01` FAQ.

---

## Clarifications / FAQ

**Q: Can the agent spawn on top of an entity?**
A: **Yes.** Agent placement is computed before entity placement (`core.py:786–787`) and the agent's cell is never added to the occupancy mask. An agent spawned with `random_start_pos=True` can land on a predator, obstacle, food, etc. The step loop's contact logic fires on the very first step.

**Q: What's the entity processing order in `resolve_overlaps_global`?**
A: Resources first, then predator-class animals, then obstacles, then neutral-class animals (the concatenation order). Earlier entities get first pick — if a resource and a predator-class animal sample the same cell, the resource keeps it and the predator is relocated.

**Q: In the unified animal array, are predators guaranteed to come before neutrals?**
A: Yes — `_load_animals` in `config_loader.py` builds the `entries` list with legacy predators first, then legacy neutrals. `predator_indices` are therefore always `[0, 1, ..., N_pred-1]` and `neutral_indices` are `[N_pred, ..., N-1]` for all 86 pre-v2.0 configs. With the new `entities:` schema the order matches the YAML declaration order.

**Q: What happens when a spawn area has fewer free cells than entities assigned to it?**
A: In `resolve_overlaps_global`, the "first valid cell" search falls back to `replacement_flat = 0` (grid origin `(0, 0)`) when no valid cell exists, because `jnp.where(first_valid_mask, global_perm, 0).sum()` returns `0` when `first_valid_mask` is all-False (`core.py:694`). Multiple entities silently stack on cell `(0, 0)`. In `place_in_area`, positions beyond `num_entities` are explicitly padded to flat index `0` (`core.py:749–753`) — the caller guards against writing these with the `valid = arange(max_per_type) < count` check. Guard against the exhaustion case in config design: every spawn area must have strictly more cells than entities assigned to it.

**Q: Why does `per_entity` mode get called "fast on small grids" but `per_type` wins on large grids?**
A: `per_entity`'s `lax.scan` runs `N_total` steps (one per entity), each step touching an `H*W`-length occupancy vector. `per_type` runs `num_types` steps, one per spawn-area group. When many entities share few areas (e.g. 50 food items in one bounding box), `per_type` reduces the scan to `num_types` steps. For small configs with few entities the grouping overhead is not worth it.

**Q: Do the two modes produce the same placement given the same seed?**
A: No — they use different sampling algorithms and consume `placement_key` differently. Switching modes changes reset-time layouts. If cross-mode reproducibility matters, also change the seed.

**Q: When `num_res == 0` (no resources), does the code still work?**
A: Yes. Empty arrays (`shape [0, ...]`) are handled by vmap / concatenation correctly in both modes. The same holds for zero predators, zero neutrals, and zero obstacles.

**Q: Is `jax_reset` cheap enough to call every episode?**
A: After the first JIT compile, yes — the whole function is pure JAX and executes in one kernel launch per device. In `ParallelEnv`, `jax_reset` is vmapped over a batch of environments, so all envs reset in parallel with no Python-level loop.

**Q: What's the return shape difference between `jax_reset` and `jax_step`?**
A: `jax_reset` returns `EnvState` only. `jax_step` returns `(new_state, reward, done, info)`. Call `get_observation(state, params)` explicitly after reset to get the first observation (see doc `09_sensors_and_observation.md`).

**Q: Are `agent_key`, `placement_key`, and `property_key` truly independent?**
A: Yes. `jax.random.split(key, 5)` produces five independent subkeys. `animal_episode_key` is derived via `fold_in(property_key, 0xAE1)` — it is correlated with `property_key` in the cryptographic sense but practically independent for sampling purposes. The agent's position, entity positions, and entity properties are statistically independent given the master seed.

**Q: What's stored in `occupancy` — Boolean per cell or entity index?**
A: Boolean per cell (`[H*W]`). The mask tracks "any entity already here" — it does not record which entity. This is sufficient for collision avoidance but means you cannot reconstruct placement order from the mask alone.

**Q: Why is `max_per_type` a static field?**
A: It is the loop bound for `place_in_area`'s padding and for `lax.scan`'s output shape in `per_type` mode. Making it static lets XLA unroll the inner loop and keeps output tensor shapes invariant across batches. Changing `max_per_type` triggers recompilation.

**Q: Can I disable random start for just one dimension?**
A: Yes — each flag is independent. Common pattern: fix `start_pos` for reproducibility but randomise nutrition / injury for domain randomisation. See doc `01` Reset Values table.

**Q: What happens if the spawn area is outside the grid (e.g. `max_r > H`)?**
A: The sampler uses the raw bounds without clipping (`core.py:688–689` for `resolve_overlaps_global`, `core.py:734–735` for `place_in_area`). If `max_r > H`, sampled rows can be ≥ H and the resulting flat index will be off-grid. Config loader does not validate this — keep spawn areas strictly within grid bounds.

**Q: What is `animal_is_damaging` and how does it relate to placement?**
A: `params.animal_is_damaging` is a precomputed boolean array `[N]` set to `True` for predator-class animals and `False` for neutral-class animals (from `ANIMAL_DAMAGING_CLASSES = {"predator"}` in `config_loader.py:34`). It is not used during placement — placement only uses `predator_indices` and `neutral_indices`. `animal_is_damaging` is used by the step function to decide whether a contact event applies damage.
