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

## `resolve_overlaps_global`

**Plain-English purpose**: After the initial `vmap`-based position sampling (which may put two entities on the same cell), this function does one sequential pass over all entities and relocates any that land on an already-occupied cell. It pre-generates a single random permutation of all grid cells (`global_perm`) and uses that to pick the first free cell within the colliding entity's spawn area — so the replacement is both reproducible (given the same key) and uniform over the valid area.

**Signature**: `resolve_overlaps_global(all_positions, all_spawn_areas, grid_height, grid_width, key) → all_positions` (`core.py:662`)

Source: `src/environment/core.py:662–708`

```python
def resolve_overlaps_global(
    all_positions: jnp.ndarray,
    all_spawn_areas: jnp.ndarray,
    grid_height: int,
    grid_width: int,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Resolve entity position overlaps via single-pass sequential scan.
    
    Uses a flat boolean occupancy mask. For each entity, if its cell is
    already taken, picks a random free cell within its spawn area using
    a pre-shuffled global permutation. All ops are JIT-compatible.
    """
    num_entities = all_positions.shape[0]
    total_cells = grid_height * grid_width
    occupancy = jnp.zeros(total_cells, dtype=jnp.bool_)
    global_perm = jax.random.permutation(key, total_cells)
    cell_rows = jnp.arange(total_cells) // grid_width
    cell_cols = jnp.arange(total_cells) % grid_width
    
    def resolve_one(carry, _unused):
        occ, positions, i = carry
        flat_idx = positions[i, 0] * grid_width + positions[i, 1]
        is_taken = occ[flat_idx]
        
        min_r, min_c, max_r, max_c = all_spawn_areas[i]
        in_area = (cell_rows >= min_r) & (cell_rows < max_r) & \
                  (cell_cols >= min_c) & (cell_cols < max_c)
        valid = in_area & (~occ)
        
        valid_in_perm = valid[global_perm]
        first_valid_mask = valid_in_perm & (jnp.cumsum(valid_in_perm) == 1)
        replacement_flat = jnp.where(first_valid_mask, global_perm, 0).sum()
        
        new_flat = jnp.where(is_taken, replacement_flat, flat_idx)
        new_r = new_flat // grid_width
        new_c = new_flat % grid_width
        
        positions = positions.at[i].set(jnp.array([new_r, new_c]))
        occ = occ.at[new_flat].set(True)
        return (occ, positions, i + 1), None
    
    init_carry = (occupancy, all_positions, jnp.array(0))
    (_, all_positions, _), _ = jax.lax.scan(
        resolve_one, init_carry, None, length=num_entities
    )
    return all_positions
```

> **API notes**
>
> - `lax.scan` here: [primer: lax.scan](00_jax_primer.md#scan). The carry is `(occ [H*W bool], positions [N,2], i scalar)`. The `_unused` second argument is `None` because the scan has no per-step input array — `length=num_entities` drives the loop count while all per-step data comes from the carry. `i` advances each step and is used to index into `positions` dynamically.
> - `positions.at[i].set(...)`: [primer: immutability](00_jax_primer.md#immutability). In NumPy you would write `positions[i] = new_pos`. In JAX all arrays are immutable; `.at[idx].set(val)` returns a new array with that slot updated. The old array is unchanged.
> - `jnp.cumsum(valid_in_perm) == 1`: [primer: masking](00_jax_primer.md#masking). This is the standard branchless idiom for "find the first True in a boolean array": `cumsum` turns the first `True` at index `k` into `[0,…,0,1,2,…]`; `== 1` isolates exactly position `k`. Combined with `jnp.where(first_valid_mask, global_perm, 0).sum()`, this extracts the flat cell index of that first free slot.
> - `jnp.where(is_taken, replacement_flat, flat_idx)`: [primer: branchless](00_jax_primer.md#branchless). JAX cannot branch on a traced boolean (`is_taken` is a JAX array, not a Python bool). `jnp.where` selects element-wise with no Python `if`, keeping the function JIT-compatible.
> - `global_perm` is generated **once** before the scan and shared across all iterations. This is intentional: the permutation provides a single consistent tiebreaking order for the whole reset. All entities see the same shuffled cell ordering; the random variety comes from which cells are still free at each step.

### Properties

- **Entity order** (implicit priority): resources first, then predator-class animals, then obstacles, then neutral-class animals (the concat order from `per_entity` mode). Earlier entities keep their sampled cell if unoccupied; later entities are relocated if there is a collision.
- **Single global permutation**: all entities share the same `global_perm`, generated once before the scan. This means the "next available cell" search is reproducible given the same `resolve_key`, but is not independent per-entity.
- **Fallback on area exhaustion**: if no valid unoccupied cell exists within an entity's spawn area, `replacement_flat = 0` (grid origin `(0, 0)`). Multiple entities can silently stack on cell `(0, 0)`. This is a known limitation — avoid by ensuring every spawn area has strictly more cells than assigned entities.
- **Spawn-area bounds**: `in_area` checks `min_r ≤ row < max_r` and `min_c ≤ col < max_c`. The upper bounds are exclusive (consistent with the `_parse_area` conversion in `config_loader.py:243`).

---

## `place_in_area`

**Plain-English purpose**: Used by the `per_type` branch. Given a rectangular spawn area and the current occupancy mask, this function picks `num_entities` non-overlapping free cells by shuffling all grid cells randomly, filtering to those inside the area and not yet occupied, and taking the first `num_entities` of that filtered sequence. Output is padded to `max_entities` (a static compile-time constant) so that XLA output shapes are fixed.

**Signature**: `place_in_area(subkey, area, occupancy, num_entities, max_entities, grid_height, grid_width) → (positions [max_entities, 2], flat_indices [max_entities])` (`core.py:711`)

Source: `src/environment/core.py:711–758`

```python
def place_in_area(
    subkey: jax.random.PRNGKey,
    area: jnp.ndarray,
    occupancy: jnp.ndarray,
    num_entities: int,
    max_entities: int,
    grid_height: int,
    grid_width: int,
) -> tuple:
    """Place entities in a rectangular area, avoiding occupied cells.
    
    Uses a single global permutation filtered to valid candidates.
    Returns (positions [max_entities, 2], flat_indices [max_entities]).
    
    Positions beyond num_entities are padded with zeros.
    """
    total_cells = grid_height * grid_width
    min_r, min_c, max_r, max_c = area[0], area[1], area[2], area[3]
    
    cell_rows = jnp.arange(total_cells) // grid_width
    cell_cols = jnp.arange(total_cells) % grid_width
    
    # Valid = in area AND not occupied
    in_area = (cell_rows >= min_r) & (cell_rows < max_r) & \
              (cell_cols >= min_c) & (cell_cols < max_c)
    valid = in_area & (~occupancy)
    
    # Permute all cells, filter to valid
    perm = jax.random.permutation(subkey, total_cells)
    valid_in_perm = valid[perm]
    
    # Take first num_entities valid cells
    cumsum = jnp.cumsum(valid_in_perm)
    selected = valid_in_perm & (cumsum <= num_entities)
    
    # Extract flat indices, padded to max_entities
    selected_flat = jnp.where(selected, perm, total_cells)  # sentinel
    selected_flat = jnp.sort(selected_flat)[:max_entities]
    selected_flat = jnp.where(
        jnp.arange(max_entities) < num_entities,
        selected_flat,
        0  # unused padding
    )
    
    pos_r = selected_flat // grid_width
    pos_c = selected_flat % grid_width
    positions = jnp.stack([pos_r, pos_c], axis=-1)
    return positions, selected_flat
```

> **API notes**
>
> - `jax.random.permutation(subkey, total_cells)`: [primer: prng](00_jax_primer.md#prng). Generates a random permutation of integers `[0, total_cells)`. This is called with a fresh subkey each time so each type group gets its own independent shuffle.
> - `jnp.cumsum(valid_in_perm) <= num_entities`: [primer: masking](00_jax_primer.md#masking). The `cumsum` converts a boolean sequence like `[F,T,T,F,T,…]` into a running count `[0,1,2,2,3,…]`. `<= num_entities` then selects exactly the first `num_entities` valid cells — a zero-branch alternative to slicing a variable-length filtered list.
> - `jnp.where(selected, perm, total_cells)` + `jnp.sort(…)[:max_entities]`: this is the scatter-then-sort idiom for compacting a sparse boolean selection into a dense fixed-length array. Unselected slots get sentinel `total_cells` (larger than any valid flat index), so after `jnp.sort` they float to the back; slicing `[:max_entities]` discards them. [primer: scatter-index](00_jax_primer.md#scatter-index) covers the `.at[].set()` variant of the same gather/scatter pattern.
> - `jnp.where(jnp.arange(max_entities) < num_entities, selected_flat, 0)`: second masking pass to zero-pad the slots beyond `num_entities`. `max_entities` is a **static Python int** (the XLA loop bound), so `jnp.arange(max_entities)` has a fixed shape and the comparison is JIT-friendly. [primer: static-dynamic](00_jax_primer.md#static-dynamic).

### Algorithm summary

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

## `jax_reset`

**Plain-English purpose**: The full episode initializer. Given compiled config (`params`) and a PRNG key, it (1) places the agent, (2) places all entities on the grid without collisions, (3) samples per-episode body state (nutrition, injury) and animal behavioral parameters (detect range, stamina, etc.), (4) samples per-episode olfactory properties for every entity, and (5) assembles the initial `EnvState`. It is a pure function decorated with `@jax.jit` — no Python side-effects, no in-place mutation.

**Signature**: `jax_reset(params: EnvParams, key: PRNGKey) → EnvState` (`core.py:762`)

---

### Chunk 1 — decorator and PRNG key splitting

Source: `src/environment/core.py:761–787`

```python
@jax.jit
def jax_reset(params: EnvParams, key: jax.random.PRNGKey) -> EnvState:
    """Functional reset for the JAX environment (v2.0 — unified animal entity).

    Placement: vmap sampling + resolve_overlaps_global (per_entity mode) or
    lax.scan over type groups (per_type mode).

    N1 fix: entity concatenation order for the overlap-resolve scan is preserved
      as [res, pred, obs, neutral] (pred = predator-class animals; neutral =
      neutral-class animals). Positions are sliced back in the same order.
    N2 fix: property-key split is 4-way (prop_key_res / prop_key_pred /
      prop_key_obs / prop_key_neutral). The predator-class / neutral-class subsets
      of animal_property are sampled with prop_key_pred / prop_key_neutral
      respectively, then concatenated in predator-first order.
    N3 fix: placement_key split remains 6-way (res, pred, obs, neutral, resolve).
    """
    # Keep the original 5-way outer split byte-for-byte (N3 fix).
    # animal_episode_key is derived from property_key via fold_in so the
    # existing agent_key / placement_key / body_key / property_key streams
    # remain byte-identical to the pre-refactor code.
    key, agent_key, placement_key, body_key, property_key = jax.random.split(key, 5)
    # Derive per-episode animal sampling key without disturbing existing streams.
    animal_episode_key = jax.random.fold_in(property_key, 0xAE1)

    # 1. Agent Position
    random_pos = jax.random.randint(agent_key, (2,), 0, jnp.array([params.height, params.width]))
    agent_pos = jnp.where(params.random_start_pos, random_pos, params.start_pos)
```

> **API notes**
>
> - `jax.random.split(key, 5)`: [primer: prng](00_jax_primer.md#prng). JAX's PRNG is **functional and explicit** — you never have a global random state. `split` consumes one key and produces `n` independent children. Every function that needs randomness receives its own subkey; no two functions share a key stream. This is why `agent_key`, `placement_key`, `body_key`, `property_key` are all guaranteed independent.
> - `jax.random.fold_in(property_key, 0xAE1)`: [primer: prng](00_jax_primer.md#prng). `fold_in` mixes a key with an integer to produce a derived key **without consuming the original key** — `property_key` remains usable for its own split later. This was added so `animal_episode_key` could be introduced without changing the byte sequence of any existing key stream (backward-compatibility requirement documented in the `N3 fix` comment).
> - `@jax.jit`: [primer: jit](00_jax_primer.md#jit). The decorator traces `jax_reset` once on first call and compiles it to XLA. Subsequent calls skip Python and go straight to device. `params` contains **static fields** (`pytree_node=False`) such as `predator_indices` — these are treated as compile-time constants; changing them forces recompilation. [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - `jnp.where(params.random_start_pos, random_pos, params.start_pos)`: [primer: branchless](00_jax_primer.md#branchless). `params.random_start_pos` is a Python bool (static), so a Python `if` would also work here — but `jnp.where` makes the intent explicit and is safe if the field is ever made dynamic.

---

### Chunk 2 — entity counts and `per_entity` placement

Source: `src/environment/core.py:789–848`

```python
    # 2. Entity Placement
    num_res = params.res_type.shape[0]
    # N1 fix: pred = animals with predator class; neutral = animals with neutral class.
    # Use static index tuples (pytree_node=False) to slice spawn areas per class.
    num_pred_class   = len(params.predator_indices)
    num_neutral_class = len(params.neutral_indices)
    num_obs = params.obs_blocking.shape[0]

    if params.placement_mode == 'per_entity':
        # ── Per-Entity Scan: vmap sample + resolve_overlaps_global ──
        # N3 fix: 6-way split (unchanged from pre-refactor split shape).
        placement_key, res_key, pred_key, obs_key, neutral_key, resolve_key = \
            jax.random.split(placement_key, 6)

        # Sample initial positions (may have overlaps)
        res_keys = jax.random.split(res_key, num_res) if num_res > 0 else jax.random.split(res_key, 1)[:0]
        res_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            res_keys, params.res_spawn_area) if num_res > 0 else jnp.zeros((0, 2), dtype=jnp.int32)

        # N1 fix: per-class spawn areas from config_loader (pred_spawn_area_for_placement
        # is not stored on params; use params.animal_spawn_area + predator_indices).
        if num_pred_class > 0:
            pred_sa = params.animal_spawn_area[jnp.array(list(params.predator_indices), dtype=jnp.int32)]
            pred_keys = jax.random.split(pred_key, num_pred_class)
            pred_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
                pred_keys, pred_sa)
        else:
            pred_pos = jnp.zeros((0, 2), dtype=jnp.int32)

        obs_keys = jax.random.split(obs_key, num_obs) if num_obs > 0 else jax.random.split(obs_key, 1)[:0]
        obs_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            obs_keys, params.obs_spawn_area) if num_obs > 0 else jnp.zeros((0, 2), dtype=jnp.int32)

        if num_neutral_class > 0:
            neutral_sa = params.animal_spawn_area[jnp.array(list(params.neutral_indices), dtype=jnp.int32)]
            neutral_keys = jax.random.split(neutral_key, num_neutral_class)
            neutral_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
                neutral_keys, neutral_sa)
        else:
            neutral_pos = jnp.zeros((0, 2), dtype=jnp.int32)

        # N1 fix: concat order [res, pred, obs, neutral] for resolve scan.
        all_positions = jnp.concatenate([res_pos, pred_pos, obs_pos, neutral_pos], axis=0)
        # Build matching spawn-area array (N1 fix: same ordering).
        if num_pred_class > 0:
            pred_sa_all = params.animal_spawn_area[jnp.array(list(params.predator_indices), dtype=jnp.int32)]
        else:
            pred_sa_all = jnp.zeros((0, 4), dtype=jnp.int32)
        if num_neutral_class > 0:
            neutral_sa_all = params.animal_spawn_area[jnp.array(list(params.neutral_indices), dtype=jnp.int32)]
        else:
            neutral_sa_all = jnp.zeros((0, 4), dtype=jnp.int32)
        all_spawn_areas = jnp.concatenate([
            params.res_spawn_area, pred_sa_all,
            params.obs_spawn_area, neutral_sa_all,
        ], axis=0)
        if all_positions.shape[0] > 0:
            all_positions = resolve_overlaps_global(
                all_positions, all_spawn_areas, params.height, params.width, resolve_key
            )
```

> **API notes**
>
> - `jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(res_keys, params.res_spawn_area)`: [primer: vmap](00_jax_primer.md#vmap). `vmap` vectorizes the lambda over the leading axis of both `res_keys` (shape `[num_res, 2]` — each row is a key) and `params.res_spawn_area` (shape `[num_res, 4]`). This replaces a Python `for` loop over entities with a single parallel operation. Each entity gets its own key (`res_keys[i]`) and its own spawn area (`res_spawn_area[i]`); the lambda body is written for a single entity.
> - `params.animal_spawn_area[jnp.array(list(params.predator_indices), ...)]`: [primer: scatter-index](00_jax_primer.md#scatter-index). `predator_indices` is a static Python tuple (not a JAX array); converting it to `jnp.array` at trace time lets it serve as a fancy-index into `animal_spawn_area`. The result is a `[num_pred_class, 4]` slice of spawn areas for the predator-class animals only.
> - `jax.random.split(placement_key, 6)`: six independent subkeys — one per entity group (`res`, `pred`, `obs`, `neutral`), one spare (unused `placement_key`), and one `resolve_key` for `resolve_overlaps_global`. The spare is consumed by the split call so the key tree remains full-rank.
> - The Python `if params.placement_mode == 'per_entity':` branch runs at **trace time** (it is a Python conditional, not a `jnp.cond`). `placement_mode` is a static string field. XLA only ever sees one branch's code; the other branch is dead-compiled. [primer: static-dynamic](00_jax_primer.md#static-dynamic).

---

### Chunk 3 — `per_type` placement

Source: `src/environment/core.py:850–883`

```python
    else:  # per_type
        # ── Type-Level: lax.scan over spawn-area groups ──
        all_positions = jnp.zeros((params.num_entities, 2), dtype=jnp.int32)
        total_cells = params.height * params.width
        occupancy = jnp.zeros(total_cells, dtype=jnp.bool_)

        def place_type_group(carry, type_idx):
            occ, positions, rng = carry
            rng, subkey = jax.random.split(rng)
            area = params.type_areas[type_idx]
            count = params.type_counts[type_idx]
            type_pos, type_flat = place_in_area(
                subkey, area, occ, count, params.max_per_type,
                params.height, params.width
            )
            valid = jnp.arange(params.max_per_type) < count
            for j in range(params.max_per_type):
                occ = jnp.where(valid[j], occ.at[type_flat[j]].set(True), occ)
            entity_indices = params.type_entity_map[type_idx]
            for j in range(params.max_per_type):
                eidx = entity_indices[j]
                positions = jnp.where(
                    valid[j],
                    positions.at[eidx].set(type_pos[j]),
                    positions
                )
            return (occ, positions, rng), None

        placement_key, scan_key = jax.random.split(placement_key)
        (_, all_positions, _), _ = jax.lax.scan(
            place_type_group,
            (occupancy, all_positions, scan_key),
            jnp.arange(params.num_types)
        )
```

> **API notes**
>
> - `jax.lax.scan(place_type_group, init_carry, jnp.arange(params.num_types))`: [primer: lax.scan](00_jax_primer.md#scan). The scan iterates `params.num_types` times (one per spawn-area group). The carry `(occ, positions, rng)` threads the evolving occupancy mask, the accumulating position array, and the PRNG key through each iteration — so each group's placement sees the cells already claimed by previous groups. `jnp.arange(params.num_types)` is the per-step input (`type_idx` in the body), providing the current group's index.
> - `rng, subkey = jax.random.split(rng)` **inside the scan body**: [primer: prng](00_jax_primer.md#prng). This is the standard pattern for generating a fresh independent key at each scan step while threading the remainder (`rng`) through the carry. Each call to `place_in_area` gets a unique subkey.
> - `for j in range(params.max_per_type)`: this is a **Python loop at trace time**, not a runtime loop. `params.max_per_type` is a static int; JAX unrolls it into `max_per_type` separate XLA operations. This is intentional: the loop body uses `valid[j]` (a scalar bool slice) with `jnp.where`, which requires a known unroll count. [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - `occ.at[type_flat[j]].set(True)` inside `jnp.where(valid[j], ..., occ)`: [primer: immutability](00_jax_primer.md#immutability). The `.at[].set()` call produces a candidate updated array; `jnp.where` then selects between the updated array and the unchanged `occ` based on whether slot `j` is a real entity (`valid[j]`). This avoids writing stale padding positions into the occupancy mask.
> - `positions.at[eidx].set(type_pos[j])` with `eidx = entity_indices[j]`: [primer: scatter-index](00_jax_primer.md#scatter-index). `entity_indices` holds the global entity indices for this type group (from `params.type_entity_map[type_idx]`). Writing via `.at[eidx].set(...)` scatters the group-local position `type_pos[j]` into the correct row of the global `positions` array.

---

### Chunk 4 — split positions back and assemble unified `animal_pos`

Source: `src/environment/core.py:885–904`

```python
    # 3. Split positions back (N1 fix: same [res, pred, obs, neutral] order).
    res_pos   = all_positions[:num_res]
    pred_pos  = all_positions[num_res:num_res + num_pred_class]
    obs_pos   = all_positions[num_res + num_pred_class:num_res + num_pred_class + num_obs]
    neutral_pos = all_positions[num_res + num_pred_class + num_obs:]

    # 4. Assemble unified animal_pos [N, 2] in predator-first order (N1 fix).
    N = params.animal_property.shape[0]
    if N > 0:
        # Scatter pred/neutral positions back into the unified [N, 2] array
        # using the static index tuples.
        animal_pos_init = jnp.zeros((N, 2), dtype=jnp.int32)
        if num_pred_class > 0:
            p_idx = jnp.array(list(params.predator_indices), dtype=jnp.int32)
            animal_pos_init = animal_pos_init.at[p_idx].set(pred_pos)
        if num_neutral_class > 0:
            n_idx = jnp.array(list(params.neutral_indices), dtype=jnp.int32)
            animal_pos_init = animal_pos_init.at[n_idx].set(neutral_pos)
    else:
        animal_pos_init = jnp.zeros((0, 2), dtype=jnp.int32)
```

> **API notes**
>
> - `animal_pos_init.at[p_idx].set(pred_pos)`: [primer: scatter-index](00_jax_primer.md#scatter-index). This is the core scatter operation. `p_idx` is a 1-D integer array of length `num_pred_class` holding the rows in `animal_pos_init` that belong to predator-class animals. `pred_pos` is `[num_pred_class, 2]`. The `.at[p_idx].set(pred_pos)` call writes each row of `pred_pos` into the corresponding row of `animal_pos_init`. The result is a new array with the predator positions filled in; all other rows remain zero until the neutral scatter follows.
> - Two successive `.at[].set()` calls, each returning a new array: [primer: immutability](00_jax_primer.md#immutability). `animal_pos_init` is reassigned twice. The two writes are to disjoint row sets (`predator_indices` and `neutral_indices` are non-overlapping by construction), so the final array has both classes correctly placed.
> - The Python `if N > 0:` / `if num_pred_class > 0:` guards run at trace time. They prevent tracing `.at[p_idx].set(pred_pos)` when `p_idx` would be an empty array — which JAX handles correctly but produces a dead scatter node that adds unnecessary XLA overhead.
> - Slices `all_positions[:num_res]`, `all_positions[num_res:num_res+num_pred_class]`, etc.: static-shape slices because `num_res`, `num_pred_class`, `num_obs` are Python ints derived from static params fields. JAX can compute these bounds at trace time. [primer: static-dynamic](00_jax_primer.md#static-dynamic).

---

### Chunk 5 — body initialisation and property sampling

Source: `src/environment/core.py:906–955`

```python
    # 5. Body (Random start support)
    body_key1, body_key2, body_key3 = jax.random.split(body_key, 3)

    if params.random_start_nutrition:
        min_start_nutr = params.max_nutrition / 2.0
        nutrition = jax.random.uniform(body_key2, (), minval=min_start_nutr, maxval=params.max_nutrition)
    else:
        nutrition = params.start_nutrition

    fullness_ratio = jnp.clip(nutrition / params.max_nutrition, 0.0, 1.0)
    satiation = params.max_satiation * jnp.power(fullness_ratio, params.nutrition_to_satiation_scaling_factor)

    if params.random_start_injury:
        max_start_injury = params.max_injury / 2.0
        injury = jax.random.uniform(body_key3, (), minval=0.0, maxval=max_start_injury)
    else:
        injury = 0.0

    injury_buffer = jnp.zeros(params.smoothing_duration)
    nociception_history_buffer = jnp.zeros(params.interoceptive_kernel_length)

    # 6. Property sampling (N2 fix: 4-way split, pred/neutral sampled separately).
    prop_key_res, prop_key_pred, prop_key_obs, prop_key_neutral = jax.random.split(property_key, 4)

    def _sample_property(sub_key, mean, std):
        noise = jax.random.normal(sub_key, shape=mean.shape)
        return jnp.clip(mean + std * noise, 0.0, 1.0)

    res_property_sampled = _sample_property(prop_key_res, params.res_property, params.res_property_std)
    obs_property_sampled = _sample_property(prop_key_obs, params.obs_property, params.obs_property_std)

    # N2 fix: sample pred-class and neutral-class properties separately then
    # scatter back into the unified [N, vector_size] array (preserving byte-parity
    # with the pre-refactor prop_key_pred / prop_key_neutral draws).
    if N > 0:
        animal_property_sampled = jnp.zeros_like(params.animal_property)
        if num_pred_class > 0:
            p_idx = jnp.array(list(params.predator_indices), dtype=jnp.int32)
            pred_prop_mean = params.animal_property[p_idx]
            pred_prop_std  = params.animal_property_std[p_idx]
            pred_prop_sampled = _sample_property(prop_key_pred, pred_prop_mean, pred_prop_std)
            animal_property_sampled = animal_property_sampled.at[p_idx].set(pred_prop_sampled)
        if num_neutral_class > 0:
            n_idx = jnp.array(list(params.neutral_indices), dtype=jnp.int32)
            neutral_prop_mean = params.animal_property[n_idx]
            neutral_prop_std  = params.animal_property_std[n_idx]
            neutral_prop_sampled = _sample_property(prop_key_neutral, neutral_prop_mean, neutral_prop_std)
            animal_property_sampled = animal_property_sampled.at[n_idx].set(neutral_prop_sampled)
    else:
        animal_property_sampled = jnp.zeros((0, params.animal_property.shape[-1] if params.animal_property.shape[0] == 0 else params.animal_property.shape[-1]), dtype=jnp.float32)
```

> **API notes**
>
> - `jax.random.split(property_key, 4)`: [primer: prng](00_jax_primer.md#prng). The 4-way split keeps `prop_key_pred` and `prop_key_neutral` byte-identical to the pre-refactor code's two separate animal property keys — the N2 fix comment explains this backward-compatibility requirement.
> - `_sample_property`: `jax.random.normal(sub_key, shape=mean.shape)` + `jnp.clip(mean + std * noise, 0.0, 1.0)` is the standard Gaussian noise pattern for property perturbation. Shape is driven by `mean.shape` so the function is polymorphic over any `[N, V]` slice.
> - The `animal_property_sampled.at[p_idx].set(pred_prop_sampled)` / `animal_property_sampled.at[n_idx].set(neutral_prop_sampled)` pattern: [primer: scatter-index](00_jax_primer.md#scatter-index). Exact same scatter idiom as Chunk 4's `animal_pos_init` assembly — allocate a zero buffer, scatter predator rows, scatter neutral rows. The two class subsets use their own independent subkeys, preserving the byte-parity of the pre-refactor code.
> - `params.random_start_nutrition` and `params.random_start_injury` are static Python bools, so the `if`/`else` branches compile to separate XLA graphs. Switching these flags triggers recompilation. [primer: jit](00_jax_primer.md#jit), [primer: static-dynamic](00_jax_primer.md#static-dynamic).

---

### Chunk 6 — per-episode behavioural parameter sampling and EnvState construction

Source: `src/environment/core.py:957–1014`

```python
    # 7. Per-episode distributional sampling for the 5 behavioural fields.
    #    5 independent uniform draws per field — shape (N,) each.
    #    For wander/static entries the ranges are [0, 0] (from _load_animals);
    #    jax.random.uniform([0,0]) = 0.0 exactly, so these are harmless.
    if N > 0:
        ep_keys = jax.random.split(animal_episode_key, 5)
        animal_detect_sampled = jax.random.uniform(
            ep_keys[0], (N,), minval=params.animal_detect_low, maxval=jnp.maximum(params.animal_detect_high, params.animal_detect_low))
        animal_max_stamina_sampled = jax.random.uniform(
            ep_keys[1], (N,), minval=params.animal_max_stamina_low, maxval=jnp.maximum(params.animal_max_stamina_high, params.animal_max_stamina_low))
        animal_recovery_sampled = jax.random.uniform(
            ep_keys[2], (N,), minval=params.animal_recovery_low, maxval=jnp.maximum(params.animal_recovery_high, params.animal_recovery_low))
        animal_hunt_thresh_sampled = jax.random.uniform(
            ep_keys[3], (N,), minval=params.animal_hunt_thresh_low, maxval=jnp.maximum(params.animal_hunt_thresh_high, params.animal_hunt_thresh_low))
        animal_lose_interest_sampled = jax.random.uniform(
            ep_keys[4], (N,), minval=params.animal_lose_interest_low, maxval=jnp.maximum(params.animal_lose_interest_high, params.animal_lose_interest_low))
    else:
        animal_detect_sampled        = jnp.zeros(0, dtype=jnp.float32)
        animal_max_stamina_sampled   = jnp.zeros(0, dtype=jnp.float32)
        animal_recovery_sampled      = jnp.zeros(0, dtype=jnp.float32)
        animal_hunt_thresh_sampled   = jnp.zeros(0, dtype=jnp.float32)
        animal_lose_interest_sampled = jnp.zeros(0, dtype=jnp.float32)

    state = EnvState(
        agent_pos=agent_pos,
        current_step=jnp.array(0, dtype=jnp.int32),
        res_pos=res_pos,
        res_active=jnp.ones(num_res, dtype=jnp.bool_),
        res_cons_count=jnp.zeros(num_res, dtype=jnp.int32),
        res_reg_timer=jnp.zeros(num_res, dtype=jnp.int32),
        res_property_sampled=res_property_sampled,
        # Unified animal fields
        animal_pos=animal_pos_init,
        animal_state=jnp.zeros(N, dtype=jnp.int32),           # PATROL=0
        animal_stamina=animal_max_stamina_sampled.copy() if N > 0 else jnp.zeros(0, dtype=jnp.float32),
        animal_move_timer=jnp.zeros(N, dtype=jnp.int32),
        animal_attack_timer=jnp.zeros(N, dtype=jnp.int32),
        animal_property_sampled=animal_property_sampled,
        animal_detect_sampled=animal_detect_sampled,
        animal_max_stamina_sampled=animal_max_stamina_sampled,
        animal_recovery_sampled=animal_recovery_sampled,
        animal_hunt_thresh_sampled=animal_hunt_thresh_sampled,
        animal_lose_interest_sampled=animal_lose_interest_sampled,
        obs_pos=obs_pos,
        obs_property_sampled=obs_property_sampled,
        satiation=jnp.array(satiation, dtype=jnp.float32),
        nutrition=jnp.array(nutrition, dtype=jnp.float32),
        injury_level=jnp.array(injury, dtype=jnp.float32),
        injury_buffer=injury_buffer,
        nociception_history_buffer=nociception_history_buffer,
        last_collision_noc=jnp.array(0.0, dtype=jnp.float32),
        rest_streak=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False, dtype=jnp.bool_),
        key=key,
        last_action=jnp.array(4 if params.rest_action_enabled else 5, dtype=jnp.int32),
    )

    return state
```

> **API notes**
>
> - `jax.random.split(animal_episode_key, 5)`: [primer: prng](00_jax_primer.md#prng). Five independent subkeys, one per behavioural field. Because `animal_episode_key` was derived via `fold_in` (not from the main 5-way split), these five draws are statistically independent from all placement and property draws.
> - `jax.random.uniform(ep_keys[0], (N,), minval=..., maxval=...)` with array-valued `minval` / `maxval`: each element `i` of the output is drawn from `Uniform(animal_detect_low[i], animal_detect_high[i])`. JAX broadcasts the scalar output shape `(N,)` against the per-element bounds. This is effectively a `vmap` over entities written as a vectorized uniform call.
> - `jnp.maximum(params.animal_detect_high, params.animal_detect_low)`: guards against degenerate ranges where `high == low` (wander/static entries). `jax.random.uniform` requires `maxval > minval`; for `[0, 0]` ranges, `jnp.maximum` keeps `maxval = 0 = minval` — and JAX's uniform returns `minval` when the interval has zero width (output is exactly `0.0`).
> - `EnvState(...)` construction: [primer: jax-pytrees](00_jax_primer.md#jax-pytrees). The `@struct.dataclass`-registered constructor takes keyword arguments for every field. All arrays must have the shapes declared in `state.py`. Any shape mismatch raises at trace time (first JIT call), not at Python import time.
> - `animal_stamina=animal_max_stamina_sampled.copy()`: `.copy()` on a JAX array produces an identical array with no aliasing. This is defensive; since JAX arrays are immutable, aliasing is safe in principle, but the explicit copy matches the intent that stamina starts at its max and evolves independently.

---

## `jax_reset` — call sequence summary

1. **PRNG key splitting** (`core.py:781–783`): 5-way outer split + `fold_in` for `animal_episode_key`.
2. **Agent position** (`core.py:786–787`): random or fixed via `jnp.where`.
3. **Entity counts** (`core.py:790–795`): static ints from params shapes and index-tuple lengths.
4. **Entity placement** (`core.py:797–883`): dispatch to `per_entity` (vmap + `resolve_overlaps_global` scan) or `per_type` (`lax.scan` over groups calling `place_in_area`).
5. **Split positions back** (`core.py:885–889`): static-index slices of `all_positions` in `[res | pred_class | obs | neutral_class]` order.
6. **Unified `animal_pos` assembly** (`core.py:891–904`): zero-init `[N, 2]`, scatter predator rows via `predator_indices`, scatter neutral rows via `neutral_indices`.
7. **Body initialisation** (`core.py:906–925`): random or fixed nutrition/injury; satiation derived via power-law; buffers zero-initialised.
8. **Property sampling** (`core.py:927–953`): 4-way key split; Gaussian noise per entity group; same scatter idiom as step 6 for the unified `animal_property_sampled` array.
9. **Behavioural parameter sampling** (`core.py:957–978`): 5-way split of `animal_episode_key`; vectorised uniform draws with per-element bounds.
10. **`EnvState` construction** (`core.py:980–1014`): all fields assembled; animals start in state `0` (Patrol/idle) with full stamina.

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

4. Call `resolve_overlaps_global(all_positions, all_spawn_areas, H, W, resolve_key)` to eliminate same-cell collisions (see above).

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
