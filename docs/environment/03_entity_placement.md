# 03 — Entity Placement

> **Source**: `src/environment/core.py` (`jax_reset`, `resolve_overlaps_global`, `place_in_area`), `src/environment/config_loader.py` | **Back to hub**: [README](README.md)

---

## Overview

At the start of every episode, all entities — resources, predators, obstacles, and neutral animals — must be placed on the grid without overlapping each other. This is done inside `jax_reset` (`core.py:614`), which is decorated with `@jax.jit` and is entirely pure-functional.

Two placement modes are available, selected by `params.placement_mode`:

- **`per_entity`**: individually sample each entity's position, then resolve collisions with a sequential scan.
- **`per_type`**: group entities by spawn area and place each group together using a constrained area sampler.

After placement, `jax_reset` initialises body state (optionally randomised) and returns an `EnvState`. It does **not** return an initial observation — the caller is responsible for calling `get_observation(state, params)`.

---

## `jax_reset`

**Signature**: `jax_reset(params: EnvParams, key: PRNGKey) → EnvState` (`core.py:614`)

Call sequence:

1. **Split PRNG key** into `agent_key`, `placement_key`, `body_key`.
2. **Agent position**: random within grid if `params.random_start_pos`, else `params.start_pos`.
3. **Entity placement**: dispatch to `per_entity` or `per_type` branch.
4. **Split positions** back into per-entity-type arrays by index ranges: `res`, `pred`, `obs`, `neutral`.
5. **Body initialisation**:
   - Nutrition: random uniform in `[max_nutrition/2, max_nutrition]` if `random_start_nutrition`, else `start_nutrition`.
   - Satiation: **always derived from nutrition** via the power-law `S = max_S * (N/max_N)^k` — never set independently.
   - Injury: random uniform in `[0, max_injury/2]` if `random_start_injury`, else `0.0`.
   - `injury_buffer`: always zero-initialised.
6. **Construct `EnvState`** with all predators in state 0 (Patrol), full stamina, zero timers, all resources active.

The `@jax.jit` decorator means the first call compiles the function for the given static fields; subsequent calls with the same `params` shape reuse the compiled version.

---

## `per_entity` Mode

**Code**: `core.py:634–664`

Steps:
1. For each entity type (res, pred, obs, neutral), split the placement key and vmap `jax.random.randint` over `spawn_area` bounds to sample an initial `(row, col)` per entity.
2. Concatenate all positions into `all_positions [N, 2]` and all spawn areas into `all_spawn_areas [N, 4]`.
3. Call `resolve_overlaps_global(all_positions, all_spawn_areas, ...)` to eliminate any same-cell collisions.

**Best for**: small grids (≤100 cells) or low entity counts. The initial vmap step is O(1) XLA instructions regardless of N.

---

## `per_type` Mode

**Code**: `core.py:666–701`

Entities that share the same spawn area bounding box are grouped into a **type group** at config-load time (see `config_loader.py:183–228`). At reset time:

1. Initialise a flat `occupancy [H*W]` boolean mask (all False).
2. Run `jax.lax.scan` over `jnp.arange(params.num_types)`, calling `place_in_area()` for each group.
3. `place_in_area` places all entities in the group at once, updating the occupancy mask so subsequent groups avoid already-taken cells.
4. `type_entity_map [T, max_per_type]` scatters group positions back to global entity indices.

**Best for**: large grids or dense configurations where many entities share an area. `lax.scan` compiles to a fixed-iteration loop — `num_types` steps regardless of N — which is more XLA-efficient than N per-entity scan steps when N is large.

---

## `resolve_overlaps_global`

**Signature**: `resolve_overlaps_global(all_positions, all_spawn_areas, grid_height, grid_width, key) → all_positions` (`core.py:514`)

Algorithm:
1. Allocate a flat `occupancy [H*W]` boolean mask.
2. Generate one random permutation of all `H*W` cell indices (`global_perm`).
3. Run `jax.lax.scan` over entities in order. For entity `i`:
   - Check if its sampled cell is already occupied.
   - If occupied: find the first valid unoccupied cell within its spawn area by scanning `global_perm` and keeping the first index where the cell is in-area and unoccupied.
   - Set the chosen cell in `occupancy`.
   - Update `all_positions[i]` to the chosen cell.
4. Return the resolved positions.

**Collision criterion**: same `(row, col)` cell. Two entities at distance 0.

**Termination guarantee**: if the spawn area has at least one free cell, the scan will always find a placement. If the area is completely full (more entities than area cells), the entity is placed at cell 0 — a known edge case that should be avoided by proper config design.

---

## `place_in_area`

**Signature**: `place_in_area(subkey, area, occupancy, num_entities, max_entities, grid_height, grid_width) → (positions [max_entities, 2], flat_indices [max_entities])` (`core.py:563`)

Algorithm:
1. Compute `in_area`: mask of all cells within `(min_r, min_c, max_r, max_c)`.
2. Compute `valid = in_area & ~occupancy`.
3. Generate a random permutation of all cells (`perm`).
4. Filter `perm` to valid cells only; take the first `num_entities`.
5. Extract row/col from flat indices and stack.
6. Pad positions to `max_entities` with `(0, 0)` for unused slots.

The caller is responsible for not scattering padded (unused) positions back to global entity indices — `per_type` mode uses `valid = jnp.arange(max_per_type) < count` to guard each write.

---

## Random Start Conditions

Controlled by static boolean flags in `EnvParams`:

| Flag | Effect when True | Range |
|------|-----------------|-------|
| `random_start_pos` | Agent spawns at uniformly random grid cell | `[0, H) × [0, W)` |
| `random_start_nutrition` | Initial nutrition sampled uniformly | `[max_nutrition/2, max_nutrition]` |
| `random_start_injury` | Initial injury sampled uniformly | `[0, max_injury/2]` |
| `random_start_satiation` | Not independently randomised; satiation is always derived from nutrition | — |

When disabled, the deterministic starting values are `start_nutrition`, `start_satiation` (body config), and `start_pos` (converted to 0-based from YAML). Initial injury is always 0.0 when not randomised.
