# 03 — Entity Placement

> **Source**: `src/environment/core.py` (`jax_reset`, `resolve_overlaps_global`, `place_in_area`), `src/environment/config_loader.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

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

> **Note**: `start_satiation` is **loaded but never used** — satiation at reset is always derived from nutrition via the power-law scaling factor. See doc `01` FAQ.

---

## Clarifications / FAQ

**Q: Can the agent spawn on top of an entity?**
A: **Yes.** Agent placement is computed at `core.py:633-634` *before* entity placement and the agent's cell is **never added to the occupancy mask**. An agent spawned with `random_start_pos=True` can land on a predator, obstacle, food, etc. The step loop's contact logic then fires on the very first step — so a random-spawn run can start with immediate damage or food gain.

**Q: What's the entity processing order in `resolve_overlaps_global`?**
A: Resources first, then predators, then obstacles, then neutral animals (the concatenation order at `per_entity` mode). Earlier entities get first pick — if a resource and a predator sample the same cell, the resource keeps it and the predator is relocated. This order is implicit in `all_positions = jnp.concatenate([res_pos, pred_pos, obs_pos, neutral_pos])`.

**Q: What happens when a spawn area has fewer free cells than entities that need to go in it?**
A: The scan's "first valid unoccupied cell" search falls back to cell `0` (global flat index 0 = grid origin `(0, 0)`) because `replacement_flat = jnp.where(first_valid_mask, global_perm, 0).sum()` returns `0` when no valid cell exists (`core.py:553-554`). This is a silent failure — **multiple entities stack on cell (0, 0)**. Guard against this in config: ensure each spawn area has strictly more cells than entities assigned to it.

**Q: Why does `per_entity` mode get called "fast on small grids" but `per_type` wins on large grids?**
A: `per_entity`'s `lax.scan` runs `N` steps (one per entity), with each step doing an `H*W`-length cumsum over the permutation. `per_type` runs `num_types` steps (one per spawn-area group). When `N` is large and entities cluster into few groups (e.g. 50 food items in one area), `per_type` is asymptotically cheaper. For small `N` (say, 5 entities) the overhead of grouping isn't worth it.

**Q: Do the two modes produce the same placement given the same seed?**
A: No — they use different sampling algorithms and consume the PRNG key differently. Switching modes changes reset-time layouts. If you want reproducibility across a mode change, also change the seed.

**Q: When `num_res == 0` (no resources), does the code still work?**
A: Yes. Empty arrays (`shape [0, ...]`) are handled by vmap/concatenation correctly. `num_res = 0` produces zero-length keys and zero-length positions, and the concat simply omits those rows. The same holds for the other entity kinds.

**Q: Is `jax_reset` cheap enough to call every episode?**
A: After the first JIT compile, yes — the whole thing is pure JAX and executes in one kernel launch on the host-to-device dispatch. In `ParallelEnv`, `jax_reset` is vmapped so a batch of envs resets in parallel without a Python-level loop.

**Q: What's `jax_reset`'s return shape difference from `jax_step`?**
A: `jax_reset` returns `EnvState` only. `jax_step` returns `(new_state, reward, done, info)`. If you want an observation at step 0, call `get_observation(state, params)` explicitly after reset (see doc `09_sensors_and_observation.md`).

**Q: Is `agent_key` independent of entity placement keys?**
A: Yes. `jax.random.split(key, 5)` produces five independent subkeys — agent, placement, body, property, and a remainder carried forward. Entity placement splits `placement_key` further. The agent's position and entity positions are therefore statistically independent given the master seed.

**Q: What's stored in `occupancy` — Boolean per cell or entity index?**
A: Boolean per cell (`[H*W]`). The mask tracks "any entity already here" — it does not record *which* entity. This is sufficient for collision avoidance but means you cannot reconstruct placement order from the mask.

**Q: Why is `max_per_type` a static field?**
A: It's the loop bound for `place_in_area`'s padding and for the `lax.scan` output shape. Making it static lets XLA unroll the inner loop and keeps output tensor shapes invariant across batches. Changing `max_per_type` triggers recompilation.

**Q: Can I disable random start for just one dimension (e.g. random_pos=False but random_nutrition=True)?**
A: Yes — each flag is independent. Common patterns: fix `start_pos` for reproducibility but randomise nutrition/injury for domain randomisation. See doc `01` Reset Values table.

**Q: What happens if the spawn area is outside the grid (e.g. `[[0, 0], [20, 20]]` on a 10×10)?**
A: The sampler uses the raw bounds without clipping (`core.py:653: jax.random.randint(k, (2,), a[:2], a[2:])`). If `a[2]` exceeds `H`, sampled rows can be ≥ H and the resulting position would be invalid. Config loader does not validate this — keep spawn areas within grid bounds.
