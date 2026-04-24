# 09 — Sensors & Observation

> **Source**: `src/environment/sensor.py` (`get_observation`, `get_observation_breakdown`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

`get_observation(state, params, apply_noise=True)` (`sensor.py:244`) assembles the flat observation vector that is fed to the agent's policy network. It concatenates active sensor outputs in a fixed, hard-coded order. Sensors are conditionally included based on static boolean flags in `EnvParams`; disabled sensors contribute zero dimensions.

`get_observation_breakdown(params)` (`sensor.py:295`) is the **authoritative** mapping from sensor name to dimension count. It is the single source of truth consulted by:
1. The perceptual noise system to target the correct slice of the observation vector.
2. The renderer to label sensor panels.
3. Any analysis code that needs to parse the observation vector.

Both functions are decorated with `@jax.jit` (the latter via `static_argnames=['apply_noise']`).

---

## Observation Vector Order

All sensors always present (dims are fixed by static params):

| # | Sensor | Enabled by | Dim | Value range | Formula |
|:-:|--------|-----------|-----|-------------|---------|
| 1 | Injury | Always on | 1 | `[0, 1]` | `injury_level / max_injury` |
| 2 | Nutrition | Always on | 1 | `[0, 1]` | `nutrition / max_nutrition` |
| 3 | Satiation | Always on | 1 | `[0, 1]` | `satiation / max_satiation` |
| 4 | Extero Nociception | `nociception_enabled` | 1 | `[0, 1]` | max intensity among contacts |
| 5 | Olfaction | `olfactory_enabled` | `olfactory_vector_size` (5) | `[0, ∞)` | Σ property·decay(dist)·active |
| 6 | Collision | Always on | `2r²+2r+1` | `{0, 1}` | binary Manhattan diamond |
| 7 | Proprioception | `proprioception_enabled` | `action_dim` | `{0, 1}` | one-hot last action |
| 8 | Visual | `visual_sensor_enabled` | `(2r²+2r+1)×8` | `{0, 1}` | 8-ch Manhattan diamond |
| 9 | Location | `location_sensor_enabled` | 2 | `[-1, 1]` | normalised (row, col) |

With `sensor_range=1`, the collision diamond has 5 cells: `{center, up, right, down, left}`.
With `visual_sensor_range=0`, the visual diamond has 1 cell (agent's own cell): `1×8=8` dims.
With `visual_sensor_range=1`, it has `5×8=40` dims.

---

## Interoceptive Sensors (1–3)

Always on. These are the core interoceptive channels — the agent's "body awareness":

```python
obs[0] = state.injury_level / params.max_injury       # sensor.py:253
obs[1] = state.nutrition / params.max_nutrition       # sensor.py:256
obs[2] = state.satiation / params.max_satiation       # sensor.py:259
```

All three are normalised to `[0, 1]`. Note: comparing raw state values (0–100) to these observations directly will show a 100× difference — always normalise first.

---

## Exteroceptive Nociception (4)

`sense_extero_nociception(agent_pos, state, params)` (`sensor.py:59`)

Aggregates contact-based pain signals from four sources using **max** aggregation (not sum):

| Source | Condition | Intensity |
|--------|-----------|-----------|
| Danger resources | `dist < 0.1` AND `res_active` | `res_nociception[n]` |
| Predators | `dist < 0.1` | `pred_nociception[p]` |
| Non-blocking obstacles | `dist < 0.1` | `obs_nociception[o]` |
| Blocking obstacle collision | `last_collision_noc` in state | stored from prev step |

Final value: `max(max_res, max_pred, max_obs_overlap, last_collision_noc)`

Returns shape `[1]`. Output range is `[0, 1]` (nociception intensities are configured in this range).

The collision bump nociception (`last_collision_noc`) is stored in `EnvState` during Stage 4 of `jax_step` and read here in the next observation.

---

## Olfaction / Chemical Gradient (5)

`sense_resource(agent_pos, res_pos, res_active, res_property, radius, decay_power)` (`sensor.py:5`)

### Config keys that control olfaction

All four keys below are **mandatory** (missing → `ValueError`):

| YAML key | `EnvParams` field | Type | Effect |
|---|---|---|---|
| `sensory.olfactory_enabled` | `olfactory_enabled` | bool (static) | Gates the entire sensor; if false, zero dims in obs |
| `sensory.vector_size` | `olfactory_vector_size` | int (static) | Length of each entity's chemical property vector (default 5) |
| `sensory.sensor_radius` | `sensor_radius` | float | Distance cutoff; entities beyond this are masked out (default 20, covers full 10×10 grid) |
| `sensory.decay_power` | `sensor_decay` | float | Exponent in the distance-decay formula (default 2.0 = inverse-square) |

The olfaction dimension reported by `get_observation_breakdown` is read directly from `params.res_property.shape[-1]` (`sensor.py:314`), not from `olfactory_vector_size` — so the two must match.

### Per-entity chemical property vectors

Each entity definition in YAML carries a chemical property vector (`properties` / `property`) indicating the mean, and optionally a standard deviation vector (`properties_std` / `property_std`). The config loader reads them and stores them in `EnvParams` as `[N_entities, vector_size]` JAX arrays. The mean and std arrays are **constant** for the whole episode (stored in `EnvParams`), but they are sampled into `EnvState` (`*_property_sampled`) at reset and upon respawn.

| Entity type | YAML key | `EnvParams` field | Shape | `EnvState` field (sampled) | Shape | Mandatory? |
|---|---|---|---|---|---|---|
| Resources | `properties` (`_std`) | `res_property` (`_std`) | `[N_res, 5]` | `res_property_sampled` | `[N_res, 5]` | Yes (mean) / No (std) |
| Predators | `property` (`_std`) | `pred_property` (`_std`) | `[N_pred, 5]` | `pred_property_sampled` | `[N_pred, 5]` | Yes (mean) / No (std) |
| Neutral animals | `property` (`_std`) | `neutral_property` (`_std`) | `[N_neutral, 5]` | `neutral_property_sampled` | `[N_neutral, 5]` | Yes (mean) / No (std) |
| Obstacles | `properties` (`_std`) | `obs_property` (`_std`) | `[N_obs, 5]` | `obs_property_sampled` | `[N_obs, 5]` | **No** — defaults to zeros |

Note the YAML key spelling inconsistency: resources and obstacles use `properties` (plural); predators and neutral animals use `property` (singular).

When an entity list is empty the loader produces a zero-row array of shape `[0, chem_dim]`, where `chem_dim` is inferred from `res_property.shape[-1]` (`config_loader.py:117, 131, 161`).

### Default chemical signatures (from `default.yaml`)

| Entity | `properties` / `property` | Interpretation |
|---|---|---|
| Food resource | `[1.0, 0.0, 0.0, 0.0, 0.0]` | Dim 0 = "food odour" |
| Danger resource | `[0.0, 0.0, 0.0, 0.0, 0.0]` | No chemical signal |
| Predator | `[0.0, 1.0, 0.0, 0.0, 0.0]` | Dim 1 = "predator odour" |
| Rabbit (neutral) | `[0.0, 0.3, 0.0, 0.0, 0.0]` | Dim 1 partial, weaker predator-like scent |
| Rock (obstacle) | `[0.0, 0.0, 0.0, 0.0, 0.0]` | No chemical signal |
| Bush (obstacle) | `[0.0, 0.0, 0.0, 1.0, 0.0]` | Dim 3 = "vegetation odour" |
| Tree (obstacle) | `[0.0, 0.0, 0.0, 0.0, 1.0]` | Dim 4 = "tree odour" |

### Runtime signal computation

Applied to all four entity types and summed (`sensor.py:266–270`):

```python
res_chem     = sense_resource(..., res_pos,     state.res_active,        state.res_property_sampled,     ...)
pred_chem    = sense_resource(..., pred_pos,    ones(N_pred, bool),       state.pred_property_sampled,    ...)
obs_chem     = sense_resource(..., obs_pos,     ones(N_obs, bool),        state.obs_property_sampled,     ...)
neutral_chem = sense_resource(..., neutral_pos, ones(N_neutral, bool),    state.neutral_property_sampled, ...)
obs_olfactory = res_chem + pred_chem + obs_chem + neutral_chem
```

Resources are masked by `res_active` (consumed/inactive resources contribute nothing). Predators, obstacles, and neutral animals are always considered present (always-ones mask).

Per-entity computation inside `sense_resource` (`sensor.py:5–22`):

```
diff  = entity_pos[n] - agent_pos              # [2]
dist  = ‖diff‖₂                               # L2 distance
decay = 2.0                  if dist < 0.001   # agent on top of entity
      = 1.0 / (dist^decay_power + 1e-10)       # otherwise (inverse-power decay)
mask  = (entity_active[n]) AND (dist ≤ sensor_radius)
signal += entity_property[n] * decay * mask    # [vector_size] accumulation
```

The final `obs_olfactory` is a `[vector_size]` vector of summed weighted properties placed at observation indices `[4 : 4+vector_size]`.

---

## Collision Sensor (6)

`sense_collision(agent_pos, state, params)` (`sensor.py:24`)

Binary Manhattan diamond of radius `sensor_range`. Each cell outputs 1 if it is blocked:

```
cell_coords = agent_pos + offsets   (offsets from get_visual_offsets(sensor_range))
blocked = out_of_bounds(cell_coord) OR blocking_obstacle_at(cell_coord)
```

Returns a flat binary vector of length `2*r²+2*r+1`. With `sensor_range=1` (default), this is 5 cells ordered `[center, up, right, down, left]`. The center cell is always 0 (agent's own cell cannot be OOB or blocked).

Cell order is defined by `get_visual_offsets(r)` (`sensor.py:89`), which for `r=1` hardcodes `[[0,0], [-1,0], [0,1], [1,0], [0,-1]]` for consistent ordering.

---

## Proprioception (7)

Inlined in `get_observation` (`sensor.py:277`):
```python
obs_parts.append(jax.nn.one_hot(state.last_action, params.action_dim))
```

One-hot encoding of `last_action` (the action taken on the previous step). Dimension = `action_dim` (4, 5, or 6 depending on which actions are enabled).

When `proprioception_enabled=False`, this sensor contributes no dimensions.

---

## Visual Sensor (8)

`sense_visual(agent_pos, state, params)` (`sensor.py:119`)

8-channel Manhattan diamond of radius `visual_sensor_range`. Each cell encodes what is present at that location via an 8-channel binary vector.

**Channel mapping**:
| Channel | Content | Source |
|---------|---------|--------|
| 0 | Grass tile | `grid_location_type == 1` |
| 1 | Sand tile | `grid_location_type == 2` |
| 2 | Plain tile | `grid_location_type == 0` |
| 3 | Food resource | `res_type == 0 AND res_active` |
| 4 | Danger resource | `res_type == 1 AND res_active` |
| 5 | Predator | always active |
| 6 | Rock/obstacle | always active |
| 7 | Neutral animal | always active |

**Implementation** (`sensor.py:147`): uses a matmul-optimised approach:
1. Compute entity presence matrix `matches [num_cells, total_entities]` by broadcasting position equality.
2. Multiply by entity property one-hot matrix `all_props [total_entities, 8]`.
3. Add background tile one-hot `vis_background [num_cells, 8]`.
4. Mask out-of-bounds cells.

Output: flat vector of length `num_cells × 8 = (2r²+2r+1) × 8`.

With `visual_sensor_range=0` (default config): 1 cell × 8 channels = 8 dims. The agent only sees what is at its own cell.

---

## Location Sensor (9)

`sense_location(agent_pos, height, width)` (`sensor.py:52`):
```
norm_r = (agent_pos[0] / (height - 1)) * 2 - 1
norm_c = (agent_pos[1] / (width  - 1)) * 2 - 1
return [norm_r, norm_c]
```

Maps grid position to `[-1, 1]`. Top-left corner = `(-1, -1)`, bottom-right = `(1, 1)`. Disabled by default in the standard config.

---

## `get_observation_breakdown`

`get_observation_breakdown(params) → dict[str, int]` (`sensor.py:295`)

Returns an ordered dict mapping sensor names to their dimension counts. Used by noise system and encoders.

```python
breakdown = {
    "Injury": 1,
    "Nutrition": 1,
    "Satiation": 1,
    "Extero Nociception": 1,           # if nociception_enabled
    "Olfaction": olfactory_vector_size, # if olfactory_enabled
    "Collision": 2*r^2 + 2*r + 1,
    "Proprioception": action_dim,       # if proprioception_enabled
    "Visual": num_vis_cells * 8,        # if visual_sensor_enabled
    "Location": 2,                      # if location_sensor_enabled
}
```

Sensors not enabled are absent from the dict entirely (not zero-dim). The noise system uses this to build per-modality index arrays.

---

## `get_observation` Assembly

```python
obs_parts = []
obs_parts.append([injury/max_injury])             # always
obs_parts.append([nutrition/max_nutrition])       # always
obs_parts.append([satiation/max_satiation])       # always
if nociception_enabled:   obs_parts.append(sense_extero_nociception(...))
if olfactory_enabled:     obs_parts.append(sense_resource(...) × 4 summed)
obs_parts.append(sense_collision(...))            # always
if proprioception_enabled: obs_parts.append(one_hot(last_action))
if visual_sensor_enabled:  obs_parts.append(sense_visual(...))
if location_sensor_enabled: obs_parts.append(sense_location(...))

obs = concatenate(obs_parts)
if apply_noise:
    obs = apply_perceptual_noise(obs, state, params, obs_key)
```

The PRNG key for noise is derived by `jax.random.fold_in(state.key, 999)` — a deterministic but independent branch from the main step key.

---

## Clarifications / FAQ

**Q: Are observations normalised or raw?**
A: Mixed. Interoceptive (1–3) are divided by max values → `[0, 1]`. Location is rescaled to `[-1, 1]`. Collision and Visual are binary `{0, 1}`. Proprioception is one-hot `{0, 1}`. **Olfaction is NOT bounded** — it's a sum of `property × decay × mask`, so it can exceed 1 with many entities or at close range (agent on top of an entity gives decay=2.0).

**Q: What's the observation order, end-to-end?**
A: `[Injury(1), Nutrition(1), Satiation(1), ExteroNoc(?), Olfaction(?), Collision(?), Proprioception(?), Visual(?), Location(?)]`. The only always-on sensors are the first 3 plus Collision. Every other sensor's dim depends on the enabled flag. To get the exact mapping at runtime, call `get_observation_breakdown(params)` — this is authoritative.

**Q: Is `get_observation` called inside `jax_step`?**
A: No. `jax_step` returns only the new state and scalars. The caller must explicitly call `get_observation(state, params)` to materialise the observation vector. `ParallelEnv` wraps this into a single step interface. See doc `11`.

**Q: What happens when `apply_noise=False`?**
A: The clean pre-noise vector is returned. Useful for evaluation/analysis. Note `apply_noise` is a static arg (`static_argnames=['apply_noise']`) — changing it triggers recompilation.

**Q: Is `last_action` proprioception from the step *just completed* or the step *about to be taken*?**
A: Just completed. `state.last_action` is set at the end of `jax_step` to the action that was just executed (`core.py:514`). The very first observation (step 0, post-reset) uses a placeholder — Rest (`4`) if rest is enabled, else Eat (`5`) (`core.py:773`).

**Q: The olfaction decay is `1/(d^p + 1e-10)` — what's the `1e-10` for?**
A: Numerical safety. At `d=0` this would divide by zero, but the `dist < 0.001` branch handles that case explicitly (returns 2.0). The `1e-10` protects against tiny but nonzero distances that would round to a huge number.

**Q: Why is agent-on-entity decay exactly 2.0?**
A: Hardcoded upper bound — chosen so that olfaction saturates predictably when the agent overlaps a source. Without this, the decay would be ≈ `1/1e-10 = 1e10`, which would swamp the entire signal. Change `sensor.py:14` if you need a different saturation value.

**Q: Does olfaction see through obstacles?**
A: Yes. There is no line-of-sight check (`sensor.py:5-22`). A chemical source on the other side of a wall is still smelled, attenuated by distance only.

**Q: Do inactive resources contribute to olfaction?**
A: No — `res_active` masks them out (`sensor.py:17`). Predators, obstacles, and neutrals are always counted (the mask passed in is `jnp.ones(...)`).

**Q: What does the collision sensor return for the center cell?**
A: Always `0`. The center is the agent's own cell — it can't be out of bounds, and the agent couldn't be there if it were blocked. Only non-center cells can be `1`.

**Q: What ordering do the collision/visual diamond cells use?**
A: Manhattan-distance shells in order `d=0, 1, 2, ...`. Within each shell, a fixed direction sweep (see `get_visual_offsets` at `sensor.py:89`). For `r=1`: `[center, (−1,0), (0,1), (1,0), (0,−1)]` — up/right/down/left. Use `get_visual_offsets(r)` if you need programmatic access.

**Q: The visual sensor has 8 channels — but grass/sand/plain are mutually exclusive. Why not 6 channels?**
A: Because a cell can simultaneously have terrain + entity (e.g. bush on grass). Channels 0–2 encode terrain (one-hot), channels 3–7 encode entities (can overlap if multiple entities stack). Total 8 = `3 + 5 entity channels`.

**Q: Do predators showing up in `Visual[5]` account for Hunt/Patrol/Return state?**
A: No. Channel 5 just marks "predator present at this cell" regardless of state. The predator's internal FSM is not observable. Inference of predator intent is the agent's job.

**Q: Is the olfaction vector sum across all entity types in one `[vector_size]` vector, or separate per-type?**
A: Summed into one (`sensor.py:270`: `res_chem + pred_chem + obs_chem + neutral_chem`). The agent cannot separate "food scent" from "predator scent" except by comparing which channel is hot — which is why channels are conventionally reserved per entity type (channel 0 = food odor, channel 1 = predator odor, etc.).

**Q: Does the sensor system know about noise, or is noise applied after?**
A: Noise is applied after. `get_observation` assembles the clean vector then optionally calls `apply_perceptual_noise` (`sensor.py:291-292`). The noise system reads `get_observation_breakdown` to know which slice belongs to which modality.

**Q: What if `olfactory_vector_size` in YAML doesn't match the actual `property` vector length?**
A: `get_observation_breakdown` reads `params.res_property.shape[-1]` directly (`sensor.py:314`) — so the breakdown uses the actual vector length. `olfactory_vector_size` is a separate static param used elsewhere (e.g. for encoder input shape). If the two diverge, you'll see a dimension mismatch downstream. Keep them in sync.

**Q: The location sensor is `[-1, 1]` but on a 1×1 grid would divide by 0. Is this a real concern?**
A: Only on pathological grids. `height=1` makes `(H-1)=0` and division would NaN. The code does not guard against this. Assume `H, W >= 2`.

**Q: The noise key uses `fold_in(state.key, 999)` — why 999?**
A: Arbitrary constant that salts the noise-RNG branch. Ensures the noise key is independent of the step RNG branches (which come from `split`, not `fold_in`). Any constant would work; 999 is just a readable sentinel.
