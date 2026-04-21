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

Applied to all four entity types and summed:
```python
res_chem     = sense_resource(..., res_pos,     res_active,  res_property,     ...)
pred_chem    = sense_resource(..., pred_pos,    ones,        pred_property,     ...)
obs_chem     = sense_resource(..., obs_pos,     ones,        obs_property,      ...)
neutral_chem = sense_resource(..., neutral_pos, ones,        neutral_property,  ...)
obs_olfactory = res_chem + pred_chem + obs_chem + neutral_chem   # sensor.py:270
```

Per-entity computation:
```
dist = ‖res_pos[n] - agent_pos‖₂
decay = 1.0 / (dist^decay_power + 1e-10)   (or 2.0 if dist < 0.001)
mask = active AND dist <= sensor_radius
obs_olfactory += res_property[n] * decay * mask
```

Each entity contributes its 5-dim property vector weighted by distance decay. The result is a 5-dim vector representing the summed chemical gradient in the agent's vicinity. Food entities have signature `[1,0,0,0,0]`, predators `[0,1,0,0,0]`, bushes `[0,0,0,1,0]` by default config.

The `sensor_radius` (default 20) effectively covers the entire 10×10 grid.

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
