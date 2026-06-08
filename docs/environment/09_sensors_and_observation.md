# 09 — Sensors & Observation

> **Source**: `src/environment/sensor.py` (`get_observation`, `get_observation_breakdown`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview — What this doc covers

The agent in GridWorld Pain does not observe the world state directly. Instead, it receives a flat numerical vector assembled from up to ten distinct sensors. Some sensors measure conditions *inside* the agent's body (body temperature, injury, hunger — called **interoceptive**). Others measure conditions *outside* — what is nearby, what is bumping into the agent, what chemical traces are in the air (called **exteroceptive**). Some sensors are optional and can be switched off by a config flag; doing so removes their dimensions from the vector entirely.

This doc catalogs every sensor in the order they appear in the assembled observation vector: the exact code that produces each value, the formula used, which config flag enables it, and how many numbers it contributes to the vector.

Two functions own the complete picture:

- `get_observation(state, params, apply_noise=True)` (`sensor.py:270`) — assembles and returns the flat observation vector, optionally adding perceptual noise at the end.
- `get_observation_breakdown(params)` (`sensor.py:328`) — returns an ordered `dict` mapping each sensor name to its dimension count. This dict is the **single source of truth** for anything that needs to parse or label the observation vector — the perceptual noise system, the renderer, and downstream analysis code all read it. If you need to know where slice `[i:j]` of the observation belongs, call this function.

`get_observation` is JIT-compiled (`@jax.jit(static_argnames=['apply_noise'])`). `get_observation_breakdown` is not JIT-compiled — it runs at trace time.

---

## Observation Vector Order

Sensors appear in the vector in the exact order listed below. Sensors toggled off by their flag contribute **zero dimensions** and are absent from both the vector and from `get_observation_breakdown`.

| # | Sensor | Enabled by (`EnvParams` flag) | Dim | Value range | Formula |
|:-:|--------|-------------------------------|-----|-------------|---------|
| 1 | Injury | `injury_observable` | 1 | `[0, 1]` | `injury_level / max_injury` |
| 2 | Nutrition | `nutrition_observable` | 1 | `[0, 1]` | `nutrition / max_nutrition` |
| 3 | Satiation | Always on | 1 | `[0, 1]` | `satiation / max_satiation` |
| 4 | Interoceptive Nociception | `interoceptive_nociception_enabled` | 1 | `[0, 1]` | Convolved (delayed) injury trace, or direct `injury/max_injury` in passthrough mode |
| 5 | Extero Nociception | `nociception_enabled` | 1 | `[0, 1]` | Max intensity among current painful contacts |
| 6 | Olfaction | `olfactory_enabled` | `vector_size` (typically 5) | `[0, ∞)` | Σ property·decay(dist)·active over **3** entity pools: resources + unified animals + obstacles |
| 7 | Collision | Always on | `2r²+2r+1` | `{0, 1}` | Binary Manhattan diamond (OOB or blocking obstacle) |
| 8 | Proprioception | `proprioception_enabled` | `action_dim` | `{0, 1}` | One-hot of last action taken |
| 9 | Visual | `visual_sensor_enabled` | `(2r²+2r+1)×8` | `{0, 1}` | 8-channel Manhattan diamond (terrain + entity type) |
| 10 | Location | `location_sensor_enabled` | 2 | `[-1, 1]` | Normalised (row, col) |

With `sensor_range=1`, the collision diamond has 5 cells: `{center, up, right, down, left}`.
With `visual_sensor_range=0`, the visual diamond has 1 cell (agent's own cell): `1×8=8` dims.
With `visual_sensor_range=1`, it has `5×8=40` dims.

---

**Olfaction is NOT bounded** — it is a weighted sum of `property × decay × mask`, so values can exceed 1.0 when multiple entities are present or the agent overlaps a source (decay = 2.0 at zero distance). All other sensors are bounded as shown above.

### Hidden States & Gating

`Injury` and `Nutrition` sensors are **gateable**. If `params.injury_observable` or `params.nutrition_observable` is `False`, the corresponding slot is removed from the observation vector entirely. This forces the agent to infer body state from indirect signals. `Satiation` is always observable as the primary nonlinear energy feedback.

---

## Sensor Details

### 1–3 · Body-state sensors (Injury, Nutrition, Satiation)

Inlined in `get_observation` (`sensor.py:277–286`). Direct normalized reads of scalar state fields:

```python
[state.injury_level / params.max_injury]      # if injury_observable
[state.nutrition / params.max_nutrition]       # if nutrition_observable
[state.satiation / params.max_satiation]       # always
```

These three sensors are sometimes called the **interoceptive triad** — they measure the agent's internal physiological state. Injury and nutrition can be hidden (gated off) to require the agent to infer them from downstream signals.

---

### 4 · Interoceptive Nociception

`sense_interoceptive_nociception(state, params)` (`sensor.py:97`)

Enabled by: `params.interoceptive_nociception_enabled`

**What it models:** A delayed, smoothed perception of ongoing injury — analogous to the "slow burn" of tonic pain that persists after the acute contact has passed. The raw injury state is optionally hidden from the agent (sensor 1 gated off), so this signal becomes the agent's only window into body damage, but with a temporal lag built in.

**Input:** `state.nociception_history_buffer` — a circular buffer of length `interoceptive_kernel_length` storing past `injury_level` values. Index 0 is the most recent entry; the current step's injury is pushed in *after* observation is assembled, so it does not appear instantaneously.

#### Convolution mode (`interoceptive_convolution_enabled=True`, default)

The buffer is convolved with a pre-computed, normalized **alpha kernel** (also called a "rise-and-fall" or alpha-function kernel):

```
i = 0, 1, …, K-1     (K = interoceptive_kernel_length)
k_raw[i] = (i / τ) * exp(1 - i / τ)
kernel[i] = k_raw[i] / Σ k_raw      (normalized so kernel sums to 1.0)
```

- **τ (tau)** (`sensory.interoceptive_kernel_tau` in YAML): peak delay in steps. The kernel weight is largest at index `i = τ`. Larger τ = more temporal lag.
- **K** (`sensory.interoceptive_kernel_length`): the FIR filter window. Must be large enough relative to τ to capture the full kernel tail.
- **`kernel[0] = 0`** always (since `0/τ = 0`), so the most recent injury value is never instantaneously visible.
- Kernel is built in `config_loader.py:814–823` and stored as `params.interoceptive_kernel`.

Sensor output:

```python
convolved = jnp.sum(state.nociception_history_buffer * params.interoceptive_kernel)
output = convolved / max(params.max_injury, 1e-6)   # normalise to [0, 1]
```

(`sensor.py:111–112`)

At steady state (constant injury), the sensor reads `injury / max_injury` because the kernel is normalized to sum 1.0.

#### Passthrough mode (`interoceptive_convolution_enabled=False`)

Skips the kernel entirely; emits `state.injury_level / max(params.max_injury, 1e-6)` directly (`sensor.py:113`). Use for ablations where temporal delay should be removed.

Returns shape `[1]`.

> **Companion figure**: `docs/environment/09_interoceptive_nociception_dynamics.png` (generator: `09_interoceptive_nociception_plot.py`) — shows kernel shape, temporal lag, and decay vs. τ.

---

### 5 · Exteroceptive Nociception

`sense_extero_nociception(agent_pos, state, params)` (`sensor.py:59`)

Enabled by: `params.nociception_enabled`

**What it models:** Phasic (instantaneous) pain signals from physical contact with harmful entities — like touching something hot. This is the *external*, contact-based pain channel, as opposed to the internal, ongoing signal above.

Aggregates contact-based pain signals from **four sources** using **max** aggregation (the worst single contact wins; pain does not add up):

| Source | Contact condition | Intensity |
|--------|-------------------|-----------|
| Danger resources (res_type == 1) | `dist < 0.1` AND `res_active` | `params.res_nociception[n]` |
| Animals — **damaging only** | `dist < 0.1` AND `params.animal_is_damaging[a]` | `params.animal_nociception[a]` |
| Non-blocking obstacles (rock overlap) | `dist < 0.1` | `params.obs_nociception[o]` |
| Blocking obstacle collision | `state.last_collision_noc` (stored previous step) | stored float |

**v2.0 B2 fix:** animals are now a single unified list at `state.animal_pos` / `params.animal_nociception`. The `params.animal_is_damaging[a]` boolean gate (precomputed from animal class at config load) ensures neutral-class animals contribute **zero** nociception even if they are in contact (`sensor.py:76–80`). The old `state.pred_pos` / `params.pred_nociception` fields are gone.

```python
# sensor.py:73–80
dist_animal = jnp.linalg.norm(state.animal_pos - agent_pos, axis=-1)
animal_intensities = jnp.where(
    jnp.logical_and(dist_animal < 0.1, params.animal_is_damaging),
    params.animal_nociception, 0.0
)
max_animal = jnp.max(animal_intensities, initial=0.0)
```

Final output: `max(max_res, max_animal, max_obs_overlap, last_collision_noc)` — shape `[1]`, range `[0, 1]`.

The collision bump nociception (`last_collision_noc`) is stored in `EnvState` during Stage 4 of `jax_step` and read here on the following step.

---

### 6 · Olfaction / Chemical Gradient

`sense_resource(agent_pos, res_pos, res_active, res_property, radius, decay_power)` (`sensor.py:5`)

Enabled by: `params.olfactory_enabled`

**What it models:** The agent smells chemical traces emitted by nearby entities. Each entity type has a "chemical signature" — a vector of floats where different dimensions represent different odour compounds (e.g. food smell, predator musk, vegetation). The sensor sums all weighted signatures within the detection radius into a single `[vector_size]` vector; the agent must learn to decode which compounds mean what.

#### Config keys

All four keys are **mandatory** (missing → `ValueError`):

| YAML key | `EnvParams` field | Type | Effect |
|---|---|---|---|
| `sensory.olfactory_enabled` | `olfactory_enabled` | bool (static) | Gates the sensor; if false, zero dims |
| `sensory.vector_size` | `olfactory_vector_size` | int (static) | Length of each entity's chemical property vector (default 5) |
| `sensory.sensor_radius` | `sensor_radius` | float | Distance cutoff; entities beyond this emit nothing (default 20, covers full 10×10 grid) |
| `sensory.decay_power` | `sensor_decay` | float | Exponent in the distance-decay formula (default 2.0 = inverse-square) |

The olfaction dimension reported by `get_observation_breakdown` is read from `params.res_property.shape[-1]` (`sensor.py:349`), not `olfactory_vector_size` — so the two must match.

#### Per-entity chemical property vectors

Each entity carries a mean `properties` vector and an optional standard deviation `properties_std`. The means are stored in `EnvParams` as `[N_entities, vector_size]` arrays; a new sample is drawn into `EnvState` (`*_property_sampled`) at reset and on respawn.

| Entity type | YAML key | `EnvParams` field | Shape | `EnvState` field (sampled) | Shape | Mandatory? |
|---|---|---|---|---|---|---|
| Resources | `properties` (`_std`) | `res_property` (`_std`) | `[N_res, 5]` | `res_property_sampled` | `[N_res, 5]` | Yes (mean) / No (std) |
| Animals (unified) | `property` (`_std`) | `animal_property` (`_std`) | `[N, 5]` | `animal_property_sampled` | `[N, 5]` | Yes (mean) / No (std) |
| Obstacles | `properties` (`_std`) | `obs_property` (`_std`) | `[N_obs, 5]` | `obs_property_sampled` | `[N_obs, 5]` | **No** — defaults to zeros |

**v2.0 B2 fix:** predators and neutral animals are now a single unified animal list. The old separate `pred_property` / `neutral_property` / `pred_property_sampled` / `neutral_property_sampled` fields are gone. All animals share `animal_property` / `animal_property_sampled`. Note YAML key spelling: animals use `property` (singular); resources and obstacles use `properties` (plural).

When an entity list is empty the loader produces a zero-row array of shape `[0, chem_dim]`, where `chem_dim` is inferred from `res_property.shape[-1]` (`config_loader.py:117, 131, 161`).

#### Default chemical signatures (from `default.yaml`)

| Entity | `properties` / `property` | Interpretation |
|---|---|---|
| Food resource | `[1.0, 0.0, 0.0, 0.0, 0.0]` | Dim 0 = "food odour" |
| Danger resource | `[0.0, 0.0, 0.0, 0.0, 0.0]` | No chemical signal |
| Predator | `[0.0, 1.0, 0.0, 0.0, 0.0]` | Dim 1 = "predator odour" |
| Rabbit (neutral animal) | `[0.0, 0.3, 0.0, 0.0, 0.0]` | Dim 1 partial — weaker, predator-like scent |
| Rock (obstacle) | `[0.0, 0.0, 0.0, 0.0, 0.0]` | No chemical signal |
| Bush (obstacle) | `[0.0, 0.0, 0.0, 1.0, 0.0]` | Dim 3 = "vegetation odour" |
| Tree (obstacle) | `[0.0, 0.0, 0.0, 0.0, 1.0]` | Dim 4 = "tree odour" |

#### Runtime signal computation

**v2.0:** three entity pools are summed (down from four in v1 — unified `animal_chem` replaces the old separate `pred_chem + neutral_chem` calls):

```python
# sensor.py:300–303
res_chem    = sense_resource(state.agent_pos, state.res_pos,    state.res_active,
                              state.res_property_sampled, params.sensor_radius, params.sensor_decay)
animal_chem = sense_resource(state.agent_pos, state.animal_pos, ones(N_animal, bool),
                              state.animal_property_sampled, params.sensor_radius, params.sensor_decay)
obs_chem    = sense_resource(state.agent_pos, state.obs_pos,    ones(N_obs, bool),
                              state.obs_property_sampled, params.sensor_radius, params.sensor_decay)
obs_olfactory = res_chem + animal_chem + obs_chem
```

Resources are masked by `res_active` (consumed/inactive resources contribute nothing). Animals and obstacles are always considered present (always-ones activity mask).

Per-entity computation inside `sense_resource` (`sensor.py:5–22`):

```
diff  = entity_pos[n] - agent_pos              # [2]
dist  = ‖diff‖₂                               # L2 distance
decay = 2.0                 if dist < 0.001    # agent on top of entity
      = 1.0 / (dist^decay_power + 1e-10)       # otherwise (inverse-power decay)
mask  = (entity_active[n]) AND (dist ≤ sensor_radius)
signal += entity_property[n] * decay * mask    # [vector_size] accumulation
```

The `1e-10` in the denominator is a numerical safety guard; the `dist < 0.001` branch handles true zero-distance cases explicitly (returning 2.0) so the guard is never reached in practice.

The final `obs_olfactory` is a `[vector_size]` vector placed at observation positions `[5 : 5+vector_size]` (0-indexed, assuming Injury + Nutrition + Satiation + InteroNoc + ExteroNoc all enabled and occupying indices 0–4).

---

### 7 · Collision Sensor

`sense_collision(agent_pos, state, params)` (`sensor.py:24`)

Always enabled (no flag). Outputs a **binary** vector — each element is 1 if the corresponding cell is impassable, 0 if clear.

```
cell_coords = agent_pos + offsets   (offsets from get_visual_offsets(sensor_range))
blocked = out_of_bounds(cell_coord) OR blocking_obstacle_at(cell_coord)
```

Returns a flat binary vector of length `2*r²+2*r+1`. With `sensor_range=1` (default), this is 5 cells. The center cell (agent's own position) is always 0.

Cell order is defined by `get_visual_offsets(r)` (`sensor.py:115`). For `r=1` hardcoded as `[[0,0], [-1,0], [0,1], [1,0], [0,-1]]` — center, up, right, down, left.

Only **blocking** obstacles (rocks with `params.obs_blocking[o]=True`) trigger this sensor. Non-blocking obstacles (bushes, trees) do NOT appear here.

---

### 8 · Proprioception

Inlined in `get_observation` (`sensor.py:309–310`):

```python
obs_parts.append(jax.nn.one_hot(state.last_action, params.action_dim))
```

Enabled by: `params.proprioception_enabled`

One-hot encoding of `last_action` — the action taken on the immediately previous step. Dimension = `action_dim` (4, 5, or 6 depending on which optional actions are enabled).

`state.last_action` is set at the end of `jax_step`. On step 0 (post-reset) it is a placeholder: Rest (`4`) if rest is enabled, else Eat (`5`) — see `core.py:773`.

---

### 9 · Visual Sensor

`sense_visual(agent_pos, state, params)` (`sensor.py:145`)

Enabled by: `params.visual_sensor_enabled`

**What it models:** A local "snapshot" of what the agent can see in nearby cells, encoded as a binary grid of object types.

8-channel Manhattan diamond of radius `visual_sensor_range`. Each cell in the diamond encodes what is present there via an 8-element binary vector.

#### Channel mapping

| Channel | Content | Source |
|---------|---------|--------|
| 0 | Grass tile | `grid_location_type == 1` |
| 1 | Sand tile | `grid_location_type == 2` |
| 2 | Plain tile | `grid_location_type == 0` |
| 3 | Food resource | `res_type == 0 AND res_active` |
| 4 | Danger/hiding-predator resource | `res_type == 1 AND res_active` |
| 5 | **Predator-class animal** | `params.animal_visual_channel[a] == 5` |
| 6 | Rock/obstacle | always active |
| 7 | **Neutral-class animal** | `params.animal_visual_channel[a] == 7` |

**v2.0 B2 fix:** channels 5 and 7 are now driven by `params.animal_visual_channel` — a per-animal integer array (`sensor.py:202`). The config loader sets `animal_visual_channel[a] = 5` for predator-class animals and `7` for neutral-class animals. This replaces the old separate `pred_pos` → channel 5, `neutral_pos` → channel 7 logic.

The visual sensor uses a matmul-optimized approach (`sensor.py:197–220`):
1. Build resource visual property matrix: `res_props = one_hot(res_type==0 ? 3 : 4, 8)` — shape `[N_res, 8]`.
2. Build animal visual property matrix: `animal_props = one_hot(params.animal_visual_channel, 8)` — shape `[N_animal, 8]`. Each animal goes to channel 5 or 7 based on its class.
3. Build obstacle visual property matrix: all go to channel 6 — shape `[N_obs, 8]`.
4. Concatenate positions and properties: `all_pos [Total_E, 2]`, `all_props [Total_E, 8]`.
5. Compute presence matrix `matches [num_cells, Total_E]` by broadcasting position equality.
6. `vis_entities = matches.astype(float32) @ all_props` — shape `[num_cells, 8]`.
7. Add terrain one-hot `vis_background [num_cells, 8]`; mask out-of-bounds cells.

Output: flat vector of length `num_cells × 8 = (2r²+2r+1) × 8`.

With `visual_sensor_range=0` (default): 1 cell × 8 channels = **8 dims**. The agent sees only its own cell.

Channels 0–2 are terrain (one-hot, mutually exclusive). Channels 3–7 are entity presence and can overlap if multiple entities share a cell.

---

### 10 · Location Sensor

`sense_location(agent_pos, height, width)` (`sensor.py:52`):

```python
norm_r = (agent_pos[0] / (height - 1)) * 2 - 1
norm_c = (agent_pos[1] / (width  - 1)) * 2 - 1
return jnp.array([norm_r, norm_c])
```

Enabled by: `params.location_sensor_enabled`

Maps grid position to `[-1, 1]`. Top-left corner = `(-1, -1)`, bottom-right = `(1, 1)`. Disabled by default in the standard config.

---

## `get_observation_breakdown`

`get_observation_breakdown(params: EnvParams) → dict[str, int]` (`sensor.py:328`)

Returns an **ordered** dict mapping sensor name → dimension count, in observation-vector order. Sensors not enabled are **absent** from the dict entirely (not zero-dim). This is the authoritative structure consumed by the perceptual noise system and encoders.

```python
# Full set; subset depends on which flags are True
breakdown = {
    "Injury": 1,                         # if params.injury_observable
    "Nutrition": 1,                      # if params.nutrition_observable
    "Satiation": 1,                      # always
    "Interoceptive Nociception": 1,      # if params.interoceptive_nociception_enabled
    "Extero Nociception": 1,             # if params.nociception_enabled
    "Olfaction": params.res_property.shape[-1],  # if params.olfactory_enabled (sensor.py:349)
    "Collision": 2*r² + 2*r + 1,        # always (r = params.sensor_range)
    "Proprioception": params.action_dim, # if params.proprioception_enabled
    "Visual": num_vis_cells * 8,         # if params.visual_sensor_enabled
    "Location": 2,                       # if params.location_sensor_enabled
}
```

Note: olfaction dimension is read from `params.res_property.shape[-1]` directly, not `params.olfactory_vector_size`. Keep the two in sync to avoid downstream mismatches.

---

## `get_observation` Assembly

Full assembly order with exact sensor.py line references:

```python
# sensor.py:275–326
obs_parts = []

if params.injury_observable:                                  # sensor.py:278–279
    obs_parts.append([injury_level / max_injury])
if params.nutrition_observable:                               # sensor.py:282–283
    obs_parts.append([nutrition / max_nutrition])
obs_parts.append([satiation / max_satiation])                 # sensor.py:285–286  (always)

if params.interoceptive_nociception_enabled:                  # sensor.py:290–291
    obs_parts.append(sense_interoceptive_nociception(state, params))

if params.nociception_enabled:                                # sensor.py:294–295
    obs_parts.append(sense_extero_nociception(state.agent_pos, state, params))

if params.olfactory_enabled:                                  # sensor.py:299–303
    res_chem    = sense_resource(..., state.res_pos,    state.res_active, ...)
    animal_chem = sense_resource(..., state.animal_pos, ones(N_animal),   ...)  # B2: unified
    obs_chem    = sense_resource(..., state.obs_pos,    ones(N_obs),      ...)
    obs_parts.append(res_chem + animal_chem + obs_chem)

obs_parts.append(sense_collision(state.agent_pos, state, params))  # sensor.py:306  (always)

if params.proprioception_enabled:                             # sensor.py:309–310
    obs_parts.append(one_hot(state.last_action, action_dim))

if params.visual_sensor_enabled:                              # sensor.py:313–314
    obs_parts.append(sense_visual(state.agent_pos, state, params))

if params.location_sensor_enabled:                            # sensor.py:317–318
    obs_parts.append(sense_location(state.agent_pos, height, width))

obs = jnp.concatenate(obs_parts)

if apply_noise:                                               # sensor.py:324–325
    obs = apply_perceptual_noise(obs, state, params, obs_key)
return obs
```

The PRNG key for noise is derived by `jax.random.fold_in(state.key, 999)` (`sensor.py:273`) — a deterministic but independent branch from the main step key.

Noise is handled by `apply_perceptual_noise` (doc 10) — sensor.py is only responsible for handing off the clean vector.

---

## Clarifications / FAQ

**Q: Are observations normalised or raw?**
A: Mixed. Interoceptive scalars (sensors 1–4) are divided by max values → `[0, 1]`. Location is rescaled to `[-1, 1]`. Collision and Visual are binary `{0, 1}`. Proprioception is one-hot `{0, 1}`. **Olfaction is NOT bounded** — it is a sum of `property × decay × mask`, so values can exceed 1.0 (agent on top of an entity gives decay=2.0; multiple entities sum together).

**Q: What's the observation order, end-to-end?**
A: `[Injury(1?), Nutrition(1?), Satiation(1), InteroNoc(1?), ExteroNoc(1?), Olfaction(V?), Collision(C), Proprioception(A?), Visual(P?), Location(2?)]`. Sensors 1–2, 4–5, 8–10 are conditional on their enable flags; 3 and 7 are always present. To get the exact runtime mapping, call `get_observation_breakdown(params)`.

**Q: Is `get_observation` called inside `jax_step`?**
A: No. `jax_step` returns only the new state and scalars. The caller must explicitly call `get_observation(state, params)` to materialise the observation vector. `ParallelEnv` wraps this into a single step interface. See doc `11`.

**Q: What happens when `apply_noise=False`?**
A: The clean pre-noise vector is returned. Useful for evaluation and analysis. Note `apply_noise` is a static arg (`static_argnames=['apply_noise']`) — changing it triggers recompilation.

**Q: Is `last_action` from the step just completed or the one about to be taken?**
A: Just completed. `state.last_action` is set at the end of `jax_step` to the action just executed (`core.py:514`). The very first observation (step 0, post-reset) uses a placeholder — Rest (`4`) if rest is enabled, else Eat (`5`) (`core.py:773`).

**Q: The olfaction decay is `1/(d^p + 1e-10)` — what's the `1e-10` for?**
A: Numerical safety for near-zero distances. The `dist < 0.001` branch catches true-zero-distance cases (returning 2.0); the `1e-10` prevents a numerical divide-by-zero for tiny-but-nonzero distances. Change `sensor.py:14` if you need a different saturation value.

**Q: Why is agent-on-entity decay exactly 2.0?**
A: Hardcoded upper bound — chosen so that olfaction saturates predictably when the agent overlaps a source. Without this, the decay would be ≈ `1/1e-10 = 1e10`, which would swamp the entire signal.

**Q: Does olfaction see through obstacles?**
A: Yes. There is no line-of-sight check (`sensor.py:5–22`). A chemical source behind a wall is still detected, attenuated by distance only.

**Q: Do inactive resources contribute to olfaction?**
A: No — `res_active` masks them out (`sensor.py:17`). Animals and obstacles are always included (their activity mask is `jnp.ones(..., bool)`).

**Q: What does the collision sensor return for the center cell?**
A: Always `0`. The center is the agent's own cell — it can't be out-of-bounds and the agent couldn't be there if it were blocked.

**Q: What ordering do the collision/visual diamond cells use?**
A: Manhattan-distance shells (d=0, 1, 2, …), within each shell a fixed direction sweep. For `r=1`: `[center, (−1,0), (0,1), (1,0), (0,−1)]` — center, up, right, down, left. See `get_visual_offsets(r)` at `sensor.py:115`.

**Q: The visual sensor has 8 channels — but grass/sand/plain are mutually exclusive. Why not 6 channels?**
A: Because a cell can simultaneously have terrain + entity (e.g. bush on grass). Channels 0–2 are terrain (one-hot), channels 3–7 are entity type (can overlap if multiple entities share a cell). `3 terrain + 5 entity = 8`.

**Q: How do predators vs. neutral animals appear in the visual sensor?**
A: Via `params.animal_visual_channel` — a per-animal integer array (`sensor.py:202`). Predator-class animals are assigned channel 5; neutral-class animals are assigned channel 7. The assignment is fixed at config load time. An agent observing channel 5 lit up knows "predator here"; channel 7 = "neutral animal here". Internal FSM state (PATROL / HUNT / RETURN) is not visible.

**Q: Do neutral-class animals emit exteroceptive nociception?**
A: No. `params.animal_is_damaging[a]` is `False` for neutral-class animals (`state.py:123`), so the `jnp.where` gate in `sense_extero_nociception` (`sensor.py:76–78`) zeroes their contribution regardless of proximity.

**Q: Is the olfaction vector a sum across all entity types in one `[vector_size]` vector, or separate per-type?**
A: Summed into one (`sensor.py:303`: `res_chem + animal_chem + obs_chem`). **v2.0:** three pools (not four — unified animal list). The agent cannot decompose "predator scent" from "neutral scent" directly — only from which chemical channels are hot (which is why channels are conventionally reserved per type).

**Q: Does the sensor system know about noise, or is noise applied after?**
A: After. `get_observation` assembles the clean vector then optionally calls `apply_perceptual_noise` (`sensor.py:324–325`). The noise system reads `get_observation_breakdown` to find which slice belongs to which modality.

**Q: What if `olfactory_vector_size` in YAML doesn't match the actual `property` vector length?**
A: `get_observation_breakdown` reads `params.res_property.shape[-1]` directly (`sensor.py:349`) — so the breakdown uses the actual vector length. `olfactory_vector_size` is a separate static param used elsewhere (e.g. encoder input shape). If the two diverge, downstream dimension mismatches follow. Keep them in sync.

**Q: The location sensor is `[-1, 1]` but on a 1×1 grid would divide by zero. Is this guarded?**
A: No. `height=1` makes `(H-1)=0` and the result would be NaN. Assume `H, W >= 2`.

**Q: The noise key uses `fold_in(state.key, 999)` — why 999?**
A: Arbitrary constant that salts the noise-RNG branch so it is independent of the step-key branches (which use `split`, not `fold_in`). Any constant works; 999 is a readable sentinel.

**Q: The code has `# 5. Olfaction` commented immediately after `# 5. Extero Nociception` in `get_observation` — is this a numbering bug?**
A: Yes, a comment-numbering error at `sensor.py:297` — olfaction is labelled `# 5.` when it should be `# 6.`. This is a documentation-only issue in the source comments; the observation assembly order itself is correct and follows the table above.
