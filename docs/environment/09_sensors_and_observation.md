# 09 — Sensors & Observation

> **Source**: `src/environment/sensor.py` (`get_observation`, `get_observation_breakdown`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview — What this doc covers

The agent in GridWorld Pain does not observe the world state directly. Instead, it receives a flat numerical vector assembled from up to ten distinct sensors. **Since v3.1 the two exteroceptive senses are also spatial** — see [Directional sensors](#directional-sensors-v31--v32) at the end of this doc. Some sensors measure conditions *inside* the agent's body (body temperature, injury, hunger — called **interoceptive**). Others measure conditions *outside* — what is nearby, what is bumping into the agent, what chemical traces are in the air (called **exteroceptive**). Some sensors are optional and can be switched off by a config flag; doing so removes their dimensions from the vector entirely.

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

#### Full implementation

Source: `src/environment/sensor.py:97–113`

```python
def sense_interoceptive_nociception(state: EnvState, params: EnvParams):
    """
    Tonic interoceptive pain. Two modes (selected statically at JIT time):

    - Convolution mode (default): convolves recent injury history with a
      normalized alpha kernel (peak at τ steps). Buffer slot 0 = most recent
      injury; kernel[0]=0 so current-step injury does not leak instantaneously.
    - Passthrough mode (`interoceptive_convolution_enabled=False`): bypasses
      the kernel and returns the current normalized injury directly. Use this
      for ablations where the agent should perceive injury without delay.

    Returns scalar in [0, 1].
    """
    if params.interoceptive_convolution_enabled:
        convolved = jnp.sum(state.nociception_history_buffer * params.interoceptive_kernel)
        return jnp.array([convolved / jnp.maximum(params.max_injury, 1e-6)])
    return jnp.array([state.injury_level / jnp.maximum(params.max_injury, 1e-6)])
```

> **API notes**
>
> - `params.interoceptive_convolution_enabled` is a static field (`struct.field(pytree_node=False)`), so the `if` branch is resolved at trace time — the compiler bakes in only one path and eliminates the other entirely. See [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - `jnp.sum(buffer * kernel)` is a standard dot-product with no explicit loop — JAX traces it as a single element-wise multiply followed by a reduction. See [primer: masking](00_jax_primer.md#masking) for the broader pattern of element-wise gating.
> - `jnp.maximum(params.max_injury, 1e-6)` is the branchless safe-divide guard — avoids a Python `if max_injury == 0` inside JIT. See [primer: branchless](00_jax_primer.md#branchless).

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

#### Full implementation

Source: `src/environment/sensor.py:59–95`

```python
def sense_extero_nociception(agent_pos, state: EnvState, params: EnvParams):
    """
    Continuous Phasic Nociceptor: Detects contact with hiding predators, animals, and rocks.
    Returns the maximum intensity among all current painful contacts.

    B2 fix: predator contact now uses unified state.animal_pos / params.animal_nociception
    masked by params.animal_is_damaging (replaces old state.pred_pos / params.pred_nociception).
    """
    # 1. Hiding Predator Contact (resources with type==1)
    dist_res = jnp.linalg.norm(state.res_pos - agent_pos, axis=-1)
    # Intensity = intensity from params if at position and active
    res_intensities = jnp.where(jnp.logical_and(state.res_active, dist_res < 0.1), params.res_nociception, 0.0)
    max_res = jnp.max(res_intensities, initial=0.0)

    # 2. Animal Contact (B2 fix — unified; only damaging animals emit nociception)
    if state.animal_pos.shape[0] > 0:
        dist_animal = jnp.linalg.norm(state.animal_pos - agent_pos, axis=-1)
        animal_intensities = jnp.where(
            jnp.logical_and(dist_animal < 0.1, params.animal_is_damaging),
            params.animal_nociception, 0.0
        )
        max_animal = jnp.max(animal_intensities, initial=0.0)
    else:
        max_animal = 0.0

    # 3. Rock Overlap Contact (Non-blocking)
    dist_obs = jnp.linalg.norm(state.obs_pos - agent_pos, axis=-1)
    obs_intensities = jnp.where(dist_obs < 0.1, params.obs_nociception, 0.0)
    max_obs_overlap = jnp.max(obs_intensities, initial=0.0)

    # 4. Rock Collision Contact (Bumping)
    # Stored in state from jax_step
    max_collision = state.last_collision_noc

    # Result is the maximum intensity
    final_noc = jnp.max(jnp.array([max_res, max_animal, max_obs_overlap, max_collision]), initial=0.0)
    return jnp.array([final_noc])
```

> **API notes**
>
> - **`jnp.linalg.norm(..., axis=-1)`** computes L2 distance for every entity in one vectorized call — shape `[N]` out, no Python loop. See [primer: linalg](00_jax_primer.md#linalg).
> - **`jnp.where(condition, true_val, false_val)`** is the branchless gate used twice: once for `res_active AND dist < 0.1`, once for `animal_is_damaging AND dist < 0.1`. Both evaluate both branches unconditionally; only the selected value is returned. See [primer: branchless](00_jax_primer.md#branchless) and [primer: masking](00_jax_primer.md#masking).
> - **`params.animal_is_damaging`** is a boolean array (`[N_animal]`) precomputed at config load — it makes the nociception gate a pure data operation rather than a string comparison at step time.
> - **`state.animal_pos.shape[0] > 0`** is a Python `if` on a static shape — shapes are known at trace time, so the compiler eliminates the dead branch. See [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - **`jnp.max(..., initial=0.0)`** is safe even on zero-length arrays (the `initial` keyword prevents the "empty sequence" error). This is the standard pattern here for optional entity lists.

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
| `sensory.decay_power` | `sensor_decay` | float | Exponent in the distance-decay formula. **Ships as `1.0`** (gentle, long-range). It was 2.0 until 2026-07-26 — see the change log in [CONFIG_CRITICAL_SETTINGS.md](CONFIG_CRITICAL_SETTINGS.md), and note eleven configs still pin 2.0. |

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
| Predator | `[0.0, 0.7, 0.5, 0.0, 0.0]` ± `[0, 0.4, 0.4, 0, 0]` | Dims 1 and 2, **deliberately overlapping with the rabbit** |
| Rabbit (neutral animal) | `[0.0, 0.5, 0.7, 0.0, 0.0]` ± `[0, 0.4, 0.4, 0, 0]` | The same two dims with 1 and 2 swapped |

> **These two are engineered to be confusable.** Predator and rabbit differ only by a swap of
> channels 1 and 2, with σ=0.4 on both, redrawn per episode and clipped to [0,1]. Smell alone
> cannot cleanly separate threat from harmless. Food, by contrast, owns a noise-free channel.
> (An earlier revision of this doc listed `[0,1,0,0,0]` / `[0,0.3,0,0,0]` — that has not matched
> `default.yaml` for some time.)
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

**All three pools are masked by their per-episode `*_active` flags** — `res_active`, `animal_active`, `obs_active` — so consumed resources and deactivated animals/obstacles contribute nothing. (An earlier revision of this doc said animals and obstacles used an always-ones mask; that stopped being true with PER_EPISODE_ENV_VARIANCE.)

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

#### Full implementation

Source: `src/environment/sensor.py:5–22`

```python
def sense_resource(agent_pos, res_pos, res_active, res_property, radius, decay_power):
    """Vectorized Resource Sensor (Chemical signature gradient)."""
    # [num_res, 2]
    diff = res_pos - agent_pos
    dist = jnp.linalg.norm(diff, axis=-1)
    
    # Handle singularity (agent ON resource)
    # Original: dist < 0.001 -> decay = 2.0
    # Else: 1.0 / (dist ** decay_power)
    decay = jnp.where(dist < 0.001, 2.0, 1.0 / (jnp.power(dist, decay_power) + 1e-10))
    
    # Mask by radius and activity
    mask = jnp.logical_and(res_active, dist <= radius)
    
    # Apply mask and sum: [num_res, vector_size] -> [vector_size]
    weighted_props = res_property * decay[:, None] * mask[:, None]
    obs = jnp.sum(weighted_props, axis=0)
    return obs
```

> **API notes**
>
> - **`jnp.linalg.norm(diff, axis=-1)`** computes the L2 distance from the agent to every entity in one vectorized call over the `[N, 2]` difference array — output shape `[N]`. See [primer: linalg](00_jax_primer.md#linalg).
> - **`jnp.where(dist < 0.001, 2.0, ...)`** is the branchless singularity guard. The condition `dist < 0.001` evaluates to a boolean array `[N]`; `jnp.where` selects element-wise between the two branches with no Python conditional. See [primer: branchless](00_jax_primer.md#branchless).
> - **`decay[:, None] * mask[:, None]`** — the `[:, None]` broadcasts a `[N]` vector to `[N, 1]` so it multiplies each row of the `[N, vector_size]` property matrix. This is the standard JAX broadcasting pattern for per-entity scaling. See [primer: masking](00_jax_primer.md#masking).
> - **`jnp.sum(weighted_props, axis=0)`** reduces `[N, vector_size]` → `[vector_size]`, accumulating all entity contributions into one chemical vector.

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

#### Full implementation

Source: `src/environment/sensor.py:24–50`

```python
def sense_collision(agent_pos, state: EnvState, params: EnvParams):
    """Manhattan Collision Sensor (checks OOB and blocking obstacles)."""
    sensor_range = params.sensor_range
    offsets = get_visual_offsets(sensor_range) # [num_cells, 2]
    num_cells = offsets.shape[0]
    cell_coords = agent_pos + offsets # [num_cells, 2]
    
    # 1. Bounds check
    is_out_of_bounds = jnp.any(jnp.logical_or(
        cell_coords < 0,
        cell_coords >= jnp.array([params.height, params.width])
    ), axis=-1)
    
    # 2. Blocking Obstacles (Rocks)
    def check_blocking_rock(coord):
        # coord: [2]
        is_here = jnp.all(state.obs_pos == coord, axis=-1)
        # Check if any rock at this position is blocking
        is_blocking = jnp.logical_and(is_here, params.obs_blocking)
        return jnp.any(is_blocking)
        
    is_blocked_by_rock = jax.vmap(check_blocking_rock)(cell_coords)
    
    # Total collision: OOB or Blocking Rock
    collision = jnp.logical_or(is_out_of_bounds, is_blocked_by_rock).astype(jnp.float32)
    
    return collision
```

> **API notes**
>
> - **`jax.vmap(check_blocking_rock)(cell_coords)`** maps a single-cell obstacle check over all `[num_cells, 2]` coordinates — one compiled kernel runs over the whole diamond in parallel. `check_blocking_rock` is written for one coordinate; `vmap` lifts it to the batch. See [primer: vmap](00_jax_primer.md#vmap).
> - **`jnp.all(..., axis=-1)`** and **`jnp.any(..., axis=-1)`** are element-wise reductions used for coordinate matching (`[N_obs, 2]` → `[N_obs]`) and bounds checking (`[num_cells, 2]` → `[num_cells]`) — both branchless. See [primer: masking](00_jax_primer.md#masking).
> - **`jnp.logical_and(is_here, params.obs_blocking)`** gates the obstacle match by the blocking flag — a boolean array operation, no Python loop over obstacles. See [primer: branchless](00_jax_primer.md#branchless).
> - **`.astype(jnp.float32)`** converts the boolean result to float for concatenation into the observation vector — the observation vector is a uniform `float32` array throughout.

---

### `get_visual_offsets` — diamond cell ordering

`get_visual_offsets(sensor_range)` (`sensor.py:115`)

This helper generates the list of `(row, col)` offsets that define the Manhattan diamond used by both the collision and visual sensors. It is called at trace time (not JIT-compiled itself) to produce a small constant array of cell offsets.

#### Full implementation

Source: `src/environment/sensor.py:115–143`

```python
def get_visual_offsets(sensor_range):
    """Generates Manhattan diamond offsets in a consistent order."""
    offsets = []
    # Ordering: Manhattan distance 0, then 1, then 2...
    # Within each distance, we can use a fixed direction order (e.g. Up, Right, Down, Left)
    for d in range(sensor_range + 1):
        if d == 0:
            offsets.append([0, 0])
        else:
            # Manhattan distance d: |dr| + |dc| = d
            # We iterate to find all pairs
            for dr in range(-d, d + 1):
                dc_abs = d - abs(dr)
                if dc_abs == 0:
                    offsets.append([dr, 0])
                else:
                    # Both + and - for dc
                    offsets.append([dr, dc_abs])
                    offsets.append([dr, -dc_abs])
    
    # Sort for consistency: primary by distance, secondary by row, tertiary by col
    offsets_arr = jnp.array(offsets)
    dist = jnp.sum(jnp.abs(offsets_arr), axis=1)
    # Use jnp.lexsort or similar if needed, but for small ranges a simple nested loop is fine
    # Let's just use the manual order for range 0 and 1 as they are common
    if sensor_range == 1:
        return jnp.array([[0,0], [-1,0], [0,1], [1,0], [0,-1]])
    
    return offsets_arr
```

> **API notes**
>
> - This function runs in **Python** at trace time — the `for` loops and `if` branches are plain Python, not JAX. The result is a small constant `jnp.array` that the compiler treats as a compile-time constant. This is safe because `sensor_range` is a static field. See [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - For `sensor_range=1` the function returns a hardcoded `jnp.array` directly, bypassing the loop-and-sort logic — a minor shortcut that avoids the unsorted ordering the general loop produces for `d>1`.
> - The `dist` variable computed mid-function (line 137) is **not used** for `sensor_range=1` (the hardcoded return fires first) — it is dead code for the common case.

---

### 8 · Proprioception

Inlined in `get_observation` (`sensor.py:309–310`):

```python
obs_parts.append(jax.nn.one_hot(state.last_action, params.action_dim))
```

Enabled by: `params.proprioception_enabled`

One-hot encoding of `last_action` — the action taken on the immediately previous step. Dimension = `action_dim` (4, 5, or 6 depending on which optional actions are enabled).

`state.last_action` is set at the end of `jax_step`. On step 0 (post-reset) it is a placeholder: Rest (`4`) if rest is enabled, else Eat (`5`) — see `core.py:773`.

> **API notes**
>
> - **`jax.nn.one_hot(index, num_classes)`** returns a float32 vector of length `num_classes` with a `1.0` at position `index` and `0.0` elsewhere. It is fully differentiable and JIT-safe. See [primer: one-hot](00_jax_primer.md#one-hot).
> - `params.action_dim` is a static field (it controls the vector length), so the compiler knows the output shape at trace time without needing the actual value of `state.last_action`.

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

#### Full implementation

Source: `src/environment/sensor.py:145–220`

```python
def sense_visual(agent_pos, state: EnvState, params: EnvParams):
    """Matmul-optimized Visual Sensor (simplified object recognition).

    Channel mapping:
      0: Grass (loc 1), 1: Sand (loc 2), 2: Plain (loc 0)
      3: Food (res_type 0), 4: Hiding Predator (res_type 1)
      5: Predator (animal_visual_channel=5), 6: Rock (obstacle), 7: Neutral Animal (channel=7)

    B2 fix: dynamic entities are now [res, animal, obs] — unified animals replace the
    old separate pred/neutral lists. Each animal uses params.animal_visual_channel for
    its per-entity visual channel index (predator=5, neutral=7).
    """
    vis_range = params.visual_sensor_range
    offsets = get_visual_offsets(vis_range)  # [num_cells, 2]
    num_cells = offsets.shape[0]
    cell_coords = agent_pos + offsets  # [num_cells, 2]

    # 1. Bounds check
    is_in_bounds = jnp.all(jnp.logical_and(
        cell_coords >= 0,
        cell_coords < jnp.array([params.height, params.width])
    ), axis=-1)

    # Safe coordinates for indexing background
    safe_coords = jnp.where(is_in_bounds[:, None], cell_coords, 0)

    # 2. Background (Grid Properties) - Vectorized Indexing
    loc_types = params.grid_location_type[safe_coords[:, 0], safe_coords[:, 1]]
    # Mapping: loc 1 -> channel 0, loc 2 -> channel 1, loc 0 -> channel 2
    vis_background = jax.nn.one_hot(jnp.where(loc_types == 1, 0, jnp.where(loc_types == 2, 1, 2)), 8)
    vis_background = vis_background * is_in_bounds[:, None]

    # 3. Dynamic Entities (Resources, Animals, Obstacles) — B2 fix: unified animal list
    num_res    = state.res_pos.shape[0]
    num_animal = state.animal_pos.shape[0]
    num_obs    = state.obs_pos.shape[0]

    # Combine all dynamic entity positions — 3-way concat
    parts_pos = [state.res_pos]
    if num_animal > 0:
        parts_pos.append(state.animal_pos)
    parts_pos.append(state.obs_pos)
    all_pos = jnp.concatenate(parts_pos, axis=0)  # [Total_E, 2]

    # Combine activity status (animals/obstacles always active)
    parts_active = [state.res_active]
    if num_animal > 0:
        parts_active.append(jnp.ones(num_animal, dtype=jnp.bool_))
    parts_active.append(jnp.ones(num_obs, dtype=jnp.bool_))
    all_active = jnp.concatenate(parts_active, axis=0)  # [Total_E]

    # Visual Property Matrix [Total_E, 8]
    res_props = jax.nn.one_hot(jnp.where(params.res_type == 0, 3, 4), 8)  # [num_res, 8]
    obs_props = jax.nn.one_hot(jnp.full((num_obs,), 6), 8)                 # [num_obs, 8]
    parts_props = [res_props]
    if num_animal > 0:
        # Each animal uses its per-entity visual channel (predator=5, neutral=7)
        animal_props = jax.nn.one_hot(params.animal_visual_channel, 8)     # [N, 8]
        parts_props.append(animal_props)
    parts_props.append(obs_props)
    all_props = jnp.concatenate(parts_props, axis=0)  # [Total_E, 8]

    # Apply activity mask
    all_props = all_props * all_active[:, None]

    # Compute Matches [num_cells, Total_E]
    matches = jnp.all(cell_coords[:, None, :] == all_pos[None, :, :], axis=-1)

    # Sum properties: [num_cells, Total_E] @ [Total_E, 8] -> [num_cells, 8]
    vis_entities = jnp.matmul(matches.astype(jnp.float32), all_props)

    # Final assembly
    total_vis = vis_background + vis_entities
    total_vis = total_vis * is_in_bounds[:, None]

    return total_vis.flatten()
```

> **API notes**
>
> - **`jax.nn.one_hot(index_array, num_classes=8)`** is called three times here — once per entity type — to build the per-entity visual property matrix. A batched array of integer channel indices in, a float32 `[N, 8]` matrix out. See [primer: one-hot](00_jax_primer.md#one-hot). The terrain case chains two `jnp.where` calls to remap `{0,1,2}` tile types to `{2,0,1}` channel indices before calling `one_hot`.
> - **`safe_coords = jnp.where(is_in_bounds[:, None], cell_coords, 0)`** is the OOB guard for grid indexing: out-of-bounds cells are clamped to coordinate `[0,0]` so the array index never raises. The `is_in_bounds` mask zeros their contribution at the end — branchless safe indexing. See [primer: masking](00_jax_primer.md#masking).
> - **`matches = jnp.all(cell_coords[:, None, :] == all_pos[None, :, :], axis=-1)`** uses broadcasting to compare every cell against every entity at once — shape `[num_cells, Total_E]`. No Python loop over entities or cells. See [primer: masking](00_jax_primer.md#masking).
> - **`jnp.matmul(matches.astype(float32), all_props)`** — casting bool to float32 and using matmul is the key efficiency trick: "which entities are in this cell" × "what channel does each entity own" in one BLAS call, shape `[num_cells, 8]`. See [primer: linalg](00_jax_primer.md#linalg).
> - **`if num_animal > 0:`** branches on a static shape, resolved at trace time — compiler eliminates the dead path for configs with no animals. See [primer: static-dynamic](00_jax_primer.md#static-dynamic).

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

#### Full implementation

Source: `src/environment/sensor.py:52–57`

```python
def sense_location(agent_pos, height, width):
    """Normalized Agent Location Sensor."""
    # Center coordinates to [-1, 1]
    norm_r = (agent_pos[0] / (height - 1)) * 2 - 1
    norm_c = (agent_pos[1] / (width - 1)) * 2 - 1
    return jnp.array([norm_r, norm_c])
```

> **API notes**
>
> - `height` and `width` are static fields passed as plain Python ints — arithmetic on them produces a Python scalar that scales the traced `agent_pos` element. No special JAX API needed; the division and linear rescaling are standard element-wise ops.
> - **No guard for `height=1`** (would produce NaN). The convention is `H, W >= 2` — see FAQ below.

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

#### Full implementation

Source: `src/environment/sensor.py:328–368`

```python
def get_observation_breakdown(params: EnvParams):
    """Returns a dict of {sensor_name: dimension} for observation components."""
    breakdown = {}
    
    # 1. Injury (hidden when injury_observable=False) — interoceptive
    if params.injury_observable:
        breakdown["Injury"] = 1
    # 2. Nutrition (hidden when nutrition_observable=False) — interoceptive
    if params.nutrition_observable:
        breakdown["Nutrition"] = 1
    # 3. Satiation — interoceptive
    breakdown["Satiation"] = 1
    # 4. Interoceptive Nociception — interoceptive (delayed/passthrough injury)
    if params.interoceptive_nociception_enabled:
        breakdown["Interoceptive Nociception"] = 1
    # 5. Extero Nociception — exteroceptive
    if params.nociception_enabled:
        breakdown["Extero Nociception"] = 1
    
    # 5. Olfaction
    if params.olfactory_enabled:
        breakdown["Olfaction"] = int(params.res_property.shape[-1])
    
    # 6. Collision
    num_coll_cells = 2 * (params.sensor_range**2) + 2 * params.sensor_range + 1
    breakdown["Collision"] = int(num_coll_cells)
    
    # 7. Proprioception
    if params.proprioception_enabled:
        breakdown["Proprioception"] = int(params.action_dim)
        
    # 8. Visual
    if params.visual_sensor_enabled:
        num_vis_cells = 2 * (params.visual_sensor_range**2) + 2 * params.visual_sensor_range + 1
        breakdown["Visual"] = int(num_vis_cells * 8)
    
    # 9. Location
    if params.location_sensor_enabled:
        breakdown["Location"] = 2
        
    return breakdown
```

> **API notes**
>
> - This function is **not** JIT-compiled. It runs at Python level on `EnvParams` static fields. Every `if` here branches on a static field value — safe as plain Python.
> - The `breakdown` dict insertion order is the **observation vector order**. Python 3.7+ dicts preserve insertion order, so iterating `breakdown.items()` is the same as iterating observation slots.
> - `int(params.res_property.shape[-1])` materializes the JAX shape integer to a plain Python `int` — `breakdown` values are pure Python ints, not JAX scalars, which matters for downstream use as `range()` arguments and slice bounds.

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

#### Full implementation

Source: `src/environment/sensor.py:269–326`

```python
@jax.jit(static_argnames=['apply_noise'])
def get_observation(state: EnvState, params: EnvParams, apply_noise=True):
    """Assembles the full observation vector, including noise if enabled."""
    # Salt the state key for observation noise
    obs_key = jax.random.fold_in(state.key, 999)
    
    obs_parts = []
    
    # 1. Injury (hidden when injury_observable=False) — interoceptive
    if params.injury_observable:
        obs_parts.append(jnp.array([state.injury_level / params.max_injury]))

    # 2. Nutrition (hidden when nutrition_observable=False) — interoceptive
    if params.nutrition_observable:
        obs_parts.append(jnp.array([state.nutrition / params.max_nutrition]))

    # 3. Satiation — interoceptive
    obs_parts.append(jnp.array([state.satiation / params.max_satiation]))

    # 4. Interoceptive Nociception — interoceptive
    #    (Tonic — delayed function of hidden injury, or passthrough if convolution disabled)
    if params.interoceptive_nociception_enabled:
        obs_parts.append(sense_interoceptive_nociception(state, params))

    # 5. Extero Nociception — exteroceptive (phasic, multi-source contact)
    if params.nociception_enabled:
        obs_parts.append(sense_extero_nociception(state.agent_pos, state, params))

    # 5. Olfaction Sensor (Resources + Animals + Obstacles)
    # B2 fix: unified animal_chem replaces separate pred_chem + neutral_chem calls.
    if params.olfactory_enabled:
        res_chem = sense_resource(state.agent_pos, state.res_pos, state.res_active, state.res_property_sampled, params.sensor_radius, params.sensor_decay)
        animal_chem = sense_resource(state.agent_pos, state.animal_pos, jnp.ones(state.animal_pos.shape[0], dtype=jnp.bool_), state.animal_property_sampled, params.sensor_radius, params.sensor_decay)
        obs_chem = sense_resource(state.agent_pos, state.obs_pos, jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_), state.obs_property_sampled, params.sensor_radius, params.sensor_decay)
        obs_parts.append(res_chem + animal_chem + obs_chem)
    
    # 6. Collision
    obs_parts.append(sense_collision(state.agent_pos, state, params))
    
    # 7. Proprioception (Previous Action)
    if params.proprioception_enabled:
        obs_parts.append(jax.nn.one_hot(state.last_action, params.action_dim))
    
    # 8. Visual Sensor
    if params.visual_sensor_enabled:
        obs_parts.append(sense_visual(state.agent_pos, state, params))
    
    # 9. Location
    if params.location_sensor_enabled:
        obs_parts.append(sense_location(state.agent_pos, params.height, params.width))
    
    # Assemble final vector
    obs = jnp.concatenate(obs_parts)
    
    # Apply Perceptual Precision Modulation
    if apply_noise:
        return apply_perceptual_noise(obs, state, params, obs_key)
    return obs
```

> **API notes**
>
> - **`@jax.jit(static_argnames=['apply_noise'])`** — `apply_noise` is a plain Python `bool` argument (not a struct field), so it cannot be traced. Marking it static means the compiler resolves the `if apply_noise:` branch at trace time, producing two compiled variants: one with the noise call and one without. Calling `get_observation(..., apply_noise=False)` triggers a second compile on first use. See [primer: static-argnames](00_jax_primer.md#static-argnames).
> - **`obs_key = jax.random.fold_in(state.key, 999)`** — `fold_in` creates an independent sub-key by hashing the base key with a constant integer (here `999`). It does not consume the key (unlike `split`), so `state.key` is unchanged. The constant `999` is a readable sentinel that distinguishes this RNG branch from all others in the step. See [primer: prng](00_jax_primer.md#prng).
> - Every `if params.<flag>:` branch here is a Python conditional on a static field — resolved at trace time; disabled sensors are compiled away entirely. The observation vector shape is therefore fixed at compile time for a given `EnvParams`. See [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - **`jnp.concatenate(obs_parts)`** assembles the final vector from the list of per-sensor arrays. Each element of `obs_parts` is a 1D array; the concatenation produces the flat observation vector of total length equal to the sum of all enabled sensor dimensions.
> - `apply_perceptual_noise` (doc 10) is not reproduced here — this function hands off the clean vector and an independent noise key.

---

## `build_sensory_viz` — renderer hookup

`build_sensory_viz(obs, state, params, true_obs=None)` (`sensor.py:370`)

This function is **renderer-facing** — it parses the flat observation vector into the structured `sensory_data` list consumed by `renderer.render_jax_state`. It is not part of the training pipeline. Full coverage is in doc 12 (renderer); only the hookup is noted here.

It iterates `get_observation_breakdown(params)` to find each sensor's slice in the flat vector, then packages each slice into a typed dict (`'type': 'intensity'`, `'type': 'diamond'`, `'type': 'visual_grid'`, etc.) that the renderer knows how to draw. The optional `true_obs` argument enables side-by-side noisy vs. clean display.

---

## Directional sensors (v3.1 / v3.2)

Both outward-facing senses gained optional spatial structure. **Everything here defaults off**;
a config that sets none of these keys produces byte-identical observations to pre-v3.1.
Config reference: [02_config_schema.md](02_config_schema.md#directional-sensors-v31--v32).
Rationale and measurements: [[VISUAL_PSF_MECHANISM_STUDY]], [[OLFACTORY_EXPANSION_STUDY]],
[[ONSOURCE_RULE_STUDY]], [[V31_IMPLEMENTATION_REPORT]].

### Olfaction over a diamond — `sense_olfaction_cells`

`olfactory_grid_range: r` evaluates the *same* field at every cell of a Manhattan diamond
instead of only at the agent. The per-cell computation is the untouched `sense_resource`, and
the three pools are still summed in the original order (`res + animal + obs`) per cell — which
is why the centre cell stays bit-identical to the old single sample **at every range**, not just
at `r=0`. Out-of-bounds cells read exactly zero, matching the visual sensor. Flattened
cell-major, contributing `(2r²+2r+1) × vector_size` dims.

`r = 0` takes a static fallback to the original single-point expression, so parity does not
depend on `vmap`-of-one compiling identically.

> **`olfactory_grid_range` is not `sensor_radius`.** The first is *where the field is sampled*
> (a property of the sensor); the second is *how far a smell carries* (a property of the field,
> and at 20 it never binds on a 10×10 grid).

### Anisotropic point-spread on vision — `_psf_weights`

`visual_blur_enabled` replaces the boolean match matrix with gaussian weights, elongated along
the agent→entity ray:

```
d      = ||e - agent||
σ_par  = max(radial_scale · d, sigma_floor)
σ_perp = max(σ_par / anisotropy, sigma_floor)
w      = exp(−v∥²/2σ_par² − v⊥²/2σ_perp²) / (2π σ_par σ_perp)
```

So *where* something is becomes vague while *which way* it lies stays sharp. Mass
normalisation is what makes distant entities fade — normalising over the visible cells instead
would cancel the falloff entirely.

Two implementation constraints that are not optional: `v = c − e` is computed **before**
projecting (the `c·û − e·û` form routes geometry through a matmul, which runs in reduced
precision on Ampere-class GPUs), and `sigma_floor` is **required**, because an entity on the
agent's own cell gives `σ_par = 0` and an infinite peak without it.

### Per-entity visibility — `_visual_mask_gate`

`visual_mask: none | far | all`. Gates on the **entity's** Manhattan distance from the agent,
not the cell's. Under exact matching the two are equivalent; under blur they are not, and
gating per cell leaves a `far`-masked entity depositing its blur tail in the agent's own cell.

### Presence versus count — `visual_value_mode`

`sum` (default) is a weighted sum, so two rocks in a cell read 2.0. `clamp` caps each channel
at 1.0, giving presence. Applied to the **entity** contribution only — terrain is ground, not
an object, and keeps its own value.

### Line-of-sight occlusion — `_occlusion_gate`

An entity is hidden when a nearer entity flagged `blocks_sight` lies inside the shadow cone of
the ray to it. A **cone**, not a strict grid line: on an integer grid exact collinearity fires
almost only along axes and perfect diagonals, which would make the sensor blind north-south and
clear-sighted obliquely. The gate is a per-entity factor, so it composes with the mask and
activity masks by plain multiplication.

**This scene is dense** — up to 36 entities on 100 cells. Obstacles-only blocking hides 31% of
live entities at 5° and 59% at 15°. Calibrate before concluding anything from a null result.

### The on-source rule

`sense_resource` returns `1/(0.5^γ)` when a sampling point sits exactly on a source — "standing
on it means half a cell away", the grid's own resolution limit. At the shipped `decay_power: 1.0`
this is bit-identical to the literal `2.0` it replaced; at other γ it stays on the curve, where
the constant did not. The eleven configs pinning `decay_power: 2.0` therefore read 4.0 rather
than 2.0 on-source.

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
