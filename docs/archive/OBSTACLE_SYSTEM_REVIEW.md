# Obstacle System Review & Bush Feature Pre-Analysis

> **Date**: 2026-02-20
> **Branch**: `feature/fixSensorFlag`
> **Scope**: Full review of obstacle generation, interaction, and sensing. Pre-analysis for the upcoming Bush obstacle type.

---

## 1. Current Obstacle Architecture

Obstacles are the only **static entities** in the environment. Unlike resources (which respawn), predators (which move), and neutral animals (which wander), obstacles are placed at reset and remain fixed for the entire episode.

### 1.1 Data Structures

**State** (`EnvState` in `state.py`):

| Field | Shape | Description |
|-------|-------|-------------|
| `obs_pos` | `[num_obs, 2]` | Row/col positions of all obstacles |

**Params** (`EnvParams` in `state.py`):

| Field | Shape | Description |
|-------|-------|-------------|
| `obs_blocking` | `[num_obs]` bool | Whether each obstacle blocks movement |
| `obs_spawn_area` | `[num_obs, 4]` | `[min_r, min_c, max_r, max_c]` (0-based, exclusive max) |
| `obs_damage` | `[num_obs, 2]` | `[min_damage, max_damage]` per obstacle |
| `obs_property` | `[num_obs, vector_size]` | Chemical/olfactory signature |
| `obs_nociception` | `[num_obs]` | Pain intensity on contact |
| `obs_type` | `[num_obs]` int32 | Index into `obstacle_names` tuple |
| `obstacle_names` | `tuple[str, ...]` | Sorted unique names (e.g., `("rock", "tree")`) |

### 1.2 Existing Obstacle Types

| Type | Blocking | Damage | Nociception | Chemical Signature | Behavior |
|------|----------|--------|-------------|--------------------|----------|
| **Rock** | No | [0.1, 0.5] | 0.3 | `[0,0,0,0,0]` (silent) | Agent walks through, takes damage on overlap |
| **Tree** | Yes | [0.1, 0.5] | 0.1 | `[0,0,0,0,1]` (channel 4) | Agent cannot enter cell; takes bump damage |

---

## 2. Obstacle Generation (Reset)

### 2.1 Config Loading (`config_loader.py`)

Obstacles are defined in YAML under `environment.obstacles`. Each entry is expanded by its `count` field:

```yaml
obstacles:
  - name: "rock"
    count: 10           # Creates 10 independent rock instances
    area: [[1, 11], [10, 20]]
    blocking: false
    damage: [0.1, 0.5]
    nociception_intensity: 0.3
    properties: [0.0, 0.0, 0.0, 0.0, 0.0]
```

The config loader performs:
1. **Count expansion**: `count: 10` produces 10 entries sharing the same template.
2. **Coordinate transform**: YAML's 1-based inclusive `[[r1,c1],[r2,c2]]` becomes 0-based exclusive-max `[r1-1, c1-1, r2, c2]` for `jax.random.randint`.
3. **Name indexing**: `obstacle_names` is a sorted deduplicated tuple. `obs_type` maps each instance to its index in that tuple.
4. **Default handling**: `blocking` defaults to `True`, `damage` defaults to `0.0`, `nociception_intensity` defaults to `0.3`.

### 2.2 Placement (`core.py::jax_reset`)

At reset, each obstacle gets a random position within its spawn area:

```python
def sample_obs_pos(ok, area):
    return jax.random.randint(ok, (2,), area[:2], area[2:])

obs_pos = jax.vmap(sample_obs_pos)(obs_keys, params.obs_spawn_area)
```

**Important**: Placement uses independent random sampling **without overlap checking**. The `jax_reset` docstring acknowledges this: overlap probability is low on a 20x20 grid with ~50 entities. Obstacles can overlap with each other, the agent, or other entity types.

> **Note**: The `OBJECT_GENERATION.md` doc describes a collision-aware spawning pipeline (`_sample_unoccupied`, `_place_entities`) that does not match the current implementation. The actual code uses simple vmapped random sampling.

---

## 3. Obstacle Interactions (Step)

The obstacle system touches four distinct areas of the step function:

### 3.1 Agent Movement Blocking (`move_agent`)

```
Agent action -> Calculate new_pos -> Check blocking obstacles at new_pos
  -> If blocked: stay at current_pos, set is_collision=True
  -> If clear: move to new_pos
```

The collision check is straightforward:

```python
is_collision = jnp.any(jnp.logical_and(
    jnp.all(obs_pos == new_pos, axis=-1),
    obs_blocking
))
final_pos = jnp.where(is_collision, pos, new_pos)
```

### 3.2 Damage Application (`jax_step`)

Obstacles deal damage through two independent paths:

**Path A: Overlap Damage (non-blocking)**

When the agent lands on a cell containing a non-blocking obstacle:

```python
at_obs = jnp.all(state.obs_pos == new_agent_pos, axis=-1)
sampled_obs_damage = jax.random.uniform(damage_key, ..., minval=obs_damage[:,0], maxval=obs_damage[:,1])
damage_obs_overlap = jnp.sum(jnp.where(
    jnp.logical_and(at_obs, jnp.logical_not(params.obs_blocking)),
    sampled_obs_damage, 0.0
))
```

**Path B: Collision Damage (blocking)**

When the agent bumps into a blocking obstacle (movement denied):

```python
at_attempted_obs = jnp.all(state.obs_pos == attempted_pos, axis=-1)
damage_obs_collision = jnp.where(
    just_collided,
    jnp.max(jnp.where(at_attempted_obs, sampled_obs_damage, 0.0)),
    0.0
)
```

Both paths use the same damage key, so the sampled values are identical per-obstacle per step. The total obstacle damage is `damage_obs_overlap + damage_obs_collision`.

### 3.3 Nociception Signal

Obstacles contribute to the extero nociception sensor through two channels:

1. **Overlap contact** (non-blocking): Direct position match triggers `obs_nociception` intensity.
2. **Collision contact** (blocking): Stored as `state.last_collision_noc` for the sensor to read.

The final nociception output is the **max** across all sources (resources, predators, obstacle overlap, obstacle collision).

### 3.4 Predator & Neutral Animal Movement

Both predators and neutral animals avoid blocking obstacles:

```python
def check_collision(p_pos, old_p_pos):
    is_coll = jnp.any(jnp.logical_and(jnp.all(obs_pos == p_pos, axis=-1), obs_blocking))
    return jnp.where(is_coll, old_p_pos, p_pos)

new_pos = jax.vmap(check_collision)(new_pos, pred_pos)
```

**No interaction with non-blocking obstacles** for predators/neutrals. They walk through rocks without consequence.

---

## 4. Obstacle Sensing

### 4.1 Olfactory Sensor

Obstacles emit chemical gradients via `obs_property`:

```python
obs_chem = sense_resource(agent_pos, state.obs_pos,
    jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_),
    params.obs_property, params.sensor_radius, params.sensor_decay)
```

Currently, rocks have `[0,0,0,0,0]` (olfactory-silent) and trees have `[0,0,0,0,1]` (channel 4). This means the agent can smell trees but not rocks.

### 4.2 Collision Sensor

The collision sensor reports a Manhattan diamond of cells, marking each as 1.0 if it's out-of-bounds OR contains a blocking obstacle:

```python
def check_blocking_rock(coord):
    is_here = jnp.all(state.obs_pos == coord, axis=-1)
    is_blocking = jnp.logical_and(is_here, params.obs_blocking)
    return jnp.any(is_blocking)
```

Non-blocking obstacles are **invisible** to the collision sensor.

### 4.3 Visual Sensor

All obstacles map to **visual channel 6** ("Rock") regardless of their actual `name` or `obs_type`:

```python
obs_props = jax.nn.one_hot(jnp.full((num_obs,), 6), 8)
```

This is a hardcoded mapping. The visual sensor currently has 8 channels:

| Channel | Entity |
|---------|--------|
| 0 | Grass terrain |
| 1 | Sand terrain |
| 2 | Plain terrain |
| 3 | Food resource |
| 4 | Danger resource |
| 5 | Predator |
| 6 | Rock/Obstacle (all types) |
| 7 | Neutral animal |

**All obstacle types are visually identical to the agent.** Trees and rocks look the same in the visual observation.

---

## 5. Renderer & Visualization

The renderer (`renderer.py`) does support distinct obstacle rendering:

```python
obs_icon = params.obstacle_names[obs_types[i]]
draw_icon(ax_grid, or_, oc, obs_icon, ...)
```

It looks up the icon name from `obstacle_names` using the `obs_type` index, so different obstacle names will render with different icons if matching assets exist in the `assets/` directory. The minimap uses a flat `COLORS['rock']` for all obstacles.

---

## 6. Discrepancies Between Code and Documentation

| Topic | `OBJECT_GENERATION.md` Says | Actual Code |
|-------|----------------------------|-------------|
| Spawn collision avoidance | Unified pipeline with `_sample_unoccupied`, `_place_entities`, sequential scan with occupancy grid | Simple `jax.vmap` random sampling, no overlap checking |
| Resource respawn collision | `_respawn_resources` with occupancy grid | Simple random sample within `spawn_area` |
| Key split at reset | 4 streams: `agent_key, place_key, body_key` | 6 streams: `agent_key, res_key, pred_key, body_key, neutral_key` |

The doc appears to describe a planned or previous implementation that was simplified. The current code explicitly notes this trade-off in the `jax_reset` docstring.

---

## 7. Pre-Analysis: Adding Bush Obstacle

### 7.1 Concept

A **bush** is a non-blocking obstacle where the agent can hide from predators. While on a bush tile, the agent should become invisible (or harder to detect) by predators, creating a strategic safe-haven mechanic.

### 7.2 What the Current System Already Supports

The obstacle infrastructure is well-suited for adding a new named type:

- **Named types**: `obstacle_names` and `obs_type` already support multiple named obstacle types (rocks, trees). Adding `"bush"` requires only a YAML entry.
- **Per-obstacle parameters**: Each obstacle instance already has independent `blocking`, `damage`, `nociception_intensity`, and `properties` arrays. A bush can be configured with `blocking: false`, `damage: [0, 0]`, and a unique chemical signature.
- **Renderer**: Already uses dynamic icon lookup from `obstacle_names`, so a bush icon just needs an asset file.
- **Config loader**: Fully supports new obstacle entries with all fields.

### 7.3 What Needs New Implementation

The core mechanic (hiding from predators) **does not exist** in the current system. Specifically:

| Component | Current State | Required Change |
|-----------|--------------|-----------------|
| **Predator detection** | `dist = manhattan(pred_pos, agent_pos)` then check `dist <= detection_range` | Must mask/reduce detection when agent is on a bush tile |
| **Visual sensor channels** | All obstacles are channel 6 | Bush needs a distinct visual channel so the agent can distinguish it from rocks/trees |
| **Predator state machine** | No concept of agent visibility | Needs a visibility check before transitioning to HUNT state |
| **EnvState tracking** | No "agent is hidden" flag | May benefit from an `agent_hidden` boolean for reward/logging, or compute it on-the-fly |

### 7.4 Design Considerations

**Detection Masking Strategy** (in `update_predators`):

There are two approaches:
- **Binary**: If agent is on any bush tile, predator detection is completely blocked (detection_range effectively becomes 0).
- **Graduated**: Bush reduces effective detection range (e.g., halved) or adds a detection probability check. More nuanced but more complex in pure JAX.

**Predator Behavior When Agent Hides Mid-Chase**:

If a predator is in HUNT state and the agent enters a bush:
- Option A: Predator immediately transitions to RETURN (loses target).
- Option B: Predator continues moving toward last-known position, then transitions to RETURN after arriving.
- Option C: Predator switches to a new SEARCH state (patrol around last-known position).

Option A is simplest and sufficient for initial implementation.

**Visual Channel Allocation**:

Adding a bush channel would expand the visual sensor from 8 to 9 channels. This changes the observation vector size and would require retraining any existing models. Alternatively, bush could share channel 6 with other obstacles if visual distinction isn't critical for the agent.

**Bush Chemical Signature**:

A unique olfactory signature (e.g., `[0,0,0,1,0]` on channel 3) would let the agent smell bushes and navigate toward them as hiding spots. This is fully supported by the existing olfactory system.

### 7.5 Files That Will Need Changes

| File | Change Type | Description |
|------|------------|-------------|
| `configs/environment/environment.yaml` | Config | Add bush obstacle entries |
| `src/environment/core.py` | Logic | Modify `update_predators` to check agent-on-bush before detection |
| `src/environment/sensor.py` | Observation | Potentially add visual channel for bush |
| `src/environment/state.py` | Possibly | Add fields if needed (e.g., new visual channel count) |
| `src/environment/renderer.py` | Visual | Add bush color/icon support |
| `assets/` | Asset | Add bush icon |

### 7.6 Minimal Implementation Sketch

The simplest viable bush implementation:

1. **YAML Config**: Add bush as a non-blocking, zero-damage obstacle with a unique chemical signature.
2. **Detection Mask**: In `update_predators`, compute `agent_on_bush = jnp.any(jnp.logical_and(jnp.all(obs_pos == agent_pos, axis=-1), is_bush_mask))`. If true, set effective detection range to 0 for all predators.
3. **Visual Channel**: Expand to 9 channels, assign bush to channel 8.
4. **No state changes needed**: `agent_on_bush` can be computed from existing `obs_pos` + a `is_bush` mask derived from `obs_type` and `obstacle_names`.

---

## 8. Summary

The obstacle system is cleanly architected as a static-entity layer with per-instance configurability. The existing `obs_type`/`obstacle_names` infrastructure was clearly designed to support multiple obstacle types, making it straightforward to add a bush.

The primary engineering challenge is not in the obstacle itself, but in the **predator-obstacle interaction** (detection masking), which is a new mechanic that crosses the boundary between the obstacle system and the predator state machine. The visual sensor channel expansion is a secondary consideration that affects observation space dimensions.
