---
title: "Object Generation & Spawning System"
topic: refactors
status: archive
created: 2026-03-04
last_updated: 2026-04-12
---

# Object Generation & Spawning System

## 1. Overview

The grid world environment generates and manages five entity types across a 20x20 grid divided into four quadrants with distinct terrain:

```
        Col 1-10        Col 11-20
       ┌───────────────┬───────────────┐
Row    │  GRASS (TL)   │  SAND (TR)    │
1-10   │  High Risk    │  Low Risk     │
       │  High Reward  │  Sparse Food  │
       ├───────────────┼───────────────┤
Row    │  SAND (BL)    │  GRASS (BR)   │
11-20  │  Low Risk     │  High Risk    │
       │  Sparse Food  │  High Reward  │
       └───────────────┴───────────────┘
```

Entity summary:

| Entity            | Count | Quadrant(s) | Respawns | Moves |
|-------------------|-------|-------------|----------|-------|
| Food (Grass)      | 6     | TL, BR      | Yes      | No    |
| Food (Sand)       | 2     | TR, BL      | Yes      | No    |
| Danger            | 16    | TL, BR      | Yes      | No    |
| Predator          | 2     | TL, BR      | No       | Yes   |
| Rock              | 20    | TR, BL      | No       | No    |
| Tree              | 2     | Fixed       | No       | No    |
| Neutral (Rabbit)  | 5     | Everywhere  | No       | Yes   |

---

## 2. Architecture

All spawning and entity management is implemented in two files:

```
src/environment/core.py          ← jax_reset(), jax_step(), update_resources(), update_predators(), update_neutral_animals()
src/environment/config_loader.py ← YAML → EnvParams conversion (spawn area transforms, count expansion)
```

### PRNG Key Management

At reset, one master key is split into 4 independent streams:

```python
key, agent_key, place_key, body_key = jax.random.split(key, 4)
```

At each step, a fresh split produces 4 keys:

```python
key, respawn_key, predator_key, neutral_key, damage_key = jax.random.split(state.key, 5)
```

This ensures deterministic reproducibility while keeping each entity's randomization independent.

### Coordinate Convention

YAML config uses **1-based inclusive** coordinates: `[[r1, c1], [r2, c2]]`.
The config loader converts these to **0-based, exclusive-max**: `[r1-1, c1-1, r2, c2]` for use with `jax.random.randint`.

---

## 3. Collision-Aware Spawning at Reset

All entities are placed during `jax_reset()` through a **unified, sequential placement pipeline** that prevents any two entities from occupying the same cell. The system is built on three utility functions in `core.py`:

### 3.1 Placement Utilities

**`_sample_unoccupied(key, spawn_area, occupancy)`** — Samples a random position within `spawn_area`, retrying up to `SPAWN_MAX_RETRIES` (10) times if the cell is occupied. Accepts the first free cell found. Falls back to the last sample on exhaustion (graceful degradation).

**`_place_entities(key, spawn_areas, init_occupancy)`** — Places N entities sequentially via `jax.lax.scan`, threading a `[H, W]` boolean occupancy grid. Each entity calls `_sample_unoccupied`, then marks its position as occupied before the next entity is placed.

**`_respawn_resources(key, res_pos, spawn_areas, respawn_mask, init_occupancy)`** — Same sequential logic for mid-episode resource respawning. Resources flagged for respawn get new collision-free positions; others keep their current positions. Both cases mark the final position in the occupancy grid.

### 3.2 Unified Placement Pipeline

The agent is placed first and marked in the occupancy grid. Then ALL entity spawn areas are concatenated into a single `[N, 4]` array and processed through `_place_entities`:

```python
occupancy = jnp.zeros((params.height, params.width), dtype=jnp.bool_)
occupancy = occupancy.at[agent_pos[0], agent_pos[1]].set(True)

all_spawn_areas = jnp.concatenate([
    params.res_spawn_area,      # [num_res, 4]
    params.obs_spawn_area,      # [num_obs, 4]
    params.pred_patrol,         # [num_pred, 4]
    params.neutral_spawn_area,  # [num_neutral, 4]
], axis=0)

all_positions, _ = _place_entities(place_key, all_spawn_areas, occupancy)

# Split positions back (order must match concatenation)
res_pos = all_positions[:num_res]
obs_pos = all_positions[num_res:num_res + num_obs]
pred_pos = all_positions[num_res + num_obs:num_res + num_obs + num_pred]
neutral_pos = all_positions[num_res + num_obs + num_pred:]
```

**To add a new entity type**: append its spawn areas to `all_spawn_areas` and add a corresponding split line. The new entity automatically benefits from collision avoidance against all existing types.

### 3.3 Agent

```python
random_pos = jax.random.randint(agent_key, (2,), 0, jnp.array([params.height, params.width]))
agent_pos = jnp.where(params.random_start_pos, random_pos, params.start_pos)
```

- If `random_start_pos: true` — uniform random anywhere on the grid.
- If `random_start_pos: false` — uses `start_pos` from config (default: row 6, col 15 in the sand TR quadrant).

The agent is placed first and its position is marked as occupied before any other entity is placed.

### 3.4 Resources

Placed via the unified pipeline. Each resource instance gets a collision-free position within its `spawn_area`. All resources start as `active=True` with `cons_count=0` and `reg_timer=0`.

The `count` field in config is expanded at load time — e.g., `count: 3` produces 3 independent resource entries sharing the same template but each receiving a separate position.

### 3.5 Predators

Placed via the unified pipeline using their `patrol_area` as the spawn area. Initial state:
- `pred_state = 0` (PATROL)
- `pred_stamina = pred_max_stamina`
- `pred_move_timer = 0`
- `pred_attack_timer = 0`

### 3.6 Obstacles

Placed via the unified pipeline. They are **intra-episode static** — once placed, they remain at their positions for the entire episode and are reshuffled at the next reset.

Two types exist:
- **Rocks** (`blocking: false`): Agent can walk onto them but takes damage.
- **Trees** (`blocking: true`): Agent cannot enter the cell. Collision triggers bump damage.

### 3.7 Neutral Animals

Placed via the unified pipeline within their `spawn_area`. Initial `move_timer = 0`.

### 3.6 Body State

```python
body_key1, body_key2, body_key3 = jax.random.split(body_key, 3)
```

| State      | Random Start         | Default     |
|------------|----------------------|-------------|
| Nutrition  | Uniform [max/2, max] | 100.0       |
| Satiation  | Derived from nutrition via scaling factor | 100.0 |
| Injury     | Uniform [0, max/2]   | 0.0         |

---

## 4. Resource Lifecycle

Resources follow a consume-deactivate-regenerate-respawn cycle:

```
  ACTIVE ──[consumed max_cons times]──▶ INACTIVE
     ▲                                      │
     │                               reg_timer = reg_delay
     │                                      │
     │                            [timer counts down each step]
     │                                      │
  RESPAWN ◀──[timer reaches 0]──────────────┘
  (new random position within spawn_area)
```

### 4.1 Consumption

When the agent overlaps an active resource (in `jax_step`, lines 317-368):

- **Danger resources**: Interaction is automatic on overlap. Damage is sampled uniformly from `[damage_min, damage_max]`. `cons_count` increments.
- **Food resources**: Interaction depends on config:
  - If `eat_action_enabled: false` — auto-eat on overlap.
  - If `eat_action_enabled: true` — requires the agent to use the Eat action while on the food cell.

### 4.2 Deactivation

```python
should_deactivate = jnp.logical_and(
    new_active,
    jnp.logical_and(params.res_max_cons > 0, next_cons_count >= params.res_max_cons)
)
final_active = jnp.where(should_deactivate, False, new_active)
next_reg_timer = jnp.where(should_deactivate, params.res_reg_delay, new_reg_timer)
```

- If `max_consumption > 0` and `cons_count >= max_consumption`, the resource deactivates.
- If `max_consumption == -1` (danger), the resource never depletes.
- On deactivation, `reg_timer` is set to `regeneration_delay`.

### 4.3 Regeneration

Each step, `update_resources()` runs before the agent moves:

```python
needs_reg_update = jnp.logical_and(jnp.logical_not(res_active), res_reg_timer > 0)
new_reg_timer = jnp.where(needs_reg_update, res_reg_timer - 1, res_reg_timer)
respawn_mask = jnp.logical_and(jnp.logical_not(res_active), new_reg_timer <= 0)
new_active = jnp.where(respawn_mask, True, res_active)
new_cons_count = jnp.where(respawn_mask, 0, res_cons_count)
```

When the timer reaches 0, the resource respawns at a **new collision-free position** within its `spawn_area`. An occupancy grid is built from the current state (agent, obstacles, predators, neutral animals, and active non-respawning resources), and `_respawn_resources` places each respawning resource sequentially:

```python
res_pos_after_reg = _respawn_resources(
    respawn_key, state.res_pos, params.res_spawn_area, respawn_mask, new_active, respawn_occupancy
)
```

The `active_mask` (`new_active`) is critical for correctness. Resources have three possible states during respawn:

| State | `new_active` | `respawn_mask` | Occupancy |
|---|---|---|---|
| Active (on grid) | True | False | Marks position |
| Ghost (consumed, timer counting) | False | False | Does NOT mark |
| Respawning (timer hit 0) | True | True | Marks new position |

**Ghost resources** (consumed but waiting to regenerate) do not physically exist on the grid. Their old positions are intentionally left unmarked so other respawning resources can use those cells.

Multiple resources respawning on the same step are processed sequentially, so they also avoid colliding with each other.

---

## 5. Predator State Machine

Predators are managed by `update_predators()` (lines 127-234) and follow a three-state machine:

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
  ┌──────────┐  detect agent   ┌──────────┐   lose interest   ┌──────────┐
  │  PATROL  │ ──────────────▶ │   HUNT   │ ────────────────▶ │  RETURN  │
  │   (0)    │   & stamina ok  │   (1)    │  or stamina = 0   │   (2)    │
  └──────────┘                 └──────────┘                   └──────────┘
       ▲                                                           │
       │                                                           │
       └────────── reach patrol center (dist <= 2) ────────────────┘
```

### State Behaviors

| State   | Movement                 | Stamina       | Bounds            |
|---------|--------------------------|---------------|-------------------|
| PATROL  | Random jitter (-1,0,+1)  | Recovers      | Clipped to patrol area |
| HUNT    | Moves toward agent       | Drains (-1/step) | Clipped to patrol area |
| RETURN  | Moves toward patrol center | Recovers    | Clipped to patrol area |

### Transition Conditions

- **PATROL -> HUNT**: Agent within `detection_range` AND `stamina >= max_stamina * hunt_threshold`
- **HUNT -> RETURN**: Agent beyond `detection_range * 2` OR `stamina <= 0`
- **RETURN -> PATROL**: Manhattan distance to patrol center <= 2

### Movement Mechanics

- Movement is tick-based: predators only move when `move_timer <= 0` AND `attack_timer <= 0`.
- Diagonal resolution: when both row and column deltas are non-zero, one axis is chosen at random (50/50).
- Obstacle avoidance: if new position overlaps a blocking obstacle, the predator stays in place.
- Spatial bounds: positions are always clipped to the predator's `patrol_area`, then to grid boundaries.

### Attack Mechanics

When a predator overlaps the agent:
- Damage is sampled uniformly from `[damage_min, damage_max]`.
- The predator's `attack_timer` is set to `attack_delay`, preventing movement for that many steps.

---

## 6. Neutral Animal Movement

Neutral animals (`update_neutral_animals()`, lines 236-269) are simple patrol entities:

- Move by random jitter (-1, 0, +1) in each axis.
- Movement gated by `move_timer` (reset to `move_interval` after each move).
- Positions clipped to `patrol_area`, then grid boundaries.
- Avoid blocking obstacles (same collision check as predators).
- They serve as **olfactory decoys** — they emit chemical signatures (`property: [0.0, 0.3, 0.0, 0.0, 0.0]`) that the agent's olfaction sensor detects, creating sensory noise.

---

## 7. Obstacle Interactions

Obstacles have two interaction modes depending on `blocking`:

### Non-blocking (Rocks)

The agent enters the cell. Damage is applied on overlap:

```python
at_obs = jnp.all(state.obs_pos == new_agent_pos, axis=-1)
damage_obs_overlap = jnp.sum(jnp.where(
    jnp.logical_and(at_obs, jnp.logical_not(params.obs_blocking)),
    sampled_obs_damage, 0.0
))
```

### Blocking (Trees)

The agent is prevented from entering. Bump damage is applied to the agent at its current position:

```python
at_attempted_obs = jnp.all(state.obs_pos == attempted_pos, axis=-1)
damage_obs_collision = jnp.where(
    just_collided,
    jnp.max(jnp.where(at_attempted_obs, sampled_obs_damage, 0.0)),
    0.0
)
```

Both types also emit nociception signals that the agent's extero nociception sensor can detect.

---

## 8. Current Configuration

### Resources

| Name              | Type   | Count | Spawn Area        | Max Cons | Regen Delay | Damage       | Properties              |
|-------------------|--------|-------|-------------------|----------|-------------|--------------|-------------------------|
| Food Grass TL     | food   | 3     | [1,1]-[10,10]     | 35       | 100 steps   | 0            | [1,0,0,0,0]            |
| Food Grass BR     | food   | 3     | [11,11]-[20,20]   | 35       | 100 steps   | 0            | [1,0,0,0,0]            |
| Food Sand TR      | food   | 1     | [1,11]-[10,20]    | 20       | 250 steps   | 0            | [1,0,0,0,0]            |
| Food Sand BL      | food   | 1     | [11,1]-[20,10]    | 20       | 250 steps   | 0            | [1,0,0,0,0]            |
| Danger Grass TL   | danger | 8     | [1,1]-[10,10]     | -1 (inf) | 20 steps    | [5.0, 15.0]  | [0,0,0,0,0]            |
| Danger Grass BR   | danger | 8     | [11,11]-[20,20]   | -1 (inf) | 20 steps    | [5.0, 15.0]  | [0,0,0,0,0]            |

### Predators

| Name               | Count | Patrol Area       | Detect | Stamina | Recovery | Hunt Thresh | Attack Delay | Damage       |
|--------------------|-------|-------------------|--------|---------|----------|-------------|--------------|--------------|
| Forest Predator TL | 1     | [1,1]-[9,9]       | 5      | 30      | 1/step   | 0.5         | 3 steps      | [5.0, 15.0]  |
| Forest Predator BR | 1     | [11,11]-[20,20]   | 5      | 30      | 1/step   | 0.5         | 3 steps      | [5.0, 15.0]  |

### Obstacles

| Name | Count | Area              | Blocking | Damage       | Nociception | Properties     |
|------|-------|-------------------|----------|--------------|-------------|----------------|
| Rock | 10    | [1,11]-[10,20]    | No       | [0.1, 0.5]   | 0.3         | [0,0,0,0,0]   |
| Rock | 10    | [11,1]-[20,10]    | No       | [0.1, 0.5]   | 0.3         | [0,0,0,0,0]   |
| Tree | 1     | [5,15] (fixed)    | Yes      | [0.1, 0.5]   | 0.1         | [0,0,0,0,1]   |
| Tree | 1     | [15,5] (fixed)    | Yes      | [0.1, 0.5]   | 0.1         | [0,0,0,0,1]   |

### Neutral Animals

| Name         | Count | Spawn Area      | Patrol Area     | Move Interval | Nociception | Properties        |
|--------------|-------|-----------------|-----------------|---------------|-------------|-------------------|
| Decoy Rabbit | 5     | [1,1]-[20,20]   | [1,1]-[20,20]   | 1 step        | 0.1         | [0,0.3,0,0,0]    |

---

## 9. Design Rationale

### Quadrant Risk/Reward Structure

The environment creates a deliberate trade-off between **risk** and **reward**:

- **Grass quadrants** (TL, BR): Dense food (3 per quadrant) but also dense danger (8 per quadrant) and an active predator. High-risk, high-reward foraging.
- **Sand quadrants** (TR, BL): Sparse food (1 per quadrant) with longer regeneration (250 vs 100 steps), but only non-blocking rocks as hazards. Low-risk, low-reward.

This pressures the agent to manage the trade-off between:
1. Entering dangerous grass to eat and survive (risk injury).
2. Staying in safe sand but risking starvation from low food density.

### Chemical Properties (5-Dimensional Olfaction Vector)

Entities emit chemical signatures that the olfaction sensor detects as spatial gradients:

| Entity       | Properties              | Interpretation                        |
|--------------|-------------------------|---------------------------------------|
| Food         | [1, 0, 0, 0, 0]        | Strong channel-0 signal               |
| Danger       | [0, 0, 0, 0, 0]        | No chemical signature (invisible to olfaction) |
| Predator     | [0, 1, 0, 0, 0]        | Strong channel-1 signal               |
| Neutral      | [0, 0.3, 0, 0, 0]      | Weak channel-1 signal (mimics predator) |
| Rock         | [0, 0, 0, 0, 0]        | No chemical signature                 |
| Tree         | [0, 0, 0, 0, 1]        | Unique channel-4 signal               |

Neutral animals emit a weaker version of the predator's chemical signature (channel 1 at 0.3 vs 1.0), acting as **olfactory decoys** that make predator localization ambiguous.

---

## 10. Known Limitations

1. **Bounded retry collision avoidance** — Spawn collision avoidance uses a bounded retry loop (`SPAWN_MAX_RETRIES = 10`). On a densely packed grid or very small spawn areas, all retries may fail and entities gracefully degrade to overlapping. For the current 20x20 grid with ~50 entities, the probability of failure is negligible. Entities with single-cell spawn areas (e.g., the fixed-position trees) cannot avoid a collision if that cell is already taken.