# 08 — Resources & Obstacles

> **Source**: `src/environment/core.py` (`update_resources`, interaction section), `src/environment/config_loader.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

The environment contains four categories of interactable objects beyond the agent:

- **Resources** (food/danger): consumable entities with regeneration timers.
- **Obstacles** (rocks, bushes, trees): static blocking or passthrough terrain features.
- **Neutral animals**: mobile non-hostile entities that wander the grid.
- **Grid location types**: tile-level terrain annotation (plain, grass, sand).

Resources and obstacles are placed at reset and interact with the agent in Stage 4 of `jax_step`. Neutral animals are mobile entities updated each step in Stage 3.5.

---

## Resource Types

`res_type [N]` int32 — 0 = food, 1 = danger.

**Food** (type 0):
- Contact provides a nutrition gain of `food_nutrition_gain - eating_nutrition_cost`.
- Olfactory signature in `res_property`: typically `[1,0,0,0,0]` (hot in channel 0).
- No damage; `res_nociception = 0.0`.

**Danger** (type 1):
- Contact deals damage sampled from `Uniform(res_damage[n,0], res_damage[n,1])`.
- `res_nociception` (default 0.9): the intensity forwarded to the nociception sensor on contact.
- No nutrition effect.
- Olfactory signature: typically `[0,0,0,0,0]` (no chemical signature) — agent must infer danger from other signals.

Both types use the same lifecycle (consumption counter, timer, activity flag). The type only determines the interaction outcome.

---

## Consumption Mechanics

**Auto-eat mode** (`eat_action_enabled=False`): any step where `agent_pos == res_pos[n] AND res_active[n]` triggers consumption. No explicit action needed.

**Eat-action mode** (`eat_action_enabled=True`): food consumption only occurs if the agent is on a food resource AND selects the eat action (action 5 if rest is enabled, action 4 if rest is disabled). Danger resources always trigger automatically regardless of this setting.

**Lifecycle tracking** (`core.py:372`):
```
next_cons_count = cons_count + 1  (if interacted)
should_deactivate = res_active AND res_max_cons > 0 AND next_cons_count >= res_max_cons
final_active = False if should_deactivate else res_active
next_reg_timer = res_reg_delay if should_deactivate else reg_timer
```

Setting `res_max_cons = -1` (or any value ≤ 0) makes the resource permanently available — it never deactivates. The default is `35` in `configs/environment/default.yaml`.

---

## Regeneration

Handled in Stage 1 of `jax_step` (`core.py:117`, called at `core.py:291`):

```
For each inactive resource:
  new_reg_timer = reg_timer - 1
  if new_reg_timer <= 0:
    new_active = True
    new_cons_count = 0
    new_res_pos = randint(spawn_area)   # sample new position in spawn area
```

Resources respawn within their `res_spawn_area` bounding box. The new position is random — it will not necessarily be the same as the original position. No overlap checking is done on respawn positions, so two resources could theoretically land on the same cell.

---

## Damage & Nociception from Resources

**Damage** (`core.py:338`): sampled independently per resource using the step's `damage_key`:
```python
sampled_res_damage = Uniform(res_damage[:, 0], res_damage[:, 1])
damage_res = sum(sampled_res_damage where (interact AND is_danger))
```

**Nociception** (`sensor.py:59`): the nociception sensor checks for active danger resources at the agent's exact position (`dist < 0.1`), returning the maximum `res_nociception` intensity among all overlapping dangers. This is a separate signal from the damage value — damage feeds the body, nociception feeds the sensor.

---

## Obstacles

Obstacles are **static** — their positions are fixed after `jax_reset` and never change during an episode.

| Field | Effect |
|-------|--------|
| `obs_blocking=True` | Agent cannot enter this cell; bounces back; `just_collided=True`; collision damage applied |
| `obs_blocking=False` | Agent can stand on the cell; overlap damage applied; nociception from overlap |
| `obs_hides_agent=True` | Agent on this cell is hidden from predators |
| `obs_hides_agent=False` | No concealment effect |

**Damage sources**:
- **Collision** (blocking): `damage_obs_collision = max(obs_damage[n] where at attempted_pos AND obs_blocking[n])` — only the specific obstacle being bumped deals damage.
- **Overlap** (non-blocking): `damage_obs_overlap = sum(obs_damage[n] where at agent_pos AND NOT obs_blocking[n])`.

**Nociception storage**: `last_collision_noc` in `EnvState` stores the nociception intensity from the most recent blocking collision. This value is consumed by `sense_extero_nociception` in the next observation. It is reset to 0 when no collision occurs.

**Olfactory signature**: `obs_property [O, 5]` — obstacles can contribute to the olfactory signal. Bush obstacles in the default config use channel 3 (`[0,0,0,1,0]`).

**Visual encoding**: `obs_type [O]` indexes into `obstacle_names` (e.g. `("bush", "rock")`). The visual sensor maps type to channel 6 (rock) regardless of obstacle name — the `obs_type` field is used by the renderer for icon selection.

---

## Neutral Animals

Neutral animals wander randomly and do not attack. They serve as **olfactory decoys** — they contribute to the olfactory sensor signal via `neutral_property`, making the olfactory scene noisier and harder to decode.

**Update** (`core.py:249`):
1. Decrement `neutral_move_timer`.
2. When timer expires, move by random jitter `(dr, dc)` each in `{-1, 0, 1}`.
3. Clamp to `neutral_patrol` bounding box.
4. Obstacle collision check via `jax.vmap`.
5. Reset timer to `neutral_move_int`.

**Olfactory contribution** (`sensor.py:269`): included in the olfactory sensor:
```python
neutral_chem = sense_resource(agent_pos, neutral_pos, ones, neutral_property, radius, decay)
obs_olfactory = res_chem + pred_chem + obs_chem + neutral_chem
```

**Nociception**: `neutral_nociception [M]` — if a neutral animal is at the agent's position, its intensity contributes to the nociception sensor. Default is 0.1 in the default config.

---

## Grid Location Types

`grid_location_type [H, W]` int32 — tiles are categorised at reset from `environment.location_areas` in the YAML.

| Value | Name | YAML type | Renderer color |
|-------|------|-----------|----------------|
| 0 | Plain | `"plain"` (default) | White `#FFFFFF` |
| 1 | Grass | `"grass"` | Light green `#ECFDF5` |
| 2 | Sand | `"sand"` | Light amber `#FFFBEB` |

**Visual sensor** (`sensor.py:143`): tile type is one-hot encoded into the first 3 channels of the 8-channel visual observation:
- channel 0 = grass (loc 1)
- channel 1 = sand (loc 2)
- channel 2 = plain (loc 0)

**Predator concealment**: predators only check `obs_hides_agent` per-obstacle, not grid location type. Bush concealment is an obstacle property, not a tile property.
