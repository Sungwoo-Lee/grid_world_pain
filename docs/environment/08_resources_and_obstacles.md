# 08 — Resources & Obstacles

> **Source**: `src/environment/core.py` (`update_resources`, `jax_step`), `src/environment/state.py`, `src/environment/config_loader.py`, `src/environment/sensor.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

This document covers three categories of non-agent interactable objects: **resources** (food and traps), **obstacles** (static terrain features), and **neutral animals** (mobile non-hostile creatures). A fourth topic, grid tile types (plain, grass, sand), is covered at the end.

**Plain-language summary of what these things do:**

- A **food resource** is a consumable item the agent can eat to restore nutrition. After being eaten a set number of times, it disappears and reappears elsewhere after a timer expires.
- A **hiding-predator resource** is a trap: it looks like a resource but deals damage and triggers the pain sensor when stepped on. It never requires an action to activate — the agent just has to step on it.
- An **obstacle** is a static object fixed at reset. A blocking obstacle (like a rock) physically stops the agent from entering its cell; a non-blocking obstacle (like a bush) the agent can walk onto, potentially taking damage each step. Blocking obstacles can also hide the agent from predators (`obs_hides_agent`).
- A **neutral animal** is the `neutral`-class member of the unified animal entity system (introduced in the env-entities CP1 refactor). Neutrals wander or stand still, contribute chemical ("olfactory") smell signals, and can optionally emit a small nociceptive signal on contact — but they do not deal damage and do not cause predators to enter hunt mode. The old `neutral_*` and `pred_*` separate arrays no longer exist; everything lives in the `animal_*` arrays described in doc [07_predator_ai](07_predator_ai.md).

Resources and obstacles are placed at `jax_reset` and interact with the agent in Stage 4 of `jax_step`. Animals (including neutrals) are updated in Stage 3 of `jax_step` via `update_animals`.

---

## Resource Types

`res_type [num_res]` int32 — two codes:

| Code | Name | YAML `type:` string |
|------|------|---------------------|
| 0 | food | `"food"` |
| 1 | hiding_predator | `"hiding_predator"` (formerly `"danger"` — deprecated, still accepted with a warning) |

**Food** (type 0):
- Contact provides a net nutrition gain of `food_nutrition_gain - eating_nutrition_cost`.
- Olfactory signature via `res_property`; convention in shipped configs is `[1,0,0,0,0]` (hot in channel 0).
- Default `res_nociception = 0.0` (no pain signal). Set nonzero to model aversive food.
- Consumption can be action-gated (see Consumption Mechanics below).

**Hiding Predator** (type 1):
- Contact deals damage sampled from `Uniform(res_damage[n,0], res_damage[n,1])`.
- `res_nociception` (default 0.9): intensity forwarded to the exteroceptive nociception sensor on contact.
- No nutrition effect.
- Olfactory signature: typically `[0,0,0,0,0]` — no chemical signature, so the agent must infer the trap from other signals (nociception, visual).
- Always triggers automatically regardless of `eat_action_enabled`.

Both types share the same lifecycle (consumption counter, regen timer, active flag). The type code only determines the interaction outcome.

**Config loading** (`config_loader.py:638-652`): `res_nociception` defaults to 0.9 for hiding-predator resources and 0.0 for food if not explicitly set. The deprecated `"danger"` type string is remapped to 1 with a `DeprecationWarning`.

---

## Resource State and Parameter Fields

| Field | Location | Shape | Description |
|-------|----------|-------|-------------|
| `res_pos` | `EnvState` | `[num_res, 2]` | Current position (row, col) |
| `res_active` | `EnvState` | `[num_res]` bool | Whether this resource can be interacted with |
| `res_cons_count` | `EnvState` | `[num_res]` int | Consumption count since last respawn |
| `res_reg_timer` | `EnvState` | `[num_res]` int | Steps remaining before respawn (counts down) |
| `res_property_sampled` | `EnvState` | `[num_res, V]` | Olfactory signature for this episode / since last respawn |
| `res_type` | `EnvParams` | `[num_res]` int32 | 0=food, 1=hiding_predator |
| `res_property` | `EnvParams` | `[num_res, V]` | Mean olfactory signature (config constant) |
| `res_property_std` | `EnvParams` | `[num_res, V]` | Std for per-reset Gaussian sampling |
| `res_nociception` | `EnvParams` | `[num_res]` | Nociceptive intensity on contact |
| `res_spawn_area` | `EnvParams` | `[num_res, 4]` | `[min_r, min_c, max_r, max_c]` bounding box for placement/respawn |
| `res_max_cons` | `EnvParams` | `[num_res]` int | Max consumptions before deactivation (<=0 = infinite) |
| `res_reg_delay` | `EnvParams` | `[num_res]` int | Steps from deactivation to respawn |
| `res_damage` | `EnvParams` | `[num_res, 2]` | `[min, max]` damage range per contact |

---

## Consumption Mechanics

**Auto-eat mode** (`eat_action_enabled=False`): any step where `agent_pos == res_pos[n] AND res_active[n]` triggers consumption. No explicit action needed.

**Eat-action mode** (`eat_action_enabled=True`): food consumption only occurs if the agent is on a food resource AND selects the eat action (action index 5 if `rest_action_enabled=True`, index 4 if `rest_action_enabled=False`). Hiding predators always trigger automatically regardless of this flag (`core.py:421-449`).

**Lifecycle tracking** (`core.py:458-465`):
```
next_cons_count = new_cons_count + (1 if interacted_this_step else 0)
should_deactivate = new_active AND (res_max_cons > 0) AND (next_cons_count >= res_max_cons)
final_active = False if should_deactivate else new_active
next_reg_timer = res_reg_delay if should_deactivate else new_reg_timer
```

Setting `res_max_cons <= 0` (e.g. `-1` or `0`) makes the resource permanently available — it never deactivates. The default in `configs/environment/default.yaml` is 35.

**`interacted_this_step`** is the union of:
1. Hiding-predator interaction: `interact_resource AND is_hiding_predator` (always auto).
2. Food lifecycle trigger: `interact_resource AND is_food AND (auto-eat OR eat-action)`.

Note: `interacted_this_step` governs the consumption counter and deactivation. The separate `ate_food` flag governs only the nutrition gain (`core.py:441-444`).

---

## Regeneration

Handled in Stage 1 of `jax_step` (`core.py:374-395`, calling `update_resources` at `core.py:375`).

`update_resources` is a pure function that advances the regen timer for every resource in parallel, using only [branchless masking](00_jax_primer.md#branchless-control-flow-jnpwhere-jaxlaxselect-jaxlaxcond) — no Python conditionals, no loop over resources.

**What it does, in plain English:**
1. Countdown: subtract 1 from the timer of every resource that is both inactive *and* has timer > 0.
2. Respawn-ready: any resource that was inactive *and* whose new timer is ≤ 0 is flagged in `respawn_mask`.
3. For flagged resources: flip `res_active` back to `True` and reset `res_cons_count` to 0.
4. Return the updated arrays plus `respawn_mask` — the caller (`jax_step`) uses the mask to sample new positions and re-draw chemical properties only for those resources.

No positions or chemical properties are touched inside `update_resources` itself.

```python
# Source: src/environment/core.py:119–130
def update_resources(res_active, res_reg_timer, res_cons_count, params):
    """Updates resource timers and regeneration."""
    # Regeneration
    needs_reg_update = jnp.logical_and(jnp.logical_not(res_active), res_reg_timer > 0)
    new_reg_timer = jnp.where(needs_reg_update, res_reg_timer - 1, res_reg_timer)
    
    # Respawn where timer hits 0
    respawn_mask = jnp.logical_and(jnp.logical_not(res_active), new_reg_timer <= 0)
    new_active = jnp.where(respawn_mask, True, res_active)
    new_cons_count = jnp.where(respawn_mask, 0, res_cons_count)
    
    return new_active, new_reg_timer, new_cons_count, respawn_mask
```

> **API notes — `update_resources`**
> - `jnp.logical_not` / `jnp.logical_and` — [primer: branchless](00_jax_primer.md#branchless-control-flow-jnpwhere-jaxlaxselect-jaxlaxcond). Elementwise boolean ops over `[num_res]` bool arrays; no Python `if`.
> - `jnp.where(mask, true_val, false_val)` — [primer: branchless](00_jax_primer.md#branchless-control-flow-jnpwhere-jaxlaxselect-jaxlaxcond). Selects element-by-element; both branches are always evaluated but only selected values write through. "If inactive AND timer > 0, decrement" is expressed entirely this way — no `for` loop, no Python `if`.
> - The intermediate `new_reg_timer` is computed first and then immediately consumed by `respawn_mask` — two sequential `jnp.where` calls in pure functional style; no array is mutated. See [primer: immutability](00_jax_primer.md#immutable-arrays-and-functional-updates-atidxset--add).
> - Scalar `True` and `0` as `true_val` arguments to `jnp.where` are broadcast to match the `[num_res]` shape — standard JAX broadcasting.

### Respawn position and property sampling (`jax_step` Stage 1 — cross-reference)

`update_resources` returns `respawn_mask` but does not move resources or re-draw their chemical properties. Immediately after the call in `jax_step`, the caller does both:

```python
# Source: src/environment/core.py:379–395  (excerpt from jax_step, Stage 1)
    # Displace resources that just respawned
    num_res = params.res_type.shape[0]
    res_keys = jax.random.split(respawn_key, num_res)

    def sample_res_pos(rk, area):
        return jax.random.randint(rk, (2,), area[:2], area[2:])

    new_potential_pos = jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)
    # Only update position IF respawn_mask is true for that resource
    res_pos_after_reg = jnp.where(respawn_mask[:, None], new_potential_pos, state.res_pos)

    # Re-sample chemical property for respawned resources
    noise = jax.random.normal(property_key, shape=params.res_property.shape)
    new_sampled_prop = jnp.clip(params.res_property + params.res_property_std * noise, 0.0, 1.0)
    res_property_sampled_after_reg = jnp.where(
        respawn_mask[:, None], new_sampled_prop, state.res_property_sampled
    )
```

> **API notes — respawn position + property**
> - `jax.random.split(respawn_key, num_res)` — [primer: prng](00_jax_primer.md#functional-prng-split-fold_in-key-threading). Produces `num_res` independent keys from one parent key so each resource gets an independent draw.
> - `jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)` — [primer: vmap](00_jax_primer.md#jaxvmap-and-in_axes). `sample_res_pos` is written for a single resource (`rk` is one key, `area` is `[4]`). `vmap` lifts it to all `num_res` resources simultaneously — both arguments batched on axis 0. No Python `for` loop.
> - `respawn_mask[:, None]` — inserts a trailing axis to broadcast `[num_res]` → `[num_res, 1]`, matching the `[num_res, 2]` position array in `jnp.where`. The shape is statically known so this is valid inside a traced/JIT context.
> - `jnp.clip(..., 0.0, 1.0)` — [primer: masking](00_jax_primer.md#fixed-shape-masking-idioms). Branchless elementwise clamp; guards the Gaussian draw without a conditional.
> - Consumption / deactivation logic that sets `res_active=False` and starts the timer lives in `jax_step` Stage 4 (`core.py:421–465`). Doc [04_jax_step](04_jax_step.md) owns that body.
> - Reset-time property sampling (`jax_reset`, `core.py:934`) follows the same `clip(mean + std * N(0,1), 0, 1)` formula. Doc [03_jax_reset](03_jax_reset.md) owns that body.

After `update_resources` returns, `jax_step` handles two additional respawn effects:

1. **New position** (`core.py:381-388`): for each respawning resource, a new `(row, col)` is sampled uniformly from its `res_spawn_area`. No occupancy check — two resources can land on the same cell.

2. **Re-sampled chemical property** (`core.py:391-395`): `new_sampled = clip(res_property + res_property_std * N(0,1), 0, 1)`. The respawned resource looks chemically "fresh" with new noise if `res_property_std > 0`.

**Timer behaviour in detail:**
- At deactivation: `next_reg_timer = res_reg_delay` (set at `core.py:465`).
- Each subsequent step: `new_reg_timer = reg_timer - 1` (decremented in `update_resources`).
- Respawn fires the first step that `new_reg_timer <= 0`, i.e. exactly `res_reg_delay` steps after deactivation.
- The timer is NOT decremented while the resource is active (`needs_reg_update` requires `NOT active`).
- At reset: `res_reg_timer` is initialized to 0 and `res_active` to all-True, so all resources start available (`core.py:984-986`).

---

## Per-Resource Property Sampling

`res_property` (`EnvParams`) is the configured mean — constant across all episodes. `res_property_sampled` (`EnvState`) is a noisy draw made at two points:

1. **At reset** (`core.py:934`, `jax_reset`): `clip(res_property + res_property_std * N(0,1), 0, 1)`.
2. **On respawn** (`core.py:391-395`): same formula, applied only to resources flagged by `respawn_mask`.

Sensors read `res_property_sampled`. If `res_property_std = 0` for all channels, the sampled value equals the mean exactly and is constant.

---

## Damage & Nociception from Resources

**Damage** (`core.py:424-427`): sampled independently per resource using `damage_key`:
```
sampled_res_damage = Uniform(res_damage[:, 0], res_damage[:, 1])
damage_res = sum(sampled_res_damage  where (interact_resource AND is_hiding_predator))
```
This is a single `jax.random.uniform` draw over all resources, so draw positions are stable. The result feeds `total_damage`.

**Exteroceptive nociception** (`sensor.py:59-95`): `sense_extero_nociception` checks:
1. Active resources at the agent's cell (`dist < 0.1`): max of `res_nociception` among overlapping resources.
2. Damaging animals at the agent's cell: max of `animal_nociception` (only `animal_is_damaging` entries contribute, `sensor.py:74-80`).
3. Non-blocking obstacles at the agent's cell: max of `obs_nociception`.
4. Blocking obstacle collision: `state.last_collision_noc` (stored from the step).

The final nociceptive signal is the maximum across all four sources — a single scalar.

---

## Obstacles

Obstacles are **static** — positions (`obs_pos` in `EnvState`) are fixed after `jax_reset` and never change during an episode.

| Field | Location | Shape | Description |
|-------|----------|-------|-------------|
| `obs_pos` | `EnvState` | `[num_obs, 2]` | Fixed position (row, col) |
| `obs_property_sampled` | `EnvState` | `[num_obs, V]` | Olfactory signature (sampled at reset) |
| `obs_blocking` | `EnvParams` | `[num_obs]` bool | True: agent cannot enter this cell |
| `obs_hides_agent` | `EnvParams` | `[num_obs]` bool | True: agent on this cell is hidden from predators |
| `obs_spawn_area` | `EnvParams` | `[num_obs, 4]` | Placement bounding box |
| `obs_damage` | `EnvParams` | `[num_obs, 2]` | `[min, max]` damage range |
| `obs_property` | `EnvParams` | `[num_obs, V]` | Mean olfactory signature |
| `obs_property_std` | `EnvParams` | `[num_obs, V]` | Std for per-reset sampling |
| `obs_nociception` | `EnvParams` | `[num_obs]` | Nociceptive intensity on overlap/collision (default 0.3) |
| `obs_type` | `EnvParams` | `[num_obs]` int32 | Index into `obstacle_names` for renderer icon |
| `obstacle_names` | `EnvParams` | `tuple[str]` | Sorted unique set of obstacle name strings |

**Config defaults** (`config_loader.py:704-711`):
- `blocking`: defaults to `True` if not specified.
- `hides_agent`: defaults to `False` if not specified.
- `damage`: defaults to `0.0` if not specified (converted to `[0.0, 0.0]`).
- `nociception_intensity`: defaults to `0.3` if not specified.

### Blocking behaviour

`move_agent` resolves all movement for the agent: it converts the action integer to a `(dr, dc)` delta, clamps the proposed position to grid boundaries, and then checks blocking obstacles — all branchlessly.

```python
# Source: src/environment/core.py:5–36
def move_agent(pos: jnp.ndarray, action: int, obs_pos: jnp.ndarray, obs_blocking: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    """Calculates New Agent position based on action, considering obstacles."""
    # 0: Up, 1: Right, 2: Down, 3: Left, 4+: Stay
    moves = jnp.array([
        [-1, 0], # Up
        [0, 1],  # Right
        [1, 0],  # Down
        [0, -1], # Left
        [0, 0],  # Rest/Stay
        [0, 0],  # Eat/Stay
    ], dtype=jnp.int32)
    
    # Clip action to valid range [0, 5]
    action = jnp.clip(action, 0, 5).astype(jnp.int32)
    move = moves[action]
    
    new_pos = pos + move
    # Clamp to grid boundaries
    new_pos = jnp.array([
        jnp.clip(new_pos[0], 0, params.height - 1),
        jnp.clip(new_pos[1], 0, params.width - 1)
    ])
    
    # Obstacle collision check
    is_collision = jnp.any(jnp.logical_and(
        jnp.all(obs_pos == new_pos, axis=-1),
        obs_blocking
    ))
    
    # If collision, stay at current position
    final_pos = jnp.where(is_collision, pos, new_pos)
    return final_pos, is_collision
```

> **API notes — `move_agent`**
> - `moves[action]` — static-shape index into a `[6, 2]` constant array; `action` is a traced integer, but the array shape is fixed at compile time, so JAX can trace this as a gather. See [primer: static-dynamic](00_jax_primer.md#static-vs-dynamic-fields-and-jit-recompilation).
> - `jnp.clip(action, 0, 5)` — [primer: masking](00_jax_primer.md#fixed-shape-masking-idioms). Branchless clamp so out-of-range actions fall back to the `[0,0]` stay move without a Python `if`.
> - `jnp.all(..., axis=-1)` over `obs_pos == new_pos` — checks both row and column match simultaneously for each obstacle. Returns `[num_obs]` bool; combined with `obs_blocking` via `logical_and` then collapsed to a scalar with `jnp.any`. See [primer: masking](00_jax_primer.md#fixed-shape-masking-idioms).
> - `jnp.where(is_collision, pos, new_pos)` — [primer: branchless](00_jax_primer.md#branchless-control-flow-jnpwhere-jaxlaxselect-jaxlaxcond). Both `pos` (stay) and `new_pos` (move) are computed; only the selected one is returned. This is the canonical JAX substitute for `if is_collision: return pos else: return new_pos`.
> - Called from `jax_step` Stage 2 (`core.py:398`): `new_agent_pos, just_collided = move_agent(state.agent_pos, action, state.obs_pos, params.obs_blocking, params)`. `obs_pos` is read from **state** (current episode positions), while `obs_blocking` is a static parameter from `params`. Doc [04_jax_step](04_jax_step.md) owns the full stage-by-stage call sequence.

If blocked, the agent stays at `pos`, `just_collided = True`, and `last_collision_noc` is written.

| `obs_blocking` | `obs_hides_agent` | Effect |
|----------------|-------------------|--------|
| `True` | `False` | Agent bounces back; `just_collided=True`; collision damage applied |
| `True` | `True` | Bounce-back AND agent hidden from predators while adjacent (same-cell check inside `_hunt_step`) |
| `False` | `False` | Agent can stand on cell; overlap damage applied each step |
| `False` | `True` | Agent can stand on cell (e.g. bush); agent hidden from predators while on this cell |

### Damage sources

**Collision damage** (blocking obstacles, `core.py:493-494`):
```
at_attempted_obs = (obs_pos == attempted_pos)
damage_obs_collision = max(sampled_obs_damage  where at_attempted_obs)  if just_collided  else  0
```
Uses `jnp.max` — only the single hardest-hitting obstacle at the attempted cell counts.

**Overlap damage** (non-blocking obstacles, `core.py:489`):
```
damage_obs_overlap = sum(sampled_obs_damage  where (at_obs AND NOT obs_blocking))
```
All non-blocking obstacles at the agent's current cell are summed. Standing on a damaging non-blocking tile bleeds the agent each step.

Both damage types are added into `total_damage` (`core.py:499`).

### Nociception

`last_collision_noc` (`EnvState`, scalar float) stores the nociceptive intensity from the most recent blocking collision:
```
collision_noc = max(obs_nociception  where at_attempted_obs)  if just_collided  else  0
```
This value is consumed by `sense_extero_nociception` on the next observation call. It is written to state every step, so it resets to 0 if no collision occurred.

### Olfactory signature

`obs_property [num_obs, V]` — obstacles contribute to the olfactory signal. Sampled at reset via the same Gaussian formula as resources. Bush obstacles in the default config use channel 3 (`[0,0,0,1,0]`). In `sensor.py:302`, `obs_chem` is added alongside `res_chem` and `animal_chem`.

### Visual encoding

`obs_type [num_obs]` int32 is built in `config_loader.py:720-722` by sorting unique obstacle names alphabetically and mapping each name to its index. The renderer uses this index for icon selection. There is no fixed rock=0, bush=1 convention — the index depends on which names appear in the config.

---

## Neutral Animals

> **Key design change (env-entities CP1 refactor)**: Neutral animals are no longer tracked in separate `neutral_*` arrays. They are the `neutral` class of the **unified animal entity system** (`animal_*` arrays). The old `neutral_pos`, `neutral_property`, `neutral_nociception` etc. fields are gone from both `EnvState` and `EnvParams`. See [07_predator_ai](07_predator_ai.md) for the full unified animal architecture.

### What neutral animals are

A neutral animal is any entry in the unified `animal_*` arrays where `animal_classes[i] == 'neutral'` (string tag, pytree-excluded) or equivalently `animal_classes_int[i] == 1` (JAX int array). The set of neutral indices is cached in `params.neutral_indices` (a static Python tuple, pytree-excluded) and used by:
- `jax_reset` for placement (N1 fix in `core.py:822-828`).
- `update_animals` (Branch B, `core.py:344-359`) to route them through `_wander_step`.

Neutral animals defined via legacy `environment.neutral_animals:` YAML (pre-v2.0 configs) are automatically projected to `class='neutral'`, `behaviour='wander'` by `_load_animals` (`config_loader.py:366-397`).

### Behaviour: wander vs. static

`animal_behaviours[i]` (string) / `animal_behaviours_int[i]` (int) describes the movement policy:

| Behaviour string | Int code | Movement |
|-----------------|----------|----------|
| `wander` | 0 | Random jitter `(-1, 0, 1)` per axis, clamped to `animal_patrol[i]` area |
| `static` | 2 | Never moves; position frozen at reset value |

The `hunt` behaviour (int 1) is exclusively used by `predator`-class animals and runs through `_hunt_step`.

**`_wander_step` update** (`core.py:242-280`):
1. Decrement `animal_move_timer`.
2. When `move_timer <= 0`, sample random jitter `(dr, dc)` each in `{-1, 0, 1}` using two independent `jax.random.randint` draws.
3. Apply move, then clamp to `animal_patrol[i]` bounding box.
4. Hard-clamp to grid boundaries.
5. Obstacle collision: if the proposed new position is on a blocking obstacle, revert to previous position.
6. Reset timer to `animal_move_int[i]`.

There is no inter-animal collision — neutrals (and predators) can stack on the same cell.

### Olfactory contribution

`sensor.py:298-303` (B2 fix): the olfactory sensor combines resources, all animals (predators and neutrals together), and obstacles into a single `animal_chem` term:
```python
animal_chem = sense_resource(
    agent_pos,
    state.animal_pos,          # unified [N, 2] -- includes both predators and neutrals
    jnp.ones(N, bool),         # always "active" (no per-animal active flag)
    state.animal_property_sampled,
    params.sensor_radius, params.sensor_decay
)
obs_olfactory = res_chem + animal_chem + obs_chem
```

Before the CP1 refactor there were separate `pred_chem` and `neutral_chem` calls; these are now collapsed into a single `animal_chem` call over the unified array. There is no mechanism to selectively mask one class from olfaction at runtime — the split is done at analysis time via `params.predator_indices` / `params.neutral_indices`.

### Nociception

`params.animal_nociception [N]` — per-animal nociceptive intensity. For neutral animals this defaults to 0.0 in the `_load_animals` NC-1 fix (`config_loader.py:391`), but any positive value can be configured via `nociception_intensity:` in the YAML.

Critically, `sense_extero_nociception` (`sensor.py:74-80`) **only** emits nociceptive signal from animals where `params.animal_is_damaging[i] == True`:
```python
animal_intensities = where(dist < 0.1 AND animal_is_damaging, animal_nociception, 0.0)
```
`animal_is_damaging` is False for all `neutral`-class animals (set in `config_loader.py:532-533`). So even if a neutral's `nociception_intensity` is nonzero, it will NOT appear in the nociception sensor.

### Damage

`params.animal_damage [N, 2]` for neutral animals is always `[0.0, 0.0]` (auto-filled by the NC-1 fix in `config_loader.py:391`). The damage accumulation at `core.py:472-477` uses `at_damaging = at_animal AND animal_is_damaging`, so neutral animals never contribute damage regardless of position.

### Hit detection in `info` dict

`info['hit_neutral']` (`core.py:529`) uses PRE-step `state.animal_pos` masked by `~animal_is_damaging`. This asymmetry with `hit_predator` (which uses POST-step positions) is intentional for byte-parity with the pre-refactor code.

### Property sampling

Neutral animals' `animal_property_sampled` is populated at reset using a separate `prop_key_neutral` from the 4-way PRNG split (N2 fix, `core.py:928-953`). The mean and std are `animal_property[neutral_indices]` and `animal_property_std[neutral_indices]`. During the episode, `animal_property_sampled` does not change (no per-step property re-sampling for animals, unlike resources). 

### Placement

At reset, neutral animals are placed using `neutral_sa` (their subset of `animal_spawn_area`) in the `per_entity` mode via `neutral_key` (N3 fix, `core.py:799-828`). In `per_type` mode, they participate in the type-grouped scan alongside resources, predators, and obstacles in the order `[res, pred, obs, neutral]` (N1 fix). Both modes call `resolve_overlaps_global` to prevent initial position overlaps.

---

## Grid Location Types

`grid_location_type [H, W]` int32 — tile annotations set at reset from `environment.location_areas` in YAML.

| Value | Name | YAML `type:` | Renderer colour |
|-------|------|-------------|-----------------|
| 0 | Plain | `"plain"` (or omit) | White `#FFFFFF` |
| 1 | Grass | `"grass"` | Light green `#ECFDF5` |
| 2 | Sand | `"sand"` | Light amber `#FFFBEB` |

**Visual sensor** (`sensor.py:143`): tile type is one-hot encoded into the first 3 channels of the 8-channel visual observation:
- channel 0 = grass (loc 1)
- channel 1 = sand (loc 2)
- channel 2 = plain (loc 0)

**Predator concealment**: concealment is determined entirely by `obs_hides_agent` (a per-obstacle flag), not by `grid_location_type`. A bush tile that hides the agent is an obstacle with `hides_agent: true`, not a grass tile.

---

## Clarifications / FAQ

**Q: Can two resources share the same cell?**
A: At reset, no — `resolve_overlaps_global` prevents it. After respawn, **yes** — respawn position sampling (`core.py:381-388`) uses `res_spawn_area` with no occupancy check. Two food items can land on the same cell, and the agent standing there interacts with both simultaneously.

**Q: What if a resource respawns onto an obstacle cell?**
A: It will. There is no check against obstacle or animal positions on respawn. The agent on that cell receives both food and obstacle effects in the same step. For dense configs, use non-overlapping `res_spawn_area` and `obs_spawn_area` to avoid this.

**Q: What's the difference between `res_max_cons = 0`, `-1`, and a positive value?**
A: The deactivation check is `should_deactivate = active AND (res_max_cons > 0) AND (next_cons_count >= res_max_cons)`:
- `res_max_cons = -1` (or any <= 0): the `> 0` guard fails, so the resource never deactivates — infinite consumption, no respawn needed.
- `res_max_cons = 0`: same as -1 (guard fails).
- `res_max_cons = N > 0`: resource deactivates after N consumptions, respawns after `res_reg_delay` steps.

**Q: Does `res_reg_delay` start counting from deactivation or from the initial step?**
A: From deactivation. `res_reg_timer` is set to `res_reg_delay` at the moment `should_deactivate` fires (`core.py:465`). A just-deactivated resource will not respawn for exactly `res_reg_delay` more steps.

**Q: When a resource respawns, does it get a new chemical property?**
A: Yes. `core.py:391-395` re-samples the property on respawn: `new_sampled = clip(res_property + res_property_std * N(0, 1), 0, 1)`. If `res_property_std > 0`, the respawned item looks chemically different to the olfaction sensor.

**Q: Can a hiding predator be "eaten" with the eat action?**
A: No. Hiding predators always trigger automatically on overlap (`core.py:421-427`). The `eat_action_enabled` flag and eat action only gate food consumption (`core.py:432-449`). Hiding predators ignore the flag entirely.

**Q: If I step onto a blocking obstacle, do I take damage once or continuously?**
A: Once per collision attempt. Each step where `just_collided=True`, collision damage is freshly sampled. Resting in place (action 4) does not re-trigger the collision — the agent stays at its current position and never enters `move_agent` with a non-zero move. Moving repeatedly against the same wall re-samples damage each time.

**Q: Does a non-blocking obstacle with `damage > 0` deal damage on every step the agent stays on it?**
A: Yes. `damage_obs_overlap` is computed every step based on `agent_pos == obs_pos`. Standing on a non-blocking damaging cell bleeds the agent each step.

**Q: What's the max number of obstacles the agent can collide with simultaneously?**
A: For collision damage, effectively one — `damage_obs_collision` uses `jnp.max(...)` over obstacles at the `attempted_pos` (`core.py:494`). For overlap damage, all matching obstacles at the agent's position are summed.

**Q: Do neutral animals affect damage or the nociception sensor?**
A: No to both. `animal_is_damaging` is False for all neutral-class animals. `sense_extero_nociception` only emits nociceptive signal from damaging animals (`sensor.py:77`). Neutrals exist solely to add olfactory ("chemical smell") clutter to the sensory scene.

**Q: What's the difference between `res_property` and `res_property_sampled`?**
A: `res_property` is the configured mean (constant across all episodes); `res_property_sampled` is the per-reset or per-respawn Gaussian draw (`mean + std x N(0,1)`, clipped to [0,1]). Sensors read `res_property_sampled`. See also doc `01_state_and_params.md`.

**Q: Do neutral animals still have their own separate PRNG stream?**
A: Yes. `wander_key` in `jax_step` feeds `_wander_step` exclusively (`core.py:372, 404`), giving neutrals byte-identical draw shapes to the pre-refactor `neutral_key`. The key split is done at the top of `jax_step`: `key, respawn_key, hunt_key, wander_key, damage_key, property_key = jax.random.split(state.key, 6)`.

**Q: What if `grid_location_type` is set for a cell that also has an obstacle?**
A: Both coexist. The location sensor reads `grid_location_type` at the agent's cell; the visual sensor also reads obstacle type. A bush on grass tile reads as "grass tile with a bush on it" — two separate signals.

**Q: Are food olfactory channels always channel 0?**
A: By convention in the shipped configs, yes — `property: [1, 0, 0, 0, 0]` for food. But the convention is not enforced in code; channel assignment is entirely user-defined.

**Q: Does the neutral animal's patrol area differ from its spawn area?**
A: Yes — both are configured per-animal as separate YAML fields. `spawn_area` (stored as `animal_spawn_area[i]`) is used only at reset for initial placement. `patrol_area` (stored as `animal_patrol[i]`) bounds movement every step in `_wander_step` (`core.py:262-265`). Same pattern as predators.

**Q: Can neutral animals walk through each other?**
A: Yes. There is no inter-animal collision — only per-animal obstacle collision (via `jax.vmap(check_collision)` in `_wander_step`, `core.py:271-275`). Multiple animals (neutral or predator) can occupy the same cell simultaneously.

**Q: What happened to the old `neutral_property`, `neutral_nociception`, `neutral_move_timer`, `neutral_patrol` fields?**
A: They were removed in the CP1 refactor. All per-animal data is now in the unified `animal_*` arrays. `neutral_property` is now `animal_property_sampled[neutral_indices]`. `neutral_nociception` is now `animal_nociception[neutral_indices]` (though it no longer feeds the nociception sensor — see Nociception section above). `neutral_move_timer` is now `animal_move_timer[neutral_indices]`. `neutral_patrol` is now `animal_patrol[neutral_indices]`. The helper `select_by_class(params, 'neutral')` (`state.py:7-28`) returns a NumPy boolean mask for host-side slicing.
