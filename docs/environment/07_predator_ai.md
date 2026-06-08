# 07 — Animal AI (Hunt / Wander / Static)

> **Source**: `src/environment/core.py` (`_hunt_step` line 132, `_wander_step` line 242, `update_animals` line 283) | `src/environment/state.py` | `src/environment/config_loader.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

The environment supports mobile non-player animals that interact with the agent. After the **v2.0 "unified animal entity" refactor** (CP1–CP6), all animals — whether hostile predators or harmless wanderers — are stored in a single set of `animal_*` arrays in `EnvState`/`EnvParams` (shape `[N, ...]`, where N is the total count). The old split into separate `pred_*` / `neutral_*` arrays is gone.

Each animal carries two orthogonal tags:

- **Class** (`animal_classes`): `"predator"` or `"neutral"`. Determines whether the animal deals damage and which sensor channel it occupies. Only predators are damaging (`animal_is_damaging`).
- **Behaviour** (`animal_behaviours`): `"hunt"`, `"wander"`, or `"static"`. Determines which update path runs each step.

The class and behaviour are independent. In practice all current configs use *predator + hunt* and *neutral + wander*, but a neutral-class hunt animal or a predator-class wander animal is structurally supported.

**Ordering**: predators are stored first in the unified arrays, then neutrals. This order is fixed at config-load time by `_load_animals` (`config_loader.py`) and respected throughout reset, step, and sensor code.

**Key concept — behaviour index tuples**: `params.hunt_idx`, `params.wander_idx`, and `params.static_idx` are static Python tuples (not JAX arrays, `pytree_node=False`) that record which positions in the unified `[N]` arrays belong to each behaviour. `update_animals` slices these subsets, calls the matching step function, and scatters results back. A parallel pair — `params.predator_indices` and `params.neutral_indices` — records per-class positions and is used during reset for placement and property sampling.

> **Note on "Hiding Predators"**: This document covers *mobile* animals. The environment also has "hiding predators" — stationary, damage-dealing resource tiles (`res_type == 1`). Those are documented in [08 — Resources & Obstacles](08_resources_and_obstacles.md).

---

## Unified Animal Arrays

### State fields (`EnvState`, `state.py:43-55`)

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `animal_pos` | `[N, 2]` | int32 | Current (row, col) position |
| `animal_state` | `[N]` | int32 | FSM state: PATROL=0, HUNT=1, RETURN=2 (always 0 for wander/static) |
| `animal_stamina` | `[N]` | float32 | Remaining stamina (unused for wander/static; kept for shape stability) |
| `animal_move_timer` | `[N]` | int32 | Countdown to next permitted move |
| `animal_attack_timer` | `[N]` | int32 | Cooldown after hitting agent (zero for non-hunt entities) |
| `animal_property_sampled` | `[N, V]` | float32 | Per-episode chemical signature (for olfaction sensor) |
| `animal_detect_sampled` | `[N]` | float32 | Per-episode detection radius (Manhattan cells) |
| `animal_max_stamina_sampled` | `[N]` | float32 | Per-episode maximum stamina |
| `animal_recovery_sampled` | `[N]` | float32 | Per-episode stamina recovery rate per non-hunt step |
| `animal_hunt_thresh_sampled` | `[N]` | float32 | Per-episode min stamina fraction needed to start a hunt |
| `animal_lose_interest_sampled` | `[N]` | float32 | Per-episode lose-interest distance multiplier |

### Params fields (`EnvParams`, `state.py:100-138`)

Static config arrays (constant across an episode's steps):

| Field | Shape | Description |
|-------|-------|-------------|
| `animal_property` | `[N, V]` | Mean chemical property vector |
| `animal_property_std` | `[N, V]` | Property noise std |
| `animal_nociception` | `[N]` | Nociception intensity forwarded to exteroceptive sensor on contact |
| `animal_move_int` | `[N]` | Movement interval (steps between moves) |
| `animal_damage` | `[N, 2]` | Damage `[min, max]` uniform — re-sampled per collision |
| `animal_attack_delay` | `[N]` | Cooldown steps after hitting agent |
| `animal_spawn_area` | `[N, 4]` | `[min_r, min_c, max_r, max_c]` for initial placement |
| `animal_patrol` | `[N, 4]` | Hard movement bounds every step |
| `animal_detect_low/high` | `[N]` | Per-episode detection radius range |
| `animal_max_stamina_low/high` | `[N]` | Per-episode max stamina range |
| `animal_recovery_low/high` | `[N]` | Per-episode recovery rate range |
| `animal_hunt_thresh_low/high` | `[N]` | Per-episode hunt-threshold range |
| `animal_lose_interest_low/high` | `[N]` | Per-episode lose-interest multiplier range |
| `animal_classes_int` | `[N]` | Int code: predator=0, neutral=1 |
| `animal_behaviours_int` | `[N]` | Int code: wander=0, hunt=1, static=2 |
| `animal_is_damaging` | `[N]` | Bool: True iff class is predator |
| `animal_visual_channel` | `[N]` | Visual sensor channel: predator=5, neutral=7 |
| `animal_classes` | tuple[str] | String class labels (pytree_node=False) |
| `animal_behaviours` | tuple[str] | String behaviour labels (pytree_node=False) |
| `animal_tags` | tuple[str] | Per-entity name tags (pytree_node=False) |
| `hunt_idx` | tuple[int] | Indices of hunt-behaviour animals in the unified array |
| `wander_idx` | tuple[int] | Indices of wander-behaviour animals in the unified array |
| `static_idx` | tuple[int] | Indices of static-behaviour animals in the unified array |
| `predator_indices` | tuple[int] | Indices of predator-class animals (used at reset) |
| `neutral_indices` | tuple[int] | Indices of neutral-class animals (used at reset) |

### Integer code constants (`config_loader.py:32-35`)

```python
ANIMAL_CLASS_TO_INT         = {"predator": 0, "neutral": 1}
ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator": 5, "neutral": 7}
ANIMAL_DAMAGING_CLASSES     = {"predator"}
ANIMAL_BEHAVIOUR_TO_INT     = {"wander": 0, "hunt": 1, "static": 2}
```

---

## Per-Episode Sampled Parameters

Five behavioural parameters are re-sampled from a uniform distribution at the start of every episode (`jax_reset`, `core.py:957-978`). Each has a `_low` and `_high` bound in `EnvParams`. Setting `low == high` gives a deterministic (scalar) value — this is how all legacy configs behave.

```python
# core.py:963-972 — 5-way PRNG split from animal_episode_key
animal_detect_sampled        = Uniform(animal_detect_low,        max(detect_high, low),        shape=(N,))
animal_max_stamina_sampled   = Uniform(animal_max_stamina_low,   max(stamina_high, low),       shape=(N,))
animal_recovery_sampled      = Uniform(animal_recovery_low,      max(recovery_high, low),      shape=(N,))
animal_hunt_thresh_sampled   = Uniform(animal_hunt_thresh_low,   max(thresh_high, low),        shape=(N,))
animal_lose_interest_sampled = Uniform(animal_lose_interest_low, max(lose_interest_high, low), shape=(N,))
```

These sampled values are stored in `EnvState` and do **not** change during a step — `jax_step` carries them forward unchanged (`core.py:642-646`). Wander/static entries have `[0, 0]` ranges, so their sampled values are always `0.0` (harmless — the wander/static code paths never read them at runtime).

**Mandatory vs optional** (`config_loader.py:225-231`):
- `behaviour: hunt` — all five distributional fields are **mandatory** in YAML. Missing → `ValueError`.
- `behaviour: wander` or `static` — all five are optional; loader auto-fills `[0, 0]` if absent.

The `animal_episode_key` is derived without disturbing the existing key streams: `animal_episode_key = fold_in(property_key, 0xAE1)` (`core.py:783`).

---

## `update_animals` — Dispatch Architecture (`core.py:283`)

`update_animals` is the top-level animal step function called from `jax_step` (`core.py:403`). It implements a per-subset dispatch pattern:

```
update_animals(state, agent_pos, params, hunt_key, wander_key):
  Branch A — HUNT subset (core.py:312-341):
    if len(params.hunt_idx) > 0:
        h_idx = jnp.array(params.hunt_idx)            # static tuple → JAX index
        (pos, state, stam, mt, at) = _hunt_step(
            animal_pos[h_idx], animal_state[h_idx], ..., hunt_key)
        new_pos[h_idx]     = pos
        new_state[h_idx]   = state
        new_stamina[h_idx] = stam
        new_mt[h_idx]      = mt
        new_at[h_idx]      = at

  Branch B — WANDER subset (core.py:344-359):
    if len(params.wander_idx) > 0:
        w_idx = jnp.array(params.wander_idx)
        (pos, mt) = _wander_step(
            animal_pos[w_idx], animal_mt[w_idx], ..., wander_key)
        new_pos[w_idx] = pos
        new_mt[w_idx]  = mt

  Branch C — STATIC subset (core.py:361-363):
    No update. Positions and timers remain unchanged.

  return new_pos, new_state, new_stamina, new_mt, new_at
```

Key invariant: `_hunt_step` receives arrays of shape `(N_hunt, ...)` and `_wander_step` receives `(N_wander, ...)`. This preserves the PRNG draw shapes from the pre-refactor `update_predators` / `update_neutral_animals` functions (B1 fix). `hunt_key` (formerly `predator_key`) feeds `_hunt_step`; `wander_key` (formerly `neutral_key`) feeds `_wander_step`.

---

## Hunt Behaviour (`_hunt_step`, `core.py:132`)

Hunt is the active-pursuit AI used by predator-class animals. It implements a 3-state FSM with stamina-limited chase, configurable detection, and an attack cooldown.

### State Machine

Each hunt animal has an integer FSM state in `animal_state`:

| State | Integer | Description |
|-------|---------|-------------|
| PATROL | 0 | Default — random jitter within `animal_patrol` bounding box |
| HUNT | 1 | Active pursuit of the agent; stamina drains 1.0 per step |
| RETURN | 2 | Moving back toward patrol centroid after giving up a chase |

**Initial state** (`jax_reset`, `core.py:990-993`): every animal starts at PATROL (0) with `animal_stamina = animal_max_stamina_sampled` (full), `animal_move_timer = 0`, `animal_attack_timer = 0`.

### Transition Diagram

```
     become_hunt (and current state != 1)
  ┌──────────────────────────────────────┐
  │                                      ▼
Patrol (0) ◄──reentered_home── Return (2) ──become_hunt──► Hunt (1)
                                   ▲                         │
                                   └────── lose_interest ────┘
```

All three states can transition to **Hunt** via `become_hunt` (line 179: guard is `hunt_state != 1`). In particular, **Return → Hunt is possible**: a returning predator that regains enough stamina while the agent is still within detection range will re-enter Hunt without first reaching Patrol.

### Transition Logic (`core.py:160-187`)

All transitions are evaluated **every step** regardless of `move_timer`. Only movement is gated by `move_timer <= 0`.

**Patrol → Hunt** *(also Return → Hunt)* (`core.py:167-179`):
```python
dist = sum(abs(hunt_pos - agent_pos), axis=-1)   # Manhattan distance
rested_enough = hunt_stamina >= hunt_max_stamina * hunt_thresh
become_hunt   = (dist <= hunt_detect) AND rested_enough AND NOT agent_hidden
next_state    = 1  if (hunt_state != 1) AND become_hunt
```
`hunt_thresh` (default 0.7 in legacy configs) prevents an exhausted returning predator from immediately re-initiating a chase.

**Hunt → Return** (`core.py:173-180`):
```python
lose_interest = (dist > hunt_detect * hunt_lose_interest) OR (hunt_stamina <= 0) OR agent_hidden
next_state    = 2  if (hunt_state == 1) AND lose_interest
```
The predator gives up if: (a) the agent moves too far (detection range × multiplier), (b) stamina reaches zero, or (c) the agent is concealed in a bush.

**Return → Patrol** (`core.py:182-187`):
```python
tr_return = (patrol[:, 0] + patrol[:, 2]) // 2    # floor-division centroid row
tc_return = (patrol[:, 1] + patrol[:, 3]) // 2    # floor-division centroid col
dist_to_center = sum(abs(hunt_pos - [tr_return, tc_return]))
reentered_home = (next_state == 2) AND (dist_to_center <= 2)
next_state     = 0  if reentered_home
```
The threshold `2` (Manhattan steps from patrol centroid) is hardcoded. The centroid uses floor division, so for an even-span patrol box it is biased toward the smaller index.

---

## Stamina Dynamics (`core.py:236-237`)

```python
new_stamina = hunt_stamina - 1.0            if next_state == HUNT (1)
            = hunt_stamina + hunt_recovery   otherwise (PATROL=0 or RETURN=2)
new_stamina = clip(new_stamina, 0.0, hunt_max_stamina)
```

- Drain is fixed at `1.0` per hunt step (hardcoded).
- Recovery rate is `animal_recovery_sampled` — per-episode, per-entity.
- Stamina is clamped to `[0.0, animal_max_stamina_sampled]`.
- Recovery applies during **both** PATROL and RETURN. A long Return walk can refill enough stamina to trigger a Return → Hunt re-engagement.

**Hunt-threshold guard**: `become_hunt` requires `stamina >= max_stamina * hunt_thresh`. Once in Hunt, stamina can fall below the threshold — the predator only exits via `lose_interest` (at stamina=0 or distance exceeded), not by crossing the threshold downward.

---

## Movement (`core.py:189-234`)

Movement for hunt animals is throttled by `animal_move_timer` (decremented each step; reset to `animal_move_int` after a move) and by `animal_attack_timer`:

```python
should_move = (new_move_timer <= 0) AND (new_attack_timer <= 0)   # core.py:190
```
Both timers must be ≤ 0 for the animal to move. State transitions occur every step regardless of either timer.

**Direction computation** (`core.py:196-203`):
- Hunt (state == 1): toward agent — `dr = agent_pos[0] - hunt_pos[row]`
- Return (state == 2): toward patrol centroid — `dr = tr_return - hunt_pos[row]`
- Patrol (state == 0): random jitter — `dr = randint(-1, 2)`, i.e. uniform `{-1, 0, 1}` per axis

**Diagonal tie-breaking** (`core.py:205-212`):
```python
step_r, step_c = sign(dr), sign(dc)
if dr != 0 AND dc != 0:             # diagonal move needed
    if rand_choice < 0.5:            # 50/50 random draw each step
        final_move = (step_r, 0)     # row-only
    else:
        final_move = (0, step_c)     # col-only
```
Prevents diagonal oscillation and produces more natural pursuit paths.

**Spatial bounds** (`core.py:218-224`):
1. Hard-clipped to `animal_patrol` bounding box `[patrol[0], patrol[2]] × [patrol[1], patrol[3]]` — enforced for all states.
2. Hard-clipped to global grid bounds `[0, H-1] × [0, W-1]`.

Predators never leave their assigned territory, even in Hunt state.

**Obstacle collision** (`core.py:227-231`): vmapped per hunt animal using `check_collision` — if the proposed new position overlaps a blocking obstacle (`params.obs_blocking`), the animal stays at its current position (uses `obs_blocking_for_collision`, which is `params.obs_blocking` for byte-parity with the old `update_predators` argument).

---

## Attack Logic (`core.py:470-480`)

Damage computation and attack-timer assignment live in `jax_step`, **not** inside `_hunt_step`. `_hunt_step` only propagates and decrements the pre-existing `animal_attack_timer`.

```python
# core.py:471-480 — post-step overlap check
at_animal    = all(new_animal_pos == new_agent_pos, axis=-1)     # POST-step positions
at_damaging  = at_animal AND params.animal_is_damaging           # predator class only

# Damage sampled over all N animals, but only at_damaging entries contribute
sampled_pred_damage = Uniform(animal_damage[:, 0], animal_damage[:, 1], shape=(N,))
damage_pred = sum(sampled_pred_damage where at_damaging)

# Set attack timer for any damaging animal that hit the agent this step
new_animal_at = where(at_damaging, params.animal_attack_delay, new_animal_at)
```

- `animal_damage` shape `[N, 2]` — columns `[min, max]`. Scalar YAML `damage: 10` becomes `[10, 10]`.
- Damage from all colliding damaging animals is **summed** into a single `damage_pred` passed to the injury system.
- The attack timer is set to `animal_attack_delay[i]` whenever animal `i` hits the agent. While this timer is > 0, `should_move` is False — the animal freezes for the cooldown, producing a pulsed attack pattern.
- Overlap is checked at POST-step positions for damaging animals (predators). Neutral animals are checked against PRE-step `state.animal_pos` for `hit_neutral` info reporting (`core.py:512-518`).

**Nociception**: `params.animal_nociception[i]` is the intensity forwarded to the exteroceptive nociception sensor when animal `i` is at the agent's position.

---

## Wander Behaviour (`_wander_step`, `core.py:242`)

Wander is the aimless movement AI used by neutral-class animals (typically rabbits). It has no FSM — the animal drifts within its patrol box.

```python
_wander_step(wand_pos, wand_mt, wand_patrol, wand_move_int, obs_pos, obs_blocking, key):
    new_move_timer = wand_mt - 1
    should_move    = new_move_timer <= 0           # only move_timer gated (no attack_timer)

    jitter_r = randint(key, (N_wander,), -1, 2)   # uniform {-1, 0, 1} per animal
    jitter_c = randint(key, (N_wander,), -1, 2)

    new_pos = wand_pos + [jitter_r, jitter_c]  if should_move else wand_pos

    # Clip to patrol box, then to grid bounds
    new_pos = clip(new_pos, patrol[:, 0:2], patrol[:, 2:4])
    new_pos = clip(new_pos, 0, [H-1, W-1])

    # Obstacle collision (vmap per wander animal — reverts to wand_pos if blocked)
    new_pos = vmap(check_collision)(new_pos, wand_pos)

    new_move_timer = wand_move_int  if should_move else new_move_timer
    return new_pos, new_move_timer
```

Key points:
- No state, no stamina — `animal_state` stays at 0 for wander animals; stamina field is unused.
- Movement gated only by `move_timer` (no `attack_timer` gate).
- `agent_pos` is never passed to `_wander_step` — wander animals cannot detect or respond to the agent.
- Neutral-class wander animals have `animal_is_damaging = False`, so they never trigger damage or set the attack timer in `jax_step`.

---

## Static Behaviour (`core.py:361-363`)

Static animals receive **no update** each step. Their entries are passed through the unified arrays unchanged (Branch C in `update_animals`). No PRNG draw is made. Positions, timers, and state are constant after reset.

---

## Bush / Hiding Concealment (`core.py:162-165`)

```python
agent_hidden = any(all(obs_pos == agent_pos, axis=-1) AND obs_hides_agent)
```

When the agent stands on any obstacle with `hides_agent=True` (a bush), `agent_hidden=True`. This affects **only** the hunt state machine:

1. **Blocks Patrol/Return → Hunt**: `become_hunt = become_hunt AND NOT agent_hidden`
2. **Forces Hunt → Return**: `lose_interest = lose_interest OR agent_hidden`

A predator in Hunt immediately abandons the chase if the agent reaches a bush. A patrolling predator cannot initiate a chase while the agent is concealed.

`obs_hides_agent [num_obs]` is set per-obstacle in YAML (`hides_agent: true`) and stored in `EnvParams` (`state.py:142`). Wander and static animals are unaffected.

---

## Reset Initialisation (`jax_reset`, `core.py:762`)

On every episode reset:

1. **Positions**: predator-class animals placed using `predator_indices` + `animal_spawn_area`; neutral-class using `neutral_indices`. The overlap-resolve scan concatenation order is `[res, pred, obs, neutral]` (N1 fix), and positions are sliced back in the same order before being scattered into `animal_pos_init`.
2. **State**: `animal_state = zeros(N, int32)` — all PATROL (0).
3. **Stamina**: `animal_stamina = animal_max_stamina_sampled` — starts at full.
4. **Timers**: `animal_move_timer = zeros(N)`, `animal_attack_timer = zeros(N)` — ready to move immediately on step 0.
5. **Per-episode sampled params**: all five fields drawn fresh via `Uniform(low, high)` using a 5-way PRNG split (`core.py:962`). The key is `animal_episode_key = fold_in(property_key, 0xAE1)` so existing key streams remain unchanged.
6. **Properties**: predator-class and neutral-class chemical properties sampled separately with `prop_key_pred` / `prop_key_neutral` then scattered back into the unified `[N, V]` array (N2 fix).

---

## vmap Parallelism

`_hunt_step` operates on the full `(N_hunt, ...)` slice using `jnp.where` over the hunt-animal batch axis. One internal `jax.vmap` at `core.py:231` handles per-animal obstacle collision. `_wander_step` follows the same pattern with its internal `jax.vmap` at `core.py:275`.

The outer `ParallelEnv` vmaps `jax_step` (and thus `update_animals`) over the environment batch axis. In a batched call the animal arrays have shape `[num_envs, N, ...]`.

Each animal is independent — no animal-to-animal interaction exists.

**Output shapes** (single env):
- `animal_pos`: `[N, 2]`
- `animal_state`: `[N]`
- `animal_stamina`: `[N]`
- `animal_move_timer`: `[N]`
- `animal_attack_timer`: `[N]`

---

## Configuration Reference

### v2.0 unified schema (`environment.entities:`)

The new schema supports arbitrary class + behaviour combinations:

```yaml
environment:
  entities:
    - class: "predator"           # animal_classes — "predator" or "neutral"
      behaviour: "hunt"           # animal_behaviours — "hunt", "wander", or "static"
      tag: "predator0"            # animal_tags — optional label
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]       # chemical signature (olfaction)
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1                              # animal_move_int — steps between moves
      nociception_intensity: 0.9                    # animal_nociception
      damage: [15.0, 45.0]                          # animal_damage [min, max] uniform
      spawn_area: [[1, 1], [10, 10]]                # 1-indexed, inclusive
      patrol_area: [[1, 1], [10, 10]]               # hard movement bounds
      attack_delay: 3                               # animal_attack_delay — cooldown steps
      # Distributional fields (MANDATORY for hunt, optional for wander/static):
      # Scalar → degenerate [s, s]; list [lo, hi] → per-episode uniform draw.
      detection_range: 5                            # scalar = degenerate [5.0, 5.0]
      max_stamina: [20, 40]                         # per-episode Uniform(20, 40)
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5

    - class: "neutral"
      behaviour: "wander"
      tag: "rabbit0"
      count: 2
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.1
      damage: [0.0, 0.0]
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
      attack_delay: 0
      # Distributional fields omitted — loader auto-fills [0, 0]
```

Example non-degenerate distributional config: `configs/experiment/v2_smoke/02-entities-distributional.yaml`.
Full parity smoke config (entities: schema vs legacy): `configs/experiment/v2_smoke/01-entities-smoke.yaml`.

### Legacy schema (`environment.predators:` + `environment.neutral_animals:`)

Still accepted by the loader (all 86 pre-v2.0 configs use this path). The loader automatically re-projects:
- Each `predators:` entry → `class='predator', behaviour='hunt'`; distributional fields are mandatory.
- Each `neutral_animals:` entry → `class='neutral', behaviour='wander'`; distributional fields are optional; `damage` and `attack_delay` auto-filled to `[0.0, 0.0]` / `0` (NC-1 fix).

The key `predator_enabled` was **removed in v2.0** (`config_loader.py:663-668`). Present in YAML → `DeprecationWarning` (does not error).

### Area index conversion (`config_loader.py:241-243`)

YAML `[[1, 1], [10, 10]]` is parsed as `[row_min-1, col_min-1, row_max, col_max] = [0, 0, 10, 10]`. Lower corner shifts to 0-indexed; upper corner is kept as-is. YAML reads as 1-indexed inclusive on both ends.

### `spawn_area` vs `patrol_area`

- `spawn_area`: used **only at reset** to sample initial position.
- `patrol_area`: hard movement bound enforced every step (`core.py:218-224`). A predator can spawn outside its patrol area if the two differ, but clips back on its first move.

---

## Legacy Aliases (`state.py:234-247`)

`params.predator_tags` and `params.neutral_tags` are read-only `@property` aliases on `EnvParams` that filter `animal_tags` by class. They exist for one release cycle to keep `dreamer_srl_main.py` and `accumulators.py` working without edits (B3 fix).

A host-side helper `select_by_class(params, class_name)` (`state.py:7-28`) returns a NumPy boolean mask for analysis scripts and renderers — not callable inside `jit`'d kernels.

---

## Clarifications / FAQ

**Q: Does the predator see the agent's old or new position?**
A: The **new** position. `jax_step` calls `move_agent` first (step 2), then `update_animals` with `new_agent_pos` (step 3). The predator reacts to the agent's position *after* the agent has moved this step.

**Q: Is detection omnidirectional?**
A: Yes. Detection is a pure Manhattan distance check — no line-of-sight, no facing direction, and obstacles do **not** block vision. A predator detects the agent through walls.

**Q: What happens if two predators try to occupy the same cell?**
A: Nothing prevents it. There is no inter-animal collision check. Two predators can stack, and both will damage the agent if the agent is there.

**Q: What if the patrol area is a single cell?**
A: The centroid equals that cell. The predator sits still in Patrol/Return. Jitter is clipped back to the single cell every step.

**Q: Does stamina recover during Return?**
A: Yes. Recovery applies whenever `next_state != 1`, which includes both PATROL (0) and RETURN (2). A long Return trip can refill enough stamina to trigger a Return → Hunt re-engagement.

**Q: Can stamina exceed `max_stamina`?**
A: No. `new_stamina = clip(new_stamina, 0.0, hunt_max_stamina)` (`core.py:237`).

**Q: How does `attack_delay` interact with `move_interval`?**
A: Both timers must be ≤ 0 for the predator to move: `should_move = (new_move_timer <= 0) AND (new_attack_timer <= 0)` (`core.py:190`). The longer of the two dominates. State transitions happen every step regardless.

**Q: Is the hunt-threshold checked on entry only, or every step?**
A: Every step, but only for the Patrol/Return → Hunt transition (`become_hunt`). Once in Hunt, stamina can fall below the threshold — the predator exits Hunt only when `lose_interest` fires, not when stamina crosses the threshold.

**Q: What does `lose_interest_multiplier = 1.0` produce?**
A: The predator gives up as soon as the agent exits detection range. Values above `1.0` create a hysteresis band — the predator continues chasing up to `detect × mult` once a hunt has started, but cannot *initiate* a hunt beyond `detect`.

**Q: Are the random diagonal tie-breaks reproducible?**
A: Yes. The PRNG key is threaded through `_hunt_step` and splits deterministically (`core.py:192, 208`). Given the same seed and state, the same moves occur.

**Q: What if `spawn_area` overlaps with an obstacle?**
A: Spawn positions are passed through `resolve_overlaps_global` (`core.py:662`), which resolves entity-to-entity overlaps. Animal-on-obstacle overlap is not separately guaranteed to be rejected — verify against the resolve logic if this matters.

**Q: Do wander animals respond to the agent?**
A: No. `_wander_step` never receives `agent_pos`. Wander is a pure random walk within the patrol territory.

**Q: Are there predator-to-predator interactions?**
A: No. Each animal is updated independently using `jnp.where` over the batch axis. No predator-to-predator collision or coordination logic exists.

**Q: Does `predator_enabled: false` still work?**
A: No. This key was removed in v2.0. To disable predators, use an empty `predators: []` list in the legacy schema, or simply omit the `entities:` section's predator entries. With N=0 hunt animals, `update_animals` Branch A is a no-op (the `if len(params.hunt_idx) > 0:` guard at `core.py:312` short-circuits).
