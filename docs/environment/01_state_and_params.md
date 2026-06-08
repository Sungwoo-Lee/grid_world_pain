# 01 — State & Parameters

> **Source**: `src/environment/state.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

This document describes the two core data structures that carry all state in the GridWorld Pain environment: `EnvState` (what is happening right now) and `EnvParams` (the fixed rules for this episode). Both are defined in `src/environment/state.py`.

**What changed in the "unified animal entity" refactor (env_entities CP1–CP6):** Before this refactor, predators and neutral animals were stored in separate arrays (`pred_pos`, `pred_state`, `pred_stamina`, …, `neutral_pos`, `neutral_move_timer`, …). These have been replaced by a single unified `animal_*` array family covering all animals — predators first, then neutral animals. A per-entity integer tag (`animal_classes`, `animal_behaviours`) tracks which kind each index represents. Code that previously wrote `pred_pos[i]` now writes `animal_pos[pred_idx]`. This affects both `EnvState` and `EnvParams`.

---

## What These Structures Are

- **`EnvState`** — mutable, per-step data that changes every call to `jax_step`. Carries all "what is happening right now" information: positions, timers, body values, RNG key.
- **`EnvParams`** — immutable, per-episode configuration. Locked at reset time and shared across all steps. Contains all rules, limits, and constant entity attributes.

Both are Flax `@struct.dataclass` pytrees. This means JAX can trace and JIT them without recompilation when values change — only when *structure* changes. The split matters for `jax.jit` because `EnvParams` fields declared with `struct.field(pytree_node=False)` are treated as **static**: they fix the compiled graph shape and force a new compilation if their value changes. All other fields are pytree leaves and can change freely between calls.

Updates always produce a new object via `.replace(**kwargs)` (aliased as `._replace(**kwargs)` for backward compatibility). No in-place mutation ever occurs.

---

## `select_by_class()` — Host-Side Class Filter

Takes `params` and a class name string, and returns a boolean NumPy mask of length N that is `True` at every index whose animal belongs to that class. Used outside JIT to extract predator or neutral subsets for analysis, rendering, and eval.

`Source: src/environment/state.py:7–28`
```python
def select_by_class(params, class_name: str) -> np.ndarray:
    """Return a boolean NumPy mask of length N selecting animals of the given class.

    Args:
        params: EnvParams — holds `animal_classes` tuple (pytree_node=False, len N).
        class_name: one of 'predator', 'neutral', or any future class string.

    Returns:
        np.ndarray[bool, shape (N,)] — True at index i iff animal_classes[i] == class_name.

    Usage:
        pred_mask = select_by_class(params, 'predator')
        pred_pos = np.array(state.animal_pos)[pred_mask]   # shape [N_pred, 2]

    Notes:
        - This is a host-side (NumPy) helper — not a JAX traced function. Call it
          from analysis scripts, renderers, and eval code, not inside jit'd kernels.
        - The mask is derived from `params.animal_classes`, a static
          `pytree_node=False` tuple, so it is config-constant and allocation-free
          when called repeatedly with the same params.
    """
    return np.array([c == class_name for c in params.animal_classes], dtype=bool)
```

> **API notes**
>
> - `params.animal_classes` is a `struct.field(pytree_node=False)` tuple — a **static** field, not a JAX array. See [primer: static-dynamic](00_jax_primer.md#static-dynamic). Iterating it with a Python list comprehension is valid host-side code, not a traced operation.
> - The return type is `np.ndarray` (plain NumPy), not `jnp.ndarray`. This is intentional: `select_by_class` is a host-only utility. Inside a JIT'd kernel, use the precomputed `params.predator_indices` / `params.neutral_indices` index tuples (also static) as slice indices — they avoid any dynamic-shape problem.
> - `params` itself is registered as a [primer: pytrees](00_jax_primer.md#jax-pytrees) node, but `select_by_class` never touches any of its array fields — only the `pytree_node=False` tuple — so no JAX machinery is invoked at all.

---

## EnvState — Mutable Per-Step

`EnvState` is a Flax `@struct.dataclass` whose fields are all JAX pytree leaves (no `pytree_node=False` here). Every call to `jax_step` returns a *new* `EnvState` via `.replace()`; no field is mutated in place.

`Source: src/environment/state.py:30–81`
```python
@struct.dataclass
class EnvState:
    # Agent
    agent_pos: jnp.ndarray      # [2] (row, col)
    current_step: jnp.ndarray   # []

    # Resources
    res_pos: jnp.ndarray        # [num_res, 2]
    res_active: jnp.ndarray     # [num_res] bool
    res_cons_count: jnp.ndarray # [num_res] int
    res_reg_timer: jnp.ndarray  # [num_res] int
    res_property_sampled: jnp.ndarray # [num_res, vector_size]

    # Animals (unified — predators + neutrals, predators-first ordering)
    animal_pos: jnp.ndarray              # [N, 2]
    animal_state: jnp.ndarray            # [N] int (PATROL=0, HUNT=1, RETURN=2; 0 for wander/static)
    animal_stamina: jnp.ndarray          # [N] float (unused for wander/static, kept for shape stability)
    animal_move_timer: jnp.ndarray       # [N] int
    animal_attack_timer: jnp.ndarray     # [N] int (zero for non-hunt entities)
    animal_property_sampled: jnp.ndarray # [N, vector_size]
    # Per-episode-sampled behavioural params (NEW — all five fields, degenerate [s,s] for legacy configs)
    animal_detect_sampled: jnp.ndarray         # [N] float
    animal_max_stamina_sampled: jnp.ndarray    # [N] float
    animal_recovery_sampled: jnp.ndarray       # [N] float
    animal_hunt_thresh_sampled: jnp.ndarray    # [N] float
    animal_lose_interest_sampled: jnp.ndarray  # [N] float

    # Obstacles
    obs_pos: jnp.ndarray        # [num_obs, 2]
    obs_property_sampled: jnp.ndarray # [num_obs, vector_size]

    # Body
    satiation: jnp.ndarray       # [] float
    nutrition: jnp.ndarray       # [] float
    injury_level: jnp.ndarray    # [] float
    injury_buffer: jnp.ndarray   # [smoothing_duration] float
    nociception_history_buffer: jnp.ndarray  # [interoceptive_kernel_length] float (past injury_level values, idx 0 = most recent)
    last_collision_noc: jnp.ndarray # float (intensity of last collision)
    rest_streak: jnp.ndarray     # [] int

    # Environment status
    terminated: jnp.ndarray      # bool

    # Random State
    key: jax.random.PRNGKey      # PRNGKey

    # Proprioception
    last_action: jnp.ndarray      # [] (int32 action index)

    def _replace(self, **kwargs):
        return self.replace(**kwargs)
```

> **API notes**
>
> - `@struct.dataclass` registers `EnvState` as a JAX pytree node. Every `jnp.ndarray` field is a pytree leaf; `jax.jit`, `jax.vmap`, and `jax.lax.scan` can all accept and return it without manual flattening. See [primer: pytrees](00_jax_primer.md#jax-pytrees).
> - **Immutability**: no field is ever mutated. Updates use `state.replace(field=new_val)` (or `state._replace(...)`) which returns a fresh struct. The step function builds the next `EnvState` in one large `.replace()` call. See [primer: immutability](00_jax_primer.md#immutability).
> - **`key: jax.random.PRNGKey`** is itself a JAX array (shape `[2]` uint32) and therefore a pytree leaf. It is split at the top of each step to derive all random events for that step. See [primer: prng](00_jax_primer.md#prng).
> - **`injury_buffer`** has shape `[smoothing_duration]`. The *length* of this buffer is a static field on `EnvParams` (`smoothing_duration: int = struct.field(pytree_node=False)`), which pins the shape at compile time and prevents recompilation as long as `smoothing_duration` is unchanged. See [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - **`vmap` batching**: `ParallelEnv` vmaps over the leading dimension of each array. Conceptually you have a batch of `EnvState` objects stacked along axis 0; the struct fields just grow a leading batch dimension. See [primer: vmap](00_jax_primer.md#vmap).

Defined in `src/environment/state.py:30`. All fields are JAX pytree leaves (no `pytree_node=False` here).

### Agent

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `agent_pos` | `[2]` | int32 | `[0, H-1] × [0, W-1]` | Agent's `(row, col)` position on the grid |
| `current_step` | `[]` | int32 | `[0, max_steps]` | Step counter within the episode |
| `last_action` | `[]` | int32 | `[0, action_dim-1]` | Action index taken on the previous step (used by the proprioception sensor); 0–3 = movement (Up/Down/Left/Right), 4 = Rest, 5 = Eat |

### Resources

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `res_pos` | `[num_res, 2]` | int32 | grid bounds | Current `(row, col)` of each resource |
| `res_active` | `[num_res]` | bool | `{0, 1}` | Whether the resource is available for interaction |
| `res_cons_count` | `[num_res]` | int32 | `[0, max_cons]` | How many times this resource has been consumed since last respawn |
| `res_reg_timer` | `[num_res]` | int32 | `[0, reg_delay]` | Countdown to respawn; decrements each step when inactive |
| `res_property_sampled` | `[num_res, vector_size]` | float32 | `[0, 1]` | Olfactory signature sampled at respawn |

### Animals (unified — predators first, then neutral animals)

After the CP1–CP6 refactor, all animal entities share a single set of arrays indexed `0…N-1`, where the first `N_pred` indices are predators and the remaining `N_neutral` indices are neutral animals. This ordering is enforced at reset time (`core.py:891`) and must be preserved whenever indices are interpreted.

`animal_state` uses the FSM integer codes PATROL=0, HUNT=1, RETURN=2. Wander and static entities always sit at state 0 (their state machine has no Hunt/Return transitions).

The five `animal_*_sampled` fields are per-episode behavioural parameters drawn fresh at every reset from the `[low, high]` ranges stored in `EnvParams`. For non-hunt entities (wander/static), both bounds are 0.0, so the sampled value is always 0.0 — the fields are kept at shape `[N]` purely for shape stability under JAX JIT.

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `animal_pos` | `[N, 2]` | int32 | grid bounds | Current `(row, col)` of every animal; predators at indices 0…N_pred-1 |
| `animal_state` | `[N]` | int32 | `{0, 1, 2}` | FSM state: 0=Patrol/Idle, 1=Hunt, 2=Return; always 0 for wander/static |
| `animal_stamina` | `[N]` | float32 | `[0, max_stamina_sampled]` | Current stamina; drains during Hunt and recovers otherwise; unused for wander/static but kept for shape stability |
| `animal_move_timer` | `[N]` | int32 | `[0, move_int]` | Countdown to next movement step |
| `animal_attack_timer` | `[N]` | int32 | `[0, attack_delay]` | Cooldown after an attack; blocks movement while nonzero; 0 for non-hunt entities |
| `animal_property_sampled` | `[N, vector_size]` | float32 | `[0, 1]` | Olfactory signature sampled at reset (predator-class and neutral-class are sampled with independent PRNG keys, per N2 fix) |
| `animal_detect_sampled` | `[N]` | float32 | `[detect_low, detect_high]` | Per-episode Manhattan detection radius; 0 for wander/static |
| `animal_max_stamina_sampled` | `[N]` | float32 | `[max_stamina_low, max_stamina_high]` | Per-episode maximum stamina; 0 for wander/static |
| `animal_recovery_sampled` | `[N]` | float32 | `[recovery_low, recovery_high]` | Per-episode stamina recovery per non-hunt step; 0 for wander/static |
| `animal_hunt_thresh_sampled` | `[N]` | float32 | `[hunt_thresh_low, hunt_thresh_high]` | Per-episode fraction of max_stamina required before switching to Hunt; 0 for wander/static |
| `animal_lose_interest_sampled` | `[N]` | float32 | `[lose_interest_low, lose_interest_high]` | Per-episode detection-range multiplier for the "lose interest" distance; 0 for wander/static |

### Obstacles

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `obs_pos` | `[num_obs, 2]` | int32 | grid bounds | `(row, col)` of each obstacle; fixed after reset |
| `obs_property_sampled` | `[num_obs, vector_size]` | float32 | `[0, 1]` | Olfactory signature sampled at reset |

### Body

| Field | Shape | dtype | Range | Description |
|-------|-------|-------|-------|-------------|
| `satiation` | `[]` | float32 | `[0, max_satiation]` | Subjective fullness; always derived from nutrition at reset via `max_S × (N/max_N)^k` — never set independently |
| `nutrition` | `[]` | float32 | `[0, max_nutrition]` | Objective energy reserve; decays each step |
| `injury_level` | `[]` | float32 | `[0, max_injury]` | Current accumulated injury after ring-buffer smoothing |
| `injury_buffer` | `[smoothing_duration]` | float32 | `[0, …]` | Ring buffer: incoming damage is spread across `smoothing_duration` future slots; slot 0 is consumed into `injury_level` each step |
| `nociception_history_buffer` | `[interoceptive_kernel_length]` | float32 | `[0, max_injury]` | FIR history of past `injury_level` values used by the interoceptive sensor; index 0 = most recent |
| `last_collision_noc` | `[]` | float32 | `[0, 1]` | Nociception intensity from the most recent obstacle collision bump; cleared the next step |
| `rest_streak` | `[]` | int32 | `[0, …]` | Consecutive steps the agent has been resting; boosts injury recovery rate |

### Meta / RNG

| Field | Shape | dtype | Description |
|-------|-------|-------------|-------------|
| `terminated` | `[]` | bool | Whether this episode has ended |
| `key` | `PRNGKey` | uint32 | JAX PRNG key, split each step to produce all random events for that step |

---

## EnvParams — Immutable Per-Episode

`EnvParams` is a Flax `@struct.dataclass` that holds every fixed rule and constant attribute for one episode. It is built once from the YAML config and never mutated during training; any update produces a new object via `.replace()`. Fields tagged `struct.field(pytree_node=False)` are **static** — they shape the compiled XLA graph and trigger recompilation if changed.

`Source: src/environment/state.py:82–250`
```python
@struct.dataclass
class EnvParams:
    # Grid
    height: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)
    max_steps: int = struct.field(pytree_node=False)
    grid_location_type: jnp.ndarray # [height, width] (0:plain, 1:grass, 2:sand)

    # Resources (Constant attributes)
    res_type: jnp.ndarray       # [num_res] int (0:food, 1:hiding_predator)
    res_property: jnp.ndarray   # [num_res, vector_size]
    res_property_std: jnp.ndarray # [num_res, vector_size]
    res_nociception: jnp.ndarray # [num_res]
    res_spawn_area: jnp.ndarray # [num_res, 4] (min_r, min_c, max_r, max_c)
    res_max_cons: jnp.ndarray   # [num_res]
    res_reg_delay: jnp.ndarray  # [num_res]
    res_damage: jnp.ndarray     # [num_res, 2] [min, max]

    # Animals (unified — predators-first ordering)
    animal_property: jnp.ndarray       # [N, V]
    animal_property_std: jnp.ndarray   # [N, V]
    animal_nociception: jnp.ndarray    # [N]
    animal_move_int: jnp.ndarray       # [N] int
    animal_damage: jnp.ndarray         # [N, 2]
    animal_attack_delay: jnp.ndarray   # [N] int
    animal_spawn_area: jnp.ndarray     # [N, 4] int
    animal_patrol: jnp.ndarray         # [N, 4] int
    # Per-episode uniform bounds (low == high for legacy scalar configs)
    animal_detect_low: jnp.ndarray         # [N]
    animal_detect_high: jnp.ndarray        # [N]
    animal_max_stamina_low: jnp.ndarray    # [N]
    animal_max_stamina_high: jnp.ndarray   # [N]
    animal_recovery_low: jnp.ndarray       # [N]
    animal_recovery_high: jnp.ndarray      # [N]
    animal_hunt_thresh_low: jnp.ndarray    # [N]
    animal_hunt_thresh_high: jnp.ndarray   # [N]
    animal_lose_interest_low: jnp.ndarray  # [N]
    animal_lose_interest_high: jnp.ndarray # [N]
    # Per-entity int-coded class/behaviour (for damage masking and visual channel)
    animal_classes_int: jnp.ndarray        # [N] int (0=predator, 1=neutral, ...)
    animal_behaviours_int: jnp.ndarray     # [N] int (0=wander, 1=hunt, 2=static)
    animal_is_damaging: jnp.ndarray        # [N] bool (precomputed from class)
    animal_visual_channel: jnp.ndarray     # [N] int (5=predator, 7=neutral, ...)
    # Static tags / labels (pytree_node=False — not JAX arrays)
    animal_classes: tuple = struct.field(pytree_node=False)    # len N strings
    animal_behaviours: tuple = struct.field(pytree_node=False) # len N strings
    animal_tags: tuple = struct.field(pytree_node=False)       # len N strings
    # Static per-subset index tuples (B1 fix — used by update_animals to slice
    # hunt / wander / static subsets while preserving today's PRNG draw shapes).
    hunt_idx: tuple = struct.field(pytree_node=False)   # tuple[int, ...], len N_pred
    wander_idx: tuple = struct.field(pytree_node=False) # tuple[int, ...], len N_neutral
    static_idx: tuple = struct.field(pytree_node=False) # tuple[int, ...], len N_static
    # Static per-class index tuples (N1/N2 fix — used by jax_reset to slice
    # predator / neutral subsets during placement + property sampling, preserving
    # today's per-type PRNG draw shapes).
    predator_indices: tuple = struct.field(pytree_node=False)  # tuple[int, ...], len N_pred_class
    neutral_indices: tuple = struct.field(pytree_node=False)   # tuple[int, ...], len N_neutral_class

    # Obstacles
    obs_blocking: jnp.ndarray   # [num_obs] bool
    obs_hides_agent: jnp.ndarray # [num_obs] bool (bush-type concealment)
    obs_spawn_area: jnp.ndarray # [num_obs, 4] (min_r, min_c, max_r, max_c)
    obs_damage: jnp.ndarray     # [num_obs, 2] [min, max]
    obs_property: jnp.ndarray   # [num_obs, vector_size]
    obs_property_std: jnp.ndarray # [num_obs, vector_size]
    obs_nociception: jnp.ndarray # [num_obs]
    obs_type: jnp.ndarray       # [num_obs] int32 index for names
    obstacle_names: tuple[str, ...] = struct.field(pytree_node=False)

    # Placement (Type-Level overlap resolution)
    type_areas: jnp.ndarray        # [T, 4] spawn area per type group
    type_counts: jnp.ndarray       # [T] entity count per type group
    type_entity_map: jnp.ndarray   # [T, max_per_type] global entity indices
    max_per_type: int = struct.field(pytree_node=False)   # max entities in any group
    num_types: int = struct.field(pytree_node=False)       # number of type groups
    num_entities: int = struct.field(pytree_node=False)    # total entities
    placement_mode: str = struct.field(pytree_node=False)  # "per_entity" or "per_type"

    # Body
    max_satiation: float
    max_nutrition: float
    max_injury: float
    food_nutrition_gain: float
    setpoint: float
    start_satiation: float              # For non-random start
    start_nutrition: float
    metabolic_cost: float
    nutrition_to_satiation_scaling_factor: float
    recovery_base_rate: float
    recovery_accel_rate: float
    smoothing_duration: int = struct.field(pytree_node=False)
    death_penalty: float
    overeating_death: bool = struct.field(pytree_node=False)
    use_homeostatic_reward: bool = struct.field(pytree_node=False)
    with_satiation: bool = struct.field(pytree_node=False)
    with_nutrition: bool = struct.field(pytree_node=False)
    with_injury: bool = struct.field(pytree_node=False)
    random_start_satiation: bool = struct.field(pytree_node=False)
    random_start_nutrition: bool = struct.field(pytree_node=False)
    random_start_injury: bool = struct.field(pytree_node=False)
    random_start_pos: bool = struct.field(pytree_node=False)
    start_pos: jnp.ndarray  # [2]
    rest_action_enabled: bool = struct.field(pytree_node=False)
    eat_action_enabled: bool = struct.field(pytree_node=False)
    eating_nutrition_cost: float
    eating_reward_penalty: float


    # Sensory
    sensor_radius: float
    sensor_decay: float
    sensor_range: int = struct.field(pytree_node=False)
    visual_sensor_enabled: bool = struct.field(pytree_node=False)
    visual_sensor_range: int = struct.field(pytree_node=False)
    local_view_size: int = struct.field(pytree_node=False)
    olfactory_enabled: bool = struct.field(pytree_node=False)
    nociception_enabled: bool = struct.field(pytree_node=False)
    location_sensor_enabled: bool = struct.field(pytree_node=False)

    # Hidden-state observability flags
    injury_observable: bool = struct.field(pytree_node=False)
    nutrition_observable: bool = struct.field(pytree_node=False)

    # Interoceptive nociception (delayed-peak perception of hidden injury)
    interoceptive_nociception_enabled: bool = struct.field(pytree_node=False)
    interoceptive_convolution_enabled: bool = struct.field(pytree_node=False)
    interoceptive_kernel_length: int = struct.field(pytree_node=False)
    interoceptive_kernel: jnp.ndarray  # [interoceptive_kernel_length] float, normalized alpha kernel (zeros when convolution disabled)

    # Proprioception
    proprioception_enabled: bool = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    olfactory_vector_size: int = struct.field(pytree_node=False)
    nociception_size: int = struct.field(pytree_node=False)

    # Perceptual Noise Parameters (Vectorized across modalities)
    perceptual_noise_enabled: bool = struct.field(pytree_node=False)
    # Order defined by YAML perceptual_noise.modalities key order (read via config_loader).
    # sensor.py builds modality_map dynamically from this tuple — do not reorder independently.
    noise_modality_order: tuple = struct.field(pytree_node=False)  # e.g. ("Injury","Nutrition",...)
    noise_modes: jnp.ndarray          # [13] int32 (0: None, 1: Constant, 2: State-Dependent)
    noise_sigmas: jnp.ndarray         # [13] float32 (Base Sigma)
    noise_injury_scales: jnp.ndarray  # [13] float32 (Injury Noise Scale)
    noise_clip_min: jnp.ndarray       # [13] float32 (Per-modality observation lower bound)
    noise_clip_max: jnp.ndarray       # [13] float32 (Per-modality observation upper bound)

    # ── Legacy @property aliases (B3 fix — kept for one release cycle) ────────
    # These accessors allow code that reads `params.predator_tags` / `params.neutral_tags`
    # (e.g., dreamer_srl_main.py:522-523, accumulators.py) to work without edits.
    # The corresponding struct.field declarations have been REMOVED (M1 fix) so these
    # properties are not shadowed by a static field.

    @property
    def predator_tags(self) -> tuple:
        """Legacy alias — derived from animal_tags filtered by class == 'predator'.

        Read by src/algorithms/dreamer_srl/dreamer_srl_main.py:522-523 and
        accumulators.py setup. Will be removed once all consumers migrate to
        consume animal_tags + class_indices() directly (post-CP6 + one release).
        """
        return tuple(t for t, c in zip(self.animal_tags, self.animal_classes) if c == 'predator')

    @property
    def neutral_tags(self) -> tuple:
        """Legacy alias — derived from animal_tags filtered by class == 'neutral'."""
        return tuple(t for t, c in zip(self.animal_tags, self.animal_classes) if c == 'neutral')

    def _replace(self, **kwargs):
        return self.replace(**kwargs)
```

> **API notes**
>
> - `@struct.dataclass` registers `EnvParams` as a JAX [primer: pytrees](00_jax_primer.md#jax-pytrees) node. Every `jnp.ndarray` field is a pytree leaf that JAX can substitute freely at runtime without recompilation.
> - Fields declared with `= struct.field(pytree_node=False)` are **static**: JAX bakes their concrete Python value into the compiled XLA graph. Python-level `if` branches inside JIT'd functions may safely test these fields. Changing any static field (e.g. `height`, `smoothing_duration`, `hunt_idx`) triggers a full recompile. See [primer: static-dynamic](00_jax_primer.md#static-dynamic).
> - **The nine static tuple fields** (`animal_classes`, `animal_behaviours`, `animal_tags`, `hunt_idx`, `wander_idx`, `static_idx`, `predator_indices`, `neutral_indices`, `obstacle_names`) are plain Python tuples, not JAX arrays. They define N and per-index semantics. Because they are static, JIT'd code can use their elements as fixed slice indices — e.g. `animal_pos[hunt_idx, :]` — without triggering a dynamic-shape error.
> - **`@property` aliases** (`predator_tags`, `neutral_tags`) are ordinary Python `@property` decorators on a Flax struct. Flax does not interfere with `@property` — they work exactly as in a regular Python class. They zip over the two static tuples `animal_tags` and `animal_classes`, which is pure Python computation, never traced by JAX.
> - **`._replace(**kwargs)`** at line 249 delegates to `.replace(**kwargs)` — Flax's generated method that returns a new struct with the named fields swapped. This is the immutability pattern described in [primer: immutability](00_jax_primer.md#immutability). Having both `._replace` and `.replace` means legacy call sites (which used `namedtuple._replace` style) work unchanged.

Defined in `src/environment/state.py:82`. Fields with `struct.field(pytree_node=False)` are **static** — they determine the compiled graph shape and trigger recompilation if changed. All other fields are JAX-array pytree leaves and can change between episodes without recompilation.

### Static vs. Dynamic — Why It Matters

Fields marked **static** below are baked into the compiled XLA program. Python-level `if` statements inside JIT'd functions may branch on them safely. Changing a static field after first compilation triggers a full JIT recompile. **Changing any dynamic (non-static) field is free at runtime** — XLA just substitutes the new values.

The most important static fields are:
- Grid dimensions (`height`, `width`, `max_steps`)
- The three label tuples for animals (`animal_classes`, `animal_behaviours`, `animal_tags`) — they define N and the per-index semantics
- The five index tuples (`hunt_idx`, `wander_idx`, `static_idx`, `predator_indices`, `neutral_indices`) — they act as static loop bounds for subset updates
- Sensor / body / noise feature-shape flags (`interoceptive_kernel_length`, `smoothing_duration`, `action_dim`, `sensor_range`, etc.)

### Grid

| Field | Type | Static? | Description |
|-------|------|---------|-------------|
| `height` | int | **yes** | Grid height (rows) |
| `width` | int | **yes** | Grid width (cols) |
| `max_steps` | int | **yes** | Episode length before truncation |
| `grid_location_type` | `[H, W]` int32 | no | Terrain type per cell: 0=plain, 1=grass, 2=sand |

### Resources

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `res_type` | `[num_res]` | int32 | Entity type tag: 0=food, 1=hiding_predator |
| `res_property` | `[num_res, vector_size]` | float32 | Olfactory chemical signature mean per resource |
| `res_property_std` | `[num_res, vector_size]` | float32 | Olfactory chemical signature std dev per resource |
| `res_nociception` | `[num_res]` | float32 | Nociception intensity emitted on contact |
| `res_spawn_area` | `[num_res, 4]` | int32 | Bounding box `(min_r, min_c, max_r, max_c)` for respawn |
| `res_max_cons` | `[num_res]` | int32 | Max consumptions before permanent deactivation; -1 = unlimited |
| `res_reg_delay` | `[num_res]` | int32 | Steps to wait before respawning after consumption |
| `res_damage` | `[num_res, 2]` | float32 | Damage `[min, max]` sampled uniformly on contact |

### Animals (unified — replaces old separate pred_* / neutral_* param blocks)

The `animal_*` param arrays all have length N = N_pred + N_neutral, in predators-first order matching `EnvState.animal_pos`. Per-class static lookup is done through the index tuples `predator_indices` / `neutral_indices` (see below).

Five pairs of `*_low` / `*_high` arrays define the per-episode sampling range for distributional behavioural parameters. For legacy configs (scalar values), low == high, so the sampled value is always the original constant. For modern configs these can specify a range, and the value is resampled every episode from `Uniform(low, high)`.

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `animal_property` | `[N, vector_size]` | float32 | Olfactory signature mean |
| `animal_property_std` | `[N, vector_size]` | float32 | Olfactory signature std dev |
| `animal_nociception` | `[N]` | float32 | Nociception intensity on contact |
| `animal_move_int` | `[N]` | int32 | Steps per move (lower = faster) |
| `animal_damage` | `[N, 2]` | float32 | Damage range `[min, max]` sampled uniformly on contact |
| `animal_attack_delay` | `[N]` | int32 | Cooldown steps after an attack |
| `animal_spawn_area` | `[N, 4]` | int32 | Spawn bounding box `(min_r, min_c, max_r, max_c)` |
| `animal_patrol` | `[N, 4]` | int32 | Patrol bounding box `(min_r, min_c, max_r, max_c)` |
| `animal_detect_low` | `[N]` | float32 | Lower bound for per-episode detect-radius sampling |
| `animal_detect_high` | `[N]` | float32 | Upper bound for per-episode detect-radius sampling |
| `animal_max_stamina_low` | `[N]` | float32 | Lower bound for per-episode max-stamina sampling |
| `animal_max_stamina_high` | `[N]` | float32 | Upper bound for per-episode max-stamina sampling |
| `animal_recovery_low` | `[N]` | float32 | Lower bound for per-episode recovery-rate sampling |
| `animal_recovery_high` | `[N]` | float32 | Upper bound for per-episode recovery-rate sampling |
| `animal_hunt_thresh_low` | `[N]` | float32 | Lower bound for per-episode hunt-threshold sampling |
| `animal_hunt_thresh_high` | `[N]` | float32 | Upper bound for per-episode hunt-threshold sampling |
| `animal_lose_interest_low` | `[N]` | float32 | Lower bound for per-episode lose-interest-multiplier sampling |
| `animal_lose_interest_high` | `[N]` | float32 | Upper bound for per-episode lose-interest-multiplier sampling |
| `animal_classes_int` | `[N]` | int32 | Integer class code per animal: 0=predator, 1=neutral, … |
| `animal_behaviours_int` | `[N]` | int32 | Integer behaviour code per animal: 0=wander, 1=hunt, 2=static |
| `animal_is_damaging` | `[N]` | bool | Precomputed from class: True iff the animal can deal damage (predator-class) |
| `animal_visual_channel` | `[N]` | int32 | Visual sensor channel index per animal: 5=predator, 7=neutral, … |

#### Static animal label/index tuples (all `pytree_node=False`)

These tuples are derived from the YAML config at load time and never change within a run. They are static so they can serve as loop bounds and slice indices inside JIT'd code without forcing dynamic shapes.

| Field | Type | Static? | Description |
|-------|------|---------|-------------|
| `animal_classes` | `tuple[str, …]` len N | **yes** | Per-animal class string (e.g. `('predator', 'predator', 'neutral')`) |
| `animal_behaviours` | `tuple[str, …]` len N | **yes** | Per-animal behaviour string (e.g. `('hunt', 'hunt', 'wander')`) |
| `animal_tags` | `tuple[str, …]` len N | **yes** | Per-animal human-readable name tag (e.g. `('wolf_0', 'wolf_1', 'rabbit_0')`) |
| `hunt_idx` | `tuple[int, …]` len N_hunt | **yes** | Global indices of hunt-behaviour animals; used by `update_animals` to slice the hunt subset |
| `wander_idx` | `tuple[int, …]` len N_wander | **yes** | Global indices of wander-behaviour animals |
| `static_idx` | `tuple[int, …]` len N_static | **yes** | Global indices of static (non-moving) animals |
| `predator_indices` | `tuple[int, …]` len N_pred_class | **yes** | Global indices of predator-class animals; used at reset for per-class placement + property sampling |
| `neutral_indices` | `tuple[int, …]` len N_neutral_class | **yes** | Global indices of neutral-class animals |

#### Legacy `@property` aliases (kept for one release cycle, post-CP6)

`EnvParams` exposes two read-only Python properties for backward compatibility with consumers that pre-date the unified refactor:

- `params.predator_tags` → filters `animal_tags` by class == `'predator'`
- `params.neutral_tags` → filters `animal_tags` by class == `'neutral'`

These are derived on access; they are not stored struct fields. Consumers should migrate to `animal_tags + predator_indices / neutral_indices` directly.

### Obstacles

| Field | Shape | dtype | Static? | Description |
|-------|-------|-------|---------|-------------|
| `obs_blocking` | `[num_obs]` | bool | no | If True, agent is bounced back on collision |
| `obs_hides_agent` | `[num_obs]` | bool | no | If True and agent is on this cell, predators cannot detect the agent |
| `obs_spawn_area` | `[num_obs, 4]` | int32 | no | Spawn bounding box |
| `obs_damage` | `[num_obs, 2]` | float32 | no | Damage range on collision/overlap |
| `obs_property` | `[num_obs, vector_size]` | float32 | no | Olfactory chemical signature mean |
| `obs_property_std` | `[num_obs, vector_size]` | float32 | no | Olfactory chemical signature std dev |
| `obs_nociception` | `[num_obs]` | float32 | no | Nociception intensity on contact |
| `obs_type` | `[num_obs]` | int32 | no | Index into `obstacle_names` tuple for visual encoding |
| `obstacle_names` | `tuple[str, …]` | — | **yes** | Static tuple of unique obstacle name strings (e.g. `("bush", "rock")`) |

### Placement

| Field | Type | Static? | Description |
|-------|------|---------|-------------|
| `type_areas` | `[T, 4]` int32 | no | Spawn area per type group |
| `type_counts` | `[T]` int32 | no | Entity count per group |
| `type_entity_map` | `[T, max_per_type]` int32 | no | Global entity indices for each group |
| `max_per_type` | int | **yes** | Max entities in any single group (static loop bound for per-type placement) |
| `num_types` | int | **yes** | Number of type groups |
| `num_entities` | int | **yes** | Total entity count (res + animals + obs) |
| `placement_mode` | str | **yes** | `"per_entity"` or `"per_type"` |

### Body

| Field | Type | Static? | Description |
|-------|------|---------|-------------|
| `max_satiation` | float | no | Upper bound on satiation |
| `max_nutrition` | float | no | Upper bound on nutrition |
| `max_injury` | float | no | Upper bound on injury (death at this value) |
| `food_nutrition_gain` | float | no | Nutrition gained per food consumption event |
| `setpoint` | float | no | Target satiation for homeostatic drive |
| `start_satiation` | float | no | Loaded from YAML but **never read by `core.py`** — satiation at reset is always derived from nutrition (see FAQ) |
| `start_nutrition` | float | no | Initial nutrition if not randomised |
| `metabolic_cost` | float | no | Nutrition drained per step |
| `nutrition_to_satiation_scaling_factor` | float | no | Exponent `k` in `S = max_S × (N/max_N)^k` |
| `recovery_base_rate` | float | no | Base injury recovery amount per rest step |
| `recovery_accel_rate` | float | no | Exponential boost factor per rest streak step |
| `smoothing_duration` | int | **yes** | Length of the injury ring buffer (fixes `injury_buffer` shape) |
| `death_penalty` | float | no | Reward penalty applied on episode termination |
| `overeating_death` | bool | **yes** | If True, satiation ≥ max_satiation triggers death |
| `use_homeostatic_reward` | bool | **yes** | Toggles reward mode (see doc 06) |
| `with_satiation` | bool | **yes** | Enable satiation tracking |
| `with_nutrition` | bool | **yes** | Enable nutrition tracking and starvation |
| `with_injury` | bool | **yes** | Enable injury tracking; if False, any damage kills instantly |
| `random_start_satiation` | bool | **yes** | Loaded but unused — see `start_satiation` note |
| `random_start_nutrition` | bool | **yes** | Randomise starting nutrition (uniform in `[max/2, max]`) |
| `random_start_injury` | bool | **yes** | Randomise starting injury (uniform in `[0, max/2]`) |
| `random_start_pos` | bool | **yes** | Randomise agent start position |
| `start_pos` | `[2]` int32 | no | Fixed start position (used when `random_start_pos=False`) |
| `rest_action_enabled` | bool | **yes** | Adds action index 4 = Rest |
| `eat_action_enabled` | bool | **yes** | Adds action index 5 = Eat (otherwise eating is automatic on overlap) |
| `eating_nutrition_cost` | float | no | Nutrition cost deducted when eating |
| `eating_reward_penalty` | float | no | Reward penalty applied when eating |

### Sensors

| Field | Type | Static? | Description |
|-------|------|---------|-------------|
| `sensor_radius` | float | no | Olfactory detection radius |
| `sensor_decay` | float | no | Distance decay power for olfaction |
| `sensor_range` | int | **yes** | Collision sensor diamond radius |
| `visual_sensor_enabled` | bool | **yes** | Enable visual sensor |
| `visual_sensor_range` | int | **yes** | Visual sensor diamond radius |
| `local_view_size` | int | **yes** | Renderer local view window size |
| `olfactory_enabled` | bool | **yes** | Enable olfaction sensor |
| `nociception_enabled` | bool | **yes** | Enable exteroceptive nociception sensor |
| `location_sensor_enabled` | bool | **yes** | Enable location sensor |
| `injury_observable` | bool | **yes** | If False, raw injury level is removed from observation |
| `nutrition_observable` | bool | **yes** | If False, raw nutrition level is removed from observation |
| `interoceptive_nociception_enabled` | bool | **yes** | Enable interoceptive nociception sensor |
| `interoceptive_convolution_enabled` | bool | **yes** | If False, interoceptive sensor emits `injury/max` directly (no delay) |
| `interoceptive_kernel_length` | int | **yes** | Static length for `nociception_history_buffer` |
| `interoceptive_kernel` | `[K]` float32 | no | Normalized alpha kernel used for interoceptive convolution |
| `proprioception_enabled` | bool | **yes** | Enable proprioception sensor |
| `action_dim` | int | **yes** | Total number of actions (4 + rest_enabled + eat_enabled) |
| `olfactory_vector_size` | int | **yes** | Olfaction vector length (equals `res_property` width) |
| `nociception_size` | int | **yes** | Currently always 1 |

### Perceptual Noise

| Field | Type | Static? | Description |
|-------|------|---------|-------------|
| `perceptual_noise_enabled` | bool | **yes** | Enable noise application |
| `noise_modality_order` | tuple | **yes** | Ordered modality names from YAML key order; used to index noise arrays |
| `noise_modes` | `[13]` int32 | no | Per-modality noise mode: 0=None, 1=Constant, 2=State-Dependent |
| `noise_sigmas` | `[13]` float32 | no | Base standard deviation per modality |
| `noise_injury_scales` | `[13]` float32 | no | Injury-scaling factor α per modality (used in mode 2) |
| `noise_clip_min` | `[13]` float32 | no | Per-modality observation lower bound after noise |
| `noise_clip_max` | `[13]` float32 | no | Per-modality observation upper bound after noise |

---

## Pytree Registration & Flax struct.dataclass

Flax `@struct.dataclass` automatically registers both classes as JAX pytrees. This means:

- `jax.jit`, `jax.vmap`, `jax.lax.scan` can accept them as inputs/outputs without manual tree registration.
- **`pytree_node=False` fields** are treated as **static**. JAX traces a new compiled function any time these values change. They are safe to use in Python-level `if` statements inside JIT.
- **All other fields** are pytree leaves; their values can change freely without recompilation.
- **Immutability**: Flax structs are immutable after creation. To produce an updated state, call `state.replace(field=new_value)` (also exposed as `state._replace(...)` for API compatibility). This is zero-copy when possible under XLA.
- **`vmap` batching**: `ParallelEnv` vmaps over the leading dimension of `EnvState` arrays. Each env in the batch has its own independent state pytree — conceptually a batch of `EnvState` objects stacked along axis 0.

---

## Reset Values (What's in `EnvState` at step 0)

Set in `core.py:980-1012` during `jax_reset`. Unified animal fields are assembled in predators-first order (N1 fix). Behavioural params are sampled with a key derived via `fold_in` so existing PRNG streams are unchanged (N2/N3 fix).

| Field | Value at reset | Source |
|-------|----------------|--------|
| `agent_pos` | `start_pos` **or** uniform random cell (if `random_start_pos=True`) | `core.py:786-787` |
| `current_step` | `0` | `core.py:982` |
| `last_action` | `4` if `rest_action_enabled` else `5` | `core.py:1011` |
| `res_active` | all `True` | `core.py:984` |
| `res_cons_count` | all `0` | `core.py:985` |
| `res_reg_timer` | all `0` | `core.py:986` |
| `animal_state` | all `0` (Patrol/Idle) | `core.py:990` |
| `animal_stamina` | `animal_max_stamina_sampled[i]` (full) for each animal | `core.py:991` |
| `animal_move_timer` | all `0` | `core.py:992` |
| `animal_attack_timer` | all `0` | `core.py:993` |
| `animal_detect_sampled` | `Uniform(detect_low[i], detect_high[i])` per animal | `core.py:963-964` |
| `animal_max_stamina_sampled` | `Uniform(max_stamina_low[i], max_stamina_high[i])` per animal | `core.py:965-966` |
| `animal_recovery_sampled` | `Uniform(recovery_low[i], recovery_high[i])` per animal | `core.py:967-968` |
| `animal_hunt_thresh_sampled` | `Uniform(hunt_thresh_low[i], hunt_thresh_high[i])` per animal | `core.py:969-970` |
| `animal_lose_interest_sampled` | `Uniform(lose_interest_low[i], lose_interest_high[i])` per animal | `core.py:971-972` |
| `nutrition` | `start_nutrition` **or** `Uniform(max_nutrition/2, max_nutrition)` if `random_start_nutrition=True` | `core.py:909-913` |
| `satiation` | **always derived** from nutrition: `max_satiation × (nutrition/max_nutrition)^k` | `core.py:915-916` |
| `injury_level` | `0.0` **or** `Uniform(0, max_injury/2)` if `random_start_injury=True` | `core.py:918-922` |
| `injury_buffer` | zeros of length `smoothing_duration` | `core.py:924` |
| `nociception_history_buffer` | zeros of length `interoceptive_kernel_length` | `core.py:925` |
| `last_collision_noc` | `0.0` | `core.py:1007` |
| `rest_streak` | `0` | `core.py:1008` |
| `terminated` | `False` | `core.py:1009` |
| `res_property_sampled` | `clip(mean + std × N(0,1), 0, 1)` — per resource | `core.py:934` |
| `obs_property_sampled` | `clip(mean + std × N(0,1), 0, 1)` — per obstacle | `core.py:935` |
| `animal_property_sampled` | `clip(mean + std × N(0,1), 0, 1)` — predator-class and neutral-class sampled with independent PRNG keys then scattered into unified array | `core.py:941-953` |

---

## Clarifications / FAQ

**Q: What happened to `pred_pos`, `pred_state`, `pred_stamina`, `neutral_pos`, `neutral_move_timer`, and the old `pred_*` / `neutral_*` param fields?**
A: They were removed in the CP1–CP6 "unified animal entity" refactor. All animal state is now in the `animal_*` array family. Use `params.predator_indices` / `params.neutral_indices` to recover the predator/neutral subsets. `params.predator_tags` and `params.neutral_tags` are temporary backward-compatibility properties that will be removed after one release cycle.

**Q: Why do entities have both `animal_property` (in params) and `animal_property_sampled` (in state)?**
A: `animal_property` is the configured **mean** chemical signature (constant across episodes); `animal_property_std` is the standard deviation. At reset, each entity samples `sampled = clip(mean + std × N(0,1), 0, 1)` (`core.py:930-932`). Predator-class and neutral-class are sampled with separate PRNG keys (N2 fix) so their noise is independent. The sampled vector is what sensors read.

**Q: The state tables show `vector_size`. Is that hardcoded to 5?**
A: `5` is whatever shape your YAML `property: [v1, v2, v3, v4, v5]` has; `olfactory_vector_size` is derived from the config. All entities in one config must share the same vector length. In practice every shipped config uses 5, hence the comment shorthand.

**Q: Are `start_satiation` and `random_start_satiation` actually used?**
A: **No.** They are loaded into `EnvParams` by `config_loader.py` but never read by `core.py`. Satiation at reset is always derived from nutrition via `S = max_S × (N/max_N)^k` (`core.py:915-916`). These fields are legacy — leaving them in YAML has no effect. Control starting satiation indirectly through `start_nutrition` / `random_start_nutrition`.

**Q: Why is `noise_*` shape `[13]` when there are only ~10 modalities?**
A: The 13 is a fixed padding size (`config_loader.py:401`: `pad = max(0, 13 - len(noise_modality_order))`). Active modalities fill the leading slots per `noise_modality_order`; remaining slots are zero-padded so the array shape stays static under JIT. Indexing must always go through `noise_modality_order` or `modality_map` — raw index positions are not semantically meaningful beyond the order your YAML declared.

**Q: How is resource respawn timing controlled?**
A: When a resource is consumed, `res_active` flips to `False` and `res_reg_timer` is set to `res_reg_delay`. Each step, inactive resources with `res_reg_timer > 0` count down (`core.py:115-123`). When the timer hits 0, `respawn_mask` fires: `res_active` flips back to `True` and a fresh `res_property_sampled` is drawn. If `res_max_cons` has been exhausted, the resource stays inactive permanently (set `res_max_cons = -1` for unlimited respawns).

**Q: What are the integer values for `last_action`?**
A: `0–3` = movement (Up, Down, Left, Right), `4` = Rest (only if `rest_action_enabled`), `5` = Eat (only if `eat_action_enabled`). `action_dim = 4 + rest_enabled + eat_enabled` (`config_loader.py:330`). The proprioception sensor one-hot-encodes this index. At reset, `last_action = 4` if `rest_action_enabled`, else `5` (`core.py:1011`).

**Q: Does `terminated=True` get reset automatically?**
A: Only on an explicit call to `jax_reset` — the step function itself never flips `terminated` back. In `ParallelEnv`, the wrapper detects `terminated=True` and calls `reset` on those environments (see doc `11_parallel_env_wrapper.md`).

**Q: When does `injury_buffer` get populated?**
A: Whenever damage is applied in a step, the raw damage value is distributed across `smoothing_duration` slots of the ring buffer (a uniform-spread injection). Each step, `injury_buffer[0]` is consumed into `injury_level` and the buffer rotates. This models "pain over time" — a single hit slowly increases injury for `smoothing_duration` steps. Full mechanics in doc `05_body_homeostasis.md`.

**Q: What's the difference between `type_areas` and `grid_location_type`?**
A: They serve orthogonal purposes:
- `grid_location_type [H, W]` encodes **terrain** per cell (plain / grass / sand) — read by the location sensor and renderer.
- `type_areas [T, 4]` encodes **spawn region bounding boxes per entity type-group** — used only during entity placement at reset. It has no runtime effect after step 0.

**Q: What is `max_per_type` used for?**
A: It's the max entity count across any single type-group. Used as a **static loop bound** for the per-type placement pass (`per_type` mode). Making it static (`pytree_node=False`) means JAX compiles the loop once; changing `max_per_type` triggers recompilation.

**Q: If `with_injury=False`, what happens on damage?**
A: The injury system is bypassed entirely and any nonzero damage terminates the episode immediately (no smoothing, no recovery). This is used in configs where the task is "touch-nothing-bad" survival, not pain modelling.

**Q: Does `overeating_death=True` kill at exactly `max_satiation` or above?**
A: At `satiation >= max_satiation` (i.e. equality triggers death). See `05_body_homeostasis.md`.

**Q: Is `EnvParams` immutable across an entire training run, or per-episode?**
A: Per-episode in principle, but most training setups reuse the same `EnvParams` for every episode in a run (it's built once from the YAML). Changing any static field (`pytree_node=False`) mid-run triggers JIT recompilation. Changing a non-static field (e.g. `max_nutrition`) is free at runtime but uncommon — the usual pattern is to rebuild `EnvParams` only for a new experiment.

**Q: Can I rely on `._replace(...)` vs `.replace(...)`?**
A: Yes — both work identically. The `._replace` alias exists for compatibility with earlier code that used `namedtuple._replace` semantics. Use either.

**Q: How do the per-episode behavioural sampling ranges work for animals?**
A: For each of the five distributional fields (detect, max_stamina, recovery, hunt_thresh, lose_interest), `EnvParams` stores a `*_low` and `*_high` array of length N. At every reset, `jax.random.uniform(key, (N,), minval=low, maxval=max(high, low))` produces this episode's sampled value for each animal. For legacy scalar configs, low == high, so the draw is deterministic (always the constant). For wander/static animals, both bounds are 0.0, so the sampled fields are always 0.0 — they are kept at shape `[N]` only for shape stability under JIT (`core.py:957-978`).

**Q: What is the `animal_episode_key` and why is it derived via `fold_in`?**
A: The five per-episode behavioural draws use a key derived as `jax.random.fold_in(property_key, 0xAE1)` (`core.py:783`). Using `fold_in` rather than adding a sixth split to the outer split means the existing PRNG streams (agent_key, placement_key, body_key, property_key) are byte-identical to the pre-refactor code — backward PRNG parity is preserved.
