import jax.numpy as jnp
import jax
import numpy as np
from flax import struct


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

@struct.dataclass
class EnvState:
    # Agent
    agent_pos: jnp.ndarray      # [2] (row, col)
    current_step: jnp.ndarray   # []

    # Resources
    res_pos: jnp.ndarray        # [num_res, 2]
    res_active: jnp.ndarray     # [num_res] bool
    res_allocated: jnp.ndarray  # [num_res] bool — set once at reset, never mutated; gates respawn
    res_cons_count: jnp.ndarray # [num_res] int
    res_reg_timer: jnp.ndarray  # [num_res] int
    res_property_sampled: jnp.ndarray # [num_res, vector_size]
    res_visual_property_sampled: jnp.ndarray  # [num_res, visual_vector_size] (per-episode sampled)

    # Animals (unified — predators + neutrals, predators-first ordering)
    animal_pos: jnp.ndarray              # [N, 2]
    animal_state: jnp.ndarray            # [N] int (PATROL=0, HUNT=1, RETURN=2; 0 for wander/static)
    animal_stamina: jnp.ndarray          # [N] float (unused for wander/static, kept for shape stability)
    animal_move_timer: jnp.ndarray       # [N] int
    animal_attack_timer: jnp.ndarray     # [N] int (zero for non-hunt entities)
    animal_property_sampled: jnp.ndarray # [N, vector_size]
    animal_visual_property_sampled: jnp.ndarray  # [N, visual_vector_size] (per-episode sampled)
    # Per-episode-sampled behavioural params (NEW — all five float fields, degenerate [s,s] for legacy configs)
    animal_detect_sampled: jnp.ndarray         # [N] float
    animal_max_stamina_sampled: jnp.ndarray    # [N] float
    animal_recovery_sampled: jnp.ndarray       # [N] float
    animal_hunt_thresh_sampled: jnp.ndarray    # [N] float
    animal_lose_interest_sampled: jnp.ndarray  # [N] float
    # Per-episode-sampled integer timer params (NEW — move_interval + attack_delay distributional sampling)
    animal_move_int_sampled: jnp.ndarray       # [N] int  (sampled from [move_int_low, move_int_high])
    animal_attack_delay_sampled: jnp.ndarray   # [N] int  (sampled from [attack_delay_low, attack_delay_high])
    # Jump/pounce feature (predator lunge attack) — sampled from an INDEPENDENT
    # fold_in key at reset (NOT part of the size-7 ep_keys split). 0 = jump disabled.
    animal_attack_range_sampled: jnp.ndarray   # [N] float (sampled from [attack_range_low, attack_range_high])

    # Animals — per-episode activation mask (NEW — per-episode count-range feature)
    animal_active: jnp.ndarray  # [N] bool; False for inactive (parked off-grid) slots

    # Obstacles
    obs_pos: jnp.ndarray        # [num_obs, 2]
    obs_property_sampled: jnp.ndarray # [num_obs, vector_size]
    obs_visual_property_sampled: jnp.ndarray  # [num_obs, visual_vector_size] (per-episode sampled)
    # Per-episode activation mask (NEW — per-episode count-range feature)
    obs_active: jnp.ndarray     # [num_obs] bool; False for inactive (parked off-grid) slots

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
    res_visual_property: jnp.ndarray  # [num_res, visual_vector_size] (visual appearance vector)
    res_visual_property_std: jnp.ndarray  # [num_res, visual_vector_size] (per-episode visual jitter std; zeros = deterministic)
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
    # Per-episode bounds (low == high for legacy scalar configs).
    # animal_detect_low/high are inclusive-integer randint bounds (int); the four
    # siblings below are float-uniform bounds.
    animal_detect_low: jnp.ndarray         # [N] int — inclusive-integer HUNT-trigger range, low bound
    animal_detect_high: jnp.ndarray        # [N] int — inclusive-integer HUNT-trigger range, high bound
    animal_max_stamina_low: jnp.ndarray    # [N]
    animal_max_stamina_high: jnp.ndarray   # [N]
    animal_recovery_low: jnp.ndarray       # [N]
    animal_recovery_high: jnp.ndarray      # [N]
    animal_hunt_thresh_low: jnp.ndarray    # [N]
    animal_hunt_thresh_high: jnp.ndarray   # [N]
    animal_lose_interest_low: jnp.ndarray  # [N]
    animal_lose_interest_high: jnp.ndarray # [N]
    # Per-episode integer randint bounds for move_interval and attack_delay
    # (low == high for legacy scalar configs → randint([s, s+1)) always yields s)
    animal_move_int_low: jnp.ndarray       # [N] int
    animal_move_int_high: jnp.ndarray      # [N] int
    animal_attack_delay_low: jnp.ndarray   # [N] int
    animal_attack_delay_high: jnp.ndarray  # [N] int
    # Jump/pounce feature (predator lunge attack) — OPTIONAL, default [0,0]/0.0 =
    # disabled. See docs/develop/active/env_entities/PREDATOR_JUMP_MECHANISM.md.
    animal_attack_range_low: jnp.ndarray      # [N] int — inclusive-integer jump-trigger Manhattan range, low bound
    animal_attack_range_high: jnp.ndarray     # [N] int — inclusive-integer jump-trigger Manhattan range, high bound
    animal_attack_success_rate: jnp.ndarray   # [N] float in [0,1] — P(fired jump lands on agent); NOT per-episode sampled
    # Per-entity int-coded class/behaviour (for damage masking and visual channel)
    animal_classes_int: jnp.ndarray        # [N] int (0=predator, 1=neutral, ...)
    animal_behaviours_int: jnp.ndarray     # [N] int (0=wander, 1=hunt, 2=static)
    animal_is_damaging: jnp.ndarray        # [N] bool (precomputed from class)
    animal_disengage_on_contact: jnp.ndarray  # [N] bool (opt-in: drain stamina→0 on agent contact)
    animal_visual_channel: jnp.ndarray     # [N] int (5=predator, 7=neutral, ...)
    animal_visual_property: jnp.ndarray   # [N, visual_vector_size] (visual appearance vector)
    animal_visual_property_std: jnp.ndarray  # [N, visual_vector_size] (per-episode visual jitter std; zeros = deterministic)
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

    # Per-episode count-range fields (NEW — per-episode count-range feature).
    # For each config entry, stores the [low, high] bounds for K-sampling at reset.
    # Slot→entry maps: res_entry_id[i] = index of the config entry that allocated slot i.
    # Degenerate entries (count_low == count_high) never trigger a K-draw at reset;
    # the activation mask is all-True (byte-identical to pre-feature behaviour).
    res_count_low: jnp.ndarray        # [num_res_entries] int32 — per-entry lower bound
    res_count_high: jnp.ndarray       # [num_res_entries] int32 — per-entry upper bound (= alloc size)
    res_entry_id: jnp.ndarray         # [num_res] int32 — slot → entry index
    animal_count_low: jnp.ndarray     # [num_animal_entries] int32
    animal_count_high: jnp.ndarray    # [num_animal_entries] int32
    animal_entry_id: jnp.ndarray      # [N] int32 — slot → entry index
    obs_count_low: jnp.ndarray        # [num_obs_entries] int32
    obs_count_high: jnp.ndarray       # [num_obs_entries] int32
    obs_entry_id: jnp.ndarray         # [num_obs] int32 — slot → entry index
    # Whether any entry in each class has a genuine range (low < high).
    # When all entries are degenerate, the K-draw is skipped entirely so the
    # PRNG stream is byte-identical to pre-feature code (parity guard).
    has_res_range: bool = struct.field(pytree_node=False)
    has_animal_range: bool = struct.field(pytree_node=False)
    has_obs_range: bool = struct.field(pytree_node=False)
    # True iff any animal has attack_range_high > 0 (jump/pounce feature enabled).
    # Guards the jump block in `_hunt_step` — a False value is a provable no-op
    # (no extra jax.random.split, no extra draws) for backward-compat.
    has_attack_feature: bool = struct.field(pytree_node=False)

    # Obstacles
    obs_blocking: jnp.ndarray      # [num_obs] bool
    obs_hides_agent: jnp.ndarray   # [num_obs] bool (bush-type concealment)
    obs_blocks_animals: jnp.ndarray # [num_obs] bool (blocks animal movement; agent still enters)
    obs_spawn_area: jnp.ndarray # [num_obs, 4] (min_r, min_c, max_r, max_c)
    obs_damage: jnp.ndarray     # [num_obs, 2] [min, max]
    obs_property: jnp.ndarray   # [num_obs, vector_size]
    obs_property_std: jnp.ndarray # [num_obs, vector_size]
    obs_visual_property: jnp.ndarray  # [num_obs, visual_vector_size] (visual appearance vector)
    obs_visual_property_std: jnp.ndarray  # [num_obs, visual_vector_size] (per-episode visual jitter std; zeros = deterministic)
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
    start_satiation: float              # For non-random start (note: dead in reset path; satiation derived from nutrition)
    start_nutrition: float
    start_nutrition_low: float          # lower bound when random_start_nutrition
    start_nutrition_high: float         # upper bound when random_start_nutrition
    start_injury_low: float             # lower bound when random_start_injury
    start_injury_high: float            # upper bound when random_start_injury
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

    # --- DIRECTIONAL_SENSORS -------------------------------------------
    # Shape-determining -> static. 0 reproduces the pre-DIRECTIONAL_SENSORS single-point sample.
    olfactory_grid_range: int = struct.field(pytree_node=False)
    # Selects the visual code path at trace time. False == exact cell match.
    visual_blur_enabled: bool = struct.field(pytree_node=False)
    # Continuous blur knobs are TRACED: sweeping them must not recompile jax_step.
    visual_blur_radial_scale: float   # sigma_par = max(scale*d, floor) -- the floor CAPS
                                      # effective anisotropy at short range (rho=1 at d=1)
    visual_blur_anisotropy: float     # rho = sigma_par / sigma_perp; 1.0 == isotropic
    visual_blur_sigma_floor: float    # cells; the grid's sampling limit
    # Per-entity visibility: 0 = none, 1 = far (visible only when co-located), 2 = all
    res_visual_mask: jnp.ndarray      # [num_res]    int32
    animal_visual_mask: jnp.ndarray   # [N]          int32
    obs_visual_mask: jnp.ndarray      # [num_obs]    int32

    # How per-cell entity contributions combine: "sum" (a count / weighted sum,
    # pre-DIRECTIONAL_SENSORS behaviour) or "clamp" (per-channel presence, capped at 1.0).
    visual_value_mode: str = struct.field(pytree_node=False)
    # Line-of-sight occlusion: a nearer sight-blocking entity inside the shadow
    # cone of the ray agent->entity hides it.
    visual_occlusion_enabled: bool = struct.field(pytree_node=False)
    visual_occlusion_cos: float       # cos(cone half-angle); traced, sweepable
    visual_occlusion_strength: float  # 1.0 = fully hidden, <1.0 = attenuated
    res_blocks_sight: jnp.ndarray     # [num_res]    bool
    animal_blocks_sight: jnp.ndarray  # [N]          bool
    obs_blocks_sight: jnp.ndarray     # [num_obs]    bool

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
    visual_vector_size: int = struct.field(pytree_node=False)  # width of visual property vectors (V); shape-determining, not traced
    visual_background_property: jnp.ndarray  # [3, visual_vector_size] (grass/sand/plain rows)
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
