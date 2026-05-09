import jax.numpy as jnp
import jax
from flax import struct

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
    
    # Predators
    pred_pos: jnp.ndarray       # [num_pred, 2]
    pred_state: jnp.ndarray      # [num_pred] int
    pred_stamina: jnp.ndarray    # [num_pred] float
    pred_move_timer: jnp.ndarray # [num_pred] int
    pred_attack_timer: jnp.ndarray # [num_pred] int
    pred_property_sampled: jnp.ndarray # [num_pred, vector_size]
    
    # Neutral Animals (Olfactory Decoys)
    neutral_pos: jnp.ndarray     # [num_neutral, 2]
    neutral_move_timer: jnp.ndarray # [num_neutral] int
    neutral_property_sampled: jnp.ndarray # [num_neutral, vector_size]

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
    
    # Predators (Constant attributes)
    pred_property: jnp.ndarray  # [num_pred, vector_size]
    pred_property_std: jnp.ndarray # [num_pred, vector_size]
    pred_nociception: jnp.ndarray # [num_pred]
    pred_move_int: jnp.ndarray  # [num_pred]
    pred_damage: jnp.ndarray    # [num_pred, 2] [min, max]
    pred_patrol: jnp.ndarray    # [num_pred, 4] (min_r, min_c, max_r, max_c)
    pred_detect: jnp.ndarray    # [num_pred]
    pred_max_stamina: jnp.ndarray
    pred_recovery: jnp.ndarray
    pred_hunt_thresh: jnp.ndarray
    pred_attack_delay: jnp.ndarray
    pred_lose_interest_mult: jnp.ndarray # [num_pred]
    predator_enabled: bool = struct.field(pytree_node=False)
    pred_spawn_area: jnp.ndarray # [num_pred, 4]
    predator_tags: tuple = struct.field(pytree_node=False)  # static metric-label tags, len = num_pred


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
    
    # Neutral Animals
    neutral_property: jnp.ndarray   # [num_neutral, vector_size]
    neutral_property_std: jnp.ndarray # [num_neutral, vector_size]
    neutral_nociception: jnp.ndarray # [num_neutral]
    neutral_move_int: jnp.ndarray    # [num_neutral]
    neutral_patrol: jnp.ndarray      # [num_neutral, 4]
    neutral_spawn_area: jnp.ndarray  # [num_neutral, 4]
    neutral_tags: tuple = struct.field(pytree_node=False)  # static metric-label tags, len = num_neutral

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

    def _replace(self, **kwargs):
        return self.replace(**kwargs)
