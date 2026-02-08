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
    
    # Predators
    pred_pos: jnp.ndarray       # [num_pred, 2]
    pred_state: jnp.ndarray      # [num_pred] int
    pred_stamina: jnp.ndarray    # [num_pred] float
    pred_move_timer: jnp.ndarray # [num_pred] int

    # Obstacles
    obs_pos: jnp.ndarray        # [num_obs, 2]
    
    # Body
    satiation: jnp.ndarray       # [] float
    injury_level: jnp.ndarray    # [] float
    injury_buffer: jnp.ndarray   # [smoothing_duration] float
    
    # Environment status
    terminated: jnp.ndarray      # bool
    
    # Random State
    key: jax.random.PRNGKey      # PRNGKey

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
    res_type: jnp.ndarray       # [num_res] int (0:food, 1:danger)
    res_property: jnp.ndarray   # [num_res, vector_size]
    res_spawn_area: jnp.ndarray # [num_res, 4] (min_r, min_c, max_r, max_c)
    res_max_cons: jnp.ndarray   # [num_res]
    res_reg_delay: jnp.ndarray  # [num_res]
    res_damage: jnp.ndarray     # [num_res]
    
    # Predators (Constant attributes)
    pred_property: jnp.ndarray  # [num_pred, vector_size]
    pred_move_int: jnp.ndarray  # [num_pred]
    pred_damage: jnp.ndarray    # [num_pred]
    pred_patrol: jnp.ndarray    # [num_pred, 4] (min_r, min_c, max_r, max_c)
    pred_detect: jnp.ndarray    # [num_pred]
    pred_max_stamina: jnp.ndarray
    pred_recovery: jnp.ndarray
    pred_hunt_thresh: jnp.ndarray

    # Obstacles
    obs_blocking: jnp.ndarray   # [num_obs] bool
    obs_spawn_area: jnp.ndarray # [num_obs, 4] (min_r, min_c, max_r, max_c)
    
    # Body
    max_satiation: float
    max_injury: float
    food_gain: float
    setpoint: float
    start_satiation: float              # For non-random start
    injury_recovery: float
    smoothing_duration: int = struct.field(pytree_node=False)
    death_penalty: float
    overeating_death: bool = struct.field(pytree_node=False)
    use_homeostatic_reward: bool = struct.field(pytree_node=False)
    with_satiation: bool = struct.field(pytree_node=False)
    with_injury: bool = struct.field(pytree_node=False)
    random_start_satiation: bool = struct.field(pytree_node=False)
    random_start_injury: bool = struct.field(pytree_node=False)
    rest_action_enabled: bool = struct.field(pytree_node=False)
    eat_action_enabled: bool = struct.field(pytree_node=False)

    
    # Sensory
    sensor_radius: float
    sensor_decay: float
    sensor_range: int = struct.field(pytree_node=False)
    visual_sensor_enabled: bool = struct.field(pytree_node=False)
    visual_sensor_range: int = struct.field(pytree_node=False)
    local_view_size: int = struct.field(pytree_node=False)

    def _replace(self, **kwargs):
        return self.replace(**kwargs)
