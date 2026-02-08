"""
Configuration loader for JAX Environment.

Translates YAML config files into JAX-compatible EnvParams.
"""
import yaml
import jax.numpy as jnp
from src.environment.state import EnvParams

from src.utils.config import Config

def load_env_params(config: Config) -> EnvParams:
    """Loads environment parameters from a Config object with strict retrieval."""
    
    # Build resource arrays
    raw_resources = config.get('environment.resources', [])
    expanded_resources = []
    if raw_resources:
        for r in raw_resources:
            count = r.get('count', 1) 
            for _ in range(count):
                expanded_resources.append(r)
    
    if expanded_resources:
        def r_get(r, key):
            val = r.get(key)
            if val is None: raise ValueError(f"Strict Config: Resource field '{key}' is required.")
            return val

        res_type = jnp.array([0 if r_get(r, 'type') == 'food' else 1 for r in expanded_resources], dtype=jnp.int32)
        res_property = jnp.array([r_get(r, 'properties') for r in expanded_resources])
        res_spawn_area = jnp.array([[*r_get(r, 'spawn_area')[0], *r_get(r, 'spawn_area')[1]] for r in expanded_resources])
        res_max_cons = jnp.array([r_get(r, 'max_consumption') for r in expanded_resources], dtype=jnp.int32)
        res_reg_delay = jnp.array([r_get(r, 'regeneration_delay') for r in expanded_resources], dtype=jnp.int32)
        res_damage = jnp.array([r_get(r, 'damage') for r in expanded_resources])
    else:
        res_type = jnp.zeros(0, dtype=jnp.int32)
        res_property = jnp.zeros((0, 5))
        res_spawn_area = jnp.zeros((0, 4))
        res_max_cons = jnp.zeros(0, dtype=jnp.int32)
        res_reg_delay = jnp.zeros(0, dtype=jnp.int32)
        res_damage = jnp.zeros(0)

    # Build predator arrays
    predators = config.get('environment.predators', [])
    if predators:
        def p_get(p, key):
            val = p.get(key)
            if val is None: raise ValueError(f"Strict Config: Predator field '{key}' is required.")
            return val

        pred_property = jnp.array([p_get(p, 'property') for p in predators])
        pred_move_int = jnp.array([p_get(p, 'move_interval') for p in predators], dtype=jnp.int32)
        pred_damage = jnp.array([p_get(p, 'damage') for p in predators])
        h = config.get_mandatory('environment.height')
        w = config.get_mandatory('environment.width')
        pred_patrol = jnp.array([[*p.get('patrol_area', [[0,0],[h-1,w-1]])[0],
                                  *p.get('patrol_area', [[0,0],[h-1,w-1]])[1]] for p in predators])
        pred_detect = jnp.array([p_get(p, 'detection_range') for p in predators])
        pred_max_stamina = jnp.array([p_get(p, 'max_stamina') for p in predators])
        pred_recovery = jnp.array([p_get(p, 'stamina_recovery_rate') for p in predators])
        pred_hunt_thresh = jnp.array([p_get(p, 'hunt_stamina_threshold') for p in predators])
    else:
        pred_property = jnp.zeros((0, 5))
        pred_move_int = jnp.zeros(0, dtype=jnp.int32)
        pred_damage = jnp.zeros(0)
        pred_patrol = jnp.zeros((0, 4))
        pred_detect = jnp.zeros(0)
        pred_max_stamina = jnp.zeros(0)
        pred_recovery = jnp.zeros(0)
        pred_hunt_thresh = jnp.zeros(0)
    
    # Build Obstacle arrays
    raw_obstacles = config.get('environment.obstacles', [])
    expanded_obstacles = []
    for o in raw_obstacles:
        count = o.get('count', 1)
        for _ in range(count):
            expanded_obstacles.append(o)
            
    if expanded_obstacles:
        def obs_get(o, key):
            val = o.get(key)
            if val is None: raise ValueError(f"Strict Config: Obstacle field '{key}' is required.")
            return val
        obs_blocking = jnp.array([o.get('blocking', True) for o in expanded_obstacles], dtype=jnp.bool_)
        obs_spawn_area = jnp.array([[*obs_get(o, 'area')[0], *obs_get(o, 'area')[1]] for o in expanded_obstacles])
    else:
        obs_blocking = jnp.zeros(0, dtype=jnp.bool_)
        obs_spawn_area = jnp.zeros((0, 4))

    # Build Grid Location Types
    import numpy as np
    height = config.get_mandatory('environment.height')
    width = config.get_mandatory('environment.width')
    grid_np = np.zeros((height, width), dtype=np.int32)
    location_areas = config.get('environment.location_areas', [])
    for area_config in location_areas:
        a_type = area_config.get('type')
        type_idx = 1 if a_type == 'grass' else 2 if a_type == 'sand' else 0
        area = area_config.get('area')
        if area:
            r1, c1, r2, c2 = area[0][0], area[0][1], area[1][0], area[1][1]
            grid_np[r1:r2+1, c1:c2+1] = type_idx
    grid_location_type = jnp.array(grid_np)
    
    return EnvParams(
        height=height,
        width=width,
        max_steps=config.get_mandatory('environment.max_steps'),
        grid_location_type=grid_location_type,
        res_type=res_type,
        res_property=res_property,
        res_spawn_area=res_spawn_area,
        res_max_cons=res_max_cons,
        res_reg_delay=res_reg_delay,
        res_damage=res_damage,
        pred_property=pred_property,
        pred_move_int=pred_move_int,
        pred_damage=pred_damage,
        pred_patrol=pred_patrol,
        pred_detect=pred_detect,
        pred_max_stamina=pred_max_stamina,
        pred_recovery=pred_recovery,
        pred_hunt_thresh=pred_hunt_thresh,
        obs_blocking=obs_blocking,
        obs_spawn_area=obs_spawn_area,
        max_satiation=config.get_mandatory('body.max_satiation'),
        max_injury=config.get_mandatory('body.max_injury'),
        food_gain=config.get_mandatory('body.food_satiation_gain'),
        setpoint=config.get_mandatory('body.satiation_setpoint'),
        start_satiation=config.get_mandatory('body.start_satiation'),
        injury_recovery=config.get_mandatory('body.injury_recovery'),
        smoothing_duration=config.get_mandatory('body.injury_smoothing_duration'),
        death_penalty=config.get_mandatory('body.death_penalty'),
        overeating_death=config.get_mandatory('body.overeating_death'),
        use_homeostatic_reward=config.get_mandatory('body.use_homeostatic_reward'),
        with_satiation=config.get_mandatory('body.with_satiation'),
        with_injury=config.get_mandatory('body.with_injury'),
        random_start_satiation=config.get_mandatory('body.random_start_satiation'),
        random_start_injury=config.get_mandatory('body.random_start_injury'),
        rest_action_enabled=config.get_mandatory('environment.rest_action_enabled'),
        eat_action_enabled=config.get_mandatory('environment.eat_action_enabled'),
        sensor_radius=config.get_mandatory('sensory.sensor_radius'),
        sensor_decay=config.get_mandatory('sensory.decay_power'),
        sensor_range=config.get_mandatory('sensory.collision_sensor_range'),
        visual_sensor_enabled=config.get('sensory.visual_sensor_enabled', False),
        visual_sensor_range=config.get('sensory.visual_sensor_range', 0),
        local_view_size=config.get('visualization.local_view_size', 9)
    )

    
    return params
