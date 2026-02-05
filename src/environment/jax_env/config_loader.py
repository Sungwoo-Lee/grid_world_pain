"""
Configuration loader for JAX Environment.

Translates YAML config files into JAX-compatible EnvParams.
"""
import yaml
import jax.numpy as jnp
from src.environment.jax_env.state import EnvParams

from src.utils.config import Config

def load_env_params(config: Config) -> EnvParams:
    """Loads environment parameters from a Config object with strict retrieval."""
    
    # Build resource arrays
    resources = config.get('environment.resources', [])
    if resources:
        res_type = jnp.array([0 if r.get('type') == 'food' else 1 for r in resources], dtype=jnp.int32)
        # Note: 'property' in YAML vs 'res_property' in JAX
        res_property = jnp.array([r.get('properties', [0,0,0,0,0]) for r in resources])
        res_spawn_area = jnp.array([[*r.get('spawn_area', [[0,0],[10,10]])[0], 
                                     *r.get('spawn_area', [[0,0],[10,10]])[1]] for r in resources])
        res_max_cons = jnp.array([r.get('max_consumption', 999) for r in resources], dtype=jnp.int32)
        res_reg_delay = jnp.array([r.get('regeneration_delay', 0) for r in resources], dtype=jnp.int32)
        res_damage = jnp.array([r.get('damage', 0.0) for r in resources])
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
        pred_property = jnp.array([p.get('property', [0,0,0,0,0]) for p in predators])
        pred_move_int = jnp.array([p.get('move_interval', 1) for p in predators], dtype=jnp.int32)
        pred_damage = jnp.array([p.get('damage', 1.0) for p in predators])
        pred_patrol = jnp.array([[*p.get('patrol_area', [[0,0],[10,10]])[0],
                                  *p.get('patrol_area', [[0,0],[10,10]])[1]] for p in predators])
        pred_detect = jnp.array([p.get('detection_range', 5.0) for p in predators])
        pred_max_stamina = jnp.array([p.get('max_stamina', 100.0) for p in predators])
        pred_recovery = jnp.array([p.get('stamina_recovery_rate', 1.0) for p in predators])
        pred_hunt_thresh = jnp.array([p.get('hunt_stamina_threshold', 0.0) for p in predators])
    else:
        pred_property = jnp.zeros((0, 5))
        pred_move_int = jnp.zeros(0, dtype=jnp.int32)
        pred_damage = jnp.zeros(0)
        pred_patrol = jnp.zeros((0, 4))
        pred_detect = jnp.zeros(0)
        pred_max_stamina = jnp.zeros(0)
        pred_recovery = jnp.zeros(0)
        pred_hunt_thresh = jnp.zeros(0)
    
    return EnvParams(
        height=config.get_mandatory('environment.height'),
        width=config.get_mandatory('environment.width'),
        max_steps=config.get_mandatory('environment.max_steps'),
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
        sensor_range=config.get_mandatory('sensory.collision_sensor_range')
    )

    
    return params
