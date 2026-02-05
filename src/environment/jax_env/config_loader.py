"""
Configuration loader for JAX Environment.

Translates YAML config files into JAX-compatible EnvParams.
"""
import yaml
import jax.numpy as jnp
from src.environment.jax_env.state import EnvParams

def load_env_params(config_path: str) -> EnvParams:
    """Loads environment parameters from a YAML config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    env_cfg = cfg.get('environment', {})
    body_cfg = cfg.get('body', {})
    sensory_cfg = cfg.get('sensory', {})
    
    # Build resource arrays
    resources = env_cfg.get('resources', [])
    if resources:
        res_type = jnp.array([0 if r.get('type') == 'food' else 1 for r in resources], dtype=jnp.int32)
        res_property = jnp.array([r.get('property', [0,0,0,0,0]) for r in resources])
        res_spawn_area = jnp.array([[*r.get('spawn_area', [[0,0],[10,10]])[0], 
                                     *r.get('spawn_area', [[0,0],[10,10]])[1]] for r in resources])
        res_max_cons = jnp.array([r.get('max_consecutive', 999) for r in resources], dtype=jnp.int32)
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
    predators = env_cfg.get('predators', [])
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
        height=env_cfg.get('height', 10),
        width=env_cfg.get('width', 10),
        max_steps=env_cfg.get('max_steps', 500),
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
        max_satiation=body_cfg.get('max_satiation', 100.0),
        max_injury=body_cfg.get('max_injury', 20.0),
        food_gain=body_cfg.get('food_gain', 10.0),
        setpoint=body_cfg.get('setpoint', 50.0),
        injury_recovery=body_cfg.get('injury_recovery', 1.0),
        smoothing_duration=body_cfg.get('smoothing_duration', 3),
        death_penalty=body_cfg.get('death_penalty', 10.0),
        overeating_death=body_cfg.get('overeating_death', False),
        use_homeostatic_reward=body_cfg.get('use_homeostatic_reward', True),
        with_satiation=body_cfg.get('with_satiation', False),
        with_injury=body_cfg.get('with_injury', True),
        rest_action_enabled=env_cfg.get('rest_action_enabled', True),
        eat_action_enabled=env_cfg.get('eat_action_enabled', True),
        sensor_radius=sensory_cfg.get('sensor_radius', 10.0),
        sensor_decay=sensory_cfg.get('sensor_decay', 2.0),
        sensor_range=sensory_cfg.get('sensor_range', 3)
    )
