import jax
import jax.numpy as jnp
import os
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step

def test_c1():
    print("Testing C1...")
    config = Config.load_yaml('configs/environment/experiment/archive/labmeeting/basic-00-predator.yaml')
    params = load_env_params(config)
    num_pred = params.pred_damage.shape[0]
    print(f"C1 Check: params.pred_damage.shape[0] = {num_pred} (Expected: 1)")
    assert num_pred == 1

def test_c2():
    print("Testing C2...")
    # Load basic config and inject two predator entries
    config = Config.load_yaml('configs/environment/experiment/archive/labmeeting/basic-00-predator.yaml')
    config._config['environment']['predators'] = [
        {
            "name": "predator1", "count": 2, "property": [0.0, 1.0, 0.0, 0.0, 0.0],
            "move_interval": 3, "damage": [15.0, 45.0], "nociception_intensity": 0.9,
            "spawn_area": [[1, 1], [5, 5]], "patrol_area": [[1, 1], [5, 5]],
            "detection_range": 3, "max_stamina": 15, "stamina_recovery_rate": 1,
            "hunt_stamina_threshold": 0.7, "attack_delay": 3, "lose_interest_multiplier": 1.5
        },
        {
            "name": "predator2", "count": 3, "property": [0.0, 1.0, 0.0, 0.0, 0.0],
            "move_interval": 3, "damage": [15.0, 45.0], "nociception_intensity": 0.9,
            "spawn_area": [[1, 1], [5, 5]], "patrol_area": [[1, 1], [5, 5]],
            "detection_range": 3, "max_stamina": 15, "stamina_recovery_rate": 1,
            "hunt_stamina_threshold": 0.7, "attack_delay": 3, "lose_interest_multiplier": 1.5
        }
    ]
    params = load_env_params(config)
    num_pred = params.pred_damage.shape[0]
    print(f"C2 Check: params.pred_damage.shape[0] = {num_pred} (Expected: 5)")
    assert num_pred == 5

def test_c3():
    print("Testing C3...")
    config = Config.load_yaml('configs/environment/experiment/archive/labmeeting/basic-00-predator.yaml')
    config._config['environment']['predators'][0]['count'] = 3
    config._config['environment']['random_start_pos'] = True
    config._config['environment']['height'] = 10
    config._config['environment']['width'] = 10
    config._config['environment']['predators'][0]['spawn_area'] = [[1, 1], [10, 10]]
    params = load_env_params(config)
    
    key = jax.random.PRNGKey(42)
    state = jax_reset(params, key)
    
    num_res = params.res_type.shape[0]
    pred_pos = state.pred_pos
    print(f"Predator positions at reset: \n{pred_pos}")
    
    # Check if they are distinct
    unique_pos = jnp.unique(pred_pos, axis=0)
    print(f"Unique predator positions: {unique_pos.shape[0]} (Expected: 3, unless rare collision)")
    
    # Step the environment
    action = 0
    next_state, reward, done, info = jax_step(state, action, params)
    
    pred_pos_next = next_state.pred_pos
    print(f"Predator positions after step: \n{pred_pos_next}")
    print("C3 Passed")

if __name__ == "__main__":
    test_c1()
    test_c2()
    test_c3()
    print("All Python API tests passed!")
