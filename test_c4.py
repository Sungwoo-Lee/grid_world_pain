import jax
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.renderer_v2 import render_jax_state_v2

def test_c4():
    print("Testing C4 (Renderer with multiple predators)...")
    config = Config.load_yaml('configs/experiment/labmeeting/basic-00-predator.yaml')
    config._config['environment']['predators'][0]['count'] = 3
    config._config['environment']['random_start_pos'] = True
    config._config['environment']['height'] = 10
    config._config['environment']['width'] = 10
    config._config['environment']['predators'][0]['spawn_area'] = [[1, 1], [10, 10]]
    params = load_env_params(config)
    
    key = jax.random.PRNGKey(42)
    state = jax_reset(params, key)
    
    try:
        frame = render_jax_state_v2(state, params, step=0)
        print(f"Rendered frame with shape: {frame.shape}")
        print("C4 Renderer check passed (no crash).")
    except Exception as e:
        print(f"C4 Renderer failed: {e}")
        raise e

if __name__ == '__main__':
    test_c4()
