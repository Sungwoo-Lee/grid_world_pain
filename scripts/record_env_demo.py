import os
import jax
import jax.numpy as jnp
from tqdm import tqdm
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.environment.config_loader import load_env_params, load_env_config
from src.environment.core import jax_step, jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown
from src.environment.renderer import render_jax_state, save_jax_video

def main():
    # 1. Load default config
    config_path = "configs/environment/default.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    config = load_env_config(config_path)  # honours `extends:` if present; standalone otherwise

    # 2. Setup Params
    params = load_env_params(config)
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # 3. Setup Sensors for rendering
    breakdown = get_observation_breakdown(params)
    
    def get_sensory_viz(obs_vec):
        from src.environment.sensor import build_sensory_viz
        return build_sensory_viz(obs_vec, state, params)

    # 4. Run loop
    key, reset_key = jax.random.split(key)
    state = jax_reset(params, reset_key)
    
    frames = []
    max_steps = 150
    fps = 10
    
    # Action labels
    rest_enabled = params.rest_action_enabled
    eat_enabled = params.eat_action_enabled
    action_dim = 4 + int(rest_enabled) + int(eat_enabled)
    
    print(f"Recording demo ({max_steps} steps)...")
    for s in tqdm(range(max_steps)):
        obs = get_observation(state, params)
        frames.append(render_jax_state(
            state, params, episode=1, step=s, 
            sensory_data=get_sensory_viz(obs)
        ))
        
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, action_dim)
        
        state, reward, done, info = jax_step(state, action, params)
        if done:
             break
             
    # 5. Save video
    output_path = "assets/temp_demo.mp4"
    save_jax_video(frames, output_path, fps=fps)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
