import os
import jax
import jax.numpy as jnp
from tqdm import tqdm
import yaml
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_step, jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown
from src.environment.renderer import render_jax_state, save_jax_video

def main():
    # 1. Load default config
    config_path = "configs/environment/environment.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    # 2. Setup Params
    params = load_env_params(config)
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # 3. Setup Sensors for rendering
    breakdown = get_observation_breakdown(params)
    
    def get_sensory_viz(obs_vec):
        ptr = 0
        chem_end = ptr + breakdown['Chemical']
        chem_vec = obs_vec[ptr:chem_end]
        ptr = chem_end
        
        noc_end = ptr + breakdown['Extero Nociception']
        noc_val = float(obs_vec[ptr]) if breakdown['Extero Nociception'] > 0 else 0.0
        ptr = noc_end
        
        coll_end = ptr + breakdown['Collision']
        coll_vec = obs_vec[ptr:coll_end]
        ptr = coll_end
        
        loc_end = ptr + breakdown['Location']
        loc_vec = obs_vec[ptr:loc_end]
        ptr = loc_end
        
        viz = [
            {'name': 'Olfactory', 'vector': chem_vec, 'type': 'spectrum'},
            {'name': 'Extero Nociception', 'intensity': noc_val, 'color': '#c0392b', 'type': 'intensity'},
            {'name': 'Collision', 'vector': coll_vec, 'type': 'diamond', 'range': params.sensor_range, 'num_features': 1, 'side_by_side': True},
            {'name': 'LOC', 'value_text': f"({loc_vec[0]:.2f}, {loc_vec[1]:.2f})", 'color': '#ADB5BD', 'type': 'text'}
        ]
        
        if 'Visual' in breakdown:
            vis_vec = obs_vec[ptr:ptr + breakdown['Visual']]
            viz.insert(3, {'name': 'Visual (One-Hot)', 'vector': vis_vec, 'type': 'diamond', 'range': params.visual_sensor_range, 'num_features': 7, 'side_by_side': True})
        
        return viz

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
