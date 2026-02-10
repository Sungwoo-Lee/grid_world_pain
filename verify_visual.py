import jax
import jax.numpy as jnp
import os
from src.environment.core import jax_step, jax_reset
from src.environment.config_loader import load_env_params
from src.environment.renderer import render_jax_state, save_jax_video
from src.environment.sensor import get_observation, get_observation_breakdown
from src.utils.config import Config
from tqdm import tqdm

# Load live config
config = Config.load_yaml("configs/environment/environment.yaml")
params = load_env_params(config)

key = jax.random.PRNGKey(42)
state = jax_reset(params, key)

# Define a sequence of actions:
# Move Right (1) for 3 steps to hit some danger (if possible, or just stay to show decay)
# Then Rest (4) for 40 steps
actions = [1, 1, 1] + [4] * 50

frames = []
breakdown = get_observation_breakdown(params)
icon_config = config.get('visualization.icons', None)

def get_sensory_viz(obs_vec):
    # Slice flat obs into renderer-friendly format
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
    
    intero_end = ptr + breakdown['Interoception']
    ptr = intero_end
    
    viz = [
        {'name': 'Olfactory', 'vector': chem_vec, 'type': 'spectrum'},
        {'name': 'Extero Nociception', 'intensity': noc_val, 'color': '#c0392b', 'type': 'intensity'},
        {'name': 'Collision', 'vector': coll_vec, 'type': 'diamond', 'range': params.sensor_range, 'num_features': 1, 'side_by_side': True},
        {'name': 'LOC', 'value_text': f"({loc_vec[0]:.2f}, {loc_vec[1]:.2f})", 'color': '#ADB5BD', 'type': 'text'}
    ]
    return viz

print("Generating visual verification video...")

# Initial frame
obs = get_observation(state, params)
frames.append(render_jax_state(state, params, episode=1, step=0, sensory_data=get_sensory_viz(obs), icon_config=icon_config))

# Take some manual damage to show recovery
state = state._replace(injury_level=jnp.array(15.0), injury_buffer=jnp.zeros(params.smoothing_duration))

for i, action in enumerate(tqdm(actions)):
    state, reward, done, info = jax_step(state, action, params)
    obs = get_observation(state, params)
    frames.append(render_jax_state(
        state, params, episode=1, step=i+1, 
        action=action, sensory_data=get_sensory_viz(obs),
        icon_config=icon_config
    ))
    if done: break

output_path = "results/verification/physiology_demo.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
save_jax_video(frames, output_path, fps=5)

print(f"Video saved to {output_path}")
print("Check the results folder for physiology_demo.mp4")
