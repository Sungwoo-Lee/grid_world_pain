import os
import jax
import jax.numpy as jnp
from flax import nnx
import yaml
import numpy as np
from PIL import Image
import orbax.checkpoint as ocp

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation, get_observation_breakdown
from src.environment.renderer import render_jax_state
from src.models.recurrent_ppo_network import ActorCriticRNN

def peel_nnx_state(st):
    if isinstance(st, dict):
        if 'value' in st:
            return st['value']
        return {k: peel_nnx_state(v) for k, v in st.items()}
    return st

def main():
    results_dir = "/media/nas01/projects/Interoceptive-AI/grid_world_pain/results/JAX_RecurrentPPO/20260212-162158_rppoNMN_128env_gsize1"
    models_dir = os.path.join(results_dir, "models")
    config_path = os.path.join(models_dir, "config.yaml")
    iteration = 1000008
    
    with open(config_path, 'r') as f:
        config = Config(yaml.safe_load(f))
    
    params = load_env_params(config)
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # 1. Initialize State
    state = jax_reset(params, key)
    obs = get_observation(state, params)
    
    # 2. Initialize Model (just to get the structure right for restoration)
    input_dim = obs.shape[0]
    rest_enabled = params.rest_action_enabled
    eat_enabled = params.eat_action_enabled
    action_dim = 4 + int(rest_enabled) + int(eat_enabled)
    
    rngs = nnx.Rngs(key)
    mod_type = config.get('agent.modulation.type')
    if mod_type is not None:
        modulation_config = {
            'type': mod_type,
            'mod_hidden_size': config.get_mandatory('agent.modulation.mod_hidden_size'),
            'grouping_size': config.get_mandatory('agent.modulation.grouping_size'),
            'percept_bias_init': config.get_mandatory('agent.modulation.percept_bias_init'),
            'memory_bias_init': config.get_mandatory('agent.modulation.memory_bias_init'),
            'temp_clip': config.get_mandatory('agent.modulation.temp_clip'),
        }
    else:
        modulation_config = None
        
    model = ActorCriticRNN(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_size=config.get_mandatory('agent.hidden_size'),
        rngs=rngs,
        rnn_type=config.get_mandatory('agent.rnn_type'),
        activation=config.get_mandatory('agent.activation'),
        modulation_config=modulation_config
    )
    
    # 3. Restore weights
    checkpointer = ocp.CheckpointManager(
        os.path.abspath(models_dir),
        checkpointers=ocp.StandardCheckpointer()
    )
    restored = checkpointer.restore(iteration)
    if 'model' in restored:
        nnx.update(model, peel_nnx_state(restored['model']))
    
    # 4. Take one step or just render initial state
    # Let's take one step to ensure sensory data is populated
    action = 0 # Up
    state, reward, done, info = jax_step(state, action, params)
    obs = get_observation(state, params)
    
    # 5. Render
    # Get breakdown for sensory viz
    breakdown = get_observation_breakdown(params)
    
    def get_sensory_viz(obs_vec, true_obs_vec=None):
        ptr = 0
        t_ptr = 0
        
        olf_dim = breakdown['Olfaction']
        olf_obs = obs_vec[ptr:ptr+olf_dim]
        olf_true = true_obs_vec[t_ptr:t_ptr+olf_dim] if true_obs_vec is not None else olf_obs
        ptr += olf_dim; t_ptr += olf_dim
        
        noc_dim = breakdown['Extero Nociception']
        noc_obs = float(obs_vec[ptr]) if noc_dim > 0 else 0.0
        noc_true = float(true_obs_vec[t_ptr]) if true_obs_vec is not None and noc_dim > 0 else noc_obs
        ptr += noc_dim; t_ptr += noc_dim
        
        coll_dim = breakdown['Collision']
        coll_obs = obs_vec[ptr:ptr+coll_dim]
        coll_true = true_obs_vec[t_ptr:t_ptr+coll_dim] if true_obs_vec is not None else coll_obs
        ptr += coll_dim; t_ptr += coll_dim
        
        loc_dim = breakdown['Location']
        loc_vec = obs_vec[ptr:ptr+loc_dim]
        ptr += loc_dim; t_ptr += loc_dim
        
        sat_dim = breakdown['Satiation']
        sat_obs = float(obs_vec[ptr]) if sat_dim > 0 else 0.0
        ptr += sat_dim; t_ptr += sat_dim
        
        nut_dim = breakdown['Nutrition']
        nut_obs = float(obs_vec[ptr]) if nut_dim > 0 else 0.0
        ptr += nut_dim; t_ptr += nut_dim
        
        inj_dim = breakdown['Injury']
        inj_obs = float(obs_vec[ptr]) if inj_dim > 0 else 0.0
        ptr += inj_dim; t_ptr += inj_dim
        
        viz = [
            {'name': 'Olfactory', 'vector': olf_obs, 'true_vector': olf_true, 'type': 'spectrum'},
            {'name': 'Extero Nociception', 'intensity': noc_obs, 'true_intensity': noc_true, 'color': '#c0392b', 'type': 'intensity'},
            {'name': 'Collision', 'vector': coll_obs, 'true_vector': coll_true, 'type': 'diamond', 'range': params.sensor_range, 'num_features': 1},
            {'name': 'LOC', 'value_text': f"({loc_vec[0]:.2f}, {loc_vec[1]:.2f})", 'color': '#ADB5BD', 'type': 'text'},
            {'name': 'Satiation', 'intensity': sat_obs, 'type': 'intensity'},
            {'name': 'Nutrition', 'intensity': nut_obs, 'type': 'intensity'},
            {'name': 'Injury', 'intensity': inj_obs, 'type': 'intensity'},
        ]
        
        if 'Visual' in breakdown:
            vis_dim = breakdown['Visual']
            vis_obs = obs_vec[ptr:ptr+vis_dim]
            vis_true = true_obs_vec[t_ptr:t_ptr+vis_dim] if true_obs_vec is not None else vis_obs
            ptr += vis_dim; t_ptr += vis_dim
            viz.append({'name': 'Visual', 'vector': vis_obs, 'true_vector': vis_true, 'type': 'visual_grid', 'num_features': 8, 'range': params.visual_sensor_range})
            
        return viz

    icon_config = config.get('visualization.icons', None)
    true_obs = get_observation(state, params, apply_noise=False)
    frame = render_jax_state(
        state, params, episode=1, step=1, train_episode=100,
        dpi=100, action=action, sensory_data=get_sensory_viz(obs, true_obs),
        icon_config=icon_config
    )
    
    # 6. Save image
    img = Image.fromarray(frame)
    output_path = "/media/nas01/projects/Interoceptive-AI/grid_world_pain/evaluation_snapshot.png"
    img.save(output_path)
    print(f"Snapshot saved to {output_path}")

if __name__ == "__main__":
    main()
