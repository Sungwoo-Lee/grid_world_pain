import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from src.environment.core import jax_step, jax_reset, calculate_drive
from src.environment.sensor import get_observation
from src.models.recurrent_ppo_network import get_action_and_value_nnx
from src.utils.wandb_utils import upload_video

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, checkpoint_pct, 
                            render_video=False, wandb_enabled=False, debug=False, quiet=True):
    """
    Runs deterministic evaluation episodes using the JAX model.
    """
    from src.environment.sensor import get_observation_breakdown
    
    key = jax.random.PRNGKey(seed)
    
    # Video output setup
    video_dir = os.path.join(results_dir, "videos")
    if render_video:
        os.makedirs(video_dir, exist_ok=True)
        from src.environment.renderer import render_jax_state, save_jax_video
    
    # Breakdown for sensory visualization
    breakdown = get_observation_breakdown(params)
    icon_config = config.get('visualization.icons', None)
    
    episode_rewards = []
    episode_lengths = []
    all_frames = []
    
    # Stats recording
    record_stats = config.get('testing.record_stats', False)
    stats_dir = os.path.join(results_dir, "stats")
    if record_stats:
        os.makedirs(stats_dir, exist_ok=True)
    
    # Progress bar for episodes
    ep_pbar = tqdm(range(num_episodes), desc="Evaluating Episodes", disable=quiet)
    # Create action name mapping
    action_map = ["Up", "Right", "Down", "Left"]
    if params.rest_action_enabled:
        action_map.append("Rest")
    if params.eat_action_enabled:
        action_map.append("Eat")

    for ep in ep_pbar:
        if debug:
            print(f"  --- Starting Evaluation Episode {ep+1}/{num_episodes} ---", flush=True)
            
        key, reset_key = jax.random.split(key)
        state = jax_reset(params, reset_key)
        obs = get_observation(state, params)
        
        # Initialize hidden state
        if hasattr(model, 'initial_state'):
            h_state = model.initial_state(batch_size=1)
        else:
            h_state = None
        
        total_reward = 0.0
        step_count = 0
        done = False
        max_steps = params.max_steps
        
        # Per-episode stats
        ep_stats = []
        if record_stats:
            ep_stats.append({
                'step': 0,
                'satiation': float(state.satiation),
                'nutrition': float(state.nutrition),
                'injury': float(state.injury_level),
                'rest_streak': int(state.rest_streak),
                'pos_r': int(state.agent_pos[0]),
                'pos_c': int(state.agent_pos[1]),
                'drive': float(calculate_drive(state.satiation, state.injury_level, params)),
                'drive_hunger': 0.0,
                'drive_injury': 0.0,
                # New "Full Spectrum" metrics
                'event_ate': False,
                'event_damage': 0.0,
                'event_collided': False,
                'event_rested': False,
                'sense_nociception': 0.0,
                'dist_to_food': 99.0,
                'dist_to_pred': 99.0,
                'reward_homeostatic': 0.0,
                'reward_extrinsic': 0.0,
                'max_satiation': float(params.max_satiation),
                'max_injury': float(params.max_injury),
                'action': "None",
                'reward': 0.0
            })
        
        def get_sensory_viz(obs_vec, true_obs_vec=None):
            # Internal helper to slice flat obs into renderer-friendly format
            ptr = 0
            t_ptr = 0
            
            # Slices
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
            
            if 'Proprioception' in breakdown:
                proprio_dim = breakdown['Proprioception']
                proprio_vec = obs_vec[ptr:ptr+proprio_dim]
                ptr += proprio_dim; t_ptr += proprio_dim
                viz.append({'name': 'Proprioception', 'vector': proprio_vec, 'type': 'radial', 'color': '#be4bdb'})

            return viz

        if render_video:
            if debug: print(f"    [Render] Initial frame...", end="", flush=True)
            true_obs = get_observation(state, params, apply_noise=False)
            all_frames.append(render_jax_state(
                state, params, episode=ep+1, step=0, 
                train_episode=checkpoint_pct,
                sensory_data=get_sensory_viz(obs, true_obs),
                icon_config=icon_config
            ))
            if debug: print(" Done", flush=True)
        
        # Progress bar for steps if rendering video
        step_pbar = tqdm(total=max_steps, desc=f"Episode {ep+1} Steps", leave=False, disable=quiet or not render_video)
        
        # Determine action space dimensions
        rest_enabled = params.rest_action_enabled
        eat_enabled = params.eat_action_enabled
        action_dim = 4 + int(rest_enabled) + int(eat_enabled)
        
        while not done and step_count < max_steps:
            if debug:
                print(f"    [Step {step_count}] Inferencing...", end="", flush=True)
            
            if model is not None:
                # Model inference (deterministic in eval_mode)
                obs_batch = obs[None, :]  # Add batch dim
                
                if h_state is not None:
                    action, log_prob, value, h_new, _ = get_action_and_value_nnx(model, obs_batch, h_state, eval_mode=True)
                    h_state = h_new
                else:
                    action, _, _, _, _ = get_action_and_value_nnx(model, obs_batch, None, eval_mode=True)
                
                action_idx = int(action)
            else:
                # Random action if no model provided
                key, action_key = jax.random.split(key)
                action_idx = int(jax.random.randint(action_key, (), 0, action_dim))
            
            if debug:
                print(f" Done (Action: {action_idx}). Stepping env...", end="", flush=True)
            
            # Step
            next_state, reward, done, info = jax_step(state, action_idx, params)
            total_reward += float(reward)
            step_count += 1
            
            state = next_state
            next_obs = get_observation(state, params)
            
            if debug:
                print(f" Done. Reward: {reward:.2f}", flush=True)
            
            if render_video:
                if debug: print(f"    [Step {step_count}] Rendering...", end="", flush=True)
                true_obs = get_observation(state, params, apply_noise=False)
                all_frames.append(render_jax_state(
                    state, params, episode=ep+1, step=step_count, 
                    train_episode=checkpoint_pct,
                    action=action_idx, sensory_data=get_sensory_viz(next_obs, true_obs),
                    icon_config=icon_config
                ))
                if debug: print(" Done", flush=True)
            
            if record_stats:
                ep_stats.append({
                    'step': step_count,
                    'satiation': float(state.satiation),
                    'nutrition': float(state.nutrition),
                    'injury': float(state.injury_level),
                    'rest_streak': int(state.rest_streak),
                    'pos_r': int(state.agent_pos[0]),
                    'pos_c': int(state.agent_pos[1]),
                    'drive': float(calculate_drive(state.satiation, state.injury_level, params)),
                    'drive_hunger': float(info.get('drive_hunger', 0.0)),
                    'drive_injury': float(info.get('drive_injury', 0.0)),
                    # New "Full Spectrum" metrics
                    'event_ate': bool(info.get('ate_food', False)),
                    'event_damage': float(info.get('damage', 0.0)),
                    'event_collided': bool(info.get('event_collided', False)),
                    'event_rested': bool(info.get('rested', False)),
                    'sense_nociception': float(next_obs[breakdown['Olfaction']]),
                    'dist_to_food': float(jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 0), jnp.linalg.norm(state.res_pos - state.agent_pos, axis=-1), 99.0))) if state.res_pos.shape[0] > 0 else 99.0,
                    'dist_to_pred': float(jnp.min(jnp.linalg.norm(state.pred_pos - state.agent_pos, axis=-1))) if state.pred_pos.shape[0] > 0 else 99.0,
                    'reward_homeostatic': float(info.get('reward_homeostatic', 0.0)),
                    'reward_extrinsic': float(info.get('reward_extrinsic', 0.0)),
                    'metabolic_drain': float(info.get('metabolic_drain', 0.0)),
                    'termination_reason': int(info.get('termination_reason', 0)),
                    'max_satiation': float(params.max_satiation),
                    'max_injury': float(params.max_injury),
                    'action': action_map[action_idx] if 0 <= action_idx < len(action_map) else "Unknown",
                    'reward': float(reward)
                })
            
            obs = next_obs
            step_pbar.update(1)
        
        # Save stats to CSV
        if record_stats and ep_stats:
            import pandas as pd
            stats_path = os.path.join(stats_dir, f"ep_{ep+1}_stats.csv")
            pd.DataFrame(ep_stats).to_csv(stats_path, index=False)
            if debug: print(f"    [Stats] Saved to {stats_path}")

        step_pbar.close()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        if debug or not WANDB_AVAILABLE:
            print(f"  --- Episode {ep+1}/{num_episodes} Complete | Steps: {step_count} | Reward: {total_reward:.2f} ---", flush=True)
        
        # Add a few pause frames between episodes
        if render_video:
            for _ in range(5):
                all_frames.append(all_frames[-1])
    
    # Save Consolidated Video
    last_video_path = None
    if render_video and all_frames:
        video_path = os.path.join(video_dir, f"eval_{checkpoint_pct}.mp4")
        fps = config.get('visualization.fps', 5)
        if debug: print(f"    [Video] Saving {len(all_frames)} frames to {video_path}...", end="", flush=True)
        save_jax_video(all_frames, video_path, fps=fps, quiet=quiet)
        if debug: print(" Done", flush=True)
        last_video_path = video_path
        print(f"  --- Consolidated Evaluation Video saved to: {video_path} ---", flush=True)
        
        # Upload to WandB if enabled
        if wandb_enabled and WANDB_AVAILABLE and wandb.run:
            from src.utils.wandb_utils import upload_video
            upload_video(video_path, episode=checkpoint_pct, step=checkpoint_pct, caption=f"Episode {checkpoint_pct}", quiet=True)
    
    # Statistics
    mean_reward = float(np.mean(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    
    return {
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "last_video_path": last_video_path
    }
