import os
import csv
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from src.environment.core import jax_step, jax_reset, calculate_drive
from src.environment.sensor import get_observation, get_visual_offsets
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
    stats_dir = os.path.join(results_dir, "stats", str(checkpoint_pct))
    if record_stats:
        os.makedirs(stats_dir, exist_ok=True)
        
        # --- Pre-calculate headers for consistent CSV structure ---
        stat_headers = ['step', 'pos_r', 'pos_c', 'action', 'reward', 
                       'satiation', 'nutrition', 'injury', 'rest_streak']
        stat_headers += ['event_ate', 'event_collided', 'event_rested',
                        'damage_total', 'damage_danger', 'damage_predator', 'damage_obstacle']
        
        # 1. Olfaction
        if 'Olfaction' in breakdown:
            for i in range(breakdown['Olfaction']): 
                stat_headers.append(f"obs_olf_{i}")
        # 2. Extero Nociception
        if 'Extero Nociception' in breakdown:
            stat_headers.append("obs_noc")
        # 3. Collision
        coll_offsets = get_visual_offsets(params.sensor_range)
        for i in range(coll_offsets.shape[0]):
            dr, dc = coll_offsets[i]
            stat_headers.append(f"obs_coll_r{dr}c{dc}")
            
        # 4. Location
        if 'Location' in breakdown:
            stat_headers += ["obs_loc_r", "obs_loc_c"]
        # 5. Interoception
        stat_headers += ["obs_sat", "obs_nut", "obs_inj"]
        # 6. Visual
        if 'Visual' in breakdown:
            vis_labels = ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']
            total_vis_dim = breakdown['Visual']
            num_channels = len(vis_labels)
            num_cells = total_vis_dim // num_channels
            for c_idx, label in enumerate(vis_labels):
                for i in range(num_cells):
                    stat_headers.append(f"obs_vis_{label}_{i}")
        # 7. Proprioception
        if 'Proprioception' in breakdown:
            for i in range(breakdown['Proprioception']): 
                stat_headers.append(f"obs_prop_{i}")
        
        # Object Positions
        for i in range(len(params.res_type)):
            r_type = "food" if params.res_type[i] == 0 else "danger"
            stat_headers += [f"res_{i}_{r_type}_r", f"res_{i}_{r_type}_c", f"res_{i}_active"]
        for i in range(len(params.pred_nociception)):
            stat_headers += [f"pred_{i}_r", f"pred_{i}_c"]
        for i in range(len(params.neutral_nociception)):
            stat_headers += [f"neutral_{i}_r", f"neutral_{i}_c"]
        for i in range(len(params.obs_blocking)):
            obs_name = params.obstacle_names[params.obs_type[i]]
            stat_headers += [f"obs_{i}_{obs_name}_r", f"obs_{i}_{obs_name}_c"]
        
        stat_headers += ['termination_reason', 'max_satiation', 'max_injury']
    
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
        
        # Per-episode deferred stats accumulation (no GPU sync during loop)
        # We collect raw JAX arrays and do a single device_get at episode end.
        ep_jax_states = []   # list of dicts of JAX arrays
        ep_jax_infos = []    # list of info dicts (JAX arrays)
        ep_actions = []      # list of int action indices
        ep_rewards = []      # list of float rewards
        ep_obs = []          # list of observation JAX arrays

        if record_stats:
            # Step 0: record initial state (deferred — no device_get)
            ep_jax_states.append({
                'agent_pos': state.agent_pos,
                'satiation': state.satiation,
                'nutrition': state.nutrition,
                'injury_level': state.injury_level,
                'rest_streak': state.rest_streak,
                'res_pos': state.res_pos,
                'res_active': state.res_active,
                'pred_pos': state.pred_pos,
                'neutral_pos': state.neutral_pos,
                'obs_pos': state.obs_pos,
            })
            ep_jax_infos.append({})  # No info at step 0
            ep_actions.append(-1)    # No action at step 0 ("None")
            ep_rewards.append(0.0)
            ep_obs.append(obs)
        
        def get_sensory_viz(obs_vec, true_obs_vec=None):
            # Internal helper to slice flat obs into renderer-friendly format
            ptr = 0
            t_ptr = 0
            viz = []
            
            # 1. Olfaction
            olf_dim = breakdown.get('Olfaction', 0)
            if olf_dim > 0:
                olf_obs = obs_vec[ptr:ptr+olf_dim]
                olf_true = true_obs_vec[t_ptr:t_ptr+olf_dim] if true_obs_vec is not None else olf_obs
                ptr += olf_dim; t_ptr += olf_dim
                viz.append({'name': 'Olfactory', 'vector': olf_obs, 'true_vector': olf_true, 'type': 'spectrum', 'labels': ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']})
            
            # 2. Extero Nociception
            noc_dim = breakdown.get('Extero Nociception', 0)
            if noc_dim > 0:
                noc_obs = float(obs_vec[ptr])
                noc_true = float(true_obs_vec[t_ptr]) if true_obs_vec is not None else noc_obs
                ptr += noc_dim; t_ptr += noc_dim
                viz.append({'name': 'Extero Nociception', 'intensity': noc_obs, 'true_intensity': noc_true, 'color': '#c0392b', 'type': 'intensity'})
            
            # 3. Collision
            coll_dim = breakdown.get('Collision', 0)
            if coll_dim > 0:
                coll_obs = obs_vec[ptr:ptr+coll_dim]
                coll_true = true_obs_vec[t_ptr:t_ptr+coll_dim] if true_obs_vec is not None else coll_obs
                ptr += coll_dim; t_ptr += coll_dim
                viz.append({'name': 'Collision', 'vector': coll_obs, 'true_vector': coll_true, 'type': 'diamond', 'range': params.sensor_range, 'num_features': 1})
            
            # 4. Location
            loc_dim = breakdown.get('Location', 0)
            if loc_dim > 0:
                loc_vec = obs_vec[ptr:ptr+loc_dim]
                ptr += loc_dim; t_ptr += loc_dim
                viz.append({'name': 'LOC', 'value_text': f"({loc_vec[0]:.2f}, {loc_vec[1]:.2f})", 'color': '#ADB5BD', 'type': 'text'})
            
            # 5. Interoception
            if 'Satiation' in breakdown:
                sat_obs = float(obs_vec[ptr])
                viz.append({'name': 'Satiation', 'intensity': sat_obs, 'type': 'intensity'})
                ptr += breakdown['Satiation']; t_ptr += breakdown['Satiation']
            if 'Nutrition' in breakdown:
                nut_obs = float(obs_vec[ptr])
                viz.append({'name': 'Nutrition', 'intensity': nut_obs, 'type': 'intensity'})
                ptr += breakdown['Nutrition']; t_ptr += breakdown['Nutrition']
            if 'Injury' in breakdown:
                inj_obs = float(obs_vec[ptr])
                viz.append({'name': 'Injury', 'intensity': inj_obs, 'type': 'intensity'})
                ptr += breakdown['Injury']; t_ptr += breakdown['Injury']
            
            # 6. Visual
            if 'Visual' in breakdown:
                vis_dim = breakdown['Visual']
                vis_obs = obs_vec[ptr:ptr+vis_dim]
                vis_true = true_obs_vec[t_ptr:t_ptr+vis_dim] if true_obs_vec is not None else vis_obs
                ptr += vis_dim; t_ptr += vis_dim
                viz.append({'name': 'Visual', 'vector': vis_obs, 'true_vector': vis_true, 'type': 'visual_grid', 'num_features': 8, 'range': params.visual_sensor_range, 'labels': ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']})
            
            # 7. Proprioception
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
                info=None,
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
                    info=jax.device_get(info),
                    icon_config=icon_config
                ))
                if debug: print(" Done", flush=True)
            
            if record_stats:
                # Deferred: collect raw JAX arrays (no device_get during loop)
                ep_jax_states.append({
                    'agent_pos': state.agent_pos,
                    'satiation': state.satiation,
                    'nutrition': state.nutrition,
                    'injury_level': state.injury_level,
                    'rest_streak': state.rest_streak,
                    'res_pos': state.res_pos,
                    'res_active': state.res_active,
                    'pred_pos': state.pred_pos,
                    'neutral_pos': state.neutral_pos,
                    'obs_pos': state.obs_pos,
                })
                ep_jax_infos.append(info)
                ep_actions.append(action_idx)
                ep_rewards.append(float(reward))
                ep_obs.append(next_obs)
            
            obs = next_obs
            step_pbar.update(1)
        
        # === BATCH TRANSFER: Single GPU→CPU sync per episode ===
        if record_stats and ep_jax_states:
            # Stack all per-step JAX arrays into batched tensors for one device_get
            batched_state = jax.device_get({
                'agent_pos': jnp.stack([s['agent_pos'] for s in ep_jax_states]),
                'satiation': jnp.stack([s['satiation'] for s in ep_jax_states]),
                'nutrition': jnp.stack([s['nutrition'] for s in ep_jax_states]),
                'injury_level': jnp.stack([s['injury_level'] for s in ep_jax_states]),
                'rest_streak': jnp.stack([s['rest_streak'] for s in ep_jax_states]),
                'res_pos': jnp.stack([s['res_pos'] for s in ep_jax_states]),
                'res_active': jnp.stack([s['res_active'] for s in ep_jax_states]),
                'pred_pos': jnp.stack([s['pred_pos'] for s in ep_jax_states]),
                'neutral_pos': jnp.stack([s['neutral_pos'] for s in ep_jax_states]),
                'obs_pos': jnp.stack([s['obs_pos'] for s in ep_jax_states]),
            })
            
            # Batch-transfer info dicts (skip step 0 which has empty dict)
            info_keys = ['ate_food', 'event_collided', 'rested', 'damage', 
                        'damage_danger', 'damage_predator', 'damage_obstacle', 'termination_reason']
            batched_info = {}
            for ik in info_keys:
                vals = []
                for inf in ep_jax_infos:
                    vals.append(inf.get(ik, 0))
                # Convert JAX arrays in the list; plain Python values stay as-is
                batched_info[ik] = np.array(jax.device_get(vals))
            
            # Batch-transfer observations
            batched_obs = np.array(jax.device_get(jnp.stack(ep_obs)))
            
            # Pre-compute observation header indices
            obs_header_indices = [i for i, h in enumerate(stat_headers) if h.startswith("obs_")]
            num_obs_headers = len(obs_header_indices)
            
            # Build all CSV rows at once from NumPy arrays
            num_steps = len(ep_jax_states)
            stats_path = os.path.join(stats_dir, f"{ep+1:06d}ep_stats.csv")
            
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(stat_headers)
                
                for t in range(num_steps):
                    action_name = "None" if ep_actions[t] < 0 else (
                        action_map[ep_actions[t]] if 0 <= ep_actions[t] < len(action_map) else "Unknown")
                    
                    row = [
                        t,                                              # step
                        int(batched_state['agent_pos'][t, 0]),          # pos_r
                        int(batched_state['agent_pos'][t, 1]),          # pos_c
                        action_name,                                    # action
                        ep_rewards[t],                                  # reward
                        float(batched_state['satiation'][t]),           # satiation
                        float(batched_state['nutrition'][t]),           # nutrition
                        float(batched_state['injury_level'][t]),        # injury
                        int(batched_state['rest_streak'][t]),           # rest_streak
                        bool(batched_info['ate_food'][t]),              # event_ate
                        bool(batched_info['event_collided'][t]),        # event_collided
                        bool(batched_info['rested'][t]),                # event_rested
                        float(batched_info['damage'][t]),               # damage_total
                        float(batched_info['damage_danger'][t]),        # damage_danger
                        float(batched_info['damage_predator'][t]),      # damage_predator
                        float(batched_info['damage_obstacle'][t]),      # damage_obstacle
                    ]
                    
                    # Observations (vectorized slice)
                    obs_vec = batched_obs[t]
                    for i in range(min(num_obs_headers, len(obs_vec))):
                        row.append(float(obs_vec[i]))
                    
                    # Resource positions
                    res_pos = batched_state['res_pos'][t]
                    res_active = batched_state['res_active'][t]
                    for i in range(res_pos.shape[0]):
                        row.append(int(res_pos[i, 0]))
                        row.append(int(res_pos[i, 1]))
                        row.append(bool(res_active[i]))
                    
                    # Predator positions
                    pred_pos = batched_state['pred_pos'][t]
                    for i in range(pred_pos.shape[0]):
                        row.append(int(pred_pos[i, 0]))
                        row.append(int(pred_pos[i, 1]))
                    
                    # Neutral positions
                    neutral_pos = batched_state['neutral_pos'][t]
                    for i in range(neutral_pos.shape[0]):
                        row.append(int(neutral_pos[i, 0]))
                        row.append(int(neutral_pos[i, 1]))
                    
                    # Obstacle positions
                    obs_pos = batched_state['obs_pos'][t]
                    for i in range(obs_pos.shape[0]):
                        row.append(int(obs_pos[i, 0]))
                        row.append(int(obs_pos[i, 1]))
                    
                    # Termination info and max values
                    row.append(int(batched_info['termination_reason'][t]))
                    row.append(float(params.max_satiation))
                    row.append(float(params.max_injury))
                    
                    writer.writerow(row)
            
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

def main():
    pass

if __name__ == "__main__":
    main()
