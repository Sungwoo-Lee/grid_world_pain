import os
import csv
import contextlib
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

@nnx.jit(static_argnames="eval_mode")
def generic_inference(model, x, h, key=None, eval_mode=False):
    """Generic inference helper that works with both RecurrentPPO and DreamerV3 NNX models.
    Supports single sample x [obs_dim] or [1, obs_dim], and batch x [N, obs_dim]."""
    # Standard signature for both models:
    # (logits, value, h_new, mod_info) = model(x, h)
    logits, value, h_new, mod_info = model(x, h)

    if eval_mode:
        action = jnp.argmax(logits, axis=-1)  # (A,) or (N,) when batched
        if logits.ndim == 1:
            log_prob = 0.0
        else:
            log_prob = jnp.zeros(logits.shape[0])
    else:
        # Note: key must be provided if not eval_mode
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]

    return action, log_prob, value.squeeze(), h_new, mod_info


def _write_episode_stats(stats_dir, episode_number, ep_jax_states, ep_jax_infos, ep_actions,
                         ep_rewards, ep_obs, stat_headers, action_map, params, debug=False,
                         ep_true_obs=None):
    """Write one episode's stats to CSV. Uses one batched device_get then writes rows."""
    info_keys = ['ate_food', 'event_collided', 'rested', 'damage',
                 'damage_danger', 'damage_predator', 'damage_obstacle', 'termination_reason']
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
    batched_info = {}
    for ik in info_keys:
        vals = [inf.get(ik, 0) for inf in ep_jax_infos]
        batched_info[ik] = np.array(jax.device_get(vals))
    batched_obs = np.array(jax.device_get(jnp.stack(ep_obs)))
    batched_true_obs = np.array(jax.device_get(jnp.stack(ep_true_obs))) if ep_true_obs is not None else None
    
    obs_header_indices = [i for i, h in enumerate(stat_headers) if h.startswith("obs_")]
    num_obs_headers = len(obs_header_indices)
    
    true_obs_header_indices = [i for i, h in enumerate(stat_headers) if h.startswith("true_")]
    num_true_obs_headers = len(true_obs_header_indices)
    num_steps = len(ep_jax_states)
    stats_path = os.path.join(stats_dir, f"{episode_number:06d}ep_stats.csv")
    with open(stats_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(stat_headers)
        for t in range(num_steps):
            action_name = "None" if ep_actions[t] < 0 else (
                action_map[ep_actions[t]] if 0 <= ep_actions[t] < len(action_map) else "Unknown")
            row = [
                t,
                int(batched_state['agent_pos'][t, 0]),
                int(batched_state['agent_pos'][t, 1]),
                action_name,
                ep_rewards[t],
                float(batched_state['satiation'][t]),
                float(batched_state['nutrition'][t]),
                float(batched_state['injury_level'][t]),
                int(batched_state['rest_streak'][t]),
                bool(batched_info['ate_food'][t]),
                bool(batched_info['event_collided'][t]),
                bool(batched_info['rested'][t]),
                float(batched_info['damage'][t]),
                float(batched_info['damage_danger'][t]),
                float(batched_info['damage_predator'][t]),
                float(batched_info['damage_obstacle'][t]),
            ]
            obs_vec = batched_obs[t]
            for i in range(min(num_obs_headers, len(obs_vec))):
                row.append(float(obs_vec[i]))
            
            # True obs (if noise diagnostics enabled)
            if batched_true_obs is not None:
                true_vec = batched_true_obs[t]
                for i in range(min(num_true_obs_headers, len(true_vec))):
                    row.append(float(true_vec[i]))
            res_pos = batched_state['res_pos'][t]
            res_active = batched_state['res_active'][t]
            for i in range(res_pos.shape[0]):
                row.append(int(res_pos[i, 0]))
                row.append(int(res_pos[i, 1]))
                row.append(bool(res_active[i]))
            pred_pos = batched_state['pred_pos'][t]
            for i in range(pred_pos.shape[0]):
                row.append(int(pred_pos[i, 0]))
                row.append(int(pred_pos[i, 1]))
            neutral_pos = batched_state['neutral_pos'][t]
            for i in range(neutral_pos.shape[0]):
                row.append(int(neutral_pos[i, 0]))
                row.append(int(neutral_pos[i, 1]))
            obs_pos = batched_state['obs_pos'][t]
            for i in range(obs_pos.shape[0]):
                row.append(int(obs_pos[i, 0]))
                row.append(int(obs_pos[i, 1]))
            row.append(int(batched_info['termination_reason'][t]))
            row.append(float(params.max_satiation))
            row.append(float(params.max_injury))
            writer.writerow(row)
    if debug:
        print(f"    [Stats] Saved to {stats_path}")


def evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, checkpoint_pct,
                            render_video=False, record_stats=None, wandb_enabled=False, debug=False, quiet=True, num_envs=1, device=None):
    """
    Runs deterministic evaluation episodes using the JAX model.
    When num_envs > 1, runs min(num_episodes, num_envs) envs in parallel (episode-ticket design).
    """
    from src.environment.sensor import get_observation_breakdown

    effective_num_envs = min(num_episodes, num_envs)
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
    if record_stats is None:
        record_stats = config.get('testing.record_stats', False)
    
    # Noise diagnostics: Record true (noise-free) observations alongside noised obs
    # Mandatory key — no fallback default.
    record_true_obs = config.get_mandatory('testing.record_true_observations') and record_stats
    
    stats_dir = os.path.join(results_dir, "stats", str(checkpoint_pct))
    # Build stat_headers and action_map whenever we might record (single or parallel path)
    stat_headers = []
    action_map = ["Up", "Right", "Down", "Left"]
    if params.rest_action_enabled:
        action_map.append("Rest")
    if params.eat_action_enabled:
        action_map.append("Eat")
    if record_stats:
        os.makedirs(stats_dir, exist_ok=True)
        stat_headers = ['step', 'pos_r', 'pos_c', 'action', 'reward', 
                       'satiation', 'nutrition', 'injury', 'rest_streak']
        stat_headers += ['event_ate', 'event_collided', 'event_rested',
                        'damage_total', 'damage_danger', 'damage_predator', 'damage_obstacle']
        # Add headers for all observation parts based on breakdown
        for sensor_name, dim in breakdown.items():
            if sensor_name == "Olfaction":
                for i in range(dim): stat_headers.append(f"obs_olf_{i}")
            elif sensor_name == "Extero Nociception":
                stat_headers.append("obs_noc")
            elif sensor_name == "Collision":
                coll_offsets = get_visual_offsets(params.sensor_range)
                for i in range(dim):
                    dr, dc = coll_offsets[i]
                    stat_headers.append(f"obs_coll_r{dr}c{dc}")
            elif sensor_name == "Location":
                stat_headers += ["obs_loc_r", "obs_loc_c"]
            elif sensor_name in ["Satiation", "Nutrition", "Injury"]:
                stat_headers.append(f"obs_intero_{sensor_name.lower()}")
            elif sensor_name == "Visual":
                for i in range(dim): stat_headers.append(f"obs_vis_{i}")
            elif sensor_name == "Proprioception":
                for i in range(dim): stat_headers.append(f"obs_prop_{i}")
        
        # Add headers for true observations if recording is enabled
        if record_true_obs:
            for sensor_name, dim in breakdown.items():
                if sensor_name == "Olfaction":
                    for i in range(dim): stat_headers.append(f"true_olf_{i}")
                elif sensor_name == "Extero Nociception":
                    stat_headers.append("true_noc")
                elif sensor_name == "Collision":
                    coll_offsets = get_visual_offsets(params.sensor_range)
                    for i in range(dim):
                        dr, dc = coll_offsets[i]
                        stat_headers.append(f"true_coll_r{dr}c{dc}")
                elif sensor_name == "Location":
                    stat_headers += ["true_loc_r", "true_loc_c"]
                elif sensor_name in ["Satiation", "Nutrition", "Injury"]:
                    stat_headers.append(f"true_intero_{sensor_name.lower()}")
                elif sensor_name == "Visual":
                    for i in range(dim): stat_headers.append(f"true_vis_{i}")
                elif sensor_name == "Proprioception":
                    for i in range(dim): stat_headers.append(f"true_prop_{i}")
        
        # Add headers for world entities (matching _write_episode_stats loop)
        # 1. Resources
        for i in range(params.res_type.shape[0]):
            res_name = f"res_{i}"
            stat_headers += [f"{res_name}_r", f"{res_name}_c", f"{res_name}_active"]
        
        # 2. Predators
        for i in range(params.pred_damage.shape[0]):
            pred_name = f"pred_{i}"
            stat_headers += [f"{pred_name}_r", f"{pred_name}_c"]
        
        # 3. Neutrals
        for i in range(params.neutral_property.shape[0]):
            neu_name = f"neutral_{i}"
            stat_headers += [f"{neu_name}_r", f"{neu_name}_c"]
        
        # 4. Obstacles
        for i in range(params.obs_blocking.shape[0]):
            obs_name = f"obs_entity_{i}"
            stat_headers += [f"{obs_name}_r", f"{obs_name}_c"]
            
        # End of row
        stat_headers += ["termination_reason", "max_satiation", "max_injury"]
    
    if not quiet:
        print(f"  [DEBUG] Starting Evaluation: {num_episodes} episodes, num_envs={num_envs}, effective={effective_num_envs}, Render={render_video}", flush=True)

    # Wrap execution in device context if provided
    device_context = jax.default_device(device) if device is not None else contextlib.nullcontext()
    
    with device_context:
        # --- Single-env path (num_envs==1 or effective_num_envs==1) ---
        if effective_num_envs == 1:
            _run_single_env_eval(
                model, params, config, num_episodes, seed, results_dir, checkpoint_pct,
                key, video_dir, breakdown, icon_config, episode_rewards, episode_lengths, all_frames,
                record_stats, stats_dir, stat_headers, action_map, params, max_steps=None,
                render_video=render_video, wandb_enabled=wandb_enabled, debug=debug, quiet=quiet,
                record_true_obs=record_true_obs,
            )
        else:
            # --- Parallel-env path (episode-ticket design) ---
            _run_parallel_env_eval(
                model, params, config, num_episodes, effective_num_envs, seed, results_dir, checkpoint_pct,
                key, video_dir, breakdown, icon_config, episode_rewards, episode_lengths,
                record_stats, stats_dir, stat_headers, action_map, params,
                render_video=render_video, wandb_enabled=wandb_enabled, debug=debug, quiet=quiet,
                record_true_obs=record_true_obs,
            )

    # Save Consolidated Video (single-env path fills all_frames; parallel path leaves it empty for now)
    last_video_path = None
    if render_video and all_frames:
        video_path = os.path.join(video_dir, f"eval_{checkpoint_pct}.mp4")
        fps = config.get('visualization.fps', 5)
        if debug:
            print(f"    [Video] Saving {len(all_frames)} frames to {video_path}...", end="", flush=True)
        from src.environment.renderer import save_jax_video
        save_jax_video(all_frames, video_path, fps=fps, quiet=quiet)
        if debug:
            print(" Done", flush=True)
        last_video_path = video_path
        if not quiet:
            print(f"  --- Consolidated Evaluation Video saved to: {video_path} ---", flush=True)
        if wandb_enabled and WANDB_AVAILABLE and wandb.run:
            from src.utils.wandb_utils import upload_video
            upload_video(video_path, episode=checkpoint_pct, step=checkpoint_pct, caption=f"Episode {checkpoint_pct}", quiet=True)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    return {
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "last_video_path": last_video_path
    }


def _run_single_env_eval(model, params, config, num_episodes, seed, results_dir, checkpoint_pct,
                         key, video_dir, breakdown, icon_config, episode_rewards, episode_lengths, all_frames,
                         record_stats, stats_dir, stat_headers, action_map, params_ref, max_steps,
                         render_video=False, wandb_enabled=False, debug=False, quiet=True,
                         record_true_obs=False):
    """Original single-env loop: one episode at a time."""
    from src.environment.sensor import get_observation_breakdown
    if render_video:
        from src.environment.renderer import render_jax_state
    max_steps = params_ref.max_steps if max_steps is None else max_steps
    ep_pbar = tqdm(total=num_episodes, desc="Evaluating Episodes", disable=quiet)
    for ep in range(num_episodes):
        if not quiet and debug:
            print(f"  --- Starting Evaluation Episode {ep+1}/{num_episodes} ---")
            
        # Reset env
        key, reset_key = jax.random.split(key)
        state = jax_reset(params_ref, reset_key)
        obs = get_observation(state, params_ref)
        
        done = False
        total_reward = 0.0
        step_count = 0
        
        # Initial state
        if model is not None and hasattr(model, 'initial_state'):
            h_state = model.initial_state(batch_size=None)
        else:
            h_state = None
        
        # Lists for deferred stats collection
        ep_jax_states = []
        ep_jax_infos = []
        ep_actions = []
        ep_rewards = []
        ep_obs = []
        ep_true_obs = [] if record_true_obs else None
        
        if record_stats:
            # Step 0 stats
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
            
            # Compute true obs once at step 0, gated by config, shared between CSV and video
            true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None
            if record_true_obs:
                ep_true_obs.append(true_obs)
        
        def get_sensory_viz(obs_vec, true_obs_vec=None):
            # Systematic iteration based on breakdown — guaranteed to match get_observation() order
            ptr = 0
            t_ptr = 0
            viz = []
            
            for sensor_name, dim in breakdown.items():
                if sensor_name == "Olfaction":
                    olf_obs = obs_vec[ptr:ptr+dim]
                    olf_true = true_obs_vec[t_ptr:t_ptr+dim] if true_obs_vec is not None else olf_obs
                    ptr += dim; t_ptr += dim
                    viz.append({'name': 'Olfactory', 'vector': olf_obs, 'true_vector': olf_true, 'type': 'spectrum', 'labels': ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']})
                
                elif sensor_name == "Extero Nociception":
                    noc_obs = float(obs_vec[ptr])
                    noc_true = float(true_obs_vec[t_ptr]) if true_obs_vec is not None else noc_obs
                    ptr += dim; t_ptr += dim
                    viz.append({'name': 'Extero Nociception', 'intensity': noc_obs, 'true_intensity': noc_true, 'color': '#c0392b', 'type': 'intensity'})
                
                elif sensor_name == "Collision":
                    coll_obs = obs_vec[ptr:ptr+dim]
                    coll_true = true_obs_vec[t_ptr:t_ptr+dim] if true_obs_vec is not None else coll_obs
                    ptr += dim; t_ptr += dim
                    viz.append({'name': 'Collision', 'vector': coll_obs, 'true_vector': coll_true, 'type': 'diamond', 'range': params_ref.sensor_range, 'num_features': 1})
                
                elif sensor_name == "Location":
                    loc_vec = obs_vec[ptr:ptr+dim]
                    ptr += dim; t_ptr += dim
                    viz.append({'name': 'LOC', 'value_text': f"({loc_vec[0]:.2f}, {loc_vec[1]:.2f})", 'color': '#ADB5BD', 'type': 'text'})
                
                elif sensor_name in ("Satiation", "Nutrition", "Injury"):
                    s_obs = float(obs_vec[ptr])
                    ptr += dim; t_ptr += dim
                    viz.append({'name': sensor_name, 'intensity': s_obs, 'type': 'intensity'})
                
                elif sensor_name == "Visual":
                    vis_obs = obs_vec[ptr:ptr+dim]
                    vis_true = true_obs_vec[t_ptr:t_ptr+dim] if true_obs_vec is not None else vis_obs
                    ptr += dim; t_ptr += dim
                    viz.append({'name': 'Visual', 'vector': vis_obs, 'true_vector': vis_true, 'type': 'visual_grid', 'num_features': 8, 'range': params_ref.visual_sensor_range, 'labels': ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']})
                
                elif sensor_name == "Proprioception":
                    proprio_vec = obs_vec[ptr:ptr+dim]
                    ptr += dim; t_ptr += dim
                    viz.append({'name': 'Proprioception', 'vector': proprio_vec, 'type': 'radial', 'color': '#be4bdb'})

            return viz

        if render_video:
            if debug: print(f"    [Render] Initial frame...", end="", flush=True)
            state_for_render = jax.device_get(state)
            # Use the already-computed true_obs (or None if diagnostics disabled)
            all_frames.append(render_jax_state(
                state_for_render, params_ref, episode=ep+1, step=0, 
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
                
                # Use algorithm-agnostic generic_inference
                action, log_prob, value, h_new, _ = generic_inference(model, obs_batch, h_state, eval_mode=True)
                h_state = h_new
                # Single-env: obs_batch is (1, D), so action can be (1,) — squeeze to scalar for int()
                action_idx = int(jnp.squeeze(action))
            else:
                # Random action if no model provided
                key, action_key = jax.random.split(key)
                action_idx = int(jax.random.randint(action_key, (), 0, action_dim))
            
            if debug:
                print(f" Done (Action: {action_idx}). Stepping env...", end="", flush=True)
            
            # Step
            next_state, reward, done, info = jax_step(state, action_idx, params_ref)
            total_reward += float(reward)
            step_count += 1
            
            state = next_state
            next_obs = get_observation(state, params_ref)
            
            if debug:
                print(f" Done. Reward: {reward:.2f}", flush=True)
            
            # Compute true obs once per step, gated by config
            true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None

            if render_video:
                if debug: print(f"    [Step {step_count}] Rendering...", end="", flush=True)
                state_for_render = jax.device_get(state)
                all_frames.append(render_jax_state(
                    state_for_render, params_ref, episode=ep+1, step=step_count, 
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
                if record_true_obs:
                    ep_true_obs.append(true_obs)
            
            obs = next_obs
            step_pbar.update(1)
        
        if record_stats and ep_jax_states:
            _write_episode_stats(stats_dir, ep + 1, ep_jax_states, ep_jax_infos, ep_actions,
                                 ep_rewards, ep_obs, stat_headers, action_map, params_ref, debug,
                                 ep_true_obs=ep_true_obs)

        step_pbar.close()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        if (debug or not WANDB_AVAILABLE) and not quiet:
            print(f"  --- Episode {ep+1}/{num_episodes} Complete | Steps: {step_count} | Reward: {total_reward:.2f} ---", flush=True)
        
        if render_video:
            for _ in range(5):
                all_frames.append(all_frames[-1])


def _run_parallel_env_eval(model, params, config, num_episodes, effective_num_envs, seed, results_dir, checkpoint_pct,
                          key, video_dir, breakdown, icon_config, episode_rewards, episode_lengths,
                          record_stats, stats_dir, stat_headers, action_map, params_ref,
                          render_video=False, wandb_enabled=False, debug=False, quiet=True,
                          record_true_obs=False):
    """Parallel env evaluation with episode-ticket design: only effective_num_envs run; when one finishes, refill if tickets remain."""
    from src.environment.wrapper import ParallelEnv
    
    penv = ParallelEnv(params_ref)
    max_steps = params_ref.max_steps
    rest_enabled = params_ref.rest_action_enabled
    eat_enabled = params_ref.eat_action_enabled
    action_dim = 4 + int(rest_enabled) + int(eat_enabled)
    
    # Initial reset: effective_num_envs envs (wrapper splits key internally)
    key, reset_key = jax.random.split(key)
    states, obs = penv.reset(reset_key, effective_num_envs)
    obs = jnp.array(obs)
    
    # Per-slot buffers for stats
    slot_states = [[] for _ in range(effective_num_envs)]
    slot_infos = [[] for _ in range(effective_num_envs)]
    slot_actions = [[] for _ in range(effective_num_envs)]
    slot_rewards = [[] for _ in range(effective_num_envs)]
    slot_obs = [[] for _ in range(effective_num_envs)]
    slot_true_obs = [[] for _ in range(effective_num_envs)] if record_true_obs else None
    
    if record_stats:
        for i in range(effective_num_envs):
            slot_states[i].append({
                'agent_pos': states.agent_pos[i], 'satiation': states.satiation[i], 'nutrition': states.nutrition[i],
                'injury_level': states.injury_level[i], 'rest_streak': states.rest_streak[i],
                'res_pos': states.res_pos[i], 'res_active': states.res_active[i],
                'pred_pos': states.pred_pos[i], 'neutral_pos': states.neutral_pos[i], 'obs_pos': states.obs_pos[i],
            })
            slot_infos[i].append({})
            slot_actions[i].append(-1)
            slot_rewards[i].append(0.0)
            slot_obs[i].append(obs[i])
            if record_true_obs:
                true_obs_i = get_observation(
                    jax.tree_util.tree_map(lambda x: x[i], states), params_ref, apply_noise=False
                )
                slot_true_obs[i].append(true_obs_i)
    
    h_state = model.initial_state(batch_size=effective_num_envs) if model is not None and hasattr(model, 'initial_state') else None
    completed_episodes = 0
    issued_tickets = effective_num_envs
    slot_active = [True] * effective_num_envs
    ep_pbar = tqdm(total=num_episodes, desc="Evaluating Episodes (parallel)", disable=quiet)
    safety_cap = max_steps * num_episodes * 2

    if not quiet:
        print(f"  [Ticket] Started: {effective_num_envs} envs running, {num_episodes} tickets (episodes to complete).", flush=True)
    if debug:
        print(f"  [Ticket] Step loop safety_cap={safety_cap}.", flush=True)
    
    for _step in range(safety_cap):
        if completed_episodes >= num_episodes:
            break
            
        if model is not None:
            action, _, _, h_state, _ = generic_inference(model, obs, h_state, eval_mode=True)
            actions = jnp.reshape(jnp.asarray(action, dtype=jnp.int32), (effective_num_envs,))
        else:
            key, subkey = jax.random.split(key)
            actions = jax.random.randint(subkey, (effective_num_envs,), 0, action_dim)
        
        next_states, next_obs, rewards, dones, infos = penv.step(states, actions)
        next_obs = jnp.array(next_obs)
        
        for i in range(effective_num_envs):
            if not slot_active[i]:
                continue
            slot_states[i].append({
                'agent_pos': next_states.agent_pos[i], 'satiation': next_states.satiation[i],
                'nutrition': next_states.nutrition[i], 'injury_level': next_states.injury_level[i],
                'rest_streak': next_states.rest_streak[i], 'res_pos': next_states.res_pos[i],
                'res_active': next_states.res_active[i], 'pred_pos': next_states.pred_pos[i],
                'neutral_pos': next_states.neutral_pos[i], 'obs_pos': next_states.obs_pos[i],
            })
            slot_infos[i].append({k: (v[i] if (hasattr(v, 'ndim') and v.ndim > 0) else v) for k, v in infos.items()})
            slot_actions[i].append(int(actions[i]))
            slot_rewards[i].append(float(rewards[i]))
            slot_obs[i].append(next_obs[i])
            if record_true_obs:
                true_obs_i = get_observation(
                    jax.tree_util.tree_map(lambda x: x[i], next_states), params_ref, apply_noise=False
                )
                slot_true_obs[i].append(true_obs_i)
        
        states = next_states
        obs = next_obs
        
        for i in range(effective_num_envs):
            if dones[i] and slot_active[i]:
                completed_episodes += 1
                total_r = sum(slot_rewards[i])
                episode_rewards.append(total_r)
                episode_lengths.append(len(slot_rewards[i]) - 1)
                tickets_left = num_episodes - completed_episodes
                if debug:
                    print(f"  [Ticket] Env {i} finished → episode {completed_episodes}/{num_episodes} written "
                          f"(reward={total_r:.2f}, steps={len(slot_rewards[i])-1}); tickets_left={tickets_left}.", flush=True)
                if record_stats and slot_states[i]:
                    _write_episode_stats(stats_dir, completed_episodes, slot_states[i], slot_infos[i],
                                         slot_actions[i], slot_rewards[i], slot_obs[i],
                                         stat_headers, action_map, params_ref, debug,
                                         ep_true_obs=slot_true_obs[i] if record_true_obs else None)
                ep_pbar.update(1)

                # Reset the slot buffer immediately
                slot_states[i] = []
                slot_infos[i] = []
                slot_actions[i] = []
                slot_rewards[i] = []
                slot_obs[i] = []
                if record_true_obs:
                    slot_true_obs[i] = []
                # Deactivate until reset
                slot_active[i] = False

                # Ticket used. Only give a new ticket (reset) if more tickets remain to be issued.
                if issued_tickets >= num_episodes:
                    if debug:
                        print(f"  [Ticket] All tickets issued; slot {i} remains inactive.", flush=True)
                    continue 
                
                if debug:
                    print(f"  [Ticket] Giving new ticket to slot {i} (reset). total_issued={issued_tickets+1}", flush=True)
                
                issued_tickets += 1
                key, reset_key = jax.random.split(key)
                new_state = jax_reset(params_ref, reset_key)
                states = jax.tree_util.tree_map(lambda x, y: x.at[i].set(y), states, new_state)
                new_obs = get_observation(new_state, params_ref)
                obs = obs.at[i].set(new_obs)
                if h_state is not None:
                    h_one = model.initial_state(batch_size=1)
                    h_state = jax.tree_util.tree_map(lambda a, b: a.at[i].set(b.squeeze(0)), h_state, h_one)
                
                # Reactivate and seed the buffer for the new episode (step 0)
                slot_active[i] = True
                slot_states[i] = [{'agent_pos': new_state.agent_pos, 'satiation': new_state.satiation, 'nutrition': new_state.nutrition,
                                   'injury_level': new_state.injury_level, 'rest_streak': new_state.rest_streak,
                                   'res_pos': new_state.res_pos, 'res_active': new_state.res_active,
                                   'pred_pos': new_state.pred_pos, 'neutral_pos': new_state.neutral_pos, 'obs_pos': new_state.obs_pos}]
                slot_infos[i] = [{}]
                slot_actions[i] = [-1]
                slot_rewards[i] = [0.0]
                slot_obs[i] = [new_obs]
                if record_true_obs:
                    true_obs_new = get_observation(new_state, params_ref, apply_noise=False)
                    slot_true_obs[i] = [true_obs_new]
        
        if completed_episodes >= num_episodes:
            if debug:
                print(f"  [Ticket] Exiting step loop (completed_episodes={completed_episodes}).", flush=True)
            break
    
    ep_pbar.close()
    if not quiet:
        if completed_episodes >= num_episodes:
            print(f"  [Ticket] Done: {completed_episodes} episodes completed (all tickets used).", flush=True)
        else:
            print(f"  [Ticket] Done: {completed_episodes}/{num_episodes} episodes (safety cap or early exit).", flush=True)


def main():
    pass

if __name__ == "__main__":
    main()
