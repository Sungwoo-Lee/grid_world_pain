import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.environment.jax_env.core import jax_step, jax_reset
from src.environment.jax_env.sensor import get_observation
from src.models.jax_models.recurrent_ppo_network import get_action_and_value_nnx
from src.utils.wandb_utils import upload_video

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, checkpoint_pct, render_video=False, wandb_enabled=False, debug=False):
    """
    Runs deterministic evaluation episodes using the JAX model.
    """
    from src.environment.jax_env.sensor import get_observation_breakdown
    
    key = jax.random.PRNGKey(seed)
    
    # Video output setup
    video_dir = os.path.join(results_dir, "videos")
    if render_video:
        os.makedirs(video_dir, exist_ok=True)
        from src.environment.jax_env.renderer import render_jax_state, save_jax_video
    
    # Breakdown for sensory visualization
    breakdown = get_observation_breakdown(params)
    
    episode_rewards = []
    episode_lengths = []
    all_frames = []
    
    for ep in range(num_episodes):
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
        
        def get_sensory_viz(obs_vec):
            # Internal helper to slice flat obs into renderer-friendly format
            viz = [
                {'name': 'Chemical', 'vector': obs_vec[0:breakdown['Chemical']], 'color': '#40C057', 'type': 'spectrum'},
                {'name': 'Collision', 'vector': obs_vec[breakdown['Chemical']:breakdown['Chemical']+breakdown['Collision']], 'color': '#FA5252', 'type': 'radial'}
            ]
            return viz

        if render_video:
            if debug: print(f"    [Render] Initial frame...", end="", flush=True)
            all_frames.append(render_jax_state(
                state, params, episode=ep+1, step=0, 
                sensory_data=get_sensory_viz(obs)
            ))
            if debug: print(" Done", flush=True)
        
        while not done and step_count < max_steps:
            if debug:
                print(f"    [Step {step_count}] Inferencing...", end="", flush=True)
            
            # Model inference (deterministic in eval_mode)
            obs_batch = obs[None, :]  # Add batch dim
            
            if h_state is not None:
                action, log_prob, value, h_new = get_action_and_value_nnx(model, obs_batch, h_state, eval_mode=True)
                h_state = h_new
            else:
                action, _, _, _ = get_action_and_value_nnx(model, obs_batch, None, eval_mode=True)
            
            action_idx = int(action)
            
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
                # We show the sensory data that CAUSED the action (obs) or the result?
                # PyTorch baseline shows the state result of the action.
                all_frames.append(render_jax_state(
                    state, params, episode=ep+1, step=step_count, 
                    action=action_idx, sensory_data=get_sensory_viz(next_obs)
                ))
                if debug: print(" Done", flush=True)
            
            obs = next_obs
        
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
        save_jax_video(all_frames, video_path, fps=fps, quiet=True)
        if debug: print(" Done", flush=True)
        last_video_path = video_path
        
        # Upload to WandB if enabled
        if wandb_enabled and WANDB_AVAILABLE and wandb.run:
            from src.utils.wandb_utils import upload_video
            upload_video(video_path, episode=checkpoint_pct, caption=f"Eval Video {checkpoint_pct}", quiet=True)
    
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
