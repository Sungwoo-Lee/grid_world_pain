"""
JAX Evaluation Script for GridWorld RL Agents.

This script mirrors evaluation.py but uses JAX-native components:
1. Loads configuration saved during training.
2. Loads JAX model checkpoint.
3. Runs deterministic evaluation episodes.
4. Generates evaluation statistics.

Arguments:
- `--results_dir <path>`: (Required) Path to results directory of the run.
- `--episodes <int>`: Number of evaluation episodes.
- `--seed <int>`: Override testing seed.
- `--checkpoint <str>`: Specific checkpoint to evaluate.
- `--wandb-run-path <str>`: WandB run path for uploads.

Usage:
    python evaluation_jax.py --results_dir results/JAX_RecurrentPPO/my_run --episodes 10
"""
import os
import argparse
import glob
import re
import yaml
import jax
import jax.numpy as jnp
from flax import nnx

from src.environment.jax_env.config_loader import load_env_params
from src.environment.jax_env.wrapper import ParallelEnv
from src.environment.jax_env.core import jax_step, jax_reset
from src.environment.jax_env.sensor import get_observation
from src.models.jax_models.recurrent_ppo_network import ActorCriticRNN, get_action_and_value_nnx
from src.utils.config import Config

# Optional WandB
try:
    import wandb
    from src.utils.wandb_utils import wandb_login
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, checkpoint_pct, render_video=False):
    """
    Runs deterministic evaluation episodes using the JAX model.
    """
    key = jax.random.PRNGKey(seed)
    
    # Video output setup
    video_dir = os.path.join(results_dir, "videos", f"ckpt_{checkpoint_pct}")
    if render_video:
        os.makedirs(video_dir, exist_ok=True)
        from src.environment.jax_env.renderer import render_jax_state, save_jax_video
    
    # For evaluation, we use a single environment for clearer logging
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state = jax_reset(params, reset_key)
        obs = get_observation(state, params)
        
        # Initialize hidden state
        h_state = model.initial_state(batch_size=1)
        
        total_reward = 0.0
        step_count = 0
        done = False
        max_steps = params.max_steps
        
        frames = []
        if render_video:
            frames.append(render_jax_state(state, params, episode=ep+1, step=0))
        
        while not done and step_count < max_steps:
            # Model inference (deterministic in eval_mode)
            obs_batch = obs[None, :]  # Add batch dim
            action, log_prob, value, h_new = get_action_and_value_nnx(model, obs_batch, h_state, eval_mode=True)
            action = int(action)  # Extract from 0-dim array
            
            # Step
            next_state, reward, done, info = jax_step(state, action, params)
            total_reward += float(reward)
            step_count += 1
            
            state = next_state
            obs = get_observation(state, params)
            h_state = h_new
            
            if render_video:
                frames.append(render_jax_state(state, params, episode=ep+1, step=step_count))
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        print(f"Episode {ep+1}/{num_episodes} | Steps: {step_count} | Reward: {total_reward:.2f}")
        
        if render_video and frames:
            video_path = os.path.join(video_dir, f"eval_ep_{ep+1:03d}.mp4")
            save_jax_video(frames, video_path, fps=5, quiet=True)
            print(f"  Saved video: {os.path.basename(video_path)}")
    
    # Statistics
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    mean_length = sum(episode_lengths) / len(episode_lengths)
    
    print(f"\n--- Evaluation Summary (Checkpoint {checkpoint_pct}%) ---")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Std Reward: {jnp.std(jnp.array(episode_rewards)):.2f}")
    
    return {
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }


def main():
    parser = argparse.ArgumentParser(description="JAX GridWorld Evaluation")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory (Required)")
    parser.add_argument("--episodes", type=int, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, help="Override testing seed")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint name or path to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints found in the directory")
    parser.add_argument("--wandb-run-path", type=str, help="WandB run path (e.g. 'entity/project/run_id') for uploads")
    parser.add_argument("--render-video", action="store_true", help="Enable video recording")
    args = parser.parse_args()

    results_dir = args.results_dir
    models_dir = os.path.join(results_dir, "models")
    config_path = os.path.join(models_dir, "config.yaml")

    # 1. Load saved configuration
    if not os.path.exists(config_path):
        print(f"Error: Training configuration file not found at {config_path}")
        return

    print(f"Loading training configuration from {config_path}...")
    with open(config_path, 'r') as f:
        saved_config_dict = yaml.safe_load(f)
        config = Config(saved_config_dict)

    # 1.1 Merge evaluation defaults (Strictly)
    eval_default_path = "configs/evaluation/default.yaml"
    if os.path.exists(eval_default_path):
        eval_defaults = Config.load_yaml(eval_default_path)
        config.merge(eval_defaults)
    else:
        # If missing, we must have them via CLI or saved config
        pass

    # 2. Resolve Parameters (No Safe Defaults)
    seed = args.seed or config.get_mandatory('testing.seed')
    num_episodes = args.episodes or config.get_mandatory('testing.evaluation_episodes')
    algorithm = config.get_mandatory('agent.algorithm')
    
    # Reconstruct JAX EnvParams from saved configuration
    # Note: load_env_params handles the mapping from YAML structure to JAX arrays
    params = load_env_params(config_path)

    # 3. Print Summary
    print(f"\n{'='*50}")
    print(f"JAX Evaluation: {algorithm}")
    print(f"{'='*50}")
    print(f"Grid: {params.height}x{params.width}")
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print(f"Video Rendering: {'Enabled' if args.render_video else 'Disabled'}")
    print(f"{'='*50}\n")

    # 4. Find checkpoints
    # Note: Modern Orbax just uses iteration numbers as folder names.
    checkpoints = []
    
    if args.checkpoint:
        # Explicit path or numeric iteration
        ckpt_path = os.path.join(models_dir, args.checkpoint)
        if os.path.isdir(ckpt_path):
            checkpoints.append(ckpt_path)
        else:
            print(f"Error: Checkpoint '{args.checkpoint}' not found at {ckpt_path}")
            return
    elif args.all:
        # Find all iteration subdirectories
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()]
        checkpoints = [os.path.join(models_dir, d) for d in sorted(subdirs, key=int)]
    else:
        # Latest numeric subdirectory
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()]
        if subdirs:
            latest = max(subdirs, key=int)
            checkpoints = [os.path.join(models_dir, latest)]

    if not checkpoints:
        print(f"No valid checkpoints found in {models_dir}")
        return

    # 5. Initialize WandB if requested
    if args.wandb_run_path and WANDB_AVAILABLE:
        try:
            wandb_login(quiet=True)
            path_parts = args.wandb_run_path.strip().split('/')
            if len(path_parts) == 3:
                wandb.init(entity=path_parts[0], project=path_parts[1], id=path_parts[2], resume="must", job_type="evaluation")
            else:
                wandb.init(id=args.wandb_run_path, resume="must", job_type="evaluation")
        except Exception as e:
            print(f"WandB init failed: {e}")

    # 6. Evaluate each checkpoint
    import orbax.checkpoint as ocp
    
    # Matching CheckpointManager setup
    checkpointer = ocp.CheckpointManager(
        os.path.abspath(models_dir),
        checkpointers=ocp.StandardCheckpointer()
    )
    
    for ckpt_path in checkpoints:
        iteration_str = os.path.basename(ckpt_path)
        iteration = int(iteration_str)
        print(f"\nEvaluating Iteration: {iteration}")
        
        # Reconstruct Model based on algorithm
        test_state = jax_reset(params, jax.random.PRNGKey(seed))
        obs = get_observation(test_state, params)
        input_dim = obs.shape[0]
        action_dim = 4
        
        rngs = nnx.Rngs(jax.random.PRNGKey(seed))
        
        if algorithm == "RecurrentPPO":
            model = ActorCriticRNN(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=config.get_mandatory('agent.hidden_size'),
                rngs=rngs
            )
            # Restore via manager (returns item named 'default' if used as single checkpointer)
            restored = checkpointer.restore(iteration)
            nnx.update(model, restored) # restored is already the Pytree if single item
            
        elif algorithm == "DreamerV3":
            from src.models.jax_models.dreamer_v3_trainer import DreamerTrainer
            dreamer_config = {
                'model_lr': config.get_mandatory('agent.model_lr'),
                'actor_lr': config.get_mandatory('agent.actor_lr'),
                'value_lr': config.get_mandatory('agent.value_lr'),
                'batch_size': config.get_mandatory('agent.batch_size'),
                'batch_length': config.get_mandatory('agent.batch_length'),
            }
            trainer = DreamerTrainer(input_dim, action_dim, dreamer_config, rngs=rngs)
            restored = checkpointer.restore(iteration)
            # restored is a dict/pytree
            nnx.update(trainer.agent.wm, restored['wm'])
            nnx.update(trainer.agent.ac.actor, restored['actor'])
            nnx.update(trainer.agent.ac.critic, restored['critic'])
            model = trainer.agent 
        else:
            raise ValueError(f"Unsupported algorithm for JAX evaluation: {algorithm}")
            
        evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, iteration, render_video=args.render_video)

    checkpointer.close()

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    print(f"\n{'='*50}")
    print(f"Evaluation Complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
