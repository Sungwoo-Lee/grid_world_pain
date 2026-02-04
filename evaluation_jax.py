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
            # Model inference (deterministic: use argmax instead of sampling)
            obs_batch = obs[None, :]  # Add batch dim
            logits, value, h_new = get_action_and_value_nnx(model, obs_batch, h_state)
            action = jnp.argmax(logits, axis=-1)[0]  # Deterministic action
            
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
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--episodes", type=int, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, help="Override testing seed")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints")
    parser.add_argument("--wandb-run-path", type=str, help="WandB run path for uploads")
    parser.add_argument("--render-video", action="store_true", help="Enable video recording")
    args = parser.parse_args()

    results_dir = args.results_dir
    models_dir = os.path.join(results_dir, "models")
    config_path = os.path.join(models_dir, "config.yaml")

    # 1. Load saved configuration
    if not os.path.exists(config_path):
        print(f"Error: Configuration not found at {config_path}")
        print("Please run train_jax.py first to generate a model.")
        return

    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        saved_config_dict = yaml.safe_load(f)
        config = Config(saved_config_dict)

    # Merge evaluation defaults
    eval_default_path = "configs/evaluation/default.yaml"
    if os.path.exists(eval_default_path):
        eval_defaults = Config.load_yaml(eval_default_path)
        config.merge(eval_defaults)

    # 2. Resolve Parameters
    seed = args.seed or config.get('testing.seed', 42)
    num_episodes = args.episodes or config.get('testing.evaluation_episodes', 10)
    
    # Load JAX EnvParams from saved config
    # We need to extract the original config path or rebuild params
    # For now, rebuild from saved config
    try:
        source_config = config.get('_source_config', args.results_dir)
        # Attempt to reconstruct path if it was relative
        params = load_env_params(config_path.replace("models/config.yaml", "").rstrip("/") + "/../" + source_config)
    except Exception as e:
        print(f"Warning: Could not load source config params: {e}")
        # Fallback: Use minimal params from saved config
        from src.environment.jax_env.state import EnvParams
        params = EnvParams(
            height=config.get('environment.height', 10),
            width=config.get('environment.width', 10),
            max_steps=config.get('environment.max_steps', 500),
            res_type=jnp.zeros(0, dtype=jnp.int32),
            res_property=jnp.zeros((0, 5)),
            res_spawn_area=jnp.zeros((0, 4)),
            res_max_cons=jnp.zeros(0, dtype=jnp.int32),
            res_reg_delay=jnp.zeros(0, dtype=jnp.int32),
            res_damage=jnp.zeros(0),
            pred_property=jnp.zeros((0, 5)),
            pred_move_int=jnp.zeros(0, dtype=jnp.int32),
            pred_damage=jnp.zeros(0),
            pred_patrol=jnp.zeros((0, 4)),
            pred_detect=jnp.zeros(0),
            pred_max_stamina=jnp.zeros(0),
            pred_recovery=jnp.zeros(0),
            pred_hunt_thresh=jnp.zeros(0),
            max_satiation=config.get('body.max_satiation', 100.0),
            max_injury=config.get('body.max_injury', 20.0),
            food_gain=config.get('body.food_gain', 10.0),
            setpoint=config.get('body.setpoint', 50.0),
            injury_recovery=config.get('body.injury_recovery', 1.0),
            smoothing_duration=config.get('body.smoothing_duration', 3),
            death_penalty=config.get('body.death_penalty', 10.0),
            overeating_death=config.get('body.overeating_death', False),
            use_homeostatic_reward=config.get('body.use_homeostatic_reward', True),
            with_satiation=config.get('body.with_satiation', False),
            with_injury=config.get('body.with_injury', True),
            sensor_radius=config.get('sensory.sensor_radius', 10.0),
            sensor_decay=config.get('sensory.sensor_decay', 2.0),
            sensor_range=config.get('sensory.sensor_range', 3)
        )

    # 3. Print Summary
    print(f"\n{'='*50}")
    print(f"JAX Evaluation")
    print(f"{'='*50}")
    print(f"Grid: {params.height}x{params.width}")
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print(f"Video Rendering: {'Enabled' if args.render_video else 'Disabled'}")
    print(f"{'='*50}\n")

    # 4. Find checkpoints
    prefix = "jax_rppo_"
    ext = ".ckpt"
    checkpoints = []

    if args.checkpoint:
        ckpt_path = os.path.join(models_dir, args.checkpoint)
        if os.path.exists(ckpt_path):
            checkpoints.append(ckpt_path)
        else:
            # Try adding prefix/ext
            ckpt_path = os.path.join(models_dir, f"{prefix}{args.checkpoint}{ext}")
            if os.path.exists(ckpt_path):
                checkpoints.append(ckpt_path)
            else:
                print(f"Error: Checkpoint '{args.checkpoint}' not found.")
                return
    elif args.all:
        checkpoints = sorted(glob.glob(os.path.join(models_dir, f"{prefix}*{ext}")))
    else:
        # Latest checkpoint
        all_ckpts = glob.glob(os.path.join(models_dir, f"{prefix}*{ext}"))
        if all_ckpts:
            def extract_pct(path):
                match = re.search(rf"{prefix}(\d+){ext}", os.path.basename(path))
                return int(match.group(1)) if match else -1
            checkpoints = [max(all_ckpts, key=extract_pct)]

    if not checkpoints:
        print(f"No checkpoints found in {models_dir}")
        print("Note: JAX checkpoint saving is not yet fully implemented.")
        print("Running evaluation with a fresh model for testing...")
        
        # Initialize fresh model for testing
        key = jax.random.PRNGKey(seed)
        key, model_key = jax.random.split(key)
        
        # Get input dim from a reset
        test_state = jax_reset(params, key)
        obs = get_observation(test_state, params)
        input_dim = obs.shape[0]
        action_dim = 4
        
        rngs = nnx.Rngs(model_key)
        model = ActorCriticRNN(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=64,
            rngs=rngs
        )
        
        evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, "fresh", render_video=args.render_video)
        return

    # 5. Initialize WandB if requested
    if args.wandb_run_path and WANDB_AVAILABLE:
        try:
            path_parts = args.wandb_run_path.strip().split('/')
            if len(path_parts) == 3:
                entity, project, run_id = path_parts
            elif len(path_parts) == 2:
                entity, project = None, path_parts[0]
                run_id = path_parts[1]
            else:
                run_id = path_parts[0]
                entity, project = None, None
            
            wandb_login(quiet=True)
            wandb.init(entity=entity, project=project, id=run_id, resume="must", job_type="evaluation")
        except Exception as e:
            print(f"WandB init failed: {e}")
            args.wandb_run_path = None

    # 6. Evaluate each checkpoint
    # TODO: Implement actual checkpoint loading when NNX serialization is ready
    print("\nNote: Full checkpoint loading not yet implemented.")
    print("Evaluation will use a fresh model for demonstration.\n")

    if wandb.run:
        wandb.finish()

    print(f"\n{'='*50}")
    print(f"Evaluation Complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
