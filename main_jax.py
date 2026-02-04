"""
JAX Debug Sandbox Script.

Purpose:
- Verify the JAX Environment Mechanics (Grid + Body) without any Learning Agent.
- Uses a Random Agent to walk around.
- Checks if resource interactions, injury, and death occur correctly.
- Renders video frames for visualization.

Arguments:
- `--episodes <int>`: Number of episodes to record.
- `--max_steps <int>`: Maximum steps per episode.
- `--seed <int>`: Random seed for reproducibility.
- `--config <path>`: Path to config YAML.
- `--num-envs <int>`: Number of parallel environments.
- `--render-video`: Enable video recording.
- `--output-dir <path>`: Directory for video output.

Usage:
    python main_jax.py --config configs/ablation/homeostatic/04_nociception.yaml --episodes 3 --render-video
"""
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from datetime import datetime

from src.environment.jax_env.state import EnvParams, EnvState
from src.environment.jax_env.core import jax_step, jax_reset
from src.environment.jax_env.sensor import get_observation
from src.environment.jax_env.config_loader import load_env_params
from src.utils.config import get_default_config, Config

def main():
    parser = argparse.ArgumentParser(description="JAX GridWorld Debug Sandbox")
    parser.add_argument("--episodes", type=int, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (default: 1 for debug)")
    parser.add_argument("--tag", type=str, default="jax_sandbox", help="Tag for the run")
    
    # Video options
    parser.add_argument("--render-video", action="store_true", help="Enable video recording")
    parser.add_argument("--output-dir", type=str, help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=5, help="Video FPS (default: 5)")
    
    args = parser.parse_args()

    # 1. Load Config (same hierarchy as main.py)
    config = get_default_config()

    # Merge Training Defaults
    train_config_path = os.path.join(os.path.dirname(__file__), "configs", "train", "default.yaml")
    if os.path.exists(train_config_path):
        train_defaults = Config.load_yaml(train_config_path)
        config.merge(train_defaults)
        print(f"Loaded training defaults from {train_config_path}")

    # Merge Evaluation Defaults
    eval_config_path = os.path.join(os.path.dirname(__file__), "configs", "evaluation", "default.yaml")
    if os.path.exists(eval_config_path):
        eval_defaults = Config.load_yaml(eval_config_path)
        config.merge(eval_defaults)
        print(f"Loaded evaluation defaults from {eval_config_path}")

    # Merge User / Ablation Config (Overrides everything)
    if args.config:
        print(f"Loading user/ablation config from: {args.config}")
        user_config = Config.load_yaml(args.config)
        config.merge(user_config)

    # Resolve Parameters (Argument > Config > Error)
    def resolve_param(arg_val, config_key, type_converter=None):
        if arg_val is not None:
            config.set(config_key, arg_val)
            return arg_val
        return config.get_mandatory(config_key, type_converter)

    # Essential Params
    episodes = int(resolve_param(args.episodes, 'testing.evaluation_episodes'))
    max_steps = int(resolve_param(args.max_steps, 'environment.max_steps'))
    seed = int(resolve_param(args.seed, 'training.seed'))
    num_envs = args.num_envs

    # Load JAX EnvParams
    if args.config:
        params = load_env_params(args.config)
    else:
        # Fallback to default environment config
        default_env_config = os.path.join(os.path.dirname(__file__), "configs", "environment", "environment.yaml")
        if os.path.exists(default_env_config):
            print(f"No config provided. Using default: {default_env_config}")
            params = load_env_params(default_env_config)
        else:
            raise ValueError(f"--config not provided and default config not found at {default_env_config}")

    # Setup video output directory
    if args.render_video:
        from src.environment.jax_env.renderer import render_jax_state, save_jax_video
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.output_dir or os.path.join("results", "JAX_Sandbox", f"{args.tag}_{timestamp}")
        # Only create if using default path or specific non-existing path
        if not args.output_dir or not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
        print(f"Video output: {output_dir}")

    # Initialize JAX
    key = jax.random.PRNGKey(seed)
    
    # Display Info
    print(f"\n{'='*50}")
    print(f"JAX GridWorld Debug Sandbox")
    print(f"{'='*50}")
    print(f"Grid Size: {params.height}x{params.width}")
    print(f"Max Steps: {max_steps}")
    print(f"Episodes: {episodes}")
    print(f"Parallel Envs: {num_envs}")
    print(f"Seed: {seed}")
    print(f"Video Recording: {'Enabled' if args.render_video else 'Disabled'}")
    print(f"{'='*50}\n")

    # Single Env mode (required for video)
    if num_envs == 1:
        for ep in range(episodes):
            print(f"\n--- Episode {ep+1}/{episodes} ---")
            key, reset_key = jax.random.split(key)
            state = jax_reset(params, reset_key)
            
            obs = get_observation(state, params)
            print(f"Initial Obs Shape: {obs.shape}")
            print(f"Agent Pos: {tuple(state.agent_pos.tolist())}")
            print(f"Satiation: {float(state.satiation):.1f}")
            print(f"Injury: {float(state.injury_level):.1f}")
            
            # Frame collection for video
            frames = []
            if args.render_video:
                frames.append(render_jax_state(state, params, episode=ep+1, step=0))
            
            total_reward = 0.0
            for step in range(max_steps):
                # Random action (0-3 for UP/DOWN/LEFT/RIGHT)
                key, action_key = jax.random.split(key)
                action = jax.random.randint(action_key, (), 0, 4)
                
                # Step
                next_state, reward, done, info = jax_step(state, action, params)
                total_reward += float(reward)
                
                # Render frame
                if args.render_video:
                    frames.append(render_jax_state(next_state, params, episode=ep+1, step=step+1))
                
                # Log interesting events
                if info.get('ate_food', False):
                    print(f"  Step {step+1}: Ate food! Reward: {float(reward):.2f}")
                if info.get('hit_danger', False):
                    print(f"  Step {step+1}: Hit danger! Injury: {float(next_state.injury_level):.1f}")
                if info.get('hit_predator', False):
                    print(f"  Step {step+1}: Hit predator! Injury: {float(next_state.injury_level):.1f}")
                
                state = next_state
                
                if done:
                    print(f"  Terminated at step {step+1}. Reason: {info.get('termination_reason', 'unknown')}")
                    break
            
            print(f"Episode {ep+1} | Steps: {step+1} | Total Reward: {total_reward:.2f}")
            
            # Save video for this episode
            if args.render_video and len(frames) > 0:
                video_path = os.path.join(output_dir, f"episode_{ep+1:03d}.mp4")
                save_jax_video(frames, video_path, fps=args.fps, quiet=False)
    else:
        # Vectorized mode (no video support)
        if args.render_video:
            print("Warning: Video rendering not supported with --num-envs > 1")
            
        from src.environment.jax_env.wrapper import ParallelEnv
        env = ParallelEnv(params)
        
        for ep in range(episodes):
            print(f"\n--- Episode {ep+1}/{episodes} ({num_envs} envs) ---")
            key, reset_key = jax.random.split(key)
            states, obs = env.reset(reset_key, num_envs)
            
            print(f"Obs Shape: {obs.shape}")
            
            total_rewards = jnp.zeros(num_envs)
            active = jnp.ones(num_envs, dtype=jnp.bool_)
            
            for step in range(max_steps):
                # Random actions
                key, action_key = jax.random.split(key)
                actions = jax.random.randint(action_key, (num_envs,), 0, 4)
                
                # Step
                states, obs, rewards, dones, infos = env.step(states, actions)
                total_rewards = total_rewards + rewards * active
                active = active & ~dones
                
                if not jnp.any(active):
                    break
            
            mean_reward = float(jnp.mean(total_rewards))
            print(f"Episode {ep+1} | Mean Reward: {mean_reward:.2f}")

    print("\n" + "="*50)
    print("JAX Sandbox Complete!")
    if args.render_video and num_envs == 1:
        print(f"Videos saved to: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
