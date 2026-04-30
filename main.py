"""
JAX Debug Sandbox Script.

Purpose:
- Verify the JAX Environment Mechanics (Grid + Body) without any Learning Agent.
- Mimics the architecture of train.py but uses a Random Agent.
- Uses centralized evaluation logic for video rendering.

Usage:
    python main.py --config configs/environment/default.yaml --episodes 3
    python main.py --no-render --num-envs 256  # High-speed throughput test
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import jax
import jax.numpy as jnp
from datetime import datetime
from tqdm import tqdm

from src.environment.config_loader import load_env_params
from src.environment.wrapper import ParallelEnv
from src.utils.config import get_default_config, Config
from src.utils.evaluation_core import evaluate_jax_checkpoint

def main():
    parser = argparse.ArgumentParser(description="JAX GridWorld Debug Sandbox")
    parser.add_argument("--episodes", type=int, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to base config YAML")
    parser.add_argument("--num-envs", type=int, help="Number of parallel envs (ignored if rendering video)")
    parser.add_argument("--tag", type=str, default="jax_sandbox", help="Tag for the run")
    
    # Video options
    parser.add_argument("--no-render", action="store_true", help="Disable video recording")
    parser.add_argument("--output-dir", type=str, help="Output directory for videos")
    
    args = parser.parse_args()

    # 1. Load Config (same hierarchy as train.py)
    config = get_default_config()

    # Merge Training Defaults
    train_config_path = os.path.join(os.path.dirname(__file__), "configs", "train", "default.yaml")
    if os.path.exists(train_config_path):
        train_defaults = Config.load_yaml(train_config_path)
        config.merge(train_defaults)

    # Merge Evaluation Defaults
    eval_config_path = os.path.join(os.path.dirname(__file__), "configs", "evaluation", "default.yaml")
    if os.path.exists(eval_config_path):
        eval_defaults = Config.load_yaml(eval_config_path)
        config.merge(eval_defaults)

    # Merge User / Ablation Config
    if args.config:
        user_config = Config.load_yaml(args.config)
        config.merge(user_config)

    # 2. Resolve Parameters
    def resolve_param(arg_val, config_key, type_converter=None):
        if arg_val is not None:
            config.set(config_key, arg_val)
            return arg_val
        return config.get_mandatory(config_key, type_converter)

    episodes = int(resolve_param(args.episodes, 'testing.evaluation_episodes'))
    max_steps = int(resolve_param(args.max_steps, 'environment.max_steps'))
    seed = int(resolve_param(args.seed, 'training.seed'))
    num_envs = args.num_envs or config.get('training.num_envs', 1)
    
    params = load_env_params(config)
    render_video = not args.no_render

    # Display Header
    print(f"\n{'='*50}")
    print(f"JAX GridWorld Debug Sandbox")
    print(f"{'='*50}")
    print(f"Grid Size: {params.height}x{params.width}")
    print(f"Episodes: {episodes}")
    print(f"Max Steps: {max_steps}")
    print(f"Video Rendering: {'Enabled (Modular)' if render_video else 'Disabled (Vectorized)'}")
    print(f"{'='*50}\n")

    if render_video:
        # Use centralized evaluation logic for video
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = args.output_dir or os.path.join("results", "JAX_Sandbox", f"{args.tag}_{timestamp}")
        
        # evaluation_core handles its own tqdm
        evaluate_jax_checkpoint(
            model=None, # None = Random Agent in evaluation_core
            params=params,
            config=config,
            num_episodes=episodes,
            seed=seed,
            results_dir=results_dir,
            checkpoint_pct=0,
            render_video=True,
            num_envs=num_envs,
            quiet=False,
            debug=False
        )
    else:
        # High-speed Vectorized mode using ParallelEnv (mirroring train.py architecture)
        print(f"Running vectorized rollout with {num_envs} environments...")
        env = ParallelEnv(params)
        key = jax.random.PRNGKey(seed)
        
        action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)
        
        t_start = datetime.now()
        for ep in range(episodes):
            key, reset_key = jax.random.split(key)
            states, obs = env.reset(reset_key, num_envs)
            
            active = jnp.ones(num_envs, dtype=jnp.bool_)
            total_reward = jnp.zeros(num_envs)
            
            for _ in range(max_steps):
                key, action_key = jax.random.split(key)
                actions = jax.random.randint(action_key, (num_envs,), 0, action_dim)
                
                states, obs, rewards, dones, infos = env.step(states, actions)
                total_reward += rewards * active
                active = active & ~dones
                
                if not jnp.any(active):
                    break
            
            if (ep + 1) % max(1, episodes // 5) == 0:
                print(f"  Batch {ep+1}/{episodes} | Mean Reward: {jnp.mean(total_reward):.2f}")
        
        t_end = datetime.now()
        duration = (t_end - t_start).total_seconds()
        fps = (episodes * num_envs * max_steps) / duration if duration > 0 else 0
        print(f"\nVectorized Rollout Complete in {duration:.2f}s ({fps:.0f} steps/s)")

    print("\n" + "="*50)
    print("Sandbox Complete!")
    print("="*50)

if __name__ == "__main__":
    main()
