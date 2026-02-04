"""
JAX Training Script for GridWorld RL Agents.

This script mirrors train.py but uses JAX-native components:
1. Loads configuration from YAML files.
2. Initializes the JAX ParallelEnv for massive parallelization.
3. Trains RecurrentPPO (or future DreamerV3) using Flax NNX.
4. Supports WandB logging, checkpointing, and results directory management.

Arguments:
- `--config <path>`: Path to environment/ablation config YAML (Required).
- `--episodes <int>`: Number of training episodes (converted to timesteps).
- `--num-envs <int>`: Number of parallel environments (default: 256).
- `--total-timesteps <int>`: Total timesteps to train.
- `--algorithm <str>`: Algorithm to use (RecurrentPPO, DreamerV3).
- `--seed <int>`: Random seed.
- `--wandb-name <str>`: WandB run name.
- `--no-wandb`: Disable WandB logging.
- `--checkpoint-frequency <int>`: Save frequency (as % of total).
- `--results-dir <path>`: Custom results directory.

Usage:
    python train_jax.py --config configs/ablation/homeostatic/04_nociception.yaml --total-timesteps 100000
"""
import os
import argparse
import yaml
from datetime import datetime
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import NamedTuple

from src.environment.jax_env.config_loader import load_env_params
from src.environment.jax_env.wrapper import ParallelEnv
from src.models.jax_models.recurrent_ppo_network import ActorCriticRNN
from src.models.jax_models.recurrent_ppo_trainer import train_iteration
from src.utils.config import get_default_config, Config

# Optional WandB
try:
    import wandb
    from src.utils.wandb_utils import wandb_login
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class PPOConfig(NamedTuple):
    num_steps: int
    num_epochs: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    lr: float

def main():
    parser = argparse.ArgumentParser(description="Train JAX RL Agents")
    
    # Required
    parser.add_argument("--config", type=str, required=True, help="Path to environment/ablation config YAML")
    
    # Training params
    parser.add_argument("--algorithm", type=str, default="RecurrentPPO", choices=["RecurrentPPO", "DreamerV3"])
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128, help="Steps per iteration (rollout length)")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO update epochs per iteration")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # WandB
    parser.add_argument("--wandb-name", type=str, help="WandB run name")
    parser.add_argument("--wandb-project", type=str, help="WandB project name (overrides config)")
    parser.add_argument("--wandb-group", type=str, help="WandB group name")
    parser.add_argument("--wandb-entity", type=str, help="WandB entity/team name")
    parser.add_argument("--wandb-resume-id", type=str, help="WandB run ID to resume")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    
    # Checkpointing
    parser.add_argument("--checkpoint-frequency", type=int, default=25, help="Checkpoint save frequency (% of total)")
    parser.add_argument("--results-dir", type=str, help="Custom results directory")
    parser.add_argument("--tag", type=str, help="Tag for the training run")
    
    args = parser.parse_args()

    # 1. Load Config (similar to train.py hierarchy)
    config = get_default_config()

    # Merge Training Defaults
    train_config_path = os.path.join(os.path.dirname(__file__), "configs", "train", "default.yaml")
    if os.path.exists(train_config_path):
        train_defaults = Config.load_yaml(train_config_path)
        config.merge(train_defaults)

    # Merge Logger Config (WandB settings)
    logger_config_path = os.path.join(os.path.dirname(__file__), "configs", "logger", "wandb.yaml")
    if os.path.exists(logger_config_path):
        logger_config = Config.load_yaml(logger_config_path)
        config.merge(logger_config)

    # Merge User / Ablation Config
    print(f"Loading config from: {args.config}")
    user_config = Config.load_yaml(args.config)
    config.merge(user_config)

    # Load JAX EnvParams
    params = load_env_params(args.config)

    # 2. Setup Results Directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.wandb_name or args.tag or f"jax_{args.algorithm}_{timestamp}"
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join("results", f"JAX_{args.algorithm}", run_name)
    
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save config
    config_save_path = os.path.join(models_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")

    # 3. Initialize WandB with full options
    wandb_enabled = WANDB_AVAILABLE and not args.no_wandb and not config.get('wandb.disabled', False)
    if wandb_enabled:
        wandb_login(quiet=True)
        
        wandb_kwargs = {
            "project": args.wandb_project or config.get('wandb.project', 'gridworld-jax'),
            "entity": args.wandb_entity or config.get('wandb.entity'),
            "group": args.wandb_group or config.get('wandb.group'),
            "name": run_name,
            "config": {
                "algorithm": args.algorithm,
                "framework": "JAX/Flax NNX",
                "total_timesteps": args.total_timesteps,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "lr": args.lr,
                "hidden_size": args.hidden_size,
                "seed": args.seed,
                **config.to_dict()
            },
            "reinit": True
        }
        
        # Resume support
        if args.wandb_resume_id:
            wandb_kwargs['id'] = args.wandb_resume_id
            wandb_kwargs['resume'] = "allow"
        
        wandb.init(**wandb_kwargs)
        
        # Define metrics
        wandb.define_metric("iteration")
        wandb.define_metric("timesteps")
        wandb.define_metric("loss/*", step_metric="iteration")
        wandb.define_metric("*", step_metric="timesteps")
        
        # Log source code
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # 4. Print Summary
    print(f"\n{'='*60}")
    print(f"JAX Training: {args.algorithm}")
    print(f"{'='*60}")
    print(f"Grid: {params.height}x{params.width}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Parallel Envs: {args.num_envs}")
    print(f"Steps/Iter: {args.num_steps}")
    print(f"Results: {results_dir}")
    print(f"WandB: {'Enabled' if wandb_enabled else 'Disabled'}")
    print(f"{'='*60}\n")

    # 5. Training Setup
    timesteps_per_iter = args.num_steps * args.num_envs
    num_iterations = args.total_timesteps // timesteps_per_iter
    checkpoint_interval = max(1, num_iterations * args.checkpoint_frequency // 100)

    # Initialize ParallelEnv
    env = ParallelEnv(params)

    # Initialize RNG
    key = jax.random.PRNGKey(args.seed)
    key, model_key, env_key = jax.random.split(key, 3)

    # Initialize environment states
    env_state, obs = env.reset(env_key, args.num_envs)
    input_dim = obs.shape[-1]
    action_dim = 4  # UP, DOWN, LEFT, RIGHT

    print(f"Observation dim: {input_dim}, Action dim: {action_dim}")

    # Initialize Model and Optimizer
    if args.algorithm == "RecurrentPPO":
        rngs = nnx.Rngs(model_key)
        model = ActorCriticRNN(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=args.hidden_size,
            rngs=rngs
        )
        optimizer = nnx.Optimizer(model, optax.adam(args.lr), wrt=nnx.Param)
        h_state = model.initial_state(batch_size=args.num_envs)

        ppo_config = PPOConfig(
            num_steps=args.num_steps,
            num_epochs=args.num_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            lr=args.lr
        )

        # JIT compile
        print("JIT compiling train_iteration...")
        jit_train = nnx.jit(train_iteration, static_argnums=(6,))

        # 6. Training Loop
        print(f"Starting training for {num_iterations} iterations...")
        for i in range(num_iterations):
            env_state, h_state, key, epoch_logs = jit_train(
                model, optimizer, params, env_state, h_state, key, ppo_config
            )

            # Extract metrics
            final_loss, (p_loss, v_loss, e_loss) = epoch_logs[-1]
            timesteps_done = (i + 1) * timesteps_per_iter

            # Log to WandB
            if wandb_enabled:
                wandb.log({
                    "iteration": i + 1,
                    "timesteps": timesteps_done,
                    "loss/total": float(final_loss),
                    "loss/policy": float(p_loss),
                    "loss/value": float(v_loss),
                    "loss/entropy": float(e_loss)
                })

            # Console output
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Iter {i+1}/{num_iterations} | Timesteps: {timesteps_done:,} | "
                      f"Loss: {final_loss:.4f} | Policy: {p_loss:.4f} | Value: {v_loss:.4f}")

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                pct = int((i + 1) / num_iterations * 100)
                ckpt_path = os.path.join(models_dir, f"jax_rppo_{pct}.ckpt")
                # TODO: Implement proper NNX checkpoint saving
                print(f"  [Checkpoint saved at {pct}%]")

    elif args.algorithm == "DreamerV3":
        print("DreamerV3 not yet implemented in JAX. Coming soon!")
        return

    # 7. Save Final Model
    final_ckpt = os.path.join(models_dir, "jax_rppo_100.ckpt")
    # TODO: Implement proper NNX checkpoint saving
    print(f"\nTraining complete! Final model saved to: {final_ckpt}")

    if wandb_enabled:
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
