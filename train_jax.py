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

GPU Memory Management:
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to prevent JAX from taking 90% VRAM per process.
- Use `CUDA_VISIBLE_DEVICES` to isolate processes on specific GPUs.
- Example: `XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_jax.py ...`

Usage:
    python train_jax.py --config configs/ablation/homeostatic/04_nociception.yaml --total-timesteps 100000
"""
import os
import argparse
import yaml
import numpy as np
from datetime import datetime
from collections import deque
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import NamedTuple

from src.environment.jax_env.config_loader import load_env_params
from src.environment.jax_env.wrapper import ParallelEnv
from src.environment.jax_env.sensor import get_observation
from src.environment.jax_env.core import jax_step
from src.models.jax_models.recurrent_ppo_network import ActorCriticRNN
from src.models.jax_models.recurrent_ppo_trainer import train_iteration
from src.utils.config import get_default_config, Config

# Orbax
import orbax.checkpoint as ocp

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
    parser.add_argument("--debug", action="store_true", help="Show verbose step-by-step progress logging")
    
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

    if args.debug:
        print(f"DEBUG: Loading JAX EnvParams from {args.config}...", flush=True)
    params = load_env_params(args.config)
    if args.debug:
        print("DEBUG: JAX EnvParams loaded.", flush=True)

    # 2. Setup Results Directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.wandb_name or args.tag or f"jax_{args.algorithm}_{timestamp}"
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join("results", f"JAX_{args.algorithm}", run_name)
    
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Orbax Setup (New API)
    checkpointer = ocp.CheckpointManager(
        os.path.abspath(models_dir),
        checkpointers=ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    )

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
    
    # 6. Algorithm Initialization
    if args.algorithm == "RecurrentPPO":
        # RecurrentPPO Setup
        
        # Init params
        key, init_key = jax.random.split(key)
        
        # Initialize NNX model state
        model = ActorCriticRNN(input_dim=input_dim, action_dim=action_dim, hidden_size=args.hidden_size, rngs=nnx.Rngs(init_key))
        
        # Optimizer
        optimizer = nnx.Optimizer(model, optax.adam(args.lr), wrt=nnx.Param)
        
        # PPO Config
        ppo_config = PPOConfig(
            num_steps=args.num_steps,
            num_epochs=args.num_epochs,
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_eps=config.get('clip_eps', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            vf_coef=config.get('vf_coef', 0.5),
            lr=args.lr
        )
        
        # Initialize Hidden State
        h_state = jnp.zeros((args.num_envs, args.hidden_size))

        # JIT compile
        print("JIT compiling train_iteration...")
        jit_train = nnx.jit(train_iteration, static_argnums=(6,))
        
    elif args.algorithm == "DreamerV3":
        from src.models.jax_models.dreamer_v3_trainer import DreamerTrainer, ReplayBuffer
        
        # Dreamer Config
        dreamer_config = {
            'model_lr': config.get('model_lr', 1e-4),
            'actor_lr': config.get('actor_lr', 3e-5),
            'value_lr': config.get('value_lr', 8e-5),
            'batch_size': config.get('batch_size', 16),
            'batch_length': config.get('batch_length', 16), # Horizon
        }
        
        if args.debug:
            print("DEBUG: Initializing DreamerTrainer...", flush=True)
        key, init_key = jax.random.split(key)
        trainer = DreamerTrainer(input_dim, action_dim, dreamer_config, rngs=nnx.Rngs(init_key))
        if args.debug:
            print("DEBUG: DreamerTrainer initialized.", flush=True)
        
        buffer = ReplayBuffer(
            capacity=int(1e5), 
            sequence_length=dreamer_config['batch_length'], 
            obs_dim=input_dim, 
            action_dim=action_dim
        )
        
        # Dreamer State
        # Prev State (RSSM) - Init for each env
        # We need to track this per env.
        # trainer.get_action handles initialization if None, but strict batch size?
        # trainer.get_action expects (B, O).
        dreamer_state = None # Will be init on first call
        
    if args.debug:
        print(f"DEBUG: Entering Training Loop ({args.num_steps} iterations)...", flush=True)
    # 7. Training Loop
    global_step = 0
    iteration = 0
    
    # Episode Metrics Tracking (Vectorized for Parallel Envs)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    completed_episodes = 0
    ep_info_buffer = deque(maxlen=100)
    
    start_time = datetime.now()
    
    try:
        while global_step < args.total_timesteps:
            iteration += 1
            
            if args.algorithm == "RecurrentPPO":
                # PPO Iteration
                env_state, h_state, key, losses = jit_train(
                    model, optimizer, params, env_state, h_state, key, ppo_config
                )
                
                # Update globals
                steps_this_iter = args.num_steps * args.num_envs
                global_step += steps_this_iter
                
                # Logging
                # PPO returns list of epoch losses. Take mean.
                avg_policy_loss = jnp.mean(jnp.array([l[1][0] for l in losses]))
                avg_value_loss = jnp.mean(jnp.array([l[1][1] for l in losses]))
                avg_ent_loss = jnp.mean(jnp.array([l[1][2] for l in losses]))
                total_loss = jnp.mean(jnp.array([l[0] for l in losses]))
                
                if wandb_enabled:
                    wandb.log({
                        "loss/total": total_loss,
                        "loss/policy": avg_policy_loss,
                        "loss/value": avg_value_loss,
                        "loss/entropy": avg_ent_loss,
                        "timesteps": global_step,
                        "iteration": iteration
                    })
                print(f"Iter {iteration} | Step {global_step} | Loss: {total_loss:.4f}", flush=True)
                    
            elif args.algorithm == "DreamerV3":
                # Dreamer Iteration (Step-based loop inside, or we do generic loop)
                # Dreamer typically alternates collection and training.
                # Train every K steps, or every step?
                # Let's collect 'num_steps' first, then train 'num_steps' times?
                # Or 1 step collect, 1 step train (standard Dreamer).
                
                # Since we have parallel envs, we collect 'num_envs' steps at once.
                # So we add 'num_envs' transitions.
                # Then we train 'num_envs' times? Ratio usually 1:1 or 1:0.x.
                # Let's train 1 batch per env step.
                
                # 1. Action Selection
                if args.debug:
                    print(f"DEBUG: Iter {iteration}/{args.num_steps} | Step {global_step} - Selecting Action...", end="\r", flush=True)
                obs_arr = jax.vmap(get_observation, in_axes=(0, None))(env_state, params) # (B, 33)
                
                # get_action uses JAX, we pass JAX array
                # Returns action_idx (B,) and updated dreamer_state
                key, act_key = jax.random.split(key)
                
                # We need to wrap get_action? It's inside nnx module.
                # We can call it directly.
                action_idx, dreamer_state = trainer.get_action(obs_arr, dreamer_state, eval_mode=False, rng=act_key)
                
                # Ensure action is integer for indexing
                action_idx = action_idx.astype(jnp.int32)
                
                # Convert to OneHot for Environment?
                # Environment expects ONEHOT action?
                # jax_env/core.py step takes 'action'
                # Check main_jax.py:
                #    action = jax.random.randint(key, (args.num_envs,), 0, 4)
                #    action_onehot = jax.nn.one_hot(action, 4)
                #    state, ... = jax.vmap(jax_step)(state, action_onehot, ...)
                # Yes, expects onehot.
                
                action_onehot = jax.nn.one_hot(action_idx, action_dim)
                # Convert to OneHot for Environment?
                # Environment expects ONEHOT action?
                # jax_env/core.py step takes 'action'
                
                # Check Debug print above
                
                # Vmap step
                if args.debug:
                    print(f"DEBUG: Iter {iteration}/{args.num_steps} | Step {global_step} - Stepping Env...", end="\r", flush=True)
                step_fn = jax.vmap(lambda s, a: jax_step(s, a, params))
                next_env_state, reward, done, info = step_fn(env_state, action_idx)
                
                # 3. Add to Buffer
                # We need to move data to CPU for numpy buffer
                # And handle 'is_first' (if done, next is first)
                
                # Current 'done' means THIS step was terminal.
                # Next step 'is_first' will be True.
                # We track is_first externally?
                # Or just use 'done' signal.
                # Dreamer: store (obs, act, reward, discount). 
                # If done, discount=0.
                
                obs_np = np.array(obs_arr) # Force sync?
                act_np = np.array(action_onehot)
                rew_np = np.array(reward)
                done_np = np.array(done)
                # is_first for THIS step.
                # If previous step was done, this step is first.
                # We need 'prev_dones'.
                if iteration == 1:
                     is_first_np = np.ones((args.num_envs,), dtype=bool)
                else:
                     # How to track? 'prev_dones'
                     pass 
                
                # Let's persistent var
                if not hasattr(main, 'prev_dones'):
                    main.prev_dones = np.zeros((args.num_envs,), dtype=bool) # assume not first except iter 1
                    if iteration == 1: main.prev_dones[:] = True
                
                is_first_np = main.prev_dones
                
                for i in range(args.num_envs):
                    buffer.add(obs_np[i], act_np[i], rew_np[i], done_np[i], is_first_np[i])
                    
                main.prev_dones = done_np
                
                # Handle auto-reset is done inside jax_step?
                # jax_step returns next_state RESETTED if done.
                # So next_env_state is valid start of new episode.
                
                env_state = next_env_state
                global_step += args.num_envs
                
                # --- Episode Metric Tracking ---
                # Update counters
                episode_returns += rew_np
                episode_lengths += 1
                
                # Check for completions
                dones = done_np.astype(bool)
                if np.any(dones):
                    for i in range(args.num_envs):
                        if dones[i]:
                            completed_episodes += 1
                            # Add to buffer
                            ep_info_buffer.append({'r': episode_returns[i], 'l': episode_lengths[i]})
                            
                            # Log aggregated stats if buffer is full enough or occasional
                            if len(ep_info_buffer) > 0 and completed_episodes % 5 == 0:
                                mean_rew = np.mean([ep['r'] for ep in ep_info_buffer])
                                mean_len = np.mean([ep['l'] for ep in ep_info_buffer])
                                
                                if wandb_enabled:
                                    wandb.log({
                                        "Episode/Reward_Mean": mean_rew,
                                        "Episode/Steps_Mean": mean_len,
                                        "Episode/Number": completed_episodes
                                    }, step=global_step)
                                
                                # Log to console occasionally
                                if completed_episodes % 10 == 0:
                                    print(f"Episode {completed_episodes} | Mean Reward: {mean_rew:.2f} | Mean Steps: {mean_len:.1f}")

                            # Reset
                            episode_returns[i] = 0.0
                            episode_lengths[i] = 0
                # -------------------------------
                
                # 4. Train Step
                metrics = {}
                loss_msg = ""
                # Train only if buffer has enough data
                if buffer.size > dreamer_config['batch_size'] * 2:
                    if args.debug:
                        print(f"DEBUG: Iter {iteration}/{args.num_steps} | Step {global_step} - Updating Models...", end="\r", flush=True)
                    metrics = trainer.train_step(batch_jax, key)
                    loss_msg = f"| Loss: {metrics.get('loss_model', 0):.2f}"
                
                # Log
                if wandb_enabled and iteration % 10 == 0:
                    wandb.log({
                        "timesteps": global_step,
                        "iteration": iteration,
                        **metrics
                    })
                
                if iteration % 1 == 0:
                     print(f"Iter {iteration}/{args.num_steps} | Step {global_step} {loss_msg}", flush=True)
                else:
                     print(f"Iter {iteration} | Step {global_step}", end="\r", flush=True)

            # Checkpoint
            checkpoint_interval = args.checkpoint_frequency
            if iteration % checkpoint_interval == 0:
                print(f"Saving checkpoint to {models_dir} at step {global_step}...")
                
                ckpt_data = {}
                if args.algorithm == "RecurrentPPO":
                    # Save PPO state
                    ckpt_data = {
                        'model': nnx.state(model, nnx.Param),
                        'optimizer': nnx.state(optimizer),
                        'h_state': h_state,
                        'key': key,
                        'iteration': iteration,
                        'step': global_step
                    }
                elif args.algorithm == "DreamerV3":
                    # Save Dreamer state
                    ckpt_data = {
                         'wm': nnx.state(trainer.agent.wm, nnx.Param),
                         'actor': nnx.state(trainer.agent.ac.actor, nnx.Param),
                         'critic': nnx.state(trainer.agent.ac.critic, nnx.Param),
                         'model_opt': nnx.state(trainer.model_opt),
                         'actor_opt': nnx.state(trainer.actor_opt),
                         'critic_opt': nnx.state(trainer.critic_opt),
                         'key': key,
                         'iteration': iteration,
                         'step': global_step
                    }
                
                checkpointer.save(iteration, args=ocp.args.StandardSave(ckpt_data))

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    print(f"Training complete. Results saved to {results_dir}")

    # 7. Save Final Model
    checkpointer.save(iteration, args=ocp.args.StandardSave(ckpt_data))
    print(f"\nTraining complete! Final model saved to: {models_dir}")

    if wandb_enabled:
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
