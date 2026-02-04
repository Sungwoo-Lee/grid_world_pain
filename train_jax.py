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
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import NamedTuple

from src.environment.jax_env.config_loader import load_env_params
from src.environment.jax_env.wrapper import ParallelEnv
from src.environment.jax_env.sensor import get_observation, get_observation_breakdown
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

# Defaults
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "environment", "environment.yaml")

def main():
    parser = argparse.ArgumentParser(description="Train JAX RL Agents")
    
    # Matching train.py flags
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to base config YAML (Environment/Ablation)")
    parser.add_argument("--agent_config", type=str, required=True, help="Path to agent config YAML (Required)")
    parser.add_argument("--tag", type=str, help="Tag for the training run")
    parser.add_argument("--device", type=str, help="Device to use (JAX handles this; kept for parity)")
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    parser.add_argument("--no-overeating-death", action="store_true", help="Disable death by overeating")
    parser.add_argument("--wandb-project", type=str, help="WandB Project Name")
    parser.add_argument("--wandb-group", type=str, help="WandB Group Name")
    parser.add_argument("--wandb-job-type", type=str, help="WandB Job Type")
    parser.add_argument("--wandb-name", type=str, help="WandB Run Name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress output and progress bar")
    parser.add_argument("--debug", action="store_true", help="Show verbose step-by-step progress logging")
    parser.add_argument("--checkpoint-frequency", type=int, help="Save checkpoint every N episodes/evals")
    parser.add_argument("--load-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-resume-id", type=str, help="WandB Run ID to resume logging")

    # JAX-specific but useful flags
    parser.add_argument("--total-timesteps", type=int, help="Total training timesteps (overrides episodes if provided)")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, help="Steps per iteration (rollout length)")
    parser.add_argument("--hidden-size", type=int, help="Hidden layer size")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides agent config if provided)")
    parser.add_argument("--results-dir", type=str, help="Custom results directory")
    parser.add_argument("--wandb-entity", type=str, help="WandB Entity Name")
    
    args = parser.parse_args()

    # 1. Load Config Hierarchy (Synchronized with train.py)
    config = get_default_config()

    # Merge Training Defaults
    train_config_path = os.path.join(os.path.dirname(__file__), "configs", "train", "default.yaml")
    if os.path.exists(train_config_path):
        if not args.quiet:
            print(f"Loading train config from {train_config_path}")
        train_defaults = Config.load_yaml(train_config_path)
        config.merge(train_defaults)

    # Merge Evaluation Defaults
    eval_config_path = os.path.join(os.path.dirname(__file__), "configs", "evaluation", "default.yaml")
    if os.path.exists(eval_config_path):
        if not args.quiet:
            print(f"Loading eval config from {eval_config_path}")
        eval_defaults = Config.load_yaml(eval_config_path)
        config.merge(eval_defaults)

    # Merge Logger Config (WandB settings)
    logger_config_path = os.path.join(os.path.dirname(__file__), "configs", "logger", "wandb.yaml")
    if os.path.exists(logger_config_path):
        if not args.quiet:
            print(f"Loading logger config from {logger_config_path}")
        logger_config = Config.load_yaml(logger_config_path)
        config.merge(logger_config)

    # Merge Base/User/Ablation Config (--config)
    if args.config:
        if not args.quiet:
            print(f"Loading override config from: {args.config}")
        user_config = Config.load_yaml(args.config)
        config.merge(user_config)

    # Merge Agent Config (--agent_config) - REQUIRED
    if not args.quiet:
        print(f"Loading agent config from: {args.agent_config}")
    agent_config = Config.load_yaml(args.agent_config)
    config.merge(agent_config)

    # CLI Overrides (Synchronized with train.py)
    if args.wandb_project: config.set('wandb.project', args.wandb_project)
    if args.wandb_group: config.set('wandb.group', args.wandb_group)
    if args.wandb_job_type: config.set('wandb.job_type', args.wandb_job_type)
    if args.wandb_name: config.set('wandb.name', args.wandb_name)
    if args.no_wandb: config.set('wandb.disabled', True)
    if args.tag: config.set('tag', args.tag)
    if args.seed is not None: config.set('seed', args.seed)
    if args.no_satiation: config.set('environment.with_satiation', False)
    if args.no_overeating_death: config.set('environment.overeating_death', False)

    # Determine Algorithm
    algorithm = config.get_mandatory('agent.algorithm')
    
    # Strictly Resolve Parameters (No Safe Defaults)
    episodes = args.episodes or config.get_mandatory('episodes')
    env_max_steps = config.get_mandatory('environment.max_steps')
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')
    
    # Budget scales with parallelization: episodes * steps per episode * num environments
    total_timesteps = args.total_timesteps or (episodes * env_max_steps * num_envs)
    
    num_steps = args.num_steps or config.get_mandatory('agent.num_steps')
    hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
    seed = args.seed if args.seed is not None else config.get_mandatory('seed')
    lr = args.lr or config.get_mandatory('agent.lr_actor') # or model_lr for dreamer... will handle below
    
    # Re-load EnvParams with full merged config for JAX core
    params = load_env_params(args.config or DEFAULT_CONFIG_PATH)
    
    # Apply CLI overrides to params if they exist in params
    if args.no_satiation: params = params.replace(with_satiation=False)
    if args.no_overeating_death: params = params.replace(overeating_death=False)

    # 2. Setup Results Directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.wandb_name or args.tag or f"jax_{algorithm}_{timestamp}"
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join("results", f"JAX_{algorithm}", run_name)
    
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
    if not args.quiet:
        print(f"Config saved to: {config_save_path}")

    # 3. Initialize WandB with full options
    wandb_enabled = WANDB_AVAILABLE and not args.no_wandb and not config.get_mandatory('wandb.disabled')
    if wandb_enabled:
        wandb_login(quiet=True)
        
        wandb_kwargs = {
            "project": args.wandb_project or config.get_mandatory('wandb.project'),
            "entity": args.wandb_entity or config.get('wandb.entity'), # entity can be None
            "group": args.wandb_group or config.get('wandb.group'), # group can be None
            "name": run_name,
            "config": {
                "algorithm": algorithm,
                "framework": "JAX/Flax NNX",
                "total_timesteps": total_timesteps,
                "num_envs": num_envs,
                "num_steps": num_steps,
                "lr": lr,
                "hidden_size": hidden_size,
                "seed": seed,
                **config.to_dict()
            },
            "reinit": True
        }
        
        # Resume support
        if args.wandb_resume_id:
            wandb_kwargs['id'] = args.wandb_resume_id
            wandb_kwargs['resume'] = "allow"
        
        wandb.init(**wandb_kwargs)
        
        # Define metrics (matching train.py)
        wandb.define_metric("iteration")
        wandb.define_metric("timesteps")
        wandb.define_metric("Episode/Number")
        wandb.define_metric("Episode/*", step_metric="Episode/Number")
        wandb.define_metric("loss/*", step_metric="iteration")
        wandb.define_metric("*", step_metric="timesteps")
        
        # Log source code
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # 3. Print Summary (matching train.py format)
    if not args.quiet:
        width = 60
        header = " JAX/FLAX RL CONFIGURATION "
        print("\n" + "=" * width)
        print(header.center(width, "="))
        print("=" * width)
        
        def print_section(title, data):
            print(f"\n[{title}]")
            for k, value in data.items():
                print(f"  \u25cf {k:.<25} {value}")
        
        # Environment Section
        with_satiation = config.get_mandatory('body.with_satiation')
        with_injury = config.get_mandatory('body.with_injury')
        use_homeostatic_reward = config.get_mandatory('body.use_homeostatic_reward')
        
        env_data = {
            "Grid Size": f"{params.height}x{params.width}",
            "Max Steps": env_max_steps,
            "Mode": "Interoceptive (Homeostasis)" if with_satiation else "Conventional (Goal-driven)",
        }
        if with_satiation:
            env_data["Reward Logic"] = "Homeostatic (Drive Reduction)" if use_homeostatic_reward else "Survival Step"
        if with_injury:
            env_data["Injury System"] = "ENABLED"
        print_section("Environment", env_data)
        
        # Training Section
        train_data = {
            "Framework": "JAX/Flax NNX",
            "Total Timesteps": f"{total_timesteps:,}",
            "Parallel Envs": num_envs,
            "Steps/Iteration": num_steps,
            "Seed": seed,
            "Results": results_dir,
            "WandB": "Enabled" if wandb_enabled else "Disabled"
        }
        print_section("Training", train_data)
        
        # Agent Section (populated after model init)

    # 5. Training Setup
    # Initialize ParallelEnv
    env = ParallelEnv(params)

    # Initialize RNG
    key = jax.random.PRNGKey(seed)
    key, model_key, env_key = jax.random.split(key, 3)

    # Initialize environment states
    env_state, obs = env.reset(env_key, num_envs)
    input_dim = obs.shape[-1]
    action_dim = 4  # UP, DOWN, LEFT, RIGHT

    # Print Observation Specs (matching train.py)
    if not args.quiet:
        print("\n--- RL API Specifications ---")
        print(f"Action Dim: {action_dim}")
        
        # Get detailed dimension breakdown
        obs_breakdown = get_observation_breakdown(params)
        breakdown_str = ", ".join([f"{k}={v}" for k, v in obs_breakdown.items()])
        total_dim = sum(obs_breakdown.values())
        
        print(f"Observation Dim: {total_dim} ({breakdown_str})")
        print(f"Dimension Breakdown:")
        for sensor_name, dim in obs_breakdown.items():
            print(f"  {sensor_name:.<20} {dim}")
        print(f"Hidden Size: {hidden_size}")
        print(f"Learning Rate: {lr}")
        print("--------------------------------------------\n")
    
    # 6. Algorithm Initialization
    if algorithm == "RecurrentPPO":
        # RecurrentPPO Setup
        
        # Init params
        key, init_key = jax.random.split(key)
        
        # Initialize NNX model state
        model = ActorCriticRNN(input_dim=input_dim, action_dim=action_dim, hidden_size=hidden_size, rngs=nnx.Rngs(init_key))
        
        # Optimizer
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        
        # PPO Config (prioritize agent.* from model config)
        ppo_config = PPOConfig(
            num_steps=num_steps,
            num_epochs=config.get_mandatory('agent.K_epochs'),
            gamma=config.get_mandatory('agent.gamma'),
            gae_lambda=config.get_mandatory('agent.gae_lambda'),
            clip_eps=config.get_mandatory('agent.eps_clip'),
            ent_coef=config.get_mandatory('agent.entropy_coef'),
            vf_coef=config.get_mandatory('agent.vf_coef'),
            lr=lr
        )
        
        # Initialize Hidden State
        h_state = jnp.zeros((num_envs, hidden_size))

        # JIT compile
        if not args.quiet:
            print("JIT compiling train_iteration...")
        jit_train = nnx.jit(train_iteration, static_argnums=(6,))
        
    elif algorithm == "DreamerV3":
        from src.models.jax_models.dreamer_v3_trainer import DreamerTrainer, ReplayBuffer
        
        # Dreamer Config (Strict)
        dreamer_config = {
            'model_lr': config.get_mandatory('agent.model_lr'),
            'actor_lr': config.get_mandatory('agent.actor_lr'),
            'value_lr': config.get_mandatory('agent.value_lr'),
            'batch_size': config.get_mandatory('agent.batch_size'),
            'batch_length': config.get_mandatory('agent.batch_length'),
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
    
    # Episode Metrics Tracking (Per-Environment Episode Depth)
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    
    # Per-environment episode counters (how many episodes each env has completed)
    env_episode_counts = np.zeros(num_envs, dtype=np.int32)
    
    # Buffer for completed episodes at each depth: {episode_depth: {'rewards': [...], 'steps': [...]}}
    episode_depth_buffer = {}
    
    # Track the current "synchronized episode" (minimum across all envs, logged to WandB)
    logged_episode_depth = 0
    
    # Legacy buffer for running mean (used in progress bar postfix)
    ep_info_buffer = deque(maxlen=100)
    
    start_time = datetime.now()
    
    # Initialize Progress Bar (Episode-based as requested)
    pbar = tqdm(total=episodes, disable=args.quiet, desc="Training")
    
    try:
        while global_step < total_timesteps:
            iteration += 1
            
            if algorithm == "RecurrentPPO":
                # PPO Iteration
                env_state, h_state, key, losses, num_completed, rollout_rew, rollout_done = jit_train(
                    model, optimizer, params, env_state, h_state, key, ppo_config
                )
                
                # Update globals
                steps_this_iter = num_steps * num_envs
                global_step += steps_this_iter
                
                # Update progress bar with fractional "average episode" progress
                pbar.update(steps_this_iter / (num_envs * env_max_steps))
                
                # --- Process Rollout Metrics (T, B) ---
                rew_np = np.array(rollout_rew)
                done_np = np.array(rollout_done)
                
                for t in range(num_steps):
                    episode_returns += rew_np[t]
                    episode_lengths += 1
                    
                    dones_t = done_np[t].astype(bool)
                    if np.any(dones_t):
                        for i in range(num_envs):
                            if dones_t[i]:
                                # Increment this environment's episode count
                                env_episode_counts[i] += 1
                                current_depth = int(env_episode_counts[i])
                                
                                # Store in buffer for this depth
                                if current_depth not in episode_depth_buffer:
                                    episode_depth_buffer[current_depth] = {'rewards': [], 'steps': []}
                                episode_depth_buffer[current_depth]['rewards'].append(episode_returns[i])
                                episode_depth_buffer[current_depth]['steps'].append(episode_lengths[i])
                                
                                # Also add to legacy buffer for progress bar
                                ep_info_buffer.append({'r': episode_returns[i], 'l': episode_lengths[i]})
                                
                                # Reset per-env buffers
                                episode_returns[i] = 0.0
                                episode_lengths[i] = 0
                                
                                # Check if all environments have reached a new synchronized depth
                                min_depth = int(np.min(env_episode_counts))
                                while logged_episode_depth < min_depth:
                                    logged_episode_depth += 1
                                    depth = logged_episode_depth
                                    
                                    # Calculate averaged metrics for this depth
                                    if depth in episode_depth_buffer:
                                        avg_reward = np.mean(episode_depth_buffer[depth]['rewards'])
                                        avg_steps = np.mean(episode_depth_buffer[depth]['steps'])
                                        
                                        # Log to WandB with episode depth as X-axis
                                        if wandb_enabled:
                                            wandb.log({
                                                "Episode/Reward": float(avg_reward),
                                                "Episode/Steps": float(avg_steps),
                                                "Episode/Number": depth,
                                                "timesteps": global_step
                                            })
                                        
                                        # Optional: Clear buffer for this depth to save memory
                                        del episode_depth_buffer[depth]

                
                # Update PPO Loss Metrics
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
                
                pbar.set_postfix({
                    "Iter": iteration,
                    "Loss": f"{total_loss:.4f}",
                    "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                })
                    
            elif algorithm == "DreamerV3":
                # Action Selection
                if args.debug:
                    pbar.set_description(f"Iter {iteration} | Selecting Action")
                obs_arr = jax.vmap(get_observation, in_axes=(0, None))(env_state, params)
                
                key, act_key = jax.random.split(key)
                action_idx, dreamer_state = trainer.get_action(obs_arr, dreamer_state, eval_mode=False, rng=act_key)
                action_idx = action_idx.astype(jnp.int32)
                
                action_onehot = jax.nn.one_hot(action_idx, action_dim)
                
                # Vmap step
                if args.debug:
                    pbar.set_description(f"Iter {iteration} | Stepping Env")
                step_fn = jax.vmap(lambda s, a: jax_step(s, a, params))
                next_env_state, reward, done, info = step_fn(env_state, action_idx)
                
                obs_np = np.array(obs_arr)
                act_np = np.array(action_onehot)
                rew_np = np.array(reward)
                done_np = np.array(done)
                
                if not hasattr(main, 'prev_dones'):
                    main.prev_dones = np.zeros((num_envs,), dtype=bool)
                    if iteration == 1: main.prev_dones[:] = True
                
                is_first_np = main.prev_dones
                for i in range(num_envs):
                    buffer.add(obs_np[i], act_np[i], rew_np[i], done_np[i], is_first_np[i])
                
                main.prev_dones = done_np
                env_state = next_env_state
                global_step += num_envs
                
                # Update progress bar with fractional "average episode" progress
                pbar.update(num_envs / (num_envs * env_max_steps))
                
                # Episode Metric Tracking
                episode_returns += rew_np
                episode_lengths += 1
                
                # Check for completions
                dones = done_np.astype(bool)
                if np.any(dones):
                    for i in range(num_envs):
                        if dones[i]:
                            # Increment this environment's episode count
                            env_episode_counts[i] += 1
                            current_depth = int(env_episode_counts[i])
                            
                            # Store in buffer for this depth
                            if current_depth not in episode_depth_buffer:
                                episode_depth_buffer[current_depth] = {'rewards': [], 'steps': []}
                            episode_depth_buffer[current_depth]['rewards'].append(episode_returns[i])
                            episode_depth_buffer[current_depth]['steps'].append(episode_lengths[i])
                            
                            # Also add to legacy buffer for progress bar
                            ep_info_buffer.append({'r': episode_returns[i], 'l': episode_lengths[i]})
                            
                            # Reset per-env buffers
                            episode_returns[i] = 0.0
                            episode_lengths[i] = 0
                            
                            # Check if all environments have reached a new synchronized depth
                            min_depth = int(np.min(env_episode_counts))
                            while logged_episode_depth < min_depth:
                                logged_episode_depth += 1
                                depth = logged_episode_depth
                                
                                # Calculate averaged metrics for this depth
                                if depth in episode_depth_buffer:
                                    avg_reward = np.mean(episode_depth_buffer[depth]['rewards'])
                                    avg_steps = np.mean(episode_depth_buffer[depth]['steps'])
                                    
                                    # Log to WandB with episode depth as X-axis
                                    if wandb_enabled:
                                        wandb.log({
                                            "Episode/Reward": float(avg_reward),
                                            "Episode/Steps": float(avg_steps),
                                            "Episode/Number": depth,
                                            "timesteps": global_step
                                        })
                                    
                                    # Clear buffer for this depth to save memory
                                    del episode_depth_buffer[depth]
                
                # Train Step
                metrics = {}
                loss_msg = ""
                if buffer.size > dreamer_config['batch_size'] * 2:
                    if args.debug:
                        pbar.set_description(f"Iter {iteration} | Updating Models")
                    metrics = trainer.train_step(batch_jax, key)
                    loss_msg = f"L: {metrics.get('loss_model', 0):.2f}"
                
                if wandb_enabled and iteration % 10 == 0:
                    wandb.log({
                        "timesteps": global_step,
                        "iteration": iteration,
                        **metrics
                    })
                
                pbar.set_postfix({
                    "Iter": iteration,
                    "Loss": loss_msg,
                    "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                })

            # Checkpoint
            checkpoint_interval = args.checkpoint_frequency or config.get_mandatory('training.checkpoint_frequency')
            
            # Populate ckpt_data for potential saving
            if algorithm == "RecurrentPPO":
                ckpt_data = {
                    'model': nnx.state(model, nnx.Param),
                    'optimizer': nnx.state(optimizer),
                    'h_state': h_state,
                    'key': key,
                    'iteration': iteration,
                    'step': global_step
                }
            elif algorithm == "DreamerV3" and 'trainer' in locals():
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
            else:
                ckpt_data = {}

            if iteration % checkpoint_interval == 0 and ckpt_data:
                print(f"Saving checkpoint to {models_dir} at step {global_step}...")
                checkpointer.save(iteration, args=ocp.args.StandardSave(ckpt_data))

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    print(f"Training complete. Results saved to {results_dir}")

    # 7. Save Final Model
    checkpointer.save(iteration, args=ocp.args.StandardSave(ckpt_data))
    checkpointer.wait_until_finished()
    print(f"\nTraining complete! Final model saved to: {models_dir}")

    if wandb_enabled:
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")
    
    # Clean up Orbax
    checkpointer.close()

if __name__ == "__main__":
    main()
