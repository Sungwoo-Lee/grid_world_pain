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
    rnn_type: str = "LSTM"
    activation: str = "tanh"
    return_mode: str = "MC"

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
    
    if args.debug:
        print(f"[DEBUG] Script started. CLI arguments: {args}", flush=True)

    # 1. Configuration Loading
    if args.debug: print(f"[DEBUG] Phase 1: Configuration Loading...", flush=True)
    
    # Matching train.py logic (Strictly no safe defaults)
    base_config_path = args.config or DEFAULT_CONFIG_PATH
    if args.debug: print(f"[DEBUG] Loading base config from {base_config_path}", flush=True)
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
    agent_config_path = args.agent_config
    if args.debug: print(f"[DEBUG] Loading agent config from {agent_config_path}", flush=True)
    if not args.quiet:
        print(f"Loading agent config from: {agent_config_path}")
    agent_config = Config.load_yaml(agent_config_path)
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

    # 1.5 Print Combined Configuration (Always)
    if not args.quiet:
        def print_pretty_config(config_dict):
            """Stylized configuration audit log."""
            width = 64
            print("\n" + "=" * width)
            print(" SYSTEM CONFIGURATION AUDIT ".center(width, "="))
            print("=" * width)
            
            def print_recursive(d, indent=0):
                keys = sorted(d.keys())
                for k in keys:
                    v = d[k]
                    prefix = "  " * indent
                    if isinstance(v, dict):
                        print(f"\n{prefix}[ {k.upper()} ]")
                        print_recursive(v, indent + 1)
                    elif isinstance(v, list):
                        if len(v) > 0 and isinstance(v[0], dict):
                            # List of dicts (like predators/resources)
                            names = [item.get('name', 'unnamed') for item in v]
                            print(f"{prefix}● {k: <20} : {len(v)} items ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})")
                        elif len(v) > 5:
                            print(f"{prefix}● {k: <20} : [List of {len(v)} items]")
                        else:
                            print(f"{prefix}● {k: <20} : {v}")
                    else:
                        print(f"{prefix}● {k: <20} : {v}")
            
            # Separate dicts from non-dicts at top level for grouping
            top_dicts = {k: v for k, v in config_dict.items() if isinstance(v, dict)}
            top_leafs = {k: v for k, v in config_dict.items() if not isinstance(v, dict)}
            
            if top_leafs:
                print("\n[ GLOBAL SETTINGS ]")
                print_recursive(top_leafs, indent=1)
            
            print_recursive(top_dicts, indent=0)
            print("\n" + "=" * width + "\n")

        print_pretty_config(config.to_dict())

    # Determine Algorithm
    algorithm = config.get_mandatory('agent.algorithm')
    
    # Strictly Resolve Parameters (No Safe Defaults)
    episodes = args.episodes or config.get_mandatory('episodes')
    env_max_steps = config.get_mandatory('environment.max_steps')
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')
    
    # Budget scales with parallelization: episodes * steps per episode * num environments
    total_timesteps = args.total_timesteps or (episodes * env_max_steps * num_envs)
    
    if algorithm == "RecurrentPPO":
        num_steps = args.num_steps or config.get_mandatory('agent.sequence_length')
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        lr = args.lr or config.get_mandatory('agent.lr_actor')
    elif algorithm == "DreamerV3":
        # Dreamer doesn't use a single sequence_length for rollout collection (usually 1 step)
        num_steps = args.num_steps or 1 
        # Dreamer has many hidden sizes; using rssm_deter_dim as a proxy for summary/logging
        hidden_size = args.hidden_size or config.get('agent.rssm_deter_dim', 512)
        lr = args.lr or config.get_mandatory('agent.actor_lr')
    else:
        num_steps = args.num_steps or 1
        hidden_size = args.hidden_size or 128
        lr = args.lr or 1e-4

    seed = args.seed if args.seed is not None else config.get_mandatory('seed')

    
    # Re-load EnvParams with full merged config for JAX core
    params = load_env_params(config)
    
    # Apply CLI overrides to params if they exist in params (redundant now but safe)
    if args.no_satiation: params = params.replace(with_satiation=False)
    if args.no_overeating_death: params = params.replace(overeating_death=False)

    # 2. Setup Results Directory
    if args.debug: print(f"[DEBUG] Phase 2: Results Directory Setup...", flush=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or config.get('tag', algorithm)
    run_name = f"{timestamp}_{tag}"
    
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

    # 3. Initialize WandB
    if args.debug: print(f"[DEBUG] Phase 3: WandB Initialization...", flush=True)
    wandb_enabled = WANDB_AVAILABLE and not args.no_wandb and not config.get_mandatory('wandb.disabled')
    if wandb_enabled:
        wandb_login(quiet=True)
        
        wandb_kwargs = {
            "project": args.wandb_project or config.get_mandatory('wandb.project'),
            "entity": args.wandb_entity or config.get('wandb.entity'),
            "group": args.wandb_group or config.get('wandb.group'),
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
        
        if args.wandb_resume_id:
            wandb_kwargs['id'] = args.wandb_resume_id
            wandb_kwargs['resume'] = "allow"
        
        wandb.init(**wandb_kwargs)
        
        wandb.define_metric("iteration")
        wandb.define_metric("timesteps")
        wandb.define_metric("Episode/Number")
        wandb.define_metric("Episode/*", step_metric="Episode/Number")
        wandb.define_metric("loss/*", step_metric="iteration")
        wandb.define_metric("*", step_metric="timesteps")
        
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # 4. Print Summary
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

    # 5. Training Setup
    if args.debug: print(f"[DEBUG] Phase 5: Training Setup...", flush=True)
    env = ParallelEnv(params)
    key = jax.random.PRNGKey(seed)
    key, model_key, env_key = jax.random.split(key, 3)

    if args.debug: print(f"[DEBUG] Performing initial environment reset for {num_envs} envs...", flush=True)
    env_state, obs = env.reset(env_key, num_envs)
    input_dim = obs.shape[-1]
    
    rest_enabled = params.rest_action_enabled
    eat_enabled = params.eat_action_enabled
    action_dim = 4 + int(rest_enabled) + int(eat_enabled)

    if not args.quiet:
        print("\n--- RL API Specifications ---")
        print(f"Action Dim: {action_dim}")
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
    if args.debug: print(f"[DEBUG] Phase 6: Algorithm Initialization ({algorithm})...", flush=True)
    if algorithm == "RecurrentPPO":
        key, init_key = jax.random.split(key)
        
        # Read parity options from config
        rnn_type = config.get('agent.rnn_type', 'LSTM')
        activation = config.get('agent.activation', 'tanh')
        return_mode = config.get('agent.return_mode', 'MC')
        
        if not args.quiet:
            print(f"RNN Type: {rnn_type}, Activation: {activation}, Return Mode: {return_mode}")
        
        model = ActorCriticRNN(
            input_dim=input_dim, 
            action_dim=action_dim, 
            hidden_size=hidden_size, 
            rngs=nnx.Rngs(init_key),
            rnn_type=rnn_type,
            activation=activation
        )
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        
        ppo_config = PPOConfig(
            num_steps=num_steps,
            num_epochs=config.get_mandatory('agent.K_epochs'),
            gamma=config.get_mandatory('agent.gamma'),
            gae_lambda=config.get_mandatory('agent.gae_lambda'),
            clip_eps=config.get_mandatory('agent.eps_clip'),
            ent_coef=config.get_mandatory('agent.entropy_coef'),
            vf_coef=config.get_mandatory('agent.vf_coef'),
            lr=lr,
            rnn_type=rnn_type,
            activation=activation,
            return_mode=return_mode
        )
        
        # Initialize hidden state (LSTM uses tuple, GRU uses array)
        if rnn_type.upper() == "LSTM":
            h_state = (jnp.zeros((num_envs, hidden_size)), jnp.zeros((num_envs, hidden_size)))
        else:
            h_state = jnp.zeros((num_envs, hidden_size))

        if not args.quiet:
            print("JIT compiling train_iteration...")
        jit_train = nnx.jit(train_iteration, static_argnums=(6,))

        
    elif algorithm == "DreamerV3":
        from src.models.jax_models.dreamer_v3_trainer import DreamerTrainer, ReplayBuffer
        
        # Pass the 'agent' section to the trainer
        dreamer_config = agent_config.get('agent')
        if dreamer_config is None:
            # Fallback if the YAML doesn't have a top-level 'agent' key (already merged into config)
            dreamer_config = config.get('agent')
        
        key, init_key = jax.random.split(key)
        trainer = DreamerTrainer(input_dim, action_dim, dreamer_config, rngs=nnx.Rngs(init_key))


        buffer = ReplayBuffer(
            capacity=int(1e5), 
            sequence_length=dreamer_config['batch_length'], 
            obs_dim=input_dim, 
            action_dim=action_dim
        )
        dreamer_state = None 
        
    # 7. Training Loop
    if args.debug: print(f"[DEBUG] Phase 7: Entering Training Loop...", flush=True)
    global_step = 0
    iteration = 0
    total_episodes_completed = 0
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_info_buffer = deque(maxlen=100)
    
    start_time = datetime.now()
    if args.debug: print(f"[DEBUG] Loop start time: {start_time.strftime('%H:%M:%S')}", flush=True)

    with tqdm(total=episodes, disable=args.quiet, desc="Training") as pbar:

        try:
            while (total_episodes_completed < episodes) if episodes > 0 else (global_step < total_timesteps):

                iteration += 1
                if args.debug: print(f"\n[DEBUG] --- Iteration {iteration} Start (Step: {global_step}) ---", flush=True)

                # Buffer for episodes that finish DURING THIS ITERATION
                iteration_episodes = []
                
                if algorithm == "RecurrentPPO":
                    if args.debug: print(f"  [DEBUG] Collecting {num_steps * num_envs} steps of experience...", end="", flush=True)
                    env_state, h_state, key, losses, num_completed, rollout_rew, rollout_done = jit_train(
                        model, optimizer, params, env_state, h_state, key, ppo_config
                    )
                    if args.debug: print(f" Done.", flush=True)
                    
                    steps_this_iter = num_steps * num_envs
                    global_step += steps_this_iter
                    
                    rew_np = np.array(rollout_rew)
                    done_np = np.array(rollout_done)
                    
                    for t in range(num_steps):
                        episode_returns += rew_np[t]
                        episode_lengths += 1
                        dones_t = done_np[t].astype(bool)
                        
                        if np.any(dones_t):
                            completed_indices = np.where(dones_t)[0]
                            for i in completed_indices:
                                total_episodes_completed += 1
                                ep_reward = float(episode_returns[i])
                                ep_length = int(episode_lengths[i])
                                
                                # Store for moving average (tqdm)
                                ep_info_buffer.append({'r': ep_reward, 'l': ep_length})
                                # Store for iteration-level logging (Stage 3)
                                iteration_episodes.append({'r': ep_reward, 'l': ep_length})
                                
                                # Reset for next episode in this slot
                                episode_returns[i] = 0.0
                                episode_lengths[i] = 0
                    
                    # Log AGGREGATED stats for the iteration (Stage 3)
                    if wandb_enabled and iteration_episodes:
                        rewards = [ep['r'] for ep in iteration_episodes]
                        lengths = [ep['l'] for ep in iteration_episodes]
                        wandb.log({
                            "Episode/Reward": np.mean(rewards),
                            "Episode/Reward_Min": np.min(rewards),
                            "Episode/Reward_Max": np.max(rewards),
                            "Episode/Steps": np.mean(lengths),
                            "Episode/Number": total_episodes_completed,
                            "timesteps": global_step,
                            "iteration": iteration
                        })

                    # Update progress bar based on total episodes completed
                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.refresh()
                    
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
                    # Collect a batch of steps to match PPO's iteration rhythm
                    for _ in range(num_steps):
                        if args.debug: print(f".", end="", flush=True)
                        obs_arr = jax.vmap(get_observation, in_axes=(0, None))(env_state, params)
                        key, act_key = jax.random.split(key)
                        action_idx, dreamer_state = trainer.get_action(obs_arr, dreamer_state, eval_mode=False, rng=act_key)
                        action_idx = action_idx.astype(jnp.int32)
                        action_onehot = jax.nn.one_hot(action_idx, action_dim)
                        
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
                        
                        episode_returns += rew_np
                        episode_lengths += 1
                        dones = done_np.astype(bool)
                        if np.any(dones):
                            completed_indices = np.where(dones)[0]
                            for i in completed_indices:
                                total_episodes_completed += 1
                                ep_reward = float(episode_returns[i])
                                ep_length = int(episode_lengths[i])
                                
                                ep_info_buffer.append({'r': ep_reward, 'l': ep_length})
                                iteration_episodes.append({'r': ep_reward, 'l': ep_length})
                                
                                episode_returns[i] = 0.0
                                episode_lengths[i] = 0

                    if wandb_enabled and iteration_episodes:
                        rewards = [ep['r'] for ep in iteration_episodes]
                        lengths = [ep['l'] for ep in iteration_episodes]
                        wandb.log({
                            "Episode/Reward": np.mean(rewards),
                            "Episode/Reward_Min": np.min(rewards),
                            "Episode/Reward_Max": np.max(rewards),
                            "Episode/Steps": np.mean(lengths),
                            "Episode/Number": total_episodes_completed,
                            "timesteps": global_step,
                            "iteration": iteration
                        })

                    # Update progress bar
                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.refresh()
    
                    metrics = {}
                    loss_msg = ""
                    if buffer.size > max(dreamer_config['batch_size'] * 2, dreamer_config['batch_length']):
                        if args.debug: print(f"  [DEBUG] DreamerV3 Training Update...", end="", flush=True)
                        batch_jax = buffer.sample(dreamer_config['batch_size'])
                        metrics = trainer.train_step(batch_jax, key)
                        loss_msg = f"L: {metrics.get('loss_model', 0):.2f}"
                    
                    if wandb_enabled and iteration % 10 == 0:
                        wandb.log({"timesteps": global_step, "iteration": iteration, **metrics})
                    
                    pbar.set_postfix({"Iter": iteration, "Loss": loss_msg, "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"})
                    if args.debug: print(f" Done.", flush=True)

                # Checkpoint Logic
                checkpoint_freq = args.checkpoint_frequency or config.get_mandatory('training.checkpoint_frequency')
                
                # Check if we've crossed an episode boundary for checkpointing
                # We save if the current episode count has reached the next checkpoint milestone
                if not hasattr(main, 'last_checkpoint_save'):
                    main.last_checkpoint_save = 0
                
                should_checkpoint = (total_episodes_completed >= main.last_checkpoint_save + checkpoint_freq)
                
                if should_checkpoint:
                    main.last_checkpoint_save = (total_episodes_completed // checkpoint_freq) * checkpoint_freq
                    
                    ckpt_data = {}
                    if algorithm == "RecurrentPPO":
                        ckpt_data = {
                            'model': nnx.state(model, nnx.Param), 
                            'optimizer': nnx.state(optimizer), 
                            'h_state': h_state, 
                            'key': key, 
                            'iteration': iteration, 
                            'step': global_step,
                            'episode': total_episodes_completed
                        }
                    elif algorithm == "DreamerV3":
                        ckpt_data = {
                            'wm': nnx.state(trainer.agent.wm, nnx.Param), 
                            'actor': nnx.state(trainer.agent.ac.actor, nnx.Param), 
                            'critic': nnx.state(trainer.agent.ac.critic, nnx.Param), 
                            'key': key, 
                            'iteration': iteration, 
                            'step': global_step,
                            'episode': total_episodes_completed
                        }

                    if ckpt_data:
                        print(f"\n[CHECKPOINT] Saving model at episode {total_episodes_completed} (Iteration {iteration})...")
                        checkpointer.save(total_episodes_completed, args=ocp.args.StandardSave(ckpt_data))
                        
                        # Trigger evaluation after checkpoint
                        vis_flag = config.get('visualization.enabled')
                    eval_v_flag = config.get('training.video_during_training')
                    if vis_flag or eval_v_flag:
                        if args.debug:
                            print(f"  [DEBUG] Starting evaluation and video saving...", flush=True)
                        try:
                            from src.utils.evaluation_jax_core import evaluate_jax_checkpoint
                            eval_results = evaluate_jax_checkpoint(
                                model=model if algorithm == "RecurrentPPO" else trainer.agent,
                                params=params, config=config, num_episodes=3, seed=seed,
                                results_dir=results_dir, checkpoint_pct=iteration,
                                render_video=True, wandb_enabled=wandb_enabled, debug=args.debug
                            )
                            if wandb_enabled:
                                wandb.log({"Eval/MeanReward": eval_results["mean_reward"], "Eval/MeanLength": eval_results["mean_length"], "iteration": iteration, "timesteps": global_step})
                        except Exception as e:
                            print(f"Warning: Evaluation failed: {e}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            
    # Final cleanup
    print(f"Training complete. Results saved to {results_dir}")
    if wandb_enabled:
        wandb.finish()
    checkpointer.close()

if __name__ == "__main__":
    main()
