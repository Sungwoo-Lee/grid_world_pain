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
import argparse
import os
import time

# --- Pre-parse arguments for Device Selection ---
# To properly set JAX_PLATFORMS, we must do this BEFORE importing jax.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--device", type=str, default="gpu")
_args, _ = _pre_parser.parse_known_args()

# Disable JAX memory pre-allocation (crucial for shared GPU environments)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if _args.device.lower().startswith("cpu"):
    os.environ["JAX_PLATFORMS"] = "cpu"
elif ":" in _args.device:
    backend, index = _args.device.lower().split(":")
    if backend in ["gpu", "cuda"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        os.environ["JAX_PLATFORMS"] = "cuda"

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

from src.environment.config_loader import load_env_params
from src.environment.wrapper import ParallelEnv
from src.environment.sensor import get_observation, get_observation_breakdown
from src.environment.core import jax_step
from src.models.recurrent_ppo_network import ActorCriticRNN
from src.models.recurrent_ppo_trainer import train_iteration
from src.models.dqn_network import DQNNetwork, get_action_dqn_nnx
from src.models.dqn_trainer import ReplayBuffer as DQNReplayBuffer, update_step_dqn
from src.models.drqn_network import DRQNNetwork, get_action_drqn_nnx
from src.models.drqn_trainer import RecurrentReplayBuffer as DRQNReplayBuffer, update_step_drqn
from src.models.ppo_network import ActorCriticMLP, get_action_and_value_ppo_nnx
from src.models.ppo_trainer import train_iteration_ppo
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

    # --- Finalize Device Selection ---
    try:
        device_str = (args.device or _args.device).lower()
        if ":" in device_str or device_str in ["gpu", "cuda"]:
            # If we set CUDA_VISIBLE_DEVICES, JAX sees the target GPU as index 0
            jax.config.update("jax_default_device", jax.devices("cuda")[0])
            if not args.quiet:
                print(f"Device: gpu ({jax.devices('cuda')[0]})")
        elif device_str == "cpu":
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
            if not args.quiet:
                print(f"Device: cpu")
    except Exception as e:
        if not args.quiet:
            print(f"Warning: Device configuration for '{args.device}' failed ({e}). Using JAX default: {jax.devices()[0]}")

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

    # Merge Visualization Defaults
    vis_config_path = os.path.join(os.path.dirname(__file__), "configs", "visualization", "visualization.yaml")
    if os.path.exists(vis_config_path):
        if not args.quiet:
            print(f"Loading visualization config from {vis_config_path}")
        vis_defaults = Config.load_yaml(vis_config_path)
        config.merge(vis_defaults)

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
        hidden_size = args.hidden_size or config.get_mandatory('agent.rssm_deter_dim')
        lr = args.lr or config.get_mandatory('agent.actor_lr')
    else:
        num_steps = args.num_steps or 1
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        lr = args.lr or config.get_mandatory('agent.lr')

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
            "entity": args.wandb_entity or config.get_mandatory('wandb.entity'),
            "group": args.wandb_group or config.get_mandatory('wandb.group'),
            "name": args.wandb_name or tag,
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
        rnn_type = config.get_mandatory('agent.rnn_type')
        activation = config.get_mandatory('agent.activation')
        return_mode = config.get_mandatory('agent.return_mode')
        
        # Read neuromodulation config (None when modulation.type is null/absent)
        modulation_config = config.get('agent.modulation', None)
        if modulation_config is not None and modulation_config.get('type') is None:
            modulation_config = None

        if not args.quiet:
            print(f"RNN Type: {rnn_type}, Activation: {activation}, Return Mode: {return_mode}")
            if modulation_config is not None:
                print(f"Neuromodulation: ENABLED (type={modulation_config['type']}, "
                      f"mod_hidden={modulation_config['mod_hidden_size']}, "
                      f"grouping={modulation_config['grouping_size']})")
            else:
                print(f"Neuromodulation: DISABLED (baseline)")
        
        model = ActorCriticRNN(
            input_dim=input_dim, 
            action_dim=action_dim, 
            hidden_size=hidden_size, 
            rngs=nnx.Rngs(init_key),
            rnn_type=rnn_type,
            activation=activation,
            modulation_config=modulation_config
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
        
        # Initialize hidden state via model (handles modulator state automatically)
        h_state = model.initial_state(num_envs)

        if not args.quiet:
            print("JIT compiling train_iteration...")
        jit_train = nnx.jit(train_iteration, static_argnums=(6,))

        
    elif algorithm == "DreamerV3":
        from src.models.dreamer_v3_trainer import DreamerTrainer, ReplayBuffer
        
        # Pass the 'agent' section to the trainer
        dreamer_config = agent_config.get_mandatory('agent')
        if dreamer_config is None:
            # Fallback if the YAML doesn't have a top-level 'agent' key (already merged into config)
            dreamer_config = config.get_mandatory('agent')
        
        # Read neuromodulation config (None when modulation.type is null/absent)
        dreamer_mod_config = config.get('agent.modulation', None)
        if dreamer_mod_config is not None and dreamer_mod_config.get('type') is None:
            dreamer_mod_config = None
        
        key, init_key = jax.random.split(key)
        trainer = DreamerTrainer(input_dim, action_dim, dreamer_config, rngs=nnx.Rngs(init_key),
                                 modulation_config=dreamer_mod_config)


        buffer = ReplayBuffer(
            capacity=int(1e5), 
            sequence_length=dreamer_config['batch_length'], 
            obs_dim=input_dim, 
            action_dim=action_dim
        )
        dreamer_state = None 

    elif algorithm == "DQN":
        key, init_key = jax.random.split(key)
        fc_layers = config.get_mandatory('agent.fc_layers')
        model = DQNNetwork(input_dim, action_dim, fc_layers, rngs=nnx.Rngs(init_key))
        target_model = DQNNetwork(input_dim, action_dim, fc_layers, rngs=nnx.Rngs(init_key))
        # NNX state sync
        nnx.update(target_model, nnx.state(model))
        
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        
        buffer_capacity = config.get_mandatory('agent.buffer_size')
        dqn_buffer = DQNReplayBuffer(capacity=buffer_capacity, obs_dim=input_dim)
        
        epsilon = config.get_mandatory('agent.epsilon_start')
        epsilon_end = config.get_mandatory('agent.epsilon_end')
        epsilon_decay = config.get_mandatory('agent.epsilon_decay')
        target_update_freq = config.get_mandatory('agent.target_update_freq')
        batch_size = config.get_mandatory('agent.batch_size')
        
        # Reuse num_steps as "collection steps per iteration"
        num_steps = args.num_steps or 1 # Standard DQN explores 1 step per env per iter

    elif algorithm == "DRQN":
        key, init_key = jax.random.split(key)
        fc_layers = config.get_mandatory('agent.fc_layers')
        recurrent_layers = config.get_mandatory('agent.recurrent_layers')
        hidden_size = recurrent_layers[0] # NNX LSTM/GRU use single hidden size
        rnn_type = config.get_mandatory('agent.rnn_type')
        
        model = DRQNNetwork(input_dim, action_dim, hidden_size, rnn_type, fc_layers, rngs=nnx.Rngs(init_key))
        target_model = DRQNNetwork(input_dim, action_dim, hidden_size, rnn_type, fc_layers, rngs=nnx.Rngs(init_key))
        nnx.update(target_model, nnx.state(model))
        
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        
        buffer_capacity = config.get_mandatory('agent.buffer_size')
        drqn_buffer = DRQNReplayBuffer(capacity=buffer_capacity, obs_dim=input_dim)
        
        epsilon = config.get_mandatory('agent.epsilon_start')
        epsilon_end = config.get_mandatory('agent.epsilon_end')
        epsilon_decay = config.get_mandatory('agent.epsilon_decay')
        target_update_freq = config.get_mandatory('agent.target_update_freq')
        batch_size = config.get_mandatory('agent.batch_size')
        trace_length = config.get_mandatory('agent.trace_length')
        burn_in_length = config.get_mandatory('agent.burn_in_length')
        
        # Hidden state for collection
        h_state = model.initial_state(num_envs)
        
        # Reuse num_steps as "collection steps per iteration"
        num_steps = args.num_steps or 1

    elif algorithm == "PPO":
        key, init_key = jax.random.split(key)
        
        activation = config.get_mandatory('agent.activation')
        return_mode = config.get_mandatory('agent.return_mode')
        actor_fc_layers = config.get_mandatory('agent.actor_fc_layers')
        critic_fc_layers = config.get_mandatory('agent.critic_fc_layers')
        
        # We can support dual LR by choosing one or using a complex optimizer
        # For now, let's use lr_actor as primary
        lr_actor = config.get_mandatory('agent.lr_actor')
        
        if not args.quiet:
            print(f"Activation: {activation}, Return Mode: {return_mode}")
            print(f"Actor Layers: {actor_fc_layers}, Critic Layers: {critic_fc_layers}")
        
        model = ActorCriticMLP(
            input_dim=input_dim, 
            action_dim=action_dim, 
            actor_fc_layers=actor_fc_layers,
            critic_fc_layers=critic_fc_layers,
            rngs=nnx.Rngs(init_key),
            activation=activation
        )
        optimizer = nnx.Optimizer(model, optax.adam(lr_actor), wrt=nnx.Param)
        
        ppo_config = PPOConfig(
            num_steps=num_steps,
            num_epochs=config.get_mandatory('agent.K_epochs'),
            gamma=config.get_mandatory('agent.gamma'),
            gae_lambda=config.get_mandatory('agent.gae_lambda'),
            clip_eps=config.get_mandatory('agent.eps_clip'),
            ent_coef=config.get_mandatory('agent.entropy_coef'),
            vf_coef=config.get_mandatory('agent.vf_coef'),
            lr=lr_actor,
            activation=activation,
            return_mode=return_mode
        )
        
        if not args.quiet:
            print("JIT compiling train_iteration_ppo...")
        jit_train = nnx.jit(train_iteration_ppo, static_argnums=(5,))
        
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
                    env_state, h_state, key, losses, num_completed, trajectories = jit_train(
                        model, optimizer, params, env_state, h_state, key, ppo_config
                    )
                    
                    rollout_rew = trajectories.reward
                    rollout_done = trajectories.done
                    mod_info = trajectories.mod_info
                    if args.debug: print(f" Done.", flush=True)
                    
                    if args.debug and mod_info is not None:
                        print(f"  [Modulator] Mean Percept: {float(jnp.mean(mod_info.z_percept)):.3f}, Mean Memory: {float(jnp.mean(mod_info.z_memory)):.3f}, Temp: {float(jnp.mean(mod_info.temperature)):.2f}")
                    
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
                    avg_grad_norm = jnp.mean(jnp.array([l[1][3] for l in losses]))
                    avg_mod_grad_norm = jnp.mean(jnp.array([l[1][4] for l in losses]))
                    total_loss = jnp.mean(jnp.array([l[0] for l in losses]))
                    
                    if wandb_enabled:
                        wandb_logs = {
                            "loss/total": total_loss,
                            "loss/policy": avg_policy_loss,
                            "loss/value": avg_value_loss,
                            "loss/entropy": avg_ent_loss,
                            "loss/grad_norm": avg_grad_norm,
                        }

                        # Add Modulator metrics if enabled
                        if mod_info is not None:
                            wandb_logs.update({
                                "modulator/grad_norm": float(avg_mod_grad_norm),
                                "modulator/gamma_mean": float(jnp.mean(mod_info.z_percept)),
                                "modulator/gamma_std": float(jnp.std(mod_info.z_percept)),
                                "modulator/z_memory_mean": float(jnp.mean(mod_info.z_memory)),
                                "modulator/z_memory_std": float(jnp.std(mod_info.z_memory)),
                                "modulator/temperature_mean": float(jnp.mean(mod_info.temperature)),
                                "modulator/temperature_min": float(jnp.min(mod_info.temperature)),
                                "modulator/temperature_max": float(jnp.max(mod_info.temperature)),
                            })
                            if modulation_config is not None and modulation_config.get('type') == "PreActivation":
                                wandb_logs.update({
                                    "modulator/beta_mean": float(jnp.mean(mod_info.z_percept_add)),
                                    "modulator/beta_std": float(jnp.std(mod_info.z_percept_add)),
                                })
                        
                        wandb_logs.update({
                            "timesteps": global_step,
                            "iteration": iteration
                        })
                        wandb.log(wandb_logs)
                    
                    postfix = {
                        "Iter": iteration,
                        "Loss": f"{total_loss:.4f}",
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    }
                    if mod_info is not None:
                        postfix["T"] = f"{float(jnp.mean(mod_info.temperature)):.2f}"
                    pbar.set_postfix(postfix)
                        
                elif algorithm == "DreamerV3":
                    # Use JITTED collect_sequence for massive speedup (approx 600x)
                    key, collect_key = jax.random.split(key)
                    env_state, dreamer_state, key, transitions = trainer.collect_sequence(
                        env_state, params, num_steps, collect_key, dreamer_state)
                    
                    # Convert transitions to NumPy and add to buffer
                    
                    # Batch transfer to host (one transfer instead of multiple)
                    transitions_np = jax.device_get(transitions)
                    
                    # Reshape for add_batch (T, B, ...) -> (T*B, ...)
                    num_items = num_steps * num_envs
                    
                    obs_flat = transitions_np['obs'].reshape(num_items, -1)
                    act_flat = transitions_np['action'].reshape(num_items, -1)
                    rew_flat = transitions_np['reward'].reshape(num_items)
                    done_flat = transitions_np['terminal'].reshape(num_items)
                    is_first_flat = transitions_np['is_first'].astype(bool).reshape(num_items)
                    
                    buffer.add_batch(obs_flat, act_flat, rew_flat, done_flat, is_first_flat)
                    
                    # Update statistics (Vectorized where possible)
                    rew_steps = transitions_np['reward'] # (T, B)
                    done_steps = transitions_np['terminal'] # (T, B)
                    
                    # More vectorized stats handling
                    done_indices = np.where(done_steps) # (t_idxs, env_idxs)
                    
                    if done_indices[0].size > 0:
                        # Track episode returns/lengths
                        for i in np.unique(done_indices[1]):
                            d_idxs = done_indices[0][done_indices[1] == i]
                            curr_start = 0
                            for d_idx in d_idxs:
                                ep_reward = float(episode_returns[i] + np.sum(rew_steps[curr_start:d_idx+1, i]))
                                ep_length = int(episode_lengths[i] + (d_idx + 1 - curr_start))
                                ep_info_buffer.append({'r': ep_reward, 'l': ep_length})
                                iteration_episodes.append({'r': ep_reward, 'l': ep_length})
                                total_episodes_completed += 1
                                episode_returns[i] = 0
                                episode_lengths[i] = 0
                                curr_start = d_idx + 1
                            
                            # Add leftover
                            if curr_start < num_steps:
                                episode_returns[i] += np.sum(rew_steps[curr_start:, i])
                                episode_lengths[i] += (num_steps - curr_start)
                        
                        # Environments with NO dones in this batch
                        no_done_mask = np.ones(num_envs, dtype=bool)
                        no_done_mask[done_indices[1]] = False
                        episode_returns[no_done_mask] += np.sum(rew_steps[:, no_done_mask], axis=0)
                        episode_lengths[no_done_mask] += num_steps
                    else:
                        # No episodes finished at all
                        episode_returns += np.sum(rew_steps, axis=0)
                        episode_lengths += num_steps
                    
                    t_buffer = time.time() - t1
                    
                    global_step += num_envs * num_steps

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
                        train_steps = dreamer_config.get('train_steps', 1)
                        for _ in range(train_steps):
                            batch_jax = buffer.sample(dreamer_config['batch_size'])
                            key, train_key = jax.random.split(key)
                            metrics = trainer.train_step(batch_jax, train_key)
                        loss_msg = f"L: {metrics.get('loss_model', 0):.2f}"
                    
                    if wandb_enabled and iteration % 10 == 0:
                        wandb_logs = {"timesteps": global_step, "iteration": iteration}
                        for mk, mv in metrics.items():
                            if mk.startswith('loss_actor') or mk.startswith('loss_critic') or \
                               mk.startswith('mean_') or mk.startswith('entropy'):
                                wandb_logs[f"Behavior/{mk}"] = float(mv)
                            elif mk.startswith('loss_model') or mk.startswith('loss_recon') or \
                                 mk.startswith('loss_kl') or mk.startswith('loss_rew') or \
                                 mk.startswith('loss_cont') or mk.startswith('loss_dyn') or \
                                 mk.startswith('loss_rep'):
                                wandb_logs[f"WorldModel/{mk}"] = float(mv)
                            elif mk.startswith('mod_'):
                                wandb_logs[f"Modulator/{mk}"] = float(mv)
                            else:
                                wandb_logs[mk] = float(mv)
                        wandb.log(wandb_logs)
                    
                    postfix = {
                        "Iter": iteration,
                        "Loss": loss_msg,
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}",
                    }
                    if dreamer_mod_config is not None and metrics and 'mod_z_reward_mean' in metrics:
                        postfix["R_mod"] = f"{float(metrics['mod_z_reward_mean']):.2f}"
                    pbar.set_postfix(postfix)
                    if args.debug: print(f" Done.", flush=True)

                elif algorithm == "DQN":
                    if args.debug: print(f"  [DEBUG] DQN Step Collection...", end="", flush=True)
                    
                    # 1. Collect Step
                    key, act_key = jax.random.split(key)
                    # vmapped action selection
                    action = jax.vmap(get_action_dqn_nnx, in_axes=(None, 0, 0, None))(
                        model, obs, jax.random.split(act_key, num_envs), epsilon
                    )
                    
                    from src.environment.core import jax_step
                    step_fn = jax.vmap(lambda s, a: jax_step(s, a, params))
                    next_env_state, reward, done, info = step_fn(env_state, action)
                    
                    next_obs = jax.vmap(get_observation, in_axes=(0, None))(next_env_state, params)
                    
                    # 2. Add to Buffer
                    dqn_buffer.add(
                        obs=np.array(obs),
                        action=np.array(action),
                        reward=np.array(reward),
                        next_obs=np.array(next_obs),
                        done=np.array(done)
                    )
                    
                    # 3. Auto-Reset and state transition
                    from src.environment.core import jax_reset
                    reset_key, key = jax.random.split(key)
                    reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(params, jax.random.split(reset_key, num_envs))
                    
                    def select_done(d, r, n):
                        d_expanded = d.reshape((d.shape[0],) + (1,) * (r.ndim - 1))
                        return jnp.where(d_expanded, r, n)
                        
                    env_state = jax.tree_util.tree_map(
                        lambda r, n: select_done(done, r, n),
                        reset_state, next_env_state
                    )
                    obs = jax.vmap(get_observation, in_axes=(0, None))(env_state, params)
                    
                    global_step += num_envs
                    iteration_episodes = []
                    
                    # Stats tracking
                    episode_returns += np.array(reward)
                    episode_lengths += 1
                    dones_np = np.array(done).astype(bool)
                    if np.any(dones_np):
                        completed_indices = np.where(dones_np)[0]
                        for i in completed_indices:
                            total_episodes_completed += 1
                            ep_reward = float(episode_returns[i])
                            ep_length = int(episode_lengths[i])
                            ep_info_buffer.append({'r': ep_reward, 'l': ep_length})
                            iteration_episodes.append({'r': ep_reward, 'l': ep_length})
                            episode_returns[i] = 0.0
                            episode_lengths[i] = 0
                            
                    # 4. Update Step
                    loss_val = 0.0
                    if dqn_buffer.size > batch_size:
                        key, sample_key = jax.random.split(key)
                        batch = dqn_buffer.sample(batch_size, sample_key)
                        loss_val = update_step_dqn(model, target_model, optimizer, batch, config.get_mandatory('agent.gamma'))
                        
                        # Epsilon Decay
                        epsilon = max(epsilon_end, epsilon * epsilon_decay)
                        
                        # Target Sync
                        if iteration % target_update_freq == 0:
                            nnx.update(target_model, nnx.state(model))
                    
                    if wandb_enabled:
                        logs = {
                            "iteration": iteration,
                            "timesteps": global_step,
                            "train/epsilon": float(epsilon),
                            "loss/dqn": float(loss_val)
                        }
                        if iteration_episodes:
                            rewards_list = [ep['r'] for ep in iteration_episodes]
                            logs.update({
                                "Episode/Reward": np.mean(rewards_list),
                                "Episode/Number": total_episodes_completed
                            })
                        wandb.log(logs)

                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.set_postfix({
                        "Iter": iteration,
                        "Loss": f"{float(loss_val):.4f}",
                        "Eps": f"{float(epsilon):.2f}",
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    })
                    pbar.refresh()

                elif algorithm == "DRQN":
                    if args.debug: print(f"  [DEBUG] DRQN Step Collection...", end="", flush=True)
                    
                    # 1. Collect Step
                    key, act_key = jax.random.split(key)
                    # vmapped action selection
                    # h_state is a PyTree (B, H) or ((B, H), (B, H))
                    # we vmap over first dimension (which is B)
                    if rnn_type.upper() == "LSTM":
                        action, h_state_new = jax.vmap(get_action_drqn_nnx, in_axes=(None, 0, (0, 0), 0, None))(
                            model, obs, h_state, jax.random.split(act_key, num_envs), epsilon
                        )
                    else:
                        action, h_state_new = jax.vmap(get_action_drqn_nnx, in_axes=(None, 0, 0, 0, None))(
                            model, obs, h_state, jax.random.split(act_key, num_envs), epsilon
                        )
                    
                    from src.environment.core import jax_step
                    step_fn = jax.vmap(lambda s, a: jax_step(s, a, params))
                    next_env_state, reward, done, info = step_fn(env_state, action)
                    
                    # 2. Add to Buffer
                    drqn_buffer.add(
                        obs=np.array(obs),
                        action=np.array(action),
                        reward=np.array(reward),
                        done=np.array(done)
                    )
                    
                    # 3. Auto-Reset and state transition
                    from src.environment.core import jax_reset
                    reset_key, key = jax.random.split(key)
                    reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(params, jax.random.split(reset_key, num_envs))
                    
                    def select_done(d, r, n):
                        d_expanded = d.reshape((d.shape[0],) + (1,) * (r.ndim - 1))
                        return jnp.where(d_expanded, r, n)
                        
                    env_state = jax.tree_util.tree_map(
                        lambda r, n: select_done(done, r, n),
                        reset_state, next_env_state
                    )
                    obs = jax.vmap(get_observation, in_axes=(0, None))(env_state, params)
                    
                    # Reset hidden state for completed envs
                    if rnn_type.upper() == "LSTM":
                        h_state = (
                            jnp.where(done[:, None], 0.0, h_state_new[0]),
                            jnp.where(done[:, None], 0.0, h_state_new[1])
                        )
                    else:
                        h_state = jnp.where(done[:, None], 0.0, h_state_new)
                    
                    global_step += num_envs
                    iteration_episodes = []
                    
                    # Stats tracking
                    episode_returns += np.array(reward)
                    episode_lengths += 1
                    dones_np = np.array(done).astype(bool)
                    if np.any(dones_np):
                        completed_indices = np.where(dones_np)[0]
                        for i in completed_indices:
                            total_episodes_completed += 1
                            ep_reward = float(episode_returns[i])
                            ep_length = int(episode_lengths[i])
                            ep_info_buffer.append({'r': ep_reward, 'l': ep_length})
                            iteration_episodes.append({'r': ep_reward, 'l': ep_length})
                            episode_returns[i] = 0.0
                            episode_lengths[i] = 0
                            
                    # 4. Update Step
                    loss_val = 0.0
                    if drqn_buffer.size > (trace_length + burn_in_length + batch_size):
                        key, sample_key = jax.random.split(key)
                        obs_seq, actions, rewards, next_obs_seq, dones = drqn_buffer.sample_sequences(
                            batch_size, trace_length + burn_in_length, sample_key
                        )
                        loss_val = update_step_drqn(
                            model, target_model, optimizer, 
                            obs_seq, actions, rewards, next_obs_seq, dones, 
                            config.get_mandatory('agent.gamma'), 
                            burn_in_length
                        )
                        
                        # Epsilon Decay
                        epsilon = max(epsilon_end, epsilon * epsilon_decay)
                        
                        # Target Sync
                        if iteration % target_update_freq == 0:
                            nnx.update(target_model, nnx.state(model))
                    
                    if wandb_enabled:
                        logs = {
                            "iteration": iteration,
                            "timesteps": global_step,
                            "train/epsilon": float(epsilon),
                            "loss/drqn": float(loss_val)
                        }
                        if iteration_episodes:
                            rewards_list = [ep['r'] for ep in iteration_episodes]
                            logs.update({
                                "Episode/Reward": np.mean(rewards_list),
                                "Episode/Number": total_episodes_completed
                            })
                        wandb.log(logs)

                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.set_postfix({
                        "Iter": iteration,
                        "Loss": f"{float(loss_val):.4f}",
                        "Eps": f"{float(epsilon):.2f}",
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    })
                    pbar.refresh()

                elif algorithm == "PPO":
                    if args.debug: print(f"  [DEBUG] Collecting {num_steps * num_envs} steps of experience...", end="", flush=True)
                    env_state, key, losses, num_completed, rollout_rew, rollout_done = jit_train(
                        model, optimizer, params, env_state, key, ppo_config
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
                                ep_info_buffer.append({'r': ep_reward, 'l': ep_length})
                                iteration_episodes.append({'r': ep_reward, 'l': ep_length})
                                episode_returns[i] = 0.0
                                episode_lengths[i] = 0
                    
                    if wandb_enabled:
                        logs = {
                            "iteration": iteration,
                            "timesteps": global_step,
                        }
                        if iteration_episodes:
                            rewards_list = [ep['r'] for ep in iteration_episodes]
                            logs.update({
                                "Episode/Reward": np.mean(rewards_list),
                                "Episode/Number": total_episodes_completed
                            })
                        if losses:
                            # losses is a list of (total_loss, (p_loss, v_loss, e_loss))
                            avg_total = np.mean([l[0] for l in losses])
                            logs.update({"loss/ppo_total": float(avg_total)})
                        wandb.log(logs)

                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    loss_msg = f"L: {float(losses[-1][0]):.2f}" if losses else ""
                    pbar.set_postfix({
                        "Iter": iteration,
                        "Loss": loss_msg,
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    })
                    pbar.refresh()

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
                        pbar.write(f"[CHECKPOINT] Saving model at episode {total_episodes_completed} (Iteration {iteration})...")
                        checkpointer.save(total_episodes_completed, args=ocp.args.StandardSave(ckpt_data))
                        checkpointer.wait_until_finished()  # Ensure sync for stability
                        
                        # Trigger evaluation after checkpoint
                        vis_flag = config.get_mandatory('visualization.enabled')
                        eval_v_flag = config.get_mandatory('training.video_during_training')

                        if vis_flag or eval_v_flag:
                            if args.debug:
                                print(f"  [DEBUG] Starting evaluation and video saving...", flush=True)
                            try:
                                from src.utils.evaluation_core import evaluate_jax_checkpoint
                                eval_results = evaluate_jax_checkpoint(
                                    model=model if algorithm == "RecurrentPPO" else trainer.agent,
                                    params=params, config=config, num_episodes=config.get_mandatory('testing.evaluation_episodes'), seed=seed,
                                    results_dir=results_dir, checkpoint_pct=total_episodes_completed,
                                    render_video=True, wandb_enabled=wandb_enabled, debug=args.debug,
                                    quiet=True
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
