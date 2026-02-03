"""
Training script for the GridWorld Reinforcement Learning agent.

This script:
1. Initializes the `GridWorld` environment (Conventional or Interoceptive) and `InteroceptiveBody`.
2. Creates an RL Agent (Tabular Q-Learning, DQN, PPO, DRQN, RecurrentPPO, DreamerV3).
3. Trains the agent for a specified number of episodes.
4. Periodically saves checkpoints (models) and visualizations.
5. Supports Continual Learning (resuming from checkpoints).

Arguments:
- `--episodes <int>`: Total number of training episodes.
- `--seed <int>`: Random seed for reproducibility.
- `--agent_config <path>`: (Required) Path to agent-specific config (e.g., `configs/models/ppo.yaml`).
- `--config <path>`: Path to base config YAML (overrides defaults).
- `--tag <str>`: Tag for the training run directory and WandB run name.
- `--device <str>`: Device to use (e.g., `cpu`, `cuda`, `cuda:0`, `auto`). Default: `auto`.
- `--no-satiation`: Disable satiation (conventional mode).
- `--no-overeating-death`: Disable death by overeating.
- `--wandb-project <str>`: WandB project name.
- `--wandb-group <str>`: WandB group name.
- `--wandb-job-type <str>`: WandB job type.
- `--wandb-name <str>`: Explicit WandB run name.
- `--no-wandb`: Disable WandB logging.
- `--quiet`: Suppress output and progress bar.
- `--debug`: Enable granular logging and debug info.
- `--checkpoint-frequency <int>`: Frequency of saving checkpoints.
- `--load-checkpoint <path>`: Path to checkpoint to resume training from.
- `--wandb-resume-id <str>`: WandB Run ID to resume logging.

Usage Examples:

1. **Train DQN**:
   ```bash
   python train.py --agent_config configs/models/dqn.yaml --episodes 1000 --tag my_dqn_run
   ```

2. **Train PPO with WandB**:
   ```bash
   python train.py --agent_config configs/models/ppo.yaml --episodes 5000 --wandb-project grid_world_pain
   ```

3. **Resume Training**:
   ```bash
   python train.py --agent_config configs/models/dqn.yaml --load-checkpoint results/DQN/RunName/models/dqn_model_500.ckpt --episodes 500 --wandb-resume-id <run_id>
   ```
"""
import matplotlib
matplotlib.use('Agg')
from src.environment import GridWorld
from src.environment.body import InteroceptiveBody
from src.models.q_learning import QLearningAgent
from src.models.dqn import DQNAgent
from src.models.drqn import DRQNAgent
from src.models.ppo import PPOAgent
from src.models.recurrent_ppo import RecurrentPPOAgent
from src.models.dreamer_v3 import DreamerV3Agent
from src.environment.sensor import SensorySystem
from src.utils.visualization import plot_q_table, plot_learning_curves
from src.utils.config import get_default_config
from src.utils.evaluation_core import evaluate_agent
from src.utils.state_utils import FrameStacker, preprocess_state
from src.utils.wandb_utils import wandb_login

import time
import numpy as np
import os
import re
import yaml
import torch
import random
import csv
from datetime import datetime
import wandb

import sys
import argparse


from tqdm import tqdm

def print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps, random_start_satiation, use_homeostatic_reward, satiation_setpoint, testing_seed,
                         with_injury, device="auto"):
    """
    Prints a professional and fancy configuration summary.
    """
    width = 60
    header = " GRIDWORLD RL CONFIGURATION "
    
    print("\n" + "=" * width)
    print(header.center(width, "="))
    print("=" * width)
    
    def print_section(title, data):
        print(f"\n[{title}]")
        for key, value in data.items():
            print(f"  \u25cf {key:.<25} {value}")

    # Environment
    resources = config_dict.get_mandatory('environment.resources')
    resource_names = [f"{r.get('count', 1)}x {r['name']} ({r['type']})" for r in resources]
    
    env_data = {
        "Grid Size": f"{config_dict.get_mandatory('environment.height', int)}x{config_dict.get_mandatory('environment.width', int)}",
        "Resources": ", ".join(resource_names),
        "Max Steps": max_steps,
        "Mode": "Interoceptive (Homeostasis)" if with_satiation else "Conventional (Goal-driven)"
    }
    if with_satiation:
        env_data["Reward Logic"] = "Homeostatic (Drive Reduction)" if use_homeostatic_reward else "Survival Step"
        if use_homeostatic_reward:
            env_data["Satiation Setpoint"] = satiation_setpoint
            
    if with_injury:
        env_data["Injury System"] = "ENABLED"
        
    print_section("Environment", env_data)
        
    print_section("Environment", env_data)

    # Body (if applicable)
    if with_satiation:
        body_data = {
            "Max Satiation": config_dict.get_mandatory('body.max_satiation'),
            "Start Satiation": config_dict.get_mandatory('body.start_satiation'),
            "Random Start Sat": "ENABLED" if random_start_satiation else "DISABLED",
            "Overeating Death": "ENABLED" if overeating_death else "DISABLED"
        }
        if with_injury:
             body_data["Max Injury"] = config_dict.get_mandatory('body.max_injury')
             body_data["Injury Recovery"] = config_dict.get_mandatory('body.injury_recovery')
             
        print_section("Body (Internal States)", body_data)

    # Agent
    # Agent
    using_sensory = config_dict.get_mandatory('sensory.using_sensory')
    algorithm = config_dict.get_mandatory('agent.algorithm')
    
    if algorithm == "DQN":
        agent_data = {
            "Algorithm": "Deep Q-Network (DQN)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Batch Size": 64
        }
        if using_sensory:
             agent_data["Sensor Radius"] = config_dict.get_mandatory('sensory.sensor_radius')
             agent_data["Decay Power"] = config_dict.get('sensory.decay_power', 1.0)
             
    elif algorithm == "DRQN":
        agent_data = {
            "Algorithm": "Deep Recurrent Q-Network (DRQN)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Batch Size": config_dict.get_mandatory('agent.batch_size'),
            "Trace Length": config_dict.get_mandatory('agent.trace_length')
        }
        if using_sensory:
             agent_data["Sensor Radius"] = config_dict.get_mandatory('sensory.sensor_radius')
             agent_data["Decay Power"] = config_dict.get('sensory.decay_power', 1.0)

    elif algorithm == "PPO":
        agent_data = {
            "Algorithm": "Proximal Policy Optimization (PPO)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Actor LR": config_dict.get_mandatory('agent.lr_actor'),
            "Critic LR": config_dict.get_mandatory('agent.lr_critic'),
            "Gamma (Discount)": config_dict.get_mandatory('agent.gamma'),
            "Update Frequency": config_dict.get_mandatory('agent.update_timestep')
        }
        if using_sensory:
             agent_data["Sensor Radius"] = config_dict.get_mandatory('sensory.sensor_radius')
             agent_data["Decay Power"] = config_dict.get('sensory.decay_power', 1.0)

    elif algorithm == "DreamerV3":
        agent_data = {
            "Algorithm": "Dreamer V3",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Batch Size": config_dict.get_mandatory('agent.batch_size'),
            "Batch Length": config_dict.get_mandatory('agent.batch_length'),
        }
        if using_sensory:
             agent_data["Sensor Radius"] = config_dict.get_mandatory('sensory.sensor_radius')
             agent_data["Decay Power"] = config_dict.get('sensory.decay_power', 1.0)

    elif algorithm == "RecurrentPPO":
        agent_data = {
            "Algorithm": "Recurrent PPO (LSTM)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Sequence Length": config_dict.get_mandatory('agent.sequence_length'),
            "Update Timestep": config_dict.get_mandatory('agent.update_timestep')
        }
        if using_sensory:
             agent_data["Sensor Radius"] = config_dict.get_mandatory('sensory.sensor_radius')
             agent_data["Decay Power"] = config_dict.get('sensory.decay_power', 1.0)

    else:
        agent_data = {
            "Algorithm": "Tabular Q-Learning",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Alpha (Learning Rate)": config_dict.get_mandatory('agent.alpha'),
            "Gamma (Discount)": config_dict.get_mandatory('agent.gamma'),
            "Min Epsilon": 0.05
        }
    print_section("RL Agent", agent_data)

    # Training
    train_data = {
        "Total Episodes": episodes,
        "Training Seed": seed,
        "Testing Seed": testing_seed,
        "Device": device
    }
    print_section("Training Schedule", train_data)

    print("\n" + "=" * width + "\n")

def train_agent(episodes=None, seed=None, with_satiation=None, overeating_death=None, food_satiation_gain=None, max_steps=None, random_start_satiation=None, use_homeostatic_reward=None, satiation_setpoint=None, death_penalty=None, testing_seed=None, config_dict=None,
                with_injury=None, max_injury=None, start_injury=None, injury_recovery=None, random_start_injury=None,
                prob_switch_to_danger=None, min_danger_duration=None, damage_amount=None,
                prob_switch_to_food=None, min_food_duration=None, device="auto", checkpoint_frequency=None, quiet=False, debug=False,
                start_episode=0, load_checkpoint_path=None):
    """
    Trains the RL Agent (Tabular Q-Learning, DQN, or PPO).
    """
    import os 
    
    # Initialize WandB
    # WandB Initialization
    if config_dict and not config_dict.get_mandatory('wandb.disabled'):
        wandb_enabled = True
    if not config_dict.get('wandb.disabled'):
        wandb_kwargs = {
            "project": config_dict.get('wandb.project'),
            "entity": config_dict.get('wandb.entity'), # Might be None
            "group": config_dict.get('wandb.group'),
            "job_type": config_dict.get('wandb.job_type'),
            "name": config_dict.get('wandb.name'),
            "config": config_dict.to_dict(),
            "reinit": True
        }
        
        # Override name with tag if present (User Request)
        if config_dict.get('tag'):
             wandb_kwargs['name'] = config_dict.get('tag')

        # Check if we should resume
        if config_dict.get('wandb.resume_id'):
            wandb_kwargs['id'] = config_dict.get('wandb.resume_id')
            wandb_kwargs['resume'] = "allow"
            
        # Login to WandB using shared key if available
        wandb_login(quiet=quiet)
        wandb.init(**wandb_kwargs)

        
        # Log Source Code
        # Explicitly log key files and src directory
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        
        # Define x-axis for different metrics
        # Episode metrics use Episode/Number as x-axis
        wandb.define_metric("Episode/*", step_metric="Episode/Number")
        # Step metrics (losses) use global_step as x-axis
        wandb.define_metric("*", step_metric="global_step") 
        wandb.define_metric("global_step", step_metric="global_step")
    
    # Extract Sensory Config
    # Strict retrieval for using_sensory?
    # User requested no safe defaults.
    using_sensory = config_dict.get_mandatory('sensory.using_sensory')
         
    if using_sensory:
        sensor_radius = config_dict.get_mandatory('sensory.sensor_radius')
        decay_power = config_dict.get_mandatory('sensory.decay_power', float)
        vector_size = config_dict.get_mandatory('sensory.vector_size', int)
        nociception_enabled = config_dict.get_mandatory('sensory.nociception_enabled', bool)
        olfactory_enabled = config_dict.get_mandatory('sensory.olfactory_enabled', bool)
        nociceptor_radius = config_dict.get_mandatory('sensory.nociceptor_radius', int)
        collision_sensor_enabled = config_dict.get_mandatory('sensory.collision_sensor_enabled', bool)
        # Size calculated from range automatically now
        collision_sensor_range = config_dict.get_mandatory('sensory.collision_sensor_range', int)
    else:
        sensor_radius = 1 # Dummy
        decay_power = 1.0 # Dummy
        vector_size = 10 # Dummy
        nociceptor_radius = 0 # Dummy
        collision_sensor_enabled = False
        collision_sensor_range = 0
    
    # Proprioception Config
    proprioception_enabled = config_dict.get_mandatory('sensory.proprioception_enabled', bool)
    
    # Professional Config Summary
    if config_dict is None:
        raise ValueError("Strict Config: 'config_dict' must be provided to train_agent")

    # Load Visualization Config (Optional)
    viz_config_path = os.path.join(os.path.dirname(__file__), "configs", "visualization", "visualization.yaml")
    if os.path.exists(viz_config_path):
        with open(viz_config_path, 'r') as f:
            viz_dict = yaml.safe_load(f)
            config_dict.merge(viz_dict)
    
    # Resolve Parameters (Argument > Config > Error)
    def resolve_param(arg_val, config_key):
        if arg_val is not None:
            config_dict.set(config_key, arg_val) # Store back for strict lookups
            return arg_val
        return config_dict.get_mandatory(config_key)

    # Essential Training Params
    episodes = int(resolve_param(episodes, 'training.training_episode'))
    
    # Calculate target end episode
    target_end_episode = start_episode + episodes
    
    if start_episode > 0:
        print(f"Resuming training from episode {start_episode}. Target end: {target_end_episode}")
    seed = int(resolve_param(seed, 'training.seed'))
    testing_seed = int(resolve_param(testing_seed, 'testing.seed'))
    
    # Environment Params
    max_steps = int(resolve_param(max_steps, 'environment.max_steps'))
    eat_action_enabled = resolve_param(None, 'environment.eat_action_enabled')
    rest_action_enabled = resolve_param(None, 'environment.rest_action_enabled')
    resources_config = config_dict.get_mandatory('environment.resources')

    # Predator Params
    predator_enabled = config_dict.get_mandatory('predator.enabled', bool)
    
    if predator_enabled:
        predator_move_interval = config_dict.get_mandatory('predator.move_interval', int)
        predator_damage = config_dict.get_mandatory('predator.damage', float)
        predator_start_pos = config_dict.get_mandatory('predator.start_pos')
        predator_random_start_pos = config_dict.get_mandatory('predator.random_start_pos', bool)
        predator_property = config_dict.get_mandatory('predator.property')
    else:
        predator_move_interval = 2
        predator_damage = 0.0
        predator_start_pos = (0, 0)
        predator_random_start_pos = False
        predator_property = None

    # Body Params
    with_satiation = resolve_param(with_satiation, 'body.with_satiation')
    max_satiation = int(resolve_param(None, 'body.max_satiation'))
    start_satiation = int(resolve_param(None, 'body.start_satiation'))
    overeating_death = resolve_param(overeating_death, 'body.overeating_death')
    random_start_satiation = resolve_param(random_start_satiation, 'body.random_start_satiation')
    food_satiation_gain = float(resolve_param(food_satiation_gain, 'body.food_satiation_gain'))
    use_homeostatic_reward = resolve_param(use_homeostatic_reward, 'body.use_homeostatic_reward')
    satiation_setpoint = float(resolve_param(satiation_setpoint, 'body.satiation_setpoint'))
    death_penalty = float(resolve_param(death_penalty, 'body.death_penalty'))
        
    with_injury = resolve_param(with_injury, 'body.with_injury')
    max_injury = float(resolve_param(max_injury, 'body.max_injury'))
    start_injury = float(resolve_param(start_injury, 'body.start_injury'))
    injury_recovery = float(resolve_param(injury_recovery, 'body.injury_recovery'))
    random_start_injury = resolve_param(random_start_injury, 'body.random_start_injury')
    injury_smoothing_duration = int(resolve_param(None, 'body.injury_smoothing_duration'))
    
    device = resolve_param(device, 'training.device')

    # Update config_dict with resolved values for consistency in logging/saving
    config_dict.set('training.training_episode', episodes)
    config_dict.set('training.seed', seed)
    config_dict.set('testing.seed', testing_seed) # Corrected key
    # ... (Update others if needed, mostly used for display/save)
    
    config_dict.set('training.device', device)
    
    if not quiet:
         print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps, random_start_satiation, use_homeostatic_reward, satiation_setpoint, testing_seed, with_injury, device)
    
    # Setup directories
    results_dir = "results"
    
    # Determine Model Name
    if using_sensory:
         model_name = "DQN"
    else:
         model_name = "Tabular_Q_Learning"
         
    # Override from agent config if available
    if config_dict.get_mandatory('agent.algorithm'):
        model_name = config_dict.get_mandatory('agent.algorithm').replace(" ", "_")
        
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = config_dict.get_mandatory('tag')
    
    run_name = f"{timestamp}_{tag}"
    
    output_dir = os.path.join(results_dir, model_name, run_name)
    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots") # Ensure plots dir is tracked
    data_dir = os.path.join(output_dir, "data")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    
    if not quiet:
        print(f"Results will be saved to: {output_dir}")

    # Save configuration
    if config_dict is not None:
        config_save_path = os.path.join(models_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict.to_dict(), f, default_flow_style=False)
        if not quiet:
            print(f"Resolved configuration saved to {config_save_path}")

    # Set numpy/torch    # Common Parameters
    # Strict Migration: try resource_pos, if not found, use food_pos if available
    resource_pos = config_dict.get('environment.resource_pos')
    if resource_pos is None:
         resource_pos = config_dict.get('environment.food_pos', (0, 0))
         
    env = GridWorld(
        height=int(resolve_param(None, 'environment.height')),
        width=int(resolve_param(None, 'environment.width')),
        start=tuple(resolve_param(None, 'environment.start_pos')),
        resources=resources_config,
        with_satiation=with_satiation,
        max_steps=max_steps,
        eat_action_enabled=eat_action_enabled,
        rest_action_enabled=rest_action_enabled,
        predator_enabled=predator_enabled,
        predator_move_interval=predator_move_interval,
        predator_damage=predator_damage,
        predator_start_pos=predator_start_pos,
        predator_random_start_pos=predator_random_start_pos,
        predator_property=predator_property,
        injury_smoothing_duration=injury_smoothing_duration
    )
    
    # Get action dimension from environment spec (dynamic based on eat/rest config)
    action_dim = env.action_spec()['n']
    
    body = InteroceptiveBody(
        max_satiation=max_satiation,
        start_satiation=start_satiation,
        overeating_death=overeating_death, 
        random_start_satiation=random_start_satiation, 
        food_satiation_gain=food_satiation_gain,
        use_homeostatic_reward=use_homeostatic_reward,
        satiation_setpoint=satiation_setpoint,
        death_penalty=death_penalty,
        with_injury=with_injury,
        max_injury=max_injury,
        start_injury=start_injury,
        injury_recovery=injury_recovery,
        random_start_injury=random_start_injury,
        injury_smoothing_duration=injury_smoothing_duration,
        with_satiation=with_satiation
    )
    
    sensory_system = None
    if using_sensory:
        if not quiet:
            print(f"Initializing Sensory System (Radius={sensor_radius}, Decay={decay_power}, VecSize={vector_size}, NociceptorR={nociceptor_radius}, Collision={collision_sensor_enabled})")
        location_sensor = config_dict.get_mandatory('sensory.location_sensor')
        nociception_size = config_dict.get_mandatory('sensory.nociception_size', int)
        location_size = config_dict.get_mandatory('sensory.location_size', int)
        sensory_system = SensorySystem(
            sensor_radius=sensor_radius, 
            vector_size=vector_size, 
            decay_power=decay_power, 
            nociceptor_radius=nociceptor_radius,
            location_sensor=location_sensor,
            nociception_enabled=nociception_enabled,
            olfactory_enabled=olfactory_enabled,
            collision_sensor_enabled=collision_sensor_enabled,
            collision_sensor_range=collision_sensor_range,
            proprioception_enabled=proprioception_enabled,
            nociception_size=nociception_size,
            location_size=location_size,
            num_actions=action_dim,
            with_satiation=with_satiation
        )



    # Initialize Agent
    agent = None
    algorithm = config_dict.get_mandatory('agent.algorithm')
    
    # Calculate Input Dimension and Breakdown using Specs
    input_dim = 0
    dims_breakdown = []
    
    # Log Specs
    print("\n--- RL API Specifications ---")
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    print(f"Environment Action Spec: {action_spec}")
    
    if using_sensory:
         sensory_spec = sensory_system.observation_spec()
         print(f"Sensory System Spec: {sensory_spec}")
         print(f"Dimension Breakdown:\n  {sensory_system.get_dimension_breakdown()}")
         
         for key, val in sensory_spec.items():
             dim = val['shape'][0]
             input_dim += dim
             dims_breakdown.append(f"{key}={dim}")

    else:
        print(f"Environment Observation Spec: {observation_spec}")
        loc_shape = observation_spec['loc']['shape']
        input_dim += loc_shape[0]
        dims_breakdown.append(f"Loc={loc_shape[0]}")
        
        if with_satiation:
            sat_shape = observation_spec['satiation']['shape']
            input_dim += sat_shape[0]
            dims_breakdown.append(f"Sat={sat_shape[0]}")
            
        if with_injury:
            hlth_shape = observation_spec['injury']['shape']
            input_dim += hlth_shape[0]
            dims_breakdown.append(f"Injury={hlth_shape[0]}")
    
    print("--------------------------------------------\n")

    input_details = f"{input_dim} ({', '.join(dims_breakdown)})"
    
    # Frame Stacking Logic
    frame_stack = config_dict.get_mandatory('agent.frame_stack', int)
    
    # input_dim IS the base dimension. We do NOT multiply it here anymore.
    base_input_dim = input_dim
    # input_dim = base_input_dim * frame_stack # REMOVED: Agent handles this internally now (DQN/PPO)
    
    breakdown_str = ', '.join(dims_breakdown)
    if frame_stack > 1:
        input_details = f"Base: {base_input_dim} [{breakdown_str}] x Stack: {frame_stack}"
    else:
        input_details = f"{input_dim} ({breakdown_str})"
        
    stacker = FrameStacker(input_dim=base_input_dim, stack_size=frame_stack)
    
    if algorithm == "DQN":
        if not quiet:
            print(f"Initializing DQN Agent (Input Dim: {input_details})...")
        agent = DQNAgent(
            state_dim=base_input_dim, # Pass BASE dim
            action_dim=action_dim, 
            lr=config_dict.get_mandatory('agent.learning_rate', float),
            gamma=config_dict.get_mandatory('agent.gamma', float),
            buffer_size=config_dict.get_mandatory('agent.buffer_size', int),
            batch_size=config_dict.get_mandatory('agent.batch_size', int),
            epsilon_start=config_dict.get_mandatory('agent.epsilon_start', float),
            epsilon_end=config_dict.get_mandatory('agent.epsilon_end', float),
            epsilon_decay=config_dict.get_mandatory('agent.epsilon_decay', float),
            target_update_freq=config_dict.get_mandatory('agent.target_update_freq', int),
            fc_layers=config_dict.get_mandatory('agent.fc_layers'),
            device=device,
            frame_stack=frame_stack # Pass frame_stack
        )

    elif algorithm == "DRQN":
        if not quiet:
            print(f"Initializing DRQN Agent (Input Dim: {input_details})...")
        agent = DRQNAgent(
            state_dim=base_input_dim, 
            action_dim=action_dim, 
            lr=config_dict.get_mandatory('agent.learning_rate', float),
            gamma=config_dict.get_mandatory('agent.gamma', float),
            buffer_size=config_dict.get_mandatory('agent.buffer_size', int),
            batch_size=config_dict.get_mandatory('agent.batch_size', int),
            trace_length=config_dict.get_mandatory('agent.trace_length', int),
            burn_in_length=config_dict.get_mandatory('agent.burn_in_length', int),
            epsilon_start=config_dict.get_mandatory('agent.epsilon_start', float),
            epsilon_end=config_dict.get_mandatory('agent.epsilon_end', float),
            epsilon_decay=config_dict.get_mandatory('agent.epsilon_decay', float),
            target_update_freq=config_dict.get_mandatory('agent.target_update_freq', int),
            fc_layers=config_dict.get_mandatory('agent.fc_layers'),
            recurrent_layers=config_dict.get_mandatory('agent.recurrent_layers'),
            device=device
        )
        
    elif algorithm == "PPO":
        if not quiet:
            print(f"Initializing PPO Agent (Input Dim: {input_details})...")
        agent = PPOAgent(
            state_dim=base_input_dim, 
            action_dim=action_dim,
            lr_actor=config_dict.get_mandatory('agent.lr_actor', float),
            lr_critic=config_dict.get_mandatory('agent.lr_critic', float),
            gamma=config_dict.get_mandatory('agent.gamma', float),
            K_epochs=config_dict.get_mandatory('agent.K_epochs', int),
            eps_clip=config_dict.get_mandatory('agent.eps_clip', float),
            update_timestep=config_dict.get_mandatory('agent.update_timestep', int),
            entropy_coef=config_dict.get_mandatory('agent.entropy_coef', float),
            actor_fc_layers=config_dict.get_mandatory('agent.actor_fc_layers'),
            critic_fc_layers=config_dict.get_mandatory('agent.critic_fc_layers'),
            device=device,
            frame_stack=frame_stack
        )

    elif algorithm == "RecurrentPPO":
        if not quiet:
            print(f"Initializing Recurrent PPO Agent (Input Dim: {input_details})...")
        agent = RecurrentPPOAgent(
            state_dim=base_input_dim, 
            action_dim=action_dim,
            lr_actor=config_dict.get_mandatory('agent.lr_actor', float),
            lr_critic=config_dict.get_mandatory('agent.lr_critic', float),
            gamma=config_dict.get_mandatory('agent.gamma', float),
            K_epochs=config_dict.get_mandatory('agent.K_epochs', int),
            eps_clip=config_dict.get_mandatory('agent.eps_clip', float),
            update_timestep=config_dict.get_mandatory('agent.update_timestep', int),
            sequence_length=config_dict.get_mandatory('agent.sequence_length', int),
            entropy_coef=config_dict.get_mandatory('agent.entropy_coef', float),
            fc_layers=config_dict.get_mandatory('agent.fc_layers'),
            recurrent_layers=config_dict.get_mandatory('agent.recurrent_layers'),
            actor_fc_layers=config_dict.get_mandatory('agent.actor_fc_layers'),
            critic_fc_layers=config_dict.get_mandatory('agent.critic_fc_layers'),
            device=device
        )

    elif algorithm == "DreamerV3":
        if not quiet:
            print(f"Initializing Dreamer V3 Agent (Input Dim: {input_details})...")
        agent = DreamerV3Agent(
            state_dim=base_input_dim, 
            action_dim=action_dim,
            batch_size=config_dict.get_mandatory('agent.batch_size', int),
            batch_length=config_dict.get_mandatory('agent.batch_length', int),
            model_lr=config_dict.get_mandatory('agent.model_lr', float),
            actor_lr=config_dict.get_mandatory('agent.actor_lr', float),
            value_lr=config_dict.get_mandatory('agent.value_lr', float),
            encoder_dim=config_dict.get_mandatory('agent.encoder_dim', int),
            encoder_fc_layers=config_dict.get_mandatory('agent.encoder_fc_layers'),
            rssm_deter_dim=config_dict.get_mandatory('agent.rssm_deter_dim', int),
            rssm_stoch_dim=config_dict.get_mandatory('agent.rssm_stoch_dim', int),
            rssm_classes=config_dict.get_mandatory('agent.rssm_classes', int),
            decoder_fc_layers=config_dict.get_mandatory('agent.decoder_fc_layers'),
            reward_fc_layers=config_dict.get_mandatory('agent.reward_fc_layers'),
            continue_fc_layers=config_dict.get_mandatory('agent.continue_fc_layers'),
            actor_fc_layers=config_dict.get_mandatory('agent.actor_fc_layers'),
            critic_fc_layers=config_dict.get_mandatory('agent.critic_fc_layers'),
            device=device
        )

    else:
        # Tabular (Default)
        if not quiet:
            print(f"Initializing Tabular Agent ({algorithm})...")
        class CompositeEnv:
            def __init__(self, env, body):
                self.height = env.height
                self.width = env.width
                self.max_satiation = body.max_satiation
                self.with_injury = body.with_injury
                self.max_injury = body.max_injury # Standardize
                
        composite_env = CompositeEnv(env, body)
        agent = QLearningAgent(composite_env, with_satiation=with_satiation)
        agent.epsilon = 1.0 # Start with full exploration

    # Load checkpoint if provided
    if load_checkpoint_path:
        print(f"Loading checkpoint weights from {load_checkpoint_path}...")
        try:
            agent.load(load_checkpoint_path)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            exit(1)
    
    if not quiet:
        print(f"Training agent (with_satiation={with_satiation}, with_injury={with_injury})...")
    start_time = time.time()
    
    # Common Epsilon Params (Agent handles its own, but we track for logs)
    # DQNAgent has internal epsilon, QLearningAgent relies on external assignment typically?
    # QLearningAgent in agent.py uses passed epsilon or default. 
    # train.py was managing it externally.
    
    # We will sync to agent's epsilon
    
    # Checkpoint Frequency
    checkpoint_freq = int(resolve_param(checkpoint_frequency, 'training.checkpoint_frequency'))
    
    episode_rewards = []
    episode_steps = []
    episode_epsilons = []
    
    # Tabular decay logic (legacy)
    tabular_decay_rate = 0.9995
    tabular_min_epsilon = 0.05
    
    # Main Training Loop
    pbar = tqdm(range(start_episode, target_end_episode), disable=quiet, desc="Training")
    
    location_sensor = config_dict.get('sensory.location_sensor', False)
    
    losses = {} # Track latest losses for debug display
    global_step = 0 # Unified counter for WandB (Environment Interactions)
    for episode in pbar:
        # Reset External
        env_state = env.reset()
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        
        # Initialize previous action (REST if enabled, or 0=Up as default)
        # Must be initialized before first sense() call
        if proprioception_enabled:
            previous_action = env.REST_ACTION if env.REST_ACTION is not None else 0
        else:
            previous_action = None
        
        # Determine internal start state
        current_agent_pos = env.agent_pos
        if using_sensory:
             resources = env.get_active_resources()
             extra_data = {
                 'injury_level': body.injury_level, 
                 'max_injury': body.max_injury,
                 'satiation': body.satiation,
                 'max_satiation': body.max_satiation
             }
             sensory_dict = sensory_system.sense(
                 current_agent_pos, resources,
                 grid_height=env.height, grid_width=env.width,
                 previous_action=previous_action,
                 extra_data=extra_data
             )

        # Reset Body (Always reset to clear injury buffers/randomize state)
        body_return = body.reset()
        
        if using_sensory:
            # Construct Dictionary State
            state = sensory_dict.copy()
            if isinstance(body_return, tuple):
                 if with_satiation:
                     state['satiation'] = body_return[0]
                 if with_injury:
                     state['injury'] = body_return[1]
            else:
                 if with_satiation:
                     state['satiation'] = body_return
                 elif with_injury:
                     state['injury'] = body_return
        else:
             state = {'loc': env_state}
             if isinstance(body_return, tuple):
                 if with_satiation:
                     state['satiation'] = body_return[0]
                 if with_injury:
                     state['injury'] = body_return[1]
             else:
                 if with_satiation:
                     state['satiation'] = body_return
                 elif with_injury:
                     state['injury'] = body_return
        
        done = False
        total_reward = 0
        steps = 0
        
        # Preprocess logic
        flat_state = None
        state_array = None
        
        if isinstance(agent, (DQNAgent, PPOAgent, DRQNAgent, RecurrentPPOAgent, DreamerV3Agent)):
            flat_state = preprocess_state(state, env.height, env.width, body.max_satiation, body.max_injury)
            # Stack the initial state and FLATTEN for agent compatibility (Seq vs Grid issues)
            state_array = stacker.reset(flat_state).flatten()


        else:
            # Tabular - Convert Dictionary to Tuple for Q-Table Indexing
            # Structure: (row, col, satiation, health) -> ints
            # Note: We rely on logic in QLearningAgent to extract logic, 
            # OR we standardize here. QLearningAgent line 84: tuple(int(x) for x in state)
            # This iterates keys if dict. We must provide values tuple.
            
            # Extract Location
            if 'loc' in state:
                r, c = state['loc']
            else:
                 # Should not happen based on preprocess logic, but strictly:
                 # If using_sensory, keys are vectors. Tabular shouldn't use sensory really.
                 # Assuming Tabular uses standard coords + body.
                 # If using_sensory=True with Tabular, it will fail unless we define a discrete mapping.
                 # For now, assume Tabular logic in repo handles (row, col, sat...)
                 # Let's extract values in order.
                 pass # Fallback to existing logic if not dict?
                 
            # Construct tuple
            # If using_sensory=False (default for Tabular), state is {'loc': (r,c), 'satiation': s...}
            tabular_list = []
            if 'loc' in state:
                tabular_list.extend(state['loc']) # r, c
            if with_satiation:
                tabular_list.append(state['satiation'])
            if with_injury:
                tabular_list.append(state.get('injury', state.get('injury', 0))) # Support both during transition
                
            state_array = tuple(int(x) for x in tabular_list)
        
        while not done:
            global_step += 1
            if debug and global_step % 100 == 0:
                print(f"[DEBUG Heartbeat] Global Step: {global_step}, Episode: {episode+1}, Step: {steps+1}")
            if isinstance(agent, (DQNAgent, PPOAgent, DRQNAgent, RecurrentPPOAgent, DreamerV3Agent)):
                action = agent.choose_action(state_array)
            else:
                action = agent.choose_action(state_array) # Pass the tuple we created
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            
            # Update Observations
            # Update Observations
            if using_sensory:
                 resources = env.get_active_resources()
                 extra_data = {
                     'injury_level': body.injury_level, 
                     'max_injury': body.max_injury,
                     'satiation': body.satiation,
                     'max_satiation': body.max_satiation
                 }
                 next_sensory_dict = sensory_system.sense(
                     next_env_state, resources,
                     grid_height=env.height, grid_width=env.width,
                     previous_action=action,
                     extra_data=extra_data
                 )
            
            # Step Body / Physiology
            body_return, body_reward, body_done = body.step(info)
            
            # Reward/Done Logic:
            # If interoceptive (homeostatic reward), body is the primary source.
            # If conventional (survival branch), env is primary but body can kill.
            if use_homeostatic_reward:
                reward = body_reward
                done = env_done or body_done
            else:
                reward = env_reward
                done = env_done or body_done
            
            # Prepare next state dictionary
            next_state = {}
            if using_sensory:
                 next_state.update(next_sensory_dict)
            else:
                 next_state['loc'] = next_env_state
                 
            if isinstance(body_return, tuple):
                 if with_satiation:
                     next_state['satiation'] = body_return[0]
                 if with_injury:
                     next_state['injury'] = body_return[1]
            else:
                 if with_satiation:
                     next_state['satiation'] = body_return
                 elif with_injury:
                     # This case: body only returns one value, and it must be injury if satiation is off
                     next_state['injury'] = body_return
            
            if isinstance(agent, (DQNAgent, PPOAgent, DRQNAgent, RecurrentPPOAgent, DreamerV3Agent)):
                # DQN/PPO/DRQN/RecurrentPPO/Dreamer Update
                # Use current action as the "previous action" for the next state
                flat_next_state_raw = preprocess_state(next_state, env.height, env.width, body.max_satiation, body.max_injury)
                # Stack next state and Flatten
                next_state_stacked = stacker.step(flat_next_state_raw).flatten()


                
                # Store Transition (Use stacked states for non-recurrent agents if needed, 
                # but usually recurrent agents manage their own history. 
                # However, our design choice was stack at env level.
                # So even DRQN receives stacked input? 
                # Config says DRQN frame_stack=1, so it's identity. Correct.)
                
                # Check agent signature for store_transition
                # PPO/RecurrentPPO: (state, action, log_prob, reward, done)
                # DQN/DRQN: (state, action, next_state, reward, done)
                # Dreamer: (state, action, reward, done) - likely adds to buffer
                
                # Unified Access: All agents support (state, action, reward, next_state, done)
                # PPO/RecurrentPPO: internally ignores s,a,ns but expects r at pos 3.
                # Dreamer: expects r at pos 3, ns at pos 4 (ignored?), d at pos 5.
                agent.store_transition(state_array, action, reward, next_state_stacked, done)
                
                # Update and capture losses
                if debug:
                    print(f"[DEBUG Update Start] Ep:{episode+1} St:{steps+1}")
                start_upd = time.time()
                update_result = agent.update()
                if debug:
                    print(f"[DEBUG Update End] Ep:{episode+1} St:{steps+1} (Took {(time.time() - start_upd):.3f}s)")
                upd_duration = (time.time() - start_upd) * 1000 # ms
                
                if isinstance(update_result, dict):
                    losses = update_result.copy() # Use copy to avoid mutating agent inner dict
                    losses["global_step"] = global_step
                    if wandb.run is not None:
                        # Log losses with global_step explicitly included as a metric
                        wandb.log(losses, step=global_step)
                
                # Dynamic Debug Print (Per Step)
                if debug and not quiet:
                    loss_str = ""
                    if losses:
                        if isinstance(agent, DreamerV3Agent):
                            m_l = losses.get('model_loss', 0)
                            r_l = losses.get('recon_loss', 0)
                            rew_l = losses.get('rew_loss', 0)
                            kl = losses.get('kl_loss', 0)
                            a_l = losses.get('actor_loss', 0)
                            c_l = losses.get('critic_loss', 0)
                            gn = losses.get('model_grad_norm', 0)
                            v_m = losses.get('value_mean', 0)
                            loss_str = f"L:[M:{m_l:.2f}(Re:{r_l:.2f},Rw:{rew_l:.2f}) A:{a_l:.2f} C:{c_l:.2f} KL:{kl:.1f}] GN:{gn:.1f} V:{v_m:.1f} "
                        else:
                            # Show primary loss for DQN/PPO etc
                            main_loss = losses.get('loss', losses.get('mean_loss', 0))
                            loss_str = f"L:{main_loss:.4f} "
                    
                    # Print as new line for granular history as requested
                    print(f"Ep:{episode+1} St:{steps+1} Act:{action} R:{total_reward+reward:.1f} {loss_str}Time:{upd_duration:.1f}ms")

                # Advance State
                state = next_state # Tuple kept for logic
                state_array = next_state_stacked # Stacked for next iter
            else:
                # Tabular Update
                # Construct next tuple
                tabular_next_list = []
                if 'loc' in next_state:
                    tabular_next_list.extend(next_state['loc'])
                if with_satiation:
                    tabular_next_list.append(next_state['satiation'])
                if with_injury:
                    tabular_next_list.append(next_state['injury'])
                
                next_state_array = tuple(int(x) for x in tabular_next_list)
                
                agent.update(state_array, action, reward, next_state_array) # Use tuples
                state = next_state
                
            # Update previous action for next iteration
            if proprioception_enabled:
                previous_action = action
                
            total_reward += reward
            steps += 1
            global_step += 1
            
        if debug and not quiet:
            print(f"--- Episode {episode+1} Finished ---")
        
        # End of Episode
        episode_epsilons.append(agent.epsilon if hasattr(agent, 'epsilon') else 0.0)

        if not isinstance(agent, (DQNAgent, PPOAgent, DRQNAgent, RecurrentPPOAgent, DreamerV3Agent)):
            # Manually decay for tabular
            agent.epsilon = max(tabular_min_epsilon, agent.epsilon * tabular_decay_rate)
        # DQN/DRQN handles decay internally in update(), also target net update handled internally
        # PPO: No epsilon decay, no target net
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Log episode metrics to WandB
        if wandb.run is not None:
            wandb.log({
                "Episode/Reward": total_reward,
                "Episode/Steps": steps,
                "Episode/Epsilon": agent.epsilon if hasattr(agent, 'epsilon') else 0.0,
                "Episode/Number": episode + 1,
            }, step=global_step)
        
        if not quiet:
            # Update tqdm postfix
            if (episode + 1) % 10 == 0:
                 current_epsilon = agent.epsilon if hasattr(agent, 'epsilon') else 0.0
                 avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0
                 pbar.set_postfix({'Rw': f'{avg_reward:.1f}', 'Eps': f'{current_epsilon:.3f}'})

        # Check milestones
            
        # Check milestones
        # Checkpoint Saving
        if (episode + 1) % checkpoint_freq == 0:
            pct = episode + 1
            
            if isinstance(agent, DQNAgent):
                model_snap_filename = os.path.join(models_dir, f"dqn_model_{pct}.ckpt")
                agent.save(model_snap_filename)
            elif isinstance(agent, DRQNAgent):
                model_snap_filename = os.path.join(models_dir, f"drqn_model_{pct}.ckpt")
                agent.save(model_snap_filename)
            elif isinstance(agent, PPOAgent):
                model_snap_filename = os.path.join(models_dir, f"ppo_model_{pct}.ckpt")
                agent.save(model_snap_filename)
            elif isinstance(agent, RecurrentPPOAgent):
                model_snap_filename = os.path.join(models_dir, f"recurrent_ppo_model_{pct}.ckpt")
                agent.save(model_snap_filename)
            elif isinstance(agent, DreamerV3Agent):
                model_snap_filename = os.path.join(models_dir, f"dreamer_model_{pct}.ckpt")
                agent.save(model_snap_filename)
            else:
                model_snap_filename = os.path.join(models_dir, f"q_table_{pct}.npy")
                agent.save(model_snap_filename)
            
            # Integrated Evaluation and Video Generation (if enabled)
            if config_dict.get('evaluation.video_during_training') or config_dict.get('visualization.enabled'):
                try:
                    if not quiet:
                         # Using tqdm.write to avoid breaking progress bar
                         tqdm.write(f"Running evaluation for checkpoint {pct}...")
                        
                    # Use evaluate_agent utility
                    # We pass the CURRENT agent and env (it resets env)
                    # We pass 'wandb_run_path=None' so it uploads to currently active run (if wandb.run is set)
                    # Note: evaluate_agent sets epsilon to 0 internally and restores it.
                    evaluate_agent(
                        agent=agent,
                        env=env,
                        body=body,
                        sensory_system=sensory_system,
                        config=config_dict,
                        num_episodes=3, # Quick check
                        results_dir=output_dir, # Same output dir
                        checkpoint_pct=pct,
                        wandb_run_path=None, # Use active run
                        quiet=True # Suppress internal prints
                    )
                except Exception as e:
                    tqdm.write(f"Warning: Evaluation failed during training: {e}")
    
    if not quiet:
        print() # Newline after progress bar
    
    training_time = time.time() - start_time
    if not quiet:
        print(f"Training completed in {training_time:.2f} seconds.")
    
    # Final model save removed (redundant with frequency checkpointing)
    # The last checkpoint is sufficient if frequency aligns or user can use latest.

    
    # Save training history
    import csv
    history_filename = os.path.join(data_dir, "training_history.csv")

    with open(history_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps', 'epsilon'])
        for i in range(len(episode_rewards)):
            writer.writerow([start_episode + i + 1, episode_rewards[i], episode_steps[i], episode_epsilons[i]])
    if not quiet:
        print(f"Training history saved to {history_filename}")

    # Generate learning curves
    # Generate milestones for plotting
    # Generate milestones for plotting
    milestones = {ep: f"Ckpt" for ep in range(start_episode + checkpoint_freq, target_end_episode + 1, checkpoint_freq)}
    plot_learning_curves(history_filename, plots_dir, config_dict, max_steps=max_steps, milestones=milestones)
    
    if not quiet:
        print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to base config YAML")
    parser.add_argument("--agent_config", type=str, required=True, help="Path to agent config YAML (Required)")
    parser.add_argument("--tag", type=str, help="Tag for the training run")
    parser.add_argument("--device", type=str, help="Device to use (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto')")
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    parser.add_argument("--no-overeating-death", action="store_true", help="Disable death by overeating")
    parser.add_argument("--wandb-project", type=str, help="WandB Project Name")
    parser.add_argument("--wandb-group", type=str, help="WandB Group Name")
    parser.add_argument("--wandb-job-type", type=str, help="WandB Job Type")
    parser.add_argument("--wandb-name", type=str, help="WandB Run Name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress output and progress bar")
    parser.add_argument("--debug", action="store_true", help="Enable dynamic debug status and granular logging")
    parser.add_argument("--checkpoint-frequency", type=int, help="Save checkpoint every N episodes")
    parser.add_argument("--load-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-resume-id", type=str, help="WandB Run ID to resume logging")
    args = parser.parse_args()
    
    # Load default config
    config = get_default_config()
    from src.utils.config import Config
    
    # Load WandB config
    # Load Train Config
    train_config_path = "configs/train/default.yaml"
    if os.path.exists(train_config_path):
        if not args.quiet:
            print(f"Loading train config from {train_config_path}")
        train_config = Config.load_yaml(train_config_path)
        config.merge(train_config)

    # Load Eval Config
    eval_config_path = "configs/evaluation/default.yaml"
    if os.path.exists(eval_config_path):
        if not args.quiet:
            print(f"Loading eval config from {eval_config_path}")
        eval_config = Config.load_yaml(eval_config_path)
        config.merge(eval_config)

    # Load Logger Config
    logger_config_path = "configs/logger/wandb.yaml"
    if os.path.exists(logger_config_path):
        if not args.quiet:
            print(f"Loading logger config from {logger_config_path}")
        logger_config = Config.load_yaml(logger_config_path)
        config.merge(logger_config)
    
    # Load and merge user override config (e.g., ablation configs)
    if args.config:
        if not args.quiet:
            print(f"Loading override config from: {args.config}")
        user_config = Config.load_yaml(args.config)
        config.merge(user_config)
        
    # Override WandB settings with CLI args
    if args.wandb_project:
        config.set('wandb.project', args.wandb_project)
    if args.wandb_group:
        config.set('wandb.group', args.wandb_group)
    if args.wandb_job_type:
        config.set('wandb.job_type', args.wandb_job_type)
    if args.wandb_name:
        config.set('wandb.name', args.wandb_name)
    if args.no_wandb:
        config.set('wandb.disabled', True)

    # Load and merge agent config
    if args.agent_config:
        if not args.quiet:
            print(f"Loading agent config from: {args.agent_config}")
        agent_config = Config.load_yaml(args.agent_config)
        config.merge(agent_config)
    
    # Set tag
    if args.tag:
        config.set('tag', args.tag)
    
    # Ensure mode uses config if not disabled via CLI
    if not config.get('wandb.disabled'):
         if config.get('wandb.mode') == 'disabled':
             config.set('wandb.disabled', True)
    

        
    # Overrides and Strict Pass-through
    # We pass explicit CLI args or None. train_agent will strictly resolve against config.
    
    # Logic for flags
    arg_with_satiation = False if args.no_satiation else None
    arg_overeating_death = False if args.no_overeating_death else None
    
    # For parameters not exposed in argparse, we pass None so train_agent enforces config.
    
    # Continual Learning Parsing
    start_episode = 0
    if args.load_checkpoint:
        import re
        ckpt_filename = os.path.basename(args.load_checkpoint)
        # Try to parse number from typical names: model_100.ckpt, dqn_model_50.ckpt
        # Regex to find the last number properly
        match = re.search(r"_(\d+)\.(ckpt|pth|npy)$", ckpt_filename)
        if match:
            start_episode = int(match.group(1))
        else:
            print(f"Warning: Could not parse episode number from {ckpt_filename}. Starting from 0.")
            start_episode = 0
            
    # Add wandb resume id to config if present
    if args.wandb_resume_id:
        config.set('wandb.resume_id', args.wandb_resume_id)
    
    train_agent(episodes=args.episodes, 
                seed=args.seed, 
                with_satiation=arg_with_satiation, 
                overeating_death=arg_overeating_death, 
                food_satiation_gain=None, 
                max_steps=None, 
                random_start_satiation=None, 
                use_homeostatic_reward=None, 
                satiation_setpoint=None, 
                death_penalty=None, 
                testing_seed=None, 
                config_dict=config,
                with_injury=None, 
                max_injury=None, 
                start_injury=None, 
                injury_recovery=None, 
                random_start_injury=None,
                prob_switch_to_danger=None, 
                min_danger_duration=None, 
                damage_amount=None,
                prob_switch_to_food=None, 
                min_food_duration=None, 
                device=args.device,
                checkpoint_frequency=args.checkpoint_frequency, 
                quiet=args.quiet,
                debug=args.debug,
                start_episode=start_episode,
                load_checkpoint_path=args.load_checkpoint)

