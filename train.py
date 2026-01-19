"""
Training script for the GridWorld Reinforcement Learning agent.

This script:
1. Initializes the `GridWorld` environment (Conventional or Interoceptive) and `InteroceptiveBody`.
2. Creates an RL Agent (Tabular Q-Learning, DQN, or PPO).
3. Trains the agent for a specified number of episodes.
4. Periodically saves checkpoints (models) and visualizations (learning curves) to `results/`.

Arguments:
- `--episodes <int>`: (Default: 100000) Total number of training episodes.
- `--seed <int>`: (Default: 42) Random seed for reproducibility.
- `--agent_config <path>`: Path to agent-specific config (e.g., `configs/models/ppo.yaml`).
- `--tag <str>`: Tag for the training run directory.
- `--device <str>`: Device to use for training (e.g., `cpu`, `cuda`, `cuda:0`, `auto`). Default: `auto`.
- `--wandb-project <str>`: WandB project name (default: "grid_world_pain").
- `--wandb-group <str>`: WandB group name for grouping runs.
- `--wandb-name <str>`: Specific name for the run.
- `--no-wandb`: Disable WandB logging.

Usage Examples:

1. **Train DQN**:
   ```bash
   python train.py --agent_config configs/models/dqn.yaml --episodes 1000 --wandb-project my_project
   ```

2. **Train PPO**:
   ```bash
   python train.py --agent_config configs/models/ppo.yaml --episodes 5000
   ```
   
3. **Train Tabular**:
   ```bash
   python train.py --agent_config configs/models/q_learning.yaml
   ```
"""
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
                         with_health, danger_prob, damage_amount, device="auto"):
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
    env_data = {
        "Grid Size": f"{config_dict.get_mandatory('environment.height', int)}x{config_dict.get_mandatory('environment.width', int)}",
        "Resource Position": str(config_dict.get_mandatory('environment.resource_pos')),
        "Max Steps": max_steps,
        "Mode": "Interoceptive (Homeostasis)" if with_satiation else "Conventional (Goal-driven)"
    }
    if with_satiation:
        env_data["Reward Logic"] = "Homeostatic (Drive Reduction)" if use_homeostatic_reward else "Survival Step"
        if use_homeostatic_reward:
            env_data["Satiation Setpoint"] = satiation_setpoint
            
    if with_health:
        env_data["Danger Prob"] = danger_prob
        env_data["Damage Amount"] = damage_amount
        
    env_data["Food Prob"] = config_dict.get_mandatory('environment.food_prob', float)
    env_data["Food Duration"] = config_dict.get_mandatory('environment.food_duration', int)
        
    print_section("Environment", env_data)

    # Body (if applicable)
    if with_satiation:
        body_data = {
            "Max Satiation": config_dict.get_mandatory('body.max_satiation'),
            "Start Satiation": config_dict.get_mandatory('body.start_satiation'),
            "Random Start Sat": "ENABLED" if random_start_satiation else "DISABLED",
            "Overeating Death": "ENABLED" if overeating_death else "DISABLED"
        }
        if with_health:
             body_data["Max Health"] = config_dict.get_mandatory('body.max_health')
             body_data["Health Recovery"] = config_dict.get_mandatory('body.health_recovery')
             
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
             agent_data["Food Radius"] = config_dict.get_mandatory('sensory.food_radius')
             agent_data["Danger Radius"] = config_dict.get_mandatory('sensory.danger_radius')
             
    elif algorithm == "DRQN":
        agent_data = {
            "Algorithm": "Deep Recurrent Q-Network (DRQN)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Batch Size": config_dict.get_mandatory('agent.batch_size'),
            "Trace Length": config_dict.get_mandatory('agent.trace_length')
        }
        if using_sensory:
             agent_data["Food Radius"] = config_dict.get_mandatory('sensory.food_radius')
             agent_data["Danger Radius"] = config_dict.get_mandatory('sensory.danger_radius')

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
             agent_data["Food Radius"] = config_dict.get_mandatory('sensory.food_radius')
             agent_data["Danger Radius"] = config_dict.get_mandatory('sensory.danger_radius')

    elif algorithm == "DreamerV3":
        agent_data = {
            "Algorithm": "Dreamer V3",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Batch Size": config_dict.get_mandatory('agent.batch_size'),
            "Batch Length": config_dict.get_mandatory('agent.batch_length'),
        }
        if using_sensory:
             agent_data["Food Radius"] = config_dict.get_mandatory('sensory.food_radius')
             agent_data["Danger Radius"] = config_dict.get_mandatory('sensory.danger_radius')

    elif algorithm == "RecurrentPPO":
        agent_data = {
            "Algorithm": "Recurrent PPO (LSTM)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Sequence Length": config_dict.get_mandatory('agent.sequence_length'),
            "Update Timestep": config_dict.get_mandatory('agent.update_timestep')
        }
        if using_sensory:
             agent_data["Food Radius"] = config_dict.get_mandatory('sensory.food_radius')
             agent_data["Danger Radius"] = config_dict.get_mandatory('sensory.danger_radius')

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
                with_health=None, max_health=None, start_health=None, health_recovery=None, start_health_random=None,
                danger_prob=None, danger_duration=None, damage_amount=None,
                food_prob=None, food_duration=None, device="auto", quiet=False, debug=False):
    """
    Trains the RL Agent (Tabular Q-Learning, DQN, or PPO).
    """
    import os 
    
    # Initialize WandB
    # WandB Initialization
    if config_dict and not config_dict.get_mandatory('wandb.disabled'):
        wandb_enabled = True
        
        # Load project-level config first
        wandb_project = config_dict.get_mandatory('wandb.project')
        wandb_group = config_dict.get_mandatory('wandb.group')
        wandb_job_type = config_dict.get_mandatory('wandb.job_type')
        wandb_name = config_dict.get_mandatory('wandb.name')
        
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            job_type=wandb_job_type,
            name=wandb_name,
            config=config_dict.to_dict(),
            reinit=True
        )
        
        # Log Source Code
        # Explicitly log key files and src directory
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        
        # Define x-axis for different metrics
        # Episode metrics use Episode/Number as x-axis
        wandb.define_metric("Episode/*", step_metric="Episode/Number")
        # Step metrics (losses) use global_step as x-axis
        wandb.define_metric("global_step", step_metric="global_step") 
        wandb.define_metric("*", step_metric="global_step")
    
    # Extract Sensory Config
    # Strict retrieval for using_sensory?
    # User requested no safe defaults.
    using_sensory = config_dict.get_mandatory('sensory.using_sensory')
         
    if using_sensory:
        food_radius = config_dict.get_mandatory('sensory.food_radius')
        danger_radius = config_dict.get_mandatory('sensory.danger_radius')
    else:
        food_radius = 1 # Dummy
        danger_radius = 1 # Dummy
    
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
    seed = int(resolve_param(seed, 'training.seed'))
    testing_seed = int(resolve_param(testing_seed, 'testing.seed'))
    
    # Environment Params
    max_steps = int(resolve_param(max_steps, 'environment.max_steps'))
    danger_prob = float(resolve_param(danger_prob, 'environment.danger_prob'))
    danger_duration = int(resolve_param(danger_duration, 'environment.danger_duration'))
    damage_amount = float(resolve_param(damage_amount, 'environment.damage_amount'))
    food_prob = float(resolve_param(food_prob, 'environment.food_prob'))
    food_duration = int(resolve_param(food_duration, 'environment.food_duration'))

    # Body Params
    with_satiation = resolve_param(with_satiation, 'body.with_satiation')
    if with_satiation:
        overeating_death = resolve_param(overeating_death, 'body.overeating_death')
        random_start_satiation = resolve_param(random_start_satiation, 'body.random_start_satiation')
        food_satiation_gain = float(resolve_param(food_satiation_gain, 'body.food_satiation_gain'))
        use_homeostatic_reward = resolve_param(use_homeostatic_reward, 'body.use_homeostatic_reward')
        satiation_setpoint = float(resolve_param(satiation_setpoint, 'body.satiation_setpoint'))
        death_penalty = float(resolve_param(death_penalty, 'body.death_penalty'))
        
    with_health = resolve_param(with_health, 'body.with_health')
    if with_health:
        max_health = float(resolve_param(max_health, 'body.max_health'))
        start_health = float(resolve_param(start_health, 'body.start_health'))
        health_recovery = float(resolve_param(health_recovery, 'body.health_recovery'))
        start_health_random = resolve_param(start_health_random, 'body.start_health_random')
    
    device = resolve_param(device, 'training.device')

    # Update config_dict with resolved values for consistency in logging/saving
    config_dict.set('training.training_episode', episodes)
    config_dict.set('training.seed', seed)
    config_dict.set('testing.seed', testing_seed) # Corrected key
    # ... (Update others if needed, mostly used for display/save)
    
    config_dict.set('training.device', device)
    
    if not quiet:
         print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps, random_start_satiation, use_homeostatic_reward, satiation_setpoint, testing_seed, with_health, danger_prob, damage_amount, device)
    
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
         resource_pos = config_dict.get_mandatory('environment.food_pos')
         
    env = GridWorld(
        height=config_dict.get_mandatory('environment.height'),
        width=config_dict.get_mandatory('environment.width'),
        start=tuple(config_dict.get_mandatory('environment.start_pos')),
        resource_pos=tuple(resource_pos),
        max_steps=max_steps, # Resolved earlier
        danger_prob=danger_prob, # Resolved earlier
        danger_duration=danger_duration,
        damage_amount=damage_amount,
        food_prob=food_prob,
        food_duration=food_duration,
        relocate_resource=config_dict.get_mandatory('environment.relocate_resource'),
        relocation_steps=config_dict.get_mandatory('environment.relocation_steps')
    )
    
    body = InteroceptiveBody(
        overeating_death=overeating_death, 
        random_start_satiation=random_start_satiation, 
        food_satiation_gain=food_satiation_gain,
        use_homeostatic_reward=use_homeostatic_reward,
        satiation_setpoint=satiation_setpoint,
        death_penalty=death_penalty,
        with_health=with_health,
        max_health=max_health,
        start_health=start_health,
        health_recovery=health_recovery,
        start_health_random=start_health_random
    )
    
    sensory_system = None
    if using_sensory:
        if not quiet:
            print(f"Initializing Sensory System (Food R={food_radius}, Danger R={danger_radius})")
        sensory_system = SensorySystem(food_radius=food_radius, danger_radius=danger_radius)

    # Define Preprocessor for DQN
    def preprocess_state(state_tuple):
        """
        Flattens state tuple to float array.
        Handles both Sensory (Vector) and Conventional (Coordinate) inputs.
        """
        flat_list = []
        
        if using_sensory:
            food_idx = state_tuple[0]
            danger_idx = state_tuple[1]
            food_vec = sensory_system.food_sensor.index_to_vector(food_idx)
            danger_vec = sensory_system.danger_sensor.index_to_vector(danger_idx)
            flat_list.extend(food_vec)
            flat_list.extend(danger_vec)
            
            # Body states follow sensory
            body_start_idx = 2
        else:
            # Conventional: (row, col)
            # Normalize coordinates?
            row = state_tuple[0]
            col = state_tuple[1]
            flat_list.append(row / env.height)
            flat_list.append(col / env.width)
            
            body_start_idx = 2
            
        # Append Body States if present
        if len(state_tuple) > body_start_idx:
            satiation = state_tuple[body_start_idx]
            flat_list.append(satiation / body.max_satiation) 
            
        if len(state_tuple) > body_start_idx + 1:
            health = state_tuple[body_start_idx + 1]
            flat_list.append(health / body.max_health)
            
        return np.array(flat_list, dtype=np.float32)

    # Initialize Agent
    agent = None
    algorithm = config_dict.get_mandatory('agent.algorithm')
    
    if algorithm == "DQN":
        # Calculate Input Dimension
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 # row, col
             
        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
        
        if not quiet:
            print(f"Initializing DQN Agent (Input Dim: {input_dim})...")
        if not quiet:
            print(f"Initializing DQN Agent (Input Dim: {input_dim})...")
        agent = DQNAgent(
            state_dim=input_dim, 
            action_dim=5, 
            lr=config_dict.get_mandatory('agent.learning_rate', float),
            gamma=config_dict.get_mandatory('agent.gamma', float),
            buffer_size=config_dict.get_mandatory('agent.buffer_size', int),
            batch_size=config_dict.get_mandatory('agent.batch_size', int),
            epsilon_start=config_dict.get_mandatory('agent.epsilon_start', float),
            epsilon_end=config_dict.get_mandatory('agent.epsilon_end', float),
            epsilon_decay=config_dict.get_mandatory('agent.epsilon_decay', float),
            target_update_freq=config_dict.get_mandatory('agent.target_update_freq', int),
            fc_layers=config_dict.get_mandatory('agent.fc_layers'),
            device=device
        )

    elif algorithm == "DRQN":
        # Calculate Input Dimension
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 # row, col
             
        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
        
        if not quiet:
            print(f"Initializing DRQN Agent (Input Dim: {input_dim})...")
        agent = DRQNAgent(
            state_dim=input_dim, 
            action_dim=5, 
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
        # Calculate Input Dimension
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 # row, col
             
        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
                 
        if not quiet:
            print(f"Initializing PPO Agent (Input Dim: {input_dim})...")
        agent = PPOAgent(
            state_dim=input_dim, 
            action_dim=5,
            lr_actor=config_dict.get_mandatory('agent.lr_actor', float),
            lr_critic=config_dict.get_mandatory('agent.lr_critic', float),
            gamma=config_dict.get_mandatory('agent.gamma', float),
            K_epochs=config_dict.get_mandatory('agent.K_epochs', int),
            eps_clip=config_dict.get_mandatory('agent.eps_clip', float),
            update_timestep=config_dict.get_mandatory('agent.update_timestep', int),
            entropy_coef=config_dict.get_mandatory('agent.entropy_coef', float),
            actor_fc_layers=config_dict.get_mandatory('agent.actor_fc_layers'),
            critic_fc_layers=config_dict.get_mandatory('agent.critic_fc_layers'),
            device=device
        )

    elif algorithm == "RecurrentPPO":
        # Calculate Input Dimension
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 # row, col
             
        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
                 
        if not quiet:
            print(f"Initializing Recurrent PPO Agent (Input Dim: {input_dim})...")
        agent = RecurrentPPOAgent(
            state_dim=input_dim, 
            action_dim=5,
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
        # Calculate Input Dimension
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 # row, col
             
        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
                 
        if not quiet:
            print(f"Initializing Dreamer V3 Agent (Input Dim: {input_dim})...")
        agent = DreamerV3Agent(
            state_dim=input_dim, 
            action_dim=5,
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
                self.with_health = body.with_health
                self.max_health = body.max_health
                
        composite_env = CompositeEnv(env, body)
        agent = QLearningAgent(composite_env, with_satiation=with_satiation)
        agent.epsilon = 1.0 # Start with full exploration

    
    if not quiet:
        print(f"Training agent (with_satiation={with_satiation}, with_health={with_health})...")
    start_time = time.time()
    
    # Common Epsilon Params (Agent handles its own, but we track for logs)
    # DQNAgent has internal epsilon, QLearningAgent relies on external assignment typically?
    # QLearningAgent in agent.py uses passed epsilon or default. 
    # train.py was managing it externally.
    
    # We will sync to agent's epsilon
    
    milestones = {int(episodes * p): int(p * 100) for p in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]}
    
    episode_rewards = []
    episode_steps = []
    episode_epsilons = []
    
    # Tabular decay logic (legacy)
    tabular_decay_rate = 0.9995
    tabular_min_epsilon = 0.05
    
    tabular_decay_rate = 0.9995
    tabular_min_epsilon = 0.05
    
    # Progress Bar with tqdm
    pbar = tqdm(range(episodes), desc="Training", unit="ep", disable=quiet or debug)
    
    losses = {} # Track latest losses for debug display
    global_step = 0 # Unified counter for WandB (Environment Interactions)
    for episode in pbar:
        # Reset External
        env_state = env.reset()
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        
        # Determine internal start state
        current_agent_pos = env.agent_pos
        current_danger_pos_list = []
        if env.is_danger:
             current_danger_pos_list = [env.resource_pos]

        if using_sensory:
             sensory_state = sensory_system.sense(current_agent_pos, env.resource_pos, current_danger_pos_list)

        if with_satiation:
            body_return = body.reset()
            if using_sensory:
                 if isinstance(body_return, tuple):
                     state = (*sensory_state, *body_return)
                 else:
                     state = (*sensory_state, body_return)
            else:
                if with_health:
                    satiation, health = body_return
                    state = (*env_state, satiation, health)
                else:
                    satiation = body_return
                    state = (*env_state, satiation)
        else:
            if using_sensory:
                state = sensory_state
            else:
                state = env_state
        
        done = False
        total_reward = 0
        steps = 0
        
        # Preprocess for DQN/PPO
        flat_state = None
        if isinstance(agent, (DQNAgent, PPOAgent, DRQNAgent, RecurrentPPOAgent, DreamerV3Agent)):
            flat_state = preprocess_state(state)
        
        while not done:
            if isinstance(agent, (DQNAgent, PPOAgent, DRQNAgent, RecurrentPPOAgent, DreamerV3Agent)):
                action = agent.choose_action(flat_state)
            else:
                action = agent.choose_action(state)
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            
            # Update Observations
            current_agent_pos = env.agent_pos
            current_danger_pos_list = []
            if env.is_danger:
                 current_danger_pos_list = [env.resource_pos]

            if using_sensory:
                 next_sensory_state = sensory_system.sense(current_agent_pos, env.resource_pos, current_danger_pos_list)
            
            if with_satiation:
                body_return, reward, body_done = body.step(info)
                done = env_done or body_done
                
                if using_sensory:
                     if isinstance(body_return, tuple):
                         next_state = (*next_sensory_state, *body_return)
                     else:
                         next_state = (*next_sensory_state, body_return)
                else:
                    if with_health:
                        next_state = (*next_env_state, *body_return)
                    else:
                        next_state = (*next_env_state, body_return)
            else:
                reward = env_reward
                done = env_done
                if using_sensory:
                    next_state = next_sensory_state
                else:
                    next_state = next_env_state
                    next_state = next_env_state
            
            if isinstance(agent, (DQNAgent, PPOAgent, DRQNAgent, RecurrentPPOAgent, DreamerV3Agent)):
                # DQN/PPO/DRQN/RecurrentPPO/Dreamer Update
                flat_next_state = preprocess_state(next_state)
                agent.store_transition(flat_state, action, reward, flat_next_state, done)
                
                # Update and capture losses
                start_upd = time.time()
                update_result = agent.update()
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

                state = next_state # Tuple kept for logic
                flat_state = flat_next_state # Flat for next iter
            else:
                # Tabular Update
                agent.update(state, action, reward, next_state)
                state = next_state
                
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
        if (episode + 1) in milestones:
            pct = milestones[episode + 1]
            
            if isinstance(agent, DQNAgent):
                model_snap_filename = os.path.join(models_dir, f"dqn_model_{pct}.pth")
                agent.save(model_snap_filename)
            elif isinstance(agent, DRQNAgent):
                model_snap_filename = os.path.join(models_dir, f"drqn_model_{pct}.pth")
                agent.save(model_snap_filename)
            elif isinstance(agent, PPOAgent):
                model_snap_filename = os.path.join(models_dir, f"ppo_model_{pct}.pth")
                agent.save(model_snap_filename)
            elif isinstance(agent, RecurrentPPOAgent):
                model_snap_filename = os.path.join(models_dir, f"recurrent_ppo_model_{pct}.pth")
                agent.save(model_snap_filename)
            elif isinstance(agent, DreamerV3Agent):
                model_snap_filename = os.path.join(models_dir, f"dreamer_model_{pct}.pth")
                agent.save(model_snap_filename)
            else:
                model_snap_filename = os.path.join(models_dir, f"q_table_{pct}.npy")
                agent.save(model_snap_filename)
    
    if not quiet:
        print() # Newline after progress bar
    
    training_time = time.time() - start_time
    if not quiet:
        print(f"Training completed in {training_time:.2f} seconds.")
    
    # Final model save
    if isinstance(agent, DQNAgent):
         model_filename = os.path.join(models_dir, "dqn_model_final.pth")
    elif isinstance(agent, DRQNAgent):
         model_filename = os.path.join(models_dir, "drqn_model_final.pth")
    elif isinstance(agent, PPOAgent):
         model_filename = os.path.join(models_dir, "ppo_model_final.pth")
    elif isinstance(agent, RecurrentPPOAgent):
         model_filename = os.path.join(models_dir, "recurrent_ppo_model_final.pth")
    elif isinstance(agent, DreamerV3Agent):
         model_filename = os.path.join(models_dir, "dreamer_model_final.pth")
    else:
         model_filename = os.path.join(models_dir, "q_table.npy")
    agent.save(model_filename)
    
    # Save training history
    import csv
    history_filename = os.path.join(data_dir, "training_history.csv")

    with open(history_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps', 'epsilon'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i], episode_steps[i], episode_epsilons[i]])
    if not quiet:
        print(f"Training history saved to {history_filename}")

    # Generate learning curves
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
    args = parser.parse_args()
    
    # Load default config
    config = get_default_config()
    from src.utils.config import Config
    
    # Load WandB config
    wandb_config_path = "configs/wandb.yaml"
    if os.path.exists(wandb_config_path):
        if not args.quiet:
            print(f"Loading WandB config from {wandb_config_path}")
        wandb_config = Config.load_yaml(wandb_config_path)
        config.merge(wandb_config)
        
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
                with_health=None, 
                max_health=None, 
                start_health=None, 
                health_recovery=None, 
                start_health_random=None,
                danger_prob=None, 
                danger_duration=None, 
                damage_amount=None,
                food_prob=None, 
                food_duration=None, 
                device=args.device, 
                quiet=args.quiet,
                debug=args.debug)
