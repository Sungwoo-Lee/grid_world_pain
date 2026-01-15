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

Usage Examples:

1. **Train DQN**:
   ```bash
   python train.py --agent_config configs/models/dqn.yaml --episodes 1000
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
from src.models.ppo import PPOAgent
from src.environment.sensory import SensorySystem
from src.environment.visualization import plot_q_table, plot_learning_curves
from src.environment.config import get_default_config
import time
import numpy as np
import os
import re
import yaml
import torch
import random

import sys
import argparse


def print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps, random_start_satiation, use_homeostatic_reward, satiation_setpoint, testing_seed,
                         with_health, danger_prob, damage_amount):
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
        "Grid Size": f"{config_dict.get('environment.height', 5)}x{config_dict.get('environment.width', 5)}",
        "Food Position": str(config_dict.get('environment.food_pos', [4, 4])),
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
        
    print_section("Environment", env_data)

    # Body (if applicable)
    if with_satiation:
        body_data = {
            "Max Satiation": config_dict.get('body.max_satiation', 20),
            "Start Satiation": config_dict.get('body.start_satiation', 10),
            "Random Start Sat": "ENABLED" if random_start_satiation else "DISABLED",
            "Overeating Death": "ENABLED" if overeating_death else "DISABLED"
        }
        if with_health:
             body_data["Max Health"] = config_dict.get('body.max_health', 20)
             body_data["Health Recovery"] = config_dict.get('body.health_recovery', 1)
             
        print_section("Body (Internal States)", body_data)

    # Agent
    using_sensory = config_dict.get('sensory.using_sensory', False)
    algorithm = config_dict.get('agent.algorithm', "Tabular Q-Learning")
    
    if algorithm == "DQN":
        agent_data = {
            "Algorithm": "Deep Q-Network (DQN)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Batch Size": 64
        }
        if using_sensory:
             agent_data["Food Radius"] = config_dict.get('sensory.food_radius', 1)
             agent_data["Danger Radius"] = config_dict.get('sensory.danger_radius', 1)
             
    elif algorithm == "PPO":
        agent_data = {
            "Algorithm": "Proximal Policy Optimization (PPO)",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Actor LR": config_dict.get('agent.lr_actor', 0.0003),
            "Critic LR": config_dict.get('agent.lr_critic', 0.001),
            "Gamma (Discount)": config_dict.get('agent.gamma', 0.99),
            "Update Frequency": config_dict.get('agent.update_timestep', 2000)
        }
        if using_sensory:
             agent_data["Food Radius"] = config_dict.get('sensory.food_radius', 1)
             agent_data["Danger Radius"] = config_dict.get('sensory.danger_radius', 1)

    else:
        agent_data = {
            "Algorithm": "Tabular Q-Learning",
            "Sensory Inputs": "Enabled" if using_sensory else "Disabled (Coordinates)",
            "Alpha (Learning Rate)": config_dict.get('agent.alpha', 0.1),
            "Gamma (Discount)": config_dict.get('agent.gamma', 0.99),
            "Min Epsilon": 0.05
        }
    print_section("RL Agent", agent_data)

    # Training
    train_data = {
        "Total Episodes": episodes,
        "Training Seed": seed,
        "Testing Seed": testing_seed
    }
    print_section("Training Schedule", train_data)

    print("\n" + "=" * width + "\n")

def train_agent(episodes=100000, seed=42, with_satiation=True, overeating_death=True, food_satiation_gain=10, max_steps=100, random_start_satiation=True, use_homeostatic_reward=False, satiation_setpoint=15, death_penalty=100, testing_seed=42, config_dict=None,
                with_health=False, max_health=20, start_health=10, health_recovery=1, start_health_random=True,
                danger_prob=0.1, danger_duration=5, damage_amount=5):
    """
    Trains the RL agent (Tabular Q-Learning, DQN, or PPO).
    """
    import os 
    
    # Extract Sensory Config
    using_sensory = config_dict.get('sensory.using_sensory', False)
    food_radius = config_dict.get('sensory.food_radius', 1)
    danger_radius = config_dict.get('sensory.danger_radius', 1)
    
    # Professional Config Summary
    if config_dict is not None:
        # Update config_dict with resolved values for saving
        config_dict.set('training.training_episode', episodes)
        config_dict.set('training.seed', seed)
        config_dict.set('body.with_satiation', with_satiation)
        config_dict.set('body.use_homeostatic_reward', use_homeostatic_reward)
        config_dict.set('body.satiation_setpoint', satiation_setpoint)
        config_dict.set('body.with_health', with_health)
        config_dict.set('environment.danger_prob', danger_prob)
        
        print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps, random_start_satiation, use_homeostatic_reward, satiation_setpoint, testing_seed, with_health, danger_prob, damage_amount)
    
    # Setup directories
    results_dir = "results"
    
    # Determine Model Name
    if using_sensory:
         model_name = "DQN"
    else:
         model_name = "Tabular_Q_Learning"
         
    # Override from agent config if available
    if config.get('agent.algorithm'):
        model_name = config.get('agent.algorithm').replace(" ", "_")
        
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = config.get('tag', 'default') # Pass tag via config/arg
    
    run_name = f"{timestamp}_{tag}"
    
    output_dir = os.path.join(results_dir, model_name, run_name)
    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots") # Ensure plots dir is tracked
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")

    # Save configuration
    if config_dict is not None:
        config_save_path = os.path.join(models_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict.to_dict(), f, default_flow_style=False)
        print(f"Resolved configuration saved to {config_save_path}")

    # Set numpy/torch random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Initialize environment, body
    env = GridWorld(with_satiation=with_satiation, max_steps=max_steps,
                    danger_prob=danger_prob, danger_duration=danger_duration, damage_amount=damage_amount)
    
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
    algorithm = config_dict.get('agent.algorithm', "Tabular Q-Learning")
    
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
        
        print(f"Initializing DQN Agent (Input Dim: {input_dim})...")
        agent = DQNAgent(state_dim=input_dim, action_dim=5)
        
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
                 
        print(f"Initializing PPO Agent (Input Dim: {input_dim})...")
        agent = PPOAgent(
            state_dim=input_dim, 
            action_dim=5,
            lr_actor=config_dict.get('agent.lr_actor', 0.0003),
            lr_critic=config_dict.get('agent.lr_critic', 0.001),
            gamma=config_dict.get('agent.gamma', 0.99),
            K_epochs=config_dict.get('agent.K_epochs', 4),
            eps_clip=config_dict.get('agent.eps_clip', 0.2),
            update_timestep=config_dict.get('agent.update_timestep', 2000)
        )

    else:
        # Tabular (Default)
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
    
    for episode in range(episodes):
        # Reset External
        env_state = env.reset()
        
        # Determine internal start state
        current_agent_pos = env.agent_pos
        current_danger_pos_list = []
        if hasattr(env, 'danger_pos') and env.danger_pos is not None:
             current_danger_pos_list = [env.danger_pos]

        if using_sensory:
             sensory_state = sensory_system.sense(current_agent_pos, env.food_pos, current_danger_pos_list)

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
        if isinstance(agent, (DQNAgent, PPOAgent)):
            flat_state = preprocess_state(state)
        
        while not done:
            if isinstance(agent, (DQNAgent, PPOAgent)):
                action = agent.choose_action(flat_state)
            else:
                action = agent.choose_action(state)
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            
            # Update Observations
            current_agent_pos = env.agent_pos
            current_danger_pos_list = []
            if hasattr(env, 'danger_pos') and env.danger_pos is not None:
                 current_danger_pos_list = [env.danger_pos]

            if using_sensory:
                 next_sensory_state = sensory_system.sense(current_agent_pos, env.food_pos, current_danger_pos_list)
            
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
            
            if isinstance(agent, (DQNAgent, PPOAgent)):
                # DQN/PPO Update
                flat_next_state = preprocess_state(next_state)
                agent.store_transition(flat_state, action, reward, flat_next_state, done)
                agent.update()
                
                state = next_state # Tuple kept for logic
                flat_state = flat_next_state # Flat for next iter
            else:
                # Tabular Update
                agent.update(state, action, reward, next_state)
                state = next_state
                
            total_reward += reward
            steps += 1
            
        # End of Episode
        episode_epsilons.append(agent.epsilon)

        if not isinstance(agent, (DQNAgent, PPOAgent)):
            # Manually decay for tabular
            agent.epsilon = max(tabular_min_epsilon, agent.epsilon * tabular_decay_rate)
        elif isinstance(agent, DQNAgent):
            # DQN handles decay internally in update(), also target net update
            if episode % 10 == 0:
                agent.update_target_network()
        # PPO: No epsilon decay, no target net
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Progress Bar
        if (episode + 1) % 100 == 0:
            progress = (episode + 1) / episodes
            bar_length = 40
            block = int(round(bar_length * progress))
            text = "\rProgress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
            
        # Check milestones
        if (episode + 1) in milestones:
            pct = milestones[episode + 1]
            
            if isinstance(agent, DQNAgent):
                model_snap_filename = os.path.join(models_dir, f"dqn_model_{pct}.pth")
                agent.save(model_snap_filename)
            elif isinstance(agent, PPOAgent):
                model_snap_filename = os.path.join(models_dir, f"ppo_model_{pct}.pth")
                agent.save(model_snap_filename)
            else:
                model_snap_filename = os.path.join(models_dir, f"q_table_{pct}.npy")
                agent.save(model_snap_filename)
    
    print() # Newline after progress bar
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Final model save
    if isinstance(agent, DQNAgent):
         model_filename = os.path.join(models_dir, "dqn_model_final.pth")
    elif isinstance(agent, PPOAgent):
         model_filename = os.path.join(models_dir, "ppo_model_final.pth")
    else:
         model_filename = os.path.join(models_dir, "q_table.npy")
    agent.save(model_filename)
    
    # Save training history
    import csv
    history_filename = os.path.join(models_dir, "training_history.csv")
    with open(history_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps', 'epsilon'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i], episode_steps[i], episode_epsilons[i]])
    print(f"Training history saved to {history_filename}")

    # Generate learning curves
    plot_learning_curves(history_filename, plots_dir, max_steps=max_steps, milestones=milestones)
    
    print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to base config YAML")
    parser.add_argument("--agent_config", type=str, default="configs/models/dqn.yaml", help="Path to agent config YAML")
    parser.add_argument("--tag", type=str, default="default", help="Tag for the training run")
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    parser.add_argument("--no-overeating-death", action="store_true", help="Disable death by overeating")
    args = parser.parse_args()
    
    # Load default config
    config = get_default_config()
    
    # Load and merge agent config
    from src.environment.config import Config
    if args.agent_config:
        print(f"Loading agent config from: {args.agent_config}")
        agent_config = Config.load_yaml(args.agent_config)
        config.merge(agent_config)
    
    # Set tag
    config.set('tag', args.tag)
        
    # Overrides
    episodes = args.episodes or config.get('training.training_episode', 100000)
    seed = args.seed or config.get('training.seed', 42)
    with_satiation = config.get('body.with_satiation', True)
    if args.no_satiation:
        with_satiation = False
        
    overeating_death = config.get('body.overeating_death', True)
    if args.no_overeating_death:
        overeating_death = False
        
    max_steps = config.get('environment.max_steps', 100)
    random_start_satiation = config.get('body.random_start_satiation', True)
    food_satiation_gain = config.get('body.food_satiation_gain', 10)
    use_homeostatic_reward = config.get('body.use_homeostatic_reward', False)
    satiation_setpoint = config.get('body.satiation_setpoint', 15)
    death_penalty = config.get('body.death_penalty', 100)
    testing_seed = config.get('testing.seed', 42)
    
    # Health / Pain Params
    with_health = config.get('body.with_health', False)
    max_health = config.get('body.max_health', 20)
    start_health = config.get('body.start_health', 10)
    health_recovery = config.get('body.health_recovery', 1)
    start_health_random = config.get('body.start_health_random', True)
    
    danger_prob = config.get('environment.danger_prob', 0.1)
    danger_duration = config.get('environment.danger_duration', 5)
    damage_amount = config.get('environment.damage_amount', 5)
    
    train_agent(episodes=episodes, seed=seed, with_satiation=with_satiation, overeating_death=overeating_death, food_satiation_gain=food_satiation_gain, max_steps=max_steps, random_start_satiation=random_start_satiation, use_homeostatic_reward=use_homeostatic_reward, satiation_setpoint=satiation_setpoint, death_penalty=death_penalty, testing_seed=testing_seed, config_dict=config,
                with_health=with_health, max_health=max_health, start_health=start_health, health_recovery=health_recovery, start_health_random=start_health_random,
                danger_prob=danger_prob, danger_duration=danger_duration, damage_amount=damage_amount)
