"""
Training script for the GridWorld Reinforcement Learning agent.

This script:
1. Initializes the `GridWorld` environment and `InteroceptiveBody`.
2. Creates a `QLearningAgent`.
3. Trains the agent for a specified number of episodes.
4. Periodically saves checkpoints and visualizations to `results/`.

Arguments:
- `--episodes <int>`: (Default: 100000) Total number of training episodes.
- `--seed <int>`: (Default: 42) Random seed for reproducibility.

Usage Examples:

1. **Quick Test** (Verify code works):
   ```bash
   python train_rl.py --episodes 100 --seed 42
   ```

2. **Full Training** (Train a robust agent):
   ```bash
   python train_rl.py --episodes 50000 --seed 123
   ```
"""
from src.grid_world_pain import GridWorld
from src.grid_world_pain.body import InteroceptiveBody
from src.grid_world_pain.agent import QLearningAgent
from src.grid_world_pain.visualization import plot_q_table, plot_learning_curves
from src.grid_world_pain.config import get_default_config
import time
import numpy as np
import os
import re
import yaml

import sys
import argparse

def print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps, random_start_satiation, use_homeostatic_reward, satiation_setpoint, testing_seed):
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
    print_section("Environment", env_data)

    # Body (if applicable)
    if with_satiation:
        body_data = {
            "Max Satiation": config_dict.get('body.max_satiation', 20),
            "Start Satiation": config_dict.get('body.start_satiation', 10),
            "Random Start Sat": "ENABLED" if random_start_satiation else "DISABLED",
            "Overeating Death": "ENABLED" if overeating_death else "DISABLED"
        }
        print_section("Body (Internal States)", body_data)

    # Agent
    agent_data = {
        "Algorithm": "Tabular Q-Learning",
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

def train_agent(episodes=100000, seed=42, with_satiation=True, overeating_death=True, food_satiation_gain=10, max_steps=100, random_start_satiation=True, use_homeostatic_reward=False, satiation_setpoint=15, death_penalty=100, testing_seed=42, config_dict=None):
    """
    Trains the Q-learning agent.
    
    Training Loop Logic:
    1. Reset Environment (and Body if with_satiation).
    2. While episode not done:
       a. Agent chooses action based on current state.
       b. External Environment steps.
       c. If with_satiation: Body steps (generates reward).
       d. Else: Environment provides reward (reaching goal).
       e. Agent updates Q-table.
       
    Artifacts Saved:
    - Models: `results/models/q_table_N.npy`
    - Configuration: `results/models/config.yaml`
    """
    import os # Import os here or at top for results_dir handling
    
    # Professional Config Summary
    if config_dict is not None:
        # Update config_dict with resolved values for saving
        # This reflects the exact environment used for training
        config_dict.set('training.training_episode', episodes)
        config_dict.set('training.seed', seed)
        config_dict.set('body.with_satiation', with_satiation)
        config_dict.set('body.use_homeostatic_reward', use_homeostatic_reward)
        config_dict.set('body.satiation_setpoint', satiation_setpoint)
        
        print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps, random_start_satiation, use_homeostatic_reward, satiation_setpoint, testing_seed)
    
    # Setup directories
    results_dir = "results"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # NEW: Save the resolved configuration for evaluation
    if config_dict is not None:
        config_save_path = os.path.join(models_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict.to_dict(), f, default_flow_style=False)
        print(f"Resolved configuration saved to {config_save_path}")

    # Set numpy random seed for determinism
    np.random.seed(seed)

    # Initialize environment, body, and agent
    env = GridWorld(with_satiation=with_satiation, max_steps=max_steps)
    body = InteroceptiveBody(
        overeating_death=overeating_death, 
        random_start_satiation=random_start_satiation, 
        food_satiation_gain=food_satiation_gain,
        use_homeostatic_reward=use_homeostatic_reward,
        satiation_setpoint=satiation_setpoint,
        death_penalty=death_penalty
    )
    
    # We need to inform the agent about expected max_satiation for sizing Q-table
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
            
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env, with_satiation=with_satiation)
    
    print(f"Training agent (with_satiation={with_satiation})...")
    start_time = time.time()
    
    agent.epsilon = 1.0 # Start with full exploration
    decay_rate = 0.9995
    min_epsilon = 0.05
    
    # Define milestones for saving intermediate visualizations
    milestones = {int(episodes * p): int(p * 100) for p in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]}
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        # Reset External
        env_state = env.reset()
        
        if with_satiation:
            # Reset Internal
            body_state = body.reset()
            state = (*env_state, body_state)
        else:
            state = env_state
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(state)
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            
            if with_satiation:
                # Step Internal - Reward comes from survival/metabolism
                next_body_state, reward, body_done = body.step(info)
                done = env_done or body_done
                next_state = (*next_env_state, next_body_state)
            else:
                # Conventional - Reward comes from environment goal
                reward = env_reward
                done = env_done
                next_state = next_env_state
            
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            
        # Decay epsilon
        agent.epsilon = max(min_epsilon, agent.epsilon * decay_rate)
        
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
            
            # Save Q-table snapshot (Only save the model during training loop)
            model_snap_filename = os.path.join(models_dir, f"q_table_{pct}.npy")
            agent.save(model_snap_filename)
    
    print() # Newline after progress bar
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Final model save
    model_filename = os.path.join(models_dir, "q_table.npy")
    agent.save(model_filename)
    
    # Save training history
    import csv
    history_filename = os.path.join(models_dir, "training_history.csv")
    with open(history_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i], episode_steps[i]])
    print(f"Training history saved to {history_filename}")

    # Generate learning curves
    plot_learning_curves(history_filename, os.path.join(results_dir, "plots"), max_steps=max_steps, milestones=milestones)
    
    print("\nTraining complete. Use evaluation.py to generate plots and videos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    parser.add_argument("--no-overeating-death", action="store_true", help="Disable death by overeating")
    args = parser.parse_args()
    
    # Load default config
    config = get_default_config()
    
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
    
    train_agent(episodes=episodes, seed=seed, with_satiation=with_satiation, overeating_death=overeating_death, food_satiation_gain=food_satiation_gain, max_steps=max_steps, random_start_satiation=random_start_satiation, use_homeostatic_reward=use_homeostatic_reward, satiation_setpoint=satiation_setpoint, death_penalty=death_penalty, testing_seed=testing_seed, config_dict=config)
