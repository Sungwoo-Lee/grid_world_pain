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
from src.grid_world_pain.visualization import plot_q_table, run_and_save_episode, generate_artifacts_from_checkpoint
from src.grid_world_pain.config import get_default_config
import time
import numpy as np

import sys
import argparse

def print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps):
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
    print_section("Environment", env_data)

    # Body (if applicable)
    if with_satiation:
        body_data = {
            "Max Satiation": config_dict.get('body.max_satiation', 20),
            "Start Satiation": config_dict.get('body.start_satiation', 10),
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
        "Random Seed": seed
    }
    print_section("Training Schedule", train_data)

    print("\n" + "=" * width + "\n")

def train_and_visualize(episodes=100000, seed=42, with_satiation=True, overeating_death=True, max_steps=100, config_dict=None):
    """
    Trains the Q-learning agent and visualizes the result.
    
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
    - Plots: `results/plots/q_table_vis_N.png`
    """
    import os # Import os here or at top for results_dir handling
    
    # NEW: Professional Config Summary
    if config_dict is not None:
        print_config_summary(config_dict, episodes, seed, with_satiation, overeating_death, max_steps)
    
    # Set numpy random seed for determinism
    np.random.seed(seed)
    
    results_dir = "results"
    models_dir = os.path.join(results_dir, "models")
    plots_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize environment, body, and agent
    env = GridWorld(with_satiation=with_satiation, max_steps=max_steps)
    body = InteroceptiveBody(overeating_death=overeating_death)
    
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
    milestones = {int(episodes * p): int(p * 100) for p in [0.1, 0.25, 0.5, 0.75, 1.0]}
    
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
        
        while not done:
            action = agent.choose_action(state)
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            
            if with_satiation:
                # Step Internal - Reward comes from survival/metabolism
                next_body_state, reward, done = body.step(info)
                next_state = (*next_env_state, next_body_state)
            else:
                # Conventional - Reward comes from environment goal
                reward = env_reward
                done = env_done
                next_state = next_env_state
            
            agent.update(state, action, reward, next_state)
            state = next_state
            
        # Decay epsilon
        agent.epsilon = max(min_epsilon, agent.epsilon * decay_rate)
        
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
    
    # Save model
    model_filename = os.path.join(models_dir, "q_table.npy")
    agent.save(model_filename)
    
    # Post-Training: Generate all visualizations and videos from saved checkpoints
    print("\nGenerating all artifact checkpoints (plots & videos)...")
    import glob
    checkpoints = glob.glob(os.path.join(models_dir, "q_table_*.npy"))
    
    # Sort checkpoints by percentage
    import re
    def extract_number(path):
        match = re.search(r"q_table_(\d+).npy", path)
        return int(match.group(1)) if match else -1
    checkpoints.sort(key=extract_number)
    
    # Process each milestone
    for checkpoint in checkpoints:
        generate_artifacts_from_checkpoint(checkpoint, results_dir, max_steps=max_steps, with_satiation=with_satiation, food_pos=env.food_pos)
    
    # Process final model
    generate_artifacts_from_checkpoint(model_filename, results_dir, max_steps=max_steps, with_satiation=with_satiation, food_pos=env.food_pos)

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
    episodes = args.episodes or config.get('training.default_episodes', 100000)
    seed = args.seed or config.get('training.seed', 42)
    with_satiation = config.get('environment.with_satiation', True)
    if args.no_satiation:
        with_satiation = False
        
    overeating_death = config.get('body.overeating_death', True)
    if args.no_overeating_death:
        overeating_death = False
        
    max_steps = config.get('environment.max_steps', 100)
    
    train_and_visualize(episodes=episodes, seed=seed, with_satiation=with_satiation, overeating_death=overeating_death, max_steps=max_steps, config_dict=config)
