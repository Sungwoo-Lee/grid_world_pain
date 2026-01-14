"""
Training script for the GridWorld Reinforcement Learning agent.

This script:
1. Initializes the `GridWorld` environment and `InteroceptiveBody`.
2. Creates a `QLearningAgent`.
3. Trains the agent for a specified number of episodes.
4. Periodically saves checkpoints and visualizations to `results/`.

Arguments:
- `--episodes <int>`: (Default: 100000) Total number of training episodes.

Usage Examples:

1. **Quick Test** (Verify code works):
   ```bash
   python train_rl.py --episodes 100
   ```

2. **Full Training** (Train a robust agent):
   ```bash
   python train_rl.py --episodes 50000
   ```
"""
from src.grid_world_pain import GridWorld
from src.grid_world_pain.body import InteroceptiveBody
from src.grid_world_pain.agent import QLearningAgent
from src.grid_world_pain.visualization import plot_q_table
import time
import numpy as np

import sys
import argparse

def train_and_visualize(episodes=100000):
    """
    Trains the Q-learning agent and visualizes the result.
    
    Training Loop Logic:
    1. Reset Environment and Body.
    2. While episode not done:
       a. Agent chooses action based on current state (Env pos + Body satiation).
       b. External Environment steps (moves agent, checks for food).
       c. Internal Body steps (metabolizes, checks eating, generates reward).
       d. Agent updates Q-table based on reward and new state.
       
    Artifacts Saved:
    - Models: `results/models/q_table_N.npy`
    - Plots: `results/plots/q_table_vis_N.png`
    """
    import os # Import os here or at top for results_dir handling
    
    # Set numpy random seed for determinism
    np.random.seed(42)
    
    results_dir = "results"
    models_dir = os.path.join(results_dir, "models")
    plots_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize environment, body, and agent
    env = GridWorld()
    body = InteroceptiveBody()
    
    # We need to inform the agent about expected max_satiation for sizing Q-table
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
            
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env)
    
    print("Training agent...")
    start_time = time.time()
    
    agent.epsilon = 1.0 # Start with full exploration
    decay_rate = 0.9995
    min_epsilon = 0.05
    
    # Define milestones for saving intermediate visualizations
    milestones = {int(episodes * p): int(p * 100) for p in [0.1, 0.25, 0.5, 0.75, 1.0]}
    
    for episode in range(episodes):
        # Reset both
        env_state = env.reset()
        body_state = body.reset()
        state = (*env_state, body_state)
        
        done = False
        
        while not done:
            action = agent.choose_action(state)
            
            # Step External
            next_env_state, _, _, info = env.step(action)
            
            # Step Internal
            next_body_state, reward, done = body.step(info)
            
            next_state = (*next_env_state, next_body_state)
            
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
            sys.stdout.write(text)
            sys.stdout.flush()
            
        # Check milestones
        if (episode + 1) in milestones:
            pct = milestones[episode + 1]
            if not os.path.exists("results"):
                 os.makedirs("results", exist_ok=True)
            
            vis_filename = os.path.join(plots_dir, f"q_table_vis_{pct}.png")
            plot_q_table(agent.q_table, vis_filename, env.food_pos)
            
            # Save Q-table snapshot
            model_snap_filename = os.path.join(models_dir, f"q_table_{pct}.npy")
            agent.save(model_snap_filename)
    
    print() # Newline after progress bar
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Save model
    model_filename = os.path.join(models_dir, "q_table.npy")
    agent.save(model_filename)
    
    # Also save as q_table_100.npy for consistency with generate_videos
    agent.save(os.path.join(models_dir, "q_table_100.npy"))
    
    # Final visualization
    vis_filename = os.path.join(plots_dir, "q_table_vis.png")
    plot_q_table(agent.q_table, vis_filename, env.food_pos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of episodes to train")
    args = parser.parse_args()
    
    train_and_visualize(episodes=args.episodes)
