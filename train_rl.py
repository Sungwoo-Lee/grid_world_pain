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
from src.grid_world_pain.visualization import plot_q_table, run_and_save_episode
from src.grid_world_pain.config import get_default_config
import time
import numpy as np

import sys
import argparse

def train_and_visualize(episodes=100000, seed=42, with_satiation=True, overeating_death=True, max_steps=100):
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
            
            vis_filename = os.path.join(plots_dir, f"q_table_vis_{pct}.png")
            plot_q_table(agent.q_table, vis_filename, env.food_pos)
            
            # Save Q-table snapshot
            model_snap_filename = os.path.join(models_dir, f"q_table_{pct}.npy")
            agent.save(model_snap_filename)
            
            # NEW: Generate milestone video
            video_dir = os.path.join(results_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            milestone_video_filename = os.path.join(video_dir, f"video_{pct}.mp4")
            
            # Temporary disable epsilon for recording
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            run_and_save_episode(env, body, agent, milestone_video_filename, max_steps=max_steps, num_episodes=1, with_satiation=with_satiation, verbose=False)
            agent.epsilon = original_epsilon
    
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
    
    # Final video generation is now handled by the 100% milestone
    # But for clarity, we can ensure results/videos/final_trained_agent.mp4 exists as a symlink or copy if needed.
    # For now, milestone video_100.mp4 is the final one.
    video_dir = os.path.join(results_dir, "videos")
    final_src = os.path.join(video_dir, "video_100.mp4")
    final_dst = os.path.join(video_dir, "final_trained_agent.mp4")
    if os.path.exists(final_src):
        import shutil
        shutil.copy(final_src, final_dst)

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
    
    train_and_visualize(episodes=episodes, seed=seed, with_satiation=with_satiation, overeating_death=overeating_death, max_steps=max_steps)
