"""
Visualization utilities for GridWorld Pain environment.
Includes Q-table plotting and video generation logic.

This module helps "peek" into the agent's brain:
1. plot_q_table: Shows what the agent thinks is the best action at different hunger levels.
2. generate_video_from_checkpoint: Replays learned behavior to confirm it works.

Arguments:
- `--demo`: Run a random agent demo instead of loading a checkpoint.
- `--max_steps <int>`: (Default: 50) Limit the number of steps per episode. Useful for debugging specific behaviors or long-term survival.
- `--episodes <int>`: (Default: 1) Number of episodes to record in a single video.

Usage Examples:

1. **Random Agent Demo** (2 episodes, max 30 steps):
   ```bash
   python -m src.grid_world_pain.visualization --demo --episodes 2 --max_steps 30
   ```

2. **Checkpoint Replay** (Default settings: 1 episode, 50 steps):
   ```bash
   python -m src.grid_world_pain.visualization
   ```

3. **Long Replay** (Watch a trained agent survive for 100 steps):
   ```bash
   python -m src.grid_world_pain.visualization --max_steps 100
   ```
"""

import os
import re
import glob
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from src.grid_world_pain.grid_world import GridWorld
from src.grid_world_pain.body import InteroceptiveBody
from src.grid_world_pain.agent import QLearningAgent

def plot_q_table(q_table, save_path, food_pos=None):
    """
    Visualizes the Q-table policy and values at different satiation levels.
    
    Why visualization?
    - The Q-table is 4D (Height, Width, Satiation, Action). We can't see 4D.
    - So we take "Slices" at specific Satiation levels:
      1. Low Satiation (Hungry): Agent should value Food.
      2. Mid Satiation: Transition zone.
      3. High Satiation (Full): Agent should avoid Food (to avoid bursting).
      
    This plot shows 3 heatmaps (one for each level) with arrows indicating the policy.
    """
    # q_table shape: (height, width, satiation_dim, actions)
    height, width, sat_dim, _ = q_table.shape
    
    # Define representative slices to visualize
    max_satiation = sat_dim - 2
    
    slices = [
        max_satiation // 4,       # Low Satiation (Hungry)
        max_satiation // 2,       # Mid Satiation
        int(max_satiation * 0.9)  # High Satiation (Full)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    action_deltas = {
        0: (-1, 0), # Up
        1: (0, 1),  # Right
        2: (1, 0),  # Down
        3: (0, -1)  # Left
    }
    
    for idx, sat_level in enumerate(slices):
        ax = axes[idx]
        
        # Extract 2D Q-table for this satiation
        q_slice = q_table[:, :, sat_level, :]
        
        # Value function: Max Q over actions
        v_values = np.max(q_slice, axis=2)
        
        # Policy: Argmax Q
        policy = np.argmax(q_slice, axis=2)
        
        # Plot Heatmap of Value
        cax = ax.imshow(v_values, cmap='viridis', interpolation='nearest')
        
        # Add simpler title
        labels = ["Low", "Mid", "High"]
        ax.set_title(f"{labels[idx]} Satiation (Sat={sat_level})")
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        
        # Overlay Arrows for Policy
        for r in range(height):
            for c in range(width):
                action = policy[r, c]
                dy, dx = action_deltas[action]
                
                arrow_char = ['↑', '→', '↓', '←'][action]
                ax.text(c, r, arrow_char, ha='center', va='center', color='white', fontsize=12, weight='bold')
        
        # Mark Food Location
        if food_pos is not None:
            fr, fc = food_pos
            ax.text(fc, fr, 'F', ha='center', va='center', color='lime', fontsize=20, weight='bold', path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
                
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cax, cax=cbar_ax, label='Max Q-Value')
    
    plt.suptitle("Learned Policy & Value at Different Satiation Levels", fontsize=16)
    plt.savefig(save_path)
    print(f"Q-table visualization saved to {save_path}")
    plt.close(fig)

def run_and_save_episode(env, body, agent, output_path, max_steps=50, num_episodes=1):
    """
    Helper to run episodes and save video.
    
    Args:
        env: The GridWorld environment.
        body: The InteroceptiveBody.
        agent: The Agent.
        output_path (str): Path to save .mp4.
        max_steps (int): Maximum steps per episode.
        num_episodes (int): Number of episodes to record in the video.
    """
    frames = []
    
    for ep in range(num_episodes):
        print(f"Recording Episode {ep+1}/{num_episodes}...")
        
        env_state = env.reset()
        body_state = body.reset()
        state = (*env_state, body_state)
        
        # Initial frame
        frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep+1, step=0))
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            action = agent.choose_action(state)
            
            next_env_state, _, _, info = env.step(action)
            next_body_state, reward, done = body.step(info)
            next_state = (*next_env_state, next_body_state)
            
            frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep+1, step=step_count+1))
            
            state = next_state
            step_count += 1
            
            if done:
                # End frames (pause on death)
                for _ in range(5):
                    frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep+1, step=step_count))
                break
                
    # Save video
    # Ensure dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    imageio.mimsave(output_path, frames, fps=5)
    print(f"Saved video to {output_path}")

def generate_video_from_checkpoint(checkpoint_path, results_dir, max_steps=50, num_episodes=1):
    """
    Generates a video for a given Q-table checkpoint.
    """
    # Extract percentage from filename (e.g., q_table_10.npy -> 10)
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"q_table_(\d+).npy", filename)
    if not match:
        if filename == "q_table.npy":
            pct = "final"
        else:
            print(f"Skipping {filename}: Does not match pattern q_table_N.npy")
            return
    else:
        pct = match.group(1)
        
    print(f"Generating video for checkpoint: {filename} ({pct}%)")

    # Initialize environment
    env = GridWorld()
    body = InteroceptiveBody()
    
    # Mock composite env for agent init
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
    
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env)
    
    # Load Q-table
    try:
        agent.load(checkpoint_path)
    except Exception as e:
        print(f"Failed to load {checkpoint_path}: {e}")
        return

    # Deterministic run
    agent.epsilon = 0
    np.random.seed(42) 
    
    output_path = os.path.join(results_dir, "videos", f"video_{pct}.mp4")
    run_and_save_episode(env, body, agent, output_path, max_steps=max_steps, num_episodes=num_episodes)

def run_random_demo(videos_dir, max_steps=50, num_episodes=1):
    """
    Runs a random agent demo.
    """
    print("Running Random Agent Demo...")
    env = GridWorld()
    body = InteroceptiveBody()
    
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
            
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env)
    agent.epsilon = 1.0
    
    np.random.seed(None) # Random seed for demo
    
    output_path = os.path.join(videos_dir, "gridworld_demo.mp4")
    run_and_save_episode(env, body, agent, output_path, max_steps=max_steps, num_episodes=num_episodes)

def main():
    parser = argparse.ArgumentParser(description="GridWorld Video Generator")
    parser.add_argument("--demo", action="store_true", help="Run a random agent demo instead of processing checkpoints.")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per episode (default: 50)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record (default: 1)")
    
    args = parser.parse_args()

    results_dir = "results"
    
    if args.demo:
        videos_dir = os.path.join(results_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        run_random_demo(videos_dir, max_steps=args.max_steps, num_episodes=args.episodes)
        return

    models_dir = os.path.join(results_dir, "models")
    videos_dir = os.path.join(results_dir, "videos")
    
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found. Run train_rl.py first.")
        return
        
    os.makedirs(videos_dir, exist_ok=True)

    # Find all q_table_*.npy files
    checkpoints = glob.glob(os.path.join(models_dir, "q_table_*.npy"))
    
    if not checkpoints:
        print(f"No checkpoints found in {models_dir}. Run train_rl.py first.")
        return
        
    # Sort by number to process in order
    def extract_number(path):
        match = re.search(r"q_table_(\d+).npy", path)
        return int(match.group(1)) if match else -1
        
    checkpoints.sort(key=extract_number)
    
    for checkpoint in checkpoints:
        generate_video_from_checkpoint(checkpoint, results_dir, max_steps=args.max_steps, num_episodes=args.episodes)

if __name__ == "__main__":
    main()
