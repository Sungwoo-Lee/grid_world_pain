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
from src.grid_world_pain.config import get_default_config

def plot_q_table(q_table, save_path, food_pos=None):
    """
    Visualizes the Q-table policy and values at different satiation levels.
    """
    # Dispatch to conventional plotter if q_table is 3D
    if len(q_table.shape) == 3:
        return plot_q_table_conventional(q_table, save_path, food_pos)

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
        
        # Normalize Value function for visualization (0 to 1 range across displayed slices)
        # We find global max/min across the slices being visualized for consistent color mapping
        v_min, v_max = np.min(v_values), np.max(v_values)
        if v_max > v_min:
            v_norm = (v_values - v_min) / (v_max - v_min)
        else:
            v_norm = np.zeros_like(v_values)
            
        # Plot Heatmap of Normalized Value
        cax = ax.imshow(v_norm, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        
        # Add simpler title
        labels = ["Low", "Mid", "High"]
        ax.set_title(f"{labels[idx]} Satiation (Sat={sat_level})")
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        
        # Overlay Arrows for Policy
        for r in range(height):
            for c in range(width):
                action = policy[r, c]
                # arrow_char = ['↑', '→', '↓', '←'][action]
                arrow_char = ['\u2191', '\u2192', '\u2193', '\u2190'][action] # Use unicode arrows
                ax.text(c, r, arrow_char, ha='center', va='center', color='white', fontsize=12, weight='bold')
        
        # Mark Food Location
        if food_pos is not None:
            fr, fc = food_pos
            ax.text(fc, fr, 'F', ha='center', va='center', color='lime', fontsize=20, weight='bold', path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
                
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cax, cax=cbar_ax, label='Max Q-Value (Normalized)')
    
    plt.suptitle("Learned Policy & Value at Different Satiation Levels", fontsize=16)
    plt.savefig(save_path)
    plt.close(fig)

def plot_q_table_conventional(q_table, save_path, food_pos=None):
    """
    Visualizes a 3D Q-table (Conventional Mode).
    """
    height, width, _ = q_table.shape
    fig, ax = plt.subplots(figsize=(7, 6))
    
    v_values = np.max(q_table, axis=2)
    policy = np.argmax(q_table, axis=2)
    
    # Normalize for visualization
    v_min, v_max = np.min(v_values), np.max(v_values)
    if v_max > v_min:
        v_norm = (v_values - v_min) / (v_max - v_min)
    else:
        v_norm = np.zeros_like(v_values)
        
    cax = ax.imshow(v_norm, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title("Learned Policy & Value (Conventional Mode)")
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    
    for r in range(height):
        for c in range(width):
            action = policy[r, c]
            arrow_char = ['\u2191', '\u2192', '\u2193', '\u2190'][action]
            ax.text(c, r, arrow_char, ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    if food_pos is not None:
        fr, fc = food_pos
        ax.text(fc, fr, 'G', ha='center', va='center', color='lime', fontsize=20, weight='bold', path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
        
    fig.colorbar(cax, ax=ax, label='Max Q-Value (Normalized)')
    plt.savefig(save_path)
    plt.close(fig)


def run_and_save_episode(env, body, agent, output_path, max_steps=50, num_episodes=1, with_satiation=True, verbose=True):
    """
    Helper to run episodes and save video.
    """
    frames = []
    
    for ep in range(num_episodes):
        if verbose:
            print(f"Recording Episode {ep+1}/{num_episodes}...")
        
        env_state = env.reset()
        if with_satiation:
            body_state = body.reset()
            state = (*env_state, body_state)
            frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep+1, step=0))
        else:
            state = env_state
            frames.append(env.render_rgb_array(episode=ep+1, step=0))
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            action = agent.choose_action(state)
            
            next_env_state, _, env_done, info = env.step(action)
            
            if with_satiation:
                next_body_state, reward, done = body.step(info)
                next_state = (*next_env_state, next_body_state)
                frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep+1, step=step_count+1))
            else:
                reward = 0 # Not used for visualization
                done = env_done
                next_state = next_env_state
                frames.append(env.render_rgb_array(episode=ep+1, step=step_count+1))
            
            state = next_state
            step_count += 1
            
            if done:
                # End frames
                for _ in range(5):
                    if with_satiation:
                        frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep+1, step=step_count))
                    else:
                        frames.append(env.render_rgb_array(episode=ep+1, step=step_count))
                break
                
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=5)
    if verbose:
        print(f"Saved video to {output_path}")

def generate_video_from_checkpoint(checkpoint_path, results_dir, max_steps=50, num_episodes=1, with_satiation=True):
    """
    Generates a video for a given Q-table checkpoint.
    """
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"q_table_(\d+).npy", filename)
    if not match:
        pct = "final" if filename == "q_table.npy" else "unknown"
    else:
        pct = match.group(1)
        
    print(f"Generating video for checkpoint: {filename} ({pct}%)")

    env = GridWorld(with_satiation=with_satiation)
    body = InteroceptiveBody()
    
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
    
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env, with_satiation=with_satiation)
    
    try:
        agent.load(checkpoint_path)
    except Exception as e:
        print(f"Failed to load {checkpoint_path}: {e}")
        return

    agent.epsilon = 0
    np.random.seed(42) 
    
    output_path = os.path.join(results_dir, "videos", f"video_{pct}.mp4")
    run_and_save_episode(env, body, agent, output_path, max_steps=max_steps, num_episodes=num_episodes, with_satiation=with_satiation)

def generate_visuals_from_checkpoint(checkpoint_path, results_dir, config=None):
    """
    Generates both Q-table plot and performance video for a given checkpoint.
    """
    if config is None:
        from src.grid_world_pain.config import get_default_config
        config = get_default_config()

    filename = os.path.basename(checkpoint_path)
    match = re.search(r"q_table_(\d+).npy", filename)
    if not match:
        pct = "final" if filename == "q_table.npy" else "unknown"
    else:
        pct = match.group(1)
        
    print(f"Processing checkpoint: {filename} ({pct}%)")
    
    # 1. Configuration extraction
    with_satiation = config.get('body.with_satiation', True)
    overeating_death = config.get('body.overeating_death', True)
    max_steps = config.get('environment.max_steps', 100)
    seed = config.get('testing.seed', 42)
    food_pos = config.get('environment.food_pos', [4, 4])
    height = config.get('environment.height', 5)
    width = config.get('environment.width', 5)
    max_satiation = config.get('body.max_satiation', 20)
    start_satiation = config.get('body.start_satiation', 10)
    random_start_satiation = config.get('body.random_start_satiation', True)
    
    # Set seed for deterministic evaluation
    np.random.seed(seed)
    
    # 2. Load Agent
    env = GridWorld(height=height, width=width, food_pos=food_pos, with_satiation=with_satiation, max_steps=max_steps)
    body = InteroceptiveBody(max_satiation=max_satiation, start_satiation=start_satiation, overeating_death=overeating_death, random_start_satiation=random_start_satiation)
    
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
    
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env, with_satiation=with_satiation)
    
    try:
        agent.load(checkpoint_path)
    except Exception as e:
        print(f"Failed to load {checkpoint_path}: {e}")
        return

    # 3. Generate Plot
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    vis_filename = os.path.join(plots_dir, f"q_table_vis_{pct}.png" if pct != "final" else "q_table_vis.png")
    plot_q_table(agent.q_table, vis_filename, food_pos)
    
    # 4. Generate Video
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_filename = os.path.join(videos_dir, f"video_{pct}.mp4" if pct != "final" else "final_trained_agent.mp4")
    
    agent.epsilon = 0
    run_and_save_episode(env, body, agent, video_filename, max_steps=max_steps, num_episodes=1, with_satiation=with_satiation, verbose=False)

def run_random_demo(videos_dir, max_steps=50, num_episodes=1, with_satiation=True):
    """
    Runs a random agent demo.
    """
    print("Running Random Agent Demo...")
    env = GridWorld(with_satiation=with_satiation)
    body = InteroceptiveBody()
    
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
            
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env, with_satiation=with_satiation)
    agent.epsilon = 1.0
    
    np.random.seed(None)
    
    output_path = os.path.join(videos_dir, "gridworld_demo.mp4")
    run_and_save_episode(env, body, agent, output_path, max_steps=max_steps, num_episodes=num_episodes, with_satiation=with_satiation)

def main():
    parser = argparse.ArgumentParser(description="GridWorld Video Generator")
    parser.add_argument("--demo", action="store_true", help="Run a random agent demo instead of processing checkpoints.")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per episode (default: 50)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record (default: 1)")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    
    args = parser.parse_args()

    # Load default config
    config = get_default_config()
    
    # Overrides
    with_satiation = config.get('environment.with_satiation', True)
    if args.no_satiation:
        with_satiation = False

    results_dir = "results"
    
    if args.demo:
        videos_dir = os.path.join(results_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        run_random_demo(videos_dir, max_steps=args.max_steps, num_episodes=args.episodes, with_satiation=with_satiation)
        return

    models_dir = os.path.join(results_dir, "models")
    videos_dir = os.path.join(results_dir, "videos")
    
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found. Run train_rl.py first.")
        return
        
    os.makedirs(videos_dir, exist_ok=True)

    checkpoints = glob.glob(os.path.join(models_dir, "q_table_*.npy"))
    
    if not checkpoints:
        print(f"No checkpoints found in {models_dir}. Run train_rl.py first.")
        return
        
    def extract_number(path):
        match = re.search(r"q_table_(\d+).npy", path)
        return int(match.group(1)) if match else -1
        
    checkpoints.sort(key=extract_number)
    
    for checkpoint in checkpoints:
        generate_visuals_from_checkpoint(checkpoint, results_dir, config=config)
    
    # Also handle the final model if it exists
    final_model = os.path.join(models_dir, "q_table.npy")
    if os.path.exists(final_model):
        generate_visuals_from_checkpoint(final_model, results_dir, config=config)

if __name__ == "__main__":
    main()
