"""
Debugging Script / Sandbox.

Purpose:
- To verify the Environmental Mechanics (Grid + Body) without any Learning Agent.
- Uses a Random Agent to simply walk around.
- Checks if "Eating" works, if "Satiation" changes, and if "Death" occurs correctly.

Arguments:
- `--episodes <int>`: (Default: 3) Number of episodes to record in the video.
- `--max_steps <int>`: (Default: 30) Maximum steps to record per episode.

Usage Examples:

1. **Short Video** (Default):
   ```bash
   python main.py
   ```

2. **Longer Observation**:
   ```bash
   python main.py --episodes 5 --max_steps 50
   ```
"""

import os
import argparse
import time
import datetime
import imageio
import numpy as np

from src.grid_world_pain import GridWorld
from src.grid_world_pain.body import InteroceptiveBody

def main():
    parser = argparse.ArgumentParser(description="GridWorld Debug Sandbox")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record in video (default: 3)")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum steps to record per episode (default: 30)")
    args = parser.parse_args()

    # Setup results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set numpy random seed for determinism
    np.random.seed(42)
    
    video_filename = os.path.join(results_dir, "gridworld_debug.mp4")
    
    print(f"Starting Debug Session")
    print(f"Video will be saved to: {video_filename}")

    # Initialize environment and body
    env = GridWorld()
    body = InteroceptiveBody()
    
    frames = []
    
    num_episodes = args.episodes
    max_steps_per_episode = args.max_steps
    
    for episode in range(num_episodes):
        ep_num = episode + 1
        print(f"\n--- Starting Episode {ep_num}/{num_episodes} ---")
        
        # Reset for new episode
        env.reset()
        satiation = body.reset()
        
        # Capture initial state
        print("Start State:")
        print(f"Satiation: {satiation}/{body.max_satiation}")
        print(f"Agent Pos: {env.agent_pos}")
        
        frames.append(env.render_rgb_array(
            satiation=body.satiation, 
            max_satiation=body.max_satiation,
            episode=ep_num,
            step=0
        ))
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            # Random action
            action = np.random.randint(0, 4)
            action_name = ["Up", "Right", "Down", "Left"][action]
            
            print(f"Step {step_count + 1}: Action {action_name}")
            
            # Step environment
            _, _, _, info = env.step(action)
            
            # Step body
            satiation, reward, done = body.step(info)
            
            # Log details
            print(f"  Info: {info}")
            print(f"  Satiation: {satiation}/{body.max_satiation}")
            print(f"  Reward: {reward}, Done: {done}")
            
            # Capture frame
            frames.append(env.render_rgb_array(
                satiation=body.satiation, 
                max_satiation=body.max_satiation,
                episode=ep_num,
                step=step_count + 1
            ))
            
            step_count += 1
            
            if done:
                print("Episode Ended (Starved or Overfed).")
                # Add pause at the end of episode
                for _ in range(5):
                    frames.append(env.render_rgb_array(
                        satiation=body.satiation, 
                        max_satiation=body.max_satiation,
                        episode=ep_num,
                        step=step_count
                    ))
                break
        
        if not done:
             print("Episode Ended (Max Steps Reached).")
             
    # Save video
    print(f"\nSaving video with {len(frames)} frames...")
    imageio.mimsave(video_filename, frames, fps=5)
    print(f"Video saved successfully: {video_filename}")

if __name__ == "__main__":
    main()
