"""
Debugging Script / Sandbox.

Purpose:
- To verify the Environmental Mechanics (Grid + Body) without any Learning Agent.
- Uses a Random Agent to simply walk around.
- Checks if "Eating" works, if "Satiation" changes, and if "Death" occurs correctly.

Arguments:
- `--episodes <int>`: (Default: 3) Number of episodes to record in the video.
- `--max_steps <int>`: (Default: 30) Maximum steps to record per episode.
- `--seed <int>`: (Default: 42) Random seed for reproducibility.

Usage Examples:

1. **Short Video** (Default):
   ```bash
   python main.py
   ```

2. **Longer Observation**:
   ```bash
   python main.py --episodes 5 --max_steps 50 --seed 123
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
from src.grid_world_pain.agent import QLearningAgent
from src.grid_world_pain.visualization import save_video
from src.grid_world_pain.config import get_default_config

def main():
    parser = argparse.ArgumentParser(description="GridWorld Debug Sandbox")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record in video (default: 3)")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum steps to record per episode (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    parser.add_argument("--no-overeating-death", action="store_true", help="Disable death by overeating")
    args = parser.parse_args()

    # Load default config
    config = get_default_config()
    
    # Overrides
    with_satiation = config.get('body.with_satiation', True)
    if args.no_satiation:
        with_satiation = False

    overeating_death = config.get('body.overeating_death', True)
    if args.no_overeating_death:
        overeating_death = False
        
    max_steps = config.get('environment.max_steps', 100)
    random_start_satiation = config.get('body.random_start_satiation', True)
    
    height = config.get('environment.height', 5)
    width = config.get('environment.width', 5)
    food_pos = config.get('environment.food_pos', [4, 4])
    start_pos = config.get('environment.start_pos', [0, 0])
    max_satiation = config.get('body.max_satiation', 20)
    start_satiation = config.get('body.start_satiation', 10)

    # Setup results directory
    results_dir = "results"
    video_dir = os.path.join(results_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Set numpy random seed for determinism
    np.random.seed(args.seed)
    
    video_filename = os.path.join(video_dir, "gridworld_debug.mp4")
    
    print(f"Starting Debug Session")
    print(f"Video will be saved to: {video_filename}")

    # Initialize components
    env = GridWorld(height=height, width=width, start=start_pos, food_pos=food_pos, with_satiation=with_satiation, max_steps=max_steps)
    body = InteroceptiveBody(max_satiation=max_satiation, start_satiation=start_satiation, overeating_death=overeating_death, random_start_satiation=random_start_satiation)
    
    # Mock composite env for agent init
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
            
    composite_env = CompositeEnv(env, body)
    agent = QLearningAgent(composite_env, with_satiation=with_satiation)
    agent.epsilon = 1.0 # Force random behavior for debug sandbox
    
    frames = []
    
    num_episodes = args.episodes
    max_steps_per_episode = args.max_steps
    
    for episode in range(num_episodes):
        ep_num = episode + 1
        print(f"\n--- Starting Episode {ep_num}/{num_episodes} ---")
        
        # Reset
        env_state = env.reset()
        if with_satiation:
            body_state = body.reset()
            state = (*env_state, body_state)
            print("Start State:")
            print(f"Satiation: {body.satiation}/{body.max_satiation}")
            print(f"Agent Pos: {env.agent_pos}")
            frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep_num, step=0))
        else:
            state = env_state
            print("Start State:")
            print(f"Agent Pos: {env.agent_pos}")
            frames.append(env.render_rgb_array(episode=ep_num, step=0))
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            action = agent.choose_action(state)
            action_names = ["Up", "Right", "Down", "Left"]
            
            print(f"Step {step_count + 1}: Action {action_names[action]}")
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            
            if with_satiation:
                next_body_state, reward, done = body.step(info)
                next_state = (*next_env_state, next_body_state)
                print(f"  Info: {info}")
                print(f"  Satiation: {body.satiation}/{body.max_satiation}")
                print(f"  Reward: {reward}, Done: {done}")
                frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep_num, step=step_count+1))
            else:
                reward = env_reward
                done = env_done
                next_state = next_env_state
                print(f"  Info: {info}")
                print(f"  Reward: {reward}, Done: {done}")
                frames.append(env.render_rgb_array(episode=ep_num, step=step_count+1))
            
            state = next_state
            step_count += 1
            
            if done:
                if with_satiation:
                    print("Episode Ended (Starved or Overfed).")
                else:
                    print("Episode Ended (Reached Goal).")
                # Add pause
                for _ in range(5):
                    if with_satiation:
                        frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep_num, step=step_count))
                    else:
                        frames.append(env.render_rgb_array(episode=ep_num, step=step_count))
                break
        
        if not done:
             print("Episode Ended (Max Steps Reached).")
             
    save_video(frames, video_filename)

if __name__ == "__main__":
    main()
