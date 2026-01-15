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
from src.grid_world_pain.sensory import SensorySystem

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
    food_satiation_gain = config.get('body.food_satiation_gain', 10)

    # Setup results directory
    results_dir = "results"
    video_dir = os.path.join(results_dir, "videos")
    
    # Set numpy random seed for determinism
    np.random.seed(args.seed)
    
    video_filename = os.path.join(video_dir, "gridworld_debug.mp4")
    
    random_start_satiation = config.get('body.random_start_satiation', True)
    food_satiation_gain = config.get('body.food_satiation_gain', 10)
    use_homeostatic_reward = config.get('body.use_homeostatic_reward', False)
    satiation_setpoint = config.get('body.satiation_setpoint', 15)
    death_penalty = config.get('body.death_penalty', 100)

    # Danger Config
    danger_prob = config.get('environment.danger_prob', 0.1)
    danger_duration = config.get('environment.danger_duration', 5)
    damage_amount = config.get('environment.damage_amount', 5)

    # Sensory Config
    using_sensory = config.get('sensory.using_sensory', False)
    food_radius = config.get('sensory.food_radius', 1)
    danger_radius = config.get('sensory.danger_radius', 1)

    print(f"Starting Debug Session")
    print(f"Video will be saved to: {video_filename}")

    # Initialize components
    env = GridWorld(height=height, width=width, start=start_pos, food_pos=food_pos, 
                    with_satiation=with_satiation, max_steps=max_steps,
                    danger_prob=danger_prob, danger_duration=danger_duration, damage_amount=damage_amount)
    body = InteroceptiveBody(
        max_satiation=max_satiation, 
        start_satiation=start_satiation, 
        overeating_death=overeating_death, 
        random_start_satiation=random_start_satiation, 
        food_satiation_gain=food_satiation_gain,
        use_homeostatic_reward=use_homeostatic_reward,
        satiation_setpoint=satiation_setpoint,
        death_penalty=death_penalty
    )
    
    sensory_system = None
    state_dims = None
    
    # Sensory System Init
    if using_sensory:
        print(f"Initializing Sensory System (Food R={food_radius}, Danger R={danger_radius})")
        sensory_system = SensorySystem(food_radius=food_radius, danger_radius=danger_radius)
        
        # Calculate State Dimensions
        # Sensory: (FoodStateSpace, DangerStateSpace)
        sensory_dims = sensory_system.state_dims
        
        # Body: (max_satiation+2, max_health+2) ??
        # Let's match Agent's expectation for Body Dimensions
        body_dims = ()
        if with_satiation:
            # Note: Agent adds +2 padding for safety/terminal states usually
            if body.with_health:
                body_dims = (body.max_satiation + 2, body.max_health + 2)
            else:
                body_dims = (body.max_satiation + 2,)
        
        state_dims = sensory_dims + body_dims
        print(f"State Dimensions: {state_dims}")

    # Mock composite env for agent init
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
            
    composite_env = CompositeEnv(env, body)
    
    # Initialize Agent with explicit state_dims if using_sensory
    agent = QLearningAgent(composite_env, with_satiation=with_satiation, state_dims=state_dims)
    agent.epsilon = 1.0 # Force random behavior for debug sandbox
    
    frames = []
    
    num_episodes = args.episodes
    max_steps_per_episode = args.max_steps
    
    for episode in range(num_episodes):
        ep_num = episode + 1
        print(f"\n--- Starting Episode {ep_num}/{num_episodes} ---")
        
        # Reset
        env_state = env.reset()
        
        # Determine Initial OBSERVATION (State)
        current_agent_pos = env.agent_pos
        # Note: GridWorld doesn't expose list of dangers easily? 
        # Actually GridWorld internal logic handles danger but maybe we need to access it for sensors.
        # Let's check GridWorld implementation. 
        # Assuming GridWorld has `danger_pos` if active.
        current_danger_pos_list = []
        if hasattr(env, 'danger_pos') and env.danger_pos is not None:
             current_danger_pos_list = [env.danger_pos]
             
        if using_sensory:
            sensory_state = sensory_system.sense(current_agent_pos, env.food_pos, current_danger_pos_list)
            # sensory_state is (food_idx, danger_idx)
            
        if with_satiation:
            body_state = body.reset()
            if using_sensory:
                 if isinstance(body_state, tuple):
                     state = (*sensory_state, *body_state)
                 else:
                     state = (*sensory_state, body_state)
            else:
                # FOMDP
                if isinstance(body_state, tuple):
                     state = (*env_state, *body_state)
                else:
                     state = (*env_state, body_state)
                     
            print("Start State:")
            print(f"Satiation: {body.satiation}/{body.max_satiation}")
            print(f"Agent Pos: {env.agent_pos}")
            
            vis_data = None
            if using_sensory:
                print(f"Sensory: {sensory_state} (FoodIdx, DangerIdx)")
                vis_data = sensory_system.get_visualization_data(sensory_state)
                
            frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep_num, step=0, sensory_data=vis_data))
        else:
            # No body
            if using_sensory:
                state = sensory_state
                vis_data = sensory_system.get_visualization_data(sensory_state)
            else:
                state = env_state
                vis_data = None
                
            print("Start State:")
            print(f"Agent Pos: {env.agent_pos}")
            if using_sensory:
                print(f"Sensory: {sensory_state}")
            frames.append(env.render_rgb_array(episode=ep_num, step=0, sensory_data=vis_data))
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            action = agent.choose_action(state)
            action_names = ["Up", "Right", "Down", "Left", "Stay"]
            
            print(f"Step {step_count + 1}: Action {action_names[action]}")
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            
            # --- CALCULATE NEXT OBSERVATION (State) ---
            current_agent_pos = env.agent_pos # Updated pos
            current_danger_pos_list = []
            if hasattr(env, 'danger_pos') and env.danger_pos is not None:
                 current_danger_pos_list = [env.danger_pos]

            if using_sensory:
                 next_sensory_state = sensory_system.sense(current_agent_pos, env.food_pos, current_danger_pos_list)

            if with_satiation:
                next_body_state, reward, body_done = body.step(info)
                done = env_done or body_done
                
                if using_sensory:
                     if isinstance(next_body_state, tuple):
                         next_state = (*next_sensory_state, *next_body_state)
                     else:
                         next_state = (*next_sensory_state, next_body_state)
                else:
                    # FOMDP
                    if isinstance(next_body_state, tuple):
                        next_state = (*next_env_state, *next_body_state)
                    else:
                        next_state = (*next_env_state, next_body_state)
                        
                print(f"  Info: {info}")
                print(f"  Satiation: {body.satiation}/{body.max_satiation}")
                vis_data = None
                if using_sensory:
                     print(f"  Sensory: {next_sensory_state}")
                     vis_data = sensory_system.get_visualization_data(next_sensory_state)
                print(f"  Reward: {reward}, Done: {done}")
                frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep_num, step=step_count+1, sensory_data=vis_data))
            else:
                reward = env_reward
                done = env_done
                
                if using_sensory:
                    next_state = next_sensory_state
                else:
                    next_state = next_env_state
                    
                print(f"  Info: {info}")
                vis_data = None
                if using_sensory:
                    print(f"  Sensory: {next_sensory_state}")
                    vis_data = sensory_system.get_visualization_data(next_sensory_state)
                print(f"  Reward: {reward}, Done: {done}")
                frames.append(env.render_rgb_array(episode=ep_num, step=step_count+1, sensory_data=vis_data))
            
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
                        frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep_num, step=step_count, sensory_data=vis_data))
                    else:
                        frames.append(env.render_rgb_array(episode=ep_num, step=step_count, sensory_data=vis_data))
                break
        
        if not done:
             print("Episode Ended (Max Steps Reached).")
             # Add pause for max steps too
             for _ in range(5):
                if with_satiation:
                    frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=ep_num, step=step_count, sensory_data=vis_data))
                else:
                    frames.append(env.render_rgb_array(episode=ep_num, step=step_count, sensory_data=vis_data))
             
    save_video(frames, video_filename)

if __name__ == "__main__":
    main()
