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
- `--config <path>`: Path to config YAML (optional).

Changelog:
- 2026-01-29: Added Predator System support with strict configuration enforcement.
"""

import os
import argparse
import time
import datetime
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import re
import yaml

from src.environment import GridWorld
from src.environment.body import InteroceptiveBody
from src.environment.sensor import SensorySystem
from src.utils.visualization import save_video
from src.utils.config import get_default_config, Config

def main():
    parser = argparse.ArgumentParser(description="GridWorld Debug Sandbox")
    parser.add_argument("--episodes", type=int, help="Number of episodes to record in video")
    parser.add_argument("--max_steps", type=int, help="Maximum steps to record per episode")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--tag", type=str, default="default", help="Tag for the run")
    
    # Optional flags for quick overrides (will override config if present)
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    parser.add_argument("--no-injury", action="store_true", help="Disable health")
    parser.add_argument("--no-overeating-death", action="store_true", help="Disable death by overeating")

    args = parser.parse_args()

    # Load default config
    config = get_default_config()

    # Merge user config if provided
    if args.config:
        print(f"Loading main config from: {args.config}")
        user_config = Config.load_yaml(args.config)
        config.merge(user_config)

    # Resolve Parameters (Argument > Config > Error)
    def resolve_param(arg_val, config_key, type_converter=None):
        if arg_val is not None:
             # Store back for consistency
            config.set(config_key, arg_val)
            return arg_val
        return config.get_mandatory(config_key, type_converter)

    # 1. Essential Params
    episodes = resolve_param(args.episodes, 'testing.evaluation_episodes', int) if args.episodes else (args.episodes or 3)
    # Note: 'testing.evaluation_episodes' might not be in environment.yaml default, so fallback to args or hard default
    # Currently args.episodes defaults to None. 
    # Let's be strict but allow fallback if config missing.
    try:
        episodes = resolve_param(args.episodes, 'testing.evaluation_episodes', int)
    except ValueError:
        episodes = args.episodes if args.episodes is not None else 3
    
    try:
        max_steps = resolve_param(args.max_steps, 'environment.max_steps', int)
    except ValueError:
        max_steps = args.max_steps if args.max_steps is not None else 30

    try:
        seed = resolve_param(args.seed, 'training.seed', int)
    except ValueError:
        seed = args.seed if args.seed is not None else 42
        
    np.random.seed(seed)

    # 2. Body Config
    with_satiation = config.get('body.with_satiation', True)
    if args.no_satiation:
        with_satiation = False
    
    with_injury = config.get('body.with_injury', False)
    if args.no_injury:
        with_injury = False
        
    overeating_death = config.get('body.overeating_death', True)
    if args.no_overeating_death:
        overeating_death = False

    # Extract Body Params
    if with_satiation:
        max_satiation = config.get_mandatory('body.max_satiation', int)
        start_satiation = config.get_mandatory('body.start_satiation', int)
        random_start_satiation = config.get_mandatory('body.random_start_satiation')
        food_satiation_gain = config.get_mandatory('body.food_satiation_gain', float)
        use_homeostatic_reward = config.get_mandatory('body.use_homeostatic_reward')
        satiation_setpoint = config.get_mandatory('body.satiation_setpoint', float)
        death_penalty = config.get_mandatory('body.death_penalty', float)
    else:
        # Dummy values if disabled, though Body handles it
        max_satiation = 20
        start_satiation = 10
        random_start_satiation = False
        food_satiation_gain = 10
        use_homeostatic_reward = False
        satiation_setpoint = 15
        death_penalty = 100

    if with_injury:
        max_injury = config.get_mandatory('body.max_injury', float)
        start_injury = config.get_mandatory('body.start_injury', float)
        injury_recovery = config.get_mandatory('body.injury_recovery', float)
        random_start_injury = config.get_mandatory('body.random_start_injury')
    else:
        max_injury = 20
        start_injury = 10
        injury_recovery = 1
        random_start_injury = False

    # 3. Environment Config
    height = config.get_mandatory('environment.height', int)
    width = config.get_mandatory('environment.width', int)
    start_pos = config.get_mandatory('environment.start_pos')
    
    # Predator Config
    predator_enabled = config.get_mandatory('predator.enabled', bool)
    
    if predator_enabled:
        predator_move_interval = config.get_mandatory('predator.move_interval', int)
        predator_damage = config.get_mandatory('predator.damage', float)
        predator_start_pos = config.get_mandatory('predator.start_pos')
        predator_random_start_pos = config.get_mandatory('predator.random_start_pos', bool)
        predator_property = config.get_mandatory('predator.property')
    else:
        # Defaults for disabled state (not used by GridWorld if enabled=False)
        predator_move_interval = 2
        predator_damage = 0.0
        predator_start_pos = (0, 0)
        predator_random_start_pos = False
        predator_property = None
    
    # Sensory Config
    using_sensory = config.get_mandatory('sensory.using_sensory')
    sensor_radius = config.get_mandatory('sensory.sensor_radius', int)
    decay_power = config.get_mandatory('sensory.decay_power', float)
    vector_size = config.get_mandatory('sensory.vector_size', int)
    nociceptor_radius = config.get_mandatory('sensory.nociceptor_radius', int)
    collision_sensor_enabled = config.get_mandatory('sensory.collision_sensor_enabled', bool)
    collision_sensor_range = config.get_mandatory('sensory.collision_sensor_range', int)
    location_sensor = config.get_mandatory('sensory.location_sensor')
    injury_smoothing_duration = config.get_mandatory('body.injury_smoothing_duration', int)

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_filename = os.path.join(current_dir, "gridworld_sandbox.mp4")

    print(f"Starting Debug Session")
    print(f"Video will be saved to: {video_filename}")
    print(f"Mode: {'Interoceptive' if with_satiation else 'Conventional'}")
    print(f"Injury: {'Enabled' if with_injury else 'Disabled'}")

    # Initialize Environment
    env = GridWorld(
        height=height, 
        width=width, 
        start=tuple(start_pos),
        resources=config.get_mandatory('environment.resources'),
        with_satiation=with_satiation, 
        max_steps=max_steps,
        eat_action_enabled=config.get_mandatory('environment.eat_action_enabled', bool),
        rest_action_enabled=config.get_mandatory('environment.rest_action_enabled', bool),
        predator_enabled=predator_enabled,
        predator_move_interval=predator_move_interval,
        predator_damage=predator_damage,
        predator_start_pos=predator_start_pos,
        predator_random_start_pos=predator_random_start_pos,
        predator_property=predator_property,
        injury_smoothing_duration=injury_smoothing_duration
    )
    
    # Get action dimension from environment spec (dynamic based on eat/rest config)
    action_dim = env.action_spec()['n']
    
    body = InteroceptiveBody(
        max_satiation=max_satiation, 
        start_satiation=start_satiation, 
        overeating_death=overeating_death, 
        random_start_satiation=random_start_satiation, 
        food_satiation_gain=food_satiation_gain,
        use_homeostatic_reward=use_homeostatic_reward,
        satiation_setpoint=satiation_setpoint,
        death_penalty=death_penalty,
        with_injury=with_injury,
        max_injury=max_injury,
        start_injury=start_injury,
        injury_recovery=injury_recovery,
        random_start_injury=random_start_injury,
        injury_smoothing_duration=injury_smoothing_duration,
        with_satiation=with_satiation
    )
    
    sensory_system = None
    if using_sensory:
        print(f"Initializing Sensory System (Radius={sensor_radius}, VecSize={vector_size}, Decay={decay_power})")
        nociception_size = config.get('sensory.nociception_size', 1)
        location_size = config.get('sensory.location_size', 2)
        olfactory_enabled = config.get('sensory.olfactory_enabled', True)
        sensory_system = SensorySystem(
            sensor_radius=sensor_radius, 
            vector_size=vector_size, 
            decay_power=decay_power, 
            nociceptor_radius=nociceptor_radius,
            location_sensor=location_sensor,
            nociception_enabled=config.get('sensory.nociception_enabled', True),
            olfactory_enabled=olfactory_enabled,
            collision_sensor_enabled=collision_sensor_enabled,
            collision_sensor_range=collision_sensor_range,
            proprioception_enabled=config.get('sensory.proprioception_enabled', False),
            nociception_size=nociception_size,
            location_size=location_size,
            num_actions=action_dim
        )
        
    # Log Specs
    print("\n--- RL API Specifications (Sandbox) ---")
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    print(f"Environment Action Spec: {action_spec}")
    if using_sensory:
         sensory_spec = sensory_system.observation_spec()
         print(f"Sensory System Spec: {sensory_spec}")
    else:
         print(f"Environment Observation Spec: {observation_spec}")
    print("--------------------------------------------\n")
        
    frames = []
    
    for episode in range(episodes):
        ep_num = episode + 1
        print(f"\n--- Starting Episode {ep_num}/{episodes} ---")
        
        # Reset
        env_state = env.reset()
        current_agent_pos = env.agent_pos
        
        sensory_dict = {}
        if using_sensory:
             resources = env.get_active_resources()
             extra_data = {'injury_level': body.injury_level if with_satiation else 0, 
                           'max_injury': body.max_injury if with_satiation else 1}
             sensory_dict = sensory_system.sense(current_agent_pos, resources, grid_height=height, grid_width=width, extra_data=extra_data)
        
        body_return = None
        if with_satiation:
            body_return = body.reset()
            
            print("Start State:")
            print(f"Satiation: {body.satiation}/{body.max_satiation}")
            if with_injury:
                print(f"Injury (Intero Nociceptor): {body.injury_level}/{body.max_injury}")
        
        if predator_enabled:
            print(f"Predator Pos: {env.predator_pos}")
        
        print(f"Agent Pos: {env.agent_pos}")
        
        # Visualization Data
        vis_data = None
        vis_data = None
        if using_sensory:
             vis_data = sensory_system.get_visualization_data(sensory_dict)



        # Load icon_scale from visualization config
        viz_config_path = os.path.join(os.path.dirname(__file__), "configs", "visualization", "visualization.yaml")
        icon_scale = 1.0
        if os.path.exists(viz_config_path):
            with open(viz_config_path, 'r') as f:
                viz_data = yaml.safe_load(f).get('visualization', {})
                icon_scale = viz_data.get('icon_scale', 1.0)

        # Capture Frame
        injury = body.injury_level if body.with_injury else None
        max_injury = body.max_injury if body.with_injury else None
        frames.append(env.render_rgb_array(
            satiation=body.satiation if with_satiation else None, 
            max_satiation=body.max_satiation if with_satiation else None, 
            injury=injury, 
            max_injury=max_injury, 
            episode=ep_num, 
            step=0, 
            sensory_data=vis_data,
            icon_scale=icon_scale
        ))
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Action Selection: Random from dynamic action space
            action = np.random.randint(0, action_dim)
            action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
            if env.REST_ACTION is not None:
                action_names[env.REST_ACTION] = "Rest"
            if env.EAT_ACTION is not None:
                action_names[env.EAT_ACTION] = "Eat"
            
            # Step External
            next_env_state, env_reward, env_done, info = env.step(action)
            current_agent_pos = env.agent_pos

            # Print Step Info
            print(f"Step {step_count+1}: Action {action_names[action]}")
            
            if using_sensory:
                 resources = env.get_active_resources()
                 extra_data = {'injury_level': body.injury_level if body.with_injury else 0, 
                               'max_injury': body.max_injury if body.with_injury else 1}
                 sensory_dict = sensory_system.sense(current_agent_pos, resources, grid_height=env.height, grid_width=env.width, extra_data=extra_data)
                 vis_data = sensory_system.get_visualization_data(sensory_dict)

            reward = env_reward
            # Step Body / Physiology
            body_state, body_reward, body_done = body.step(info)

            # Logic: If using homeostatic reward, use body_reward.
            # Otherwise (conventional), use env_reward and env_done, but allow body to trigger death.
            if config.get('body.use_homeostatic_reward', True):
                 reward = body_reward
                 done = env_done or body_done
            else:
                 reward = env_reward
                 done = env_done or body_done

            print(f"  Info: {info}")
            if with_satiation:
                print(f"  Satiation: {body.satiation}/{body.max_satiation}")
            if with_injury:
                print(f"  Injury: {body.injury_level}/{body.max_injury}")
            if predator_enabled:
                print(f"  Predator Pos: {env.predator_pos}")
            print(f"  Reward: {reward}, Done: {done}")

            # Capture Frame
            injury = body.injury_level if body.with_injury else None
            max_injury = body.max_injury if body.with_injury else None
            frames.append(env.render_rgb_array(
                satiation=body.satiation if with_satiation else None, 
                max_satiation=body.max_satiation if with_satiation else None, 
                injury=injury, 
                max_injury=max_injury, 
                episode=ep_num, 
                step=step_count+1, 
                sensory_data=vis_data,
                icon_scale=icon_scale
            ))
            
            step_count += 1
            
            if done:
                print("Episode Ended.")
                # Add pause frames
                for _ in range(5):
                     frames.append(env.render_rgb_array(
                        satiation=body.satiation if with_satiation else None, 
                        max_satiation=body.max_satiation if with_satiation else None, 
                        injury=injury, 
                        max_injury=max_injury, 
                        episode=ep_num, 
                        step=step_count, 
                        sensory_data=vis_data,
                        icon_scale=icon_scale
                    ))
                break
        
        if not done:
             print("Episode Ended (Max Steps Reached).")
             for _ in range(5):
                 frames.append(env.render_rgb_array(
                    satiation=body.satiation if with_satiation else None, 
                    max_satiation=body.max_satiation if with_satiation else None, 
                    injury=injury, 
                    max_injury=max_injury, 
                    episode=ep_num, 
                    step=step_count, 
                    sensory_data=vis_data,
                    icon_scale=icon_scale
                ))

    save_video(frames, video_filename)
    print(f"Video saved to {video_filename}")

if __name__ == "__main__":
    main()
