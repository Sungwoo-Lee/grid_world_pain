"""
Debugging Script / Sandbox.

Purpose:
- To verify the Environmental Mechanics (Grid + Body) without any Learning Agent.
- Uses a Random Agent to simply walk around.
- Checks if "Eating" works, if "Satiation" changes, and if "Death" occurs correctly.

Arguments:
- `--episodes <int>`: Number of episodes to record in the video.
- `--max_steps <int>`: Maximum steps to record per episode.
- `--seed <int>`: Random seed for reproducibility.
- `--config <path>`: Path to config YAML (optional).

Changelog:
- 2026-01-29: Added Predator System support with strict configuration enforcement.
- 2026-02-03: Synchronized with train.py strict configuration logic.
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

    # 1. Start with Base Config (Environment Defaults)
    config = get_default_config()

    # 2. Merge Training Defaults
    train_config_path = os.path.join(os.path.dirname(__file__), "configs", "train", "default.yaml")
    if os.path.exists(train_config_path):
        train_defaults = Config.load_yaml(train_config_path)
        config.merge(train_defaults)
        print(f"Loaded training defaults from {train_config_path}")

    # 3. Merge Visualization Defaults
    viz_config_path = os.path.join(os.path.dirname(__file__), "configs", "visualization", "visualization.yaml")
    if os.path.exists(viz_config_path):
        with open(viz_config_path, 'r') as f:
            viz_dict = yaml.safe_load(f)
            config.merge(viz_dict)
            print(f"Merged visualization settings from {viz_config_path}")

    # 4. Merge Evaluation Defaults
    eval_config_path = os.path.join(os.path.dirname(__file__), "configs", "evaluation", "default.yaml")
    if os.path.exists(eval_config_path):
        eval_defaults = Config.load_yaml(eval_config_path)
        config.merge(eval_defaults)
        print(f"Loaded evaluation defaults from {eval_config_path}")

    # 5. Merge User / Ablation Config (Overrides everything)
    if args.config:
        print(f"Loading user/ablation config from: {args.config}")
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
    episodes = int(resolve_param(args.episodes, 'testing.evaluation_episodes'))
    max_steps = int(resolve_param(args.max_steps, 'environment.max_steps'))
    seed = int(resolve_param(args.seed, 'training.seed'))
    np.random.seed(seed)

    # 2. Body Config
    with_satiation = resolve_param(None, 'body.with_satiation')
    if args.no_satiation:
        with_satiation = False
        config.set('body.with_satiation', False)
    
    with_injury = resolve_param(None, 'body.with_injury')
    if args.no_injury:
        with_injury = False
        config.set('body.with_injury', False)
        
    overeating_death = resolve_param(None, 'body.overeating_death')
    if args.no_overeating_death:
        overeating_death = False
        config.set('body.overeating_death', False)

    # Extract Body Params
    max_satiation = config.get_mandatory('body.max_satiation', int)
    start_satiation = config.get_mandatory('body.start_satiation', int)
    random_start_satiation = config.get_mandatory('body.random_start_satiation')
    food_satiation_gain = config.get_mandatory('body.food_satiation_gain', float)
    use_homeostatic_reward = config.get_mandatory('body.use_homeostatic_reward')
    satiation_setpoint = config.get_mandatory('body.satiation_setpoint', float)
    death_penalty = config.get_mandatory('body.death_penalty', float)

    max_injury = config.get_mandatory('body.max_injury', float)
    start_injury = config.get_mandatory('body.start_injury', float)
    injury_recovery = config.get_mandatory('body.injury_recovery', float)
    random_start_injury = config.get_mandatory('body.random_start_injury')
    injury_smoothing_duration = config.get_mandatory('body.injury_smoothing_duration', int)

    # 3. Environment Config
    height = config.get_mandatory('environment.height', int)
    width = config.get_mandatory('environment.width', int)
    start_pos = config.get_mandatory('environment.start_pos')
    
    # Predator Config
    predator_enabled = config.get_mandatory('environment.predator_enabled', bool)
    
    # Sensory Config
    using_sensory = config.get_mandatory('sensory.using_sensory')
    sensor_radius = config.get_mandatory('sensory.sensor_radius', int)
    decay_power = config.get_mandatory('sensory.decay_power', float)
    vector_size = config.get_mandatory('sensory.vector_size', int)
    nociceptor_radius = config.get_mandatory('sensory.nociceptor_radius', int)
    collision_sensor_enabled = config.get_mandatory('sensory.collision_sensor_enabled', bool)
    collision_sensor_range = config.get_mandatory('sensory.collision_sensor_range', int)
    location_sensor = config.get_mandatory('sensory.location_sensor')
    icon_scale = config.get_mandatory('visualization.icon_scale', float)

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
        predators=config.get_mandatory('environment.predators'), # New Mandatory
        with_satiation=with_satiation, 
        max_steps=max_steps,
        eat_action_enabled=config.get_mandatory('environment.eat_action_enabled', bool),
        rest_action_enabled=config.get_mandatory('environment.rest_action_enabled', bool),
        predator_enabled=predator_enabled,
        injury_smoothing_duration=injury_smoothing_duration,
        vector_size=vector_size
        # Legacy/Extra arguments (move_interval, etc.) are now in the 'predators' list config
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
        nociception_size = config.get_mandatory('sensory.nociception_size', int)
        location_size = config.get_mandatory('sensory.location_size', int)
        olfactory_enabled = config.get_mandatory('sensory.olfactory_enabled', bool)
        sensory_system = SensorySystem(
            sensor_radius=sensor_radius, 
            vector_size=vector_size, 
            decay_power=decay_power, 
            nociceptor_radius=nociceptor_radius,
            location_sensor=location_sensor,
            nociception_enabled=config.get_mandatory('sensory.nociception_enabled', bool),
            olfactory_enabled=olfactory_enabled,
            collision_sensor_enabled=collision_sensor_enabled,
            collision_sensor_range=collision_sensor_range,
            proprioception_enabled=config.get_mandatory('sensory.proprioception_enabled', bool),
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
        env.reset()
        current_agent_pos = env.agent_pos
        
        sensory_dict = {}
        if using_sensory:
             resources = env.get_active_resources()
             extra_data = {'injury_level': body.injury_level if with_satiation else 0, 
                           'max_injury': body.max_injury if with_satiation else 1}
             sensory_dict = sensory_system.sense(current_agent_pos, resources, grid_height=height, grid_width=width, extra_data=extra_data)
        
        if with_satiation:
            body.reset()
            
            print("Start State:")
            print(f"Satiation: {body.satiation}/{body.max_satiation}")
            if with_injury:
                print(f"Injury (Intero Nociceptor): {body.injury_level}/{body.max_injury}")
        
        if predator_enabled:
            for i, p in enumerate(env.predators):
                print(f"Predator {i} Pos: {p.pos} | State: {p.state}")
        
        print(f"Agent Pos: {env.agent_pos}")
        
        # Visualization Data
        vis_data = None
        if using_sensory:
             vis_data = sensory_system.get_visualization_data(sensory_dict)

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
                for i, p in enumerate(env.predators):
                    print(f"  Predator {i} Pos: {p.pos} | State: {p.state}")
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
