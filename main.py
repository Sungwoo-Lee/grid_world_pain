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
    parser.add_argument("--no-health", action="store_true", help="Disable health")
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
    
    with_health = config.get('body.with_health', False)
    if args.no_health:
        with_health = False
        
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

    if with_health:
        max_health = config.get_mandatory('body.max_health', float)
        start_health = config.get_mandatory('body.start_health', float)
        health_recovery = config.get_mandatory('body.health_recovery', float)
        start_health_random = config.get_mandatory('body.start_health_random')
    else:
        max_health = 20
        start_health = 10
        health_recovery = 1
        start_health_random = False

    # 3. Environment Config
    height = config.get_mandatory('environment.height', int)
    width = config.get_mandatory('environment.width', int)
    
    # Resource Pos fallback logic
    resource_pos = config.get('environment.resource_pos')
    if resource_pos is None:
         resource_pos = config.get_mandatory('environment.food_pos')
    
    start_pos = config.get_mandatory('environment.start_pos')
    
    prob_switch_to_danger = config.get_mandatory('environment.prob_switch_to_danger', float)
    min_danger_duration = config.get_mandatory('environment.min_danger_duration', int)
    damage_amount = config.get_mandatory('environment.damage_amount', float)
    
    prob_switch_to_food = config.get_mandatory('environment.prob_switch_to_food', float)
    min_food_duration = config.get_mandatory('environment.min_food_duration', int)
    
    relocate_resource = config.get_mandatory('environment.relocate_resource')
    relocation_steps = config.get_mandatory('environment.relocation_steps', int)
    
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
    decay_power = config.get('sensory.decay_power', 1.0)
    vector_size = config.get('sensory.vector_size', 10)
    food_property = config.get('sensory.food_property', None)
    danger_property = config.get('sensory.danger_property', None)
    nociceptor_radius = config.get('sensory.nociceptor_radius', 0)
    collision_sensor_enabled = config.get_mandatory('sensory.collision_sensor_enabled', bool)
    collision_sensor_range = config.get_mandatory('sensory.collision_sensor_range', int)
    location_sensor = config.get_mandatory('sensory.location_sensor')
    injury_smoothing_duration = config.get('body.injury_smoothing_duration', 3)

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_filename = os.path.join(current_dir, "gridworld_sandbox.mp4")

    print(f"Starting Debug Session")
    print(f"Video will be saved to: {video_filename}")
    print(f"Mode: {'Interoceptive' if with_satiation else 'Conventional'}")
    print(f"Health: {'Enabled' if with_health else 'Disabled'}")

    # Initialize Environment
    env = GridWorld(
        height=height, 
        width=width, 
        start=tuple(start_pos),
        resource_pos=tuple(resource_pos),
        with_satiation=with_satiation, 
        max_steps=max_steps,
        prob_switch_to_danger=prob_switch_to_danger, 
        min_danger_duration=min_danger_duration, 
        damage_amount=damage_amount,
        prob_switch_to_food=prob_switch_to_food, 
        min_food_duration=min_food_duration,
        relocate_resource=relocate_resource, 
        relocation_steps=relocation_steps,
        vector_size=vector_size, 
        food_property=food_property, 
        danger_property=danger_property,
        eat_action_enabled=config.get('environment.eat_action_enabled', True),
        rest_action_enabled=config.get('environment.rest_action_enabled', True),
        predator_enabled=predator_enabled,
        predator_move_interval=predator_move_interval,
        predator_damage=predator_damage,
        predator_start_pos=tuple(predator_start_pos),
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
        with_health=with_health,
        max_health=max_health,
        start_health=start_health,
        health_recovery=health_recovery,
        start_health_random=start_health_random,
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
            if with_health:
                print(f"Injury (Intero Nociceptor): {body.injury_level}/{body.max_injury}")
        
        if predator_enabled:
            print(f"Predator Pos: {env.predator_pos}")
        
        print(f"Agent Pos: {env.agent_pos}")
        
        # Visualization Data
        vis_data = None
        vis_data = None
        if using_sensory:
             vis_data = sensory_system.get_visualization_data(sensory_dict)



        # Capture Frame
        injury = body.injury_level if body.with_health else None
        max_injury = body.max_injury if body.with_health else None
        frames.append(env.render_rgb_array(
            satiation=body.satiation if with_satiation else None, 
            max_satiation=body.max_satiation if with_satiation else None, 
            injury=injury, 
            max_injury=max_injury, 
            episode=ep_num, 
            step=0, 
            sensory_data=vis_data
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
                 extra_data = {'injury_level': body.injury_level if body.with_health else 0, 
                               'max_injury': body.max_injury if body.with_health else 1}
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
            if with_health:
                print(f"  Injury: {body.injury_level}/{body.max_injury}")
            if predator_enabled:
                print(f"  Predator Pos: {env.predator_pos}")
            print(f"  Reward: {reward}, Done: {done}")

            # Capture Frame
            injury = body.injury_level if body.with_health else None
            max_injury = body.max_injury if body.with_health else None
            frames.append(env.render_rgb_array(
                satiation=body.satiation if with_satiation else None, 
                max_satiation=body.max_satiation if with_satiation else None, 
                injury=injury, 
                max_injury=max_injury, 
                episode=ep_num, 
                step=step_count+1, 
                sensory_data=vis_data
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
                        sensory_data=vis_data
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
                    sensory_data=vis_data
                ))

    save_video(frames, video_filename)
    print(f"Video saved to {video_filename}")

if __name__ == "__main__":
    main()
