
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import glob
from src.environment import GridWorld
from src.environment.sensor import SensorySystem
from src.environment.body import InteroceptiveBody
from src.utils.config import get_default_config

def verify_rendering():
    # Discover all configs in subfolders
    base_dir = "configs/ablation"
    config_paths = []
    for branch in ["survival", "homeostatic"]:
        branch_dir = os.path.join(base_dir, branch)
        if os.path.isdir(branch_dir):
            config_paths.extend(glob.glob(os.path.join(branch_dir, "*.yaml")))
    
    # Sort for consistent output
    config_paths.sort()
    
    # Save directory in brain artifacts
    save_dir = "/home/vncuser/.gemini/antigravity/brain/ac30859c-a5af-40aa-a2de-325649406f81/verification"
    os.makedirs(save_dir, exist_ok=True)
    
    print("=== Ablation Branch Rendering Verification ===\n")
    
    for path in config_paths:
        # Extract metadata
        parts = path.split(os.sep)
        branch = parts[-2]
        filename = os.path.basename(path)
        
        print(f"Testing [{branch}] {filename}...")
        
        # Load config
        config = get_default_config()
        with open(path, 'r') as f:
            overrides = yaml.safe_load(f)
            config.merge(overrides)
            
        # Init Env
        with_satiation = config.get_mandatory('body.with_satiation')
        env = GridWorld(
            height=config.get_mandatory('environment.height'),
            width=config.get_mandatory('environment.width'),
            start=tuple(config.get_mandatory('environment.start_pos')),
            resource_pos=tuple(config.get_mandatory('environment.resource_pos')),
            with_satiation=with_satiation,
            max_steps=config.get_mandatory('environment.max_steps'),
            prob_switch_to_danger=config.get_mandatory('environment.prob_switch_to_danger'),
            min_danger_duration=config.get_mandatory('environment.min_danger_duration'),
            damage_amount=config.get_mandatory('environment.damage_amount'),
            prob_switch_to_food=config.get_mandatory('environment.prob_switch_to_food'),
            min_food_duration=config.get_mandatory('environment.min_food_duration'),
            relocate_resource=config.get_mandatory('environment.relocate_resource'),
            relocation_steps=config.get_mandatory('environment.relocation_steps'),
            vector_size=config.get_mandatory('sensory.vector_size'),
            food_property=config.get_mandatory('sensory.food_property'),
            danger_property=config.get_mandatory('sensory.danger_property'),
            eat_action_enabled=config.get_mandatory('environment.eat_action_enabled'),
            rest_action_enabled=config.get_mandatory('environment.rest_action_enabled')
        )
        
        body = InteroceptiveBody(
            max_satiation=config.get_mandatory('body.max_satiation'),
            start_satiation=config.get_mandatory('body.start_satiation'),
            overeating_death=config.get_mandatory('body.overeating_death'),
            random_start_satiation=config.get_mandatory('body.random_start_satiation'),
            food_satiation_gain=config.get_mandatory('body.food_satiation_gain'),
            use_homeostatic_reward=config.get_mandatory('body.use_homeostatic_reward'),
            satiation_setpoint=config.get_mandatory('body.satiation_setpoint'),
            death_penalty=config.get_mandatory('body.death_penalty'),
            with_injury=config.get_mandatory('body.with_injury'),
            max_injury=config.get_mandatory('body.max_injury'),
            start_injury=config.get_mandatory('body.start_injury'),
            injury_recovery=config.get_mandatory('body.injury_recovery'),
            start_injury_random=config.get_mandatory('body.start_injury_random')
        )
        
        sensory_system = None
        if config.get_mandatory('sensory.using_sensory'):
            sensory_system = SensorySystem(
                sensor_radius=config.get_mandatory('sensory.sensor_radius'),
                vector_size=config.get_mandatory('sensory.vector_size'),
                decay_power=config.get_mandatory('sensory.decay_power'),
                nociceptor_radius=config.get_mandatory('sensory.nociceptor_radius'),
                location_sensor=config.get_mandatory('sensory.location_sensor'),
                num_actions=env.action_spec()['n'],
                nociception_enabled=config.get('sensory.nociception_enabled', True),
                olfactory_enabled=config.get('sensory.olfactory_enabled', True),
                collision_sensor_enabled=config.get('sensory.collision_sensor_enabled', True),
                collision_sensor_range=config.get('sensory.collision_sensor_range', 0),
                proprioception_enabled=config.get('sensory.proprioception_enabled', False),
                nociception_size=config.get('sensory.nociception_size', 1),
                location_size=config.get('sensory.location_size', 2)
            )
            
        # Reset
        env.reset()
        
        # Force states for visualization completeness
        if "danger" in filename or "nociception" in filename:
            env.agent_pos = env.resource_pos
            env.resource_state = 'DANGER'
            env.step(4 if env.rest_action_enabled else 0)
            
        # Get Sensory Data
        sensory_dict = {}
        if sensory_system:
            resources = env.get_active_resources()
            sensory_dict = sensory_system.sense(
                env.agent_pos, resources,
                grid_height=env.height, grid_width=env.width, resource_pos=env.resource_pos
            )
            
        # Get Visualization Data
        vis_data = None
        if sensory_system:
            vis_data = sensory_system.get_visualization_data(sensory_dict)
            
        # Render
        frame = env.render_rgb_array(
            satiation=body.satiation if with_satiation else None,
            max_satiation=body.max_satiation if with_satiation else None,
            injury=body.injury if body.with_injury else None,
            max_injury=body.max_injury if body.with_injury else None,
            episode=1,
            step=env.current_step,
            sensory_data=vis_data
        )
        
        # Save frame in branch-specific subfolder in results
        branch_save_dir = os.path.join(save_dir, branch)
        os.makedirs(branch_save_dir, exist_ok=True)
        frame_path = os.path.join(branch_save_dir, f"{filename.replace('.yaml', '.png')}")
        plt.imsave(frame_path, frame)
        
        print(f"  Frame saved: {branch}/{os.path.basename(frame_path)}")
        
        # Check reward type
        rew_type = "Homeostatic" if body.use_homeostatic_reward else "Survival"
        if filename == "01_goal_only.yaml": rew_type = "Goal (Fixed)"
        print(f"  Reward Mode: {rew_type}")
            
    print("\nVerification complete. Check artifacts directory for branch-wise images.")

if __name__ == "__main__":
    verify_rendering()
