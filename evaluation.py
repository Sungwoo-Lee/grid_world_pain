"""
Evaluation script for the GridWorld Reinforcement Learning agent.

This script:
1. Loads the configuration saved during training (results/.../models/config.yaml).
2. Sets up the evaluation environment (Grid, Body, Sensory) to match training.
3. Instantiates the appropriate agent (Tabular, DQN, or PPO).
4. Evaluates checkpoints:
   - Runs evaluation episodes (deterministic).
   - Collects frames for video generation.
   - Generates Q-table plots (if Tabular).
   - Saves artifacts to `results/.../RunName/`.

Usage:
    python evaluation.py --results_dir results/DQN/MyRun_... --episodes 3

Notes:
- Uses the configuration saved during training (`config.yaml`).
- Supports CPU/GPU execution (auto-detected or inherited from config).
- Ensures at least one resource (Food or Danger) is active at all times.
"""
import os
import glob
import re
import yaml
import numpy as np
import argparse
from src.environment import GridWorld
from src.environment.body import InteroceptiveBody
from src.models.q_learning import QLearningAgent
from src.environment.sensor import SensorySystem
from src.utils.config import Config, get_default_config
from src.utils.visualization import plot_q_table, save_video, visualize_activations, combine_frame_and_activations
from src.utils.activation_monitor import ActivationMonitor
import torch


def evaluate_checkpoint(checkpoint_path, results_dir, config):
    """
    Evaluates a single checkpoint:
    - Sets up environment and body based on config.
    - Loads agent.
    - Runs evaluation episodes to collect frames.
    - Generates Q-table plot and performance video.
    """
    filename = os.path.basename(checkpoint_path)
    # Try Q-table pattern
    match = re.search(r"q_table_(\d+).npy", filename)
    if not match:
        match = re.search(r"dqn_model_(\d+).pth", filename)
    if not match:
        match = re.search(r"ppo_model_(\d+).pth", filename)
    if not match:
        match = re.search(r"drqn_model_(\d+).pth", filename)
    if not match:
        match = re.search(r"recurrent_ppo_model_(\d+).pth", filename)
    if not match:
        match = re.search(r"dreamer_model_(\d+).pth", filename)
        
    pct = match.group(1) if match else ("final" if "final" in filename or filename == "q_table.npy" else "unknown")
    
    print(f"Evaluating checkpoint: {filename} ({pct}%)")

    # 1. Component Extraction from Config
    with_satiation = config.get('body.with_satiation', True)
    overeating_death = config.get('body.overeating_death', True)
    max_steps = config.get('environment.max_steps', 100)
    seed = config.get('testing.seed', 42)
    num_episodes = config.get('testing.evaluation_episodes', 1)
    resource_pos = config.get('environment.resource_pos')
    if resource_pos is None:
         resource_pos = config.get('environment.food_pos', [4, 4])
         
    height = config.get('environment.height', 5)
    width = config.get('environment.width', 5)
    max_satiation = config.get('body.max_satiation', 20)
    start_satiation = config.get('body.start_satiation', 10)
    random_start_satiation = config.get('body.random_start_satiation', True)
    food_satiation_gain = config.get('body.food_satiation_gain', 10)
    use_homeostatic_reward = config.get('body.use_homeostatic_reward', False)
    satiation_setpoint = config.get('body.satiation_setpoint', 15)
    death_penalty = config.get('body.death_penalty', 100)
    
    # New Health/Pain Params
    with_health = config.get('body.with_health', False)
    max_health = config.get('body.max_health', 20)
    start_health = config.get('body.start_health', 10)
    health_recovery = config.get('body.health_recovery', 1)
    start_health_random = config.get('body.start_health_random', True)
    
    pain_prob = config.get('environment.danger_prob', 0.1)
    pain_duration = config.get('environment.danger_duration', 5)
    damage_amount = config.get('environment.damage_amount', 5)
    
    food_prob = config.get('environment.food_prob', 0.2)
    food_duration = config.get('environment.food_duration', 10)
    
    # Extract Relocation Config
    relocate_resource = config.get('environment.relocate_resource', False)
    relocation_steps = config.get('environment.relocation_steps', 20)

    # 2. Environment & Body Setup
    # Set seed for deterministic evaluation
    np.random.seed(seed)
    
    env = GridWorld(height=height, width=width, resource_pos=resource_pos, with_satiation=with_satiation, max_steps=max_steps,
                    danger_prob=pain_prob, danger_duration=pain_duration, damage_amount=damage_amount,
                    food_prob=food_prob, food_duration=food_duration,
                    relocate_resource=relocate_resource, relocation_steps=relocation_steps)
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
        start_health_random=start_health_random
    )
    
    # Sensory System
    using_sensory = config.get('sensory.using_sensory', False)
    sensory_system = None
    if using_sensory:
        food_radius = config.get('sensory.food_radius', 1)
        danger_radius = config.get('sensory.danger_radius', 1)
        sensory_system = SensorySystem(food_radius=food_radius, danger_radius=danger_radius)

    # Preprocessor for DQN
    def preprocess_state(state_tuple):
        flat_list = []
        if using_sensory:
            food_idx = state_tuple[0]
            danger_idx = state_tuple[1]
            food_vec = sensory_system.food_sensor.index_to_vector(food_idx)
            danger_vec = sensory_system.danger_sensor.index_to_vector(danger_idx)
            flat_list.extend(food_vec)
            flat_list.extend(danger_vec)
            body_start_idx = 2
        else:
            # Coords
            row = state_tuple[0]
            col = state_tuple[1]
            flat_list.append(row / height)
            flat_list.append(col / width)
            body_start_idx = 2

        if len(state_tuple) > body_start_idx:
            satiation = state_tuple[body_start_idx]
            flat_list.append(satiation / body.max_satiation) 
        if len(state_tuple) > body_start_idx + 1:
            health = state_tuple[body_start_idx + 1]
            flat_list.append(health / body.max_health)
        return np.array(flat_list, dtype=np.float32)

    # Initialize Agent
    agent = None
    algorithm = config.get('agent.algorithm', "Tabular Q-Learning")
    
    device = config.get('training.device', 'auto') # Use training device setting or auto
    
    if algorithm == "DQN":
        # DQN inputs
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 # row, col

        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
        
        from src.models.dqn import DQNAgent
        agent = DQNAgent(state_dim=input_dim, action_dim=5)
        # Load weights
        import torch
        # Map location if needed? default load is fine
        agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device if device != 'auto' else None))
        agent.epsilon = 0.0 # Eval mode

    elif algorithm == "DRQN":
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 

        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
        
        from src.models.drqn import DRQNAgent
        agent = DRQNAgent(state_dim=input_dim, action_dim=5, device=device)
        import torch
        agent.load(checkpoint_path)
        agent.epsilon = 0.0

    elif algorithm == "PPO":
        from src.models.ppo import PPOAgent
        # PPO inputs same as DQN
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 # row, col

        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
        
        agent = PPOAgent(state_dim=input_dim, action_dim=5, device=device)
        agent.load(checkpoint_path)

    elif algorithm == "RecurrentPPO":
        from src.models.recurrent_ppo import RecurrentPPOAgent
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 

        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
        
        agent = RecurrentPPOAgent(state_dim=input_dim, action_dim=5, sequence_length=config.get('agent.sequence_length', 8), device=device)
        agent.load(checkpoint_path)

    elif algorithm == "DreamerV3":
        from src.models.dreamer_v3 import DreamerV3Agent
        input_dim = 0
        if using_sensory:
             input_dim += sensory_system.food_sensor.vector_size + \
                          sensory_system.danger_sensor.vector_size
        else:
             input_dim += 2 

        if with_satiation:
            input_dim += 1
            if with_health:
                 input_dim += 1
        
        agent = DreamerV3Agent(state_dim=input_dim, action_dim=5, batch_size=config.get('agent.batch_size', 16), batch_length=config.get('agent.batch_length', 16), device=device)
        agent.load(checkpoint_path)

    else:
        # Tabular
        class CompositeEnv:
            def __init__(self, env, body):
                self.height = env.height
                self.width = env.width
                self.max_satiation = body.max_satiation
                self.with_health = body.with_health
                self.max_health = body.max_health
                
        agent = QLearningAgent(CompositeEnv(env, body), with_satiation=with_satiation)
        try:
            agent.load(checkpoint_path)
            agent.epsilon = 0 # No exploration during evaluation
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            return

    # Setup Activation Monitor

    monitor = None
    if algorithm != "Tabular Q-Learning":
        model_to_monitor = None
        if isinstance(agent, torch.nn.Module):
            model_to_monitor = agent
        elif hasattr(agent, 'policy_net'):
            model_to_monitor = agent.policy_net
        elif hasattr(agent, 'policy'):
            # For PPO, policy is ActorCritic which has actor and critic
            model_to_monitor = agent.policy
            
        if model_to_monitor:
            print(f"  Monitoring activations for {type(model_to_monitor).__name__}...")
            # We track Linear and Conv layers primarily
            monitor = ActivationMonitor(model_to_monitor, tracked_layers=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM, torch.nn.GRU])
            
            # Warm-up to prompt monitor to capture layer structure and determine frame size
            # Create a dummy input based on input_dim
            # input_dim logic is complex above, but we can just use a zero tensor of approx shape?
            # Or just wait for first frame? No, first frame is rendered BEFORE first step.
            
            # Helper to generate zero frame of correct size
            # We need to know the dummy input shape.
            # Reuse logic for `input_dim` or just one forward.
            try:
                # Construct dummy state
                # We need input_dim from code above.
                # Code above defines `input_dim` inside blocks (DQN, etc).
                # We can access it if we move monitor setup AFTER agent init fully.
                # Monitor setup IS after agent init.
                # But `input_dim` variable is local to blocks.
                # Let's try to infer from agent.policy_net first layer?
                dummy_input = None
                first_layer = None
                for module in model_to_monitor.modules():
                     if isinstance(module, torch.nn.Linear):
                         dummy_input = torch.zeros(1, module.in_features).to(device)
                         break
                     elif isinstance(module, torch.nn.Conv2d):
                         # Assuming square input? tough.
                         pass
                
                if dummy_input is not None:
                     # Run forward
                     with torch.no_grad():
                          if hasattr(agent, 'reset_hidden'): agent.reset_hidden()
                          if "DRQN" in type(agent).__name__ or "Recurrent" in type(agent).__name__:
                               # DRQN forward expects (batch, seq, dim)
                               dummy_input = dummy_input.unsqueeze(0) 
                               model_to_monitor(dummy_input)
                               if hasattr(agent, 'reset_hidden'): agent.reset_hidden()
                          else:
                               model_to_monitor(dummy_input)
                     
                     monitor.template_activations = monitor.get_current_activations().copy()
                     monitor.clear_history() # Clear the dummy record
                else:
                    monitor.template_activations = None
            except Exception as e:
                print(f"Activation warm-up failed: {e}")
                monitor.template_activations = None


    # Helper to process frame
    def append_frame_with_activations(game_frame):
        if monitor:
            acts = monitor.get_current_activations()
            
            # If empty (first frame), try to use template with zeros
            if not acts and hasattr(monitor, 'template_activations') and monitor.template_activations:
                 # Create zero-fill using template structure
                 acts = {k: np.zeros_like(v) for k,v in monitor.template_activations.items()}
            
            # If still empty (warmup failed or no layers), visualize returns None
            act_frame = visualize_activations(acts, game_frame.shape[1])
            combined = combine_frame_and_activations(game_frame, act_frame)
            frames.append(combined)
            
            if acts and acts is not getattr(monitor, 'template_activations', None):
                 monitor.record_step()
        else:
            frames.append(game_frame)



    # 3. Run Evaluation Episodes (Collect Frames)
    agent.epsilon = 0 # No exploration during evaluation
    frames = []
    
    for ep in range(num_episodes):
        ep_idx = ep + 1
        
        # Reset environment
        env_state = env.reset() # Returns (row, col)
        
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        
        # Determine initial sensory state
        current_agent_pos = env.agent_pos
        current_danger_pos_list = []
        if env.is_danger:
             current_danger_pos_list = [env.resource_pos]
        if using_sensory:
            sensory_state = sensory_system.sense(current_agent_pos, env.resource_pos, current_danger_pos_list)

        if with_satiation:
            body_return = body.reset()
            if using_sensory:
                if isinstance(body_return, tuple):
                     state = (*sensory_state, *body_return)
                else:
                     state = (*sensory_state, body_return)
            else:
                if with_health:
                    satiation, health = body_return
                    state = (*env_state, satiation, health)
                    append_frame_with_activations(env.render_rgb_array(satiation, max_satiation, health, max_health, episode=ep_idx, step=0))
                else:
                    satiation = body_return
                    state = (*env_state, satiation)
                    append_frame_with_activations(env.render_rgb_array(satiation, max_satiation, episode=ep_idx, step=0))
            
            # Initial frame handling for POMDP?
            # Existing code only handled FOMDP rendering logic above for initial frame.
            # POMDP initial frame logic:
            if using_sensory and with_satiation:
                health = body.health if with_health else None
                max_h = body.max_health if with_health else None
                append_frame_with_activations(env.render_rgb_array(body.satiation, max_satiation, health, max_h, episode=ep_idx, step=0, sensory_data=sensory_system.get_visualization_data(sensory_state)))

        else:
            if using_sensory:
                state = sensory_state
                append_frame_with_activations(env.render_rgb_array(episode=ep_idx, step=0, sensory_data=sensory_system.get_visualization_data(sensory_state)))
            else:
                state = env_state
                append_frame_with_activations(env.render_rgb_array(episode=ep_idx, step=0))

        
        done = False
        step_count = 0
        
        # Preprocess if DQN
        if using_sensory:
            flat_state = preprocess_state(state)
        
        while not done and step_count < max_steps:
            if using_sensory:
                action = agent.choose_action(flat_state)
            else:
                action = agent.choose_action(state)
            
            next_env_state, _, env_done, info = env.step(action)
            
            # Observations
            current_agent_pos = env.agent_pos
            current_danger_pos_list = []
            if env.is_danger:
                 current_danger_pos_list = [env.resource_pos]
            
            if using_sensory:
                 next_sensory_state = sensory_system.sense(current_agent_pos, env.resource_pos, current_danger_pos_list)
            
            if with_satiation:
                body_return, _, body_done = body.step(info)
                done = env_done or body_done
                
                vis_data = None
                if using_sensory:
                     vis_data = sensory_system.get_visualization_data(next_sensory_state)
                     if isinstance(body_return, tuple):
                         next_state = (*next_sensory_state, *body_return)
                     else:
                         next_state = (*next_sensory_state, body_return)
                else:
                     # FOMDP logic
                     if with_health:
                        next_sat, next_health = body_return
                        next_state = (*next_env_state, next_sat, next_health)
                     else:
                        next_sat = body_return
                        next_state = (*next_env_state, next_sat)

                health = body.health if with_health else None
                max_h = body.max_health if with_health else None
                append_frame_with_activations(env.render_rgb_array(body.satiation, max_satiation, health, max_h, episode=ep_idx, step=step_count+1, sensory_data=vis_data))     
            else:
                done = env_done
                vis_data = None
                
                if using_sensory:
                    next_state = next_sensory_state
                    vis_data = sensory_system.get_visualization_data(next_sensory_state)
                else:
                    next_state = next_env_state

                append_frame_with_activations(env.render_rgb_array(episode=ep_idx, step=step_count+1, sensory_data=vis_data))

            
            state = next_state
            if using_sensory:
                flat_state = preprocess_state(state)
            step_count += 1
            
            if done:
                # Buffer end frames
                # Buffer end frames
                for _ in range(5):
                    if with_satiation:
                        if with_health:
                             # Use last known state values
                             append_frame_with_activations(env.render_rgb_array(body.satiation, max_satiation, body.health, max_health, episode=ep_idx, step=step_count, sensory_data=vis_data))
                        else:
                             append_frame_with_activations(env.render_rgb_array(body.satiation, max_satiation, episode=ep_idx, step=step_count, sensory_data=vis_data))
                    else:
                        append_frame_with_activations(env.render_rgb_array(episode=ep_idx, step=step_count, sensory_data=vis_data))

                break

    # 4. Generate Visual Artifacts
    
    # Plot Q-Table (Tabular only)
    if not using_sensory and hasattr(agent, 'q_table'):
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        vis_filename = os.path.join(plots_dir, f"q_table_vis_{pct}.png" if pct != "final" else "q_table_vis.png")
        plot_q_table(agent.q_table, vis_filename, resource_pos)
    
    # Save Video
    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_filename = os.path.join(videos_dir, f"video_{pct}.mp4" if pct != "final" else "final_trained_agent.mp4")
    save_video(frames, video_filename)

    # Save Activations
    if monitor:
        activations_file = os.path.join(videos_dir, f"activations_{pct}.npz" if pct != "final" else "activations_final.npz")
        monitor.save_history(activations_file)
        monitor.close()


def main():
    parser = argparse.ArgumentParser(description="GridWorld Evaluation")
    parser.add_argument("--seed", type=int, help="Override testing seed")
    parser.add_argument("--episodes", type=int, help="Number of episodes to evaluate")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    models_dir = os.path.join(results_dir, "models")
    config_path = os.path.join(models_dir, "config.yaml")

    # 1. Load saved configuration
    if not os.path.exists(config_path):
        print(f"Error: Training configuration file not found at {config_path}")
        print("Please run train.py first to generate a model and its configuration.")
        return

    print(f"Loading training configuration from {config_path}...")
    with open(config_path, 'r') as f:
        saved_config_dict = yaml.safe_load(f)
        config = Config(saved_config_dict)

    # 2. Key Overrides (Allow user to change testing seed/episodes)
    global_config = get_default_config()
    testing_seed = args.seed or global_config.get('testing.seed', 42)
    eval_episodes = args.episodes or global_config.get('testing.evaluation_episodes', 1)
    
    config.set('testing.seed', testing_seed)
    config.set('testing.evaluation_episodes', eval_episodes)
    
    # 3. Print Summary
    print("-" * 40)
    print(f"Evaluation Mode: {'Interoceptive' if config.get('body.with_satiation') else 'Conventional'}")
    print(f"Grid Size: {config.get('environment.height')}x{config.get('environment.width') if config.get('environment.width') else '?'}")
    print(f"Testing Seed: {testing_seed}")
    print(f"Num Episodes: {eval_episodes}")
    print("-" * 40)

    # 4. Find all checkpoints
    algorithm = config.get('agent.algorithm', "Tabular Q-Learning")
    
    if algorithm == "DQN":
        checkpoints = glob.glob(os.path.join(models_dir, "dqn_model_*.pth"))
        def extract_number(path):
            match = re.search(r"dqn_model_(\d+).pth", path)
            return int(match.group(1)) if match else -1
        final_model = os.path.join(models_dir, "dqn_model_final.pth")
    elif algorithm == "PPO":
        checkpoints = glob.glob(os.path.join(models_dir, "ppo_model_*.pth"))
        def extract_number(path):
            match = re.search(r"ppo_model_(\d+).pth", path)
            return int(match.group(1)) if match else -1
        final_model = os.path.join(models_dir, "ppo_model_final.pth")
    elif algorithm == "DRQN":
        checkpoints = glob.glob(os.path.join(models_dir, "drqn_model_*.pth"))
        def extract_number(path):
            match = re.search(r"drqn_model_(\d+).pth", path)
            return int(match.group(1)) if match else -1
        final_model = os.path.join(models_dir, "drqn_model_final.pth")
    elif algorithm == "RecurrentPPO":
        checkpoints = glob.glob(os.path.join(models_dir, "recurrent_ppo_model_*.pth"))
        def extract_number(path):
            match = re.search(r"recurrent_ppo_model_(\d+).pth", path)
            return int(match.group(1)) if match else -1
        final_model = os.path.join(models_dir, "recurrent_ppo_model_final.pth")
    elif algorithm == "DreamerV3":
        checkpoints = glob.glob(os.path.join(models_dir, "dreamer_model_*.pth"))
        def extract_number(path):
            match = re.search(r"dreamer_model_(\d+).pth", path)
            return int(match.group(1)) if match else -1
        final_model = os.path.join(models_dir, "dreamer_model_final.pth")
    else:
        checkpoints = glob.glob(os.path.join(models_dir, "q_table_*.npy"))
        def extract_number(path):
            match = re.search(r"q_table_(\d+).npy", path)
            return int(match.group(1)) if match else -1
        final_model = os.path.join(models_dir, "q_table.npy")
    
    checkpoints.sort(key=extract_number)
    
    if os.path.exists(final_model):
        checkpoints.append(final_model)

    if not checkpoints:
        print(f"No model checkpoints found in {models_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints. Starting evaluation...")

    # 5. Evaluate each checkpoint
    for checkpoint in checkpoints:
        evaluate_checkpoint(checkpoint, results_dir, config)

    print("-" * 40)
    print(f"Evaluation complete! Visualizations are in {results_dir}/plots/ and {results_dir}/videos/")

if __name__ == "__main__":
    main()
