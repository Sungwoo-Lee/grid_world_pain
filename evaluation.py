"""
Evaluation script for the GridWorld Reinforcement Learning agent.

This script:
1. Loads the configuration saved during training (`results/.../models/config.yaml`).
2. Sets up the evaluation environment (Grid, Body, Sensory) to match training.
3. Instantiates the appropriate agent (Tabular, DQN, PPO, DRQN, RecurrentPPO, DreamerV3).
4. Evaluates checkpoints:
   - Runs evaluation episodes (deterministic).
   - Collects frames for video generation.
   - Generates Q-table plots (if Tabular).
   - Saves artifacts to `results/.../RunName/`.

Arguments:
- `--results_dir <path>`: (Required) Path to the results directory of the run to evaluate.
- `--episodes <int>`: Override the number of evaluation episodes.
- `--seed <int>`: Override the random seed.
- `--checkpoint <str/int>`: Specific checkpoint to evaluate (e.g., `model_50.ckpt`, `50`).
- `--all`: Evaluate ALL checkpoints found in the directory.
- `--wandb-run-path <str>`: WandB run path (entity/project/run_id) to upload evaluation videos.

Usage Examples:

1. **Basic Evaluation (Latest Checkpoint)**:
   ```bash
   python evaluation.py --results_dir results/DQN/20260118-120000_my_run --episodes 5
   ```

2. **Evaluate Specific Checkpoint**:
   ```bash
   python evaluation.py --results_dir results/DQN/RunName --checkpoint 500
   ```

3. **Evaluate All Checkpoints & Upload to WandB**:
   ```bash
   python evaluation.py --results_dir results/DQN/RunName --all --wandb-run-path my_entity/my_project/run_id
   ```

Notes:
- Uses the configuration saved during training (`config.yaml`).
- **Strict Configuration**: Raises errors if required parameters are missing.
- Default behavior (no args): Evaluates the *latest* numeric checkpoint found.
"""
import os
import glob
import wandb
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
from src.utils.lrp_monitor import LRPMonitor
from src.utils.state_utils import FrameStacker
from src.utils.wandb_utils import wandb_login
import torch


def evaluate_checkpoint(checkpoint_path, results_dir, config, wandb_run_path=None):
    """
    Evaluates a single checkpoint:
    - Sets up environment and body based on config.
    - Loads agent.
    - Runs evaluation episodes to collect frames.
    - Generates Q-table plot and performance video.
    - Optionally uploads video to WandB.
    """
    import torch # Explicit import to fix UnboundLocalError
    filename = os.path.basename(checkpoint_path)
    # Create Data Dir
    data_dir = os.path.join(results_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Try Q-table pattern
    match = re.search(r"q_table_(\d+).npy", filename)
    if not match:
        match = re.search(r"dqn_model_(\d+).ckpt", filename)
    if not match:
        match = re.search(r"ppo_model_(\d+).ckpt", filename)
    if not match:
        match = re.search(r"drqn_model_(\d+).ckpt", filename)
    if not match:
        match = re.search(r"recurrent_ppo_model_(\d+).ckpt", filename)
    if not match:
        match = re.search(r"dreamer_model_(\d+).ckpt", filename)
        
    pct = match.group(1) if match else "unknown"
    
    print(f"Evaluating checkpoint: {filename} ({pct}%)")

    # 1. Component Extraction from Config
    with_satiation = config.get_mandatory('body.with_satiation')
    overeating_death = config.get_mandatory('body.overeating_death')
    max_steps = config.get_mandatory('environment.max_steps', int)
    seed = config.get_mandatory('testing.seed', int)
    num_episodes = config.get_mandatory('testing.evaluation_episodes', int)
    # Common Parameters
    resource_pos = config.get('environment.resource_pos')
    if resource_pos is None:
         # Fallback to food_pos if resource_pos missing (legacy compatibility)
         resource_pos = config.get_mandatory('environment.food_pos')
         
    height = config.get_mandatory('environment.height', int)
    width = config.get_mandatory('environment.width', int)
    
    max_satiation = config.get_mandatory('body.max_satiation', int)
    start_satiation = config.get_mandatory('body.start_satiation', int)
    random_start_satiation = config.get_mandatory('body.random_start_satiation')
    food_satiation_gain = config.get_mandatory('body.food_satiation_gain', float)
    use_homeostatic_reward = config.get_mandatory('body.use_homeostatic_reward')
    satiation_setpoint = config.get_mandatory('body.satiation_setpoint', float)
    death_penalty = config.get_mandatory('body.death_penalty', float)
    
    # New Health/Pain Params (Check if body.with_health is present, strictly)
    with_health = config.get_mandatory('body.with_health')
    
    # If with_health is True, we enforce params. If False, maybe optional or safe defaults?
    # User said strict. Config file should have them regardless if generated by strict train.py.
    # But basic config might not have them? `environment.yaml` has defaults.
    # I will be strict.
    max_health = config.get_mandatory('body.max_health', float)
    start_health = config.get_mandatory('body.start_health', float)
    health_recovery = config.get_mandatory('body.health_recovery', float)
    start_health_random = config.get_mandatory('body.start_health_random')
    
    prob_switch_to_danger = config.get_mandatory('environment.prob_switch_to_danger', float)
    min_danger_duration = config.get_mandatory('environment.min_danger_duration', int)
    damage_amount = config.get_mandatory('environment.damage_amount', float)
    
    prob_switch_to_food = config.get_mandatory('environment.prob_switch_to_food', float)
    min_food_duration = config.get_mandatory('environment.min_food_duration', int)
    
    # Extract Relocation Config
    relocate_resource = config.get_mandatory('environment.relocate_resource')
    relocation_steps = config.get_mandatory('environment.relocation_steps', int)

    # 2. Environment & Body Setup
    # Set seed for deterministic evaluation
    np.random.seed(seed)
    
    env = GridWorld(height=height, width=width, start=tuple(config.get_mandatory('environment.start_pos')), 
                    resource_pos=resource_pos, with_satiation=with_satiation, max_steps=max_steps,
                    prob_switch_to_danger=prob_switch_to_danger, min_danger_duration=min_danger_duration, damage_amount=damage_amount,
                    prob_switch_to_food=prob_switch_to_food, min_food_duration=min_food_duration,
                    relocate_resource=relocate_resource, relocation_steps=relocation_steps,
                    vector_size=config.get_mandatory('sensory.vector_size', int),
                    food_property=config.get_mandatory('sensory.food_property'),
                    danger_property=config.get_mandatory('sensory.danger_property'))
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
    using_sensory = config.get_mandatory('sensory.using_sensory')
    sensory_system = None
    if using_sensory:
        sensor_radius = config.get_mandatory('sensory.sensor_radius', int)
        decay_power = config.get_mandatory('sensory.decay_power', float)
        vector_size = config.get_mandatory('sensory.vector_size', int)
        nociception_enabled = config.get_mandatory('sensory.nociception_enabled', bool)
        nociceptor_radius = config.get_mandatory('sensory.nociceptor_radius', int)
        collision_sensor_enabled = config.get_mandatory('sensory.collision_sensor_enabled', bool)
        collision_sensor_range = config.get_mandatory('sensory.collision_sensor_range', int)
        location_sensor = config.get_mandatory('sensory.location_sensor')
        
        sensory_system = SensorySystem(
            sensor_radius=sensor_radius, 
            vector_size=vector_size, 
            decay_power=decay_power, 
            nociceptor_radius=nociceptor_radius,
            location_sensor=location_sensor,
            nociception_enabled=nociception_enabled,
            collision_sensor_enabled=collision_sensor_enabled,
            collision_sensor_range=collision_sensor_range
        )


    # Determine Input Dimension using Specs
    input_dim = 0
    dims_breakdown = []

    # Log Specs
    print("\n--- RL API Specifications (Evaluation) ---")
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    print(f"Environment Action Spec: {action_spec}")

    if using_sensory:
         sensory_spec = sensory_system.observation_spec()
         # Pretty print
         import json
         print(f"Sensory System Spec: {sensory_spec}")
         
         for key, val in sensory_spec.items():
             dim = val['shape'][0]
             input_dim += dim
             dims_breakdown.append(f"{key}={dim}")
    else:
         print(f"Environment Observation Spec: {observation_spec}")
         loc_shape = observation_spec['loc']['shape']
         input_dim += loc_shape[0]
         dims_breakdown.append(f"Loc={loc_shape[0]}")
         
    if with_satiation:
        sat_shape = observation_spec['satiation']['shape']
        input_dim += sat_shape[0]
        dims_breakdown.append(f"Sat={sat_shape[0]}")
        if with_health:
             hlth_shape = observation_spec['health']['shape']
             input_dim += hlth_shape[0]
             dims_breakdown.append(f"Health={hlth_shape[0]}")
    
    print("--------------------------------------------\n")


    # Frame Stacking Logic
    frame_stack = config.get_mandatory('agent.frame_stack', int)
    base_input_dim = input_dim
    
    if frame_stack > 1:
        print(f"Input Dimension: Base: {base_input_dim} x Stack: {frame_stack}")
    else:
        print(f"Input Dimension: {input_dim}")
    
    # Initialize Agent
    agent = None
    algorithm = config.get_mandatory('agent.algorithm')
    
    device = config.get_mandatory('training.device')
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{algorithm} Agent using device: {device}")
    
    if algorithm == "DQN":
        from src.models.dqn import DQNAgent
        agent = DQNAgent(
            state_dim=base_input_dim, 
            action_dim=5,
            frame_stack=frame_stack,
            lr=config.get_mandatory('agent.learning_rate', float),
            gamma=config.get_mandatory('agent.gamma', float),
            buffer_size=config.get_mandatory('agent.buffer_size', int),
            batch_size=config.get_mandatory('agent.batch_size', int),
            epsilon_start=0.0, # Eval override
            epsilon_end=0.0,
            epsilon_decay=1.0,
            target_update_freq=config.get_mandatory('agent.target_update_freq', int),
            fc_layers=config.get_mandatory('agent.fc_layers'),
            device=device
        )
        agent.load(checkpoint_path, weights_only=True)

    elif algorithm == "DRQN":
        from src.models.drqn import DRQNAgent
        agent = DRQNAgent(
            state_dim=base_input_dim, 
            action_dim=5, 
            lr=config.get_mandatory('agent.learning_rate', float),
            gamma=config.get_mandatory('agent.gamma', float),
            buffer_size=config.get_mandatory('agent.buffer_size', int),
            batch_size=config.get_mandatory('agent.batch_size', int),
            trace_length=config.get_mandatory('agent.trace_length', int),
            burn_in_length=config.get_mandatory('agent.burn_in_length', int),
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            target_update_freq=config.get_mandatory('agent.target_update_freq', int),
            fc_layers=config.get_mandatory('agent.fc_layers'),
            recurrent_layers=config.get_mandatory('agent.recurrent_layers'),
            device=device
        )
        agent.load(checkpoint_path, weights_only=True)

    elif algorithm == "PPO":
        from src.models.ppo import PPOAgent
        agent = PPOAgent(
            state_dim=base_input_dim, 
            action_dim=5,
            frame_stack=frame_stack, 
            lr_actor=config.get_mandatory('agent.lr_actor', float),
            lr_critic=config.get_mandatory('agent.lr_critic', float),
            gamma=config.get_mandatory('agent.gamma', float),
            K_epochs=config.get_mandatory('agent.K_epochs', int),
            eps_clip=config.get_mandatory('agent.eps_clip', float),
            update_timestep=config.get_mandatory('agent.update_timestep', int),
            entropy_coef=config.get_mandatory('agent.entropy_coef', float),
            actor_fc_layers=config.get_mandatory('agent.actor_fc_layers'),
            critic_fc_layers=config.get_mandatory('agent.critic_fc_layers'),
            device=device
        )
        agent.load(checkpoint_path, weights_only=True)


    elif algorithm == "RecurrentPPO":
        from src.models.recurrent_ppo import RecurrentPPOAgent
        agent = RecurrentPPOAgent(
            state_dim=base_input_dim, 
            action_dim=5, 
            lr_actor=config.get_mandatory('agent.lr_actor', float), 
            lr_critic=config.get_mandatory('agent.lr_critic', float), 
            gamma=config.get_mandatory('agent.gamma', float), 
            K_epochs=config.get_mandatory('agent.K_epochs', int), 
            eps_clip=config.get_mandatory('agent.eps_clip', float), 
            update_timestep=config.get_mandatory('agent.update_timestep', int), 
            sequence_length=config.get_mandatory('agent.sequence_length', int), 
            entropy_coef=config.get_mandatory('agent.entropy_coef', float),
            fc_layers=config.get_mandatory('agent.fc_layers'),
            recurrent_layers=config.get_mandatory('agent.recurrent_layers'),
            actor_fc_layers=config.get_mandatory('agent.actor_fc_layers'),
            critic_fc_layers=config.get_mandatory('agent.critic_fc_layers'),
            device=device
        )
        agent.load(checkpoint_path, weights_only=True)

    elif algorithm == "DreamerV3":
        from src.models.dreamer_v3 import DreamerV3Agent
        agent = DreamerV3Agent(
            state_dim=base_input_dim,
            action_dim=5,
            device=device,
            batch_size=config.get_mandatory('agent.batch_size', int),
            batch_length=config.get_mandatory('agent.batch_length', int),
            model_lr=config.get_mandatory('agent.model_lr', float),
            actor_lr=config.get_mandatory('agent.actor_lr', float),
            value_lr=config.get_mandatory('agent.value_lr', float),
            encoder_dim=config.get_mandatory('agent.encoder_dim', int),
            encoder_fc_layers=config.get_mandatory('agent.encoder_fc_layers'),
            rssm_deter_dim=config.get_mandatory('agent.rssm_deter_dim', int),
            rssm_stoch_dim=config.get_mandatory('agent.rssm_stoch_dim', int),
            rssm_classes=config.get_mandatory('agent.rssm_classes', int),
            decoder_fc_layers=config.get_mandatory('agent.decoder_fc_layers'),
            reward_fc_layers=config.get_mandatory('agent.reward_fc_layers'),
            continue_fc_layers=config.get_mandatory('agent.continue_fc_layers'),
            actor_fc_layers=config.get_mandatory('agent.actor_fc_layers'),
            critic_fc_layers=config.get_mandatory('agent.critic_fc_layers')
        )
        agent.load(checkpoint_path, weights_only=True)

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

    # 3. Run Evaluation via Core Utility
    from src.utils.evaluation_core import evaluate_agent
    
    evaluate_agent(
        agent=agent,
        env=env,
        body=body,
        sensory_system=sensory_system,
        config=config,
        num_episodes=num_episodes,
        device=device,
        results_dir=results_dir,
        checkpoint_pct=pct,
        wandb_run_path=wandb_run_path # Pass the path, or let it fallback to active run if main() inited it
    )

    # 4. Generate Visual Artifacts (Tabular Q-Table only, Video handled by evaluate_agent)
    
    # Plot Q-Table (Tabular only)
    if not using_sensory and hasattr(agent, 'q_table'):
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True) # Ensure directory exists
        # Visualization Plotting
        if algorithm == "Tabular Q-Learning":
            vis_filename = os.path.join(plots_dir, f"q_table_{pct}.png")
            plot_q_table(agent.q_table, vis_filename, config, resource_pos)
            
    # Video and Activations are handled by evaluate_agent now.
    # LRP is handled by evaluate_agent now.




def main():
    parser = argparse.ArgumentParser(description="GridWorld Evaluation")
    parser.add_argument("--seed", type=int, help="Override testing seed")
    parser.add_argument("--episodes", type=int, help="Number of episodes to evaluate")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory (Required)")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint name or path to evaluate (e.g. 'model_100.ckpt' or full path)")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints found in the directory")
    parser.add_argument("--wandb-run-path", type=str, help="WandB run path (e.g. 'entity/project/run_id') to upload evaluation videos")
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
    # 2. Key Overrides (Allow user to change testing seed/episodes)
    # Load evaluation defaults since environment.yaml no longer has them
    eval_default_path = "configs/evaluation/default.yaml"
    if os.path.exists(eval_default_path):
        eval_defaults = Config.load_yaml(eval_default_path)
    else:
        eval_defaults = Config() # Empty if missing
        
    # Determine params: CLI > Default Config > Hardcoded Fallback
    testing_seed = args.seed or eval_defaults.get('testing.seed')
    if testing_seed is None:
        raise ValueError("Strict Config: 'testing.seed' must be provided via CLI or configs/evaluation/default.yaml")
    
    eval_episodes = args.episodes or eval_defaults.get('testing.evaluation_episodes')
    if eval_episodes is None:
        raise ValueError("Strict Config: 'testing.evaluation_episodes' must be provided via CLI or configs/evaluation/default.yaml")
    
    config.set('testing.seed', testing_seed)
    config.set('testing.evaluation_episodes', eval_episodes)
    
    # 3. Print Summary
    # Print Info
    print(f"\nEvaluation Mode: {'Interoceptive' if config.get_mandatory('body.with_satiation') else 'Conventional'}")
    print(f"Grid Size: {config.get_mandatory('environment.height')}x{config.get_mandatory('environment.width')}")
    print(f"Testing Seed: {testing_seed}")
    print(f"Num Episodes: {eval_episodes}")
    print("-" * 40)

    # 4. Find all checkpoints
    # 4. Find checkpoints
    algorithm = config.get_mandatory('agent.algorithm')
    
    # Helper to clean model naming
    prefix = ""
    if algorithm == "DQN": prefix = "dqn_model_"
    elif algorithm == "PPO": prefix = "ppo_model_"
    elif algorithm == "DRQN": prefix = "drqn_model_"
    elif algorithm == "RecurrentPPO": prefix = "recurrent_ppo_model_"
    elif algorithm == "DreamerV3": prefix = "dreamer_model_"
    else: prefix = "q_table_"

    ext = ".npy" if algorithm == "Tabular Q-Learning" else ".ckpt"

    checkpoints = []
    
    if args.checkpoint:
        # User specified a specific checkpoint
        ckpt_arg = args.checkpoint
        
        # Check 1: Is it a full path?
        if os.path.exists(ckpt_arg):
             checkpoints.append(ckpt_arg)
        else:
             # Check 2: specific name in models_dir
             ckpt_path = os.path.join(models_dir, ckpt_arg)
             if os.path.exists(ckpt_path):
                 checkpoints.append(ckpt_path)
             else:
                 # Check 3: maybe just the number? e.g. "100"
                 ckpt_name = f"{prefix}{ckpt_arg}{ext}"
                 ckpt_path = os.path.join(models_dir, ckpt_name)
                 if os.path.exists(ckpt_path):
                     checkpoints.append(ckpt_path)
                 else:
                     print(f"Error: Specified checkpoint '{args.checkpoint}' not found.")
                     return

    elif args.all:
        # Evaluate ALL found
        if algorithm == "Tabular Q-Learning":
            checkpoints = glob.glob(os.path.join(models_dir, f"{prefix}*{ext}"))
        else:
            checkpoints = glob.glob(os.path.join(models_dir, f"{prefix}*{ext}"))
            
        # Sort by number
        def extract_number(path):
            filename = os.path.basename(path)
            # Match number
            match = re.search(rf"{prefix}(\d+){ext}", filename)
            if match:
                return int(match.group(1))
            return -1
        
        checkpoints.sort(key=extract_number)
        
    else:
        # Default: Evaluate LATEST numeric checkpoint
        # Find latest numeric
        all_ckpts = glob.glob(os.path.join(models_dir, f"{prefix}*{ext}"))
        if all_ckpts:
            def extract_number(path):
                    match = re.search(rf"{prefix}(\d+){ext}", os.path.basename(path))
                    return int(match.group(1)) if match else -1
            
            # Filter out any that didn't match (e.g. if some other file exists)
            valid_ckpts = [c for c in all_ckpts if extract_number(c) != -1]
            
            if valid_ckpts:
                latest = max(valid_ckpts, key=extract_number)
                checkpoints.append(latest)
    
    if not checkpoints:
         print(f"No valid checkpoints found in {models_dir}")
         return
         
    print(f"Found {len(checkpoints)} checkpoint(s). Starting evaluation...")

    
    # 5. Initialize WandB if requested
    if args.wandb_run_path:
        try:
            print(f"Initializing WandB run: {args.wandb_run_path}...")
            path_parts = args.wandb_run_path.strip().split('/')
            entity = None
            project = None
            run_id = None
            
            if len(path_parts) == 3:
                entity, project, run_id = path_parts
            elif len(path_parts) == 2:
                project, run_id = path_parts
            else:
                run_id = path_parts[0]
                
            if run_id:
                 wandb_login(quiet=False)
                 wandb.init(
                    entity=entity,
                    project=project,
                    id=run_id,
                    resume="must",
                    job_type="evaluation"
                )
            else:
                print(f"Error: Could not parse run ID from {args.wandb_run_path}. Upload skipped.")
                args.wandb_run_path = None # Disable upload
                
        except Exception as e:
             print(f"Error initializing WandB: {e}")
             args.wandb_run_path = None

    # 6. Evaluate each checkpoint
    for checkpoint in checkpoints:
        evaluate_checkpoint(checkpoint, results_dir, config, wandb_run_path=args.wandb_run_path)
    
    if wandb.run:
        wandb.finish()

    print("-" * 40)
    print(f"Evaluation complete! Visualizations are in {results_dir}/plots/ and {results_dir}/videos/")

if __name__ == "__main__":
    main()
