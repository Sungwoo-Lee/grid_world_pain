
import os
import numpy as np
import torch
import wandb
from src.utils.visualization import save_video, visualize_activations, combine_frame_and_activations
from src.utils.activation_monitor import ActivationMonitor
from src.utils.lrp_monitor import LRPMonitor
from src.utils.state_utils import FrameStacker, preprocess_state
from src.utils.wandb_utils import upload_video

def evaluate_agent(
    agent, 
    env, 
    body, 
    sensory_system, 
    config, 
    num_episodes=1, 
    device="cpu", 
    results_dir=None, 
    checkpoint_pct=None, 
    wandb_run_path=None,
    quiet=False
):
    """
    Evaluates the agent for a given number of episodes.
    Generates video with activations/LRP if configured.
    Uploads to WandB if active or run path provided.
    
    Args:
        agent: The RL agent instance.
        env: The GridWorld environment instance.
        body: The InteroceptiveBody instance (or None if conventional logic handled).
        sensory_system: SensorySystem instance (or None).
        config: Config object.
        num_episodes: Number of episodes to run (default 1).
        device: 'cpu' or 'cuda'.
        results_dir: Directory to save videos/plots.
        checkpoint_pct: Checkpoint number/percentage/epoch (int or str) for labeling.
        wandb_run_path: Optional WandB run path to upload to.
        quiet: If True, suppresses non-critical print output.
        
    Returns:
        video_filename (str): Path to generated video, or None.
    """
    
    # Unpack Key Configs
    using_sensory = config.get_mandatory('sensory.using_sensory')
    with_satiation = config.get_mandatory('body.with_satiation')
    with_injury = config.get_mandatory('body.with_injury')
    proprioception_enabled = config.get_mandatory('sensory.proprioception_enabled', bool)
    
    max_satiation = config.get_mandatory('body.max_satiation', int) if with_satiation else None
    max_injury = config.get_mandatory('body.max_injury', float) if with_injury else None
    max_steps = config.get_mandatory('environment.max_steps', int)
    
    # Visualization Config
    vis_enabled = config.get_mandatory('visualization.enabled')
    vis_activations = config.get_mandatory('visualization.activations.enabled') and vis_enabled
    vis_lrp = config.get_mandatory('visualization.activations.with_lrp') and vis_enabled
    vis_lrp = config.get_mandatory('visualization.activations.with_lrp') and vis_enabled
    vis_fps = config.get_mandatory('visualization.fps', int)
    
    location_sensor = config.get('sensory.location_sensor', False)
    
    # Setup Activation/LRP Monitors
    monitor = None
    lrp_monitor = None
    input_structure = []
    
    # Capture Frames Closure
    frames = []

    def append_frame_with_activations(game_frame, action=None, state=None):
        nonlocal frames
        act_frame = None
        acts = None
        if monitor:
            acts = monitor.get_current_activations()
            
            # Template fallback
            if not acts and hasattr(monitor, 'template_activations') and monitor.template_activations:
                    acts = {k: np.zeros_like(v) for k,v in monitor.template_activations.items()}
            
            # LRP
            attributions = None
            if lrp_monitor and action is not None and state is not None:
                try:
                    input_tensor = None
                    if using_sensory:
                        if isinstance(state, np.ndarray):
                            input_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    else:
                        flat = preprocess_state(state, env.height, env.width, max_satiation, max_injury)
                        input_tensor = torch.FloatTensor(flat).unsqueeze(0).to(device)
                        
                    algorithm_name = type(agent).__name__
                    if "Recurrent" in algorithm_name or "DRQN" in algorithm_name or "Dreamer" in algorithm_name or "LSTM" in algorithm_name:
                         if input_tensor is not None and input_tensor.ndim == 2:
                             input_tensor = input_tensor.unsqueeze(1) # (Batch, Seq, Dim)

                    if input_tensor is not None:
                        attributions = lrp_monitor.compute_relevance(input_tensor, action)
                except Exception as e:
                    # print(f"LRP Error: {e}") 
                    pass

            # Visualize
            act_frame = visualize_activations(
                acts, 
                game_frame.shape[1], 
                config, 
                input_structure=input_structure, 
                attributions=attributions
            )

        combined = combine_frame_and_activations(game_frame, act_frame)
        frames.append(combined)
    
        if acts and monitor and acts is not getattr(monitor, 'template_activations', None):
                monitor.record_step()

    # Initialize Monitors
    algorithm = config.get_mandatory('agent.algorithm')
    if algorithm != "Tabular Q-Learning" and vis_activations:
        model_to_monitor = None
        if isinstance(agent, torch.nn.Module):
            model_to_monitor = agent
        elif hasattr(agent, 'policy_net'):
            model_to_monitor = agent.policy_net
        elif hasattr(agent, 'policy'):
            model_to_monitor = agent.policy
            
        if model_to_monitor:
            if not quiet: print(f"  Monitoring activations for {type(model_to_monitor).__name__}...")
            # We track Linear and Conv layers primarily
            monitor = ActivationMonitor(model_to_monitor, tracked_layers=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM, torch.nn.GRU])
            
            # Warm-up (Required to capture structure)
            try:
                dummy_input = None
                for module in model_to_monitor.modules():
                     if isinstance(module, torch.nn.Linear):
                         dummy_input = torch.zeros(1, module.in_features).to(device)
                         break
                
                if dummy_input is not None:
                     with torch.no_grad():
                          if hasattr(agent, 'reset_hidden'): agent.reset_hidden()
                          is_recurrent = "DRQN" in type(agent).__name__ or "Recurrent" in type(agent).__name__ or "Dreamer" in type(agent).__name__
                          
                          if is_recurrent:
                               dummy_input = dummy_input.unsqueeze(0) 
                               model_to_monitor(dummy_input)
                               if hasattr(agent, 'reset_hidden'): agent.reset_hidden()
                          else:
                               model_to_monitor(dummy_input)
                     
                     monitor.template_activations = monitor.get_current_activations().copy()
                     monitor.clear_history() 
            except Exception as e:
                print(f"  Activation warm-up failed (non-fatal): {e}")

            # Input Structure for Visualization Labeling
            if using_sensory and sensory_system:
                input_structure.append(("Sensory", sensory_system.vector_size))
                # Add Collision/Nociceptor placeholders if needed?
                # But mostly adding Location if requested
                if location_sensor:
                    input_structure.append(("Loc", 2))
            else:
                input_structure.append(("Agent", 2)) 

            if with_satiation:
                input_structure.append(("Sat", 1))
                if with_injury:
                    input_structure.append(("Inj", 1))

            # Initialize LRP
            if vis_lrp:
                try:
                    target_net = model_to_monitor
                    if algorithm == "PPO":
                        target_net = model_to_monitor.actor
                    elif algorithm in ["DRQN", "DreamerV3"]:
                        class OutputWrapper(torch.nn.Module):
                             def __init__(self, model, index=0):
                                 super().__init__()
                                 self.model = model
                                 self.index = index
                             def forward(self, x):
                                 return self.model(x)[self.index]
                        target_net = OutputWrapper(model_to_monitor, 0)
                    
                    lrp_monitor = LRPMonitor(target_net)
                except Exception as e:
                    print(f"Failed to initialize LRP: {e}")
    
    # ---------------------------------------------------------
    # Execution Loop
    # ---------------------------------------------------------
    
    # Frame Stacker Re-Init for Evaluation
    input_dim = 0 
    if using_sensory and sensory_system: input_dim += sensory_system.vector_size + 1
    else: input_dim += 2
    if with_satiation: input_dim += 1
    if with_injury: input_dim += 1
    
    frame_stack = config.get('agent.frame_stack', 1)
    # Important: FrameStacker expects base_input_dim
    # Wait, preprocess_state returns base_input_dim size.
    stacker = FrameStacker(input_dim=input_dim, stack_size=frame_stack)

    # Exploration Off
    original_epsilon = agent.epsilon if hasattr(agent, 'epsilon') else 0
    if hasattr(agent, 'epsilon'): agent.epsilon = 0

    try:
        for ep in range(num_episodes):
            ep_idx = ep + 1
            
            # Reset Environment
            env_state = env.reset()
            if hasattr(agent, 'reset_hidden'): agent.reset_hidden()
            
            # Initial Observations
            current_agent_pos = env.agent_pos
            if using_sensory and sensory_system:
                resources = env.get_active_resources()
                extra_data = {
                    'injury_level': body.injury_level if with_satiation else 0, 
                    'max_injury': body.max_injury if with_satiation else 1,
                    'satiation': body.satiation if with_satiation else 0,
                    'max_satiation': body.max_satiation if with_satiation else 1
                }
                sensory_dict = sensory_system.sense(
                    current_agent_pos, resources,
                    grid_height=env.height, grid_width=env.width,
                    extra_data=extra_data
                )

            if with_satiation:
                body_return = body.reset()
                state = {}
                # Strictly separate sensory and loc
                if using_sensory:
                    if sensory_system: 
                        state.update(sensory_dict)
                        if location_sensor:
                            state['loc'] = current_agent_pos
                else:
                    state['loc'] = env_state
                
                if isinstance(body_return, tuple):
                     state['satiation'] = body_return[0]
                     state['injury'] = body_return[1]
                else:
                     state['satiation'] = body_return
            else:
                if using_sensory and sensory_system:
                    state = sensory_dict.copy()
                    if location_sensor:
                        state['loc'] = current_agent_pos
                else:
                    state = {'loc': env_state}
            
            # Initial Frame
            vis_data = None
            if using_sensory and sensory_system:
                 vis_data = sensory_system.get_visualization_data(sensory_dict)
            
            # Render Helper
            # Need to extract body state for render
            sat_val = state.get('satiation', 0)
            injury_val = state.get('injury')
            
            # Arg, render_rgb_array signature varies based on satiation?
            # env.render_rgb_array signature:
            # def render_rgb_array(self, satiation=None, max_satiation=None, injury=None, max_injury=None, episode=0, step=0, sensory_data=None, action=None):
            
            frame = env.render_rgb_array(
                satiation=sat_val if with_satiation else None, 
                max_satiation=max_satiation, 
                injury=injury_val, 
                max_injury=max_injury, 
                episode=ep_idx, 
                step=0, 
                sensory_data=vis_data
            )
            append_frame_with_activations(frame, state=state)
            
            # Loop
            done = False
            step_count = 0
            
            # Initialize previous action (4 = Stay, representing initial stationary state)
            previous_action = 4 if proprioception_enabled else None
            
            # Preprocess
            flat_state = preprocess_state(state, env.height, env.width, max_satiation, max_injury)
            state_array = stacker.reset(flat_state).flatten()
            
            # Tabular State Handling
            if algorithm == "Tabular Q-Learning":
                 tabular_list = []
                 if 'loc' in state:
                      tabular_list.extend(state['loc'])
                 if with_satiation:
                      tabular_list.append(state.get('satiation', 0))
                 if with_injury:
                      tabular_list.append(state.get('injury', 0))
                 state_array = tuple(int(x) for x in tabular_list)
                 
            while not done and step_count < max_steps:
                # Check if choose_action supports eval_mode
                import inspect
                sig = inspect.signature(agent.choose_action)
                if 'eval_mode' in sig.parameters:
                    action = agent.choose_action(state_array, eval_mode=True)
                else:
                    action = agent.choose_action(state_array)
                
                next_env_state, _, env_done, info = env.step(action)
                current_agent_pos = env.agent_pos
                
                if using_sensory and sensory_system:
                     resources = env.get_active_resources()
                     extra_data = {
                         'injury_level': body.injury_level if with_satiation else 0, 
                         'max_injury': body.max_injury if with_satiation else 1,
                         'satiation': body.satiation if with_satiation else 0,
                         'max_satiation': body.max_satiation if with_satiation else 1
                     }
                     next_sensory_dict = sensory_system.sense(
                         current_agent_pos, resources,
                         grid_height=env.height, grid_width=env.width,
                         extra_data=extra_data
                     )
                
                if with_satiation:
                    body_return, _, body_done = body.step(info)
                    done = env_done or body_done
                    
                    next_state = {}
                    if using_sensory:
                        if sensory_system: 
                            next_state.update(next_sensory_dict)
                    else:
                        next_state['loc'] = next_env_state
                        
                    if isinstance(body_return, tuple):
                         next_state['satiation'] = body_return[0]
                         next_state['injury'] = body_return[1]
                    else:
                         next_state['satiation'] = body_return
                else:
                    done = env_done
                    if using_sensory and sensory_system:
                        next_state = next_sensory_dict.copy()
                    else:
                        next_state = {'loc': next_env_state}
                        
                # Render
                vis_data = None
                if using_sensory and sensory_system:
                     vis_data = sensory_system.get_visualization_data(next_sensory_dict)

                
                sat_val = next_state.get('satiation', 0)
                injury_val = next_state.get('injury')
                
                frame = env.render_rgb_array(
                    satiation=sat_val if with_satiation else None,
                    max_satiation=max_satiation,
                    injury=injury_val,
                    max_injury=max_injury,
                    episode=ep_idx,
                    step=step_count+1,
                    sensory_data=vis_data,
                    action=action
                )
                
                # We explain the action taken at `state` (inputs used to generate action)
                # But append_frame usually attaches to the RESULTING frame.
                # Logic: We see frame T+1. We want to see activations that caused transition T -> T+1.
                l_state = flat_state if using_sensory else state
                append_frame_with_activations(frame, action=action, state=l_state)
                
                state = next_state
                # Use current action as "previous action" for next state
                flat_next = preprocess_state(next_state, env.height, env.width, max_satiation, max_injury)
                
                # Update previous action for next iteration
                if proprioception_enabled:
                    previous_action = action
                
                if algorithm != "Tabular Q-Learning":
                     state_array = stacker.step(flat_next).flatten()
                     flat_state = flat_next
                else:
                     tabular_list = []
                     if 'loc' in next_state:
                          tabular_list.extend(next_state['loc'])
                     if with_satiation:
                          tabular_list.append(next_state.get('satiation', 0))
                     if with_injury:
                          tabular_list.append(next_state.get('injury', 0))
                     state_array = tuple(int(x) for x in tabular_list)
                     
                step_count += 1
                
                if done:
                    # Buffer frames
                    for _ in range(5):
                        frame = env.render_rgb_array(
                            satiation=sat_val if with_satiation else None,
                            max_satiation=max_satiation,
                            injury=injury_val,
                            max_injury=max_injury,
                            episode=ep_idx,
                            step=step_count,
                            sensory_data=vis_data
                        )
                        append_frame_with_activations(frame)
                    break
    finally:
        # Restore Epsilon
        if hasattr(agent, 'epsilon'): agent.epsilon = original_epsilon
        if monitor: monitor.close()

    # Save Video
    video_filename = None
    if results_dir and checkpoint_pct is not None:
         videos_dir = os.path.join(results_dir, "videos")
         os.makedirs(videos_dir, exist_ok=True)
         video_filename = os.path.join(videos_dir, f"video_{checkpoint_pct}.mp4")
         save_video(frames, video_filename, fps=vis_fps, quiet=quiet)
         
         if wandb_run_path or wandb.run:
             upload_video(video_filename, run_path=wandb_run_path, episode=checkpoint_pct, caption=f"Eval Video {checkpoint_pct}", quiet=quiet)
             
    return video_filename
