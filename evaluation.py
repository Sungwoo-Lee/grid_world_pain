"""
Evaluation script for the GridWorld Reinforcement Learning agent.

This script:
1. Loads the configuration saved during training (results/models/config.yaml).
2. Sets up the evaluation environment (Grid, Body, Agent).
3. Evaluates checkpoints:
   - Runs evaluation episodes.
   - Collects frames and Q-tables.
   - Uses visualization utilities to generate plots and videos.

Usage:
    python evaluation.py [--seed SEED] [--results_dir DIR]
"""
import os
import glob
import re
import yaml
import numpy as np
import argparse
from src.grid_world_pain import GridWorld
from src.grid_world_pain.body import InteroceptiveBody
from src.grid_world_pain.agent import QLearningAgent
from src.grid_world_pain.config import Config, get_default_config
from src.grid_world_pain.visualization import plot_q_table, save_video

def evaluate_checkpoint(checkpoint_path, results_dir, config):
    """
    Evaluates a single checkpoint:
    - Sets up environment and body based on config.
    - Loads agent.
    - Runs evaluation episode to collect frames.
    - Generates Q-table plot and performance video.
    """
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"q_table_(\d+).npy", filename)
    pct = match.group(1) if match else ("final" if filename == "q_table.npy" else "unknown")
    
    print(f"Evaluating checkpoint: {filename} ({pct}%)")

    # 1. Component Extraction from Config
    with_satiation = config.get('body.with_satiation', True)
    overeating_death = config.get('body.overeating_death', True)
    max_steps = config.get('environment.max_steps', 100)
    seed = config.get('testing.seed', 42)
    food_pos = config.get('environment.food_pos', [4, 4])
    height = config.get('environment.height', 5)
    width = config.get('environment.width', 5)
    max_satiation = config.get('body.max_satiation', 20)
    start_satiation = config.get('body.start_satiation', 10)
    random_start_satiation = config.get('body.random_start_satiation', True)

    # 2. Environment & Body Setup
    # Set seed for deterministic evaluation
    np.random.seed(seed)
    
    env = GridWorld(height=height, width=width, food_pos=food_pos, with_satiation=with_satiation, max_steps=max_steps)
    body = InteroceptiveBody(max_satiation=max_satiation, start_satiation=start_satiation, overeating_death=overeating_death, random_start_satiation=random_start_satiation)
    
    # Mock composite env for agent
    class CompositeEnv:
        def __init__(self, env, body):
            self.height = env.height
            self.width = env.width
            self.max_satiation = body.max_satiation
            
    agent = QLearningAgent(CompositeEnv(env, body), with_satiation=with_satiation)
    
    try:
        agent.load(checkpoint_path)
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return

    # 3. Run Evaluation Episode (Collect Frames)
    agent.epsilon = 0 # No exploration during evaluation
    frames = []
    
    env_state = env.reset()
    if with_satiation:
        body_state = body.reset()
        state = (*env_state, body_state)
        frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=1, step=0))
    else:
        state = env_state
        frames.append(env.render_rgb_array(episode=1, step=0))
    
    done = False
    step_count = 0
    while not done and step_count < max_steps:
        action = agent.choose_action(state)
        next_env_state, _, env_done, info = env.step(action)
        
        if with_satiation:
            next_body_state, _, done = body.step(info)
            next_state = (*next_env_state, next_body_state)
            frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=1, step=step_count+1))
        else:
            done = env_done
            next_state = next_env_state
            frames.append(env.render_rgb_array(episode=1, step=step_count+1))
        
        state = next_state
        step_count += 1
        
        if done:
            # Buffer end frames
            for _ in range(5):
                if with_satiation:
                    frames.append(env.render_rgb_array(body.satiation, body.max_satiation, episode=1, step=step_count))
                else:
                    frames.append(env.render_rgb_array(episode=1, step=step_count))
            break

    # 4. Generate Visual Artifacts
    # Plot Q-Table
    plots_dir = os.path.join(results_dir, "plots")
    vis_filename = os.path.join(plots_dir, f"q_table_vis_{pct}.png" if pct != "final" else "q_table_vis.png")
    plot_q_table(agent.q_table, vis_filename, food_pos)
    
    # Save Video
    videos_dir = os.path.join(results_dir, "videos")
    video_filename = os.path.join(videos_dir, f"video_{pct}.mp4" if pct != "final" else "final_trained_agent.mp4")
    save_video(frames, video_filename)

def main():
    parser = argparse.ArgumentParser(description="GridWorld Evaluation")
    parser.add_argument("--seed", type=int, help="Override testing seed")
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

    # 2. Key Overrides (Allow user to change testing seed)
    global_config = get_default_config()
    testing_seed = args.seed or global_config.get('testing.seed', 42)
    config.set('testing.seed', testing_seed)
    
    # 3. Print Summary
    print("-" * 40)
    print(f"Evaluation Mode: {'Interoceptive' if config.get('body.with_satiation') else 'Conventional'}")
    print(f"Testing Seed: {testing_seed}")
    print("-" * 40)

    # 4. Find all checkpoints
    checkpoints = glob.glob(os.path.join(models_dir, "q_table_*.npy"))
    def extract_number(path):
        match = re.search(r"q_table_(\d+).npy", path)
        return int(match.group(1)) if match else -1
    checkpoints.sort(key=extract_number)
    
    final_model = os.path.join(models_dir, "q_table.npy")
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
    print(f"Evaluation complete! Artifacts are in {results_dir}/plots/ and {results_dir}/videos/")

if __name__ == "__main__":
    main()
