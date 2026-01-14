"""
Evaluation script for the GridWorld Reinforcement Learning agent.

This script:
1. Loads the configuration saved during training (results/models/config.yaml).
2. Loads all available checkpoints from results/models/.
3. Generates Q-table visualizations and performance videos for each checkpoint.

Usage:
    python evaluation.py
"""
import os
import glob
import re
import yaml
import sys
import argparse
from src.grid_world_pain.visualization import generate_visuals_from_checkpoint
from src.grid_world_pain.config import get_default_config

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
        # Load into an actual Config object for ease of use
        from src.grid_world_pain.config import Config
        saved_config_dict = yaml.safe_load(f)
        config = Config(saved_config_dict)

    # 2. Key Overrides (Allow user to change testing seed without retraining)
    global_config = get_default_config()
    testing_seed = args.seed or global_config.get('testing.seed', 42)
    config.set('testing.seed', testing_seed)
    
    # 3. Print Summary
    print("-" * 40)
    print(f"Evaluation Mode: {'Interoceptive' if config.get('body.with_satiation') else 'Conventional'}")
    print(f"Grid Size: {config.get('environment.height')}x{config.get('environment.width') if config.get('environment.width') else '?'}")
    print(f"Testing Seed: {testing_seed}")
    print("-" * 40)

    # 4. Find all checkpoints
    checkpoints = glob.glob(os.path.join(models_dir, "q_table_*.npy"))
    
    # Sort checkpoints by percentage milestone
    def extract_number(path):
        match = re.search(r"q_table_(\d+).npy", path)
        return int(match.group(1)) if match else -1
    checkpoints.sort(key=extract_number)
    
    # Also include the final model if it exists
    final_model = os.path.join(models_dir, "q_table.npy")
    if os.path.exists(final_model):
        checkpoints.append(final_model)

    if not checkpoints:
        print(f"No model checkpoints found in {models_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints. Starting visual generation...")

    # 5. Generate visuals for each checkpoint
    for checkpoint in checkpoints:
        generate_visuals_from_checkpoint(checkpoint, results_dir, config=config)

    print("-" * 40)
    print(f"Evaluation complete! Artifacts are in {results_dir}/plots/ and {results_dir}/videos/")

if __name__ == "__main__":
    main()
