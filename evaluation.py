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
from src.grid_world_pain.visualization import generate_visuals_from_checkpoint

def main():
    results_dir = "results"
    models_dir = os.path.join(results_dir, "models")
    config_path = os.path.join(models_dir, "config.yaml")

    # 1. Load saved configuration
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        print("Please run train.py first to generate a model and its configuration.")
        return

    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Find all checkpoints
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

    print(f"Found {len(checkpoints)} checkpoints. Starting evaluation...")
    print("-" * 40)

    # 3. Generate visuals for each checkpoint
    for checkpoint in checkpoints:
        generate_visuals_from_checkpoint(checkpoint, results_dir, config=config)

    print("-" * 40)
    print(f"Evaluation complete! Artifacts are in {results_dir}/plots/ and {results_dir}/videos/")

if __name__ == "__main__":
    main()
