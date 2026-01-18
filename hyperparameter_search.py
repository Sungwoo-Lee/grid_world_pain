import os
import yaml
import itertools
import argparse
import subprocess
import time
import sys
from datetime import datetime

# Define Search Spaces
SEARCH_SPACES = {
    "dqn": {
        "agent.batch_size": [32, 64],
        "agent.learning_rate": [1e-3, 5e-4],
        "agent.epsilon_decay": [0.99, 0.995],
        "agent.algorithm": ["DQN"]
    },
    "drqn": {
        "agent.batch_size": [32, 64],
        "agent.learning_rate": [1e-3, 5e-4],
        "agent.epsilon_decay": [0.99, 0.995],
        "agent.algorithm": ["DRQN"]
    },
    "ppo": {
        "agent.lr_actor": [3e-4, 1e-4],
        "agent.lr_critic": [1e-3, 5e-4],
        "agent.gamma": [0.99],
        "agent.K_epochs": [4, 10],
        "agent.algorithm": ["PPO"]
    },
    "recurrent_ppo": {
        "agent.lr_actor": [3e-4, 1e-4],
        "agent.lr_critic": [1e-3, 5e-4],
        "agent.gamma": [0.99],
        "agent.K_epochs": [4, 10],
        "agent.algorithm": ["RecurrentPPO"]
    }
}

TEMP_CONFIG_DIR = "temp_configs"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_config(base_config_path, params, output_path):
    """
    Loads base config, updates with params, and saves to output_path.
    """
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Helper to set nested dict values
    def set_nested(d, key, value):
        keys = key.split('.')
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    for k, v in params.items():
        set_nested(config, k, v)

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

def run_search(algorithm, episodes, seed, dry_run=False):
    if algorithm not in SEARCH_SPACES:
        print(f"Error: Algorithm '{algorithm}' not found in search spaces.")
        print(f"Available: {list(SEARCH_SPACES.keys())}")
        return

    space = SEARCH_SPACES[algorithm]
    keys = list(space.keys())
    values = list(space.values())
    combinations = list(itertools.product(*values))

    print(f"Starting search for {algorithm} with {len(combinations)} combinations.")
    
    ensure_dir(TEMP_CONFIG_DIR)
    
    base_config_path = f"configs/models/{algorithm}.yaml"
    if not os.path.exists(base_config_path):
        print(f"Warning: Base config {base_config_path} not found. Starting from empty.")
        # Create a basic config file structure if it doesn't exist? 
        # Actually it's better to just ensure the key params are present.
        # But for now, let's assume the user has the base files.

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Create a unique tag
        tag_parts = [f"{k.split('.')[-1]}_{v}" for k, v in params.items() if k != 'agent.algorithm']
        tag = f"search_{algorithm}_{'_'.join(tag_parts)}"
        
        # Limit tag length just in case
        if len(tag) > 200:
            tag = tag[:200]
            
        print(f"\n--- Run {i+1}/{len(combinations)}: {tag} ---")
        print(f"Params: {params}")

        temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{tag}.yaml")
        generate_config(base_config_path, params, temp_config_path)

        cmd = [
            sys.executable, "train.py",
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--agent_config", temp_config_path,
            "--tag", tag
        ]

        if dry_run:
            print(f"Dry Run Command: {' '.join(cmd)}")
        else:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running training for {tag}: {e}")
            
        # Cleanup
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    print("\nSearch complete.")
    try:
        os.rmdir(TEMP_CONFIG_DIR)
    except OSError:
        pass # Directory might not be empty

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search Runner")
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm to search (dqn, ppo, drqn, recurrent_ppo)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    
    args = parser.parse_args()
    
    run_search(args.algorithm, args.episodes, args.seed, args.dry_run)

if __name__ == "__main__":
    main()
