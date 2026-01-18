"""
Hyperparameter Search Runner for GridWorld RL Agents.

This script automates the process of running multiple training sessions with different
hyperparameter combinations. It generates temporary configuration files and executes
`train.py` for each combination.

Supported Algorithms:
- DQN
- DRQN
- PPO
- RecurrentPPO

features:
- defined search spaces for each algorithm.
- automatic grouping of runs in weights & biases (wandb).
- dry-run mode to preview commands.

Arguments:
- `--algorithm <str>`: Algorithm to search (dqn, ppo, drqn, recurrent_ppo).
- `--episodes <int>`: Number of episodes per run (default: 1000).
- `--seed <int>`: Random seed (default: 42).
- `--wandb-project <str>`: WandB project name (default: "grid_world_pain").
- `--dry-run`: Print commands without executing.

Usage:
    # Run a DQN search with 1000 episodes per run
    python hyperparameter_search.py --algorithm dqn --episodes 1000 --wandb-project my_project

    # Preview commands for PPO search
    python hyperparameter_search.py --algorithm ppo --episodes 5000 --dry-run
"""
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
        "agent.learning_rate": [1e-4, 3e-4, 1e-3],
        "agent.batch_size": [64, 128],
        "agent.epsilon_decay": [0.995, 0.999],
        "agent.target_update_freq": [1000, 5000],
        "agent.algorithm": ["DQN"]
    },
    "drqn": {
        "agent.learning_rate": [1e-4, 3e-4, 1e-3],
        "agent.batch_size": [32, 64],
        "agent.trace_length": [8, 16],
        "agent.target_update_freq": [1000, 5000],
        "agent.algorithm": ["DRQN"]
    },
    "ppo": {
        "agent.lr_actor": [1e-4, 3e-4],
        "agent.lr_critic": [5e-4, 1e-3],
        "agent.clip_param": [0.1, 0.2],
        "agent.entropy_coef": [0.001, 0.01],
        "agent.K_epochs": [4, 10],
        "agent.algorithm": ["PPO"]
    },
    "recurrent_ppo": {
        "agent.lr_actor": [1e-4, 3e-4],
        "agent.lr_critic": [5e-4, 1e-3],
        "agent.clip_param": [0.1, 0.2],
        "agent.entropy_coef": [0.001, 0.01],
        "agent.K_epochs": [4, 10],
        "agent.sequence_length": [8],
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
        for i, k in enumerate(keys[:-1]):
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    for k, v in params.items():
        set_nested(config, k, v)

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

def run_search(algorithm, episodes, seed, wandb_project, dry_run=False):
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

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Create a unique tag
        # Shorten keys for readability
        tag_parts = []
        for k, v in params.items():
            if k == 'agent.algorithm':
                continue
            short_key = k.split('.')[-1]
            tag_parts.append(f"{short_key}_{v}")
            
        tag = f"search_{algorithm}_{'_'.join(tag_parts)}"
        
        # Limit tag length
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
            "--tag", tag,
            "--wandb-project", wandb_project,
            "--wandb-group", f"search_{algorithm}_{int(time.time())}", # Group all runs in this search together
            "--wandb-name", tag
        ]

        if dry_run:
            print(f"Dry Run Command: {' '.join(cmd)}")
            # Cleanup immediately in dry run
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
        else:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running training for {tag}: {e}")
            finally:
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
    
    # Load default project from config if available
    default_project = "grid_world_pain"
    if os.path.exists("configs/wandb.yaml"):
        with open("configs/wandb.yaml", 'r') as f:
            wc = yaml.safe_load(f)
            if wc and 'wandb' in wc and 'project' in wc['wandb']:
                default_project = wc['wandb']['project']

    parser.add_argument("--wandb-project", type=str, default=default_project, help=f"WandB Project Name (default: {default_project})")
    
    args = parser.parse_args()
    
    run_search(args.algorithm, args.episodes, args.seed, args.wandb_project, args.dry_run)

if __name__ == "__main__":
    main()
