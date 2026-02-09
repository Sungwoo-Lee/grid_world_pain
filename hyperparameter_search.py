"""
Hyperparameter Search Runner for JAX GridWorld RL Agents.

This script automates the process of running multiple training sessions with different
hyperparameter combinations using JAX-native components. It generates temporary 
configuration files and executes `train.py` (JAX version) for each combination. 
Supports parallel execution and strict configuration protocols.

Features:
- Updated search spaces for JAX RecurrentPPO and DreamerV3.
- Network layer synchronization via SYNC_GROUPS.
- Forced usage of explicit Conda environment python.
- Automatic grouping and job-type tagging for WandB.
- Parallel process management with log redirection.

Usage:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python hyperparameter_search.py \\
        --algorithm recurrent_ppo --episodes 1000 --num-processes 4 --seed 42
"""

import os
import yaml
import itertools
import argparse
import subprocess
import time
import sys
from datetime import datetime
import multiprocessing
from tqdm import tqdm

# Constants for Strict Protocol
PYTHON_EXEC = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
TRAIN_SCRIPT = "train.py"
TEMP_CONFIG_DIR = "temp_configs"

# Define Search Spaces for JAX Algorithms
SEARCH_SPACES = {
    "recurrent_ppo": {
        "agent.lr_actor": [1e-4, 5e-4],
        "agent.entropy_coef": [0.001, 0.01],
        "agent.sequence_length": [64, 128],
        "agent.actor_fc_layers": [[128, 128], [256, 256]],
        "agent.rnn_type": ["LSTM", "GRU"],
        "agent.activation": ["tanh", "relu"],
        "agent.gamma": [0.95, 0.99],
        "agent.algorithm": ["RecurrentPPO"]
    },
    "dreamer_v3": {
        "agent.model_lr": [1e-4, 3e-4],
        "agent.actor_lr": [8e-5, 2e-4],
        "agent.value_lr": [8e-5, 2e-4],
        "agent.rssm_deter_dim": [256, 512],
        "agent.rssm_stoch_dim": [32],
        "agent.encoder_dim": [128, 256],
        "agent.algorithm": ["DreamerV3"]
    }
}

# Parameter Synchronization Groups
SYNC_GROUPS = {
    "recurrent_ppo": [
        ("agent.actor_fc_layers", "agent.critic_fc_layers", "agent.fc_layers"),
    ],
    "dreamer_v3": [
        ("agent.actor_fc_layers", "agent.critic_fc_layers", "agent.reward_fc_layers", "agent.continue_fc_layers"),
    ]
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except FileExistsError:
            pass

def generate_config(base_config_path, params, output_path):
    """Loads base config, updates with params, and saves to output_path."""
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    def set_nested(d, key, value):
        keys = key.split('.')
        for i, k in enumerate(keys[:-1]):
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    for k, v in params.items():
        set_nested(config, k, v)

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

def run_single_combination(params, algorithm, episodes, seed, config, wandb_project, wandb_group, wandb_job_type, dry_run, index, total, log_dir):
    """Worker function to run a single hyperparameter combination."""
    tag_parts = []
    for k, v in params.items():
        if k == 'agent.algorithm': continue
        short_key = k.split('.')[-1]
        tag_parts.append(f"{short_key}_{v}")
        
    tag = f"search_{algorithm}_{'_'.join(tag_parts)}"
    if len(tag) > 200: tag = tag[:200]
        
    if not dry_run:
        print(f"\n--- Run {index}/{total}: {tag} ---")
        print(f"Log: {os.path.join(log_dir, f'run_{index}.log')}")

    ensure_dir(TEMP_CONFIG_DIR)
    temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{tag}_{os.getpid()}.yaml")
    
    base_config_path = f"configs/models/{algorithm}.yaml"
    generate_config(base_config_path, params, temp_config_path)

    cmd = [
        PYTHON_EXEC, TRAIN_SCRIPT,
        "--agent_config", temp_config_path,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--tag", tag,
        "--wandb-project", wandb_project,
        "--wandb-group", wandb_group,
        "--wandb-name", tag,
        "--wandb-job-type", wandb_job_type,
        "--quiet"
    ]
    
    if config:
        cmd.extend(["--config", config])

    if dry_run:
        print(f"Dry Run Command: {' '.join(cmd)}")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    else:
        log_path = os.path.join(log_dir, f"run_{index}.log")
        try:
            with open(log_path, "w") as log_file:
                subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error running training for {tag}: {e}")
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

def run_single_combination_wrapper(args):
    """Wrapper to unpack arguments for imap."""
    return run_single_combination(*args)

def run_search(algorithm, episodes, seed, config, wandb_project, num_processes, dry_run, wandb_job_type):
    if algorithm not in SEARCH_SPACES:
        print(f"Error: Algorithm '{algorithm}' not found in search spaces.")
        return

    space = SEARCH_SPACES[algorithm]
    sync_groups = SYNC_GROUPS.get(algorithm, [])
    
    synced_params = set()
    for group in sync_groups:
        synced_params.update(group)
    
    independent_keys = [k for k in space.keys() if k not in synced_params]
    
    leaders = []
    followers = {}
    for group in sync_groups:
        leader = group[0]
        leaders.append(leader)
        followers[leader] = list(group[1:])
    
    product_keys = independent_keys + leaders
    product_values = [space[k] for k in product_keys]
    
    raw_combinations = list(itertools.product(*product_values))
    
    combinations = []
    for raw_combo in raw_combinations:
        params = {}
        for k, v in zip(product_keys, raw_combo):
            params[k] = v
        for leader, group_followers in followers.items():
            val = params[leader]
            for f in group_followers:
                params[f] = val
        if "agent.algorithm" not in params and "agent.algorithm" in space:
             params["agent.algorithm"] = space["agent.algorithm"][0]
        combinations.append(params)

    total_combinations = len(combinations)
    print(f"Starting JAX search for {algorithm} with {total_combinations} combinations.")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    wandb_group = f"search_{algorithm}_{timestamp}"
    log_dir = os.path.join("logs", f"{timestamp}_search_{algorithm}")
    ensure_dir(log_dir)
    
    tasks = []
    for i, params in enumerate(combinations):
        tasks.append((params, algorithm, episodes, seed, config, wandb_project, wandb_group, wandb_job_type, dry_run, i+1, total_combinations, log_dir))
    
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            try:
                for _ in tqdm(pool.imap_unordered(run_single_combination_wrapper, tasks), total=len(tasks), desc="JAX HPO Search"):
                    pass
            except KeyboardInterrupt:
                pool.terminate()
                sys.exit(1)
    else:
        for task in tqdm(tasks, desc="JAX HPO Search"):
            run_single_combination(*task)

    if not dry_run:
        print(f"\nSearch complete. Logs: {log_dir}")
    try:
        os.rmdir(TEMP_CONFIG_DIR)
    except OSError:
        pass

def main():
    parser = argparse.ArgumentParser(description="JAX Hyperparameter Search Runner")
    parser.add_argument("--algorithm", type=str, required=True, choices=["recurrent_ppo", "dreamer_v3"])
    parser.add_argument("--episodes", type=int, required=True, help="Episodes per run")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--num-processes", type=int, default=1, help="Parallel processes")
    parser.add_argument("--config", type=str, help="Base config file (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--wandb-project", type=str, help="WandB Project Name")

    args = parser.parse_args()

    # Import Config to read defaults
    sys.path.append(os.getcwd())
    from src.utils.config import Config
    
    wandb_project = args.wandb_project
    if not wandb_project:
        wandb_config_path = "configs/logger/wandb.yaml"
        if os.path.exists(wandb_config_path):
            wc = Config.load_yaml(wandb_config_path)
            wandb_project = wc.get('wandb.project')

    if not wandb_project:
        raise ValueError("Strict Protocol: WandB project must be specified via --wandb-project or in configs/logger/wandb.yaml")

    run_search(args.algorithm, args.episodes, args.seed, args.config, wandb_project, args.num_processes, args.dry_run, "hyperparameter search")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
