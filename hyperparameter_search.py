"""
Hyperparameter Search Runner for GridWorld RL Agents.

This script automates the process of running multiple training sessions with different
hyperparameter combinations. It generates temporary configuration files and executes
`train.py` for each combination. Supports parallel execution to speed up the search.

Supported Algorithms:
- DQN
- DRQN
- PPO
- RecurrentPPO

features:
- defined search spaces for each algorithm.
- automatic grouping of runs in weights & biases (wandb).
- parallel execution of training runs.
- dry-run mode to preview commands.

Arguments:
- `--algorithm <str>`: Algorithm to search (dqn, ppo, drqn, recurrent_ppo).
- `--episodes <int>`: Number of episodes per run (default: 1000).
- `--seed <int>`: Random seed (default: 42).
- `--num-processes <int>`: Number of parallel processes to run (default: 1).
- `--device <str>`: Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0', 'auto').
- `--config <str>`: Path to base training configuration file.
- `--wandb-project <str>`: WandB project name.
- `--dry-run`: Print commands without executing.

Usage:
    # Run a DQN search with 1000 episodes per run
    python hyperparameter_search.py --algorithm dqn --episodes 1000 --wandb-project my_project

    # Run PPO search with 4 parallel processes
    python hyperparameter_search.py --algorithm ppo --episodes 5000 --num-processes 4

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
import multiprocessing
from src.utils.wandb_utils import wandb_login


# Define Search Spaces
SEARCH_SPACES = {
    "dqn": {
        "agent.learning_rate": [1e-4, 1e-3],
        "agent.batch_size": [64, 128],
        "agent.epsilon_decay": [0.995, 0.999],
        "agent.target_update_freq": [1000, 5000],
        "agent.fc_layers": [[256, 128, 64], [128, 128]],
        "agent.gamma": [0.95, 0.99],
        "agent.algorithm": ["DQN"]
    },
    "drqn": {
        "agent.learning_rate": [1e-4, 1e-3],
        "agent.batch_size": [32, 64],
        "agent.trace_length": [8, 16],
        "agent.epsilon_decay": [0.995, 0.999],
        "agent.target_update_freq": [1000, 5000],
        "agent.fc_layers": [[64], [128]],
        "agent.recurrent_layers": [[64], [128]],
        "agent.gamma": [0.95, 0.99],
        "agent.algorithm": ["DRQN"]
    },
    "ppo": {
        "agent.lr_actor": [1e-4, 3e-4],
        "agent.lr_critic": [5e-4, 1e-3],
        # "agent.clip_param": [0.1, 0.2],
        "agent.entropy_coef": [0.001, 0.01],
        "agent.K_epochs": [4, 10],
        "agent.actor_fc_layers": [[64, 64], [128, 128]],
        "agent.critic_fc_layers": [[64, 64], [128, 128]],
        "agent.algorithm": ["PPO"]
    },
    "recurrent_ppo": {
        # "agent.lr_actor": [1e-4, 3e-4],
        # "agent.lr_critic": [5e-4, 1e-3],
        # "agent.clip_param": [0.1, 0.2],
        "agent.entropy_coef": [0.001, 0.01],
        # "agent.K_epochs": [4, 10],
        "agent.sequence_length": [64, 128],
        # "agent.fc_layers": [[64], [128]],
        # "agent.recurrent_layers": [[64]],
        "agent.actor_fc_layers": [[128, 128]],
        # "agent.critic_fc_layers": [[64], [128]],
        "agent.gamma": [0.95, 0.99],
        "agent.algorithm": ["RecurrentPPO"]
    },
    "dreamer_v3": {
        "agent.model_lr": [1e-4, 3e-4],
        "agent.actor_lr": [8e-5, 2e-4],
        "agent.value_lr": [8e-5, 2e-4],
        "agent.rssm_deter_dim": [256, 512],
        "agent.rssm_stoch_dim": [32],
        "agent.encoder_dim": [256],
        "agent.algorithm": ["DreamerV3"]
    }
}

# Define Synchronized Groups
# Parameters in the same group will always share the same value (from the first parameter in the group)
SYNC_GROUPS = {
    "ppo": [
        ("agent.actor_fc_layers", "agent.critic_fc_layers"),
    ],
    "recurrent_ppo": [
        ("agent.actor_fc_layers", "agent.critic_fc_layers", "agent.fc_layers"),
    ],
}

TEMP_CONFIG_DIR = "temp_configs"

def ensure_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except FileExistsError:
            pass

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

def run_single_combination(params, algorithm, episodes, seed, config, wandb_project, wandb_group, wandb_job_type, dry_run, index, total, log_dir, device):
    """
    Worker function to run a single hyperparameter combination.
    """
    # Unique tag generation
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
        
    print(f"\n--- Run {index}/{total}: {tag} ---")
    if not dry_run:
        print(f"Log: {os.path.join(log_dir, f'run_{index}.log')}")
        print(f"Params: {params}")

    ensure_dir(TEMP_CONFIG_DIR)
    
    # Unique config path for this process
    temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{tag}_{os.getpid()}.yaml")
    
    base_config_path = f"configs/models/{algorithm}.yaml"
    if not os.path.exists(base_config_path):
        print(f"Warning: Base config {base_config_path} not found. Starting from empty.")

    # Generate config
    generate_config(base_config_path, params, temp_config_path)

    cmd = [
        sys.executable, "train.py",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--agent_config", temp_config_path,
        "--tag", tag,
        "--wandb-project", wandb_project,
        "--wandb-group", wandb_group,
        "--wandb-name", tag,
        "--wandb-job-type", wandb_job_type,
        "--quiet"
    ]
    
    if config:
        cmd.extend(["--config", config])
    if device:
        cmd.extend(["--device", device])

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

from tqdm import tqdm

def run_single_combination_wrapper(args):
    """Wrapper to unpack arguments for imap."""
    return run_single_combination(*args)

def run_search(algorithm, episodes, seed, config, wandb_project, num_processes, dry_run, wandb_job_type, device):
    if algorithm not in SEARCH_SPACES:
        print(f"Error: Algorithm '{algorithm}' not found in search spaces.")
        print(f"Available: {list(SEARCH_SPACES.keys())}")
        return

    space = SEARCH_SPACES[algorithm]
    sync_groups = SYNC_GROUPS.get(algorithm, [])
    
    # Identify synced and independent parameters
    synced_params = set()
    for group in sync_groups:
        synced_params.update(group)
    
    independent_keys = [k for k in space.keys() if k not in synced_params]
    
    # Define leaders and followers for sync groups
    leaders = []
    followers = {} # leader -> list of followers
    for group in sync_groups:
        leader = group[0]
        leaders.append(leader)
        followers[leader] = list(group[1:])
    
    # Build product keys and values
    product_keys = independent_keys + leaders
    product_values = [space[k] for k in product_keys]
    
    raw_combinations = list(itertools.product(*product_values))
    
    # Generate full parameter dictionaries
    combinations = []
    for raw_combo in raw_combinations:
        params = {}
        # Set independent and leader values
        for k, v in zip(product_keys, raw_combo):
            params[k] = v
        
        # Sync followers with leaders
        for leader, group_followers in followers.items():
            val = params[leader]
            for f in group_followers:
                params[f] = val
        
        # Add a placeholder for agent.algorithm if not present in the combination itself
        # (though it usually is in SEARCH_SPACES)
        if "agent.algorithm" not in params and "agent.algorithm" in space:
             params["agent.algorithm"] = space["agent.algorithm"][0]

        combinations.append(params)

    total_combinations = len(combinations)

    print(f"Starting search for {algorithm} with {total_combinations} combinations.")
    print(f"Parallel processes: {num_processes}")
    
    ensure_dir(TEMP_CONFIG_DIR)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    wandb_group = f"search_{algorithm}_{timestamp}"
    log_dir = os.path.join("logs", f"{timestamp}_search_{algorithm}")
    ensure_dir(log_dir)
    
    print(f"Logs will be saved to: {log_dir}")
    
    # Prepare arguments for each task
    tasks = []
    for i, params in enumerate(combinations):
        tasks.append((params, algorithm, episodes, seed, config, wandb_project, wandb_group, wandb_job_type, dry_run, i+1, total_combinations, log_dir, device))
    
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            try:
                # Use imap_unordered to update progress bar as tasks complete
                for _ in tqdm(pool.imap_unordered(run_single_combination_wrapper, tasks), total=len(tasks), desc="Hyperparameter Search"):
                    pass
            except KeyboardInterrupt:
                print("\nInterrupted! Terminating parallel processes...")
                pool.terminate()
                pool.join()
                sys.exit(1)
    else:
        # Sequential execution
        try:
            for task in tqdm(tasks, desc="Hyperparameter Search"):
                run_single_combination(*task)
        except KeyboardInterrupt:
            print("\nInterrupted! Terminating search...")
            sys.exit(1)

    print("\nSearch complete.")
    try:
        os.rmdir(TEMP_CONFIG_DIR)
    except OSError:
        pass # Directory might not be empty

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search Runner")
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm to search (dqn, ppo, drqn, recurrent_ppo)")
    parser.add_argument("--episodes", type=int, help="Number of episodes per run (Required)")
    parser.add_argument("--seed", type=int, help="Random seed (Required)")
    parser.add_argument("--num-processes", type=int, required=True, help="Number of parallel processes (Required)")
    parser.add_argument("--config", type=str, help="Path to config file (Optional)")
    parser.add_argument("--device", type=str, help="Device to use for training (e.g., 'cpu', 'cuda', 'auto')")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--wandb-project", type=str, help="WandB Project Name")

    
    args = parser.parse_args()
    
    # Ensure src is importable
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    from src.utils.config import Config

    # Load defaults from WandB config
    wandb_project = None
    wandb_job_type = "hyperparameter search"
    
    wandb_config_path = "configs/logger/wandb.yaml"
    if os.path.exists(wandb_config_path):
        try:
            wc = Config.load_yaml(wandb_config_path)
            wandb_project = wc.get('wandb.project')
        except Exception as e:
            print(f"Warning: Failed to load {wandb_config_path}: {e}")

    # Override with CLI args
    if args.wandb_project:
        wandb_project = args.wandb_project
        
    # Strict Validation
    if args.episodes is None:
        raise ValueError("Strict Config: '--episodes' is a required argument.")
    if args.seed is None:
        raise ValueError("Strict Config: '--seed' is a required argument.")
    if wandb_project is None:
        raise ValueError("Strict Config: 'wandb.project' must be specified in configs/wandb.yaml or via --wandb-project")
    
    
    # Ensure WandB login for the search runner
    wandb_login(quiet=False)
    
    run_search(args.algorithm, args.episodes, args.seed, args.config, wandb_project, args.num_processes, args.dry_run, wandb_job_type, args.device)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True) # Safe for pytorch/cuda
    main()
