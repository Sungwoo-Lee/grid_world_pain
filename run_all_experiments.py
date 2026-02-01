"""
Parallel Execution Script for GridWorld RL Experiments.

This script launches multiple training processes in parallel for different algorithms (DQN, PPO, DRQN, etc.).
It manages the subprocesses, logs their output to `logs/`, and monitors their status.

Arguments:
- `--episodes <int>`: (Default: 10000) Number of training episodes per agent.
- `--tag <str>`: (Required) Base tag for the experiment (e.g., `experiment_v1`).

Usage Examples:

1. **Run Experiments**:
   ```bash
   python run_all_experiments.py --tag baseline_run
   ```

2. **Long Run**:
   ```bash
   python run_all_experiments.py --episodes 50000 --tag long_run
   ```

Notes:
- Ensure that the `ALGORITHMS` list in the script contains the desired models and config paths.
- Each process is pinned to a specific device (e.g., `cuda:1`) as defined in `ALGORITHMS`.
"""

import subprocess
import argparse
import sys
import time
import os
import glob
import re

# Algorithms to run with specific device assignment
# Format: (Algorithm Name, Config Path, Device)
ALGORITHMS = [
    # ("DQN", "configs/models/dqn.yaml", "cuda:1"),
    # ("PPO", "configs/models/ppo.yaml", "cuda:1"),
    # ("DRQN", "configs/models/drqn.yaml", "cuda:1"),
    ("RecurrentPPO", "configs/models/recurrent_ppo.yaml", "cuda:1"),
    # ("DreamerV3", "configs/models/dreamer_v3.yaml", "cuda:3"),
]

def main():
    parser = argparse.ArgumentParser(description="Run Deep RL algorithms in parallel for ablation study.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes per agent")
    parser.add_argument("--tag", type=str, required=True, help="Base tag for WandB/Results (e.g. 'ablation_v1')")
    parser.add_argument("--config", type=str, help="Specific config file to run (optional)")
    
    args = parser.parse_args()
    
    # 1. Discover Configs
    config_queue = []
    if args.config:
        config_queue.append(args.config)
    else:
        # Default: Full ablation sweep
        base_dir = "configs/ablation"
        for branch in ["survival", "homeostatic"]:
            branch_dir = os.path.join(base_dir, branch)
            if os.path.isdir(branch_dir):
                files = sorted(glob.glob(os.path.join(branch_dir, "*.yaml")))
                config_queue.extend(files)
    
    if not config_queue:
        print("No configurations found to run.")
        return

    print(f"Starting {len(config_queue)} experiment levels...")
    print(f"Algorithms per level: {[a[0] for a in ALGORITHMS]}")
    print(f"Base Tag: {args.tag}")
    print(f"Episodes: {args.episodes}")
    print("-" * 50)

    # 2. Iterate through configs sequentially
    for config_file in config_queue:
        # Extract branch and level name for tagging
        # Path format: configs/ablation/{branch}/{level}.yaml
        parts = config_file.split(os.sep)
        branch_name = parts[-2] if len(parts) >= 2 else "unknown"
        level_name = os.path.basename(config_file).replace(".yaml", "")
        
        level_tag = f"{args.tag}_{branch_name}_{level_name}"
        
        print(f"\n>>> Running Level: {level_name} (Branch: {branch_name})")
        print(f"    Config: {config_file}")
        
        # Process State for this set of algorithms
        process_states = []
        
        # Create log directory for this level
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", f"{timestamp}_{level_tag}")
        os.makedirs(log_dir, exist_ok=True)
        
        for algo_name, algo_config, device in ALGORITHMS:
            full_tag = f"{level_tag}_{algo_name.lower()}"
            
            cmd = [
                sys.executable, "train.py",
                "--agent_config", algo_config,
                "--episodes", str(args.episodes),
                "--tag", full_tag,
                "--device", device,
                "--wandb-group", level_tag,
                "--config", config_file,
            ]
            
            log_path = os.path.join(log_dir, f"{algo_name}_train.log")
            log_file = open(log_path, "w")
            
            p = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            
            process_states.append({
                "name": algo_name,
                "process": p,
                "log": log_file
            })
            
        print(f"    Launched {len(ALGORITHMS)} agents. Monitoring...")
        
        # Wait for all processes in this level to finish
        try:
            while True:
                all_complete = True
                for state in process_states:
                    if state["process"].poll() is None:
                        all_complete = False
                        break
                
                if all_complete:
                    break
                time.sleep(5)
                
            # Close logs
            for state in process_states:
                state["log"].close()
                ret = state["process"].poll()
                if ret != 0:
                    print(f"    [!] {state['name']} FAILED with code {ret}")
                else:
                    print(f"    [+] {state['name']} DONE")
                    
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt! Terminating all processes...")
            for state in process_states:
                if state["process"].poll() is None:
                    state["process"].terminate()
                state["log"].close()
            sys.exit(1)

    print("\n--- All Ablation Levels Complete ---")

if __name__ == "__main__":
    main()
