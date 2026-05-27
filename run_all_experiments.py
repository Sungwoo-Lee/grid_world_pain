"""
Parallel Execution Script for JAX GridWorld RL Experiments.

This script launches multiple training processes in parallel for different algorithms 
(RecurrentPPO, DreamerV3) across various ablation configurations. It manages 
subprocesses, logs output to `logs/`, and monitors status, adhering to strict 
project protocols.

Features:
- Discovery of ablation configs in survival/ and homeostatic/ branches.
- Parallel execution of multiple algorithms per ablation level.
- Forced usage of explicit Conda environment python.
- Automatic result directory and log management.
- Integration with JAX-native train.py.

Usage:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python run_all_experiments.py \\
        --tag ablation_sweep --episodes 10000
"""

import subprocess
import argparse
import sys
import time
import os
import glob

# Constants for Strict Protocol
PYTHON_EXEC = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
TRAIN_SCRIPT = "train.py"

# Algorithms to run in parallel per level
# Format: (Algorithm Name, Config Path)
ALGORITHMS = [
    ("RecurrentPPO", "configs/models/recurrent_ppo/recurrent_ppo.yaml"),
    # ("DreamerV3", "configs/models/dreamer_v3/dreamer_v3.yaml"),
]

def main():
    parser = argparse.ArgumentParser(description="Run JAX RL experiments in parallel.")
    parser.add_argument("--episodes", type=int, default=10000, help="Episodes per agent")
    parser.add_argument("--tag", type=str, required=True, help="Base tag for WandB/Results")
    parser.add_argument("--config", type=str, help="Specific config file to run (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    
    args = parser.parse_args()
    
    # 1. Discover Configs
    config_queue = []
    if args.config:
        config_queue.append(args.config)
    else:
        # Default: Full ablation sweep
        base_dir = "configs/experiment/ablation"
        for branch in ["survival", "homeostatic"]:
            branch_dir = os.path.join(base_dir, branch)
            if os.path.isdir(branch_dir):
                files = sorted(glob.glob(os.path.join(branch_dir, "*.yaml")))
                config_queue.extend(files)
    
    if not config_queue:
        print("No configurations found to run.")
        return

    print(f"Starting {len(config_queue)} JAX experiment levels...")
    print(f"Algorithms per level: {[a[0] for a in ALGORITHMS]}")
    print(f"Base Tag: {args.tag}")
    print("-" * 50)

    # 2. Iterate through configs sequentially (parallelizing algorithms within each level)
    for config_file in config_queue:
        parts = config_file.split(os.sep)
        branch_name = parts[-2] if len(parts) >= 2 else "unknown"
        level_name = os.path.basename(config_file).replace(".yaml", "")
        
        level_tag = f"{args.tag}_{branch_name}_{level_name}"
        
        print(f"\n>>> Running Level: {level_name} (Branch: {branch_name})")
        print(f"    Config: {config_file}")
        
        process_states = []
        
        # Create log directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", f"{timestamp}_{level_tag}")
        if not args.dry_run:
            os.makedirs(log_dir, exist_ok=True)
        
        for algo_name, algo_config in ALGORITHMS:
            full_tag = f"{level_tag}_{algo_name.lower()}"
            
            cmd = [
                PYTHON_EXEC, TRAIN_SCRIPT,
                "--agent_config", algo_config,
                "--episodes", str(args.episodes),
                "--tag", full_tag,
                "--wandb-group", level_tag,
                "--config", config_file,
                "--quiet"
            ]
            
            if args.dry_run:
                print(f"    [DRY RUN] {' '.join(cmd)}")
                continue

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
            
        if args.dry_run:
            continue

        print(f"    Launched {len(ALGORITHMS)} agents. Monitoring...")
        
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
                
            for state in process_states:
                state["log"].close()
                ret = state["process"].poll()
                if ret != 0:
                    print(f"    [!] {state['name']} FAILED with code {ret}")
                else:
                    print(f"    [+] {state['name']} DONE")
                    
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt! Terminating processes...")
            for state in process_states:
                if state["process"].poll() is None:
                    state["process"].terminate()
                state["log"].close()
            sys.exit(1)

    print("\n--- All JAX Ablation Levels Complete ---")

if __name__ == "__main__":
    main()
