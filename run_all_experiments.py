
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
    ("DQN", "configs/models/dqn.yaml", "cuda:1"),
    ("PPO", "configs/models/ppo.yaml", "cuda:1"),
    ("DRQN", "configs/models/drqn.yaml", "cuda:1"),
    ("RecurrentPPO", "configs/models/recurrent_ppo.yaml", "cuda:1"),
    # ("DreamerV3", "configs/models/dreamer_v3.yaml", "cuda:1"),
]

def find_latest_results_dir(algorithm_name, tag, start_time):
    """
    Finds the results directory for the given algorithm and tag.
    Assumes standard train.py output: results/{Algorithm}/{Timestamp}_{Tag}
    Returns the most recently created directory matching the pattern
    AND modified/created AFTER start_time.
    """
    # train.py uses folder names matching algorithm config loader usually,
    # but let's assume 'algorithm_name' in ALGORITHMS matches folder name.
    # Note: train.py uses 'DQN', 'PPO' etc as keys.
    
    base_path = os.path.join("results", algorithm_name)
    if not os.path.exists(base_path):
        return None
        
    # Search for directories containing the tag
    # Pattern: *_{tag}
    # Note: 'tag' passed to train.py is full_tag e.g. 'experiment_dqn'
    # Directory pattern: YYYYMMDD-HHMMSS_experiment_dqn
    
    search_pattern = os.path.join(base_path, f"*_{tag}")
    candidates = glob.glob(search_pattern)
    
    if not candidates:
        return None
        
    # Filter candidates created after start_time
    valid_candidates = []
    for c in candidates:
        if os.path.getmtime(c) > start_time:
            valid_candidates.append(c)
            
    if not valid_candidates:
        return None

    # Sort by creation/modification time (latest first)
    valid_candidates.sort(key=os.path.getmtime, reverse=True)
    return valid_candidates[0]

def find_best_checkpoint(results_dir):
    """
    Finds the checkpoint with the highest episode number in results_dir/models.
    """
    models_dir = os.path.join(results_dir, "models")
    if not os.path.exists(models_dir):
        return None
        
    checkpoints = glob.glob(os.path.join(models_dir, "*.ckpt"))
    if not checkpoints:
        return None
        
    # extract numbers
    # format: algo_model_1000.ckpt
    best_ckpt = None
    max_ep = -1
    
    for ckpt in checkpoints:
        match = re.search(r"_(\d+)\.ckpt$", ckpt)
        if match:
            ep = int(match.group(1))
            if ep > max_ep:
                max_ep = ep
                best_ckpt = ckpt
                
    return best_ckpt

def main():
    parser = argparse.ArgumentParser(description="Run all Deep RL algorithms in parallel.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes per agent")
    parser.add_argument("--tag", type=str, required=True, help="Base tag for WandB/Results (e.g. 'experiment_v1')")
    
    args = parser.parse_args()
    
    # Process State:
    # (Algorithm, Train_Process, Log_File, Status, Eval_Process, Eval_Log_File)
    # Status: "TRAINING", "EVALUATING", "DONE", "FAILED"
    process_states = []
    
    print(f"Starting {len(ALGORITHMS)} experiments in parallel...")
    print(f"Base Tag: {args.tag}")
    print(f"Episodes: {args.episodes}")
    print("-" * 50)
    
    # Capture start time to filter results later
    script_start_time = time.time()
    
    # Create unique log directory for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"{timestamp}_{args.tag}")
    os.makedirs(log_dir, exist_ok=True)
    
    for algo_name, config_path, device in ALGORITHMS:
        # Construct specific tag
        full_tag = f"{args.tag}_{algo_name.lower()}"
        
        cmd = [
            sys.executable, "train.py",
            "--agent_config", config_path,
            "--episodes", str(args.episodes),
            "--tag", full_tag,
            "--device", device
        ]
        
        print(f"[{algo_name}] Launching Training: {' '.join(cmd)}")
        
        log_path = os.path.join(log_dir, f"{algo_name}_train.log")
        log_file = open(log_path, "w")
        
        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        # Store state
        process_states.append({
            "name": algo_name,
            "tag": full_tag,
            "train_process": p,
            "train_log": log_file,
            "status": "TRAINING",
            "eval_process": None,
            "eval_log": None
        })
        
    print("-" * 50)
    print("All training processes launched. Monitoring...")
    print(f"Logs are being written to '{log_dir}/'")
    
    try:
        while True:
            all_complete = True
            
            for state in process_states:
                status = state["status"]
                name = state["name"]
                
                if status == "TRAINING":
                    all_complete = False
                    ret = state["train_process"].poll()
                    
                    if ret is not None:
                        # Training finished
                        state["train_log"].close()
                        
                        if ret == 0:
                            print(f"[{name}] Training finished successfully. Starting Evaluation...")
                            
                            # Find results dir and checkpoint
                            # Wait a brief moment for FS sync?
                            time.sleep(2)
                            results_dir = find_latest_results_dir(name, state["tag"], script_start_time)
                            
                            if results_dir:
                                ckpt = find_best_checkpoint(results_dir)
                                if ckpt:
                                    # Launch Evaluation
                                    ckpt_name = os.path.basename(ckpt)
                                    print(f"[{name}] Found checkpoint: {ckpt_name}. Launching evaluation...")
                                    
                                    eval_cmd = [
                                        sys.executable, "evaluation.py",
                                        "--results_dir", results_dir,
                                        "--checkpoint", ckpt_name,
                                        "--episodes", "5" # Default eval episodes
                                    ]
                                    
                                    eval_log_path = os.path.join(log_dir, f"{state['tag']}_eval.log")
                                    eval_log = open(eval_log_path, "w")
                                    
                                    ep = subprocess.Popen(
                                        eval_cmd,
                                        stdout=eval_log,
                                        stderr=subprocess.STDOUT
                                    )
                                    
                                    state["status"] = "EVALUATING"
                                    state["eval_process"] = ep
                                    state["eval_log"] = eval_log
                                else:
                                    print(f"[{name}] Error: No checkpoint found in {results_dir}. Skipping evaluation.")
                                    state["status"] = "DONE" # Or failed?
                            else:
                                print(f"[{name}] Error: Could not locate results directory for tag {state['tag']}. Skipping evaluation.")
                                state["status"] = "DONE"
                        else:
                            print(f"[{name}] Training FAILED with code {ret}.")
                            state["status"] = "FAILED"
                            
                elif status == "EVALUATING":
                    all_complete = False
                    ret = state["eval_process"].poll()
                    
                    if ret is not None:
                        state["eval_log"].close()
                        if ret == 0:
                            print(f"[{name}] Evaluation completed successfully.")
                            state["status"] = "DONE"
                        else:
                            print(f"[{name}] Evaluation FAILED with code {ret}.")
                            state["status"] = "FAILED"
            
            if all_complete:
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt! Terminating all processes...")
        for state in process_states:
            if state["train_process"] and state["train_process"].poll() is None:
                print(f"[{state['name']}] Killing Training...")
                state["train_process"].terminate()
            if state["eval_process"] and state["eval_process"].poll() is None:
                print(f"[{state['name']}] Killing Evaluation...")
                state["eval_process"].terminate()
            
            if state["train_log"] and not state["train_log"].closed: state["train_log"].close()
            if state["eval_log"] and not state["eval_log"].closed: state["eval_log"].close()
            
    # Summary
    print("\n--- Final Summary ---")
    for state in process_states:
        print(f"{state['name']}: {state['status']}")

if __name__ == "__main__":
    main()
