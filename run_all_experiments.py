
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



def main():
    parser = argparse.ArgumentParser(description="Run all Deep RL algorithms in parallel.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes per agent")
    parser.add_argument("--tag", type=str, required=True, help="Base tag for WandB/Results (e.g. 'experiment_v1')")
    
    args = parser.parse_args()
    
    # Process State:
    # (Algorithm, Train_Process, Log_File, Status)
    # Status: "TRAINING", "DONE", "FAILED"
    process_states = []
    
    print(f"Starting {len(ALGORITHMS)} experiments in parallel...")
    print(f"Base Tag: {args.tag}")
    print(f"Episodes: {args.episodes}")
    print("-" * 50)
    
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
            "status": "TRAINING"
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
                            print(f"[{name}] Training finished successfully.")
                            state["status"] = "DONE"
                        else:
                            print(f"[{name}] Training FAILED with code {ret}.")
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
            
            if state["train_log"] and not state["train_log"].closed: state["train_log"].close()
            
            if all_complete:
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt! Terminating all processes...")
        for state in process_states:
            if state["train_process"] and state["train_process"].poll() is None:
                print(f"[{state['name']}] Killing Training...")
                state["train_process"].terminate()
            if state["train_log"] and not state["train_log"].closed: state["train_log"].close()
            
    # Summary
    print("\n--- Final Summary ---")
    for state in process_states:
        print(f"{state['name']}: {state['status']}")

if __name__ == "__main__":
    main()
