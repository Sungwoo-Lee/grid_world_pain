
import subprocess
import argparse
import sys
import time
import os

# Algorithms to run (excluding Tabular Q-Learning)
ALGORITHMS = [
    ("DQN", "configs/models/dqn.yaml"),
    ("PPO", "configs/models/ppo.yaml"),
    ("DRQN", "configs/models/drqn.yaml"),
    ("RecurrentPPO", "configs/models/recurrent_ppo.yaml"),
    ("DreamerV3", "configs/models/dreamer_v3.yaml"),
]

def main():
    parser = argparse.ArgumentParser(description="Run all Deep RL algorithms in parallel.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes per agent")
    parser.add_argument("--tag", type=str, required=True, help="Base tag for WandB/Results (e.g. 'experiment_v1')")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g. cuda:0)")
    
    args = parser.parse_args()
    
    processes = []
    
    print(f"Starting {len(ALGORITHMS)} experiments in parallel...")
    print(f"Base Tag: {args.tag}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {args.device}")
    print("-" * 50)
    
    for algo_name, config_path in ALGORITHMS:
        # Construct specific tag
        full_tag = f"{args.tag}_{algo_name.lower()}"
        
        cmd = [
            sys.executable, "train.py",
            "--agent_config", config_path,
            "--episodes", str(args.episodes),
            "--tag", full_tag,
            "--device", args.device
        ]
        
        print(f"[{algo_name}] Launching: {' '.join(cmd)}")
        
        # Open separate log files for each process to avoid console clutter
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, f"{full_tag}.log"), "w")
        
        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        processes.append((algo_name, p, log_file))
        
    print("-" * 50)
    print("All processes launched. Waiting for completion...")
    print("Logs are being written to the 'logs/' directory.")
    
    # Monitor loop
    try:
        while True:
            all_done = True
            for name, p, _ in processes:
                ret = p.poll()
                if ret is None:
                    all_done = False
                else:
                    # check if we already printed completion
                    pass 
            
            if all_done:
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt! Terminating all processes...")
        for name, p, _ in processes:
            print(f"Killing {name}...")
            p.terminate()
            
    # Close files and print status
    print("\n--- Summary ---")
    for name, p, f in processes:
        f.close()
        ret = p.returncode
        status = "SUCCESS" if ret == 0 else f"FAILED (Code {ret})"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main()
