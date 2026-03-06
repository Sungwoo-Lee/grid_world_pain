#!/usr/bin/env python3
"""
Lab Server Background Execution — Launch scripts on 14-node Ubuntu cluster.

This script allows you to execute commands on remote nodes in the background,
with automated conda environment activation and timestamped logging.

Architecture:
    - 14 nodes (101-114).
    - IPs: 192.168.0.101 to 192.168.0.114.
    - SSH Port: 1800 (mapped to docker port 22).

Execution Flow:
    1. Connects to the specified node via SSH.
    2. Changes directory to the project root.
    3. Executes the command using 'nohup' and 'conda run'.
    4. Redirects stdout and stderr to logs/YYYYMMDD_HHMMSS.log.
    5. Disconnects while the process continues to run on the node.

Usage Examples:
    ./run_command.py 101 grid_world_pain "bash train_command.sh"
    ./run_command.py 114 grid_world_pain "python3 main.py --epochs 10"

Notes:
    - Logs are stored at: /media/nas01/projects/Interoceptive-AI/grid_world_pain/logs/
    - Use 'tail -f logs/<timestamp>.log' on the remote node to monitor progress.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# --- Configuration ---
PROJECT_ROOT = "/media/nas01/projects/Interoceptive-AI/grid_world_pain"
CONDA_BIN = "/home/vncuser/miniconda3/bin/conda"
REMOTE_USER = "vncuser"
SSH_PORT = 1800

def get_node_ip(node_id):
    """Map node_id (101-114) to IP address (192.168.0.101-114)."""
    if 101 <= node_id <= 114:
        return f"192.168.0.{node_id}"
    else:
        print(f"Error: Invalid node_id {node_id}. Must be between 101 and 114.")
        sys.exit(1)

def run_remote(node_id, conda_env, script_cmd, dry_run=False):
    ip = get_node_ip(node_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    full_log_dir = os.path.join(PROJECT_ROOT, "logs")
    log_file = os.path.join(full_log_dir, f"{timestamp}.log")

    # Construct the remote command
    # Using bash -c for safe execution of complex commands
    safe_script_cmd = script_cmd.replace('"', '\\"')
    remote_cmd = (
        f"cd {PROJECT_ROOT} && "
        f"nohup {CONDA_BIN} run --no-capture-output -n {conda_env} bash -c \"{safe_script_cmd}\" "
        f"> {log_file} 2>&1 &"
    )

    ssh_cmd = [
        "ssh", "-p", str(SSH_PORT),
        f"{REMOTE_USER}@{ip}",
        remote_cmd
    ]

    print(f"🚀 Executing on Node {node_id} (IP {ip}, Port {SSH_PORT})...")
    if dry_run:
        print(f"DEBUG: [DRY RUN] Command: {' '.join(ssh_cmd)}")
        return

    print(f"📂 Project: {PROJECT_ROOT}")
    print(f"🐍 Env: {conda_env}")
    print(f"📝 Log: {log_file}")
    print(f"💻 Command: {script_cmd}")
    
    try:
        # Run SSH command.
        subprocess.run(ssh_cmd, check=True)
        print(f"\n✅ Successfully started in background!")
        
        # Automatic tailing
        print(f"\n📺 Automatically tailing logs... (Press Ctrl+C to stop viewing, the process will keep running)")
        time.sleep(1) # Give a moment for the file to be started
        
        tail_cmd = [
            "ssh", "-p", str(SSH_PORT),
            f"{REMOTE_USER}@{ip}",
            f"tail -n 20 -f {log_file}"
        ]
        
        try:
            subprocess.run(tail_cmd)
        except KeyboardInterrupt:
            print("\n👋 Stopped tailing. The process is still running remotely.")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error executing SSH command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run background script on Lab Server nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s 101 grid_world_pain "bash train_command.sh"
  %(prog)s 105 grid_world_pain "python3 train.py"
        """
    )
    parser.add_argument("node", type=int, help="Node ID (101-114)")
    parser.add_argument("env", type=str, help="Conda environment name (e.g., 'grid_world_pain')")
    parser.add_argument("script", type=str, help="Script/command to execute (wrapped in quotes if it has spaces)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    
    args = parser.parse_args()
    
    run_remote(args.node, args.env, args.script, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
