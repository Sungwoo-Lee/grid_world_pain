#!/usr/bin/env python3
"""
Run a shell command on a lab cluster node (101-114).

Thin SSH wrapper. Backgrounds the remote command under nohup, redirects
stdout/stderr to a timestamped log file under `<PROJECT_ROOT>/logs/`, and
optionally tails the log until Ctrl+C.

What this script does NOT do (push these into your bash workload script):
  - `cd` into the project root
  - Activate a conda environment
  - Validate configs or pre-flight the env

The intent is "just run this command on that node" — everything project-
specific lives in the bash script you pass in.

Cluster facts (hardcoded — change here only if the cluster changes):
  - Nodes 101-114 map to IPs 192.168.0.101-114.
  - SSH port: 1800 (docker-mapped).
  - User: vncuser (SSH key auth via ~/.ssh/id_ed25519_gridworld).
  - Project root (NAS-mounted, same path on every node):
    /media/nas01/projects/Interoceptive-AI/grid_world_pain

Usage:
    ./run_command.py 113 "bash train_command-new.sh"
    ./run_command.py 114 "bash scripts/lab/launch_sheeprl.sh basic/01-5X5_Pred.yaml 0 gwp_5x5"
    ./run_command.py --foreground 113 "nvidia-smi"
    ./run_command.py --no-tail 114 "bash my_overnight_run.sh"
    ./run_command.py --log /tmp/custom.log 113 "echo hello"
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# --- Cluster constants ---
PROJECT_ROOT = "/media/nas01/projects/Interoceptive-AI/grid_world_pain"
REMOTE_USER = "vncuser"
SSH_PORT = 1800
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def get_node_ip(node_id):
    if 101 <= node_id <= 114:
        return f"192.168.0.{node_id}"
    print(f"Error: invalid node_id {node_id}. Must be 101-114.")
    sys.exit(1)


def build_ssh_opts():
    return [
        "-p", str(SSH_PORT),
        "-o", "ControlMaster=auto",
        "-o", "ControlPath=/tmp/ssh_mux_%h_%p_%r",
        "-o", "ControlPersist=600",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
    ]


def run_foreground(node_id, command, dry_run=False):
    ip = get_node_ip(node_id)
    ssh_cmd = ["ssh", *build_ssh_opts(), f"{REMOTE_USER}@{ip}", command]
    print(f"Node {node_id} ({ip}:{SSH_PORT}) [foreground]")
    print(f"Cmd:  {command}")
    if dry_run:
        print(f"\n[DRY RUN] {' '.join(ssh_cmd)}")
        return
    subprocess.run(ssh_cmd)


def run_background(node_id, command, log_path=None, tail=True, dry_run=False):
    ip = get_node_ip(node_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_path is None:
        log_path = os.path.join(DEFAULT_LOG_DIR, f"{timestamp}.log")

    # Build the remote command. The remote user runs `<command>` under
    # nohup, redirected to the log, and backgrounded. No cd, no conda.
    safe_cmd = command.replace('"', '\\"')
    remote_cmd = (
        f'mkdir -p "$(dirname {log_path})" && '
        f'nohup bash -c "{safe_cmd}" > {log_path} 2>&1 &'
    )

    ssh_cmd = ["ssh", *build_ssh_opts(), f"{REMOTE_USER}@{ip}", remote_cmd]

    print(f"Node {node_id} ({ip}:{SSH_PORT})")
    print(f"Log:  {log_path}")
    print(f"Cmd:  {command}")

    if dry_run:
        print(f"\n[DRY RUN] {' '.join(ssh_cmd)}")
        return

    try:
        subprocess.run(ssh_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: SSH launch failed: {e}")
        sys.exit(1)

    print("Launched in background.")

    if not tail:
        return

    time.sleep(1)
    print(f"\nTailing {log_path} (Ctrl+C to stop tailing; remote keeps running)\n")
    tail_cmd = ["ssh", *build_ssh_opts(), f"{REMOTE_USER}@{ip}", f"tail -n 20 -f {log_path}"]
    try:
        subprocess.run(tail_cmd)
    except KeyboardInterrupt:
        print("\nStopped tailing.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a shell command on a lab cluster node (101-114).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s 113 "bash train_command-new.sh"
  %(prog)s 114 "bash scripts/lab/launch_sheeprl.sh configs/environment/experiment/archive/basic/01-5X5_Pred.yaml 0 tag"
  %(prog)s --foreground 113 "nvidia-smi"
  %(prog)s --no-tail 114 "bash long_running.sh"
""",
    )
    parser.add_argument("node", type=int, help="Node ID (101-114).")
    parser.add_argument("command", type=str, help="Shell command to execute on the node.")
    parser.add_argument(
        "--log", type=str, default=None,
        help=f"Log path on the remote node. Default: {DEFAULT_LOG_DIR}/<timestamp>.log",
    )
    parser.add_argument(
        "--no-tail", action="store_true",
        help="Don't auto-tail the log after launching.",
    )
    parser.add_argument(
        "--foreground", action="store_true",
        help="Run synchronously, stream output to terminal. No nohup, no log. "
             "Use for short commands (nvidia-smi, pgrep, ls).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the SSH command without executing.",
    )
    args = parser.parse_args()

    if args.foreground:
        if args.log:
            print("Warning: --log ignored in --foreground mode.")
        run_foreground(args.node, args.command, dry_run=args.dry_run)
    else:
        run_background(
            args.node, args.command,
            log_path=args.log, tail=not args.no_tail, dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
