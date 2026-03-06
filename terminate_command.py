#!/usr/bin/env python3
"""
Lab Server Process Terminator — Selective pkill across 14-node Ubuntu cluster.

This script allows you to terminate processes across multiple nodes with a two-stage process:
    1. Scan: Lists all matching processes across targeted nodes for review.
    2. Confirmation: Requires explicit user consent before performing the kill.

By default, it sends SIGINT (2) to trigger a graceful shutdown (e.g., finishing
the current training iteration and closing WandB). Use --force for SIGTERM (15).

Usage:
    ./kill_command.py <nodes> <pattern> [-f]

Node Selection Formats:
    - Single Node:  101
    - Range:         101-105
    - List:          101,102,110
    - All Nodes:     all (maps to 101-114)
    - Mixed:         101,105-107,110

Examples:
    # Gracefully kill all 'train.py' processes (send SIGINT)
    ./kill_command.py all "train.py"

    # Forcefully kill processes on specific nodes (send SIGTERM)
    ./kill_command.py 101-105 "python" --force
"""

import argparse
import getpass
import sys
import pexpect
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration (Synced with run_command.py) ---
PROJECT_ROOT = "/media/nas01/projects/Interoceptive-AI/grid_world_pain"
REMOTE_USER = "vncuser"
SSH_PORT = 1800

# ANSI Colors
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BOLD = "\033[1m"
NC = "\033[0m"

def get_node_ip(node_id):
    """Map node_id (101-114) to IP address (192.168.0.101-114)."""
    if 101 <= node_id <= 114:
        return f"192.168.0.{node_id}"
    else:
        raise ValueError(f"Invalid node_id {node_id}. Must be between 101 and 114.")

def parse_nodes(node_str):
    """Parse node string into a list of integers."""
    if node_str.lower() == "all":
        return list(range(101, 115))
    
    nodes = set()
    parts = node_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    start, end = end, start
                nodes.update(range(start, end + 1))
            except ValueError:
                print(f"{RED}Error: Invalid range format '{part}'. Use START-END.{NC}")
                sys.exit(1)
        else:
            try:
                nodes.add(int(part))
            except ValueError:
                print(f"{RED}Error: Invalid node ID '{part}'.{NC}")
                sys.exit(1)
    
    return sorted(list(nodes))

def ssh_execute(ip, command, password, timeout=10):
    """Execute a command via SSH using pexpect and return the output and exit status."""
    ssh_cmd = f"ssh -o ConnectTimeout=5 -p {SSH_PORT} {REMOTE_USER}@{ip} \"{command}\""
    try:
        child = pexpect.spawn(ssh_cmd, encoding='utf-8', timeout=timeout)
        index = child.expect(["(?i)password:", "(?i)are you sure you want to continue connecting", pexpect.EOF, pexpect.TIMEOUT])
        
        if index == 1:
            child.sendline("yes")
            child.expect("(?i)password:")
            child.sendline(password)
        elif index == 0:
            child.sendline(password)
        
        output = child.read()
        child.close()
        return output, child.exitstatus
    except Exception as e:
        return str(e), -1

def scan_node(node_id, pattern, password):
    """Scan a node for matching processes. Returns (node_id, processes, status)."""
    ip = get_node_ip(node_id)
    # pgrep -af: a=show full command line, f=match against full command line
    command = f"pgrep -af '{pattern}'"
    output, status = ssh_execute(ip, command, password)
    
    processes = []
    if status == 0 and output:
        for line in output.strip().splitlines():
            if line:
                processes.append(line)
    return node_id, processes, status

def kill_node(node_id, pattern, password, signal_num=2):
    """Kill matching processes on a node using the specified signal."""
    ip = get_node_ip(node_id)
    # pkill -<sig> -f <pattern>
    command = f"pkill -{signal_num} -f '{pattern}'"
    output, status = ssh_execute(ip, command, password)
    return status == 0

def main():
    parser = argparse.ArgumentParser(
        description="Selective process terminator for Lab Server nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Node Selection Examples:
  - Single:  101
  - Range:   101-105
  - List:    101,102,110
  - All:     all (maps to 101-114)
  - Mixed:   101,105-107,110

Usage Examples:
  %(prog)s all "train.py"         # Graceful (SIGINT)
  %(prog)s 101-105 "python" -f    # Forceful (SIGTERM)
        """
    )
    parser.add_argument("nodes", type=str, help="Node(s) to target (e.g. 101, 101-105, all)")
    parser.add_argument("pattern", type=str, help="Process pattern to pkill (passed to pkill -f)")
    parser.add_argument("-f", "--force", action="store_true", help="Send SIGTERM (15) instead of SIGINT (2)")

    args = parser.parse_args()

    try:
        target_nodes = parse_nodes(args.nodes)
    except Exception as e:
        print(f"{RED}Error parsing nodes: {e}{NC}")
        sys.exit(1)

    print(f"{BOLD}{CYAN}🚀 STAGE 1: Scanning for processes matching '{args.pattern}'...{NC}")
    password = getpass.getpass(f"🔑 Enter SSH password for {REMOTE_USER}: ")

    node_results = {}
    unresponsive_nodes = []
    clean_nodes = []

    # Parallel scanning
    with ThreadPoolExecutor(max_workers=len(target_nodes)) as executor:
        futures = {executor.submit(scan_node, node_id, args.pattern, password): node_id for node_id in target_nodes}
        
        # Use tqdm for progress bar
        with tqdm(total=len(target_nodes), desc="Scanning nodes", unit="node") as pbar:
            for future in as_completed(futures):
                node_id, procs, status = future.result()
                if status == -1 or "Timeout" in str(procs): # Error or timeout
                    unresponsive_nodes.append(node_id)
                elif procs:
                    node_results[node_id] = procs
                else:
                    clean_nodes.append(node_id)
                pbar.update(1)

    # Report results
    if unresponsive_nodes:
        print(f"\n{BOLD}{RED}⚠️  NODES WITH NO REACTION / TIMEOUT:{NC}")
        print(f"  {', '.join(map(str, sorted(unresponsive_nodes)))}")

    if not node_results:
        print(f"\n{GREEN}✨ No matching processes found on any responsive nodes.{NC}")
        if not unresponsive_nodes:
            return
        if input(f"\n{YELLOW}Continue with remaining nodes if any? (y/N): {NC}").lower() != 'y':
            return
    else:
        print(f"\n{BOLD}{YELLOW}⚠️  FOUND MATCHING PROCESSES:{NC}")
        for node_id in sorted(node_results.keys()):
            procs = node_results[node_id]
            print(f"{BOLD}Node {node_id}:{NC}")
            for proc in procs:
                print(f"  {proc}")
        
        print(f"\n{BOLD}Total nodes with processes: {len(node_results)}{NC}")
    
    signal_num = 15 if args.force else 2
    sig_name = "SIGTERM (Force)" if args.force else "SIGINT (Graceful)"
    
    confirm = input(f"\n{BOLD}{RED}Proceed with termination using {sig_name}? (y/N): {NC}")
    if confirm.lower() != 'y':
        print(f"{CYAN}Aborted.{NC}")
        return

    print(f"\n{BOLD}{CYAN}🚀 STAGE 2: Terminating processes with {sig_name}...{NC}")
    success_count = 0
    # Use parallel termination as well if many nodes
    with ThreadPoolExecutor(max_workers=min(len(node_results), 10)) as executor:
        future_to_node = {executor.submit(kill_node, node_id, args.pattern, password, signal_num=signal_num): node_id for node_id in node_results.keys()}
        for future in as_completed(future_to_node):
            node_id = future_to_node[future]
            try:
                if future.result():
                    print(f"  Node {node_id}: {GREEN}✅ Success{NC}")
                    success_count += 1
                else:
                    print(f"  Node {node_id}: {RED}❌ Failed{NC}")
            except Exception as exc:
                print(f"  Node {node_id}: {RED}❌ Exception: {exc}{NC}")

    print(f"\n{BOLD}🏁 Finished. Terminated processes on {success_count}/{len(node_results)} nodes.{NC}")

if __name__ == "__main__":
    main()
