#!/usr/bin/env python
"""Lab cluster GPU status — spec, live state, and usage across nodes 101-114.

Queries each node's `nvidia-smi` over DIRECT SSH with per-node captured output.
Do NOT route this through run_command.py: that wrapper writes every remote command's
output to one timestamped NAS log and tails it, so parallel queries collide on a shared
log and every node reports one node's output (this produced a WRONG homogeneous spec on
2026-06-30). This tool captures each node independently and is the source of truth for
both the static spec doc (docs/environment/LAB_NODE_GPU_SPEC.md) and live free/busy state.

Usage:
  gpu_status.py                  # full status (spec + util + mem + processes), all nodes
  gpu_status.py --nodes 108 109  # only these nodes
  gpu_status.py --spec-only      # hardware inventory only (model + VRAM + indices)
  gpu_status.py --free           # only GPUs that are free right now, weakest tier first
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

SSH_PORT = 1800
USER = "vncuser"
SSH_OPTS = ["-p", str(SSH_PORT), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=no"]
ALL_NODES = list(range(101, 115))

# Rough capability ranking (higher = stronger) for free-GPU recommendations.
def tier_rank(name):
    n = name.lower()
    if "6000 ada" in n: return 4
    if "4090" in n:     return 3
    if "3090" in n:     return 2
    if "2080 ti" in n:  return 1
    return 0

FREE_UTIL = 5      # % utilisation below which a GPU counts as idle
FREE_MEM  = 1000   # MiB used below which a GPU counts as free


def ssh(node, cmd, timeout=15):
    full = ["ssh", *SSH_OPTS, f"{USER}@192.168.0.{node}", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            err = (r.stderr.strip().splitlines() or ["ssh error"])[-1]
            return None, err[:70]
        return r.stdout, None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:70]


def query_node(node, spec_only):
    if spec_only:
        out, err = ssh(node, "nvidia-smi --query-gpu=index,name,memory.total "
                             "--format=csv,noheader,nounits")
        if out is None:
            return node, {"error": err}
        gpus = []
        for line in out.strip().splitlines():
            idx, name, mtot = [p.strip() for p in line.split(",")]
            gpus.append({"idx": idx, "name": name, "mem_total": int(float(mtot))})
        return node, {"gpus": gpus}

    out, err = ssh(node, "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,"
                         "utilization.gpu,temperature.gpu --format=csv,noheader,nounits")
    if out is None:
        return node, {"error": err}
    gpus = []
    by_uuid = {}
    for line in out.strip().splitlines():
        idx, uuid, name, mtot, mused, util, temp = [p.strip() for p in line.split(",")]
        g = {"idx": idx, "uuid": uuid, "name": name,
             "mem_total": int(float(mtot)), "mem_used": int(float(mused)),
             "util": int(float(util)), "temp": temp, "procs": []}
        gpus.append(g)
        by_uuid[uuid] = g
    apps, _ = ssh(node, "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name "
                        "--format=csv,noheader,nounits")
    if apps:
        for line in apps.strip().splitlines():
            if not line.strip():
                continue
            uuid, pid, mem, pname = [p.strip() for p in line.split(",")]
            if uuid in by_uuid:
                by_uuid[uuid]["procs"].append({"pid": pid, "mem": mem,
                                               "name": pname.split("/")[-1][:24]})
    return node, {"gpus": gpus}


def is_free(g):
    return g["util"] <= FREE_UTIL and g["mem_used"] <= FREE_MEM


def main():
    ap = argparse.ArgumentParser(description="Lab cluster GPU status.")
    ap.add_argument("--nodes", type=int, nargs="+", default=ALL_NODES)
    ap.add_argument("--spec-only", action="store_true")
    ap.add_argument("--free", action="store_true")
    args = ap.parse_args()

    with ThreadPoolExecutor(max_workers=len(args.nodes)) as ex:
        results = dict(ex.map(lambda n: query_node(n, args.spec_only), args.nodes))

    if args.spec_only:
        print(f"{'NODE':<6}{'GPUs':<10}{'MODEL':<32}{'VRAM':>10}")
        for n in args.nodes:
            r = results[n]
            if "error" in r:
                print(f"{n:<6}{'-':<10}{'UNREACHABLE: '+r['error']}")
                continue
            g = r["gpus"]
            idxs = ",".join(x["idx"] for x in g)
            print(f"{n:<6}{idxs:<10}{g[0]['name']:<32}{str(g[0]['mem_total'])+' MiB':>10}")
        return

    if args.free:
        free = []
        for n in args.nodes:
            r = results[n]
            if "error" in r:
                continue
            for g in r["gpus"]:
                if is_free(g):
                    free.append((tier_rank(g["name"]), n, g))
        free.sort(key=lambda x: (x[0], x[1], x[2]["idx"]))   # weakest tier first
        print("FREE GPUs (weakest tier first — prefer the weakest card adequate for the job):")
        if not free:
            print("  none idle right now")
        for _, n, g in free:
            print(f"  {n}:{g['idx']}  {g['name']:<24} {g['mem_total']} MiB  (util {g['util']}%, {g['mem_used']} MiB used)")
        return

    # full status
    for n in args.nodes:
        r = results[n]
        if "error" in r:
            print(f"node {n}: UNREACHABLE — {r['error']}")
            continue
        print(f"node {n}:")
        for g in r["gpus"]:
            tag = "FREE" if is_free(g) else "BUSY"
            print(f"  GPU {g['idx']} [{tag}] {g['name']:<32} util {g['util']:>3}%  "
                  f"mem {g['mem_used']:>6}/{g['mem_total']} MiB  {g['temp']}C")
            for p in g["procs"]:
                print(f"        pid {p['pid']:<8} {p['mem']:>6} MiB  {p['name']}")


if __name__ == "__main__":
    main()
