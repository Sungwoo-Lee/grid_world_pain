# Lab Node GPU Spec — `gw-101` … `gw-114`

> **Purpose (read this before assigning a node:GPU for any training launch).** This documents
> the GPU hardware on every lab node so a launch never requests a GPU that does not exist or
> mis-matches the job. The single most important fact: **every node has exactly TWO GPUs,
> indices `0` and `1` — there is NO GPU `2`** (requesting `cuda:2` fails). The cluster is
> **homogeneous** — every node is the same card — so GPU selection is about which GPU is
> *free*, not about "spec": there is no high-spec/low-spec tier to match to task complexity.

**Probed:** 2026-06-30 via `run_command.py <node> "nvidia-smi --query-gpu=index,name,memory.total"`.

## The hard rules
1. **Valid GPU indices are `0` and `1` only.** Never request `cuda:2` (or higher) on any node — there is no third GPU. (This is the bug this doc exists to prevent.)
2. **All GPUs are identical** — `NVIDIA GeForce RTX 2080 Ti`, **11 GB** (11264 MiB) each. There is no "use the fast GPU for the hard job" decision to make; pick by **availability** (which of `0`/`1` is free per `nvidia-smi`).
3. **Memory budget:** ~11 GB per GPU. A standard rPPO run (`--num-envs 16`, 10×10 grid) uses well under that; two small jobs can share one GPU but will contend — prefer one job per GPU when GPUs are free.

## Per-node inventory

| Node | GPUs | Model (each) | VRAM each | Notes |
|---|---|---|---|---|
| 101 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed 2026-06-30 |
| 102 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 103 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 104 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 105 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 106 | 0, 1 | RTX 2080 Ti | 11 GB | **down at probe time** (SSH refused); spec from cluster homogeneity — re-verify when up |
| 107 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 108 | 0, 1 | RTX 2080 Ti (inferred) | 11 GB | 2 GPUs confirmed (no GPU 2); model inferred — couldn't re-query live (active run held SSH) |
| 109 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 110 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 111 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 112 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 113 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |
| 114 | 0, 1 | RTX 2080 Ti | 11 GB | confirmed |

## How to use when assigning GPUs for a launch
- **Count:** at most **2 concurrent runs per node** (one per GPU). For N runs across the cluster, spread them: e.g. 4 runs → `108:0, 108:1, 109:0, 109:1`.
- **Index:** only `0` or `1`. If you catch yourself about to write `:2`, stop.
- **Availability ≠ existence:** this doc says the GPU *exists*; the training-runner still does a live `nvidia-smi` pre-flight to confirm the chosen GPU is *free* (and that the node is reachable — see the NAS-mount / reachability caveats in memory). A GPU being busy is transient; the index map here is static hardware.
- **Reachability caveat:** some nodes intermittently drop off (NAS unmount or SSH refused — 106 was down at this probe; 107 has had no-NAS-mount incidents). Reachability is not in this doc because it changes hour to hour — the runner's pre-flight is the source of truth for "is it up right now".

## Maintenance
Re-run the probe and update this table if hardware changes or a previously-unverified node (106 down, 108 model inferred) is confirmed:
```bash
for n in $(seq 101 114); do echo "== $n =="; ./run_command.py $n "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"; done
```
