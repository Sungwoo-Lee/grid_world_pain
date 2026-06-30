# Lab Node GPU Spec — `gw-101` … `gw-114`

> **Purpose (read before assigning a node:GPU for any training launch).** This documents the
> GPU hardware on every lab node so a launch never (a) requests a GPU index that does not exist,
> or (b) wastes a high-end card on a small job (or starves a big job onto a weak card). The
> cluster is **HETEROGENEOUS** — four different GPU classes — and **GPU count varies per node**
> (most have 2; node 114 has 4). Match the card to the job: this project's standard rPPO/Dreamer
> runs (10×10 grid, `--num-envs 16`) are small and belong on the **low/mid tier**; reserve the
> RTX 6000 Ada / 4090 nodes for jobs that actually need the compute or VRAM.

**Probed:** 2026-06-30 via **direct SSH** `nvidia-smi --query-gpu=index,name,memory.total` with
per-node captured output. (An earlier `run_command.py`-based probe was WRONG — its parallel calls
shared one timestamped NAS log, so every node reported one node's output. Always probe with the
`scripts/lab/gpu_status.py` tool, which captures per-node.)

## The hard rules
1. **GPU indices are per-node — check the table.** Most nodes have GPUs `0,1` only (no `2`).
   **Node 114 has `0,1,2,3`.** Never assume an index exists.
2. **The cluster is heterogeneous — match card to job:**
   - **Low (RTX 2080 Ti, 11 GB):** 101, 103, 104, 105 — fine for standard small rPPO/Dreamer runs.
   - **Mid-high (RTX 3090, 24 GB):** 106–112 — the workhorse bulk; default for routine runs.
   - **High (RTX 4090, 24 GB):** 102, 113 — fast; use when 3090s are full or the job is compute-heavy.
   - **Top (RTX 6000 Ada, 49 GB ×4):** 114 — reserve for big-VRAM / many-parallel / heaviest jobs.
   - **Anti-pattern:** do NOT put a tiny rPPO job on 114 while a 2080 Ti sits idle, and do NOT
     starve a large job onto an 11 GB 2080 Ti when a 24/49 GB card is free.

## Per-node inventory (confirmed 2026-06-30)

| Node | GPUs | Model (each) | VRAM each | Tier |
|---|---|---|---|---|
| 101 | 0, 1 | RTX 2080 Ti | 11 GB | low |
| 102 | 0, 1 | RTX 4090 | 24 GB | high |
| 103 | 0, 1 | RTX 2080 Ti | 11 GB | low |
| 104 | 0, 1 | RTX 2080 Ti | 11 GB | low |
| 105 | 0, 1 | RTX 2080 Ti | 11 GB | low |
| 106 | 0, 1 | RTX 3090 | 24 GB | mid-high |
| 107 | 0, 1 | RTX 3090 | 24 GB | mid-high |
| 108 | 0, 1 | RTX 3090 | 24 GB | mid-high |
| 109 | 0, 1 | RTX 3090 | 24 GB | mid-high |
| 110 | 0, 1 | RTX 3090 | 24 GB | mid-high |
| 111 | 0, 1 | RTX 3090 | 24 GB | mid-high |
| 112 | 0, 1 | RTX 3090 | 24 GB | mid-high |
| 113 | 0, 1 | RTX 4090 | 24 GB | high |
| 114 | 0, 1, 2, 3 | RTX 6000 Ada Generation | 49 GB | top |

**Totals:** 14 nodes, **30 GPUs** — 8× 2080 Ti, 14× 3090, 4× 4090, 4× RTX 6000 Ada.

## How to use when assigning GPUs for a launch
- **Existence:** only use indices listed above for that node (114 → up to 3; everyone else → 0/1).
- **Tier match:** default routine rPPO/Dreamer → a low or mid-high node; escalate to 4090/Ada only
  for genuinely heavier jobs or when lower tiers are saturated.
- **Live availability:** this table is static *hardware*. For *current* free/busy state + running
  processes, run `scripts/lab/gpu_status.py` (or the `gpu-status` skill) — that is the source of
  truth for "is this GPU free right now / is the node up". Reachability changes hour to hour
  (NAS unmounts, SSH refused), so it is intentionally NOT baked into this static table.

## Maintenance
Re-probe and update this table if hardware changes:
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/lab/gpu_status.py --spec-only
```
