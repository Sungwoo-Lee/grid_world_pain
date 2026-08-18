---
name: gpu-status
description: "Show live GPU spec, state, and usage across the lab nodes (101-114) by running scripts/lab/gpu_status.py. ALWAYS use this BEFORE assigning a node:GPU for a training launch — it is the source of truth for which GPUs are free right now, what is running on each, and which card each node has (the cluster is HETEROGENEOUS: RTX 2080 Ti 11GB / RTX 3090 24GB / RTX 4090 24GB / RTX 6000 Ada 49GB, and node 114 has 4 GPUs while all others have 2). Also trigger on /gpu-status, 'check gpu usage', 'which gpus are free', 'gpu status', 'show cluster gpus', 'what's running on the nodes', or any request to pick a node/GPU for a run. Distinct from the static spec doc docs/environment/LAB_NODE_GPU_SPEC.md (hardware only) — this skill shows the LIVE state. Queries each node over direct SSH with per-node captured output; never routes through run_command.py (whose shared NAS log corrupts parallel queries)."
---

# gpu-status — live lab-cluster GPU availability + spec

Run `scripts/lab/gpu_status.py` to see the real-time GPU state across nodes 101–114, then
use it to assign a `node:GPU` for a launch. The cluster is **heterogeneous** and **GPU count
varies per node**, so never assume an index/spec — check.

## When to use
- **Before every training launch** that needs a node + GPU (you, or before spawning the
  `training-runner`). Pick the GPU from this tool's live output, not from memory.
- On `/gpu-status` or any "which GPUs are free / what's running / show the cluster" request.
- To refresh the static spec doc after a hardware change (`--spec-only`).

## How to run (conda interpreter)
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/lab/gpu_status.py [MODE]
```
| Mode | Output |
|---|---|
| (none) | Full status: per node, each GPU's util / mem / temp + the processes running on it, FREE/BUSY tag |
| `--free` | Only GPUs idle right now, **weakest tier first** (so you pick the weakest card adequate for the job) |
| `--spec-only` | Hardware inventory only (model + VRAM + valid indices) — used to maintain the spec doc |
| `--nodes 108 109` | Restrict to specific nodes |

## How to choose (match card to job)
The project's standard rPPO runs (10×10 grid) use `--num-envs 128` — the `configs/train/default.yaml` default. **Do NOT override to 16** (a long-standing habit that silently shadowed the 128 default; the CLI flag wins over the config). Even at 128 envs they stay modest (a few GB), so they fit
on any card, including an 11 GB RTX 2080 Ti, and do **not** need a 49 GB RTX 6000 Ada. DreamerV3's `num_envs` is a **per-run decision** — check the config/launch for the intended value; do not assume 16.

1. Run `--free`.
2. **Prefer the weakest free card that fits the job** (the list is already sorted that way):
   routine runs → a free **2080 Ti (101/103/104/105)** or **3090 (106–112)**.
3. **Reserve the high tier** — RTX 4090 (102, 113) and RTX 6000 Ada (114, 4 GPUs) — for jobs
   that genuinely need the compute/VRAM, or when the lower tiers are full.
4. Confirm the chosen node:GPU is `FREE` and the **index exists on that node** (114 → 0–3;
   everyone else → 0/1). Then hand node + GPU to the `training-runner`.

## References
- Script: `scripts/lab/gpu_status.py`.
- Static hardware doc (tiers, per-node table): [docs/environment/LAB_NODE_GPU_SPEC.md](../../../docs/environment/LAB_NODE_GPU_SPEC.md).
- Launch flow: the `training-runner` agent still does its own live `nvidia-smi` pre-flight on the chosen GPU before launching.
