# Lab Node GPU Spec — `gw-101` … `gw-114`

> **Purpose (read before assigning a node:GPU for any training launch).** This documents the
> GPU hardware on every lab node so a launch never (a) requests a GPU index that does not exist,
> or (b) wastes a high-end card on a small job (or starves a big job onto a weak card). The
> cluster is **HETEROGENEOUS** — four different GPU classes — and **GPU count varies per node**
> (most have 2; node 114 has 4). Match the card to the job: this project's standard rPPO/Dreamer
> runs (10×10 grid, `--num-envs 128`) are small and belong on the **low/mid tier**; reserve the
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
3. **Pack-node-first — fill one node's free GPUs before spilling to the next.** People claim whole
   nodes per job set; scattering one person's runs one-GPU-each across many nodes blocks colleagues
   from getting a clean node. See the allocation-policy section below.

4. **Eval sweeps are CPU-bound (multithreaded XLA compile) — thread-cap them, don't blame the NAS.**
   A frozen-checkpoint / behaviour-probe eval sweep spawns one short-lived Python process per
   checkpoint. Each process's cost is dominated by **JAX import + XLA compile (~13 s), which is
   multithreaded and sizes its threadpools to the whole core count** — so a handful of concurrent
   processes oversubscribe the CPU (measured: 16 procs → load 100 on a 20-core node; *stacked*
   un-killable workers → load 137, throughput collapse). The shared ceiling is **CPU threads, not
   NAS I/O** — parallel checkpoint reads are fine. See the "Eval-sweep parallelism" section below
   for the measured fix (compile cache + 1-thread cap + node fan-out).

## Eval-sweep parallelism — CPU-bound, thread-cap it (measured 2026-07-13)
All code, configs, checkpoints, and results live on one shared NAS, mounted at the **same path on
every node** (`/media/nas01/projects/Interoceptive-AI/grid_world_pain`). A frozen-checkpoint eval
sweep reads each checkpoint off that NAS, so the intuition is "don't fan out — you'll jam the NAS."
**That intuition was wrong.** A measured breakdown of one `eval_rollout.py --batched` invocation
(10x10 grid, 30 episodes, CPU) on a 20-core node:

| Config | s/eval | load (20 cores) |
|---|---|---|
| single eval, cold | 14.7 | — |
| NPAR=16, no cache, no thread-cap | 3.43 | 100 |
| NPAR=16, warm compile cache | 1.92 | 99 |
| **NPAR=18, warm cache + 1-thread cap** | **0.66** | **29** |

- **The bottleneck is CPU threads, not the NAS.** ~13 s of each eval is JAX import + XLA compile,
  which is **multithreaded and sizes threadpools to the full core count**. Running N such processes
  oversubscribes the box (16 -> load 100). The earlier "150-way across five nodes stalled at load
  137, un-killable" incident was **CPU thread oversubscription amplified by *stacked* workers I
  couldn't kill from a sandboxed container** — not NAS I/O saturation. Parallel checkpoint *reads*
  are fine (as routine parallel work on this NAS has always shown).
- **The fix (measured 5x speedup):**
  1. **Persistent XLA compilation cache** — the compiled program depends only on *shapes*, which are
     identical across a model's checkpoints (only weights differ), so eval #2..N hit the cache and
     skip the ~7 s compile. Set `JAX_COMPILATION_CACHE_DIR` (+ `JAX_PERSISTENT_CACHE_MIN_*=0`).
  2. **Thread-cap each process to 1 core** — `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
     MKL_NUM_THREADS=1`, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.
     This is what keeps load ~= core count instead of ~5x over.
  3. **NPAR ~= core_count** (≈18 on a 20-core node) -> load ~29, ~0.66 s/eval.
- **Fan-out across nodes now distributes fine** — each thread-capped node sits at a safe load ~29,
  so nodes 101-105 give a near-linear wall-clock win (18.7k evals: ~3.4 h on one node -> ~41 min on
  five). The earlier failure was thread thrash + stacking, *not* the shared NAS.
- **One worker per node, never stack.** The disaster's real trigger was launching a second worker
  before the first was dead. Launch exactly one worker per node; if you must adjust, kill first,
  confirm zero `eval_rollout` procs, then relaunch.
- **Outputs are shared; concurrent writes race.** Anything under `results/` is visible on every
  node, but two jobs writing the *same* file race — give parallel jobs **distinct output paths**.
- Aside: `nproc` under a detached/`nohup` remote context can misreport `1` — never size a worker
  pool from `nproc`; pass the parallelism explicitly.

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

## GPU allocation policy: pack-node-first (fill one node before spilling to the next)

When a single person launches **multiple** runs, assign them to **all usable GPUs on one node
first**, and only move to a second node once the first node's GPUs are exhausted. Lab convention is
that people claim a **whole node** per job set, so scattering one person's runs one-GPU-each across
many nodes leaves every node half-occupied and blocks colleagues from getting a clean node.

- **Pack order:** run 1 → `nodeA:0`, run 2 → `nodeA:1` (node 114 → `0,1,2,3`), then spill to
  `nodeB:0`, `nodeB:1`, … — never open a third node while node A or B still has a free GPU that fits.
- **Tier match still applies:** first pick the node whose tier fits the job (routine rPPO → a
  low/mid node), THEN fill that node before opening another. Don't cram tiny jobs onto a 4090/Ada
  node just to keep them together — choose the right-tier node first, then pack it.
- **Live state first:** consult `scripts/lab/gpu_status.py` (or the `gpu-status` skill) so "fill
  this node" means filling its actually-free GPUs, not ones a colleague is already using.

## Maintenance
Re-probe and update this table if hardware changes:
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/lab/gpu_status.py --spec-only
```
