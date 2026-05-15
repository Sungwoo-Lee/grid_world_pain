---
title: "JAXVectorEnv v2 — GPU-placement spike (option A: numpy boundary, JAX on GPU)"
topic: dreamer
status: active
created: 2026-05-13
last_updated: 2026-05-13
supersedes: JAX_VECTOR_ENV_V1_PLAN.md
---

# JAXVectorEnv v2 — GPU-placement spike

> **Status**: PLANNED — 2026-05-13
> **Related**: [v1 plan (now superseded)](../../archive/sheeprl_bridge/JAX_VECTOR_ENV_V1_PLAN.md) · [Parallel-env benchmark](PARALLEL_ENV_BENCHMARK.md) · [CLAUDE.md doc-framing rule](../../../../CLAUDE.md)
> **Type**: spike (minimum effort to answer ONE question; halt-at-design-phase clause enforced)

---

## Question

We have a vmap-batched gridworld env wrapper (`JAXVectorEnv`) that should run all N parallel environments in one fused kernel instead of N sequential Python loop iterations. **v1 placed JAX on CPU and gave no speedup** at any N from 1 to 16 — at N=1 and N=2 it was actually slower than the plain Python loop (0.52×, 0.69×), and at higher N it merely tied. The training-runner's diagnosis: on CPU, the gridworld is so small (5×5 grid, ~28 state leaves, scalar reward) that `vmap` has no real work to parallelise; meanwhile the per-step JAX↔numpy boundary cost (one `np.array()` copy per step) dominates wall time at low N.

**The hypothesis this spike tests**: if we move JAX to the GPU, the vmapped kernel becomes genuinely parallel hardware work, and at N=4 we should clear the 1.5× speedup bar that v1 missed. **The risk**: torch already owns the GPU (the DreamerV3 world model lives on cuda:3), so JAX-on-GPU introduces (a) memory-allocator contention with torch, and (b) a GPU→host device-copy on every step at the numpy boundary, which could swamp any parallel-kernel win.

**The spike chooses ONE boundary strategy** — the cheapest one — and tests it. If it clears the bar, we declare GPU placement viable and v2.1 hardens it (accumulators, noise, per-env seed). If it misses, we decide whether a more expensive boundary strategy (DLPack zero-copy GPU↔GPU) is worth a follow-up spike, or we abandon the JAX-vector-env path entirely and look elsewhere for parallelism.

The single GO/NO-GO criterion is **G4**: at `num_envs=4`, JAX-vmap (GPU) SPS ≥ 1.5× SyncVectorEnv 4-env baseline (640 SPS measured on `2kkrsh1k`), i.e. ≥ 960 SPS. Close-miss in [768, 960] triggers a 3000-step rerun to amortize JIT-compile overhead before declaring fail. Anything <768 is hard fail.

---

## Why v1 failed (one paragraph)

v1's G3 sweep on n114 cuda:3 at 1000 steps showed JAX-vmap **slower** than the Python loop at N=1 (0.52×) and N=2 (0.69×), and only tying at N=4–16 (1.01–1.05×). Full numbers in the [v1 plan](../../archive/sheeprl_bridge/JAX_VECTOR_ENV_V1_PLAN.md). The diagnosis: `vmap` on CPU has no parallel hardware to use — XLA-CPU compiles it to a loop with auto-vectorised inner ops, which for a 5×5 env with a handful of int operations gives no measurable win. Per-step Python↔JAX boundary cost (the v1.0 `np.array(jax_arr)` copy plus the `jax.tree.map` auto-reset blend) dominates at low N. **GPU placement is the only intervention left** that could plausibly flip the result — vmap on GPU is real parallel hardware work — but it introduces a fresh set of risks (the rest of this plan addresses them).

---

## The key design decision: three boundary strategies, pick one

Once JAX is on the GPU, the JAX↔numpy boundary becomes a GPU→host device copy on every step (instead of a free host-memory copy on CPU). This is the single biggest unknown. Three candidates were considered:

### (a) Naive — keep `np.array(jax_arr)` at the boundary [CHOSEN FOR SPIKE]

- **Cost**: one GPU→host device-copy of obs + reward + done per step. For our env: obs is `(N, 19)` float32 ≈ 0.3 KB at N=4, reward is `(N,)` float32 = 16 B, done is `(N,)` bool = 4 B. Total per-step copy ≈ 0.35 KB. PCIe Gen3 x16 (modern lab nodes) sustains ~12 GB/s effective for tiny transfers; the *bandwidth* cost is trivial. The real cost is **launch + sync latency** — each `device_get` triggers a CUDA stream sync that stalls the JAX kernel completion (~5–20 µs per call, hardware-dependent). At 686 SPS the per-step budget is 1.46 ms, so a 10–20 µs sync is 0.7–1.4% overhead. Tolerable.
- **Pros**: zero changes to sheeprl's consumer side. The downstream code (`run_dreamer_v3.py:351-423`) consumes obs / reward / done as numpy arrays and writes into them in-place (the read-only crash fix from v1.0 already established this). Numpy boundary is the only path that lets us reuse all that code unchanged.
- **Cons**: if device-sync latency turns out to be the dominant cost at N=4, this option caps the achievable speedup well below the parallel-kernel ceiling.
- **Dev cost**: ~3 lines (env-var changes + remove the `JAX_PLATFORMS=cpu` line in `jax_vector_env.py`). Truly minimum-effort.

### (b) DLPack — zero-copy GPU↔GPU via `jax.dlpack` + `torch.utils.dlpack`

- **Cost**: zero device-copy; the JAX device tensor is reinterpreted as a torch device tensor without leaving GPU memory.
- **Pros**: theoretical maximum speedup.
- **Cons**: requires patching **every downstream consumer** of the obs / reward / done in `run_dreamer_v3.py`. The replay buffer at line 396 stores `next_obs[k][np.newaxis]` — that's numpy indexing. Line 351 builds `dones = np.logical_or(terminated, truncated)` — pure numpy. Line 393 mutates `real_next_obs[k][idx] = v` in-place — numpy assignment. Converting to torch tensors throughout means a multi-site sheeprl vendor patch.
- **Dev cost**: estimated 6–10 patch sites, all in `run_dreamer_v3.py` plus the replay-buffer interface. **Outside spike budget.** Documented in the "future v2.1 ideas" appendix for a follow-up if (a) falls short.

### (c) Hybrid — keep numpy boundary, sync only at episode boundaries

- **Cost**: same numpy boundary as (a), but reduce sync frequency by buffering several JAX-only steps before a host copy.
- **Cons**: sheeprl's loop drives one step at a time and consumes obs immediately to compute the next action (the policy is torch on GPU). You cannot defer the obs copy — the action selection needs it now. Option (c) only works if the policy is also moved into JAX, which is a fundamental architecture change. **Out of spike scope by orders of magnitude.**

### Decision

**Option (a) is the only option that fits a spike budget.** The spike answers: "with naive numpy boundary, does GPU placement deliver ≥1.5× at N=4?". If yes → ship (a), close the spike, plan v2.1 for accumulators + production configs. If no, with the close-miss criterion in G4 triggering a 3000-step rerun, and if the 3000-step rerun still misses → **STOP**, write a memory insight, escalate to user with the data on whether option (b) is worth a separate, larger spike.

---

## Hard constraints

These must be in the implementation, not optional:

1. **GPU memory contention with torch.** JAX defaults to `XLA_PYTHON_CLIENT_PREALLOCATE=true`, which grabs 90% of GPU memory at startup. Torch reserves CUDA memory lazily but aggressively. Without bounding JAX, the two collide on torch's first parameter allocation and one of them OOMs. The launch script must export:
   - `XLA_PYTHON_CLIENT_PREALLOCATE=false`  (disable preallocation; JAX grows on demand)
   - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`   (cap JAX's max share at 20% of the visible GPU)
   The 20% is a generous ceiling — our env state is ~28 small int/float arrays × N=4–16 ≈ kilobytes of GPU memory in practice. Even 5% would suffice, but 20% leaves headroom for XLA compile artefacts. Verified safe in G1.

2. **JAX_PLATFORMS must NOT be forced to CPU.** v1 hardcodes `os.environ.setdefault("JAX_PLATFORMS", "cpu")` at line 19 of `jax_vector_env.py` AND `export JAX_PLATFORMS=cpu` at line 50 of `launch_sheeprl.sh`. Both must be removed (or changed to `cuda` / unset) for JAX to see the GPU. The same `CUDA_VISIBLE_DEVICES=$GPU` line that pins torch to one GPU also pins JAX to the same GPU — both libraries see the same single device — which is exactly what we want.

3. **The toggle stays the existing `env.use_jax_vector_env` Hydra flag.** No new `jax_device: cpu|gpu` knob is added; the spike's mode is "JAX on GPU, period". Justification: a spike that adds a config-surface knob is no longer a spike. If GPU placement clears G4 we ship it as the only mode; if it misses we delete it. Either way, an intermediate `cpu|gpu` toggle is dead weight. (User's instinct in the prompt matched this; this paragraph confirms.)

4. **Gate criterion is G4-verbatim from v1**, restated below for the record:
   > **G4**: At `num_envs=4`, JAX-vmap (now GPU) SPS ≥ 1.5× SyncVectorEnv 4-env baseline (640 SPS, `2kkrsh1k`). Numerically: ≥ 960 SPS. Close-miss [768, 960] triggers a 3000-step rerun. <768 = hard FAIL.

## Risks the spike must surface, not bury

- **JIT-compile cost on GPU is higher than on CPU.** XLA-GPU compile of `jax_step` + `jax_reset` + `get_observation` is empirically 5–15× slower than XLA-CPU compile. At 1000 steps the compile is amortized but visible; at 3000 steps it should be negligible. The close-miss [768, 960] re-run at 3000 steps explicitly tests for this.
- **First-step OOM-fight with torch.** Even with `MEM_FRACTION=0.2`, the order of init matters: if JAX touches the GPU before torch reserves its parameter slabs, torch might fail. The `JAXVectorEnv.__init__` runs in `run_dreamer_v3.py` line 116 **before** `build_agent(...)` at line 178, so JAX touches GPU first and torch second. With JAX bounded to 20%, torch has 80% left, which is plenty for DreamerV3-XS (the model is ~10–20 MB of parameters). G1 explicitly verifies this ordering does not OOM.
- **Multi-GPU separation alternative (flagged, not in spike).** If contention or sync latency turns out to dominate, an alternative is running JAX on cuda:1 and torch on cuda:3, fully sidestepping both problems. But this introduces a PCIe cross-device copy on every step (instead of an intra-device GPU→host copy), and requires separate `CUDA_VISIBLE_DEVICES` masks for the two libraries — non-trivial to plumb through the launcher. **Out of spike scope**; recorded here as a v2.2 path if v2.1 also misses.

## Halt-at-design-phase clause (explicit gate)

If during implementation the developer discovers that option (a)'s numpy boundary is structurally impossible (e.g. a sheeprl call-site requires the obs to already be on the same device as the torch policy, not a numpy array), the developer must **STOP** before writing G2/G3 protocol code and escalate to senior-developer. The spike does not silently expand into option (b).

This is unlikely — v1 already proved the numpy boundary works for end-to-end sheeprl training (the v1 G2 smoke ran 1000 steps cleanly after the read-only fix). The plan flags it as a gate because the user explicitly requested it.

---

## File Changes

Three small files. Total expected diff: ~10 lines added, ~3 lines removed.

### 1. `pytorch_agents/pytorch_agents/envs/jax_vector_env.py` (lines 16–19, modify)

```python
# BEFORE (lines 16–19):
# Force JAX to CPU before any JAX import (PyTorch owns the GPU).
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# AFTER:
# JAX device is controlled by the launcher (XLA_PYTHON_CLIENT_*); do NOT
# force a platform here. v2 spike places JAX on the same GPU as torch so the
# vmap kernel runs as real parallel hardware work; v1 forced CPU and missed
# the speedup bar. See JAX_VECTOR_ENV_V2_GPU_SPIKE.md for the design.
import os
import sys
```

That's it for this file. The boundary helper `_to_numpy()` (the v1.0 read-only fix) stays unchanged — `np.array(jax_device_arr)` already handles GPU→host transfer correctly (JAX inserts a `device_get` automatically when the destination is host).

### 2. `scripts/launch_sheeprl.sh` (lines 49–51, modify)

```bash
# BEFORE (lines 49–51):
export GWP_CONFIG_PATH="$CONFIG"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES="$GPU"

# AFTER:
export GWP_CONFIG_PATH="$CONFIG"
# v2 spike: JAX shares the same GPU as torch (bound to 20% of GPU memory).
# Preallocate=false is mandatory — otherwise JAX would grab 90% at startup
# and OOM-fight torch's parameter slabs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export CUDA_VISIBLE_DEVICES="$GPU"
```

The deleted `export JAX_PLATFORMS=cpu` line lets JAX auto-detect the GPU made visible by `CUDA_VISIBLE_DEVICES`. The two `XLA_PYTHON_CLIENT_*` vars bound JAX so it does not collide with torch.

### 3. `pytorch_agents/pytorch_agents/run_dreamer_v3.py` (no change required)

GWP-PATCH-C from v1 already instantiates `JAXVectorEnv` when `cfg.env.use_jax_vector_env=true`. No code change here. The Hydra flag stays as-is — the user's instinct is confirmed: the env-var change in the launcher controls device placement implicitly.

### Configs unchanged

No new config keys, no YAML edits. The existing `env.use_jax_vector_env: false` default still rolls back to SyncVectorEnv with zero behaviour change.

---

## Implementation steps + pre/post-conditions

1. **Apply file change #1** (`jax_vector_env.py`): delete the one-line `os.environ.setdefault("JAX_PLATFORMS", "cpu")` and update the comment. Pre-condition: v1.0 plan is in `Verified`. Post-condition: `git diff` shows ONLY that file modified with ≤ 5 lines net change.
2. **Apply file change #2** (`launch_sheeprl.sh`): replace the `JAX_PLATFORMS=cpu` line with the two `XLA_PYTHON_CLIENT_*` lines. Post-condition: `git diff` net change ≤ 5 lines.
3. **G1** — instantiation + co-residency check (developer-side, ~30 seconds on a lab node).
4. **G2** — end-to-end 1000-step smoke at N=4 (one run).
5. **G3** — SPS sweep `N ∈ {1, 2, 4, 8, 16}` at 1000 steps (5 runs).
6. **G4** — read the N=4 cell, compare to 640 SPS × 1.5 = 960. Decide GO / close-miss / FAIL per the gate.
7. **(close-miss only)** — re-run G3 at 3000 steps; decide GO / FAIL.

If any of G1 / G2 fails, halt and escalate; do not proceed to the next gate.

---

## Smoke + benchmark protocol

All gates run on **n114 cuda:3** for direct comparison with v1's baseline. Same conda env (`sheeprl_bridge`).

### G1 — instantiation + co-residency (one minute)

A tiny standalone check the developer runs once during implementation. Verifies (a) JAX sees the GPU, (b) JAX-on-GPU + torch-on-same-GPU co-exist without OOM.

```python
# tmp/jax_vector_env_v2_g1.py — DO NOT COMMIT
import os
# Mimic launch_sheeprl.sh env-var setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import jax, torch, numpy as np
print("jax devices:", jax.devices())            # expect: [CudaDevice(id=0)]
print("torch cuda available:", torch.cuda.is_available())  # expect: True

# Allocate a small torch tensor on cuda:0 (mapped from cuda:3 by CUDA_VISIBLE_DEVICES)
t = torch.zeros(1024, 1024, device="cuda:0")
print("torch tensor on", t.device, "OK")

from pytorch_agents.envs.jax_vector_env import JAXVectorEnv
ve = JAXVectorEnv(
    config_path="/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/dreamer_curriculum/01_food_only.yaml",
    num_envs=4, seed=42,
)
obs, info = ve.reset()
assert obs["state"].shape[0] == 4, obs["state"].shape

actions = np.array([0, 1, 2, 3], dtype=np.int32)
obs2, r, term, trunc, info2 = ve.step(actions)
assert obs2["state"].shape == obs["state"].shape
assert not np.any(np.isnan(obs2["state"]))

# Co-residency check: torch tensor still alive after JAX work
assert t.sum().item() == 0
print("G1 PASS: JAX on GPU, torch on same GPU, no OOM, env steps OK")
```

**Pass**: prints the three lines, no exception, no OOM. **Fail**: `RuntimeError: CUDA out of memory` (means MEM_FRACTION needs lowering further, or torch must be allocated first — escalate before proceeding).

### G2 — 1000-step end-to-end smoke (one run, ~2 min)

```bash
./run_command.py 114 "bash scripts/launch_sheeprl.sh \
  configs/experiment/dreamer_curriculum/01_food_only.yaml \
  3 jaxvec_gpu_smoke_n4 1000 4 env.use_jax_vector_env=true"
```

**Pass**: sheeprl exits cleanly, WandB summary contains `agent.algorithm=DreamerV3` and any non-zero `Time/sps_env_interaction`. **Fail**: traceback (likely OOM or a CUDA stream-sync error) — escalate.

### G3 — SPS sweep `N ∈ {1, 2, 4, 8, 16}` × 1000 steps (5 runs, ~10 min)

```bash
for N in 1 2 4 8 16; do
  ./run_command.py 114 "bash scripts/launch_sheeprl.sh \
    configs/experiment/dreamer_curriculum/01_food_only.yaml \
    3 jaxvec_gpu_sps_n${N} 1000 ${N} env.use_jax_vector_env=true"
done
```

(Sequential — each waits for the previous so we don't share the GPU.) Fill this aggregation table (training-runner writes back):

| num_envs | wall (s) | sps (JAX-vmap GPU) | sps (Sync baseline) | speedup | wandb id |
|---|---|---|---|---|---|
| 1 | … | … | 686 (`6p386gwm`) | … | … |
| 2 | … | … | 660 (`no66f0j4`) | … | … |
| **4** | … | … | **640 (`2kkrsh1k`)** | … | **G4 cell** |
| 8 | … | … | 625 (`xggwhch8`) | … | … |
| 16 | … | … | 612 (`naoupbs4`) | … | … |

Read SPS from the LAST `Time/sps_env_interaction` value in each run's WandB summary (matches v1's protocol).

### G4 — GO/NO-GO gate (verbatim)

**At `num_envs=4`, JAX-vmap (now GPU) SPS ≥ 1.5× SyncVectorEnv 4-env baseline (640 SPS, `2kkrsh1k`). Numerically ≥ 960 SPS. Close-miss [768, 960] triggers a 3000-step rerun. <768 = hard FAIL.**

- **GO (≥ 960)**: spike succeeds. Close the spike. Plan v2.1 for accumulator preservation + production configs + per-env seed + terminated/truncated split (the existing v1 v2-backlog items).
- **Close-miss [768, 960]**: re-run G3 at `total_steps=3000` on the same hardware. If the re-run hits ≥ 960 → GO (JIT-compile cost was the bottleneck, amortized away). If still < 960 → FAIL.
- **FAIL (< 768, or close-miss re-run still < 960)**: STOP. Write a memory insight under `docs/memory/dreamer_diagnosis/` titled "JAX-vmap-GPU also misses 1.5× — boundary cost dominates". Mark this spike doc as superseded with a `superseded_by:` pointing to the memory-insight path. Escalate to user with the option-(b) DLPack plan as a possible follow-up spike, but **do not implement it without explicit user approval**.

---

## Reversibility

- `env.use_jax_vector_env: false` is the YAML default — anyone running an existing command (no Hydra override) goes through the unchanged SyncVectorEnv path. **The spike's risk is bounded to whoever explicitly opts in.**
- To revert entirely: re-add the `JAX_PLATFORMS=cpu` line in both files (one in `launch_sheeprl.sh`, one in `jax_vector_env.py`). ~3 minutes.
- The `XLA_PYTHON_CLIENT_*` env vars only affect JAX, which only runs in the explicit-opt-in code path; they cannot break SyncVectorEnv runs.

## Future v2.1 ideas (NOT in this spike, recorded for the next planning pass)

- Option (b) — DLPack zero-copy boundary. ~6–10 sheeprl patch sites. Only justified if option (a) misses by a large margin AND we still believe parallel env collection is the right bottleneck to chase.
- Multi-GPU JAX/torch split (JAX on cuda:1, torch on cuda:3). Sidesteps contention; introduces a cross-device PCIe copy per step. Worth ~30 minutes of investigation if option (a) misses for clearly-contention-shaped reasons (e.g. torch slowdown observed alongside flat env SPS).
- Accumulator preservation (BM / distance / episode) — required before any production run that reads `Episode/*` WandB metrics. ~150 LoC across `src/behavior/*` and the vector env. Not on the critical path for the spike.
- `apply_noise=True` path for `perceptual_noise.enabled: true` configs.
- Per-env seed convention (`cfg.seed + rank * num_envs + i`).
- Proper `terminated` vs `truncated` split via `termination_reason` info field.

---

## Cross-references

- [v1 plan](../../archive/sheeprl_bridge/JAX_VECTOR_ENV_V1_PLAN.md) (superseded by this spike) — full file-change history, v1 G3 numbers, read-only-buffer bug-fix report.
- [Parallel-env benchmark + audit](PARALLEL_ENV_BENCHMARK.md) — original audit that established the SyncVectorEnv-is-sequential diagnosis.
- [Diary 2026-05-12 progress report](../../../diary/2026-05-12.md) — v1 G3 result entry that triggered this spike.
- [CLAUDE.md doc-framing rule](../../../../CLAUDE.md) — first-section plain-English entry-point requirement (this doc's "Question" section satisfies it).

## Handoff

- **To developer**: apply the two file changes above. Run G1. Run G2. Stop and report after G2; do **not** launch the G3 sweep until senior-developer signs off the verification of G1+G2. Specifically check the developer report includes (a) the `jax.devices()` output from G1, (b) the WandB run ID from G2, and (c) confirmation no OOM was hit at start-of-train.
- **To senior-developer (verification, post-G2)**: read the developer's report; verify the diff is bounded to the two files above; sign off on G1+G2 verification before authorizing G3.
- **To training-runner (after G2 sign-off)**: launch the 5-run G3 sweep on n114 cuda:3, sequential, fill the aggregation table, decide G4 GO / close-miss / FAIL per the gate. If close-miss, launch the 3000-step re-run automatically; if FAIL, escalate per the halt clause.

## Implementation Report

> **Implemented by**: developer (Claude Sonnet 4.6)
> **Date**: 2026-05-13

### Summary

#### Files changed (per plan)

1. **`pytorch_agents/pytorch_agents/envs/jax_vector_env.py`** — removed `os.environ.setdefault("JAX_PLATFORMS", "cpu")` (line 19) and replaced the surrounding comment with the v2 rationale per the plan's File Changes spec. Net: -1 line code, +3 lines comment.

2. **`scripts/launch_sheeprl.sh`** — replaced `export JAX_PLATFORMS=cpu` with the two `XLA_PYTHON_CLIENT_*` env vars and a comment block per the plan's File Changes spec. Net: -1 line, +5 lines.

3. **`pytorch_agents/pytorch_agents/envs/grid_world_pain.py`** — NOT changed. The user's instruction mentioned this file; the plan does not. Analysis confirmed it is not imported when `use_jax_vector_env=True` (it is in the `else` branch only), so removing its `JAX_PLATFORMS=cpu` is out of spike scope and was left for flagging.

#### `grep` confirmation

```
grep -rn "JAX_PLATFORMS" pytorch_agents/ scripts/
→ pytorch_agents/pytorch_agents/envs/grid_world_pain.py:21:  (SyncVectorEnv path only — correct)
```

No remaining `JAX_PLATFORMS` references in `jax_vector_env.py` or `launch_sheeprl.sh`.

#### Commit

`5ed4c6f` — `feat(jax_vector_env): ⚡ v2 spike — place JAX on GPU alongside torch (XLA_PYTHON_CLIENT_PREALLOCATE=false)`

---

### G1 Attempt 1 — BLOCKER (halt-at-design-phase, 2026-05-13 previous session)

G1 was run on n114 using the `sheeprl_bridge` conda env (the env the launcher uses):

```
jax.devices(): [CpuDevice(id=0)]
JAX platform: cpu
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed.
Falling back to cpu.
AssertionError: FAIL: expected 'gpu', got 'cpu'
```

**Root cause**: `jaxlib 0.10.0` in the `sheeprl_bridge` env is the **CPU-only build** — it has no CUDA backend compiled in. Setting `CUDA_VISIBLE_DEVICES` has no effect on a CPU-only jaxlib; JAX cannot see any GPU regardless of env vars.

For comparison, the `grid_world_pain` env (jax 0.9.0.1) has CUDA jaxlib installed and sees all 4 GPUs cleanly.

---

### G1 Attempt 2 — NEW BLOCKER: PTX version mismatch (2026-05-13 current session)

User approved installing CUDA jaxlib in `sheeprl_bridge`. Installed the following packages with `--no-deps`:

```
jax-cuda12-plugin==0.10.0   (6.7 MB, from storage.googleapis.com/jax-releases/jax_cuda_releases.html)
jax-cuda12-pjrt==0.10.0     (164.8 MB)
nvidia-cudnn-cu12==9.22.0.52 (718.4 MB, upgraded from 9.1.0.70 — required because plugin compiled with CuDNN 9.8+)
```

Step 2 verification confirmed JAX sees GPU:
```
jax.devices(): [CudaDevice(id=0)]
platform: gpu
```

G1 re-run then failed at the first JAX compute call during `JAXVectorEnv.__init__` (`load_env_params` → `jnp.array`):

```
jax.errors.JaxRuntimeError: INTERNAL: CUDA error: ptxas application ptx input, line 5;
fatal   : Unsupported .version 8.8; current version is '8.2':
CUDA_ERROR_UNSUPPORTED_PTX_VERSION: the provided PTX was compiled with an unsupported toolchain.
```

**Root cause**: `jax-cuda12-plugin==0.10.0` was compiled against CUDA 12.8+ (PTX ISA 8.8). n114's NVIDIA driver 535.230.02 supports CUDA 12.2 (PTX ISA 8.2 maximum). This is a **fundamental incompatibility** — no env-var or install flag can bridge it.

**Evidence**:
- `nvidia-smi` on n114: `Driver Version: 535.230.02   CUDA Version: 12.2`
- `grid_world_pain` env works with GPU because it uses `jax-cuda12-plugin==0.9.0.1`, which was compiled for CUDA 12.2 (PTX 8.2 or lower).
- jax 0.10.0 requires **exactly** `jax-cuda12-plugin==0.10.0` — no older plugin version can be used with this jax version.

**kfsvh1qk confirmed alive**: PID 48052, running 272+ min on GPU 0 (UUID GPU-ac4d1be1), consuming ~1938 MiB. The pip installs did NOT affect the running process.

---

### Deviations from plan

| Deviation | Reason |
|-----------|--------|
| G1 failed (halt) × 2 | First: CPU-only jaxlib. Second: PTX version mismatch — `jax==0.10.0` requires CUDA 12.8+, n114 has CUDA 12.2 |
| G2/G3/G4 not run | Correctly halted per halt-at-design-phase clause |
| `grid_world_pain.py` not changed | Not in the plan's File Changes; not on the JAX-path code path; flagged here rather than silently changed |

---

### Speed check

Not applicable — G1 failed before any training run.

---

### Blockers for user (escalation required)

The `jax==0.10.0` pin is incompatible with n114's CUDA 12.2 driver for GPU acceleration. Options to unblock:

**Option A — Downgrade jax to 0.9.x in `sheeprl_bridge`** (recommended, lowest risk)
- `jax==0.9.0.1` + `jax-cuda12-plugin==0.9.0.1` is proven to work on n114 (already used by `grid_world_pain` env on the same node).
- `jax` 0.9.x and 0.10.x have no API changes relevant to the gridworld env code (`vmap`, `jit`, `lax.scan`).
- Install: `pip install 'jax==0.9.0.1' 'jax-cuda12-plugin[with-cuda]==0.9.0.1' 'jax-cuda12-pjrt==0.9.0.1' --no-deps --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- Risk: sheeprl or flax may have a transitive jax version constraint — check `pip check` after install.

**Option B — Upgrade n114 driver to ≥560.x (CUDA 12.6+) or ≥570.x (CUDA 12.8+)**
- Requires system admin action. Not something the developer can do.
- If the driver is upgraded to ≥570.x, `jax==0.10.0` with its CUDA plugin would work as-is with no code change.
- Risk: driver upgrade could break the running kfsvh1qk process and other training runs.

**Option C — Run the spike on a different node** with a newer driver (≥570.x)
- If another lab node (101–113) has a newer driver, the spike could run there with `jax==0.10.0`.
- n114 would remain as the kfsvh1qk host.

**Developer recommendation**: Option A. Downgrading jax in `sheeprl_bridge` from 0.10.0 to 0.9.0.1 unblocks the GPU spike without any driver change, system admin action, or process disruption. The `--no-deps` install pattern is established and safe for this env (per the `20260512_1756_pip_install_namespace_shadow_numpy_cap` memory insight). Requires user approval to change the jax pin.

---

**Signed**: Implemented by: developer

## Verification Report

> **Verified by**: (senior-developer — to fill)
> **Date**: (to fill)

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `pytorch_agents/pytorch_agents/envs/jax_vector_env.py` | -1 line (`JAX_PLATFORMS=cpu` setdefault) | | |
| `scripts/launch_sheeprl.sh` | -1 line, +2 lines (XLA env vars) | | |

**Speed check**: (record G3 sweep result + G4 verdict — to fill)

**Conclusion**: (one-line — to fill)
