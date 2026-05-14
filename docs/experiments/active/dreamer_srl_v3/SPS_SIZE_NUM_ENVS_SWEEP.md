---
title: "Dreamer-SRL v3 — SPS × num_envs × network-size feasibility sweep (node 114)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
wandb_tag: dreamer_srl_sweep
---

# Dreamer-SRL v3 — SPS × num_envs × network-size feasibility sweep (node 114)

> **Status**: PLANNED — 2026-05-14
> **Author**: experiment-designer
> **Related**:
> - [PI call — D-013 / parity-launch disposition](../../../pi/calls/2026-05-14_d013_parity_launch_disposition.md) (decided "Go with XS"; this sweep informs whether XS-single-env is the right launch point or whether a bigger size / multi-env launch is now feasible on node 114)
> - [Config-correction plan](../../../develop/active/dreamer_srl_v3/CONFIG_CORRECTION_PLAN.md) (the corrected XS values this sweep extends)
> - [Sheeprl bridge parallel-env benchmark](../../../develop/active/sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md) (the analogous benchmark on the PyTorch sheeprl side — found env-collection is single-threaded, so multi-env did not speed up that stack; this sweep tests whether the JAX dreamer-srl stack behaves the same)

---

## 1. Context (plain language)

We are about to launch the **parity gate** for the JAX dreamer-srl rebuild — a 3-seed, ~25-hour-each training run that confirms our JAX port reaches the same survival score as the upstream PyTorch reference (`sheeprl`) on a small grid-world. Before spending ~75 GPU-hours on that gate, we want to know what **hardware envelope** is actually available on node 114 — the lab cluster's biggest single-GPU surface (4× RTX 6000 Ada, 49 GB VRAM each). That envelope decides three things: (1) how many environment copies we should run in parallel (`num_envs`), (2) whether a bigger network size than the parity target (XS) actually fits and trains at a reasonable speed, and (3) where the **JIT-compile out-of-memory boundary** lands — JAX compiles the forward+backward pass into a single fused graph the first time it runs, and that compile pass spikes GPU memory roughly proportionally to model size × env count. If the graph compiles, the run is fine. If it OOMs at compile, the run dies before the first training step.

The agent is **DreamerV3** — a model-based RL algorithm that learns a small "world model" of the environment, then trains a policy by rolling out imagined trajectories inside that world model. Sheeprl ships **five named size presets** for it — **XS, S, M, L, XL** — that scale the dense-layer width (256 → 1024), the number of stacked MLP layers (1 → 5), and the recurrent-state width (256 → 4096). XS is the parity target. XL is what `01_food_only.yaml` carried before today's config correction.

The sweep is a **2-D grid** — size {XS, S, M, L, XL} × num_envs {1, 2, 4, 8, 16}, 25 cells, 1000 env-steps each — and the deliverable is a table saying "this cell fits at this many steps-per-second (SPS), and this cell OOMs at compile". The result decides what we launch as the parity gate: keep the planned XS / num_envs=1 default, switch to XS / num_envs=N for throughput, or upsize to S / M for a stronger parity claim.

This sweep is **decision-support for the parity launch, not a separate research result.** No training conclusion is drawn from it — only a hardware-envelope map.

---

## 2. Research questions

This sweep is profiling, not hypothesis-testing — so the "questions" below are operational rather than scientific.

- **Q1 — Fit boundary.** For each of the 5 size presets, what is the largest `num_envs` that JIT-compiles successfully on a single RTX 6000 Ada (49 GB)? Equivalently: where does the **OOM frontier** on the 5×5 grid lie?
- **Q2 — Steady-state throughput.** For each cell that fits, what is the steady-state SPS (`Time/sps_env`, the driver's logged metric — env-steps per wall-clock second, averaged over the post-JIT-warmup window)?
- **Q3 — Multi-env payoff at XS.** Does increasing `num_envs` from 1 to 16 at the XS preset meaningfully increase throughput, or is the rest of the training step (gradient update + replay sample) the bottleneck? If multi-env does not speed things up at XS (mirroring the 2026-05-11 finding on the sheeprl PyTorch side), the parity launch can stay at `num_envs=1`. If it does speed things up, the parity launch should run multi-env.
- **Q4 — Bigger-size feasibility.** At `num_envs=1`, does any size larger than XS (S, M, L, XL) fit on a single RTX 6000 Ada, and at what SPS? If S or M fits with steady-state SPS comparable to XS at multi-env, the parity claim could be strengthened by upsizing.

**Expectations (informal, sanity-only — these are not falsifiable predictions):**
- XS at every `num_envs` ∈ {1, 2, 4, 8, 16} fits comfortably (the corrected XS is ~250 MB of model params; even at num_envs=16 the compile graph is well below 49 GB).
- XL at `num_envs=1` likely fits on the 49 GB GPU (the 14.38 GB compile-memory observation from D-013 was on a 24 GB GPU). XL at `num_envs ≥ 4` likely OOMs.
- The OOM frontier most likely traces a diagonal from (size=XS, num_envs=16) safe → (size=XL, num_envs=2) at-the-edge.
- Multi-env at XS may or may not increase SPS — the sheeprl PyTorch finding (2026-05-11, see [PARALLEL_ENV_BENCHMARK.md](../../../develop/active/sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md)) says the env-collection step is not the bottleneck on that stack; the JAX dreamer-srl stack has the same single-threaded `ParallelEnv` design, so the same negative result is plausible. This sweep measures it.

None of the above are pass/fail criteria. The sweep collects facts; the parity-launch decision uses those facts.

---

## 3. Experimental design

### 3.1 Independent variables

| Variable | Values | Rationale |
|---|---|---|
| `size` | {XS, S, M, L, XL} | The five sheeprl named presets. Each corresponds to one agent-config YAML in `configs/dreamer_srl/01_food_only_<size>.yaml` (XS uses the unsuffixed `01_food_only.yaml`). Size-bearing keys taken byte-identically from `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_<size>.yaml`. |
| `num_envs` | {1, 2, 4, 8, 16} | Powers of 2 spanning the realistic range for a single-process `SyncVectorEnv`-style wrapper. `num_envs=1` is the parity-launch default; `num_envs=16` is the largest practical for a single-process step loop. Skipping {3, 5, 6, …} is intentional — the OOM frontier is monotone in `num_envs`, so a power-of-2 grid suffices to locate it. |

### 3.2 Fixed (controlled) variables — held constant across all 25 cells

| Variable | Value | Source |
|---|---|---|
| `algo.learning_starts` | 0 | Sweep-only deviation (skip §S3 prefill — pure profiling, no training). Same pattern as CP9 smoke per D-012. |
| `algo.replay_ratio` | 1 | sheeprl `dreamer_v3.yaml:L16` |
| `algo.per_rank_sequence_length` | 64 | sheeprl `exp/dreamer_v3.yaml:L15` |
| `algo.per_rank_batch_size` | 16 | sheeprl `exp/dreamer_v3.yaml:L14` |
| `algo.horizon` | 15 | sheeprl `dreamer_v3.yaml:L24` |
| `algo.stochastic_size` | 32 | sheeprl `exp/dreamer_v3.yaml:L23` |
| `algo.discrete_size` | 32 | sheeprl `exp/dreamer_v3.yaml:L24` |
| `total_steps` (CLI) | 1000 | Sweep budget — JIT warmup + ~100-step steady-state window. |
| `seed` | 0 | All cells use seed=0; the sweep measures throughput, not stochastic variance. |
| Env config | `configs/experiment/dreamer_curriculum/01_food_only.yaml` | The same 5×5 food-only environment used at CP10b. |
| Env | `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Lets JAX grow the pool dynamically; required so the driver does not silently consume 90% of the 49 GB regardless of need. |
| Hardware | node 114, RTX 6000 Ada (49 GB VRAM) — one GPU per cell | Cells are pinned to GPUs 1/2/3 (GPU 0 reserved for CP10b until ~2026-05-14T16:30). |

### 3.3 Dependent variables (what we measure)

| Variable | Source | Notes |
|---|---|---|
| Steady-state SPS | `Time/sps_env` in WandB, averaged over the last 70% of the run | The driver computes this each log iteration as `policy_step / max(time.time() - t_start, 1e-9)` (`src/algorithms/dreamer_srl/dreamer_srl_main.py:559`). Last-70% averaging skips the JIT-compile-dominated warmup. |
| JIT-compile success (binary) | Did the run complete `total_steps=1000` without OOM? | Pass = WandB run shows non-NaN losses and final-time print. Fail = process exit with CUDA OOM or JAX `RESOURCE_EXHAUSTED` error in stderr. |
| Peak GPU memory | `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` polled every 10 s during the run; the per-cell peak goes into the results table | Captured via the runner script's tee-to-log pattern (training-runner owns this). |
| Time-to-first-step (optional) | Wall-clock from `python` invocation to first `[iter N/total_iters]` stdout line | Diagnostic only; useful for understanding where memory pressure surfaces (compile vs. runtime). |

### 3.4 Confounds & limitations

| Confound | Severity | Notes / Mitigation |
|---|---|---|
| **JIT cache reuse across cells** — JAX persists compiled artifacts on disk by default. A second run of the same cell skips compile and reports artificially fast first-step time. | Low | The sweep visits each (size, num_envs) cell **once**, so cross-cell cache reuse is not possible (each cell has different shapes → different compile). Within a cell, time-to-first-step is reported only as a diagnostic, not a primary metric. |
| **Background load on the GPU** — if another job runs on the same GPU during a sweep cell, peak-memory and SPS measurements are contaminated. | Med | GPUs are pinned: GPU 0 → CP10b (locked until ~16:30), GPUs 1/2/3 → sweep. Runner enforces `CUDA_VISIBLE_DEVICES` per cell. |
| **`num_envs=1` is the only setting where the env-step is meaningfully observed; higher num_envs amortize env cost** — this is intentional (it's what we're measuring), not a confound. | — | — |
| **Single seed** — the sweep does not measure SPS variance across seeds. A given (size, num_envs) cell's SPS could vary ±5–10% across seeds due to GPU thermal state, system noise, etc. | Low | Acceptable for profiling. The parity launch (downstream) uses 3 seeds with variance reporting; the sweep does not need to. |
| **`learning_starts=0` deviates from parity** — the sweep runs without §S3 prefill, so the first 1024 steps that would normally be uniform-random instead come from the (zero-initialized, near-uniform) actor. SPS is barely affected by this distinction (actor forward is cheap compared to world-model train step). | Low | Documented as a sweep-only deviation. Parity launch uses `learning_starts=1024`. |

---

## 4. Launch manifest

System-of-record for every cell in this sweep. The **designer** writes the planned columns. The **training-runner** fills the actual columns (Node, GPU, Launched at, WandB run ID, Log path) at launch time, in place. The **experiment-analyzer** reads this table to find run folders and the results table in §6.

All 25 cells share:
- **wandb-group**: `dreamer_srl_sweep_2026-05-14`
- **wandb-job-type**: `profile`
- **Seed**: 0
- **wandb-project**: `grid_world_pain_dreamer_srl_sweep`
- **Node**: 114 (only)

| Run | Status | Cell (size, num_envs) | Tag (= wandb-name) | GPU | Launched at | WandB run ID | Log path |
|---|---|---|---|---|---|---|---|
| 1  | planned | (XS, 1)  | `dreamer_srl_sweep_XS_envs1`  | — | — | — | — |
| 2  | planned | (XS, 2)  | `dreamer_srl_sweep_XS_envs2`  | — | — | — | — |
| 3  | planned | (XS, 4)  | `dreamer_srl_sweep_XS_envs4`  | — | — | — | — |
| 4  | planned | (XS, 8)  | `dreamer_srl_sweep_XS_envs8`  | — | — | — | — |
| 5  | planned | (XS, 16) | `dreamer_srl_sweep_XS_envs16` | — | — | — | — |
| 6  | planned | (S, 1)   | `dreamer_srl_sweep_S_envs1`   | — | — | — | — |
| 7  | planned | (S, 2)   | `dreamer_srl_sweep_S_envs2`   | — | — | — | — |
| 8  | planned | (S, 4)   | `dreamer_srl_sweep_S_envs4`   | — | — | — | — |
| 9  | planned | (S, 8)   | `dreamer_srl_sweep_S_envs8`   | — | — | — | — |
| 10 | planned | (S, 16)  | `dreamer_srl_sweep_S_envs16`  | — | — | — | — |
| 11 | planned | (M, 1)   | `dreamer_srl_sweep_M_envs1`   | — | — | — | — |
| 12 | planned | (M, 2)   | `dreamer_srl_sweep_M_envs2`   | — | — | — | — |
| 13 | planned | (M, 4)   | `dreamer_srl_sweep_M_envs4`   | — | — | — | — |
| 14 | planned | (M, 8)   | `dreamer_srl_sweep_M_envs8`   | — | — | — | — |
| 15 | planned | (M, 16)  | `dreamer_srl_sweep_M_envs16`  | — | — | — | — |
| 16 | planned | (L, 1)   | `dreamer_srl_sweep_L_envs1`   | — | — | — | — |
| 17 | planned | (L, 2)   | `dreamer_srl_sweep_L_envs2`   | — | — | — | — |
| 18 | planned | (L, 4)   | `dreamer_srl_sweep_L_envs4`   | — | — | — | — |
| 19 | planned | (L, 8)   | `dreamer_srl_sweep_L_envs8`   | — | — | — | — |
| 20 | planned | (L, 16)  | `dreamer_srl_sweep_L_envs16`  | — | — | — | — |
| 21 | planned | (XL, 1)  | `dreamer_srl_sweep_XL_envs1`  | — | — | — | — |
| 22 | planned | (XL, 2)  | `dreamer_srl_sweep_XL_envs2`  | — | — | — | — |
| 23 | planned | (XL, 4)  | `dreamer_srl_sweep_XL_envs4`  | — | — | — | — |
| 24 | planned | (XL, 8)  | `dreamer_srl_sweep_XL_envs8`  | — | — | — | — |
| 25 | planned | (XL, 16) | `dreamer_srl_sweep_XL_envs16` | — | — | — | — |

### 4.1 Configs to produce (designer pre-launch)

All 25 runs use the same env config; the agent config picks the size preset. `num_envs` is passed via CLI `--num-envs`, not the YAML.

| Size | Agent config (relative to repo root) | Env config | Status |
|---|---|---|---|
| XS | `configs/dreamer_srl/01_food_only.yaml` | `configs/experiment/dreamer_curriculum/01_food_only.yaml` | **exists** (corrected 2026-05-14, commit `4fe3d8b`) |
| S  | `configs/dreamer_srl/01_food_only_S.yaml`  | same | **created this design** |
| M  | `configs/dreamer_srl/01_food_only_M.yaml`  | same | **created this design** |
| L  | `configs/dreamer_srl/01_food_only_L.yaml`  | same | **created this design** |
| XL | `configs/dreamer_srl/01_food_only_XL.yaml` | same | **created this design** |

**Note on the XS agent config in the sweep.** The parity-track XS config sets `learning_starts: 1024` and `total_steps: 5000`. For the sweep, the runner overrides these via CLI (`--total-steps 1000`) but the YAML's `learning_starts: 1024` is NOT CLI-overridable. The 5 sweep cells with XS will therefore burn ~1024 of their 1000 steps on random-action prefill, which is fine for SPS measurement (prefill exercises the env step but skips the train step — slightly inflates `Time/sps_env`). **Alternative if the analyzer wants strict apples-to-apples with the bigger sizes** (which all have `learning_starts: 0`): the runner can pass `--total-steps 2048` for the XS cells (1024 prefill + ~1024 train) so the steady-state window covers genuine train-step throughput. Recommendation: **use 2048 for the 5 XS cells, 1000 for the other 20 cells**, and let the analyzer note the difference in the results table.

### 4.2 Per-cell command template

Authoritative reference; the runner adapts as needed (e.g., adds env vars for GPU pinning, log redirect):

```bash
# Inside a script that sets CUDA_VISIBLE_DEVICES + XLA_PYTHON_CLIENT_PREALLOCATE
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=<GPU> \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    src/algorithms/dreamer_srl/dreamer_srl_main.py \
    --env-config configs/experiment/dreamer_curriculum/01_food_only.yaml \
    --agent-config configs/dreamer_srl/01_food_only<SIZE_SUFFIX>.yaml \
    --total-steps <1000_or_2048> --num-envs <N> --seed 0 \
    --wandb-project grid_world_pain_dreamer_srl_sweep \
    --wandb-name dreamer_srl_sweep_<size>_envs<N>
```

Where `<SIZE_SUFFIX>` is empty for XS (uses `01_food_only.yaml`) and `_S` / `_M` / `_L` / `_XL` for the four new variants.

### 4.3 Failure-handling policy

| Failure mode | Action |
|---|---|
| **CUDA OOM at JIT compile** (most common — log shows `RESOURCE_EXHAUSTED` from XLA) | Log the cell as **OOM**, do **not** retry. Move on to next cell. The OOM is a data point. |
| **CUDA OOM at runtime** (rare — would mean memory grew after compile) | Log as **OOM-runtime**, do not retry. Note in results table. |
| **Process crashes for non-memory reason** (e.g., NaN losses, segfault) | Log as **error**, do **not** retry. Note error class in results table. |
| **`Time/sps_env` is NaN at end of run** | Treat as soft fail — record the run as "completed but unmeasured" and continue. |
| **Wall-clock exceeds 30 minutes for any cell** | Kill the cell, log as **timeout**, continue. (1000 steps at the slowest expected SPS of ~5 should finish in ~3 minutes; 30 minutes is a generous timeout against runaway compile.) |

The runner **does not preempt other cells** when one fails. The whole sweep continues. If more than ~30% of cells fail with an unexpected pattern (e.g., not OOM but a real bug), the runner halts and notifies the user.

---

## 5. Execution plan

### 5.1 Parallelization — recommended: 3-way parallel across GPUs 1/2/3

| Constraint | Detail |
|---|---|
| GPU 0 status | CP10b training (single-GPU XS wall-clock, WandB `s31wc1a1`). Reserved until CP10b finishes (~30–90 min ETA at design time, 2026-05-14T15:24 launch). |
| GPUs 1, 2, 3 status | Idle; available for sweep cells. |
| Expected per-cell wall-clock | ~5 min for small cells (XS/S at low num_envs), ~15 min for large cells (L/XL at high num_envs that compile but run slowly). Use **20 min** as the worst-case budget for scheduling. |

**Batch pattern.** Group the 25 cells into 9 batches of ≤3 cells each. Within a batch, the 3 cells run in parallel on GPUs 1, 2, 3. The runner waits for all 3 to finish (or hit timeout), then launches the next batch.

Suggested batch ordering — **size-major, num_envs-minor** (so within each batch the 3 GPUs all do similar-size work, which keeps per-batch wall-clock predictable):

| Batch | Cells | GPU 1 | GPU 2 | GPU 3 | Est. wall-clock |
|---|---|---|---|---|---|
| 1 | XS  | (XS,1)  | (XS,2)  | (XS,4)  | ~5 min |
| 2 | XS  | (XS,8)  | (XS,16) | (S,1)   | ~5 min |
| 3 | S   | (S,2)   | (S,4)   | (S,8)   | ~8 min |
| 4 | S/M | (S,16)  | (M,1)   | (M,2)   | ~10 min |
| 5 | M   | (M,4)   | (M,8)   | (M,16)  | ~12 min |
| 6 | L   | (L,1)   | (L,2)   | (L,4)   | ~15 min |
| 7 | L   | (L,8)   | (L,16)  | (XL,1)  | ~15 min |
| 8 | XL  | (XL,2)  | (XL,4)  | (XL,8)  | ~20 min |
| 9 | XL  | (XL,16) | —       | —       | ~20 min |

**Total wall-clock estimate**: ~2.0–2.5 hours for the full sweep, three GPUs in parallel.

**Alternative (rejected): serial on one GPU.** Simpler but ~6.25 hours, no savings to justify the additional latency given that GPUs 1/2/3 are idle and the cells are throughput-bound, not memory-bound.

### 5.2 Sparse-grid fallback

If 25 cells is too many or wall-clock pressure increases (e.g., CP10b runs over, or a node 114 GPU drops out), the runner can execute a **9-cell sparse subset** that still localizes the OOM frontier:

- 4 corners: (XS, 1), (XS, 16), (XL, 1), (XL, 16)
- 4 edge midpoints: (XS, 4), (M, 1), (M, 16), (XL, 4)
- 1 diagonal interior: (M, 4)

The sparse sweep takes ~45 min on 3 GPUs in parallel. If a frontier emerges inside an unmeasured region, the runner infills 1–2 cells around it.

**Default recommendation**: run the full 25-cell sweep. Sparse-grid is a contingency.

---

## 6. Results

To be filled by the training-runner (manifest columns) and the experiment-analyzer (this section) once the sweep completes.

### 6.1 Headline result table

5×5 grid; each cell shows steady-state SPS (avg over last 70% of run) on the top line and peak GPU memory in MiB on the bottom line. OOM cells are marked `OOM` and the cell's failure mode is noted in §6.2.

|  | num_envs=1 | num_envs=2 | num_envs=4 | num_envs=8 | num_envs=16 |
|---|---|---|---|---|---|
| **XS** | — / — | — / — | — / — | — / — | — / — |
| **S**  | — / — | — / — | — / — | — / — | — / — |
| **M**  | — / — | — / — | — / — | — / — | — / — |
| **L**  | — / — | — / — | — / — | — / — | — / — |
| **XL** | — / — | — / — | — / — | — / — | — / — |

### 6.2 Per-cell notes (failure modes, JIT compile time, anomalies)

To be filled by analyzer.

### 6.3 OOM frontier description

Plain-language summary of where the OOM boundary lies, with a verdict on each of Q1–Q4 from §2.

### 6.4 Verdict for the parity launch

One of:
- **Stay at XS / num_envs=1**: multi-env did not buy SPS at XS.
- **Move to XS / num_envs=N**: multi-env increases SPS by ≥X%; use the cheapest N that captures most of the gain.
- **Upsize to S or M at num_envs=1**: a bigger size fits and runs at comparable SPS; strengthens the parity claim.
- **No change recommended**: sweep result is ambiguous; default to the planned XS / num_envs=1.

---

## 7. Forward-looking implications

This sweep is **decision-support for the parity gate**, not an independent research result. The output is a single one-line recommendation appended to the PI call's record:

> "Sweep result: parity launch will run at <size> / num_envs=<N> on node 114 GPU <X>, projected wall-clock <T>h per seed."

Three downstream consequences flow from that line:

1. **Parity-launch config selection.** If S or M fits at num_envs=1 with SPS comparable to XS at multi-env, the parity launch can be promoted to that bigger size. This strengthens the like-for-like claim against sheeprl's `dreamer_v3_XS` only if sheeprl's claim itself was at the same bigger size — which it is NOT (sheeprl baseline = real XS). So upsizing changes the parity-claim shape ("matches sheeprl XS *or larger*") rather than matching sheeprl 1:1. The user should consciously make this call rather than letting the sweep silently dictate it. Surface this as an `AskUserQuestion` if the sweep returns a "promote-to-S/M" recommendation.
2. **Parity-launch throughput envelope.** Three 25-hour seeds at sequential single-GPU XS = ~75 GPU-hours. Three 25-hour seeds at parallel single-GPU XS on 3 GPUs of node 114 = ~25 wall-clock hours. The sweep's SPS table lets us refine that 25h estimate per seed.
3. **Future scaling decisions for downstream neuromodulation experiments.** The neuromodulator hooks add per-step compute and a small amount of per-graph compile memory. Knowing the size×num_envs envelope at the bare-XS dreamer-srl scale is the starting point for projecting how much headroom the modulator can consume before the same envelope shrinks.

---

## 8. Pre-flight

This sweep is **profiling, not training** — no novel env settings, no obs/noise/sensor changes, no new schema. The agent configs are direct ports of sheeprl preset overlays. The five YAMLs have been parsed via `Config.load_yaml` and every mandatory key for the driver (`algo.learning_starts`, `algo.replay_ratio`, …, `algo.critic.optimizer.eps`) resolves cleanly (verified 2026-05-14 — see commit message). **No `env-config-auditor` pre-flight is strictly required** — the env config (`configs/experiment/dreamer_curriculum/01_food_only.yaml`) is unchanged from CP9/CP9b/CP10b which the auditor has already cleared.

If the user requests it, the auditor can re-check the four new size-variant YAMLs for structural parity against the corrected XS file; the expected diff is exactly the six size-bearing keys per the table in §3.1.

---

## Appendix A — Config diffs (each variant vs. corrected XS)

Each of S/M/L/XL differs from `01_food_only.yaml` (corrected real-XS) only in:
- 6 numeric size keys (per the §3.1 sheeprl-reference table)
- `algo.learning_starts: 1024 → 0` (sweep-only)
- `algo.total_steps: 5000 → 1000` (sweep budget)

All other keys (`gamma`, `lmbda`, `horizon`, `unimix`, `kl_*`, `actor.moments.*`, optimizer LRs / eps, buffer size, etc.) are byte-identical. Each variant carries inline citations to `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_<size>.yaml:L<n>` for every overridden value.

## Appendix B — Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-14 | Initial design + 4 size-variant configs (S/M/L/XL) authored. Sweep planned but not yet launched. | experiment-designer |
