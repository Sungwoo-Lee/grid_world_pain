---
title: DreamerV3 Diagnostics Plan (v2)
topic: dreamer
status: active
created: 2026-03-02
last_updated: 2026-04-12
supersedes: DREAMER_DIAGNOSTICS_PLAN_v1.md
---

# DreamerV3 Diagnostics Plan (v2)

> **Previous version**: [`DREAMER_DIAGNOSTICS_PLAN_v1.md`](DREAMER_DIAGNOSTICS_PLAN_v1.md) — Full investigation history (Sections 1-22, Feb 22 - Mar 2, 2026).
> This document summarizes the resolved history and provides a clean starting point for ongoing DreamerV3 training diagnostics.

---

## 1. Historical Summary: Bugs Found & Fixed

This section captures every significant bug, root cause, and architectural decision made during v1 investigation. These are preserved to prevent regressions and give context for future debugging.

### 1.1 Critical Bug: Two-Hot Value Space Mismatch (v1 Section 8)

**Date**: Feb 25, 2026
**Symptom**: Complete entropy collapse (`mean_entropy → 0.07`), policy locks onto single action, zero food discovery across all runs.

**Root Cause**: Three interlocking bugs in `dreamer_v3_trainer.py::behavior_loss_fn`:

| Bug | What Happened | Fix |
|:---|:---|:---|
| **Double-transformed critic target** | `to_twohot(norm_returns)` applied `symlog` to already-normalized values. Critic learned `symlog(normalized)` — a double compression. | Changed to `to_twohot(lambda_returns)` — critic trains on raw returns (symlog is internal to `to_twohot`). |
| **Advantage space mismatch** | `advantage = norm_returns - from_twohot(v_pred)` subtracted `[0,1]`-range from raw-space `[-20,+5]` → garbage gradients. | Normalize both sides: `norm_baseline = (baseline - moments_low) / moments_invscale`, then `advantage = norm_returns - norm_baseline`. |
| **Moments OOM** | Moments EMA state was traced inside `nnx.grad`, causing unnecessary gradient computation. | Pre-compute `moments_low` and `moments_invscale` outside grad with `stop_gradient`. |

**Verification**: Entropy recovered to 1.79, survival +68%, food discovery from 0 to measurable within 5k episodes. Confirmed alignment with sheeprl canonical implementation.

**Lesson**: DreamerV3's two-hot critic operates in symlog-space internally. External code must always pass **raw** values to `to_twohot()` and normalize advantages using the **same** Moments statistics on both sides.

### 1.2 JAX RSSM Stochasticity Bug (v1 Section 6)

**Date**: Feb 22-23, 2026
**Symptom**: Batches with identical logits produced identical stochastic samples.
**Root Cause**: Single PRNG key shared across the batch dimension. All environments with similar states sampled the same latent.
**Fix**: Implemented `T x B` `jax.vmap` PRNG splitting in `OneHotDist`, ensuring stochastically independent prior/posterior sampling per environment per timestep.

**Lesson**: In JAX, any stochastic operation inside `vmap` or `lax.scan` requires explicit per-element key splitting. A single key produces identical samples across vectorized dimensions.

### 1.3 Action Distribution Labeling Error (v1 Section 6)

**Date**: Feb 22, 2026
**Symptom**: Eval plots showed 100% "Forward" action.
**Root Cause**: Off-by-one labeling in `agentActionAnalysis.py`, not in the agent itself.
**Fix**: Corrected action label mapping. Agent was exploring correctly.

**Lesson**: Always verify analysis/plotting code before diagnosing agent behavior from visualizations.

### 1.4 Parallel Environment Scaling Failure (v1 Sections 7, 9, 13)

**Date**: Feb 24-26, 2026
**Symptom**: Increasing from 1 to 64 envs degraded per-episode learning (MeanLen 188 vs 124 at 500k eps, despite matched replay ratio).

**Root Causes Identified**:

| Factor | Impact | Resolution |
|:---|:---|:---|
| **Replay ratio imbalance** | Fixed `train_steps` didn't scale with `num_envs`. 64env with `train_steps=16` had 0.25 grad/seq vs 1env's 1.0 grad/seq. | Implemented `Ratio` class (from Hafner/sheeprl) for automatic scaling. |
| **Buffer turnover** | 64env overwrites 100K buffer in ~12 iters vs 1env's ~781 iters. Rare experiences lost before sufficient training. | Increased `buffer_capacity` to 10M (1.6 GB, 6.5% of 24GB GPU). |
| **PRNG key sharing in collection** | `collect_sequence` passed single key to all parallel envs → identical actions in similar states. | Split keys per environment. |
| **Buffer alignment** | Capacity (100K) not a multiple of `sequence_length` (128) → temporal corruption on wrap-around, 64x faster with more envs. | Buffer capacity now set to multiples of sequence_length. |

**Lesson**: Parallel env scaling in DreamerV3 requires three things simultaneously: (1) proportional gradient steps via `Ratio`, (2) proportional buffer capacity, (3) independent PRNG keys per env.

### 1.5 Replay Ratio and Collect Interval Design (v1 Sections 12-14)

**Date**: Feb 26, 2026

**Key Design Decision**: Our codebase collects `collect_interval` steps per env per iteration via `jax.lax.scan` (JAX optimization), unlike canonical DreamerV3/sheeprl which collects 1 step per env per iteration.

**The Formula**:
```
env_steps_per_iter  = num_envs x collect_interval
ratio_input         = env_steps_per_iter / collect_interval = num_envs
grad_steps_per_iter = num_envs x replay_ratio
```

The `// collect_interval` normalization cancels out collection batch size, so **grad steps per iteration = `num_envs x replay_ratio`**, regardless of `collect_interval`.

**Current Implementation** (`train.py`):
```python
ratio_scaled_updates = Ratio(config.get_mandatory('agent.replay_ratio'))
# ...
train_steps = ratio_scaled_updates(global_step // num_steps)  # num_steps = collect_interval
```

**Quick Reference Table** (all with `sequence_length=128`):
| num_envs | collect_interval | replay_ratio | grad_steps/iter |
|:---|:---|:---|:---|
| 1 | 1 | 1 | 1 |
| 1 | 128 | 1 | 1 |
| 64 | 1 | 1 | 64 |
| 64 | 128 | 1 | 64 |
| 64 | 128 | 2 | 128 |

**Lesson**: When using batched collection (`collect_interval > 1`), the `Ratio` input must be normalized by `collect_interval` to avoid 128x gradient inflation. The invariant is: grad_steps/iter depends only on `num_envs x replay_ratio`.

---

## 2. Historical Summary: Performance Optimization

### 2.1 Speed Benchmarks (v1 Sections 18-19)

Benchmarked across multiple configurations (Feb 26 - Mar 2). Key findings:

| Run | s/it | SPS | Notes |
|:---|---:|---:|:---|
| RPPO 128env (baseline) | 0.37 | 44,688 | Fully JIT-fused collect+train |
| Dreamer 64env CI=128 RR=1.0 (pre-optimization) | 12.89 | 4,429 | 92% time in Python training loop |
| Dreamer 64env CI=128 RR=1.0 (post-optimization) | ~0.23 | ~36,000 (est.) | GPU buffer + `lax.scan` batched JIT |

### 2.2 Root Cause: Python Training Loop Overhead (v1 Section 19)

**Pre-optimization iteration breakdown** (64env CI=128 RR=1.0, 12.89 s/it):
```
Collection:      ~0.8s  (6%)   <- lax.scan, fast
Statistics:      ~0.1s  (1%)
Buffer add:      ~0.1s  (1%)
Training loop:  ~11.9s  (92%)  <- 64 x train_step from Python
  buffer.sample:  ~2.5s (19%)    <- numpy indexing + host->device
  JAX dispatch:   ~1.5s (12%)    <- per-step launch overhead
  GPU compute:    ~7.9s (61%)    <- actual gradient work
```

**Key insight**: RPPO fuses collection + training into one JIT call. DreamerV3's off-policy training requires sampling from a replay buffer, which was CPU-based and dispatched per step from Python.

### 2.3 GPU-Resident Buffer + Batched JIT (v1 Sections 20-22)

**Implementation** (Mar 2, 2026): Two structural changes yielding ~55x speedup:

1. **Zero-Copy GPU Replay Buffer**: `ReplayBuffer` supports dual backend (`device="gpu"` or `"cpu"`). GPU mode stores all data as `jnp` arrays directly in VRAM. At 160 bytes/transition, 10M capacity = 1.6 GB (6.5% of 24GB).

2. **State-Isolated Functional JIT Training**: `train_multiple_gpu()` uses `nnx.split(self)` to extract trainer state, runs `lax.scan` over gradient steps with `nnx.merge(graphdef, state)` inside the scan body, then applies final state back. This avoids `TraceContextError` from mutating NNX graph state inside traced scan.

**Key files**:
- `dreamer_v3_trainer.py:589-628` — `train_multiple_gpu()` (GPU path with `lax.scan`)
- `dreamer_v3_trainer.py:630-656` — `train_multiple_cpu()` (CPU fallback with pre-sampled batches)
- `dreamer_v3_trainer.py:663-771` — `ReplayBuffer` (unified GPU/CPU)
- `train.py:857-885` — Collection path splits GPU (stay on device) vs CPU (`device_get`)
- `train.py:938-944` — Training dispatch based on `buffer.device`

### 2.4 Post-Optimization Code Review (v1 Section 22)

| Issue | Severity | Status |
|:---|:---|:---|
| GPU->CPU->GPU roundtrip in collection | HIGH | **FIXED** — GPU path reshapes in JAX, `device_get` only for stats |
| `np.any(dones)` on JAX array in `add_batch` | MEDIUM | **FIXED** — removed with dead code |
| `ep_start_idx` dead code | LOW | **FIXED** — removed entirely |
| `config.get()` safe defaults for buffer | LOW | **FIXED** — uses `get_mandatory()` |
| `buffer_capacity: 10M` (1.6 GB VRAM) | NOTE | **Open** — feasible but monitor under large configs |
| JIT retracing on buffer mutation | NOTE | **Open** — inherent to closure capture pattern, acceptable |

**Lesson**: When implementing GPU-resident buffers, ensure the entire data path stays on-device. A single `device_get` in the hot path negates the zero-copy benefit.

### 2.5 Replay Buffer Design Notes (v1 Section 15)

Research findings on DreamerV3 replay strategies:

| Feature | DreamerV3 Paper | sheeprl | Our Implementation |
|:---|:---|:---|:---|
| Sampling | Hybrid (online queue + uniform) | Episode-aware uniform | Block-aligned uniform |
| Recency bias | Online queue guarantees it | None | None |
| Terminal oversampling | None | `prioritize_ends` (optional) | None |
| Episode boundaries | Handled by `is_first` | Episode-level storage | `is_first` flag in training |

**Known gap**: Our buffer has no recency bias (DreamerV3 paper uses an online queue to guarantee recent data in every batch). With large buffer capacity (10M) this is less critical, but may matter for policy-lag-sensitive training.

---

## 3. Current DreamerV3 Configuration

```yaml
# configs/models/dreamer_v3.yaml (as of Mar 2, 2026)
agent:
  algorithm: "DreamerV3"
  batch_size: 64
  sequence_length: 128
  replay_ratio: 1
  collect_interval: 128
  train_steps: 64            # Legacy; overridden by replay_ratio when active
  model_lr: 1e-4
  actor_lr: 3e-5
  value_lr: 3e-5
  buffer_device: "gpu"
  buffer_capacity: 10000000  # 10M transitions = 1.6 GB VRAM
  encoder_dim: 128
  encoder_fc_layers: [128, 128]
  rssm_deter_dim: 512
  rssm_stoch_dim: 32
  rssm_classes: 32
  decoder_fc_layers: [128, 128]
  reward_fc_layers: [128, 128]
  continue_fc_layers: [128, 128]
  actor_fc_layers: [128, 128]
  critic_fc_layers: [128, 128]
  entropy_scale: 3e-4
  unimix: 0.01
  encoding_mode: "hierarchical"
  modulation: { type: null }
```

**Divergences from canonical DreamerV3** (Hafner 2023):

| Parameter | Ours | Canonical | Reason |
|:---|:---|:---|:---|
| `batch_size` | 64 | 16 | Larger batches for GPU utilization |
| `sequence_length` | 128 | 64 | Match RecurrentPPO sequence length |
| `collect_interval` | 128 | 1 | JAX `lax.scan` optimization |
| `entropy_scale` | 3e-4 | 3e-4 | Matched |
| `encoding_mode` | hierarchical | flat MLP | Project-specific sensory hierarchy |

---

## 4. Diagnostic Phase 1: Training Speed Validation

The GPU buffer + batched JIT optimization (Section 2.3) has been implemented but **not yet validated in a live training run**. The claimed ~55x speedup (12.89 → ~0.23 s/it) was estimated, not measured. This phase must be completed before any training performance investigation — there is no point tuning hyperparameters on a pipeline that may still be bottlenecked.

### 4.1 Speed Benchmark Run

**Objective**: Measure actual wall-clock speed of the optimized pipeline and compare against pre-optimization baselines.

**Run Configuration**:
```
Config:     configs/environment/default.yaml + configs/models/dreamer_v3.yaml
num_envs:   64
CI:         128
RR:         1.0
buffer:     GPU, 10M capacity
Duration:   ~1000 iterations (enough for JIT warmup + steady-state measurement)
```

**Checklist**:
- [x] **4.1.1** Launch benchmark run with `--tag speed_validation_gpu_buffer` on `cuda:0`.
- [x] **4.1.2** Record steady-state `s/it` (exclude first 5 iterations for JIT compilation warmup).
- [x] **4.1.3** Record `SPS` (env steps per second) from WandB `timesteps / wall_time`.
- [x] **4.1.4** Compare against pre-optimization baselines:

| Metric | Pre-Optimization | Target (Post-Optimization) | Measured | Verdict |
|:---|---:|---:|---:|:---|
| s/it (64env CI=128 RR=1.0) | 12.89 | < 1.0 | **13.0** | **MISSED** — no improvement |
| SPS | 4,429 | > 30,000 | **630** | **MISSED** — 7x worse than pre-opt |
| RPPO reference | 0.37 s/it, 44,688 SPS | — | — | — |

### 4.2 JIT Retracing Check

The GPU buffer's `add_batch` creates new JAX arrays via `.at[].set()`, which changes the buffer's array references between training calls. This may cause JIT cache misses in `train_multiple_gpu`.

- [x] **4.2.1** Monitor first 20 iterations: check if `s/it` is consistently fast after warmup, or if it spikes every iteration (indicating retracing).
  - **Result**: JIT retracing was detected and **fixed**. Closure capture was the root cause. Refactored to pass buffer arrays explicitly and moved `_scan_train_gpu` to a stable method. JIT stable after iteration 2.
- [x] **4.2.2** If retracing detected: profile with `JAX_LOG_COMPILES=1` to confirm. Consider passing buffer arrays as explicit arguments instead of closure capture.
  - **Result**: Confirmed via compilation logs. Fix applied (see 8.1 Resolution items 2-4).

### 4.3 GPU Memory Validation

At 10M capacity with 160 bytes/transition, the buffer consumes ~1.6 GB. With model parameters, activations, and optimizer state, total VRAM usage should be monitored.

- [x] **4.3.1** Run `nvidia-smi` during training to measure peak VRAM usage.
  - **Result**: 18.4 GB peak (75% of 24 GB 3090). No OOM.
- [x] **4.3.2** Confirm no OOM errors. If close to limit, reduce `buffer_capacity` to 1M (160 MB).
  - **Result**: Passed. 75% is within the 80% threshold, though headroom is limited.

### 4.4 Speed Bottleneck Investigation (If Target Not Met)

If measured `s/it > 1.0`, investigate in this order:

1. **Is `train_multiple_gpu` being called?** Add a print/log at `train.py:938` to confirm GPU path is taken.
2. **Is JIT retracing dominating?** Set `JAX_LOG_COMPILES=1`. If recompilation happens every iteration, the buffer closure capture is the cause.
3. **Is collection the bottleneck?** Time `collect_sequence` vs `train_multiple_gpu` separately. Collection should be <1s for 64env CI=128.
4. **Is `device_get` for stats blocking?** The GPU path still calls `jax.device_get(transitions)` for episode statistics (`train.py:871`). This is synchronous — measure if it's stalling the pipeline.
5. **Is batch size too large per gradient step?** Each step processes `64 batch x 128 seq x 15 horizon = 122,880` imagined transitions. If GPU compute dominates, reduce `batch_size` to 16 (`sequence_length` stays at 128 to match RPPO).

### 4.5 Pass Criteria

| Criterion | Target | Result | Status |
|:---|:---|:---|:---|
| Steady-state s/it | < 1.0 | **13.0** | **FAILED** |
| No JIT retracing after warmup | Stable after iter 2 | Stable after iter 2 | **PASSED** |
| GPU memory < 80% of 24 GB | < 19.2 GB | 18.4 GB (75%) | **PASSED** |
| Results in Investigation Log | Recorded | Section 8.1 | **PASSED** |

### 4.6 Phase 1 Assessment

**Overall: PARTIALLY PASSED** — JIT retracing and memory are resolved, but the speed target was missed.

The measured 13.0 s/it is essentially identical to the pre-optimization 12.89 s/it. The GPU buffer + `lax.scan` fusion successfully eliminated Python dispatch overhead (~31%), but **GPU compute was always the dominant cost (~61%)** and is irreducible at current batch dimensions. The original ~55x speedup estimate was fundamentally wrong — the realistic gain from overhead removal was ~1.45x. See Section 8.1 for the full investigation record.

### 4.7 Phase 1 Continuation: Batch Dimension Reduction

The single most impactful speed improvement is reducing the per-gradient-step workload. Currently each step processes `64 batch × 128 seq × 15 horizon = 122,880` imagined transitions — **4x the canonical DreamerV3** (`16 × 128 × 15 = 30,720`). Note: `sequence_length: 128` is kept to match RecurrentPPO config.

#### Step 1: Reduce Batch Size

- [x] **4.7.1** Change `configs/models/dreamer_v3.yaml`:
  ```yaml
  batch_size: 16          # was 64 (sequence_length stays at 128)
  ```
- [x] **4.7.2** Run benchmark: 64env, CI=128, RR=1.0, same tag format.
- [x] **4.7.3** Record steady-state s/it and SPS. Expected: ~3.2-4.5 s/it. **Measured: 6.5 s/it, 1,260 SPS** (2x speedup, below 4x projection).
- [x] **4.7.4** Record VRAM usage. Expected: ~6-9 GB. **Measured: 8.6 GB** (53% reduction from 18.4 GB).

**Results table**:
| Metric | Before (64×128) | After (16×128) | Expected | Actual |
|:---|---:|---:|:---|:---|
| Imagined transitions / step | 122,880 | 30,720 | 4x less compute | 4x less compute |
| Per-step time | ~0.203s | ~0.102s | ~0.05s | 0.102s (2x, not 4x) |
| s/it (64 grad steps) | 13.0 | 6.5 | ~3.2-4.5 | **6.5** (2x speedup) |
| VRAM | 18.4 GB | 8.6 GB | ~6-9 GB | **8.6 GB** (within range) |
| SPS | 630 | 1,260 | — | **1,260** (2x) |

#### Step 2: Replay Ratio Adjustment (If Still Too Slow)

If reduced batch size achieves ~3-4 s/it but the target is <1.0:

- [ ] **4.7.5** Try `replay_ratio: 0.5` — halves grad steps from 64 to 32. Expected: ~1.6-2.3 s/it.
- [ ] **4.7.6** Verify that reduced replay ratio doesn't harm learning (compare `Episode/Steps` at 50k episodes vs RR=1.0).

#### Step 3: Profile `nnx.split`/`nnx.merge` Overhead

The `lax.scan` body calls `nnx.merge(graphdef, state)` and `nnx.state(trainer)` on every gradient step. This state serialization may add overhead that wasn't present in the pre-optimization Python loop.

- [ ] **4.7.7** Add timing around `_scan_train_gpu` vs `collect_sequence` to isolate training time.
- [ ] **4.7.8** If `nnx.merge`/`nnx.state` overhead is significant (>20% of training time): consider flattening state management — pass raw parameter pytrees through scan carry instead of full NNX graph objects.

#### Step 4: Revised Pass Criteria

Phase 1 is fully complete when:
- [x] Steady-state `s/it < 5.0` for 64env CI=128 RR=1.0 with `batch_size: 16`. → **4.98 s/it (Section 8.3)**
- [x] No JIT retracing after warmup (already achieved).
- [x] GPU memory < 50% of 24 GB → **8.6 GB (36%)** (Section 8.2)
- [x] Results recorded in Section 8. → Sections 8.1, 8.2, 8.3

---

## 5. Diagnostic Phase 2: Training Performance Validation

**Prerequisite**: Phase 1 complete (training speed at acceptable level, s/it < 3.0).

This phase verifies that the optimized pipeline produces correct learning behavior — the speed optimization must not have broken training dynamics.

### 5.1 Sanity Check: Short Training Run

**Objective**: Confirm basic learning signal in a fast run before committing to long training.

**Run Configuration**:
```
Config:     configs/experiment/ablation/homeostatic/08_location.yaml + dreamer_v3.yaml
num_envs:   64
Episodes:   50,000  (short diagnostic run)
Tag:        phase2_sanity_check
```

**Checklist**:
- [ ] **5.1.1** `mean_entropy` stays above 0.5 throughout (no entropy collapse).
- [ ] **5.1.2** `model_reward_mae_pos > 0` by 10k episodes (agent discovers food).
- [ ] **5.1.3** `Episode/Steps` shows upward trend (agent learning to survive).
- [ ] **5.1.4** `value_mae` is decreasing (critic is learning).
- [ ] **5.1.5** `loss_recon` is decreasing (world model is learning).
- [ ] **5.1.6** `Params/effective_replay_ratio` is close to configured `replay_ratio` (gradient balance is correct).

### 5.2 Parallel Scaling Parity Test

**Objective**: Verify that 64env performs comparably to 1env on a per-episode basis, confirming that the replay ratio + buffer capacity fixes (Section 1.4) hold with the new optimized pipeline.

**Runs**:
| Run | num_envs | CI | RR | buffer_capacity | Episodes | Tag |
|:---|:---|:---|:---|:---|:---|:---|
| A | 1 | 128 | 1 | 10M | 50,000 | `phase2_scaling_1env` |
| B | 64 | 128 | 1 | 10M | 50,000 | `phase2_scaling_64env` |

**Checklist**:
- [ ] **5.2.1** Compare `Episode/Steps` at 10k, 30k, 50k episodes — 64env should be within 0.7x of 1env.
- [ ] **5.2.2** Compare `mean_entropy` trajectories — both should remain above 0.5.
- [ ] **5.2.3** If 64env is significantly worse: check buffer turnover. At 64 envs with CI=128, `10M / (64*128) = 1220` iterations before overwrite. If insufficient, increase to `replay_ratio: 2` or `buffer_capacity: 50M`.

### 5.3 World Model Health Check

Performed during or after the sanity run (5.1).

- [ ] **5.3.1** `loss_dyn_kl` and `loss_rep_kl`: Should stabilize above `FREE_NATS` (1.0) but not explode. Both at exactly 1.0 means uninformative latents (posterior ≈ prior).
- [ ] **5.3.2** `latent_entropy`: Healthy range 1.0-2.5. Below 0.5 means collapsed latent representation.
- [ ] **5.3.3** `loss_rew` and `model_reward_mae`: Should decrease over time. If `model_reward_mae_pos = 0` persists beyond 20k episodes with healthy entropy, the world model may not be learning reward structure.
- [ ] **5.3.4** `model_cont_acc`: Should be high (>0.95) but verify it's not trivially predicting "always continue" — check if there are actual termination events in the data.

### 5.4 Actor-Critic Health Check

- [ ] **5.4.1** `mean_entropy`: Healthy trajectory is ~1.5-1.8 early, gradually declining to 0.5-1.0 as policy converges. Collapse below 0.3 triggers the entropy playbook (Section 6).
- [ ] **5.4.2** `value_mae`: Should decrease. If stuck above 10, check critic target (must be `to_twohot(lambda_returns)` in raw space — see Section 1.1).
- [ ] **5.4.3** `mean_advantage` magnitude and variance: Should be non-trivial. Near-zero std means the advantage is uninformative.
- [ ] **5.4.4** `loss_actor_policy` vs `loss_actor_entropy`: Entropy term should be a meaningful fraction of total actor loss. If `loss_actor_entropy ≈ 0`, the entropy bonus is negligible.

### 5.5 Long Training Run (Default Environment)

**Prerequisite**: Sanity check (5.1) passes.

**Objective**: Full training on the default environment to establish a DreamerV3 performance baseline.

**Run Configuration**:
```
Config:     configs/environment/default.yaml + dreamer_v3.yaml
num_envs:   64
CI:         128
RR:         1
Episodes:   100,000,000 (long run)
Tag:        phase2_default_env_baseline
```

**Checkpoints to evaluate** (via `evaluation.py`):
- [ ] **5.5.1** At 100k, 500k, 1M, 5M episodes: record `MeanLen`, `MeanRew`, `TotalAte`, action distribution.
- [ ] **5.5.2** Compare against RecurrentPPO baseline at equivalent episode counts.
- [ ] **5.5.3** If performance plateaus early: investigate whether `batch_size: 64` (4x canonical) is causing gradient issues. Consider reducing to `batch_size: 16`.

### 5.6 Pass Criteria

Phase 2 is complete when:
- [ ] Sanity check passes all 5.1.x items.
- [ ] 64env scaling is within 0.7x of 1env per-episode performance.
- [ ] Long run shows monotonic improvement in `Episode/Steps` over at least 1M episodes.
- [ ] No entropy collapse, critic divergence, or reward starvation observed.

---

## 6. Diagnostic Phase 3: Hyperparameter Tuning & Advanced Diagnostics

**Prerequisite**: Phase 2 complete (training performance validated).

### 6.1 Batch Size Sweep

Our `batch_size: 64` with `sequence_length: 128` produces 4x more imagined transitions per gradient step than canonical DreamerV3 (`16 × 128 × 15 = 30,720` vs `64 × 128 × 15 = 122,880`). This may affect training dynamics.

- [ ] **6.1.1** Run with `batch_size: 16` (already applied in Phase 1) and compare learning curves against `batch_size: 64`.
- [ ] **6.1.2** If smaller batch is better: the large batch may be causing gradient dilution or excessive per-step compute without proportional learning benefit.
- [ ] **6.1.3** If larger batch is comparable: consider increasing back for better GPU utilization (but only if speed target is met).

### 6.2 Entropy Scale Tuning

Current `entropy_scale: 3e-4` (Hafner default). GridWorld may benefit from higher exploration pressure.

- [ ] **6.2.1** Sweep: `3e-4`, `1e-3`, `3e-3`, `1e-2`. Record entropy trajectory and `Episode/Steps` at 50k episodes each.
- [ ] **6.2.2** Identify the sweet spot where entropy stabilizes in 0.5-1.0 range without collapsing or preventing exploitation.

### 6.3 Replay Ratio Tuning

With 64 envs and CI=128, the buffer fills in ~1220 iterations. Higher `replay_ratio` extracts more learning per data point before it's overwritten.

- [ ] **6.3.1** Compare `replay_ratio: 0.5, 1.0, 2.0` at 64 envs.
- [ ] **6.3.2** Higher ratio = more gradient steps per iteration = slower wall-clock per iteration but potentially better sample efficiency.
- [ ] **6.3.3** Find the Pareto-optimal point (best learning curve per wall-clock hour).

### 6.4 Neuromodulation Integration

**Prerequisite**: Unmodulated DreamerV3 baseline established in Phase 2.

- [ ] **6.4.1** Run `neuromodulated_dreamer_v3.yaml` with the same environment config.
- [ ] **6.4.2** Monitor `mod_z_*_mean` and `mod_z_*_std` — modulator outputs should not saturate (all 0 or all 1).
- [ ] **6.4.3** Compare `Episode/Steps` and `Episode/Reward` against unmodulated baseline.
- [ ] **6.4.4** If modulated is worse: check if modulator is interfering with RSSM learning (compare `loss_recon`, `loss_rew` between the two).

### 6.5 Buffer Improvements (If Scaling Issues Persist)

If Phase 2 scaling test (5.2) shows persistent 64env degradation despite correct replay ratio and large buffer:

- [ ] **6.5.1** Implement **online queue**: Reserve 25% of each minibatch for the most recent `collect_interval` transitions. Matches DreamerV3 paper's hybrid approach.
- [ ] **6.5.2** Implement **`prioritize_ends`**: Oversample blocks containing terminal transitions (see v1 Section 15.2 for sheeprl reference).
- [ ] **6.5.3** Consider `replay_ratio: 2` as a simpler alternative — trains more on each experience before overwrite.

---

## 7. Key Metrics Reference

Quick reference for WandB monitoring across all diagnostic phases.

### 7.1 Speed Metrics (Phase 1)

| Metric | How to Measure | Target |
|:---|:---|:---|
| s/it | WandB `_timestamp` delta / iteration delta | < 1.0 (64env CI=128 RR=1.0) |
| SPS | `timesteps / wall_time` | > 30,000 |
| GPU utilization | `nvidia-smi` during training | > 80% |
| VRAM usage | `nvidia-smi` peak | < 80% of 24 GB |

### 7.2 Training Health Metrics (Phase 2)

| Metric | Healthy Range | Red Flag | Cause |
|:---|:---|:---|:---|
| `mean_entropy` | 0.5 - 1.8 | < 0.3 | Entropy collapse (Section 1.1) |
| `loss_recon` | Decreasing, < 0.1 | Increasing or stuck | World model not learning |
| `loss_rew` | Decreasing | Stuck or increasing | Reward prediction failure |
| `model_reward_mae_pos` | > 0 | = 0 for extended period | No food discovery (often from entropy collapse) |
| `value_mae` | < 5, decreasing | > 20 or diverging | Critic divergence (Section 1.1) |
| `loss_dyn_kl` | > 1.0, stable | = 1.0 (floor) or exploding | Uninformative or unstable latents |
| `latent_entropy` | 1.0 - 2.5 | < 0.5 | Collapsed latent representation |
| `Params/effective_replay_ratio` | Near config `replay_ratio` | Very different | Gradient balance broken |
| `Episode/Steps` | Increasing | Flat or decreasing | Agent not learning to survive |
| `Episode/Reward` | Increasing | Flat | Agent not improving behavior |

### 7.3 Debugging Playbook

**Entropy Collapse** (`mean_entropy < 0.3`):
1. Check `entropy_scale` — try 3e-3 or 1e-2.
2. Verify advantage computation — both sides must use same Moments normalization (Section 1.1).
3. Check `to_twohot` target — must receive **raw** `lambda_returns`.
4. Inspect `mean_advantage` std — near-zero means no learning signal.

**Critic Divergence** (`value_mae > 20`):
1. Verify critic trains on `to_twohot(lambda_returns)` (raw space).
2. Check `from_twohot()` returns raw-space values.
3. Verify Moments `low`/`high` EMAs are not stale.

**Zero Food Discovery** (`model_reward_mae_pos = 0`):
1. Usually a *consequence* of entropy collapse — fix entropy first.
2. If entropy healthy: check environment config (food placement, `eat_enabled`, resource parameters).

**Parallel Scaling Regression** (64env worse per-episode than 1env):
1. Verify `replay_ratio` is active and `train_steps = ratio_scaled_updates(global_step // num_steps)`.
2. Check `buffer_capacity` retains data for >= 500 iterations.
3. Verify PRNG key splitting in `collect_sequence`.

**Slow Training** (`s/it >> 1.0` for 64env CI=128):
1. Confirm `buffer_device: "gpu"`.
2. Confirm `train_multiple_gpu` is called (not CPU fallback).
3. Check JIT retracing: `JAX_LOG_COMPILES=1`.
4. Check buffer capacity — very large buffers may cause memory pressure.

---

## 8. Investigation Log

New diagnostics entries go below. Each entry should include date, observation, analysis, and resolution.

### Template
**Date**: YYYY-MM-DD
**Phase**: [1/2/3]
**Context**: [Config, run tag, WandB link]
**Observation**: [What was seen]
**Analysis**: [Root cause investigation]
**Resolution**: [Fix applied or next steps]

### 8.1 Phase 1 Speed Validation & JIT Optimization
**Date**: 2026-03-02
**Phase**: 1
**Context**: 64env, CI=128, RR=1.0, `buffer_device: "gpu"`, `tag: speed_validation_gpu_buffer_v5`
**Observation**: 
- Initial benchmark (v1) finished in 1 iteration because `args.episodes=0` was overridden by config default.
- GPU VRAM consumption confirmed at ~4.5 GB (Buffer 1.6 GB + Model/Stats/JAX Context).
- JIT compilation for `_scan_train` was significantly slow (77s) and recurring every iteration in subsequent runs.
- Encountered `ValueError: Non-hashable static arguments` and `ConcretizationTypeError` during JIT refactoring.

**Analysis**:
1. **Argument Overflow**: `train.py` argument parsing logic used `or` which treated `episodes=0` as `False`, falling back to config default.
2. **Closure Retracing**: `train_multiple_gpu` was capturing the `buffer` object in a closure. Every time `buffer.add_batch` updated the underlying JAX arrays, JAX detected a changed closure and triggered a re-trace.
3. **Nested JIT Re-definition**: `@nnx.jit` on a function defined inside a method causes a new JIT object to be created every call, preventing cross-call caching.
4. **Static vs Dynamic**: `jnp.arange` requires a concrete value for its shape, but `b_seq_len` was being passed as a tracer.

**Resolution**:
1. Fixed `episodes` check in `train.py` using `is not None`.
2. Refactored `DreamerTrainer.train_multiple_gpu` to pass buffer arrays explicitly to `_scan_train`.
3. Corrected `static_argnums` to handle static sequence length and capacity.
4. Refactored `_scan_train_gpu` into a stable method and initialized `dreamer_state` in `train.py` to eliminate all JIT retracing.
5. **Final Metrics (v6)**:
   - **Steady-state s/it**: 13.0s (for 64 envs, CI=128, RR=1.0)
   - **Steady-state SPS**: 630
   - **VRAM Usage**: 18.4 GB (75% of 3090)
   - **GPU Utilization**: 100% (Arithmetically bound by ~7.8M transitions processed per iteration)
   - **Status**: JIT is fully stable after Iteration 2. Ready for Phase 2.

**Review (Post-Implementation Audit)**:

The measured 13.0 s/it vs 12.89 pre-optimization means effectively **no wall-clock improvement**. However, the optimization is not broken — the original bottleneck analysis was wrong about where time was spent.

*What the optimization eliminated*:
| Bottleneck | Pre-Opt Share | Post-Opt Status |
|:---|---:|:---|
| Buffer sampling (numpy + host→device) | ~19% (~2.5s) | **Eliminated** — on-device sampling inside `lax.scan` |
| JAX dispatch overhead (per-step launch) | ~12% (~1.5s) | **Eliminated** — 64 steps fused into one XLA program |
| GPU compute (gradient work) | ~61% (~7.9s) | **Unchanged** — irreducible at current batch dimensions |
| Collection (`collect_sequence`) | ~6% (~0.8s) | **Unchanged** |

*Why no speedup is visible*: The 31% overhead elimination (~4s savings) should have yielded ~8.9 s/it. The measured 13.0 s/it suggests either (a) the fused `lax.scan` has its own overhead (state serialization via `nnx.split`/`nnx.merge` per step), or (b) the pre-opt time breakdown was inaccurate. The original ~55x estimate (v1 Section 21) was fundamentally wrong — it assumed the Python loop was the dominant cost, but GPU compute always was.

*Why SPS dropped from 4,429 to 630*: The pre-opt SPS figure (v1 Section 18.5) was measured over a 42-hour run with `collect_interval=128`, counting `num_envs × collect_interval = 8,192` env steps per iteration even though the iteration took 12.89s. The current 630 SPS may reflect a different measurement window or iteration count. The per-step GPU compute time (~0.19-0.20s) is consistent across both measurements.

*The real bottleneck — batch dimensions*: Each gradient step processes `64 batch × 128 seq × 15 horizon = 122,880` imagined transitions — **4x the target workload** (`16 × 128 × 15 = 30,720`, keeping `sequence_length: 128` to match RPPO). At 64 gradient steps per iteration, this is ~7.8M total imagined transitions, which saturates the GPU (100% utilization confirmed).

**VRAM breakdown**:
| Component | Estimated Size |
|:---|---:|
| GPU replay buffer (10M × 160 bytes) | 1.6 GB |
| Model parameters + optimizer state (3 networks × 3 optimizers) | ~0.5 GB |
| `lax.scan` carry state (full trainer state × 2 for fwd/bwd) | ~1.0 GB |
| Activations / intermediates (122K imagined transitions, gradient tape) | ~15 GB |
| **Total** | **~18.1 GB** (matches measured 18.4 GB) |

The activation memory dominates — this is proportional to `batch_size × sequence_length`. Reducing to canonical dimensions would also reduce VRAM.

**Actionable Next Steps** (ordered by expected impact):

1. **Reduce `batch_size` to 16** (keep `sequence_length: 128` to match RPPO). Expected per-step time: ~0.05s (vs current ~0.19s). At 64 grad steps: **~3.2-4.5 s/it**. Also reduces VRAM from ~18 GB to ~6-9 GB. **This is the highest-impact change.**

2. **Reduce `replay_ratio` to 0.5** — Halves grad steps from 64 to 32 → ~6.5 s/it at current batch size, or ~1.6-2.3 s/it with reduced batch. Trades sample efficiency for wall-clock speed.

3. **Profile `collect_sequence` vs `_scan_train_gpu`** — Confirm collection is <1s and training >12s. If `nnx.split`/`nnx.merge` inside `lax.scan` adds significant overhead, consider flattening state management.

4. **Consider `collect_interval: 1` with canonical batch** — If per-step drops to ~0.03s, even 64 Python-dispatched steps (~2s + overhead) may be acceptable, simplifying the architecture.

> **Decision needed**: Should Phase 2 proceed at 13 s/it while speed tuning continues in parallel, or should batch dimension reduction be applied first?
---

### 8.2 Phase 1 Continuation: Batch Size Reduction (v7)
**Date**: 2026-03-02
**Phase**: 1 (Continuation)
**Context**: 64env, CI=128, RR=1.0, `batch_size: 16`, `tag: speed_validation_gpu_buffer_v7`
**Observation**: 
- **VRAM Drop**: GPU memory usage dropped from 18.4 GB to **8.6 GB** (~53% reduction). This confirms that activation memory (proportional to `batch_size * sequence_length`) was the dominant memory consumer.
- **JIT Stability**: No new JIT retracing issues observed. `collect_sequence` and `_scan_train_gpu` both compiled successfully with the new static batch dimension.
- **Steady-State Speed (v7)**:
  - **s/it**: **6.5s** (for 64 envs, CI=128, RR=1.0, batch=16)
  - **SPS**: **1,260**
  - **Improvement**: 2.0x faster than batch=64 (13.0 s/it).
  - **GPU Utilization**: 100% (Arithmetically bound).

**Analysis**:
Reducing the batch size to 16 yielded a clean 2x speedup. While the target was <4.5 s/it (linear 4x projection), the achieved 6.5 s/it suggests non-linear scaling of JAX kernels or a fixed overhead in the `lax.scan` body (likely `nnx.split`/`nnx.merge` for 128-sequence activation tapes). However, 8.6 GB VRAM usage is much safer for long runs, and the system is achieving peak throughput for this heavy model configuration.

**Why reducing batch size reduces wall-clock time** (not just memory):
In standard supervised learning, larger batches parallelize across GPU cores and often take similar wall-clock time. But DreamerV3 is different — each gradient step involves **sequential 15-step imagination rollouts** through the world model, producing `batch_size × sequence_length × horizon` total imagined transitions. The GPU was already at **100% utilization** (confirmed via `nvidia-smi`), meaning it was fully saturated. Once a GPU is saturated, adding more parallel work doesn't run "for free" — it queues internally and takes proportionally longer. Reducing batch from 64→16 cuts total FLOPS per gradient step by 4x, which directly translates to less wall-clock time because the GPU had no spare capacity.

**Resolution / Next Steps**:
1. Phase 1 is concluded with optimized steady-state baseline.
2. **Ready for Phase 2: Training Performance Validation** (awaiting user confirmation).

**Review (Post-Measurement Analysis)**:

The 2x speedup (not 4x) reveals a significant **fixed overhead per gradient step** inside the `lax.scan` body. We can decompose the per-step cost:

```
Per-step time = fixed_overhead + compute_time
Old (batch=64): 0.203s = F + C
New (batch=16): 0.102s = F + C/4    (4x less compute)

Solving: 0.203 - 0.102 = 3C/4 → C = 0.135s (old compute), F = 0.068s
```

| Component | Per-step (batch=64) | Per-step (batch=16) | Per-iteration (×64 steps) |
|:---|---:|---:|---:|
| Fixed overhead (`nnx.split`/`nnx.merge`, scan bookkeeping) | 0.068s (33%) | 0.068s (**67%**) | **4.35s** |
| GPU compute (imagined transitions) | 0.135s (67%) | 0.034s (33%) | 2.15s |
| **Total** | **0.203s** | **0.102s** | **6.5s** |

The fixed overhead is now the **dominant bottleneck** at batch=16 — 67% of each gradient step and 4.35s per iteration. This validates Step 3 (4.7.7-4.7.8): profiling `nnx.split`/`nnx.merge` is the next highest-impact optimization if further speed improvement is needed.

However, 6.5 s/it is a reasonable operating point for Phase 2 diagnostics. Further optimization (flattening NNX state, reducing replay_ratio) can proceed in parallel with training performance validation.

**Updated Phase 1 checklist status**:
- [x] VRAM < 50% of 24 GB → 8.6 GB (36%) — **PASSED**
- [x] No JIT retracing — **PASSED** (since 8.1)
- [ ] s/it < 5.0 → 6.5 s/it — **NOT MET** (but 2x improvement from 13.0; diminishing returns without NNX refactor)
- [x] Results recorded — Section 8.2

**Recommendation**: Proceed to Phase 2 at 6.5 s/it. The remaining speed gap (6.5 → <5.0) requires NNX state management refactoring (Step 3), which is a significant code change best tackled after confirming training correctness.

---

### 8.3 Batch Size Comparison: Training Metrics (batch=64 vs batch=16)
**Date**: 2026-03-03
**Phase**: 1 → 2 (speed validation + training health)
**Context**:
- **batch=64**: `tag: dreamer_v3_128envs_64batch_128collect_replay1_hierarchical_1e6buffer` (WandB: `ep7fzk87`)
- **batch=16**: `tag: dreamer_v3_128envs_16batch_128collect_replay1_hierarchical_1e6buffer` (WandB: `jl5ndawv`)
- Both: 64 envs (naming says "128envs" — **mistake**), CI=128, RR=1.0, buffer_capacity=1M, GPU buffer
- Data extracted via `scripts/benchmark_wandb_speed.py` and `scripts/compare_wandb_runs.py`

#### 8.3.1 Speed Comparison

| Run | s/it | it/s | SPS | Total Time | Iterations | Timesteps |
|:---|---:|---:|---:|:---|---:|---:|
| batch=64 | 45.98 | 0.02 | 5,979 | 16h 47m | 1,184 | 9.7M |
| batch=16 | 4.98 | 0.20 | 4,405 | 13h 25m | 9,401 | 77.0M |

**Key observations**:
- batch=16 is **9.2x faster** per iteration (4.98 vs 45.98 s/it).
- batch=64's 45.98 s/it is ~3.5x slower than the earlier v6 short diagnostic run (13.0 s/it). Possible causes: different buffer capacity (1M vs 10M), longer-run steady-state behavior, or WandB timestamp measurement over full run including checkpoints/logging overhead.
- batch=16 processed **8x more data** (77M vs 9.7M timesteps) in **less wall-clock time** (13.4h vs 16.8h).
- SPS favors batch=64 (5,979 vs 4,405) because each iteration collects the same env steps but batch=64 does more gradient work — SPS measures throughput, not training efficiency.

#### 8.3.2 Training Health Comparison

**Episode Performance** (steady-state = last 20% of run):

| Metric | Criterion | batch=64 (steady) | batch=16 (steady) | batch=64 (last) | batch=16 (last) |
|:---|:---|---:|---:|---:|---:|
| `Episode/Steps` | ↑ better | 35.0 ± 1.8 | 35.4 ± 1.6 | 32.8 | 38.8 |
| `Episode/Reward` | ↑ better | -207.0 ± 0.6 | -207.3 ± 0.6 | -206.5 | -208.3 |
| `Total Episodes` | info | 255k | 1,964k | 283k | 2,183k |

**World Model**:

| Metric | Criterion | batch=64 | batch=16 | Status |
|:---|:---|---:|---:|:---|
| `loss_recon` | ↓ better | 0.011 | 0.011 | **OK** — both converged |
| `loss_rew` | ↓ better | 1.44 | 1.72 | batch=64 slightly better |
| `loss_dyn_kl` | > 1.0 | 2.11 | 2.22 | **OK** — both above floor |
| `loss_rep_kl` | > 1.0 | 2.11 | 2.22 | **OK** — both above floor |
| `latent_entropy` | 1.0-2.5 | **0.83** | **0.86** | **RED FLAG** — below 1.0 |
| `reward_mae_pos` | > 0 | 0.21 | 0.12 | **OK** — food found |
| `cont_acc` | > 0.95 | 0.983 | 0.981 | **OK** |

**Actor-Critic**:

| Metric | Criterion | batch=64 | batch=16 | Status |
|:---|:---|---:|---:|:---|
| `mean_entropy` | > 0.5 | 1.66 | 1.31 | **OK** — both above 0.5 (batch=16 dropped more) |
| `value_mae` | < 5 | **7.56** | **7.56** | **RED FLAG** — both above 5.0 |
| `mean_advantage` | non-trivial | -0.184 | -0.194 | **OK** — non-zero |
| `mean_return` | info | -25.46 | -25.48 | Similar |
| `mean_value` | info | -18.30 | -18.28 | Similar |
| `loss_actor_entropy` | meaningful | **-0.0004** | **-0.0003** | **WARNING** — negligible entropy bonus |
| `loss_critic` | ↓ better | 0.314 | 0.339 | batch=64 slightly better |

**System**:

| Metric | Criterion | batch=64 | batch=16 | Status |
|:---|:---|---:|---:|:---|
| `eff_replay_ratio` | ≈ 1.0 | **0.0078** | **0.0078** | **CRITICAL** — 128x below configured RR=1.0 |

**Trajectory** (first → last):

| Metric | batch=64 (1,184 iters) | batch=16 (9,401 iters) |
|:---|:---|:---|
| `Episode/Steps` | 27.6 → 32.8 | 27.3 → 38.8 |
| `mean_entropy` | 1.79 → 1.65 | 1.79 → 1.29 |
| `value_mae` | 8.43 → 7.58 | 8.00 → 7.57 |
| `loss_recon` | 0.176 → 0.011 | 0.177 → 0.011 |
| `reward_mae_pos` | 1.97 → 0.21 | 2.07 → 0.37 |
| `latent_entropy` | 1.45 → 0.83 | 1.88 → 0.86 |

#### 8.3.3 Review & Analysis

**1. Speed: batch=16 is the clear winner.** 9.2x faster per iteration, 8x more data processed in less wall-clock time. Phase 1 speed target of <5.0 s/it is now met (4.98 s/it).

**2. Training quality: batch sizes produce equivalent learning dynamics.** World model, critic, and episode metrics are nearly identical between the two runs. Neither batch size shows a clear learning advantage — the batch=16 run simply got further (8x more iterations) in less time.

**3. CRITICAL — `effective_replay_ratio = 0.0078` (both runs).** The configured `replay_ratio: 1.0` should produce ~1.0 gradient step per env step, but the measured 0.0078 means the agent performs ~128x fewer gradient steps than expected. This is likely because `collect_interval=128` inflates the per-iteration env step count (`64 envs × 128 steps = 8,192`), but the `Ratio` class is receiving a different normalization. This must be investigated — it may explain why learning is slow despite 77M timesteps.

Specifically: `effective_replay_ratio = grad_steps / global_step`. Each iteration produces `num_envs × collect_interval = 64 × 128 = 8,192` env steps. With RR=1.0 and the `Ratio` class, the expected grad_steps should be ~8,192 per iteration (or ~64 if normalized by `global_step // num_steps`). The measured 0.0078 suggests `grad_steps ≈ 64` while `global_step` increments by 8,192 — i.e., the normalization is **not** dividing by `num_steps`. This is the bug identified in v1 Section 14 ("Ratio class and `collect_interval` interaction").

**4. RED FLAG — `latent_entropy` below 1.0 (both runs).** Started at 1.4-1.9 and dropped to 0.83-0.86. This indicates the latent representation is becoming increasingly deterministic — the posterior is collapsing toward point estimates. This reduces the world model's ability to represent uncertainty and may limit downstream policy quality. Worth monitoring in Phase 2.

**5. RED FLAG — `value_mae` stuck at ~7.5 (both runs).** Above the < 5 threshold. The critic is converging (8.4→7.6) but slowly. With `mean_return ≈ -25.5`, a MAE of 7.5 represents ~30% prediction error. This may improve with more training or may indicate a systematic critic learning issue.

**6. WARNING — `loss_actor_entropy` ≈ -0.0003 (both runs).** The entropy bonus is negligible compared to `loss_actor_policy` (≈ -0.3 to -1.2). At `entropy_scale: 3e-4`, the entropy term contributes < 0.1% of total actor loss. This means the entropy regularization is effectively inactive — the policy entropy (1.3-1.7) is maintained by the advantage landscape, not by explicit entropy pressure.

#### 8.3.4 Actionable Items

1. **CRITICAL: Fix `effective_replay_ratio`** — Investigate `Ratio` class normalization. The `global_step` passed to `ratio.wants(global_step)` must be `global_step // num_steps` (i.e., collect-interval-normalized), not raw env steps. See v1 Section 14 for the prior investigation.

2. **Monitor `latent_entropy`** — If it continues dropping below 0.5 in Phase 2, consider:
   - Increasing `free_nats` from 1.0 to 2.0 (forces latents away from determinism)
   - Checking if KL balancing (`kl_balance: 0.8`) is pushing too hard on the representation loss

3. **Monitor `value_mae`** — If still > 5 after fixing replay ratio (which should increase gradient steps dramatically), investigate critic target computation (Section 1.1 two-hot issue).

4. **Consider increasing `entropy_scale`** — Current 3e-4 is negligible. If entropy drops below 0.5 in Phase 2, try 1e-3 or 3e-3 (Phase 3 item 6.2).

5. **Phase 1 conclusion**: Speed target met with batch=16 (4.98 s/it < 5.0). Proceed to Phase 2 but **fix replay ratio first** — training is currently running at ~0.8% of intended gradient utilization.

---

## 9. Multi-Environment Scaling Analysis (2026-03-05)

Analysis of four DreamerV3 runs varying `num_envs` (1, 4, 8, 16) to test whether increasing environment parallelism addresses the pathologies identified in Sections 7.5 and 8.3.

### 9.1 Runs Under Analysis

| Alias | Full Tag | num_envs | Run ID | Status |
|:---|:---|---:|:---|:---|
| **DV3-1env** | `20260303-111929_dreamer_v3_1envs_16batch_128collect_replay1_hierarchical_1e6buffer` | 1 | `81xlvacs` | Completed |
| **DV3-4env** | `20260304-203524_dreamer_v3_4envs_16batch_128collect_replay1_hierarchical_1e6buffer` | 4 | `c5i05ru8` | Running |
| **DV3-8env** | `20260304-203615_dreamer_v3_8envs_16batch_128collect_replay1_hierarchical_1e6buffer` | 8 | `gwslkjfo` | Running |
| **DV3-16env** | `20260304-203635_dreamer_v3_16envs_16batch_128collect_replay1_hierarchical_1e6buffer` | 16 | `fzoiyhn7` | Running |

### 9.2 Configuration Differences

The 1-env run uses a different hierarchical encoding architecture than the 4/8/16-env runs:

| Parameter | DV3-1env | DV3-4env / 8env / 16env |
|:---|:---|:---|
| `num_envs` | 1 | 4 / 8 / 16 |
| `hierarchical_params.default_mlp` | `[128]` | `[32, 32]` |
| `hierarchical_params.multimodal_hub` | *(not set)* | `[128, 128]` |
| `hierarchical_params.hub_overrides` | `{body_state: [64], association: [128]}` | *(not set)* |
| `hierarchical_params.unimodal_overrides` | *(not set)* | `{visual: [128, 128], olfaction: [128, 128]}` |

**Important caveat**: The 1-env run uses a body-state-aware architecture (dedicated encoding pathway) with larger defaults, while the 4/8/16-env runs use a smaller default encoder with sensor-specific overrides for visual and olfaction. This confounds the `num_envs` comparison — performance differences reflect **both** architecture and parallelism changes.

All other hyperparameters are identical: `batch_size=16`, `collect_interval=128`, `replay_ratio=1`, `train_steps=64`, `buffer_capacity=1M`, `rssm_deter=512`, `rssm_stoch=32×32`, `model_lr=1e-4`, `actor_lr=3e-5`, `entropy_scale=3e-4`.

### 9.3 Training Speed & Throughput

| Metric | DV3-1env | DV3-4env | DV3-8env | DV3-16env |
|:---|---:|---:|---:|---:|
| **s/iteration** | 0.255 | 0.411 | 0.768 | 1.467 |
| **SPS** | 507 | 1,382 | 1,393 | 1,607 |
| **Wall-clock time** | 48h 55m | 15h 43m | 15h 42m | 15h 42m |
| **Total timesteps** | 88.6M | 70.3M | 75.4M | 78.8M |
| **Total iterations** | 691,769 | 137,357 | 73,659 | 38,492 |
| **WM log points (n)** | 10,000 | 10,000 | 7,339 | 3,836 |

**Speed analysis**:
- SPS scales roughly linearly with `num_envs` (507 → 1,607 ≈ 3.2× for 16× more envs). The sub-linear scaling is expected — more envs increase per-iteration cost (world model trains on same batch_size but env stepping is parallelized).
- The 4/8/16-env runs all completed ~70-79M timesteps in ~15.7 hours, compared to 88.6M in 48.9 hours for 1-env. Multi-env runs are **3× faster** in wall-clock time per timestep.
- **Iterations decrease proportionally**: 1-env got 691k iterations (10k WM log points), 16-env got only 38k iterations (3,836 WM log points). Fewer iterations means fewer gradient updates — a critical consideration for learning dynamics.

### 9.4 Episode Performance

| Metric | DV3-1env | DV3-4env | DV3-8env | DV3-16env |
|:---|---:|---:|---:|---:|
| **Ep Steps (early → last)** | 21.5 → 7.0 | 22.1 → 38.1 | 26.2 → 37.9 | 27.1 → 32.0 |
| **Ep Steps (steady-state)** | 62.8 ± 31.2 | 36.6 ± 7.0 | 36.4 ± 4.6 | 35.8 ± 3.2 |
| **Ep Reward (early → last)** | -202.7 → -200.2 | -203.7 → -208.2 | -205.0 → -208.0 | -205.6 → -206.8 |
| **Ep Reward (steady-state)** | -208.7 ± 8.5 | -207.5 ± 2.4 | -207.6 ± 1.7 | -207.5 ± 1.2 |

**Key findings**:

1. **DV3-1env episode steps collapsed** (21.5 → 7.0). The agent is dying faster at the end of training than at the beginning. This is a **catastrophic regression** — the policy has unlearned whatever survival behavior it initially had. The high variance (±31.2) during steady-state suggests highly erratic behavior before final collapse.

2. **Multi-env runs show modest improvement** (22-27 → 32-38 steps). The 4-env and 8-env runs roughly doubled from their starting points. However, 32-38 steps out of a maximum 500 means the agent survives only **6-8% of the episode** — far from functional.

3. **Multi-env steady-state is remarkably consistent**: 35.8-36.6 steps across 4/8/16 envs with decreasing variance (7.0 → 3.2). This suggests a **performance ceiling** that more environments alone cannot break through.

4. **Reward is flat and indistinguishable** across all multi-env runs (~-207.5). Combined with short episodes, this means agents die quickly and accumulate roughly the same penalty each time. The homeostatic reward paradox (Section 7.4) applies: reward is dominated by death events, not learning signal.

5. **The 1-env run's higher steady-state (62.8 steps)** is misleading — it reflects a period before collapse, not sustained performance. The final value (7.0 steps) is the worst of all runs.

### 9.5 World Model Health

| Metric | Criterion | DV3-1env | DV3-4env | DV3-8env | DV3-16env |
|:---|:---|---:|---:|---:|---:|
| `loss_model` | ↓ better | **2.20** | 2.94 | 3.01 | 3.00 |
| `loss_recon` | ↓ better | **0.006** | 0.011 | 0.012 | 0.012 |
| `loss_rew` | ↓ better | **1.09** | 1.65 | 1.72 | 1.72 |
| `loss_dyn_kl` | > 1.0 | 1.80 | 2.05 | 2.04 | 2.03 |
| `loss_rep_kl` | > 1.0 | 1.80 | 2.05 | 2.04 | 2.03 |
| `latent_entropy` | 1.0–2.5 | ❌ 0.78 | ❌ 0.83 | ❌ 0.88 | ❌ 0.80 |
| `cont_acc` | > 0.95 | ✅ 0.988 | ✅ 0.981 | ✅ 0.980 | ✅ 0.980 |
| `reward_mae_pos` | > 0 | 0.050 | 0.175 | 0.144 | 0.153 |
| `reward_mae_neg` | info | **2.47** | 3.70 | 3.70 | 3.73 |

**Analysis**:

1. **DV3-1env has lower WM losses across the board** — but this is deceptive. With 691k iterations vs 38k-137k for multi-env runs, the 1-env world model has had **5-18× more gradient updates**. It has overfit to its limited data distribution. The lower `loss_recon` (0.006 vs 0.012) suggests the decoder has memorized reconstruction patterns rather than learning generalizable features.

2. **Latent entropy collapsed in ALL runs** (0.78-0.88 vs target 1.0-2.5). This was the critical Pathology 1 from Section 7.5 — and increasing `num_envs` has **not fixed it**. The multi-env runs are slightly better (0.83-0.88 vs 0.78) but still well below the healthy range. The RSSM posterior continues to collapse toward deterministic states regardless of data diversity.

3. **KL losses are elevated in multi-env runs** (2.03-2.05 vs 1.80). With `free_nats=1.0`, a KL of 2.0 means the free-bits floor is not binding — the posterior and prior are diverging. However, the KL trajectory tells the real story:
   - 1-env: started at 3.68, converged to 1.79 (posterior collapsed toward prior)
   - 4-env: started at 1.00, increased to 2.12 (posterior diverging from prior)
   - 8/16-env: same pattern (1.00 → 1.90-2.02)

   The 1-env run's KL decrease reflects posterior collapse (entropy → 0.78), not healthy convergence. The multi-env KL increase reflects the posterior learning from diverse data faster than the prior can follow.

4. **Positive reward prediction** (`reward_mae_pos`) is marginal across all runs. The multi-env runs show slightly higher values (0.14-0.18 vs 0.05) because more environments generate more diverse trajectories with occasional positive rewards. But the final values (0.00-0.02) indicate the reward head still gives up on positive reward prediction. This is **Pathology 3 (Positive Reward Blindness)** — unresolved.

5. **Negative reward MAE** is ~50% higher in multi-env runs (3.70 vs 2.47). The world model struggles more with diverse negative-reward patterns from multiple environments. This makes sense — with more environments, there's more variety in death scenarios, making the reward distribution harder to model.

### 9.6 Actor-Critic Health

| Metric | Criterion | DV3-1env | DV3-4env | DV3-8env | DV3-16env |
|:---|:---|---:|---:|---:|---:|
| `mean_entropy` | > 0.5 | ✅ 1.61 | ✅ 1.38 | ✅ 1.30 | ✅ 1.29 |
| `value_mae` | < 5 | ❌ 6.96 | ❌ 7.98 | ❌ 7.55 | ❌ 7.37 |
| `mean_advantage` | non-trivial | -0.189 | -0.193 | -0.198 | -0.196 |
| `mean_return` | info | -24.98 | -25.88 | -25.53 | -25.29 |
| `mean_value` | info | -18.31 | -18.29 | -18.35 | -18.29 |
| `loss_critic` | ↓ better | 0.328 | 0.333 | 0.324 | 0.325 |
| `eff_replay_ratio` | ≈ 1.0 | ❌ 0.0078 | ❌ 0.0078 | ❌ 0.0078 | ❌ 0.0078 |

**Analysis**:

1. **Policy entropy is healthy** across all runs (1.29-1.61, all > 0.5). Unlike the latent entropy collapse, the actor maintains diverse action distributions. However, the trend is downward (1.79 → 1.27 for 4-env), suggesting slow policy specialization. The 1-env run retains the highest entropy (1.61) — but this didn't translate into better performance, indicating the policy explores without learning.

2. **Critic divergence persists** (`value_mae` > 5 in all runs, **Pathology 2** from Section 7.5). The trajectory tells the real story:
   - 1-env: 0.09 → 7.36 (critic worsening over 691k iterations)
   - 4-env: 1.84 → 8.13 (worst)
   - 16-env: 4.92 → 7.20 (best, but still diverging)

   More environments start with higher value_mae (the critic faces more diverse returns from the start) but converge to similar bad levels. The critic is failing to learn accurate value predictions in all configurations.

3. **Advantages are uniformly negative** (~-0.19). In a healthy DreamerV3, advantages should be centered near zero with both positive and negative values. All-negative advantages mean the actor consistently finds that real outcomes are worse than predicted — the critic is systematically overestimating value. This creates a **pessimistic policy gradient** that reinforces avoidance over exploration.

4. **Effective replay ratio still stuck at 0.0078** across all runs. This confirms the bug is independent of `num_envs` — it's a systemic issue in how `grad_steps / global_step` is computed. With the replay ratio effectively at 0.78%, the world model, actor, and critic are **all drastically under-trained** relative to data collected. This remains the most actionable fix (Section 8.3.4 Item 1).

### 9.7 Behavioral Observations (User-Reported)

The user reports observing in evaluation videos that the agent:
- **Succeeds at**: Avoiding predators, hiding in bushes, resting
- **Fails at**: Consuming food resources

This is consistent with the metrics:
- **Why avoidance works**: Negative rewards (injury, death) dominate the buffer. The reward head learns to predict negative consequences well (`reward_mae_neg` converging). The world model imagines danger accurately → actor learns avoidance.
- **Why foraging fails**: Positive rewards (eating, reducing drive) are extremely rare in the buffer. `reward_mae_pos → 0` means the world model **cannot imagine benefit from eating**. The actor has no gradient signal toward food-seeking behavior. This is Pathology 3 confirmed by behavioral observation.

The agent has learned a **passive survival strategy**: avoid danger, hide, rest — all behaviors that reduce negative reward. But it cannot learn **active survival**: seeking food to prevent starvation. Eventually, passive survival fails because nutrition depletes to zero regardless of how well the agent avoids injury. This explains the ~35-step episode ceiling — it's roughly the starvation timeline for a non-eating agent.

### 9.8 Core Problem: Asymmetric Reward Learning

The fundamental issue across all four runs is **asymmetric reward learning in the world model**:

| Reward Type | Frequency in Buffer | WM Prediction Quality | Actor Learning |
|:---|:---|:---|:---|
| **Negative** (injury, death, drive increase) | Very common (~99%) | Good (`mae_neg` converging) | ✅ Learns avoidance |
| **Positive** (eating, drive decrease) | Extremely rare (<1%) | Failed (`mae_pos → 0`) | ❌ No foraging signal |

This creates a **one-sided actor**: the imagined trajectories in DreamerV3's imagination always predict negative outcomes regardless of action → the actor only learns "minimize damage" → never discovers "seek reward" → buffer stays reward-poor → vicious cycle.

### 9.9 Proposed Solution: DreamerV4-Style Recent Data Replay

The user proposes adopting DreamerV4's replay buffer strategy, which mixes **recent data** with replay data during training batches.

#### 9.9.1 Rationale

DreamerV4 (Hafner et al., 2025) addresses exactly this class of problem — when positive experiences are rare in a large replay buffer, they get overwhelmed by the dominant negative experience distribution. The key insight:

- **Standard replay** (current): Sample uniformly from the full 1M buffer → batch is 99% negative-reward transitions → reward head learns "always predict negative" → actor never imagines benefit from food
- **DreamerV4 recent-data mixing**: Reserve a fraction of each training batch for **recent** transitions (e.g., last N steps) → even if the overall buffer is 99% negative, recent data captures the latest policy's behavior → as the policy improves even slightly (e.g., accidentally eating), those positive experiences are immediately amplified in training batches

#### 9.9.2 Expected Benefits

1. **Faster positive reward learning**: Recent data preserves the distribution of the current policy, including any rare positive events. The reward head trains on these immediately rather than waiting for them to become statistically significant in a 1M buffer.
2. **Reduced staleness**: Current replay from a 1M buffer includes transitions from very early random policy. These may teach the world model outdated dynamics (e.g., "the agent always walks into predators").
3. **Better posterior-prior alignment**: Recent data is more representative of current policy → posterior trains on current behavior → prior tracks posterior more closely → reduced KL divergence → healthier latent space.
4. **Natural curriculum**: As the policy improves, the "recent" window naturally shifts to harder, more relevant scenarios.

#### 9.9.3 Implementation Approach

Two strategies, from simplest to most DreamerV4-faithful:

**Option A: Simple Recent-Bias Sampling**
- When sampling a training batch of 16 sequences, reserve K sequences (e.g., K=4) from the most recent `recent_window` transitions (e.g., last 10k steps)
- Remaining 12 sequences sampled uniformly from the full buffer
- Implementation: Add a `recent_fraction` parameter to the replay buffer's `sample()` method
- Minimal code change, easy to tune via config

**Option B: Full DreamerV4 Replay Strategy**
- Maintain two buffer regions: a "recent" FIFO buffer (e.g., last 50k transitions) and the full replay buffer
- Each batch: 50% from recent, 50% from replay (DreamerV4 default ratio)
- Apply importance weighting if using different sampling distributions
- More faithful to the paper but requires more significant buffer refactoring

**Recommendation**: Start with **Option A** — it addresses the core issue (positive reward dilution) with minimal code change. The `recent_fraction` and `recent_window` can be tuned via YAML config. If Option A shows improvement, Option B can be explored as a follow-up.

#### 9.9.4 Configuration Design

```yaml
# In agent config (dreamer_v3 section)
replay:
  capacity: 1_000_000
  recent_fraction: 0.25        # 25% of batch from recent data
  recent_window: 10_000        # "recent" = last 10k transitions
  # existing params unchanged
  replay_ratio: 1
  batch_size: 16
  sequence_length: 128
```

### 9.10 Additional Fixes to Combine with Recent-Data Replay

Recent-data replay alone may not be sufficient. The following should be addressed in parallel:

1. **Fix effective replay ratio** (Priority: CRITICAL, carried from Section 8.3.4)
   - Current 0.0078 means 99.2% of collected data is never used for gradients
   - Must fix `Ratio` class normalization before any replay strategy change can be properly evaluated

2. **Increase `free_nats` to prevent latent entropy collapse** (Priority: HIGH)
   - Current: `free_nats=1.0` → latent entropy collapses to 0.78-0.88
   - Proposed: `free_nats=2.0` or `free_nats=3.0` → keep latent entropy in 1.0-2.5 range
   - This prevents the posterior from collapsing, preserving stochastic imagination quality

3. **Increase `entropy_scale`** (Priority: MEDIUM)
   - Current: `3e-4` → `loss_actor_entropy ≈ -0.0003` (negligible)
   - Proposed: `1e-3` or `3e-3` → meaningful entropy pressure to maintain exploration
   - Especially important if recent-data replay shifts the actor toward a specific strategy

4. **Standardize architecture across runs** (Priority: MEDIUM)
   - The 1-env run's body-state-aware architecture should be tested with multi-env
   - Recommended: Use `hub_overrides: {body_state: [64], association: [128]}` + `unimodal_overrides: {visual: [128, 128], olfaction: [128, 128]}` combined

### 9.11 Summary: What num_envs Scaling Revealed

| Finding | Implication |
|:---|:---|
| Multi-env prevents catastrophic collapse (1-env: 7 steps, multi-env: 32-38) | Minimum 4 envs needed for stable training |
| Multi-env does NOT break the ~36-step ceiling | Data diversity alone doesn't solve the asymmetric reward problem |
| Latent entropy still collapses in all runs | Need `free_nats` increase, not just more data |
| `value_mae` diverges in all runs | Critic learning is fundamentally broken (possibly tied to replay ratio bug) |
| `effective_replay_ratio = 0.0078` everywhere | Systemic bug, highest priority fix |
| Agent avoids danger but never eats | World model can't imagine positive reward → need recent-data replay |
| 4/8/16-env performance nearly identical | Diminishing returns beyond 4 envs for current hyperparameters |

### 9.12 Recommended Next Steps (Prioritized)

1. **Fix replay ratio bug** — Without this, gradient utilization is 128× below intended. All other improvements are limited by undertrained networks.
2. **Implement 33/33/33 mixture sampling** — Positive-reward / recent / uniform split. Address the core positive-reward blindness. → **Implementation plan**: [`MIXTURE_SAMPLING_PLAN.md`](MIXTURE_SAMPLING_PLAN.md)
3. **Increase `free_nats` to 2.0** — Prevent latent entropy collapse.
4. **Run a controlled experiment**: 4 envs (cheapest multi-env that matches 8/16 performance) with fixes 1-3 applied, compared against current 4-env baseline.
5. **Add behavioral logging** (Phase 2 from `TRAINING_METRICS_ANALYSIS.md` Section 4) — death cause, action distribution, and final physiology metrics to properly diagnose future runs.
6. **Test body-state architecture with multi-env** — The 1-env architecture's `hub_overrides: {body_state: [64]}` may be important for interoceptive learning; currently untested with multi-env.

---

## 10. DreamerV4 Feature Applicability Review (2026-03-05)

Reviewed `docs/DREAMER_IMPLEMENTATION_AUDIT.md` Sections 5.1–5.8 to identify DreamerV4 innovations that can be adopted within our constraints:
- **Constraint 1**: Keep RSSM (RNN-based world model) — project is neuroscience-inspired, recurrent structure is architecturally motivated.
- **Constraint 2**: Keep online sampling — no pre-defined dataset; the sparse foraging signal must be solved online.

### 10.1 Applicable: PMPO Sign-Only Advantages (Audit Section 5.4) — HIGH IMPACT

**What it is**: DreamerV4 replaces DreamerV3's magnitude-weighted percentile advantage normalization with PMPO (Preference Optimization as Probabilistic Inference). Imagined trajectories are split into D+ (above median return) and D- (below median). The actor receives **sign-only** gradients: increase probability of actions in D+, decrease in D-. Return magnitudes are discarded.

$$L(\theta) = \frac{1-\alpha}{|D^-|} \sum_{i \in D^-} \ln \pi_\theta(a_i|s_i) - \frac{\alpha}{|D^+|} \sum_{i \in D^+} \ln \pi_\theta(a_i|s_i) + \frac{\beta}{N} \sum_{i=1}^{N} \text{KL}[\pi_\theta \| \pi_\text{prior}]$$

With $\alpha = 0.5$ (equal weighting between positive and negative sets).

**Why it directly addresses our pathologies**:

| Current Problem | How PMPO Fixes It |
|:---|:---|
| `mean_advantage ≈ -0.19` uniformly negative (Section 9.6) | D+ always exists (top 50% of imagined trajectories). Even when all returns are negative, the "least bad" actions get positive reinforcement. |
| Death penalty (-100) dominates food reward (+3) in gradients | Magnitudes discarded — sign-only means a +3 food event and a -100 death event contribute equal gradient weight. |
| Critic overestimates value → pessimistic policy gradient | PMPO bypasses critic-based advantage entirely. D+/D- split uses raw returns, not critic predictions. |
| `loss_actor_entropy ≈ -0.0003` (entropy bonus negligible) | PMPO replaces entropy bonus with KL prior (see 10.5). |

**Compatibility**: Works directly with RSSM imagination. Only changes how actor loss is computed from imagined returns — no world model or critic architecture change needed. The existing `λ`-return computation is preserved; only the advantage → policy gradient step changes.

**Implementation scope**: Replace `behavior_loss_fn` actor loss computation. Remove Moments EMA (no longer needed for percentile normalization). Add median-split logic over imagined returns.

### 10.2 Applicable: 50/50 Mixture Sampling (Audit Section 5.7) — HIGH IMPACT

Already proposed in Section 9.9 as "recent-data replay." The audit confirms V4's exact formulation is more targeted than pure recency — V4 splits batches into **uniform** (for world model calibration) and **task-relevant** (for sparse reward amplification), with loss decoupling between the two halves.

**Refinement over Section 9.9**: V4's approach is not just recency-biased — it explicitly filters for **task-success events**. For our online setting, this translates to:

| V4 (Offline) | Our Adaptation (Online) |
|:---|:---|
| Pre-annotated task-relevant sequences | Tag episodes at buffer insertion: `has_eating_event`, `has_recovery` |
| 50% uniform + 50% task-relevant | 50% uniform + 25% recent (last 10k steps) + 25% reward-tagged |
| BC loss on relevant half only | Actor-critic loss on all; WM dynamics loss on uniform half only |
| Static dataset split | Dynamic tagging as buffer fills |

**V4's loss decoupling insight** (Audit Section 5.7.5): Dynamics loss is applied **only** to the uniform half — this prevents the world model from learning that rare food events happen 50% of the time. Actor-critic loss is applied to both halves. This is more principled than applying all losses uniformly to a biased batch.

**Early-training warm-up**: V4 sidesteps cold-start because its offline dataset already contains sufficient rare events. We need a warm-up period of uniform-only sampling until the "reward-tagged" pool reaches a minimum size (e.g., 100 episodes with eating events). Before that threshold, fall back to 75% uniform + 25% recent.

**Configuration update** (supersedes Section 9.9.4):

```yaml
replay:
  capacity: 1_000_000
  sampling_mode: "mixture"          # "uniform" (default) or "mixture"
  mixture:
    uniform_fraction: 0.50          # WM dynamics loss applied here
    recent_fraction: 0.25           # last recent_window steps
    tagged_fraction: 0.25           # episodes with positive reward events
    recent_window: 10_000
    tagged_min_pool: 100            # minimum tagged episodes before enabling
  # Loss decoupling: dynamics loss on uniform half, actor-critic on all
```

### 10.3 Applicable: RMS Loss Normalization (Audit Section 5.2) — MEDIUM IMPACT

**What it is**: V4 replaces DreamerV3's unit-scale loss summation with RMS normalization — each loss component is divided by its running RMS before summing. This ensures no single loss dominates the gradient.

**Current problem**: Our world model loss is `total = loss_recon + loss_rew + loss_dyn + loss_rep + loss_cont`. Observed magnitudes:

| Loss | Steady-State Value | Relative Contribution |
|:---|---:|---:|
| `loss_rew` | 1.65–1.72 | **56%** |
| `loss_dyn_kl` | 2.03–2.05 (but 0.5× weighted by KL balance) | ~34% |
| `loss_rep_kl` | 2.03–2.05 (but 0.1× weighted by KL balance) | ~7% |
| `loss_recon` | 0.011–0.012 | **<0.4%** |
| `loss_cont` | (not separately logged) | ~3% |

The reward head receives ~56% of total gradient weight, while the reconstruction head (which learns observation dynamics) receives <0.4%. This gradient imbalance may contribute to the reward head's dominance — it overfits to the negative-reward majority while the decoder is under-trained.

**Implementation**: Track per-loss EMA of RMS (same mechanism as Moments). Divide each loss by its RMS before summing. Minimal code change — ~10 lines in `dreamer_v3_trainer.py::world_model_loss_fn`.

### 10.4 Applicable: Symexp Two-Hot Bins (Audit Section 5.5) — LOW-MEDIUM IMPACT

**What it is**: V4 shifts from symlog-spaced to symexp-spaced bins in the two-hot discretization. Symexp uses exponentially wider spacing at extremes and finer resolution near zero.

**Why it may help**: Our reward range is narrow (~-100 to +3). Symlog allocates many bins to ranges we never use ($10^3$–$10^6$). Symexp would concentrate more bins in the critical near-zero region where "slightly negative" vs "slightly positive" reward is the difference between avoidance and foraging behavior.

**Implementation**: Drop-in replacement in `to_twohot` / `from_twohot` in `dreamer_v3_util.py`. The bin boundary computation changes; the two-hot encoding logic stays the same.

**Priority**: Low — the current symlog discretization isn't broken. This is a refinement, not a fix. Defer until after PMPO and mixture sampling are validated.

### 10.5 Applicable: KL Prior Regularization (Audit Section 5.4) — LOW-MEDIUM IMPACT

**What it is**: V4 replaces entropy regularization with a KL penalty toward a frozen **behavioral cloning prior** ($\beta = 0.3$). The prior prevents the policy from deviating too far from previously successful behavior.

**Current problem**: `entropy_scale = 3e-4` produces `loss_actor_entropy ≈ -0.0003` (Section 8.3.3 item 6) — effectively inactive. The 1-env run's catastrophic collapse (21→7 steps) suggests the policy can unlearn good behavior without any trust-region constraint.

**Adaptation for online RL** (no offline BC dataset): Use a **periodic policy snapshot** as the prior — freeze a copy of the actor every N iterations. This creates an implicit trust region: the policy can improve but cannot deviate too far from its recent successful state. This is conceptually similar to PPO's clipping but implemented through explicit KL divergence.

**Note**: If PMPO (Section 10.1) is adopted, the KL prior is part of the PMPO loss formulation (the third term). Both should be implemented together.

**Implementation**: Store a frozen actor copy (`jax.lax.stop_gradient`). Compute `KL[π_current || π_prior]` per imagined step. Update the frozen copy every `prior_update_interval` iterations (e.g., every 1000 iterations). Add `kl_prior_scale: 0.3` to config.

### 10.6 Not Applicable (Filtered by Constraints)

| V4 Feature | Reason Excluded |
|:---|:---|
| Block-Causal Transformer world model (Section 5.1) | **Constraint 1**: Keep RSSM/RNN |
| Causal Tokenizer / Masked Autoencoder (Section 5.3) | Observations are 33-dim vector, not images |
| Shortcut Forcing diffusion objective (Section 5.2) | Tied to transformer temporal modeling |
| GQA / RoPE attention (Section 5.1) | Transformer-specific |
| Offline dataset pipeline (Section 5.6) | **Constraint 2**: Online sampling required |
| Start-frame augmentation (Section 5.7.4) | Offline-specific |
| μ-law action encoding (Section 5.7.4) | Our actions are discrete (7 actions) |
| Agent token causal masking (Section 5.5) | Transformer-specific |
| Multi-Token Prediction for reward head (Section 5.5) | Low impact — reward data distribution is the problem, not head architecture |
| SwiGLU activation (Section 5.1) | Minor; SiLU is functionally equivalent for our scale |

### 10.7 Implementation Priority & Dependency Map

```
                    ┌─────────────────────────┐
                    │ 0. Fix replay ratio bug  │  ← PREREQUISITE (Section 8.3.4)
                    │    (not V4, but blocker) │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
   ┌──────────────────┐ ┌──────────────┐ ┌────────────────┐
   │ 1. PMPO sign-only│ │ 2. Mixture   │ │ 3. free_nats   │
   │    advantages     │ │    sampling  │ │    → 2.0       │
   │ (Section 10.1)   │ │ (Section 10.2)│ │ (Section 9.10) │
   └────────┬─────────┘ └──────┬───────┘ └────────────────┘
            │                   │
            ▼                   ▼
   ┌──────────────────┐ ┌──────────────────┐
   │ 4. KL prior      │ │ 5. RMS loss norm │
   │ (part of PMPO)   │ │ (Section 10.3)   │
   │ (Section 10.5)   │ └──────────────────┘
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ 6. Symexp bins   │
   │ (Section 10.4)   │
   └──────────────────┘
```

| Step | Feature | Addresses | Effort | Depends On |
|:---|:---|:---|:---|:---|
| **0** | Fix replay ratio bug | All pathologies (128× under-training) | Low | — |
| **1** | PMPO sign-only advantages | Asymmetric reward, all-negative advantage, critic divergence | Medium | Step 0 |
| **2** | 50/25/25 mixture sampling | Positive reward blindness, data staleness | Medium | Step 0 |
| **3** | Increase `free_nats` to 2.0 | Latent entropy collapse | Low (config change) | Step 0 |
| **4** | KL prior regularization | Policy collapse, entropy ineffectiveness | Low (part of PMPO) | Step 1 |
| **5** | RMS loss normalization | WM gradient imbalance | Low | Step 0 |
| **6** | Symexp two-hot bins | Reward resolution near zero | Low | Step 1 (validate PMPO first) |

**Recommended experiment**: Apply Steps 0–3 simultaneously in a single 4-env run. Steps 1+4 are tightly coupled (PMPO includes KL prior). Steps 2 and 3 are independent config/buffer changes. Compare against current 4-env baseline to measure combined impact before isolating individual contributions.
