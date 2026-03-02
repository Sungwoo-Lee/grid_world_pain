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

- [ ] **4.7.1** Change `configs/models/dreamer_v3.yaml`:
  ```yaml
  batch_size: 16          # was 64 (sequence_length stays at 128)
  ```
- [ ] **4.7.2** Run benchmark: 64env, CI=128, RR=1.0, same tag format.
- [ ] **4.7.3** Record steady-state s/it and SPS. Expected: **~3.2-4.5 s/it** (per-step drops from ~0.19s to ~0.05s).
- [ ] **4.7.4** Record VRAM usage. Expected: **~6-9 GB** (down from 18.4 GB — activation memory is proportional to batch × seq).

**Expected results table**:
| Metric | Current (64×128) | Reduced (16×128) | Improvement |
|:---|---:|---:|:---|
| Imagined transitions / step | 122,880 | 30,720 | 4x less compute |
| Per-step time (est.) | ~0.19s | ~0.05s | ~4x faster |
| s/it (64 grad steps) | 13.0 | ~3.2-4.5 | ~3-4x faster |
| VRAM | 18.4 GB | ~6-9 GB | ~2-3x less memory |

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
- [ ] Steady-state `s/it < 5.0` for 64env CI=128 RR=1.0 with `batch_size: 16`.
- [ ] No JIT retracing after warmup (already achieved).
- [ ] GPU memory < 50% of 24 GB (with reduced batch, should be ~6-9 GB).
- [ ] Results recorded in Section 8.

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
