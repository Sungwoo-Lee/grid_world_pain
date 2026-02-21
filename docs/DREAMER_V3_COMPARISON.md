# DreamerV3 Implementation Comparison: Local (JAX/Flax NNX) vs SheepRL (PyTorch)

This document provides a function-by-function comparison of our DreamerV3 implementation
(`src/models/dreamer_v3_nnx.py`, `src/models/dreamer_v3_trainer.py`, `src/models/dreamer_v3_util.py`)
against the reference SheepRL implementation ([Eclectic-Sheep/sheeprl](https://github.com/Eclectic-Sheep/sheeprl),
`sheeprl/algos/dreamer_v3/`), with a focus on **training speed** differences.

---

## 1. High-Level Architecture Comparison

| Component | Local (JAX/Flax NNX) | SheepRL (PyTorch) |
|---|---|---|
| Framework | JAX + Flax NNX | PyTorch + Lightning Fabric |
| Distributed | Single-device | Multi-device via Fabric |
| RSSM loop | `jax.lax.scan` | Python `for` loop |
| Imagination loop | `jax.lax.scan` | Python `for` loop |
| Encoder type | MLP only (vector obs) | CNN + MLP (image + vector) |
| Observation model | Single MSE decoder | Multi-decoder (MSE for CNN, Symlog for MLP) |
| Reward model | TwoHot (255 bins) | TwoHot (255 bins) |
| Continue model | Binary cross-entropy | Bernoulli log-prob |
| Actor distribution | OneHotCategorical (discrete only) | Continuous (Normal/TanhNormal) + Discrete |
| Weight init | Flax defaults (Lecun) | Custom truncated normal (`init_weights`) |
| Gradient clipping | None | Configurable per-module clipping |
| Replay buffer | Simple circular NumPy | Per-env `SequentialReplayBuffer` with memmap |

---

## 2. Function-by-Function Comparison

### 2.1 RSSM

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Class** | `RSSM` (monolithic) | `RSSM` with separate `RecurrentModel`, `representation_model`, `transition_model` | Neutral |
| **GRU cell** | Custom `LayerNormGRUCell` (inline) | `LayerNormGRUCell` from `sheeprl.models.models` | Neutral |
| **Pre-GRU projection** | `img_in` → `elu` | `RecurrentModel.mlp`: Linear → LayerNorm → SiLU | **Local is faster** (1 Linear vs Linear+LN+SiLU) but less expressive |
| **Posterior computation** | `obs_out`: single Linear | `representation_model`: MLP (hidden + output) with LN+SiLU | **Local is faster** (1 layer vs 2+) |
| **Prior computation** | `img_out`: single Linear | `transition_model`: MLP (hidden + output) with LN+SiLU | **Local is faster** (1 layer vs 2+) |
| **Initial state** | Fixed zeros | **Learnable** `initial_recurrent_state` (nn.Parameter) | Local missing feature |
| **Unimix** | Not implemented | `_uniform_mix` injects 1% uniform into categoricals | Local missing feature — may affect exploration |
| **is_first handling** | Binary mask on deter+stoch | Resets to learnable initial state + zeros actions | Local is simpler but less correct |
| **Scan vs loop** | `jax.lax.scan` (compiled) | Python `for i in range(T)` | **Major: Local potentially faster** via XLA fusion |

**Key difference**: SheepRL's RSSM has a **separate MLP for the recurrent model input** (Linear → LN → SiLU → GRU), a **separate MLP for representation** (posterior), and a **separate MLP for transition** (prior). Each is a multi-layer network. The local implementation uses single Linear projections for all three, making it significantly lighter but less expressive.

### 2.2 Encoder

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Architecture** | MLP: N×(Linear → LN → SiLU) + Linear → LN → SiLU | `MLPEncoder`: MLP (N layers, LN, SiLU) | Equivalent |
| **Symlog input** | Applied externally in `train_step` | Built-in `self.symlog_inputs` flag | Neutral |
| **CNN support** | Not present | Full CNN encoder with configurable stages | N/A (local is vector-only) |
| **Output dim** | Configurable `embed_dim` | `dense_units` (last hidden size) | Neutral |
| **Body/act split** | Split for neuromodulation injection | Not split | Neutral |

### 2.3 Decoder

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Architecture** | Single MLP → scalar output | Multi-head decoder: shared MLP body + per-key Linear heads | SheepRL more flexible |
| **Loss distribution** | MSE on raw output | `SymlogDistribution` log-prob for MLP keys, `MSEDistribution` for CNN keys | Different gradient scale — SheepRL matches paper |
| **Decoder uses symlog** | No (raw MSE) | Yes (Symlog distribution) | **Local deviates from paper** |

### 2.4 Reward Model

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Output** | 255 logits → TwoHot | 255 bins → `TwoHotEncodingDistribution` | Equivalent |
| **Loss** | Manual cross-entropy: `-sum(target * log_softmax(pred))` | `-pr.log_prob(rewards)` via distribution class | Equivalent computation |

### 2.5 Continue Model

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Output** | 1 logit | 1 logit | Same |
| **Loss** | `optax.sigmoid_binary_cross_entropy` | `BernoulliSafeMode.log_prob` | Equivalent |
| **Scale factor** | 1.0 (implicit) | Configurable `continue_scale_factor` (default 10.0) | **Different weighting — SheepRL uses 10x** |

### 2.6 Actor

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Class** | `MLP` (generic) | Dedicated `Actor` class | Neutral |
| **Discrete actions** | `OneHotDist` with Gumbel ST | `OneHotCategoricalStraightThrough` + unimix | Local missing unimix |
| **Continuous actions** | Not supported | `Normal`, `TanhNormal`, `ScaledNormal` | N/A |
| **Entropy coeff** | Fixed `3e-4` | Configurable `ent_coef` | Neutral |
| **Actor loss** | `log_prob * advantage` (REINFORCE) | Same for discrete; continuous uses `objective = advantage` directly | Equivalent for discrete |
| **Discount weighting** | Not applied to actor loss | `discount[:-1].detach() * (objective + entropy)` | **Local missing — SheepRL weights by cumulative discount** |

### 2.7 Critic

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Architecture** | `MLP` → 255 TwoHot logits | Same architecture | Same |
| **Loss** | `-sum(target_twohot * log_softmax(pred))` | `-qv.log_prob(lambda_values) - qv.log_prob(target_values)` | **SheepRL uses dual targets (lambda + target critic)** |
| **Discount weighting** | Not applied | `discount[:-1]` weighting | **Local missing** |
| **Target network update** | EMA (τ=0.02) every step | EMA with configurable τ and frequency | Neutral |

### 2.8 Lambda Returns

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Implementation** | `jax.lax.scan` (reverse) | Python `for` loop (reverse) | **Local faster** (compiled scan) |
| **Formula** | `r + c * ((1-λ)*v + λ*next_return)` | Same | Same |
| **Discount** | `conts` (sigmoid of continue logits) | `conts * gamma` (continues multiplied by discount) | **Different: local doesn't multiply by gamma** |

### 2.9 Moments / Return Normalization

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Implementation** | `Moments` NNX Module (EMA percentiles) | `Moments` nn.Module | Equivalent |
| **Usage** | `normalize()` then use directly | `(lambda_values - offset) / invscale` on both lambda and baseline | **SheepRL normalizes both advantage and baseline** |
| **Distributed** | Single-device | `fabric.all_gather` for multi-device sync | N/A |

---

## 3. Training Loop Comparison

### 3.1 World Model Training

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Sequence processing** | `jax.lax.scan` over T steps inside `nnx.grad` | Python `for` loop over T steps | **Major: scan fuses operations, avoids Python overhead** |
| **Encoder call** | Inside scan (per-step) | **Outside loop** — `encoder(batch_obs)` on full batch at once | **SheepRL is faster here**: single batched encoder forward, not T sequential calls |
| **Gradient computation** | `nnx.grad(model_loss_fn)` | `fabric.backward(rec_loss)` | Equivalent |
| **Gradient clipping** | Not implemented | `fabric.clip_gradients(max_norm=...)` | **Local missing — affects training stability** |

**Critical speed difference**: SheepRL encodes **all observations at once** before entering the RSSM loop (`embedded_obs = world_model.encoder(batch_obs)` on full `[T, B, D]` tensor). The local implementation calls `encoder(o)` inside the scan step, processing one timestep at a time. This prevents the encoder from benefiting from large batch parallelism.

### 3.2 Behavior Learning (Imagination)

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Imagination loop** | `jax.lax.scan` over HORIZON steps | Python `for` loop over HORIZON steps | **Local potentially faster** (scan compilation) |
| **Start state** | Reshape posts from WM scan | Same | Same |
| **Actor/Critic grads** | Joint `nnx.grad(behavior_loss_fn, argnums=(0,1))` | Separate `fabric.backward(policy_loss)` then `fabric.backward(value_loss)` | **Local computes both gradients in one pass** |
| **Pre-allocated tensors** | No (scan outputs) | `torch.empty(...)` pre-allocated | Neutral (JAX handles allocation) |

### 3.3 Replay Buffer

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Type** | Simple circular buffer (`ReplayBuffer`) | `EnvIndependentReplayBuffer` wrapping `SequentialReplayBuffer` | **Major difference** |
| **Per-env tracking** | No — single buffer for all envs | Yes — separate buffer per env | SheepRL ensures episode boundaries are respected |
| **Storage** | NumPy arrays in RAM | NumPy arrays, optional **memory-mapped** (`memmap`) | SheepRL can handle larger buffers |
| **Sampling** | Random contiguous slice (no episode boundary check) | Proper sequential sampling respecting episode boundaries | **Local may sample across episode boundaries** — affects learning quality |
| **Batch sampling** | Python loop: `for _ in range(batch_size)` with `np.random.randint` | Vectorized sampling with `sample_tensors` | **SheepRL faster for large batches** |
| **Data transfer** | `np.array(batch)` → JAX arrays (implicit) | `dtype=None, device=fabric.device, from_numpy=True` | Equivalent |

### 3.4 Data Collection

| Aspect | Local | SheepRL | Speed Impact |
|---|---|---|---|
| **Steps per iteration** | `num_steps` (from config `sequence_length`, typically 1) | 1 step per iteration, multiple gradient steps per sample | **Very different cadence** |
| **Replay ratio** | Fixed 1 gradient step per iteration | `Ratio` class — configurable replay ratio | **SheepRL more efficient data re-use** |
| **Gradient steps** | 1 per iteration | `per_rank_gradient_steps` computed by `Ratio` | SheepRL can do many updates per env step |
| **Inference** | No `@torch.no_grad` equivalent | `torch.inference_mode()` for env interaction | JAX handles this differently (no autograd during collection) |

---

## 4. Training Speed Bottleneck Analysis

### 4.1 Critical Bottlenecks in Local Implementation

#### **B1. Encoder called inside scan (per-timestep) — HIGH IMPACT**
The local implementation calls `wm.encoder(o)` inside `jax.lax.scan`, processing one `(B,)` slice at a time.
SheepRL calls `world_model.encoder(batch_obs)` once on the full `(T, B, D)` tensor before the loop.

**Fix**: Pre-compute `embedded_obs = wm.encoder(obs)` for all timesteps before the scan, then pass
embedded observations as scan inputs instead of raw observations.

**Estimated speedup**: 2–5x for the world model training step, depending on encoder depth.

#### **B2. Single-step data collection — HIGH IMPACT**
The DreamerV3 training loop collects only `num_steps` (= `sequence_length` = 1 by default) environment steps
per iteration, then immediately trains. This means:
- Only 4 transitions per iteration (with 4 envs)
- Training doesn't start until buffer reaches `batch_size * batch_length` (16 × 64 = 1024 transitions)
- ~256 iterations of pure collection before any training

SheepRL collects 1 step per iteration but uses a **replay ratio** to determine how many gradient steps
to take, amortizing training over many more gradient updates per environment step.

**Fix**: Increase `num_steps` per collection iteration, or implement a replay ratio mechanism.

#### **B3. Replay buffer sampling is Python-loop — MEDIUM IMPACT**
The `ReplayBuffer.sample()` method uses a Python `for` loop to collect `batch_size` sequences.
SheepRL uses vectorized `sample_tensors` that efficiently samples multiple sequences.

**Fix**: Vectorize sampling with NumPy index arrays instead of per-sample Python loops.

#### **B4. No gradient clipping — MEDIUM IMPACT (stability, not speed)**
SheepRL clips gradients for world model, actor, and critic independently. The local implementation
has no gradient clipping, which can cause training instability (NaN losses, divergence) that wastes time.

**Fix**: Add `optax.clip_by_global_norm` to each optimizer chain.

#### **B5. No JIT on train_step — MEDIUM IMPACT**
The local `DreamerTrainer.train_step` is not explicitly JIT-compiled. While `nnx.grad` triggers
tracing, wrapping the entire `train_step` in `nnx.jit` would ensure the full training pipeline
(WM + behavior + moment update + EMA) is compiled as a single XLA program.

**Fix**: Wrap `train_step` with `nnx.jit`.

#### **B6. Missing is_first / episode boundary handling in buffer — MEDIUM IMPACT (learning quality)**
The replay buffer can sample sequences that cross episode boundaries without proper `is_first` marking.
SheepRL's `EnvIndependentReplayBuffer` ensures valid sequential samples per-environment.

**Fix**: Track episode boundaries and ensure sampled sequences don't span multiple episodes without is_first flags.

### 4.2 Algorithmic Differences Affecting Convergence Speed

| Difference | Impact |
|---|---|
| **No unimix** (1% uniform in categorical) | May reduce exploration diversity, slower convergence |
| **No learnable initial state** | Less robust episode resets |
| **No discount weighting** on actor/critic losses | Incorrect credit assignment for long horizons |
| **Critic loss uses single target** (not dual) | Less stable value estimation |
| **continue_scale_factor = 1** (SheepRL uses 10) | Under-weighted terminal prediction |
| **No gamma in lambda returns** | Lambda returns don't discount properly |
| **Decoder uses MSE** (not Symlog distribution) | Gradient scale mismatch for observation reconstruction |

---

## 5. Recommendations (Ordered by Impact)

### Immediate (High Impact on Training Speed)

1. **Move encoder outside scan**: Pre-compute embeddings for all timesteps, pass as scan input.
   Expected speedup: **2–5x on world model step**.

2. **JIT-compile `train_step`**: Wrap with `nnx.jit` to compile the full training pipeline.
   Expected speedup: **1.5–3x overall** (eliminates Python dispatch overhead between WM/behavior steps).

3. **Implement replay ratio**: Instead of 1 gradient step per N env steps, use configurable
   replay ratio (SheepRL default: 1 gradient step per 2 env steps at 1024 batch).
   Expected improvement: **Better sample efficiency**, faster wall-clock convergence.

4. **Vectorize replay buffer sampling**: Replace Python loop with NumPy batch indexing.
   Expected speedup: **2–5x on sampling** (especially with batch_size=16+).

### Medium Term (Correctness + Stability)

5. **Add gradient clipping**: Use `optax.clip_by_global_norm(1000.0)` matching SheepRL defaults.
6. **Add unimix**: Inject 1% uniform distribution into categorical state posteriors/priors.
7. **Add gamma to lambda returns**: `continues * gamma` instead of just `continues`.
8. **Implement discount weighting**: Weight actor and critic losses by cumulative discount.
9. **Dual critic targets**: Add `target_critic` prediction to critic loss (Eq. 10 in paper).
10. **Use Symlog distribution for decoder**: Replace MSE with Symlog log-prob.

### Long Term (Feature Parity)

11. **Learnable initial recurrent state** for RSSM.
12. **Hafner-style weight initialization** (`init_weights` / `uniform_init_weights`).
13. **Separate representation and transition MLPs** (multi-layer instead of single Linear).
14. **Per-env replay buffer** to properly respect episode boundaries.
15. **Memory-mapped buffer** support for large-scale experiments.

---

## 6. Summary

The local DreamerV3 is a **minimal but functional** JAX port. Its main speed advantages come from
`jax.lax.scan` (compiled sequence processing) and joint actor-critic gradient computation. However,
the **encoder-inside-scan** pattern, **lack of JIT on the full train step**, and **naive replay buffer**
are the primary bottlenecks. The most impactful single change would be **pre-computing encoder
embeddings outside the scan**, which SheepRL already does and which could yield a 2–5x speedup
on the world model training step alone.

On the algorithmic side, several DreamerV3 paper details are missing (unimix, discount weighting,
dual critic targets, gamma in lambda returns, Symlog decoder), which affect convergence speed
more than wall-clock speed.
