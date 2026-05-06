---
title: "NMN Metrics Reference: Calculation Details"
topic: neuromodulation
status: active
created: 2026-03-10
last_updated: 2026-04-12
---

# NMN Metrics Reference: Calculation Details

> **Date**: 2026-03-10
> **Author**: Claude
> **Scope**: Detailed reference for all neuromodulator (NMN) metrics logged to WandB during RecurrentPPO training.
> **Key files**: `train.py`, `src/models/neuromodulator.py`, `src/models/recurrent_ppo_trainer.py`, `src/models/recurrent_ppo_network.py`, `src/models/modulated_gru_cell.py`
> **Related**: [NMN_ARCHITECTURE_REVIEW.md](NMN_ARCHITECTURE_REVIEW.md), [NMN_PERFORMANCE_DIAGNOSIS.md](NMN_PERFORMANCE_DIAGNOSIS.md), [WANDB_METRICS_REFERENCE.md](WANDB_METRICS_REFERENCE.md)

---

## 1. Data Pipeline Overview

### 1.1 Where `mod_info` Is Produced

During rollout collection, each environment step calls the `ActorCriticRNN.__call__()` method, which runs the modulator forward pass and returns a `ModulatorOutput` namedtuple:

```python
# recurrent_ppo_network.py:257-258
mod_output, mod_h_new = self.modulator(x, mod_h)
# Returns: ModulatorOutput(z_unimodal, z_unimodal_add, z_multimodal,
#                           z_multimodal_add, z_memory, temperature)
```

This `mod_output` is returned as the 4th value from the network call:

```python
# recurrent_ppo_network.py:282
return logits, value, h_combined_new, mod_output
```

### 1.2 How `mod_info` Is Collected Across Timesteps

The rollout collector uses `jax.lax.scan` over `num_steps` timesteps. At each step, the per-env `mod_info` is stored in a `Transition` namedtuple:

```python
# recurrent_ppo_trainer.py:160-163
action, log_prob, value, h_new, mod_info = jax.vmap(
    get_action_and_value_nnx, in_axes=(None, 0, h_axes, 0)
)(model, obs, h_state, act_keys)

# recurrent_ppo_trainer.py:199-202
trans = Transition(
    obs=obs, action=action, reward=reward, done=done,
    log_prob=log_prob, value=value, mod_info=mod_info,
    step_info=step_info
)
```

Because `jax.lax.scan` stacks outputs along a new leading axis, the final `trajectories.mod_info` is a `ModulatorOutput` where **each field has shape `(T, B, ...)`** — `T` timesteps, `B` parallel environments:

| Field | Per-step shape `(B, ...)` | Stacked shape `(T, B, ...)` |
|-------|--------------------------|----------------------------|
| `z_unimodal` | `(B, 9)` | `(T, B, 9)` |
| `z_unimodal_add` | `(B, 9)` | `(T, B, 9)` |
| `z_multimodal` | `(B, 128)` | `(T, B, 128)` |
| `z_multimodal_add` | `(B, 128)` | `(T, B, 128)` |
| `z_memory` | `(B, 128)` | `(T, B, 128)` |
| `temperature` | `(B, 1)` | `(T, B, 1)` |

Where `T = num_steps` (e.g., 128), `B = num_envs` (e.g., 32), `9 = num_modalities`, `128 = target_hidden_size`.

### 1.3 How Metrics Are Logged

After the PPO update, `train.py` reads `mod_info` from the collected trajectories:

```python
# train.py:823
mod_info = trajectories.mod_info
```

All logged metrics use `jnp.mean()`, `jnp.std()`, `jnp.min()`, or `jnp.max()` **flattened across the entire `(T, B, ...)` tensor** — collapsing timesteps, environments, and neurons/modalities into a single scalar.

---

## 2. Metric Definitions

### 2.1 `modulator/grad_norm`

**What it measures**: The L2 norm of gradients for the modulator sub-network parameters only.

**Computation** (`recurrent_ppo_trainer.py:229-239`):

```python
# After computing full-model gradients via nnx.value_and_grad:
(loss, aux), grads = nnx.value_and_grad(batch_loss_wrapped, has_aux=True)(model)

grad_norm = optax.global_norm(grads)       # Full model gradient norm

# Extract modulator-only gradient norm:
mod_grad_norm = 0.0
if hasattr(model, 'modulation_enabled') and model.modulation_enabled:
    if 'modulator' in grads:
        mod_grad_norm = optax.global_norm(grads['modulator'])
```

The `optax.global_norm()` computes:

$$\text{global\_norm} = \sqrt{\sum_i \| g_i \|_2^2}$$

where $g_i$ are all parameter tensors within the modulator sub-tree (GRU weights, head weights, baselines).

**Logged value** (`train.py:920, 935`):

```python
avg_mod_grad_norm = jnp.mean(jnp.array([l[1][4] for l in losses]))
# losses contains one entry per PPO epoch (num_epochs entries)
# l[1][4] is the mod_grad_norm from each epoch's update_step
# Result: mean of mod_grad_norm across PPO epochs within this iteration
```

**Shape flow**: scalar per epoch → mean across epochs → single scalar logged.

**Interpretation**:
- **High values** (>1.0): Modulator parameters are receiving strong gradient signal; rapid learning.
- **Near-zero** (<0.01): Modulator has converged or is in a gradient plateau; modulation signals are not contributing to loss reduction.
- **Spikes**: May indicate instability — check if correlated with loss spikes or behavioral changes.

---

### 2.2 `modulator/gamma_uni_mean` and `modulator/gamma_uni_std`

**What they measure**: Summary statistics of the **raw (pre-sigmoid) unimodal gain signal** `z_unimodal` across all timesteps, environments, and modalities.

**Important**: These log the **pre-sigmoid** value. The actual gain applied during injection is `sigmoid(z_unimodal)`.

#### Signal Computation (`neuromodulator.py:137-157`)

```python
def _get_signal(head, baseline, ..., is_unimodal=False):
    raw = head(h_mod_new)          # Linear(mod_hidden_size → 9), shape (B, 9)
    if is_unimodal:
        sig = baseline.value + raw # shape (B, 9)
    ...

z_uni, z_uni_add = _get_signal(
    self.head_unimodal,            # Linear(16 → 9), bias_init=2.0
    self.z_unimodal_baseline,      # Param(zeros(9))
    ...
    is_unimodal=True
)
```

Step-by-step for a single env at one timestep:

1. **GRU update**: `h_mod_new = GRU(h_mod_prev, obs)` — shape `(16,)`
2. **Head projection**: `raw = W @ h_mod_new + b` — `W` shape `(16, 9)`, `b` shape `(9,)` init to `2.0` → output shape `(9,)`
3. **Add baseline**: `sig = z_unimodal_baseline + raw` — baseline shape `(9,)` init to `0.0`
4. **Result**: `z_unimodal` shape `(9,)` — one value per sensor modality

At initialization: `raw ≈ 0 + 2.0 = 2.0` (bias dominates), `baseline = 0.0`, so `z_unimodal ≈ 2.0` for all modalities.

#### How It's Injected (`recurrent_ppo_network.py:139-144`)

```python
encoded_all = self.unimodal_grouped(x_padded)     # shape (B, 9, 128)
gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)    # shape (B, 9) → values in (0, 1)
beta1 = mod_output.z_unimodal_add                  # shape (B, 9)
encoded_all = jax.nn.relu(encoded_all * gamma1[..., None] + beta1[..., None])
#  gamma1[..., None] broadcasts (B, 9, 1) × (B, 9, 128) → (B, 9, 128)
#  Each modality's 128 neurons share the SAME gamma1 value
```

#### Logged Values (`train.py:936-937`)

```python
"modulator/gamma_uni_mean": float(jnp.mean(mod_info.z_unimodal)),
"modulator/gamma_uni_std":  float(jnp.std(mod_info.z_unimodal)),
```

- `jnp.mean(mod_info.z_unimodal)` flattens `(T, B, 9)` → mean over all `T×B×9` values.
- `jnp.std(mod_info.z_unimodal)` — standard deviation over the same flattened tensor.

#### Conversion Table: Pre-Sigmoid → Actual Gain

| Logged `gamma_uni_mean` | `sigmoid(value)` | Effect |
|------------------------|------------------|--------|
| -5.0 | 0.007 | Near-complete suppression of all modalities |
| -2.0 | 0.119 | Strong attenuation (~88% signal loss) |
| 0.0 | 0.500 | Half-gain (50% signal passed) |
| 2.0 (init) | 0.881 | Near pass-through (~12% attenuation) |
| 5.0 | 0.993 | Effectively transparent gate |
| 10.0 | 0.99995 | Fully open (ceiling of sigmoid) |

#### Worked Example

With `T=128`, `B=32`, `9` modalities:
- Total elements: `128 × 32 × 9 = 36,864`
- If the modulator has learned to suppress injury (modality 0) with `z=-4.0` and leave others near init at `z≈2.0`:
  - `gamma_uni_mean ≈ (8 × 2.0 + 1 × (-4.0)) / 9 ≈ 1.33`
  - `gamma_uni_std ≈ 2.0` (high, reflecting the spread between -4.0 and 2.0)
- The mean/std collapse per-modality detail — always check per-modality values in debug mode for diagnosis.

---

### 2.3 `modulator/gamma_multi_mean` and `modulator/gamma_multi_std`

**What they measure**: Summary statistics of the **raw (pre-sigmoid) multimodal gain signal** `z_multimodal` across all timesteps, environments, and neurons.

#### Signal Computation (`neuromodulator.py:137-161`)

```python
raw = head(h_mod_new)  # Linear(mod_hidden_size → num_groups_hidden)
                        # With G=64, hidden=128: Linear(16 → 2), shape (B, 2)

# Expand groups to per-neuron:
sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
# jnp.repeat([a, b], 64) → [a,a,...×64, b,b,...×64] → shape (B, 128)
# Slice [:128] is no-op when 2×64=128

sig = baseline.value + sig  # z_hidden_baseline shape (128,) + sig shape (B, 128)
```

Step-by-step for a single env at one timestep (with `grouping_size=64`, `hidden_size=128`):

1. **Head projection**: `raw = W @ h_mod_new + b` — `W` shape `(16, 2)`, `b` shape `(2,)` init to `2.0` → output `[a, b]`
2. **Repeat**: `[a, b]` → `[a, a, ...(×64), b, b, ...(×64)]` = 128 values
3. **Add per-neuron baseline**: `sig[i] = z_hidden_baseline[i] + repeated[i]`
   - Neurons 0–63: `baseline[0..63] + a`
   - Neurons 64–127: `baseline[64..127] + b`
4. **Result**: `z_multimodal` shape `(128,)` — per-neuron but grouped

At initialization: `a ≈ b ≈ 2.0`, baselines `= 0.0`, so `z_multimodal ≈ 2.0` for all 128 neurons.

#### How It's Injected (`recurrent_ppo_network.py:146-151`)

```python
mm_in = encoded_all.reshape(batch_shape + (-1,))   # (B, 9×128) = (B, 1152)
mm_latent = self.multimodal_hub(mm_in)              # MLP → (B, 128)
gamma2 = jax.nn.sigmoid(mod_output.z_multimodal)    # (B, 128), values in (0, 1)
beta2 = mod_output.z_multimodal_add                  # (B, 128)
return jax.nn.relu(mm_latent * gamma2 + beta2)       # element-wise
```

#### Logged Values (`train.py:938-939`)

```python
"modulator/gamma_multi_mean": float(jnp.mean(mod_info.z_multimodal)),
"modulator/gamma_multi_std":  float(jnp.std(mod_info.z_multimodal)),
```

- Flattens `(T, B, 128)` → mean/std over all `T×B×128` values.
- Total elements: `128 × 32 × 128 = 524,288`.

#### Interpretation

The per-neuron baselines cause `gamma_multi_std` to be nonzero even if the head outputs are constant — each neuron has its own learned offset. A low `gamma_multi_mean` (negative) means the multimodal hub output is being suppressed, which is the "feature collapse" pathology documented in NMN_PERFORMANCE_DIAGNOSIS.md.

---

### 2.4 `modulator/z_memory_mean` and `modulator/z_memory_std`

**What they measure**: Summary statistics of the **raw memory gate-bias signal** `z_memory` that shifts the GRU update gate.

#### Signal Computation (`neuromodulator.py:163-164`)

```python
z_mem, _ = _get_signal(self.head_memory, self.z_mem_baseline)
# head_memory: Linear(mod_hidden_size → num_groups_hidden)
# With G=64: Linear(16 → 2), bias_init=0.0
# z_mem_baseline: Param(zeros(128))
```

Same repeat-and-slice as multimodal, but with `memory_bias_init=0.0` (not 2.0).

At initialization: `raw ≈ 0 + 0 = 0`, baseline `= 0`, so `z_memory ≈ 0.0` → no gate bias at start.

#### How It's Injected (`modulated_gru_cell.py:63-76`)

```python
# Reset gate (NOT modulated):
r = jax.nn.sigmoid(self.W_ir(x) + self.W_hr(h))

# Update gate (modulated by gate_bias):
u_pre = self.W_iu(x) + self.W_hu(h)
if gate_bias is not None:
    u_pre = u_pre + gate_bias     # <<< z_memory added here
u = jax.nn.sigmoid(u_pre)

# Candidate hidden state:
n = jnp.tanh(self.W_in(x) + r * self.W_hn(h))

# New hidden state:
h_new = (1.0 - u) * h + u * n
```

The injection point is at `recurrent_ppo_network.py:269`:

```python
h_new, x_h = self.rnn_cell(task_h, x_proj, gate_bias=mod_output.z_memory)
```

#### Effect on GRU Dynamics

The update gate `u` controls the interpolation between old state `h` and new candidate `n`:
- `h_new = (1 - u) * h + u * n`

The gate-bias shifts the sigmoid operating point:

| `z_memory` value | Effect on `sigmoid(u_pre + z_memory)` | GRU behavior |
|-----------------|---------------------------------------|--------------|
| -6.0 | `sigmoid(anything - 6.0) ≈ 0.002` | **Memory freeze**: `h_new ≈ h` (old state preserved, new input ignored) |
| -2.0 | Shifts sigmoid left by 2.0 | Biased toward retention |
| 0.0 (init) | No effect — standard GRU | Normal dynamics |
| +2.0 | Shifts sigmoid right by 2.0 | Biased toward updating |
| +6.0 | `sigmoid(anything + 6.0) ≈ 0.998` | **Memory wipe**: `h_new ≈ n` (old state discarded, fully replaced by new input) |

#### Logged Values (`train.py:940-941`)

```python
"modulator/z_memory_mean": float(jnp.mean(mod_info.z_memory)),
"modulator/z_memory_std":  float(jnp.std(mod_info.z_memory)),
```

- Flattens `(T, B, 128)` → mean/std over all `T×B×128` values.

#### Diagnostic Value

- `z_memory_mean` drifting strongly negative (e.g., -5 to -7) indicates the modulator is freezing the task GRU — converting the recurrent network into a feedforward one. This was a key pathology in early NMN training.
- `z_memory_std` near zero means the gate-bias is uniform across neurons (all frozen or all normal). High std means different neuron groups have different memory dynamics.

---

### 2.5 `modulator/temperature_mean`, `modulator/temperature_min`, `modulator/temperature_max`

**What they measure**: Summary statistics of the **bounded temperature scalar** used to scale policy logits.

#### Signal Computation (`neuromodulator.py:167-168`)

```python
z_act_raw = self.head_action(h_mod_new)  # Linear(16 → 1), default bias init
temperature = jnp.clip(
    jax.nn.softplus(z_act_raw) + 0.5,
    self.temp_clip[0],    # default 0.1
    self.temp_clip[1]     # default 10.0
)
```

Step-by-step:

1. **Head projection**: `z_act_raw = W @ h_mod_new + b` — `W` shape `(16, 1)`, `b` shape `(1,)` → scalar
2. **Softplus**: `softplus(z) = log(1 + exp(z))` — ensures non-negativity, smooth approximation of ReLU
3. **Offset**: `+0.5` — prevents temperature from being near-zero at initialization. At init `z ≈ 0`, `softplus(0) = ln(2) ≈ 0.693`, so `temperature_init ≈ 0.693 + 0.5 = 1.193`
4. **Clip**: Bounded to `[0.1, 10.0]`

#### How It's Injected (`recurrent_ppo_network.py:276`)

```python
logits = logits / mod_output.temperature
```

Applied after the actor head, before action sampling. The temperature divides the raw logits:

| Temperature (τ) | Effect on `softmax(logits / τ)` | Behavior |
|-----------------|--------------------------------|----------|
| 0.1 (min clip) | Logits multiplied by 10× | Near-deterministic (argmax) |
| 0.5 | Logits multiplied by 2× | Sharp, confident actions |
| 1.0 | No change | Standard policy |
| 1.19 (init) | Slightly flatter | Mild initial exploration boost |
| 3.0 | Logits divided by 3× | High exploration, diffuse actions |
| 10.0 (max clip) | Logits divided by 10× | Near-uniform random (each of 5 actions ≈ 20%) |

#### Logged Values (`train.py:942-944`)

```python
"modulator/temperature_mean": float(jnp.mean(mod_info.temperature)),
"modulator/temperature_min":  float(jnp.min(mod_info.temperature)),
"modulator/temperature_max":  float(jnp.max(mod_info.temperature)),
```

- `mod_info.temperature` has shape `(T, B, 1)`.
- `mean` flattens across all `T×B` values → average temperature across the rollout.
- `min`/`max` report the extreme values seen in the rollout — useful for detecting if temperature is hitting the clip bounds.

#### Diagnostic Value

- `temperature_mean ≈ 1.0`: Modulator is not interfering with the policy — healthy.
- `temperature_mean > 2.0`: Modulator is flattening the policy — can indicate the modulator is using temperature inflation as a crutch (see NMN_PERFORMANCE_DIAGNOSIS.md §5.5).
- `temperature_max` pinned at `10.0`: Temperature has hit the upper clip bound — the modulator is pushing for maximum randomness in at least some states. A warning sign.
- `temperature_min` pinned at `0.1`: Some states have near-deterministic action selection — may be appropriate in well-learned states.

---

### 2.6 `modulator/beta_uni_mean` and `modulator/beta_uni_std` (PreActivation only)

**What they measure**: Summary statistics of the **unimodal additive threshold shift** `z_unimodal_add`. Only logged when `modulation.type == "PreActivation"`.

#### Signal Computation (`neuromodulator.py:91-93, 145-152`)

```python
# Only constructed in PreActivation mode:
self.head_unimodal_add = nnx.Linear(
    mod_hidden_size, self.num_groups_unimodal,
    bias_init=nnx.initializers.constant(percept_add_bias_init),  # default 0.0
    rngs=rngs
)
self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(self.num_groups_unimodal))

# In _get_signal with head_add provided:
raw_add = head_add(h_mod_new)        # Linear(16 → 9), shape (B, 9)
sig_add = baseline_add.value + raw_add  # shape (B, 9)
```

At initialization: `raw_add ≈ 0 + 0 = 0`, baseline `= 0`, so `z_unimodal_add ≈ 0.0` → no threshold shift.

#### How It's Injected (`recurrent_ppo_network.py:139-144`)

```python
encoded_all = self.unimodal_grouped(x_padded)     # shape (B, 9, 128) — pre-activation
gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)    # gain ∈ (0, 1)
beta1 = mod_output.z_unimodal_add                  # threshold shift ∈ (-∞, +∞)
encoded_all = jax.nn.relu(encoded_all * gamma1[..., None] + beta1[..., None])
```

The beta shifts the pre-ReLU activation:
- **Positive beta**: Lowers the effective activation threshold → more neurons fire (disinhibition)
- **Negative beta**: Raises the threshold → fewer neurons fire (selective gating of weak signals)

#### Logged Values (`train.py:948-949`)

```python
"modulator/beta_uni_mean": float(jnp.mean(mod_info.z_unimodal_add)),
"modulator/beta_uni_std":  float(jnp.std(mod_info.z_unimodal_add)),
```

- Flattens `(T, B, 9)` → mean/std over all `T×B×9` values.

---

### 2.7 `modulator/beta_multi_mean` and `modulator/beta_multi_std` (PreActivation only)

**What they measure**: Summary statistics of the **multimodal additive threshold shift** `z_multimodal_add`. Only logged when `modulation.type == "PreActivation"`.

#### Signal Computation (`neuromodulator.py:98-100, 145-152`)

```python
# Only constructed in PreActivation mode:
self.head_multimodal_add = nnx.Linear(
    mod_hidden_size, self.num_groups_hidden,
    bias_init=nnx.initializers.constant(percept_add_bias_init),  # default 0.0
    rngs=rngs
)
self.z_hidden_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))  # (128,)

# In _get_signal:
raw_add = head_add(h_mod_new)        # Linear(16 → num_groups), shape (B, 2) with G=64
sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
sig_add = baseline_add.value + sig_add  # shape (B, 128)
```

Same repeat-and-slice + per-neuron baseline pattern as the multimodal gain, but with `bias_init=0.0`.

#### How It's Injected (`recurrent_ppo_network.py:146-151`)

```python
mm_latent = self.multimodal_hub(mm_in)              # (B, 128)
gamma2 = jax.nn.sigmoid(mod_output.z_multimodal)    # (B, 128)
beta2 = mod_output.z_multimodal_add                  # (B, 128)
return jax.nn.relu(mm_latent * gamma2 + beta2)
```

#### Logged Values (`train.py:950-951`)

```python
"modulator/beta_multi_mean": float(jnp.mean(mod_info.z_multimodal_add)),
"modulator/beta_multi_std":  float(jnp.std(mod_info.z_multimodal_add)),
```

- Flattens `(T, B, 128)` → mean/std over all `T×B×128` values.

---

## 3. Summary: All Logged Metrics

| WandB Key | Source Field | Shape (T,B,...) | Aggregation | Units | Init Value |
|-----------|-------------|-----------------|-------------|-------|------------|
| `modulator/grad_norm` | gradient tree | scalar per epoch | Mean across PPO epochs | L2 norm | N/A |
| `modulator/gamma_uni_mean` | `z_unimodal` | `(T, B, 9)` | `jnp.mean` (flatten all) | Pre-sigmoid | ~2.0 |
| `modulator/gamma_uni_std` | `z_unimodal` | `(T, B, 9)` | `jnp.std` (flatten all) | Pre-sigmoid | ~0.0 |
| `modulator/gamma_multi_mean` | `z_multimodal` | `(T, B, 128)` | `jnp.mean` (flatten all) | Pre-sigmoid | ~2.0 |
| `modulator/gamma_multi_std` | `z_multimodal` | `(T, B, 128)` | `jnp.std` (flatten all) | Pre-sigmoid | ~0.0 |
| `modulator/z_memory_mean` | `z_memory` | `(T, B, 128)` | `jnp.mean` (flatten all) | Raw additive bias | ~0.0 |
| `modulator/z_memory_std` | `z_memory` | `(T, B, 128)` | `jnp.std` (flatten all) | Raw additive bias | ~0.0 |
| `modulator/temperature_mean` | `temperature` | `(T, B, 1)` | `jnp.mean` (flatten all) | Bounded scalar | ~1.19 |
| `modulator/temperature_min` | `temperature` | `(T, B, 1)` | `jnp.min` (flatten all) | Bounded scalar | ~1.19 |
| `modulator/temperature_max` | `temperature` | `(T, B, 1)` | `jnp.max` (flatten all) | Bounded scalar | ~1.19 |
| `modulator/beta_uni_mean` | `z_unimodal_add` | `(T, B, 9)` | `jnp.mean` (flatten all) | Raw additive | ~0.0 |
| `modulator/beta_uni_std` | `z_unimodal_add` | `(T, B, 9)` | `jnp.std` (flatten all) | Raw additive | ~0.0 |
| `modulator/beta_multi_mean` | `z_multimodal_add` | `(T, B, 128)` | `jnp.mean` (flatten all) | Raw additive | ~0.0 |
| `modulator/beta_multi_std` | `z_multimodal_add` | `(T, B, 128)` | `jnp.std` (flatten all) | Raw additive | ~0.0 |

---

## 4. Interpreting Metrics: Healthy vs Pathological Ranges

Based on training observations documented in NMN_PERFORMANCE_DIAGNOSIS.md:

| Metric | Healthy Range | Warning | Pathological |
|--------|--------------|---------|-------------|
| `gamma_uni_mean` | 1.0 – 3.0 | < 0.0 or > 5.0 | < -3.0 (modalities suppressed) |
| `gamma_multi_mean` | 1.0 – 3.0 | < 0.0 or > 5.0 | < -5.0 (feature collapse) |
| `z_memory_mean` | -1.0 – 1.0 | < -2.0 or > 2.0 | < -5.0 (GRU frozen) |
| `temperature_mean` | 0.8 – 1.5 | > 2.0 or < 0.5 | > 3.0 (policy override) or = 0.1 (deterministic collapse) |
| `temperature_max` | < 3.0 | = 10.0 (hitting clip) | Sustained at 10.0 |
| `grad_norm` | 0.01 – 1.0 | > 5.0 | > 10.0 (exploding) or < 0.001 (vanishing) |

---

## 5. Key Caveats

### 5.1 Gamma Metrics Are Pre-Sigmoid

The logged `gamma_uni_mean` and `gamma_multi_mean` are the **raw pre-sigmoid values**, not the actual gains applied. To interpret the actual gating effect, apply `sigmoid()`:

```
Actual gain = sigmoid(logged_value)
```

This is a common source of confusion. A logged value of `-4.0` means `sigmoid(-4.0) = 0.018` — the gate is 98% closed, not "slightly negative."

### 5.2 Flattened Aggregation Masks Per-Modality Behavior

Because `gamma_uni_mean` averages across all 9 modalities, a situation where 8 modalities have `z=3.0` and 1 has `z=-10.0` would show:
- `gamma_uni_mean ≈ 1.56` (looks benign)
- `gamma_uni_std ≈ 4.3` (reveals the spread)

Always check **both mean and std** together. A high std relative to the mean indicates heterogeneous modulation across modalities/neurons — which may be desirable (selective gating) or pathological (collapse of specific modalities).

### 5.3 `mod_info` Contains the Last Iteration's Rollout Only

The logged `mod_info` comes from `trajectories.mod_info` which is the rollout collected in the current iteration — **before** the PPO update. The gradient norm, however, is from the update that follows. There is a one-iteration lag between the modulator state being logged and the gradient that will change it.

### 5.4 Beta Metrics Are Only Logged for PreActivation Type

The `beta_uni_*` and `beta_multi_*` metrics are gated by:

```python
if modulation_config is not None and modulation_config.get('type') == "PreActivation":
```

In Multiplicative mode, these fields exist in `ModulatorOutput` but are filled with zeros. They are not logged to WandB.
