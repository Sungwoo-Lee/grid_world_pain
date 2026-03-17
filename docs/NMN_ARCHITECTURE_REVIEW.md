# NMN Architecture Review: RecurrentPPO Neuromodulatory Network

> **Reviewer**: Claude | **Date**: 2026-03-06 (updated 2026-03-12)
> **Scope**: Structural review of the neuromodulatory network as implemented in RecurrentPPO.
> **Key files**: `src/models/neuromodulator.py`, `src/models/recurrent_ppo_network.py`, `src/models/modulated_gru_cell.py`, `configs/models/neuromodulated_ppo.yaml`

---

## 1. Architecture Summary

The NMN is a **branched recurrent modulator** that runs in parallel with the task network. A small GRU maintains an "affective state" and produces signals that modulate three injection points in the task network: perception, memory, and action selection.

```
Observation ──┬──────────────────────────────────── ObservationEncoder ──── RNN ──── Actor/Critic
              │                                         ▲    (Inj A)        ▲ (Inj B)   ▲ (Inj C)
              └──► NeuromodulatorRNN ── GRU(16) ──┬── z_unimodal ──────────┘            │
                                                  ├── z_multimodal ────────┘            │
                                                  ├── z_memory ─────────────────────────┘ (gate-bias)
                                                  └── temperature ──────────────────────┘ (logit scaling)
```

### Component Inventory

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `NeuromodulatorRNN` | `neuromodulator.py` | 41–172 | Recurrent core + branched output heads |
| `ModulatorOutput` | `neuromodulator.py` | 31–38 | NamedTuple carrying all modulation signals |
| `ModulatedGRUCell` | `modulated_gru_cell.py` | 17–78 | Standard GRU with external update gate bias |
| `ObservationEncoder` | `recurrent_ppo_network.py` | 72–151 | Hierarchical encoder with modulated forward path |
| `ActorCriticRNN` | `recurrent_ppo_network.py` | 154–321 | Main network integrating all injections |

---

## 2. NeuromodulatorRNN Internals

### 2.1 Recurrent Core

- **Single GRU**: `GRUCell(obs_dim, mod_hidden_size)` — receives the raw observation directly (after symlog compression in `ActorCriticRNN`).
- **Hidden size**: 16 (config). This is deliberately small — intended to represent slowly-evolving affective tone rather than rich state representation.
- **No input projection**: The raw observation (dimension = `obs_dim`, typically ~30) feeds directly into the GRU. This means the GRU input-to-hidden weights are shape `(obs_dim, 3 * mod_hidden_size)` = `(30, 48)` — a significant compression.

### 2.2 Branched Output Heads

Six linear heads project from the shared GRU hidden state. With current config `grouping_size=4`, `hidden_size=128`: `num_groups = ceil(128/4) = 32`.

| Head | Output Shape | Activation | Init Bias | Role |
|------|-------------|------------|-----------|------|
| `head_unimodal` | `(num_groups_hidden,)` = 32 | — (sigmoid applied at injection) | 3.0 | Spatial gain pattern applied uniformly to all modalities (gamma) |
| `head_unimodal_add` | `(num_groups_hidden,)` = 32 | — | 0.0 | Spatial bias pattern applied uniformly to all modalities (beta, PreActivation only) |
| `head_multimodal` | `(num_groups_hidden,)` = 32 | — (sigmoid applied at injection) | 3.0 | Multimodal fusion gain |
| `head_multimodal_add` | `(num_groups_hidden,)` = 32 | — | 0.0 | Multimodal fusion bias (PreActivation only) |
| `head_memory` | `(num_groups_hidden,)` = 32 | — (raw added to GRU gate) | 0.0 | GRU update gate bias |
| `head_action` | `(1,)` | softplus + offset + clip | default | Temperature scalar |

**Note on Multiplicative mode**: `head_unimodal_add` and `head_multimodal_add` are **not constructed** — the corresponding `z_*_add` outputs are filled with zeros. Only the gamma (gain) heads are active.

### 2.3 Per-Neuron Learned Baselines

Each head's output is added to a **learned baseline parameter** before use. All baselines are per-neuron (shape `(target_hidden_size,)` = `(128,)`):

```python
# neuromodulator.py:115-117
self.z_unimodal_baseline = nnx.Param(jnp.zeros(target_hidden_size))  # shape (128,)
self.z_hidden_baseline = nnx.Param(jnp.zeros(target_hidden_size))    # shape (128,)
self.z_mem_baseline = nnx.Param(jnp.zeros(target_hidden_size))       # shape (128,)
```

Signal computation (unified for all heads):
```python
# neuromodulator.py:139-142
raw = head(h_mod_new)                                              # shape (batch, 32) with G=4
sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
sig = baseline.value + sig                                         # shape (batch, 128)
```

These baselines allow each neuron to learn its own "resting" modulation level independently of the GRU dynamics. At initialization, baseline + head_bias = 0 + 3.0 = 3.0 for perceptual heads → sigmoid(3.0) ≈ 0.95 initial gain.

### 2.4 Spatial Grouping

All modulation heads — unimodal, multimodal, and memory — use the **same unified grouping mechanism** controlled by the `grouping_size` config parameter. The number of output groups is:

```python
# neuromodulator.py:82-83
self.num_groups_hidden = math.ceil(target_hidden_size / grouping_size)
self.num_groups_unimodal = self.num_groups_hidden  # unified — same as multimodal
```

#### 2.4.1 Repeat-and-Slice Grouping

All heads produce a small number of group signals that are then **repeated** to cover the full hidden dimension:

```python
# neuromodulator.py:82
self.num_groups_hidden = math.ceil(target_hidden_size / grouping_size)
# With grouping_size=4, hidden_size=128: ceil(128/4) = 32 groups

# neuromodulator.py:91-92
self.head_unimodal = nnx.Linear(mod_hidden_size, self.num_groups_unimodal, ...)
# Linear(16 → 32)

# neuromodulator.py:98-99
self.head_multimodal = nnx.Linear(mod_hidden_size, self.num_groups_hidden, ...)
# Linear(16 → 32)
```

**Signal computation** — In the unified `_get_signal()`, all heads use the same repeat-and-slice path:

```python
# neuromodulator.py:139-142
def _get_signal(head, baseline, head_add=None, baseline_add=None):
    raw = head(h_mod_new)                                              # shape (batch, 32) with G=4
    sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
    #     jnp.repeat([a, b, c, ..., z], 4) → [a,a,a,a, b,b,b,b, ..., z,z,z,z]  shape (128,)
    sig = baseline.value + sig                                         # shape (batch, 128)
```

**Key design point**: The unimodal signal has shape `(batch, 128)` — the same as multimodal. At injection, it broadcasts over the modality dimension: `(batch, 1, 128) × (batch, 9, 128)`. This means all 9 modalities receive the **same spatial gate pattern** — the modulator controls **which neuron positions** are amplified/suppressed, not which modalities.

**Worked example** — With `grouping_size=4` and `hidden_size=128` (32 groups):

```
head_unimodal(h_mod) → raw = [a, b, c, ..., z]   # 32 raw group values
jnp.repeat(raw, 4) → [a,a,a,a, b,b,b,b, ..., z,z,z,z]  # expanded to 128

+ z_unimodal_baseline (128 per-neuron values):
  sig[0]  = baseline[0]  + a    ┐
  sig[1]  = baseline[1]  + a    │ group 0: same 'a' but different baselines
  sig[2]  = baseline[2]  + a    │
  sig[3]  = baseline[3]  + a    ┘
  sig[4]  = baseline[4]  + b    ┐
  ...                           │ group 1: same 'b' but different baselines
  sig[7]  = baseline[7]  + b    ┘
  ...

Applied to ALL modalities uniformly (broadcast over dim 0):
  Injury    neurons 0-3: × sigmoid(sig[0:4])   neurons 4-7: × sigmoid(sig[4:8])   ...
  Nutrition neurons 0-3: × sigmoid(sig[0:3])   neurons 4-7: × sigmoid(sig[4:8])   ...
  ...
  Location  neurons 0-3: × sigmoid(sig[0:4])   neurons 4-7: × sigmoid(sig[4:8])   ...
```

**Key subtlety**: Although neurons within a group share the same raw head signal, the **per-neuron baselines** allow each neuron to learn a different resting modulation level. The head controls group-level *dynamics* (how modulation changes over time), while the baselines control per-neuron *static offsets*.

The **memory head** follows the same repeat-and-slice pattern:

```python
# neuromodulator.py:105-109
self.head_memory = nnx.Linear(mod_hidden_size, self.num_groups_hidden, ...)
# Same num_groups_hidden (32 with G=4)

self.z_mem_baseline = nnx.Param(jnp.zeros(target_hidden_size))  # shape (128,)
# Memory signal also gets per-neuron baselines + grouped dynamics
```

#### 2.4.2 Grouping Size Impact

| `grouping_size` (G) | `num_groups_hidden` | Neurons per group | Granularity |
|---------------------|--------------------|--------------------|-------------|
| 1 | 128 | 1 | Per-neuron (finest) |
| 4 | 32 | 4 | Fine-grained |
| 40 | 4 | 32–40 | Coarse |
| 64 | 2 | 64 | Very coarse |
| 128 | 1 | 128 | Global (single scalar for all) |

**Current config**: `grouping_size=4` → **32 groups** for all heads (unimodal, multimodal, memory). Each group of 4 neurons shares one dynamic signal. This is fine-grained enough that the modulator can selectively gate small clusters of neurons without wholesale suppression.

**Design history**: Earlier configurations used `grouping_size=64` (2 groups) or `grouping_size=40` (4 groups), which enabled wholesale feature suppression — a contributor to the feature collapse observed in training (see NMN_PERFORMANCE_DIAGNOSIS.md §5.4). The current `G=4` significantly mitigates this risk.

### 2.5 Temperature Computation

```python
# neuromodulator.py:164-165
z_act_raw = self.head_action(h_mod_new)          # Linear(16 → 1)
temperature = clip(softplus(z_act_raw) + 0.5, 0.5, 3.0)
```

- `softplus` ensures positivity
- `+0.5` offset prevents temperature from collapsing to near-zero at init
- Clip bounds: `[0.5, 3.0]` — a 6× range. Temperature of 3.0 flattens action logits moderately; temperature of 0.5 sharpens them modestly.

---

## 3. Injection Points

### 3.1 Injection A — Perceptual Modulation (Two Modulation Types)

**Location**: `ObservationEncoder.forward_with_modulation()` (lines 121–151)

The implementation supports two modulation types selected via `modulation.type` in config. Both apply the same two-phase structure (unimodal → multimodal) but differ in **where** and **how** the modulator signal enters the activation function. The choice determines whether the modulator controls only the **magnitude** of features (Multiplicative) or both **magnitude and threshold** (PreActivation).

#### 3.1.1 Multiplicative Mode (Ben-Iwhiwhu style)

**Config**: `modulation.type: "Multiplicative"`
**Reference**: Ben-Iwhiwhu et al. (2022) — multiplicative masking

**Equation**: `output = relu(Wx + b) * sigmoid(z)`

In this mode, the modulator applies a **post-linear, pre-final-ReLU sigmoid gate** to each encoding layer's output. The gate value is always in (0, 1), meaning the modulator can only **attenuate** features — never amplify above their unmodulated magnitude.

**Worked example** — Suppose a single olfaction neuron's linear output is `Wx + b = 3.5`:

| Modulator state (z) | gamma = sigmoid(z) | Computation | Output |
|---|---|---|---|
| z = 3.0 (init) | 0.95 | relu(3.5) × 0.95 | **3.33** (5% attenuated) |
| z = 5.0 (gate open) | 0.993 | relu(3.5) × 0.993 | **3.48** (near pass-through) |
| z = 0.0 (neutral) | 0.50 | relu(3.5) × 0.50 | **1.75** (halved) |
| z = -5.0 (suppressed) | 0.007 | relu(3.5) × 0.007 | **0.024** (nearly silenced) |
| z = +10.0 (max open) | 0.99995 | relu(3.5) × 0.99995 | **3.4998** (ceiling — cannot exceed 3.5) |

The output is always ≤ the unmodulated value (`relu(3.5) = 3.5`). No z value can produce output > 3.5. This is the fundamental asymmetry: suppression is easy, amplification is impossible.

**Modulator heads constructed**: Only gamma heads (`head_unimodal`, `head_multimodal`). The additive heads (`head_unimodal_add`, `head_multimodal_add`) are **not constructed** — the `z_*_add` fields in `ModulatorOutput` are filled with zeros.

**Phase 1 (Unimodal) — spatial gating broadcast over all modalities**:
```python
# recurrent_ppo_network.py:139-144
encoded_all = self.unimodal_grouped(x_padded)              # GroupedMLP: (batch, 9, 128)
gamma1 = sigmoid(mod_output.z_unimodal)                    # shape (batch, 128)
beta1 = mod_output.z_unimodal_add                          # always zeros in Multiplicative mode
# Broadcast (batch, 1, 128) × (batch, 9, 128) — same gate pattern for all modalities
encoded_all = relu(encoded_all * gamma1[..., None, :] + beta1[..., None, :])
```

The modulation signal `(batch, 128)` broadcasts over the 9 modalities via `[..., None, :]`. All modalities receive the same spatial gate pattern — neurons at position `i` across all 9 modalities share the same gate value `sigmoid(z_unimodal[i])`.

**Phase 2 (Multimodal) — fusion hub gating**:
```python
# recurrent_ppo_network.py:146-151
mm_latent = self.multimodal_hub(mm_in)                     # MLP: 1152 → [128,128] → 128
gamma2 = sigmoid(mod_output.z_multimodal)                  # shape (batch, 128)
beta2 = mod_output.z_multimodal_add                        # always zeros
return relu(mm_latent * gamma2 + beta2)
```

The multimodal hub fuses all sensor outputs. Its 128-dim output is then element-wise gated by the grouped multimodal signal (32 groups → repeated to 128 dims).

**Behavior summary**: The Multiplicative modulator acts as a **soft binary mask**. It can silence features (gamma → 0) or pass them through nearly unchanged (gamma → 1), but cannot boost them. This is computationally simple and stable, but limits expressiveness — the modulator cannot learn "amplify nociceptive signals after injury," only "suppress non-nociceptive signals."

#### 3.1.2 PreActivation Mode (Ferguson & Cardin style)

**Config**: `modulation.type: "PreActivation"`
**Reference**: Ferguson & Cardin (2020) — VIP-SST disinhibitory gating; Shine et al. (2021) — energy landscape flattening

**Equation**: `output = relu(gamma * (Wx + b) + beta)`

In this mode, the modulator applies **both a multiplicative gain (gamma) and an additive threshold shift (beta)** inside the activation function, before the ReLU. This provides two independent degrees of freedom per layer:

- **gamma = sigmoid(z_percept)** — controls the **slope/gain** of the input-output mapping. High gamma amplifies the dynamic range of the linear output; low gamma compresses it toward zero. Inspired by Shine et al.'s neural gain model where noradrenaline rescales the energy landscape.

- **beta = z_percept_add** — controls the **threshold/bias** of the ReLU activation. Positive beta shifts the pre-activation upward → more neurons pass the ReLU threshold (disinhibition, lowering the activation barrier). Negative beta shifts downward → fewer neurons activate (inhibition, raising the threshold to filter weak/noisy signals). Inspired by Ferguson & Cardin's VIP→SST disinhibitory circuit where VIP interneurons lift inhibitory gates to allow sensory propagation.

**Worked example** — Same olfaction neuron with `Wx + b = 3.5`, now with both gamma and beta:

| Scenario | z_gamma | gamma | beta | Computation: relu(3.5 × gamma + beta) | Output | Interpretation |
|---|---|---|---|---|---|---|
| Init state | 3.0 | 0.95 | 0.0 | relu(3.5 × 0.95 + 0.0) = relu(3.33) | **3.33** | Same as Multiplicative at init |
| Disinhibited (danger nearby) | 3.0 | 0.95 | +2.0 | relu(3.5 × 0.95 + 2.0) = relu(5.33) | **5.33** | **Exceeds unmodulated 3.5** — beta lifts output above what Multiplicative can achieve |
| Strongly inhibited (safe area) | -2.0 | 0.12 | -1.0 | relu(3.5 × 0.12 - 1.0) = relu(-0.58) | **0.0** | Signal killed — compressed by low gamma AND pushed below ReLU threshold by negative beta |
| Selective (only strong signals) | 3.0 | 0.95 | -2.5 | relu(3.5 × 0.95 - 2.5) = relu(0.83) | **0.83** | Weak signals would fail: relu(1.0 × 0.95 - 2.5) = relu(-1.55) = 0 |
| Noise gate (weak signal) | 3.0 | 0.95 | -2.5 | relu(**0.5** × 0.95 - 2.5) = relu(-2.03) | **0.0** | Weak input (0.5) doesn't survive the raised threshold |

Key differences from Multiplicative visible in these examples:
- **Row 2 (Disinhibited)**: Output = 5.33 > 3.5 — positive beta achieves **effective amplification** that Multiplicative cannot do. The modulator can learn "boost olfaction sensitivity after injury."
- **Row 3 (Inhibited)**: gamma compresses the signal AND beta pushes it below zero. Two knobs working together produce stronger suppression than gamma alone.
- **Rows 4–5 (Selective)**: With `gamma=0.95, beta=-2.5`, the effective ReLU threshold shifts from 0 to `2.5/0.95 ≈ 2.63` in input space. Only inputs > 2.63 activate the neuron. This is **selectivity sharpening** — the modulator filters weak/noisy signals while preserving strong ones. Multiplicative mode cannot do this; it attenuates all signals equally regardless of magnitude.

**Modulator heads constructed**: Both gamma heads AND beta heads (`head_unimodal`, `head_unimodal_add`, `head_multimodal`, `head_multimodal_add`). Additional learned baselines are also created (`z_unimodal_add_baseline`, `z_hidden_add_baseline`).

**Phase 1 (Unimodal)**:
```python
# recurrent_ppo_network.py:139-144
encoded_all = self.unimodal_grouped(x_padded)              # GroupedMLP raw output (pre-activation)
gamma1 = sigmoid(mod_output.z_unimodal)                    # gain ∈ (0, 1), shape (batch, 128)
beta1 = mod_output.z_unimodal_add                          # threshold shift, shape (batch, 128)
# Broadcast (batch, 1, 128) × (batch, 9, 128)
encoded_all = relu(encoded_all * gamma1[..., None, :] + beta1[..., None, :])
```

Here gamma and beta work together, broadcast uniformly across all modalities:
- **High gamma + positive beta**: Strong signal amplification + lowered threshold → aggressive feature detection (disinhibited state)
- **Low gamma + negative beta**: Compressed signal + raised threshold → strict noise filtering (inhibited state)
- **High gamma + negative beta**: Only strong signals survive the raised threshold → sharpened selectivity

**Phase 2 (Multimodal)**:
```python
# recurrent_ppo_network.py:146-151
mm_latent = self.multimodal_hub(mm_in)                     # Hub raw output (pre-activation)
gamma2 = sigmoid(mod_output.z_multimodal)                  # gain, shape (batch, 128)
beta2 = mod_output.z_multimodal_add                        # threshold shift, shape (batch, 128)
return relu(mm_latent * gamma2 + beta2)
```

Same two-parameter control applied to the fusion layer.

**Behavior summary**: PreActivation provides richer modulation — the modulator can independently control both **how much signal gets through** (gamma) and **what activation threshold signals must exceed** (beta). This maps more closely to biological disinhibitory circuits where VIP interneurons and SST interneurons independently regulate gain and threshold. However, the extra expressiveness also means more parameters to learn and a larger space of degenerate solutions.

#### 3.1.3 Side-by-Side Comparison

| Aspect | Multiplicative | PreActivation |
|--------|---------------|---------------|
| **Equation** | `relu(Wx * sigmoid(z))` | `relu(sigmoid(z_gamma) * Wx + z_beta)` |
| **Degrees of freedom** | 1 per layer (gamma only) | 2 per layer (gamma + beta) |
| **Gain range** | (0, 1) — attenuation only | (0, 1) — attenuation only* |
| **Threshold control** | None — ReLU threshold fixed at 0 | Yes — beta shifts the ReLU boundary |
| **Can amplify features?** | No (sigmoid ≤ 1) | Indirectly — positive beta can push sub-threshold features above ReLU |
| **Can sharpen selectivity?** | No | Yes — low gamma + negative beta raises the bar for activation |
| **Biological analog** | Binary masking (Ben-Iwhiwhu) | VIP-SST disinhibition (Ferguson & Cardin) + neural gain (Shine) |
| **Extra modulator heads** | 0 | 2 (`head_unimodal_add`, `head_multimodal_add`) |
| **Extra baselines** | 0 | 2 (`z_unimodal_add_baseline`, `z_hidden_add_baseline`) |
| **Config init** | `percept_bias_init: 3.0` | `percept_bias_init: 3.0`, `percept_add_bias_init: 0.0` |
| **Init behavior** | sigmoid(3.0) ≈ 0.95 gain, no bias | sigmoid(3.0) ≈ 0.95 gain, zero bias → same as Multiplicative at init |
| **Risk profile** | Simpler, biased toward suppression | More expressive, but larger degenerate solution space |

*\*Note: In both modes, gamma uses `sigmoid()` which is bounded to (0, 1). Neither mode supports gain > 1 (true amplification). The PreActivation mode compensates partially through the beta term — a positive beta can push features above the ReLU threshold even when gamma attenuates them, achieving an effective amplification of the number of active neurons (though not their magnitude).*

#### 3.1.4 Flat Encoder Fallback

When `encoding_mode: "flat"` (non-hierarchical), both modes collapse to a single-phase modulation using the full `z_unimodal` signal:

```python
# recurrent_ppo_network.py:123-130
if self.mode != 'hierarchical':
    x_proj = self.monolith(x)
    if modulation_type == "PreActivation":
        gamma = sigmoid(z_unimodal)          # (batch, 128) — full signal
        beta = z_unimodal_add                # (batch, 128)
        return relu(x_proj * gamma + beta)
    else:
        return relu(x_proj) * sigmoid(z_unimodal)
```

The `z_unimodal` signal is now `(batch, 128)` — the same shape as `x_proj` — so it applies element-wise. The multimodal phase is skipped entirely. The flat encoder path exists for backward compatibility but should not be used with modulation.

#### 3.1.5 Key Implementation Detail: Unified Code Path

Both modulation types share the **same forward code** in `forward_with_modulation()` (lines 139–151). The differentiation happens upstream in `NeuromodulatorRNN.__init__()`:
- In Multiplicative mode, the `head_*_add` layers and `z_*_add_baseline` params are **never constructed**. The `_get_signal()` helper returns `jnp.zeros_like(sig)` for the additive component.
- In PreActivation mode, both heads exist and produce learned signals.

The encoder code always computes `relu(encoded * gamma + beta)` — it's just that `beta = 0` in Multiplicative mode. This means switching modes requires only a config change and model re-initialization; no code branches exist in the encoder's hot path.

#### 3.1.6 Key Implementation Detail: Unified Signal Computation

The `_get_signal()` helper in `NeuromodulatorRNN.__call__()` uses a single code path for all heads — unimodal, multimodal, and memory (lines 139–149). There is no `is_unimodal` flag or branching. All heads produce grouped outputs that are repeat-and-sliced to the target hidden dimension:

```python
# neuromodulator.py:139-149
def _get_signal(head, baseline, head_add=None, baseline_add=None):
    raw = head(h_mod_new)
    sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
    sig = baseline.value + sig

    if head_add is not None:
        raw_add = head_add(h_mod_new)
        sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
        sig_add = baseline_add.value + sig_add
        return sig, sig_add
    return sig, jnp.zeros_like(sig)
```

### 3.2 Injection B — Memory Gate-Bias

**Location**: `ModulatedGRUCell.__call__()` (lines 42–78)

```python
u_pre = self.W_iu(x) + self.W_hu(h)     # Standard update gate pre-activation
if gate_bias is not None:
    u_pre = u_pre + gate_bias             # Add modulator signal
u = sigmoid(u_pre)                        # Update gate
h_new = (1 - u) * h + u * n              # GRU update
```

- **Negative `z_memory`** → sigmoid shifts left → lower `u` → **retain old hidden state** (memory persistence)
- **Positive `z_memory`** → sigmoid shifts right → higher `u` → **incorporate new candidate** (faster forgetting)

The gate-bias is additive on the pre-sigmoid activation — a lightweight intervention that preserves the GRU's internal dynamics while shifting its operating point.

**Bounded output**: The `z_memory` signal is clamped before injection:
```python
# neuromodulator.py:161
z_mem = jnp.clip(z_mem, self.memory_clip[0], self.memory_clip[1])
```
With `memory_clip: [-2.0, 2.0]` (config), this prevents the gate-bias from drifting to extreme values that would freeze or force-reset the GRU update gate. At the bounds: `sigmoid(pre + 2.0)` shifts the gate but cannot fully override it; `sigmoid(pre - 2.0)` shifts toward retention but doesn't freeze completely.

### 3.3 Injection C — Temperature Scaling

**Location**: `ActorCriticRNN.__call__()` (line 278)

```python
logits = logits / mod_output.temperature
```

Applied after actor head, before action sampling. Temperature τ controls exploration:
- τ < 1 → sharper logits → deterministic behavior
- τ > 1 → flatter logits → stochastic exploration

Current bounds `[0.5, 3.0]` give a 6× range — enough for meaningful exploration control without allowing the modulator to fully override the learned policy.

---

## 4. Data Flow Through the Full Network

```
obs (raw)
  │
  ├── symlog compression: sign(x) * log(|x| + 1)
  │
  ├──► NeuromodulatorRNN
  │      GRU(obs_dim=30, hidden=16): obs → h_mod_new
  │      Heads: h_mod_new → {z_uni(128), z_multi(128), z_mem(128), temperature(1)}
  │
  ├──► ObservationEncoder (with modulation)
  │      Phase 1: GroupedMLP(9 modalities, 30→128→128→128) × sigmoid(z_uni)
  │               z_uni broadcasts (batch, 1, 128) × (batch, 9, 128)
  │      Phase 2: MLP(1152→128→128→128) × sigmoid(z_multi)
  │      Output: x_proj (batch, 128)
  │
  ├──► ModulatedGRUCell
  │      Input: x_proj (128), h_prev (128), gate_bias=clip(z_mem, -2, 2) (128)
  │      Output: h_new (128)
  │
  ├──► Actor Head
  │      FC(128→128, relu) → FC(128→action_dim)
  │      logits = logits / temperature   ← Injection C
  │
  └──► Critic Head
         FC(128→128, relu) → FC(128→1)
         value (unmodulated)
```

**Hidden state composition**:
- With modulation: `h = (task_h, mod_h)` where `task_h: (128,)`, `mod_h: (16,)`
- Without modulation: `h = task_h: (128,)` only

---

## 5. Parameter Count Analysis

### 5.1 Modulator Parameters

With `grouping_size=4`, `mod_hidden_size=16`, `hidden_size=128`:

| Component | Shape | Params |
|-----------|-------|--------|
| GRU input-to-hidden (3 gates) | (30, 16) × 3 | 1,440 |
| GRU hidden-to-hidden (3 gates) | (16, 16) × 3 | 768 |
| GRU biases (6 total) | 16 × 6 | 96 |
| `head_unimodal` | (16, 32) + 32 | 544 |
| `head_multimodal` | (16, 32) + 32 | 544 |
| `head_memory` | (16, 32) + 32 | 544 |
| `head_action` | (16, 1) + 1 | 17 |
| Baselines (z_unimodal, z_hidden, z_mem) | 128 + 128 + 128 | 384 |
| **Total Modulator (Multiplicative)** | | **~4,337** |

PreActivation mode adds:
| Component | Shape | Params |
|-----------|-------|--------|
| `head_unimodal_add` | (16, 32) + 32 | 544 |
| `head_multimodal_add` | (16, 32) + 32 | 544 |
| Baselines (z_unimodal_add, z_hidden_add) | 128 + 128 | 256 |
| **Total Modulator (PreActivation)** | | **~5,681** |

### 5.2 Task Network Parameters (for context)

| Component | Params (approx) |
|-----------|----------------|
| Observation Encoder (9 grouped MLPs + multimodal hub) | ~320,000 |
| ModulatedGRUCell (6 weight matrices) | ~65,000 |
| Actor Head (2 layers) | ~33,000 |
| Critic Head (2 layers) | ~33,000 |
| **Total Task Network** | **~451,000** |

**Modulator is ~1.0% of total parameters** (Multiplicative) — lightweight. The overhead comes from the extra forward pass and sigmoid computations, not from parameter count.

---

## 6. Observations and Concerns

### 6.1 No Input Projection on the Modulator GRU

The modulator GRU receives the full raw observation (after symlog) directly:
```python
self.gru = nnx.GRUCell(obs_dim, mod_hidden_size, rngs=rngs)  # (30 → 16)
```

This is a 30→16 compression in a single GRU step. The DreamerV3 variant adds `proj_obs = Linear(obs_dim, mod_hidden_size)` with ReLU before the GRU, giving it a learned feature extraction stage. The RecurrentPPO variant lacks this.

**Concern**: The GRU must simultaneously compress 30 observation dims and maintain temporal state in only 16 units. An input projection would decouple feature extraction from temporal dynamics.

### 6.2 Modulator Sees Symlog-Compressed Observations

The symlog compression `sign(x) * log(|x| + 1)` is applied at line 254, before the modulator receives `x`. This compresses high-magnitude modalities (olfaction ~0–40, visual ~0–13) to ~0–3.7 range.

This is appropriate — it prevents olfaction from dominating the modulator's GRU dynamics. However, it also compresses interoceptive signals (injury, nutrition, satiation already in [0,1]) further: `symlog(0.5) = 0.405`. The modulator's ability to distinguish fine-grained interoceptive state may be reduced.

### 6.3 Sigmoid Gain Gates Are Bounded [0, 1]

In Multiplicative mode, perceptual modulation uses:
```python
gamma = sigmoid(z_unimodal)  # always in (0, 1)
encoded = relu(raw * gamma)
```

This means the modulator can only **attenuate** features — it cannot amplify them above their unmodulated value. The theoretical motivation for gain control (Shine et al., 2021) includes **amplification** (gain > 1) as well as suppression. The sigmoid ceiling at 1.0 prevents the modulator from ever increasing sensory gain.

**Consequence**: The modulator cannot learn "after injury, amplify nociceptive signals" — only "suppress non-nociceptive signals." This asymmetry may bias the modulator toward suppression as the default strategy, which is exactly what was observed in training.

### 6.4 Critic Head Is Not Modulated

The critic receives the same GRU hidden state as the actor but its output is **not temperature-scaled** or otherwise modulated:
```python
c_h = relu(self.critic_fc1(x_h))
value = self.critic_fc2(c_h)    # No modulation applied
```

This means the value function cannot benefit from modulation-dependent context. If the modulator is learning "this is a dangerous state," the critic has no direct signal for this — it must infer it from the (already modulated) hidden state `x_h`, which passes through the modulated GRU and modulated encoder.

This is likely acceptable since indirect modulation via the shared hidden state should suffice. But it's worth noting that the critic and actor operate under different modulation regimes (actor has temperature scaling, critic does not).

### 6.5 Shared Optimizer / No Learning Rate Separation

Both task network and modulator parameters are trained with a single Adam optimizer at `lr=0.0005`. The NEUROMODULATION_ALGORITHM.md §4 recommends separate learning rates, but this is not implemented.

**Consequence**: The modulator adapts at the same speed as the task network. Since the modulator has far fewer parameters (~4,337 vs ~451,000), it can converge much faster in practice. This allows the modulator to find and exploit degenerate shortcuts before the task network has learned useful features to modulate.

---

## 7. Comparison: RecurrentPPO vs DreamerV3 NMN

| Aspect | RecurrentPPO (`NeuromodulatorRNN`) | DreamerV3 (`DreamerNeuromodulatorRNN`) |
|--------|-----------------------------------|---------------------------------------|
| GRU input | Raw obs (no projection) | `relu(proj_obs(obs))` — learned projection |
| Dual modes | No — single forward path | Yes — `forward_obs()` vs `forward_imagine()` |
| Injection C | Temperature head (τ ∈ [0.5, 3.0]) | Reward head (sigmoid-bounded `z_reward`) |
| Imagination mode | N/A | Zeros perceptual heads, keeps memory + reward |
| Input to GRU | `obs_dim` directly | `mod_hidden_size` (after projection) |
| Lazy init | No | `set_imagine_input_dim()` for imagination projection |
| Unimodal grouping | `num_groups_hidden` (same as multimodal) | `num_groups_percept` (same as multimodal) |
| Signal computation | Unified `_get_signal()` — no branching | Unified `_get_signal()` — `is_percept` flag for imagination zeroing only |
| Unimodal baseline shape | `(target_hidden_size,)` | `(embed_dim,)` |
| z_memory clamp | `memory_clip: [-2.0, 2.0]` | Not clamped (TODO) |

The DreamerV3 variant is architecturally cleaner: the input projection decouples observation dimensionality from GRU state size, and dual modes handle the distinction between real observation and imagined rollouts.

Both variants use the **same unified grouping** for unimodal and multimodal heads — the `is_unimodal` branching was eliminated in the unification refactor (see [UNIFY_UNIMODAL_GROUPING.md](docs/UNIFY_UNIMODAL_GROUPING.md)).
