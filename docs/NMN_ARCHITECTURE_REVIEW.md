# NMN Architecture Review: RecurrentPPO Neuromodulatory Network

> **Reviewer**: Claude | **Date**: 2026-03-06
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
| `NeuromodulatorRNN` | `neuromodulator.py` | 41–182 | Recurrent core + branched output heads |
| `ModulatorOutput` | `neuromodulator.py` | 31–38 | NamedTuple carrying all modulation signals |
| `ModulatedGRUCell` | `modulated_gru_cell.py` | 17–78 | Standard GRU with external update gate bias |
| `ObservationEncoder` | `recurrent_ppo_network.py` | 72–151 | Hierarchical encoder with modulated forward path |
| `ActorCriticRNN` | `recurrent_ppo_network.py` | 154–319 | Main network integrating all injections |

---

## 2. NeuromodulatorRNN Internals

### 2.1 Recurrent Core

- **Single GRU**: `GRUCell(obs_dim, mod_hidden_size)` — receives the raw observation directly (after symlog compression in `ActorCriticRNN`).
- **Hidden size**: 16 (config). This is deliberately small — intended to represent slowly-evolving affective tone rather than rich state representation.
- **No input projection**: The raw observation (dimension = `obs_dim`, typically ~30) feeds directly into the GRU. This means the GRU input-to-hidden weights are shape `(obs_dim, 3 * mod_hidden_size)` = `(30, 48)` — a significant compression.

### 2.2 Branched Output Heads

Six linear heads project from the shared GRU hidden state:

| Head | Output Shape | Activation | Init Bias | Role |
|------|-------------|------------|-----------|------|
| `head_unimodal` | `(num_modalities,)` = 9 | — (sigmoid applied at injection) | 2.0 | Per-sensor perceptual gain (gamma) |
| `head_unimodal_add` | `(num_modalities,)` = 9 | — | 0.0 | Per-sensor bias (beta, PreActivation only) |
| `head_multimodal` | `(ceil(128/40),)` = 4 | — (sigmoid applied at injection) | 2.0 | Multimodal fusion gain |
| `head_multimodal_add` | `(ceil(128/40),)` = 4 | — | 0.0 | Multimodal fusion bias (PreActivation only) |
| `head_memory` | `(ceil(128/40),)` = 4 | — (raw added to GRU gate) | 0.0 | GRU update gate bias |
| `head_action` | `(1,)` | softplus + offset + clip | default | Temperature scalar |

**Note on Multiplicative mode**: `head_unimodal_add` and `head_multimodal_add` are **not constructed** — the corresponding `z_*_add` outputs are filled with zeros. Only the gamma (gain) heads are active.

### 2.3 Per-Neuron Learned Baselines

Each head's output is added to a **learned baseline parameter** before use:

```python
sig = baseline.value + raw   # for unimodal (per-group)
sig = baseline.value + repeat(raw, G)[..., :target_hidden_size]  # for multimodal/memory
```

- `z_unimodal_baseline`: shape `(9,)`, init zeros
- `z_hidden_baseline`: shape `(128,)`, init zeros
- `z_mem_baseline`: shape `(128,)`, init zeros

These baselines allow each neuron/group to learn its own "resting" modulation level independently of the GRU dynamics. At initialization, baseline + head_bias = 0 + 2.0 = 2.0 for perceptual heads → sigmoid(2.0) ≈ 0.88 initial gain.

### 2.4 Spatial Grouping

With `grouping_size=40` and `hidden_size=128`:
- **Unimodal**: 1 signal per modality (9 groups). Each group gates all output neurons of that modality's encoder.
- **Multimodal/Memory**: `ceil(128/40) = 4` groups, each repeated 40 times to cover 128 hidden dims. This means 32-neuron blocks share the same modulation signal.

**Implication**: The modulator can shut off an entire 40-neuron block with a single negative value. This coarse grouping was identified as a contributor to the feature collapse observed in training (see NMN_PERFORMANCE_DIAGNOSIS.md §5.4).

### 2.5 Temperature Computation

```python
z_act_raw = self.head_action(h_mod_new)          # Linear(16 → 1)
temperature = clip(softplus(z_act_raw) + 0.5, 0.1, 10.0)
```

- `softplus` ensures positivity
- `+0.5` offset prevents temperature from collapsing to near-zero at init
- Clip bounds: `[0.1, 10.0]` — extremely wide. Temperature of 10.0 makes action logits near-uniform.

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
| z = 2.0 (init) | 0.88 | relu(3.5) × 0.88 | **3.08** (12% attenuated) |
| z = 5.0 (gate open) | 0.993 | relu(3.5) × 0.993 | **3.48** (near pass-through) |
| z = 0.0 (neutral) | 0.50 | relu(3.5) × 0.50 | **1.75** (halved) |
| z = -5.0 (suppressed) | 0.007 | relu(3.5) × 0.007 | **0.024** (nearly silenced) |
| z = +10.0 (max open) | 0.99995 | relu(3.5) × 0.99995 | **3.4998** (ceiling — cannot exceed 3.5) |

The output is always ≤ the unmodulated value (`relu(3.5) = 3.5`). No z value can produce output > 3.5. This is the fundamental asymmetry: suppression is easy, amplification is impossible.

**Modulator heads constructed**: Only gamma heads (`head_unimodal`, `head_multimodal`). The additive heads (`head_unimodal_add`, `head_multimodal_add`) are **not constructed** — the `z_*_add` fields in `ModulatorOutput` are filled with zeros.

**Additional parameters over baseline**: ~221 (gamma heads only, no beta heads or beta baselines).

**Phase 1 (Unimodal) — per-modality gating**:
```python
encoded_all = self.unimodal_grouped(x_padded)            # GroupedMLP: (batch, 9, max_in) → (batch, 9, 128)
gamma1 = sigmoid(mod_output.z_unimodal)                  # shape (batch, 9), range (0, 1)
beta1 = mod_output.z_unimodal_add                        # always zeros in Multiplicative mode
encoded_all = relu(encoded_all * gamma1[..., None] + beta1[..., None])
# Effective: relu(encoded_all * gamma1[..., None])
```

Each of the 9 sensor modalities gets a single scalar gate ∈ (0, 1). This gate uniformly scales all 128 output neurons of that modality's encoder. At init: `sigmoid(2.0) ≈ 0.88` — modest initial attenuation.

**Phase 2 (Multimodal) — fusion hub gating**:
```python
mm_latent = self.multimodal_hub(mm_in)                   # MLP: 1152 → [128,128] → 128
gamma2 = sigmoid(mod_output.z_multimodal)                # shape (batch, 128), range (0, 1)
beta2 = mod_output.z_multimodal_add                      # always zeros
return relu(mm_latent * gamma2 + beta2)
# Effective: relu(mm_latent * gamma2)
```

The multimodal hub fuses all sensor outputs. Its 128-dim output is then element-wise gated by the grouped multimodal signal (4 groups → repeated to 128 dims).

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
| Init state | 2.0 | 0.88 | 0.0 | relu(3.5 × 0.88 + 0.0) = relu(3.08) | **3.08** | Same as Multiplicative at init |
| Disinhibited (danger nearby) | 2.0 | 0.88 | +2.0 | relu(3.5 × 0.88 + 2.0) = relu(5.08) | **5.08** | **Exceeds unmodulated 3.5** — beta lifts output above what Multiplicative can achieve |
| Strongly inhibited (safe area) | -2.0 | 0.12 | -1.0 | relu(3.5 × 0.12 - 1.0) = relu(-0.58) | **0.0** | Signal killed — compressed by low gamma AND pushed below ReLU threshold by negative beta |
| Selective (only strong signals) | 2.0 | 0.88 | -2.5 | relu(3.5 × 0.88 - 2.5) = relu(0.58) | **0.58** | Weak signals would fail: relu(1.0 × 0.88 - 2.5) = relu(-1.62) = 0 |
| Noise gate (weak signal) | 2.0 | 0.88 | -2.5 | relu(**0.5** × 0.88 - 2.5) = relu(-2.06) | **0.0** | Weak input (0.5) doesn't survive the raised threshold |

Key differences from Multiplicative visible in these examples:
- **Row 2 (Disinhibited)**: Output = 5.08 > 3.5 — positive beta achieves **effective amplification** that Multiplicative cannot do. The modulator can learn "boost olfaction sensitivity after injury."
- **Row 3 (Inhibited)**: gamma compresses the signal AND beta pushes it below zero. Two knobs working together produce stronger suppression than gamma alone.
- **Rows 4–5 (Selective)**: With `gamma=0.88, beta=-2.5`, the effective ReLU threshold shifts from 0 to `2.5/0.88 ≈ 2.84` in input space. Only inputs > 2.84 activate the neuron. This is **selectivity sharpening** — the modulator filters weak/noisy signals while preserving strong ones. Multiplicative mode cannot do this; it attenuates all signals equally regardless of magnitude.

**Modulator heads constructed**: Both gamma heads AND beta heads (`head_unimodal`, `head_unimodal_add`, `head_multimodal`, `head_multimodal_add`). Additional learned baselines are also created (`z_unimodal_add_baseline`, `z_hidden_add_baseline`).

**Additional parameters over Multiplicative**: ~230 extra (beta heads + beta baselines), totaling ~451 modulator-specific parameters for the perceptual heads.

**Phase 1 (Unimodal)**:
```python
encoded_all = self.unimodal_grouped(x_padded)            # GroupedMLP raw output (pre-activation)
gamma1 = sigmoid(mod_output.z_unimodal)                  # gain ∈ (0, 1)
beta1 = mod_output.z_unimodal_add                        # threshold shift ∈ (-∞, +∞)
encoded_all = relu(encoded_all * gamma1[..., None] + beta1[..., None])
```

Here gamma and beta work together:
- **High gamma + positive beta**: Strong signal amplification + lowered threshold → aggressive feature detection (disinhibited state)
- **Low gamma + negative beta**: Compressed signal + raised threshold → strict noise filtering (inhibited state)
- **High gamma + negative beta**: Only strong signals survive the raised threshold → sharpened selectivity

**Phase 2 (Multimodal)**:
```python
mm_latent = self.multimodal_hub(mm_in)                   # Hub raw output (pre-activation)
gamma2 = sigmoid(mod_output.z_multimodal)                # gain
beta2 = mod_output.z_multimodal_add                      # threshold shift
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
| **Extra params (approx)** | 0 | ~230 |
| **Config init** | `percept_bias_init: 2.0` | `percept_bias_init: 2.0`, `percept_add_bias_init: 0.0` |
| **Init behavior** | sigmoid(2.0) ≈ 0.88 gain, no bias | sigmoid(2.0) ≈ 0.88 gain, zero bias → same as Multiplicative at init |
| **Risk profile** | Simpler, biased toward suppression | More expressive, but larger degenerate solution space |

*\*Note: In both modes, gamma uses `sigmoid()` which is bounded to (0, 1). Neither mode supports gain > 1 (true amplification). The PreActivation mode compensates partially through the beta term — a positive beta can push features above the ReLU threshold even when gamma attenuates them, achieving an effective amplification of the number of active neurons (though not their magnitude).*

#### 3.1.4 Flat Encoder Fallback

When `encoding_mode: "flat"` (non-hierarchical), both modes collapse to a single-phase modulation using only the first element of `z_unimodal`:

```python
# Multiplicative flat fallback:
return relu(monolith(x)) * sigmoid(z_unimodal[..., 0:1])

# PreActivation flat fallback:
gamma = sigmoid(z_unimodal[..., 0:1])
beta = z_unimodal_add[..., 0:1]
return relu(monolith(x) * gamma + beta)
```

This is a degenerate case — only 1 of 9 unimodal groups is used, and the multimodal phase is skipped entirely. The flat encoder path exists for backward compatibility but should not be used with modulation.

#### 3.1.5 Key Implementation Detail: Unified Code Path

Both modulation types share the **same forward code** in `forward_with_modulation()` (lines 139–151). The differentiation happens upstream in `NeuromodulatorRNN.__init__()`:
- In Multiplicative mode, the `head_*_add` layers and `z_*_add_baseline` params are **never constructed**. The `_get_signal()` helper returns `jnp.zeros_like(sig)` for the additive component.
- In PreActivation mode, both heads exist and produce learned signals.

The encoder code always computes `relu(encoded * gamma + beta)` — it's just that `beta = 0` in Multiplicative mode. This means switching modes requires only a config change and model re-initialization; no code branches exist in the encoder's hot path.

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

### 3.3 Injection C — Temperature Scaling

**Location**: `ActorCriticRNN.__call__()` (line 276)

```python
logits = logits / mod_output.temperature
```

Applied after actor head, before action sampling. Temperature τ controls exploration:
- τ < 1 → sharper logits → deterministic behavior
- τ > 1 → flatter logits → stochastic exploration

---

## 4. Data Flow Through the Full Network

```
obs (raw)
  │
  ├── symlog compression: sign(x) * log(|x| + 1)
  │
  ├──► NeuromodulatorRNN
  │      GRU(obs_dim=30, hidden=16): obs → h_mod_new
  │      Heads: h_mod_new → {z_uni, z_multi, z_mem, temperature}
  │
  ├──► ObservationEncoder (with modulation)
  │      Phase 1: GroupedMLP(9 modalities, 30→128→128→128) × sigmoid(z_uni)
  │      Phase 2: MLP(1152→128→128→128) × sigmoid(z_multi)
  │      Output: x_proj (batch, 128)
  │
  ├──► ModulatedGRUCell
  │      Input: x_proj (128), h_prev (128), gate_bias=z_mem (128)
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

| Component | Shape | Params |
|-----------|-------|--------|
| GRU input-to-hidden (3 gates) | (30, 16) × 3 | 1,440 |
| GRU hidden-to-hidden (3 gates) | (16, 16) × 3 | 768 |
| GRU biases (6 total) | 16 × 6 | 96 |
| `head_unimodal` | (16, 9) + 9 | 153 |
| `head_multimodal` | (16, 4) + 4 | 68 |
| `head_memory` | (16, 4) + 4 | 68 |
| `head_action` | (16, 1) + 1 | 17 |
| Baselines (z_unimodal, z_hidden, z_mem) | 9 + 128 + 128 | 265 |
| **Total Modulator** | | **~2,875** |

### 5.2 Task Network Parameters (for context)

| Component | Params (approx) |
|-----------|----------------|
| Observation Encoder (9 grouped MLPs + multimodal hub) | ~320,000 |
| ModulatedGRUCell (6 weight matrices) | ~65,000 |
| Actor Head (2 layers) | ~33,000 |
| Critic Head (2 layers) | ~33,000 |
| **Total Task Network** | **~451,000** |

**Modulator is ~0.6% of total parameters** — very lightweight. The 24% wall-clock overhead comes from the extra forward pass and sigmoid computations, not from parameter count.

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

The symlog compression `sign(x) * log(|x| + 1)` is applied at line 252, before the modulator receives `x`. This compresses high-magnitude modalities (olfaction ~0–40, visual ~0–13) to ~0–3.7 range.

This is appropriate — it prevents olfaction from dominating the modulator's GRU dynamics. However, it also compresses interoceptive signals (injury, nutrition, satiation already in [0,1]) further: `symlog(0.5) = 0.405`. The modulator's ability to distinguish fine-grained interoceptive state may be reduced.

### 6.3 Sigmoid Gain Gates Are Bounded [0, 1]

In Multiplicative mode, perceptual modulation uses:
```python
gamma = sigmoid(z_unimodal)  # always in (0, 1)
encoded = relu(raw * gamma)
```

This means the modulator can only **attenuate** features — it cannot amplify them above their unmodulated value. The theoretical motivation for gain control (Shine et al., 2021) includes **amplification** (gain > 1) as well as suppression. The sigmoid ceiling at 1.0 prevents the modulator from ever increasing sensory gain.

**Consequence**: The modulator cannot learn "after injury, amplify nociceptive signals" — only "suppress non-nociceptive signals." This asymmetry may bias the modulator toward suppression as the default strategy, which is exactly what was observed in training.

### 6.4 Temperature Bounds Are Extremely Wide

Current config: `temp_clip: [0.1, 10.0]`

- At τ=10.0 with 5 actions: `softmax(logits/10)` → near-uniform (each action ≈ 20% ± tiny perturbation)
- At τ=0.1 with 5 actions: `softmax(logits/0.1)` → near-deterministic (winner-take-all)

The range spans **100x** from min to max. This gives the modulator enormous leverage over the policy — it can effectively disable the learned policy by setting temperature to 10.0. Training confirmed this: temperature mean reached 3.29 with max pinned at 10.0.

### 6.5 No Clamp on `z_memory`

The memory gate-bias `z_memory` has no bounds — it can drift to arbitrarily large negative values, which fully freezes the GRU update gate. At `z_memory = -6.5`, the update gate receives a constant -6.5 bias, making `sigmoid(anything - 6.5) ≈ 0` regardless of input.

This effectively converts the recurrent network into a feedforward one — a drastic architectural change that the modulator can impose unilaterally.

### 6.6 Critic Head Is Not Modulated

The critic receives the same GRU hidden state as the actor but its output is **not temperature-scaled** or otherwise modulated:
```python
c_h = relu(self.critic_fc1(x_h))
value = self.critic_fc2(c_h)    # No modulation applied
```

This means the value function cannot benefit from modulation-dependent context. If the modulator is learning "this is a dangerous state," the critic has no direct signal for this — it must infer it from the (already modulated) hidden state `x_h`, which passes through the modulated GRU and modulated encoder.

This is likely acceptable since indirect modulation via the shared hidden state should suffice. But it's worth noting that the critic and actor operate under different modulation regimes (actor has temperature scaling, critic does not).

### 6.7 Shared Optimizer / No Learning Rate Separation

Both task network and modulator parameters are trained with a single Adam optimizer at `lr=0.0005`. The NEUROMODULATION_ALGORITHM.md §4 recommends separate learning rates, but this is not implemented.

**Consequence**: The modulator adapts at the same speed as the task network. Since the modulator has far fewer parameters (~2,875 vs ~451,000), it can converge much faster in practice. This allows the modulator to find and exploit degenerate shortcuts before the task network has learned useful features to modulate.

### 6.8 Unimodal Modulation Applies Per-Group, Not Per-Neuron

The unimodal gamma has shape `(9,)` — one scalar per sensor modality. This scalar gates all 128 output neurons of that modality's encoder:

```python
encoded_all = relu(encoded_all * gamma1[..., None])  # gamma1: (9,), encoded_all: (9, 128)
```

This is an all-or-nothing gate per modality. The modulator cannot selectively suppress certain features within a modality while preserving others. For example, it cannot keep "food direction" from olfaction while suppressing "food distance" — it must gate the entire olfaction encoding uniformly.

---

## 7. Comparison: RecurrentPPO vs DreamerV3 NMN

| Aspect | RecurrentPPO (`NeuromodulatorRNN`) | DreamerV3 (`DreamerNeuromodulatorRNN`) |
|--------|-----------------------------------|---------------------------------------|
| GRU input | Raw obs (no projection) | `relu(proj_obs(obs))` — learned projection |
| Dual modes | No — single forward path | Yes — `forward_obs()` vs `forward_imagine()` |
| Injection C | Temperature head (τ ∈ [0.1, 10.0]) | Reward head (sigmoid-bounded `z_reward`) |
| Imagination mode | N/A | Zeros perceptual heads, keeps memory + reward |
| Input to GRU | `obs_dim` directly | `mod_hidden_size` (after projection) |
| Lazy init | No | `set_imagine_input_dim()` for imagination projection |

The DreamerV3 variant is architecturally cleaner: the input projection decouples observation dimensionality from GRU state size, and dual modes handle the distinction between real observation and imagined rollouts.

---

## 8. Summary of Architectural Risks

| Risk | Severity | Current Impact | Mitigation |
|------|----------|---------------|------------|
| Sigmoid gain ≤ 1 (no amplification) | **High** | Modulator can only suppress, biasing toward feature collapse | Use `2 * sigmoid(z)` or `softplus(z)` for gain ∈ (0, ∞) |
| Temperature bounds [0.1, 10.0] too wide | **High** | Modulator overrides policy (τ → 10.0) | Tighten to [0.5, 2.0] or [0.8, 1.5] |
| No `z_memory` clamp | **High** | GRU frozen (z_mem → -6.5) | Clamp to [-2, +2] |
| Coarse grouping (G=40) | **Medium** | 4 groups for 128 dims enables wholesale suppression | Reduce to G=1 (per-neuron) |
| No input projection on modulator GRU | **Medium** | 30→16 compression in single step | Add `Linear(obs_dim, mod_hidden_size) + relu` |
| Shared optimizer (no LR separation) | **Medium** | Modulator converges to degenerate shortcuts first | Use `optax.multi_transform` with 5–10× lower modulator LR |
| Unimodal gate is per-modality scalar | **Low** | Cannot do within-modality selective gating | Acceptable for current architecture |
| Critic not directly modulated | **Low** | Critic infers context from shared hidden state | Acceptable — indirect modulation sufficient |

---

## 9. Relationship to Known Training Pathologies

The architectural features above directly map to the pathological training outcomes documented in `NMN_PERFORMANCE_DIAGNOSIS.md`:

1. **Feature collapse** (gamma_body → -4, gamma_assoc → -15): Enabled by sigmoid ≤ 1 (suppression is easy, amplification impossible) + coarse grouping (one value kills 40 neurons) + no environmental pressure for adaptive modulation.

2. **Memory freeze** (z_memory → -6.5): Enabled by unbounded z_memory + no timescale separation. Once body-state features are suppressed, the GRU carries no useful info, and freezing it reduces value loss variance.

3. **Temperature inflation** (τ → 3.3, max pinned at 10.0): Enabled by excessively wide bounds. Compensatory response to frozen memory — the agent needs random exploration since its hidden state doesn't update.

4. **Stable degenerate equilibrium**: The modulator's tiny parameter count (~2,875) converges quickly to a local optimum that suppresses features + freezes memory + inflates temperature. The shared optimizer allows this to happen before the task network has learned features worth modulating.

All four pathologies are **architecturally enabled** — they are not bugs in the code but rather consequences of design choices that permit the modulator to find degenerate shortcuts. The fixes involve constraining the modulator's action space (bounded outputs, tighter temperature, per-neuron grouping) and enforcing timescale separation (lower learning rate).
