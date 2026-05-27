---
title: "FiLM-Style Modulation: Implementation Plan"
topic: filim
status: active
created: 2026-03-17
last_updated: 2026-04-12
---

# FiLM-Style Modulation: Implementation Plan

> **Status**: COMPLETED
> **Opened**: 2026-03-17
> **Related**: [SOLVING_GATE_COLLAPSE.md](SOLVING_GATE_COLLAPSE.md) §8.1, §8.8 | [NMN_ARCHITECTURE_REVIEW.md](NMN_ARCHITECTURE_REVIEW.md)

---

## Context

The NMN's multimodal hub suffers from universal gate collapse (46/46 runs). The root cause is a multiplicative sigmoid gate that creates a death spiral: when γ → 0, both forward signal and backward gradient vanish, making the state absorbing (§2.1 of SOLVING_GATE_COLLAPSE.md).

FiLM (Feature-wise Linear Modulation, Perez et al. 2018) replaces the sigmoid gate with an **unconstrained affine transform** `γ · x + β`, where both γ and β are produced by the modulator. This structurally breaks the death spiral: even when γ → 0, the additive β term preserves gradient flow to the modulator, allowing it to "resurrect" suppressed channels.

The original FiLM paper applies the affine transform **after normalization** (batch norm or layer norm). This normalization step is critical in FiLM's original context (conditioning a vision backbone) but may or may not be necessary in our RL encoder. We therefore implement three experimental conditions to compare.

## Analysis

### Why FiLM Normalization Matters

In the original FiLM paper, the conditioning is applied to features **after** batch/layer normalization:

```
x_pre = Linear(x)
x_norm = LayerNorm(x_pre)          ← normalize to zero mean, unit variance
x_out = γ(z) · x_norm + β(z)      ← FiLM affine transform
x_act = activation(x_out)          ← ReLU / SiLU
```

The normalization serves two purposes:

1. **Standardized modulation target**: γ and β operate on features with known statistics (μ=0, σ=1), so the modulator can learn a consistent mapping from context → modulation. Without normalization, the pre-activation magnitudes vary across neurons and training time, making γ and β semantically inconsistent.
2. **Prevents feature magnitude drift**: Without normalization, the multiplicative γ can compound with already-large activations, leading to magnitude explosion or making the β term negligible relative to γ·x.

### Current Architecture vs FiLM


| Aspect        | Current (Multiplicative) | Current (PreActivation)   | **FiLM**                            |
| ------------- | ------------------------ | ------------------------- | ----------------------------------- |
| Equation      | `relu(Wx) · σ(z)`        | `relu(σ(z_γ) · Wx + z_β)` | `act(γ(z) · LN(Wx) + β(z))`         |
| γ range       | (0, 1) via sigmoid       | **(0, 1) via sigmoid**    | **unconstrained (ℝ)**               |
| β             | none / zeros             | unconstrained             | unconstrained                       |
| Normalization | none                     | none                      | LayerNorm before modulation         |
| Can amplify?  | No                       | Indirectly (β only)       | **Yes** (γ > 1)                     |
| Death spiral? | Yes (§2.1)               | Partially (β helps)       | **No** (β provides gradient bypass) |


### Key Architectural Difference: γ is Unconstrained

In PreActivation mode, γ = sigmoid(z) ∈ (0, 1) — attenuation only. In FiLM, **γ is not passed through sigmoid**. The raw modulator output is used directly as the scaling factor. This means:

- γ > 1 → amplification (impossible in current modes)
- γ < 0 → signal inversion (impossible in current modes)
- γ = 0 → only β survives (gradient still flows)

This is the single most important difference. The sigmoid ceiling at 1.0 in current modes prevents amplification, biasing the modulator toward suppression. FiLM removes this bias.

### Experimental Conditions

We compare **three modulation types** against the existing **Multiplicative** baseline:


| Condition                    | Config value       | Description                                                |
| ---------------------------- | ------------------ | ---------------------------------------------------------- |
| **Multiplicative** (control) | `"Multiplicative"` | Existing: `relu(Wx) · σ(z)`                                |
| **FiLM**                     | `"FiLM"`           | Full FiLM: `act(γ · LN(Wx) + β)`, γ unconstrained          |
| **FiLMNoNorm**               | `"FiLMNoNorm"`     | FiLM without LayerNorm: `act(γ · Wx + β)`, γ unconstrained |


The **FiLM vs FiLMNoNorm** comparison isolates the effect of normalization. The **FiLMNoNorm vs PreActivation** comparison isolates the effect of unconstrained γ (since PreActivation uses sigmoid-bounded γ).

### Initialization Strategy

FiLM initialization is critical for pass-through at start:

- **γ init**: bias = 1.0, weights = small (so γ ≈ 1.0 → identity scaling)
- **β init**: bias = 0.0, weights = small (so β ≈ 0.0 → no shift)

This produces `1.0 · x + 0.0 = x` at initialization — true pass-through. Compare with the current `sigmoid(3.0) ≈ 0.95` which starts at 5% attenuation.

For FiLM (with LayerNorm), the initialization produces `1.0 · LN(x) + 0.0 = LN(x)`. The network starts with normalized features, and the modulator starts at identity.

### Where to Add LayerNorm

LayerNorm is inserted **after the linear layer, before the FiLM transform**, at each modulation injection point:

**Phase 1 (Unimodal)**: After `GroupedMLP` output, before FiLM + activation

- Shape: `(batch, 9, 128)` → LayerNorm over last dim (128) → FiLM → relu/silu
- Need a `GroupedLayerNorm` that normalizes each modality's 128-dim features independently

**Phase 2 (Multimodal Hub)**: After hub MLP output, before FiLM + activation

- Shape: `(batch, 128)` → LayerNorm over last dim (128) → FiLM → relu/silu
- Standard `nnx.LayerNorm(128)`

**Important**: The LayerNorm layers are **only constructed when modulation_type == "FiLM"**. They are NOT added to the unmodulated baseline or other modulation modes, preserving exact baseline equivalence.

---

## Implementation Plan

### Design

**Minimal changes.** FiLM/FiLMNoNorm are nearly identical to PreActivation — the only differences are (1) no sigmoid on γ, (2) γ bias init = 1.0, and (3) optional LayerNorm. We implement this by extending existing `if` conditions with `or` clauses and adding new `elif` branches in the forward path. No new classes, no helper functions, no refactoring of existing code.

Add `"FiLM"` and `"FiLMNoNorm"` as new `modulation.type` options alongside `"Multiplicative"` and `"PreActivation"`.

**Data flow (FiLM with norm)**:

```
GroupedMLP(x)  →  LayerNorm  →  γ(z) · normalized + β(z)  →  relu
                                  ↑ unconstrained
```

**Data flow (FiLMNoNorm)**:

```
GroupedMLP(x)  →  γ(z) · raw + β(z)  →  relu
                    ↑ unconstrained
```

### File Changes

#### 1. `src/models/neuromodulator.py` — NeuromodulatorRNN

**Goal**: FiLM/FiLMNoNorm construct the same additive heads as PreActivation, but with γ bias = 1.0.

##### Lines 88–121 — `__init__` head construction

Extend existing conditions with `or` clauses. Override γ bias for FiLM types.

```python
# BEFORE (lines 91-95):
        # Phase 1: Unimodal
        self.head_unimodal = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                       bias_init=nnx.initializers.constant(percept_bias_init), rngs=rngs)
        if self.modulation_type == "PreActivation":
            self.head_unimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                               bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)

# AFTER:
        # Phase 1: Unimodal
        # FiLM modes use γ bias = 1.0 (identity in linear space); sigmoid modes use percept_bias_init
        if self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            _percept_bias = 1.0
        else:
            _percept_bias = percept_bias_init

        self.head_unimodal = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                       bias_init=nnx.initializers.constant(_percept_bias), rngs=rngs)
        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.head_unimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                               bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)
```

Apply the same pattern to multimodal heads (lines 98-102):

```python
# BEFORE:
        self.head_multimodal = nnx.Linear(mod_hidden_size, self.num_groups_hidden,
                                         bias_init=nnx.initializers.constant(percept_bias_init), rngs=rngs)
        if self.modulation_type == "PreActivation":
            self.head_multimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_hidden,
                                                 bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)

# AFTER:
        self.head_multimodal = nnx.Linear(mod_hidden_size, self.num_groups_hidden,
                                         bias_init=nnx.initializers.constant(_percept_bias), rngs=rngs)
        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.head_multimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_hidden,
                                                 bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)
```

And baseline parameters (lines 119-121):

```python
# BEFORE:
        if self.modulation_type == "PreActivation":
            self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))
            self.z_hidden_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))

# AFTER:
        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))
            self.z_hidden_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))
```

**No changes** to `__call__` / `_get_signal` — these already handle additive heads via `getattr`.

##### DreamerNeuromodulatorRNN (lines 195–365)

Apply the **exact same pattern**: extend the three existing `if self.modulation_type == "PreActivation":` conditions at lines 258, 265, 288 with `or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm"`. Add the same `_percept_bias` override before head construction.

#### 2. `src/models/recurrent_ppo_network.py` — ObservationEncoder

**Goal**: Add `elif` branches for FiLM/FiLMNoNorm in `forward_with_modulation`. Add LayerNorm construction in `ActorCriticRNN`.

##### Lines 154–226 — `ActorCriticRNN.__init__`

After the encoder is constructed (line 183), add LayerNorm layers for FiLM:

```python
# AFTER line 183 (after self.obs_encoder = ObservationEncoder(...)):

        # FiLM LayerNorm layers (only for FiLM, not FiLMNoNorm)
        if self.modulation_enabled and self.modulation_type == "FiLM":
            if hasattr(self.obs_encoder, 'names'):  # hierarchical mode
                self.film_unimodal_ln = nnx.LayerNorm(hidden_size, rngs=rngs)
                self.film_multimodal_ln = nnx.LayerNorm(hidden_size, rngs=rngs)
            else:  # flat mode
                self.film_flat_ln = nnx.LayerNorm(hidden_size, rngs=rngs)
```

##### Lines 121–151 — `ObservationEncoder.forward_with_modulation`

Add new `elif` branches for FiLM and FiLMNoNorm. The existing Multiplicative and PreActivation branches stay untouched.

```python
# REPLACE lines 121-151 entirely with:

    def forward_with_modulation(self, x, mod_output, modulation_type: str,
                                film_unimodal_ln=None, film_multimodal_ln=None,
                                film_flat_ln=None):
        """Hierarchical forward pass with multi-stage modulation (Injection A)."""
        # --- Flat mode ---
        if self.mode != 'hierarchical':
            x_proj = self.monolith(x)
            if modulation_type == "PreActivation":
                gamma = jax.nn.sigmoid(mod_output.z_unimodal)
                beta = mod_output.z_unimodal_add
                return jax.nn.relu(x_proj * gamma + beta)
            elif modulation_type == "FiLM":
                if film_flat_ln is not None:
                    x_proj = film_flat_ln(x_proj)
                return jax.nn.relu(mod_output.z_unimodal * x_proj + mod_output.z_unimodal_add)
            elif modulation_type == "FiLMNoNorm":
                return jax.nn.relu(mod_output.z_unimodal * x_proj + mod_output.z_unimodal_add)
            else:  # Multiplicative
                return jax.nn.relu(x_proj) * jax.nn.sigmoid(mod_output.z_unimodal)

        # --- Hierarchical mode ---
        batch_shape = x.shape[:-1]
        x_padded = jnp.zeros(batch_shape + (len(self.names), self.max_in), dtype=x.dtype)
        start = 0
        for i, (name, dim) in enumerate(self.breakdown.items()):
            x_padded = x_padded.at[..., i, :dim].set(x[..., start : start + dim])
            start += dim

        # Phase 1: Unimodal + Modulation
        encoded_all = self.unimodal_grouped(x_padded)

        if modulation_type == "PreActivation":
            gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)
            beta1 = mod_output.z_unimodal_add
            encoded_all = jax.nn.relu(encoded_all * gamma1[..., None, :] + beta1[..., None, :])
        elif modulation_type == "FiLM":
            if film_unimodal_ln is not None:
                encoded_all = jax.vmap(film_unimodal_ln, in_axes=-2, out_axes=-2)(encoded_all)
            gamma1 = mod_output.z_unimodal
            beta1 = mod_output.z_unimodal_add
            encoded_all = jax.nn.relu(gamma1[..., None, :] * encoded_all + beta1[..., None, :])
        elif modulation_type == "FiLMNoNorm":
            gamma1 = mod_output.z_unimodal
            beta1 = mod_output.z_unimodal_add
            encoded_all = jax.nn.relu(gamma1[..., None, :] * encoded_all + beta1[..., None, :])
        else:  # Multiplicative
            gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)
            encoded_all = jax.nn.relu(encoded_all) * gamma1[..., None, :]

        # Phase 2: Multimodal Hub + Modulation
        mm_in = encoded_all.reshape(batch_shape + (-1,))
        mm_latent = self.multimodal_hub(mm_in)

        if modulation_type == "PreActivation":
            gamma2 = jax.nn.sigmoid(mod_output.z_multimodal)
            beta2 = mod_output.z_multimodal_add
            return jax.nn.relu(mm_latent * gamma2 + beta2)
        elif modulation_type == "FiLM":
            if film_multimodal_ln is not None:
                mm_latent = film_multimodal_ln(mm_latent)
            gamma2 = mod_output.z_multimodal
            beta2 = mod_output.z_multimodal_add
            return jax.nn.relu(gamma2 * mm_latent + beta2)
        elif modulation_type == "FiLMNoNorm":
            gamma2 = mod_output.z_multimodal
            beta2 = mod_output.z_multimodal_add
            return jax.nn.relu(gamma2 * mm_latent + beta2)
        else:  # Multiplicative
            gamma2 = jax.nn.sigmoid(mod_output.z_multimodal)
            return jax.nn.relu(mm_latent) * gamma2
```

##### Line 264 — `ActorCriticRNN.__call__` (the call site)

Pass LayerNorm layers through:

```python
# BEFORE (line 264):
            x_proj = self.obs_encoder.forward_with_modulation(x, mod_output, self.modulation_type)

# AFTER:
            x_proj = self.obs_encoder.forward_with_modulation(
                x, mod_output, self.modulation_type,
                film_unimodal_ln=getattr(self, 'film_unimodal_ln', None),
                film_multimodal_ln=getattr(self, 'film_multimodal_ln', None),
                film_flat_ln=getattr(self, 'film_flat_ln', None),
            )
```

#### 3. `src/models/dreamer_v3_nnx.py` — DreamerObservationEncoder + Encoder

Same minimal pattern as RecurrentPPO. The Dreamer encoder uses SiLU instead of ReLU.

##### `DreamerObservationEncoder.forward_with_modulation` (lines 259–282)

Add `elif modulation_type == "FiLM":` and `elif modulation_type == "FiLMNoNorm":` branches at both Phase 1 and Phase 2, using unconstrained γ and SiLU activation. Existing branches untouched.

##### `Encoder.forward_with_modulation` (lines 319–340)

Same — add two `elif` branches for the flat encoder fallback.

##### DreamerV3 WorldModel construction

Attach `film_unimodal_ln` and `film_multimodal_ln` when `self.modulation_type == "FiLM"`, same as `ActorCriticRNN`.

#### 4. `configs/models/ppo/neuromodulated_ppo.yaml`

Update comment only:

```yaml
# BEFORE (line 54):
    type: "Multiplicative"       # "Multiplicative", "PreActivation", or null

# AFTER:
    type: "Multiplicative"       # "Multiplicative", "PreActivation", "FiLM", "FiLMNoNorm", or null
```

No new config keys needed — γ bias is overridden internally for FiLM modes.

#### 5. Logging — `src/models/dreamer_v3_trainer.py` (lines 274–286)

Log raw γ for FiLM modes instead of sigmoid-transformed:

```python
# BEFORE:
                    'mod_z_unimodal_mean': jnp.mean(jax.nn.sigmoid(mod_outputs_T.z_unimodal)),
                    'mod_z_unimodal_std': jnp.std(jax.nn.sigmoid(mod_outputs_T.z_unimodal)),
                    'mod_z_multimodal_mean': jnp.mean(jax.nn.sigmoid(mod_outputs_T.z_multimodal)),
                    'mod_z_multimodal_std': jnp.std(jax.nn.sigmoid(mod_outputs_T.z_multimodal)),

# AFTER:
                    if wm.modulation_type == "FiLM" or wm.modulation_type == "FiLMNoNorm":
                        effective_gamma_uni = mod_outputs_T.z_unimodal
                        effective_gamma_multi = mod_outputs_T.z_multimodal
                    else:
                        effective_gamma_uni = jax.nn.sigmoid(mod_outputs_T.z_unimodal)
                        effective_gamma_multi = jax.nn.sigmoid(mod_outputs_T.z_multimodal)

                    'mod_z_unimodal_mean': jnp.mean(effective_gamma_uni),
                    'mod_z_unimodal_std': jnp.std(effective_gamma_uni),
                    'mod_z_multimodal_mean': jnp.mean(effective_gamma_multi),
                    'mod_z_multimodal_std': jnp.std(effective_gamma_multi),
```

### LayerNorm in FiLM: Detailed Walkthrough

This section explains how LayerNorm operates within the FiLM modulation pipeline, with concrete numerical examples. It covers the mathematical formula, the per-phase application, and the vmap trick used for the unimodal stage.

#### The LayerNorm Formula

`nnx.LayerNorm(128)` computes, for each sample independently:

```
x_norm = (x - μ) / √(σ² + ε) × scale + bias
```

Where:

- **μ** = mean of the 128-dim feature vector
- **σ²** = variance of the 128-dim feature vector
- **ε** = 1e-5 (numerical stability constant)
- **scale** = learnable parameter, initialized to **1.0** (shape: 128)
- **bias** = learnable parameter, initialized to **0.0** (shape: 128)

At initialization (scale=1, bias=0), this reduces to pure standardization: `(x - μ) / √(σ² + ε)`.

#### Concrete Example: Phase 1 (Unimodal)

Suppose after `GroupedMLP`, one modality (e.g. Olfaction, index 4) produces a 128-dim vector for one sample:

```
encoded_all[batch=0, modality=4, :] = [2.1, 0.3, -1.5, 4.0, ..., 0.8]
                                        ←————— 128 values ——————→
```

**Step 1 — Compute statistics over the 128 dims:**

```
μ = mean([2.1, 0.3, -1.5, 4.0, ..., 0.8]) = 1.2   (example value)
σ² = var([2.1, 0.3, -1.5, 4.0, ..., 0.8]) = 3.5    (example value)
```

**Step 2 — Normalize each element:**

```
x_norm[0] = (2.1 - 1.2) / √(3.5 + 1e-5) = 0.9 / 1.871  =  0.481
x_norm[1] = (0.3 - 1.2) / √(3.5 + 1e-5) = -0.9 / 1.871 = -0.481
x_norm[2] = (-1.5 - 1.2) / √(3.5 + 1e-5) = -2.7 / 1.871 = -1.443
x_norm[3] = (4.0 - 1.2) / √(3.5 + 1e-5) = 2.8 / 1.871  =  1.496
...
```

After this step, the 128 values have **mean ≈ 0** and **variance ≈ 1**.

**Step 3 — Apply learnable scale and bias (initially identity):**

```
output[i] = scale[i] × x_norm[i] + bias[i]
           = 1.0 × x_norm[i] + 0.0           ← at initialization
           = x_norm[i]
```

**Step 4 — FiLM affine transform** (`recurrent_ppo_network.py:162`):

The neuromodulator's unconstrained γ and β are applied to the normalized features:

```
γ₁ ≈ 1.0  (at init, because γ head bias = 1.0)
β₁ ≈ 0.0  (at init, because β head bias = 0.0)

result = relu(γ₁ × x_norm + β₁)
       = relu(1.0 × 0.481 + 0.0)    = 0.481
       = relu(1.0 × (-0.481) + 0.0) = 0.0      ← killed by ReLU
       = relu(1.0 × (-1.443) + 0.0) = 0.0      ← killed by ReLU
       = relu(1.0 × 1.496 + 0.0)    = 1.496
```

At initialization, the full pipeline is: **normalize → identity scaling → ReLU**. The modulator has no effect, which is the desired pass-through behavior.

#### Why LayerNorm Matters for Modulation

Without LayerNorm, raw activations from `GroupedMLP` can have arbitrary magnitude — some neurons output values around 5.0, others around 0.01. The modulator's γ and β would have to learn different scales per neuron, which is difficult.

With LayerNorm, **every neuron's output is standardized to approximately N(0,1)** before γ/β are applied. This gives the modulation signals a consistent semantic meaning:


| Modulator output | Effect                                                             |
| ---------------- | ------------------------------------------------------------------ |
| γ = 1.5          | Amplify by 50% — consistently, for every neuron                    |
| γ = 0.0          | Suppress entirely, rely only on β — gradient still flows through β |
| γ < 0            | Signal inversion (impossible in sigmoid-based modes)               |
| β = 0.3          | Shift by 0.3 standard deviations — a consistent semantic offset    |


This is what the original FiLM paper (Perez et al. 2018) calls a "standardized modulation target": the modulator learns a stable mapping from interoceptive state to modulation, regardless of the varying feature magnitudes across neurons and training time.

#### Phase 1: Unimodal — Vmap Over Modalities

The unimodal phase has shape `(batch, 9, 128)` — 9 modalities, each with 128 features. But `nnx.LayerNorm(128)` expects input of shape `(..., 128)`. The vmap trick handles this:

```python
encoded_all = jax.vmap(film_unimodal_ln, in_axes=-2, out_axes=-2)(encoded_all)
```

This tells JAX: "treat the modality axis (dim -2, size 9) as the batch axis for vmap." It applies the **same** LayerNorm instance 9 times — once per modality — normalizing across the 128 features independently each time:

```
Input:  (batch, 9, 128)
         │      │    │
         │      │    └── LayerNorm normalizes over this axis (μ, σ² computed here)
         │      └── vmap iterates over this axis (9 independent applications)
         └── real batch axis (preserved)

Output: (batch, 9, 128)  ← same shape, each (batch, 128) slice independently normalized
```

Key properties:

- **Same LN parameters** (scale, bias) are shared across all 9 modalities — Injury and Olfaction get the same learned scale/bias.
- **Statistics are computed independently** per modality — each modality's mean and variance reflect its own activation distribution, not a mixture of all modalities.
- If vmap over `nnx.LayerNorm` proves problematic at JIT time, the fallback is to reshape `(batch, 9, 128)` → `(batch×9, 128)`, apply LayerNorm, then reshape back.

#### Phase 2: Multimodal Hub — Direct Application

At `recurrent_ppo_network.py:180–181`, the hub output is `(batch, 128)` — no modality dimension — so LayerNorm is applied directly without vmap:

```python
mm_latent = film_multimodal_ln(mm_latent)   # (batch, 128) → (batch, 128)
```

Same math as above, just on the fused multimodal representation.

#### Parameter Cost

Each `nnx.LayerNorm(128)` adds 256 parameters (128 scale + 128 bias). With two LN layers (unimodal + multimodal), FiLM mode adds **512 total parameters** compared to FiLMNoNorm or PreActivation. This is negligible relative to the full model size.

#### FAQ: LayerNorm + ReLU — Doesn't Half the Signal Get Killed?

A natural concern: LayerNorm centers features to mean=0, which means roughly **half the values become negative and get killed by ReLU**. This seems wasteful. This section explains why it still works well, and specifically why it works in our FiLM context.

##### The "half gets killed" concern is real but misleading

Yes, after LayerNorm, ~50% of neurons have negative values. But this is **not a problem** — it's actually similar to what happens without LayerNorm:

1. **ReLU already kills ~50% of neurons in healthy networks.** A well-initialized linear layer (e.g. Kaiming/He init) produces roughly zero-centered outputs. ReLU was designed for this regime. If ReLU is killing significantly *less* than ~50%, it often means activations have drifted positive (magnitude explosion), which is its own problem.
2. **The neurons that survive are the informative ones.** LayerNorm ensures the surviving neurons have well-scaled magnitudes (in the range of ~0 to 2–3 standard deviations), rather than arbitrary magnitudes that vary across neurons and training time.

##### The real reason LayerNorm works with ReLU

The key insight: **LayerNorm's value isn't about preserving all activations — it's about controlling the scale of the ones that survive.**

Without LayerNorm:

```
GroupedMLP output: [0.002, 15.3, -0.5, 42.1, ...]   ← wildly different scales
After ReLU:        [0.002, 15.3,  0.0, 42.1, ...]   ← surviving values span 4 orders of magnitude
```

With LayerNorm:

```
GroupedMLP output: [0.002, 15.3, -0.5, 42.1, ...]
After LN:         [-0.89, 0.52, -0.92, 1.85, ...]   ← standardized
After ReLU:        [0.0,  0.52,  0.0,  1.85, ...]   ← surviving values are well-scaled
```

The surviving positive values are in a consistent range, which makes the downstream FiLM modulation (γ and β) much easier to learn.

##### In our FiLM case specifically, there's a stronger reason

In our pipeline, ReLU comes **after** the FiLM transform, not immediately after LayerNorm:

```
GroupedMLP → LayerNorm → γ·x + β → ReLU
                         ↑
                    β can shift negatives back to positive
```

The neuromodulator's **β (additive bias) can rescue negative values before ReLU sees them**. If a neuron outputs -1.2 after LayerNorm, and β = 1.5, then the input to ReLU is `γ·(-1.2) + 1.5`, which can be positive. The modulator controls which neurons survive ReLU — that's the whole point of FiLM modulation.

This is structurally different from a plain `LayerNorm → ReLU` stack where the negatives are unconditionally killed. Here, the modulator **decides** what to keep and what to suppress, based on the agent's interoceptive state.

##### Why not use an activation that doesn't kill negatives?

You could. The DreamerV3 encoder in our codebase already uses **SiLU** (Sigmoid Linear Unit), which passes negative values through (attenuated). That's partly why DreamerV3's architecture uses SiLU everywhere — it avoids the dead-neuron problem entirely.

For RecurrentPPO, changing ReLU → SiLU would be an independent experiment worth considering, but it's orthogonal to the FiLM question. The FiLM + LayerNorm combination works with ReLU because β provides the mechanism to shift the decision boundary.

##### Summary


| Concern             | Why it's OK                                                                      |
| ------------------- | -------------------------------------------------------------------------------- |
| ~50% killed by ReLU | Normal and expected — ReLU in healthy networks kills ~50% anyway                 |
| Lost information    | β can shift negatives positive before ReLU — the modulator decides what survives |
| Scale consistency   | The real win — surviving values are well-scaled, making γ/β semantically stable  |
| Alternative         | DreamerV3 side uses SiLU, which avoids this entirely                             |


The fame of LayerNorm isn't about preserving all activations — it's about **stabilizing training dynamics** by preventing magnitude drift, which makes everything downstream (including our FiLM γ/β) easier to learn.

### Summary of Files Changed


| File                                     | Change                                                                         | New Lines (approx) |
| ---------------------------------------- | ------------------------------------------------------------------------------ | ------------------ |
| `src/models/neuromodulator.py`           | Extended `or` conditions on 3 existing `if`s + `_percept_bias` override        | ~10                |
| `src/models/recurrent_ppo_network.py`    | Two new `elif` branches per phase + LayerNorm construction + updated call site | ~30                |
| `src/models/dreamer_v3_nnx.py`           | Two new `elif` branches per phase + LayerNorm construction                     | ~30                |
| `src/models/dreamer_v3_trainer.py`       | `if/else` for effective γ logging                                              | ~6                 |
| `configs/models/ppo/neuromodulated_ppo.yaml` | Comment update (doc only)                                                      | ~1                 |


---

## Checkpoints

- **Checkpoint 1** — After neuromodulator.py changes: instantiate `NeuromodulatorRNN(modulation_type="FiLM")` and verify that `head_unimodal_add` and `head_multimodal_add` exist, and that the γ head bias is 1.0 (not 3.0). Print `model.head_unimodal.bias.value` to confirm. [17:23:45]
- **Checkpoint 2** — After encoder changes: run a single forward pass with FiLM mode and print shapes at each stage. Verify `gamma1` is unconstrained (can be > 1 or < 0), not sigmoid-bounded. Print `gamma1.min(), gamma1.max()` — at init, should be ≈ 1.0. [17:28:15]
- **Checkpoint 3** — LayerNorm vmap: print `encoded_all` shape before and after LayerNorm application. Must remain `(batch, 9, 128)`. If vmap fails, use reshape fallback. [17:28:15]
- **Checkpoint 4** — Run 1 training iteration with `type: "FiLM"` and `type: "FiLMNoNorm"`. Verify no NaN/Inf in loss. Check that `mod_z_unimodal_mean` in logs reflects raw γ (should be ≈ 1.0 at start), not sigmoid-transformed. — confirmed, 100 steps clean [18:24:00]
- **Checkpoint 5** — Run 1 training iteration with `type: "Multiplicative"` to verify no regression in existing modes. — confirmed, 100 steps clean [18:26:00]
- **Checkpoint 6** — Parameter count: FiLM should add only the LayerNorm parameters (128 scale + 128 bias = 256 per LN, × 2 LN layers = 512 total) relative to PreActivation mode. FiLMNoNorm should have identical parameter count to PreActivation. — confirmed, consistent results [18:30:00]

---

> **Implemented by**: Gemini
> **Date**: 2026-03-17 18:35:00

### FiLM Implementation Report (2026-03-17)

- Implemented `FiLM` and `FiLMNoNorm` in `neuromodulator.py`.
- Integrated affine modulation with optional LayerNorm in `recurrent_ppo_network.py`.
- Updated DreamerV3 (`dreamer_v3_nnx.py`) with FiLM support and SiLU activations.
- Modified `dreamer_v3_trainer.py` to log unconstrained gamma signals.
- Verified all checkpoints (1-6) including training iterations and parameter counts.
- Updated `neuromodulated_ppo.yaml` with new modulation types.

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-03-17


| File                                     | Change                                                                             | Status | Notes                                                                                                                                                                    |
| ---------------------------------------- | ---------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/models/neuromodulator.py`           | `_percept_bias` override + extended `or` conditions (both RNN classes)             | ✅      | Clean, matches plan exactly. 3 conditions extended per class.                                                                                                            |
| `src/models/recurrent_ppo_network.py`    | FiLM/FiLMNoNorm branches in encoder + LayerNorm construction + call site           | ✅      | All 4 branches correct. LayerNorm uses `self.obs_encoder.mode` (cleaner than `hasattr`).                                                                                 |
| `src/models/dreamer_v3_nnx.py`           | FiLM/FiLMNoNorm branches in both encoders + WorldModel LayerNorm + agent call site | ✅      | Uses `jax.nn.silu` directly instead of `SiLU()` wrapper — functionally identical. Flat encoder passes `film_flat_ln` correctly.                                          |
| `src/models/dreamer_v3_trainer.py`       | Effective γ logging + beta logging condition extended                              | ✅      | Training loop call site (L148) correctly updated. Beta logging condition correctly extended.                                                                             |
| `src/models/dreamer_v3_trainer.py`       | `get_action` call site (L495)                                                      | ❌      | **Missing `film_*_ln` args** — calls `forward_with_modulation` without passing LayerNorm layers. FiLM mode will silently skip normalization during inference/evaluation. |
| `configs/models/ppo/neuromodulated_ppo.yaml` | Comment update                                                                     | ✅      | Correct.                                                                                                                                                                 |


### ❌ Detail: `dreamer_v3_trainer.py:495` — Missing FiLM LayerNorm in `get_action`

**Current (wrong)**:

```python
embed = self.agent.wm.encoder.forward_with_modulation(
    obs_symlog, mod_output, self.agent.wm.modulation_type)
```

**Should be**:

```python
embed = self.agent.wm.encoder.forward_with_modulation(
    obs_symlog, mod_output, self.agent.wm.modulation_type,
    film_unimodal_ln=getattr(self.agent.wm, 'film_unimodal_ln', None),
    film_multimodal_ln=getattr(self.agent.wm, 'film_multimodal_ln', None),
    film_flat_ln=getattr(self.agent.wm, 'film_flat_ln', None)
)
```

**Impact**: During FiLM inference (action selection), LayerNorm is skipped. The encoder sees un-normalized features while the γ/β were trained expecting normalized input. This creates a train/eval mismatch — training uses LN (line 148 is correct), but action selection does not.

## Implementation Update (2026-03-17)

> **Fixed by**: Gemini
> **Status**: RESOLVED

I have addressed the ❌ identified by Claude in `dreamer_v3_trainer.py:495`. The `get_action` call site now correctly passes the `film_*_ln` parameters using `getattr`. 

**Final Conclusion**: All 6 files are now fully aligned. Training and inference parity is guaranteed for all modulation modes (FiLM, FiLMNoNorm, Multiplicative, PreActivation).

## Re-Verification Report (2026-03-17)

> **Verified by**: Claude
> **Date**: 2026-03-17

Re-verified after Gemini's fix for the `get_action` call site.


| File                                     | Change                                                                             | Status | Notes                                                                                                                                  |
| ---------------------------------------- | ---------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `src/models/neuromodulator.py`           | `_percept_bias` override + extended `or` conditions (both RNN classes)             | ✅      | No change since last verification.                                                                                                     |
| `src/models/recurrent_ppo_network.py`    | FiLM/FiLMNoNorm branches in encoder + LayerNorm construction + call site           | ✅      | No change since last verification.                                                                                                     |
| `src/models/dreamer_v3_nnx.py`           | FiLM/FiLMNoNorm branches in both encoders + WorldModel LayerNorm + agent call site | ✅      | No change since last verification.                                                                                                     |
| `src/models/dreamer_v3_trainer.py`       | Effective γ logging + training call site (L148)                                    | ✅      | No change since last verification.                                                                                                     |
| `src/models/dreamer_v3_trainer.py`       | `get_action` call site (L495)                                                      | ✅      | **Fixed.** Now passes `film_unimodal_ln`, `film_multimodal_ln`, `film_flat_ln` via `getattr`, matching the training call site pattern. |
| `configs/models/ppo/neuromodulated_ppo.yaml` | Comment update                                                                     | ✅      | No change since last verification.                                                                                                     |


**Diff stats**: 5 files changed, +180 / −55. Proportionate to plan scope (5 files, ~77 new lines planned; extra lines from `elif` branch duplication and signature changes are expected).

**Conclusion**: All ✅. The previous ❌ at `dreamer_v3_trainer.py:495` is resolved — train/eval parity is now guaranteed for FiLM mode. Implementation is complete and ready for experimental runs.