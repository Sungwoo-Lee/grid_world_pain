# FiLM-Style Modulation: Implementation Plan

> **Status**: IN PROGRESS
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

| Aspect | Current (Multiplicative) | Current (PreActivation) | **FiLM** |
|--------|-------------------------|------------------------|----------|
| Equation | `relu(Wx) · σ(z)` | `relu(σ(z_γ) · Wx + z_β)` | `act(γ(z) · LN(Wx) + β(z))` |
| γ range | (0, 1) via sigmoid | (0, 1) via sigmoid | **unconstrained** (ℝ) |
| β | none / zeros | unconstrained | unconstrained |
| Normalization | none | none | LayerNorm before modulation |
| Can amplify? | No | Indirectly (β only) | **Yes** (γ > 1) |
| Death spiral? | Yes (§2.1) | Partially (β helps) | **No** (β provides gradient bypass) |

### Key Architectural Difference: γ is Unconstrained

In PreActivation mode, γ = sigmoid(z) ∈ (0, 1) — attenuation only. In FiLM, **γ is not passed through sigmoid**. The raw modulator output is used directly as the scaling factor. This means:
- γ > 1 → amplification (impossible in current modes)
- γ < 0 → signal inversion (impossible in current modes)
- γ = 0 → only β survives (gradient still flows)

This is the single most important difference. The sigmoid ceiling at 1.0 in current modes prevents amplification, biasing the modulator toward suppression. FiLM removes this bias.

### Experimental Conditions

We compare **three modulation types** against the existing **Multiplicative** baseline:

| Condition | Config value | Description |
|-----------|-------------|-------------|
| **Multiplicative** (control) | `"Multiplicative"` | Existing: `relu(Wx) · σ(z)` |
| **FiLM** | `"FiLM"` | Full FiLM: `act(γ · LN(Wx) + β)`, γ unconstrained |
| **FiLMNoNorm** | `"FiLMNoNorm"` | FiLM without LayerNorm: `act(γ · Wx + β)`, γ unconstrained |

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

#### 4. `configs/models/neuromodulated_ppo.yaml`

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

### LayerNorm Vmapping Detail

The unimodal phase has shape `(batch, 9, 128)`. We need LayerNorm over the last dim (128) applied independently per modality. With a single `nnx.LayerNorm(128)`:

```python
encoded_all = jax.vmap(film_unimodal_ln, in_axes=-2, out_axes=-2)(encoded_all)
```

This vmaps the LayerNorm over the 9-modality dimension, applying it to each `(batch, 128)` slice. The same LayerNorm parameters are shared across all 9 modalities.

If vmap over `nnx.LayerNorm` proves problematic at JIT time, the fallback is to reshape `(batch, 9, 128)` → `(batch*9, 128)`, apply LayerNorm, reshape back.

### Summary of Files Changed

| File | Change | New Lines (approx) |
|------|--------|-------------------|
| `src/models/neuromodulator.py` | Extended `or` conditions on 3 existing `if`s + `_percept_bias` override | ~10 |
| `src/models/recurrent_ppo_network.py` | Two new `elif` branches per phase + LayerNorm construction + updated call site | ~30 |
| `src/models/dreamer_v3_nnx.py` | Two new `elif` branches per phase + LayerNorm construction | ~30 |
| `src/models/dreamer_v3_trainer.py` | `if/else` for effective γ logging | ~6 |
| `configs/models/neuromodulated_ppo.yaml` | Comment update (doc only) | ~1 |

---

## Checkpoints

- [ ] **Checkpoint 1** — After neuromodulator.py changes: instantiate `NeuromodulatorRNN(modulation_type="FiLM")` and verify that `head_unimodal_add` and `head_multimodal_add` exist, and that the γ head bias is 1.0 (not 3.0). Print `model.head_unimodal.bias.value` to confirm.
- [ ] **Checkpoint 2** — After encoder changes: run a single forward pass with FiLM mode and print shapes at each stage. Verify `gamma1` is unconstrained (can be > 1 or < 0), not sigmoid-bounded. Print `gamma1.min(), gamma1.max()` — at init, should be ≈ 1.0.
- [ ] **Checkpoint 3** — LayerNorm vmap: print `encoded_all` shape before and after LayerNorm application. Must remain `(batch, 9, 128)`. If vmap fails, use reshape fallback.
- [ ] **Checkpoint 4** — Run 1 training iteration with `type: "FiLM"` and `type: "FiLMNoNorm"`. Verify no NaN/Inf in loss. Check that `mod_z_unimodal_mean` in logs reflects raw γ (should be ≈ 1.0 at start), not sigmoid-transformed.
- [ ] **Checkpoint 5** — Run 1 training iteration with `type: "Multiplicative"` to verify no regression in existing modes.
- [ ] **Checkpoint 6** — Parameter count: FiLM should add only the LayerNorm parameters (128 scale + 128 bias = 256 per LN, × 2 LN layers = 512 total) relative to PreActivation mode. FiLMNoNorm should have identical parameter count to PreActivation.

---

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-03-17 17:21:45

## Verification Report

> **Verified by**: [pending]
> **Date**: [pending]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [pending]
