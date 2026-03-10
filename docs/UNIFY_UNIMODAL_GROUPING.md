# Unify Unimodal Grouping with Multimodal Grouping

> **Status**: PLANNED
> **Opened**: 2026-03-10
> **Related**: [NMN Architecture Review §2.4](docs/NMN_ARCHITECTURE_REVIEW.md)

---

## Context

The unimodal and multimodal modulation heads use different grouping mechanisms, but the original design intent was for both to use the same `grouping_size`-based spatial grouping. Currently:

- **Unimodal**: One scalar per modality (9 groups total). All 128 output neurons within a modality share a single gate. The number of groups is structurally fixed to `len(obs_breakdown)` — `grouping_size` has no effect.
- **Multimodal/Memory**: Uses `grouping_size` to produce `ceil(hidden_size / G)` groups, then repeat-and-slice to cover the full hidden dimension.

This inconsistency means the unimodal phase is always coarser than intended. With `grouping_size=64`, multimodal gets 2 groups per 128 neurons, but unimodal gets 1 group per 128 neurons (per modality). The `grouping_size` config parameter should control both.

## Analysis

### Current unimodal path

**Neuromodulator head** (`neuromodulator.py:80,89-90`):
```python
self.num_groups_unimodal = len(obs_breakdown)  # = 9, ignores grouping_size
self.head_unimodal = nnx.Linear(mod_hidden_size, self.num_groups_unimodal, ...)
# Output shape: (batch, 9)
```

**Signal computation** (`neuromodulator.py:137-140`):
```python
if is_unimodal:
    sig = baseline.value + raw  # shape (batch, 9) — no repeat
```

**Injection** (`recurrent_ppo_network.py:141-144`):
```python
gamma1 = sigmoid(mod_output.z_unimodal)            # shape (batch, 9)
encoded_all = relu(encoded_all * gamma1[..., None]) # broadcasts (batch, 9, 1) × (batch, 9, 128)
```

### Current multimodal path (target behavior)

**Neuromodulator head** (`neuromodulator.py:81,96-97`):
```python
self.num_groups_hidden = math.ceil(target_hidden_size / grouping_size)  # e.g. ceil(128/64) = 2
self.head_multimodal = nnx.Linear(mod_hidden_size, self.num_groups_hidden, ...)
# Output shape: (batch, 2)
```

**Signal computation** (`neuromodulator.py:142-143`):
```python
sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
sig = baseline.value + sig  # shape (batch, 128)
```

**Injection** (`recurrent_ppo_network.py:149-151`):
```python
gamma2 = sigmoid(mod_output.z_multimodal)           # shape (batch, 128)
return relu(mm_latent * gamma2 + beta2)             # element-wise
```

### Desired unimodal behavior

The unimodal head should produce the **same grouping pattern as multimodal**, shared uniformly across all modalities. The groups are not per-modality — they are a spatial pattern over neuron positions that applies identically to every modality's encoder output.

With `G=64, hidden_size=128`:

- Groups: `ceil(128/64) = 2` (same as multimodal)
- Head output: `(batch, 2)` → repeat-and-slice → `(batch, 128)`
- At injection: broadcast over the 9-modality dimension → `(batch, 1, 128) × (batch, 9, 128)`

This means all modalities receive the same modulation pattern: neurons 0–63 share one gate value, neurons 64–127 share another — uniformly across Injury, Nutrition, Olfaction, etc. The modulator controls **which neuron positions** are amplified/suppressed, not which modalities.

```
head_unimodal(h_mod) → [a, b]                       # 2 values, shared across modalities
repeat → [a,a,...×64, b,b,...×64]                     # expanded to 128

Applied to ALL modalities uniformly (broadcast over dim 0):
  Injury    neurons 0-63: × sigmoid(a)   neurons 64-127: × sigmoid(b)
  Nutrition neurons 0-63: × sigmoid(a)   neurons 64-127: × sigmoid(b)
  ...
  Location  neurons 0-63: × sigmoid(a)   neurons 64-127: × sigmoid(b)
```

The baseline is per-neuron `(128,)` — the same offset pattern for every modality, matching the multimodal baseline structure.

## Implementation Plan

### Design

Make the unimodal head structurally identical to the multimodal head: same `num_groups_hidden` output size, same repeat-and-slice signal computation, same per-neuron `(hidden_size,)` baseline. The only difference is where the signal is injected (at the per-modality encoder outputs, broadcast over modalities, vs. at the fusion hub output).

The `is_unimodal` flag in `_get_signal` is eliminated — both paths use the exact same repeat-and-slice logic with the same `target_hidden_size`.

**Applies to both** `NeuromodulatorRNN` (RecurrentPPO) and `DreamerNeuromodulatorRNN` (DreamerV3).

### File Changes

#### `src/models/neuromodulator.py` — `NeuromodulatorRNN.__init__()` (lines 80–119)

```python
# BEFORE (line 80):
self.num_groups_unimodal = len(obs_breakdown)

# AFTER — use the same group count as multimodal:
self.num_groups_unimodal = self.num_groups_hidden  # = math.ceil(target_hidden_size / grouping_size)
```

Note: `self.num_groups_hidden` is already computed on line 81. The unimodal head now has the same output dimension as the multimodal head.

The `head_unimodal` Linear layer (lines 89-90) and `head_unimodal_add` (lines 92-93) don't need code changes — they already reference `self.num_groups_unimodal`, which now equals `self.num_groups_hidden`.

```python
# BEFORE (line 113):
self.z_unimodal_baseline = nnx.Param(jnp.zeros(self.num_groups_unimodal))

# AFTER — per-neuron baseline, same shape as z_hidden_baseline:
self.z_unimodal_baseline = nnx.Param(jnp.zeros(target_hidden_size))
```

```python
# BEFORE (line 118):
self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(self.num_groups_unimodal))

# AFTER:
self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))
```

#### `src/models/neuromodulator.py` — `NeuromodulatorRNN.__call__()` `_get_signal` (lines 137–164)

Eliminate the `is_unimodal` flag — both unimodal and multimodal now use the same code path:

```python
# BEFORE (lines 137-153):
def _get_signal(head, baseline, head_add=None, baseline_add=None, is_unimodal=False):
    raw = head(h_mod_new)
    if is_unimodal:
        sig = baseline.value + raw
    else:
        sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
        sig = baseline.value + sig

    if head_add is not None:
        raw_add = head_add(h_mod_new)
        if is_unimodal:
            sig_add = baseline_add.value + raw_add
        else:
            sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
            sig_add = baseline_add.value + sig_add
        return sig, sig_add
    return sig, jnp.zeros_like(sig)

# AFTER — unified path, no is_unimodal branching:
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

```python
# BEFORE (lines 155-157):
z_uni, z_uni_add = _get_signal(self.head_unimodal, self.z_unimodal_baseline,
                              getattr(self, 'head_unimodal_add', None),
                              getattr(self, 'z_unimodal_add_baseline', None), is_unimodal=True)

# AFTER — remove is_unimodal=True:
z_uni, z_uni_add = _get_signal(self.head_unimodal, self.z_unimodal_baseline,
                              getattr(self, 'head_unimodal_add', None),
                              getattr(self, 'z_unimodal_add_baseline', None))
```

The multimodal and memory calls (lines 159-164) remain unchanged — they already don't pass `is_unimodal`.

#### `src/models/neuromodulator.py` — `DreamerNeuromodulatorRNN.__init__()` (lines 240–293)

Apply the same changes:

```python
# BEFORE (line 246):
self.num_groups_unimodal = len(obs_breakdown)

# AFTER — use the same group count as the percept head:
self.num_groups_unimodal = self.num_groups_percept  # = math.ceil(embed_dim / grouping_size)
```

```python
# BEFORE (line 287):
self.z_unimodal_baseline = nnx.Param(jnp.zeros(self.num_groups_unimodal))

# AFTER:
self.z_unimodal_baseline = nnx.Param(jnp.zeros(embed_dim))
```

```python
# BEFORE (line 292):
self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(self.num_groups_unimodal))

# AFTER:
self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(embed_dim))
```

#### `src/models/neuromodulator.py` — `DreamerNeuromodulatorRNN._compute_heads()` (lines 295–337)

Eliminate the `is_unimodal` flag from `_get_signal` and update its callers:

```python
# BEFORE (lines 299-320):
def _get_signal(head, baseline, head_add=None, baseline_add=None, target_dim=None, is_unimodal=False):
    if not include_percept and (is_unimodal or head == self.head_multimodal):
        dim = self.num_groups_unimodal if is_unimodal else target_dim
        return jnp.zeros(h_mod.shape[:-1] + (dim,)), jnp.zeros(h_mod.shape[:-1] + (dim,))

    raw = head(h_mod)
    if is_unimodal:
        sig = baseline.value + raw
    else:
        sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :target_dim]
        sig = baseline.value + sig

    if head_add is not None:
        raw_add = head_add(h_mod)
        if is_unimodal:
            sig_add = baseline_add.value + raw_add
        else:
            sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :target_dim]
            sig_add = baseline_add.value + sig_add
        return sig, sig_add
    return sig, jnp.zeros_like(sig)

# AFTER — unified path:
def _get_signal(head, baseline, head_add=None, baseline_add=None, target_dim=None, is_percept=False):
    if not include_percept and is_percept:
        return jnp.zeros(h_mod.shape[:-1] + (target_dim,)), jnp.zeros(h_mod.shape[:-1] + (target_dim,))

    raw = head(h_mod)
    sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :target_dim]
    sig = baseline.value + sig

    if head_add is not None:
        raw_add = head_add(h_mod)
        sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :target_dim]
        sig_add = baseline_add.value + sig_add
        return sig, sig_add
    return sig, jnp.zeros_like(sig)
```

Note: `is_unimodal` is replaced with `is_percept` — this flag now only controls whether to zero-out during imagination mode (both unimodal and multimodal are perceptual heads that should be zeroed during imagination).

```python
# BEFORE (lines 322-324):
z_uni, z_uni_add = _get_signal(self.head_unimodal, self.z_unimodal_baseline,
                              getattr(self, 'head_unimodal_add', None),
                              getattr(self, 'z_unimodal_add_baseline', None), is_unimodal=True)

# AFTER:
z_uni, z_uni_add = _get_signal(self.head_unimodal, self.z_unimodal_baseline,
                              getattr(self, 'head_unimodal_add', None),
                              getattr(self, 'z_unimodal_add_baseline', None),
                              target_dim=self.embed_dim, is_percept=True)
```

The multimodal call (lines 326-328) adds `is_percept=True` (it was already using `head == self.head_multimodal` to detect this, which is now replaced by the flag):

```python
# BEFORE (lines 326-328):
z_multi, z_multi_add = _get_signal(self.head_multimodal, self.z_hidden_baseline,
                                  getattr(self, 'head_multimodal_add', None),
                                  getattr(self, 'z_hidden_add_baseline', None), target_dim=self.embed_dim)

# AFTER:
z_multi, z_multi_add = _get_signal(self.head_multimodal, self.z_hidden_baseline,
                                  getattr(self, 'head_multimodal_add', None),
                                  getattr(self, 'z_hidden_add_baseline', None),
                                  target_dim=self.embed_dim, is_percept=True)
```

The memory call (line 330) remains unchanged (`is_percept` defaults to `False`).

#### `src/models/recurrent_ppo_network.py` — `forward_with_modulation()` (lines 139–151)

The unimodal signal shape changes from `(batch, 9)` to `(batch, 128)`. At injection, it must broadcast over the 9 modalities: `(batch, 1, 128) × (batch, 9, 128)`. Change `[..., None]` (which broadcast over neuron dim) to `[..., None, :]` (which broadcasts over modality dim):

```python
# BEFORE (lines 139-144):
# Phase 1: Unimodal + Modulation (z_unimodal)
encoded_all = self.unimodal_grouped(x_padded)
gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)
beta1 = mod_output.z_unimodal_add
# Apply per-group modulation
encoded_all = jax.nn.relu(encoded_all * gamma1[..., None] + beta1[..., None])

# AFTER — broadcast over modality dim instead of neuron dim:
# Phase 1: Unimodal + Modulation (z_unimodal)
encoded_all = self.unimodal_grouped(x_padded)              # (batch, 9, 128)
gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)              # (batch, 128)
beta1 = mod_output.z_unimodal_add                           # (batch, 128)
# Broadcast (batch, 1, 128) × (batch, 9, 128) — same gate pattern for all modalities
encoded_all = jax.nn.relu(encoded_all * gamma1[..., None, :] + beta1[..., None, :])
```

#### `src/models/recurrent_ppo_network.py` — `forward_with_modulation()` flat fallback (lines 123–130)

The unimodal signal is now `(batch, 128)` — same shape as multimodal. The flat fallback can use it directly:

```python
# BEFORE (lines 123-130):
if self.mode != 'hierarchical':
    x_proj = self.monolith(x)
    if modulation_type == "PreActivation":
        gamma = jax.nn.sigmoid(mod_output.z_unimodal[..., 0:1])
        beta = mod_output.z_unimodal_add[..., 0:1]
        return jax.nn.relu(x_proj * gamma + beta)
    else:
        return jax.nn.relu(x_proj) * jax.nn.sigmoid(mod_output.z_unimodal[..., 0:1])

# AFTER — z_unimodal is now (batch, 128), can apply directly:
if self.mode != 'hierarchical':
    x_proj = self.monolith(x)
    if modulation_type == "PreActivation":
        gamma = jax.nn.sigmoid(mod_output.z_unimodal)
        beta = mod_output.z_unimodal_add
        return jax.nn.relu(x_proj * gamma + beta)
    else:
        return jax.nn.relu(x_proj) * jax.nn.sigmoid(mod_output.z_unimodal)
```

#### DreamerV3 encoder (if applicable)

The implementing agent should search for all consumers of `mod_output.z_unimodal` across the codebase (grep for `z_unimodal`) and verify each handles the new `(batch, 128)` shape correctly. If a DreamerV3 encoder has a similar `forward_with_modulation`, the same `[..., None]` → `[..., None, :]` change applies.

### Shape Summary

With `grouping_size=64`, `hidden_size=128`, 9 modalities:

| Signal | Before | After |
|--------|--------|-------|
| `num_groups_unimodal` | 9 (= num modalities) | 2 (= `ceil(128/64)`, same as multimodal) |
| `head_unimodal` output | `(batch, 9)` | `(batch, 2)` |
| `z_unimodal` (after expand) | `(batch, 9)` | `(batch, 128)` |
| `z_unimodal_baseline` | `(9,)` | `(128,)` |
| Injection broadcast | `(batch, 9, 1) × (batch, 9, 128)` | `(batch, 1, 128) × (batch, 9, 128)` |
| `head_multimodal` output | `(batch, 2)` | `(batch, 2)` — unchanged |
| `z_multimodal` (after expand) | `(batch, 128)` | `(batch, 128)` — unchanged |

### Config Changes

No new config keys needed. The existing `grouping_size` parameter now controls both unimodal and multimodal grouping as originally intended.

### Parameter Impact

With `grouping_size=64, mod_hidden_size=16, hidden_size=128`:

| Component | Before | After | Delta |
|-----------|--------|-------|-------|
| `head_unimodal` weights | 16×9 + 9 = 153 | 16×2 + 2 = 34 | -119 |
| `z_unimodal_baseline` | 9 | 128 | +119 |
| `head_unimodal_add` (PreAct only) | 153 | 34 | -119 |
| `z_unimodal_add_baseline` (PreAct only) | 9 | 128 | +119 |
| **Total (Multiplicative)** | **162** | **162** | **0** |
| **Total (PreActivation)** | **324** | **324** | **0** |

Net parameter count is unchanged — the head shrinks (fewer outputs) while the baseline grows (per-neuron instead of per-group).

## Checkpoints

- [ ] Checkpoint 1 — After modifying `NeuromodulatorRNN.__init__()`, print `self.num_groups_unimodal` and `self.num_groups_hidden` to confirm they are equal (both 2 with G=64, hidden_size=128).
- [ ] Checkpoint 2 — After modifying `_get_signal()`, print `z_uni.shape` in `__call__()` to confirm it is `(batch, 128)` not `(batch, 9)`.
- [ ] Checkpoint 3 — In `forward_with_modulation()`, assert `gamma1.shape[-1] == encoded_all.shape[-1]` (both 128) to verify broadcast alignment.
- [ ] Checkpoint 4 — Run a single training step (`python -m src.train --max_steps 10`) and verify no shape errors occur.
- [ ] Checkpoint 5 — Search codebase for all consumers of `mod_output.z_unimodal` (grep for `z_unimodal`) and verify each handles the new `(batch, 128)` shape correctly.
- [ ] Checkpoint 6 — If DreamerV3 encoder exists, verify the same shape change is propagated there.

## Implementation Report

> **Implemented by**:
> **Date**:

## Verification Report

> **Verified by**:
> **Date**:

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/neuromodulator.py` | Unified unimodal grouping (NeuromodulatorRNN) | | |
| `src/models/neuromodulator.py` | Unified unimodal grouping (DreamerNeuromodulatorRNN) | | |
| `src/models/recurrent_ppo_network.py` | Change broadcast axis, update flat fallback | | |

**Conclusion**:
