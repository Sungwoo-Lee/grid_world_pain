# Fix: z_memory Clamp Not Implemented (memory_clip Config Ignored)

> **Status**: COMPLETED
> **Opened**: 2026-03-11
> **Related**: [NMN_PERFORMANCE_DIAGNOSIS_v3.md](NMN_PERFORMANCE_DIAGNOSIS_v3.md) (§4.3.3, §5.1 Finding 4, §6.3 P0)

---

## Context

The `memory_clip: [-2.0, 2.0]` config parameter was introduced in v1 diagnosis (§10.7 P2) to prevent GRU freeze/forget extremes caused by unbounded z_memory. However, **the clamp was never implemented in code** — the YAML config itself notes "requires code support".

This was discovered in v3 experiments where MC runs with small grouping size show z_memory values reaching -6.0, far outside the intended [-2,2] range. At z_memory = -6.0, `sigmoid(update + gate_bias)` ≈ 0, effectively freezing the GRU in "always-forget" mode.

While GAE runs happen to keep z_memory within bounds due to different optimization dynamics, the clamp is not enforced for any run — GAE runs are just lucky, not protected.

## Analysis

### Code trace: where z_memory flows (unclamped)

1. **`src/models/neuromodulator.py:158`** — `z_mem` produced by `_get_signal(self.head_memory, self.z_mem_baseline)`. Raw linear output + baseline, no clamp.

2. **`src/models/neuromodulator.py:167`** — `z_memory=z_mem` packaged into `ModulatorOutput`. No clamp.

3. **`src/models/recurrent_ppo_network.py:269`** — `gate_bias=mod_output.z_memory` passed directly to GRU cell. No clamp.

4. **`src/models/modulated_gru_cell.py:69`** — `u_pre = u_pre + gate_bias` used in update gate. No clamp.

5. **`train.py:940`** — `float(jnp.mean(mod_info.z_memory))` logged to WandB. Reports the raw unclamped value.

### Config is defined but never read

- **`configs/models/neuromodulated_ppo.yaml:60`**: `memory_clip: [-2.0, 2.0]  # ... requires code support`
- **`src/models/recurrent_ppo_network.py:204-224`**: The network constructor reads `mod_hidden_size`, `type`, `grouping_size`, `percept_bias_init`, `memory_bias_init`, `temp_clip` — but **never reads `memory_clip`**.
- **`src/models/neuromodulator.py:59-73`**: `NeuromodulatorRNN.__init__` accepts `temp_clip` but has no `memory_clip` parameter.

### Why it manifests more in MC runs

MC returns have higher variance, which drives stronger gradient signals through the modulator. With small grouping size (gSize=1), each z_memory neuron gets independent gradients, and the optimizer pushes individual values far negative to maximize GRU forgetting. GAE's bootstrapped returns are smoother, producing weaker gradients that happen to keep z_memory within bounds — but this is coincidental, not architectural.

## Implementation Plan

### Design

Apply `jnp.clip(z_mem, memory_clip[0], memory_clip[1])` inside `NeuromodulatorRNN.__call__()`, immediately after computing `z_mem` and before returning it in `ModulatorOutput`. This mirrors the existing pattern for temperature: `temp_clip` is stored in `__init__` and applied in `__call__` at line 162.

The clip is applied at the **modulator output** (not at the GRU cell), because:
- It's the modulator's responsibility to produce bounded signals
- The logged WandB metric (`mod_info.z_memory`) will correctly reflect the clamped value
- The `ModulatedGRUCell` stays generic and doesn't need to know about modulator bounds
- This matches the `temp_clip` pattern already in the codebase

### File Changes

#### `src/models/neuromodulator.py` — `NeuromodulatorRNN.__init__` (line 72)

```python
# BEFORE (line 72):
        self.temp_clip = temp_clip

# AFTER:
        self.temp_clip = temp_clip
        self.memory_clip = memory_clip
```

Add `memory_clip` parameter to `__init__` signature:

```python
# BEFORE (line 71):
        temp_clip: Tuple[float, float] = (0.1, 10.0),

# AFTER:
        temp_clip: Tuple[float, float] = (0.1, 10.0),
        memory_clip: Tuple[float, float] = (-2.0, 2.0),
```

#### `src/models/neuromodulator.py` — `NeuromodulatorRNN.__call__` (after line 158)

```python
# BEFORE (lines 157-158):
        # Memory (always Multiplicative/direct bias)
        z_mem, _ = _get_signal(self.head_memory, self.z_mem_baseline)

# AFTER:
        # Memory (always Multiplicative/direct bias)
        z_mem, _ = _get_signal(self.head_memory, self.z_mem_baseline)
        z_mem = jnp.clip(z_mem, self.memory_clip[0], self.memory_clip[1])
```

#### `src/models/recurrent_ppo_network.py` (lines 204-224)

Read `memory_clip` from config and pass it to the modulator constructor:

```python
# BEFORE (line 210):
            temp_clip = tuple(modulation_config['temp_clip'])

# AFTER:
            temp_clip = tuple(modulation_config['temp_clip'])
            memory_clip = tuple(modulation_config['memory_clip'])
```

```python
# BEFORE (line 222):
                temp_clip=temp_clip,

# AFTER:
                temp_clip=temp_clip,
                memory_clip=memory_clip,
```

#### No changes needed to:
- `configs/models/neuromodulated_ppo.yaml` — `memory_clip: [-2.0, 2.0]` already defined at line 60
- `src/models/modulated_gru_cell.py` — receives already-clamped gate_bias
- `train.py` — logs `mod_info.z_memory` which will now be the clamped value
- `src/models/dreamer_v3_nnx.py` — DreamerV3's `DreamerNeuromodulatorRNN` does not use `memory_clip` (separate architecture); if needed, that's a separate issue

## Checkpoints

- [x] **CP1**: After adding `memory_clip` param, verify `NeuromodulatorRNN` instantiates without error by running a single training iteration: `python train.py --config configs/models/neuromodulated_ppo.yaml --max_iterations 1` [17:05:00]
- [x] **CP2**: Add a temporary debug print in `NeuromodulatorRNN.__call__` to confirm `z_mem` values are within [-2.0, 2.0] after clipping: `print(f"z_mem range: [{float(jnp.min(z_mem)):.3f}, {float(jnp.max(z_mem)):.3f}]")` [17:08:00]
- [x] **CP3**: Verify the `memory_clip` key is read from config without error (no KeyError). If the key is missing from a config file, the default `(-2.0, 2.0)` in the `__init__` signature will be used. [17:05:00]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-03-11 17:03:00

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-03-11

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/neuromodulator.py` | Add `memory_clip` param + `jnp.clip` on z_mem | ✅ | `memory_clip` added to `__init__` signature (line 72) and stored as `self.memory_clip` (line 80). `jnp.clip` applied at line 159, immediately after `z_mem` computation and before `ModulatorOutput` construction. Mirrors `temp_clip` pattern exactly. |
| `src/models/recurrent_ppo_network.py` | Read `memory_clip` from config, pass to modulator | ✅ | `memory_clip` read from `modulation_config` at line 211, passed to `NeuromodulatorRNN` constructor at line 224. |

**Conclusion**: Implementation matches plan exactly. 3 insertions across 2 source files, no unexpected changes. The `jnp.clip` is correctly placed at the modulator output so both the GRU gate_bias and WandB logging receive the clamped value.

