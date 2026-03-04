# Observation Scale Normalization

> **Status**: PLANNED — not yet implemented.
> **Opened**: 2026-03-04
> **Related**: [NOISE_DEBUGGING_PLAN.md](NOISE_DEBUGGING_PLAN.md) (independent issue)

---

## Context

Olfaction ([0, ~40]) and Visual ([0, ~13]) dominate all other modalities ([0, 1]), causing gradient imbalance, under-representation of interoceptive signals, and near-zero noise SNR. Discovered during observation analysis after the noise modality ordering fix.

## Analysis

### Per-Modality Scale Ranges

| Modality | Dims | Raw Range | Source / Calculation |
|----------|:----:|-----------|---------------------|
| **Injury** | 1 | [0, 1] | `state.injury_level / max_injury` (normalized) |
| **Nutrition** | 1 | [0, 1] | `state.nutrition / max_nutrition` (normalized) |
| **Satiation** | 1 | [0, 1] | `state.satiation / max_satiation` (normalized) |
| **Extero Nociception** | 1 | [0, 0.9] | Max configured intensity = 0.9 |
| **Olfaction** | 5 | [0, ~40] | Worst case: 20 bushes × 2.0 on-entity decay = 40.0 |
| **Collision** | 5 | [0, 1] | Binary per cell. 5 cells at `sensor_range=1` |
| **Proprioception** | 6 | [0, 1] | One-hot of last action |
| **Visual** | 8 | [0, ~13] | Rock channel: 12 rocks can stack → 13.0 |
| **Location** | 2 | [-1, 1] | Symmetric normalized coords |

```
Modality            Example max    Relative to [0,1]
─────────────────────────────────────────────────────
Injury                   1.0       1×   (baseline)
Nutrition                1.0       1×
Satiation                1.0       1×
Extero Nociception       0.9       ~1×
Collision                1.0       1×   (binary)
Proprioception           1.0       1×   (one-hot)
Location                 1.0       1×   (abs value)
─────────────────────────────────────────────────────
Olfaction               40.0       40×  ← DOMINANT
Visual                  13.0       13×  ← HIGH
```

### Impact on Training

**1. Gradient Magnitude Imbalance**
Gradient ∝ input magnitude. Olfaction weights get 40× larger gradients than injury weights. Interoceptive signals (the core homeostatic signals) are learned slowly and under-represented.

**2. GroupedMLP Partially Mitigates This**
Per-modality sub-MLPs (`recurrent_ppo_network.py`) isolate within-modality gradients, but the multimodal hub still sees unbalanced sub-MLP outputs. Weight init `* 0.1` is tuned for [0,1] inputs — olfaction (max 40) causes ~4× intended initial activations.

**3. Noise-to-Signal Ratio (SNR) Inconsistency**

| Modality | Signal Range | Noise σ | SNR (σ / max) |
|----------|:-----------:|:-------:|:-------------:|
| Injury | 1.0 | 0.05 | 5% |
| Nutrition | 1.0 | 0.10 | 10% |
| Satiation | 1.0 | 0.10 | 10% |
| Extero Noc | 0.9 | 0.01 | ~1% |
| **Olfaction** | **40.0** | **0.15** | **0.4%** ← near-zero |
| Collision | 1.0 | 0.01 | 1% |
| Proprioception | 1.0 | 0.05 | 5% |
| **Visual** | **13.0** | **0.05** | **0.4%** ← near-zero |
| Location | 1.0 | 0.01 | 1% |

**4. Satiation ≡ Nutrition Redundancy**
With `nutrition_to_satiation_scaling_factor = 1.0`: `obs_satiation = obs_nutrition` identically.

## Implementation Plan

### Design

Apply `symlog(x) = sign(x) * log(|x| + 1)` to olfaction and visual in `get_observation()`.

**Why symlog over [0,1] normalization**: Theoretical max ≠ practical max for olfaction/visual (config-dependent, scenario-dependent). Symlog compresses without needing known bounds, preserves discrimination at small values, and is already used by DreamerV3 globally.

**Post-Symlog Scale Summary**:

| Modality | Raw Max | Symlog Max | Relative (was) |
|----------|:-------:|:----------:|:--------------:|
| Olfaction | 40.0 | 3.71 | 3.7× (was 40×) |
| Visual | 13.0 | 2.64 | 2.6× (was 13×) |

**DreamerV3 double-symlog**: DreamerV3 trainer applies `symlog(batch['obs'])` globally (line 121). With this change, olfaction/visual get double-symlog'd: `symlog(symlog(40)) = symlog(3.71) ≈ 1.55`. Other modalities (already [0,1]): `symlog(1.0) ≈ 0.69`. Gap: 1.55 / 0.69 ≈ 2.2×. Acceptable — no changes to DreamerV3 trainer needed.

**PPO**: No existing symlog — benefits most from this fix.

**Noise SNR improvement**: Olfaction improves from 0.4% → ~4% without changing σ values.

### File Changes

#### `src/environment/sensor.py` — Add symlog helper (top of file)

```python
# AFTER (new function, add near top of file):
def _symlog(x):
    """Symmetric log compression for unbounded observations."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)
```

#### `src/environment/sensor.py` — Wrap olfaction output (line ~270)

```python
# BEFORE:
obs_parts.append(res_chem + pred_chem + obs_chem + neutral_chem)

# AFTER:
obs_parts.append(_symlog(res_chem + pred_chem + obs_chem + neutral_chem))
```

#### `src/environment/sensor.py` — Wrap visual output (line ~281)

```python
# BEFORE:
obs_parts.append(sense_visual(state.agent_pos, state, params))

# AFTER:
obs_parts.append(_symlog(sense_visual(state.agent_pos, state, params)))
```

No other files change. Trainers, configs, and network code remain untouched.

## Checkpoints

What the implementing agent should verify **during** implementation:

- [ ] `_symlog` function added and returns correct values: `_symlog(jnp.array(40.0))` ≈ 3.71, `_symlog(jnp.array(0.0))` = 0.0
- [ ] Olfaction slice of `get_observation(state, params, apply_noise=False)` has values in [0, ~3.7] not [0, ~40]
- [ ] Visual slice of `get_observation(state, params, apply_noise=False)` has values in [0, ~2.6] not [0, ~13]
- [ ] All other modality slices unchanged (injury, nutrition, satiation, collision, proprioception, location)
- [ ] No NaN/Inf in observation vector after symlog
- [ ] Run single episode to confirm no runtime errors

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the implementing agent after code changes are made. -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/sensor.py` | Add `_symlog()` helper | ❌ | Not implemented |
| `src/environment/sensor.py` | Wrap olfaction with `_symlog()` (line ~270) | ❌ | Not implemented |
| `src/environment/sensor.py` | Wrap visual with `_symlog()` (line ~281) | ❌ | Not implemented |

**Conclusion**: Implementation pending.

---

## Future Considerations

### Scale-Aware Noise Sigmas (Deferred)
After symlog, current σ values are more meaningful. Revisit if experiments show further tuning needed.

### Satiation Scaling Factor (Open)
With `scaling_factor = 1.0`, satiation ≡ nutrition. Consider changing to 2.0+ for non-linear relationship, or document as intentionally redundant.
