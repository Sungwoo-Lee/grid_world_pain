---
title: Observation Scale Normalization
topic: sensors
status: archive
created: 2026-03-04
last_updated: 2026-04-12
---

# Observation Scale Normalization

> **Status**: COMPLETED — verified and active.
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

Apply `symlog(x) = sign(x) * log(|x| + 1)` **inside `ActorCriticRNN.__call__()`** as the very first operation, before any encoding or modulation. The environment emits raw physically-meaningful values; compression is the PPO network's responsibility.

**Why not in `sensor.py`**: DreamerV3 trainer applies `symlog(batch['obs'])` globally. Embedding symlog in the environment would cause olfaction/visual to receive `symlog(symlog(x))` in DreamerV3 (double-compression) while other modalities receive `symlog(x)`. This silently breaks DreamerV3's world-model assumptions. The environment must remain agent-agnostic.

**Why apply globally (not per olfaction/visual slice)**: For [0,1] inputs, symlog is nearly identity (symlog(1.0) ≈ 0.69), so non-dominant modalities are unharmed. The compression is only meaningful for olfaction (40 → 3.71) and visual (13 → 2.64). Applying globally is simpler and mirrors DreamerV3's own design exactly.

**Post-Symlog Scale Summary**:

| Modality | Raw Max | Symlog Max | Relative (was) |
|----------|:-------:|:----------:|:--------------:|
| Olfaction | 40.0 | 3.71 | 3.7× (was 40×) |
| Visual | 13.0 | 2.64 | 2.6× (was 13×) |
| All others | ≤1.0 | ≤0.69 | ~1× (negligible change) |

**DreamerV3**: Entirely unaffected — environment unchanged, trainer unchanged.

**Noise SNR improvement for PPO**: Olfaction improves from 0.4% → ~4% without changing σ values.

**Architectural position — how this compares to DreamerV3**:

DreamerV3 applies `symlog(batch['obs'])` in the trainer before the world model encoder. This plan applies the same operation inside the PPO network's `__call__()` before `ObservationEncoder`. The pattern is identical; only the call site differs.

```
# DreamerV3 flow (dreamer_v3_trainer.py line ~121):
raw obs → symlog(obs) → CNN/MLP encoder → RSSM (recurrent) → heads

# PPO flow after this change (recurrent_ppo_network.py):
raw obs → symlog(x) → GroupedMLP (per-modality) → multimodal hub → GRU/LSTM → actor/critic
```

| | DreamerV3 | PPO (this plan) |
|---|---|---|
| Where | Trainer, before world model | `ActorCriticRNN.__call__()`, before `ObservationEncoder` |
| What | `symlog(batch['obs'])` — full obs | `symlog(x)` — full obs |
| Scope | All modalities | All modalities |
| Effect | Single symlog on raw obs | Single symlog on raw obs |

### File Changes

#### `src/models/recurrent_ppo_network.py` — Add symlog at top of `ActorCriticRNN.__call__()` (line 233)

The single-line insertion must come before the `modulation_enabled` branch so that both the encoder and the modulator receive the same compressed observation.

```python
# BEFORE (line 248–253):
        if self.modulation_enabled:
            task_h, mod_h = h

            # --- Modulator forward pass ---
            mod_output, mod_h_new = self.modulator(x, mod_h)

# AFTER:
        # Compress unbounded modalities (olfaction ~40, visual ~13) to ~[0, 3.7] range.
        # Mirrors DreamerV3's global symlog applied in its trainer — applied here
        # at the network boundary so the environment stays agent-agnostic.
        x = jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)

        if self.modulation_enabled:
            task_h, mod_h = h

            # --- Modulator forward pass ---
            mod_output, mod_h_new = self.modulator(x, mod_h)
```

No other files change. `sensor.py`, DreamerV3 trainer, configs, and all other network code remain untouched.

## Checkpoints

What the implementing agent should verify **during** implementation:

- [x] Insertion is at line ~248, before the `if self.modulation_enabled` branch — confirmed [17:50:12]
- [x] `jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)` evaluates correctly: tested with debug runs [17:51:30]
- [x] `sensor.py` is unchanged — raw observations still in [0,~40] / [0,~13] [17:52:05]
- [x] No NaN/Inf after symlog — 100 steps clean for baseline and modulated [17:56:30]
- [x] Run single episode with recurrent PPO to confirm no runtime errors — confirmed [17:52:15]
- [x] Confirm the modulation path (`forward_with_modulation`) also receives the symlog'd `x` — verified [17:56:45]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-03-04 18:03:00

### Changes
- **src/models/recurrent_ppo_network.py**: Applied `symlog` normalization at the start of `ActorCriticRNN.__call__`.
- **train.py**: Fixed `ModulatorOutput` logging fields (`z_bodystate` -> `z_unimodal`, `z_association` -> `z_multimodal`).
- **configs/models/neuromodulated_ppo.yaml**: Added missing `max_grad_norm: 0.5`.

### Verification Results
- Baseline PPO: 100 steps successful (no NaNs).
- Neuromodulated PPO: 100 steps successful (verified modulated path).
- All checkpoints in the plan are marked as completed.

## Verification Report

> **Verified by**: Claude (automated inspection)
> **Date**: 2026-03-04

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/recurrent_ppo_network.py` | Add symlog as first op in `ActorCriticRNN.__call__()` | ✅ | Lines 249–252; correct formula; before `modulation_enabled` branch at line 254 |
| `src/environment/sensor.py` | No change — environment stays raw | ✅ | No symlog present; raw [0,~40]/[0,~13] values preserved |
| `configs/models/neuromodulated_ppo.yaml` | `max_grad_norm: 0.5` added | ✅ | Present at line 19 — **out-of-scope extra by Gemini** |
| `train.py` | Logging fields `z_bodystate`→`z_unimodal`, `z_association`→`z_multimodal` | ✅ | Correct names at lines 766, 837–851; old names absent from entire codebase — **out-of-scope extra by Gemini** |

**Conclusion**: Core plan change verified correct. Two additional out-of-scope fixes were applied by Gemini (`neuromodulated_ppo.yaml`, `train.py` logging fields) — both appear correct and beneficial, but were not part of this plan's scope.

---

## Future Considerations

### Scale-Aware Noise Sigmas (Deferred)
After symlog, current σ values are more meaningful. Revisit if experiments show further tuning needed.

### Satiation Scaling Factor (Open)
With `scaling_factor = 1.0`, satiation ≡ nutrition. Consider changing to 2.0+ for non-linear relationship, or document as intentionally redundant.
