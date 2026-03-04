# Perceptual Noise Debugging Plan

> **Status**: COMPLETED — sensory modality order fixed, verified.
> **Opened**: 2026-03-04
> **Trigger**: Observed large discrepancy between true and noised nutrition values
> **Root Cause**: Index mismatch — noise arrays in `config_loader.py` built in wrong order relative to `modality_map` in `sensor.py`

---

## Executive Summary

There is a **confirmed systematic bug**: the noise parameter arrays in `config_loader.py` (lines 320–374) are built in a different modality order than the `modality_map` used by `sensor.py:apply_perceptual_noise()`. This causes every sensor modality (except Visual, which is correct by accident) to receive the wrong noise parameters. The fix makes the **YAML config the single source of truth** — both `config_loader.py` and `sensor.py` read the modality order from `default.yaml`.

---

## Root Cause: The Index Mismatch

### Canonical modality order — defined in `default.yaml`, consumed by `sensor.py`

This is the intended index assignment (will be defined by YAML key order after the fix):

| Index | Modality |
|:-----:|----------|
| 0 | Injury |
| 1 | Nutrition |
| 2 | Satiation |
| 3 | Extero Nociception |
| 4 | Olfaction |
| 5 | Collision |
| 6 | Proprioception |
| 7 | Visual |
| 8 | Location |

### Current (wrong) ordering in `config_loader.py` (lines 320–374)

The arrays are built in this mismatched order:

| Array index | What `config_loader` puts here | What `modality_map` expects |
|:-----------:|-------------------------------|---------------------------|
| 0 | olfaction (σ=0.15) | **Injury** (σ=0.05, state_dep) |
| 1 | extero_nociception (σ=0.01) | **Nutrition** (σ=0.10) |
| 2 | collision (σ=0.01) | **Satiation** (σ=0.10) |
| 3 | location (σ=0.01) | **Extero Nociception** (σ=0.01) |
| 4 | satiation (σ=0.10) | **Olfaction** (σ=0.15) |
| 5 | nutrition (σ=0.10) | **Collision** (σ=0.01) |
| 6 | injury (σ=0.05, state_dep) | **Proprioception** (σ=0.05) |
| 7 | visual (σ=0.05) | **Visual** (σ=0.05) ← correct! |
| 8 | proprioception (σ=0.05) | **Location** (σ=0.01) |

---

## Full Impact: Intended vs Actual Noise Per Sensor

| Sensor | Intended σ | **Actual σ** | Intended mode | **Actual mode** | Intended clip | **Actual clip** |
|--------|:----------:|:------------:|:-------------:|:---------------:|:-------------:|:---------------:|
| Injury | 0.05 | **0.15** (+3×) | state_dependent | **constant** ← lost! | [0,1] | **[0,100]** |
| Nutrition | 0.10 | **0.01** (÷10) | constant | constant | [0,1] | **[0,100]** |
| Satiation | 0.10 | **0.01** (÷10) | constant | constant | [0,1] | [0,1] OK |
| Extero Noc | 0.01 | **0.01** OK σ | constant | constant | [0,100] | **[-1,1]** ← wrong! |
| Olfaction | 0.15 | **0.10** | constant | constant | [0,100] | **[0,1]** ← clips signals! |
| Collision | 0.01 | **0.10** (+10×) | constant | constant | [0,1] | [0,1] OK |
| Proprioception | 0.05 | **0.05** OK σ | constant | **state_dependent** ← wrong! | [0,1] | [0,1] OK |
| Visual | 0.05 | **0.05** ✓ | constant | constant ✓ | [0,100] | [0,100] ✓ |
| Location | 0.01 | **0.05** (+5×) | constant | constant | [-1,1] | **[0,1]** ← wrong! |

### Most Impactful Consequences

1. **Olfaction clipped at 1.0** instead of 100.0 — Any strong chemical signal > 1.0 is silently clamped.
2. **Injury gets 3× too much noise and loses state-dependent mode** — Injury perception noisier-when-injured design is broken.
3. **Nutrition and Satiation get 10× less noise** — Nearly noiseless (σ=0.01) instead of intended σ=0.10.
4. **Collision gets 10× too much noise** — Binary obstacle detection (0/1) gets σ=0.10.
5. **Proprioception becomes state-dependent** — Last-action one-hot noise scaled by injury level.
6. **Extero Nociception clipped to [-1, 1]** — Pain sensor values can go negative.
7. **Location clipped to [0, 1]** instead of [-1, 1] — Negative normalized coords clipped to 0.

---

## Implementation Plan: YAML-Driven Ordering

### Design Goal

Make the **YAML config the single source of truth** for modality ordering. No hardcoded integer-to-modality mapping that can go out of sync.

### Data Flow

```
YAML config (defines canonical modality order via key order in perceptual_noise.modalities)
    ↓
config_loader.py (reads YAML keys in order → builds noise arrays → stores order in EnvParams)
    ↓
sensor.py (reads order from params.noise_modality_order → builds modality_map dynamically)
    ↓
apply_perceptual_noise (uses modality_map to index into noise arrays)
```

- **Config is the authority** — the YAML key order defines the index mapping
- **`sensor.py` is a consumer** — it reads the order from `params`, never hardcodes it
- **Noise system reads from `params`** — most efficient, no extra imports needed

**How it works**:
- PyYAML preserves dict insertion order (Python 3.7+)
- `config_loader.py` reads the YAML `modalities` keys in order, builds noise arrays in that order
- The modality order is stored as a tuple in `EnvParams.noise_modality_order`
- `sensor.py` builds `modality_map` dynamically from `params.noise_modality_order`

**Why `Config.merge()` is safe**: `deep_update` in `config.py` updates existing dict keys in-place. In Python 3.7+, updating an existing key preserves its position. So the YAML key order from `default.yaml` survives config merges.

### Files to Modify (4 files)

#### 1. `configs/environment/default.yaml`

Reorder `perceptual_noise.modalities` keys to match canonical order. Add a comment explaining the contract:

```yaml
perceptual_noise:
  enabled: true
  # Key order here is the single source of truth for noise array indices (0–8).
  # config_loader reads this order; sensor.py consumes it via params.noise_modality_order.
  modalities:
    injury:          # index 0
      mode: "state_dependent"
      sigma: 0.05
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 1.0
    nutrition:       # index 1
      mode: "constant"
      sigma: 0.1
      ...
    satiation:       # index 2
    extero_nociception:  # index 3
    olfaction:       # index 4
    collision:       # index 5
    proprioception:  # index 6
    visual:          # index 7
    location:        # index 8
```

Values (sigma, mode, clip) remain unchanged — only the key order changes.

#### 2. `src/environment/state.py`

Add `noise_modality_order` field to `EnvParams`:

```python
noise_modality_order: tuple = struct.field(pytree_node=False)  # e.g. ("Injury","Nutrition",...)
```

**Note**: This field was already added in a partial edit. Verify it's present.

#### 3. `src/environment/config_loader.py` (lines 317–375)

**Add** a name mapping at module level:

```python
_YAML_KEY_TO_SENSOR_NAME = {
    "injury":              "Injury",
    "nutrition":           "Nutrition",
    "satiation":           "Satiation",
    "extero_nociception":  "Extero Nociception",
    "olfaction":           "Olfaction",
    "collision":           "Collision",
    "proprioception":      "Proprioception",
    "visual":              "Visual",
    "location":            "Location",
}
```

**Replace** the 5 hardcoded noise arrays with YAML-order-driven construction:

```python
modalities_cfg = config.get('perceptual_noise.modalities') or {}

def _parse_mode(s):
    return 2 if s == 'state_dependent' else 1 if s == 'constant' else 0

noise_modality_order = tuple(
    _YAML_KEY_TO_SENSOR_NAME[k]
    for k in modalities_cfg
    if k in _YAML_KEY_TO_SENSOR_NAME
)
pad = max(0, 12 - len(noise_modality_order))

noise_modes     = jnp.pad(jnp.array([_parse_mode(modalities_cfg[k].get('mode','none'))
                    for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME], dtype=jnp.int32), (0, pad))
noise_sigmas    = jnp.pad(jnp.array([modalities_cfg[k].get('sigma', 0.0)
                    for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME], dtype=jnp.float32), (0, pad))
noise_injury_scales = jnp.pad(jnp.array([modalities_cfg[k].get('injury_noise_scale', 0.0)
                    for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME], dtype=jnp.float32), (0, pad))
noise_clip_min  = jnp.pad(jnp.array([modalities_cfg[k].get('clip_min', -100.0)
                    for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME], dtype=jnp.float32), (0, pad))
noise_clip_max  = jnp.pad(jnp.array([modalities_cfg[k].get('clip_max', 100.0)
                    for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME], dtype=jnp.float32), (0, pad))
```

**Pass** `noise_modality_order=noise_modality_order` into the `EnvParams(...)` constructor.

#### 4. `src/environment/sensor.py` (lines 204–216)

Replace the hardcoded `modality_map` dict:

```python
# BEFORE (hardcoded):
modality_map = {
    "Injury": 0, "Nutrition": 1, "Satiation": 2,
    "Extero Nociception": 3, "Olfaction": 4, "Collision": 5,
    "Proprioception": 6, "Visual": 7, "Location": 8
}

# AFTER (derived from YAML order stored in params):
modality_map = {name: i for i, name in enumerate(params.noise_modality_order)}
```

No other changes to `sensor.py`. After this change, `sensor.py` has no hardcoded ordering — it reads entirely from config via `params`.

---

## Debugging Protocol

### Step 0: Confirm the Bug (Quick Sanity Check)

```python
import sys
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

from src.utils.config import Config
from src.environment.config_loader import load_env_params

config = Config()
config.load('configs/environment/default.yaml')
config.load('configs/train/default.yaml')
config.merge({'visualization': {'local_view_size': 3}})

params = load_env_params(config)

sensor_order = ["Injury","Nutrition","Satiation","Extero Nociception",
                "Olfaction","Collision","Proprioception","Visual","Location"]

print("=== WHAT SENSORS ACTUALLY RECEIVE (current) ===")
print(f"{'Sensor':<22} idx   sigma  mode  clip")
for i, name in enumerate(sensor_order):
    print(f"  {name:<22} {i}  ->  {float(params.noise_sigmas[i]):.3f}  {int(params.noise_modes[i])}     [{float(params.noise_clip_min[i]):.1f},{float(params.noise_clip_max[i]):.1f}]")

print()
print("Expected (from default.yaml):")
expected = {
    "Injury":             (0.05, 2, 0.0, 1.0),
    "Nutrition":          (0.10, 1, 0.0, 1.0),
    "Satiation":          (0.10, 1, 0.0, 1.0),
    "Extero Nociception": (0.01, 1, 0.0, 100.0),
    "Olfaction":          (0.15, 1, 0.0, 100.0),
    "Collision":          (0.01, 1, 0.0, 1.0),
    "Proprioception":     (0.05, 1, 0.0, 1.0),
    "Visual":             (0.05, 1, 0.0, 100.0),
    "Location":           (0.01, 1, -1.0, 1.0),
}
for name, (sig, mode, cmin, cmax) in expected.items():
    print(f"  {name:<22} sigma={sig:.2f}  mode={mode}  clip=[{cmin},{cmax}]")
```

### Step 1: Compare True vs Noised Per Modality

```python
import jax
import jax.numpy as jnp
from src.environment.core import jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown

key = jax.random.PRNGKey(42)
state, _ = jax_reset(key, params)

obs_clean = get_observation(state, params, apply_noise=False)
obs_noisy = get_observation(state, params, apply_noise=True)

breakdown = get_observation_breakdown(params)
start = 0
print("=== CLEAN vs NOISY OBSERVATION ===")
for name, dim in breakdown.items():
    end = start + dim
    clean_val = obs_clean[start:end]
    noisy_val = obs_noisy[start:end]
    delta = jnp.abs(noisy_val - clean_val)
    print(f"{name:<22} clean={jnp.round(clean_val,3)}  |delta|_max={float(jnp.max(delta)):.4f}")
    start = end

print()
print("=== RAW STATE (for reference) ===")
print(f"  state.nutrition:    {float(state.nutrition):.2f}  (obs = {float(state.nutrition/params.max_nutrition):.4f})")
print(f"  state.satiation:    {float(state.satiation):.2f}  (obs = {float(state.satiation/params.max_satiation):.4f})")
print(f"  state.injury_level: {float(state.injury_level):.2f}  (obs = {float(state.injury_level/params.max_injury):.4f})")
```

### Step 2: Verify the Fix

After implementation, rerun Step 0 and confirm:

```python
# Confirm noise_modality_order matches expected canonical order
assert list(params.noise_modality_order) == [
    "Injury","Nutrition","Satiation","Extero Nociception",
    "Olfaction","Collision","Proprioception","Visual","Location"
]

# Confirm per-sensor noise values are correct
assert abs(float(params.noise_sigmas[0]) - 0.05) < 1e-5   # Injury
assert abs(float(params.noise_sigmas[1]) - 0.10) < 1e-5   # Nutrition
assert abs(float(params.noise_sigmas[4]) - 0.15) < 1e-5   # Olfaction
assert int(params.noise_modes[0]) == 2                     # Injury = state_dependent
assert int(params.noise_modes[6]) == 1                     # Proprioception = constant
assert abs(float(params.noise_clip_max[4]) - 100.0) < 1e-5 # Olfaction unbounded
assert abs(float(params.noise_clip_min[8]) - (-1.0)) < 1e-5 # Location min = -1
```

---

## Why Nutrition Appears "Too Different"

Two layered causes:

1. **Normalization gap**: `state.nutrition` stores the raw value (0–100). `obs[1]` is `state.nutrition / max_nutrition` (0–1). Comparing raw state directly to obs shows ~100× difference regardless of noise.

2. **Wrong clip from the bug**: Nutrition's slot gets extero_noc's `clip_max=100.0` instead of `1.0`. After the fix, σ becomes 0.10 (intended) and the clip correctly constrains the observation to [0, 1].

---

## Secondary Issues to Check After Fix

### A. Sigma calibration for unbounded sensors

Log `mean(|obs_olfaction|)` and `std(noise_olfaction)` over a rollout. Target: noise_std / signal_mean ≈ 5–20% for meaningful perceptual uncertainty.

### B. State-dependent injury noise magnitude

Injury gets: `σ_eff = 0.05 × (1 + 1.5 × injury_norm)`.
- Uninjured: σ=0.05
- Fully injured: σ=0.125
Verify this is biologically intended (injury perception gets noisier as damage worsens).

### C. PRNG key correlation across parallel environments

`get_observation()` uses `obs_key = jax.random.fold_in(state.key, 999)`. Since `state.key` may be shared across the vmap batch between resets, all environments could get **identical obs noise** each step. Consider whether each env should receive an independent noise sample.

---

## Observation Scale Analysis

### Per-Modality Scale Ranges

| Modality | Dims | Raw Range | Source / Calculation |
|----------|:----:|-----------|---------------------|
| **Injury** | 1 | [0, 1] | `state.injury_level / max_injury` (normalized in `get_observation`) |
| **Nutrition** | 1 | [0, 1] | `state.nutrition / max_nutrition` (normalized in `get_observation`) |
| **Satiation** | 1 | [0, 1] | `state.satiation / max_satiation` (normalized in `get_observation`) |
| **Extero Nociception** | 1 | [0, 0.9] | `max(res_nociception, pred_nociception, obs_nociception, collision_noc)`. Max configured intensity = 0.9 (danger/predator) |
| **Olfaction** | 5 | [0, ~40] | **Unbounded aggregate**. `sum(property * decay * mask)` per channel. Worst case: bush channel = 20 bushes × 2.0 on-entity decay = **40.0** |
| **Collision** | 5 | [0, 1] | Binary per cell (OOB or blocked). 5 cells at `sensor_range=1` |
| **Proprioception** | 6 | [0, 1] | One-hot of last action. Exactly one element = 1.0, rest = 0.0 |
| **Visual** | 8 | [0, ~13] | Per-channel entity count at agent's cell (`visual_sensor_range=0` → 1 cell). Background one-hot max = 1.0. Rock channel: 12 rocks can stack → **13.0**. Food: 4 max. When `visual_sensor_range > 0`, scales proportionally with cells |
| **Location** | 2 | [-1, 1] | `(pos / (dim-1)) * 2 - 1`. Symmetric around 0 |

### Scale Imbalance Detail

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

Most modalities sit in [0, 1]. Two outliers — **Olfaction (up to 40×)** and **Visual (up to 13×)** — dominate the input space.

### Impact on Training

#### 1. Gradient Magnitude Imbalance

In a flat (monolithic) encoder, the gradient with respect to weight `W_ij` scales with `∂L/∂W_ij ∝ x_i`. When olfaction values are 40× larger than injury values, the corresponding weights receive 40× larger gradients. This causes:
- Olfaction and visual features are learned faster and dominate the representation
- Interoceptive signals (injury, nutrition, satiation) — the core homeostatic signals — are learned slowly and may be under-represented

#### 2. GroupedMLP Partially Mitigates This

The hierarchical encoder (`GroupedMLP` in `recurrent_ppo_network.py`) processes each modality through its own sub-MLP before the multimodal hub. This **isolates gradient flow within each modality's weights**. However:

- **Within each sub-MLP**: Weights are initialized at `* 0.1` scale, which is tuned for [0,1] inputs. For olfaction (max 40), initial forward activations will be ~4× the intended magnitude, potentially saturating ReLU paths or causing large initial gradients.
- **At the multimodal hub**: All sub-MLP outputs are concatenated and fed into the hub MLP. If olfaction's sub-MLP produces larger activations (due to larger inputs), it still dominates the hub's input, recreating the scale imbalance at the integration layer.

#### 3. Noise-to-Signal Ratio (SNR) Inconsistency

The noise sigmas are specified as absolute values, not relative to signal scale:

| Modality | Signal Range | Noise σ | SNR (σ / max) |
|----------|:-----------:|:-------:|:-------------:|
| Injury | 1.0 | 0.05 | 5% |
| Nutrition | 1.0 | 0.10 | 10% |
| Satiation | 1.0 | 0.10 | 10% |
| Extero Noc | 0.9 | 0.01 | ~1% |
| **Olfaction** | **40.0** | **0.15** | **0.4%** ← near-zero noise |
| Collision | 1.0 | 0.01 | 1% |
| Proprioception | 1.0 | 0.05 | 5% |
| **Visual** | **13.0** | **0.05** | **0.4%** ← near-zero noise |
| Location | 1.0 | 0.01 | 1% |

**Olfaction and visual are effectively noiseless** relative to their signal magnitude. The perceptual uncertainty design assumes [0,1]-range signals. If the goal is to model noisy perception, these sensors need proportionally larger σ values.

#### 4. Satiation ≡ Nutrition Redundancy

With `nutrition_to_satiation_scaling_factor = 1.0` (line 219 in default.yaml):
```
satiation = max_satiation * (nutrition / max_nutrition) ^ 1.0
         = max_satiation * nutrition / max_nutrition
```

Since both are normalized by their max in the observation, the agent sees `obs_satiation = obs_nutrition` identically. This is a redundant input dimension.

### Recommendations & Resolution

#### R1: Symlog Compression (IMPLEMENTED)

Applied `symlog(x) = sign(x) * log(|x| + 1)` to olfaction and visual in `get_observation()` (`sensor.py`). This compresses unbounded modalities while preserving discrimination:

| Modality | Raw Max | Symlog Max | Relative (was) |
|----------|:-------:|:----------:|:--------------:|
| Olfaction | 40.0 | 3.71 | 3.7× (was 40×) |
| Visual | 13.0 | 2.64 | 2.6× (was 13×) |

- **DreamerV3**: already applies global `symlog(batch['obs'])`, so these get double-symlog'd → olfaction max ≈ 1.55, gap ≈ 2.2×. Acceptable.
- **PPO**: no existing symlog, benefits most from this fix.
- **Noise SNR**: olfaction improves from 0.4% → ~4% without changing σ values.

#### R2: Scale-Aware Noise Sigmas (Deferred)
After symlog (R1), noise σ values are more meaningful relative to compressed range. Current values are reasonable. Revisit if noise experiments show further tuning needed.

#### R3: Satiation Scaling Factor (Open)
With `nutrition_to_satiation_scaling_factor = 1.0`, satiation ≡ nutrition in observation space. Consider changing to 2.0+ for non-linear relationship, or document as intentionally redundant.

---

## Implementation Verification Report

> **Verified**: 2026-03-04

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/environment/default.yaml` | Reorder modality keys to canonical order | ✅ | Contract comment at lines 260–261 |
| `src/environment/state.py` | Add `noise_modality_order` tuple field | ✅ | `pytree_node=False`, line 161 |
| `src/environment/config_loader.py` | Replace hardcoded arrays with `_parse_noise_config()` | ✅ | YAML-order-driven loop, lines 334–379 |
| `src/environment/sensor.py` | Replace hardcoded `modality_map` with dynamic derivation | ✅ | `{name: i for i, name in enumerate(params.noise_modality_order)}`, line 206 |
| `src/environment/sensor.py` | R1: Symlog compression on olfaction/visual | ❌ | Not yet implemented (lines 270, 281 still raw) |
| `src/environment/sensor.py` | Asymmetric guard in clip construction | ⚠️ | Non-blocking. Lines 237–238 have `if name in modality_map` guard but sigma/mode loop (213–217) does not. Recommend making consistent. |

**Conclusion**: Noise modality ordering fix is correctly implemented. Every sensor now receives its intended noise parameters.
