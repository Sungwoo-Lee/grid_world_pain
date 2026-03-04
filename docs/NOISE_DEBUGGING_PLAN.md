# Noise Modality Ordering Bug

> **Status**: COMPLETED — fixed and verified.
> **Opened**: 2026-03-04
> **Trigger**: Observed large discrepancy between true and noised nutrition values
> **Root Cause**: Index mismatch — noise arrays in `config_loader.py` built in wrong order relative to `modality_map` in `sensor.py`
> **Related**: [OBSERVATION_SCALE_PLAN.md](OBSERVATION_SCALE_PLAN.md) (independent issue)

---

## Root Cause

The noise parameter arrays in `config_loader.py` were built in a different modality order than `sensor.py:apply_perceptual_noise()` expected. Every sensor (except Visual, correct by accident) received wrong noise parameters.

### Canonical modality order — defined in `default.yaml`

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

### Mismatch (before fix)

| Array idx | What `config_loader` put | What `modality_map` expected |
|:---------:|--------------------------|------------------------------|
| 0 | olfaction (σ=0.15) | **Injury** (σ=0.05, state_dep) |
| 1 | extero_nociception (σ=0.01) | **Nutrition** (σ=0.10) |
| 2 | collision (σ=0.01) | **Satiation** (σ=0.10) |
| 3 | location (σ=0.01) | **Extero Nociception** (σ=0.01) |
| 4 | satiation (σ=0.10) | **Olfaction** (σ=0.15) |
| 5 | nutrition (σ=0.10) | **Collision** (σ=0.01) |
| 6 | injury (σ=0.05, state_dep) | **Proprioception** (σ=0.05) |
| 7 | visual (σ=0.05) | **Visual** (σ=0.05) ← correct! |
| 8 | proprioception (σ=0.05) | **Location** (σ=0.01) |

## Impact (before fix)

| Sensor | Intended σ | **Actual σ** | Intended mode | **Actual mode** | Intended clip | **Actual clip** |
|--------|:----------:|:------------:|:-------------:|:---------------:|:-------------:|:---------------:|
| Injury | 0.05 | **0.15** (+3×) | state_dependent | **constant** | [0,1] | **[0,100]** |
| Nutrition | 0.10 | **0.01** (÷10) | constant | constant | [0,1] | **[0,100]** |
| Satiation | 0.10 | **0.01** (÷10) | constant | constant | [0,1] | [0,1] OK |
| Extero Noc | 0.01 | **0.01** OK | constant | constant | [0,100] | **[-1,1]** |
| Olfaction | 0.15 | **0.10** | constant | constant | [0,100] | **[0,1]** ← clips! |
| Collision | 0.01 | **0.10** (+10×) | constant | constant | [0,1] | [0,1] OK |
| Proprioception | 0.05 | **0.05** OK | constant | **state_dependent** | [0,1] | [0,1] OK |
| Visual | 0.05 | **0.05** ✓ | constant | constant ✓ | [0,100] | [0,100] ✓ |
| Location | 0.01 | **0.05** (+5×) | constant | constant | [-1,1] | **[0,1]** |

Most impactful: olfaction clipped at 1.0, injury lost state-dependent mode, nutrition/satiation 10× less noise, collision 10× more noise.

---

## Fix: YAML-Driven Ordering

### Design

Make **YAML config the single source of truth** for modality ordering:

```
default.yaml (canonical key order) → config_loader._parse_noise_config() (builds arrays in YAML order)
    → EnvParams.noise_modality_order (tuple) → sensor.py modality_map (dynamic derivation)
```

- PyYAML preserves dict insertion order (Python 3.7+)
- `Config.merge()` preserves key position (updates in-place)

### Implementation (4 files)

#### 1. `configs/environment/default.yaml`

Reorder `perceptual_noise.modalities` keys to canonical order:

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

#### 2. `src/environment/state.py`

```python
noise_modality_order: tuple = struct.field(pytree_node=False)  # e.g. ("Injury","Nutrition",...)
```

#### 3. `src/environment/config_loader.py` (lines 317–375)

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

def _parse_noise_config(config):
    modalities_cfg = config.get('perceptual_noise.modalities') or {}
    def _parse_mode(s):
        return 2 if s == 'state_dependent' else 1 if s == 'constant' else 0

    noise_modality_order = tuple(
        _YAML_KEY_TO_SENSOR_NAME[k] for k in modalities_cfg if k in _YAML_KEY_TO_SENSOR_NAME
    )
    pad = max(0, 12 - len(noise_modality_order))
    # Build all 5 arrays in YAML key order, pad to 12...
    return {"noise_modality_order": noise_modality_order, "noise_modes": ..., ...}
```

#### 4. `src/environment/sensor.py` (lines 204–216)

```python
# BEFORE: hardcoded modality_map = {"Injury": 0, "Nutrition": 1, ...}
# AFTER:
modality_map = {name: i for i, name in enumerate(params.noise_modality_order)}
```

---

## Verification Report

> **Verified**: 2026-03-04

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/environment/default.yaml` | Reorder modality keys to canonical order | ✅ | Contract comment at lines 260–261 |
| `src/environment/state.py` | Add `noise_modality_order` tuple field | ✅ | `pytree_node=False`, line 161 |
| `src/environment/config_loader.py` | Replace hardcoded arrays with `_parse_noise_config()` | ✅ | YAML-order-driven loop, lines 334–379 |
| `src/environment/sensor.py` | Replace hardcoded `modality_map` with dynamic derivation | ✅ | Line 206 |
| `src/environment/sensor.py` | Asymmetric guard in clip construction | ⚠️ | See detail below |

**Conclusion**: Noise modality ordering fix is correctly implemented.

#### ⚠️ Asymmetric guard in clip construction

`sensor.py` lines 237–238: The clip construction has `if name in modality_map` guard, but the sigma/mode loop (lines 213–217) does not. If a sensor name is ever missing from `modality_map`, sigma/mode would `KeyError` but clips would silently skip, causing a dimension mismatch between `sigma_eff` and `clip_min`/`clip_max`. Non-blocking (all 9 modalities always present), but recommend making both consistent.

---

## Debugging Protocol

### Step 0: Confirm the Bug

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
print(f"{'Sensor':<22} idx   sigma  mode  clip")
for i, name in enumerate(sensor_order):
    print(f"  {name:<22} {i}  ->  {float(params.noise_sigmas[i]):.3f}  {int(params.noise_modes[i])}     [{float(params.noise_clip_min[i]):.1f},{float(params.noise_clip_max[i]):.1f}]")
```

### Step 1: Verify the Fix

```python
assert list(params.noise_modality_order) == [
    "Injury","Nutrition","Satiation","Extero Nociception",
    "Olfaction","Collision","Proprioception","Visual","Location"
]
assert abs(float(params.noise_sigmas[0]) - 0.05) < 1e-5   # Injury
assert abs(float(params.noise_sigmas[1]) - 0.10) < 1e-5   # Nutrition
assert abs(float(params.noise_sigmas[4]) - 0.15) < 1e-5   # Olfaction
assert int(params.noise_modes[0]) == 2                     # Injury = state_dependent
assert int(params.noise_modes[6]) == 1                     # Proprioception = constant
assert abs(float(params.noise_clip_max[4]) - 100.0) < 1e-5 # Olfaction unbounded
assert abs(float(params.noise_clip_min[8]) - (-1.0)) < 1e-5 # Location min = -1
```

### Why Nutrition Appeared "Too Different"

1. **Normalization gap**: `state.nutrition` is raw (0–100), `obs[1]` is normalized (0–1).
2. **Wrong clip from the bug**: Nutrition got `clip_max=100.0` instead of `1.0`.

---

## Secondary Issues

### A. Sigma calibration for unbounded sensors
Log `mean(|obs_olfaction|)` and `std(noise_olfaction)` over a rollout. Target: noise_std / signal_mean ≈ 5–20%.

### B. State-dependent injury noise magnitude
`σ_eff = 0.05 × (1 + 1.5 × injury_norm)`. Uninjured: σ=0.05, fully injured: σ=0.125. Verify biological intent.

### C. PRNG key correlation across parallel environments
`get_observation()` uses `obs_key = jax.random.fold_in(state.key, 999)`. All environments could get identical obs noise if `state.key` is shared across vmap batch.
