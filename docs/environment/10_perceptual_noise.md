# 10 — Perceptual Noise

> **Source**: `src/environment/sensor.py` (`apply_perceptual_noise`, `sensor.py:197`), `src/environment/config_loader.py` (`_parse_noise_config`, `config_loader.py:334`) | **Back to hub**: [README](README.md)

---

## Overview

Perceptual noise simulates the imprecision of biological sensing. After the full observation vector is assembled by `get_observation`, Gaussian noise is added independently to each sensor modality. The key design principle is **state-dependent noise**: when the agent is injured, sensory precision degrades across multiple modalities — a biological analogue of pain-induced perceptual distortion.

The system is enabled/disabled by `params.perceptual_noise_enabled` (static flag). When disabled, `apply_perceptual_noise` is a pass-through returning `obs` unchanged.

**Critical invariant**: the YAML key order under `perceptual_noise.modalities` is the single source of truth for which index in the noise arrays corresponds to which modality. Do not reorder arrays independently of the YAML or `noise_modality_order` tuple.

---

## Noise Modes

Three modes per modality (`noise_modes[i]`):

| Mode | Value | σ_effective | Description |
|------|-------|-------------|-------------|
| None | 0 | 0.0 | No noise applied |
| Constant | 1 | `σ_base` | Fixed Gaussian noise regardless of body state |
| State-Dependent | 2 | `σ_base × (1 + α × injury_norm)` | Noise scales with normalised injury level |

**State-dependent formula** (`sensor.py:230`):
```
injury_norm = injury_level / max(max_injury, 1e-6)
σ_eff = σ_base × (1 + α × injury_norm)
```

- At `injury_norm = 0.0` (healthy): `σ_eff = σ_base`.
- At `injury_norm = 1.0` (maximum injury): `σ_eff = σ_base × (1 + α)`.
- With defaults `σ_base=0.1`, `α=1.5`: max σ = `0.1 × 2.5 = 0.25`.

**Motivation**: pain degrades interoceptive and exteroceptive perception. A highly injured agent has less reliable sensory information about its own state and its environment, requiring it to act under greater uncertainty.

---

## Configuration

YAML structure under `perceptual_noise.modalities`:

```yaml
perceptual_noise:
  enabled: true
  modalities:
    <modality_key>:
      mode: "none" | "constant" | "state_dependent"
      sigma: <float>                  # σ_base
      injury_noise_scale: <float>     # α (used in mode 2 only)
      clip_min: <float>               # observation lower bound after noise
      clip_max: <float>               # observation upper bound after noise
```

**All 9 modalities** (default config values):

| YAML key | Sensor name | Mode | σ_base | α | clip |
|----------|------------|------|--------|---|------|
| `injury` | Injury | state_dependent | 0.1 | 1.5 | [0, 1] |
| `nutrition` | Nutrition | state_dependent | 0.1 | 1.5 | [0, 1] |
| `satiation` | Satiation | state_dependent | 0.1 | 1.5 | [0, 1] |
| `extero_nociception` | Extero Nociception | state_dependent | 0.1 | 1.5 | [0, 100] |
| `olfaction` | Olfaction | state_dependent | 0.2 | 1.5 | [0, 100] |
| `collision` | Collision | constant | 0.01 | 0.0 | [0, 1] |
| `proprioception` | Proprioception | constant | 0.05 | 0.0 | [0, 1] |
| `visual` | Visual | state_dependent | 0.2 | 1.5 | [0, 100] |
| `location` | Location | constant | 0.01 | 0.0 | [-1, 1] |

---

## `_parse_noise_config` (config_loader)

`_parse_noise_config(config) → dict` (`config_loader.py:334`)

**Step-by-step**:
1. Read `perceptual_noise.modalities` dict (Python dict, insertion order preserved in Python ≥ 3.7).
2. Emit `noise_modality_order` tuple by mapping YAML keys through `_YAML_KEY_TO_SENSOR_NAME` — only keys present in that mapping are included.
3. Build five parallel arrays by iterating the same filtered key order:
   - `noise_modes [K]` int32: `0 / 1 / 2` from mode string
   - `noise_sigmas [K]` float32: `sigma` field
   - `noise_injury_scales [K]` float32: `injury_noise_scale` field
   - `noise_clip_min [K]` float32: `clip_min` field (default -100.0)
   - `noise_clip_max [K]` float32: `clip_max` field (default 100.0)
4. Zero-pad all five arrays to length 12 using `jnp.pad(..., (0, pad))`.

The returned dict is unpacked into `EnvParams` with `**_parse_noise_config(config)`.

---

## `apply_perceptual_noise` (sensor)

`apply_perceptual_noise(obs, state, params, key)` (`sensor.py:197`)

**Step-by-step**:
1. Call `get_observation_breakdown(params)` to get the ordered `{name: dim}` mapping.
2. Build `modality_map = {name: index}` from `params.noise_modality_order`.
3. For each `(sensor_name, dim)` in breakdown, look up the modality index and broadcast the noise params (`σ_base`, `α`, `mode`) to a vector of length `dim`.
4. Concatenate all broadcasted vectors into `sigma_base [obs_dim]`, `alpha [obs_dim]`, `mode [obs_dim]`.
5. Compute `norm_injury = injury_level / max_injury`.
6. Compute effective sigma per element:
   ```
   σ_eff = mode==2 ? σ_base*(1+α*inj) : mode==1 ? σ_base : 0.0
   ```
7. Build clip bounds similarly, broadcasting from `noise_clip_min/max` by modality.
8. Sample `noise = Normal(0, σ_eff)` once for the full obs vector.
9. Return `clip(obs + noise, clip_min, clip_max)`.

The PRNG key comes from `jax.random.fold_in(state.key, 999)` in `get_observation` — a fixed fold-in that creates a per-step, per-env observation noise key without consuming a key split from the main step.

---

## Array Layout

**Why 12, not 9**: the pad to 12 keeps array shapes static across configs with fewer modalities. Adding or removing modalities from the YAML changes `noise_modality_order` length (a static field — triggers recompilation) but keeps the array shape at 12, which avoids GPU memory reallocations. The 3 extra slots are zero-padded and never accessed.

**Index assignment**: indices 0–8 correspond to YAML keys in declaration order. An example with the default config:

| Array index | Modality |
|-------------|---------|
| 0 | Injury |
| 1 | Nutrition |
| 2 | Satiation |
| 3 | Extero Nociception |
| 4 | Olfaction |
| 5 | Collision |
| 6 | Proprioception |
| 7 | Visual |
| 8 | Location |
| 9–11 | Zero padding |

**Important**: the arrays are indexed by name at runtime via `modality_map`. If a modality listed in `noise_modality_order` is not in `get_observation_breakdown(params)` (e.g. sensor disabled but noise configured), `apply_perceptual_noise` will raise a `KeyError`. The safe pattern is to always have noise config match the enabled sensor set.
