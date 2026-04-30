# 10 — Perceptual Noise

> **Source**: `src/environment/sensor.py` (`apply_perceptual_noise`, `sensor.py:197`), `src/environment/config_loader.py` (`_parse_noise_config`, `config_loader.py:334`) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

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
| `interoceptive_nociception` | Interoceptive Nociception | state_dependent | 0.1 | 1.5 | [0, 1] |
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
4. Zero-pad all five arrays to length 13 using `jnp.pad(..., (0, pad))`.

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

**Why 13, not 10**: the pad to 13 keeps array shapes static across configs with fewer modalities. Adding or removing modalities from the YAML changes `noise_modality_order` length (a static field — triggers recompilation) but keeps the array shape at 13, which avoids GPU memory reallocations. The 3 extra slots are zero-padded and never accessed.

**Index assignment**: indices 0–8 correspond to YAML keys in declaration order. An example with the default config:

| Array index | Modality |
|-------------|---------|
| 0 | Injury |
| 1 | Nutrition |
| 2 | Satiation |
| 3 | Interoceptive Nociception |
| 4 | Extero Nociception |
| 5 | Olfaction |
| 6 | Collision |
| 7 | Proprioception |
| 8 | Visual |
| 9 | Location |
| 10–12 | Zero padding |

**Important**: the arrays are indexed by name at runtime via `modality_map`. If a modality listed in `noise_modality_order` is not in `get_observation_breakdown(params)` (e.g. sensor disabled but noise configured), `apply_perceptual_noise` will raise a `KeyError`. The safe pattern is to always have noise config match the enabled sensor set.

---

## Clarifications / FAQ

**Q: What's the lookup failure mode — KeyError or silent skip?**
A: **KeyError**, but in the opposite direction from the note above. Line 214 (`idx = modality_map[sensor_name]`) raises when an *enabled sensor* has no noise entry in `noise_modality_order`. If you enable Olfaction but forget to declare `olfaction:` under `perceptual_noise.modalities`, you'll crash at JIT-trace time. To safely skip noise for a sensor, set its mode to `none` in YAML — don't just omit it.

**Q: Does noise apply when the sensor has no signal (e.g. no predator in sight)?**
A: Yes. Noise is added unconditionally per dimension. An "all zeros" olfaction vector will become "all small Gaussians" after noise. The clip bounds then ensure the value stays in range. Agents must learn to distinguish "weak signal" from "weak signal + noise" — this is the whole point of perceptual noise.

**Q: Is noise correlated across dimensions within one modality?**
A: No. Each dimension gets an independent draw from `Normal(0, σ_eff)` (`sensor.py:240` samples one noise vector of full obs length). Two dimensions of the same modality (e.g. olfaction channels) have uncorrelated noise.

**Q: Is noise correlated across modalities in one step?**
A: No, same reason. One PRNG key, one vector draw of full-obs-length, one sample per element. Different modalities have different σ_eff but the underlying N(0,1) draws are independent.

**Q: Is noise correlated across steps?**
A: No. `obs_key = jax.random.fold_in(state.key, 999)` and `state.key` changes every step via the split-and-store pattern (`core.py:513`). Sequential observations see uncorrelated noise.

**Q: What happens at `injury_norm > 1.0`?**
A: Not possible — injury is clipped to `[0, max_injury]` in `update_body`. But if it somehow were, `σ_eff` would grow without bound: `σ_base × (1 + α × 1.5)` etc. No guard.

**Q: What does `clip_min/clip_max` do beyond the natural sensor range?**
A: Defines the valid range of the *noisy* observation. For interoceptive sensors with natural range `[0, 1]`, the clip typically matches. For olfaction with `clip_max: 100`, the clip is wide because olfaction can saturate well above 1 when the agent is on top of an entity. The clip is applied only after noise — **the pre-noise obs is never clipped** by this step.

**Q: Why are `clip_min/clip_max` defaults `[-100, 100]` in `_parse_noise_config`?**
A: Permissive bounds so that missing clip config doesn't mangle the signal. If you want a narrower clip, explicitly set `clip_min: 0, clip_max: 1` per modality.

**Q: If I set `mode: state_dependent` and `injury_noise_scale: 0`, does it behave like `constant`?**
A: Yes. `σ_eff = σ_base × (1 + 0 × injury_norm) = σ_base`. Functionally identical. The `mode` dispatch is then redundant.

**Q: Can I set `sigma: 0` in `state_dependent` mode?**
A: Yes but pointless. `σ_eff = 0 × (...) = 0`. No noise added. Equivalent to `mode: none`.

**Q: How does noise affect the "Visual" sensor's binary channels?**
A: Small Gaussians centered on 0/1, then clipped. A channel that reads 1 might become 0.87, and a channel that reads 0 might become 0.08. The agent sees this continuous smear, not binary. Consider whether your policy network expects binary inputs.

**Q: Is perceptual noise applied during training but not eval?**
A: That depends on how the caller invokes `get_observation`. `apply_noise` is a function parameter. Training wrappers pass `True`; eval wrappers typically pass `False` for deterministic rollout. See `ParallelEnv` for the pattern.

**Q: What's the CPU/GPU cost of `apply_perceptual_noise`?**
A: One vector draw from `Normal(0, 1)` sized to the observation, one elementwise multiply-add, one clip. Negligible compared to the rest of the step. The `breakdown` loop is Python-side but runs only at trace time (unrolled by `@jax.jit`).

**Q: Why the 999 fold-in constant?**
A: Documented in doc `09` — a readable salt constant to create an independent noise PRNG branch without splitting the main key. Any constant would work.

**Q: Can I add a eleventh modality?**
A: Yes, if it corresponds to a real sensor. Steps: (1) add the YAML key to `_YAML_KEY_TO_SENSOR_NAME`; (2) add the sensor name to `get_observation_breakdown`; (3) if there will be > 13 active modalities, increase the pad-to-13 to the new cap in `config_loader.py:401`. All noise arrays are currently sized `[13]` — adding modalities beyond that needs a schema change.

**Q: Do `noise_modality_order` and `get_observation_breakdown` have to declare modalities in the same order?**
A: No. Lookup is by name (`modality_map[sensor_name]`). Order determines array *indices*; names determine *lookup*. But it's convention to keep them aligned for readability.
