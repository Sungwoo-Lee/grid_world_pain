---
title: Perceptual Precision Modulation
topic: precision
status: active
created: 2026-02-11
last_updated: 2026-04-12
---

# Perceptual Precision Modulation

## 1. Theoretical Framework

In computational neuroscience and **Active Inference**, "Perceptual Precision" refers to the reliability or confidence assigned to sensory data. Mathematically, it is the **inverse variance** ($\pi = 1/\sigma^2$) of the likelihood distribution $P(o|s)$.

### High Precision ($\uparrow \pi$, $\downarrow \sigma^2$)
Sensory data is "trusted." The agent's posterior belief is strongly driven by observations, allowing for rapid environment tracking but making the agent vulnerable to sensory artifacts.

### Low Precision ($\downarrow \pi$, $\uparrow \sigma^2$)
Sensory data is ignored or "blurred." The agent relies more on its **prior (internal model)** and temporal integration. This is biologically observed during high-arousal states, severe injury, or high-velocity movement.

---

## 2. Mathematical Models for Noise

Three levels of noise are supported by the environment:

### A. None (Deterministic)
No noise is applied. The observation is passed through unchanged:
$$o_{noisy} = o_{true}$$

### B. Constant Additive Gaussian
For continuous sensors, a fixed-variance Gaussian is added:
$$o_{noisy} = o_{true} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2_{base})$$

### C. State-Dependent Modulation
Precision is modulated by the agent's normalized injury level. Higher injury leads to greater noise (lower precision):
$$\sigma_{eff} = \sigma_{base} \cdot (1 + \alpha \cdot \hat{I})$$
where $\hat{I} = \frac{\text{injury\_level}}{\text{max\_injury}}$ is the normalized injury (clamped to $[0, 1]$) and $\alpha$ is the `injury_noise_scale` coefficient. The noisy observation is then:
$$o_{noisy} = o_{true} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2_{eff})$$

---

## 3. Architecture

The system is split across four files spanning configuration, state, loading, and runtime:

```
configs/environment/environment.yaml   ← Per-modality noise settings
src/environment/state.py               ← EnvParams fields (vectorized arrays)
src/environment/config_loader.py       ← YAML → JAX array conversion
src/environment/sensor.py              ← Runtime noise application (JIT-compiled)
```

### 3.1 Per-Modality Configuration

Each of the 9 sensory modalities is independently configured with three parameters:

| Parameter             | Type   | Description                                         |
|-----------------------|--------|-----------------------------------------------------|
| `mode`                | string | `"none"`, `"constant"`, or `"state_dependent"`      |
| `sigma`               | float  | Base standard deviation of Gaussian noise            |
| `injury_noise_scale`  | float  | Multiplier ($\alpha$) for injury-dependent scaling   |
| `clip_min`            | float  | Lower bound for post-noise clamping                  |
| `clip_max`            | float  | Upper bound for post-noise clamping                  |

This "mix and match" design allows heterogeneous noise profiles. For example, olfaction can be state-dependent while location sensing remains constant.

### 3.2 Modality Index Mapping

Modalities are mapped to fixed array indices, synchronized between `config_loader.py` and `sensor.py`:

| Index | Modality             |
|-------|----------------------|
| 0     | Olfaction            |
| 1     | Extero Nociception   |
| 2     | Collision            |
| 3     | Location             |
| 4     | Satiation            |
| 5     | Nutrition            |
| 6     | Injury               |
| 7     | Visual               |
| 8     | Proprioception       |
| 9–11  | *(padding buffer)*   |

---

## 4. Implementation

### 4.1 State Parameters (`state.py`)

Three parallel JAX arrays of length 12 (9 modalities + 3 padding) are stored in `EnvParams`:

```python
# Perceptual Noise Parameters (Vectorized across modalities)
perceptual_noise_enabled: bool = struct.field(pytree_node=False)
noise_modes: jnp.ndarray          # [12] int32 (0: None, 1: Constant, 2: State-Dependent)
noise_sigmas: jnp.ndarray         # [12] float32 (Base Sigma)
noise_injury_scales: jnp.ndarray  # [12] float32 (Injury Noise Scale)
noise_clip_min: jnp.ndarray       # [12] float32 (Per-modality observation lower bound)
noise_clip_max: jnp.ndarray       # [12] float32 (Per-modality observation upper bound)
```

Five parallel JAX arrays of length 12 (9 modalities + 3 padding buffer) ensure a fixed shape for JIT compilation, with headroom for future modalities.

### 4.2 Config Loading (`config_loader.py`)

The loader converts YAML string modes to integer codes and pads each array:

- `"none"` → `0`
- `"constant"` → `1`
- `"state_dependent"` → `2`

Each of the five arrays (`noise_modes`, `noise_sigmas`, `noise_injury_scales`, `noise_clip_min`, `noise_clip_max`) is constructed by reading the 9 modality entries from the YAML config and padding with 3 zeros via `jnp.pad(..., (0, 3))`. The `clip_max` padding uses `100.0` (effectively unbounded) as the default.

### 4.3 Vectorized Sigma Mask (`sensor.py`)

The observation vector $o$ is a concatenation of variable-length slices (one per modality). The `apply_perceptual_noise` function constructs element-wise `sigma_base`, `alpha`, and `mode` vectors that match the full observation shape, using `get_observation_breakdown()` to determine each modality's slice width:

```python
for sensor_name, dim in breakdown.items():
    idx = modality_map[sensor_name]
    sigma_base_list.append(jnp.full((dim,), params.noise_sigmas[idx]))
    alpha_list.append(jnp.full((dim,), params.noise_injury_scales[idx]))
    mode_list.append(jnp.full((dim,), params.noise_modes[idx]))
    clip_min_list.append(jnp.full((dim,), params.noise_clip_min[idx]))
    clip_max_list.append(jnp.full((dim,), params.noise_clip_max[idx]))
```

This produces parallel vectors ($\vec{\sigma}$, $\vec{\alpha}$, $\vec{m}$, $\vec{c}_{lo}$, $\vec{c}_{hi}$) of the same dimension as $o$, enabling fully vectorized noise application and clamping.

### 4.4 Core Noise Function (`sensor.py`)

```python
@jax.jit
def apply_perceptual_noise(obs, state, params, key):
    # ... builds per-element sigma_base, alpha, mode, clip_lo, clip_hi vectors ...

    norm_injury = state.injury_level / jnp.maximum(params.max_injury, 1e-6)

    sigma_eff = jnp.where(
        mode == 2,
        sigma_base * (1.0 + alpha * norm_injury),       # State-dependent
        jnp.where(mode == 1, sigma_base, 0.0)           # Constant or None
    )

    noise = jax.random.normal(key, obs.shape) * sigma_eff
    return jnp.clip(obs + noise, clip_lo, clip_hi)
```

The effective sigma per element is:
- **Mode 0 (none):** $\sigma_{eff} = 0$
- **Mode 1 (constant):** $\sigma_{eff} = \sigma_{base}$
- **Mode 2 (state-dependent):** $\sigma_{eff} = \sigma_{base} \cdot (1 + \alpha \cdot \hat{I})$

### 4.5 Observation Assembly & Noise Key (`sensor.py`)

Noise is applied as the final step of `get_observation()`:

```python
@jax.jit(static_argnames=['apply_noise'])
def get_observation(state, params, apply_noise=True):
    obs_key = jax.random.fold_in(state.key, 999)
    # ... assemble olfaction, nociception, collision, location,
    #     interoception, visual, proprioception into obs ...
    if apply_noise:
        return apply_perceptual_noise(obs, state, params, obs_key)
    return obs
```

- **Training:** called with `apply_noise=True` (default) — observations are noisy.
- **Evaluation:** called with `apply_noise=False` — observations are clean and deterministic.

The noise key is derived deterministically via `fold_in(state.key, 999)`, ensuring reproducibility: the same state always produces the same noise realization.

---

## 5. Current Configuration

The active settings in `configs/environment/environment.yaml`:

```yaml
perceptual_noise:
  enabled: true
  modalities:
    olfaction:
      mode: "state_dependent"
      sigma: 0.15
      injury_noise_scale: 2.0
      clip_min: 0.0
      clip_max: 100.0       # Unbounded (aggregated chemical gradient)
    extero_nociception:
      mode: "constant"
      sigma: 0.01
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 100.0       # Unbounded (pain intensity)
    collision:
      mode: "constant"
      sigma: 0.01
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    location:
      mode: "constant"
      sigma: 0.01
      injury_noise_scale: 0.0
      clip_min: -1.0         # Normalized to [-1, 1]
      clip_max: 1.0
    satiation:
      mode: "constant"
      sigma: 0.1
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    nutrition:
      mode: "constant"
      sigma: 0.1
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    injury:
      mode: "state_dependent"
      sigma: 0.05
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 1.0
    visual:
      mode: "state_dependent"
      sigma: 0.05
      injury_noise_scale: 3.0
      clip_min: 0.0
      clip_max: 100.0       # Unbounded (one-hot counts)
    proprioception:
      mode: "constant"
      sigma: 0.05
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
```

### Effective Noise Summary

| Modality             | Mode             | $\sigma_{base}$ | $\alpha$ | $\sigma_{eff}$ at $\hat{I}=1$ | Clamp Range   |
|----------------------|------------------|------------------|----------|-------------------------------|---------------|
| Olfaction            | state_dependent  | 0.15             | 2.0      | 0.45                          | [0, 100]      |
| Extero Nociception   | constant         | 0.01             | —        | 0.01                          | [0, 100]      |
| Collision            | constant         | 0.01             | —        | 0.01                          | [0, 1]        |
| Location             | constant         | 0.01             | —        | 0.01                          | [-1, 1]       |
| Satiation            | constant         | 0.10             | —        | 0.10                          | [0, 1]        |
| Nutrition            | constant         | 0.10             | —        | 0.10                          | [0, 1]        |
| Injury               | state_dependent  | 0.05             | 1.5      | 0.125                         | [0, 1]        |
| Visual               | state_dependent  | 0.05             | 3.0      | 0.20                          | [0, 100]      |
| Proprioception       | constant         | 0.05             | —        | 0.05                          | [0, 1]        |

### Design Rationale

- **Olfaction** has the highest baseline noise and strongest injury scaling, reflecting that chemical gradient sensing is inherently noisy and degrades significantly under injury.
- **Visual** has the largest $\alpha$ (3.0), modeling severe perceptual degradation under injury — at maximum injury the effective sigma quadruples.
- **Injury sensing** is itself state-dependent ($\alpha = 1.5$), capturing the idea that proprioceptive pain signals become less precise as injury worsens.
- **Extero nociception, collision, and location** use minimal constant noise (0.01), treating these as relatively reliable signals.
- **Satiation and nutrition** have moderate constant noise (0.1), reflecting imprecise interoceptive sensing.
- **Proprioception** (previous action encoding) has mild constant noise (0.05).

---

## 6. Usage in Research

By varying `injury_noise_scale` ($\alpha$) per modality, you can simulate different **perceptual regimes** — from agents that are nearly immune to pain-induced confusion to those whose world becomes illegible the moment they take damage.

Example experimental conditions:

| Condition            | Olfaction $\alpha$ | Visual $\alpha$ | Injury $\alpha$ | Description                          |
|----------------------|--------------------|-----------------|-----------------|--------------------------------------|
| Baseline (current)   | 2.0                | 3.0             | 1.5             | Moderate degradation under injury    |
| Resilient            | 0.5                | 0.5             | 0.5             | Mild precision loss                  |
| Fragile              | 5.0                | 5.0             | 3.0             | Severe sensory collapse under injury |
| No modulation        | 0.0                | 0.0             | 0.0             | Constant noise only (ablation)       |

---

## 7. Known Limitations & Future Work

1. **Per-modality output clamping** — After adding noise, observations are clamped to per-modality bounds via `jnp.clip(obs + noise, clip_lo, clip_hi)`. This prevents physically meaningless values (e.g., negative satiation, location outside [-1, 1]). The bounds are configurable via `clip_min` and `clip_max` in the YAML. Modalities with no natural upper bound (olfaction, nociception, visual) use 100.0 as an effectively unbounded ceiling.

2. **Deterministic noise within a step** — The noise key is derived via `fold_in(state.key, 999)`, so repeated calls to `get_observation` on the same state produce identical noise. This is desirable for JAX reproducibility but prevents stochastic re-sampling within a single timestep.

3. **Olfaction signal-to-noise ratio** — At maximum injury, olfaction's $\sigma_{eff} = 0.45$. Since olfaction values are often small gradient signals, this may overwhelm the signal entirely. The signal-to-noise ratio at high injury levels should be monitored during training.

4. **Weber-Fechner scaling not implemented** — The original proposal included intensity-dependent noise ($\sigma = \sigma_{base} \cdot o_{true}$). This remains a potential future extension for modalities where biological noise scales with stimulus intensity.

5. **Padding buffer** — The 3-element padding buffer (indices 9–11) is undocumented in code. If a 10th modality is added, the buffer silently shrinks. Consider adding a runtime assertion that the modality count does not exceed the array length.
