---
title: Verify Perceptual Noise Disabling
topic: noise
status: archive
created: 2026-03-12
last_updated: 2026-04-12
---

# Verify Perceptual Noise Disabling

> Status: IN PROGRESS
> **Opened**: 2026-03-12
> Implemented by: Gemini
> Date: 2026-03-12 14:15:00
> **Related**: [configs/environment/default.yaml](../configs/environment/default.yaml), [src/environment/sensor.py](../src/environment/sensor.py)

---

## Context

The user wants to ensure that when `perceptual_noise.enabled` is set to `false` in the configuration, no noise is added to the observations. This is critical for deterministic evaluation and for research conditions that require a "clean" environment.

## Analysis

In `src/environment/sensor.py`, the `apply_perceptual_noise` function includes an early return if noise is disabled:

```python
def apply_perceptual_noise(obs: jnp.ndarray, state: EnvState, params: EnvParams, key: jax.random.PRNGKey):
    if not params.perceptual_noise_enabled:
        return obs
    ...
```

The `perceptual_noise_enabled` parameter is loaded in `src/environment/config_loader.py`:

```python
perceptual_noise_enabled=config.get('perceptual_noise.enabled', False),
```

Currently, `configs/environment/default.yaml` has `enabled: false`.

## Checkpoints

- [x] Checkpoint 1: Verify `perceptual_noise.enabled: false` is correctly loaded from YAML. [14:18:22]
- [x] Checkpoint 2: Direct test of `apply_perceptual_noise` with `perceptual_noise_enabled=False`. [14:18:22]
- [x] Checkpoint 3: Verify `get_observation` produces identical results across multiple keys when noise is disabled. [14:18:22]
- [x] Checkpoint 4: Verify `get_observation` produces different results when noise is enabled with non-zero sigma. [14:18:22]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-03-12 14:15:00

| File | Change | Status | Notes |
|:-----|:-------|:------:|:------|
| [scripts/verify_noise.py](../scripts/verify_noise.py) | Created verification script | [x] | Success |

**Conclusion**:
The verification script `scripts/verify_noise.py` confirmed that perceptual noise is correctly disabled when `perceptual_noise.enabled` is set to `false` in the configuration. The observations are deterministic and identical across different PRNG keys when disabled, and they show stochastic variation when enabled.

> Status: COMPLETED
