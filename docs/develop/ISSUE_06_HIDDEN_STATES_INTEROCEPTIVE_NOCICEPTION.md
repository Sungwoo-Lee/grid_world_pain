# Hidden Body States + Delayed Interoceptive Nociception

> **Status**: COMPLETED
> **Implemented by**: Gemini
> **Date**: 2026-04-30 22:15:42
> **Opened**: 2026-04-30
> **Related**: [docs/environment/01_state_and_params.md](../environment/01_state_and_params.md), [docs/environment/05_body_homeostasis.md](../environment/05_body_homeostasis.md), [docs/environment/09_sensors_and_observation.md](../environment/09_sensors_and_observation.md), [docs/environment/10_perceptual_noise.md](../environment/10_perceptual_noise.md)

---

## Context

The current observation vector exposes **raw `injury_level` and `nutrition` directly to the agent** as the first two scalar slices ([sensor.py:252-255](../../src/environment/sensor.py#L252-L255)). For interoceptive-AI research we want these to behave as **hidden body states**, with the agent receiving only **derived perceptual signals**:

- `nutrition` (hidden) → `satiation` (already a non-linear derivative via `S = MaxS·(N/MaxN)^k`, [core.py:60-66](../../src/environment/core.py#L60-L66)). Satiation is already observable; we just need to stop emitting raw nutrition.
- `injury_level` (hidden) → a new **interoceptive nociception** signal that lags injury via a temporal convolution with a delayed-peak (alpha) kernel — consistent with the slow C-fiber pain dynamics being modelled. The existing `Extero Nociception` (slice [3]) is a *phasic contact detector* and stays untouched as a separate sensor.

Design decisions (confirmed with user):
1. **Kernel**: alpha function (delayed peak, single time-constant τ).
2. **Sensor wiring**: NEW `Interoceptive Nociception` sensor placed alongside the existing `Extero Nociception` — two separate slices, independent enable flags.
3. **Buffer**: NEW dedicated `nociception_history_buffer` on `EnvState`, sized by a new static param. Decoupled from `injury_buffer` (which serves body-side damage smoothing).
4. **Hide mechanism**: per-signal config flags `sensory.injury_observable` and `sensory.nutrition_observable`. Default `true` for backward compatibility; experiments that want hidden states explicitly set them to `false`.
5. **Convolution disable (NEW)**: a `sensory.interoceptive_convolution_enabled` flag. When `false`, the sensor bypasses the alpha kernel entirely and emits `injury_level / max_injury` directly — i.e. **`intero == injury` exactly, no delay, no smoothing**. This is needed because even τ=1 leaves a 1-step lag (since `k[0]=0` is a structural property of the alpha function). Passthrough mode lets researchers ablate the perceptual delay without rewiring the obs vector. The hidden-state flag (`injury_observable`) is independent — passthrough on the *interoceptive* sensor still produces a separate slice with its own perceptual-noise channel, distinct from the (now optional) raw injury slice.

## Analysis

### Current observation layout (default config, all sensors on)

The new Interoceptive Nociception sensor is **inserted into the interoceptive cluster** (next to Satiation), not after Extero Nociception. The interoceptive cluster (Injury, Nutrition, Satiation, Interoceptive Nociception) is the body-state group; Extero Nociception is exteroceptive (phasic contact detector). Grouping them together keeps the renderer's tile order consistent with sensor semantics.

| Slice | Sensor | Source | Hidden in this plan? |
|-------|--------|--------|----------------------|
| `[0]` | Injury (raw `injury_level/max_injury`) — **interoceptive** | always | **YES** (gated off) |
| `[1]` | Nutrition (raw `nutrition/max_nutrition`) — **interoceptive** | always | **YES** (gated off) |
| `[2]` | Satiation — **interoceptive** | always | NO (kept observable) |
| **NEW** | Interoceptive Nociception (delayed/passthrough injury) — **interoceptive** | `interoceptive_nociception_enabled` | added |
| `[3]` | Extero Nociception (phasic contact) — **exteroceptive** | `nociception_enabled` | NO (kept) |
| `[4:4+V]` | Olfaction | `olfactory_enabled` | NO |
| ... | Collision / Proprio / Visual / Location | flags | NO |

`get_observation_breakdown()` ([sensor.py:295-333](../../src/environment/sensor.py#L295-L333)) is the single source of truth that the perceptual-noise system consumes ([sensor.py:206](../../src/environment/sensor.py#L206)). Every entry must have a matching `perceptual_noise.modalities.<key>` block, otherwise `apply_perceptual_noise` raises `KeyError` (see [10_perceptual_noise.md](../environment/10_perceptual_noise.md) FAQ). The new sensor must be registered in both places.

### Where hidden state currently leaks

`get_observation` builds `obs_parts` unconditionally for `Injury` and `Nutrition`:
```python
# sensor.py:251-255
obs_parts.append(jnp.array([state.injury_level / params.max_injury]))
obs_parts.append(jnp.array([state.nutrition / params.max_nutrition]))
```

Gating these on the new flags removes them from the observation but keeps the underlying **state dynamics intact** — `update_body` ([core.py:44-113](../../src/environment/core.py#L44-L113)) still updates `injury_level`/`nutrition` exactly as today. They become hidden state (computed, never observed).

### Existing `injury_buffer` is NOT reused

`injury_buffer` ([state.py:39](../../src/environment/state.py#L39), [core.py:71-80](../../src/environment/core.py#L71-L80)) is a **damage-spreading queue** used by body dynamics: each new damage event is divided by `smoothing_duration` and added to every slot, then the front slot is applied to `injury_level` and the buffer rolls. It is *not* a history of past injury values. Reusing it would couple physiology smoothing to perceptual delay (one knob for two unrelated phenomena). Per the user's decision, we add a separate buffer that stores past `injury_level` values.

## Implementation Plan

### Design

#### Alpha kernel

For kernel length `K` and time-to-peak `τ` (steps), discrete alpha kernel over indices `i = 0, 1, ..., K-1`:

```
k_raw[i] = (i / τ) * exp(1 - i / τ)         # k_raw[0] = 0, k_raw[τ] = 1
k[i]     = k_raw[i] / sum(k_raw)            # normalized: sum(k) = 1
```

Properties:
- `k[0] = 0` ⇒ injury at the *current* step contributes zero ⇒ clean perceptual delay (no instantaneous leak of hidden injury).
- Peak weight at `i = τ` ⇒ pain perception ramps up over τ steps after an injury event.
- Exponential decay for `i > τ` ⇒ pain fades smoothly.
- Normalization (sum=1) ⇒ a sustained injury at full intensity produces an interoceptive reading equal to that injury level (steady-state DC gain = 1). This makes the signal directly comparable to `injury_level / max_injury` in the limit, just delayed.

Recommended defaults: `τ = 3.0`, `kernel_length = 12` (covers ~4τ; tail weight <2%).

#### History buffer & convolution

A new `nociception_history_buffer` of shape `[K]` is added to `EnvState`. Index 0 is the most recent injury; index `K-1` is the oldest in the window.

After each step, the buffer rolls and the new injury is written to slot 0:
```python
new_buffer = jnp.roll(prev_buffer, 1).at[0].set(new_injury)
```

The sensor reads the buffer and computes:
```python
intero_noc = jnp.sum(buffer * params.interoceptive_kernel) / params.max_injury
```

This is a length-K causal FIR filter applied to the injury-level time series. With `k[0]=0`, the filter is delay-free of *current* injury but progressively integrates past injury.

#### Convolution disable (passthrough mode)

When `interoceptive_convolution_enabled=False`, `sense_interoceptive_nociception` short-circuits the kernel/buffer machinery and returns the current normalized injury directly:

```python
if params.interoceptive_convolution_enabled:
    convolved = jnp.sum(state.nociception_history_buffer * params.interoceptive_kernel)
    return jnp.array([convolved / jnp.maximum(params.max_injury, 1e-6)])
else:
    # Passthrough: intero == injury (no delay, no smoothing).
    return jnp.array([state.injury_level / jnp.maximum(params.max_injury, 1e-6)])
```

The buffer is still rolled in `update_body` regardless of the flag (so that toggling the flag at runtime/restart does not change `EnvState` shape). The `interoceptive_kernel` array is still computed at config load — when the flag is off, the loader produces a placeholder zero array of length `interoceptive_kernel_length` (the field stays present so `EnvParams` shape is invariant). The buffer/kernel waste is one length-`K` float vector per env — negligible.

Why this matters: even at τ=1 the alpha kernel concentrates ~75% of weight at lag 1 and 0% at lag 0, so the perceived signal is always at least one step behind the hidden injury. Passthrough is the only way to get `intero == injury` exactly. It is also the natural ablation baseline for studies measuring how much temporal smearing the convolution adds.

The perceptual-noise channel still applies to the intero slot in passthrough mode, which is what makes it functionally distinct from the raw `Injury` slice — researchers can keep `injury_observable=False` and use a noisy passthrough intero as the only injury-derived signal the agent sees.

#### Static vs runtime params

- `interoceptive_nociception_enabled`: static bool (gates obs slot — JIT recompile on toggle).
- `interoceptive_convolution_enabled`: static bool (selects passthrough vs alpha-kernel mode — JIT recompile on toggle).
- `interoceptive_kernel_length`: static int (controls buffer shape — JIT recompile on toggle).
- `interoceptive_kernel`: float array of shape `[K]` (runtime; alpha kernel when convolution enabled, zero placeholder when disabled).
- `injury_observable`, `nutrition_observable`: static bools (gate obs slots).

### File Changes

#### `src/environment/state.py`

**Add 1 field to `EnvState`** (next to existing body fields, around line 39):

```python
# BEFORE:
    injury_level: jnp.ndarray    # [] float
    injury_buffer: jnp.ndarray   # [smoothing_duration] float
    last_collision_noc: jnp.ndarray # float (intensity of last collision)

# AFTER:
    injury_level: jnp.ndarray    # [] float
    injury_buffer: jnp.ndarray   # [smoothing_duration] float
    nociception_history_buffer: jnp.ndarray  # [interoceptive_kernel_length] float (past injury_level values, idx 0 = most recent)
    last_collision_noc: jnp.ndarray # float (intensity of last collision)
```

**Add 6 fields to `EnvParams`** (next to the sensory block, after line 157):

```python
# AFTER existing `location_sensor_enabled`:
    injury_observable: bool = struct.field(pytree_node=False)
    nutrition_observable: bool = struct.field(pytree_node=False)
    interoceptive_nociception_enabled: bool = struct.field(pytree_node=False)
    interoceptive_convolution_enabled: bool = struct.field(pytree_node=False)
    interoceptive_kernel_length: int = struct.field(pytree_node=False)
    interoceptive_kernel: jnp.ndarray  # [interoceptive_kernel_length] float, normalized alpha kernel (zeros when convolution disabled)
```

#### `src/environment/sensor.py`

**1. New sensor function** (insert after `sense_extero_nociception`, around line 88):

```python
def sense_interoceptive_nociception(state: EnvState, params: EnvParams):
    """
    Tonic interoceptive pain. Two modes (selected statically at JIT time):

    - Convolution mode (default): convolves recent injury history with a
      normalized alpha kernel (peak at τ steps). Buffer slot 0 = most recent
      injury; kernel[0]=0 so current-step injury does not leak instantaneously.
    - Passthrough mode (`interoceptive_convolution_enabled=False`): bypasses
      the kernel and returns the current normalized injury directly. Use this
      for ablations where the agent should perceive injury without delay.

    Returns scalar in [0, 1].
    """
    if params.interoceptive_convolution_enabled:
        convolved = jnp.sum(state.nociception_history_buffer * params.interoceptive_kernel)
        return jnp.array([convolved / jnp.maximum(params.max_injury, 1e-6)])
    return jnp.array([state.injury_level / jnp.maximum(params.max_injury, 1e-6)])
```

> **Note**: the `if` is on a static field, so JAX traces only one branch per JIT compile — no runtime cost for the unused branch.

**2. Gate Injury and Nutrition; insert Interoceptive Nociception** in `get_observation` ([sensor.py:251-263](../../src/environment/sensor.py#L251-L263)):

```python
# BEFORE:
    # 1. Injury
    obs_parts.append(jnp.array([state.injury_level / params.max_injury]))

    # 2. Nutrition
    obs_parts.append(jnp.array([state.nutrition / params.max_nutrition]))

    # 3. Satiation
    obs_parts.append(jnp.array([state.satiation / params.max_satiation]))

    # 4. Extero Nociception (Phasic - Multi-source)
    if params.nociception_enabled:
        obs_parts.append(sense_extero_nociception(state.agent_pos, state, params))

# AFTER (interoceptive cluster grouped first; Extero stays exteroceptive):
    # 1. Injury (hidden when injury_observable=False) — interoceptive
    if params.injury_observable:
        obs_parts.append(jnp.array([state.injury_level / params.max_injury]))

    # 2. Nutrition (hidden when nutrition_observable=False) — interoceptive
    if params.nutrition_observable:
        obs_parts.append(jnp.array([state.nutrition / params.max_nutrition]))

    # 3. Satiation — interoceptive
    obs_parts.append(jnp.array([state.satiation / params.max_satiation]))

    # 4. Interoceptive Nociception — interoceptive
    #    (Tonic — delayed function of hidden injury, or passthrough if convolution disabled)
    if params.interoceptive_nociception_enabled:
        obs_parts.append(sense_interoceptive_nociception(state, params))

    # 5. Extero Nociception — exteroceptive (phasic, multi-source contact)
    if params.nociception_enabled:
        obs_parts.append(sense_extero_nociception(state.agent_pos, state, params))
```

**3. Mirror gating in `get_observation_breakdown`** ([sensor.py:295-333](../../src/environment/sensor.py#L295-L333)):

```python
# BEFORE:
    # 1. Injury
    breakdown["Injury"] = 1
    # 2. Nutrition
    breakdown["Nutrition"] = 1
    # 3. Satiation
    breakdown["Satiation"] = 1
    # 4. Extero Nociception
    if params.nociception_enabled:
        breakdown["Extero Nociception"] = 1

# AFTER (mirrors the new ordering in get_observation):
    # 1. Injury (hidden when injury_observable=False) — interoceptive
    if params.injury_observable:
        breakdown["Injury"] = 1
    # 2. Nutrition (hidden when nutrition_observable=False) — interoceptive
    if params.nutrition_observable:
        breakdown["Nutrition"] = 1
    # 3. Satiation — interoceptive
    breakdown["Satiation"] = 1
    # 4. Interoceptive Nociception — interoceptive (delayed/passthrough injury)
    if params.interoceptive_nociception_enabled:
        breakdown["Interoceptive Nociception"] = 1
    # 5. Extero Nociception — exteroceptive
    if params.nociception_enabled:
        breakdown["Extero Nociception"] = 1
```

> **Important**: insertion order in this dict is the **single source of truth** for the observation slice layout, the renderer tile order, and the perceptual-noise modality registration order. Keep all three in sync (see the `default.yaml` change below).

**4. Renderer hook** in `build_sensory_viz` ([sensor.py:354-394](../../src/environment/sensor.py#L354-L394)) — group the new sensor with the existing interoceptive cluster (Satiation/Nutrition/Injury) rather than next to the exteroceptive Extero Nociception branch. Update the existing tuple-membership branch:

```python
# BEFORE (existing — the interoceptive cluster):
        elif sensor_name in ("Satiation", "Nutrition", "Injury"):
            s_obs = float(obs[ptr])
            ptr += dim; t_ptr += dim
            viz.append({'name': sensor_name, 'intensity': s_obs, 'type': 'intensity'})

# AFTER (extend the membership tuple to include the new sensor; preserves
# the visual grouping in the dashboard. true_intensity is passed because
# this sensor IS in the perceptual-noise system, unlike the others in the
# tuple which were never noise-renderable in the existing code):
        elif sensor_name in ("Satiation", "Nutrition", "Injury", "Interoceptive Nociception"):
            s_obs = float(obs[ptr])
            s_true = float(true_obs[t_ptr]) if true_obs is not None else s_obs
            ptr += dim; t_ptr += dim
            display_name = "Intero Nociception" if sensor_name == "Interoceptive Nociception" else sensor_name
            color = "#8e44ad" if sensor_name == "Interoceptive Nociception" else None  # purple distinguishes pain
            tile = {'name': display_name, 'intensity': s_obs, 'true_intensity': s_true, 'type': 'intensity'}
            if color is not None:
                tile['color'] = color
            viz.append(tile)
```

Visual result: the dashboard renders the four interoceptive tiles in a single contiguous group — Injury, Nutrition, Satiation, Intero Nociception — followed by Extero Nociception elsewhere. Keep the existing `Extero Nociception` branch unchanged.

#### `src/environment/core.py`

**1. Update buffer in `update_body`** — extend the return tuple and roll-and-set the history buffer using the **new** `injury_level` (after recovery, after clipping). Insert just before the termination block ([core.py:96-101](../../src/environment/core.py#L96-L101)):

```python
# BEFORE (lines 95-101):
        new_injury = jnp.where(can_recover, new_injury - recovery_amount, new_injury)
        new_injury = jnp.clip(new_injury, 0.0, params.max_injury)
    else:
        new_injury = prev_injury
        new_buffer = state.injury_buffer
        new_rest_streak = prev_rest_streak

# AFTER:
        new_injury = jnp.where(can_recover, new_injury - recovery_amount, new_injury)
        new_injury = jnp.clip(new_injury, 0.0, params.max_injury)
    else:
        new_injury = prev_injury
        new_buffer = state.injury_buffer
        new_rest_streak = prev_rest_streak

    # Roll the perceptual history buffer and write the new injury at slot 0.
    # Buffer is non-conditional on `with_injury`: if injury never updates, slot 0 stays at prev_injury (0 from reset).
    new_nociception_history = jnp.roll(state.nociception_history_buffer, 1).at[0].set(new_injury)
```

**2. Extend the return tuple of `update_body`** ([core.py:113](../../src/environment/core.py#L113)):

```python
# BEFORE:
    return new_satiation, new_nutrition, new_injury, new_buffer, new_rest_streak, done

# AFTER:
    return new_satiation, new_nutrition, new_injury, new_buffer, new_nociception_history, new_rest_streak, done
```

**3. Receive the new value in `jax_step`** ([core.py:438](../../src/environment/core.py#L438)):

```python
# BEFORE:
    new_satiation, new_nutrition, new_injury, next_injury_buffer, new_rest_streak, done = update_body(state, info, params)

# AFTER:
    new_satiation, new_nutrition, new_injury, next_injury_buffer, next_nociception_history, new_rest_streak, done = update_body(state, info, params)
```

**4. Pass it through `state._replace`** ([core.py:493-517](../../src/environment/core.py#L493-L517)):

```python
# Add to the kwargs:
        injury_buffer=next_injury_buffer,
        nociception_history_buffer=next_nociception_history,   # ← NEW
        last_collision_noc=collision_noc,
```

**5. Initialize buffer in `jax_reset`** ([core.py:736-738](../../src/environment/core.py#L736-L738)):

```python
# BEFORE:
    injury_buffer = jnp.zeros(params.smoothing_duration)

# AFTER:
    injury_buffer = jnp.zeros(params.smoothing_duration)
    nociception_history_buffer = jnp.zeros(params.interoceptive_kernel_length)
```

**6. Add to EnvState constructor** ([core.py:768](../../src/environment/core.py#L768)):

```python
# Add after `injury_buffer=injury_buffer,`:
        nociception_history_buffer=nociception_history_buffer,
```

#### `src/environment/config_loader.py`

**1. Read new mandatory keys and build kernel** — insert in `load_env_params` near the sensory block (around the `nociception_size` line, before `EnvParams(...)` construction):

```python
# Hidden-state observability flags
injury_observable = bool(config.get_mandatory('sensory.injury_observable'))
nutrition_observable = bool(config.get_mandatory('sensory.nutrition_observable'))

# Interoceptive nociception (delayed-peak perception of hidden injury)
interoceptive_nociception_enabled = bool(config.get_mandatory('sensory.interoceptive_nociception_enabled'))
interoceptive_convolution_enabled = bool(config.get_mandatory('sensory.interoceptive_convolution_enabled'))
interoceptive_kernel_length = int(config.get_mandatory('sensory.interoceptive_kernel_length'))
interoceptive_kernel_tau = float(config.get_mandatory('sensory.interoceptive_kernel_tau'))

if interoceptive_convolution_enabled:
    # Build normalized alpha kernel: k_raw[i] = (i/τ)·exp(1 - i/τ); k[i] = k_raw[i] / Σk_raw
    _k_idx = np.arange(interoceptive_kernel_length, dtype=np.float32)
    _k_raw = (_k_idx / interoceptive_kernel_tau) * np.exp(1.0 - _k_idx / interoceptive_kernel_tau)
    _k_sum = float(_k_raw.sum())
    if _k_sum <= 0.0:
        raise ValueError(
            f"interoceptive_kernel produced non-positive sum ({_k_sum}). "
            f"Check tau ({interoceptive_kernel_tau}) and length ({interoceptive_kernel_length})."
        )
    interoceptive_kernel = jnp.array(_k_raw / _k_sum, dtype=jnp.float32)
else:
    # Passthrough mode — kernel is unused but kept as zeros for shape stability.
    interoceptive_kernel = jnp.zeros(interoceptive_kernel_length, dtype=jnp.float32)
```

**2. Pass to `EnvParams(...)`** — append to the constructor call ([config_loader.py:283-376](../../src/environment/config_loader.py#L283-L376)):

```python
# Add alongside the other sensory fields, e.g. after location_sensor_enabled:
    injury_observable=injury_observable,
    nutrition_observable=nutrition_observable,
    interoceptive_nociception_enabled=interoceptive_nociception_enabled,
    interoceptive_convolution_enabled=interoceptive_convolution_enabled,
    interoceptive_kernel_length=interoceptive_kernel_length,
    interoceptive_kernel=interoceptive_kernel,
```

**3. Register the new modality for perceptual noise** — extend `_YAML_KEY_TO_SENSOR_NAME` ([config_loader.py:378-388](../../src/environment/config_loader.py#L378-L388)):

```python
# BEFORE:
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

# AFTER (add interoceptive_nociception entry):
_YAML_KEY_TO_SENSOR_NAME = {
    "injury":                    "Injury",
    "nutrition":                 "Nutrition",
    "satiation":                 "Satiation",
    "extero_nociception":        "Extero Nociception",
    "interoceptive_nociception": "Interoceptive Nociception",
    "olfaction":                 "Olfaction",
    "collision":                 "Collision",
    "proprioception":            "Proprioception",
    "visual":                    "Visual",
    "location":                  "Location",
}
```

**4. Bump noise array padding from 12 → 13** ([config_loader.py:401](../../src/environment/config_loader.py#L401)):

```python
# BEFORE:
    pad = max(0, 12 - len(noise_modality_order))

# AFTER:
    pad = max(0, 13 - len(noise_modality_order))
```

**5. Update fixed array shape comments in `state.py`** for the noise arrays (around [state.py:170-174](../../src/environment/state.py#L170-L174)):

```python
# BEFORE:  noise_modes: jnp.ndarray  # [12] int32 ...
# AFTER:   noise_modes: jnp.ndarray  # [13] int32 ...
```
(repeat for `noise_sigmas`, `noise_injury_scales`, `noise_clip_min`, `noise_clip_max` — comment-only).

#### `configs/environment/default.yaml`

**1. Sensory block additions** ([default.yaml:266-285](../../configs/environment/default.yaml#L266-L285)):

```yaml
# AFTER existing sensory keys, append:
sensory:
  ...
  # Hidden-state observability gates (default true = backward compatible)
  injury_observable: true
  nutrition_observable: true
  # Interoceptive nociception: tonic, delayed perception of hidden injury via alpha-kernel convolution.
  # alpha kernel: k_raw[i] = (i/tau) * exp(1 - i/tau), normalized so sum(k)=1. k[0]=0 → no instantaneous leak.
  # Peak at i=tau steps. kernel_length should cover ~4*tau for low truncation error.
  interoceptive_nociception_enabled: true
  # When false, the sensor bypasses the alpha kernel entirely and emits
  # injury_level/max_injury directly (intero == injury, no delay, no smoothing).
  # Use for ablations or to recover the pre-convolution baseline. tau and
  # kernel_length below are still parsed for shape stability but unused.
  interoceptive_convolution_enabled: true
  interoceptive_kernel_tau: 3.0
  interoceptive_kernel_length: 12
```

**2. Perceptual-noise modality entry** ([default.yaml:289-347](../../configs/environment/default.yaml#L289-L347)) — insert `interoceptive_nociception` **between `satiation` and `extero_nociception`** so YAML key order (which is the noise-array index order) matches the new observation ordering with the interoceptive cluster contiguous:

```yaml
    satiation:       # index 2 — interoceptive
      mode: "state_dependent"
      sigma: 0.1
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 1.0
    # NEW — sits inside the interoceptive cluster
    interoceptive_nociception:  # index 3 — interoceptive (new)
      mode: "state_dependent"
      sigma: 0.1
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 1.0
    extero_nociception:  # index 4 (was 3) — exteroceptive
      mode: "state_dependent"
      sigma: 0.1
      injury_noise_scale: 1.5
      clip_min: 0.0
      clip_max: 100.0
    olfaction:       # index 5 (was 4)
      ...
```

> **Important**: YAML key order defines the modality index used by `noise_modality_order` and must match the insertion order in `get_observation_breakdown`. The implementing agent must shift the index comments for every modality after the insertion point (`extero_nociception`, `olfaction`, `collision`, `proprioception`, `visual`, `location` all bump by 1). Any sibling experiment configs that override `perceptual_noise.modalities` wholesale must also include `interoceptive_nociception` in the new position — otherwise `apply_perceptual_noise` raises `KeyError` at runtime ([sensor.py:206](../../src/environment/sensor.py#L206)). Configs that only override the *outer* `enabled` flag, or that inherit `modalities` from the default, are unaffected.

#### Documentation cross-references (after implementation)

- [docs/environment/01_state_and_params.md](../environment/01_state_and_params.md) — add `nociception_history_buffer` to the `EnvState` table; add 5 new params to the `EnvParams` table.
- [docs/environment/05_body_homeostasis.md](../environment/05_body_homeostasis.md) — add a "Hidden-state separation" section noting that `injury_level` and `nutrition` remain authoritative body state and become hidden when the new flags are off.
- [docs/environment/09_sensors_and_observation.md](../environment/09_sensors_and_observation.md) — update observation table (gate Injury/Nutrition rows; add Interoceptive Nociception row) and document `sense_interoceptive_nociception`.
- [docs/environment/10_perceptual_noise.md](../environment/10_perceptual_noise.md) — bump the modality count (9 → 10), update padding (12 → 13), add modality-order example.
- [docs/environment/ENVIRONMENT_SUMMARY.md](../environment/ENVIRONMENT_SUMMARY.md) — update Observation Layout Reference Table and Config-to-EnvParams mapping table.

## Checkpoints

- [x] **Kernel sanity** — after config load, print `params.interoceptive_kernel`. Verify `k[0]==0`, peak index ≈ `round(tau)`, `sum(k) ≈ 1.0`. [13:11:05]
- [x] **Obs shape** — with all defaults, `len(get_observation(state, params))` should equal previous default + 1 (new sensor adds 1 dim). [13:11:12]
- [x] **Hide flags** — set `injury_observable=false` and `nutrition_observable=false`. `get_observation_breakdown(params)` must omit "Injury" and "Nutrition" keys; obs vector shrinks by 2; first emitted slice is "Satiation". [13:11:21]
- [x] **Delay behaviour** — in a smoke episode where the agent takes a single damage hit at step `t0`: log `state.injury_level` and the new sensor reading per step. The sensor should be ~0 at `t0`, ramp up over τ steps, peak around `t0 + τ`, and decay smoothly. [13:13:05]
- [x] **Steady state** — drive injury to a constant value (no recovery). Sensor reading should converge to `injury_level / max_injury` (DC gain = 1 because kernel is normalized). [13:13:06]
- [x] **Passthrough mode** — set `interoceptive_convolution_enabled=false`. At every step the sensor reading must equal `state.injury_level / params.max_injury` *exactly* (modulo perceptual noise), with zero lag. Toggle on/off without changing `EnvState` shape (buffer still allocated). [13:13:08]
- [x] **Backward compat** — running with the unmodified default YAML (both `*_observable` flags = true) reproduces the same observation dim **+ 1** (the new intero sensor). If you also set `interoceptive_nociception_enabled=false`, obs dim must equal the pre-change default (regression check). [13:13:08]
- [x] **No-injury config** — with `with_injury=false`, the buffer stays at zero; sensor returns 0; no NaN/Inf. [13:13:08]
- [x] **Noise system** — confirm `apply_perceptual_noise` does not raise. Check that `params.noise_sigmas.shape == (13,)`. [13:13:08]
- [x] **Renderer grouping** — render one frame; confirm the four interoceptive tiles appear contiguously in this order: **Injury → Nutrition → Satiation → Intero Nociception**. The Extero Nociception tile must come *after* this group, not interleaved with it. Tile order should match `get_observation_breakdown` insertion order exactly. [13:13:08]
- [x] **Single-env smoke** — run `python -c "..."` (or the project's smoke-test entry) for ~50 steps without exceptions. [13:13:08]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-04-30 22:12:35

### `src/environment/state.py`
- Added `nociception_history_buffer` to `EnvState`.
- Added 6 new fields to `EnvParams`: `injury_observable`, `nutrition_observable`, `interoceptive_nociception_enabled`, `interoceptive_convolution_enabled`, `interoceptive_kernel_length`, `interoceptive_kernel`.
- Updated noise array size comments (12 -> 13).

### `src/environment/core.py`
- Updated `update_body` to return `new_nociception_history`.
- Integrated history buffer management in `jax_step`.
- Initialized `nociception_history_buffer` in `jax_reset`.

### `src/environment/sensor.py`
- Implemented `sense_interoceptive_nociception` with convolution and passthrough modes.
- Updated `get_observation` to gate `Injury`/`Nutrition` and insert `Interoceptive Nociception`.
- Updated `get_observation_breakdown` to match new layout.
- Updated `build_sensory_viz` to group and color the new sensor.

### `src/environment/config_loader.py`
- Added logic to load new sensory keys and compute the normalized alpha kernel.
- Updated `_YAML_KEY_TO_SENSOR_NAME` mapping.
- Bumped noise array padding to 13.

### `configs/environment/default.yaml`
- Added default values for new sensory parameters.
- Inserted `interoceptive_nociception` into noise modalities with state-dependent mode.
- Updated index comments for all subsequent modalities.

<!-- Filled by the implementing agent. -->

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-04-30

### Diff stats

```
 configs/environment/default.yaml               | 35 +++++++++---
 docs/environment/01_state_and_params.md        | 22 +++++---
 docs/environment/05_body_homeostasis.md        | 16 +++++-
 docs/environment/09_sensors_and_observation.md | 52 ++++++++++++------
 docs/environment/10_perceptual_noise.md        | 24 +++++----
 docs/environment/ENVIRONMENT_SUMMARY.md        | 24 +++++----
 src/environment/config_loader.py               | 56 ++++++++++++++++----
 src/environment/core.py                        | 13 +++--
 src/environment/sensor.py                      | 73 +++++++++++++++++++-------
 src/environment/state.py                       | 21 ++++++--
 10 files changed, 246 insertions(+), 90 deletions(-)
```

All five planned code-change targets touched, plus the five documentation cross-references called out in the plan. No out-of-scope files modified. Net line counts proportional to the planned scope.

### File-by-file

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/state.py` | +1 EnvState field, +6 EnvParams fields | ✅ | All fields present at correct positions; `nociception_history_buffer` typed and commented; static fields use `struct.field(pytree_node=False)`; noise-array shape comments bumped 12→13. |
| `src/environment/sensor.py` | new `sense_interoceptive_nociception` (with passthrough branch), gated Injury/Nutrition slices, breakdown + viz updates | ✅ | Function follows the planned static-`if` branch on `interoceptive_convolution_enabled`. `get_observation` and `get_observation_breakdown` both reordered so the interoceptive cluster (Injury, Nutrition, Satiation, **Interoceptive Nociception**) precedes Extero Nociception. `build_sensory_viz` extends the membership tuple with `"Interoceptive Nociception"`, applies purple `#8e44ad` color, sets display name to `"Intero Nociception"`, and now passes `true_intensity` for the whole interoceptive cluster. |
| `src/environment/core.py` | extend `update_body` return, roll-and-set buffer, init in reset, threading in `jax_step` | ✅ | Buffer roll placed at function-body level (correct — runs unconditionally on `with_injury`). Return tuple expanded; `jax_step` unpacks the new value and passes through `state._replace`; `jax_reset` initializes the buffer to zeros of length `params.interoceptive_kernel_length`. |
| `src/environment/config_loader.py` | read 6 new keys, compute alpha kernel (or zero placeholder when convolution disabled), register new modality, bump pad 12→13 | ✅ | All 6 keys read via `get_mandatory`. Alpha kernel built with normalization and a non-positive-sum `ValueError` guard; passthrough mode produces a zeros placeholder. `_YAML_KEY_TO_SENSOR_NAME` extended; pad bumped to 13. |
| `configs/environment/default.yaml` | new sensory keys (incl. `interoceptive_convolution_enabled`), new noise modality entry | ✅ | All 6 sensory keys added (defaults: `*_observable=true`, `interoceptive_nociception_enabled=true`, `interoceptive_convolution_enabled=true`, `tau=3.0`, `length=12`). Noise-modality block has `interoceptive_nociception` inserted between `satiation` and `extero_nociception`; downstream index comments shifted by +1. |
| `docs/environment/01_state_and_params.md` | EnvState + EnvParams tables, reset table, FAQ | ✅ | New row for `nociception_history_buffer`; 6 new EnvParams rows; reset table extended; FAQ updated 12→13. |
| `docs/environment/05_body_homeostasis.md` | new "Hidden Body States" section | ✅ | Documents the masking flags and the `nociception_history_buffer` push in `core.py:102`. |
| `docs/environment/09_sensors_and_observation.md` | observation table + new "Interoceptive Nociception" section | ✅ | Reordered the per-slice table to match the implementation; documents both convolution and passthrough modes. ⚠️ Uses informal flag name `intero_enabled` in two table cells — actual EnvParams field is `interoceptive_nociception_enabled` (cosmetic doc nit). |
| `docs/environment/10_perceptual_noise.md` | modality count, padding, FAQ | ✅ | Modality table updated; pad 12→13; index table extended. ⚠️ Two stale references: line 119 still reads "indices 0–8" (should be "0–9"); line 183 says "a eleventh modality" (grammatical, should be "an eleventh"). Both are doc-only nits. |
| `docs/environment/ENVIRONMENT_SUMMARY.md` | step-flow diagram, obs table, EnvParams mapping table, FAQ | ✅ | Stage-5 read/write list now includes `nociception_history_buffer`; obs table reordered; 5 new YAML→param rows added. ⚠️ Same `intero_enabled` informal name used (cosmetic). |

### Functional verification (live smoke run with `get_default_config`)

| Check | Result |
|---|---|
| Kernel: shape `(12,)`, `k[0]=0.0`, `sum(k)=1.000000`, `argmax(k)=3` (matches τ=3) | ✅ |
| Breakdown order = `Injury → Nutrition → Satiation → Interoceptive Nociception → Extero Nociception → Olfaction → Collision → Proprioception → Visual` | ✅ |
| Default obs dim = 29 (pre-change baseline 28 + 1 for new sensor) | ✅ |
| `params.noise_sigmas.shape == (13,)` | ✅ |
| State `nociception_history_buffer.shape == (12,)`, initialized to zeros | ✅ |
| Convolution mode: injury just landed at slot 0 → sensor returns 0.0 (no instantaneous leak) | ✅ |
| Convolution mode: injury at slot 3 (kernel peak) → sensor returns `50 · k[3] / 100 = 0.0692` exactly | ✅ |
| Convolution mode: sustained injury=50 across buffer → sensor returns `0.5` (DC gain = 1) | ✅ |
| Passthrough mode (`interoceptive_convolution_enabled=False`): injury=75 with empty buffer → sensor returns `0.75 = injury/max_injury` | ✅ |
| Hide flags (`injury_observable=False`, `nutrition_observable=False`): obs dim = 27, first slice is `Satiation` | ✅ |
| Disable intero (`interoceptive_nociception_enabled=False`): obs dim = 28 (matches pre-change default) | ✅ |

### Out-of-scope changes

None. All 10 modified files were either explicitly listed in the plan's File Changes section or named under "Documentation cross-references".

### Minor nits (non-blocking)

1. `docs/environment/10_perceptual_noise.md:119` — "indices 0–8" should now read "indices 0–9" (10 active modalities → indices 0..9).
2. `docs/environment/10_perceptual_noise.md:183` — "Can I add **a eleventh** modality?" — should be "**an eleventh**".
3. `docs/environment/09_sensors_and_observation.md` and `ENVIRONMENT_SUMMARY.md` — observation tables list the gating flag as `intero_enabled` (informal); the actual EnvParams field is `interoceptive_nociception_enabled`. Cosmetic only.

These do not affect behavior and can be patched in a follow-up doc-only edit.

**Conclusion**: ✅ **Implementation matches the plan and all 11 checkpoints pass.** Convolution mode produces the expected delayed-peak alpha-kernel response; passthrough mode produces `intero == injury` exactly; hide flags drop the corresponding obs slices; the new sensor sits inside the interoceptive cluster as required; noise array padding correctly bumped to 13. Only three small doc-text nits remain (listed above) — none functional.

---

## Appendix: Why the alpha kernel (vs alternatives)

The alpha function `α(t) = (t/τ)·exp(1 − t/τ)` is the canonical model of post-synaptic / EPSP-like response and matches the qualitative C-fiber pain time-course: zero at onset, smooth rise, peak at τ, exponential decay.

Compared to the rejected options:
- **Exponential IIR** has no rise time — pain reads non-zero immediately, defeating the "delayed perception" goal.
- **Boxcar** has a flat lag profile (no peak) and a sharp turn-on/off, less biological.
- **Gaussian FIR** is symmetric so it has acausal-like ringing (peak occurs *before* injury would in the buffer indexing), and needs two parameters.

A single τ also keeps the YAML surface minimal. The sum-normalization choice is deliberate: it makes the steady-state interoceptive reading numerically comparable to the (now hidden) injury level, which simplifies reward design and noise calibration.
