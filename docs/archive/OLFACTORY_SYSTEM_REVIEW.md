# Olfactory System Review

> **Date**: 2026-02-24
> **Scope**: Review of the chemical sensing (olfactory) system implementation, configuration, and integration.

---

## 1. [Current] Core Architecture

The olfactory system is a **gradient-based distal sensor** that allows the agent to detect the "chemical signature" (property vector) of entities across the grid. It acts as the primary long-range navigation aid, complementing the short-range Visual and Manhattan-based Collision sensors.

### 1.1 Mathematical Model

The sensor implements a weighted summation of chemical properties from all active entities within a specified radius. The intensity of an entity's signal at the agent's position follows an inverse-power law decay:

$$I = \sum_{i \in \text{Entities}} \text{Property}_i \cdot \text{Decay}(d_i)$$

Where:
- $d_i$ is the Euclidean distance between the agent and entity $i$.
- $\text{Decay}(d) = \frac{1}{d^p + \epsilon}$ (where $p$ is `sensor_decay`).
- Signals are masked by `sensor_radius`: if $d_i > \text{radius}$, the signal is zero.

### 1.2 Implementation Details (`sensor.py`)

The core logic resides in `sense_resource`:

```python
def sense_resource(agent_pos, res_pos, res_active, res_property, radius, decay_power):
    diff = res_pos - agent_pos
    dist = jnp.linalg.norm(diff, axis=-1)
    
    # Handle singularity (agent ON resource)
    decay = jnp.where(dist < 0.001, 2.0, 1.0 / (jnp.power(dist, decay_power) + 1e-10))
    
    # Mask by radius and activity
    mask = jnp.logical_and(res_active, dist <= radius)
    
    # Apply mask and sum
    weighted_props = res_property * decay[:, None] * mask[:, None]
    obs = jnp.sum(weighted_props, axis=0)
    return obs
```

**Key Features**:
- **Singularity Handling**: If the distance is near zero, the decay multiplier is capped at `2.0` to prevent infinite gradients.
- **Vectorized Summation**: Uses JAX's vectorized operations to process all entities of a type (e.g., all 50 food items) in a single pass.
- **Dimensionality**: The output is a vector of size `olfactory_vector_size` (typically 5), representing different "smell" channels (e.g., Food, Danger, Predator, Tree).

---

## 2. [Current] Multi-Source Integration

The full olfactory observation is an **additive mixture** of chemical signatures from four distinct entity categories:

| Source Category | Implementation Detail |
|-----------------|-----------------------|
| **Resources** | Food and Danger items. Active only when not consumed. |
| **Predators** | Always active. Allows detection of approaching threats before they enter visual range. |
| **Obstacles** | Static entities like Trees. Rocks are currently olfactory-silent (`[0,0,0,0,0]`). |
| **Neutral Animals**| Serve as "olfactory decoys" or environmental noise. |

In `get_observation`, these are combined:
```python
obs_parts.append(res_chem + pred_chem + obs_chem + neutral_chem)
```

---

## 3. [Current] Configuration Parameters

The system is highly parameterized, allowing for different sensing "physics" per level:

| Parameter | Type | Description |
|-----------|------|-------------|
| `sensor_radius` | float | Max distance at which a signal can be detected. |
| `sensor_decay` | float | The power ($p$) in the inverse-decay law (e.g., 2.0 for quadratic decay). |
| `olfactory_enabled` | bool | Global toggle for the sensor. |
| `res_property` | ndarray| Vector defined in YAML for each resource type (e.g., `[1,0,0,0,0]` for Food). |
| `pred_property` | ndarray| Chemical signature for predators. |

---

## 4. [Current] Perceptual Precision & Noise

The olfactory system is subject to the **Perceptual Precision Modulation** system. During training and evaluation, Gaussian noise can be added to the olfactory signal.

### 4.1 State-Dependent Noise
If `noise_mode` for Olfaction is set to `2` (State-Dependent), the variance of the noise increases as the agent's `injury_level` increases:

$$\sigma_{\text{eff}} = \sigma_{\text{base}} \cdot (1 + \alpha \cdot \text{Injury}_{\text{norm}})$$

This simulates "sensory degradation" when the agent is damaged. The noise is clipped by `noise_clip_min` and `noise_clip_max` to prevent extreme outliers from destabilizing the agent's neural network.

---

---

## 5. [Discussion] Closest-Object Refinement

> [!NOTE]
> The following section documents ongoing research and proposed architectural refinements discussed with the professor. These features are **not currently implemented** in the active codebase.

A refined model is under discussion to address technical artifacts (such as "clumping") in the current summation-based system.

### 5.1 The "Clumping" Problem (Current System)
In the current implementation, signals are linearly additive. This creates a "clumping" effect where a large cluster of distant resources can produce a signal intensity identical to a single, much closer resource.
- **Navigational Risk**: The agent may be mathematically "pulled" toward distant resource clusters while ignoring lower-density but more immediately critical items.

### 5.2 Proposed Solution: Hard Winner-Takes-All (WTA)
The proposed solution transitions the sensor to a **Hard WTA** model, where only the single closest active entity of each type is reported at any given time.
- **Benefit**: Provides a much sharper navigational gradient and prevents "clumping" artifacts.
- **Technical Risk: Signal Jitter**: A major risk with Hard WTA is the **discontinuity** of the signal. If an agent is halfway between two objects, the reported signal can instantly "flip" 180 degrees as the agent moves only locally.

#### Background: Jitter in Robotics and RL
In robotics and control systems, this phenomenon is often referred to as "Chattering" (sliding mode control) or "Bang-Bang" instability. When a system uses discrete logical selection (like `argmin`) to drive a continuous control policy (a Neural Network), it introduces **mathematical discontinuities**.
- **Robotics**: A robot arm trying to decide between two targets may physically vibrate or "stutter" if the decision logic toggles at high frequency.
- **RL Training**: Discontinuous inputs create "spiky" loss landscapes. A tiny step in the environment can lead to a massive change in the agent's observation, making it difficult for the optimizer to find a stable policy.

#### Case Study: The Equidistant Trap
Imagine the agent is at $(0, 0)$ with two Food items:
- **Food A** at $(-1, 0)$
- **Food B** at $(+1, 0)$

1. **Step 1**: Agent moves slightly left. `dist(A) < dist(B)`. Scent input is $\vec{Scent}_A$ (Directly West).
2. **Step 2**: Agent oscillates slightly right due to noise or a secondary goal. Suddenly `dist(B) < dist(A)`. Scent input instantly "teleports" to $\vec{Scent}_B$ (Directly East).

The agent's "brain" receives a 180-degree phase shift in a single step. Without sophisticated internal state (memory), the agent may lose its navigational heading or enter an infinite loop of indecision.

### 5.3 Recommendation: Soft-Max (Soft Selection)
To achieve the benefits of the professor's "closest-only" goal while maintaining mathematical stability, a **Soft-Max Selection** is recommended. By using a soft-selection based on inverse distance, we can ensure:
1.  **Dominance**: The closest object provides the vast majority (e.g., 95%) of the signal.
2.  **Continuity**: Transitions between objects are smooth rather than instantaneous flips, allowing the agent to learn stable cross-boundary behaviors.

---

## 6. Limitations & Future Improvements

1.  **Linear Summation**: Signals sum linearly. In some environments, strong signals (e.g., a massive food pile) might saturate the sensor or mask weaker signals (e.g., a nearby predator).
2.  **Euclidean Distance only**: The system does not account for wind or obstacles blocking smells (diffusion).
3.  **Identical Channels**: If multiple resources share the same `res_property` vector, they are indistinguishable to the olfactory sensor.
4.  **Static Decay**: The decay power $p$ is global for all entities. Future iterations could allow different "scent volatility" per entity type.

---

## 6. Summary

The olfactory system provides a robust, JAX-native implementation of long-range chemical sensing. Its additive nature makes it efficient but demands careful balancing of `res_property` vectors to avoid channel crosstalk. Its integration with the injury-based noise system provides a key homeostatic pressure, forcing the agent to maintain high health to ensure reliable navigation signals.
