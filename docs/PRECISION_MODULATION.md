# Research Proposal: Perceptual Precision Modulation

## 1. Theoretical Framework
In computational neuroscience and **Active Inference**, "Perceptual Precision" refers to the reliability or confidence assigned to sensory data. Mathematically, it is the **inverse variance** ($\pi = 1/\sigma^2$) of the likelihood distribution $P(o|s)$.

### High Precision ($\uparrow \pi$, $\downarrow \sigma^2$)
Sensory data is "trusted." The agent's posterior belief is strongly driven by observations, allowing for rapid environment tracking but making the agent vulnerable to sensory artifacts.

### Low Precision ($\downarrow \pi$, $\uparrow \sigma^2$)
Sensory data is ignored or "blurred." The agent relies more on its **prior (internal model)** and temporal integration. This is biologically observed during high-arousal states, severe injury, or high-velocity movement.

---

## 2. Mathematical Models for Noise

We propose three levels of implementation for the Grid World environment:

### A. Constant Additive Gaussian (Baseline)
For continuous sensors (Chemical, Interoception):
$$o_{noisy} = o_{true} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2_{fixed})$$

### B. Weber-Fechner Scaling (Intensity-Dependent)
Noise scales with the intensity of the signal (common in biological sensory systems):
$$\sigma = \sigma_{base} \cdot o_{true}$$

### C. State-Dependent Modulation (Research Focus)
Precision is modulated by the agent's internal state $s_{int}$ (e.g., `injury_level`):
$$\sigma_{dynamic} = \sigma_{base} \cdot (1 + \alpha \cdot \text{Injury})$$
where $\alpha$ is a sensitivity coefficient. High injury $\rightarrow$ higher noise $\rightarrow$ lower precision.

---

### 3. Modular Implementation (Heterogeneous Noise)

To apply different noise levels to different observation types (**Olfaction** vs. Interoception vs. Visual), we utilize a **Vectorized Sigma Mask**.

#### The "Mix and Match" Logic
The global `modulation_type` defines the **complexity budget** of the environment, but the specific behavior of each sensor is determined by its individual $(\sigma, \alpha)$ parameters.

*   **To make a sensor Deterministic**: Set its `mode` to `"none"`.
*   **To make a sensor Constant (No Injury Effect)**: Set its `mode` to `"constant"`.
*   **To make a sensor State-Dependent**: Set its `mode` to `"state_dependent"`.

This allows you to have an environment where, for example, **Olfaction sensing is state-dependent** but **Location sensing is constant**.

### 1. The Sigma Vector
Since the observation $o$ is a concatenated vector, we define a corresponding $\vec{\sigma}$ vector of the same dimension:
$$\vec{\sigma} = [\sigma_{olf}, \dots, \sigma_{olf}, \sigma_{noc}, \sigma_{coll}, \dots, \sigma_{intero}, \dots]$$

### 2. Implementation in JAX
Using the `get_observation_breakdown` utility, we can construct this sigma vector during environment initialization:

```python
def get_sigma_vector(params: EnvParams):
    breakdown = get_observation_breakdown(params)
    sigmas = []
    for sensor, dim in breakdown.items():
        # Get specific sigma from config for this sensor
        s = getattr(params.noise_sigmas, sensor.lower().replace(" ", "_"))
        sigmas.append(jnp.full((dim,), s))
    return jnp.concatenate(sigmas)
```

---

## 4. Implementation Design (JAX-Native)

### Core Changes in `EnvParams`
We will add a new PyTree-node-free configuration structure:
```python
@struct.dataclass
class NoiseParams:
    enabled: bool
    base_sigma_olf: float
    base_sigma_intero: float
    injury_noise_scale: float
```

### Sensor Modification logic in `sensor.py`
We will wrap `get_observation` with a noise utility:

```python
def apply_precision_modulation(obs, state: EnvState, params: EnvParams, key: jax.random.PRNGKey):
    # Split key for JIT-safe randomness
    noise_key = jax.random.split(key)[0]
    
    # Calculate Dynamic Sigma based on Injury level
    # sigma = base_sigma * (1.0 + sense_params.alpha * state.injury_level)
    
    # Generate Gaussian Noise
    noise = jax.random.normal(noise_key, shape=obs.shape) * sigma_vector
    return obs + noise
```

---

---

## 5. Exemplary Standardized Configuration (`perceptual_noise.yaml`)

Every modality follows the same schema for clarity and consistency.

```yaml
perceptual_noise:
  enabled: true
  
  modalities: # These modalities are mapped to indices 0-8 in internal noise parameters.
    olfaction:
      mode: "state_dependent"  # none, constant, state_dependent
      sigma: 0.15
      injury_noise_scale: 2.0  # Used in state_dependent mode
      
    satiation:
      mode: "constant"
      sigma: 0.1
      injury_noise_scale: 0.0 # Not used in constant mode
      
    nutrition:
      mode: "constant"
      sigma: 0.1
      injury_noise_scale: 0.0 # Not used in constant mode
      
    injury:
      mode: "state_dependent"
      sigma: 0.05
      injury_noise_scale: 1.5 # Used in state_dependent mode
      
    extero_nociception:
      mode: "none"
      sigma: 0.0               # Not used in none mode
      injury_noise_scale: 0.0
      
    visual:
      mode: "state_dependent"
      sigma: 0.05
      injury_noise_scale: 3.0
      
    location:
      mode: "constant"
      sigma: 0.001
      injury_noise_scale: 0.0
```

### Implementation Logic Update
With this schema, the `EnvParams` will hold a structured mapping. The `get_observation` function will iterate (or use a vectorized lookup) to apply the specific noise model requested for each slice of the observation vector.

### Usage in Research
By varying `injury_noise_scale`, you can simulate different "perceptual regimes"—from agents that are nearly immune to pain-induced confusion to those whose world becomes "illegible" the moment they take damage.
