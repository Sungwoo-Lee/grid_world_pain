# Intensive Review: Structural Evolution of the Dreamer Algorithm

**Date**: February 22, 2026  
**Subject**: Advanced Mechanics from World Models to DreamerV3  
**Audience**: Graduate Students / RL Researchers

---

## 1. The Core Paradigm: Latent Dynamics

The fundamental innovation of the Dreamer lineage is the **Recurrent State Space Model (RSSM)**. Unlike a standard RNN which collapses past information into a single deterministic vector, RSSM models the world as a sequence of both deterministic and stochastic variables.

### 1.1 Mathematical Formulation of RSSM
For each timestep $t$, the state is defined by specific interactions:

1.  **Deterministic State ($h_t$)**: Models historical context.
    $$h_t = \text{Deterministic}(h_{t-1}, z_{t-1}, a_{t-1})$$
    *Implementation: `LayerNormGRUCell` in `sheeprl/algos/dreamer_v3/agent.py`.*

2.  **Stochastic State (Posterior $z_t$)**: Incorporates current sensory input $o_t$.
    $$z_t \sim q_\phi(z_t \mid h_t, \text{Encoder}(o_t))$$
    *Implementation: `RepresentationModel` (Categorical Linear layers).*

3.  **Stochastic State (Prior $\hat{z}_t$)**: Predicts the next state without seeing the observation.
    $$\hat{z}_t \sim p_\theta(\hat{z}_t \mid h_t)$$
    *Implementation: `TransitionModel`.*

---

## 2. Generational Shifts: Loss Functions

### 2.1 DreamerV1: Gaussian & Free Nats
V1 used Gaussian distributions for $z_t$. To prevent the posterior $q$ from overpowering the prior $p$, a "Free Nats" threshold was used:
$$\mathcal{L}_{KL} = \max(\text{FreeNats}, KL(q \Vert p))$$
This allowed the model to ignore unimportant sensory details if the KL was already low.

### 2.2 DreamerV2: Categorical & KL Balancing
V2 introduced **discrete latents** ($32 \times 32$ categoricals) and **KL Balancing**. KL Balancing forces the prior to move toward the posterior faster than the posterior moves toward the prior:
$$\mathcal{L}_{KL} = \alpha KL[\text{sg}(q) \Vert p] + (1-\alpha) KL[q \Vert \text{sg}(p)]$$
Where $\alpha = 0.8$ (default in `sheeprl`). This prevents the posterior from collapsing before the prior can learn the structure of the world.

### 2.3 DreamerV3: Unimix & Free Bits
V3 adds **Unimix** to stabilize discrete gradients. It mixes the categorical distribution with a uniform distribution (1%) to ensure no probability ever reaches zero:
$$P_{unimix}(z) = (1 - 0.01) P(z) + 0.01 \cdot \text{Uniform}$$
*Implementation: `agent.py:L437` (`_uniform_mix` method).*

---

## 3. Mastering Stability: The DreamerV3 "Bag of Tricks"

### 3.1 Symlog Transformation
To handle unbounded reward scales and observation intensities, DreamerV3 uses the **Symlog** function.
$$\text{symlog}(x) = \text{sign}(x) \ln(1 + |x|)$$
This compresses the dynamic range of inputs, protecting the network from exploding gradients when transitioning between highly different sensory environments.

### 3.2 Two-Hot Regression (Softmax Classification)
Instead of predicting a scalar $y$ via Mean Squared Error (MSE), V3 uses a discrete distribution over a fixed set of bins $B$.
1.  **Encoding**: A target value $x$ is encoded into two adjacent bins $b_i, b_{i+1}$:
    $$w_i = \max(0, 1 - |x - b_i| / \Delta)$$
2.  **Loss**: The critic is trained to minimize the Cross-Entropy between its predicted distribution and this two-hot target.

#### Practical Example: Coding $x = 10.5$
Assume we have bins from $[-20, 20]$ with a bin width $\Delta = 1.0$.
-   **Step 1**: Find the neighbors. $10.5$ sits between $10.0$ and $11.0$.
-   **Step 2**: Calculate distance-based weights.
    -   Distance to $10.0$ is $0.5$. Weight for bin $11.0$ is $0.5$.
    -   Distance to $11.0$ is $0.5$. Weight for bin $10.0$ is $0.5$.
-   **Result**: The "Target" vector is all zeros except for $0.5$ at index $(10)$ and $0.5$ at index $(11)$.
-   **Reasoning**: This prevents the network from "averaging" multi-modal targets (which would happen with MSE) and keeps the gradients bounded by the softmax temperature.

*Implementation: `sheeprl/utils/distribution.py` (`TwoHotEncodingDistribution` class).*

### 3.3 Percentile Return Normalization
To handle the "Value Scale" problem without task-specific tuning, V3 normalizes the advantage $A_t$ using the 5th and 95th percentiles of the return distribution:
$$S = \text{EMA}(\text{Percentile}_{95}) - \text{EMA}(\text{Percentile}_{5})$$
$$A_{norm} = \frac{V^\lambda - V}{S}$$
*Implementation: `Moments` class in `sheeprl/algos/dreamer_v3/utils.py`.*

---

## 4. Why This Works: Inductive Biases
*   **Discrete Latents**: Act as an information bottleneck. They provide a "concept" based representation rather than a pixel-perfect reconstruction, which aids in generalization across visually different tasks.
*   **Scale Invariance**: By turning regression into classification (Two-Hot) and normalizing returns (Moments), DreamerV3 is mathematically "scale-blind"—it treats a reward of $1$ and $1,000,000$ identically in terms of gradient magnitude.

---

## 5. Summary Tracking for Students

| Version | Feature | `sheeprl` Location |
| :--- | :--- | :--- |
| **v1** | Gaussian Posteriors | [loss.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/sheeprl/sheeprl/algos/dreamer_v1/loss.py) |
| **v2** | 32x32 One-Hot Latents | [agent.py:L344](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/sheeprl/sheeprl/algos/dreamer_v2/agent.py#L344) |
| **v3** | Symlog + Two-Hot | [distribution.py:L224](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/sheeprl/sheeprl/utils/distribution.py#L224) |
| **v3** | Percentile Normalization | [utils.py:L40](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/sheeprl/sheeprl/algos/dreamer_v3/utils.py#L40) |

---

## 5. Diagnostic Guide: Debugging for Students

When implementing or tuning Dreamer, these are the most common failure modes:

### 5.1 Vanishing Latent Information
*   **Symptoms**: The `observation_loss` decreases, but the agent fails to perform any task.
*   **Cause**: The KL term is too large, or FreeNats is too high, causing the model to prioritize "boring" prior matching over accurate sensory state modeling.
*   **Fix**: Lower `kl_regularizer` or decrease `kl_free_nats` to force a higher information bottleneck capacity.

### 5.2 Exploding Value Estimates
*   **Symptoms**: Reward loss stays low, but `value_loss` spikes to infinity.
*   **Cause**: Symlog not applied to the critic targets, or the Two-Hot bin range $(low, high)$ is too narrow for the environment's return scale.
*   **Fix**: Verify `SymlogDistribution` is wrapping your MLP decoder outputs, and extend bin ranges in `TwoHotEncodingDistribution`.

### 5.3 Deterministic Policy Collapse
*   **Symptoms**: Action entropy drops to exactly $0.0$, and the agent spins in circles.
*   **Cause**: Policy entropy coefficient is too low, or the world model is "too perfect," leaving no room for stochastic exploration.
*   **Fix**: Increase `ent_coef` in the actor config and ensure `Unimix` is enabled ($0.01$).

---

> [!NOTE]
> **Implementation Warning**: 
> When training your own agent, if you see the **State Entropy** dropping to zero rapidly, it indicates a distribution collapse. Check the `Unimix` mixture ratio. If the agent ignores rewards, verify if `symlog` is correctly applied to the reward inputs.
