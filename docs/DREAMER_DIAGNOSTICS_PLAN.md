# Diagnostic Plan: DreamerV3 Performance Investigation

The objective is to identify why DreamerV3 (and its neuromodulated variant) is performing significantly worse than Recurrent PPO in the `grid_world_pain` environment.

## 1. World Model (RSSM) Audit
If the World Model cannot predict the future or the rewards, the Actor cannot plan effectively.
- **[ ] Reconstruction Loss**: Verify if `observation_loss` is decreasing. If pixels/sensors aren't reconstructed, latents are useless.
- **[ ] Reward Prediction**: GridWorld rewards are often sparse. Check if `reward_loss` is converging or if it's trapped in a local minimum (predicting 0 reward everywhere).
- **[ ] KL Balancing**: Ensure KL loss isn't collapsing ($KL \approx 0$) or exploding. Collapsed KL means the posterior is just the prior (no information from observation).

## 2. Actor-Critic Policy Audit
- **[ ] Imagined Entropy**: Measure the entropy of the actor's policy during training. If it's too low too early, it's stuck in a suboptimal deterministic policy.
- **[ ] Value Accuracy**: Compare imagined values vs. actual returns. If the Critic is wrong, the Actor gradients are noise.
- **[ ] Action Distribution**: Check if the agent is outputting a diverse range of actions or is stuck in one direction (e.g., always going 'Up').

## 3. Configuration & Scaling Gaps
- **[x] Learning Rate Comparison**: Recurrent PPO uses $5 \times 10^{-4}$ for the Actor. Dreamer uses $8 \times 10^{-5}$. We might need to bump the Actor LR.  *(Resolved: Migrated to canonical 1e-4/3e-5 split with corrected Adam epsilons)*.
- **[x] Entropy Scale**: Recurrent PPO uses `0.01`. Dreamer uses `3e-4` (Hafner's default). For GridWorld, this might be insufficient for early discovery. *(Resolved: Verified `loss_actor_entropy` logic and aligned with 3e-4)*.
- **[x] Batch Interaction**: Recurrent PPO updates every 2000 steps. Dreamer iterates every step (or `train_steps`). Check the "Data-to-Update" ratio. *(Resolved: Identified Replay Ratio Inflation as a primary performance driver when reducing envs)*.
- **[ ] Symlog Interaction**: Verify that the rewards in GridWorld (which might be small or specific) are correctly handled by the Symlog transform.

## 4. Neuromodulation Interference
- **[ ] Baseline vs Modulated**: Run a baseline DreamerV3 (no modulation) to see if the neuromodulation component is the source of instability.
- **[ ] Modulator Weights**: Check if the modulator outputs are saturating (e.g., all 0s or all 1s).

## 5. Execution Steps
1. **Pilot Run (Baseline)**: Run `dreamer_v3.yaml` (unmodulated) for 100k steps and monitor WandB.
2. **Log Audit**: Insert specific metrics for Reward Accuracy and KL balancing into `dreamer_v3_trainer.py`.
3. **Hyperparameter Sweep**: Increase Actor LR and decrease model capacity (MLP depth) to match environment complexity.

---

## 6. Resolved Diagnostics (Feb 22-23 Verifications)
- **[x] Action Space Mapping**: Investigated action distributions showing 100% "Forward". Identified and resolved an off-by-one labeling error in `--plot-all` generation within `agentActionAnalysis.py`. The agent *is* exploring.
- **[x] Latent Imagination Verification**: Audited `behavior_loss_fn` and confirmed the Actor/Critic correctly learn from a 15-step purely latent rollout via `RSSM.imagine_step`, detached from real observations.
- **[x] JAX-Specific RSSM Stochasticity**: Addressed a critical state-overlap issue by implementing $T \times B$ `jax.vmap` PRNG splitting in `OneHotDist`, ensuring that batches with identical logits still sample stochastically independent latent prior/posterior states.

## 7. Parallel Environment Audit (Feb 24 - Deep Dive Investigation)

Significant performance gains were observed when reducing the number of parallel environments (e.g., from 64 to 1). The audit identifies deep architectural and implementation gaps that favor low-parallelism runs in the current codebase.

### 7.1 Replay Ratio Inflation & Sample Intensity
In `train.py`, `train_steps` and `batch_size` are independent of `num_envs` ($B$).
*   **Mathematical Context**: 
    *   Steps Collected per Iteration: $N_{steps} = B \times \text{sequence\_length}$
    *   Updates per Iteration: $U = \text{train\_steps}$
    *   **Replay Ratio** (Intensity): $\frac{U \times \text{batch\_size}}{N_{steps}}$
*   **Comparison**:
    *   **64 Envs**: $\frac{1 \times 64}{64 \times 128} = 0.0078$ updates per step.
    *   **1 Env**: $\frac{1 \times 64}{1 \times 128} = 0.5$ updates per step.
*   **Finding**: The 1-env agent undergoes **64x more gradient updates per environment interaction time-step**. This explains why it survives longer in fewer real-time steps—it is simply training much harder on less data.

### 7.2 Dataset Persistence (Memory Flush Rate)
The `ReplayBuffer` has a fixed `capacity` ($C = 10^5$).
*   **Data Turnover**: The number of iterations before the buffer is completely overwritten is $I_{flush} = \frac{C}{B \times \text{sequence\_length}}$.
*   **64 Envs**: $I_{flush} \approx 12$ iterations. The model never sees transitions older than ~1500 steps.
*   **1 Env**: $I_{flush} \approx 781$ iterations. The model trains on a massive historical diversity.
*   **Finding**: High parallelism leads to catastrophic forgetting or overfitting to the current local policy's "distribution drift."

### 7.3 Stochasticity Collapse (The PRNG Key Bug)
In `dreamer_v3_trainer.py:L537-539`:
```python
current_key, act_key = jax.random.split(current_key)
action_idx, next_d_state = self.get_action(obs, d_state, eval_mode=False, rng=act_key)
```
*   **The Issue**: A single `act_key` is passed to `get_action`, which handles a batch of $B$ observations. Inside `OneHotDist.sample` in `dreamer_v3_util.py`, the code only vectorizes sampling if the key itself has a batch dimension. 
*   **Consequence**: All $B$ environments use the **same random seed for action sampling**. If environments are in similar initial states (which they are), they will take identical actions.
*   **Finding**: Increasing $num\_envs$ does not increase exploration diversity; it just creates redundant copies of the same trajectory, wasting compute and slowing down the "effective" learning rate.

### 7.4 Circular Buffer Alignment Bug (Corrupted Sequences)
`ReplayBuffer.sample` (Line 661) uses a block-sampling strategy:
```python
starts = block_indices * self.sequence_length
indices = (starts[:, None] + seq_range[None, :]) % self.capacity
```
*   **The Issue**: This logic assumes the buffer is a perfect chain of consecutive `sequence_length` blocks. 
*   **The Math**: $100,000 \, \text{capacity} \div 128 \, \text{seq\_len} = 781.25$.
*   **Consequence**: The ".25" remainder means every time the buffer wraps around, the start of every trajectory in the buffer "shifts" by 32 slots.
*   **Finding**: After the first wrap-around, `sample()` starts returning "Frankenstein" sequences—combinations of different environments or non-sequential time-steps. High parallelism (64 envs) reaches this corruption state 64x faster than a single environment.

---

### Phase 8: Strategic Fixes (Next Steps)
1.  **Stochastic Collection**: Split `act_key` into $B$ keys in `collect_sequence` to ensure independent exploration.
2.  **Adaptive Replay Ratio**: Scale `train_steps` or decrease `batch_size` based on `num_envs` to maintain constant sample intensity.
3.  **Buffer Hygiene**: Enforce `capacity % (num_envs * sequence_length) == 0` during buffer initialization.
