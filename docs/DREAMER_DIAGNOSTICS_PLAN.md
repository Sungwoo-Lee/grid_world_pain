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

---

## 7. Parallel Environment Audit (Feb 24 - Current Investigation)

Significant performance gains were observed when reducing the number of parallel environments (e.g., from 64 to 1). Audit identifies four systemic reasons:

### 7.1 Replay Ratio Inflation
In the current implementation, `train_steps` is fixed while collection volume scales with `num_envs` ($B$).
- **64 Envs**: 8192 steps collected per 1 update (Replay Ratio $\approx 1.0$).
- **1 Env**: 128 steps collected per 1 update (Replay Ratio $\approx 64.0$).
The 1-env agent undergoes **64x more training intensity per environment step**, leading to faster survival time improvement.

### 7.2 Dataset Persistence (Memory Depth)
Fixed buffer capacity ($10^5$) means high parallelism flushes memory 64x faster.
- **64 Envs**: Buffer is overwritten every ~12 iterations (overfits to immediate past).
- **1 Env**: Buffer persists for ~781 iterations (trains on diverse historical transitions).

### 7.3 Stochasticity Bug in Collection
`DreamerTrainer.collect_sequence` passed a single PRNG key to all parallel environments. This caused environments in similar states to sample **identical actions**, collapsing the effective exploration diversity of the parallel batch.

### 7.4 Buffer Alignment Bug
Replay buffer `capacity` (100,000) is not a multiple of `sequence_length` (128). Upon wrap-around, the indexing shifts, causing the temporal sampler to retrieve non-sequential "jumbled" trajectories. This corruption occurs 64x faster in highly parallel runs.

**Status**: Root cause identified. Implementation of fixes (scaling `train_steps`, fixing key splitting, and aligning buffer capacity) is required to restore parallel efficiency.
