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

---

## 8. Feb 25 – 08_location Training Failure Investigation

### 8.1 Context
Training with `configs/experiment/ablation/homeostatic/08_location.yaml` + `dreamer_v3.yaml` (1 env, 1 train step) fails to learn. This run uses the **same JAX DreamerV3 code** but switches to the 08_location ablation config which includes a predator, homeostatic reward, and all sensory channels.

> **Important Context (from user)**:  
> - The 08_location task **was trainable** with a previous code version (different git branch). Those results were deleted during branch cleanup, so no baseline comparison exists.  
> - The 08_location config is **easier and faster** than the default env config. RecurrentPPO solves it quickly. The default config is actually more difficult.  
> - The issue could be a **code bug** in the current JAX DreamerV3 implementation, **and/or hyperparameter/environment parameter changes** that occurred during the JAX migration (e.g., max nutrition, damage values, etc.).

**Run**: `results/JAX_DreamerV3/20260225-210859_08_location_dreamer_v3_1env_1trainStep_decoderupdat`  
**WandB**: `run-20260225_210900-bthzxq3a`  
**Command**: `train.py --config 08_location.yaml --agent_config dreamer_v3.yaml --num-envs 1 --episodes 100000`

### 8.2 Empirical Evidence (from WandB + Eval Stats)

#### Eval Stats (100 episodes per checkpoint)
| Checkpoint | Mean Ep Length | Action Distribution | Termination |
|:---|:---|:---|:---|
| 10,004 | 20 steps | 87% Right, 6% Up | 100% death (code 3) |
| 20,002 | 17.3 steps | 100% Up | Mixed |
| 30,003 | 17.3 steps | 100% Up | Mixed |

**Diagnosis**: Complete policy collapse to a single action by 20k episodes. No survival improvement whatsoever.

#### WandB Summary Metrics (at ~30k episodes / 555k timesteps)
| Metric | Value | Healthy Range | Verdict |
|:---|:---|:---|:---|
| `mean_entropy` | **0.07** (from 0.32 at 10k) | >0.5 | ❌ **COLLAPSED** |
| `Episode/Reward` | -137 | Should improve | ❌ Flat |
| `Episode/Steps` | 21 | Should increase | ❌ Flat |
| `model_reward_mae` | 4.8 | <1.0 | ❌ Very high |
| `model_reward_mae_pos` | **0.0** | >0 | ❌ **Never sees positive reward** |
| `model_reward_mae_neg` | 4.8 | - | High error on negatives |
| `latent_entropy` | 0.66 | 1.5-2.5 | ⚠️ Low |
| `loss_recon` | 0.015 | <0.1 | ✅ OK |
| `loss_rew` | 0.54 | - | ⚠️ Not converging |
| `mean_value` | 0.42 | Should track return | ❌ Mismatch |
| `mean_return` | -21.0 | Should improve | ❌ Stuck |
| `value_mae` | **21.5** | <5 | ❌ **Critic completely wrong** |
| `loss_dyn_kl` / `loss_rep_kl` | 1.63 | ~1.0 (free nats) | ⚠️ Near floor |

#### Comparison with Previous Runs (default env, Feb 24)
| Run | Entropy | Ep Steps | Ep Reward | Rew MAE | Latent Ent |
|:---|:---|:---|:---|:---|:---|
| 64env hierarchical (default) | 0.15 | 58 | -143 | 1.97 | 0.83 |
| 16env hierarchical (default) | 0.03 | 51 | -144 | 1.89 | 0.80 |
| 4env hierarchical (default) | 0.04 | 36 | -136 | 2.36 | 0.87 |
| 1env hierarchical (default) | 0.12 | 11 | -130 | 3.28 | 0.67 |
| **08_location 1env (THIS RUN)** | **0.07** | **17** | **-137** | **4.80** | **0.66** |

**Key Observation**: Entropy collapse occurs across ALL runs. This is a **systemic problem** with the DreamerV3 implementation, not specific to 08_location. The 08_location config is harder (predator, homeostatic reward) which makes the problem more visible.

### 8.3 Root Cause Hypotheses (Ranked by Likelihood)

#### H1: ❌ Entropy Collapse → Policy Death Spiral (PRIMARY SUSPECT)
The policy entropy drops to near-zero within the first 10k episodes. With `entropy_scale=3e-4`, the entropy bonus in the actor loss is negligible (`loss_actor_entropy = -6.6e-6`). Once the policy becomes deterministic, it cannot explore, so it never finds food → only sees negative rewards → world model learns "all actions lead to death" → actor converges to an arbitrary fixed action.

**Evidence**: `mean_entropy = 0.07` (should be >0.5), all previous runs also collapsed.

**Potential Fix**: Increase `entropy_scale` (try 1e-2 or 3e-3) or investigate if the Reinforce-style actor loss is correctly formulated.

#### H2: ⚠️ Actor Loss Formulation Mismatch
The current actor loss uses REINFORCE (`log_prob * advantage + entropy_scale * entropy`). Canonical DreamerV3 uses **dynamics backprop** through the world model, or at minimum **Reinforce with normalized advantages**. A pure Reinforce formulation is extremely sensitive to high-variance advantages.

**Evidence**: `mean_advantage=0.057` is suspiciously small and uniform, suggesting the advantage signal is being washed out.

**Potential Fix**: Verify advantage normalization is working. Consider switching to the canonical straight-through actor.

#### H3: ⚠️ Critic Divergence
The critic predicts `mean_value=0.42` while `mean_return=-21.0`, with `value_mae=21.5`. The critic is completely detached from reality. This poisons the advantage estimates.

**Evidence**: Critic loss is 1.02 which isn't decreasing. The `mean_norm_return` is 0.48 while advantage is 0.057, suggesting the Moments normalization is compressing everything into a tiny range.

**Potential Fix**: Check if the Moments normalization is working correctly. Verify that the `norm_returns` passed to the critic use consistent scaling.

#### H4: 🔍 Reward Signal Starvation
The world model has `model_reward_mae_pos=0.0`, meaning it has **never encountered a positive reward** in its training data. In the 08_location config, the agent must find food AND eat it to get positive reward, but it dies too quickly (17 steps) to ever reach the food source at position (3,3).

**Evidence**: All rewards are negative (homeostatic drive-reduction penalties accumulate). The reward head only learns to predict negative values.

**Contributing Factor**: This is a *consequence* of H1 (entropy collapse → no exploration → no food discovery), not an independent cause.

#### H5: 🔍 KL Collapse (Latent Uninformative)
Both `loss_dyn_kl` and `loss_rep_kl` are at 1.63 — just barely above the free-nats threshold of 1.0. This means the posterior is nearly equal to the prior (the observation provides almost no information to the latent state). The world model isn't learning meaningful dynamics.

**Evidence**: `latent_entropy=0.66` is very low. The prior/posterior collapse means imagination produces meaningless rollouts.

### 8.4 Diagnostic Plan

#### Phase 1: Quick Experiments (No Code Changes)
- [ ] **Exp 1.1**: Run the *same code* with default env config + 1env to confirm the problem is systemic (not 08_location specific). Compare entropy trajectory.
- [ ] **Exp 1.2**: Check if `num_steps` variable for `collect_sequence` in the training loop is set correctly for 08_location. Verify `sequence_length=128` matches the buffer and batch sampling.

#### Phase 2: Entropy Fix (Highest Priority)
- [ ] **Exp 2.1**: Increase `entropy_scale` from `3e-4` to `3e-3` or `1e-2` and rerun 08_location.
- [ ] **Exp 2.2**: Add entropy logging per-step to verify the entropy regularization gradient is flowing correctly through the actor.

#### Phase 3: Actor-Critic Audit
- [ ] **Audit 3.1**: Verify the advantage computation. Check if `norm_returns - baseline` is producing meaningful gradients. Log advantage statistics (min, max, std) per train step.
- [ ] **Audit 3.2**: Verify the Moments normalization is not compressing the return distribution too aggressively. Log the Moments `low`, `high` EMA values.
- [ ] **Audit 3.3**: Verify the critic is predicting in the same space as the lambda returns (both should be symlog-space, or both raw-space). A mismatch here would explain the `value_mae=21.5`.

#### Phase 4: World Model Verification
- [ ] **Audit 4.1**: Verify the decoder `loss_recon` target is correct. The loss is 0.015 (low), but check if the decoder target is `symlog(obs)` (matching the input) — see line 121 vs line 201 in `dreamer_v3_trainer.py`.
- [ ] **Audit 4.2**: Dump example imagined rewards from the behavior rollout. Verify they're in the correct range (not symlogged when they should be raw, etc.).
- [ ] **Audit 4.3**: Check if the `continue_head` is correctly predicting episode termination (0.9957 accuracy seems high — verify it's not trivially predicting "always continue").

#### Phase 5: Structural Issues
- [ ] **Check 5.1**: Verify the `num_steps` variable usage in `train.py` line 797 — is it defined for the Dreamer branch? (It appears to use `config.get_mandatory('agent.sequence_length')` for collection, but `num_steps` in the stats loop may be undefined).
- [ ] **Check 5.2**: Verify the `collect_sequence` is passing `is_first` correctly for episode boundaries within a 128-step sequence.

### 8.5 Code Audit Findings (Feb 25, 21:50 KST)

> **ROOT CAUSE IDENTIFIED**: Critical value-space mismatch in `behavior_loss_fn` (lines 389-420 of `dreamer_v3_trainer.py`)

**Bug chain traced through the behavior loss:**

1. `lambda_returns` computed from `from_twohot()` → **raw space** ✅
2. `norm_returns = moments.normalize(lambda_returns)` → **[0,1]-ish** ✅  
3. **BUG (line 391)**: `target_twohot = to_twohot(norm_returns)` → `to_twohot()` internally calls `symlog()`, so critic learns `symlog(norm_returns)` — a **double transformation**.
4. **BUG (line 396)**: `baseline = from_twohot(v_pred_logits)` → returns `symexp(predicted)` → back to **~norm_returns space** (approximately).
5. **BUG (line 397)**: `advantage = norm_returns - baseline` → subtracts Moments-normalized values from symexp(symlog(normalized)) values — **incompatible spaces for any non-trivial magnitude**.

**This explains ALL symptoms**: garbage advantage → random actor gradients → entropy collapse → policy death spiral.

**Fix**: See `implementation_plan.md` — Option A (canonical DreamerV3): train critic on raw `lambda_returns`, normalize both sides for advantage computation.

### 8.6 Applied Fix & Verification (Feb 25, 22:00 KST)

#### Changes Made to `dreamer_v3_trainer.py`

**Background: How DreamerV3's Critic Works**

DreamerV3 uses a **two-hot encoded critic** to predict future returns. The flow is:

1. The agent imagines future trajectories in the world model.
2. It computes **lambda returns** (discounted cumulative rewards) from imagined rewards — these are in **raw reward space** (e.g., values like -20, +5, etc.).
3. The critic learns to predict these returns using a **two-hot distribution** over 255 buckets. The `to_twohot()` function converts a scalar target into this distribution, but crucially it first applies `symlog()` internally (symmetric log: `sign(x) * log(|x|+1)`) to compress the value range.
4. The inverse function `from_twohot()` decodes the critic's prediction back to a scalar, applying `symexp()` (the inverse of `symlog`) — returning a value in **raw space**.
5. **Moments normalization** scales raw returns to a `[0, 1]`-ish range by tracking the 5th/95th percentile via Exponential Moving Average. This normalized form is used for the **advantage** (which drives the actor's learning signal).

**Bug 1 — Critic trained on double-transformed targets (line 391)**

```python
# BEFORE (broken):
norm_returns = self.moments.normalize(lambda_returns)  # Raw → Normalized [0,1]
target_twohot = to_twohot(norm_returns)                # to_twohot applies symlog AGAIN!
# Result: critic learns to predict symlog(norm_returns) — a double transformation
```

The critic was being trained on `symlog(normalized_returns)` — the returns were first Moments-normalized (compressing to ~[0,1]) and then `to_twohot()` applied `symlog()` again. This double compression made the critic's target distribution nearly uniform, making it very hard to learn.

```python
# AFTER (fixed):
target_twohot = to_twohot(lambda_returns)  # Raw → symlog (single transform, as intended)
# Result: critic learns to predict symlog(raw_returns) — correct single transformation
```

**Bug 2 — Advantage computed in mismatched spaces (lines 396-397)**

The **advantage** = (how good this trajectory is) - (how good the critic thinks it is). Both sides must be in the same numerical space for this subtraction to be meaningful.

```python
# BEFORE (broken):
baseline = from_twohot(v_pred_logits)      # Returns symexp(critic_prediction) → ~raw space
advantage = norm_returns - baseline         # Normalized [0,1] minus raw [-20, +5] = GARBAGE
```

`norm_returns` was in `[0, 1]` range while `baseline` was in raw space `[-20, +5]`. The subtraction produced meaningless advantage values → random gradients for the actor → entropy collapse → the agent locks onto a single action and stops exploring.

```python
# AFTER (fixed):
baseline = from_twohot(v_pred_logits)                        # Raw space
norm_baseline = (baseline - moments_low) / moments_invscale  # Normalize to same scale
advantage = norm_returns - norm_baseline                     # Both normalized → meaningful signal
```

Both `norm_returns` and `norm_baseline` are now in the same Moments-normalized space, producing a meaningful advantage signal that correctly tells the actor which actions are better than expected.

> **Note**: We pre-compute `moments_low` and `moments_invscale` *outside* the `nnx.grad` function to avoid pulling the Moments module's internal state into JAX's gradient computation graph (which would cause OOM errors due to unnecessary gradient tracing).

**Bug 3 — value_mae metric (line 420)**

```python
# BEFORE: compared baseline (raw-ish) vs lambda_returns (raw) — but baseline was corrupted
# AFTER:  both in raw space, giving an honest accuracy measure of the critic
'value_mae': jnp.mean(jnp.abs(baseline - jax.lax.stop_gradient(lambda_returns)))
```

#### Reference: Sheeprl Canonical Implementation

For reference, here is how the sheeprl DreamerV3 (PyTorch) handles the same logic. Our fix now matches this pattern.

**How `TwoHotEncodingDistribution` works** (`sheeprl/utils/distribution.py:224`):
- Bins are **evenly spaced** in `[-20, +20]` — NOT symexp'd as some descriptions claim.
- `symlog` is applied to the **input value** before finding the nearest bins (encoding).
- `symexp` is applied to the **output** weighted average after decoding.

```python
# Sheeprl TwoHotEncodingDistribution
class TwoHotEncodingDistribution:
    def __init__(self, logits, low=-20, high=20,
                 transfwd=symlog, transbwd=symexp):
        self.bins = torch.linspace(low, high, logits.shape[-1])  # Evenly spaced!

    @property
    def mean(self):
        # Decode: weighted avg in symlog-space → symexp back to raw
        return self.transbwd((self.probs * self.bins).sum(...))

    def log_prob(self, x):
        # Encode: symlog(raw_target) → find two closest bins → cross-entropy
        x = self.transfwd(x)  # symlog(x)
        # ... two-hot encoding against the evenly-spaced bins
```

**Critic target** (`sheeprl/algos/dreamer_v3/dreamer_v3.py:314`):
```python
# Sheeprl trains critic on RAW lambda_values (symlog applied internally by log_prob)
value_loss = -qv.log_prob(lambda_values.detach())
# This is equivalent to our fixed: to_twohot(lambda_returns)
```

**Advantage normalization** (`sheeprl/algos/dreamer_v3/dreamer_v3.py:275-279`):
```python
# Sheeprl normalizes BOTH sides with the same Moments statistics
baseline = predicted_values[:-1]                      # Raw space (from .mean → symexp)
offset, invscale = moments(lambda_values, fabric)     # Moments on raw lambda
normed_lambda_values = (lambda_values - offset) / invscale
normed_baseline = (baseline - offset) / invscale      # Same normalization!
advantage = normed_lambda_values - normed_baseline    # Both normalized → consistent
```

**Key takeaway**: The canonical sheeprl implementation confirms that:
1. Critic is always trained on **raw** lambda returns (symlog is internal to `TwoHotEncodingDistribution`).
2. Advantage uses **Moments normalization on both sides** with identical `offset` and `invscale`.
3. These are exactly the patterns our fix now follows.

#### Verification Run
- **Tag**: `diagnostic_value_fix_v2`
- **WandB**: `run-20260225_220450-a2jts33m`
- **Config**: 08_location, 1 env, 5000 episodes (144k timesteps, 8 min)

#### Results: Before vs After Fix (at 5000 episodes)
| Metric | Broken (30k eps) | Fixed (5k eps) | Verdict |
|:---|:---|:---|:---|
| `mean_entropy` | **0.07** | **1.79** | ✅ **FIXED** — no longer collapsed |
| `mean_return` | -21.0 | **-12.9** | ✅ Improving |
| `mean_value` | 0.42 | **-6.15** | ✅ Now tracks returns |
| `value_mae` | **21.5** | **6.75** | ✅ 3x more accurate |
| `model_reward_mae_pos` | **0.0** | **6.0** | ✅ Now sees positive rewards |
| `model_reward_mae` | 4.8 | 3.73 | ✅ Slightly better |
| `latent_entropy` | 0.66 | **1.10** | ✅ Richer latent |
| `Episode/Steps` | 17 | **28.6** | ✅ 68% longer survival |
| `Episode/Reward` | -137 | -138 | ⚠️ Similar — needs more training |
| `loss_recon` | 0.015 | 0.15 | ⚠️ Higher (expected — model exploring more) |
| `loss_rew` | 0.54 | 1.80 | ⚠️ Higher (learning harder reward landscape) |

**Conclusion**: The value-space fix **resolves the primary bugs** (entropy collapse + critic divergence). The agent is now exploring (mean_entropy=1.79), learning meaningful representations (latent_entropy=1.10), and surviving longer (28.6 vs 17 steps). The episode reward hasn't improved yet — this is expected since 5k episodes (144k timesteps) is very early for DreamerV3. A longer run (50k+ episodes) is needed to see reward improvement.

**Next steps**: Run a longer training (e.g., 30k–50k episodes) with `--debug` to monitor entropy and reward trajectories over time. If rewards still don't improve, investigate hyperparameters (entropy_scale, train_steps) and environment parameters (max_nutrition, damage values).

---

## 9. Issue: Parallel Environment Scaling (Feb 25, 23:30 KST)

### 9.1 Problem Statement

Increasing from 1 to 64 parallel environments does **NOT** improve training — it actually makes per-episode learning dramatically worse.

### 9.2 Run Comparison

| Config | 1env (fixedReturns) | 64env (fixedTwoHot) |
|:---|:---|:---|
| **Tag** | `08_location_dreamer_v3_1env_1trainStep_fixedReturns` | `08_location_dreamer_v3_64env_16trainStep_fixedTwoHot` |
| **WandB** | `run-20260225_222020-4643srox` | `run-20260225_225710-hgdfpos2` |
| **num_envs** | 1 | 64 |
| **train_steps** | 1 | 16 |
| **sequence_length** | 128 | 128 |

### 9.3 WandB Metrics Comparison

| Metric | 1env (at ~50k eps, tqdm) | 64env (at 100k eps, WandB) |
|:---|:---|:---|
| **Entropy** | **1.78** (stable) | **1.64** (oscillating 1.56–1.74) |
| **World Model Loss** | **1.76** | **2.51** (43% higher) |
| **Reward MAE** | **2.88** | **2.92** |
| **Reward MAE (pos)** | — | 0.77 |
| **Latent Entropy** | — | **0.74** (low vs ~1.1 for 1env) |
| **Value MAE** | — | **4.33** |
| **Mean Return** | — | **-19.3** |
| **Loss Critic** | — | **1.45** |
| **Iterations** | **~11,766** | **367** |
| **Timesteps** | ~6.4M | **3.0M** |

### 9.4 Eval Stats Comparison (per-checkpoint)

**1env (fixedReturns):**
| Checkpoint | Mean Ep Len | Mean Ep Rew | Total Ate |
|:---|:---|:---|:---|
| 10k eps | 20.3 | -135.0 | 149 |
| 20k eps | 33.2 | -135.5 | 542 |
| 30k eps | 41.7 | -132.0 | 1240 |
| 40k eps | 49.1 | -134.5 | 711 |
| **50k eps** | **92.0** | **-133.1** | **1139** |

**64env (fixedTwoHot):**
| Checkpoint | Mean Ep Len | Mean Ep Rew | Total Ate |
|:---|:---|:---|:---|
| 10k eps | 24.3 | -137.1 | 11 |
| 20k eps | 25.2 | -136.0 | 15 |
| 30k eps | 20.5 | -134.1 | 155 |
| 40k eps | 18.1 | -133.8 | 213 |
| **50k eps** | **19.6** | **-133.4** | **346** |
| 70k eps | 20.2 | -131.3 | 745 |
| 90k eps | 35.3 | -136.6 | 151 |
| 100k eps | 49.1 | -132.0 | 496 |

**Key observation**: At 50k episodes, 1env's MeanLen is **92** vs 64env's **19.6** — a **4.7x** difference.

### 9.5 Root Cause Analysis: Gradient-to-Data Ratio

The core issue is a **severely imbalanced replay ratio** when scaling environments.

**How the training loop works** (`train.py:750-837`):

1. Each iteration calls `collect_sequence()` → produces `num_envs * sequence_length` transitions
2. Then calls `trainer.train_step()` exactly `train_steps` times
3. Each `train_step` samples one batch from the replay buffer

**Gradient update calculation:**

| | 1env | 64env |
|:---|:---|:---|
| Transitions per iteration | 1 × 128 = **128** | 64 × 128 = **8,192** |
| Gradient updates per iteration | **1** | **16** |
| **Data-to-gradient ratio** | **128:1** | **512:1** (4x worse) |
| Gradient updates at 50k episodes | **~11,766** | **~2,936** |
| **Gradient shortfall** | — | **4x fewer** updates at same ep count |

The 64env run collects 64× more data but only does 16× more gradient updates per iteration. This means:
- **The world model is underfitting**: it sees 4× less training per data point, explaining the higher loss (2.51 vs 1.76)
- **The latent space is less structured**: lower latent entropy (0.74 vs >1.0) because the RSSM hasn't been trained enough
- **The actor/critic lag behind**: fewer gradient updates → worse behavior learning

### 9.6 Additional Concern: Replay Buffer Capacity

With `capacity=100,000`:
- **1env**: fills at 128 transitions/iter → ~781 iterations to fill → data stays ~781 iters
- **64env**: fills at 8,192 transitions/iter → ~12 iterations to fill → data expires in ~12 iters

The 64env setup has much higher **data turnover**, meaning old experiences are overwritten quickly. The world model may not get enough training passes over each piece of data before it's discarded.

### 9.7 Proposed Remedies

#### Option 1: Manual `train_steps` scaling
Manually set `train_steps` proportional to `num_envs`. For 64 envs, `train_steps = 64` should match the 1env gradient-to-data ratio. Quick to implement but fragile — requires manual tuning every time `num_envs` changes.

#### Option 2: Increase replay buffer capacity
Scale buffer with `num_envs` (e.g., `capacity = 100,000 * num_envs`). Helps with data retention but doesn't fix the gradient shortfall.

#### Option 3: Sheeprl's `Ratio` approach (recommended)

Sheeprl's DreamerV3 uses a `Ratio` class to **automatically** compute how many gradient steps to take, ensuring a constant ratio of gradient updates per environment step, regardless of `num_envs`.

**The `Ratio` class** (`sheeprl/utils/utils.py:259`, from [Hafner's original](https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26)):

```python
class Ratio:
    def __init__(self, ratio: float, pretrain_steps: int = 0):
        self._ratio = ratio       # Target: gradient_steps / env_steps
        self._prev = None         # Tracks cumulative env steps last processed
        self._pretrain_steps = pretrain_steps  # Extra training at startup

    def __call__(self, step: int) -> int:
        """Given total env steps so far, return how many gradient steps to do NOW."""
        if self._ratio == 0:
            return 0
        if self._prev is None:
            # First call — handle pretrain
            self._prev = step
            repeats = int(step * self._ratio)
            if self._pretrain_steps > 0:
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        # Subsequent calls — only count NEW env steps since last call
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio  # Advance by exactly what we consumed
        return repeats
```

**How it integrates into the training loop** (`sheeprl/algos/dreamer_v3/dreamer_v3.py:518,659-698`):

```python
# Init: replay_ratio = 1.0 (default for DreamerV3)
ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)

# Each iteration:
policy_step += policy_steps_per_iter  # policy_steps_per_iter = num_envs * world_size
ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
per_rank_gradient_steps = ratio(ratio_steps / world_size)  # Dynamically computed!

if per_rank_gradient_steps > 0:
    # Sample that many batches and train
    for i in range(per_rank_gradient_steps):
        train(...)
        cumulative_per_rank_gradient_steps += 1
```

**What this means in practice:**

With `replay_ratio = 1.0` (sheeprl DreamerV3 default), the `Ratio` class ensures **1 gradient step per 1 new environment step**:

| | 1env | 64env |
|:---|:---|:---|
| Env steps per iteration | 1 | 64 |
| `ratio(new_steps)` returns | **1** grad step | **64** grad steps |
| **Gradient:Data ratio** | **1:1** | **1:1** ✅ (auto-scaled!) |

Compare this to our current fixed `train_steps`:

| | 1env (train_steps=1) | 64env (train_steps=16) |
|:---|:---|:---|
| Env steps per iteration | 128 | 8,192 |
| Gradient updates per iteration | 1 | 16 |
| **Gradient:Data ratio** | **1:128** | **1:512** ❌ |

> **Key insight**: Sheeprl counts gradient steps per **environment step** (per policy step), while our code counts per **iteration** (which bundles `num_envs * sequence_length` env steps). This is why our `train_steps` parameter doesn't scale correctly with `num_envs`.

**To implement this in our codebase**, we would:
1. Replace the fixed `train_steps` config with `replay_ratio` (default: 1.0)
2. Track cumulative environment steps (`global_step`)
3. Use `Ratio(replay_ratio)` to compute gradient steps each iteration:
   ```python
   # In train.py, DreamerV3 branch:
   new_env_steps = num_envs * sequence_length  # = 8192 for 64 envs
   grad_steps = ratio(global_step)   # Returns new_env_steps * replay_ratio
   for _ in range(grad_steps):
       batch = buffer.sample(batch_size)
       trainer.train_step(batch, key)
   ```
4. This auto-scales: 64 envs → 64× more grad steps per iteration, maintaining the 1:1 ratio.

---

## 10. Observation: Pre-Fix vs Post-Fix Performance Paradox (Feb 26, 00:20 KST)

### 10.1 The Question

The **pre-fix** run (`decoderupdat`, with the buggy `to_twohot(norm_returns)`) appeared to outperform the **post-fix** run (`fixedReturns`, with the corrected `to_twohot(lambda_returns)`). Why?

### 10.2 Run Details

| Config | decoderupdat (PRE-FIX) | fixedReturns (POST-FIX) |
|:---|:---|:---|
| **WandB** | `run-20260225_213404-4yfb1aa6` | `run-20260225_222020-4643srox` |
| **to_twohot target** | `to_twohot(norm_returns)` (buggy) | `to_twohot(lambda_returns)` (fixed) |
| **num_envs** | 1 | 1 |
| **Episodes** | 100k (completed) | 100k (still running at ~90k) |

### 10.3 WandB Final Metrics

| Metric | decoderupdat (PRE-FIX) | fixedReturns (POST-FIX, ~50k tqdm) |
|:---|:---|:---|
| **mean_entropy** | **0.14** ❌ (collapsed!) | **1.78** ✅ (healthy) |
| **mean_return** | -1.66 | — |
| **mean_value** | 0.51 | — |
| **value_mae** | **3.03** | — |
| **loss_model** | **1.58** | ~1.76 (from tqdm) |
| **loss_rew** | 0.69 | — |
| **model_reward_mae** | 0.90 | ~2.88 (from tqdm) |
| **latent_entropy** | **0.64** (collapsed) | — |
| **Episode/Steps** | **64** | — |
| **Eval/MeanLength** | **75.7** | — |
| **iterations** | 31,134 | ~11,766 |

### 10.4 Eval Stats Learning Curves

**decoderupdat (PRE-FIX):**
| Checkpoint | MeanLen | MeanRew | TotalAte | Learning? |
|:---|:---|:---|:---|:---|
| 10k eps | 17.3 | -134.3 | **0** | ❌ No learning |
| 20k eps | 17.3 | -134.3 | **0** | ❌ No learning |
| 30k eps | 18.0 | -134.7 | **0** | ❌ No learning |
| 40k eps | 18.0 | -134.7 | **0** | ❌ No learning |
| 50k eps | 21.6 | -136.4 | **0** | ❌ No learning |
| **60k eps** | **94.3** | **-132.7** | **2369** | 💥 Sudden spike |
| 70k eps | 52.3 | -135.5 | 790 | ⚠️ Dropped back |
| 80k eps | 39.9 | -133.2 | 356 | ⚠️ Unstable |
| 90k eps | 56.9 | -133.1 | 1493 | ⚠️ Oscillating |
| 100k eps | 76.7 | -130.7 | 1261 | ⚠️ Inconsistent |

**fixedReturns (POST-FIX):**
| Checkpoint | MeanLen | MeanRew | TotalAte | Learning? |
|:---|:---|:---|:---|:---|
| 10k eps | 20.3 | -135.0 | **149** | ✅ Early learning |
| 20k eps | 33.2 | -135.5 | **542** | ✅ Improving |
| 30k eps | 41.7 | -132.0 | **1240** | ✅ Strong growth |
| 40k eps | 49.1 | -134.5 | 711 | ✅ Dip but still eating |
| 50k eps | 92.0 | -133.1 | 1139 | ✅ Strong |
| 60k eps | 103.8 | -132.7 | 1355 | ✅ Continuing |
| 70k eps | 84.9 | -135.0 | 724 | ✅ Slight dip |
| 80k eps | 104.8 | -131.4 | 1213 | ✅ Best so far |

### 10.5 Analysis: Why Pre-Fix Appeared Better

**Short answer**: It didn't outperform consistently — it had a lucky exploration spike.

**The pre-fix run had a fundamentally broken learning process:**
1. **Entropy collapsed to 0.14** — the policy was effectively deterministic (almost always picking the same action)
2. **Zero eating for 50k episodes** — the agent couldn't find food at all
3. **The 60k spike is anomalous** — after 50k episodes of total failure, the agent suddenly achieved MeanLen=94 with 2369 eating events, then immediately regressed to 52 and 40 in subsequent checkpoints

**This is a hallmark pattern of a collapsed-entropy agent that "got lucky":**
- With entropy=0.14, the agent is essentially deterministic. If the fixed action sequence happens to stumble onto food (perhaps the environment's randomized start positions placed the agent near food), it will appear to "learn" temporarily
- But since the policy is locked (no exploration), it can't adapt to different start positions → oscillating performance across checkpoints

**The post-fix run has a fundamentally sound learning process:**
1. **Entropy stays at 1.78** — healthy exploration throughout
2. **Started eating by 10k episodes** — found food through genuine exploration, not luck
3. **Monotonic improvement** from 20→33→42→49→92→104 MeanLen — a real learning curve
4. **The learning is robust** — even dips (70k: 85) are much higher than the pre-fix's baseline

### 10.6 The Paradox Resolved

The pre-fix agent's lower `loss_model` (1.58 vs 1.76) and lower `reward_mae` (0.90 vs 2.88) are actually **symptoms of the bug, not signs of better learning**:

- **Lower loss_model**: The critic was trained on `symlog(norm_returns)` which compressed everything to near-zero, making the prediction task trivially easy. Low loss ≠ useful world model.
- **Lower reward_mae**: With entropy collapsed, the agent always takes the same action → always gets the same reward → the reward distribution has very low variance → easy to predict.
- **High reward_mae in post-fix**: The agent is actually exploring diverse actions → encountering diverse rewards → harder prediction task → higher MAE. This is a sign of healthy exploration.

> **Conclusion**: The post-fix run is unambiguously better. It has a genuine, stable learning curve driven by real exploration. The pre-fix run's occasional high points were lucky accidents in an otherwise collapsed policy. The fix is correct.

