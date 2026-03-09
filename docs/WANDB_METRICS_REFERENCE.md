# WandB Metrics Reference

> **Status**: COMPLETED
> **Opened**: 2026-03-09
> **Related**: [train.py](../train.py), [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## Context

This document provides a comprehensive reference of all metrics logged to Weights & Biases (WandB) during training in `train.py`. It serves as context for LLM agents working on training analysis, metric additions, or dashboard configuration.

## Analysis

All metrics are extracted from `train.py`. The logging structure varies by algorithm, with some metrics shared across all algorithms and others algorithm-specific. WandB step metrics are configured at initialization (lines 385–390) to control which x-axis each metric group uses.

### Step Metric Definitions

Defined at WandB init (`train.py:385–390`):

| Pattern | X-Axis |
|---------|--------|
| `Episode/*` | `Episode/Number` |
| `loss/*` | `iteration` |
| `*` (everything else) | `timesteps` |

---

## Metrics by Algorithm

### Shared Metrics (All Algorithms)

Logged whenever episodes complete during an iteration.

| Metric Key | Type | Description |
|------------|------|-------------|
| `Episode/Reward` | float | Mean reward of episodes completed in the current iteration |
| `Episode/Reward_Min` | float | Minimum episode reward in the iteration |
| `Episode/Reward_Max` | float | Maximum episode reward in the iteration |
| `Episode/Steps` | float | Mean episode length in the iteration |
| `Episode/Number` | int | Cumulative total episodes completed |
| `Episode/FoodEaten` | float | Mean food eaten per episode |
| `Episode/PredatorHits` | float | Mean predator hits per episode |
| `Episode/DangerHits` | float | Mean danger hits per episode |
| `Episode/RestCount` | float | Mean rest actions per episode |
| `Episode/Collisions` | float | Mean collision count per episode |
| `Episode/TotalDamage` | float | Mean total damage taken per episode |
| `Episode/DamagePredator` | float | Mean damage from predators per episode |
| `Episode/DamageDanger` | float | Mean damage from danger zones per episode |
| `Episode/DamageObstacle` | float | Mean damage from obstacle collisions per episode |
| `Episode/MeanDistFood` | float | Average distance to food per step across entire episode |
| `Episode/MeanDistPredator`| float | Average distance to nearest predator per step |
| `Episode/Term_Starvation` | float | Fraction of episodes ending in starvation (energy < 0.0) |
| `Episode/Term_Injury` | float | Fraction of episodes ending in injury (health < 0.0) |
| `Episode/Term_Overeating` | float | Fraction of episodes ending in overeating (stomach > stomach_capacity) |
| `Episode/Term_MaxSteps` | float | Fraction of episodes reaching maximum episode length |
| `timesteps` | int | Global environment step counter |
| `iteration` | int | Training iteration counter |

> **Note**: All behavioral metrics are now shared across all algorithms (RecurrentPPO, DreamerV3, DQN, DRQN).

### Evaluation Metrics (All Algorithms, at Checkpoints)

Logged when `training.stats_during_training` is enabled, triggered at checkpoint intervals (`train.py:1465–1466`).

| Metric Key | Type | Description |
|------------|------|-------------|
| `Eval/MeanReward` | float | Mean reward over evaluation episodes |
| `Eval/MeanLength` | float | Mean episode length over evaluation episodes |

---

### RecurrentPPO

**Loss metrics** — logged every iteration (`train.py:866–900`):

| Metric Key | Type | Description |
|------------|------|-------------|
| `loss/total` | float | Combined PPO loss (policy + value + entropy) |
| `loss/policy` | float | Policy (actor) surrogate loss |
| `loss/value` | float | Value function MSE loss |
| `loss/entropy` | float | Entropy bonus (negative = encouraging exploration) |
| `loss/grad_norm` | float | Global gradient norm across all parameters |

**Modulator metrics** — logged only when `agent.modulation` is configured and non-null (`train.py:876–894`):

| Metric Key | Type | Description |
|------------|------|-------------|
| `modulator/grad_norm` | float | Gradient norm of modulator parameters specifically |
| `modulator/gamma_uni_mean` | float | Mean of unimodal multiplicative gain (z_unimodal) |
| `modulator/gamma_uni_std` | float | Std of unimodal multiplicative gain |
| `modulator/gamma_multi_mean` | float | Mean of multimodal multiplicative gain (z_multimodal) |
| `modulator/gamma_multi_std` | float | Std of multimodal multiplicative gain |
| `modulator/z_memory_mean` | float | Mean of memory gate signal (z_memory) |
| `modulator/z_memory_std` | float | Std of memory gate signal |
| `modulator/temperature_mean` | float | Mean policy temperature |
| `modulator/temperature_min` | float | Min policy temperature across batch |
| `modulator/temperature_max` | float | Max policy temperature across batch |

**PreActivation-only modulator metrics** — logged only when `modulation.type == "PreActivation"` (`train.py:888–894`):

| Metric Key | Type | Description |
|------------|------|-------------|
| `modulator/beta_uni_mean` | float | Mean of unimodal additive bias (z_unimodal_add) |
| `modulator/beta_uni_std` | float | Std of unimodal additive bias |
| `modulator/beta_multi_mean` | float | Mean of multimodal additive bias (z_multimodal_add) |
| `modulator/beta_multi_std` | float | Std of multimodal additive bias |

---

### DreamerV3

**Episode metrics** — logged every iteration when episodes complete (`train.py:1036–1047`). Uses the shared Episode metrics listed above.

**Training/World Model metrics** — logged every 10 iterations (`train.py:1077–1103`). The specific keys depend on what the `DreamerTrainer` returns in its `metrics` dict. They are routed by prefix:

| Prefix Pattern | WandB Panel | Example Keys |
|----------------|-------------|--------------|
| `loss_actor*`, `loss_critic*`, `mean_*`, `entropy*` | `Behavior/` | `Behavior/loss_actor`, `Behavior/mean_entropy` |
| `loss_model*`, `loss_recon*`, `loss_kl*`, `loss_rew*`, `loss_cont*`, `loss_dyn*`, `loss_rep*`, `model_*` | `WorldModel/` | `WorldModel/loss_model`, `WorldModel/model_reward_mae` |
| `mod_*` | `Modulator/` | `Modulator/mod_z_reward_mean` |
| Other keys | Root namespace | — |

**Buffer/Ratio metrics** — logged every 10 iterations (`train.py:1078–1089`):

| Metric Key | Type | Description |
|------------|------|-------------|
| `Params/effective_replay_ratio` | float | Cumulative gradient steps / global env steps |
| `Params/positive_buffer_blocks` | int | Sequence blocks stored in positive-reward buffer (mixture mode only) |
| `Params/positive_buffer_utilization` | float | Positive buffer fill ratio (mixture mode only) |
| `Params/main_buffer_blocks` | int | Sequence blocks stored in main replay buffer |

---

### DQN

Logged every iteration (`train.py:1192–1205`):

| Metric Key | Type | Description |
|------------|------|-------------|
| `loss/dqn` | float | TD loss |
| `train/epsilon` | float | Current epsilon-greedy exploration rate |

---

### DRQN

Logged every iteration (`train.py:1308–1321`):

| Metric Key | Type | Description |
|------------|------|-------------|
| `loss/drqn` | float | TD loss |
| `train/epsilon` | float | Current epsilon-greedy exploration rate |

---

### PPO (non-recurrent)

Logged every iteration (`train.py:1361–1376`):

| Metric Key | Type | Description |
|------------|------|-------------|
| `loss/ppo_total` | float | Combined PPO loss |

> **Note**: Unlike RecurrentPPO, vanilla PPO does **not** log per-component loss breakdown (policy/value/entropy) to WandB. Only the aggregate total is logged.

---

## WandB Configuration

Controlled by CLI flags and config YAML (`train.py:354–392`):

| Setting | CLI Flag | Config Key | Default |
|---------|----------|------------|---------|
| Disable WandB | `--no-wandb` | `wandb.disabled` | `false` |
| Project | `--wandb-project` | `wandb.project` | (mandatory) |
| Entity | `--wandb-entity` | `wandb.entity` | (mandatory) |
| Group | `--wandb-group` | `wandb.group` | (mandatory) |
| Run name | `--wandb-name` | — | Falls back to `tag` |
| Resume ID | `--wandb-resume-id` | — | — |

Code is also logged via `wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))` (line 392).
