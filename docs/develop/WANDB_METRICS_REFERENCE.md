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

## Episode Metrics: Aggregation Pipeline & Interpretation

All `Episode/*` metrics follow a two-stage aggregation pipeline. Understanding this pipeline is essential for interpreting values correctly.

### Stage 1: Per-Episode Accumulation

During each episode, per-step data from the environment's `info` dict is accumulated into per-environment buffers (`episode_behavior`, `episode_dist_sums`). When an episode ends (done=True), these accumulators produce a single `ep_data` dict for that episode:

| Metric Category | Per-Step Accumulation | Per-Episode Value | Example |
|---|---|---|---|
| **Event counts** (FoodEaten, PredatorHits, etc.) | `episode_behavior[k] += info[k]` each step | Raw sum over all steps in the episode | Agent ate food 3 times → `ate_food = 3` |
| **Damage metrics** (TotalDamage, DamagePredator, etc.) | `episode_behavior[k] += info[k]` each step | Raw sum of damage over the episode | 5.0 damage per predator hit × 15 hits → `damage_predator = 75.0` |
| **Distance metrics** (MeanDistFood, MeanDistPredator) | `episode_dist_sums[k] += info[k]` each step | Sum divided by episode length: `dist_sum / max(ep_length, 1)` | Sum of distances = 150.0 over 50 steps → `dist_to_food = 3.0` |
| **Termination reason** | Captured once at episode end | Integer code: 1=MaxSteps, 2=Starvation, 3=Overeating, 4=Injury | Agent died from injury → `termination_reason = 4` |
| **Reward** | Accumulated by the environment | Total episode return | — |
| **Steps** | Counted by the environment | Episode length (integer) | — |

After creating `ep_data`, the accumulators for that environment are reset to zero. The episode is appended to `iteration_episodes`.

### Stage 2: Iteration-Level Aggregation (what gets logged to WandB)

At the end of each training iteration, **all episodes that completed during that iteration** are aggregated with `np.mean()` into a single WandB log call. This means:

- **Each WandB data point represents the mean across N episodes**, where N is the number of episodes that happened to finish during that iteration.
- **N varies per iteration.** With parallel environments, multiple episodes can finish in the same iteration, or none at all (in which case no `Episode/*` data is logged for that iteration).
- **This is standard RL logging convention** — the same pattern used by Stable Baselines3, CleanRL, and other frameworks.

#### How to interpret each metric type

| Metric | WandB Value Represents | Why It Can Be Non-Integer |
|---|---|---|
| `Episode/Reward` | Mean total return across N episodes | Average of different episode rewards |
| `Episode/Steps` | Mean episode length across N episodes | e.g., episodes of length 42 and 58 → 50.0 |
| `Episode/FoodEaten` | Mean food eaten per episode across N episodes | e.g., 3 episodes ate [0, 1, 0] food → 0.33 |
| `Episode/PredatorHits` | Mean predator hits per episode | Same averaging as above |
| `Episode/TotalDamage` | Mean cumulative damage per episode | Already float from per-step damage values |
| `Episode/MeanDistFood` | Mean of per-episode mean distances | Double-averaged: per-step → per-episode → per-iteration |
| `Episode/Term_Injury` | Fraction of N episodes ending in injury | e.g., 8 of 10 episodes → 0.80 |

#### Practical example

If an iteration has 4 completed episodes with `FoodEaten = [0, 0, 1, 3]`:
- `Episode/FoodEaten` = `np.mean([0, 0, 1, 3])` = **1.0**
- Early in training when agents rarely eat, most episodes have 0 food → mean is a small float like **0.14**

#### WandB x-axis

`Episode/*` metrics use `Episode/Number` (cumulative episode count) as the x-axis, not `timesteps` or `iteration`. This is set via `wandb.define_metric("Episode/*", step_metric="Episode/Number")`.

---

## Metrics by Algorithm

### Shared Metrics (All Algorithms)

Logged whenever episodes complete during an iteration. See above for aggregation details.

| Metric Key | Type | Per-Episode Aggregation | Description |
|------------|------|------------------------|-------------|
| `Episode/Reward` | float | sum of rewards | Mean episode return across the iteration |
| `Episode/Reward_Min` | float | — | Minimum single-episode return in the iteration |
| `Episode/Reward_Max` | float | — | Maximum single-episode return in the iteration |
| `Episode/Steps` | float | step count | Mean episode length across the iteration |
| `Episode/Number` | int | — | Cumulative total episodes completed (x-axis) |
| `Episode/FoodEaten` | float | sum of `ate_food` events | Mean food eaten per episode |
| `Episode/PredatorHits` | float | sum of `hit_predator` events | Mean predator hits per episode |
| `Episode/DangerHits` | float | sum of `hit_hiding_predator` events | Mean danger zone hits per episode |
| `Episode/RestCount` | float | sum of `rested` events | Mean rest actions per episode |
| `Episode/Collisions` | float | sum of `event_collided` events | Mean collisions per episode |
| `Episode/TotalDamage` | float | sum of `damage` | Mean total damage taken per episode |
| `Episode/DamagePredator` | float | sum of `damage_predator` | Mean damage from predators per episode |
| `Episode/DamageDanger` | float | sum of `damage_hiding_predator` | Mean damage from danger zones per episode |
| `Episode/DamageObstacle` | float | sum of `damage_obstacle` | Mean damage from obstacles per episode |
| `Episode/MeanDistFood` | float | sum of `dist_to_food` / ep_length | Mean per-step distance to food, averaged across episodes |
| `Episode/MeanDistPredator`| float | sum of `dist_to_pred` / ep_length | Mean per-step distance to nearest predator, averaged across episodes |
| `Episode/Term_Starvation` | float | binary (1 if reason==2) | Fraction of episodes ending in starvation (energy < 0.0) |
| `Episode/Term_Injury` | float | binary (1 if reason==4) | Fraction of episodes ending in injury (health < 0.0) |
| `Episode/Term_Overeating` | float | binary (1 if reason==3) | Fraction of episodes ending in overeating (stomach > capacity) |
| `Episode/Term_MaxSteps` | float | binary (1 if reason==1) | Fraction of episodes reaching maximum episode length |
| `timesteps` | int | — | Global environment step counter |
| `iteration` | int | — | Training iteration counter |

> **Note**: All behavioral metrics are shared across all algorithms (RecurrentPPO, DreamerV3, DQN, DRQN, PPO). The 4 termination fractions sum to 1.0 within each iteration.

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
