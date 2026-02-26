# Recurrent PPO Diagnostics Plan

## 1. Gradient Update Mechanics

### 1.1 Overview

Recurrent PPO is an **on-policy** algorithm. Unlike DreamerV3's off-policy replay-based training, PPO collects a fixed rollout of experience, trains on it for a small number of epochs, then **discards** it and collects fresh data. There is no replay buffer.

### 1.2 Per-Iteration Flow

```
train_iteration() [recurrent_ppo_trainer.py]:
1. COLLECT:  collect_trajectories()
   - jax.lax.scan for num_steps=128 steps across num_envs environments
   - Produces rollout: (128, num_envs, ...) tensors of obs, action, reward, done, log_prob, value
   - Total data collected: num_steps × num_envs = 128 × num_envs transitions

2. COMPUTE ADVANTAGES:
   - GAE mode: compute_gae() with gamma=0.95, lambda=0.95
   - MC mode:  compute_mc_returns() with gamma=0.95
   - Normalize advantages

3. TRAIN (K_epochs = 4):
   for epoch in range(K_epochs):
       update_step():
         - vmap(ppo_loss_fn) over batch dimension
         - Single forward+backward pass over ENTIRE rollout
         - optimizer.update(model, grads)
   
   Total gradient updates per iteration: K_epochs = 4
   
4. DISCARD the rollout (no replay buffer)
```

### 1.3 Gradient Steps Per Iteration

| Parameter | Value | Role |
|:---|:---|:---|
| `num_steps` (`sequence_length`) | 128 | Steps collected per env per iteration |
| `num_envs` | CLI argument | Number of parallel environments |
| `K_epochs` | 4 | Number of gradient passes over the rollout |
| **grad_steps/iter** | **4** | Always 4, regardless of num_envs |

> **Key difference from DreamerV3**: In PPO, `num_envs` does NOT affect the number of gradient steps. It affects the **batch size** of each gradient step. More envs = larger batch = richer gradient signal per update, not more updates.

### 1.4 How `num_envs` Affects Training

```
Each gradient step processes: (num_steps, num_envs, ...) = (128, num_envs, ...)

1env:   4 gradient steps × batch of (128, 1)   = 4 updates, 128 samples each
64env:  4 gradient steps × batch of (128, 64)  = 4 updates, 8192 samples each
256env: 4 gradient steps × batch of (128, 256) = 4 updates, 32768 samples each
```

More environments give:
- **Better gradient estimates** (larger batch → lower variance)
- **More diverse experience** per iteration (64 different trajectories vs 1)
- **More episodes completed** per iteration (faster episode throughput)
- **Same number of gradient updates** (always K_epochs = 4)

### 1.5 Effective Training Intensity

| Metric | PPO | DreamerV3 (replay_ratio=1) |
|:---|:---|:---|
| **Data reuse** | K_epochs passes (4), then discard | Stored in buffer, sampled many times |
| **Grad steps / iter** | 4 (fixed) | num_envs × replay_ratio (scales) |
| **num_envs effect** | Batch size ↑ | Grad steps ↑ AND batch diversity ↑ |
| **Buffer** | None (on-policy) | 100k capacity replay buffer |
| **Data freshness** | Always current policy | Mix of old and new data |

## 2. Comparison: PPO vs DreamerV3 Scaling Behavior

### 2.1 With 1 Environment

| | PPO | DreamerV3 |
|:---|:---|:---|
| Data per iter | 128 steps | 128 steps (collect_interval=128) or 1 step (collect_interval=1) |
| Grad steps per iter | **4** | **1** (replay_ratio=1) |
| Data per grad step | 128 samples | 64 × 128 = 8192 samples (batch × seq) |
| Data reuse | 4× (K_epochs) | Many× (replay buffer re-sampling) |

### 2.2 With 64 Environments

| | PPO | DreamerV3 |
|:---|:---|:---|
| Data per iter | 8,192 steps | 8,192 steps (collect_interval=128) or 64 steps (collect_interval=1) |
| Grad steps per iter | **4** | **64** (replay_ratio=1) |
| Data per grad step | 8,192 samples | 64 × 128 = 8,192 samples (batch × seq) |
| Data reuse | 4× (K_epochs) | Many× (replay buffer re-sampling) |

### 2.3 Key Insight

PPO's `K_epochs` plays a **similar role** to DreamerV3's `replay_ratio` — both control how many times the model trains on collected data before moving on. The difference:

- **PPO**: Trains exactly K_epochs times on the **same batch** (all envs, all steps), then throws it away
- **DreamerV3**: Trains num_envs × replay_ratio times on **random samples from the buffer**, which mixes old and new data

## 3. Configuration Reference

### 3.1 Current Config (`configs/models/recurrent_ppo.yaml`)

```yaml
agent:
  algorithm: "RecurrentPPO"
  sequence_length: 128      # Steps per rollout
  K_epochs: 4               # Gradient passes per rollout
  gamma: 0.95
  gae_lambda: 0.95
  eps_clip: 0.1
  entropy_coef: 0.01
  vf_coef: 0.5
  lr_actor: 0.0005
  rnn_type: "GRU"
  activation: "relu"
  return_mode: "GAE"
```

### 3.2 Practical Guidance

| Goal | Setting |
|:---|:---|
| More training per rollout | Increase `K_epochs` (e.g., 8) — but too high risks overfitting on-policy data |
| Richer gradients | Increase `num_envs` — larger batch per update |
| Longer context | Increase `sequence_length` — but increases memory and iteration time |
| Faster episodes | Increase `num_envs` — more episodes complete per iteration |
