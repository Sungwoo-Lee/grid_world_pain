# Training Metrics & Analysis System — Full Research Context

> **Purpose**: Comprehensive analysis of what metrics are needed to properly evaluate RecurrentPPO and DreamerV3 training in the GridWorld Pain interoceptive homeostasis environment, what's currently logged vs missing, and a phased plan to close the gaps.
> **Date**: 2026-03-04
> **Status**: PLANNED

---

## 1. Why Standard RL Metrics Are Insufficient Here

### The Environment Is Not a Standard Benchmark

GridWorld Pain is an **interoceptive homeostasis** task. The agent must simultaneously:
- **Forage** for food to avoid starvation (nutrition → 0 = death)
- **Avoid danger** (predators, hazards) that cause injury (injury → max = death)
- **Rest** to recover from injury (exponential recovery with rest streaks)
- **Balance drives** — in homeostatic mode, reward = `prev_drive - curr_drive` where `drive = sqrt((1 - satiation/max)² + (injury/max)²)`

This means:
1. **Reward alone doesn't tell the story** — in homeostatic mode, reward converges toward zero as the agent maintains setpoint. A flat reward curve could mean either "agent learned perfectly" or "agent learned nothing."
2. **Episode length = survival** — longer episodes always means the agent is doing better (avoiding death).
3. **Death cause matters enormously** — dying from starvation vs injury vs truncation (surviving to max_steps) tells completely different stories about agent competence.
4. **Action distribution reveals strategy** — an agent that never rests can't recover from injury. An agent that never eats will starve. A healthy agent shows a mixed policy with context-dependent behavior.

---

## 2. Current State: What's Already Logged to WandB

### 2.1 RecurrentPPO Metrics (logged in `train.py:800-859`)

**Episode metrics** (logged when episodes complete within an iteration):
```
Episode/Reward          — mean reward of completed episodes this iteration
Episode/Reward_Min      — min reward
Episode/Reward_Max      — max reward
Episode/Steps           — mean episode length
Episode/Number          — cumulative episodes completed
```

**Loss metrics** (logged every iteration):
```
loss/total              — combined PPO loss
loss/policy             — clipped surrogate policy loss
loss/value              — value function loss
loss/entropy            — entropy bonus (negative = encouraging exploration)
loss/grad_norm          — global gradient norm
```

**System metrics**:
```
timesteps               — cumulative environment steps
iteration               — training iteration count
```

**Neuromodulator metrics** (only when modulation enabled):
```
modulator/grad_norm, modulator/gamma_uni_mean, modulator/gamma_uni_std,
modulator/gamma_multi_mean, modulator/gamma_multi_std,
modulator/z_memory_mean, modulator/z_memory_std,
modulator/temperature_mean, modulator/temperature_min, modulator/temperature_max
(+ beta variants for PreActivation type)
```

### 2.2 DreamerV3 Metrics (logged in `train.py:984-1002`)

**Episode metrics** — same as RecurrentPPO (logged when episodes complete).

**World Model metrics** (logged every 10 iterations, prefixed with `WorldModel/`):
```
WorldModel/loss_model           — total WM loss
WorldModel/loss_recon           — observation reconstruction MSE
WorldModel/loss_rew             — reward prediction cross-entropy
WorldModel/loss_cont            — continue prediction BCE
WorldModel/loss_dyn_kl          — dynamics KL (posterior detached)
WorldModel/loss_rep_kl          — representation KL (prior detached)
WorldModel/loss_kl              — combined KL loss
WorldModel/model_reward_mae     — reward prediction MAE
WorldModel/model_reward_mae_pos — MAE on positive rewards only
WorldModel/model_reward_mae_neg — MAE on negative rewards only
WorldModel/model_latent_entropy — posterior categorical entropy
WorldModel/model_cont_acc       — continue prediction accuracy
```

**Behavior metrics** (prefixed with `Behavior/`):
```
Behavior/loss_actor_policy      — policy gradient component
Behavior/loss_actor_entropy     — entropy bonus component
Behavior/loss_critic            — critic loss
Behavior/mean_entropy           — policy entropy
Behavior/mean_return            — mean imagined return
Behavior/mean_norm_return       — mean normalized return
Behavior/mean_value             — mean value estimate
Behavior/mean_advantage         — mean advantage
```

**System**:
```
Params/effective_replay_ratio   — cumulative_grad_steps / global_step
value_mae                       — value prediction error (logged without prefix)
```

**Neuromodulator metrics** (if modulation enabled, prefixed `Modulator/`):
```
Modulator/mod_z_unimodal_mean, mod_z_unimodal_std,
Modulator/mod_z_multimodal_mean, mod_z_multimodal_std,
Modulator/mod_memory_mean, mod_memory_std,
Modulator/mod_z_reward_mean, mod_z_reward_std
(+ beta variants for PreActivation)
```

### 2.3 What the Existing Analysis Skill Can Do

The `wandb-analysis` skill (`scripts/wandb_metrics.py`) provides:
- **discover**: List all metrics in any run
- **config**: Show run hyperparameters
- **extract**: Pull time-series stats (steady-state mean/std, trajectory, min/max)
- **compare**: Side-by-side comparison of runs
- **presets**: `dreamer_v3` and `recurrent_ppo` curated metric lists with criteria

### 2.4 Known Preset Bugs

The `recurrent_ppo` preset in `wandb_metrics.py:90-107` references metrics that **don't exist**:

| Preset Metric | Actually Logged As | Status |
|---|---|---|
| `Policy/entropy` | `loss/entropy` | **Wrong prefix** |
| `Policy/policy_loss` | `loss/policy` | **Wrong prefix** |
| `Policy/value_loss` | `loss/value` | **Wrong prefix** |
| `Policy/approx_kl` | *Not computed at all* | **Missing from code** |
| `Policy/clip_fraction` | *Not computed at all* | **Missing from code** |

This means `--preset recurrent_ppo` currently shows "—" for most metrics. The preset is essentially broken.

---

## 3. What a Researcher Needs to Measure

### 3.1 Algorithm Health Checklist (Generic RL)

These apply to any RL algorithm and verify the training machinery is working:

| Check | What to Look For | Current Coverage |
|---|---|---|
| **Reward trend** | Improving over training | ✅ Logged |
| **Episode length** | Increasing (survival = longer) | ✅ Logged |
| **Loss convergence** | Losses decreasing/stabilizing | ✅ Logged for both |
| **Policy entropy** | Gradual decrease (not collapse) | ✅ RPPO: `loss/entropy`; DV3: `Behavior/mean_entropy` |
| **Value prediction quality** | Error decreasing | ⚠️ DV3: `value_mae` logged; RPPO: no explicit value MAE |
| **Gradient health** | Stable norms, no explosion | ⚠️ RPPO: `loss/grad_norm`; DV3: **not logged** |
| **NaN/missing data** | All metrics consistently logged | ✅ Can check via metric count |
| **Training speed** | Stable s/it over time | ✅ Via `benchmark_wandb_speed.py` |

### 3.2 Environment-Specific Behavioral Metrics (Homeostasis)

These are **critical for this project** and **none are currently logged**:

#### A. Death Cause Distribution
**Why**: An agent dying 100% from starvation needs to learn foraging. An agent dying 100% from injury needs to learn avoidance. An agent that survives to max_steps (truncation) is succeeding.

**Data available**: `info['termination_reason']` in `core.py:438-444`:
- 0 = active (shouldn't appear at episode end)
- 1 = max_steps (truncation — **good**, agent survived)
- 2 = starvation (nutrition ≤ 0)
- 3 = overeating (satiation ≥ max, if enabled)
- 4 = injury death (injury ≥ max_injury)

**Proposed WandB keys**:
```
Behavior/death_truncated_frac   — fraction of episodes ending by max_steps (goal: → 1.0)
Behavior/death_starvation_frac  — fraction dying from starvation (goal: → 0.0)
Behavior/death_injury_frac      — fraction dying from injury (goal: → 0.0)
Behavior/death_overeating_frac  — fraction dying from overeating (goal: → 0.0)
```

**Interpretation**: As training progresses, `death_truncated_frac` should increase toward 1.0 — the agent learns to survive the full episode. Early training will show mostly starvation/injury deaths.

#### B. End-of-Episode Physiological State
**Why**: Even if the agent dies, we want to know HOW close it got to homeostasis. Final nutrition/injury at episode end tracks gradual improvement.

**Data available**: `EnvState.nutrition`, `EnvState.injury_level`, `EnvState.satiation` — accessible from env_state when episodes terminate.

**Proposed WandB keys**:
```
Behavior/final_nutrition_mean   — mean nutrition at episode end (goal: → setpoint)
Behavior/final_injury_mean      — mean injury at episode end (goal: → 0.0)
Behavior/final_satiation_mean   — mean satiation at episode end (goal: → setpoint)
Behavior/mean_drive             — mean homeostatic drive = sqrt(drive_hunger + drive_injury)
```

**Interpretation**:
- Nutrition trending up = agent learning to eat
- Injury trending down = agent learning to avoid/rest
- Drive trending down = agent approaching homeostatic balance

#### C. Action Distribution
**Why**: In this environment, action choice directly reveals strategy. The agent has 4-6 actions: {Up, Right, Down, Left, Rest (optional), Eat (optional)}. A degenerate policy (e.g., always moves right) is easy to detect from action distribution.

**Data available**: `trajectories.action` (already in the rollout).

**Proposed WandB keys**:
```
Behavior/action_move_frac       — fraction of actions that are movement (0-3)
Behavior/action_rest_frac       — fraction of rest actions (4, if enabled)
Behavior/action_eat_frac        — fraction of eat actions (5, if enabled)
```

**Interpretation**:
- A healthy agent should show mixed actions, context-dependent
- If `rest_frac ≈ 0` when injury system is enabled → agent hasn't learned recovery
- If `eat_frac ≈ 0` with explicit eat action → agent isn't using the eat mechanic

#### D. Foraging & Avoidance Metrics
**Why**: Direct measures of the two core survival skills.

**Data available**:
- `info['dist_to_food']` — Manhattan distance to nearest food (computed in `core.py:480`)
- `info['dist_to_pred']` — Manhattan distance to nearest predator (`core.py:481`)
- Food consumption events can be inferred from reward in survival mode, or tracked via `ate_food` in core.py

**Proposed WandB keys**:
```
Behavior/food_eaten_per_ep      — food items consumed per episode
Behavior/mean_dist_to_food      — avg distance to nearest food (should decrease)
Behavior/mean_dist_to_pred      — avg distance to nearest predator (should increase)
```

**Interpretation**:
- Decreasing food distance + increasing food consumption = improving foraging
- Increasing predator distance = learning avoidance

#### E. Reward Decomposition (Homeostatic Mode)
**Why**: In homeostatic mode, total reward = `reward_homeostatic + reward_extrinsic - eating_penalty`. Understanding which component dominates reveals what the agent is actually optimizing.

**Data available**: `info['reward_homeostatic']`, `info['reward_extrinsic']` (computed in `core.py:472-473`).

**Proposed WandB keys**:
```
Reward/homeostatic_mean         — mean homeostatic reward component
Reward/extrinsic_mean           — mean extrinsic reward component
Reward/drive_hunger_mean        — mean hunger drive component
Reward/drive_injury_mean        — mean injury drive component
```

### 3.3 RecurrentPPO-Specific Diagnostics

Currently missing metrics that are standard PPO diagnostics:

| Metric | Why It Matters | How to Compute |
|---|---|---|
| **Approx KL** | Measures policy change per update. If KL > 0.03, updates are too aggressive (instability risk). If KL ≈ 0, agent isn't learning. | `mean(old_log_prob - new_log_prob)` — both values already computed in `ppo_loss_fn` |
| **Clip fraction** | Fraction of samples where the PPO clip is active. Should be 0.1-0.3. If 0, clipping is useless. If > 0.5, policy is changing too fast. | `mean(abs(ratio - 1) > clip_eps)` — ratio is already computed in `ppo_loss_fn` |
| **Explained variance** | How well the value function predicts returns. `1 - Var(targets - values) / Var(targets)`. Should approach 1.0. If near 0, critic is useless. | Computed from `batch.targets` and `batch.values` in `train_iteration` |

**Source files**:
- `ppo_loss_fn` in `src/models/recurrent_ppo_trainer.py:55-90` — ratio and log_probs already computed but approx_kl and clip_frac not returned
- `update_step` returns `(ppo_loss, v_loss, ent_loss, grad_norm, mod_grad_norm)` — needs to also return the new diagnostics
- `train.py:817-821` — extracts and logs the loss tuple

**Implementation**: Modify `ppo_loss_fn` to also return `approx_kl`, `clip_fraction`. Modify `update_step` to pass them through. Modify `train.py` to log them.

### 3.4 DreamerV3-Specific Diagnostics

#### Gradient Norms (Critical — Currently Missing)
DreamerV3 has three separate optimizers (world model, actor, critic). Currently **none** of their gradient norms are logged. This is a major diagnostic gap — gradient explosion or vanishing is undetectable.

**Source**: `src/models/dreamer_v3_trainer.py`
- Line 291-293: `grads_model` computed via `nnx.grad(model_loss_fn, ...)` then `self.model_opt.update()`
- Line 427-438: `grads_actor`, `grads_critic` computed and applied

**Proposed metrics**:
```
WorldModel/grad_norm            — optax.global_norm(grads_model)
Behavior/actor_grad_norm        — optax.global_norm(grads_actor)
Behavior/critic_grad_norm       — optax.global_norm(grads_critic)
```

**Implementation**: Add `optax.global_norm()` calls in `train_step()` and include in the returned metrics dict.

#### Replay Buffer Health
**Why**: If positive rewards are extremely rare in the buffer (< 1%), the reward head can't learn to predict them — which is the known `model_reward_mae_pos = 0` issue.

**Proposed metrics**:
```
System/buffer_pos_reward_frac   — fraction of buffer transitions with reward > 0.01
System/buffer_size              — current buffer utilization
System/buffer_mean_reward       — mean reward in buffer
```

**Implementation**: Cheap to compute from `buffer.rewards` array. Log in `train.py` alongside other DreamerV3 metrics.

#### Per-Modality Reconstruction Error (Advanced — Phase 3)
**Why**: The decoder reconstructs the full observation vector, but which modalities is it learning well vs ignoring? The observation is composed of: Injury(1d), Nutrition(1d), Satiation(1d), Extero_Noc(1d), Olfaction(5d), Collision(~9d), Proprioception(6d), Visual(var), Location(2d).

**Implementation**: In `model_loss_fn` (`dreamer_v3_trainer.py:200-201`), `loss_recon = mean(square(recon - obs))` averages over all dims. Instead, split the reconstruction error by modality using `get_observation_breakdown(params)` dimension mapping.

**Proposed metrics**:
```
WorldModel/recon_interoception  — reconstruction MSE for injury+nutrition+satiation dims
WorldModel/recon_olfaction      — reconstruction MSE for olfaction dims
WorldModel/recon_collision      — reconstruction MSE for collision dims
WorldModel/recon_location       — reconstruction MSE for location dims
... etc per enabled modality
```

**Note**: This is lower priority and more complex (requires passing `obs_breakdown` into the JIT-compiled loss). Could be Phase 3.

---

## 4. Implementation Plan

### Phase 1: Fix & Enhance Analysis Tools (No train.py changes)

**Goal**: Make the existing skill work correctly with currently logged metrics.

#### Step 1: Fix `recurrent_ppo` preset (`scripts/wandb_metrics.py:90-107`)
- Replace `Policy/entropy` → `loss/entropy`
- Replace `Policy/policy_loss` → `loss/policy`
- Replace `Policy/value_loss` → `loss/value`
- Remove `Policy/approx_kl` and `Policy/clip_fraction` (not logged yet)
- Add `loss/total`, `loss/grad_norm`, `Episode/Reward_Min`, `Episode/Reward_Max`

#### Step 2: Verify `dreamer_v3` preset (`scripts/wandb_metrics.py:56-88`)
- Cross-reference every metric key against actual `wandb.log()` calls in `train.py:984-1002`
- Note: Some metrics go through prefix routing logic (`train.py:990-1001`) — verify the final key names match

#### Step 3: Add `health` subcommand to `wandb_metrics.py`
- Auto-detect algorithm from run config
- Apply the health checklist programmatically
- Output structured diagnosis with pass/warn/fail per check
- Include environment-specific interpretation (homeostatic vs survival mode)

#### Step 4: Update `.claude/skills/wandb-analysis/SKILL.md`
- Add environment-specific health interpretation
- Document `health` subcommand
- Add "What metrics mean in this environment" section
- Note known limitations (behavioral metrics not yet logged)

### Phase 2: Add Behavioral & Diagnostic Logging (train.py changes)

**Goal**: Log behavioral metrics for both algorithms and fix diagnostic gaps.

#### Step 1: Extract behavioral data from training loops

Both algorithms use `lax.scan` for rollout collection. The `info` dict from `jax_step` contains termination_reason and body state but is currently **discarded** (assigned to `_`).

**Key problem**: `env_state` after the scan shows the *reset* state (auto-reset happens inside scan), not the terminal state. To get terminal physiological state, we must capture it during the scan.

**Recommended approach (Option A — minimal scan modification)**:
- In `recurrent_ppo_trainer.py:149-174`: Inside `scan_fn`, after `jax_step`, capture `info['termination_reason']` and body state from `next_state_raw` (before auto-reset). Add these to the `Transition` NamedTuple.
- In `dreamer_v3_trainer.py:533-577`: Same pattern — capture from the transition dict inside `scan_fn`.
- Then in `train.py`, extract these from the returned trajectories at `done=True` timesteps.

Memory impact: Adding 4 scalar fields (termination_reason, nutrition, injury, satiation) per timestep per env is negligible (~4 * T * B * 4 bytes).

#### Step 2: Add PPO diagnostic metrics

Modify `ppo_loss_fn` in `src/models/recurrent_ppo_trainer.py:55-90`:
```python
# Already computed: ratio = exp(new_log_prob - old_log_prob)
approx_kl = jnp.mean(old_log_prob - new_log_prob)  # mean(log(pi_old/pi_new))
clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
```
Return them alongside existing `(ppo_loss, v_loss, ent_loss)`.

Modify `update_step` (`recurrent_ppo_trainer.py:182-214`) to pass through the new values.

Modify `train.py:817-821` to extract and log:
```
Policy/approx_kl
Policy/clip_fraction
Policy/explained_var    (computed in train_iteration from batch.targets and batch.values)
```

#### Step 3: Add DreamerV3 gradient norms

Modify `train_step` in `src/models/dreamer_v3_trainer.py`:
- After line 291 (`grads_model`): `model_metrics['wm_grad_norm'] = optax.global_norm(grads_model)`
- After line 436-437 (`grads_actor`, `grads_critic`): add to behavior_metrics

These flow through the existing metric routing in `train.py:989-1001` automatically.

#### Step 4: Add behavioral logging to `train.py`

For both algorithms, after computing episode stats in the iteration loop:
```python
if wandb_enabled and iteration_episodes:
    wandb.log({
        # Existing episode metrics ...
        "Behavior/death_truncated_frac": ...,
        "Behavior/death_starvation_frac": ...,
        "Behavior/death_injury_frac": ...,
        "Behavior/action_move_frac": ...,
        "Behavior/action_rest_frac": ...,
        "Behavior/final_nutrition_mean": ...,
        "Behavior/final_injury_mean": ...,
    })
```

#### Step 5: Add buffer health logging for DreamerV3

After buffer updates (`train.py:889/905`), log periodically:
```python
if wandb_enabled and iteration % 50 == 0:
    wandb.log({
        "System/buffer_pos_reward_frac": float((buffer.rewards[:buffer.size] > 0.01).mean()),
        "System/buffer_size": buffer.size,
    })
```

#### Step 6: Update presets and SKILL.md
Add all new metrics to both presets with appropriate criteria. Update SKILL.md with behavioral analysis workflow.

---

## 5. Files to Modify

### Phase 1 (Analysis only — no retraining needed)
| File | Action | Description |
|---|---|---|
| `scripts/wandb_metrics.py` | EDIT | Fix recurrent_ppo preset, verify dreamer_v3 preset, add `health` subcommand |
| `.claude/skills/wandb-analysis/SKILL.md` | EDIT | Environment-specific health guidance, document `health` command |

### Phase 2 (New logging — requires retraining)
| File | Action | Description |
|---|---|---|
| `src/models/recurrent_ppo_trainer.py` | EDIT | Add approx_kl, clip_fraction to loss returns; add body state + termination_reason to Transition |
| `src/models/dreamer_v3_trainer.py` | EDIT | Add gradient norms to train_step metrics; add body state + termination_reason to transition dict |
| `train.py` | EDIT | Add behavioral metric logging for both algorithms; add buffer health logging for DV3 |
| `scripts/wandb_metrics.py` | EDIT | Update presets with new behavioral + diagnostic metrics |
| `.claude/skills/wandb-analysis/SKILL.md` | EDIT | Full behavioral analysis documentation |

---

## 6. Verification

### Phase 1 Verification
1. `PYTHONPATH=scripts python scripts/wandb_metrics.py extract <existing_rppo_run> --preset recurrent_ppo` — all metrics should resolve (no "—")
2. `PYTHONPATH=scripts python scripts/wandb_metrics.py health <existing_run>` — structured health report
3. Manual: Confirm preset keys match `wandb.log()` keys in train.py

### Phase 2 Verification
1. Short RecurrentPPO training (~100 episodes) → check WandB for:
   - `Behavior/death_*_frac`, `Behavior/action_*_frac`, `Behavior/final_*_mean`
   - `Policy/approx_kl`, `Policy/clip_fraction`, `Policy/explained_var`
2. Short DreamerV3 training (~100 episodes) → check WandB for:
   - `WorldModel/grad_norm`, `Behavior/actor_grad_norm`, `Behavior/critic_grad_norm`
   - `System/buffer_pos_reward_frac`
   - All behavioral metrics
3. Benchmark speed before/after to confirm no JIT recompilation overhead
4. Run full `health` command on new runs — should cover all new metrics

---

## 7. Live Run Analysis (2026-03-04)

Analysis of three active training runs using existing WandB metrics. This section demonstrates what we can currently diagnose and where the gaps from Section 3 become painfully visible.

### 7.1 Runs Under Analysis

| Alias | Full Tag | Algorithm | Run ID | Status |
|---|---|---|---|---|
| **RPPO** | `rppo_128envs_GAE_gradNorm_128default_128vis_olf_hub` | RecurrentPPO | `6jqakzwr` | Running |
| **DV3-noBody** | `dreamer_v3_1envs_16batch_128collect_replay1_hierarchical_1e6buffer_noBodyEncoding` | DreamerV3 | `lp7e8axs` | Running |
| **DV3-original** | `dreamer_v3_1envs_16batch_128collect_replay1_hierarchical_1e6buffer` | DreamerV3 | `m3oeo0dj` | Running |

### 7.2 Configuration Comparison

#### Key Hyperparameters

| Parameter | RPPO | DV3-noBody | DV3-original |
|---|---|---|---|
| **num_envs** | 128 | 1 | 1 |
| **encoding_mode** | hierarchical | hierarchical | hierarchical |
| **hidden_size** | 128 | 512 (RSSM deter) | 512 (RSSM deter) |
| **batch_size** | — (on-policy) | 16 | 16 |
| **collect_interval** | — | 128 | 128 |
| **sequence_length** | 128 | 128 | 128 |
| **replay_ratio** | — | 1 | 1 |
| **buffer_capacity** | — | 1,000,000 | 1,000,000 |
| **lr (actor)** | 5e-4 | 3e-5 | 3e-5 |
| **lr (critic/value)** | 1e-4 | 3e-5 | 3e-5 |
| **lr (world model)** | — | 1e-4 | 1e-4 |
| **entropy** | coef=0.01 | scale=3e-4 | scale=3e-4 |
| **gamma** | 0.95 | — (in RSSM) | — (in RSSM) |
| **GAE lambda** | 0.95 | — | — |
| **max_grad_norm** | 0.5 | — | — |
| **rnn_type** | GRU | RSSM (32×32 stoch) | RSSM (32×32 stoch) |
| **K_epochs** | 4 | — | — |

#### Critical Architecture Difference: `hierarchical_params`

| Component | RPPO | DV3-noBody | DV3-original |
|---|---|---|---|
| `default_mlp` | [128, 128] | [32, 32] | [128] |
| `multimodal_hub` | [128, 128] | [32, 32] | — |
| `unimodal_overrides` | visual=[128,128], olfaction=[128,128] | — | — |
| `hub_overrides` | — | — | body_state=[64], association=[128] |

**Key insight**: DV3-noBody uses **much smaller** default encoders (32-wide) with no body-state encoding path. DV3-original has a dedicated body_state pathway ([64]) and larger defaults ([128]). This architectural difference directly impacts how well each model can learn the interoceptive observation space.

### 7.3 Training Speed & Throughput

| Metric | RPPO | DV3-noBody | DV3-original |
|---|---|---|---|
| **s/iteration** | 0.27 | 0.24 | 0.26 |
| **Steps per second (SPS)** | **63,129** | 525 | 505 |
| **Wall-clock time** | 8h 25m | 21h 18m | 32h 51m |
| **Total timesteps** | 1.85B | 40M | 59M |
| **Total iterations** | 114k | 311k | 455k |

**Analysis**:
- **RPPO is 120× faster in SPS** than DreamerV3. This is expected: 128 parallel envs vs 1, and on-policy collection is inherently simpler than world model training.
- **Iteration speed is comparable** (~0.25 s/it for all three), but each RPPO iteration processes 128 envs × 128 steps = 16,384 transitions, while each DV3 iteration processes 1 env × 128 steps = 128 transitions.
- **DV3 effective data efficiency**: Despite 46× fewer timesteps (40M vs 1.85B), DV3 should compensate via replay — but the replay ratio is broken (see Section 7.5).
- **Practical implication**: At current SPS, DV3 needs ~22 days to reach 1B timesteps. RPPO reached 1.85B in 8.4 hours. If DreamerV3's data efficiency doesn't compensate for this speed gap, the algorithm choice is hard to justify.

### 7.4 Performance Comparison

#### Episode Length (Primary Success Metric)

| Metric | RPPO | DV3-noBody | DV3-original |
|---|---|---|---|
| **Early episode steps** | 27 | 21 | 66 |
| **Latest episode steps** | 229 | 75 | 100 |
| **Improvement** | **+202 (+748%)** | +54 (+257%) | +34 (+52%) |
| **% of max_steps (500)** | 45.8% | 15.0% | 20.0% |

**Interpretation**:
- RPPO has learned meaningful survival — agents live nearly half the episode. This is strong early-training progress.
- DV3-noBody barely survives 75 steps — likely dying from the first threat encountered.
- DV3-original starts better (66 vs 21) due to body-state encoding and larger networks, but improvement rate is slow.
- **None of the agents are close to surviving full episodes** (500 steps). Without death-cause logging (Section 3.2A), we cannot tell if they're dying from starvation, injury, or both.

#### Reward Trends

| Metric | RPPO | DV3-noBody | DV3-original |
|---|---|---|---|
| **Early reward** | -205 | -203 | -221 |
| **Latest reward** | -211 | -226 | -200 |
| **Trend** | Flat/slightly worse | **Getting worse** | **Improving** |
| **Steady-state mean** | -208 | -217 | -212 |
| **Reward min** | -1,102 | -802 | -867 |
| **Reward max** | -12 | -6 | -12 |

**Interpretation in context of homeostatic reward**:
- With `death_penalty = 100` and episode terminations causing large negative spikes, a mean reward of -200 to -220 is dominated by death events.
- The reward floor (~-1,100 for RPPO) corresponds to dying early with large accumulated negative drive changes.
- RPPO's flat reward despite massive step improvement: **this is the homeostatic reward paradox** described in Section 1. Longer survival means more timesteps of small negative drive deltas, while fewer deaths removes big death-penalty spikes. These effects roughly cancel.
- DV3-original improving from -221 to -200: fewer deaths per episode, consistent with modest step improvement.
- **DV3-noBody getting worse** (-203 → -226): this is a red flag. The agent is dying more frequently or accumulating more negative reward per step.

#### Loss Metrics

**RecurrentPPO:**

| Metric | Early | Latest | Trend |
|---|---|---|---|
| loss/total | 1,958 | 64 | ✅ Converging |
| loss/policy | 0.01 | 0.007 | ✅ Stable |
| loss/value | 1,953 | 57 | ✅ Converging (30× reduction) |
| loss/entropy | -1.79 | -0.29 | ⚠️ Entropy declining significantly |
| loss/grad_norm | 14 | 22 (max: 194) | ⚠️ High gradient norm with spikes |

**DV3-noBody:**

| Metric | Early | Latest | Trend |
|---|---|---|---|
| loss_model | 10.0 | 8.7 | Slow decrease |
| loss_recon | 0.31 | 0.06 | ✅ Converging |
| loss_rew | 4.86 | 1.60 | ✅ Decreasing but still high |
| loss_dyn_kl | 0.94 | 4.25 | ⚠️ **Increasing** |
| loss_rep_kl | 0.91 | 3.59 | ⚠️ **Increasing** |
| model_latent_entropy | 3.4 | 0.9 | ❌ **Entropy collapsed** |
| model_reward_mae | 0.47 | 0.15 | ✅ Decreasing |
| model_reward_mae_pos | 0.74 | 0.15 | ⚠️ Near zero — see Section 7.5 |
| model_cont_acc | 0.97 | 0.93 | ✅ Stable |
| value_mae | 0.57 | 8.47 | ❌ **Increasing — critic diverging** |
| effective_replay_ratio | 0.008 | 0.008 | ❌ **Stuck** — see Section 7.5 |

**DV3-original:**

| Metric | Early | Latest | Trend |
|---|---|---|---|
| loss_model | 10.3 | 8.1 | Slow decrease |
| loss_recon | 0.31 | 0.06 | ✅ Converging |
| loss_rew | 4.53 | 1.17 | ✅ Better than noBody |
| loss_dyn_kl | 1.13 | 4.92 | ⚠️ **Increasing** |
| loss_rep_kl | 1.28 | 4.31 | ⚠️ **Increasing** |
| model_latent_entropy | 3.4 | 0.80 | ❌ **Entropy collapsed** |
| model_reward_mae | 0.52 | 0.07 | ✅ Good |
| model_reward_mae_pos | 0.79 | 0.02 | ❌ **Near zero — positive reward invisible** |
| model_cont_acc | 0.96 | 0.93 | ✅ Stable |
| value_mae | 0.09 | 6.55 | ❌ **Increasing — critic diverging** |
| effective_replay_ratio | 0.008 | 0.008 | ❌ **Stuck** |

### 7.5 DreamerV3 Pathology Diagnosis

Both DreamerV3 runs exhibit multiple concerning pathologies. These are listed in order of severity.

#### Pathology 1: Entropy Collapse (CRITICAL)

**Symptom**: `model_latent_entropy` drops from 3.4 → 0.8-0.9 (should stay in 1.0–2.5 range for 32×32 categoricals).

**What this means**: The RSSM posterior is collapsing to near-deterministic states. Instead of maintaining a diverse latent space that captures environmental stochasticity, the model is "memorizing" — mapping observations to fixed latent categories. This destroys the stochastic imagination that DreamerV3's actor-critic training depends on.

**Likely causes**:
1. **KL losses increasing** (dyn_kl: 0.9→4.5, rep_kl: 0.9→4.0) — the posterior and prior are diverging. The prior can't keep up with the posterior, which means imagination (which uses the prior) generates increasingly unrealistic trajectories.
2. **Only 1 environment** — with a single env, the data distribution is highly correlated and non-stationary, making posterior collapse more likely.
3. **Possible free-bit threshold issue** — DreamerV3 uses a free-bits KL regularization. If the threshold is too low, it doesn't prevent collapse.

**Impact**: The actor trains on imagined trajectories generated by the prior. If prior ≠ posterior (KL diverging), the imagined experience is unrealistic → actor learns a bad policy → bad policy generates bad real data → vicious cycle.

#### Pathology 2: Critic Divergence (CRITICAL)

**Symptom**: `value_mae` increasing over training: 0.57→8.47 (noBody), 0.09→6.55 (original). The critic is getting WORSE at predicting returns.

**What this means**: The value function, which is trained on imagined returns from the world model, is diverging. Since the world model's imagination is degrading (Pathology 1), the critic is fitting to garbage targets.

**This creates a cascading failure**: Bad world model → bad imagined returns → bad value estimates → bad advantages → bad actor updates → bad policy → bad data collection → bad world model updates.

#### Pathology 3: Positive Reward Blindness (HIGH)

**Symptom**: `model_reward_mae_pos` → 0.02-0.15, while `model_reward_mae` → 0.07-0.15.

**What this means at first glance**: The MAE for positive rewards is near zero — this looks like perfect prediction. But it's actually **near zero because there are almost no positive reward samples**. When positive rewards are extremely rare in the buffer (< 1% of transitions), the metric effectively becomes `mean([])` or is dominated by noise.

**Why positive rewards are rare**: In homeostatic mode, positive reward occurs when `curr_drive < prev_drive` (the agent reduced its deviation from homeostasis). Early in training, the random policy almost never achieves this — it bumbles into danger, takes damage, and the drive monotonically increases until death.

**This is the core DreamerV3 challenge for this environment**: The world model can't learn to predict positive reward if it almost never sees it. The reward head learns "always predict zero or negative" — which means the actor's imagined rollouts never show benefit from eating, resting, or avoiding danger. The actor has no gradient signal toward beneficial behavior.

**This connects to the missing buffer health metrics** proposed in Section 3.4: `System/buffer_pos_reward_frac` would directly diagnose this.

#### Pathology 4: Effective Replay Ratio Anomaly (MEDIUM)

**Symptom**: `effective_replay_ratio` stuck at 0.0078 for both runs.

**Expected value**: With `replay_ratio = 1` and `batch_size = 16`, each collected sequence should trigger 1 gradient step → ratio ≈ 1.0.

**Calculation**: `ratio = cumulative_grad_steps / global_step`. At ~40M steps with 311k iterations and 1 gradient step per iteration: `311k / 40M = 0.0078`. This means each gradient step covers only 0.78% of collected data — the model is drastically under-training relative to data collected.

**Root cause**: With `collect_interval = 128` and `train_steps = 64` (noBody config), each collection of 128 transitions triggers 64 gradient steps. But `global_step` increments by 128 per collection while `grad_steps` increments by 64: `64/128 = 0.5`. However, `effective_replay_ratio` in the code uses `cumulative_grad_steps / global_step` where `global_step` is the total environment step count, not the number of collections. This makes the metric misleading — it's not a true replay ratio but rather `grad_steps / env_steps`.

**True concern**: With batch_size=16 and sequence_length=128, each batch uses 2,048 transitions from a 1M buffer. With 64 train_steps per collection, the model sees `64 × 2,048 = 131,072` transition-updates per 128 new transitions. The actual replay ratio (updates per new transition) is ~1,024, which is very high — potentially causing overfitting to old data.

### 7.6 RPPO Health Assessment

RPPO is in significantly better shape than DreamerV3, but has its own concerns:

#### Positive Signs
- **Episode length +748%** — the strongest learning signal across all three runs.
- **Value loss converging** (1,953 → 57) — the critic is fitting well.
- **Policy loss stable** (~0.007) — PPO clipping is working, updates are controlled.

#### Concerns

**Entropy decline** (loss/entropy: -1.79 → -0.29): Policy is becoming more deterministic. With `entropy_coef = 0.01`, the entropy bonus is very weak. At -0.29, the policy has lost most of its exploration capacity. In a homeostatic environment with multiple interacting systems, premature policy collapse means the agent may have found a local strategy (e.g., "always move in one direction") without discovering rest/eat mechanics.

**Without action distribution logging** (Section 3.2C), we cannot confirm this — but the entropy trend is suggestive of policy collapse toward a dominant action.

**Gradient spikes** (max: 194 vs steady-state: 22): Despite `max_grad_norm = 0.5`, gradient norms reach 194 before clipping. These spikes indicate episodes where value targets change dramatically — likely when the agent encounters death events that create large TD errors. The clipping prevents divergence but the underlying signal is noisy.

**Missing diagnostics**: Without `approx_kl` and `clip_fraction` (Section 3.3), we can't assess whether the PPO clipping is doing meaningful work or if the policy is changing too aggressively per update.

### 7.7 Cross-Algorithm Insights

#### DreamerV3 Body Encoding Matters

Comparing the two DV3 runs with the only architectural difference being hierarchical_params:

| Aspect | DV3-noBody (small, no body) | DV3-original (larger, body path) |
|---|---|---|
| Episode steps | 21 → 75 | 66 → 100 |
| Reward trend | Worsening (-203 → -226) | Improving (-221 → -200) |
| loss_rew | 4.86 → 1.60 | 4.53 → 1.17 |
| value_mae | 0.57 → 8.47 | 0.09 → 6.55 |

DV3-original is better on every metric. The dedicated body_state encoding pathway (`hub_overrides: {body_state: [64]}`) gives the world model a structured way to learn interoceptive dynamics, and the larger default_mlp ([128] vs [32,32]) provides more capacity. This confirms that **hierarchical encoding with body-aware pathways is important for interoceptive tasks**.

However, DV3-original still exhibits all four pathologies — the body encoding slows the collapse but doesn't prevent it. The fundamental issues are algorithmic (single environment, KL divergence, sparse positive rewards), not architectural.

#### Data Efficiency: The Broken Promise

DreamerV3's raison d'être is data efficiency — learning from fewer environment interactions by leveraging a learned world model. In this experiment:

- **RPPO**: 1.85B timesteps, 229 episode steps → 0.12 episode-steps per 1M timesteps
- **DV3-original**: 59M timesteps, 100 episode steps → 1.69 episode-steps per 1M timesteps
- **DV3-noBody**: 40M timesteps, 75 episode steps → 1.88 episode-steps per 1M timesteps

**Per-timestep, DreamerV3 is ~14× more data-efficient than RPPO**. But the world model pathologies are capping absolute performance far below what RPPO achieves. The world model is learning *something* from its limited data, but what it learns is increasingly corrupted by entropy collapse and critic divergence.

### 7.8 Recommendations

#### Immediate (No Code Changes)
1. **Monitor DV3-noBody closely** — reward is trending worse. If it doesn't recover within another 20M timesteps, the run should be considered failed.
2. **Let DV3-original continue** — reward is improving, but slowly. The body encoding architecture is validated.
3. **RPPO is the current best** — let it continue to at least 5B timesteps to see if episode steps approach 400+ (80% survival).

#### Short-Term (Phase 1 from Section 4)
1. **Fix the wandb_metrics.py presets** — the recurrent_ppo preset is broken, making automated health checks impossible.
2. **Add the `health` subcommand** — manual metric interpretation (as done in this section) should be automated.

#### Medium-Term (Phase 2 from Section 4)
1. **Add behavioral logging** — death cause distribution would immediately clarify whether agents die from starvation or injury, informing whether the issue is foraging or avoidance.
2. **Add action distribution logging** — verify the RPPO entropy collapse correlates with action collapse.
3. **Add DV3 gradient norms** — the critic divergence (value_mae increasing) could be caused by gradient explosion in the critic optimizer, but we can't check without norms.

#### DreamerV3-Specific Fixes for Next Experiments
1. **Increase num_envs** to at least 4-8 for DreamerV3. Single-env DreamerV3 leads to highly correlated buffer data, contributing to posterior collapse. The original DreamerV3 paper uses 1 env for Atari but with much simpler dynamics — this environment's stochastic predators and multi-system homeostasis likely needs more diverse data.
2. **Investigate KL balance** — the free-bits threshold may need tuning. Increasing it from the default could prevent the entropy collapse. Alternatively, try KL balancing with a higher `kl_free` value (e.g., 1.0 → 3.0).
3. **Positive reward augmentation** — consider adding small reward shaping (e.g., +0.01 for being near food, -0.01 for being near predators) to make positive rewards less sparse, giving the reward head more training signal. This can be done in the reward function without changing the fundamental homeostatic objective.
4. **Buffer prioritization** — prioritized experience replay weighting positive-reward transitions higher could help the reward head learn faster. This is a more complex change but addresses the root cause of Pathology 3.

### 7.9 What We Can't Diagnose (Gaps Confirmed)

This analysis hit every diagnostic gap identified in Section 3:

| Missing Metric | What We Couldn't Determine |
|---|---|
| **Death cause distribution** | Are agents dying from starvation or injury? Which survival skill is lacking? |
| **Action distribution** | Is RPPO's entropy decline actually policy collapse, or a healthy specialization? |
| **Final physiology** | How close to homeostasis are agents at death? Is nutrition trending up? |
| **DV3 gradient norms** | Is critic divergence caused by gradient explosion? |
| **Buffer reward distribution** | What fraction of DV3 buffer contains positive rewards? |
| **Reward decomposition** | Is the homeostatic component or extrinsic component dominating? |

**This directly motivates the Phase 2 implementation plan.**
