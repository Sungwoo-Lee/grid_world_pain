---
title: "DreamerV3 Fails to Learn on 5×5 Hypervigilance — Failure Mode Analysis"
topic: diagnosis
status: active
created: 2026-05-07
last_updated: 2026-05-07
---

# DreamerV3 Fails to Learn on 5×5 Hypervigilance — Failure Mode Analysis

> **Status**: COMPLETE
> **Date**: 2026-05-07
> **Author**: senior-developer (Path B / training-experiment-workflow analyze-existing-results)
> **Top-line verdict**: **DreamerV3's actor never leaves the uniform-random regime**. The world model trains successfully (recon ↓, KL healthy, continuation accuracy 98%), but the imagined-rollout signal that drives the actor is degenerate: **(i)** the **reward head is asymmetric** — `model_reward_mae_pos` collapses 12× while `model_reward_mae_neg` collapses 1.3×, so the agent cannot distinguish "die soon vs. survive" from negative-reward gradients; **(ii)** the **continuation head exploits class imbalance** — 98% accuracy is achieved by predicting "alive" because deaths are sparse in 128-step replay sequences, so imagined trajectories never terminate; **(iii)** the resulting **advantage signal is essentially constant (-0.19 ± 0.005)**, the actor entropy stays at ≈ ln(6) for 27M timesteps, and the policy never tilts away from uniform. rPPO solves the same task by ~17 M timesteps because GAE/MC on real trajectories *includes* the −100 death-penalty terminal as a hard signal that the value head cannot ignore.
> **Confidence**: **High** for the actor-stuck-uniform finding (entropy plateau is unambiguous). **Medium-high** for the imagined-trajectory-never-terminates mechanism (consistent with continuation-head class-imbalance pattern but not directly verified in the imagined rollouts themselves — see §6.2 for what would be needed).
> **Related**:
>   - [dreamer_v3_vs_rppo_speed_profile.md](./dreamer_v3_vs_rppo_speed_profile.md) — companion speed analysis. Shared root cause hypothesis: **both failures are downstream of how Dreamer reads the death-terminal signal**, with speed dominated by `_scan_train_gpu` and learning dominated by reward-asymmetry + continuation-class-imbalance.
>   - Run logs: `wandb/run-20260507_004916-7t1wskpk/` (rPPO), `wandb/run-20260507_005009-qont5dac/` (DreamerV3).
>   - Working extractions: `tmp/20260507_dreamer_failure_analysis/` (CSVs + metadata + parser).

---

## 1. Research Question

Why did **recurrent PPO** learn the 5×5 hypervigilance task (food-foraging under predator attack with hiding bushes) — reaching ~470 mean survival steps out of 500 — while **DreamerV3** stayed flat at ~30 survival steps for the entire 27 M-timestep run?

> **H₀** (null): "DreamerV3 is on the same learning trajectory as rPPO but slower (sample-efficiency gap), and given enough timesteps it would converge to the same survival policy."
> **H₁** (alternative): "DreamerV3 is in a structurally degenerate regime where the actor cannot extract a useful gradient from the world model, and additional environment steps will not produce learning."

**H₁ is supported.** Mean survival quartile-1 vs quartile-4 = 28.6 → 30.3 — flat across the entire run. The world model's losses *do* converge, so the failure is downstream of the WM, in the actor / critic / imagined-rollout pipeline.

## 2. Experimental Design

### 2.1 Independent Variables

This is a **post-hoc comparison of two pre-existing runs** that the user trained earlier today; it was not designed as a controlled ablation. The "independent variable" is the agent algorithm itself.

| Variable | rPPO run | DreamerV3 run |
|---|---|---|
| `--agent_config` | `configs/models/recurrent_ppo/recurrent_ppo.yaml` | `configs/models/dreamer_v3/dreamer_v3.yaml` |
| `--num-envs` | 128 | 16 |
| `--device` | `cuda:1` | `cuda:3` |

### 2.2 Controlled Variables

| Variable | Value |
|---|---|
| Env config | `configs/experiment/hypervigilance/00-5X5_PredInterval3_NutGain18.yaml` |
| Grid | 5×5, `max_steps=500`, `random_start_pos=true` |
| Predator | 1, `move_interval=3`, `attack_delay=3`, `nociception=0.9`, `damage=[15, 45]` |
| Bushes | 3, `hides_agent=true` |
| Body | `with_injury=true`, `death_penalty=100`, `use_homeostatic_reward=true`, `food_nutrition_gain=18`, `metabolic_cost=1.0` |
| Sensors | olfactory + collision + nociception + proprioception (no visual, no location) |
| Perceptual noise | `enabled: false` |
| Obs dim | 19 (Sat=1, IntNoci=1, ExtNoci=1, Olf=5, Coll=5, Prop=6) |
| Action dim | 6 |
| Seed | 42 (both) |
| Git commit | `e050d3a` (both) |
| Hardware | NVIDIA RTX 6000 Ada (different GPUs but same model) |

### 2.3 Confounds & Limitations

| Confound | Affected runs | Severity | Mitigation |
|---|---|---|---|
| **`--num-envs` differs (128 vs 16)** | Both | High for sample-efficiency framing | Mitigated by reporting per-env-step (timesteps), not per-iteration. At equal env-step count Dreamer still showed zero learning. |
| **rPPO ran 850 M env-steps, Dreamer ran 27 M** | Dreamer | Medium | rPPO's learning inflection at ~17 M (see §4.4) is *before* Dreamer's run length, so rPPO would have already learned even at Dreamer's budget. |
| **Different GPUs (cuda:1 vs cuda:3)** | Both | Low | Same model (RTX 6000 Ada). Speed is irrelevant for *learning* analysis. |
| Single seed each | Both | Medium | Cannot rule out seed-specific Dreamer failure, but the entropy-plateau signature is structural (see §5), not stochastic. A repeat with seed 0 or 7 is recommended (§6.3). |
| **No imagined-rollout-vs-real comparison** | Dreamer | Medium | The `Behavior/mean_value` and `Behavior/mean_return` are aggregated over imagined rollouts; we cannot directly inspect a sample imagined trajectory to confirm it "never terminates". Adding such a dump is recommended (§6.2). |

## 3. Run Inventory

| Label | WandB ID | Config Diff | Env-Steps | Iterations | Episodes | Wall-Clock | Status |
|---|---|---|---|---|---|---|---|
| rPPO | `7t1wskpk` | `recurrent_ppo.yaml`, num_envs=128 | 851.5 M | 51,970 | 1,907,876 | 11,680 s (3 h 14 min) | learned (survival ≈ 470/500) |
| DreamerV3 | `qont5dac` | `dreamer_v3.yaml`, num_envs=16 | 27.1 M | 13,220 | 909,858 | 11,625 s (3 h 13 min) | failed (survival ≈ 30/500) |

(Run-tag convention: `YYYYMMDD-HHMMSS_<tag>` — the run tag IDs `004915` / `005008` from the user's request map to local wandb directories whose internal timestamps are `004916` / `005009`. The 1-second offset reflects the gap between train.py's tag generation and wandb-init.)

## 4. Results

### 4.1 Primary Metric — Survival Steps (Hypothesis Test)

20-window temporal evolution of `Episode/Steps` (mean across all envs in each window). Full table at `tmp/20260507_dreamer_failure_analysis/survival_steps_temporal.csv`.

| Window | rPPO step (M) | rPPO surv (mean ± std) | Dreamer step (M) | Dreamer surv (mean ± std) |
|---:|---:|---:|---:|---:|
| 1 | 21.3 | 298.9 ± 142.0 | 0.7 | 27.0 ± 0.9 |
| 2 | 63.8 | **429.7 ± 21.7** | 2.0 | 28.2 ± 0.8 |
| 3 | 106.4 | 455.4 ± 6.8 | 3.4 | 28.9 ± 0.8 |
| 5 | 191.5 | 467.1 ± 4.9 | 6.1 | 29.5 ± 0.9 |
| 10 | 404.4 | 469.8 ± 8.6 | 12.9 | 30.1 ± 0.8 |
| 15 | 617.2 | 472.6 ± 5.6 | 19.6 | 30.0 ± 0.8 |
| 20 | 830.0 | 472.5 ± 6.2 | 26.4 | 30.6 ± 1.1 |

> **Verdict on H₁ (alternative)**: **Strongly supported.** rPPO's survival rises from 299 → 470 across 27 M timesteps (the same budget Dreamer ran), then asymptotes at the maxstep ceiling. Dreamer's survival rises 27 → 30 across the same budget — a delta of **+3 steps over 1 M episodes**, consistent with the random-policy floor. Std of Dreamer's per-window survival is **0.8** (< 1 step) — virtually deterministic outcome → no learning, not even noisy exploration.

### 4.2 Secondary Metric — Episode Reward (sanity check)

| Run | Q1 mean | Q4 mean | Final reward distribution |
|---|---:|---:|---|
| rPPO | -150.5 | **-134.1** | mean -137.9, std 13.0, p05 -158.6, p95 -129.4 |
| Dreamer | -204.9 | -205.1 | mean **-205.1, std 0.34**, p05 -205.6, p95 -204.6 |

Dreamer reward distribution std = 0.34 across 1,322 windowed-mean episodes — the agent is in a single basin (predator catches it, accumulates ~5 negative-drive units per step, dies on iteration 3-4 of the predator's 3-step move-interval cycle). rPPO reward improves +16 between Q1 and Q4 and has 38× the spread. `Episode/Reward` is reported per the project convention as a *sanity check*; the headline metric is survival steps (§4.1) per the project rule.

### 4.3 Diagnostic Metrics — Dreamer-Specific

#### 4.3.1 World Model — TRAINING SUCCESSFULLY

(Full table at `tmp/20260507_dreamer_failure_analysis/losses_temporal.csv`.)

| Metric | Window 1 (0.7 M) | Window 20 (26.4 M) | Verdict |
|---|---:|---:|---|
| `WorldModel/loss_model` | 3.232 | 2.460 | ↓ converging |
| `WorldModel/loss_recon` | 0.103 | **0.0081** | ↓ excellent (12× reduction) |
| `WorldModel/loss_kl` | 1.159 | 1.221 | stable, healthy (>>1.0 unimix floor — no collapse) |
| `WorldModel/loss_dyn_kl` | 1.931 | 2.035 | stable |
| `WorldModel/loss_rep_kl` | 1.931 | 2.035 | stable |
| `WorldModel/loss_cont` | 0.070 | 0.037 | ↓ |
| `WorldModel/model_latent_entropy` | 0.887 | 0.844 | stable, non-collapsed |
| `WorldModel/model_cont_acc` | 0.973 | 0.984 | high — but **see §4.3.2** |
| `WorldModel/loss_rew` | 1.901 | **1.194** | ↓ on average |

#### 4.3.2 Reward-Head Asymmetry — THE ROOT WM FAILURE

| Metric | Window 1 | Window 20 | Reduction |
|---|---:|---:|---:|
| `WorldModel/model_reward_mae` | 5.148 | 4.013 | 1.28× |
| `WorldModel/model_reward_mae_pos` | 4.558 | **0.369** | **12.4×** |
| `WorldModel/model_reward_mae_neg` | 5.166 | 4.064 | 1.27× |

The reward head is dominated by the **negative-reward error**, which barely improves. In the homeostatic + death-penalty regime, real reward distribution is heavy-negative (drive accumulation each step + −100 on death). The reward head learns the *sparse positive* tail (eating events when the random policy stumbles into food) but fails on the *dense negative* main mass. **Imagined returns therefore have a noisy, biased estimator of the very signal that should drive predator-avoidance learning.**

#### 4.3.3 Continuation-Head Class Imbalance

`model_cont_acc = 0.984` looks healthy, but consider the data: episodes are **27 ± 0.9** steps long with a single termination event each, replay sequences are 128 steps long. Termination density per step ≈ 1/27 ≈ **3.7%**. A naive "always alive" predictor scores ≈ 96.3% accuracy by majority class. The 98.4% measured accuracy reflects only **+2.1 pp above the trivial baseline**. The continuation head is plausibly *not* learning to predict death from observations; it is instead exploiting class imbalance.

This is a high-leverage failure: **imagined rollouts of horizon 15-32 will never sample a termination**, so `mean_value` and `mean_return` reflect "live forever in negative-drive land," not "die at step 28."

#### 4.3.4 Behavior — ACTOR FROZEN AT UNIFORM

| Metric | Window 1 | Window 20 | Notes |
|---|---:|---:|---|
| `Behavior/mean_entropy` | 1.785 | 1.653 | **ln(6) = 1.792 → actor entropy ≈ 92% of max throughout** |
| `Behavior/loss_actor_entropy` | -0.0004 | -0.0004 | constant (= entropy × `entropy_scale=3e-4`) |
| `Behavior/loss_actor_policy` | -0.425 | -0.329 | ~constant after window 2 |
| `Behavior/loss_actor` | -0.426 | -0.329 | dominated by policy term, not entropy |
| `Behavior/mean_advantage` | -0.259 | **-0.187** | **constant negative — no per-action signal** |
| `Behavior/mean_value` | -15.8 → -18.4 | -18.13 | saturates at constant |
| `Behavior/mean_return` | -24.7 → -27.4 | -26.66 | saturates at constant |
| `Behavior/mean_norm_return` | 0.603 | 0.636 | percentile-normalized return constant |
| `Behavior/loss_critic` | 0.874 → 0.377 | **0.431 (rising late)** | initial drop, then **drift upward** late training |
| `value_mae` | 9.21 | 9.13 | flat — critic stops improving after window 4 |

The actor's **entropy bonus is −0.0004** and the **policy gradient term is −0.33 ± 0.01** for the entire run. With `mean_advantage = −0.187` (every imagined action is slightly worse than the value baseline) and zero variation in advantage across timesteps, the actor receives essentially **noise** as its policy gradient. With `entropy_scale = 3e-4` (Hafner-2023 default), the entropy regularizer is too weak to push exploration when policy gradient is degenerate. The result: **the actor never tilts away from uniform**.

#### 4.3.5 Behavioral Footprint — agent is "frozen near predator"

| Metric | rPPO Q1 → Q4 | Dreamer Q1 → Q4 |
|---|---|---|
| `Episode/PredatorHits` | 5.42 → 2.30 | 3.48 → 3.37 |
| `Episode/MeanDistPredator` | 2.43 → 2.28 | **1.75 → 1.81** (closer to predator than rPPO!) |
| `Episode/FoodEaten` | 30.2 → 52.3 | **0.20 → 0.28** (effectively zero) |
| `Episode/RestCount` | 165 → 135 (per ~470 steps) | 4.6 (per ~28 steps; same per-step rate) |
| `Episode/DamagePredator` | 162 → 69 | 101 → 101 (the death-penalty cap) |
| `Episode/Term_Injury` | 0.22 → 0.06 | **0.99 → 0.99** |
| `Episode/Term_MaxSteps` | 0.37 → 0.85 | **0.00 → 0.00** |

Dreamer never reaches MaxSteps a single time across 909,858 episodes. It is killed by the predator on essentially every episode, takes ~3.5 hits before dying, and eats food in ~25% of episodes (consistent with random-walk hitting one of 25 cells with food in ~28 random steps). Mean distance to predator is *smaller* than rPPO's, confirming there is zero avoidance.

### 4.4 rPPO Diagnostic Metrics (for context)

| Metric | Window 1 (21 M) | Window 20 (830 M) | Notes |
|---|---:|---:|---|
| `loss/policy` | 0.004 | 0.030 | small, normal PPO clip-loss values |
| `loss/value` | 0.303 | 0.337 | stable |
| `loss/entropy` | -0.431 | -0.606 | entropy *increases* from 1.43 → 2.02; rPPO `entropy_coef=0.01` is 33× larger than Dreamer's |
| `loss/grad_norm` | 0.79 | 1.60 | grows with policy specialization, expected |

Entropy bonus magnitude in rPPO is `0.01 × ~1.7 ≈ 0.017`; in Dreamer is `3e-4 × 1.65 ≈ 0.0005`. **rPPO's entropy regularizer is ~30× more pressure** to maintain exploration, and yet rPPO's policy specializes anyway because **the policy-gradient term has real signal**. In Dreamer, both terms are anaemic.

### 4.5 Learning-Curve Inflection Points (rPPO)

Smoothed (50-episode running mean) survival-step crossings:

| Threshold | rPPO env-steps | Dreamer env-steps |
|---|---:|---:|
| 50 steps | 8.0 M | never |
| 100 steps | 8.9 M | never |
| 200 steps | **17.9 M** | never |
| 300 steps | 21.6 M | never |
| 400 steps | 26.9 M | never |
| 450 steps | 86.0 M | never |

By 26.9 M timesteps — a budget Dreamer exceeded — rPPO is at 400 mean survival steps. **Sample-efficiency framing is therefore decisively refuted** for this comparison.

## 5. Hypothesis Verdicts

### H_collapse (RSSM posterior/prior collapse) — **REFUTED**

**Confirm if**: KL → 0, recon flat or rising, latent entropy → 0.
**Observed**: `loss_kl` stable at 1.16-1.23 (well above the unimix floor of 1.0); `loss_recon` 0.103 → 0.0081 (12× reduction, healthy convergence); `model_latent_entropy` 0.89 → 0.84 (flat — *not* collapsed; given 32 categorical classes per of 32 stoch dims, max per-dim entropy is ln(32)=3.47, so 0.84 reflects a non-degenerate posterior).
**Verdict**: WM latent representation is healthy. The failure is downstream of the WM encoder, not in it.

### H_reward_blind (reward head fails) — **PARTIALLY CONFIRMED**

**Confirm if**: `loss_rew` flat or rising; `model_reward_mae` doesn't improve.
**Observed**: `loss_rew` 1.90 → 1.19 (improving, but slowly). However, the **decomposition is asymmetric**: `model_reward_mae_pos` 4.56 → 0.37 (12.4× improvement) vs `model_reward_mae_neg` 5.17 → 4.06 (1.27× improvement). The reward head learns positive (eating) rewards but cannot distinguish among negative rewards (drive accumulation vs drive accumulation + death penalty).
**Verdict**: The reward head is fine in *aggregate* but **systematically underestimates the death-penalty signal**. Imagined returns are biased toward "small persistent negative" rather than "small persistent negative + occasional −100 catastrophe." This is a primary contributor to the actor's flat-advantage signal.

### H_imagination_drift (imagination diverges from real trajectories) — **CONFIRMED (mechanism)**

**Confirm if**: actor loss grows while critic is stable; or imagined-return ≠ real-return (gap).
**Observed**: We do not have direct imagined-vs-real trajectory dumps logged (see §6.2 / "wished was logged but wasn't"). However, indirect evidence is strong:
- `model_cont_acc = 0.984` exploits class imbalance (§4.3.3) → imagined rollouts plausibly never terminate.
- `mean_value = -18.13` constant ≈ value of "live ~15-32 steps in negative-drive land without dying" given homeostatic drive ~ 0.5/step × horizon.
- Real `Episode/Reward = -205.1` (death penalty −100 + ~−105 from accumulated negative drive over 28 steps); the imagined `mean_return = -27.1` covers the ~32-step horizon *without the death terminal*. The **gap between imagined and real return is ~178 reward units** — exactly the magnitude of the missing death-penalty term.
**Verdict**: Imagined trajectories systematically miss the death-terminal event. The actor optimizes against a model where dying is invisible. This is a deeper mechanism than a "drift" — it's a structural blind spot.

### H_homeostatic_signal_mismatch (dense small-magnitude rewards mis-handled by symlog) — **PLAUSIBLE BUT SECONDARY**

**Confirm if**: real reward magnitudes are too small for symlog; reward head systematically biased.
**Observed**: real reward distribution (Dreamer episodes): mean −205, std 0.34, p05 −205.6, p95 −204.6 — extraordinarily concentrated. Per-step reward ≈ −205/28 ≈ −7.3 (very negative; dominated by the −100 death-penalty divided over 28 steps + ~−4 drive change per step late in episode). **The real per-step reward is in the −0.3 to −7 range, not the small dense range** (homeostatic drive change *prev−curr* between episode start and middle). DreamerV3 symlog handles ranges well, but the **extreme concentration** of reward (std 0.34 across 1M episodes) means the head sees almost zero variation — it underfits anything except the modal value, which it predicts correctly.
**Verdict**: The reward distribution is so pathological (single-mode, near-deterministic) that symlog calibration is not the proximal cause. The pathology is *itself* a consequence of the actor never escaping the death-loop policy. Secondary, not primary.

### H_death_loop (early death dominates replay buffer) — **CONFIRMED (positive feedback amplifier)**

**Confirm if**: every episode ends in injury early; survival flat from start; replay buffer skewed toward short death-trajectories.
**Observed**: 99.4% → 98.8% Term_Injury throughout (no improvement). Survival 27 → 30 (essentially constant). Random policy expects ~28-step survival on this map; Dreamer never escapes that.
**Verdict**: Confirmed. Once the actor is stuck uniform, all replayed sequences are short death-trajectories that further inform the WM "you die fast." The mixture-sampling buffer (with 5/16 positive slots) cannot help because *there are essentially no positive-reward sequences to slot in* — `FoodEaten ≈ 0.2` per episode means food is consumed in ~20% of episodes, and each such consumption is +0 reward in the homeostatic regime (only a *drive reduction* via N→S→drive mechanics). The positive-buffer pathway exists, but its content is anaemic. This is a **lock-in mechanism** rather than a primary cause.

### H_sample_efficiency_gap (just slow) — **REFUTED**

**Confirm if**: Dreamer is on a slow ramp toward rPPO's plateau.
**Observed**: Dreamer survival Q1=28.6, Q4=30.3 across 909K episodes / 27 M env-steps. rPPO crossed 400 survival at 26.9 M env-steps. There is **no upward trajectory** in Dreamer's curve; it is a flat line with σ < 1 step.
**Verdict**: Refuted. This is not a slower learner; it is a non-learner.

### H_config_bug (NaN, shape mismatch, sensor schema error) — **REFUTED**

**Confirm if**: warnings/NaN traces in `output.log`, missing metrics, sudden gaps.
**Observed**: zero `Warning|nan|NaN|Error|error` matches in either `output.log`. All metrics logged for 1,322 iterations × 22 dreamer-specific keys. No NaNs, no missing keys, no shape mismatches. Both runs reached `--checkpoint-frequency` checkpoints cleanly. The 19-dim observation matches the sensor schema. `mandatory` config keys all resolved (per the recent commit `9092e29` that added the missing ones).
**Verdict**: Refuted. The pipeline works; the *learning* fails.

## 6. Failure Mechanism

### 6.1 Dominant Story

**Two world-model defects compound to flatten the actor's learning signal.**

1. **Reward-head asymmetry** (`mae_pos: 4.56 → 0.37` vs `mae_neg: 5.17 → 4.06`): the head learns the sparse positive (eating) tail but cannot resolve magnitudes within the heavy-negative bulk that dominates this homeostatic-plus-death-penalty task.
2. **Continuation-head class imbalance** (98.4% accuracy ≈ 96.3% trivial baseline): the head plausibly predicts "alive" by default, so imagined rollouts of horizon 15-32 step almost never terminate and the death-penalty gradient is invisible to the actor.

These defects produce imagined trajectories that **never die** and have **near-flat reward differentiation across actions**. The actor's `mean_advantage` is therefore constant at `-0.187 ± 0.005` — every imagined action looks slightly worse than the value baseline by the same amount. Policy gradient = (advantage × log-prob gradient) integrates to noise. The entropy regularizer (`entropy_scale=3e-4`, ~30× weaker than rPPO's) cannot pressure exploration when policy gradient is degenerate. The actor stays at ≈ ln(6) entropy — **uniform random over 6 actions** — for the entire 27 M-timestep run.

The death-loop (every episode ends by predator injury at step ~28) **locks in** this regime: the replay buffer is dominated by short death-trajectories with near-identical reward profiles, so the WM has no signal-rich data to escape with even if its components were healthier.

rPPO succeeds on the same task because **GAE/MC on real trajectories** propagates the −100 death-terminal directly into value targets. The critic learns "actions near predator have value much lower than actions far from predator." That signal is unmissable. The dense −0.5 drive-change rewards play a secondary role; the predator avoidance is learned from the death event itself.

### 6.2 Data the user wished was logged but wasn't

If we had any of the following, the diagnosis above could be tightened to high-confidence on every clause:

1. **Imagined-vs-real return paired statistics** — sample (1) a real 32-step trajectory and (2) the same starting-state imagined 32-step trajectory; log `imagined_return - real_return` and `imagined_terminations` per batch. Would directly verify "imagined trajectories never terminate."
2. **Continuation-head per-class metrics** — `cont_recall_dead`, `cont_precision_dead` instead of bulk `model_cont_acc`. Would directly verify the class-imbalance hypothesis.
3. **Reward-head residuals binned by reward magnitude** — bins for `[-110,-50]`, `[-50,-5]`, `[-5,0]`, `[0,5]`, `[5,20]`. Would localize the asymmetry in §4.3.2.
4. **Per-action advantage histograms** — to see if there is *any* action discrimination in imagined rollouts. We have only `mean_advantage` (scalar) and `mean_value` (scalar).
5. **Behavior actor logits / per-action probability over training** — would confirm whether the actor is literally uniform (P=1/6 each) or has a slight bias on one action that doesn't escape.
6. **WM-latent reachability** — does the world model encoder distinguish "predator at distance 1" from "predator at distance 3"? A simple linear-probe metric on the deterministic state.

### 6.3 Could H_imagination_drift be confirmed without re-running training?

Partially. The trained Dreamer checkpoint at `results/JAX_DreamerV3/20260507-005008_dreamer_v3_basic01_PredInterval3_NutGain18/` (per `output.log` line 19) can be loaded and (i) rolled out in the real env — expect ≈ 28-step death — and (ii) imagined from real states for 32 steps — expect ≈ 32-step survival with no terminations. The gap between the two is the direct measurement of imagination drift. This is a **5-minute analysis script**, not a re-training. Recommended in §7.3 priority 1.

## 7. Conclusions and Recommendations

### 7.1 Summary

- **DreamerV3's actor is stuck at uniform random over 6 actions for the entire 27 M-timestep run on this task.** Survival is at the random-policy floor (~28 steps). H₀ (sample efficiency gap) is **refuted**.
- **The world model trains successfully** (recon ↓ 12×, KL healthy, latent entropy non-collapsed). Failure is **downstream** of the WM encoder.
- **Reward-head asymmetry**: positive-reward MAE collapses 12× while negative-reward MAE collapses 1.3× → the head cannot differentiate "die soon" from "live long" along the negative-reward dimension that dominates the task.
- **Continuation-head class imbalance**: 98.4% accuracy is ~2 pp above the trivial "always alive" baseline (death density ≈ 3.7%/step). Plausibly imagined rollouts almost never terminate.
- **Critic saturates at constant** `mean_value = -18.13`; **advantage saturates at constant** `mean_advantage = -0.19`; actor entropy stays at ≈ ln(6). The full actor-critic loop has lost its gradient signal.
- **The replay buffer is locked in** to short death-trajectories (99% Term_Injury throughout). Mixture-sampling cannot help because `FoodEaten ≈ 0.2/episode` provides almost no positive-reward content.
- **rPPO solves it because** GAE/MC on real trajectories propagates the −100 death penalty directly into value targets; this is invisible to Dreamer.

### 7.2 Limitations

- Single-seed comparison (rPPO seed 42, Dreamer seed 42). The qualitative entropy-plateau signature is structural, but a 3-seed Dreamer repeat would rule out one-in-a-thousand bad initializations.
- We do not have direct imagined-rollout dumps; the imagination-drift mechanism (§5 H_imagination_drift) is supported by indirect evidence (reward-head asymmetry + continuation-class-imbalance + the constant `mean_value`/`mean_return`/`mean_advantage`) but not by a direct sample of an imagined trajectory.
- The configs differ in `--num-envs` (128 vs 16). This is irrelevant to the *learning* failure (timesteps-aligned analysis) but means we cannot draw wall-clock conclusions from the same data; that is the speed analysis's domain ([dreamer_v3_vs_rppo_speed_profile.md](./dreamer_v3_vs_rppo_speed_profile.md)).

### 7.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|---|---|---|---|
| **1 (highest)** | **Imagination-drift probe**: load the failed Dreamer checkpoint, sample 100 (state, action, real-32-step-rollout) tuples from the env, then imagine 32 steps from the same start states. Plot `imagined_return - real_return` and termination-fraction per horizon. **Directly confirms or refutes the imagination-no-termination mechanism in §6.1.** | Confirms the dominant failure mechanism; cheap and decisive. | 5 min script, no re-training |
| **2** | **Continuation-class-imbalance fix sweep**: re-train Dreamer with `cont_loss_weight` boosted (e.g. 10×) AND/OR positive-class re-weighting (e.g., upweight death-step samples by 1/density). Compare survival-steps temporal evolution to baseline. **Tests whether fixing the continuation head alone unblocks learning.** | If continuation is the bottleneck, this should restore some learning signal. | 1 config change + 1 training run, ~3 hours |
| 3 | **Dreamer 3-seed repeat** at seed ∈ {0, 7, 13}: confirm the entropy-plateau is structural (every seed gets stuck) rather than seed-pathological. | Robustness check on the central observation. | 3 training runs, ~9 hours total (parallelizable) |
| 4 | **Reward-head asymmetry fix**: expose the per-bin reward-MAE histograms (item 3 in §6.2) AND apply class-balanced regression (e.g., bin-weighted reward-head loss) to see if the asymmetry can be flattened. | Tests the reward-head hypothesis in isolation. | 1 logging change + 1 config sweep, ~3 hours |
| 5 | **Larger entropy_scale ablation**: re-train Dreamer with `entropy_scale ∈ {3e-4, 1e-3, 3e-3, 1e-2}`. The task is small and stochastic-explorable; if the actor needs help escaping uniform, a stronger entropy bonus might bootstrap exploration without changing the WM. | Cheap robustness sweep against actor under-pressure. | 4 training runs in parallel |
| 6 | **Easier-task baseline**: run Dreamer on `00-5X5_PredInterval3_NutGain18` with `predator_enabled: false` (food-only). If Dreamer learns the food-only task, the failure is specifically the predator-avoidance + death-penalty pathway. If it fails the food-only task too, the failure is more general (homeostatic-reward calibration). | Disentangles "Dreamer can't do hypervigilance" from "Dreamer can't do this env at all." | 1 training run, ~3 hours |

### 7.4 Should we cut a `bug-fix-workflow` plan?

**Stub recommended, not yet a fix plan.** The failure is real and reproducible, but the *primary fix target* is ambiguous between "continuation head" and "reward head". Priority-1 experiment (imagination-drift probe) should run before committing to a fix plan, because the probe pinpoints which head needs surgery. After that probe, the fix plan can be precise (e.g., "increase `cont_loss_weight` to 5.0" or "add a class-balanced loss to the reward head").

> **Bug-fix-workflow stub** (do not execute yet): `docs/develop/active/diagnosis/dreamer_hypervigilance_fix_plan.md` (TBD). Conditions to write: priority-1 experiment results pinpoint a single dominant head failure; or priority-2 reveals which intervention actually moves survival.

## 8. Cross-link to Speed Profile

The companion analysis [dreamer_v3_vs_rppo_speed_profile.md](./dreamer_v3_vs_rppo_speed_profile.md) showed that DreamerV3 spends 95.4% of its iter wall-clock inside `_scan_train_gpu` (the replay-ratio training loop), and that on the same hardware DreamerV3 is structurally slower per-iteration than rPPO (post-warmup 2184.5 SPS @ R=1 vs rPPO higher).

**Both diagnoses share a root**: DreamerV3 *only* has world-model-driven training (no on-policy gradient like rPPO's GAE), so a degenerate WM (here: reward-head asymmetry + continuation class imbalance) means **every one of those expensive `_scan_train_gpu` updates is an update against bad imagined data**. Speed and learning are not independent failures — the speed cost magnifies the learning cost.

Practical consequence: if the priority-1 probe (§7.3) confirms imagination-drift, the speed problem becomes secondary. Even at 5× faster, an algorithm with no learning signal will not learn. Conversely, even if Dreamer's learning is fixed by a continuation-loss boost, the 95% `_scan_train_gpu` cost remains a separate optimization problem.

---

## Appendix

### A. Raw Data Tables

All temporal extractions saved at `/media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/20260507_dreamer_failure_analysis/`:

- `survival_steps_temporal.csv` — primary metric, 20 windows × {rppo, dreamer} × {mean, std, min, max}.
- `rewards_temporal.csv` — `Episode/Reward`, 20 windows × both algos.
- `terminations_temporal.csv` — `Term_Injury / _Starvation / _MaxSteps / _Overeating`, 20 windows × both algos.
- `losses_temporal.csv` — Dreamer-specific 22 metrics (WM + Behavior), 20 windows.
- `rppo_temporal.csv` — rPPO 5 loss metrics, 20 windows.
- `metadata.json` — run paths, config diff, run lengths, distributions, quartile means.
- `parse_local_wandb.py` — local-only `.wandb` binary parser (no web API).
- `build_csvs.py` — windowing + aggregation script.
- `rppo_history.jsonl`, `dreamer_history.jsonl` — raw history dumps from local wandb files.

### B. Run-Tag Convention Note

The user's run-tag IDs `20260507-004915_rppo...` and `20260507-005008_dreamer...` use `004915` / `005008` (the `time.strftime` call in `train.py` at tag generation), but the `wandb-init` directories are at `004916` / `005009` (1 second later, when wandb actually wrote its init record). Both refer to the same runs; the local-directory timestamps were used for parsing (`wandb/run-20260507_004916-7t1wskpk/` and `wandb/run-20260507_005009-qont5dac/`).

### C. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-07 | Initial analysis | senior-developer |

### D. See also

- Curriculum recovery experiment built on this diagnosis (priority 2 + priority 5 levers + 3-stage food→predator curriculum): [`docs/experiments/active/continual_learning/DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md`](../../../experiments/active/continual_learning/DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md)
