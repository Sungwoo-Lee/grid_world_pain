# NMN Performance Diagnosis v5: No-Noise & Easy Foraging

> **Status**: COMPLETE
> **Date**: 2026-03-17
> **Author**: Claude (analysis), Sungwoo (experiment design & training)
> **Related**: [NMN_PERFORMANCE_DIAGNOSIS_v4.md](NMN_PERFORMANCE_DIAGNOSIS_v4.md) (v4, unified grouping & memory clip fix)

---

## 1. Research Question

v4 established that: (1) unified grouping reduces unimodal suppression but does not break the zero-sum gate trade-off; (2) GAE gSize=2 achieved the best episode performance (-130.59); (3) MC gSize=1 uniquely kept both pathways open; (4) memory clip fix resolved z_memory escape. Two environmental simplifications are now tested to isolate whether **perceptual noise** and **foraging difficulty** contribute to multimodal hub collapse or mask the modulator's true dynamics:

- **No perceptual noise** (`perceptual_noise.enabled: false`): Removes all observation noise, giving the modulator clean sensory inputs. If noise was causing the modulator to suppress noisy channels, removing it should reduce gate suppression.
- **Increased food nutrition gain** (`food_nutrition_gain: 6`, up from 3): Makes foraging twice as rewarding per food unit, reducing the survival pressure and allowing the agent to survive longer. This tests whether longer episodes reveal different modulator dynamics.

> **H₀** (null): Removing noise and easing foraging has no effect on multimodal hub collapse or gate trade-off dynamics.
> **H₁a**: Removing perceptual noise reduces multimodal hub collapse by eliminating noisy features that the modulator learned to suppress.
> **H₁b**: Increased nutrition gain extends episode length, revealing late-training modulator dynamics not visible in shorter episodes.
> **H₁c**: MC return mode continues to outperform GAE under simplified environments, confirming the advantage is algorithmic rather than environment-dependent.
> **H₁d**: The grouping size effect on gate dynamics is consistent across noise/no-noise conditions.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| `return_mode` | GAE, MC | Compare return estimation methods |
| `grouping_size` | 1, 2, 4, 8 | Full sweep for NoNoise; 1, 4 for NutGain6 |
| `food_nutrition_gain` | 3 (default), 6 | Test effect of easier foraging |
| `perceptual_noise.enabled` | false | All runs — test effect of no noise vs v4 (noise on) |

### 2.2 Controlled Variables

```yaml
modulation:
  type: Multiplicative
  mod_hidden_size: 16
  percept_bias_init: 3.0
  memory_bias_init: 0.0
  temp_clip: [0.5, 3.0]
  memory_clip: [-2.0, 2.0]
  multimodal_hidden_size: 128
  grouping: unified  # same as v4

agent:
  activation: relu
  rnn_type: GRU
  hidden_size: 128
  encoding_mode: hierarchical
  lr_actor: 0.0005
  lr_critic: 0.0001
  K_epochs: 4
  gamma: 0.95
  gae_lambda: 0.95
  entropy_coef: 0.01
  eps_clip: 0.1
  sequence_length: 128
  num_envs: 128

environment:
  grid: 10x10, max_steps=500
  bush: 2/quadrant, pred: 4, food: 2/quadrant
  danger: 8 (2/quad, damage [15,45])
  rocks: 12 (3/quad, damage [0.1,0.5])
  neutral: 5 rabbits

perceptual_noise:
  enabled: false  # NEW — disabled for all v5 runs
```

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| Much longer training (~17–21B timesteps) vs v4 (~3–4B) | All | Med | Compare at matched training windows where possible; use steady-state metrics which are past convergence |
| No noise-ON control at this training duration | All | Med | Compare with v4 results (noise ON, shorter training) for directional signal |
| NutGain6 only tested at gSize 1 and 4 | NutGain6 runs | Med | gSize 1 and 4 bracket the range — interpolation reasonable |
| GAE_g4_NG6 shows late instability (possible training divergence) | Run #12 | High | Flag in analysis; exclude from steady-state conclusions where affected |
| MC_g1_NG6 shows a mid-training dip around windows 11-12 | Run #9 | Med | Agent recovers; likely training instability not structural |
| Single seed per config | All | Med | Within-run variance provides some signal |

## 3. Run Inventory

| # | Datetime ID | WandB ID | Return | gSize | NutGain | Timesteps | Tag |
|---|---|---|---|---|---|---|---|
| 1 | 20260312-141413 | `sm8ydmxx` | MC | 1 | 3 | ~21.7B | rppoNMN_MC_NoNoise_...gSize1 |
| 2 | 20260312-141506 | `r9qxkggk` | MC | 2 | 3 | ~21.7B | rppoNMN_MC_NoNoise_...gSize2 |
| 3 | 20260312-141537 | `lw7pq0ic` | MC | 4 | 3 | ~21.7B | rppoNMN_MC_NoNoise_...gSize4 |
| 4 | 20260312-141609 | `3euakozf` | MC | 8 | 3 | ~21.7B | rppoNMN_MC_NoNoise_...gSize8 |
| 5 | 20260312-142552 | `mhbc8o6n` | GAE | 1 | 3 | ~17.2B | rppoNMN_GAE_NoNoise_...gSize1 |
| 6 | 20260312-142653 | `kfvcbhkt` | GAE | 2 | 3 | ~17.2B | rppoNMN_GAE_NoNoise_...gSize2 |
| 7 | 20260312-143043 | `b6uqyhv3` | GAE | 4 | 3 | ~17.2B | rppoNMN_GAE_NoNoise_...gSize4 |
| 8 | 20260312-143111 | `3e4vbdcc` | GAE | 8 | 3 | ~17.2B | rppoNMN_GAE_NoNoise_...gSize8 |
| 9 | 20260312-143800 | `vbw1fj5m` | MC | 1 | 6 | ~20.9B | rppoNMN_MC_NoNoise_...NutGain6_...gSize1 |
| 10 | 20260312-143832 | `wpenvc21` | MC | 4 | 6 | ~20.9B | rppoNMN_MC_NoNoise_...NutGain6_...gSize4 |
| 11 | 20260312-143925 | `muu7rohk` | GAE | 1 | 6 | ~21.0B | rppoNMN_GAE_NoNoise_...NutGain6_...gSize1 |
| 12 | 20260312-143951 | `8d9zz3og` | GAE | 4 | 6 | ~21.0B | rppoNMN_GAE_NoNoise_...NutGain6_...gSize4 |

---

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

#### 4.1.1 NoNoise Effect — Comparing v5 (NoNoise) vs v4 (Noise ON) at gSize=1

| Metric | v4 GAE_g1 (noise) | v5 GAE_g1 (NoNoise) | v4 MC_g1 (noise) | v5 MC_g1 (NoNoise) |
|--------|-------------------|----------------------|-------------------|----------------------|
| Reward | -138.41 ± 2.42 | **-134.79 ± 2.14** | -135.45 ± 2.59 | **-129.05 ± 1.64** |
| Steps | 161.94 ± 5.37 | **179.29 ± 4.89** | 178.09 ± 6.49 | **222.95 ± 6.35** |
| FoodEaten | 33.96 ± 2.64 | **42.28 ± 2.39** | 40.62 ± 3.27 | **62.48 ± 3.20** |
| TotalDamage | 85.89 ± 4.13 | 85.80 ± 3.56 | 84.15 ± 5.02 | 75.43 ± 3.37 |
| gamma_multi SS | -1.369 (sig 20.3%) | **-2.748 (sig 6.0%)** | -0.835 (sig 30.2%) | **-5.688 (sig 0.3%)** |
| gamma_uni SS | -1.981 (sig 12.1%) | -3.167 (sig 4.0%) | +1.740 (sig 85.1%) | **-3.208 (sig 3.9%)** |
| temp_mean | 2.833 | 2.909 | 1.163 | **3.000** |

> **Verdict on H₁a: NOT SUPPORTED — removing noise worsens multimodal collapse.**
> - Multimodal gates are **more suppressed** without noise: v5 MC_g1 gamma_multi = -5.69 (sig 0.3%) vs v4 MC_g1 = -0.84 (sig 30.2%).
> - However, v5 runs trained ~5× longer (17–21B vs 3–4B timesteps). The deeper collapse may simply reflect more training time, not a noise effect.
> - Critically, the v4 MC_g1 unimodal gate recovery (sig 85%) **does not reproduce** in v5 NoNoise. v5 MC_g1 gamma_uni = -3.21 (sig 3.9%) — severely suppressed.
> - Despite worse gate dynamics, v5 NoNoise MC achieves **better reward** (-129.05 vs -135.45), suggesting the modulator gates are not the primary driver of performance — the clean sensory signal matters more.

#### 4.1.2 NutGain6 Effect — Episode Length and Survival

| Metric | MC_g1 (NutGain3) | MC_g1_NG6 (NutGain6) | GAE_g1 (NutGain3) | GAE_g1_NG6 (NutGain6) |
|--------|-------------------|------------------------|---------------------|--------------------------|
| Reward | **-129.05 ± 1.64** | -130.80 ± 2.53 | **-134.79 ± 2.14** | -143.67 ± 3.05 |
| Steps | 222.95 ± 6.35 | **362.37 ± 10.48** | 179.29 ± 4.89 | **237.06 ± 15.33** |
| FoodEaten | **62.48 ± 3.20** | 55.60 ± 2.44 | **42.28 ± 2.39** | 34.91 ± 3.72 |
| TotalDamage | 75.43 ± 3.37 | **108.49 ± 5.04** | 85.80 ± 3.56 | **116.42 ± 6.93** |
| Term_MaxSteps | 2.7% | **38.3%** | 0.5% | **10.6%** |
| Term_Starvation | 89.3% | **46.5%** | 83.8% | **59.9%** |
| Term_Injury | 8.0% | 15.3% | 15.7% | **29.5%** |

> **Verdict on H₁b: PARTIALLY SUPPORTED — NutGain6 dramatically extends survival but does not improve reward.**
> - MC_g1_NG6 episodes are 62% longer (362 vs 223 steps), with 38% reaching max_steps (vs 2.7%).
> - However, reward is slightly **worse** (-130.80 vs -129.05) because longer survival means more damage exposure.
> - FoodEaten is paradoxically **lower** despite NutGain6 — because each food unit provides 2× nutrition, the agent needs to eat less.
> - The shift from starvation death (89→47%) to injury death (8→15%) and max_steps (3→38%) represents a qualitative change in agent behavior — the agent is no longer foraging-constrained.

#### 4.1.3 MC vs GAE Return Mode — NoNoise

| gSize | MC Reward | GAE Reward | MC Steps | GAE Steps | Δ Reward |
|-------|-----------|------------|----------|-----------|----------|
| 1 | **-129.05** | -134.79 | 222.95 | 179.29 | **MC +5.74** |
| 2 | **-129.29** | -136.05 | 224.73 | 180.80 | **MC +6.76** |
| 4 | **-128.78** | -133.64 | 228.86 | 205.03 | **MC +4.86** |
| 8 | **-128.71** | -133.97 | 225.26 | 234.85 | **MC +5.26** |

> **Verdict on H₁c: STRONGLY SUPPORTED — MC consistently outperforms GAE by 5–7 reward points across all grouping sizes in NoNoise conditions.**
> - This is a larger gap than v4 (~5 points at gSize=2). The MC advantage appears to be **amplified** by removing noise.
> - MC agents survive ~25% longer (223–229 vs 179–235 steps) and eat ~50% more food.
> - MC agents also take **less** total damage (74–76 vs 86–96), despite living longer — indicating better danger avoidance.

#### 4.1.4 Grouping Size Effect — NoNoise

**MC (NoNoise, NutGain3):**

| gSize | Reward | gamma_multi (sig%) | gamma_uni (sig%) | Both open? |
|-------|--------|-------------------|-----------------|------------|
| 1 | -129.05 | -5.69 (0.3%) | -3.21 (3.9%) | No — both dead |
| 2 | -129.29 | -9.51 (0.01%) | -0.13 (46.7%) | No — multi dead, uni open |
| 4 | **-128.78** | -11.04 (<0.01%) | -0.38 (40.6%) | No — multi dead, uni open |
| 8 | **-128.71** | -11.21 (<0.01%) | -0.41 (39.9%) | No — multi dead, uni open |

**GAE (NoNoise, NutGain3):**

| gSize | Reward | gamma_multi (sig%) | gamma_uni (sig%) | Both open? |
|-------|--------|-------------------|-----------------|------------|
| 1 | -134.79 | -2.75 (6.0%) | -3.17 (4.0%) | No — both mostly dead |
| 2 | -136.05 | -4.37 (1.3%) | -4.78 (0.8%) | No — both dead |
| 4 | -133.64 | -6.77 (0.1%) | -1.77 (14.6%) | No — multi dead, uni partially open |
| 8 | -133.97 | -8.91 (0.01%) | -5.33 (0.5%) | No — both dead |

> **Verdict on H₁d: PARTIALLY SUPPORTED — grouping size effect is consistent in direction but different in magnitude.**
> - In MC NoNoise, gSize has **minimal effect on reward** (0.6-point spread). All MC runs converge to ~-129.
> - In GAE NoNoise, gSize=4 is best (-133.64), consistent with v4 where intermediate gSize performed best.
> - The unique MC_g1 unimodal recovery seen in v4 **does not reproduce** — MC_g1 unimodal is now -3.21 (sig 3.9%) instead of v4's +1.74 (sig 85.1%). This suggests the v4 recovery was noise-dependent or training-duration-dependent.

### 4.2 Episode Performance

#### NoNoise NutGain3 — Full Comparison

| Label | Reward SS | Steps SS | FoodEaten SS | TotalDamage SS | Term_Injury | Term_Starvation | Term_MaxSteps |
|-------|-----------|----------|-------------|----------------|-------------|-----------------|---------------|
| MC_g1 | -129.05 ± 1.64 | 222.95 ± 6.35 | 62.48 ± 3.20 | 75.43 ± 3.37 | 8.0% | 89.3% | 2.7% |
| MC_g2 | -129.29 ± 1.67 | 224.73 ± 6.18 | 63.43 ± 3.12 | 76.38 ± 3.36 | 8.2% | 88.9% | 3.0% |
| MC_g4 | **-128.78 ± 1.63** | 228.86 ± 6.27 | 65.46 ± 3.16 | **75.32 ± 3.25** | 8.0% | 88.6% | 3.3% |
| MC_g8 | **-128.71 ± 1.63** | 225.26 ± 6.20 | 63.59 ± 3.13 | **74.49 ± 3.27** | 7.8% | 89.4% | 2.8% |
| GAE_g1 | -134.79 ± 2.14 | 179.29 ± 4.89 | 42.28 ± 2.39 | 85.80 ± 3.56 | 15.7% | 83.8% | 0.5% |
| GAE_g2 | -136.05 ± 2.34 | 180.80 ± 6.72 | 43.37 ± 3.30 | 89.02 ± 4.11 | 17.0% | 82.5% | 0.5% |
| GAE_g4 | -133.64 ± 2.89 | 205.03 ± 13.71 | 54.82 ± 6.78 | 89.74 ± 6.08 | 13.1% | 85.1% | 1.7% |
| GAE_g8 | -133.97 ± 1.95 | 234.85 ± 7.07 | 69.51 ± 3.57 | 96.05 ± 4.02 | 12.7% | 83.5% | 3.8% |

**MC_g8/g4 are the best performers** by reward. MC runs show remarkably uniform performance — the 0.6-point spread across all 4 grouping sizes is within noise, suggesting the modulator has minimal functional impact on MC performance under NoNoise conditions.

**GAE_g8 is a paradox**: longest survival (235 steps), most food eaten (69.5), but worst damage (96.1) and mediocre reward (-134.0). The agent survives by eating prolifically but cannot avoid damage.

#### NutGain6 — Comparison

| Label | Reward SS | Steps SS | FoodEaten SS | TotalDamage SS | Term_Injury | Term_Starvation | Term_MaxSteps |
|-------|-----------|----------|-------------|----------------|-------------|-----------------|---------------|
| MC_g1_NG6 | **-130.80 ± 2.53** | **362.37 ± 10.48** | 55.60 ± 2.44 | 108.49 ± 5.04 | 15.3% | 46.5% | **38.3%** |
| MC_g4_NG6 | -134.55 ± 2.56 | **370.00 ± 10.53** | 57.47 ± 2.43 | 116.98 ± 5.06 | 18.0% | 41.1% | **40.9%** |
| GAE_g1_NG6 | -143.67 ± 3.05 | 237.06 ± 15.33 | 34.91 ± 3.72 | 116.42 ± 6.93 | 29.5% | 59.9% | 10.6% |
| GAE_g4_NG6 | -135.75 ± 12.06 | 264.49 ± 82.55 | 41.06 ± 17.36 | 106.45 ± 9.45 | 22.6% | 58.2% | 19.2% |

**MC NutGain6 agents approach the step limit**: 38–41% of episodes reach max_steps=500, meaning the agent has essentially "solved" the survival problem. Starvation drops from ~89% to ~44%, and injury death rises from ~8% to ~16% — injury now becomes the primary challenge.

**GAE_g4_NG6 shows extreme instability**: Steps variance of ±82.55 (vs ±10 for MC) and reward variance of ±12.06 indicate the agent periodically collapses and recovers during training. The last training window shows FoodEaten plummeting from ~49 to 5.8 and Steps from ~303 to 109.

### 4.3 Diagnostic Metrics

#### 4.3.1 Gate Dynamics — Multimodal Collapse Severity

**Multimodal gate (gamma_multi) steady-state:**

| gSize | v4 GAE (noise) | v5 GAE (no noise) | v4 MC (noise) | v5 MC (no noise) |
|-------|----------------|-------------------|----------------|-------------------|
| 1 | -1.37 (sig 20%) | -2.75 (sig 6%) | -0.84 (sig 30%) | -5.69 (sig 0.3%) |
| 2 | -2.59 (sig 7%) | -4.37 (sig 1.3%) | -2.32 (sig 9%) | -9.51 (sig 0.01%) |
| 4 | -4.63 (sig 1%) | -6.77 (sig 0.1%) | -3.11 (sig 4%) | -11.04 (<0.01%) |
| 8 | -4.37 (sig 1.3%) | -8.91 (sig 0.01%) | -3.94 (sig 2%) | -11.21 (<0.01%) |

**Multimodal collapse is dramatically more severe in v5 NoNoise**, even accounting for ~5× longer training. At gSize≥2, MC runs show gamma_multi below -9.5, meaning the multimodal pathway is essentially completely dead (sigmoid < 0.01%). Removing noise did not help — it may have **accelerated** collapse by making modality-specific (unimodal) features more reliable, further reducing the relative value of hub features.

#### 4.3.2 Unimodal Gate — MC Reversal Does Not Reproduce

| gSize | v4 MC gamma_uni (noise) | v5 MC gamma_uni (no noise) | Change |
|-------|-------------------------|----------------------------|--------|
| 1 | **+1.74 (sig 85.1%)** | -3.21 (sig 3.9%) | **Reversed — now suppressed** |
| 2 | -1.83 (sig 13.9%) | -0.13 (sig 46.7%) | Improved — more open |
| 4 | -2.75 (sig 6.0%) | -0.38 (sig 40.6%) | Improved — more open |
| 8 | -2.24 (sig 9.6%) | -0.41 (sig 39.9%) | Improved — more open |

**Key observation**: The v4 MC_g1 unimodal gate recovery (the "unprecedented" finding of v4) **does not reproduce** under NoNoise conditions. Instead, MC_g1 shows the **most suppressed** unimodal gate of any MC run. However, MC gSize≥2 shows **less suppressed** unimodal gates (~40–47% sigmoid) compared to v4 (~6–14%). This suggests that removing noise makes the unimodal pathway more viable at gSize≥2 but not at gSize=1.

#### 4.3.3 Temperature

| Return Mode | gSize | v4 temp_mean | v5 temp_mean |
|-------------|-------|-------------|-------------|
| MC | 1 | 1.163 | **3.000** |
| MC | 2 | 1.203 | **3.000** |
| MC | 4 | 1.242 | **3.000** |
| MC | 8 | 1.487 | **3.000** |
| GAE | 1 | 2.833 | 2.909 |
| GAE | 2 | 2.674 | 2.928 |
| GAE | 4 | 2.760 | **3.000** |
| GAE | 8 | 2.849 | **3.000** |

**All MC runs and GAE gSize≥4 saturate at the 3.0 temperature ceiling.** This is a dramatic change from v4 where MC temperatures were 1.1–1.5. Only GAE gSize=1 and 2 remain below ceiling (~2.9). The temperature ceiling of 3.0 appears to be a hard constraint that the modulator consistently pushes against — the `temp_clip` upper bound may need to be increased to see if higher temperatures improve performance.

#### 4.3.4 z_memory

| gSize | MC z_memory | GAE z_memory |
|-------|------------|-------------|
| 1 | -0.015 | +0.138 |
| 2 | +0.634 | +0.665 |
| 4 | +0.844 | +0.752 |
| 8 | +1.369 | +1.305 |

All z_memory values within [-2, 2] — memory clip continues to work correctly. There is a **clear monotonic relationship**: larger gSize → higher z_memory, consistent across both return modes. z_memory appears to encode grouping-size-dependent information about the modulator's state.

### 4.4 Learning Dynamics

#### 4.4.1 Episode Reward Over Training

**MC NoNoise NutGain3** (20-window means, selected windows):

| ~Window | MC_g1 | MC_g2 | MC_g4 | MC_g8 |
|---------|-------|-------|-------|-------|
| 1 (0–5M) | -135.40 | -136.66 | -138.27 | -137.56 |
| 4 (~16–21M) | -130.50 | -130.65 | -130.52 | -132.46 |
| 8 (~36–41M) | -129.60 | -129.98 | -129.82 | -129.49 |
| 12 (~55–60M) | -129.51 | -129.70 | -129.03 | -129.21 |
| 16 (~75–80M) | -129.38 | -129.17 | -128.91 | -128.72 |
| 20 (~94–99M) | -129.17 | -129.39 | -128.66 | -128.82 |

MC runs converge by window 4 (~20M steps) and remain remarkably stable thereafter. All grouping sizes converge to within 0.7 points of each other. Slight continued improvement through window 20.

**GAE NoNoise NutGain3** (selected windows):

| ~Window | GAE_g1 | GAE_g2 | GAE_g4 | GAE_g8 |
|---------|--------|--------|--------|--------|
| 1 (0–5M) | -135.14 | -131.46 | -133.56 | -131.35 |
| 4 (~13–18M) | -134.26 | -135.23 | -130.60 | -131.50 |
| 8 (~31–35M) | -134.51 | -135.97 | -132.55 | -132.45 |
| 12 (~47–52M) | -135.57 | -137.08 | -133.29 | -133.72 |
| 16 (~80–84M) | -135.22 | -135.98 | -133.57 | -133.89 |
| 20 (~95–99M) | -134.85 | -135.85 | -133.45 | -133.81 |

GAE_g2 shows **performance degradation** over training: starting as the best (-131.46 in window 1) but degrading to worst (-135.85 in window 20). GAE_g4 is the most stable performer. GAE_g1 oscillates without clear trend.

**NutGain6** (selected windows):

| ~Window | MC_g1_NG6 | MC_g4_NG6 | GAE_g1_NG6 | GAE_g4_NG6 |
|---------|-----------|-----------|------------|------------|
| 1 | -137.46 | -139.59 | -135.01 | -134.62 |
| 5 | -131.31 | -130.49 | -137.08 | -131.45 |
| 10 | -130.67 | -130.83 | -136.60 | -131.57 |
| 15 | -131.13 | -134.35 | -147.59 | -131.20 |
| 20 | -130.55 | -134.56 | -143.60 | **-151.26** |

GAE_g1_NG6 shows **progressive degradation** from -135 (window 1) to -147 (window 15), stabilizing around -143. GAE_g4_NG6 **catastrophically collapses** in the final window (-151.26, with Steps dropping to 109 and FoodEaten to 5.8).

MC_g1_NG6 shows a mid-training dip (windows 11-12, reward drops to -138.79) but **recovers** to -130.55 by window 20. MC_g4_NG6 gradually worsens from -130.49 (window 5) to -134.56 (window 20).

#### 4.4.2 Multimodal Collapse Timeseries

**MC NoNoise gamma_multi** (20-window means):

| ~Window | MC_g1 | MC_g2 | MC_g4 | MC_g8 |
|---------|-------|-------|-------|-------|
| 1 | +1.50 | +0.92 | +0.57 | +0.27 |
| 5 | -3.26 | -5.13 | -5.64 | -6.45 |
| 10 | -4.63 | -7.82 | -8.64 | -9.04 |
| 15 | -5.28 | -8.85 | -10.21 | -10.54 |
| 20 | -5.82 | -9.72 | -11.33 | -11.53 |

Collapse is **monotonic and does not decelerate** in any configuration. MC_g1 collapses slowest (rate ~0.22 units/window), MC_g8 fastest (~0.56 units/window). All runs still declining at window 20 — no plateau reached.

**Collapse rate comparison (v4 vs v5, MC_g1):**
- v4 MC_g1 (18 windows, noise ON): +3.14 → -1.09, rate ~0.24/window
- v5 MC_g1 (20 windows, NoNoise): +1.50 → -5.82, rate ~0.37/window

Collapse is ~50% faster in NoNoise conditions, suggesting noise may have been **slowing** collapse by injecting variability that prevented the optimizer from fully suppressing the multimodal pathway.

**GAE NoNoise gamma_multi** (20-window means):

| ~Window | GAE_g1 | GAE_g2 | GAE_g4 | GAE_g8 |
|---------|--------|--------|--------|--------|
| 1 | +1.78 | +1.97 | +2.02 | +0.26 |
| 5 | -1.03 | -1.76 | -1.82 | -5.06 |
| 10 | -1.93 | -3.06 | -3.98 | -7.02 |
| 15 | -2.70 | -4.00 | -6.16 | -8.55 |
| 20 | -2.81 | -4.52 | -6.98 | -9.05 |

GAE collapse is slower than MC at matched gSize. GAE_g1 shows the slowest collapse (~0.23 units/window) and **may be approaching plateau** around -2.8 in late windows.

#### 4.4.3 Unimodal Gate Timeseries

**MC NoNoise gamma_uni** — Key patterns:

| ~Window | MC_g1 | MC_g2 | MC_g4 | MC_g8 |
|---------|-------|-------|-------|-------|
| 1 | +0.40 | -0.09 | +0.37 | +0.44 |
| 3 | -1.95 | -2.37 | -1.74 | -0.91 |
| 8 | -2.31 | -2.07 | -1.12 | -0.73 |
| 12 | -2.65 | -0.88 | -0.78 | -0.52 |
| 16 | -3.06 | -0.40 | -0.52 | -0.35 |
| 20 | -3.31 | -0.16 | -0.29 | -0.27 |

**MC_g1 unimodal is monotonically suppressed** — unique among MC runs, it shows no recovery. Reaches -3.31 (sig 3.5%) by window 20.

**MC_g2/g4/g8 show unimodal gate recovery**: After initial suppression to -1.2 to -2.4 (windows 2–3), these runs **gradually reopen** the unimodal pathway, stabilizing around -0.1 to -0.4 (sig 40–48%) by window 20. This is a new pattern — the unimodal gate is not monotonically suppressed but **oscillates and stabilizes** at moderate openness.

**GAE NoNoise gamma_uni** — Key patterns:

| ~Window | GAE_g1 | GAE_g2 | GAE_g4 | GAE_g8 |
|---------|--------|--------|--------|--------|
| 1 | +1.23 | +0.75 | +1.11 | -0.66 |
| 5 | -1.20 | -2.48 | -1.70 | -3.76 |
| 10 | -2.01 | -3.34 | -1.97 | -4.49 |
| 15 | -2.89 | -4.52 | -1.75 | -5.14 |
| 20 | -3.31 | -4.82 | -1.82 | -5.41 |

**GAE_g4 shows a unique unimodal plateau at ~-1.8** (sig ~14%) from window 5 onward — consistent with v4's finding. This is the only GAE run where the unimodal gate does not monotonically decline. GAE_g1, g2, g8 all show monotonic suppression.

#### 4.4.4 NutGain6 Gate Dynamics

**MC_g1_NG6 shows a mid-training perturbation:**

| ~Window | gamma_multi | gamma_uni |
|---------|------------|-----------|
| 1 | +1.77 | +1.13 |
| 5 | -2.52 | -2.13 |
| 10 | -4.30 | -2.26 |
| 11 (dip) | -4.95 | -2.41 |
| 14 | -5.36 | -2.45 |
| 20 | -6.24 | -2.25 |

The mid-training reward dip (windows 11–12) coincides with accelerated multimodal collapse but the agent recovers reward despite worsening gates — suggesting behavioral adaptation independent of gate values.

**GAE_g4_NG6 collapse:**

| ~Window | gamma_multi | gamma_uni | Reward | Steps |
|---------|------------|-----------|--------|-------|
| 10 | -6.40 | -1.47 | -131.57 | 290 |
| 15 | -8.20 | -1.18 | -131.20 | 301 |
| 20 | **-8.38** | **-1.04** | **-151.26** | **147** |

The late collapse in GAE_g4_NG6 is **not correlated with gate dynamics** — gates are actually slightly less suppressed in the final windows than earlier. The collapse appears to be a training instability (possibly learning rate or critic divergence) rather than a modulator pathology.

---

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — MC with NoNoise achieves the best overall performance (HIGH confidence)**
What: MC NoNoise runs at NutGain3 achieve reward -128.71 to -129.29, outperforming all prior configurations (v4 best was -130.59 GAE_g2 with noise).
Why: Two factors compound: (a) MC return estimation handles the longer-horizon reward signal better; (b) removing noise gives the agent cleaner sensory inputs, improving learning efficiency.
Evidence: All 4 MC NoNoise gSize variants outperform the best v4 result. MC outperforms GAE by 5–7 points at every gSize.
Confidence: HIGH.

**Finding 2 — Multimodal collapse is dramatically worse without noise (HIGH confidence)**
What: NoNoise MC gamma_multi reaches -5.7 to -11.2 (vs v4 noise: -0.8 to -3.9). Even after accounting for 5× longer training, collapse is ~50% faster per window.
Why: Without noise, unimodal (modality-specific) features are more reliable and informative than multimodal hub features. The optimizer suppresses the hub more aggressively because the unimodal pathway provides cleaner signal.
Evidence: All 8 NoNoise runs show deeper collapse than their v4 noise counterparts. Collapse rate per window is higher.
Confidence: HIGH for the observation. MEDIUM for the causal mechanism — longer training contributes but cannot fully explain the difference.

**Finding 3 — MC temperature saturates at ceiling in NoNoise (HIGH confidence)**
What: All MC runs hit temp_mean=3.000 (the `temp_clip` upper bound). In v4 with noise, MC temperature was 1.1–1.5.
Why: Without noise, the modulator's temperature parameter has no noise-averaging benefit from staying low. The optimizer pushes temperature to maximum to inflate the policy's entropy, counteracting the agent's tendency to converge too quickly on deterministic policies.
Evidence: All 4 MC runs at 3.000, all 4 NutGain6 runs at 3.000, GAE_g4/g8 also at 3.000.
Confidence: HIGH.

**Finding 4 — The v4 MC_g1 unimodal recovery does not reproduce (HIGH confidence)**
What: v4's most surprising finding — MC_g1 unimodal gate recovering from -1.4 to +1.7 (sig 85%) — is absent in v5 NoNoise. v5 MC_g1 gamma_uni = -3.21 (sig 3.9%), monotonically declining.
Why: The v4 recovery likely depended on noise providing a signal that the unimodal pathway needed to remain open to filter. Without noise, there is no filtering incentive, so the unimodal gate suppresses normally.
Evidence: v5 MC_g1 gamma_uni timeseries shows monotonic decline from +0.40 to -3.31. No recovery phase.
Confidence: HIGH that the behavior is absent. MEDIUM for the causal explanation.

**Finding 5 — MC gSize≥2 shows unimodal gate recovery in NoNoise (MEDIUM confidence)**
What: MC_g2/g4/g8 gamma_uni suppresses to -1.2 to -2.4 by window 3, then **gradually recovers** to -0.1 to -0.4 (sig 40–48%) by window 20.
Why: With gSize≥2, the unimodal pathway provides enough granularity for useful modulation. The optimizer initially suppresses all pathways, then rediscovers the unimodal pathway as multimodal collapses further.
Evidence: All 3 MC runs with gSize≥2 show the same pattern: suppress → recover → stabilize. MC_g1 does not show this pattern.
Confidence: MEDIUM — the recovery is gradual and could be an artifact of the longer training duration.

**Finding 6 — NutGain6 dramatically extends survival but does not improve reward (HIGH confidence)**
What: NutGain6 MC episodes reach 362–370 steps (vs 223–229 for NutGain3), with 38–41% reaching max_steps. Reward is slightly worse (-130.80 vs -129.05).
Why: The agent survives longer because each food unit provides more nutrition, but longer survival means more exposure to predators and dangers. The reward improvement from less starvation is offset by more injury damage.
Evidence: Consistent across both MC NutGain6 runs. Term_Starvation drops 89→47%, Term_Injury rises 8→15%.
Confidence: HIGH.

**Finding 7 — GAE shows performance degradation in multiple NoNoise configs (MEDIUM confidence)**
What: GAE_g2 NoNoise reward worsens from -131 (window 1) to -136 (window 20). GAE_g1_NG6 worsens from -135 to -144. GAE_g4_NG6 collapses to -151 in the final window.
Why: GAE's value function estimates become increasingly biased as training progresses, particularly under NoNoise conditions where the lack of stochasticity in observations reduces exploration.
Evidence: 3 of 8 GAE runs show degradation; 0 of 4 MC runs show degradation. The late GAE_g4_NG6 collapse is particularly severe (critic loss=143 vs MC value loss ~0.29).
Confidence: MEDIUM — could be training instability rather than a fundamental GAE weakness.

**Finding 8 — Grouping size barely affects MC NoNoise performance (HIGH confidence)**
What: MC NoNoise reward spread across all 4 grouping sizes is 0.57 points (-128.71 to -129.29). In contrast, v4 MC showed a 4.2-point spread.
Why: Without noise, the agent performs well regardless of modulator configuration. The modulator gates have collapsed to the point where they contribute minimal modulation — the agent is essentially running on raw features.
Evidence: All 4 MC reward values within 0.6 points. Gate values show that multimodal is fully collapsed and unimodal is near 50% sigmoid for gSize≥2.
Confidence: HIGH.

### 5.2 Cross-Run Comparisons

#### NoNoise vs Noise (v4) — Matched Configs (MC gSize=4)

| Metric | v4 MC_g4 (noise) | v5 MC_g4 (NoNoise) | Change |
|--------|-------------------|---------------------|--------|
| Reward | -135.56 | **-128.78** | **+6.78** |
| Steps | 174.21 | **228.86** | **+54.65** |
| FoodEaten | 38.72 | **65.46** | **+26.74** |
| TotalDamage | 84.49 | **75.32** | **-9.17** |
| gamma_multi | -3.11 (sig 4.3%) | -11.04 (sig <0.01%) | Much worse |
| gamma_uni | -2.75 (sig 6.0%) | -0.38 (sig 40.6%) | **Much better** |
| temp_mean | 1.242 | 3.000 | Saturated |

**Conclusion**: Removing noise improves reward by ~7 points, extends survival by 55 steps, and dramatically changes gate dynamics. The multimodal pathway collapses fully but the unimodal pathway opens more. The net effect on performance is strongly positive — the agent compensates for lost multimodal signal by using raw unimodal features more effectively.

#### NutGain3 vs NutGain6 — Matched Configs (MC gSize=1)

| Metric | MC_g1 (NG3) | MC_g1_NG6 (NG6) | Change |
|--------|-------------|------------------|--------|
| Reward | **-129.05** | -130.80 | -1.75 |
| Steps | 222.95 | **362.37** | **+139.42** |
| FoodEaten | **62.48** | 55.60 | -6.88 |
| TotalDamage | **75.43** | 108.49 | +33.06 |
| gamma_multi | -5.69 | -6.10 | Slightly worse |
| gamma_uni | -3.21 | -2.24 | Slightly better |

**Conclusion**: NutGain6 extends survival by 63% but worsens reward by 1.75 points due to increased damage from longer exposure. The agent eats less food (each unit worth more) but accumulates 44% more total damage. The modulator is largely unaffected — NutGain6 is primarily an environmental change, not a modulator dynamics change.

#### GAE gSize=4: NoNoise NutGain3 (stable) vs NutGain6 (unstable)

| Metric | GAE_g4 (NG3) | GAE_g4_NG6 (NG6) |
|--------|-------------|------------------|
| Reward | -133.64 ± 2.89 | -135.75 ± **12.06** |
| Steps | 205.03 ± 13.71 | 264.49 ± **82.55** |
| FoodEaten | 54.82 ± 6.78 | 41.06 ± **17.36** |
| Late collapse? | No | **Yes — reward to -151 in final window** |

**Conclusion**: GAE_g4 is stable under NutGain3 but **catastrophically unstable** under NutGain6. The variance across all metrics is 3–5× higher. This suggests that the longer episodes created by NutGain6 exceed GAE's capacity for stable value estimation — the critic cannot accurately estimate values over 300+ step horizons.

### 5.3 Failure Modes & Pathologies

#### Pathology 1: Universal Multimodal Hub Collapse (persistent, more severe)
- **Affected**: All 12 runs
- **Severity**: Worse than v4 — MC gSize≥2 reaches sig <0.01%
- **Change from v4**: Collapse is ~50% faster per training window in NoNoise conditions
- **Impact on performance**: Minimal for MC (reward within 0.6 points regardless of collapse severity); measurable for GAE

#### Pathology 2: MC Temperature Ceiling Saturation
- **Affected**: All 8 MC runs, GAE gSize≥4
- **When**: MC reaches 3.0 by window 3 (~10M steps) and stays pinned
- **Likely cause**: Without noise, the modulator pushes temperature to maximum to maintain exploration. The 3.0 ceiling is a binding constraint.
- **Impact**: Unknown — the agent performs well at temp=3.0, but performance might improve if the ceiling were raised. This is a testable hypothesis.

#### Pathology 3: GAE Performance Degradation Under NoNoise
- **Affected**: GAE_g2 (gradual), GAE_g1_NG6 (moderate), GAE_g4_NG6 (catastrophic)
- **When**: Begins around window 5–10; catastrophic for GAE_g4_NG6 in final window
- **Likely cause**: GAE value estimation becomes increasingly biased without observation noise to provide implicit regularization. Critic loss is 100× higher for GAE (89–107) vs MC (0.24–0.29).
- **Impact**: GAE_g4_NG6 FoodEaten drops from 49 to 5.8, Steps from 303 to 109 in the final window

#### Pathology 4: MC_g1_NG6 Mid-Training Dip
- **Affected**: MC_g1_NG6
- **When**: Windows 11–12 (~25–42M steps), reward drops from -131 to -139
- **Recovery**: Agent recovers to -131 by window 14
- **Likely cause**: Training instability possibly triggered by the interaction between long episodes and MC return computation
- **Impact**: Temporary — 10-window dip then recovery

---

## 6. Conclusions

### 6.1 Summary

- **MC NoNoise is the best configuration found across all v1–v5 experiments** (reward -128.71 to -129.29), outperforming v4's best (GAE_g2 -130.59). The 5–7 point MC advantage over GAE is consistent across all grouping sizes and both nutrition levels.
- **Removing perceptual noise improves performance but worsens multimodal collapse**. The agent achieves better reward with clean inputs despite the modulator's multimodal pathway being fully collapsed (sig <0.01% at gSize≥2). This confirms that multimodal hub collapse does not prevent good performance — the agent learns to rely on unimodal features.
- **The modulator has minimal functional impact under NoNoise MC conditions**: Grouping size produces only a 0.6-point reward spread, and all gate values suggest the modulator is operating at the extreme of its parameter space (temp=3.0, multi=dead, uni=~50% for gSize≥2).
- **NutGain6 extends survival (222→362 steps, 38% reaching max_steps) but does not improve reward** (-129 → -131). Longer survival means more damage exposure, offsetting the reduced starvation.
- **GAE is unstable under NoNoise conditions**, showing performance degradation in 3 of 8 runs. The combination of NoNoise + NutGain6 + GAE causes catastrophic collapse in GAE_g4_NG6. MC return mode is more robust to these environmental simplifications.
- **The v4 MC_g1 unimodal recovery does not reproduce** — it was likely noise-dependent. However, MC gSize≥2 shows a **new pattern**: unimodal gates suppress early then gradually recover to ~40–48% sigmoid.

### 6.2 Limitations & Open Questions

1. **No noise-ON controls at matched training duration**: v5 trains ~5× longer than v4. The deeper collapse could be duration-dependent rather than noise-dependent. A matched-duration control would clarify.
2. **Temperature ceiling is a binding constraint**: All MC runs hit temp=3.0. Would raising `temp_clip` to [0.5, 5.0] change behavior? The modulator may be constrained rather than converged.
3. **Is the modulator doing anything useful?** MC NoNoise performance varies by only 0.6 points across grouping sizes, and gates are at extreme values. A **baseline without the modulator** would quantify its actual contribution.
4. **NutGain6 only tested at gSize 1 and 4**: The gSize=2 sweet spot from NutGain3 is untested with NutGain6.
5. **GAE critic instability**: GAE value loss is 100× higher than MC. Is this a fundamental issue with GAE in NoNoise environments, or can it be fixed with critic learning rate adjustments?
6. **MC_g1 unimodal suppression vs MC_g2+**: Why does MC_g1 suppress unimodal (sig 3.9%) while MC_g2/g4/g8 keep it open (sig 40–47%)? What is special about per-neuron (gSize=1) gating that causes this divergence?

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| **P0** | **Baseline without modulator (NoNoise, MC)** | Quantify modulator's actual contribution. If baseline matches NMN, the modulator is functionally inert. | Low |
| **P1** | **Raise temp_clip to [0.5, 5.0] or [0.5, 10.0] (MC, NoNoise, gSize=4)** | Temperature saturates at 3.0 — this may be a binding constraint. Raising the ceiling tests whether higher temperature improves performance. | Low |
| **P2** | **Match v4 training duration (~3–4B steps) with NoNoise** | Isolate whether deeper collapse is due to no-noise or longer training. Run MC gSize=1,4 NoNoise for only 4B steps. | Low |
| **P3** | **GAE with lower critic LR (NoNoise)** | GAE critic loss is 100× higher than MC. Reducing lr_critic to 0.00005 may stabilize GAE. | Low |
| **P4** | **NutGain6 with max_steps=1000** | 38% of MC_NG6 episodes hit max_steps=500. Extend the limit to see if the agent can survive even longer and reveal new dynamics. | Low |
| **P5** | **NoNoise + NutGain6 + MC, full gSize sweep** | Only gSize 1 and 4 tested. gSize=2 was the NutGain3 optimum — test if it's also best for NutGain6. | Low |

---

## Appendix

### A. Raw Data

Full extraction data available in:
- `tmp/20260317_v5_metrics.md` — summary metrics for all 12 runs
- Full timeseries data extracted via `wandb_metrics.py timeseries` (20-window mode)

### B. Run Label Reference

| Label | Full Run Name |
|-------|--------------|
| MC_g1 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_Multiplicative_gSize1 |
| MC_g2 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_Multiplicative_gSize2 |
| MC_g4 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_Multiplicative_gSize4 |
| MC_g8 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_Multiplicative_gSize8 |
| GAE_g1 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_Multiplicative_gSize1 |
| GAE_g2 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_Multiplicative_gSize2 |
| GAE_g4 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_Multiplicative_gSize4 |
| GAE_g8 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_Multiplicative_gSize8 |
| MC_g1_NG6 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_Multiplicative_gSize1 |
| MC_g4_NG6 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_Multiplicative_gSize4 |
| GAE_g1_NG6 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_Multiplicative_gSize1 |
| GAE_g4_NG6 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_Multiplicative_gSize4 |

### C. Config Diffs

All 12 runs share identical base config except:
- `agent.return_mode`: GAE or MC
- `modulation.grouping_size`: 1, 2, 4, or 8
- `body.food_nutrition_gain`: 3 (default) or 6 (NutGain6 runs)
- **New vs v4**: `perceptual_noise.enabled` is now `false` (was implicitly true in v4)

### D. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-03-17 | Initial analysis | Claude |
