---
title: "NMN Performance Diagnosis v8: Perceptual Noise + FiLM Grouping Size Sweep"
topic: diagnosis
status: active
created: 2026-03-20
last_updated: 2026-04-12
supersedes: NMN_PERFORMANCE_DIAGNOSIS_v7.md
---

# NMN Performance Diagnosis v8: Perceptual Noise + FiLM Grouping Size Sweep

> **Status**: COMPLETE (runs still training — analysis based on ~12–19M episodes, ~4B timesteps)
> **Date**: 2026-03-20
> **Author**: Claude (analysis), Sungwoo (experiment design & training)
> **Related**: [NMN_PERFORMANCE_DIAGNOSIS_v7.md](NMN_PERFORMANCE_DIAGNOSIS_v7.md) (v7, LayerNorm orthogonal experiment — this v8 is the v7 P0 follow-up)

---

## 1. Research Question

v7 established that LayerNorm is the primary driver of survival performance, not the modulation mechanism. However, v7 was conducted under NoNoise conditions, where all sensory channels deliver perfect information — the modulator has nothing to filter. v7's top-priority recommendation (P0) was to enable perceptual noise, giving the modulator a genuine signal-vs-noise discrimination task.

This experiment answers two questions:

1. **Does the FiLM modulator provide survival benefit under perceptual noise?** (v7 P0 — the key test of whether NMN has value beyond LayerNorm)
2. **Does grouping size affect FiLM performance under noise?** (v7 tested only gSize 1 and 4; this extends to 16 and 32)

> **H₀** (null): FiLM modulation has no effect on survival performance compared to the unmodulated LN baseline when perceptual noise is enabled.
> **H₁a**: Under perceptual noise, the FiLM modulator improves survival beyond the LN-only baseline by learning to filter noisy sensory channels.
> **H₁b**: Smaller grouping sizes (g1, g4) outperform larger ones (g16, g32) because finer-grained modulation enables better noise discrimination.
> **H₁c**: The MC advantage over GAE persists under noise conditions.
> **H₁d**: Perceptual noise reduces survival relative to v7 NoNoise conditions for all configurations.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| `modulation.type` | FiLM, null | FiLM is the reference modulation type from v6/v7; null baseline for comparison |
| `modulation.grouping_size` | 1, 4, 16, 32 | Extended sweep; v7 only tested 1 and 4 |
| `return_mode` | MC, GAE | Consistent with v4–v7 |

### 2.2 Controlled Variables

```yaml
perceptual_noise:
  enabled: true                  # KEY CHANGE from v7 (was false)
  modalities:
    injury:      { sigma: 0.1, mode: state_dependent, injury_noise_scale: 1.5, clip: [0, 1] }
    nutrition:   { sigma: 0.1, mode: state_dependent, injury_noise_scale: 1.5, clip: [0, 1] }
    satiation:   { sigma: 0.1, mode: state_dependent, injury_noise_scale: 1.5, clip: [0, 1] }
    extero_noc:  { sigma: 0.1, mode: state_dependent, injury_noise_scale: 1.5, clip: [0, 100] }
    olfaction:   { sigma: 0.2, mode: state_dependent, injury_noise_scale: 1.5, clip: [0, 100] }
    collision:   { sigma: 0.01, mode: constant, clip: [0, 1] }
    proprioception: { sigma: 0.05, mode: constant, clip: [0, 1] }
    visual:      { sigma: 0.2, mode: state_dependent, injury_noise_scale: 1.5, clip: [0, 100] }
    location:    { sigma: 0.01, mode: constant, clip: [-1, 1] }

modulation:
  mod_hidden_size: 16
  percept_bias_init: 3.0
  memory_bias_init: 0.0
  temp_clip: [0.5, 3.0]
  memory_clip: [-2.0, 2.0]
  multimodal_hidden_size: 128
  grouping: unified

agent:
  activation: relu
  rnn_type: GRU
  hidden_size: 128
  encoding_mode: hierarchical
  use_layer_norm: true           # All runs use LN (v7 established LN is critical)
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

body:
  food_nutrition_gain: 6
```

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| Runs still training (~12–19M episodes, ~4B timesteps) | All | Med | MC runs appear converged (flat windows 5–10); GAE still improving. Steady-state values are approximate. |
| Single seed per config | All | Med | Consistent with v7; differences >10 steps likely real based on v7's 2-seed variance estimate (Δ = 0.44 steps) |
| Different log_intervals (baselines N=10000 vs FiLM N=4000–5200) | Cross-config comparisons | Low | Timeseries windows show similar total episode counts (~12–19M); SS values are comparable |
| Noise config may have index mismatch bug (see MEMORY.md) | All noise runs | Med-High | Bug affects which modality gets which noise params; all runs affected equally, so relative comparisons remain valid |
| Only FiLM tested (no PreActivation/Multiplicative under noise) | All | Med | v7 showed all mod types perform similarly with LN; FiLM is the reference type |

## 3. Run Inventory

### v8 FiLM + Noise Runs (8 runs — all use_layer_norm: true, noise: true)

| # | Datetime ID | WandB ID | Return | gSize | Episodes | Label |
|---|------------|----------|--------|-------|----------|-------|
| 1 | 20260319-175239 | `0mumh9u6` | GAE | 1 | ~19M | GAE_FiLM_Noise_g1 |
| 2 | 20260319-175551 | `2c5uehzi` | GAE | 4 | ~18M | GAE_FiLM_Noise_g4 |
| 3 | 20260319-175718 | `aud9drk8` | MC | 1 | ~15M | MC_FiLM_Noise_g1 |
| 4 | 20260319-175827 | `gs98a6cc` | MC | 4 | ~16M | MC_FiLM_Noise_g4 |
| 5 | 20260319-180327 | `r2ptbe2f` | GAE | 16 | ~18M | GAE_FiLM_Noise_g16 |
| 6 | 20260319-180434 | `se935kp6` | GAE | 32 | ~15M | GAE_FiLM_Noise_g32 |
| 7 | 20260319-180533 | `rxnfrerg` | MC | 16 | ~12M | MC_FiLM_Noise_g16 |
| 8 | 20260319-180744 | `151rfhy8` | MC | 32 | ~12M | MC_FiLM_Noise_g32 |

### v8 Unmodulated Baselines (2 runs — LN: true, noise: true)

| # | Datetime ID | WandB ID | Return | LN | Episodes | Label |
|---|------------|----------|--------|-----|----------|-------|
| 9 | 20260319-180913 | `ptje3he2` | GAE | true | ~18M | GAE_Unmod_Noise_LN |
| 10 | 20260319-180955 | `auuuayj6` | MC | true | ~15M | MC_Unmod_Noise_LN |

---

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

#### 4.1.1 Survival Steps Under Noise — Ranked

**MC Runs — Ranked by Survival Steps:**

| Label | Steps SS | Reward SS | FoodEaten SS | TotalDamage SS | Term_Injury | Term_Starvation | Term_MaxSteps |
|-------|----------|-----------|-------------|----------------|-------------|-----------------|---------------|
| MC_FiLM_Noise_g16 | 283.44 ± 8.76 | -138.41 | 39.13 | 112.60 | 18.0% | 64.8% | 17.2% |
| **MC_Unmod_Noise_LN** | **283.64 ± 10.35** | -139.80 | 39.22 | 113.84 | 18.4% | 64.3% | 17.3% |
| MC_FiLM_Noise_g1 | 282.80 ± 9.92 | -139.03 | 39.04 | 113.69 | 18.3% | 64.8% | 17.0% |
| MC_FiLM_Noise_g4 | 278.75 ± 11.49 | -140.13 | 38.29 | 113.82 | 19.4% | 64.1% | 16.4% |
| MC_FiLM_Noise_g32 | 276.81 ± 9.25 | -138.16 | 37.83 | 110.73 | 17.5% | 66.9% | 15.6% |

**GAE Runs — Ranked by Survival Steps:**

| Label | Steps SS | Reward SS | FoodEaten SS | TotalDamage SS | Term_Injury | Term_Starvation | Term_MaxSteps |
|-------|----------|-----------|-------------|----------------|-------------|-----------------|---------------|
| **GAE_Unmod_Noise_LN** | **264.52 ± 11.22** | -137.00 | 38.60 | 106.78 | 18.7% | 67.4% | 13.9% |
| GAE_FiLM_Noise_g1 | 246.75 ± 5.92 | -137.66 | 34.77 | 105.01 | 20.5% | 68.2% | 11.3% |
| GAE_FiLM_Noise_g4 | 243.31 ± 5.80 | -136.65 | 34.10 | 103.39 | 20.9% | 69.0% | 10.2% |
| GAE_FiLM_Noise_g32 | 229.03 ± 6.79 | -138.05 | 31.18 | 100.70 | 20.3% | 71.0% | 8.8% |
| GAE_FiLM_Noise_g16 | 212.89 ± 7.02 | -140.24 | 27.55 | 100.59 | 22.6% | 70.1% | 7.3% |

#### 4.1.2 The v7 P0 Answer — Does the Modulator Help Under Noise?

**MC (comparing against MC_Unmod_Noise_LN = 283.64 steps):**

| Config | Steps SS | Δ vs Unmod | Interpretation |
|--------|----------|------------|----------------|
| MC_FiLM_Noise_g16 | 283.44 | **-0.20** | Tied |
| MC_FiLM_Noise_g1 | 282.80 | **-0.84** | Tied |
| MC_FiLM_Noise_g4 | 278.75 | **-4.89** | Slightly worse |
| MC_FiLM_Noise_g32 | 276.81 | **-6.83** | Worse |

> **Verdict on H₁a for MC: NOT SUPPORTED — the FiLM modulator provides zero benefit under noise.** No modulated config exceeds the unmodulated baseline. The best modulated MC config (g16, 283.44) is 0.2 steps *below* the baseline (283.64). The noise-filtering hypothesis fails for MC.

**GAE (comparing against GAE_Unmod_Noise_LN = 264.52 steps):**

| Config | Steps SS | Δ vs Unmod | Interpretation |
|--------|----------|------------|----------------|
| GAE_FiLM_Noise_g1 | 246.75 | **-17.77** | Substantially worse |
| GAE_FiLM_Noise_g4 | 243.31 | **-21.21** | Substantially worse |
| GAE_FiLM_Noise_g32 | 229.03 | **-35.49** | Much worse |
| GAE_FiLM_Noise_g16 | 212.89 | **-51.63** | Dramatically worse |

> **Verdict on H₁a for GAE: STRONGLY NOT SUPPORTED — the modulator actively hurts GAE survival under noise (-18 to -52 steps).** This is consistent with v7's finding (modulator hurts GAE) and is now amplified under noisy conditions.

#### 4.1.3 Grouping Size Effect Under Noise

**MC — Grouping Size:**

| gSize | Steps SS | gamma_multi SS | gamma_multi_std SS |
|-------|----------|----------------|-------------------|
| 1 | 282.80 | 0.966 | 1.409 |
| 4 | 278.75 | 0.922 | 0.983 |
| 16 | 283.44 | 0.461 | 0.230 |
| 32 | 276.81 | 0.389 | 0.200 |

> **Verdict on H₁b for MC: NOT SUPPORTED for performance** — grouping size has no meaningful effect on MC survival (range: 277–283, ~2% spread). However, **grouping size strongly affects gate behavior**: g1/g4 maintain near-identity gamma (~0.92–0.97) with high diversity (std 1.0–1.4), while g16/g32 attenuate to 0.39–0.46 with low diversity (std 0.20–0.23).

**GAE — Grouping Size:**

| gSize | Steps SS | gamma_multi SS | gamma_multi_std SS |
|-------|----------|----------------|-------------------|
| 1 | 246.75 | 0.962 | 1.219 |
| 4 | 243.31 | 0.948 | 0.994 |
| 32 | 229.03 | 0.417 | 0.678 |
| 16 | 212.89 | 0.711 | 0.832 |

> **Verdict on H₁b for GAE: PARTIALLY SUPPORTED — smaller grouping sizes (g1, g4) clearly outperform larger ones (g16, g32).** GAE_FiLM_g1 survives 34 steps longer than g16. However, even the best grouping size (g1) still performs -18 steps below the unmodulated baseline.

#### 4.1.4 Noise Impact — v8 Noise vs v7 NoNoise (at matched ~4B timesteps)

| Config | v7 NoNoise Steps | v8 Noise Steps | Δ (Noise Penalty) | % Drop |
|--------|-----------------|---------------|-------------------|--------|
| MC_FiLM_LN_g1 | 357.26 | 282.80 | **-74.46** | -20.8% |
| MC_FiLM_LN_g4 | 355.56 | 278.75 | **-76.81** | -21.6% |
| MC_Unmod_LN | 352.49 | 283.64 | **-68.85** | -19.5% |
| GAE_FiLM_LN_g1 | 289.49 | 246.75 | **-42.74** | -14.8% |
| GAE_FiLM_LN_g4 | 293.76 | 243.31 | **-50.45** | -17.2% |
| GAE_Unmod_LN | 313.68 | 264.52 | **-49.16** | -15.7% |

> **Verdict on H₁d: STRONGLY SUPPORTED — perceptual noise reduces survival by 43–77 steps (15–22%) across all configurations.** The noise penalty is larger for MC (~69–77 steps) than GAE (~43–50 steps) in absolute terms, but comparable in relative terms (~15–22%).

#### 4.1.5 MC vs GAE Under Noise

| Config | MC Steps | GAE Steps | Δ MC–GAE |
|--------|----------|-----------|----------|
| Unmod_Noise_LN | 283.64 | 264.52 | **+19.12** |
| FiLM_Noise_g1 | 282.80 | 246.75 | **+36.05** |
| FiLM_Noise_g4 | 278.75 | 243.31 | **+35.44** |
| FiLM_Noise_g16 | 283.44 | 212.89 | **+70.55** |
| FiLM_Noise_g32 | 276.81 | 229.03 | **+47.78** |

> **Verdict on H₁c: SUPPORTED — MC consistently outperforms GAE under noise.** The MC–GAE gap ranges from +19 to +71 steps. For the unmodulated baseline, the gap is +19 (smallest), consistent with v7's finding that the modulator hurts GAE more than MC. The modulator amplifies the MC–GAE gap.

### 4.2 Secondary Metrics

#### Loss / Critic

| Label | loss/value SS | loss/total SS | loss/grad_norm SS | loss/entropy SS |
|-------|-------------|-------------|-------------------|----------------|
| MC_Unmod_Noise_LN | 0.302 | 0.141 | 0.232 | -0.967 |
| MC_FiLM_Noise_g1 | 0.300 | 0.140 | 0.437 | -0.974 |
| MC_FiLM_Noise_g4 | 0.301 | 0.139 | 0.765 | -0.979 |
| MC_FiLM_Noise_g16 | 0.299 | 0.139 | 0.521 | -0.979 |
| MC_FiLM_Noise_g32 | 0.296 | 0.138 | 0.570 | -0.982 |
| GAE_Unmod_Noise_LN | 85.95 | 42.97 | 738.5† | -0.690 |
| GAE_FiLM_Noise_g1 | 91.67 | 45.83 | 155.5 | -0.727 |
| GAE_FiLM_Noise_g4 | 90.69 | 45.34 | 122.4 | -0.710 |
| GAE_FiLM_Noise_g16 | 104.94 | 52.46 | 170.2 | -0.684 |
| GAE_FiLM_Noise_g32 | 95.04 | 47.51 | 113.7 | -0.673 |

†GAE_Unmod_Noise_LN has an extreme gradient spike (max 1.1M), inflating the SS mean to 738.5. Final value: 22.5.

- MC value loss: ~0.30 across all configs — noise does not increase MC value loss (v7 NoNoise was also ~0.29).
- GAE critic loss: 86–105 under noise vs 69–76 under NoNoise (v7). **Noise increases GAE critic loss by ~20–30%.** Larger grouping sizes show higher critic loss (g16: 105, g32: 95 vs g1: 92).
- MC grad norms remain small (0.23–0.77). Modulated MC runs have slightly higher grad norms than unmodulated (0.44–0.77 vs 0.23).

#### Comparison with v7 Critic Loss

| Config | v7 NoNoise loss/value | v8 Noise loss/value | Δ |
|--------|----------------------|--------------------|----|
| MC_Unmod_LN | 0.292 | 0.302 | +0.010 |
| GAE_Unmod_LN | 68.67 | 85.95 | **+17.28** |
| GAE_FiLM_LN_g1 | 72.92 | 91.67 | **+18.75** |

Noise significantly worsens GAE critic learning but barely affects MC.

### 4.3 Diagnostic Metrics

#### 4.3.1 Gate Dynamics — FiLM Under Noise

**Multimodal Gate (gamma_multi_mean) — Steady-State:**

| gSize | MC | GAE |
|-------|-----|------|
| 1 | 0.966 | 0.962 |
| 4 | 0.922 | 0.948 |
| 16 | 0.461 | 0.711 |
| 32 | 0.389 | 0.417 |

**Unimodal Gate (gamma_uni_mean) — Steady-State:**

| gSize | MC | GAE |
|-------|-----|------|
| 1 | 1.262 | 1.057 |
| 4 | 0.367 | 1.479 |
| 16 | 0.953 | 1.445 |
| 32 | 0.711 | 1.809 |

**Key observations:**
1. **Grouping size strongly determines gamma_multi**: g1/g4 ≈ 0.92–0.97 (near-identity), g16/g32 ≈ 0.39–0.71 (substantial attenuation). With fewer gamma parameters, the network learns a conservative average rather than selective gating.
2. **All gates are healthy (no collapse)** — even the lowest (g32 MC = 0.39) is far above the v6 collapsed values (-4 to -10). LN continues to prevent collapse under noise.
3. **Comparing with v7 NoNoise FiLM gates**: v7 FiLM_LN_g1 gamma_multi ≈ 0.76–1.26; v8 FiLM_Noise_g1 gamma_multi ≈ 0.96. The gates are slightly different but both functional.
4. **Unimodal gates show an unexpected pattern**: MC_FiLM_g4 has very low gamma_uni (0.37), while GAE_FiLM_g32 has the highest (1.81). This suggests the unimodal and multimodal gates may compensate for each other.

#### 4.3.2 Gate Variance

| Label | gamma_multi_std | gamma_uni_std |
|-------|----------------|---------------|
| MC_FiLM_Noise_g1 | **1.409** | **2.217** |
| MC_FiLM_Noise_g4 | 0.983 | 1.172 |
| MC_FiLM_Noise_g16 | 0.230 | 0.739 |
| MC_FiLM_Noise_g32 | 0.200 | 0.392 |
| GAE_FiLM_Noise_g1 | **1.219** | **1.383** |
| GAE_FiLM_Noise_g4 | 0.994 | 1.490 |
| GAE_FiLM_Noise_g16 | 0.832 | 1.516 |
| GAE_FiLM_Noise_g32 | 0.678 | 1.478 |

Larger grouping sizes dramatically reduce gamma_multi_std for MC (1.41 → 0.20), confirming that fewer gamma parameters mean less diverse modulation. GAE shows a milder reduction. **Despite the high diversity at g1, this does not translate to better survival vs the unmodulated baseline.**

#### 4.3.3 Temperature

| Label | temp_mean SS | temp_min SS |
|-------|-------------|-------------|
| MC_FiLM_Noise_g1 | **3.000** | **3.000** |
| MC_FiLM_Noise_g4 | **3.000** | **3.000** |
| MC_FiLM_Noise_g16 | **3.000** | **3.000** |
| MC_FiLM_Noise_g32 | **3.000** | 2.992 |
| GAE_FiLM_Noise_g1 | 2.704 | 1.472 |
| GAE_FiLM_Noise_g4 | 2.867 | 1.428 |
| GAE_FiLM_Noise_g16 | 2.745 | 1.525 |
| GAE_FiLM_Noise_g32 | 2.862 | 1.496 |

**All MC FiLM runs saturate temperature at the 3.0 ceiling** — the modulator is NOT performing meaningful temperature modulation for MC. GAE runs maintain sub-ceiling temperatures (temp_min ≈ 1.4–1.5), suggesting some state-dependent temperature modulation.

#### 4.3.4 Beta (Additive Term)

| Label | beta_multi_mean | beta_uni_mean |
|-------|----------------|---------------|
| MC_FiLM_Noise_g1 | -0.582 | -1.281 |
| MC_FiLM_Noise_g4 | -0.442 | -1.929 |
| MC_FiLM_Noise_g16 | -0.182 | -1.163 |
| MC_FiLM_Noise_g32 | -0.144 | -0.976 |
| GAE_FiLM_Noise_g1 | -0.529 | -1.092 |
| GAE_FiLM_Noise_g4 | -0.589 | -1.719 |
| GAE_FiLM_Noise_g16 | -0.654 | -1.890 |
| GAE_FiLM_Noise_g32 | -0.425 | -2.104 |

All beta values are negative, consistent with v7. Larger grouping sizes show larger negative beta_uni for GAE (up to -2.1 for g32), suggesting stronger subtractive bias on unimodal features with coarser grouping.

### 4.4 Learning Dynamics

#### 4.4.1 Survival Steps Over Training (10-window means)

**MC Runs:**

| Window | MC_Unmod_LN | MC_FiLM_g1 | MC_FiLM_g4 | MC_FiLM_g16 | MC_FiLM_g32 |
|--------|------------|-----------|-----------|------------|------------|
| 1 | 224 | 236 | 222 | 216 | 224 |
| 2 | 264 | 274 | 254 | 268 | 261 |
| 3 | 276 | 281 | 264 | 275 | 267 |
| 5 | 283 | 282 | 277 | 281 | 268 |
| 7 | 282 | 285 | 276 | 284 | 274 |
| 10 | 283 | 282 | 279 | 285 | 277 |

All MC runs show smooth convergence by window 4–5 (~6–8M episodes), plateauing around 277–285. The unmodulated baseline converges at the same level as the best modulated configs.

**GAE Runs:**

| Window | GAE_Unmod_LN | GAE_FiLM_g1 | GAE_FiLM_g4 | GAE_FiLM_g16 | GAE_FiLM_g32 |
|--------|-------------|-----------|-----------|------------|------------|
| 1 | 152 | 153 | 155 | 135 | 148 |
| 2 | 212 | 202 | 210 | 166 | 199 |
| 3 | 231 | 214 | 226 | 179 | 212 |
| 5 | 253 | 228 | 237 | 195 | 222 |
| 7 | 261 | 238 | 243 | 205 | 230 |
| 10 | 266 | 245 | 243 | 213 | 229 |

**GAE_Unmod_LN is ahead of all FiLM configs from window 2 onward and is still climbing.** The unmodulated baseline leads by 21–53 steps by window 10. GAE_FiLM_g16 is the worst performer, consistently lagging.

#### 4.4.2 Multimodal Gate Timeseries — Grouping Size Effect

**gamma_multi_mean over training (10 windows, MC runs):**

| Window | MC_g1 | MC_g4 | MC_g16 | MC_g32 |
|--------|-------|-------|--------|--------|
| 1 | 1.00 | 0.94 | 0.88 | 0.79 |
| 3 | 1.03 | 0.76 | 0.54 | 0.45 |
| 5 | 1.06 | 0.82 | 0.43 | 0.42 |
| 7 | 0.96 | 0.93 | 0.44 | 0.42 |
| 10 | 0.93 | 0.93 | 0.46 | 0.38 |

**gamma_multi_mean over training (10 windows, GAE runs):**

| Window | GAE_g1 | GAE_g4 | GAE_g16 | GAE_g32 |
|--------|--------|--------|---------|---------|
| 1 | 0.82 | 0.91 | 0.93 | 0.86 |
| 3 | 0.82 | 0.89 | 0.92 | 0.62 |
| 5 | 0.89 | 0.92 | 0.83 | 0.50 |
| 7 | 0.92 | 0.91 | 0.75 | 0.43 |
| 10 | 0.97 | 0.91 | 0.68 | 0.41 |

**Key patterns:**
1. **g1 and g4 maintain near-identity gates (~0.9–1.0) throughout training** — stable, healthy.
2. **g16 and g32 show progressive attenuation** — starting near 0.9 and declining to 0.38–0.46 for MC, 0.41–0.68 for GAE. This is NOT collapse (sigmoid death spiral) — the decline is gradual and bounded. Rather, with coarse grouping, the optimizer finds that attenuating the modulated features is better than amplifying them.
3. **GAE_g1 gamma increases slowly** over training (0.82 → 0.97), suggesting the modulator is learning to open gates wider as training progresses.

---

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — The FiLM modulator provides no survival benefit under perceptual noise (HIGH confidence)**
What: Under noise, the unmodulated LN baseline achieves MC: 284 steps, GAE: 265 steps. The best FiLM config (MC_g16: 283, GAE_g1: 247) does not exceed the baseline. For GAE, all FiLM configs perform 18–52 steps worse than unmodulated.
Why: The modulator does not learn useful noise filtering. The FiLM mechanism modulates features via learned γ and β applied uniformly across time — it cannot distinguish "this observation is noisy right now" from "this feature should always be scaled this way." Noise-robust inference likely requires a fundamentally different architecture (e.g., Bayesian filtering, attention over observation reliability).
Evidence: MC: 0/4 FiLM configs exceed baseline. GAE: 0/4 FiLM configs exceed baseline. This is consistent across both return modes.
Confidence: HIGH — replicated across MC and GAE, 4 grouping sizes each.

**Finding 2 — Perceptual noise reduces survival by ~69–77 steps for MC and ~43–50 steps for GAE (HIGH confidence)**
What: Compared to v7 NoNoise: MC drops from 352–357 → 279–284 (-19 to -22%). GAE drops from 290–314 → 213–265 (-15 to -17% for unmodulated).
Why: Noisy observations corrupt the sensory information the agent relies on for foraging, predator avoidance, and navigation. State-dependent noise (injury-scaled) adds more noise when the agent is injured, creating a vicious cycle.
Evidence: Every v8 configuration shows dramatically lower survival than its v7 NoNoise counterpart.
Confidence: HIGH.

**Finding 3 — Larger grouping sizes cause gamma attenuation without gate collapse (MEDIUM confidence)**
What: g16 and g32 show gamma_multi declining to 0.38–0.71, while g1/g4 maintain ~0.92–0.97. The decline is gradual and bounded — not the exponential death spiral seen in v6.
Why: With grouping_size=32, a single γ scales 32 neurons simultaneously. The optimizer cannot selectively enhance/attenuate individual features, so it converges to a conservative attenuation (γ < 1). With g=1, each neuron has its own γ, allowing fine-grained identity-like modulation.
Evidence: gamma_multi_std at g1 (1.2–1.4) >> g32 (0.20–0.68). g16/g32 gamma timeseries show monotonic decline.
Confidence: MEDIUM — consistent pattern but limited to FiLM under noise.

**Finding 4 — MC temperature saturates at ceiling under noise (HIGH confidence)**
What: All MC FiLM runs hit temp_mean = temp_min = 3.0 (the configured ceiling). GAE maintains sub-ceiling temperatures (temp_min ≈ 1.4–1.5).
Why: MC's stable value function allows the policy to become confident quickly, pushing temperature to maximum. Under noise, this may be suboptimal — higher temperature means the modulator increases exploration noise in the hidden state on top of the already-noisy observations.
Evidence: All 4 MC FiLM configs show temp = 3.0; all 4 GAE FiLM configs show temp < 3.0.
Confidence: HIGH.

**Finding 5 — The modulator amplifies the MC-GAE gap under noise (MEDIUM confidence)**
What: For unmodulated configs, the MC–GAE gap is +19 steps. For FiLM g1, it grows to +36. For g16, it balloons to +71.
Why: The modulator adds gradient noise that destabilizes GAE's critic more than MC's value function (v7 Finding 5). Under perceptual noise, the already-noisy observations further corrupt the modulator's inputs, amplifying this instability for GAE.
Evidence: MC–GAE gap: Unmod (+19) < g1 (+36) < g4 (+35) < g32 (+48) < g16 (+71).
Confidence: MEDIUM — single seeds.

### 5.2 Cross-Run Comparisons

#### Noise Effect — Controlled (same architecture, LN=true, Noise vs NoNoise)

| Metric | MC_Unmod_NoNoise (v7) | MC_Unmod_Noise (v8) | Δ | GAE_Unmod_NoNoise (v7) | GAE_Unmod_Noise (v8) | Δ |
|--------|----------------------|--------------------|----|------------------------|---------------------|----|
| Steps SS | 352.49 | 283.64 | **-68.85** | 313.68 | 264.52 | **-49.16** |
| Reward SS | -131.34 | -139.80 | -8.46 | -130.08 | -137.00 | -6.92 |
| FoodEaten | 53.70 | 39.22 | **-14.48** | 50.58 | 38.60 | **-11.98** |
| Term_MaxSteps | 34.6% | 17.3% | **-17.3%** | 25.5% | 13.9% | **-11.6%** |
| loss/value | 0.292 | 0.302 | +0.010 | 68.67 | 85.95 | **+17.28** |

Noise reduces food eaten by ~12–14 meals, Term_MaxSteps by ~12–17%, and increases GAE critic loss by ~25%. MC value loss is barely affected (+3%).

#### Grouping Size Effect — Controlled (FiLM+LN+Noise, MC vs GAE)

| gSize | MC Steps | Δ vs g1 | GAE Steps | Δ vs g1 |
|-------|----------|---------|-----------|---------|
| 1 | 282.80 | — | 246.75 | — |
| 4 | 278.75 | -4.05 | 243.31 | -3.44 |
| 16 | 283.44 | +0.64 | 212.89 | **-33.86** |
| 32 | 276.81 | -5.99 | 229.03 | **-17.72** |

For MC, grouping size has negligible effect (range: 6.6 steps). For GAE, larger grouping sizes substantially degrade performance, with g16 being the worst (-34 steps vs g1). The GAE sensitivity to grouping size is consistent with GAE's sensitivity to any added complexity.

### 5.3 Failure Modes & Pathologies

#### Pathology 1: Temperature Saturation (MC FiLM under noise)
- **Affected**: All 4 MC FiLM configs
- **Severity**: temp saturates at 3.0 ceiling throughout training
- **Impact**: The modulator's temperature channel is non-functional for MC under noise
- **Likely cause**: MC's low value loss (0.30) enables fast policy convergence → temperature pushed to max. May also reflect that higher temperature is not harmful under MC (unlike GAE where critic instability penalizes it)

#### Pathology 2: GAE_Unmod_Noise_LN Gradient Spike
- **Affected**: GAE_Unmod_Noise_LN
- **Severity**: Single extreme gradient spike (loss/grad_norm max 1.1M), inflating SS mean to 738
- **Impact**: None apparent — steps continue climbing after spike
- **Recurrence**: This is the same pathology seen in v7's GAE_Unmod_LN (max 3.35M)

#### Pathology 3: Large Grouping Size Gamma Attenuation
- **Affected**: MC_FiLM_g16, MC_FiLM_g32, GAE_FiLM_g16, GAE_FiLM_g32
- **Severity**: gamma_multi declines to 0.38–0.46 (MC) and 0.41–0.71 (GAE) over training
- **Impact**: Moderate for GAE (g16/g32 lose 18–52 steps vs baseline); negligible for MC
- **Likely cause**: Coarse grouping forces a single γ per group of 16–32 neurons; optimizer cannot find a useful group-level modulation, so attenuates toward identity (σ(0) = 0.5)

---

## 6. Conclusions

### 6.1 Summary

- **The FiLM modulator does NOT provide noise-filtering benefit.** Under perceptual noise, the unmodulated LN baseline matches or exceeds all FiLM configurations for both MC (284 vs 277–283) and GAE (265 vs 213–247). **The v7 P0 hypothesis is rejected** — noise does not reveal a hidden value of neuromodulation.

- **Perceptual noise substantially degrades survival** — ~69 steps (-20%) for MC and ~49 steps (-16%) for GAE compared to v7 NoNoise conditions. The noise penalty primarily reduces food consumption (-12 to -14 meals) and max-step survival rate (-12 to -17%).

- **Grouping size has minimal effect on MC but substantially harms GAE at larger sizes.** g16 and g32 cause gamma attenuation (not collapse) and degrade GAE performance by 18–52 steps. For MC, all grouping sizes perform within 7 steps of each other.

- **The modulator architecture (FiLM) cannot perform noise filtering** because it applies static learned γ/β to features, not adaptive filtering based on observation reliability. A noise-robust modulator would need to estimate per-timestep observation confidence — fundamentally different from the current mechanism.

- **All findings from v7 are reinforced and amplified under noise**: (1) LN is the performance driver, not modulation, (2) modulator hurts GAE, (3) MC > GAE. The noise condition makes these patterns stronger, not weaker.

### 6.2 Limitations & Open Questions

1. **Runs still training**: Most runs are at ~12–19M episodes. GAE runs are still improving; final values may change.
2. **Only FiLM tested under noise**: PreActivation and Multiplicative were not included. PreActivation + LN was the best modulated config in v7 — it might perform differently under noise.
3. **Single seeds**: Differences of <7 steps (MC range) are likely within seed variance.
4. **Noise config index mismatch bug**: The known bug in `config_loader.py` means modalities receive wrong noise parameters. All runs are equally affected, so relative comparisons hold, but absolute survival values may change after the bug is fixed.
5. **Temperature ceiling**: Would raising `temp_clip` above 3.0 allow MC to learn useful temperature modulation? Or would it just saturate at the new ceiling?
6. **Is the noise too strong or too weak?** The current noise config may overwhelm any modulation benefit, or it may be too uniform (all modalities get similar noise) to reward selective filtering.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| **P0** | **Fix noise index mismatch bug, re-run key configs** | The bug means every modality receives the wrong noise sigma/mode. Fixing it could change the noise landscape enough to reveal modulation benefits — or confirm they don't exist with correct noise. | Med |
| **P1** | **PreActivation + LN under noise (g1)** | PreActivation was the best modulated config in v7 (healthiest gates, highest variance). It may handle noise differently from FiLM. | Low |
| **P2** | **Multi-seed replication (MC_Unmod_Noise_LN, MC_FiLM_Noise_g1, GAE_Unmod_Noise_LN)** | Confirm that the null result (modulator = no benefit) is not a seed artifact. 3–5 seeds needed. | Med |
| **P3** | **Attention-based or Bayesian modulator architecture** | The fundamental issue is that FiLM cannot distinguish "noisy timestep" from "always-attenuated feature." An architecture that estimates per-timestep observation reliability could provide genuine noise filtering. | High |
| **P4** | **Heterogeneous noise profiles** | Current noise is relatively uniform across modalities. A more heterogeneous profile (e.g., very noisy olfaction, clean proprioception) might create a stronger selective pressure for modulation. | Low |

---

## Appendix

### A. Raw Data

Full extraction data available in:
- `tmp/20260320_v8_extract_results.md` — summary metrics for all 10 runs
- `tmp/20260320_v8_timeseries_results.md` — 10-window timeseries for all runs

### B. Run Label Reference

| Label | Full Run Name |
|-------|--------------|
| GAE_FiLM_Noise_g1 | rppoNMN_GAE_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize1 |
| GAE_FiLM_Noise_g4 | rppoNMN_GAE_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize4 |
| MC_FiLM_Noise_g1 | rppoNMN_MC_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize1 |
| MC_FiLM_Noise_g4 | rppoNMN_MC_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize4 |
| GAE_FiLM_Noise_g16 | rppoNMN_GAE_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize16 |
| GAE_FiLM_Noise_g32 | rppoNMN_GAE_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize32 |
| MC_FiLM_Noise_g16 | rppoNMN_MC_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize16 |
| MC_FiLM_Noise_g32 | rppoNMN_MC_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize32 |
| GAE_Unmod_Noise_LN | rppo_GAE_2bush4pred2food_NutGain6_LayerNorm |
| MC_Unmod_Noise_LN | rppo_MC_2bush4pred2food_NutGain6_LayerNorm |

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-03-20 | Initial analysis (runs still in progress) | Claude |
