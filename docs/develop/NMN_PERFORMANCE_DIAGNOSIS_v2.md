# NMN Performance Diagnosis v2: Environment Sweep with Fixed Modulator

> **Status**: INVESTIGATING (mid-training snapshot)
> **Date**: 2026-03-10
> **Previous**: [NMN_PERFORMANCE_DIAGNOSIS.md](NMN_PERFORMANCE_DIAGNOSIS.md) (v1, §1–11)
> **Data**: All runs still running as of 2026-03-10. Metrics are mid-training snapshots (~10K iterations each). Trends are established but final values will change. Re-analyze when runs complete.

---

## Background (from v1)

The v1 diagnosis (§1–9) found the NMN agent matched baseline performance because the modulator learned **degenerate shortcuts**: suppressing interoceptive features (gamma_body → sigmoid 1.9%), freezing the GRU (z_memory → -6.45), and inflating temperature to 10x (pinned at ceiling). The §10 ablation identified `mod_hidden_size=16` as the critical fix — smaller modulators are too constrained to find extreme degenerate strategies. The §11 discussion deferred environment curriculum to Phase 2.

**This document (v2)** analyzes a new 15-run sweep that applies all v1-recommended fixes across 5 different environment densities.

---

## 12. Environment Sweep Experiment (2026-03-09)

### 12.1 Experiment Design

15 runs testing **Multiplicative vs PreActivation vs Baseline** across 5 environment configurations that vary bush count, predator count, and food density.

**Fixed NMN config (all NMN runs):**
```yaml
modulation:
  type: Multiplicative | PreActivation  # the only NMN variable
  mod_hidden_size: 16       # v1 §10 fix
  grouping_size: 60
  percept_bias_init: 3.0
  memory_bias_init: 0.0
  temp_clip: [0.5, 3.0]    # v1 §8 fix (tighter than original [0.1, 10.0])
  memory_clip: [-2.0, 2.0]  # v1 §8 fix (NEW — prevents GRU freeze)
```

**Shared agent config (all runs):**
```yaml
agent:
  activation: relu           # No more activation mismatch (v1 §3)
  return_mode: GAE
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
```

**Shared environment (all runs, except density variables):**
- Grid: 10x10, max_steps=500
- Danger: 8 (2/quadrant, damage [15,45])
- Rocks: 12 (3/quadrant, damage [0.1,0.5])
- Neutral animals: 5 rabbits
- Perceptual noise: injury only state_dependent (sigma=0.05, alpha=1.5); all others constant

### 12.2 Config Verification

All 15 run configs verified against WandB API. **No mismatches found.**

| Run ID | Tag | Modulation | Config Match |
|---|---|---|---|
| 20260309-212615 | rppoNMN_..._2bush4pred1food_..._Multiplicative_... | `Multiplicative` | YES |
| 20260309-213440 | rppoNMN_..._3bush5pred2food_..._Multiplicative_... | `Multiplicative` | YES |
| 20260309-213532 | rppoNMN_..._2bush3pred2food_..._Multiplicative_... | `Multiplicative` | YES |
| 20260309-220707 | rppoNMN_..._2bush4pred2food_..._Multiplicative_... | `Multiplicative` | YES |
| 20260309-220117 | rppoNMN_..._2bush4pred1food_..._PreActivation_... | `PreActivation` | YES |
| 20260309-220237 | rppoNMN_..._3bush5pred2food_..._PreActivation_... | `PreActivation` | YES |
| 20260309-220432 | rppoNMN_..._2bush3pred2food_..._PreActivation_... | `PreActivation` | YES |
| 20260309-220618 | rppoNMN_..._2bush4pred2food_..._PreActivation_... | `PreActivation` | YES |
| 20260309-221708 | rppo_..._2bush4pred2food_behavMetrics | `null` (baseline) | YES |
| 20260309-222226 | rppo_..._2bush3pred2food_behavMetrics | `null` (baseline) | YES |
| 20260309-222427 | rppo_..._3bush5pred2food_behavMetrics | `null` (baseline) | YES |
| 20260309-222606 | rppo_..._2bush4pred1food_behavMetrics | `null` (baseline) | YES |
| 20260309-222800 | rppo_..._1bush3pred1food_behavMetrics | `null` (baseline) | YES |
| 20260309-222923 | rppoNMN_..._1bush3pred1food_..._Multiplicative_..._BehavMetrics | `Multiplicative` | YES |
| 20260309-223610 | rppoNMN_..._1bush3pred1food_..._PreActivation_..._BehavMetrics | `PreActivation` | YES |

All NMN runs share identical modulation config: `temp_clip=[0.5,3]`, `memory_clip=[-2,2]`, `grouping_size=60`, `mod_hidden_size=16`, `percept_bias_init=3`.

### 12.3 Run Inventory

| # | Datetime ID | WandB ID | Type | Env (bush/pred/food) | Status |
|---|---|---|---|---|---|
| 1 | 20260309-212615 | `ql6yd7gf` | Mult | 2b/4p/1f | Running |
| 2 | 20260309-213440 | `uepbf8w8` | Mult | 3b/5p/2f | Running |
| 3 | 20260309-213532 | `3q1f0rhs` | Mult | 2b/3p/2f | Running |
| 4 | 20260309-220707 | `qpy3xa8n` | Mult | 2b/4p/2f | Running |
| 5 | 20260309-220117 | `t8zawna2` | PreAct | 2b/4p/1f | Running |
| 6 | 20260309-220237 | `5pept2kl` | PreAct | 3b/5p/2f | Running |
| 7 | 20260309-220432 | `3eoz8uh7` | PreAct | 2b/3p/2f | Running |
| 8 | 20260309-220618 | `adox8yze` | PreAct | 2b/4p/2f | Running |
| 9 | 20260309-221708 | `acewh2tg` | Baseline | 2b/4p/2f | Running |
| 10 | 20260309-222226 | `qknfof7q` | Baseline | 2b/3p/2f | Running |
| 11 | 20260309-222427 | `r6aqss8x` | Baseline | 3b/5p/2f | Running |
| 12 | 20260309-222606 | `398a5gm4` | Baseline | 2b/4p/1f | Running |
| 13 | 20260309-222800 | `bgubemyk` | Baseline | 1b/3p/1f | Running |
| 14 | 20260309-222923 | `vnfaaomy` | Mult+BM | 1b/3p/1f | Running |
| 15 | 20260309-223610 | `dvs6rgh2` | PreAct+BM | 1b/3p/1f | Running |

"BM" = BehavMetrics logging variant. bush/pred/food = per-quadrant bush count / total predators / per-quadrant food count.

---

### 12.4 Episode Performance by Environment

All values are **steady-state** (last 20% of training so far). Bold indicates best in group.

#### 12.4.1 Environment: 2bush / 4pred / 1food (low food density)

| | Mult | PreAct | Baseline |
|---|---|---|---|
| **Reward (SS)** | -117.33 ± 2.95 | **-108.73 ± 2.08** | -114.19 ± 2.69 |
| **Steps (SS)** | 127.29 ± 4.47 | 98.82 ± 1.48 | 112.18 ± 3.27 |
| FoodEaten | 15.27 ± 2.09 | 1.07 ± 0.44 | 7.87 ± 1.45 |
| TotalDamage | 53.16 ± 4.71 | **34.46 ± 3.39** | 43.70 ± 4.12 |
| DamagePredator | 21.99 ± 2.55 | **17.10 ± 2.14** | 20.62 ± 2.48 |
| DamageDanger | 30.39 ± 3.42 | **16.99 ± 2.22** | 22.56 ± 2.89 |
| RestCount | 57.58 | 50.70 | 53.71 |
| MeanDistPredator | 5.42 | 5.45 | 5.41 |
| Term_Injury | 0.069 | **0.057** | 0.068 |
| Term_Starvation | 0.931 | 0.943 | 0.932 |
| Trajectory (Reward) | -177.83 → -117.0 | -178.27 → -106.6 | -179.74 → -113.6 |

**Result: PreAct WINS (+5.5 pts over baseline).** PreAct achieves the best reward (-108.73) with lowest total damage (34.46) and lowest danger hits. However, it eats almost no food (1.07) and has the shortest episodes (98.82 steps) — it learned a **pure avoidance strategy**: minimize damage, die of starvation quickly but with minimal penalty. Mult survives longest (127 steps) and eats the most food (15.27) but accumulates more damage.

#### 12.4.2 Environment: 2bush / 4pred / 2food (medium density)

| | Mult | PreAct | Baseline |
|---|---|---|---|
| **Reward (SS)** | **-131.97 ± 4.29** | -135.96 ± 4.39 | -136.09 ± 4.55 |
| **Steps (SS)** | 178.07 ± 12.51 | 170.24 ± 9.19 | 173.24 ± 10.55 |
| FoodEaten | **41.07 ± 6.11** | 37.38 ± 4.46 | 39.05 ± 5.04 |
| TotalDamage | **81.20 ± 7.13** | 85.03 ± 6.61 | 86.02 ± 6.95 |
| DamagePredator | **32.62 ± 4.13** | 33.90 ± 3.71 | 34.12 ± 4.07 |
| RestCount | 69.58 | 66.33 | 68.78 |
| Term_Injury | **0.119** | 0.138 | 0.143 |
| Term_Starvation | 0.879 | 0.860 | 0.853 |
| Trajectory (Reward) | -176.42 → -132.7 | -198.67 → -135.7 | -176.45 → -140.2 |

**Result: Mult slightly WINS (+4.1 pts over baseline).** All three agents converge to similar behavior — long episodes (~170-178 steps), heavy food consumption (~37-41), and similar damage profiles. The Mult agent has a slight edge in all metrics. Differences are within ~1 standard deviation.

#### 12.4.3 Environment: 2bush / 3pred / 2food (easier — fewer predators)

| | Mult | PreAct | Baseline |
|---|---|---|---|
| **Reward (SS)** | -134.73 ± 4.37 | **-132.89 ± 4.36** | -136.97 ± 4.53 |
| **Steps (SS)** | **186.31 ± 11.56** | 183.68 ± 10.41 | 164.76 ± 10.96 |
| FoodEaten | **45.31 ± 5.68** | 43.85 ± 5.09 | 35.18 ± 5.21 |
| TotalDamage | 89.39 ± 7.05 | **84.76 ± 6.77** | 86.64 ± 7.27 |
| DamagePredator | 33.32 ± 3.78 | **32.59 ± 3.84** | 32.31 ± 3.94 |
| DamageDanger | **54.74 ± 5.51** | 50.65 ± 5.13 | 53.12 ± 5.79 |
| RestCount | **72.24** | 69.43 | 65.28 |
| Term_Injury | 0.132 | **0.123** | 0.162 |
| Term_Starvation | 0.857 | 0.873 | 0.834 |
| Trajectory (Reward) | -176.87 → -146.0 | -180.85 → -145.0 | -183.66 → -138.5 |

**Result: Both NMN variants slightly better than baseline (+2–4 pts).** NMN agents survive ~20 steps longer and eat ~10 more food items. Lower injury death rate. PreAct has slightly better reward; Mult survives longest. Both still within noise of baseline.

#### 12.4.4 Environment: 3bush / 5pred / 2food (high density — most predators, most cover)

| | Mult | PreAct | Baseline |
|---|---|---|---|
| **Reward (SS)** | -132.56 ± 4.20 | -131.19 ± 4.26 | **-128.38 ± 4.09** |
| **Steps (SS)** | **195.61 ± 12.45** | 178.06 ± 9.81 | 176.39 ± 9.57 |
| FoodEaten | **49.63 ± 6.13** | 40.90 ± 4.75 | 39.92 ± 4.68 |
| TotalDamage | 87.89 ± 7.40 | 80.60 ± 6.55 | **74.67 ± 6.49** |
| DamagePredator | 34.59 ± 4.21 | 30.57 ± 3.61 | **32.25 ± 3.93** |
| DamageDanger | **51.95 ± 5.66** | 48.77 ± 4.97 | 41.23 ± 4.70 |
| RestCount | **77.22** | 69.61 | 71.78 |
| Term_Injury | 0.113 | 0.115 | **0.099** |
| Term_Starvation | 0.875 | 0.882 | **0.899** |
| Trajectory (Reward) | -174.22 → -124.8 | -182.96 → -135.3 | -179.94 → -131.5 |

**Result: Baseline WINS (-128.38, +3–4 pts over NMN).** The baseline achieves the best reward with the lowest total damage (74.67). Despite NMN agents surviving longer (Mult 196 steps vs baseline 176), they accumulate more danger damage. The extra bushes (3/quad) provide ample hiding — the baseline's fixed architecture is sufficient. NMN modulation provides no benefit here.

#### 12.4.5 Environment: 1bush / 3pred / 1food (sparse — least cover, least food)

| | Mult+BM | PreAct+BM | Baseline |
|---|---|---|---|
| **Reward (SS)** | **-116.72 ± 2.92** | -119.59 ± 3.04 | -120.53 ± 3.44 |
| **Steps (SS)** | 109.07 ± 3.50 | 114.50 ± 3.76 | **123.73 ± 4.44** |
| FoodEaten | 6.80 ± 1.60 | **9.64 ± 1.66** | **14.02 ± 1.94** |
| TotalDamage | **51.00 ± 4.48** | 56.18 ± 4.76 | 59.11 ± 5.09 |
| DamagePredator | **21.04 ± 2.48** | 22.33 ± 2.51 | 23.39 ± 3.09 |
| DamageDanger | **29.29 ± 3.17** | 32.87 ± 3.34 | 34.78 ± 3.65 |
| RestCount | 52.33 | 52.25 | 54.52 |
| Term_Injury | **0.090** | 0.102 | 0.094 |
| Term_Starvation | 0.910 | 0.898 | 0.906 |
| Trajectory (Reward) | -182.65 → -118.5 | -199.78 → -120.6 | -176.01 → -123.9 |

**Result: Mult WINS (+3.8 pts over baseline).** Mult achieves the best reward by minimizing damage (51.00 vs baseline 59.11). It eats least food and has shortest episodes — similar to the PreAct strategy in 2b4p1f: **damage avoidance > food seeking** in sparse environments. Baseline survives longest but accumulates more damage.

### 12.4.6 Cross-Environment Summary

| Environment | Winner | NMN Advantage | Mechanism |
|---|---|---|---|
| 2b/4p/1f (low food) | PreAct (+5.5 pts) | Significant | Damage avoidance strategy |
| 2b/4p/2f (medium) | Mult (+4.1 pts) | Moderate | Slightly better foraging |
| 2b/3p/2f (easy) | PreAct (+4.1 pts) | Moderate | Longer survival, more food |
| 3b/5p/2f (dense) | Baseline (+3.2 pts) | **Negative** | Baseline has lower damage |
| 1b/3p/1f (sparse) | Mult (+3.8 pts) | Moderate | Damage avoidance |

**Pattern**: NMN wins in low-to-medium density environments (3–5.5 pts). Baseline wins in the highest-density environment. The NMN advantage comes primarily from **damage avoidance** (lower TotalDamage), not from better foraging or adaptive sensory modulation.

---

### 12.5 Modulator Metrics Analysis

#### 12.5.1 Gamma — Perceptual Gain

##### Unimodal Gates

| Run (Type, Env) | gamma_uni_mean (SS) | sigmoid | Interpretation |
|---|---|---|---|
| Mult 2b4p1f | +0.324 | 58% | Moderate gating |
| Mult 3b5p2f | -0.490 | 38% | Suppressed |
| Mult 2b3p2f | -1.504 | 18% | **Heavy suppression** |
| Mult 2b4p2f | +0.256 | 56% | Moderate gating |
| Mult 1b3p1f+BM | +0.432 | 61% | Moderate gating |
| PreAct 2b4p1f | +1.238 | 78% | Near pass-through |
| PreAct 3b5p2f | +2.067 | 89% | Near pass-through |
| PreAct 2b3p2f | +1.725 | 85% | Near pass-through |
| PreAct 2b4p2f | +1.845 | 86% | Near pass-through |
| PreAct 1b3p1f+BM | +1.789 | 86% | Near pass-through |

**Finding**: PreActivation universally preserves unimodal features (78–89% pass-through). Multiplicative is more variable (18–61%) with one run heavily suppressing unimodal signals. PreAct's beta term provides an alternative pathway that reduces pressure on gamma, explaining why gamma stays higher.

**Comparison to v1 §10**: In the previous ablation, Mult h=16 runs had gamma_uni at 27–35% sigmoid. These new runs show a wider range (18–61%), suggesting environment density influences unimodal gating — but the trend is similar.

##### Multimodal (Association Hub) Gates

| Run (Type, Env) | gamma_multi_mean (SS) | sigmoid | Interpretation |
|---|---|---|---|
| Mult 2b4p1f | **-9.192** | **0.01%** | Collapsed |
| Mult 3b5p2f | -7.857 | 0.04% | Collapsed |
| Mult 2b3p2f | -7.041 | 0.09% | Collapsed |
| Mult 2b4p2f | -7.055 | 0.09% | Collapsed |
| Mult 1b3p1f+BM | -6.389 | 0.17% | Collapsed |
| PreAct 2b4p1f | -7.060 | 0.09% | Collapsed |
| PreAct 3b5p2f | -7.379 | 0.06% | Collapsed |
| PreAct 2b3p2f | -7.073 | 0.08% | Collapsed |
| PreAct 2b4p2f | -7.249 | 0.07% | Collapsed |
| PreAct 1b3p1f+BM | -6.786 | 0.11% | Collapsed |

**CRITICAL FINDING: Multimodal hub collapse is UNIVERSAL.**

All 10 NMN runs in this sweep show gamma_multi < -6.4 (sigmoid < 0.17%). Combined with v1's 2 runs and §10's 11 runs, this is now **22 out of 22 NMN configurations** that collapse the multimodal hub. This is independent of:
- Modulation type (Multiplicative, PreActivation)
- Grouping size (20, 40, 60, 80, 100)
- Mod hidden size (16, 32)
- Environment density (1b3p1f through 3b5p2f)
- Training duration (2.5B to 14B timesteps)

**Conclusion**: The multimodal association hub is not producing useful features for the task. This is an **architectural problem**, not a modulator problem. The modulator correctly identifies that suppressing the hub reduces variance without losing useful information.

#### 12.5.2 Temperature

| Run (Type, Env) | temp_mean (SS) | temp_max (SS) | temp_min (SS) | Interpretation |
|---|---|---|---|---|
| Mult 2b4p1f | 2.542 | **3.000 PIN** | 1.201 | High inflation |
| Mult 3b5p2f | 2.572 | **3.000 PIN** | 0.922 | High inflation |
| Mult 2b3p2f | 2.521 | **3.000 PIN** | 0.793 | High inflation |
| Mult 2b4p2f | 2.099 | 2.999 | 0.703 | Moderate inflation |
| Mult 1b3p1f+BM | **1.917** | 2.980 | 0.704 | **Best Mult** |
| PreAct 2b4p1f | **2.906** | **3.000 PIN** | 0.664 | **Heavy inflation** |
| PreAct 3b5p2f | 2.625 | **3.000 PIN** | 1.026 | High inflation |
| PreAct 2b3p2f | 2.740 | **3.000 PIN** | 1.158 | High inflation |
| PreAct 2b4p2f | 2.719 | **3.000 PIN** | 0.779 | High inflation |
| PreAct 1b3p1f+BM | **2.877** | **3.000 PIN** | 1.025 | **Heavy inflation** |

**Finding**: Temperature inflation persists despite the tighter [0.5, 3.0] bounds. All runs pin or nearly pin temp_max at 3.0. PreActivation runs consistently inflate more than Multiplicative (mean 2.63–2.91 vs 1.92–2.57).

**Comparison to v1**: In v1, temperature with [0.1, 10.0] bounds reached mean 3.29 with max pinned at 10.0. The [0.5, 3.0] bounds reduced absolute inflation but the modulator still pushes to the ceiling. In v1 §10, Mult h=16 runs had healthy temperature (mean 1.3–2.0) — the new runs show higher inflation (1.9–2.6), possibly because the environment sweep includes harder environments that push the modulator toward exploration.

**Recommendation**: Tighten to [0.5, 2.0]. The modulator would push temperature higher if allowed — the ceiling is the constraint, not the equilibrium.

#### 12.5.3 z_memory (GRU Gate Bias)

| Run (Type, Env) | z_mem_mean (SS) | z_mem_std (SS) | Interpretation |
|---|---|---|---|
| Mult 2b4p1f | +0.859 | 2.053 | Mild forgetting, **high variance** |
| Mult 3b5p2f | +0.267 | 0.838 | Near neutral — **healthy** |
| Mult 2b3p2f | -0.263 | 0.986 | Near neutral — **healthy** |
| Mult 2b4p2f | -0.215 | 0.700 | Near neutral — **healthy** |
| Mult 1b3p1f+BM | +0.212 | 0.641 | Near neutral — **healthy** |
| PreAct 2b4p1f | +0.191 | 0.666 | Near neutral — **healthy** |
| PreAct 3b5p2f | -0.206 | 0.755 | Near neutral — **healthy** |
| PreAct 2b3p2f | +0.088 | 0.909 | Near neutral — **healthy** |
| PreAct 2b4p2f | **+1.330** | **2.179** | Approaching boundary — **watch** |
| PreAct 1b3p1f+BM | +0.554 | 1.415 | Mild forgetting, moderate variance |

**FINDING: The [-2, 2] memory clamp WORKS.**

In v1, z_memory reached -6.45 (always-retain) or +7 (always-forget), completely disabling recurrent memory. With the clamp, 8 of 10 runs maintain z_memory in the healthy range (-0.26 to +0.86). Two runs show elevated values (Mult 2b4p1f at 0.86/std=2.05 and PreAct 2b4p2f at 1.33/std=2.18) — these are worth monitoring as training continues, but are not yet pathological.

**Recommendation**: Keep the [-2, 2] clamp. Consider tightening to [-1.5, 1.5] if the high-variance runs deteriorate.

#### 12.5.4 Beta (PreActivation Only)

| Run (Env) | beta_uni_mean (SS) | beta_multi_mean (SS) | Interpretation |
|---|---|---|---|
| PreAct 2b4p1f | **-2.236** | -0.574 | Strong unimodal inhibition |
| PreAct 3b5p2f | **-2.827** | -0.248 | Strong unimodal inhibition |
| PreAct 2b3p2f | -2.141 | -0.139 | Strong unimodal inhibition |
| PreAct 2b4p2f | -1.678 | -0.154 | Moderate inhibition |
| PreAct 1b3p1f+BM | -0.789 | -0.166 | Mild inhibition |

**Finding**: Beta is **consistently negative** (inhibitory) across all PreActivation runs. The unimodal beta ranges from -0.79 to -2.83 — the modulator is raising activation thresholds, filtering weak signals. This is the **opposite** of the intended disinhibition effect described in PreActivation theory.

The beta term adds degrees of freedom that the optimizer uses for inhibition rather than the intended flexible gating. Combined with PreActivation's higher temperature inflation, the extra parameters appear to enable **more sophisticated pathologies** rather than better adaptive modulation.

**Recommendation**: PreActivation does not offer clear advantages over Multiplicative. Consider standardizing on Multiplicative for simplicity.

#### 12.5.5 Gradient Stability (NEW)

| Run | loss/grad_norm (SS) | modulator/grad_norm (SS) | Concern |
|---|---|---|---|
| Mult 2b4p1f | 47.22 ± 27.37 | 41.81 ± 25.39 | Normal |
| Mult 3b5p2f | 57.79 ± 233.55 | 51.08 ± 150.48 | **Spiky** |
| Mult 2b3p2f | 31.88 ± 17.06 | 28.87 ± 16.73 | Normal |
| **Mult 2b4p2f** | **9,051 ± 130,910** | **3,825 ± 68,071** | **UNSTABLE** |
| Mult 1b3p1f+BM | 58.23 ± 33.34 | 32.62 ± 18.57 | Normal |
| **PreAct 2b4p1f** | **8,596 ± 329,376** | **6,644 ± 254,071** | **UNSTABLE** |
| PreAct 3b5p2f | 50.84 ± 173.51 | 45.90 ± 154.75 | **Spiky** |
| PreAct 2b3p2f | 35.25 ± 17.79 | 32.72 ± 17.53 | Normal |
| PreAct 2b4p2f | 44.20 ± 66.80 | 38.99 ± 55.64 | Mild |
| PreAct 1b3p1f+BM | 48.08 ± 34.06 | 45.77 ± 32.47 | Normal |
| Baseline 2b4p2f | 75.49 ± 76.03 | — | Mild |
| Baseline 2b3p2f | 106.77 ± 253.32 | — | **Spiky** |
| Baseline 3b5p2f | 78.64 ± 62.36 | — | Normal |
| Baseline 2b4p1f | 59.14 ± 36.01 | — | Normal |
| Baseline 1b3p1f | 64.25 ± 55.46 | — | Normal |

**Finding**: Two NMN runs show extreme gradient instability (mean grad_norm > 8,000 with std > 100,000). These correspond to max spikes of 129M (Mult 2b4p2f) and 14.6M (PreAct 2b4p1f). Despite this, both runs still show learning in episode metrics — the spikes may be rare outliers. However, this warrants investigation:

- Are the gradient spikes correlated with specific environment events (e.g., sudden death)?
- Does `max_grad_norm: 0.5` clipping prevent damage? (The high values suggest it may not be applied to the modulator.)
- Could gradient clipping be applied separately to modulator parameters?

---

### 12.6 Environment Difficulty Analysis

| Env Config | Reward Range | Steps (SS) | TotalDamage (SS) | FoodEaten (SS) | Character |
|---|---|---|---|---|---|
| 2b/4p/1f | -109 to -117 | 99–127 | 34–53 | 1–15 | Short life, low food, moderate damage |
| 1b/3p/1f | -117 to -121 | 109–124 | 51–59 | 7–14 | Short life, low food, higher damage (less cover) |
| 2b/4p/2f | -132 to -136 | 170–178 | 81–86 | 37–41 | Long life, high food, high damage |
| 2b/3p/2f | -133 to -137 | 165–186 | 85–89 | 35–45 | Long life, high food, high damage |
| 3b/5p/2f | -128 to -133 | 176–196 | 75–88 | 40–50 | Long life, most food, variable damage |

**Pattern**: More food (2/quad) leads to dramatically longer episodes (~170-196 vs ~100-127 steps) because the agent can sustain nutrition. But longer episodes mean more cumulative damage. The NMN advantage is largest in **low-food environments** where survival strategy matters most — avoidance vs foraging tradeoff is sharpest.

In high-food environments, the baseline can adopt a simple "eat everything, rest often" strategy that works well without adaptive modulation. The NMN's advantage from damage avoidance is offset by the increased complexity.

---

### 12.7 Critical Findings

#### Finding 1: Multimodal Hub Collapse is Structural (Confirmed)

22 out of 22 NMN configurations across all experiments (v1 + §10 + v2) collapse the multimodal association hub (gamma_multi sigmoid < 0.2%). This is independent of every variable tested: modulation type, grouping size, mod hidden size, environment density, and training duration. The multimodal hub architecture (1152→128→128→128) is not producing features useful enough to justify their variance cost.

**This is the #1 priority to fix.** No amount of modulator tuning will help if the hub it's gating produces nothing useful.

Possible fixes:
- Residual connections from unimodal outputs to the final representation (bypassing the hub)
- Wider hub (128→256 or larger) to reduce information bottleneck
- Skip connections within the hub
- Remove the hub entirely and concatenate unimodal features directly
- Train the hub separately with an auxiliary loss

#### Finding 2: z_memory Clamp Works (Confirmed)

The v1 recommendation to clamp z_memory to [-2, +2] successfully prevents the GRU from being disabled. 8/10 runs maintain z_memory in healthy range. This was the most impactful config change from v1.

#### Finding 3: Temperature Bounds Need Further Tightening

The [0.5, 3.0] bounds are better than v1's [0.1, 10.0] but the modulator still pushes to the ceiling. Recommendation: [0.5, 2.0].

#### Finding 4: PreActivation Does Not Outperform Multiplicative

Across 5 environment configs:
- PreAct wins 2b4p1f (reward), ties or loses the other 4
- PreAct has higher temperature inflation (mean 2.63–2.91 vs Mult 1.92–2.57)
- PreAct's beta term is consistently inhibitory (opposite of design intent)
- PreAct has 4 extra parameters (beta_uni, beta_multi, beta_uni_std, beta_multi_std) that don't improve outcomes

**Recommendation**: Standardize on Multiplicative mode. It's simpler, has healthier temperature, and performs at least as well.

#### Finding 5: NMN Advantage is Small, Inconsistent, and Comes from Damage Avoidance

The NMN beats baseline by 3–5.5 points in 4 of 5 environments, but loses in the densest environment (3b5p2f). The advantage comes from **lower TotalDamage** (less danger and predator damage), not from better food-seeking or adaptive sensory modulation. In low-food environments, the NMN learns a "die quickly but safely" strategy that minimizes the death penalty.

This suggests the NMN's benefit is a mild **regularization effect** (the modulator smooths the loss landscape) rather than the intended adaptive precision-weighting. The modulator is not learning "upweight olfaction when healthy, suppress when injured" — it's learning "suppress the noisy hub, adjust exploration temperature."

---

### 12.8 Updated Experiment Priority List

| Priority | Experiment | Rationale | Effort |
|---|---|---|---|
| **P0** | **Fix multimodal hub architecture** | 22/22 runs collapse. The modulator cannot help if the hub is dead. | Medium-High |
| **P1** | Tighten temp_clip to [0.5, 2.0] | Still hitting 3.0 ceiling in all runs. | Low |
| **P2** | Enable state-dependent noise (olfaction + visual) | Still no environmental pressure for adaptive modulation (v1 §4). | Low |
| **P3** | Investigate gradient spikes | 2 runs show extreme instability (grad_norm > 10^6). | Medium |
| **P4** | Standardize on Multiplicative mode | PreActivation adds complexity without benefit. | Low |
| **P5** | Wait for runs to complete, re-analyze | Current data is mid-training. Trends may shift. | — |

### Recommended Next Run Configuration

```yaml
modulation:
  type: "Multiplicative"
  mod_hidden_size: 16
  grouping_size: 60
  percept_bias_init: 3.0
  memory_bias_init: 0.0
  temp_clip: [0.5, 2.0]       # P1: further tightened
  memory_clip: [-2.0, 2.0]    # Keep working fix

# P2: Enable state-dependent noise
perceptual_noise:
  enabled: true
  modalities:
    olfaction:
      mode: "state_dependent"  # Was: constant
      sigma: 0.15
      injury_noise_scale: 2.0
    visual:
      mode: "state_dependent"  # Was: constant
      sigma: 0.05
      injury_noise_scale: 3.0
    injury:
      mode: "state_dependent"
      sigma: 0.05
      injury_noise_scale: 1.5
```

With a separate experiment to test hub architecture fixes (P0).

---

### 12.9 Summary

The v1-recommended fixes (mod_hidden=16, temp_clip=[0.5,3], memory_clip=[-2,2]) successfully addressed the **z_memory pathology** — recurrent memory is no longer disabled. Temperature inflation is reduced but not eliminated. However, the **core problem persists**: the multimodal association hub is universally collapsed across all 22 NMN configurations tested to date, and the modulator's contribution to performance is limited to a mild regularization effect rather than meaningful adaptive sensory modulation.

The NMN shows small, inconsistent advantages (3–5.5 points in 4/5 environments, -3 points in 1/5) that come from damage avoidance rather than adaptive precision-weighting. Until the multimodal hub architecture is fixed and state-dependent noise creates genuine pressure for sensory adaptation, the modulator cannot learn its intended function.

**Next steps**: Fix the hub (P0), tighten temperature (P1), enable state-dependent noise (P2), then re-evaluate whether neuromodulation provides meaningful adaptive behavior beyond regularization.
