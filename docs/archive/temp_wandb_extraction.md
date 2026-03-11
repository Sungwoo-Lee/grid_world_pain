# Temporary WandB Extraction — 2026-03-09 Runs

## Run 1: Mult 2bush4pred1food (20260309-212615, ql6yd7gf) — RUNNING

### Config Verification
- **Tag**: rppoNMN_128env_2bush4pred1food_gSize60_Multiplicative_modHidden16_tempClip
- **Modulation**: Multiplicative, mod_hidden=16, grouping=60, temp_clip=[0.5,3], memory_clip=[-2,2], percept_bias_init=3
- **Environment**: 10x10, 4 predators, 4 food (1/quad), 8 bushes (2/quad), 8 danger (2/quad), 12 rocks (3/quad), 5 rabbits
- **Agent**: relu, GAE, GRU, hidden=128, lr_actor=0.0005, lr_critic=0.0001, K_epochs=4, gamma=0.95, gae_lambda=0.95, entropy=0.01, eps_clip=0.1, seq_len=128, num_envs=128
- **Config match**: YES — bush=2/quad, pred=4, food=1/quad matches tag "2bush4pred1food"

### Episode Metrics (SS = last 20%)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| Reward | -117.33 ± 2.95 | -116.99 | -177.83 → -116.99 |
| Steps | 127.29 ± 4.47 | 133.06 | 49.00 → 133.06 |
| FoodEaten | 15.27 ± 2.09 | 18.24 | 0.21 → 18.24 |
| DamagePredator | 21.99 ± 2.55 | 24.80 | 63.50 → 24.80 |
| DamageDanger | 30.39 ± 3.42 | 27.43 | 40.31 → 27.43 |
| TotalDamage | 53.16 ± 4.71 | 53.03 | 105.71 → 53.03 |
| PredatorHits | 0.73 ± 0.08 | 0.85 | 2.13 → 0.85 |
| DangerHits | 1.01 ± 0.11 | 0.95 | 1.37 → 0.95 |
| RestCount | 57.58 ± 3.30 | 56.58 | 4.11 → 56.58 |
| MeanDistFood | 2.45 ± 0.08 | 2.43 | 2.73 → 2.43 |
| MeanDistPredator | 5.42 ± 0.14 | 5.31 | 5.02 → 5.31 |
| Term_Injury | 0.069 | 0.053 | 0.685 → 0.053 |
| Term_Starvation | 0.931 | 0.947 | 0.316 → 0.947 |
| Term_MaxSteps | 0.000 | 0.000 | — |

### Loss Metrics (SS)
| Metric | SS | Final |
|---|---|---|
| loss/total | 32.80 ± 4.71 | 36.51 |
| loss/value | 65.61 ± 9.41 | 73.03 |
| loss/policy | 0.0006 ± 0.0006 | 0.0011 |
| loss/entropy | -0.531 ± 0.013 | -0.524 |
| loss/grad_norm | 47.22 ± 27.37 | 60.95 |

### Modulator Metrics (SS)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| gamma_uni_mean | 0.324 ± 0.082 | 0.446 | 3.146 → 0.446 |
| gamma_uni_std | 2.029 ± 0.080 | 2.211 | 0.302 → 2.211 |
| gamma_multi_mean | -9.192 ± 0.207 | -9.557 | 2.921 → -9.557 |
| gamma_multi_std | 1.409 ± 0.035 | 1.441 | 0.280 → 1.441 |
| temperature_mean | 2.542 ± 0.115 | 2.477 | 1.138 → 2.477 |
| temperature_max | 3.000 ± 0.000 | 3.000 | 1.316 → 3.000 (PINNED) |
| temperature_min | 1.201 ± 0.228 | 1.526 | 0.951 → 1.526 |
| z_memory_mean | 0.859 ± 0.105 | 1.096 | 0.236 → 1.096 |
| z_memory_std | 2.053 ± 0.110 | 2.131 | 0.117 → 2.131 |
| modulator/grad_norm | 41.81 ± 25.39 | 55.37 | 0.428 → 55.37 |

---

## Run 2: Mult 3bush5pred2food (20260309-213440, uepbf8w8) — RUNNING

### Config Verification
- **Tag**: rppoNMN_128env_3bush5pred2food_gSize60_Multiplicative_modHidden16_tempClip
- **Config match**: bush=3/quad→12 total, pred=5, food=2/quad→8 total matches tag "3bush5pred2food"

### Episode Metrics (SS)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| Reward | -132.56 ± 4.20 | -124.82 | -174.22 → -124.82 |
| Steps | 195.61 ± 12.45 | 180.78 | 53.35 → 180.78 |
| FoodEaten | 49.63 ± 6.13 | 41.28 | 0.21 → 41.28 |
| DamagePredator | 34.59 ± 4.21 | 30.42 | 60.64 → 30.42 |
| DamageDanger | 51.95 ± 5.66 | 42.28 | 38.44 → 42.28 |
| TotalDamage | 87.89 ± 7.40 | 73.96 | 101.15 → 73.96 |
| PredatorHits | 1.15 ± 0.14 | 1.00 | 2.00 → 1.00 |
| DangerHits | 1.73 ± 0.19 | 1.41 | 1.27 → 1.41 |
| RestCount | 77.22 ± 5.02 | 77.64 | 2.55 → 77.64 |
| MeanDistFood | 1.48 ± 0.08 | 1.45 | 2.02 → 1.45 |
| MeanDistPredator | 5.39 ± 0.16 | 5.21 | 5.55 → 5.21 |
| Term_Injury | 0.113 | 0.060 | 0.660 → 0.060 |
| Term_Starvation | 0.875 | 0.940 | 0.340 → 0.940 |
| Term_MaxSteps | 0.012 | 0.000 | — |

### Loss (SS)
| loss/total | 42.82 ± 4.72 | loss/value | 85.65 ± 9.44 | loss/entropy | -0.535 ± 0.018 | loss/grad_norm | 57.79 ± 233.55 |

### Modulator (SS)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| gamma_uni_mean | -0.490 ± 0.075 | -0.474 | 3.066 → -0.474 |
| gamma_multi_mean | -7.857 ± 0.195 | -8.158 | 3.137 → -8.158 |
| temperature_mean | 2.572 ± 0.105 | 2.678 | 1.141 → 2.678 |
| temperature_max | 3.000 (PINNED) | 3.000 | 1.304 → 3.000 |
| temperature_min | 0.922 ± 0.116 | 0.930 | 0.960 → 0.930 |
| z_memory_mean | 0.267 ± 0.151 | 0.525 | 0.270 → 0.525 |
| z_memory_std | 0.838 ± 0.064 | 0.952 | 0.109 → 0.952 |

---

## Run 3: Mult 2bush3pred2food (20260309-213532, 3q1f0rhs) — RUNNING

### Config Verification
- **Tag**: rppoNMN_128env_2bush3pred2food_gSize60_Multiplicative_modHidden16_tempClip
- **Config match**: bush=2/quad, pred=3, food=2/quad matches tag "2bush3pred2food"

### Episode Metrics (SS)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| Reward | -134.73 ± 4.37 | -145.96 | -176.87 → -145.96 |
| Steps | 186.31 ± 11.56 | 158.95 | 50.67 → 158.95 |
| FoodEaten | 45.31 ± 5.68 | 33.87 | 0.43 → 33.87 |
| DamagePredator | 33.32 ± 3.78 | 41.16 | 57.94 → 41.16 |
| DamageDanger | 54.74 ± 5.51 | 60.75 | 43.24 → 60.75 |
| TotalDamage | 89.39 ± 7.05 | 102.97 | 102.97 → 102.97 |
| PredatorHits | 1.11 ± 0.12 | 1.40 | 1.97 → 1.40 |
| DangerHits | 1.82 ± 0.18 | 1.99 | 1.43 → 1.99 |
| RestCount | 72.24 ± 4.73 | 60.61 | 2.49 → 60.61 |
| Term_Injury | 0.132 | 0.274 | 0.703 → 0.274 |
| Term_Starvation | 0.857 | 0.726 | 0.302 → 0.726 |

### Loss (SS)
| loss/total | 46.02 ± 5.14 | loss/value | 92.04 ± 10.28 | loss/entropy | -0.523 ± 0.017 | loss/grad_norm | 31.88 ± 17.06 |

### Modulator (SS)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| gamma_uni_mean | -1.504 ± 0.073 | -1.557 | 3.080 → -1.557 |
| gamma_multi_mean | -7.041 ± 0.134 | -7.254 | 3.246 → -7.254 |
| temperature_mean | 2.521 ± 0.094 | 2.620 | 1.201 → 2.620 |
| temperature_max | 3.000 (PINNED) | 3.000 | 1.390 → 3.000 |
| temperature_min | 0.793 ± 0.053 | 0.829 | 0.976 → 0.829 |
| z_memory_mean | -0.263 ± 0.047 | -0.285 | 0.288 → -0.285 |
| z_memory_std | 0.986 ± 0.041 | 1.003 | 0.105 → 1.003 |

---

## Run 4: Mult 2bush4pred2food (20260309-220707, qpy3xa8n) — RUNNING

### Config Verification
- **Tag**: rppoNMN_128env_2bush4pred2food_gSize60_Multiplicative_modHidden16_tempClip
- **Config match**: bush=2/quad, pred=4, food=2/quad matches tag "2bush4pred2food"

### Episode Metrics (SS)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| Reward | -131.97 ± 4.29 | -132.69 | -176.42 → -132.69 |
| Steps | 178.07 ± 12.51 | 186.04 | 50.79 → 186.04 |
| FoodEaten | 41.07 ± 6.11 | 44.62 | 0.25 → 44.62 |
| DamagePredator | 32.62 ± 4.13 | 35.67 | 64.60 → 35.67 |
| DamageDanger | 47.11 ± 5.36 | 54.71 | 37.81 → 54.71 |
| TotalDamage | 81.20 ± 7.13 | 91.75 | 104.20 → 91.75 |
| PredatorHits | 1.09 ± 0.13 | 1.19 | 2.17 → 1.19 |
| DangerHits | 1.57 ± 0.18 | 1.77 | 1.30 → 1.77 |
| RestCount | 69.58 ± 5.10 | 76.14 | 2.56 → 76.14 |
| Term_Injury | 0.119 | 0.132 | 0.680 → 0.132 |
| Term_Starvation | 0.879 | 0.868 | 0.321 → 0.868 |

### Loss (SS)
| loss/total | 44.83 ± 5.60 | loss/value | 89.68 ± 11.20 | loss/entropy | -0.526 ± 0.040 | loss/grad_norm | 9051 ± 130910 (UNSTABLE!) |

### Modulator (SS)
| Metric | SS | Final | Trajectory |
|---|---|---|---|
| gamma_uni_mean | 0.256 ± 0.071 | 0.352 | 3.084 → 0.352 |
| gamma_multi_mean | -7.055 ± 0.184 | -7.282 | 3.216 → -7.282 |
| temperature_mean | 2.099 ± 0.170 | 2.413 | 1.100 → 2.413 |
| temperature_max | 2.999 ± 0.008 | 3.000 | 1.278 → 3.000 |
| temperature_min | 0.703 ± 0.046 | 0.705 | 0.943 → 0.705 |
| z_memory_mean | -0.215 ± 0.059 | -0.194 | 0.256 → -0.194 |
| z_memory_std | 0.700 ± 0.041 | 0.722 | 0.093 → 0.722 |
| modulator/grad_norm | 3825 ± 68071 (UNSTABLE!) | 228.09 | — |

---

---

# SUMMARY TABLES — ALL 15 RUNS

## Episode Performance (SS = Steady-State last 20%)

### Grouped by Environment Config

#### Environment: 2bush4pred1food (low density: 1 food/quad, 4 pred)
| Run | Type | Reward (SS) | Steps (SS) | FoodEaten (SS) | TotalDamage (SS) | Term_Injury | Term_Starv |
|---|---|---|---|---|---|---|---|
| 20260309-212615 | Mult | -117.33 ± 2.95 | 127.29 ± 4.47 | 15.27 ± 2.09 | 53.16 ± 4.71 | 0.069 | 0.931 |
| 20260309-220117 | PreAct | **-108.73 ± 2.08** | 98.82 ± 1.48 | 1.07 ± 0.44 | **34.46 ± 3.39** | 0.057 | 0.943 |
| 20260309-222606 | Baseline | -114.19 ± 2.69 | 112.18 ± 3.27 | 7.87 ± 1.45 | 43.70 ± 4.12 | 0.068 | 0.932 |

#### Environment: 2bush4pred2food (medium density: 2 food/quad, 4 pred)
| Run | Type | Reward (SS) | Steps (SS) | FoodEaten (SS) | TotalDamage (SS) | Term_Injury | Term_Starv |
|---|---|---|---|---|---|---|---|
| 20260309-220707 | Mult | -131.97 ± 4.29 | 178.07 ± 12.51 | 41.07 ± 6.11 | 81.20 ± 7.13 | 0.119 | 0.879 |
| 20260309-220618 | PreAct | -135.96 ± 4.39 | 170.24 ± 9.19 | 37.38 ± 4.46 | 85.03 ± 6.61 | 0.138 | 0.860 |
| 20260309-221708 | Baseline | -136.09 ± 4.55 | 173.24 ± 10.55 | 39.05 ± 5.04 | 86.02 ± 6.95 | 0.143 | 0.853 |

#### Environment: 2bush3pred2food (medium density: 2 food/quad, 3 pred)
| Run | Type | Reward (SS) | Steps (SS) | FoodEaten (SS) | TotalDamage (SS) | Term_Injury | Term_Starv |
|---|---|---|---|---|---|---|---|
| 20260309-213532 | Mult | -134.73 ± 4.37 | 186.31 ± 11.56 | 45.31 ± 5.68 | 89.39 ± 7.05 | 0.132 | 0.857 |
| 20260309-220432 | PreAct | -132.89 ± 4.36 | 183.68 ± 10.41 | 43.85 ± 5.09 | 84.76 ± 6.77 | 0.123 | 0.873 |
| 20260309-222226 | Baseline | -136.97 ± 4.53 | 164.76 ± 10.96 | 35.18 ± 5.21 | 86.64 ± 7.27 | 0.162 | 0.834 |

#### Environment: 3bush5pred2food (high density: 2 food/quad, 5 pred, 3 bush)
| Run | Type | Reward (SS) | Steps (SS) | FoodEaten (SS) | TotalDamage (SS) | Term_Injury | Term_Starv |
|---|---|---|---|---|---|---|---|
| 20260309-213440 | Mult | -132.56 ± 4.20 | 195.61 ± 12.45 | 49.63 ± 6.13 | 87.89 ± 7.40 | 0.113 | 0.875 |
| 20260309-220237 | PreAct | -131.19 ± 4.26 | 178.06 ± 9.81 | 40.90 ± 4.75 | 80.60 ± 6.55 | 0.115 | 0.882 |
| 20260309-222427 | Baseline | **-128.38 ± 4.09** | 176.39 ± 9.57 | 39.92 ± 4.68 | **74.67 ± 6.49** | 0.099 | 0.899 |

#### Environment: 1bush3pred1food (low density: 1 food/quad, 3 pred, 1 bush)
| Run | Type | Reward (SS) | Steps (SS) | FoodEaten (SS) | TotalDamage (SS) | Term_Injury | Term_Starv |
|---|---|---|---|---|---|---|---|
| 20260309-222923 | Mult+BM | -116.72 ± 2.92 | 109.07 ± 3.50 | 6.80 ± 1.60 | 51.00 ± 4.48 | 0.090 | 0.910 |
| 20260309-223610 | PreAct+BM | -119.59 ± 3.04 | 114.50 ± 3.76 | 9.64 ± 1.66 | 56.18 ± 4.76 | 0.102 | 0.898 |
| 20260309-222800 | Baseline | -120.53 ± 3.44 | 123.73 ± 4.44 | 14.02 ± 1.94 | 59.11 ± 5.09 | 0.094 | 0.906 |

## Modulator Metrics Summary (SS)

### Multiplicative Runs
| Env | gamma_uni (sigmoid) | gamma_multi (sigmoid) | temp_mean | temp_max | z_mem_mean | z_mem_std |
|---|---|---|---|---|---|---|
| 2b4p1f | 0.324 (58%) | -9.192 (0.01%) | 2.542 | 3.000 PIN | 0.859 | 2.053 |
| 3b5p2f | -0.490 (38%) | -7.857 (0.04%) | 2.572 | 3.000 PIN | 0.267 | 0.838 |
| 2b3p2f | -1.504 (18%) | -7.041 (0.09%) | 2.521 | 3.000 PIN | -0.263 | 0.986 |
| 2b4p2f | 0.256 (56%) | -7.055 (0.09%) | 2.099 | 2.999 | -0.215 | 0.700 |
| 1b3p1f+BM | 0.432 (61%) | -6.389 (0.17%) | 1.917 | 2.980 | 0.212 | 0.641 |

### PreActivation Runs
| Env | gamma_uni (sigmoid) | gamma_multi (sigmoid) | temp_mean | temp_max | z_mem_mean | z_mem_std | beta_uni | beta_multi |
|---|---|---|---|---|---|---|---|---|
| 2b4p1f | 1.238 (78%) | -7.060 (0.09%) | 2.906 | 3.000 PIN | 0.191 | 0.666 | -2.236 | -0.574 |
| 3b5p2f | 2.067 (89%) | -7.379 (0.06%) | 2.625 | 3.000 PIN | -0.206 | 0.755 | -2.827 | -0.248 |
| 2b3p2f | 1.725 (85%) | -7.073 (0.08%) | 2.740 | 3.000 PIN | 0.088 | 0.909 | -2.141 | -0.139 |
| 2b4p2f | 1.845 (86%) | -7.249 (0.07%) | 2.719 | 3.000 PIN | 1.330 | 2.179 | -1.678 | -0.154 |
| 1b3p1f+BM | 1.789 (86%) | -6.786 (0.11%) | 2.877 | 3.000 PIN | 0.554 | 1.415 | -0.789 | -0.166 |

## Key Observations for v2 Diagnosis

### 1. Multimodal Hub Collapse — STILL UNIVERSAL
All 10 NMN runs: gamma_multi_mean ranges from -6.39 to -9.19 (sigmoid 0.01%–0.17%). Universal collapse confirmed again.

### 2. Temperature — Pinned at Ceiling in Most Runs
All runs have temp_max = 3.000 (the new tighter ceiling). temp_mean ranges from 1.92 to 2.91.
PreActivation runs have HIGHER temp_mean (2.63–2.91) than Multiplicative (1.92–2.57).

### 3. z_memory — MUCH HEALTHIER than v1 diagnosis
Most runs: z_memory_mean between -0.26 and +0.86. The [-2,2] clamp is working.
Exception: PreAct 2b4p2f (z_mem=1.33, std=2.18) — approaching boundary.
Exception: Mult 2b4p1f (z_mem=0.86, std=2.05) — high variance.

### 4. Unimodal Gating — PreActivation Preserves Better
- Multiplicative: gamma_uni ranges from -1.50 to +0.43 (18%–61% sigmoid). Two runs suppress unimodal.
- PreActivation: gamma_uni ranges from 1.24 to 2.07 (78%–89% sigmoid). Near pass-through.

### 5. Environment Difficulty Impact
- Low-density envs (1food, fewer pred): shorter episodes (~100-130 steps), lower damage, lower reward magnitude
- High-density envs (2food, more pred): longer episodes (~170-196 steps), higher damage, higher food eaten
- More food → more reason to stay alive → more exposure to damage

### 6. NMN vs Baseline — Mixed Results by Environment
- 2b4p1f: PreAct WINS (-108.73 vs baseline -114.19, +5.5 pts)
- 2b4p2f: Mult slightly WINS (-131.97 vs baseline -136.09, +4.1 pts)
- 2b3p2f: Both NMN slightly better (Mult -134.73, PreAct -132.89 vs baseline -136.97)
- 3b5p2f: Baseline WINS (-128.38 vs Mult -132.56, PreAct -131.19)
- 1b3p1f: Mult+BM slightly WINS (-116.72 vs baseline -120.53, +3.8 pts)

### 7. Gradient Instability — NEW Issue
- Mult 2b4p2f: loss/grad_norm SS = 9051 ± 130910, modulator/grad_norm = 3825 ± 68071
- PreAct 2b4p1f: loss/grad_norm SS = 8596 ± 329376
- PreAct 3b5p2f: loss/grad_norm SS = 50.84 ± 173.51, final spike to 3132
- Several runs show occasional extreme gradient spikes (>10^6)

### 8. PreActivation Beta Behavior
- beta_uni is consistently negative (-0.79 to -2.83): inhibitory — raising activation thresholds
- beta_multi is mildly negative (-0.14 to -0.57): slight inhibition
- Both decrease over training — the modulator learns to inhibit rather than disinhibit
