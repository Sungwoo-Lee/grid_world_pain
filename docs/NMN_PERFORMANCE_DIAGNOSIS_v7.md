# NMN Performance Diagnosis v7: LayerNorm as Orthogonal Variable — The P0 Experiment

> **Status**: COMPLETE
> **Date**: 2026-03-19
> **Author**: Claude (analysis), Sungwoo (experiment design & training)
> **Related**: [NMN_PERFORMANCE_DIAGNOSIS_v6.md](NMN_PERFORMANCE_DIAGNOSIS_v6.md) (v6, FiLM modulation comparison), [LAYERNORM_ORTHOGONAL_REFACTOR_PLAN.md](LAYERNORM_ORTHOGONAL_REFACTOR_PLAN.md) (refactor that enabled this experiment)

---

## 1. Research Question

v6 established that FiLM (with LayerNorm) structurally prevents multimodal hub collapse and dramatically improves GAE performance (+88–99 steps over FiLMNoNorm). However, v6 had a critical confound: **LayerNorm was entangled with modulation type** — only FiLM had LayerNorm. The [LAYERNORM_ORTHOGONAL_REFACTOR_PLAN.md](LAYERNORM_ORTHOGONAL_REFACTOR_PLAN.md) refactored `use_layer_norm` into an orthogonal config axis, enabling this v7 experiment to fill the complete 2D factorial: `{null, FiLM, PreActivation, Multiplicative} × {LN, NoLN}`.

This experiment answers two questions:

1. **Is LayerNorm or the modulation type the primary driver of v6's FiLM advantage?** (v6 Finding 2 identified LN as critical but could not disentangle it from FiLM)
2. **Does the modulator add value beyond what LN alone provides?** (v6 P0 — the highest-priority recommended experiment)

> **H₀** (null): Modulation type (FiLM, PreActivation, Multiplicative, or null) has no effect on survival performance when LayerNorm is held constant.
> **H₁a**: LayerNorm prevents gate collapse for sigmoid-bounded modes (PreActivation, Multiplicative), not just FiLM.
> **H₁b**: LayerNorm alone (unmodulated + LN) explains most of FiLM's v6 survival advantage.
> **H₁c**: Adding a modulator to an LN-equipped network improves survival beyond the unmodulated LN baseline.
> **H₁d**: The MC advantage over GAE persists across all configurations, including unmodulated baselines.
> **H₁e**: Gate dynamics differ qualitatively when LN is added to sigmoid-bounded modes.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| `modulation.type` | FiLM, PreActivation, Multiplicative, null | Compare all modulation types + unmodulated baseline |
| `agent.use_layer_norm` | true, false | Orthogonal LN axis (enabled by refactor) |
| `return_mode` | MC, GAE | Consistent with v4–v6 |
| `grouping_size` | 1, 4 | For modulated runs only; bracket the range per v5 |

### 2.2 Controlled Variables

```yaml
modulation:
  mod_hidden_size: 16
  percept_bias_init: 3.0     # overridden to 1.0 for FiLM internally
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
  food_nutrition_gain: 6    # NutGain6

perceptual_noise:
  enabled: false             # NoNoise
```

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| v6 Multiplicative runs trained ~5× longer (~20B steps) | Cross-version Multiplicative comparisons | Med | Compare v7 runs at matched training duration (~3–4B steps); v6 Multiplicative data noted where used |
| Single seed per config | All | Med | 2 MC_Unmod_NoLN seeds provide some variance estimate |
| All v7 runs use refactored code; v6 used pre-refactor code | Cross-version comparisons | Low | FiLM+LN re-runs in v7 should match v6 FiLM behavior — used as sanity check |
| v7 runs still in progress (state: running) | All | Low | Steady-state metrics are past convergence for MC; GAE still improving in some runs |

## 3. Run Inventory

### v7 Modulated Runs (12 runs — all use_layer_norm: true)

| # | Datetime ID | WandB ID | Return | ModType | gSize | Timesteps | Label |
|---|------------|----------|--------|---------|-------|-----------|-------|
| 1 | 20260318-173952 | `1f5he70x` | GAE | PreActivation | 1 | ~4.6B | GAE_PreAct_LN_g1 |
| 2 | 20260318-174122 | `9zart4w7` | GAE | PreActivation | 4 | ~4.4B | GAE_PreAct_LN_g4 |
| 3 | 20260318-174252 | `rnlt09ke` | MC | PreActivation | 1 | ~4.6B | MC_PreAct_LN_g1 |
| 4 | 20260318-174748 | `qplp0i9q` | MC | PreActivation | 4 | ~4.6B | MC_PreAct_LN_g4 |
| 5 | 20260318-174926 | `e7qrk420` | GAE | Multiplicative | 1 | ~3.7B | GAE_Mult_LN_g1 |
| 6 | 20260318-175029 | `9qobjpjn` | GAE | Multiplicative | 4 | ~3.7B | GAE_Mult_LN_g4 |
| 7 | 20260318-175143 | `ptm2mmrd` | MC | Multiplicative | 1 | ~3.7B | MC_Mult_LN_g1 |
| 8 | 20260318-175307 | `j2yzqcuu` | MC | Multiplicative | 4 | ~3.6B | MC_Mult_LN_g4 |
| 9 | 20260318-175427 | `wwssq5o2` | GAE | FiLM | 1 | ~3.6B | GAE_FiLM_LN_g1 |
| 10 | 20260318-175540 | `xwhafw5g` | GAE | FiLM | 4 | ~3.4B | GAE_FiLM_LN_g4 |
| 11 | 20260318-175802 | `1seg2jqd` | MC | FiLM | 1 | ~3.6B | MC_FiLM_LN_g1 |
| 12 | 20260318-175923 | `oqfjwaou` | MC | FiLM | 4 | ~3.3B | MC_FiLM_LN_g4 |

### v7 Unmodulated Baselines (5 runs — the P0 experiment)

| # | Datetime ID | WandB ID | Return | LN | Timesteps | Label |
|---|------------|----------|--------|-----|-----------|-------|
| 13 | 20260318-180254 | `qb02ygg8` | GAE | true | ~3.4B | GAE_Unmod_LN |
| 14 | 20260318-180346 | `u8d0yc7f` | GAE | false | ~3.7B | GAE_Unmod_NoLN |
| 15 | 20260318-180506 | `h0n6ak0w` | MC | true | ~4.6B | MC_Unmod_LN |
| 16 | 20260318-180551 | `o431n7ig` | MC | false | ~4.9B | MC_Unmod_NoLN_1 |
| 17 | 20260318-180735 | `5s0ah0pu` | MC | false | ~3.8B | MC_Unmod_NoLN_2 |

---

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

#### 4.1.1 Complete 2D Factorial — Survival Steps (Steady-State)

**MC Runs — Ranked by Survival Steps:**

| Label | Steps SS | Reward SS | FoodEaten SS | TotalDamage SS | Term_Injury | Term_Starvation | Term_MaxSteps |
|-------|----------|-----------|-------------|----------------|-------------|-----------------|---------------|
| MC_PreAct_LN_g1 | **360.57 ± 10.22** | -133.04 | 55.44 | 113.95 | 17.0% | 45.3% | 37.7% |
| MC_Mult_LN_g4 | 360.12 ± 10.48 | -130.53 | 55.08 | 108.59 | 15.1% | 47.9% | 37.0% |
| MC_FiLM_LN_g1 | 357.26 ± 10.98 | -131.91 | 54.89 | 111.14 | 16.6% | 46.6% | 36.8% |
| MC_PreAct_LN_g4 | 357.20 ± 9.94 | -132.90 | 54.57 | 115.29 | 17.3% | 46.4% | 36.3% |
| MC_Mult_LN_g1 | 356.75 ± 10.41 | -130.03 | 54.12 | 109.44 | 14.4% | 49.3% | 36.3% |
| MC_FiLM_LN_g4 | 355.56 ± 10.80 | -131.48 | 54.27 | 113.30 | 15.9% | 48.0% | 36.1% |
| **MC_Unmod_LN** | **352.49 ± 10.45** | -131.34 | 53.70 | 110.27 | 14.8% | 50.6% | 34.6% |
| MC_Unmod_NoLN_1 | 344.26 ± 10.69 | -131.79 | 52.15 | 112.05 | 15.3% | 51.9% | 32.9% |
| MC_Unmod_NoLN_2 | 343.82 ± 10.57 | -131.73 | 51.95 | 111.66 | 14.9% | 52.8% | 32.3% |

Note: v6 MC runs (for reference): MC_FiLM_g1 = 361, MC_FiLM_g4 = 360, MC_PreAct_g1 = 349, MC_PreAct_g4 = 353, MC_Mult_g1 = 363*, MC_Mult_g4 = 370* (*trained ~5× longer).

**GAE Runs — Ranked by Survival Steps:**

| Label | Steps SS | Reward SS | FoodEaten SS | TotalDamage SS | Term_Injury | Term_Starvation | Term_MaxSteps |
|-------|----------|-----------|-------------|----------------|-------------|-----------------|---------------|
| GAE_Mult_LN_g1 | **321.13 ± 12.56** | -131.21 | 54.02 | 111.11 | 15.8% | 57.2% | 27.1% |
| **GAE_Unmod_LN** | **313.68 ± 9.35** | -130.08 | 50.58 | 106.43 | 14.1% | 60.4% | 25.5% |
| GAE_PreAct_LN_g1 | 307.15 ± 10.15 | -133.98 | 48.96 | 111.39 | 17.4% | 58.5% | 24.1% |
| GAE_FiLM_LN_g4 | 293.76 ± 9.78 | -132.17 | 46.57 | 107.37 | 16.5% | 62.7% | 20.8% |
| GAE_Mult_LN_g4 | 293.26 ± 12.17 | -132.89 | 46.58 | 107.45 | 16.5% | 61.6% | 21.8% |
| GAE_FiLM_LN_g1 | 289.49 ± 14.78 | -132.42 | 44.93 | 106.10 | 16.4% | 64.0% | 19.6% |
| GAE_PreAct_LN_g4 | 288.15 ± 9.30 | -134.01 | 44.44 | 109.63 | 17.4% | 63.1% | 19.5% |
| GAE_Unmod_NoLN | 256.89 ± 15.45 | -134.57 | 37.14 | 97.99 | 17.4% | 68.9% | 13.7% |

Note: v6 GAE runs (for reference): GAE_FiLM_g1 = 294, GAE_FiLM_g4 = 277, GAE_PreAct_g1 = 201, GAE_PreAct_g4 = 242, GAE_Mult_g1 = 238*, GAE_Mult_g4 = 250*† (*trained ~5× longer, †catastrophic collapse).

#### 4.1.2 The P0 Answer — Does the Modulator Add Value?

**MC (comparing against MC_Unmod_LN = 352.49 steps):**

| Config | Steps SS | Δ vs Unmod_LN | Interpretation |
|--------|----------|---------------|----------------|
| MC_PreAct_LN_g1 | 360.57 | **+8.08** | Small positive |
| MC_Mult_LN_g4 | 360.12 | **+7.63** | Small positive |
| MC_FiLM_LN_g1 | 357.26 | **+4.77** | Marginal |
| MC_PreAct_LN_g4 | 357.20 | **+4.71** | Marginal |
| MC_Mult_LN_g1 | 356.75 | **+4.26** | Marginal |
| MC_FiLM_LN_g4 | 355.56 | **+3.07** | Marginal |

> **Verdict on H₁c for MC: WEAKLY SUPPORTED — the modulator adds +3 to +8 steps over the LN-only baseline.** The best modulated config (MC_PreAct_LN_g1, 361 steps) is only 2.3% better than no modulator (352 steps). Within MC, the modulator provides a real but small benefit.

**GAE (comparing against GAE_Unmod_LN = 313.68 steps):**

| Config | Steps SS | Δ vs Unmod_LN | Interpretation |
|--------|----------|---------------|----------------|
| GAE_Mult_LN_g1 | 321.13 | **+7.45** | Small positive |
| GAE_PreAct_LN_g1 | 307.15 | **-6.53** | Slightly worse |
| GAE_FiLM_LN_g4 | 293.76 | **-19.92** | Worse |
| GAE_Mult_LN_g4 | 293.26 | **-20.42** | Worse |
| GAE_FiLM_LN_g1 | 289.49 | **-24.19** | Worse |
| GAE_PreAct_LN_g4 | 288.15 | **-25.53** | Worse |

> **Verdict on H₁c for GAE: NOT SUPPORTED — the modulator generally hurts GAE performance compared to the LN-only baseline.** Only GAE_Mult_LN_g1 slightly exceeds the unmodulated baseline. 5 of 6 modulated GAE+LN configs perform worse than no modulator. The unmodulated baseline with LN (314 steps) exceeds v6's best GAE result (FiLM_g1 = 294 steps).

#### 4.1.3 LayerNorm Effect — Isolated

**Unmodulated baseline (pure LN effect, no modulator):**

| Return | Unmod_LN Steps | Unmod_NoLN Steps | Δ LN Effect |
|--------|---------------|-----------------|-------------|
| MC | 352.49 | 344.04† | **+8.45** |
| GAE | 313.68 | 256.89 | **+56.79** |

†Average of MC_Unmod_NoLN_1 (344.26) and MC_Unmod_NoLN_2 (343.82).

> **Verdict on H₁b: STRONGLY SUPPORTED — LayerNorm alone explains most of FiLM's v6 advantage.**
> - **MC**: LN alone gives 352 steps — within the range of all v6 modulated MC configs (344–362 for matched training durations). v6 MC_FiLM_g1 was 361, of which ~352 is attributable to LN alone.
> - **GAE**: LN alone gives 314 steps — **exceeding** v6's best result (GAE_FiLM_g1 = 294). The entire v6 GAE_FiLM advantage (+99 over FiLMNoNorm, +56 over Multiplicative) is attributable to LayerNorm, not to the FiLM modulation mechanism.
> - **The LN effect is ~7× larger for GAE (+57 steps) than MC (+8 steps)**, consistent with v6's finding that LN is critical for GAE's value function stability.

#### 4.1.4 Does LayerNorm Prevent Gate Collapse for Sigmoid-Bounded Modes?

**Multimodal gate (gamma_multi) — v7 (all +LN) vs v6 (no LN for PreAct/Mult):**

| ModType | v7 MC_g1 +LN | v6 MC_g1 (no LN) | v7 MC_g4 +LN | v6 MC_g4 (no LN) |
|---------|-------------|-------------------|-------------|-------------------|
| **PreActivation** | **+2.336** (σ≈91%) | -4.137 (σ≈1.6%) | **+4.236** (σ≈99%) | -5.519 (σ≈0.4%) |
| **Multiplicative** | **+0.127** (σ≈53%) | -6.133 (σ≈0.2%) | **-0.745** (σ≈32%) | -9.572 (σ<0.01%) |
| **FiLM** | +1.262 | +1.153 (v6) | +1.052 | +0.740 (v6) |

| ModType | v7 GAE_g1 +LN | v6 GAE_g1 (no LN) | v7 GAE_g4 +LN | v6 GAE_g4 (no LN) |
|---------|-------------|---------------------|-------------|---------------------|
| **PreActivation** | **+1.590** (σ≈83%) | -2.124 (σ≈10.7%) | **+1.887** (σ≈87%) | -5.084 (σ≈0.6%) |
| **Multiplicative** | **+0.659** (σ≈66%) | -3.827 (σ≈2.1%) | **+1.397** (σ≈80%) | -8.555 (σ<0.01%) |
| **FiLM** | +0.757 | +0.716 (v6) | +0.762 | +0.782 (v6) |

> **Verdict on H₁a: STRONGLY SUPPORTED — LayerNorm prevents gate collapse for ALL modulation types, including sigmoid-bounded modes.**
> - **PreActivation + LN**: gamma_multi ranges from +1.59 to +4.24 (sigmoid 83–99%). Compare v6 PreActivation without LN: -2.1 to -5.5 (sigmoid 0.4–10.7%). **Complete reversal from collapsed to fully open.** MC_PreAct_LN_g4 reaches gamma_multi = 4.24 (sigmoid ≈ 99%) — the highest gate opening of any sigmoid-bounded run in any version.
> - **Multiplicative + LN**: gamma_multi ranges from -0.75 to +1.40 (sigmoid 32–80%). Compare v6 Multiplicative without LN: -3.8 to -9.6 (sigmoid <0.01–2.1%). **Dramatic improvement from deeply collapsed to functional.** However, MC_Mult_LN_g4 still shows moderate attenuation (sigma ≈ 32%) — not fully healthy.
> - **FiLM + LN**: v7 values closely match v6, confirming the refactor preserved behavior.
> - **The key insight**: Gate collapse was not caused by sigmoid bounding per se, but by the **interaction** of sigmoid bounding with unnormalized features. LayerNorm breaks the death spiral by ensuring consistent feature magnitudes.

#### 4.1.5 MC vs GAE

| Config | MC Steps | GAE Steps | Δ MC–GAE |
|--------|----------|-----------|----------|
| Unmod_LN | 352.49 | 313.68 | **+38.81** |
| Unmod_NoLN | 344.04 | 256.89 | **+87.15** |
| PreAct_LN_g1 | 360.57 | 307.15 | **+53.42** |
| Mult_LN_g1 | 356.75 | 321.13 | **+35.62** |
| FiLM_LN_g1 | 357.26 | 289.49 | **+67.77** |

> **Verdict on H₁d: SUPPORTED — MC consistently outperforms GAE.** The MC–GAE gap ranges from +36 to +87 steps. **LN alone narrows the gap from 87 (NoLN) to 39 (LN)** — a 55% reduction. This confirms that much of GAE's disadvantage is due to critic instability, which LN alleviates.

### 4.2 Secondary Metrics

#### Value/Critic Loss

| Label | loss/value SS |
|-------|-------------|
| MC_Unmod_LN | 0.292 |
| MC_Unmod_NoLN_1 | 0.292 |
| MC_PreAct_LN_g1 | 0.296 |
| MC_Mult_LN_g1 | 0.293 |
| MC_FiLM_LN_g1 | 0.298 |
| GAE_Unmod_LN | **68.67** |
| GAE_Unmod_NoLN | **80.83** |
| GAE_PreAct_LN_g1 | 74.30 |
| GAE_Mult_LN_g1 | **69.43** |
| GAE_FiLM_LN_g1 | 72.92 |

- MC value loss: ~0.29 across all configs (no effect of LN or modulation)
- GAE critic loss: LN reduces from 80.8 (NoLN) to 68.7 (LN) for unmodulated. Modulated configs range 69.4–76.4. The modulator does NOT reduce critic loss below the unmodulated LN baseline.
- v6 comparison: GAE_FiLM (v6) = 75.2; v7 GAE_FiLM_LN_g1 = 72.9 — similar.

#### Gradient Norms

| Label | loss/grad_norm SS | modulator/grad_norm SS |
|-------|------------------|----------------------|
| MC_Unmod_LN | 0.22 | — |
| MC_Unmod_NoLN_1 | 0.46 | — |
| MC_PreAct_LN_g1 | 0.24 | 0.13 |
| MC_Mult_LN_g1 | 0.23 | 0.04 |
| MC_FiLM_LN_g1 | 0.24 | 0.12 |
| GAE_Unmod_LN | 1,705† | — |
| GAE_Unmod_NoLN | 141.4 | — |
| GAE_PreAct_LN_g1 | 302.8 | 163.7 |
| GAE_Mult_LN_g1 | 35.6 | 4.91 |
| GAE_FiLM_LN_g1 | 241.4 | 137.4 |

†GAE_Unmod_LN has an extreme grad_norm outlier (max 3.35M). The mean is inflated; final value is 23.8.

- MC gradient norms are small and stable across all configs.
- GAE gradient norms are ~100–1000× higher, explaining GAE instability.
- GAE_Mult_LN has notably lower gradient norms (35.6 policy, 4.9 modulator) compared to other modulated GAE configs.

### 4.3 Diagnostic Metrics

#### 4.3.1 Gate Dynamics — Full v7 Summary

**Multimodal Gate (gamma_multi_mean) — Steady-State:**

| ModType | MC_g1 | MC_g4 | GAE_g1 | GAE_g4 |
|---------|-------|-------|--------|--------|
| PreActivation +LN | **+2.336** (σ≈91%) | **+4.236** (σ≈99%) | **+1.590** (σ≈83%) | **+1.887** (σ≈87%) |
| Multiplicative +LN | +0.127 (σ≈53%) | -0.745 (σ≈32%) | **+0.659** (σ≈66%) | **+1.397** (σ≈80%) |
| FiLM +LN | **+1.262** | **+1.052** | **+0.757** | **+0.762** |

**Unimodal Gate (gamma_uni_mean) — Steady-State:**

| ModType | MC_g1 | MC_g4 | GAE_g1 | GAE_g4 |
|---------|-------|-------|--------|--------|
| PreActivation +LN | +1.870 (σ≈87%) | **+2.949** (σ≈95%) | +1.768 (σ≈85%) | **+2.501** (σ≈92%) |
| Multiplicative +LN | +1.315 (σ≈79%) | +0.779 (σ≈69%) | +1.919 (σ≈87%) | +2.116 (σ≈89%) |
| FiLM +LN | +1.204 | +1.467 | +1.206 | +1.405 |

**Key observations:**
1. **PreActivation + LN shows the healthiest gates** — gamma_multi = +2.3 to +4.2, far exceeding even FiLM (+0.8 to +1.3). MC_PreAct_LN_g4 reaches 4.24, meaning the sigmoid gate operates at ~99% — essentially transparent.
2. **Multiplicative + LN shows mixed results** — GAE Multiplicative gates are healthy (66–80%), but MC_Mult_LN_g4 attenuates to 32%. The Multiplicative mode still tends toward attenuation but far less than without LN.
3. **FiLM + LN in v7 closely matches v6 FiLM** — confirming the refactor preserved behavior (v7: 0.76–1.26 vs v6: 0.72–1.15).

#### 4.3.2 Gate Variance

| Label | gamma_multi_std | gamma_uni_std |
|-------|----------------|---------------|
| MC_PreAct_LN_g1 | **2.118** | **2.897** |
| MC_PreAct_LN_g4 | **2.908** | **3.497** |
| MC_Mult_LN_g1 | 2.141 | 3.211 |
| MC_Mult_LN_g4 | 1.544 | 2.497 |
| MC_FiLM_LN_g1 | 1.072 | 1.681 |
| MC_FiLM_LN_g4 | 0.720 | 1.389 |

PreActivation + LN shows the **highest gate variance** of any modulation type — individual neurons are modulated very differently. Combined with the high mean (sigmoid ≈ 91–99%), this indicates rich, selective modulation. FiLM + LN has lower variance, suggesting more uniform modulation.

#### 4.3.3 Temperature

| Label | temp_mean SS | temp_min SS |
|-------|-------------|-------------|
| MC_PreAct_LN_g1 | 2.903 | 1.416 |
| MC_PreAct_LN_g4 | **3.000** | **3.000** |
| MC_Mult_LN_g1 | **3.000** | 2.965 |
| MC_Mult_LN_g4 | **3.000** | **3.000** |
| MC_FiLM_LN_g1 | 2.855 | 1.425 |
| MC_FiLM_LN_g4 | 2.950 | 1.585 |
| GAE_PreAct_LN_g1 | 2.766 | 1.132 |
| GAE_Mult_LN_g1 | 2.461 | 0.978 |
| GAE_FiLM_LN_g1 | 2.474 | 1.207 |

FiLM and PreActivation_g1 maintain temperatures below the 3.0 ceiling (temp_min ≈ 1.1–1.6), suggesting genuine state-dependent temperature modulation. Multiplicative MC and PreAct_g4 saturate at 3.0.

#### 4.3.4 Beta (Additive Term) — PreActivation + LN and FiLM + LN

| Label | beta_multi_mean | beta_uni_mean |
|-------|----------------|---------------|
| MC_PreAct_LN_g1 | -0.560 | -1.349 |
| MC_PreAct_LN_g4 | -1.149 | -1.686 |
| MC_FiLM_LN_g1 | -0.260 | -1.251 |
| MC_FiLM_LN_g4 | -0.348 | -2.166 |
| GAE_PreAct_LN_g1 | -0.621 | -1.237 |
| GAE_FiLM_LN_g1 | -0.394 | -1.018 |

All beta values are negative, acting as a subtractive bias on the pre-activation features. The magnitude is larger for unimodal (−1.0 to −2.2) than multimodal (−0.3 to −1.1), suggesting the modulator applies a stronger offset to individual sensory channels.

### 4.4 Learning Dynamics

#### 4.4.1 Survival Steps Over Training

**MC Runs (10-window means):**

| Window | MC_Unmod_LN | MC_Unmod_NoLN | MC_PreAct_LN_g1 | MC_Mult_LN_g1 | MC_FiLM_LN_g1 |
|--------|------------|--------------|----------------|---------------|---------------|
| 1 | 291 | 284 | 289 | 288 | — |
| 3 | 329 | 320 | 341 | 329 | — |
| 5 | 341 | 334 | 352 | 344 | — |
| 7 | 348 | 341 | 355 | 351 | — |
| 10 | 353 | 346 | 360 | 359 | — |

All MC runs show smooth convergence. MC_Unmod_LN converges ~8 steps below modulated configs. MC_Unmod_NoLN runs ~8 steps below MC_Unmod_LN, confirming the LN effect.

**GAE Runs (10-window means):**

| Window | GAE_Unmod_LN | GAE_Unmod_NoLN | GAE_PreAct_LN_g1 | GAE_Mult_LN_g1 | GAE_FiLM_LN_g1 |
|--------|-------------|---------------|-----------------|----------------|----------------|
| 1 | 165 | 146 | — | — | — |
| 3 | 273 | 225 | — | — | — |
| 5 | 296 | 244 | — | — | — |
| 7 | 308 | 245 | — | — | — |
| 10 | 314 | 260 | — | — | — |

GAE_Unmod_LN shows continuous improvement throughout training, reaching 314 steps and still climbing. GAE_Unmod_NoLN plateaus around 250–260. The LN effect opens up at window 3 (~50 steps gap) and stabilizes.

#### 4.4.2 Multimodal Gate Timeseries — PreActivation + LN (Most Dramatic Change)

**MC_PreAct_LN gamma_multi (10 windows):**

| Window | MC_PreAct_LN_g1 | MC_PreAct_LN_g4 |
|--------|-----------------|-----------------|
| 1 | +3.045 (σ≈95%) | +2.805 (σ≈94%) |
| 3 | +2.428 (σ≈92%) | +2.275 (σ≈91%) |
| 5 | +2.194 (σ≈90%) | +2.778 (σ≈94%) |
| 7 | +2.241 (σ≈90%) | +3.514 (σ≈97%) |
| 10 | +2.324 (σ≈91%) | **+4.519** (σ≈99%) |

These are **positive and stable throughout training** — a complete reversal from v6 PreActivation (no LN), which showed monotonic collapse to -4.1 to -5.5 (sigmoid 0.4–1.6%). MC_PreAct_LN_g4 actually **increases** from 2.8 to 4.5 over training — the gates open wider as the modulator learns.

**MC_Mult_LN gamma_multi (10 windows):**

| Window | MC_Mult_LN_g1 | MC_Mult_LN_g4 |
|--------|--------------|--------------|
| 1 | +2.550 (σ≈93%) | +2.294 (σ≈91%) |
| 3 | +1.127 (σ≈76%) | +0.201 (σ≈55%) |
| 5 | +0.746 (σ≈68%) | -0.314 (σ≈42%) |
| 7 | +0.502 (σ≈62%) | -0.393 (σ≈40%) |
| 10 | +0.064 (σ≈52%) | -0.762 (σ≈32%) |

Multiplicative + LN shows a **gradual decline** but stabilizes above the deeply collapsed v6 values (-6 to -10). MC_Mult_LN_g1 stays near 50% (identity) while MC_Mult_LN_g4 attenuates to 32%. The decline is slow and bounded — not the exponential death spiral seen without LN.

---

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — LayerNorm is the primary driver of performance, not the modulation type (HIGH confidence)**
What: The unmodulated baseline with LN (MC: 352, GAE: 314 steps) is competitive with or exceeds all modulated configurations. For GAE, 5 of 6 modulated+LN configs actually perform *worse* than the LN-only baseline.
Why: LayerNorm standardizes pre-activation features to zero-mean, unit-variance, providing the network with well-conditioned representations for learning. This is a general deep learning benefit, independent of modulation. The modulator introduces additional parameters and complexity that can interfere with GAE's already-unstable critic learning.
Evidence: MC_Unmod_LN (352) is within 8 steps of the best modulated MC config (361). GAE_Unmod_LN (314) exceeds GAE_FiLM_LN_g1 (289) by 25 steps — the modulator hurts GAE.
Confidence: HIGH — replicated across MC and GAE with consistent pattern.

**Finding 2 — LayerNorm prevents gate collapse for ALL modulation types, including sigmoid-bounded (HIGH confidence)**
What: PreActivation + LN maintains gamma_multi at +1.6 to +4.2 (sigmoid 83–99%). Multiplicative + LN maintains gamma_multi at -0.7 to +1.4 (sigmoid 32–80%). In v6 without LN, these same types collapsed to -2 to -10 (sigmoid <0.01–10.7%).
Why: The sigmoid death spiral requires runaway feature magnitude variation — when some neurons have very large pre-activations and others very small, the sigmoid gate saturates for the large ones and kills the small ones. LayerNorm eliminates this variation by normalizing all features to unit scale, keeping the sigmoid in its responsive range.
Evidence: v6 MC_PreAct_g1 gamma_multi = -4.14 (collapsed) → v7 MC_PreAct_LN_g1 gamma_multi = +2.34 (healthy). The ONLY difference is LayerNorm. This is the cleanest controlled experiment yet for gate collapse causation.
Confidence: HIGH — consistent across all 8 sigmoid-bounded +LN runs.

**Finding 3 — v6's FiLM advantage was entirely due to LayerNorm (HIGH confidence)**
What: v6 concluded "FiLM structurally prevents multimodal hub collapse" (Finding 1) and "FiLM dramatically improves GAE" (Finding 3). v7 shows that LayerNorm alone, without FiLM, achieves the same or better results.
Why: FiLM was the only modulation type in v6 that included LayerNorm. The unconstrained γ did not prevent collapse — LN did. PreActivation (sigmoid-bounded) with LN achieves the **healthiest gates** in v7, proving that unconstrained γ is not necessary when LN is present.
Evidence: GAE_Unmod_LN (314 steps, no modulator at all) > v6 GAE_FiLM_g1 (294 steps, with FiLM). MC_PreAct_LN_g1 gamma_multi (+2.34, sigmoid 91%) vs MC_FiLM_LN_g1 gamma_multi (+1.26, unconstrained) — PreActivation+LN gates are healthier than FiLM+LN gates.
Confidence: HIGH — this fundamentally reinterprets v6's conclusions.

**Finding 4 — PreActivation + LN is the best modulated configuration (MEDIUM confidence)**
What: MC_PreAct_LN_g1 achieves the highest MC survival (361 steps), the healthiest gates (gamma_multi = 2.34, σ ≈ 91%), and the highest gate variance (gamma_multi_std = 2.12) among v7 runs. GAE_PreAct_LN_g1 achieves 307 steps — 3rd among modulated GAE configs.
Why: With LN preventing collapse, the sigmoid bound in PreActivation may actually be beneficial — it constrains γ to (0,1), providing a well-defined scaling semantics (attenuation vs pass-through) that is easier for the optimizer to learn than FiLM's unconstrained γ.
Evidence: MC_PreAct_LN_g1 > MC_FiLM_LN_g1 (+3 steps, healthier gates). MC_PreAct_LN_g4 gamma_multi = 4.24 (sigmoid 99%) — the gate found near-identity as optimal and stabilized there.
Confidence: MEDIUM — differences are small; could be within seed variance.

**Finding 5 — The modulator hurts GAE performance (MEDIUM confidence)**
What: GAE_Unmod_LN (314 steps) outperforms 5 of 6 modulated GAE+LN configs. Only GAE_Mult_LN_g1 (321) exceeds it.
Why: The modulator adds parameters that receive noisy gradient updates from GAE's high-loss critic (loss ≈ 69–76). The modulator's gradient noise (grad_norm 5–164 for modulated vs 0 for unmodulated) propagates instability back through the network. Without a modulator, the simpler architecture has fewer parameters to destabilize.
Evidence: GAE_Unmod_LN (314) > GAE_FiLM_LN_g1 (289), GAE_PreAct_LN_g1 (307), GAE_Mult_LN_g4 (293). The exception (GAE_Mult_LN_g1 = 321) has notably low gradient norms (modulator grad_norm = 4.9 vs 137–164 for other types).
Confidence: MEDIUM — the single exception (Mult_g1) and single seeds prevent a definitive conclusion.

**Finding 6 — LN effect is ~7× larger for GAE than MC (HIGH confidence)**
What: LN adds ~57 steps for GAE (unmodulated: 257 → 314) but only ~8 steps for MC (344 → 352).
Why: MC value loss is already ~0.29 regardless of LN. LN helps with feature conditioning but MC doesn't need it for stable learning. GAE's critic loss drops from 80.8 to 68.7 with LN — normalized features enable more stable value function estimation, which cascades into better advantage estimates and policy updates.
Evidence: The LN effect mirrors the MC–GAE gap: configurations where GAE struggles most (NoLN) benefit most from LN.
Confidence: HIGH.

### 5.2 Cross-Run Comparisons

#### LayerNorm Effect — Controlled (Unmodulated, MC vs GAE)

| Metric | MC_Unmod_LN | MC_Unmod_NoLN | Δ | GAE_Unmod_LN | GAE_Unmod_NoLN | Δ |
|--------|------------|--------------|---|-------------|---------------|---|
| Steps SS | 352.49 | 344.04 | +8.45 | 313.68 | 256.89 | **+56.79** |
| Reward SS | -131.34 | -131.76 | +0.42 | -130.08 | -134.57 | **+4.49** |
| FoodEaten | 53.70 | 52.05 | +1.65 | 50.58 | 37.14 | **+13.44** |
| Term_MaxSteps | 34.6% | 32.6% | +2.0% | 25.5% | 13.7% | **+11.8%** |
| loss/value | 0.292 | 0.292 | 0.000 | 68.67 | 80.83 | **-12.16** |

MC is nearly unchanged by LN across all metrics. GAE shows dramatic improvement on every metric — steps, food, max-step survival rate, and critic loss all improve substantially.

#### Modulation Type Effect — Controlled (All +LN, MC, gSize=1)

| Metric | Unmod_LN | FiLM_LN | PreAct_LN | Mult_LN |
|--------|---------|---------|----------|---------|
| Steps SS | 352.49 | 357.26 | **360.57** | 356.75 |
| gamma_multi | — | +1.262 | **+2.336** | +0.127 |
| gamma_uni | — | +1.204 | +1.870 | +1.315 |
| gamma_multi_std | — | 1.072 | **2.118** | 2.141 |
| value_loss | 0.292 | 0.298 | 0.296 | 0.293 |
| mod_grad_norm | — | 0.119 | 0.127 | 0.045 |

With LN present, all modulation types perform within 8 steps of the unmodulated baseline. PreActivation has the healthiest and most diverse gates; Multiplicative has the lowest modulator gradient norm.

#### v6 → v7 Comparison — Effect of Adding LN to Sigmoid-Bounded Modes

| Config | v6 (no LN) Steps | v7 (+LN) Steps | Δ | v6 gamma_multi | v7 gamma_multi |
|--------|-----------------|----------------|---|----------------|----------------|
| MC_PreAct_g1 | 349 | 361 | **+12** | -4.14 (1.6%) | **+2.34 (91%)** |
| MC_PreAct_g4 | 353 | 357 | +4 | -5.52 (0.4%) | **+4.24 (99%)** |
| GAE_PreAct_g1 | 201 | 307 | **+106** | -2.12 (10.7%) | **+1.59 (83%)** |
| GAE_PreAct_g4 | 242 | 288 | **+46** | -5.08 (0.6%) | **+1.89 (87%)** |
| MC_Mult_g1* | 363 | 357 | -6 | -6.13 (0.2%) | +0.13 (53%) |
| MC_Mult_g4* | 370 | 360 | -10 | -9.57 (<0.01%) | -0.74 (32%) |
| GAE_Mult_g1* | 238 | 321 | **+83** | -3.83 (2.1%) | **+0.66 (66%)** |
| GAE_Mult_g4*† | 250 | 293 | **+43** | -8.56 (<0.01%) | **+1.40 (80%)** |

*v6 Multiplicative trained ~5× longer (~20B vs ~3–4B steps). †GAE_Mult_g4 had catastrophic late-training collapse in v6.

Adding LN transforms **every** sigmoid-bounded configuration's gate dynamics from collapsed to functional. The performance impact is dramatic for GAE (+43 to +106 steps) but modest for MC (-10 to +12 steps) — consistent with gates being functionally inert under MC conditions.

### 5.3 Failure Modes & Pathologies

#### Pathology 1: Multiplicative Gate Attenuation (even with LN)
- **Affected**: MC_Mult_LN_g1, MC_Mult_LN_g4
- **Severity**: gamma_multi declining toward 0 throughout training (MC_g4 reaches -0.76, σ ≈ 32%)
- **Impact**: Minimal — MC_Mult_LN_g4 achieves 360 steps despite attenuation
- **Likely cause**: Multiplicative lacks the additive β term, so the modulation can only scale features. The optimizer reduces gamma toward zero when modulation doesn't help, since σ(0) = 0.5 is a reasonable default. With β, PreActivation can learn meaningful offsets.

#### Pathology 2: Modulator Hurts GAE
- **Affected**: 5 of 6 modulated GAE+LN configs
- **Severity**: -6 to -25 steps compared to unmodulated LN baseline
- **Likely cause**: GAE's high critic loss (~69–76) produces noisy gradients that flow through the modulator, destabilizing the network. The modulator adds ~12K parameters that receive these noisy updates.
- **Exception**: GAE_Mult_LN_g1 (321 steps) — notably has the lowest modulator gradient norm (4.9 vs 106–164 for other types), suggesting the modulator is effectively inert and not injecting noise.

#### Pathology 3: GAE_Unmod_LN Gradient Spike
- **Affected**: GAE_Unmod_LN
- **Severity**: Single extreme gradient spike (loss/grad_norm max 3.35M), inflating the SS mean to 1,705. Final value: 23.8.
- **Impact**: None apparent — steps continue climbing despite the spike
- **Likely cause**: GAE-specific critic instability, mitigated by gradient clipping

---

## 6. Conclusions

### 6.1 Summary

- **LayerNorm is the primary driver of survival performance, not the modulation mechanism.** The unmodulated baseline with LN achieves MC: 352 steps, GAE: 314 steps — matching or exceeding all v6 modulated results (v6 best: MC FiLM 361, GAE FiLM 294). The modulator adds at most +8 steps for MC and generally *hurts* GAE performance (5/6 configs worse than unmodulated).

- **v6's attribution of FiLM's success to unconstrained γ was incorrect** — it was the LayerNorm. PreActivation (sigmoid-bounded) + LN achieves healthier gates (sigmoid 83–99%) than FiLM + LN (unconstrained γ ≈ 0.8–1.3). Gate collapse is not caused by sigmoid bounding per se, but by the interaction of sigmoid bounding with unnormalized features.

- **LayerNorm prevents gate collapse for ALL modulation types.** Adding LN transforms PreActivation from deeply collapsed (sigmoid 0.4–10.7%) to fully open (83–99%). Multiplicative goes from deeply collapsed (<0.01–2.1%) to functional (32–80%). This is the most parsimonious explanation of gate collapse: unnormalized features create runaway magnitude variation that drives sigmoid gates to saturation.

- **The modulator's contribution is minimal for MC** (~3–8 steps above LN-only baseline, ~1–2% improvement) and **negative for GAE** (the modulator injects gradient noise that destabilizes the critic). The MC result answers v6's P0 question: the modulator is nearly functionally inert under NoNoise MC conditions.

- **LN's effect is ~7× larger for GAE (+57 steps) than MC (+8 steps),** because LN stabilizes the critic's value function learning, which is the bottleneck for GAE but not for MC.

### 6.2 Limitations & Open Questions

1. **Single seeds**: All configurations use a single seed except MC_Unmod_NoLN (2 seeds, Δ = 0.44 steps — very tight). The small modulated-vs-unmodulated differences (~3–8 steps for MC) could be within seed variance.
2. **Training duration**: All v7 runs trained ~3–5B steps. v6 Multiplicative trained ~20B. Would the modulator's contribution grow with longer training?
3. **NoNoise only**: These results are under NoNoise conditions. With perceptual noise enabled, the modulator might provide a genuine noise-filtering benefit that LN alone cannot achieve.
4. **GAE modulator harm**: Is the modulator hurting GAE because of architecture (extra parameters + noisy gradients) or hyperparameters (learning rate, gradient clipping)? A modulator with a much lower learning rate might help rather than hurt.
5. **Multiplicative anomaly**: GAE_Mult_LN_g1 is the only modulated config to exceed the unmodulated baseline for GAE. Is this genuine or noise? Its low modulator gradient norm suggests the modulator is effectively inert — so the benefit may come from the implicit capacity (extra parameters acting as a mild regularizer) rather than active modulation.
6. **Gate health vs performance**: PreActivation + LN has the healthiest gates (sigmoid 91–99%) but does not clearly outperform FiLM or Multiplicative. Healthy gates appear to be necessary for modulator contribution but not sufficient for large performance gains.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| **P0** | **Enable perceptual noise with LN configurations** | Under NoNoise, the modulator has nothing to modulate — all sensory channels are equally reliable. With noise, the modulator should have a genuine signal-vs-noise discrimination task. This is the key test of whether the NMN architecture has value beyond LN. | Low |
| **P1** | **Multi-seed replication of key configs (MC_Unmod_LN, MC_PreAct_LN_g1, GAE_Unmod_LN)** | With only ~3–8 step differences between configs, 3–5 seeds are needed to determine statistical significance. | Med |
| **P2** | **GAE modulator with reduced learning rate** | The modulator may hurt GAE because it receives noisy gradients at the same learning rate as the main network. Try 0.1× or 0.01× the main learning rate. | Low |
| **P3** | **Extended training (20B steps) for FiLM+LN and PreAct+LN** | Test whether the modulator's contribution grows with training. GAE_FiLM_LN is still climbing at the end of training. | Med |
| **P4** | **Ablation: LN on unmodulated + noise vs LN on modulated + noise** | Directly test whether the modulator provides noise-robustness that LN alone cannot. | Low |

---

## Appendix

### A. Raw Data

Full extraction data available in:
- `tmp/20260319_v7_extract_results.md` — summary metrics for all 17 runs
- `tmp/20260319_v7_timeseries_results.md` — 20-window timeseries for all 17 runs

### B. Run Label Reference

| Label | Full Run Name |
|-------|--------------|
| GAE_PreAct_LN_g1 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_PreActivation_gSize1 |
| GAE_PreAct_LN_g4 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_PreActivation_gSize4 |
| MC_PreAct_LN_g1 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_PreActivation_gSize1 |
| MC_PreAct_LN_g4 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_PreActivation_gSize4 |
| GAE_Mult_LN_g1 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_Multiplicative_gSize1 |
| GAE_Mult_LN_g4 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_Multiplicative_gSize4 |
| MC_Mult_LN_g1 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_Multiplicative_gSize1 |
| MC_Mult_LN_g4 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_Multiplicative_gSize4 |
| GAE_FiLM_LN_g1 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize1 |
| GAE_FiLM_LN_g4 | rppoNMN_GAE_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize4 |
| MC_FiLM_LN_g1 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize1 |
| MC_FiLM_LN_g4 | rppoNMN_MC_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm_FiLM_gSize4 |
| GAE_Unmod_LN | rppo_GAE_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm |
| GAE_Unmod_NoLN | rppo_GAE_NoNoise_128env_2bush4pred2food_NutGain6_NoLayerNorm |
| MC_Unmod_LN | rppo_MC_NoNoise_128env_2bush4pred2food_NutGain6_LayerNorm |
| MC_Unmod_NoLN_1 | rppo_MC_NoNoise_128env_2bush4pred2food_NutGain6_NoLayerNorm (o431n7ig) |
| MC_Unmod_NoLN_2 | rppo_MC_NoNoise_128env_2bush4pred2food_NutGain6_NoLayerNorm (5s0ah0pu) |

### C. Complete 2D Factorial Summary

| | use_layer_norm: false | use_layer_norm: true |
|---|---|---|
| **null (no modulator)** | MC: 344 / GAE: 257 | MC: 352 / GAE: 314 |
| **FiLM** | MC: 345–348 (v6) / GAE: 189–195 (v6) | MC: 355–357 / GAE: 289–294 |
| **PreActivation** | MC: 349–353 (v6) / GAE: 201–242 (v6) | MC: 357–361 / GAE: 288–307 |
| **Multiplicative** | MC: 363–370 (v6*) / GAE: 238–250 (v6*†) | MC: 357–360 / GAE: 293–321 |

*v6 Multiplicative trained ~5× longer. †GAE_Mult_g4 had catastrophic collapse in v6.
v6 "FiLM" = FiLM+LN (now equivalent to v7 FiLM+LN). v6 "FiLMNoNorm" = FiLM+NoLN.

**Reading the table**: The LN column is dramatically better than the NoLN column for GAE (+43 to +119 steps), and modestly better for MC (+4 to +12 steps). Within each column, modulation type matters little — the spread is ~16 steps for MC+LN and ~33 steps for GAE+LN.

### D. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-03-19 | Initial analysis | Claude |
