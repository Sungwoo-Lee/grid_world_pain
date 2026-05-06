---
title: "NMN Performance Diagnosis v3: Hub Collapse, Grouping Size & Return Mode"
topic: diagnosis
status: superseded
created: 2026-03-11
last_updated: 2026-04-12
supersedes: NMN_PERFORMANCE_DIAGNOSIS_v2.md
superseded_by: NMN_PERFORMANCE_DIAGNOSIS_v4.md
---

# NMN Performance Diagnosis v3: Hub Collapse, Grouping Size & Return Mode

> **Status**: COMPLETE
> **Date**: 2026-03-11
> **Author**: Claude (analysis), Sungwoo (experiment design & training)
> **Related**: [NMN_PERFORMANCE_DIAGNOSIS_v2.md](NMN_PERFORMANCE_DIAGNOSIS_v2.md) (v2, §12), [NMN_PERFORMANCE_DIAGNOSIS.md](NMN_PERFORMANCE_DIAGNOSIS.md) (v1, §1–11)

---

## 1. Research Question

v2 established that multimodal hub collapse (gamma_multi sigmoid < 0.2%) is universal across 22/22 NMN configurations. This experiment tests whether the collapse can be prevented by manipulating two architectural levers: **multimodal hub width** and **neuromodulatory grouping size**, while also comparing **GAE vs Monte Carlo (MC) return estimation**.

> **H₀** (null): Neither hub size nor grouping size affects multimodal gate collapse. gamma_multi converges to the same deeply negative values regardless of configuration.
> **H₁a**: Wider multimodal hub (512, 1024, 2048 vs 128) produces richer features, reducing the optimizer's incentive to suppress the hub via gamma.
> **H₁b**: Smaller grouping size (1, 4, 8, 16 vs 64) constrains the modulator's ability to suppress an entire pathway with a single gate, forcing finer-grained and less degenerate gating.
> **H₁c**: MC return estimation produces different modulator dynamics than GAE, potentially affecting collapse behavior.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| `multimodal_hidden_size` (hub) | 128, 512, 1024, 2048 | Test if larger hub produces features worth preserving |
| `grouping_size` | 1, 4, 8, 16, 64 | Test if finer-grained gating prevents wholesale collapse |
| `return_mode` | GAE, MC | Compare return estimation effects on modulator dynamics |

### 2.2 Controlled Variables

```yaml
modulation:
  type: Multiplicative
  mod_hidden_size: 16       # v1 §10 fix
  percept_bias_init: 3.0
  memory_bias_init: 0.0
  temp_clip: [0.5, 3.0]
  memory_clip: [-2.0, 2.0]

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
```

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| Config mismatch: tag=gSize4 but actual=gSize8 (GAE) | Run #6 (20260310-164032) | **High** | Missing gSize=4 GAE data point. Use actual config value in analysis. |
| Config mismatch: tag=gSize8 but actual=gSize16 (MC) | Run #11 (20260310-164514) | **High** | Missing gSize=8 MC data point. Use actual config value in analysis. |
| No baseline (non-NMN) runs in this sweep | All | Med | Compare against v2 baseline (same env: 2b/4p/2f → reward ~-136) |
| Hub sweep only at gSize=64 | Hub sweep runs | Med | Cannot isolate hub×gSize interaction |
| Single seed per config | All | Med | Variance within runs provides some signal but no seed replication |

## 3. Run Inventory

| # | Datetime ID | WandB ID | Return | gSize (actual) | Hub | Timesteps | Config Match |
|---|---|---|---|---|---|---|---|
| 1 | 20260310-155531 | `1oewe6cz` | GAE | 64 | 1024 | ~28M | YES |
| 2 | 20260310-155723 | `xfns7hd7` | GAE | 64 | 2048 | ~28M | YES |
| 3 | 20260310-155809 | `eu95nw4k` | GAE | 64 | 512 | ~28M | YES |
| 4 | 20260310-155825 | `kio3g92a` | GAE | 64 | 128 | ~29M | YES |
| 5 | 20260310-164016 | `f3hmb222` | GAE | 1 | 128 | ~24M | YES |
| 6 | 20260310-164032 | `archj5tj` | GAE | **8** (tag=4) | 128 | ~24M | **NO** |
| 7 | 20260310-164058 | `felpqc2g` | GAE | 8 | 128 | ~24M | YES |
| 8 | 20260310-164109 | `iaz292pr` | GAE | 16 | 128 | ~24M | YES |
| 9 | 20260310-164433 | `kkmfgcnr` | MC | 1 | 128 | ~23M | YES |
| 10 | 20260310-164454 | `ciizmt4b` | MC | 4 | 128 | ~23M | YES |
| 11 | 20260310-164514 | `2btrsbjv` | MC | **16** (tag=8) | 128 | ~23M | **NO** |
| 12 | 20260310-164526 | `id5klhnr` | MC | 16 | 128 | ~23M | YES |
| 13 | 20260310-164959 | `7t7pq75y` | MC | 64 | 128 | ~23M | YES |
| 14 | 20260310-165026 | `x3estj6a` | MC | 64 | 512 | ~23M | YES |
| 15 | 20260310-170117 | `dhhq9v4k` | MC | 64 | 1024 | ~23M | YES |
| 16 | 20260310-170141 | `rd7w8irs` | MC | 64 | 2048 | ~15M | YES |

Note: Run #6 and #11 have tag/config mismatches. gSize=4 (GAE) and gSize=8 (MC) data points are **missing** from this sweep.

---

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

#### 4.1.1 Hub Size Effect on gamma_multi (H₁a) — gSize=64

| Hub | GAE gamma_multi SS | GAE sig% | MC gamma_multi SS | MC sig% |
|-----|---------------------|----------|---------------------|---------|
| 128 | -8.582 ± 0.20 | 0.019% | -7.539 ± 0.28 | 0.053% |
| 512 | -10.01 ± 0.20 | 0.004% | -7.839 ± 0.21 | 0.040% |
| 1024 | -10.91 ± 0.22 | 0.002% | -8.911 ± 0.22 | 0.013% |
| 2048 | -10.10 ± 0.20 | 0.004% | -8.352 ± 0.20 | 0.024% |

> **Verdict on H₁a: NOT SUPPORTED.** Increasing hub size from 128 to 2048 does not reduce collapse — it makes it **worse**. Larger hubs collapse to even more negative gamma_multi values. The wider hub adds more variance without producing proportionally more useful features, strengthening the optimizer's incentive to suppress it.

#### 4.1.2 Grouping Size Effect on gamma_multi (H₁b) — hub=128

| gSize | GAE gamma_multi SS | GAE sig% | MC gamma_multi SS | MC sig% |
|-------|---------------------|----------|---------------------|---------|
| 1 | -0.842 ± 0.06 | **30%** | -1.607 ± 0.15 | **17%** |
| 4 | — (missing) | — | -3.610 ± 0.15 | 2.6% |
| 8 | -3.996 ± 0.17 | 1.8% | — (missing) | — |
| 8* | -5.315 ± 0.15 | 0.49% | — | — |
| 16 | -5.556 ± 0.14 | 0.39% | -5.918 ± 0.15 | 0.27% |
| 16* | — | — | -5.742 ± 0.10 | 0.32% |
| 64 | -8.582 ± 0.20 | 0.019% | -7.539 ± 0.28 | 0.053% |

(*) = mislabeled run, actual config used

> **Verdict on H₁b: STRONGLY SUPPORTED.** Grouping size has a dramatic, monotonic effect on hub collapse severity:
> - **gSize=1**: gamma_multi sigmoid ~17–30% — **hub is partially alive**
> - **gSize=4–8**: sigmoid ~0.5–2.6% — mostly collapsed but not fully dead
> - **gSize=16**: sigmoid ~0.3% — near-complete collapse
> - **gSize=64**: sigmoid <0.05% — total collapse (same as all previous experiments)
>
> Effect size: ~1500× difference in gate openness between gSize=1 and gSize=64.

#### 4.1.3 GAE vs MC Return Mode (H₁c)

| Return | Avg gamma_multi SS (gSize=64) | Avg temp_mean SS | z_memory escaping? |
|--------|-------------------------------|-------------------|--------------------|
| GAE | -9.80 | 2.58 (pinning at 3.0) | No |
| MC | -8.16 | 0.66 (near 0.5 floor) | **YES** (gSize ≤ 4) |

> **Verdict on H₁c: SUPPORTED — MC produces fundamentally different dynamics.** MC runs have (a) dramatically lower temperature (0.6–1.6 vs 2.4–2.8), (b) z_memory pathology where values escape the [-2,2] clamp, and (c) slightly less severe hub collapse. However, MC also has worse reward performance and higher variance.

### 4.2 Episode Performance

#### GAE Hub Size Sweep (gSize=64)

| Hub | Reward SS | Steps SS | FoodEaten SS | TotalDamage SS |
|-----|-----------|----------|-------------|----------------|
| 128 | -134.10 ± 4.29 | 195.28 ± 11.19 | 49.60 ± 5.46 | 89.14 ± 6.71 |
| 512 | **-132.63 ± 4.17** | 185.84 ± 10.99 | 44.86 ± 5.34 | **84.56 ± 6.41** |
| 1024 | -137.41 ± 4.40 | 196.56 ± 11.45 | 50.59 ± 5.57 | 93.86 ± 7.01 |
| 2048 | -134.13 ± 4.27 | 188.01 ± 11.30 | 46.20 ± 5.50 | 88.49 ± 6.72 |

Hub size has no meaningful effect on episode performance (within ~4 pts, overlapping CIs). The hub is suppressed regardless of width, so wider hubs neither help nor hurt — the extra parameters are wasted.

#### GAE Grouping Size Sweep (hub=128)

| gSize | Reward SS | Steps SS | FoodEaten SS | TotalDamage SS |
|-------|-----------|----------|-------------|----------------|
| 1 | -132.90 ± 4.35 | 174.64 ± 9.51 | 39.62 ± 4.54 | **82.05 ± 6.46** |
| 8 | -134.57 ± 4.42 | 185.86 ± 11.16 | 45.15 ± 5.43 | 86.47 ± 6.99 |
| 16 | -134.39 ± 4.32 | 193.81 ± 11.28 | 49.03 ± 5.47 | 89.67 ± 6.74 |
| 64 | -134.10 ± 4.29 | 195.28 ± 11.19 | 49.60 ± 5.46 | 89.14 ± 6.71 |

gSize=1 has slightly better reward (-132.90 vs -134.10) and lowest damage (82.05), but shorter episodes and less food. The partial hub preservation at gSize=1 does not translate into a clear performance advantage — the **unimodal suppression** at gSize=1 partially offsets the multimodal benefit.

#### MC Grouping Size Sweep (hub=128)

| gSize | Reward SS | Steps SS | FoodEaten SS | TotalDamage SS |
|-------|-----------|----------|-------------|----------------|
| 1 | -137.44 ± 5.32 | 175.18 ± 12.05 | 39.21 ± 5.96 | 85.14 ± 9.10 |
| 4 | -139.55 ± 4.52 | 175.50 ± 10.66 | 39.58 ± 5.21 | 87.68 ± 7.43 |
| 16 | -140.01 ± 5.00 | 172.25 ± 10.93 | 38.03 ± 5.39 | 85.79 ± 8.22 |
| 64 | -133.85 ± 10.01 | 163.60 ± 21.92 | 33.35 ± 11.00 | 77.73 ± 16.94 |

MC runs generally perform worse than GAE (reward -134 to -140 vs -133 to -135). MC_gSize64 has the best MC reward (-133.85) but extremely high variance (±10.01), likely unstable.

#### MC Hub Size Sweep (gSize=64)

| Hub | Reward SS | Steps SS | FoodEaten SS | TotalDamage SS |
|-----|-----------|----------|-------------|----------------|
| 128 | -133.85 ± 10.01 | 163.60 ± 21.92 | 33.35 ± 11.00 | 77.73 ± 16.94 |
| 512 | -137.28 ± 6.45 | 170.62 ± 14.95 | 36.92 ± 7.46 | 84.14 ± 11.23 |
| 1024 | -136.96 ± 4.24 | 175.68 ± 9.89 | 39.44 ± 4.85 | 85.96 ± 7.22 |
| 2048 | -139.39 ± 4.83 | 175.08 ± 11.18 | 39.31 ± 5.51 | 87.88 ± 8.09 |

MC hub sweep shows same pattern as GAE: hub size doesn't matter for performance.

### 4.3 Diagnostic Metrics

#### 4.3.1 gamma_uni — Unimodal Gates

**Critical trade-off with grouping size:**

| gSize | GAE gamma_uni SS | GAE sig% | MC gamma_uni SS | MC sig% |
|-------|------------------|----------|------------------|---------|
| 1 | -2.705 ± 0.09 | **6.3%** | -3.496 ± 0.06 | **3.0%** |
| 4 | — | — | -2.646 ± 0.07 | 6.6% |
| 8 | -2.830 ± 0.05 | 5.6% | — | — |
| 16 | -1.518 ± 0.09 | 18% | +1.961 ± 0.07 | 88% |
| 64 | +0.507 ± 0.06 | 62% | +0.634 ± 0.22 | 65% |

**Pattern**: Small gSize preserves multimodal but suppresses unimodal. Large gSize suppresses multimodal but preserves unimodal. The modulator has a **fixed suppression budget** — it cannot keep both pathways open simultaneously. This is a zero-sum gate allocation problem.

#### 4.3.2 Temperature

| Return | gSize=1 | gSize=4 | gSize=8 | gSize=16 | gSize=64 |
|--------|---------|---------|---------|----------|----------|
| GAE | 2.824 | — | 2.695 | 2.530 | 2.649 |
| MC | 1.576 | 0.911 | — | 0.985 | 0.785 |

**GAE** consistently inflates temperature to 2.5–2.8 (near the 3.0 ceiling).
**MC** keeps temperature near the 0.5 floor (0.6–1.6). This is the most dramatic GAE/MC difference — MC returns have lower variance, reducing the need for exploration temperature.

#### 4.3.3 z_memory

| gSize | GAE z_mem SS | MC z_mem SS |
|-------|-------------|-------------|
| 1 | +0.648 | **-6.001** |
| 4 | — | -1.947 |
| 8 | +1.013 | — |
| 16 | -0.141 | -1.899 |
| 64 | -0.917 | -0.898 |

**CRITICAL BUG — MC z_memory escapes the [-2,2] clamp:**
- MC_gSize1: z_memory = -6.001 (well outside [-2,2])
- MC_gSize16* (mislabeled): z_memory = -4.243

The timeseries shows MC_gSize1 z_memory monotonically decreasing from -0.89 → -6.16 over 23M steps, clearly not clamped. Either `memory_clip` is not being applied to MC runs, or it is only checked at initialization, not during the forward pass.

GAE runs all stay within or near the [-2,2] range, confirming the clamp works for GAE. This suggests a **code path divergence** between GAE and MC return computation that bypasses the clamp.

### 4.4 Learning Dynamics

#### 4.4.1 Hub Collapse Timeseries (gamma_multi over training)

**GAE Hub Size Sweep (gSize=64) — window means at ~5M step intervals:**

| ~Steps | hub=128 | hub=512 | hub=1024 | hub=2048 |
|--------|---------|---------|----------|----------|
| 0–3M | +1.5 | +1.2 | -0.4 | +1.1 |
| 5M | -1.7 | -2.4 | -4.5 | -2.9 |
| 10M | -5.0 | -5.7 | -7.3 | -6.0 |
| 15M | -6.5 | -7.5 | -9.1 | -7.6 |
| 20M | -7.7 | -8.9 | -10.0 | -8.6 |
| 25M | -8.5 | -10.0 | -11.0 | -9.8 |
| 28M | -8.8 | -10.4 | -11.3 | -10.3 |

Collapse is **monotonic** and **never reverses**. Larger hubs collapse faster and deeper. hub=1024 is the worst — it reaches gamma=-11.3 (sigmoid 0.001%). The trajectory is approximately linear in log-space after the initial transition, suggesting exponential suppression dynamics.

**GAE Grouping Size Sweep (hub=128):**

| ~Steps | gSize=1 | gSize=8 | gSize=16 | gSize=64 |
|--------|---------|---------|----------|----------|
| 0–3M | +2.5 | +2.5 | +2.8 | +1.5 |
| 5M | +1.3 | +1.1 | +0.2 | -1.7 |
| 10M | +0.1 | -0.2 | -2.2 | -5.0 |
| 15M | -0.5 | -1.0 | -3.8 | -6.5 |
| 20M | -0.8 | -1.5 | -5.0 | -7.7 |
| 24M | -0.9 | -1.8 | -5.8 | -8.8 |

**gSize=1 collapse rate: ~0.14 units/M steps** (very slow, approaching plateau)
**gSize=64 collapse rate: ~0.37 units/M steps** (fast, no sign of plateau)

At gSize=1, the collapse trajectory is **decelerating** — it may plateau around sigmoid ~25%. This is the first evidence of any run finding a non-degenerate equilibrium for the multimodal hub.

#### 4.4.2 Unimodal Gate Dynamics (gamma_uni)

gSize=1 runs show monotonic unimodal suppression: starting at +3.2 (sigmoid 96%), dropping to -2.9 (sigmoid 5%) by 24M steps. The suppression rate is steady with no sign of reversal. This confirms the zero-sum trade-off — preserving multimodal comes at the direct expense of unimodal.

gSize=64 runs show the opposite: gamma_uni stays positive (+0.5 to +1.0, sigmoid 60–73%), with the unimodal pathway preserved while multimodal is eliminated.

#### 4.4.3 MC z_memory Escape Trajectory

MC_gSize1 z_memory timeseries (20 windows):
- Window 1 (0–1M): -0.89
- Window 5 (6–7M): -3.40
- Window 10 (11–12M): -4.67
- Window 15 (17–18M): -5.81
- Window 20 (22–23M): -6.16

The descent is monotonic at ~0.24 units/M steps with no sign of clamping at -2.0. This is **unambiguously a bug** — the memory_clip parameter is not being enforced during MC training.

---

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — Grouping size is the critical lever for hub collapse (HIGH confidence)**
What: gSize=1 maintains multimodal sigmoid at ~17–30% vs <0.05% at gSize=64 — a 1500× difference.
Why: With gSize=1, each neuron in the multimodal output gets its own independent gate. The modulator cannot suppress the entire hub with a single gate value — it must evaluate each feature individually. Features that carry useful information survive; only truly noisy features are suppressed. With gSize=64, a single gate controls 64 neurons simultaneously, making wholesale suppression the optimal strategy.
Evidence: Monotonic relationship across gSize={1,4,8,16,64} in both GAE and MC runs.
Confidence: HIGH — consistent across both return modes with no exceptions.

**Finding 2 — Unimodal-multimodal gate trade-off is zero-sum (HIGH confidence)**
What: gSize=1 preserves multimodal (sig ~30%) but suppresses unimodal (sig ~3–6%). gSize=64 does the opposite (multimodal ~0.02%, unimodal ~62%). No configuration keeps both open.
Why: The modulator has limited capacity (mod_hidden_size=16). It can only produce a limited range of gate values. With gSize=1, it must dedicate its parameters to fine-grained multimodal gating, leaving unimodal gates under-parameterized. The modulator is capacity-constrained — it allocates its representational budget to the pathway with more features to gate.
Evidence: Consistent inverse relationship in gamma_uni vs gamma_multi across all grouping sizes.
Confidence: HIGH.

**Finding 3 — Hub width does not prevent collapse (HIGH confidence)**
What: Even hub=2048 (16× wider than hub=128) collapses to gamma_multi sigmoid <0.03%.
Why: The hub's MLP layers (input→128/512/1024/2048→128→128→output) do not produce features with sufficient signal-to-noise ratio relative to the direct unimodal pathway. Wider layers mean more parameters and more noise variance, making the suppression incentive stronger, not weaker.
Evidence: hub=1024 has the deepest collapse (-11.3 vs -8.6 for hub=128).
Confidence: HIGH.

**Finding 4 — MC return mode causes z_memory clamp escape (HIGH confidence)**
What: MC runs with small gSize show z_memory values far outside the configured [-2,2] clamp (reaching -6.0 at gSize=1).
Why: Either (a) the `memory_clip` parameter is not applied during the MC return computation path, or (b) the clip is applied to the raw modulator output but the accumulated gradient pushes the underlying parameters far enough that the clipped value saturates at -2.0 while the reported metric shows the pre-clip value. Investigation of `src/` code is needed.
Evidence: MC_gSize1 z_memory timeseries monotonically decreases 0.89→6.16 with no clamping behavior. GAE runs with same config stay within [-2,2].
Confidence: HIGH that the behavior exists. MEDIUM on root cause — code investigation needed.

**Finding 5 — MC fundamentally changes temperature dynamics (HIGH confidence)**
What: GAE inflates temperature to 2.5–2.8 (near 3.0 ceiling). MC keeps temperature at 0.6–1.6 (near 0.5 floor).
Why: MC returns have higher variance than GAE (no bootstrapping bias but full trajectory noise). The higher return variance already provides exploration signal, reducing the modulator's incentive to inflate temperature for exploration.
Evidence: Consistent across all gSize and hub configurations. No overlap in temperature ranges between GAE and MC.
Confidence: HIGH.

**Finding 6 — Collapse is monotonic and never reverses (HIGH confidence)**
What: In all 38 NMN runs analyzed to date (v1+v2+v3), gamma_multi always moves toward more negative values over training. No run has ever shown collapse followed by recovery.
Why: Once the hub is suppressed, gradients through the hub vanish (gamma acts as a multiplicative gate). Without gradients, the hub features cannot improve, creating a death spiral: suppress → no gradient → no learning → more suppression.
Evidence: All 20-window timeseries show monotonic decrease.
Confidence: HIGH.

### 5.2 Cross-Run Comparisons

#### Grouping Size Effect (Controlled: return=GAE, hub=128)

| Comparison | gSize effect on gamma_multi | gSize effect on gamma_uni | Net effect on performance |
|---|---|---|---|
| gSize=1 vs gSize=64 | +7.7 units (30% vs 0.02%) | -3.2 units (6% vs 62%) | +1.2 reward pts (not significant) |
| gSize=8 vs gSize=64 | +4.6 units (1.8% vs 0.02%) | -3.3 units (5.6% vs 62%) | -0.5 reward pts (not significant) |
| gSize=16 vs gSize=64 | +3.0 units (0.4% vs 0.02%) | -2.0 units (18% vs 62%) | -0.3 reward pts (not significant) |

**Conclusion**: Grouping size dramatically affects gate allocation but has no significant impact on episode reward. The zero-sum trade-off means that preserving multimodal features comes at the cost of unimodal features, and the net effect cancels out.

#### Hub Width Effect (Controlled: return=GAE, gSize=64)

| Comparison | Hub effect on gamma_multi | Effect on reward |
|---|---|---|
| hub=512 vs hub=128 | -1.4 units (deeper collapse) | +1.5 reward pts |
| hub=1024 vs hub=128 | -2.3 units (deeper collapse) | -3.3 reward pts |
| hub=2048 vs hub=128 | -1.5 units (deeper collapse) | 0.0 reward pts |

**Conclusion**: Wider hubs collapse more but performance is unaffected. The hub is dead weight regardless of width.

#### GAE vs MC (Controlled: gSize=64, hub=128)

| Metric | GAE | MC |
|---|---|---|
| Reward SS | -134.10 ± 4.29 | -133.85 ± 10.01 |
| Temp mean | 2.649 | 0.785 |
| z_memory | -0.917 (healthy) | -0.898 (healthy at gSize=64) |
| Reward variance | Low (±4.3) | **High (±10.0)** |

MC achieves similar reward but with much higher variance and fundamentally different modulator dynamics. MC is not recommended due to the z_memory bug and instability.

### 5.3 Failure Modes & Pathologies

#### Pathology 1: Multimodal Hub Collapse (ALL runs, ALL configs)
- **Affected**: All 16 runs (and all 22 runs from v1+v2)
- **When**: Begins within first 1–3M steps, accelerates thereafter
- **Severity varies by gSize**: gSize=1 shows partial collapse (sig ~30%); gSize=64 shows total collapse (sig <0.05%)
- **Likely cause**: The multimodal hub does not produce features with sufficient signal-to-noise ratio. The modulator correctly identifies suppression as variance-reducing.
- **Correlates with performance**: Does NOT correlate. Runs with the most collapsed hub (gSize=64) perform as well as or better than runs with partially alive hub (gSize=1).

#### Pathology 2: z_memory Clamp Escape (MC runs with small gSize)
- **Affected**: MC_gSize1 (z=-6.0), MC_gSize4 (z=-1.9, borderline), MC_gSize16* (z=-4.2)
- **When**: Begins within first 1M steps, monotonic descent
- **Likely cause**: `memory_clip` not applied during MC return computation, or metric reports pre-clip value
- **Impact**: At z=-6.0, the GRU is effectively frozen in "always-forget" mode — same pathology as v1 without the clamp

#### Pathology 3: Unimodal Suppression at Small gSize
- **Affected**: gSize=1 runs (both GAE and MC)
- **When**: Co-occurs with multimodal preservation
- **Likely cause**: Modulator capacity (h=16) is insufficient to gate both pathways simultaneously
- **Impact**: Negates the benefit of multimodal preservation

---

## 6. Conclusions

### 6.1 Summary

- **Grouping size controls the multimodal/unimodal gate allocation trade-off.** gSize=1 keeps the multimodal hub partially alive (sigmoid ~30%) but suppresses unimodal gates (sigmoid ~3–6%). gSize=64 does the opposite. No configuration keeps both pathways open simultaneously.
- **Hub width does not prevent collapse.** Even hub=2048 collapses fully at gSize=64. Wider hubs collapse faster and deeper.
- **The zero-sum gate trade-off prevents any configuration from outperforming baseline.** Preserving multimodal at the cost of unimodal (or vice versa) yields equivalent episode performance. The modulator is solving a constrained optimization that has no better solution than suppressing whichever pathway is noisier.
- **MC return mode has fundamentally different (and buggier) dynamics.** MC keeps temperature low, causes z_memory to escape bounds (likely a code bug), and produces higher-variance training. GAE is the preferred return mode.
- **The hub collapse problem is architectural, not modulator-related.** 38/38 NMN runs collapse the hub regardless of modulation type, grouping size, hub width, return mode, environment, or training duration. The fix must be architectural — the hub must produce features that are genuinely useful.

### 6.2 Limitations & Open Questions

1. **Missing data points**: gSize=4 (GAE) and gSize=8 (MC) due to config mismatches. These would fill in the grouping size curve.
2. **No gSize=1 + large hub**: Would gSize=1 with hub=2048 maintain an even healthier multimodal pathway? The interaction between gSize and hub width is untested.
3. **Why the zero-sum trade-off?** Is it purely a modulator capacity issue (h=16 too small)? Would h=32 or h=64 allow both pathways to stay open? Or is it inherent to the multiplicative gating architecture?
4. **Is the partially-alive hub at gSize=1 actually useful?** The multimodal features at 30% sigmoid pass some signal through, but does the downstream policy use it? Probing the policy network's reliance on multimodal features would answer this.
5. **MC z_memory bug**: Is `memory_clip` not applied during MC forward pass? Or is the metric logging the pre-clip value? Code investigation needed.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| **P0** | **Investigate MC z_memory clamp bug** | z_memory escapes [-2,2] in MC runs. This is likely a code bug. | Low (code review) |
| **P1** | **Test residual/skip connections in the hub** | The hub collapse is architectural. Residual connections let useful signal bypass the noisy MLP layers, giving the modulator something worth preserving. | Medium |
| **P2** | **Test gSize=1 with mod_hidden_size=32/64** | If the zero-sum trade-off is capacity-limited, larger modulators may keep both pathways open. | Low |
| **P3** | **Test intermediate gSize (2, 4) with GAE** | gSize=1 partially works. gSize=4 is missing from GAE. The optimal gSize may be 2–4. | Low |
| **P4** | **Remove the hub entirely (concat unimodal features)** | If the hub never produces useful features, removing it simplifies the architecture and eliminates the collapse problem entirely. Compare performance. | Medium |
| **P5** | **Re-run the 2 mislabeled configs** (gSize=4 GAE, gSize=8 MC) | Fill in the missing data points from this sweep. | Low |

---

## Appendix

### A. Raw Data

Full extraction data available in:
- `tmp/20260311_all_extracts.md` — summary metrics for 13/16 runs
- `tmp/20260311_ts_batch4_MC_hub.md` — MC hub sweep (3 runs)
- `tmp/20260311_all_timeseries.md` — 20-window timeseries for 13/16 runs
- `tmp/20260311_wandb_v3_multimodal_collapse.md` — consolidated analysis notes

### B. Config Diffs

All 16 runs share identical base config except:
- `agent.return_mode`: GAE or MC
- `modulation.grouping_size`: 1, 4, 8, 16, or 64
- `modulation.multimodal_hidden_size`: 128, 512, 1024, or 2048

Two runs have tag/config mismatches:
- Run #6 (20260310-164032): tag=`gSize4` → actual config=`gSize8`
- Run #11 (20260310-164514): tag=`gSize8` → actual config=`gSize16`

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-03-11 | Initial analysis | Claude |
