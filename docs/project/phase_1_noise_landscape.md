# Phase 1 — Reshape the Noise Landscape Until the Baseline Bleeds

> **Status**: PLANNED
> **Opened**: 2026-04-13
> **Related**: [project_plan.md](project_plan.md) (§4, Phase 1), [PRECISION_MODULATION.md](../develop/active/precision/PRECISION_MODULATION.md), [NMN_PERFORMANCE_DIAGNOSIS_v8.md](../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)

---

## Context

The [project plan](project_plan.md) identifies four interacting causes (§4.1–4.4) for why the neuromodulated agent does not outperform the baseline. All four share a common prerequisite: **the noise landscape must create measurable headroom** — a gap between the baseline's noise-free and noisy performance — before any modulation mechanism can demonstrate value.

The four causes are:

1. **No precision training signal (§4.1).** The RL loss rewards behavior, not calibrated gating. The modulator has no gradient pressure to estimate per-channel reliability.
2. **Static FiLM cannot express per-timestep reliability (§4.2).** FiLM applies learned (γ, β) uniformly across time — it cannot distinguish "this channel is noisy right now" from "this feature should always be scaled this way."
3. **Noise landscape does not reward precision (§4.3).** The perceptual noise profile is too uniform across modalities to make selective gating pay off. An unmodulated agent can do about as well as a modulated one.
4. **Temperature saturation and critic instability (§4.4).** The PPO temperature head saturates at the clip ceiling; under GAE, the modulator destabilizes the critic under noisy inputs.

Phase 1 directly addresses **cause §4.3** and establishes the environmental conditions required before causes §4.1, §4.2, and §4.4 can be meaningfully addressed. If noise does not hurt the baseline, there is no room for precision modulation to help, and no subsequent phase is meaningful.

**Exit gate (G1):** An unmodulated LayerNorm baseline's survival must drop *measurably and reproducibly* under the new noise profile compared to the no-noise condition. The drop must be seed-stable (≥3 seeds).

---

## Analysis

### Evidence from v1–v8: The Current Noise Landscape Is Insufficient

Eight rounds of diagnosis experiments (v1–v8) have progressively established that the noise landscape does not create enough selective pressure for modulation to matter.

#### The v8 Null Result (Clearest Statement)

v8 ([NMN_PERFORMANCE_DIAGNOSIS_v8.md](../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)) tested FiLM modulation under perceptual noise with the following noise config:

| Modality | Mode | σ_base | α (injury_scale) | σ_eff at Î=1 |
|----------|------|--------|-------------------|--------------|
| Injury | state_dependent | 0.1 | 1.5 | 0.25 |
| Nutrition | state_dependent | 0.1 | 1.5 | 0.25 |
| Satiation | state_dependent | 0.1 | 1.5 | 0.25 |
| Extero Nociception | state_dependent | 0.1 | 1.5 | 0.25 |
| Olfaction | state_dependent | 0.2 | 1.5 | 0.50 |
| Collision | constant | 0.01 | — | 0.01 |
| Proprioception | constant | 0.05 | — | 0.05 |
| Visual | state_dependent | 0.2 | 1.5 | 0.50 |
| Location | constant | 0.01 | — | 0.01 |

**Key v8 findings:**

- **MC baseline (unmodulated + LN):** 283.64 steps. Best FiLM config (MC_g16): 283.44 steps. **The modulator provides zero benefit** (v8 §4.1.2).
- **GAE baseline (unmodulated + LN):** 264.52 steps. All FiLM configs perform **18–52 steps worse** than unmodulated (v8 §4.1.2).
- **Noise penalty vs NoNoise (v7):** MC dropped from 352→284 (-19.5%), GAE dropped from 314→265 (-15.7%) (v8 §4.1.4). This shows noise *does* hurt — but the modulator cannot exploit the gap.

#### Why the v8 Noise Profile Fails to Reward Precision

Three structural problems with the v8 noise configuration prevent selective gating from being useful:

**1. Insufficient reliability contrast across channels.**

Five of six state-dependent channels share identical parameters (σ=0.1, α=1.5), and the two high-noise channels (olfaction, visual) also share identical parameters (σ=0.2, α=1.5). The noise profile is essentially two tiers:
- Tier 1: σ=0.1 with α=1.5 (injury, nutrition, satiation, extero_noc)
- Tier 2: σ=0.2 with α=1.5 (olfaction, visual)
- Tier 3: near-zero noise (collision σ=0.01, proprioception σ=0.05, location σ=0.01)

With only two meaningful noise levels among the noisy channels, there is little for a precision head to discriminate. An agent that simply "trusts everything equally" or "distrusts everything equally" performs as well as one with per-channel precision.

**2. Uniform injury_noise_scale across state-dependent channels.**

All state-dependent channels use α=1.5. Under injury, all channels degrade at the same rate. A precision-weighting mechanism needs *differential* degradation — some channels becoming unreliable while others remain stable — to learn which channels to down-weight when injured. Uniform degradation means the optimal response is uniform down-weighting, which a simple amplitude reduction (or doing nothing) achieves equally well.

**3. Threat-relevant channels are not selectively degraded.**

The project's target behavior (hypervigilance, §2.1 of the project plan) requires that *threat-relevant* channels (olfaction, nociception, visual) become particularly unreliable after injury, forcing the agent to rely on memory and cautious behavior. Currently, threat-relevant channels are not more degraded than non-threat channels (nutrition and satiation have the same α=1.5 as nociception and olfaction).

#### Progressive Evidence Across v1–v8

| Version | Key Finding Relevant to Noise Landscape |
|---------|----------------------------------------|
| v1 | Only 1/9 modalities used state-dependent noise (injury only). Olfaction and visual were `constant`. The modulator had almost no injury-dependent signal to learn from. |
| v5 | Removing noise entirely made reward *better* (MC: -129 vs -135 with noise) but made gate collapse *worse*. Noise was paradoxically *slowing* collapse by injecting variability. |
| v5 | Noise-free MC collapse rate was ~50% faster than noise-on (0.37 vs 0.24 units/window). Noise acts as a regularizer, not a signal the modulator learns to exploit. |
| v7 | LayerNorm alone explains most of the survival advantage previously attributed to FiLM. Under NoNoise conditions, modulator adds only +3 to +8 steps for MC. |
| v8 | Under noise, the modulator provides **zero** benefit for MC and **actively hurts** GAE (-18 to -52 steps). FiLM γ settles near identity (~0.92–0.97 for g1/g4). |
| v8 | All MC FiLM runs saturate temperature at 3.0 ceiling — the temperature injection site is non-functional. |
| v8 | Noise penalty is real (MC: -69 steps, GAE: -49 steps vs NoNoise) but the modulator cannot exploit it. The gap exists but is not addressable by the current noise profile. |

#### The Current Active Noise Config (post-v8)

The current `configs/environment/default.yaml` has been updated since v8:

| Modality | Mode | σ_base | α | Notes |
|----------|------|--------|---|-------|
| Injury | state_dependent | 0.1 | 1.5 | |
| Nutrition | state_dependent | 0.1 | 1.5 | |
| Satiation | state_dependent | 0.1 | 1.5 | |
| Extero Nociception | state_dependent | 0.1 | 1.5 | |
| Olfaction | state_dependent | 0.2 | 1.5 | |
| Collision | constant | 0.01 | 0.0 | |
| Proprioception | constant | 0.05 | 0.0 | |
| Visual | state_dependent | 0.2 | 1.5 | |
| Location | constant | 0.01 | 0.0 | |

This is identical to the v8 profile — the same problems apply.

#### The Noise Index Mismatch Bug (RESOLVED)

A noise modality index mismatch bug was discovered and fixed (see [MEMORY.md](../../.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md)). `_parse_noise_config` now iterates modalities in YAML key order and emits a `noise_modality_order` tuple that `sensor.py` consumes by name. Every modality now receives the σ/mode/clip it was configured with. This fix is a prerequisite for Phase 1 — without it, noise tuning would be meaningless since modalities would receive wrong parameters.

---

## Implementation Plan

### Design

Phase 1 is **tuning, not engineering**. The environment already supports everything needed — state-dependent noise, per-modality σ/mode/clip, and injury-scaled noise. The work is:

1. **Design a heterogeneous noise profile** with sharp reliability contrast across channels.
2. **Validate the profile against G1** by sweeping on the LayerNorm baseline.
3. **Lock the canonical noise preset** for all subsequent phases.

The design is guided by three principles from the project plan (§4, Phase 1):

- **Heterogeneous per-modality noise** — sharp reliability contrast between channels. The precision head (Phase 3) needs a landscape where some channels are trustworthy and others are not — not a flat landscape where everything is equally noisy.
- **State-dependent, injury-scaled noise on threat-relevant channels** — reliability itself becomes time-varying and correlated with interoceptive state. This is the signal the recurrent neuromodulatory core can latch onto (per [NEUROMODULATION_ALGORITHM.md §1.4](../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) hypotheses H1–H5).
- **Magnitudes tuned against G1** — the baseline must bleed. Sweep σ_base and injury_scale on the unmodulated LayerNorm baseline (no modulation) across ≥3 seeds.

### Proposed Noise Profile: "Sharp Contrast"

The proposed profile maximizes **reliability contrast** across modalities by grouping them into three tiers:

| Tier | Channels | Design Rationale |
|------|----------|-----------------|
| **High noise, injury-scaled** | Olfaction, Visual | Primary navigation senses. Biologically, these degrade most under injury/stress. Highest σ_base and highest α — these channels become unreliable when injured. |
| **Moderate noise, injury-scaled** | Extero Nociception, Injury (interoceptive) | Threat-relevant interoceptive channels. Moderate baseline noise with injury scaling — pain signals become less precise as injury worsens (cf. clinical pain literature on nociceptive precision loss). |
| **Low/zero noise, constant** | Collision, Proprioception, Location, Nutrition, Satiation | Reliable channels. Collision is binary (touched or not), proprioception is the agent's own action encoding, location is spatial. Nutrition and satiation are internal counters with minimal noise. These serve as the "anchor" channels a precision head would learn to trust. |

**Proposed YAML values (starting point for sweep):**

```yaml
perceptual_noise:
  enabled: true
  modalities:
    # === HIGH NOISE, INJURY-SCALED (Tier 1) ===
    olfaction:
      mode: "state_dependent"
      sigma: 0.3              # ↑ from 0.2 — high baseline noise
      injury_noise_scale: 3.0  # ↑ from 1.5 — strong injury degradation
      clip_min: 0.0
      clip_max: 100.0
    visual:
      mode: "state_dependent"
      sigma: 0.3              # ↑ from 0.2
      injury_noise_scale: 4.0  # ↑ from 1.5 — strongest injury degradation
      clip_min: 0.0
      clip_max: 100.0

    # === MODERATE NOISE, INJURY-SCALED (Tier 2) ===
    extero_nociception:
      mode: "state_dependent"
      sigma: 0.15             # ↑ from 0.1
      injury_noise_scale: 2.0  # ↑ from 1.5
      clip_min: 0.0
      clip_max: 100.0
    injury:
      mode: "state_dependent"
      sigma: 0.1              # unchanged
      injury_noise_scale: 2.5  # ↑ from 1.5
      clip_min: 0.0
      clip_max: 1.0

    # === LOW/ZERO NOISE, CONSTANT (Tier 3 — anchor channels) ===
    nutrition:
      mode: "constant"         # CHANGED from state_dependent → constant
      sigma: 0.02             # ↓ from 0.1 — reliable internal counter
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    satiation:
      mode: "constant"         # CHANGED from state_dependent → constant
      sigma: 0.02             # ↓ from 0.1 — reliable internal counter
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    collision:
      mode: "constant"
      sigma: 0.01             # unchanged — near-perfect binary signal
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    proprioception:
      mode: "constant"
      sigma: 0.02             # ↓ from 0.05 — agent's own action encoding
      injury_noise_scale: 0.0
      clip_min: 0.0
      clip_max: 1.0
    location:
      mode: "constant"
      sigma: 0.01             # unchanged — reliable spatial signal
      injury_noise_scale: -1.0
      clip_max: 1.0
```

**Key changes from current config:**

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| Olfaction σ | 0.2 | 0.3 | Increase baseline noise to degrade navigation |
| Olfaction α | 1.5 | 3.0 | Strong injury-dependent degradation (σ_eff at Î=1: 0.2×2.5=0.5 → 0.3×4.0=1.2) |
| Visual σ | 0.2 | 0.3 | Match olfaction tier |
| Visual α | 1.5 | 4.0 | Strongest injury scaling — vision severely degrades under injury |
| Extero Nociception mode | state_dependent (α=1.5) | state_dependent (α=2.0) | Moderate injury-scaled — pain signals degrade but less than distance senses |
| Injury α | 1.5 | 2.5 | Interoceptive pain sensing degrades under injury |
| Nutrition mode | state_dependent | **constant** | Internal counters should not degrade with injury — the agent knows how hungry it is |
| Nutrition σ | 0.1 | 0.02 | Low noise — reliable anchor channel |
| Satiation mode | state_dependent | **constant** | Same rationale as nutrition |
| Satiation σ | 0.1 | 0.02 | Low noise — reliable anchor channel |
| Proprioception σ | 0.05 | 0.02 | Agent's own action encoding should be reliable |

**Effective noise summary under the proposed profile:**

| Modality | σ_eff (healthy, Î=0) | σ_eff (injured, Î=1) | Contrast Ratio |
|----------|---------------------|---------------------|----------------|
| Olfaction | 0.30 | **1.20** | 4.0× |
| Visual | 0.30 | **1.50** | 5.0× |
| Extero Nociception | 0.15 | **0.45** | 3.0× |
| Injury | 0.10 | **0.35** | 3.5× |
| Nutrition | 0.02 | 0.02 | 1.0× (stable) |
| Satiation | 0.02 | 0.02 | 1.0× (stable) |
| Collision | 0.01 | 0.01 | 1.0× (stable) |
| Proprioception | 0.02 | 0.02 | 1.0× (stable) |
| Location | 0.01 | 0.01 | 1.0× (stable) |

This creates a **4-tier reliability landscape**: very unreliable (olfaction/visual when injured), moderately unreliable (nociception/injury when injured), mildly noisy (olfaction/visual when healthy), and highly reliable (all constant channels). The healthy-to-injured contrast ratio ranges from 1.0× (anchor channels) to 5.0× (visual), giving a precision head something to discriminate.

### Experimental Design: G1 Sweep

**Goal:** Find a noise profile where the unmodulated LN baseline's survival drops by a meaningful margin vs the no-noise condition, seed-stably.

**Phase 1a — Validate the proposed profile:**

| Run | Config | Seeds | Purpose |
|-----|--------|-------|---------|
| 1–3 | Unmodulated + LN + **NoNoise** | 3 | No-noise baseline (reference) |
| 4–6 | Unmodulated + LN + **Sharp Contrast** (proposed above) | 3 | Test proposed profile |

**Controlled variables (all runs):**

```yaml
agent:
  use_layer_norm: true
  activation: relu
  rnn_type: GRU
  hidden_size: 128
  encoding_mode: hierarchical
  return_mode: MC           # MC is the reference return mode (v7/v8)
  lr_actor: 0.0005
  lr_critic: 0.0001
  K_epochs: 4
  gamma: 0.95
  gae_lambda: 0.95
  entropy_coef: 0.01
  eps_clip: 0.1
  sequence_length: 128
  num_envs: 128
modulation:
  type: null                # Unmodulated baseline — no modulator
environment:
  grid: 10x10, max_steps: 500
  bush: 2/quadrant, pred: 4, food: 2/quadrant
  danger: 8 (2/quad, damage [15,45])
  rocks: 12 (3/quad, damage [0.1,0.5])
  neutral: 5 rabbits
body:
  food_nutrition_gain: 6    # NutGain6 (v5+ standard)
```

**G1 gate criteria:**

The proposed noise profile passes G1 if:
- **Mean survival drop ≥ 30 steps** (NoNoise baseline − Sharp Contrast baseline), measured at steady-state across ≥3 seeds.
- **The drop is seed-stable**: the worst-seed noise run must still be below the best-seed NoNoise run.
- Reference from v8: MC unmodulated went from 352 (NoNoise, v7) → 284 (Noise, v8), a 69-step drop (19.5%). The v8 profile was already close to sufficient — the Sharp Contrast profile should produce an equal or larger drop.

**Phase 1b — If the drop is too small or too large:**

If the proposed profile produces <30 steps of drop, increase σ_base on Tier 1 channels (olfaction, visual) in increments of 0.1, or increase α in increments of 1.0. If the drop is >120 steps (agent cannot survive at all), decrease σ_base on Tier 1 channels.

The target is a **sweet spot**: noise hurts enough to create headroom for precision modulation (~50–90 steps of drop), but not so much that the task becomes impossible (agent still survives >200 steps on average).

| Profile Variant | Olfaction (σ, α) | Visual (σ, α) | When to use |
|----------------|------------------|---------------|-------------|
| Sharp Contrast (default) | (0.3, 3.0) | (0.3, 4.0) | Start here |
| Sharp Contrast Mild | (0.2, 2.5) | (0.2, 3.0) | If default drop > 120 steps |
| Sharp Contrast Severe | (0.4, 4.0) | (0.4, 5.0) | If default drop < 30 steps |

**Phase 1c — Confirm with GAE (optional):**

If Phase 1a passes G1 with MC, run 3 additional seeds with GAE to verify the noise penalty is also present for the alternative return mode. This is informational — MC is the primary return mode per v7/v8, but GAE data helps Phase 2 characterization.

### File Changes

#### `configs/environment/default.yaml` (lines ~258–316)

Replace the `perceptual_noise` section with the Sharp Contrast profile (YAML block shown above in the Proposed Noise Profile section). The exact values may be adjusted during the Phase 1b sweep — the implementing agent should use whichever profile variant passes G1.

**Config keys changed:**

| YAML Path | Current | New |
|-----------|---------|-----|
| `perceptual_noise.modalities.olfaction.sigma` | 0.2 | 0.3 |
| `perceptual_noise.modalities.olfaction.injury_noise_scale` | 1.5 | 3.0 |
| `perceptual_noise.modalities.visual.sigma` | 0.2 | 0.3 |
| `perceptual_noise.modalities.visual.injury_noise_scale` | 1.5 | 4.0 |
| `perceptual_noise.modalities.extero_nociception.sigma` | 0.1 | 0.15 |
| `perceptual_noise.modalities.extero_nociception.injury_noise_scale` | 1.5 | 2.0 |
| `perceptual_noise.modalities.injury.injury_noise_scale` | 1.5 | 2.5 |
| `perceptual_noise.modalities.nutrition.mode` | state_dependent | constant |
| `perceptual_noise.modalities.nutrition.sigma` | 0.1 | 0.02 |
| `perceptual_noise.modalities.nutrition.injury_noise_scale` | 1.5 | 0.0 |
| `perceptual_noise.modalities.satiation.mode` | state_dependent | constant |
| `perceptual_noise.modalities.satiation.sigma` | 0.1 | 0.02 |
| `perceptual_noise.modalities.satiation.injury_noise_scale` | 1.5 | 0.0 |
| `perceptual_noise.modalities.proprioception.sigma` | 0.05 | 0.02 |

No source code changes required. All parameters are configurable through YAML.

---

## Checkpoints

What the implementing agent should verify **during** implementation:

- [ ] **Checkpoint 1 — Noise index mapping is correct.** After updating the YAML, run a single-episode test and print the effective σ per modality (from `sensor.py`). Verify that olfaction gets σ=0.3, visual gets σ=0.3, nutrition gets σ=0.02, etc. The noise index mismatch bug was fixed, but this confirms the fix holds with the new config.
- [ ] **Checkpoint 2 — Injury-scaled noise actually scales.** In the single-episode test, set injury to 50% (Î=0.5) and print σ_eff for each modality. Olfaction should show σ_eff = 0.3 × (1 + 3.0 × 0.5) = 0.75. Visual should show σ_eff = 0.3 × (1 + 4.0 × 0.5) = 0.90. Constant channels should be unchanged.
- [ ] **Checkpoint 3 — NoNoise baseline runs complete.** 3 seeds, unmodulated + LN + NoNoise. Record mean survival steps ± std across seeds.
- [ ] **Checkpoint 4 — Sharp Contrast baseline runs complete.** 3 seeds, unmodulated + LN + Sharp Contrast profile. Record mean survival steps ± std across seeds.
- [ ] **Checkpoint 5 — G1 gate evaluation.** Compute: (NoNoise mean steps) − (Sharp Contrast mean steps). If ≥ 30 and seed-stable → G1 passes. If < 30 → escalate to Phase 1b (increase noise). If > 120 → escalate to Phase 1b (decrease noise).

---

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the implementing agent after config changes and training runs are made.
     Describe what was done, any deviations from the plan, and why.
     Record the G1 result: NoNoise steps (mean ± std, N seeds) vs Sharp Contrast steps (mean ± std, N seeds).
     If Phase 1b was needed, record which variant was used and why. -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/environment/default.yaml` | Noise profile update | | |

**G1 Result:**

| Condition | Seed 1 | Seed 2 | Seed 3 | Mean ± Std |
|-----------|--------|--------|--------|------------|
| NoNoise | | | | |
| Sharp Contrast | | | | |
| **Δ (drop)** | | | | |

**G1 Pass?**: [ ] Yes / [ ] No — drop = ___ steps

**Conclusion**: [one-line summary]

---

<!--
After G1 is passed, cross-link:
- The chosen canonical noise preset back to this document
- Forward-link to Phase 2 (docs/project/phase_2_film_characterization.md) when created
- Update project_plan.md §4 Phase 1 exit gate with the result
-->
