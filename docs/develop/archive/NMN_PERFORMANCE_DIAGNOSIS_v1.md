---
title: "NMN Performance Diagnosis: Why the Modulated Agent Matches Baseline"
topic: diagnosis
status: superseded
created: 2026-03-03
last_updated: 2026-04-12
superseded_by: NMN_PERFORMANCE_DIAGNOSIS_v2.md
---

# NMN Performance Diagnosis: Why the Modulated Agent Matches Baseline

## 1. Problem Statement

The Neuromodulated Recurrent PPO (NMN) agent shows **equivalent performance** to the unmodulated baseline — no measurable improvement in episode reward, survival, or learning speed. The theoretical expectation was that the modulator would learn injury-adaptive behavior (hypervigilance, cautious exploration, memory persistence) that outperforms a fixed-architecture agent, particularly in environments with state-dependent sensory noise and nociceptive threats.

This document diagnoses the likely causes, organized from **most impactful** to **least impactful**, and proposes concrete experiments to resolve each.

> **Updated 2026-03-06**: Both runs have completed. Data refreshed from final WandB metrics (~14B timesteps each, ~72–89h wall-clock). All pathological trends from the mid-training analysis **worsened** with extended training, confirming the diagnosis.

---

## 2. WandB Empirical Analysis

### 2.1 Runs Compared

| | NMN | Baseline |
|---|---|---|
| **Run** | `20260301-213626_rppoNMN_MC_relu_128hidden_GRU_hierarchical` | `20260302-143850_rppo_MC_relu_128hidden_GRU_hierarchical` |
| **WandB ID** | `uyhuypji` | `ddrpu91m` |
| **Status** | **Finished** (89h 16m) | **Finished** (72h 11m) |
| **Timesteps** | **14.01B** | **13.97B** |
| **Episodes** | 55.5M | 57.4M |

### 2.2 Config Parity Check (from WandB)

These runs are **well-controlled** — most hyperparameters match:

| Parameter | NMN | Baseline | Match? |
|---|---|---|---|
| `activation` | `tanh` | `relu` | **MISMATCH** |
| `return_mode` | `MC` | `MC` | OK |
| `hidden_size` | `128` | `128` | OK |
| `hierarchical_params` | `default_mlp=[128,128]`, `visual=[128,128]`, hub_overrides: `body_state=[128,128]`, `association=[128,128]` | Same | OK |
| `K_epochs` | `4` | `4` | OK |
| `gamma` | `0.95` | `0.95` | OK |
| `entropy_coef` | `0.01` | `0.01` | OK |
| `lr_actor` | `0.0005` | `0.0005` | OK |
| `modulation.type` | `Multiplicative` | `null` | Intended variable |
| `modulation.grouping_size` | `40` | — | — |

**Finding**: The actual WandB runs use matched encoder configs (`default_mlp=[128,128]`, `visual=[128,128]`, same hub overrides), unlike the on-disk YAML files which differed. The only remaining confound is `activation: tanh` (NMN) vs `relu` (baseline). This is a minor confound but should still be fixed in future runs.

### 2.3 Performance Comparison — Episode Metrics

| Metric | NMN (steady-state) | Baseline (steady-state) | Verdict |
|---|---|---|---|
| **Episode/Reward** | **-208.53 ± 1.77** | **-209.09 ± 1.94** | Effectively identical |
| Episode/Reward_Max | -185.01 ± 15.40 | -190.43 ± 13.06 | NMN slightly better peak |
| Episode/Reward_Min | -238.82 ± 2.78 | -239.59 ± 2.11 | Identical |
| **Episode/Steps** | **259.56 ± 13.00** | **240.42 ± 21.28** | NMN survives ~19 steps longer on average |
| Reward trajectory | -215.31 → -206.27 | -209.64 → -210.60 | NMN improved ~9 pts; Baseline flat/slightly worsened |

**Conclusion**: With full training, the NMN shows a **marginal edge** — ~0.5 points better steady-state reward and ~19 more survival steps. However, this is within noise bounds and **far below** what adaptive modulation should achieve. The NMN's trajectory improved more than the baseline's (9 pts vs flat), but the final performance remains in the same band (~-207 to -211). Neither agent demonstrates strong learning — both plateau early and show minimal improvement over 14B timesteps.

**New observation (vs mid-training)**: The baseline's reward trajectory is essentially **flat** (-209.64 → -210.60), while the NMN's trajectory shows a slow upward drift (-215.31 → -206.27). This suggests the NMN is extracting marginal signal, possibly from the unimodal gates it kept open, but the benefit is negligible.

### 2.4 Training Speed

| Run | s/it | SPS | Wall-Clock | Overhead |
|---|---|---|---|---|
| NMN | 0.374s | 52,669 | 89h 16m | +23.6% slower |
| Baseline | 0.303s | 58,284 | 72h 11m | — |

The modulator adds **~24% wall-clock overhead** for negligible performance gain. SPS values are higher than the mid-training report due to JIT warmup effects stabilizing over the full run.

### 2.5 Loss Curves

| Metric | NMN (steady-state) | Baseline (steady-state) | Note |
|---|---|---|---|
| loss/total | 0.1035 ± 0.054 | 0.1054 ± 0.042 | Identical |
| loss/policy | -0.0006 ± 0.053 | 0.0006 ± 0.039 | Both near zero |
| loss/value | 0.2171 ± 0.018 | 0.2188 ± 0.021 | Identical |
| loss/entropy | -0.4521 ± 0.026 | -0.4617 ± 0.031 | NMN slightly higher entropy (see §2.6) |
| loss/grad_norm | 0.3996 ± 0.447 | 0.5672 ± 0.919 | Baseline has higher grad norm variance; baseline max spike 269.96 vs NMN max 21.67 |

Losses are functionally identical. The value loss plateaus around 0.22 for both — the critic is equally imprecise.

**New observation**: The baseline shows occasional gradient norm spikes (max 269.96 vs NMN's 21.67). The modulator may be providing a mild gradient-smoothing effect by absorbing some loss variance through its own parameters, though this doesn't translate to performance.

### 2.6 Modulator Metrics — IS the Modulator Learning?

**Yes — the modulator is learning aggressively, but not helpfully. All pathological trends from mid-training worsened.**

| Metric | Init | Mid-Training SS | **Final SS** | **Final** | Interpretation |
|---|---|---|---|---|---|
| `gamma_uni_mean` | 1.99 | 3.12 ± 0.33 | **5.30 ± 0.40** | **5.23** | **Further increased** — sigmoid(5.23) ≈ 0.995. Unimodal gates now essentially hard-open (was 0.98 mid-training). |
| `gamma_uni_std` | 0.77 | 5.34 ± 0.34 | **6.80 ± 0.31** | **6.25** | **Variance grew further** — unimodal signals vary even more wildly |
| `gamma_body_mean` | 3.03 | -4.26 ± 0.15 | **-4.13 ± 0.14** | **-3.92** | **Stable at collapse** — sigmoid(-3.92) ≈ 1.9%. Body-state features remain **shut off**. |
| `gamma_assoc_mean` | 3.18 | -11.48 ± 0.32 | **-14.19 ± 0.26** | **-14.67** | **Worsened** — sigmoid(-14.67) ≈ 0.000004%. Association hub pushed even deeper into suppression. |
| `z_memory_mean` | -0.09 | -5.41 ± 0.61 | **-6.45 ± 0.37** | **-5.48** | **Worsened** — GRU update gate biased even more toward 0. Memory is frozen. |
| `z_memory_std` | 0.87 | 1.47 ± 0.29 | **1.43 ± 0.35** | **2.05** | Increasing variance — some timesteps may have partially active memory |
| `temperature_mean` | 0.70 | 2.13 ± 0.16 | **3.09 ± 0.24** | **3.29** | **Worsened significantly** — policy is now 3.3x more stochastic (was 2.4x). |
| `temperature_max` | 1.13 | 8.53 ± 0.88 | **10.00 ± 0.04** | **10.00** | **Pinned at ceiling** — near-permanent max temperature for some timesteps |
| `temperature_min` | 0.53 | 0.54 ± 0.02 | **0.56 ± 0.04** | **0.54** | Near floor — wide temperature spread persists |
| `modulator/grad_norm` | 0.04 | 0.236 ± 0.098 | **0.298 ± 0.15** | **0.22** | Modulator still receiving gradients, stable |

### 2.7 Critical Findings from Modulator Metrics

**Finding 1 — Feature Collapse DEEPENED (body_state and association gates)**

The association gate went from sigmoid(-12) ≈ 0.0006% at mid-training to sigmoid(-14.67) ≈ 0.000004% at completion — a **150x further suppression**. The body-state gate remained stable at sigmoid ≈ 1.6–1.9%. The modulator continued to aggressively suppress interoceptive and cross-modal features throughout the full training run.

- Interoceptive signals (injury, satiation, nutrition) remain **completely silenced**
- The association hub (cross-modal fusion) is **functionally dead**
- Unimodal gates pushed further open (sigmoid ≈ 0.995), confirming the agent relies only on individual sensory channels

This is the **opposite** of the hypothesis — rather than learning to upregulate pain-related features, the modulator learned to suppress them entirely. The agent is effectively operating without its interoceptive senses and cross-modal integration.

**Why**: Suppressing noisy or unhelpful features can reduce variance in the value function, which PPO rewards. If interoceptive signals are noisy (constant noise on satiation/nutrition) and the reward signal is dominated by survival (avoiding death penalty), the optimal strategy may be to ignore interoception entirely. The modulator discovered this shortcut.

**Finding 2 — Memory Gate Freezing WORSENED (z_memory → -6.5)**

A strongly negative `z_memory` biases the GRU update gate toward 0:
$$u_t = \sigma(W_u x + U_u h + z_{memory}) \approx \sigma(\cdot - 6.5) \approx 0$$

This means $h_{new} \approx (1 - 0) \cdot h_{prev} + 0 \cdot \tilde{h} = h_{prev}$. The GRU **stops updating its hidden state**. The agent's recurrent memory is frozen — it's operating as a near-feedforward network. The z_memory value drifted from -5.41 (mid-training) to -6.45 (steady-state), showing continued deepening.

**Why**: If the modulator cannot improve task performance through adaptive modulation, freezing the hidden state is a stable local minimum. A frozen hidden state produces constant value estimates (low value loss variance), which PPO's clipped objective tolerates.

**Finding 3 — Temperature Inflation WORSENED SIGNIFICANTLY (mean = 3.29, ceiling pinned)**

The modulator pushed temperature from 2.43 (mid-training) to **3.29** at completion — a **35% further increase**. The temperature_max is now **permanently pinned at 10.0** (std = 0.04). Higher temperature means softer action distributions, which:
- Increases entropy (confirmed: NMN entropy loss = -0.452 vs baseline = -0.462, NMN is more entropic)
- Counteracts the entropy coefficient's natural decay
- Compensates for the frozen GRU (if the hidden state doesn't update, the agent needs random exploration to cover the state space)

The temperature ceiling of 10.0 is now a **hard constraint** — the modulator would push it higher if allowed. This confirms the bounds are far too wide.

**Finding 4 — The modulator IS receiving gradient signal**

`modulator/grad_norm` stabilized at ~0.30 (steady-state). The modulator is not starved of gradients — it learned these extreme values deliberately through optimization. The gradient signal remained stable, confirming these pathological values are **intentional optima** for the modulator, not artifacts of gradient death.

---

## 3. Updated Config Confound Analysis

### 3.1 Remaining Confound: activation mismatch

The WandB configs confirm that the actual runs share identical encoder architectures, but differ in `activation: tanh` (NMN) vs `relu` (baseline). This is a minor confound — tanh saturates in [-1,1] while relu is unbounded positive. However, given that both runs achieve essentially identical reward, this is unlikely to be a dominant factor.

### 3.2 Recommended Fix: Controlled Configs

Future runs should use identical configs where the ONLY difference is `modulation.type`:

```yaml
# configs/models/rppo_baseline_controlled.yaml
agent:
  activation: "relu"
  return_mode: "MC"
  max_grad_norm: 0.5
  hidden_size: 128
  hierarchical_params:
    default_mlp: [128, 128]
    unimodal_overrides:
      visual: [128, 128]
    hub_overrides:
      body_state: [128, 128]
      association: [128, 128]
  modulation:
    type: null             # <-- ONLY this differs

# configs/models/rppo_nmn_controlled.yaml
agent:
  # ... identical hyperparameters ...
  modulation:
    type: "Multiplicative"  # <-- ONLY this differs
    mod_hidden_size: 64
    grouping_size: 1         # Per-neuron (start fine-grained)
    percept_bias_init: 3.0   # Higher for truer pass-through
    memory_bias_init: 0.0
    temp_clip: [0.5, 2.0]   # Much tighter bounds
```

---

## 4. Insufficient Environmental Pressure

### 4.1 State-Dependent Noise Is Nearly Disabled

The active environment config (`configs/environment/default.yaml`) has **only 1 of 9 modalities** using `state_dependent` noise:

| Modality | Mode in `default.yaml` | Mode in `PRECISION_MODULATION.md` (Documented Ideal) |
|---|---|---|
| Olfaction | `constant` | `state_dependent` ($\alpha = 2.0$) |
| Visual | `constant` | `state_dependent` ($\alpha = 3.0$) |
| Injury | `state_dependent` ($\alpha = 1.5$) | `state_dependent` ($\alpha = 1.5$) |
| All others | `constant` | `constant` |

The PRECISION_MODULATION.md document describes the theory assuming olfaction, visual, and injury are all state-dependent. But the actual running config only has injury as state-dependent. This means:

- **The modulator has almost no injury-dependent signal to learn from.** Olfaction (the primary navigation sense) is equally noisy whether the agent is healthy or dying. Visual sensing doesn't degrade with injury. The modulator cannot learn "increase olfactory precision when healthy, decrease when injured" because olfaction precision doesn't change with injury.
- **The task doesn't differentially reward adaptive sensing.** Without state-dependent noise, a fixed-architecture agent faces the same sensory challenge regardless of injury level. There is no "perceptual regime" to adapt to.

### 4.2 Recommended Fix: Enable the Designed Noise Profile

Update `configs/environment/default.yaml` to match the documented design:

```yaml
perceptual_noise:
  enabled: true
  modalities:
    olfaction:
      mode: "state_dependent"    # Was: "constant"
      sigma: 0.15
      injury_noise_scale: 2.0    # Was: 0.0
    visual:
      mode: "state_dependent"    # Was: "constant"
      sigma: 0.05
      injury_noise_scale: 3.0    # Was: 0.0
    # ... rest unchanged ...
```

This creates the environmental pressure the modulator was designed to address.

---

## 5. Root Cause Analysis (WandB-Informed)

The modulator metrics (§2.6–2.7) reveal that the modulator IS learning — but it found a **degenerate shortcut** rather than useful adaptive behavior. Three pathological patterns emerged and **all worsened** with extended training:

### 5.1 Feature Collapse — The Modulator Kills Interoceptive Signals

`gamma_body_mean` → -3.92 (sigmoid ≈ 1.9%) and `gamma_assoc_mean` → -14.67 (sigmoid ≈ 0.000004%) mean the modulator learned to **suppress** body-state and cross-modal features entirely. The association suppression **deepened 150x** from mid-training to completion. This is rational from PPO's perspective: if interoceptive signals are noisy and don't reliably predict reward, silencing them reduces value function variance.

**Root cause**: The environment's perceptual noise is `constant` (not `state_dependent`) for satiation/nutrition, and only `injury` uses state-dependent noise. The modulator cannot learn "amplify these signals when healthy, suppress when injured" because the signals don't change with injury. Instead, it learns the simpler rule: "always suppress."

**Fix**: Enable state-dependent noise on olfaction and visual (§4) to create a regime where adaptive modulation outperforms static suppression.

### 5.2 Memory Gate Freezing — The GRU Stops Updating

`z_memory_mean` → -6.45 (steady-state) biases the GRU update gate to $\sigma(\cdot - 6.5) \approx 0$, effectively freezing $h_{prev}$. The agent loses its recurrent memory. This worsened from -5.41 at mid-training.

**Root cause**: With body-state and association features already suppressed (§5.1), the GRU hidden state carries little useful information. Freezing it is a stable local minimum — constant hidden state → constant value predictions → low value loss variance → PPO is satisfied.

**Fix**: This is downstream of §5.1. If the feature collapse is fixed, the GRU has useful information to maintain. Additionally, consider **clamping `z_memory` to [-2, +2]** to prevent the modulator from fully disabling the recurrent dynamics.

### 5.3 Temperature Inflation — Compensatory Exploration

`temperature_mean` → 3.29 (up from 2.43 mid-training), with temperature_max **permanently pinned at 10.0**. With the GRU frozen and features suppressed, the agent compensates by exploring randomly. This is a second-order pathology caused by §5.1 and §5.2.

**Root cause**: The temperature bounds `[0.1, 10.0]` are far too wide. A temperature of 10 makes the policy nearly uniform-random. The current bounds allow the modulator to effectively disable the learned policy. The modulator would push temperature even higher if allowed — 10.0 is a hard constraint, not an equilibrium.

**Fix**: Tighten temperature bounds to `[0.5, 2.0]` or even `[0.8, 1.5]`. This constrains the modulator to *fine-tuning* exploration rather than overriding the policy entirely.

### 5.4 Coarse Spatial Grouping Enables Wholesale Suppression

With `grouping_size: 40` and `hidden_size: 128`, each head output controls 32–40 neurons simultaneously. This makes it easy for the modulator to shut off entire feature groups with a single negative value — which is exactly what happened with `gamma_body` and `gamma_assoc`.

**Fix**: Use `grouping_size: 1` (per-neuron) so the modulator must make fine-grained decisions. Wholesale suppression becomes harder when each neuron requires an independent decision.

### 5.5 Shared Optimizer — No Timescale Separation

Both the task network and the modulator are trained with a single Adam optimizer at `lr=0.0005`. The NEUROMODULATION_ALGORITHM.md §4 recommends a lower learning rate for the modulator.

With equal learning rates, the modulator learns as fast as the task network, which allows it to quickly find degenerate shortcuts (§5.1–5.3) before the task network has converged enough to provide stable gradient signal.

**Fix**: Use `optax.multi_transform` to assign a 5–10x lower learning rate to modulator parameters. This forces the modulator to adapt slowly to the task network's stable features rather than racing ahead to suppress them.

---

## 6. Training Loop Issues

### 6.1 PPO Loss Re-evaluation Discards Modulation State Continuity

In `recurrent_ppo_trainer.py:ppo_loss_fn`, the scan calls `model(obs, h)` at each timestep. This re-runs the modulator from `h_init` (the stored initial hidden state). During the K_epochs=4 re-evaluations, the modulator re-processes the same trajectory each time.

However, the **rollout was collected with a different modulator trajectory** (the one that actually evolved during collection). During PPO re-evaluation, the modulator hidden state trajectory may differ slightly from the collection trajectory because the task network weights have been updated between epochs. This creates a subtle distribution mismatch between the modulator signals used during collection (which determined actions) and those computed during loss evaluation.

For the baseline (no modulator), this issue doesn't exist — the only state is the task GRU, which is re-processed identically.

**Impact**: Likely minor, but worth noting. Standard in recurrent PPO.

### 6.2 Modulator Gradient Norm Not Properly Extracted

In `update_step()`, the modulator gradient norm extraction uses:
```python
if 'modulator' in grads:
    mod_grad_norm = optax.global_norm(grads['modulator'])
```

But NNX gradients are structured as nested module states, not string-keyed dicts. This check likely never fires, meaning `mod_grad_norm` is always 0.0 in WandB logs. Without modulator gradient diagnostics, we can't tell if the modulator is receiving meaningful learning signal.

**Update**: Despite this code issue, the WandB logs show `modulator/grad_norm` growing from 0.04 to 0.30, indicating the logging path IS working (perhaps through a different code path). The modulator IS receiving gradients.

**Recommendation**: Verify the gradient extraction code path and ensure it's correctly measuring the modulator's gradient contribution.

---

## 7. Theoretical Considerations

### 7.1 The Environment May Not Need Modulation

The 10x10 grid with 4 food, 8 danger, 3 predators, and 500-step episodes may be **simple enough** that a fixed-architecture GRU can learn all necessary behaviors without adaptive modulation. The baseline can memorize:
- Danger locations via the visual sensor
- Predator avoidance via olfaction
- Food seeking via olfaction

**Evidence from final runs**: Both agents plateau at ~-209 reward and ~250 steps (half the max 500). Neither shows significant improvement over 14B timesteps. This suggests both agents have converged to a **behavioral ceiling** — the environment is not differentiating between architectures because both are equally limited by the task structure, not by their representational capacity.

**Recommendation**: If this is the case, increase environment difficulty:
- Larger grid (20x20 or 30x30)
- More danger zones
- State-dependent noise (see §4)
- Shorter episodes (forces faster adaptation)

### 7.2 The Modulator Learned a Degenerate But Stable Strategy

The full-training data confirms the modulator's strategy is not just a transient pathology but a **stable equilibrium**:
- All pathological trends continued deepening over 14B timesteps
- The modulator gradient norm remained stable (~0.30), meaning it's still receiving gradient signal but the current strategy is a local optimum
- The modulator effectively reduces the NMN to: open unimodal gates + suppress interoception + freeze memory + randomize actions

This is a valid (if undesirable) optimization outcome: the modulator simplified the agent's architecture by removing components that weren't helping, then compensated with exploration noise. The ~0.5-point reward advantage over baseline may come from the unimodal gate amplification (sigmoid ≈ 0.995 vs the baseline's implicit 1.0 — effectively identical).

---

## 8. Experiment Priority List

| Priority | Experiment | What It Tests | Effort |
|---|---|---|---|
| **P0** | Tighten temperature bounds to [0.5, 2.0] | Prevent policy override (§5.3) | Low |
| **P1** | Clamp z_memory to [-2, +2] | Prevent GRU freeze (§5.2) | Low |
| **P2** | Enable state-dependent noise for olfaction + visual (§4) | Create reason for adaptive modulation | Low |
| **P3** | Set `grouping_size: 1` for max expressiveness | Prevent wholesale feature suppression (§5.4) | Low |
| **P4** | Fix activation mismatch (both relu) | Eliminate last config confound (§3.1) | Low |
| **P5** | Lower modulator learning rate 5–10x | Enforce timescale separation (§5.5) | Medium |
| **P6** | Increase `percept_bias_init` to 3.0+ | Truer pass-through at init | Low |
| **P7** | Increase environment difficulty (larger grid, more threats) | Force the baseline to fail so modulation has room to help (§7.1) | Medium |

### Recommended First Run

Apply P0 + P1 + P2 + P3 + P4 together (all low-effort):

```yaml
modulation:
  type: "Multiplicative"
  mod_hidden_size: 64
  grouping_size: 1              # P3: per-neuron
  percept_bias_init: 3.0        # P6: truer pass-through
  memory_bias_init: 0.0
  temp_clip: [0.5, 2.0]         # P0: tight bounds
  memory_clip: [-2.0, 2.0]      # P1: prevent GRU freeze (requires code change)
```

With `activation: relu` for both (P4) and state-dependent noise on olfaction + visual (P2).

If the NMN still matches baseline, move to P5 (separate learning rate) and P7 (harder environment).

---

## 9. Summary

Both runs have **completed** (~14B timesteps, 72–89h). The full-training WandB analysis confirms and strengthens the mid-training diagnosis. The modulator **learned aggressively but pathologically**, and all pathological trends **deepened** with extended training:

1. **Feature collapse DEEPENED**: Body-state gates → sigmoid(-3.92) ≈ 1.9%, association gates → sigmoid(-14.67) ≈ 0.000004% (150x worse than mid-training). The modulator discovered that suppressing interoceptive signals reduces variance, which PPO rewards.
2. **Memory freeze WORSENED**: z_memory → -6.45 steady-state (was -5.41 mid-training), forcing GRU update gate to ~0. The recurrent memory is effectively disabled.
3. **Temperature inflation WORSENED**: Mean temperature 3.29x (was 2.43x mid-training), max **permanently pinned at ceiling 10.0**. The agent compensates for frozen memory by exploring randomly. The modulator would push temperature higher if bounds allowed.
4. **~24% speed overhead** for negligible performance benefit (~0.5 reward points, within noise).
5. **Stable equilibrium**: The modulator gradient norm remained stable (~0.30), confirming these extreme values are intentional local optima, not artifacts. Extended training did not self-correct — it made things worse.

The core problem is that **the environment doesn't create sufficient pressure for adaptive modulation**. With mostly constant noise and a 10x10 grid, static feature suppression is as good as (or better than) dynamic modulation. The modulator found the easiest gradient-reducing shortcut instead of learning the intended hypervigilance behavior.

**Fix order**: Constrain the modulator's action space (tighter bounds on temperature and memory) → create environmental pressure (state-dependent noise) → refine architecture (grouping, learning rate) → increase task difficulty if needed.

---

## 10. Ablation Study: Grouping Size × Modulation Type × Mod Hidden Size

> **Updated 2026-03-07**: New ablation sweep with 11 NMN variants + 1 baseline, all running on a revised environment (128 envs, 2 bush quadrants, 4 predators, 1 food per quadrant). All runs are **still running** (~14h, ~2.5B timesteps). Results below are mid-training snapshots — trends are established but final values will change.

### 10.1 Experiment Design

All runs share identical environment and agent hyperparameters (see §10.2). The only variables are:

| Variable | Values Tested |
|---|---|
| **Modulation type** | Multiplicative (7 runs), PreActivation (4 runs), null/Baseline (1 run) |
| **Grouping size** | 20, 40, 60, 80, 100 (Multiplicative default mod_hidden=32); 20, 60 (both types with mod_hidden=16 or 32) |
| **Mod hidden size** | 16 (explicit in name), 32 (default, implicit when omitted) |

### 10.2 Shared Environment Config

```yaml
environment:
  width: 10, height: 10
  max_steps: 500
  predators: 4 (damage [15,45], detection_range=3, attack_delay=3)
  food: 4 (1 per quadrant, max_consumption=35, regen_delay=250)
  danger: 8 (2 per quadrant, damage [15,45])
  rocks: 12 (3 per quadrant, damage [0.1,0.5])
  bushes: 8 (2 per quadrant, hides_agent=true)
  neutral_animals: 5 rabbits

agent:
  algorithm: RecurrentPPO
  activation: relu
  return_mode: GAE
  rnn_type: GRU
  hidden_size: 128
  encoding_mode: hierarchical
  lr_actor: 0.0005, lr_critic: 0.0001
  K_epochs: 4, gamma: 0.95, gae_lambda: 0.95
  entropy_coef: 0.01, eps_clip: 0.1
  sequence_length: 128, num_envs: 128

perceptual_noise:
  enabled: true
  injury: state_dependent (sigma=0.05, alpha=1.5)
  all others: constant
```

**Key difference from previous runs (§2)**: This environment uses `return_mode: GAE` (vs MC), 4 predators (vs 3), and explicitly matched configs across all runs — no activation mismatch confound.

### 10.3 Run Inventory

| Tag | WandB ID | Type | G | mod_h | Timesteps | Status |
|---|---|---|---|---|---|---|
| `rppo_128env_2bush4pred1food` | `a1h6g56v` | Baseline | — | — | 1.89B | Running |
| `rppoNMN_..._gSize20_multiplicative` | `y43bcakx` | Mult | 20 | 32 | 2.76B | Running |
| `rppoNMN_..._gSize40_multiplicative` | `qfi6qfzy` | Mult | 40 | 32 | 2.47B | Running |
| `rppoNMN_..._gSize60_multiplicative` | `pu438wd8` | Mult | 60 | 32 | 2.68B | Running |
| `rppoNMN_..._gSize80_multiplicative` | `8x99hsgr` | Mult | 80 | 32 | 2.71B | Running |
| `rppoNMN_..._gSize100_multiplicative` | `pmmg589y` | Mult | 100 | 32 | 2.67B | Running |
| `rppoNMN_..._gSize20_multiplicative_modHidden16` | `imavc7fo` | Mult | 20 | 16 | 2.52B | Running |
| `rppoNMN_..._gSize60_multiplicative_modHidden16` | `2tw7n7nj` | Mult | 60 | 16 | 2.77B | Running |
| `rppoNMN_..._gSize20_preActivation_modHidden16` | `v1pdnlkf` | PreAct | 20 | 16 | 2.59B | Running |
| `rppoNMN_..._gSize60_preActivation_modHidden16` | `vbkeujm3` | PreAct | 60 | 16 | 2.55B | Running |
| `rppoNMN_..._gSize20_preActivation_modHidden32` | `v76stpj6` | PreAct | 20 | 32 | 2.50B | Running |
| `rppoNMN_..._gSize60_preActivation_modHidden32` | `1jb899lc` | PreAct | 60 | 32 | 2.44B | Running |

### 10.4 Episode Performance Comparison

| Run | Reward (SS) | Steps (SS) | loss/value (SS) | loss/total (SS) |
|---|---|---|---|---|
| **Baseline** | **-113.24 ± 2.69** | **105.12 ± 2.75** | **63.90 ± 11.6** | **31.95 ± 5.8** |
| Mult G=20 h=32 | -113.29 ± 2.73 | 97.25 ± 1.68 | 66.65 ± 13.2 | 33.32 ± 6.6 |
| Mult G=40 h=32 | -114.98 ± 2.78 | 111.42 ± 3.32 | 65.21 ± 11.1 | 32.60 ± 5.5 |
| Mult G=60 h=32 | -118.46 ± 3.21 | 112.85 ± 3.57 | 78.61 ± 12.5 | 39.30 ± 6.2 |
| Mult G=80 h=32 | -117.57 ± 3.03 | 116.36 ± 3.45 | 70.30 ± 11.1 | 35.14 ± 5.5 |
| Mult G=100 h=32 | -113.83 ± 2.63 | 103.04 ± 2.42 | 63.44 ± 11.1 | 31.71 ± 5.6 |
| **Mult G=20 h=16** | -115.14 ± 2.78 | **121.45 ± 3.97** | 61.36 ± 9.7 | 30.67 ± 4.9 |
| **Mult G=60 h=16** | **-108.42 ± 1.91** | 98.86 ± 1.35 | **44.48 ± 9.4** | **22.23 ± 4.7** |
| PreAct G=20 h=16 | -118.68 ± 3.25 | 118.65 ± 3.99 | 73.11 ± 10.4 | 36.55 ± 5.2 |
| PreAct G=60 h=16 | -116.65 ± 2.87 | **122.09 ± 3.81** | 65.94 ± 10.0 | 32.97 ± 5.0 |
| PreAct G=20 h=32 | -113.01 ± 2.48 | 97.73 ± 1.52 | 63.63 ± 11.2 | 31.81 ± 5.6 |
| PreAct G=60 h=32 | -112.82 ± 2.44 | 97.63 ± 1.54 | 63.39 ± 11.3 | 31.69 ± 5.7 |

**Key observations**:
- **Best reward**: `Mult G=60 h=16` at **-108.42** — ~5 points better than baseline (-113.24). Also has the lowest value loss (44.48 vs 63.90).
- **Longest survival**: `PreAct G=60 h=16` at **122.09 steps** and `Mult G=20 h=16` at **121.45 steps** — both ~16 steps longer than baseline (105.12).
- **Most baseline-like**: `PreAct G=60 h=32` (-112.82) and `PreAct G=20 h=32` (-113.01) are indistinguishable from baseline.
- **Worst reward**: `Mult G=60 h=32` (-118.46) and `PreAct G=20 h=16` (-118.68) — ~5 points worse than baseline.
- **mod_hidden=16 generally outperforms mod_hidden=32** in Multiplicative mode, suggesting the smaller modulator is more constrained and less prone to degenerate shortcuts.

### 10.5 Modulator Behavior Comparison

#### 10.5.1 Gamma (Perceptual Gain) — Unimodal

| Run | gamma_uni_mean (SS) | sigmoid(gamma) | Interpretation |
|---|---|---|---|
| Mult G=20 h=32 | -2.447 | 8.0% | Suppressed |
| Mult G=40 h=32 | -4.459 | 1.1% | **Hard suppressed** |
| Mult G=60 h=32 | -1.856 | 13.5% | Suppressed |
| Mult G=80 h=32 | -3.030 | 4.6% | Hard suppressed |
| Mult G=100 h=32 | -2.664 | 6.5% | Suppressed |
| **Mult G=20 h=16** | **-1.004** | **26.8%** | **Moderate gating** |
| **Mult G=60 h=16** | **-0.644** | **34.5%** | **Mild gating** |
| PreAct G=20 h=16 | 1.327 | 79.0% | Near pass-through |
| PreAct G=60 h=16 | 0.840 | 69.8% | Moderate pass-through |
| PreAct G=20 h=32 | 0.315 | 57.8% | Mild gating |
| PreAct G=60 h=32 | 0.326 | 58.1% | Mild gating |

**Finding**: All Multiplicative h=32 runs suppress unimodal features (sigmoid < 15%). The h=16 variants are much less aggressive (27–35%). PreActivation runs maintain near-pass-through or mild gating — the beta term provides an alternative pathway, reducing pressure to suppress via gamma.

#### 10.5.2 Gamma (Perceptual Gain) — Multimodal

| Run | gamma_multi_mean (SS) | sigmoid(gamma) | Interpretation |
|---|---|---|---|
| Mult G=20 h=32 | -2.351 | 8.7% | Suppressed |
| Mult G=40 h=32 | -1.333 | 20.9% | Moderate suppression |
| Mult G=60 h=32 | -1.778 | 14.5% | Suppressed |
| Mult G=80 h=32 | -1.852 | 13.6% | Suppressed |
| Mult G=100 h=32 | -1.985 | 12.1% | Suppressed |
| **Mult G=20 h=16** | **-5.753** | **0.3%** | **Collapsed** |
| **Mult G=60 h=16** | **-6.956** | **0.1%** | **Collapsed** |
| PreAct G=20 h=16 | -7.235 | 0.07% | Collapsed |
| PreAct G=60 h=16 | -7.473 | 0.06% | Collapsed |
| PreAct G=20 h=32 | -7.175 | 0.08% | Collapsed |
| PreAct G=60 h=32 | -7.925 | 0.04% | Collapsed |

**Finding**: **ALL runs show multimodal (association hub) collapse** — sigmoid < 1% across all configurations. This is universal. The multimodal fusion layer is being shut off regardless of modulation type, grouping size, or mod_hidden_size. This strongly suggests the multimodal hub itself is the problem, not the modulator configuration.

#### 10.5.3 Temperature

| Run | temp_mean (SS) | temp_max (SS) | Interpretation |
|---|---|---|---|
| Baseline | — | — | No temperature modulation |
| Mult G=20 h=32 | **8.055** | **10.00** | Extreme inflation, pinned |
| Mult G=40 h=32 | 3.879 | 7.60 | Moderate inflation |
| Mult G=60 h=32 | 4.088 | 7.66 | Moderate inflation |
| Mult G=80 h=32 | 4.099 | 7.56 | Moderate inflation |
| Mult G=100 h=32 | 3.496 | 7.74 | Moderate inflation |
| **Mult G=20 h=16** | **1.341** | **2.30** | **Healthy range** |
| **Mult G=60 h=16** | **2.022** | **2.97** | **Healthy range** |
| PreAct G=20 h=16 | 3.945 | 6.43 | Moderate inflation |
| PreAct G=60 h=16 | 1.954 | 3.34 | Mild inflation |
| PreAct G=20 h=32 | 5.852 | 9.45 | Heavy inflation |
| PreAct G=60 h=32 | 6.244 | **10.00** | Heavy inflation, pinned |

**Finding**: The h=16 Multiplicative runs show the healthiest temperature behavior (mean 1.3–2.0, not pinned at ceiling). All h=32 runs inflate temperature significantly. PreActivation h=32 runs are the worst offenders (mean 5.9–6.2, near ceiling).

#### 10.5.4 z_memory (GRU Gate Bias)

| Run | z_mem_mean (SS) | Interpretation |
|---|---|---|
| Mult G=20 h=32 | +6.75 | **GRU forced to forget** (high update gate) |
| Mult G=40 h=32 | +6.51 | Forced to forget |
| Mult G=60 h=32 | +5.96 | Forced to forget |
| Mult G=80 h=32 | +6.96 | Forced to forget |
| Mult G=100 h=32 | +6.98 | Forced to forget |
| **Mult G=20 h=16** | **+1.28** | **Mild forgetting bias — healthy** |
| **Mult G=60 h=16** | **-0.14** | **Near neutral — healthy** |
| PreAct G=20 h=16 | +3.44 | Moderate forgetting |
| PreAct G=60 h=16 | +0.34 | Near neutral |
| PreAct G=20 h=32 | +5.49 | Forced to forget |
| PreAct G=60 h=32 | +8.72 | **Extreme forgetting** |

**Critical observation**: In the previous diagnosis (§2.7, §5.2), z_memory was **negative** (-5.41 to -6.45), meaning the GRU was frozen (update gate → 0, retaining old state). In these new runs, z_memory is **positive** (+5 to +9 for h=32), meaning the opposite — the GRU update gate is pushed toward 1, causing **complete forgetting** at every timestep ($h_{new} \approx \tilde{h}$, discarding all previous state). Both extremes disable effective recurrent memory; only the h=16 runs maintain z_memory in a healthy range (-0.14 to +1.28).

**Note**: The sign reversal from the previous experiment (z_mem ≈ -6 then, ≈ +7 now) does not indicate improved behavior — it indicates the same pathology (memory disabled) via the opposite mechanism (always-forget vs always-retain). The h=32 modulator has enough capacity to find either extreme; the h=16 modulator is too constrained to do so.

#### 10.5.5 Beta (PreActivation Threshold Shift)

| Run | beta_uni_mean (SS) | beta_multi_mean (SS) | Interpretation |
|---|---|---|---|
| PreAct G=20 h=16 | -1.345 | -0.438 | Inhibitory (raising threshold) |
| PreAct G=60 h=16 | -0.972 | -0.244 | Mild inhibition |
| PreAct G=20 h=32 | -0.618 | -0.030 | Near neutral |
| PreAct G=60 h=32 | **+0.445** | -0.727 | **Disinhibited unimodal** / inhibited multimodal |

**Finding**: Beta (threshold shift) is mostly negative (inhibitory) — the modulator raises activation thresholds, filtering weak signals. Only `PreAct G=60 h=32` shows positive unimodal beta (+0.445), but this run still collapses the multimodal hub (gamma_multi = -7.93). The beta term is not being used for the intended disinhibition effect described in the PreActivation theory.

### 10.6 Analysis Summary

#### 10.6.1 Headline: mod_hidden_size=16 is the critical factor

The strongest predictor of healthy modulator behavior is **mod_hidden_size=16** (not grouping size or modulation type):

| Factor | Healthy Temperature | Healthy z_memory | Best Reward |
|---|---|---|---|
| mod_hidden=16 + Multiplicative | Yes (1.3–2.0) | Yes (-0.14 to +1.28) | **-108.42** (best overall) |
| mod_hidden=16 + PreActivation | Partial (2.0–3.9) | Partial (0.3–3.4) | -116 to -119 |
| mod_hidden=32 + Multiplicative | No (3.5–8.1) | No (+5 to +7) | -113 to -118 |
| mod_hidden=32 + PreActivation | No (5.9–6.2) | No (+5 to +9) | -113 (similar to baseline) |

The h=16 modulator is too constrained to find degenerate extremes. With only 16 hidden units, the GRU cannot memorize shortcuts as easily, forcing it to learn more useful modulation patterns. The h=32 modulator consistently finds pathological strategies regardless of other settings.

#### 10.6.2 Multimodal Hub Collapse is Universal

Every NMN run — all 11 configurations — collapsed the multimodal hub gate to sigmoid < 1%. This is not a modulator pathology but a signal that **the multimodal fusion layer is not producing useful features**. Possible explanations:
- The task is solvable with unimodal features alone (olfaction for food/predator, visual for threats)
- The multimodal hub's architecture (1152→128→128→128) is too compressed
- The hub receives already-modulated unimodal features, so its input quality depends on unimodal gate behavior

#### 10.6.3 Grouping Size Has Less Impact Than Expected

Comparing G=20 vs G=60 vs G=100 within Multiplicative h=32:
- Reward range: -113.29 to -118.46 (all within noise of baseline)
- All show similar pathological modulator values
- G=20 has the worst temperature inflation (mean 8.05)
- G=100 is closest to baseline behavior

Grouping size does not prevent degenerate convergence — it changes which local optimum the modulator finds, but all are equally pathological when mod_hidden=32.

#### 10.6.4 PreActivation Does Not Outperform Multiplicative

Despite the theoretical advantage of having both gamma and beta controls, PreActivation runs do not outperform Multiplicative ones. The best-performing run (`Mult G=60 h=16`) uses Multiplicative mode. PreActivation h=32 runs achieve baseline-equivalent reward but with heavily pathological modulator internals (temperature 5.9–6.2, z_memory 5.5–8.7). The extra degrees of freedom in PreActivation appear to enable more pathways to degenerate solutions rather than better adaptive modulation.

### 10.7 Updated Recommendations

Based on this ablation study, the experiment priority list from §8 is revised:

| Priority | Experiment | Rationale |
|---|---|---|
| **P0** | **Fix mod_hidden_size to 16** | h=16 prevents degenerate convergence in Multiplicative mode. The best-performing run uses h=16. |
| **P1** | Tighten temp_clip to [0.5, 3.0] | Even the best run (Mult G=60 h=16) reaches temp_mean=2.0, temp_max=3.0. Tighter bounds keep it constrained. |
| **P2** | Clamp z_memory to [-2, +2] | Prevents both always-retain (old pathology) and always-forget (new pathology). h=16 runs are already near this range naturally. |
| **P3** | Enable state-dependent noise for olfaction + visual | Still needed — the environment doesn't reward adaptive modulation. All runs collapse multimodal hub. |
| **P4** | Investigate multimodal hub architecture | Universal collapse suggests the hub itself is the bottleneck, not the modulator. Consider: wider hub, residual connections, or skip connections from unimodal to RNN. |
| **P5** | Separate modulator learning rate (5x lower) | Still relevant for h=32 if revisited, but h=16 + shared LR already works. |
| **P6** | Test Multiplicative G=60 h=16 as the new standard NMN config | This is the current best performer — use it as the reference for future ablations. |

---

## 11. Discussion: Dynamic Config Scheduling / Environment Curriculum (2026-03-09)

> Record of critical analysis of proposed approaches for addressing the NMN performance gap.

### 11.1 Proposal A: Algorithm Config Scheduling (Rejected)

**Idea**: A system that loads different algorithm/modulator configs at predefined training stages — e.g., tighter modulator bounds early, looser later; annealing learning rates on a schedule.

**Assessment: Not recommended.** Reasons:

1. **Treats symptoms, not causes.** The diagnosis identifies clear root causes (insufficient environmental pressure, excessive modulator capacity, multimodal hub collapse). None are *timing* problems — the optimization landscape has degenerate attractors regardless of when configs are applied.
2. **Combinatorial hyperparameter explosion.** N_params × N_stages × N_values. The 11-run ablation (§10) already shows mixed results from *static* configs. Scheduling multiplies the search space dramatically.
3. **The best run (Mult G=60 h=16) doesn't need scheduling.** Simply constraining modulator capacity (h=16) produces healthy temperature, z_memory, and best reward. This is a *structural* fix, not a temporal one.
4. **Mid-training config changes destabilize PPO.** PPO assumes a stationary MDP. Changing clipping bounds or architecture behavior mid-training invalidates the value function, causing performance crashes at each transition.

### 11.2 Proposal B: Non-Stationary Environment Curriculum (Deferred to Phase 2)

**Idea**: Periodically change environment parameters (predator count, bush count, predator olfactory properties, food availability) to create non-stationarity that forces adaptive behavior. Not algorithm config changes — purely environment changes.

**Assessment: Scientifically interesting, but premature.** Should be Phase 2 after the known issues are fixed.

#### 11.2.1 What's Good About This

- **The core intuition is correct**: the diagnosis repeatedly states the environment doesn't create pressure for adaptive modulation. Non-stationarity *does* create pressure for adaptation.
- **Biologically motivated**: animals evolved neuromodulation precisely because environments change — predator density fluctuates, food availability is seasonal, shelter comes and goes.
- **Could differentiate NMN from baseline**: a fixed-architecture agent that memorizes one strategy should fail when the environment shifts; a modulated agent that dynamically adjusts processing *should* have an advantage.

#### 11.2.2 Critical Concerns

**Concern 1 — Non-stationarity creates pressure for *general* adaptation, not *neuromodulation* specifically.**

The modulator's design is about precision-weighting sensory channels based on internal state (injury). Changing the number of predators or bushes doesn't create pressure to *modulate sensory precision based on injury*. It creates pressure to learn different policies for different environment configurations. These are fundamentally different problems.

If predator count doubles mid-training, the agent needs a different *policy* (more cautious movement), not different *sensory weighting* (upweight olfaction). The modulator isn't designed to solve "the world changed" — it's designed to solve "my body state changed and I need to re-weight my senses."

**Risk**: The NMN outperforms baseline under non-stationarity, but the real reason is extra parametric capacity absorbing distribution shift, not meaningful sensory modulation. The modulator internals must be checked to distinguish these explanations.

**Concern 2 — PPO is the wrong algorithm for non-stationary environments.**

PPO assumes a stationary MDP. Its value function estimates expected returns under the *current* environment. When the environment changes:
- Value function is immediately wrong → value loss spikes
- Wasted samples as the policy adapts to outdated value estimates
- Performance crashes at every transition

This hurts *both* agents equally, making the experimental signal noisy. Meta-RL algorithms (RL², MAML) are designed for this; PPO is not.

**Concern 3 — Confounding two variables simultaneously.**

The current diagnosis gives a clean experimental question: *does the modulator learn useful sensory modulation?* Adding non-stationarity introduces a second question: *can the agent handle changing environments?*

If NMN wins under non-stationarity, possible explanations include: (a) proper sensory gain control, (b) extra capacity helps with distribution shift, (c) baseline was more brittle. If NMN loses: (a) modulation doesn't help, (b) non-stationarity broke PPO for both, (c) modulator shortcuts got worse under instability. Clean attribution becomes impossible.

**Concern 4 — The unsolved problems from the diagnosis persist.**

Even in a non-stationary environment:
- Multimodal hub will still collapse (§10.6.2 — universal across ALL 11 configs)
- State-dependent noise is still off — modulator has no reason to modulate precision based on injury
- h=32 will still find degenerate shortcuts — capacity problem is independent of environment stationarity

Non-stationarity adds a new dimension without fixing the existing broken ones.

#### 11.2.3 Recommended Phasing

**Phase 1 — Fix the known problems first (static environment)**
- h=16, tight bounds, state-dependent noise (P0–P3 from §10.7)
- Verify the modulator actually learns meaningful sensory modulation (gamma responds to injury state, not collapsed to a constant)
- This answers: *does neuromodulation work at all under ideal conditions?*

**Phase 2 — Test robustness with non-stationarity (THEN)**
- Only after confirming the modulator learns meaningful modulation in a static environment
- Use non-stationary environment to test whether the modulated agent *generalizes* better than baseline
- Now the comparison is meaningful: both agents learned useful behavior, but the modulated one adapts better to change

**If environment curriculum is pursued in Phase 2**, the highest-value variant is **state-dependent noise curriculum** (not entity count changes): start with strong state-dependent noise on all channels (making sensory modulation essential), then gradually reduce it. This directly pressures the modulator to learn its intended function — precision weighting based on body state.

#### 11.2.4 Alternative: Modulator Freeze/Unfreeze

A simpler mechanism that addresses the timescale separation issue (§5.5) without a full config scheduling system: freeze the modulator for the first N timesteps (e.g., 1B) while the task network learns stable representations, then unfreeze. This is one boolean flag with one timestep threshold — not a full scheduling system — and directly prevents the modulator from racing ahead to find degenerate shortcuts before the task network has converged.

### 11.3 Conclusion

**Do not build a generic config scheduling system.** The NMN's core problem is that suppressing features is a *better strategy* than modulating them in the current environment. No amount of scheduling changes that fact. Fix the fundamentals first (h=16, tighter bounds, state-dependent noise, multimodal hub investigation), confirm the modulator works under ideal conditions, then stress-test with non-stationarity as a Phase 2 experiment.
