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
