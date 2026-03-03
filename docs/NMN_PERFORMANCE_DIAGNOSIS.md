# NMN Performance Diagnosis: Why the Modulated Agent Matches Baseline

## 1. Problem Statement

The Neuromodulated Recurrent PPO (NMN) agent shows **equivalent performance** to the unmodulated baseline — no measurable improvement in episode reward, survival, or learning speed. The theoretical expectation was that the modulator would learn injury-adaptive behavior (hypervigilance, cautious exploration, memory persistence) that outperforms a fixed-architecture agent, particularly in environments with state-dependent sensory noise and nociceptive threats.

This document diagnoses the likely causes, organized from **most impactful** to **least impactful**, and proposes concrete experiments to resolve each.

---

## 2. WandB Empirical Analysis

### 2.1 Runs Compared

| | NMN | Baseline |
|---|---|---|
| **Run** | `20260301-213626_rppoNMN_MC_relu_128hidden_GRU_hierarchical` | `20260302-143850_rppo_MC_relu_128hidden_GRU_hierarchical` |
| **WandB ID** | `uyhuypji` | `ddrpu91m` |
| **Status** | Running (44h) | Running (27h) |
| **Timesteps** | 6.94B | 5.25B |

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
| **Episode/Reward** | **-209.12 ± 1.70** | **-209.55 ± 1.89** | Effectively identical |
| Episode/Reward_Max | -186.61 ± 14.55 | -188.01 ± 13.40 | Identical |
| Episode/Reward_Min | -239.18 ± 2.35 | -239.62 ± 2.04 | Identical |
| **Episode/Steps** | **255.42 ± 14.48** | **252.84 ± 12.80** | Identical (both ~half max_steps) |
| Reward trajectory | -210.98 → -207.49 | -209.74 → -208.53 | Both improved ~2 points total |

**Conclusion**: Performance is statistically indistinguishable. Both agents plateau at the same reward (~-209) and survive ~255 of 500 steps. Neither shows a meaningful learning curve beyond early training. The NMN modulator provides zero measurable benefit.

### 2.4 Training Speed

| Run | s/it | SPS | Overhead |
|---|---|---|---|
| NMN | 0.373s | 44,496 | +20.7% slower |
| Baseline | 0.309s | 54,985 | — |

The modulator adds **~20% wall-clock overhead** for zero performance gain.

### 2.5 Loss Curves

| Metric | NMN (steady-state) | Baseline (steady-state) | Note |
|---|---|---|---|
| loss/total | 0.1041 ± 0.050 | 0.1031 ± 0.038 | Identical |
| loss/policy | -0.0008 ± 0.049 | -0.0011 ± 0.036 | Both near zero |
| loss/value | 0.2191 ± 0.019 | 0.2179 ± 0.019 | Identical |
| loss/entropy | -0.4647 ± 0.029 | -0.4776 ± 0.025 | Similar (NMN slightly higher — see §2.6) |
| loss/grad_norm | 0.3341 ± 0.133 | 0.3602 ± 0.129 | NMN slightly lower |

Losses are functionally identical. The value loss plateaus around 0.22 for both — the critic is equally imprecise.

### 2.6 Modulator Metrics — IS the Modulator Learning?

**Yes — the modulator is learning aggressively, but not helpfully.**

| Metric | Init | Steady-State | Final | Interpretation |
|---|---|---|---|---|
| `gamma_uni_mean` | 1.95 | 3.12 ± 0.33 | 3.77 | **Increasing** — unimodal gain amplified (sigmoid(3.77) ≈ 0.98). Modulator learned to keep unimodal gates ~open. |
| `gamma_uni_std` | 0.66 | 5.34 ± 0.34 | 5.46 | **Huge variance** — unimodal signals vary wildly across timesteps/envs |
| `gamma_body_mean` | 2.80 | -4.26 ± 0.15 | -4.09 | **Collapsed negative** — sigmoid(-4.09) ≈ 0.016. Body-state features are being **shut off** (~98% attenuation). |
| `gamma_assoc_mean` | 2.93 | -11.48 ± 0.32 | -12.08 | **Collapsed to near-zero** — sigmoid(-12) ≈ 0.000006. Association hub features are **completely killed**. |
| `z_memory_mean` | 0.94 | -5.41 ± 0.61 | -6.05 | **Strongly negative** — biasing GRU update gate toward 0 (maximal memory retention / refusing to update). |
| `z_memory_std` | 1.35 | 1.47 ± 0.29 | 1.99 | Moderate variance |
| `temperature_mean` | 0.68 | 2.13 ± 0.16 | 2.43 | **Doubled** — policy is 2.4x more stochastic than baseline. The agent is exploring excessively. |
| `temperature_max` | 1.10 | 8.53 ± 0.88 | **10.00** | **Hitting clip ceiling** — some environments/timesteps have max temperature |
| `temperature_min` | 0.57 | 0.54 ± 0.02 | 0.50 | Near floor for other timesteps |
| `modulator/grad_norm` | 0.017 | 0.236 ± 0.098 | 0.311 | Modulator IS receiving gradients |

### 2.7 Critical Findings from Modulator Metrics

**Finding 1 — Feature Collapse (body_state and association gates shut off)**

The modulator has learned to completely suppress body-state features ($\gamma \to -4$, sigmoid ≈ 1.6%) and association features ($\gamma \to -12$, sigmoid ≈ 0.0006%). This means:
- Interoceptive signals (injury, satiation, nutrition) are being **silenced**
- The association hub (which fuses cross-modal information) is **dead**

This is the **opposite** of the hypothesis — rather than learning to upregulate pain-related features, the modulator learned to suppress them entirely. The agent is effectively operating without its interoceptive senses and cross-modal integration.

**Why**: Suppressing noisy or unhelpful features can reduce variance in the value function, which PPO rewards. If interoceptive signals are noisy (constant noise on satiation/nutrition) and the reward signal is dominated by survival (avoiding death penalty), the optimal strategy may be to ignore interoception entirely. The modulator discovered this shortcut.

**Finding 2 — Memory Gate Freezing (z_memory → -6)**

A strongly negative `z_memory` biases the GRU update gate toward 0:
$$u_t = \sigma(W_u x + U_u h + z_{memory}) \approx \sigma(\cdot - 6) \approx 0$$

This means $h_{new} \approx (1 - 0) \cdot h_{prev} + 0 \cdot \tilde{h} = h_{prev}$. The GRU **stops updating its hidden state**. The agent's recurrent memory is frozen — it's operating as a near-feedforward network.

**Why**: If the modulator cannot improve task performance through adaptive modulation, freezing the hidden state is a stable local minimum. A frozen hidden state produces constant value estimates (low value loss variance), which PPO's clipped objective tolerates.

**Finding 3 — Temperature Inflation (mean = 2.43, some at ceiling 10.0)**

The modulator has pushed temperature to 2.4x the baseline's implicit 1.0. Higher temperature means softer action distributions, which:
- Increases entropy (confirmed: NMN entropy loss = -0.464 vs baseline = -0.478, NMN is slightly more entropic)
- Counteracts the entropy coefficient's natural decay
- May be compensating for the frozen GRU (if the hidden state doesn't update, the agent needs random exploration to cover the state space)

**Finding 4 — The modulator IS receiving gradient signal**

`modulator/grad_norm` grew from 0.017 to 0.311 over training. The modulator is not starved of gradients — it learned these extreme values deliberately through optimization.

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

## 3. Insufficient Environmental Pressure

### 3.1 State-Dependent Noise Is Nearly Disabled

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

### 3.2 Recommended Fix: Enable the Designed Noise Profile

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

## 4. Root Cause Analysis (WandB-Informed)

The modulator metrics (§2.6–2.7) reveal that the modulator IS learning — but it found a **degenerate shortcut** rather than useful adaptive behavior. Three pathological patterns emerged:

### 4.1 Feature Collapse — The Modulator Kills Interoceptive Signals

`gamma_body_mean` → -4.09 (sigmoid ≈ 1.6%) and `gamma_assoc_mean` → -12.08 (sigmoid ≈ 0%) mean the modulator learned to **suppress** body-state and cross-modal features entirely. This is rational from PPO's perspective: if interoceptive signals are noisy and don't reliably predict reward, silencing them reduces value function variance.

**Root cause**: The environment's perceptual noise is `constant` (not `state_dependent`) for satiation/nutrition, and only `injury` uses state-dependent noise. The modulator cannot learn "amplify these signals when healthy, suppress when injured" because the signals don't change with injury. Instead, it learns the simpler rule: "always suppress."

**Fix**: Enable state-dependent noise on olfaction and visual (§3 below) to create a regime where adaptive modulation outperforms static suppression.

### 4.2 Memory Gate Freezing — The GRU Stops Updating

`z_memory_mean` → -6.05 biases the GRU update gate to $\sigma(\cdot - 6) \approx 0$, effectively freezing $h_{prev}$. The agent loses its recurrent memory.

**Root cause**: With body-state and association features already suppressed (§4.1), the GRU hidden state carries little useful information. Freezing it is a stable local minimum — constant hidden state → constant value predictions → low value loss variance → PPO is satisfied.

**Fix**: This is downstream of §4.1. If the feature collapse is fixed, the GRU has useful information to maintain. Additionally, consider **clamping `z_memory` to [-2, +2]** to prevent the modulator from fully disabling the recurrent dynamics.

### 4.3 Temperature Inflation — Compensatory Exploration

`temperature_mean` → 2.43 (some timesteps at the clip ceiling of 10.0). With the GRU frozen and features suppressed, the agent compensates by exploring randomly. This is a second-order pathology caused by §4.1 and §4.2.

**Root cause**: The temperature bounds `[0.1, 10.0]` are far too wide. A temperature of 10 makes the policy nearly uniform-random. The current bounds allow the modulator to effectively disable the learned policy.

**Fix**: Tighten temperature bounds to `[0.5, 2.0]` or even `[0.8, 1.5]`. This constrains the modulator to *fine-tuning* exploration rather than overriding the policy entirely.

### 4.4 Coarse Spatial Grouping Enables Wholesale Suppression

With `grouping_size: 40` and `hidden_size: 128`, each head output controls 32–40 neurons simultaneously. This makes it easy for the modulator to shut off entire feature groups with a single negative value — which is exactly what happened with `gamma_body` and `gamma_assoc`.

**Fix**: Use `grouping_size: 1` (per-neuron) so the modulator must make fine-grained decisions. Wholesale suppression becomes harder when each neuron requires an independent decision.

### 4.5 Shared Optimizer — No Timescale Separation

Both the task network and the modulator are trained with a single Adam optimizer at `lr=0.0005`. The NEUROMODULATION_ALGORITHM.md §4 recommends a lower learning rate for the modulator.

With equal learning rates, the modulator learns as fast as the task network, which allows it to quickly find degenerate shortcuts (§4.1–4.3) before the task network has converged enough to provide stable gradient signal.

**Fix**: Use `optax.multi_transform` to assign a 5–10x lower learning rate to modulator parameters. This forces the modulator to adapt slowly to the task network's stable features rather than racing ahead to suppress them.

---

## 5. Training Loop Issues

### 5.1 PPO Loss Re-evaluation Discards Modulation State Continuity

In `recurrent_ppo_trainer.py:ppo_loss_fn`, the scan calls `model(obs, h)` at each timestep. This re-runs the modulator from `h_init` (the stored initial hidden state). During the K_epochs=4 re-evaluations, the modulator re-processes the same trajectory each time.

However, the **rollout was collected with a different modulator trajectory** (the one that actually evolved during collection). During PPO re-evaluation, the modulator hidden state trajectory may differ slightly from the collection trajectory because the task network weights have been updated between epochs. This creates a subtle distribution mismatch between the modulator signals used during collection (which determined actions) and those computed during loss evaluation.

For the baseline (no modulator), this issue doesn't exist — the only state is the task GRU, which is re-processed identically.

**Impact**: Likely minor, but worth noting. Standard in recurrent PPO.

### 5.2 Modulator Gradient Norm Not Properly Extracted

In `update_step()`, the modulator gradient norm extraction uses:
```python
if 'modulator' in grads:
    mod_grad_norm = optax.global_norm(grads['modulator'])
```

But NNX gradients are structured as nested module states, not string-keyed dicts. This check likely never fires, meaning `mod_grad_norm` is always 0.0 in WandB logs. Without modulator gradient diagnostics, we can't tell if the modulator is receiving meaningful learning signal.

**Recommendation**: Fix gradient extraction to use NNX-compatible tree traversal, or log the modulator's parameter norms over time.

---

## 6. Theoretical Considerations

### 6.1 The Environment May Not Need Modulation

The 10x10 grid with 4 food, 8 danger, 3 predators, and 500-step episodes may be **simple enough** that a fixed-architecture GRU can learn all necessary behaviors without adaptive modulation. The baseline can memorize:
- Danger locations via the visual sensor
- Predator avoidance via olfaction
- Food seeking via olfaction

If the baseline already achieves near-optimal behavior, the modulator has no room to improve. We should check WandB: if the baseline plateaus at a high reward with low variance, the environment is too easy.

**Recommendation**: If this is the case, increase environment difficulty:
- Larger grid (20x20 or 30x30)
- More danger zones
- State-dependent noise (see §3)
- Shorter episodes (forces faster adaptation)

### 6.2 The Modulator May Be Learning — But Not Helping

It's possible the modulator IS learning to produce non-trivial signals (gamma deviating from 0.88, memory bias deviating from 0, temperature varying) but these signals are orthogonal to task performance. The modulator may be encoding interoceptive state without this encoding translating into better decisions.

**Diagnostic**: Check WandB metrics:
- If `modulator/gamma_uni_mean` stays ~0.88 and `modulator/z_memory_mean` stays ~0.0, the modulator learned nothing.
- If these values change significantly over training, the modulator learned *something* — but it didn't translate to reward.

---

## 7. Experiment Priority List

| Priority | Experiment | What It Tests | Effort |
|---|---|---|---|
| **P0** | Tighten temperature bounds to [0.5, 2.0] | Prevent policy override (§4.3) | Low |
| **P1** | Clamp z_memory to [-2, +2] | Prevent GRU freeze (§4.2) | Low |
| **P2** | Enable state-dependent noise for olfaction + visual (§3) | Create reason for adaptive modulation | Low |
| **P3** | Set `grouping_size: 1` for max expressiveness | Prevent wholesale feature suppression (§4.4) | Low |
| **P4** | Fix activation mismatch (both relu) | Eliminate last config confound (§3.1) | Low |
| **P5** | Lower modulator learning rate 5–10x | Enforce timescale separation (§4.5) | Medium |
| **P6** | Increase `percept_bias_init` to 3.0+ | Truer pass-through at init | Low |
| **P7** | Increase environment difficulty (larger grid, more threats) | Force the baseline to fail so modulation has room to help (§6.1) | Medium |

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

## 8. Summary

The WandB analysis reveals the modulator **learned aggressively but pathologically**:

1. **Feature collapse**: Body-state gates → sigmoid(-4) ≈ 1.6%, association gates → sigmoid(-12) ≈ 0%. The modulator discovered that suppressing interoceptive signals reduces variance, which PPO rewards.
2. **Memory freeze**: z_memory → -6, forcing GRU update gate to ~0. The recurrent memory is effectively disabled.
3. **Temperature inflation**: Mean temperature 2.4x, some timesteps at ceiling 10.0. The agent compensates for frozen memory by exploring randomly.
4. **20% speed overhead** for zero performance benefit.

The core problem is that **the environment doesn't create sufficient pressure for adaptive modulation**. With mostly constant noise and a 10x10 grid, static feature suppression is as good as (or better than) dynamic modulation. The modulator found the easiest gradient-reducing shortcut instead of learning the intended hypervigilance behavior.

**Fix order**: Constrain the modulator's action space (tighter bounds on temperature and memory) → create environmental pressure (state-dependent noise) → refine architecture (grouping, learning rate) → increase task difficulty if needed.
