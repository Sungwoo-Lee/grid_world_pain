# Fix PPO Beta Logging for FiLM Modes

> **Status**: COMPLETED
> **Opened**: 2026-03-18
> **Related**: [FILM_MODULATION_PLAN.md](FILM_MODULATION_PLAN.md) — FiLM implementation (completed)

---

## Context

The RecurrentPPO training loop in `train.py` does not log beta (β) metrics for FiLM/FiLMNoNorm modes. The condition at line 955 only checks `"PreActivation"`, so FiLM runs silently drop all beta metrics from WandB. The DreamerV3 trainer (`dreamer_v3_trainer.py:295`) already handles this correctly.

Note: gamma (γ) is intentionally logged as raw logits for all modes — this is preferred because raw values make gate collapse directly visible (large negative values for Multiplicative/PreActivation).

## Analysis

### What the neuromodulator outputs

`_get_signal` in `src/models/neuromodulator.py:145-155` always returns two values per phase:

```python
def _get_signal(head, baseline, head_add=None, baseline_add=None):
    raw = head(h_mod_new)
    sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
    sig = baseline.value + sig

    if head_add is not None:          # ← exists for PreActivation, FiLM, FiLMNoNorm
        raw_add = head_add(h_mod_new)
        sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
        sig_add = baseline_add.value + sig_add
        return sig, sig_add
    return sig, jnp.zeros_like(sig)   # ← Multiplicative: beta is zeros
```

The `ModulatorOutput` dataclass always has both `z_unimodal` and `z_unimodal_add` fields. For Multiplicative, `z_unimodal_add` is zeros (no additive head exists).

### How each mode uses γ and β in the encoder

From `src/models/recurrent_ppo_network.py:126-188`:

| Mode | Effective γ | Effective β | Equation |
|------|------------|------------|----------|
| **Multiplicative** | `sigmoid(z_unimodal)` ∈ (0,1) | *not used* | `relu(Wx) * sigmoid(z)` |
| **PreActivation** | `sigmoid(z_unimodal)` ∈ (0,1) | `z_unimodal_add` (raw) | `relu(sigmoid(z) * Wx + β)` |
| **FiLM** | `z_unimodal` (raw, unbounded ℝ) | `z_unimodal_add` (raw) | `relu(z * LN(Wx) + β)` |
| **FiLMNoNorm** | `z_unimodal` (raw, unbounded ℝ) | `z_unimodal_add` (raw) | `relu(z * Wx + β)` |

Key difference: Multiplicative and PreActivation pass γ through sigmoid; FiLM and FiLMNoNorm use it raw (unconstrained).

### What train.py currently logs (lines 942–961)

```python
# GAMMA — logged for ALL modes as raw logit (intentional, preferred for collapse diagnosis)
"modulator/gamma_uni_mean":  jnp.mean(mod_info.z_unimodal)
"modulator/gamma_uni_std":   jnp.std(mod_info.z_unimodal)
"modulator/gamma_multi_mean": jnp.mean(mod_info.z_multimodal)
"modulator/gamma_multi_std":  jnp.std(mod_info.z_multimodal)

# BETA — logged ONLY for PreActivation
if type == "PreActivation":                                        # ← BUG: misses FiLM/FiLMNoNorm
    "modulator/beta_uni_mean":   jnp.mean(mod_info.z_unimodal_add)
    "modulator/beta_uni_std":    jnp.std(mod_info.z_unimodal_add)
    "modulator/beta_multi_mean": jnp.mean(mod_info.z_multimodal_add)
    "modulator/beta_multi_std":  jnp.std(mod_info.z_multimodal_add)
```

### What dreamer_v3_trainer.py does correctly (lines 295–299)

```python
# Beta: logged for all modes that have additive heads
if wm.modulation_type == "PreActivation" or wm.modulation_type == "FiLM" or wm.modulation_type == "FiLMNoNorm":
    'mod_beta_unimodal_mean': jnp.mean(mod_outputs_T.z_unimodal_add),
    'mod_beta_multimodal_mean': jnp.mean(mod_outputs_T.z_multimodal_add),
```

## Implementation Plan

### Design

Extend the beta logging condition in `train.py` to include FiLM and FiLMNoNorm. One-line change.

No new metrics, no new config keys.

### File Changes

#### `train.py` (line 955)

```python
# BEFORE (line 955):
                            if modulation_config is not None and modulation_config.get('type') == "PreActivation":

# AFTER:
                            if modulation_config is not None and modulation_config.get('type') in ("PreActivation", "FiLM", "FiLMNoNorm"):
```

### Summary of Files Changed

| File | Change | New Lines (approx) |
|------|--------|-------------------|
| `train.py` | Extend beta logging condition to include FiLM/FiLMNoNorm | 0 net new (condition change only) |

## Checkpoints

- [x] Checkpoint 1 — Enable WandB for a short FiLM or FiLMNoNorm PPO run. Confirm `modulator/beta_uni_mean` and `modulator/beta_multi_mean` appear in WandB (they were previously missing). [14:40:00]
- [x] Checkpoint 2 — Run with `type: "Multiplicative"`. Confirm beta metrics do NOT appear (Multiplicative has no additive head, `z_unimodal_add` is zeros — logging zeros would be misleading). [14:41:00]

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-03-18 14:33:29

- Modified `train.py` line 955: Extended the condition for logging beta metrics to include `"FiLM"` and `"FiLMNoNorm"`.

## Verification Report

> **Verified by**: Gemini
> **Date**: 2026-03-18 14:42:00

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `train.py` | Extend beta logging | ✅ | Verified via debug runs for FiLM and Multiplicative |

**Conclusion**: Beta metrics are now correctly logged for FiLM and FiLMNoNorm modes in RecurrentPPO, matching DreamerV3's behavior. Multiplicative remains correctly excluded.
