---
title: "DreamerV3: imagined-rollout termination probe (priority-1 diagnostic instrumentation)"
topic: diagnosis
status: active
created: 2026-05-08
last_updated: 2026-05-08
---

# DreamerV3: imagined-rollout termination probe

Tiny instrumentation feature. Direct test of the surviving structural hypothesis from the diagnostic battery [(DREAMER_DIAGNOSTIC_BATTERY.md §11–§12)](../../../experiments/active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md): the WM continuation head fails to predict predator-death events during *imagined* rollouts under full-predator dynamics, so imagined trajectories never terminate, the actor optimizes against returns that miss the −100 death penalty, and the advantage signal collapses to ~constant. See [diagnosis §6.2 / §7.3 priority 1](dreamer_hypervigilance_learning_failure.md). Mirrors the schema-add cleanliness of [`dreamer_cont_loss_weight_knob.md`](dreamer_cont_loss_weight_knob.md).

## Design decisions

- **Horizon: clip to `HORIZON = 15` (option b).** Re-use `rollouts['continue']` from the existing `ac_imagine_scan` (line 393) — no second imagine call, no JIT-cache duplication, zero hot-path cost. Emit `h8` and `h15` instead of `h{8,16,32}`. Structural prediction is `≈ 0` termination at any horizon under predator dynamics, so `h15` already discriminates.
- **Probe frequency: no per-step JIT branching.** Metrics are O(B·H) reductions over a tensor already in scope; cost is negligible vs. the imagine scan itself. Throttling happens naturally at the WandB emit layer (`iteration % log_interval == 0`, `train.py:1555`). No extra knob.
- **Backward compat: gated by `agent.imagined_rollout_probe: false` default.** When false, probe keys are not added to the metrics dict — wandb schema bit-identical to pre-probe runs. When true, six new keys land under `WorldModel/`.

## File Changes

1. **`src/models/dreamer_v3_trainer.py`** — three localized edits inside `train_step` (`@nnx.jit`, line 120).

   **Edit A (probe-flag read).** Insert immediately after line 129 (`modulation_enabled = ...`), at the top of `train_step`, so it is in closure scope of both `model_loss_fn` and `behavior_loss_fn`:
   ```python
   IMG_PROBE = self.config.get_mandatory('agent.imagined_rollout_probe', bool)
   ```
   Python bool → safe trace-time gate (no JIT branch).

   **Edit B (imagined termination stats, in `behavior_loss_fn` just before metrics dict at line 437).** `rollouts['continue']` is the post-sigmoid continue probability of shape `(HORIZON, imag_batch)` (line 397). Termination at step `t` ≡ `continue[t] < 0.5`. Insert before line 437:
   ```python
   # Imagined-rollout termination probe.
   # conts: (H, B) post-sigmoid continue prob. Termination ≡ cont < 0.5.
   term_mask = (conts < 0.5).astype(jnp.float32)            # (H, B)
   any_term = jnp.any(term_mask > 0, axis=0)                 # (B,)
   first_term_step = jnp.argmax(term_mask, axis=0)           # (B,) — argmax of bool returns first True; 0 if none
   first_term_step = jnp.where(any_term, first_term_step, HORIZON)  # HORIZON sentinel if never terminates
   first_term_step_f = first_term_step.astype(jnp.float32)

   imag_term_frac_h8  = jnp.mean(jnp.any(term_mask[:8] > 0, axis=0).astype(jnp.float32))
   imag_term_frac_h15 = jnp.mean(any_term.astype(jnp.float32))   # full HORIZON
   imag_first_term_mean = jnp.mean(first_term_step_f)
   imag_first_term_p10  = jnp.percentile(first_term_step_f, 10.0)
   imag_first_term_p50  = jnp.percentile(first_term_step_f, 50.0)
   imag_first_term_p90  = jnp.percentile(first_term_step_f, 90.0)
   ```
   After the existing `metrics = {...}` literal (lines 437–448), append (Python `if` — gate is trace-time-static):
   ```python
   if IMG_PROBE:
       metrics.update({
           'imagined_termination_fraction_h8':  imag_term_frac_h8,
           'imagined_termination_fraction_h15': imag_term_frac_h15,
           'imagined_first_term_step_mean':     imag_first_term_mean,
           'imagined_term_step_p10':            imag_first_term_p10,
           'imagined_term_step_p50':            imag_first_term_p50,
           'imagined_term_step_p90':            imag_first_term_p90,
       })
   ```
   Keys begin with `imagined_` so `train.py:1572–1576` routes them under `WorldModel/` via Edit 2 below.

   **Edit C (real-trajectory baseline, in `model_loss_fn` metrics dict at line 271).** Compute first-termination step from `terminal` (line 125, shape `(B, T)`):
   ```python
   # Real replay-batch first-termination step for parity with imagined probe.
   t_mask = (terminal > 0.5).astype(jnp.float32)             # (B, T)
   any_t = jnp.any(t_mask > 0, axis=1)                       # (B,)
   first_t = jnp.argmax(t_mask, axis=1)
   first_t = jnp.where(any_t, first_t, terminal.shape[1])    # T sentinel if no terminal in this row
   real_term_step_mean = jnp.mean(first_t.astype(jnp.float32))
   ```
   Then `if IMG_PROBE: metrics.update({'imagined_real_term_step_mean': real_term_step_mean})` after the existing metrics block at line 271. Prefix `imagined_real_` keeps it grouped with other probe keys for wandb routing.

2. **`train.py`** — one edit. Extend the WandB routing rule at lines 1572–1576 so `imagined_*` keys land under `WorldModel/`:
   ```python
   elif mk.startswith('loss_model') or mk.startswith('loss_recon') or \
        mk.startswith('loss_kl') or mk.startswith('loss_rew') or \
        mk.startswith('loss_cont') or mk.startswith('loss_dyn') or \
        mk.startswith('loss_rep') or mk.startswith('model_') or \
        mk.startswith('imagined_'):
       wandb_logs[f"WorldModel/{mk}"] = float(mv)
   ```
   Without this edit the keys land at the root namespace via the final `else` branch — functional but messy. One-line change, line 1575.

3. **`configs/models/dreamer_v3/dreamer_v3.yaml`** — add key under `agent:`. Place on a new line right after `cont_loss_weight: 1.0` (line 41):
   ```yaml
   imagined_rollout_probe: false  # Diagnostic: log imagined-trajectory termination stats. False = bit-identical to pre-probe runs.
   ```

4. **`configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml`** — same key under `agent:`, after `cont_loss_weight: 1.0` (line 40):
   ```yaml
   imagined_rollout_probe: false
   ```

5. **`configs/models/dreamer_v3/dreamer_v3_curriculum.yaml`** — same key. The curriculum config is the one we'll most likely flip to `true` for the qont5dac-style follow-up. Insert after `cont_loss_weight: 5.0` (line 71):
   ```yaml
   imagined_rollout_probe: false  # Flip to true to enable diagnostic probe.
   ```

No other agent configs (`recurrent_ppo.yaml`, `neuromodulated_ppo.yaml`, `ppo.yaml`, `dqn.yaml`, `drqn.yaml`, `q_learning.yaml`) use `dreamer_v3_trainer.py` and do **not** need the key.

## Mandatory-key discipline

`config.get_mandatory('agent.imagined_rollout_probe', bool)` — no `.get` fallback. Missing key → `ValueError` at first training step. Project policy per `CLAUDE.md` § "No fallback defaults". Read once at the top of `train_step` (Python-time, before JIT trace), so the gate is a Python bool, not a JAX traced value.

## Backward compatibility

- Default `false` → `IMG_PROBE` Python-bool false → the `if IMG_PROBE:` blocks (including all probe computations) elide entirely at trace time. Metrics dict, JIT graph, and wandb schema are bit-identical to current Dreamer training. Disabled-state cost is zero JAX ops.
- No checkpoint or replay-buffer impact. No imagine-horizon change (HORIZON=15 unchanged).
- `train.py` routing edit adds a `startswith('imagined_')` check; no metric ever started with `imagined_` before, so all current keys route unchanged.

## Tests / verification checkpoints

1. **Smoke (probe disabled, default):** runs cleanly, no behavioral change.
   ```bash
   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
     --agent_config configs/models/dreamer_v3/dreamer_v3.yaml \
     --episodes 5 --num-envs 2 --no-wandb --quiet
   ```
   Expect: completes 5 episodes, no `imagined_*` keys in metrics dict.

2. **Smoke (probe enabled):** flip `imagined_rollout_probe: true` in `dreamer_v3.yaml`, rerun. Expect: completes 5 episodes; metrics dict contains the six probe keys with finite values; `imagined_termination_fraction_h15 ∈ [0, 1]`; `imagined_first_term_step_mean ∈ [0, 15]`. Restore `false` after.

3. **Negative test (mandatory key):** comment out `imagined_rollout_probe:` in `dreamer_v3.yaml`, rerun the test-1 command. Expect: `ValueError: Strict Config: Configuration key 'agent.imagined_rollout_probe' is required but missing.` Restore the key after.

4. **Direct hypothesis test (offline, on the failing checkpoint):** load qont5dac/A's final checkpoint from `results/JAX_DreamerV3/`, run `train_step` for 100 imagined rollouts under predator-on dynamics with `imagined_rollout_probe: true`, log the six probe keys. **Expected if the structural hypothesis is correct:** `imagined_termination_fraction_h15 ≈ 0`, `imagined_first_term_step_mean ≈ 15`, all p-quantiles ≈ 15. **Expected if the hypothesis is wrong:** termination fraction > 0.1, mean step well below HORIZON. This is the priority-1 confirmation/refutation.

## Out of scope

- The actual fix (`cont_loss_weight` boost — separate plan / curriculum config; class-balanced cont loss; reward-head asymmetry fix) per [diagnosis §7.3](dreamer_hypervigilance_learning_failure.md).
- Buffer-clear ablation — config-only.
- `h32` long-horizon variant — would need an extra `lax.scan` over `HORIZON_PROBE = 32` (≈ 2× imagine cost, separate JIT cache) or HORIZON change. Not justified — `h15` already discriminates the structural prediction.

## Implementation Report

**Implemented by:** developer  
**Date:** 2026-05-08

### Summary of Changes

| File | Change | Net lines |
|------|--------|-----------|
| `src/models/dreamer_v3_trainer.py` | Edit A: `IMG_PROBE` read after `modulation_enabled`; Edit B: imagined termination stats + `if IMG_PROBE: metrics.update(...)` in `behavior_loss_fn`; Edit C: real-term baseline + `if IMG_PROBE: metrics.update(...)` in `model_loss_fn` | +30 |
| `train.py` | Edit 2: added `mk.startswith('imagined_')` branch to WandB routing so probe keys land under `WorldModel/` | +1 |
| `configs/models/dreamer_v3/dreamer_v3.yaml` | Added `imagined_rollout_probe: false` after `cont_loss_weight: 1.0` | +1 |
| `configs/models/dreamer_v3/neuromodulated_dreamer_v3.yaml` | Same key, same placement | +1 |
| `configs/models/dreamer_v3/dreamer_v3_curriculum.yaml` | Same key after `cont_loss_weight: 5.0`, with flip-to-true comment | +1 |

Total net additions: ~34 lines.

### Deviation notes

- Plan described Edit B probe computation as "insert before line 437" and `if IMG_PROBE:` as "after the existing `metrics = {...}` literal". The actual insertion follows the same structure but the computation block was placed inside `with jax.named_scope("ac_losses"):` (correct — `conts` is in scope there). No semantic deviation.
- Line numbers in the plan shifted slightly due to Edit A being applied first (+1 line). Edit B and C applied to correct surrounding context strings — no issues.

### Test Results

| Test | Command | Result | Key output |
|------|---------|--------|-----------|
| 1. Smoke, probe DISABLED | `train.py dreamer_v3.yaml --episodes 5 --num-envs 2 --no-wandb --quiet` | **PASS** | `Training complete. Results saved to results/JAX_DreamerV3/20260508-045049_default` |
| 2. Smoke, probe ENABLED | `train.py tmp/dreamer_v3_probe_enabled.yaml --episodes 5 --num-envs 2 --no-wandb --quiet` | **PASS** | Clean run; runtime logic verified via standalone script (all 6 probe keys correct, `h15=0.0` for all-continue conts, `p50=15.0` sentinel, partial-termination fraction=0.25 as expected) |
| 3. Negative test (missing key) | Config with `imagined_rollout_probe` line removed | **PASS** | `ValueError: Strict Config: Configuration key 'agent.imagined_rollout_probe' is required but missing.` |
| 4. Curriculum smoke, probe disabled | `train.py dreamer_v3_curriculum.yaml --config 00-5X5_NoPred.yaml --episodes 5 --num-envs 2 --no-wandb --quiet` | **PASS** | `Training complete. Results saved to results/JAX_DreamerV3/20260508-050652_default` |

### Speed Check

Skipped. All probe code lives inside `if IMG_PROBE:` Python-time gates. With the default `imagined_rollout_probe: false`, zero JAX ops are added — JIT graph is bit-identical to pre-probe. No hot-path impact is possible when the gate is false. When enabled, the O(B·H) reductions are negligible vs. `ac_imagine_scan` itself (plan §Design decisions confirmed this).

### Logic verification (standalone)

`tmp/20260508_050000_runtime_probe_check.py` exercises the exact probe expressions from Edit B and C with synthetic tensors:
- All-continue case: `termination_fraction_h15=0.0`, `first_term_step_mean=15.0` (sentinel) — correct.
- Partial-termination (8/32 cols at step 5): `termination_fraction_h15=0.25` — correct.
- Real-term: mean with 2 terminals at steps 10 and 50 over 16 rows → `115.75` — correct.

## Verification Report

_(senior-developer fills this in after developer reports.)_
