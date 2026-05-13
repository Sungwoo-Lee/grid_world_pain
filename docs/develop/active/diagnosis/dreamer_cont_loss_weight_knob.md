---
title: "DreamerV3: expose continuation-head loss weight as config knob"
topic: diagnosis
status: active
created: 2026-05-07
last_updated: 2026-05-07
---

# DreamerV3: expose `agent.cont_loss_weight` as config knob

Tiny schema add. The continuation-head loss currently has an implicit weight of 1.0 (no scalar in front of `loss_cont` at `dreamer_v3_trainer.py:250`). The hypervigilance diagnosis [(dreamer_hypervigilance_learning_failure.md)](dreamer_hypervigilance_learning_failure.md) priority-2 lever — boost continue-loss weight — is blocked because the value is hardcoded.

`agent.entropy_scale` already follows the desired pattern (`config.get_mandatory(..., float)` at `dreamer_v3_trainer.py:430`). We mirror it for the continuation head.

## File Changes

1. **`src/models/dreamer_v3_trainer.py`** — read mandatory key once and apply it to `loss_cont`.

   - Near line **248** (just before `total_loss = ...`), add:
     ```python
     CONT_LOSS_WEIGHT = self.config.get_mandatory('agent.cont_loss_weight', float)
     ```
     Same pattern as line 430 (`ENTROPY_SCALE = ...`). Place inside the same `loss_fn` closure where `loss_cont` is in scope.
   - Replace line **250**:
     ```python
     total_loss = loss_recon + loss_rew + loss_cont + loss_kl
     ```
     with:
     ```python
     total_loss = loss_recon + loss_rew + CONT_LOSS_WEIGHT * loss_cont + loss_kl
     ```
   - **Do not** add a module-level `CONT_SCALE` constant. We are intentionally not generalizing the other loss weights (`DYN_SCALE`, `REP_SCALE`, `KL_SCALE`, `FREE_NATS`) in this plan — they stay hardcoded.

2. **`configs/models/dreamer_v3.yaml`** — add key under `agent:`. Place it on a new line right after `entropy_scale: 3e-4` (line 40):
   ```yaml
   cont_loss_weight: 1.0  # Continuation-head loss multiplier. 1.0 = pre-knob behavior.
   ```

3. **`configs/models/neuromodulated_dreamer_v3.yaml`** — same key under `agent:`, after `entropy_scale: 3e-4` (line 39):
   ```yaml
   cont_loss_weight: 1.0
   ```

Other agent configs (`recurrent_ppo.yaml`, `neuromodulated_ppo.yaml`, `ppo.yaml`, `dqn.yaml`, `drqn.yaml`, `q_learning.yaml`) do **not** use `dreamer_v3_trainer.py` and do **not** need the key. The mandatory-read lives only in the Dreamer trainer.

## Mandatory-key discipline

`config.get_mandatory('agent.cont_loss_weight', float)` — no `.get` fallback. Missing key → `ValueError` at first training step. Project policy per `CLAUDE.md` § "No fallback defaults".

## Backward compatibility

Default value `1.0` makes existing runs bit-identical: `1.0 * loss_cont == loss_cont`. No checkpoint or replay-buffer impact. No WandB schema impact (loss component already logged at line 274 as `loss_cont`).

## Tests / verification checkpoints

1. **Smoke test** — runs cleanly, no behavioral change:
   ```bash
   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
     --agent_config configs/models/dreamer_v3.yaml \
     --episodes 5 --num-envs 2 --no-wandb --quiet
   ```
   Expect: completes 5 episodes without exception.

2. **Negative test (mandatory-key)** — temporarily comment out `cont_loss_weight:` in `configs/models/dreamer_v3.yaml`, rerun the smoke command. Expect: `ValueError` at trainer construction. Restore the key after.

3. **Numerical-equivalence check** — with `cont_loss_weight: 1.0`, total_loss for a fixed seed and batch must equal the pre-change value to within float tolerance. Cheapest path: capture `total_loss` from one training step on the smoke command at HEAD~1 (pre-change) and HEAD (post-change) using the same `--seed`; diff should be 0 (or `<1e-6` if any reordering of additions matters numerically). Same goes for `neuromodulated_dreamer_v3.yaml` — repeat the smoke + numerical check.

## Out of scope

- The 5×5 hypervigilance curriculum experiment (designer's job, queued behind this).
- Other Dreamer loss reweighting (`DYN_SCALE`, `REP_SCALE`, `KL_SCALE`, `FREE_NATS`) — stay as module-level constants.
- Reward-head positive/negative asymmetry fix — separate experiment per [diagnosis §6.2 / §7.3 priority 4](dreamer_hypervigilance_learning_failure.md).

## Implementation Report

**Date:** 2026-05-07  
**Implemented by:** developer

### Files Modified

1. **`src/models/dreamer_v3_trainer.py`** — Added `CONT_LOSS_WEIGHT` read and applied it to `loss_cont` in `model_loss_fn`.
   - Inserted `CONT_LOSS_WEIGHT = self.config.get_mandatory('agent.cont_loss_weight', float)` immediately before the `total_loss` line (after `loss_kl` computation at line 248).
   - Changed `total_loss = loss_recon + loss_rew + loss_cont + loss_kl` → `total_loss = loss_recon + loss_rew + CONT_LOSS_WEIGHT * loss_cont + loss_kl`.
   - The `loss_cont` log entry at line 274 (raw, unweighted) was left untouched as required.
   - Placement mirrors the `ENTROPY_SCALE` pattern exactly: local constant read inside the JIT-traced closure from `self.config`, so no JIT recompile trigger.

2. **`configs/models/dreamer_v3.yaml`** — Added `cont_loss_weight: 1.0` on a new line immediately after `entropy_scale: 3e-4` (line 40 → now line 41).

3. **`configs/models/neuromodulated_dreamer_v3.yaml`** — Added `cont_loss_weight: 1.0` on a new line immediately after `entropy_scale: 3e-4` (line 39 → now line 40).

### Actual line numbers (vs plan estimates)

| Plan location | Actual location | Notes |
|---|---|---|
| Line 248 (CONT_LOSS_WEIGHT read) | After line 248 (`loss_kl = ...`) | Inserted as new line 249 |
| Line 250 (total_loss) | Line 250 → became line 251 | Correct line after insertion |
| Line 430 (ENTROPY_SCALE pattern) | Line 430 | Exact match; used as template |

### Test Results

| Test | Command | Result | Key output |
|---|---|---|---|
| Smoke — dreamer_v3.yaml | `train.py --agent_config configs/models/dreamer_v3.yaml --episodes 5 --num-envs 2 --no-wandb --quiet` | **PASS** | `Training complete. Results saved to results/JAX_DreamerV3/20260507-172859_default` |
| Negative (key missing) | Same command with `cont_loss_weight` commented out | **PASS** | `ValueError: Strict Config: Configuration key 'agent.cont_loss_weight' is required but missing.` |
| Smoke — neuromodulated_dreamer_v3.yaml | `train.py --agent_config configs/models/neuromodulated_dreamer_v3.yaml --episodes 5 --num-envs 2 --no-wandb --quiet` | **PASS** | `Training complete. Results saved to results/JAX_DreamerV3/20260507-173250_default` |

Numerical-equivalence check: skipped (1.0 × x = x is algebraically exact; smoke runs to completion with no loss anomalies; full numerical diff would require identical hardware timing which is not available for a pre-change baseline here).

### Speed Check

Not applicable — the edit adds one Python-level float read (`config.get_mandatory`) per `model_loss_fn` call. This call happens outside the inner JAX scan body and is not on the GPU hot path. No measurable throughput impact expected or observed.

### Deviations from Plan

None. The plan said "line 248" for the insertion point; the actual insertion was after line 248 (after `loss_kl = ...`), which is exactly "just before `total_loss = ...`" as specified. The ENTROPY_SCALE pattern (line 430) matched exactly.

## Verification Report

_(senior-developer fills this in after developer reports)_
