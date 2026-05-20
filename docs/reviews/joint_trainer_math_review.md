---
title: JointTrainer refactor — math review (construction-preserves-equivalence)
topic: dreamer_srl_v2
status: active
created: 2026-05-20
last_updated: 2026-05-20
reviewer: math-reviewer
---

# JointTrainer refactor — math review

**Verdict**: `construction-preserves-equivalence`. The refactor is purely a Python-side reorganisation of attribute ownership. No equation, stop-gradient boundary, vmap axis, PRNG split, REINFORCE pairing, or optimizer-update ordering is touched. The L2 math-equivalence test at `atol=1e-5` is a sufficient gate, with L1 bit-identity providing the regression guard against accidental drift. One minor ambiguity flagged (Q3 / EMA at-step-0 semantics) — does NOT block implementation.

## Findings per audit question

| # | Audit question | Plan section | Verdict | Rationale |
|---|---|---|---|---|
| 1 | jax.grad boundaries | "Why this is safe" + BEFORE/AFTER | ✅ unchanged-by-construction | `joint.world_model is world_model` (NNX reference semantics, plan R1/R6). `nnx.value_and_grad(wm_loss_fn)(world_model)` inside `make_train_step` is unchanged; the closure receives the SAME object whether passed as `world_model` or `joint.world_model`. No new intermediate object is introduced; `nnx.merge` inside scan is the same merge that already exists today. |
| 2 | Optimizer update order | BEFORE/AFTER block lines 306-315 (Polyak) → 318-322 (train_step) | ✅ unchanged-by-construction | AFTER block preserves the exact `Polyak → train_step` sequence inside `_scan_body`. Inside `train_step`, the WM/actor/critic order at `train.py:793-982` is untouched (plan §Non-goals "NO change to make_train_step"). |
| 3 | Polyak EMA semantics | BEFORE/AFTER lines 305-315, plan §S7 quote | ⚠️ verify-empirically-at-L2 | The EMA formula `new = (1-tau)*target + tau*online` is preserved verbatim. AFTER block writes `nnx.update(trainer.target_critic, new_target_params)` then immediately reads `nnx.state(trainer)` — this relies on NNX reference semantics propagating the update through the composite trainer. The L2 test (`test_lax_scan_train.py` at atol=1e-5) is the right gate for this. The plan R6 already documents the reference-semantics expectation. The `tau=1.0` hard-copy at `step_idx==0` is preserved (line 308). The `do_update = (step_idx % target_update_freq) == 0` predicate is preserved (line 307). |
| 4 | symlog / twohot transforms | §Non-goals "NO math change" + §"Files that do NOT change: train.py, loss.py" | ✅ unchanged-by-construction | `src/algorithms/dreamer_srl/loss.py` (TwoHotEncoding with `linspace(-20, +20, 255)` in symlog space) is in the "NOT change" list. `symlog`/`symexp` in `utils.py` is not in the refactor scope. The bins-in-symlog-space contract is preserved by exclusion. |
| 5 | λ-return computation | §"Files that do NOT change: train.py" | ✅ unchanged-by-construction | `compute_imagined_returns` and `compute_lambda_values` live in `train.py` / `utils.py` which are explicitly out-of-scope. The call-site enumeration table (19 rows in `dreamer_srl_main.py` only) does not touch any line inside `train.py`. The §S5 true-continue splice is untouched. |
| 6 | REINFORCE log-prob × advantage pairing | §Non-goals "NO change to make_train_step signature in train.py:613-660" | ✅ unchanged-by-construction | `actor_loss_fn` at `train.py:891-947` computes `log_prob(sg(imagined_actions[h]))` via `sg(action) · log_softmax(logits_h)` — this pairing IS the v2-CP9 fix for the resampling bug. Both `sg_imagined_actions` and `actor_module` are now reached via `trainer.actor.forward_logits(sg_latents[h])` instead of `actor.forward_logits(...)` — but the SAME object, SAME stop_gradient'd action tensor, SAME log_prob formula. R1 (stale-reference) mitigation explicitly forbids re-binding short names that could break this guarantee. |
| 7 | jax.lax.stop_gradient boundaries | BEFORE/AFTER preserves train_step internals | ✅ unchanged-by-construction | All stop_gradient calls live in `train.py` (`sg_latents`, `sg_imagined_actions`, `sg(advantage)`, `sg(lambda_values)`, `sg(predicted_values)`, `sg(moments_offset)`, `sg(moments_invscale)`, `sg(target_critic_values)`) — not touched by the refactor. |
| 8 | vmap axis correctness | §Non-goals + train.py untouched | ✅ unchanged-by-construction | `jax.vmap(world_model.reward_model)`, `jax.vmap(critic)`, `jax.vmap(target_critic)`, `jax.vmap(world_model.continue_model)` all live in `train.py:831-963`, out of scope. No vmap axis is touched. |
| 9 | PRNG semantics | BEFORE/AFTER line 318 (`carry_key, k_train = jax.random.split(carry_key)`) | ✅ unchanged-by-construction | The scan-carry's `key` is split inside the scan body exactly as today (line 1041 of current `dreamer_srl_main.py` vs. line 318 of the AFTER block). The `jax.random.split` call site is character-identical. Iteration-N's key equals what the legacy for-loop's iteration-N would produce, by construction. The L2 test already validates that the scan-vs-for-loop PRNG sequencing produces math-equivalent results. The refactor does not perturb this. |
| 10 | per_rank_gradient_steps semantics | §C4 win condition + §Non-goals "NO buffer change" | ✅ unchanged-by-construction | `n_grad_steps = ratio(ratio_steps)` at line 867 of the driver is not in the rewire table. `cumulative_grad_steps += n_grad_steps` at line 1085 (scan path) is preserved by C3's row-19 collapse. The smoke config's `per_rank_gradient_steps: 1` means `n_grad_steps = 1` per iteration — the scan has length 1 in that smoke, which still exercises the same single-step math the for-loop produces. C4's perf win-condition is about the multi-iteration regime in the bench config, not the smoke. |
| 11 | Bit-identity under the new container | §"Why this is safe" + R1 + R6 + C2.a/b checkpoints | ✅ unchanged-by-construction | The core reasoning is correct: `JointTrainer.__init__` only binds attributes, it does not call any module's `__init__` or copy any parameter array. `joint.world_model IS world_model` holds (verified at C2.a checkpoint: `assert joint.world_model is world_model`). `nnx.split(joint)` walks the composite and returns the SAME parameter arrays the seven separate splits would return — just gathered under one graphdef. `nnx.merge(graphdef_joint, state)` inside the scan body produces a `trainer` whose `.world_model` substructure has bit-identical params to what the legacy seven-merge path would have produced. Plan R2 (optimizer-state pytree shape) is the one place where structural drift COULD occur; the plan's mitigation (debug assertion at C1) is appropriate. |

## Critical findings

None. The plan claims construction-preserves-equivalence and the audit confirms it. The L1 (bit-identity, 11 tests) and L2 (math equivalence at atol=1e-5, 5 tests) gates are sufficient to catch any latent drift, with C2.a/b runtime assertions catching the reference-semantics class of bug (R1/R6).

## Minor recommendation (non-blocking)

For audit question 3, the plan's AFTER block at line 308 reads `tau_val = jnp.where(step_idx == 0, jnp.float32(1.0), jnp.float32(critic_tau))`. This is identical to the current scan path (lines 1006-1008 of `dreamer_srl_main.py`). It already differs from the legacy for-loop's `tau = 1.0 if cumulative_grad_steps == 0 else critic_tau` (line 922) in ONE subtle way: the for-loop's `cumulative_grad_steps` is the GLOBAL counter across all iterations, while the scan's `step_idx` carry is also seeded from `cumulative_grad_steps` at line 973 (`jnp.array(cumulative_grad_steps, dtype=jnp.int32)`). So the first-call tau=1.0 fires ONLY on the very first grad step of the very first training iteration — same as the for-loop. The refactor preserves this exactly. No issue, just calling it out so the L2 test's `step_idx==0` branch coverage stays explicit when the developer ports the assertion.

## Cross-links

- Plan: [`joint_trainer_refactor_plan.md`](../develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md)
- Parity-pass memory: [`20260518_1511_dreamer_srl_v2_parity_pass_outperform`](../memory/memories/dreamer_diagnosis/20260518_1511_dreamer_srl_v2_parity_pass_outperform.md)
- REINFORCE-bug memory: [`20260518_1512_reinforce_resampling_bug_imag_action_threading`](../memory/memories/dreamer_diagnosis/20260518_1512_reinforce_resampling_bug_imag_action_threading.md)
- CPU-buffer-regression memory: [`20260519_1507_dreamer_srl_v2_cpu_buffer_regression`](../memory/memories/dreamer_diagnosis/20260519_1507_dreamer_srl_v2_cpu_buffer_regression.md)
- nnx split/merge pattern memory: [`20260519_1509_nnx_lax_scan_split_merge_pattern`](../memory/memories/dreamer_diagnosis/20260519_1509_nnx_lax_scan_split_merge_pattern.md)
- Code-reviewer doc (companion): [`joint_trainer_plan_review.md`](joint_trainer_plan_review.md)
- Working source: `src/algorithms/dreamer_srl/train.py` (loss formulae), `src/algorithms/dreamer_srl/dreamer_srl_main.py:860-1090` (scan-body to rewire), `src/models/dreamer_v3_trainer.py:685-833` (template).

Reviewer: `math-reviewer` (delegated; review transcribed by top-level Claude into this file because the agent returned findings inline).
