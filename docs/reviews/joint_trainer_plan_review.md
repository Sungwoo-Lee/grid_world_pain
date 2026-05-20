---
title: "Code review — JointTrainer refactor plan (dreamer-srl v2)"
topic: dreamer
status: complete
created: 2026-05-20
last_updated: 2026-05-20
reviewer: code-reviewer
plan_under_review: docs/develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md
---

# Code review — JointTrainer refactor plan (dreamer-srl v2)

## Purpose

This document is a pre-implementation correctness audit of the senior-developer's plan to collapse dreamer-srl v2's seven separately-tracked trainable artefacts (world model, actor, critic, target critic, plus three Adam optimizers) into a single composite Flax-NNX module called `JointTrainer`. The goal of the refactor is to unlock a previously-blocked perf win: the current scan path, despite passing math-equivalence tests at `atol=1e-5`, runs roughly seventy times slower than the equivalent Python for-loop and leaks GPU memory between iterations — both symptoms of XLA bookkeeping cost that scales with the number of independent NNX module instances threaded through `jax.lax.scan`. The plan asks the developer to mirror the structural shape that the original JAX-Dreamer trainer (an unrelated codebase in the same repo) has been using for months without trouble: one composite owner, one `nnx.split` outside the scan, one `nnx.merge` inside, one `nnx.update` after. This review checks fifteen specific JAX / NNX / Flax structural-correctness questions against the plan text and against the actual current source. It does NOT cover the math (a parallel math review under `docs/reviews/joint_trainer_plan_math_review.md` owns that question).

## Verdict

**Accept-with-changes.** The plan is structurally sound: the worked AFTER block on the scan body is a faithful one-for-one re-rendering of the proven `dreamer_v3_trainer.py:685-833` template; the C1–C6 commit topology has correct gating tests at each step; the rollback flag is preserved; R1 (stale references) and R3 (no inner split) are explicitly named with concrete mitigations. Three issues need to be resolved before the developer starts coding — none are show-stoppers, all are small clarifications. (1) The plan's AFTER block does not split the per-iteration training PRNG key from the carry PRNG key — the line `carry_key, k_train = jax.random.split(carry_key)` is shown inside the body, which is correct, but a contributor reading the BEFORE block could easily mis-port the existing pattern. (2) The R2 debug-assertion in C1 uses pseudocode (`joint_state['wm_opt']`) that does not match how `nnx.State` is actually keyed (attribute access, not dict subscript) — the developer needs the corrected form before C1 lands. (3) Checkpoint C3.a's grep ("`nnx.merge` count = 1 inside scan body") is one merge short — the AFTER block uses `nnx.update(trainer.target_critic, …)` to write Polyak params back, which is correct, but neighbouring code that needs `nnx.state(trainer.critic, nnx.Param)` and `nnx.state(trainer.target_critic, nnx.Param)` should be flagged as expected so the developer does not over-tighten the grep. None of these block hand-off; all three are addressable with sub-paragraph edits to the plan doc.

## Findings per audit question

Each row cites the plan section being audited and gives a one-line rationale. `📎 matches reference` = the plan text matches the canonical NNX pattern at `src/models/dreamer_v3_trainer.py:685-833`.

| # | Question | Plan reference | Verdict | Rationale |
|---|---|---|:---:|---|
| 1 | `nnx.split(joint)` called ONCE outside scan, graphdef in closure | Plan §AFTER, line 291: `graphdef_joint, state_joint = nnx.split(joint)` placed before `init_carry` and outside `_scan_body` | 📎 matches reference | Mirrors `dreamer_v3_trainer.py:797` (`graphdef, _ = nnx.split(self)`). One split, captured in enclosing closure that the scan body reads as a free variable. Conforms to insight `20260519_1509` rule 1+3. |
| 2 | `nnx.merge(graphdef, current_state)` called inside scan body | Plan §AFTER, line 304: `trainer = nnx.merge(graphdef_joint, state_joint)` as the first line of `_scan_body` | 📎 matches reference | Mirrors `dreamer_v3_trainer.py:705` (`trainer = nnx.merge(graphdef, current_state)`). One merge, reconstructs the full composite. |
| 3 | `nnx.state(joint)` called ONCE at bottom of scan body | Plan §AFTER, line 329: `new_state = nnx.state(trainer)` before assembling `new_carry` | 📎 matches reference | Mirrors `dreamer_v3_trainer.py:784` (`new_state = nnx.state(trainer)`). One extract; the recursive walk over the composite handles every sub-module. |
| 4 | `nnx.update(self, final_state)` called ONCE after scan returns | Plan §AFTER, line 338: `nnx.update(joint, state_joint_f)` after the scan returns | 📎 matches reference | Mirrors `dreamer_v3_trainer.py:831`. One update, propagates via NNX reference semantics to every child. |
| 5 | No stale module references across the joint boundary | Plan §R1 names the pattern explicitly and prescribes "use only `trainer.world_model`, `trainer.actor`, etc. inside the scan body. Never bind a short name." Plan §AFTER applies this discipline throughout (e.g. line 320: `trainer.world_model, trainer.actor, …` passed positionally to `train_step`) | ✅ correct | Mitigation is consistently applied throughout the worked AFTER block. The temptation pattern (`world_model = trainer.world_model` rebind, then use bare name after `nnx.update`) is correctly flagged as the high-risk path. |
| 6 | No `nnx.split` inside scan body | Plan §R3: "the developer must NOT write `nnx.split(trainer)` inside `_scan_body`" — and Plan §AFTER contains zero `nnx.split` calls inside `_scan_body` (the only split is the one outside, line 291) | ✅ correct | Conforms to insight `20260519_1509` anti-pattern. C3.a grep enforcement (`grep -c 'nnx.merge'` inside scan body == 1) does not directly check this — a complementary `grep -c 'nnx.split'` inside scan body == 0 would strengthen the gate; this is a soft suggestion, not a blocker. |
| 7 | No nested merges | Plan §R4 names this explicitly: "Do not `nnx.merge(graphdef_joint, state_joint)` to get `trainer`, then `nnx.merge(trainer.wm_opt_graphdef, …)`". Plan §AFTER respects this — there is exactly one merge per body iteration | ✅ correct | The composite handles sub-module reconstruction transparently. |
| 8 | Optimizer-state pytree layout preserved | Plan §"Optimizer-state verification (please read before C1)" verifies that all three optimizers are `nnx.Optimizer` wrappers (confirmed at `dreamer_srl_main.py:321-323`). Plan §R2 prescribes a one-line debug assertion at C1 comparing `nnx.split(joint).state['wm_opt']` to standalone `nnx.split(wm_opt).state` | ⚠️ concern | **The R2 assertion uses `joint_state['wm_opt']` (dict-subscript) pseudocode**, but `nnx.State` is accessed via attribute (`joint_state.wm_opt`) — and even that is not guaranteed: the actual access pattern depends on which NNX version is pinned. The plan does say "Adjust the key access — `joint_state['wm_opt']` is the conceptual idea; the actual access depends on how `nnx.split` keys composite children. The developer should empirically confirm the access pattern in C1" — but this leaves the developer guessing. Suggested fix: in the plan revision, add a one-sentence note that the developer should `print(jax.tree_util.tree_paths(joint_state))` once and paste the result into the Implementation Report so the C1 review can validate the assertion form before C2 starts. Note: the test fixtures at `test_lax_scan_train.py:127-129` keep the three optimizers as standalone `nnx.Optimizer` instances. The L2 test thus does NOT exercise the JointTrainer-wrapped optimizer state directly — it exercises the seven-everything carry shape. If the JointTrainer carry produces a subtly different pytree key path, L2 cannot catch it. This is a real gap; mitigation is the R2 assertion (with correct syntax). |
| 9 | PRNG threading inside the scan | Plan §AFTER lines 318: `carry_key, k_train = jax.random.split(carry_key)` (same line as the BEFORE block at `dreamer_srl_main.py:1041`) | 📎 matches reference | The carry's `key` slot is updated each iteration to the post-split residual `carry_key`; `k_train` is consumed by the train_step and not re-used. Reproducibility property is preserved — same initial key + same scan_xs = same trajectory. No accidental fresh PRNG inside the body. |
| 10 | Static argnames + closure capture for `lax.scan` | Plan §AFTER captures `graphdef_joint`, `target_update_freq`, `critic_tau` as free variables in the enclosing function scope; `n_grad_steps` is implicit in `scan_xs`'s leading axis (per `dreamer_srl_main.py:979` — `scan_xs = local_data_gpu`; `lax.scan` infers length from axis 0) | ✅ correct | The closure pattern is identical to the working reference. `target_update_freq` (int) and `critic_tau` (float) are hashable Python constants and trace as constants. The scan length is inferred from `scan_xs`, matching the existing BEFORE block. |
| 11 | vmap / axis correctness | Plan §"Files that do NOT change" explicitly lists `train.py`, `agent.py`, `loss.py`, `eval.py`, `buffers.py`. The call-site enumeration touches only attribute-binding rewires — no axis-related code | ✅ correct | No vmap axes change anywhere in the plan's File Changes section. The refactor is structural, not vectorization-shaped. |
| 12 | `@struct.dataclass` mutation discipline | Plan §"The JointTrainer class" declares `JointTrainer(nnx.Module)`, not `@struct.dataclass`. NNX module attribute mutation via `self.attr = …` is the canonical NNX-side write path; in-place pytree-field mutation is not at issue here | ✅ correct | The `moments` accumulator stays in the scan carry (Plan §"Carry-and-scan shape post-refactor", line 146: "The `moments`, `key`, and `step_idx` items stay in the carry because they are NOT `nnx.Module` attributes"), so the `@struct.dataclass` semantics of `MomentsState` are not affected by this refactor. Decision explicitly justified at Plan line 146. |
| 13 | JIT recompile triggers | Plan §"Non-goals" line 166 forbids optimizer hyperparameter changes (`wm_lr`, `*_eps`, `wrt=nnx.Param` unchanged) — these are what would trigger an XLA cache miss. The graphdef will change shape once (seven children → one composite), causing one fresh trace at the first call after C1 lands; this is expected and amortized over the run | ✅ correct | The single fresh trace at first call is the canonical cost of changing the graphdef structure. Subsequent calls will hit the cache. The plan does not explicitly call this out — a one-line note in the C3 commit ("expect a ~1-2 min XLA cold-start at the first scan call post-refactor; subsequent calls hit the cache") would be a polish item, not a blocker. |
| 14 | CUDA-graph proliferation / `XLA_FLAGS` parity in C4 bench | Plan §C4: "Use `XLA_FLAGS='--xla_gpu_enable_command_buffer='` for parity with the Step-0 baseline measurement" — single flag value, applied to **both** legs of the head-to-head bench (legacy for-loop and scan path) per the back-to-back run instruction | ✅ correct | Empty value disables command buffer caching for both legs; the comparison is symmetric. The plan does not bias the result toward or away from scan by selective flag application. |
| 15 | `--legacy-grad-loop` flag preservation | Plan §R7: "DO NOT remove this flag. It is the rollback. The plan explicitly keeps it through C5 + C6." Plan §C5 may "merge into C3 — see planner's call below" but the flag itself stays. The current default at `dreamer_srl_main.py:195` is `default=False` (scan path is default at CLI), so C5's "default flip" is really a help-text polish, not a behavior change | ✅ correct | The flag persists. The rollback path is intact. R8 also reminds the developer that between C2 and C3 the scan path is still slow, so contributors should pass `--legacy-grad-loop` until C3 lands. |

## Recommended plan revisions

Three small clarifications, ordered by impact:

1. **R2 debug-assertion pseudocode is non-runnable as written.** Plan lines 422-425 show `joint_state['wm_opt']` — `nnx.State` does not subscript like a dict in current flax-nnx releases (`flax==0.12.4` per the project pin, verified at `dreamer_srl_main.py:321` comment). Recommended edit: replace the pseudocode block with the following note (no specific syntax, since it depends on the NNX version):

   > At C1, before the existing scan path runs, add a one-shot diagnostic:
   >
   > ```python
   > _, joint_state = nnx.split(joint)
   > print("[C1 diag] joint_state tree_paths:", jax.tree_util.tree_paths(joint_state))
   > ```
   >
   > Paste the resulting tree-path list into the Implementation Report. The code-reviewer / senior-developer can then verify in PR review that the wm_opt substate path matches the standalone `nnx.split(wm_opt)` tree paths byte-for-byte. If the paths diverge in a way that affects `train_step` (e.g. an extra wrapping level), flag before C3.

2. **L2 test does not exercise JointTrainer-wrapped state.** Plan line 209 says the test fixtures will NOT be rewritten ("Symmetry note: this is intentional"). That is the right call for parity reasons, but it leaves a gap: the L2 test passes today against a seven-everything carry, and will continue to pass after C3 against a JointTrainer carry — but it never compares the **two carry shapes**. If the JointTrainer's substate pytree differs (key naming, extra wrapping, etc.) from the standalone optimizers' substate in a way that doesn't affect the math but DOES affect e.g. checkpoint shape, L2 will not catch it. Recommended edit: add one paragraph to Plan §"Testing Strategy" (between L2 and L3) noting that the C1 R2 diagnostic IS the load-bearing test for "JointTrainer carry shape equals seven-everything carry shape", and that the developer must commit to inspecting the diagnostic output before advancing past C1.

3. **C3.a grep gate is one assertion short.** Plan line 526: "Grep enforcement: `grep -c 'nnx.merge' dreamer_srl_main.py` inside the scan-body block should equal 1." This is correct for the AFTER shape, but the developer also needs:
   - `grep -c 'nnx.split' dreamer_srl_main.py` **outside** scan body, in the post-refactor grad-step block: should equal 1 (only the one `nnx.split(joint)`).
   - `grep -c 'nnx.split' dreamer_srl_main.py` **inside** `_scan_body`: should equal 0 (no inner splits — R3 anti-pattern).
   - `grep -c 'nnx.update' dreamer_srl_main.py` in the post-scan code block: should equal 1 outside the body (the `nnx.update(joint, state_joint_f)` line) plus 1 inside the body (the Polyak `nnx.update(trainer.target_critic, new_target_params)` line). So the total is 2, not 1.

   Recommended edit: rewrite C3.a as a four-line checklist (split outside == 1, split inside == 0, merge inside == 1, update outside == 1, update inside == 1 [Polyak]). Five constraints, not one.

## Critical findings affecting implementation

None of the above three findings block the developer from starting C1 — they are clarifications, not blockers. The plan is structurally sound and matches the proven `dreamer_v3_trainer.py:685-833` reference at every load-bearing point. **The developer may proceed to C1 after the three plan-doc edits above are applied.** All three edits are sub-paragraph in size.

## Conventions audit checklist

The project's `code-reviewer` profile lists a fixed audit checklist. None of the items below apply directly to this plan (the refactor does not touch env, sensor, config, or vmap surfaces), but they are recorded for completeness:

- pytree immutability: ✅ (plan explicitly uses `nnx.update` and `nnx.state`, no in-place mutation of `EnvState`-style structs)
- JIT static-vs-traced fields: ✅ (no `EnvParams` fields touched; `graphdef` is correctly closure-captured, not carry-traced)
- vmap axis correctness: ✅ (no vmap changes anywhere in the plan)
- PRNG threading: ✅ (`carry_key, k_train = jax.random.split(carry_key)` inside body; carry updated; same initial key + same `scan_xs` = same trajectory)
- sensor / observation breakdown sync: n/a (refactor does not touch sensors)
- config protocol (`get_mandatory`): n/a (refactor does not introduce new YAML keys)

## Conclusion

The JointTrainer refactor plan is structurally faithful to the canonical NNX-with-scan pattern at `src/models/dreamer_v3_trainer.py:685-833` and respects every JAX / NNX / Flax structural-correctness rule the project tracks. Three sub-paragraph plan revisions are recommended (R2 pseudocode → tree_paths diagnostic; L2-gap acknowledgment paragraph; C3.a grep checklist expansion). After those land, hand off to the `developer` for C1.

Reviewed by: code-reviewer

## Cross-links

- Plan under review: [`docs/develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md`](../develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md)
- Math review (parallel; forward reference): `docs/reviews/joint_trainer_plan_math_review.md`
- Canonical NNX-with-scan pattern: [`docs/memory/memories/dreamer_diagnosis/20260519_1509_nnx_lax_scan_split_merge_pattern.md`](../memory/memories/dreamer_diagnosis/20260519_1509_nnx_lax_scan_split_merge_pattern.md)
- Triggering perf regression: [`docs/memory/memories/dreamer_diagnosis/20260519_1507_dreamer_srl_v2_cpu_buffer_regression.md`](../memory/memories/dreamer_diagnosis/20260519_1507_dreamer_srl_v2_cpu_buffer_regression.md)
- 4-phase original-Dreamer retrofit history: [`docs/memory/memories/dreamer_diagnosis/20260519_1508_dreamer_jax_perf_retrofit_4_phases.md`](../memory/memories/dreamer_diagnosis/20260519_1508_dreamer_jax_perf_retrofit_4_phases.md)
- Working reference (proven-correct composite-trainer pattern): `src/models/dreamer_v3_trainer.py:685-833` (`_scan_train_gpu` + `train_multiple_gpu`)
- Current dreamer-srl v2 grad-step block under refactor: `src/algorithms/dreamer_srl/dreamer_srl_main.py:860-1090`
- L2 test fixture (will not be touched per Plan §line 209): `tests/algorithms/dreamer_srl/test_lax_scan_train.py:90-286`
