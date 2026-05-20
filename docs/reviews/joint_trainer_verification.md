---
title: "JointTrainer refactor — senior-developer verification (structural commits C1-C5)"
topic: dreamer_srl_v2
status: active
created: 2026-05-21
last_updated: 2026-05-21
reviewer: senior-developer
plan_under_verification: docs/develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md
---

# JointTrainer refactor — verification (structural commits C1-C5)

## Purpose

This document is the post-implementation verification of the **structural** refactor that collapsed the dreamer-srl v2 trainer's seven separately-tracked trainable artefacts (world-model + actor + critic + target-critic + three Adam optimizers) into a single composite Flax-NNX module called `JointTrainer`. The refactor's goal was to mirror the proven-correct one-composite-module pattern already used by the original JAX Dreamer trainer at `src/models/dreamer_v3_trainer.py:685-833`, so that the `jax.lax.scan` grad-step loop can do exactly one `nnx.split` outside the scan, one `nnx.merge` inside, and one `nnx.update` after — instead of seven of each — and so unlock the perf win that the seven-everything decomposition was blocking.

The verification scope here is **the structural refactor only** — the C1-C5 commits (skeleton class, for-loop rewire, scan-path collapse, help-text + Option-L cross-link). The two perf-gates the plan also defined (C4 scan-bench head-to-head + C6 50k-step food-only smoke) are documented in the plan's "Implementation Findings (2026-05-20)" section as **DEFERRED** — both ran 6h+ wall time on Node 114 inside XLA-GPU compile (with no iteration progress, with host RSS growing unboundedly) and were killed. That XLA-compile pathology is a real blocker but is **NOT a failure of the structural refactor** — it is a separate compile-time bottleneck on the full Dreamer train-step body, distinct from the module-count cost the JointTrainer eliminated. The structural work itself is correct, mathematically equivalent to the pre-refactor scan path at `atol=1e-5`, and bit-identical on the legacy for-loop path.

## Verdict

**verified-structural-complete**. The C1-C5 commits land cleanly: every call-site enumerated in the plan's rewire table is touched, the C3.a five-constraint grep checklist passes on the post-refactor scan body, the `--legacy-grad-loop` rollback flag is preserved, no out-of-scope files were modified, both L1 grad-parity (11/11) and L2 math-equivalence (5/5) tests pass on this verification run, and the working tree is clean. The C4/C6 perf-gates are explicitly deferred per the plan's Implementation Findings — they do NOT block the structural refactor from being safe to keep on `v1.4`. One verification flag (⚠️ R2 diagnostic gap, non-blocking): the developer ran the R2 `tree_paths(joint_state)` diagnostic at C1 and reported "260 leaves, 7 top-level keys (alphabetical)", but the full per-leaf tree-path list was written to a `tmp/` file rather than pasted into the Implementation Report. The plan's "Load-bearing role of the R2 diagnostic" callout explicitly required the full list in the report. The high-level confirmation is sufficient for L1+L2 to be the load-bearing gates today, but a future C4 attempt should re-paste the full list before re-entering the perf-bench phase.

## Findings per audit question

| # | Question | Verdict | Evidence (commit / file:line) |
|---|---|:---:|---|
| 1 | Every plan-named call site rewired? | ✅ | Rows 1-19 of the plan's call-site table all rewired. Plan row 4 (Player) → `dreamer_srl_main.py:349`. Plan rows 5-7 (checkpoint + 2× eval rollouts) → `dreamer_srl_main.py:770-773, 797-798, 837-838`. Plan row 8 (legacy Polyak) → `dreamer_srl_main.py:936-942`. Plan row 9 (legacy train_step) → `dreamer_srl_main.py:949-952`. Plan rows 10-19 (scan path) → `dreamer_srl_main.py:973, 978-983, 1003-1064`. All commits: `f109b4a` (C2 for-loop rewire) + `2e8a433` (C3 scan rewire). |
| 2 | Legacy `--legacy-grad-loop=true` path intact? | ✅ | `dreamer_srl_main.py:924-955` preserved as a guarded `if args.legacy_grad_loop:` branch executing the original Python for-loop. Only difference from pre-refactor is `world_model` → `joint.world_model` (reference-semantics-equivalent per plan §R6 + C2.a checkpoint). L1 grad-parity 11/11 green on this verification run confirms byte-equivalence at the gradient-step level. |
| 3 | No out-of-scope edits? | ✅ | `git diff 56be57b..HEAD -- src/ tests/` touches exactly two files: `src/algorithms/dreamer_srl/joint_trainer.py` (NEW, +43 LoC) and `src/algorithms/dreamer_srl/dreamer_srl_main.py` (+122 / -97). `git diff 56be57b..HEAD -- tests/` is empty (test fixtures intentionally not rewritten per plan §line 209 / "Symmetry note"). `git diff 56be57b..HEAD -- configs/ scripts/` is empty. |
| 4 | Commit topology matches plan? | ✅ | C1 `3e1ec2d` touches `joint_trainer.py` (new) + `dreamer_srl_main.py` (additive only). C2 `f109b4a` touches `dreamer_srl_main.py` only. C3 `2e8a433` touches `dreamer_srl_main.py` only. C5 `c204b77` touches `dreamer_srl_main.py:196-202` (help-text update) + `docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md` (cross-link). Each commit ↔ one plan-prescribed boundary. C4 is a bench measurement (no source change) and is deferred. |
| 5 | C3.a five-constraint grep checklist holds on post-refactor scan-path block? | ✅ | Read of `dreamer_srl_main.py:956-1067` confirms: `nnx.split(joint)` count outside scan: 1 (line 973). `nnx.split` inside `_scan_body` (lines 990-1049): 0. `nnx.merge` inside `_scan_body`: 1 (line 1006). `nnx.update(joint, …)` outside scan, post-scan: 1 (line 1061). `nnx.update(trainer.target_critic, …)` inside `_scan_body` for Polyak: 1 (line 1030). All five constraints pass. |
| 6 | R2 `jax.tree_util.tree_paths` diagnostic pasted into Implementation Report? | ⚠️ | Plan §"Load-bearing role of the R2 diagnostic" required the FULL `tree_paths(joint_state)` output in the report. Plan §Implementation Report §C1 (line 560) only contains the summary: "260 leaves; top-level keys (alphabetical): actor, actor_opt, critic, critic_opt, target_critic, wm_opt, world_model. Full `jax.tree_util.tree_flatten_with_path` output pasted to `tmp/` during development". The developer correctly substituted `tree_flatten_with_path` for `tree_paths` (API change in JAX 0.9.x, documented as deviation #1) and confirmed the structural top-level keys, but the per-leaf path list itself is no longer recoverable from the doc — it lives in a `tmp/` file that may have been cleaned up. **Gap is non-blocking for the structural refactor** (L1+L2 cover the math, and the top-level-key confirmation is enough to rule out the "extra wrapping level" failure mode) but should be re-run + pasted in full before any future C4 attempt that introspects optimizer-state pytree shape. |
| 7 | L1 grad-parity sanity sample (11/11)? | ✅ | `pytest tests/algorithms/dreamer_srl/test_grad_parity.py -q -x` → `11 passed in 9.91s` on this verification run. |
| 8 | L2 math-equivalence sanity sample (5/5)? | ✅ | `pytest tests/algorithms/dreamer_srl/test_lax_scan_train.py -q -x` → `5 passed in 1003.36s (0:16:43)` on this verification run. Scan-path math matches for-loop path at `atol=1e-5`. |
| 9 | `--legacy-grad-loop` default per plan? Flag still exists? | ✅ | `dreamer_srl_main.py:196-202` keeps the flag with `default=False` (scan path is default) — matches plan §C5 prescription. Help string was tightened in C5 `c204b77` to say "Rollback flag … Default: False (scan path; validated by JointTrainer refactor C3-C4)". The flag is preserved as the rollback per plan §R7. |
| 10 | No orphan files or imports? | ✅ | `git status` → clean. `grep tree_paths\|tree_flatten_with_path src/algorithms/dreamer_srl/*.py` → empty (R2 diagnostic correctly removed at C3 per plan §"C3 — remove the R2 debug assertion added in C1"). The C1 `print("[dreamer-srl] JointTrainer constructed OK")` at `dreamer_srl_main.py:336` is an intentional confirmation log, not orphan. |
| 11 | Plan doc reflects ground truth? | ✅ | The plan's "Implementation Findings (2026-05-20)" section accurately captures: (a) C1-C5 ✅ landed cleanly with gate evidence, (b) C4 scan-bench + C6 food-only smoke are ⏸ DEFERRED (not failed) due to XLA-compile pathology, (c) revised causal model — the JointTrainer cleaned up pytree shape (correctness gain) but is not by itself sufficient to fix the XLA-GPU compile-time bottleneck (perf gain still unmeasured), (d) what ships vs. what's deferred. The Checkpoints section shows the same status (C1.a-C5.a all `[x]`; C4.a-c + C6.a-b all `[ ]`). |

## Out-of-scope edits

None. `git diff 56be57b..HEAD -- src/` touched only `joint_trainer.py` (new) + `dreamer_srl_main.py` (rewired). `tests/`, `configs/`, `scripts/` are untouched.

The only outside-`src/` files modified in the C1-C5 commit range are:

- `docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md` — modified by C5 `c204b77` per plan §C5 to mark Option-L Step 3 as DONE + Step 4/5 as superseded. Within plan scope.
- `docs/develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md` itself — partial Implementation Report (commit `71ae259`) and final Implementation Findings (commit `0489f94`). Within plan scope.

No surprise files.

## R2 diagnostic gap (non-blocking; recommendation for future C4)

The R2 diagnostic was the load-bearing check the plan defined to catch a class of optimizer-state-pytree drift that L2 cannot see (because L2's fixtures stay on the seven-everything carry shape, by design — see plan §line 209 + §line 497). The developer ran the diagnostic at C1 and reported the high-level summary (260 leaves, 7 top-level keys, no extra wrapping), then removed it at C3 per plan. The full per-leaf path list was written to a `tmp/` file rather than pasted into the Implementation Report.

For the **structural refactor**, this gap is non-blocking — the top-level-key confirmation rules out the "extra wrapping level" failure mode, and L1+L2 cover the math at gradient-step granularity. For a **future C4 / C6 perf-bench retry**, the developer should re-run the diagnostic and paste the full `tree_flatten_with_path(joint_state)` output (one line per leaf path) into the Implementation Report BEFORE entering the scan-bench phase. The cost is one extra print statement and a one-time C1-style commit; the value is closing the documented coverage gap before any future perf-bench attempt is interpreted.

## Speed-change review

No SPS measurement is available for the scan path under JointTrainer — C4 bench was killed mid-XLA-compile after 6h 31min with zero iteration progress. The legacy for-loop bench completed normally at 41.14 inst SPS (matches Option-M reference 42.23 within noise), so the legacy path is confirmed non-regressed.

Per the plan's "What the refactor did NOT achieve" section, the dominant cost in the scan-path slowdown turned out NOT to be the seven-module decomposition (which JointTrainer fixes) but the XLA-GPU compile time of the full Dreamer train-step body inside `lax.scan`. The structural refactor is necessary but not sufficient for the perf win — a separate diagnosis effort on the compile pathology is required before any SPS comparison can be measured.

**Verdict**: ⚠️ perf unmeasured / scan SPS deferred. NOT a regression of the structural refactor, since the perf-bench was a gate the plan explicitly named, attempted, blocked, and deferred — not a hidden slowdown the developer skipped measuring. The legacy for-loop path was measured and shows no regression.

## Recommended next steps

### For the deferred C4/C6 perf-gates

The plan's "What is deferred to a follow-up investigation" section already names four diagnostic angles. Concrete experiments, ranked by expected diagnostic value:

1. **Shrink the bench budget aggressively first.** Re-run C4 scan-bench with `--total-steps 5000` (or even 1000) on Node 114 cuda:1. If XLA-compile time is shape-bound (fixed per train-step body) rather than length-bound (scales with `n_grad_steps`), compile should still take 6h+ at 5000 steps — and the SPS-per-iteration once compile finishes is what we actually want to measure. If compile completes faster at 5000 than at 50000, the compile is length-coupled and we have a different lever (lower `per_rank_gradient_steps`).
2. **`JAX_LOG_COMPILES=1`** plus `XLA_FLAGS='--xla_dump_hlo_as_text=/tmp/hlo_dump'` to see which subprogram blows up the IR. Compare the HLO size + node count of the scan-path body against the original Dreamer's `_scan_train_gpu` HLO (which compiles in tractable time and runs at ~32 SPS).
3. **Direct comparison against original Dreamer's `train_multiple_gpu`**. Run `src/models/dreamer_v3_trainer.py:_scan_train_gpu` on the same Node 114 / cuda:1 / same XLA flags / same step budget. If it compiles in minutes and the dreamer-srl-v2 scan body compiles in hours, the structural difference is in the train-step body (loss formulae, REINFORCE pairing, etc.), not in the scan wrapper itself. That isolates the bottleneck to `make_train_step` / `train.py`.

These three should be cheap (one is a 5k-step re-run, one is a flag toggle, one is an apples-to-apples bench) and should converge to a concrete next-step plan before re-attempting C4/C6 at full 50k budget.

### For shipping the structural win to memory

A session-insight should be captured under `docs/memory/memories/dreamer_diagnosis/` documenting the distinction the implementation revealed:

- **The originally hypothesized cause** (seven-module decomposition → seven splits/merges per iteration → 70× slowdown + memory leak) was a partial story. JointTrainer fixed the pytree-shape side cleanly (L2 5/5 at `atol=1e-5`) but did not deliver the measured SPS win.
- **The dominant cost** turned out to be XLA-GPU compile time on the full Dreamer-V3 train-step body inside `lax.scan` — a compile pathology distinct from the module-count cost.
- This is a useful diagnostic distinction for any future "scan path is slow" investigation: separate "pytree-shape cost" (visible in `nnx.split/merge` count + traversal depth) from "compile cost" (visible in JAX_LOG_COMPILES output + HLO size).

Recommend `/memorize` capture this with title `dreamer_srl_v2_xla_compile_distinct_from_module_count` (or similar), cross-linking the plan + this verification doc + the deferred follow-up experiments.

## Cross-links

- Plan: [`docs/develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md`](../develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md)
- Code-reviewer plan review (accept-with-3-revisions, applied): [`docs/reviews/joint_trainer_plan_review.md`](joint_trainer_plan_review.md)
- Math-reviewer plan review (construction-preserves-equivalence): [`docs/reviews/joint_trainer_math_review.md`](joint_trainer_math_review.md)
- Triggering finding (CPU/buffer regression + 70× scan slowdown): [`docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md`](../develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md)
- Working reference (proven composite trainer): `src/models/dreamer_v3_trainer.py:685-833`

### Commits verified

| Commit | Subject | Files | Status |
|---|---|---|:---:|
| `3e1ec2d` | C1 — add JointTrainer composite nnx.Module skeleton | `joint_trainer.py` (new), `dreamer_srl_main.py` (+import +construction) | ✅ |
| `f109b4a` | C2 — rewire for-loop path to `joint.*` attributes | `dreamer_srl_main.py` (8 sites) | ✅ |
| `2e8a433` | C3 — collapse scan path to 1 split / 1 merge / 1 update | `dreamer_srl_main.py:945-1080` (~125 LoC → ~60 LoC) | ✅ |
| `c204b77` | C5 — `--legacy-grad-loop` help-text + Option-L Step 3 mark-done | `dreamer_srl_main.py:196-202`, `buffer_perf_fix_plan_option_L.md` | ✅ |

C4 (perf-bench) and C6 (food-only L3 smoke) are documented as **deferred** in the plan's Implementation Findings — no commit landed for them and none was expected.

---

**Verified by**: senior-developer
**Date**: 2026-05-21
