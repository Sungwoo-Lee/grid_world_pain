---
id: 20260521_0151_xla_scan_body_compile_dominates_module_count
date: 2026-05-21
time: "01:51"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, refutation, design, meta]
summary: "Earlier claim — that dreamer-srl's 7-module decomposition CAUSED the 70× lax.scan regression — was a partial story. The JointTrainer refactor (composite of 7 → 1) landed cleanly with full math-equivalence preserved (5/5 atol=1e-5) but the XLA-GPU scan-body compile-time was still pathologically unbounded (6h31m / 85 GB RSS, no iter progress). The real dominant cost is XLA-GPU compiling the full Dreamer train_step body inside lax.scan; the module-count cost is real but secondary."
related: ["20260518_1511_dreamer_srl_v2_parity_pass_outperform", "20260519_1507_dreamer_srl_v2_cpu_buffer_regression", "20260519_1508_dreamer_jax_perf_retrofit_4_phases", "20260519_1509_nnx_lax_scan_split_merge_pattern"]
session_origin: claude_code
session_label: "dreamer-srl v2 perf fix — JointTrainer refactor session"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# XLA scan-body compile-time dominates module-count cost in dreamer-srl

## Key conclusion
The JointTrainer refactor (collapsing dreamer-srl's 7 separate `nnx.Module` / optimizer instances into 1 composite — matching the original JAX Dreamer's `train_multiple_gpu` template at `src/models/dreamer_v3_trainer.py:685-833`) was the structurally correct change but **did not resolve the lax.scan perf regression**. The refactor's math is verified at atol=1e-5 (5/5 L2 math-equiv tests pass on CPU in 4 minutes), bit-identity is preserved on the legacy for-loop path (11/11 L1 grad-parity), and the C3.a 5-constraint grep checklist confirms the post-refactor scan body uses 1 split / 1 merge / 1 update — but the actual GPU benchmark ran 6h 31min with **zero iteration progress** and host RSS growing unboundedly 44 → 85 GB. The earlier claim ([[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]]) — that 7 split/merge per scan iteration was the *cause* of the 70× regression — was at best a partial story. The dominant cost is XLA-GPU compiling the full Dreamer training step (world model + actor + critic + target Polyak + 3 optimizers all unrolled inside a `lax.scan` body), and that cost is determined by **the train_step body's pytree size and instruction count**, not by the module count surrounding it. JointTrainer is necessary (the legacy 7-module path also exhibits the issue, just amplified) but not sufficient.

## Evidence, measurements, facts

**Bench infrastructure ruled out a transient issue:**
- Three benches across three nodes (n105, n113, n114) all exhibited the same long-XLA-compile pattern when the scan path was the default.
- The pattern was independent of GPU model (n105's 11 GB, n113's 24 GB, n114's 48 GB) and independent of whether the buffer was on CPU or GPU.
- The `--xla_gpu_enable_command_buffer=` flag (empty, disabling CUDA-graph caching) did not change the symptom.
- Math-equivalence tests on CPU complete in 4 minutes — proving the math composition is correct and the compile-time blowup is GPU-specific.

**JointTrainer refactor (the "structural fix") landed cleanly:**
- 5 commits C1-C5 (`3e1ec2d` → `c204b77` on v1.4) with all parity gates green.
- L1 grad-parity (bit-identity vs sheeprl reference): 11/11 pass.
- L2 math-equivalence (atol=1e-5 between for-loop and scan path): 5/5 pass.
- C3.a grep checklist on `src/algorithms/dreamer_srl/dreamer_srl_main.py`: `nnx.split` outside scan = 1, inside scan = 0; `nnx.merge` inside scan = 1; `nnx.update` outside = 1, inside (Polyak) = 1. All five constraints hold simultaneously.
- R2 `tree_paths(joint_state)` diagnostic: 260 leaves, 7 alphabetical top-level keys, no extra wrapping vs the standalone splits.
- Out-of-scope edit count: zero (verified by senior-developer Verification Protocol at `docs/reviews/joint_trainer_verification.md`).
- Refactor matches the original JAX Dreamer's `train_multiple_gpu` pattern at `src/models/dreamer_v3_trainer.py:685-833`, which DOES sustain ~32 SPS in production on the `yxij4lrc` XS/16/4M run — so the structural pattern IS correct in principle.

**The XLA-compile pathology that the refactor did NOT resolve:**
- C4 scan bench on n114/cuda:1: 6h 31min wall time, RSS grew 44 → 85 GB unboundedly, zero iteration progress logged. Killed.
- C6 food-only smoke on n114/cuda:2: 3h 28min wall time, RSS 35 GB, same symptom. Killed.
- Step 3's pre-refactor scan bench on n105 (with 7-module decomposition): 5h 13min wall time before OOMing at iter 1202/3125 with `RESOURCE_EXHAUSTED: 35 alive command buffers, CUDA_ERROR_OUT_OF_MEMORY`. Sustained inst SPS during that run was 0.4 (70× slower than the 28.33 legacy for-loop on the same node).
- The pre-refactor and post-refactor benches BOTH exhibit the same long-compile pathology; the post-refactor version is structurally cleaner but no faster to compile.

**Three possible root-cause angles for the dominant XLA cost** (untested in this session, captured for follow-up):
1. **Train_step body size**: the full Dreamer-V3 train_step has ~15 separate gradient calls (WM + actor + critic + their losses + target Polyak), each with its own backward pass. Unrolling all of them inside a `lax.scan` body multiplies the IR size by `n_grad_steps`. Original Dreamer's working `train_multiple_gpu` may have a smaller per-step body that JIT-compiles in tractable time.
2. **Pytree leaf count**: even with JointTrainer (260 leaves), the scan body's input/output pytree is large. XLA's auto-fusion / loop-unrolling heuristics may explode on pytrees this large.
3. **Per-leaf gradient broadcasting**: dreamer-srl uses 3 separate optax optimizers, each with its own gradient pytree. The scan body must thread 3 such pytrees through every iteration. The original-Dreamer composite trainer may use 1 unified optimizer that produces 1 gradient pytree.

## Decisions and actions

- **Keep JointTrainer on v1.4**: it's a structurally correct refactor whose math is parity-verified at both bit-identity (legacy) and atol=1e-5 (scan). Reverting would lose a clean architectural alignment with the proven `train_multiple_gpu` template without diagnostic benefit. (Senior-developer Verification Protocol explicitly recommended KEEP at `docs/reviews/joint_trainer_verification.md`.)
- **`--legacy-grad-loop` flag remains the production default** as the rollback path. The scan path stays opt-in until the XLA-compile pathology is resolved.
- **Refines [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]]'s causal claim**: that insight named the 7-module decomposition as "the" cause. This new insight refines that to "*a* cost, but not *the* dominant cost." The old insight's `valid_until: 2026-06-19` date passes by then anyway; future-Claude reading both should treat this newer insight as the refined model.
- **Architectural-defense lesson for future refactors**: the 4-agent verification chain (senior-developer plan → code-reviewer + math-reviewer in parallel → developer C1-C5 commit-by-commit with parity gates → senior-developer Verification Protocol) successfully caught integration mismatch risk. Specifically: the code-reviewer's pre-impl audit caught 3 plan-doc bugs (non-runnable pseudocode, L2-coverage gap, incomplete grep checklist) that would have produced silent failures during implementation. This methodology is reusable beyond this one refactor and should be the template for any future parity-tested codebase refactor (the user explicitly requested this captured as a pattern).

## Open questions and follow-ups

**For future XLA-compile-pathology investigation** (the deferred C4/C6 perf gates):
- Run `JAX_LOG_COMPILES=1` against the scan path to see which subprogram blows up the IR.
- Try `--total-steps=5000` (10× smaller) to test whether compile time scales with `n_grad_steps` or is fixed-per-shape. If it's fixed, the IR explosion is per-leaf; if it scales linearly, it's per-iteration unrolling.
- Compare with original Dreamer's `train_multiple_gpu` directly via `train.py` — does it also have a long compile? If not, what's structurally different about its train_step body or optimizer wrapping?
- Consider `jax.disable_jit()` for a correctness-only run to isolate whether the issue is the compile itself or some recursive trace.
- Consider refactoring the train_step body to reduce its per-iteration size (e.g., split into world-model-only and actor-critic-only scans).

**Related but separate work that this session unblocked**:
- The death-penalty ablation (Phase 3 design at `docs/experiments/active/dreamer_srl_v2/death_penalty_ablation_design.md`) can now launch on the v1.4 codebase as designed — it uses the `--legacy-grad-loop=true` default which is unchanged behaviour vs pre-refactor.
- The §10.4 second-pass synthesis (task #41) remains pending and is independent of this perf work.

## References

- Plan: [`joint_trainer_refactor_plan.md`](../../../develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md) — includes the full "Implementation Findings (2026-05-20)" section documenting what landed and what was deferred.
- Code-review: [`joint_trainer_plan_review.md`](../../../reviews/joint_trainer_plan_review.md) — accept-with-3-revisions; all 3 applied.
- Math-review: [`joint_trainer_math_review.md`](../../../reviews/joint_trainer_math_review.md) — construction-preserves-equivalence verified.
- Sr-dev verification: [`joint_trainer_verification.md`](../../../reviews/joint_trainer_verification.md) — verified-structural-complete; KEEP recommendation.
- Refined by this insight: [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] (whose causal claim is refined here, not superseded — the original perf-regression diagnosis stands; the *cause hierarchy* is what this insight refines).
- Companion technique insight: [[20260519_1509_nnx_lax_scan_split_merge_pattern]] (the canonical `nnx.split/merge` pattern — JointTrainer applied this correctly, proving the pattern itself isn't the bottleneck).
- Historical retrofit: [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]] (original JAX Dreamer's 2026-02-21 retrofit; their composite-trainer pattern is what JointTrainer matches).
- Parity-pass anchor: [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]] (v2's parity success that the JointTrainer refactor preserves).
- Working reference: `src/models/dreamer_v3_trainer.py:685-833` (`train_multiple_gpu` / `_scan_train_gpu`) and `:923-1024` (`ReplayBuffer` with `device="gpu"` mode).
- Per-step bench progression: `docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md` "User Decision (2026-05-20)" section.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260521_0151_xla_scan_body_compile_dominates_module_count.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260521_0152_parity_tested_refactor_verification_chain]] (subagent_engineering, 2026-05-21) — A 4-agent verification chain (senior-developer plan → code-reviewer + math-revie
<!-- END BACKLINKS -->
