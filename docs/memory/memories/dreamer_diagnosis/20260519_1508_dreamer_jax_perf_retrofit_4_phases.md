---
id: 20260519_1508_dreamer_jax_perf_retrofit_4_phases
date: 2026-05-19
time: "15:08"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, design, decision]
summary: "On 2026-02-21, the original JAX Dreamer trainer was retrofitted in 4 measured phases that took DreamerV3 training from 'impractical' to 'production-ready': encoder outside scan (Phase 1), JIT-compile train_step (130×), JIT collection loop via lax.scan (596×), vectorize replay sampling (4.8×). This insight preserves the lesson at a point in time AFTER the memory/diary system existed."
related: ["20260519_1507_dreamer_srl_v2_cpu_buffer_regression", "20260519_1509_nnx_lax_scan_split_merge_pattern"]
session_origin: claude_code
session_label: "dreamer-srl v2 perf diagnosis + M-cell wakeup chain"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# The 4-phase JAX Dreamer performance retrofit (2026-02-21)

## Key conclusion
The original-JAX-Dreamer trainer was retrofitted on 2026-02-21 (commit `0dec9e0`) in four named phases with measured speedups. Two of those phases (1 and 3) are about putting work inside or outside `lax.scan` correctly; one (2) is about JIT-compiling the inner training step; one (4) is about vectorising replay sampling. Together they took DreamerV3 from "impractical" (Python-loop-bound) to production-ready. **This insight is captured now because the original implementation predates the project's memory and diary systems** — the retrofit history was at risk of being known only to the user and recoverable only via `git log + git show`. Future-Claude (and the upcoming dreamer-srl v2 perf work — see [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]]) should treat the original Dreamer's `train_multiple_gpu` at `src/models/dreamer_v3_trainer.py:791-833` as the **near-direct template** for any NNX+JAX training-loop optimisation.

## Evidence, measurements, facts
**Source**: commit `0dec9e0` (2026-02-21) with message: *"perf: Implement and document DreamerV3 performance optimizations, including encoder outside scan, JIT compilation, and vectorized sampling"*. The retrofit doc lived at `docs/DREAMER_V3_COMPARISON.md` (deleted later; content recoverable via `git show 0dec9e0:docs/DREAMER_V3_COMPARISON.md`).

**The 4 phases:**

| Phase | Optimisation | Measured speedup | Mechanism |
|---|---|---|---|
| 1 | **Encoder outside scan** | (qualitative; documented as critical) | Encode the full `[T, B, obs_dim]` slab in one batched encoder pass → embeddings `[T, B, D]`. The `lax.scan` over time then runs on embeddings, not raw obs. Avoids $T$ sequential encoder calls inside the scan body. Doc quote: *"Allowed the GPU to process all observations in a single batched pass instead of T sequential calls."* |
| 2 | **JIT-compile `train_step`** | **130×** | Combined World Model + Behavior Learning + Moment Updates into a single XLA program. Doc quote: *"Achieved 130× speedup on the core training update."* |
| 3 | **JIT collection loop via `lax.scan`** | **596×** | Moved env interaction (inference + step + reset) into `jax.lax.scan` instead of a Python `for` loop. Doc quote: *"Collection is no longer a bottleneck."* |
| 4 | **Vectorize replay sampling** | **4.8×** | Replaced Python `for` loops in sampling with broadcasted NumPy indexing. Doc quote: *"4.8× speedup on batch preparation."* |

**Phase 1 generalisable rule:** anything that doesn't need to be sequential (encoders, MLPs, layer-norm passes) should run **once on the whole `[T, B]` slab** before/after the scan, not inside the scan body. The scan body should carry only the recurrent state + the truly-sequential math. Scaling cost of an N-layer MLP on `[T*B, D]` is one matmul; the same MLP inside a scan body over T steps is T matmuls with launch overhead each — JIT cannot fuse them because the scan's intra-iteration data dependency is on the carry, not on the parameters.

**Phase 3 generalisable rule:** if a `for` loop in Python ever runs more than a few iterations of GPU work, JIT it via `lax.scan` (or `lax.fori_loop`). The 596× speedup the original Dreamer measured was on env collection, but the same logic applies to a training-step loop. Caveat: the inner body of the scan has to satisfy `nnx.split / nnx.merge` discipline if it touches a flax module (see [[20260519_1509_nnx_lax_scan_split_merge_pattern]] for the technique).

**The follow-up commit** `6705735` ("Update GPU training loop to prevent retracing") fixed a separate symptom that emerged after Phase 3: the `lax.scan` body initially re-traced on every call because the buffer object identity wasn't stable. The fix lives in `src/models/dreamer_v3_trainer.py:799-828`: pass buffer arrays explicitly (`buffer_arrays = (buffer.obs, buffer.actions, ...)`) instead of passing the buffer object. See [[20260519_1509_nnx_lax_scan_split_merge_pattern]] §Issue 2 for the full pattern.

**Working reference in-repo:**
- `src/models/dreamer_v3_trainer.py:791-833` — `train_multiple_gpu` (the entrypoint).
- `src/models/dreamer_v3_trainer.py:685-790` — `_scan_train_gpu` (the JIT'd scan body).
- `src/models/dreamer_v3_trainer.py:923-1024` — `ReplayBuffer` class with `device="gpu"` mode.
- Test fixtures (preserved in commit `0dec9e0` history but possibly deleted later): `test_phase1_encoder_optimization.py`, `test_phase3_collection_jit.py`, `test_phase4_sampling_jit.py` — verified each phase's parity vs the pre-optimisation baseline.

## Decisions and actions
- This insight is the **engineering template** for two related upcoming pieces of work:
  1. The dreamer-srl v2 perf fix (option M/L per [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]]) — uses Phase 3 + buffer-on-GPU + the split/merge pattern.
  2. Any future JAX/NNX training-loop perf work in the project.
- **Do NOT lose Phase 1's rule** ("encoder outside scan"). It's the easiest to violate by accident — a future contributor might put the encoder inside a scan body for "cleanliness" and silently lose 100× on perf.
- **Reference the 4 commits as the audit trail**:
  - `0dec9e0` — main retrofit commit (the 4 phases together).
  - `6705735` — retracing fix (the split/merge pattern).
  - `6d46e3d` — buffer config mandatorisation.
  - `742fdb6` — unified GPU/CPU buffer dispatch.
- **If `docs/DREAMER_V3_COMPARISON.md` ever needs to be restored** as a living doc: pull from `git show 0dec9e0:docs/DREAMER_V3_COMPARISON.md`. The doc was later removed (likely during the v2/v3 docs reorganisation); its content is preserved in git and now in this insight.

## Open questions and follow-ups
- The original implementation's parity tests (`test_phase1_encoder_optimization.py` etc.) — are they still in the repo or were they deleted? Search the current tree for `phase1_encoder` to confirm. If deleted, that's a knowledge-loss risk for the retrofit's verification path; consider restoring as part of any future re-retrofit work.

## References
- See [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] for the current dreamer-srl v2 regression that this template will be applied to fix.
- See [[20260519_1509_nnx_lax_scan_split_merge_pattern]] for the specific technique needed to make `nnx.Module` work inside `lax.scan`.
- Working reference in-repo: `src/models/dreamer_v3_trainer.py:791-1024`.
- Historical doc (deleted; restorable from git): `docs/DREAMER_V3_COMPARISON.md` at commit `0dec9e0`.
- Commits: `0dec9e0` (main retrofit), `6705735` (retracing fix), `6d46e3d` (buffer config), `742fdb6` (unified buffer).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260519_1508_dreamer_jax_perf_retrofit_4_phases.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] (dreamer_diagnosis, 2026-05-19) — dreamer-srl v2's replay buffer is 100% numpy/CPU (faithful to sheeprl's PyTorch 
- [[20260519_1509_nnx_lax_scan_split_merge_pattern]] (dreamer_diagnosis, 2026-05-19) — To run a flax NNX Module inside jax.lax.scan without re-tracing on every call, s
- [[20260521_0151_xla_scan_body_compile_dominates_module_count]] (dreamer_diagnosis, 2026-05-21) — Earlier claim — that dreamer-srl's 7-module decomposition CAUSED the 70× lax.sca
- [[20260630_1720_dreamer_srl_train_step_jit_compile_once]] (dreamer_diagnosis, 2026-06-30) — The dreamer_srl gradient-step lax.scan was written bare in the Python loop with 
<!-- END BACKLINKS -->
