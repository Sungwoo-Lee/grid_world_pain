---
id: 20260519_1507_dreamer_srl_v2_cpu_buffer_regression
date: 2026-05-19
time: "15:07"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, meta]
summary: "dreamer-srl v2's replay buffer is 100% numpy/CPU (faithful to sheeprl's PyTorch design), but this is a perf regression vs the original JAX Dreamer trainer which kept the buffer on GPU. The hottest line is dreamer_srl_main.py:885 — jnp.asarray() inside the gradient-step loop triggers a host→device transfer per gradient step (~tens per training iteration)."
related: ["20260518_1511_dreamer_srl_v2_parity_pass_outperform", "20260519_1508_dreamer_jax_perf_retrofit_4_phases", "20260519_1509_nnx_lax_scan_split_merge_pattern"]
session_origin: claude_code
session_label: "dreamer-srl v2 perf diagnosis + M-cell wakeup chain"
importance: high
status: settled
valid_until: 2026-06-19
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# dreamer-srl v2's replay buffer is CPU/numpy — perf regression vs original JAX Dreamer

## Key conclusion
dreamer-srl v2 ports sheeprl's PyTorch replay buffer as pure numpy/CPU (`src/algorithms/dreamer_srl/buffers.py:25-66`), which was a faithful port choice but is a performance regression against the original-JAX-Dreamer trainer (`src/models/dreamer_v3_trainer.py:923-1024`), which kept the entire buffer on GPU as JAX arrays. The current dreamer-srl driver's gradient-step loop (`dreamer_srl_main.py:866-895`) calls `jnp.asarray(v[i], dtype=jnp.float32)` **inside** the loop at line 885, triggering one host→device transfer per gradient step (with sheeprl's grad-step ratio, that's tens of H2D transfers per training iteration). The empirical signature shows up in the M-cell sweep throughput: E8 M/num_envs=128 ran ~30% slower per env-step than E7 M/num_envs=64, consistent with buffer-bandwidth scaling sub-linearly with envs because each new env doubles buffer-add + buffer-sample volume without adding GPU compute.

## Evidence, measurements, facts
- **Buffer storage type** (CPU/numpy):
  - `src/algorithms/dreamer_srl/buffers.py:25` — `import numpy as np` (no jax).
  - `src/algorithms/dreamer_srl/buffers.py:63` — `self._buf: Dict[str, np.ndarray] = {}`.
  - `src/algorithms/dreamer_srl/buffers.py:66` — `self._rng: np.random.Generator = np.random.default_rng()`.
  - `src/algorithms/dreamer_srl/buffers.py:355-376` — sample = `np.take(np.reshape(...))` + `np.swapaxes(...)` (all CPU).
- **Original Dreamer JAX trainer's GPU buffer mode** (the right pattern):
  - `src/models/dreamer_v3_trainer.py:924` — `device="gpu"` is the default.
  - `src/models/dreamer_v3_trainer.py:930-935` — when on GPU, all buffer arrays allocated as `jnp.zeros(...)`.
  - `src/models/dreamer_v3_trainer.py:962-967` — GPU `add_batch` uses `jnp.arange` + `.at[indices].set(...)`.
  - `src/models/dreamer_v3_trainer.py:997-1000` — GPU `sample` uses `jax.random.randint(key, ...)` + `jnp.arange`.
  - `src/models/dreamer_v3_trainer.py:1017-1018` — explicit comment: *"For GPU: not needed — sample inside lax.scan instead"*.
- **The hot line** (per-gradient-step H2D transfer):
  - `src/algorithms/dreamer_srl/dreamer_srl_main.py:866-895` — for-loop over `n_grad_steps`.
  - `src/algorithms/dreamer_srl/dreamer_srl_main.py:885` — `arr = jnp.asarray(v[i], dtype=jnp.float32)` is called inside the loop on a numpy slice → H2D transfer per gradient step.
- **Empirical signature** (throughput sub-linear in num_envs):
  - E7 M/num_envs=64/2M: 9.05 env-steps/sec average (avg), 8.40 recent.
  - E8 M/num_envs=128/2M: 6.21-6.85 env-steps/sec — ~30% slower per env-step.
  - If GPU compute were the bottleneck, num_envs doubling would scale near-linearly. Sub-linear scaling matches buffer-bandwidth contention.
- **Per-iteration transfer volume estimate**: with num_envs=128, batch_size=16, seq_len=64, ~5 keys (obs+actions+rewards+dones+is_first), data per batch ≈ 200 KB. With ~64 grad steps per iteration (sheeprl ratio=0.5 × num_envs=128), total H2D ≈ 12.8 MB/iter. The dominant cost is per-call CUDA launch overhead, not bandwidth.

## Decisions and actions
- **Captured in this insight** specifically because *the original JAX Dreamer implementation predated the project's memory + diary systems* — the GPU-buffer optimisation history was at risk of being lost to git archaeology. See companion insights [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]] for the full 4-phase retrofit story and [[20260519_1509_nnx_lax_scan_split_merge_pattern]] for the NNX-with-scan technique.
- **Fix ladder** (3 options, stackable; option S non-blocking on option L):
  - **Option S (small)**: hoist the H2D transfer out of the gradient-step loop. Call `jax.tree.map(jnp.asarray, local_data)` once at `dreamer_srl_main.py:865` (after `buffer.sample(...)`), so `local_data[k][i]` slices are JAX-native. **One H2D per iteration instead of `n_grad_steps`.** ~5-10 line change. Expected: 2-5× iteration speedup.
  - **Option M (medium)**: port the original-Dreamer `_on_gpu` pattern to dreamer-srl. Replace `np.ndarray` storage with `jnp.ndarray`. Implement `.at[].set(...)`-based `add()` and `jax.random.randint(key, ...)`-based `sample()`. ~100-line patch. Expected: 3-10× iteration speedup.
  - **Option L (large)**: vectorize the whole train loop inside `lax.scan` (like original-Dreamer's `train_multiple_gpu` at `src/models/dreamer_v3_trainer.py:791-833`). The original Dreamer measured **596× speedup** on its collection loop with this pattern (from [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]]). ~200-line port; needs grad-parity tests.
- **Constraint**: the bit-identity grad-parity tests in `tests/algorithms/dreamer_srl/test_grad_parity.py` sample via numpy paths for sheeprl-comparability. Options M and L break this byte-equality even when the math is right; needs an opt-in flag or a separate test path. Option S preserves the test path entirely.

## Open questions and follow-ups
- **Is option S sufficient or do we need M/L?** Option S addresses only the H2D launch-overhead-per-grad-step; it doesn't remove the numpy-side compute (`np.take`/`np.swapaxes` in `buffers.py:355-376`) or the bandwidth cost. If option S gives 2-3× speedup, option M may still be worth ~2× more on top. Decision: ship option S first, measure, then decide on M.
- **Does the slow throughput on the M cells explain the rPPO performance gap entirely?** Probably not — the gap was on the 10×10 hypervigilance task at the XS recipe where buffer pressure is lower (num_envs=16 vs 128); see Phase 1 anchor doc for the reward-head asymmetry story which is the leading H1 hypothesis there.
- **`buffer._pos` and `buffer._full` Python-side state** under option M: if buffer arrays move to GPU but `_pos` stays Python-int, the host has to sync with device on every `add()` to know fullness. Original Dreamer kept `self.idx` and `self.size` as Python ints — fine, but verify we don't accidentally JIT through a host-side condition.

## References
- See [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]] for the original-Dreamer 4-phase perf retrofit (the engineering template for option L).
- See [[20260519_1509_nnx_lax_scan_split_merge_pattern]] for the NNX-module-inside-`lax.scan` pattern (the technique needed for option L).
- Phase 2a code audit: `docs/reviews/dreamer_srl_reward_head_audit.md` — §Q3 covers buffer sampling correctness (uniform, unweighted) but did NOT flag the CPU/GPU placement issue. The audit was scoped to correctness, not performance — this insight closes that gap.
- Phase 1 anchor: `docs/experiments/active/dreamer_srl_v2/REWARD_HEAD_ASYMMETRY_ANALYSIS.md` — the rPPO-vs-Dreamer gap analysis. The buffer regression here is a SEPARATE performance bottleneck from the reward-head asymmetry; both are real.
- Sheeprl upstream reference: `vendor/sheeprl/sheeprl/data/buffers.py` — sheeprl's PyTorch buffer is also CPU-side (PyTorch's `.to(device)` is fast, so it's fine for sheeprl). The faithful port to JAX inherited the slow path because JAX's host↔device boundary is heavier than PyTorch's.
- Settled-companion insight on parity (food-only task succeeded with same buffer): [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]] — explains why parity passed (low buffer pressure on food-only) while hypervigilance shows the gap.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260519_1507_dreamer_srl_v2_cpu_buffer_regression.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]] (dreamer_diagnosis, 2026-05-19) — On 2026-02-21, the original JAX Dreamer trainer was retrofitted in 4 measured ph
- [[20260519_1509_nnx_lax_scan_split_merge_pattern]] (dreamer_diagnosis, 2026-05-19) — To run a flax NNX Module inside jax.lax.scan without re-tracing on every call, s
- [[20260521_0151_xla_scan_body_compile_dominates_module_count]] (dreamer_diagnosis, 2026-05-21) — Earlier claim — that dreamer-srl's 7-module decomposition CAUSED the 70× lax.sca
- [[20260529_1825_log_interval_anchored_rows_per_session]] (cluster_ops, 2026-05-29) — log_interval should be anchored to rows-per-session (target ~140 rows in a 24h r
<!-- END BACKLINKS -->
