---
id: 20260721_0422_batched_dreamer_eval_rng_and_unify
date: 2026-07-21
time: 04:22
folder: dreamer_diagnosis
tags: [dreamer, design, decision, learned_lesson, meta]
summary: "Unified Dreamer eval into eval_rollout.py (one script, both algorithms) and added a batched (parallel-env) Dreamer eval = 6.4x speedup. Dreamer's per-step stochastic RSSM sampling + sequential master RNG key make bit-exact batching structurally impossible (unlike rPPO's deterministic argmax), so batched uses independent per-episode keys and is validated by DISTRIBUTIONAL equivalence, not bit-exact parity."
related: ["20260721_0421_eval_sweep_cpu_bound_not_nas", "20260721_0423_batched_eval_layout_and_npar"]
session_origin: claude_code
session_label: "eval speed + batched-dreamer + size-sweep result generation (v3.0)"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# Unified + batched Dreamer eval; stochastic-RSSM forbids bit-exact batching

## Key conclusion
Dreamer (`dreamer_srl`) eval was single-env and ~10x slower than rPPO purely because the eval helper was written batch=1, not any model limit - the world model, actor, and env all already run batched in training (`Player` uses `get_initial_states(num_envs)`; `ParallelEnv` vmaps reset/step/obs). Two changes: (1) **unified** the Dreamer path INTO `scripts/eval/eval_rollout.py` (filled its Dreamer stub; one script now dispatches rPPO vs Dreamer by `--agent_config`), retiring the need for the separate `dreamer_srl_probe_eval.py` driver; (2) added a **batched** Dreamer eval (`dreamer_srl_eval_rollout_batched`, nnx.jit lax.scan over max_steps + per-episode death-step slicing) = **6.4x** faster (123 s -> 19 s at N=30 on an idle node). The deep finding: **Dreamer's RSSM draws a random latent sample every step**, and the legacy single-env eval threads ONE master RNG key sequentially across all episodes+steps, so episode k's start randomness depends on how many steps episode k-1 took. That makes bit-exact batching **structurally impossible** for N>1 (vmap needs each episode's RNG stream known up front). Contrast rPPO, whose eval is pure deterministic argmax and IS bit-exact-batchable. So batched Dreamer uses **independent per-episode keys** and is validated by **distributional equivalence** (aggregate measures within Monte Carlo noise), plus a deterministic same-keys cross-check that the vmap/scan machinery is bit-exact.

## Evidence, measurements, facts
- Speedup (idle node 103, N=30, same checkpoint): single-env 123 s -> batched 19 s = **6.4x**. Under contention (node 102) the ratio was ~7.6x (inflated single-env baseline). Pure-rollout speedup ~18x; end-to-end capped at 6.4x by the ~13 s per-process JAX-import+compile fixed cost.
- Correctness gates: (a) machinery deterministic cross-check (identical fixed keys -> bit-identical actions) MUST pass; (b) aggregate measures single-env vs batched agree within |z|<2 at N=60 (same distribution, different draws); (c) 11/11 legacy + 5/5 new batched pytests pass.
- rPPO's batched path (`_run_episodes_batched` + `_rollout_scan_jit`) is the template - it already solved variable-episode-length via `T = np.argmax(done_seq, axis=0) + 1` then per-env slicing, and documents that the scan MUST be wrapped in nnx.jit (bare lax.scan reads a stale nnx graphdef/state split and diverges after step 0).
- Design "Option A" (purely additive): the single-env `dreamer_srl_eval_rollout` is UNTOUCHED (0 lines removed) because it is on the TRAINING path (`dreamer_srl_main.py` calls it for periodic in-training eval) and live size-sweep runs were training; shared agent/wrapper/core are only CALLED, never modified. `--batched` on Dreamer falls back to single-env with a warning if it cannot apply.
- Commits: unification 766938d; batched c023145.

## Decisions and actions
- Chose Option A (add a batched sibling; never modify the training-path function) after confirming rPPO did exactly this (kept single-env default, added `--batched` sibling with a bit-parity contract).
- Redefined the acceptance gate from bit-exact to distributional-within-MC-noise once the stochastic-RSSM RNG structure was found - bit-exact is the right bar only for deterministic-argmax rPPO.
- Follow-up (deferred until the running slow sweep cleared): point the production Dreamer worker at `eval_rollout.py --batched`; retire `dreamer_srl_probe_eval.py` to a shim; promote the tmp orchestration scripts into `scripts/`.

## Open questions and follow-ups
- Future full-curve regenerations should use batched THROUGHOUT to avoid mixing RNG schemes (single-env legacy points + batched per-episode-key points are same-distribution but different draws).
- The production wiring + driver-to-shim retirement is not yet committed.

## References
- `scripts/eval/eval_rollout.py`, `src/algorithms/dreamer_srl/eval.py`, `tests/algorithms/dreamer_srl/test_eval_rollout_batched.py`, `docs/develop/active/refactors/EVAL_ROLLOUT_BATCHING_PERF.md`.
- Batched-eval operational gotchas (output layout + NPAR) captured separately in [[20260721_0423_batched_eval_layout_and_npar]].
- CPU-bound eval fix (compile cache + thread cap) in [[20260721_0421_eval_sweep_cpu_bound_not_nas]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read: `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` or `python scripts/claude/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl /tmp/20260721_0422.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260721_0421_eval_sweep_cpu_bound_not_nas]] (cluster_ops, 2026-07-21) — The frozen-checkpoint eval sweep is CPU-bound (multithreaded XLA compile), NOT N
- [[20260721_0423_batched_eval_layout_and_npar]] (behavior_measures, 2026-07-21) — Two operational gotchas when driving Dreamer probe sweeps through eval_rollout.p
<!-- END BACKLINKS -->
