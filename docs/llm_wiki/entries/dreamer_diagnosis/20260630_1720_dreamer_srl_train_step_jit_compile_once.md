---
id: 20260630_1720_dreamer_srl_train_step_jit_compile_once
date: 2026-06-30
time: "17:20"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, meta]
summary: "The dreamer_srl gradient-step lax.scan was written bare in the Python loop with NO enclosing @jax.jit, so the ~76k-HLO-line train_step (WM+actor+critic+target+3 optimizers, T/H loops unrolled) recompiled from scratch EVERY iteration — the multi-hour, GPU-0%, RSS-climbing 'hang' on every real --episodes run. Fix: wrap the scan in one persistent @jax.jit built once + make n_grad_steps constant. Then it compiles once (~75s) and trains at 40-71 SPS."
related: ["20260519_1507_dreamer_srl_v2_cpu_buffer_regression", "20260519_1508_dreamer_jax_perf_retrofit_4_phases", "20260521_0151_xla_scan_body_compile_dominates_module_count", "20260622_1746_dreamer_srl_recompile_storm_done_count", "20260623_1616_rppo_reset_recompile_immune"]
session_origin: claude_code
session_label: "dreamer_srl basic training-speed hang: two-bug fix (reset storm + missing jit)"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/3c11010b-dbfb-4d17-af58-5b86e54d6815.jsonl
raw_completeness: full
---

# dreamer_srl train_step hang = gradient lax.scan with no outer @jax.jit (recompiled every iter)

## Key conclusion
Every real `--episodes` dreamer_srl training run "hung" for hours at the gradient phase: process state `R` (CPU-bound), **GPU 0%, RSS climbing linearly toward OOM, no progress**. Root cause (measured by senior-developer): the gradient-step `jax.lax.scan` was written **bare in the Python training loop with NO enclosing `@jax.jit`**. So the giant fused `train_step` program (~76,000 HLO lines: RSSM observe T=64 + imagine H=15 + actor forward H+1=16 unrolled, × world-model + actor + critic + target-Polyak + 3 Adam optimizer states) was **recompiled from scratch on essentially every training iteration**. The original working JAX Dreamer (`src/models/dreamer_v3_trainer.py:683` `_scan_train_gpu`, `@nnx.jit`) wraps its scan in exactly this outer jit — that is the structural difference. Fix: extract the scan into a function `@jax.jit`-decorated and built ONCE before the loop, and make `n_grad_steps` constant per iteration (the `Ratio` scheduler drifts 15/16/17 → each length is a new shape → fresh compile even with the jit). After the fix: compiles ONCE (~75s), then trains at 40-71 SPS (≥ the documented ~40 baseline), world-model loss converging, RSS flat.

## Evidence, measurements, facts
- Measured (CPU, this session): single `one_train_step` → 76,005 HLO lines, 63s compile. Whole 16-step gradient loop wrapped in ONE `@jax.jit` → 76,952 HLO lines (1.01×), 75.7s compile, paid once (amortization 0.07). Without the outer jit, that ~76k-line compile recurs per iteration.
- The two-phase pipeline: COLLECTION (env-step) works fine — short tests hit 344 episodes in 31s at 127-184 SPS. The hang is purely the GRADIENT phase, which only starts after the buffer fills (~9 min in, learning_starts=1024). **`grad_steps=0` short tests (`--total-steps`) pass because they never reach it** — this masked the bug for a long time.
- Diagnostic signature of a CPU-bound XLA compile (NOT a deadlock): `/proc` shows state `R`, `wchan=0`, ~470 threads, 130-145% CPU, RSS growing; GPU 0%. A futex-blocked deadlock would be state `S`. py-spy needs ptrace (sudo-blocked); faulthandler (`PYTHONFAULTHANDLER=1` + `kill -ABRT`) is the workaround but here the short run had already finished (grad_steps=0).
- Fix lives entirely in `dreamer_srl_main.py` (~:905-947 the `@jax.jit _scan_grad_steps`, ~:1449-1591 the call). agent.py untouched. Commits `0119e87` (Fix 1+2), report `aa75050` / `ecdcf0d`. 103 tests pass.
- Confirmed end-to-end: a 34.5h / 156k-episode run (`sps=40.1`, `world_model_loss` 9.7→2.5) + a fresh measurement run (`sps=71.1` on node 114, `world_model_loss` 9.68→2.3). 6-run basic sweep (L0-L5) all crossed into training within 4-8 min at 62-100% GPU.

## Decisions and actions
- **Two distinct bugs blocked basic-level dreamer training, both now fixed**: (1) the reset recompile storm — variable `len(dones_idxes)`-sized per-step reset — fixed with a fixed-width masked reset matching rPPO's reset-all-then-`where(done)` pattern [[20260622_1746_dreamer_srl_recompile_storm_done_count]] / [[20260623_1616_rppo_reset_recompile_immune]]; (2) THIS missing `@jax.jit`, which was the actual "can't train at all" blocker. The reset storm was collection-phase overhead (+37% once fixed); the missing jit was the gradient-phase hang.
- An earlier "Fix 3" (SCAN_BUCKET padding + masking the scan body) was tried, found to INFLATE the train_step compile, and dropped — it was solving a non-problem for `replay_ratio=1` (n_grad_steps already constant 16). See `docs/reviews/dreamer_srl_basic_recompile_diagnosis.md`.
- **Meta-lesson (measure, don't guess)**: this took ~5 wrong hypotheses (Fix-3 inflation, "my reset fix is the regression," config-type-dependent, node-degraded, "always-slow inherent compile") and repeated premature run-kills before the senior-developer's direct HLO/compile-time MEASUREMENT (with vs without the outer jit) found it. For dreamer GPU-compile issues: (a) check `grad_steps` and the buffer-fill phase before concluding "hang"; (b) measure HLO instruction count + compile time directly rather than inferring from symptoms; (c) don't kill runs mid-compile.

## Open questions and follow-ups
- Reaching rPPO's "compile once" fully would require folding collection + per-env autoreset into one jitted `vmap(jax_reset)`+`where` (the ~300 remaining one-time warmup compiles are the un-fused eager ops); optional, doesn't affect steady-state speed.

## References
- Diagnosis + plan: `docs/develop/active/dreamer_srl_v2/TRAIN_STEP_COMPILE_DIAGNOSIS_AND_FIX_PLAN.md`; `docs/reviews/dreamer_srl_basic_recompile_diagnosis.md`; report `docs/develop/active/dreamer_srl_v2/recompile_storm_fix.md`.
- The slow-scan-compile pathology this resolves: [[20260521_0151_xla_scan_body_compile_dominates_module_count]]; the perf-retrofit template: [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]]; CPU-buffer regression (next speed target): [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]]. Same-session reset-storm fix: [[20260622_1746_dreamer_srl_recompile_storm_done_count]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 3c11010b-dbfb-4d17-af58-5b86e54d6815` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260630_1721_per_episode_variance_dreamer_recompile_safe]] (env_entities, 2026-06-30) — The per-episode environment-variance feature (count ranges via count_high-static
<!-- END BACKLINKS -->
