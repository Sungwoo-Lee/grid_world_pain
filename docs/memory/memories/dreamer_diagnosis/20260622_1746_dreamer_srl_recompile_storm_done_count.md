---
id: 20260622_1746_dreamer_srl_recompile_storm_done_count
date: 2026-06-22
time: "17:46"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, meta]
summary: "dreamer_srl crashed all 5 basic-curriculum runs because the per-step env-reset path sizes arrays to the variable done-env count len(dones_idxes), a faithful PyTorch idiom that is a JAX trap: each distinct width compiles a new XLA executable, the executables/CUDA-graphs accumulate, and the GPU OOMs (24GB) / stalls in endless compiles (49GB). Fixed with a masked fixed-width reset (always num_envs, boolean done mask) + a fixed scan bucket."
related: ["20260529_1826_lazy_import_schema_drift_first_call_crash", "20260622_1747_dreamer_srl_single_config_budget_source"]
session_origin: claude_code
session_label: "dreamer_srl basic-curriculum: 5-run crash → recompile-storm diagnosis + fix"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/3c11010b-dbfb-4d17-af58-5b86e54d6815.jsonl
raw_completeness: full
---

# dreamer_srl recompile storm: variable done-count is a PyTorch→JAX port trap

## Key conclusion
Five `dreamer_srl` runs on the basic difficulty curriculum (L0–L4, one per level, single-config `--env-config` mode) all crashed within ~1–1.5 days with **zero checkpoints and essentially no training** (the node-113 runs advanced only ~1,800 loop iterations in 35 h). Root cause: the per-step environment-reset path sizes several arrays to `len(dones_idxes)` — the number of parallel envs that finished **this** step — which on the basic levels varies 1…16 every step because a lethal hiding predator from step 0 makes the 16 envs die at random, **desynchronized** times. Variable leading dims are free in eager PyTorch (the code's origin: a faithful port of sheeprl's "reset only the done envs" idiom) but are a **trap in JAX**: each distinct width forces a fresh XLA executable, the executables/CUDA-graphs accumulate ("alive graphs" 9→27 in the logs), and the GPU either OOMs (24 GB node-113) or stalls in endless 15-min compiles until the node falls over (49 GB node-114). The 3-stage curriculum was immune only because its benign Stage 1 lets all 16 envs end together at the 500-step cap → constant count 16 → one compile. **General rule: any PyTorch→JAX port must convert variable-length "operate on the subset that changed" idioms to fixed-width masked operations.**

## Evidence, measurements, facts
- Failure signature: `RESOURCE_EXHAUSTED: ... instantiate command buffer ... (total of 9 / 27 alive graphs)` on 24 GB GPUs; repeated `[Compiling module jit_scan for GPU] Very slow compile?` + a `slow_operation_alarm took 15m3s` on 49 GB GPUs; node 114 went SSH-unreachable (needed manual restart).
- Reproduced with `JAX_LOG_COMPILES=1` on `02-fast_predator_8x8`: the bug emits reset-path compiles at widths `[1,256],[2,256],[3,256],[4,256],…,[16,256]` and `[N,32,32]`.
- The varying input is `len(dones_idxes)` (`dreamer_srl_main.py:1003`), driving: (1) `player.init_states(reset_envs=dones_idxes)` → `rssm.get_initial_states(n_done)` scatter (`agent.py` ~:212-244, :931); (2) `jax.random.split(k_autoreset, len(dones_idxes))` + the buffer reset row (~:1055-1064, :1092); (3) secondarily `n_grad_steps` as the `lax.scan` leading dim (`:1321`, scan at `:1518`) — the `jit_scan` 15-min compiles.
- `ParallelEnv` (`wrapper.py`) is vmap-only with NO outer `jax.jit`, so nothing pins the shapes; the OOM that surfaced *inside* `jax_reset` (`:1098`) was incidental (next allocation on an already-saturated GPU). `jax_step`/`jax_reset` themselves are correctly jitted, fixed-shape.
- **Fix (3 parts, all in the hot loop):** (1) masked fixed-width recurrent/posterior reset — always `get_initial_states(num_envs)`, select per slot with `(1-m)*state + m*h0_full`, the same arithmetic-mask idiom already at `agent.py:993-1004`; (2) build the autoreset key split + reset_data at width `num_envs`; (3) pad `n_grad_steps` to a fixed `SCAN_BUCKET` and mask surplus scan iterations.
- **Post-fix verification (independent):** `JAX_LOG_COMPILES=1` on the same config shows the intermediate reset widths `[2..15,256]` are GONE (only `[1,256]` single-step + `[16,256]` fixed reset remain); the compile set saturates (last 40 compile events introduce zero new functions); the run completes its budget with no OOM. code-reviewer verdict APPROVE-WITH-NITS: masked reset is bit-for-bit equivalent (living envs untouched); Fix-3 padding contributes exactly zero to params/optimizer/target-EMA/moments/buffer; one dormant PRNG nit that only fires under a fractional `replay_ratio` (all shipped configs use `replay_ratio=1`, 16 envs → padding never fires).

## Decisions and actions
- Fix implemented in `src/algorithms/dreamer_srl/dreamer_srl_main.py` + `agent.py`, committed; diagnosis at `docs/reviews/dreamer_srl_basic_recompile_diagnosis.md`, implementation report at `docs/develop/active/dreamer_srl_v2/recompile_storm_fix.md`.
- The 5 basic runs were re-launched on the fixed code (113:0,1 + 114:0,1,2, seed 42, num_envs 16); they entered a one-time ~15-20 min GPU first-compile (single compile of this model is genuinely that slow — the bug was REPEATING it).
- Follow-up handed to `developer` only if a fractional replay-ratio is ever used: mask `carry_key` like the other scan-carry channels.
- This is the same failure CLASS as other "latent bug surfaces only under a specific runtime condition" dreamer findings — see [[20260529_1826_lazy_import_schema_drift_first_call_crash]].

## Open questions and follow-ups
- The dormant PRNG nit (unmasked `carry_key` on padding iterations) is untested because `test_grad_parity.py` is in the ignored-test list; would matter only under fractional `replay_ratio`.
- Whether the basic runs train healthily to convergence on the fixed code is still pending (they were mid first-compile at capture time).

## References
- Diagnosis: `docs/reviews/dreamer_srl_basic_recompile_diagnosis.md`; impl report: `docs/develop/active/dreamer_srl_v2/recompile_storm_fix.md`.
- Launch/budget gotcha from the same session: [[20260622_1747_dreamer_srl_single_config_budget_source]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 3c11010b-dbfb-4d17-af58-5b86e54d6815` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260622_1747_dreamer_srl_single_config_budget_source]] (cluster_ops, 2026-06-22) — In dreamer_srl single-config (--env-config) mode the training budget is read fro
- [[20260623_1616_rppo_reset_recompile_immune]] (env_entities, 2026-06-23) — rPPO (recurrent_ppo) is recompile-IMMUNE to the new per-episode count-variance: 
<!-- END BACKLINKS -->
