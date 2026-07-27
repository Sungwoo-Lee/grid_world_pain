---
id: 20260513_1417_jax_vmap_no_speedup_tiny_env
date: 2026-05-13
time: "14:17"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, refutation, decision, meta]
summary: "JAX-vmap parallel env over our 5×5 NoPred gridworld delivered no speedup vs sheeprl's SyncVectorEnv, on CPU (v1: 1.01× at N=4) or GPU (v2: 0.47× at N=4). Root cause is the Python↔JAX boundary cost dominating microsecond-cheap env-step compute — naive `np.array()` round-trip plus `jax.tree.map` auto-reset blend. Closes the spike; only DLPack zero-copy bridge could plausibly win, and that is a separate fresh plan."
related: ["20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned", "20260512_1755_pytorch_agents_pip_dep_layout", "20260512_1756_pip_install_namespace_shadow_numpy_cap"]
session_origin: claude_code
session_label: "JAXVectorEnv v1+v2 spike — vmap on tiny envs doesn't deliver speedup, CPU or GPU"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/39e1dece-ad07-470f-9d4c-d659599e8d6c.jsonl
raw_completeness: full
---

# JAX-vmap on a 5×5 env doesn't beat SyncVectorEnv — naive numpy boundary kills the speedup on both CPU and GPU

## Key conclusion

For tiny envs (≤ ~10×10 grid, single-digit-microsecond per-step compute), `jax.vmap`-batched single-env-state stepping does **not** beat sheeprl's `gym.vector.SyncVectorEnv` python-loop, **on either CPU or GPU**, when the boundary between JAX outputs and the consumer (sheeprl/torch) is the naive numpy copy. The per-step Python↔JAX boundary cost (forced `np.array()` device copy plus `jax.tree.map`-blended auto-reset on done masks) is larger than the env-step compute itself, so vmap has no math to amortize. The only architecture that could plausibly win is zero-copy DLPack bridging between JAX device tensors and torch device tensors — out of spike scope; a separate fresh plan.

## Evidence, measurements, facts

- **v1 — CPU JAX**, n114 cuda:3, 1000 steps each, on the 5×5 NoPred food-only config `configs/experiment/dreamer_curriculum/01_food_only.yaml`:

| N | SyncVectorEnv SPS | CPU JAXVectorEnv SPS | speedup |
|---|---|---|---|
| 1 | 686 | 360 | 0.52× |
| 2 | 660 | 458 | 0.69× |
| 4 | 640 | 646 | 1.01× |
| 8 | 625 | 656 | 1.05× |
| 16 | 612 | 620 | 1.01× |

  v1 G4 gate (≥1.5× at N=4) FAILED. Pattern (slow at low N, parity at high N) is the canonical fingerprint of "compute negligible, boundary dominates".

- **v2 — GPU JAX**, same node/GPU/config/steps, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`:

| N | SyncVectorEnv SPS | GPU JAXVectorEnv SPS | speedup |
|---|---|---|---|
| 1 | 686 | 156 | 0.23× |
| 2 | 660 | 224 | 0.34× |
| 4 | 640 | 301 | 0.47× |
| 8 | 625 | 337 | 0.54× |
| 16 | 612 | 370 | 0.60× |

  v2 G4 gate FAILED hard. GPU JAX was *worse* than CPU JAX — host↔device CUDA round-trip per step added cost on top of the already-dominant Python↔JAX boundary.

- **WandB run IDs**:
  - v1 SyncVectorEnv baseline (original 2026-05-12 sweep): `6p386gwm` `no66f0j4` `2kkrsh1k` `xggwhch8` `naoupbs4`
  - v1 JAXVectorEnv CPU sweep: `cu7i0jly` `xgpmvi2s` `yv94yeu6` `xqpgwyl6` `a0pdc3rc`
  - v2 G2 smoke (GPU, num_envs=4, no crash): `ggifduku`
  - v2 G3 sweep: `l4iprhh5` `orwdkxzq` `tu2myidr` `e43224mf` `qla1wedj`
  - v1 G2 smoke (CPU, num_envs=4): `9l07wq7z`

- **Why vmap can't help here**: the env's `jax_step` on a 5×5 grid is a handful of integer/array ops, microseconds of compute. The boundary work per step is `np.array(jax_out)` (forced GPU→host copy on GPU, host-copy on CPU) plus a `jax.tree.map(lambda done, step, reset: jnp.where(done, reset, step), ...)` auto-reset blend over each pytree leaf. That overhead is constant-per-step regardless of N, and at low N it exceeds the entire per-instance work the SyncVectorEnv loop was doing.

- **Env-install side-effects** that persist in `sheeprl_bridge` (recipe for future revisits):
  - `nvidia-cudnn-cu12: 9.1.0.70 → 9.10.2.21` (jax-cuda12-pjrt 0.9.0.1 needs ≥9.8.x)
  - `nvidia-cuda-nvcc-cu12: (missing) → 12.9.86` (kernel compile toolchain)
  - `jax/jaxlib/jax-cuda12-{plugin,pjrt}: 0.10.0 → 0.9.0.1` (downgrade because 0.10.0's PTX 8.8 exceeds n114 driver 535's CUDA 12.2 / PTX 8.2 ceiling; same version `grid_world_pain` env already uses on n114)
  - **`torch: 2.5.0+cu121 → 2.8.0+cu128`** — major bump, training-runner did this unilaterally during pre-flight to resolve a cuDNN metadata conflict; user accepted "verify via smoke" rather than revert. Running `kfsvh1qk` NoPred replica (cuda:2, ~7h ETA from this insight) is unaffected because its python process loaded torch 2.5 into memory before the disk upgrade.

- **What stays in the repo**: `pytorch_agents/envs/jax_vector_env.py` (155 lines, retained as no-op fallback behind `env.use_jax_vector_env: false` default); `scripts/launch_sheeprl.sh` env-var changes (harmless when JAX-on-GPU is disabled, but should consider re-adding `JAX_PLATFORMS=cpu setdefault` for safety so accidental opt-in doesn't suddenly route to slow GPU JAX). v2 plan to be archived `status: superseded` by senior-developer.

- **Generalizable heuristic** for future spikes: before betting on `vmap`-driven speedup, measure per-step env-compute time. If it's < ~100 µs, the boundary cost will dominate any vmap gain — vmap is for *bulk* compute (per-step ≥ ms range or large arrays), not for cheap envs. The same heuristic explains why the 2026-05-11 SPS sweep also found no speedup at `num_envs=4` even with training active: gradient compute (the actual ms-range work) was the bottleneck, not env-stepping.

## Decisions and actions

- **STOP** the JAXVectorEnv investigation. v1 (CPU) and v2 (GPU) both fail the G4 gate; the design hypothesis (vmap parallelism wins on tiny envs with naive numpy boundary) is refuted.
- **Code stays** as `use_jax_vector_env: false` no-op fallback — useful reference for a future DLPack-shaped revisit. Default is opt-out; existing launchers see no change.
- **Plans archived** to `docs/develop/archive/sheeprl_bridge/` with `status: superseded` (v1 already archived; v2 archival queued for senior-developer).
- **Torch 2.8 upgrade** stays in `sheeprl_bridge` — user chose smoke-verification over revert. Future sheeprl launches in this env will use torch 2.8.0+cu128 instead of 2.5.0+cu121.
- **DLPack option (b)** queued as a fresh plan if/when speedup becomes a priority again — would require patching ~6–10 sheeprl call sites in `run_dreamer_v3.py` plus the replay-buffer interface to consume torch device tensors directly. Real failure risk; not a spike.

## Open questions and follow-ups

- **Torch 2.8 + sheeprl compatibility smoke** — pending. The user accepted "verify via smoke" rather than revert; a 1000-step smoke on the food-only NoPred config with `use_jax_vector_env=false` (default SyncVectorEnv path) under torch 2.8 should confirm no regression in the unmodified bridge. Hold for `kfsvh1qk` to finish first to avoid GPU contention, or use cuda:3 (free).
- **Restore `JAX_PLATFORMS=cpu` setdefault** in `pytorch_agents/envs/grid_world_pain.py` and `scripts/launch_sheeprl.sh` as a safety net? Currently removed (commit `5ed4c6f`); without it, an accidental opt-in to `use_jax_vector_env=true` would route to the slow GPU JAX path. Low priority but worth a one-line safety patch.
- **DLPack design questions** for the future revisit: do we keep sheeprl's main loop on torch and JAX-side outputs DLPack-bridged into torch tensors? Or move the full step to JAX and only DLPack-bridge the actor outputs back? Different patch sizes and different failure modes.

## References

- v1 plan (archived): `docs/develop/archive/sheeprl_bridge/JAX_VECTOR_ENV_V1_PLAN.md`
- v2 plan (active → to be archived): `docs/develop/active/sheeprl_bridge/JAX_VECTOR_ENV_V2_GPU_SPIKE.md`
- Parent SPS benchmark doc: `docs/develop/active/sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md`
- Diary day: `docs/diary/2026-05-12.md` and `docs/diary/2026-05-13.md`
- Related insight (env-install gotchas in sheeprl_bridge): `[[20260512_1756_pip_install_namespace_shadow_numpy_cap]]`
- Related insight (pytorch_agents pip-dep layout): `[[20260512_1755_pytorch_agents_pip_dep_layout]]`
- Related insight (sheeprl pivot context): `[[20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned]]`
- Commits: `d729e2c` (writable numpy copies fix), `5ed4c6f` (v2 spike — JAX on GPU), `9921006` (launcher Hydra-override pass-through), and the initial JAXVectorEnv landing commit
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 39e1dece-ad07-470f-9d4c-d659599e8d6c` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
