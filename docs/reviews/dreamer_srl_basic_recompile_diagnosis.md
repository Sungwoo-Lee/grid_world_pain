# Diagnosis — dreamer_srl recompilation storm on the basic curriculum

## What this is about (plain language)

Five `dreamer_srl` (in-house DreamerV3) training runs — one per "basic" difficulty
level (L0–L4) — were launched on 2026-06-20 and **all crashed within ~1–1.5 days
without producing a single checkpoint**. They were not training; they were thrashing
on compilation (the node-113 runs advanced only ~1,800 internal iterations in 35 hours).
This doc records the root cause, why the earlier 3-stage curriculum never hit it, the
recommended fix, and the status of a launch-time stopgap.

**Headline:** the training loop re-compiles XLA code **every time the number of parallel
environments that just finished an episode changes from one step to the next.** On the
basic levels a lethal hiding predator is present from step 0, so the 16 envs die at
random, desynchronized times and the per-step done-count cycles 1, 2, … 16 endlessly.
Each distinct count is a new array shape flowing through code that is **not wrapped in a
single compiled function**, so XLA builds a fresh executable for it and keeps the old
ones alive. Those accumulate on the GPU ("alive graphs" 9 → 27 in the logs) until it
OOMs (24 GB node-113) or stalls in endless 15-min compiles and takes the node down
(49 GB node-114).

## Root cause (specific)

The varying input is `len(dones_idxes)` — the count of done envs this step
(`dreamer_srl_main.py:1003`), a Python int that changes every iteration on the basic
configs. It drives three un-jitted, variable-shape array paths, each re-compiled per
distinct value:

1. **Primary: player recurrent-state reset.** `player.init_states(reset_envs=dones_idxes)`
   (`dreamer_srl_main.py:1067`) → `rssm.get_initial_states(n_done)` (`agent.py:931`,
   called at `:226-227`) builds `[n_done,256]` / `[n_done,32,32]` arrays and scatters
   with a variable-length `idx` (`dreamer_srl_main.py:238-241`). Reproduced recompiles:
   `[1,256],[2,256],[3,256],[4,256],[16,256]` and `[1,32,32]…[16,32,32]`.
2. **Done-key split + buffer reset row** (`dreamer_srl_main.py:1055-1064, 1092`): arrays
   of leading width `R = len(dones_idxes)` → recompiles of `_threefry_split`,
   `broadcast_in_dim`, etc.
3. **Secondary: the gradient-step scan.** `n_grad_steps = ratio(ratio_steps)`
   (`dreamer_srl_main.py:1321`) is the leading `lax.scan` dim at `:1518`. When it varies,
   `jit_scan` recompiles — the literal `Compiling module jit_scan … Very slow compile?`
   and the 15-minute `slow_operation_alarm` on node 114.

Shared mechanism: `ParallelEnv` (`wrapper.py`) is vmap-only with **no outer `jax.jit`**,
and the done-handling block is plain Python orchestration. With no enclosing jit to pin
shapes, every distinct done-count / grad-step count is a new XLA program that JAX caches
(keeps alive). The OOM that surfaced *inside* `jax_reset` (`:1098`) on node 113 is
incidental — `jax_reset`/`jax_step` are correctly jitted and fixed-shape (`core.py:782`);
that line is just where the next allocation landed once the GPU was already saturated.

## Why the 3-stage curriculum was immune

Curriculum Stage 1 is a benign 5×5 world the agent quickly learns to survive; episodes
end predominantly at the shared `max_steps=500` truncation, so all 16 envs finish on the
**same** step → `len(dones_idxes)` is a constant (16, or 0) → one shape → one compile →
no accumulation. The basic configs invert this (lethal hiding predator from step 0 on a
tiny grid → desynchronized deaths → done-count visits every value 1…16 continuously).

## Fix (hand-off: developer)

Make the done-handling shapes **static (always width `num_envs`)** so XLA compiles once:

- Replace the variable-length `player.init_states(reset_envs=dones_idxes)` scatter with a
  **fixed-shape masked reset**: always reset over all `num_envs`, selecting with a boolean
  `done` mask (`jnp.where(done_mask[:,None], h0_full, h)`), so `get_initial_states(num_envs)`
  gets a constant batch and `.at[idx].set` becomes a mask-select. This is the same
  `(1-is_first)*x + is_first*init` idiom already used at `agent.py:993-1004`.
- Build `reset_data` and the autoreset keys at fixed width `num_envs` (mask unused rows).
- Secondary: **pad `n_grad_steps` to a fixed bucket** (scan a constant length, mask the
  surplus) so `jit_scan` compiles once.

Cleanest structural option: fold env-step + auto-reset into a single jitted auto-reset
step (sketched but unused at `wrapper.py:35-61`). The masked-fixed-width change is the
minimal correctness fix and needs no loop rearchitecture.

## On the `XLA_FLAGS=--xla_gpu_enable_command_buffer=` stopgap

Partial mask, not a fix. Disabling command buffers removes the per-executable CUDA-graph
allocation (the `instantiate command buffer … alive graphs` OOM vector) and lets runs
limp further, but the underlying unbounded distinct-executable compilation remains —
continuous compile latency and slower executable-memory leak persist. Emergency stopgap
only; must not substitute for the fix.

## Out of scope (recorded)

Node 114 went down (SSH port 1800 refused) during these runs and needs a **manual
restart** — infrastructure, not part of the recompilation root cause.

## Conventions audit
pytree ✅ · JIT ❌ (variable-shape leading dims `len(dones_idxes)`/`n_grad_steps`;
un-jitted env wrapper) · vmap ✅ · PRNG ✅ · sensor sync ✅ (obs constant 27-dim across
all basic levels) · config protocol ✅.

---
*Reviewed by code-reviewer (reproduced with `JAX_LOG_COMPILES=1`). Source runs:
`logs/20260620_16470{9}.log`, `…1647{10,11,12}.log`; WandB
zcg33koq/1sphik88/yvyx8x42/2uc2yg7f/c9lelny1.*
