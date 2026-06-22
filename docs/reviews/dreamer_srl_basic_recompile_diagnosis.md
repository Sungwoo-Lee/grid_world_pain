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

## Post-fix correctness review

### Plain-language verdict

The recompile-storm fix was reviewed not for "does the storm go away" (the user already
confirmed empirically that it does) but for "does the fix still train the same agent."
The answer is **yes, with one cosmetic-but-worth-noting caveat about random-number
bookkeeping that does not fire in any of the configs actually being run.**

**Verdict: APPROVE-WITH-NITS.**

The three fixes preserve training semantics. Fix 1 (the masked reset of the agent's
memory state when an environment dies) is exactly equivalent to the old code — living
environments keep their memory untouched, dead ones get the correct fresh-start memory.
Fix 3 (running a fixed number of gradient-step slots and "masking off" the extra ones)
correctly nulls every extra slot's effect on the network weights, the optimizer, the
slow-moving target network, the running normalizer, and the logged numbers — extra slots
are genuine no-ops, not just "ignored in logging." The one nit: the extra slots still
*consume random keys* even though they do nothing else, so the random stream is advanced
slightly further than the old code would advance it. This only matters when the number of
real gradient steps per iteration is less than the fixed slot count, which **never happens
for any current config** (all use replay-ratio 1 with 16 environments, so real-steps =
slot-count = 16 exactly, every iteration). It would only surface with a fractional
replay-ratio, and even then it does not corrupt training — it just breaks bit-for-bit
parity with the old `--legacy-grad-loop` path. No code changes are required to ship the
launched runs.

### Risk 1 — Masked reset bit-for-bit equivalence (Fix 1): PASS

`Player.init_states(done_mask=...)`, `dreamer_srl_main.py:236-256`.

- **(a) Living envs unchanged — PASS.** For a living slot `mask[i]=0`, the select is
  `(1.0 - 0.0)*state + 0.0*h0 = state`. Multiply-by-1.0 and add-0.0 are exact in IEEE
  fp32 (no rounding, no drift). Recurrent (`:246`), posterior (`:250`) and prev-action
  (`:255`) all use this idiom. No accidental reset.
- **(b) Dead envs get the true initial state — PASS.** For a done slot `mask[i]=1`, the
  select is `h0_full[i]`. `get_initial_states(num_envs)` (`agent.py:931-969`) returns the
  deterministic mode-based `(h0, z0)` — **no PRNG consumed** (CP4 mandate, `agent.py:945`),
  so `h0_full[i]` is identical to what the old `get_initial_states(n_done)` scatter
  produced for that slot. The fix changes the *batch size* the function is called with
  (16 vs n_done), not the per-row value — and the per-row value is independent of batch
  size (a tiled constant + a deterministic transition). Bit-identical.
- **(c) Slot alignment — PASS.** `_done_mask = dones.astype(np.float32)`
  (`dreamer_srl_main.py:1130`), shape `[num_envs]`, same `dones` array indexed
  env-wise everywhere else in the loop (episode bookkeeping `:1139`, autoreset `:1161`).
  No transpose, no off-by-one. `mask[:,None]` / `mask[:,None,None]` broadcast against
  `[B,recurrent]` and `[B,S,D]` on the leading (env) axis — correct.
- **(d) Recurrent AND posterior masked consistently — PASS.** Both use the same `mask`
  with shape-appropriate broadcasts; prev-action is masked the same way. The Player stores
  the posterior in *unflattened* `[B,S,D]` form (`dreamer_srl_main.py:316,325`), and
  `z0_full` is `[B,S,D]` (`agent.py:957`), so `m_z = mask[:,None,None]` is the right rank.
  None scattered while the other is masked.

### Risk 2 — Surplus gradient-step masking (Fix 3): PASS on the dangerous channels, ONE nit

`_scan_body`, `dreamer_srl_main.py:1528-1631`; `is_real = scan_step_local < n_real_steps`
(`:1601`).

The high-risk question — does a padding (surplus) scan iteration silently corrupt
training? — is **answered cleanly NO for every state-bearing channel:**

| Channel | Masked? | Line | Padding effect |
|---|---|---|---|
| World-model params/state | `jnp.where(is_real, new, old)` | 1603 | none (old kept) |
| Actor params/state | same | 1604 | none |
| Critic params/state | same | 1605 | none |
| Target-critic (EMA) | same, over `s_tg_new` | 1606 | none |
| WM / actor / critic **optimizer** state (Adam moments) | same | 1607-1609 | none |
| Moments normalizer (§S7) | same | 1610 | none |
| `step_idx` (Polyak schedule counter) | `+ where(is_real,1,0)` | 1614 | not advanced |
| Logged losses | `where(is_real, v, 0)` | 1612 | zeroed; `last_losses` taken from `losses_stack[_n_real-1]` (`:1654-1655`), a real row |
| `cumulative_grad_steps` | `+= _n_real` (Python) | 1658 | counts real steps only |

Padding rows feed the last *real* data row (`:1517-1523`, tiled), so even the
forward/backward pass inside a padding step runs on valid (not garbage/NaN) data — and
its result is then discarded by the masks above. The target-critic Polyak update is
double-protected: even if `do_update` fires on a padding step (because `step_idx` is
frozen at a multiple of `target_update_freq`), its result is masked out by `s_tg_out`
(`:1606`). Replay-buffer pointers are untouched by the scan entirely (buffer is sampled
once before the scan, `:1391-1411`; `cumulative_grad_steps` advances by `_n_real` only).

**The one nit — PRNG over-advancement on padding steps (🟡 concern, dormant):**
`carry_key, k_train = jax.random.split(carry_key)` (`:1589`) runs on *every* scan
iteration including padding, and `carry_key` is propagated **unmasked** into the carry
(`:1626`). So when `_n_real < _SCAN_BUCKET`, the `key` returned to the outer loop
(`:1641`) has been split `_SCAN_BUCKET` times instead of `_n_real` times.

- **Does this corrupt the real gradient steps?** No. Real steps occupy scan indices
  `0.._n_real-1` and consume their keys *before* any padding step runs on the same chain,
  so the random numbers the real `train_step`s see are unaffected by padding.
- **Does it break determinism?** No. The advancement is a fixed function of `_SCAN_BUCKET`
  (a startup constant), so same seed + same config still reproduces the same run.
- **What it does break:** bit-for-bit parity with the `--legacy-grad-loop` reference,
  which splits the key `n_grad_steps` times — when padding fires, downstream draws (next
  iteration's actions, autoreset keys, buffer sampling) diverge from the legacy path.
  `tests/algorithms/dreamer_srl/test_grad_parity.py` is in the ignored-test list, so this
  divergence is untested.
- **When does padding actually fire?** Only when `n_grad_steps < _SCAN_BUCKET`, i.e. when
  `replay_ratio * num_envs` is non-integer (e.g. ratio 0.3). **Every shipped dreamer_srl
  config uses `replay_ratio: 1`** (`configs/models/dreamer_srl/*.yaml`), so with 16 envs
  `_SCAN_BUCKET = 16` and steady-state `n_grad_steps = int(16*1.0) = 16` — `_n_real ==
  _SCAN_BUCKET` every iteration, **padding never fires**, and the key advances by exactly
  16 splits as the legacy path would. The first gated iteration (`Ratio._prev is None`
  returns a large `int(ratio_steps)`) overflows the bucket and is handled by the legacy
  fallback loop (`:1662-1681`) with `_n_real = 16` — still no padding.

**Recommendation (only if a fractional replay-ratio is ever used — hand to `developer`):**
mask the key like every other carry channel so the legacy and scan paths stay
bit-identical. Inside `_scan_body`, replace the unmasked `carry_key` at `:1626` with
`jnp.where(is_real, carry_key, prev_carry_key)` (capturing the pre-split key), OR — simpler
and clearer — make `train_step`'s key consumption itself conditional. Not required for the
current `replay_ratio=1` runs; do this before any fractional-ratio sweep.

### Risk 3 — New recompile trigger / shape bug introduced: PASS

- `_SCAN_BUCKET = max(1, math.ceil(replay_ratio * num_envs))` (`:668`) is a Python int
  computed once from two startup constants — it does not vary, so the scan leading dim is
  constant (confirmed empirically: single `float32[16,...]` scan compile).
- `scan_xs` is always built at width `_SCAN_BUCKET` — the `_n_real < bucket` branch pads
  by tiling (`:1517-1523`), the `else` branch slices `[:_SCAN_BUCKET]` (`:1525`). Constant
  shape either way; the mask (`is_real`) is a scalar comparison, not a variable-width
  array.
- Edge cases checked: the scan block only runs when `n_grad_steps > 0` (`:1391`), so
  `_n_real >= 1` and `_n_real-1 >= 0` — the filler-row slice (`:1519`) and the
  last-real-loss index (`:1654`) are always in-bounds. Overflow slicing
  (`[:_SCAN_BUCKET]` scan + `[_n_real:]` legacy, `:1525/1663`) partitions the rows with no
  overlap or gap.
- Fix 2 (`jax.random.split(k_autoreset, num_envs)`, `:1160`, indexed by `env_idx`) is
  constant-width and correct — each done env reads its own row `reset_keys[env_idx]`,
  preserving per-env key independence.

### Post-fix conventions audit
pytree ✅ · JIT ✅ (all three variable-shape leading dims now constant-width) ·
vmap ✅ · PRNG ⚠️ (sound + deterministic; non-bit-identical to legacy path only under a
fractional replay-ratio, which no current config uses) · sensor sync ✅ · config protocol ✅.

**Conclusion: APPROVE-WITH-NITS — safe to ship the launched `replay_ratio=1` runs as-is;
mask `carry_key` (Fix 3) before any fractional-replay-ratio sweep.**

*Post-fix review by: code-reviewer*

---
*Reviewed by code-reviewer (reproduced with `JAX_LOG_COMPILES=1`). Source runs:
`logs/20260620_16470{9}.log`, `…1647{10,11,12}.log`; WandB
zcg33koq/1sphik88/yvyx8x42/2uc2yg7f/c9lelny1.*
