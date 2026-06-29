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

---

## Fix 3 compile-cost investigation

### Plain-language headline

The recompile-storm fix shipped three changes. Two of them (Fix 1 and Fix 2) fixed
real bugs — they made array shapes that genuinely changed size every step into
fixed-size arrays, which is what stopped the storm. The third change ("Fix 3") solved
a problem that **did not exist for the runs we are actually launching**, and in doing so
it made the one big compile those runs need **measurably slower**. The five basic runs
that have now been "compiling at 0% GPU for 60+ minutes" are almost certainly paying that
inflated compile cost.

Here is the chain in plain terms. The training loop does a batch of gradient updates each
iteration. The number of updates per iteration is set by the "replay ratio." For every
config we ship, that ratio is 1 and we run 16 parallel environments, which works out to
**exactly 16 gradient updates every single iteration — a constant, never varying.** Fix 3
was built to handle the case where that number *wobbles* (it pads the batch up to a fixed
size of 16 and then "masks off" the padding so the extra slots do nothing). But since the
number is already a rock-solid 16, there is never any padding to mask. The masking
machinery still gets compiled, though — XLA cannot delete it, because whether a slot is
"real" or "padding" is decided by a runtime number, not a compile-time constant. So we pay
the full compile price for machinery that, at run time, is a no-op on every step.

**Verdict: Fix 3 inflated the gradient-step compile and is NOT needed for any
`replay_ratio=1` config. Fix 1 is the load-bearing fix and must stay. The leanest correct
fix is to take the original un-padded gradient-step loop whenever the per-iteration update
count is provably constant (which is every shipped config), and only use Fix 3's
pad-and-mask machinery under a fractional replay ratio.**

### What I measured (this dev node, CPU — instruction count is backend-independent)

I built the real dreamer_srl XS agent (the architecture all five basic runs use) plus its
three optimizers, split them into the exact `(graphdef, state)` pytrees the scan carry
holds, and lowered two versions of the gradient-step scan to HLO: the **original
un-masked** body and the **Fix-3 masked** body. The model's forward/backward math was
replaced by a cheap arithmetic stand-in so the measured difference is *purely* the masking
overhead Fix 3 adds, not the model's own cost. Probe scripts:
`tmp/20260622_fix3_compile_probe.py`, `tmp/20260622_fix3_foldcheck.py`.

| Quantity | Original (un-masked) | Fix-3 (masked) | Inflation |
|---|---|---|---|
| Combined trainer-state leaf count | 146 leaves (140 float) | — | — |
| HLO op-lines in scan body | 1592 | 1763 | **+10.7%** |
| `stablehlo.select` ops emitted | 0 | 20 | **+20** (one per float-leaf group) |
| Compiled-executable flops | 3.96e7 | 4.96e7 | **+25%** |
| Lower+compile wall time (CPU) | 0.61 s | 0.74 s | **+21%** |

These deltas are a **lower bound**. The stand-in train_step is tiny; the real Dreamer
train_step body (world model + actor + critic + target Polyak + three Adam optimizer
states, all unrolled inside the scan) is orders of magnitude larger, and per the memory
insight [[20260521_0151_xla_scan_body_compile_dominates_module_count]] the XLA-GPU compile
time scales with that body's **instruction count and pytree size**. Fix 3 wraps a
`jnp.where(is_real, new, old)` around **every one of the 140 float leaves** of the combined
trainer state — i.e. it adds a full-size `select` over each parameter / optimizer-moment
array, and forces XLA to keep both the pre-update and post-update copy of every parameter
buffer live simultaneously at the select point. That roughly doubles live-buffer pressure
inside the body and enlarges the scheduling/optimization search space that dominates
GPU-side compile *time* — which is exactly the 60-minute, 0%-GPU, stable-RSS symptom.

### Refuting the "XLA folds it away" claim (the load-bearing error)

Both `recompile_storm_fix.md` (lines 98–100) and the Risk-2 review above asserted that
"when `_n_real == _SCAN_BUCKET` … `is_real` is always True and XLA may constant-fold these
selects — no runtime overhead in the steady state." **This is false, and it is the crux of
the regression.** XLA constant-folds on *compile-time* constants, not on values that merely
happen to be true at run time. In the shipped code, `n_real_steps` enters the carry as
`_n_real_jax = jnp.array(_n_real, dtype=jnp.int32)` (`dreamer_srl_main.py:1503,1510`) — a
**traced device array** — so `is_real = scan_step_local < n_real_steps`
(`:1601`) is a traced predicate of unknown value at compile time. XLA must emit every
select. Direct test (`tmp/20260622_fix3_foldcheck.py`):

```
TRACED  n_real (real Fix-3 code): selects = 20   op-lines = 1763
STATIC  n_real (lean fix)       : selects =  0   op-lines = 1592
=> masking NOT foldable; the lean path removes 171 op-lines and all 20 selects
```

The selects vanish only when the predicate is a Python-level constant (i.e. when the code
*structurally* takes an un-masked path), never from XLA folding a runtime-true value.

### Was Fix 3 even targeting the right thing? (No, for these configs)

The diagnosis named the gradient-step scan as the *secondary* recompile source (line
37–40). I re-checked whether `n_grad_steps` actually varies for the launched configs by
replaying the real `Ratio` scheduler with the production parameters (`replay_ratio=1`,
`num_envs=16`, `learning_starts=1024`, `ratio_steps = policy_step - learning_starts*num_envs`):

```
distinct n_grad_steps values over 2000 iterations (when >0): {16: 1999}
```

**`n_grad_steps` is a constant 16 on every single training iteration** — there is not even
a startup catchup burst (the first positive `ratio_steps` is small because `Ratio._prev`
is seeded to it). Therefore the *original un-masked scan* would have compiled **exactly
once** (`scan_xs` leading dim = 16, invariant) for these configs. The scan was never a
storm contributor here; at worst it was a single one-time large compile. Fix 3's bucketing
solved a non-problem for `replay_ratio=1` **and** inflated the one compile those runs do
need. (Fractional replay ratios — e.g. 0.3 — *would* make `n_grad_steps` wobble and *do*
need bucketing, but no shipped config uses one.)

By contrast, Fix 1's target is genuinely variable: `len(dones_idxes)` cycles 1…16 every
step as predators kill envs at desynchronized times. That is the real storm and Fix 1 (the
masked fixed-width agent-memory reset) is the load-bearing fix that must remain. Fix 2
(autoreset key split widened to `num_envs`) likewise fixes a genuinely variable width and
stays.

### Leanest correct fix (hand-off: developer)

Gate the scan path on whether the per-iteration gradient-step count is provably constant.
The candidate in the prompt is correct; the precise change:

1. **At startup**, alongside `_SCAN_BUCKET` (`dreamer_srl_main.py:668`), compute a Python
   bool: `_grad_steps_constant = (replay_ratio == 1.0)` — or, more robustly, derive it from
   whether `replay_ratio * num_envs` is a positive integer, since that is the exact
   condition under which `Ratio(replay_ratio)` returns a constant `int(num_envs *
   replay_ratio)` every steady-state iteration. (`replay_ratio == 1` is the only shipped
   case and is sufficient; the integer test is the general guard.)
2. **In the scan branch** (the `else` at `:1465`), when `_grad_steps_constant` is True and
   `n_grad_steps == _SCAN_BUCKET` (no overflow), take an **un-masked** scan body: the
   original `_scan_body` form **without** the `scan_step_local` / `n_real_steps` carry
   fields and **without** the ten `jnp.where(is_real, …)` wraps (`:1601-1614`). Return the
   post-`train_step` states directly (the NO-MASK body measured above). This restores the
   original compile cost — 0 selects, 171 fewer op-lines, ~25% less flops in the body.
3. **Keep the existing pad-and-mask `_scan_body` for the fractional-ratio path only** — when
   `_grad_steps_constant` is False, `n_grad_steps` can wobble and the bucket+mask is
   required (and the dormant `carry_key` over-advance nit from the Risk-2 review still
   applies there and should be fixed before any fractional-ratio sweep).

Simplest possible implementation: keep one `_scan_body` but make the masking
*structurally* conditional on the Python bool `_grad_steps_constant` — when True, skip
building the `is_real` predicate and the `jnp.where` wraps entirely (return `new` states
straight), and drop the two extra carry ints. Because the branch is a Python `if` evaluated
at trace time, the un-masked body is what gets compiled — no select ops emitted. This is a
~30-line localized change in the single `else` block; no change to Fix 1, Fix 2, the legacy
loop, the overflow fallback, or `agent.py`.

**Do NOT touch the five running processes on nodes 113/114** — this fix is for the *next*
launch. Whether to kill-and-relaunch the current five (which appear stuck in the inflated
compile) vs. let them finish the one-time compile is the user's call; if a 60-minute
compile is the only cost and they then train, they may simply be slow-but-correct. The
lean fix removes the inflation for all future launches.

### Fix-3 investigation conventions audit
pytree ✅ · JIT ⚠️ (masking ops compiled-but-runtime-noop for `replay_ratio=1`; not a
*recompile* trigger — `_SCAN_BUCKET` is constant — but a compile-*cost* inflation) ·
vmap ✅ · PRNG ✅ (no change) · sensor sync ✅ · config protocol ✅.

**Conclusion: Fix 3 inflated the gradient-step compile (+~11% HLO op-lines / +25% flops /
+21% compile time as a lower bound, larger in the real body) and is unnecessary for every
`replay_ratio=1` config; Fix 1 is load-bearing and stays. Take the original un-masked scan
when the per-iteration update count is provably constant; reserve pad-and-mask for
fractional replay ratios. Hand to `developer`.**

*Fix-3 compile-cost investigation by: code-reviewer (measured on dev node CPU via
`tmp/20260622_fix3_compile_probe.py` + `tmp/20260622_fix3_foldcheck.py`).*
