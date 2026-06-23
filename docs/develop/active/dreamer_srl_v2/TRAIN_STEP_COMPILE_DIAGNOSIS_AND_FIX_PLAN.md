---
title: "dreamer-srl train_step compile pathology — diagnosis + prioritized fix plan"
topic: dreamer
status: active
created: 2026-06-24
last_updated: 2026-06-24
---

# dreamer-srl train_step compile pathology — diagnosis + prioritized fix plan

## Plain-language purpose

Every real `dreamer_srl` training run (our in-house DreamerV3) appears to **hang for hours**:
the process is busy on the CPU, the GPU sits at 0%, memory creeps upward toward an out-of-memory
crash, and **zero training progress** is logged. Short test runs that stop before any learning
happens finish fine. So the freeze is specifically in the **gradient-update phase** — the part
that runs once the replay buffer has filled (about 9 minutes in).

This document pins down **why** that phase hangs, backs the explanation with fresh measurements,
and hands the `developer` agent a ranked list of fixes. The headline:

- **Root cause (confirmed): the gradient loop is a `jax.lax.scan` that is NOT wrapped in an outer
  compiled function (`@jax.jit`).** Because of that, the giant fused program XLA builds for the
  whole gradient loop has to be **re-compiled from scratch on essentially every training iteration**
  — and that single compile is so large it takes hours on the GPU and leaks memory until the node
  dies. The original, working JAX Dreamer wraps the same kind of loop in one outer compiled
  function, so it compiles **once** and then runs.
- **The fix that matters most** is a roughly 15-line change: hoist the existing `lax.scan` body
  into one `@jax.jit`-decorated function and call that function from the training loop — exactly
  what the original plan ([[buffer_perf_fix_plan_option_L]] Step 3) prescribed but the
  implementation skipped.
- **A second issue compounds it:** the number of gradient steps per iteration **fluctuates**
  (15, 16, 17, …) because of how the gradient-step scheduler accumulates. Each new count is a new
  array shape, which forces yet another fresh compile even after the outer-jit fix. The fix is to
  pad/fix the per-iteration gradient-step count to a constant.

Measured proof (this session, CPU): one DreamerV3 gradient step lowers to **76,005 lines of XLA
HLO** and compiles in **63 s** *on CPU* (GPU is far slower). When the gradient loop is wrapped in
one outer jit, all 16 steps compile together in **~76 s** — essentially the same cost as a single
step, paid **once**. Without the wrapper, that ~76 s (hours on GPU) is paid **per iteration**, on a
loop body that also re-compiles whenever the step count changes.

---

## How the gradient phase is built today

The training loop's gradient phase lives in
[`src/algorithms/dreamer_srl/dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py)
lines **1364–1593** (the "TRAIN GATE"). The relevant structure:

1. The driver computes `n_grad_steps = ratio(ratio_steps)` (line 1373) — a **Python int that
   varies per iteration** (the `Ratio` scheduler in
   [`utils.py`](../../../../src/algorithms/dreamer_srl/utils.py) lines 311–333 accumulates a float
   and rounds, so the count drifts 15/16/17 even at a fixed replay-ratio).
2. It samples `local_data` for all grad steps at once and does one host→device copy
   (lines 1411–1414 — the Option-S fix; this part is fine).
3. **Scan path (the default, lines 1449–1591):** it calls `nnx.split` on all 7 modules/optimizers,
   defines a closure `_scan_body`, and calls `jax.lax.scan(_scan_body, init_carry, scan_xs)`
   **directly in the Python loop body — with no enclosing `@jax.jit`.** Inside `_scan_body` it
   reconstructs the modules with `nnx.merge` and calls `train_step` (the `@nnx.jit`'d
   `one_train_step` from [`train.py`](../../../../src/algorithms/dreamer_srl/train.py) line 648).

The inner `one_train_step` is itself enormous, because it contains **three unrolled Python loops**:

| Loop | Location | Length | What it unrolls |
|---|---|---|---|
| RSSM `observe` over time | `agent.py:1683` `for t in range(T)` | T = `per_rank_sequence_length` = **64** | 64 sequential RSSM transition steps (GRU + transition MLP + representation), fully unrolled |
| `imagine` rollout | `agent.py:1785` `for i in range(1, horizon+1)` | horizon = **15** | 15 imagination steps (recurrent MLP + GRU + transition + actor forward) |
| actor `forward_logits` | `train.py:903` `for h in range(H_plus_1)` | H+1 = **16** | 16 actor forward passes |

The encoder is correctly **outside** these loops (one batched `jax.vmap(self.encoder)` over the
whole `[T*B, obs_dim]` slab at `agent.py:1667`), so Phase 1 ("encoder outside scan") from the
historical retrofit is present. The decoder/reward/continue heads are also batched. **The size of
the HLO comes from the three unrolled time loops inside the train step, not from the encoder.**

## Why it hangs — the root cause

`jax.lax.scan` keeps its body in the compiled program **once** (it does not unroll the scan), so a
scan over 16 grad steps is roughly the size of one grad step, not 16×. That is good. **But a
`lax.scan` written bare in Python still has to be lowered and compiled into an executable before it
can run** — and that executable is keyed on the body's shapes and the scan length.

The original, working JAX Dreamer
([`src/models/dreamer_v3_trainer.py`](../../../../src/models/dreamer_v3_trainer.py)) wraps the whole
scan in ONE outer compiled function: `_scan_train_gpu` is decorated
`@nnx.jit(static_argnums=...)` at **line 683**, and `train_multiple_gpu` (line 791) calls it. The
scan length is passed as a **static** argument. Result: XLA compiles that one big program **once**,
caches it, and every subsequent call dispatches the cached executable. This is the documented
130×/596× retrofit ([[20260519_1508_dreamer_jax_perf_retrofit_4_phases]]).

dreamer_srl **does not have that wrapper.** The `lax.scan` runs in the bare Python loop body. The
consequences, in order of severity:

1. **No persistent compiled executable for the gradient loop.** Each training iteration re-enters
   the Python loop, re-runs `nnx.split`, re-defines the `_scan_body` closure, and calls `lax.scan`
   again. The single huge compile (≈76 k HLO lines) is re-paid. On GPU this single compile is the
   multi-hour "hang" — consistent with the documented 6h31m / RSS 44→85 GB case
   ([[20260521_0151_xla_scan_body_compile_dominates_module_count]]).
2. **Variable scan length forces re-compiles even if a cache existed.** `n_grad_steps` drifts
   (15/16/17…), so `scan_xs` has a different leading-axis length each time. Even XLA's normal jit
   cache (keyed on shapes) would miss on each new length. With no outer jit there is no cache at all,
   but this is *also* why simply adding the wrapper is necessary-but-not-sufficient — see Fix 2.
3. **CUDA-graph / command-buffer accumulation → OOM.** Each fresh compile registers new CUDA graphs;
   the prior bench saw `RESOURCE_EXHAUSTED: ... alive command buffers, CUDA_ERROR_OUT_OF_MEMORY`
   after ~38% of a run. That is the linear RSS climb the user observes.

### This is an implementation deviation from the approved plan

The Option-L plan ([[buffer_perf_fix_plan_option_L]]) **explicitly prescribed the outer jit** — its
Step 3 pseudocode (plan lines 389–438) shows:

```python
@jax.jit                       # <-- THE OUTER WRAPPER THE PLAN REQUIRED
def _scan_grad_steps(...):
    def scan_body(carry, scan_inputs): ...
    final_carry, losses_stack = jax.lax.scan(scan_body, init_carry, local_data_gpu)
    ...
```

The shipped code at `dreamer_srl_main.py:1449–1591` implemented the `scan_body` and the
`jax.lax.scan` call **inline in the training loop with the `@jax.jit` wrapper dropped.** That single
missing decorator is the difference between "compiles once, then trains" and "re-compiles a 76 k-line
program every iteration." The Step-3 bench in the plan recorded the symptom ("scan path is 70× SLOWER
than the legacy for-loop … OOM'd") but attributed it to scan being inherently slow; the real cause is
the missing wrapper.

## Evidence (measured this session, 2026-06-24)

Probe scripts (read-only, no `src/` edits): `tmp/20260624_dsrl_scan_compile_probe.py` and
`tmp/20260624_dsrl_scan_nojit_repro.py`. Config: `01_food_only.yaml` (XS recipe), obs_dim 27,
action_dim 6, T=64, batch=16, horizon=15, scan length 16. Backend: CPU (instruction counts are
backend-independent; GPU compile *time* is strictly worse).

| Lowering | HLO lines | Lower time | Compile time |
|---|---|---|---|
| **(A)** single `one_train_step` (one grad step) | **76,005** | 13.4 s | **63.3 s** |
| **(B)** whole gradient loop wrapped in one `@jax.jit` (scan length 16) | **76,952** | 2.6 s | **75.7 s** |

Readouts:

- **HLO ratio B/A = 1.01.** The scan keeps the body once — 16 grad steps are ~the same program size
  as one. (If the loop were *unrolled* it would be ~16×; it is not. The size problem is the train
  step itself, via the three internal time loops.)
- **Compile amortization B / (16 × A) = 0.07.** Wrapped in one jit, all 16 grad steps compile for
  ~76 s — barely more than a single step, paid once. That ~76 s on CPU is the GPU multi-hour hang
  when paid per-iteration.
- The `nnx.split` + bare-`lax.scan` reproduction (`..._nojit_repro.py`) shows the gradient loop being
  lowered as `jit(scan)` each Python iteration with the per-iteration `n_grad_steps` baked in — i.e.
  the compile recurs and shifts shape with the fluctuating step count.

Live corroboration: on node 114 the definitive run `dsrl_definitive_n114`
(`01_food_only_buf256k.yaml`, `--num-envs 16`) was in state `R` (CPU-bound), RSS ~6.7 GB and a
sibling run at 3.4 GB after ~6 h — matching the "CPU-bound, GPU idle, RSS climbing, no progress"
signature.

## Prioritized fix plan

Ranked by impact on the compile-time pathology. **Fix 1 is the whole ballgame** — it converts the
gradient phase from "recompiles a 76 k-line program every iteration (hours on GPU)" to "compiles once
(~1–2 min on GPU), then runs." Fixes 2–4 remove the remaining recompile triggers and a throughput
tax. The `developer` agent should implement **Fix 1 first, measure, then Fix 2**, before touching 3/4.

> Note for `developer`: the legacy Python for-loop path (`--legacy-grad-loop`) is the current
> production default per [[20260521_0151_xla_scan_body_compile_dominates_module_count]]; the scan
> path is opt-in. Fixes 1+2 make the scan path the correct fast path. Keep `--legacy-grad-loop` as
> the rollback. **All math must stay parity-verified** — run
> `tests/algorithms/dreamer_srl/test_grad_parity.py` and the L2 math-equiv suite after each fix.

### Fix 1 (P0 — the headline) — wrap the gradient-step scan in one outer `@jax.jit`

**File:** `src/algorithms/dreamer_srl/dreamer_srl_main.py`, scan path lines **1449–1591**.

**Change:** lift the scan into a single jitted function, built once before the training loop (so the
graphdefs are captured once and the executable is cached across iterations), following the original
template `src/models/dreamer_v3_trainer.py:683–833` (`_scan_train_gpu` / `train_multiple_gpu`) and the
plan's own Step-3 pseudocode (`buffer_perf_fix_plan_option_L.md` lines 389–438).

Concretely:
- Define `_scan_grad_steps(init_carry, scan_xs)` (or a module-state-in/-out signature) decorated
  `@jax.jit` (or `@nnx.jit`), with the 7 `graphdef_*` captured in its closure and the scan **length
  treated as static** (it is the leading axis of `scan_xs`, so it is fixed per compiled instance — see
  Fix 2 for making it constant). Build this jitted callable **once**, outside the `while` loop, the
  same way `train_step` is built once at line 657.
- In the loop body, replace lines 1461–1591 (the inline `nnx.split` + `_scan_body` + `jax.lax.scan` +
  `nnx.update`) with: split → call the cached `_scan_grad_steps(...)` → `nnx.update` the live modules
  from the returned final state. The `nnx.split`/`nnx.update` bookkeeping stays outside the jit; only
  the scan body is inside.
- The Polyak update, moments threading, and key splitting already inside `_scan_body` move unchanged
  into the jitted function.

**Expected effect:** gradient-loop compile drops from per-iteration (hours on GPU) to **once per
distinct shape** (~1–2 min on GPU, then cached). GPU utilization goes from 0% to active; RSS stops
climbing. This is the fix that makes training actually run.

**Risk / subtlety:** `train_step` is already `@nnx.jit`; nesting it inside an outer `@jax.jit` is fine
(the inner jit is traced through during the outer trace — JAX inlines it). If nesting causes friction,
the cleaner form is to make the inner `one_train_step` a **plain (un-jitted) function** and let the
single outer jit be the only compilation boundary — exactly how `dreamer_v3_trainer.py` does it
(`train_step` at line 125 is `@nnx.jit`, but it is called inside the outer-jitted `_scan_train_gpu`;
either works, plain-inner is marginally cleaner). Developer to pick whichever compiles cleanly and
keeps parity.

### Fix 2 (P0 — required for Fix 1 to actually cache) — make `n_grad_steps` constant per iteration

**File:** `src/algorithms/dreamer_srl/dreamer_srl_main.py` around line **1373**; scheduler in
`src/algorithms/dreamer_srl/utils.py:311–333` (`Ratio.__call__`).

**Problem:** even with Fix 1, the leading axis of `scan_xs` (= `n_grad_steps`) drifts 15/16/17… per
iteration. Each distinct length is a distinct compiled instance → a fresh ~76 s compile each time a new
length first appears. With only a handful of distinct values the storm is bounded (compile 3–4 times,
then cache), but it still stalls early training and re-triggers whenever the buffer/ratio dynamics
shift.

**Change (pick one, developer to choose the lower-risk option and note it):**
- **(a) Fixed bucket:** clamp/pad `n_grad_steps` to a constant `G` per iteration (e.g. the steady-state
  `int(replay_ratio * num_envs)`), carrying any remainder forward so the long-run gradient-step total is
  unchanged (preserves the `Ratio` contract). The scan always sees length `G` → one compiled instance.
- **(b) Cap the distinct set:** round `n_grad_steps` to the nearest of a small fixed set (e.g. snap to
  `G`), accepting a tiny deviation from sheeprl's exact ratio. Cheaper, slightly less faithful.

**Expected effect:** the Fix-1 executable is compiled **once** and reused for the entire run. This is
what turns "compiles 3–4 times then runs" into "compiles once then runs."

**Constraint:** the `Ratio` scheduler is parity-tested
(`tests/algorithms/dreamer_srl/test_utils.py::test_ratio_matches_sheeprl`). Option (a) with
remainder-carry keeps the long-run count identical; option (b) needs a documented, ratified deviation.
Prefer (a). Add a regression test asserting `n_grad_steps` is constant across iterations at steady
state.

### Fix 3 (P2 — throughput, not compile) — move the replay buffer onto the GPU

**File:** `src/algorithms/dreamer_srl/buffers.py` (the `device="gpu"` path already exists, lines 21–24,
68–80) and the launch command (no `--buffer-device gpu` today).

**Problem:** the buffer defaults to CPU/numpy. Sampling (`np.take`/`np.swapaxes`) runs on the host every
iteration, and the result is copied to the device. The Option-S fix already collapsed the per-grad-step
H2D copies into **one** copy per iteration (`dreamer_srl_main.py:1411–1414`), so the worst of the
[[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] regression is gone — but host-side sampling +
one big H2D per iteration is still a tax, and it grows with `num_envs`.

**Change:** after Fixes 1+2 land and training actually runs, benchmark `--buffer-device gpu` vs `cpu`.
If GPU-buffer sampling-inside-the-jit wins, make it the default. This is the Option-M/L convergence with
the original `dreamer_v3_trainer.py:923–1024` GPU buffer. **Do not do this before Fix 1** — a GPU buffer
cannot help while every iteration re-compiles for hours.

**Expected effect:** moderate per-iteration throughput gain (the prior estimate was 2–10× on the buffer
component alone), and it removes a `num_envs`-scaling bottleneck. Secondary to the compile fix.

### Fix 4 (P3 — optional, only if the train step is still too big) — `lax.scan` the internal time loops

**File:** `src/algorithms/dreamer_srl/agent.py` (`observe` `for t in range(T)` at 1683; `imagine`
`for i in range(1, horizon+1)` at 1785) and `train.py` (`actor_loss_fn` `for h in range(H_plus_1)` at
903).

**Problem:** the train step is **76 k HLO lines** mostly because these three Python loops are unrolled
(T=64 + horizon=15 + H+1=16 = 95 unrolled sub-steps). After Fix 1 this compiles **once**, so it is no
longer fatal — but the one-time compile is still ~1–2 min on GPU, and a smaller body compiles faster and
uses less memory.

**Change:** convert the RSSM `observe` time loop (the biggest, T=64) to `lax.scan` over time (carry =
recurrent + posterior state), and optionally the `imagine` loop likewise. This is a math-sensitive
refactor (the carry threading and PRNG-per-step must stay bit-identical), so it is **last** and only if
the one-time compile or memory footprint proves troublesome after Fixes 1–3.

**Expected effect:** smaller HLO → faster one-time compile and lower peak compile memory. Pure
compile-cost/quality-of-life; no steady-state throughput change (the unrolled and scanned forms run the
same FLOPs). Defer unless needed.

## Top fixes to hand to `developer`

1. **Fix 1 — wrap the gradient-step `lax.scan` in one outer `@jax.jit` built once before the loop**
   (`dreamer_srl_main.py:1449–1591`). The single highest-impact change; converts hours-per-iteration
   compile into one cached compile.
2. **Fix 2 — make `n_grad_steps` constant per iteration** (`dreamer_srl_main.py:1373` + `utils.py`
   `Ratio`). Required so Fix 1's executable actually caches across the whole run.
3. **Fix 3 — benchmark and (if it wins) default the GPU buffer** (`buffers.py`, launch flag). Only
   after 1+2 land; a throughput win, not a compile fix.

Fix 4 (scan the internal time loops) is deferred unless the one-time compile/memory is still a problem
after 1–3.

## Implementation Report

*(to be filled by `developer`)*

| Fix | File(s) | Status | Speed before → after | Parity tests | Notes |
|---|---|---|---|---|---|
| Fix 1 | dreamer_srl_main.py | | | | |
| Fix 2 | dreamer_srl_main.py, utils.py | | | | |
| Fix 3 | buffers.py, launch | | | | |
| Fix 4 | agent.py, train.py | | | | |

## Verification Report

*(to be filled by `senior-developer` after implementation)*

## References

- Diagnosis insight (the multi-hour compile pathology): [[20260521_0151_xla_scan_body_compile_dominates_module_count]]
- Historical retrofit template (the working pattern): [[20260519_1508_dreamer_jax_perf_retrofit_4_phases]]
- NNX split/merge technique: [[20260519_1509_nnx_lax_scan_split_merge_pattern]]
- CPU-buffer regression + Option-S fix: [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]]
- The plan whose Step 3 prescribed the outer jit (and whose impl dropped it): [[buffer_perf_fix_plan_option_L]]
- Companion recompile-storm fix (env-reset variable shapes; a different layer): [[recompile_storm_fix]]
- Working reference code: `src/models/dreamer_v3_trainer.py:683-833` (`_scan_train_gpu` / `train_multiple_gpu`)
- dreamer_srl gradient phase: `src/algorithms/dreamer_srl/dreamer_srl_main.py:1364-1593`
- Train step + internal loops: `src/algorithms/dreamer_srl/train.py:613-1022`, `src/algorithms/dreamer_srl/agent.py:1622-1827`
- Probe scripts (this session): `tmp/20260624_dsrl_scan_compile_probe.py`, `tmp/20260624_dsrl_scan_nojit_repro.py`
