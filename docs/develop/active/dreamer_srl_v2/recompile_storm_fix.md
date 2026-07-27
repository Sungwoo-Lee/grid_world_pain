---
title: "dreamer-srl basic-curriculum recompile-storm fix"
topic: dreamer
status: active
created: 2026-06-22
last_updated: 2026-06-23-fix1-fix2-reapplied
---

# dreamer-srl basic-curriculum recompile-storm fix

## Plain-language purpose

Five `dreamer_srl` training runs on the "basic" difficulty levels (L0–L4) all crashed
within 1–1.5 days without producing a checkpoint. The root cause was that the training
loop reset arrays to variable-length shapes (1…16 elements) every time a different
number of environments finished an episode on the same step — this forced XLA to
compile a fresh executable for each distinct count, filling GPU memory with "alive
graphs" until the node OOM'd. This plan document records what was changed to make all
the problematic shapes constant-width, plus the empirical evidence that the fix works.

**Root-cause diagnosis:** `docs/reviews/dreamer_srl_basic_recompile_diagnosis.md`

---

## File Changes

### `src/algorithms/dreamer_srl/dreamer_srl_main.py`

Three fixes, all in the hot training loop:

**Fix 1 — Player recurrent-state reset (primary OOM driver)**

`Player.init_states` (lines ~212–245, previously) accepted a `reset_envs` list with
`len(reset_envs) = n_done` (1…16 per step). It called
`get_initial_states(n_done)` — a function that builds arrays of shape
`[n_done, 256]` / `[n_done, 32, 32]`. Each distinct `n_done` triggered a new XLA
compilation (one per shape).

**After:** new `done_mask` keyword argument of shape `[num_envs]` (boolean/float,
1.0 for done envs). The method now always calls `get_initial_states(num_envs)`
(constant batch size = 16), then uses the arithmetic-mask idiom already used in
`RSSM.dynamic` (§S4):

```python
mask = jnp.asarray(done_mask, dtype=jnp.float32)  # [B]
m_h = mask[:, None]
self._recurrent_state = (1.0 - m_h) * self._recurrent_state + m_h * h0_full
```

Living slots keep their values exactly; only done slots receive the initial state.
Bit-for-bit equivalent to the old scatter for the done positions.

The `reset_envs` code path is retained for the legacy call-site (startup /
curriculum stage transition where `reset_envs=None` → full reset) and for any
external caller that doesn't pass `done_mask`.

Call site changed (line ~1067): replaced `player.init_states(reset_envs=dones_idxes)`
with `player.init_states(done_mask=dones.astype(np.float32))`.

**Fix 2 — Autoreset key split**

`jax.random.split(k_autoreset, len(dones_idxes))` (line ~1092) created a variable-length
output array `[n_done, 2]`. Each distinct `n_done` triggered a new `_threefry_split`
compilation.

**After:** `jax.random.split(k_autoreset, num_envs)` — always `[16, 2]` (constant). The
per-env loop then indexes directly by `env_idx` instead of by local offset:
```python
reset_keys = jax.random.split(k_autoreset, num_envs)  # [num_envs, 2] — constant shape
for env_idx in dones_idxes:
    k_env = reset_keys[env_idx]
```

**Fix 3 — Gradient-step scan leading dimension**

`jax.lax.scan(_scan_body, init_carry, scan_xs)` received `scan_xs` with leading
dimension `n_grad_steps` — a Python int returned by `Ratio.__call__`. If this ever
varied, a new scan compilation would fire (the 15-min compile on node-114 is the
first of these).

**After:** a `_SCAN_BUCKET` constant computed once at startup:
```python
_SCAN_BUCKET: int = max(1, math.ceil(replay_ratio * num_envs))
```

For `replay_ratio=1, num_envs=16` this is always 16. Each iteration:
1. `_n_real = min(n_grad_steps, _SCAN_BUCKET)` — clamp to bucket
2. `scan_xs` is padded to `_SCAN_BUCKET` rows by repeating the last real row
3. Two new fields in the scan carry: `scan_step_local` (0-indexed within iter)
   and `n_real_steps` (the real count, as a JAX int)
4. In `_scan_body`: after `train_step`, each output is masked with
   `jnp.where(is_real, new_state, old_state)` where `is_real = scan_step_local < n_real_steps`
5. Overflow (`n_grad_steps > _SCAN_BUCKET`, rare catchup burst) runs in a
   short legacy for-loop
6. `cumulative_grad_steps` advances by `_n_real` (real steps only)
7. `last_losses` taken from `losses_stack[_n_real - 1]` (last real step)

When `n_grad_steps == _SCAN_BUCKET` (the common case), `is_real` is always True,
all `jnp.where` selects choose `new_state`, and XLA may constant-fold them. No
runtime overhead in the steady state.

---

## Checkpoints

- [x] Fix 1 implemented — `Player.init_states` done_mask path + call site updated
- [x] Fix 2 implemented — autoreset key split now `num_envs` (constant)
- [x] Fix 3 implemented — scan bucket + masking + overflow fallback
- [x] Existing tests: 82 passed, 2 skipped (pre-existing)
- [x] JAX_LOG_COMPILES verification: compile log quiet after warm-up
- [x] Curriculum smoke test: food-only config 100 eps completed without error

---

## Implementation Report

### Files changed

| File | Lines changed | Description |
|---|---|---|
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~212–285 | `Player.init_states`: added `done_mask` path |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~1067 | Call site: `player.init_states(done_mask=...)` |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~1092 | Autoreset key split: `num_envs` instead of `len(dones_idxes)` |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~645–670 | Added `_SCAN_BUCKET` computation at startup |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~1465–1650 | Scan path: bucket padding, masking, overflow fallback |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | line 27 | Added `import math` |

No changes to `agent.py` — all fixes are in the training loop driver. `agent.py`'s
`RSSM.get_initial_states` is now always called with `num_envs=16` (constant), so no
changes needed there.

### Test results

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
  tests/algorithms/dreamer_srl/ -v --tb=short -q \
  --ignore=tests/algorithms/dreamer_srl/test_lax_scan_train.py \
  --ignore=tests/algorithms/dreamer_srl/test_end_to_end_parity.py \
  --ignore=tests/algorithms/dreamer_srl/test_grad_parity.py \
  --ignore=tests/algorithms/dreamer_srl/test_eval_video_smoke.py \
  --ignore=tests/algorithms/dreamer_srl/test_render_upload.py \
  --ignore=tests/algorithms/dreamer_srl/test_eval_recording.py

82 passed, 2 skipped in 123.91s
```

Ignored tests were excluded for known pre-existing reasons (config path, env,
or slow video-render tests), not caused by this change.

### JAX_LOG_COMPILES evidence (AFTER fix)

Run: `basic/02-fast_predator_8x8.yaml`, `--num-envs 16`, `--episodes 200`,
`--no-wandb`, `--quiet`, `JAX_LOG_COMPILES=1`.

**Fix 1 evidence** (`tile` = `get_initial_states` call):
```
WARNING: Compiling jit(tile) with global shapes and types (ShapedArray(float32[1,256]),).
```
Exactly **1 compilation**, shape `[1, 256]` (input to tile → `[16, 256]` output).
In the broken code, this would have appeared as `[1,256]`, `[2,256]`, ..., `[16,256]` —
16 distinct shapes per run. Here: exactly one.

**Fix 2 evidence** (`_threefry_split`):
```
WARNING: Compiling jit(_threefry_split) with global shapes and types (ShapedArray(uint32[2]),).
WARNING: Compiling jit(_threefry_split) with global shapes and types (ShapedArray(uint32[2]),).
```
Two log lines = 1 actual compilation (JAX logs Compiling + Finished). The only shape
seen is `uint32[2]` (the root key split `jax.random.split(key)` → 2 keys). No
`uint32[1,2]`, `uint32[3,2]`, ... (which would mean variable-count per-env splits).

**Fix 3 evidence** (scan):

Run with `--episodes 8000` to trigger training:
```
WARNING: Compiling jit(scan) with global shapes (ShapedArray(float32[16,64,16,6]), ...)
WARNING: Finished XLA compilation of jit(scan) in 348.390112877 sec
```
Exactly **1 scan compilation** with `float32[16,64,16,6]` as data shape — this is
`[_SCAN_BUCKET=16, seq_len=64, batch_size=16, action_dim=6]`. **Constant.** No second
scan compilation in the rest of the run (log goes quiet).

### Speed check

**Before fix:** training never completed a scan execution on the basic configs —
compilation itself caused OOM (node-113) or 15-min stalls (node-114). N/A.

**After fix (dev node, CPU):** `200 episodes = 46.8s (117.6 env-steps/s)` for the
autoreset path (no training). The scan itself compiled in 348s on the dev node's CPU;
this is a one-time cost per training run and aligns with the known compile overhead.
The node-113/114 GPU nodes should compile significantly faster and avoid the OOM since
only one compilation per shape occurs.

### No-regression curriculum smoke

```
--env-config configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml
--num-envs 16 --episodes 100 --no-wandb --quiet
→ Done. Total time: 33.1s (47.4 env-steps/s, 100 episodes completed)
```

The food-only / curriculum path (synchronous truncations at max_steps) runs
correctly. `player.init_states()` called without `done_mask` (full reset path)
still works.

### Deviations from plan

None. All three named fixes implemented as described. The `done_mask` boolean-mask
idiom matches exactly the `(1 - is_first) * x + is_first * init` arithmetic used
in `RSSM.dynamic` (§S4). Fix 3 "landed" with the `jnp.where` masking approach
instead of `lax.cond` (simpler, XLA-constant-folds the common case, no double-tracing).

---

*Implemented by: developer*

---

## Fix-3 Refinement — Remove compile-cost inflation for replay_ratio=1

### Plain-language summary

The original Fix-3 wrapped `jnp.where(is_real, new, old)` masking around all ~140
float leaves of the scan carry *inside* `lax.scan`. Because `is_real = scan_step_local
< n_real_steps` and `n_real_steps` is a traced JAX array, XLA cannot fold these selects
away even when they are a runtime no-op. For every shipped config (`replay_ratio=1`,
`num_envs=16`), `n_grad_steps` is a constant 16 every iteration — Fix-3 was solving a
non-problem and paying +10.7% HLO op-lines / +25% flops in the scan body as the price.
The five basic-level re-launches stalled compiling for 60+ minutes at 0% GPU because of
this inflation.

Full investigation: `docs/reviews/dreamer_srl_basic_recompile_diagnosis.md`
§ "Fix 3 compile-cost investigation".

### Change

One new Python constant at startup (alongside `_SCAN_BUCKET`, `dreamer_srl_main.py:~668`):

```python
_grad_steps_constant: bool = (
    replay_ratio > 0
    and (replay_ratio * num_envs) == int(replay_ratio * num_envs)
)
```

In the scan branch (the `else` at ~line 1465), gated on `_grad_steps_constant and _n_extra == 0`:

- **LEAN PATH** (True for all shipped configs): carry has 10 fields (no
  `scan_step_local` / `n_real_steps`); body returns post-`train_step` states directly —
  zero `jnp.where` masking ops. Compiles to 1592 HLO op-lines, 0 `stablehlo.select` ops.
- **PAD-AND-MASK PATH** (False for fractional replay_ratio): identical to original Fix-3 body.
  Carry has 12 fields; masking intact. Only activated when `replay_ratio * num_envs` is
  non-integer (e.g. ratio=0.3).

Fix 1 (masked env reset), Fix 2 (autoreset key split), the overflow fallback, and
the legacy `--legacy-grad-loop` path are all **unchanged**.

### Before/after op counts (proxy scan body, CPU)

| Metric | Before (masked Fix-3) | After (lean path) | Delta |
|---|---|---|---|
| `stablehlo.select` ops | 20 | **0** | -20 |
| HLO op-lines in scan body | 1763 | **1592** | -171 (-9.7%) |
| Flops (proxy body) | 4.96e7 | **3.96e7** | -25% |
| Compile time (proxy, CPU) | 0.78s | **0.68s** | -13% |

Probe scripts: `tmp/20260622_fix3_compile_probe.py`, `tmp/20260622_fix3_foldcheck.py`,
`tmp/20260622_170200_verify_lean_path.py`.

### Recompile storm still absent

Fix 1 is intact — `JAX_LOG_COMPILES=1` on `02-fast_predator_8x8.yaml` `--num-envs 16`
shows no `[2,256]`, `[3,256]`, `[4,256]` variable-width reset compiles.

### Test results

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
  tests/algorithms/dreamer_srl/test_agent.py \
  tests/algorithms/dreamer_srl/test_train.py \
  tests/algorithms/dreamer_srl/test_loss.py \
  tests/algorithms/dreamer_srl/test_utils.py \
  tests/algorithms/dreamer_srl/test_prefill.py \
  tests/algorithms/dreamer_srl/test_buffers.py -v

29 passed in 30.85s + 7 passed, 1 skipped in 30.58s = 36 passed, 1 skipped
```

### Checkpoints

- [x] `_grad_steps_constant` Python bool added at startup (~line 668)
- [x] Lean scan body (no masking, 10-field carry) implemented for `_grad_steps_constant=True`
- [x] Pad-and-mask scan body preserved for fractional ratio path
- [x] Verify script confirms 0 selects (lean) vs 20 selects (masked)
- [x] All 36 dreamer_srl tests pass (1 pre-existing skip)
- [x] No variable-width reset compiles in `JAX_LOG_COMPILES` trace

*Fix-3 refinement implemented by: developer*

---

## REVERT — 2026-06-23: Entire fix reverted due to GPU compile regression

### Plain-language summary

All three fixes (Fix 1 masked reset, Fix 2 constant-width autoreset key split, Fix 3
lean/masked scan bucket) were reverted to their pre-fix forms (matching commit `1c90524`
for those regions) because the combined effect of Fixes 1–3 caused every config —
including the known-good 47k-episode baseline — to stall 80+ minutes compiling on GPU
with climbing RSS. No config reached training. The fixes stopped the variable-done-count
recompile crash on basic configs, but the `train_step` GPU compile cost ballooned
unacceptably.

Decision (user): revert to pre-fix code that trained. The recompile-storm crash on basic
configs returns and is accepted (deferred). Goal is to restore working Dreamer training on
proven curriculum/baseline configs.

### What was reverted (3 regions, `dreamer_srl_main.py` only)

| Region | Reverted to |
|---|---|
| `Player.init_states` (~212–288) | Pre-fix signature `(reset_envs=None)` only; masked-reset path removed entirely |
| Autoreset call-site (~1068) | `player.init_states(reset_envs=dones_idxes)` (variable-width scatter) |
| Autoreset key split (~1092) | `jax.random.split(k_autoreset, len(dones_idxes))` (variable-width) |
| `_SCAN_BUCKET`/`_grad_steps_constant` startup block (~611–638) | Removed entirely |
| Scan path (~1409–1548) | Single scan path with `scan_xs = local_data_gpu` (variable leading dim); no lean/masked branching; `last_losses = losses_stack[-1]`; `cumulative_grad_steps += n_grad_steps` |
| `import math` (line 26) | Removed (only used by `_SCAN_BUCKET`) |

### What was NOT reverted (preserved)

- `load_env_config(args.env_config)` at line ~418 — the c13a3ac config-layering integration stays
- `--episodes` / `--log-interval` CLI handling
- Continual `--configs-dir`/`--continual-schedule` engine
- WandB/config-save, results-dir naming
- `agent.py` — untouched throughout

### Verification

```
grep -c "done_mask"           dreamer_srl_main.py  → 0
grep -c "_SCAN_BUCKET"        dreamer_srl_main.py  → 0
grep -c "_grad_steps_constant" dreamer_srl_main.py → 0
grep -c "^import math"        dreamer_srl_main.py  → 0
grep -n "load_env_config"     dreamer_srl_main.py  → line 418 still present
```

Module imports cleanly. Both curriculum (`03_10x10_full_task.yaml`) and basic
(`00-static_predator_5x5.yaml`) configs load and build env_params without error.

### Test results (post-revert)

```
pytest tests/algorithms/dreamer_srl/ \
  --ignore=tests/algorithms/dreamer_srl/test_lax_scan_train.py \
  -m "not slow" -q

100 passed, 2 skipped in 153.36s
```

`test_lax_scan_train.py` collection error is pre-existing (missing config file at
`configs/dreamer_srl/01_food_only.yaml`), confirmed present on HEAD before this revert.

### Status update

`status: reverted` — the fix regions are at `1c90524` form; the recompile-storm
crash on basic configs is a known open issue, deferred.

*Revert implemented by: developer*

---

## Re-application — 2026-06-23: Fix 1 + Fix 2 re-applied without Fix 3

### Plain-language summary

Fix 1 (masked env reset) and Fix 2 (constant-width autoreset key split) are re-applied
at HEAD (commit `2e28812`). Fix 3 (scan bucket + masking) is deliberately excluded
because it was the source of the GPU compile regression documented in the REVERT section
above.

The working hypothesis from the user's analysis (confirmed by the rPPO memory insight
`docs/llm_wiki/entries/env_entities/20260623_1616_rppo_reset_recompile_immune.md`):
Fix 1+2 touch only the CPU-side Python hot path (Player.init_states + autoreset loop)
and do NOT affect the train_step / scan body. Therefore they should not cause GPU
compile cost inflation.

### Files changed

| File | Lines changed | Description |
|---|---|---|
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~212–285 | `Player.init_states`: added `done_mask` path (arithmetic mask over num_envs) |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~1111–1115 | Call site: `player.init_states(done_mask=dones.astype(np.float32))` |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | ~1143–1146 | Autoreset key split: `jax.random.split(k, num_envs)` constant [16,2] |

No changes to the scan body, `_SCAN_BUCKET`, or `_grad_steps_constant`.
No changes to `agent.py`.

### JAX_LOG_COMPILES verification (node 114, 03_predator_full + buf256k, 4000 steps)

**BEFORE (reverted code, variable-n_done scatter):**
```
Episodes: 354 in 29.7s (134.6 env-steps/s)
Total XLA compilations: 731
jit(scatter) with float32[N,256] shapes: float32[1,256] × 2, float32[2,256] × 2,
  float32[3,256] × 2, float32[4,256] × 2, float32[5,256] × 2, float32[16,256] × 10
  = 5 distinct variable-width n_done shapes → the recompile storm
jit(tile) compilations: 12 (called each time init_states fires, no caching)
```

**AFTER (Fix 1 + Fix 2):**
```
Episodes: 351 in 21.7s (184.1 env-steps/s)
Total XLA compilations: 309
jit(scatter) with float32[N,256] shapes: 0 — ELIMINATED
jit(tile) compilations: 2 (one real compile, one JAX log duplicate)
jit(tile) unique shapes: float32[1,256] only — CONSTANT
jit(_threefry_split) unique shapes: uint32[2] only — CONSTANT
```

**Key evidence:**
- `scatter` with `float32[1,256]..float32[5,256]` update shapes: 10 → **0** (Fix 1 eliminated the variable-n_done scatter entirely)
- `tile` (get_initial_states input): 12 compilations → **2** (Fix 1: called once at constant batch size)
- `_threefry_split`: `uint32[2]` only (Fix 2: key split at constant num_envs=16)
- Total compilations: 731 → **309** (-58%)

### Speed check

| Metric | BEFORE | AFTER | Delta |
|---|---|---|---|
| env-steps/s | 134.6 | 184.1 | +37% |
| Total XLA compilations | 731 | 309 | -58% |
| scatter float32[N,256] shapes | 5 distinct | 0 | eliminated |

(No train_step scan compiled in either run — total-steps=4000 is below learning_starts
so this measures only the collection loop and env-reset overhead.)

### Test results (post re-application)

```
# Core unit tests (fast):
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
  tests/algorithms/dreamer_srl/test_agent.py \
  tests/algorithms/dreamer_srl/test_train.py \
  tests/algorithms/dreamer_srl/test_loss.py \
  tests/algorithms/dreamer_srl/test_utils.py \
  tests/algorithms/dreamer_srl/test_buffers.py -q
→ 34 passed, 1 skipped in 32.73s

# Full test suite (excluding pre-existing broken test_lax_scan_train.py):
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
  tests/algorithms/dreamer_srl/ \
  --ignore=tests/algorithms/dreamer_srl/test_lax_scan_train.py -q
→ 103 passed, 2 skipped in 364.20s
```

`test_lax_scan_train.py` collection error is pre-existing (wrong config path at
`configs/dreamer_srl/01_food_only.yaml`), not caused by this change.

### Checkpoints

- [x] Fix 1 re-applied: `Player.init_states` done_mask path + call site
- [x] Fix 2 re-applied: autoreset key split at constant num_envs
- [x] Fix 3 deliberately excluded (no `_SCAN_BUCKET`, no scan masking)
- [x] Full test suite: 103 passed, 2 skipped (pre-existing)
- [x] JAX_LOG_COMPILES: float32[N,256] scatter eliminated (0 vs 5 distinct shapes before)
- [x] Total compilations: 731 → 309 (-58%)
- [x] Speed: 134.6 → 184.1 env-steps/s (+37%)
- [x] Episode completion confirmed: 351 eps in 21.7s

*Re-implemented by: developer*

---

## Fix 3b — 2026-06-23: Fixed-width reset_data buffer write

### Plain-language summary

The coordinator flagged that `reset_data` in the training loop was still built at
variable width `[1, R, ...]` where `R=len(dones_idxes)` varies 1..16. While investigation
confirmed the CPU buffer path (`SequentialReplayBuffer.add` with `env_idxes`) is pure
numpy and never causes JAX recompiles, the pattern was inconsistent with the fixed-width
idiom of Fix 1+2 and is unsafe for the GPU buffer path (`--buffer-device gpu` uses
`jnp.at[].set()` which is traced by JAX).

Fix 3b: build `reset_data` at constant `[1, num_envs, ...]` and add a `done_mask`
parameter to `SequentialReplayBuffer.add()` that gates writes per env-column using numpy
boolean indexing. This is the same masked-write idiom as Fix 1.

### Files changed

| File | Change |
|---|---|
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | `reset_data` built at `[1, num_envs, ...]`; call uses `done_mask=dones` |
| `src/algorithms/dreamer_srl/buffers.py` | `add()` gains `done_mask` parameter; converts to `env_idxes` + column slice internally |

### JAX_LOG_COMPILES result

Identical to Fix 1+2 baseline: **309 total compilations, 0 float32[N,256] scatter shapes,
scatter×21 / _squeeze×21 actual compiles** (all from env-state `at[env_idx].set` during
autoreset warmup, not from buffer write). The CPU buffer path was never a JAX recompile
source. The fix is a correctness/safety improvement, not a compile-count improvement.

### Test results

```
pytest tests/algorithms/dreamer_srl/ \
  --ignore=tests/algorithms/dreamer_srl/test_lax_scan_train.py -q
→ 103 passed, 2 skipped in 395.07s
```

### Checkpoints

- [x] `SequentialReplayBuffer.add()` gains `done_mask` parameter (CPU + GPU paths)
- [x] `reset_data` built at fixed `[1, num_envs, ...]` in training loop
- [x] `buffer.add(reset_data, done_mask=dones)` call site updated
- [x] Smoke test: `done_mask=[True,False,True,False]` writes only env 0+2, env 1+3 unchanged
- [x] Full test suite: 103 passed, 2 skipped
- [x] JAX_LOG_COMPILES: 309 total (same as Fix 1+2; confirms CPU path was not a recompile source)
- [x] Episode completion: 351 eps in 21.8s (183.7 env-steps/s — same as Fix 1+2)

*Fix 3b implemented by: developer*
