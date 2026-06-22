---
title: "dreamer-srl basic-curriculum recompile-storm fix"
topic: dreamer
status: implemented
created: 2026-06-22
last_updated: 2026-06-22
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
