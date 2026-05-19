---
title: "dreamer-srl v2 buffer perf fix — Option L (lax.scan + GPU buffer), in measurable steps"
topic: dreamer
status: active
created: 2026-05-19
last_updated: 2026-05-19
phase: 2
---

# dreamer-srl v2 buffer perf fix — Option L (lax.scan + GPU buffer), in measurable steps

> **Status**: PLANNED
> **Opened**: 2026-05-19
> **Related**:
> - Diagnosis: `docs/memory/memories/dreamer_diagnosis/20260519_1507_dreamer_srl_v2_cpu_buffer_regression.md`
> - Engineering template: `docs/memory/memories/dreamer_diagnosis/20260519_1508_dreamer_jax_perf_retrofit_4_phases.md`
> - Technique: `docs/memory/memories/dreamer_diagnosis/20260519_1509_nnx_lax_scan_split_merge_pattern.md`
> - Working reference: `src/models/dreamer_v3_trainer.py:791-833` (`train_multiple_gpu`)
> - Companion v2 docs: `docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md`, `GRAD_PARITY_METHODOLOGY.md`

---

## Context

The new "dreamer-srl v2" trainer (the audit-and-fix rebuild that finally parity-passes against sheeprl) is **slower per environment step than the original JAX Dreamer trainer**, especially as the number of parallel environments grows. The cause is that the replay buffer at `src/algorithms/dreamer_srl/buffers.py` is a pure-numpy/CPU port of sheeprl's PyTorch buffer, and the training-step loop at `src/algorithms/dreamer_srl/dreamer_srl_main.py:866-895` calls `jnp.asarray(...)` once per gradient step (line 885) — each call is a **host→device transfer** ("H2D": copy from CPU RAM to the GPU). With sheeprl's grad-step ratio that's tens of CUDA-launch overheads per training iteration. The empirical signature is a ~30% per-env-step slowdown going from 64 to 128 parallel envs, which is sub-linear scaling consistent with buffer-bandwidth contention. The original JAX Dreamer fixed an analogous bottleneck back on 2026-02-21 by porting the buffer to GPU and putting the whole gradient-step loop inside a JIT'd `lax.scan` (a JAX construct that turns a Python `for` loop over a fixed number of iterations into one fused GPU kernel) — that retrofit measured a **596× speedup** on the collection loop and **130×** on `train_step`. This plan ports the same retrofit to dreamer-srl v2 as **Option L** from the diagnosis insight's S/M/L fix ladder.

The work is split into **six small, individually reversible steps**, each with (a) a parity test that must still pass and (b) an SPS micro-benchmark whose delta is recorded against the previous step. The goal is a measurable contribution per step rather than one big-bang commit. A separate **Step 0** is dedicated to investigating the secondary, user-reported pattern that `Time/sps_env` appears to **decay over the course of a training run** — which may be partly explained by the fact that the logged quantity at `dreamer_srl_main.py:913` is `policy_step / (time.time() - t_start)`, i.e. a **cumulative running average**, not a rate-over-window. Step 0 distinguishes the metric artefact from any real throughput degradation (buffer-fill pressure, Python GC, WandB logging cost, etc.) before the implementation steps begin.

## Goals

- (a) **Implement Option L** — a `lax.scan`-based training loop with a GPU-resident replay buffer — in a series of **small commits**, each independently testable.
- (b) **Per-step measurement** — after each step, run a **parity test** (math equivalence vs the pre-step baseline) and an **SPS micro-benchmark** (a short canned training run); record the delta.
- (c) **Final speed comparison** — at the end, tabulate Step 0 (baseline) vs each intermediate step vs final cumulative speedup. Baseline is the current worktree HEAD at plan-authoring time (`3d1f0c7` or whatever the implementing developer's HEAD is when Step 0 runs).
- (d) **Diagnose the `Time/sps_env` decay** observed in the current Dreamer training. Distinguish metric artefact (cumulative-average rendering as decay) from real throughput regression (e.g., buffer growing past size threshold, GC, JIT cache pressure, monotonic WandB log overhead). If a real cause exists, fix it.

## Analysis

### Why the current code is slow

| Location | Issue | Cost |
|---|---|---|
| `buffers.py:25,63,66` | Buffer storage is `np.ndarray` + numpy RNG | Sampling runs on CPU; output must be copied to GPU. |
| `dreamer_srl_main.py:885` | `arr = jnp.asarray(v[i], dtype=jnp.float32)` **inside** the `for i in range(n_grad_steps)` loop | One H2D launch per gradient step → tens per training iteration. |
| `dreamer_srl_main.py:866-895` | Python `for` loop drives gradient steps | No XLA fusion across grad-step boundaries; per-call Python overhead. |
| `buffers.py:355-376` (`_get_samples`) | `np.take` + `np.swapaxes` on the CPU buffer | Per-iteration CPU work that scales with `batch_size * seq_len * num_envs`. |

### Why the original Dreamer is fast

`src/models/dreamer_v3_trainer.py:791-833` (`train_multiple_gpu`) keeps every byte of the replay buffer on GPU as `jnp.ndarray`, splits the NNX trainer module into `(graphdef, state)` via `nnx.split` ("**split / merge** pattern", see insight `20260519_1509`), then drives `num_steps` gradient steps inside a single `jax.lax.scan` whose body re-creates a working trainer instance via `nnx.merge(graphdef, current_state)` and writes the final state back with `nnx.update` *once* after the scan returns. Zero H2D transfers inside the scan; one `nnx.update` at the end.

### Why `Time/sps_env` may *appear* to decay even with no real regression

Look at `dreamer_srl_main.py:913`:
```python
sps_env = policy_step / max(time.time() - t_start, 1e-9)
```
This is the **cumulative average** since training started, not the recent rate. If the *instantaneous* SPS is constant at some value `r` after a slow warm-up (compilation, first-iteration JIT, prefill), the cumulative average will rise from below `r` toward `r` asymptotically. Conversely, if there's a brief fast period followed by a steady slower regime (e.g., prefill is fast → training is slower), the cumulative average will **decay from above toward the steady value**, looking exactly like the user-reported "decreasing SPS" pattern. Step 0 must compute an **instantaneous SPS** (Δpolicy_step / Δwall_time over the last log window) and plot both side-by-side before claiming there is a real throughput problem to fix.

### Cross-reference: the original retrofit's 4 phases vs this plan

| Original retrofit phase | This plan's equivalent | Why ordered this way |
|---|---|---|
| Phase 1: encoder outside scan | n/a (dreamer-srl already calls encoder once per batch inside `train_step`) | Pre-existing; no work. |
| Phase 2: JIT-compile `train_step` (130×) | already done — `train_step` in dreamer-srl is JIT'd | Pre-existing; no work. |
| Phase 3: JIT collection / grad-step loop via `lax.scan` (596×) | **Step 3 + Step 4** | The big lever. |
| Phase 4: vectorize replay sampling (4.8×) | absorbed into **Step 2** (GPU buffer's `.at[].set` / `jax.random.randint` paths) | Falls out of GPU-mode sampling. |

The S → M → L fix ladder in the diagnosis insight maps to **Step 1 (S)**, **Step 2 (M slice)**, **Step 3 + Step 4 (L)** of this plan. Step 1 is non-blocking on later steps — it captures most of the easy win first.

---

## Implementation Plan

### Design

Six steps. Each one is one logical commit with one measurement.

```
Step 0 — Baseline + SPS-decay investigation        (no code change; investigation only)
Step 1 — Option S: hoist H2D out of grad-step loop  (~10 lines; bit-identity preserved)
Step 2 — Option M slice: GPU-mode buffer flag       (~120 lines; opt-in, CPU mode default)
Step 3 — lax.scan over grad-step loop (CPU-mode)    (~80 lines; uses split/merge pattern)
Step 4 — Combine 2+3, default to device="gpu"       (~30 lines + test reshuffle)
Step 5 — (optional, deferred) lax.scan over outer collection iteration
Step 6 — Final benchmark + write-up                 (measurement + plan close-out)
```

Each commit has the same gate at the bottom: **parity test ✓, SPS micro-benchmark ran, delta recorded**.

The **SPS micro-benchmark fixture** is defined once in Step 0 and re-used identically at every step so deltas are comparable. See "SPS micro-benchmark fixture" below.

---

### Step 0 — Baseline measurement + SPS-decay investigation

**No code change** in this step. This is an investigation step that produces (a) a baseline number every later step is measured against and (b) a verdict on whether `Time/sps_env` decay is real or a metric artefact.

#### 0.1 SPS micro-benchmark fixture (define once, re-use at every step)

- **Config**: `configs/dreamer_srl/01_food_only_smoke.yaml` (already exists; small XS recipe, food-only env).
- **Override at the CLI** (do **not** edit the config — keep it stable across steps): `num_envs=16`, total `policy_steps=50000`, `seed=0`, `wandb.enabled=false` (we want a clean SPS, not WandB I/O overhead in the measurement), `learning_starts` set to a small value (e.g., `1000 / num_envs = ~63`) so the gradient-step loop actually runs.
- **Hardware**: launch via `run_command.py` on whichever node is free, **must be the same node for every step** (`node101` recommended — the regular dreamer training box). Record the node ID in every measurement.
- **Wall-time budget**: ≤ 5 minutes per run. 50k env-steps × ~50 SPS ≈ ~16 min if perf is bad; if a run is too slow to finish in 5 min, fall back to 25k steps and note the change in the table.
- **Helper script** (the developer to create): `tests/algorithms/dreamer_srl/bench_sps.py`. It runs the trainer end-to-end with the overrides above and prints two numbers — **instantaneous SPS** averaged over the last 25% of training, and **cumulative SPS** over the full run — plus a CSV trace of `(policy_step, wall_time, instantaneous_sps)` rows to `tmp/sps_bench_<step>_<timestamp>.csv` for plotting.
- **Invocation**: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python tests/algorithms/dreamer_srl/bench_sps.py --config configs/dreamer_srl/01_food_only_smoke.yaml --num-envs 16 --total-steps 50000 --seed 0 --label step0_baseline`.

The script does **not** require WandB and can run offline.

#### 0.2 SPS-decay investigation methodology

- **Pull `Time/sps_env` history from a long completed run** that already has the data. Candidates: `yxij4lrc` (XS/16/4M, the production winner) or `02n94uzu` (XS/64/4M). Use the WandB API:
  ```python
  import wandb
  api = wandb.Api()
  run = api.run("<entity>/<project>/yxij4lrc")  # adjust entity/project
  hist = run.history(keys=["Time/sps_env", "timesteps"], pandas=True)
  ```
- **Plot `Time/sps_env` vs `timesteps`** (matplotlib; save to `tmp/sps_decay_<runid>.png`).
- **Compute the implied instantaneous SPS** from the cumulative rate. If `sps_cum(t) = t / wall_time(t)`, then `wall_time(t) = t / sps_cum(t)`, and the instantaneous rate over a window `[t1, t2]` is `(t2 - t1) / (wall_time(t2) - wall_time(t1))`. Plot the **instantaneous SPS** vs `timesteps` on the same chart.
- **Cross-reference with `buffer._pos` growth.** The buffer fills around `policy_step ≈ buffer_size × num_envs` (with `buffer_size=1M, num_envs=16` that's ~16M env-steps — past the end of a 4M run, so the buffer is **never full** in production runs). If sps_inst is flat through the 4M run, the decay is purely the cumulative-average artefact.
- **If sps_inst** *is* **declining**, the candidate causes are:
  - (i) Replay-buffer numpy compute scaling — should be ruled out via the same WandB chart on a `buffer_size=10k` configured run, but we have no such run. Defer to Step 0.3 micro-experiment.
  - (ii) Python GC fragmentation / heap growth — characteristic: linear-in-time decline; rule out with a `psutil` memory trace alongside.
  - (iii) JIT cache pressure (shape changes triggering retraces) — characteristic: step-discontinuities in sps_inst; rule out with `JAX_LOG_COMPILES=1`.
  - (iv) WandB log overhead growing with run length — characteristic: linear with log count; rule out by toggling `wandb.enabled=false` on a short repro and comparing.

#### 0.3 Controlled micro-experiment (only if sps_inst decay is real)

- Run **two** Step-0 benchmarks back-to-back with the same recipe but different `agent.buffer_size`:
  - `buffer_size=10000` (small)
  - `buffer_size=1000000` (default)
- Total env-steps fixed at 50k (so the small buffer fills, the big buffer does not).
- If the small buffer **decays faster** than the big buffer → numpy compute scales with `_pos`. Option S/M directly addresses this.
- If both decay at the same rate → cause is NOT the buffer; flag and investigate one of (ii)/(iii)/(iv) above.

#### 0.4 Step 0 deliverables

- `tmp/sps_bench_step0_baseline_<timestamp>.csv` — the baseline trace.
- `tmp/sps_decay_<runid>.png` — the WandB-pulled decay chart with both cumulative and instantaneous SPS.
- A "Step 0 findings" subsection appended to this plan doc with: (a) baseline numbers (cumulative + instantaneous, last-25%-window), (b) the verdict on decay-is-artefact vs decay-is-real, (c) if real, the candidate cause (i)–(iv).

#### Step 0 parity test
- N/A (no code change).

#### Step 0 SPS gate
- Record the baseline. This is the number all later steps compare against.

#### Step 0 time estimate
- 1–2 hours: ~30 min to set up `bench_sps.py`, 30 min for the WandB pull + plot, 30 min for the optional micro-experiment if needed, 30 min to write up findings.

---

### Step 1 — Option S: hoist H2D out of grad-step loop

The smallest possible fix: do the `np.ndarray → jnp.ndarray` conversion **once per training iteration** instead of once per gradient step. This is non-blocking on every later step.

#### Change

**`src/algorithms/dreamer_srl/dreamer_srl_main.py` (around lines 859–895)**

```python
# BEFORE (current):
local_data = buffer.sample(
    batch_size=batch_size,
    sequence_length=seq_len,
    n_samples=n_grad_steps,
)
# local_data shape: [n_grad_steps, seq_len, batch_size, ...]

for i in range(n_grad_steps):
    # ... target update ...
    batch = {}
    for k, v in local_data.items():
        arr = jnp.asarray(v[i], dtype=jnp.float32)  # ← H2D PER GRAD STEP
        batch[k] = arr
    # ... train_step ...

# AFTER:
local_data = buffer.sample(
    batch_size=batch_size,
    sequence_length=seq_len,
    n_samples=n_grad_steps,
)
# Single H2D for the whole iteration (Option S — see plan §Step 1)
local_data_gpu = jax.tree.map(
    lambda x: jnp.asarray(x, dtype=jnp.float32),
    local_data,
)
# local_data_gpu[k] shape: [n_grad_steps, seq_len, batch_size, ...] on device

for i in range(n_grad_steps):
    # ... target update ...
    batch = {k: v[i] for k, v in local_data_gpu.items()}  # JAX slice — no H2D
    # ... train_step ...
```

Slicing a JAX array with a Python int (`v[i]`) is a free trace-time operation and does not retrigger H2D — it dispatches to `lax.dynamic_index_in_dim`. The hot transfer is gone.

#### Step 1 parity test

- Existing `tests/algorithms/dreamer_srl/test_grad_parity.py` — must pass unchanged. Bit-identity is preserved because the per-step batch dict is exactly the same numbers, just reached via one H2D instead of N.
- Run: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/algorithms/dreamer_srl/test_grad_parity.py -v`.

#### Step 1 SPS gate

- Run `bench_sps.py --label step1_option_s` with the **identical** Step-0 invocation. Record the delta in the Implementation Report table.
- **Expected improvement** (from the diagnosis insight): 2–5× iteration speedup. If the measured delta is < 1.5×, something else is dominating — flag and investigate before Step 2.

#### Step 1 time estimate
- 1 hour: 10 min change, 20 min parity test, 20 min SPS bench, 10 min write-up.

---

### Step 2 — Option M slice: GPU-mode buffer flag

Add a `device: "cpu" | "gpu"` flag to `SequentialReplayBuffer`, defaulting to `"cpu"` (preserves all existing bit-identity tests). When `device="gpu"`, the buffer allocates `jnp.zeros(...)` and uses `.at[indices].set(...)` / `jax.random.randint(key, ...)` paths. This is a *capability* commit — no caller is forced to use GPU mode yet. Step 4 is where we flip the default.

#### Change A

**`src/algorithms/dreamer_srl/buffers.py:49-66`** — add `device` parameter and dual storage:

```python
def __init__(
    self,
    buffer_size: int,
    n_envs: int = 1,
    obs_keys: Sequence[str] = ("observations",),
    device: str = "cpu",   # ← NEW
) -> None:
    # ... existing validation ...
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    self._device: str = device
    self._on_gpu: bool = (device == "gpu")
    # ... existing fields ...
    if self._on_gpu:
        # Lazy-init in add(); we don't know dtypes/trailing shapes until first add.
        self._xnp = jnp
    else:
        self._xnp = np
```

#### Change B

**`buffers.py:105-210` (`add`)** — branch the np vs jnp paths. The structure is a near-copy of `dreamer_v3_trainer.py:946-977` (the original ReplayBuffer `add_batch` on GPU mode):
- `np.zeros / np.empty` → `jnp.zeros` when `self._on_gpu`.
- `self._buf[k][idxes] = data` → `self._buf[k] = self._buf[k].at[idxes].set(data)` when `self._on_gpu`. (Note: `.at[].set` returns a new array; the buffer dict is rebound.)
- `np.ix_(idxes, env_idxes)` (used in the `env_idxes is not None` branch at line 195) — JAX has no `ix_`; use broadcasting: `self._buf[k] = self._buf[k].at[idxes[:, None], env_idxes[None, :]].set(data)`.

#### Change C

**`buffers.py:216-298` (`sample`)** — accept an **optional** `key: jax.Array | None = None` parameter; when `self._on_gpu`, require a key and use `jax.random.randint(key, ...)` instead of `self._rng.integers(...)`. Existing CPU-path callers pass no key; new GPU-mode callers pass one. Internal flow:
- `valid_idxes` construction stays the same (Python ints into `jnp.array`).
- `start_idxes = jax.random.randint(key, (batch_dim,), 0, len(valid_idxes), dtype=jnp.int32)`, then `valid_idxes[start_idxes]` instead of `valid_idxes[self._rng.integers(...)]`.
- `chunk_length = jnp.arange(sequence_length)` and the broadcast composition stays identical (just `jnp` instead of `np`).

#### Change D

**`buffers.py:300-376` (`_get_samples`)** — branch on `self._on_gpu`. CPU path unchanged. GPU path uses `jnp.take` / `jnp.reshape` / `jnp.swapaxes`. Env-selection PRNG path (`self._rng.integers(...)` on line 344) also needs a key for GPU mode — propagate the same `key` through.

#### Change E (new test)

**`tests/algorithms/dreamer_srl/test_gpu_buffer.py`** (new file) — assert math equivalence between `device="cpu"` and `device="gpu"` modes on the same fixed seed:

```python
# Pseudocode of the assertion shape — developer to flesh out:
def test_gpu_mode_matches_cpu_mode_after_add_sample_roundtrip():
    rng = np.random.default_rng(0)
    # Construct deterministic add() input
    data_in = {...}
    cpu_buf = SequentialReplayBuffer(buffer_size=128, n_envs=4, obs_keys=("obs",), device="cpu")
    gpu_buf = SequentialReplayBuffer(buffer_size=128, n_envs=4, obs_keys=("obs",), device="gpu")
    cpu_buf.add(data_in)
    gpu_buf.add(jax.tree.map(jnp.asarray, data_in))
    # Compare stored buffer contents
    for k in cpu_buf._buf:
        np.testing.assert_allclose(np.asarray(gpu_buf._buf[k]), cpu_buf._buf[k], atol=1e-5)
    # Sample with matched keys — math equivalence, not bit-identity
    # (CPU uses np.random.Generator, GPU uses jax PRNG → different RNG streams; check shapes/dtypes only)
    cpu_sample = cpu_buf.sample(batch_size=4, sequence_length=8, n_samples=2)
    gpu_sample = gpu_buf.sample(batch_size=4, sequence_length=8, n_samples=2, key=jax.random.PRNGKey(0))
    for k in cpu_sample:
        assert cpu_sample[k].shape == gpu_sample[k].shape
        assert cpu_sample[k].dtype == gpu_sample[k].dtype.name or True  # dtype kinds match
```

#### Step 2 parity test

- Existing `tests/algorithms/dreamer_srl/test_grad_parity.py` and `tests/algorithms/dreamer_srl/test_buffers.py` — **must pass unchanged** because `device="cpu"` is the default.
- New `test_gpu_buffer.py` — passes.
- Run: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/algorithms/dreamer_srl/ -v`.

#### Step 2 SPS gate

- The main trainer still constructs the buffer with `device="cpu"` (no caller change in Step 2), so the SPS micro-benchmark should be **unchanged** from Step 1. Record this as a parity gate, not an improvement gate — if Step 2 *slows things down*, something regressed; investigate.

#### Step 2 time estimate
- 4–6 hours: 1 hr `add` port, 1 hr `sample`/`_get_samples` port, 1 hr test, 30 min benchmark, 1 hr write-up.

---

### Step 3 — `lax.scan` over the grad-step loop (CPU-mode buffer still)

Replace the Python `for i in range(n_grad_steps)` at `dreamer_srl_main.py:866-895` with a `jax.lax.scan` over the gradient-step axis of `local_data_gpu` (from Step 1). The buffer stays on CPU in this step (sampled once at iteration start, transferred at iteration start — that's the Step 1 H2D); the win is removing Python-loop overhead between gradient steps and giving XLA the freedom to fuse work across steps.

This is the step that **requires the `nnx.split` / `nnx.merge` pattern** from insight `20260519_1509`. We split *all three* trainable NNX modules (`world_model`, `actor`, `critic`) plus their optimizers, carry their `state` pytrees through the scan, and `nnx.merge` inside the scan body. The `target_critic` is also a module — it can be carried similarly, but its updates are pure pytree arithmetic (Polyak) and can be expressed as part of the carry without a `merge` if cleaner.

#### Design sketch

```python
# OUTSIDE the JIT boundary, replacing the Python for-loop at line 866:
graphdef_wm,    state_wm    = nnx.split(world_model)
graphdef_ac,    state_ac    = nnx.split(actor)
graphdef_cr,    state_cr    = nnx.split(critic)
graphdef_tg,    state_tg    = nnx.split(target_critic)
graphdef_wm_opt, state_wm_opt = nnx.split(wm_opt)
graphdef_ac_opt, state_ac_opt = nnx.split(actor_opt)
graphdef_cr_opt, state_cr_opt = nnx.split(critic_opt)

(state_wm, state_ac, state_cr, state_tg,
 state_wm_opt, state_ac_opt, state_cr_opt,
 moments, key, last_losses_stack) = _scan_grad_steps(
    graphdef_wm, graphdef_ac, graphdef_cr, graphdef_tg,
    graphdef_wm_opt, graphdef_ac_opt, graphdef_cr_opt,
    state_wm, state_ac, state_cr, state_tg,
    state_wm_opt, state_ac_opt, state_cr_opt,
    moments, key,
    local_data_gpu,                              # carry input: pre-stacked batch
    n_grad_steps,
    cumulative_grad_steps,                       # static or carry; see Risks
    target_update_freq, critic_tau,              # static (Python ints/floats)
)

nnx.update(world_model,     state_wm)
nnx.update(actor,           state_ac)
nnx.update(critic,          state_cr)
nnx.update(target_critic,   state_tg)
nnx.update(wm_opt,          state_wm_opt)
nnx.update(actor_opt,       state_ac_opt)
nnx.update(critic_opt,      state_cr_opt)
last_losses = jax.tree.map(lambda x: x[-1], last_losses_stack)  # last step's losses
```

```python
# The JIT'd scan body — graphdefs captured via closure:
@jax.jit
def _scan_grad_steps(...):
    def scan_body(carry, scan_inputs):
        (s_wm, s_ac, s_cr, s_tg,
         s_wm_opt, s_ac_opt, s_cr_opt,
         moments, key, step_idx) = carry
        batch = scan_inputs  # one slice of local_data_gpu along axis 0

        # Polyak update on target_critic (replaces the Python conditional at lines 870-878)
        do_update = (step_idx % target_update_freq) == 0
        tau = jnp.where(step_idx == 0, 1.0, critic_tau)
        # Pull Param substate, blend, write back
        online_params = filter_param_substate(s_cr)   # helper — see below
        target_params = filter_param_substate(s_tg)
        new_target = jax.tree.map(
            lambda c, t: jnp.where(do_update, (1.0 - tau) * t + tau * c, t),
            online_params, target_params,
        )
        s_tg = write_param_substate(s_tg, new_target)

        # Reconstruct trainable modules
        wm   = nnx.merge(graphdef_wm,    s_wm)
        ac   = nnx.merge(graphdef_ac,    s_ac)
        cr   = nnx.merge(graphdef_cr,    s_cr)
        tg   = nnx.merge(graphdef_tg,    s_tg)
        wm_o = nnx.merge(graphdef_wm_opt, s_wm_opt)
        ac_o = nnx.merge(graphdef_ac_opt, s_ac_opt)
        cr_o = nnx.merge(graphdef_cr_opt, s_cr_opt)

        key, k_train = jax.random.split(key)
        moments, losses = train_step(
            wm, ac, cr, tg, wm_o, ac_o, cr_o, moments, batch, k_train,
        )

        return (
            (nnx.state(wm), nnx.state(ac), nnx.state(cr), nnx.state(tg),
             nnx.state(wm_o), nnx.state(ac_o), nnx.state(cr_o),
             moments, key, step_idx + 1),
            losses,
        )

    init_carry = (state_wm, state_ac, state_cr, state_tg,
                  state_wm_opt, state_ac_opt, state_cr_opt,
                  moments, key, jnp.array(cumulative_grad_steps, dtype=jnp.int32))
    # Stack local_data_gpu along the leading axis as the scan inputs
    final_carry, losses_stack = jax.lax.scan(scan_body, init_carry, local_data_gpu)
    ...
```

**Subtleties** (developer to handle; if any is ambiguous, halt and ask):
- **`cumulative_grad_steps` as a carry**: it's a Python int *outside* the scan and gets re-bound *after* the scan completes (`cumulative_grad_steps += n_grad_steps`). Inside the scan, the Polyak-condition needs a JAX int (jnp.array) so `do_update` is a traced bool. The carry handles that.
- **`train_step`'s signature** currently takes module references; with split/merge it now receives reconstructed merges. Verify `train_step` is `@jax.jit`-friendly under this calling convention — it should be, but Step 3 should ship with a CP that runs `train_step` once outside the scan with `nnx.merge(graphdef, state)` arguments to confirm pre-scan-integration.
- **NNX `Param` vs `BatchStat` substate filtering**: the Polyak update touches only `Param`. The `filter_param_substate` / `write_param_substate` helpers above are not standard NNX API names — developer to use whatever the current flax-nnx version provides (e.g., `nnx.state(target_critic, nnx.Param)` to filter, plus `nnx.update`-shaped write-back inside JIT). See the working `train_multiple_gpu` for the form.

#### Step 3 parity test

This step **may break bit-identity** on the existing `test_grad_parity.py` — `jax.lax.scan` may compose operations differently from a Python for-loop even when math is equivalent. Tolerance must be agreed before this step lands.

- **Primary parity test**: `tests/algorithms/dreamer_srl/test_lax_scan_train.py` (new) — run the *same fixed seed* training for `n_grad_steps=4` with (a) the pre-Step-3 Python for-loop, (b) the new `lax.scan` path. Assert `nnx.state(world_model)` matches to `atol=1e-5, rtol=1e-5` after both runs. Same for actor and critic. **This is math equivalence, not bit-identity** — and that is *deliberate*; document the relaxation explicitly in the test docstring.
- **Existing `test_grad_parity.py`** — gate behind an env var (e.g., `DREAMER_SRL_BITIDENTITY=1`) so the *legacy code path* still gets exercised on CI. If the developer cannot preserve bit-identity at all under the new code path, the original Python loop should remain as a `--legacy-grad-loop` opt-in flag in `dreamer_srl_main.py` for the duration of the audit window. **The user must decide whether bit-identity loss is acceptable before this step ships.**

#### Step 3 SPS gate

- Run `bench_sps.py --label step3_scan_cpu_buffer`. Record delta vs Step 1.
- **Expected improvement**: 2–10× (Phase-3-equivalent territory, but partially gated on the buffer still being on CPU). If < 2×, the bottleneck has shifted to the CPU-side buffer sampling — that's fine; Step 4 picks it up.

#### Step 3 time estimate
- 8–12 hours: 4 hr scaffold of split/merge over all three modules + optimizers, 2 hr Polyak-update-in-JIT, 2 hr new parity test, 1 hr SPS bench, 2 hr write-up + checkpoint discussion with user re: bit-identity loss.

---

### Step 4 — Combine 2+3: default `device="gpu"`, sample inside the scan

Now flip the buffer default to `device="gpu"` and move the `buffer.sample(...)` call *inside* the scan body so we sample-and-train without any host involvement during the gradient-step block. This is the configuration equivalent to the original Dreamer's `train_multiple_gpu`.

#### Changes

**`src/algorithms/dreamer_srl/dreamer_srl_main.py`** — at the buffer-construction site (search for `SequentialReplayBuffer(`), pass `device="gpu"`. Restructure the training-loop block: instead of calling `buffer.sample(n_samples=n_grad_steps)` once and feeding the stack to `lax.scan`, sample one batch *per scan iteration* using a JAX PRNG key carried in the scan state. The scan body becomes:

```python
def scan_body(carry, _):
    (s_wm, s_ac, s_cr, s_tg, s_opts..., moments, key, step_idx) = carry
    key, sample_key, train_key = jax.random.split(key, 3)
    batch = sample_one_batch_from_gpu_buffer(
        buffer_arrays=(obs, actions, rewards, dones, is_first),  # passed via closure or scan input
        buf_pos=buf_pos, buf_full=buf_full,                       # JAX ints
        batch_size=batch_size, sequence_length=seq_len,
        key=sample_key,
    )
    # ... rest of body unchanged from Step 3 ...
```

`sample_one_batch_from_gpu_buffer` is a pure JAX function (no Python state) that takes buffer arrays + position metadata + a PRNG key and returns a sampled batch dict. It's essentially the inner body of `SequentialReplayBuffer.sample` lifted out so it's JIT-friendly.

**Buffer arrays as scan-body closure**: per the split/merge pattern §rule 2, extract `(obs, actions, rewards, dones, is_first)` as a tuple of `jnp.ndarray` *outside* the scan and reference them inside the body — do **not** access them via `self._buf[k]`. The same retracing trap that bit the original Dreamer (commit `6705735`) applies here.

**`buf_pos` and `buf_full`** — keep them as Python ints outside the JIT; pass as `jnp.array(...)` for the call. If they change between calls (they do — `_pos` grows monotonically), the JIT will retrace on each call until the buffer is full. Once full, `_pos` stabilises (still rotates but always equals `buffer_size - 1` value-wise — wait, no, it rotates). The original Dreamer's `idx` is in the same position; it accepts retracing while the buffer fills, then stabilises after fill. Document this in the change.

**Add `device: gpu` to the dreamer-srl YAML schema.** If a config key like `agent.buffer_device` doesn't exist, add it via `config.get_mandatory(...)` (no fallback default). Search the existing dreamer_srl configs (`configs/dreamer_srl/01_food_only*.yaml`) and add the key to each, defaulting to `"gpu"`. The `device="cpu"` legacy path stays in the codebase but is no longer the default.

#### Step 4 parity test

- `test_lax_scan_train.py` (the math-equivalence test from Step 3) — must still pass.
- A new test `test_gpu_mode_end_to_end.py` runs a *short* training (1 iteration, 1 grad step) with `device="gpu"` and asserts the output dict / metrics shape / dtype matches the CPU-mode counterpart from Step 3.
- Existing `test_grad_parity.py` is now strictly a legacy-path test (CPU-mode buffer + Python for-loop) — keep it green by ensuring the `--legacy-grad-loop` opt-in still works.

#### Step 4 SPS gate

- Run `bench_sps.py --label step4_scan_gpu_buffer`. Record cumulative delta vs Step 0.
- **Expected**: this is the headline number. Target is **≥ 5×** over Step 0 (matching the original Dreamer's regime). If we hit < 3×, halt and investigate before Step 5 — there's a missed lever somewhere (e.g., the sample-inside-scan isn't actually JIT'ing cleanly).

#### Step 4 time estimate
- 6–8 hours: 3 hr lift `sample` into a pure JAX function, 1 hr config plumbing, 2 hr test scaffold, 30 min SPS bench, 1.5 hr write-up.

---

### Step 5 — (optional, deferred) `lax.scan` over the env-collection iteration

Only enacted if Step 4's SPS does *not* meet the original Dreamer's collection throughput. The env-collection loop (the outer `while policy_step < total_policy_steps:` block in `dreamer_srl_main.py:525-907`) is harder to put inside `lax.scan` because env step calls are not pure — they touch Python state (env objects, reward sensors, episode bookkeeping). The original Dreamer's Phase 3 fix was for an envpool-style vectorised env that *is* JAX-pure; if dreamer-srl's env stack supports the same, port the pattern; if not, this step is deferred indefinitely.

**Gate before starting**: if Step 4 cumulative speedup ≥ 5× and the user is satisfied, **skip Step 5** and go to Step 6.

#### Step 5 time estimate (if enacted)
- 1–2 days. Plan to be drafted as a separate doc if the gate-condition triggers.

---

### Step 6 — Final benchmark + write-up

#### Tasks

1. Re-run `bench_sps.py --label step6_final` on the post-Step-4 (or post-Step-5) code with the **identical** invocation from Step 0.
2. Re-pull the `Time/sps_env` decay chart and **check whether the decay pattern from Step 0 still appears** in the new code. If yes:
   - If Step 0's verdict was "metric artefact": confirm the artefact persists (it's a metric question, not a code question) and add a side issue to fix the metric (log instantaneous SPS over a window, not cumulative).
   - If Step 0's verdict was "real throughput decline": verify it's gone; if it isn't, second-pass root-cause investigation needed (open as Issue #2 below).
3. Tabulate the cumulative speedup chain:

   | Step | Code state | Cumulative SPS (last-25%-window) | Δ vs prev | Δ vs baseline |
   |---|---|---|---|---|
   | 0 | HEAD `<sha>` | TBD | — | 1.00× |
   | 1 | + Option S | TBD | TBD× | TBD× |
   | 2 | + GPU buffer (opt-in) | TBD (parity) | 1.00× | TBD× |
   | 3 | + lax.scan (CPU buffer) | TBD | TBD× | TBD× |
   | 4 | + lax.scan + GPU buffer | TBD | TBD× | TBD× |
   | 5 (opt) | + outer-loop scan | TBD | TBD× | TBD× |

4. Write a closing "Step 6 findings" subsection in this plan doc and update the **status** in the frontmatter to `superseded` (point `superseded_by:` at any follow-on doc) or `archive` if no follow-on is needed.
5. Run `python scripts/regen_dev_index.py` to refresh `docs/develop/INDEX.md`.

#### Step 6 time estimate
- 2 hours: 30 min benchmark, 30 min WandB pull comparison, 1 hr write-up.

---

### Aggregate time estimate (Steps 0 → 6)

| Step | Estimate | Sequential cumulative |
|---|---|---|
| 0 | 1–2 hr | 1–2 hr |
| 1 | 1 hr | 2–3 hr |
| 2 | 4–6 hr | 6–9 hr |
| 3 | 8–12 hr | 14–21 hr |
| 4 | 6–8 hr | 20–29 hr |
| 5 | (optional, 1–2 days if enacted) | 20–29 hr + 8–16 hr if enacted |
| 6 | 2 hr | 22–31 hr |

Roughly **3 working days end-to-end** for Steps 0–4 + 6, with Step 5 as a possible extension.

---

## File Changes

Grouped by step. Files marked `(new)` don't yet exist; files marked `(edit)` are modified.

### Step 0
- `tests/algorithms/dreamer_srl/bench_sps.py` **(new)** — SPS micro-benchmark harness; pure-Python launcher around the trainer entrypoint with overrides for `num_envs`, `total_steps`, `seed`, `wandb.enabled=false`. Writes a CSV trace + prints summary numbers.
- This plan doc (`docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md`) **(edit)** — append "Step 0 findings" subsection.

### Step 1
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` **(edit)** — around lines 859–895, hoist H2D out of the gradient-step `for` loop (single `jax.tree.map(jnp.asarray, local_data)` call, then JAX-slice per step). ~10-line diff.

### Step 2
- `src/algorithms/dreamer_srl/buffers.py` **(edit)** — add `device: str = "cpu"` param to `__init__` (~5 lines); branch the `add` storage (`np.zeros / np.empty` vs `jnp.zeros` and `[idxes] = v` vs `.at[idxes].set(v)`) (~30 lines); branch the `sample` RNG (`np.random.Generator` vs `jax.random.randint`) and accept an optional `key` parameter (~20 lines); branch `_get_samples` (`np.take/reshape/swapaxes` vs `jnp` equivalents) (~25 lines).
- `tests/algorithms/dreamer_srl/test_gpu_buffer.py` **(new)** — math-equivalence assertions between CPU and GPU buffer modes; shape/dtype parity for `add` + `sample`.

### Step 3
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` **(edit)** — replace the Python `for i in range(n_grad_steps)` block at lines 866–895 with a `_scan_grad_steps` helper call + `nnx.update` writebacks; ~80-line restructure.
- `src/algorithms/dreamer_srl/_scan_grad_steps.py` **(new, optional)** — if the scan body is long enough to warrant its own module; otherwise inline.
- `tests/algorithms/dreamer_srl/test_lax_scan_train.py` **(new)** — `atol=1e-5, rtol=1e-5` math-equivalence test between pre-Step-3 Python for-loop path and the new `lax.scan` path; runs 4 grad steps from a fixed seed.
- `tests/algorithms/dreamer_srl/test_grad_parity.py` **(edit, conditional)** — if bit-identity cannot be preserved, gate the legacy assertions behind a `DREAMER_SRL_LEGACY_GRAD_LOOP=1` env-var path; document the relaxation.
- `dreamer_srl_main.py` **(edit)** — add a `--legacy-grad-loop` CLI flag if bit-identity preservation requires keeping the Python loop available.

### Step 4
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` **(edit)** — buffer-construction site: pass `device=config.get_mandatory('agent.buffer_device', str)`; rewire the training block to sample inside the scan body via a pure JAX `sample_one_batch_from_gpu_buffer` helper; ~30-line diff.
- `src/algorithms/dreamer_srl/buffers.py` **(edit)** — expose a free-function form of `sample` (or a `@staticmethod` that takes buffer arrays + pos/full ints + key as explicit inputs), suitable for closure-capture inside `lax.scan`. ~30 lines.
- `configs/dreamer_srl/01_food_only*.yaml` **(edit, all variants)** — add `agent.buffer_device: gpu`. No fallback default per project rule; the driver uses `config.get_mandatory(...)`.
- `tests/algorithms/dreamer_srl/test_gpu_mode_end_to_end.py` **(new)** — 1-iteration end-to-end smoke test with `device="gpu"`; assert metrics dict has expected keys + shapes.

### Step 5 (optional, deferred)
- TBD — separate plan doc to be written if Step 4's gate fails.

### Step 6
- This plan doc **(edit)** — fill the final tabulation, update frontmatter status.
- `docs/develop/INDEX.md` **(regen)** — via `python scripts/regen_dev_index.py`.

---

## Checkpoints

What the implementing agent should verify **during** implementation:

- [ ] **CP0 — bench_sps.py self-check.** Run twice with the same seed on the same node; the cumulative-SPS values should agree to within ±2%. If not, the benchmark itself is noisy and needs more warm-up / longer total-steps before any step's delta is trustworthy.
- [ ] **CP0 — WandB pull works.** The decay-chart pull from `yxij4lrc` returns rows; if not, surface the error and the dev/user pick an alternate run.
- [ ] **CP1 — bit-identity preserved.** `test_grad_parity.py` passes byte-for-byte. (`jnp.asarray` outside the loop is semantically equivalent to inside; no math change.)
- [ ] **CP2 — CPU default unchanged.** Constructing `SequentialReplayBuffer(...)` with no `device` arg → same numpy buffer as before; all existing tests green.
- [ ] **CP2 — GPU mode dtype parity.** `gpu_buf._buf[k].dtype` matches `cpu_buf._buf[k].dtype` for every key.
- [ ] **CP3 — `train_step` callable after `nnx.merge`.** Before integrating the scan, smoke-test `train_step(*nnx.merge_results, batch, key)` once at module scope; confirm it returns valid metrics. If it errors, the split/merge mapping is wrong and the scan won't help.
- [ ] **CP3 — Polyak update inside JIT.** Ensure the `jnp.where(do_update, ...)` form actually skips the update when `step_idx % freq != 0` — print `s_tg`'s mean before/after a scan with `freq=1000, n_grad_steps=10` and verify it's unchanged on 9 out of 10 steps.
- [ ] **CP3 — `cumulative_grad_steps` continuity.** After the scan, the outer Python counter advances by exactly `n_grad_steps`; verify this matches the pre-Step-3 behaviour.
- [ ] **CP4 — JIT retrace audit.** Set `JAX_LOG_COMPILES=1` for a short training run; confirm that after the buffer fills, the scan body is not retraced. If it is, the issue is `_pos` continuing to rotate as a traced-value input (the same `idx` bug commit `6705735` fixed in the original); apply the same fix.
- [ ] **CP4 — config plumbing.** `agent.buffer_device` is read via `config.get_mandatory`; missing key → `ValueError` on a removed-key smoke test.
- [ ] **CP6 — final SPS table.** Tabulate, plot, and write up.

---

## Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | **`test_grad_parity.py` byte-identity loss at Step 3+** — `lax.scan` may compose op order differently from the Python loop. | Document the relaxation explicitly; keep the legacy Python-loop path behind a `--legacy-grad-loop` opt-in for the audit window. Surface to user before Step 3 lands. |
| R2 | **`nnx.split` / `nnx.merge` overhead inside the scan body** — a Python-level reconstruction per scan iteration. The original Dreamer measured negligible overhead; ours might differ if the modules are bigger or the graphdef carries more metadata. | Measure at Step 3's SPS gate. If split/merge is the bottleneck (rare, but possible), fall back to a manual functional `train_step(params, batch, key) -> new_params, losses` rewrite that bypasses NNX entirely inside the scan. |
| R3 | **`buffer._pos` and `_full` as scan inputs** — they grow monotonically until the buffer is full, then `_pos` rotates. Passing as `jnp.array(_pos)` triggers retraces on every call until fill. | Accept the retrace cost during prefill (a known-bounded cost: log2(buffer_size) traces at most due to step-size doubling — actually more like O(buffer_size / n_grad_steps), still finite). After fill, `_pos` becomes traced-constant-equivalent (still changes but stably). If retracing dominates SPS, change `_pos` to a traced jax-int inside a class that wraps it as a State variable. |
| R4 | **`Time/sps_env` decay is real and is from a cause Step 0 doesn't find** — e.g., a Python-side accumulator in our own code we haven't searched for. | Step 0 is explicit about ruling causes in/out. If real-and-unexplained: open Issue #2 below; do **not** silently expand Option L's scope to "also fix the decay" — they're independent. |
| R5 | **The `sample_one_batch_from_gpu_buffer` lift** (Step 4) — extracting a method body into a pure JAX function risks losing a corner case the OOP form was handling (e.g., the `env_idxes is not None` write-only path, sample-next-obs branches). | The test in `test_gpu_buffer.py` (Step 2) already exercises both paths; Step 4 keeps the legacy `device="cpu"` code path callable for back-comparison; new test `test_gpu_mode_end_to_end.py` runs one full iteration end-to-end. |
| R6 | **Float64 vs float32 in JAX defaults** — `jnp.asarray(np_arr)` on a float64 numpy array yields float64 in JAX unless `JAX_ENABLE_X64=False` (the default) silently downcasts. The original Dreamer is float32 throughout. | Already enforced via `jnp.asarray(..., dtype=jnp.float32)` at line 885 (Step 1 preserves the dtype kwarg). New GPU-buffer allocation uses `jnp.zeros(..., dtype=jnp.float32)` explicitly. Add an `assert batch[k].dtype == jnp.float32` at scan-body entry (CP3) to catch slips. |
| R7 | **Cross-node measurement drift** — running Step 0 on node101 and Step 4 on node103 would invalidate the comparison. | Lock the SPS-benchmark node for the whole campaign; record it in every CSV. Use `run_command.py` consistently. |

---

## Implementation Report

> **Implemented by**: [developer]
> **Date**: [TBD]

<!-- The developer fills this section per-step. Suggested structure: -->
<!--
### Step 0
- Baseline cumulative SPS: __ env-steps/sec (node __, seed 0, 50k steps).
- Baseline instantaneous SPS (last 25%): __ env-steps/sec.
- Decay verdict: [artefact | real cause i/ii/iii/iv | other].
- Notes / deviations: __

### Step 1
- Code change: hoisted H2D, ~10 lines at lines 859–895.
- test_grad_parity.py: passes.
- bench_sps step1_option_s: cum __ inst __ (Δ vs Step 0: __×).
- Notes: __

(repeat for Steps 2-6)
-->

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [TBD]

| Step | Code change | Parity test | SPS delta | Status | Notes |
|------|-------------|-------------|-----------|:------:|-------|
| 0 | (no code) | n/a | baseline | | |
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | (optional) | | | | |
| 6 | (write-up) | | | | |

**Conclusion**: [one-line summary at end of campaign]

---

<!--
NEW ISSUES discovered during implementation/verification:
- If Step 0 surfaces a real `Time/sps_env` decay that Option L's fixes do NOT address, append as "## Issue #2: Time/sps_env real decay — root cause TBD" below with the same template structure.
- If `nnx.split`/`merge` overhead becomes the new bottleneck (R2), append as a separate issue.
-->
