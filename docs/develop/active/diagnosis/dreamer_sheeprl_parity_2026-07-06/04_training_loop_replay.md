---
title: "dreamer-srl ↔ sheeprl parity audit — Area 4: Training loop, replay buffer, optimizers, live config"
topic: dreamer
status: active
created: 2026-07-06
last_updated: 2026-07-06
phase: 2
---

# Area 4 — Training loop, replay buffer, optimizers + live config

## Purpose (plain-language entry point)

**What this is.** A fresh, line-by-line comparison of the JAX/Flax "dreamer-srl"
training-loop driver against the PyTorch reference it was ported from —
[sheeprl](https://github.com/Eclectic-Sheep/sheeprl)'s DreamerV3, pinned at commit
`33b6366`. "dreamer-srl" is the project's from-scratch JAX rebuild of DreamerV3, a
model-based RL agent. This audit covers **five surfaces**: the per-iteration
env-interaction + train loop (`dreamer_srl_main.py`), the replay buffer
(`buffers.py`), the checkpoint helpers (`checkpoint.py`), the three Adam optimizers,
and — critically — the **live YAML config values actually launched**, not just code
defaults.

**Why it exists.** The project measures agent performance in **survival steps**, and
the dreamer-srl rebuild has repeatedly landed at the random-policy floor. Prior audits
found gradient-side bugs; this one asks a narrower question: *does the training loop
feed the gradient the right data, at the right cadence, with the right recipe?* The
answer is **mostly yes at the single-environment parity config, but with several
un-logged deviations that bite hard the moment `num_envs > 1`** — which the live smoke
tests do use (the H5 verification smoke ran `num_envs=4`).

**Headline findings, in plain English.**

1. **The replay buffer is the wrong shape.** Sheeprl gives each parallel environment
   its *own* independent buffer; dreamer-srl uses **one shared buffer with a single
   write-head across all envs**. Two consequences: (a) every time *any* environment
   finishes an episode, the buffer punches a **hole (a stale/garbage row)** into every
   *other* environment's data column — I verified this empirically (a sampled sequence
   reads `[21, 31, 0, 41]` where the `0` is the hole); (b) the buffer holds
   `num_envs ×` more transitions than sheeprl's, because the reference divides the
   configured size by the environment count and dreamer-srl does not. **At `num_envs=1`
   (the live parity config) neither bites**; at `num_envs=4/16` (smokes, sweeps) both do.

2. **No gradient clipping anywhere** (confirmed: plain `optax.adam`, no clip). Sheeprl
   clips the world-model gradient at norm 1000 and actor/critic at 100. Live smoke runs
   show the world-model loss exploding to ~1e29–1e31 — the clip's absence, now with
   empirical proof.

3. **The discount factor is silently wrong.** The live config sets `gamma =
   0.996840347`; sheeprl's DreamerV3 uses `0.996996996996997`. This shortens the agent's
   effective planning horizon from ~333 steps to ~316 (~5%) and is present at **every**
   env count, including the `num_envs=1` parity launch.

4. **Prefill, replay-ratio quantization, and episode-metric bookkeeping** all deviate in
   ways that are invisible at `num_envs=1, replay_ratio=1` but distort multi-env or
   fractional-ratio runs.

**The one thing that got fixed.** The H5 fix (commit `b1dd90a`) landed correctly: the
post-`done` "zero the staged reward/flags, set is_first, write the terminal-obs reset
row" block is now **byte-for-byte the sheeprl pattern**. That surface is now PARITY.

**Bottom line.** The `num_envs=1` parity config is close to sheeprl on the gradient path
(modulo no-clip and the gamma typo), but the **deviation record (`DEVIATION_LOG.md`) is
incomplete** — none of the seven deviations below is logged there, even though several
are already written up as Fable-5 "Findings". This doc exists to make that record
complete. Full detail, with file:line on both sides, in §3–§5.

**Scope note.** The imagined-rollout math, actor/critic objective internals, RSSM, and
loss functions live in `train.py` / `loss.py` / `agent.py` and are **area 3's** job, not
this one. This audit stops at the boundary where the driver hands a sampled batch to
`train_step(...)`.

---

## 1. Classification summary

| # | Item | Verdict | In `DEVIATION_LOG.md`? | Also in Fable-5 / KNOWN_BUGS? | Bites live parity config (`num_envs=1`)? |
|---|------|---------|:---:|:---:|:---:|
| P-01 | Adam lr/eps/betas/weight_decay per module | ✅ PARITY | — | — | — |
| P-02 | Adam eps placement (inside sqrt) | ✅ PARITY (substrate) | — | — | — |
| P-03 | `Ratio` scheduler class | ✅ PARITY (line-for-line) | — | — | — |
| P-04 | Buffer ring-wrap index math (`add`) | ✅ PARITY | — | — | — |
| P-05 | `sample()` one-env-per-sequence + valid-idx exclusion | ✅ PARITY | — | — | — |
| P-06 | Polyak: before train, τ=1 first then 0.02, freq=1 | ✅ PARITY | D-011 (mechanism) | — | — |
| P-07 | Prefill uses uniform-random actions, actor not called | ✅ PARITY | — | — | — |
| P-08 | buffer-add BEFORE env.step | ✅ PARITY | — | — | — |
| P-09 | Train gate `iter_num >= learning_starts` (`>=`, not `>`) | ✅ PARITY | — | — | — |
| P-10 | `is_first` on startup=1; H5 reset-row + is_first-next block | ✅ PARITY (post-H5) | — | Finding 1 (fixed) | — |
| P-11 | reset_data content (terminal obs, actions=0, is_first=0) | ✅ PARITY | — | Finding 1 (fixed) | — |
| P-12 | `action_repeat=1`, `clip_rewards=False`, sample_next_obs=False | ✅ PARITY | — | — | — |
| P-13 | `policy_step += num_envs` (world_size=1) | ✅ PARITY | — | — | — |
| P-14 | Buffer sampling RNG unseeded (`default_rng()`) | ✅ PARITY (repro caveat) | — | — | — |
| D-01 | **No gradient clipping** (WM 1000 / actor 100 / critic 100 dropped) | 🔴 UNDECLARED (in log) | ❌ | Finding 3 + KNOWN_BUGS (OPEN) | **YES** |
| D-02 | **`gamma` mismatch** `0.996840347` vs `0.996996996996997` | 🔴 UNDECLARED | ❌ | ❌ | **YES** |
| D-03 | **Shared-pos buffer** replaces `EnvIndependentReplayBuffer` → hole rows + capacity | 🔴 UNDECLARED | ❌ (D-004 = memmap only) | ❌ | No (only `num_envs>1`) |
| D-04 | **`learning_starts` counted in iterations, not env-steps** → `num_envs×` prefill | 🟡 UNDECLARED | ❌ | Finding 6 | No (only `num_envs>1`) |
| D-05 | **Replay-ratio remainder is dead code**; `_G=max(1,·)` quantizes ratio | 🟡 UNDECLARED | ❌ (D-014 ≠ this) | Finding 4 | No (only `replay_ratio<1`) |
| D-06 | **Episode length/reward re-increment after done reset** (+1 step, +terminal reward) | 🟡 UNDECLARED | ❌ | Finding 5 | **YES** (metric only) |
| D-07 | **Train gate ignores `buffer._full`** (`_pos >= seq_len` only) | 🟢 UNDECLARED | ❌ | Finding 7 | No (only at wrap) |
| K-01 | **Checkpoint saves params only** — no optimizer state, no restore path | 🟡 DECLARED elsewhere | ❌ | KNOWN_BUGS A2 + Finding 9 | (resume only) |
| S-01 | **`DEVIATION_LOG` D-014 is stale** vs code (`ratio_steps` now subtracts) | 🟡 doc drift | stale row | — | (audit-trail) |

**Counts:** 14 PARITY · 7 UNDECLARED deviations (D-01…D-07) · 1 declared-elsewhere-but-absent-from-log (K-01) · 1 deviation-log staleness (S-01).

Of the 7 undeclared deviations, **D-01 (no clip), D-02 (gamma), D-06 (episode metric bleed)** bite the live `num_envs=1` parity config; **D-03, D-04, D-05, D-07** are dormant there but active in multi-env / fractional-ratio / long-run regimes.

---

## 2. Live config vs sheeprl YAML

Live values read from `configs/models/dreamer_srl/01_food_only.yaml` (parity track) and
confirmed against a saved run config
(`results/JAX_DreamerSRL/20260630-051517_.../models/agent_config.yaml`, `num_envs=1`).
Sheeprl values from `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` (+ its
`dreamer_v3_XS.yaml` overlay, `optim/adam.yaml`, `exp/dreamer_v3.yaml`,
`env/default.yaml`).

| Param | dreamer-srl (live) | sheeprl | Match? |
|---|---|---|---|
| WM optimizer lr / eps / weight_decay | `1e-4` / `1e-8` / 0 | `1e-4` / `1e-8` / 0 | ✅ |
| Actor optimizer lr / eps / weight_decay | `8e-5` / `1e-5` / 0 | `8e-5` / `1e-5` / 0 | ✅ |
| Critic optimizer lr / eps / weight_decay | `8e-5` / `1e-5` / 0 | `8e-5` / `1e-5` / 0 | ✅ |
| Adam betas | `(0.9, 0.999)` (optax default) | `[0.9, 0.999]` (`optim/adam.yaml`) | ✅ |
| **WM clip_gradients** | **none** (plain `optax.adam`) | **`1000.0`** | ❌ **D-01** |
| **Actor clip_gradients** | **none** | **`100.0`** | ❌ **D-01** |
| **Critic clip_gradients** | **none** | **`100.0`** | ❌ **D-01** |
| **gamma** | **`0.996840347`** | **`0.996996996996997`** | ❌ **D-02** |
| lmbda | `0.95` | `0.95` | ✅ |
| horizon | `15` | `15` | ✅ |
| unimix (consumed in `agent.py`, area 2) | `0.01` | `0.01` | ✅ |
| replay_ratio | `1` | `1` | ✅ |
| learning_starts | `1024` (**as iterations**, see D-04) | `1024` (`// policy_steps_per_iter` → iterations) | ⚠️ D-04 |
| per_rank_pretrain_steps | `0` | `0` | ✅ |
| per_rank_batch_size | `16` | `16` (`exp/dreamer_v3.yaml`) | ✅ |
| per_rank_sequence_length | `64` | `64` (`exp/dreamer_v3.yaml`) | ✅ |
| critic tau | `0.02` | `0.02` | ✅ |
| per_rank_target_network_update_freq | `1` | `1` | ✅ |
| ent_coef | `3e-4` | `3e-4` | ✅ |
| moments decay / max / pct low / high | `0.99` / `1.0` / `0.05` / `0.95` | `0.99` / `1.0` / `0.05` / `0.95` | ✅ |
| kl_dynamic / representation / free_nats / regularizer | `0.5` / `0.1` / `1.0` / `1.0` | `0.5` / `0.1` / `1.0` / `1.0` | ✅ |
| continue_scale_factor | `1.0` | `1.0` | ✅ |
| buffer.size | `1_000_000` (256k in variants) | `1_000_000` (`exp`) **then `// num_envs`** | ⚠️ D-03 |
| num_envs (parity launch) | `1` | `1` (atari default) | ✅ |
| precision / dtype | JAX float32 (default) | Fabric `32-true` + `float32_matmul_precision: high` | ✅ (substrate) |

**Note on the gamma comment.** `01_food_only.yaml` annotates the value `# sheeprl
dreamer_v3.yaml:L22`, but sheeprl's gamma lives at **line 11** and reads
`0.996996996996997`. The comment cites both the wrong line and (implicitly) the wrong
value. Provenance of `0.996840347` traces to `CP9_PLAN.md:441` — it was inlined by hand
and never matched the reference. **D-02.**

---

## 3. Findings in detail (file:line, both sides)

### D-01 — No gradient clipping on any optimizer 🔴 UNDECLARED (relative to `DEVIATION_LOG`)

- **Ours:** `dreamer_srl_main.py:658-660` builds all three optimizers as plain
  `nnx.Optimizer(module, optax.adam(lr, eps=eps), wrt=nnx.Param)`. There is no
  `optax.clip_by_global_norm` anywhere in the chain, and the per-step update in
  `train.py` (`wm_opt.update(...)`, `actor_opt.update(...)`, `critic_opt.update(...)` at
  `train.py:797/954/982`) applies raw gradients.
- **Sheeprl:** `dreamer_v3.py:193-199` (WM, `max_norm=cfg.algo.world_model.clip_gradients`
  = **1000**), `:300-303` (actor, **100**), `:320-326` (critic, **100**), via
  `fabric.clip_gradients(..., error_if_nonfinite=False)`. Config: `dreamer_v3.yaml:52`
  (WM), `:127` (actor), `:154` (critic).
- **Status:** Documented as **Finding 3** in
  [[04_dreamer_srl]] and as an **OPEN — priority raised** row in KNOWN_BUGS, but **absent
  from `DEVIATION_LOG.md`** (D-001…D-014 have no clip row). The deviation record is
  therefore incomplete.
- **Impact — empirically confirmed.** `tmp/20260706_h5_speed_verify.log` (a `num_envs=4`
  smoke) shows `world_model_loss` reaching `2.2e29`, `1.4e29`, `2.2e29` at several log
  points before recovering — exactly the divergence a gradient clip is designed to
  prevent. Bites at **all** env counts. **HIGH.**

### D-02 — `gamma` mismatch 🔴 UNDECLARED

- **Ours:** `01_food_only.yaml` → `algo.gamma: 0.996840347`; loaded at
  `dreamer_srl_main.py:513` (`gamma = agent_cfg.get_mandatory("algo.gamma", float)`) and
  baked into `make_train_step(..., gamma=gamma, ...)` (`:681`).
- **Sheeprl:** `dreamer_v3.yaml:11` → `gamma: 0.996996996996997`.
- **Impact.** Effective horizon `1/(1-γ)`: sheeprl ≈ **333** steps, ours ≈ **316.5** steps
  — a ~5% shorter discount horizon on the value/λ-return targets. Present at **every** env
  count including the `num_envs=1` parity launch. Small but systematic, and it silently
  breaks the "same recipe as the benchmark" claim the whole parity effort rests on. **MED.**

### D-03 — Shared-position buffer replaces `EnvIndependentReplayBuffer` 🔴 UNDECLARED

- **Ours:** `dreamer_srl_main.py:667-672` constructs a single
  `SequentialReplayBuffer(buffer_size, n_envs=num_envs, ...)`. That class
  (`buffers.py:35-299`) stores each key as one array of shape `[buffer_size, n_envs, ...]`
  with a **single shared `_pos` / `_full`** advanced for the whole array on every `add`
  (`buffers.py:297-299`).
- **Sheeprl:** `dreamer_v3.py:479-485` constructs an
  **`EnvIndependentReplayBuffer`** (`data/buffers.py:529-590`) that holds **`n_envs`
  separate `SequentialReplayBuffer`s, each with `n_envs=1` and its own `_pos`**, and sizes
  each at `buffer_size = cfg.buffer.size // (num_envs * world_size)` (`dreamer_v3.py:478`).
  Its `add(data, indices)` routes each env-column to its own sub-buffer
  (`data/buffers.py:645-654`), and its `sample` allocates the batch across sub-buffers via
  `bincount` (`data/buffers.py:683-699`).

  Two distinct semantic consequences follow from collapsing that into one shared-pos array:

  **(a) Hole/stale rows at every reset write (`num_envs>1`).** When a subset of envs
  finish, the driver writes the second "reset row" via
  `buffer.add(reset_data, done_mask=dones)` (`dreamer_srl_main.py:1272`). In `buffers.py`
  the `done_mask` path (`:224-230`) slices to the done columns *but still advances the
  shared `_pos` by 1 for the whole array* (`:297-299`) — so the **non-done columns get no
  write at that time-slot and retain uninitialized / stale ring-buffer bytes**. I verified
  this directly:

  ```
  # n_envs=2; env0 done, env1 not; reset row written with done_mask=[True, False]
  env1 sampled sequence rows 1–4 (obs): [21.0, 31.0, 0.0, 41.0]
                                                    ^^^ hole (env1 never written at that slot)
  ```

  In sheeprl the reset write (`rb.add(reset_data, dones_idxes)`, `dreamer_v3.py:650`) goes
  **only** to the done envs' independent buffers; env1's write head never moves, so its
  sequence stays contiguous `[21, 31, 41, …]`. The dreamer-srl world model, sampling a
  sequence that straddles such a slot for a non-done env, trains on a garbage transition.

  **(b) Capacity / recency.** Because the `// num_envs` division is dropped, dreamer-srl
  retains `num_envs ×` more transitions than sheeprl (e.g. at `num_envs=4`, sheeprl keeps
  `256000/4 = 64000` per env vs dreamer-srl's `256000` per env column), so the sampled
  data distribution is older/staler than the reference's.
- **Status:** `DEVIATION_LOG` D-004 covers only the *memmap* omission; the
  `EnvIndependentReplayBuffer → shared-pos` structural substitution and its hole-row
  behaviour are **not logged anywhere**.
- **Impact.** Dormant at `num_envs=1` (single column, no non-done neighbours, `//1`
  no-op). Active at `num_envs>1` — the H5 smoke ran `num_envs=4`, and the recompile-storm
  fixes in this file exist specifically for `num_envs=16` desync, so multi-env is a real
  operating regime. **HIGH (structural) at `num_envs>1`.**

### D-04 — `learning_starts` counted in iterations, not env-steps 🟡 UNDECLARED

- **Ours:** `dreamer_srl_main.py:1131` gates prefill with `if iter_num <= learning_starts`,
  where `learning_starts` is the raw config value `1024` (`:484`). Each iteration is
  `num_envs` env-steps, so prefill length in env-steps is `1024 * num_envs`.
- **Sheeprl:** `dreamer_v3.py:510` `learning_starts = cfg.algo.learning_starts //
  policy_steps_per_iter` — i.e. the `1024` config value is **divided by `num_envs`** to
  get an iteration count, so prefill is always ~`1024` env-steps regardless of env count.
- **Impact (measured):** `num_envs=1` → 1024 vs 1024 env-steps (**parity**); `num_envs=4`
  → **4×** longer prefill; `num_envs=16` → **16×** (16384 vs 1024 env-steps). Finding 6.
  Dormant at the parity config. **MED at `num_envs>1`.**

### D-05 — Replay-ratio remainder is dead code; `_G = max(1, ·)` quantizes the ratio 🟡 UNDECLARED

- **Ours:** `dreamer_srl_main.py:1064` sets `_G = max(1, int(replay_ratio * num_envs))` and
  `:1065` a `_grad_step_remainder = 0.0`. At `:1549-1554` the scan path accumulates
  `_grad_step_remainder += n_grad_steps - _G` but then **always** runs exactly `_G` steps
  (`n_grad_steps_scan = _G`); the remainder is **never read back** to add/skip a step.
- **Sheeprl:** `dreamer_v3.py:662` uses the exact `Ratio` return
  (`per_rank_gradient_steps = ratio(ratio_steps / world_size)`) with no quantization.
- **Impact (measured):** At `replay_ratio=1, num_envs=1` → `_G=1`, steady-state
  `Ratio` return ≈1, remainder ≈0 → **parity**. But at a fractional `replay_ratio=0.5,
  num_envs=1`, `int(0.5)=0` and `max(1, 0)=1` forces **1 grad step/iter (2× the intended
  0.5)**, and the dead remainder cannot correct it. This matters for the replay-ratio
  sweep work ([[dreamer_replay_ratio_sweep]]). Finding 4. **MED for `replay_ratio<1`.**

### D-06 — Episode length/reward re-increment after the done reset 🟡 UNDECLARED (metric-only)

- **Ours:** at a done, `dreamer_srl_main.py:1213-1214` logs `ep_len = episode_lengths[i]+1`
  and `ep_rew = episode_rewards[i]+rewards[i]` (correct for the finished episode), then
  `:1291-1292` zeros the counters. But **after** the done block, `:1526-1527` unconditionally
  runs `episode_lengths += 1` / `episode_rewards += rewards` for **all** envs including the
  just-reset ones — so the done-transition (which belongs to the *old* episode) also
  increments the *new* episode's counter and adds the old terminal reward into it.
- **Sheeprl:** episode stats come from the env wrapper's `final_info["episode"]`
  (`dreamer_v3.py:610-618`); there is no driver-side running counter to double-count.
- **Impact.** Every logged episode length is **+1 too high** and every logged episode
  reward carries the predecessor's terminal reward. Because the project's headline metric
  is **survival steps**, this is a systematic bias on the number that decides success
  (~1% at 100-step episodes). Finding 5. Bites at **all** env counts. **MED (reported
  metric, not the gradient).**

### D-07 — Train gate ignores `buffer._full` 🟢 UNDECLARED

- **Ours:** `dreamer_srl_main.py:1558` gates training on `n_grad_steps > 0 and buffer._pos
  >= seq_len`. Once the ring buffer wraps (`_full=True`), `_pos` cycles back toward 0, so
  for `seq_len` iterations after each wrap `_pos < seq_len` and training is **skipped** even
  though the buffer is full of valid data. The `Ratio._prev` has already advanced, so those
  owed gradient steps are lost (not repaid).
- **Sheeprl:** no such gate — it relies on `learning_starts` guaranteeing enough data, and
  the per-env buffers never present a below-threshold `_pos` mid-run.
- **Impact.** At `buffer_size=256k–1M` this fires for `seq_len=64` iterations once per
  full-buffer cycle — negligible in practice, but a latent correctness papercut. Correct
  gate would be `(buffer._full or buffer._pos >= seq_len)`. Finding 7. **LOW.**

### K-01 — Checkpoints save params only; no optimizer state; no restore path 🟡 DECLARED elsewhere

- **Ours:** `checkpoint.py:84-96` stores `nnx.state(module, nnx.Param)` for WM/actor/
  critic/target-critic plus moments + bookkeeping scalars — **no optimizer state** (Adam
  `mu`/`nu` momentum). `load_checkpoint` exists (`:113-123`) but is **never called** in
  `dreamer_srl_main.py` — there is no `--resume` wiring and no `checkpoint.resume_from`
  branch.
- **Sheeprl:** `dreamer_v3.py:741-755` saves `world_optimizer.state_dict()`,
  `actor_optimizer.state_dict()`, `critic_optimizer.state_dict()`, `moments`, `ratio`,
  `iter_num`, `last_log`, etc., and the top of `main` (`:366-367, :453-520`) fully restores
  optimizer + ratio + buffer + counters.
- **Status:** Declared in **KNOWN_BUGS (A2, OPEN)** and **Finding 9**, but **absent from
  `DEVIATION_LOG.md`**. **MED (resume / continual-learning correctness).**

### S-01 — `DEVIATION_LOG` D-014 is stale vs the code 🟡 doc drift

- **`DEVIATION_LOG.md` D-014** states the JAX driver uses `ratio_steps = policy_step` and
  *omits* sheeprl's `prefill_steps * policy_steps_per_iter` subtraction (`dreamer_v3.py:661`).
- **The code now subtracts:** `dreamer_srl_main.py:1540` reads
  `ratio_steps = policy_step - learning_starts * num_envs`, with a "D-014 fix" comment at
  `:1534-1539`. So the deviation-log row **no longer describes the code**. Moreover the
  subtracted quantity differs from sheeprl's: sheeprl subtracts
  `prefill_steps * policy_steps_per_iter` where `prefill_steps = learning_starts_iters - 1`
  (`dreamer_v3.py:511`), i.e. it has an intentional off-by-one and uses the *iteration*
  count; ours subtracts `learning_starts(config) * num_envs`. Because ours already
  mis-reads `learning_starts` as iterations (D-04), the two never agree except at
  `num_envs=1` where both reduce to `policy_step - 1024`. **Audit-trail hygiene: update
  the D-014 row to match the code, and cross-reference D-04.**

---

## 4. PARITY confirmations (spot-checked, both sides)

- **Optimizers (P-01/02):** lr/eps/betas/weight_decay all match (table §2);
  `optax.adam` places `eps` inside the sqrt denominator like `torch.optim.Adam` (documented
  substrate boundary). `dreamer_srl_main.py:658-660` ↔ `dreamer_v3.py:448-452` + `optim/adam.yaml`.
- **`Ratio` class (P-03):** `utils.py:270-333` is a line-for-line port of
  `sheeprl/utils/utils.py:259-291` — same `_prev is None` first-call branch, same
  `int((step - self._prev) * ratio)` + `self._prev += repeats / self._ratio` self-correction.
- **Buffer ring-wrap (P-04):** `buffers.py:207-217` (`next_pos`, wrap-idxes, oversized-data
  slice) is identical to `data/buffers.py:193-217` (`ReplayBuffer.add`).
- **`sample()` (P-05):** `buffers.py:403-431` + `_get_samples:516-558` reproduce sheeprl's
  valid-index exclusion of the `[_pos-seq_len, _pos)` chunk and the one-env-per-sequence
  tiling (`data/buffers.py:419-465, 476-526`).
- **Polyak (P-06):** legacy loop `dreamer_srl_main.py:1612-1620` and scan body `:992-1011`
  both update the target critic **before** `train_step`, with `tau=1.0` on the first grad
  step and `critic_tau=0.02` after, at `per_rank_target_network_update_freq=1` — matching
  `dreamer_v3.py:674-680`.
- **Prefill (P-07):** `dreamer_srl_main.py:1131-1138` samples uniform-random one-hot actions
  and does **not** call `player.get_actions` during prefill — matching `dreamer_v3.py:558-571`.
- **buffer-add-before-step (P-08):** `buffer.add(step_data)` at `:1145` precedes `env.step`
  at `:1150`, matching `dreamer_v3.py:587` before `:589`.
- **Train gate `>=` (P-09):** `:1532` `if iter_num >= learning_starts` ↔ `dreamer_v3.py:660`.
- **H5 reset block (P-10/11):** `_reset_terminal_step_data` (`:342-360`) zeros
  rewards/terminated/truncated and sets `is_first=1` for done columns, and the reset_data
  write (`:1263-1272`) carries the true terminal obs with `actions=0, is_first=0` — a
  faithful port of `dreamer_v3.py:645-656`. **This is the H5 fix (`b1dd90a`) and it is
  correct.**
- **action_repeat / clip_rewards / sample_next_obs (P-12):** all default to the sheeprl
  food-task values (1 / False / False).

---

## 5. Verdict + unverifiable list

**Verdict.** The `num_envs=1` parity config feeds the gradient a recipe that is close to
sheeprl **except** for (i) the missing gradient clip (D-01, empirically load-bearing —
WM loss to 1e29), (ii) the `gamma` typo (D-02), and (iii) the +1 survival-step / reward
bleed in the reported episode metric (D-06). The buffer, prefill-length, and
replay-ratio-quantization deviations (D-03/D-04/D-05) are **dormant at `num_envs=1,
replay_ratio=1` but active in every multi-env or fractional-ratio run** — including the
smokes that actually ran (`num_envs=4`). None of the seven deviations is recorded in
`DEVIATION_LOG.md`; three (D-01, D-04, D-05, D-06, D-07) exist as Fable-5 Findings and one
(K-01) as a KNOWN_BUGS row, but the parity deviation record itself is incomplete, and one
existing row (D-014, S-01) is stale.

**Recommended (for `developer`, not applied here):**
1. Add `optax.clip_by_global_norm` (1000/100/100) — D-01, highest priority (empirical divergence).
2. Fix `gamma` to `0.996996996996997` — D-02, one-character config change, affects the parity launch.
3. Decide the buffer question — either restore per-env buffers / the `//num_envs` sizing or
   fix the shared-pos hole-row write — before any `num_envs>1` parity claim (D-03).
4. Reconcile `learning_starts` iteration-vs-env-step semantics (D-04) and the D-014 log row (S-01).
5. Move the D-06 `episode_lengths += 1` / `episode_rewards += rewards` update into the
   non-done branch (or skip just-reset envs) so survival-steps is unbiased.
6. Log D-01…D-07 in `DEVIATION_LOG.md` for record completeness.

**Unverifiable from static read (need a live A/B or are substrate-class):**
- The **aggregate training-time impact** of D-01+D-02+D-06 at `num_envs=1` (needs a paired
  run vs sheeprl at matched seed/recipe — the standing parity gate).
- **float32 matmul precision** parity: sheeprl runs Fabric `32-true` +
  `float32_matmul_precision: high` (TF32 on Ampere); JAX's default GPU matmul precision may
  differ per-op. Substrate-class (D-007/D-008 family), not resolvable by reading source.
- Whether any **production parity launch used `num_envs>1`** — the one saved config I
  inspected was `num_envs=1`, but the H5 smoke used `num_envs=4`; the set of launched
  configs was not exhaustively audited here.
- The exact **hole-row contents in a wrapped/full buffer** — my probe showed zeros/garbage
  in a fresh buffer (`np.empty` tail); in a filled buffer the hole would carry *stale valid
  data* from a prior cycle, which is harder to detect and arguably worse. Not exhaustively
  characterized.

---

*Reviewed by: code-reviewer — Area 4 of the 2026-07-06 dreamer-srl ↔ sheeprl parity audit.
Read-only; no source, config, or `DEVIATION_LOG` edits made.*
