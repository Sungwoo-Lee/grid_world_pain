# Diagnosis findings — dreamer_srl training loop + replay buffer (2026-07-23)

Unit: `src/algorithms/dreamer_srl/train.py` + `src/algorithms/dreamer_srl/buffers.py`.
Because train.py holds only the JIT'd step functions, the actual training loop
(`src/algorithms/dreamer_srl/dreamer_srl_main.py`) was read in full as well — the
FIXED rows under regression check live there. Supporting reads: `utils.py`, `loss.py`,
`agent.py` (action_shift / RSSM.dynamic / observe / imagine / Actor), `checkpoint.py`,
`vendor/sheeprl/sheeprl/data/buffers.py`. Read-only Python simulations were run against
the real buffer classes (driver-faithful write pattern; 35k+ sampled sequences on the
CPU path, 3k+ on the GPU path).

Severity: P0 = corrupts training now; P1 = wrong results in realistic configs;
P2 = latent / quality.

---

## New findings

### F1 [P2] dreamer_srl_main.py:1799-1812 — Ratio debt silently consumed while the buffer is not yet sequence-ready

**Claim.** On every iteration past `learning_starts`, `n_grad_steps = ratio(ratio_steps)`
is called (line 1800) *before* the `buffer.ready_to_sample(seq_len)` gate (line 1812).
`Ratio.__call__` advances its internal `_prev` accumulator by `repeats / ratio`
(utils.py:393-394) the moment it returns, so if the ready gate then fails, the owed
gradient steps are permanently dropped — they are never repaid once the buffer becomes
sampleable.

**Concrete failure scenario.** `configs/models/dreamer_srl/01_food_only_M_seqlen128.yaml`:
`learning_starts=1024`, `per_rank_sequence_length=128`, `replay_ratio=1`, and
`training.num_envs=16` (configs/train/dreamer_srl.yaml:36). Prefill ends at iteration
`1024 // 16 = 64`, but each per-env sub-buffer only reaches `_pos >= 128` at iteration
~128 (one row per iteration, plus done extras). Iterations 64..~127 each consume
16 owed grad steps from the Ratio scheduler and execute none -> ~1000 gradient steps
silently lost at the start of the run, and `Params/effective_replay_ratio` starts
depressed. Upstream sheeprl has no such gate — its `rb.sample()` would raise
`ValueError: Cannot sample a sequence of length 128 ...`, surfacing the
learning_starts/seq_len misconfiguration loudly instead of masking it.

**Evidence.** utils.py:373-395 (Ratio advances `_prev` unconditionally on positive
return); dreamer_srl_main.py:1789-1812 (call order); config values above. Transient
(bounded by `seq_len - learning_starts_iters` iterations), hence P2 not P1.

**Fix direction.** Call `ratio()` only when `buffer.ready_to_sample(seq_len)` is true
(or warn/fail at startup when `learning_starts_cfg // num_envs < seq_len`).

### F2 [P2] dreamer_srl_main.py:1969-1985 — legacy logging path: prefill-phase episodes withheld until first train log; unbounded growth with --no-wandb

**Claim.** On the legacy path (`logging_cfg is None`), episode rows are emitted only
inside the step-log gate, which requires `bool(last_losses)` (line 1970). During the
entire prefill phase `last_losses == {}`, so every completed episode accumulates in
`iteration_episodes` and is emitted as one giant averaged row at the first
post-training log — early-training Episode/* curves get a distorted first point whose
window is much larger than `log_every`. Additionally, the clear happens only under
`use_wandb` (lines 1983-1985): with `--no-wandb`, `iteration_episodes` grows without
bound for the whole run (memory-only, no metric impact).

**Failure scenario.** Any legacy-logging config with a long prefill and short
episodes (e.g. num_envs=16, 1024-step prefill, ~30-step survival at init -> hundreds
of buffered episodes averaged into row one).

**Fix direction.** Decouple episode emission from `bool(last_losses)` on the legacy
path (as the two-level path already does), and clear the list regardless of
`use_wandb`. Two-level logging path is unaffected.

### F3 [P2, latent] buffers.py:224-225 — sheeprl-inherited wrong slice when a single add() exceeds capacity

**Claim.** `data_to_store = {k: v[-self._buffer_size - next_pos:]}` keeps the last
`buffer_size + next_pos` elements, while `idxes` (line 221) has
`buffer_size - pos + next_pos` entries — a length mismatch that would raise a numpy
broadcast error (and `idxes` itself contains duplicate indices when
`data_len > buffer_size`). Faithful port of the same bug in upstream sheeprl
(vendor/sheeprl/sheeprl/data/buffers.py:200).

**Failure scenario.** Only fires when a single `add()` carries more rows than the
buffer capacity. The driver always adds `data_len == 1` (dreamer_srl_main.py:1371,
1508), so this is unreachable today. Recorded so a future bulk-writer doesn't trip it.

**Fix direction.** Slice to `v[-len(idxes):]` (or assert `data_len <= buffer_size`).

### F4 [P2, declared deviation — note only] dreamer_srl_main.py:1224 — D-015 quantization floors fractional replay ratios

`_G = max(1, int(replay_ratio * num_envs))` truncates: ratio 0.3 x 16 envs -> 4
grad steps/iter instead of the owed 4.8 (-17% effective replay ratio vs. config).
This is the declared deviation D-015 ("fractional replay ratios are QUANTIZED on
this path") and is observable via `Params/effective_replay_ratio`; recorded here only
so future fractional-ratio experiments don't misattribute the gap. Exact cadence is
available via `--legacy-grad-loop`. Not a defect.

### F5 [P2, cosmetic] dreamer_srl_main.py:1710/1750 — curriculum checkpoint eval passes stage-0 `env_cfg` as `config`

In curriculum mode `env_cfg` stays bound to `schedule.stage_configs[0]` forever; the
checkpoint-triggered eval receives the *current* stage's `env_params` (correct env
dynamics — the "continual eval rebuilt stage-0 env" fix holds) but `config=env_cfg`.
eval.py uses `config` only for `visualization.icons` / `source_path` (eval.py:82-88,
379-385), so the only effect is stage-0 icon styling on post-swap eval videos when
stages define different icons. No metric impact.

---

## Fixed-bug regression check

| Fixed row | Verdict | Evidence |
|---|---|---|
| Shared replay write-head corruption (D-03) | **Holds** | Driver constructs `EnvIndependentSequentialReplayBuffer` (dreamer_srl_main.py:775) with sheeprl sizing `buffer.size // num_envs` (:749) + `validate_per_env_capacity` fail-fast; done-boundary `add(done_mask=...)` routes rows only to done envs' sub-buffers (buffers.py:832-839) — non-done heads do not move. Simulated 2-env run with desynchronized deaths: per-env streams show no hole rows, correct pos advance (+2 on done iterations). The dangerous partial-column `done_mask` branch in the plain buffer is retained but unreachable multi-env: GPU buffer mode is hard-guarded to `num_envs == 1` (dreamer_srl_main.py:760-765), where it is semantically safe (verified: non-done -> early return, pos unchanged). |
| Episode logging inherited step + prior death reward (P5/C1) | **Holds** | `_advance_episode_counters` (:380-406) skips envs done this iteration and no-ops on stage-swap iterations; done block counts the terminal transition exactly once (`ep_len = counter+1`, `ep_rew = counter + rewards[i]`, :1443-1444) then zeroes counters (:1527-1528). Terminal-step behavior events accumulate *before* the done block reads them (:1395-1399) — included exactly once. |
| Replay leaked prior episode's death into new episode (H5) | **Holds** | `_reset_terminal_step_data` (:359-377) zeroes staged rewards/terminated/truncated and sets is_first for done envs, called at :1524 — *after* `buffer.add(reset_data, ...)` at :1508 copies the terminal values (reset_data aliases the very arrays being zeroed; numpy assignment inside add() copies first, so the ordering is load-bearing and currently correct). Simulation confirms the post-death row is (fresh reset obs, r=0, term=0, is_first=1) and the terminal row is (terminal obs, action=0, r_term, term=1, is_first=0) — exact sheeprl two-row semantics. The `step_data["obs"]` -> `next_obs` view aliasing (autoreset mutates next_obs in place at :1560) is intentional and verified to propagate the fresh obs into the next add. |
| Prefill counted in iterations not env steps (P6) | **Holds** | `derive_prefill` (utils.py:62-82) = `cfg // num_envs` with sheeprl's intentional off-by-one on `prefill_steps`; random-action gate `iter_num <= learning_starts` (:1357) and train gate `iter_num >= learning_starts` (:1789) reproduce sheeprl's inclusive overlap; `ratio_steps = policy_step - prefill_steps * num_envs` (:1799) is line-for-line sheeprl dreamer_v3.py:508-511/660-661 at world_size=1. Hand-checked at learning_starts_cfg=1024, num_envs=16: 64 prefill iterations = exactly 1024 env steps; first Ratio call sees step=16 -> 16 grad steps. |
| Gradient clipping (P1) | **Holds** | All three optimizers built via `make_optim_tx` = `optax.chain(clip_by_global_norm, adam)` (:734-736; utils.py:50-55) with WM 1000 / actor 100 / critic 100, matching sheeprl's clip values and clip-then-step order. |
| Persistent-compilation fixes | **Holds** | (a) per-iteration recompile: single `@jax.jit _scan_grad_steps` with graphdefs closed over, built once (:1121-1207); (b) per-step re-upload: one `jax.tree.map(jnp.asarray)` H2D per iteration (:1850-1853); (c) reset-storm: fixed-width `done_mask` player reset (:251-271, :1514), fixed-width reset_data + buffer done_mask add (:1499-1508), constant-shape `jax.random.split(k, num_envs)` autoreset keys (:1547-1548), constant scan length `_G` (Fix 2, :1224). |
| Recon loss half-weighted + missing symlog (WP-SRL P3) | **Holds** | `SymlogDistribution` (loss.py:49-87): no 0.5 factor, decoder output treated as symlog-space prediction, `tol=1e-8` clamp present; wired in train.py:720. |
| REINFORCE resampled actions (v1) | **Holds** | Actor loss computes `sum(sg(rollout_action) * log_softmax(forward_logits(sg(latent))))` (train.py:908-948) — rollout actions threaded from `imagine()` (train.py:843), no PRNG and no resample inside the loss. |
| Curriculum-swap counter skew | **Holds** | `_stage_swapped_this_iter` set in the swap block (:1614) and honored by `_advance_episode_counters` (:1783-1784); swap wipes counters, behavior/dist/BM accumulators, rolling windows, and re-syncs the staged step_data row (:1593-1601) so the cleared buffer's first row pairs the new stage's obs with is_first=1. |
| Resume fixes ("resume silently restored nothing"; "continual resume ran wrong stage's world") | **Not regressable in this unit** | `dreamer_srl_main.py` contains no resume/restore path at all (grep: zero hits); checkpoint.py explicitly declares restore wiring out of scope (:78-80). Those fixes live in the legacy stack. The OPEN checkpoint-momentum row (checkpoint.py:85-88) is owned elsewhere and is not re-reported. |

---

## Reviewed but clean

- **Ring-buffer wrap-around sampling** (buffers.py:379-441): re-derived the
  `valid_idxes` exclusion (`[0, pos-L] U [pos, size-1 or size+pos-L]`) — windows can
  wrap size-1 -> 0 (temporally adjacent in a full ring) but can never straddle the
  pos-1 -> pos write-head discontinuity; verified empirically over 31,936 CPU-path and
  3,168 GPU-path sampled sequences across many ring laps: zero non-contiguous, zero
  stale/future reads. `sequence_length == buffer_size` edge and post-wrap
  `ready_to_sample` (P8) both OK.
- **One-step shift semantics**: buffer row t = (obs_t, action-taken-FROM-obs_t,
  reward-received-ON-ARRIVAL-at-obs_t, terminated-at-obs_t, is_first_t) — verified by
  simulation; `action_shift` (agent.py:171-213) then feeds RSSM step t with a_{t-1};
  reward head trains against the arrival-state reward; §S5 splice uses per-latent
  `terminated` with `[T,B] -> [BT]` row-major flatten consistent with the posterior
  reshape (train.py:827-873). All sheeprl-exact.
- **is_first placement**: force-set `is_first[0]=1` at train time (train.py:690);
  §S4 three-quantity arithmetic-mask reset in RSSM.dynamic (agent.py:1037-1071);
  is_first=1 row written with the fresh reset obs; reset_data row carries is_first=0.
- **Multi-env interleaving**: per-env sub-buffers; `_get_samples` env tiling keeps
  each sequence inside one env column; bincount batch allocation + axis-2 concat
  (buffers.py:845-883) matches sheeprl EnvIndependentReplayBuffer.
- **Actor/critic/lambda math** (train.py): discount cumprod/gamma with §S5 splice;
  two-term critic NLL against un-normalised lambda-targets; per-term Moments
  normalization with offset cancellation preserved; polyak-before-train ordering
  (both legacy loop and scan body, with tau=1 hard copy at step 0); moments EMA
  updated before use (sheeprl Moments.forward order). Live critic for
  lambda-bootstrap/baseline, target critic only in the two-term loss.
- **Terminated vs truncated split** (dreamer_srl_main.py:1389-1391): reason>=2 ->
  terminated, ==1 -> truncated; truncations bootstrap through §S5 (continue=1) —
  correct DreamerV3 semantics for the survival metric.
- **PRNG threading**: k_player / autoreset keys / sample_key / scan carry key all
  split from the single loop key with no reuse; `_scan_grad_steps` returns the
  advanced key which replaces the outer key on the scan path; legacy path splits
  per grad step.
- **Replay-ratio accounting** (integer ratio x num_envs case): Ratio debt formula
  exact; `cumulative_grad_steps` advanced by the actually-executed count on both
  paths; `Params/effective_replay_ratio` computed from real counts.
- **Device transfers**: single batched H2D per train iteration; GPU buffer mode
  keeps data resident; `v[i]` slices inside the legacy loop are device-side.
- **Stage-swap hygiene**: env rebuild -> full reset -> staged-row re-sync -> player
  reset -> buffer reset -> accumulator wipe -> tag-roster rebuild, in a safe order;
  `_emit_episode_row` closure reads current tag bindings (documented hazard, intact).
- **flax/nnx API details**: `nnx.sigmoid` exists (checked); optimizer state
  round-trips through split/merge/update in the scan carry (Adam moments preserved
  across iterations in-process).
