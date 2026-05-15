---
title: "dreamer-srl v3 — CP3b spec: buffer + cadence state-evolution gate"
topic: dreamer
status: active
created: 2026-05-13
last_updated: 2026-05-14
phase: 2
---

# dreamer-srl v3 — CP3b spec: buffer + cadence state-evolution gate

## Plain-language entry point

How the replay buffer stores transitions, how training-time samples are drawn from
it, and how often gradient updates fire per environment step are upstream of every
single algorithmic checkpoint that follows in the dreamer-srl v3 rebuild. If JAX
stores `is_first` at a slightly different time index than sheeprl, or if the JAX
cadence wiring fires gradient updates at 1/16 the rate of sheeprl's even though
both YAMLs say `replay_ratio: 1.0`, no amount of bit-identity in the two-hot encoder
or the RSSM transition will recover parity at the parity gate — the world model
trains on a different data distribution served at a different rate. The original
v3 sketch listed this layer as "no Lever-A gate; integration smoke only"; this
spec promotes it to a full checkpoint (CP3b) because the 16× `replay_ratio` drift
documented in [SPS_COMPARISON_JAX_VS_SHEEPRL.md §3.6](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#36-where-the-jax-advantage-is-coming-from)
was the exact silent-divergence class this v3 plan exists to prevent — and it
slipped through anyway because the layer had no gate.

**CP3b is state-evolution bit-identity, not pure-function bit-identity.** Pure-
function identity (drive both implementations with the same RNG, assert byte-
identical output) is impossible in this layer because PyTorch and JAX have
different PRNG streams (deviation D-002 class — see [DEVIATION_LOG.md](DEVIATION_LOG.md)).
State-evolution identity is achievable: drive both buffers with an identical
deterministic `add()` sequence and serialize both buffers' state; or pre-compute
sheeprl's sample-index sequence and apply both buffers' index-selection path to
it. Both paths produce byte-identical comparisons against the vendored sheeprl
reference at commit `33b6366`.

The spec defines 6 Lever-A tests (4 buffer storage/sample + 2 cadence-trace),
the per-test reviewer focus, the file-change spec for the `developer` agent, and
the deviation-log entry [D-004](DEVIATION_LOG.md#deviation-table) (`memmap` omission
pre-declared for PI ratification).

Parent plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). This spec is the
detailed content for the **CP3b** row of that plan's §"Checkpoint table (v3)".

## High-risk algorithmic details that MUST match sheeprl

The 8 items below are the spec for the four storage/sample tests plus the cadence
sub-gate. Each is cited against the vendored `vendor/sheeprl/sheeprl/` at commit
`33b6366`.

| # | Item | Sheeprl source line | Why it is high-risk (failure mode) |
|---|---|---|---|
| 1 | **Per-env buffer lanes** — N parallel envs' transitions stay in N separate column-lanes (axis 1); sampling pulls contiguous windows from ONE lane at a time | `vendor/sheeprl/sheeprl/data/buffers.py:467-526` (`SequentialReplayBuffer._get_samples`, env_idxes tiled to seq_length so flattened idx selects within one env) | Cross-lane interleaving → sequences cross env boundaries → world model trains on inter-env-spliced trajectories with garbage transition statistics |
| 2 | **Sequence-window sampling may straddle dones** | `vendor/sheeprl/sheeprl/data/buffers.py:395-465` (`SequentialReplayBuffer.sample`, valid_idxes only excludes the chunk that would overlap `self._pos`, NOT dones) | If JAX silently forbids straddling, `is_first=1` markers never appear inside sampled sequences → RSSM reset code path (§S1+§S4 / CP4b) never exercised at training time → CP4b's "fix" is silently dead code |
| 3 | **`replay_ratio` semantics** — `Ratio(replay_ratio)(policy_step / world_size)` returns per-rank-gradient-steps as a function of policy steps NOT macro-steps | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:518` (`ratio = Ratio(cfg.algo.replay_ratio, ...)`), `:661-662` (`ratio_steps = policy_step - prefill_steps * policy_steps_per_iter; per_rank_gradient_steps = ratio(ratio_steps / world_size)`) | Single biggest training-density lever; the 16× JAX/sheeprl drift (effective `replay_ratio = 1/128` on JAX vs 1.0 on sheeprl, both YAMLs writing `1.0`) is exactly this |
| 4 | **`learning_starts` boundary** (§S3) — off-by-one `prefill_steps = learning_starts - int(learning_starts > 0)` | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:510-515` (`learning_starts = cfg.algo.learning_starts // policy_steps_per_iter ...; prefill_steps = learning_starts - int(learning_starts > 0)`; resume-from-checkpoint adjusts both `learning_starts += start_iter` and `prefill_steps += start_iter`) | Mis-counted prefill phase boundary → train begins on tiny stale buffer OR never fills the buffer; the resume-from-checkpoint case is the silent variant |
| 5 | **`per_rank_gradient_steps`** — `n_samples=per_rank_gradient_steps` passed to `rb.sample_tensors`, then `for i in range(per_rank_gradient_steps): batch = {k: v[i].float() ...}; train(...)` | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:664-698` (per-iteration sample of `n_samples` independent batches → looped `train()` calls) | Combines with `replay_ratio` × `num_envs` × `world_size` to determine actual per-iteration update count |
| 6 | **`collect_interval`** semantics — env-steps-per-iteration before the next gradient block | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:550-551` (outer `for iter_num in range(...): policy_step += policy_steps_per_iter`); compare against our `configs/dreamer_v3_rr06.yaml` `collect_interval: 128` vs sheeprl XS `1` | 128× granularity difference between current JAX cascade and sheeprl reference; CP3b binds the dreamer-srl YAML to the sheeprl-XS semantic so this drift cannot recur |
| 7 | **Action storage timing** (works with §S2) — `step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1)); rb.add(step_data, ...)` BEFORE `envs.step(...)` runs | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:586-591` (`step_data["actions"] = ...; rb.add(step_data); next_obs, ... = envs.step(...)`) | Buffer must STORE actions at the right time index so the prepend-zero-action shift downstream (§S2 / CP2b) produces the correct alignment; storing actions one step late silently breaks CP2b's premise |
| 8 | **`is_first` storage** (works with §S1) — initial `step_data["is_first"] = np.ones_like(...)` BEFORE any step; reset to zeros after `rb.add`; set back to ones for dones-idxes after env reset | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:546` (initial), `:594` (reset to zero post-add), `:656` (re-mark for done-idxes) | Buffer must STORE `is_first` correctly so the force-set-to-1-at-chunk-start downstream (§S1 / CP4b) can find the marker; mis-storage silently breaks CP4b |

## Lever-A tests for CP3b (state-evolution flavour)

Each test below replaces the impossible "pure-function bit-identity" with an
achievable state-evolution variant. Same `1e-6` threshold, same fixed seed
`0xD3EAF`, same fixture convention (`tests/fixtures/dreamer_srl/cp3b_*.npz`).

1. **`test_buffer_storage_state_after_deterministic_adds`** — drive both the
   vendored PyTorch `SequentialReplayBuffer` and our JAX buffer with an identical
   1000-step `add()` sequence (per-step dict produced by a fixed-seed numpy
   generator, NOT each buffer's internal RNG). Serialize both buffers' `_buf`
   dicts (and `_pos`, `_full`, `_n_envs`, `_buffer_size`) to numpy. Assert byte-identical.
   This isolates the storage path from the sampling RNG.

2. **`test_buffer_sample_at_indices_matches_sheeprl`** — pre-compute sheeprl's
   sample-index sequence on a fixed seed (sheeprl's `_rng.integers(...)` output
   for given `batch_size`, `n_samples`, `sequence_length`, and the buffer state
   from test 1). Expose a private `_sample_at_indices(precomputed_indices)` path
   on the JAX buffer that bypasses JAX's PRNG entirely. Call both — assert the
   `_get_samples` output is byte-identical. This isolates the index-flattening
   and reshape path from the sampling RNG (which is D-002 class).

3. **`test_buffer_is_first_marker_placement_in_straddling_window`** — construct
   a buffer state where the 500th transition has `is_first=1` (i.e. a done
   landed at step 499 → env reset → step 500 is the first step of the new
   episode). Force-sample a window with start_idx=495, sequence_length=10
   (straddles the done). Assert: (a) the JAX output is byte-identical to
   sheeprl's; (b) `is_first[5] == 1` and `is_first[other indices] == 0`. This is
   the test that catches the silently-dead-CP4b-code-path failure mode (item
   #2 in the high-risk table).

4. **`test_buffer_parallel_env_lane_non_interference`** — add 200 transitions
   across 4 envs (50 per env, distinct sentinel values per env-column in the
   `observations` dict). Sample 100 windows of length 8 from each lane. Assert
   no cross-lane leakage: for every sampled window, the sentinel value at
   every time-index matches the source env's sentinel. This locks in item #1
   from the high-risk table.

## Cadence sub-gate (separate Lever-A tests, same CP-id)

5. **`test_cadence_yaml_key_parity_with_sheeprl_xs`** — mandatory-key audit on
   `configs/dreamer_srl/agent_xs.yaml`. Assert exact-match against sheeprl XS
   defaults for the following keys: `learning_starts`, `replay_ratio`,
   `per_rank_gradient_steps`, `per_rank_sequence_length`, `per_rank_batch_size`,
   `per_rank_pretrain_steps`, `per_rank_target_network_update_freq`,
   `total_steps`, `num_envs`. Document the canonical key list at the top of
   the YAML (a `# CP3b canonical key list` block comment) so downstream readers
   know the cadence-parity contract.

6. **`test_cadence_env_grad_step_trace_5000_iters`** — drive both implementations
   with the same `agent_xs.yaml` config and a fixed seed; mock out the actual
   `train()` and `envs.step()` calls so they only return a deterministic
   `(env_step_count, grad_step_count)` tuple per iteration; run the outer loop
   for 5000 iterations. Assert the produced sequence of `(env_step, grad_step,
   per_rank_gradient_steps)` tuples is byte-identical across the two
   implementations. This is "training-schedule bit-identity" — distinct from
   the within-train-step gradient-output identity that the other CPs cover, and
   it catches the exact 16× drift class from SPS §3.6.

## Reviewer-chain focus per CP3b's scope (Lever C trap table)

| Trap (active in CP3b) | What the reviewer trio looks for |
|---|---|
| `_get_samples` env_idxes tile-and-broadcast (item #1) | `code-reviewer`: numpy / `jnp.tile` axis discipline matches `np.tile(env_idxes, (1, sequence_length))`. `math-reviewer`: re-derives the flat-index formula `flat_idx = batch_idx * n_envs + env_idx` from sheeprl line 489. `professor`: confirms semantic — "each sequence comes from one env". |
| Straddle-dones permitted (item #2) | `code-reviewer`: JAX `valid_idxes` formula matches sheeprl line 444-451 (excludes only `self._pos` overlap, NOT dones). `math-reviewer`: confirms the 'first_range_end' / 'second_range_end' arithmetic. `professor`: confirms that S1+S4 reset code is what handles the within-window `is_first` marker — buffer must let the marker through. |
| `Ratio` semantics mismatch (item #3) | `code-reviewer`: JAX `Ratio` call site receives `policy_step / world_size`, NOT `policy_step // collect_interval`. `math-reviewer`: cumulative-grad-steps formula matches `cumulative_per_rank_gradient_steps * world_size / policy_step ≈ replay_ratio`. `professor`: confirms the binding constraint is "grad updates per env step", not "per macro step". |
| `prefill_steps` off-by-one (item #4) | `code-reviewer`: JAX line reproduces `prefill_steps = learning_starts - int(learning_starts > 0)` literally. `math-reviewer`: spot-checks the resume-from-checkpoint adjustment `learning_starts += start_iter; prefill_steps += start_iter`. `professor`: confirms semantic — prefill phase ends one iteration before learning_starts. |
| Action storage timing (item #7) | `code-reviewer`: `rb.add(step_data)` fires before `envs.step(...)`. `math-reviewer`: confirms §S2 prepend-zero-shift consumes this storage. `professor`: confirms the alignment with CP2b. |
| `is_first` storage (item #8) | `code-reviewer`: three placement sites match sheeprl lines 546, 594, 656. `math-reviewer`: confirms §S1 force-set-to-1 consumes this storage. `professor`: confirms the alignment with CP4b. |

## What CP3b explicitly does NOT cover

- **`EpisodeBuffer`, `EnvIndependentReplayBuffer`** — out of scope per v2 non-goals.
- **`memmap` mode** — pre-declared as deviation [D-004](DEVIATION_LOG.md#deviation-table)
  (memmap omission). The v2 plan declared this at design time (see v2 §"Non-goals"
  line referencing memmap); D-004 makes the omission an explicit, PI-ratifiable
  deviation entry rather than an implicit one.
- **`prioritize_ends`, `from_numpy`** — out of scope per v2 non-goals.

## File-change spec for `developer` to execute when CP3b launches

> **Senior-developer note:** the items below are *target-state* descriptions for
> the `developer` agent. Senior-developer has not edited these files (per the
> no-implementation policy); the `developer` agent makes the changes when this
> CP launches.

1. **`scripts/sheeprl_jax_diff.py:CHECKPOINT_REGISTRY`** — add a new entry
   *between the existing `"CP3"` and `"CP4"` rows* (preserve numeric order):
   ```python
   "CP3b": ["buffer_storage_state_after_deterministic_adds",
            "buffer_sample_at_indices_matches_sheeprl",
            "buffer_is_first_marker_placement_in_straddling_window",
            "buffer_parallel_env_lane_non_interference",
            "cadence_yaml_key_parity_with_sheeprl_xs",
            "cadence_env_grad_step_trace_5000_iters"],
   ```
   Also append the corresponding six entries to the top-of-file CP-comment
   block (the `# CP1 (utils.py): ...` ledger near line 161-172) between the
   `CP3` and `CP4` lines, in this form:
   ```
   # CP3b (buffers.py + train.py): buffer_storage_state_after_deterministic_adds,
   #                        buffer_sample_at_indices_matches_sheeprl,
   #                        buffer_is_first_marker_placement_in_straddling_window,
   #                        buffer_parallel_env_lane_non_interference,
   #                        cadence_yaml_key_parity_with_sheeprl_xs,
   #                        cadence_env_grad_step_trace_5000_iters
   #                        (State-evolution bit-identity, not pure-function;
   #                        D-004 memmap omission pre-declared.)
   ```
2. **`scripts/sheeprl_jax_diff.py:FUNCTION_REGISTRY`** — add six `_run_<name>`
   functions following the CP1 runner pattern (`_run_symlog` is the template:
   loads fixture, runs both sides, returns `(jax_out, torch_out, metadata)`).
   Each runner returns the state-evolution-equivalent comparison output (e.g.
   for `_run_buffer_storage_state_after_deterministic_adds`: serialize both
   buffers' `_buf` dicts to numpy and compare element-wise). The two cadence
   runners return the produced `(env_step, grad_step, per_rank_gradient_steps)`
   trace tuples.
3. **`tests/algorithms/dreamer_srl/test_buffers.py`** — create paired pytest
   tests for the six functions above. (Replaces the placeholder file currently
   noted in `tests/algorithms/dreamer_srl/README.md` line 63 as
   *"no CP — buffers.py is an inter-CP sanity round-trip"*.)
4. **`tests/algorithms/dreamer_srl/README.md`** — update line 63 from
   `# (no CP — buffers.py is an inter-CP sanity round-trip; see plan §"Implementation order" step 3)`
   to `# CP3b: buffer state-evolution parity + cadence trace (state-evolution bit-identity, not pure-function — see CP3B_SPEC.md)`.
   Also update `test_train.py`'s comment (line 66) to include CP3b's cadence
   tests if they live in `test_train.py` rather than `test_buffers.py` — at the
   developer's discretion; both placements are acceptable as long as the
   `CHECKPOINT_REGISTRY` resolves the fixture paths correctly.
5. **`tests/fixtures/dreamer_srl/`** — create six `.npz` fixture files and a
   `scripts/fixtures/gen_cp3b_fixtures.py` generator script. Seed `0xD3EAF`,
   conventions per `tests/fixtures/dreamer_srl/README.md`.
6. **`configs/dreamer_srl/agent_xs.yaml`** — add a top-of-file canonical-key
   block comment listing the cadence-parity contract enforced by
   `test_cadence_yaml_key_parity_with_sheeprl_xs`:
   ```yaml
   # CP3b canonical key list — these keys are byte-identical with sheeprl XS at
   # commit 33b6366. Any change requires updating both this YAML and the test
   # fixture at tests/fixtures/dreamer_srl/cadence_yaml_key_parity_input.npz.
   # Source: vendor/sheeprl/sheeprl/configs/exp/dreamer_v3_XS.yaml
   #   - algo.learning_starts
   #   - algo.replay_ratio
   #   - algo.per_rank_gradient_steps
   #   - algo.per_rank_sequence_length
   #   - algo.per_rank_batch_size
   #   - algo.per_rank_pretrain_steps
   #   - algo.critic.per_rank_target_network_update_freq
   #   - algo.total_steps
   #   - env.num_envs
   ```

## Links

- Parent plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — the v3 rebuild
  plan; this CP3b spec is the detailed content for the **CP3b** row of
  §"Checkpoint table (v3)".
- Deviation log: [DEVIATION_LOG.md](DEVIATION_LOG.md) — D-004 (memmap omission)
  is pre-declared for PI ratification alongside this spec.
- Empirical motivation: [SPS_COMPARISON_JAX_VS_SHEEPRL.md §3.6](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#36-where-the-jax-advantage-is-coming-from) —
  the 16× `replay_ratio` semantic drift between JAX and sheeprl, caught at the
  parity gate.
- v2 archived plan: [docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md](../../archive/dreamer_srl/IMPLEMENTATION_PLAN.md) —
  §"Training-loop semantics" defines §S1–§S10; CP3b interlocks with §S1 (`is_first`
  force-set), §S2 (action shift), §S3 (random-action prefill), §S4 (three-quantity
  reset).
- Code-reviewer Finding-6 ruling on stable CP-IDs: [docs/reviews/dreamer_srl_v3_pre_cp0_code_review.md](../../../reviews/dreamer_srl_v3_pre_cp0_code_review.md) —
  the basis for choosing the `Nb`-suffix convention for CP3b rather than
  renumbering downstream CPs.

---

## Implementation Report

**Implemented by: developer**
**Date: 2026-05-14**
**Branch: v1.4**

### Summary of changes (file-by-file)

| File | Status | Notes |
|---|---|---|
| `src/algorithms/dreamer_srl/buffers.py` | New | `SequentialReplayBuffer` ported from sheeprl@33b6366:sheeprl/data/buffers.py:L363-L526. In-memory only (D-004). Added `_sample_at_indices()` for CP3b explicit-index test path (not in sheeprl). All methods carry Lever-B citation headers. |
| `configs/dreamer_srl/agent_xs.yaml` | New | 9 cadence keys matching sheeprl XS defaults at commit 33b6366. Top-of-file CP3b canonical-key-list block comment. |
| `scripts/fixtures/gen_cp3b_fixtures.py` | New | Generates 6 `.npz` fixture files using sheeprl's Ratio + SRB. Seed 0xD3EAF. Run in `sheeprl_bridge` env. |
| `tests/algorithms/dreamer_srl/test_buffers.py` | New | 6 Lever-A tests (state-evolution bit-identity). All fixtures loaded from pre-computed `.npz` — no PyTorch needed at test time. |
| `tests/fixtures/dreamer_srl/*.npz` (6 files) | New | Pre-computed reference fixtures for CP3b. |
| `scripts/sheeprl_jax_diff.py` | Edited | CP3b added to `CHECKPOINT_REGISTRY` (between CP3 and CP4). 6 `_run_*` runners added to `FUNCTION_REGISTRY`. CP-comment ledger updated. |
| `tests/algorithms/dreamer_srl/README.md` | Edited | Line 63 updated from `(no CP — buffers.py is an inter-CP sanity round-trip...)` to CP3b status. |
| `docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md` | Edited | D-005 logged: filled-region-only comparison for test 1 (unfilled `np.empty` garbage excluded). |

### Test results

```
pytest tests/algorithms/dreamer_srl/test_buffers.py -v
  14/14 passed (6 CP3b + 8 CP1 regression)

python scripts/sheeprl_jax_diff.py --checkpoint CP3b → exit 0
  buffer_storage_state_after_deterministic_adds      PASS  max_abs_diff=0.000e+00
  buffer_sample_at_indices_matches_sheeprl           PASS  max_abs_diff=0.000e+00
  buffer_is_first_marker_placement_in_straddling_window PASS  max_abs_diff=0.000e+00
  buffer_parallel_env_lane_non_interference          PASS  max_abs_diff=0.000e+00
  cadence_yaml_key_parity_with_sheeprl_xs            PASS  max_abs_diff=0.000e+00
  cadence_env_grad_step_trace_5000_iters             PASS  max_abs_diff=0.000e+00
```

### Speed check

Skipped — CP3b adds no hot-path code (buffer is CPU NumPy; the JAX training hot path is not yet wired). No changes to `src/models/` or JAX-compiled functions. Speed measurement is not applicable at this stage.

### Deviations from plan

| ID | Description | Logged before commit |
|---|---|---|
| D-004 | memmap omission — pre-declared | Yes (pre-existing in DEVIATION_LOG.md) |
| D-005 | Test 1 compares only filled region `[:_pos]`; unfilled `[_pos:]` has undefined `np.empty` garbage that is not semantically meaningful | Yes (logged in DEVIATION_LOG.md before commit) |

**D-005 detail:** Both sheeprl and our JAX buffer allocate with `np.empty`. The uninitialized portion `[_pos:]` of the buffer contains garbage values from the host allocator — these differ between the two Python processes and cannot be compared. The test was originally written to compare the full buffer array, which caused `max_abs_diff=nan` (NaN from comparing uninitialized floats). The fix restricts the comparison to `[:_pos]` — the only region with semantically defined content. This is not a logic deviation; it is a test-scope constraint from the shared `np.empty` allocation pattern. The written state is byte-identical (confirmed by `max_abs_diff=0.000e+00`).

### YAML cadence keys confirmed

All 9 keys in `configs/dreamer_srl/agent_xs.yaml` match sheeprl XS defaults at commit 33b6366:
`learning_starts=1024`, `replay_ratio=1`, `per_rank_gradient_steps=1`, `per_rank_sequence_length=64`,
`per_rank_batch_size=16`, `per_rank_pretrain_steps=0`, `per_rank_target_network_update_freq=1`,
`total_steps=5000000`, `num_envs=1`.

### Isolation rule confirmed

`grep -r "from src.models.dreamer_v3" src/algorithms/dreamer_srl/` → no output.

### Checkpoints

- [x] `buffers.py` created with Lever-B citation headers on all methods
- [x] `agent_xs.yaml` created with CP3b canonical-key block comment
- [x] `gen_cp3b_fixtures.py` created, fixtures generated, all 6 `.npz` files written
- [x] `test_buffers.py` created, all 6 tests pass
- [x] `sheeprl_jax_diff.py` updated, `--checkpoint CP3b` exits 0
- [x] `README.md` updated
- [x] `DEVIATION_LOG.md` updated with D-005
- [x] No memmap code in `buffers.py` or `test_buffers.py`
- [x] No imports from `src.models.dreamer_v3_*`
- [x] D-004 + D-005 logged before commit
