---
title: "dreamer-srl v3 — CP3b Math Review (buffer + cadence)"
topic: dreamer
status: active
reviewer: math-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/buffers.py
audited_commit: 9c57c06
---

# dreamer-srl v3 — CP3b Math Review

## Verdict (plain language)

**Purpose.** Second of three sequential reviewer gates on the buffer + cadence checkpoint (CP3b) of the dreamer-srl v3 rebuild. The rebuild ports a PyTorch DreamerV3 reference (vendored sheeprl) into JAX with the binding constraint that **the math does not change**. CP3b is the layer that decides (a) *what bytes* every gradient update sees from the replay buffer and (b) *how often* gradient updates fire per environment step. The 16× replay-ratio drift that motivated promoting this layer to a real checkpoint was a pure cadence-arithmetic bug at the integration layer between three pieces — `learning_starts // num_envs`, `prefill_steps = learning_starts - 1`, and `Ratio((policy_step - prefill_steps) / world_size)` — that each looked correct in isolation. The 6 Lever-A tests gate that integration.

**Headline.** The math is bit-faithful. The ring-buffer arithmetic, the modular-index sample-window formula, the C-order flatten that interleaves storage as `(time, env, feature)` and recovers a flat element via `flat = t*n_envs + e`, the `is_first[5] == 1` placement for a window that crosses a done boundary, the `prefill_steps = learning_starts - int(learning_starts > 0)` off-by-one, the `Ratio.__call__(step)` fractional-debt accumulator, and the `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` subtraction at every iteration after `learning_starts` are all line-for-line identical to sheeprl. The two declared deviations (D-004 memmap-omission, D-005 `[:_pos]`-filled-region slice in Test 1) are both math-invariant: D-004 is a storage-backend choice that does not touch the buffer's bytes, and D-005 is a test-side semantic match against `np.empty`'s uninitialized-tail-garbage convention that the JAX implementation inherits unchanged from sheeprl's allocator path.

**Outcome.** No math findings. The cadence trace's first non-zero `grad_step` lands at iter 1024 (not iter 256 — the brief's iter-256 figure assumed `num_envs=4`, but the XS YAML pins `num_envs=1`, so `learning_starts // policy_steps_per_iter = 1024 // 1 = 1024`). The brief's description of the parallel-env buffer layout as `[env0×50, env1×50, ...]` is inverted from the actual interleaved-by-time-major layout, but the `t*n_envs + e` flat-index formula is correct for the actual layout, so the implementation math is right. **Verdict: PASS.** Professor-rl-bayesian-dl can fire next.

## Equations under review

### Eq. 1 — Ring-buffer index wrap

For an add of `n` transitions starting at write position `p` into a buffer of size `B`:

$$p' = (p + n) \bmod B, \quad \text{idxes} = \begin{cases} [p, p{+}1, \ldots, B{-}1] \cup [0, 1, \ldots, p'{-}1] & \text{if } p' \le p \\ [p, p{+}1, \ldots, p'{-}1] & \text{otherwise} \end{cases}$$

Sheeprl source (`vendor/sheeprl/sheeprl/data/buffers.py:193-198`). JAX port (`src/algorithms/dreamer_srl/buffers.py:153-158`). Line-for-line identical.

### Eq. 2 — Sample-window modular construction

Given start indices $\mathbf{s} \in \mathbb{Z}^{B \cdot N}$ and a sequence length $L$:

$$\text{idxes}[i, l] = (s_i + l) \bmod B, \quad l \in \{0, 1, \ldots, L{-}1\}$$

Sheeprl source (`vendor/sheeprl/sheeprl/data/buffers.py:459-460`). JAX port (`buffers.py:256-257`, `:378-379` in `_sample_at_indices`). Identical.

### Eq. 3 — Valid-start arithmetic when buffer is full

When `_full` is true, valid start indices are the union of two ranges that exclude the chunk that would overlap the current write head:

$$e_1 = p - L + 1, \quad e_2 = \begin{cases} B & \text{if } e_1 \ge 0 \\ B + e_1 & \text{otherwise} \end{cases}$$
$$\text{valid} = [0, e_1) \cup [p, e_2)$$

Sheeprl source (`buffers.py:444-451`). JAX port (`buffers.py:238-246`). Identical.

### Eq. 4 — Env-tiled flat-index formula (lane arithmetic)

Storage shape: `(B, N_\text{envs}, F)`. After C-order flatten of the first two axes to `(B \cdot N_\text{envs}, F)`, the row at flat index `i` corresponds to `(t = i // N_\text{envs}, e = i % N_\text{envs})`. To select element `(t, e)` the inverse is:

$$\text{flat}(t, e) = t \cdot N_\text{envs} + e$$

Sheeprl source (`buffers.py:489`). JAX port (`buffers.py:315`, `:389`). Identical.

### Eq. 5 — `is_first` window-relative offset

Given a window of length `L` starting at absolute index `s`, a transition at absolute index `j` lands at window-relative offset `j - s`. For a done at absolute index $d$ and the new-episode marker at $d{+}1$:

$$\text{offset}_{\text{is\_first}} = (d + 1) - s$$

Fixture sets $d = 499$, $s = 495$, $L = 10$ ⇒ offset = 5. Verified by the fixture generator (`gen_cp3b_fixtures.py:276`).

### Eq. 6 — `prefill_steps` off-by-one

Given a user-configured `learning_starts_steps` and `policy_steps_per_iter = num_envs * world_size`:

$$L_\text{start} = \left\lfloor \frac{L_\text{start, steps}}{P} \right\rfloor, \quad L_\text{prefill} = L_\text{start} - \mathbb{1}[L_\text{start} > 0]$$

The indicator handles the edge case `learning_starts_steps = 0` (no off-by-one needed when prefill is skipped entirely). Sheeprl source (`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:510-511`). JAX port (`tests/algorithms/dreamer_srl/test_buffers.py:407-408`). Identical.

### Eq. 7 — `ratio_steps` cadence shift

At iteration $i \ge L_\text{start}$, the Ratio scheduler is called with a shifted policy step:

$$\text{ratio\_steps}_i = \pi_i - L_\text{prefill} \cdot P, \quad \pi_i = i \cdot P \text{ (cumulative policy step)}$$

Sheeprl source (`dreamer_v3.py:661`). JAX port (`test_buffers.py:422`). Identical.

### Eq. 8 — Ratio scheduler with fractional-debt accumulator

First call (`_prev is None`):

$$k = \lfloor n \cdot r \rfloor, \quad \text{prev} \leftarrow n$$

Steady-state calls:

$$k = \lfloor (n - \text{prev}) \cdot r \rfloor, \quad \text{prev} \leftarrow \text{prev} + \frac{k}{r}$$

Sheeprl source (`vendor/sheeprl/sheeprl/utils/utils.py:273-291`). JAX port (`src/algorithms/dreamer_srl/utils.py:311-333`). Identical. (Same Eq. 7 from the CP1 math review — re-used at the integration layer here.)

## Per-test math audit

| # | Test | Formula correctness | Constants | Index arithmetic | Issues |
|---|---|---|---|---|---|
| 1 | `test_buffer_state_evolution_matches_sheeprl` | Eq. 1 ring-buffer wrap verified. After 100 deterministic adds with `buffer_size=200, data_len=1` per add, `_pos = 100, _full = False`, no wrap occurs. The two-branch wrap path is exercised by Eq. 1's `next_pos <= self._pos` predicate, which here evaluates `(100+1) % 200 = 101 ≤ 100`? False ⇒ takes the linear-range branch `range(self._pos, next_pos)` correctly. | `BUFFER_SIZE=200, N_STEPS_ADD=100` (fixture L45,49); add data_len=1 per step (`_make_step_data` returns seq_len=1) | `[:_pos]` slice with `_pos=100` selects the filled region. Slice math is conservative-correct: pre-wraparound, `_pos` is an absolute count (sheeprl L194 / JAX L154 are byte-identical `_pos` mutators). | none — see D-005 audit below |
| 2 | `test_buffer_indexed_sample_matches_sheeprl` | Eq. 2 + Eq. 4 verified. `precomputed_start_idxes` is supplied by the fixture (sheeprl-RNG-derived, then passed to both buffers). Eq. 2's modular sum constructs the same window in both implementations. Eq. 4's `t*n_envs + e` flat-index produces the same row selection. | `SEQUENCE_LENGTH=10, BATCH_SIZE=4, n_samples=1, N_ENVS_1=1` (fixture L50-52). With `n_envs=1`, Eq. 4 degenerates to `flat = t` — Test 2 does NOT exercise the lane arithmetic (that's Test 4's job). | The output shape `[n_samples, sequence_length, batch_size, ...]` results from `np.reshape(..., (n_samples, batch_size, sequence_length))` then `np.swapaxes(axis1=1, axis2=2)` — verified to be axis-consistent with sheeprl L505-L511 line-for-line. | none |
| 3 | `test_is_first_marker_at_done_boundary` | Eq. 5 offset formula verified. With `DONE_AT=499`, the new-episode `is_first=1` is set at index `500` (fixture L252). With `START_IDX_3=495`, window length 10: offset = `500 - 495 = 5`. The fixture generator pre-asserts `is_first_flat[5] == 1.0` (L279). The JAX test asserts the same byte-identical sample. | `BUFFER_SIZE_3=1000, N_STEPS_3=600, DONE_AT=499, START_IDX_3=495, SEQ_LEN_3=10` (fixture L235-239). Done at 499 places `is_first=1` at 500 (the NEW-episode first step, not the done step itself). | The window spans absolute indices `[495..504]`. With `_pos=600, _full=False`, no wraparound. Modular index `(495 + l) % 1000 = 495+l` for `l∈[0..9]`. Index 5 in the relative window = absolute 500 = the new-episode start. ✓ | none — the convention "is_first marks the step AFTER done" is the sheeprl convention (matches dreamer_v3.py:656 "set back to ones for dones-idxes after env reset") and the fixture encodes it exactly |
| 4 | `test_parallel_env_lane_non_interference` | Eq. 4 flat-index lane arithmetic verified. With `N_ENVS_4=4`, env-$e$'s sentinel is $e+1$ (fixture L332). The buffer storage shape `(B, 4, F)` is interleaved-by-time-major; a C-order flatten gives `[(t=0,e=0), (t=0,e=1), (t=0,e=2), (t=0,e=3), (t=1,e=0), ...]`. Selecting env $e$ uses flat index `t*4 + e` — distinct from the brief's description but mathematically the **correct** formula for the actual storage layout. | `N_ENVS_4=4, N_STEPS_4=50, BUFFER_SIZE_4=100, SEQ_LEN_4=8, N_WINDOWS_4=10` (fixture L319-323). `max_start = _pos - seq_len = 50 - 8 = 42`; valid starts are `[0..42]`, capped at `min(n_windows, 43) = 10`. | The `np.allclose(obs_flat, sentinel)` assertion checks every element of `[1, seq_len, batch, obs_dim] = [1, 8, 10, 8] = 640` floats — all must equal $e+1$. With correct lane arithmetic, env $e$'s window contains only `(t, e)` indices, all storing sentinel $e+1$. ✓ | none — the brief's description of layout as `[env0×50, env1×50, ...]` is inverted from the actual time-major interleaving, but Eq. 4 is the correct formula either way (the brief and impl describe the same flat-row pattern in different mental models) |
| 5 | `test_cadence_yaml_key_parity_vs_sheeprl_xs` | No formula — YAML-key constant audit only. The 9 expected values are hard-coded against sheeprl XS defaults (`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`, `exp/dreamer_v3.yaml`, `env/default.yaml`). YAML loaded by `yaml.safe_load`; each `assert algo.get(key) == expected_value` is an exact-equality check. | `learning_starts=1024, replay_ratio=1, per_rank_gradient_steps=1, per_rank_sequence_length=64, per_rank_batch_size=16, per_rank_pretrain_steps=0, per_rank_target_network_update_freq=1, total_steps=5000000, num_envs=1` (test L345-372). YAML at `configs/dreamer_srl/agent_xs.yaml:21-33` mirrors these values exactly. | n/a (constant audit) | none — the YAML keys and assert order are exact-match |
| 6 | `test_env_grad_step_trace_matches_sheeprl` | Eq. 6, Eq. 7, Eq. 8 verified at the integration layer. The Python loop in the test (L416-426) recomputes the cadence trace using **the same JAX `Ratio` class** (which is bit-identical to sheeprl per CP1 Eq. 7) called with **the same shifted-policy-step argument** (Eq. 7) under **the same prefill-off-by-one boundary** (Eq. 6). | `num_envs=1, world_size=1, total_iters=5000, replay_ratio=1.0, learning_starts_steps=1024, per_rank_pretrain_steps=0` (fixture L424-430). `policy_steps_per_iter=1, learning_starts=1024, prefill_steps=1023`. First non-zero `grad_step` at iter 1024 (not 256 — the brief's iter-256 assumed `num_envs=4`; XS pins `num_envs=1`). | At iter `i ≥ 1024`: `ratio_steps = i - 1023*1 = i - 1023`. First call (iter 1024): `ratio_steps = 1`, Ratio first-call special case, `repeats = int(1 * 1.0) = 1`, `_prev = 1`. Iter 1025: `ratio_steps = 2`, steady-state, `repeats = int((2-1)*1.0) = 1`. By iter 5000 the cumulative grad_step is `5000 - 1023 = 3977`. | none — the bit-identical assertion at L435 catches any single-iteration divergence; pre-computed sheeprl trace is the ground truth |

## Critical-points audit (the 7 named in the brief)

### Critical point 1 — `prefill_steps` off-by-one arithmetic

Sheeprl L510-L511 (line-by-line, no parens dropped):

```python
learning_starts = cfg.algo.learning_starts // policy_steps_per_iter
prefill_steps   = learning_starts - int(learning_starts > 0)
```

For `learning_starts_steps = 1024`, `policy_steps_per_iter = 1`:
- $L_\text{start} = 1024 // 1 = 1024$
- $L_\text{prefill} = 1024 - 1 = 1023$ (the indicator evaluates to 1)

For `learning_starts_steps = 0`:
- $L_\text{start} = 0$
- $L_\text{prefill} = 0 - 0 = 0$ (the indicator evaluates to 0 — no off-by-one when prefill is skipped)

The training-start gate is `if iter_num >= learning_starts:` (dreamer_v3.py L660), so iter 1024 is the **first** iter that triggers a gradient update. The ratio_steps shift then makes the Ratio scheduler see step `1024 - 1023 = 1` on its very first call, mapping the first policy-step-past-prefill to ratio-step 1 — which is the conventional first-step-of-the-train semantics. Verified line-for-line in `test_buffers.py:407-408` and `gen_cp3b_fixtures.py:434-436`. ✓

**Note on brief's iter-256 figure.** The brief states "gradient updates begin at iteration `1024 // (num_envs * action_repeat) = 256` for `num_envs=4, action_repeat=1`." The fixture and YAML pin `num_envs=1` (not 4), and there is no `action_repeat` term in the formula (sheeprl uses `policy_steps_per_iter = num_envs * world_size` — `action_repeat` is an env-wrapper concern that affects raw env-step counting but does NOT enter `policy_steps_per_iter`). The math for the XS config is iter 1024, not 256. This is a brief-side restatement issue, not an implementation issue. ✓ PASS.

### Critical point 2 — `Ratio` scheduler formula

Eq. 8 verified. The `collect_interval` term mentioned in the brief is implicit in sheeprl XS: there is no separate `collect_interval` key — the outer loop runs `policy_step += policy_steps_per_iter` per iteration (dreamer_v3.py L551), so "collect interval = 1 iteration = `policy_steps_per_iter` env-steps." The Ratio scheduler is called with `ratio_steps / world_size`; with `world_size = 1` this is just `ratio_steps`. The integer-floor `int((step - prev) * ratio)` with `ratio = 1.0` and integer step arg produces `(step - prev)` exactly (no fractional debt to accumulate in this special configuration; the `prev += repeats / ratio` increment is exact). ✓

### Critical point 3 — `is_first` marker placement at done boundary

Eq. 5 verified. The convention `is_first[d+1] = 1` (NOT `is_first[d] = 1`) matches sheeprl `dreamer_v3.py:656`: "set back to ones for dones-idxes after env reset." The reset happens AFTER `rb.add(step_data)` (L587) and BEFORE the next add (L656), so the next transition stored carries `is_first = 1`. Verified at fixture-generation time by `gen_cp3b_fixtures.py:248-252` (done at 499 → is_first=0; index 500 → is_first=1) and re-asserted at L279. The window-relative offset $d+1-s = 500-495 = 5$ matches the test assertion. ✓

### Critical point 4 — Parallel-env lane-index math

Eq. 4 verified. The brief's described layout `[env0×50, env1×50, env2×50, env3×50]` (env-major contiguous) is **inverted** from the actual storage layout, which is time-major interleaved: shape `(buffer_size, n_envs, F)` flattens C-order to `[(t=0,e=0), (t=0,e=1), (t=0,e=2), (t=0,e=3), (t=1,e=0), ...]`. The formula `flat = t * n_envs + e` is the inverse of this time-major C-order flatten — and it is the formula sheeprl uses (L489) and the formula JAX uses (L315, L389). Test 4's assertion `np.allclose(obs_flat, sentinel)` relies only on the **per-env constancy** of the sentinel value across time (every (t, e=$e_0$) cell holds $e_0+1$), so the test would pass under either layout — but the formula `t*n_envs + e` actually computes the time-major case. ✓ PASS, with a notation note that the brief's worded layout is inverted from the actual storage but the math is right.

### Critical point 5 — `buffer_size × num_envs` capacity

For `buffer_size = 1_000_000` total and `num_envs = 4`: per-env capacity is `1_000_000 // 4 = 250_000` ⇒ no rounding loss. For `buffer_size = 1_000_001` (the brief's edge case): `1_000_001 // 4 = 250_000` per env ⇒ 1 transition is discarded by integer floor. The XS YAML pins `buffer_size = 1_000_000, num_envs = 1`, so per-env capacity = 1_000_000 (no division). The buffer's `__init__` (`buffers.py:60-61`) accepts `buffer_size` and `n_envs` as separate constructor arguments — there is no implicit divide; the caller is responsible for the splitting. The sheeprl XS path in `dreamer_v3.py` does NOT split (it uses `EnvIndependentReplayBuffer` for env-major splitting, which is out of scope here). For CP3b's `SequentialReplayBuffer`, the math is: `buffer.shape = (buffer_size, n_envs, F)` ⇒ total transitions = `buffer_size * n_envs`. The `1_000_000 // 4 = 250_000` figure in the brief is an `EnvIndependentReplayBuffer` semantic, not the `SequentialReplayBuffer` semantic that CP3b ports. ✓ PASS — the math is the simpler "axes-multiply" form here.

### Critical point 6 — Test 1's `[:_pos]` slicing semantics (D-005)

`_pos` math: sheeprl L221 / JAX L174 set `self._pos = next_pos`, where `next_pos = (self._pos + data_len) % self._buffer_size` (L194 / L154). Pre-wraparound, this is an absolute count of transitions written. Post-wraparound, this is a modular write-position. The transition between regimes is gated by `self._full`, which is set true at the line just before `_pos` is reassigned (L219 / L172): `if self._pos + data_len >= self._buffer_size: self._full = True`.

Test 1 uses 100 adds × data_len=1 = 100 transitions into a 200-slot buffer ⇒ `_pos = 100, _full = False`. The slice `[:100]` is a true filled-region slice (pre-wraparound regime). The unfilled portion `[100:]` is `np.empty`-allocated uninitialized memory (sheeprl L214 / JAX L165-167 — both call `np.empty`). The two host allocations cannot agree on the uninitialized tail, hence D-005 restricts the comparison to `[:_pos]`.

**Wraparound edge case (the brief asks).** The slice `[:_pos]` is **not** semantically correct post-wraparound (it would miss valid transitions in `[_pos:buffer_size]`). But Test 1 explicitly guards against this case: `assert jax_rb._full == expected_full` (L94), and the fixture has `expected_full = False` (L108 in gen_cp3b_fixtures.py). The test would fail loudly if the wraparound regime were entered. There is no hidden edge case. The `_pos` semantics are identical between sheeprl and JAX because the JAX port is line-for-line — both treat `_pos` as the same wraparound-modular index. ✓ PASS for D-005's math.

### Critical point 7 — `_get_samples` index arithmetic for straddling-done windows

Eq. 2 + Eq. 5 verified. With `start = 495, seq_len = 10, done_at = 499, buffer_size = 1000`:
- Window absolute indices: $\{(495 + l) \bmod 1000\}_{l=0}^{9} = \{495, 496, 497, 498, 499, 500, 501, 502, 503, 504\}$ (no wrap)
- `is_first[t]` for these absolute indices: $\{0, 0, 0, 0, 0, 1, 0, 0, 0, 0\}$ (set at the new-episode start at $t=500$)
- Window-relative position of $t=500$: $500 - 495 = 5$
- ⇒ window's `is_first[5] = 1`, rest zero

Formula `offset = done_at - start + 1 = 499 - 495 + 1 = 5`. ✓ Matches Eq. 5.

The "+1" in `done_at - start + 1` is the new-episode-marker convention: `is_first` is at index `done_at + 1`, not `done_at`. Eq. 5 makes this explicit by writing `(d + 1) - s` rather than `d - s + 1` (algebraically the same, semantically clearer). ✓ PASS.

## Deviation math review

### D-004 — memmap omission

**Math lens verdict: PASS.** Memmap-vs-RAM is a storage-backend distinction (how the bytes are persisted to disk), not a computational distinction (what the bytes are). Both `np.empty(shape, dtype)` (in-RAM path, sheeprl L214 / JAX L165) and `MemmapArray(filename, dtype, shape, mode)` (memmap path, sheeprl L205-L210, dropped on JAX side) produce the same uninitialized buffer of the same shape and dtype, addressable by the same `[idxes]` assignment. Subsequent reads of written slots return identical bytes. No equation in CP3b's math is sensitive to the storage backend. ✓

### D-005 — `[:_pos]`-filled-region slice in Test 1

**Math lens verdict: PASS.** As audited in Critical point 6 above. The `_pos` semantics are byte-identical between sheeprl and JAX (the JAX `add()` method is a line-for-line port of sheeprl L193-L221 modulo the dropped memmap branch). Pre-wraparound, `_pos` is the absolute count of written transitions, and `[:_pos]` is the semantically-correct slice for "filled region." The unfilled tail `[_pos:]` is uninitialized host memory (`np.empty`-allocated by both implementations), so excluding it from the comparison is semantically required — comparing uninitialized memory would be comparing arbitrary garbage. The test's guard `assert _full == expected_full == False` blocks the wraparound regime where `[:_pos]` would silently miss valid transitions. There is no hidden math deviation. ✓

## Conclusion

✅ **PASS** — math is bit-faithful to vendored sheeprl@`33b6366`. The 6 Lever-A tests cover the 8 equations under review (Eq. 1-3 storage + sample, Eq. 4 lane arithmetic, Eq. 5 is_first offset, Eq. 6-8 cadence). All formulas, constants, and index arithmetic match sheeprl line-for-line. The two declared deviations (D-004, D-005) are math-invariant. Professor-rl-bayesian-dl can fire next.

Reviewed by: math-reviewer
