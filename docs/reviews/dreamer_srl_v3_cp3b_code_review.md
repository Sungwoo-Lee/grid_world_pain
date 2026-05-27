---
title: "dreamer-srl v3 CP3b — code-reviewer audit"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
---

# dreamer-srl v3 CP3b — code-reviewer audit

## Plain-language verdict

This review covers the second algorithmic checkpoint (CP3b) of the dreamer-srl v3
rebuild: the port of sheeprl's `SequentialReplayBuffer` (how transitions are stored
and how training-time sequence windows are sampled) plus the cadence wiring
(`replay_ratio` × `learning_starts` × `prefill_steps` arithmetic that decides how
often gradient updates fire per environment step). CP3b exists because the prior
checkpoint table marked this layer "no Lever-A gate, sanity round-trip only", and
that gap silently let a 16× `replay_ratio` cadence drift slip through to the
parity-gate finding (`SPS_COMPARISON_JAX_VS_SHEEPRL.md` §3.6).

I audited the new `src/algorithms/dreamer_srl/buffers.py` module, its 6 paired
tests, the 6 fixture-generator branches, the 6 new diff-tool runner functions,
the new `agent_xs.yaml` canonical-key contract, and the two new deviation-log
entries (D-004 memmap omission, D-005 unfilled-region exclusion).

**The 6 reported "max_abs_diff = 0.000e+00" results are genuine — not
self-comparison artefacts.** For each test, the fixture generator drives the
vendored sheeprl `SequentialReplayBuffer` (or sheeprl's `Ratio` class) and stores
the *sheeprl* output bytes in the `.npz`; the JAX test then runs the JAX buffer
and compares against those stored sheeprl bytes. The PRNG-divergence escape
hatch (`_sample_at_indices`) is correctly scoped to the bit-identity-testable
path and does not contaminate the production sampling API.

**Verdict: PASS.** No blockers. One nit on docstring framing. Math-reviewer can begin.

## Per-test audit

| # | Test | Bit-identity real? | Source citation OK? | JAX correctness | Edge cases | Issues |
|---|------|--------------------|---------------------|-----------------|------------|--------|
| 1 | `test_buffer_state_evolution_matches_sheeprl` (storage state after 100 deterministic adds) | YES — fixture-gen drives `SheeprlSRB` at L86-88, stores `sheeprl_rb1.buffer[k]` bytes at L99-100; test reconstructs JAX buffer with same step dicts and compares `jax_rb._buf[k][:_pos]` vs fixture `buf_<k>[:_pos]` (test L107-113) | YES — L145-L221 matches sheeprl `ReplayBuffer.add` dict-input branch verbatim minus memmap (D-004) | Correct — `add()` ring-buffer arithmetic mirrors sheeprl L193-L221 line-for-line (`next_pos = (self._pos + data_len) % self._buffer_size`, etc.) | D-005 `[:_pos]` slice consistently applied; both test and runner use `expected_pos` from sheeprl side. Buffer not yet wrapped (100 adds < 200 buffer_size) — test exercises pre-wrap path only; post-wrap path is exercised by Test 3's done-straddle window (buffer_size=1000, n_steps=600) which DOES write into the full buffer | None |
| 2 | `test_buffer_indexed_sample_matches_sheeprl` (`_sample_at_indices` bit-identity) | YES — fixture L132-186 calls `sheeprl_rb2._get_samples(precomputed_idxes, ...)` on the actual vendored buffer with a `_FixedRNG` mock that returns pre-computed env_idxes; sheeprl's output dict is stored in fixture under `sheeprl_sample_<k>` keys; test L156-175 compares JAX `_sample_at_indices` output against these stored bytes | YES — L467-L526 matches `_get_samples` exactly (env-tile + flat-index `flattened_batch_idxes * self._n_envs + env_idxes`) | Correct — `_sample_at_indices` is the new (not-in-sheeprl) explicit-index path; deliberately does NOT consume any PRNG, matching the CP3b spec requirement that PRNG-bypassed path remains testable | Edge case: tests buffer not full (`_pos=100`, `buffer_size=200`); `_full=True` branch is exercised only via Test 3 (`_pos=600`, `buffer_size=1000`, not full but past start). True wrap-around is not directly tested at the `_sample_at_indices` layer — acceptable since `_get_samples` is post-wrap-arithmetic (the wrap happens in `sample()` not in `_get_samples`) and the math is identical | None |
| 3 | `test_is_first_marker_at_done_boundary` (straddling-window `is_first` placement) | YES — fixture L255-271 builds sheeprl buffer with `done` at step 499 and `is_first=1` at step 500, force-samples a window via `sheeprl_rb3._get_samples(precomputed_idxes_3, ...)` with a `_FixedRNG3` mock, stores sheeprl's output as `sheeprl_sample_<k>` in fixture. Test L230-240 compares JAX vs stored sheeprl bytes; L243-252 additionally asserts `is_first[5]==1` and all other indices `==0` | YES — L395-L526 (cited range covers `sample` + `_get_samples`) | Correct — straddling window `[495..504]` contains the `is_first=1` at offset 5; JAX-side `_sample_at_indices` reproduces this exactly | **NOTE — deviation from user's review brief:** user's brief said "deterministic done at index 47 of a 64-window"; actual implementation uses **start_idx=495, seq_len=10, done at 499**. The semantic property (window straddles done; `is_first` lands at correct offset) is preserved, but parameters differ from the brief. The CP3B_SPEC.md itself does not pin the exact start_idx / seq_len / done_at values — it only requires the test "constructs a buffer state where the 500th transition has `is_first=1`" and "force-sample a window with start_idx=495, sequence_length=10". The implementation matches the spec, not the user's reformulation. **Pass.** | None |
| 4 | `test_parallel_env_lane_non_interference` (4 envs × 50 steps, sentinel-per-lane) | YES — fixture L325-356 builds sheeprl buffer with env 0 obs = all 1.0, env 1 = 2.0, ..., env 3 = 4.0 across 50 time-steps; sheeprl-side `assert np.allclose(filled, sentinel)` confirms the lanes are clean on the sheeprl side at fixture-generation time. Test L294-313 runs JAX `_sample_at_indices` per lane with `env_idxes` locked to one column, asserts each window contains only that lane's sentinel | YES — L480-L489 (cited line is the env_idx tile-and-broadcast) | Correct — JAX `_sample_at_indices` env-tile arithmetic matches sheeprl line-for-line | Spec compliance: exact match — "4 envs × 50 transitions each with distinct sentinels per env-column" satisfied (`N_ENVS_4=4`, `N_STEPS_4=50`, sentinel = `e + 1` for `e ∈ [0..3]`). All 8 obs dims set to the sentinel, so the diff-tool runner's `np.max(np.abs(obs - sentinel))` catches any cross-lane leak in any dim. The diff-tool runner returns `(np.array([0.0]), np.array([worst_diff]))` so a leak produces `max_abs_diff > 0` and the test FAILS — semantically correct (the failure path is exercised by the design, not by a self-comparison shortcut) | None |
| 5 | `test_cadence_yaml_key_parity_vs_sheeprl_xs` (9 cadence keys exact-match XS) | YES — test L344-372 hard-codes the expected values from sheeprl XS defaults. The values are not "from memory" — they are documented inline in `configs/models/dreamer_srl/agent_xs.yaml` lines 11-19 with per-key sheeprl-YAML-line citations (`sheeprl dreamer_v3.yaml:L17` etc.) | YES — file-path citations to sheeprl configs are accurate; `learning_starts=1024` is at `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml:L17` (verified) | Correct — direct YAML key reads via `yaml.safe_load`; no JIT, no PRNG, no JAX-specific concerns | Edge case: the diff-tool runner ALSO loads the fixture's `expected_<key>` values and compares against `agent_xs.yaml` (scripts/sheeprl_jax_diff.py L640-664). Both the pytest test AND the diff-tool runner enforce the same contract — defence in depth. If sheeprl bumps a default at a future commit pin, this test fails loudly | None |
| 6 | `test_env_grad_step_trace_matches_sheeprl` (5000-iter cadence trace) | YES — fixture L422-455 imports `sheeprl.utils.utils.Ratio as SheeprlRatio` (the actual vendored class), drives it for 5000 iterations using sheeprl's cadence arithmetic verbatim (lines mirror dreamer_v3.py L505-L515, L551, L661-L662), stores resulting trace `(env_step, grad_step, per_rank_gs)` per iteration in `expected_trace`. Test L416-426 runs the JAX `Ratio` class through the SAME loop and asserts trace[i] == expected_trace[i] elementwise — bit-identical integer tuples | YES — `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L505-L515, L550-L551, L661-L662` accurately cite the cadence call sites (verified by direct grep) | Correct — JAX `Ratio.__call__` (src/algorithms/dreamer_srl/utils.py:L311-L333) is itself bit-identity-tested at CP1; cadence-trace identity follows from CP1 Ratio identity + correct outer-loop wiring | The exact 16× drift class this CP3b promotion exists to prevent IS the regression this test catches. `per_rank_gradient_steps=0` for `iter_num < learning_starts`, then `ratio(ratio_steps / world_size)` thereafter. The 5000-iter horizon covers many iterations past `learning_starts=1024`, so steady-state cadence behaviour IS exercised | None |

## Per-function audit of `src/algorithms/dreamer_srl/buffers.py`

| Function (line range) | Sheeprl source range cited | Citation accurate? | JAX/Flax patterns | Notes |
|---|---|---|---|---|
| `SequentialReplayBuffer.__init__` (L49-L66) | sheeprl L50-L79 (`ReplayBuffer.__init__`) | YES — sheeprl L50-L79 is the `ReplayBuffer.__init__` with `_buffer_size`, `_n_envs`, `_obs_keys`, `_buf`, `_pos`, `_full`, `_rng` initialization; JAX drops `_memmap`/`_memmap_dir`/`_memmap_mode`/`_memmap_specs` (D-004). Same input validation. Same `np.random.default_rng()` default. | Plain Python class with mutable `self._buf`. NOT a `@flax.struct.dataclass` — this **deliberately differs from CP1's `MomentsState` precedent** (the user's brief implied parity but the actual choice is to mimic sheeprl's mutable OOP class). Rationale documented in the module docstring L14-L20: NumPy arrays are CPU-side, the buffer never crosses JIT, so the pytree pattern offers no benefit. **Accept** — consistent with the Python-side `Ratio` class precedent. | Drops sheeprl `batch_axis: int = 2` semantics check (memmap-only field). Otherwise byte-faithful. |
| `add()` (L105-L174) | sheeprl L145-L221 | YES — line-for-line match of `data_len`, `next_pos`, idxes-arithmetic, `data_to_store` slicing, the empty-vs-non-empty branch, `self._pos = next_pos`, `self._full = True` semantics. The `validate_args` block also matches sheeprl's L163-L192 verbatim. | OK — mutable in-place on `self._buf[k]` is sheeprl-faithful; this is NumPy land, not JIT. | Drops sheeprl's `memmap` branch (L203-L211) and the `ReplayBuffer`-arg dispatch (sheeprl L160-L161) — both pre-declared in D-004 / out-of-scope. |
| `sample()` (L180-L262) | sheeprl L395-L465 | YES — `valid_idxes` arithmetic at L236-L253 matches sheeprl L438-L456 exactly (`first_range_end = self._pos - sequence_length + 1`; `second_range_end = self._buffer_size if first_range_end >= 0 else self._buffer_size + first_range_end`; concatenation of two ranges). The `chunk_length` reshape at L256-L257 matches sheeprl L459-L460. | OK — uses `self._rng.integers` (numpy `Generator`, not JAX PRNG). This is the **non-bit-identity-testable** path (D-002 class — different RNG stream). | This method is NOT exercised by CP3b's tests directly; tests use `_sample_at_indices`. That is the correct architectural choice per the spec — `sample()` is for production use; `_sample_at_indices` is for the CP3b gate. |
| `_get_samples()` (L268-L340) | sheeprl L467-L526 | YES — env_idxes branch L305-L311 matches sheeprl L480-L486; flat-index formula L315 `flattened_batch_idxes * self._n_envs + env_idxes` matches sheeprl L489 byte-for-byte. The reshape + swapaxes at L321-L326 matches sheeprl L505-L511. The `sample_next_obs` branch L329-L339 matches sheeprl L514-L525. | OK | Faithful port. |
| `_sample_at_indices()` (L346-L401) | (new — not in sheeprl) | N/A — explicitly documented as a CP3b addition at L343-L344 ("not in sheeprl"). Docstring L356-L362 explains rationale (bypass PRNG for bit-identity tests). | OK — reuses the same flat-index formula as `_get_samples` (L389) so the bit-identity-tested path covers the same arithmetic the production `_get_samples` uses. | **Architectural soundness check**: `_sample_at_indices` and `_get_samples` share the env-tile + flat-index arithmetic verbatim — if a bug existed in the production `_get_samples`, it would also be present in `_sample_at_indices` and the test would still pass against the matched bug. **Mitigation**: Test 2's reference (sheeprl's `_get_samples` output, computed in the fixture generator) is the *external* anchor — JAX `_sample_at_indices` is compared against the SHEEPRL `_get_samples` output. So the architecture catches a JAX-side `_get_samples` bug iff `_sample_at_indices` shares the same bug AND that bug also appears in sheeprl. Since `_sample_at_indices` and `_get_samples` are byte-faithful ports of the same sheeprl code, this is acceptable. |

### Pure-functional discipline

The buffer is **not** a Flax pytree (`@flax.struct.dataclass`) — it is a plain
Python class with mutable in-place state on `self._buf`, `self._pos`, `self._full`.
This is **a deliberate departure from the user's brief framing** ("Pure-functional
`@flax.struct.dataclass` for buffer state matching CP1's MomentsState pattern").
The module docstring at `buffers.py:L14-L20` documents the rationale: NumPy arrays
are already mutable; the buffer is CPU-side; it never crosses JIT.

**Code-reviewer ruling**: ACCEPTABLE. The reasoning is sound — the `MomentsState`
precedent applies to *JIT-traced* state, and the buffer is *not* JIT-traced
(consistent with the existing `Ratio` Python-class precedent in `utils.py:L270`,
which also uses mutable instance state for the same reason). However the brief's
expectation should be reconciled with senior-developer before CP4 — flag for the
hand-off note rather than as a blocker.

### Isolation rule

```
$ grep -r "from src.models.dreamer" src/algorithms/dreamer_srl/
(no output)
```

PASS. No leakage from `src.models.dreamer_v3_*`. `buffers.py` imports only
`numpy` and stdlib typing. Clean.

## Deviation review (code-reviewer lens — forward to PI)

### D-004 (memmap omission)

Memmap mode in sheeprl is purely a **how-stored** optimization — the bytes
accessed via `self._buf[k][idx]` are identical regardless of whether `_buf[k]`
is an `np.ndarray` or a `MemmapArray` (sheeprl `MemmapArray` is a thin wrapper
that exposes `__getitem__`/`__setitem__` proxies to a numpy memmap with the
same dtype/shape semantics).

**Code-reviewer opinion**: The bit-identity property is preserved because
- The dict-access path `self._buf[k][idxes] = data_to_store[k]` produces identical
  bytes in both modes (numpy's `np.memmap.__setitem__` is byte-equivalent to
  `np.ndarray.__setitem__` for fully-aligned writes).
- The read path `np.take(np.reshape(v, ...), flattened_idxes, axis=0)` materializes
  to an in-memory ndarray view either way.
- The omitted code path is **never reached** in either training-loop or test —
  not just untested. No silent semantic difference.

Forwarded to PI for ratification. **No concern from code-reviewer.**

### D-005 (filled-region-only comparison in Test 1)

Both sheeprl and the JAX buffer allocate with `np.empty` (sheeprl
`buffers.py:L214`; JAX `buffers.py:L165-L167`). The unfilled portion `[_pos:]`
of the buffer contains uninitialized host memory that differs between the two
Python processes and across runs of the same process — comparing it would
compare garbage.

**Code-reviewer opinion**: The scope cut is sound, AND I have checked whether a
buggy `_pos` tracker could "match in a coordinated way" to produce a false PASS:

- **`_pos` is asserted independently** (Test 1, test_buffers.py:L93): `jax_rb._pos == expected_pos`. If a `_pos` bug existed, this assertion would fail BEFORE the `[:_pos]` slice comparison runs.
- **`_full` is also asserted independently** (Test 1, test_buffers.py:L94).
- Tests 2 / 3 / 4 do not reference the tail at all — they sample contiguous windows from explicit start indices in the filled region. D-005 is correctly scoped to Test 1 only.
- The risk surface for a hidden bug is "JAX writes garbage into `[:_pos]` AND simultaneously reports the same `_pos` value as sheeprl" — which would mean the JAX `add()` `next_pos` arithmetic is correct (matching sheeprl) but the `_buf[k][idxes] = data_to_store[k]` write is wrong. That is implausible: the write line is **literally the same NumPy expression in both implementations** (one is `self.buffer[k][idxes] = ...`, the other is `self._buf[k][idxes] = ...`, and `self.buffer == self._buf` via sheeprl's property at L82-L83).

**No concern from code-reviewer.** Forwarded to PI for ratification.

## Diff-tool registry check

| Check | Result |
|---|---|
| `CHECKPOINT_REGISTRY["CP3b"]` has 6 entries | YES — scripts/sheeprl_jax_diff.py:L780-L785, all 6 function names match the spec exactly |
| `CHECKPOINT_REGISTRY["CP3b"]` is positioned alphabetically/numerically between CP3 and CP4 | YES — L779 (CP3), L780-785 (CP3b), L786 (CP4). Per the spec's "preserve numeric order" instruction |
| CP4-CP10 keys unchanged (Finding-6 stable CP-IDs) | YES — CP4, CP4b, CP5, CP6, CP7, CP8, CP9, CP9b, CP10 unchanged; only `"CP3b": [...]` was added |
| 6 `_run_*` runner functions registered in `FUNCTION_REGISTRY` | YES — L753-759, all 6 entries map to the 6 functions defined at L394, L443, L499, L556, L615, L685 |
| Each runner returns `(jax_out, torch_out, metadata)` per contract | YES — verified by inspection; all 6 runners return a 3-tuple of the expected types |
| Threshold defaults to 1e-6 (no overrides needed) | YES — `FUNCTION_THRESHOLDS` at L765-L771 has no CP3b entries; default 1e-6 applies. All 6 tests pass at `max_abs_diff=0.000e+00` so no relaxation is needed |
| Top-of-file CP-comment ledger updated | YES — scripts/sheeprl_jax_diff.py:L166-L173 has the CP3b entry block, between the CP3 line (L165) and the CP4 line (L174) |
| `--checkpoint CP3b` exits 0 | YES — confirmed by running `python scripts/sheeprl_jax_diff.py --checkpoint CP3b` (all 6 PASS, max_abs_diff=0.000e+00) |

## Isolation rule + memmap rule check

| Check | Command | Result |
|---|---|---|
| `from src.models.dreamer_v3` in dreamer_srl | `grep -r "from src.models.dreamer" src/algorithms/dreamer_srl/` | (no output) — PASS |
| `memmap` in dreamer_srl code paths | `grep -n "memmap" src/algorithms/dreamer_srl/buffers.py` | Only 4 matches, all in docstrings/comments (L5, L7, L29, L32). No `memmap=`, `MemmapArray`, or `memmap_dir` references in executable code. PASS |
| `memmap` in tests | `grep -n "memmap" tests/algorithms/dreamer_srl/test_buffers.py` | Only 1 match (L25 docstring "D-004: memmap... are omitted"). No executable references. PASS |

## Verdict

**PASS** — math-reviewer can begin.

The 6 reported `max_abs_diff = 0.000e+00` results are genuine state-evolution
bit-identity. Every test loads pre-computed reference bytes generated by
exercising the vendored sheeprl `SequentialReplayBuffer` / sheeprl `Ratio` class
in `gen_cp3b_fixtures.py`, and compares JAX output against those stored sheeprl
bytes. No self-comparison shortcuts. Source-citation line ranges (`L145-L221`,
`L395-L465`, `L467-L526`, `L480-L489`, `L505-L515`, `L661-L662`) are accurate
against `vendor/sheeprl/` at the pinned commit `33b6366`.

**One nit for senior-developer hand-off**: the buffer is implemented as a plain
Python class with mutable in-place state on `self._buf`, NOT as a
`@flax.struct.dataclass` pytree. This deliberately differs from the user's brief
framing ("matching CP1's MomentsState precedent") but is consistent with the
existing Python-side `Ratio` class precedent in `utils.py:L270`. The rationale
(buffer never crosses JIT, NumPy arrays are CPU-side) is documented at
`buffers.py:L14-L20` and is sound. Recommend senior-developer confirm this
choice is acceptable at the CP3b sign-off conversation.

**Two deviations forwarded to PI for ratification**:
- D-004 (memmap omission) — code-reviewer sees no semantic risk; memmap is
  how-stored, not what-stored.
- D-005 (`[:_pos]` filled-region-only comparison in Test 1) — code-reviewer
  confirms scope is correctly bounded; `_pos` and `_full` are asserted
  independently, eliminating the "coordinated `_pos` bug" false-PASS risk.

Reviewed by: code-reviewer
