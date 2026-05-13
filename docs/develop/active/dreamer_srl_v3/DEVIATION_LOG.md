---
title: "dreamer-srl v3 — Deviation Log"
topic: dreamer
status: active
created: 2026-05-13
last_updated: 2026-05-14  # D-005 added (CP3b impl: unfilled-region exclusion in test 1)
phase: 2
---

# dreamer-srl v3 — Deviation Log

## Purpose

This file lists every place the JAX `dreamer-srl` implementation deviates from
the vendored sheeprl@`33b6366` reference at `vendor/sheeprl/`. It is the **single
source of truth** for "what's different" — the bit-identity test suite
(Lever A in the [implementation plan](IMPLEMENTATION_PLAN.md#lever-a--per-function-bit-identity-test-suite))
fails at `1e-6` for any deviation, so any failing test either becomes a fix or a
row in this log.

**The user's binding constraint is "nothing has to be changed in the meaning of
functions."** This log is the mechanism that enforces it: every deviation is
visible, every deviation has a sheeprl-source citation, and every deviation is
PI-signed-off before merge.

## Schema

| Field | Description |
|---|---|
| **ID** | `D-NNN`, monotonically increasing. Used as anchor in plan + tests + code. |
| **CP** | Which checkpoint the deviation surfaced in (`CP1`–`CP10`). |
| **Function** | The JAX function name (and file path under `src/algorithms/dreamer_srl/`). |
| **Sheeprl source line** | Path under `vendor/sheeprl/` + line range (e.g. `sheeprl/utils/distribution.py:L237`). Pinned to commit `33b6366`. |
| **What we did instead** | One sentence describing the JAX deviation. |
| **Why** | One sentence (or short paragraph) — the rationale. Must reference either a JAX-platform constraint or a measured numerical effect (e.g. `max_abs_diff = 3.2e-7` against sheeprl). |
| **Bit-identity test result** | The `max_abs_diff` measured by the paired Lever-A test, or `n/a` if the deviation is structural (e.g. dropped function). |
| **PI verdict** | `☐ pending`, `✅ APPROVED — <date>`, `❌ REJECTED — fix required`. |
| **Resolution link** | If rejected, link to the fix commit; if approved, link to the PI review note (typically a comment block below the table or a `docs/pi/` entry). |

## Enforcement rules

1. **No silent deviations.** If a Lever-A bit-identity test exceeds `1e-6`, OR
   the developer cannot match sheeprl line-for-line (missing JAX API, structural
   incompatibility, conscious simplification), the developer logs an entry here
   **before** marking the function done. The CP gate does not close until the
   PI verdict is `✅ APPROVED` or the deviation is resolved.
2. **Before the parity-gate launch (the 3-seed run at the end of CP8)**, this log
   must have **zero `☐ pending` rows**. Every entry is either approved or
   resolved.
3. **The PI reviews per CP**, not per deviation in real time. When a CP closes,
   the senior-developer pings `pi` with the per-CP delta to this log (the new
   rows since the last CP).
4. **Approval criteria** the PI typically uses:
   - JAX-platform constraint (e.g. no float64 path on this GPU) ⇒ likely ✅.
   - "Simplification for readability" ⇒ likely ❌. The whole point is bit-identity.
   - Measured `max_abs_diff` ≤ `1e-5` AND the function is in the forward-pass
     (not the gradient flow) ⇒ likely ✅ with a note.
   - Measured `max_abs_diff` > `1e-4` ⇒ likely ❌; the deviation is too large to
     be float-precision drift, it's a semantic difference.

## Deviation table

> Add new rows at the bottom; never reuse a retired ID.

| ID | CP | Function | Sheeprl source line | What we did instead | Why | Bit-identity test result | PI verdict | Resolution link |
|---|---|---|---|---|---|---|---|---|
| D-001 | CP1 | `moments_update` (`utils.py`) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L57` | Omitted `fabric.all_gather(x)` call; use `x` directly | We run single-process (single GPU, no Lightning Fabric). `all_gather` is a no-op in single-process mode — the gathered tensor is identical to the input tensor. The numerical output is bit-identical to single-process sheeprl. | `max_abs_diff = 8.2e-8` (within `1e-6` threshold) | ✅ APPROVED — 2026-05-13 ([pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md)) | — |
| D-002 | CP1 | `init_weights`, `uniform_init_weights` (`utils.py`) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L143-L186` | Distribution property test instead of bit-identity comparison | JAX's `jax.random.truncated_normal` / `jax.random.uniform` use a different RNG implementation than PyTorch's `nn.init.trunc_normal_` / `nn.init.uniform_`. The mathematical formula (fan-avg scale, Hafner constant, truncation bounds) is identical. Individual kernel values cannot be bit-compared across different RNG streams. Test verifies: same shape, same `std_theoretical`, bounds respected (`max_abs_val <= limit`), zero-init case produces all-zeros. | n/a — distribution test; statistical properties pass (std rel-err=12%, within 15% bound) | ✅ APPROVED — 2026-05-13 ([pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md)) — conditional on F2 fixture tighten (`out_features` 256 → 16384, test bound 15% → <1%); follow-up commit on `v1.4` | — |
| D-003 | CP1 | `symexp` (`utils.py`) | `vendor/sheeprl/sheeprl/utils/utils.py:L152-L153` | Threshold relaxed to `2e-5` for the bit-identity test | JAX's CUDA float32 `exp` and PyTorch's CUDA float32 `exp` differ by up to 1 ULP for large `|x|` (~5). At `|x|=5`, `exp(5)≈148` and 1 ULP is ~`1.5e-5`. Maximum observed `max_abs_diff = 1.526e-5` (= exactly 1 ULP at the fixture's largest x). Maximum relative diff = `2.1e-7` (< 0.25 ULP relative), confirming this is a platform rounding difference not a semantic error. The mathematical formula `sign(x)*(exp(|x|)-1)` is identical to sheeprl's. | `max_abs_diff = 1.526e-5` (exceeds `1e-6`; within `2e-5` relaxed threshold) | ✅ APPROVED — 2026-05-13 ([pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md)) | — |
| D-004 | CP3b | `SequentialReplayBuffer` (`buffers.py`) — entire memmap path | `vendor/sheeprl/sheeprl/data/buffers.py:60-78` (memmap dispatch), `:203-211` (memmap-array allocation), `:349-356` (memmap setter); `vendor/sheeprl/sheeprl/utils/memmap.py` (whole module) | Omitted `memmap` / `memmap_dir` / `memmap_mode` argument trio entirely; JAX buffer stores arrays in-RAM only | We run with `buffer_size=1_000_000` transitions which fits comfortably in RAM on the lab nodes; sheeprl's memmap path is a scale convenience for huge buffers, not a semantic requirement. Memmap mode is purely how-stored, not what-stored — the in-memory tensors `_buf[k]` are byte-identical between the two modes. The v2 plan declared this at design time (v2 §"Non-goals" line referencing memmap; v2 §"Scaffolding to reuse from `train.py` and `src/`" reiterates "No memmap replay"); D-004 makes the omission an explicit PI-ratifiable deviation entry alongside D-001/D-002/D-003 rather than an implicit one. | n/a — structural omission (the function does not exist on the JAX side); CP3b's six Lever-A tests cover the in-memory storage and sample paths that *do* exist | ☐ pending (PI ratification fires alongside CP3b plan-revision approval; see [CP3B_SPEC.md](CP3B_SPEC.md)) | — |
| D-005 | CP3b | `test_buffer_state_evolution_matches_sheeprl` (test scope only) | `vendor/sheeprl/sheeprl/data/buffers.py:L212-L215` (`np.empty` allocation) | Bit-identity comparison restricted to filled region `[:_pos]`; unfilled slots `[_pos:]` excluded from diff | Both sheeprl and our JAX buffer allocate with `np.empty` — the unfilled portion `[_pos:]` contains uninitialized host memory values that are undefined and differ between the two Python allocations. This is not a semantic divergence in the buffer logic; the _written_ state `[:_pos]` is byte-identical (confirmed by test). Comparing the unfilled portion would compare garbage values and is meaningless. The semantic invariant — "every written slot stores the same byte pattern" — is preserved. | `max_abs_diff = 0.0` (filled region only; unfilled region excluded) | ☐ pending (PI ratification fires alongside CP3b; logged here as D-005 per the no-silent-deviations rule) | — |

## Approved deviations — PI rationale notes

> When the PI approves a deviation, the rationale goes here (with date and the
> `D-NNN` anchor). One block per approval. This is the audit trail for
> "we deliberately did not match sheeprl exactly because <X>."

### D-001 — APPROVED 2026-05-13

JAX-mechanical + Fabric-architectural. `flax.struct.dataclass` is the standard JAX pattern for state-carrying objects that must JIT; in-place state mutation is structurally disallowed inside JIT, not a stylistic choice. `fabric.all_gather(x)` is a no-op in single-process mode (the only mode we run), so dropping it is mathematically equivalent. Measured `max_abs_diff = 8.2e-8` is well inside the `1e-6` threshold. The "meaning" of `Moments` — what EMA the running statistics implement, with what decay constants and clamping — is unchanged. The user's binding constraint ("nothing changed in the meaning of functions") is preserved. See [pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md#d-001--moments-pure-functional--all_gather-dropped).

### D-002 — APPROVED 2026-05-13 (conditional on F2 fixture tighten)

Cross-PRNG fundamental. PyTorch's `nn.init.trunc_normal_` and JAX's `jax.random.truncated_normal` consume RNG state in different orders and use different underlying PRNG algorithms; bit-identity across the two streams is mathematically impossible. The mathematical formula (fan-avg scale, Hafner constant `0.99996…`, truncation bounds) is identical line-for-line with sheeprl, and the test verifies the resulting distribution. The approval is **conditional** on tightening the property test: raise `out_features` 256 → 16384 in [`scripts/fixtures/gen_cp1_fixtures.py`](../../../../scripts/fixtures/gen_cp1_fixtures.py) (a 64× sample-size increase → 8× tighter empirical-std variance), regenerate the `.npz` fixtures, and tighten the test bound from 15% to <1%. This makes the historical Hafner-truncation bug class (the `0.99996 → 0.8796` ~12% drift that wasted six months) re-detectable at >10× margin. F2 is a follow-up `developer` task on branch `v1.4`. See [pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md#d-002--init_weights--uniform_init_weights-tested-by-distribution-property-not-bit-identity) and the F2 section in the same call.

### D-003 — APPROVED 2026-05-13

Hardware float32 ULP drift. JAX's CUDA float32 `exp` and PyTorch's CUDA float32 `exp` differ by up to one ULP at large `|x|`; the measured `max_abs_diff = 1.526e-5` is exactly one ULP at `exp(5) ≈ 148`, and the corresponding relative error is `2.1e-7` (less than a quarter of a ULP relative) — the signature of platform rounding, not semantic divergence. The formula `sign(x) * (exp(|x|) - 1)` is identical to sheeprl's. The relaxed threshold (`2e-5`) is still over six orders of magnitude tighter than what would let a sign error or off-by-one constant through. See [pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md](../../../pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md#d-003--symexp-test-threshold-relaxed-from-1e-6-to-2e-5).

## Rejected deviations — fix-required notes

> When the PI rejects a deviation, the rejection rationale and the required fix
> direction go here. One block per rejection.

_(no rejections yet)_

## Links

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — the v3 plan that defines
  the five guardrail layers
- [Lever A — bit-identity test suite](IMPLEMENTATION_PLAN.md#lever-a--per-function-bit-identity-test-suite)
- [Lever B — source-citation discipline](IMPLEMENTATION_PLAN.md#lever-b--source-citation-discipline)
- [Lever C — per-checkpoint 3-reviewer gate](IMPLEMENTATION_PLAN.md#lever-c--per-checkpoint-3-reviewer-gate)
- [Lever D — vendored sheeprl + diff tool](IMPLEMENTATION_PLAN.md#lever-d--vendored-sheeprl--diff-tool)
- [Lever E — this file](IMPLEMENTATION_PLAN.md#lever-e--deviation_logmd--pi-consultation)
- [PI agent profile](../../../../.claude/agents/pi.md) — for how PI consultation
  works mechanically
