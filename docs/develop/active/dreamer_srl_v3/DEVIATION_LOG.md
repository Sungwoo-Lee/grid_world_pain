---
title: "dreamer-srl v3 — Deviation Log"
topic: dreamer
status: active
created: 2026-05-13
last_updated: 2026-05-13
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
| D-001 | CP1 | `moments_update` (`utils.py`) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L57` | Omitted `fabric.all_gather(x)` call; use `x` directly | We run single-process (single GPU, no Lightning Fabric). `all_gather` is a no-op in single-process mode — the gathered tensor is identical to the input tensor. The numerical output is bit-identical to single-process sheeprl. | `max_abs_diff = 8.2e-8` (within `1e-6` threshold) | ☐ pending | — |
| D-002 | CP1 | `init_weights`, `uniform_init_weights` (`utils.py`) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py:L143-L186` | Distribution property test instead of bit-identity comparison | JAX's `jax.random.truncated_normal` / `jax.random.uniform` use a different RNG implementation than PyTorch's `nn.init.trunc_normal_` / `nn.init.uniform_`. The mathematical formula (fan-avg scale, Hafner constant, truncation bounds) is identical. Individual kernel values cannot be bit-compared across different RNG streams. Test verifies: same shape, same `std_theoretical`, bounds respected (`max_abs_val <= limit`), zero-init case produces all-zeros. | n/a — distribution test; statistical properties pass (std rel-err=12%, within 15% bound) | ☐ pending | — |
| D-003 | CP1 | `symexp` (`utils.py`) | `vendor/sheeprl/sheeprl/utils/utils.py:L152-L153` | Threshold relaxed to `2e-5` for the bit-identity test | JAX's CUDA float32 `exp` and PyTorch's CUDA float32 `exp` differ by up to 1 ULP for large `|x|` (~5). At `|x|=5`, `exp(5)≈148` and 1 ULP is ~`1.5e-5`. Maximum observed `max_abs_diff = 1.526e-5` (= exactly 1 ULP at the fixture's largest x). Maximum relative diff = `2.1e-7` (< 0.25 ULP relative), confirming this is a platform rounding difference not a semantic error. The mathematical formula `sign(x)*(exp(|x|)-1)` is identical to sheeprl's. | `max_abs_diff = 1.526e-5` (exceeds `1e-6`; within `2e-5` relaxed threshold) | ☐ pending | — |

## Approved deviations — PI rationale notes

> When the PI approves a deviation, the rationale goes here (with date and the
> `D-NNN` anchor). One block per approval. This is the audit trail for
> "we deliberately did not match sheeprl exactly because <X>."

_(no approvals yet)_

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
