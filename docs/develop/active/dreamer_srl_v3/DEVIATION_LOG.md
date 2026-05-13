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
| _example_ | _CP5_ | `TwoHotEncoding.log_prob` (`loss.py`) | `vendor/sheeprl/sheeprl/utils/distribution.py:L237-L260` | Reduction in float32 (JAX default) where sheeprl uses float64 internally for the `torch.nn.functional.one_hot` indexing | JAX has no efficient float64 path on the RTX 6000 Ada; performing the reduction in float64 would 4× the step time. Forward-pass only — does not affect gradient flow. | `max_abs_diff = 3.2e-7` (over `1e-6` threshold by ~3×) | ☐ pending | _(no rows yet)_ |

<!-- DELETE THE EXAMPLE ROW WHEN THE FIRST REAL ROW IS ADDED. -->

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
