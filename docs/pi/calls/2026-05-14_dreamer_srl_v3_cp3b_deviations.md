---
title: "PI Call — dreamer-srl v3 CP3b deviation gate: approve D-004 (memmap omission) + D-005 (filled-region test slice)"
date: 2026-05-14
trigger: CP3b deviation gate — buffer + cadence layer
status: decided
---

# PI Call — dreamer-srl v3 CP3b deviation gate: approve D-004 (memmap omission) + D-005 (filled-region test slice)

## Question

**At the close of CP3b — the second algorithmic checkpoint in the dreamer-srl v3 rebuild, covering the replay buffer (`buffers.py`'s `SequentialReplayBuffer`) and the training-cadence wiring (`learning_starts` / `prefill_steps` / `Ratio(replay_ratio)`) — do we approve the two logged differences between our JAX implementation and the vendored sheeprl reference?**

The two deviations are:

- **D-004** — the JAX buffer drops sheeprl's `memmap` storage path entirely (RAM-only, no on-disk backing).
- **D-005** — Test 1's bit-identity comparison is scoped to the filled region `[:_pos]` of the buffer, excluding the uninitialised tail `[_pos:]`.

## Headline

**Both APPROVED.** CP3b's four-gate closure (Lever A 6/6 PASS at `0.000e+00`; Lever B source-citation discipline verified by code-reviewer; Lever C three-reviewer chain all `PASS`; Lever E both deviations now `APPROVED`) is **complete**. The CP3b row in the v3 plan's checkpoint table is eligible to flip from `IN PROGRESS` to `CP-PASS (2026-05-14)`. That flip is a `senior-developer` task — the same pattern as the CP1 → CP-PASS transition on 2026-05-13. With CP3b closed, the next slot in the user-reordered build queue is **CP5** (the two-hot reward-distribution port — the historical-scar checkpoint that motivated this entire five-lever guardrail system).

## Context for a fresh reader

The project is rebuilding DreamerV3 (the world-model RL agent we use) from scratch in JAX, copying the working PyTorch reference at [`vendor/sheeprl/`](../../../../vendor/sheeprl/) commit `33b6366` function-by-function. The rebuild is called **dreamer-srl v3** and is gated by a five-lever guardrail system designed to prevent the kind of silent off-by-a-constant bug that wasted six months last time. The five levers (Lever A bit-identity tests at `1e-6`, Lever B source-citation discipline, Lever C three-reviewer chain, Lever D vendored sheeprl, Lever E this deviation log) are summarised in the prior PI call ([2026-05-13_dreamer_srl_v3_cp1_deviations.md](2026-05-13_dreamer_srl_v3_cp1_deviations.md)) and defined in full in the [v3 implementation plan](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md).

### Why CP3b exists at all

The original v3 plan listed the buffer + cadence layer as "no Lever-A gate; integration smoke only." That gap silently let through a **16× `replay_ratio` cadence drift** at the parity-gate launch — both the JAX-side YAML and the sheeprl-side YAML wrote `replay_ratio: 1.0`, but the JAX outer-loop wiring was firing gradient updates at one-sixteenth the rate of sheeprl's. The arithmetic at the integration point — `learning_starts // policy_steps_per_iter` (sheeprl `dreamer_v3.py:510`), `prefill_steps = learning_starts - int(learning_starts > 0)` (sheeprl `:511`), and `per_rank_gradient_steps = ratio(ratio_steps / world_size)` (sheeprl `:661`) — looked correct in three isolated places, and the integration drift only surfaced as a measurable parity gap (see [SPS_COMPARISON_JAX_VS_SHEEPRL.md §3.6](../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#36-where-the-jax-advantage-is-coming-from)).

CP3b was promoted into a real checkpoint on 2026-05-13 with [CP3B_SPEC.md](../../develop/active/dreamer_srl_v3/CP3B_SPEC.md), defining six Lever-A tests (four buffer-storage / sample state-evolution tests + two cadence-trace tests) at the same `1e-6` threshold the other checkpoints use. The `developer` agent landed the port (commit `9c57c06`); all six tests PASS at `max_abs_diff = 0.000e+00`; the three-reviewer chain (code → math → professor-rl-bayesian-dl) issued PASS verdicts. Two deviations were logged for PI review under the no-silent-deviations rule.

### The "Strong (A+B+C+D+E)" deviation-prevention strategy

The Strong strategy — that all five levers must close before a CP closes — exists because reviewer-PASS alone is not enough. The historical Hafner-truncation bug ran the same three-reviewer chain on each cascade fix one at a time and the bug class still wasted six months. Lever E (this gate) adds an explicit PI portfolio question on top of the technical-correctness review: **is this deviation the kind of drift that could compound into a multi-week parity gap?** For each item below, the answer is no — and the technical-reviewer chain backs that up.

## The two deviations

### D-004 — `SequentialReplayBuffer` drops the `memmap` storage path entirely

**Sheeprl source.** Sheeprl's `SequentialReplayBuffer` (and its parent `ReplayBuffer`) supports two storage backends: in-RAM (`np.ndarray`) and on-disk-paged (`MemmapArray`, a thin wrapper around `np.memmap`). The selection is made by the `memmap` / `memmap_dir` / `memmap_mode` argument trio at construction time. Source: [`vendor/sheeprl/sheeprl/data/buffers.py:60-78`](../../../../vendor/sheeprl/sheeprl/data/buffers.py) (constructor dispatch), `:203-211` (memmap allocation), `:349-356` (memmap setter), plus the whole [`vendor/sheeprl/sheeprl/utils/memmap.py`](../../../../vendor/sheeprl/sheeprl/utils/memmap.py) module.

**What the JAX code does instead.** The JAX `SequentialReplayBuffer` omits the entire memmap path — no `memmap` constructor argument, no `MemmapArray` wrapper, no on-disk backing file. Storage is `np.ndarray`-only.

**Why.** We run with `buffer_size = 1_000_000` transitions, which fits comfortably in RAM on the lab nodes. Sheeprl's memmap path is a **scale convenience** for huge buffers (tens of millions of transitions on memory-constrained hardware), not an algorithmic requirement. Memmap-vs-RAM is a **how-stored** distinction, not a **what-stored** distinction: in both modes, the bytes accessible via `self._buf[k][idxes]` are identical, because `np.memmap.__setitem__` and `np.ndarray.__setitem__` are byte-equivalent for fully-aligned writes.

**Why this is safe.** The omission is **structural** — the memmap code path does not exist on the JAX side, and it would not be exercised in either training-loop or test even if it did. No equation in CP3b's math review is sensitive to the storage backend (math-reviewer Eq. 1–8 all PASS under either backend). The professor-rl-bayesian-dl review explicitly confirms that "no `is_first` placement, `action` storage timing, `_pos` arithmetic, `_full` flag transition, or sample-window arithmetic differs between modes." The user's binding constraint ("nothing has to be changed in the meaning of functions") is preserved — what's missing is a storage backend, not a function semantic.

This deviation was **pre-declared at the v2 plan stage** (the v2 "Non-goals" section already excluded memmap replay); D-004 promotes it from an implicit omission to an explicit PI-ratifiable entry alongside D-001/D-002/D-003, matching the no-silent-deviations rule.

### D-005 — Test 1's bit-identity comparison is restricted to the filled region `[:_pos]`

**Sheeprl source.** Sheeprl's `ReplayBuffer.__init__` allocates each storage tensor with `np.empty(shape, dtype)` (or its `MemmapArray` equivalent) — see [`vendor/sheeprl/sheeprl/data/buffers.py:212-215`](../../../../vendor/sheeprl/sheeprl/data/buffers.py). `np.empty` returns **uninitialised host memory**: the bytes at the allocated address are whatever happened to be there before the allocation, and they differ between Python processes and across runs of the same process.

**What the JAX code does instead.** The JAX port allocates with `np.empty` too (the line-for-line port carries the allocator over). But the **test** — `test_buffer_state_evolution_matches_sheeprl` — restricts its byte-identity comparison to the filled region `[:_pos]` of the buffer, excluding the unfilled tail `[_pos:]` from the diff.

**Why.** Both implementations write the same bytes into the same logical slots in `[:_pos]` (verified by the test: `max_abs_diff = 0.000e+00`). But the unfilled tail `[_pos:]` contains the uninitialised host memory left over from `np.empty`, which is **garbage and undefined**. Comparing it across two separate Python allocations would be comparing arbitrary platform-rounding artefacts that have nothing to do with buffer semantics. The semantic invariant the test is trying to verify — *"every written slot stores the same byte pattern in both implementations"* — is preserved at `[:_pos]`, and is meaningless at `[_pos:]`.

**Why this is safe.** Three independent guardrails in Test 1 itself prevent a false-PASS from a hypothetical "JAX writes garbage to `[:_pos]` but reports a matching `_pos`" coordinated bug:

1. **`_pos` is asserted independently** (`tests/algorithms/dreamer_srl/test_buffers.py:L93`): `assert jax_rb._pos == expected_pos`. A `_pos`-arithmetic bug would fail this line *before* the `[:_pos]` slice comparison runs.
2. **`_full` is asserted independently** (`:L94`): `assert jax_rb._full == expected_full`. A wraparound-regime entry that would silently invalidate the `[:_pos]` semantics would fail this line.
3. **The write line is literally the same NumPy expression in both implementations** (one is `self.buffer[k][idxes] = ...`, the other is `self._buf[k][idxes] = ...`, with `self.buffer == self._buf` via sheeprl's property at L82-L83) — a "coordinated `_pos` bug" would require the JAX `add()` `next_pos` arithmetic to be correct (matching sheeprl) while the assignment line was wrong, which is implausible given the line-for-line port.

Additionally, Tests 2 / 3 / 4 do not reference the unfilled tail at all — they sample contiguous windows from explicit start indices in the filled region. D-005 is correctly scoped to Test 1 only.

The professor-rl-bayesian-dl review also confirms that the unfilled-tail bytes **never flow into the algorithm**: sheeprl's `sample()` raises `ValueError` if the buffer doesn't have `≥ seq_len` transitions yet, and the production training loop only samples past `learning_starts=1024` (`buffer-warmup ≫ seq_len=64`). Under sheeprl-XS semantics, the algorithm provably never reads bytes from `[_pos:]`, so the test scope exactly matches the algorithm's bytes-of-interest.

## Options considered

For each deviation, the option boxed `[X]` is the user's pick.

### D-004 — memmap storage path omitted entirely

1. **[X] APPROVE.** Storage backend, not algorithm semantics. Memmap-vs-RAM produces byte-identical reads/writes; the omitted path is never exercised in training-loop or test. Pre-declared at v2 plan stage; D-004 is the explicit-deviation-log form of the same decision. Code-reviewer: "no semantic risk; memmap is how-stored, not what-stored." Math-reviewer: "no equation in CP3b's math is sensitive to the storage backend." Professor-rl-bayesian-dl: "deployment-regime simplification."
2. REJECT — require a `MemmapArray`-equivalent JAX shim wrapping `np.memmap` for future huge-buffer scaling. *Cost:* a non-trivial wrapper for a capability we will not use in this paper; buffer-size ceiling is `1M` transitions, which fits in RAM.
3. DEFER — leave D-004 `☐ pending` and revisit at CP8 (the merge-gate that requires the deviation log to be empty). *Cost:* a known-acceptable deviation kept artificially open; CP3b cannot close until D-004 closes.

### D-005 — Test 1's `[:_pos]` filled-region slice

1. **[X] APPROVE.** Test-scope honesty about `np.empty`'s uninitialised-tail semantics. Filled region `[:_pos]` is byte-identical at `0.000e+00`; unfilled tail `[_pos:]` is undefined garbage by `np.empty`'s contract, identical in both implementations only by accident. False-PASS risk pre-checked: `_pos` and `_full` are independently asserted in the same test; the write line is the same NumPy expression in both implementations; Tests 2/3/4 don't reference the tail. Three independent reviewers confirmed: code-reviewer "the scope cut is sound"; math-reviewer "no hidden math deviation"; professor-rl-bayesian-dl "the test scope exactly matches the algorithm's bytes-of-interest."
2. REJECT — require a deterministic-fill convention (`np.zeros` instead of `np.empty`) so the unfilled tail is comparable. *Cost:* changes the buffer's allocator from line-for-line with sheeprl to deliberately-different; the deviation moves from test-scope to implementation-scope, which is a larger surface.
3. EXPAND SCOPE — require the test to compare the unfilled tail by computing both sides' tail bytes and asserting they match. *Cost:* mathematically impossible — `np.empty` bytes are not reproducible across Python processes.

## User decision

**D-004 APPROVE. D-005 APPROVE.**

Both picks match the PI's recommended option. Both match the unanimous verdict of the three technical reviewers (code-reviewer, math-reviewer, professor-rl-bayesian-dl). The two picks are coherent under the user's binding constraint ("nothing has to be changed in the meaning of functions"): D-004 omits a storage backend, not a function semantic; D-005 restricts a test's comparison region, not the buffer's behaviour.

## Rationale captured

- **The user's binding constraint is preserved.** Neither deviation changes what the buffer's functions compute. D-004 omits a parallel storage backend (the in-RAM path that IS implemented is byte-identical to the in-RAM path on the sheeprl side). D-005 adjusts the *test's* comparison region to match the algorithm's actual bytes-of-interest, while the buffer's allocator (`np.empty`) is line-for-line with sheeprl.
- **The cascade-debugging lesson is operationalised.** The 16× `replay_ratio` cadence drift — the empirical motivator that promoted CP3b from "no-CP sanity round-trip" to a full state-evolution checkpoint — is what Test 6 (the 5000-iter cadence-trace bit-identity test) catches. Tests 1+3+4 cover the buffer-storage substrate that downstream §S1+§S2+§S4 reset paths rely on. The substrate-mechanical / test-scope-honesty character of D-004 and D-005 is the OPPOSITE of the "silent semantic divergence" class CP3b exists to prevent — they are explicit, documented, reviewed, and ratified.
- **The three-reviewer chain converged independently.** Each reviewer applies a different lens (line-for-line port, equations under review, algorithm-fidelity over downstream §S consumers). All three concluded both deviations are substrate-mechanical and pose no algorithm risk. Lever E adds the portfolio question on top of that technical-correctness consensus.
- **No PI disagreement to log.** The PI recommended APPROVE for both; the user concurred; the technical-reviewer chain consensus matches.

## What this enables

CP3b's row in [the v3 checkpoint table](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#checkpoint-table-v3) is eligible to flip from `IN PROGRESS` to `CP-PASS (2026-05-14)`. The four gates are now all closed:

- **Lever A** — 6/6 paired bit-identity tests PASS at `max_abs_diff = 0.000e+00`.
- **Lever B** — line-for-line source citations verified by code-reviewer (sheeprl `buffers.py:L145-L221` for `add()`, `:L395-L465` for `sample()`, `:L467-L526` for `_get_samples()`, `:L489` for the flat-index formula, `dreamer_v3.py:L505-L515, L550-L551, L661-L662` for cadence wiring — all accurate against `vendor/sheeprl/` at commit `33b6366`).
- **Lever C** — three-reviewer chain all PASS ([code review](../../reviews/dreamer_srl_v3_cp3b_code_review.md), [math review](../../reviews/dreamer_srl_v3_cp3b_math_review.md), [professor-rl-bayesian-dl review](../../reviews/dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md)).
- **Lever E** — D-004 and D-005 both APPROVED in this call.

With CP3b closed, the next slot in the user-reordered build queue is **CP5** (the two-hot reward-distribution port — the historical-scar checkpoint whose `0.99996 → 0.8796` constant-truncation bug wasted six months at the v2 stage). CP5 is slot #3 in the [revised implementation order](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#implementation-order-revised) (after CP1 and CP3b), but slot #2 by historical-scar priority — the reason it was moved early in the build queue is precisely because the v3 plan's five-lever system was designed in response to this checkpoint's failure mode.

## Hand-off

- **This call doc** — written here (`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp3b_deviations.md`).
- **DEVIATION_LOG.md** — PI flips the verdict cells for D-004 and D-005 to `✅ APPROVED — 2026-05-14 (pi/calls/2026-05-14_dreamer_srl_v3_cp3b_deviations.md)`, and appends one rationale block per deviation under "Approved deviations — PI rationale notes." Done as part of this call.
- **Diary** — append a `note` row pointing at this call doc. Done as part of this call.
- **CP-PASS flip on the v3 plan's checkpoint table** — held by `senior-developer`, matching the CP1 → CP-PASS pattern. The PI closes Lever E; senior-developer flips the CP3b row from `IN PROGRESS` to `CP-PASS (2026-05-14)` in the v3 plan's checkpoint table and regenerates `docs/develop/INDEX.md`. Not done as part of this call (the call only signs off the deviations and confirms the four gates are closed).
- **CP5 start authorization** — separate decision from the user; the senior-developer does not spawn `developer` for CP5 without that explicit authorization (matching the CP1 → CP5 transition pattern on 2026-05-13).
- **Stop rule.** If the senior-developer's CP-PASS-flip step surfaces a state inconsistency (e.g. a test that PASSes individually but fails when run alongside another CP3b test, or a Lever-B citation that grep'ing fails to confirm against the pinned commit), escalate back to PI before flipping — that would indicate a gate that was reported closed but isn't.

## Links

- [DEVIATION_LOG.md](../../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) — the Lever-E log (D-004 and D-005 verdict cells flipped to APPROVED as part of this call; rationale-notes block appended).
- [CP3B_SPEC.md](../../develop/active/dreamer_srl_v3/CP3B_SPEC.md) — the spec that defines why CP3b exists (the 16× `replay_ratio` drift class) and the six Lever-A tests.
- [v3 implementation plan](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md) — five-lever guardrail system + checkpoint table (CP3b row at L519).
- [Lever E — DEVIATION_LOG.md + PI consultation](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#lever-e--deviation_logmd--pi-consultation) — the section that mandates this gate.
- CP3b reviewer audits — [code review](../../reviews/dreamer_srl_v3_cp3b_code_review.md), [math review](../../reviews/dreamer_srl_v3_cp3b_math_review.md), [professor-rl-bayesian-dl review](../../reviews/dreamer_srl_v3_cp3b_professor_rl_bayesian_dl_review.md).
- [Prior PI call — CP1 deviation gate (D-001/D-002/D-003 + F2 fixture tighten)](2026-05-13_dreamer_srl_v3_cp1_deviations.md) — style precedent and Lever-E pattern.
- [Prior PI call — Dreamer backend decision](2026-05-12_dreamer_backend.md) — the call that authorised the rebuild this gate is gating.
- [SPS_COMPARISON_JAX_VS_SHEEPRL.md §3.6](../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#36-where-the-jax-advantage-is-coming-from) — empirical motivation for promoting CP3b (the 16× `replay_ratio` drift class).
- CP3b port commit `9c57c06` — the developer's implementation.
