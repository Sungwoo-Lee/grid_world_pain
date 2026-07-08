---
title: "Code review — WP-SRL parity fixes (P1-P8, dreamer_srl)"
topic: dreamer_srl
status: active
created: 2026-07-08
last_updated: 2026-07-08
---

# Review: WP-SRL parity fixes for the JAX DreamerV3 port

## Verdict

**APPROVE-WITH-NITS.** This diff fixes seven divergences between our JAX DreamerV3
training code (`dreamer_srl`) and the sheeprl reference implementation it ports. The
three most consequential: gradient clipping was missing entirely (a single exploding
gradient could freeze the optimizer for thousands of steps — now all three optimizers
clip before Adam, at the reference norms); the world-model's observation loss used a
divergent inline formula that under-weighted reconstruction exactly 2x and trained the
decoder in the wrong space (now routed through the faithful ported loss function); and
the replay buffer shared one write head across all parallel environments, so every
episode-end in one environment punched a garbage "hole row" into every other
environment's stored history (now each environment gets its own independent buffer,
matching sheeprl's design). I verified the JAX correctness of each change — gradient
paths, JIT shape stability, PRNG stream preservation, and the ordering of the
episode-boundary bookkeeping — and ran all six new/extended test files plus the
neighboring suites (73 tests pass, 9 GPU-only skips). I found **no blockers**: one
low-severity concern (a one-step counter leak that survives only at curriculum
stage swaps) and three nits. The fix set is correct and safe to commit.

Reviewed scope: uncommitted changes to `src/algorithms/dreamer_srl/{utils,loss,train,dreamer_srl_main,buffers}.py`
and `tests/algorithms/dreamer_srl/` (3 new + 3 extended files). Plan:
[[fix_plan_srl_parity]] (`docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/fix_plan_srl_parity.md`).

## Findings

| # | Severity | Location | Issue | Suggested fix |
|---|----------|----------|-------|---------------|
| C1 | 🟡 concern | `src/algorithms/dreamer_srl/dreamer_srl_main.py:1592` vs `:1436` | Curriculum stage-swap edge of the P5 fix: on a stage-transition iteration, step 4 of the swap block wipes ALL envs' episode counters (line 1436), but `_advance_episode_counters` then runs (line 1592) with the **pre-swap** `rewards`/`dones` — non-done envs get +1 length and the pre-swap step's reward credited to the new stage's first episode. One step per swap, curriculum mode only; partially pre-existing (the old unconditional increment leaked for all envs incl. done ones). | Track a `_stage_swapped_this_iter` flag and skip the advance, or move the counter wipe (step 4) to after line 1592. |
| N1 | 🟢 nit | `src/algorithms/dreamer_srl/buffers.py:598` + `dreamer_srl_main.py:706` | `ready_to_sample()` returns `True` for a wrapped-full buffer even when `sequence_length > _buffer_size`; `sample()` then raises (`buffers.py:367-371`). With P2's `buffer_size // num_envs` sizing, a small configured buffer + many envs makes `per_env_buffer_size < seq_len` reachable — the run crashes only at the first post-prefill sample instead of at startup. | Fail fast in the driver: assert `buffer_size // num_envs >= seq_len` at buffer construction. |
| N2 | 🟢 nit | `src/algorithms/dreamer_srl/agent.py:1283, 1655, 1712` | P3 silently redefines `reconstructed_obs` semantics: the raw decoder output is now the **symlog-space** prediction (real-space only via `symexp` at `SymlogDistribution.mode/mean`). The decoder/`observe` docstrings still describe it as the reconstructed observation with no space annotation — a future consumer (e.g. eval-time reconstruction rendering) could plot symlog values as real ones. Only consumer today is the loss (verified). | One-line docstring update at the three sites: "symlog-space; apply `symexp` for real-space values". |
| N3 | 🟢 nit | `tests/algorithms/dreamer_srl/test_lax_scan_train.py:564, 592` | The two P7 guards are source-text regex assertions, not behavioral tests — they pass as long as the strings `_G = max(1, ...)` and `n_grad_steps_scan = _G` exist, even if a refactor made those lines dead. Acceptable as declared tripwires (the accumulator was loop-local and unimportable), noting the limitation. | Optional: complement with a behavioral check if the grad-step dispatch ever gets factored into a testable function. |

## Per-fix audit

**P1 — gradient clipping (`utils.py` `make_optim_tx`, `dreamer_srl_main.py:701-703`).** ✅
Clip placed **inside** `optax.chain` **before** `optax.adam` — clips raw gradients, then
Adam consumes the clipped gradients; this is the correct analogue of torch's
`clip_grad_norm_` before `optimizer.step()`, and critically it protects Adam's second
moment (clipping *updates* after Adam would not). No pre-existing clip anywhere in
`train.py`/`dreamer_srl_main.py` (grepped) — no double-counting. `wrt=nnx.Param`
preserved on all three `nnx.Optimizer` wrappers, and the scan path's
`nnx.split`/`nnx.update` round-trip of the optimizer states is structure-generic, so
the extra `EmptyState` the chain prepends is carried transparently. Checkpoint
compatibility is a non-issue: `_save_checkpoint` (`dreamer_srl_main.py:1488-1502`)
does not persist optimizer state. Norms 1000/100/100 pinned as constants with an
explicit "recipe constant, not config" rationale — consistent with how the two-hot
bin count is handled; the no-fallback-defaults rule is not violated because nothing
is read from config with a default.

**P3 — faithful observation loss (`loss.py:49-87` `SymlogDistribution`, `train.py:703-808`).** ✅
The gradient path is right: `log_prob = -Σ_dims (mode − symlog(target))²` applies
`symlog` to the **target** only; the raw decoder output is the symlog-space prediction
and receives gradient directly. The `tol=1e-8` clamp uses `jnp.where` on a traced
value — element-wise select, no Python control flow leak, and zero gradient in the
sub-tolerance branch exactly like the torch reference. No dead-code inversion: the
`total` returned by `reconstruction_loss` **is** the value returned by `wm_loss_fn`
and consumed by `nnx.value_and_grad` at `train.py:810-814` (the exact failure mode
this fix addresses — verified line-by-line). Reduction scale is consistent: obs,
reward, continue, and KL terms are all per-`[T,B]` sums over event dims, jointly
`.mean()`-ed once (`loss.py:613`), matching sheeprl's sum-over-dims / mean-over-[T,B]
form. The logging-only dyn/rep KL recompute (`train.py:758-764`) is sound —
`KL(stop_grad(p)‖q)` and `KL(p‖stop_grad(q))` are value-identical, and the recompute
is wrapped in `stop_gradient`, so it contributes nothing to the objective.

**P2 — per-env independent buffers (`buffers.py:683-871`, `dreamer_srl_main.py:706-735`).** ✅
- *Done-mask routing:* `np.where(done_mask)[0]` yields absolute env indices; sub-buffer
  `e` receives column `v[:, e:e+1]` — index identity preserved, non-done heads untouched.
  Verified against the two-rows-per-done pattern: for a done env the sequence is
  (regular row at iteration top) → (reset row with true terminal obs, `is_first=0`,
  via `done_mask`) → (next iteration's fresh-obs row with `is_first=1`), landing in the
  right env's sub-buffer; the `[21,31,0,41]` hole-row fixture confirms non-done envs
  stay contiguous.
- *JIT shape stability:* bincount allocation is host-side NumPy; per-sub-buffer sample
  sizes vary but the axis-2 concat always restores exactly `batch_size`, so the array
  entering `_scan_grad_steps` is `[_G, seq_len, batch_size, ...]` every iteration —
  no recompile trigger. The GPU-traced path is correctly walled off (`num_envs==1`
  hard `ValueError`), and at `num_envs==1` the plain buffer is semantically a
  wrapper-of-one.
- *Wrap arithmetic:* each sub-buffer wraps independently at `buffer_size // num_envs`;
  done envs fill faster (extra reset rows) — handled per-head. Division semantics
  match sheeprl `dreamer_v3.py:478`, done in the driver as documented.
- *Sequence assembly:* sub-buffers are `n_envs=1`, so sequences physically cannot
  cross env boundaries; the wrapped-full valid-index exclusion around `_pos`
  (`buffers.py:415-432`, incl. the negative-`first_range_end` wrap case) is the
  pre-existing, sheeprl-matching logic and composes correctly with P8.
- The superseded shared-head `done_mask` branch on the base class is retained with an
  accurate SUPERSEDED note and remains valid at `n_envs==1` (the GPU path) — good.

**P5 — episode-counter reorder (`dreamer_srl_main.py:371-388, 1592`).** ✅ (with C1)
The masked advance runs at the same loop position as the old unconditional increment —
strictly **after** the done block — so the H5-critical ordering inside the done block
(`buffer.add(reset_data, done_mask)` at :1333 **before** `_reset_terminal_step_data`
at :1349) is untouched. Done-env logging (`counters+1`, `counters+rewards[i]`) plus
the alive-only advance is exact for back-to-back episodes. Remaining edge is C1
(curriculum swaps only).

**P6 — `derive_prefill` (`utils.py:59-84`).** ✅ Line-for-line sheeprl
`dreamer_v3.py:508-511` at `world_size==1`, including the intentional off-by-one in
`prefill_steps`. `ratio_steps = policy_step - prefill_steps * num_envs` (:1607)
matches L661; at the first gated iteration the scheduler sees `num_envs` steps — the
old one-shot debt-repayment burst is gone. `derive_prefill(0, n) == (0, 0)` keeps the
zero-prefill smoke path. Note (accepted, sheeprl-identical): non-divisible
`learning_starts_cfg` floor-truncates, and `cfg < num_envs` yields zero prefill.

**P7 — remainder deletion.** ✅ No references to `_grad_step_remainder` remain
anywhere in `src/`/`scripts/` (grepped); only the tripwire test mentions it. `_G`
quantization unchanged; legacy path keeps exact Ratio cadence.

**P8 — `ready_to_sample` (`buffers.py:576-591, 855-859`).** ✅ `_full or _pos >= seq_len`
is exactly the condition under which `sample()` succeeds (its wrapped-full branch
handles any `_pos`, including `_pos < seq_len`, via the two-range valid-index set).
Defined on **both** classes, so the GPU path (plain buffer) is gated identically —
not incorrectly blocked. The wrapper's `all(...)` over sub-buffers is safe: regular
adds advance every head in lockstep and done adds only add extra rows, so no
sub-buffer ever lags the gate. Edge case N1 noted above.

## Cross-cutting checks

**PRNG threading.** ✅ Stream-preserving. No `jax.random` split was added, removed, or
reordered in the diff (the deleted inline loss consumed no keys; `reconstruction_loss`
consumes none; `k_wm`/`k_player`/`k_autoreset`/GPU `sample_key` usage unchanged). The
new host-side `np.random.default_rng()` in the wrapper (`buffers.py:749`) is unseeded
by design — it mirrors sheeprl and the base class's own pre-existing unseeded RNG
(`buffers.py:76`); the reproducibility caveat is explicitly carried over (P-14 note in
the class docstring). Not a regression.

**Tests — discriminative power.** ✅ Genuinely discriminating, not absence-only:
- The ν-poisoning test (`test_grad_clip.py:63-118`) runs **both arms**: clipped final
  update norm must stay > 1e-5 AND the unclipped control must freeze below 1e-6 — the
  second arm guards fixture vacuity.
- The hole-row fixture (`test_env_independent_buffer.py:51-90`) asserts exact stored
  contents `[21,31,41]` / `[20,30,999,40]` and documents the pre-fix `[21,31,0,41]`.
- The 2x-ratio test pins `new/old == 2.0` exactly (no sub-tol elements → clamp inert),
  and the clamp test uses a full sub-tol row so float32 absorption can't hide a missing
  clamp — with an explicit anti-vacuity assertion.
- Episode-metrics fixture discriminates (2, 12.0) vs the old (3, 22.0).
- P7's two tests are source-text tripwires (N3) — declared as such.

**Test execution** (CPU, `grid_world_pain` env): `test_env_independent_buffer` +
`test_episode_metrics` + `test_grad_clip` + `test_prefill` + `test_loss` → 24 passed;
`test_lax_scan_train` → 7 passed; neighboring `test_buffers`, `test_buffer_reset`,
`test_terminal_step_data_reset`, `test_continual_schedule`, `test_gpu_buffer`,
`test_utils` → 47 passed, 9 skipped (GPU-only).

## Conventions audit

| Convention | Status | Note |
|---|---|---|
| Pytree immutability | ✅ | Buffers are deliberate host-side mutable objects (documented design, never crosses JIT); env pytrees updated via `.at[].set()` only. |
| JIT recompilation | ✅ | Concat restores constant `batch_size`; `_G` constant scan length preserved; GPU/bincount interaction correctly forbidden. |
| vmap/batch conventions | ✅ | No new vmap; env-axis-0 conventions untouched; per-env autoreset still uses `tree_map` + `.at[env_idx]`. |
| PRNG threading | ✅ | No new key consumption or reordering; numpy RNG deviation declared. |
| Sensor/obs-breakdown sync | ✅ n/a | No sensor or observation-layout code touched. |
| Config protocol | ✅ | `algo.learning_starts` still `get_mandatory`; clip norms are recipe constants with explicit rationale, not silent defaults; no new YAML keys in the reviewed files. |

## Conclusion

APPROVE-WITH-NITS — no blockers; one curriculum-only counter edge (C1) and three nits, none commit-gating.

Reviewed by: code-reviewer
