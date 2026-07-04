---
title: "Review — H4 fix: MC-mode returns bootstrap at the rollout-window edge (rPPO)"
topic: reviews
status: done
created: 2026-07-05
last_updated: 2026-07-05
---

# Review — H4 MC window-edge bootstrap (rPPO trainer)

## Verdict

**APPROVE (with nits).** This change fixes a real bias in every live recurrent-PPO
training run: experience is collected in fixed 128-step windows, and until now the
"Monte Carlo" return computation simply stopped at the window edge — a step near the
edge was credited with almost none of its true future reward, distorting the critic's
learning targets position-dependently. The fix seeds the reverse return-scan with the
critic's own estimate of the remaining future (a "bootstrap" value) taken from the true,
pre-auto-reset state after the window's last step. The bootstrap is zeroed only when the
agent actually died exactly at the edge; a timeout or a mid-episode window cut keeps it —
the same death-vs-truncation semantics the earlier GAE-path fix (commit `3c60f6f`)
established.

I hand-traced the reverse scan against all seven test cases, verified the vmap axis
alignment of the new arguments, confirmed the carried-out state really is the
pre-reset one, confirmed zero change to random-number consumption (the network forward
is pure — no dropout, no RNG use outside init), and ran the new test file (7/7 pass).
No blockers. Three informational nits below; none require code changes before commit.

Scope reviewed: the uncommitted diff to `src/models/recurrent_ppo_trainer.py` (+62/−18)
and the new `tests/models/test_mc_window_bootstrap.py`. The unrelated parallel work
package in the working tree (train.py, src/utils/*, tests/training/*) was not reviewed.

## Findings

| Sev | Location | Issue | Suggested fix |
|---|---|---|---|
| 🟢 nit | `tests/models/test_mc_window_bootstrap.py:14` | `import pytest` is unused (no fixtures, marks, or `pytest.approx`). | Drop the import (cosmetic; can ride along with any later touch). |
| 🟢 nit | `src/models/recurrent_ppo_trainer.py:90-131` | Semantic asymmetry, deliberate per plan: a timeout at the window edge bootstraps (infinite-horizon treatment) while a mid-window timeout still resets the return to 0 (finite-horizon treatment). So timeout steps are still treated position-dependently — just with far smaller bias than before. Plan and docstring document this explicitly ("deliberately unchanged"). | None now. If MC-mode timeout handling is ever revisited, this is the seam. |
| 🟢 nit | `src/models/recurrent_ppo_trainer.py:299` | In GAE mode `bootstrap_value = trajectories.next_value[-1]` is computed but never consumed by `train_iteration`'s GAE branch — a dead value kept for uniform return signature. XLA dead-code-eliminates it; zero runtime cost. | None. Intentional per plan. |
| 🟢 note | `src/models/ppo_trainer.py:80` | Sibling non-recurrent trainer keeps its own old 3-arg `compute_mc_returns` with the identical window-edge defect. Separate copy, so no import breakage from this diff. Plan explicitly defers it to a `bug-curator` row (no live study uses plain PPO). | Confirm the post-merge `bug-curator` action actually happens. |

## Detailed verification

### 1. Reverse-scan direction and carry semantics — correct

`jax.lax.scan(..., reverse=True)` runs t = T−1 → 0 with the carry flowing backward.
At t = T−1 the carry entering `mc_scan` is `bootstrap_value`; the in-scan gate
`ret = where(done, 0, ret)` then either discards it (edge death) or folds it into
`r[T−1] + γ·V(s′)`. Hand-trace with γ=0.9, r=[1,2,3], V=10 reproduces every expected
array in the tests, including the backward-equivalence anchor (bootstrap=0 ⇒ old
outputs exactly). Mid-window resets are untouched: `dones_for_reset` differs from
`dones` only at index −1, and only when the edge step is *not* a real death.

Carry dtype is stable (`bootstrap_value` arrives float32 per vmap slice; reward is
float32) — no scan carry-structure mismatch.

### 2. `.at[-1].set` edge-gate under vmap — correct

`compute_mc_returns` is vmapped with `in_axes=(1, 1, 1, 0, None)`. Inside the mapped
function each array is logically `(T,)`, so `dones[-1]` and `.at[-1].set(...)` index
the **time** axis; JAX's batching rule for scatter turns the edit into a per-env update
of the last time index of the `(T, N)` array. `edge_death` is a per-env scalar;
`.astype(dones.dtype)` round-trips bool→bool. `jnp.ndarray.at` is functional — the
caller's `trajectories.done` is never mutated (pytree immutability ✅).

### 3. vmap axis alignment of new arguments — correct

`trajectories.reward/done` and the new `terminateds` mask are `(T, N)` → `in_axes=1`;
`bootstrap_value` is `(N,)` → `in_axes=0`; `out_axes=1` restores `(T, N)` returns.
The `terminateds` mask uses the identical `termination_reason >= 2` mapping as the GAE
branch (`recurrent_ppo_trainer.py:361` vs `:374`) — reasons 2/3/4 = real death,
1 = timeout, 0 = active. The overeating quirk (reason 3 without `done=True`, per
KNOWN_BUGS) is guarded by the `done AND terminated` edge gate; the dedicated test
(`test_overeating_quirk_does_not_zero_bootstrap`) pins it.

### 4. Pre-reset carry — really pre-reset, no stale closure

In `scan_fn`, `next_state` is the raw `jax_step` output (`:221`), captured into the
carry return (`:286`) **before** the auto-reset selection produces `final_state`
(`:252-255`); `h_new` is the post-forward hidden **before** `_h_reset_on_done`
(`:258`). So `(boot_state, boot_h)` after the scan is the last step's TRUE
continuation state — exactly the pair the GAE branch uses per-step (`:236-237`).
The two new carry slots are write-only inside the scan (unpacked as
`_prev_next_state, _prev_h_new`, never read); the init dummies
`(last_state, last_h_state)` share structure/shape/dtype with the per-step values, so
the scan carry contract holds. The post-scan forward uses the same in-scope `model`,
`env_params`, and `h_axes` (built from `last_h_state`, structure-identical to
`boot_h`) — no closure capture of stale values. Carries are not stacked, so the extra
EnvState + hidden in the carry is constant memory.

### 5. PRNG — stream-preserving claim confirmed

The diff touches **no** `jax.random` call. Key consumption per step is unchanged:
one split at `:209` (act keys), one split at `:244` (reset keys). The pre-existing
`reset_key` aliasing at `:244` (`reset_key, _ = jax.random.split(key)` splits the same
carried key the next iteration splits again) is untouched, as the plan's scope fence
requires. The new post-scan value forward consumes no randomness: `rngs` appears in
`recurrent_ppo_network.py` only in `__init__` signatures (param init); the forward has
no dropout/batchnorm. Rollout data for a given seed is therefore byte-identical
pre/post fix up to the first parameter update — the plan's claim holds.

### 6. JIT / recompilation — no new hazards

`use_gae_bootstrap` remains a trace-time Python bool derived from the static `config`
jit argument; the new post-scan `if use_gae_bootstrap:` is the same static-branch
pattern as the existing in-scan one. `bootstrap_value` is `(N,)` in both branches
(`next_value[-1]` and `v_boot.squeeze(-1)`), so the return structure is
branch-uniform. `.at[-1]` uses a static index. No new static arguments, no
shape-varying inputs, no traced Python control flow.

### 7. Stop-gradient / detach semantics — no gradient leak

`bootstrap_value` is computed inside `collect_trajectories`, entirely outside any
gradient transform. Targets reach the loss as plain arrays in `PPOBatch`;
`nnx.value_and_grad` (`update_step:320`) differentiates only through
`ppo_loss_fn`'s own re-forward. So no gradient flows from the return targets back
into the value network through the bootstrap — exactly the same (correct) treatment
as `trajectories.value` and the GAE branch's `next_value`, neither of which needs an
explicit `stop_gradient` for the same reason.

### 8. Tests — hand-computations verified, all pass

All seven expected arrays were re-derived by hand (γ=0.9) and match; the suite passes
(`7 passed in 2.54s` under the project conda env). Coverage maps 1:1 onto the plan's
gating table: window cut / edge death / edge timeout / mid-window boundary
(death + timeout) / overeating quirk / zero-bootstrap backward-equivalence anchor.
The pre-fix values quoted in docstrings ([5.23, 4.7, 3.0]) are also correct.

## Conventions audit

| Convention | Status |
|---|---|
| Pytree immutability (no in-place mutation; functional `.at`) | ✅ |
| JIT (static branch structure intact, no new recompile triggers) | ✅ |
| vmap (axis 0 = env for bootstrap, axis 1 = env for (T,N) arrays; `EnvParams` broadcast, not batched) | ✅ |
| PRNG (zero change in consumption order/count; `:244` aliasing untouched per scope fence) | ✅ |
| Sensor / observation-breakdown sync | ✅ n/a (no sensor change; `get_observation` used unchanged) |
| Config protocol | ✅ n/a (no config keys added — matches plan's File Changes) |
| Known-bug recurrence (overeating `termination_reason=3` without `done`) | ✅ explicitly guarded + tested |

## Conclusion

Approve — the H4 fix is JAX-correct, plan-faithful, PRNG-stream-preserving, and well-pinned by tests; only cosmetic nits remain.

Reviewed by: code-reviewer
