---
title: "Fix plan H4 — bootstrap the MC return at the rollout-window edge (rPPO live path)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-05
---

# Fix plan H4 — bootstrap the MC return at the rollout-window edge

> **Status**: PLANNED (approved for implementation — work package WP-B)
> **Opened**: 2026-07-04
> **Related**: [[02_rppo_stack]] (Finding 1 — source of record), [[00_combined_diagnosis]] (§2 row H4), [[FIX_TRUNCATION_TREATED_AS_DEATH]] (the sibling GAE fix `3c60f6f` whose truncation semantics this fix adopts)

---

## Context

Every live recurrent-PPO training run scores its experience with "Monte Carlo" returns
that are computed **per 128-step collection window** — and the computation simply stops at
the end of the window, crediting a step near the window's edge with almost none of its true
future reward. All 13 live rPPO configs use this MC mode, while episodes run up to 500
steps, so roughly the last quarter of every window feeds the critic and the policy
position-dependently distorted learning targets on **every iteration of every live run**.
This was found by the 2026-07-04 Fable 5 re-diagnosis (finding H4; the rPPO area report's
Finding 1). The fix is small: initialise the return computation at the window edge with the
critic's own estimate of the remaining future (a "bootstrap" value), instead of zero — the
same repair that an earlier fix (`3c60f6f`) already gave the *other* (GAE) scoring mode,
which no live config uses. This plan specifies that change, its exact death-vs-timeout
gating, the regression tests that pin it, and a comparability caveat: runs trained after
this fix are **not** directly comparable to runs trained before it.

## Decision record (made by the user/parent, 2026-07-04)

**Fix direction: bootstrap the MC return at the window edge — do NOT migrate live configs
to GAE.** Rationale: a GAE migration changes the return estimator the current studies were
designed around (MC was chosen deliberately — "cleaner per v8: smaller seed variance +
lower critic loss" per the config comments), so switching estimators is an
**experiment-level decision**, not a bug fix. The bootstrap fix removes the window-edge
bias while keeping the MC estimator's semantics. GAE migration remains available as a
future experiment-level choice (route through `experiment-designer` if taken up; the GAE
path is already correct since `3c60f6f`).

## Analysis

### Root cause

`compute_mc_returns` (`src/models/recurrent_ppo_trainer.py:90-106`) runs a reverse scan
over the T=128 window with the carry initialised to `0.0`:

```python
def compute_mc_returns(rewards, dones, gamma):
    def mc_scan(carry, x):
        ret = carry
        reward, done = x
        ret = jnp.where(done, 0.0, ret)   # reset at episode boundary
        ret = reward + gamma * ret
        return ret, ret
    _, returns = jax.lax.scan(mc_scan, 0.0, (rewards, dones), reverse=True)
    return returns
```

The window edge (t = T−1) is an *arbitrary collection boundary*, not an episode event —
yet the carry starts at 0, so a step at window position 0 integrates up to 128 steps of
future reward while a step at position 120 integrates at most 8. The critic trains on
these truncated targets (`targets = returns`, `:315-324`), and advantages
(`returns − V`) inherit the bias. Window position is not in the state, so the critic
cannot learn it away — the bias lands directly in the policy gradient. With predominantly
negative per-step homeostatic rewards, late-window steps systematically look "better"
than identical early-window states.

### Why the collector gives MC no bootstrap value today

`collect_trajectories` computes `Transition.next_value` (V of the TRUE, pre-auto-reset
next state, using the un-reset hidden `h_new`) **only in GAE mode** — a deliberate
static (trace-time) branch that keeps the extra value forward pass out of the MC hot
path (`:198-212`; MC mode fills `next_value` with zeros). The fix must therefore obtain
V(s′) of the window's *last* step without paying a per-step forward pass: we carry the
last step's pre-reset `(next_state, h_new)` through the scan carry (constant memory —
carries are not stacked) and do **one** value forward per window after the scan.
~1/128 extra forwards relative to collection ⇒ expected speed impact ≈ 0.

### Death-vs-timeout gating at the edge (decided semantics)

Consistent with the landed `3c60f6f` truncation semantics (real death zeroes the
bootstrap; truncation retains it):

| Edge-step condition | Carry entering the reverse scan |
|---|---|
| episode continues past the window (`done[-1]=False`) | `V(s′_last)` — bootstrap |
| timeout at the edge (`done[-1]=True`, `terminated[-1]=False`) | `V(s′_last)` — bootstrap retained |
| real death at the edge (`done[-1]=True`, `terminated[-1]=True`) | `0` — future truly gone |

`terminated` uses the same mapping as the GAE branch: `termination_reason >= 2`
(2/3/4 = starvation / over-eating / injury death; 1 = timeout; 0 = active).
The edge gate is `done AND terminated` — the AND guards against the known env quirk
where `overeating_death` sets `termination_reason=3` **without** `done=True`
(Finding 3 of [[02_rppo_stack]]): a continuing episode must keep its bootstrap.

Because the in-scan reset (`ret = where(done, 0, ret)`) fires on merged `done`, a naive
carry-init mask would be cancelled by the edge step's own reset on a timeout. The fix
therefore also replaces the **edge step's** reset gate with `done AND terminated`
(mid-window steps keep the merged-`done` gate — see "deliberately unchanged" below).

### Deliberately unchanged (scope-relevant semantics)

- **Mid-window episode boundaries**: the return still resets to 0 on merged `done`
  mid-window — including mid-window timeouts. MC has no per-step `next_value` available
  (that is the GAE-only field), and MC-treating-timeout-as-end-of-return is the project's
  documented finite-horizon design ([[02_rppo_stack]] Finding 1 sibling note). Only the
  *window edge* — where a bootstrap value is obtainable for one forward pass — changes.
- **Per-iteration z-normalization of MC targets** (`:322`) stays as-is (PyTorch-parity
  design; the "critic target scale is nonstationary" nit is Low-severity, pre-existing,
  and out of scope). The bootstrap V enters the raw return sum before normalization,
  exactly as `advantages = normalized_returns − raw V` already mixes these scales.

### Sibling-path check (asked for explicitly)

- **GAE branch, same file** (`compute_gae`, `:55-88`): **no equivalent defect.** Every
  step's delta already contains its own bootstrap `γ·V(s′)·(1−terminated)`; the zero
  carry at the edge only truncates the λ-averaging chain, which is standard truncated
  GAE with bounded bias. No change.
- **Plain-PPO sibling file** `src/models/ppo_trainer.py:80` (`compute_mc_returns`):
  **identical defect exists** (carry 0, no edge bootstrap). No live study uses plain
  PPO, and its `return_mode` even defaults to GAE (`ppo_trainer.py:207`). **Out of
  scope here** — to be recorded with `bug-curator` as an open low-priority row
  (see Post-merge actions).

### Existing tests that pin return values — will any legitimately change?

Checked (`grep compute_mc_returns|compute_gae|recurrent_ppo_trainer` across `tests/`,
`scripts/`): the **only** test file touching this module is
`tests/models/test_gae_truncation.py`, and it exercises `compute_gae` (both trainers)
only — the GAE path is untouched by this fix, so **all existing golden values stay
valid; no test changes are expected or permitted**. There is no existing test on
`compute_mc_returns` (that gap is exactly what the new regression tests close).
The A1 parity fixtures (`tests/env/test_unified_parity.py`) test env observability,
not returns — unrelated (see Known-red baseline below).

## Implementation Plan

### Design

Three coordinated edits in one file, plus one new test file:

1. `compute_mc_returns` gains `terminateds` and `bootstrap_value` parameters; the
   reverse-scan carry is initialised with `bootstrap_value`, and the edge step's reset
   gate becomes `done[-1] AND terminated[-1]` (mid-window gates unchanged).
2. `collect_trajectories` carries the last step's **pre-reset** `(next_state, h_new)`
   through the scan carry and, in MC mode, does one post-scan value forward on them to
   produce `bootstrap_value` (shape `(num_envs,)`), returned as a new sixth output.
   In GAE mode `bootstrap_value = trajectories.next_value[-1]` (the exact same quantity,
   already computed per-step) keeps the return signature uniform.
3. `train_iteration` unpacks the new output and passes it (plus the `terminateds` mask,
   built with the same `termination_reason >= 2` mapping as the GAE branch) into the
   vmapped `compute_mc_returns`.

No PRNG key is consumed or re-threaded anywhere in this change — the rollout data
stream (observations, actions, rewards) for a given seed is byte-identical pre/post
fix up to the first parameter update; only the learning targets change.

No config keys are added or changed. `train_iteration`'s public signature is unchanged
(`train.py` needs no edit). `collect_trajectories` is called only by `train_iteration`
in the same module (verified by grep — no other caller anywhere in the repo).

### File Changes

#### 1. `src/models/recurrent_ppo_trainer.py:90-106` — `compute_mc_returns`

```python
# BEFORE:
def compute_mc_returns(rewards, dones, gamma):
    """Computes Monte Carlo returns."""
    def mc_scan(carry, x):
        ret = carry
        reward, done = x
        # Reset return at episode boundary
        ret = jnp.where(done, 0.0, ret)
        ret = reward + gamma * ret
        return ret, ret

    _, returns = jax.lax.scan(
        mc_scan,
        0.0,
        (rewards, dones),
        reverse=True
    )
    return returns

# AFTER:
def compute_mc_returns(rewards, dones, terminateds, bootstrap_value, gamma):
    """Computes Monte Carlo returns with a value bootstrap at the rollout-window edge.

    H4 fix (docs/develop/active/issues/diag_fable5_20260704/fix_plan_h4_mc_window_bootstrap.md):
    the reverse-scan carry is initialised with V(s') of the window's LAST step instead of
    0.0, so steps near the window edge keep an estimate of their future return. Edge-step
    gating follows the 3c60f6f truncation semantics:
      - real death at the edge  (done AND terminated)  -> bootstrap zeroed (future truly gone)
      - timeout at the edge     (done, NOT terminated) -> bootstrap retained
      - window cut mid-episode  (NOT done)             -> bootstrap retained
    Mid-window episode boundaries are unchanged: the return still resets on merged `done`
    (finite-horizon MC treatment of mid-window timeouts is deliberate — 02_rppo_stack.md,
    Finding 1 sibling note).

    Args:
        rewards:         (T,) rewards
        dones:           (T,) merged episode-end flags (death OR timeout)
        terminateds:     (T,) real-death flags (termination_reason >= 2)
        bootstrap_value: ()  V(s') of the TRUE (pre-auto-reset) next state after step T-1
        gamma:           discount factor
    """
    # Edge gate: zero the carry only on REAL death at the window edge. The AND with done
    # guards against the known env quirk where overeating sets termination_reason=3
    # without done=True (KNOWN_BUGS) — a continuing episode must keep its bootstrap.
    edge_death = jnp.logical_and(dones[-1].astype(bool), terminateds[-1].astype(bool))
    dones_for_reset = dones.at[-1].set(edge_death.astype(dones.dtype))

    def mc_scan(carry, x):
        ret = carry
        reward, done = x
        # Reset return at episode boundary
        ret = jnp.where(done, 0.0, ret)
        ret = reward + gamma * ret
        return ret, ret

    _, returns = jax.lax.scan(
        mc_scan,
        bootstrap_value,
        (rewards, dones_for_reset),
        reverse=True
    )
    return returns
```

Note: `bootstrap_value` arrives as a float32 scalar (per vmap slice), so the scan carry
dtype is float32 as before (the old `0.0` was weakly-typed). `dones` is bool from
`jax_step` (`src/environment/core.py:118-126, 716`); the `.astype` round-trip keeps the
code robust if a float mask is ever passed (the GAE branch already casts `terminateds`
to `done.dtype`).

#### 2. `src/models/recurrent_ppo_trainer.py:163-264` — `collect_trajectories`

Three small edits (the auto-reset block at `:214-227` — including the `reset_key` line
`:216` — is **not touched**; see Scope fence):

**(a) `:175` — scan carry unpack** (two new write-only slots holding the *previous*
step's pre-reset next-state/hidden; only the final carry is consumed):

```python
# BEFORE:
    def scan_fn(carry, _):
        state, h_state, key = carry

# AFTER:
    def scan_fn(carry, _):
        state, h_state, key, _prev_next_state, _prev_h_new = carry
```

**(b) `:258` — scan carry return** (surface this step's PRE-reset `next_state` and
`h_new`; `next_state` at this point is the raw `jax_step` output from `:196`, before the
`final_state` reset-selection at `:224-227`; `h_new` is the pre-reset hidden from `:190-192`):

```python
# BEFORE:
        return (final_state, final_h, key), (trans, h_state)

# AFTER:
        return (final_state, final_h, key, next_state, h_new), (trans, h_state)
```

**(c) `:260-264` — scan init + post-scan bootstrap + return** (init the two new slots
with same-structure dummies; compute the MC edge bootstrap once per window):

```python
# BEFORE:
    (final_state, final_h, final_key), (trajectories, h_states) = jax.lax.scan(
        scan_fn, (last_state, last_h_state, last_key), None, length=num_steps
    )

    return trajectories, h_states, final_state, final_h, final_key

# AFTER:
    (final_state, final_h, final_key, boot_state, boot_h), (trajectories, h_states) = jax.lax.scan(
        scan_fn, (last_state, last_h_state, last_key, last_state, last_h_state), None,
        length=num_steps
    )

    # H4 fix: window-edge bootstrap value = V(TRUE next state) of the LAST window step,
    # from the pre-auto-reset (boot_state, boot_h) carried out of the scan. One value
    # forward per 128-step window — negligible vs the per-step collection forward.
    # In GAE mode the exact per-step quantity already exists; expose its edge slice so
    # the return signature is uniform (train_iteration's GAE branch does not consume it).
    if use_gae_bootstrap:
        bootstrap_value = trajectories.next_value[-1]
    else:
        with jax.named_scope("rppo_mc_edge_bootstrap"):
            obs_boot = jax.vmap(get_observation, in_axes=(0, None))(boot_state, env_params)
            _, v_boot, _, _ = jax.vmap(model, in_axes=(0, h_axes))(obs_boot, boot_h)
            bootstrap_value = v_boot.squeeze(-1)

    return trajectories, h_states, final_state, final_h, final_key, bootstrap_value
```

Also update the now-stale sentence in the comment block at `:202-205` — it currently
says "`compute_gae` is the only consumer of `next_value` (MC mode uses
`compute_mc_returns`, which never reads it)". That remains true for the *per-step*
`next_value`, but append one line noting MC mode now takes a **single** window-edge
bootstrap computed post-scan (H4 fix) instead of per-step values.

#### 3. `src/models/recurrent_ppo_trainer.py:307-324` — `train_iteration`

```python
# BEFORE:
        trajectories, h_states, next_env_state, next_h_state, key = collect_trajectories(
            model, env_params, env_state, h_state, key, config.num_steps, rnn_type=rnn_type, return_mode=return_mode
        )

    # 2. Compute Advantages and Targets
    with jax.named_scope("rppo_advantages"):
        if return_mode.upper() == "MC":
            # Monte Carlo Returns (PyTorch parity)
            # Use vmap over batch dimension (axis 1)
            returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, None), out_axes=1)(
                trajectories.reward, trajectories.done, config.gamma
            )

# AFTER:
        trajectories, h_states, next_env_state, next_h_state, key, bootstrap_value = collect_trajectories(
            model, env_params, env_state, h_state, key, config.num_steps, rnn_type=rnn_type, return_mode=return_mode
        )

    # 2. Compute Advantages and Targets
    with jax.named_scope("rppo_advantages"):
        if return_mode.upper() == "MC":
            # Monte Carlo Returns (PyTorch parity) with window-edge bootstrap (H4 fix).
            # Real-death mask: same termination_reason >= 2 mapping as the GAE branch below.
            # Use vmap over batch dimension (axis 1); bootstrap_value is (num_envs,) -> axis 0.
            terminateds = (trajectories.step_info.termination_reason >= 2).astype(trajectories.done.dtype)
            returns = jax.vmap(compute_mc_returns, in_axes=(1, 1, 1, 0, None), out_axes=1)(
                trajectories.reward, trajectories.done, terminateds, bootstrap_value, config.gamma
            )
```

The lines after this (`returns` z-normalization, `targets = returns`,
`advantages = returns - trajectories.value`) are unchanged. The GAE `else` branch
(`:325-340`) is unchanged.

#### 4. `tests/models/test_mc_window_bootstrap.py` — NEW regression test file

Direct unit tests on `compute_mc_returns` (no env rollout), mirroring the style of
`tests/models/test_gae_truncation.py`. Full content to implement (γ = 0.9 throughout;
hand-computed expectations shown):

```python
"""Regression tests for the H4 fix: MC returns must bootstrap at the rollout-window edge.

Plain-language context: rPPO collects experience in fixed 128-step windows while
episodes run up to 500 steps. Before this fix, the MC return computation started its
reverse scan from 0.0 at the window edge, so a step near the edge was credited with
almost none of its true future return — a position-dependent bias in every live rPPO
run's value targets and advantages. The fix initialises the scan carry with the
critic's V(s') of the window's last step, gated by the 3c60f6f death-vs-truncation
semantics (real death -> 0; timeout or mid-episode cut -> bootstrap retained).

See docs/develop/active/issues/diag_fable5_20260704/fix_plan_h4_mc_window_bootstrap.md.
"""
import jax.numpy as jnp
import pytest

from src.models.recurrent_ppo_trainer import compute_mc_returns

GAMMA = 0.9
REWARDS = jnp.array([1.0, 2.0, 3.0])
NO_DONES = jnp.array([False, False, False])
NO_TERMS = jnp.array([False, False, False])
V_BOOT = jnp.float32(10.0)


def test_window_edge_step_includes_bootstrap():
    """Window cut mid-episode (no done anywhere): every step's return must include the
    discounted bootstrap. PRE-FIX this fails: the old carry=0.0 gave [5.23, 4.7, 3.0]."""
    returns = compute_mc_returns(REWARDS, NO_DONES, NO_TERMS, V_BOOT, GAMMA)
    # t=2: 3 + 0.9*10 = 12; t=1: 2 + 0.9*12 = 12.8; t=0: 1 + 0.9*12.8 = 12.52
    assert jnp.allclose(returns, jnp.array([12.52, 12.8, 12.0]), atol=1e-5)


def test_no_bootstrap_on_real_death_at_window_edge():
    """Real death exactly at the edge (done & terminated): future truly gone -> carry 0."""
    dones = jnp.array([False, False, True])
    terms = jnp.array([False, False, True])
    returns = compute_mc_returns(REWARDS, dones, terms, V_BOOT, GAMMA)
    # t=2: 3; t=1: 2 + 0.9*3 = 4.7; t=0: 1 + 0.9*4.7 = 5.23  (identical to pre-fix)
    assert jnp.allclose(returns, jnp.array([5.23, 4.7, 3.0]), atol=1e-5)


def test_bootstrap_retained_on_timeout_at_window_edge():
    """Timeout at the edge (done, NOT terminated): 3c60f6f truncation semantics ->
    bootstrap retained."""
    dones = jnp.array([False, False, True])
    returns = compute_mc_returns(REWARDS, dones, NO_TERMS, V_BOOT, GAMMA)
    assert jnp.allclose(returns, jnp.array([12.52, 12.8, 12.0]), atol=1e-5)


def test_bootstrap_does_not_leak_across_mid_window_episode_boundary():
    """Episode ends mid-window (t=1): the bootstrap belongs to the NEW episode's steps
    (t=2) only; episode A's steps (t=0, t=1) must not see it."""
    dones = jnp.array([False, True, False])
    terms = jnp.array([False, True, False])
    returns = compute_mc_returns(REWARDS, dones, terms, V_BOOT, GAMMA)
    # t=2: 3 + 0.9*10 = 12; t=1: reset -> 2; t=0: 1 + 0.9*2 = 2.8
    assert jnp.allclose(returns, jnp.array([2.8, 2.0, 12.0]), atol=1e-5)


def test_mid_window_timeout_semantics_unchanged():
    """A mid-window TIMEOUT still ends the return (finite-horizon MC design is
    deliberately unchanged mid-window — only the window EDGE gained a bootstrap)."""
    dones = jnp.array([False, True, False])
    returns = compute_mc_returns(REWARDS, dones, NO_TERMS, V_BOOT, GAMMA)
    assert jnp.allclose(returns, jnp.array([2.8, 2.0, 12.0]), atol=1e-5)


def test_overeating_quirk_does_not_zero_bootstrap():
    """Known env quirk: overeating can set termination_reason=3 WITHOUT done=True.
    The edge gate is (done AND terminated), so a continuing episode keeps its bootstrap."""
    terms = jnp.array([False, False, True])  # terminated flag without done
    returns = compute_mc_returns(REWARDS, NO_DONES, terms, V_BOOT, GAMMA)
    assert jnp.allclose(returns, jnp.array([12.52, 12.8, 12.0]), atol=1e-5)


def test_zero_bootstrap_reproduces_old_behaviour():
    """With bootstrap_value=0 the new code must reproduce the pre-fix outputs exactly
    (backward-equivalence anchor for the scan restructure)."""
    dones = jnp.array([False, True, False])
    returns = compute_mc_returns(REWARDS, dones, NO_TERMS, jnp.float32(0.0), GAMMA)
    # old behaviour on these inputs: t=2: 3; t=1: reset -> 2; t=0: 1 + 0.9*2 = 2.8
    assert jnp.allclose(returns, jnp.array([2.8, 2.0, 3.0]), atol=1e-5)
```

**Fail-pre-fix requirement**: before applying the `src/` change, the developer must run
the old `compute_mc_returns` on the inputs of `test_window_edge_step_includes_bootstrap`
(old 3-arg signature: `compute_mc_returns(REWARDS, NO_DONES, GAMMA)`) and record the
biased output `[5.23, 4.7, 3.0]` in the Implementation Report — this is the pre-fix
evidence that the test's expected `[12.52, 12.8, 12.0]` fails on current code (over and
above the trivial signature `TypeError`).

**How to run**:
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
  tests/models/test_mc_window_bootstrap.py tests/models/test_gae_truncation.py -v
```

### Files NOT changed (explicit)

- `train.py` — `train_iteration`'s signature is unchanged; the new return flows only inside the module.
- `src/models/ppo_trainer.py` — identical MC defect exists there, deliberately out of scope (not live; `bug-curator` row instead).
- `configs/**` — no keys added/changed (no CONFIG_GUIDE update needed).
- `scripts/**` — untouched (no SCRIPTS_DEPENDENCY_MAP update needed).
- `docs/develop/active/issues/KNOWN_BUGS.md` — owned by `bug-curator`; see Post-merge actions.

## Scope fence (binding on the developer)

**ONLY the MC window-edge bootstrap changes.** Specifically:

- The **PRNG reset-key aliasing at `recurrent_ppo_trainer.py:216`** (`reset_key, _ =
  jax.random.split(key)` — Finding 2 of [[02_rppo_stack]], triple-confirmed by the
  re-diagnosis) is **OUT OF SCOPE**. It is stream-breaking (changes every subsequent
  random draw for a given seed) and is scheduled separately together with the A1
  parity-fixture regeneration. Do not touch it, even though it sits inside the function
  being edited. This fix consumes **no** new PRNG keys, precisely so it stays
  stream-preserving.
- The plain-PPO sibling defect (`ppo_trainer.py:80`) is out of scope.
- The MC z-normalization nit and the GAE branch are out of scope.
- Nothing else in `recurrent_ppo_trainer.py` changes beyond the three regions in File
  Changes (plus the one stale-comment touch-up at `:202-205`).

## Comparability caveat (say it plainly)

Every historical rPPO run — all 13 live configs, every study to date — was trained with
window-truncated MC returns. They all share the same bias, so **historical run-to-run
comparisons remain internally valid**. But a run trained **after** this fix learns from a
different (less biased) signal: **post-fix runs are NOT directly comparable to pre-fix
runs.** Any study that mixes pre- and post-fix runs in one comparison must re-run its
baselines under the fixed trainer. Same-seed reruns will produce identical rollout data
only until the first parameter update, then diverge. Practical rule: note the fix commit
hash in any analysis; treat it as a hard "before/after" partition line for rPPO results.
`experiment-designer` and `experiment-analyzer` should be pointed at this section when
the fix lands (the diagnosis folder's combined report already links here).

## Known-red baseline note (for the developer)

`tests/env/test_unified_parity.py` `observability_gates_S1-S4` are currently **RED for an
unrelated known reason (A1** — parity-fixture staleness, scheduled with the Finding-2 PRNG
work**)**. This is **not** your breakage — do not attempt to fix it, and do not count it
against this change in the test summary. Everything else in the suite is expected green;
`tests/models/test_gae_truncation.py` in particular must stay green untouched.

## Checkpoints

What the implementing agent should verify **during** implementation:

- [x] Pre-fix evidence recorded: old `compute_mc_returns(REWARDS, NO_DONES, GAMMA)` →
      `[5.2299995, 4.7, 3.0]` (run before editing; see Implementation Report).
- [x] New tests all green post-fix (7/7); `test_gae_truncation.py` still green (10/10).
- [x] **Cross-mode consistency check**: MC `bootstrap_value` bit-identical (uint32 view)
      to GAE `trajectories.next_value[-1]` on identical inputs/key — PASSED
      (`tmp/20260704_h4_crossmode_check.py`; output in Implementation Report).
- [x] Smoke train: 5 jitted `train_iteration` calls on the base rPPO config (MC mode,
      32 envs) — all losses finite, no NaN, no shape errors; plus two full `train.py`
      runs (20/120 iterations, 128 envs) completed cleanly.
- [x] Speed check: 120-iteration budget wall time 120.21 s (pre-fix) vs 120.08 s
      (post-fix) — Δ −0.1 % ≈ 0, as expected. Details in Implementation Report.
- [x] Full suite run: 388 passed / 494 skipped / 8 failed — 4 are the A1 known-red
      parity gates; the other 4 are pre-existing failures unrelated to this change
      (3 × FileNotFoundError on configs retired by `b093023`, 1 × dreamer-SRL offline
      WM smoke data issue; none import the changed module).
- [x] `git diff` footprint of THIS work package: exactly `src/models/recurrent_ppo_trainer.py`
      (6 hunks, all inside the three planned regions; line 216 `reset_key` untouched) +
      new `tests/models/test_mc_window_bootstrap.py`. (Note: the working tree also
      carries a parallel work package's uncommitted changes — WP H1–H3 resume-config —
      in `train.py`, `src/utils/config.py`, etc.; not part of this change.)

## Post-merge actions (not the developer's)

1. `senior-developer` runs the Verification Protocol on this plan.
2. `bug-curator` records: (a) H4 as fixed with the commit hash; (b) the clarification
   that `3c60f6f`'s registry row covers the GAE branch only (no live config used it —
   the wording currently overstates live coverage, per [[02_rppo_stack]]); (c) a new
   open low-priority row for the identical MC defect in `src/models/ppo_trainer.py:80`.
3. Diary `implemented` / `verified` rows per the diary protocol.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-04

### Summary of changes (file-by-file)

- **`tests/models/test_mc_window_bootstrap.py` (NEW)** — written FIRST, exactly per the
  plan's §File Changes 4: 7 direct unit tests on `compute_mc_returns` (edge bootstrap
  inclusion, real-death zeroing, edge-timeout retention, mid-window no-leak, mid-window
  timeout unchanged, overeating-quirk gate, zero-bootstrap backward equivalence).
- **`src/models/recurrent_ppo_trainer.py`** — the three planned edits, verbatim from the
  plan:
  1. `compute_mc_returns(rewards, dones, terminateds, bootstrap_value, gamma)` — carry
     initialised with `bootstrap_value`; edge reset gate swapped to
     `done[-1] AND terminated[-1]` via `dones.at[-1].set(edge_death)`; mid-window gates
     unchanged.
  2. `collect_trajectories` — scan carry extended with two write-only slots holding the
     previous step's pre-reset `(next_state, h_new)`; post-scan MC branch does ONE value
     forward per window (`rppo_mc_edge_bootstrap` scope) on `(boot_state, boot_h)`;
     GAE branch exposes `trajectories.next_value[-1]`; new sixth return
     `bootstrap_value`. Stale comment at the old `:202-205` updated per plan.
  3. `train_iteration` — unpacks `bootstrap_value`, builds `terminateds` with the same
     `termination_reason >= 2` mapping as the GAE branch, passes both into the vmapped
     `compute_mc_returns` (`in_axes=(1, 1, 1, 0, None)`).
- Scope fence respected: `reset_key` line untouched (verified in the diff — the only
  hunk near it is the comment-block update); `ppo_trainer.py`, GAE branch,
  z-normalization, `configs/`, `train.py` untouched by this work package. No PRNG key
  consumed anywhere in the change.

### Pre-fix evidence (regression tests written and run FIRST)

Old 3-arg `compute_mc_returns(REWARDS, NO_DONES, 0.9)` on the edge-test inputs:

```
old compute_mc_returns(REWARDS, NO_DONES, 0.9) = [5.2299995 4.7       3.       ]
```

i.e. the biased window-truncated returns, not the expected `[12.52, 12.8, 12.0]`.
Running the new test file against pre-fix code: **7/7 FAILED**
(`TypeError: compute_mc_returns() takes 3 positional arguments but 5 were given` — over
and above the substantive value evidence recorded above, per the plan's fail-pre-fix
requirement). Log: `tmp/20260704_h4_prefix_test_run.log`.

### Post-fix test results

```
tests/models/test_mc_window_bootstrap.py  — 7 passed
tests/models/test_gae_truncation.py       — 10 passed  (untouched, still green)
```

### Cross-mode bit-identity check (checkpoint 3)

`tmp/20260704_h4_crossmode_check.py` — identical env state, model, and rollout key;
`collect_trajectories` run once per mode (4 envs, 16 steps; rollout data asserted equal):

```
MC  bootstrap_value          : [-0.07487796 -0.13693738 -0.07665118  0.05316849]
GAE trajectories.next_value[-1]: [-0.07487796 -0.13693738 -0.07665118  0.05316849]
bit-identical (uint32 view)  : True
CROSS-MODE CHECK PASSED
```

### Smoke train

`tmp/20260704_h4_smoke_train.py` — base rPPO config (MC mode, GRU), 32 envs, 5 jitted
`train_iteration` iterations: all losses finite (e.g. iter 4:
`total=0.20307 policy=0.04852 value=0.34331 ent=-1.71003`), no NaN, no shape errors.
Two full `train.py` runs (128 envs, 20 and 120 iterations) also completed cleanly.

### Speed check

Command (both sides; local RTX 4090, `CUDA_VISIBLE_DEVICES=0`, seed 42, 128 envs):

```
python train.py --config configs/environment/default.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 128 --seed 42 --episodes 0 --total-timesteps {327680 | 1966080} \
  --no-wandb --quiet
```

| Budget | Pre-fix wall | Post-fix wall |
|---|---|---|
| 20 iterations (327,680 steps) | 95.01 s | 98.73 s |
| 120 iterations (1,966,080 steps) | 120.21 s | 120.08 s |
| steady-state s/it (Δ of the two budgets / 100) | 0.252 s/it | 0.213 s/it |

Like-for-like 120-iteration total: **−0.1 % (≈ 0, within noise)** — as predicted for one
extra value forward per 128-step window. The difference-method s/it estimates carry a few
seconds of startup/JIT variance between runs; the matched 120-it totals are the cleaner
comparison. **No regression.**

### Full suite

`python -m pytest tests/ -q` → **388 passed, 494 skipped, 8 failed** (29 min 57 s;
log: `tmp/20260704_h4_full_suite.log`). The 8 failures:

- 4 × `tests/env/test_unified_parity.py` `observability_gates_S1–S4` — the **A1 known-red
  baseline** named in this plan (parity-fixture staleness; not this change).
- 3 × `FileNotFoundError` on retired configs (`tests/env/test_inactive_animal_offgrid.py`,
  `tests/env/test_truncation_not_death.py` ×2) — pre-existing breakage from the basic-ladder
  re-leveling commit `b093023` (the configs they load, e.g.
  `configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml`, were retired);
  these tests do not import the changed module.
- 1 × `tests/scripts/test_dreamer_srl_offline_wm_test.py::test_offline_wm_smoke` —
  dreamer-SRL data/starting-state issue, unrelated (does not import the changed module).

Everything that was green stays green.

### Deviations from the plan

**None.** All edits are verbatim from the plan's File Changes section.

One environmental note (not a deviation): the working tree also contains a **parallel,
uncommitted work package** (WP H1–H3 resume-config: `train.py`, `src/utils/config.py`,
`src/utils/checkpoint_restore.py`, related tests/docs). This change's own footprint is
exactly the two planned files; the verifier should diff them independently.

### Follow-ups (for `senior-developer` / post-merge)

- `bug-curator` rows per the plan's Post-merge actions (H4 fixed; `3c60f6f` wording
  clarification; new open row for `ppo_trainer.py:80`).
- The 3 config-retirement test failures (`b093023` fallout) and the dreamer-SRL smoke
  failure are pre-existing and unowned — worth routing to `bug-curator` as separate rows.

Implemented by: developer

## Verification Report

> **Verified by**: senior-developer (independent — did not implement)
> **Date**: 2026-07-05

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/recurrent_ppo_trainer.py` | `compute_mc_returns` — new `(rewards, dones, terminateds, bootstrap_value, gamma)` signature; carry init = `bootstrap_value`; edge reset gate = `done[-1] AND terminated[-1]` via `dones.at[-1].set(edge_death)` | ✅ | Verbatim to plan §File Changes 1. Mid-window gates untouched (only index −1 altered). Docstring matches plan text. |
| `src/models/recurrent_ppo_trainer.py` | `collect_trajectories` — carry extended with pre-reset `(next_state, h_new)`; one post-scan value forward in MC mode (`rppo_mc_edge_bootstrap`); GAE mode exposes `next_value[-1]`; new sixth return | ✅ | Verbatim to plan §File Changes 2 incl. the stale-comment update. Carry surfaces the RAW `jax_step` output (line 286), i.e. pre-`final_state` reset selection — bootstrap is V of the TRUE next state. `h_axes` (line 194, function scope) valid at the post-scan site. |
| `src/models/recurrent_ppo_trainer.py` | `train_iteration` — unpacks `bootstrap_value`; `terminateds = termination_reason >= 2` (same mapping as GAE branch); vmap `in_axes=(1, 1, 1, 0, None)` | ✅ | Verbatim to plan §File Changes 3. z-normalization, `targets`, `advantages`, and the GAE `else` branch unchanged. |
| `src/models/recurrent_ppo_trainer.py` | Scope fence: `reset_key` aliasing + everything else | ✅ | `reset_key, _ = jax.random.split(key)` (now line 244, formerly 216) UNTOUCHED — no diff hunk modifies it. `compute_gae` has zero hunks. PRNG call-site set identical pre/post (lines 209/210/244/246 only); the new bootstrap forward takes no key, mirroring the GAE per-step forward at line 237. Diff stat 62+/18− — proportionate to the docstring-heavy edit; no out-of-plan hunks in this file. |
| `tests/models/test_mc_window_bootstrap.py` (NEW) | 7 regression tests on `compute_mc_returns` | ✅ | Matches plan §File Changes 4 exactly. **Expected values independently recomputed** with a standalone pure-Python script (no project code): all 8 vectors confirmed, incl. pre-fix `[5.23, 4.7, 3.0]` vs post-fix `[12.52, 12.8, 12.0]` for γ=0.9 — the edge test genuinely encodes the bug and would fail on pre-fix code (substantively, over and above the signature `TypeError`). |

**Semantics check (all confirmed against the diff + hand computation):**
- Real death at edge (`done ∧ terminated`) → carry 0 (returns identical to pre-fix `[5.23, 4.7, 3.0]`).
- Timeout at edge (`done`, ¬`terminated`) → bootstrap retained (`3c60f6f` truncation semantics).
- Window cut mid-episode (¬`done`) → bootstrap retained.
- Overeating quirk (`terminated` without `done`) → gate's AND keeps the bootstrap.
- Mid-window boundaries (incl. mid-window timeouts) reset on merged `done`, unchanged; bootstrap does not leak across a mid-window boundary (`[2.8, 2.0, 12.0]` confirmed).
- Zero-bootstrap backward-equivalence anchor confirmed.

**Test re-run (verifier's own run, 2026-07-05):**
`tests/models/test_mc_window_bootstrap.py` 7/7 PASSED + `tests/models/test_gae_truncation.py` 10/10 PASSED (17 passed in 3.9 s). GAE golden values untouched, as the plan requires.

**Known-red baseline:** the 4 `test_unified_parity.py` observability-gate failures (A1 parity-fixture staleness), 3 config-retirement `FileNotFoundError`s (`b093023` fallout), and 1 dreamer-SRL offline-WM smoke failure are pre-existing and unrelated — not counted against this package, per plan. Developer's full-suite result (388 passed / 8 failed, all 8 accounted for) accepted.

**Speed verdict: ✅ no regression.** Matched 120-iteration budgets, same hardware/config/seed: 120.21 s pre vs 120.08 s post (−0.1 %, within noise) — consistent with the predicted ~1/128 extra forward. The 20-iteration pair carries JIT/startup variance; the long matched budget is the valid comparison, and the developer used it correctly.

**Out-of-scope note:** the working tree also carries the parallel WP H1–H3 package (`train.py`, `src/utils/config.py`, `src/utils/checkpoint_restore.py`, related tests/docs) — explicitly excluded from this verification per the work-package split; not contamination.

**Conclusion**: PASS — implementation is verbatim to plan, the scope fence held (reset_key and GAE branch untouched, no new PRNG consumption), the regression tests genuinely pin the bug (expectations independently recomputed), targeted tests re-run green by the verifier, and speed is unchanged. Ready to commit; post-merge `bug-curator` actions (H4 fixed row, `3c60f6f` wording clarification, `ppo_trainer.py:80` open row) remain outstanding per plan.

Verified by: senior-developer
