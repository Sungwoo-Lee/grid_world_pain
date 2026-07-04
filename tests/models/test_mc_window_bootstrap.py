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
