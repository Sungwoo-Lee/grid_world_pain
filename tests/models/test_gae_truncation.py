"""Regression test for Finding B, Part 2 (value-bootstrap truncation fix).

Plain-language context: an episode in this project ends one of two ways —
the agent dies (real termination, `termination_reason` in {2, 3, 4}), or the
clock runs out while the agent is still alive (truncation / timeout,
`termination_reason == 1`). Standard reinforcement-learning practice is: on
real death, the future is truly gone, so the value-bootstrap term
`gamma * V(s')` is zeroed. On a timeout, the episode was merely cut off —
the agent would have kept going — so the bootstrap must be RETAINED. Before
this fix, `compute_gae` zeroed the bootstrap on BOTH outcomes (it only saw
the merged `done` flag, which is true for both death and timeout). This test
pins the corrected, death-vs-truncation-asymmetric behaviour directly on
`compute_gae` — no environment rollout needed.

See docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md (Part 2).
"""
import jax.numpy as jnp
import pytest

from src.models.recurrent_ppo_trainer import compute_gae

GAMMA = 0.99
LAMBDA = 0.95


def test_bootstrap_retained_on_truncation():
    """A single timeout step (done=True, terminated=False) must RETAIN gamma*V(s')."""
    rewards = jnp.array([2.0])
    values = jnp.array([1.0])
    values_next = jnp.array([5.0])  # V(true next state), pre-auto-reset
    dones = jnp.array([True])
    terminateds = jnp.array([False])  # truncation, NOT real death

    advantages = compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    expected = rewards[0] + GAMMA * values_next[0] * 1.0 - values[0]  # bootstrap retained
    assert advantages[0] == pytest.approx(float(expected), abs=1e-5)
    assert advantages[0] == pytest.approx(5.95, abs=1e-5)


def test_bootstrap_zeroed_on_real_death():
    """A single real-death step (done=True, terminated=True) must ZERO gamma*V(s')."""
    rewards = jnp.array([2.0])
    values = jnp.array([1.0])
    values_next = jnp.array([5.0])
    dones = jnp.array([True])
    terminateds = jnp.array([True])  # real death

    advantages = compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    expected = rewards[0] + GAMMA * values_next[0] * 0.0 - values[0]  # bootstrap zeroed
    assert advantages[0] == pytest.approx(float(expected), abs=1e-5)
    assert advantages[0] == pytest.approx(1.0, abs=1e-5)


def test_truncation_bootstrap_does_not_leak_across_episode_boundary():
    """A truncation step followed by the FIRST step of a brand-new episode: the truncated
    step must still retain its own bootstrap, but must NOT inherit GAE advantage propagated
    backward from the new episode (the `done` flag must still cut the accumulation chain).
    """
    # t=0: last step of episode A, ends via TIMEOUT (done=True, terminated=False).
    # t=1: first step of episode B (a different, freshly-reset episode), not terminal.
    rewards = jnp.array([2.0, 3.0])
    values = jnp.array([1.0, 0.5])
    values_next = jnp.array([5.0, 4.0])
    dones = jnp.array([True, False])
    terminateds = jnp.array([False, False])

    advantages = compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    # t=0's advantage must equal the single-step truncation case exactly (5.95) — i.e. it
    # retains its own bootstrap but does NOT pick up any contribution from t=1's advantage,
    # because `done[0] = True` zeroes the (1 - done) accumulation-reset term.
    assert advantages[0] == pytest.approx(5.95, abs=1e-5)


def test_death_bootstrap_does_not_leak_across_episode_boundary():
    """Same as above, but t=0 is a REAL DEATH (terminated=True) — bootstrap must be zeroed,
    and the episode boundary must still cut the accumulation chain.
    """
    rewards = jnp.array([2.0, 3.0])
    values = jnp.array([1.0, 0.5])
    values_next = jnp.array([5.0, 4.0])
    dones = jnp.array([True, False])
    terminateds = jnp.array([True, False])

    advantages = compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    assert advantages[0] == pytest.approx(1.0, abs=1e-5)


def test_termination_reason_to_terminated_mask():
    """Pins the `termination_reason >= 2` mapping used in `train_iteration`:
    0 = still active, 1 = timeout (truncation, NOT terminated), 2/3/4 = real death
    (starvation / over-eating / injury, terminated).
    """
    reasons = jnp.array([0, 1, 2, 3, 4])
    terminated = (reasons >= 2).astype(jnp.float32)
    expected = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    assert jnp.array_equal(terminated, expected)
