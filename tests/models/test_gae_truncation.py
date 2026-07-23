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


def test_bootstrap_retained_on_overeating_quirk_mid_episode():
    """Overeating quirk guard: a step with terminated=1 but done=0 must RETAIN the bootstrap.

    The known env quirk (KNOWN_BUGS): `overeating_death=True` stamps
    `termination_reason=3` (so `terminateds`=1) WITHOUT setting `done` — the episode
    actually continues. A continuing episode must keep its `gamma * V(s')` bootstrap;
    only a REAL death (done AND terminated) may zero it. `compute_mc_returns` already
    guards this at the window edge (trainer:114); this pins the same gate in
    `compute_gae`'s delta. Pre-fix, `compute_gae` zeroed the bootstrap on `terminated`
    alone, producing an incoherent mid-episode hybrid (bootstrap zeroed, accumulation
    chain still running).
    """
    # t=0: quirk step — terminated=1 (reason=3 stamped) but done=0 (episode continues).
    # t=1: ordinary non-terminal step.
    rewards = jnp.array([2.0, 3.0])
    values = jnp.array([1.0, 0.5])
    values_next = jnp.array([5.0, 4.0])
    dones = jnp.array([False, False])
    terminateds = jnp.array([True, False])  # the quirk: terminated without done

    advantages = compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    # Since done=0 everywhere, the quirk step must behave exactly like an ordinary
    # mid-episode step: delta_0 = r + gamma*V(s')*1 - V(s), plus the accumulation
    # from t=1 (chain NOT cut — done=0).
    delta_1 = rewards[1] + GAMMA * values_next[1] - values[1]  # 3 + 0.99*4 - 0.5 = 6.46
    delta_0 = rewards[0] + GAMMA * values_next[0] - values[0]  # 2 + 0.99*5 - 1 = 5.95
    expected_0 = delta_0 + GAMMA * LAMBDA * delta_1
    assert advantages[0] == pytest.approx(float(expected_0), abs=1e-5), (
        f"quirk step (terminated=1, done=0) must retain the bootstrap; got {advantages[0]}, "
        f"expected {float(expected_0):.5f} (pre-fix zeroed-bootstrap value would be "
        f"{float(expected_0 - GAMMA * values_next[0]):.5f})"
    )


def test_termination_reason_to_terminated_mask():
    """Pins the `termination_reason >= 2` mapping used in `train_iteration`:
    0 = still active, 1 = timeout (truncation, NOT terminated), 2/3/4 = real death
    (starvation / over-eating / injury, terminated).
    """
    reasons = jnp.array([0, 1, 2, 3, 4])
    terminated = (reasons >= 2).astype(jnp.float32)
    expected = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    assert jnp.array_equal(terminated, expected)


# ---------------------------------------------------------------------------
# Finding B-2 sibling: plain (non-recurrent) PPO trainer's `compute_gae`.
#
# `src.models.ppo_trainer.compute_gae` now takes the same `(rewards, values,
# values_next, dones, terminateds, gamma, lmbda)` signature as the recurrent
# trainer's version above (advantage-baseline fix — see
# test_ppo_trainer_advantage_baseline_uses_v_st below). These tests mirror the
# death-vs-truncation bootstrap cases already pinned for the recurrent trainer.
# ---------------------------------------------------------------------------
from src.models.ppo_trainer import compute_gae as ppo_compute_gae


def test_ppo_trainer_bootstrap_retained_on_truncation():
    """A single timeout step (done=True, terminated=False) must RETAIN gamma*V(s')."""
    rewards = jnp.array([2.0])
    values = jnp.array([1.0])
    values_next = jnp.array([5.0])  # V(true next state), pre-auto-reset
    dones = jnp.array([True])
    terminateds = jnp.array([False])  # truncation, NOT real death

    advantages = ppo_compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    expected = rewards[0] + GAMMA * values_next[0] * 1.0 - values[0]  # bootstrap retained
    assert advantages[0] == pytest.approx(float(expected), abs=1e-5)
    assert advantages[0] == pytest.approx(5.95, abs=1e-5)


def test_ppo_trainer_bootstrap_zeroed_on_real_death():
    """A single real-death step (done=True, terminated=True) must ZERO gamma*V(s')."""
    rewards = jnp.array([2.0])
    values = jnp.array([1.0])
    values_next = jnp.array([5.0])
    dones = jnp.array([True])
    terminateds = jnp.array([True])  # real death

    advantages = ppo_compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    expected = rewards[0] + GAMMA * values_next[0] * 0.0 - values[0]  # bootstrap zeroed
    assert advantages[0] == pytest.approx(float(expected), abs=1e-5)
    assert advantages[0] == pytest.approx(1.0, abs=1e-5)


def test_ppo_trainer_truncation_bootstrap_does_not_leak_across_episode_boundary():
    """A truncation step followed by the first step of a brand-new episode: the truncated
    step must still retain its own bootstrap, but must NOT inherit GAE advantage propagated
    backward from the new episode (the `done` flag must still cut the accumulation chain).
    """
    rewards = jnp.array([2.0, 3.0])
    values = jnp.array([1.0, 0.5])
    values_next = jnp.array([5.0, 4.0])
    dones = jnp.array([True, False])
    terminateds = jnp.array([False, False])

    advantages = ppo_compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    assert advantages[0] == pytest.approx(5.95, abs=1e-5)


def test_ppo_trainer_death_bootstrap_does_not_leak_across_episode_boundary():
    """Same as above, but t=0 is a REAL DEATH (terminated=True) — bootstrap must be zeroed,
    and the episode boundary must still cut the accumulation chain.
    """
    rewards = jnp.array([2.0, 3.0])
    values = jnp.array([1.0, 0.5])
    values_next = jnp.array([5.0, 4.0])
    dones = jnp.array([True, False])
    terminateds = jnp.array([True, False])

    advantages = ppo_compute_gae(rewards, values, values_next, dones, terminateds, GAMMA, LAMBDA)

    assert advantages[0] == pytest.approx(1.0, abs=1e-5)


def test_ppo_trainer_advantage_baseline_uses_v_st():
    """Regression test for the advantage-baseline bug found during the B-2 sibling port
    (commit 926c2c3): `ppo_trainer.compute_gae` used to have no separate `values` (V(s_t))
    argument — it subtracted a value SHIFTED from the neighbouring timestep as the baseline,
    instead of V(s_t), which diverges from the textbook GAE recursion
    `delta_t = r_t + gamma*V(s_{t+1})*(1-terminated) - V(s_t)`.

    Hand-computed 3-step example (mid-rollout, non-terminal steps, real death at t=2):
        rewards      = [1.0, 2.0, 3.0]
        values       = [5.0, 4.0, 3.0]   # V(s_t)
        values_next  = [4.0, 3.0, 0.0]   # V(s_{t+1}), true next-state value
        dones        = [False, False, True]
        terminateds  = [False, False, True]

    Textbook-correct advantages (backward GAE recursion, gamma=lambda=0.9):
        t=2: delta = 3 + 0.9*0*(1-1) - 3 = 0.0;  gae2 = 0.0
        t=1: delta = 2 + 0.9*3*(1-0) - 4 = 0.7;  gae1 = 0.7 + 0.81*1*gae2 = 0.7
        t=0: delta = 1 + 0.9*4*(1-0) - 5 = -0.4; gae0 = -0.4 + 0.81*1*gae1 = 0.167
        -> [0.167, 0.7, 0.0]

    Before this fix, the buggy shifted-value baseline produced [7.3753, 7.13, 3.0] on
    this exact input (confirmed by re-running the pre-fix `compute_gae` against this
    test data) — wildly different in both sign and magnitude from the textbook values.
    """
    rewards = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([5.0, 4.0, 3.0])
    values_next = jnp.array([4.0, 3.0, 0.0])
    dones = jnp.array([False, False, True])
    terminateds = jnp.array([False, False, True])

    gamma = 0.9
    lmbda = 0.9

    advantages = ppo_compute_gae(rewards, values, values_next, dones, terminateds, gamma, lmbda)

    expected = jnp.array([0.167, 0.7, 0.0])
    assert jnp.allclose(advantages, expected, atol=1e-3), (
        f"expected textbook-correct advantages {expected}, got {advantages} "
        "(pre-fix shifted-value baseline would give [7.3753, 7.13, 3.0])"
    )
