"""Regression tests for WP-NNX F4 (U2) — DreamerV2-style value learning:
lambda-return bootstrap from the slow (EMA "target") critic, and the recipe's
slow-critic regularizer missing.

Plain-language context: canonical DreamerV3 computes ALL imagination values
(the lambda-return bootstrap and the start value) from the LIVE online critic
(sheeprl dreamer_v3.py:244); the slow EMA critic is used only as a
regularizer — an extra critic-loss term pulling the online critic's
prediction toward the slow critic's (D:307-316). Our implementation did the
opposite (DreamerV2-style): it bootstrapped lambda-returns from the slow
critic and had no regularizer at all — slower value propagation, missing
anti-overfitting term. The fix routes val/v_start through the online critic
and adds the two-hot cross-entropy regularizer `loss_critic_slow_reg`.

See docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/
fix_plan_nnx_parity.md (F4) and 05_dreamer_v3_nnx_conventions.md (U2).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dreamer_nnx_fixtures import build_trainer_and_env

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.models.dreamer_v3_util import to_twohot, from_twohot
from dreamer_nnx_rollout_replica import (
    replica_rollout, replica_discount_weights,
)


def _setup(n_rows=6, seed=3):
    # Random (non-zero-init) heads: with the live zero_init_reward_critic the
    # critics output constants at init and the bootstrap-source shift would be
    # invisible. See test_dreamer_nnx_stop_gradients._setup.
    trainer, params, env_state, _ = build_trainer_and_env(
        max_steps=50, cfg_overrides={'agent.zero_init_reward_critic': False})
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    fresh = trainer.agent.wm.rssm.initial(n_rows)
    start_state = dict(fresh)
    start_state['deter'] = jax.random.normal(k1, fresh['deter'].shape)
    start_state['stoch'] = jax.random.normal(k2, fresh['stoch'].shape)
    rng = jax.random.PRNGKey(17)
    return trainer, start_state, rng, jnp.array(0.0), jnp.array(1.0)


def _run_behavior_metrics(trainer, start_state, rng, low, inv):
    _, (metrics, _) = trainer._behavior_loss(
        trainer.agent.ac.actor, trainer.agent.ac.critic, rng, start_state,
        None, low, inv)
    return metrics


def _corrupt_target_critic(trainer, delta=1e3):
    st = nnx.state(trainer.target_critic, nnx.Param)
    corrupted = jax.tree.map(lambda x: x + delta, st)
    nnx.update(trainer.target_critic, corrupted)


def test_lambda_returns_independent_of_target_critic():
    """T4(a): corrupt every target-critic param by +1e3 and rerun the behavior
    loss with the identical rng — the lambda-returns and baseline metrics
    (mean_return, mean_value) must be BIT-IDENTICAL: post-fix they depend only
    on the online critic. Pre-fix: mean_return shifts (the bootstrap values
    came from the target critic)."""
    trainer, start_state, rng, low, inv = _setup()

    m_clean = _run_behavior_metrics(trainer, start_state, rng, low, inv)
    _corrupt_target_critic(trainer)
    m_corrupt = _run_behavior_metrics(trainer, start_state, rng, low, inv)

    for key in ('mean_return', 'mean_value'):
        assert jnp.array_equal(m_clean[key], m_corrupt[key]), (
            f"{key} moved when the TARGET critic was corrupted — lambda-return "
            f"bootstrap still sourced from the slow critic (U2); "
            f"{float(m_clean[key])} vs {float(m_corrupt[key])}"
        )


def test_slow_critic_regularizer_present_and_correct():
    """T4(b): the critic loss must carry the slow-critic regularizer.
    Pre-fix: the 'loss_critic_slow_reg' metric does not exist (KeyError-red).
    Post-fix: it is >= 0, equals the hand-computed two-hot cross-entropy
    between the online critic's logits and the target critic's (symexp'd)
    predictions on the same rollout feats, and MOVES when the target critic
    moves (while mean_return does not — see test above)."""
    trainer, start_state, rng, low, inv = _setup()
    bins = trainer._paper_canonical_twohot_bins

    metrics = _run_behavior_metrics(trainer, start_state, rng, low, inv)
    assert 'loss_critic_slow_reg' in metrics, (
        "slow-critic regularizer metric missing — the recipe's D:307-316 "
        "regularizer term is absent (U2)"
    )
    reg = metrics['loss_critic_slow_reg']
    assert float(reg) >= 0.0, "two-hot CE regularizer cannot be negative"

    # Hand-compute on the replicated rollout (same PRNG stream).
    ro = replica_rollout(trainer, trainer.agent.ac.actor, rng, start_state,
                         value_net=trainer.agent.ac.critic)
    weights = replica_discount_weights(ro['conts'])
    v_pred_logits = trainer.agent.ac.critic(ro['feats'])
    slow_vals = from_twohot(trainer.target_critic(ro['feats']),
                            paper_canonical_bins=bins)
    slow_twohot = to_twohot(slow_vals, paper_canonical_bins=bins)
    ce = -jnp.sum(slow_twohot * jax.nn.log_softmax(v_pred_logits), axis=-1)
    expected = jnp.mean(ce * weights)

    np.testing.assert_allclose(
        np.asarray(reg), np.asarray(expected), rtol=1e-5, atol=1e-7,
        err_msg=("loss_critic_slow_reg does not equal the hand-computed "
                 "two-hot CE(online logits, slow-critic predictions)"))

    # It must MOVE when the target critic moves (it is the ONLY term that
    # depends on the slow critic post-fix).
    _corrupt_target_critic(trainer)
    metrics2 = _run_behavior_metrics(trainer, start_state, rng, low, inv)
    assert not jnp.array_equal(metrics['loss_critic_slow_reg'],
                               metrics2['loss_critic_slow_reg']), (
        "regularizer did not respond to a corrupted target critic — it is "
        "not actually reading the slow critic"
    )
