"""Regression test for WP-NNX F5 (U5) — imagination discount weights ignore
the start state's true continue.

Plain-language context: every imagined rollout starts from a replayed state.
If that replayed state was a real death (terminal) row, the recipe gives the
whole imagined trajectory zero weight in both actor and critic losses —
sheeprl sets continues[0] = 1 - terminated from the replay batch
(dreamer_v3.py:247-248) and builds the cumulative discount weights from it
(D:260). Our implementation started every rollout's weight at 1 and never
consulted the replay terminals during imagination, so rollouts imagined from
death rows trained the actor/critic on post-death futures at full weight.

The crisp pin: a replay batch whose EVERY row is a real death (termination
reason 2 = starvation) must produce exactly zero actor and critic losses
(row-0 weight 0 -> cumprod -> all weights 0). Pre-fix: both losses nonzero.

See docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/
fix_plan_nnx_parity.md (F5) and 05_dreamer_v3_nnx_conventions.md (U5).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dreamer_nnx_fixtures import build_trainer_and_env, make_replay_batch

import jax
import jax.numpy as jnp


def _train_step_metrics(term_reason_value):
    trainer, params, env_state, _ = build_trainer_and_env(
        max_steps=50, cfg_overrides={'agent.zero_init_reward_critic': False})
    batch = make_replay_batch(trainer, params, env_state, B=3, T=6,
                              key=jax.random.PRNGKey(11),
                              term_reason_value=term_reason_value)
    return trainer.train_step(batch, jax.random.PRNGKey(99))


def test_all_terminal_start_rows_zero_behavior_losses():
    """Every replay row a real death (term_reason=2) -> every imagination
    start row has true continue 0 -> all discount weights 0 -> loss_actor and
    loss_critic are exactly 0. Pre-fix: nonzero (start weight hardcoded 1)."""
    metrics = _train_step_metrics(term_reason_value=2.0)
    assert float(metrics['loss_actor']) == 0.0, (
        f"loss_actor = {float(metrics['loss_actor'])} != 0 on an all-death "
        f"batch — rollouts imagined from terminal rows are being trained on "
        f"at full weight (U5)"
    )
    assert float(metrics['loss_critic']) == 0.0, (
        f"loss_critic = {float(metrics['loss_critic'])} != 0 on an all-death "
        f"batch — rollouts imagined from terminal rows are being trained on "
        f"at full weight (U5)"
    )


def test_all_alive_batch_nonzero_behavior_losses():
    """Positive control (pre- and post-fix): an all-alive batch
    (term_reason=0 -> true continue 1) must give nonzero behavior losses —
    guards against the fix zeroing everything."""
    metrics = _train_step_metrics(term_reason_value=0.0)
    assert float(metrics['loss_actor']) != 0.0
    assert float(metrics['loss_critic']) != 0.0
