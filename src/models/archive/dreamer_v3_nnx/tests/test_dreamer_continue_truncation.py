"""Regression test for the DreamerV3 analog of Finding B, Part 2 (continue-head target).

Plain-language context: an episode in this project ends one of two ways — the agent
dies (real termination, `termination_reason` in {2, 3, 4}: starvation / over-eating /
injury), or the clock runs out while the agent is still alive (truncation / timeout,
`termination_reason == 1`). DreamerV3's world model has a "continue head" that predicts
whether the episode goes on; this prediction is used to bootstrap value during imagined
(model-based) rollouts. Before this fix, the continue-head TARGET was built from the
merged `terminal` flag (real death OR timeout), so the model was trained to believe
"ran out of time" == "the world ended" — corrupting imagined-return value learning
exactly the way the (already-fixed) GAE truncation-bootstrap bug corrupted Recurrent
PPO. This test pins the corrected, death-vs-truncation-asymmetric continue target
directly on `compute_continue_target` — no environment rollout needed.

See docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md.
"""
import jax.numpy as jnp
import pytest

from src.models.dreamer_v3_trainer import compute_continue_target


def test_continue_target_is_one_on_timeout():
    """A TIMEOUT step (termination_reason == 1) must yield continue-target 1.0 —
    the episode was cut off, not ended; the model should predict it would continue."""
    term_reason = jnp.array([1.0])
    target = compute_continue_target(term_reason)
    assert target[0] == pytest.approx(1.0)


def test_continue_target_is_zero_on_real_death():
    """Real-death steps (termination_reason in {2, 3, 4}: starvation / over-eating /
    injury) must yield continue-target 0.0."""
    term_reason = jnp.array([2.0, 3.0, 4.0])
    target = compute_continue_target(term_reason)
    assert jnp.array_equal(target, jnp.array([0.0, 0.0, 0.0]))


def test_continue_target_is_one_while_active():
    """A still-active step (termination_reason == 0) must yield continue-target 1.0."""
    term_reason = jnp.array([0.0])
    target = compute_continue_target(term_reason)
    assert target[0] == pytest.approx(1.0)


def test_continue_target_mixed_batch():
    """Pins the full `termination_reason -> continue_target` mapping used in
    `train_step`'s continue loss: 0/1 (active/timeout) -> 1.0, 2/3/4 (real death) -> 0.0.
    Mirrors `test_termination_reason_to_terminated_mask` in test_gae_truncation.py."""
    term_reason = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    target = compute_continue_target(term_reason)
    expected = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert jnp.array_equal(target, expected)


def test_old_merged_terminal_formula_would_wrongly_zero_on_timeout():
    """Demonstrates the bug this fix removes: the OLD formula `1.0 - terminal` (terminal =
    real-death OR timeout merged flag) zeroed the continue-target on timeout too, unlike
    the corrected `compute_continue_target`, which only zeros on real death."""
    # A batch of one TIMEOUT step: terminal=True (episode-end flag merges timeout), but
    # termination_reason=1 (timeout, NOT real death).
    terminal = jnp.array([1.0])
    term_reason = jnp.array([1.0])

    old_target = 1.0 - terminal  # pre-fix formula
    new_target = compute_continue_target(term_reason)  # post-fix formula

    assert old_target[0] == pytest.approx(0.0)   # OLD (buggy): timeout treated as death
    assert new_target[0] == pytest.approx(1.0)   # NEW (fixed): timeout != death
