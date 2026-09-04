"""Tests for `scripts/analysis/core/env.py`, the consolidated config-derived helpers.

Plain-language context: the agent cannot see its own wound. The only route from an injury to its
behaviour is an interoceptive nociceptor, which does not report the injury directly — it reports a
*smoothed, delayed* version of it, built by convolving the last twelve steps of injury history with
a decaying weight curve (an "alpha kernel"). The analysis has to rebuild that signal from the
recorded injury column, because the store does not save the observation vector itself.

The subtle part, and the reason this file exists, is the **reset boundary**. When an episode ends
the environment zeroes the twelve-slot history buffer, so at the first step of a new episode the
agent feels *nothing*, no matter how badly wounded it woke up. An earlier version of this
reconstruction let the previous episode's injuries — and the new episode's own starting wound —
bleed across that boundary. That is not a rounding error: this project's sensor-ladder study turns
on the claim that the agent's behaviour tracks what it FEELS rather than the wound it HAS, and the
evidence for that is precisely the delay at the start of an episode. A reconstruction that leaks
across the reset manufactures the feeling early and destroys the finding it was built to test.

The bug is recorded in the Known Bugs registry (2026-08-25). These tests pin the boundary so it
cannot come back silently.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts", "analysis", "core"))
import env as E  # noqa: E402


def _episode(lengths):
    """Build (t, estart) for consecutive episodes of the given lengths, as the store lays them out."""
    t = np.concatenate([np.arange(n) for n in lengths])
    starts = np.cumsum([0] + list(lengths))[:-1]
    estart = np.repeat(starts, lengths)
    return t, estart


def test_the_reset_row_feels_nothing():
    """At t=0 the buffer has just been zeroed, so the percept is 0 however large the wound."""
    t, estart = _episode([6])
    inj = np.full(6, 100.0)                       # maximally wounded for the whole episode
    k = np.array([0.0, 0.5, 0.3, 0.2])
    out = E.perceived_nociception(inj, t, estart, k)
    assert out[0] == 0.0


def test_the_second_row_feels_nothing_either():
    """`kernel[0]` is zero by construction, so the wound at row t is not in the percept at row t.

    The agent feels the step BEFORE, weighted by kernel[1] — and at t=1 the only prior row is the
    reset row, which the buffer excludes. So the first two rows are both silent.
    """
    t, estart = _episode([6])
    inj = np.full(6, 100.0)
    k = np.array([0.0, 0.5, 0.3, 0.2])
    out = E.perceived_nociception(inj, t, estart, k)
    assert out[1] == 0.0
    assert out[2] > 0.0                            # by t=2 there is a real prior step to feel


def test_nothing_leaks_across_an_episode_boundary():
    """A brutal episode followed by an unhurt one: the unhurt episode must feel nothing at all."""
    lengths = [5, 5]
    t, estart = _episode(lengths)
    inj = np.concatenate([np.full(5, 100.0), np.zeros(5)])
    k = np.array([0.0, 0.5, 0.3, 0.2])
    out = E.perceived_nociception(inj, t, estart, k)
    assert np.all(out[5:] == 0.0), (
        "the second episode felt the first episode's injuries across the reset — this is the "
        "2026-08-25 bug, and it manufactures the very early-response signal the ladder study tests")


def test_the_percept_lags_and_smooths_a_single_spike():
    """One injured step should be felt afterwards, spread over the kernel, never on the step itself."""
    t, estart = _episode([8])
    inj = np.zeros(8); inj[2] = 100.0
    k = np.array([0.0, 0.5, 0.3, 0.2])
    out = E.perceived_nociception(inj, t, estart, k)
    assert out[2] == 0.0                           # not felt on the step it happened
    assert out[3] == pytest.approx(50.0)           # kernel[1]
    assert out[4] == pytest.approx(30.0)           # kernel[2]
    assert out[5] == pytest.approx(20.0)           # kernel[3]
    assert out[6] == 0.0                           # kernel exhausted


def test_kernel_is_normalised_and_starts_at_zero():
    """The real kernel: weights sum to 1, and slot 0 is zero so the current step is never felt."""
    cfg = {"sensory": {"interoceptive_kernel_length": 12, "interoceptive_kernel_tau": 3.0}}
    k = E.nociception_kernel(cfg)
    assert len(k) == 12
    assert k[0] == 0.0
    assert k.sum() == pytest.approx(1.0)
    assert np.argmax(k) == 3                       # peaks at tau, as an alpha kernel does


def test_slot_layout_returns_the_superset_every_caller_needs():
    """Consolidating three drifted copies is only safe if the merged one is a superset."""
    cfg = {"environment": {
        "entities": [{"class": "predator", "count_high": 2, "properties": [1, 1, 0]},
                     {"class": "neutral", "count_high": 3, "properties": [1, 0, 1]}],
        "obstacles": [{"count_high": 4, "hides_agent": True}, {"count_high": 2}],
        "resources": [{"count_high": 5}, {"count_high": 1, "damage": [0, 9]}]}}
    lay = E.slot_layout(cfg)
    assert lay["pred"] == [0, 1]
    assert lay["neutral"] == [2, 3, 4]
    assert lay["bush"] == [0, 1, 2, 3]
    assert lay["rock"] == [4, 5]
    assert lay["food"] == [0, 1, 2, 3, 4]
    assert lay["ambush"] == [5]
    assert (lay["n_animal"], lay["n_obs"], lay["n_res"]) == (5, 6, 6)


def test_smell_channels_refuses_a_config_that_does_not_separate():
    """Deriving the channels rather than assuming them means an undecidable config must fail loudly."""
    cfg = {"environment": {"entities": [
        {"class": "predator", "count_high": 1, "properties": [1.0, 1.0]},
        {"class": "neutral", "count_high": 1, "properties": [1.0, 1.0]}]}}
    with pytest.raises(SystemExit):
        E.smell_channels(cfg)
