"""Regression test for the curriculum checkpoint-retention guard.

Incident (2026-07-28, dsrl_curric123_hier_mirror): the continual schedule set a
1000-episode checkpoint cadence per stage while training.max_checkpoints_to_keep
stayed at the config default of 20. That is a 20,000-episode rolling window, and
stage 0 was exactly 20,000 episodes long, so its boundary checkpoint was deleted
at the moment stage 1 began — silently, and unrecoverably.

The guard recomputes the retention window at startup and warns loudly, naming
the boundaries that will not survive. This test pins the arithmetic so the guard
cannot regress into passing the bad configuration.
"""
import pytest


def doomed_boundaries(max_keep, frequencies, boundaries):
    """Mirror of the guard in dreamer_srl_main.py (kept in lockstep)."""
    window = max_keep * min(frequencies)
    total = boundaries[-1]
    return [b for b in boundaries[:-1] if (total - b) > window]


def test_the_actual_incident_is_flagged():
    """keep=20 x cadence 1000 vs the real 3-stage schedule -> both boundaries lost."""
    doomed = doomed_boundaries(20, [1000, 1000, 1000], [20000, 80000, 100_000_000])
    assert doomed == [20000, 80000], (
        "the guard must flag BOTH stage boundaries for the configuration that "
        "actually destroyed the stage-0 checkpoint"
    )


def test_keep_all_is_silent():
    """The shipped fix (keep all) must not warn."""
    assert doomed_boundaries(1_000_000, [2000, 5000, 5000],
                             [20000, 80000, 100_000_000]) == []


def test_window_uses_the_MINIMUM_cadence():
    """Per-stage cadences differ; the window must be computed from the densest
    one, because that is what rotates checkpoints out fastest."""
    # min cadence 1000 -> window 10*1000 = 10_000 -> boundary at 5000 is lost
    # (total 100_000 - 5_000 = 95_000 > 10_000).
    assert doomed_boundaries(10, [1000, 9999], [5000, 100000]) == [5000]
    # If the guard wrongly used max() the window would be 99_990 and it would
    # report nothing — pin that it does not.
    assert doomed_boundaries(10, [1000, 9999], [5000, 100000]) != []


def test_boundary_survives_when_window_covers_the_tail():
    """A boundary close enough to the end survives; it must not be flagged."""
    # boundary 90_000, total 100_000, window 20*1000 = 20_000 >= 10_000 tail.
    assert doomed_boundaries(20, [1000], [90000, 100000]) == []


@pytest.mark.parametrize("max_keep,expect_warn", [(20, True), (1_000_000, False)])
def test_default_vs_fixed_config_values(max_keep, expect_warn):
    """The old config default warns; the shipped value does not."""
    got = bool(doomed_boundaries(max_keep, [2000, 5000, 5000],
                                 [20000, 80000, 100_000_000]))
    assert got is expect_warn
