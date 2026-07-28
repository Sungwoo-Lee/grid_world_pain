"""Regression/unit tests for src/utils/rolling_logging.py (CP2 of
docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md).

Reproduces the two worked examples in the plan doc (updated 2026-07-28 for
partial-window emission — the original warm-up gate blacked out episode rows
on short curriculum stages and at the start of every run), and checks the
smoothing<=interval warning path of resolve_logging_cfg.
"""
import numpy as np
import pytest

from src.utils.rolling_logging import RollingWindow, spread, resolve_logging_cfg


def test_episode_level_worked_example():
    """smoothing=4, interval=2 over [20,35,50,12,60,18,44].

    Plan doc worked example (episode level), UPDATED 2026-07-28 for
    partial-window emission: the warm-up gate was removed (it blacked out
    short curriculum stages entirely), so an additional PARTIAL point fires
    at ep2 (mean of [20,35] = 27.5). The full-window points at ep4 (29.25)
    and ep6 (35.0) are unchanged.
    """
    stream = [20, 35, 50, 12, 60, 18, 44]
    w = RollingWindow(smoothing=4, interval=2, name="episode")
    emissions = []
    for sample in stream:
        if w.push(sample):
            emissions.append((w.count, float(np.mean(w.buf))))

    assert emissions == [(2, 27.5), (4, 29.25), (6, 35.0)]


def test_step_level_worked_example():
    """smoothing=3, interval=2 over [8,6,5,4,4.5,3.5].

    Plan doc worked example (step level), UPDATED 2026-07-28 for
    partial-window emission: an additional PARTIAL point fires at iter2
    (mean of [8,6] = 7.0); the full-window points at iter4 (5.0) and
    iter6 (4.0) are unchanged.
    """
    stream = [8.0, 6.0, 5.0, 4.0, 4.5, 3.5]
    w = RollingWindow(smoothing=3, interval=2, name="step")
    emissions = []
    for sample in stream:
        if w.push(sample):
            emissions.append((w.count, float(np.mean(w.buf))))

    assert emissions == [(2, 7.0), (4, 5.0), (6, 4.0)]


def test_emission_fires_before_full_window():
    """UPDATED 2026-07-28 (was test_no_emission_before_full_window, asserting
    the opposite): the warm-up gate is gone — emission fires on the interval
    even before `smoothing` samples accumulate. The interval gate itself is
    unchanged."""
    w = RollingWindow(smoothing=4, interval=1, name="episode")
    fired = [w.push(x) for x in [1, 2, 3]]
    assert fired == [True, True, True]
    assert w.push(4) is True  # window now full — still fires
    # Interval gate still respected when interval > 1.
    w2 = RollingWindow(smoothing=4, interval=3, name="episode")
    assert [w2.push(x) for x in [1, 2, 3]] == [False, False, True]


def test_buffer_evicts_not_clears():
    """Buffer B evicts (deque maxlen) rather than clearing after emission —
    the key behavioral change from the legacy clear-after-emit path."""
    w = RollingWindow(smoothing=4, interval=2, name="episode")
    for sample in [20, 35, 50, 12]:
        w.push(sample)
    assert len(w.buf) == 4  # full, not cleared
    w.push(60)
    assert len(w.buf) == 4  # still full — evicted 20, kept the rest
    assert list(w.buf) == [35, 50, 12, 60]


def test_partial_window_emission_on_interval():
    """Bug fix (curriculum audit 2026-07-28): push() must emit on the interval
    even when the window is NOT yet full. The old full-window gate blacked out
    all episode rows for any curriculum stage shorter than `smoothing` episodes
    (the stage swap clears the window), and for the first `smoothing` episodes
    of every run."""
    w = RollingWindow(smoothing=5000, interval=200, name="episode")
    fired_at = [i for i in range(1, 601) if w.push(float(i))]
    assert fired_at == [200, 400, 600], (
        f"Partial-window emission must fire on the interval; got {fired_at}")


def test_partial_window_emission_after_stage_swap_clear():
    """Simulate a curriculum stage swap (buf.clear() + count = 0, the exact
    idiom in dreamer_srl_main.py's stage-transition block): a post-swap stage
    shorter than `smoothing` must still emit rows."""
    w = RollingWindow(smoothing=1000, interval=10, name="episode")
    for i in range(25):
        w.push(float(i))
    # Stage swap: driver clears window + re-arms the counter.
    w.buf.clear()
    w.count = 0
    fired_at = [i for i in range(1, 31) if w.push(float(i))]
    assert fired_at == [10, 20, 30], (
        f"Short post-swap stage must still emit on the interval; got {fired_at}")
    # Post-swap emissions contain ONLY post-swap samples (no tag contamination).
    assert len(w.buf) == 30


def test_window_sample_count_exposed():
    """The driver logs the current sample count (Episode/_window_n) so a
    partial-window point is never mistaken for a full-window one."""
    w = RollingWindow(smoothing=4, interval=2, name="episode")
    assert w.n == 0
    w.push(1.0)
    assert w.n == 1
    for x in [2.0, 3.0, 4.0, 5.0]:
        w.push(x)
    assert w.n == 4  # capped at smoothing (deque maxlen)


def test_full_still_reports_strict_fullness():
    """full() keeps the strict semantics for callers that want them."""
    w = RollingWindow(smoothing=3, interval=1, name="episode")
    w.push(1.0)
    assert not w.full()
    w.push(2.0)
    w.push(3.0)
    assert w.full()


def test_spread_writes_mean_std_min_max():
    out = {}
    spread([1.0, 2.0, 3.0, 4.0], "Episode/Steps", out)
    assert out["Episode/Steps"] == pytest.approx(2.5)
    assert out["Episode/Steps_Std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))
    assert out["Episode/Steps_Min"] == pytest.approx(1.0)
    assert out["Episode/Steps_Max"] == pytest.approx(4.0)


def test_spread_skips_nan_and_empty():
    out = {}
    spread([], "X", out)
    assert out == {}
    out2 = {}
    spread([float("nan"), float("nan")], "X", out2)
    assert out2 == {}


def test_resolve_logging_cfg_absent_returns_none():
    """No `logging.*` key anywhere → LEGACY path (returns None)."""
    def cfg_get(key, default=None):
        return default  # simulates a Config with no `logging:` block
    assert resolve_logging_cfg(cfg_get, defaults={
        'smoothing_episodes': 5000, 'interval_episodes': 200,
        'smoothing_iters': 200, 'interval_iters': 100,
    }) is None


def test_resolve_logging_cfg_present_uses_defaults_for_missing_keys():
    raw = {'logging.episode.smoothing_episodes': 1000}

    def cfg_get(key, default=None):
        return raw.get(key, default)

    out = resolve_logging_cfg(cfg_get, defaults={
        'smoothing_episodes': 5000, 'interval_episodes': 200,
        'smoothing_iters': 200, 'interval_iters': 100,
    })
    assert out == {
        'smoothing_episodes': 1000,   # explicit override
        'interval_episodes': 200,     # falls back to default
        'smoothing_iters': 200,
        'interval_iters': 100,
    }


def test_resolve_logging_cfg_warns_when_smoothing_le_interval(capsys):
    raw = {
        'logging.episode.smoothing_episodes': 50,
        'logging.episode.interval_episodes': 100,   # smoothing < interval -> gaps
    }

    def cfg_get(key, default=None):
        return raw.get(key, default)

    out = resolve_logging_cfg(cfg_get, defaults={
        'smoothing_episodes': 5000, 'interval_episodes': 200,
        'smoothing_iters': 200, 'interval_iters': 100,
    })
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "logging.episode" in captured.out
    # Still runs (returns resolved values, does not raise).
    assert out['smoothing_episodes'] == 50
    assert out['interval_episodes'] == 100


def test_rolling_window_rejects_nonpositive_smoothing_or_interval():
    with pytest.raises(ValueError):
        RollingWindow(smoothing=0, interval=1)
    with pytest.raises(ValueError):
        RollingWindow(smoothing=1, interval=0)
