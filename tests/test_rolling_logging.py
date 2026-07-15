"""Regression/unit tests for src/utils/rolling_logging.py (CP2 of
docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md).

Reproduces the two worked examples in the plan doc exactly, including the
warm-up gate (no emission before the window is full), and checks the
smoothing<=interval warning path of resolve_logging_cfg.
"""
import numpy as np
import pytest

from src.utils.rolling_logging import RollingWindow, spread, resolve_logging_cfg


def test_episode_level_worked_example():
    """smoothing=4, interval=2 over [20,35,50,12,60,18,44].

    Plan doc worked example (episode level): emits at ep4 (mean 29.25) and
    ep6 (mean 35.0) — NOT at ep2, because the warm-up gate requires a full
    window before the first emission.
    """
    stream = [20, 35, 50, 12, 60, 18, 44]
    w = RollingWindow(smoothing=4, interval=2, name="episode")
    emissions = []
    for sample in stream:
        if w.push(sample):
            emissions.append((w.count, float(np.mean(w.buf))))

    assert emissions == [(4, 29.25), (6, 35.0)]


def test_step_level_worked_example():
    """smoothing=3, interval=2 over [8,6,5,4,4.5,3.5].

    Plan doc worked example (step level): emits at iter4 (mean 5.0) and
    iter6 (mean 4.0) — not at iter2, for the same warm-up-gate reason.
    """
    stream = [8.0, 6.0, 5.0, 4.0, 4.5, 3.5]
    w = RollingWindow(smoothing=3, interval=2, name="step")
    emissions = []
    for sample in stream:
        if w.push(sample):
            emissions.append((w.count, float(np.mean(w.buf))))

    assert emissions == [(4, 5.0), (6, 4.0)]


def test_no_emission_before_full_window():
    """No emission fires before `smoothing` samples have been pushed, even if
    `interval` would otherwise be satisfied (e.g. interval=1)."""
    w = RollingWindow(smoothing=4, interval=1, name="episode")
    fired = [w.push(x) for x in [1, 2, 3]]
    assert fired == [False, False, False]
    assert w.push(4) is True  # window now full


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
