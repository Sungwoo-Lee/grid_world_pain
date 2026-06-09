"""Unit tests for SequentialReplayBuffer.reset() — curriculum stage-boundary clear.

Tests:
  1. After reset(), _pos==0 and _full==False.
  2. sample() raises after reset() (buffer logically empty).
  3. New data can be added and sampled after reset().
  4. reset() on an un-initialized buffer (empty dict) is safe.

Must FAIL on pre-change code (no reset() method) and PASS after.

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_buffer_reset.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer


def _make_buffer(buffer_size: int = 100, n_envs: int = 2, obs_dim: int = 8) -> SequentialReplayBuffer:
    return SequentialReplayBuffer(buffer_size=buffer_size, n_envs=n_envs, obs_keys=("obs",))


def _make_step_data(n_envs: int, obs_dim: int, action_dim: int = 4) -> dict:
    """Create one time-step of data with shape [1, n_envs, ...]."""
    return {
        "obs":        np.random.randn(1, n_envs, obs_dim).astype(np.float32),
        "actions":    np.random.randn(1, n_envs, action_dim).astype(np.float32),
        "rewards":    np.random.randn(1, n_envs, 1).astype(np.float32),
        "terminated": np.zeros((1, n_envs, 1), dtype=np.float32),
        "truncated":  np.zeros((1, n_envs, 1), dtype=np.float32),
        "is_first":   np.zeros((1, n_envs, 1), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Test 1: reset() clears pos and full flag
# ---------------------------------------------------------------------------

def test_reset_clears_pos_and_full():
    """After adding rows, reset() sets _pos=0 and _full=False."""
    buf = _make_buffer(buffer_size=50, n_envs=2, obs_dim=8)
    n_envs, obs_dim = 2, 8

    # Add enough data to fill the buffer
    for _ in range(60):
        buf.add(_make_step_data(n_envs, obs_dim))

    assert buf._full, "Buffer should be full after 60 adds into size-50 buffer"
    assert buf._pos != 0 or buf._full, "pos should be non-zero or buffer full"

    buf.reset()

    assert buf._pos == 0, f"Expected _pos=0 after reset, got {buf._pos}"
    assert buf._full is False, f"Expected _full=False after reset, got {buf._full}"


# ---------------------------------------------------------------------------
# Test 2: sample() raises after reset() (empty gate)
# ---------------------------------------------------------------------------

def test_sample_raises_after_reset():
    """sample() must raise ValueError after reset() (buffer logically empty)."""
    buf = _make_buffer(buffer_size=50, n_envs=2, obs_dim=8)
    n_envs, obs_dim = 2, 8

    for _ in range(10):
        buf.add(_make_step_data(n_envs, obs_dim))

    buf.reset()

    with pytest.raises((ValueError, RuntimeError)):
        buf.sample(batch_size=4, sequence_length=2)


# ---------------------------------------------------------------------------
# Test 3: new data can be added and sampled after reset
# ---------------------------------------------------------------------------

def test_add_and_sample_after_reset():
    """After reset(), the buffer accepts new data and samples correctly."""
    buf = _make_buffer(buffer_size=50, n_envs=2, obs_dim=8)
    n_envs, obs_dim, seq_len = 2, 8, 3

    # Pre-fill
    for _ in range(20):
        buf.add(_make_step_data(n_envs, obs_dim))

    buf.reset()

    # Re-fill past seq_len threshold
    for _ in range(seq_len + 5):
        buf.add(_make_step_data(n_envs, obs_dim))

    # Should now be able to sample
    samples = buf.sample(batch_size=2, sequence_length=seq_len)
    assert "obs" in samples, "samples must contain 'obs' key"
    # Shape: [n_samples=1, seq_len, batch_size, obs_dim]
    assert samples["obs"].shape == (1, seq_len, 2, obs_dim), (
        f"Unexpected obs shape: {samples['obs'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: reset() on an empty (never-added-to) buffer is safe
# ---------------------------------------------------------------------------

def test_reset_on_empty_buffer_is_safe():
    """Calling reset() on a fresh buffer (no adds yet) does not raise."""
    buf = _make_buffer(buffer_size=50, n_envs=2, obs_dim=8)
    # buf._buf is empty dict at this point
    buf.reset()  # Must not raise
    assert buf._pos == 0
    assert buf._full is False


# ---------------------------------------------------------------------------
# Test 5: backing arrays are retained after reset (no re-alloc needed)
# ---------------------------------------------------------------------------

def test_backing_arrays_retained_after_reset():
    """After reset(), the internal _buf dict still holds the arrays."""
    buf = _make_buffer(buffer_size=50, n_envs=2, obs_dim=8)
    n_envs, obs_dim = 2, 8

    for _ in range(10):
        buf.add(_make_step_data(n_envs, obs_dim))

    keys_before = set(buf._buf.keys())
    buf.reset()
    keys_after = set(buf._buf.keys())

    assert keys_before == keys_after, (
        "reset() must not remove backing arrays from _buf"
    )
