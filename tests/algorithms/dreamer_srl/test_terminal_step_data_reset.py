"""Regression test for H5 — episode-start buffer rows must NOT inherit the
previous episode's terminal reward / death flag.

At a done boundary sheeprl zeroes the staged step_data rewards/terminated/
truncated for done envs ("Reset already inserted step data",
sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L652-L656). The JAX
port previously performed only the is_first=1 set and dropped the three
zeroing lines, so every episode's first stored row carried the previous
episode's terminal reward and death flag (H5).

Tests:
  1. test_helper_zeroes_done_env_terminal_fields — helper semantics: done-env
     columns zeroed + is_first=1; non-done env columns untouched.
  2. test_next_buffer_row_is_clean — real two-buffer.add sequence through a
     real SequentialReplayBuffer; the persisted is_first==1 row has
     rewards == terminated == truncated == 0.

Must FAIL on pre-fix code and PASS after.

Fix plan: docs/develop/active/issues/diag_fable5_20260704/
fix_plan_h5_dreamer_srl_buffer_reset.md

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer
from src.algorithms.dreamer_srl.dreamer_srl_main import _reset_terminal_step_data


# ---------------------------------------------------------------------------
# Test 1: helper semantics — done env zeroed + is_first=1, live env untouched
# ---------------------------------------------------------------------------

def test_helper_zeroes_done_env_terminal_fields():
    """Done-env columns are zeroed (+ is_first=1); non-done columns survive."""
    step_data = {
        "rewards":    np.array([[[-5.0], [0.3]]], dtype=np.float32),  # [1, 2, 1]
        "terminated": np.array([[[1.0], [0.0]]], dtype=np.float32),
        "truncated":  np.array([[[0.0], [0.0]]], dtype=np.float32),
        "is_first":   np.array([[[0.0], [0.0]]], dtype=np.float32),
    }

    _reset_terminal_step_data(step_data, dones_idxes=[0])

    # Done env (col 0): terminal fields zeroed, is_first set.
    assert step_data["rewards"][0, 0, 0] == 0.0, (
        f"done-env reward not zeroed: {step_data['rewards'][0, 0, 0]}"
    )
    assert step_data["terminated"][0, 0, 0] == 0.0, (
        f"done-env terminated not zeroed: {step_data['terminated'][0, 0, 0]}"
    )
    assert step_data["truncated"][0, 0, 0] == 0.0, (
        f"done-env truncated not zeroed: {step_data['truncated'][0, 0, 0]}"
    )
    assert step_data["is_first"][0, 0, 0] == 1.0, (
        f"done-env is_first not set: {step_data['is_first'][0, 0, 0]}"
    )

    # Non-done env (col 1): live in-flight values untouched.
    assert step_data["rewards"][0, 1, 0] == np.float32(0.3), (
        f"non-done env reward was clobbered: {step_data['rewards'][0, 1, 0]}"
    )
    assert step_data["terminated"][0, 1, 0] == 0.0
    assert step_data["truncated"][0, 1, 0] == 0.0
    assert step_data["is_first"][0, 1, 0] == 0.0, (
        f"non-done env is_first was clobbered: {step_data['is_first'][0, 1, 0]}"
    )


# ---------------------------------------------------------------------------
# Test 2: buffer contents — the persisted is_first==1 row is clean
# ---------------------------------------------------------------------------

def test_next_buffer_row_is_clean():
    """Drive the driver's real two-write done-boundary sequence through a real
    SequentialReplayBuffer; the persisted episode-start row (is_first==1) must
    have rewards == terminated == truncated == 0."""
    n_envs, obs_dim, action_dim = 2, 4, 3
    buf = SequentialReplayBuffer(buffer_size=16, n_envs=n_envs, obs_keys=("obs",))

    # Terminal step: env 0 dies with a nonzero terminal reward; env 1 mid-episode.
    terminal_obs = np.full((1, n_envs, obs_dim), 7.0, dtype=np.float32)
    step_data = {
        "obs":        terminal_obs.copy(),
        "actions":    np.ones((1, n_envs, action_dim), dtype=np.float32),
        "rewards":    np.array([[[-5.0], [0.3]]], dtype=np.float32),
        "terminated": np.array([[[1.0], [0.0]]], dtype=np.float32),
        "truncated":  np.array([[[0.0], [0.0]]], dtype=np.float32),
        "is_first":   np.zeros((1, n_envs, 1), dtype=np.float32),
    }
    # Row 0 — the terminal row (mirrors buffer.add(step_data) at the loop top).
    buf.add(step_data, validate_args=False)

    # Row 1 — the reset_data second write for the done env
    # (mirrors dreamer_srl_main.py:1242-1251).
    dones = np.array([True, False])
    reset_data = {
        "obs":        step_data["obs"],          # true terminal obs
        "actions":    np.zeros((1, n_envs, action_dim), dtype=np.float32),
        "rewards":    step_data["rewards"],
        "terminated": step_data["terminated"],
        "truncated":  step_data["truncated"],
        "is_first":   np.zeros((1, n_envs, 1), dtype=np.float32),
    }
    buf.add(reset_data, done_mask=dones, validate_args=False)

    # The fix under test: reset the already-staged step_data for the done env,
    # then mirror the env auto-reset (fresh obs for env 0).
    _reset_terminal_step_data(step_data, [0])
    fresh_obs = np.full((obs_dim,), 2.0, dtype=np.float32)
    step_data["obs"][0, 0] = fresh_obs

    # Row 2 — the episode-start row (mirrors the next iteration's buffer.add).
    buf.add(step_data, validate_args=False)

    # Locate the persisted is_first==1 row for env column 0.
    n_written = buf._pos
    is_first_col0 = buf._buf["is_first"][:n_written, 0, 0]
    start_rows = np.where(is_first_col0 == 1.0)[0]
    assert len(start_rows) == 1, (
        f"expected exactly one is_first==1 row for env 0, got rows {start_rows}"
    )
    row = int(start_rows[0])

    # Sanity: it is the episode-start row (fresh reset obs, not the terminal obs).
    np.testing.assert_array_equal(buf._buf["obs"][row, 0], fresh_obs)

    # The core H5 assertion: the episode-start row is clean.
    assert buf._buf["rewards"][row, 0, 0] == 0.0, (
        f"episode-start row inherited terminal reward: "
        f"{buf._buf['rewards'][row, 0, 0]} (expected 0.0)"
    )
    assert buf._buf["terminated"][row, 0, 0] == 0.0, (
        f"episode-start row inherited death flag: "
        f"{buf._buf['terminated'][row, 0, 0]} (expected 0.0)"
    )
    assert buf._buf["truncated"][row, 0, 0] == 0.0, (
        f"episode-start row inherited truncation flag: "
        f"{buf._buf['truncated'][row, 0, 0]} (expected 0.0)"
    )

    # Non-done env (col 1) keeps its live in-flight values in the same row.
    assert buf._buf["rewards"][row, 1, 0] == np.float32(0.3), (
        f"non-done env reward was clobbered in the buffer: "
        f"{buf._buf['rewards'][row, 1, 0]}"
    )
    assert buf._buf["is_first"][row, 1, 0] == 0.0
