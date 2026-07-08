"""WP-SRL P2 + P8 regression tests — EnvIndependentSequentialReplayBuffer.

P2 ([[00_master_comparison]] §3 P2, area report 04 D-03): the port used ONE
shared write-head (`_pos`/`_full`) across all env columns of a single
[size, n_envs, ...] store. Every partial-done reset write advanced the shared
`_pos` for ALL envs while writing only the done envs' columns — punching a
stale/garbage row into every non-done env's stored history (area-report-4
probe: env-1's sampled sequence came back `[21, 31, 0, 41]` instead of
`[21, 31, 41]`). It also sized the store at `buffer.size` PER ENV, i.e.
num_envs x the sheeprl capacity (missing `// num_envs`, dreamer_v3.py:478).

Fix: `EnvIndependentSequentialReplayBuffer` — a structural port of sheeprl's
`EnvIndependentReplayBuffer` (vendor/sheeprl/sheeprl/data/buffers.py:529-699)
composing n_envs independent single-env SequentialReplayBuffers, each with its
own write head; reset writes are routed ONLY to done envs' sub-buffers
(dreamer_v3.py:650); sampling allocates the batch across sub-buffers via
bincount and concatenates on the batch axis (buffers.py:683-699).

P8 ([[00_master_comparison]] §3 P8, area report 04 D-07): the train gate
checked `buffer._pos >= seq_len` only; after a ring wrap `_pos` cycles low and
up to seq_len-1 owed grad steps were skipped. Fix: `ready_to_sample(seq_len)`
counts wrapped-full buffers as sampleable (`_full or _pos >= seq_len`).

Red evidence (pre-fix): all tests fail at collection with ImportError —
`EnvIndependentSequentialReplayBuffer` does not exist ("red by absence").
The no-hole fixture is additionally a TRUE behavioural discriminator: run
against the old shared-head `SequentialReplayBuffer(n_envs=2)`, the same
write sequence leaves env-1 rows `[21, 31, 0, 41]` (hole at the reset slot) —
reproduced pre-fix in tmp/20260708_wp_srl_p2_probe_red.log.

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
        -m pytest tests/algorithms/dreamer_srl/test_env_independent_buffer.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.dreamer_srl.buffers import (
    EnvIndependentSequentialReplayBuffer,
    SequentialReplayBuffer,
    validate_per_env_capacity,
)


def _row(vals: list[float]) -> dict[str, np.ndarray]:
    """One time-step of full-width data: {'obs': [1, n_envs, 1]}."""
    return {"obs": np.asarray(vals, dtype=np.float32).reshape(1, len(vals), 1)}


def _partial_done_write_sequence(buf) -> None:
    """The area-report-4 probe as a fixture (n_envs=2).

    Steps 1-2 write env0=[20,30], env1=[21,31]; env0 finishes its episode
    after step 2 -> reset row (obs 999 for env0; the env1 column of the
    full-width reset_data is a DON'T-CARE 0, mirroring the driver's
    fixed-width Fix-3 convention) routed via done_mask=[True, False];
    then step 3 writes env0=40, env1=41.
    """
    buf.add(_row([20.0, 21.0]))
    buf.add(_row([30.0, 31.0]))
    buf.add(_row([999.0, 0.0]), done_mask=np.array([True, False]))
    buf.add(_row([40.0, 41.0]))


def test_no_hole_rows_on_partial_done() -> None:
    """Non-done env's stored sequence stays contiguous across a partial done.

    On the old shared-head buffer this exact write sequence stored env-1 rows
    `[21, 31, 0, 41]` — the reset write advanced the SHARED `_pos` past env-1's
    untouched row 2, leaving a hole that later got sampled into training
    sequences. The per-env wrapper must store env-1 as contiguous
    `[21, 31, 41]` (its head never moved at the reset write) and env-0 as
    `[20, 30, 999, 40]` (reset row present).
    """
    buf = EnvIndependentSequentialReplayBuffer(buffer_size=8, n_envs=2, obs_keys=("obs",))
    _partial_done_write_sequence(buf)

    env0_rows = buf._buf[0]._buf["obs"][: buf._buf[0]._pos, 0, 0].tolist()
    env1_rows = buf._buf[1]._buf["obs"][: buf._buf[1]._pos, 0, 0].tolist()

    assert env1_rows == [21.0, 31.0, 41.0], (
        f"env-1 rows {env1_rows} != [21, 31, 41]: a hole row leaked into the "
        "non-done env's history (WP-SRL P2 shared-write-head regression; old "
        "shared-head buffer stored [21, 31, 0, 41])."
    )
    assert env0_rows == [20.0, 30.0, 999.0, 40.0], (
        f"env-0 rows {env0_rows} != [20, 30, 999, 40]: the done env's reset "
        "row was not routed to its own sub-buffer (sheeprl dreamer_v3.py:650)."
    )


def test_per_env_heads_advance_independently() -> None:
    """After the partial-done write, env-0's head is one ahead of env-1's."""
    buf = EnvIndependentSequentialReplayBuffer(buffer_size=8, n_envs=2, obs_keys=("obs",))
    _partial_done_write_sequence(buf)
    assert buf._buf[0]._pos == buf._buf[1]._pos + 1, (
        f"env-0 _pos={buf._buf[0]._pos}, env-1 _pos={buf._buf[1]._pos}: "
        "expected independent heads (env-0 got 4 writes incl. the reset row, "
        "env-1 got 3)."
    )


def test_capacity_division() -> None:
    """Driver-mirroring sizing: cfg buffer.size // num_envs per sub-buffer.

    sheeprl sizes each sub-buffer at cfg.buffer.size // (num_envs * world_size)
    (dreamer_v3.py:478, world_size == 1); the division happens in the DRIVER,
    mirrored here. The old port passed the full cfg size per env column —
    num_envs x the reference capacity.
    """
    cfg_buffer_size, num_envs = 256000, 4
    per_env_buffer_size = cfg_buffer_size // num_envs  # driver-side division
    buf = EnvIndependentSequentialReplayBuffer(
        buffer_size=per_env_buffer_size, n_envs=num_envs, obs_keys=("obs",)
    )
    assert len(buf._buf) == num_envs
    for i, sub in enumerate(buf._buf):
        assert sub._buffer_size == 64000, (
            f"sub-buffer {i} _buffer_size={sub._buffer_size} != 64000 "
            "(= 256000 // 4, sheeprl dreamer_v3.py:478 capacity semantics)"
        )
        assert sub._n_envs == 1


def test_sample_shape_and_env_isolation() -> None:
    """Sampled sequences have sheeprl's output shape and never cross envs.

    Fill 2 envs with disjoint value ranges (env0 in [100, 200), env1 in
    [1000, 2000)); sample and assert (a) output shape
    [n_samples, seq_len, batch_size, ...] with the batch allocated across
    sub-buffers (buffers.py:683-699 bincount + axis-2 concat), (b) every
    sampled sequence lies entirely inside ONE env's value range.
    """
    buf = EnvIndependentSequentialReplayBuffer(buffer_size=64, n_envs=2, obs_keys=("obs",))
    for t in range(32):
        buf.add(_row([100.0 + t, 1000.0 + t]))

    out = buf.sample(batch_size=8, sequence_length=3, n_samples=2)
    assert out["obs"].shape == (2, 3, 8, 1), (
        f"sample shape {out['obs'].shape} != (2, 3, 8, 1) "
        "([n_samples, seq_len, batch_size, obs_dim])"
    )

    obs = out["obs"]
    for s in range(2):
        for b in range(8):
            seq = obs[s, :, b, 0]
            in_env0 = np.all((seq >= 100.0) & (seq < 200.0))
            in_env1 = np.all((seq >= 1000.0) & (seq < 2000.0))
            assert in_env0 or in_env1, (
                f"sample {s} batch {b} sequence {seq.tolist()} mixes env value "
                "ranges: cross-env sequence leaked through the per-env sampler."
            )


@pytest.mark.parametrize("cls_name", ["plain", "wrapper"])
def test_ready_to_sample_after_wrap(cls_name: str) -> None:
    """WP-SRL P8: a wrapped-full buffer is sampleable even when _pos < seq_len.

    Tiny buffer (size 8), seq_len=4: write 10 rows so the ring wraps
    (_full=True, _pos=2). The OLD train-gate expression `_pos >= seq_len`
    evaluates False in this exact state — up to seq_len-1 owed grad steps
    were skipped after every ring wrap (area report 04 D-07). The new
    `ready_to_sample(seq_len)` must return True. Covers both the plain
    single-env class (still used at num_envs=1 GPU-buffer mode and as the
    sub-buffer) and the wrapper.
    """
    seq_len = 4
    if cls_name == "plain":
        buf = SequentialReplayBuffer(buffer_size=8, n_envs=1, obs_keys=("obs",))
        for t in range(10):
            buf.add(_row([float(t)]))
        pos, full = buf._pos, buf._full
    else:
        buf = EnvIndependentSequentialReplayBuffer(buffer_size=8, n_envs=1, obs_keys=("obs",))
        for t in range(10):
            buf.add(_row([float(t)]))
        pos, full = buf._buf[0]._pos, buf._buf[0]._full

    assert full is True and pos == 2, f"fixture drift: _full={full}, _pos={pos}"
    # Documentation of what was wrong: the old gate expression is False here.
    assert not (pos >= seq_len), "fixture no longer reproduces the post-wrap state"
    assert buf.ready_to_sample(seq_len) is True, (
        "ready_to_sample(4) returned False on a wrapped-full buffer "
        "(WP-SRL P8: the old `_pos >= seq_len` gate skipped owed grad steps "
        "after every ring wrap)."
    )
    # And the buffer really is sampleable in this state.
    out = buf.sample(batch_size=2, sequence_length=seq_len, n_samples=1)
    assert out["obs"].shape == (1, seq_len, 2, 1)


def test_filled_size_and_reset() -> None:
    """filled_size sums per-sub-buffer fill; reset clears every head."""
    buf = EnvIndependentSequentialReplayBuffer(buffer_size=8, n_envs=2, obs_keys=("obs",))
    _partial_done_write_sequence(buf)          # env0: 4 rows, env1: 3 rows
    assert buf.filled_size == 7

    plain = SequentialReplayBuffer(buffer_size=8, n_envs=1, obs_keys=("obs",))
    for t in range(10):                        # wraps: full -> filled == capacity
        plain.add(_row([float(t)]))
    assert plain.filled_size == 8

    buf.reset()
    assert buf.filled_size == 0
    assert all((not b._full) and b._pos == 0 for b in buf._buf)


def test_constructor_validation() -> None:
    """Non-positive sizes are rejected (mirrors sheeprl buffers.py:556-559)."""
    with pytest.raises(ValueError):
        EnvIndependentSequentialReplayBuffer(buffer_size=0, n_envs=2)
    with pytest.raises(ValueError):
        EnvIndependentSequentialReplayBuffer(buffer_size=8, n_envs=0)
    with pytest.raises(ValueError):
        EnvIndependentSequentialReplayBuffer(buffer_size=8, n_envs=2).sample(
            batch_size=0, sequence_length=1
        )


def test_per_env_capacity_guard() -> None:
    """N1 (review_srl_parity_fixes.md): fail fast when per-env capacity < seq_len.

    With P2's driver-side sizing (`buffer.size // num_envs`), a small
    configured buffer + many envs makes `per_env_buffer_size < seq_len`
    reachable. `ready_to_sample()` returns True once the ring wraps full, so
    the run crashes only at the FIRST post-prefill `sample()` (buffers.py:
    367-371) instead of at startup. The driver now calls
    `validate_per_env_capacity(...)` before buffer construction.

    Red evidence (pre-fix): whole file fails at collection with ImportError —
    `validate_per_env_capacity` does not exist ("red by absence"; the
    deferred-crash pathology itself is pinned behaviorally below).
    """
    # Guard fires: 128 // 16 = 8 < seq_len 64.
    with pytest.raises(ValueError, match="per-env"):
        validate_per_env_capacity(
            per_env_buffer_size=8,
            sequence_length=64,
            configured_buffer_size=128,
            num_envs=16,
        )
    # Guard passes at the boundary (per-env capacity == seq_len).
    validate_per_env_capacity(
        per_env_buffer_size=64,
        sequence_length=64,
        configured_buffer_size=1024,
        num_envs=16,
    )


def test_deferred_crash_without_guard() -> None:
    """The pathology the N1 guard preempts: gate says yes, sample() crashes.

    A wrapped-full buffer with capacity < seq_len passes `ready_to_sample()`
    (P8's `_full or _pos >= seq_len` is True via `_full`) yet every
    `sample()` raises — i.e. without the startup guard the failure surfaces
    only after the full prefill phase has been paid for.
    """
    buf = SequentialReplayBuffer(buffer_size=4, n_envs=1, obs_keys=("obs",))
    for t in range(6):                      # wrap: _full = True
        buf.add(_row([float(t)]))
    assert buf.ready_to_sample(8)           # gate is (misleadingly) open
    with pytest.raises(ValueError, match="greater than"):
        buf.sample(batch_size=1, sequence_length=8)
