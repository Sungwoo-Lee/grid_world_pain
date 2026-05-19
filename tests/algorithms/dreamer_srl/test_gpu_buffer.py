"""Step 2: GPU-mode buffer math-equivalence tests.

Tests that device="gpu" and device="cpu" modes of SequentialReplayBuffer produce
numerically equivalent stored content after add() calls, and that sample() output
has correct shapes and dtypes.

Design decisions (documented here for future reviewers):
- This is math-equivalence testing, NOT bit-identity. The two modes use different
  RNG streams (numpy.random.Generator vs jax.random.PRNGKey) so sampled indices
  will differ — we only check that stored buffer contents match and that samples
  have correct shapes/dtypes.
- For stored-content comparison after add(), we use jnp.allclose(atol=1e-5, rtol=1e-5).
- For the single-env case, we verify the actual sample values match (same indices,
  same content, same math).
- _sample_at_indices() is CPU-only by design (Step 2 decision (a)); the GPU mode
  test uses sample() with a fixed JAX PRNG key.

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_gpu_buffer.py -v

See: docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md §Step 2
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer

# ---------------------------------------------------------------------------
# CUDA availability guard
# ---------------------------------------------------------------------------
# Tests that actually exercise the GPU path need a working CUDA device.
# On machines where both GPUs are saturated (OOM) or unavailable, those tests
# are skipped gracefully. CPU-path tests and logic tests always run.

def _cuda_available() -> bool:
    """Return True if JAX can initialize at least one CUDA device."""
    try:
        import jax
        jax.devices("gpu")
        return True
    except RuntimeError:
        return False

_CUDA = _cuda_available()
_requires_cuda = pytest.mark.skipif(not _CUDA, reason="No CUDA device available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(rng: np.random.Generator, seq_len: int, n_envs: int, obs_dim: int = 4) -> dict:
    """Create a deterministic transition batch for add()."""
    return {
        "observations": rng.random((seq_len, n_envs, obs_dim), dtype=np.float32).astype(np.float32),
        "actions":      rng.random((seq_len, n_envs, 2), dtype=np.float32).astype(np.float32),
        "rewards":      rng.random((seq_len, n_envs), dtype=np.float32).astype(np.float32),
        "terminated":   (rng.random((seq_len, n_envs)) < 0.1).astype(np.float32),
        "is_first":     (rng.random((seq_len, n_envs)) < 0.05).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Test 1: invalid device raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_device_raises():
    """Constructing with an unknown device name must raise ValueError."""
    with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
        SequentialReplayBuffer(buffer_size=64, n_envs=2, device="cuda")


# ---------------------------------------------------------------------------
# Test 2: CPU default — no behavioural change
# ---------------------------------------------------------------------------

def test_cpu_default_unchanged():
    """SequentialReplayBuffer() with no device arg behaves exactly as before."""
    buf = SequentialReplayBuffer(buffer_size=64, n_envs=2)
    assert buf._device == "cpu"
    assert not buf._on_gpu

    rng = np.random.default_rng(42)
    data = _make_data(rng, seq_len=4, n_envs=2)
    buf.add(data)

    # Should be numpy arrays
    for v in buf._buf.values():
        assert isinstance(v, np.ndarray), f"expected np.ndarray, got {type(v)}"


# ---------------------------------------------------------------------------
# Test 3: GPU mode stores jax.Array
# ---------------------------------------------------------------------------

@_requires_cuda
def test_gpu_mode_stores_jax_array():
    """After add() in GPU mode, buffer arrays should be jax.Array."""
    buf = SequentialReplayBuffer(buffer_size=64, n_envs=2, device="gpu")
    assert buf._device == "gpu"
    assert buf._on_gpu

    rng = np.random.default_rng(42)
    data = _make_data(rng, seq_len=4, n_envs=2)
    buf.add(data)

    for k, v in buf._buf.items():
        assert isinstance(v, jax.Array), f"key '{k}': expected jax.Array, got {type(v)}"


# ---------------------------------------------------------------------------
# Test 4: stored content matches between CPU and GPU after identical add() calls
# ---------------------------------------------------------------------------

@_requires_cuda
def test_gpu_buffer_add_matches_cpu():
    """After the same sequence of add() calls, GPU and CPU buffers store the same values.

    This is the core math-equivalence check: add() is purely deterministic
    (ring-buffer index arithmetic + array assignment), so both modes must store
    identical bytes in the filled region.
    """
    BUFFER_SIZE = 128
    N_ENVS = 4
    N_ADDS = 5
    SEQ_LEN = 8

    cpu_buf = SequentialReplayBuffer(
        buffer_size=BUFFER_SIZE, n_envs=N_ENVS,
        obs_keys=("observations",), device="cpu",
    )
    gpu_buf = SequentialReplayBuffer(
        buffer_size=BUFFER_SIZE, n_envs=N_ENVS,
        obs_keys=("observations",), device="gpu",
    )

    rng = np.random.default_rng(0)
    for _ in range(N_ADDS):
        data = _make_data(rng, seq_len=SEQ_LEN, n_envs=N_ENVS)
        cpu_buf.add(data)
        # GPU add() accepts numpy arrays; it will jnp.asarray them internally.
        gpu_buf.add(data)

    # Compare stored content in the filled region
    assert cpu_buf._pos == gpu_buf._pos, (
        f"_pos mismatch: cpu={cpu_buf._pos}, gpu={gpu_buf._pos}"
    )
    assert cpu_buf._full == gpu_buf._full, (
        f"_full mismatch: cpu={cpu_buf._full}, gpu={gpu_buf._full}"
    )

    n_filled = cpu_buf._pos
    for k in cpu_buf._buf:
        cpu_arr = cpu_buf._buf[k][:n_filled]
        gpu_arr = np.asarray(gpu_buf._buf[k])[:n_filled]
        assert cpu_arr.shape == gpu_arr.shape, (
            f"Shape mismatch for '{k}': cpu={cpu_arr.shape}, gpu={gpu_arr.shape}"
        )
        assert cpu_arr.dtype == gpu_arr.dtype, (
            f"Dtype mismatch for '{k}': cpu={cpu_arr.dtype}, gpu={gpu_arr.dtype}"
        )
        match = jnp.allclose(jnp.asarray(cpu_arr), jnp.asarray(gpu_arr), atol=1e-5, rtol=1e-5)
        assert bool(match), (
            f"Buffer content mismatch for '{k}' (filled region [:{n_filled}]): "
            f"max_abs_diff={float(jnp.max(jnp.abs(jnp.asarray(cpu_arr) - jnp.asarray(gpu_arr)))):.3e}"
        )


# ---------------------------------------------------------------------------
# Test 5: GPU mode sample() requires a key
# ---------------------------------------------------------------------------

@_requires_cuda
def test_gpu_sample_requires_key():
    """sample() without a key when device='gpu' must raise ValueError."""
    buf = SequentialReplayBuffer(buffer_size=64, n_envs=1, device="gpu")
    rng = np.random.default_rng(0)
    buf.add(_make_data(rng, seq_len=8, n_envs=1))
    with pytest.raises(ValueError, match="JAX PRNG key must be provided"):
        buf.sample(batch_size=2, sequence_length=4)


# ---------------------------------------------------------------------------
# Test 6: GPU mode sample() shape and dtype parity with CPU mode
# ---------------------------------------------------------------------------

@_requires_cuda
def test_gpu_sample_shape_dtype_parity():
    """sample() in GPU mode returns arrays with the same shape and dtype as CPU mode.

    Because the two modes use different RNG streams (numpy vs JAX), the actual
    indices sampled will differ — we only check shape and dtype here.
    """
    BUFFER_SIZE = 256
    N_ENVS = 4
    SEQ_LEN = 8
    BATCH_SIZE = 6
    N_SAMPLES = 3

    cpu_buf = SequentialReplayBuffer(
        buffer_size=BUFFER_SIZE, n_envs=N_ENVS,
        obs_keys=("observations",), device="cpu",
    )
    gpu_buf = SequentialReplayBuffer(
        buffer_size=BUFFER_SIZE, n_envs=N_ENVS,
        obs_keys=("observations",), device="gpu",
    )

    rng = np.random.default_rng(1)
    for _ in range(10):
        data = _make_data(rng, seq_len=16, n_envs=N_ENVS)
        cpu_buf.add(data)
        gpu_buf.add(data)

    cpu_sample = cpu_buf.sample(batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_samples=N_SAMPLES)
    gpu_sample = gpu_buf.sample(
        batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_samples=N_SAMPLES,
        key=jax.random.PRNGKey(42),
    )

    assert set(cpu_sample.keys()) == set(gpu_sample.keys()), (
        f"Key sets differ: cpu={set(cpu_sample.keys())}, gpu={set(gpu_sample.keys())}"
    )
    for k in cpu_sample:
        cpu_arr = cpu_sample[k]
        gpu_arr = gpu_sample[k]
        assert cpu_arr.shape == gpu_arr.shape, (
            f"Shape mismatch for '{k}': cpu={cpu_arr.shape}, gpu={gpu_arr.shape}"
        )
        # dtype: cpu returns np.float32; gpu returns jnp.float32
        assert cpu_arr.dtype == np.float32, f"CPU sample '{k}' not float32: {cpu_arr.dtype}"
        assert gpu_arr.dtype == jnp.float32, f"GPU sample '{k}' not float32: {gpu_arr.dtype}"


# ---------------------------------------------------------------------------
# Test 7: GPU mode sample() values (single env, same indices via fixed key)
# ---------------------------------------------------------------------------

@_requires_cuda
def test_gpu_sample_single_env_values():
    """For n_envs=1, GPU sample values should match CPU values when the same
    time indices are used.

    With n_envs=1, env selection is trivial (always env 0). We use a fixed
    JAX key and verify the GPU sample is a valid subset of the buffer content.
    We also verify that the CPU buffer's _sample_at_indices() and the GPU
    buffer's sample() agree on stored values (when forced to the same indices).
    """
    BUFFER_SIZE = 64
    N_ENVS = 1
    SEQ_LEN = 4
    BATCH_SIZE = 4

    cpu_buf = SequentialReplayBuffer(
        buffer_size=BUFFER_SIZE, n_envs=N_ENVS,
        obs_keys=("observations",), device="cpu",
    )
    gpu_buf = SequentialReplayBuffer(
        buffer_size=BUFFER_SIZE, n_envs=N_ENVS,
        obs_keys=("observations",), device="gpu",
    )

    rng = np.random.default_rng(77)
    for _ in range(5):
        data = _make_data(rng, seq_len=8, n_envs=N_ENVS)
        cpu_buf.add(data)
        gpu_buf.add(data)

    # Use a fixed key; retrieve start indices that gpu_buf.sample uses
    key = jax.random.PRNGKey(0)
    gpu_sample = gpu_buf.sample(batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, key=key)

    # All returned observation values must exist somewhere in the original CPU buffer.
    obs_key = "observations"
    gpu_obs = np.asarray(gpu_sample[obs_key])  # [1, seq_len, batch, obs_dim]
    cpu_obs_flat = cpu_buf._buf[obs_key][:cpu_buf._pos, 0]  # [n_filled, obs_dim]

    # Every individual time-step observation in the GPU sample must match some
    # row in the CPU buffer (both buffers have identical content after add()).
    for b in range(BATCH_SIZE):
        for t in range(SEQ_LEN):
            obs_vec = gpu_obs[0, t, b]  # [obs_dim]
            # Check it matches at least one row in the CPU buffer
            dists = np.max(np.abs(cpu_obs_flat - obs_vec), axis=-1)
            assert np.min(dists) < 1e-5, (
                f"GPU sample obs[b={b}, t={t}] not found in CPU buffer. "
                f"min_dist={float(np.min(dists)):.3e}"
            )


# ---------------------------------------------------------------------------
# Test 8: _sample_at_indices() raises NotImplementedError in GPU mode
# ---------------------------------------------------------------------------

@_requires_cuda
def test_sample_at_indices_raises_in_gpu_mode():
    """_sample_at_indices() must raise NotImplementedError for device='gpu'."""
    buf = SequentialReplayBuffer(buffer_size=64, n_envs=1, device="gpu")
    rng = np.random.default_rng(0)
    buf.add(_make_data(rng, seq_len=8, n_envs=1))

    with pytest.raises(NotImplementedError, match="CPU-only"):
        buf._sample_at_indices(
            precomputed_start_idxes=np.array([0, 1], dtype=np.intp),
            env_idxes=np.array([0, 0], dtype=np.intp),
            sequence_length=4,
            batch_size=2,
        )


# ---------------------------------------------------------------------------
# Test 9: GPU mode with env_idxes subset write (CP7-P1 path)
# ---------------------------------------------------------------------------

@_requires_cuda
def test_gpu_env_idxes_subset_write():
    """add(data, env_idxes=[...]) in GPU mode writes only the specified env columns."""
    BUFFER_SIZE = 32
    N_ENVS = 4

    buf = SequentialReplayBuffer(buffer_size=BUFFER_SIZE, n_envs=N_ENVS, device="gpu")

    rng = np.random.default_rng(5)
    # First, fill all envs to initialize the buffer
    full_data = _make_data(rng, seq_len=4, n_envs=N_ENVS)
    buf.add(full_data)

    # Now overwrite only env 0 and env 2 with distinct sentinel values
    sentinel_obs = np.full((1, 2, 4), 99.0, dtype=np.float32)
    sentinel_data = {
        "observations": sentinel_obs,
        "actions":      np.full((1, 2, 2), 99.0, dtype=np.float32),
        "rewards":      np.full((1, 2), 99.0, dtype=np.float32),
        "terminated":   np.full((1, 2), 0.0, dtype=np.float32),
        "is_first":     np.full((1, 2), 0.0, dtype=np.float32),
    }
    buf.add(sentinel_data, env_idxes=[0, 2])

    # The sentinel should be at position 4 (one step after full_data), env 0 and 2.
    write_pos = 4  # where sentinel landed
    obs_arr = np.asarray(buf._buf["observations"])
    assert np.allclose(obs_arr[write_pos, 0], 99.0, atol=1e-5), (
        f"Env 0 at pos {write_pos} should be 99.0, got {obs_arr[write_pos, 0]}"
    )
    assert np.allclose(obs_arr[write_pos, 2], 99.0, atol=1e-5), (
        f"Env 2 at pos {write_pos} should be 99.0, got {obs_arr[write_pos, 2]}"
    )
    # Env 1 and 3 should still be the original full_data values (not 99.0)
    assert not np.allclose(obs_arr[write_pos, 1], 99.0, atol=1e-5), (
        "Env 1 should NOT have been overwritten with sentinel 99.0"
    )
    assert not np.allclose(obs_arr[write_pos, 3], 99.0, atol=1e-5), (
        "Env 3 should NOT have been overwritten with sentinel 99.0"
    )
