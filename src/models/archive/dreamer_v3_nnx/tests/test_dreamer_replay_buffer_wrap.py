"""Regression tests for H7 — DreamerV3-NNX replay buffer wrap splices two envs
(WP-D, Finding 2 of the 2026-07-04 Fable 5 re-diagnosis).

Plain-language context: the Dreamer replay buffer stores env-major blocks of
`sequence_length` consecutive transitions (one environment's trajectory per
block) and `sample()` only reads windows that start at multiples of
`sequence_length` from index 0. Writes, however, wrap at `% capacity`. If
capacity is not a multiple of sequence_length (the live config's 1,000,000 %
128 = 64), then after the first wrap every write lands shifted relative to the
fixed sampling grid, and sampled "sequences" contain the tail of one
environment's block glued to the head of another's — a hard mid-sequence
teleport with no is_first marker, which the RSSM trains through as if it were
a real transition. Each further wrap shifts the grid again, so corruption
grows toward the whole buffer. The fix floors capacity to a multiple of
sequence_length at construction (runtime rounding — config YAML intentionally
untouched), mirroring the positive-buffer precedent in train.py.

These are pure ReplayBuffer unit tests (no env, no trainer). The content-level
test tags every transition of block b with obs = float(b), so a spliced sample
window is directly visible as a mid-window change of the obs channel.

See docs/develop/active/issues/diag_fable5_20260704/
fix_plan_h6h7_dreamer_v3_world_model.md (plan) and 05_dreamer_v3_nnx.md
(Finding 2).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import pytest

from src.models.dreamer_v3_trainer import ReplayBuffer

SEQ_LEN = 8


def _make_buffer(capacity=100):
    # "gpu" device mode = JAX arrays (they run fine on the CPU backend) —
    # this is the buffer mode the live Dreamer configs use.
    return ReplayBuffer(capacity=capacity, sequence_length=SEQ_LEN,
                        obs_dim=1, action_dim=2, device="gpu")


def _write_blocks(buf, num_blocks, blocks_per_call=2, after_each_call=None):
    """Write `num_blocks` env-major blocks of SEQ_LEN transitions, in
    add_batch calls of `blocks_per_call` blocks each (mimicking the live
    `num_envs x collect_interval` write shape). Every transition of block b
    carries obs = float(b) (a block-identity channel); is_first = 1 only on
    each block's first slot."""
    for start in range(0, num_blocks, blocks_per_call):
        n = min(blocks_per_call, num_blocks - start)
        num_items = n * SEQ_LEN
        block_ids = jnp.repeat(jnp.arange(start, start + n, dtype=jnp.float32),
                               SEQ_LEN)
        obs = block_ids[:, None]                                   # (items, 1)
        actions = jnp.zeros((num_items, 2), dtype=jnp.float32)
        rewards = jnp.zeros((num_items,), dtype=jnp.float32)
        dones = jnp.zeros((num_items,), dtype=jnp.float32)
        is_firsts = jnp.tile(
            jnp.eye(SEQ_LEN, dtype=jnp.float32)[0], n)             # 1,0,...,0 per block
        term_reasons = jnp.zeros((num_items,), dtype=jnp.float32)
        buf.add_batch(obs, actions, rewards, dones, is_firsts, term_reasons)
        if after_each_call is not None:
            after_each_call(buf)


def test_capacity_rounded_down_to_sequence_multiple():
    """FAILS PRE-FIX: capacity must be floored to a multiple of
    sequence_length at construction (100 -> 96 for sequence_length 8), so the
    write grid and the seq-aligned sampling grid can never drift apart."""
    buf = _make_buffer(capacity=100)
    assert buf.capacity == 96, (
        f"capacity must be rounded down to a multiple of sequence_length "
        f"({SEQ_LEN}); got {buf.capacity}"
    )


def test_capacity_below_sequence_length_raises():
    """FAILS PRE-FIX: a capacity smaller than one sequence_length rounds to
    zero blocks — the buffer could never serve a single training sequence and
    must refuse construction loudly (no-fallback-defaults discipline)."""
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=5, sequence_length=SEQ_LEN,
                     obs_dim=1, action_dim=2, device="gpu")


def test_wrapped_buffer_never_splices_two_envs_mid_sequence():
    """FAILS PRE-FIX (content-level, the actual H7 corruption): after the ring
    buffer wraps, every sampled sequence must still contain a single block's
    content. Pre-fix (capacity 100, seq 8): the first wrap lands writes at
    index 112 % 100 = 12 (≡ 4 mod 8), so 8-aligned sample windows straddle two
    blocks — the obs channel changes mid-window with no episode marker.
    Post-fix (capacity 96): the grid stays aligned forever."""
    buf = _make_buffer(capacity=100)
    _write_blocks(buf, num_blocks=30)  # wraps the buffer several times

    key = jax.random.PRNGKey(0)
    for _ in range(50):
        key, subkey = jax.random.split(key)
        batch = buf.sample(32, key=subkey)
        assert batch is not None
        obs = batch['obs'][..., 0]  # (32, SEQ_LEN) block-identity channel
        constant = jnp.all(obs == obs[:, :1], axis=1)
        assert bool(jnp.all(constant)), (
            "sampled sequence splices two env blocks mid-window (H7): "
            f"offending windows (block-id channel rows): "
            f"{obs[~constant][:4].tolist()}"
        )


def test_write_index_stays_block_aligned_across_wrap():
    """FAILS PRE-FIX (supplementary arithmetic pin): the write index must stay
    a multiple of sequence_length after every add_batch, including across
    wraps — the invariant that keeps writes on the sampling grid."""
    buf = _make_buffer(capacity=100)

    def _assert_aligned(b):
        assert b.idx % b.sequence_length == 0, (
            f"write index {b.idx} is not a multiple of sequence_length "
            f"{b.sequence_length} — the write grid has drifted off the "
            f"sampling grid (H7)"
        )

    _write_blocks(buf, num_blocks=30, after_each_call=_assert_aligned)
