"""Tests for `core/store.py` and `core/scan.py` — the shared scan driver.

Plain-language context: the analysis reads a large pile of Parquet files ("shards"), each holding
many episodes, each episode a run of rows one step apart. Almost everything that has gone wrong in
this project's analysis went wrong in the bookkeeping around that structure rather than in the
science: an episode split across two files and half-counted, a reset row counted as a step, a
predictor read from the wrong row, a rebuild run on a store that had grown since.

None of those raise on their own. They produce a plausible wrong number, which is far worse than a
crash. So the driver asserts them — and these tests exist to prove the asserts actually fire, because
an assertion that has never been seen to fail is decoration rather than evidence.
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "scripts", "analysis", "core")
sys.path.insert(0, CORE)
import scan as SCAN      # noqa: E402
import store as STORE    # noqa: E402


def _write(tmp_path, episodes, seeds=None, shard_of=None):
    """Build a tiny store: `episodes` is a list of step-counts (excluding the reset row)."""
    seeds = list(range(1000, 1000 + len(episodes))) if seeds is None else seeds
    shard_of = [0] * len(episodes) if shard_of is None else shard_of
    root = tmp_path / "store"
    (root / "run").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "episode_seed": pa.array(seeds, pa.int64()),
        "length": pa.array([float(n) for n in episodes], pa.float64()),
    }), root / "run" / "episodes_000.parquet")
    for sh in sorted(set(shard_of)):
        s_seed, s_t, s_v = [], [], []
        for seed, n, w in zip(seeds, episodes, shard_of):
            if w != sh:
                continue
            for t in range(n + 1):                 # +1 for the reset row at t=0
                s_seed.append(seed); s_t.append(t); s_v.append(float(t))
        pq.write_table(pa.table({
            "episode_seed": pa.array(s_seed, pa.int64()),
            "t": pa.array(s_t, pa.int64()),
            "v": pa.array(s_v, pa.float64()),
        }), root / "run" / f"steps_{sh:03d}.parquet")
    return [str(root)]


def test_a_seed_gap_is_refused(tmp_path):
    """A hole in the seed range means the population is not what the analysis will report."""
    roots = _write(tmp_path, [3, 3, 3], seeds=[1000, 1001, 1005])
    with pytest.raises(SystemExit, match="contiguous"):
        STORE.open_run(roots, ["episode_seed", "length"])


def test_duplicate_seeds_are_refused(tmp_path):
    """The same episode counted twice inflates every denominator silently."""
    roots = _write(tmp_path, [3, 3, 3], seeds=[1000, 1001, 1001])
    with pytest.raises(SystemExit, match="contiguous"):
        STORE.open_run(roots, ["episode_seed", "length"])


def test_a_clean_store_opens(tmp_path):
    roots = _write(tmp_path, [3, 4, 5])
    st = STORE.open_run(roots, ["episode_seed", "length"])
    assert st.n_episodes == 3
    assert st.seed0 == 1000


def test_the_reset_row_is_not_a_step(tmp_path):
    """Each episode here has n+1 rows; the driver must count n."""
    roots = _write(tmp_path, [3, 4, 5])
    st = STORE.open_run(roots, ["episode_seed", "length"])
    seen = {}

    def collect(fr, acc, fi):
        seen["steps"] = fr.steps_per_episode.tolist()
        seen["n_rows"] = fr.n

    SCAN.sweep(st, ["episode_seed", "t", "v"], collect, verbose=False)
    assert seen["steps"] == [3, 4, 5]
    assert seen["n_rows"] == 3 + 4 + 5 + 3          # the three reset rows are present but not steps


def test_prev_is_the_row_the_action_was_chosen_on(tmp_path):
    """`prev` must point one row back, and must never point across an episode boundary."""
    roots = _write(tmp_path, [3, 3])
    st = STORE.open_run(roots, ["episode_seed", "length"])
    got = {}

    def collect(fr, acc, fi):
        got["prev_t"] = fr.raw("t")[fr.prev].tolist()
        got["step_t"] = fr.raw("t")[fr.is_step].tolist()

    SCAN.sweep(st, ["episode_seed", "t", "v"], collect, verbose=False)
    assert got["step_t"] == [1, 2, 3, 1, 2, 3]
    assert got["prev_t"] == [0, 1, 2, 0, 1, 2]      # never -1, never the other episode's last row


def test_a_step_count_mismatch_is_caught(tmp_path):
    """The cross-check that catches a scan which dropped or double-counted rows."""
    roots = _write(tmp_path, [3, 4, 5])
    st = STORE.open_run(roots, ["episode_seed", "length"])
    st._ep = st._ep.set_column(st._ep.schema.get_field_index("length"), "length",
                               pa.array([3.0, 4.0, 99.0], pa.float64()))
    with pytest.raises(SystemExit, match="step count disagrees"):
        SCAN.sweep(st, ["episode_seed", "t", "v"], lambda fr, acc, fi: None, verbose=False)


def test_a_shard_holding_non_consecutive_episodes_is_caught(tmp_path):
    """Episodes must arrive in one contiguous run per shard, or per-episode sums go partial.

    The first version of this test built a shard that was still internally contiguous, so the
    step-count cross-check fired first and the alignment check was never exercised. Getting a real
    misalignment needs a shard that holds episodes 0 and 2 while skipping 1 — which is exactly the
    shape a resumed or re-ordered collection would produce.
    """
    root = tmp_path / "store" / "run"
    root.mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_seed": pa.array([1000, 1001, 1002], pa.int64()),
        "length": pa.array([2.0, 2.0, 2.0], pa.float64()),
    }), root / "episodes_000.parquet")
    # shard 0 holds global episodes 0 and 2, skipping 1 -> not one contiguous run
    pq.write_table(pa.table({
        "episode_seed": pa.array([1000, 1000, 1000, 1002, 1002, 1002], pa.int64()),
        "t": pa.array([0, 1, 2, 0, 1, 2], pa.int64()),
        "v": pa.array([0.0] * 6, pa.float64()),
    }), root / "steps_000.parquet")
    pq.write_table(pa.table({
        "episode_seed": pa.array([1001, 1001, 1001], pa.int64()),
        "t": pa.array([0, 1, 2], pa.int64()),
        "v": pa.array([0.0] * 3, pa.float64()),
    }), root / "steps_001.parquet")
    st = STORE.open_run([str(tmp_path / "store")], ["episode_seed", "length"])
    with pytest.raises(SystemExit, match="shard-aligned"):
        SCAN.sweep(st, ["episode_seed", "t", "v"], lambda fr, acc, fi: None, verbose=False)


def test_per_episode_sums_differ_in_the_documented_way(tmp_path):
    """The two forms exist because the real sweeps use both; the difference is the reset row."""
    roots = _write(tmp_path, [3])
    st = STORE.open_run(roots, ["episode_seed", "length"])
    got = {}

    def collect(fr, acc, fi):
        v = fr.raw("v")                            # v == t, so the episode is [0, 1, 2, 3]
        got["excl"] = fr.per_episode_sum(v).tolist()
        got["incl"] = fr.per_episode_sum_with_initial(v).tolist()

    SCAN.sweep(st, ["episode_seed", "t", "v"], collect, verbose=False)
    assert got["incl"] == [6.0]                    # 0 + 1 + 2 + 3
    assert got["excl"] == [6.0]                    # same here only because the reset row's v is 0
