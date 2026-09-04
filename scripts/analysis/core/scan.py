#!/usr/bin/env python3
"""scan.py - THE scan. One loop, every check, always.

Three programs in this repo read the trajectory store and each re-implemented this loop, months
apart, each picking up whichever correctness checks its author had in mind that day. Of the twenty
applicable check-cells across them, ten were empty - and all three of the bugs this analysis has
actually hit sat in an empty cell. None of them raised: a third of episodes with no predator being
filed into the comparison group, a rebuild running on stale data, and a dose-response reading its
predictor from the wrong row all produced figures that looked entirely fine.

A DRIVER, NOT A MINI-LANGUAGE. An earlier draft proposed a declarative API where a study would
describe what to count and never write a loop. That does not survive contact with what these sweeps
actually do - crossed two-dimensional grids, per-episode-by-bin matrices, composite geometry
conditions, quantile-derived edges, and a sequential within-episode convolution with a reset
boundary that no per-row primitive expresses. Either the language grows a dozen constructs or the
escape hatch becomes the normal path, and an abstraction everybody bypasses is worse than none.

So the driver owns the parts that have gone wrong before, and hands the study a frame it can trust.
The distinction matters and is deliberate:

    ASSERTS    shard alignment, seed contiguity, the step-count cross-check. These are properties of
               the data. They cannot be bypassed and they cannot move a number - they either pass or
               they fire.
    PROVIDES   `is_step`, `prev`, `initial`, `present`, `active`. These are CONVENTIONS. A callback
               is free to ignore them and index the raw arrays, so the guarantee they offer is
               auditable, not structural. `frame.raw()` is the deliberately conspicuous way to do
               that, so a breach is one grep away.
"""
from __future__ import annotations
import time

import numpy as np
import pyarrow.parquet as pq


class Frame:
    """One shard, validated, with the conventions this analysis has repeatedly got wrong built in."""

    def __init__(self, table, store, path: str):
        self._tb = table
        self._store = store
        self.path = path

        self.t = table.column("t").to_numpy()
        self.n = len(self.t)
        seed = table.column("episode_seed").to_numpy()
        self.episode_id = seed - store.seed0            # global episode index, per ROW

        # Episode boundaries. `t == 0` is the reset row: the world as handed to the agent, before it
        # has acted. It is not a step, and counting it as one inflates every denominator.
        self.initial = np.flatnonzero(self.t == 0)      # row indices of the reset rows
        self.episodes = self.episode_id[self.initial]   # global index of each episode in this shard
        if not np.array_equal(self.episodes,
                              np.arange(self.episodes[0], self.episodes[0] + len(self.episodes))):
            raise SystemExit(f"{path}: episodes are not shard-aligned - an episode's rows are split "
                             f"across shard files, so any per-episode sum here is partial")
        ends = np.append(self.initial[1:], self.n)
        self.estart = np.repeat(self.initial, ends - self.initial)   # first row of MY episode

        row = np.arange(self.n)
        self.is_step = row > self.estart                # excludes every reset row
        self.prev = row[self.is_step] - 1               # the row the action was chosen on
        self.step_rows = row[self.is_step]
        self.episode_of_step = self.episode_id[self.is_step]
        self.steps_per_episode = np.diff(np.append(self.initial, self.n)) - 1

    def raw(self, name: str, dtype=np.float64) -> np.ndarray:
        """The column as stored, with no convention applied. Deliberately conspicuous."""
        return self._tb.column(name).to_numpy(zero_copy_only=False).astype(dtype)

    def list_raw(self, name: str, width: int) -> np.ndarray:
        from env import listcol
        return listcol(self._tb.column(name), width)

    def at_initial(self, values: np.ndarray) -> np.ndarray:
        """Per-episode values read off the reset rows of this shard."""
        return values[self.initial]

    def per_episode_sum(self, values: np.ndarray) -> np.ndarray:
        """Sum within each episode in this shard, EXCLUDING the reset row.

        This is the right form for anything counted per STEP - bush occupancy, for instance -
        because the reset row is the world as handed to the agent, not something it did.
        """
        return np.add.reduceat(values, self.initial) - values[self.initial]

    def per_episode_sum_with_initial(self, values: np.ndarray) -> np.ndarray:
        """Sum within each episode INCLUDING the reset row.

        Both forms are offered because the existing sweeps genuinely use both, and the difference
        is not an oversight: a column like `damage` or `ate_food` records what the environment
        applied on arriving at a row, so its reset-row entry is a real zero rather than a
        miscount, and subtracting it would be arithmetic theatre. A port must keep whichever form
        its original used, which is why the choice is explicit at every call site rather than
        hidden in one default.
        """
        return np.add.reduceat(values, self.initial)


def sweep(store, columns: list[str], on_shard, init=None, verbose: bool = True, label: str = ""):
    """Walk every shard once, hand each to `on_shard`, and cross-check the population at the end.

    `on_shard(frame, acc, shard_index)` does the study's own accumulation in ordinary NumPy. The
    driver counts steps per episode itself and asserts the total against the episode table, so a
    callback that silently drops rows is caught even though the driver cannot see what it counted.
    """
    acc = init() if callable(init) else ({} if init is None else init)
    counted = np.zeros(store.n_episodes)
    t0 = time.time()
    for fi, path in enumerate(store.step_files):
        table = pq.read_table(path, columns=columns)
        frame = Frame(table, store, path)
        counted[frame.episodes] += frame.steps_per_episode
        on_shard(frame, acc, fi)
        if verbose and (fi % 15 == 14 or fi == len(store.step_files) - 1):
            print(f"  {label}: shard {fi+1}/{len(store.step_files)}  {time.time()-t0:5.0f}s",
                  flush=True)
    store.assert_step_counts(counted)
    return acc
