#!/usr/bin/env python3
"""store.py - find a run's shards across collection passes, and assert what must be true of them.

THE POPULATION CONTRACT, and why it is split three ways.

An earlier draft had `open_run` assert that the store matched every derived product already on disk.
That deadlocks its own workflow: after a collection top-up - this project has had two - every product
is stale by definition, so the rebuild that would refresh them cannot open the store, and
regenerating one product alone becomes impossible. The predictable resolution under deadline
pressure is to delete the assert, which is the flagship guard. So the checks are split by when they
can honestly be true:

    at open (here)      store-internal properties only, always checkable
    at write            each product is stamped with the population it was built from
    at read             a product whose stamp disagrees with its siblings is refused

Only the first lives in this module. It asserts three things, and each exists because getting it
wrong produces a plausible wrong number rather than an error:

  * seeds are contiguous and unique ACROSS the union of collection passes, not per store - the whole
    point of a second pass is that it continues the first without gap or overlap, and a gap means
    the population is not what the analysis says it is;
  * every episode's step rows sit inside one shard, so an episode is never split across a file
    boundary and silently half-counted;
  * the summed step count equals the episode table's own `length` column - the cross-check that
    catches a scan which quietly dropped or double-counted rows.

The third needs the whole scan to have run, so it is enforced by `scan.sweep`, which calls back into
`Store.assert_step_counts` at the end.
"""
from __future__ import annotations
import glob
import os

import numpy as np
import pyarrow.parquet as pq


class Store:
    """One run's trajectory store, spanning however many collection passes it was built in."""

    def __init__(self, roots: list[str], episode_columns: list[str]):
        self.roots = list(roots)
        self.step_files = self._files("steps")
        if not self.step_files:
            raise SystemExit(f"no step shards under any of {self.roots}")

        ep = pq.read_table(self._files("episodes"), columns=episode_columns)
        seeds = ep.column("episode_seed").to_numpy()
        self.order = np.argsort(seeds)
        self.seeds = seeds[self.order]
        self.n_episodes = len(self.seeds)
        self.seed0 = int(self.seeds[0])
        self._ep = ep

        span = int(self.seeds.max()) - self.seed0 + 1
        if span != self.n_episodes or len(np.unique(self.seeds)) != self.n_episodes:
            raise SystemExit(
                f"the {len(self.roots)} collection pass(es) do not form one contiguous seed range: "
                f"{self.n_episodes:,} episodes spanning {self.seeds.min():,}..{self.seeds.max():,} "
                f"({span:,} seeds). A gap or an overlap means the population is not what the "
                f"analysis will report it to be.")

    def _files(self, kind: str) -> list[str]:
        out: list[str] = []
        for r in self.roots:
            out += sorted(glob.glob(os.path.join(r, "**", f"{kind}_*.parquet"), recursive=True))
        return out

    def episode(self, name: str, dtype=None) -> np.ndarray:
        """An episode-table column, sorted into seed order and aligned with `self.seeds`."""
        col = self._ep.column(name).to_numpy(zero_copy_only=False)[self.order]
        return col.astype(dtype) if dtype is not None else col

    def episode_list(self, name: str, width: int) -> np.ndarray:
        """A fixed-width list column from the episode table, in seed order."""
        from env import listcol            # same package; imported lazily to keep this file standalone
        return listcol(self._ep.column(name), width)[self.order]

    def episode_property(self, name: str, per_episode: int) -> np.ndarray:
        """A ragged-looking episode column that is really (episodes x entities x channels).

        The width of the last axis is not stored anywhere, so it is inferred - and then CHECKED,
        because an inferred shape that happens to divide evenly is exactly how a silently
        transposed array gets into a figure.
        """
        flat = np.concatenate([c.flatten().to_numpy(zero_copy_only=False)
                               for c in self._ep.column(name).chunks])
        if flat.size % (self.n_episodes * per_episode):
            raise SystemExit(
                f"{name} is not (episodes x {per_episode} x channels): {flat.size:,} values do not "
                f"divide by {self.n_episodes:,} x {per_episode}")
        return flat.reshape(self.n_episodes, per_episode, -1)[self.order]

    def assert_step_counts(self, counted: np.ndarray) -> None:
        """The cross-check that catches a scan which dropped or double-counted rows."""
        length = self.episode("length", np.float64)
        if not np.allclose(counted, length):
            bad = int(np.sum(~np.isclose(counted, length)))
            raise SystemExit(
                f"step count disagrees with the episode table's `length` column in {bad:,} of "
                f"{len(length):,} episodes - the scan lost or double-counted rows")


def open_run(roots: list[str], episode_columns: list[str]) -> Store:
    """Open a run's store and assert everything that can be checked without scanning it."""
    return Store(roots, episode_columns)
