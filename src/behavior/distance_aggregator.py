"""Per-tag distance running-mean aggregator.

Extracted from ``train.py`` (lines 1285-1298 plus per-episode
distance accumulation logic).  Pure numpy — no JAX, no PyTorch.

Both the JAX trainer and the sheeprl bridge import from here.

API:
  DistState            — running sum + step count arrays.
  make_dist_state()    — factory (zeroed).
  dist_step_update()   — accumulate per-step distances.
  dist_reset_env()     — episode-end reset for one env slot.
  dist_finalise_episode() — returns per-episode mean-distance dict.
  dist_wandb_keys()    — enumerates all WandB keys emitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class DistState:
    """Running sums and step counts for per-tag distance metrics.

    ``num_envs``      — vectorized env width (use 1 for sheeprl).
    ``num_predator``  — number of predator entities tracked.
    ``num_neutral``   — number of neutral entities tracked.
    """

    num_envs: int
    num_predator: int
    num_neutral: int

    sum_food:    np.ndarray = field(init=False)  # [num_envs]
    sum_pred:    np.ndarray = field(init=False)
    sum_neutral: np.ndarray = field(init=False)
    sum_hide:    np.ndarray = field(init=False)
    sum_per_pred:    np.ndarray = field(init=False)  # [num_envs, num_predator]
    sum_per_neutral: np.ndarray = field(init=False)  # [num_envs, num_neutral]
    step_count:  np.ndarray = field(init=False)  # [num_envs]

    def __post_init__(self):
        ne = self.num_envs
        self.sum_food    = np.zeros(ne, dtype=np.float32)
        self.sum_pred    = np.zeros(ne, dtype=np.float32)
        self.sum_neutral = np.zeros(ne, dtype=np.float32)
        self.sum_hide    = np.zeros(ne, dtype=np.float32)
        self.sum_per_pred    = np.zeros((ne, self.num_predator), dtype=np.float32)
        self.sum_per_neutral = np.zeros((ne, self.num_neutral),  dtype=np.float32)
        self.step_count  = np.zeros(ne, dtype=np.int64)


def make_dist_state(num_envs: int, num_predator: int, num_neutral: int) -> DistState:
    """Factory — returns a zero-initialised DistState."""
    return DistState(num_envs=num_envs, num_predator=num_predator, num_neutral=num_neutral)


def dist_step_update(dist: DistState, info_np_t: dict) -> None:
    """Accumulate per-step distances from ``info_np_t``.

    Expected keys:
      ``dist_to_food``, ``dist_to_pred``, ``dist_to_neutral``,
      ``dist_to_hiding_predator`` — shape [num_envs].
      ``dist_per_predator``       — shape [num_envs, num_predator] (optional).
      ``dist_per_neutral``        — shape [num_envs, num_neutral]  (optional).
    """
    dist.sum_food    += np.asarray(info_np_t['dist_to_food'],            dtype=np.float32)
    dist.sum_pred    += np.asarray(info_np_t['dist_to_pred'],            dtype=np.float32)
    dist.sum_neutral += np.asarray(info_np_t['dist_to_neutral'],         dtype=np.float32)
    dist.sum_hide    += np.asarray(info_np_t['dist_to_hiding_predator'], dtype=np.float32)
    dpp = info_np_t.get('dist_per_predator')
    dpn = info_np_t.get('dist_per_neutral')
    if dpp is not None and dpp.ndim >= 2 and dpp.shape[-1] > 0:
        dist.sum_per_pred    += np.asarray(dpp, dtype=np.float32)
    if dpn is not None and dpn.ndim >= 2 and dpn.shape[-1] > 0:
        dist.sum_per_neutral += np.asarray(dpn, dtype=np.float32)
    dist.step_count += 1


def dist_reset_env(dist: DistState, i: int) -> None:
    """Reset distance accumulators for env slot *i* at episode end."""
    dist.sum_food[i]    = 0.0
    dist.sum_pred[i]    = 0.0
    dist.sum_neutral[i] = 0.0
    dist.sum_hide[i]    = 0.0
    dist.sum_per_pred[i, :]    = 0.0
    dist.sum_per_neutral[i, :] = 0.0
    dist.step_count[i]  = 0


def dist_finalise_episode(
    dist: DistState,
    i: int,
    neutral_tags: Tuple[str, ...],
    predator_tags: Tuple[str, ...],
) -> dict:
    """Return per-episode mean-distance dict for env slot *i*.

    Keys match the JAX-side WandB surface:
      ``Episode/MeanDistFood``, ``Episode/MeanDistPredator``,
      ``Episode/MeanDistRabbit``, ``Episode/MeanDistHidingPredator``,
      ``Episode/MeanDistPredator_<tag>``, ``Episode/MeanDistRabbit_<tag>``.
    """
    n = max(int(dist.step_count[i]), 1)
    out = {
        "Episode/MeanDistFood":           float(dist.sum_food[i]    / n),
        "Episode/MeanDistPredator":       float(dist.sum_pred[i]    / n),
        "Episode/MeanDistRabbit":         float(dist.sum_neutral[i] / n),
        "Episode/MeanDistHidingPredator": float(dist.sum_hide[i]    / n),
    }
    for j, tag in enumerate(predator_tags):
        if j < dist.num_predator:
            out[f"Episode/MeanDistPredator_{tag}"] = float(dist.sum_per_pred[i, j] / n)
    for j, tag in enumerate(neutral_tags):
        if j < dist.num_neutral:
            out[f"Episode/MeanDistRabbit_{tag}"] = float(dist.sum_per_neutral[i, j] / n)
    return out


def dist_wandb_keys(
    neutral_tags: Tuple[str, ...],
    predator_tags: Tuple[str, ...],
) -> list:
    """Enumerate all WandB keys the distance aggregator emits.

    Used by the sheeprl aggregator to register MeanMetrics at startup.
    """
    keys = [
        "Episode/MeanDistFood",
        "Episode/MeanDistPredator",
        "Episode/MeanDistRabbit",
        "Episode/MeanDistHidingPredator",
    ]
    for tag in predator_tags:
        keys.append(f"Episode/MeanDistPredator_{tag}")
    for tag in neutral_tags:
        keys.append(f"Episode/MeanDistRabbit_{tag}")
    return keys
