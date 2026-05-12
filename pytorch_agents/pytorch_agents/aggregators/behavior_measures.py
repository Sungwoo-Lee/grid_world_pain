"""Bridge between wrapper's terminal-info keys and sheeprl's MetricAggregator.

Sheeprl's MetricAggregator (sheeprl/utils/metric.py) expects a fixed key list
at instantiation.  Our per-tag Episode/* keys are dynamic — they come from the
env YAML at runtime.  This module provides:

  (a) ``register_dynamic_keys()`` — adds per-tag distance and behavior-measure
      MeanMetrics to an existing aggregator post-instantiation.  Idempotent.
  (b) ``update_from_final_info()`` — reads every ``Episode/*`` key off a
      final_info[i] dict and calls aggregator.update(...) for each present key.

Called from ``pytorch_agents.run_dreamer_v3`` at two hook points:
  GWP-PATCH-A: register_dynamic_keys() just after aggregator instantiation.
  GWP-PATCH-B: update_from_final_info() inside the "final_info in infos" block.
"""
from __future__ import annotations

from typing import Iterable

from torchmetrics import MeanMetric  # sheeprl already depends on torchmetrics

from src.behavior.accumulators import bm_wandb_keys
from src.behavior.distance_aggregator import dist_wandb_keys


def register_dynamic_keys(
    aggregator,
    neutral_tags: Iterable[str],
    predator_tags: Iterable[str],
    device: str = "cpu",
) -> None:
    """Add per-tag distance + behavior-measure MeanMetrics to the aggregator.

    Safe to call post-instantiation — MetricAggregator.add() does not lock
    the metrics dict after to().  Idempotent: skips keys already present.

    Args:
        aggregator:    sheeprl MetricAggregator instance.
        neutral_tags:  tuple/list of neutral entity tag strings (e.g. ('TL', 'BR')).
        predator_tags: tuple/list of predator entity tag strings.
        device:        torch device string to move new MeanMetric objects onto.
    """
    if aggregator is None or aggregator.disabled:
        return
    neutral_tags  = tuple(neutral_tags)
    predator_tags = tuple(predator_tags)
    all_keys = (
        dist_wandb_keys(neutral_tags, predator_tags)
        + bm_wandb_keys(predator_tags, neutral_tags)
    )
    for k in all_keys:
        if k not in aggregator.metrics:
            aggregator.add(k, MeanMetric().to(device))


def update_from_final_info(aggregator, agent_ep_info: dict) -> None:
    """Walk every ``Episode/*`` key on final_info[i] and update the aggregator.

    Silently skips any key not registered in the aggregator (avoids the
    MetricAggregatorException that would fire with raise_on_missing=True).

    Args:
        aggregator:    sheeprl MetricAggregator instance (may be None or disabled).
        agent_ep_info: one entry from infos["final_info"] — the terminal info
                       dict our wrapper placed Episode/* scalars onto.
    """
    if aggregator is None or aggregator.disabled:
        return
    for key, val in agent_ep_info.items():
        if isinstance(key, str) and key.startswith("Episode/"):
            if key in aggregator.metrics:
                aggregator.update(key, float(val))
