"""episode_logging.py — shared per-iteration episode-level WandB helpers.

Lifted verbatim from train.py:L1000-L1059 (_append_per_measure_mean,
_append_per_tag_means, _bm_log_wandb) so that both train.py and
dreamer_srl_main.py can import from a single source of truth rather than
copy-pasting the logic.

Commit 4 of MIGRATION_PLAN.md introduces this module;
Commit 5 adds bm_log_wandb.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _append_per_measure_mean(
    ep_log: dict,
    iteration_episodes: List[dict],
    ep_key_raw: str,
    wandb_key: str,
) -> None:
    """Mean the same per-episode raw scalar (skipping NaN) across the window.

    Lifted from train.py:L1000-L1006.
    """
    vals = [
        ep[ep_key_raw]
        for ep in iteration_episodes
        if ep_key_raw in ep
        and not (isinstance(ep[ep_key_raw], float) and ep[ep_key_raw] != ep[ep_key_raw])
    ]
    if vals:
        ep_log[wandb_key] = float(np.mean(vals))


def append_per_tag_means(
    ep_log: dict,
    iteration_episodes: List[dict],
    tags: Tuple[str, ...],
    ep_key_prefix: str,
    wandb_key_prefix: str,
) -> None:
    """Group ep_data raw per-tag scalars by tag, mean across instances + episodes.

    Writes ep_log[f'{wandb_key_prefix}_{tag}'] for each unique tag.

    Lifted from train.py:L1046-L1059.
    """
    for tag in sorted(set(tags)):
        matching = [j for j, t in enumerate(tags) if t == tag]
        per_ep = []
        for ep in iteration_episodes:
            vals = [
                ep[f'{ep_key_prefix}_{tags[j]}_raw']
                for j in matching
                if f'{ep_key_prefix}_{tags[j]}_raw' in ep
            ]
            if vals:
                per_ep.append(np.mean(vals))
        if per_ep:
            ep_log[f'{wandb_key_prefix}_{tag}'] = float(np.mean(per_ep))


def bm_log_wandb(
    ep_log: dict,
    iteration_episodes: List[dict],
    predator_tags: Tuple[str, ...],
    neutral_tags: Tuple[str, ...],
) -> None:
    """Append all BM WandB keys to ep_log from the iteration's episodes.

    Lifted from train.py:L1008-L1044.
    """
    for cname in ("predator", "rabbit"):
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"interrupted_feeding_rate_{cname}_raw",   f"Episode/InterruptedFeedingRate_{cname}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"interrupted_feeding_denom_{cname}_raw",  f"Episode/InterruptedFeedingDenominator_{cname}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"bush_dive_rate_{cname}_raw",             f"Episode/BushDiveRate_{cname}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"bush_dive_denom_{cname}_raw",            f"Episode/BushDiveDenominator_{cname}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_under_threat_ratio_{cname}_raw",     f"Episode/EatUnderThreatRatio_{cname}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_under_threat_rate_{cname}_raw",      f"Episode/EatUnderThreatRate_{cname}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_safe_rate_{cname}_raw",              f"Episode/EatSafeRate_{cname}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_under_threat_safe_steps_{cname}_raw", f"Episode/EatUnderThreatSafeSteps_{cname}")
    for tag in predator_tags:
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"interrupted_feeding_rate_predator_{tag}_raw", f"Episode/InterruptedFeedingRate_predator_{tag}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"bush_dive_rate_predator_{tag}_raw",           f"Episode/BushDiveRate_predator_{tag}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_under_threat_ratio_predator_{tag}_raw",   f"Episode/EatUnderThreatRatio_predator_{tag}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_under_threat_rate_predator_{tag}_raw",    f"Episode/EatUnderThreatRate_predator_{tag}")
    for tag in neutral_tags:
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"interrupted_feeding_rate_rabbit_{tag}_raw",   f"Episode/InterruptedFeedingRate_rabbit_{tag}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"bush_dive_rate_rabbit_{tag}_raw",             f"Episode/BushDiveRate_rabbit_{tag}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_under_threat_ratio_rabbit_{tag}_raw",     f"Episode/EatUnderThreatRatio_rabbit_{tag}")
        _append_per_measure_mean(ep_log, iteration_episodes,
            f"eat_under_threat_rate_rabbit_{tag}_raw",      f"Episode/EatUnderThreatRate_rabbit_{tag}")
