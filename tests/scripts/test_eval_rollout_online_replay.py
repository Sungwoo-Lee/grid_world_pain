"""Regression test — offline replay M1 (interrupted-feeding) rate must be able
to fire.

Bug under test (H9, docs/develop/active/issues/diag_fable5_20260704/
07_behavior_measures.md Finding 3 + fix_plan_h8h9_eval_output_correctness.md):
`_compute_online_replay` in scripts/eval/eval_rollout.py resolved a pending M1
candidate BEFORE updating `steps_since_eat` for the current step. A candidate
is recorded at an eat step (counter 0); at resolution (K steps later) the
pre-update value read is at most K-1, so `steps_since_eat >= K` was
unsatisfiable — `interrupted_feeding_rate_*` was structurally 0.0 for every
agent whenever candidates existed.

Fix under test: reorder to the online accumulator's ordering (update
`steps_since_eat` first, then age/resolve, then record — src/behavior/
accumulators.py:211/215-232/234-257, fix 3e1e53e), so "interrupted" means: the
agent did not eat at any of the K steps following the candidate eat.

NOTE on expected numbers (hand-traced against the fixed code, per plan):
the replay tracks a SINGLE pending candidate per class; a new candidate
overwrites an unresolved one, and the offline denominator is counted at
RESOLUTION time (known open divergence vs online record-time counting —
07_behavior_measures.md Finding 4, deliberately NOT changed here). So an
overwritten candidate never enters the offline denominator.
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "eval"))

import eval_rollout as er  # noqa: E402


def _episode(T, eat_steps, dist=1.0):
    """Synthetic episode dict in the shape _compute_online_replay consumes.

    `dist` may be a scalar (constant predator distance) or a length-T sequence
    (time-varying distance; cue_radius below is 3.0, so <3 means 'in radius').
    """
    if np.isscalar(dist):
        dist_arr = np.full((T, 1), dist, dtype=np.float32)
    else:
        dist_arr = np.asarray(dist, dtype=np.float32).reshape(T, 1)
    return {
        "length": T,
        "dist_per_predator": dist_arr,
        "dist_per_neutral": np.zeros((T, 0), dtype=np.float32),  # no neutrals
        "ate_food": np.isin(np.arange(T), eat_steps),
        "agent_in_bush": np.zeros(T, dtype=bool),
    }


def _bm_cfg():
    return SimpleNamespace(cue_radius=3.0, obs_window=5)


def test_interrupted_feeding_fires():
    """Agent eats once under threat at t=2, threat stays in radius, agent never
    eats again -> the K=5 window after the eat contains no eat -> interrupted.
    Pre-fix: rate structurally 0.0."""
    res = er._compute_online_replay([_episode(T=20, eat_steps=[2])], _bm_cfg())
    assert res["m1_candidates_predator"] == 1
    assert res["m1_interrupted_predator"] == 1              # pre-fix: 0
    assert res["interrupted_feeding_rate_predator"] == 1.0  # pre-fix: 0.0
    # No neutral animals -> no rabbit candidates -> NaN rate.
    assert math.isnan(res["interrupted_feeding_rate_rabbit"])


def test_not_interrupted_when_agent_eats_within_window():
    """Negative control (guards against over-fixing): candidate eat at t=2
    under threat; the predator then LEAVES the radius, and the agent eats again
    at t=5 — inside the 5-step window but out of radius, so no new candidate is
    recorded and the counter resets. The candidate resolves not-interrupted."""
    dist = [1.0, 1.0, 1.0] + [10.0] * 17  # in radius through t=2, then gone
    res = er._compute_online_replay(
        [_episode(T=20, eat_steps=[2, 5], dist=dist)], _bm_cfg())
    assert res["m1_candidates_predator"] == 1
    assert res["m1_interrupted_predator"] == 0
    assert res["interrupted_feeding_rate_predator"] == 0.0


def test_second_eat_under_threat_overwrites_candidate():
    """Two eats under sustained threat (t=2 and t=5): the t=5 eat overwrites
    the unresolved t=2 candidate (single-slot tracker) and — offline
    denominator counted at resolution time (fenced divergence, Finding 4) —
    the overwritten candidate is never counted. The surviving candidate's
    window (t=6..10) has no eat -> interrupted. Pre-fix: rate 0.0."""
    res = er._compute_online_replay([_episode(T=20, eat_steps=[2, 5])], _bm_cfg())
    assert res["m1_candidates_predator"] == 1
    assert res["m1_interrupted_predator"] == 1              # pre-fix: 0
    assert res["interrupted_feeding_rate_predator"] == 1.0  # pre-fix: 0.0
