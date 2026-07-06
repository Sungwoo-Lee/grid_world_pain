"""H10 regression tests — DreamerV3 batch BM driver (``bm_drive_batch``).

The train.py DreamerV3 branch ("Site 2") collects whole [T, B] batches and
drives the online behavior-measure (M1/M2/M5) state machine from them. The
pre-fix ordering ran the per-step update over ALL T steps first and only then
finalised/reset the episodes that ended mid-batch, so a finished episode
absorbed post-death steps, the next episode lost its opening steps, and a
double-done env slot produced an empty all-NaN second finalisation.

These tests drive ``bm_drive_batch`` with synthetic [T, B] arrays (no env
needed) and assert on the unambiguous M5-family step-counting keys only
(``eat_under_threat_rate_predator_raw``, ``eat_safe_rate_predator_raw``,
``eat_under_threat_safe_steps_predator_raw``) plus NaN-ness. They deliberately
do NOT assert on how a pending M1 candidate / M2 onset resolves at episode end
— that end-of-episode rule is a known-undecided user decision (B1).

Fix plan: docs/develop/active/issues/diag_fable5_20260704/
fix_plan_h10_dreamer_batch_bm.md

Run with:
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
      -m pytest tests/environment/test_bm_dreamer_batch_driver.py -v
"""
import math
import os
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.behavior.accumulators import (
    make_bm_state,
    bm_step_update,
    bm_reset_env,
    bm_finalise_episode,
    bm_drive_batch,
)

# ---------------------------------------------------------------------------
# Common fixture parameters
# ---------------------------------------------------------------------------
NEAR = 1.0   # < bm_R=2.0 → threat in radius
FAR = 5.0    # > bm_R=2.0 → safe
PRED_TAGS = ("wolf",)
NEUT_TAGS = ()


def _fresh_bm(num_envs=2):
    return make_bm_state(
        num_envs=num_envs,
        num_predator_tags=1,
        num_neutral_tags=0,
        bm_R=2.0,
        bm_K=3,
    )


def _quiet_env1_arrays(T):
    """Env 1 is the quiet control: never done, threat always far, never eats.

    Returns per-env-1 columns to be composed into [T, 2] / [T, 2, 1] arrays.
    """
    ate = np.zeros(T, dtype=bool)
    bush = np.zeros(T, dtype=bool)
    dist = np.full(T, FAR)
    done = np.zeros(T, dtype=bool)
    return ate, bush, dist, done


def _compose(T, env0_ate, env0_dist, env0_done):
    """Build the [T, B=2] batch arrays with env 1 as the quiet control."""
    e1_ate, e1_bush, e1_dist, e1_done = _quiet_env1_arrays(T)
    ate_food = np.stack([env0_ate, e1_ate], axis=1)                      # [T, 2]
    in_bush = np.zeros((T, 2), dtype=bool)                               # [T, 2]
    dist_pred = np.stack([env0_dist, e1_dist], axis=1)[..., None]        # [T, 2, 1]
    done = np.stack([env0_done, e1_done], axis=1)                        # [T, 2]
    return ate_food, in_bush, dist_pred, done


def _assert_env1_control(bm, T_total):
    """Env 1's counters (finalised manually) must match its own quiet script
    and be independent of env 0's dones."""
    ep1 = bm_finalise_episode(bm, 1, PRED_TAGS, NEUT_TAGS)
    assert ep1["eat_under_threat_safe_steps_predator_raw"] == T_total
    assert ep1["eat_safe_rate_predator_raw"] == 0.0
    assert math.isnan(ep1["eat_under_threat_rate_predator_raw"])


def _mid_batch_death_arrays():
    """T=12, env 0 done at t=5 only.

    Episode 1 (t=0..5): threat far all 6 steps, eat at t=2.
    Post-death steps (t=6..11): threat near all 6 steps, eat at t=8.
    """
    T = 12
    env0_ate = np.zeros(T, dtype=bool)
    env0_ate[2] = True
    env0_ate[8] = True
    env0_dist = np.full(T, FAR)
    env0_dist[6:] = NEAR
    env0_done = np.zeros(T, dtype=bool)
    env0_done[5] = True
    return (T,) + _compose(T, env0_ate, env0_dist, env0_done)


def _double_done_arrays():
    """T=12, env 0 done at t=3 AND t=9.

    Segment 1 (t=0..3): threat near all 4 steps, eat at t=1.
    Segment 2 (t=4..9): threat far all 6 steps, eat at t=6.
    Trailing steps (t=10..11): quiet (far, no eat) — episode 3 in progress.
    """
    T = 12
    env0_ate = np.zeros(T, dtype=bool)
    env0_ate[1] = True
    env0_ate[6] = True
    env0_dist = np.full(T, FAR)
    env0_dist[0:4] = NEAR
    env0_done = np.zeros(T, dtype=bool)
    env0_done[3] = True
    env0_done[9] = True
    return (T,) + _compose(T, env0_ate, env0_dist, env0_done)


# ---------------------------------------------------------------------------
# Scenario tests (red pre-fix)
# ---------------------------------------------------------------------------

def test_mid_batch_death_no_contamination():
    """The episode finished at t=5 must not absorb post-death steps t=6..11.

    Pre-fix failure mode: all 12 steps land in the state before finalise →
    eat_under_threat_rate == 1/6 (not NaN).
    """
    T, ate_food, in_bush, dist_pred, done = _mid_batch_death_arrays()
    bm = _fresh_bm()
    results = bm_drive_batch(
        bm,
        ate_food_steps=ate_food,
        agent_in_bush_steps=in_bush,
        dist_per_predator_steps=dist_pred,
        dist_per_neutral_steps=None,
        done_steps=done,
        predator_tags=PRED_TAGS,
        neutral_tags=NEUT_TAGS,
    )
    assert (5, 0) in results
    ep = results[(5, 0)]
    # Episode 1: 6 safe steps, 1 safe eat, ZERO threat steps.
    assert ep["eat_under_threat_safe_steps_predator_raw"] == 6
    assert ep["eat_safe_rate_predator_raw"] == 1 / 6
    assert math.isnan(ep["eat_under_threat_rate_predator_raw"])
    _assert_env1_control(bm, T)


def test_new_episode_opening_steps_counted():
    """The episode that starts mid-batch (t=6..11) must keep its opening steps.

    Pre-fix failure mode: state was wiped after the batch-end finalise →
    all-NaN / zeros for the still-open episode 2.
    """
    T, ate_food, in_bush, dist_pred, done = _mid_batch_death_arrays()
    bm = _fresh_bm()
    bm_drive_batch(
        bm,
        ate_food_steps=ate_food,
        agent_in_bush_steps=in_bush,
        dist_per_predator_steps=dist_pred,
        dist_per_neutral_steps=None,
        done_steps=done,
        predator_tags=PRED_TAGS,
        neutral_tags=NEUT_TAGS,
    )
    # Episode 2 (t=6..11) is still open: 6 threat steps, 1 threat eat.
    ep2 = bm_finalise_episode(bm, 0, PRED_TAGS, NEUT_TAGS)
    assert ep2["eat_under_threat_rate_predator_raw"] == 1 / 6
    assert ep2["eat_under_threat_safe_steps_predator_raw"] == 0
    _assert_env1_control(bm, T)


def test_double_done_two_valid_finalisations():
    """Two dones for the same env slot in one batch → two VALID finalisations.

    Pre-fix failure mode: first finalise mixes all 12 steps; second finalise
    runs on a fully-reset state → safe_steps 0, every rate NaN.
    """
    T, ate_food, in_bush, dist_pred, done = _double_done_arrays()
    bm = _fresh_bm()
    results = bm_drive_batch(
        bm,
        ate_food_steps=ate_food,
        agent_in_bush_steps=in_bush,
        dist_per_predator_steps=dist_pred,
        dist_per_neutral_steps=None,
        done_steps=done,
        predator_tags=PRED_TAGS,
        neutral_tags=NEUT_TAGS,
    )
    assert set(results) == {(3, 0), (9, 0)}
    # Segment 1: 4 threat steps, 1 threat eat.
    assert results[(3, 0)]["eat_under_threat_rate_predator_raw"] == 1 / 4
    # Segment 2: 6 safe steps, 1 safe eat — NOT the empty all-NaN finalise.
    assert results[(9, 0)]["eat_safe_rate_predator_raw"] == 1 / 6
    assert results[(9, 0)]["eat_under_threat_safe_steps_predator_raw"] == 6
    _assert_env1_control(bm, T)


# ---------------------------------------------------------------------------
# Non-scenario tests
# ---------------------------------------------------------------------------

def test_no_done_batch_accumulates():
    """No dones across two consecutive batches → no finalisations, no resets;
    a manual finalise afterwards reflects all 12 steps (guards against
    over-eager resets)."""
    T = 6
    env0_ate = np.zeros(T, dtype=bool)
    env0_ate[2] = True
    env0_dist = np.full(T, FAR)
    env0_done = np.zeros(T, dtype=bool)
    ate_food, in_bush, dist_pred, done = _compose(T, env0_ate, env0_dist, env0_done)

    bm = _fresh_bm()
    for _ in range(2):
        results = bm_drive_batch(
            bm,
            ate_food_steps=ate_food,
            agent_in_bush_steps=in_bush,
            dist_per_predator_steps=dist_pred,
            dist_per_neutral_steps=None,
            done_steps=done,
            predator_tags=PRED_TAGS,
            neutral_tags=NEUT_TAGS,
        )
        assert results == {}

    ep = bm_finalise_episode(bm, 0, PRED_TAGS, NEUT_TAGS)
    # 12 safe steps total, 2 safe eats, zero threat steps.
    assert ep["eat_under_threat_safe_steps_predator_raw"] == 12
    assert ep["eat_safe_rate_predator_raw"] == 2 / 12
    assert math.isnan(ep["eat_under_threat_rate_predator_raw"])
    _assert_env1_control(bm, 12)


def _nan_equal(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_matches_interleaved_reference_pattern():
    """Consistency pin: bm_drive_batch must equal a hand-rolled per-step loop
    mirroring the rPPO Site-1 call pattern verbatim (bm_step_update → per-done
    bm_finalise_episode + bm_reset_env), key-for-key (NaN == NaN)."""
    T, ate_food, in_bush, dist_pred, done = _double_done_arrays()

    # Driver under test.
    bm_a = _fresh_bm()
    results_a = bm_drive_batch(
        bm_a,
        ate_food_steps=ate_food,
        agent_in_bush_steps=in_bush,
        dist_per_predator_steps=dist_pred,
        dist_per_neutral_steps=None,
        done_steps=done,
        predator_tags=PRED_TAGS,
        neutral_tags=NEUT_TAGS,
    )

    # Independent reference: rPPO Site-1 interleaved pattern.
    bm_b = _fresh_bm()
    results_b = {}
    for t in range(T):
        info_t = {
            "ate_food": ate_food[t].astype(bool),
            "agent_in_bush": in_bush[t].astype(bool),
            "dist_per_predator": dist_pred[t],
        }
        done_t = done[t].astype(bool)
        bm_step_update(bm_b, info_t, done_t)
        for i in np.where(done_t)[0]:
            i = int(i)
            results_b[(t, i)] = bm_finalise_episode(bm_b, i, PRED_TAGS, NEUT_TAGS)
            bm_reset_env(bm_b, i)

    assert set(results_a) == set(results_b)
    for key in results_b:
        ep_a, ep_b = results_a[key], results_b[key]
        assert set(ep_a) == set(ep_b), f"key set mismatch at {key}"
        for k in ep_b:
            assert _nan_equal(ep_a[k], ep_b[k]), (
                f"mismatch at {key} / {k}: {ep_a[k]!r} != {ep_b[k]!r}"
            )
