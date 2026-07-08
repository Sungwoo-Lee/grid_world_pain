"""WP-SRL P5 regression test — episode-counter double-count at done boundaries.

The 2026-07-06 re-audit ([[00_master_comparison]] §3 P5, area report 04 D-06)
found that the driver's unconditional per-iteration increment

    episode_lengths += 1
    episode_rewards += rewards

ran AFTER the done-block already (a) logged the finished episode as
`ep_len = counters + 1`, `ep_rew = counters + rewards[i]` and (b) zeroed the
counters. The increment then re-counted the terminal transition into the
SUCCESSOR episode's counters: every logged episode after the first carried
+1 survival step (the project's headline metric) and the PREDECESSOR episode's
terminal reward.

sheeprl needs no such counters — it reads the gym wrapper's
`final_info["episode"]` (dreamer_v3.py:610-618).

Fix: module-level helper `_advance_episode_counters(...)` advances the
counters only for envs NOT done this iteration; the driver calls it in place
of the unconditional increment. The done-block logging (`counters + 1` /
`counters + rewards[i]`) and zeroing stay unchanged — with the leak removed
they are exactly right.

Red evidence (pre-fix): fails at collection with ImportError —
`_advance_episode_counters` does not exist ("red by absence"; the buggy
increment was inline in the driver loop and not importable). The two-episode
fixture below is the discriminator: under the old unconditional increment,
episode 2 logs (l=3, r=22.0) instead of the true (l=2, r=12.0).

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
        -m pytest tests/algorithms/dreamer_srl/test_episode_metrics.py -v
"""
from __future__ import annotations

import numpy as np

from src.algorithms.dreamer_srl.dreamer_srl_main import _advance_episode_counters


def test_second_episode_counters_unbiased() -> None:
    """Two back-to-back episodes: BOTH logged (length, reward) pairs are exact.

    Scripted single-env trace replaying the driver's exact order per iteration
    (dreamer_srl_main.py: done-block logs `l = counters + 1`,
    `r = counters + rewards[i]`, then zeroes done-env counters, THEN the
    counters advance for the next iteration):

      episode 1: 3 steps, rewards [1, 2, 10]  -> logged (3, 13.0)
      episode 2: 2 steps, rewards [5, 7]      -> logged (2, 12.0)

    Old-code values (unconditional `episode_lengths += 1;
    episode_rewards += rewards` after the done block): episode 1 logs
    (3, 13.0) correctly, but episode 2 logs (3, 22.0) — the terminal step of
    episode 1 leaks +1 length and its terminal reward (10.0) into episode 2's
    counters.
    """
    num_envs = 1
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)

    # (reward, done) per iteration for env 0.
    trace = [
        (1.0, False), (2.0, False), (10.0, True),   # episode 1
        (5.0, False), (7.0, True),                   # episode 2
    ]

    logged: list[tuple[int, float]] = []
    for reward, done in trace:
        rewards = np.array([reward], dtype=np.float32)
        dones = np.array([done], dtype=bool)

        # Done block — mirrors dreamer_srl_main.py logging + zeroing order.
        dones_idxes = list(np.where(dones)[0])
        for i in dones_idxes:
            ep_len = int(episode_lengths[i]) + 1
            ep_rew = float(episode_rewards[i]) + float(rewards[i])
            logged.append((ep_len, ep_rew))
        episode_lengths[dones_idxes] = 0
        episode_rewards[dones_idxes] = 0.0

        # Counter advance — the WP-SRL P5 helper under test.
        _advance_episode_counters(episode_lengths, episode_rewards, rewards, dones)

    assert logged[0] == (3, 13.0), (
        f"Episode 1 logged {logged[0]}, expected (3, 13.0). Episode 1 is "
        "unaffected by the P5 leak — if this fails, the fixture ordering "
        "itself has drifted from the driver."
    )
    assert logged[1] == (2, 12.0), (
        f"Episode 2 logged {logged[1]}, expected (2, 12.0). Old-code value was "
        "(3, 22.0): +1 step and +10.0 terminal reward leaked from episode 1 "
        "(WP-SRL P5 double-count regression)."
    )


def test_advance_skips_only_done_envs() -> None:
    """Multi-env: alive envs advance, done envs stay at their post-reset zeros."""
    episode_lengths = np.array([4, 0, 7], dtype=np.int32)   # env 1 just reset
    episode_rewards = np.array([2.5, 0.0, -1.0], dtype=np.float32)
    rewards = np.array([1.0, 9.0, 0.5], dtype=np.float32)
    dones = np.array([False, True, False])

    _advance_episode_counters(episode_lengths, episode_rewards, rewards, dones)

    np.testing.assert_array_equal(episode_lengths, np.array([5, 0, 8], dtype=np.int32))
    np.testing.assert_allclose(episode_rewards, np.array([3.5, 0.0, -0.5], dtype=np.float32))


def test_stage_swap_iteration_skips_advance() -> None:
    """C1 (review_srl_parity_fixes.md): swap-iteration advance must be skipped.

    Curriculum stage-swap edge of the P5 fix: on a stage-transition iteration
    the swap block (dreamer_srl_main.py step 4, ~:1436) wipes ALL envs'
    episode counters — partial episodes of NON-done envs are deliberately
    dropped. `_advance_episode_counters` then runs at the end of the same
    iteration (~:1592) with the PRE-swap `rewards`/`dones`; unguarded, it
    credits the pre-swap step's +1 length and reward to the non-done envs'
    freshly wiped counters, i.e. into the NEW stage's first episode.

    Fix: the driver tracks `_stage_swapped_this_iter` and passes it as
    `stage_swapped=`; the helper no-ops when True, so all counters stay at
    the swap block's zeros and every env starts the new stage clean —
    consistent with the behavior/dist accumulators, which are wiped at the
    same point and receive no post-wipe credit.

    Red evidence (pre-fix): TypeError — `_advance_episode_counters` had no
    `stage_swapped` parameter (the advance at :1592 was unconditional).
    Old-code behavioral value: env 1 (alive through the swap) ends the swap
    iteration with length=1, reward=3.0 instead of 0/0.0.
    """
    num_envs = 2
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)

    # Iteration 1 — ordinary step, no dones, no swap.
    rewards = np.array([1.0, 2.0], dtype=np.float32)
    dones = np.array([False, False])
    _advance_episode_counters(episode_lengths, episode_rewards, rewards, dones,
                              stage_swapped=False)
    np.testing.assert_array_equal(episode_lengths, np.array([1, 1], dtype=np.int32))
    np.testing.assert_allclose(episode_rewards, np.array([1.0, 2.0], dtype=np.float32))

    # Iteration 2 — env 0 finishes (logged by the done block), completing the
    # episode quota for the current stage -> stage swap fires this iteration.
    rewards = np.array([10.0, 3.0], dtype=np.float32)
    dones = np.array([True, False])

    # Done block: log env 0 as (counters+1, counters+rewards[0]), zero env 0.
    ep_len = int(episode_lengths[0]) + 1
    ep_rew = float(episode_rewards[0]) + float(rewards[0])
    assert (ep_len, ep_rew) == (2, 11.0)
    episode_lengths[0] = 0
    episode_rewards[0] = 0.0

    # Swap block step 4: wipe ALL counters (env 1's in-flight partial episode
    # is dropped — mirrors dreamer_srl_main.py:1436-1437).
    episode_lengths[:] = 0
    episode_rewards[:] = 0.0

    # End-of-iteration advance with the PRE-swap rewards/dones: must no-op.
    _advance_episode_counters(episode_lengths, episode_rewards, rewards, dones,
                              stage_swapped=True)

    np.testing.assert_array_equal(
        episode_lengths, np.zeros(num_envs, dtype=np.int32),
        err_msg="Pre-swap step leaked +1 length into the new stage's counters "
                "(C1: swap-iteration advance not skipped)",
    )
    np.testing.assert_allclose(
        episode_rewards, np.zeros(num_envs, dtype=np.float32),
        err_msg="Pre-swap step's reward leaked into the new stage's counters "
                "(C1: swap-iteration advance not skipped)",
    )
