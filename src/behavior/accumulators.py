"""M1/M2/M5 online behavior-measure accumulators.

Extracted verbatim from ``train.py`` (lines 967-1283). Pure numpy — no JAX,
no PyTorch, no Lightning dependency.  Both the JAX trainer and the sheeprl
bridge import from here.

Class summary:
  BMState  — holds all numpy counter arrays for one vectorized environment.
  make_bm_state()          — factory (zeroed arrays).
  bm_step_update()         — one-step counter update (matches train.py exactly).
  bm_reset_env()           — episode-end reset for one env slot.
  bm_finalise_episode()    — returns *_raw dict of per-episode scalars.
  bm_finalise_to_wandb_keys() — maps *_raw -> WandB key namespace (for sheeprl).
  bm_wandb_keys()          — enumerates all WandB keys this module emits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

_NUM_CLASSES = 2  # 0 = predator, 1 = rabbit


@dataclass
class BMState:
    """All numpy accumulators for M1/M2/M5, per-env per-class and per-tag.

    ``num_envs``          — vectorized env width (use 1 for sheeprl single-instance).
    ``num_predator_tags`` — number of predator tags from params.predator_tags.
    ``num_neutral_tags``  — number of neutral (rabbit) tags from params.neutral_tags.
    ``bm_R``              — cue radius (from behavior_measures.cue_radius YAML key).
    ``bm_K``              — observation window in steps (behavior_measures.obs_window).
    """

    num_envs: int
    num_predator_tags: int
    num_neutral_tags: int
    bm_R: float
    bm_K: int

    # Per-env, per-class M1/M2/M5 counters (int64). Populated by __post_init__.
    m1_candidates:  np.ndarray = field(init=False)
    m1_interrupted: np.ndarray = field(init=False)
    m2_onsets:      np.ndarray = field(init=False)
    m2_dives:       np.ndarray = field(init=False)
    m5_threat_steps: np.ndarray = field(init=False)
    m5_safe_steps:   np.ndarray = field(init=False)
    m5_eat_threat:   np.ndarray = field(init=False)
    m5_eat_safe:     np.ndarray = field(init=False)

    # Per-tag accumulators (same semantics, indexed [env, tag_slot]).
    # Layout: first num_predator_tags entries = predator, then neutral_tags.
    m1_candidates_tag:  np.ndarray = field(init=False)
    m1_interrupted_tag: np.ndarray = field(init=False)
    m2_onsets_tag:      np.ndarray = field(init=False)
    m2_dives_tag:       np.ndarray = field(init=False)
    m5_threat_steps_tag: np.ndarray = field(init=False)
    m5_safe_steps_tag:   np.ndarray = field(init=False)
    m5_eat_threat_tag:   np.ndarray = field(init=False)
    m5_eat_safe_tag:     np.ndarray = field(init=False)

    # K-buffer state for M1 / M2.
    m1_candidate_age:     np.ndarray = field(init=False)   # [num_envs, _NUM_CLASSES]; -1=no pending
    m1_candidate_tag_idx: np.ndarray = field(init=False)
    m1_steps_since_eat:   np.ndarray = field(init=False)   # [num_envs]
    m2_onset_age:         np.ndarray = field(init=False)
    m2_onset_tag_idx:     np.ndarray = field(init=False)
    m2_in_bush_seen:      np.ndarray = field(init=False)   # [num_envs, _NUM_CLASSES]

    # Prior-step "threat in radius" state (for M2 onset detection).
    m_prev_threat_in_R:     np.ndarray = field(init=False)  # [num_envs, _NUM_CLASSES]
    m_prev_threat_in_R_tag: np.ndarray = field(init=False)

    def __post_init__(self):
        ne = self.num_envs
        nc = _NUM_CLASSES
        nt = self.num_predator_tags + self.num_neutral_tags

        self.m1_candidates  = np.zeros((ne, nc), dtype=np.int64)
        self.m1_interrupted = np.zeros((ne, nc), dtype=np.int64)
        self.m2_onsets      = np.zeros((ne, nc), dtype=np.int64)
        self.m2_dives       = np.zeros((ne, nc), dtype=np.int64)
        self.m5_threat_steps = np.zeros((ne, nc), dtype=np.int64)
        self.m5_safe_steps   = np.zeros((ne, nc), dtype=np.int64)
        self.m5_eat_threat   = np.zeros((ne, nc), dtype=np.int64)
        self.m5_eat_safe     = np.zeros((ne, nc), dtype=np.int64)

        self.m1_candidates_tag  = np.zeros((ne, nt), dtype=np.int64)
        self.m1_interrupted_tag = np.zeros((ne, nt), dtype=np.int64)
        self.m2_onsets_tag      = np.zeros((ne, nt), dtype=np.int64)
        self.m2_dives_tag       = np.zeros((ne, nt), dtype=np.int64)
        self.m5_threat_steps_tag = np.zeros((ne, nt), dtype=np.int64)
        self.m5_safe_steps_tag   = np.zeros((ne, nt), dtype=np.int64)
        self.m5_eat_threat_tag   = np.zeros((ne, nt), dtype=np.int64)
        self.m5_eat_safe_tag     = np.zeros((ne, nt), dtype=np.int64)

        self.m1_candidate_age     = np.full((ne, nc), -1, dtype=np.int32)
        self.m1_candidate_tag_idx = np.full((ne, nc), -1, dtype=np.int32)
        self.m1_steps_since_eat   = np.zeros(ne, dtype=np.int32)
        self.m2_onset_age         = np.full((ne, nc), -1, dtype=np.int32)
        self.m2_onset_tag_idx     = np.full((ne, nc), -1, dtype=np.int32)
        self.m2_in_bush_seen      = np.zeros((ne, nc), dtype=bool)
        self.m_prev_threat_in_R     = np.zeros((ne, nc), dtype=bool)
        self.m_prev_threat_in_R_tag = np.zeros((ne, nt), dtype=bool)


def make_bm_state(
    num_envs: int,
    num_predator_tags: int,
    num_neutral_tags: int,
    bm_R: float,
    bm_K: int,
) -> BMState:
    """Factory — returns a zero-initialised BMState."""
    return BMState(
        num_envs=num_envs,
        num_predator_tags=num_predator_tags,
        num_neutral_tags=num_neutral_tags,
        bm_R=bm_R,
        bm_K=bm_K,
    )


def bm_reset_env(bm: BMState, i: int) -> None:
    """Reset all behavior-measure accumulators for env slot *i* at episode end.

    Body mirrors ``train.py`` ``_bm_reset_env`` verbatim.
    """
    bm.m1_candidates[i, :]  = 0;  bm.m1_interrupted[i, :] = 0
    bm.m2_onsets[i, :]      = 0;  bm.m2_dives[i, :]       = 0
    bm.m5_threat_steps[i, :] = 0; bm.m5_safe_steps[i, :]  = 0
    bm.m5_eat_threat[i, :]   = 0; bm.m5_eat_safe[i, :]    = 0
    bm.m1_candidates_tag[i, :]  = 0; bm.m1_interrupted_tag[i, :] = 0
    bm.m2_onsets_tag[i, :]      = 0; bm.m2_dives_tag[i, :]       = 0
    bm.m5_threat_steps_tag[i, :] = 0; bm.m5_safe_steps_tag[i, :] = 0
    bm.m5_eat_threat_tag[i, :]   = 0; bm.m5_eat_safe_tag[i, :]   = 0
    bm.m1_candidate_age[i, :]     = -1; bm.m1_candidate_tag_idx[i, :] = -1
    bm.m2_onset_age[i, :]         = -1; bm.m2_onset_tag_idx[i, :]     = -1
    bm.m2_in_bush_seen[i, :]      = False
    bm.m_prev_threat_in_R[i, :]     = False
    bm.m_prev_threat_in_R_tag[i, :] = False
    bm.m1_steps_since_eat[i]        = 0


def bm_step_update(bm: BMState, info_np_t: dict, done_mask: np.ndarray) -> None:
    """One-step update of M1/M2/M5 counters across all envs, both classes.

    Body mirrors ``train.py`` ``_bm_step_update`` verbatim.

    Args:
        bm:          BMState holding all accumulator arrays.
        info_np_t:   per-env arrays for this single step — keys ``ate_food``,
                     ``agent_in_bush``, ``dist_per_predator``, ``dist_per_neutral``.
                     Shape [num_envs] or [num_envs, N].
        done_mask:   bool array [num_envs] — True for envs completing an episode.
                     (Passed for signature parity with the old closure; not used
                     inside the step update itself — the caller calls
                     ``bm_reset_env`` *after* ``bm_step_update``.)
    """
    num_envs = bm.num_envs
    bm_R = bm.bm_R
    bm_K = bm.bm_K
    num_predator_for_log = bm.num_predator_tags
    num_neutral_for_log  = bm.num_neutral_tags
    _num_tag_slots = num_predator_for_log + num_neutral_for_log
    _pred_slice    = slice(0, num_predator_for_log)
    _neutral_slice = slice(num_predator_for_log, num_predator_for_log + num_neutral_for_log)

    ate_food_t = info_np_t['ate_food'].astype(bool)          # [num_envs]
    in_bush_t  = info_np_t['agent_in_bush'].astype(bool)     # [num_envs]

    # Per-class threat distances
    dist_pred_t = info_np_t.get('dist_per_predator')   # [num_envs, num_pred] or None
    dist_neut_t = info_np_t.get('dist_per_neutral')    # [num_envs, num_neut] or None

    # threat_in_R[env, class]: True iff min distance to class-c entity < bm_R
    threat_in_R = np.zeros((num_envs, _NUM_CLASSES), dtype=bool)
    if dist_pred_t is not None and dist_pred_t.shape[1] > 0:
        threat_in_R[:, 0] = np.min(dist_pred_t, axis=1) < bm_R
    if dist_neut_t is not None and dist_neut_t.shape[1] > 0:
        threat_in_R[:, 1] = np.min(dist_neut_t, axis=1) < bm_R

    # threat_in_R_tag[env, tag_slot]: per-tag threat flag
    threat_in_R_tag = np.zeros((num_envs, _num_tag_slots), dtype=bool)
    if dist_pred_t is not None and num_predator_for_log > 0:
        for j in range(min(num_predator_for_log, dist_pred_t.shape[1])):
            threat_in_R_tag[:, _pred_slice.start + j] = dist_pred_t[:, j] < bm_R
    if dist_neut_t is not None and num_neutral_for_log > 0:
        for j in range(min(num_neutral_for_log, dist_neut_t.shape[1])):
            threat_in_R_tag[:, _neutral_slice.start + j] = dist_neut_t[:, j] < bm_R

    # --- M5: per-class threat/safe step and eat counters ---
    for c in range(_NUM_CLASSES):
        under_threat = threat_in_R[:, c]
        bm.m5_threat_steps[:, c] += under_threat.astype(np.int64)
        bm.m5_safe_steps[:, c]   += (~under_threat).astype(np.int64)
        bm.m5_eat_threat[:, c]   += (under_threat & ate_food_t).astype(np.int64)
        bm.m5_eat_safe[:, c]     += (~under_threat & ate_food_t).astype(np.int64)
    # M5 per-tag
    for j in range(_num_tag_slots):
        ut = threat_in_R_tag[:, j]
        bm.m5_threat_steps_tag[:, j] += ut.astype(np.int64)
        bm.m5_safe_steps_tag[:, j]   += (~ut).astype(np.int64)
        bm.m5_eat_threat_tag[:, j]   += (ut & ate_food_t).astype(np.int64)
        bm.m5_eat_safe_tag[:, j]     += (~ut & ate_food_t).astype(np.int64)

    # --- M1: interrupted-feeding detection ---
    # 1. Update steps_since_eat
    bm.m1_steps_since_eat[:] = np.where(ate_food_t, 0, bm.m1_steps_since_eat + 1)

    for c in range(_NUM_CLASSES):
        # 2. Age pending candidates FIRST (before recording new ones this step).
        for env_i in range(num_envs):
            if bm.m1_candidate_age[env_i, c] >= 0:
                bm.m1_candidate_age[env_i, c] += 1
                if bm.m1_candidate_age[env_i, c] >= bm_K:
                    # Resolve: interrupted iff agent has not eaten for >= bm_K steps after
                    # the candidate event.
                    if bm.m1_steps_since_eat[env_i] >= bm_K:
                        bm.m1_interrupted[env_i, c] += 1
                        tag_j = bm.m1_candidate_tag_idx[env_i, c]
                        if tag_j >= 0:
                            bm.m1_interrupted_tag[env_i, tag_j] += 1
                    # NOTE: the per-tag denominator (m1_candidates_tag) is incremented at
                    # candidate-RECORD time below (same instant as the per-class
                    # m1_candidates denominator), not here at resolution time — Finding
                    # C-math #2 (per-class vs per-tag denominators counted at different
                    # instants, diverging on overwrite / episode-end).
                    bm.m1_candidate_age[env_i, c] = -1
                    bm.m1_candidate_tag_idx[env_i, c] = -1

        # 3. Record new candidates: ate_food this step AND threat in radius
        new_cand = ate_food_t & threat_in_R[:, c]
        for env_i in range(num_envs):
            if new_cand[env_i]:
                # Resolve old pending candidate (conservative: not-interrupted)
                # before overwriting with the new one.
                if bm.m1_candidate_age[env_i, c] >= 0:
                    pass  # counters already incremented at candidate-record time
                bm.m1_candidates[env_i, c] += 1
                bm.m1_candidate_age[env_i, c] = 0
                # Find nearest predator/neutral tag for per-tag bucket
                if c == 0 and dist_pred_t is not None and num_predator_for_log > 0 and dist_pred_t.shape[1] > 0:
                    best_j = int(np.argmin(dist_pred_t[env_i]))
                    bm.m1_candidate_tag_idx[env_i, c] = _pred_slice.start + min(best_j, num_predator_for_log - 1)
                elif c == 1 and dist_neut_t is not None and num_neutral_for_log > 0 and dist_neut_t.shape[1] > 0:
                    best_j = int(np.argmin(dist_neut_t[env_i]))
                    bm.m1_candidate_tag_idx[env_i, c] = _neutral_slice.start + min(best_j, num_neutral_for_log - 1)
                else:
                    bm.m1_candidate_tag_idx[env_i, c] = -1
                # Per-tag denominator counted HERE, at record time — same instant as
                # the per-class m1_candidates denominator (Finding C-math #2 fix).
                tag_j = bm.m1_candidate_tag_idx[env_i, c]
                if tag_j >= 0:
                    bm.m1_candidates_tag[env_i, tag_j] += 1

    # --- M2: bush-dive detection ---
    for c in range(_NUM_CLASSES):
        threat_now = threat_in_R[:, c]

        for env_i in range(num_envs):
            # 1. Detect onset: prev not in R, now in R, agent not already in bush
            if (not bm.m_prev_threat_in_R[env_i, c]) and threat_now[env_i] and (not in_bush_t[env_i]):
                # Resolve old pending onset before overwriting
                if bm.m2_onset_age[env_i, c] >= 0:
                    pass  # count will be tallied at age==K
                bm.m2_onsets[env_i, c] += 1
                bm.m2_onset_age[env_i, c] = 0
                bm.m2_in_bush_seen[env_i, c] = False
                # Per-tag: find triggering instance
                if c == 0 and dist_pred_t is not None and num_predator_for_log > 0 and dist_pred_t.shape[1] > 0:
                    best_j = int(np.argmin(dist_pred_t[env_i]))
                    bm.m2_onset_tag_idx[env_i, c] = _pred_slice.start + min(best_j, num_predator_for_log - 1)
                elif c == 1 and dist_neut_t is not None and num_neutral_for_log > 0 and dist_neut_t.shape[1] > 0:
                    best_j = int(np.argmin(dist_neut_t[env_i]))
                    bm.m2_onset_tag_idx[env_i, c] = _neutral_slice.start + min(best_j, num_neutral_for_log - 1)
                else:
                    bm.m2_onset_tag_idx[env_i, c] = -1
                # Per-tag denominator counted HERE, at onset-record time — same instant
                # as the per-class m2_onsets denominator (Finding C-math #2 fix).
                tag_j = bm.m2_onset_tag_idx[env_i, c]
                if tag_j >= 0:
                    bm.m2_onsets_tag[env_i, tag_j] += 1

            # 2. Mark if bush entered during window
            if bm.m2_onset_age[env_i, c] >= 0 and in_bush_t[env_i]:
                bm.m2_in_bush_seen[env_i, c] = True

            # 3. Age and resolve
            if bm.m2_onset_age[env_i, c] >= 0:
                bm.m2_onset_age[env_i, c] += 1
                if bm.m2_onset_age[env_i, c] >= bm_K:
                    if bm.m2_in_bush_seen[env_i, c]:
                        bm.m2_dives[env_i, c] += 1
                        tag_j = bm.m2_onset_tag_idx[env_i, c]
                        if tag_j >= 0:
                            bm.m2_dives_tag[env_i, tag_j] += 1
                    # NOTE: the per-tag denominator (m2_onsets_tag) was counted at
                    # onset-RECORD time above, not here — see Finding C-math #2 fix.
                    bm.m2_onset_age[env_i, c] = -1
                    bm.m2_onset_tag_idx[env_i, c] = -1
                    bm.m2_in_bush_seen[env_i, c] = False

    # Update prev threat state for next step
    bm.m_prev_threat_in_R[:, :]     = threat_in_R
    bm.m_prev_threat_in_R_tag[:, :] = threat_in_R_tag


def _bm_finalise_tag(
    bm: BMState,
    i: int,
    j: int,
    tag: str,
    class_name: str,
    ep_data: dict,
) -> None:
    """Compute per-tag BM scalars for env *i*, tag slot *j* and add to ep_data.

    Body mirrors ``train.py`` ``_bm_finalise_tag`` verbatim.
    """
    # M1
    denom = int(bm.m1_candidates_tag[i, j])
    if denom > 0:
        ep_data[f"interrupted_feeding_rate_{class_name}_{tag}_raw"] = float(bm.m1_interrupted_tag[i, j]) / float(denom)
    else:
        ep_data[f"interrupted_feeding_rate_{class_name}_{tag}_raw"] = float("nan")

    # M2
    denom2 = int(bm.m2_onsets_tag[i, j])
    if denom2 > 0:
        ep_data[f"bush_dive_rate_{class_name}_{tag}_raw"] = float(bm.m2_dives_tag[i, j]) / float(denom2)
    else:
        ep_data[f"bush_dive_rate_{class_name}_{tag}_raw"] = float("nan")

    # M5
    ts  = int(bm.m5_threat_steps_tag[i, j])
    ss  = int(bm.m5_safe_steps_tag[i, j])
    et  = int(bm.m5_eat_threat_tag[i, j])
    es  = int(bm.m5_eat_safe_tag[i, j])
    pet = (et / ts) if ts > 0 else float("nan")
    pes = (es / ss) if ss > 0 else float("nan")
    ep_data[f"eat_under_threat_rate_{class_name}_{tag}_raw"]  = pet
    ep_data[f"eat_safe_rate_{class_name}_{tag}_raw"]          = pes
    if ts > 0 and es > 0:
        ep_data[f"eat_under_threat_ratio_{class_name}_{tag}_raw"] = float(pet) / float(pes)
    else:
        ep_data[f"eat_under_threat_ratio_{class_name}_{tag}_raw"] = float("nan")


def bm_finalise_episode(
    bm: BMState,
    i: int,
    predator_tags: Tuple[str, ...],
    neutral_tags: Tuple[str, ...],
) -> dict:
    """Compute per-episode BM scalars for env slot *i*.

    Returns a dict of ``*_raw`` keys.  Body mirrors ``train.py``
    ``_bm_finalise_episode`` verbatim.
    """
    num_predator_for_log = bm.num_predator_tags
    _pred_slice    = slice(0, num_predator_for_log)
    _neutral_slice = slice(num_predator_for_log, num_predator_for_log + bm.num_neutral_tags)

    ep_data: dict = {}
    for c, cname in enumerate(("predator", "rabbit")):
        # M1
        denom = int(bm.m1_candidates[i, c])
        if denom > 0:
            ep_data[f"interrupted_feeding_rate_{cname}_raw"] = float(bm.m1_interrupted[i, c]) / float(denom)
        else:
            ep_data[f"interrupted_feeding_rate_{cname}_raw"] = float("nan")
        ep_data[f"interrupted_feeding_denom_{cname}_raw"] = denom

        # M2
        denom2 = int(bm.m2_onsets[i, c])
        if denom2 > 0:
            ep_data[f"bush_dive_rate_{cname}_raw"] = float(bm.m2_dives[i, c]) / float(denom2)
        else:
            ep_data[f"bush_dive_rate_{cname}_raw"] = float("nan")
        ep_data[f"bush_dive_denom_{cname}_raw"] = denom2

        # M5
        threat_steps = int(bm.m5_threat_steps[i, c])
        safe_steps   = int(bm.m5_safe_steps[i, c])
        eat_threat   = int(bm.m5_eat_threat[i, c])
        eat_safe     = int(bm.m5_eat_safe[i, c])
        p_eat_threat = (eat_threat / threat_steps) if threat_steps > 0 else float("nan")
        p_eat_safe   = (eat_safe / safe_steps)     if safe_steps   > 0 else float("nan")
        ep_data[f"eat_under_threat_rate_{cname}_raw"]       = p_eat_threat
        ep_data[f"eat_safe_rate_{cname}_raw"]               = p_eat_safe
        ep_data[f"eat_under_threat_safe_steps_{cname}_raw"] = safe_steps
        if threat_steps > 0 and eat_safe > 0:
            ep_data[f"eat_under_threat_ratio_{cname}_raw"] = float(p_eat_threat) / float(p_eat_safe)
        else:
            ep_data[f"eat_under_threat_ratio_{cname}_raw"] = float("nan")

    # Per-tag M1, M2, M5
    for j_in_slice, tag in enumerate(predator_tags):
        j = _pred_slice.start + j_in_slice
        _bm_finalise_tag(bm, i, j, tag, "predator", ep_data)
    for j_in_slice, tag in enumerate(neutral_tags):
        j = _neutral_slice.start + j_in_slice
        _bm_finalise_tag(bm, i, j, tag, "rabbit", ep_data)

    return ep_data


def bm_drive_batch(
    bm: BMState,
    ate_food_steps: np.ndarray,          # [T, B]
    agent_in_bush_steps: np.ndarray,     # [T, B]
    dist_per_predator_steps,             # [T, B, P] or None
    dist_per_neutral_steps,              # [T, B, N] or None
    done_steps: np.ndarray,              # [T, B]
    predator_tags: Tuple[str, ...],
    neutral_tags: Tuple[str, ...],
) -> dict:
    """Drive the BM state machine over a whole [T, B] collected batch
    (train.py DreamerV3 Site 2).

    Interleaves the per-step update with per-done finalise/reset — the same
    step -> finalise -> reset ordering as the per-step drivers (train.py
    Site 1 rPPO, dreamer_srl_main) — so no step after a done leaks into the
    finished episode, the next episode keeps its opening steps, and a second
    done for the same env within the batch yields a second valid finalisation.
    (H10 fix — see docs/develop/active/issues/diag_fable5_20260704/
    fix_plan_h10_dreamer_batch_bm.md.)

    Mutates ``bm`` (per-env reset at each done). Returns
    ``{(t, i): ep_data}`` — one finalised ``*_raw`` dict per done event at
    step ``t`` for env ``i``.
    """
    results: dict = {}
    T = done_steps.shape[0]
    for t in range(T):
        info_t = {
            'ate_food':      ate_food_steps[t].astype(bool),
            'agent_in_bush': agent_in_bush_steps[t].astype(bool),
        }
        if dist_per_predator_steps is not None:
            info_t['dist_per_predator'] = dist_per_predator_steps[t]
        if dist_per_neutral_steps is not None:
            info_t['dist_per_neutral'] = dist_per_neutral_steps[t]
        done_t = done_steps[t].astype(bool)
        bm_step_update(bm, info_t, done_t)
        for i in np.where(done_t)[0]:
            i = int(i)
            results[(t, i)] = bm_finalise_episode(bm, i, predator_tags, neutral_tags)
            bm_reset_env(bm, i)
    return results


def bm_finalise_to_wandb_keys(
    ep_data_raw: dict,
    predator_tags: Tuple[str, ...],
    neutral_tags: Tuple[str, ...],
) -> dict:
    """Map ``*_raw`` scalars from ``bm_finalise_episode`` into the WandB key namespace.

    For sheeprl: the wrapper places these on the terminal info dict;
    sheeprl's aggregator averages across envs/episodes for free.
    For JAX: ``train.py`` uses ``_append_per_measure_mean`` which does
    NaN-skipping iteration-level averaging — that path is preserved
    as-is in ``train.py`` calling ``_bm_log_wandb``.
    """
    out: dict = {}
    for cname in ("predator", "rabbit"):
        for src_suffix, dst_name in [
            ("interrupted_feeding_rate",       "InterruptedFeedingRate"),
            ("interrupted_feeding_denom",      "InterruptedFeedingDenominator"),
            ("bush_dive_rate",                 "BushDiveRate"),
            ("bush_dive_denom",                "BushDiveDenominator"),
            ("eat_under_threat_ratio",         "EatUnderThreatRatio"),
            ("eat_under_threat_rate",          "EatUnderThreatRate"),
            ("eat_safe_rate",                  "EatSafeRate"),
            ("eat_under_threat_safe_steps",    "EatUnderThreatSafeSteps"),
        ]:
            src_key = f"{src_suffix}_{cname}_raw"
            if src_key in ep_data_raw:
                v = ep_data_raw[src_key]
                if not (isinstance(v, float) and v != v):  # skip NaN
                    out[f"Episode/{dst_name}_{cname}"] = float(v)
    for tag in predator_tags:
        for src_suffix, dst_name in [
            ("interrupted_feeding_rate_predator", "InterruptedFeedingRate_predator"),
            ("bush_dive_rate_predator",           "BushDiveRate_predator"),
            ("eat_under_threat_ratio_predator",   "EatUnderThreatRatio_predator"),
            ("eat_under_threat_rate_predator",    "EatUnderThreatRate_predator"),
        ]:
            src_key = f"{src_suffix}_{tag}_raw"
            if src_key in ep_data_raw:
                v = ep_data_raw[src_key]
                if not (isinstance(v, float) and v != v):
                    out[f"Episode/{dst_name}_{tag}"] = float(v)
    for tag in neutral_tags:
        for src_suffix, dst_name in [
            ("interrupted_feeding_rate_rabbit", "InterruptedFeedingRate_rabbit"),
            ("bush_dive_rate_rabbit",           "BushDiveRate_rabbit"),
            ("eat_under_threat_ratio_rabbit",   "EatUnderThreatRatio_rabbit"),
            ("eat_under_threat_rate_rabbit",    "EatUnderThreatRate_rabbit"),
        ]:
            src_key = f"{src_suffix}_{tag}_raw"
            if src_key in ep_data_raw:
                v = ep_data_raw[src_key]
                if not (isinstance(v, float) and v != v):
                    out[f"Episode/{dst_name}_{tag}"] = float(v)
    return out


def bm_wandb_keys(
    predator_tags: Tuple[str, ...],
    neutral_tags: Tuple[str, ...],
) -> list:
    """Return the full list of WandB keys this module emits, in stable order.

    Used by the sheeprl aggregator to register MeanMetrics at startup.
    """
    keys = []
    for cname in ("predator", "rabbit"):
        keys.append(f"Episode/InterruptedFeedingRate_{cname}")
        keys.append(f"Episode/InterruptedFeedingDenominator_{cname}")
        keys.append(f"Episode/BushDiveRate_{cname}")
        keys.append(f"Episode/BushDiveDenominator_{cname}")
        keys.append(f"Episode/EatUnderThreatRatio_{cname}")
        keys.append(f"Episode/EatUnderThreatRate_{cname}")
        keys.append(f"Episode/EatSafeRate_{cname}")
        keys.append(f"Episode/EatUnderThreatSafeSteps_{cname}")
    for tag in predator_tags:
        keys.append(f"Episode/InterruptedFeedingRate_predator_{tag}")
        keys.append(f"Episode/BushDiveRate_predator_{tag}")
        keys.append(f"Episode/EatUnderThreatRatio_predator_{tag}")
        keys.append(f"Episode/EatUnderThreatRate_predator_{tag}")
    for tag in neutral_tags:
        keys.append(f"Episode/InterruptedFeedingRate_rabbit_{tag}")
        keys.append(f"Episode/BushDiveRate_rabbit_{tag}")
        keys.append(f"Episode/EatUnderThreatRatio_rabbit_{tag}")
        keys.append(f"Episode/EatUnderThreatRate_rabbit_{tag}")
    return keys


# ── CP5: per-episode sampled-behavioural-parameter logging ────────────────

def build_episode_log_dict(state, params) -> dict:
    """Build per-episode WandB log dict for the 5 sampled behavioural fields.

    Called once per episode-done event, after ``jax_reset``, using the STATE
    that was active DURING the just-completed episode (the sampled values are
    constant within an episode — they change only at the next reset).

    Returns a dict with one key per animal entity per field:
      ``Episode/sampled_detect_<tag>``
      ``Episode/sampled_max_stamina_<tag>``
      ``Episode/sampled_recovery_<tag>``
      ``Episode/sampled_hunt_thresh_<tag>``
      ``Episode/sampled_lose_interest_<tag>``

    Values are Python floats (host-side); the caller averages them across
    episodes within an iteration window before passing to ``wandb.log``.

    CP5 deliverable — plan §"Logging":
      "For each animal entity with a static tag, log five WandB metrics per
       episode (after each reset): Episode/sampled_detect_<tag>, etc."

    Args:
        state: EnvState — the state at the START of the episode (after reset)
               or equivalently any step within it (sampled values are constant).
        params: EnvParams — holds animal_tags (static tuple, pytree_node=False).

    Returns:
        dict[str, float]: one key per (entity, field) combination.
    """
    out: dict = {}
    detect_arr = np.asarray(state.animal_detect_sampled)
    max_stam_arr = np.asarray(state.animal_max_stamina_sampled)
    recovery_arr = np.asarray(state.animal_recovery_sampled)
    hunt_thresh_arr = np.asarray(state.animal_hunt_thresh_sampled)
    lose_int_arr = np.asarray(state.animal_lose_interest_sampled)

    for i, tag in enumerate(params.animal_tags):
        out[f"Episode/sampled_detect_{tag}"] = float(detect_arr[i])
        out[f"Episode/sampled_max_stamina_{tag}"] = float(max_stam_arr[i])
        out[f"Episode/sampled_recovery_{tag}"] = float(recovery_arr[i])
        out[f"Episode/sampled_hunt_thresh_{tag}"] = float(hunt_thresh_arr[i])
        out[f"Episode/sampled_lose_interest_{tag}"] = float(lose_int_arr[i])

    return out


def sampled_wandb_keys(animal_tags: Tuple[str, ...]) -> list:
    """Return the full list of ``Episode/sampled_*_<tag>`` WandB keys in stable order.

    Companion to ``build_episode_log_dict`` — used by consumers that need to
    pre-register the key names (e.g. sheeprl MeanMetrics at startup).

    CP5 deliverable — plan §"Logging".

    Args:
        animal_tags: ``params.animal_tags`` (one entry per animal entity).

    Returns:
        list[str]: keys for all 5 fields × all N animal entities.
    """
    keys = []
    for tag in animal_tags:
        keys.append(f"Episode/sampled_detect_{tag}")
        keys.append(f"Episode/sampled_max_stamina_{tag}")
        keys.append(f"Episode/sampled_recovery_{tag}")
        keys.append(f"Episode/sampled_hunt_thresh_{tag}")
        keys.append(f"Episode/sampled_lose_interest_{tag}")
    return keys
