#!/usr/bin/env python3
"""eval_rollout.py — Offline evaluation rollout for behavior-measure toolkit v1.

Loads a frozen training checkpoint, runs N deterministic evaluation episodes
(seeded deterministically per behavior_measures.eval_seeds), and dumps per-episode
numpy archives + a threat-onset window index for offline motif clustering.

Usage
-----
    python scripts/eval_rollout.py \\
        --config <env-config-yaml>  \\
        --agent_config <agent-config-yaml>  \\
        --checkpoint <path-to-checkpoint-dir>  \\
        --output-root <output-dir>  \\
        [--eval-n-episodes N]  \\
        [--eval-seeds S1 S2 …]  \\
        [--device cpu|gpu]  \\
        [--quiet]

Output schema (under <output-root>/)
--------------------------------------
    metadata.json              — config snapshot + wall-clock + git commit
    episodes/<idx>.npz         — per-episode step arrays
    windows/threat_onsets.parquet  — one row per threat-onset event
    online_replay.json         — M1/M2/M5 re-computed offline (sanity cross-check)

Dependencies
------------
    Same conda env as training (grid_world_pain).
    pyarrow is required for parquet output:
        pip install pyarrow pandas
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.config_loader import Config, load_env_params, load_behavior_measure_cfg, load_env_config
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation, get_observation_breakdown


def _get_git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"



def _run_episode(
    params,
    policy_fn,
    rng_key,
    max_steps: int,
    deterministic: bool = True,
):
    """Run a single episode and return per-step arrays.

    Returns a dict of arrays (T,) or (T, D).
    """
    state = jax_reset(params, rng_key)
    carry = None  # recurrent hidden state; policy_fn handles init

    records = {
        "agent_pos": [],
        "action": [],
        "ate_food": [],
        "agent_in_bush": [],
        "dist_per_predator": [],
        "dist_per_neutral": [],
        "hit_predator": [],
        "hit_neutral": [],
        "nociception": [],
        "termination_reason": None,
    }
    step = 0
    done = False
    step_key = rng_key
    while step < max_steps and not done:
        obs = state  # obs = full state (policy sees sensory obs internally)
        step_key, act_key = jax.random.split(step_key)
        action, carry = policy_fn(obs, carry, act_key, deterministic=deterministic)
        next_state, reward, done_flag, info = jax_step(state, action, params)

        records["agent_pos"].append(np.array(state.agent_pos))
        records["action"].append(int(action))
        records["ate_food"].append(bool(np.array(info.get("ate_food", False))))
        records["agent_in_bush"].append(bool(np.array(info.get("agent_in_bush", False))))

        dp = info.get("dist_per_predator")
        records["dist_per_predator"].append(np.array(dp) if dp is not None else np.array([]))
        dn = info.get("dist_per_neutral")
        records["dist_per_neutral"].append(np.array(dn) if dn is not None else np.array([]))

        records["hit_predator"].append(bool(np.array(info.get("hit_predator", False))))
        records["hit_neutral"].append(bool(np.array(info.get("hit_neutral", False))))
        noci = info.get("nociception", info.get("exteroception_nociception", 0.0))
        records["nociception"].append(float(np.array(noci)))

        state = next_state
        done = bool(done_flag)
        step += 1

    records["termination_reason"] = int(np.array(info.get("termination_reason", -1)))
    T = step
    return {
        "agent_pos": np.array(records["agent_pos"], dtype=np.int32),
        "action": np.array(records["action"], dtype=np.int32),
        "ate_food": np.array(records["ate_food"], dtype=bool),
        "agent_in_bush": np.array(records["agent_in_bush"], dtype=bool),
        "dist_per_predator": _pad_ragged(records["dist_per_predator"]),
        "dist_per_neutral": _pad_ragged(records["dist_per_neutral"]),
        "hit_predator": np.array(records["hit_predator"], dtype=bool),
        "hit_neutral": np.array(records["hit_neutral"], dtype=bool),
        "nociception": np.array(records["nociception"], dtype=np.float32),
        "termination_reason": np.int32(records["termination_reason"]),
        "length": np.int32(T),
    }


# Checkpoint-container dir names: "checkpoints" (Dreamer), "models" (rPPO, train.py:623).
# A step-dir invocation is <run>/<container>/<step>; the run identity is the grandparent.
_CKPT_CONTAINER_DIRNAMES = {"checkpoints", "models"}


def _derive_run_tag(ckpt_path: Path) -> str:
    """Derive the run-identity tag from a checkpoint path.

    If the checkpoint's immediate parent directory name is a known
    checkpoint-container name (`_CKPT_CONTAINER_DIRNAMES`), the run tag is the
    grandparent directory name (the run directory); otherwise the run tag is
    the parent directory name.
    """
    return (
        ckpt_path.parent.parent.name
        if ckpt_path.parent.name in _CKPT_CONTAINER_DIRNAMES
        else ckpt_path.parent.name
    )


def _assert_eval_obs_noise_supported(bm_cfg) -> None:
    """Fail loud if `behavior_measures.eval_obs_noise` requests a mode the
    rollout does not actually enforce.

    Only "training" (the env's configured/training-time noise) is honored
    today; "zero" and "custom" would require threading `apply_noise` through
    four independent obs-computation code paths (see docs/develop/active/
    issues/FIX_EVAL_LOGGING_TRACK_B_20260723.md, Bug 5). Rather than silently
    running with training noise while `metadata.json` claims otherwise, raise
    at startup before any episode runs.
    """
    if bm_cfg.eval_obs_noise != "training":
        raise NotImplementedError(
            f"behavior_measures.eval_obs_noise={bm_cfg.eval_obs_noise!r} is not enforced by the "
            "rollout — the policy always sees the env's configured (training) noise. Only "
            "'training' is currently honored; 'zero'/'custom' would require threading "
            "apply_noise through every obs path (see docs/develop/active/issues/"
            "FIX_EVAL_LOGGING_TRACK_B_20260723.md, Bug 5). Set eval_obs_noise: 'training'."
        )


def _pad_ragged(rows):
    """Stack a list of 0- or 1-D arrays into a 2-D array, padding missing dims."""
    if not rows:
        return np.zeros((0,), dtype=np.float32)
    max_d = max(r.shape[0] if r.ndim > 0 and r.shape[0] > 0 else 0 for r in rows)
    if max_d == 0:
        return np.zeros((len(rows), 0), dtype=np.float32)
    out = np.zeros((len(rows), max_d), dtype=np.float32)
    for i, r in enumerate(rows):
        if r.ndim > 0 and r.shape[0] > 0:
            d = min(r.shape[0], max_d)
            out[i, :d] = r[:d]
    return out


def _run_episode_with_recording(
    params,
    policy_fn,
    rng_key,
    max_steps: int,
    deterministic: bool = True,
    episode_index: int = 0,
    seed: int = 0,
):
    """Recorder-aware variant of _run_episode.

    Returns (ep_data, EpisodeRecorder).  Mirrors the call pattern used by
    evaluation_core.py (_run_single_env_eval) so recordings are compatible
    with render_recordings.py.  Handles zero-predator configs (dist shape [T,0]).
    """
    from src.utils.eval_recording import EpisodeRecorder
    from src.environment.sensor import get_observation as _get_obs

    recorder = EpisodeRecorder(
        episode_index=episode_index,
        train_episode=episode_index,   # no training-episode concept here; reuse index
        seed=seed,
    )

    state = jax_reset(params, rng_key)
    carry = None

    records = {
        "agent_pos": [],
        "action": [],
        "ate_food": [],
        "agent_in_bush": [],
        "dist_per_predator": [],
        "dist_per_neutral": [],
        "hit_predator": [],
        "hit_neutral": [],
        "nociception": [],
        "termination_reason": None,
    }

    # Record initial state (action=-1, reward=0.0 -- matches evaluation_core.py)
    obs0 = _get_obs(state, params)
    true_obs0 = _get_obs(state, params, apply_noise=False)
    recorder.append(state, obs0, true_obs0, action_idx=-1, reward=0.0)

    step = 0
    done = False
    info = {}
    step_key = rng_key
    while step < max_steps and not done:
        step_key, act_key = jax.random.split(step_key)
        action, carry = policy_fn(state, carry, act_key, deterministic=deterministic)
        next_state, reward, done_flag, info = jax_step(state, action, params)

        records["agent_pos"].append(np.array(state.agent_pos))
        records["action"].append(int(action))
        records["ate_food"].append(bool(np.array(info.get("ate_food", False))))
        records["agent_in_bush"].append(bool(np.array(info.get("agent_in_bush", False))))

        dp = info.get("dist_per_predator")
        records["dist_per_predator"].append(np.array(dp) if dp is not None else np.array([]))
        dn = info.get("dist_per_neutral")
        records["dist_per_neutral"].append(np.array(dn) if dn is not None else np.array([]))

        records["hit_predator"].append(bool(np.array(info.get("hit_predator", False))))
        records["hit_neutral"].append(bool(np.array(info.get("hit_neutral", False))))
        noci = info.get("nociception", info.get("exteroception_nociception", 0.0))
        records["nociception"].append(float(np.array(noci)))

        # Record post-step state
        next_obs = _get_obs(next_state, params)
        true_next_obs = _get_obs(next_state, params, apply_noise=False)
        recorder.append(next_state, next_obs, true_next_obs,
                        action_idx=int(action), reward=float(reward))

        state = next_state
        done = bool(done_flag)
        step += 1

    records["termination_reason"] = int(np.array(info.get("termination_reason", -1)))
    T = step
    ep_data = {
        "agent_pos": np.array(records["agent_pos"], dtype=np.int32),
        "action": np.array(records["action"], dtype=np.int32),
        "ate_food": np.array(records["ate_food"], dtype=bool),
        "agent_in_bush": np.array(records["agent_in_bush"], dtype=bool),
        "dist_per_predator": _pad_ragged(records["dist_per_predator"]),
        "dist_per_neutral": _pad_ragged(records["dist_per_neutral"]),
        "hit_predator": np.array(records["hit_predator"], dtype=bool),
        "hit_neutral": np.array(records["hit_neutral"], dtype=bool),
        "nociception": np.array(records["nociception"], dtype=np.float32),
        "termination_reason": np.int32(records["termination_reason"]),
        "length": np.int32(T),
    }
    return ep_data, recorder


def _rollout_scan_jit(model, params, states0, h0, max_steps):
    """nnx.jit-wrapped scan body. MUST be entered via `nnx.jit`, not called eagerly.

    Root-cause note (found empirically while building the parity harness — flag
    for `code-reviewer`): after `nnx.update(model, restored_tree)` restores a
    checkpoint, calling `model(...)` **eagerly** (no `nnx.jit` anywhere in the call
    stack) reads a stale/inconsistent view of the restored parameters — a plain
    Python attribute read does not see the same values an `nnx.jit`-traced read
    does, even though `nnx.state(model)` checksums identically either way. The
    legacy per-episode path never hits this because its only forward-pass entry
    point, `get_action_and_value_nnx`, is itself `@nnx.jit`-decorated, so the
    first-ever forward pass of the whole process (and every one after) already
    goes through the split/merge that correctly materializes the restored state.
    A prior version of this function called `model(...)` directly inside a bare
    `jax.lax.scan` with no enclosing `nnx.jit` — `lax.scan` alone does not fix
    this (it only compiles the OUTER XLA loop; it does not perform nnx's
    graphdef/state split) — and every batched trajectory silently diverged from
    the legacy reference after step 0. Wrapping the whole scan in `nnx.jit` (this
    function) makes `_run_episodes_batched` self-contained and exactly correct.
    """
    v_step = jax.vmap(jax_step, in_axes=(0, 0, None))
    v_obs = jax.vmap(get_observation, in_axes=(0, None))
    v_obs_true = jax.vmap(lambda s, p: get_observation(s, p, apply_noise=False), in_axes=(0, None))

    def scan_fn(carry, _):
        state, h = carry
        logits, _value, h_new, _mod_info = model(v_obs(state, params), h)
        action = jnp.argmax(logits, axis=-1)
        next_state, reward, done, info = v_step(state, action, params)
        next_obs = v_obs(next_state, params)
        next_true_obs = v_obs_true(next_state, params)

        step_out = {
            "pre_agent_pos": state.agent_pos,
            "action": action,
            "reward": reward,
            "done": done,
            "ate_food": info["ate_food"],
            "agent_in_bush": info["agent_in_bush"],
            "dist_per_predator": info["dist_per_predator"],
            "dist_per_neutral": info["dist_per_neutral"],
            "hit_predator": info["hit_predator"],
            "hit_neutral": info["hit_neutral"],
            "termination_reason": info["termination_reason"],
            "snap_agent_pos": next_state.agent_pos,
            "snap_satiation": next_state.satiation,
            "snap_nutrition": next_state.nutrition,
            "snap_injury_level": next_state.injury_level,
            "snap_rest_streak": next_state.rest_streak,
            "snap_res_pos": next_state.res_pos,
            "snap_res_active": next_state.res_active,
            "snap_animal_pos": next_state.animal_pos,
            "snap_obs_pos": next_state.obs_pos,
            "obs": next_obs,
            "true_obs": next_true_obs,
        }
        return (next_state, h_new), step_out

    (_final_state, _final_h), scan_out = jax.lax.scan(
        scan_fn, (states0, h0), None, length=max_steps
    )
    return scan_out


def _run_episodes_batched(
    params,
    model,
    seeds,
    max_steps: int,
    record: bool = False,
    record_n_episodes: int = 0,
):
    """Batched (single-process, vmapped) rollout of ALL episodes at once (Tier 2).

    Bit-for-bit parity contract with `_run_episode` / `_run_episode_with_recording`:
    see docs/develop/active/refactors/EVAL_ROLLOUT_BATCHING_PERF.md for the full
    correctness argument. The two facts that make exact parity possible: (1) an
    episode is a pure function of its reset key `jax.random.PRNGKey(seed)` — all
    per-step stochasticity is carried in `state.key` — and (2) the eval policy is
    deterministic argmax with no cross-batch coupling (LayerNorm normalizes over
    the feature axis, never the batch axis).

    PRNG parity crux (concern a): the per-env reset key MUST be the stacked,
    directly-vmapped `jax.random.PRNGKey(seed)` — NOT `ParallelEnv.reset`, whose
    internal `jax.random.split(key, num_envs)` derives a completely different key
    set and would silently corrupt every trajectory.

    Returns (episodes, recorders):
      - episodes: list of per-episode dicts, SAME schema as `_run_episode`
        (agent_pos, action, ate_food, agent_in_bush, dist_per_predator,
        dist_per_neutral, hit_predator, hit_neutral, nociception,
        termination_reason, length). Caller adds `seed` afterward, exactly as
        the legacy per-episode loop does.
      - recorders: list of EpisodeRecorder|None (None for episodes beyond
        `record_n_episodes`, or when `record=False`).
    """
    import flax.nnx as nnx
    from src.utils.eval_recording import EpisodeRecorder, _snapshot_state
    from types import SimpleNamespace

    num_envs = len(seeds)
    keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])

    v_reset = jax.vmap(jax_reset, in_axes=(None, 0))
    v_obs = jax.vmap(get_observation, in_axes=(0, None))
    v_obs_true = jax.vmap(lambda s, p: get_observation(s, p, apply_noise=False), in_axes=(0, None))

    states0 = v_reset(params, keys)

    # --- Checkpoint (concern a): PRNG reset-key parity guard. Permanent regression
    # guard, not just a one-off manual check — this is THE correctness-critical
    # invariant of the whole batched path. Cheap: num_envs unbatched jax_reset calls,
    # once per config eval (negligible next to the max_steps scan).
    for i, s in enumerate(seeds):
        ref_key = jax_reset(params, jax.random.PRNGKey(int(s))).key
        if not bool(jnp.array_equal(states0.key[i], ref_key)):
            raise RuntimeError(
                f"Batched reset PRNG parity check failed at seed index {i} (seed={s}): "
                "state.key from the batched vmap(jax_reset) does not match "
                "jax_reset(params, PRNGKey(seed)) called directly. The batched reset "
                "must use jnp.stack([PRNGKey(s) for s in seeds]) + vmap(jax_reset), NOT "
                "ParallelEnv.reset (whose internal jax.random.split(key, num_envs) "
                "derives a different key set)."
            )

    h0 = model.initial_state(batch_size=num_envs)
    obs0 = v_obs(states0, params)
    true_obs0 = v_obs_true(states0, params)

    # --- Checkpoint (concern b): vmap axis correctness. logits are (num_envs,
    # action_dim) by construction (model(obs, h) with a batched leading axis) and
    # scan_fn's `jnp.argmax(logits, axis=-1)` gives (num_envs,) — the bare
    # `jnp.argmax(logits)` in get_action_and_value_nnx is only correct unbatched.
    # (Read `model.action_dim` rather than probing with an eager forward call —
    # see _rollout_scan_jit's docstring for why an eager, non-nnx.jit call here
    # would itself be numerically unsafe.)
    action_dim = model.action_dim

    scan_out = nnx.jit(_rollout_scan_jit, static_argnames=("max_steps",))(
        model, params, states0, h0, max_steps
    )
    assert scan_out["action"].shape == (max_steps, num_envs), (
        f"Expected batched action shape (max_steps={max_steps}, num_envs={num_envs}), "
        f"got {scan_out['action'].shape}"
    )
    assert bool(jnp.all(scan_out["action"] < action_dim)), (
        f"Batched action indices out of range for action_dim={action_dim} — "
        "argmax likely reduced over the wrong axis."
    )

    # Bring everything to host ONCE (per Design step 4).
    scan_out = jax.tree_util.tree_map(np.asarray, scan_out)
    states0_np = jax.tree_util.tree_map(np.asarray, states0)
    obs0_np = np.asarray(obs0)
    true_obs0_np = np.asarray(true_obs0)

    done_seq = scan_out["done"]  # (max_steps, num_envs) bool
    # `max_steps` truncation guarantees every env reaches done=True within the
    # scan window (core.py: truncated = next_step >= params.max_steps fires on the
    # max_steps-th step call at the latest), so argmax always finds a real hit.
    T = np.argmax(done_seq, axis=0) + 1  # (num_envs,) — episode length, matches legacy `T`

    episodes = []
    recorders = []
    for i in range(num_envs):
        Ti = int(T[i])

        ep_data = {
            "agent_pos": np.asarray(scan_out["pre_agent_pos"][:Ti, i], dtype=np.int32),
            "action": np.asarray(scan_out["action"][:Ti, i], dtype=np.int32),
            "ate_food": np.asarray(scan_out["ate_food"][:Ti, i], dtype=bool),
            "agent_in_bush": np.asarray(scan_out["agent_in_bush"][:Ti, i], dtype=bool),
            "dist_per_predator": np.asarray(scan_out["dist_per_predator"][:Ti, i], dtype=np.float32),
            "dist_per_neutral": np.asarray(scan_out["dist_per_neutral"][:Ti, i], dtype=np.float32),
            "hit_predator": np.asarray(scan_out["hit_predator"][:Ti, i], dtype=bool),
            "hit_neutral": np.asarray(scan_out["hit_neutral"][:Ti, i], dtype=bool),
            # No live info dict ever carries a 'nociception'/'exteroception_nociception'
            # key (verified against core.py's jax_step info schema) — the legacy
            # `.get(..., 0.0)` fallback always fires, so this is always exactly 0.0.
            "nociception": np.zeros(Ti, dtype=np.float32),
            "termination_reason": np.int32(scan_out["termination_reason"][Ti - 1, i]),
            "length": np.int32(Ti),
        }
        episodes.append(ep_data)

        if record and i < record_n_episodes:
            recorder = EpisodeRecorder(episode_index=i, train_episode=i, seed=int(seeds[i]))

            # Initial snapshot (pre-loop), exactly matching
            # _run_episode_with_recording's recorder.append(state, obs0, true_obs0,
            # action_idx=-1, reward=0.0) call.
            init_state = SimpleNamespace(
                agent_pos=states0_np.agent_pos[i], satiation=states0_np.satiation[i],
                nutrition=states0_np.nutrition[i], injury_level=states0_np.injury_level[i],
                rest_streak=states0_np.rest_streak[i], res_pos=states0_np.res_pos[i],
                res_active=states0_np.res_active[i], animal_pos=states0_np.animal_pos[i],
                obs_pos=states0_np.obs_pos[i],
            )
            recorder.snapshots.append(_snapshot_state(init_state))
            recorder.obs.append(np.asarray(obs0_np[i]))
            recorder.true_obs.append(np.asarray(true_obs0_np[i]))
            recorder.actions.append(-1)
            recorder.rewards.append(0.0)

            for t in range(Ti):
                step_state = SimpleNamespace(
                    agent_pos=scan_out["snap_agent_pos"][t, i],
                    satiation=scan_out["snap_satiation"][t, i],
                    nutrition=scan_out["snap_nutrition"][t, i],
                    injury_level=scan_out["snap_injury_level"][t, i],
                    rest_streak=scan_out["snap_rest_streak"][t, i],
                    res_pos=scan_out["snap_res_pos"][t, i],
                    res_active=scan_out["snap_res_active"][t, i],
                    animal_pos=scan_out["snap_animal_pos"][t, i],
                    obs_pos=scan_out["snap_obs_pos"][t, i],
                )
                recorder.snapshots.append(_snapshot_state(step_state))
                recorder.obs.append(np.asarray(scan_out["obs"][t, i]))
                recorder.true_obs.append(np.asarray(scan_out["true_obs"][t, i]))
                recorder.actions.append(int(scan_out["action"][t, i]))
                recorder.rewards.append(float(scan_out["reward"][t, i]))

            recorders.append(recorder)
        else:
            recorders.append(None)

    return episodes, recorders


def _detect_threat_onsets(episodes, bm_cfg):
    """Scan episodes for threat-onset events (rising edge of dist < R per class).

    Returns a list of dicts (one per onset).
    """
    R = float(bm_cfg.cue_radius)
    K = int(bm_cfg.motif_window_K)
    onsets = []
    for ep_idx, ep in enumerate(episodes):
        T = int(ep["length"])
        dp = ep["dist_per_predator"]   # [T, num_pred]
        dn = ep["dist_per_neutral"]    # [T, num_neutral]

        prev_pred_in_R = False
        prev_neut_in_R = False
        for t in range(T):
            num_pred = dp.shape[1] if dp.ndim == 2 else 0
            num_neut = dn.shape[1] if dn.ndim == 2 else 0

            pred_in_R = (num_pred > 0 and float(np.min(dp[t])) < R)
            neut_in_R = (num_neut > 0 and float(np.min(dn[t])) < R)

            if pred_in_R and not prev_pred_in_R:
                best_j = int(np.argmin(dp[t])) if num_pred > 0 else -1
                onsets.append({
                    "episode_idx": ep_idx,
                    "t_star": t,
                    "triggering_class": "predator",
                    "triggering_tag": f"pred_{best_j}",
                    "window_start_t": max(0, t - 2),
                    "window_end_t": min(T - 1, t + K),
                })

            if neut_in_R and not prev_neut_in_R:
                best_j = int(np.argmin(dn[t])) if num_neut > 0 else -1
                onsets.append({
                    "episode_idx": ep_idx,
                    "t_star": t,
                    "triggering_class": "rabbit",
                    "triggering_tag": f"neut_{best_j}",
                    "window_start_t": max(0, t - 2),
                    "window_end_t": min(T - 1, t + K),
                })

            prev_pred_in_R = pred_in_R
            prev_neut_in_R = neut_in_R
    return onsets


def _compute_online_replay(episodes, bm_cfg):
    """Re-compute M1/M2/M5 offline from episode dumps (sanity cross-check)."""
    R = float(bm_cfg.cue_radius)
    K = int(bm_cfg.obs_window)
    EPS = 1e-9

    totals = {
        "m1_candidates_predator": 0, "m1_interrupted_predator": 0,
        "m1_candidates_rabbit": 0, "m1_interrupted_rabbit": 0,
        "m2_onsets_predator": 0, "m2_dives_predator": 0,
        "m2_onsets_rabbit": 0, "m2_dives_rabbit": 0,
        "m5_threat_steps_predator": 0, "m5_safe_steps_predator": 0,
        "m5_eat_threat_predator": 0, "m5_eat_safe_predator": 0,
        "m5_threat_steps_rabbit": 0, "m5_safe_steps_rabbit": 0,
        "m5_eat_threat_rabbit": 0, "m5_eat_safe_rabbit": 0,
    }

    for ep in episodes:
        T = int(ep["length"])
        dp = ep["dist_per_predator"]
        dn = ep["dist_per_neutral"]
        ate = ep["ate_food"]
        in_bush = ep["agent_in_bush"]

        for c, (dists, prefix) in enumerate([(dp, "predator"), (dn, "rabbit")]):
            num_threats = dists.shape[1] if dists.ndim == 2 else 0
            prev_in_R = False
            cand_age = -1
            onset_age = -1
            onset_bush_seen = False
            steps_since_eat = 0

            for t in range(T):
                in_R = num_threats > 0 and float(np.min(dists[t])) < R

                # M5
                if in_R:
                    totals[f"m5_threat_steps_{prefix}"] += 1
                    if ate[t]:
                        totals[f"m5_eat_threat_{prefix}"] += 1
                else:
                    totals[f"m5_safe_steps_{prefix}"] += 1
                    if ate[t]:
                        totals[f"m5_eat_safe_{prefix}"] += 1

                # M1 — update steps_since_eat FIRST, then age/resolve, then record:
                # same ordering as the online accumulator (accumulators.py:211 update,
                # :215-232 resolve, :234-257 record; fix 3e1e53e), so "interrupted"
                # means: no eat at any of the K steps after the candidate eat.
                # NOTE: the denominator below still counts at RESOLUTION time, unlike
                # the online record-time counting — known open divergence, see
                # diag_fable5_20260704/07_behavior_measures.md Finding 4. Do not
                # change it here.
                steps_since_eat = 0 if ate[t] else steps_since_eat + 1

                if cand_age >= 0:
                    cand_age += 1
                    if cand_age >= K:
                        totals[f"m1_candidates_{prefix}"] += 1
                        if steps_since_eat >= K:
                            totals[f"m1_interrupted_{prefix}"] += 1
                        cand_age = -1

                if ate[t] and in_R:
                    cand_age = 0  # record new candidate

                # M2
                if not prev_in_R and in_R and not in_bush[t]:
                    totals[f"m2_onsets_{prefix}"] += 1
                    onset_age = 0
                    onset_bush_seen = False

                if onset_age >= 0 and in_bush[t]:
                    onset_bush_seen = True
                if onset_age >= 0:
                    onset_age += 1
                    if onset_age >= K:
                        if onset_bush_seen:
                            totals[f"m2_dives_{prefix}"] += 1
                        onset_age = -1
                        onset_bush_seen = False

                prev_in_R = in_R

    # Derive rates
    results = {}
    for prefix in ("predator", "rabbit"):
        cands = totals[f"m1_candidates_{prefix}"]
        results[f"interrupted_feeding_rate_{prefix}"] = (
            totals[f"m1_interrupted_{prefix}"] / cands if cands > 0 else float("nan")
        )
        onsets = totals[f"m2_onsets_{prefix}"]
        results[f"bush_dive_rate_{prefix}"] = (
            totals[f"m2_dives_{prefix}"] / onsets if onsets > 0 else float("nan")
        )
        ts = totals[f"m5_threat_steps_{prefix}"]
        ss = totals[f"m5_safe_steps_{prefix}"]
        p_t = totals[f"m5_eat_threat_{prefix}"] / ts if ts > 0 else float("nan")
        p_s = totals[f"m5_eat_safe_{prefix}"] / ss if ss > 0 else float("nan")
        results[f"eat_under_threat_ratio_{prefix}"] = (
            float(p_t) / max(float(p_s), EPS)
            if ts > 0 and ss > 0
            else float("nan")
        )
    results.update({k: int(v) for k, v in totals.items()})
    return results


def _resolve_continual_stage_config(config_arg: str, checkpoint_arg: str,
                                     quiet: bool = False) -> Optional[str]:
    """Resolve the environment config for a continual (multi-stage curriculum)
    checkpoint to its OWN stage, instead of always evaluating against stage 0.

    Background (see docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md,
    Finding E / L2): a continual run's CheckpointManager root (`models/`) always
    contains a stage-0 `config.yaml` PLUS per-stage `stage_{i:02d}_<name>.yaml`
    dumps and a `schedule.yaml` manifest (train.py ~line 592-605). Evaluating every
    checkpoint against `config.yaml` silently evaluates later-stage checkpoints in
    the wrong (stage-0) environment.

    Detection is conservative and backward-compatible: this only fires when
    (a) `checkpoint_arg`'s CheckpointManager root has a `schedule.yaml` (i.e. this
    is actually a continual run) AND (b) `config_arg` is exactly that run's own
    stage-0 `config.yaml` -- the common `--config <run>/models/config.yaml`
    invocation pattern that the diagnosis found buggy. An explicitly different
    `--config` (e.g. an out-of-distribution / generalization eval config) is left
    untouched; the caller's choice always wins. Non-continual runs (no
    `schedule.yaml`) return None and the caller uses `--config` exactly as before.

    The checkpoint's stage is read directly from the checkpoint's own saved
    payload (`ckpt_data['stage']`, train.py ~line 2428) rather than recomputed
    from the episode count and `episode_boundaries` -- the per-iteration
    stage-transition check means a checkpoint's *recorded* stage can lag one
    behind what the boundary alone would imply (confirmed empirically on a real
    continual checkpoint: episode 811 with boundary 800 was still saved as stage
    0, not the recomputed stage 1). Reading the checkpoint's own field is ground
    truth; recomputing from boundaries is not.
    """
    ckpt_path = Path(checkpoint_arg).resolve()
    try:
        explicit_step = int(ckpt_path.name)
        ckpt_root = ckpt_path.parent
    except ValueError:
        explicit_step = None
        ckpt_root = ckpt_path

    schedule_path = ckpt_root / "schedule.yaml"
    if not schedule_path.exists():
        return None  # not a continual run; unchanged behavior

    default_cfg_path = (ckpt_root / "config.yaml").resolve()
    if Path(config_arg).resolve() != default_cfg_path:
        return None  # caller explicitly chose a different config; leave it alone

    sched = Config.load_yaml(str(schedule_path))
    stage_names = sched.get_mandatory("continual.stage_names")

    restore_mngr = ocp.CheckpointManager(str(ckpt_root))
    step = explicit_step if explicit_step is not None else restore_mngr.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint steps found under {ckpt_root}.")

    _cpu_device = jax.local_devices()[0]

    def _cpu_restore_arg(_x):
        return ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=jax.sharding.SingleDeviceSharding(_cpu_device),
        )

    target = {"stage": 0}
    restore_args = jax.tree_util.tree_map(_cpu_restore_arg, target)
    try:
        restored = restore_mngr.restore(
            step,
            args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args,
                                         partial_restore=True),
        )
        stage_idx = int(restored["stage"])
    except Exception as e:
        raise ValueError(
            f"Continual run detected ({schedule_path}) but could not read the "
            f"'stage' field from checkpoint step {step} under {ckpt_root}: {e}. "
            f"Pass a --config other than the stage-0 config.yaml to bypass "
            f"stage auto-detection."
        ) from e

    if not (0 <= stage_idx < len(stage_names)):
        raise ValueError(
            f"Checkpoint step {step} reports stage index {stage_idx}, out of "
            f"range for {len(stage_names)} stages in {schedule_path}."
        )

    stage_cfg_path = ckpt_root / f"stage_{stage_idx:02d}_{stage_names[stage_idx]}.yaml"
    if not stage_cfg_path.exists():
        raise ValueError(
            f"Resolved stage config {stage_cfg_path} does not exist "
            f"(stage {stage_idx}: {stage_names[stage_idx]})."
        )

    if not quiet:
        print(f"[eval_rollout] Continual run detected at {ckpt_root}; checkpoint "
              f"step {step} belongs to stage {stage_idx} "
              f"('{stage_names[stage_idx]}'). Using stage config: {stage_cfg_path} "
              f"(overrides stage-0 {default_cfg_path})", flush=True)

    return str(stage_cfg_path)


def _parse_config_list(path: str):
    """Parse a `--config-list` file: one pending probe condition per line,
    tab-separated `<config_path>\\t<output_root>`. Blank lines and `#`-comment
    lines are skipped. Returns a list of (config_path, output_root) string tuples,
    in file order (order only affects which entry is "first" -- used to build
    the model/restore the checkpoint -- it does not affect per-condition output).
    """
    entries = []
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<config_path>\\t<output_root>', "
                    f"got: {line!r}"
                )
            cfg, out_root = parts
            entries.append((cfg, out_root))
    if not entries:
        raise ValueError(f"--config-list file {path} contains no entries.")
    return entries


def _load_dreamer_probe_env_cfg(env_config_path: str) -> Config:
    """Merged env Config for a Dreamer probe eval, mirroring
    `dreamer_srl_main.py`'s single-config path (L507-L520) and ported verbatim
    from `scripts/eval/dreamer_srl_probe_eval.py::_load_probe_env_cfg`.

    Deliberately NOT the same as the plain `load_env_config(config_path)` used
    for the rPPO branch's `params`/`bm_cfg` above: Dreamer's own driver always
    layers `get_default_config()` + `configs/{train,evaluation,visualization}/
    default.yaml` UNDER the probe config before resolving its `extends:` chain,
    so matching that chain exactly is required for restore/rollout parity with
    `dreamer_srl_probe_eval.py`.
    """
    from src.utils.config import get_default_config
    env_cfg = get_default_config()
    for rel in ('configs/train/default.yaml', 'configs/evaluation/default.yaml',
                'configs/visualization/default.yaml'):
        path = PROJECT_ROOT / rel
        if path.exists():
            env_cfg.merge(Config.load_yaml(str(path)))
    env_cfg.merge(load_env_config(env_config_path))
    return env_cfg


def _build_dreamer_restore_target(world_model, actor, critic, target_critic):
    """Full abstract target tree matching every key `save_checkpoint()` writes
    (src/algorithms/dreamer_srl/checkpoint.py:save_checkpoint), ported verbatim
    from `scripts/eval/dreamer_srl_probe_eval.py::_build_restore_target`.

    Values are REAL arrays built from the already-constructed (CPU-resident,
    when `--device cpu` forced the platform before agent construction) agent,
    so restoring into this target via `ocp.args.StandardRestore` makes orbax
    derive each leaf's destination sharding from ITS OWN device placement
    rather than the checkpoint's saved (GPU) sharding — this is what makes
    restoring a GPU-trained checkpoint onto CPU work. See the module docstring
    of `dreamer_srl_probe_eval.py` for the full topology-mismatch rationale.
    """
    import flax.nnx as nnx
    from src.algorithms.dreamer_srl.utils import moments_init

    moments = moments_init()
    moments_target = {k: v for k, v in moments.__dict__.items() if isinstance(v, jax.Array)}
    return {
        'world_model': nnx.state(world_model, nnx.Param),
        'actor': nnx.state(actor, nnx.Param),
        'critic': nnx.state(critic, nnx.Param),
        'target_critic': nnx.state(target_critic, nnx.Param),
        'key': jax.random.PRNGKey(0),
        'iter_num': jnp.array(0, dtype=jnp.int32),
        'policy_step': jnp.array(0, dtype=jnp.int32),
        'total_episodes_completed': jnp.array(0, dtype=jnp.int32),
        'cumulative_grad_steps': jnp.array(0, dtype=jnp.int32),
        'stage': jnp.array(0, dtype=jnp.int32),
        'moments': moments_target,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluation rollout for behavior-measure toolkit v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="Path to environment/training config YAML. Mutually exclusive "
                             "with --config-list.")
    parser.add_argument("--config-list", default=None,
                        help="Path to a file listing MULTIPLE probe conditions to evaluate "
                             "against the SAME checkpoint in one process: one condition per "
                             "line, tab-separated '<config_path>\\t<output_root>' (blank "
                             "lines and '#' comments skipped). The model is built and the "
                             "checkpoint restored ONCE, then every condition's rollout runs "
                             "in a loop, writing recordings under its own <output_root> "
                             "exactly as a standalone --config run would -- this is the "
                             "dwell-sweep perf path (avoids rebuilding+restoring the same "
                             "checkpoint once per condition). Mutually exclusive with "
                             "--config; --output-root is ignored (each line supplies its own "
                             "output root instead).")
    parser.add_argument("--agent_config", default=None,
                        help="Path to agent config YAML (used to infer agent type).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to orbax checkpoint directory. For rPPO: the "
                             "CheckpointManager root or a <root>/<step> dir. For Dreamer: "
                             "the checkpoint step dir, following the "
                             "<run_dir>/checkpoints/<episode> convention (episode parsed "
                             "from the basename, run_dir from two levels up). If "
                             "--checkpoint doesn't parse this way (e.g. you passed a bare "
                             "run_dir or its checkpoints/ dir), pass --episode explicitly.")
    parser.add_argument("--episode", type=int, default=None,
                        help="Dreamer checkpoint step to restore. Only needed when "
                             "--checkpoint does not follow the <run_dir>/checkpoints/<episode> "
                             "convention (i.e. its basename isn't an integer). Ignored for rPPO.")
    parser.add_argument("--output-root", default=None,
                        help="Output root dir. Defaults to behavior_measures.eval_output_root/<run_tag>/.")
    parser.add_argument("--eval-n-episodes", type=int, default=None,
                        help="Override behavior_measures.eval_n_episodes.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None,
                        help="Override behavior_measures.eval_seeds.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master RNG seed for the Dreamer rollout (dreamer_srl_eval_rollout "
                             "derives all per-episode reset keys from this ONE seed via "
                             "successive jax.random.split calls — unlike rPPO, which resets each "
                             "episode from its own entry in --eval-seeds). Ignored for rPPO. "
                             "Default 0 matches dreamer_srl_probe_eval.py's default, for parity.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="JAX device.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Emit .rec.gz recordings compatible with render_recordings.py.")
    parser.add_argument("--record-n-episodes", type=int, default=10,
                        help="Number of episodes to record (first N of --eval-n-episodes).")
    parser.add_argument("--batched", action="store_true", default=False,
                        help="Run ALL episodes as one vmapped batch + lax.scan (Tier 2 perf "
                             "path) instead of the legacy per-episode Python loop. Additive: "
                             "the legacy loop stays the default until parity is proven per "
                             "docs/develop/active/refactors/EVAL_ROLLOUT_BATCHING_PERF.md. "
                             "Requires eval_policy_mode == 'deterministic'.")
    args = parser.parse_args()

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    # --- Resolve --config vs --config-list into a list of (config_arg, output_root_arg) ---
    if args.config and args.config_list:
        parser.error("Specify exactly one of --config or --config-list, not both.")
    if not args.config and not args.config_list:
        parser.error("Specify one of --config or --config-list.")
    raw_entries = _parse_config_list(args.config_list) if args.config_list else [
        (args.config, args.output_root)
    ]

    # --- Checkpoint-derived identifiers (config-independent; needed by _resolve_entry) ---
    ckpt_path = Path(args.checkpoint).resolve()
    run_tag = _derive_run_tag(ckpt_path)
    # Use the step number from the checkpoint dir name if parseable, else "eval".
    try:
        _pct_label = str(int(ckpt_path.name))
    except ValueError:
        _pct_label = "eval"

    # --- Load configs ---
    # Finding E / L2 fix: for a continual (multi-stage curriculum) run whose
    # checkpoint is being evaluated via the default `--config <run>/models/config.yaml`
    # pattern, evaluate against the checkpoint's OWN stage config, not stage 0's.
    # See _resolve_continual_stage_config for the detection rule and rationale.
    # Applied PER ENTRY (a no-op for probe configs, which never match the default
    # stage-0 config.yaml path that triggers stage resolution).
    def _resolve_entry(config_arg: str, output_root_arg):
        """Load one probe condition's env config + every field the ORIGINAL
        single-`--config` flow computed before branching on agent_type. Called once
        per `--config-list` line (or once, for the plain `--config` path) -- this is
        what makes the single-`--config` path's output byte-identical to before:
        with exactly one entry, this reproduces the old top-level code verbatim.
        """
        stage_cfg = _resolve_continual_stage_config(config_arg, args.checkpoint,
                                                      quiet=args.quiet)
        config_path_i = stage_cfg if stage_cfg is not None else config_arg
        config_i = load_env_config(config_path_i)  # honours `extends:` if present
        bm_cfg_i = load_behavior_measure_cfg(config_i)
        if bm_cfg_i is None:
            print(f"WARNING: behavior_measures block absent in config {config_arg}; "
                  "using defaults for eval.", flush=True)
            # Construct a minimal BM cfg for offline use
            from src.environment.config_loader import BehaviorMeasureCfg
            bm_cfg_i = BehaviorMeasureCfg(
                enabled=True,
                cue_radius=3.0,
                obs_window=5,
                eval_n_episodes=args.eval_n_episodes or 10,
                eval_seeds=tuple(range(args.eval_n_episodes or 10)),
                eval_policy_mode="deterministic",
                eval_max_steps=500,
                eval_obs_noise="training",
                motif_window_K=7,
                motif_features=(
                    "net_displacement", "path_length", "threat_distance_change_rate",
                    "min_threat_distance", "bush_occupancy_fraction", "eat_events_per_window",
                    "action_entropy", "mode_action_fraction", "stay_in_place_fraction",
                    "drive_injury_change",
                ),
                motif_kmeans_k=6,
                motif_kmeans_seed=42,
                motif_standardise="zscore_pooled",
                eval_output_root="results/eval",
            )

        n_eps_i = args.eval_n_episodes or bm_cfg_i.eval_n_episodes
        seeds_i = list(args.eval_seeds) if args.eval_seeds else list(bm_cfg_i.eval_seeds)
        if len(seeds_i) < n_eps_i:
            # Extend seeds if fewer than n_eps provided
            extra = [seeds_i[-1] + i + 1 for i in range(n_eps_i - len(seeds_i))]
            seeds_i = seeds_i + extra
        seeds_i = seeds_i[:n_eps_i]

        if args.batched and bm_cfg_i.eval_policy_mode != "deterministic":
            raise NotImplementedError(
                "--batched only supports eval_policy_mode == 'deterministic' (argmax). "
                f"Got '{bm_cfg_i.eval_policy_mode}' for config {config_arg!r}. The batched "
                "path's parity argument (docs/develop/active/refactors/"
                "EVAL_ROLLOUT_BATCHING_PERF.md) relies on the eval policy being "
                "deterministic argmax with the RNG key unused; stochastic batched "
                "sampling is out of scope for this change. Drop --batched to use the "
                "legacy per-episode loop."
            )

        params_i = load_env_params(config_i)
        _assert_eval_obs_noise_supported(bm_cfg_i)
        max_steps_i = int(config_i.get_mandatory("environment.max_steps"))

        # --- Determine output dir ---
        out_root_i = Path(output_root_arg) if output_root_arg else Path(bm_cfg_i.eval_output_root)
        out_dir_i = out_root_i / run_tag / ckpt_path.name
        out_dir_i.mkdir(parents=True, exist_ok=True)
        (out_dir_i / "episodes").mkdir(exist_ok=True)
        (out_dir_i / "windows").mkdir(exist_ok=True)

        # --- Recording setup (--record) ---
        rec_dir_i = out_dir_i / "recordings" / _pct_label
        if args.record:
            rec_dir_i.mkdir(parents=True, exist_ok=True)

        return {
            "config_arg": config_arg, "config_path": config_path_i, "config": config_i,
            "bm_cfg": bm_cfg_i, "n_eps": n_eps_i, "seeds": seeds_i, "params": params_i,
            "max_steps": max_steps_i, "out_dir": out_dir_i, "rec_dir": rec_dir_i,
        }

    entries = [_resolve_entry(cfg, out_root_arg) for cfg, out_root_arg in raw_entries]

    if not args.quiet:
        print(f"[eval_rollout] Config: {args.config if args.config else args.config_list}", flush=True)
        print(f"[eval_rollout] Checkpoint: {args.checkpoint}", flush=True)
        if len(entries) == 1:
            print(f"[eval_rollout] Output: {entries[0]['out_dir']}", flush=True)
        else:
            print(f"[eval_rollout] {len(entries)} condition(s) from --config-list "
                  "(model + checkpoint built/restored ONCE, looped per condition)", flush=True)
        print(f"[eval_rollout] Episodes: {entries[0]['n_eps']}, "
              f"seeds: {entries[0]['seeds'][:5]}{'...' if entries[0]['n_eps'] > 5 else ''}",
              flush=True)

    # --- Load policy ---
    # Detect agent type from checkpoint directory structure or agent_config
    agent_type = "rppo"
    if args.agent_config:
        if "dreamer" in args.agent_config.lower():
            agent_type = "dreamer"
    elif (ckpt_path / "wm").exists() or (ckpt_path.parent / "wm").exists():
        agent_type = "dreamer"

    if not args.quiet:
        print(f"[eval_rollout] Agent type: {agent_type}", flush=True)

    if agent_type == "rppo":
        import flax.nnx as nnx
        from src.models.recurrent_ppo_network import ActorCriticRNN, get_action_and_value_nnx

        # --- Locate agent config ---
        # Prefer an explicit --agent_config; fall back to config.yaml saved alongside
        # the checkpoint (written by train.py at checkpoint-save time).
        # The config.yaml lives in the CheckpointManager root directory; check
        # both ckpt_path itself and its parent (handles step-dir vs. root-dir arg).
        if args.agent_config:
            agent_config = Config.load_yaml(args.agent_config)
        else:
            saved_cfg = ckpt_path / "config.yaml"
            if not saved_cfg.exists():
                saved_cfg = ckpt_path.parent / "config.yaml"
            if saved_cfg.exists():
                agent_config = Config.load_yaml(str(saved_cfg))
            else:
                raise FileNotFoundError(
                    f"No --agent_config provided and no config.yaml found near "
                    f"checkpoint: {ckpt_path}. Pass --agent_config explicitly."
                )

        # --- Derive model dimensions from env params ---
        # Built ONCE from the FIRST entry's params. With --config-list, every later
        # entry is checked below (right before its rollout) for the SAME obs/action
        # shape -- the multi-config perf path requires all conditions to share the
        # checkpoint's architecture (see --config-list help text).
        params = entries[0]["params"]
        obs_breakdown = get_observation_breakdown(params)
        input_dim  = sum(obs_breakdown.values())
        action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)

        # --- Read model hyper-params from saved agent config ---
        hidden_size       = agent_config.get_mandatory("agent.hidden_size")
        rnn_type          = agent_config.get_mandatory("agent.rnn_type")
        activation        = agent_config.get_mandatory("agent.activation")
        encoding_config   = agent_config.to_dict().get("agent", {})
        modulation_config = agent_config.get("agent.modulation")
        if modulation_config is not None and modulation_config.get("type") is None:
            modulation_config = None

        if not args.quiet:
            print(
                f"[eval_rollout] Building ActorCriticRNN: input_dim={input_dim}, "
                f"action_dim={action_dim}, hidden={hidden_size}, rnn={rnn_type}",
                flush=True,
            )

        # --- Build a fresh model with the same architecture ---
        init_key = jax.random.PRNGKey(0)
        model = ActorCriticRNN(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            rngs=nnx.Rngs(init_key),
            rnn_type=rnn_type,
            activation=activation,
            modulation_config=modulation_config,
            observation_breakdown=obs_breakdown,
            encoding_config=encoding_config,
        )

        # --- Restore checkpoint weights (mirrors train.py lines ~1327-1368) ---
        # Auto-detect: if ckpt_path is a step directory (integer name), open its
        # parent as the CheckpointManager root and use that step; otherwise treat
        # ckpt_path as the root and let orbax pick the latest step.
        try:
            _explicit_step = int(ckpt_path.name)
            _ckpt_root     = ckpt_path.parent
        except ValueError:
            _explicit_step = None
            _ckpt_root     = ckpt_path

        restore_mngr = ocp.CheckpointManager(str(_ckpt_root))
        step = _explicit_step if _explicit_step is not None else restore_mngr.latest_step()
        if not args.quiet:
            print(f"[eval_rollout] Restoring RPPO checkpoint at step {step}", flush=True)

        # Build restore_args with explicit CPU sharding so orbax can restore a
        # GPU-saved checkpoint to CPU (PyTreeRestore() without a target fails when
        # the checkpoint has no sharding metadata for the target device).
        current_model_state = nnx.state(model)
        _cpu_device = jax.local_devices()[0]

        def _cpu_restore_arg(_x):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                sharding=jax.sharding.SingleDeviceSharding(_cpu_device),
            )

        _restore_args = jax.tree_util.tree_map(_cpu_restore_arg, current_model_state)
        restored = restore_mngr.restore(
            step,
            args=ocp.args.PyTreeRestore(
                item={"model": current_model_state},
                restore_args={"model": _restore_args},
                partial_restore=True,
            ),
        )

        restored_model_state = restored["model"]

        flat_restored, _        = jax.tree_util.tree_flatten_with_path(restored_model_state)
        flat_current,  tree_def = jax.tree_util.tree_flatten_with_path(current_model_state)

        restored_dict = {str(k): v for k, v in flat_restored}

        # Strict architecture check (same as train.py)
        mismatches = []
        for k, cur_v in flat_current:
            k_str = str(k)
            if k_str not in restored_dict:
                mismatches.append(
                    f"Missing in checkpoint: '{k_str}' "
                    f"(current shape {getattr(cur_v, 'shape', '?')})"
                )
            elif getattr(cur_v, "shape", None) != getattr(restored_dict[k_str], "shape", None):
                mismatches.append(
                    f"Shape mismatch '{k_str}': ckpt "
                    f"{getattr(restored_dict[k_str], 'shape', '?')} "
                    f"vs model {getattr(cur_v, 'shape', '?')}"
                )
        if mismatches:
            raise ValueError(
                "Architecture mismatch between checkpoint and env params:\n"
                + "\n".join(f"  {m}" for m in mismatches)
            )

        valid_flat  = [restored_dict[str(k)] for k, _ in flat_current]
        valid_tree  = jax.tree_util.tree_unflatten(tree_def, valid_flat)
        nnx.update(model, valid_tree)

        if not args.quiet:
            print(f"[eval_rollout] RPPO model restored from step {step}.", flush=True)

        # --- Policy closure factory (per entry: `params` differs per probe condition
        # even though obs/action SHAPE is shared -- entity positions/content differ) ---
        h_init = model.initial_state(batch_size=None)  # no batch dim for single-agent eval

        def _make_policy_fn(params_i):
            def policy_fn(state, carry, key, deterministic=True):
                obs = get_observation(state, params_i)   # (input_dim,)
                if carry is None:
                    carry = h_init
                action, _log_prob, _value, h_new, _mod = get_action_and_value_nnx(
                    model, obs, carry, key=key if not deterministic else None,
                    eval_mode=deterministic,
                )
                return action, h_new
            return policy_fn

        def _run_rppo_entry(entry, policy_fn):
            """Run the recording setup + episodes + threat-onset index + online-replay
            cross-check + metadata.json for ONE probe condition. Mirrors, verbatim, what
            the original single-`--config` flow did after building/restoring the model
            (this function's body IS that trailing block, just parameterized per entry
            so it can be called once for `--config`, or once per line for
            `--config-list` reusing the SAME already-built/restored `model`)."""
            config = entry["config"]; config_path_e = entry["config_path"]
            bm_cfg = entry["bm_cfg"]; params_e = entry["params"]
            max_steps = entry["max_steps"]; n_eps = entry["n_eps"]; seeds = entry["seeds"]
            out_dir = entry["out_dir"]; rec_dir = entry["rec_dir"]

            # --- Recording metadata (written once before the loop) ---
            if args.record:
                from src.utils.eval_recording import write_run_meta
                _action_map = ["Up", "Right", "Down", "Left"]
                if params_e.rest_action_enabled:
                    _action_map.append("Rest")
                if params_e.eat_action_enabled:
                    _action_map.append("Eat")
                _icon_config = config.get("visualization.icons", None)
                write_run_meta(
                    rec_dir, params_e, _icon_config, _action_map, entry["config_arg"],
                    extras={"checkpoint_pct": _pct_label},
                )
                if not args.quiet:
                    print(f"[eval_rollout] Recording to: {rec_dir}", flush=True)

            # --- Run episodes ---
            t_start = time.time()
            rec_n_eps = args.record_n_episodes if args.record else 0
            episodes = []
            if args.batched:
                # Tier 2 — all n_eps episodes as one vmapped batch + lax.scan over max_steps.
                # See _run_episodes_batched docstring + the plan doc for the parity argument.
                if not args.quiet:
                    print(f"[eval_rollout] Running batched rollout: {n_eps} episodes as one "
                          f"vmapped batch (max_steps={max_steps})...", flush=True)
                episodes, recorders = _run_episodes_batched(
                    params_e, model, seeds, max_steps,
                    record=args.record, record_n_episodes=rec_n_eps,
                )
                for ep_idx, ep_data in enumerate(episodes):
                    ep_data["seed"] = np.int32(seeds[ep_idx])
                    np.savez_compressed(
                        out_dir / "episodes" / f"{ep_idx:04d}.npz",
                        **ep_data,
                    )
                    recorder = recorders[ep_idx]
                    if recorder is not None:
                        out_path = rec_dir / f"episode_{ep_idx:06d}.rec.gz"
                        recorder.write(out_path)
                        if not args.quiet:
                            print(f"[eval_rollout] Wrote recording: {out_path}", flush=True)
            else:
                for ep_idx in range(n_eps):
                    key = jax.random.PRNGKey(seeds[ep_idx])
                    if not args.quiet:
                        print(f"[eval_rollout] Episode {ep_idx + 1}/{n_eps} (seed={seeds[ep_idx]})...", end="\r", flush=True)

                    # When recording, use the recorder-aware variant; otherwise use the fast path.
                    should_record = args.record and ep_idx < rec_n_eps
                    if should_record:
                        ep_data, recorder = _run_episode_with_recording(
                            params_e, policy_fn, key, max_steps=max_steps,
                            deterministic=(bm_cfg.eval_policy_mode == "deterministic"),
                            episode_index=ep_idx,
                            seed=seeds[ep_idx],
                        )
                    else:
                        ep_data = _run_episode(
                            params_e, policy_fn, key, max_steps=max_steps,
                            deterministic=(bm_cfg.eval_policy_mode == "deterministic"),
                        )
                        recorder = None

                    ep_data["seed"] = np.int32(seeds[ep_idx])
                    episodes.append(ep_data)

                    # Save .npz (unchanged schema)
                    np.savez_compressed(
                        out_dir / "episodes" / f"{ep_idx:04d}.npz",
                        **ep_data,
                    )

                    # Write recording file if applicable
                    if should_record and recorder is not None:
                        out_path = rec_dir / f"episode_{ep_idx:06d}.rec.gz"
                        recorder.write(out_path)
                        if not args.quiet:
                            print(f"[eval_rollout] Wrote recording: {out_path}", flush=True)

            if not args.quiet:
                print(f"\n[eval_rollout] {n_eps} episodes done in {time.time() - t_start:.1f}s", flush=True)

            # --- Threat-onset index ---
            onsets = _detect_threat_onsets(episodes, bm_cfg)
            if not args.quiet:
                print(f"[eval_rollout] {len(onsets)} threat-onset events detected.", flush=True)

            try:
                import pandas as pd
                df_onsets = pd.DataFrame(onsets) if onsets else pd.DataFrame(
                    columns=["episode_idx", "t_star", "triggering_class", "triggering_tag",
                             "window_start_t", "window_end_t"]
                )
                df_onsets.to_parquet(out_dir / "windows" / "threat_onsets.parquet", index=False)
            except ImportError:
                # Fallback: JSON
                with open(out_dir / "windows" / "threat_onsets.json", "w") as f:
                    json.dump(onsets, f, indent=2)
                if not args.quiet:
                    print("[eval_rollout] pyarrow/pandas not installed; wrote threat_onsets.json instead.", flush=True)

            # --- Online replay (sanity cross-check) ---
            replay = _compute_online_replay(episodes, bm_cfg)
            with open(out_dir / "online_replay.json", "w") as f:
                json.dump(replay, f, indent=2, default=lambda x: None if (isinstance(x, float) and x != x) else x)

            # --- Metadata ---
            metadata = {
                "config": entry["config_arg"],
                "config_resolved": config_path_e,
                "checkpoint": args.checkpoint,
                "agent_type": agent_type,
                "rollout_mode": "batched" if args.batched else "legacy",
                "n_episodes": n_eps,
                "seeds": seeds,
                "eval_policy_mode": bm_cfg.eval_policy_mode,
                "eval_obs_noise": bm_cfg.eval_obs_noise,
                "cue_radius": bm_cfg.cue_radius,
                "obs_window": bm_cfg.obs_window,
                "motif_window_K": bm_cfg.motif_window_K,
                "wall_clock_s": time.time() - t_start,
                "jax_version": jax.__version__,
                "git_commit": _get_git_commit(),
                "jax_devices": [str(d) for d in jax.devices()],
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            if not args.quiet:
                print(f"[eval_rollout] Results saved to: {out_dir}", flush=True)
                for k, v in replay.items():
                    if isinstance(v, float) and v != v:
                        print(f"  {k}: NaN")
                    elif isinstance(v, float):
                        print(f"  {k}: {v:.4f}")

        # --- Loop over every entry (1 for plain --config; N for --config-list),
        # reusing the SAME model/checkpoint restored above. Each entry's episodes
        # list is local to this iteration and goes out of scope (and is GC-able)
        # before the next entry starts -- recordings are written to disk per
        # entry, not accumulated in memory across entries. ---
        for idx, entry in enumerate(entries):
            if idx > 0:
                obs_breakdown_i = get_observation_breakdown(entry["params"])
                action_dim_i = (4 + int(entry["params"].rest_action_enabled)
                                 + int(entry["params"].eat_action_enabled))
                if obs_breakdown_i != obs_breakdown or action_dim_i != action_dim:
                    raise ValueError(
                        f"--config-list entry {entry['config_arg']!r} has a different "
                        f"observation/action shape than the first entry "
                        f"({entries[0]['config_arg']!r}) -- obs_breakdown {obs_breakdown_i} "
                        f"vs {obs_breakdown}, action_dim {action_dim_i} vs {action_dim}. "
                        "The multi-config path requires every condition to share the "
                        "checkpoint's model architecture; run this condition through its "
                        "own single --config invocation instead."
                    )
            _run_rppo_entry(entry, _make_policy_fn(entry["params"]))

    elif agent_type == "dreamer":
        import flax.nnx as nnx
        from src.algorithms.dreamer_srl.agent import build_agent
        from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout
        if args.batched:
            from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout_batched

        # --- Checkpoint-arg reconciliation: <run_dir>/checkpoints/<episode> ---
        # dreamer_srl's restore needs (run_dir, episode:int), not a single path,
        # so parse both out of --checkpoint. `_pct_label` (computed above from
        # `ckpt_path.name`) already tells us whether the basename parsed as an
        # int; reuse it instead of re-deriving.
        if _pct_label == "eval":
            if args.episode is None:
                raise ValueError(
                    f"--checkpoint '{args.checkpoint}' does not follow the "
                    f"<run_dir>/checkpoints/<episode> convention (basename "
                    f"'{ckpt_path.name}' is not an integer episode). Pass --episode "
                    f"explicitly, with --checkpoint pointing at either <run_dir> or "
                    f"<run_dir>/checkpoints."
                )
            dreamer_episode = args.episode
        else:
            dreamer_episode = int(_pct_label)

        if ckpt_path.name == "checkpoints":
            dreamer_run_dir = ckpt_path.parent
        elif ckpt_path.parent.name == "checkpoints":
            dreamer_run_dir = ckpt_path.parent.parent
        else:
            dreamer_run_dir = ckpt_path  # --checkpoint given as the run_dir itself

        if not args.agent_config:
            raise FileNotFoundError(
                "Dreamer eval requires --agent_config (the run's models/agent_config.yaml)."
            )
        dreamer_agent_cfg = Config.load_yaml(args.agent_config)

        # --- Env config: the FULL merge chain dreamer_srl_main.py uses, not the
        # plain load_env_config() already used above for params/bm_cfg. --- Computed
        # PER ENTRY (each probe condition merges its own config), but the agent is
        # built + restored only ONCE below, from the FIRST entry's obs_dim/action_dim
        # -- every later entry is shape-checked against it (see the loop after restore).
        dreamer_envs = []
        for entry in entries:
            d_env_cfg = _load_dreamer_probe_env_cfg(entry["config_path"])
            d_env_params = load_env_params(d_env_cfg)
            d_obs_breakdown = get_observation_breakdown(d_env_params)
            d_obs_dim = sum(d_obs_breakdown.values())
            d_action_dim = (4 + int(d_env_params.rest_action_enabled)
                             + int(d_env_params.eat_action_enabled))
            dreamer_envs.append({
                "env_cfg": d_env_cfg, "env_params": d_env_params,
                "obs_breakdown": d_obs_breakdown, "obs_dim": d_obs_dim,
                "action_dim": d_action_dim,
            })

        obs_dim = dreamer_envs[0]["obs_dim"]
        action_dim_d = dreamer_envs[0]["action_dim"]
        for i, de in enumerate(dreamer_envs[1:], start=1):
            if de["obs_breakdown"] != dreamer_envs[0]["obs_breakdown"] or de["action_dim"] != action_dim_d:
                raise ValueError(
                    f"--config-list entry {entries[i]['config_arg']!r} has a different "
                    f"observation/action shape than the first entry "
                    f"({entries[0]['config_arg']!r}) -- obs_dim {de['obs_dim']} vs {obs_dim}, "
                    f"action_dim {de['action_dim']} vs {action_dim_d}. The multi-config "
                    "path requires every condition to share the checkpoint's agent "
                    "architecture; run this condition through its own single --config "
                    "invocation instead."
                )

        if not args.quiet:
            print(f"[eval_rollout] Building dreamer_srl agent: obs_dim={obs_dim}, "
                  f"action_dim={action_dim_d}", flush=True)

        rngs = nnx.Rngs(args.seed)
        world_model, actor, critic, target_critic = build_agent(
            obs_dim=obs_dim, action_dim=action_dim_d, cfg=dreamer_agent_cfg.to_dict(), rngs=rngs,
        )

        # --- Restore checkpoint weights (topology-agnostic; see
        # _build_dreamer_restore_target docstring) ---
        # Eval only ever needs the model params, never the optimizer state --
        # but training checkpoints saved after the --load-checkpoint resume
        # feature was added (checkpoint.py:save_checkpoint) ALSO carry
        # wm_opt/actor_opt/critic_opt (Adam state). A `StandardRestore` demands
        # the target tree match the saved tree exactly, so a model-only target
        # fails structurally against those full-state checkpoints. Use
        # `PyTreeRestore(..., partial_restore=True)` instead: it restores only
        # the keys present in `restore_target` and silently ignores any extra
        # keys in the checkpoint (optimizer subtrees included), so the same
        # code path works for both model-only and full-training-state
        # checkpoints. This is the same pattern already used for the rPPO
        # restore (~L1148 above) and `scripts/dreamer/visualize_dream.py`.
        # A plain `ocp.CheckpointManager(<checkpoints dir>)` (no `checkpointers=`
        # kwarg) is required here -- `make_checkpoint_manager()` binds the
        # manager to `StandardCheckpointer` only, which registers just
        # Standard{Save,Restore} handlers and rejects `PyTreeRestore` args.
        if not args.quiet:
            print(f"[eval_rollout] Restoring Dreamer checkpoint step={dreamer_episode} from "
                  f"{dreamer_run_dir}/checkpoints/", flush=True)

        manager = ocp.CheckpointManager(
            os.path.abspath(os.path.join(str(dreamer_run_dir), "checkpoints")))
        restore_target = _build_dreamer_restore_target(world_model, actor, critic, target_critic)
        _cpu_device = jax.local_devices()[0]

        def _cpu_restore_arg(_x):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                sharding=jax.sharding.SingleDeviceSharding(_cpu_device),
            )

        restore_args = jax.tree_util.tree_map(_cpu_restore_arg, restore_target)
        try:
            restored = manager.restore(
                dreamer_episode,
                args=ocp.args.PyTreeRestore(item=restore_target, restore_args=restore_args,
                                             partial_restore=True),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to restore Dreamer checkpoint step={dreamer_episode} from "
                f"{dreamer_run_dir}/checkpoints/ into the agent built from --agent_config "
                f"(obs_dim={obs_dim}, action_dim={action_dim_d}). This usually means either "
                f"the step doesn't exist (ls {dreamer_run_dir}/checkpoints/) or --config "
                f"produces a different obs_dim/action_dim than the environment the "
                f"checkpoint was trained on. Original error: {e}"
            ) from e
        if restored is None:
            raise RuntimeError(
                f"manager.restore returned None for step={dreamer_episode} under "
                f"{dreamer_run_dir}/checkpoints/ — confirm the step exists."
            )
        nnx.update(world_model, restored["world_model"])
        nnx.update(actor, restored["actor"])
        if not args.quiet:
            print(f"[eval_rollout] Dreamer model restored from step {dreamer_episode}.", flush=True)

        # --- Rollout + record, ONE loop iteration per probe condition (writes the
        # same .rec.gz format as the rPPO --record path, under the SAME
        # out_dir/recordings/<pct> layout). Reuses the SAME world_model/actor
        # restored once above -- no rebuild/restore per condition. Each condition
        # calls dreamer_srl_eval_rollout(_batched) fresh with seed=args.seed, so it
        # gets the SAME per-episode key sequence a standalone single-`--config` call
        # for that condition would produce (the function reseeds from `seed` inside,
        # it does not carry state across calls).
        # --batched: all n_eps episodes as one vmapped batch (Tier 2 perf path,
        # ~10-30x speedup for offline probe sweeps). Uses an INDEPENDENT
        # per-episode-key RNG convention, different from the legacy sequential-
        # master-key single-env path -- same seed reproduces the same behavior
        # DISTRIBUTION, not bit-identical trajectories. See
        # dreamer_srl_eval_rollout_batched's docstring for the full argument.
        for entry, de in zip(entries, dreamer_envs):
            n_eps = entry["n_eps"]; out_dir = entry["out_dir"]
            t_start = time.time()
            if args.batched:
                if not args.quiet:
                    print(f"[eval_rollout] Running batched Dreamer rollout: {n_eps} "
                          f"episodes as one vmapped batch...", flush=True)
                result = dreamer_srl_eval_rollout_batched(
                    world_model=world_model,
                    actor=actor,
                    env_params=de["env_params"],
                    config=de["env_cfg"],
                    num_episodes=n_eps,
                    seed=args.seed,
                    results_dir=str(out_dir),
                    checkpoint_pct=dreamer_episode,
                    render_video=args.record,
                    quiet=args.quiet,
                )
            else:
                result = dreamer_srl_eval_rollout(
                    world_model=world_model,
                    actor=actor,
                    env_params=de["env_params"],
                    config=de["env_cfg"],
                    num_episodes=n_eps,
                    seed=args.seed,
                    results_dir=str(out_dir),
                    checkpoint_pct=dreamer_episode,
                    render_video=args.record,
                    quiet=args.quiet,
                )
            wall_clock_s = time.time() - t_start

            if not args.quiet:
                print(f"[eval_rollout] {n_eps} episodes done in {wall_clock_s:.1f}s", flush=True)
                print(f"[eval_rollout] mean_reward={result['mean_reward']:.3f} "
                      f"mean_length={result['mean_length']:.1f}", flush=True)
                if result["recordings_dir"]:
                    print(f"[eval_rollout] Wrote recordings under: {result['recordings_dir']}", flush=True)

            # Dreamer's rollout writes its own recordings/run_meta (via
            # dreamer_srl_eval_rollout -> write_run_meta) and doesn't produce the
            # rPPO branch's per-step arrays needed for episodes/*.npz, the
            # threat-onset index, or the online-replay cross-check — those are
            # rPPO-specific behavior-measure-toolkit-v1 outputs. Write a Dreamer-
            # appropriate metadata.json and move to the next entry (if any).
            metadata = {
                "config": entry["config_arg"],
                "config_resolved": entry["config_path"],
                "checkpoint": args.checkpoint,
                "agent_type": agent_type,
                "rollout_mode": "dreamer_batched" if args.batched else "dreamer_single_env",
                "n_episodes": n_eps,
                "seeds": entry["seeds"],
                "mean_reward": result["mean_reward"],
                "mean_length": result["mean_length"],
                "wall_clock_s": wall_clock_s,
                "jax_version": jax.__version__,
                "git_commit": _get_git_commit(),
                "jax_devices": [str(d) for d in jax.devices()],
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            if not args.quiet:
                print(f"[eval_rollout] Results saved to: {out_dir}", flush=True)
        return

    else:
        raise NotImplementedError(f"Agent type '{agent_type}' not yet supported by eval_rollout.py.")


if __name__ == "__main__":
    main()
