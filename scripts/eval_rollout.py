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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    while step < max_steps and not done:
        obs = state  # obs = full state (policy sees sensory obs internally)
        action, carry = policy_fn(obs, carry, rng_key, deterministic=deterministic)
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
    while step < max_steps and not done:
        action, carry = policy_fn(state, carry, rng_key, deterministic=deterministic)
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

                # M1 — age FIRST (age-first ordering matches train.py)
                if cand_age >= 0:
                    cand_age += 1
                    if cand_age >= K:
                        totals[f"m1_candidates_{prefix}"] += 1
                        if steps_since_eat >= K:
                            totals[f"m1_interrupted_{prefix}"] += 1
                        cand_age = -1

                steps_since_eat = 0 if ate[t] else steps_since_eat + 1

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


def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluation rollout for behavior-measure toolkit v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to environment/training config YAML.")
    parser.add_argument("--agent_config", default=None,
                        help="Path to agent config YAML (used to infer agent type).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to orbax checkpoint directory.")
    parser.add_argument("--output-root", default=None,
                        help="Output root dir. Defaults to behavior_measures.eval_output_root/<run_tag>/.")
    parser.add_argument("--eval-n-episodes", type=int, default=None,
                        help="Override behavior_measures.eval_n_episodes.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None,
                        help="Override behavior_measures.eval_seeds.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="JAX device.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Emit .rec.gz recordings compatible with render_recordings.py.")
    parser.add_argument("--record-n-episodes", type=int, default=10,
                        help="Number of episodes to record (first N of --eval-n-episodes).")
    args = parser.parse_args()

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    # --- Load configs ---
    config = load_env_config(args.config)  # honours `extends:` if present; standalone otherwise
    bm_cfg = load_behavior_measure_cfg(config)
    if bm_cfg is None:
        print("WARNING: behavior_measures block absent in config; using defaults for eval.", flush=True)
        # Construct a minimal BM cfg for offline use
        from src.environment.config_loader import BehaviorMeasureCfg
        bm_cfg = BehaviorMeasureCfg(
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

    n_eps = args.eval_n_episodes or bm_cfg.eval_n_episodes
    seeds = list(args.eval_seeds) if args.eval_seeds else list(bm_cfg.eval_seeds)
    if len(seeds) < n_eps:
        # Extend seeds if fewer than n_eps provided
        extra = [seeds[-1] + i + 1 for i in range(n_eps - len(seeds))]
        seeds = seeds + extra
    seeds = seeds[:n_eps]

    params = load_env_params(config)
    max_steps = int(config.get_mandatory("environment.max_steps"))

    # --- Derive checkpoint_pct label (directory-name-safe) ---
    # Use the step number from the checkpoint dir name if parseable, else "eval".
    _ckpt_name_for_pct = Path(args.checkpoint).resolve().name
    try:
        _pct_label = str(int(_ckpt_name_for_pct))
    except ValueError:
        _pct_label = "eval"

    # --- Determine output dir ---
    ckpt_path = Path(args.checkpoint).resolve()
    run_tag = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else ckpt_path.parent.parent.name
    out_root = Path(args.output_root) if args.output_root else Path(bm_cfg.eval_output_root)
    out_dir = out_root / run_tag / ckpt_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    (out_dir / "windows").mkdir(exist_ok=True)

    # --- Recording setup (--record) ---
    rec_n_eps = args.record_n_episodes if args.record else 0
    rec_dir = out_dir / "recordings" / _pct_label
    if args.record:
        rec_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"[eval_rollout] Config: {args.config}", flush=True)
        print(f"[eval_rollout] Checkpoint: {args.checkpoint}", flush=True)
        print(f"[eval_rollout] Output: {out_dir}", flush=True)
        print(f"[eval_rollout] Episodes: {n_eps}, seeds: {seeds[:5]}{'...' if n_eps > 5 else ''}", flush=True)

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

        # --- Policy closure ---
        h_init = model.initial_state(batch_size=None)  # no batch dim for single-agent eval

        def policy_fn(state, carry, key, deterministic=True):
            obs = get_observation(state, params)   # (input_dim,)
            if carry is None:
                carry = h_init
            action, _log_prob, _value, h_new, _mod = get_action_and_value_nnx(
                model, obs, carry, key=key if not deterministic else None,
                eval_mode=deterministic,
            )
            return action, h_new
    else:
        raise NotImplementedError(f"Agent type '{agent_type}' not yet supported by eval_rollout.py. "
                                  f"Implement DreamerV3 checkpoint loading and add here.")

    # --- Recording metadata (written once before the loop) ---
    if args.record:
        from src.utils.eval_recording import write_run_meta
        _action_map = ["Up", "Right", "Down", "Left"]
        if params.rest_action_enabled:
            _action_map.append("Rest")
        if params.eat_action_enabled:
            _action_map.append("Eat")
        _icon_config = config.get("visualization.icons", None)
        write_run_meta(
            rec_dir, params, _icon_config, _action_map, args.config,
            extras={"checkpoint_pct": _pct_label},
        )
        if not args.quiet:
            print(f"[eval_rollout] Recording to: {rec_dir}", flush=True)

    # --- Run episodes ---
    t_start = time.time()
    episodes = []
    for ep_idx in range(n_eps):
        key = jax.random.PRNGKey(seeds[ep_idx])
        if not args.quiet:
            print(f"[eval_rollout] Episode {ep_idx + 1}/{n_eps} (seed={seeds[ep_idx]})...", end="\r", flush=True)

        # When recording, use the recorder-aware variant; otherwise use the fast path.
        should_record = args.record and ep_idx < rec_n_eps
        if should_record:
            ep_data, recorder = _run_episode_with_recording(
                params, policy_fn, key, max_steps=max_steps,
                deterministic=(bm_cfg.eval_policy_mode == "deterministic"),
                episode_index=ep_idx,
                seed=seeds[ep_idx],
            )
        else:
            ep_data = _run_episode(
                params, policy_fn, key, max_steps=max_steps,
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
        "config": args.config,
        "checkpoint": args.checkpoint,
        "agent_type": agent_type,
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


if __name__ == "__main__":
    main()
