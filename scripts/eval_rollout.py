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

from src.environment.config_loader import Config, load_env_params, load_behavior_measure_cfg
from src.environment.core import jax_reset, jax_step


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


def _load_rppo_policy(checkpoint_path: str, device: str):
    """Load RecurrentPPO policy from an orbax checkpoint directory.

    Returns (trainer, key) where trainer.agent can be called for inference.
    """
    from src.environment.config_loader import Config
    # We import the trainer lazily to avoid device-init on import
    import importlib
    rppo_mod = importlib.import_module("src.models.recurrent_ppo_trainer")
    RecurrentPPOTrainer = rppo_mod.RecurrentPPOTrainer

    # Load the saved config alongside the checkpoint
    ckpt_dir = Path(checkpoint_path)
    config_path = ckpt_dir.parent / "config.yaml"
    if not config_path.exists():
        # Try grandparent (results/<run>/checkpoints/<step>/ layout)
        config_path = ckpt_dir.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml near checkpoint: {checkpoint_path}")

    agent_cfg = Config.load_yaml(str(config_path))
    return RecurrentPPOTrainer, agent_cfg, ckpt_dir


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
    args = parser.parse_args()

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    # --- Load configs ---
    config = Config.load_yaml(args.config)
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

    # --- Determine output dir ---
    ckpt_path = Path(args.checkpoint).resolve()
    run_tag = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else ckpt_path.parent.parent.name
    out_root = Path(args.output_root) if args.output_root else Path(bm_cfg.eval_output_root)
    out_dir = out_root / run_tag / ckpt_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    (out_dir / "windows").mkdir(exist_ok=True)

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
        from src.models.recurrent_ppo_trainer import RecurrentPPOTrainer
        agent_config = Config.load_yaml(args.agent_config) if args.agent_config else Config({})
        trainer = RecurrentPPOTrainer(params, agent_config)

        restore_mngr = ocp.CheckpointManager(str(ckpt_path))
        step = restore_mngr.latest_step()
        if not args.quiet:
            print(f"[eval_rollout] Restoring RPPO checkpoint at step {step}", flush=True)
        restored = restore_mngr.restore(step, args=ocp.args.PyTreeRestore())
        import flax.nnx as nnx
        from flax.nnx import graph as nnx_graph
        restored_model_state = restored["model"]
        flat_restored, _ = jax.tree_util.tree_flatten_with_path(restored_model_state)
        restored_dict = {str(k): v for k, v in flat_restored}
        flat_current, tree_def = jax.tree_util.tree_flatten_with_path(
            nnx.state(trainer.agent)
        )
        valid_flat = [restored_dict.get(str(k), v) for k, v in flat_current]
        new_state = jax.tree_util.tree_unflatten(tree_def, valid_flat)
        nnx.update(trainer.agent, new_state)

        def policy_fn(state, carry, key, deterministic=True):
            obs = trainer._get_obs(state)
            if carry is None:
                carry = trainer.agent.initialize_hidden(batch_size=1)
            action, carry = trainer.agent.get_action(obs[None], carry, key, deterministic=deterministic)
            return action[0], carry
    else:
        raise NotImplementedError(f"Agent type '{agent_type}' not yet supported by eval_rollout.py. "
                                  f"Implement DreamerV3 checkpoint loading and add here.")

    # --- Run episodes ---
    t_start = time.time()
    episodes = []
    for ep_idx in range(n_eps):
        key = jax.random.PRNGKey(seeds[ep_idx])
        if not args.quiet:
            print(f"[eval_rollout] Episode {ep_idx + 1}/{n_eps} (seed={seeds[ep_idx]})...", end="\r", flush=True)
        ep_data = _run_episode(
            params, policy_fn, key, max_steps=max_steps,
            deterministic=(bm_cfg.eval_policy_mode == "deterministic"),
        )
        ep_data["seed"] = np.int32(seeds[ep_idx])
        episodes.append(ep_data)

        # Save .npz
        np.savez_compressed(
            out_dir / "episodes" / f"{ep_idx:04d}.npz",
            **ep_data,
        )

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
