#!/usr/bin/env python
"""dreamer_srl_probe_eval.py — Standalone eval driver: run a frozen dreamer_srl
checkpoint through a behavior-probe environment, writing the same `.rec.gz`
recordings the behavior-measure pipeline reads.

Plain-language purpose
-----------------------
`scripts/eval/eval_rollout.py` (the existing offline-eval driver) supports the
rPPO agent but raises `NotImplementedError` for Dreamer checkpoints. This
script fills that gap for dreamer_srl: it loads a frozen checkpoint (world
model + actor), builds one of the 12 avoidance "behavior probe" environments
(configs under `configs/environment/experiment/behavior_probes/`) — NOT the
environment the model was trained in — and runs deterministic (argmax) eval
episodes through it. Each episode is written to
`<output-root>/recordings/<episode>/episode_NNNNNN.rec.gz`, which is exactly
the recording format `scripts/behavior_measures/avoidance_stats_heatmap.py`
already reads for the rPPO agent, so no downstream code changes are needed.

Config-loading gotcha (read before using --env-config)
--------------------------------------------------------
The checkpoint's own saved `models/env_config.yaml` describes the TRAINING
environment, not the probe. This script deliberately does NOT load that file
for env_params. Instead, `--env-config` (a probe YAML, e.g.
`avoid_pred_inj00.yaml`) is merged through the exact same chain
`dreamer_srl_main.py` uses in single-config mode (`get_default_config()` +
`configs/{train,evaluation,visualization}/default.yaml` +
`load_env_config(probe_path)`), so any `extends:` in the probe YAML resolves
identically to how it resolves at training time. Only `--agent-config` is
loaded from the checkpoint's own `models/agent_config.yaml`, since that
describes the frozen network architecture whose shapes must match the
checkpoint's saved parameters.

Usage
-----
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        scripts/eval/dreamer_srl_probe_eval.py \\
        --agent-config results/JAX_DreamerSRL/<run>/models/agent_config.yaml \\
        --env-config configs/environment/experiment/behavior_probes/core/avoidance/avoid_pred_inj00.yaml \\
        --run-dir results/JAX_DreamerSRL/<run> \\
        --episode 30000 \\
        --output-root tmp/<probe_name>_eval \\
        --n-episodes 30 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
from flax import nnx

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_config, load_env_params
from src.environment.sensor import get_observation_breakdown
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.checkpoint import make_checkpoint_manager, load_checkpoint
from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout


def _normalize_checkpoint(d):
    """Normalize an orbax-restored checkpoint pytree for nnx.update.

    Ported verbatim from scripts/dreamer/dreamer_srl_offline_wm_test.py:167-181
    (the same two transforms applied there): digit-string keys -> int, and
    unwrap single-key {'value': array} leaf dicts.
    """
    if isinstance(d, dict):
        if set(d.keys()) == {'value'}:
            return _normalize_checkpoint(d['value'])
        return {
            (int(k) if isinstance(k, str) and k.isdigit() else k): _normalize_checkpoint(v)
            for k, v in d.items()
        }
    return d


def _load_probe_env_cfg(env_config_path: str) -> Config:
    """Merged env Config, mirroring dreamer_srl_main.py's single-config path
    (dreamer_srl_main.py:L507-L520): default env config + train/evaluation/
    visualization defaults + the probe config (with `extends:` resolved by
    load_env_config).
    """
    env_cfg = get_default_config()
    for rel in ('configs/train/default.yaml', 'configs/evaluation/default.yaml',
                'configs/visualization/default.yaml'):
        path = PROJECT_ROOT / rel
        if path.exists():
            env_cfg.merge(Config.load_yaml(str(path)))
    env_cfg.merge(load_env_config(env_config_path))
    return env_cfg


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent-config', required=True,
                         help="Path to the run's models/agent_config.yaml (network architecture).")
    parser.add_argument('--env-config', required=True,
                         help="Path to a PROBE env config (e.g. behavior_probes/core/avoidance/*.yaml).")
    parser.add_argument('--run-dir', required=True,
                         help="Run dir containing checkpoints/ (the CheckpointManager's home).")
    parser.add_argument('--episode', type=int, required=True,
                         help="Checkpoint step to restore (e.g. 30000).")
    parser.add_argument('--output-root', required=True,
                         help="Root dir for recordings/<episode>/episode_*.rec.gz output.")
    parser.add_argument('--n-episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # --- 1. Load configs ---
    print(f'[probe-eval] Loading probe env config: {args.env_config}')
    env_cfg = _load_probe_env_cfg(args.env_config)
    env_params = load_env_params(env_cfg)

    print(f'[probe-eval] Loading agent config: {args.agent_config}')
    agent_cfg = Config.load_yaml(args.agent_config)

    obs_breakdown = get_observation_breakdown(env_params)
    obs_dim = sum(obs_breakdown.values())
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)
    print(f'[probe-eval] obs_dim={obs_dim}  action_dim={action_dim}')
    print(f'[probe-eval] obs breakdown: {dict(obs_breakdown)}')

    # --- 2. Build agent (fresh architecture matching agent_config) ---
    rngs = nnx.Rngs(args.seed)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim, action_dim=action_dim, cfg=agent_cfg.to_dict(), rngs=rngs,
    )
    print('[probe-eval] build_agent OK')

    # --- 3. Restore checkpoint ---
    run_dir = os.path.abspath(args.run_dir)
    print(f'[probe-eval] Restoring checkpoint step={args.episode} from {run_dir}/checkpoints/')
    manager = make_checkpoint_manager(run_dir, max_to_keep=100)
    ckpt = load_checkpoint(manager, args.episode)
    if ckpt is None:
        raise RuntimeError(
            f'load_checkpoint returned None for step={args.episode} under {run_dir}/checkpoints/ '
            f'— confirm the step exists (ls {run_dir}/checkpoints/).'
        )
    ckpt = _normalize_checkpoint(ckpt)
    for key in ('world_model', 'actor'):
        if key not in ckpt:
            raise ValueError(f"Checkpoint at step={args.episode} is missing '{key}' — "
                              f"cannot restore. Available top-level keys: {list(ckpt.keys())}")

    try:
        nnx.update(world_model, ckpt['world_model'])
        nnx.update(actor, ckpt['actor'])
    except Exception as e:
        raise ValueError(
            f"Failed to restore checkpoint step={args.episode} into the agent built from "
            f"--agent-config (obs_dim={obs_dim}, action_dim={action_dim}). This usually means "
            f"the probe env config (--env-config) produces a different obs_dim/action_dim than "
            f"the environment the checkpoint was trained on. Original error: {e}"
        ) from e
    print(f'[probe-eval] Checkpoint restored OK at step={args.episode}')

    # --- 4. Run probe rollout ---
    os.makedirs(args.output_root, exist_ok=True)
    t0 = time.time()
    result = dreamer_srl_eval_rollout(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=args.n_episodes,
        seed=args.seed,
        results_dir=args.output_root,
        checkpoint_pct=args.episode,
        render_video=True,
        quiet=False,
    )
    elapsed = time.time() - t0
    per_ep = elapsed / max(args.n_episodes, 1)
    print(f'[probe-eval] Done: {args.n_episodes} episodes in {elapsed:.1f}s '
          f'({per_ep:.2f}s/episode)')
    print(f'[probe-eval] mean_reward={result["mean_reward"]:.3f}  '
          f'mean_length={result["mean_length"]:.1f}')
    print(f'[probe-eval] recordings written to: {result["recordings_dir"]}')


if __name__ == '__main__':
    main()
