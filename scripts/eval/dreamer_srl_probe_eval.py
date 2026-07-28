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

Concurrent-sweep gotcha (CPU restore + topology)
--------------------------------------------------
A parallel sweep launches many of these processes at once. On GPU, each
process's JAX init reserves GPU memory, so ~18 concurrent processes exhaust
the GPU and the later ones fail with `CUDA_ERROR_OUT_OF_MEMORY` — hence
`--device cpu` (default, matching `scripts/eval/eval_rollout.py`'s
convention), which forces `jax_platform_name` to `cpu` before any array is
created.

Forcing CPU surfaces a second problem: `checkpoint.load_checkpoint()` (a bare
`manager.restore(episode)` with no restore target) is "topology-locked" to
whatever device sharding the checkpoint was SAVED with (GPU, at training
time). Restoring on CPU with no target then raises `ValueError: Topology
mismatch detected. ... Please provide a target tree with the desired
topology`. The fix: build the FULL abstract target tree (matching every
top-level key `save_checkpoint()` writes — `world_model`, `actor`, `critic`,
`target_critic`, `key`, `iter_num`, `policy_step`,
`total_episodes_completed`, `cumulative_grad_steps`, `stage`, `moments`) out
of the freshly-built agent's own real (already CPU-resident, since
`--device cpu` forced the platform before these were created) arrays, then
restore INTO that target via `ocp.args.StandardRestore(item=target)`. Orbax
then derives each leaf's destination sharding from the target leaf's OWN
current device placement (CPU) rather than the checkpoint's saved GPU
sharding, sidestepping the mismatch entirely — see
`_get_sharding_for_target_leaf` in
`orbax/checkpoint/_src/handlers/standard_checkpoint_handler.py`. A targeted
restore like this also comes back typed exactly like the target (nnx.State /
MomentsState, not a raw digit-keyed dict), so the old
`_normalize_checkpoint()` post-processing (needed only for the untargeted
`load_checkpoint()` path used in `dreamer_srl_offline_wm_test.py`) is not
needed here — confirmed by the parity check in this file's validation.

Usage
-----
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        scripts/eval/dreamer_srl_probe_eval.py \\
        --agent-config results/JAX_DreamerSRL/<run>/models/agent_config.yaml \\
        --env-config configs/environment/experiment/behavior_probes/core/avoidance/avoid_pred_inj00.yaml \\
        --run-dir results/JAX_DreamerSRL/<run> \\
        --episode 30000 \\
        --output-root tmp/<probe_name>_eval \\
        --n-episodes 30 --seed 0 --device cpu

`--device cpu` is the default (matches `eval_rollout.py`'s rPPO convention),
so it is safe to launch many of these concurrently on the SAME GPU-trained
run without GPU memory contention. Exporting `JAX_PLATFORMS=cpu` on the CLI
too is redundant-but-harmless belt-and-suspenders for the same guarantee.
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
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_config, load_env_params
from src.environment.sensor import get_observation_breakdown
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.checkpoint import make_checkpoint_manager
from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout
from src.algorithms.dreamer_srl.utils import moments_init


def _build_restore_target(world_model, actor, critic, target_critic):
    """Full abstract target tree matching every key `save_checkpoint()` writes
    (src/algorithms/dreamer_srl/checkpoint.py:save_checkpoint).

    Values are REAL arrays (not jax.ShapeDtypeStruct) built from the
    already-constructed agent + fresh placeholders, so each leaf carries its
    OWN current device sharding. Passing this as `item=` to
    `ocp.args.StandardRestore` makes orbax restore onto that device (CPU when
    `--device cpu` forced the platform before this function ran), instead of
    the checkpoint's saved GPU sharding — this is what fixes the topology
    mismatch under forced-CPU execution. The exact VALUES here are
    placeholders; only shape/dtype/device matter, since restore overwrites
    them with the checkpoint's saved data.
    """
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
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                         help="JAX device (default cpu — matches eval_rollout.py's rPPO "
                              "convention; safe for high-concurrency sweeps on a shared GPU).")
    args = parser.parse_args()

    # Force the platform BEFORE any array is created (must happen before the
    # first jax.random.PRNGKey / nnx.Rngs / build_agent call below). Mirrors
    # scripts/eval/eval_rollout.py's `--device cpu` handling.
    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

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
        # D-018: required for hierarchical checkpoints (optional kwarg; flat ignores it).
        observation_breakdown=get_observation_breakdown(env_params),
    )
    print('[probe-eval] build_agent OK')

    # --- 3. Restore checkpoint (topology-agnostic: restore INTO a real,
    #     already-CPU-resident target tree so orbax uses ITS sharding, not
    #     the checkpoint's saved GPU sharding — see module docstring) ---
    run_dir = os.path.abspath(args.run_dir)
    print(f'[probe-eval] Restoring checkpoint step={args.episode} from {run_dir}/checkpoints/ '
          f'(device={args.device})')
    # Training checkpoints written after the --load-checkpoint resume feature
    # landed (checkpoint.py:save_checkpoint) ALSO carry wm_opt/actor_opt/
    # critic_opt (Adam state). `StandardRestore` demands the target tree match
    # the saved tree EXACTLY, so this model-only target fails structurally
    # against those full-training-state checkpoints. Use `PyTreeRestore(...,
    # partial_restore=True)`: it restores only the keys present in `target` and
    # ignores extra keys in the checkpoint (optimizer subtrees included), so one
    # code path serves both old model-only and new full-state checkpoints.
    # A plain CheckpointManager (no `checkpointers=` kwarg) is required --
    # make_checkpoint_manager() binds StandardCheckpointer, which registers only
    # Standard{Save,Restore} handlers and rejects PyTreeRestore args.
    # Same pattern as scripts/eval/eval_rollout.py and
    # scripts/dreamer/visualize_dream.py.
    manager = ocp.CheckpointManager(os.path.join(run_dir, 'checkpoints'))
    target = _build_restore_target(world_model, actor, critic, target_critic)

    try:
        restored = manager.restore(
            args.episode,
            args=ocp.args.PyTreeRestore(item=target, partial_restore=True))
    except Exception as e:
        raise ValueError(
            f"Failed to restore checkpoint step={args.episode} from {run_dir}/checkpoints/ into "
            f"the agent built from --agent-config (obs_dim={obs_dim}, action_dim={action_dim}). "
            f"This usually means either the step doesn't exist (ls {run_dir}/checkpoints/) or the "
            f"probe env config (--env-config) produces a different obs_dim/action_dim than the "
            f"environment the checkpoint was trained on. Original error: {e}"
        ) from e
    if restored is None:
        raise RuntimeError(
            f'manager.restore returned None for step={args.episode} under {run_dir}/checkpoints/ '
            f'— confirm the step exists (ls {run_dir}/checkpoints/).'
        )

    # Targeted StandardRestore returns leaves typed to match `target` (nnx.State,
    # not a raw digit-keyed dict), so no _normalize_checkpoint step is needed here.
    nnx.update(world_model, restored['world_model'])
    nnx.update(actor, restored['actor'])
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
