"""Orbax checkpointing helpers for dreamer-srl.

Mirrors the checkpoint pattern in train.py:L2411-L2426 (DreamerV3 branch)
using ocp.StandardSave / ocp.StandardCheckpointer via CheckpointManager.

Usage::

    from src.algorithms.dreamer_srl.checkpoint import (
        make_checkpoint_manager, save_checkpoint, load_checkpoint,
    )

    manager = make_checkpoint_manager(results_dir, max_to_keep=20)
    save_checkpoint(manager, episode, world_model, actor, critic,
                    target_critic, moments, key, iter_num, policy_step,
                    total_episodes_completed, cumulative_grad_steps)
    ckpt = load_checkpoint(manager, episode)  # returns dict of pytrees
"""
from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx


def make_checkpoint_manager(results_dir: str, max_to_keep: int = 20) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager under <results_dir>/checkpoints/.

    Ported from train.py:L555-L559 (ocp.CheckpointManager setup).
    """
    ckpt_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    manager = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir),
        checkpointers=ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True),
    )
    return manager


def save_checkpoint(
    manager: ocp.CheckpointManager,
    episode: int,
    world_model,
    actor,
    critic,
    target_critic,
    moments,
    key: jax.Array,
    iter_num: int,
    policy_step: int,
    total_episodes_completed: int,
    cumulative_grad_steps: int,
    stage: int = 0,
) -> None:
    """Save dreamer-srl state to an Orbax checkpoint at the given episode count.

    Dict structure mirrors train.py:L2411-L2421 (DreamerV3 ckpt_data).
    Ported from train.py:L2423-L2426.

    Args:
        manager: CheckpointManager returned by make_checkpoint_manager().
        episode: Step label for the checkpoint (= total_episodes_completed).
        world_model: dreamer-srl WorldModel (nnx module).
        actor: dreamer-srl Actor (nnx module).
        critic: dreamer-srl Critic (nnx module).
        target_critic: dreamer-srl target Critic (nnx module).
        moments: moments dataclass (plain JAX arrays, already picklable).
        key: current PRNG key.
        iter_num: current iteration number.
        policy_step: current policy_step counter.
        total_episodes_completed: running episode count.
        cumulative_grad_steps: total gradient steps taken.
        stage: current curriculum stage index (0-based). Written to the
            checkpoint payload so a future resume plan can land in the right
            stage. Defaults to 0 (single-config / non-curriculum runs).
            NOTE: restore wiring is out of scope for this plan (no resume
            path exists in the dreamer-srl driver today). This field is
            write-only until a future plan adds restore.
    """
    ckpt_data = {
        'world_model':              nnx.state(world_model, nnx.Param),
        'actor':                    nnx.state(actor, nnx.Param),
        'critic':                   nnx.state(critic, nnx.Param),
        'target_critic':            nnx.state(target_critic, nnx.Param),
        # Moments + bookkeeping scalars stored as 0-d arrays for Orbax compat
        'key':                      key,
        'iter_num':                 jnp.array(iter_num, dtype=jnp.int32),
        'policy_step':              jnp.array(policy_step, dtype=jnp.int32),
        'total_episodes_completed': jnp.array(total_episodes_completed, dtype=jnp.int32),
        'cumulative_grad_steps':    jnp.array(cumulative_grad_steps, dtype=jnp.int32),
        'stage':                    jnp.array(stage, dtype=jnp.int32),
    }
    # Moments is a NamedTuple of JAX arrays — include field-by-field for Orbax
    # (Orbax StandardSave handles plain JAX pytrees natively).
    if hasattr(moments, '_asdict'):
        ckpt_data['moments'] = dict(moments._asdict())
    elif isinstance(moments, dict):
        ckpt_data['moments'] = moments
    else:
        # Fallback: store the object fields as a dict
        ckpt_data['moments'] = {k: v for k, v in moments.__dict__.items()
                                 if isinstance(v, jax.Array)}

    # Ported from train.py:L2425-L2426
    manager.save(episode, args=ocp.args.StandardSave(ckpt_data))
    manager.wait_until_finished()


def load_checkpoint(manager: ocp.CheckpointManager, episode: int) -> dict:
    """Load a dreamer-srl checkpoint saved at `episode`.

    Returns the raw pytree dict as stored (call nnx.update(module, ckpt['actor'])
    etc. to restore into live modules).

    Ported from train.py:L1071-L1073 (ocp.CheckpointManager restore pattern).
    """
    # Orbax StandardCheckpointer restore needs an abstract_target that matches
    # the saved structure.  Use restore() with None to get a raw dict back.
    return manager.restore(episode)
