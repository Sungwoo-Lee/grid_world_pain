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
    wm_opt=None,
    actor_opt=None,
    critic_opt=None,
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

    # Optimizer state (Adam mu/nu). Saved so a resumed run continues with warm
    # Adam moments instead of restarting them. Checkpoints written before this
    # was added lack these keys; restore_dreamer_training_state() detects that
    # and falls back to a network-only restore.
    if wm_opt is not None and actor_opt is not None and critic_opt is not None:
        ckpt_data['wm_opt']     = nnx.state(wm_opt)
        ckpt_data['actor_opt']  = nnx.state(actor_opt)
        ckpt_data['critic_opt'] = nnx.state(critic_opt)

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


def checkpoint_has_optimizers(manager: ocp.CheckpointManager, step: int) -> bool:
    """True if the checkpoint at `step` stores optimizer state.

    Checkpoints written before optimizer state was added to `save_checkpoint`
    hold only the four network trees + scalars; restoring one with an
    optimizer-bearing target fails Orbax's structure match. Detection reads the
    on-disk array index and falls back to False (the older, smaller format)
    when the layout cannot be parsed — it never guesses "present", so an
    unreadable index degrades to the restore that is always valid.
    """
    import json
    try:
        meta = os.path.join(str(manager.directory), str(step), 'default', '_METADATA')
        with open(meta) as fh:
            raw = json.load(fh)
        found = []

        def _walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(v, f"{path}/{k}")
            else:
                found.append(path)

        _walk(raw)
        return any('_opt' in p for p in found)
    except Exception:
        return False


def restore_dreamer_training_state(
    load_path,
    world_model,
    actor,
    critic,
    target_critic,
    moments,
    key,
    wm_opt=None,
    actor_opt=None,
    critic_opt=None,
    *,
    episode: int = None,
    quiet: bool = False,
):
    """Restore dreamer-srl state in place; return the restored counters.

    Mirrors `src/utils/checkpoint_restore.restore_rppo_training_state` — the
    proven restore-to-target pattern. Orbax reconstructs the exact live pytree
    structure and raises on any structure/shape/dtype mismatch. Failure is
    ALWAYS fatal: a silently-skipped restore would train from scratch while
    reporting resumed counters (the H1 bug this pattern exists to prevent).

    Args:
        load_path: the run's ``checkpoints/`` dir (CheckpointManager root).
        world_model, actor, critic, target_critic: live NNX modules, used as
            restore targets and updated in place via ``nnx.update``.
        moments, key: live moments + PRNG key, used as structure templates.
        wm_opt, actor_opt, critic_opt: live optimizers. Updated in place only
            when the checkpoint actually carries optimizer state.
        episode: checkpoint step to load; defaults to the latest.
        quiet: suppress the success print.

    Returns:
        dict with ``key, iter_num, policy_step, total_episodes_completed,
        cumulative_grad_steps, stage, moments, restored_optimizers``.

    Raises:
        FileNotFoundError: no checkpoint steps under ``load_path``.
        Whatever orbax raises on structure/shape/dtype mismatch or corruption.
    """
    manager = ocp.CheckpointManager(os.path.abspath(str(load_path)))
    step = episode if episode is not None else manager.latest_step()
    if step is None:
        raise FileNotFoundError(
            f"--load-checkpoint given but no checkpoint steps found under "
            f"{load_path!r} — refusing to silently train from scratch."
        )
    if step not in manager.all_steps():
        raise FileNotFoundError(
            f"Checkpoint episode {step} not found under {load_path!r}. "
            f"Available: {sorted(manager.all_steps())}"
        )

    # Scalars were saved as 0-d int32 arrays (see save_checkpoint) — the target
    # must match that dtype exactly or StandardRestore rejects the tree.
    _i32 = lambda: jnp.array(0, dtype=jnp.int32)
    target = {
        'world_model':              nnx.state(world_model, nnx.Param),
        'actor':                    nnx.state(actor, nnx.Param),
        'critic':                   nnx.state(critic, nnx.Param),
        'target_critic':            nnx.state(target_critic, nnx.Param),
        'key':                      key,
        'iter_num':                 _i32(),
        'policy_step':              _i32(),
        'total_episodes_completed': _i32(),
        'cumulative_grad_steps':    _i32(),
        'stage':                    _i32(),
    }
    if hasattr(moments, '_asdict'):
        target['moments'] = dict(moments._asdict())
    elif isinstance(moments, dict):
        target['moments'] = dict(moments)
    else:
        target['moments'] = {k: v for k, v in moments.__dict__.items()
                             if isinstance(v, jax.Array)}

    has_opt = checkpoint_has_optimizers(manager, step)
    opt_pairs = (('wm_opt', wm_opt), ('actor_opt', actor_opt), ('critic_opt', critic_opt))
    if has_opt and all(o is not None for _, o in opt_pairs):
        for name, opt in opt_pairs:
            target[name] = nnx.state(opt)
    else:
        has_opt = False

    restored = manager.restore(step, args=ocp.args.StandardRestore(item=target))

    nnx.update(world_model,   restored['world_model'])
    nnx.update(actor,         restored['actor'])
    nnx.update(critic,        restored['critic'])
    nnx.update(target_critic, restored['target_critic'])
    if has_opt:
        nnx.update(wm_opt,     restored['wm_opt'])
        nnx.update(actor_opt,  restored['actor_opt'])
        nnx.update(critic_opt, restored['critic_opt'])

    if not quiet:
        n_leaves = sum(len(jax.tree_util.tree_leaves(restored[k]))
                       for k in ('world_model', 'actor', 'critic', 'target_critic'))
        print(f"  -> Restored {n_leaves} network param leaves "
              f"(checkpoint episode {step}).")
        if has_opt:
            print("  -> Restored optimizer state (Adam moments continue).")
        else:
            print("  [RESUME WARNING] This checkpoint predates optimizer-state "
                  "saving: Adam moments restart from zero. They re-warm within "
                  "a few hundred gradient steps.")
        print("  [RESUME WARNING] The replay buffer is NOT checkpointed and "
              "starts EMPTY — the driver refills it with the restored policy "
              "before resuming gradient steps.")

    out = {k: restored[k] for k in
           ('key', 'iter_num', 'policy_step', 'total_episodes_completed',
            'cumulative_grad_steps', 'stage')}
    out['moments'] = restored['moments']
    out['restored_optimizers'] = has_opt
    return out
