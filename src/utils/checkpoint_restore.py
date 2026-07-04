"""Fatal-on-failure checkpoint restore for train.py's RecurrentPPO resume path.

Fixes H1 of docs/develop/active/issues/diag_fable5_20260704/01_train_entry_config.md:
the old strict-match restore compared DictKey vs GetAttrKey path strings (0 leaves
ever matched) and a blanket except swallowed the failure. This module uses the
canonical flax-NNX + orbax restore-to-target pattern: orbax reconstructs the exact
live pytree structure (nnx.State, real optax tuples) and raises on any
structure/shape/dtype mismatch. Failure is ALWAYS fatal — never print-and-continue.
"""
import os

import jax
import orbax.checkpoint as ocp
from flax import nnx


def restore_rppo_training_state(load_path, model, optimizer, h_state, key, *, quiet=False):
    """Restore model/optimizer in place; return the restored counters.

    Args:
        load_path: CheckpointManager root (the run's ``models/`` dir).
        model, optimizer: live NNX model + optimizer — used as restore targets
            and updated in place via ``nnx.update``.
        h_state, key: live recurrent state and PRNG key — used only as structure
            templates for the restore target.
        quiet: suppress the success print.

    Returns:
        dict with keys ``h_state, key, step, iteration, episode, stage``
        (counters as saved; caller casts to int).

    Raises:
        FileNotFoundError: no checkpoint steps under ``load_path``.
        Whatever orbax raises on structure/shape/dtype mismatch or corruption.
    """
    mngr = ocp.CheckpointManager(os.path.abspath(load_path))
    step = mngr.latest_step()
    if step is None:
        raise FileNotFoundError(
            f"--load-checkpoint given but no checkpoint steps found under "
            f"{load_path!r} — refusing to silently train from scratch."
        )
    # Target mirrors the exact StandardSave payload written at train.py
    # checkpoint-save time (rPPO branch).
    target = {
        'model': nnx.state(model, nnx.Param),
        'optimizer': nnx.state(optimizer),
        'h_state': h_state,
        'key': key,
        'iteration': 0,
        'step': 0,
        'episode': 0,
        'stage': 0,
    }
    restored = mngr.restore(step, args=ocp.args.StandardRestore(item=target))
    nnx.update(model, restored['model'])
    nnx.update(optimizer, restored['optimizer'])
    if not quiet:
        n_leaves = len(jax.tree_util.tree_leaves(restored['model']))
        print(f"  -> Restored {n_leaves} model param leaves + optimizer state "
              f"(checkpoint step {step}).")
    return {k: restored[k] for k in
            ('h_state', 'key', 'step', 'iteration', 'episode', 'stage')}
