"""Regression test (H1): rPPO checkpoint restore must actually restore.

Bug context
-----------
diag_fable5_20260704/01_train_entry_config.md Finding 1: train.py's old
strict-match restore string-compared tree paths of the raw
``ocp.args.PyTreeRestore()`` dict (leaf paths ending ``DictKey(key='value')``)
against live ``nnx.state(model)`` (paths ending ``GetAttrKey(name='value')``)
— 100 % of leaves mismatched, the architecture-mismatch ``ValueError`` fired,
and a blanket ``except`` swallowed it: training silently continued from
freshly random weights. Independently,
``nnx.update(optimizer, restored['optimizer'])`` with the raw dict failed on
the optax-chain tuple (serialized as a dict with string integer keys).

Fix: ``src/utils/checkpoint_restore.restore_rppo_training_state`` uses the
canonical restore-to-target pattern (``ocp.args.StandardRestore(item=...)``
with live ``nnx.state`` targets) so orbax reconstructs the exact live pytree
structure and raises on any structure/shape/dtype mismatch. Failure is
always fatal.

Red -> green contract
---------------------
BEFORE the fix: all tests fail with ImportError (``src.utils.checkpoint_restore``
does not exist). The behavioral pre-fix proof is the diagnosis scripts
``tmp/20260704_diag_ckpt_restore_test.py`` / ``..._test2.py`` (44/44 path
mismatches; optimizer tuple failure), whose assertions this test replicates
against the production helper.
AFTER the fix: bit-exact weight + optimizer round-trip; mismatch raises;
missing checkpoint raises FileNotFoundError.

Run with::

    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/training/test_checkpoint_restore_roundtrip.py -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # before jax import

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest
from flax import nnx

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.models.recurrent_ppo_network import ActorCriticRNN
from src.utils.checkpoint_restore import restore_rppo_training_state

_OBS_BD = {"Olfaction": 5, "Collision": 4, "Satiation": 1}
_ENC_CFG = {"encoding_mode": "flat", "use_layer_norm": False}
_NUM_ENVS = 2


def _make_model(seed: int, hidden_size: int = 16) -> ActorCriticRNN:
    return ActorCriticRNN(
        input_dim=10, action_dim=4, hidden_size=hidden_size,
        rngs=nnx.Rngs(seed), rnn_type="LSTM", activation="tanh",
        modulation_config=None, observation_breakdown=_OBS_BD,
        encoding_config=_ENC_CFG,
    )


def _make_optimizer(model: ActorCriticRNN) -> nnx.Optimizer:
    # Exactly as train.py builds it (optax.chain(clip_by_global_norm, adam)).
    return nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4)),
        wrt=nnx.Param,
    )


def _take_one_optimizer_step(model: ActorCriticRNN, optimizer: nnx.Optimizer):
    """One real gradient step so Adam moments are non-trivial (non-zero)."""
    def loss_fn(m):
        return sum(jnp.sum(jnp.square(leaf))
                   for leaf in jax.tree_util.tree_leaves(nnx.state(m, nnx.Param)))
    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)


def _save_train_py_style(tmp_path, model, optimizer, h_state, key, counters):
    """Save with the exact CheckpointManager + StandardSave payload of
    train.py's rPPO checkpoint branch (train.py ~2450-2476)."""
    models_dir = os.path.join(str(tmp_path), "models")
    mngr = ocp.CheckpointManager(
        os.path.abspath(models_dir),
        checkpointers=ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )
    ckpt_data = {
        "model": nnx.state(model, nnx.Param),
        "optimizer": nnx.state(optimizer),
        "h_state": h_state,
        "key": key,
        "iteration": counters["iteration"],
        "step": counters["step"],
        "episode": counters["episode"],
        "stage": counters["stage"],
    }
    mngr.save(counters["episode"], args=ocp.args.StandardSave(ckpt_data))
    mngr.wait_until_finished()
    return models_dir


def _leaves_all_equal(tree_a, tree_b) -> bool:
    eq = jax.tree_util.tree_map(
        lambda a, b: bool(jnp.array_equal(a, b)), tree_a, tree_b)
    return all(jax.tree_util.tree_leaves(eq))


def test_rppo_restore_roundtrip_bit_exact(tmp_path):
    counters = {"iteration": 7, "step": 1234, "episode": 42, "stage": 1}

    # Model A (seed 0) + one optimizer step -> non-trivial Adam moments.
    model_a = _make_model(seed=0)
    opt_a = _make_optimizer(model_a)
    _take_one_optimizer_step(model_a, opt_a)
    h_state_a = model_a.initial_state(_NUM_ENVS)
    key_a = jax.random.PRNGKey(0)

    models_dir = _save_train_py_style(
        tmp_path, model_a, opt_a, h_state_a, key_a, counters)

    # Sanity: at least one Adam moment leaf is non-zero, so the optimizer
    # equality assert below is meaningful.
    opt_leaves = jax.tree_util.tree_leaves(nnx.state(opt_a))
    assert any(jnp.any(l != 0) for l in opt_leaves
               if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating)), \
        "optimizer state is all-zero; the one-step setup failed"

    # Model B (different seed) — weights differ before restore.
    model_b = _make_model(seed=99)
    opt_b = _make_optimizer(model_b)
    assert not _leaves_all_equal(nnx.state(model_a, nnx.Param),
                                 nnx.state(model_b, nnx.Param)), \
        "seeds 0 and 99 produced identical inits — test is vacuous"

    restored_meta = restore_rppo_training_state(
        models_dir, model_b, opt_b,
        model_b.initial_state(_NUM_ENVS), jax.random.PRNGKey(123),
        quiet=True,
    )

    # 1. Every Param leaf bit-exact.
    assert _leaves_all_equal(nnx.state(model_a, nnx.Param),
                             nnx.state(model_b, nnx.Param))
    # 2. Every optimizer-state leaf equal (incl. the non-zero Adam moments).
    assert _leaves_all_equal(nnx.state(opt_a), nnx.state(opt_b))
    # 3. Counters round-trip.
    assert int(restored_meta["step"]) == counters["step"]
    assert int(restored_meta["iteration"]) == counters["iteration"]
    assert int(restored_meta["episode"]) == counters["episode"]
    assert int(restored_meta["stage"]) == counters["stage"]
    # 4. h_state / key round-trip.
    assert _leaves_all_equal(restored_meta["h_state"], h_state_a)
    assert _leaves_all_equal(restored_meta["key"], key_a)


def test_rppo_restore_architecture_mismatch_raises(tmp_path):
    model_a = _make_model(seed=0, hidden_size=16)
    opt_a = _make_optimizer(model_a)
    models_dir = _save_train_py_style(
        tmp_path, model_a, opt_a, model_a.initial_state(_NUM_ENVS),
        jax.random.PRNGKey(0),
        {"iteration": 1, "step": 10, "episode": 2, "stage": 0})

    model_b = _make_model(seed=1, hidden_size=32)
    opt_b = _make_optimizer(model_b)
    # Orbax must refuse the shape mismatch — never return normally.
    # Observed type (orbax 0.11.33): builtins.ValueError, message
    # "Requested shape: (32, 4) is not compatible with the stored shape: (16, 4)...".
    with pytest.raises(ValueError, match="not compatible with the stored shape"):
        restore_rppo_training_state(
            models_dir, model_b, opt_b,
            model_b.initial_state(_NUM_ENVS), jax.random.PRNGKey(1),
            quiet=True,
        )


def test_rppo_restore_missing_checkpoint_raises(tmp_path):
    model = _make_model(seed=0)
    opt = _make_optimizer(model)
    empty_dir = os.path.join(str(tmp_path), "empty_models")
    os.makedirs(empty_dir)
    with pytest.raises(FileNotFoundError):
        restore_rppo_training_state(
            empty_dir, model, opt,
            model.initial_state(_NUM_ENVS), jax.random.PRNGKey(0),
            quiet=True,
        )
