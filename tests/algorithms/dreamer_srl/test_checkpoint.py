"""Commit B regression test — Orbax checkpoint round-trip for dreamer-srl.

Verifies:
  1. make_checkpoint_manager() creates the checkpoints/ dir.
  2. save_checkpoint() fires without error and writes a checkpoint step dir.
  3. load_checkpoint() reads back a pytree with the expected top-level keys.
  4. The loaded actor param shapes match the original (round-trip correctness).

Uses a freshly-initialised agent (no training required) for speed.
"""
import os
import sys
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.utils import moments_init
from src.algorithms.dreamer_srl.checkpoint import (
    make_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
)


@pytest.fixture(scope='module')
def tiny_agent():
    """Build a tiny dreamer-srl agent for checkpoint tests (no env needed)."""
    from src.utils.config import get_default_config
    _project_root = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'
    env_cfg = get_default_config()
    env_cfg.merge(Config.load_yaml(f'{_project_root}/configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml'))
    agent_cfg = Config.load_yaml(f'{_project_root}/configs/models/dreamer_srl/01_food_only.yaml')

    from src.environment.config_loader import load_env_params
    env_params = load_env_params(env_cfg)
    obs_dim = 19  # food-only 5x5 obs dim (known fixture)
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=agent_cfg.to_dict(),
        rngs=rngs,
    )
    moments = moments_init()
    return world_model, actor, critic, target_critic, moments, key


def test_checkpoint_manager_creates_dir(tmp_path, tiny_agent):
    """make_checkpoint_manager creates checkpoints/ under results_dir."""
    manager = make_checkpoint_manager(str(tmp_path), max_to_keep=5)
    assert (tmp_path / 'checkpoints').is_dir(), "checkpoints/ dir not created"


def test_save_checkpoint_writes_step(tmp_path, tiny_agent):
    """save_checkpoint writes a step directory inside checkpoints/."""
    world_model, actor, critic, target_critic, moments, key = tiny_agent
    manager = make_checkpoint_manager(str(tmp_path), max_to_keep=5)

    save_checkpoint(
        manager=manager,
        episode=50,
        world_model=world_model,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        moments=moments,
        key=key,
        iter_num=100,
        policy_step=5000,
        total_episodes_completed=50,
        cumulative_grad_steps=200,
    )

    ckpt_dir = tmp_path / 'checkpoints'
    # Orbax saves each step in a numbered directory
    step_dirs = [d for d in ckpt_dir.iterdir() if d.is_dir()]
    assert len(step_dirs) >= 1, f"Expected >=1 checkpoint step dir; got {step_dirs}"


def test_load_checkpoint_has_keys(tmp_path, tiny_agent):
    """load_checkpoint() returns a dict with expected top-level keys."""
    world_model, actor, critic, target_critic, moments, key = tiny_agent
    manager = make_checkpoint_manager(str(tmp_path), max_to_keep=5)

    save_checkpoint(
        manager=manager,
        episode=100,
        world_model=world_model,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        moments=moments,
        key=key,
        iter_num=200,
        policy_step=10000,
        total_episodes_completed=100,
        cumulative_grad_steps=400,
    )

    restored = load_checkpoint(manager, 100)
    assert restored is not None, "load_checkpoint returned None"
    expected_keys = {'world_model', 'actor', 'critic', 'target_critic',
                     'key', 'iter_num', 'policy_step',
                     'total_episodes_completed', 'cumulative_grad_steps'}
    missing = expected_keys - set(restored.keys())
    assert not missing, f"Missing keys in restored checkpoint: {missing}"


def test_checkpoint_actor_shapes_preserved(tmp_path, tiny_agent):
    """Round-trip: restored actor pytree has same leaf shapes as original."""
    world_model, actor, critic, target_critic, moments, key = tiny_agent
    manager = make_checkpoint_manager(str(tmp_path), max_to_keep=5)

    original_actor_state = nnx.state(actor, nnx.Param)

    save_checkpoint(
        manager=manager,
        episode=200,
        world_model=world_model,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        moments=moments,
        key=key,
        iter_num=400,
        policy_step=20000,
        total_episodes_completed=200,
        cumulative_grad_steps=800,
    )

    restored = load_checkpoint(manager, 200)
    restored_actor = restored['actor']

    # Compare leaf shapes (not values, since Orbax may return numpy arrays)
    orig_leaves = jax.tree_util.tree_leaves(original_actor_state)
    rest_leaves = jax.tree_util.tree_leaves(restored_actor)
    assert len(orig_leaves) == len(rest_leaves), \
        f"Leaf count mismatch: {len(orig_leaves)} vs {len(rest_leaves)}"
    for orig_l, rest_l in zip(orig_leaves, rest_leaves):
        assert np.asarray(orig_l).shape == np.asarray(rest_l).shape, \
            f"Shape mismatch: {np.asarray(orig_l).shape} vs {np.asarray(rest_l).shape}"
