"""Commit C regression test — dreamer_srl_eval_rollout().

Verifies:
  1. Runs 2 episodes with a freshly-initialized agent (random behaviour).
  2. episode_lengths are within [1, env_params.max_steps].
  3. If render_video=True, recordings_dir exists with 2 .rec.gz + run_meta.pkl.
  4. mean_reward and mean_length are reasonable scalars.
"""
import os
import sys
import pytest
import jax
import jax.numpy as jnp

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

from flax import nnx
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout


@pytest.fixture(scope='module')
def eval_agent_and_params():
    """Build a tiny agent + env_params for eval tests."""
    _root = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'
    env_cfg = get_default_config()
    env_cfg.merge(Config.load_yaml(f'{_root}/configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml'))
    for rel in ['configs/train/default.yaml', 'configs/evaluation/default.yaml',
                'configs/visualization/default.yaml']:
        env_cfg.merge(Config.load_yaml(os.path.join(_root, rel)))

    agent_cfg = Config.load_yaml(f'{_root}/configs/models/dreamer_srl/01_food_only.yaml')

    env_params = load_env_params(env_cfg)
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)
    obs_dim = 19  # known for 5x5 food-only

    key = jax.random.PRNGKey(7)
    rngs = nnx.Rngs(key)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=agent_cfg.to_dict(),
        rngs=rngs,
    )
    return world_model, actor, env_params, env_cfg


def test_eval_rollout_returns_stats(eval_agent_and_params, tmp_path):
    """dreamer_srl_eval_rollout runs 2 episodes, returns plausible stats."""
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    result = dreamer_srl_eval_rollout(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=2,
        seed=0,
        results_dir=str(tmp_path),
        checkpoint_pct=50,
        render_video=False,
        quiet=True,
    )
    assert 'mean_reward' in result
    assert 'mean_length' in result
    assert 'episode_rewards' in result
    assert 'episode_lengths' in result
    assert len(result['episode_rewards']) == 2
    assert len(result['episode_lengths']) == 2


def test_eval_rollout_episode_lengths_valid(eval_agent_and_params, tmp_path):
    """Episode lengths are in [1, max_steps]."""
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    result = dreamer_srl_eval_rollout(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=2,
        seed=1,
        results_dir=str(tmp_path),
        checkpoint_pct=100,
        render_video=False,
        quiet=True,
    )
    for ep_len in result['episode_lengths']:
        assert 1 <= ep_len <= int(env_params.max_steps), \
            f"episode_length={ep_len} outside [1, {env_params.max_steps}]"


def test_eval_rollout_recordings_exist(eval_agent_and_params, tmp_path):
    """With render_video=True: recordings_dir has 2 .rec.gz + run_meta.pkl."""
    world_model, actor, env_params, env_cfg = eval_agent_and_params
    result = dreamer_srl_eval_rollout(
        world_model=world_model,
        actor=actor,
        env_params=env_params,
        config=env_cfg,
        num_episodes=2,
        seed=2,
        results_dir=str(tmp_path),
        checkpoint_pct=200,
        render_video=True,
        quiet=True,
    )
    rdir = result['recordings_dir']
    assert rdir is not None
    from pathlib import Path
    rpath = Path(rdir)
    assert rpath.is_dir(), f"recordings_dir {rpath} not created"
    rec_files = list(rpath.glob('episode_*.rec.gz'))
    assert len(rec_files) == 2, f"Expected 2 .rec.gz files, got {len(rec_files)}"
    meta_file = rpath / 'run_meta.pkl'
    assert meta_file.exists(), "run_meta.pkl not written"
