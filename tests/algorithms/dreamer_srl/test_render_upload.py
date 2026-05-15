"""Commit E test — _render_and_upload() smoke test.

Verifies that _render_and_upload():
  1. Is importable from src.algorithms.dreamer_srl.eval.
  2. Returns None gracefully when given a recordings_dir with no .rec.gz files
     (render_recordings.py will error out; we verify the None path).
  3. Returns the MP4 path (str) when render succeeds on a real 1-episode recording.

Test 3 runs render_recordings.py in a subprocess (JAX_PLATFORMS=cpu) on
a real recording written by dreamer_srl_eval_rollout. It is slow (~30s) but
required to confirm the subprocess→MP4 flow end-to-end.

Note: WandB upload is NOT tested here (requires live WandB run). That hop is
verified by the Commit G end-to-end smoke test.
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')


def test_render_and_upload_importable():
    """_render_and_upload is importable from eval module."""
    from src.algorithms.dreamer_srl.eval import _render_and_upload  # noqa: F401


def test_render_and_upload_empty_dir(tmp_path):
    """_render_and_upload returns None gracefully when no .rec.gz in dir."""
    from src.algorithms.dreamer_srl.eval import _render_and_upload

    # Create empty recordings dir (no .rec.gz files)
    rdir = tmp_path / 'recordings' / '0'
    rdir.mkdir(parents=True)

    result = _render_and_upload(
        recordings_dir=str(rdir),
        results_dir=str(tmp_path),
        checkpoint_pct=0,
        fps=5,
        wandb_enabled=False,
        quiet=True,
    )
    # render_recordings.py should fail (no episodes) → returns None
    assert result is None


@pytest.mark.slow
def test_render_and_upload_produces_mp4(tmp_path):
    """Full render pipeline: 1 episode recording → eval_N.mp4 exists."""
    import jax
    from flax import nnx
    from src.utils.config import Config, get_default_config
    from src.environment.config_loader import load_env_params
    from src.algorithms.dreamer_srl.agent import build_agent
    from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout, _render_and_upload

    _root = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'
    env_cfg = get_default_config()
    env_cfg.merge(Config.load_yaml(f'{_root}/configs/experiment/dreamer_curriculum/01_food_only.yaml'))
    for rel in ['configs/train/default.yaml', 'configs/evaluation/default.yaml',
                'configs/visualization/default.yaml']:
        env_cfg.merge(Config.load_yaml(os.path.join(_root, rel)))
    agent_cfg = Config.load_yaml(f'{_root}/configs/dreamer_srl/01_food_only.yaml')
    env_params = load_env_params(env_cfg)
    obs_dim = 19
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)

    key = jax.random.PRNGKey(55)
    rngs = nnx.Rngs(key)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim, action_dim=action_dim, cfg=agent_cfg.to_dict(), rngs=rngs,
    )

    result = dreamer_srl_eval_rollout(
        world_model=world_model, actor=actor, env_params=env_params, config=env_cfg,
        num_episodes=1, seed=0, results_dir=str(tmp_path), checkpoint_pct=5,
        render_video=True, quiet=True,
    )

    mp4_path = _render_and_upload(
        recordings_dir=result['recordings_dir'],
        results_dir=str(tmp_path),
        checkpoint_pct=5,
        fps=5,
        wandb_enabled=False,
        quiet=True,
    )
    assert mp4_path is not None, (
        "render_and_upload returned None — render_recordings.py subprocess likely failed. "
        "Check that JAX_PLATFORMS=cpu works and scripts/render_recordings.py is on PATH."
    )
    assert Path(mp4_path).exists(), f"MP4 not found at {mp4_path}"
    assert Path(mp4_path).stat().st_size > 1024, f"MP4 suspiciously small: {Path(mp4_path).stat().st_size} bytes"
