"""Track C telemetry-fix regression tests (T1-T3).

Covers:
  T1 — video upload now uses the env-step (`policy_step`) WandB clock instead
       of the episode-count `checkpoint_pct` clock (Item 1: dropped videos).
  T2 — the `_eval_scalar_prefix` helper routes the video-pass scalar keys to
       `Eval/video/*` only when the stats pass is also enabled, otherwise it
       falls back to `Eval/*` (Item 3: estimator-mixing sawtooth).
  T3 — the fixed per-checkpoint eval seed is a *locked* design decision: the
       video pass's episodes are a bit-identical prefix of the stats pass's
       episodes when both share the same seed. This property is intentional
       (paired eval curve) and this test is a guard against a future
       accidental seed change, not a bug regression test — it is expected to
       pass both before and after the Track C fix.

See docs/develop/active/dreamer/DREAMER_SRL_EVAL_TELEMETRY_FIX.md.
"""
import os
import sys
from unittest import mock

import jax
import pytest

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

from flax import nnx
from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout


# ---------------------------------------------------------------------------
# T1 — video upload uses the policy_step (env-step) WandB clock
# ---------------------------------------------------------------------------

def test_video_upload_uses_policy_step_clock(tmp_path, monkeypatch):
    """_render_and_upload must pass step=policy_step (env-step clock) to
    upload_video, keeping episode=checkpoint_pct as the episode label.

    Pre-fix: _render_and_upload has no `policy_step` parameter, so this call
    raises TypeError (unexpected keyword argument).
    """
    from src.algorithms.dreamer_srl import eval as eval_mod

    # Fake a consolidated MP4 so the post-subprocess `os.path.exists` check passes.
    recordings_dir = tmp_path / 'recordings' / '10000'
    recordings_dir.mkdir(parents=True)
    videos_dir = tmp_path / 'videos'
    videos_dir.mkdir(parents=True)
    consolidated = videos_dir / 'eval_10000.mp4'
    consolidated.write_bytes(b'\x00' * 2048)

    def _fake_subprocess_run(cmd, **kwargs):
        return mock.Mock(returncode=0, stdout='', stderr='')

    captured = {}

    def _fake_upload_video(video_path, episode=None, step=None, caption=None, quiet=True):
        captured['video_path'] = video_path
        captured['episode'] = episode
        captured['step'] = step

    monkeypatch.setattr(eval_mod.subprocess, 'run', _fake_subprocess_run)
    monkeypatch.setattr('src.utils.wandb_utils.upload_video', _fake_upload_video)

    result = eval_mod._render_and_upload(
        recordings_dir=str(recordings_dir),
        results_dir=str(tmp_path),
        checkpoint_pct=10_000,
        fps=5,
        wandb_enabled=True,
        policy_step=1_000_000,
        quiet=True,
    )

    assert result == str(consolidated)
    assert captured['step'] == 1_000_000, (
        f"expected step=policy_step (1_000_000), got {captured.get('step')!r}"
    )
    assert captured['episode'] == 10_000, (
        f"expected episode=checkpoint_pct (10_000), got {captured.get('episode')!r}"
    )


# ---------------------------------------------------------------------------
# T2 — Eval scalar key routing helper
# ---------------------------------------------------------------------------

def test_eval_scalar_prefix_routing():
    """_eval_scalar_prefix routes video-pass keys to Eval/video/* only when
    the stats pass is also enabled; otherwise both fall back to Eval/*.

    Pre-fix: _eval_scalar_prefix does not exist -> import/collection error.
    """
    from src.algorithms.dreamer_srl.dreamer_srl_main import _eval_scalar_prefix

    assert _eval_scalar_prefix(is_video_pass=True, stats_during_training=False) == "Eval/"
    assert _eval_scalar_prefix(is_video_pass=True, stats_during_training=True) == "Eval/video/"
    assert _eval_scalar_prefix(is_video_pass=False, stats_during_training=True) == "Eval/"
    assert _eval_scalar_prefix(is_video_pass=False, stats_during_training=False) == "Eval/"


# ---------------------------------------------------------------------------
# T3 — locked seed policy: paired eval (video pass is a bit-identical prefix
#       of the stats pass when both share the same seed). Expected to pass
#       both before and after the fix -- it documents/pins a decision, not a
#       regression.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def eval_agent_and_params():
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


def test_video_and_stats_share_seed_paired_eval(eval_agent_and_params, tmp_path):
    world_model, actor, env_params, env_cfg = eval_agent_and_params

    result3 = dreamer_srl_eval_rollout(
        world_model=world_model, actor=actor, env_params=env_params, config=env_cfg,
        num_episodes=3, seed=0, results_dir=str(tmp_path / 'v3'), checkpoint_pct=50,
        render_video=False, quiet=True,
    )
    result8 = dreamer_srl_eval_rollout(
        world_model=world_model, actor=actor, env_params=env_params, config=env_cfg,
        num_episodes=8, seed=0, results_dir=str(tmp_path / 'v8'), checkpoint_pct=50,
        render_video=False, quiet=True,
    )

    assert result3['episode_rewards'] == result8['episode_rewards'][:3], (
        "video-pass (N=3) episode_rewards must be a bit-identical prefix of the "
        "stats-pass (N=8) episode_rewards under a shared seed -- this is the "
        "locked, deliberate paired-eval design (see plan §Decided semantics)."
    )
    assert result3['episode_lengths'] == result8['episode_lengths'][:3], (
        "video-pass (N=3) episode_lengths must be a bit-identical prefix of the "
        "stats-pass (N=8) episode_lengths under a shared seed."
    )
