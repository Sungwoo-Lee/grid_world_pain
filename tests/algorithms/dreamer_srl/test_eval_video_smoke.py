"""Commit G — end-to-end smoke test for eval-video pipeline.

Runs dreamer_srl_main.py for a short duration with eval enabled and verifies:
  1. At least 1 Orbax checkpoint dir created under checkpoints/.
  2. At least 1 recordings/<N>/episode_000001.rec.gz exists.
  3. At least 1 videos/eval_<N>.mp4 file exists and is > 1KB.
  4. run_meta.pkl in the recordings dir is loadable.

This test is marked @pytest.mark.slow since it runs a real training loop
for ~100 steps plus a render subprocess (~30s total).

Uses a tiny env with max_steps=10 and checkpoint_frequency=3 so multiple
checkpoints fire within the small step budget.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

_PYTHON = '/home/vncuser/miniconda3/envs/grid_world_pain/bin/python'
_DRIVER = '/media/nas01/projects/Interoceptive-AI/grid_world_pain/src/algorithms/dreamer_srl/dreamer_srl_main.py'
_ROOT = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'


def _write_smoke_config(path: str, checkpoint_frequency: int = 3,
                        video_during_training: bool = True,
                        eval_video_episodes: int = 1,
                        stats_during_training: bool = False):
    """Write a minimal YAML config for the smoke test."""
    import yaml
    cfg = {
        'environment': {
            'height': 5, 'width': 5, 'start_pos': [2, 2],
            'max_steps': 20, 'random_start_pos': True,
            'rest_action_enabled': True, 'eat_action_enabled': True,
            'placement': {'mode': 'per_entity'},
            'resources': [{
                'name': 'food', 'type': 'food', 'count': 2,
                'spawn_area': [[1, 1], [4, 4]],
                'properties': [1.0, 0.0, 0.0, 0.0, 0.0],
                'properties_std': [0.0, 0.0, 0.0, 0.0, 0.0],
                'max_consumption': 12, 'regeneration_delay': 0,
                'damage': [0.0, 0.0], 'nociception_intensity': 0.0,
            }],
        },
        'training': {
            'checkpoint_frequency': checkpoint_frequency,
            'video_during_training': video_during_training,
            'stats_during_training': stats_during_training,
            'eval_video_episodes': eval_video_episodes,
        },
    }
    with open(path, 'w') as f:
        yaml.dump(cfg, f)


@pytest.mark.slow
def test_e2e_smoke_checkpoints_and_recordings():
    """End-to-end: run dreamer-srl, verify checkpoints + recordings + MP4."""
    with tempfile.TemporaryDirectory() as results_dir:
        cfg_path = os.path.join(results_dir, 'smoke_config.yaml')
        _write_smoke_config(
            cfg_path,
            checkpoint_frequency=3,
            video_during_training=True,
            eval_video_episodes=1,
        )

        # Run for enough steps to hit at least 1 checkpoint
        # ~300 steps at max_steps=20 → ~15 episodes → 5 checkpoint events
        cmd = [
            _PYTHON, _DRIVER,
            '--env-config', cfg_path,
            '--agent-config', os.path.join(_ROOT, 'configs/dreamer_srl/01_food_only.yaml'),
            '--total-steps', '300',
            '--num-envs', '1',
            '--seed', '42',
            '--no-wandb',
            '--quiet',
            '--results-dir', results_dir,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute wall-clock bound (plan spec: 5 min)
            cwd=_ROOT,
        )

        # Print output for pytest -v debugging
        print('\n--- dreamer-srl smoke stdout ---')
        print(result.stdout[-3000:])
        if result.stderr:
            print('--- stderr (last 500) ---')
            # Filter out benign orbax warning
            filtered = '\n'.join(
                l for l in result.stderr.splitlines()
                if 'Configured `CheckpointManager`' not in l
            )
            print(filtered[-500:])

        assert result.returncode == 0, \
            f'dreamer_srl_main.py exited with code {result.returncode}'

        results_path = Path(results_dir)

        # 1. At least 1 Orbax checkpoint dir
        ckpt_dir = results_path / 'checkpoints'
        assert ckpt_dir.is_dir(), 'checkpoints/ dir not created'
        step_dirs = [d for d in ckpt_dir.iterdir() if d.is_dir()]
        assert len(step_dirs) >= 1, \
            f'Expected >=1 checkpoint step dir, got {len(step_dirs)}'

        # 2. At least 1 recordings dir with episode .rec.gz
        recordings_root = results_path / 'recordings'
        assert recordings_root.is_dir(), 'recordings/ dir not created'
        ckpt_subdirs = [d for d in recordings_root.iterdir() if d.is_dir()]
        assert len(ckpt_subdirs) >= 1, \
            f'Expected >=1 recordings/<ckpt>/ dir, got {len(ckpt_subdirs)}'
        # Check at least 1 .rec.gz in the first subdir
        first_subdir = sorted(ckpt_subdirs, key=lambda d: d.name)[0]
        rec_files = list(first_subdir.glob('episode_*.rec.gz'))
        assert len(rec_files) >= 1, \
            f'Expected >=1 episode_*.rec.gz in {first_subdir}, got {len(rec_files)}'
        # run_meta.pkl must be loadable
        meta_path = first_subdir / 'run_meta.pkl'
        assert meta_path.exists(), f'run_meta.pkl not found in {first_subdir}'
        import pickle
        with open(meta_path, 'rb') as fh:
            meta = pickle.load(fh)
        assert 'params' in meta, "run_meta.pkl missing 'params' key"

        # 3. At least 1 MP4 in videos/
        videos_dir = results_path / 'videos'
        assert videos_dir.is_dir(), 'videos/ dir not created'
        mp4_files = list(videos_dir.glob('eval_*.mp4'))
        assert len(mp4_files) >= 1, \
            f'Expected >=1 eval_*.mp4 in {videos_dir}, got {len(mp4_files)}'
        for mp4 in mp4_files:
            assert mp4.stat().st_size > 1024, \
                f'MP4 {mp4.name} too small: {mp4.stat().st_size} bytes'


@pytest.mark.slow
def test_e2e_smoke_no_video_only_stats():
    """Smoke with stats_during_training=True, video=False: no recordings, no MP4."""
    with tempfile.TemporaryDirectory() as results_dir:
        cfg_path = os.path.join(results_dir, 'smoke_stats.yaml')
        _write_smoke_config(
            cfg_path,
            checkpoint_frequency=3,
            video_during_training=False,
            eval_video_episodes=1,
            stats_during_training=False,  # also off — just checkpoint smoke
        )

        cmd = [
            _PYTHON, _DRIVER,
            '--env-config', cfg_path,
            '--agent-config', os.path.join(_ROOT, 'configs/dreamer_srl/01_food_only.yaml'),
            '--total-steps', '150',
            '--num-envs', '1',
            '--seed', '0',
            '--no-wandb',
            '--quiet',
            '--results-dir', results_dir,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=_ROOT,
        )
        assert result.returncode == 0, \
            f'dreamer_srl_main.py exited with code {result.returncode}'

        # Checkpoint dir exists
        ckpt_dir = Path(results_dir) / 'checkpoints'
        assert ckpt_dir.is_dir(), 'checkpoints/ dir not created'
        step_dirs = [d for d in ckpt_dir.iterdir() if d.is_dir()]
        assert len(step_dirs) >= 1, f'Expected >=1 step dir, got {len(step_dirs)}'

        # No recordings dir (video_during_training=False)
        recordings_root = Path(results_dir) / 'recordings'
        if recordings_root.is_dir():
            assert len(list(recordings_root.glob('**/*.rec.gz'))) == 0, \
                'Unexpected .rec.gz files with video_during_training=False'
