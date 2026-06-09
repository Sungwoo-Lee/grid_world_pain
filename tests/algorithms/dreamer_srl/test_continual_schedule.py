"""Unit tests for ContinualSchedule + _build_continual_schedule.

Tests:
  A. Validation — _build_continual_schedule raises on:
      (a) non-increasing boundaries
      (b) length mismatch boundaries vs stages
      (c) length mismatch freqs vs stages
      (d) boundaries[0] <= 0
      (e) freq <= 0
  B. Accepts a valid 3-stage schedule and returns correct fields.
  C. stage_for_episode mapping: for boundaries [10, 25, 40]:
      episodes 0..9   -> 0
      episodes 10..24 -> 1
      episodes 25..39 -> 2
      episodes 40+    -> 2  (last stage clamped)

Must FAIL on pre-change code (_build_continual_schedule does not exist) and
PASS after implementation.

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_continual_schedule.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import yaml

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

from src.algorithms.dreamer_srl.dreamer_srl_main import (
    ContinualSchedule,
    _build_continual_schedule,
)

_PROJECT_ROOT = _REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers: build a minimal temporary configs_dir + schedule YAML
# ---------------------------------------------------------------------------

# We use the real curriculum configs to avoid writing large env YAMLs.
_CURRICULUM_DIR = os.path.join(
    _REPO_ROOT, "configs", "experiment", "dreamer_srl_curriculum"
)
_SCHEDULE_PATH = os.path.join(
    _REPO_ROOT, "configs", "continual", "dreamer_srl_3stage_size_curriculum.yaml"
)

# Skip all tests that need the real configs if they're not present.
_REAL_CONFIGS_AVAILABLE = (
    os.path.isdir(_CURRICULUM_DIR)
    and len([f for f in os.listdir(_CURRICULUM_DIR) if f.endswith(".yaml")]) >= 3
    and os.path.isfile(_SCHEDULE_PATH)
)


def _write_tiny_yaml(path: str, content: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(content, f)


def _make_tmp_configs_dir(tmpdir: str, n: int = 3) -> str:
    """Create n minimal env stub YAMLs in tmpdir, returning its path."""
    configs_dir = os.path.join(tmpdir, "stages")
    os.makedirs(configs_dir, exist_ok=True)
    for i in range(n):
        stub = {"environment": {"height": 5, "width": 5, "max_steps": 100}}
        _write_tiny_yaml(os.path.join(configs_dir, f"0{i+1}_stage{i}.yaml"), stub)
    return configs_dir


def _make_tmp_schedule(tmpdir: str, boundaries: list, freqs: list) -> str:
    path = os.path.join(tmpdir, "schedule.yaml")
    _write_tiny_yaml(path, {"continual": {
        "episode_boundaries": boundaries,
        "checkpoint_frequencies": freqs,
    }})
    return path


# ---------------------------------------------------------------------------
# A. Validation — raises on bad inputs
# ---------------------------------------------------------------------------

class TestValidation:

    def test_non_increasing_boundaries(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 5, 40], [1, 1, 1])
        with pytest.raises(ValueError, match="strictly increasing"):
            _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)

    def test_equal_boundaries_not_strictly_increasing(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 10, 40], [1, 1, 1])
        with pytest.raises(ValueError, match="strictly increasing"):
            _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)

    def test_boundaries_length_mismatch(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        # 2 boundaries for 3 stages
        schedule_path = _make_tmp_schedule(tmpdir, [10, 40], [1, 1])
        with pytest.raises(ValueError, match="episode_boundaries length"):
            _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)

    def test_freqs_length_mismatch(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        # 2 freqs for 3 stages
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [5, 5])
        with pytest.raises(ValueError, match="checkpoint_frequencies length"):
            _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)

    def test_first_boundary_zero(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [0, 25, 40], [1, 1, 1])
        with pytest.raises(ValueError, match="episode_boundaries\\[0\\] must be > 0"):
            _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)

    def test_freq_zero(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [1, 0, 1])
        with pytest.raises(ValueError, match="checkpoint_frequencies must all be > 0"):
            _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)

    def test_missing_configs_dir(self, tmp_path):
        tmpdir = str(tmp_path)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [1, 1, 1])
        with pytest.raises(ValueError, match="is not a directory"):
            _build_continual_schedule(
                _PROJECT_ROOT,
                os.path.join(tmpdir, "nonexistent"),
                schedule_path,
            )

    def test_empty_configs_dir(self, tmp_path):
        tmpdir = str(tmp_path)
        empty_dir = os.path.join(tmpdir, "empty")
        os.makedirs(empty_dir)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [1, 1, 1])
        with pytest.raises(ValueError, match="No \\*.yaml files found"):
            _build_continual_schedule(_PROJECT_ROOT, empty_dir, schedule_path)


# ---------------------------------------------------------------------------
# B. Valid schedule — returns correct ContinualSchedule fields
# ---------------------------------------------------------------------------

class TestValidSchedule:

    def test_returns_correct_num_stages(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [5, 10, 20])
        sched = _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)
        assert sched.num_stages == 3

    def test_returns_correct_boundaries(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [5, 10, 20])
        sched = _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)
        assert sched.episode_boundaries == [10, 25, 40]
        assert sched.checkpoint_frequencies == [5, 10, 20]

    def test_stage_names_alphabetical(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [5, 10, 20])
        sched = _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)
        assert sched.stage_names == sorted(sched.stage_names), (
            "Stage names must be in alphabetical order"
        )

    def test_stage_configs_list_length(self, tmp_path):
        tmpdir = str(tmp_path)
        configs_dir = _make_tmp_configs_dir(tmpdir, n=3)
        schedule_path = _make_tmp_schedule(tmpdir, [10, 25, 40], [5, 10, 20])
        sched = _build_continual_schedule(_PROJECT_ROOT, configs_dir, schedule_path)
        assert len(sched.stage_configs) == 3

    @pytest.mark.skipif(
        not _REAL_CONFIGS_AVAILABLE,
        reason="Real curriculum configs not found — skipping real-file test"
    )
    def test_real_curriculum_configs_load(self):
        """Smoke test: the real 3-stage dreamer_srl curriculum loads without error."""
        sched = _build_continual_schedule(
            _PROJECT_ROOT, _CURRICULUM_DIR, _SCHEDULE_PATH
        )
        assert sched.num_stages == 3
        assert sched.episode_boundaries == [15000, 75000, 760000]
        assert sched.checkpoint_frequencies == [7500, 30000, 100000]


# ---------------------------------------------------------------------------
# C. stage_for_episode mapping
# ---------------------------------------------------------------------------

class TestStageForEpisode:
    """For boundaries [10, 25, 40]:
      ep 0..9   -> stage 0
      ep 10..24 -> stage 1
      ep 25..39 -> stage 2
      ep 40+    -> stage 2 (clamped to last)
    """

    @pytest.fixture(autouse=True)
    def schedule(self):
        self.sched = ContinualSchedule(
            stage_config_paths=["/fake/01.yaml", "/fake/02.yaml", "/fake/03.yaml"],
            stage_names=["01_a", "02_b", "03_c"],
            stage_configs=[None, None, None],
            episode_boundaries=[10, 25, 40],
            checkpoint_frequencies=[5, 10, 20],
        )

    @pytest.mark.parametrize("ep,expected", [
        (0, 0), (1, 0), (9, 0),
        (10, 1), (15, 1), (24, 1),
        (25, 2), (30, 2), (39, 2),
        (40, 2), (99, 2), (1000, 2),
    ])
    def test_stage_for_episode(self, ep, expected):
        result = self.sched.stage_for_episode(ep)
        assert result == expected, (
            f"stage_for_episode({ep}) = {result}, expected {expected}"
        )

    def test_num_stages(self):
        assert self.sched.num_stages == 3
