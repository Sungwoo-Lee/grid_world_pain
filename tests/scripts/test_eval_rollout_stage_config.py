"""Regression test — scripts/eval/eval_rollout.py continual-run stage config selection.

Diagnosis finding (Finding E / L2, docs/develop/active/diagnosis/
v3_pipeline_correctness_diagnosis.md): for a continual (multi-stage curriculum)
run, evaluation always loaded the STAGE-0 environment config (`models/config.yaml`),
so a later-stage checkpoint was evaluated in the wrong environment.

Fix under test: `_resolve_continual_stage_config()` in scripts/eval/eval_rollout.py.
It detects a continual run via `models/schedule.yaml`, reads the checkpoint's OWN
saved `stage` field (not a recomputation from episode_boundaries -- the two can
disagree, see test below), and resolves the matching `stage_XX_<name>.yaml`.

Tests:
  1. A checkpoint saved as stage 0 resolves to the stage-0 config file.
  2. A checkpoint saved as stage 1 resolves to the stage-1 config file (the bug:
     without the fix, both would resolve to the same stage-0 config.yaml).
  3. A checkpoint's *recorded* stage can lag behind what a naive recompute from
     episode_boundaries would give (per-iteration transition-check drift); the
     fix must trust the checkpoint's own field, not the recompute.
  4. An explicitly different --config (not the run's own stage-0 config.yaml) is
     left untouched -- the caller's choice always wins.
  5. A non-continual run (no schedule.yaml) returns None -- unchanged behavior.
"""
import os
import sys

import jax
import orbax.checkpoint as ocp
import pytest
import yaml

jax.config.update("jax_platform_name", "cpu")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "eval"))

import eval_rollout as er  # noqa: E402


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _save_stage_checkpoint(models_dir, step, stage):
    """Mirror train.py's checkpoint save: a StandardSave payload containing a
    'stage' leaf alongside the (here, dummy) model/optimizer state."""
    checkpointer = ocp.CheckpointManager(
        os.path.abspath(models_dir),
        checkpointers=ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True),
    )
    ckpt_data = {"stage": stage, "episode": step, "dummy_param": 0.0}
    checkpointer.save(step, args=ocp.args.StandardSave(ckpt_data))
    checkpointer.wait_until_finished()


def _make_continual_run(tmp_path):
    """Build a minimal on-disk continual-run layout: schedule.yaml, a stage-0
    config.yaml (the default eval target), two stage_XX configs, and two saved
    checkpoints tagged stage 0 and stage 1 respectively."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    _write_yaml(str(models_dir / "schedule.yaml"), {
        "continual": {
            "episode_boundaries": [10, 20],
            "checkpoint_frequencies": [5, 5],
            "stage_names": ["00-a", "01-b"],
        }
    })
    _write_yaml(str(models_dir / "config.yaml"), {"marker": "stage0-dump"})
    _write_yaml(str(models_dir / "stage_00_00-a.yaml"), {"marker": "stage0"})
    _write_yaml(str(models_dir / "stage_01_01-b.yaml"), {"marker": "stage1"})

    return models_dir


def test_stage0_checkpoint_resolves_to_stage0_config(tmp_path):
    models_dir = _make_continual_run(tmp_path)
    _save_stage_checkpoint(str(models_dir), step=8, stage=0)

    resolved = er._resolve_continual_stage_config(
        str(models_dir / "config.yaml"), str(models_dir / "8"), quiet=True)

    assert resolved == str(models_dir / "stage_00_00-a.yaml")


def test_stage1_checkpoint_resolves_to_stage1_config(tmp_path):
    """The bug this fixes: a later-stage checkpoint must NOT resolve to
    stage-0's config.yaml."""
    models_dir = _make_continual_run(tmp_path)
    _save_stage_checkpoint(str(models_dir), step=25, stage=1)

    resolved = er._resolve_continual_stage_config(
        str(models_dir / "config.yaml"), str(models_dir / "25"), quiet=True)

    assert resolved == str(models_dir / "stage_01_01-b.yaml")
    assert resolved != str(models_dir / "config.yaml")


def test_checkpoint_recorded_stage_overrides_naive_boundary_recompute(tmp_path):
    """A checkpoint saved with step=12 (past episode_boundaries[0]=10, so a
    naive recompute via stage_for_episode would say stage 1) but whose OWN
    recorded 'stage' field is still 0 (the per-iteration transition-check
    drift documented in train.py) must resolve to stage 0 -- the checkpoint's
    own field is ground truth, not the boundary recompute."""
    models_dir = _make_continual_run(tmp_path)
    _save_stage_checkpoint(str(models_dir), step=12, stage=0)

    resolved = er._resolve_continual_stage_config(
        str(models_dir / "config.yaml"), str(models_dir / "12"), quiet=True)

    assert resolved == str(models_dir / "stage_00_00-a.yaml")


def test_explicit_non_default_config_is_left_untouched(tmp_path):
    models_dir = _make_continual_run(tmp_path)
    _save_stage_checkpoint(str(models_dir), step=25, stage=1)

    other_cfg = tmp_path / "some_other_env_config.yaml"
    _write_yaml(str(other_cfg), {"marker": "ood-eval-config"})

    resolved = er._resolve_continual_stage_config(
        str(other_cfg), str(models_dir / "25"), quiet=True)

    assert resolved is None


def test_non_continual_run_returns_none(tmp_path):
    """No schedule.yaml next to the checkpoint -> not a continual run ->
    unchanged single-config behavior."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_yaml(str(models_dir / "config.yaml"), {"marker": "single-config"})
    _save_stage_checkpoint(str(models_dir), step=819, stage=0)

    resolved = er._resolve_continual_stage_config(
        str(models_dir / "config.yaml"), str(models_dir / "819"), quiet=True)

    assert resolved is None
