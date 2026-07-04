"""Regression test (H2): continual resume must rebuild the env for the
restored stage.

Bug context
-----------
diag_fable5_20260704/01_train_entry_config.md Finding 2: the environment is
built from stage 0's config BEFORE checkpoint restoration, and restoring the
saved stage number makes the in-loop transition check
(``schedule.stage_for_episode(ep) != current_stage``) compare equal — so the
rebuild never fires. A stage-N resume therefore trained stage-N counters and
WandB tags on a stage-0 world.

Fix: an unconditional post-restore rebuild block in train.py derives the
stage from the schedule (``stage_for_episode(total_episodes_completed)``),
rebuilds the env from ``schedule.stage_configs[stage]``, resets the rPPO
recurrent state, and prints a ``[RESUME] Stage k:`` marker.

Red -> green contract
---------------------
BEFORE the fix: the ``[RESUME]`` marker does not exist anywhere in train.py
-> the marker assert fails (and behaviorally the resumed run really is on
the stage-0 world; with H1 unfixed the restore itself also silently no-ops).
AFTER the fix: the resumed run prints ``[RESUME] Stage 1:`` and no spurious
stage-0 transition.

Run with::

    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/training/test_continual_resume_rebuild.py -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap

import pytest

_PYTHON = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TRAIN_PY = os.path.join(_REPO_ROOT, "train.py")
_AGENT_CONFIG = os.path.join(
    _REPO_ROOT, "configs", "models", "recurrent_ppo", "recurrent_ppo.yaml"
)

# ---------------------------------------------------------------------------
# Stage configs — same tiny 5x5 scene as tests/training/test_continual_bm_transition.py
# (identical obs/action dims across stages, mandatory for a valid continual run).
# ---------------------------------------------------------------------------

_STAGE_YAML = textwrap.dedent("""\
    # Continual-resume regression test — stage {name} (5x5, food + hiding predator)
    extends: environment/default

    # Keep the checkpoint saves fast: no video/stats eval passes.
    training:
      video_during_training: false
      stats_during_training: false

    environment:
      height: 5
      width: 5
      start_pos: [3, 3]
      max_steps: 20
      location_areas:
        - type: "grass"
          area: [[1, 1], [5, 5]]
        - type: "sand"
          area: [[0, 0], [0, 0]]
      resources:
        - name: "food"
          type: "food"
          count: 2
          spawn_area: [[1, 1], [5, 5]]
          properties: [1.0, 0.0, 0.0, 0.0, 0.0]
          properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
          visual_properties: [0, 0, 0, 1, 0, 0, 0, 0]
          visual_properties_std: [0, 0, 0, 0, 0, 0, 0, 0]
          max_consumption: 12
          regeneration_delay: 0
          damage: [0.0, 0.0]
          nociception_intensity: 0.0
        - name: "hiding_predator"
          type: "hiding_predator"
          count: 1
          spawn_area: [[1, 1], [5, 5]]
          properties: [0.0, 0.0, 0.0, 0.0, 0.0]
          properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
          visual_properties: [0, 0, 0, 0, 1, 0, 0, 0]
          visual_properties_std: [0, 0, 0, 0, 0, 0, 0, 0]
          max_consumption: -1
          regeneration_delay: 20
          damage: [15.0, 45.0]
          nociception_intensity: 0.9
      entities: []
      obstacles: []
""")

# Schedule: stage 0 = episodes [0, 3), stage 1 = [3, 8).
# checkpoint_frequencies 2 -> checkpoints land at episode-numbered steps
# 2, 4, 6, 8 (approximately; >= milestones), so some are saved in stage 1.
_SCHEDULE_YAML = textwrap.dedent("""\
    continual:
      episode_boundaries: [3, 8]
      checkpoint_frequencies: [2, 2]
""")


def _write(path: str, content: str) -> None:
    with open(path, "w") as fh:
        fh.write(content)


def _run_train(extra_args, cwd):
    cmd = [
        _PYTHON, _TRAIN_PY,
        "--num-envs", "2",
        "--num-steps", "10",
        "--seed", "42",
        "--device", "cpu",
        "--no-wandb",
        # No --quiet: the [RESUME]/[STAGE] markers must reach stdout.
    ] + extra_args
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, cwd=cwd)


@pytest.mark.slow
def test_continual_resume_rebuilds_stage_env():
    """Create a 2-stage run, doctor the checkpoint dir so the latest step is
    mid-stage-1, resume, and assert the env was rebuilt for stage 1."""
    with tempfile.TemporaryDirectory(prefix="resume_rebuild_test_") as tmpdir:
        stages_dir = os.path.join(tmpdir, "stages")
        os.makedirs(stages_dir)
        _write(os.path.join(stages_dir, "00_a.yaml"), _STAGE_YAML.format(name="A"))
        _write(os.path.join(stages_dir, "01_b.yaml"), _STAGE_YAML.format(name="B"))
        schedule_path = os.path.join(tmpdir, "schedule.yaml")
        _write(schedule_path, _SCHEDULE_YAML)

        common = [
            "--configs-dir", stages_dir,
            "--continual-schedule", schedule_path,
            "--agent_config", _AGENT_CONFIG,
        ]

        # --- Run 1: create checkpoints across the stage boundary. ---
        results1 = os.path.join(tmpdir, "results_run1")
        os.makedirs(results1)
        r1 = _run_train(common + ["--results-dir", results1], cwd=_REPO_ROOT)
        assert r1.returncode == 0, (
            f"Run 1 (checkpoint creation) failed rc={r1.returncode}.\n"
            f"STDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}")

        models_dir = os.path.join(results1, "models")
        step_dirs = sorted(int(d) for d in os.listdir(models_dir) if d.isdigit())
        assert step_dirs, f"No checkpoint step dirs under {models_dir}"

        # --- Doctor: make the LATEST step a mid-stage-1 step (in [3, 7]). ---
        # Stage 1 covers episodes [3, 8); delete every step dir > 4 so the
        # latest remaining step is <= 4 but >= 3 (stage 1, episodes remain).
        for s in step_dirs:
            if s > 4:
                shutil.rmtree(os.path.join(models_dir, str(s)))
        remaining = sorted(int(d) for d in os.listdir(models_dir) if d.isdigit())
        assert remaining, "Doctoring removed every checkpoint step"
        latest = remaining[-1]
        assert 3 <= latest <= 7, (
            f"Latest remaining checkpoint step {latest} is not mid-stage-1 "
            f"(expected in [3, 7]); saved steps were {step_dirs} — adjust the "
            f"doctoring threshold.")

        # --- Run 2: resume from the doctored checkpoint dir. ---
        results2 = os.path.join(tmpdir, "results_run2")
        os.makedirs(results2)
        r2 = _run_train(
            common + ["--results-dir", results2,
                      "--load-checkpoint", models_dir],
            cwd=_REPO_ROOT)
        combined = r2.stdout + r2.stderr

        # 1. Clean exit, no traceback. (Pre-fix H1 alone would silently pass
        #    this — hence the marker assert below.)
        assert "Traceback" not in combined, (
            f"Traceback in resume run.\nSTDOUT:\n{r2.stdout}\n"
            f"STDERR:\n{r2.stderr}")
        assert r2.returncode == 0, (
            f"Resume run failed rc={r2.returncode}.\nSTDOUT:\n{r2.stdout}\n"
            f"STDERR:\n{r2.stderr}")

        # 2. The H2 marker: env rebuilt for stage 1, not left at stage 0.
        assert "[RESUME] Stage 1:" in combined, (
            f"No '[RESUME] Stage 1:' marker — the post-restore rebuild did "
            f"not fire (resume is on the stage-0 world).\n"
            f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")

        # 3. No spurious back-transition to stage 0 and no stage-0 transition
        #    after the resume marker (episodes 4..8 are all stage 1).
        post_resume = combined.split("[RESUME] Stage 1:", 1)[1]
        assert "-> 0:" not in post_resume, (
            f"Spurious back-transition to stage 0 after resume.\n"
            f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")
        assert "[STAGE] 0:" not in post_resume, (
            f"Spurious stage-0 transition line after resume.\n"
            f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")
