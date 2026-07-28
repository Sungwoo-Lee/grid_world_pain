"""Regression test: dreamer-srl curriculum resume must rebuild the env for the
restored stage.

Bug context (curriculum audit 2026-07-28)
-----------------------------------------
``--load-checkpoint`` restores the stage index (``dreamer_srl_main.py``
``current_stage = int(_restored['stage'])``) but never rebuilds the
environment for that stage. ``ParallelEnv`` is only constructed at setup
(from stage 0) and inside the in-loop transition block — and on a stage-N
resume the in-loop check ``_new_stage != current_stage`` compares equal, so
the transition never fires. Training silently proceeds on the STAGE-0 world
with stage-N bookkeeping. This is the dreamer-srl twin of the rPPO H2 bug
(diag_fable5_20260704/01 Finding 2), whose fix at train.py:1368-1392 is the
ported reference; the rPPO regression test is
tests/training/test_continual_resume_rebuild.py.

Red -> green contract
---------------------
BEFORE the fix: the resume run prints no ``[RESUME] Stage 1:`` marker and its
startup banner shows the STAGE-0 grid (5x5) -> the marker + grid asserts fail.
AFTER the fix: the resumed run prints the marker and the banner shows the
stage-1 grid (8x8) — a directly stage-distinguishing observable.

Run with::

    JAX_PLATFORMS=cpu /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_continual_resume_rebuild.py -v
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap

import pytest

_PYTHON = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_DRIVER = os.path.join(
    _REPO_ROOT, "src", "algorithms", "dreamer_srl", "dreamer_srl_main.py"
)
_AGENT_CONFIG = os.path.join(
    _REPO_ROOT, "configs", "models", "dreamer_srl", "01_food_only.yaml"
)

# ---------------------------------------------------------------------------
# Stage configs — same tiny food-only scene at two DIFFERENT grid sizes
# (5x5 stage 0, 8x8 stage 1). Grid size does not affect the modality
# fingerprint (the real 3-stage size curriculum runs 5x5 -> 10x10), so the
# pre-flight check passes while the grid dimension remains a directly
# assertable stage-distinguishing observable in the startup banner.
# ---------------------------------------------------------------------------

_STAGE_YAML = textwrap.dedent("""\
    # Dreamer curriculum-resume regression test — stage {name} ({hw}x{hw} grid)
    extends: environment/default

    # Keep checkpoint saves fast: no video/stats eval passes.
    training:
      video_during_training: false
      stats_during_training: false

    environment:
      height: {hw}
      width: {hw}
      start_pos: [3, 3]
      max_steps: 20
      location_areas:
        - type: "grass"
          area: [[1, 1], [{hw}, {hw}]]
        - type: "sand"
          area: [[0, 0], [0, 0]]
      resources:
        - name: "food"
          type: "food"
          count: 2
          spawn_area: [[1, 1], [{hw}, {hw}]]
          properties: [1.0, 0.0, 0.0, 0.0, 0.0]
          properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
          visual_properties: [0, 0, 0, 1, 0, 0, 0, 0]
          visual_properties_std: [0, 0, 0, 0, 0, 0, 0, 0]
          max_consumption: 12
          regeneration_delay: 0
          damage: [0.0, 0.0]
          nociception_intensity: 0.0
      entities: []
      obstacles: []
""")

# Schedule: stage 0 = episodes [0, 3), stage 1 = [3, 8).
# checkpoint_frequencies 2 -> saves land at episode labels ~2, 4, 6, 8, so at
# least one checkpoint is saved mid-stage-1 (label in [3, 7]).
_SCHEDULE_YAML = textwrap.dedent("""\
    continual:
      episode_boundaries: [3, 8]
      checkpoint_frequencies: [2, 2]
""")


def _write(path: str, content: str) -> None:
    with open(path, "w") as fh:
        fh.write(content)


def _run_driver(extra_args, cwd):
    cmd = [
        _PYTHON, _DRIVER,
        "--agent-config", _AGENT_CONFIG,
        "--num-envs", "2",
        "--seed", "42",
        "--no-wandb",
        # No --quiet: the [RESUME]/[STAGE] markers + banner must reach stdout.
    ] + extra_args
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, cwd=cwd, env=env)


@pytest.mark.slow
def test_dreamer_curriculum_resume_rebuilds_stage_env():
    """Create a 2-stage run with checkpoints across the stage boundary, resume
    from a mid-stage-1 checkpoint, and assert the env was rebuilt for stage 1
    (8x8 grid in the banner + [RESUME] marker), not left on the stage-0 5x5."""
    with tempfile.TemporaryDirectory(prefix="dsrl_resume_rebuild_") as tmpdir:
        stages_dir = os.path.join(tmpdir, "stages")
        os.makedirs(stages_dir)
        _write(os.path.join(stages_dir, "01_a.yaml"),
               _STAGE_YAML.format(name="A", hw=5))
        _write(os.path.join(stages_dir, "02_b.yaml"),
               _STAGE_YAML.format(name="B", hw=8))
        schedule_path = os.path.join(tmpdir, "schedule.yaml")
        _write(schedule_path, _SCHEDULE_YAML)

        common = [
            "--configs-dir", stages_dir,
            "--continual-schedule", schedule_path,
        ]

        # --- Run 1: full 8-episode curriculum, checkpoints across the boundary.
        results1 = os.path.join(tmpdir, "results_run1")
        os.makedirs(results1)
        r1 = _run_driver(common + ["--results-dir", results1], cwd=_REPO_ROOT)
        assert r1.returncode == 0, (
            f"Run 1 (checkpoint creation) failed rc={r1.returncode}.\n"
            f"STDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}")

        ckpt_dir = os.path.join(results1, "checkpoints")
        assert os.path.isdir(ckpt_dir), f"No checkpoints/ dir under {results1}"
        labels = sorted(int(d) for d in os.listdir(ckpt_dir) if d.isdigit())
        assert labels, f"No checkpoint episode dirs under {ckpt_dir}"

        # Pick a mid-stage-1 checkpoint: episode label in [3, 7] (stage 1
        # covers eps [3, 8); label 8 would leave no episodes to train).
        stage1_labels = [l for l in labels if 3 <= l <= 7]
        assert stage1_labels, (
            f"No checkpoint label in [3, 7] (mid-stage-1); saved labels were "
            f"{labels} — adjust the schedule/frequencies.")
        resume_ep = stage1_labels[-1]

        # --- Run 2: resume from the mid-stage-1 checkpoint. ---
        results2 = os.path.join(tmpdir, "results_run2")
        os.makedirs(results2)
        r2 = _run_driver(
            common + ["--results-dir", results2,
                      "--load-checkpoint", ckpt_dir,
                      "--load-episode", str(resume_ep)],
            cwd=_REPO_ROOT)
        combined = r2.stdout + r2.stderr

        # 1. Clean exit, no traceback.
        assert "Traceback" not in combined, (
            f"Traceback in resume run.\nSTDOUT:\n{r2.stdout}\n"
            f"STDERR:\n{r2.stderr}")
        assert r2.returncode == 0, (
            f"Resume run failed rc={r2.returncode}.\nSTDOUT:\n{r2.stdout}\n"
            f"STDERR:\n{r2.stderr}")

        # 2. The rebuild marker: env rebuilt for stage 1, not left at stage 0.
        assert "[RESUME] Stage 1:" in combined, (
            f"No '[RESUME] Stage 1:' marker — the post-restore rebuild did "
            f"not fire (resume is training on the stage-0 world).\n"
            f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")

        # 3. Stage-distinguishing observable: the startup banner prints
        #    env_params.height x width AFTER the resume block, so a rebuilt
        #    stage-1 env shows the 8x8 grid (stage 0 is 5x5).
        m = re.search(r"Grid Size\.*\s*(\d+x\d+)", combined)
        assert m, f"No 'Grid Size' banner line found.\nSTDOUT:\n{r2.stdout}"
        assert m.group(1) == "8x8", (
            f"Banner grid is {m.group(1)} — resume is on the WRONG stage's "
            f"world (expected stage 1's 8x8).\nSTDOUT:\n{r2.stdout}")

        # 4. No spurious stage transition after the resume marker (episodes
        #    resume_ep..8 are all stage 1).
        post_resume = combined.split("[RESUME] Stage 1:", 1)[1]
        assert "[STAGE]" not in post_resume, (
            f"Spurious in-loop stage transition after resume.\n"
            f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")
