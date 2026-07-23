"""Regression test — Bug 1 (Track B, 2026-07-23 eval/logging fix plan):
`run_tag` derivation in scripts/eval/eval_rollout.py collapsed run identity
for the rPPO step-dir invocation form `<run>/models/<step>`.

Pre-fix, the derivation special-cased only the Dreamer container dir name
("checkpoints"), so for the rPPO form `run_tag == "models"` and different
runs' evaluations collided under `results/eval/models/<step>`.

Fix (docs/develop/active/issues/FIX_EVAL_LOGGING_TRACK_B_20260723.md, Bug 1):
walk up past EITHER known checkpoint-container dir name ("checkpoints" for
Dreamer, "models" for rPPO) to find the run directory.
"""
import os
import sys
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "eval"))

import eval_rollout as er  # noqa: E402


def test_step_dir_invocation_preserves_run_identity():
    # rPPO step-dir form: <run>/models/<step> -- the bug this test targets.
    assert er._derive_run_tag(Path("/x/runA/models/8900007")) == "runA"

    # Dreamer step-dir form: <run>/checkpoints/<step> -- must remain unchanged.
    assert er._derive_run_tag(Path("/x/runB/checkpoints/500")) == "runB"

    # CheckpointManager-root invocation form: <run>/models -- must remain unchanged.
    assert er._derive_run_tag(Path("/x/runC/models")) == "runC"

    # Bare-dir form (no known container dirname in the parent) falls back to
    # the parent directory name.
    assert er._derive_run_tag(Path("/x/runD/some_other_dir")) == "runD"
