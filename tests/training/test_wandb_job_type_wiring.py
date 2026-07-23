"""Regression test — Bug 3 (Track B, 2026-07-23 eval/logging fix plan):
`wandb.job_type` was never wired into `wandb.init`'s kwargs in train.py, and
its YAML default was misspelled ("defualt").

Fix (docs/develop/active/issues/FIX_EVAL_LOGGING_TRACK_B_20260723.md, Bug 3):
- configs/logger/wandb.yaml: "defualt" -> "default".
- train.py: wandb_kwargs now includes "job_type": config.get_mandatory('wandb.job_type').

Invoking train.py's wandb-init block standalone requires a fully-populated
CLI/config/tag context that is impractical to construct in a unit test, so
this test asserts the two textual facts the fix guarantees instead (per the
plan's documented lighter-assertion fallback):
  1. configs/logger/wandb.yaml's job_type value is no longer the typo.
  2. train.py's wandb_kwargs dict literal includes a "job_type" key sourced
     from config.get_mandatory('wandb.job_type').
"""
import os
import re

import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WANDB_YAML = os.path.join(_REPO, "configs", "logger", "wandb.yaml")
_TRAIN_PY = os.path.join(_REPO, "train.py")


def _wandb_kwargs_literal():
    """Extract the `wandb_kwargs = { ... }` dict literal text from train.py."""
    with open(_TRAIN_PY) as f:
        src = f.read()
    m = re.search(r"wandb_kwargs\s*=\s*\{(.*?)\n\s*\}", src, re.DOTALL)
    assert m is not None, "could not locate `wandb_kwargs = {...}` literal in train.py"
    return m.group(1)


def test_job_type_passed_to_wandb_init():
    # (1) config default typo fixed
    with open(_WANDB_YAML) as f:
        wandb_cfg = yaml.safe_load(f)
    assert wandb_cfg["wandb"]["job_type"] == "default"

    # (2) train.py's wandb_kwargs literal wires job_type from config
    kwargs_literal = _wandb_kwargs_literal()
    assert '"job_type"' in kwargs_literal
    assert "wandb.job_type" in kwargs_literal
