"""Regression test (H3): a missing config path must be a hard error.

Bug context
-----------
diag_fable5_20260704/01_train_entry_config.md Finding 5: ``Config.load_yaml``
printed a one-line warning and returned an EMPTY config when the file did not
exist. A typo'd ``--config`` path therefore trained silently on
``configs/environment/default.yaml`` — bypassing the project's strict
no-fallback config rule (every mandatory key resolved from the default base,
so no ``ValueError`` ever fired).

Fix: ``Config.load_yaml`` raises ``FileNotFoundError`` on a missing path.
Intentional optional loads must guard with ``os.path.exists`` at the call
site (all existing ones already do — see the caller enumeration in
docs/develop/active/issues/diag_fable5_20260704/fix_plan_h1h2h3_resume_config.md).

Red -> green contract
---------------------
BEFORE the fix: the two missing-path tests FAIL (load_yaml returns an empty
``Config`` after a print — a genuine behavioral red).
AFTER the fix: all three tests pass.

Run with::

    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/env/test_config_strict_load.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

# Project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.utils.config import Config
from src.environment.config_loader import load_env_config


def test_load_yaml_missing_path_raises():
    """A nonexistent path must raise, never return an empty config."""
    with pytest.raises(FileNotFoundError):
        Config.load_yaml("/nonexistent/definitely_missing.yaml")


def test_load_env_config_missing_path_raises():
    """The exact train.py --config path (load_env_config -> _resolve_extends
    -> Config.load_yaml) must raise on a typo'd path."""
    with pytest.raises(FileNotFoundError):
        load_env_config("/nonexistent/typo.yaml")


def test_load_yaml_existing_path_still_works():
    """Guard against over-tightening: a real tracked config still loads."""
    cfg = Config.load_yaml(os.path.join(_ROOT, "configs", "environment", "default.yaml"))
    assert cfg.to_dict(), "default.yaml loaded as empty — load_yaml is broken"
    assert cfg.get_mandatory("environment.height") is not None
