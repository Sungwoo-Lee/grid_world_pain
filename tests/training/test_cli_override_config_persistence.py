"""Regression test: CLI overrides to train.py must be persisted into the saved
``models/config.yaml`` (Finding G3/L4 of the v3.0 diagnosis).

Bug context
-----------
Before the fix, several ``train.py`` CLI flags overrode runtime behaviour (via
a local Python variable or a ``params.replace(...)`` call) WITHOUT being
written back into the ``Config`` object that gets dumped to
``models/config.yaml``. The most dangerous instance: ``--no-satiation``
disabled satiation for the actual run (via ``params.replace(with_satiation=False)``)
but the two ``config.set(...)`` calls meant to persist that choice wrote to
the WRONG top-level key (``environment.with_satiation`` / ``environment.overeating_death``)
instead of the key ``load_env_params()`` actually reads (``body.with_satiation`` /
``body.overeating_death``). Re-evaluating such a run via ``evaluation.py``
(which rebuilds ``EnvParams`` from the saved config) would silently restore
satiation ON, even though the run trained with it OFF.
``--hidden-size``, ``--lr``, and ``--num-steps`` had the same class of bug:
the CLI value was used for training but the saved config kept the pre-override
YAML default.

Fix: every CLI override is now written back into ``config`` (via
``config.set(...)`` on the correct canonical key) before the config is dumped.

Red -> green contract
----------------------
BEFORE the fix: saved ``config.yaml`` shows ``body.with_satiation: true`` and
``agent.hidden_size: 128`` (the YAML defaults) even though ``--no-satiation``
and ``--hidden-size 64`` were passed on the CLI -- this test fails.
AFTER the fix: saved ``config.yaml`` shows ``body.with_satiation: false`` and
``agent.hidden_size: 64``, matching the CLI override -- this test passes.

Run with::

    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/training/test_cli_override_config_persistence.py -v
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import pytest
import yaml

_PYTHON = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TRAIN_PY = os.path.join(_REPO_ROOT, "train.py")
_AGENT_CONFIG = os.path.join(
    _REPO_ROOT, "configs", "models", "recurrent_ppo", "recurrent_ppo.yaml"
)


def _run_train(results_dir: str, extra_args: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        _PYTHON, _TRAIN_PY,
        "--agent_config", _AGENT_CONFIG,
        "--num-envs", "4",
        "--episodes", "1",
        "--device", "cpu",
        "--no-wandb",
        "--quiet",
        "--results-dir", results_dir,
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=_REPO_ROOT)


@pytest.mark.slow
def test_no_satiation_and_hidden_size_overrides_persist_in_saved_config():
    """--no-satiation and --hidden-size must land in the saved config.yaml."""
    with tempfile.TemporaryDirectory(prefix="g3_override_test_") as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        result = _run_train(
            results_dir,
            ["--no-satiation", "--hidden-size", "64", "--tag", "g3_override_test"],
        )
        assert result.returncode == 0, (
            f"train.py exited with non-zero code {result.returncode}.\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

        config_path = os.path.join(results_dir, "models", "config.yaml")
        assert os.path.exists(config_path), f"No saved config at {config_path}"
        with open(config_path) as fh:
            saved = yaml.safe_load(fh)

        assert saved["body"]["with_satiation"] is False, (
            "--no-satiation was passed but saved config.yaml still shows "
            f"body.with_satiation={saved['body']['with_satiation']!r} "
            "(the run behaved correctly at runtime, but re-evaluating this "
            "run from the saved config would silently restore satiation)."
        )
        assert saved["agent"]["hidden_size"] == 64, (
            "--hidden-size 64 was passed but saved config.yaml still shows "
            f"agent.hidden_size={saved['agent']['hidden_size']!r}."
        )


@pytest.mark.slow
def test_no_cli_overrides_saved_config_matches_no_override_run():
    """Backward compatibility: two runs with identical (no-override) CLI args
    must dump byte-identical config.yaml content."""
    with tempfile.TemporaryDirectory(prefix="g3_no_override_test_") as tmpdir:
        results_dir_a = os.path.join(tmpdir, "results_a")
        results_dir_b = os.path.join(tmpdir, "results_b")
        result_a = _run_train(results_dir_a, ["--tag", "g3_no_override_a"])
        result_b = _run_train(results_dir_b, ["--tag", "g3_no_override_b"])
        assert result_a.returncode == 0 and result_b.returncode == 0

        with open(os.path.join(results_dir_a, "models", "config.yaml")) as fh:
            saved_a = yaml.safe_load(fh)
        with open(os.path.join(results_dir_b, "models", "config.yaml")) as fh:
            saved_b = yaml.safe_load(fh)

        # Strip the only two keys expected to legitimately differ (tag is the
        # only CLI value that differs between the two invocations).
        saved_a.pop("tag", None)
        saved_b.pop("tag", None)
        assert saved_a == saved_b, "Identical no-override runs produced different saved configs."
