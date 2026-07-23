"""Regression test — Bug 5 (Track B, 2026-07-23 eval/logging fix plan, latent):
`eval_obs_noise` ("training" | "zero" | "custom") passed config validation and
was echoed into `metadata.json` as if honored, but the rollout never actually
applied it -- the policy always saw the env's configured (training) noise, so
any non-"training" value was a silent lie in the saved metadata.

Decision (recorded in the plan): fail loud, do not implement enforcement.
Since no config sets anything but "training" today, the fix hard-errors at
eval startup on any `eval_obs_noise != "training"` rather than proceeding
silently.

Fix: scripts/eval/eval_rollout.py::_assert_eval_obs_noise_supported(bm_cfg),
called from main() before any episode runs.
"""
import os
import sys
from types import SimpleNamespace

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "eval"))

import eval_rollout as er  # noqa: E402


def test_non_training_noise_mode_raises():
    bm_cfg_zero = SimpleNamespace(eval_obs_noise="zero")
    with pytest.raises(NotImplementedError):
        er._assert_eval_obs_noise_supported(bm_cfg_zero)

    bm_cfg_custom = SimpleNamespace(eval_obs_noise="custom")
    with pytest.raises(NotImplementedError):
        er._assert_eval_obs_noise_supported(bm_cfg_custom)


def test_training_noise_mode_does_not_raise():
    bm_cfg = SimpleNamespace(eval_obs_noise="training")
    er._assert_eval_obs_noise_supported(bm_cfg)  # should not raise
