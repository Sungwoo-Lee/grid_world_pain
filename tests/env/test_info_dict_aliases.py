"""CP1 info-dict legacy-alias parity test (C5 from code-reviewer).

Loads the parity-reference config (01-interoNocicept_sameProp.yaml), runs
100 steps from seed 0, and asserts that every legacy info-dict key:
  (a) is present in the returned info dict, AND
  (b) has the same value as stored in the pre-refactor fixture.

Legacy keys checked:
  hit_predator, hit_neutral, damage_predator, damage_obstacle,
  damage_hiding_predator, dist_per_predator, dist_per_neutral,
  dist_to_pred, dist_to_neutral, agent_in_bush.

All keys are expected to be present even when there are zero predators /
neutrals (M3 guard: zero-N → empty array or False / 99.0 sentinel).
"""
import os
import sys
import yaml

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import jax

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step

# ── Reference config ─────────────────────────────────────────────────────────
_PARITY_REF_CFG = os.path.join(
    _ROOT, "configs", "experiment", "hypervigilance",
    "01-interoNocicept_sameProp.yaml"
)
_FIXTURE_DIR = os.path.join(_ROOT, "tests", "env", "fixtures", "parity")

import re as _re
import glob as _glob


def _config_slug(config_path: str) -> str:
    rel = os.path.relpath(config_path, _ROOT)
    slug = _re.sub(r'[/\\]', '__', rel)
    slug = _re.sub(r'\.yaml$', '', slug)
    slug = _re.sub(r'[^A-Za-z0-9_.-]', '_', slug)
    return slug


ACTIONS = [0, 1, 2, 3, 4] * 20  # 100 steps

_LEGACY_SCALAR_KEYS = [
    "hit_predator",
    "hit_neutral",
    "damage_predator",
    "damage_obstacle",
    "damage_hiding_predator",
    "dist_to_pred",
    "dist_to_neutral",
    "agent_in_bush",
]
_LEGACY_ARRAY_KEYS = [
    "dist_per_predator",
    "dist_per_neutral",
]


@pytest.fixture(scope="module")
def ref_data():
    """Load the parity reference config + fixture, run 100 steps."""
    if not os.path.exists(_PARITY_REF_CFG):
        pytest.skip(f"Reference config not found: {_PARITY_REF_CFG}")

    slug = _config_slug(_PARITY_REF_CFG)
    fixture_path = os.path.join(_FIXTURE_DIR, slug + ".npz")
    if not os.path.exists(fixture_path):
        pytest.skip(f"No parity fixture for {slug}")

    with open(_PARITY_REF_CFG) as f:
        cfg_dict = yaml.safe_load(f)
    config = Config(cfg_dict)
    params = load_env_params(config)

    fixture = np.load(fixture_path)

    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)
    infos = []
    for action in ACTIONS:
        state, _, _, info = jax_step(state, action, params)
        infos.append({k: np.array(v) for k, v in info.items()})

    return infos, fixture


def test_all_legacy_scalar_keys_present(ref_data):
    """Every legacy scalar info-dict key is present in every step's info."""
    infos, fixture = ref_data
    for step_i, info in enumerate(infos):
        for key_name in _LEGACY_SCALAR_KEYS:
            assert key_name in info, (
                f"Legacy key '{key_name}' missing from info at step {step_i}"
            )


def test_all_legacy_array_keys_present(ref_data):
    """Every legacy array info-dict key is present in every step's info."""
    infos, fixture = ref_data
    for step_i, info in enumerate(infos):
        for key_name in _LEGACY_ARRAY_KEYS:
            assert key_name in info, (
                f"Legacy key '{key_name}' missing from info at step {step_i}"
            )


def test_legacy_scalar_values_match_fixture(ref_data):
    """Legacy scalar info keys match pre-refactor fixture values."""
    infos, fixture = ref_data
    for step_i, info in enumerate(infos):
        for key_name in _LEGACY_SCALAR_KEYS:
            old_val = fixture.get(f"info{step_i:03d}_{key_name}")
            if old_val is None:
                continue  # key not captured in fixture; skip
            new_val = info[key_name]
            np.testing.assert_allclose(
                new_val, old_val, rtol=1e-5, atol=1e-5,
                err_msg=(
                    f"info['{key_name}'] mismatch at step {step_i}: "
                    f"new={new_val}, old={old_val}"
                )
            )


def test_legacy_array_values_match_fixture(ref_data):
    """Legacy array info keys match pre-refactor fixture values (dist_per_*)."""
    infos, fixture = ref_data
    for step_i, info in enumerate(infos):
        for key_name in _LEGACY_ARRAY_KEYS:
            old_val = fixture.get(f"info{step_i:03d}_{key_name}")
            if old_val is None:
                continue
            new_val = info[key_name]
            np.testing.assert_allclose(
                new_val, old_val, rtol=1e-5, atol=1e-5,
                err_msg=(
                    f"info['{key_name}'] mismatch at step {step_i}: "
                    f"new={new_val}, old={old_val}"
                )
            )


def test_dist_per_animal_present(ref_data):
    """New unified info['dist_per_animal'] is also present (additive — not a breaking change)."""
    infos, _ = ref_data
    for step_i, info in enumerate(infos):
        assert "dist_per_animal" in info, (
            f"New key 'dist_per_animal' missing from info at step {step_i}"
        )
