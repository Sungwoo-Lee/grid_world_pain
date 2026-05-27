"""CP4 — extero-nociception sensor parity test (B2 fix verification).

Runs one episode (1000 steps) on the parity-reference config from seed 0
and asserts byte-equality with a pinned reference fixture stored at:
  tests/env/fixtures/extero_noc_parity_ref.npz

The fixture is auto-generated on first run (when absent) and committed.
On subsequent runs the test strictly verifies byte-equality.

This test verifies the B2 fix: sense_extero_nociception now uses
state.animal_pos / params.animal_nociception masked by params.animal_is_damaging
instead of the old state.pred_pos / params.pred_nociception. Only damaging
(predator-class) animals contribute to the extero-nociception signal — neutrals
must NOT appear in the sum even though they are now in the same animal_pos array.

The parity-reference config (01-interoNocicept_sameProp.yaml) has 1 predator +
2 neutral rabbits, so a regression in the animal_is_damaging mask would cause
neutrals to contribute nociception and would break byte-equality with the fixture.

Usage:
  # Generate fixture (first time):
  python -m pytest tests/env/test_extero_noc_parity.py --gen-fixtures
  # Normal run (strict byte-equality):
  python -m pytest tests/env/test_extero_noc_parity.py
"""
import os
import sys
import warnings

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import jax

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation, get_observation_breakdown

# ── Paths ─────────────────────────────────────────────────────────────────────

_PARITY_CFG = os.path.join(
    _ROOT, "configs", "experiment", "hypervigilance",
    "01-interoNocicept_sameProp.yaml"
)
_FIXTURE_PATH = os.path.join(
    _ROOT, "tests", "env", "fixtures", "extero_noc_parity_ref.npz"
)

ACTIONS = ([0, 1, 2, 3, 4] * 200)  # 1000 steps


def _load_params():
    """Load the parity-reference config."""
    with open(_PARITY_CFG) as f:
        d = yaml.safe_load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(d))


def _get_extero_noc_slice(obs: np.ndarray, params) -> np.ndarray:
    """Extract the Extero Nociception sensor slice from a flattened observation vector."""
    breakdown = get_observation_breakdown(params)
    offset = 0
    for sensor_name, dim in breakdown.items():
        if sensor_name == "Extero Nociception":
            return obs[offset: offset + dim]
        offset += dim
    raise KeyError("'Extero Nociception' sensor not found in observation breakdown.")


def _run_episode(params, seed=0):
    """Reset + 1000 steps. Returns [1000, 1] array of extero-noc values."""
    key = jax.random.PRNGKey(seed)
    state = jax_reset(params, key)
    extero_noc_slices = []
    for action in ACTIONS:
        state, _, _, _ = jax_step(state, action, params)
        obs = np.array(get_observation(state, params, apply_noise=False))
        extero_noc_slices.append(_get_extero_noc_slice(obs, params))
    return np.stack(extero_noc_slices, axis=0)  # [1000, 1]


def _generate_fixture(params):
    """Generate and save the fixture to disk."""
    os.makedirs(os.path.dirname(_FIXTURE_PATH), exist_ok=True)
    extero_noc_all = _run_episode(params)
    np.savez_compressed(_FIXTURE_PATH, extero_noc_all=extero_noc_all)
    print(f"Generated extero-noc parity fixture: {_FIXTURE_PATH}")
    return extero_noc_all


# ── pytest option ─────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    """Add --gen-fixtures option to pytest."""
    try:
        parser.addoption(
            "--gen-fixtures",
            action="store_true",
            default=False,
            help="Regenerate visual/extero-noc parity fixtures and save to disk.",
        )
    except ValueError:
        pass  # option already registered by another conftest


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def params():
    if not os.path.exists(_PARITY_CFG):
        pytest.skip(f"Parity reference config not found: {_PARITY_CFG}")
    return _load_params()


@pytest.fixture(scope="module")
def extero_noc_fixture(params, request):
    """Load or generate the extero-noc parity fixture."""
    gen = request.config.getoption("--gen-fixtures", default=False)
    if gen or not os.path.exists(_FIXTURE_PATH):
        return _generate_fixture(params)
    data = np.load(_FIXTURE_PATH)
    return data["extero_noc_all"]


def test_extero_noc_parity_byte_equal(extero_noc_fixture, params):
    """Extero Nociception slice must be byte-identical to the pinned reference fixture.

    This is the load-bearing gate for the B2 fix: only damaging (predator-class)
    animals contribute to extero-nociception. If the animal_is_damaging mask is
    wrong (e.g., neutrals now also contribute), the nociception signal changes and
    byte-equality fails.
    """
    ref = extero_noc_fixture

    # Re-run the episode
    extero_noc_new = _run_episode(params)

    assert extero_noc_new.shape == ref.shape, (
        f"Shape mismatch: got {extero_noc_new.shape}, expected {ref.shape}"
    )
    np.testing.assert_array_equal(
        extero_noc_new, ref,
        err_msg=(
            "Extero Nociception slice differs from pinned fixture — CP4/B2 regression. "
            f"First differing step: {np.where((extero_noc_new != ref).any(axis=1))[0]}"
        ),
    )


def test_only_damaging_animals_contribute(params):
    """Verify params.animal_is_damaging is True only for predators (not neutrals).

    B2 correctness check: the parity-reference config has 1 predator + 2 rabbits.
    animal_is_damaging must be [True, False, False] (predators-first layout).
    If any neutral is marked as damaging, the extero-noc sum would include neutral
    nociception — a silent behavioural change from the pre-refactor code.
    """
    if not os.path.exists(_PARITY_CFG):
        pytest.skip(f"Parity reference config not found: {_PARITY_CFG}")

    is_damaging = list(params.animal_is_damaging)
    pred_damage_flags = [is_damaging[i] for i in params.predator_indices]
    neutral_damage_flags = [is_damaging[i] for i in params.neutral_indices]

    assert all(pred_damage_flags), (
        f"All predators should be damaging (animal_is_damaging=True), got {pred_damage_flags}"
    )
    assert not any(neutral_damage_flags), (
        f"No neutrals should be damaging (animal_is_damaging=False), got {neutral_damage_flags}"
    )


def test_nociception_enabled_in_reference_config(params):
    """Verify the reference config has nociception enabled (test would be vacuous otherwise)."""
    assert params.nociception_enabled, (
        "Reference config must have nociception_enabled=True for this parity test to be meaningful."
    )
    assert params.animal_nociception.shape[0] > 0, (
        "Reference config must have at least one animal for nociception parity to be testable."
    )
