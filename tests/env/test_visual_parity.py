"""CP4 — visual sensor parity test.

Runs one episode (1000 steps) on the parity-reference config from seed 0
and asserts byte-equality with a pinned reference fixture stored at:
  tests/env/fixtures/visual_parity_ref.npz

The fixture is auto-generated on first run (when absent) and committed.
On subsequent runs the test strictly verifies byte-equality.

Usage:
  # Generate fixture (first time):
  python -m pytest tests/env/test_visual_parity.py --gen-fixtures
  # Normal run (strict byte-equality):
  python -m pytest tests/env/test_visual_parity.py
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
    _ROOT, "tests", "env", "fixtures", "visual_parity_ref.npz"
)

ACTIONS = ([0, 1, 2, 3, 4] * 200)  # 1000 steps


def _load_params():
    """Load the parity-reference config."""
    with open(_PARITY_CFG) as f:
        d = yaml.safe_load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(d))


def _get_visual_slice(obs: np.ndarray, params) -> np.ndarray:
    """Extract the Visual sensor slice from a flattened observation vector."""
    breakdown = get_observation_breakdown(params)
    offset = 0
    for sensor_name, dim in breakdown.items():
        if sensor_name == "Visual":
            return obs[offset: offset + dim]
        offset += dim
    raise KeyError("'Visual' sensor not found in observation breakdown.")


def _run_episode(params, seed=0):
    """Reset + 1000 steps. Returns dict of arrays for the fixture."""
    key = jax.random.PRNGKey(seed)
    state = jax_reset(params, key)
    visual_slices = []
    for action in ACTIONS:
        state, _, _, _ = jax_step(state, action, params)
        obs = np.array(get_observation(state, params, apply_noise=False))
        visual_slices.append(_get_visual_slice(obs, params))
    return np.stack(visual_slices, axis=0)  # [1000, visual_dim]


def _generate_fixture(params):
    """Generate and save the fixture to disk."""
    os.makedirs(os.path.dirname(_FIXTURE_PATH), exist_ok=True)
    visual_all = _run_episode(params)
    np.savez_compressed(_FIXTURE_PATH, visual_all=visual_all)
    print(f"Generated visual parity fixture: {_FIXTURE_PATH}")
    return visual_all


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
def visual_fixture(params, request):
    """Load or generate the visual parity fixture."""
    gen = request.config.getoption("--gen-fixtures", default=False)
    if gen or not os.path.exists(_FIXTURE_PATH):
        return _generate_fixture(params)
    data = np.load(_FIXTURE_PATH)
    return data["visual_all"]


def test_visual_parity_byte_equal(visual_fixture, params):
    """Visual sensor slice must be byte-identical to the pinned reference fixture."""
    ref = visual_fixture  # loaded or generated above

    # Re-run the episode
    visual_new = _run_episode(params)

    assert visual_new.shape == ref.shape, (
        f"Shape mismatch: got {visual_new.shape}, expected {ref.shape}"
    )
    np.testing.assert_array_equal(
        visual_new, ref,
        err_msg=(
            "Visual sensor slice differs from pinned fixture — CP4 regression. "
            f"First differing step: {np.where((visual_new != ref).any(axis=1))[0]}"
        ),
    )


def test_visual_channel_layout(params):
    """Predators use channel 5, neutral animals use channel 7 (CP1/CP4 parity claim)."""
    key = jax.random.PRNGKey(99)
    state = jax_reset(params, key)
    # Verify params.animal_visual_channel encodes [predator=5, neutral=7, neutral=7]
    # for the 1-pred + 2-neutral reference config.
    channels = list(params.animal_visual_channel)
    pred_channels = [channels[i] for i in params.predator_indices]
    neutral_channels = [channels[i] for i in params.neutral_indices]
    assert all(c == 5 for c in pred_channels), (
        f"Predator visual channels should all be 5, got {pred_channels}"
    )
    assert all(c == 7 for c in neutral_channels), (
        f"Neutral visual channels should all be 7, got {neutral_channels}"
    )
