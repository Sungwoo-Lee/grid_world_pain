"""Tests for configurable initial internal-state randomization ranges.

Plan: docs/develop/active/refactors/CONFIGURABLE_INITIAL_STATE_RANGES.md

Verifies:
  Test 1 — configured nutrition range draws in [low, high] (hungry band reachable).
  Test 2 — configured injury range draws in [low, high] (badly-injured band reachable).
  Test 3 — flags-off path: absent range keys do NOT error; reset byte-identical.
  Test 4 — missing range key raises ValueError (conditional-mandatory enforcement).
  Test 5 — low > high raises ValueError at load time.
  Test 6 — legacy parity range reproduces the old max/2 hard-coded behaviour.
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset

# ---------------------------------------------------------------------------
# Shared minimal YAML base (no entities, no noise, no random start by default)
# ---------------------------------------------------------------------------

_BASE_YAML = """
environment:
  height: 10
  width: 10
  start_pos: [5, 5]
  random_start_pos: false
  max_steps: 500
  rest_action_enabled: true
  eat_action_enabled: false
  resources: []
  obstacles: []
  entities: []
  location_areas:
    - type: grass
      area: [[1, 1], [10, 10]]
  placement:
    mode: per_entity
body:
  with_satiation: true
  with_nutrition: true
  with_injury: true
  random_start_satiation: false
  random_start_nutrition: false
  random_start_injury: false
  max_satiation: 100.0
  max_nutrition: 100.0
  max_injury: 100.0
  food_nutrition_gain: 20.0
  satiation_setpoint: 70.0
  start_satiation: 70.0
  start_nutrition: 100.0
  metabolic_cost: 1.0
  nutrition_to_satiation_scaling_factor: 0.5
  recovery_base_rate: 0.5
  recovery_accel_rate: 0.5
  injury_smoothing_duration: 3
  death_penalty: 10.0
  overeating_death: false
  use_homeostatic_reward: true
  eating_nutrition_cost: 0.0
  eating_reward_penalty: 0.0
sensory:
  injury_observable: false
  nutrition_observable: false
  sensor_radius: 5.0
  decay_power: 1.0
  olfactory_grid_range: 0
  visual_blur_enabled: false
  visual_blur_radial_scale: 0.5
  visual_blur_anisotropy: 3.0
  visual_blur_sigma_floor: 0.5
  collision_sensor_range: 1
  visual_sensor_enabled: false
  visual_sensor_range: 1
  proprioception_enabled: false
  olfactory_enabled: false
  nociception_enabled: false
  location_sensor: false
  vector_size: 5
  nociception_size: 1
  interoceptive_nociception_enabled: false
  interoceptive_convolution_enabled: false
  interoceptive_kernel_length: 5
  interoceptive_kernel_tau: 3.0
visualization:
  local_view_size: 3
perceptual_noise:
  enabled: false
  modalities: {}
"""


def _make_config(overrides: dict) -> Config:
    """Load _BASE_YAML and deep-merge overrides into it, then wrap in Config."""
    base = yaml.safe_load(_BASE_YAML)
    for section, values in overrides.items():
        if section not in base:
            base[section] = {}
        if isinstance(values, dict):
            base[section].update(values)
        else:
            base[section] = values
    return Config(base)


def _make_params_with_overrides(overrides: dict):
    """Helper: build EnvParams from base YAML + body overrides."""
    return load_env_params(_make_config(overrides))


# ---------------------------------------------------------------------------
# Test 1 — Configured nutrition range draws in bounds (hungry band reachable)
# ---------------------------------------------------------------------------

def test_configured_nutrition_range_draws_in_bounds():
    """Range [0, 30] produces draws in [0, 30] and below 50 (previously unreachable)."""
    params = _make_params_with_overrides({
        'body': {
            'random_start_nutrition': True,
            'start_nutrition_low': 0.0,
            'start_nutrition_high': 30.0,
        }
    })

    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    draws = np.array([
        float(jax_reset(params, k).nutrition)
        for k in keys
    ])

    assert np.all(draws >= 0.0), f"Draws below 0: {draws[draws < 0.0]}"
    assert np.all(draws <= 30.0), f"Draws above 30: {draws[draws > 30.0]}"
    # Proves the hungry band (below max_nutrition/2 = 50) is now reachable.
    assert np.any(draws < 50.0), (
        "Expected at least some draws below 50 (max_nutrition/2), "
        "proving the previously-unreachable hungry band is now accessible"
    )


# ---------------------------------------------------------------------------
# Test 2 — Configured injury range draws in bounds (badly-injured band reachable)
# ---------------------------------------------------------------------------

def test_configured_injury_range_draws_in_bounds():
    """Range [60, 100] produces draws in [60, 100] and above 50 (previously unreachable)."""
    params = _make_params_with_overrides({
        'body': {
            'random_start_injury': True,
            'start_injury_low': 60.0,
            'start_injury_high': 100.0,
        }
    })

    keys = jax.random.split(jax.random.PRNGKey(1), 200)
    draws = np.array([
        float(jax_reset(params, k).injury_level)
        for k in keys
    ])

    assert np.all(draws >= 60.0), f"Draws below 60: {draws[draws < 60.0]}"
    assert np.all(draws <= 100.0), f"Draws above 100: {draws[draws > 100.0]}"
    # Proves the badly-injured band (above max_injury/2 = 50) is now reachable.
    assert np.any(draws > 50.0), (
        "Expected at least some draws above 50 (max_injury/2), "
        "proving the previously-unreachable badly-injured band is now accessible"
    )


# ---------------------------------------------------------------------------
# Test 3 — Flags-off path: absent range keys do NOT error; reset byte-identical
# ---------------------------------------------------------------------------

def test_flags_off_path_unchanged():
    """With both flags false and range keys ABSENT, load succeeds and reset is unchanged."""
    # Explicitly do NOT include start_nutrition_low/high or start_injury_low/high.
    # This is the Fork B2 contract: absent keys must NOT raise when flags are false.
    cfg = _make_config({})  # pure base, both flags already false, no range keys
    params = load_env_params(cfg)

    # Reset should give fixed start_nutrition=100, injury=0.0 (flags off => deterministic).
    state = jax_reset(params, jax.random.PRNGKey(42))
    assert float(state.nutrition) == pytest.approx(100.0, abs=1e-5), (
        f"Expected nutrition=100.0 (start_nutrition), got {float(state.nutrition)}"
    )
    assert float(state.injury_level) == pytest.approx(0.0, abs=1e-5), (
        f"Expected injury_level=0.0, got {float(state.injury_level)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Missing range key raises ValueError (conditional-mandatory)
# ---------------------------------------------------------------------------

def test_missing_range_key_raises():
    """random_start_nutrition=true with start_nutrition_high ABSENT raises ValueError."""
    with pytest.raises(ValueError):
        _make_params_with_overrides({
            'body': {
                'random_start_nutrition': True,
                'start_nutrition_low': 0.0,
                # start_nutrition_high is ABSENT — must raise
            }
        })


# ---------------------------------------------------------------------------
# Test 5 — low > high raises ValueError at load time
# ---------------------------------------------------------------------------

def test_low_greater_than_high_raises():
    """start_nutrition_low=80 > start_nutrition_high=20 raises ValueError at load time."""
    with pytest.raises(ValueError):
        _make_params_with_overrides({
            'body': {
                'random_start_nutrition': True,
                'start_nutrition_low': 80.0,
                'start_nutrition_high': 20.0,
            }
        })


# ---------------------------------------------------------------------------
# Test 6 — Legacy parity: [max/2, max] nutrition + [0, max/2] injury reproduce old behaviour
# ---------------------------------------------------------------------------

def test_legacy_parity_range_reproduces_old_behaviour():
    """Range [max/2, max] for nutrition and [0, max/2] for injury matches old hard-coded draws.

    The old code used:
        min_start_nutr  = max_nutrition / 2.0
        nutrition = jax.random.uniform(body_key2, (), minval=min_start_nutr, maxval=max_nutrition)
        max_start_injury = max_injury / 2.0
        injury = jax.random.uniform(body_key3, (), minval=0.0, maxval=max_start_injury)

    With the new params-carried bounds set to those exact values, we feed the same
    body_key2 / body_key3 streams to jax.random.uniform with identical minval/maxval,
    so the output must be bit-identical.
    """
    max_nutr = 100.0
    max_inj  = 100.0

    params_new = _make_params_with_overrides({
        'body': {
            'random_start_nutrition': True,
            'start_nutrition_low':  max_nutr / 2.0,
            'start_nutrition_high': max_nutr,
            'random_start_injury': True,
            'start_injury_low':  0.0,
            'start_injury_high': max_inj / 2.0,
        }
    })

    # Simulate old behaviour: same PRNG key derivation as jax_reset.
    # jax_reset splits the top-level key as:
    #   key, agent_key, placement_key, body_key, property_key = jax.random.split(key, 5)
    # then: body_key1, body_key2, body_key3 = jax.random.split(body_key, 3)
    # nutrition uses body_key2, injury uses body_key3.
    seed = jax.random.PRNGKey(7)
    _, _, _, body_key, _ = jax.random.split(seed, 5)
    _, body_key2, body_key3 = jax.random.split(body_key, 3)

    expected_nutr = float(jax.random.uniform(body_key2, (), minval=max_nutr / 2.0, maxval=max_nutr))
    expected_inj  = float(jax.random.uniform(body_key3, (), minval=0.0, maxval=max_inj / 2.0))

    state = jax_reset(params_new, seed)

    assert float(state.nutrition) == pytest.approx(expected_nutr, abs=1e-6), (
        f"Nutrition parity failed: expected {expected_nutr}, got {float(state.nutrition)}"
    )
    assert float(state.injury_level) == pytest.approx(expected_inj, abs=1e-6), (
        f"Injury parity failed: expected {expected_inj}, got {float(state.injury_level)}"
    )
