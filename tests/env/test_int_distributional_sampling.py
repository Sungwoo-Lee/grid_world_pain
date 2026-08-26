"""Per-episode integer distributional sampling tests for move_interval and attack_delay.

Verifies that move_interval and attack_delay accept scalar-or-[lo, hi] YAML syntax
and are sampled per-episode as integers from the inclusive range [lo, hi].

Tests:
  1. move_interval: [1, 3] → over ~300 resets sampled value is an integer in {1, 2, 3}
     and varies across resets.
  2. attack_delay: [2, 4] → sampled integer in {2, 3, 4} and varies;
     scalar attack_delay: 3 → always 3 (degenerate).
  3. Backward-compat: an existing scalar config (basic/04 style) loads and a
     jax_reset + jax_step runs with no behavioural change vs degenerate range.
  4. Both new sampled fields live on EnvState, NOT on EnvParams.
  5. Degenerate guard: scalar move_interval / attack_delay produce exactly the
     scalar value on every reset.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step

# ---------------------------------------------------------------------------
# Shared YAML base
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
  start_nutrition: 70.0
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

# 1 predator entity with move_interval range [1, 3] and attack_delay range [2, 4]
_ENTITY_RANGES_YAML = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: p1
      count: 1
      properties: [0.0, 0.8, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: [1, 3]
      nociception_intensity: 0.9
      damage: [10.0, 30.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: [2, 4]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
"""

# 1 predator with scalar move_interval=1 and scalar attack_delay=3 (degenerate)
_ENTITY_SCALAR_YAML = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: p1
      count: 1
      properties: [0.0, 0.8, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [10.0, 30.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 3
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
"""

# Legacy predator/neutral format (backward-compat check)
_LEGACY_SCALAR_YAML = """
environment:
  predators:
    - tag: wolf
      count: 1
      properties: [0.0, 0.9, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      attack_delay: 2
      damage: [10.0, 30.0]
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
  neutral_animals: []
"""


def _load(extra_yaml: str):
    """Merge extra_yaml environment section into base and return EnvParams."""
    base = yaml.safe_load(_BASE_YAML)
    extra = yaml.safe_load(extra_yaml)
    for k, v in extra.get("environment", {}).items():
        base["environment"][k] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def params_range():
    return _load(_ENTITY_RANGES_YAML)


@pytest.fixture(scope="module")
def params_scalar():
    return _load(_ENTITY_SCALAR_YAML)


@pytest.fixture(scope="module")
def params_legacy():
    return _load(_LEGACY_SCALAR_YAML)


# ---------------------------------------------------------------------------
# Test 1 — move_interval: [1, 3] samples integer in {1, 2, 3} and varies
# ---------------------------------------------------------------------------

def test_move_interval_range_sampling(params_range):
    """move_interval: [1, 3] → over 300 resets sampled value is in {1, 2, 3} and varies."""
    keys = jax.random.split(jax.random.PRNGKey(42), 300)
    samples = []
    for k in keys:
        state = jax_reset(params_range, k)
        val = int(state.animal_move_int_sampled[0])
        samples.append(val)

    arr = np.array(samples)
    assert np.all(arr >= 1), f"Values below lo=1 found: {arr[arr < 1]}"
    assert np.all(arr <= 3), f"Values above hi=3 found: {arr[arr > 3]}"
    # All three values should appear with 300 draws (probability of missing any is ~(2/3)^300 ≈ 0)
    assert set(samples) == {1, 2, 3}, (
        f"Expected all three values {{1, 2, 3}}, got {set(samples)}. "
        "Integer sampling may not be uniform."
    )
    assert len(set(samples)) > 1, "move_int_sampled does not vary across resets."


# ---------------------------------------------------------------------------
# Test 2 — attack_delay: [2, 4] samples integer in {2, 3, 4} and varies;
#           scalar attack_delay: 3 always gives 3
# ---------------------------------------------------------------------------

def test_attack_delay_range_sampling(params_range):
    """attack_delay: [2, 4] → sampled integer in {2, 3, 4} and varies."""
    keys = jax.random.split(jax.random.PRNGKey(7), 300)
    samples = []
    for k in keys:
        state = jax_reset(params_range, k)
        val = int(state.animal_attack_delay_sampled[0])
        samples.append(val)

    arr = np.array(samples)
    assert np.all(arr >= 2), f"Values below lo=2 found: {arr[arr < 2]}"
    assert np.all(arr <= 4), f"Values above hi=4 found: {arr[arr > 4]}"
    assert set(samples) == {2, 3, 4}, (
        f"Expected all three values {{2, 3, 4}}, got {set(samples)}."
    )
    assert len(set(samples)) > 1, "attack_delay_sampled does not vary across resets."


def test_attack_delay_scalar_always_constant(params_scalar):
    """scalar attack_delay: 3 → always samples 3 (degenerate range [3, 3])."""
    keys = jax.random.split(jax.random.PRNGKey(99), 100)
    for k in keys:
        state = jax_reset(params_scalar, k)
        val = int(state.animal_attack_delay_sampled[0])
        assert val == 3, f"Expected 3 from scalar attack_delay=3, got {val}"


# ---------------------------------------------------------------------------
# Test 3 — Backward-compat: scalar configs load and step runs correctly
# ---------------------------------------------------------------------------

def test_backward_compat_scalar_config_loads_and_steps(params_legacy):
    """Legacy scalar-config loads, jax_reset + jax_step runs without error."""
    # Verify sampled values match scalars
    state = jax_reset(params_legacy, jax.random.PRNGKey(0))
    mi = int(state.animal_move_int_sampled[0])
    ad = int(state.animal_attack_delay_sampled[0])
    assert mi == 1, f"Expected move_int_sampled=1 for scalar move_interval=1, got {mi}"
    assert ad == 2, f"Expected attack_delay_sampled=2 for scalar attack_delay=2, got {ad}"

    # 10 steps should not raise
    for action in range(5):
        state, _, _, _ = jax_step(state, action, params_legacy)

    # Scalar degenerate range: low/high arrays should be equal
    np.testing.assert_array_equal(
        np.array(params_legacy.animal_move_int_low),
        np.array(params_legacy.animal_move_int_high),
        err_msg="Scalar move_interval should produce degenerate range (low == high)",
    )
    np.testing.assert_array_equal(
        np.array(params_legacy.animal_attack_delay_low),
        np.array(params_legacy.animal_attack_delay_high),
        err_msg="Scalar attack_delay should produce degenerate range (low == high)",
    )


def test_backward_compat_scalar_byte_identical(params_scalar):
    """Scalar config: sampled move_int and attack_delay equal the scalar on every reset.

    Verifies that randint([s, s+1)) == s always, so degenerate range is byte-identical
    to the pre-feature fixed-array behaviour (same integer value used at runtime).
    """
    mi_scalar = 1
    ad_scalar = 3
    keys = jax.random.split(jax.random.PRNGKey(11), 50)
    for k in keys:
        state = jax_reset(params_scalar, k)
        mi = int(state.animal_move_int_sampled[0])
        ad = int(state.animal_attack_delay_sampled[0])
        assert mi == mi_scalar, (
            f"Degenerate move_interval={mi_scalar} yielded sampled={mi} from key {k}"
        )
        assert ad == ad_scalar, (
            f"Degenerate attack_delay={ad_scalar} yielded sampled={ad} from key {k}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Sampled fields live on EnvState, NOT on EnvParams
# ---------------------------------------------------------------------------

def test_sampled_fields_on_state_not_params(params_range):
    """Per-episode sampled fields must be on EnvState, not on EnvParams."""
    key = jax.random.PRNGKey(0)
    state = jax_reset(params_range, key)
    params = params_range

    for field in ("animal_move_int_sampled", "animal_attack_delay_sampled"):
        assert hasattr(state, field), (
            f"EnvState is missing field '{field}'"
        )
        assert not hasattr(params, field), (
            f"EnvParams unexpectedly has field '{field}' — per-episode samples "
            "belong on EnvState only."
        )

    # Bounds are on EnvParams
    for field in (
        "animal_move_int_low", "animal_move_int_high",
        "animal_attack_delay_low", "animal_attack_delay_high",
    ):
        assert hasattr(params, field), (
            f"EnvParams is missing bound field '{field}'"
        )


# ---------------------------------------------------------------------------
# Test 5 — Degenerate guard: scalar produces exactly the scalar value
# ---------------------------------------------------------------------------

def test_degenerate_guard_move_int(params_scalar):
    """move_interval: 1 (scalar) → params low==high==1, sampled always 1."""
    assert int(params_scalar.animal_move_int_low[0]) == 1
    assert int(params_scalar.animal_move_int_high[0]) == 1
    state = jax_reset(params_scalar, jax.random.PRNGKey(5))
    assert int(state.animal_move_int_sampled[0]) == 1


def test_degenerate_guard_attack_delay(params_scalar):
    """attack_delay: 3 (scalar) → params low==high==3, sampled always 3."""
    assert int(params_scalar.animal_attack_delay_low[0]) == 3
    assert int(params_scalar.animal_attack_delay_high[0]) == 3
    state = jax_reset(params_scalar, jax.random.PRNGKey(5))
    assert int(state.animal_attack_delay_sampled[0]) == 3
