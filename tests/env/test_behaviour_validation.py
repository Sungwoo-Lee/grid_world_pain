"""CP1 behaviour-validation tests.

Tests the _load_animals() guard that raises ValueError on unknown behaviour
strings (case-sensitive: 'Hunt' must fail, 'hunt' must pass).

Also covers:
  - NC-1: legacy neutral_animals auto-fill of attack_delay=0, damage=[0,0].
  - predator_enabled guard: load_env_params raises ValueError if present.
  - Legacy aliases predator_tags / neutral_tags on EnvParams (B3 / M1 fix).
"""
import os
import sys
import yaml

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import jax.numpy as jnp
from src.utils.config import Config
from src.environment.config_loader import load_env_params


# ── Minimal YAML template ─────────────────────────────────────────────────────
_MINIMAL_YAML = """
environment:
  height: 5
  width: 5
  start_pos: [1, 1]
  random_start_pos: false
  max_steps: 10
  rest_action_enabled: true
  eat_action_enabled: false
  resources: []
  predators: []
  neutral_animals: []
  obstacles: []
  location_areas:
    - type: grass
      area: [[1, 1], [5, 5]]
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
  injury_observable: true
  nutrition_observable: true
  sensor_radius: 5.0
  decay_power: 1.0
  collision_sensor_range: 1
  visual_sensor_enabled: false
  visual_sensor_range: 1
  proprioception_enabled: false
  olfactory_enabled: true
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
"""

_PRED_ENTRY = """
    - tag: wolf
      count: 1
      properties: [0.0, 0.9, 0.0, 0.0, 0.9]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      attack_delay: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      detection_range: 5
      max_stamina: 30.0
      stamina_recovery_rate: 1.0
      hunt_stamina_threshold: 0.5
      lose_interest_multiplier: 2.0
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
"""

_NEUTRAL_ENTRY = """
    - tag: rabbit
      count: 1
      properties: [0.5, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 3
      nociception_intensity: 0.0
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
"""


def _make_config(extra_yaml=""):
    raw = yaml.safe_load(_MINIMAL_YAML)
    if extra_yaml:
        extra = yaml.safe_load(extra_yaml)
        # Deep-merge environment block
        for k, v in extra.get("environment", {}).items():
            raw["environment"][k] = v
    return Config(raw)


def _load(extra_yaml=""):
    return load_env_params(_make_config(extra_yaml))


# ── Behaviour string validation ───────────────────────────────────────────────

def test_valid_hunt_behaviour():
    """'hunt' (lowercase) is accepted for predator entries."""
    params = _load(f"environment:\n  predators:\n{_PRED_ENTRY}")
    assert len(params.hunt_idx) == 1


def test_valid_wander_behaviour():
    """'wander' (lowercase) is accepted for neutral entries."""
    params = _load(f"environment:\n  neutral_animals:\n{_NEUTRAL_ENTRY}")
    assert len(params.wander_idx) == 1


def test_unknown_behaviour_raises():
    """Unknown behaviour string raises ValueError (entities: schema)."""
    entities_yaml = """
environment:
  entities:
    - tag: wolf
      class: predator
      behaviour: Hunt
      count: 1
      properties: [0.0, 0.9, 0.0, 0.0, 0.9]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      attack_delay: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      detection_range: 5
      max_stamina: 30.0
      stamina_recovery_rate: 1.0
      hunt_stamina_threshold: 0.5
      lose_interest_multiplier: 2.0
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
"""
    with pytest.raises(ValueError, match="Unknown behaviour"):
        _load(entities_yaml)


# ── NC-1: neutral auto-fill ───────────────────────────────────────────────────

def test_nc1_neutral_damage_autofill():
    """Legacy neutral_animals: auto-fills attack_delay=0 and damage=[0,0] (NC-1 fix)."""
    params = _load(f"environment:\n  neutral_animals:\n{_NEUTRAL_ENTRY}")
    import numpy as np
    # Should have 1 neutral animal
    assert params.animal_property.shape[0] == 1
    n_idx = jnp.array(list(params.neutral_indices), dtype=jnp.int32)
    assert int(params.animal_attack_delay[n_idx[0]]) == 0
    np.testing.assert_array_equal(params.animal_damage[n_idx[0]], [0.0, 0.0])


# ── predator_enabled guard ────────────────────────────────────────────────────

def test_predator_enabled_raises():
    """load_env_params raises ValueError if predator_enabled is still present in config."""
    raw = yaml.safe_load(_MINIMAL_YAML)
    raw["environment"]["predator_enabled"] = True
    config = Config(raw)
    with pytest.raises(ValueError, match="predator_enabled"):
        load_env_params(config)


# ── Legacy @property aliases (B3 / M1 fix) ───────────────────────────────────

def test_predator_tags_property():
    """params.predator_tags is a @property alias that filters by class == 'predator'."""
    params = _load(f"environment:\n  predators:\n{_PRED_ENTRY}\n  neutral_animals:\n{_NEUTRAL_ENTRY}")
    assert "wolf" in params.predator_tags
    assert "rabbit" not in params.predator_tags


def test_neutral_tags_property():
    """params.neutral_tags is a @property alias that filters by class == 'neutral'."""
    params = _load(f"environment:\n  predators:\n{_PRED_ENTRY}\n  neutral_animals:\n{_NEUTRAL_ENTRY}")
    assert "rabbit" in params.neutral_tags
    assert "wolf" not in params.neutral_tags


def test_predator_enabled_not_a_field():
    """predator_enabled must NOT be a struct field on EnvParams (M1 fix)."""
    params = _load()
    assert not hasattr(params, 'predator_enabled'), (
        "predator_enabled is still a struct field — M1 fix not applied"
    )


def test_predator_tags_not_a_field():
    """predator_tags must be a @property, NOT a struct.field (M1 fix prevents double-def)."""
    from src.environment.state import EnvParams
    # @property objects are defined on the class, not on struct fields.
    # If it's a struct field, the pytree would have it — check by inspecting _fields
    fields = getattr(EnvParams, '__dataclass_fields__', {})
    assert 'predator_tags' not in fields, (
        "predator_tags is a struct.field — M1 fix not applied (should be @property only)"
    )
    assert 'neutral_tags' not in fields, (
        "neutral_tags is a struct.field — M1 fix not applied (should be @property only)"
    )


# ── M3: zero-N dist fallback ─────────────────────────────────────────────────

def test_zero_animals_dist_fallback():
    """With no animals, dist_to_pred and dist_to_neutral fall back to 99.0 (M3 guard)."""
    import jax
    from src.environment.core import jax_reset, jax_step
    params = _load()  # default minimal: no predators, no neutrals
    assert params.animal_property.shape[0] == 0, "Expected 0 animals in minimal config"
    key = jax.random.PRNGKey(42)
    state = jax_reset(params, key)
    _, _, _, info = jax_step(state, 0, params)
    assert float(info["dist_to_pred"]) == 99.0, "dist_to_pred should fall back to 99.0"
    assert float(info["dist_to_neutral"]) == 99.0, "dist_to_neutral should fall back to 99.0"
    assert bool(info["hit_predator"]) is False
    assert bool(info["hit_neutral"]) is False
