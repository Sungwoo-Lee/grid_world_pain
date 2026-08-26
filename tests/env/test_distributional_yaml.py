"""CP5 — Distributional YAML parsing tests.

Verifies that the config loader's _parse_distributional() helper correctly
handles all four input shapes for the 5 distributional fields:

  detection_range, max_stamina, stamina_recovery_rate,
  hunt_stamina_threshold, lose_interest_multiplier

Four scalar/range cases tested against the loaded EnvParams bounds:

  1. Scalar 5          → bounds (5.0, 5.0)  [degenerate range]
  2. [0, 5]            → bounds (0.0, 5.0)  [true non-degenerate range]
  3. [5]               → raises ValueError  [malformed: one-element list]
  4. "five"            → raises ValueError  [malformed: non-numeric string]
  5. Non-degenerate config (02-entities-distributional.yaml) loads and
     stores correct low/high for all 5 fields.

Plan ref: §"Test Plan §(f) — Distributional schema test" (CP5).
"""
import os
import sys
import warnings

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.utils.config import Config
from src.environment.config_loader import load_env_params


# ── Shared base YAML (minimal env, no resources, no obstacles) ─────────────

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

# Predator-only entity template; DETECTION_RANGE_PLACEHOLDER is substituted per test.
_PRED_ENTITY_TEMPLATE = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: p
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [10.0, 20.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 2
      detection_range: {detect}
      max_stamina: 30
      stamina_recovery_rate: 1.0
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
"""


def _load_with_detect(detect_repr: str):
    """Build a minimal config with the given detection_range YAML value and load it."""
    base = yaml.safe_load(_BASE_YAML)
    extra_raw = _PRED_ENTITY_TEMPLATE.format(detect=detect_repr)
    extra = yaml.safe_load(extra_raw)
    base["environment"]["entities"] = extra["environment"]["entities"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ── Tests: scalar and range cases ─────────────────────────────────────────

class TestParseSingleField:
    """_parse_distributional() for detection_range in isolation."""

    def test_scalar_5_gives_degenerate_range(self):
        """Scalar 5 → bounds (5.0, 5.0): uniform([5, 5]) == 5 every episode."""
        params = _load_with_detect("5")
        lo = float(params.animal_detect_low[0])
        hi = float(params.animal_detect_high[0])
        assert lo == 5.0, f"Expected low=5.0, got {lo}"
        assert hi == 5.0, f"Expected high=5.0, got {hi}"

    def test_range_0_5_gives_correct_bounds(self):
        """[0, 5] → bounds (0.0, 5.0): true non-degenerate range."""
        params = _load_with_detect("[0, 5]")
        lo = float(params.animal_detect_low[0])
        hi = float(params.animal_detect_high[0])
        assert lo == 0.0, f"Expected low=0.0, got {lo}"
        assert hi == 5.0, f"Expected high=5.0, got {hi}"

    def test_single_element_list_raises_valueerror(self):
        """[5] → ValueError: must be scalar or 2-element list."""
        with pytest.raises(ValueError, match="2-element list"):
            _load_with_detect("[5]")

    def test_non_numeric_string_raises_valueerror(self):
        """'five' → ValueError: could not convert string to float."""
        with pytest.raises(ValueError):
            _load_with_detect("five")

    def test_inverted_range_raises_valueerror(self):
        """[5, 3] → ValueError: low must be <= high (inverted range).

        C-CP5-1 regression test. Without the lo <= hi guard, jax.random.uniform
        with minval=5, maxval=3 silently returns 5.0 every episode (deterministic
        and wrong). With the guard, a clear ValueError is raised at load time.
        """
        with pytest.raises(ValueError, match="low <= high"):
            _load_with_detect("[5, 3]")


# ── Tests: all 5 fields via non-degenerate config file ───────────────────

_DISTRIBUTIONAL_CONFIG = os.path.join(
    _ROOT, "configs", "environment", "experiment", "archive", "v2_smoke", "02-entities-distributional.yaml"
)


@pytest.fixture(scope="module")
def dist_params():
    """Load 02-entities-distributional.yaml once for all field tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config.load_yaml(_DISTRIBUTIONAL_CONFIG))


class TestDistributionalConfig:
    """End-to-end check that 02-entities-distributional.yaml stores all 5 field bounds."""

    def test_config_file_exists(self):
        """Smoke: the config file was created as part of CP5."""
        assert os.path.isfile(_DISTRIBUTIONAL_CONFIG), (
            f"Missing CP5 config: {_DISTRIBUTIONAL_CONFIG}"
        )

    def test_detect_bounds(self, dist_params):
        """detection_range: [0, 5] → animal_detect_low[0]==0, animal_detect_high[0]==5."""
        assert float(dist_params.animal_detect_low[0]) == 0.0
        assert float(dist_params.animal_detect_high[0]) == 5.0

    def test_max_stamina_bounds(self, dist_params):
        """max_stamina: [20, 40] → low=20.0, high=40.0."""
        assert float(dist_params.animal_max_stamina_low[0]) == 20.0
        assert float(dist_params.animal_max_stamina_high[0]) == 40.0

    def test_recovery_bounds(self, dist_params):
        """stamina_recovery_rate: [0.5, 1.5] → low=0.5, high=1.5."""
        assert abs(float(dist_params.animal_recovery_low[0]) - 0.5) < 1e-6
        assert abs(float(dist_params.animal_recovery_high[0]) - 1.5) < 1e-6

    def test_hunt_thresh_bounds(self, dist_params):
        """hunt_stamina_threshold: [0.5, 0.9] → low=0.5, high=0.9."""
        assert abs(float(dist_params.animal_hunt_thresh_low[0]) - 0.5) < 1e-6
        assert abs(float(dist_params.animal_hunt_thresh_high[0]) - 0.9) < 1e-6

    def test_lose_interest_degenerate(self, dist_params):
        """lose_interest_multiplier: 1.5 (scalar) → degenerate [1.5, 1.5]."""
        assert abs(float(dist_params.animal_lose_interest_low[0]) - 1.5) < 1e-6
        assert abs(float(dist_params.animal_lose_interest_high[0]) - 1.5) < 1e-6

    def test_wander_animals_have_zero_bounds(self, dist_params):
        """Neutral (wander) animals get auto-filled [0, 0] for all 5 dist fields."""
        # Indices 1 and 2 are the two wander rabbits (predator-first ordering).
        for i in [1, 2]:
            assert float(dist_params.animal_detect_low[i]) == 0.0
            assert float(dist_params.animal_detect_high[i]) == 0.0
            assert float(dist_params.animal_max_stamina_low[i]) == 0.0
            assert float(dist_params.animal_max_stamina_high[i]) == 0.0
