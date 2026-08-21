"""CP5 — Per-episode sampled-parameter WandB logging tests.

Verifies that build_episode_log_dict() produces the five Episode/sampled_*_<tag>
WandB keys for each animal entity, and that the values:

  1. Appear in the per-episode log dict (all 5 keys × all tags present).
  2. Are within [low, high] for each tag and field.
  3. Change across episodes when non-degenerate bounds are used.
  4. Are constant across episodes for degenerate (scalar) bounds.

Plan ref: §"Test Plan §(e) — JIT-recompile test" and §"Per-episode logging"
         description (CP5, accumulators.py build_episode_log_dict).

Config used for non-degenerate tests:
  configs/environment/experiment/archive/v2_smoke/02-entities-distributional.yaml
  — 1 predator (tag='predator0', detection_range=[0,5]) + 2 wander rabbits
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
import jax.numpy as jnp

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset
from src.behavior.accumulators import build_episode_log_dict, sampled_wandb_keys


# ── Config paths ─────────────────────────────────────────────────────────────

_DIST_CONFIG = os.path.join(
    _ROOT, "configs", "environment", "experiment", "archive", "v2_smoke", "02-entities-distributional.yaml"
)

# Minimal legacy config with degenerate ranges (all scalars).
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
  olfactory_sensor_range: 0
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

# Degenerate-range predator (all scalars → [s, s] internally).
_DEGENERATE_ENTITIES_YAML = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: pred
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [10.0, 20.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 2
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1.0
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
"""


def _load_degenerate():
    base = yaml.safe_load(_BASE_YAML)
    extra = yaml.safe_load(_DEGENERATE_ENTITIES_YAML)
    base["environment"]["entities"] = extra["environment"]["entities"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dist_params():
    """Load non-degenerate distributional config."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config.load_yaml(_DIST_CONFIG))


@pytest.fixture(scope="module")
def deg_params():
    """Load degenerate-range (scalar) config."""
    return _load_degenerate()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildEpisodeLogDict:
    """build_episode_log_dict() key-presence and value-range tests."""

    def test_all_five_keys_present_for_each_tag(self, dist_params):
        """All 5 Episode/sampled_*_<tag> keys appear for every animal tag."""
        state = jax_reset(dist_params, jax.random.PRNGKey(0))
        log = build_episode_log_dict(state, dist_params)

        expected_suffixes = [
            "sampled_detect",
            "sampled_max_stamina",
            "sampled_recovery",
            "sampled_hunt_thresh",
            "sampled_lose_interest",
        ]
        for tag in dist_params.animal_tags:
            for suffix in expected_suffixes:
                key = f"Episode/{suffix}_{tag}"
                assert key in log, (
                    f"Missing WandB key: {key!r}. "
                    f"log keys = {sorted(log.keys())}"
                )

    def test_logged_values_are_python_floats(self, dist_params):
        """All values in the dict are Python floats (not jax arrays)."""
        state = jax_reset(dist_params, jax.random.PRNGKey(1))
        log = build_episode_log_dict(state, dist_params)
        for k, v in log.items():
            assert isinstance(v, float), (
                f"Key {k!r} has type {type(v).__name__}, expected float"
            )

    def test_predator_values_within_bounds(self, dist_params):
        """Predator sampled values are within their configured [low, high]."""
        # predator0 is at index 0 — check 20 resets cover the range.
        for seed in range(20):
            state = jax_reset(dist_params, jax.random.PRNGKey(seed))
            log = build_episode_log_dict(state, dist_params)
            tag = "predator0"
            # detection_range: [0, 5]
            detect_val = log[f"Episode/sampled_detect_{tag}"]
            assert 0.0 <= detect_val <= 5.0, (
                f"detection_range out of [0, 5]: got {detect_val}"
            )
            # max_stamina: [20, 40]
            stam_val = log[f"Episode/sampled_max_stamina_{tag}"]
            assert 20.0 <= stam_val <= 40.0, (
                f"max_stamina out of [20, 40]: got {stam_val}"
            )
            # lose_interest_multiplier: 1.5 (degenerate)
            li_val = log[f"Episode/sampled_lose_interest_{tag}"]
            assert abs(li_val - 1.5) < 1e-5, (
                f"lose_interest should be 1.5 (degenerate), got {li_val}"
            )

    def test_values_change_across_episodes_non_degenerate(self, dist_params):
        """Non-degenerate ranges: sampled detect values differ across episodes."""
        detect_vals = []
        for seed in range(10):
            state = jax_reset(dist_params, jax.random.PRNGKey(seed * 13 + 7))
            log = build_episode_log_dict(state, dist_params)
            detect_vals.append(log["Episode/sampled_detect_predator0"])
        # With [0, 5] range and 10 distinct seeds, all-identical is astronomically unlikely.
        assert len(set(detect_vals)) > 1, (
            f"Expected detection_range values to vary across episodes; "
            f"got {detect_vals}"
        )

    def test_degenerate_values_constant_across_episodes(self, deg_params):
        """Degenerate ranges: sampled values are identical across all episodes."""
        detect_vals = []
        for seed in range(5):
            state = jax_reset(deg_params, jax.random.PRNGKey(seed))
            log = build_episode_log_dict(state, deg_params)
            detect_vals.append(log["Episode/sampled_detect_pred"])
        assert all(v == 5.0 for v in detect_vals), (
            f"Degenerate detect should always be 5.0; got {detect_vals}"
        )

    def test_wander_animals_have_zero_sampled_values(self, dist_params):
        """Wander animals carry zero-filled sampled fields (auto-filled [0,0])."""
        state = jax_reset(dist_params, jax.random.PRNGKey(42))
        log = build_episode_log_dict(state, dist_params)
        # rabbit0 and rabbit1 are wander animals with [0, 0] bounds → sampled = 0.0
        for tag in ("rabbit0", "rabbit1"):
            for suffix in ["sampled_detect", "sampled_max_stamina",
                           "sampled_recovery", "sampled_hunt_thresh",
                           "sampled_lose_interest"]:
                val = log[f"Episode/{suffix}_{tag}"]
                assert val == 0.0, (
                    f"Wander animal {tag!r} should have {suffix}=0.0, got {val}"
                )


class TestSampledWandBKeys:
    """sampled_wandb_keys() enumerates the expected key list."""

    def test_key_count(self, dist_params):
        """5 fields × N animals = total expected keys."""
        N = len(dist_params.animal_tags)
        keys = sampled_wandb_keys(dist_params.animal_tags)
        assert len(keys) == 5 * N, (
            f"Expected {5 * N} keys for {N} animals, got {len(keys)}"
        )

    def test_all_five_suffixes_per_tag(self, dist_params):
        """Every animal tag appears with all 5 field suffixes."""
        keys = sampled_wandb_keys(dist_params.animal_tags)
        key_set = set(keys)
        for tag in dist_params.animal_tags:
            for suffix in ["sampled_detect", "sampled_max_stamina",
                           "sampled_recovery", "sampled_hunt_thresh",
                           "sampled_lose_interest"]:
                assert f"Episode/{suffix}_{tag}" in key_set

    def test_keys_match_build_dict_keys(self, dist_params):
        """sampled_wandb_keys() keys are exactly the keys returned by build_episode_log_dict()."""
        state = jax_reset(dist_params, jax.random.PRNGKey(0))
        log = build_episode_log_dict(state, dist_params)
        expected_keys = set(sampled_wandb_keys(dist_params.animal_tags))
        actual_keys = set(log.keys())
        assert actual_keys == expected_keys, (
            f"Key mismatch.\n"
            f"In sampled_wandb_keys but not log: {expected_keys - actual_keys}\n"
            f"In log but not sampled_wandb_keys: {actual_keys - expected_keys}"
        )
