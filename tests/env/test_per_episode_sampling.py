"""CP2 — per-episode sampling tests.

Verifies that the 5 distributional behavioural fields
(detection_range, max_stamina, stamina_recovery_rate,
hunt_stamina_threshold, lose_interest_multiplier) are sampled
uniformly from [low, high] at every env reset, independently
per-entity, with byte-identical reproducibility from the same key.

Tests:
  1. Same key -> same sampled values (reproducibility).
  2. Different key -> different sampled values (divergence).
  3. N entities of the same class -> N independent samples (not N copies).
  4. Per-episode-sampled fields live on EnvState, NOT on EnvParams
     (code-reviewer Missed-failure-mode #1).
  5. Cross-field independence: pairwise Pearson |r| < 0.5 across 100 resets.
  6. Degenerate-range guard: low==high -> sampled value is exactly low.
  7. Wander/static entities carry the fields but state.animal_state stays 0
     for wander/static after 1000 steps (Missed-failure-modes #3, #4).
  8. Zero-animal smoke (M6 v0.3): empty config -> no exception, correct
     empty shapes, hit_* False, dist_to_* == 99.0.
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
from src.environment.core import jax_reset, jax_step

# ── Shared YAML pieces ────────────────────────────────────────────────────────

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
  visual_value_mode: sum
  visual_occlusion_enabled: false
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
"""

# Config with 4 predators (class=predator, behaviour=hunt) with NON-DEGENERATE ranges.
# Each predator uses the same ranges but gets independent draws.
_ENTITIES_4PRED_YAML = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: p1
      count: 4
      properties: [0.0, 0.8, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [10.0, 30.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 2
      detection_range: [1, 8]
      max_stamina: [20, 50]
      stamina_recovery_rate: [0.5, 2.0]
      hunt_stamina_threshold: [0.3, 0.8]
      lose_interest_multiplier: [1.0, 3.0]
"""

# Config with 1 predator + 1 wander + 1 static — for wander/static state check.
_ENTITIES_MIXED_YAML = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: pred
      count: 1
      properties: [0.0, 0.8, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [10.0, 30.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 2
      detection_range: [1, 8]
      max_stamina: [20, 50]
      stamina_recovery_rate: [0.5, 2.0]
      hunt_stamina_threshold: [0.3, 0.8]
      lose_interest_multiplier: [1.0, 3.0]
    - class: neutral
      behaviour: wander
      tag: wand
      count: 1
      properties: [0.0, 0.5, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 0
"""

# Config with ZERO animals — zero-animal smoke (M6).
_ENTITIES_EMPTY_YAML = """
environment:
  predators: []
  neutral_animals: []
"""


def _load(extra_yaml: str):
    """Merge extra_yaml into base and return loaded EnvParams."""
    base = yaml.safe_load(_BASE_YAML)
    extra = yaml.safe_load(extra_yaml)
    for k, v in extra.get("environment", {}).items():
        base["environment"][k] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def params_4pred():
    return _load(_ENTITIES_4PRED_YAML)


@pytest.fixture(scope="module")
def params_mixed():
    return _load(_ENTITIES_MIXED_YAML)


@pytest.fixture(scope="module")
def params_empty():
    return _load(_ENTITIES_EMPTY_YAML)


# ── Test 1: Reproducibility (same key -> same sampled values) ─────────────────

def test_same_key_same_samples(params_4pred):
    """Same PRNG key produces byte-identical per-episode samples."""
    key = jax.random.PRNGKey(7)
    state_a = jax_reset(params_4pred, key)
    state_b = jax_reset(params_4pred, key)
    np.testing.assert_array_equal(
        np.array(state_a.animal_detect_sampled),
        np.array(state_b.animal_detect_sampled),
        err_msg="animal_detect_sampled not reproducible with same key",
    )
    np.testing.assert_array_equal(
        np.array(state_a.animal_max_stamina_sampled),
        np.array(state_b.animal_max_stamina_sampled),
    )
    np.testing.assert_array_equal(
        np.array(state_a.animal_recovery_sampled),
        np.array(state_b.animal_recovery_sampled),
    )
    np.testing.assert_array_equal(
        np.array(state_a.animal_hunt_thresh_sampled),
        np.array(state_b.animal_hunt_thresh_sampled),
    )
    np.testing.assert_array_equal(
        np.array(state_a.animal_lose_interest_sampled),
        np.array(state_b.animal_lose_interest_sampled),
    )


# ── Test 2: Different key -> different sampled values ─────────────────────────

def test_different_key_different_samples(params_4pred):
    """Different PRNG keys produce different per-episode samples (with high probability)."""
    key_a = jax.random.PRNGKey(0)
    key_b = jax.random.PRNGKey(1)
    state_a = jax_reset(params_4pred, key_a)
    state_b = jax_reset(params_4pred, key_b)
    # At least one field should differ (probability of all being equal is negligible)
    all_same = all(
        np.array_equal(
            np.array(getattr(state_a, f)),
            np.array(getattr(state_b, f)),
        )
        for f in [
            "animal_detect_sampled",
            "animal_max_stamina_sampled",
            "animal_recovery_sampled",
            "animal_hunt_thresh_sampled",
            "animal_lose_interest_sampled",
        ]
    )
    assert not all_same, (
        "All per-episode sampled fields are identical across different keys — "
        "sampling is not consuming the key correctly."
    )


# ── Test 3: N entities of the same class -> N independent samples ─────────────

def test_per_instance_independence(params_4pred):
    """4 predators get 4 independent detection_range draws (not 4 copies)."""
    key = jax.random.PRNGKey(42)
    state = jax_reset(params_4pred, key)
    vals = np.array(state.animal_detect_sampled)
    # With 4 draws from [1, 8], the probability of all 4 being equal is < 1e-10.
    assert len(set(vals.tolist())) > 1, (
        f"All 4 per-instance detection_range values are identical: {vals}. "
        "Sampling is producing copies instead of independent draws."
    )


# ── Test 4: Sampled fields on EnvState, NOT EnvParams ─────────────────────────

def test_sampled_fields_on_state_not_params(params_4pred):
    """Per-episode-sampled fields must be on EnvState, not on EnvParams (code-reviewer MF#1)."""
    key = jax.random.PRNGKey(0)
    state = jax_reset(params_4pred, key)
    params = params_4pred
    for field in [
        "animal_detect_sampled",
        "animal_max_stamina_sampled",
        "animal_recovery_sampled",
        "animal_hunt_thresh_sampled",
        "animal_lose_interest_sampled",
    ]:
        assert hasattr(state, field), (
            f"EnvState is missing field '{field}' — CP2 sampling not stored on state."
        )
        assert not hasattr(params, field), (
            f"EnvParams unexpectedly has field '{field}' — per-episode samples "
            "belong on EnvState only."
        )


# ── Test 5: Cross-field independence (pairwise Pearson |r| < 0.5) ─────────────

def test_cross_field_independence(params_4pred):
    """Pairwise Pearson |r| < 0.5 across 100 resets for each pair of sampled fields.

    Guards against reusing the same sub-key for multiple fields.
    """
    N_RESETS = 100
    field_names = [
        "animal_detect_sampled",
        "animal_max_stamina_sampled",
        "animal_recovery_sampled",
        "animal_hunt_thresh_sampled",
        "animal_lose_interest_sampled",
    ]
    # Collect mean value per field across N_RESETS resets.
    # (Mean across entities so we get a 1-D vector per field.)
    samples = {f: [] for f in field_names}
    for i in range(N_RESETS):
        key = jax.random.PRNGKey(i + 1000)
        state = jax_reset(params_4pred, key)
        for f in field_names:
            vals = np.array(getattr(state, f))
            samples[f].append(float(np.mean(vals)))

    for f in field_names:
        samples[f] = np.array(samples[f])

    # Pairwise correlation check
    from itertools import combinations
    for fa, fb in combinations(field_names, 2):
        corr = float(np.corrcoef(samples[fa], samples[fb])[0, 1])
        assert abs(corr) < 0.5, (
            f"Fields '{fa}' and '{fb}' are correlated: r={corr:.3f} >= 0.5. "
            "They likely share a sub-key."
        )


# ── Test 6: Degenerate-range guard (low==high -> sampled == low) ───────────────

def test_degenerate_range_returns_low(params_4pred):
    """Legacy scalar ranges (low==high) produce exactly `low`, not NaN or random.

    The 4 predators in params_4pred use non-degenerate ranges, but the bounds for
    wander/static entities are auto-filled to [0, 0] — i.e., degenerate.
    Test the degenerate sub-case directly by loading a fresh config with scalar
    detection_range, and confirm the sampled value == scalar.
    """
    _DEGENERATE_YAML = """
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
    params_degen = _load(_DEGENERATE_YAML)
    key = jax.random.PRNGKey(0)
    state = jax_reset(params_degen, key)
    detect_val = float(state.animal_detect_sampled[0])
    assert detect_val == pytest.approx(5.0), (
        f"Degenerate detection_range=5 produced sampled value {detect_val}, expected 5.0."
    )
    stamina_val = float(state.animal_max_stamina_sampled[0])
    assert stamina_val == pytest.approx(30.0), (
        f"Degenerate max_stamina=30 produced sampled value {stamina_val}, expected 30.0."
    )


# ── Test 7: Wander/static entities keep animal_state==0 after 1000 steps ──────

def test_wander_static_animal_state_stays_zero(params_mixed):
    """Wander/static entities carry per-episode draws but animal_state stays 0 throughout.

    Verifies intentional shape uniformity: the wander branch never reads the
    per-episode distributional fields (MF#3, MF#4 from code-reviewer).
    """
    key = jax.random.PRNGKey(0)
    state = jax_reset(params_mixed, key)

    wander_idx = list(params_mixed.wander_idx)
    static_idx = list(params_mixed.static_idx)
    combined_idx = wander_idx + static_idx
    assert len(combined_idx) > 0, "Expected at least 1 wander/static entity in mixed config."

    actions = [0, 1, 2, 3, 4] * 200  # 1000 steps
    for action in actions:
        state, _, _, _ = jax_step(state, action, params_mixed)
        for idx in combined_idx:
            s = int(state.animal_state[idx])
            assert s == 0, (
                f"Wander/static entity at index {idx} has animal_state={s} at step — "
                "should stay 0 (PATROL state) throughout."
            )


# ── Test 8: Zero-animal smoke (M6 v0.3) ─────────────────────────────────────

def test_zero_animal_smoke(params_empty):
    """Config with no animals: reset+100 steps, no exception, correct shapes (M6)."""
    key = jax.random.PRNGKey(0)
    # (a) no exception during reset
    state = jax_reset(params_empty, key)

    # (b) correct empty shapes
    assert state.animal_pos.shape == (0, 2), (
        f"Expected animal_pos.shape (0, 2), got {state.animal_pos.shape}"
    )
    for field in [
        "animal_detect_sampled",
        "animal_max_stamina_sampled",
        "animal_recovery_sampled",
        "animal_hunt_thresh_sampled",
        "animal_lose_interest_sampled",
    ]:
        shape = getattr(state, field).shape
        assert shape == (0,), (
            f"Expected {field}.shape (0,), got {shape}"
        )

    # (c) hit_* False, dist_to_* == 99.0 across 100 steps (M3 guard)
    actions = [0, 1, 2, 3, 4] * 20  # 100 steps
    for action in actions:
        state, _, _, info = jax_step(state, action, params_empty)
        assert not bool(info["hit_predator"]), "hit_predator should be False with no animals"
        assert not bool(info["hit_neutral"]), "hit_neutral should be False with no animals"

    # Check dist_to_pred and dist_to_neutral after last step
    assert float(info["dist_to_pred"]) == pytest.approx(99.0), (
        f"dist_to_pred should fall back to 99.0 with no animals, got {info['dist_to_pred']}"
    )
    assert float(info["dist_to_neutral"]) == pytest.approx(99.0), (
        f"dist_to_neutral should fall back to 99.0 with no animals, got {info['dist_to_neutral']}"
    )
