"""Tests for configurable per-entity visual properties (v3.0).

Covers the opt-in flexibility and custom-size (V≠8) paths — NOT parity tests.
These tests verify behaviour that is NEW in v3.0.

Checkpoints covered:
  CP3 — custom visual_properties changes obs as expected (only that cell).
  CP4 — visual_vector_size: 4 end-to-end; breakdown["Visual"] = num_vis_cells * 4.
  CP5 — obs↔noise width sync at V=4 (noised width == clean width).
  CP6 — V≠8 with missing visual_properties raises ValueError;
          length-mismatch raises ValueError.
"""
import os
import sys
import textwrap
import warnings

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

from src.utils.config import Config
from src.environment.config_loader import load_env_params, _resolve_extends
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation, get_observation_breakdown, apply_perceptual_noise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_params_from_yaml_str(yaml_str: str, extends_base: str = None):
    """Load env params from a YAML string, with optional extends: resolution."""
    d = yaml.safe_load(yaml_str)
    if extends_base is not None:
        # Use the extends: mechanism by creating a tmp config that extends base
        base_cfg = _resolve_extends(
            os.path.join(_ROOT, f"configs/{extends_base}.yaml"), frozenset()
        )
        from src.utils.config import Config as Cfg
        base_d = dict(base_cfg._data)
        # Deep-merge the override on top
        from src.utils.config import Config as C
        override = C(d)
        # Use _resolve_extends logic: merge base + override
        merged = base_cfg.merge(C(d))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return load_env_params(merged)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(d))


def _load_params_from_extends(cfg_path: str):
    """Load params using extends: resolution."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = _resolve_extends(os.path.join(_ROOT, cfg_path), frozenset())
        return load_env_params(cfg)


def _get_visual_slice(obs: np.ndarray, params) -> np.ndarray:
    """Extract the Visual sensor slice from a flattened observation vector."""
    breakdown = get_observation_breakdown(params)
    offset = 0
    for name, dim in breakdown.items():
        if name == "Visual":
            return obs[offset: offset + dim]
        offset += dim
    raise KeyError("'Visual' not in breakdown")


def _num_vis_cells(params) -> int:
    r = params.visual_sensor_range
    return int(2 * r * r + 2 * r + 1)


# ── CP3: Custom visual_properties changes only that entity's contribution ──────

def test_custom_vector_changes_cell(tmp_path):
    """CP3: A predator with all-zero visual_properties produces all-zero in its cell."""
    # Use 01-slowPred_5x5 as base (1 predator, visual_sensor_range=0 → agent's own cell)
    base_path = os.path.join(_ROOT, "configs", "environment", "experiment", "basic", "01-slowPred_5x5.yaml")
    if not os.path.exists(base_path):
        pytest.skip(f"Base config not found: {base_path}")

    # Load default params
    params_default = _load_params_from_extends("configs/environment/experiment/basic/01-slowPred_5x5.yaml")

    # Patch animal_visual_property: replace predator row with all-zeros
    pred_idx = list(params_default.predator_indices)
    vp_arr = np.array(params_default.animal_visual_property)
    for i in pred_idx:
        vp_arr[i] = 0.0
    # Rebuild params with custom property
    params_zero = params_default.replace(
        animal_visual_property=jnp.array(vp_arr, dtype=jnp.float32)
    )

    # Run 50 steps for each
    def run(p):
        key = jax.random.PRNGKey(42)
        state = jax_reset(p, key)
        slices = []
        for a in [0, 1, 2, 3, 4] * 10:
            state, _, _, _ = jax_step(state, a, p)
            obs = np.array(get_observation(state, p, apply_noise=False))
            slices.append(_get_visual_slice(obs, p))
        return np.stack(slices)

    default_vis = run(params_default)
    zero_vis = run(params_zero)

    V = params_default.visual_vector_size
    assert V == 8

    # The predator channel (5) in the default should have some non-zero values
    pred_channel = 5
    assert np.any(default_vis[:, pred_channel] != 0), (
        "Expected at least some non-zero predator-channel values in default"
    )

    # In the zero-property variant, the predator channel should be all-zero
    assert np.all(zero_vis[:, pred_channel] == 0), (
        "With all-zero visual_properties, predator channel should be zero everywhere"
    )

    # All other channels (non-predator) should be unchanged
    other_channels = [i for i in range(V) if i != pred_channel]
    np.testing.assert_array_equal(
        default_vis[:, other_channels],
        zero_vis[:, other_channels],
        err_msg="Non-predator channels should be identical when only predator VP is zeroed"
    )


# ── CP4: Custom width V≠8 end-to-end ─────────────────────────────────────────

def _make_v4_config() -> str:
    """Return a minimal YAML string for a V=4 environment."""
    return textwrap.dedent("""\
        environment:
          height: 5
          width: 5
          max_steps: 100
          random_start_pos: false
          start_pos: [3, 3]
          rest_action_enabled: false
          eat_action_enabled: false
          resources:
            - type: food
              count: 1
              spawn_area: [[1, 1], [5, 5]]
              max_consumption: 3
              regeneration_delay: 10
              damage: 0.0
              properties: [0.0, 0.0, 0.0, 1.0, 0.0]
              properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
              visual_properties: [1.0, 0.0, 0.0, 0.0]
          obstacles: []
          location_areas: []
          placement:
            mode: per_entity
        sensory:
          sensor_radius: 3.0
          decay_power: 1.5
          olfactory_sensor_range: 0
          visual_blur_enabled: false
          visual_blur_radial_scale: 0.5
          visual_blur_anisotropy: 3.0
          visual_blur_sigma_floor: 0.5
          collision_sensor_range: 1
          olfactory_enabled: true
          vector_size: 5
          nociception_enabled: false
          nociception_size: 1
          interoceptive_nociception_enabled: false
          interoceptive_convolution_enabled: false
          interoceptive_kernel_length: 1
          interoceptive_kernel_tau: 1.0
          visual_sensor_enabled: true
          visual_sensor_range: 0
          visual_vector_size: 4
          visual_background_properties:
            - [1.0, 0.0, 0.0, 0.0]
            - [0.0, 1.0, 0.0, 0.0]
            - [0.0, 0.0, 1.0, 0.0]
          proprioception_enabled: false
          location_sensor: false
          injury_observable: false
          nutrition_observable: false
        body:
          max_satiation: 1.0
          max_nutrition: 1.0
          max_injury: 1.0
          food_nutrition_gain: 0.3
          satiation_setpoint: 0.7
          start_satiation: 0.7
          start_nutrition: 0.7
          metabolic_cost: 0.01
          nutrition_to_satiation_scaling_factor: 1.0
          recovery_base_rate: 0.0
          recovery_accel_rate: 0.0
          injury_smoothing_duration: 1
          death_penalty: 0.0
          overeating_death: false
          use_homeostatic_reward: false
          with_satiation: true
          with_nutrition: true
          with_injury: true
          eating_nutrition_cost: 0.0
          eating_reward_penalty: 0.0
          random_start_satiation: false
          random_start_nutrition: false
          random_start_injury: false
          start_injury_low: 0.0
          start_injury_high: 0.0
        visualization:
          local_view_size: 5
        perceptual_noise:
          enabled: false
    """)


def test_custom_width_v4_end_to_end():
    """CP4: V=4 config runs end-to-end; breakdown['Visual'] == num_vis_cells * 4."""
    yaml_str = _make_v4_config()
    d = yaml.safe_load(yaml_str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        params = load_env_params(Config(d))

    assert params.visual_vector_size == 4

    # Check breakdown
    bd = get_observation_breakdown(params)
    assert "Visual" in bd
    nvc = _num_vis_cells(params)
    assert bd["Visual"] == nvc * 4, (
        f"Expected breakdown['Visual'] = {nvc * 4}, got {bd['Visual']}"
    )

    # Total obs width = sum of breakdown
    total = sum(bd.values())

    # Run episode — must not raise
    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)
    for a in [0, 1, 2, 3, 4] * 20:
        state, _, _, _ = jax_step(state, a, params)
        obs = np.array(get_observation(state, params, apply_noise=False))
        assert len(obs) == total, f"obs length {len(obs)} != expected {total}"
        vis = _get_visual_slice(obs, params)
        assert len(vis) == nvc * 4


# ── CP5: Obs↔noise width sync at V=4 ─────────────────────────────────────────

def _make_v4_config_with_noise() -> str:
    """V=4 config with visual noise enabled."""
    return textwrap.dedent("""\
        environment:
          height: 5
          width: 5
          max_steps: 100
          random_start_pos: false
          start_pos: [3, 3]
          rest_action_enabled: false
          eat_action_enabled: false
          resources:
            - type: food
              count: 1
              spawn_area: [[1, 1], [5, 5]]
              max_consumption: 3
              regeneration_delay: 10
              damage: 0.0
              properties: [0.0, 0.0, 0.0, 1.0, 0.0]
              properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
              visual_properties: [1.0, 0.0, 0.0, 0.0]
          obstacles: []
          location_areas: []
          placement:
            mode: per_entity
        sensory:
          sensor_radius: 3.0
          decay_power: 1.5
          olfactory_sensor_range: 0
          visual_blur_enabled: false
          visual_blur_radial_scale: 0.5
          visual_blur_anisotropy: 3.0
          visual_blur_sigma_floor: 0.5
          collision_sensor_range: 1
          olfactory_enabled: true
          vector_size: 5
          nociception_enabled: false
          nociception_size: 1
          interoceptive_nociception_enabled: false
          interoceptive_convolution_enabled: false
          interoceptive_kernel_length: 1
          interoceptive_kernel_tau: 1.0
          visual_sensor_enabled: true
          visual_sensor_range: 0
          visual_vector_size: 4
          visual_background_properties:
            - [1.0, 0.0, 0.0, 0.0]
            - [0.0, 1.0, 0.0, 0.0]
            - [0.0, 0.0, 1.0, 0.0]
          proprioception_enabled: false
          location_sensor: false
          injury_observable: false
          nutrition_observable: false
        body:
          max_satiation: 1.0
          max_nutrition: 1.0
          max_injury: 1.0
          food_nutrition_gain: 0.3
          satiation_setpoint: 0.7
          start_satiation: 0.7
          start_nutrition: 0.7
          metabolic_cost: 0.01
          nutrition_to_satiation_scaling_factor: 1.0
          recovery_base_rate: 0.0
          recovery_accel_rate: 0.0
          injury_smoothing_duration: 1
          death_penalty: 0.0
          overeating_death: false
          use_homeostatic_reward: false
          with_satiation: true
          with_nutrition: true
          with_injury: true
          eating_nutrition_cost: 0.0
          eating_reward_penalty: 0.0
          random_start_satiation: false
          random_start_nutrition: false
          random_start_injury: false
          start_injury_low: 0.0
          start_injury_high: 0.0
        visualization:
          local_view_size: 5
        perceptual_noise:
          enabled: true
          modalities:
            satiation:
              mode: constant
              sigma: 0.01
              injury_noise_scale: 0.0
              clip_min: 0.0
              clip_max: 1.0
            nutrition:
              mode: constant
              sigma: 0.01
              injury_noise_scale: 0.0
              clip_min: 0.0
              clip_max: 1.0
            injury:
              mode: constant
              sigma: 0.0
              injury_noise_scale: 0.0
              clip_min: 0.0
              clip_max: 1.0
            olfaction:
              mode: none
              sigma: 0.0
              injury_noise_scale: 0.0
              clip_min: -10.0
              clip_max: 10.0
            collision:
              mode: none
              sigma: 0.0
              injury_noise_scale: 0.0
              clip_min: 0.0
              clip_max: 1.0
            visual:
              mode: constant
              sigma: 0.01
              injury_noise_scale: 0.0
              clip_min: 0.0
              clip_max: 1.0
    """)


def test_obs_noise_width_sync_v4():
    """CP5: With visual noise enabled at V=4, noised obs has same width as clean obs."""
    yaml_str = _make_v4_config_with_noise()
    d = yaml.safe_load(yaml_str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        params = load_env_params(Config(d))

    assert params.visual_vector_size == 4
    assert params.perceptual_noise_enabled

    key = jax.random.PRNGKey(7)
    state = jax_reset(params, key)
    state, _, _, _ = jax_step(state, 0, params)

    clean_obs = get_observation(state, params, apply_noise=False)
    noise_key = jax.random.PRNGKey(99)
    noised_obs = apply_perceptual_noise(clean_obs, state, params, noise_key)

    assert clean_obs.shape == noised_obs.shape, (
        f"Noised obs shape {noised_obs.shape} != clean obs shape {clean_obs.shape}"
    )
    # The visual block in the breakdown should be nvc * 4
    bd = get_observation_breakdown(params)
    assert bd["Visual"] == _num_vis_cells(params) * 4


# ── CP6: Guard tests — V≠8 with missing visual_properties raises ───────────────

def _base_yaml_v4_resource_missing_vp() -> str:
    """V=4 config where resource is MISSING visual_properties (should raise)."""
    return textwrap.dedent("""\
        environment:
          height: 5
          width: 5
          max_steps: 100
          random_start_pos: false
          start_pos: [3, 3]
          rest_action_enabled: false
          eat_action_enabled: false
          resources:
            - type: food
              count: 1
              spawn_area: [[1, 1], [5, 5]]
              max_consumption: 3
              regeneration_delay: 10
              damage: 0.0
              properties: [0.0, 0.0, 0.0, 1.0, 0.0]
              properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
              # visual_properties intentionally OMITTED
          obstacles: []
          location_areas: []
          placement:
            mode: per_entity
        sensory:
          sensor_radius: 3.0
          decay_power: 1.5
          olfactory_sensor_range: 0
          visual_blur_enabled: false
          visual_blur_radial_scale: 0.5
          visual_blur_anisotropy: 3.0
          visual_blur_sigma_floor: 0.5
          collision_sensor_range: 1
          olfactory_enabled: true
          vector_size: 5
          nociception_enabled: false
          nociception_size: 1
          interoceptive_nociception_enabled: false
          interoceptive_convolution_enabled: false
          interoceptive_kernel_length: 1
          interoceptive_kernel_tau: 1.0
          visual_sensor_enabled: true
          visual_sensor_range: 0
          visual_vector_size: 4
          visual_background_properties:
            - [1.0, 0.0, 0.0, 0.0]
            - [0.0, 1.0, 0.0, 0.0]
            - [0.0, 0.0, 1.0, 0.0]
          proprioception_enabled: false
          location_sensor: false
          injury_observable: false
          nutrition_observable: false
        body:
          max_satiation: 1.0
          max_nutrition: 1.0
          max_injury: 1.0
          food_nutrition_gain: 0.3
          satiation_setpoint: 0.7
          start_satiation: 0.7
          start_nutrition: 0.7
          metabolic_cost: 0.01
          nutrition_to_satiation_scaling_factor: 1.0
          recovery_base_rate: 0.0
          recovery_accel_rate: 0.0
          injury_smoothing_duration: 1
          death_penalty: 0.0
          overeating_death: false
          use_homeostatic_reward: false
          with_satiation: true
          with_nutrition: true
          with_injury: true
          eating_nutrition_cost: 0.0
          eating_reward_penalty: 0.0
          random_start_satiation: false
          random_start_nutrition: false
          random_start_injury: false
          start_injury_low: 0.0
          start_injury_high: 0.0
        visualization:
          local_view_size: 5
        perceptual_noise:
          enabled: false
    """)


def test_v4_missing_resource_visual_properties_raises():
    """CP6a: V=4 config that omits resource visual_properties raises ValueError."""
    yaml_str = _base_yaml_v4_resource_missing_vp()
    d = yaml.safe_load(yaml_str)
    with pytest.raises(ValueError, match="visual_properties"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            load_env_params(Config(d))


def test_v4_missing_background_properties_raises():
    """CP6b: V=4 config that omits visual_background_properties raises ValueError."""
    yaml_str = _make_v4_config()
    d = yaml.safe_load(yaml_str)
    # Remove visual_background_properties
    del d['sensory']['visual_background_properties']
    with pytest.raises(ValueError, match="visual_background_properties"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            load_env_params(Config(d))


def test_length_mismatch_visual_properties_raises():
    """CP6c: visual_properties of wrong length raises ValueError."""
    yaml_str = _make_v4_config()
    d = yaml.safe_load(yaml_str)
    # Set resource visual_properties to length 3 (wrong for V=4)
    d['environment']['resources'][0]['visual_properties'] = [1.0, 0.0, 0.0]  # length 3 ≠ 4
    with pytest.raises(ValueError, match="length"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            load_env_params(Config(d))


def test_v8_default_no_visual_properties_needed():
    """At V=8 (default), omitting visual_properties everywhere is fine (no YAML edits needed)."""
    base_path = "configs/environment/experiment/basic/01-slowPred_5x5.yaml"
    if not os.path.exists(os.path.join(_ROOT, base_path)):
        pytest.skip(f"Base config not found: {base_path}")
    # This should load without error — no visual_properties in the YAML
    params = _load_params_from_extends(base_path)
    assert params.visual_vector_size == 8
    # Verify the default one-hot rows
    vp = np.array(params.animal_visual_property)
    for i in params.predator_indices:
        expected = np.zeros(8)
        expected[5] = 1.0
        np.testing.assert_array_equal(vp[i], expected)
