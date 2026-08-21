"""CP3 — entities: unified schema tests.

Tests:
  1. Smoke config (01-entities-smoke.yaml) loads without error.
  2. Byte-parity: smoke config (unified schema) vs legacy config
     (01-interoNocicept_sameProp.yaml) — 100 steps from seed 0,
     byte-identical obs.
  3. Both legacy + unified sections present → loader prefers the legacy sections
     (precedence flipped by FIX_CONFIG_LAYER_SILENT_FAILURES_20260723 Bug 1),
     emits DeprecationWarning.
  4. Config with residual predator_enabled key raises ValueError.
  5. Unified config with behaviour: hunt and missing detection_range
     raises ValueError.
  6. Unified config with behaviour: wander and no distributional fields
     loads fine (auto-fill [0, 0]).
  7. 4-entity mixed-behaviour config loads correctly with expected idx tuples
     (hunt_idx==(0,2), wander_idx==(1,), static_idx==(3,)) — MF#2.

Pre-flight note: grep -rln "mode: per_type" configs/ produced no results
(verified in CP1 pre-flight and re-confirmed in CP3). All configs use
per_entity or global placement mode. No type_entity_map re-projection
needed at this stage.
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
from src.environment.sensor import get_observation

# ── Config paths ──────────────────────────────────────────────────────────────

_SMOKE_CFG = os.path.join(
    _ROOT, "configs", "experiment", "v2_smoke", "01-entities-smoke.yaml"
)
_LEGACY_CFG = os.path.join(
    _ROOT, "configs", "experiment", "hypervigilance", "01-interoNocicept_sameProp.yaml"
)

ACTIONS = [0, 1, 2, 3, 4] * 20  # 100 steps


def _load_file(path: str):
    with open(path) as f:
        d = yaml.safe_load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(d))


def _run_episode(params, seed=0):
    """Reset + 100 steps. Returns list of obs arrays."""
    key = jax.random.PRNGKey(seed)
    state = jax_reset(params, key)
    obs_list = []
    for action in ACTIONS:
        state, _, _, _ = jax_step(state, action, params)
        obs_list.append(np.array(get_observation(state, params, apply_noise=False)))
    return obs_list


# ── Minimal YAML base (for inline schema tests) ───────────────────────────────

_MINIMAL_BASE = """
environment:
  height: 5
  width: 5
  start_pos: [1, 1]
  random_start_pos: false
  max_steps: 10
  rest_action_enabled: true
  eat_action_enabled: false
  resources: []
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
"""

_HUNT_ENTITY = """
    - class: predator
      behaviour: hunt
      tag: wolf
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

_WANDER_ENTITY = """
    - class: neutral
      behaviour: wander
      tag: rabbit
      count: 1
      properties: [0.5, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 3
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
      attack_delay: 0
"""


def _make_config(extra_yaml=""):
    base = yaml.safe_load(_MINIMAL_BASE)
    if extra_yaml:
        extra = yaml.safe_load(extra_yaml)
        for k, v in extra.get("environment", {}).items():
            base["environment"][k] = v
    return Config(base)


def _load(extra_yaml=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(_make_config(extra_yaml))


# ── Test 1: Smoke config loads ────────────────────────────────────────────────

def test_smoke_config_loads():
    """01-entities-smoke.yaml loads without error."""
    if not os.path.exists(_SMOKE_CFG):
        pytest.skip(f"Smoke config not found: {_SMOKE_CFG}")
    params = _load_file(_SMOKE_CFG)
    # 1 predator + 2 neutrals = 3 animals total
    assert params.animal_property.shape[0] == 3, (
        f"Expected 3 animals (1 pred + 2 neutrals), got {params.animal_property.shape[0]}"
    )
    assert len(params.hunt_idx) == 1
    assert len(params.wander_idx) == 2
    assert len(params.static_idx) == 0


# ── Test 2: Byte-parity vs legacy config ─────────────────────────────────────

def test_entities_smoke_byte_parity():
    """Unified-schema smoke config produces byte-identical obs to legacy config (100 steps)."""
    if not os.path.exists(_SMOKE_CFG):
        pytest.skip(f"Smoke config not found: {_SMOKE_CFG}")
    if not os.path.exists(_LEGACY_CFG):
        pytest.skip(f"Legacy config not found: {_LEGACY_CFG}")

    params_smoke = _load_file(_SMOKE_CFG)
    params_legacy = _load_file(_LEGACY_CFG)

    obs_smoke = _run_episode(params_smoke)
    obs_legacy = _run_episode(params_legacy)

    for step_i, (os_, ol) in enumerate(zip(obs_smoke, obs_legacy)):
        np.testing.assert_array_equal(
            os_, ol,
            err_msg=(
                f"Observation mismatch at step {step_i}: "
                f"smoke obs != legacy obs.\n"
                f"First differing element index: {np.where(os_ != ol)[0]}"
            ),
        )


# ── Test 3: Both schemas present → unified wins + DeprecationWarning ──────────

def test_both_schemas_warns_and_prefers_legacy():
    """Loader emits DeprecationWarning when both entities: and legacy sections present.

    Precedence updated by FIX_CONFIG_LAYER_SILENT_FAILURES_20260723 (Bug 1): when the
    user file itself authors legacy predators:/neutral_animals: sections, THOSE win —
    even though an entities: list is also present here (in this test, authored
    directly in the same file; in practice this precedence matters when entities:
    instead arrives via the default.yaml base underlay under train.py). This keeps
    train.py and eval_rollout.py agreeing on legacy configs' animal scene.
    """
    both_yaml = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: wolf
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
  predators:
    - tag: legacy_wolf
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        params = load_env_params(_make_config(both_yaml))
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) > 0, "Expected DeprecationWarning when both schemas present."
        assert any("entities" in str(dw.message).lower() or "unified" in str(dw.message).lower()
                   for dw in dep_warnings), (
            f"DeprecationWarning not about entities/unified: {[str(dw.message) for dw in dep_warnings]}"
        )
    # Legacy took precedence: animal_tags from predators:/neutral_animals: schema
    assert "legacy_wolf" in params.animal_tags, (
        f"Expected 'legacy_wolf' (legacy) in animal_tags, got {params.animal_tags}"
    )
    assert "wolf" not in params.animal_tags, (
        f"'wolf' (unified) should have been ignored; animal_tags={params.animal_tags}"
    )


# ── Test 4: Residual predator_enabled raises ValueError ───────────────────────

def test_predator_enabled_still_raises():
    """Configs carrying residual predator_enabled: raise ValueError (sanity check)."""
    raw = yaml.safe_load(_MINIMAL_BASE)
    raw["environment"]["predator_enabled"] = True
    raw["environment"]["predators"] = []
    raw["environment"]["neutral_animals"] = []
    config = Config(raw)
    with pytest.raises(ValueError, match="predator_enabled"):
        load_env_params(config)


# ── Test 5: hunt entity with missing detection_range → ValueError ─────────────

def test_hunt_missing_dist_field_raises():
    """behaviour: hunt with missing detection_range raises ValueError (no fallback default)."""
    missing_yaml = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: wolf
      count: 1
      properties: [0.0, 0.9, 0.0, 0.0, 0.9]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      attack_delay: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      # detection_range is intentionally MISSING — should raise ValueError
      max_stamina: 30.0
      stamina_recovery_rate: 1.0
      hunt_stamina_threshold: 0.5
      lose_interest_multiplier: 2.0
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
"""
    with pytest.raises(ValueError, match="detection_range"):
        _load(missing_yaml)


# ── Test 6: wander entity with no distributional fields loads fine ─────────────

def test_wander_without_dist_fields_ok():
    """behaviour: wander without distributional fields loads fine (auto-fill [0, 0])."""
    wander_yaml = f"environment:\n  entities:\n{_WANDER_ENTITY}"
    params = _load(wander_yaml)
    assert params.animal_property.shape[0] == 1
    assert len(params.wander_idx) == 1
    # Auto-filled bounds should be 0.0
    assert float(params.animal_detect_low[0]) == pytest.approx(0.0)
    assert float(params.animal_detect_high[0]) == pytest.approx(0.0)


# ── Test 7: 4-entity mixed-behaviour config — idx tuples correct ──────────────

def test_mixed_behaviour_idx_tuples():
    """[hunt, wander, hunt, static] → hunt_idx==(0,2), wander_idx==(1,), static_idx==(3,).

    Covers code-reviewer Missed-failure-mode #2 (behaviour-mask axis test).
    """
    mixed_yaml = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: h0
      count: 1
      properties: [0.0, 0.9, 0.0, 0.0, 0.9]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      attack_delay: 2
      damage: [5.0, 15.0]
      nociception_intensity: 0.9
      detection_range: 3
      max_stamina: 20.0
      stamina_recovery_rate: 1.0
      hunt_stamina_threshold: 0.5
      lose_interest_multiplier: 1.5
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
    - class: neutral
      behaviour: wander
      tag: w1
      count: 1
      properties: [0.5, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
      attack_delay: 0
    - class: predator
      behaviour: hunt
      tag: h2
      count: 1
      properties: [0.0, 0.9, 0.0, 0.0, 0.9]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      attack_delay: 2
      damage: [5.0, 15.0]
      nociception_intensity: 0.9
      detection_range: 3
      max_stamina: 20.0
      stamina_recovery_rate: 1.0
      hunt_stamina_threshold: 0.5
      lose_interest_multiplier: 1.5
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
    - class: neutral
      behaviour: static
      tag: s3
      count: 1
      properties: [0.3, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      spawn_area: [[1, 1], [5, 5]]
      patrol_area: [[1, 1], [5, 5]]
      attack_delay: 0
"""
    params = _load(mixed_yaml)
    assert params.hunt_idx == (0, 2), (
        f"Expected hunt_idx==(0,2), got {params.hunt_idx}"
    )
    assert params.wander_idx == (1,), (
        f"Expected wander_idx==(1,), got {params.wander_idx}"
    )
    assert params.static_idx == (3,), (
        f"Expected static_idx==(3,), got {params.static_idx}"
    )
