"""Regression tests for inclusive-integer sampling of `attack_range` and
`detection_range`.

Plan: docs/develop/active/issues/INCLUSIVE_INTEGER_RANGE_SAMPLING.md

Before the fix, both fields were sampled as FLOATS uniformly from [lo, hi)
and compared against an INTEGER Manhattan distance via `dist <= sampled`.
Because a float draw from [lo, hi) is always strictly < hi, the top of the
range was unreachable: `attack_range: [2, 3]` behaved identically to
`[2, 2]` (a jump could never fire at distance 3), and `detection_range:
[1, 7]` could never trigger HUNT at distance 7.

After the fix both fields are sampled via `jax.random.randint([lo, hi+1))`,
mirroring `move_interval`/`attack_delay`, so `[lo, hi]` means the inclusive
integer set {lo, ..., hi}. Scalar configs (`detection_range: 5`) are
byte-identical before and after (degenerate range low==high).

Tests:
  1. attack_range: [2, 3] -> sampled in {2, 3}, both occur over many resets.
  2. Jump gate fires at Manhattan distance 3 when the per-episode sample
     lands on 3 (the case that is unreachable pre-fix).
  3. detection_range: [1, 7] -> sampled reaches all of {1..7}, including 7.
  4. Scalar attack_range=2 / detection_range=5 -> always sample exactly the
     scalar value (byte-identical to pre-fix behaviour).
  5. A fractional bound on either field raises ValueError at load time.
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

REST = 4

# ---------------------------------------------------------------------------
# Shared YAML base (mirrors tests/env/test_int_distributional_sampling.py)
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


def _load(extra_yaml: str):
    """Merge extra_yaml environment section into base and return EnvParams."""
    base = yaml.safe_load(_BASE_YAML)
    extra = yaml.safe_load(extra_yaml)
    for k, v in extra.get("environment", {}).items():
        base["environment"][k] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# 1 predator entity: attack_range: [2, 3], detection_range: [1, 7]
_ENTITY_RANGE_YAML = """
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
      detection_range: [1, 7]
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
      attack_range: [2, 3]
      attack_success_rate: 1.0
"""

# 1 predator with scalar attack_range=2 and scalar detection_range=5 (degenerate)
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
      attack_range: 2
      attack_success_rate: 1.0
"""

# Fractional attack_range bound -> must raise ValueError at load.
_ENTITY_FRACTIONAL_ATTACK_RANGE_YAML = """
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
      attack_range: [2, 3.5]
      attack_success_rate: 1.0
"""

# Fractional detection_range bound -> must raise ValueError at load.
_ENTITY_FRACTIONAL_DETECTION_RANGE_YAML = """
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
      detection_range: [1, 6.5]
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
"""

# Jump-gate geometry: agent at 0-indexed [0,0] (corner), predator forced to a
# fixed spawn cell at 0-indexed [3,0] -> Manhattan distance exactly 3.
# detection_range=[8,8] and hunt_stamina_threshold=0.1 guarantee HUNT on step 1
# (mirrors tests/env/test_predator_jump.py's _pred_entry geometry).
_JUMP_ENTITY_YAML = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: p1
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [10.0, 10.0]
      spawn_area: [[4, 1], [4, 1]]
      patrol_area: [[1, 1], [10, 10]]
      attack_delay: 3
      detection_range: [8, 8]
      max_stamina: [50, 50]
      stamina_recovery_rate: [2, 2]
      hunt_stamina_threshold: [0.1, 0.1]
      lose_interest_multiplier: [3.0, 3.0]
      attack_range: [2, 3]
      attack_success_rate: 1.0
"""


def _load_jump_gate_params():
    base = yaml.safe_load(_BASE_YAML)
    base["environment"]["start_pos"] = [1, 1]  # 1-indexed -> 0-indexed [0,0] corner
    extra = yaml.safe_load(_JUMP_ENTITY_YAML)
    base["environment"]["entities"] = extra["environment"]["entities"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


@pytest.fixture(scope="module")
def params_range():
    return _load(_ENTITY_RANGE_YAML)


@pytest.fixture(scope="module")
def params_scalar():
    return _load(_ENTITY_SCALAR_YAML)


# ---------------------------------------------------------------------------
# 1 — attack_range: [2, 3] -> sampled in {2, 3}, both occur
# ---------------------------------------------------------------------------

def test_attack_range_inclusive_both_ends_occur(params_range):
    keys = jax.random.split(jax.random.PRNGKey(42), 300)
    samples = []
    for k in keys:
        state = jax_reset(params_range, k)
        samples.append(int(state.animal_attack_range_sampled[0]))

    arr = np.array(samples)
    assert np.all(arr >= 2), f"Values below lo=2 found: {arr[arr < 2]}"
    assert np.all(arr <= 3), f"Values above hi=3 found: {arr[arr > 3]}"
    assert set(samples) == {2, 3}, (
        f"Expected both {{2, 3}} to occur over 300 draws; got {set(samples)}. "
        "Pre-fix (float-uniform [2,3)) never reaches 3."
    )


# ---------------------------------------------------------------------------
# 2 — jump gate fires at Manhattan distance 3 (unreachable pre-fix)
# ---------------------------------------------------------------------------

def test_jump_gate_fires_at_distance_3():
    """attack_range: [2, 3], attack_success_rate: 1.0, predator fixed at
    Manhattan distance 3 from the agent. Whenever the per-episode sample
    lands on 3, the jump MUST fire on step 1 (predator lands on the agent).

    Pre-fix, `animal_attack_range_sampled` was a float drawn from [2, 3) and
    could never equal exactly 3 -- so `found_3` below would stay False and
    this test would fail with an AssertionError (confirmed by temporarily
    reverting the core.py sampling call to `jax.random.uniform`; see the
    Implementation Report in INCLUSIVE_INTEGER_RANGE_SAMPLING.md).
    """
    params = _load_jump_gate_params()
    assert params.has_attack_feature is True

    found_3 = False
    for seed in range(60):
        key = jax.random.PRNGKey(seed)
        state = jax_reset(params, key)
        sampled = int(state.animal_attack_range_sampled[0])
        assert sampled in (2, 3), f"Sampled attack_range out of range: {sampled}"

        # Pre-jump geometry sanity: predator at 0-indexed [3,0], agent at [0,0], dist=3.
        assert list(np.array(state.animal_pos[0])) == [3, 0]
        assert list(np.array(state.agent_pos)) == [0, 0]

        if sampled == 3:
            found_3 = True
            state_after, reward, done, info = jax_step(state, REST, params)
            assert list(np.array(state_after.animal_pos[0])) == [0, 0], (
                "Jump must fire at exact boundary distance 3 when the sampled "
                "attack_range == 3"
            )
            assert bool(info["hit_predator"]) is True

    assert found_3, (
        "Sampled attack_range never equalled 3 across 60 seeds -- the top of "
        "the inclusive range [2,3] is unreachable (the pre-fix float-uniform bug)."
    )


# ---------------------------------------------------------------------------
# 3 — detection_range: [1, 7] -> sampled reaches all of {1..7}, including 7
# ---------------------------------------------------------------------------

def test_detection_range_reaches_top_of_range(params_range):
    keys = jax.random.split(jax.random.PRNGKey(7), 300)
    samples = []
    for k in keys:
        state = jax_reset(params_range, k)
        samples.append(int(state.animal_detect_sampled[0]))

    arr = np.array(samples)
    assert np.all(arr >= 1), f"Values below lo=1 found: {arr[arr < 1]}"
    assert np.all(arr <= 7), f"Values above hi=7 found: {arr[arr > 7]}"
    assert set(samples) == set(range(1, 8)), (
        f"Expected all of {{1..7}} to occur over 300 draws; got {sorted(set(samples))}. "
        "Pre-fix (float-uniform [1,7)) never reaches 7."
    )
    assert 7 in samples, "Top of detection_range [1,7] (value 7) never sampled."


# ---------------------------------------------------------------------------
# 4 — Scalar configs unchanged (byte-identical)
# ---------------------------------------------------------------------------

def test_scalar_attack_range_byte_identical(params_scalar):
    """scalar attack_range: 2 -> always samples exactly 2."""
    keys = jax.random.split(jax.random.PRNGKey(11), 50)
    for k in keys:
        state = jax_reset(params_scalar, k)
        val = int(state.animal_attack_range_sampled[0])
        assert val == 2, f"Degenerate attack_range=2 yielded sampled={val} from key {k}"


def test_scalar_detection_range_byte_identical(params_scalar):
    """scalar detection_range: 5 -> always samples exactly 5."""
    keys = jax.random.split(jax.random.PRNGKey(13), 50)
    for k in keys:
        state = jax_reset(params_scalar, k)
        val = int(state.animal_detect_sampled[0])
        assert val == 5, f"Degenerate detection_range=5 yielded sampled={val} from key {k}"


# ---------------------------------------------------------------------------
# 5 — Fractional bound raises ValueError
# ---------------------------------------------------------------------------

def test_fractional_attack_range_bound_raises():
    with pytest.raises(ValueError, match="attack_range"):
        _load(_ENTITY_FRACTIONAL_ATTACK_RANGE_YAML)


def test_fractional_detection_range_bound_raises():
    with pytest.raises(ValueError, match="detection_range"):
        _load(_ENTITY_FRACTIONAL_DETECTION_RANGE_YAML)
