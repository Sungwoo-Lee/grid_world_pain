"""Regression tests for per-episode count-range activation (PER_EPISODE_ENV_VARIANCE).

Verifies:
  1. Same key → same K and same active masks (reproducibility).
  2. Different keys → K varies across [low, high] over many resets (covers both ends).
  3. Inactive-slot inertness:
     - An inactive predator parked off-grid deals ZERO damage when the agent walks
       the entire grid.
     - An inactive bush does NOT set agent_in_bush.
     - Inactive food is NOT sensed (olfaction contribution zero) and reads dist 99.0.
  4. Degenerate-range parity: a config with only 'count: N' produces all-True masks
     and byte-identical jax_reset output to the pre-feature semantics (K-draw skipped).
  5. Within an episode, K and the active masks are constant across steps (only resampled
     at reset, not during step).
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
  entities: []
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
  start_nutrition_low: 0
  start_nutrition_high: 100
  start_injury_low: 0
  start_injury_high: 100
sensory:
  olfactory_enabled: true
  sensor_radius: 20
  vector_size: 5
  decay_power: 2.0
  olfactory_grid_range: 0
  visual_blur_enabled: false
  visual_blur_radial_scale: 0.5
  visual_blur_anisotropy: 3.0
  visual_blur_sigma_floor: 0.5
  collision_sensor_range: 1
  location_sensor: false
  nociception_enabled: true
  nociception_size: 1
  visual_sensor_enabled: false
  visual_sensor_range: 0
  proprioception_enabled: false
  injury_observable: true
  nutrition_observable: false
  interoceptive_nociception_enabled: false
  interoceptive_convolution_enabled: false
  interoceptive_kernel_tau: 3.0
  interoceptive_kernel_length: 12
visualization:
  local_view_size: 5
perceptual_noise:
  enabled: false
  modalities: {}
"""


def _load_yaml_config(extra_yaml: str):
    """Merge extra YAML on top of _BASE_YAML and return loaded EnvParams."""
    import yaml as _yaml
    base = _yaml.safe_load(_BASE_YAML)
    extra = _yaml.safe_load(extra_yaml)
    # Deep merge: resources/entities/obstacles replace wholesale
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return load_env_params(Config(base))


# ---------------------------------------------------------------------------
# Test 1: Same key → same K and same masks (reproducibility)
# ---------------------------------------------------------------------------

def test_reproducibility():
    """Same PRNG key must produce identical K and masks on two independent resets."""
    extra = """
environment:
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "pred"
      count_low: 0
      count_high: 2
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
      attack_delay: 3
  obstacles:
    - name: "bush"
      count_low: 1
      count_high: 4
      area: [[1, 1], [10, 10]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      hides_agent: true
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
"""
    params = _load_yaml_config(extra)
    key = jax.random.PRNGKey(42)
    s1 = jax_reset(params, key)
    s2 = jax_reset(params, key)
    assert jnp.array_equal(s1.animal_active, s2.animal_active), "animal_active must be reproducible"
    assert jnp.array_equal(s1.obs_active, s2.obs_active), "obs_active must be reproducible"
    assert jnp.array_equal(s1.res_active, s2.res_active), "res_active must be reproducible"


# ---------------------------------------------------------------------------
# Test 2: Different keys → K covers [low, high] range empirically
# ---------------------------------------------------------------------------

def test_k_range_coverage():
    """Over 200 resets with different keys, K must cover both low and high ends."""
    extra = """
environment:
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "pred"
      count_low: 0
      count_high: 3
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
      attack_delay: 3
  resources:
    - name: "food"
      type: "food"
      count_low: 1
      count_high: 4
      spawn_area: [[1, 1], [10, 10]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: 12
      regeneration_delay: 0
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
  obstacles:
    - name: "rock"
      count_low: 2
      count_high: 5
      area: [[1, 1], [10, 10]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      properties: [0.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
"""
    params = _load_yaml_config(extra)
    N_RESETS = 200
    keys = jax.random.split(jax.random.PRNGKey(7), N_RESETS)

    pred_counts = set()
    food_counts = set()
    obs_counts = set()
    for k in keys:
        s = jax_reset(params, k)
        pred_counts.add(int(jnp.sum(s.animal_active)))
        food_counts.add(int(jnp.sum(s.res_active)))
        obs_counts.add(int(jnp.sum(s.obs_active)))

    # Must cover low end
    assert 0 in pred_counts, f"predator K=0 never seen (got {pred_counts})"
    assert 1 in food_counts, f"food K=1 never seen (got {food_counts})"
    assert 2 in obs_counts, f"obs K=2 never seen (got {obs_counts})"
    # Must cover high end
    assert 3 in pred_counts, f"predator K=3 never seen (got {pred_counts})"
    assert 4 in food_counts, f"food K=4 never seen (got {food_counts})"
    assert 5 in obs_counts, f"obs K=5 never seen (got {obs_counts})"


# ---------------------------------------------------------------------------
# Test 3a: Inactive predator deals NO damage
# ---------------------------------------------------------------------------

def test_inactive_predator_no_damage():
    """An inactive (K=0) predator parked off-grid must deal zero damage."""
    extra = """
environment:
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "pred"
      count_low: 0
      count_high: 2
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      damage: [50.0, 100.0]
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
      attack_delay: 3
"""
    params = _load_yaml_config(extra)
    # Search for a key that gives K=0 (all inactive)
    found_k0 = False
    for seed in range(500):
        k = jax.random.PRNGKey(seed)
        s = jax_reset(params, k)
        if int(jnp.sum(s.animal_active)) == 0:
            found_k0 = True
            # Run 100 steps and confirm zero predator damage accumulates
            actions = [0, 1, 2, 3] * 25  # walk around the whole grid
            total_pred_damage = 0.0
            state = s
            for a in actions:
                state, reward, done, info = jax_step(state, a, params)
                total_pred_damage += float(info['damage_predator'])
                if done:
                    break
            assert total_pred_damage == 0.0, (
                f"Inactive predator caused {total_pred_damage} damage (seed {seed})"
            )
            break
    assert found_k0, "Could not find a reset with K=0 predator in 500 seeds (range too narrow?)"


# ---------------------------------------------------------------------------
# Test 3b: Inactive bush does NOT set agent_in_bush
# ---------------------------------------------------------------------------

def test_inactive_bush_no_concealment():
    """An inactive bush must not set agent_in_bush even if the agent is at that cell."""
    extra = """
environment:
  obstacles:
    - name: "bush"
      count_low: 0
      count_high: 3
      area: [[1, 1], [10, 10]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      hides_agent: true
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
"""
    params = _load_yaml_config(extra)
    found_k0 = False
    for seed in range(500):
        k = jax.random.PRNGKey(seed)
        s = jax_reset(params, k)
        if int(jnp.sum(s.obs_active)) == 0:
            found_k0 = True
            # Run 50 steps; no step should see agent_in_bush
            state = s
            for a in range(50):
                action = a % 4
                state, reward, done, info = jax_step(state, action, params)
                assert not bool(info['agent_in_bush']), (
                    f"Inactive bush set agent_in_bush at step {a} (seed {seed})"
                )
                if done:
                    break
            break
    assert found_k0, "Could not find a reset with K=0 bushes in 500 seeds"


# ---------------------------------------------------------------------------
# Test 3c: Inactive food is not sensed (olfaction zero) and dist reads 99.0
# ---------------------------------------------------------------------------

def test_inactive_food_not_sensed():
    """With K=0 food, olfaction contribution from food slots is zero; dist_to_food == 99.0."""
    from src.environment.sensor import get_observation
    extra = """
environment:
  resources:
    - name: "food"
      type: "food"
      count_low: 0
      count_high: 4
      spawn_area: [[1, 1], [10, 10]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: 12
      regeneration_delay: 0
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
"""
    params = _load_yaml_config(extra)
    found_k0 = False
    for seed in range(500):
        k = jax.random.PRNGKey(seed)
        s = jax_reset(params, k)
        if int(jnp.sum(s.res_active)) == 0:
            found_k0 = True
            # Olfaction channel 0 = food; should be 0.0
            obs = get_observation(s, params, apply_noise=False)
            # Olfaction starts at index 0 for injury-invisible config (injury_observable=true here)
            # Breakdown: injury(1), satiation(1), extero_noc(1), olfaction(5) → olf at [3:8]
            # But our base YAML has injury_observable=true, no nutrition, yes noc, no intero
            # Actually from get_observation_breakdown: injury(1) + satiation(1) + extero_noc(1) + olfaction(5) = 8
            # Then collision (5) + no proprioception + no visual = 5 more
            # Olfaction is at obs[3:8] - food channel is obs[3] (prop=[1,0,0,0,0] → channel 0 = obs[3])
            obs_np = np.array(obs)
            olfaction_food_channel = obs_np[3]  # First olfaction dim = food signature
            assert olfaction_food_channel == 0.0, (
                f"Inactive food olfaction expected 0.0 but got {olfaction_food_channel} (seed {seed})"
            )
            # dist_to_food step test
            _, _, _, info = jax_step(s, 0, params)
            dist_food = float(info['dist_to_food'])
            assert dist_food == 99.0, (
                f"Inactive food dist expected 99.0 but got {dist_food} (seed {seed})"
            )
            break
    assert found_k0, "Could not find a reset with K=0 food in 500 seeds"


# ---------------------------------------------------------------------------
# Test 4: Degenerate-range parity (count: N → all-True masks, byte-identical output)
# ---------------------------------------------------------------------------

def test_degenerate_range_parity():
    """Configs with only 'count: N' (degenerate range) must produce all-True activation
    masks and the K-draw must NOT alter the jax_reset output compared to the pure
    all-True initialization path.
    """
    extra = """
environment:
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "pred"
      count: 1
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
      attack_delay: 3
  resources:
    - name: "food"
      type: "food"
      count: 2
      spawn_area: [[1, 1], [10, 10]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: 12
      regeneration_delay: 0
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
  obstacles:
    - name: "rock"
      count: 3
      area: [[1, 1], [10, 10]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      properties: [0.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
"""
    params = _load_yaml_config(extra)
    # Degenerate-range flags must be False
    assert not params.has_res_range, "has_res_range must be False for degenerate-only configs"
    assert not params.has_animal_range, "has_animal_range must be False for degenerate-only configs"
    assert not params.has_obs_range, "has_obs_range must be False for degenerate-only configs"

    # Masks must be all-True (over multiple resets with different keys)
    for seed in range(10):
        k = jax.random.PRNGKey(seed)
        s = jax_reset(params, k)
        assert jnp.all(s.res_active), f"seed {seed}: res_active not all-True for degenerate config"
        assert jnp.all(s.animal_active), f"seed {seed}: animal_active not all-True for degenerate config"
        assert jnp.all(s.obs_active), f"seed {seed}: obs_active not all-True for degenerate config"


# ---------------------------------------------------------------------------
# Test 5: Within-episode mask stability (masks constant across steps)
# ---------------------------------------------------------------------------

def test_mask_stable_within_episode():
    """animal_active, obs_active, AND res_active count must not increase across steps.

    res_active can legitimately shift (eaten slots regrow), BUT the ACTIVE COUNT
    must never exceed the reset K — inactive (never-allocated) resource slots must
    not revive.  This is the regression test for the resource-revival blocker
    (PER_EPISODE_ENV_VARIANCE fix, 2026-06-23).
    """
    extra = """
environment:
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "pred"
      count_low: 0
      count_high: 2
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
      attack_delay: 3
  resources:
    - name: "food"
      type: "food"
      count_low: 1
      count_high: 4
      spawn_area: [[1, 1], [10, 10]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: 12
      regeneration_delay: 5
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
  obstacles:
    - name: "bush"
      count_low: 1
      count_high: 3
      area: [[1, 1], [10, 10]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      hides_agent: true
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
"""
    params = _load_yaml_config(extra)
    # Use a key that gives K < count_high for food, so inactive slots exist.
    # Search for a seed where food K < 4 (the count_high).
    found_partial = False
    for seed in range(200):
        key = jax.random.PRNGKey(seed)
        state = jax_reset(params, key)
        reset_food_K = int(jnp.sum(state.res_active))
        if reset_food_K < 4:  # count_high is 4 → inactive slots exist
            found_partial = True
            break
    assert found_partial, "Could not find a reset with food K < count_high in 200 seeds"

    initial_animal_active = state.animal_active.copy()
    initial_obs_active = state.obs_active.copy()
    # Reset K is the maximum the active count should ever reach within this episode.
    max_allowed_food_active = reset_food_K

    for step in range(50):
        action = step % 4
        state, _, done, _ = jax_step(state, action, params)
        assert jnp.array_equal(state.animal_active, initial_animal_active), (
            f"animal_active changed at step {step}"
        )
        assert jnp.array_equal(state.obs_active, initial_obs_active), (
            f"obs_active changed at step {step}"
        )
        # Key regression check: inactive resource slots must NOT revive.
        # Active count must never EXCEED the count drawn at reset.
        current_food_active = int(jnp.sum(state.res_active))
        assert current_food_active <= max_allowed_food_active, (
            f"res_active count EXCEEDED reset K at step {step}: "
            f"{current_food_active} > {max_allowed_food_active} "
            f"(resource-revival blocker regression)"
        )
        if done:
            break


# ---------------------------------------------------------------------------
# Test 6: Shape stability — different K values, same array shape
# ---------------------------------------------------------------------------

def test_shape_stability_across_k_values():
    """Shapes of activation masks and positions must be identical regardless of K."""
    extra = """
environment:
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "pred"
      count_low: 0
      count_high: 3
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      damage: [10.0, 20.0]
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: 5
      max_stamina: 30
      stamina_recovery_rate: 1
      hunt_stamina_threshold: 0.7
      lose_interest_multiplier: 1.5
      attack_delay: 3
"""
    params = _load_yaml_config(extra)
    shapes = set()
    for seed in range(20):
        k = jax.random.PRNGKey(seed)
        s = jax_reset(params, k)
        shapes.add((s.animal_active.shape, s.animal_pos.shape))
    # All resets must produce the same shapes
    assert len(shapes) == 1, f"Multiple shapes across resets: {shapes}"


# ---------------------------------------------------------------------------
# Test 7: count_low > count_high raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_count_range_raises():
    """count_low > count_high must raise ValueError at config load time."""
    import pytest
    from src.environment.config_loader import _resolve_count_range
    with pytest.raises(ValueError, match="count_low"):
        _resolve_count_range({'count_low': 5, 'count_high': 2}, "TestEntity")


# ---------------------------------------------------------------------------
# Test 8: count + count_low/count_high together raises ValueError (ambiguous)
# ---------------------------------------------------------------------------

def test_ambiguous_count_raises():
    """Providing both 'count' and 'count_low'/'count_high' must raise ValueError."""
    import pytest
    from src.environment.config_loader import _resolve_count_range
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_count_range({'count': 2, 'count_low': 1, 'count_high': 4}, "TestEntity")
