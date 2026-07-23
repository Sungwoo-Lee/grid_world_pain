"""Regression test for ghost-predator bug (PER_EPISODE_ENV_VARIANCE).

Bug: update_animals moved ALL animal slots each step, including inactive ones
(animal_active=False).  Inactive slots were parked at the off-grid sentinel
(params.height, params.width) at reset, but the hunt/wander movement logic had
NO animal_active gate.  So after step 1 the inactive slot was clipped by the
hard-grid boundary (jnp.clip(..., 0, [grid_h-1, grid_w-1])) and ended up at
(9, 9) — on-grid — chasing the agent visually while dealing 0 damage.

Fix (core.py update_animals): after all subset movement updates compute new_pos,
re-park inactive slots off-grid every step:
    off_grid = jnp.array([params.height, params.width], dtype=jnp.int32)
    new_pos = jnp.where(state.animal_active[:, None], new_pos, off_grid[None, :])
This mirrors the reset parking and is vmap-safe / recompile-safe.  For all-active
configs animal_active is all-True, so jnp.where(True, new_pos, off_grid) == new_pos
— byte-identical, no-op.

Plan: docs/develop/active/refactors/PER_EPISODE_ENV_VARIANCE.md
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
from src.environment.core import jax_reset, jax_step

# ── Shared YAML: count_low=0, count_high=2 predator (triggers count-range path) ──

_RANGE_YAML = """
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
  entities:
    - class: "predator"
      behaviour: "hunt"
      tag: "pred"
      count_low: 0
      count_high: 2
      properties: [0.0, 0.7, 0.5, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      visual_properties: [0, 0, 0, 0, 0, 1, 0, 0]
      visual_properties_std: [0, 0, 0, 0, 0, 0, 0, 0]
      move_interval: 1
      damage: [50.0, 80.0]
      attack_delay: 3
      nociception_intensity: 0.9
      spawn_area: [[1, 1], [10, 10]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: 8
      max_stamina: 30
      stamina_recovery_rate: 2
      hunt_stamina_threshold: 0.3
      lose_interest_multiplier: 2.0
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
  metabolic_cost: 0.1
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
  olfactory_enabled: false
  sensor_radius: 5.0
  vector_size: 5
  decay_power: 1.0
  collision_sensor_range: 1
  location_sensor: false
  nociception_enabled: true
  nociception_size: 1
  visual_sensor_enabled: false
  visual_sensor_range: 1
  proprioception_enabled: false
  injury_observable: true
  nutrition_observable: false
  interoceptive_nociception_enabled: false
  interoceptive_convolution_enabled: false
  interoceptive_kernel_tau: 3.0
  interoceptive_kernel_length: 5
visualization:
  local_view_size: 3
perceptual_noise:
  enabled: false
  modalities: {}
"""


def _range_params():
    cfg = yaml.safe_load(_RANGE_YAML)
    return load_env_params(Config(cfg))


def _find_state_with_inactive(params, max_seeds: int = 500):
    """Return first (state, seed) where at least one predator slot is inactive (K < 2)."""
    for seed in range(max_seeds):
        k = jax.random.PRNGKey(seed)
        state = jax_reset(params, k)
        if not jnp.all(state.animal_active):
            return state, seed
    pytest.skip(
        f"No inactive predator slot found in {max_seeds} seeds — widen count range"
    )


# ---------------------------------------------------------------------------
# Test 1 — Core regression: inactive slot stays off-grid every step
# ---------------------------------------------------------------------------

def test_inactive_slot_stays_offgrid():
    """Ghost-predator regression: inactive predator slot must remain at (height, width) every step.

    On the buggy code update_animals had no animal_active gate, so after step 1
    the inactive slot was clipped by the hard-grid boundary to (9, 9) — on-grid.
    The fix re-parks inactive slots to (height, width) after every movement update.

    Procedure:
      1. Reset until at least one predator slot is inactive (K < count_high=2).
      2. Assert inactive slot is at (height, width) — confirms reset still works.
      3. Step 15 times with action=4 (rest); agent stays at (5,5).
      4. Assert inactive slot position == (height, width) at EVERY step.
         Pre-fix: fails at step 1 (slot clipped to (9,9) by grid boundaries).
         Post-fix: passes all 15 steps.
    """
    params = _range_params()
    off_grid = (params.height, params.width)  # (10, 10) for a 10x10 grid

    state, seed = _find_state_with_inactive(params)

    inactive_np = np.array(state.animal_active)
    inactive_indices = np.where(~inactive_np)[0]
    assert len(inactive_indices) > 0, "Test setup error: no inactive slot in state"
    inactive_idx = int(inactive_indices[0])

    # Verify reset correctly parks the slot off-grid
    pos_at_reset = tuple(int(x) for x in np.array(state.animal_pos[inactive_idx]))
    assert pos_at_reset == off_grid, (
        f"Reset did not park inactive slot at {off_grid}: got {pos_at_reset} "
        f"(seed {seed}, slot {inactive_idx})"
    )

    # Step 15 times with rest; assert inactive slot stays off-grid
    for step_i in range(15):
        state, _reward, done, _info = jax_step(state, 4, params)
        pos = tuple(int(x) for x in np.array(state.animal_pos[inactive_idx]))
        assert pos == off_grid, (
            f"Step {step_i + 1}: inactive slot moved to {pos} instead of staying "
            f"off-grid at {off_grid}. Ghost-predator bug: slot un-parked by "
            f"movement update without animal_active gate. "
            f"(seed {seed}, slot {inactive_idx})"
        )
        if done:
            break


# ---------------------------------------------------------------------------
# Test 2 — Active slot still chases and can damage (guards against over-masking)
# ---------------------------------------------------------------------------

def test_active_slot_still_chases_and_damages():
    """Active predator slots must still move and deal damage after the fix.

    Guards against the fix accidentally masking ACTIVE slots.  We manually
    place an active predator adjacent to the resting agent and confirm that
    injury rises — i.e. the fix only re-parks INACTIVE slots.

    Procedure:
      1. Find a seed with exactly K=1 (one active, one inactive predator).
      2. Use state.replace() to put the active predator at (5, 6) — one step
         from the agent at (5, 5) — in hunt state with full stamina.
      3. Step up to 20 times.
      4. Assert injury_level rises above 0 (active predator dealt damage).
      5. Assert the inactive slot stayed at (height, width).
    """
    params = _range_params()
    off_grid = (params.height, params.width)

    # Find a seed with exactly 1 active predator
    found = False
    for seed in range(500):
        k = jax.random.PRNGKey(seed)
        state = jax_reset(params, k)
        active_count = int(jnp.sum(state.animal_active))
        if active_count == 1:
            found = True
            break
    if not found:
        pytest.skip("No seed in 500 gave exactly K=1 active predator")

    # Identify active and inactive slot indices
    active_np = np.array(state.animal_active)
    active_idx = int(np.where(active_np)[0][0])
    inactive_idx = int(np.where(~active_np)[0][0])

    # Place the active predator adjacent to the agent (at [5, 6]), in hunt
    # state with max stamina so it immediately chases
    agent_pos = np.array(state.agent_pos)  # (5, 5) from fixed start_pos
    adjacent = jnp.array([agent_pos[0], agent_pos[1] + 1], dtype=jnp.int32)

    new_animal_pos = state.animal_pos.at[active_idx].set(adjacent)
    new_animal_state = state.animal_state.at[active_idx].set(1)     # HUNT
    new_animal_stamina = state.animal_stamina.at[active_idx].set(30.0)  # full
    new_animal_mt = state.animal_move_timer.at[active_idx].set(0)    # ready
    new_animal_at = state.animal_attack_timer.at[active_idx].set(0)  # no delay
    state = state.replace(
        animal_pos=new_animal_pos,
        animal_state=new_animal_state,
        animal_stamina=new_animal_stamina,
        animal_move_timer=new_animal_mt,
        animal_attack_timer=new_animal_at,
    )

    initial_injury = float(state.injury_level)
    injury_rose = False

    for step_i in range(20):
        state, _reward, done, _info = jax_step(state, 4, params)  # agent rests

        # Guard: inactive slot must stay off-grid
        inact_pos = tuple(int(x) for x in np.array(state.animal_pos[inactive_idx]))
        assert inact_pos == off_grid, (
            f"Step {step_i + 1}: inactive slot moved to {inact_pos}, expected {off_grid} "
            f"(over-masking check within active-predator test)"
        )

        if float(state.injury_level) > initial_injury:
            injury_rose = True
            break
        if done:
            break

    assert injury_rose, (
        f"Active predator (slot {active_idx}) placed adjacent to agent never dealt "
        f"damage in 20 steps. Final injury={float(state.injury_level)}, "
        f"initial={initial_injury}. The fix may be over-masking active slots."
    )


# ---------------------------------------------------------------------------
# Test 3 — Legacy all-active parity: fix is a no-op for fixed-count configs
# ---------------------------------------------------------------------------

def test_allactive_config_no_offgrid_parking():
    """For fixed-count (all-active) configs, the fix must be a strict no-op.

    A config using 'count: N' has all slots active (animal_active all-True).
    jnp.where(True, new_pos, off_grid) == new_pos so no slot should ever end
    up at (height, width) during normal stepping (the predator starts in-bounds
    and stays in-bounds due to grid boundary clipping).

    Uses configs/environment/experiment/basic_curriculum/04-far_sight_predator_10x10.yaml
    (the basic-ladder re-level b093023 archived the old basic/ copy; the curriculum
    copy is identical) — all entity groups use fixed 'count: N', no count-range feature.
    """
    from src.environment.config_loader import _resolve_extends

    config_path = os.path.join(
        _ROOT, "configs", "environment", "experiment", "basic_curriculum",
        "04-far_sight_predator_10x10.yaml"
    )
    config = _resolve_extends(config_path, frozenset())
    params = load_env_params(config)

    # All animals must be active for a fixed-count config
    state = jax_reset(params, jax.random.PRNGKey(0))
    assert jnp.all(state.animal_active), (
        "Expected all animal slots active for a fixed-count config"
    )

    off_grid = (params.height, params.width)

    # Step 20 times — no slot should reach (height, width)
    for step_i in range(20):
        state, _reward, done, _info = jax_step(state, 4, params)
        for i in range(state.animal_pos.shape[0]):
            pos = tuple(int(x) for x in np.array(state.animal_pos[i]))
            assert pos != off_grid, (
                f"Step {step_i + 1}: animal slot {i} ended up at off-grid sentinel "
                f"{off_grid} in a fixed-count (all-active) config. The fix incorrectly "
                f"re-parked an active slot."
            )
        if done:
            break
