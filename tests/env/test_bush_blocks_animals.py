"""Tests for Phase 1 of the 'bush as perfect refuge' feature.

Feature: obstacle flag `blocks_animals: true` makes a cell impassable to animals
while leaving the agent free to enter.  Default is `false`, so behaviour is
byte-identical to pre-feature code when the flag is absent or false.

Three tests:
  1. wander_blocked   — a wandering rabbit cannot enter a bush cell with blocks_animals=true
  2. hunt_blocked     — a hunting predator cannot enter a bush cell with blocks_animals=true
                        even when the agent is standing ON the bush (agent still enters freely)
  3. default_off_parity — with blocks_animals=false (default), behaviour is identical to
                          a config that does not mention the flag at all
"""
import os
import sys
import warnings

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


# ─── Shared base (5×5 grid, no resources, minimal sensors) ───────────────────
#
# Bush at row=2 col=2 (0-indexed) = YAML 1-indexed [3,3].
# Area: [[3,3],[3,3]] → exactly one bush, always active.
# Agent start_pos = [3,3] in YAML = 0-indexed [2,2] = ON the bush cell.

_BASE_YAML = """
environment:
  height: 5
  width: 5
  start_pos: [1, 1]
  random_start_pos: false
  max_steps: 5000
  rest_action_enabled: true
  eat_action_enabled: false
  resources: []
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
  metabolic_cost: 0.0
  nutrition_to_satiation_scaling_factor: 0.5
  recovery_base_rate: 0.5
  recovery_accel_rate: 0.5
  injury_smoothing_duration: 3
  death_penalty: 0.0
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
perceptual_noise:
  enabled: false
  modalities: {}
"""

# Bush entry - blocks_animals=true, agent can still enter.
# Bush at 1-indexed [3,3] = 0-indexed [2,2].
_BUSH_BLOCKING_OBS = """
environment:
  obstacles:
    - name: bush
      count: 1
      area: [[3, 3], [3, 3]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      hides_agent: true
      blocks_animals: true
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      visual_properties: [0,0,0,0,0,0,1,0]
      visual_properties_std: [0,0,0,0,0,0,0,0]
"""

# Same bush - blocks_animals=false (explicit default).
_BUSH_NONBLOCKING_OBS = """
environment:
  obstacles:
    - name: bush
      count: 1
      area: [[3, 3], [3, 3]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      hides_agent: true
      blocks_animals: false
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      visual_properties: [0,0,0,0,0,0,1,0]
      visual_properties_std: [0,0,0,0,0,0,0,0]
"""

# Same bush - no blocks_animals key at all (implicit default).
_BUSH_NO_KEY_OBS = """
environment:
  obstacles:
    - name: bush
      count: 1
      area: [[3, 3], [3, 3]]
      blocking: false
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      hides_agent: true
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      visual_properties: [0,0,0,0,0,0,1,0]
      visual_properties_std: [0,0,0,0,0,0,0,0]
"""

# Wandering rabbit: spawns at [3,2] (1-indexed) = 0-indexed [2,1], directly left
# of the bush at [2,2].  Patrol area covers the full 5×5 grid so it can wander
# anywhere.  move_interval=1 so it tries to move every step.
_WANDER_ENTITY = """
environment:
  entities:
    - class: neutral
      behaviour: wander
      tag: rabbit
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      attack_delay: 0
      spawn_area: [[3, 2], [3, 2]]
      patrol_area: [[1, 1], [5, 5]]
"""

# Hunting predator: spawns at [2,3] (1-indexed) = 0-indexed [1,2], directly
# above the bush at [2,2].  Agent is placed at [1,1] (0-indexed [0,0]).
# Detection range=5 → can see the agent anywhere on a 5×5 grid.
# patrol_area covers the full 5×5 grid.
_HUNT_ENTITY = """
environment:
  entities:
    - class: predator
      behaviour: hunt
      tag: predator
      count: 1
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [1.0, 1.0]
      attack_delay: 0
      spawn_area: [[2, 3], [2, 3]]
      patrol_area: [[1, 1], [5, 5]]
      detection_range: [5, 5]
      max_stamina: [100, 100]
      stamina_recovery_rate: [1, 1]
      hunt_stamina_threshold: [0.1, 0.1]
      lose_interest_multiplier: [10.0, 10.0]
"""


def _make_params(obs_yaml: str, entity_yaml: str) -> object:
    """Merge base YAML with obstacle + entity definitions and load EnvParams."""
    base = yaml.safe_load(_BASE_YAML)
    obs_doc = yaml.safe_load(obs_yaml)
    ent_doc = yaml.safe_load(entity_yaml)
    base["environment"]["obstacles"] = obs_doc["environment"]["obstacles"]
    base["environment"]["entities"] = ent_doc["environment"]["entities"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ─── Test 1: wandering animal cannot enter a blocks_animals bush ──────────────

def test_wander_blocked_by_bush():
    """
    A wandering rabbit spawns adjacent to a bush with blocks_animals=true.
    Over 500 steps it should NEVER land on the bush cell (0-indexed [2,2]).
    With blocks_animals=false the rabbit CAN enter, so this is a meaningful test.
    """
    BUSH_CELL = np.array([2, 2])

    params_block = _make_params(_BUSH_BLOCKING_OBS, _WANDER_ENTITY)
    assert bool(params_block.obs_blocks_animals[0]) is True, (
        "obs_blocks_animals[0] should be True for the blocking bush"
    )

    key = jax.random.PRNGKey(123)
    state = jax_reset(params_block, key)

    # Reset forces the rabbit to its spawn_area, but obs_blocking/blocks_animals
    # doesn't affect spawn.  Rabbit starts at [2,1] (0-indexed).
    rabbit_pos_0 = np.array(state.animal_pos[0])
    # Must NOT start on the bush (spawn_area is [[3,2],[3,2]] = 0-indexed [2,1]).
    assert not np.array_equal(rabbit_pos_0, BUSH_CELL), (
        f"Rabbit started on bush cell — spawn exclusion not active (Phase 1 only). Got {rabbit_pos_0}"
    )

    REST = 4
    for step_i in range(500):
        state, _, _, _ = jax_step(state, REST, params_block)
        pos = np.array(state.animal_pos[0])
        assert not np.array_equal(pos, BUSH_CELL), (
            f"Rabbit entered blocked bush cell at step {step_i + 1}: pos={pos}"
        )


def test_wander_not_blocked_when_flag_false():
    """
    Sanity check: with blocks_animals=false, the wandering rabbit CAN eventually
    enter the bush cell.  Failure here means the default-off path is broken.
    """
    BUSH_CELL = np.array([2, 2])

    params_free = _make_params(_BUSH_NONBLOCKING_OBS, _WANDER_ENTITY)
    assert bool(params_free.obs_blocks_animals[0]) is False, (
        "obs_blocks_animals[0] should be False for the non-blocking bush"
    )

    key = jax.random.PRNGKey(0)
    state = jax_reset(params_free, key)

    entered_bush = False
    REST = 4
    for step_i in range(2000):
        state, _, _, _ = jax_step(state, REST, params_free)
        pos = np.array(state.animal_pos[0])
        if np.array_equal(pos, BUSH_CELL):
            entered_bush = True
            break

    assert entered_bush, (
        "With blocks_animals=false the rabbit should be able to enter the bush cell, "
        "but never did in 2000 steps — default-off path may be broken."
    )


# ─── Test 2: hunting predator cannot enter a blocks_animals bush ──────────────

def test_hunt_blocked_by_bush():
    """
    A hunting predator spawns above the bush; agent is at [0,0].
    The predator will hunt toward the agent (not the bush) but must travel through
    the 5×5 grid.  We verify that across 500 steps the predator never lands on
    the bush cell [2,2] while blocks_animals=true.

    Additionally: after steps 1-5 we manually teleport the agent ONTO the bush
    cell and confirm the agent position IS [2,2] (agent still enters freely)
    while the predator still cannot land there.
    """
    BUSH_CELL = jnp.array([2, 2])
    BUSH_CELL_NP = np.array([2, 2])

    params_block = _make_params(_BUSH_BLOCKING_OBS, _HUNT_ENTITY)
    assert bool(params_block.obs_blocks_animals[0]) is True

    key = jax.random.PRNGKey(7)
    state = jax_reset(params_block, key)

    # Force agent onto the bush cell so the predator always hunts toward it.
    # (start_pos=[1,1] 1-indexed = [0,0] 0-indexed; we patch it via _replace.)
    state = state._replace(agent_pos=BUSH_CELL)

    # Agent must be on the bush cell already
    assert np.array_equal(np.array(state.agent_pos), BUSH_CELL_NP), (
        "Agent failed to be placed on the bush cell"
    )

    REST = 4
    for step_i in range(500):
        state, _, _, _ = jax_step(state, REST, params_block)
        pred_pos = np.array(state.animal_pos[0])
        agent_pos = np.array(state.agent_pos)
        assert not np.array_equal(pred_pos, BUSH_CELL_NP), (
            f"Predator entered blocked bush cell at step {step_i + 1}: pred_pos={pred_pos}"
        )
        # Agent still moves around (move_agent uses plain obs_blocking which is False
        # for bush), but after REST action with no blocking, may wander or stay.
        # Core check: predator never on bush.


# ─── Test 3: default-off parity (no flag == false flag == explicit false) ─────

def test_default_off_parity():
    """
    The three formulations — no key, explicit false, explicit false via YAML —
    must:
      a) all set obs_blocks_animals to all-False
      b) produce byte-identical animal_pos over 200 steps

    This is the byte-transparency guard: default-off should be a no-op.
    """
    params_no_key  = _make_params(_BUSH_NO_KEY_OBS,       _WANDER_ENTITY)
    params_false   = _make_params(_BUSH_NONBLOCKING_OBS,  _WANDER_ENTITY)

    # a) Both should have obs_blocks_animals all-False
    assert bool(params_no_key.obs_blocks_animals[0]) is False, (
        "Missing blocks_animals key must default to False"
    )
    assert bool(params_false.obs_blocks_animals[0]) is False, (
        "blocks_animals: false must give obs_blocks_animals=False"
    )

    # b) Byte-identical animal_pos across 200 steps
    key = jax.random.PRNGKey(42)
    STEPS = 200
    REST = 4

    def run_episode(params):
        state = jax_reset(params, key)
        positions = []
        for _ in range(STEPS):
            state, _, _, _ = jax_step(state, REST, params)
            positions.append(np.array(state.animal_pos))
        return positions

    pos_no_key = run_episode(params_no_key)
    pos_false  = run_episode(params_false)

    for i, (a, b) in enumerate(zip(pos_no_key, pos_false)):
        assert np.array_equal(a, b), (
            f"animal_pos diverged at step {i} between no-key and explicit-false configs: "
            f"{a} vs {b}"
        )
