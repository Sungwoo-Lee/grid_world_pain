"""Tests for the disengage_on_contact feature (strike-and-retreat for hunting animals).

Three tests per the plan:
  1. full_cycle   — contact → stamina 0 → HUNT→RETURN → recover → re-engage
  2. byte_parity  — no-flag config produces all-False params.animal_disengage_on_contact
                    and two identical 100-step rollouts agree
  3. vmap_mixed   — mixed entity config, flag per animal, vmap over envs
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

# ─── Shared YAML base (minimal env, no sensors/resources) ─────────────────────
# Agent is at start_pos [1,1] (YAML, 1-indexed) = 0-indexed [0,0].
# The spawn_area [[1,1],[1,1]] puts the animal on the same cell as the agent.
# The patrol_area [[1,1],[10,10]] covers the full 10x10 grid → stored [0,0,10,10]
# → patrol center at (5,5), which is 10 steps (Manhattan) from the agent at [0,0].
# So after HUNT→RETURN the animal will NOT immediately satisfy dist_to_center<=2.

_BASE_YAML = """
environment:
  height: 10
  width: 10
  start_pos: [1, 1]
  random_start_pos: false
  max_steps: 5000
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

# Hunt animal with disengage_on_contact=True.
# Spawns at [1,1] (same cell as agent).  Patrol area covers full grid so it
# can chase the agent anywhere.  Patrol center = (5,5) in 0-indexed, far enough
# from the agent at (0,0) so reentered_home (dist<=2) is False right after
# HUNT→RETURN.
_HUNT_DOC_YAML = """
environment:
  entities:
    - class: neutral
      behaviour: hunt
      tag: chaser
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      attack_delay: 0
      disengage_on_contact: true
      spawn_area: [[1, 1], [1, 1]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: [8, 8]
      max_stamina: [50, 50]
      stamina_recovery_rate: [2, 2]
      hunt_stamina_threshold: [0.5, 0.5]
      lose_interest_multiplier: [2.0, 2.0]
"""

# Same animal WITHOUT the flag (default False → byte-parity test)
_NO_FLAG_DOC_YAML = """
environment:
  entities:
    - class: neutral
      behaviour: hunt
      tag: chaser
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      attack_delay: 0
      spawn_area: [[1, 1], [1, 1]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: [8, 8]
      max_stamina: [50, 50]
      stamina_recovery_rate: [2, 2]
      hunt_stamina_threshold: [0.5, 0.5]
      lose_interest_multiplier: [2.0, 2.0]
"""


def _make_params(extra_entity_yaml: str) -> object:
    """Merge base YAML with entity definition and load EnvParams."""
    base = yaml.safe_load(_BASE_YAML)
    extra = yaml.safe_load(extra_entity_yaml)
    base["environment"]["entities"] = extra["environment"]["entities"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ─── Test 1 — full strike-and-retreat cycle ───────────────────────────────────

def test_full_cycle():
    """
    Animal starts ON the agent's cell (both at 0-indexed [0,0]).
    Patrol center is at [5,5] (far from agent).  The full cycle:

      Step 1:  agent=[0,0] == animal=[0,0] → at_animal=True → override:
               stamina → 0.  Animal entered HUNT this step (rested+in range).
      Step 2:  _hunt_step receives stamina=0 → lose_interest → HUNT→RETURN.
               dist_to_center([0,0],[5,5]) = 10 > 2 → NOT reentered_home.
      Steps 3+: animal retreats toward [5,5], stamina recovers (+2/step).
               Once stamina >= 25 (0.5 * 50) AND agent in detect range (8),
               become_hunt fires → state back to HUNT.

    State codes: 0=PATROL, 1=HUNT, 2=RETURN.
    """
    params = _make_params(_HUNT_DOC_YAML)

    # Verify flag is set
    assert params.animal_disengage_on_contact.shape == (1,)
    assert bool(params.animal_disengage_on_contact[0]) is True

    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)

    REST = 4
    HUNT_STATE = 1
    RETURN_STATE = 2

    # Step 1: animal and agent are both at [0,0] → contact on first step
    state_after_1, _, _, _ = jax_step(state, REST, params)

    # Override must have fired: stamina → 0
    assert float(state_after_1.animal_stamina[0]) == 0.0, (
        f"Stamina must be 0 after contact, got {state_after_1.animal_stamina[0]}"
    )

    # Step 2: lose_interest fires (stamina=0) → HUNT→RETURN
    state_after_2, _, _, _ = jax_step(state_after_1, REST, params)

    animal_state_2 = int(state_after_2.animal_state[0])
    assert animal_state_2 == RETURN_STATE, (
        f"Expected RETURN (2) after disengage on step 2, got {animal_state_2}"
    )
    # Stamina should be recovering already (was 0, +2/step while in RETURN)
    assert float(state_after_2.animal_stamina[0]) > 0.0, (
        f"Stamina must be recovering after disengagement, got {state_after_2.animal_stamina[0]}"
    )

    # Steps 3..N: retreat → recover → re-engage HUNT
    hunt_state = state_after_2
    seen_recovery = False
    seen_re_hunt = False

    for step_j in range(400):
        hunt_state, _, _, _ = jax_step(hunt_state, REST, params)
        st = int(hunt_state.animal_state[0])
        stamina = float(hunt_state.animal_stamina[0])

        if st != HUNT_STATE and stamina > 5.0:
            seen_recovery = True
        if st == HUNT_STATE and seen_recovery:
            seen_re_hunt = True
            break

    assert seen_re_hunt, (
        "Animal never completed the full strike-and-retreat cycle "
        f"(seen_recovery={seen_recovery}, seen_re_hunt={seen_re_hunt})"
    )


# ─── Test 2 — byte-parity for non-opted-in configs ────────────────────────────

def test_byte_parity_no_flag():
    """
    A config without the disengage_on_contact flag must:
      a) have params.animal_disengage_on_contact all-False
      b) produce byte-identical animal_stamina over 100 steps compared to
         itself (two identical runs → same result).
    The load-bearing backward-compat proof is test_unified_parity.py passing
    for all 86 existing configs.
    """
    params_no = _make_params(_NO_FLAG_DOC_YAML)
    assert params_no.animal_disengage_on_contact.shape == (1,)
    assert bool(params_no.animal_disengage_on_contact[0]) is False, (
        "Missing disengage_on_contact flag must default to False"
    )

    # Two identical runs from same seed must produce identical stamina arrays
    key = jax.random.PRNGKey(42)
    ACTIONS = ([0, 1, 2, 3, 4] * 20)  # 100 steps

    def run_episode(params):
        state = jax_reset(params, key)
        staminas = []
        for a in ACTIONS:
            state, _, _, _ = jax_step(state, a, params)
            staminas.append(np.array(state.animal_stamina))
        return staminas

    staminas_a = run_episode(params_no)
    staminas_b = run_episode(params_no)
    for i, (sa, sb) in enumerate(zip(staminas_a, staminas_b)):
        assert np.array_equal(sa, sb), f"Stamina diverged at step {i}"


# ─── Test 3 — vmap + mixed entity config ──────────────────────────────────────

_MIXED_DOC_YAML = """
environment:
  entities:
    - class: neutral
      behaviour: hunt
      tag: chaser_flagged
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      attack_delay: 0
      disengage_on_contact: true
      spawn_area: [[3, 3], [4, 4]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: [8, 8]
      max_stamina: [50, 50]
      stamina_recovery_rate: [2, 2]
      hunt_stamina_threshold: [0.5, 0.5]
      lose_interest_multiplier: [2.0, 2.0]
    - class: neutral
      behaviour: wander
      tag: wanderer
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      attack_delay: 0
      spawn_area: [[8, 8], [9, 9]]
      patrol_area: [[7, 7], [10, 10]]
    - class: predator
      behaviour: hunt
      tag: pred_no_flag
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 2
      nociception_intensity: 0.9
      damage: [5.0, 5.0]
      attack_delay: 2
      spawn_area: [[1, 1], [2, 2]]
      patrol_area: [[1, 1], [10, 10]]
      detection_range: [8, 8]
      max_stamina: [50, 50]
      stamina_recovery_rate: [2, 2]
      hunt_stamina_threshold: [0.5, 0.5]
      lose_interest_multiplier: [2.0, 2.0]
"""


def test_vmap_mixed_entities():
    """
    Mixed config: [hunt+flagged, wander, hunt+no_flag].
    Checks:
      1. animal_disengage_on_contact shape is (3,) with [True, False, False]
      2. hunt_idx=(0, 2), wander_idx=(1,) — flag is on full unified array not
         just the hunt subset
      3. jax.vmap(jax_step) runs without shape errors over a batch of 4 envs
    """
    params = _make_params(_MIXED_DOC_YAML)

    # Check shape and per-entry flags
    assert params.animal_disengage_on_contact.shape == (3,), (
        f"Expected shape (3,), got {params.animal_disengage_on_contact.shape}"
    )
    flags = [bool(params.animal_disengage_on_contact[i]) for i in range(3)]
    assert flags == [True, False, False], (
        f"Expected [True, False, False], got {flags}"
    )

    # hunt_idx should be (0, 2); wander_idx should be (1,)
    assert params.hunt_idx == (0, 2), f"Expected hunt_idx=(0,2), got {params.hunt_idx}"
    assert params.wander_idx == (1,), f"Expected wander_idx=(1,), got {params.wander_idx}"

    # Wander animal (index 1) has disengage=False — override is a no-op for wander
    assert not bool(params.animal_disengage_on_contact[1])
    # Predator (index 2) has disengage=False
    assert not bool(params.animal_disengage_on_contact[2])

    # vmap over a batch of 4 environments
    BATCH = 4
    keys = jax.random.split(jax.random.PRNGKey(7), BATCH)
    states = jax.vmap(jax_reset, in_axes=(None, 0))(params, keys)

    # Step all 4 envs in parallel
    actions = jnp.zeros(BATCH, dtype=jnp.int32)  # action 0 for all

    def step_one(state, action):
        return jax_step(state, action, params)

    batch_states, rewards, dones, infos = jax.vmap(step_one)(states, actions)

    # Shapes should broadcast correctly
    assert batch_states.animal_stamina.shape == (BATCH, 3), (
        f"Expected stamina shape (4, 3), got {batch_states.animal_stamina.shape}"
    )
    assert batch_states.animal_state.shape == (BATCH, 3), (
        f"Expected state shape (4, 3), got {batch_states.animal_state.shape}"
    )
    # No NaN stamina values
    assert not jnp.any(jnp.isnan(batch_states.animal_stamina)), (
        "NaN in animal_stamina after vmap step"
    )
