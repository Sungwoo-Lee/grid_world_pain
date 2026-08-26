"""Tests for the predator jump/pounce feature (attack_range + attack_success_rate).

Plan: docs/develop/active/env_entities/PREDATOR_JUMP_MECHANISM.md

Checklist (a)-(h) from the plan's Test Plan section:
  (a) Fires only when all conditions hold -> predator lands on agent, damage applied.
  (b) Success -> on-agent hit, no double-count.
  (c) Miss -> adjacent free cell, no damage.
  (d) Cooldown blocks the next-step jump.
  (e) Bushed agent is never jumped.
  (f) Disabled (attack_range absent/[0,0]) = no jump + parity.
  (g) Recompile-safety.
  (h) vmap + mixed entities.

Geometry used throughout (a)-(e): agent fixed at 0-indexed [0,0] (corner, REST
action every step so it never moves). Predator spawns at a fixed cell so all
draws are deterministic given a fixed PRNG seed.
"""
import io
import logging
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

REST = 4
PATROL, HUNT, RETURN = 0, 1, 2

# ─── Shared YAML base (minimal env, no sensors/resources) ─────────────────────
# Agent is at start_pos [1,1] (YAML, 1-indexed) = 0-indexed [0,0], a grid corner
# (so the miss-neighbour candidate set is small and deterministic: only
# (0,1), (1,0), (1,1) are in-bounds).
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


def _pred_entry(tag: str, attack_range=None, attack_success_rate=None,
                spawn_area="[[3, 3], [3, 3]]", damage="[10.0, 10.0]",
                attack_delay: int = 3) -> str:
    """A hunt predator, spawned at a fixed cell, distance 4 (Manhattan) from
    the agent's corner [0,0] when spawn_area is the default [[3,3],[3,3]]
    (0-indexed [2,2]).  detection_range=8 and hunt_stamina_threshold=0.1 mean
    it always becomes HUNT on step 1 (full stamina, in range, agent visible).
    """
    extra = ""
    if attack_range is not None:
        extra += f"\n      attack_range: {attack_range}"
    if attack_success_rate is not None:
        extra += f"\n      attack_success_rate: {attack_success_rate}"
    return f"""
    - class: predator
      behaviour: hunt
      tag: {tag}
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: {damage}
      attack_delay: {attack_delay}
      spawn_area: {spawn_area}
      patrol_area: [[1, 1], [10, 10]]
      detection_range: [8, 8]
      max_stamina: [50, 50]
      stamina_recovery_rate: [2, 2]
      hunt_stamina_threshold: [0.1, 0.1]
      lose_interest_multiplier: [3.0, 3.0]{extra}"""


def _wand_entry(tag: str) -> str:
    return f"""
    - class: neutral
      behaviour: wander
      tag: {tag}
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.0
      damage: [0.0, 0.0]
      attack_delay: 0
      spawn_area: [[8, 8], [9, 9]]
      patrol_area: [[7, 7], [10, 10]]"""


_BUSH_OBSTACLE_AT_ORIGIN = """
environment:
  obstacles:
    - name: bush
      count: 1
      area: [[1, 1], [1, 1]]
      blocking: false
      blocks_animals: false
      hides_agent: true
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
      properties: [0.0, 0.0, 0.0, 1.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
"""


def _make_params(entity_yaml_body: str, obstacles_yaml: str = None) -> object:
    """Combine base YAML with an entities: body (and optional obstacles:) and load EnvParams."""
    full_entities_yaml = "environment:\n  entities:" + entity_yaml_body
    base = yaml.safe_load(_BASE_YAML)
    extra = yaml.safe_load(full_entities_yaml)
    base["environment"]["entities"] = extra["environment"]["entities"]
    if obstacles_yaml is not None:
        obs_extra = yaml.safe_load(obstacles_yaml)
        base["environment"]["obstacles"] = obs_extra["environment"]["obstacles"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ─── (a)/(b) — fires only when all conditions hold; success -> hit, no double-count ──

def test_jump_success_lands_on_agent_and_deals_damage_once():
    """attack_range=4 (== Manhattan dist from spawn to agent), attack_success_rate=1.0
    -> the fired jump ALWAYS succeeds -> predator lands exactly on the agent's cell
    on step 1, and info['damage_predator'] equals exactly ONE damage draw (10.0,
    not 20.0 -- confirms no double-counting with the existing on-cell damage block).
    """
    params = _make_params(_pred_entry("p", attack_range=4, attack_success_rate=1.0))
    assert params.has_attack_feature is True

    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)

    # Sanity: sampled attack_range is exactly 4 (degenerate scalar range).
    assert float(state.animal_attack_range_sampled[0]) == 4.0
    # Pre-jump geometry sanity: predator at (2,2), agent at (0,0), dist=4.
    assert list(np.array(state.animal_pos[0])) == [2, 2]
    assert list(np.array(state.agent_pos)) == [0, 0]

    state_after, reward, done, info = jax_step(state, REST, params)

    # Predator jumped onto the agent's cell.
    assert list(np.array(state_after.animal_pos[0])) == [0, 0], (
        f"Expected predator to land on agent [0,0]; got {np.array(state_after.animal_pos[0])}"
    )
    assert bool(info["hit_predator"]) is True
    # Exactly one damage draw (10.0), not doubled.
    assert float(info["damage_predator"]) == 10.0
    assert float(info["damage"]) == 10.0
    # Cooldown set on the attempt (matches attack_delay=3; idempotent with the
    # existing on-cell-hit reset at core.py ~line 562).
    assert int(state_after.animal_attack_timer[0]) == 3


# ─── (c) — miss lands on a valid adjacent cell, never on the agent, no damage ──────

def test_jump_miss_lands_on_valid_neighbour_no_damage():
    """attack_success_rate=0.0 -> the fired jump ALWAYS misses. Agent sits in the
    grid corner [0,0], so only 3 of the 8 Chebyshev neighbours are in-bounds:
    (0,1), (1,0), (1,1). The predator must land on one of these three cells.
    """
    params = _make_params(_pred_entry("p", attack_range=4, attack_success_rate=0.0))

    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)

    state_after, reward, done, info = jax_step(state, REST, params)

    pos = tuple(int(x) for x in np.array(state_after.animal_pos[0]))
    valid_misses = {(0, 1), (1, 0), (1, 1)}
    assert pos in valid_misses, f"Predator landed at {pos}, expected one of {valid_misses}"
    assert pos != (0, 0), "A missed jump must never land on the agent's cell"
    assert bool(info["hit_predator"]) is False
    assert float(info["damage_predator"]) == 0.0
    # Cooldown is still set on ANY attempt (hit or miss).
    assert int(state_after.animal_attack_timer[0]) == 3


# ─── (d) — cooldown blocks the next-step jump ──────────────────────────────────

def test_cooldown_blocks_second_jump():
    """After a fired jump (miss, so the predator is off the agent's cell),
    animal_attack_timer == attack_delay. On the NEXT step should_move is also
    gated on attack_timer<=0, so the predator does not move at all (natural
    post-pounce recovery pause) -- confirming no second jump/move fires.
    """
    params = _make_params(_pred_entry("p", attack_range=4, attack_success_rate=0.0, attack_delay=3))

    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)

    state_1, _, _, _ = jax_step(state, REST, params)
    assert int(state_1.animal_attack_timer[0]) == 3
    pos_after_jump = tuple(int(x) for x in np.array(state_1.animal_pos[0]))
    assert pos_after_jump != (0, 0)

    state_2, _, _, info_2 = jax_step(state_1, REST, params)
    # Frozen: predator does not move (should_move requires attack_timer<=0).
    pos_after_2 = tuple(int(x) for x in np.array(state_2.animal_pos[0]))
    assert pos_after_2 == pos_after_jump, "Predator must stay put during cooldown"
    assert int(state_2.animal_attack_timer[0]) == 2  # decremented, no re-trigger
    assert bool(info_2["hit_predator"]) is False


# ─── (e) — bushed agent is never jumped ────────────────────────────────────────

def test_bushed_agent_never_jumped():
    """Agent's start cell carries a hides_agent obstacle (bush). Even though the
    predator spawns within attack_range and attack_success_rate=1.0, the jump
    (and even HUNT itself, via the existing agent_hidden gate on become_hunt)
    never fires: the predator never lands on the agent.
    """
    params = _make_params(
        _pred_entry("p", attack_range=4, attack_success_rate=1.0),
        obstacles_yaml=_BUSH_OBSTACLE_AT_ORIGIN,
    )

    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)

    for _ in range(10):
        state, _, _, info = jax_step(state, REST, params)
        assert list(np.array(state.animal_pos[0])) != [0, 0], (
            "Predator must never land on a bushed (hidden) agent"
        )
        assert bool(info["hit_predator"]) is False
        # become_hunt is gated on ~agent_hidden -> predator never even enters HUNT.
        assert int(state.animal_state[0]) != HUNT


# ─── (f) — disabled (attack_range absent/[0,0]) = no jump + parity ────────────

def test_disabled_no_jump_and_deterministic():
    """No attack_range/attack_success_rate keys -> has_attack_feature is False,
    the predator only ever advances by <=1 cell (Manhattan) per step (never
    teleports onto the agent from range-4), and two identical seeded rollouts
    agree byte-for-byte (determinism sanity; the load-bearing global parity
    proof is test_unified_parity.py staying green with no fixture re-capture).
    """
    params = _make_params(_pred_entry("p"))  # no attack_range / attack_success_rate
    assert params.has_attack_feature is False
    assert bool(jnp.all(params.animal_attack_range_high == 0.0))
    assert bool(jnp.all(params.animal_attack_success_rate == 0.0))

    key = jax.random.PRNGKey(0)

    def run_episode():
        state = jax_reset(params, key)
        positions = []
        for _ in range(20):
            state, _, _, _ = jax_step(state, REST, params)
            positions.append(np.array(state.animal_pos[0]).copy())
        return positions

    pos_a = run_episode()
    pos_b = run_episode()
    for i, (pa, pb) in enumerate(zip(pos_a, pos_b)):
        assert np.array_equal(pa, pb), f"Predator position diverged at step {i}"

    # Never teleports: consecutive positions differ by at most 1 cell (Chebyshev).
    prev = np.array([2, 2])  # spawn
    for i, p in enumerate(pos_a):
        step_dist = np.max(np.abs(p - prev))
        assert step_dist <= 1, f"Predator teleported at step {i}: {prev} -> {p}"
        prev = p


# ─── (g) — recompile safety ────────────────────────────────────────────────────

class _CompileCounter:
    """Context manager that captures jax._src.interpreters.pxla WARNING logs
    and counts 'Compiling jit(jax_step)' occurrences. (Model: test_no_recompile.py)"""

    def __init__(self):
        self._buf = io.StringIO()
        self._handler = logging.StreamHandler(self._buf)
        self._handler.setLevel(logging.WARNING)
        self._logger = logging.getLogger("jax._src.interpreters.pxla")
        self._original_level = self._logger.level

    def __enter__(self):
        self._buf.truncate(0)
        self._buf.seek(0)
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *args):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._original_level)

    @property
    def count(self) -> int:
        return self._buf.getvalue().count("Compiling jit(jax_step)")

    @property
    def log(self) -> str:
        return self._buf.getvalue()


jax.config.update("jax_log_compiles", True)


def test_recompile_safety_values_and_enable_toggle():
    """Part 1 (negative control): two jump-enabled configs, same N + class
    ordering, different attack_range/attack_success_rate VALUES -> exactly 1
    compile (traced leaves, not static -- B reuses A's trace).

    Part 2 (positive control): a jump-enabled vs jump-disabled pair -> 2
    compiles total (has_attack_feature is a static bool -> distinct traces).
    """
    jax.clear_caches()

    params_A = _make_params(_pred_entry("p", attack_range=4, attack_success_rate=0.5))
    params_B = _make_params(_pred_entry("p", attack_range=6, attack_success_rate=0.9))
    assert params_A.has_attack_feature is True
    assert params_B.has_attack_feature is True
    assert params_A.animal_classes == params_B.animal_classes
    assert params_A.hunt_idx == params_B.hunt_idx

    key = jax.random.PRNGKey(0)

    with _CompileCounter() as counter:
        state_A = jax_reset(params_A, key)
        for _ in range(5):
            state_A, _, _, _ = jax_step(state_A, REST, params_A)
        count_after_A = counter.count
        assert count_after_A == 1, f"Expected 1 compile after config A; got {count_after_A}"

        state_B = jax_reset(params_B, key)
        for _ in range(5):
            state_B, _, _, _ = jax_step(state_B, REST, params_B)
        count_after_B = counter.count
        assert count_after_B == 1, (
            f"Expected still 1 compile after config B (values-only change); got {count_after_B}"
        )

        # Part 2: jump-disabled config -- static has_attack_feature flips -> recompile.
        params_C = _make_params(_pred_entry("p"))  # no attack_range -> disabled
        assert params_C.has_attack_feature is False
        state_C = jax_reset(params_C, key)
        for _ in range(5):
            state_C, _, _, _ = jax_step(state_C, REST, params_C)
        count_after_C = counter.count
        assert count_after_C == 2, (
            f"Expected 2 compiles after the enabled->disabled switch; got {count_after_C}.\n"
            f"Log tail:\n{counter.log[-500:]}"
        )


# ─── (h) — vmap + mixed entities ───────────────────────────────────────────────

_MIXED_YAML = (
    _pred_entry("pred_jump", attack_range=4, attack_success_rate=1.0)
    + _wand_entry("wanderer")
    + _pred_entry("pred_no_jump", spawn_area="[[6, 6], [6, 6]]")
)


def test_vmap_mixed_entities():
    """Config [hunt+jump, wander, hunt+no-jump]: animal_attack_range_high has
    shape (3,), and jax.vmap(jax_step) over a batch of environments runs
    without shape errors and produces no NaNs.
    """
    params = _make_params(_MIXED_YAML)

    assert params.animal_attack_range_high.shape == (3,)
    assert params.hunt_idx == (0, 2)
    assert params.wander_idx == (1,)
    # Per-animal enable flags: pred_jump enabled, wanderer/pred_no_jump not.
    flags = [bool(params.animal_attack_range_high[i] > 0) for i in range(3)]
    assert flags == [True, False, False]

    BATCH = 4
    keys = jax.random.split(jax.random.PRNGKey(7), BATCH)
    states = jax.vmap(jax_reset, in_axes=(None, 0))(params, keys)

    actions = jnp.full((BATCH,), REST, dtype=jnp.int32)

    def step_one(state, action):
        return jax_step(state, action, params)

    batch_states, rewards, dones, infos = jax.vmap(step_one)(states, actions)

    assert batch_states.animal_pos.shape == (BATCH, 3, 2)
    assert not jnp.any(jnp.isnan(batch_states.animal_stamina))
    assert not jnp.any(jnp.isnan(batch_states.animal_attack_range_sampled))


# ─── (extra) — empirical success-rate sanity over many draws ──────────────────

def test_empirical_success_rate_matches_configured_rate():
    """Over many independent seeds, the empirical hit-rate of a fired jump
    should approximate the configured attack_success_rate (0.5)."""
    params = _make_params(_pred_entry("p", attack_range=4, attack_success_rate=0.5))

    n_trials = 300
    hits = 0
    for seed in range(n_trials):
        key = jax.random.PRNGKey(seed)
        state = jax_reset(params, key)
        state_after, _, _, info = jax_step(state, REST, params)
        if bool(info["hit_predator"]):
            hits += 1

    rate = hits / n_trials
    assert 0.35 < rate < 0.65, f"Empirical hit rate {rate:.3f} too far from configured 0.5"
