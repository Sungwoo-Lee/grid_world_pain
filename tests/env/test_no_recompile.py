"""CP5 — JIT no-recompile tests.

Verifies that changing only the distributional bounds (detection_range, etc.)
for an env with the same animal count and class ordering does NOT trigger a
JAX JIT recompile of jax_step.

Conversely, changing the per-entity class ordering (which alters the static
hunt_idx / wander_idx pytree_node=False tuples) DOES trigger a recompile.

This guards the plan's JIT-shape-stability claim (§"JIT recompile" and
§"Test Plan §(e) — JIT-recompile test").

Implementation:
  Uses jax.config.update("jax_log_compiles", True) to surface compile events
  through the jax._src.interpreters.pxla WARNING logger, then counts
  occurrences of "Compiling jit(jax_step)" in the captured log.

  Between tests, jax.clear_caches() ensures clean JIT state so counts are
  exactly 1 (not 0 due to a previous run's cached trace).

Two parts:
  Part 1 — negative control: same N + same class ordering, different bounds
            → exactly 1 compile (config A only; config B reuses A's trace).
  Part 2 — positive control: same N but different class ordering
            → exactly 2 compiles (one per unique class-ordering trace).

Plan ref: §"Test Plan §(e)" (CP5).
"""
import io
import logging
import os
import sys
import warnings

import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_step, jax_reset

# ── Enable JAX compile-event logging (module-level) ──────────────────────────
# Setting this once at import time is sufficient; it persists across tests.
jax.config.update("jax_log_compiles", True)


# ── Shared YAML base ──────────────────────────────────────────────────────────

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

# ── Entity YAML building blocks ───────────────────────────────────────────────

def _pred_entry(tag: str, detect_range: str = "[0, 5]") -> str:
    return f"""
    - class: predator
      behaviour: hunt
      tag: {tag}
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.9
      damage: [10.0, 20.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 2
      detection_range: {detect_range}
      max_stamina: [20, 40]
      stamina_recovery_rate: [0.5, 1.5]
      hunt_stamina_threshold: [0.5, 0.9]
      lose_interest_multiplier: 1.5"""


def _wand_entry(tag: str) -> str:
    return f"""
    - class: neutral
      behaviour: wander
      tag: {tag}
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.1
      damage: [0.0, 0.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 0"""


def _stat_entry(tag: str) -> str:
    return f"""
    - class: neutral
      behaviour: static
      tag: {tag}
      count: 1
      properties: [0.0, 1.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      move_interval: 1
      nociception_intensity: 0.1
      damage: [0.0, 0.0]
      spawn_area: [[1, 1], [9, 9]]
      patrol_area: [[1, 1], [9, 9]]
      attack_delay: 0"""


def _make_params(entity_yaml_body: str):
    """Combine base YAML with entity body and load EnvParams."""
    full_entities_yaml = "environment:\n  entities:" + entity_yaml_body
    base = yaml.safe_load(_BASE_YAML)
    extra = yaml.safe_load(full_entities_yaml)
    base["environment"]["entities"] = extra["environment"]["entities"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_env_params(Config(base))


# ── Log-capture context manager ───────────────────────────────────────────────

class _CompileCounter:
    """Context manager that captures jax._src.interpreters.pxla WARNING logs
    and counts 'Compiling jit(jax_step)' occurrences."""

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
        """Number of 'Compiling jit(jax_step)' messages captured so far."""
        return self._buf.getvalue().count("Compiling jit(jax_step)")

    @property
    def log(self) -> str:
        """Full captured log text (for debugging)."""
        return self._buf.getvalue()


# ── Part 1: negative control — bounds-only change, no recompile ───────────────

class TestNegativeControl:
    """Same N + same class ordering, different distributional bounds → 1 compile."""

    def test_bounds_change_no_recompile(self):
        """Config A: [0,5] bounds. Config B: [2,7] bounds. Same 1 pred + 2 rabbits.

        Exactly 1 compile expected: A triggers it; B reuses A's cached trace
        because hunt_idx / wander_idx / animal_classes are identical.

        Plan §(e) Part 1 — negative control.
        """
        # Clear the JIT cache so the compile count starts from scratch.
        jax.clear_caches()

        params_A = _make_params(
            _pred_entry("p", "[0, 5]") + _wand_entry("r1") + _wand_entry("r2")
        )
        params_B = _make_params(
            _pred_entry("p", "[2, 7]") + _wand_entry("r1") + _wand_entry("r2")
        )

        # Sanity: both configs have the same static keys.
        assert params_A.animal_classes == params_B.animal_classes, (
            "Test setup error: class tuples must be identical for the negative control."
        )
        assert params_A.hunt_idx == params_B.hunt_idx
        assert params_A.wander_idx == params_B.wander_idx

        key = jax.random.PRNGKey(0)

        with _CompileCounter() as counter:
            # Run config A — should compile once.
            state_A = jax_reset(params_A, key)
            for _ in range(10):
                state_A, _, _, _ = jax_step(state_A, 0, params_A)

            count_after_A = counter.count
            assert count_after_A == 1, (
                f"Expected exactly 1 compile after config A; got {count_after_A}.\n"
                f"Log tail:\n{counter.log[-500:]}"
            )

            # Run config B (same layout, different bounds) — must NOT recompile.
            state_B = jax_reset(params_B, key)
            for _ in range(10):
                state_B, _, _, _ = jax_step(state_B, 0, params_B)

            count_after_B = counter.count
            assert count_after_B == 1, (
                f"Expected still exactly 1 compile after config B (no recompile); "
                f"got {count_after_B}.\n"
                f"Log tail:\n{counter.log[-500:]}"
            )


# ── Part 2: positive control — class-ordering change triggers recompile ────────

class TestPositiveControl:
    """Same N but different class ordering → 2 compiles (one per distinct trace)."""

    def test_class_ordering_swap_triggers_recompile(self):
        """Config C: [pred, neutral, pred]. Config D: [pred, pred, neutral].

        Same N=3 but hunt_idx differs: C has hunt_idx=(0,2), D has hunt_idx=(0,1).
        Because animal_classes is pytree_node=False, JAX must re-trace for D.
        Exactly 2 compiles expected.

        Plan §(e) Part 2 — positive control.
        """
        # Clear the JIT cache so the compile count starts from scratch.
        jax.clear_caches()

        # Config C: pred / neutral / pred → hunt_idx=(0,2), wander_idx=(1,)
        params_C = _make_params(
            _pred_entry("p0") + _wand_entry("r0") + _pred_entry("p1")
        )
        # Config D: pred / pred / neutral → hunt_idx=(0,1), wander_idx=(2,)
        params_D = _make_params(
            _pred_entry("p0") + _pred_entry("p1") + _wand_entry("r0")
        )

        # Sanity: same N, but different class orderings.
        assert len(params_C.animal_classes) == len(params_D.animal_classes) == 3
        assert params_C.animal_classes != params_D.animal_classes, (
            "Test setup error: class tuples must differ for the positive control."
        )
        assert params_C.hunt_idx != params_D.hunt_idx, (
            "Test setup error: hunt_idx must differ for the positive control."
        )

        key = jax.random.PRNGKey(0)

        with _CompileCounter() as counter:
            # Run config C — should compile once.
            state_C = jax_reset(params_C, key)
            for _ in range(10):
                state_C, _, _, _ = jax_step(state_C, 0, params_C)

            count_after_C = counter.count
            assert count_after_C == 1, (
                f"Expected exactly 1 compile after config C; got {count_after_C}."
            )

            # Run config D (different class ordering) — MUST recompile.
            state_D = jax_reset(params_D, key)
            for _ in range(10):
                state_D, _, _, _ = jax_step(state_D, 0, params_D)

            count_after_D = counter.count
            assert count_after_D == 2, (
                f"Expected exactly 2 compiles after config D (recompile expected); "
                f"got {count_after_D}.\n"
                f"Log tail:\n{counter.log[-500:]}"
            )


# ── Part 3: same animal_classes but different animal_behaviours → recompile ───

class TestBehaviourChangeTriggersRecompile:
    """Same N + same class tuple but different behaviours → 2 compiles.

    wander_idx / static_idx are pytree_node=False tuples.  Changing a neutral
    entity from wander to static changes wander_idx=(0,) → wander_idx=() and
    static_idx=() → static_idx=(0,), so JAX must re-trace.

    N-CP5-3: reviewer-verified edge case; this test documents and guards it.
    Plan ref: §"Test Plan §(e)" (CP5 / N-CP5-3).
    """

    def test_behaviour_change_triggers_recompile(self):
        """Config E: 1 neutral (wander). Config F: 1 neutral (static).

        Same animal_classes=('neutral',) but wander_idx vs static_idx differ.
        Exactly 2 compiles expected.
        """
        jax.clear_caches()

        params_E = _make_params(_wand_entry("r0"))   # wander_idx=(0,), static_idx=()
        params_F = _make_params(_stat_entry("r0"))   # wander_idx=(),   static_idx=(0,)

        # Sanity: same class tuple, different behaviour tuples.
        assert params_E.animal_classes == params_F.animal_classes, (
            "Test setup error: class tuples must be identical."
        )
        assert params_E.animal_behaviours != params_F.animal_behaviours, (
            "Test setup error: behaviour tuples must differ (wander vs static)."
        )
        assert params_E.wander_idx != params_F.wander_idx, (
            "Test setup error: wander_idx must differ."
        )

        key = jax.random.PRNGKey(0)

        with _CompileCounter() as counter:
            # Config E — wander neutral — should compile once.
            state_E = jax_reset(params_E, key)
            for _ in range(5):
                state_E, _, _, _ = jax_step(state_E, 0, params_E)

            count_after_E = counter.count
            assert count_after_E == 1, (
                f"Expected exactly 1 compile after config E; got {count_after_E}.\n"
                f"Log tail:\n{counter.log[-500:]}"
            )

            # Config F — static neutral — MUST recompile (different wander_idx/static_idx).
            state_F = jax_reset(params_F, key)
            for _ in range(5):
                state_F, _, _, _ = jax_step(state_F, 0, params_F)

            count_after_F = counter.count
            assert count_after_F == 2, (
                f"Expected exactly 2 compiles after config F (recompile expected); "
                f"got {count_after_F}.\n"
                f"Log tail:\n{counter.log[-500:]}"
            )
