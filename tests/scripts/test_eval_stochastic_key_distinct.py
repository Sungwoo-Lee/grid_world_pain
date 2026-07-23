"""Regression test — Bug 4 (Track B, 2026-07-23 eval/logging fix plan, latent):
`_run_episode` (and `_run_episode_with_recording`) in scripts/eval/
eval_rollout.py passed the same frozen episode-level `rng_key` into
`policy_fn` on every step, so stochastic eval sampling (`eval_policy_mode:
"stochastic"`) drew every step's action with an identical key -- a
perfectly-correlated, near-deterministic "stochastic" eval.

Fix (docs/develop/active/issues/FIX_EVAL_LOGGING_TRACK_B_20260723.md, Bug 4):
split a fresh per-step key from a running `step_key` before each call to
`policy_fn`.

Uses a stub `policy_fn` that records every key it is handed and asserts the
recorded keys are pairwise distinct.
"""
import os
import sys

import jax
import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "eval"))

from src.utils.config import Config
from src.environment.config_loader import load_env_params

import eval_rollout as er  # noqa: E402

_ENV_YAML = """\
environment:
  height: 10
  width: 10
  start_pos: [5, 5]
  max_steps: 5
  rest_action_enabled: false
  eat_action_enabled: false
  random_start_pos: false
  placement:
    mode: per_entity
  resources: []
  neutral_animals: []
  predators: []
  obstacles: []
  location_areas:
    - type: "grass"
      area: [[1, 1], [10, 10]]
body:
  with_satiation: false
  with_nutrition: false
  with_injury: false
  random_start_satiation: false
  random_start_nutrition: false
  random_start_injury: false
  max_satiation: 100
  max_nutrition: 100
  max_injury: 100
  metabolic_cost: 1.0
  food_nutrition_gain: 6
  eating_nutrition_cost: 0.0
  eating_reward_penalty: 0.0
  nutrition_to_satiation_scaling_factor: 1.0
  satiation_setpoint: 100
  start_satiation: 100
  start_nutrition: 100
  recovery_base_rate: 0.0
  recovery_accel_rate: 0.0
  injury_smoothing_duration: 1
  use_homeostatic_reward: false
  death_penalty: 0.0
  overeating_death: false
sensory:
  using_sensory: true
  olfactory_enabled: false
  sensor_radius: 5
  vector_size: 5
  decay_power: 2.0
  collision_sensor_range: 1
  location_sensor: false
  nociception_enabled: false
  nociception_size: 1
  visual_sensor_enabled: false
  visual_sensor_range: 0
  proprioception_enabled: false
  injury_observable: false
  nutrition_observable: false
  interoceptive_nociception_enabled: false
  interoceptive_convolution_enabled: false
  interoceptive_kernel_tau: 3.0
  interoceptive_kernel_length: 1
visualization:
  local_view_size: 3
"""


def _make_config(yaml_str):
    return Config(yaml.safe_load(yaml_str))


def test_policy_fn_receives_distinct_keys_per_step():
    config = _make_config(_ENV_YAML)
    params = load_env_params(config)

    seen_keys = []

    def stub_policy_fn(obs, carry, key, deterministic=True):
        seen_keys.append(np.asarray(key))
        return 0, carry  # always move (action 0); carry unused by this stub

    er._run_episode(
        params=params,
        policy_fn=stub_policy_fn,
        rng_key=jax.random.PRNGKey(0),
        max_steps=params.max_steps,
        deterministic=False,
    )

    assert len(seen_keys) == params.max_steps
    unique_keys = {tuple(k.tolist()) for k in seen_keys}
    assert len(unique_keys) == len(seen_keys), (
        f"expected {len(seen_keys)} pairwise-distinct per-step keys, got "
        f"{len(unique_keys)} unique (pre-fix: every step reuses the same frozen key)"
    )
