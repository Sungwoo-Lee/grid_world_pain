"""Regression test — Bug 2 (Track B, 2026-07-23 eval/logging fix plan):
`_run_parallel_env_eval` in src/utils/evaluation_core.py gated the
initial (step-0) per-slot buffer seeding on `if record_stats:`, but the
post-episode refill seeding is unconditional and `episode_lengths.append(
len(slot_rewards[i]) - 1)` always subtracts one assuming a step-0 sentinel
entry is present. With `record_stats=False` (the project default) and
`effective_num_envs > 1`, the first `effective_num_envs` episodes are missing
their step-0 sentinel, so their recorded length is undercounted by exactly 1.

Fix (docs/develop/active/issues/FIX_EVAL_LOGGING_TRACK_B_20260723.md, Bug 2):
seed the step-0 sentinel unconditionally (matching the refill block).

Uses a death-disabled tiny env (no nutrition/injury termination, no damage
sources) with a small max_steps so every episode truncates at exactly
max_steps -- both first-generation and refilled episodes must then report the
same length.
"""
import os
import sys

import jax
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.utils.evaluation_core import _run_parallel_env_eval

_ENV_YAML = """\
environment:
  height: 10
  width: 10
  start_pos: [5, 5]
  max_steps: 8
  rest_action_enabled: false
  eat_action_enabled: false
  random_start_pos: false
  placement:
    mode: per_entity
  resources:
    - name: "food"
      type: "food"
      count: 1
      spawn_area: [[1, 1], [3, 3]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: 5
      regeneration_delay: 0
      damage: [0.0, 0.0]
      nociception_intensity: 0.0
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
  olfactory_sensor_range: 0
  visual_blur_enabled: false
  visual_blur_radial_scale: 0.5
  visual_blur_anisotropy: 3.0
  visual_blur_sigma_floor: 0.5
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


def test_first_generation_episodes_not_undercounted():
    config = _make_config(_ENV_YAML)
    params = load_env_params(config)
    max_steps = params.max_steps

    episode_rewards = []
    episode_lengths = []

    _run_parallel_env_eval(
        model=None,
        params=params,
        config=config,
        num_episodes=4,
        effective_num_envs=2,
        seed=0,
        results_dir="",
        checkpoint_pct=0,
        key=jax.random.PRNGKey(0),
        video_dir=None,
        breakdown=None,
        icon_config=None,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        record_stats=False,
        stats_dir=None,
        stat_headers=None,
        action_map=None,
        params_ref=params,
        render_video=False,
        wandb_enabled=False,
        debug=False,
        quiet=True,
        record_true_obs=False,
    )

    assert len(episode_lengths) == 4
    assert all(L == max_steps for L in episode_lengths), (
        f"episode_lengths={episode_lengths}, expected all == max_steps={max_steps} "
        "(pre-fix: the first effective_num_envs=2 first-generation episodes are "
        "undercounted by 1 step)"
    )
