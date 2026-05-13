"""Gymnasium bridge for the grid_world_pain JAX env into sheeprl.

This module wraps our JAX-based 5x5 gridworld env in a single-instance
gymnasium.Env that sheeprl can vectorize via gym.vector.SyncVectorEnv.
Sheeprl owns: vectorization, model, optimizer, replay buffer, training loop.
Our project provides: env step/reset logic and YAML config.

JAX must run on CPU here — sheeprl's torch owns the GPU.  The env-step is
microseconds on a 5x5 grid so there is no meaningful overhead.

The wrapper accumulates per-step episode signals (M1/M2/M5 behavior measures
and per-tag distance means) using the shared src/behavior/* modules and
places them on the terminal info dict at episode-done, so sheeprl's
SyncVectorEnv auto-promotes them to infos["final_info"][i].  The sheeprl
main loop can then update its MetricAggregator from these keys.
"""

# Force JAX to CPU before any JAX import so torch can claim the GPU.
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Ensure the grid_world_pain project root is on sys.path so that
# ``from src.xxx`` imports work regardless of working directory.
# Hydra changes cwd at runtime, so CWD-relative imports are unreliable.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params, load_behavior_measure_cfg
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation
from src.behavior.accumulators import (
    make_bm_state, bm_step_update, bm_reset_env,
    bm_finalise_episode, bm_finalise_to_wandb_keys,
)
from src.behavior.distance_aggregator import (
    make_dist_state, dist_step_update, dist_reset_env, dist_finalise_episode,
)
from src.behavior.episode_metrics import (
    make_episode_state, episode_reset_env,
    episode_step_update, episode_finalise_episode, episode_wandb_keys,
)


class GridWorldPainWrapper(gym.Env):
    """Single-instance gym.Env wrapping the grid_world_pain JAX environment.

    Sheeprl handles vectorization externally (SyncVectorEnv around N thunks).

    Args:
        config_path: Absolute path to the project's YAML env config (e.g.
            ``configs/experiment/dreamer_curriculum/01_food_only.yaml``).
        seed: Integer seed passed to the initial JAX PRNG key.
        apply_noise: If True (default for production), pass apply_noise=True to
            get_observation so injury-modulated sensory noise is realized.
            The smoke run (jzgkcep4) used False for cleanliness; production
            neuromodulation experiments require True.  Override via the
            GWP_APPLY_NOISE environment variable in the Hydra env config:
            ``${oc.env:GWP_APPLY_NOISE,true}``.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str, seed: int = 0, apply_noise: bool = True):
        super().__init__()
        # Load defaults first, then merge experiment config on top — this is
        # the same pattern used by train.py.  Experiment configs only override
        # keys they explicitly specify; missing keys fall back to the defaults.
        cfg = get_default_config()
        cfg.merge(Config.load_yaml(config_path))
        self._params = load_env_params(cfg)
        self._apply_noise = apply_noise

        # Action space: discrete, size read directly from EnvParams.action_dim.
        # action_dim = 4 + rest_action_enabled + eat_action_enabled (per config_loader.py:547).
        n_actions = int(self._params.action_dim)
        self.action_space = gym.spaces.Discrete(n_actions)

        # Determine observation shape by running a probe reset.
        # Use apply_noise=False for the probe regardless of self._apply_noise so
        # the obs shape is always derived cleanly (noise does not change shape).
        self._rng = jax.random.PRNGKey(seed)
        self._state = jax_reset(self._params, self._rng)
        obs0 = np.asarray(
            get_observation(self._state, self._params, apply_noise=False),
            dtype=np.float32,
        )
        # Obs space: single flat key "state", 1-D float32 vector.
        # Sheeprl DreamerV3 MLP encoder expects a gym.spaces.Dict with flat keys.
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32
            )
        })

        self._step_count = 0

        # --- Behavior-measure and distance accumulators ---
        # Tags come from env YAML params (possibly empty for food-only configs).
        self._neutral_tags  = tuple(self._params.neutral_tags)
        self._predator_tags = tuple(self._params.predator_tags)

        bm_cfg = load_behavior_measure_cfg(cfg)
        self._bm_enabled = bm_cfg is not None and bm_cfg.enabled
        if self._bm_enabled:
            self._bm_state = make_bm_state(
                num_envs=1,  # single-instance; sheeprl vectorizes externally
                num_predator_tags=len(self._predator_tags),
                num_neutral_tags=len(self._neutral_tags),
                bm_R=float(bm_cfg.cue_radius),
                bm_K=int(bm_cfg.obs_window),
            )
        else:
            self._bm_state = None
        self._dist_state = make_dist_state(
            1, len(self._predator_tags), len(self._neutral_tags)
        )
        self._ep_state = make_episode_state(num_envs=1)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        else:
            self._rng, _ = jax.random.split(self._rng)
        self._state = jax_reset(self._params, self._rng)
        self._step_count = 0
        obs = np.asarray(
            get_observation(self._state, self._params, apply_noise=self._apply_noise),
            dtype=np.float32,
        )
        # Reset accumulators in case sheeprl calls reset() mid-episode
        # (unusual but possible at training-start).
        if self._bm_enabled:
            bm_reset_env(self._bm_state, 0)
        dist_reset_env(self._dist_state, 0)
        episode_reset_env(self._ep_state, 0)
        return {"state": obs}, {}

    def step(self, action):
        a = int(action)
        self._state, reward, done, info_jax = jax_step(self._state, a, self._params)
        self._step_count += 1

        obs = np.asarray(
            get_observation(self._state, self._params, apply_noise=self._apply_noise),
            dtype=np.float32,
        )
        r = float(np.asarray(reward))
        terminated = bool(np.asarray(done))
        truncated = False  # max_steps handled inside our env via done flag

        # Unpack JAX info to numpy with [1, ...] batch axis so accumulator code
        # (written for [num_envs, ...]) works unchanged with num_envs=1.
        dist_per_pred_raw = np.asarray(info_jax['dist_per_predator'])
        dist_per_neut_raw = np.asarray(info_jax['dist_per_neutral'])
        info_np_t = {
            # Keys consumed by BM / distance accumulators (pre-existing)
            'ate_food':      np.asarray(info_jax['ate_food']).reshape(1).astype(bool),
            'agent_in_bush': np.asarray(info_jax['agent_in_bush']).reshape(1).astype(bool),
            'dist_to_food':  np.asarray(info_jax['dist_to_food']).reshape(1).astype(np.float32),
            'dist_to_pred':  np.asarray(info_jax['dist_to_pred']).reshape(1).astype(np.float32),
            'dist_to_neutral': np.asarray(info_jax['dist_to_neutral']).reshape(1).astype(np.float32),
            'dist_to_hiding_predator': np.asarray(info_jax['dist_to_hiding_predator']).reshape(1).astype(np.float32),
            'dist_per_predator': dist_per_pred_raw.reshape(1, -1).astype(np.float32),
            'dist_per_neutral':  dist_per_neut_raw.reshape(1, -1).astype(np.float32),
            # Keys consumed by EpisodeAccumulator (new)
            'reward':                 np.array([r], dtype=np.float32),
            'damage':                 np.asarray(info_jax['damage']).reshape(1).astype(np.float32),
            'damage_predator':        np.asarray(info_jax['damage_predator']).reshape(1).astype(np.float32),
            'damage_hiding_predator': np.asarray(info_jax['damage_hiding_predator']).reshape(1).astype(np.float32),
            'damage_obstacle':        np.asarray(info_jax['damage_obstacle']).reshape(1).astype(np.float32),
            'hit_predator':           np.asarray(info_jax['hit_predator']).reshape(1).astype(bool),
            'hit_hiding_predator':    np.asarray(info_jax['hit_hiding_predator']).reshape(1).astype(bool),
            'hit_neutral':            np.asarray(info_jax['hit_neutral']).reshape(1).astype(bool),
            'rested':                 np.asarray(info_jax['rested']).reshape(1).astype(bool),
            'event_collided':         np.asarray(info_jax['event_collided']).reshape(1).astype(bool),
        }
        done_mask = np.array([terminated], dtype=bool)

        # Per-step accumulation
        if self._bm_enabled:
            bm_step_update(self._bm_state, info_np_t, done_mask)
        dist_step_update(self._dist_state, info_np_t)
        episode_step_update(self._ep_state, info_np_t, done_mask)

        # At episode-done: finalise and build terminal info dict
        info_out: dict = {}
        if terminated:
            # Episode/* scalars (20 keys) — merged first so BM/dist keys can override if needed
            ep_scalar_dict = episode_finalise_episode(
                self._ep_state, 0, int(np.asarray(info_jax['termination_reason']))
            )
            info_out.update(ep_scalar_dict)
            episode_reset_env(self._ep_state, 0)

            if self._bm_enabled:
                ep_data_raw = bm_finalise_episode(
                    self._bm_state, 0, self._predator_tags, self._neutral_tags
                )
                info_out.update(
                    bm_finalise_to_wandb_keys(
                        ep_data_raw, self._predator_tags, self._neutral_tags
                    )
                )
                bm_reset_env(self._bm_state, 0)
            info_out.update(
                dist_finalise_episode(
                    self._dist_state, 0, self._neutral_tags, self._predator_tags
                )
            )
            dist_reset_env(self._dist_state, 0)

        return {"state": obs}, r, terminated, truncated, info_out

    def close(self):
        pass
