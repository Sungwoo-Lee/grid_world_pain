"""Gymnasium bridge for the grid_world_pain JAX env into sheeprl.

This module wraps our JAX-based 5x5 gridworld env in a single-instance
gymnasium.Env that sheeprl can vectorize via gym.vector.SyncVectorEnv.
Sheeprl owns: vectorization, model, optimizer, replay buffer, training loop.
Our project provides: env step/reset logic and YAML config.

JAX must run on CPU here — sheeprl's torch owns the GPU.  The env-step is
microseconds on a 5x5 grid so there is no meaningful overhead.
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
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation


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
        return {"state": obs}, {}

    def step(self, action):
        a = int(action)
        self._state, reward, done, info = jax_step(self._state, a, self._params)
        self._step_count += 1

        obs = np.asarray(
            get_observation(self._state, self._params, apply_noise=self._apply_noise),
            dtype=np.float32,
        )
        r = float(np.asarray(reward))
        terminated = bool(np.asarray(done))
        truncated = False  # max_steps handled inside our env via done flag

        # Drop info entirely — SyncVectorEnv cannot stack JAX arrays across
        # env workers, and info is unused by the DreamerV3 training loop.
        return {"state": obs}, r, terminated, truncated, {}

    def close(self):
        pass
