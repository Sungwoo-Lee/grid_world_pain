"""vmap-batched JAX vector env for the grid_world_pain gridworld.

Replaces gym.vector.SyncVectorEnv around N python-instance wrappers with a
single instance that holds one batched EnvState and steps via jax.vmap.
Implements the gymnasium vector-env API just enough for sheeprl DreamerV3.

v1 scope (smoke-only):
  - No per-step behavior-measure / distance / episode accumulators
    (the smoke env config does not fire them).
  - apply_noise=False (smoke env has perceptual_noise.enabled: false).
  - No per-env seed convention (single top-level seed split into N).

Production parity (full accumulator support, noise, per-env seed) is v2.
"""

# Force JAX to CPU before any JAX import (PyTorch owns the GPU).
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import functools
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation


class JAXVectorEnv:
    """vmap-batched gym vector env.

    Replaces gym.vector.SyncVectorEnv for the grid_world_pain env, exposing
    parallelism that is currently hidden inside sequential Python loop.

    Gym vector-env API surface implemented:
        reset(seed=None) -> (obs_dict, info_list)
        step(actions) -> (obs_dict, reward, terminated, truncated, info_list)
        close()
        num_envs (int attribute)
        single_action_space, single_observation_space  (per-env)
        action_space, observation_space                (batched per gym convention)
        __len__

    Args:
        config_path: Path to env YAML.
        num_envs: Batch size.
        seed: Top-level seed; split into N sub-keys for the initial reset.
    """

    def __init__(self, config_path: str, num_envs: int, seed: int = 0):
        cfg = get_default_config()
        cfg.merge(Config.load_yaml(config_path))
        self._params = load_env_params(cfg)
        self.num_envs = int(num_envs)
        self._apply_noise = False  # v1: smoke-only, noise disabled

        # Per-env action/obs spaces (the "single_" versions).
        n_actions = int(self._params.action_dim)
        self.single_action_space = gym.spaces.Discrete(n_actions)

        # Build vmapped functions once.
        self._vmap_reset = jax.jit(
            jax.vmap(jax_reset, in_axes=(None, 0))
        )
        self._vmap_step = jax.jit(
            jax.vmap(jax_step, in_axes=(0, 0, None))
        )
        # get_observation has a non-pytree static kwarg apply_noise; partial it out
        _obs_fn = functools.partial(get_observation, apply_noise=False)
        self._vmap_obs = jax.jit(jax.vmap(_obs_fn, in_axes=(0, None)))

        # Initial reset to determine obs shape.
        self._rng = jax.random.PRNGKey(int(seed))
        self._rng, sub = jax.random.split(self._rng)
        init_keys = jax.random.split(sub, self.num_envs)
        self._state = self._vmap_reset(self._params, init_keys)
        obs0 = np.asarray(self._vmap_obs(self._state, self._params), dtype=np.float32)

        # Per-env obs space (strip batch dim from obs0 for single_observation_space).
        obs_dim = obs0.shape[1:]
        self.single_observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32
            )
        })

        # Batched spaces (gym vector-env convention: stack of single).
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

        # `envs` attr — sheeprl reads `envs.envs[0]` to introspect inner env
        # for per-tag aggregator registration (GWP-PATCH-A in run_dreamer_v3.py).
        # Provide a single-element list with a duck-typed shim exposing
        # _neutral_tags / _predator_tags so the existing patch succeeds.
        self.envs = [self._make_introspection_shim()]

    def _make_introspection_shim(self):
        """Build a minimal object that GWP-PATCH-A can introspect.

        GWP-PATCH-A reads `inner_env._neutral_tags` and `inner_env._predator_tags`.
        We expose those from EnvParams so the patch's per-tag aggregator
        registration still works in the JAXVectorEnv path.
        """
        class _Shim:
            pass
        s = _Shim()
        s._neutral_tags  = tuple(self._params.neutral_tags)
        s._predator_tags = tuple(self._params.predator_tags)
        return s

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        self._rng, sub = jax.random.split(self._rng)
        keys = jax.random.split(sub, self.num_envs)
        self._state = self._vmap_reset(self._params, keys)
        obs = np.asarray(self._vmap_obs(self._state, self._params), dtype=np.float32)
        return {"state": obs}, [{} for _ in range(self.num_envs)]

    def step(self, actions):
        # 1. Step all N in one fused kernel.
        actions_j = jnp.asarray(np.asarray(actions), dtype=jnp.int32)
        next_state, reward, done, _info_j = self._vmap_step(
            self._state, actions_j, self._params
        )

        # 2. Build fresh reset candidates for every env.
        self._rng, sub = jax.random.split(self._rng)
        reset_keys = jax.random.split(sub, self.num_envs)
        reset_state = self._vmap_reset(self._params, reset_keys)

        # 3. Auto-reset blend: per-leaf jnp.where on the done mask.
        #    Broadcast the (N,) done mask up to each leaf's shape.
        def _blend(nx, rx):
            # done has shape (N,); pad with trailing 1s to match leaf rank.
            d = done.reshape((self.num_envs,) + (1,) * (nx.ndim - 1))
            return jnp.where(d, rx, nx)
        self._state = jax.tree.map(_blend, next_state, reset_state)

        # 4. Compute obs from the BLENDED state (gym convention: obs at t+1
        #    after auto-reset is the obs OF the new episode, not the dead one).
        obs = np.asarray(self._vmap_obs(self._state, self._params), dtype=np.float32)
        r = np.asarray(reward, dtype=np.float32)
        d_np = np.asarray(done, dtype=bool)
        # truncated tracked inside jax_step's termination_reason==1; lumped into
        # done for v1 simplicity. SyncVectorEnv's separation of terminated vs
        # truncated affects bootstrapping in sheeprl — for the smoke at
        # learning_starts=1024 with 1000-step runs this is irrelevant.
        terminated = d_np
        truncated  = np.zeros_like(d_np)
        info_list  = [{} for _ in range(self.num_envs)]
        return {"state": obs}, r, terminated, truncated, info_list

    def close(self):
        pass

    def __len__(self):
        return self.num_envs
