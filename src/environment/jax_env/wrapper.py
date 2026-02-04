import jax
import jax.numpy as jnp
from .state import EnvState, EnvParams
from .core import jax_step, jax_reset
from .sensor import get_observation

class ParallelEnv:
    """Vectorized Environment Wrapper for JAX."""
    def __init__(self, params: EnvParams):
        self.params = params
        
        # Vectorize reset and step
        self._v_reset = jax.vmap(jax_reset, in_axes=(None, 0))
        self._v_step = jax.vmap(jax_step, in_axes=(0, 0, None))
        self._v_obs = jax.vmap(get_observation, in_axes=(0, None))

    def reset(self, key: jax.random.PRNGKey, num_envs: int) -> tuple[EnvState, jnp.ndarray]:
        keys = jax.random.split(key, num_envs)
        states = self._v_reset(self.params, keys)
        obs = self._v_obs(states, self.params)
        return states, obs

    def step(self, states: EnvState, actions: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Steps all environments in parallel."""
        # Standard vmapped step
        next_states, rewards, dones, infos = self._v_step(states, actions, self.params)
        
        # Note: Auto-reset is usually handled at the training loop level in JAX 
        # (e.g. jax.lax.cond) to keep it pure. 
        # But we can provide a helper here if needed.
        obs = self._v_obs(next_states, self.params)
        
        return next_states, obs, rewards, dones, infos

def auto_reset_step(states: EnvState, actions: jnp.ndarray, params: EnvParams, key: jax.random.PRNGKey):
    """vmapped step with automatic reset for finished environments."""
    
    def step_fn(state, action):
        next_state, reward, done, info = jax_step(state, action, params)
        
        # If done, reset immediately for the NEXT step's input
        # Note: In most RL loops (CleanRL/Gymnax), the 'obs' returned is for the next state.
        # If terminal, we often want the reset obs.
        
        reset_key, _ = jax.random.split(state.key)
        reset_state = jax_reset(params, reset_key)
        
        # Replace state if done
        # We use tree_utils to swap the entire Pytree
        final_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y),
            reset_state, next_state
        )
        
        obs = get_observation(next_state, params) # Return terminal obs or masked?
        # Usually, for 'done' transitions, we return the terminal obs, 
        # but the state passed back should be reset.
        
        return final_state, obs, reward, done, info

    return jax.vmap(step_fn, in_axes=(0, 0))(states, actions)
