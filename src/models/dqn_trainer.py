import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import NamedTuple, Any

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray

class ReplayBuffer:
    """A simple Replay Buffer for JAX using JNP arrays."""
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = jnp.zeros((capacity, obs_dim))
        self.actions = jnp.zeros((capacity,), dtype=jnp.int32)
        self.rewards = jnp.zeros((capacity,))
        self.next_obs = jnp.zeros((capacity, obs_dim))
        self.dones = jnp.zeros((capacity,), dtype=jnp.bool_)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Adds a batch of transitions to the buffer."""
        batch_size = obs.shape[0]
        indices = (jnp.arange(self.ptr, self.ptr + batch_size) % self.capacity)
        
        self.obs = self.obs.at[indices].set(obs)
        self.actions = self.actions.at[indices].set(action)
        self.rewards = self.rewards.at[indices].set(reward)
        self.next_obs = self.next_obs.at[indices].set(next_obs)
        self.dones = self.dones.at[indices].set(done)
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, key: jax.random.PRNGKey) -> Transition:
        """Samples a batch of transitions from the buffer."""
        indices = jax.random.randint(key, (batch_size,), 0, self.size)
        return Transition(
            obs=self.obs[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_obs=self.next_obs[indices],
            done=self.dones[indices]
        )

def dqn_loss_fn(model, target_model, batch, gamma):
    """Computes the DQN loss (MSE)."""
    q_values = model(batch.obs)
    # Get Q-value of the action taken
    q_val = jax.vmap(lambda q, a: q[a])(q_values, batch.action)
    
    # Compute target Q-value
    next_q_values = target_model(batch.next_obs)
    next_q_max = jnp.max(next_q_values, axis=-1)
    target_q = batch.reward + gamma * next_q_max * (1.0 - batch.done)
    
    loss = jnp.mean(jnp.square(q_val - jax.lax.stop_gradient(target_q)))
    return loss

def update_step_dqn(model, target_model, optimizer, batch, gamma):
    """Performs a single DQN update step."""
    def loss_fn(m):
        return dqn_loss_fn(m, target_model, batch, gamma)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss
