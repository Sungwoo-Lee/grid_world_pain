import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import NamedTuple, Any, Tuple, Union

class RecurrentTransition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

class RecurrentReplayBuffer:
    """A recurrent replay buffer for storing and sampling sequences."""
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = jnp.zeros((capacity, obs_dim))
        self.actions = jnp.zeros((capacity,), dtype=jnp.int32)
        self.rewards = jnp.zeros((capacity,))
        self.dones = jnp.zeros((capacity,), dtype=jnp.bool_)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, done):
        """Adds a batch of transitions."""
        batch_size = obs.shape[0]
        indices = (jnp.arange(self.ptr, self.ptr + batch_size) % self.capacity)
        
        self.obs = self.obs.at[indices].set(obs)
        self.actions = self.actions.at[indices].set(action)
        self.rewards = self.rewards.at[indices].set(reward)
        self.dones = self.dones.at[indices].set(done)
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample_sequences(self, batch_size: int, seq_len: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Samples a batch of sequences. 
        Note: This is a simple implementation that might cross episode boundaries.
        In a more robust version, we'd ensure sequences stay within an episode.
        """
        # Start indices for sequences (must be at least seq_len steps from current ptr if buffer is not full)
        max_idx = self.size - seq_len - 1
        starts = jax.random.randint(key, (batch_size,), 0, max_idx)
        
        def get_seq(start):
            obs = jax.lax.dynamic_slice(self.obs, (start, 0), (seq_len + 1, self.obs.shape[1]))
            actions = jax.lax.dynamic_slice(self.actions, (start,), (seq_len + 1,))
            rewards = jax.lax.dynamic_slice(self.rewards, (start,), (seq_len + 1,))
            dones = jax.lax.dynamic_slice(self.dones, (start,), (seq_len + 1,))
            return obs, actions, rewards, dones

        obs_v, act_v, rew_v, done_v = jax.vmap(get_seq)(starts)
        
        # obs_seq: (B, L, D), next_obs_seq: (B, L, D)
        obs_seq = obs_v[:, :-1, :]
        next_obs_seq = obs_v[:, 1:, :]
        actions = act_v[:, :-1]
        rewards = rew_v[:, :-1]
        dones = done_v[:, :-1]
        
        return obs_seq, actions, rewards, next_obs_seq, dones

def drqn_loss_fn(model, target_model, obs_seq, actions, rewards, next_obs_seq, dones, gamma, burn_in_len):
    """Computes the DRQN loss with burn-in."""
    batch_size = obs_seq.shape[0]
    
    def scan_fn(model_instance, target_instance):
        def forward_scan(h_carry, x):
            o, a, r, no, d = x
            q_vals, h_next = model_instance(o, h_carry)
            
            # Target calculation
            target_q_vals, h_target_next = target_instance(no, h_carry) # Should target use same h?
            # Actually, target should also scan to get proper hidden states
            return h_next, (q_vals, target_q_vals, h_target_next)
        return forward_scan

    # Initial hidden states
    h_init = model.initial_state(batch_size)
    h_target_init = target_model.initial_state(batch_size)
    
    # We need to scan over the sequence dimension (axis 1)
    # Re-order to (seq_len, batch, dim) for scan
    obs_t = jnp.transpose(obs_seq, (1, 0, 2))
    next_obs_t = jnp.transpose(next_obs_seq, (1, 0, 2))
    
    def step_fn(carry, x):
        h, h_target = carry
        o, no = x
        q, h_new = model(o, h)
        q_target, h_target_new = target_model(no, h_target)
        return (h_new, h_target_new), (q, q_target)

    _, (all_q, all_target_q) = jax.lax.scan(step_fn, (h_init, h_target_init), (obs_t, next_obs_t))
    
    # all_q: (L, B, A), all_target_q: (L, B, A)
    # Back to (B, L, A)
    all_q = jnp.transpose(all_q, (1, 0, 2))
    all_target_q = jnp.transpose(all_target_q, (1, 0, 2))
    
    # Slice for training (after burn-in)
    q_train = all_q[:, burn_in_len:]
    target_q_train = jax.lax.stop_gradient(all_target_q[:, burn_in_len:])
    
    act_train = actions[:, burn_in_len:]
    rew_train = rewards[:, burn_in_len:]
    done_train = dones[:, burn_in_len:]
    
    # Get Q-value of actions
    q_val = jax.vmap(lambda q, a: q[jnp.arange(len(a)), a])(q_train, act_train)
    max_target_q = jnp.max(target_q_train, axis=-1)
    
    expected_q = rew_train + gamma * max_target_q * (1.0 - done_train)
    
    loss = jnp.mean(jnp.square(q_val - expected_q))
    return loss

def update_step_drqn(model, target_model, optimizer, obs_seq, actions, rewards, next_obs_seq, dones, gamma, burn_in_len):
    """Performs a single DRQN update step."""
    def loss_fn(m):
        return drqn_loss_fn(m, target_model, obs_seq, actions, rewards, next_obs_seq, dones, gamma, burn_in_len)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss
