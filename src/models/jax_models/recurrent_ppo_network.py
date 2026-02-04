import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple

class ActorCriticRNN(nnx.Module):
    """Actor-Critic RNN (GRU) using Flax NNX."""
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        # Input projection
        self.input_proj = nnx.Linear(input_dim, hidden_size, rngs=rngs)
        
        # RNN Layer (GRU) - takes hidden_size input after projection
        self.gru = nnx.GRUCell(hidden_size, hidden_size, rngs=rngs)
        
        # Actor Head
        self.actor_fc1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.actor_fc2 = nnx.Linear(hidden_size, action_dim, rngs=rngs)
        
        # Critic Head
        self.critic_fc1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.critic_fc2 = nnx.Linear(hidden_size, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass for a single step."""
        # Project input to hidden size
        x_proj = jax.nn.relu(self.input_proj(x))
        
        # GRU expects (h, x) -> (h_new, y)
        h_new, x_h = self.gru(h, x_proj)
        
        # Actor
        a_h = jax.nn.relu(self.actor_fc1(x_h))
        logits = self.actor_fc2(a_h)
        
        # Critic
        c_h = jax.nn.relu(self.critic_fc1(x_h))
        value = self.critic_fc2(c_h)
        
        return logits, value, h_new

    def initial_state(self, batch_size: int = None):
        if batch_size is None:
            return jnp.zeros((self.hidden_size,))
        return jnp.zeros((batch_size, self.hidden_size))

def get_action_and_value_nnx(model, x, h, key=None, eval_mode=False):
    """Helper for inference with NNX."""
    logits, value, h_new = model(x, h)
    
    if eval_mode:
        action = jnp.argmax(logits)
        log_prob = 0.0
    else:
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        
    return action, log_prob, value.squeeze(), h_new
