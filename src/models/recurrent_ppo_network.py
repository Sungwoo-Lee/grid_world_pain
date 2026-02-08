import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Union

class ActorCriticRNN(nnx.Module):
    """Actor-Critic RNN (LSTM or GRU) using Flax NNX."""
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int, rngs: nnx.Rngs,
                 rnn_type: str = "LSTM", activation: str = "tanh"):
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.rnn_type = rnn_type.upper()
        self.activation = activation.lower()
        
        # Input projection
        self.input_proj = nnx.Linear(input_dim, hidden_size, rngs=rngs)
        
        # RNN Layer (LSTM or GRU)
        if self.rnn_type == "LSTM":
            self.rnn_cell = nnx.LSTMCell(hidden_size, hidden_size, rngs=rngs)
        else:
            self.rnn_cell = nnx.GRUCell(hidden_size, hidden_size, rngs=rngs)
        
        # Actor Head
        self.actor_fc1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.actor_fc2 = nnx.Linear(hidden_size, action_dim, rngs=rngs)
        
        # Critic Head
        self.critic_fc1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.critic_fc2 = nnx.Linear(hidden_size, 1, rngs=rngs)
    
    def _activate(self, x):
        """Apply the configured activation function."""
        if self.activation == "tanh":
            return jnp.tanh(x)
        else:
            return jax.nn.relu(x)

    def __call__(self, x: jnp.ndarray, h: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, jnp.ndarray, Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]]:
        """Forward pass for a single step."""
        # Project input to hidden size (always ReLU for input projection)
        x_proj = jax.nn.relu(self.input_proj(x))
        
        # RNN forward
        if self.rnn_type == "LSTM":
            # LSTM: h is a tuple (h_state, c_state)
            # LSTMCell expects (carry, x) -> returns (new_carry, y)
            # carry is (h, c)
            h_new, x_h = self.rnn_cell(h, x_proj)
        else:
            # GRU: h is a single array
            h_new, x_h = self.rnn_cell(h, x_proj)
        
        # Actor (with configurable activation)
        a_h = self._activate(self.actor_fc1(x_h))
        logits = self.actor_fc2(a_h)
        
        # Critic (with configurable activation)
        c_h = self._activate(self.critic_fc1(x_h))
        value = self.critic_fc2(c_h)
        
        return logits, value, h_new

    def initial_state(self, batch_size: int = None):
        """Returns the initial hidden state for the RNN."""
        if self.rnn_type == "LSTM":
            if batch_size is None:
                return (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))
            return (jnp.zeros((batch_size, self.hidden_size)), jnp.zeros((batch_size, self.hidden_size)))
        else:
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
