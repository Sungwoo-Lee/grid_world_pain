import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Union, List

class DRQNNetwork(nnx.Module):
    """Deep Recurrent Q-Network using Flax NNX."""
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int, rnn_type: str, fc_layers: List[int], rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.rnn_type = rnn_type.upper()
        
        # FC Pre-processing
        layers = []
        in_dim = input_dim
        for hidden_dim in fc_layers:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.fc_layers = nnx.List(layers)
        
        # RNN Layer
        if self.rnn_type == "LSTM":
            self.rnn_cell = nnx.LSTMCell(in_dim, hidden_size, rngs=rngs)
        else:
            self.rnn_cell = nnx.GRUCell(in_dim, hidden_size, rngs=rngs)
        
        # Output Head
        self.output_head = nnx.Linear(hidden_size, action_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, h: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]]:
        """Forward pass for a single step."""
        # FC Pre-processing
        for layer in self.fc_layers:
            x = jax.nn.relu(layer(x))
            
        # RNN forward
        if self.rnn_type == "LSTM":
            h_new, x_h = self.rnn_cell(h, x)
        else:
            h_new, x_h = self.rnn_cell(h, x)
            
        # Q-values
        q_values = self.output_head(x_h)
        return q_values, h_new

    def initial_state(self, batch_size: int = None):
        """Returns the initial hidden state."""
        if self.rnn_type == "LSTM":
            if batch_size is None:
                return (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))
            return (jnp.zeros((batch_size, self.hidden_size)), jnp.zeros((batch_size, self.hidden_size)))
        else:
            if batch_size is None:
                return jnp.zeros((self.hidden_size,))
            return jnp.zeros((batch_size, self.hidden_size))

def get_action_drqn_nnx(model, obs, h, key, epsilon, eval_mode=False):
    """Epsilon-greedy action selection for DRQN."""
    q_values, h_new = model(obs, h)
    
    if eval_mode:
        return jnp.argmax(q_values), h_new
    
    key, subkey = jax.random.split(key)
    do_random = jax.random.uniform(subkey) < epsilon
    
    key, subkey = jax.random.split(key)
    random_action = jax.random.randint(subkey, (), 0, q_values.shape[-1])
    greedy_action = jnp.argmax(q_values)
    
    action = jnp.where(do_random, random_action, greedy_action)
    return action, h_new
