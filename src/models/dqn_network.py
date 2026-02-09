import jax
import jax.numpy as jnp
from flax import nnx
from typing import List

class DQNNetwork(nnx.Module):
    """Deep Q-Network using Flax NNX."""
    def __init__(self, input_dim: int, action_dim: int, fc_layers: List[int], rngs: nnx.Rngs):
        self.action_dim = action_dim
        
        layers = []
        in_dim = input_dim
        for hidden_dim in fc_layers:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
            
        self.fc_layers = nnx.List(layers)
        self.output_layer = nnx.Linear(in_dim, action_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass to compute Q-values."""
        for layer in self.fc_layers:
            x = jax.nn.relu(layer(x))
        return self.output_layer(x)

def get_action_dqn_nnx(model, obs, key, epsilon, eval_mode=False):
    """Epsilon-greedy action selection for DQN."""
    q_values = model(obs)
    
    if eval_mode:
        return jnp.argmax(q_values)
    
    # Epsilon-greedy
    key, subkey = jax.random.split(key)
    do_random = jax.random.uniform(subkey) < epsilon
    
    key, subkey = jax.random.split(key)
    random_action = jax.random.randint(subkey, (), 0, q_values.shape[-1])
    greedy_action = jnp.argmax(q_values)
    
    action = jnp.where(do_random, random_action, greedy_action)
    return action
