import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, List

class ActorCriticMLP(nnx.Module):
    """Actor-Critic MLP using Flax NNX with separate actor and critic networks."""
    def __init__(self, input_dim: int, action_dim: int, actor_fc_layers: List[int], critic_fc_layers: List[int], 
                 rngs: nnx.Rngs, activation: str = "tanh"):
        self.action_dim = action_dim
        self.activation = activation.lower()
        
        # Actor Network
        actor_layers = []
        in_dim = input_dim
        for hidden_dim in actor_fc_layers:
            actor_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.actor_layers = nnx.List(actor_layers)
        self.actor_head = nnx.Linear(in_dim, action_dim, rngs=rngs)
        
        # Critic Network
        critic_layers = []
        in_dim = input_dim
        for hidden_dim in critic_fc_layers:
            critic_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.critic_layers = nnx.List(critic_layers)
        self.critic_head = nnx.Linear(in_dim, 1, rngs=rngs)
    
    def _activate(self, x):
        """Apply the configured activation function."""
        if self.activation == "tanh":
            return jnp.tanh(x)
        else:
            return jax.nn.relu(x)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass to compute logits and value."""
        # Actor path
        a_x = x
        for layer in self.actor_layers:
            a_x = self._activate(layer(a_x))
        logits = self.actor_head(a_x)
        
        # Critic path
        c_x = x
        for layer in self.critic_layers:
            c_x = self._activate(layer(c_x))
        value = self.critic_head(c_x)
        
        return logits, value

def get_action_and_value_ppo_nnx(model, x, key=None, eval_mode=False):
    """Helper for inference with NNX."""
    logits, value = model(x)
    
    if eval_mode:
        action = jnp.argmax(logits)
        log_prob = 0.0
    else:
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        
    return action, log_prob, value.squeeze()
