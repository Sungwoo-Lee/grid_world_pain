import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Union, Optional, Any

from .modulated_gru_cell import ModulatedGRUCell
from .neuromodulator import NeuromodulatorRNN


class ActorCriticRNN(nnx.Module):
    """Actor-Critic RNN (LSTM or GRU) using Flax NNX, with optional neuromodulation.

    When modulation_config is None, this behaves identically to the original
    unmodulated network (true baseline control, §5.2).
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int, rngs: nnx.Rngs,
                 rnn_type: str = "LSTM", activation: str = "tanh",
                 modulation_config: Optional[dict] = None):
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.rnn_type = rnn_type.upper()
        self.activation = activation.lower()
        self.modulation_enabled = modulation_config is not None and modulation_config.get('type') is not None

        # Input projection
        self.input_proj = nnx.Linear(input_dim, hidden_size, rngs=rngs)

        # RNN Layer (LSTM or ModulatedGRU)
        if self.rnn_type == "LSTM":
            self.rnn_cell = nnx.LSTMCell(hidden_size, hidden_size, rngs=rngs)
        else:
            if self.modulation_enabled:
                # Use custom GRU with gate-bias support
                self.rnn_cell = ModulatedGRUCell(hidden_size, hidden_size, rngs=rngs)
            else:
                self.rnn_cell = nnx.GRUCell(hidden_size, hidden_size, rngs=rngs)

        # Actor Head
        self.actor_fc1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.actor_fc2 = nnx.Linear(hidden_size, action_dim, rngs=rngs)

        # Critic Head
        self.critic_fc1 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.critic_fc2 = nnx.Linear(hidden_size, 1, rngs=rngs)

        # Neuromodulator (only constructed when enabled — §5.2 baseline control)
        if self.modulation_enabled:
            # All values must be explicitly set in config (no safe defaults)
            mod_hidden = modulation_config['mod_hidden_size']
            grouping = modulation_config['grouping_size']
            percept_bias = modulation_config['percept_bias_init']
            memory_bias = modulation_config['memory_bias_init']
            temp_clip = tuple(modulation_config['temp_clip'])

            self.modulator = NeuromodulatorRNN(
                input_dim=input_dim,
                target_hidden_size=hidden_size,
                mod_hidden_size=mod_hidden,
                grouping_size=grouping,
                percept_bias_init=percept_bias,
                memory_bias_init=memory_bias,
                temp_clip=temp_clip,
                rngs=rngs,
            )

    def _activate(self, x):
        """Apply the configured activation function."""
        if self.activation == "tanh":
            return jnp.tanh(x)
        else:
            return jax.nn.relu(x)

    def __call__(
        self,
        x: jnp.ndarray,
        h: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
        """Forward pass for a single step.

        Args:
            x: Observation, shape (..., input_dim).
            h: Hidden state. When modulation is disabled, this is the task RNN
               state (array for GRU, tuple for LSTM). When modulation is enabled,
               this is a tuple (task_h, mod_h).

        Returns:
            (logits, value, h_new): Policy logits, value estimate, and new hidden state.
        """
        if self.modulation_enabled:
            # Unpack combined hidden state
            task_h, mod_h = h

            # --- Modulator forward pass ---
            mod_output, mod_h_new = self.modulator(x, mod_h)

            # --- Task path with modulation ---
            # Layer 1: Input projection
            x_proj = jax.nn.relu(self.input_proj(x))

            # INJECTION A: Perceptual Gate (§5.2)
            x_proj = x_proj * jax.nn.sigmoid(mod_output.z_percept)

            # Layer 2: RNN forward with gate-bias injection
            if self.rnn_type == "LSTM":
                # For LSTM, gate_bias not yet supported — pass through normally
                h_new, x_h = self.rnn_cell(task_h, x_proj)
            else:
                # INJECTION B: Internal Gate-Bias (§5.1a)
                h_new, x_h = self.rnn_cell(task_h, x_proj, gate_bias=mod_output.z_memory)

            # Layer 3: Actor/Critic heads
            a_h = self._activate(self.actor_fc1(x_h))
            logits = self.actor_fc2(a_h)

            # INJECTION C: Bounded Temperature (§5.3)
            logits = logits / mod_output.temperature

            c_h = self._activate(self.critic_fc1(x_h))
            value = self.critic_fc2(c_h)

            # Pack combined hidden state
            h_combined_new = (h_new, mod_h_new)
            return logits, value, h_combined_new, mod_output

        else:
            # --- Original unmodulated path (exact baseline) ---
            x_proj = jax.nn.relu(self.input_proj(x))

            if self.rnn_type == "LSTM":
                h_new, x_h = self.rnn_cell(h, x_proj)
            else:
                h_new, x_h = self.rnn_cell(h, x_proj)

            a_h = self._activate(self.actor_fc1(x_h))
            logits = self.actor_fc2(a_h)

            c_h = self._activate(self.critic_fc1(x_h))
            value = self.critic_fc2(c_h)

            return logits, value, h_new, None

    def initial_state(self, batch_size: int = None):
        """Returns the initial hidden state for the RNN."""
        if self.rnn_type == "LSTM":
            if batch_size is None:
                task_h = (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))
            else:
                task_h = (jnp.zeros((batch_size, self.hidden_size)), jnp.zeros((batch_size, self.hidden_size)))
        else:
            if batch_size is None:
                task_h = jnp.zeros((self.hidden_size,))
            else:
                task_h = jnp.zeros((batch_size, self.hidden_size))

        if self.modulation_enabled:
            mod_h = self.modulator.initial_state(batch_size)
            return (task_h, mod_h)
        else:
            return task_h


@nnx.jit(static_argnames="eval_mode")
def get_action_and_value_nnx(model, x, h, key=None, eval_mode=False):
    """Helper for inference with NNX."""
    logits, value, h_new, mod_info = model(x, h)

    if eval_mode:
        action = jnp.argmax(logits)
        log_prob = 0.0
    else:
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]

    return action, log_prob, value.squeeze(), h_new, mod_info
