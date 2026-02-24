import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Union, Optional, Any

from .modulated_gru_cell import ModulatedGRUCell
from .neuromodulator import NeuromodulatorRNN



class MLP(nnx.Module):
    """Simple Multi-Layer Perceptron helper."""
    def __init__(self, input_dim: int, hidden_layers: list, output_dim: int, rngs: nnx.Rngs):
        layers = []
        in_d = input_dim
        for h in hidden_layers:
            layers.append(nnx.Linear(in_d, h, rngs=rngs))
            layers.append(jax.nn.relu)
            in_d = h
        layers.append(nnx.Linear(in_d, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)


class ObservationEncoder(nnx.Module):
    """Unified observation encoder supporting flat and hierarchical modes."""
    def __init__(self, input_dim: int, hidden_size: int, breakdown: dict,
                 config: Optional[dict], rngs: nnx.Rngs):
        if config is None:
            raise ValueError("Strict Config: encoding_config is required for ObservationEncoder.")
            
        self.mode = config['encoding_mode']
        self.breakdown = breakdown
        self.hidden_size = hidden_size
        
        if self.mode == 'hierarchical':
            h_params = config['hierarchical_params']
            default_mlp = h_params['default_mlp']
            unimodal_overrides = h_params.get('unimodal_overrides', {})
            hub_overrides = h_params.get('hub_overrides', {})
            
            # 1. Unimodal Encoders
            encoders = {}
            for name, dim in breakdown.items():
                # Overrides are optional, but if a name is in overrides, we use it.
                # Otherwise we MUST have a default_mlp.
                target_mlp = unimodal_overrides.get(name.lower(), default_mlp)
                # Output dim for unimodal encoders: using hidden_size
                encoders[name] = MLP(dim, target_mlp, hidden_size, rngs=rngs)
            self.encoders = nnx.Dict(encoders)
            
            # 2. Body-State Hub (Intero + Extero Nocicep + Collision)
            body_sensors = ["Satiation", "Nutrition", "Injury", "Extero Nociception", "Collision"]
            body_in_dim = sum([hidden_size for name in body_sensors if name in breakdown])
            body_mlp_struct = hub_overrides.get('body_state', default_mlp)
            self.body_hub = MLP(body_in_dim, body_mlp_struct, hidden_size, rngs=rngs)
            
            # 3. Association Hub
            assoc_sensors = ["Olfaction", "Location", "Visual", "Proprioception"]
            assoc_in_dim = sum([hidden_size for name in assoc_sensors if name in breakdown]) + hidden_size
            assoc_mlp_struct = hub_overrides.get('association', default_mlp)
            self.assoc_hub = MLP(assoc_in_dim, assoc_mlp_struct, hidden_size, rngs=rngs)
            
        else:
            self.monolith = nnx.Linear(input_dim, hidden_size, rngs=rngs)

    def __call__(self, x):
        # Universal slicing - robust to any batch shape
        slices = {}
        start = 0
        for name, dim in self.breakdown.items():
            slices[name] = x[..., start : start + dim]
            start += dim
            
        if self.mode == 'hierarchical':
            # Phase 1: Unimodal Encoding
            encoded = {name: self.encoders[name](s) for name, s in slices.items()}
            
            # Phase 2: Body-State Sub-Fusion
            body_sensors = ["Satiation", "Nutrition", "Injury", "Extero Nociception", "Collision"]
            body_inputs = [encoded[name] for name in body_sensors if name in encoded]
            body_latent = self.body_hub(jnp.concatenate(body_inputs, axis=-1))
            
            # Phase 3: Global Association
            assoc_sensors = ["Olfaction", "Location", "Visual", "Proprioception"]
            assoc_inputs = [encoded[name] for name in assoc_sensors if name in encoded]
            assoc_inputs.append(body_latent)
            return jax.nn.relu(self.assoc_hub(jnp.concatenate(assoc_inputs, axis=-1)))
            
        else:
            return jax.nn.relu(self.monolith(x))


class ActorCriticRNN(nnx.Module):
    """Actor-Critic RNN (LSTM or GRU) using Flax NNX, with optional neuromodulation.

    When modulation_config is None, this behaves identically to the original
    unmodulated network (true baseline control, §5.2).

    Supports two perceptual modulation styles via modulation_config['type']:
    - "Multiplicative": Post-activation gating (Ben-Iwhiwhu style, original).
    - "PreActivation": Pre-activation gain + threshold shift (Ferguson & Cardin style).
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int, rngs: nnx.Rngs,
                 rnn_type: str = "LSTM", activation: str = "tanh",
                 modulation_config: Optional[dict] = None,
                 observation_breakdown: Optional[dict] = None,
                 encoding_config: Optional[dict] = None):
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.rnn_type = rnn_type.upper()
        self.activation = activation.lower()
        self.modulation_enabled = modulation_config is not None and modulation_config.get('type') is not None
        self.modulation_type = modulation_config['type'] if self.modulation_enabled else None

        # Observation Encoding (Flat or Hierarchical)
        if observation_breakdown is None:
            # Fallback for compatibility or if breakdown not provided
            observation_breakdown = {"Observation": input_dim}
            
        self.obs_encoder = ObservationEncoder(
            input_dim, hidden_size, observation_breakdown, encoding_config, rngs
        )

        # RNN Layer (LSTM or ModulatedGRU)
        if self.rnn_type == "LSTM":
            self.rnn_cell = nnx.LSTMCell(hidden_size, hidden_size, rngs=rngs)
        else:
            if self.modulation_enabled:
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
            mod_hidden = modulation_config['mod_hidden_size']
            mod_type = modulation_config['type']
            grouping = modulation_config['grouping_size']
            percept_bias = modulation_config['percept_bias_init']
            percept_add_bias = modulation_config.get('percept_add_bias_init', 0.0)
            memory_bias = modulation_config['memory_bias_init']
            temp_clip = tuple(modulation_config['temp_clip'])

            self.modulator = NeuromodulatorRNN(
                input_dim=input_dim,
                target_hidden_size=hidden_size,
                mod_hidden_size=mod_hidden,
                modulation_type=mod_type,
                grouping_size=grouping,
                percept_bias_init=percept_bias,
                percept_add_bias_init=percept_add_bias,
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
            task_h, mod_h = h

            # --- Modulator forward pass ---
            mod_output, mod_h_new = self.modulator(x, mod_h)

            # --- Task path with modulation ---
            # INJECTION A: Perceptual modulation (type-dependent)
            if self.modulation_type == "PreActivation":
                # Ferguson & Cardin style: gain + threshold shift INSIDE activation
                x_linear = self.obs_encoder(x)
                gamma = jax.nn.sigmoid(mod_output.z_percept)
                beta = mod_output.z_percept_add
                x_proj = jax.nn.relu(x_linear * gamma + beta)
            else:
                # Multiplicative (original): post-activation gating
                x_proj = self.obs_encoder(x)
                x_proj = x_proj * jax.nn.sigmoid(mod_output.z_percept)

            # Layer 2: RNN forward with gate-bias injection
            if self.rnn_type == "LSTM":
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

            h_combined_new = (h_new, mod_h_new)
            return logits, value, h_combined_new, mod_output

        else:
            # --- Original unmodulated path (exact baseline) ---
            x_proj = self.obs_encoder(x)

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
