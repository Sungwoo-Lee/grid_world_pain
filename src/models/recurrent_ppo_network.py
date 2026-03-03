import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Union, Optional, Any

from .modulated_gru_cell import ModulatedGRUCell
from .neuromodulator import NeuromodulatorRNN



class GroupedLinear(nnx.Module):
    """
    Applies independent linear layers to N groups in parallel using einsum.
    Weights shape: [num_groups, in_features, out_features]
    """
    def __init__(self, num_groups: int, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias for each group independently
        w_key = rngs.params()
        self.weights = nnx.Param(jax.random.normal(w_key, (num_groups, in_features, out_features)) * 0.1)
        b_key = rngs.params()
        self.bias = nnx.Param(jnp.zeros((num_groups, out_features)))

    def __call__(self, x):
        # x shape: [..., G, I], weights shape: [G, I, O]
        # output shape: [..., G, O]
        return jnp.einsum('...gi,gio->...go', x, self.weights) + self.bias


class GroupedMLP(nnx.Module):
    """
    Applies independent MLPs to a collection of sensory inputs in parallel.
    Uses GroupedLinear to ensure a single kernel launch for each Phase 1 layer.
    """
    def __init__(self, num_groups: int, max_in: int, hidden_layers: list, output_dim: int, rngs: nnx.Rngs):
        self.num_groups = num_groups
        self.max_in = max_in
        
        layers = []
        in_d = max_in
        for h in hidden_layers:
            layers.append(GroupedLinear(num_groups, in_d, h, rngs=rngs))
            layers.append(jax.nn.relu)
            in_d = h
            
        layers.append(GroupedLinear(num_groups, in_d, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x_padded):
        return self.net(x_padded)


class MLP(nnx.Module):
    """Simple Multi-Layer Perceptron helper for non-grouped paths (hubs)."""
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
    """Unified observation encoder supporting flat and hierarchical modes with grouped processing."""
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
            
            # 1. Grouped Unimodal Encoders
            self.names = list(breakdown.keys())
            input_dims = [breakdown[name] for name in self.names]
            self.max_in = max(input_dims)
            # Use grouped MLP structure
            self.unimodal_grouped = GroupedMLP(len(self.names), self.max_in, default_mlp, hidden_size, rngs=rngs)
            
            # 2. Multimodal Hub
            multimodal_in_dim = len(self.names) * hidden_size
            multimodal_mlp_struct = h_params.get('multimodal_hub', default_mlp)
            self.multimodal_hub = MLP(multimodal_in_dim, multimodal_mlp_struct, hidden_size, rngs=rngs)
            
        else:
            self.monolith = nnx.Linear(input_dim, hidden_size, rngs=rngs)

    def __call__(self, x):
        """Standard forward pass (no modulation)."""
        if self.mode != 'hierarchical':
            return jax.nn.relu(self.monolith(x))
        
        batch_shape = x.shape[:-1]
        x_padded = jnp.zeros(batch_shape + (len(self.names), self.max_in), dtype=x.dtype)
        start = 0
        for i, (name, dim) in enumerate(self.breakdown.items()):
            x_padded = x_padded.at[..., i, :dim].set(x[..., start : start + dim])
            start += dim

        # Phase 1: Grouped encoding
        encoded_all = jax.nn.relu(self.unimodal_grouped(x_padded))

        # Phase 2: Multimodal Hub
        mm_in = encoded_all.reshape(batch_shape + (-1,))
        return jax.nn.relu(self.multimodal_hub(mm_in))

    def forward_with_modulation(self, x, mod_output, modulation_type: str):
        """Hierarchical forward pass with multi-stage modulation (Injection A)."""
        if self.mode != 'hierarchical':
            x_proj = self.monolith(x)
            if modulation_type == "PreActivation":
                gamma = jax.nn.sigmoid(mod_output.z_unimodal[..., 0:1]) # Fallback to first group or similar
                beta = mod_output.z_unimodal_add[..., 0:1]
                return jax.nn.relu(x_proj * gamma + beta)
            else:
                return jax.nn.relu(x_proj) * jax.nn.sigmoid(mod_output.z_unimodal[..., 0:1])

        batch_shape = x.shape[:-1]
        x_padded = jnp.zeros(batch_shape + (len(self.names), self.max_in), dtype=x.dtype)
        start = 0
        for i, (name, dim) in enumerate(self.breakdown.items()):
            x_padded = x_padded.at[..., i, :dim].set(x[..., start : start + dim])
            start += dim

        # Phase 1: Unimodal + Modulation (z_unimodal)
        encoded_all = self.unimodal_grouped(x_padded)
        gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)
        beta1 = mod_output.z_unimodal_add
        # Apply per-group modulation
        encoded_all = jax.nn.relu(encoded_all * gamma1[..., None] + beta1[..., None])

        # Phase 2: Multimodal Hub + Modulation (z_multimodal)
        mm_in = encoded_all.reshape(batch_shape + (-1,))
        mm_latent = self.multimodal_hub(mm_in)
        gamma2 = jax.nn.sigmoid(mod_output.z_multimodal)
        beta2 = mod_output.z_multimodal_add
        return jax.nn.relu(mm_latent * gamma2 + beta2)


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
                obs_dim=input_dim,
                target_hidden_size=hidden_size,
                obs_breakdown=observation_breakdown,
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
            # INJECTION A: Perceptual modulation (Phase-dependent)
            x_proj = self.obs_encoder.forward_with_modulation(x, mod_output, self.modulation_type)

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
