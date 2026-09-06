import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Union, Optional, Any

from .modulated_gru_cell import ModulatedGRUCell
from .neuromodulator import NeuromodulatorRNN



def _mod_required(cfg: dict, key: str):
    """Mirror Config.get_mandatory for a plain modulation dict.

    ActorCriticRNN is built from a hand-assembled dict at several call sites
    (save_snapshot.py, and historically evaluation.py — see Finding A / 2ad9104),
    so a missing key must raise a NAMED ValueError here rather than a bare
    KeyError deep in the modulator.
    """
    if key not in cfg or cfg[key] is None:
        raise ValueError(
            f"Strict Config: modulation key '{key}' is required but missing. "
            f"See docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md"
        )
    return cfg[key]


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

    def __call__(self, x, unimodal_ln=None, multimodal_ln=None, flat_ln=None):
        """Standard forward pass (no modulation). Optional LayerNorm on pre-activations."""
        if self.mode != 'hierarchical':
            x_proj = self.monolith(x)
            if flat_ln is not None:
                x_proj = flat_ln(x_proj)
            return jax.nn.relu(x_proj)
        
        batch_shape = x.shape[:-1]
        x_padded = jnp.zeros(batch_shape + (len(self.names), self.max_in), dtype=x.dtype)
        start = 0
        for i, (name, dim) in enumerate(self.breakdown.items()):
            x_padded = x_padded.at[..., i, :dim].set(x[..., start : start + dim])
            start += dim

        # Phase 1: Grouped encoding
        encoded_all = self.unimodal_grouped(x_padded)
        if unimodal_ln is not None:
            encoded_all = jax.vmap(unimodal_ln, in_axes=-2, out_axes=-2)(encoded_all)
        encoded_all = jax.nn.relu(encoded_all)

        # Phase 2: Multimodal Hub
        mm_in = encoded_all.reshape(batch_shape + (-1,))
        mm_latent = self.multimodal_hub(mm_in)
        if multimodal_ln is not None:
            mm_latent = multimodal_ln(mm_latent)
        return jax.nn.relu(mm_latent)

    def forward_with_modulation(self, x, mod_output, modulation_type: str,
                                unimodal_ln=None, multimodal_ln=None,
                                flat_ln=None):
        """Hierarchical forward pass with multi-stage modulation (Injection A)."""
        # --- Flat mode ---
        if self.mode != 'hierarchical':
            x_proj = self.monolith(x)
            if flat_ln is not None:
                x_proj = flat_ln(x_proj)
            if modulation_type == "PreActivation":
                gamma = jax.nn.sigmoid(mod_output.z_unimodal)
                beta = mod_output.z_unimodal_add
                return jax.nn.relu(x_proj * gamma + beta)
            elif modulation_type == "FiLM":
                return jax.nn.relu(mod_output.z_unimodal * x_proj + mod_output.z_unimodal_add)
            else:  # Multiplicative
                return jax.nn.relu(x_proj) * jax.nn.sigmoid(mod_output.z_unimodal)

        # --- Hierarchical mode ---
        batch_shape = x.shape[:-1]
        x_padded = jnp.zeros(batch_shape + (len(self.names), self.max_in), dtype=x.dtype)
        start = 0
        for i, (name, dim) in enumerate(self.breakdown.items()):
            x_padded = x_padded.at[..., i, :dim].set(x[..., start : start + dim])
            start += dim

        # Phase 1: Unimodal + Modulation
        encoded_all = self.unimodal_grouped(x_padded)
        if unimodal_ln is not None:
            encoded_all = jax.vmap(unimodal_ln, in_axes=-2, out_axes=-2)(encoded_all)

        if modulation_type == "PreActivation":
            gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)
            beta1 = mod_output.z_unimodal_add
            encoded_all = jax.nn.relu(encoded_all * gamma1[..., None, :] + beta1[..., None, :])
        elif modulation_type == "FiLM":
            gamma1 = mod_output.z_unimodal
            beta1 = mod_output.z_unimodal_add
            encoded_all = jax.nn.relu(gamma1[..., None, :] * encoded_all + beta1[..., None, :])
        else:  # Multiplicative
            gamma1 = jax.nn.sigmoid(mod_output.z_unimodal)
            encoded_all = jax.nn.relu(encoded_all) * gamma1[..., None, :]

        # Phase 2: Multimodal Hub + Modulation
        mm_in = encoded_all.reshape(batch_shape + (-1,))
        mm_latent = self.multimodal_hub(mm_in)
        if multimodal_ln is not None:
            mm_latent = multimodal_ln(mm_latent)

        if modulation_type == "PreActivation":
            gamma2 = jax.nn.sigmoid(mod_output.z_multimodal)
            beta2 = mod_output.z_multimodal_add
            return jax.nn.relu(mm_latent * gamma2 + beta2)
        elif modulation_type == "FiLM":
            gamma2 = mod_output.z_multimodal
            beta2 = mod_output.z_multimodal_add
            return jax.nn.relu(gamma2 * mm_latent + beta2)
        else:  # Multiplicative
            gamma2 = jax.nn.sigmoid(mod_output.z_multimodal)
            return jax.nn.relu(mm_latent) * gamma2


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

        # === Modulation site / mechanism / temperature validation (plan A) ===
        # This lives HERE, in the one place all model-construction call sites funnel
        # through, so a hand-built modulation dict at any call site fails with a clear
        # named ValueError instead of a bare KeyError deep inside the modulator —
        # which is exactly how Finding A (2ad9104) escaped to a second site.
        if self.modulation_enabled:
            # --- A3 migration guard: the flat temp_clip key was replaced by
            # temperature.{enabled, clip}. The message must be self-explaining for the
            # TWO distinct readers who hit it (D16): someone editing a live config, and
            # someone re-evaluating an ARCHIVED run from its saved models/config.yaml.
            if 'temp_clip' in modulation_config:
                raise ValueError(
                    "modulation.temp_clip is no longer supported: it was replaced by "
                    "modulation.temperature.{enabled, clip} in the modulation-site refactor "
                    "(2026-08-31). Temperature modulation is now OPT-IN and defaults to disabled.\n"
                    "\n"
                    "If you are EDITING A LIVE CONFIG, replace the flat key with:\n"
                    "    temperature:\n      enabled: true\n      clip: <the old temp_clip value>\n"
                    "(plus the now-mandatory `sites:` block and `rnn_mechanism:`).\n"
                    "\n"
                    "If you are RE-EVALUATING AN ARCHIVED RUN, this config was saved BEFORE the "
                    "modulation-site refactor, so the architecture it describes no longer exists in "
                    "this code. Its checkpoint cannot be restored, and results produced from it would "
                    "not be comparable with either its own original results or any post-refactor run. "
                    "The correct response is to RE-TRAIN under the current architecture, NOT to "
                    "re-evaluate this checkpoint. This refusal is intentional (plan decision D16).\n"
                    "\n"
                    "See docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md."
                )

            sites = _mod_required(modulation_config, 'sites')
            for _s in ('encoder', 'rnn', 'actor', 'critic'):
                if _s not in sites or sites[_s] is None:
                    raise ValueError(f"Strict Config: modulation.sites.{_s} is required but missing.")
            # Four SEPARATE scalar bools, never a dict attribute: a bare bool on an
            # nnx.Module lands in the graphdef (the static half of the jit cache key),
            # so it is resolved once at trace time and never becomes a traced leaf.
            self.site_encoder = bool(sites['encoder'])
            self.site_rnn     = bool(sites['rnn'])
            self.site_actor   = bool(sites['actor'])
            self.site_critic  = bool(sites['critic'])

            self.rnn_mechanism = _mod_required(modulation_config, 'rnn_mechanism')
            if self.rnn_mechanism not in ("activation", "gate_bias"):   # D12 — explicit whitelist
                raise ValueError(
                    f"modulation.rnn_mechanism must be 'activation' or 'gate_bias', "
                    f"got {self.rnn_mechanism!r}."
                )

            _temp = _mod_required(modulation_config, 'temperature')
            self.temperature_enabled = bool(_mod_required(_temp, 'enabled'))
            temp_clip = tuple(_mod_required(_temp, 'clip')) if self.temperature_enabled else None

            if not (self.site_encoder or self.site_rnn or self.site_actor
                    or self.site_critic or self.temperature_enabled):    # D13 — no dead modulator
                raise ValueError(
                    "modulation.type is set but every site is disabled and temperature is off: "
                    "the modulator GRU would run every step and receive no gradient (dead weight). "
                    "Enable at least one of sites.{encoder,rnn,actor,critic} or temperature.enabled."
                )
        else:
            self.site_encoder = False
            self.site_rnn = False
            self.site_actor = False
            self.site_critic = False
            self.rnn_mechanism = None
            self.temperature_enabled = False
            temp_clip = None

        # ModulatedGRUCell differs from nnx.GRUCell in init scheme and gate polarity
        # (KNOWN_BUGS: modulated/baseline init confound). Only pay that cost when the
        # gate-bias mechanism actually needs it (D6).
        self._uses_gate_bias = (self.modulation_enabled and self.site_rnn
                                and self.rnn_mechanism == "gate_bias")

        if self.modulation_enabled and self.modulation_type == "FiLMNoNorm":
            raise ValueError(
                "modulation.type='FiLMNoNorm' has been removed. "
                "Use type='FiLM' with use_layer_norm=false instead."
            )

        if self.modulation_enabled and self.rnn_type == "LSTM":
            raise ValueError(
                "modulation with rnn_type='LSTM' is unsupported: Injection B "
                "(memory gate bias) requires the GRU cell (ModulatedGRUCell); "
                "nnx.LSTMCell has no gate_bias input, so the modulator's z_memory "
                "head would be silently dropped (dead weight). Use rnn_type: \"GRU\"."
            )

        # Observation Encoding (Flat or Hierarchical)
        if observation_breakdown is None:
            # Fallback for compatibility or if breakdown not provided
            observation_breakdown = {"Observation": input_dim}
            
        self.obs_encoder = ObservationEncoder(
            input_dim, hidden_size, observation_breakdown, encoding_config, rngs
        )

        # LayerNorm on encoder pre-activations (orthogonal to modulation type)
        # Read from agent-level config, not modulation config
        self.use_layer_norm = encoding_config['use_layer_norm']  # mandatory — no fallback default
        if self.use_layer_norm:
            if self.obs_encoder.mode == 'hierarchical':
                self.mod_unimodal_ln = nnx.LayerNorm(hidden_size, rngs=rngs)
                self.mod_multimodal_ln = nnx.LayerNorm(hidden_size, rngs=rngs)
            else:  # flat mode
                self.mod_flat_ln = nnx.LayerNorm(hidden_size, rngs=rngs)

        # RNN Layer (LSTM or ModulatedGRU)
        if self.rnn_type == "LSTM":
            self.rnn_cell = nnx.LSTMCell(hidden_size, hidden_size, rngs=rngs)
        else:
            if self._uses_gate_bias:
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
            memory_clip = tuple(modulation_config['memory_clip'])

            # D3 — one global grouping_size is only valid because every modulation
            # target in this network is exactly `hidden_size` wide. Assert it, so a
            # future width change fails loudly instead of mis-broadcasting.
            _widths = {
                'encoder_unimodal': self.obs_encoder.hidden_size,
                'actor_fc1': self.actor_fc1.out_features,
                'critic_fc1': self.critic_fc1.out_features,
                'task_rnn': hidden_size,
            }
            _bad = {k: v for k, v in _widths.items() if v != hidden_size}
            if _bad:
                raise ValueError(
                    f"modulation assumes every FiLM target is hidden_size={hidden_size} wide "
                    f"so a single grouping_size applies to all of them, but these differ: "
                    f"{_bad}. Add a per-site grouping_size before allowing this."
                )

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
                memory_clip=memory_clip,
                sites={'encoder': self.site_encoder, 'rnn': self.site_rnn,
                       'actor': self.site_actor, 'critic': self.site_critic},
                rnn_mechanism=self.rnn_mechanism,
                temperature_enabled=self.temperature_enabled,
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
        # Compress unbounded modalities (olfaction ~40, visual ~13) to ~[0, 3.7] range.
        # Mirrors DreamerV3's global symlog applied in its trainer — applied here
        # at the network boundary so the environment stays agent-agnostic.
        x = jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)

        if self.modulation_enabled:
            task_h, mod_h = h

            # --- Modulator forward pass ---
            mod_output, mod_h_new = self.modulator(x, mod_h)

            # --- Task path with modulation ---
            # SITE: encoder — perceptual modulation (Phase-dependent)
            if self.site_encoder:
                x_proj = self.obs_encoder.forward_with_modulation(
                    x, mod_output, self.modulation_type,
                    unimodal_ln=getattr(self, 'mod_unimodal_ln', None),
                    multimodal_ln=getattr(self, 'mod_multimodal_ln', None),
                    flat_ln=getattr(self, 'mod_flat_ln', None)
                )
            else:
                x_proj = self.obs_encoder(
                    x,
                    unimodal_ln=getattr(self, 'mod_unimodal_ln', None),
                    multimodal_ln=getattr(self, 'mod_multimodal_ln', None),
                    flat_ln=getattr(self, 'mod_flat_ln', None)
                )

            # SITE: rnn — either the legacy internal gate bias (§5.1a) or FiLM on
            # the cell's emitted output.
            if self.rnn_type == "LSTM":
                h_new, x_h = self.rnn_cell(task_h, x_proj)
            elif self._uses_gate_bias:
                h_new, x_h = self.rnn_cell(task_h, x_proj, gate_bias=mod_output.z_memory)
            else:
                h_new, x_h = self.rnn_cell(task_h, x_proj)
                if self.site_rnn:      # rnn_mechanism == "activation"
                    # D5: modulate the EMITTED output only. `h_new` (the carry) is
                    # deliberately left untouched — scaling the carry re-creates the
                    # double-gating pathology that §5.1 flags as Critical. No
                    # nonlinearity: x_h is already an activation.
                    x_h = mod_output.z_rnn * x_h + mod_output.z_rnn_add

            # SITE: actor — pre-activation FiLM on the single hidden layer
            a_pre = self.actor_fc1(x_h)
            if self.site_actor:
                a_pre = mod_output.z_actor * a_pre + mod_output.z_actor_add
            a_h = self._activate(a_pre)
            logits = self.actor_fc2(a_h)

            # INJECTION C: Bounded Temperature (§5.3) — now opt-in (A3)
            if self.temperature_enabled:
                logits = logits / mod_output.temperature

            # SITE: critic — pre-activation FiLM on the single hidden layer
            c_pre = self.critic_fc1(x_h)
            if self.site_critic:
                c_pre = mod_output.z_critic * c_pre + mod_output.z_critic_add
            c_h = self._activate(c_pre)
            value = self.critic_fc2(c_h)

            h_combined_new = (h_new, mod_h_new)
            return logits, value, h_combined_new, mod_output

        else:
            # --- Original unmodulated path (exact baseline) ---
            x_proj = self.obs_encoder(
                x,
                unimodal_ln=getattr(self, 'mod_unimodal_ln', None),
                multimodal_ln=getattr(self, 'mod_multimodal_ln', None),
                flat_ln=getattr(self, 'mod_flat_ln', None)
            )

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
