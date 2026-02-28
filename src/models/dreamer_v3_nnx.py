import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
from typing import Tuple, Dict, Any, Optional

from src.models.dreamer_v3_util import symlog, symexp, to_twohot, from_twohot, OneHotDist, hafner_init
from src.models.modulated_layer_norm_gru_cell import ModulatedLayerNormGRUCell

# -----------------------------------------------------------------------------
# Core Modules
# -----------------------------------------------------------------------------

class SiLU(nnx.Module):
    def __call__(self, x):
        return jax.nn.silu(x)

class LayerNormGRUCell(nnx.Module):
    def __init__(self, hidden_size: int, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.dense_ih = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, kernel_init=hafner_init(), rngs=rngs)
        self.dense_hh = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, kernel_init=hafner_init(), rngs=rngs)

        self.ln_ih = nnx.LayerNorm(3 * hidden_size, rngs=rngs)
        self.ln_hh = nnx.LayerNorm(3 * hidden_size, rngs=rngs)

    def __call__(self, x, h):
        gates_ih = self.ln_ih(self.dense_ih(x))
        gates_hh = self.ln_hh(self.dense_hh(h))
        gates = gates_ih + gates_hh

        reset, update, cand = jnp.split(gates, 3, axis=-1)

        reset = nnx.sigmoid(reset)
        update = nnx.sigmoid(update)
        cand = jnp.tanh(cand)

        h_new = (1 - update) * h + update * cand
        return h_new

class RSSM(nnx.Module):
    def __init__(self, action_dim: int, deter_dim: int, stoch_dim: int,
                 discrete: int, embed_dim: int,
                 modulation_enabled: bool, rngs: nnx.Rngs):
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.discrete = discrete
        self.action_dim = action_dim
        self.modulation_enabled = modulation_enabled

        self.img_in = nnx.Linear(stoch_dim * discrete + action_dim, deter_dim, kernel_init=hafner_init(), rngs=rngs)

        if modulation_enabled:
            self.cell = ModulatedLayerNormGRUCell(deter_dim, rngs=rngs)
        else:
            self.cell = LayerNormGRUCell(deter_dim, rngs=rngs)

        self.img_out = nnx.Linear(deter_dim, stoch_dim * discrete, kernel_init=hafner_init(), rngs=rngs)
        self.obs_out = nnx.Linear(deter_dim + embed_dim, stoch_dim * discrete, kernel_init=hafner_init(), rngs=rngs)

    def initial(self, batch_size: int):
        return {
            'deter': jnp.zeros((batch_size, self.deter_dim)),
            'stoch': jnp.zeros((batch_size, self.stoch_dim * self.discrete)),
            'logits': jnp.zeros((batch_size, self.stoch_dim, self.discrete)),
            'prev_action': jnp.zeros((batch_size, self.action_dim))
        }

    def step(self, prev_state, embed, action, is_first, key, gate_bias=None):
        """Single step transition with optional neuromodulation.

        Args:
            prev_state: RSSM state dict.
            embed: Encoded observation, shape (B, embed_dim).
            action: One-hot action, shape (B, action_dim).
            is_first: Episode boundary flag, shape (B,) or (B, 1).
            key: PRNG key for stochastic sampling.
            gate_bias: Optional gate-bias from modulator for GRU update gate,
                       shape (B, deter_dim). None = no modulation.
        """
        mask = (1.0 - is_first).astype(jnp.float32).reshape((-1, 1))
        deter = prev_state['deter'] * mask
        stoch = prev_state['stoch'] * mask

        x = jnp.concatenate([stoch, action], axis=-1)
        x = self.img_in(x)
        x = jax.nn.silu(x)

        if self.modulation_enabled and gate_bias is not None:
            deter = self.cell(x, deter, gate_bias=gate_bias)
        else:
            deter = self.cell(x, deter)

        prior_logits = self.img_out(deter)
        prior_logits = prior_logits.reshape(prior_logits.shape[:-1] + (self.stoch_dim, self.discrete))

        post_input = jnp.concatenate([deter, embed], axis=-1)
        post_logits = self.obs_out(post_input)
        post_logits = post_logits.reshape(post_logits.shape[:-1] + (self.stoch_dim, self.discrete))

        dist = OneHotDist(post_logits)
        stoch = dist.sample(key)
        stoch = stoch.reshape(stoch.shape[:-2] + (-1,))

        post = {'deter': deter, 'stoch': stoch, 'logits': post_logits, 'prev_action': action}
        prior = {'deter': deter, 'stoch': None, 'logits': prior_logits, 'prev_action': action}

        return post, prior

    def imagine_step(self, prev_state, action, key, gate_bias=None):
        """Dynamics-only step with optional neuromodulation (for imagination).

        Args:
            prev_state: RSSM state dict.
            action: One-hot action, shape (B, action_dim).
            key: PRNG key for stochastic sampling.
            gate_bias: Optional gate-bias from modulator, shape (B, deter_dim).
        """
        deter = prev_state['deter']
        stoch = prev_state['stoch']

        x = jnp.concatenate([stoch, action], axis=-1)
        x = self.img_in(x)
        x = jax.nn.silu(x)

        if self.modulation_enabled and gate_bias is not None:
            deter = self.cell(x, deter, gate_bias=gate_bias)
        else:
            deter = self.cell(x, deter)

        prior_logits = self.img_out(deter)
        prior_logits = prior_logits.reshape(prior_logits.shape[:-1] + (self.stoch_dim, self.discrete))

        dist = OneHotDist(prior_logits)
        stoch = dist.sample(key)
        stoch = stoch.reshape(stoch.shape[:-2] + (-1,))

        prior = {'deter': deter, 'stoch': stoch, 'logits': prior_logits, 'prev_action': action}
        return prior

class DreamerGroupedLinear(nnx.Module):
    """
    Applies independent linear layers to N groups in parallel using einsum.
    Uses hafner_init for parameters to match DreamerV3 standards.
    """
    def __init__(self, num_groups: int, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias for each group independently
        w_key = rngs.params()
        init_fn = hafner_init()
        # We need to apply init_fn per group or handle the 3D shape correctly.
        # hafner_init calculates fan_in from the first dimension of the shape.
        # For [G, I, O], fan_in should be I.
        # Our hafner_init: fan_in = shape[0] if len(shape) > 0 else 1.
        # This will use G as fan_in, which is WRONG.
        # We'll manually implement the init here to be safe and correct.
        stddev = 0.8796 / jnp.sqrt(in_features)
        self.weights = nnx.Param(jax.random.truncated_normal(w_key, -2.0, 2.0, (num_groups, in_features, out_features)) * stddev)
        
        b_key = rngs.params()
        self.bias = nnx.Param(jnp.zeros((num_groups, out_features)))

    def __call__(self, x):
        # x shape: [..., G, I], weights shape: [G, I, O]
        return jnp.einsum('...gi,gio->...go', x, self.weights) + self.bias


class DreamerGroupedMLP(nnx.Module):
    """
    DreamerV3-compatible grouped MLP with LayerNorm and SiLU.
    """
    def __init__(self, num_groups: int, max_in: int, hidden_layers: list, output_dim: int, rngs: nnx.Rngs):
        self.num_groups = num_groups
        
        layers = []
        in_d = max_in
        for h in hidden_layers:
            layers.append(DreamerGroupedLinear(num_groups, in_d, h, rngs=rngs))
            # nnx.vmap(nnx.LayerNorm) would be ideal, but einsum is already handled.
            # LayerNorm over the feature dimension [..., G, H]
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
            
        layers.append(DreamerGroupedLinear(num_groups, in_d, output_dim, rngs=rngs))
        layers.append(nnx.LayerNorm(output_dim, rngs=rngs))
        # Note: No final SiLU here, similar to Encoder.body
        self.net = nnx.Sequential(*layers)

    def __call__(self, x_padded):
        return self.net(x_padded)


class DreamerObservationEncoder(nnx.Module):
    """
    Hierarchical observation encoder for DreamerV3.
    Replicates the Grouped Encoding optimization from PPO but with Dreamer components.
    """
    def __init__(self, input_dim: int, embed_dim: int, breakdown: dict,
                 config: Optional[dict], rngs: nnx.Rngs):
        if config is None:
            raise ValueError("Strict Config: Hierarchical config is required for DreamerObservationEncoder.")
            
        self.mode = config['encoding_mode']
        self.breakdown = breakdown
        self.embed_dim = embed_dim
        
        if self.mode == 'hierarchical':
            if 'hierarchical_params' not in config:
                raise ValueError("Strict Config: 'hierarchical_params' is required when encoding_mode is 'hierarchical'.")
            h_params = config['hierarchical_params']
            default_mlp = h_params['default_mlp']
            # We use embed_dim as the hidden_size for hubs to match Dreamer's bottleneck
            hidden_size = embed_dim 
            
            # 1. Grouped Unimodal Encoders
            self.names = list(breakdown.keys())
            input_dims = [breakdown[name] for name in self.names]
            self.max_in = max(input_dims)
            self.unimodal_grouped = DreamerGroupedMLP(len(self.names), self.max_in, default_mlp, hidden_size, rngs=rngs)
            
            # 2. Body-State Hub
            body_sensors = ["Satiation", "Nutrition", "Injury", "Extero Nociception", "Collision"]
            self.body_indices = [i for i, name in enumerate(self.names) if name in body_sensors]
            body_in_dim = len(self.body_indices) * hidden_size
            body_mlp_struct = h_params.get('hub_overrides', {}).get('body_state', default_mlp)
            self.body_hub = MLP(body_in_dim, hidden_size, body_mlp_struct, rngs=rngs)
            
            # 3. Association Hub
            assoc_sensors = ["Olfaction", "Location", "Visual", "Proprioception"]
            self.assoc_indices = [i for i, name in enumerate(self.names) if name in assoc_sensors]
            assoc_in_dim = (len(self.assoc_indices) * hidden_size) + hidden_size
            assoc_mlp_struct = h_params.get('hub_overrides', {}).get('association', default_mlp)
            self.assoc_hub = MLP(assoc_in_dim, embed_dim, assoc_mlp_struct, rngs=rngs)
            
            self.final_act = SiLU()
        else:
            # Replicate standard Encoder behavior
            fc_layers = config['encoder_fc_layers']
            self.flat_encoder = Encoder(input_dim, embed_dim, fc_layers, rngs=rngs)

    def __call__(self, x):
        if self.mode == 'hierarchical':
            return self.final_act(self._forward_body(x))
        else:
            return self.flat_encoder(x)

    def _forward_body(self, x):
        """Processes the observation through the hierarchy (pre-activation)."""
        batch_shape = x.shape[:-1]
        x_padded = jnp.zeros(batch_shape + (len(self.names), self.max_in), dtype=x.dtype)
        
        start = 0
        for i, (name, dim) in enumerate(self.breakdown.items()):
            x_padded = x_padded.at[..., i, :dim].set(x[..., start : start + dim])
            start += dim
            
        # Phase 1: Grouped encoding
        encoded_all = self.unimodal_grouped(x_padded)
        
        # Phase 2: Body-State Sub-Fusion
        body_inputs = encoded_all[..., self.body_indices, :]
        body_in = body_inputs.reshape(batch_shape + (-1,))
        body_latent = self.body_hub(body_in)
        
        # Phase 3: Global Association
        assoc_inputs = encoded_all[..., self.assoc_indices, :]
        assoc_in = assoc_inputs.reshape(batch_shape + (-1,))
        assoc_in = jnp.concatenate([assoc_in, body_latent], axis=-1)
        return self.assoc_hub(assoc_in)

    def forward_with_modulation(self, x, mod_output, modulation_type: str):
        if self.mode != 'hierarchical':
            return self.flat_encoder.forward_with_modulation(x, mod_output, modulation_type)
            
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
        encoded_all = SiLU()(encoded_all * gamma1[..., None] + beta1[..., None])

        # Phase 2: Body-State Hub + Modulation (z_bodystate)
        body_inputs = encoded_all[..., self.body_indices, :]
        body_in = body_inputs.reshape(batch_shape + (-1,))
        body_latent = self.body_hub(body_in)
        gamma2 = jax.nn.sigmoid(mod_output.z_bodystate)
        beta2 = mod_output.z_bodystate_add
        body_latent = body_latent * gamma2 + beta2

        # Phase 3: Association Hub + Modulation (z_association)
        assoc_inputs = encoded_all[..., self.assoc_indices, :]
        assoc_in = assoc_inputs.reshape(batch_shape + (-1,))
        assoc_in = jnp.concatenate([assoc_in, body_latent], axis=-1)
        assoc_latent = self.assoc_hub(assoc_in)
        gamma3 = jax.nn.sigmoid(mod_output.z_association)
        beta3 = mod_output.z_association_add
        return self.final_act(assoc_latent * gamma3 + beta3)


class Encoder(nnx.Module):
    def __init__(self, input_dim: int, embed_dim: int, fc_layers: list, rngs: nnx.Rngs):
        """
        Configurable Encoder matching PyTorch architecture.

        The body produces a pre-activation embedding, and final_act applies SiLU.
        This split allows neuromodulation (Injection A) to be applied between
        the body and final activation (PreActivation style).

        Args:
            input_dim: Input observation dimension
            embed_dim: Output embedding dimension
            fc_layers: List of hidden layer sizes (e.g., [128, 128, 128, 128, 128])
        """
        self.embed_dim = embed_dim

        body_layers = []
        in_d = input_dim
        for h in fc_layers:
            body_layers.append(nnx.Linear(in_d, h, kernel_init=hafner_init(), rngs=rngs))
            body_layers.append(nnx.LayerNorm(h, rngs=rngs))
            body_layers.append(SiLU())
            in_d = h
        body_layers.append(nnx.Linear(in_d, embed_dim, kernel_init=hafner_init(), rngs=rngs))
        body_layers.append(nnx.LayerNorm(embed_dim, rngs=rngs))
        self.body = nnx.Sequential(*body_layers)

        # Final activation (separate for modulation injection point)
        self.final_act = SiLU()

    def __call__(self, x):
        """Standard forward pass (no modulation)."""
        return self.final_act(self.body(x))

    def forward_with_modulation(self, x, mod_output, modulation_type: str):
        """Forward pass with neuromodulation (Injection A).

        Applies modulation between body (pre-activation) and final activation.

        Args:
            x: Observation vector, shape (..., input_dim).
            mod_output: DreamerModulatorOutput with hierarchical fields.
            modulation_type: "Multiplicative" or "PreActivation".

        Returns:
            Modulated embedding, shape (..., embed_dim).
        """
        x_pre = self.body(x)

        # Flat encoder fallback: just use z_unimodal's first dimension
        if modulation_type == "PreActivation":
            gamma = jax.nn.sigmoid(mod_output.z_unimodal[..., 0:1])
            beta = mod_output.z_unimodal_add[..., 0:1]
            return self.final_act(x_pre * gamma + beta)
        else:
            return self.final_act(x_pre) * jax.nn.sigmoid(mod_output.z_unimodal[..., 0:1])

class DreamerObservationDecoder(nnx.Module):
    """
    Symmetric observation decoder for DreamerV3.
    Matches the structure of DreamerObservationEncoder, handling both 
    flat and hierarchical modes.
    """
    def __init__(self, feat_dim: int, obs_dim: int, breakdown: dict,
                 config: dict, rngs: nnx.Rngs):
        self.mode = config['encoding_mode']
        self.breakdown = breakdown
        self.obs_dim = obs_dim
        
        if self.mode == 'hierarchical':
            h_params = config['hierarchical_params']
            default_mlp = h_params['default_mlp']
            hidden_size = config['encoder_dim'] # hidden_size per sensor
            
            self.names = list(breakdown.keys())
            self.sensor_dims = [breakdown[name] for name in self.names]
            self.max_out = max(self.sensor_dims)
            
            # 1. Global Association Decoder (Phase 1)
            assoc_sensors = ["Olfaction", "Location", "Visual", "Proprioception"]
            self.assoc_indices = [i for i, name in enumerate(self.names) if name in assoc_sensors]
            assoc_mlp_struct = h_params.get('hub_overrides', {}).get('association', default_mlp)
            self.assoc_decoder = MLP(feat_dim, (len(self.assoc_indices) * hidden_size) + hidden_size, assoc_mlp_struct, rngs=rngs)
            
            # 2. Body-State Hub Decoder (Phase 2)
            body_sensors = ["Satiation", "Nutrition", "Injury", "Extero Nociception", "Collision"]
            self.body_indices = [i for i, name in enumerate(self.names) if name in body_sensors]
            body_mlp_struct = h_params.get('hub_overrides', {}).get('body_state', default_mlp)
            self.body_decoder = MLP(hidden_size, len(self.body_indices) * hidden_size, body_mlp_struct, rngs=rngs)
            
            # 3. Grouped Unimodal Decoders (Phase 3)
            self.unimodal_grouped_decoder = DreamerGroupedMLP(len(self.names), hidden_size, default_mlp, self.max_out, rngs=rngs)
        else:
            # Replicate standard Decoder behavior
            decoder_fc = config['decoder_fc_layers']
            self.flat_decoder = Decoder(feat_dim, obs_dim, decoder_fc, rngs=rngs)

    def __call__(self, feat):
        if self.mode == 'hierarchical':
            batch_shape = feat.shape[:-1]
            H = self.body_decoder.net.layers[0].in_features # hidden_size
            
            # Phase 1: Global Expansion
            assoc_body_flat = self.assoc_decoder(feat)
            assoc_flat = assoc_body_flat[..., :len(self.assoc_indices) * H]
            body_latent = assoc_body_flat[..., len(self.assoc_indices) * H:]
            
            # Phase 2: Body-State Expansion
            body_flat = self.body_decoder(body_latent)
            
            # Phase 3: Per-Sensor Reconstruction
            latents_all = jnp.zeros(batch_shape + (len(self.names), H), dtype=feat.dtype)
            assoc_reshaped = assoc_flat.reshape(batch_shape + (len(self.assoc_indices), H))
            latents_all = latents_all.at[..., self.assoc_indices, :].set(assoc_reshaped)
            body_reshaped = body_flat.reshape(batch_shape + (len(self.body_indices), H))
            latents_all = latents_all.at[..., self.body_indices, :].set(body_reshaped)
            
            decoded_all_padded = self.unimodal_grouped_decoder(latents_all)
            
            parts = []
            for i, dim in enumerate(self.sensor_dims):
                parts.append(decoded_all_padded[..., i, :dim])
            return jnp.concatenate(parts, axis=-1)
        else:
            return self.flat_decoder(feat)

class Decoder(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, fc_layers: list, rngs: nnx.Rngs):
        """
        Configurable Decoder matching PyTorch architecture.
        Args:
            input_dim: Input feature dimension (deter + stoch*discrete)
            output_dim: Output observation dimension
            fc_layers: List of hidden layer sizes (e.g., [128, 128, 128, 128, 128])
        """
        layers = []
        in_d = input_dim
        for h in fc_layers:
            layers.append(nnx.Linear(in_d, h, kernel_init=hafner_init(), rngs=rngs))
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
        layers.append(nnx.Linear(in_d, output_dim, kernel_init=hafner_init(), rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)

class MLP(nnx.Module):
    def __init__(self, input_dim, output_dim, hidden: list, rngs: nnx.Rngs):
        """
        Configurable MLP with LayerNorm + SiLU.
        """
        layers = []
        in_d = input_dim
        for h in hidden:
            layers.append(nnx.Linear(in_d, h, kernel_init=hafner_init(), rngs=rngs))
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
        layers.append(nnx.Linear(in_d, output_dim, kernel_init=hafner_init(), rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)

class WorldModel(nnx.Module):
    def __init__(self, obs_dim, act_dim, config: dict, rngs: nnx.Rngs,
                 obs_breakdown: Optional[dict] = None,
                 modulation_config: Optional[dict] = None):
        """
        Configurable World Model matching PyTorch architecture.

        Args:
            obs_dim: Observation vector dimension.
            act_dim: Action space dimension.
            config: Dictionary containing model hyperparameters.
            rngs: Flax NNX random number generators.
            obs_breakdown: Dict of {sensor_name: dim} for hierarchical encoding.
            modulation_config: Optional neuromodulation config dict.
        """
        self.deter_dim = config['rssm_deter_dim']
        self.stoch_dim = config['rssm_stoch_dim']
        self.discrete = config['rssm_classes']
        encoder_dim = config['encoder_dim']
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        decoder_fc = config['decoder_fc_layers']
        reward_fc = config['reward_fc_layers']
        continue_fc = config['continue_fc_layers']

        # --- Neuromodulation ---
        self.modulation_enabled = (modulation_config is not None
                                   and modulation_config.get('type') is not None)
        self.modulation_type = modulation_config['type'] if self.modulation_enabled else None

        # Use the new hierarchical-capable encoder
        self.encoder = DreamerObservationEncoder(
            obs_dim, encoder_dim, obs_breakdown, config, rngs=rngs
        )
        
        self.rssm = RSSM(
            act_dim, self.deter_dim, self.stoch_dim, self.discrete,
            embed_dim=encoder_dim,
            modulation_enabled=self.modulation_enabled,
            rngs=rngs,
        )

        feat_dim = self.deter_dim + self.stoch_dim * self.discrete
        self.feat_dim = feat_dim

        self.decoder = DreamerObservationDecoder(
            feat_dim, obs_dim, obs_breakdown, config, rngs=rngs
        )
            
        self.reward_head = MLP(feat_dim, 255, reward_fc, rngs=rngs)
        self.continue_head = MLP(feat_dim, 1, continue_fc, rngs=rngs)

        # --- Construct modulator when enabled ---
        if self.modulation_enabled:
            from src.models.neuromodulator import DreamerNeuromodulatorRNN
            mod_hidden = modulation_config['mod_hidden_size']
            grouping = modulation_config['grouping_size']
            percept_bias = modulation_config['percept_bias_init']
            percept_add_bias = modulation_config['percept_add_bias_init']
            memory_bias = modulation_config['memory_bias_init']
            reward_bias = modulation_config['reward_bias_init']

            self.modulator = DreamerNeuromodulatorRNN(
                obs_dim=obs_dim,
                embed_dim=encoder_dim,
                deter_dim=self.deter_dim,
                action_dim=act_dim,
                obs_breakdown=obs_breakdown,
                mod_hidden_size=mod_hidden,
                modulation_type=modulation_config['type'],
                grouping_size=grouping,
                percept_bias_init=percept_bias,
                percept_add_bias_init=percept_add_bias,
                memory_bias_init=memory_bias,
                reward_bias_init=reward_bias,
                rngs=rngs,
            )
            self.modulator.set_imagine_input_dim(feat_dim + act_dim, rngs)

    def get_feat(self, state):
        return jnp.concatenate([state['deter'], state['stoch']], axis=-1)

class ActorCritic(nnx.Module):
    def __init__(self, feat_dim, act_dim, config: dict, rngs: nnx.Rngs):
        """
        Configurable Actor-Critic matching PyTorch architecture.
        Args:
            config: Dictionary containing:
                - actor_fc_layers: Actor hidden layers
                - critic_fc_layers: Critic hidden layers
        """
        actor_fc = config['actor_fc_layers']
        critic_fc = config['critic_fc_layers']

        self.actor = MLP(feat_dim, act_dim, actor_fc, rngs=rngs)
        self.critic = MLP(feat_dim, 255, critic_fc, rngs=rngs)

class DreamerV3Agent(nnx.Module):
    """
    Container for the full agent with configurable architecture.
    """
    def __init__(self, obs_dim, act_dim, config: dict, rngs: nnx.Rngs,
                 obs_breakdown: Optional[dict] = None,
                 modulation_config: Optional[dict] = None):
        """
        Args:
            obs_dim: Observation vector dimension.
            act_dim: Action space dimension.
            config: Full agent config dictionary from YAML.
            rngs: Flax NNX random number generators.
            obs_breakdown: Breakdown metadata for hierarchical encoding.
            modulation_config: Optional neuromodulation config dict.
        """
        self.config = config
        self.modulation_config = modulation_config
        self.wm = WorldModel(obs_dim, act_dim, config, rngs=rngs,
                             obs_breakdown=obs_breakdown,
                             modulation_config=modulation_config)
        feat_dim = self.wm.deter_dim + self.wm.stoch_dim * self.wm.discrete
        self.ac = ActorCritic(feat_dim, act_dim, config, rngs=rngs)

    def initial_state(self, batch_size: int = None):
        """Returns initial RSSM state (and modulator state if enabled)."""
        if batch_size is None:
            rssm_h = self.wm.rssm.initial(1)
        else:
            rssm_h = self.wm.rssm.initial(batch_size)
            
        if self.wm.modulation_enabled:
            mod_h = self.wm.modulator.initial_state(batch_size)
            return (rssm_h, mod_h)
        return rssm_h

    def __call__(self, x: jnp.ndarray, h: Any, key: jax.Array = None):
        """
        Inference step for evaluation/rollouts.
        Args:
            x: Observation, shape (B, obs_dim)
            h: Hidden state (RSSM state or (RSSM, Modulator) tuple)
            key: Optional PRNG key for stochastic RSSM sampling
        Returns:
            (logits, value, h_new, mod_info)
        """
        if self.wm.modulation_enabled:
            rssm_h, mod_h = h
            mod_out, mod_h_new = self.wm.modulator.forward_obs(x, mod_h)
            embed = self.wm.encoder.forward_with_modulation(x, mod_out, self.wm.modulation_type)
            mod_info = mod_out
        else:
            rssm_h = h
            embed = self.wm.encoder(x)
            mod_h_new = None
            mod_info = None

        if key is None:
            key = jax.random.PRNGKey(0)

        # RSSM step needs prev_action. We use transitions stored in RSSM state.
        # During eval_mode, is_first is 0.0 (resets handled by env loops).
        is_first = jnp.zeros((x.shape[0],))
        post, _ = self.wm.rssm.step(rssm_h, embed, rssm_h['prev_action'], is_first, key)
        
        feat = self.wm.get_feat(post)
        logits = self.ac.actor(feat)
        value_logits = self.ac.critic(feat)
        value = from_twohot(value_logits, num_buckets=value_logits.shape[-1])
        
        # In evaluation (inference), we should store the action we just took 
        # so it's available for the next step.
        action = jnp.argmax(logits, axis=-1)
        action_onehot = jax.nn.one_hot(action, self.wm.rssm.action_dim)
        post = post.copy()
        post['prev_action'] = action_onehot

        if self.wm.modulation_enabled:
            h_new = (post, mod_h_new)
        else:
            h_new = post
            
        return logits, value, h_new, mod_info
