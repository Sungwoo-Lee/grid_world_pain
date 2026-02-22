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
    def __init__(self, action_dim: int, deter_dim: int = 512, stoch_dim: int = 32,
                 discrete: int = 32, embed_dim: int = 512,
                 modulation_enabled: bool = False, rngs: nnx.Rngs = None):
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
            mod_output: DreamerModulatorOutput with z_percept (gamma) and
                        z_percept_add (beta) fields.
            modulation_type: "Multiplicative" or "PreActivation".

        Returns:
            Modulated embedding, shape (..., embed_dim).
        """
        x_pre = self.body(x)

        if modulation_type == "PreActivation":
            gamma = jax.nn.sigmoid(mod_output.z_percept)
            beta = mod_output.z_percept_add
            return self.final_act(x_pre * gamma + beta)
        else:
            return self.final_act(x_pre) * jax.nn.sigmoid(mod_output.z_percept)

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
    def __init__(self, input_dim, output_dim, hidden: list = None, rngs: nnx.Rngs = None):
        """
        Configurable MLP with LayerNorm + SiLU.
        Args:
            hidden: List of hidden layer sizes (default: [256, 256])
        """
        if hidden is None:
            hidden = [256, 256]
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
                 modulation_config: Optional[dict] = None):
        """
        Configurable World Model matching PyTorch architecture.

        Args:
            obs_dim: Observation vector dimension.
            act_dim: Action space dimension.
            config: Dictionary containing model hyperparameters.
            rngs: Flax NNX random number generators.
            modulation_config: Optional neuromodulation config dict.
                When not None and type is not None, constructs a
                DreamerNeuromodulatorRNN and uses ModulatedLayerNormGRUCell.
        """
        self.deter_dim = config.get('rssm_deter_dim', 512)
        self.stoch_dim = config.get('rssm_stoch_dim', 32)
        self.discrete = config.get('rssm_classes', 32)
        encoder_dim = config.get('encoder_dim', self.deter_dim)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        encoder_fc = config.get('encoder_fc_layers', [128, 128])
        decoder_fc = config.get('decoder_fc_layers', [128, 128])
        reward_fc = config.get('reward_fc_layers', [128, 128])
        continue_fc = config.get('continue_fc_layers', [128, 128])

        # --- Neuromodulation ---
        self.modulation_enabled = (modulation_config is not None
                                   and modulation_config.get('type') is not None)
        self.modulation_type = modulation_config['type'] if self.modulation_enabled else None

        self.encoder = Encoder(obs_dim, encoder_dim, encoder_fc, rngs=rngs)
        self.rssm = RSSM(
            act_dim, self.deter_dim, self.stoch_dim, self.discrete,
            embed_dim=encoder_dim,
            modulation_enabled=self.modulation_enabled,
            rngs=rngs,
        )

        feat_dim = self.deter_dim + self.stoch_dim * self.discrete
        self.feat_dim = feat_dim

        self.decoder = Decoder(feat_dim, obs_dim, decoder_fc, rngs=rngs)
        self.reward_head = MLP(feat_dim, 255, reward_fc, rngs=rngs)
        self.continue_head = MLP(feat_dim, 1, continue_fc, rngs=rngs)

        # --- Construct modulator when enabled ---
        if self.modulation_enabled:
            from src.models.neuromodulator import DreamerNeuromodulatorRNN
            mod_hidden = modulation_config.get('mod_hidden_size', 64)
            grouping = modulation_config.get('grouping_size', 1)
            percept_bias = modulation_config.get('percept_bias_init', 2.0)
            percept_add_bias = modulation_config.get('percept_add_bias_init', 0.0)
            memory_bias = modulation_config.get('memory_bias_init', 0.0)
            reward_bias = modulation_config.get('reward_bias_init', 2.0)

            self.modulator = DreamerNeuromodulatorRNN(
                obs_dim=obs_dim,
                embed_dim=encoder_dim,
                deter_dim=self.deter_dim,
                action_dim=act_dim,
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
        actor_fc = config.get('actor_fc_layers', [256, 256])
        critic_fc = config.get('critic_fc_layers', [256, 256])

        self.actor = MLP(feat_dim, act_dim, actor_fc, rngs=rngs)
        self.critic = MLP(feat_dim, 255, critic_fc, rngs=rngs)

class DreamerV3Agent(nnx.Module):
    """
    Container for the full agent with configurable architecture.
    """
    def __init__(self, obs_dim, act_dim, config: dict, rngs: nnx.Rngs,
                 modulation_config: Optional[dict] = None):
        """
        Args:
            obs_dim: Observation vector dimension.
            act_dim: Action space dimension.
            config: Full agent config dictionary from YAML.
            rngs: Flax NNX random number generators.
            modulation_config: Optional neuromodulation config dict.
        """
        self.config = config
        self.modulation_config = modulation_config
        self.wm = WorldModel(obs_dim, act_dim, config, rngs=rngs,
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
