import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
from typing import Tuple, Dict, Any, Optional

from src.models.dreamer_v3_util import symlog, symexp, to_twohot, from_twohot, OneHotDist

# -----------------------------------------------------------------------------
# Core Modules
# -----------------------------------------------------------------------------

class SiLU(nnx.Module):
    def __call__(self, x):
        return jax.nn.silu(x)

class LayerNormGRUCell(nnx.Module):
    def __init__(self, hidden_size: int, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        # GRU Gates: Reset, Update, Candidate
        # Combined Linear layer for input -> 3*hidden
        self.dense_ih = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)
        self.dense_hh = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)
        
        self.ln_ih = nnx.LayerNorm(3 * hidden_size, rngs=rngs)
        self.ln_hh = nnx.LayerNorm(3 * hidden_size, rngs=rngs)

    def __call__(self, x, h):
        # x is assumed to be projected to hidden_size already? 
        # No, typically input z is projected before GRU. 
        # Here we assume x matches shape or dense_ih handles it.
        # But dense_ih is hidden->3*hidden. So x must be hidden_size? 
        # Standard GRUCell takes (input_size, hidden_size).
        # We'll assume external projection for now to match 'deter_dim'.
        
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
    def __init__(self, action_dim: int, deter_dim: int = 512, stoch_dim: int = 32, discrete: int = 32, embed_dim: int = 512, rngs: nnx.Rngs = None):
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.discrete = discrete
        self.action_dim = action_dim
        
        # Cell Input: stoch + action -> deter_dim
        self.img_in = nnx.Linear(stoch_dim * discrete + action_dim, deter_dim, rngs=rngs)
        self.cell = LayerNormGRUCell(deter_dim, rngs=rngs)
        
        # Prior: deter -> stoch_logits
        self.img_out = nnx.Linear(deter_dim, stoch_dim * discrete, rngs=rngs)
        
        # Posterior: deter + embed -> stoch_logits
        self.obs_out = nnx.Linear(deter_dim + embed_dim, stoch_dim * discrete, rngs=rngs)


    def initial(self, batch_size: int):
        return {
            'deter': jnp.zeros((batch_size, self.deter_dim)),
            'stoch': jnp.zeros((batch_size, self.stoch_dim * self.discrete)), # One-hot flat
            'logits': jnp.zeros((batch_size, self.stoch_dim, self.discrete)),
            'prev_action': jnp.zeros((batch_size, self.action_dim))
        }

    def observe(self, embed, action, is_first, state=None, rngs: nnx.Rngs = None):
        # embed: (B, T, E)
        # action: (B, T, A)
        # is_first: (B, T)
        
        B, T, _ = embed.shape
        if state is None:
            state = self.initial(B)
            
        # Unroll loop
        # We can use scan, but for now manual loop or scan wrapper.
        # For simplicity inside Module, let's look at structure.
        # It's better to expose single step and use scan in call/external.
        pass # To be implemented in __call__ or scan method

    def step(self, prev_state, embed, action, is_first, key):
        # Single step transition
        # Mask state if first step
        # CRITICAL: Use .reshape((-1, 1)) to be robust to is_first shape (B,) or (B, 1)
        # Failure to reshape causes broadcasting that results in Rank-3 states.
        mask = (1.0 - is_first).astype(jnp.float32).reshape((-1, 1))
        deter = prev_state['deter'] * mask
        stoch = prev_state['stoch'] * mask
        
        # 1. Deterministic Step
        x = jnp.concatenate([stoch, action], axis=-1)
        x = self.img_in(x)
        x = nnx.elu(x) # Activation before GRU typically? DreamerV3 uses SiLU/ELU
        deter = self.cell(x, deter)
        
        # 2. Prior
        prior_logits = self.img_out(deter)
        prior_logits = prior_logits.reshape(prior_logits.shape[:-1] + (self.stoch_dim, self.discrete))
        
        # Sample Prior (for imagination usually, but here just computing logits)
        # We don't sample prior for posterior calculation usually, unless for KL.
        
        # 3. Posterior
        # Embed is input here
        post_input = jnp.concatenate([deter, embed], axis=-1)
        post_logits = self.obs_out(post_input)
        post_logits = post_logits.reshape(post_logits.shape[:-1] + (self.stoch_dim, self.discrete))
        
        # Sample Posterior
        dist = OneHotDist(post_logits)
        stoch = dist.sample(key) # (B, S, D)
        stoch = stoch.reshape(stoch.shape[:-2] + (-1,)) # Flatten
        
        post = {'deter': deter, 'stoch': stoch, 'logits': post_logits, 'prev_action': action}
        prior = {'deter': deter, 'stoch': None, 'logits': prior_logits, 'prev_action': action} # Stoch sampled later if needed
        
        return post, prior

    def imagine_step(self, prev_state, action, key):
        # Dynamics only
        deter = prev_state['deter']
        stoch = prev_state['stoch']
        
        x = jnp.concatenate([stoch, action], axis=-1)
        x = self.img_in(x)
        x = nnx.elu(x)
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
        Args:
            input_dim: Input observation dimension
            embed_dim: Output embedding dimension
            fc_layers: List of hidden layer sizes (e.g., [128, 128, 128, 128, 128])
        """
        layers = []
        in_d = input_dim
        for h in fc_layers:
            layers.append(nnx.Linear(in_d, h, rngs=rngs))
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
        layers.append(nnx.Linear(in_d, embed_dim, rngs=rngs))
        layers.append(nnx.LayerNorm(embed_dim, rngs=rngs))
        layers.append(SiLU())
        self.net = nnx.Sequential(*layers)
        
    def __call__(self, x):
        return self.net(x)

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
            layers.append(nnx.Linear(in_d, h, rngs=rngs))
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
        layers.append(nnx.Linear(in_d, output_dim, rngs=rngs))
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
            layers.append(nnx.Linear(in_d, h, rngs=rngs))
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
        layers.append(nnx.Linear(in_d, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)
        
    def __call__(self, x):
        return self.net(x)

class WorldModel(nnx.Module):
    def __init__(self, obs_dim, act_dim, config: dict, rngs: nnx.Rngs):
        """
        Configurable World Model matching PyTorch architecture.
        Args:
            config: Dictionary containing:
                - encoder_dim: Embedding dimension
                - encoder_fc_layers: Encoder hidden layers
                - rssm_deter_dim: Deterministic state dimension
                - rssm_stoch_dim: Number of stochastic classes
                - rssm_classes: Number of discrete classes per stoch unit
                - decoder_fc_layers: Decoder hidden layers
                - reward_fc_layers: Reward head hidden layers
                - continue_fc_layers: Continue head hidden layers
        """
        self.deter_dim = config.get('rssm_deter_dim', 512)
        self.stoch_dim = config.get('rssm_stoch_dim', 32)
        self.discrete = config.get('rssm_classes', 32)
        encoder_dim = config.get('encoder_dim', self.deter_dim)
        
        encoder_fc = config.get('encoder_fc_layers', [128, 128])
        decoder_fc = config.get('decoder_fc_layers', [128, 128])
        reward_fc = config.get('reward_fc_layers', [128, 128])
        continue_fc = config.get('continue_fc_layers', [128, 128])
        
        self.encoder = Encoder(obs_dim, encoder_dim, encoder_fc, rngs=rngs)
        self.rssm = RSSM(act_dim, self.deter_dim, self.stoch_dim, self.discrete, embed_dim=encoder_dim, rngs=rngs)

        
        feat_dim = self.deter_dim + self.stoch_dim * self.discrete
        
        self.decoder = Decoder(feat_dim, obs_dim, decoder_fc, rngs=rngs)
        self.reward_head = MLP(feat_dim, 255, reward_fc, rngs=rngs)
        self.continue_head = MLP(feat_dim, 1, continue_fc, rngs=rngs)

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
    def __init__(self, obs_dim, act_dim, config: dict, rngs: nnx.Rngs):
        """
        Args:
            config: Full agent config dictionary from YAML
        """
        self.config = config
        self.wm = WorldModel(obs_dim, act_dim, config, rngs=rngs)
        feat_dim = self.wm.deter_dim + self.wm.stoch_dim * self.wm.discrete
        self.ac = ActorCritic(feat_dim, act_dim, config, rngs=rngs)

