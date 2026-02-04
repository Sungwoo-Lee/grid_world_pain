import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict, Any

class RSSM(nn.Module):
    deter_dim: int
    stoch_dim: int
    stoch_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, prev_deter: jnp.ndarray, prev_stoch: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """One step of the deterministic transition: s_{t-1}, a_{t-1} -> h_t"""
        x = jnp.concatenate([prev_stoch.reshape(prev_stoch.shape[:-2] + (-1,)), action], axis=-1)
        
        # Deterministic transition (GRU)
        gru_cell = nn.GRUCell(features=self.deter_dim)
        deter, _ = gru_cell(prev_deter, x)
        
        # Stochastic prior: h_t -> p(z_t | h_t)
        prior_logits = self._get_stoch_logits(deter)
        prior_stoch = self._sample_stoch(prior_logits)
        
        return deter, prior_stoch, prior_logits

    def _get_stoch_logits(self, deter):
        x = nn.Dense(self.hidden_size)(deter)
        x = nn.relu(x)
        logits = nn.Dense(self.stoch_dim * self.stoch_classes)(x)
        return logits.reshape(logits.shape[:-1] + (self.stoch_dim, self.stoch_classes))

    def _sample_stoch(self, logits):
        # DreamerV3 uses unimodal sampling (ST-estimator)
        return jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.stoch_classes)

class WorldModel(nn.Module):
    obs_dim: int
    deter_dim: int
    stoch_dim: int
    stoch_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, prev_deter: jnp.ndarray, prev_stoch: jnp.ndarray, action: jnp.ndarray):
        rssm = RSSM(self.deter_dim, self.stoch_dim, self.stoch_classes, self.hidden_size)
        
        # 1. Deterministic state and Prior
        deter, prior_stoch, prior_logits = rssm(prev_deter, prev_stoch, action)
        
        # 2. Posterior: h_t, Encoder(obs_t) -> q(z_t | h_t, obs_t)
        embed = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.relu,
            nn.Dense(self.hidden_size),
            nn.relu
        ])(obs)
        
        post_input = jnp.concatenate([deter, embed], axis=-1)
        post_logits = nn.Dense(self.stoch_dim * self.stoch_classes)(post_input)
        post_logits = post_logits.reshape(post_logits.shape[:-1] + (self.stoch_dim, self.stoch_classes))
        post_stoch = rssm._sample_stoch(post_logits)
        
        return deter, post_stoch, post_logits, prior_stoch, prior_logits

class Decoder(nn.Module):
    output_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, deter: jnp.ndarray, stoch: jnp.ndarray):
        x = jnp.concatenate([deter, stoch.reshape(stoch.shape[:-2] + (-1,))], axis=-1)
        out = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.relu,
            nn.Dense(self.hidden_size),
            nn.relu,
            nn.Dense(self.output_dim)
        ])(x)
        return out
