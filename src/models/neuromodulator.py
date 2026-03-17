"""
Neuromodulatory networks with branched heads for RecurrentPPO and DreamerV3.

Implements the modulator architecture from NEUROMODULATION_ALGORITHM.md §2:
- Recurrent core (GRU) for "affective inertia"
- Branched output heads with spatial grouping (AlKilany & Goodman, 2025)
- Per-neuron learned baselines for additive modulation
- Pass-through initialization (§5.2) and bounded temperature (§5.3)

Supports two perceptual modulation styles (Injection A):
- "Multiplicative": Post-activation gating (Ben-Iwhiwhu style).
    h = relu(Wx + b) * sigmoid(z_percept)
- "PreActivation": Pre-activation gain + threshold shift (Ferguson & Cardin style).
    h = relu(gamma(z_percept) * (Wx + b) + beta(z_percept_add))
    gamma = multiplicative gain (Shine et al. energy landscape)
    beta  = additive threshold shift (Ferguson disinhibitory gating)

Two modulator classes:
- NeuromodulatorRNN: For RecurrentPPO (single target_hidden_size, temperature head).
- DreamerNeuromodulatorRNN: For DreamerV3 (per-head target dims, reward head,
  dual input projections for observation vs imagination modes).
"""

import math
import jax
import jax.numpy as jnp
from flax import nnx
from typing import NamedTuple, Tuple


class ModulatorOutput(NamedTuple):
    """Output from the neuromodulator's branched heads."""
    z_unimodal: jnp.ndarray      # Phase 1: Unimodal gains
    z_unimodal_add: jnp.ndarray  # Phase 1: Unimodal biases
    z_multimodal: jnp.ndarray    # Phase 2: Multimodal gains
    z_multimodal_add: jnp.ndarray # Phase 2: Multimodal biases
    z_memory: jnp.ndarray        # Memory gate-bias signal
    temperature: jnp.ndarray     # Bounded temperature scalar


class NeuromodulatorRNN(nnx.Module):
    """Recurrent neuromodulatory network with branched output heads.

    Args:
        input_dim: Dimension of the observation vector (same as task network).
        target_hidden_size: Hidden size of the task network (determines head output dims).
        mod_hidden_size: Hidden size of the modulator GRU (default: 64).
        modulation_type: "Multiplicative" (post-activation gate) or "PreActivation"
                         (Ferguson-style gain + threshold shift inside activation).
        grouping_size: Spatial grouping G (1 = per-neuron, target_hidden_size = global).
        percept_bias_init: Bias initialization for the perceptual gain head (gamma).
        percept_add_bias_init: Bias initialization for the perceptual additive head (beta).
                               Only used when modulation_type = "PreActivation".
        memory_bias_init: Bias initialization for the memory gate-bias head.
        temp_clip: (min, max) bounds for the temperature output.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        obs_dim: int,
        target_hidden_size: int,
        obs_breakdown: dict,
        *,
        mod_hidden_size: int = 64,
        modulation_type: str = "Multiplicative",
        grouping_size: int = 1,
        percept_bias_init: float = 2.0,
        percept_add_bias_init: float = 0.0,
        memory_bias_init: float = 0.0,
        temp_clip: Tuple[float, float] = (0.1, 10.0),
        memory_clip: Tuple[float, float] = (-2.0, 2.0),
        rngs: nnx.Rngs,
    ):
        self.mod_hidden_size = mod_hidden_size
        self.target_hidden_size = target_hidden_size
        self.modulation_type = modulation_type
        self.grouping_size = grouping_size
        self.temp_clip = temp_clip
        self.memory_clip = memory_clip

        self.num_groups_hidden = math.ceil(target_hidden_size / grouping_size)
        self.num_groups_unimodal = self.num_groups_hidden

        # Recurrent core
        self.gru = nnx.GRUCell(obs_dim, mod_hidden_size, rngs=rngs)

        # === Branched Hierarchical Heads ===

        # Phase 1: Unimodal
        # FiLM modes use γ bias = 1.0 (identity in linear space); sigmoid modes use percept_bias_init
        if self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            _percept_bias = 1.0
        else:
            _percept_bias = percept_bias_init

        self.head_unimodal = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                       bias_init=nnx.initializers.constant(_percept_bias), rngs=rngs)
        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.head_unimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                               bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)

        # Phase 2: Multimodal
        self.head_multimodal = nnx.Linear(mod_hidden_size, self.num_groups_hidden,
                                         bias_init=nnx.initializers.constant(_percept_bias), rngs=rngs)
        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.head_multimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_hidden,
                                                 bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)

        # Memory gate-bias head: raw output, injected into GRU update gate (§5.1a)
        self.head_memory = nnx.Linear(
            mod_hidden_size, self.num_groups_hidden,
            bias_init=nnx.initializers.constant(memory_bias_init),
            rngs=rngs,
        )

        # Temperature head: scalar → softplus → clip
        self.head_action = nnx.Linear(mod_hidden_size, 1, rngs=rngs)

        # === Per-neuron learned baselines ===
        self.z_unimodal_baseline = nnx.Param(jnp.zeros(target_hidden_size))
        self.z_hidden_baseline = nnx.Param(jnp.zeros(target_hidden_size))
        self.z_mem_baseline = nnx.Param(jnp.zeros(target_hidden_size))

        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))
            self.z_hidden_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))

    def __call__(
        self,
        obs: jnp.ndarray,
        h_mod: jnp.ndarray,
    ) -> Tuple[ModulatorOutput, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: Observation vector, shape (..., input_dim).
            h_mod: Previous modulator hidden state, shape (..., mod_hidden_size).

        Returns:
            (output, h_mod_new): ModulatorOutput and new hidden state.
        """
        h_mod_new, _ = self.gru(h_mod, obs)

        def _get_signal(head, baseline, head_add=None, baseline_add=None):
            raw = head(h_mod_new)
            sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
            sig = baseline.value + sig
            
            if head_add is not None:
                raw_add = head_add(h_mod_new)
                sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :self.target_hidden_size]
                sig_add = baseline_add.value + sig_add
                return sig, sig_add
            return sig, jnp.zeros_like(sig)

        z_uni, z_uni_add = _get_signal(self.head_unimodal, self.z_unimodal_baseline, 
                                      getattr(self, 'head_unimodal_add', None), 
                                      getattr(self, 'z_unimodal_add_baseline', None))
        
        z_multi, z_multi_add = _get_signal(self.head_multimodal, self.z_hidden_baseline,
                                          getattr(self, 'head_multimodal_add', None),
                                          getattr(self, 'z_hidden_add_baseline', None))

        # Memory (always Multiplicative/direct bias)
        z_mem, _ = _get_signal(self.head_memory, self.z_mem_baseline)
        z_mem = jnp.clip(z_mem, self.memory_clip[0], self.memory_clip[1])

        # Temperature (bounded)
        z_act_raw = self.head_action(h_mod_new)
        temperature = jnp.clip(jax.nn.softplus(z_act_raw) + 0.5, self.temp_clip[0], self.temp_clip[1])

        output = ModulatorOutput(
            z_unimodal=z_uni, z_unimodal_add=z_uni_add,
            z_multimodal=z_multi, z_multimodal_add=z_multi_add,
            z_memory=z_mem, temperature=temperature
        )
        return output, h_mod_new

    def initial_state(self, batch_size: int = None) -> jnp.ndarray:
        """Returns the initial modulator hidden state (zeros)."""
        if batch_size is None:
            return jnp.zeros((self.mod_hidden_size,))
        return jnp.zeros((batch_size, self.mod_hidden_size))


# =============================================================================
# DreamerV3 Neuromodulator
# =============================================================================

class DreamerModulatorOutput(NamedTuple):
    """Output from the DreamerV3 neuromodulator's branched heads."""
    z_unimodal: jnp.ndarray      # Phase 1: Unimodal gains
    z_unimodal_add: jnp.ndarray  # Phase 1: Unimodal biases
    z_multimodal: jnp.ndarray    # Phase 2: Multimodal gains
    z_multimodal_add: jnp.ndarray # Phase 2: Multimodal biases
    z_memory: jnp.ndarray        # Memory gate-bias signal
    z_reward: jnp.ndarray        # Reward interpretation scale


class DreamerNeuromodulatorRNN(nnx.Module):
    """Recurrent neuromodulatory network for DreamerV3 with dual input modes.

    Operates in two modes:
    - Observation mode (forward_obs): Receives raw observations during RSSM step().
      Produces z_percept, z_percept_add, z_memory, z_reward.
    - Imagination mode (forward_imagine): Receives concat(feat, action) during
      RSSM imagine_step(). Produces z_memory and z_reward (z_percept is zeros
      since the encoder doesn't run during imagination).

    Args:
        obs_dim: Dimension of the raw observation vector.
        embed_dim: Dimension of the encoder output (target for percept heads).
        deter_dim: Dimension of the RSSM deterministic state (target for memory head).
        action_dim: Dimension of the action space.
        mod_hidden_size: Hidden size of the modulator GRU.
        modulation_type: "Multiplicative" or "PreActivation".
        grouping_size: Spatial grouping G (1 = per-neuron).
        percept_bias_init: Bias init for perceptual gain head (gamma).
        percept_add_bias_init: Bias init for perceptual additive head (beta).
        memory_bias_init: Bias init for memory gate-bias head.
        reward_bias_init: Bias init for reward head (pass-through at start).
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int,
        deter_dim: int,
        action_dim: int,
        obs_breakdown: dict,
        *,
        mod_hidden_size: int = 64,
        modulation_type: str = "Multiplicative",
        grouping_size: int = 1,
        percept_bias_init: float = 2.0,
        percept_add_bias_init: float = 0.0,
        memory_bias_init: float = 0.0,
        reward_bias_init: float = 2.0,
        rngs: nnx.Rngs,
    ):
        self.mod_hidden_size = mod_hidden_size
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.modulation_type = modulation_type
        self.grouping_size = grouping_size

        self.num_groups_percept = math.ceil(embed_dim / grouping_size)
        self.num_groups_unimodal = self.num_groups_percept
        self.num_groups_memory = math.ceil(deter_dim / grouping_size)

        # === Dual input projections ===
        self.proj_obs = nnx.Linear(obs_dim, mod_hidden_size, rngs=rngs)

        # Recurrent core (shared between both modes)
        self.gru = nnx.GRUCell(mod_hidden_size, mod_hidden_size, rngs=rngs)

        # === Branched Hierarchical Heads ===

        # Phase 1: Unimodal
        # FiLM modes use γ bias = 1.0 (identity in linear space); sigmoid modes use percept_bias_init
        if self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            _percept_bias = 1.0
        else:
            _percept_bias = percept_bias_init

        self.head_unimodal = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                       bias_init=nnx.initializers.constant(_percept_bias), rngs=rngs)
        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.head_unimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_unimodal,
                                               bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)

        # Phase 2: Multimodal
        self.head_multimodal = nnx.Linear(mod_hidden_size, self.num_groups_percept,
                                         bias_init=nnx.initializers.constant(_percept_bias), rngs=rngs)
        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.head_multimodal_add = nnx.Linear(mod_hidden_size, self.num_groups_percept,
                                                 bias_init=nnx.initializers.constant(percept_add_bias_init), rngs=rngs)

        # Memory gate-bias head: targets deter_dim (§5.1a)
        self.head_memory = nnx.Linear(
            mod_hidden_size, self.num_groups_memory,
            bias_init=nnx.initializers.constant(memory_bias_init),
            rngs=rngs,
        )

        # Reward head: sigmoid-bounded for nociceptive interpretation
        self.head_reward = nnx.Linear(
            mod_hidden_size, 1,
            bias_init=nnx.initializers.constant(reward_bias_init),
            rngs=rngs,
        )

        # === Per-neuron learned baselines ===
        self.z_unimodal_baseline = nnx.Param(jnp.zeros(embed_dim))
        self.z_hidden_baseline = nnx.Param(jnp.zeros(embed_dim))
        self.z_mem_baseline = nnx.Param(jnp.zeros(deter_dim))

        if self.modulation_type == "PreActivation" or self.modulation_type == "FiLM" or self.modulation_type == "FiLMNoNorm":
            self.z_unimodal_add_baseline = nnx.Param(jnp.zeros(embed_dim))
            self.z_hidden_add_baseline = nnx.Param(jnp.zeros(embed_dim))

    def _compute_heads(self, h_mod: jnp.ndarray, include_percept: bool = True
                       ) -> DreamerModulatorOutput:
        """Compute head outputs from modulator hidden state."""
        
        def _get_signal(head, baseline, head_add=None, baseline_add=None, target_dim=None, is_percept=False):
            if not include_percept and is_percept:
                # Imagination mode: return zeros for perceptual heads
                return jnp.zeros(h_mod.shape[:-1] + (target_dim,)), jnp.zeros(h_mod.shape[:-1] + (target_dim,))
            
            raw = head(h_mod)
            sig = jnp.repeat(raw, self.grouping_size, axis=-1)[..., :target_dim]
            sig = baseline.value + sig
            
            if head_add is not None:
                raw_add = head_add(h_mod)
                sig_add = jnp.repeat(raw_add, self.grouping_size, axis=-1)[..., :target_dim]
                sig_add = baseline_add.value + sig_add
                return sig, sig_add
            return sig, jnp.zeros_like(sig)

        z_uni, z_uni_add = _get_signal(self.head_unimodal, self.z_unimodal_baseline,
                                      getattr(self, 'head_unimodal_add', None),
                                      getattr(self, 'z_unimodal_add_baseline', None), 
                                      target_dim=self.embed_dim, is_percept=True)
        
        z_multi, z_multi_add = _get_signal(self.head_multimodal, self.z_hidden_baseline,
                                          getattr(self, 'head_multimodal_add', None),
                                          getattr(self, 'z_hidden_add_baseline', None), 
                                          target_dim=self.embed_dim, is_percept=True)

        z_mem, _ = _get_signal(self.head_memory, self.z_mem_baseline, target_dim=self.deter_dim)
        z_rew = jax.nn.sigmoid(self.head_reward(h_mod))

        return DreamerModulatorOutput(
            z_unimodal=z_uni, z_unimodal_add=z_uni_add,
            z_multimodal=z_multi, z_multimodal_add=z_multi_add,
            z_memory=z_mem, z_reward=z_rew
        )

    def forward_obs(
        self,
        obs: jnp.ndarray,
        h_mod: jnp.ndarray,
    ) -> Tuple[DreamerModulatorOutput, jnp.ndarray]:
        """Observation mode: called during RSSM step() and get_action()."""
        x = jax.nn.relu(self.proj_obs(obs))
        h_mod_new, _ = self.gru(h_mod, x)
        output = self._compute_heads(h_mod_new, include_percept=True)
        return output, h_mod_new

    def forward_imagine(
        self,
        imagine_input: jnp.ndarray,
        h_mod: jnp.ndarray,
    ) -> Tuple[DreamerModulatorOutput, jnp.ndarray]:
        """Imagination mode: called during RSSM imagine_step()."""
        x = jax.nn.relu(self.proj_imagine(imagine_input))
        h_mod_new, _ = self.gru(h_mod, x)
        output = self._compute_heads(h_mod_new, include_percept=False)
        return output, h_mod_new

    def set_imagine_input_dim(self, imagine_input_dim: int, rngs: nnx.Rngs):
        """Lazily initialize the imagination projection layer.

        Called after construction once feat_dim + action_dim is known
        (feat_dim = deter_dim + stoch_dim * discrete).
        """
        self.proj_imagine = nnx.Linear(imagine_input_dim, self.mod_hidden_size, rngs=rngs)

    def initial_state(self, batch_size: int = None) -> jnp.ndarray:
        """Returns the initial modulator hidden state (zeros)."""
        if batch_size is None:
            return jnp.zeros((self.mod_hidden_size,))
        return jnp.zeros((batch_size, self.mod_hidden_size))
