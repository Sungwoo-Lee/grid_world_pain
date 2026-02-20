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
    z_percept: jnp.ndarray       # Perceptual gain signal (gamma), shape (hidden_size,)
    z_percept_add: jnp.ndarray   # Perceptual additive signal (beta), shape (hidden_size,)
    z_memory: jnp.ndarray        # Memory gate-bias signal, shape (hidden_size,)
    temperature: jnp.ndarray     # Bounded temperature scalar, shape (1,)


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
        input_dim: int,
        target_hidden_size: int,
        *,
        mod_hidden_size: int = 64,
        modulation_type: str = "Multiplicative",
        grouping_size: int = 1,
        percept_bias_init: float = 2.0,
        percept_add_bias_init: float = 0.0,
        memory_bias_init: float = 0.0,
        temp_clip: Tuple[float, float] = (0.1, 10.0),
        rngs: nnx.Rngs,
    ):
        self.mod_hidden_size = mod_hidden_size
        self.target_hidden_size = target_hidden_size
        self.modulation_type = modulation_type
        self.grouping_size = grouping_size
        self.temp_clip = temp_clip

        self.num_groups = math.ceil(target_hidden_size / grouping_size)

        # Recurrent core
        self.gru = nnx.GRUCell(input_dim, mod_hidden_size, rngs=rngs)

        # === Branched Heads ===

        # Perceptual gain head (gamma): sigmoid(2.0) ≈ 0.88 (near pass-through, §5.2)
        self.head_percept = nnx.Linear(
            mod_hidden_size, self.num_groups,
            bias_init=nnx.initializers.constant(percept_bias_init),
            rngs=rngs,
        )

        # Perceptual additive head (beta): only for PreActivation mode
        if self.modulation_type == "PreActivation":
            self.head_percept_add = nnx.Linear(
                mod_hidden_size, self.num_groups,
                bias_init=nnx.initializers.constant(percept_add_bias_init),
                rngs=rngs,
            )

        # Memory gate-bias head: raw output, injected into GRU update gate (§5.1a)
        self.head_memory = nnx.Linear(
            mod_hidden_size, self.num_groups,
            bias_init=nnx.initializers.constant(memory_bias_init),
            rngs=rngs,
        )

        # Temperature head: scalar → softplus → clip
        self.head_action = nnx.Linear(mod_hidden_size, 1, rngs=rngs)

        # === Per-neuron learned baselines (AlKilany & Goodman, 2025) ===
        self.z_perc_baseline = nnx.Param(jnp.zeros(target_hidden_size))
        self.z_mem_baseline = nnx.Param(jnp.zeros(target_hidden_size))

        if self.modulation_type == "PreActivation":
            self.z_perc_add_baseline = nnx.Param(jnp.zeros(target_hidden_size))

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

        # Perceptual gain gamma (with spatial grouping broadcast)
        z_perc_raw = self.head_percept(h_mod_new)
        z_perc = jnp.repeat(z_perc_raw, self.grouping_size, axis=-1)
        z_perc = z_perc[..., :self.target_hidden_size]
        z_perc = self.z_perc_baseline.value + z_perc

        # Perceptual additive beta (only for PreActivation, zeros otherwise)
        if self.modulation_type == "PreActivation":
            z_perc_add_raw = self.head_percept_add(h_mod_new)
            z_perc_add = jnp.repeat(z_perc_add_raw, self.grouping_size, axis=-1)
            z_perc_add = z_perc_add[..., :self.target_hidden_size]
            z_perc_add = self.z_perc_add_baseline.value + z_perc_add
        else:
            z_perc_add = jnp.zeros_like(z_perc)

        # Memory gate-bias (with spatial grouping broadcast)
        z_mem_raw = self.head_memory(h_mod_new)
        z_mem = jnp.repeat(z_mem_raw, self.grouping_size, axis=-1)
        z_mem = z_mem[..., :self.target_hidden_size]
        z_mem = self.z_mem_baseline.value + z_mem

        # Temperature (bounded, §5.3a)
        z_act_raw = self.head_action(h_mod_new)
        temperature = jnp.clip(
            jax.nn.softplus(z_act_raw) + 0.5,
            self.temp_clip[0],
            self.temp_clip[1],
        )

        output = ModulatorOutput(
            z_percept=z_perc,
            z_percept_add=z_perc_add,
            z_memory=z_mem,
            temperature=temperature,
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
    z_percept: jnp.ndarray       # Perceptual gain signal (gamma), shape (embed_dim,)
    z_percept_add: jnp.ndarray   # Perceptual additive signal (beta), shape (embed_dim,)
    z_memory: jnp.ndarray        # Memory gate-bias signal, shape (deter_dim,)
    z_reward: jnp.ndarray        # Reward interpretation scale, shape (1,)


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
        self.num_groups_memory = math.ceil(deter_dim / grouping_size)

        # === Dual input projections ===
        self.proj_obs = nnx.Linear(obs_dim, mod_hidden_size, rngs=rngs)

        # Recurrent core (shared between both modes)
        self.gru = nnx.GRUCell(mod_hidden_size, mod_hidden_size, rngs=rngs)

        # === Branched Heads ===

        # Perceptual gain head (gamma): sigmoid(2.0) ≈ 0.88 (§5.2)
        self.head_percept = nnx.Linear(
            mod_hidden_size, self.num_groups_percept,
            bias_init=nnx.initializers.constant(percept_bias_init),
            rngs=rngs,
        )

        if self.modulation_type == "PreActivation":
            self.head_percept_add = nnx.Linear(
                mod_hidden_size, self.num_groups_percept,
                bias_init=nnx.initializers.constant(percept_add_bias_init),
                rngs=rngs,
            )

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
        self.z_perc_baseline = nnx.Param(jnp.zeros(embed_dim))
        self.z_mem_baseline = nnx.Param(jnp.zeros(deter_dim))

        if self.modulation_type == "PreActivation":
            self.z_perc_add_baseline = nnx.Param(jnp.zeros(embed_dim))

    def _compute_heads(self, h_mod: jnp.ndarray, include_percept: bool = True
                       ) -> DreamerModulatorOutput:
        """Compute head outputs from modulator hidden state.

        Args:
            h_mod: Modulator hidden state, shape (..., mod_hidden_size).
            include_percept: If True, compute z_percept/z_percept_add.
                             If False (imagination mode), return zeros for percept.
        """
        if include_percept:
            z_perc_raw = self.head_percept(h_mod)
            z_perc = jnp.repeat(z_perc_raw, self.grouping_size, axis=-1)
            z_perc = z_perc[..., :self.embed_dim]
            z_perc = self.z_perc_baseline.value + z_perc
        else:
            z_perc = jnp.zeros(h_mod.shape[:-1] + (self.embed_dim,))

        if include_percept and self.modulation_type == "PreActivation":
            z_perc_add_raw = self.head_percept_add(h_mod)
            z_perc_add = jnp.repeat(z_perc_add_raw, self.grouping_size, axis=-1)
            z_perc_add = z_perc_add[..., :self.embed_dim]
            z_perc_add = self.z_perc_add_baseline.value + z_perc_add
        else:
            z_perc_add = jnp.zeros_like(z_perc)

        z_mem_raw = self.head_memory(h_mod)
        z_mem = jnp.repeat(z_mem_raw, self.grouping_size, axis=-1)
        z_mem = z_mem[..., :self.deter_dim]
        z_mem = self.z_mem_baseline.value + z_mem

        z_rew = jax.nn.sigmoid(self.head_reward(h_mod))

        return DreamerModulatorOutput(
            z_percept=z_perc,
            z_percept_add=z_perc_add,
            z_memory=z_mem,
            z_reward=z_rew,
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
