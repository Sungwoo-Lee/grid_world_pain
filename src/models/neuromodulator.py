"""
NeuromodulatorRNN: A standalone recurrent neuromodulatory network with branched heads.

Implements the modulator architecture from NEUROMODULATION_ALGORITHM.md §2:
- Recurrent core (GRU) for "affective inertia"
- Branched output heads with spatial grouping (AlKilany & Goodman, 2025)
- Per-neuron learned baselines for additive modulation
- Pass-through initialization (§5.2) and bounded temperature (§5.3)
"""

import math
import jax
import jax.numpy as jnp
from flax import nnx
from typing import NamedTuple, Tuple


class ModulatorOutput(NamedTuple):
    """Output from the neuromodulator's branched heads."""
    z_percept: jnp.ndarray   # Perceptual gate signal, shape (hidden_size,)
    z_memory: jnp.ndarray    # Memory gate-bias signal, shape (hidden_size,)
    temperature: jnp.ndarray # Bounded temperature scalar, shape (1,)


class NeuromodulatorRNN(nnx.Module):
    """Recurrent neuromodulatory network with branched output heads.

    Args:
        input_dim: Dimension of the observation vector (same as task network).
        target_hidden_size: Hidden size of the task network (determines head output dims).
        mod_hidden_size: Hidden size of the modulator GRU (default: 64).
        grouping_size: Spatial grouping G (1 = per-neuron, target_hidden_size = global).
        percept_bias_init: Bias initialization for the perceptual gate head.
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
        grouping_size: int = 1,
        percept_bias_init: float = 2.0,
        memory_bias_init: float = 0.0,
        temp_clip: Tuple[float, float] = (0.1, 10.0),
        rngs: nnx.Rngs,
    ):
        self.mod_hidden_size = mod_hidden_size
        self.target_hidden_size = target_hidden_size
        self.grouping_size = grouping_size
        self.temp_clip = temp_clip

        # Number of groups (head output dimension before broadcast)
        self.num_groups = math.ceil(target_hidden_size / grouping_size)

        # Recurrent core
        self.gru = nnx.GRUCell(input_dim, mod_hidden_size, rngs=rngs)

        # === Branched Heads ===

        # Perceptual gate head: output → sigmoid → gate input features
        # Bias init +2.0 so sigmoid(2.0) ≈ 0.88 (near pass-through at start, §5.2)
        self.head_percept = nnx.Linear(
            mod_hidden_size, self.num_groups,
            bias_init=nnx.initializers.constant(percept_bias_init),
            rngs=rngs,
        )

        # Memory gate-bias head: raw output (no sigmoid), injected into GRU update gate
        # Bias init 0.0 (neutral — no shift at start, §5.1a)
        self.head_memory = nnx.Linear(
            mod_hidden_size, self.num_groups,
            bias_init=nnx.initializers.constant(memory_bias_init),
            rngs=rngs,
        )

        # Temperature head: scalar output → softplus → clip
        self.head_action = nnx.Linear(mod_hidden_size, 1, rngs=rngs)

        # === Per-neuron learned baselines for additive modulation ===
        # These preserve heterogeneity when G > 1 (AlKilany & Goodman, 2025)
        self.z_perc_baseline = nnx.Param(jnp.zeros(target_hidden_size))
        self.z_mem_baseline = nnx.Param(jnp.zeros(target_hidden_size))

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
        # Recurrent step
        h_mod_new, _ = self.gru(h_mod, obs)

        # --- Head outputs ---

        # Perceptual gate (with spatial grouping broadcast)
        z_perc_raw = self.head_percept(h_mod_new)  # (num_groups,)
        z_perc = jnp.repeat(z_perc_raw, self.grouping_size, axis=-1)
        z_perc = z_perc[..., :self.target_hidden_size]  # Trim to exact size
        z_perc = self.z_perc_baseline.value + z_perc     # Additive: preserve heterogeneity

        # Memory gate-bias (with spatial grouping broadcast)
        z_mem_raw = self.head_memory(h_mod_new)    # (num_groups,)
        z_mem = jnp.repeat(z_mem_raw, self.grouping_size, axis=-1)
        z_mem = z_mem[..., :self.target_hidden_size]
        z_mem = self.z_mem_baseline.value + z_mem

        # Temperature (bounded, §5.3a)
        z_act_raw = self.head_action(h_mod_new)    # (1,)
        temperature = jnp.clip(
            jax.nn.softplus(z_act_raw) + 0.5,      # Always >= 0.5
            self.temp_clip[0],
            self.temp_clip[1],
        )

        output = ModulatorOutput(
            z_percept=z_perc,
            z_memory=z_mem,
            temperature=temperature,
        )

        return output, h_mod_new

    def initial_state(self, batch_size: int = None) -> jnp.ndarray:
        """Returns the initial modulator hidden state (zeros)."""
        if batch_size is None:
            return jnp.zeros((self.mod_hidden_size,))
        return jnp.zeros((batch_size, self.mod_hidden_size))
