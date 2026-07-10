"""
ModulatedLayerNormGRUCell: A LayerNorm GRU cell with external gate-bias injection
for neuromodulation in the DreamerV3 RSSM.

Extends the LayerNormGRUCell from dreamer_v3_nnx.py with an optional gate_bias
parameter on the update gate (see NEUROMODULATION_ALGORITHM.md §5.1a).

When gate_bias is None, this is functionally identical to LayerNormGRUCell.

Architecture:
    gates_ih = LN(W_ih @ x)
    gates_hh = LN(W_hh @ h)
    gates = gates_ih + gates_hh
    reset, update, cand = split(gates, 3)
    update = sigmoid(update + gate_bias)   # <-- modulated
    h_new = (1 - update) * h + update * tanh(cand)
"""

import jax.numpy as jnp
from flax import nnx
from typing import Optional


class ModulatedLayerNormGRUCell(nnx.Module):
    """LayerNorm GRU cell with optional external gate-bias injection on the update gate.

    Follows the same structure as the DreamerV3 LayerNormGRUCell: combined input/hidden
    projections with LayerNorm applied before gate splitting.

    Args:
        hidden_size: Dimension of the hidden state (= deter_dim in RSSM).
        rngs: Flax NNX random number generators.
        apply_reset_gate:
            Mirrors LayerNormGRUCell.apply_reset_gate. See that
            docstring + the plan at
            docs/develop/active/diagnosis/dreamer_gru_reset_gate_fix.md
            for semantics. Default False keeps direct-import / test
            behaviour bit-identical pre-fix; the RSSM constructor passes
            the configured value explicitly via agent_config.
    """

    def __init__(self, hidden_size: int, rngs: nnx.Rngs,
                 apply_reset_gate: bool = False):
        self.hidden_size = hidden_size
        self.apply_reset_gate = apply_reset_gate

        self.dense_ih = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)
        self.dense_hh = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)

        self.ln_ih = nnx.LayerNorm(3 * hidden_size, rngs=rngs)
        self.ln_hh = nnx.LayerNorm(3 * hidden_size, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        h: jnp.ndarray,
        gate_bias: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input vector (projected to hidden_size by img_in), shape (..., hidden_size).
            h: Previous hidden state h_{t-1}, shape (..., hidden_size).
            gate_bias: Optional external bias for the update gate from the
                       neuromodulator, shape (..., hidden_size). Added to the
                       update gate pre-activation. None = no modulation.

        Returns:
            h_new: New hidden state, shape (..., hidden_size).
        """
        gates_ih = self.ln_ih(self.dense_ih(x))
        gates_hh = self.ln_hh(self.dense_hh(h))
        gates = gates_ih + gates_hh

        reset, update, cand = jnp.split(gates, 3, axis=-1)

        reset = nnx.sigmoid(reset)

        if gate_bias is not None:
            update = nnx.sigmoid(update + gate_bias)
        else:
            update = nnx.sigmoid(update)

        if self.apply_reset_gate:
            cand = jnp.tanh(reset * cand)
        else:
            cand = jnp.tanh(cand)

        h_new = (1.0 - update) * h + update * cand
        return h_new
