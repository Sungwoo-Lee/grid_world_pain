"""
ModulatedGRUCell: A GRU cell with external gate-bias injection for neuromodulation.

When gate_bias is provided, it is added to the update gate pre-activation,
shifting the gate's operating point without introducing a second multiplicative
bottleneck (see NEUROMODULATION_ALGORITHM.md §5.1a).

When gate_bias is None or zeros, this is functionally identical to nnx.GRUCell.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional


class ModulatedGRUCell(nnx.Module):
    """GRU cell with optional external gate-bias injection on the update gate.

    The standard GRU equations are:
        r_t = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)           (reset gate)
        u_t = sigmoid(W_iu @ x + b_iu + W_hu @ h + b_hu + bias)    (update gate, modulated)
        n_t = tanh(W_in @ x + b_in + r_t * (W_hn @ h + b_hn))      (candidate)
        h_t = (1 - u_t) * h + u_t * n_t

    Where `bias` is the external gate_bias from the neuromodulator.
    """

    def __init__(self, in_features: int, hidden_size: int, *, rngs: nnx.Rngs):
        self.hidden_size = hidden_size

        # Input-to-hidden weights (reset, update, candidate)
        self.W_ir = nnx.Linear(in_features, hidden_size, use_bias=True, rngs=rngs)
        self.W_iu = nnx.Linear(in_features, hidden_size, use_bias=True, rngs=rngs)
        self.W_in = nnx.Linear(in_features, hidden_size, use_bias=True, rngs=rngs)

        # Hidden-to-hidden weights (reset, update, candidate)
        self.W_hr = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.W_hu = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.W_hn = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)

    def __call__(
        self,
        carry: jnp.ndarray,
        x: jnp.ndarray,
        gate_bias: Optional[jnp.ndarray] = None,
    ):
        """Forward pass.

        Args:
            carry: Previous hidden state h_{t-1}, shape (..., hidden_size).
            x: Input vector, shape (..., in_features).
            gate_bias: Optional external bias for the update gate from the
                       neuromodulator, shape (..., hidden_size). Added to the
                       update gate pre-activation. None = no modulation.

        Returns:
            (new_carry, output): Both are h_t with shape (..., hidden_size).
                                 Matches the (carry, y) convention of Flax RNN cells.
        """
        h = carry

        # Reset gate
        r = jax.nn.sigmoid(self.W_ir(x) + self.W_hr(h))

        # Update gate (with optional external modulation)
        u_pre = self.W_iu(x) + self.W_hu(h)
        if gate_bias is not None:
            u_pre = u_pre + gate_bias
        u = jax.nn.sigmoid(u_pre)

        # Candidate hidden state
        n = jnp.tanh(self.W_in(x) + r * self.W_hn(h))

        # New hidden state
        h_new = (1.0 - u) * h + u * n

        return h_new, h_new
