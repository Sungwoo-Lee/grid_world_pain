"""agent.py — neural-network modules for the dreamer-srl v3 rebuild.

This module ports neural-network components from
sheeprl@33b6366 (vendor/sheeprl/) to JAX/Flax-NNX.

Isolation rule (v2 Risks §13):
    This module does NOT import from src.models.dreamer_v3_* or any other
    file in src/models/. All components are ported from sheeprl source.
    Only shared imports allowed: src.utils.config.Config (for config.get_mandatory)
    and src.environment.* (if needed in train.py).

NNX conventions (docs/develop/active/dreamer_srl_v3/NNX_CONVENTIONS.md):
    - Every nnx.Module.__init__ takes `rngs: nnx.Rngs` as its last positional arg.
    - rngs is NOT stored on the module — only used during __init__.
    - Forward-pass randomness uses explicit `key: jax.Array` args, NOT rngs.
    - JIT boundaries use nnx.split / nnx.merge / nnx.state / nnx.update.

Contents
--------
    LayerNormGRUCell    CP2  — recurrent GRU cell with LayerNorm + 1+1 fused gates
    action_shift        CP2b — §S2 action-shift: prepend-zero, drop-last
"""
import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# CP2 — LayerNormGRUCell
# ---------------------------------------------------------------------------

class LayerNormGRUCell(nnx.Module):
    """GRU cell with LayerNorm applied after the fused input projection.

    Ported from sheeprl@33b6366:sheeprl/models/models.py:L331-L410
    (LayerNormGRUCell class, including __init__ and forward).

    GOTCHA — reset-before-tanh (cascade fix #28):
        sheeprl applies the reset gate INSIDE the tanh for the candidate state:
            cand = tanh(reset * cand_proj)   ← reset multiplied BEFORE tanh
        NOT:
            cand = reset * tanh(cand_proj)   ← this would be WRONG
        The first form (reset inside tanh) produces a different activation basin
        than the second form. The difference is silent at training time because
        both are valid GRU variants — but sheeprl specifically uses the former,
        and matching it exactly is required for bit-identity parity.
        This trap is cascade fix #28 in the v2 plan:
            docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md#the-five-cascade-items

    Architecture:
        - 1+1 fused-gate form: a single linear projection [input; hx] → 3*hidden
          followed by LayerNorm, then split into (reset, cand, update).
        - Chunk order: (reset, cand, update) at indices [:H], [H:2H], [2H:3H].
        - Gate formulas:
            reset  = sigmoid(reset_proj)
            cand   = tanh(reset * cand_proj)           ← reset BEFORE tanh
            update = sigmoid(update_proj - 1)          ← bias shift -1
            hx_new = update * cand + (1 - update) * hx

    Bit-identity test:
        tests/algorithms/dreamer_srl/test_agent.py::test_layernorm_gru_cell_matches_sheeprl

    Args:
        input_size  (int): size of the input vector.
        hidden_size (int): size of the hidden (recurrent) state.
        rngs        (nnx.Rngs): Flax NNX RNG container — used only during __init__
                    for parameter initialization.  NOT stored on the module.
    """

    def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Fused projection: [hx || input] → 3 * hidden_size
        # Matches sheeprl L365:
        #   self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=True)
        self.linear = nnx.Linear(
            input_size + hidden_size,
            3 * hidden_size,
            use_bias=True,
            rngs=rngs,
        )

        # LayerNorm over the projected dimension (3 * hidden_size)
        # Matches sheeprl L368:
        #   self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)
        # where layer_norm_cls = nn.LayerNorm and layer_norm_kw = {}.
        self.layer_norm = nnx.LayerNorm(
            num_features=3 * hidden_size,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, hx: jax.Array) -> jax.Array:
        """Single-step GRU-LN forward pass.

        Args:
            x  : input  [B, input_size]
            hx : hidden state [B, hidden_size]

        Returns:
            hx_new : updated hidden state [B, hidden_size]

        Ported from sheeprl@33b6366:sheeprl/models/models.py:L370-L410
        (LayerNormGRUCell.forward).

        GOTCHA: reset gate is applied BEFORE tanh (see class docstring).
        """
        H = self.hidden_size

        # Sheeprl L396: input = torch.cat((hx, input), -1)
        # NOTE: sheeprl concatenates hx FIRST, then input.
        cat = jnp.concatenate([hx, x], axis=-1)  # [B, I+H]

        # Sheeprl L397-L398: x = self.linear(input); x = self.layer_norm(x)
        z = self.linear(cat)          # [B, 3H]
        z = self.layer_norm(z)        # [B, 3H]

        # Sheeprl L399-L403: chunk → gate formulas
        # reset, cand, update = torch.chunk(x, 3, -1)
        reset_proj  = z[..., :H]           # [B, H]
        cand_proj   = z[..., H:2 * H]      # [B, H]
        update_proj = z[..., 2 * H:]       # [B, H]

        reset  = jax.nn.sigmoid(reset_proj)
        # CRITICAL: reset multiplied INSIDE tanh — NOT reset * tanh(cand_proj)
        cand   = jnp.tanh(reset * cand_proj)
        # Sheeprl L402: update = torch.sigmoid(update - 1)  ← bias shift -1
        update = jax.nn.sigmoid(update_proj - 1.0)

        # Sheeprl L403: hx = update * cand + (1 - update) * hx
        hx_new = update * cand + (1.0 - update) * hx

        return hx_new


# ---------------------------------------------------------------------------
# CP2b — action_shift (§S2)
# ---------------------------------------------------------------------------

def action_shift(actions: jax.Array) -> jax.Array:
    """Shift actions by one time-step: prepend zeros, drop the last action.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L102-L104
    (training function, world-model rollout section).

    Call site in sheeprl (L104):
        batch_actions = torch.cat(
            (torch.zeros_like(data["actions"][:1]), data["actions"][:-1]),
            dim=0,
        )

    Purpose (§S2 — "silent training-loop semantic #2"):
        In the world-model rollout, the action at time t in the RSSM recurrence
        is the action that WAS TAKEN at t-1 (i.e. the action that produced obs_t),
        NOT the action taken AT t. This is correct because:
            obs_t = env.step(action_{t-1})
        so the action conditioning the RSSM transition that produced obs_t is
        action_{t-1}, not action_t.

        Concretely, for a sequence obs_0, obs_1, ..., obs_{T-1}:
            RSSM step at t=0 receives zeros (no prior action)
            RSSM step at t=k receives action_{k-1} (the action that produced obs_k)

    GOTCHA: if you wire actions[t] to the RSSM step at t (instead of actions[t-1]),
    the model learns to predict obs_{t+1} given obs_t conditioned on the wrong action.
    This is a silent divergence — the model still trains without crashing, but the
    world-model semantics are wrong. This was §S2 in the v2 plan:
        docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md#training-loop-semantics

    Args:
        actions: [T, B, A] float array of actions (unshifted).

    Returns:
        shifted: [T, B, A] float array where
                 shifted[0]  = zeros  (no prior action for the first step)
                 shifted[1:] = actions[:-1]

    Bit-identity test:
        tests/algorithms/dreamer_srl/test_agent.py::test_action_shift_matches_sheeprl
    """
    zeros = jnp.zeros_like(actions[:1])   # [1, B, A]
    return jnp.concatenate([zeros, actions[:-1]], axis=0)   # [T, B, A]
