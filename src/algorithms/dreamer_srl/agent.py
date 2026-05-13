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
    RewardHead          CP3  — reward MLP head with zero-init output linear (cascade fix #27)
    CriticHead          CP3  — critic MLP head with zero-init output linear (cascade fix #27)
"""
import jax
import jax.numpy as jnp
from flax import nnx

from src.algorithms.dreamer_srl.utils import uniform_init_weights


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
        eps         (float): LayerNorm epsilon. Default 1e-3 matches sheeprl
                    production CP4 wire-up (agent.py:L305-L306), NOT the nnx
                    default 1e-6. Using nnx default produces 1.72e-3 forward-pass
                    drift vs the production RSSM, silently degrading CP4 parity.
        rngs        (nnx.Rngs): Flax NNX RNG container — used only during __init__
                    for parameter initialization.  NOT stored on the module.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        eps: float = 1e-3,
        *,
        rngs: nnx.Rngs,
    ) -> None:
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
        # where layer_norm_cls = nn.LayerNorm and layer_norm_kw = {"eps": 1e-3}.
        # epsilon=1e-3 matches sheeprl production CP4 wire-up (agent.py:L305-L306),
        # NOT the nnx default 1e-6.
        self.layer_norm = nnx.LayerNorm(
            num_features=3 * hidden_size,
            epsilon=eps,
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


# ---------------------------------------------------------------------------
# CP3 — RewardHead (zero-init output linear, cascade fix #27)
# ---------------------------------------------------------------------------

class RewardHead(nnx.Module):
    """Single-layer output head for the reward model with zero-init output linear.

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1170-L1180

    Cascade fix #27: the output linear layer (kernel + bias) is initialized to
    all-zeros via uniform_init_weights(scale=0.0). This forces the reward head to
    start as an uninformative predictor, preventing spurious high-magnitude gradient
    signals from random-init logits on the first training step.

    In sheeprl the zero-init is applied in the build_agent final-init phase (L1175):
        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))

    Architecture (this class exposes only the output linear — the full MLP body is
    wired in CP4. For CP3 we test only the zero-init discipline on the output linear):
        output_linear : Linear(in_features, out_features)  ← zero-init kernel + bias

    GOTCHA: uniform_init_weights(scale=0.0) produces limit = sqrt(3 * 0) = 0, so
    uniform(-0, 0) = 0.0 for all elements. This is the intended behavior (all-zeros
    is the sentinel for "uninformative at init"). The scale=0.0 case is not a
    degenerate input — it is cascade fix #27's explicit mechanism.

    Bit-identity test:
        tests/algorithms/dreamer_srl/test_agent.py::test_zero_init_reward_head_matches_sheeprl

    Args:
        in_features  (int): size of the MLP body's last hidden state (dense_units).
        out_features (int): number of two-hot bins (255 for DreamerV3 XS).
        key          (jax.Array): PRNG key used for zero-init (produces all-zeros
                     regardless of key value — supplied for API consistency with
                     uniform_init_weights).
        rngs         (nnx.Rngs): Flax NNX RNG container — used only during __init__
                     to construct the nnx.Linear; NOT stored on the module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        # Initialize the output linear with placeholder random init (rngs),
        # then immediately overwrite with zero-init kernel + zero bias.
        # This matches sheeprl's two-phase init:
        #   1. build_agent constructs the module (random or truncated-normal init)
        #   2. build_agent applies uniform_init_weights(0.0) to model[-1] (L1175)
        self.output_linear = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=True,
            rngs=rngs,
        )

        # Zero-init kernel (cascade fix #27) — uniform_init_weights(scale=0.0)
        # produces all-zeros (limit = sqrt(3 * 0 / denom) = 0).
        zero_kernel = uniform_init_weights(0.0, in_features, out_features, key)
        zero_bias = jnp.zeros((out_features,), dtype=jnp.float32)

        self.output_linear.kernel = nnx.Param(zero_kernel)
        self.output_linear.bias   = nnx.Param(zero_bias)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: linear projection of the final hidden state.

        Args:
            x : [..., in_features]

        Returns:
            logits : [..., out_features]  (two-hot bin logits over 255 bins)
        """
        return self.output_linear(x)


# ---------------------------------------------------------------------------
# CP3 — CriticHead (zero-init output linear, cascade fix #27)
# ---------------------------------------------------------------------------

class CriticHead(nnx.Module):
    """Single-layer output head for the critic with zero-init output linear.

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1170-L1180

    Cascade fix #27 (critic branch): the output linear layer is initialized to
    all-zeros via uniform_init_weights(scale=0.0). Same rationale as RewardHead —
    forces the critic to start as an uninformative value predictor.

    In sheeprl the zero-init is applied in the build_agent final-init phase (L1172):
        critic.model[-1].apply(uniform_init_weights(0.0))

    Architecture: mirrors RewardHead exactly (same in/out dimensions in XS config).

    GOTCHA: same as RewardHead — scale=0.0 is the intentional mechanism, not an edge
    case. The PRNG key is consumed but irrelevant (limit=0 → all-zeros output).

    Bit-identity test:
        tests/algorithms/dreamer_srl/test_agent.py::test_zero_init_critic_head_matches_sheeprl

    Args:
        in_features  (int): size of the MLP body's last hidden state (dense_units).
        out_features (int): number of two-hot bins (255 for DreamerV3 XS).
        key          (jax.Array): PRNG key (result is all-zeros regardless of value).
        rngs         (nnx.Rngs): Flax NNX RNG container — used only during __init__.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.Array,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        self.output_linear = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=True,
            rngs=rngs,
        )

        # Zero-init kernel (cascade fix #27)
        zero_kernel = uniform_init_weights(0.0, in_features, out_features, key)
        zero_bias = jnp.zeros((out_features,), dtype=jnp.float32)

        self.output_linear.kernel = nnx.Param(zero_kernel)
        self.output_linear.bias   = nnx.Param(zero_bias)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: linear projection of the final hidden state.

        Args:
            x : [..., in_features]

        Returns:
            logits : [..., out_features]  (two-hot bin logits over 255 bins)
        """
        return self.output_linear(x)
