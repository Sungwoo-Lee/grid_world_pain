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
    RSSM                CP4  — recurrent state-space model (cascade fix #30 one-hidden-MLP)
                        CP4b — §S4 three-quantity arithmetic-mask reset (action + recurrent +
                               posterior; arithmetic-mask form, posterior reshape before mask)
"""
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple

from src.algorithms.dreamer_srl.utils import uniform_init_weights, init_weights


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
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Fused projection: [hx || input] → 3 * hidden_size
        # Matches sheeprl models.py:L365:
        #   self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        # bias defaults to True in models.py but when used inside RecurrentModel (agent.py:L322)
        # it is called with bias=False (LayerNorm in the GRU follows the projection).
        self.linear = nnx.Linear(
            input_size + hidden_size,
            3 * hidden_size,
            use_bias=use_bias,
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


# ---------------------------------------------------------------------------
# CP4 — RSSM (Recurrent State-Space Model) with cascade fix #30 one-hidden-MLP
# CP4b — §S4 three-quantity arithmetic-mask reset (action, recurrent, posterior)
# ---------------------------------------------------------------------------

class RSSM(nnx.Module):
    """Recurrent State-Space Model (RSSM) for DreamerV3.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L281-L498
    (RecurrentModel class L281-L341 + RSSM class L344-L498).

    ============================================================
    CP4 — RecurrentModel: MLP pre-projection BEFORE LayerNormGRUCell
    ============================================================
    Sheeprl wraps the GRU in a mandatory MLP pre-projection layer
    (sheeprl agent.py:L309-L326, RecurrentModel.__init__):

        self.mlp = MLP(
            input_dims=input_size,          # stochastic_size + action_dim
            output_dim=None,
            hidden_sizes=[dense_units],     # [recurrent_dense_units]
            activation=SiLU,
            layer_args={"bias": False},     # bias=False (LayerNorm follows)
            norm_layer=[LayerNorm],
            norm_args=[{"eps": 1e-3, "normalized_shape": dense_units}],
        )
        self.rnn = LayerNormGRUCell(
            dense_units,
            recurrent_state_size,
            bias=False,                     # bias=False in the GRU linear too
            ...
        )

    RecurrentModel.forward (L328-L341):
        feat = self.mlp(input)         # [B, recurrent_dense_units]
        out = self.rnn(feat, recurrent_state)  # [B, recurrent_state_size]
        return out

    The MLP pre-projection has the layer order (per MLP.miniblock() at models.py:L95):
        Linear(stochastic_size + action_dim, recurrent_dense_units, bias=False)
        LayerNorm(recurrent_dense_units, eps=1e-3)
        SiLU

    CRITICAL: without this MLP, the GRU receives raw [posterior_flat, action]
    (shape [B, stochastic_size + action_dim]) rather than the pre-projected
    [B, recurrent_dense_units]. This is a dimensional mismatch AND a semantic
    difference — the MLP is part of the Hafner architecture specification.

    ============================================================
    CP4 — Cascade fix #30: ONE hidden MLP layer (NOT bare Linear)
    ============================================================
    The transition and representation models are each a TWO-layer MLP:
        input → Linear(in, hidden) → LayerNorm(hidden, eps=1e-3) → SiLU
              → Linear(hidden, stochastic_size)
    This is "one hidden layer" (hidden_sizes=[transition_hidden_size]) plus
    an output linear.  A bare Linear (no hidden layer) would be cascade fix #30's
    wrong version — it silently trains but has different representational capacity
    than sheeprl's reference.

    Sheeprl's MLP construction (agent.py:L1021-L1051):
        transition_model = MLP(
            input_dims=recurrent_state_size,
            output_dim=stochastic_size,
            hidden_sizes=[world_model_cfg.transition_model.hidden_size],  ← ONE hidden
            activation=SiLU,
            layer_args={"bias": False},   ← bias=False (LayerNorm follows)
            norm_layer=[LayerNorm],
            norm_args=[{"eps": 1e-3, "normalized_shape": hidden_size}],
        )
    The resulting layer order per miniblock() at L95:
        Linear(in, hidden, bias=False) → LayerNorm(hidden, eps=1e-3) → SiLU
        Linear(hidden, stochastic_size)  ← final output layer, no norm/act

    The same structure applies to the representation model, except:
        input_dims = recurrent_state_size + encoder_output_dim  (concatenated)

    ============================================================
    CP4 — get_initial_states: returns mode, NOT a sample (no PRNG)
    ============================================================
    Sheeprl (agent.py:L391-L394):
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
        return initial_recurrent_state, initial_posterior

    `sample_state=False` → `compute_stochastic_state(..., sample=False)` → `.mode`
    This is deterministic (mode of a Categorical = argmax); NO PRNG key consumed.
    DO NOT add a `key` parameter to `get_initial_states`. If a `key` parameter is
    added by a future developer, tests will catch it (CP4 Test 3 checks the signature).

    ============================================================
    CP4b — §S4 three-quantity arithmetic-mask reset
    ============================================================
    Sheeprl (agent.py:L423-L430, inside RSSM.dynamic):
        action = (1 - is_first) * action
        initial_recurrent_state, initial_posterior = self.get_initial_states(...)
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)          ← reshape FIRST
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

    THREE quantities are reset (not two):
      1. action  = (1 - is_first) * action   (zeroed when is_first=1)
      2. recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent
      3. posterior is FIRST reshaped [B, S, D] → [B, S*D], THEN masked:
             posterior = (1 - is_first) * posterior + is_first * initial_posterior_flat

    FORM: arithmetic mask `(1 - is_first) * x + is_first * init`, NOT `jnp.where`.
    The two forms are numerically equivalent at infinite precision, but the arithmetic
    form matches sheeprl source 1:1 for bit-identity. `jnp.where` is forbidden for
    §S4 in this module (grep check enforced in CI).

    This reset lives in the `dynamic` rollout body / scan-step closure.  It is
    applied OUTSIDE the LayerNormGRUCell — the GRU cell does NOT receive `is_first`.

    `is_first` shape expected at scan-step level: `[B, 1]` (trailing singleton,
    per CP3b's buffer storage layout — one float per env per timestep).

    ============================================================
    `_uniform_mix`: unimix smoothing for categorical logits
    ============================================================
    Sheeprl (agent.py:L437-L449):
        logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
    The output is flattened logits of shape [..., S*D] (stochastic_size = S * D = 32*32).

    ============================================================
    Hafner init_weights: applied AT THE RSSM CONSTRUCTION SITE
    ============================================================
    Sheeprl (agent.py:L1057-L1065):
        rssm = RSSM(
            recurrent_model=recurrent_model.apply(init_weights),
            representation_model=representation_model.apply(init_weights),
            transition_model=transition_model.apply(init_weights),
            ...
        )
    In JAX/NNX we apply `init_weights` to each Linear layer's kernel immediately
    after construction (inside RSSM.__init__), NOT inside the sub-modules.
    This matches the sheeprl apply(init_weights) pattern which traverses the module
    tree and re-initializes every nn.Linear.

    Bit-identity tests:
        tests/algorithms/dreamer_srl/test_agent.py::test_rssm_transition_matches_sheeprl
        tests/algorithms/dreamer_srl/test_agent.py::test_rssm_representation_matches_sheeprl
        tests/algorithms/dreamer_srl/test_agent.py::test_get_initial_states_matches_sheeprl
        tests/algorithms/dreamer_srl/test_agent.py::test_is_first_force_set
        tests/algorithms/dreamer_srl/test_agent.py::test_is_first_three_quantity_reset

    Args:
        recurrent_state_size    (int): size of the GRU hidden state (hx), h_t.
        recurrent_dense_units   (int): width of the MLP pre-projection BEFORE the GRU
                                       (sheeprl RecurrentModel.mlp hidden dim).
                                       Must match sheeprl's recurrent_model.dense_units.
                                       256 for XS config.
        action_dim              (int): size of the action vector. Needed to size the
                                       recurrent pre-projection MLP input correctly
                                       (input = cat([posterior_flat, action])).
        stochastic_size         (int): total flat size of z_t = num_categoricals * num_classes
                                       (32 * 32 = 1024 for XS).
        transition_hidden_size  (int): width of the one hidden layer in the transition MLP.
                                       Cascade fix #30: must be an MLP hidden layer, not bare
                                       Linear. Default 256 (sheeprl XS).
        repr_hidden_size        (int): width of the one hidden layer in the repr MLP.
                                       Default 256 (sheeprl XS).
        num_categoricals        (int): number of categorical variables (S). Default 32.
        num_classes             (int): number of classes per categorical (D). Default 32.
        unimix                  (float): unimix smoothing coefficient. Default 0.01.
        encoder_output_dim      (int): dimension of the encoder embedding (for repr input).
        rngs                    (nnx.Rngs): NNX RNG container — used only in __init__.
    """

    def __init__(
        self,
        recurrent_state_size: int,
        recurrent_dense_units: int,
        action_dim: int,
        stochastic_size: int,
        transition_hidden_size: int,
        repr_hidden_size: int,
        num_categoricals: int,
        num_classes: int,
        encoder_output_dim: int,
        unimix: float = 0.01,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.recurrent_state_size = recurrent_state_size
        self.stochastic_size = stochastic_size
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.unimix = unimix

        # ---------------------------------------------------------------
        # Recurrent pre-projection MLP (before the GRU)
        # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L309-L326
        # (RecurrentModel.__init__ — the self.mlp block)
        #
        # Sheeprl RecurrentModel:
        #   self.mlp = MLP(
        #       input_dims=input_size,       # stochastic_size + action_dim
        #       hidden_sizes=[dense_units],  # ONE hidden layer = recurrent_dense_units
        #       activation=SiLU,
        #       layer_args={"bias": False},  # bias=False (LayerNorm follows)
        #       norm_layer=[LayerNorm],
        #       norm_args=[{"eps": 1e-3, "normalized_shape": dense_units}],
        #   )
        # MLP miniblock() at models.py:L95 produces:
        #   Linear(stochastic_size + action_dim, recurrent_dense_units, bias=False)
        #   LayerNorm(recurrent_dense_units, eps=1e-3)
        #   SiLU
        # ---------------------------------------------------------------
        recurrent_input_size = stochastic_size + action_dim
        self.recurrent_mlp_linear = nnx.Linear(
            recurrent_input_size, recurrent_dense_units,
            use_bias=False,   # bias=False: LayerNorm follows (matches sheeprl layer_args)
            rngs=rngs,
        )
        self.recurrent_mlp_norm = nnx.LayerNorm(
            num_features=recurrent_dense_units,
            epsilon=1e-3,
            rngs=rngs,
        )

        # ---------------------------------------------------------------
        # LayerNormGRUCell (recurrent core)
        # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L318-L325
        # (RecurrentModel.__init__ — the self.rnn block)
        #
        # Sheeprl:
        #   self.rnn = LayerNormGRUCell(
        #       dense_units,                # recurrent_dense_units (MLP output)
        #       recurrent_state_size,
        #       bias=False,                 # GRU linear also has bias=False
        #       layer_norm_cls=LayerNorm,
        #       layer_norm_kw={"eps": 1e-3},
        #   )
        # bias=False in the GRU's fused Linear (sheeprl RecurrentModel L322).
        # eps=1e-3 matches sheeprl production wire-up.
        # ---------------------------------------------------------------
        self.gru_cell = LayerNormGRUCell(
            input_size=recurrent_dense_units,  # output of MLP pre-projection
            hidden_size=recurrent_state_size,
            eps=1e-3,                          # must be 1e-3 per sheeprl production wire-up
            use_bias=False,                    # bias=False matches sheeprl RecurrentModel L322
            rngs=rngs,
        )

        # ---------------------------------------------------------------
        # Transition model (prior): hx → logits over z_t
        # Cascade fix #30: ONE hidden layer (hidden_sizes=[transition_hidden_size])
        # NOT a bare Linear. The layer order (matching sheeprl MLP + miniblock):
        #   Linear(recurrent_state_size, hidden_size, bias=False)
        #   LayerNorm(hidden_size, eps=1e-3)
        #   SiLU
        #   Linear(hidden_size, stochastic_size)   ← output linear, no norm/act
        # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1036-L1051
        # ---------------------------------------------------------------
        self.transition_hidden = nnx.Linear(
            recurrent_state_size, transition_hidden_size,
            use_bias=False,   # LayerNorm follows (bias=False matches sheeprl miniblock)
            rngs=rngs,
        )
        self.transition_norm = nnx.LayerNorm(
            num_features=transition_hidden_size,
            epsilon=1e-3,     # matches sheeprl mlp_layer_norm.kw.eps
            rngs=rngs,
        )
        self.transition_out = nnx.Linear(
            transition_hidden_size, stochastic_size,
            use_bias=True,
            rngs=rngs,
        )

        # ---------------------------------------------------------------
        # Representation model (posterior): cat(hx, obs_embed) → logits over z_t
        # Same one-hidden-layer structure as transition.
        # Input = recurrent_state_size + encoder_output_dim (concatenated)
        # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1017-L1035
        # ---------------------------------------------------------------
        repr_input_size = recurrent_state_size + encoder_output_dim
        self.repr_hidden = nnx.Linear(
            repr_input_size, repr_hidden_size,
            use_bias=False,   # LayerNorm follows
            rngs=rngs,
        )
        self.repr_norm = nnx.LayerNorm(
            num_features=repr_hidden_size,
            epsilon=1e-3,
            rngs=rngs,
        )
        self.repr_out = nnx.Linear(
            repr_hidden_size, stochastic_size,
            use_bias=True,
            rngs=rngs,
        )

        # ---------------------------------------------------------------
        # Learnable initial recurrent state (vector, shape [recurrent_state_size])
        # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L382-L385
        #   self.initial_recurrent_state = nn.Parameter(
        #       torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
        #   )
        # Initialized to all-zeros per sheeprl; learnable during training.
        # ---------------------------------------------------------------
        self.initial_recurrent_state = nnx.Param(
            jnp.zeros(recurrent_state_size, dtype=jnp.float32)
        )

        # ---------------------------------------------------------------
        # Apply Hafner init_weights to all Linear layers AT THE RSSM SITE
        # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1057-L1065
        #   rssm = RSSM(
        #       recurrent_model=recurrent_model.apply(init_weights),
        #       representation_model=representation_model.apply(init_weights),
        #       transition_model=transition_model.apply(init_weights),
        #       ...
        #   )
        # apply(init_weights) traverses the module tree and re-initializes
        # every nn.Linear. We do the same here per-layer explicitly.
        # The GRU cell's linear is also re-initialized.
        # ---------------------------------------------------------------
        key = rngs.params()  # draw a single PRNG key for the whole init sequence
        # Recurrent pre-projection linear
        key, k = jax.random.split(key)
        I, O = self.recurrent_mlp_linear.kernel[...].shape
        self.recurrent_mlp_linear.kernel = nnx.Param(init_weights(I, O, k))
        # GRU cell linear (fused [hx; x] → 3H projection)
        key, k = jax.random.split(key)
        I, O = self.gru_cell.linear.kernel[...].shape
        self.gru_cell.linear.kernel = nnx.Param(init_weights(I, O, k))
        # Transition hidden
        key, k = jax.random.split(key)
        I, O = self.transition_hidden.kernel[...].shape
        self.transition_hidden.kernel = nnx.Param(init_weights(I, O, k))
        # Transition output
        key, k = jax.random.split(key)
        I, O = self.transition_out.kernel[...].shape
        self.transition_out.kernel = nnx.Param(init_weights(I, O, k))
        # Repr hidden
        key, k = jax.random.split(key)
        I, O = self.repr_hidden.kernel[...].shape
        self.repr_hidden.kernel = nnx.Param(init_weights(I, O, k))
        # Repr output
        key, k = jax.random.split(key)
        I, O = self.repr_out.kernel[...].shape
        self.repr_out.kernel = nnx.Param(init_weights(I, O, k))

    # -------------------------------------------------------------------
    # _transition (prior): hx → logits + stochastic state (mode or sample)
    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L467-L480
    # -------------------------------------------------------------------
    def _transition(
        self,
        recurrent_state: jax.Array,
        sample_state: bool = True,
        key: jax.Array = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute prior logits and stochastic state from the recurrent state.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L467-L480
        (_transition method).

        Layer order (cascade fix #30 — one hidden MLP, NOT bare Linear):
            Linear(recurrent_state_size, hidden_size, bias=False)
            LayerNorm(hidden_size, eps=1e-3)
            SiLU
            Linear(hidden_size, stochastic_size)   ← logits
            _uniform_mix(logits)
            compute_stochastic_state(logits, sample=sample_state)

        Args:
            recurrent_state : [..., recurrent_state_size]
            sample_state    : if True, sample via straight-through estimator;
                              if False, return mode (argmax one-hot). Used
                              by get_initial_states to return a deterministic
                              initial posterior without consuming a PRNG key.
            key             : PRNG key for sampling; ignored when sample_state=False.

        Returns:
            logits : [..., stochastic_size]  (after _uniform_mix)
            state  : [..., num_categoricals, num_classes]  stochastic state
        """
        # One-hidden-layer MLP: Linear → LayerNorm → SiLU → Linear
        h = self.transition_hidden(recurrent_state)   # [..., hidden]
        h = self.transition_norm(h)                   # [..., hidden]
        h = jax.nn.silu(h)                            # [..., hidden]
        logits = self.transition_out(h)               # [..., stochastic_size]

        # Uniform mixing (unimix smoothing)
        logits = self._uniform_mix(logits)            # [..., stochastic_size]

        # Compute stochastic state
        state = self._compute_stochastic_state(logits, sample=sample_state, key=key)
        return logits, state

    # -------------------------------------------------------------------
    # _representation (posterior): cat(hx, obs_embed) → logits + state
    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L451-L465
    # -------------------------------------------------------------------
    def _representation(
        self,
        recurrent_state: jax.Array,
        embedded_obs: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute posterior logits and stochastic state from hx + obs embedding.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L451-L465
        (_representation method).

        Sheeprl L463:
            logits = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))

        Layer order (one hidden layer, same as transition):
            Linear(recurrent_state_size + encoder_dim, hidden_size, bias=False)
            LayerNorm(hidden_size, eps=1e-3)
            SiLU
            Linear(hidden_size, stochastic_size)
            _uniform_mix(logits)
            compute_stochastic_state(logits, sample=True)

        Args:
            recurrent_state : [..., recurrent_state_size]
            embedded_obs    : [..., encoder_output_dim]
            key             : PRNG key for straight-through sampling.

        Returns:
            logits : [..., stochastic_size]  (after _uniform_mix)
            state  : [..., num_categoricals, num_classes]  posterior state
        """
        # Concatenate hx + obs_embed (sheeprl L463)
        x = jnp.concatenate([recurrent_state, embedded_obs], axis=-1)  # [..., hx+enc]

        # One-hidden-layer MLP
        h = self.repr_hidden(x)       # [..., hidden]
        h = self.repr_norm(h)         # [..., hidden]
        h = jax.nn.silu(h)            # [..., hidden]
        logits = self.repr_out(h)     # [..., stochastic_size]

        # Uniform mixing
        logits = self._uniform_mix(logits)

        # Posterior state (sampled via straight-through estimator)
        state = self._compute_stochastic_state(logits, sample=True, key=key)
        return logits, state

    # -------------------------------------------------------------------
    # _uniform_mix: unimix smoothing for categorical logits
    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L437-L449
    # -------------------------------------------------------------------
    def _uniform_mix(self, logits: jax.Array) -> jax.Array:
        """Apply uniform mixing (unimix) to flatten logits.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L437-L449
        (_uniform_mix method).

        Sheeprl:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
            if self.unimix > 0.0:
                probs = logits.softmax(dim=-1)
                uniform = torch.ones_like(probs) / self.discrete
                probs = (1 - self.unimix) * probs + self.unimix * uniform
                logits = probs_to_logits(probs)
            logits = logits.view(*logits.shape[:-2], -1)

        `probs_to_logits(probs)` is `torch.log(probs)` for Categorical
        (no sigmoid; it's the log of the probability mass function).

        Args:
            logits: [..., stochastic_size]  flat logits

        Returns:
            mixed logits: [..., stochastic_size]  after unimix smoothing
        """
        # Reshape to [..., num_categoricals, num_classes]
        shape = logits.shape[:-1] + (self.num_categoricals, self.num_classes)
        logits = logits.reshape(shape)  # [..., S, D]

        if self.unimix > 0.0:
            probs = jax.nn.softmax(logits, axis=-1)               # [..., S, D]
            uniform = jnp.ones_like(probs) / self.num_classes     # [..., S, D]
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform  # [..., S, D]
            # probs_to_logits for Categorical = log(probs)
            logits = jnp.log(probs)                                # [..., S, D]

        # Flatten back to [..., stochastic_size]
        logits = logits.reshape(logits.shape[:-2] + (-1,))        # [..., S*D]
        return logits

    # -------------------------------------------------------------------
    # _compute_stochastic_state: sample or mode from categorical logits
    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v2/utils.py:L44-L62
    # (compute_stochastic_state, called inside _transition + _representation)
    # -------------------------------------------------------------------
    def _compute_stochastic_state(
        self,
        logits: jax.Array,
        sample: bool = True,
        key: jax.Array = None,
    ) -> jax.Array:
        """Sample or return mode of the Categorical stochastic state.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v2/utils.py:L44-L62
        (compute_stochastic_state, called by _transition and _representation).

        Sheeprl:
            logits = logits.view(*logits.shape[:-1], -1, discrete)
            dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
            stochastic_state = dist.rsample() if sample else dist.mode
            return stochastic_state

        JAX equivalent:
          - mode: one_hot(argmax(logits), num_classes) → [..., num_categoricals, num_classes]
          - sample: straight-through gumbel-softmax (equivalent to rsample() of
            OneHotCategoricalStraightThrough for the purposes of gradient flow)

        get_initial_states ALWAYS calls this with sample=False (no key needed).
        DO NOT add a key requirement when sample=False — it would break the
        determinism guarantee of initial-state generation.

        Args:
            logits  : [..., stochastic_size]  flat logits (after _uniform_mix)
            sample  : True → straight-through sample; False → argmax mode
            key     : PRNG key for sampling (required if sample=True; ignored if False)

        Returns:
            state : [..., num_categoricals, num_classes]
        """
        # Reshape flat logits → [..., S, D]
        shape = logits.shape[:-1] + (self.num_categoricals, self.num_classes)
        logits_2d = logits.reshape(shape)  # [..., S, D]

        if not sample:
            # Mode: argmax → one-hot (deterministic, no PRNG)
            # Matches sheeprl's `dist.mode` for OneHotCategorical
            indices = jnp.argmax(logits_2d, axis=-1)   # [..., S]
            state = jax.nn.one_hot(indices, self.num_classes)  # [..., S, D]
        else:
            # Straight-through Gumbel-softmax sample
            # Equivalent to OneHotCategoricalStraightThrough.rsample()
            # Forward pass: one_hot(argmax(logits + gumbel_noise), D)
            # Backward pass: gradient flows through softmax(logits)
            assert key is not None, (
                "_compute_stochastic_state: key required when sample=True. "
                "If you see this in get_initial_states, sample=False must be passed."
            )
            gumbel_noise = jax.random.gumbel(key, shape=logits_2d.shape)
            perturbed = logits_2d + gumbel_noise
            # Hard one-hot (stop-gradient for forward)
            hard_indices = jnp.argmax(perturbed, axis=-1)   # [..., S]
            hard = jax.nn.one_hot(hard_indices, self.num_classes)  # [..., S, D]
            # Soft for gradient flow (straight-through estimator)
            soft = jax.nn.softmax(logits_2d, axis=-1)       # [..., S, D]
            # STE: forward = hard, backward via soft
            state = hard - jax.lax.stop_gradient(soft) + soft  # [..., S, D]

        return state

    # -------------------------------------------------------------------
    # get_initial_states: deterministic initial (h0, z0) — NO PRNG consumed
    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L391-L394
    # -------------------------------------------------------------------
    def get_initial_states(
        self,
        batch_size: int,
    ) -> Tuple[jax.Array, jax.Array]:
        """Return deterministic initial recurrent state and initial posterior.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L391-L394
        (get_initial_states method).

        Sheeprl:
            initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
            initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
            return initial_recurrent_state, initial_posterior

        CP4 mandate — NO PRNG consumed:
            - `sample_state=False` → `_compute_stochastic_state(sample=False)` → mode (argmax)
            - The mode of a Categorical distribution is deterministic.
            - This function deliberately has NO `key` parameter.
            - A future developer adding `key` here would break reproducibility.
            - CP4 Test 3 checks the function signature to catch this.

        Args:
            batch_size (int): number of parallel environments B.

        Returns:
            initial_recurrent_state : [B, recurrent_state_size]
            initial_posterior       : [B, num_categoricals, num_classes]
        """
        # tanh of learnable initial state, expanded to batch
        # Sheeprl L392: torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        h0 = jnp.tanh(self.initial_recurrent_state[...])  # [recurrent_state_size]
        h0 = jnp.tile(h0[None, :], (batch_size, 1))         # [B, recurrent_state_size]

        # Run transition model in mode (not sample) to get initial posterior
        # Sheeprl L393: self._transition(initial_recurrent_state, sample_state=False)[1]
        # DO NOT pass a key — sample=False means no PRNG is needed or consumed.
        _, z0 = self._transition(h0, sample_state=False, key=None)  # [B, S, D]

        return h0, z0

    # -------------------------------------------------------------------
    # dynamic: one scan-step of the RSSM rollout (CP4b §S4 reset here)
    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L396-L435
    # -------------------------------------------------------------------
    def dynamic(
        self,
        posterior: jax.Array,
        recurrent_state: jax.Array,
        action: jax.Array,
        embedded_obs: jax.Array,
        is_first: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """One-step RSSM dynamic: §S4 reset + GRU + transition + representation.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L396-L435
        (RSSM.dynamic method).

        ============================================================
        §S4 THREE-QUANTITY ARITHMETIC-MASK RESET (CP4b)
        ============================================================
        Sheeprl L425-L430:
            action = (1 - is_first) * action
            initial_recurrent_state, initial_posterior = self.get_initial_states(...)
            recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
            posterior = posterior.view(*posterior.shape[:-2], -1)   ← reshape BEFORE mask
            posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        The THREE quantities that are reset:
          1. action         → zeroed when is_first=1
          2. recurrent_state → replaced by initial_recurrent_state when is_first=1
          3. posterior      → FIRST reshaped [B, S, D] → [B, S*D], THEN replaced

        FORM: arithmetic mask `(1 - is_first) * x + is_first * init`.
        NOT jnp.where — forbidden for §S4 per plan discipline (grep enforced in CI).

        is_first shape: [B, 1] (trailing singleton from buffer per CP3b layout).

        ============================================================
        THEN the RSSM forward:
        ============================================================
        Sheeprl L432-L435:
            recurrent_state = self.recurrent_model(cat(posterior, action), recurrent_state)
            prior_logits, prior = self._transition(recurrent_state)
            posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)

        Note: the recurrent_model forward (sheeprl agent.py:L328-L341) first passes
        cat(posterior_flat, action) through an MLP pre-projection (Linear + LayerNorm
        + SiLU → recurrent_dense_units), then through the LayerNormGRUCell.
        After the §S4 reset, `posterior` is already in flat [B, S*D] form.

        Args:
            posterior       : [B, num_categoricals, num_classes]  posterior from t-1
            recurrent_state : [B, recurrent_state_size]           h_{t-1}
            action          : [B, action_dim]                     a_{t-1} (already action-shifted by CP2b)
            embedded_obs    : [B, encoder_output_dim]             phi(o_t)
            is_first        : [B, 1]                              1.0 if first step after episode reset
            key             : PRNG key for posterior sampling

        Returns:
            recurrent_state  : [B, recurrent_state_size]      h_t
            posterior        : [B, num_categoricals, num_classes]  z_t (posterior)
            prior            : [B, num_categoricals, num_classes]  z_t (prior)
            posterior_logits : [B, stochastic_size]           logits for posterior
            prior_logits     : [B, stochastic_size]           logits for prior
        """
        # ---------------------------------------------------------------
        # §S4 THREE-QUANTITY ARITHMETIC-MASK RESET (CP4b)
        # Applied BEFORE the GRU forward pass.
        # is_first shape: [B, 1]
        # ---------------------------------------------------------------

        # Quantity 1: action → zeroed when is_first=1
        # Sheeprl L425: action = (1 - is_first) * action
        action = (1.0 - is_first) * action   # [B, A]

        # Get initial states for the reset (deterministic, no PRNG)
        batch_size = recurrent_state.shape[0]
        initial_recurrent_state, initial_posterior = self.get_initial_states(batch_size)
        # initial_posterior shape: [B, S, D] → need flat [B, S*D] for masking (step 3 below)

        # Quantity 2: recurrent_state → replaced by initial when is_first=1
        # Sheeprl L428: recurrent_state = (1-is_first)*recurrent_state + is_first*initial_recurrent_state
        recurrent_state = (
            (1.0 - is_first) * recurrent_state
            + is_first * initial_recurrent_state
        )  # [B, recurrent_state_size]

        # Quantity 3: posterior → RESHAPE FIRST, then arithmetic-mask reset
        # Sheeprl L429: posterior = posterior.view(*posterior.shape[:-2], -1)  ← flatten [S, D] → [S*D]
        # Sheeprl L430: posterior = (1-is_first)*posterior + is_first*initial_posterior.view_as(posterior)
        posterior_flat = posterior.reshape(
            posterior.shape[:-2] + (-1,)
        )  # [B, S*D]  ← reshape BEFORE mask (critical order)
        initial_posterior_flat = initial_posterior.reshape(
            initial_posterior.shape[:-2] + (-1,)
        )  # [B, S*D]
        posterior_flat = (
            (1.0 - is_first) * posterior_flat
            + is_first * initial_posterior_flat
        )  # [B, S*D]

        # ---------------------------------------------------------------
        # RecurrentModel forward: MLP pre-projection THEN GRU
        # Sheeprl L432: recurrent_state = self.recurrent_model(cat(posterior, action), recurrent_state)
        # sheeprl RecurrentModel.forward (agent.py:L328-L341):
        #   feat = self.mlp(input)                              # Linear + LayerNorm + SiLU
        #   out = self.rnn(feat, recurrent_state)               # LayerNormGRUCell
        #   return out
        # ---------------------------------------------------------------
        recurrent_input = jnp.concatenate([posterior_flat, action], axis=-1)  # [B, S*D+A]
        # MLP pre-projection: Linear(S*D+A → recurrent_dense_units) + LayerNorm + SiLU
        recurrent_feat = self.recurrent_mlp_linear(recurrent_input)  # [B, recurrent_dense_units]
        recurrent_feat = self.recurrent_mlp_norm(recurrent_feat)     # [B, recurrent_dense_units]
        recurrent_feat = jax.nn.silu(recurrent_feat)                  # [B, recurrent_dense_units]
        # GRU cell: feat → new recurrent state
        recurrent_state = self.gru_cell(recurrent_feat, recurrent_state)   # [B, hx]

        # ---------------------------------------------------------------
        # Transition (prior): hx → prior logits + prior state
        # Sheeprl L433: prior_logits, prior = self._transition(recurrent_state)
        # ---------------------------------------------------------------
        key, k_prior = jax.random.split(key)
        prior_logits, prior = self._transition(
            recurrent_state, sample_state=True, key=k_prior
        )  # prior_logits: [B, S*D], prior: [B, S, D]

        # ---------------------------------------------------------------
        # Representation (posterior): cat(hx, obs_embed) → posterior logits + state
        # Sheeprl L434: posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        # ---------------------------------------------------------------
        key, k_post = jax.random.split(key)
        posterior_logits, posterior = self._representation(
            recurrent_state, embedded_obs, key=k_post
        )  # posterior_logits: [B, S*D], posterior: [B, S, D]

        return recurrent_state, posterior, prior, posterior_logits, prior_logits
