"""agent.py — neural-network modules for the dreamer-srl v3 rebuild.

This module ports neural-network components from
sheeprl@33b6366 (vendor/sheeprl/) to JAX/Flax-NNX.

Isolation rule (v2 Risks §13):
    This module does NOT import from src.models.dreamer_v3_* or any other
    file in src/models/. All components are ported from sheeprl source.
    Only shared imports allowed: src.utils.config.Config (for config.get_mandatory)
    and src.environment.* (if needed in train.py).

NNX conventions (docs/develop/active/dreamer_srl_v1/NNX_CONVENTIONS.md):
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
    MLPEncoder          CP9  — MLP encoder for single-key vector observations (symlog input)
    MLPDecoder          CP9  — MLP decoder from latent state to observation dimension
    ContinueHead        CP9  — Bernoulli continue head (logits output)
    Actor               CP9  — discrete actor MLP with unimix + Hafner-init final layer
    WorldModel          CP9  — composite module: encoder + RSSM + decoder + reward + continue
    build_agent         CP9  — factory: constructs all modules, applies zero-init + Hafner-init
"""
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Dict, List, Tuple

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


# ---------------------------------------------------------------------------
# CP9 — MLPEncoder
# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L100-L153
# ---------------------------------------------------------------------------

class MLPEncoder(nnx.Module):
    """MLP encoder for single-key vector observations (no CNN — gridworld is vector obs).

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L100-L153
    # (MLPEncoder class, MLP-only single-key variant)

    Architecture per sheeprl MLPEncoder.__init__ (L137-L146) using MLP miniblock():
        For each of mlp_layers layers:
            Linear(in, dense_units, bias=False)   ← bias=False when LayerNorm follows
            LayerNorm(dense_units, eps=1e-3)
            SiLU
        Final layer: one more Linear(dense_units, dense_units, bias=False) + LN + SiLU
        (sheeprl MLP with hidden_sizes=[dense_units]*mlp_layers, output_dim=None)
        output_dim = dense_units (the final hidden size after the last hidden layer)

    Sheeprl symlog input (L150):
        x = torch.cat([symlog(obs[k]) if self.symlog_inputs else obs[k] for k in self.keys], -1)
    We apply symlog to the flat observation vector before the MLP.

    Init: apply Hafner init_weights to all Linear kernels (sheeprl L1129 encoder.apply(init_weights)).

    Args:
        obs_dim (int): flat observation vector size (sum of input_dims for all mlp keys).
        dense_units (int): width of each hidden layer (and encoder output_dim).
        mlp_layers (int): number of hidden layers.
        rngs (nnx.Rngs): NNX RNG container — used only during __init__.
    """

    def __init__(
        self,
        obs_dim: int,
        dense_units: int,
        mlp_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.obs_dim = obs_dim
        self.dense_units = dense_units
        self.mlp_layers = mlp_layers
        self.output_dim = dense_units  # matches sheeprl MLPEncoder.output_dim = dense_units

        # Build mlp_layers hidden layers: each is Linear(bias=False) + LayerNorm + SiLU
        # Sheeprl MLP with hidden_sizes=[dense_units]*mlp_layers, output_dim=None
        # miniblock order: Linear(in, hidden, bias=False) → LayerNorm → SiLU (L95)
        # NNX requires nnx.List for storing lists of submodules inside a Module.
        linears = []
        norms = []
        in_size = obs_dim
        for _ in range(mlp_layers):
            lin = nnx.Linear(in_size, dense_units, use_bias=False, rngs=rngs)
            norm = nnx.LayerNorm(num_features=dense_units, epsilon=1e-3, rngs=rngs)
            linears.append(lin)
            norms.append(norm)
            in_size = dense_units
        self.hidden_linears = nnx.List(linears)
        self.hidden_norms = nnx.List(norms)

        # Apply Hafner init_weights to all Linear kernels
        # Sheeprl L1129: encoder.apply(init_weights)
        key = rngs.params()
        for lin in self.hidden_linears:
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward pass: symlog(obs) → MLP hidden layers → dense_units output.

        Args:
            obs: [..., obs_dim] float array (raw observation, not symlog'd)

        Returns:
            embedded: [..., dense_units]
        """
        from src.algorithms.dreamer_srl.utils import symlog as _symlog
        x = _symlog(obs)
        for lin, norm in zip(self.hidden_linears, self.hidden_norms):
            x = lin(x)
            x = norm(x)
            x = jax.nn.silu(x)
        return x


# ---------------------------------------------------------------------------
# CP9 — MLPDecoder
# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L229-L279
# ---------------------------------------------------------------------------

class MLPDecoder(nnx.Module):
    """MLP decoder from latent state to reconstructed observation.

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L229-L279
    # (MLPDecoder class, single-key variant)

    Architecture per sheeprl MLPDecoder.__init__ (L265-L274):
        MLP body: mlp_layers hidden layers each:
            Linear(in, dense_units, bias=False)
            LayerNorm(dense_units, eps=1e-3)
            SiLU
        Output head: Linear(dense_units, obs_dim)  — matches sheeprl L274:
            self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])

    Forward (L276-L278):
        x = self.model(latent_states)
        return {k: h(x) for k, h in zip(self.keys, self.heads)}
    We return the raw output tensor (single key, no dict wrapping needed in caller).

    The output head uses Hafner init with scale=1.0 (sheeprl L1178:
        mlp_decoder.heads.apply(uniform_init_weights(1.0))
    )

    Args:
        latent_dim (int): input latent state size (stochastic_size + recurrent_state_size).
        obs_dim (int): output observation dimension.
        dense_units (int): hidden layer width.
        mlp_layers (int): number of hidden layers.
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        dense_units: int,
        mlp_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.dense_units = dense_units
        self.mlp_layers = mlp_layers

        # MLP body: mlp_layers hidden layers
        linears = []
        norms = []
        in_size = latent_dim
        for _ in range(mlp_layers):
            lin = nnx.Linear(in_size, dense_units, use_bias=False, rngs=rngs)
            norm = nnx.LayerNorm(num_features=dense_units, epsilon=1e-3, rngs=rngs)
            linears.append(lin)
            norms.append(norm)
            in_size = dense_units
        self.hidden_linears = nnx.List(linears)
        self.hidden_norms = nnx.List(norms)

        # Output head: Linear(dense_units, obs_dim) — sheeprl L274
        self.output_head = nnx.Linear(dense_units, obs_dim, use_bias=True, rngs=rngs)

        # Apply Hafner init_weights to all Linear kernels in the body
        # Sheeprl L1129: observation_model.apply(init_weights) (world_model contains decoder)
        key = rngs.params()
        for lin in self.hidden_linears:
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))
        # Output head: uniform_init_weights(1.0) — sheeprl L1178
        key, k = jax.random.split(key)
        I, O = self.output_head.kernel[...].shape
        self.output_head.kernel = nnx.Param(uniform_init_weights(1.0, I, O, k))

    def __call__(self, latent: jax.Array) -> jax.Array:
        """Forward pass: latent → MLP body → output head → obs_dim.

        Args:
            latent: [..., latent_dim]

        Returns:
            reconstructed_obs: [..., obs_dim] — symlog-space prediction
                (WP-SRL P3: the obs loss symlogs the TARGET, so the raw
                decoder output lives in symlog space; apply `symexp` for
                real-space values, cf. loss.SymlogDistribution.mode/mean).
        """
        x = latent
        for lin, norm in zip(self.hidden_linears, self.hidden_norms):
            x = lin(x)
            x = norm(x)
            x = jax.nn.silu(x)
        return self.output_head(x)


# ---------------------------------------------------------------------------
# D-018 — Modality-hierarchical encoder/decoder (opt-in; flat default untouched)
#
# Three designs (docs/develop/active/dreamer_srl_v2/
# hierarchical_modality_encoder_plan.md, user decision 2026-07-28):
#   1. flat (default)             — MLPEncoder + MLPDecoder above (sheeprl parity)
#   2. hierarchical + heads       — HierarchicalMLPEncoder + HeadsMLPDecoder
#   3. hierarchical + mirror      — HierarchicalMLPEncoder + MirrorMLPDecoder
#
# All new blocks use Dreamer's idiom — Linear(bias=False) → LayerNorm(eps=1e-3)
# → SiLU with Hafner truncated-normal init (init_weights) — NOT rPPO's
# Linear+bias → ReLU idiom. Output projections use uniform_init_weights(1.0)
# with bias, matching the flat decoder head (sheeprl L1178 idiom).
# No imports from src/models/recurrent_ppo_network.py (stacks stay decoupled);
# the GroupedLinear einsum pattern is re-implemented locally.
# Attribute names are deliberately DISJOINT from the flat modules'
# (branch_/hub_/trunk_/heads vs hidden_/output_head) so a flat checkpoint can
# never structurally restore into a hierarchical module or vice versa
# (plan-review N2 — cross-mode Orbax restore must fail loudly).
# ---------------------------------------------------------------------------

class HierGroupedLinear(nnx.Module):
    """Per-modality (grouped) linear layer via a single einsum kernel.

    Weights: [G, I, O] — one independent linear map per modality group,
    applied in a single kernel launch (pattern from
    src/models/recurrent_ppo_network.py:11-30, re-idiomed for dreamer_srl).

    Init: per-group Hafner truncated-normal (`init_weights`, the encoder/trunk
    idiom) or per-group `uniform_init_weights(1.0)` (the output-head idiom).
    Bias-free by default (a LayerNorm follows in every non-output position).

    Args:
        num_groups (int): number of modality groups G.
        in_features (int): per-group input width I.
        out_features (int): per-group output width O.
        use_bias (bool): add a per-group bias [G, O] (zeros init).
        init (str): 'hafner' | 'uniform1' — kernel initializer per group.
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        num_groups: int,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool,
        init: str,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        key = rngs.params()
        kernels = []
        for _ in range(num_groups):
            key, k = jax.random.split(key)
            if init == 'hafner':
                kernels.append(init_weights(in_features, out_features, k))
            elif init == 'uniform1':
                kernels.append(uniform_init_weights(1.0, in_features, out_features, k))
            else:
                raise ValueError(f"HierGroupedLinear: unknown init {init!r} "
                                 "(expected 'hafner' or 'uniform1')")
        self.weights = nnx.Param(jnp.stack(kernels))  # [G, I, O]
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((num_groups, out_features), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        """[..., G, I] → [..., G, O] (independent linear map per group)."""
        y = jnp.einsum('...gi,gio->...go', x, self.weights[...])
        if self.use_bias:
            y = y + self.bias[...]
        return y


class HierarchicalMLPEncoder(nnx.Module):
    """Per-modality branched encoder with a multimodal fusion hub (designs 2+3).

    Structure (every block: Linear(bias=False) → LayerNorm(eps=1e-3) → SiLU):
        symlog(obs)                                       [..., obs_dim]
          (uniform on all dims — professor-rl memo §3 ruling, no exemptions)
        static split by breakdown widths, zero-pad to max_in
          → stack                                         [..., G, max_in]
        branches: for h in default_mlp + [hidden_size]:
          HierGroupedLinear → LN → SiLU                   [..., G, hidden_size]
        concat                                            [..., G*hidden_size]
        hub: for h in multimodal_hub + [hidden_size]:
          Linear → LN → SiLU                              [..., hidden_size]

    output_dim = hidden_size — consumed by build_agent for the RSSM's
    representation-model input width (plan-review C3: must NOT be conflated
    with encoder.dense_units, which the flat path uses).

    Branch LayerNorm scale/bias are shared across groups (the LN normalises
    each group's feature vector independently; sharing parameters across
    groups mirrors rPPO's vmap-of-one-LN idiom at recurrent_ppo_network.py:120).

    vmap-safe: the split/pad uses only static shapes, so
    `jax.vmap(encoder)(obs[B, obs_dim])` works as used by WorldModel.observe.

    Args:
        observation_breakdown (dict): ordered {modality_name: width} from
            src/environment/sensor.py:get_observation_breakdown (order is
            parallel-by-construction with get_observation's assembly order).
        default_mlp (list[int]): per-modality branch hidden layer widths.
        multimodal_hub (list[int]): fusion-hub hidden layer widths.
        hidden_size (int): branch output width = hub output width = output_dim.
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        observation_breakdown: dict,
        default_mlp: list,
        multimodal_hub: list,
        hidden_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.modality_names = tuple(observation_breakdown.keys())
        self.modality_widths = tuple(int(w) for w in observation_breakdown.values())
        self.obs_dim = int(sum(self.modality_widths))
        self.num_groups = len(self.modality_widths)
        self.max_in = int(max(self.modality_widths))
        self.hidden_size = hidden_size
        self.output_dim = hidden_size  # C3: RSSM must consume THIS, not dense_units

        # Phase 1 — per-modality branches (grouped): default_mlp hidden widths,
        # ending in a grouped projection to hidden_size. Every layer LN+SiLU.
        branch_widths = list(default_mlp) + [hidden_size]
        b_lins, b_norms = [], []
        in_d = self.max_in
        for h in branch_widths:
            b_lins.append(HierGroupedLinear(
                self.num_groups, in_d, h, use_bias=False, init='hafner', rngs=rngs))
            b_norms.append(nnx.LayerNorm(num_features=h, epsilon=1e-3, rngs=rngs))
            in_d = h
        self.branch_linears = nnx.List(b_lins)
        self.branch_norms = nnx.List(b_norms)

        # Phase 2 — multimodal hub: concat(G × hidden_size) → multimodal_hub
        # widths → hidden_size. Every layer LN+SiLU, Hafner init.
        hub_widths = list(multimodal_hub) + [hidden_size]
        h_lins, h_norms = [], []
        in_d = self.num_groups * hidden_size
        key = rngs.params()
        for h in hub_widths:
            lin = nnx.Linear(in_d, h, use_bias=False, rngs=rngs)
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))
            h_lins.append(lin)
            h_norms.append(nnx.LayerNorm(num_features=h, epsilon=1e-3, rngs=rngs))
            in_d = h
        self.hub_linears = nnx.List(h_lins)
        self.hub_norms = nnx.List(h_norms)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward: symlog → split/pad → grouped branches → fusion hub.

        Args:
            obs: [..., obs_dim] float array (raw observation, not symlog'd)

        Returns:
            embedded: [..., hidden_size]
        """
        from src.algorithms.dreamer_srl.utils import symlog as _symlog
        x = _symlog(obs)  # uniform symlog on all dims (memo §3)

        batch_shape = x.shape[:-1]
        # Static split by breakdown widths + zero-pad to max_in → [..., G, max_in]
        x_padded = jnp.zeros(batch_shape + (self.num_groups, self.max_in), dtype=x.dtype)
        start = 0
        for i, width in enumerate(self.modality_widths):
            x_padded = x_padded.at[..., i, :width].set(x[..., start:start + width])
            start += width

        h = x_padded
        for lin, norm in zip(self.branch_linears, self.branch_norms):
            h = lin(h)
            h = norm(h)
            h = jax.nn.silu(h)
        # [..., G, hidden_size] → [..., G*hidden_size]
        fused = h.reshape(batch_shape + (self.num_groups * self.hidden_size,))
        for lin, norm in zip(self.hub_linears, self.hub_norms):
            fused = lin(fused)
            fused = norm(fused)
            fused = jax.nn.silu(fused)
        return fused


class HeadsMLPDecoder(nnx.Module):
    """Shared-trunk decoder with thin per-modality output heads (design 2).

    The trunk is structurally identical to the flat MLPDecoder's body
    (mlp_layers × [Linear(bias=False) → LN(eps=1e-3) → SiLU], Hafner init) and
    reads the SAME sizing source as the flat decoder (encoder dense_units /
    mlp_layers — plan-review N1: designs 1 vs 2 differ only in the head).
    The single Linear(dense_units → obs_dim) head is replaced by one
    Linear(dense_units → d_k) per modality, kernels uniform_init_weights(1.0)
    (sheeprl L274 + L1178 native multi-key idiom), concatenated in breakdown
    order — an exact reparameterisation of the single head (memo §6.1: same
    function class, parameter count, and gradients; only RNG draw order
    differs).

    Args:
        latent_dim (int): input latent size (stochastic + recurrent).
        observation_breakdown (dict): ordered {modality_name: width}.
        dense_units (int): trunk hidden width (same source as flat decoder).
        mlp_layers (int): trunk depth (same source as flat decoder).
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        latent_dim: int,
        observation_breakdown: dict,
        dense_units: int,
        mlp_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.modality_names = tuple(observation_breakdown.keys())
        self.modality_widths = tuple(int(w) for w in observation_breakdown.values())
        self.obs_dim = int(sum(self.modality_widths))
        self.dense_units = dense_units
        self.mlp_layers = mlp_layers

        # Shared trunk — same structure as flat MLPDecoder body
        t_lins, t_norms = [], []
        in_size = latent_dim
        for _ in range(mlp_layers):
            t_lins.append(nnx.Linear(in_size, dense_units, use_bias=False, rngs=rngs))
            t_norms.append(nnx.LayerNorm(num_features=dense_units, epsilon=1e-3, rngs=rngs))
            in_size = dense_units
        self.trunk_linears = nnx.List(t_lins)
        self.trunk_norms = nnx.List(t_norms)

        # Thin per-modality output heads (sheeprl L274 idiom)
        heads = []
        for width in self.modality_widths:
            heads.append(nnx.Linear(dense_units, width, use_bias=True, rngs=rngs))
        self.heads = nnx.List(heads)

        # Init: Hafner on trunk, uniform_init_weights(1.0) on heads (L1178)
        key = rngs.params()
        for lin in self.trunk_linears:
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))
        for head in self.heads:
            key, k = jax.random.split(key)
            I, O = head.kernel[...].shape
            head.kernel = nnx.Param(uniform_init_weights(1.0, I, O, k))

    def __call__(self, latent: jax.Array) -> jax.Array:
        """latent [..., latent_dim] → symlog-space obs prediction [..., obs_dim]."""
        x = latent
        for lin, norm in zip(self.trunk_linears, self.trunk_norms):
            x = lin(x)
            x = norm(x)
            x = jax.nn.silu(x)
        outs = [head(x) for head in self.heads]
        return jnp.concatenate(outs, axis=-1)  # breakdown order


class MirrorMLPDecoder(nnx.Module):
    """Depth-symmetric per-modality decoder (design 3).

    Mirrors the HierarchicalMLPEncoder back-to-front:
        latent → hub-mirror: for h in reversed(multimodal_hub):
                   Linear(bias=False) → LN → SiLU
                 grouped expansion Linear(→ G*hidden_size) → LN → SiLU
        reshape                                            [..., G, hidden_size]
        branches: for h in reversed(default_mlp):
                   HierGroupedLinear(bias=False) → LN → SiLU
        per-modality output: Linear(branch_width → d_k), kernels
        uniform_init_weights(1.0) + bias, concat in breakdown order → [..., obs_dim]

    The memo (§6.2) predicts the 1-dim branches starve on 1/27 of the
    reconstruction signal; this class exists to test that empirically.

    Args:
        latent_dim (int): input latent size.
        observation_breakdown (dict): ordered {modality_name: width}.
        default_mlp (list[int]): encoder branch widths (reversed here).
        multimodal_hub (list[int]): encoder hub widths (reversed here).
        hidden_size (int): per-branch entry width (= encoder branch output).
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        latent_dim: int,
        observation_breakdown: dict,
        default_mlp: list,
        multimodal_hub: list,
        hidden_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.modality_names = tuple(observation_breakdown.keys())
        self.modality_widths = tuple(int(w) for w in observation_breakdown.values())
        self.obs_dim = int(sum(self.modality_widths))
        self.num_groups = len(self.modality_widths)
        self.hidden_size = hidden_size

        # Hub-mirror: reversed hub widths, then grouped expansion to G*hidden_size
        hub_widths = list(reversed(multimodal_hub)) + [self.num_groups * hidden_size]
        m_lins, m_norms = [], []
        in_d = latent_dim
        for h in hub_widths:
            m_lins.append(nnx.Linear(in_d, h, use_bias=False, rngs=rngs))
            m_norms.append(nnx.LayerNorm(num_features=h, epsilon=1e-3, rngs=rngs))
            in_d = h
        self.mirror_hub_linears = nnx.List(m_lins)
        self.mirror_hub_norms = nnx.List(m_norms)

        # Per-modality branches (grouped): reversed default_mlp widths
        branch_widths = list(reversed(default_mlp))
        b_lins, b_norms = [], []
        in_d = hidden_size
        for h in branch_widths:
            b_lins.append(HierGroupedLinear(
                self.num_groups, in_d, h, use_bias=False, init='hafner', rngs=rngs))
            b_norms.append(nnx.LayerNorm(num_features=h, epsilon=1e-3, rngs=rngs))
            in_d = h
        self.branch_linears = nnx.List(b_lins)
        self.branch_norms = nnx.List(b_norms)
        self._branch_out_width = in_d

        # Per-modality output projections: Linear(branch_width → d_k)
        heads = []
        for width in self.modality_widths:
            heads.append(nnx.Linear(in_d, width, use_bias=True, rngs=rngs))
        self.output_heads = nnx.List(heads)

        # Init: Hafner on hub-mirror linears (branch HierGroupedLinears are
        # Hafner-init in their own __init__); uniform_init_weights(1.0) on outputs
        key = rngs.params()
        for lin in self.mirror_hub_linears:
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))
        for head in self.output_heads:
            key, k = jax.random.split(key)
            I, O = head.kernel[...].shape
            head.kernel = nnx.Param(uniform_init_weights(1.0, I, O, k))

    def __call__(self, latent: jax.Array) -> jax.Array:
        """latent [..., latent_dim] → symlog-space obs prediction [..., obs_dim]."""
        batch_shape = latent.shape[:-1]
        x = latent
        for lin, norm in zip(self.mirror_hub_linears, self.mirror_hub_norms):
            x = lin(x)
            x = norm(x)
            x = jax.nn.silu(x)
        # [..., G*hidden_size] → [..., G, hidden_size]
        h = x.reshape(batch_shape + (self.num_groups, self.hidden_size))
        for lin, norm in zip(self.branch_linears, self.branch_norms):
            h = lin(h)
            h = norm(h)
            h = jax.nn.silu(h)
        outs = [self.output_heads[i](h[..., i, :]) for i in range(self.num_groups)]
        return jnp.concatenate(outs, axis=-1)  # breakdown order


# ---------------------------------------------------------------------------
# CP9 — ContinueHead
# Sheeprl builds inline at agent.py:L1114-L1127 (discount_model = MLP(..., output_dim=1, ...))
# and L1176: world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
# ---------------------------------------------------------------------------

class ContinueHead(nnx.Module):
    """Continue (discount) head — Bernoulli logit over latent state.

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1114-L1127
    # (continue_model / discount_model inline construction in build_agent)

    Architecture: same as RewardHead / CriticHead structure but output_dim=1.
        mlp_layers hidden layers:
            Linear(in, dense_units, bias=False)
            LayerNorm(dense_units, eps=1e-3)
            SiLU
        Output linear: Linear(dense_units, 1)
            Hafner init_weights(1.0) applied to output linear (sheeprl L1176)

    Forward returns raw logit (scalar per batch element). The IndependentBernoulli
    wrap happens in train.py / one_train_step (§S9 compliance).

    Args:
        latent_dim (int): input latent state size.
        dense_units (int): hidden layer width.
        mlp_layers (int): number of hidden layers.
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        latent_dim: int,
        dense_units: int,
        mlp_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.dense_units = dense_units
        self.mlp_layers = mlp_layers

        # MLP body
        linears = []
        norms = []
        in_size = latent_dim
        for _ in range(mlp_layers):
            lin = nnx.Linear(in_size, dense_units, use_bias=False, rngs=rngs)
            norm = nnx.LayerNorm(num_features=dense_units, epsilon=1e-3, rngs=rngs)
            linears.append(lin)
            norms.append(norm)
            in_size = dense_units
        self.hidden_linears = nnx.List(linears)
        self.hidden_norms = nnx.List(norms)

        # Output linear → single logit
        self.output_linear = nnx.Linear(dense_units, 1, use_bias=True, rngs=rngs)

        # Apply Hafner init_weights to body linears
        # Sheeprl L1129: continue_model.apply(init_weights) via world_model
        key = rngs.params()
        for lin in self.hidden_linears:
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))
        # Output linear: uniform_init_weights(1.0) — sheeprl L1176
        key, k = jax.random.split(key)
        I, O = self.output_linear.kernel[...].shape
        self.output_linear.kernel = nnx.Param(uniform_init_weights(1.0, I, O, k))

    def __call__(self, latent: jax.Array) -> jax.Array:
        """Forward pass: latent → MLP body → scalar logit.

        Args:
            latent: [..., latent_dim]

        Returns:
            logits: [..., 1]  (single Bernoulli logit per batch element)
        """
        x = latent
        for lin, norm in zip(self.hidden_linears, self.hidden_norms):
            x = lin(x)
            x = norm(x)
            x = jax.nn.silu(x)
        return self.output_linear(x)


# ---------------------------------------------------------------------------
# CP9 — Actor
# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L694-L846
# (Actor class, discrete-action branch only — gridworld is always discrete)
# ---------------------------------------------------------------------------

class Actor(nnx.Module):
    """Discrete actor — MLP body + categorical head over actions.

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L694-L846
    # (Actor class, discrete-action branch only — grid-world is always discrete)

    Architecture (sheeprl L761-L774):
        MLP body: mlp_layers hidden layers each:
            Linear(in, dense_units, bias=False)
            LayerNorm(dense_units, eps=1e-3)
            SiLU
        Output head: Linear(dense_units, action_dim) with Hafner init scale=1.0
            sheeprl L1171: actor.mlp_heads.apply(uniform_init_weights(1.0))
        Then apply Hafner init_weights (truncated-normal) to all body linears.
            sheeprl L1167-L1168: actor.apply(init_weights)

    Unimix (sheeprl L839-L845):
        if unimix > 0:
            probs = logits.softmax(dim=-1)
            uniform = ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            logits = log(probs)  # probs_to_logits for Categorical

    Forward returns (actions, log_probs):
        actions: [..., action_dim] one-hot via straight-through Gumbel-softmax
        log_probs: [..., 1] sum of log-probs (scalar per batch, matching sheeprl)
        entropy: [...] entropy of the Categorical distribution

    §S7 sg(action) discipline: the caller (one_train_step) must apply
        jax.lax.stop_gradient(actions) before passing to log_prob computation.
        This class computes log_probs internally from the distribution logits
        (not from the stop_gradient'd action) so gradients flow through the actor.

    Args:
        latent_dim (int): input latent state size (stochastic_size + recurrent_state_size).
        action_dim (int): total action space size (for single discrete action head).
        dense_units (int): hidden layer width.
        mlp_layers (int): number of hidden layers.
        unimix (float): uniform mixing coefficient.
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        dense_units: int,
        mlp_layers: int,
        unimix: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.dense_units = dense_units
        self.mlp_layers = mlp_layers
        self.unimix = unimix

        # MLP body: mlp_layers hidden layers
        linears = []
        norms = []
        in_size = latent_dim
        for _ in range(mlp_layers):
            lin = nnx.Linear(in_size, dense_units, use_bias=False, rngs=rngs)
            norm = nnx.LayerNorm(num_features=dense_units, epsilon=1e-3, rngs=rngs)
            linears.append(lin)
            norms.append(norm)
            in_size = dense_units
        self.hidden_linears = nnx.List(linears)
        self.hidden_norms = nnx.List(norms)

        # Output head: Linear(dense_units, action_dim)
        # sheeprl L774: self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, action_dim)])
        self.output_linear = nnx.Linear(dense_units, action_dim, use_bias=True, rngs=rngs)

        # Hafner init: apply init_weights to all body linears (sheeprl L1167: actor.apply(init_weights))
        key = rngs.params()
        for lin in self.hidden_linears:
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))
        # Output head: uniform_init_weights(1.0) — sheeprl L1171: actor.mlp_heads.apply(uniform_init_weights(1.0))
        key, k = jax.random.split(key)
        I, O = self.output_linear.kernel[...].shape
        self.output_linear.kernel = nnx.Param(uniform_init_weights(1.0, I, O, k))

    def _uniform_mix(self, logits: jax.Array) -> jax.Array:
        """Apply unimix smoothing to logits — sheeprl Actor._uniform_mix (L839-L845).

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L839-L845.
        """
        if self.unimix > 0.0:
            probs = jax.nn.softmax(logits, axis=-1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
            logits = jnp.log(probs)  # probs_to_logits for Categorical
        return logits

    def forward_logits(self, latent: jax.Array) -> jax.Array:
        """Return post-unimix logits for a given latent, WITHOUT sampling.

        Used by the actor-loss path in train.py to compute log_prob on the
        rollout-time imagined actions (not on a freshly-sampled action).
        Gradient flows through the actor MLP parameters via this method.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L828-L835
        (the logits path inside Actor.forward, before the rsample() call).
        Authorized by: docs/reviews/dreamer_srl_v2_cp3_actor_objective_review.md §2.4
                       docs/reviews/dreamer_srl_v2_cp6_orchestrator_review.md §3.5

        Args:
            latent: [..., latent_dim]

        Returns:
            logits: [..., action_dim] post-unimix logits (equivalent to the
                    logits fed into OneHotCategoricalStraightThrough in sheeprl).
        """
        x = latent
        for lin, norm in zip(self.hidden_linears, self.hidden_norms):
            x = lin(x)
            x = norm(x)
            x = jax.nn.silu(x)
        raw_logits = self.output_linear(x)
        return self._uniform_mix(raw_logits)

    def __call__(
        self, latent: jax.Array, key: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass: latent → MLP → logits → action + log_prob + entropy.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L783-L837
        (Actor.forward, discrete branch).

        §S7 stop_gradient discipline:
            log_probs returned here are computed from the DISTRIBUTION LOGITS
            (with gradient flowing through the actor parameters).
            The CALLER must apply jax.lax.stop_gradient(actions) before passing
            to compute_actor_objective — the stop_gradient is on the sampled
            action TOKEN, not on the log_prob computation itself.

        Args:
            latent: [..., latent_dim]
            key: PRNG key for straight-through Gumbel-softmax sampling

        Returns:
            actions: [..., action_dim] one-hot via straight-through Gumbel-softmax
            log_probs: [..., 1] sum of log-probs over action dim (matching sheeprl L286)
            entropy: [...] entropy of the Categorical distribution
        """
        # MLP body + output head + unimix (reuse forward_logits for DRY principle)
        logits = self.forward_logits(latent)  # [..., action_dim] post-unimix

        # Straight-through Gumbel-softmax sample (sheeprl L834: actions_dist[-1].rsample())
        gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
        perturbed = logits + gumbel_noise
        hard_indices = jnp.argmax(perturbed, axis=-1)  # [...] int indices
        hard = jax.nn.one_hot(hard_indices, self.action_dim)  # [..., action_dim]
        soft = jax.nn.softmax(logits, axis=-1)
        # STE: forward = hard, backward through soft
        actions = hard - jax.lax.stop_gradient(soft) + soft   # [..., action_dim]

        # Log-prob from distribution logits (NOT from stop_gradient'd action)
        # sheeprl L286: p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
        # For discrete: log_prob = sum(one_hot * log_softmax(logits), axis=-1)
        # We use STOP_GRADIENT on the action token here to match §S7 from the actor side.
        # The caller's compute_actor_objective receives these log_probs and uses them.
        log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)  # [..., action_dim]
        # stop_gradient on the sampled action (§S7 sg(action) discipline)
        sg_actions = jax.lax.stop_gradient(actions)
        log_probs = jnp.sum(sg_actions * log_softmax_logits, axis=-1, keepdims=True)  # [..., 1]

        # Entropy of Categorical: -sum(p * log p, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)  # [...]

        return actions, log_probs, entropy


# ---------------------------------------------------------------------------
# CP9 — WorldModel
# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L600-L692
# (PlayerDV3 class for the inference path + dreamer_v3.py train() for the train path)
# ---------------------------------------------------------------------------

class WorldModel(nnx.Module):
    """Composite world model: encoder + RSSM + decoder + reward head + continue head.

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L596-L692
    # (WorldModel container class + PlayerDV3 inference path)
    # Training-time observe() and imagination() follow dreamer_v3.py:L107-L241.

    Holds references to:
        encoder      — MLPEncoder (obs → dense_units embedding)
        rssm         — RSSM (observe/imagine dynamics)
        decoder      — MLPDecoder (latent → reconstructed obs)
        reward_model — RewardHead (latent → 255-bin reward logits, zero-init)
        continue_model — ContinueHead (latent → 1-dim Bernoulli logit)

    observe():
        Training-time forward. Runs encoder over the full [T, B] sequence, then
        RSSM.dynamic step-by-step (Python loop for T steps, like sheeprl L134-L145),
        returns all RSSM outputs + decoder/reward/continue predictions.

    imagine():
        Imagination rollout. Starts from posterior latent states, rolls forward
        `horizon` steps using actor's sampled actions through RSSM.imagination
        (transition model only — no observation encoder during imagination).

    Note on RSSM.imagination (sheeprl agent.py:L396 via dreamer_v3.py:L236):
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        This is the RSSM's prior-only step (no observation conditioning).
        In our JAX RSSM, the equivalent is _transition(recurrent_state) after running
        the recurrent step with the action.

    Args:
        encoder (MLPEncoder): the observation encoder.
        rssm (RSSM): the recurrent state-space model.
        decoder (MLPDecoder): the observation decoder.
        reward_model (RewardHead): the reward head.
        continue_model (ContinueHead): the continue head.
    """

    def __init__(
        self,
        encoder: "MLPEncoder",
        rssm: RSSM,
        decoder: "MLPDecoder",
        reward_model: RewardHead,
        continue_model: "ContinueHead",
    ) -> None:
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_model = reward_model
        self.continue_model = continue_model

    def observe(
        self,
        obs: jax.Array,
        actions: jax.Array,
        is_first: jax.Array,
        key: jax.Array,
    ) -> Dict[str, jax.Array]:
        """Training-time world-model forward over a [T, B] sequence.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L112-L149
        (train() dynamic-learning section).

        Steps:
        1. Encode all T observations: embedded_obs = encoder(obs)  → [T, B, dense_units]
        2. RSSM observe loop (Python for-loop, T steps):
           a. rssm.dynamic(posterior, recurrent_state, actions[t], embedded_obs[t], is_first[t], key_t)
           b. Collect recurrent_states, posterior_logits, prior_logits, posteriors
        3. Build latent_states = cat(posteriors_flat, recurrent_states) → [T, B, latent_dim]
        4. Compute decoder, reward, continue predictions.

        Args:
            obs:       [T, B, obs_dim] float — raw observations (symlog applied in encoder)
            actions:   [T, B, action_dim] float — action-shifted (§S2 already applied by caller)
            is_first:  [T, B, 1] float — is-first flags (§S1 already applied by caller)
            key:       PRNG key — split into T per-step keys

        Returns:
            dict with keys:
                latent_states:       [T, B, latent_dim]
                posteriors:          [T, B, num_cat, num_cls]
                recurrent_states:    [T, B, recurrent_state_size]
                posterior_logits:    [T, B, stochastic_size]
                prior_logits:        [T, B, stochastic_size]
                reconstructed_obs:   [T, B, obs_dim] — symlog-space (WP-SRL P3);
                                     apply `symexp` for real-space values
                reward_logits:       [T, B, bins]
                continue_logits:     [T, B, 1]
        """
        T, B = obs.shape[:2]
        stochastic_size = self.rssm.stochastic_size
        num_categoricals = self.rssm.num_categoricals
        num_classes = self.rssm.num_classes
        recurrent_state_size = self.rssm.recurrent_state_size

        # Encode all observations
        # sheeprl L113: embedded_obs = world_model.encoder(batch_obs)
        embedded_obs = jax.vmap(self.encoder)(obs.reshape(T * B, -1)).reshape(T, B, -1)

        # Initialize RSSM state
        # sheeprl L108-L109: recurrent_state = zeros; posterior = zeros with [S,D] shape
        recurrent_state, posterior_state = self.rssm.get_initial_states(B)
        # recurrent_state: [B, recurrent_state_size], posterior_state: [B, S, D]

        # Split key into T per-step keys
        step_keys = jax.random.split(key, T)

        # Python loop over sequence (matches sheeprl L134-L145)
        all_recurrent = []
        all_posterior = []
        all_posterior_logits = []
        all_prior_logits = []

        for t in range(T):
            recurrent_state, posterior_state, prior_state, post_logits, prior_logits = \
                self.rssm.dynamic(
                    posterior_state,
                    recurrent_state,
                    actions[t],      # [B, action_dim] — already action-shifted
                    embedded_obs[t], # [B, dense_units]
                    is_first[t],     # [B, 1]
                    step_keys[t],
                )
            all_recurrent.append(recurrent_state)
            all_posterior.append(posterior_state)
            all_posterior_logits.append(post_logits)
            all_prior_logits.append(prior_logits)

        # Stack into [T, B, ...] tensors
        recurrent_states = jnp.stack(all_recurrent, axis=0)    # [T, B, recurrent_state_size]
        posteriors = jnp.stack(all_posterior, axis=0)           # [T, B, S, D]
        posterior_logits = jnp.stack(all_posterior_logits, axis=0)  # [T, B, S*D]
        prior_logits = jnp.stack(all_prior_logits, axis=0)     # [T, B, S*D]

        # Build latent states (sheeprl L146):
        # latent_states = cat(posteriors.view(T, B, -1), recurrent_states, dim=-1)
        posteriors_flat = posteriors.reshape(T, B, -1)            # [T, B, S*D]
        latent_states = jnp.concatenate([posteriors_flat, recurrent_states], axis=-1)  # [T, B, latent_dim]

        # Decoder, reward, continue predictions (sheeprl L149-L167)
        # Reshape to [T*B, latent_dim] for batched forward, then reshape back
        latent_flat = latent_states.reshape(T * B, -1)
        # N2 note: symlog-space prediction (WP-SRL P3) — apply symexp for real-space obs
        reconstructed_obs = jax.vmap(self.decoder)(latent_flat).reshape(T, B, -1)
        reward_logits = jax.vmap(self.reward_model)(latent_flat).reshape(T, B, -1)
        continue_logits = jax.vmap(self.continue_model)(latent_flat).reshape(T, B, 1)

        return {
            "latent_states":     latent_states,
            "posteriors":        posteriors,
            "recurrent_states":  recurrent_states,
            "posterior_logits":  posterior_logits,
            "prior_logits":      prior_logits,
            "reconstructed_obs": reconstructed_obs,
            "reward_logits":     reward_logits,
            "continue_logits":   continue_logits,
        }

    def imagine(
        self,
        init_latent: jax.Array,
        actor: "Actor",
        horizon: int,
        key: jax.Array,
    ) -> Dict[str, jax.Array]:
        """Imagination rollout for behavior learning.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L202-L244
        (train() imagination-rollout section).

        Starts from the initial latent state (posterior from observe()), runs
        `horizon` steps using actor's sampled actions through the RSSM transition
        model (prior-only — no observation encoder).

        Args:
            init_latent:  [BT, latent_dim] — flattened posterior latent from observe()
                          (posteriors_flat + recurrent_states concatenated, then reshaped)
            actor:        the Actor module
            horizon:      number of imagination steps (H)
            key:          PRNG key

        Returns:
            dict with keys:
                imagined_latents:  [H+1, BT, latent_dim]
                imagined_actions:  [H+1, BT, action_dim]
                imagined_log_probs:[H+1, BT, 1]   log_probs from actor
                imagined_entropies:[H+1, BT]       entropy from actor
        """
        BT = init_latent.shape[0]
        latent_dim = init_latent.shape[1]
        stochastic_size = self.rssm.stochastic_size      # S*D flat
        recurrent_state_size = self.rssm.recurrent_state_size

        imagined_latents = [init_latent]                  # [BT, latent_dim]
        imagined_actions = []
        imagined_log_probs = []
        imagined_entropies = []

        # Decode initial latent into prior + recurrent components
        # init_latent = cat(posterior_flat [BT, S*D], recurrent [BT, H])
        imagined_prior_flat = init_latent[:, :stochastic_size]   # [BT, S*D]
        recurrent_state = init_latent[:, stochastic_size:]        # [BT, recurrent_state_size]

        # Step 0: actor action from initial latent
        key, k_act = jax.random.split(key)
        actions_0, log_probs_0, entropy_0 = actor(init_latent, k_act)
        imagined_actions.append(actions_0)
        imagined_log_probs.append(log_probs_0)
        imagined_entropies.append(entropy_0)

        # Reshape prior to [BT, S, D] for RSSM imagination step
        num_cat = self.rssm.num_categoricals
        num_cls = self.rssm.num_classes
        imagined_prior = imagined_prior_flat.reshape(BT, num_cat, num_cls)

        # Imagination loop (sheeprl L235-L241)
        for i in range(1, horizon + 1):
            key, k_rssm, k_act = jax.random.split(key, 3)

            # RSSM imagination step: prior transition only (no observation)
            # sheeprl L236: imagined_prior, recurrent_state = rssm.imagination(prior, recurrent_state, actions)
            # Our RSSM equivalent: recurrent_mlp + GRU + _transition (no _representation)
            prior_flat = imagined_prior.reshape(BT, -1)   # [BT, S*D]
            action = imagined_actions[-1]                  # [BT, action_dim]

            # RecurrentModel forward (MLP pre-projection + GRU)
            recurrent_input = jnp.concatenate([prior_flat, action], axis=-1)
            recurrent_feat = self.rssm.recurrent_mlp_linear(recurrent_input)
            recurrent_feat = self.rssm.recurrent_mlp_norm(recurrent_feat)
            recurrent_feat = jax.nn.silu(recurrent_feat)
            recurrent_state = self.rssm.gru_cell(recurrent_feat, recurrent_state)

            # Transition model → prior logits + state
            prior_logits_new, imagined_prior = self.rssm._transition(
                recurrent_state, sample_state=True, key=k_rssm
            )

            # Build new latent = cat(prior_flat_new, recurrent_state)
            prior_flat_new = imagined_prior.reshape(BT, -1)
            new_latent = jnp.concatenate([prior_flat_new, recurrent_state], axis=-1)
            imagined_latents.append(new_latent)

            # Actor forward from new latent
            actions_i, log_probs_i, entropy_i = actor(new_latent, k_act)
            imagined_actions.append(actions_i)
            imagined_log_probs.append(log_probs_i)
            imagined_entropies.append(entropy_i)

        imagined_latents_arr = jnp.stack(imagined_latents, axis=0)   # [H+1, BT, latent_dim]
        imagined_actions_arr = jnp.stack(imagined_actions, axis=0)   # [H+1, BT, action_dim]
        imagined_log_probs_arr = jnp.stack(imagined_log_probs, axis=0)  # [H+1, BT, 1]
        imagined_entropies_arr = jnp.stack(imagined_entropies, axis=0)  # [H+1, BT]

        return {
            "imagined_latents":   imagined_latents_arr,
            "imagined_actions":   imagined_actions_arr,
            "imagined_log_probs": imagined_log_probs_arr,
            "imagined_entropies": imagined_entropies_arr,
        }


# ---------------------------------------------------------------------------
# CP9 — FullMLPHead
# Full MLP head with configurable hidden layers + output linear.
# Used by build_agent for reward_model, continue_model, and critic.
# ---------------------------------------------------------------------------

class FullMLPHead(nnx.Module):
    """Full MLP head: mlp_layers hidden layers (Linear+LN+SiLU) + output linear.

    # CP9 helper — not a direct sheeprl port, but faithfully implements the
    # sheeprl MLP(hidden_sizes=[du]*layers, output_dim=bins) structure used
    # in build_agent for reward, continue, and critic models.
    # Sheeprl reference: vendor/sheeprl/sheeprl/models/models.py:L87-L115 (MLP class)

    The output linear is initialized with either:
    - zero_init_output=True  → uniform_init_weights(0.0) — cascade fix #27 (reward, critic)
    - zero_init_output=False → uniform_init_weights(1.0) — Hafner scale=1 (continue head)

    All body linears are initialized with Hafner init_weights (truncated-normal fan-avg).

    Args:
        in_dim (int): input dimension (latent_dim).
        out_dim (int): output dimension (bins or 1).
        dense_units (int): hidden layer width.
        mlp_layers (int): number of hidden layers.
        zero_init_output (bool): if True, zero-init output linear (cascade fix #27).
                                  if False, use uniform_init_weights(1.0) Hafner scale.
        rngs (nnx.Rngs): NNX RNG container.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dense_units: int,
        mlp_layers: int,
        zero_init_output: bool,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dense_units = dense_units
        self.mlp_layers = mlp_layers

        # Hidden layers
        linears = []
        norms = []
        in_size = in_dim
        for _ in range(mlp_layers):
            lin = nnx.Linear(in_size, dense_units, use_bias=False, rngs=rngs)
            norm = nnx.LayerNorm(num_features=dense_units, epsilon=1e-3, rngs=rngs)
            linears.append(lin)
            norms.append(norm)
            in_size = dense_units
        self.hidden_linears = nnx.List(linears)
        self.hidden_norms = nnx.List(norms)

        # Output linear
        self.output_linear = nnx.Linear(in_size, out_dim, use_bias=True, rngs=rngs)

        # Init: Hafner truncated-normal for body linears
        key = rngs.params()
        for lin in self.hidden_linears:
            key, k = jax.random.split(key)
            I, O = lin.kernel[...].shape
            lin.kernel = nnx.Param(init_weights(I, O, k))

        # Output linear init
        key, k = jax.random.split(key)
        I, O = self.output_linear.kernel[...].shape
        if zero_init_output:
            # Cascade fix #27: zero-init the output linear
            # Sheeprl: world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
            # Sheeprl: critic.model[-1].apply(uniform_init_weights(0.0))
            self.output_linear.kernel = nnx.Param(uniform_init_weights(0.0, I, O, k))
            self.output_linear.bias = nnx.Param(jnp.zeros((O,), dtype=jnp.float32))
        else:
            # Hafner scale=1.0 for continue head and decoder heads
            # Sheeprl: world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
            self.output_linear.kernel = nnx.Param(uniform_init_weights(1.0, I, O, k))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: hidden layers → output linear.

        Args:
            x: [..., in_dim]

        Returns:
            out: [..., out_dim]
        """
        for lin, norm in zip(self.hidden_linears, self.hidden_norms):
            x = lin(x)
            x = norm(x)
            x = jax.nn.silu(x)
        return self.output_linear(x)


# ---------------------------------------------------------------------------
# CP9 — build_agent
# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L935-L1237
# (build_agent function, MLP-only single-key variant)
# ---------------------------------------------------------------------------

def build_agent(
    obs_dim: int,
    action_dim: int,
    cfg: dict,
    rngs: nnx.Rngs,
    *,
    observation_breakdown: dict = None,
) -> Tuple["WorldModel", "Actor", CriticHead, CriticHead]:
    """Factory: construct WorldModel, Actor, Critic, target_critic.

    # Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L935-L1237
    # (build_agent function, Fabric-stripped, MLP-only single-key variant)

    Construction order (matches sheeprl L970-L1220):
    1. encoder = MLPEncoder(obs_dim, dense_units, mlp_layers)
    2. rssm = RSSM(...)
    3. decoder = MLPDecoder(latent_dim, obs_dim, dense_units, mlp_layers)
    4. reward_model = RewardHead(latent_dim, bins)
    5. continue_model = ContinueHead(latent_dim, dense_units, mlp_layers)
    6. world_model = WorldModel(encoder, rssm, decoder, reward_model, continue_model)
    7. actor = Actor(latent_dim, action_dim, dense_units, mlp_layers, unimix)
    8. critic = CriticHead(latent_dim, bins)
    9. target_critic = CriticHead(latent_dim, bins) — same arch, separate params

    Zero-init enforcement (cascade fix #27):
        reward_model.output_linear.kernel -> zero (already in RewardHead.__init__)
        critic.output_linear.kernel -> zero (already in CriticHead.__init__)
    Sanity check at startup:
        assert max(|reward_model.output_linear.kernel|) == 0.0
        assert max(|critic.output_linear.kernel|) == 0.0

    Polyak init: target_critic starts as a COPY of critic (tau=1.0 on first call).
        This is handled in the main loop (first polyak_update call with tau=1.0).

    Args:
        obs_dim (int): flat observation vector size.
        action_dim (int): action space size (number of discrete actions).
        cfg (dict): configuration dictionary. Required keys:
            cfg['algo']['world_model']['encoder']['dense_units']
            cfg['algo']['world_model']['encoder']['mlp_layers']
            cfg['algo']['world_model']['recurrent_model']['recurrent_state_size']
            cfg['algo']['world_model']['recurrent_model']['dense_units']
            cfg['algo']['world_model']['transition_model']['hidden_size']
            cfg['algo']['world_model']['representation_model']['hidden_size']
            cfg['algo']['world_model']['reward_model']['bins']
            cfg['algo']['world_model']['reward_model']['dense_units']
            cfg['algo']['world_model']['reward_model']['mlp_layers']
            cfg['algo']['world_model']['continue_model']['dense_units']
            cfg['algo']['world_model']['continue_model']['mlp_layers']
            cfg['algo']['world_model']['stochastic_size']
            cfg['algo']['world_model']['discrete_size']
            cfg['algo']['actor']['dense_units']
            cfg['algo']['actor']['mlp_layers']
            cfg['algo']['critic']['dense_units']
            cfg['algo']['critic']['mlp_layers']
            cfg['algo']['critic']['bins']
            cfg['algo']['unimix']
        rngs (nnx.Rngs): NNX RNG container.
        observation_breakdown (dict, optional): ordered {modality_name: width}
            from src/environment/sensor.py:get_observation_breakdown. REQUIRED
            when cfg['algo']['world_model']['encoding_mode'] == 'hierarchical'
            (D-018); ignored (but validated harmless) under the default 'flat'
            mode, so existing flat callers keep working unchanged.

    D-018 optional keys (read with .get — D-017 precedent; the default 'flat'
    path is bit-identical to the pre-D-018 code and pinned by the parity suite):
            cfg['algo']['world_model']['encoding_mode']: 'flat' (default) | 'hierarchical'
            cfg['algo']['world_model']['hierarchical_params']:   # required iff hierarchical
                mirror_encoder_in_decoder: bool  # REQUIRED there; no default.
                    false → HeadsMLPDecoder (design 2), true → MirrorMLPDecoder (design 3)
                default_mlp: list[int]
                multimodal_hub: list[int]
                hidden_size: int
        A 'flat' config that nevertheless contains hierarchical_params is
        REJECTED with ValueError (a silently inert block is config drift).

    Returns:
        (world_model, actor, critic, target_critic)
    """
    # Extract config values (no fallback defaults — missing key raises KeyError)
    wm_cfg = cfg['algo']['world_model']
    actor_cfg = cfg['algo']['actor']
    critic_cfg = cfg['algo']['critic']

    enc_dense_units = wm_cfg['encoder']['dense_units']
    enc_mlp_layers = wm_cfg['encoder']['mlp_layers']

    recurrent_state_size = wm_cfg['recurrent_model']['recurrent_state_size']
    recurrent_dense_units = wm_cfg['recurrent_model']['dense_units']
    transition_hidden_size = wm_cfg['transition_model']['hidden_size']
    repr_hidden_size = wm_cfg['representation_model']['hidden_size']

    stochastic_size = wm_cfg['stochastic_size']   # num_categoricals (S)
    discrete_size = wm_cfg['discrete_size']        # num_classes per categorical (D)
    stoch_flat_size = stochastic_size * discrete_size  # S*D
    latent_dim = stoch_flat_size + recurrent_state_size

    reward_bins = wm_cfg['reward_model']['bins']
    reward_dense_units = wm_cfg['reward_model']['dense_units']
    reward_mlp_layers = wm_cfg['reward_model']['mlp_layers']

    continue_dense_units = wm_cfg['continue_model']['dense_units']
    continue_mlp_layers = wm_cfg['continue_model']['mlp_layers']

    actor_dense_units = actor_cfg['dense_units']
    actor_mlp_layers = actor_cfg['mlp_layers']

    critic_dense_units = critic_cfg['dense_units']
    critic_mlp_layers = critic_cfg['mlp_layers']
    critic_bins = critic_cfg['bins']

    unimix = cfg['algo']['unimix']

    # -----------------------------------------------------------------------
    # D-018 — encoding-mode resolution + fail-loud validation.
    # ALL mode selection and validation happens BEFORE any constructor call,
    # so the default flat path consumes the rngs stream identically to the
    # pre-D-018 code (init bits cannot move; parity suite pins this).
    # -----------------------------------------------------------------------
    encoding_mode = wm_cfg.get('encoding_mode', 'flat')  # D-017 .get pattern: default = parity
    if encoding_mode not in ('flat', 'hierarchical'):
        raise ValueError(
            f"algo.world_model.encoding_mode must be 'flat' or 'hierarchical', "
            f"got {encoding_mode!r}"
        )
    if encoding_mode == 'flat' and 'hierarchical_params' in wm_cfg:
        # Rejected combination — a silently inert hierarchical_params block
        # (incl. one that only sets mirror_encoder_in_decoder) is exactly the
        # config drift the project's no-fallback rule exists to prevent.
        raise ValueError(
            "algo.world_model.hierarchical_params is present but encoding_mode "
            "is 'flat' (explicit or defaulted). Either set "
            "encoding_mode: hierarchical or delete the hierarchical_params block."
        )
    if encoding_mode == 'hierarchical':
        # Sub-keys hard-indexed — missing key raises KeyError (no fallback).
        h_params = wm_cfg['hierarchical_params']
        hier_mirror_decoder = h_params['mirror_encoder_in_decoder']  # REQUIRED boolean
        if not isinstance(hier_mirror_decoder, bool):
            raise ValueError(
                "algo.world_model.hierarchical_params.mirror_encoder_in_decoder "
                f"must be a boolean, got {hier_mirror_decoder!r} "
                f"({type(hier_mirror_decoder).__name__})"
            )
        hier_default_mlp = h_params['default_mlp']
        hier_multimodal_hub = h_params['multimodal_hub']
        hier_hidden_size = h_params['hidden_size']
        if observation_breakdown is None:
            raise ValueError(
                "encoding_mode: hierarchical requires observation_breakdown. "
                "Callers must pass observation_breakdown="
                "get_observation_breakdown(env_params) "
                "(src/environment/sensor.py) to build_agent — see "
                "dreamer_srl_main.py for the reference wiring."
            )
        _bd_sum = sum(observation_breakdown.values())
        if _bd_sum != obs_dim:
            raise ValueError(
                f"observation_breakdown sums to {_bd_sum} but obs_dim is "
                f"{obs_dim} — the breakdown does not describe this observation "
                f"vector (breakdown: {dict(observation_breakdown)})"
            )

    if encoding_mode == 'flat':
        # 1. Encoder
        encoder = MLPEncoder(
            obs_dim=obs_dim,
            dense_units=enc_dense_units,
            mlp_layers=enc_mlp_layers,
            rngs=rngs,
        )

        # 2. RSSM
        rssm = RSSM(
            recurrent_state_size=recurrent_state_size,
            recurrent_dense_units=recurrent_dense_units,
            action_dim=action_dim,
            stochastic_size=stoch_flat_size,
            transition_hidden_size=transition_hidden_size,
            repr_hidden_size=repr_hidden_size,
            num_categoricals=stochastic_size,
            num_classes=discrete_size,
            encoder_output_dim=enc_dense_units,
            unimix=unimix,
            rngs=rngs,
        )

        # 3. Decoder
        decoder = MLPDecoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            dense_units=enc_dense_units,    # matches encoder dense_units (symmetric)
            mlp_layers=enc_mlp_layers,
            rngs=rngs,
        )
    else:
        # D-018 hierarchical mode (designs 2/3). Same construction ORDER as
        # flat (encoder → RSSM → decoder) on the same rngs stream.
        # 1. Encoder — per-modality branches + fusion hub
        encoder = HierarchicalMLPEncoder(
            observation_breakdown=observation_breakdown,
            default_mlp=hier_default_mlp,
            multimodal_hub=hier_multimodal_hub,
            hidden_size=hier_hidden_size,
            rngs=rngs,
        )

        # 2. RSSM — consumes encoder.output_dim (= hierarchical hidden_size),
        # NOT enc_dense_units (plan-review C3: the two coincide at XS only).
        rssm = RSSM(
            recurrent_state_size=recurrent_state_size,
            recurrent_dense_units=recurrent_dense_units,
            action_dim=action_dim,
            stochastic_size=stoch_flat_size,
            transition_hidden_size=transition_hidden_size,
            repr_hidden_size=repr_hidden_size,
            num_categoricals=stochastic_size,
            num_classes=discrete_size,
            encoder_output_dim=encoder.output_dim,
            unimix=unimix,
            rngs=rngs,
        )

        # 3. Decoder — heads (design 2) or mirror (design 3), selected by the
        # required boolean hierarchical_params.mirror_encoder_in_decoder.
        if hier_mirror_decoder:
            decoder = MirrorMLPDecoder(
                latent_dim=latent_dim,
                observation_breakdown=observation_breakdown,
                default_mlp=hier_default_mlp,
                multimodal_hub=hier_multimodal_hub,
                hidden_size=hier_hidden_size,
                rngs=rngs,
            )
        else:
            # N1: trunk reads the SAME sizing source as the flat decoder
            # (encoder dense_units / mlp_layers) so designs 1 vs 2 differ
            # only in the output head.
            decoder = HeadsMLPDecoder(
                latent_dim=latent_dim,
                observation_breakdown=observation_breakdown,
                dense_units=enc_dense_units,
                mlp_layers=enc_mlp_layers,
                rngs=rngs,
            )

    # 4. RewardMLP — full MLP with zero-init output linear (cascade fix #27)
    # Sheeprl L1100-L1112: reward_model = MLP(input_dims=latent, output_dim=bins, hidden_sizes=[du]*layers, ...)
    # then L1175: world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
    # We use FullMLPHead which has hidden layers + zero-init output linear.
    reward_model = FullMLPHead(
        in_dim=latent_dim,
        out_dim=reward_bins,
        dense_units=reward_dense_units,
        mlp_layers=reward_mlp_layers,
        zero_init_output=True,   # cascade fix #27
        rngs=rngs,
    )

    # 5. ContinueHead (full MLP) — Hafner init (scale=1.0) on output linear (sheeprl L1176)
    continue_model = FullMLPHead(
        in_dim=latent_dim,
        out_dim=1,
        dense_units=continue_dense_units,
        mlp_layers=continue_mlp_layers,
        zero_init_output=False,  # Hafner scale=1.0 (not zero) — sheeprl L1176
        rngs=rngs,
    )

    # 6. WorldModel
    world_model = WorldModel(
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_model=reward_model,
        continue_model=continue_model,
    )

    # 7. Actor
    actor = Actor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        dense_units=actor_dense_units,
        mlp_layers=actor_mlp_layers,
        unimix=unimix,
        rngs=rngs,
    )

    # 8. Critic (full MLP) — zero-init output (cascade fix #27)
    # Sheeprl L1154-L1166: critic = MLP(...)  then L1172: critic.model[-1].apply(uniform_init_weights(0.0))
    critic = FullMLPHead(
        in_dim=latent_dim,
        out_dim=critic_bins,
        dense_units=critic_dense_units,
        mlp_layers=critic_mlp_layers,
        zero_init_output=True,   # cascade fix #27
        rngs=rngs,
    )

    # 9. Target critic — same arch, separate params (zero-init on fresh init, then tau=1.0 polyak copies)
    target_critic = FullMLPHead(
        in_dim=latent_dim,
        out_dim=critic_bins,
        dense_units=critic_dense_units,
        mlp_layers=critic_mlp_layers,
        zero_init_output=True,
        rngs=rngs,
    )

    # Sanity check: zero-init enforcement (cascade fix #27)
    # These assertions catch any future regression where the output linear is not zero-initialized.
    reward_kernel_max = jnp.abs(reward_model.output_linear.kernel[...]).max()
    critic_kernel_max = jnp.abs(critic.output_linear.kernel[...]).max()
    if float(reward_kernel_max) != 0.0:
        raise RuntimeError(
            f"CP3 zero-init regression: reward_model.output_linear.kernel max = {float(reward_kernel_max):.3e} != 0.0"
        )
    if float(critic_kernel_max) != 0.0:
        raise RuntimeError(
            f"CP3 zero-init regression: critic.output_linear.kernel max = {float(critic_kernel_max):.3e} != 0.0"
        )

    # D-018: per-module param-count print (run-manifest record; plan File
    # Change 1). Pure logging — consumes no rngs, touches no params.
    def _n_params(module) -> int:
        return sum(int(leaf.size) for leaf in
                   jax.tree_util.tree_leaves(nnx.state(module, nnx.Param)))

    _counts = {
        'encoder':        _n_params(world_model.encoder),
        'rssm':           _n_params(world_model.rssm),
        'decoder':        _n_params(world_model.decoder),
        'reward_model':   _n_params(world_model.reward_model),
        'continue_model': _n_params(world_model.continue_model),
        'actor':          _n_params(actor),
        'critic':         _n_params(critic),
        'target_critic':  _n_params(target_critic),
    }
    print(f"[build_agent] encoding_mode={encoding_mode} | params: "
          + ", ".join(f"{k}={v:,}" for k, v in _counts.items())
          + f" | total={sum(_counts.values()):,}")

    return world_model, actor, critic, target_critic
