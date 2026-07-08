"""dreamer-srl utility functions — leaf utilities ported from sheeprl@33b6366.

All functions in this module carry source-citation docstring headers (Lever B)
naming the vendored sheeprl file and exact line range. Every function is
paired with a bit-identity test in tests/algorithms/dreamer_srl/test_utils.py
that asserts max-absolute-difference < 1e-6 against the sheeprl PyTorch side.

Isolation rule (v2 Risks §13): this module does NOT import from
src.models.dreamer_v3_nnx, src.models.dreamer_v3_trainer, or any other file
in src.models. The only allowed shared import is src.utils.config.Config.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------------------
# WP-SRL P1: gradient-clip recipe constants + optimizer factory
# ---------------------------------------------------------------------------
# Pinned to sheeprl@33b6366:sheeprl/configs/algo/dreamer_v3.yaml
#   :52  algo.world_model.clip_gradients: 1000.0
#   :127 algo.actor.clip_gradients:       100.0
#   :154 algo.critic.clip_gradients:      100.0
# Applied in sheeprl at dreamer_v3.py:193-197 (WM), :300-302 (actor), :320-324
# (critic) via fabric.clip_gradients(..., error_if_nonfinite=False).
#
# These are hardcoded reference-recipe constants, NOT config reads with
# defaults (no-fallback-defaults rule does not apply) — exactly as fixed as
# the two-hot bin count. If a future experiment needs them tunable, that is a
# separate plan (config keys via get_mandatory + YAML additions).
#
# Substrate note: torch's clip_grad_norm_ uses
# clip_coef = max_norm / (total_norm + 1e-6); optax.clip_by_global_norm omits
# the 1e-6. Relative effect < 1e-6 at the clip boundary — substrate-class
# (same family as D-003/D-007), no DEVIATION_LOG row needed.
#
# Regression test: tests/algorithms/dreamer_srl/test_grad_clip.py
WM_CLIP_NORM: float = 1000.0
ACTOR_CLIP_NORM: float = 100.0
CRITIC_CLIP_NORM: float = 100.0


def make_optim_tx(lr: float, eps: float, clip_norm: float) -> optax.GradientTransformation:
    """clip-by-global-norm → Adam, matching sheeprl's clip-then-step order."""
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(lr, eps=eps),
    )


# ---------------------------------------------------------------------------
# WP-SRL P6: learning_starts is an ENV-STEP count in the config
# ---------------------------------------------------------------------------

def derive_prefill(learning_starts_cfg: int, num_envs: int) -> Tuple[int, int]:
    """Return (learning_starts_iters, prefill_steps) from the config env-step value.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L508-L511
    (world_size == 1):
        policy_steps_per_iter = num_envs
        learning_starts = cfg.algo.learning_starts // policy_steps_per_iter
        prefill_steps   = learning_starts - int(learning_starts > 0)

    The config value is an ENV-STEP count (sheeprl semantics); the pre-fix
    driver consumed it as an ITERATION count, making the prefill phase
    num_envs x too long ([[00_master_comparison]] §3 P6; DEVIATION_LOG D-014
    is historical as of this fix). The off-by-one in prefill_steps is
    intentional (sheeprl L511). derive_prefill(0, n) == (0, 0) keeps the
    D-012 zero-prefill smoke path unchanged.

    Regression test: tests/algorithms/dreamer_srl/test_prefill.py::test_derive_prefill
    """
    learning_starts = learning_starts_cfg // num_envs
    prefill_steps = learning_starts - int(learning_starts > 0)
    return learning_starts, prefill_steps


# ---------------------------------------------------------------------------
# symlog / symexp
# ---------------------------------------------------------------------------

def symlog(x: jax.Array) -> jax.Array:
    """Element-wise symlog transform: sign(x) * log(|x| + 1).

    Ported from sheeprl@33b6366:sheeprl/utils/utils.py:L148-L149
    (symlog function).

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_symlog_matches_sheeprl
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Element-wise symexp transform (inverse of symlog): sign(x) * (exp(|x|) - 1).

    Ported from sheeprl@33b6366:sheeprl/utils/utils.py:L152-L153
    (symexp function).

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_symexp_matches_sheeprl
    """
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


# ---------------------------------------------------------------------------
# Weight initializers
# ---------------------------------------------------------------------------

def init_weights(
    in_features: int,
    out_features: int,
    key: jax.Array,
) -> jax.Array:
    """Hafner truncated-normal weight initializer for Linear layers.

    Returns a kernel array of shape (in_features, out_features) using the
    fan-average scale and the precise Hafner constant 0.87962566103423978.
    Bias is not returned — callers initialize bias to zero separately.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/utils.py:L143-L166
    (init_weights function, Linear branch only — Conv/LayerNorm branches are
    out of scope for dreamer-srl which uses only Linear layers).

    GOTCHA: PyTorch's nn.init.trunc_normal_ truncates at [-2*std, +2*std] for
    Linear layers (sheeprl L150: a=-2.0*std, b=2.0*std). We match this exactly
    via jax.random.truncated_normal with lower=-2, upper=+2, then scale by std.

    GOTCHA: The Hafner constant must be the FULL-PRECISION value
    0.87962566103423978. Do NOT use the truncated 0.8796 from the existing
    src/models/dreamer_v3_util.py — that truncation is a known bug (v2
    Risks §5). Bit-identity test will catch any mismatch.

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_init_weights_matches_sheeprl
    """
    denoms = (in_features + out_features) / 2.0
    scale = 1.0 / denoms
    std = float(np.sqrt(scale) / 0.87962566103423978)
    # truncated_normal samples from a standard normal truncated to [lower, upper],
    # then we multiply by std to match nn.init.trunc_normal_(mean=0, std=std, a=-2*std, b=2*std)
    kernel = (
        jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=(in_features, out_features))
        * std
    )
    return kernel


def uniform_init_weights(
    given_scale: float,
    in_features: int,
    out_features: int,
    key: jax.Array,
) -> jax.Array:
    """Uniform weight initializer for output head Linear layers.

    Returns a kernel array of shape (in_features, out_features).
    When given_scale=0.0 this produces a zero-initialized kernel (limit=0,
    uniform(-0,0)=0) — this is cascade fix #27 for the reward head and critic
    output linear (see v2 cascade items). Bias is not returned; callers
    initialize to zero separately.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/utils.py:L170-L186
    (uniform_init_weights function, Linear branch only).

    GOTCHA: given_scale=0.0 encodes zero-init — there is no separate zero-init
    code path. The function gracefully handles scale=0 (limit=0 → all zeros).

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_uniform_init_weights_matches_sheeprl
    """
    denoms = (in_features + out_features) / 2.0
    scale = given_scale / denoms
    limit = float(np.sqrt(3 * scale))
    kernel = jax.random.uniform(key, shape=(in_features, out_features), minval=-limit, maxval=limit)
    return kernel


# ---------------------------------------------------------------------------
# Lambda return
# ---------------------------------------------------------------------------

def compute_lambda_values(
    rewards: jax.Array,
    values: jax.Array,
    continues: jax.Array,
    lmbda: float = 0.95,
) -> jax.Array:
    """Compute lambda-return targets via reverse recursion.

    Arguments match sheeprl exactly:
        rewards   : [T, B]   — reward at each step
        values    : [T+1, B] — bootstrap values (values[-1] is the terminal bootstrap)
        continues : [T, B]   — discount mask (0 at terminal, 1 otherwise; NOT γ itself)
        lmbda     : float    — lambda mixing coefficient (default 0.95)

    Returns a [T, B] tensor of lambda-return targets.

    The recursion (sheeprl L73-L76):
        interm[t] = rewards[t] + continues[t] * values[t] * (1 - lmbda)
        V_lambda[T] = values[T]          (terminal bootstrap)
        V_lambda[t] = interm[t] + continues[t] * lmbda * V_lambda[t+1]

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/utils.py:L66-L77
    (compute_lambda_values function).

    GOTCHA: sheeprl's Python for-loop builds the result as a list that gets
    reversed and concatenated — the loop iterates in REVERSE (t=T-1 down to 0).
    We replicate this exactly with a Python loop (no lax.scan) because the
    input length T is small (horizon=16 typically) and the Python overhead is
    negligible. Using lax.scan would require tracing T steps and produces
    identical numerics; the Python loop is cleaner for readability and matches
    sheeprl's structure more directly.

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_compute_lambda_values_matches_sheeprl
    """
    # Sheeprl L72-L76: vals starts with the terminal bootstrap value[-1:],
    # interm uses the FULL values array (same length as rewards/continues),
    # the loop appends in REVERSE order, then we reverse+concat and drop the last.
    vals = [values[-1:]]  # [1, B] bootstrap
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = jnp.concatenate(list(reversed(vals))[:-1], axis=0)
    return ret


# ---------------------------------------------------------------------------
# Moments — pure-functional pytree
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class MomentsState:
    """Pure-functional state for the Moments percentile-EMA normalizer.

    Both fields are scalar float32 JAX arrays. Using flax.struct.dataclass
    per the v3 implementation plan (IMPLEMENTATION_PLAN.md line 328) — both
    flax.struct.dataclass and NamedTuple are JAX pytrees, so the numerical
    behavior is identical.

    DEVIATION NOTE: Sheeprl's Moments is a PyTorch nn.Module that mutates
    self.low / self.high in-place via register_buffer (L53-L54, L60-L61).
    We cannot use in-place mutation inside JAX-traced functions. Instead, the
    state is explicit: callers pass in the old MomentsState and receive a new
    MomentsState plus the (offset, invscale) pair. This is logged in
    DEVIATION_LOG.md as D-001.
    """
    low: jax.Array   # scalar float32, EMA of the 5th-percentile
    high: jax.Array  # scalar float32, EMA of the 95th-percentile


def moments_init(
    decay: float = 0.99,
    max_: float = 1e8,
    percentile_low: float = 0.05,
    percentile_high: float = 0.95,
) -> "MomentsState":
    """Create the initial MomentsState with low=0.0, high=0.0.

    Mirrors sheeprl Moments.__init__ (L41-L54): buffers are zero-initialized.
    The hyperparameters (decay, max_, percentile_low, percentile_high) are NOT
    stored in the state — they are passed as arguments to moments_update so
    they stay in Python (not traced into JAX arrays).

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/utils.py:L40-L54
    (Moments.__init__).

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_moments_update_matches_sheeprl
    """
    return MomentsState(
        low=jnp.zeros((), dtype=jnp.float32),
        high=jnp.zeros((), dtype=jnp.float32),
    )


def moments_update(
    state: MomentsState,
    x: jax.Array,
    decay: float = 0.99,
    max_: float = 1e8,
    percentile_low: float = 0.05,
    percentile_high: float = 0.95,
) -> tuple[MomentsState, jax.Array, jax.Array]:
    """Update the EMA percentile statistics and return (new_state, new_low, invscale).

    This is the JAX port of sheeprl's Moments.forward (L56-L63). Single-process
    version: no fabric.all_gather (we run single-process; see DEVIATION_LOG D-001).

    The numerics MATCH sheeprl exactly for single-process (fabric.all_gather on
    a single process returns x unchanged). The only structural difference is that
    state is explicit and returned, rather than mutated in-place on self.

    Args:
        state          : current MomentsState (low, high)
        x              : [T, B] or any float32 array of lambda-return values
        decay          : EMA decay coefficient (default 0.99, sheeprl L43)
        max_           : floor for invscale denominator (default 1e8, sheeprl L44)
        percentile_low : lower percentile (default 0.05, sheeprl L45)
        percentile_high: upper percentile (default 0.95, sheeprl L46)

    Returns:
        new_state : updated MomentsState
        new_low   : low EMA (detached scalar) — used as advantage baseline
                    (sheeprl names this ``offset`` at the call site)
        invscale  : max(1/max_, high - low) — advantage scale

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/utils.py:L40-L63
    (Moments.forward).

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_moments_update_matches_sheeprl
    """
    x_flat = x.astype(jnp.float32).ravel()
    # Sheeprl L58-L59: quantile on gathered tensor
    low_new = jnp.quantile(x_flat, percentile_low)
    high_new = jnp.quantile(x_flat, percentile_high)
    # Sheeprl L60-L61: EMA update
    new_low = decay * state.low + (1 - decay) * low_new
    new_high = decay * state.high + (1 - decay) * high_new
    new_state = MomentsState(low=new_low, high=new_high)
    # Sheeprl L62-L63: invscale and return
    invscale = jnp.maximum(1.0 / max_, new_high - new_low)
    return new_state, new_low, invscale


# ---------------------------------------------------------------------------
# Ratio — replay-ratio scheduler (Python-side, never crosses JIT)
# ---------------------------------------------------------------------------

class Ratio:
    """Replay-ratio scheduler with fractional-debt accumulator.

    Directly ported from sheeprl@33b6366:sheeprl/utils/utils.py:L259-L301
    (Ratio class), which itself is taken from Hafner et al. (2023):
    https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26

    Semantics: given the current cumulative env-step count, returns the integer
    number of gradient steps owed. Accumulates fractional debt to avoid drift
    over long runs. The __call__ method must be called with MONOTONICALLY
    INCREASING step values (no repeated or out-of-order calls).

    GOTCHA: Ratio is a Python-side object. NEVER pass it inside a jax.jit or
    @nnx.jit boundary — the _prev accumulator would be a stale captured value
    and replay-ratio would silently be wrong. The training loop calls
    ratio(env_step) in plain Python, then executes `one_train_step` that many
    times. See v2 plan Risks §12 and §2.

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_ratio_matches_sheeprl
    """

    def __init__(self, ratio: float, pretrain_steps: int = 0) -> None:
        """Initialize.

        Ported from sheeprl@33b6366:sheeprl/utils/utils.py:L264-L271.

        Args:
            ratio          : desired gradient-steps-per-env-step ratio (e.g. 0.5)
            pretrain_steps : number of env steps to burn down at a potentially
                             higher ratio before the steady-state ratio kicks in
                             (default 0 — no pretrain burst, which is correct for
                             the XS config: sheeprl's per_rank_pretrain_steps=0)
        """
        if pretrain_steps < 0:
            raise ValueError(f"'pretrain_steps' must be non-negative, got {pretrain_steps}")
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._pretrain_steps = pretrain_steps
        self._ratio = ratio
        self._prev: Optional[float] = None

    def __call__(self, step: int) -> int:
        """Return the number of gradient steps owed at this env-step count.

        Ported from sheeprl@33b6366:sheeprl/utils/utils.py:L273-L291.
        """
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            repeats = int(step * self._ratio)
            if self._pretrain_steps > 0:
                if step < self._pretrain_steps:
                    warnings.warn(
                        "The number of pretrain steps is greater than the number of current steps. "
                        f"This could lead to a higher ratio than the one specified ({self._ratio}). "
                        "Setting the 'pretrain_steps' equal to the number of current steps."
                    )
                    self._pretrain_steps = step
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state dict.

        Ported from sheeprl@33b6366:sheeprl/utils/utils.py:L293-L294.
        """
        return {
            "_ratio": self._ratio,
            "_prev": self._prev,
            "_pretrain_steps": self._pretrain_steps,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> "Ratio":
        """Restore state from a state dict.

        Ported from sheeprl@33b6366:sheeprl/utils/utils.py:L296-L300.
        """
        self._ratio = state_dict["_ratio"]
        self._prev = state_dict["_prev"]
        self._pretrain_steps = state_dict["_pretrain_steps"]
        return self


# ---------------------------------------------------------------------------
# prepare_obs — obs-dict-to-array helper
# ---------------------------------------------------------------------------

def prepare_obs(
    obs: Dict[str, np.ndarray],
    *,
    cnn_keys: Sequence[str] = (),
    num_envs: int = 1,
) -> Dict[str, jax.Array]:
    """Convert a numpy observation dict to JAX arrays with the RSSM's [T=1, B, ...] shape.

    Ports sheeprl's prepare_obs (L80-L91) without the Lightning Fabric device
    dispatch — JAX arrays are always on the accelerator already.

    For each key in obs:
      - MLP keys  : reshape to [1, num_envs, -1] (flattened)
      - CNN keys  : reshape to [1, num_envs, C, H, W] and rescale to [-0.5, 0.5]
                    (uint8 divide by 255 then subtract 0.5)

    GOTCHA: The leading T=1 axis is MANDATORY. The RSSM encoder consumes
    [T, B, ...] and will silently broadcast if the axis is omitted, producing
    wrong-shape embeddings. Do not drop this axis even when calling for a
    single observation.

    GOTCHA: Our environment is vector-obs only. The CNN branch is ported for
    completeness but will not be exercised in the XS config (cnn_keys is empty).
    Do not add a guard that errors on non-empty cnn_keys — future configs may use
    pixel obs.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/utils.py:L80-L91
    (prepare_obs function, minus Fabric device dispatch).

    Bit-identity test: tests/algorithms/dreamer_srl/test_utils.py::test_prepare_obs_shape_contract
    """
    jax_obs: Dict[str, jax.Array] = {}
    for k, v in obs.items():
        arr = jnp.asarray(v, dtype=jnp.float32)
        if k in cnn_keys:
            # [H, W, C] or [B, H, W, C] → [1, num_envs, C, H, W] then rescale
            jax_obs[k] = arr.reshape(1, num_envs, -1, *v.shape[-2:]) / 255.0 - 0.5
        else:
            jax_obs[k] = arr.reshape(1, num_envs, -1)
    return jax_obs
