import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

def symlog(x):
    """
    Symmetric logarithmic function: sign(x) * log(|x| + 1).
    Used to compress the scale of inputs and targets.
    """
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)

def symexp(x):
    """
    Cyclic inverse of symlog: sign(x) * (exp(|x|) - 1).
    """
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)

def to_twohot(x, min_v=-20.0, max_v=20.0, num_buckets=255, paper_canonical_bins=False):
    """
    Converts a scalar to a Two-Hot distribution (soft discretization).
    Used for Value and Reward targets in DreamerV3.

    Args:
        x: scalar(s), in raw space.
        min_v, max_v: bin-grid edges in SYMLOG space when paper_canonical_bins=True
            (paper-canonical convention: ±20 are already symlog-space edges; raw-space
            bin centres span ±symexp(20) ≈ ±4.85·10⁸). Matches Hafner
            embodied/jax/heads.py:87–97 and sheeprl TwoHotEncodingDistribution.
        num_buckets: 255 (paper).
        paper_canonical_bins: if True, use ±20 as symlog-space edges directly
            (paper-canonical; raw-space support ±4.85e8). If False (legacy default),
            apply symlog(±20) to the edges (legacy buggy behaviour; raw-space support
            narrows to ±20). The trainer always passes this flag explicitly via
            agent.paper_canonical_twohot_bins; the False default keeps import-time
            tests bit-identical.
    """
    x = symlog(x)

    if paper_canonical_bins:
        # Paper-canonical: min_v, max_v are already symlog-space edges.
        bottom = jnp.array(min_v, dtype=x.dtype)
        top    = jnp.array(max_v, dtype=x.dtype)
    else:
        # Legacy behaviour: treat min_v, max_v as raw-space edges and apply symlog.
        # Reproduces pre-fix bin layout (raw-space support ±20).
        bottom = symlog(jnp.array(min_v, dtype=x.dtype))
        top    = symlog(jnp.array(max_v, dtype=x.dtype))

    # Clip value to range (in symlog space)
    x = jnp.clip(x, bottom, top)

    # Map to [0, num_buckets - 1]
    rel = (x - bottom) / (top - bottom) * (num_buckets - 1)

    floor = jnp.floor(rel).astype(jnp.int32)
    ceil = jnp.ceil(rel).astype(jnp.int32)

    prob_ceil = rel - floor
    prob_floor = 1.0 - prob_ceil

    # One-hot encoding
    # We return the target PROBABILITIES directly
    # Shape: (*x.shape, num_buckets)

    def scatter(idx, val):
        # idx shape: (...)
        # val shape: (...)
        # output shape: (..., num_buckets)
        return jax.nn.one_hot(idx, num_buckets) * val[..., None]

    target = scatter(floor, prob_floor) + scatter(ceil, prob_ceil)
    return target

def from_twohot(logits, min_v=-20.0, max_v=20.0, num_buckets=255, paper_canonical_bins=False):
    """
    Converts logits from Two-Hot distribution back to scalar (expectation).
    Returns value in RAW space (inverse symlog).

    Args mirror `to_twohot` — see that docstring for the `paper_canonical_bins`
    flag semantics. Bin layout in this function MUST match the layout used in
    `to_twohot` (same flag value).
    """
    probs = jax.nn.softmax(logits, axis=-1)

    if paper_canonical_bins:
        # Paper-canonical: min_v, max_v are already symlog-space edges.
        bottom = jnp.array(min_v, dtype=probs.dtype)
        top    = jnp.array(max_v, dtype=probs.dtype)
    else:
        # Legacy behaviour: apply symlog to min_v, max_v.
        bottom = symlog(jnp.array(min_v, dtype=probs.dtype))
        top    = symlog(jnp.array(max_v, dtype=probs.dtype))

    # Bucket centres in symlog space.
    bucket_vals = jnp.linspace(bottom, top, num_buckets)

    # Expected value in symlog space, then mapped back to raw via symexp.
    sym_val = jnp.sum(probs * bucket_vals, axis=-1)

    return symexp(sym_val)

class OneHotDist:
    """
    One-Hot Categorical Distribution with Straight-Through Estimator and Unimix.
    Used for Stochastic State (z) in RSSM.
    """
    def __init__(self, logits, unimix=0.01):
        self.logits = logits
        self.num_classes = logits.shape[-1]
        probs = jax.nn.softmax(logits, axis=-1)
        if unimix > 0:
            probs = (1.0 - unimix) * probs + unimix / self.num_classes
        self.probs = probs
        
    def sample(self, key):
        # Sample from probabilities (equivalent to categorical sampling)
        # If key is (B, -1), we check if the probs also have a leading batch dimension B.
        # This handles both latent z (B, stoch, discrete) and actions (B, action_dim).
        if key.ndim == 2 and self.probs.ndim >= 2:
            sample_idx = jax.vmap(lambda p, k: random.categorical(k, jnp.log(p + 1e-10)))(self.probs, key)
        else:
            sample_idx = random.categorical(key, jnp.log(self.probs + 1e-10))
            
        sample_onehot = jax.nn.one_hot(sample_idx, self.num_classes)
        
        # Straight-Through Estimator
        # Forward: One-hot sample
        # Backward: Gradients flow through softmax probs
        sample_st = sample_onehot - jax.lax.stop_gradient(self.probs) + self.probs
        return sample_st
    
    def mode(self):
        sample_idx = jnp.argmax(self.logits, axis=-1)
        return jax.nn.one_hot(sample_idx, self.num_classes)


class Moments(nnx.Module):
    """
    Exponential Moving Average (EMA) of percentile-based moments for return normalization.
    Matches PyTorch DreamerV3 implementation.
    
    Now an NNX Module for correct state management in JIT.
    """
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1.0,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ):
        self.decay = decay
        self.max_ = max_
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        # State: EMA of low and high percentiles (init as jnp arrays for NNX)
        self.low = nnx.Variable(jnp.array(0.0))
        self.high = nnx.Variable(jnp.array(1.0))
        
    def update(self, x):
        """
        Update moments with new data and return normalization parameters.
        """
        x_flat = jnp.ravel(x).astype(jnp.float32)
        
        # Compute percentiles
        low_p = jnp.percentile(x_flat, self.percentile_low * 100)
        high_p = jnp.percentile(x_flat, self.percentile_high * 100)
        
        # EMA update (NNX handles attribute assignment in jit)
        self.low.value = self.decay * self.low.value + (1.0 - self.decay) * low_p
        self.high.value = self.decay * self.high.value + (1.0 - self.decay) * high_p
        
        # Compute inverse scale (clamped as in PyTorch)
        invscale = jnp.maximum(1.0 / self.max_, self.high.value - self.low.value)
        return self.low.value, invscale
    
    def normalize(self, x):
        """
        Normalize values using current moments.
        """
        invscale = jnp.maximum(1.0 / self.max_, self.high.value - self.low.value)
        return (x - self.low.value) / invscale


class Ratio:
    """
    Computes the number of gradient steps to perform based on a target ratio 
    of gradient steps per environment step.
    
    Directly taken from Hafner et al. (2023) implementation.
    """
    def __init__(self, ratio: float, pretrain_steps: int = 0):
        if pretrain_steps < 0:
            raise ValueError(f"'pretrain_steps' must be non-negative, got {pretrain_steps}")
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._pretrain_steps = pretrain_steps
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            repeats = int(step * self._ratio)
            if self._pretrain_steps > 0:
                # If pretrain_steps is provided, we use it for the first call
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        
        # Subsequent calls compute steps since last call
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


def hafner_init(scale=0.8796):
    """
    Hafner initialization: Truncated normal with stddev = scale / sqrt(fan_in).
    Matches the 'secret sauce' of the original DreamerV3 implementation.
    """
    def init(key, shape, dtype=jnp.float32):
        fan_in = shape[0] if len(shape) > 0 else 1
        stddev = scale / jnp.sqrt(fan_in)
        # Using 2.0 * stddev as truncation bound is standard for truncated_normal
        return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype=dtype) * stddev
    return init
