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

def to_twohot(x, min_v=-20.0, max_v=20.0, num_buckets=255):
    """
    Converts a scalar to a Two-Hot distribution (soft discretization).
    Used for Value and Reward targets in DreamerV3.
    """
    x = symlog(x)
    # Using raw values for boundaries as per common implementations, but mapped to symlog space?
    # Actually DreamerV3 paper uses symlog(x) for targets.
    # We define buckets in the TRANSFORMED space usually, or raw?
    # The official implementation defines buckets in SYMLOG space.
    # range: symlog(-20) ~ -3 to symlog(20) ~ 3.
    
    # Let's interpret min_v and max_v as RAW values, and we transform them.
    bottom = symlog(jnp.array(min_v))
    top = symlog(jnp.array(max_v))
    
    # Clip value to range
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

def from_twohot(logits, min_v=-20.0, max_v=20.0, num_buckets=255):
    """
    Converts logits from Two-Hot distribution back to scalar (expectation).
    Returns value in RAW space (inverse symlog).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    
    bottom = symlog(jnp.array(min_v))
    top = symlog(jnp.array(max_v))
    
    # Bucket values in symlog space
    bucket_vals = jnp.linspace(bottom, top, num_buckets)
    
    # Expected value in symlog space
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

