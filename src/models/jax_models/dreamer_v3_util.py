import jax
import jax.numpy as jnp
from jax import random

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
    One-Hot Categorical Distribution with Straight-Through Estimator.
    Used for Stochastic State (z) in RSSM.
    """
    def __init__(self, logits):
        self.logits = logits
        self.probs = jax.nn.softmax(logits, axis=-1)
        self.num_classes = logits.shape[-1]
        
    def sample(self, key):
        # Gumbel-Max trick
        u = random.uniform(key, self.logits.shape)
        gumbel = -jnp.log(-jnp.log(u + 1e-10) + 1e-10)
        # Add temperature if needed, but standard is argmax
        sample_idx = jnp.argmax(self.logits + gumbel, axis=-1)
        sample_onehot = jax.nn.one_hot(sample_idx, self.num_classes)
        
        # Straight-Through Estimator
        # Forward: One-hot sample
        # Backward: Gradients flow through softmax probs
        sample_st = sample_onehot + self.probs - self.probs # No stop_gradient needed here explicitly in JAX if we do it right?
        # Typically: y = y_hard - y_soft.stop_gradient() + y_soft
        sample_st = sample_onehot - jax.lax.stop_gradient(self.probs) + self.probs
        return sample_st
    
    def mode(self):
        sample_idx = jnp.argmax(self.logits, axis=-1)
        return jax.nn.one_hot(sample_idx, self.num_classes)
