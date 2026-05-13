---
title: "Sheeprl Reference: distribution.py"
source: tmp/sheeprl/sheeprl/utils/distribution.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `distribution.py`

> **Source**: `tmp/sheeprl/sheeprl/utils/distribution.py` — 416 lines, 57 def/class sections.
> **Purpose** (one-line): All probabilistic distributions used by DreamerV3 — `TwoHotEncodingDistribution` (reward / critic head), `SymlogDistribution` (observation reconstruction), `MSEDistribution` (alternative continuous), `BernoulliSafeMode` (continue head), `TruncatedNormal` / `TruncatedStandardNormal` (for some continuous-action variants), `OneHotCategoricalValidateArgs` / `OneHotCategoricalStraightThroughValidateArgs` (categorical latents + actions, the latter with straight-through gradients), plus a registered KL divergence between two `OneHotCategoricalValidateArgs`.
> **Imports from elsewhere in this index**: `sheeprl.utils.utils.symlog` and `sheeprl.utils.utils.symexp` (covered in [`utils_core.md`](utils_core.md)) — the `symlog(x) = sign(x) * log(|x| + 1)` and inverse `symexp(x) = sign(x) * (exp(|x|) - 1)` scale-invariant transforms.

The file's docstring credits two upstream sources: the truncated-normal block is lifted from `torch_truncnorm` by `toshas`, and `SymlogDistribution` is adapted from Danijar Hafner's `dreamerv3` JAX repository. Everything else is sheeprl-native or copied verbatim from PyTorch's `one_hot_categorical.py` with `validate_args` plumbing added.

The mathematical reason DreamerV3 needs two-hot/symlog distributions: rewards and returns are unbounded scalars whose scale varies wildly across tasks and over the course of training. Squashing them through `symlog` and then predicting a discrete distribution over a fixed grid of bins gives the value head a bounded, well-conditioned target while still letting it represent extreme values via the bin grid `symexp(linspace(-20, 20, K))` — i.e., the bins span roughly ±exp(20) in true reward space.

---

## Table of Contents

- [Lines 1–22 — Module docstring, imports, constants, `__all__`](#lines-122--module-docstring-imports-constants-__all__)
- [Line 25 — `class TruncatedStandardNormal`](#line-25--class-truncatedstandardnormal)
- [Line 37 — `TruncatedStandardNormal.__init__`](#line-37--truncatedstandardnormal__init__)
- [Line 67 — `TruncatedStandardNormal.support`](#line-67--truncatedstandardnormalsupport)
- [Line 71 — `TruncatedStandardNormal.mean`](#line-71--truncatedstandardnormalmean)
- [Line 75 — `TruncatedStandardNormal.variance`](#line-75--truncatedstandardnormalvariance)
- [Line 79 — `TruncatedStandardNormal.auc`](#line-79--truncatedstandardnormalauc)
- [Line 83 — `TruncatedStandardNormal._little_phi`](#line-83--truncatedstandardnormal_little_phi)
- [Line 87 — `TruncatedStandardNormal._big_phi`](#line-87--truncatedstandardnormal_big_phi)
- [Line 91 — `TruncatedStandardNormal._inv_big_phi`](#line-91--truncatedstandardnormal_inv_big_phi)
- [Line 94 — `TruncatedStandardNormal.cdf`](#line-94--truncatedstandardnormalcdf)
- [Line 99 — `TruncatedStandardNormal.icdf`](#line-99--truncatedstandardnormalicdf)
- [Line 102 — `TruncatedStandardNormal.log_prob`](#line-102--truncatedstandardnormallog_prob)
- [Line 107 — `TruncatedStandardNormal.rsample`](#line-107--truncatedstandardnormalrsample)
- [Line 112 — `TruncatedStandardNormal.entropy`](#line-112--truncatedstandardnormalentropy)
- [Line 116 — `class TruncatedNormal`](#line-116--class-truncatednormal)
- [Line 124 — `TruncatedNormal.__init__`](#line-124--truncatednormal__init__)
- [Line 134 — `TruncatedNormal._to_std_rv`](#line-134--truncatednormal_to_std_rv)
- [Line 137 — `TruncatedNormal._from_std_rv`](#line-137--truncatednormal_from_std_rv)
- [Line 140 — `TruncatedNormal.cdf`](#line-140--truncatednormalcdf)
- [Line 143 — `TruncatedNormal.icdf`](#line-143--truncatednormalicdf)
- [Line 146 — `TruncatedNormal.log_prob`](#line-146--truncatednormallog_prob)
- [Line 152 — `class SymlogDistribution`](#line-152--class-symlogdistribution)
- [Line 153 — `SymlogDistribution.__init__`](#line-153--symlogdistribution__init__)
- [Line 170 — `SymlogDistribution.mode`](#line-170--symlogdistributionmode)
- [Line 174 — `SymlogDistribution.mean`](#line-174--symlogdistributionmean)
- [Line 177 — `SymlogDistribution.log_prob`](#line-177--symlogdistributionlog_prob)
- [Line 196 — `class MSEDistribution`](#line-196--class-msedistribution)
- [Line 197 — `MSEDistribution.__init__`](#line-197--msedistribution__init__)
- [Line 205 — `MSEDistribution.mode`](#line-205--msedistributionmode)
- [Line 209 — `MSEDistribution.mean`](#line-209--msedistributionmean)
- [Line 212 — `MSEDistribution.log_prob`](#line-212--msedistributionlog_prob)
- [Line 224 — `class TwoHotEncodingDistribution`](#line-224--class-twohotencodingdistribution)
- [Line 225 — `TwoHotEncodingDistribution.__init__`](#line-225--twohotencodingdistribution__init__)
- [Line 246 — `TwoHotEncodingDistribution.mean`](#line-246--twohotencodingdistributionmean)
- [Line 250 — `TwoHotEncodingDistribution.mode`](#line-250--twohotencodingdistributionmode)
- [Line 253 — `TwoHotEncodingDistribution.log_prob`](#line-253--twohotencodingdistributionlog_prob)
- [Line 281 — `class OneHotCategoricalValidateArgs`](#line-281--class-onehotcategoricalvalidateargs)
- [Line 315 — `OneHotCategoricalValidateArgs.__init__`](#line-315--onehotcategoricalvalidateargs__init__)
- [Line 321 — `OneHotCategoricalValidateArgs.expand`](#line-321--onehotcategoricalvalidateargsexpand)
- [Line 329 — `OneHotCategoricalValidateArgs._new`](#line-329--onehotcategoricalvalidateargs_new)
- [Line 333 — `OneHotCategoricalValidateArgs._param`](#line-333--onehotcategoricalvalidateargs_param)
- [Line 337 — `OneHotCategoricalValidateArgs.probs`](#line-337--onehotcategoricalvalidateargsprobs)
- [Line 341 — `OneHotCategoricalValidateArgs.logits`](#line-341--onehotcategoricalvalidateargslogits)
- [Line 345 — `OneHotCategoricalValidateArgs.mean`](#line-345--onehotcategoricalvalidateargsmean)
- [Line 349 — `OneHotCategoricalValidateArgs.mode`](#line-349--onehotcategoricalvalidateargsmode)
- [Line 355 — `OneHotCategoricalValidateArgs.variance`](#line-355--onehotcategoricalvalidateargsvariance)
- [Line 359 — `OneHotCategoricalValidateArgs.param_shape`](#line-359--onehotcategoricalvalidateargsparam_shape)
- [Line 362 — `OneHotCategoricalValidateArgs.sample`](#line-362--onehotcategoricalvalidateargssample)
- [Line 369 — `OneHotCategoricalValidateArgs.log_prob`](#line-369--onehotcategoricalvalidateargslog_prob)
- [Line 375 — `OneHotCategoricalValidateArgs.entropy`](#line-375--onehotcategoricalvalidateargsentropy)
- [Line 378 — `OneHotCategoricalValidateArgs.enumerate_support`](#line-378--onehotcategoricalvalidateargsenumerate_support)
- [Line 387 — `class OneHotCategoricalStraightThroughValidateArgs`](#line-387--class-onehotcategoricalstraightthroughvalidateargs)
- [Line 398 — `OneHotCategoricalStraightThroughValidateArgs.rsample`](#line-398--onehotcategoricalstraightthroughvalidateargsrsample)
- [Line 405 — `_kl_onehotcategoricalvalidateargs_onehotcategoricalvalidateargs`](#line-405--_kl_onehotcategoricalvalidateargs_onehotcategoricalvalidateargs)
- [Line 409 — `class BernoulliSafeMode`](#line-409--class-bernoullisafemode)
- [Line 410 — `BernoulliSafeMode.__init__`](#line-410--bernoullisafemode__init__)
- [Line 414 — `BernoulliSafeMode.mode`](#line-414--bernoullisafemodemode)

---

## Lines 1–22 — Module docstring, imports, constants, `__all__`

```python
"""From: https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py"""

import math
from numbers import Number
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli, Categorical, Distribution, constraints
from torch.distributions.kl import _kl_categorical_categorical, register_kl
from torch.distributions.utils import broadcast_all

from sheeprl.utils.utils import symexp, symlog

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

__all__ = ["OneHotCategoricalValidateArgs", "OneHotCategoricalStraightThroughValidateArgs"]
```

**What it does**: Sets up the module. The docstring credits `toshas/torch_truncnorm` (the original source of the `TruncatedStandardNormal` / `TruncatedNormal` block). Imports cover `Number` (for the scalar-bounds branch in `TruncatedStandardNormal.__init__`), `Callable` (for `TwoHotEncodingDistribution`'s `transfwd` / `transbwd` typing), torch's `Distribution` base + `constraints` registry, the `register_kl` decorator + the underlying `_kl_categorical_categorical` reused by the KL registration at line 404, `broadcast_all` (used to unify shapes of `loc`/`scale`/`a`/`b`), and `symlog`/`symexp` from `utils.py`. The five `CONST_*` numerics are constants used in the standard-normal density and entropy formulas: `√2`, `1/√(2π)`, `1/√2`, `log(1/√(2π))`, and `½·log(2πe)` (the entropy of a standard normal). The `__all__` deliberately exports **only** the two categorical classes — the rest are internal but still importable by fully-qualified name.

---

## Line 25 — `class TruncatedStandardNormal`

```python
class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True
```

**What it does**: A standard normal `N(0,1)` truncated to the interval `[a, b]`. Subclasses `torch.distributions.Distribution`. Declares its two parameters (`a`, `b`) as real-valued via `arg_constraints` and advertises `has_rsample = True` so that `rsample` works through inverse-CDF sampling (so a reparameterization gradient flows through `a`, `b` — see line 107). Not used directly by DreamerV3's discrete-only configuration in this repo, but kept as part of the upstream `torch_truncnorm` block. The class is the parent of `TruncatedNormal` (line 116), which adds `loc`/`scale`.

---

## Line 37 — `TruncatedStandardNormal.__init__`

```python
    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any((self.a >= self.b).view(-1).tolist()):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z
```

**What it does**: Pre-computes all closed-form moments of the truncated normal so subsequent calls to `mean`/`variance`/`entropy` are O(1). `_Z = Φ(b) − Φ(a)` is the normalizing constant (the probability mass that survives truncation); it's `clamp_min`-ed to machine epsilon to avoid division-by-zero when bounds are nearly equal. `_mean = −(φ(b) − φ(a)) / Z` and the standard textbook variance formula `1 − [bφ(b) − aφ(a)]/Z − [(φ(b) − φ(a))/Z]²` follow. `_entropy = ½log(2πe) + log Z − ½·(bφ(b) − aφ(a))/Z` is the entropy of a truncated standard normal. The `nan_to_num` calls guard against `a = −∞` or `b = +∞` causing `0 · ∞` in the variance term. Raises `ValueError` if `a ≥ b` (degenerate or empty range) or if the two bounds have different dtypes.

---

## Line 67 — `TruncatedStandardNormal.support`

```python
    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)
```

**What it does**: Returns the support `[a, b]` as a torch `constraints.interval` object. `@constraints.dependent_property` is the canonical pattern for a support that depends on the distribution's own parameters (the bounds), so that `validate_args=True` can check that samples lie inside `[a, b]` per-instance rather than against a fixed global constraint.

---

## Line 71 — `TruncatedStandardNormal.mean`

```python
    @property
    def mean(self):
        return self._mean
```

**What it does**: Returns the pre-computed mean `−(φ(b) − φ(a)) / Z` cached in `__init__`. The closed form: for a standard normal truncated to `[a, b]`, the mean shifts toward whichever bound carries more of the density.

---

## Line 75 — `TruncatedStandardNormal.variance`

```python
    @property
    def variance(self):
        return self._variance
```

**What it does**: Returns the pre-computed variance from `__init__`, `1 − [bφ(b) − aφ(a)]/Z − ((φ(b) − φ(a))/Z)²`. Always ≤ 1 (truncation can only reduce variance vs. the underlying `N(0,1)`).

---

## Line 79 — `TruncatedStandardNormal.auc`

```python
    @property
    def auc(self):
        return self._Z
```

**What it does**: Returns the "area under curve" `Z = Φ(b) − Φ(a)`, i.e. the probability mass of the *untruncated* standard normal that lies inside `[a, b]`. Useful for inverse importance weighting if you re-cast samples back to the untruncated normal.

---

## Line 83 — `TruncatedStandardNormal._little_phi`

```python
    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI
```

**What it does**: The standard-normal *pdf* `φ(x) = (1/√(2π))·exp(−x²/2)`. Static method because it's a pure helper that doesn't depend on instance state. Called four times in `__init__` (once for each of `a`, `b` × pdf, cdf).

---

## Line 87 — `TruncatedStandardNormal._big_phi`

```python
    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
```

**What it does**: The standard-normal *cdf* `Φ(x) = ½·(1 + erf(x/√2))`. Uses the relation between the Gauss error function and the normal cdf. Cached in `__init__` as `_big_phi_a` / `_big_phi_b`; also called from `cdf` (line 94).

---

## Line 91 — `TruncatedStandardNormal._inv_big_phi`

```python
    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()
```

**What it does**: The inverse cdf (quantile function) `Φ⁻¹(x) = √2 · erfinv(2x − 1)`. The cornerstone of inverse-CDF sampling: feed a uniform `u ∈ (0, 1)` to get a standard-normal sample. Used in both `icdf` (line 99) and indirectly in `rsample` (line 107).

---

## Line 94 — `TruncatedStandardNormal.cdf`

```python
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)
```

**What it does**: Truncated-normal cdf: `F(x) = (Φ(x) − Φ(a)) / Z` for `x ∈ [a, b]`. The result is clamped to `[0, 1]` defensively in case of floating-point drift. Validates that `value` lies in `[a, b]` only when `validate_args=True`.

---

## Line 99 — `TruncatedStandardNormal.icdf`

```python
    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)
```

**What it does**: Truncated-normal inverse cdf: given `u ∈ [0, 1]`, returns `Φ⁻¹(Φ(a) + u·Z)`. This maps `u = 0 → a` and `u = 1 → b`. Used in `rsample` (line 107) — sample uniform then push through this to get a truncated-normal sample.

---

## Line 102 — `TruncatedStandardNormal.log_prob`

```python
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5
```

**What it does**: Truncated standard-normal log-density: `log φ(x) − log Z = log(1/√(2π)) − x²/2 − log Z`. Subtracting `log_Z` is the normalization correction for truncation. Inputs outside `[a, b]` are not handled here (they'd give a finite log-prob even though the true density is zero) — the caller is expected to validate.

---

## Line 107 — `TruncatedStandardNormal.rsample`

```python
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)
```

**What it does**: Reparameterized sampling via inverse-CDF: draw `p ~ Uniform(ε, 1−ε)` (the `ε` margin avoids `erfinv(±1) = ±∞`), then map through `icdf`. Because `icdf` is a deterministic function of `a`, `b` and the uniform noise, gradients flow back through the bounds (this is what makes `has_rsample = True` honest). `_extended_shape` combines `sample_shape` with `batch_shape`.

---

## Line 112 — `TruncatedStandardNormal.entropy`

```python
    def entropy(self):
        return self._entropy
```

**What it does**: Returns the pre-computed entropy `½log(2πe) + log Z − ½·(bφ(b) − aφ(a))/Z`. The first term is the entropy of an untruncated `N(0,1)`; the next two terms correct for the lost mass and the redistributed density.

---

## Line 116 — `class TruncatedNormal`

```python
class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True
```

**What it does**: General location-scale truncated normal `N(loc, scale²)` restricted to `[a, b]`. Implemented as a thin wrapper that *standardizes* its bounds and shifts/scales the parent class's moments/sampling. Inherits `cdf`/`icdf`/`log_prob`/`rsample` and overrides them to insert the affine transform. `has_rsample = True` is redeclared for clarity — gradients still flow through `loc`, `scale`, `a`, `b`.

---

## Line 124 — `TruncatedNormal.__init__`

```python
    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale
```

**What it does**: Broadcasts all four parameters to a common shape, standardizes `(a, b) → ((a−loc)/scale, (b−loc)/scale)`, then delegates to `TruncatedStandardNormal.__init__` with the standardized bounds. After the super-init has cached the standardized moments, this method **re-scales them back to original units**: `mean → mean·scale + loc`, `variance → variance·scale²`, `entropy → entropy + log scale` (the entropy of a scaled rv shifts by `log|scale|`).

---

## Line 134 — `TruncatedNormal._to_std_rv`

```python
    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale
```

**What it does**: Z-score a value: subtract `loc`, divide by `scale`. Used by `cdf` / `log_prob` to dispatch to the parent class which works on the standardized scale.

---

## Line 137 — `TruncatedNormal._from_std_rv`

```python
    def _from_std_rv(self, value):
        return value * self.scale + self.loc
```

**What it does**: Inverse of `_to_std_rv`: shift standardized samples back to the original `(loc, scale)` scale. Used by `icdf` (which calls into the standardized inverse-cdf and then de-standardizes).

---

## Line 140 — `TruncatedNormal.cdf`

```python
    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))
```

**What it does**: Standardize then evaluate the parent's cdf. `P(X ≤ value)` for the location-scale rv equals `P(Z ≤ (value − loc)/scale)` for the standardized rv.

---

## Line 143 — `TruncatedNormal.icdf`

```python
    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))
```

**What it does**: Compute the standardized quantile, then de-standardize to the original scale. Underlies `rsample` (inherited unchanged from the parent — but since the parent calls `self.icdf`, dispatch picks up this override).

---

## Line 146 — `TruncatedNormal.log_prob`

```python
    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale
```

**What it does**: Standardize, evaluate standardized log-density, then **subtract `log scale`** to account for the change-of-variables Jacobian: if `X = scale·Z + loc`, then `p_X(x) = p_Z(z)/scale`, so `log p_X(x) = log p_Z(z) − log scale`.

---

## Line 152 — `class SymlogDistribution`

```python
class SymlogDistribution:
```

**What it does**: A non-`torch.distributions.Distribution` ("duck-typed") wrapper that represents the prediction `mode` in **symlog space** — i.e. you predict `symlog(target)`, and `log_prob` measures the squared (or abs) error in that transformed space. Used by DreamerV3 for **observation reconstruction** and as one option for the reward head. The `TODO` comment above (line 150) flags that this is intentionally not yet API-compliant. Note the upstream credit to Hafner's `dreamerv3/jaxutils.py`.

---

## Line 153 — `SymlogDistribution.__init__`

```python
    def __init__(
        self,
        mode: Tensor,
        dims: int,
        dist: str = "mse",
        agg: str = "sum",
        tol: float = 1e-8,
    ):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]
```

**What it does**: Stores `mode` (the network's raw prediction, already in *symlog* space — so the actual point estimate is `symexp(mode)`). `dims` is the number of trailing dimensions that constitute the event (e.g. 3 for an image `H × W × C`); `_dims = (-1, -2, ..., -dims)` is the tuple of axes that `log_prob` reduces over. `dist` selects the inner per-element loss (`"mse"` → squared error, `"abs"` → L1). `tol` is the threshold below which per-element distances are zeroed out (a "free-bits" / "noise-floor" trick to ignore sub-pixel jitter). `agg` is how to reduce across event dims (`"sum"` or `"mean"`). The two `_shape` attributes mimic torch's API.

---

## Line 170 — `SymlogDistribution.mode`

```python
    @property
    def mode(self) -> Tensor:
        return symexp(self._mode)
```

**What it does**: Converts the symlog-space prediction back to the original scale. Because `symexp` is the inverse of `symlog`, and the network was trained to output `symlog(target)`, `symexp(self._mode)` is the point estimate in real units.

---

## Line 174 — `SymlogDistribution.mean`

```python
    @property
    def mean(self) -> Tensor:
        return symexp(self._mode)
```

**What it does**: Same as `mode` — for a symmetric loss centered at `mode`, the mean coincides with the mode. (Strictly, this is the predicted point estimate, not the true expectation under the implied density; DreamerV3 treats them as interchangeable in practice.)

---

## Line 177 — `SymlogDistribution.log_prob`

```python
    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss
```

**What it does**: Asserts shape compatibility between target and prediction. Pushes the **target** through `symlog`, then measures squared (or L1) deviation from the prediction `self._mode` in symlog space — so the network never sees raw unbounded values. Distances below `tol` are zeroed to suppress noise. Reduces across the event dims with `sum`/`mean`, then negates so it has the sign of a log-density (loss = `−log_prob`). This is the **observation-reconstruction term** in the world-model loss when the obs decoder uses symlog (see `loss.py`).

---

## Line 196 — `class MSEDistribution`

```python
class MSEDistribution:
```

**What it does**: Same shape and API as `SymlogDistribution`, but **no `symlog` transform** — just plain mean-squared error against `self._mode`. Used when the target is already in a well-behaved range (bounded observations, normalized features). The simplest possible continuous "distribution" sheeprl ships.

---

## Line 197 — `MSEDistribution.__init__`

```python
    def __init__(self, mode: Tensor, dims: int, agg: str = "sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]
```

**What it does**: Mirrors `SymlogDistribution.__init__` minus the `dist` and `tol` knobs (MSE is the only mode, and there's no symlog tolerance to discuss). Splits the input shape into `batch_shape` (leading dims) and `event_shape` (last `dims` dims), and pre-builds the reduction axis tuple `_dims = (-1, ..., -dims)`.

---

## Line 205 — `MSEDistribution.mode`

```python
    @property
    def mode(self) -> Tensor:
        return self._mode
```

**What it does**: Trivial — the prediction *is* the mode in plain-MSE space (no symlog transform to invert).

---

## Line 209 — `MSEDistribution.mean`

```python
    @property
    def mean(self) -> Tensor:
        return self._mode
```

**What it does**: Same as `mode` — for a Gaussian centered at `mode`, mean = mode.

---

## Line 212 — `MSEDistribution.log_prob`

```python
    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss
```

**What it does**: Plain squared-error log-density up to a constant: `−Σ (mode − value)²`. Treats the underlying distribution as a unit-variance Gaussian (the `−½ log(2πσ²)` constant is dropped). Used as the obs-reconstruction term in world-model loss when the decoder predicts raw values.

---

## Line 224 — `class TwoHotEncodingDistribution`

```python
class TwoHotEncodingDistribution:
```

**What it does**: The **critical** distribution for DreamerV3's reward and critic heads. Discrete distribution over a fixed grid of `K` bins (logits shape `[..., K]`). The grid spans `[low, high]` in *symlog* space, so the bin centers in real space are `symexp(linspace(low, high, K))` — see line 237. A scalar target is encoded across the **two adjacent bins** that bracket its symlog value, weighted so the expectation matches the target exactly. This gives the network a discrete classification target while preserving regression-style differentiability with respect to the target value.

---

## Line 225 — `TwoHotEncodingDistribution.__init__`

```python
    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
    ):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)
```

**What it does**: Builds the bin grid. `low=-20`, `high=20`, `K = logits.shape[-1]` (typically 255 in the canonical config). The grid is `linspace(low, high, K)` in *symlog* space — equivalent to bin centers at `symexp(linspace(-20, 20, K))` in real reward space. (This is the "paper-canonical bins" debate referenced in the project notes: the bins are uniform in symlog space, not in linear space.) `transfwd = symlog` and `transbwd = symexp` are kept as parameters so the encoding can in principle use other transforms. `probs = softmax(logits)` is cached for `mean`/`mode`. `dims` selects how many trailing dims (other than the bin axis) participate in the event — typically `0` for a scalar head (reward, value), giving `_event_shape = (..., 1)`.

---

## Line 246 — `TwoHotEncodingDistribution.mean`

```python
    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))
```

**What it does**: Computes the **expected value in symlog space** (`Σ probs · bin_center_symlog`) and then maps it back to real units with `transbwd = symexp`. This is the standard way to extract a scalar prediction from a two-hot head (used by the critic to read off the predicted value, and by the reward head for evaluation).

---

## Line 250 — `TwoHotEncodingDistribution.mode`

```python
    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))
```

**What it does**: Identical to `mean` — sheeprl uses the expectation as the "mode" for downstream callers. Strictly, the true mode would be the bin with highest probability, but the expectation is a smoother point estimate and is what DreamerV3 actually uses for value estimation.

---

## Line 253 — `TwoHotEncodingDistribution.log_prob`

```python
    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(self.bins) - 1]
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = below + 1

        # above in [0, len(self.bins) - 1]
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        # below in [0, len(self.bins) - 1]
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        return (target * log_pred).sum(dim=self.dims)
```

**What it does**: The two-hot encoding + cross-entropy. (1) Push the scalar target `x` through `symlog`. (2) Find the two bracketing bin indices `below` and `above` (clamped to valid range). (3) The target distribution puts weight `dist_to_above / total` on `below` and `dist_to_below / total` on `above` — note the **inverted** assignment: the closer the target is to the upper bin, the more weight goes to the upper bin. This is the algebraic identity that makes the expectation `weight_below · bins[below] + weight_above · bins[above]` equal `x` exactly. (4) The `equal` mask handles the edge case where target hits a bin exactly (both distances are 0 → divide-by-zero → replaced with 1, putting all weight on whichever bin). (5) `log_pred = logits − logsumexp(logits)` is `log softmax`. (6) Return `Σ target · log_pred` — the standard cross-entropy log-likelihood. This is the loss term used by both the reward head and the value head in DreamerV3.

---

## Line 281 — `class OneHotCategoricalValidateArgs`

```python
class OneHotCategoricalValidateArgs(Distribution):
```

**What it does**: Verbatim copy of PyTorch's stock `OneHotCategorical` with the *one* difference that it threads `validate_args` through to the inner `Categorical`. This matters because DreamerV3 wants to validate categorical probs/logits during world-model training (e.g. to catch NaN or zero-sum simplex inputs). The full class docstring (lines 282–309 in the source, included in the code block of `__init__` below) explains: samples are one-hot vectors; `probs` must be non-negative, finite, non-zero-sum; `logits` are interpreted as unnormalized log-probs.

`arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}`, `support = constraints.one_hot`, `has_enumerate_support = True`. Exported in `__all__` (line 22).

---

## Line 315 — `OneHotCategoricalValidateArgs.__init__`

```python
    def __init__(self, probs=None, logits=None, validate_args=None):
        self._categorical = Categorical(probs, logits, validate_args=validate_args)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
```

**What it does**: Delegates almost everything to an inner `torch.distributions.Categorical`. Pulls `batch_shape` from the categorical and sets `event_shape = (K,)` where `K = param_shape[-1]` is the number of categories. Forwards `validate_args` both into the inner categorical and into the `Distribution` superclass init — this is the line that fixes the upstream omission.

---

## Line 321 — `OneHotCategoricalValidateArgs.expand`

```python
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(OneHotCategoricalValidateArgs, _instance)
        batch_shape = torch.Size(batch_shape)
        new._categorical = self._categorical.expand(batch_shape)
        super(OneHotCategoricalValidateArgs, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
```

**What it does**: Standard torch-distributions `expand`: produces a new instance whose `batch_shape` is broader (broadcasting). Re-uses `_get_checked_instance` to optionally fill an existing object (subclass-friendly). Propagates `_validate_args` manually since super-init is called with `validate_args=False` to skip re-validation of already-validated parameters.

---

## Line 329 — `OneHotCategoricalValidateArgs._new`

```python
    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)
```

**What it does**: Delegates `_new` (a torch-internal helper that creates a fresh tensor on the right device/dtype) to the inner categorical. Used by torch's machinery during sampling and `_extended_shape`.

---

## Line 333 — `OneHotCategoricalValidateArgs._param`

```python
    @property
    def _param(self):
        return self._categorical._param
```

**What it does**: Internal access to whichever of `probs` or `logits` was passed in (whichever is the "owning" parameter). Used in `enumerate_support` (line 378) to pick the right dtype/device for the identity tensor.

---

## Line 337 — `OneHotCategoricalValidateArgs.probs`

```python
    @property
    def probs(self):
        return self._categorical.probs
```

**What it does**: Returns the normalized probabilities from the inner categorical. Because `Categorical` lazy-normalizes when given `probs` and computes `softmax(logits)` when given `logits`, this gets the right answer either way.

---

## Line 341 — `OneHotCategoricalValidateArgs.logits`

```python
    @property
    def logits(self):
        return self._categorical.logits
```

**What it does**: Returns the normalized log-probabilities (i.e. `log_softmax(logits)` if logits were passed, else `log(probs)`).

---

## Line 345 — `OneHotCategoricalValidateArgs.mean`

```python
    @property
    def mean(self):
        return self._categorical.probs
```

**What it does**: For a one-hot categorical, the mean (per-component) is exactly the probability vector — `E[one_hot(i)] = probs`. Returns a vector in the probability simplex, not a one-hot.

---

## Line 349 — `OneHotCategoricalValidateArgs.mode`

```python
    @property
    def mode(self):
        probs = self._categorical.probs
        mode = probs.argmax(axis=-1)
        return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)
```

**What it does**: Argmax + one-hot encoding — the most-likely category, returned as a one-hot vector cast to the same dtype as `probs`. Ties are broken by `argmax`'s "first occurrence" rule.

---

## Line 355 — `OneHotCategoricalValidateArgs.variance`

```python
    @property
    def variance(self):
        return self._categorical.probs * (1 - self._categorical.probs)
```

**What it does**: Per-component Bernoulli variance: `p_k · (1 − p_k)` for each category `k`. Note: this ignores the across-category covariance (which is `−p_i p_j` for `i ≠ j`); just the diagonal.

---

## Line 359 — `OneHotCategoricalValidateArgs.param_shape`

```python
    @property
    def param_shape(self):
        return self._categorical.param_shape
```

**What it does**: Shape of the underlying parameter tensor, i.e. `(*batch_shape, K)`. Forwarded directly from the inner categorical.

---

## Line 362 — `OneHotCategoricalValidateArgs.sample`

```python
    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        probs = self._categorical.probs
        num_events = self._categorical._num_events
        indices = self._categorical.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).to(probs)
```

**What it does**: Sample category index via the inner categorical's `sample` (multinomial sampling), then one-hot encode. Cast back to the dtype of `probs` so downstream tensor ops stay consistent. **Non-reparameterized** (no gradient through sampling) — the straight-through variant at line 387 fixes this.

---

## Line 369 — `OneHotCategoricalValidateArgs.log_prob`

```python
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indices = value.max(-1)[1]
        return self._categorical.log_prob(indices)
```

**What it does**: Converts the one-hot `value` back into an index via `argmax`, then delegates to the inner categorical's `log_prob`. The `max(-1)[1]` idiom returns the indices (`max` returns a `(values, indices)` named-tuple). Validates that `value` is actually a valid one-hot if `validate_args=True`.

---

## Line 375 — `OneHotCategoricalValidateArgs.entropy`

```python
    def entropy(self):
        return self._categorical.entropy()
```

**What it does**: Entropy of the categorical: `−Σ p_k log p_k`. Identical to the underlying categorical's entropy because one-hot encoding is a bijection.

---

## Line 378 — `OneHotCategoricalValidateArgs.enumerate_support`

```python
    def enumerate_support(self, expand=True):
        n = self.event_shape[0]
        values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        if expand:
            values = values.expand((n,) + self.batch_shape + (n,))
        return values
```

**What it does**: Returns all `n` one-hot vectors of the support, as the rows of an `n × n` identity matrix. The view-then-expand pattern broadcasts those `n` values across the entire `batch_shape` if `expand=True`, otherwise keeps them at minimum size. Used by `register_kl`-style enumeration and for analytic-expectation utilities.

---

## Line 387 — `class OneHotCategoricalStraightThroughValidateArgs`

```python
class OneHotCategoricalStraightThroughValidateArgs(OneHotCategoricalValidateArgs):
    r"""
    Creates a reparameterizable :class:`OneHotCategoricalValidateArgs` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al, 2013)
    """

    has_rsample = True
```

**What it does**: Subclass that *adds* `rsample` via the **straight-through estimator** (Bengio et al. 2013): forward pass is a hard one-hot sample, backward pass uses the gradient of the soft probabilities. This is what DreamerV3 uses for its **categorical latent states** in the RSSM — discrete on the forward pass (for the world model's expressiveness) but differentiable on the backward pass (so the encoder can be trained end-to-end). Also used for the actor's discrete-action distribution. The class docstring credits Bengio's original ST estimator paper.

---

## Line 398 — `OneHotCategoricalStraightThroughValidateArgs.rsample`

```python
    def rsample(self, sample_shape=torch.Size()):
        samples = self.sample(sample_shape)
        probs = self._categorical.probs  # cached via @lazy_property
        return samples + (probs - probs.detach())
```

**What it does**: The straight-through trick in one line. `samples` is the hard one-hot (no gradient). `probs - probs.detach()` is **zero-valued in the forward pass** (`probs == probs.detach()` numerically) but **has the gradient of `probs`** in the backward pass. So forward = one-hot, backward = ∂L/∂probs. The result is a tensor that behaves discretely in the forward pass and continuously for gradient computation — this is what lets DreamerV3 train through discrete latents and discrete actions with standard backprop.

---

## Line 405 — `_kl_onehotcategoricalvalidateargs_onehotcategoricalvalidateargs`

```python
@register_kl(OneHotCategoricalValidateArgs, OneHotCategoricalValidateArgs)
def _kl_onehotcategoricalvalidateargs_onehotcategoricalvalidateargs(p, q):
    return _kl_categorical_categorical(p._categorical, q._categorical)
```

**What it does**: Registers a KL divergence between two `OneHotCategoricalValidateArgs` (and any subclass, including the straight-through one) by delegating to the existing `_kl_categorical_categorical` on the inner categoricals. `register_kl` populates torch's dispatch table so that `torch.distributions.kl_divergence(p, q)` picks this up. **Critical for DreamerV3**: the world-model loss includes a KL between the *posterior* RSSM latent (output of the representation model conditioned on observation) and the *prior* RSSM latent (output of the transition model from previous state only) — both are categorical, both go through this KL.

---

## Line 409 — `class BernoulliSafeMode`

```python
class BernoulliSafeMode(Bernoulli):
```

**What it does**: A `torch.distributions.Bernoulli` with one tiny override: a numerically safe `mode`. The stock Bernoulli's `mode` is `(probs > 0.5)` cast back to `probs`'s dtype — but the docstring on stock Bernoulli's `mode` warns it can return NaN when `probs == 0.5` exactly. This subclass simply hard-codes `probs > 0.5` and skips the NaN case (always returns 0 at the boundary). Used by DreamerV3's **continue head**: a Bernoulli over "episode continues" (1) vs "episode ends" (0), and the predicted mode is what gates whether imagined rollouts continue.

---

## Line 410 — `BernoulliSafeMode.__init__`

```python
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)
```

**What it does**: Verbatim pass-through to `torch.distributions.Bernoulli.__init__`. Exists solely to make the class non-abstract and to document that no special parameter handling happens at construction.

---

## Line 414 — `BernoulliSafeMode.mode`

```python
    @property
    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode
```

**What it does**: Returns a hard 0/1 tensor: 1 when the success probability strictly exceeds 0.5, 0 otherwise (including the boundary case `probs == 0.5`). Casts back to `self.probs`'s dtype so downstream arithmetic stays in float. This is consumed in the imagination rollout: when the continue-head says "more likely to terminate than continue", the rollout discounts subsequent rewards (or stops).
