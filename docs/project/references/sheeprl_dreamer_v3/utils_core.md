---
title: "Sheeprl Reference: sheeprl/utils/utils.py (core utilities)"
source: tmp/sheeprl/sheeprl/utils/utils.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `sheeprl/utils/utils.py` (core utilities)

> **Source**: `tmp/sheeprl/sheeprl/utils/utils.py` — 313 lines, 23 def/class sections.
> **Purpose** (one-line): Cross-algorithm utilities: `Ratio` (replay-ratio scheduler), `polyak_update` (EMA target network — note: not present in this file; lives elsewhere), `save_configs`, and assorted helpers (`dotdict`, `gae`, `symlog`/`symexp`, two-hot codec, `print_config`, `unwrap_fabric`, `safetanh`/`safeatanh`).
> **Imports from elsewhere in this index**: (none significant; pure utility module).
> **Note**: this file is `sheeprl/utils/utils.py`, distinct from `sheeprl/algos/dreamer_v3/utils.py` which is documented separately as [`utils.md`](utils.md).

---

## Table of Contents

- [Lines 1–16 — Imports](#lines-116--imports)
- [Lines 18–31 — Module-level constants (`NUMPY_TO_TORCH_DTYPE_DICT`, `TORCH_TO_NUMPY_DTYPE_DICT`)](#lines-1831--module-level-constants-numpy_to_torch_dtype_dict-torch_to_numpy_dtype_dict)
- [Line 34 — `class dotdict`](#line-34--class-dotdict)
- [Line 43 — `dotdict.__init__`](#line-43--dotdict__init__)
- [Line 49 — `dotdict.__getstate__`](#line-49--dotdict__getstate__)
- [Line 52 — `dotdict.__setstate__`](#line-52--dotdict__setstate__)
- [Line 55 — `dotdict.as_dict`](#line-55--dotdictas_dict)
- [Line 64 — `gae`](#line-64--gae)
- [Line 103 — `init_weights`](#line-103--init_weights)
- [Line 121 — `normalize_tensor`](#line-121--normalize_tensor)
- [Line 133 — `polynomial_decay`](#line-133--polynomial_decay)
- [Line 148 — `symlog`](#line-148--symlog)
- [Line 152 — `symexp`](#line-152--symexp)
- [Line 156 — `two_hot_encoder`](#line-156--two_hot_encoder)
- [Line 191 — `two_hot_decoder`](#line-191--two_hot_decoder)
- [Line 209 — `print_config`](#line-209--print_config)
- [Line 240 — `unwrap_fabric`](#line-240--unwrap_fabric)
- [Line 255 — `save_configs`](#line-255--save_configs)
- [Line 259 — `class Ratio`](#line-259--class-ratio)
- [Line 264 — `Ratio.__init__`](#line-264--ratio__init__)
- [Line 273 — `Ratio.__call__`](#line-273--ratio__call__)
- [Line 293 — `Ratio.state_dict`](#line-293--ratiostate_dict)
- [Line 296 — `Ratio.load_state_dict`](#line-296--ratioload_state_dict)
- [Line 304 — `safetanh`](#line-304--safetanh)
- [Line 311 — `safeatanh`](#line-311--safeatanh)

---

## Lines 1–16 — Imports

```python
from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rich.syntax
import rich.tree
import torch
import torch.nn as nn
from lightning.fabric.wrappers import _FabricModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
```

The `from __future__ import annotations` enables PEP 604 union syntax (`_FabricModule | nn.Module`) under older Pythons by deferring annotation evaluation. The `copy` module backs `unwrap_fabric`, `os` backs config-save paths, and `warnings` is used by `Ratio` to flag a pretrain misconfiguration. Numpy + torch dtype tables (next block) need `np` and `torch`. `rich.syntax` / `rich.tree` drive the pretty `print_config`. `_FabricModule` is Lightning Fabric's wrapper class that `unwrap_fabric` peels off. `DictConfig` + `OmegaConf` underpin Hydra config save/print. `rank_zero_only` ensures `print_config` only fires on the rank-0 process under distributed training. `Tensor` is the type alias used throughout.

---

## Lines 18–31 — Module-level constants (`NUMPY_TO_TORCH_DTYPE_DICT`, `TORCH_TO_NUMPY_DTYPE_DICT`)

```python
NUMPY_TO_TORCH_DTYPE_DICT = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
TORCH_TO_NUMPY_DTYPE_DICT = {value: key for key, value in NUMPY_TO_TORCH_DTYPE_DICT.items()}
```

Two lookup tables mapping NumPy dtypes ↔ torch dtypes. Used by the replay buffer and env-side glue code to keep numpy ndarrays (env outputs) and torch tensors (model inputs) in sync without scattering `if/elif` chains. The inverse table is built by dict-comprehension from the forward table to guarantee they cannot drift. Note that `torch` has no exact analogue for some numpy dtypes (e.g. `float128`) so the table is deliberately a curated subset of common ML dtypes. These are referenced throughout sheeprl wherever a numpy array must be cast to a torch tensor with a matching dtype.

---

## Line 34 — `class dotdict`

```python
class dotdict(dict):
    """
    A dictionary supporting dot notation.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
```

A thin `dict` subclass that exposes attribute-style access (`cfg.algo` instead of `cfg["algo"]`). `__getattr__` is bound to `dict.get` so missing keys return `None` rather than raising `AttributeError`, which makes Hydra config consumers tolerant of optional fields. `__setattr__` / `__delattr__` are aliased to the corresponding dict mutators so writes go through the dict storage rather than into `__dict__`. Used throughout sheeprl to wrap the OmegaConf-resolved config in a pythonic, attribute-addressable container.

---

## Line 43 — `dotdict.__init__`

```python
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)
```

Constructs from any `dict`-compatible argument set, then **recursively** wraps every nested `dict` value as a `dotdict` so `cfg.algo.world_model.recurrent_state_size`-style chained access works at arbitrary depth. The recursion runs at construction time (not lazily) so attribute access has no per-call overhead. Note: nested non-`dict` containers (e.g. lists of dicts) are not walked — only top-level dict values per level — so a `list[dict]` would yield raw `dict` elements.

---

## Line 49 — `dotdict.__getstate__`

```python
    def __getstate__(self):
        return self
```

Pickle hook: returns the instance itself as its serialised state. Because `dotdict` inherits from `dict`, the dict contents are pickled via the built-in dict protocol; this hook just makes the round-trip explicit and lets `__setstate__` re-hydrate correctly. Needed for checkpointing runs where the config is stored as part of training state.

---

## Line 52 — `dotdict.__setstate__`

```python
    def __setstate__(self, state):
        self.update(state)
```

Pickle restore: rebuilds the instance by `.update`-ing from the serialised state. Pairs with `__getstate__`. Note that this does **not** re-run `__init__`, so nested dicts coming out of a pickle may be raw `dict` objects rather than `dotdict` — a subtle quirk if you pickle, unpickle, and then try `cfg.algo.world_model` chained access on a checkpoint-restored config.

---

## Line 55 — `dotdict.as_dict`

```python
    def as_dict(self) -> Dict[str, Any]:
        _copy = dict(self)
        for k, v in _copy.items():
            if isinstance(v, dotdict):
                _copy[k] = v.as_dict()
        return _copy
```

Recursively un-wraps a `dotdict` tree back into plain `dict`s. Required before handing the config to `OmegaConf.save` (see `save_configs` below) because OmegaConf's serializer expects plain mappings. The recursion mirrors `__init__`'s wrap-on-construct pattern. Returns a shallow copy at each level — values are not deep-copied, so mutating tensors / arrays inside the returned dict will mutate the originals.

---

## Line 64 — `gae`

```python
@torch.no_grad()
def gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Compute returns and advantages following https://arxiv.org/abs/1506.02438

    Args:
        rewards (Tensor): all rewards collected from the last rollout
        values (Tensor): all values collected from the last rollout
        dones (Tensor): all dones collected from the last rollout
        next_values (Tensor): estimated values for the next observations
        num_steps (int): the number of steps played
        gamma (float): discout factor
        gae_lambda (float): lambda for GAE estimation

    Returns:
        estimated returns
        estimated advantages
    """
    lastgaelam = 0
    nextvalues = next_value
    not_dones = torch.logical_not(dones)
    nextnonterminal = not_dones[-1]
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        if t < num_steps - 1:
            nextnonterminal = not_dones[t]
            nextvalues = values[t + 1]
        delta = rewards[t] + nextvalues * nextnonterminal * gamma - values[t]
        advantages[t] = lastgaelam = delta + nextnonterminal * lastgaelam * gamma * gae_lambda
    returns = advantages + values
    return returns, advantages
```

Generalised Advantage Estimation (Schulman et al. 2015). Walks the rollout backwards in time computing TD residuals `δ_t = r_t + γ · V(s_{t+1}) · (1-done_t) - V(s_t)` and accumulating them into `A_t = δ_t + γλ(1-done_t)A_{t+1}`. Returns `(returns, advantages)` where `returns = advantages + values` (the value-baseline-corrected n-step bootstrap target). Decorated `@torch.no_grad()` because GAE is computed from already-detached value estimates and the returned tensors are treated as fixed targets. Note this is used by PPO-family algorithms — DreamerV3 uses lambda-returns computed inside its actor-critic, not this function.

---

## Line 103 — `init_weights`

```python
def init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to the method described in
    [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852) using a uniform distribution.

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
```

He-style (Kaiming uniform) initialisation applied module-by-module via `model.apply(init_weights)`. Conv2d / ConvTranspose2d weights are init'd with `nonlinearity="relu"` (gain = √2). Linear weights use the default `nonlinearity="leaky_relu"` (which behaves like `relu` for the standard ReLU case at slope 0). All biases zero-init. Other module types (LayerNorm, GRU, etc.) are skipped — those keep their PyTorch defaults. Note: DreamerV3 does its own initialisation inside `models.py`, so this helper is mostly used by the simpler algos.

---

## Line 121 — `normalize_tensor`

```python
@torch.no_grad()
def normalize_tensor(tensor: Tensor, eps: float = 1e-8, mask: Optional[Tensor] = None) -> Tensor:
    unmasked = mask is None
    if unmasked:
        mask = torch.ones_like(tensor, dtype=torch.bool)
    masked_tensor = tensor[mask]
    normalized = (masked_tensor - masked_tensor.mean()) / (masked_tensor.std() + eps)
    if unmasked:
        return normalized.reshape_as(mask)
    else:
        return normalized
```

Standardises a tensor to zero mean and unit standard deviation under `@torch.no_grad()`, with optional boolean mask to restrict the statistics to a subset (e.g. exclude padding entries). When unmasked, the result is reshaped back to the input shape; when masked, only the masked entries are returned (1-D flattened). The `eps=1e-8` floor on the denominator prevents division by zero if the (masked) tensor is constant. Commonly used to normalise GAE advantages before PPO updates.

---

## Line 133 — `polynomial_decay`

```python
def polynomial_decay(
    current_step: int,
    *,
    initial: float = 1.0,
    final: float = 0.0,
    max_decay_steps: int = 100,
    power: float = 1.0,
) -> float:
    if current_step > max_decay_steps or initial == final:
        return final
    else:
        return (initial - final) * ((1 - current_step / max_decay_steps) ** power) + final
```

Polynomial decay schedule of the form `f(t) = (initial - final) · (1 - t/T)^power + final` for `t ∈ [0, T]`, clamped to `final` outside that range. With `power=1.0` (the default) this collapses to a linear decay from `initial` to `final` over `max_decay_steps`. Used for things like entropy-coefficient annealing or clip-coefficient decay in PPO. Note the keyword-only marker `*` — all decay parameters must be passed by name, only `current_step` is positional.

---

## Line 148 — `symlog`

```python
# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))
```

Symmetric logarithm: `sign(x) · log(1 + |x|)`. Compresses large-magnitude values toward zero on both sides of the origin while keeping the function smooth and differentiable everywhere (unlike a plain `log` which is undefined for `x ≤ 0`). This is DreamerV3's core trick for stabilising training across environments whose reward / return scales vary by orders of magnitude — the world model regresses on `symlog(reward)` and `symlog(return)` targets rather than raw values. Inverse is `symexp` below.

---

## Line 152 — `symexp`

```python
def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

Inverse of `symlog`: `sign(x) · (exp(|x|) - 1)`. Applied at inference / target-decoding time to map the model's symlog-space prediction back to the original reward / return scale. Together with `symlog` and the two-hot codec below, this is the numerical-stability backbone of DreamerV3's reward predictor and critic.

---

## Line 156 — `two_hot_encoder`

```python
def two_hot_encoder(tensor: Tensor, support_range: int = 300, num_buckets: Optional[int] = None) -> Tensor:
    """Encode a tensor representing a floating point number `x` as a tensor with all zeros except for two entries in the
    indexes of the two buckets closer to `x` in the support of the distribution.
    Check https://arxiv.org/pdf/2301.04104v1.pdf equation 9 for more details.

    Args:
        tensor (Tensor): tensor to encode of shape (..., batch_size, 1)
        support_range (int): range of the support of the distribution, going from -support_range to support_range
        num_buckets (int): number of buckets in the support of the distribution

    Returns:
        Tensor: tensor of shape (..., batch_size, support_size)
    """
    if tensor.shape == torch.Size([]):
        tensor = tensor.unsqueeze(0)
    if num_buckets is None:
        num_buckets = support_range * 2 + 1
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    tensor = tensor.clip(-support_range, support_range)
    buckets = torch.linspace(-support_range, support_range, num_buckets, device=tensor.device)
    bucket_size = buckets[1] - buckets[0] if len(buckets) > 1 else 1.0

    right_idxs = torch.bucketize(tensor, buckets)
    left_idxs = (right_idxs - 1).clip(min=0)

    two_hot = torch.zeros(tensor.shape[:-1] + (num_buckets,), device=tensor.device)
    left_value = torch.abs(buckets[right_idxs] - tensor) / bucket_size
    right_value = 1 - left_value
    two_hot.scatter_add_(-1, left_idxs, left_value)
    two_hot.scatter_add_(-1, right_idxs, right_value)

    return two_hot
```

DreamerV3 equation 9 (two-hot categorical encoding of a scalar). Discretises the support `[-support_range, support_range]` into `num_buckets` equally spaced bins (default `2 · range + 1`, must be odd so zero lands on a bucket centre). For each scalar `x`, the encoder finds the two adjacent buckets `b_left ≤ x ≤ b_right` and places mass proportional to the linear-interpolation weights at those two indices — the two weights sum to 1, so the result is a probability distribution. Scatter-add is used so the same scalar handled at the bucket boundary (right and left indices coincide) accumulates correctly. The clip-to-range step ensures out-of-range scalars saturate at the boundary buckets rather than throw. Output shape: `(..., num_buckets)` (the trailing scalar dim is replaced by the bucket dim).

---

## Line 191 — `two_hot_decoder`

```python
def two_hot_decoder(tensor: torch.Tensor, support_range: int) -> torch.Tensor:
    """Decode a tensor representing a two-hot vector as a tensor of floating point numbers.

    Args:
        tensor (Tensor): tensor to decode of shape (..., batch_size, support_size)
        support_range (int): range of the support of the values, going from -support_range to support_range

    Returns:
        Tensor: tensor of shape (..., batch_size, 1)
    """
    num_buckets = tensor.shape[-1]
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    support = torch.linspace(-support_range, support_range, num_buckets).to(tensor.device)
    return torch.sum(tensor * support, dim=-1, keepdim=True)
```

Inverse of `two_hot_encoder`: dot-products the bucket-probability vector with the bucket-centre values to recover the expected scalar. Works equally well on a true two-hot vector (recovering `x` exactly when `x` lies on a bin boundary, otherwise the linear interpolation of the two bucket centres) and on a softmax-over-buckets distribution (recovering `E[x]` under that distribution). The DreamerV3 critic emits a softmax over buckets and this decoder converts it to the scalar value estimate.

---

## Line 209 — `print_config`

```python
@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = ("algo", "buffer", "checkpoint", "env", "fabric", "metric"),
    resolve: bool = True,
    cfg_save_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config: Configuration composed by Hydra.
        fields: Determines which main fields from config will
            be printed and in what order.
        resolve: Whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if cfg_save_path is not None:
        with open(os.path.join(os.getcwd(), "config_tree.txt"), "w") as fp:
            rich.print(tree, file=fp)
```

Pretty-prints the Hydra-composed config as a Rich tree, one branch per top-level field in the canonical sheeprl order (`algo`, `buffer`, `checkpoint`, `env`, `fabric`, `metric`). Each sub-config is rendered as YAML with syntax highlighting via `rich.syntax.Syntax`. `@rank_zero_only` suppresses the print on all non-rank-0 processes under distributed training. When `cfg_save_path` is non-None, a plain-text version of the tree is also dumped to `<cwd>/config_tree.txt` for log archival. Note: `OmegaConf.to_yaml(..., resolve=resolve)` controls whether `${...}` interpolations are expanded — defaulting to True means the printed config matches the resolved runtime config.

---

## Line 240 — `unwrap_fabric`

```python
def unwrap_fabric(model: _FabricModule | nn.Module) -> nn.Module:
    """Recursively unwrap the model from _FabricModule. This method returns a deep copy of the model.

    Args:
        model (_FabricModule | nn.Module): the model to unwrap.

    Returns:
        nn.Module: the unwrapped model.
    """
    model = copy.deepcopy(getattr(model, "module", model))
    for name, child in model.named_children():
        setattr(model, name, unwrap_fabric(child))
    return model
```

Peels Lightning Fabric's `_FabricModule` wrapper off the model so the underlying `nn.Module` can be checkpointed / inspected as plain torch. `getattr(model, "module", model)` returns the wrapped `.module` attribute if present, otherwise the model itself — so the helper is idempotent. Recursion walks the named-children tree so nested wrapped modules are unwrapped at every level. `copy.deepcopy` at each level returns a clone, leaving Fabric's live training model untouched. Used at checkpoint-write time to get a clean state-dict.

---

## Line 255 — `save_configs`

```python
def save_configs(cfg: dotdict, log_dir: str):
    OmegaConf.save(cfg.as_dict(), os.path.join(log_dir, "config.yaml"), resolve=True)
```

Writes the resolved Hydra config to `<log_dir>/config.yaml` for run reproducibility. Calls `cfg.as_dict()` first to peel off all the `dotdict` wrappers (OmegaConf's serialiser needs plain `dict`s). The `resolve=True` argument expands any `${...}` interpolations so the saved file is self-contained — the run can be re-instantiated without needing the original `defaults:` chain. This is invoked once at the start of every training run.

---

## Line 259 — `class Ratio`

```python
class Ratio:
    """Directly taken from Hafner et al. (2023) implementation:
    https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26
    """
```

The replay-ratio scheduler that drives DreamerV3's "train K gradient steps per env step" loop semantics. Ported verbatim from Hafner's reference implementation. Critical for matching paper-canonical training dynamics: under-firing gradient updates yields a slow learner, over-firing yields the staleness pathologies the DreamerV3 paper specifically warns about. The class maintains internal state (`_prev`, `_pretrain_steps`) so the schedule is **stateful across calls**, which is why it implements `state_dict` / `load_state_dict` for checkpoint round-trip.

---

## Line 264 — `Ratio.__init__`

```python
    def __init__(self, ratio: float, pretrain_steps: int = 0):
        if pretrain_steps < 0:
            raise ValueError(f"'pretrain_steps' must be non-negative, got {pretrain_steps}")
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._pretrain_steps = pretrain_steps
        self._ratio = ratio
        self._prev = None
```

Stores the target gradient-steps-per-env-step `ratio` and optional `pretrain_steps` budget (extra updates fired during the warm-up phase). Both arguments are validated as non-negative. `_prev = None` is the sentinel signalling "first call" — on the first `__call__`, the scheduler treats the entire elapsed step count as new and may fire the pretrain budget. After the first call, `_prev` tracks the policy-step index at which the most recent batch of gradient steps was triggered.

---

## Line 273 — `Ratio.__call__`

```python
    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            repeats = int(step * self._ratio)
            if self._pretrain_steps > 0:
                if step < self._pretrain_steps:
                    warnings.warn(
                        "The number of pretrain steps is greater than the number of current steps. This could lead to "
                        f"a higher ratio than the one specified ({self._ratio}). Setting the 'pretrain_steps' equal to "
                        "the number of current steps."
                    )
                    self._pretrain_steps = step
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats
```

The arithmetic core. Given the current policy step, returns the integer number of gradient updates to fire **right now**. Branches:
- `ratio == 0` → never train (e.g. for pure data-collection runs).
- First call (`_prev is None`) → set anchor, and if `pretrain_steps > 0` fire `pretrain_steps · ratio` updates (clamped to `step · ratio` if the warm-up exceeds the elapsed steps, with a warning).
- Subsequent calls → fire `int((step - _prev) · ratio)` updates and advance the anchor `_prev` by exactly `repeats / ratio`. This fractional bookkeeping is what makes non-integer ratios (e.g. `ratio=0.5` → one update every two env steps; `ratio=2.0` → two updates per env step) accumulate correctly without drift over long runs. Returning `int(...)` truncates toward zero, so leftover fractional debt is carried into the next call via the `_prev += repeats / _ratio` accumulator.

---

## Line 293 — `Ratio.state_dict`

```python
    def state_dict(self) -> Dict[str, Any]:
        return {"_ratio": self._ratio, "_prev": self._prev, "_pretrain_steps": self._pretrain_steps}
```

Serialises the scheduler's internal state so a checkpoint can resume training with the correct fractional-debt accumulator. Returning the dict directly (rather than picking a sub-set) means any future field additions to `__init__` must also be added here to stay round-trippable.

---

## Line 296 — `Ratio.load_state_dict`

```python
    def load_state_dict(self, state_dict: Mapping[str, Any]):
        self._ratio = state_dict["_ratio"]
        self._prev = state_dict["_prev"]
        self._pretrain_steps = state_dict["_pretrain_steps"]
        return self
```

Restores all three state fields from a `state_dict` (uses `Mapping` so any dict-like works). Returns `self` to allow `ratio = Ratio(...).load_state_dict(ckpt["ratio"])` one-liners. Note that **all three fields are restored** including `_ratio` and `_pretrain_steps` — so a checkpoint resume cannot silently change the schedule, you must construct a new `Ratio` and load to inherit the on-disk values.

---

## Line 304 — `safetanh`

```python
# https://github.com/pytorch/rl/blob/824f6d192e88c115790cf046e4df416ce2d7aaf6/torchrl/modules/distributions/utils.py#L156
def safetanh(x, eps):
    lim = 1.0 - eps
    y = x.tanh()
    return y.clamp(-lim, lim)
```

Numerically safe tanh that clamps the output to `[-(1-eps), 1-eps]` rather than letting it saturate at `±1`. Saturated tanh is the killer for tanh-squashed Gaussian policies because the subsequent atanh would return `±∞`; the clamp keeps the inverse well-defined. Ported from torchrl. Used inside continuous-action policy heads where the action distribution is `tanh(N(μ, σ))`.

---

## Line 311 — `safeatanh`

```python
# https://github.com/pytorch/rl/blob/824f6d192e88c115790cf046e4df416ce2d7aaf6/torchrl/modules/distributions/utils.py#L161
def safeatanh(y, eps):
    lim = 1.0 - eps
    return y.clamp(-lim, lim).atanh()
```

Inverse of `safetanh`: clamps the input to `[-(1-eps), 1-eps]` **before** taking `atanh`, so the result stays finite even if upstream code passes a value at the boundary. Together with `safetanh` this gives a numerically stable round-trip for tanh-squashed policies — crucial when computing log-probabilities with the change-of-variables term `log|det dy/dx|` that involves `atanh`. Ported from torchrl.
