---
title: "Sheeprl Reference: agent.py"
source: tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `agent.py`

> **Source**: `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py` — 1236 lines.
> **Purpose** (one-line): All networks for DreamerV3: RSSM (recurrent + transition + representation models), multi-modality encoder/decoder, reward + continue heads, Actor, Critic, target critic, PlayerDV3 env-interaction wrapper, and `build_agent` factory.
> **Imports from elsewhere in this index**: [`models.md`](models.md) (`MLP`, `CNN`, `DeCNN`, `MultiEncoder`, `MultiDecoder`, `LayerNorm`, `LayerNormChannelLast`, `LayerNormGRUCell`), [`distribution.md`](distribution.md) (`TwoHotEncodingDistribution`, `SymlogDistribution`, `MSEDistribution`, `BernoulliSafeMode` — used downstream in `loss.py` and via the reward/critic bins config), [`utils.md`](utils.md) (`Moments`, `compute_lambda_values` — consumed by the training loop, not by this file directly), and [`utils_core.md`](utils_core.md) (`symlog` helper). The `WorldModel` container class is inherited from `sheeprl/algos/dreamer_v2/agent.py` (a sibling DreamerV2 file).

---

## Table of Contents

- **Lines 1–39** — Imports and module-level setup.
- **Line 42** — `class CNNEncoder` — the 4-stage Conv2d image encoder.
- **Line 64** — `CNNEncoder.__init__` — build the conv stack with optional channel-last LayerNorm.
- **Line 95** — `CNNEncoder.forward` — concat image keys on channel dim, flatten to a vector.
- **Line 100** — `class MLPEncoder` — vector encoder with symlog input squashing.
- **Line 123** — `MLPEncoder.__init__` — build N-layer MLP with LayerNorm.
- **Line 149** — `MLPEncoder.forward` — symlog inputs then MLP.
- **Line 154** — `class CNNDecoder` — inverse of `CNNEncoder` (4 stages of ConvTranspose2d).
- **Line 180** — `CNNDecoder.__init__` — Linear projection → unflatten 4×4 → DeCNN stack.
- **Line 224** — `CNNDecoder.forward` — DeCNN, split channels per key.
- **Line 229** — `class MLPDecoder` — inverse of `MLPEncoder` (shared MLP + per-key Linear heads).
- **Line 251** — `MLPDecoder.__init__` — shared trunk + `nn.ModuleList` of output heads.
- **Line 276** — `MLPDecoder.forward` — apply trunk then each head.
- **Line 281** — `class RecurrentModel` — MLP + `LayerNormGRUCell` deterministic state update.
- **Line 299** — `RecurrentModel.__init__` — one-layer MLP feeding a `LayerNormGRUCell`.
- **Line 328** — `RecurrentModel.forward` — MLP(input) → GRUCell(feat, h).
- **Line 344** — `class RSSM` — the recurrent state-space model (h, z) world-model backbone.
- **Line 365** — `RSSM.__init__` — wire recurrent + representation + transition models; optional learnable initial recurrent state.
- **Line 391** — `RSSM.get_initial_states` — tanh-of-learned-init recurrent + prior-from-init posterior.
- **Line 396** — `RSSM.dynamic` — one teacher-forced step: reset on `is_first`, run recurrent → prior → posterior.
- **Line 437** — `RSSM._uniform_mix` — inject `unimix` uniform mass into categorical logits.
- **Line 451** — `RSSM._representation` — posterior network (h, embedded_obs) → categorical logits.
- **Line 467** — `RSSM._transition` — prior network (h) → categorical logits.
- **Line 482** — `RSSM.imagination` — one-step latent rollout using prior only (no embedded obs).
- **Line 501** — `class DecoupledRSSM` — variant where the posterior ignores the recurrent state.
- **Line 522** — `DecoupledRSSM.__init__` — forwards arguments to `RSSM.__init__`.
- **Line 542** — `DecoupledRSSM.dynamic` — same as `RSSM.dynamic` but does not compute the posterior here.
- **Line 582** — `DecoupledRSSM._representation` — posterior takes only embedded obs as input.
- **Line 596** — `class PlayerDV3` — env-interaction wrapper: holds carry state, samples actions.
- **Line 617** — `PlayerDV3.__init__` — store encoder, RSSM, actor, dims, device.
- **Line 644** — `PlayerDV3.init_states` — reset recurrent + stochastic state + last action for given env indices.
- **Line 661** — `PlayerDV3.get_actions` — encode obs, advance RSSM by one step, sample/greedy action.
- **Line 694** — `class Actor` — policy network with discrete or continuous action distributions.
- **Line 729** — `Actor.__init__` — shared MLP trunk + per-action heads (Linear); decide distribution type.
- **Line 783** — `Actor.forward` — build distribution(s) from logits, rsample (or argmax-of-100 if greedy), action-clip.
- **Line 839** — `Actor._uniform_mix` — `unimix` injection for discrete action logits.
- **Line 848** — `class MinedojoActor` — `Actor` subclass with Minedojo action-mask handling.
- **Line 849** — `MinedojoActor.__init__` — forwards all args to `Actor.__init__` (no `max_std`).
- **Line 881** — `MinedojoActor.forward` — apply hierarchical action masks based on previously sampled functional action.
- **Line 935** — `build_agent` — factory that wires every network, applies Hafner init, wraps with Fabric, ties weights to the player.

---

## Lines 1–39 — Imports and module-level setup

```python
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
)
from torch.distributions.utils import probs_to_logits

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.models.models import (
    CNN,
    MLP,
    DeCNN,
    LayerNorm,
    LayerNormChannelLast,
    LayerNormGRUCell,
    MultiDecoder,
    MultiEncoder,
)
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.model import ModuleType, cnn_forward
from sheeprl.utils.utils import symlog
```

**What this is**: PyTorch + Lightning-Fabric + Hydra + Gymnasium are the framework stack. `WorldModel` (the container holding encoder + RSSM + obs/reward/continue heads) is imported from the DreamerV2 file and reused unchanged. `compute_stochastic_state` (the categorical-with-straight-through sampler) and `init_weights` / `uniform_init_weights` (Hafner-style initialization) are reused from DreamerV2/V3 utility files. All network primitives ([`MLP`](models.md), [`CNN`](models.md), [`DeCNN`](models.md), [`LayerNormGRUCell`](models.md), [`MultiEncoder`](models.md), [`MultiDecoder`](models.md), [`LayerNorm`](models.md), [`LayerNormChannelLast`](models.md)) come from `sheeprl/models/models.py`. The `torch.distributions` imports power the actor's continuous and discrete action distributions. `symlog` is the signed-log squashing function applied to vector observations.

---

## Line 42 — `class CNNEncoder`

```python
class CNNEncoder(nn.Module):
    """The Dreamer-V3 image encoder. This is composed of 4 `nn.Conv2d` with
    kernel_size=3, stride=2 and padding=1. No bias is used if a `nn.LayerNorm`
    is used after the convolution. This 4-stages model assumes that the image
    is a 64x64 and it ends with a resolution of 4x4. If more than one image is to be encoded, then those will
    be concatenated on the channel dimension and fed to the encoder.

    Args:
        keys (Sequence[str]): the keys representing the image observations to encode.
        input_channels (Sequence[int]): the input channels, one for each image observation to encode.
        image_size (Tuple[int, int]): the image size as (Height,Width).
        channels_multiplier (int): the multiplier for the output channels. Given the 4 stages, the 4 output channels
            will be [1, 2, 4, 8] * `channels_multiplier`.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNormChannelLast.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        activation (ModuleType, optional): the activation function.
            Defaults to nn.SiLU.
        stages (int, optional): how many stages for the CNN.
    """
```

**What it does**: Image observation encoder. 4 conv stages with stride 2 halve resolution each step (64→32→16→8→4). Channels double each stage. If multiple image keys are present they are concatenated along the channel dimension before encoding. Note that the docstring says `kernel_size=3` but the actual code uses `kernel_size=4` — a minor doc/code mismatch in the source. No bias when followed by LayerNorm (the LayerNorm absorbs the affine shift). Channel-last LayerNorm is used so it normalises over the channel dim correctly after Conv2d's `[B,C,H,W]` layout.

---

## Line 64 — `CNNEncoder.__init__`

```python
    def __init__(
        self,
        keys: Sequence[str],
        input_channels: Sequence[int],
        image_size: Tuple[int, int],
        channels_multiplier: int,
        layer_norm_cls: Callable[..., nn.Module] = LayerNormChannelLast,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        activation: ModuleType = nn.SiLU,
        stages: int = 4,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = (sum(input_channels), *image_size)
        self.model = nn.Sequential(
            CNN(
                input_channels=self.input_dim[0],
                hidden_channels=(torch.tensor([2**i for i in range(stages)]) * channels_multiplier).tolist(),
                cnn_layer=nn.Conv2d,
                layer_args={"kernel_size": 4, "stride": 2, "padding": 1, "bias": layer_norm_cls == nn.Identity},
                activation=activation,
                norm_layer=[layer_norm_cls] * stages,
                norm_args=[
                    {**layer_norm_kw, "normalized_shape": (2**i) * channels_multiplier} for i in range(stages)
                ],
            ),
            nn.Flatten(-3, -1),
        )
        with torch.no_grad():
            self.output_dim = self.model(torch.zeros(1, *self.input_dim)).shape[-1]
```

**What it does**: Builds the convolutional stack via the generic [`CNN`](models.md) builder. `hidden_channels = [1,2,4,8]*multiplier` (with default `multiplier=32` → `[32,64,128,256]`). Bias is only included when there is no LayerNorm. A trailing `Flatten` collapses spatial dims into one feature vector. The output dimension is computed *empirically* by running a zero tensor through the model so callers don't have to hard-code the 4×4×(8·M) figure. `stages` is computed by `build_agent` as `log2(screen_size) - log2(4)` so 64×64 → 4 stages.

---

## Line 95 — `CNNEncoder.forward`

```python
    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-3)  # channels dimension
        return cnn_forward(self.model, x, x.shape[-3:], (-1,))
```

**What it does**: Concatenate image observations from every CNN key along the channel dim (`dim=-3` is C in `[…, C, H, W]`). `cnn_forward` is a sheeprl helper that flattens any leading batch/time dims, runs the CNN, and restores them — so the encoder works seamlessly with shape `[B, C, H, W]` and `[T, B, C, H, W]`.

---

## Line 100 — `class MLPEncoder`

```python
class MLPEncoder(nn.Module):
    """The Dreamer-V3 vector encoder. This is composed of N `nn.Linear` layers, where
    N is specified by `mlp_layers`. No bias is used if a `nn.LayerNorm` is used after the linear layer.
    If more than one vector is to be encoded, then those will concatenated on the last
    dimension before being fed to the encoder.

    Args:
        keys (Sequence[str]): the keys representing the vector observations to encode.
        input_dims (Sequence[int]): the dimensions of every vector to encode.
        mlp_layers (int, optional): how many mlp layers.
            Defaults to 4.
        dense_units (int, optional): the dimension of every mlp.
            Defaults to 512.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        activation (ModuleType, optional): the activation function after every layer.
            Defaults to nn.SiLU.
        symlog_inputs (bool, optional): whether to squash the input with the symlog function.
            Defaults to True.
    """
```

**What it does**: Vector-observation encoder. N (default 4) linear layers, each followed by LayerNorm and SiLU. Multiple vector keys are concatenated along the last dim and passed through one shared trunk. The hallmark of DreamerV3 here is `symlog_inputs=True` — input vectors get `sign(x)·log(1+|x|)` squashing before entering the MLP, which stabilises the encoder against unbounded reward-/value-scale observations.

---

## Line 123 — `MLPEncoder.__init__`

```python
    def __init__(
        self,
        keys: Sequence[str],
        input_dims: Sequence[int],
        mlp_layers: int = 4,
        dense_units: int = 512,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        activation: ModuleType = nn.SiLU,
        symlog_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = sum(input_dims)
        self.model = MLP(
            self.input_dim,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.output_dim = dense_units
        self.symlog_inputs = symlog_inputs
```

**What it does**: Build the [`MLP`](models.md) trunk with `output_dim=None` (so the last layer is just dense+norm+activation, no projection). Records `output_dim` for the downstream consumer (representation model). Saves the `symlog_inputs` flag for `forward`.

---

## Line 149 — `MLPEncoder.forward`

```python
    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([symlog(obs[k]) if self.symlog_inputs else obs[k] for k in self.keys], -1)
        return self.model(x)
```

**What it does**: Optionally apply `symlog` per-key, concatenate on the last (feature) dim, run the MLP. Symlog is applied **per-observation** before concatenation so each vector key gets its own squashing.

---

## Line 154 — `class CNNDecoder`

```python
class CNNDecoder(nn.Module):
    """The exact inverse of the `CNNEncoder` class. It assumes an initial resolution
    of 4x4, and in 4 stages reconstructs the observation image to 64x64. If multiple
    images are to be reconstructed, then it will create a dictionary with an entry
    for every reconstructed image. No bias is used if a `nn.LayerNorm` is used after
    the `nn.Conv2dTranspose` layer.

    Args:
        keys (Sequence[str]): the keys of the image observation to be reconstructed.
        output_channels (Sequence[int]): the output channels, one for every image observation.
        channels_multiplier (int): the channels multiplier, same for the encoder network.
        latent_state_size (int): the size of the latent state. Before applying the decoder,
            a `nn.Linear` layer is used to project the latent state to a feature vector
            of dimension [8 * `channels_multiplier`, 4, 4].
        cnn_encoder_output_dim (int): the output of the image encoder. It should be equal to
            8 * `channels_multiplier` * 4 * 4.
        image_size (Tuple[int, int]): the final image size.
        activation (nn.Module, optional): the activation function.
            Defaults to nn.SiLU.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNormChannelLast.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        stages (int): how many stages in the CNN decoder.
    """
```

**What it does**: Symmetric inverse of `CNNEncoder`. Takes the latent state `(h, z)` flattened, projects it back to a 4×4 feature map with `8·M` channels, then 4 ConvTranspose2d stages double the resolution back to 64×64. Multiple image keys are stored as a dict on output, splitting along the channel dim. The final ConvTranspose layer has no LayerNorm and no activation — it produces the raw reconstruction (consumed by `MSEDistribution` or `SymlogDistribution` in [`distribution.md`](distribution.md)).

---

## Line 180 — `CNNDecoder.__init__`

```python
    def __init__(
        self,
        keys: Sequence[str],
        output_channels: Sequence[int],
        channels_multiplier: int,
        latent_state_size: int,
        cnn_encoder_output_dim: int,
        image_size: Tuple[int, int],
        activation: nn.Module = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = LayerNormChannelLast,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        stages: int = 4,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.output_channels = output_channels
        self.cnn_encoder_output_dim = cnn_encoder_output_dim
        self.image_size = image_size
        self.output_dim = (sum(output_channels), *image_size)
        self.model = nn.Sequential(
            nn.Linear(latent_state_size, cnn_encoder_output_dim),
            nn.Unflatten(1, (-1, 4, 4)),
            DeCNN(
                input_channels=(2 ** (stages - 1)) * channels_multiplier,
                hidden_channels=(
                    torch.tensor([2**i for i in reversed(range(stages - 1))]) * channels_multiplier
                ).tolist()
                + [self.output_dim[0]],
                cnn_layer=nn.ConvTranspose2d,
                layer_args=[
                    {"kernel_size": 4, "stride": 2, "padding": 1, "bias": layer_norm_cls == nn.Identity}
                    for _ in range(stages - 1)
                ]
                + [{"kernel_size": 4, "stride": 2, "padding": 1}],
                activation=[activation for _ in range(stages - 1)] + [None],
                norm_layer=[layer_norm_cls for _ in range(stages - 1)] + [None],
                norm_args=[
                    {**layer_norm_kw, "normalized_shape": (2 ** (stages - i - 2)) * channels_multiplier}
                    for i in range(stages - 1)
                ]
                + [None],
            ),
        )
```

**What it does**: `Linear` projects `[B, latent_state_size]` → `[B, 8·M·4·4]`, `Unflatten` reshapes to `[B, 8·M, 4, 4]`, then `DeCNN` runs `stages-1` ConvTranspose2d blocks (each with LayerNorm + SiLU) followed by a final raw ConvTranspose2d to `output_channels`. Channel counts mirror the encoder reversed: `[4,2,1]·M` then final output channels. The final block omits norm and activation so the reconstruction is unconstrained.

---

## Line 224 — `CNNDecoder.forward`

```python
    def forward(self, latent_states: Tensor) -> Dict[str, Tensor]:
        cnn_out = cnn_forward(self.model, latent_states, (latent_states.shape[-1],), self.output_dim)
        return {k: rec_obs for k, rec_obs in zip(self.keys, torch.split(cnn_out, self.output_channels, -3))}
```

**What it does**: Runs the decoder (with batch-dim flattening through `cnn_forward`) and splits the multi-channel output back into per-key tensors via `torch.split(..., output_channels, dim=-3)`. Returns a dict matching the obs schema, e.g. `{"rgb": tensor[..., 3, 64, 64]}`.

---

## Line 229 — `class MLPDecoder`

```python
class MLPDecoder(nn.Module):
    """The exact inverse of the MLPEncoder. This is composed of N `nn.Linear` layers, where
    N is specified by `mlp_layers`. No bias is used if a `nn.LayerNorm` is used after the linear layer.
    If more than one vector is to be decoded, then it will create a dictionary with an entry
    for every reconstructed vector.

    Args:
        keys (Sequence[str]): the keys representing the vector observations to decode.
        output_dims (Sequence[int]): the dimensions of every vector to decode.
        latent_state_size (int): the dimension of the latent state.
        mlp_layers (int, optional): how many mlp layers.
            Defaults to 4.
        dense_units (int, optional): the dimension of every mlp.
            Defaults to 512.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        activation (ModuleType, optional): the activation function after every layer.
            Defaults to nn.SiLU.
    """
```

**What it does**: Vector decoder. One shared MLP trunk + a list of per-key linear heads (one head per reconstructed vector). Returns a dict matching the obs schema. Used downstream by `SymlogDistribution` to reconstruct symlog-encoded vector observations.

---

## Line 251 — `MLPDecoder.__init__`

```python
    def __init__(
        self,
        keys: Sequence[str],
        output_dims: Sequence[str],
        latent_state_size: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        activation: ModuleType = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        super().__init__()
        self.output_dims = output_dims
        self.keys = keys
        self.model = MLP(
            latent_state_size,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])
```

**What it does**: Build a shared trunk ending in `dense_units`-dim features, then one `nn.Linear(dense_units, out_dim)` head per output key. Heads are stored in `nn.ModuleList` so they get registered for state-dict and gradient propagation. The Hafner-init pass in `build_agent` later applies `uniform_init_weights(1.0)` to `self.heads`.

---

## Line 276 — `MLPDecoder.forward`

```python
    def forward(self, latent_states: Tensor) -> Dict[str, Tensor]:
        x = self.model(latent_states)
        return {k: h(x) for k, h in zip(self.keys, self.heads)}
```

**What it does**: Trunk produces shared features; each head produces its own reconstruction tensor. The output dict aligns with the obs space's vector keys.

---

## Line 281 — `class RecurrentModel`

```python
class RecurrentModel(nn.Module):
    """Recurrent model for the model-base Dreamer-V3 agent.
    This implementation uses the `sheeprl.models.models.LayerNormGRUCell`, which combines
    the standard GRUCell from PyTorch with the `nn.LayerNorm`, where the normalization is applied
    right after having computed the projection from the input to the weight space.

    Args:
        input_size (int): the input size of the model.
        dense_units (int): the number of dense units.
        recurrent_state_size (int): the size of the recurrent state.
        activation_fn (nn.Module): the activation function.
            Default to SiLU.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
    """
```

**What it does**: Computes the deterministic part `h_t` of the RSSM state. Input is `concat(z_{t-1}, a_{t-1})`. Internally runs a one-layer MLP (with LayerNorm + SiLU) projecting input to `dense_units`, then feeds that into a [`LayerNormGRUCell`](models.md) updating the recurrent state. The `LayerNormGRUCell` is the DreamerV3-specific bit: standard GRU cell but with LayerNorm on the input-to-hidden projection — known cascade target #28 in the project's fix cascade.

---

## Line 299 — `RecurrentModel.__init__`

```python
    def __init__(
        self,
        input_size: int,
        recurrent_state_size: int,
        dense_units: int,
        activation_fn: nn.Module = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=[dense_units],
            activation=activation_fn,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=[layer_norm_cls],
            norm_args=[{**layer_norm_kw, "normalized_shape": dense_units}],
        )
        self.rnn = LayerNormGRUCell(
            dense_units,
            recurrent_state_size,
            bias=False,
            batch_first=False,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
        )
        self.recurrent_state_size = recurrent_state_size
```

**What it does**: Build the pre-projection MLP and the LayerNorm-GRU cell. Two key flags: `bias=False` on the GRU (LayerNorm replaces the bias term) and `batch_first=False` (sheeprl uses `[T, B, …]` time-major layout throughout). `recurrent_state_size` is saved on the module so `RSSM.__init__` can size the learnable initial state.

---

## Line 328 — `RecurrentModel.forward`

```python
    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tensor:
        """
        Compute the next recurrent state from the latent state (stochastic and recurrent states) and the actions.

        Args:
            input (Tensor): the input tensor composed by the stochastic state and the actions concatenated together.
            recurrent_state (Tensor): the previous recurrent state.

        Returns:
            the computed recurrent output and recurrent state.
        """
        feat = self.mlp(input)
        out = self.rnn(feat, recurrent_state)
        return out
```

**What it does**: One-step recurrent update `h_t = GRUCell(MLP(z_{t-1} ⊕ a_{t-1}), h_{t-1})`. The returned tensor is both the new hidden state and the "output" used by downstream consumers (prior/transition network).

---

## Line 344 — `class RSSM`

```python
class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in
            [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a
            multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multi-layer perceptron to predict the stochastic part of the latent state.
        distribution_cfg (Dict[str, Any]): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = `1 / self.discrete`.
            Defaults to 0.01.
    """
```

**What it does**: The central piece of the DreamerV3 world model. Bundles three sub-networks: (1) `recurrent_model` produces the deterministic state `h_t`; (2) `transition_model` (the *prior*) predicts `p(z_t | h_t)` — what we'd expect the stochastic state to be without seeing the observation; (3) `representation_model` (the *posterior*) computes `q(z_t | h_t, x_t)` — corrected by the actual observation embedding. The KL between posterior and prior is the world-model dynamics loss. `discrete=32` and stochastic_size=32 means `z_t` is `32×32 = 1024` one-hot categoricals — DreamerV3's signature representation. `unimix=0.01` inject 1% uniform noise to prevent collapsing categoricals from dominating training.

---

## Line 365 — `RSSM.__init__`

```python
    def __init__(
        self,
        recurrent_model: RecurrentModel | _FabricModule,
        representation_model: nn.Module | _FabricModule,
        transition_model: nn.Module | _FabricModule,
        distribution_cfg: Dict[str, Any],
        discrete: int = 32,
        unimix: float = 0.01,
        learnable_initial_recurrent_state: bool = True,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.distribution_cfg = distribution_cfg
        self.discrete = discrete
        self.unimix = unimix
        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = nn.Parameter(
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state", torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )
```

**What it does**: Stores the three sub-models and key hyperparameters. The initial recurrent state can be either a *learned* parameter (default) or a non-trainable buffer — Hafner's recipe in the original paper uses a learnable initial recurrent state, which the agent learns to bias according to a sensible reset state for the task. Note that the initial state is stored *before* a `tanh` is applied in `get_initial_states`.

---

## Line 391 — `RSSM.get_initial_states`

```python
    def get_initial_states(self, batch_shape: Sequence[int] | torch.Size) -> Tuple[Tensor, Tensor]:
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
        return initial_recurrent_state, initial_posterior
```

**What it does**: Compute the recurrent/stochastic state to use at episode boundaries (`is_first=True`). The learned initial state is tanh-squashed to `[-1, 1]` for numerical stability, then expanded to the requested batch shape (e.g. `(1, num_envs)`). For the initial *stochastic* state, the prior network is queried with the initial recurrent state and `sample_state=False` (use the mode, i.e. the soft probabilities — see `compute_stochastic_state`).

---

## Line 396 — `RSSM.dynamic`

```python
    def dynamic(
        self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor, is_first: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the prior from the recurrent output.
            Representation model: compute the posterior from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
        and [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

        Args:
            posterior (Tensor): the stochastic state computed by the representation model (posterior). It is expected
                to be of dimension `[stoch_size, self.discrete]`, which by default is `[32, 32]`.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.
            is_first (Tensor): if this is the first step in the episode.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The posterior stochastic state (Tensor): computed by the representation model
            The prior stochastic state (Tensor): computed by the transition model
            The logits of the posterior state (Tensor): computed by the transition model from the recurrent state.
            The logits of the prior state (Tensor): computed by the transition model from the recurrent state.
            from the recurrent state and the embbedded observation.
        """
        action = (1 - is_first) * action

        initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        return recurrent_state, posterior, prior, posterior_logits, prior_logits
```

**What it does**: The single dynamic-learning step. Key subtleties: (1) **soft reset on `is_first`** — actions and previous state are zeroed at episode start, and recurrent/stochastic states are replaced with the learned initial states. This is the "first-step is_first masking" that Hafner introduced in DreamerV3. (2) `posterior` is reshaped from `[…, stoch, discrete]` to `[…, stoch·discrete]` because the recurrent model expects a flat vector. (3) Three sequential calls: recurrent → transition (prior) → representation (posterior). Returns five tensors so the loss function can compute reconstruction (using posterior), reward/continue prediction (using posterior), and KL(posterior || prior). The prior is also returned for downstream imagination rollouts.

---

## Line 437 — `RSSM._uniform_mix`

```python
    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits
```

**What it does**: Inject a uniform component into the categorical distribution over stochastic states: `p ← (1-α)·p + α·(1/K)` where `α = self.unimix` (default 0.01) and `K = self.discrete` (default 32). Reshape logits between flat `[…, stoch·discrete]` and grouped `[…, stoch, discrete]` views since softmax must apply *per-categorical-group*, not over all 32·32 = 1024 entries. `probs_to_logits` converts the mixed probabilities back to logit space so downstream sampling (via `OneHotCategoricalStraightThrough` in `compute_stochastic_state`) uses the mixed distribution. Unimix is a regularizer that prevents categoricals from collapsing to delta distributions, which would zero out the gradient through the straight-through estimator.

---

## Line 451 — `RSSM._representation`

```python
    def _representation(self, recurrent_state: Tensor, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in
                [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: Tensor = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)
```

**What it does**: Posterior network `q(z_t | h_t, x_t)`. Concat the deterministic state with the encoder output and project through the representation MLP to logits. Apply unimix, then sample with straight-through (`compute_stochastic_state` returns `OneHotCategoricalStraightThrough.rsample()` reshaped to `[…, stoch·discrete]`). The posterior is *observation-conditioned* — it knows what the agent actually saw.

---

## Line 467 — `RSSM._transition`

```python
    def _transition(self, recurrent_out: Tensor, sample_state=True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.
            sampler_state (bool): whether or not to sample the stochastic state.
                Default to True

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits: Tensor = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete, sample=sample_state)
```

**What it does**: Prior network `p(z_t | h_t)`. Only takes the deterministic state — does *not* see the embedded observation. The KL between posterior and prior teaches the prior to predict what the posterior will see, even though the prior can't peek. `sample_state=False` (used by `get_initial_states`) returns the softmax probabilities (the mode of the categorical) instead of a hard sample.

---

## Line 482 — `RSSM.imagination`

```python
    def imagination(self, prior: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step imagination of the next latent state.
        It can be used several times to imagine trajectories in the latent space (Transition Model).

        Args:
            prior (Tensor): the prior state.
            recurrent_state (Tensor): the recurrent state of the recurrent model.
            actions (Tensor): the actions taken by the agent.

        Returns:
            The imagined prior state (Tuple[Tensor, Tensor]): the imagined prior state.
            The recurrent state (Tensor).
        """
        recurrent_state = self.recurrent_model(torch.cat((prior, actions), -1), recurrent_state)
        _, imagined_prior = self._transition(recurrent_state)
        return imagined_prior, recurrent_state
```

**What it does**: One step of pure-prior latent rollout — used during actor-critic training to generate the imagination horizon (default 15 steps). Unlike `dynamic`, there's no `is_first`, no embedded observation, no posterior. The actor produces `actions` greedily from `(h, z)` and the prior produces the next stochastic state. Called in a loop with the actor in the main training file's `train()` function to produce imagined trajectories that are scored by the reward model + critic.

---

## Line 501 — `class DecoupledRSSM`

```python
class DecoupledRSSM(RSSM):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in
            [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a
            multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multi-layer perceptron to predict the stochastic part of the latent state.
        distribution_cfg (Dict[str, Any]): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = `1 / self.discrete`.
            Defaults to 0.01.
    """
```

**What it does**: Optional variant where the posterior is computed from the **embedded observation alone**, decoupled from the recurrent state. This corresponds to a setup where the posterior is more of an "encoder readout" than a true Bayesian update. Selected by `cfg.algo.world_model.decoupled_rssm=true`. The hypothesis is that decoupling can prevent the posterior from over-fitting to specific `h_t` trajectories, but DreamerV3 default is the standard `RSSM`.

---

## Line 522 — `DecoupledRSSM.__init__`

```python
    def __init__(
        self,
        recurrent_model: nn.Module | _FabricModule,
        representation_model: nn.Module | _FabricModule,
        transition_model: nn.Module | _FabricModule,
        distribution_cfg: Dict[str, Any],
        discrete: int = 32,
        unimix: float = 0.01,
        learnable_initial_recurrent_state: bool = True,
    ) -> None:
        super().__init__(
            recurrent_model,
            representation_model,
            transition_model,
            distribution_cfg,
            discrete,
            unimix,
            learnable_initial_recurrent_state,
        )
```

**What it does**: Pure delegation to `RSSM.__init__`. The class only overrides `dynamic` and `_representation`; everything else (the unimix routine, `get_initial_states`, `imagination`) is inherited unchanged.

---

## Line 542 — `DecoupledRSSM.dynamic`

```python
    def dynamic(
        self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, is_first: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the prior from the recurrent output.
            Representation model: compute the posterior from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
        and [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

        Args:
            posterior (Tensor): the stochastic state computed by the representation model (posterior). It is expected
                to be of dimension `[stoch_size, self.discrete]`, which by default is `[32, 32]`.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.
            is_first (Tensor): if this is the first step in the episode.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The posterior stochastic state (Tensor): computed by the representation model
            The prior stochastic state (Tensor): computed by the transition model
            The logits of the posterior state (Tensor): computed by the transition model from the recurrent state.
            The logits of the prior state (Tensor): computed by the transition model from the recurrent state.
            from the recurrent state and the embbedded observation.
        """
        action = (1 - is_first) * action

        initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        return recurrent_state, prior, prior_logits
```

**What it does**: Decoupled variant of the dynamic step. Same is-first resetting and recurrent update, but only returns the *prior* — the posterior is not computed here. The main training loop computes the posterior separately by calling `_representation(embedded_obs)` (without `recurrent_state`). The return tuple has 3 entries instead of 5; callers in `dreamer_v3.py` branch on `decoupled_rssm` to handle this difference.

---

## Line 582 — `DecoupledRSSM._representation`

```python
    def _representation(self, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: Tensor = self.representation_model(embedded_obs)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)
```

**What it does**: Posterior network with only `embedded_obs` as input — no `recurrent_state` concatenation. The representation_model is sized in `build_agent` to take only `encoder.output_dim` when `decoupled_rssm=True`. Output is the same as `RSSM._representation`: (logits, sampled stochastic state).

---

## Line 596 — `class PlayerDV3`

```python
class PlayerDV3(nn.Module):
    """
    The model of the Dreamer_v3 player.

    Args:
        encoder (MultiEncoder): the encoder.
        rssm (RSSM | DecoupledRSSM): the RSSM model.
        actor (_FabricModule): the actor.
        actions_dim (Sequence[int]): the dimension of the actions.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        transition_model (_FabricModule): the transition model.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
        actor_type (str, optional): which actor the player is using ('task' or 'exploration').
            Default to None.
        decoupled_rssm (bool, optional): whether to use the DecoupledRSSM model.
    """
```

**What it does**: Env-interaction wrapper. Unlike the training-time RSSM which is called over batched sequences, the player holds the **carry state** (`recurrent_state`, `stochastic_state`, last `actions`) across env steps. It's only used at rollout time (collecting trajectories that go into the replay buffer), not during gradient updates. Wraps a copy of the encoder, RSSM, and actor whose weights are tied to the trainable ones via parameter-aliasing in `build_agent`.

---

## Line 617 — `PlayerDV3.__init__`

```python
    def __init__(
        self,
        encoder: MultiEncoder | _FabricModule,
        rssm: RSSM | DecoupledRSSM,
        actor: Actor | MinedojoActor | _FabricModule,
        actions_dim: Sequence[int],
        num_envs: int,
        stochastic_size: int,
        recurrent_state_size: int,
        device: str | torch.device,
        discrete_size: int = 32,
        actor_type: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.device = device
        self.discrete_size = discrete_size
        self.actor_type = actor_type
        self.decoupled_rssm = isinstance(rssm, DecoupledRSSM)
```

**What it does**: Stores references to encoder + RSSM + actor (these are *copied* in `build_agent`, then re-tied via parameter aliasing). Caches dimensions for state shaping. `decoupled_rssm` flag triggers the alternative posterior call in `get_actions`.

---

## Line 644 — `PlayerDV3.init_states`

```python
    @torch.no_grad()
    def init_states(self, reset_envs: Optional[Sequence[int]] = None) -> None:
        """Initialize the states and the actions for the ended environments.

        Args:
            reset_envs (Optional[Sequence[int]], optional): which environments' states to reset.
                If None, then all environments' states are reset.
                Defaults to None.
        """
        if reset_envs is None or len(reset_envs) == 0:
            self.actions = torch.zeros(1, self.num_envs, np.sum(self.actions_dim), device=self.device)
            self.recurrent_state, stochastic_state = self.rssm.get_initial_states((1, self.num_envs))
            self.stochastic_state = stochastic_state.reshape(1, self.num_envs, -1)
        else:
            self.actions[:, reset_envs] = torch.zeros_like(self.actions[:, reset_envs])
            self.recurrent_state[:, reset_envs], stochastic_state = self.rssm.get_initial_states((1, len(reset_envs)))
            self.stochastic_state[:, reset_envs] = stochastic_state.reshape(1, len(reset_envs), -1)
```

**What it does**: Reset the carry state for vector-env workers. Either all envs (when `reset_envs=None`, e.g. at startup) or a subset (when individual envs finish episodes mid-rollout). The leading `1` dim is the time dimension (one step). Stochastic state is reshaped from `[1, B, stoch, discrete]` to `[1, B, stoch·discrete]` to match the flat representation used downstream. `@torch.no_grad()` because rollout is inference-only.

---

## Line 661 — `PlayerDV3.get_actions`

```python
    def get_actions(
        self,
        obs: Dict[str, Tensor],
        greedy: bool = False,
        mask: Optional[Dict[str, Tensor]] = None,
    ) -> Sequence[Tensor]:
        """
        Return the greedy actions.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            greedy (bool): whether or not to sample the actions.
                Default to False.

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs)
        self.recurrent_state = self.rssm.recurrent_model(
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        if self.decoupled_rssm:
            _, self.stochastic_state = self.rssm._representation(embedded_obs)
        else:
            _, self.stochastic_state = self.rssm._representation(self.recurrent_state, embedded_obs)
        self.stochastic_state = self.stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions
```

**What it does**: One env-interaction step. (1) Encode the current observation. (2) Advance the recurrent state using the *previous* stochastic state and *previous* action (the carry state). (3) Compute the posterior from the new recurrent state and the embedded obs (or just the embedded obs in the decoupled variant) — this becomes the new stochastic state. (4) Flatten `[stoch, discrete]` → `[stoch·discrete]`. (5) Run the actor on `concat(z_t, h_t)`. (6) Cache the chosen action for the next call. Returns the list of action tensors (one per action head — see `Actor`).

---

## Line 694 — `class Actor`

```python
class Actor(nn.Module):
    """
    The wrapper class of the Dreamer_v2 Actor model.

    Args:
        latent_state_size (int): the dimension of the latent state (stochastic size + recurrent_state_size).
        actions_dim (Sequence[int]): the dimension in output of the actor.
            The number of actions if continuous, the dimension of the action if discrete.
        is_continuous (bool): whether or not the actions are continuous.
        distribution_cfg (Dict[str, Any]): The configs of the distributions.
        init_std (float): the amount to sum to the standard deviation.
            Default to 0.0.
        min_std (float): the minimum standard deviation for the actions.
            Default to 1.0.
        max_std (float): the maximum standard deviation for the actions.
            Default to 1.0.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 1024.
        activation (int): the activation function to apply after the dense layers.
            Default to nn.SiLU.
        mlp_layers (int): the number of dense layers.
            Default to 5.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over actions, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`,
            where `unif = `1 / self.discrete`.
            Defaults to 0.01.
        action_clip (float): the action clip parameter.
            Default to 1.0.
    """
```

**What it does**: Policy network mapping `(h_t, z_t)` to an action distribution. Handles four distribution modes: `discrete` (one `OneHotCategoricalStraightThrough` per action head), `scaled_normal` (continuous default), `normal`, and `tanh_normal` (with `TanhTransform`). For `auto` mode, picks `scaled_normal` if continuous, `discrete` otherwise. The unimix injection (1% by default) is applied to discrete action logits for the same reason as in RSSM — preventing categorical collapse.

---

## Line 729 — `Actor.__init__`

```python
    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        is_continuous: bool,
        distribution_cfg: Dict[str, Any],
        init_std: float = 0.0,
        min_std: float = 1.0,
        max_std: float = 1.0,
        dense_units: int = 1024,
        activation: nn.Module = nn.SiLU,
        mlp_layers: int = 5,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        unimix: float = 0.01,
        action_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.distribution_cfg = distribution_cfg
        self.distribution = distribution_cfg.get("type", "auto").lower()
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete", "scaled_normal"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal`, `tanh_normal` and `scaled_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "discrete" and is_continuous:
            raise ValueError("You have choose a discrete distribution but `is_continuous` is true")
        if self.distribution == "auto":
            if is_continuous:
                self.distribution = "scaled_normal"
            else:
                self.distribution = "discrete"
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            flatten_dim=None,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        if is_continuous:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, np.sum(actions_dim) * 2)])
        else:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, action_dim) for action_dim in actions_dim])
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self._unimix = unimix
        self._action_clip = action_clip
```

**What it does**: Validate the distribution type, then build (a) a shared 5-layer MLP trunk (1024 units each, LayerNorm + SiLU) and (b) the output heads. For continuous actions: a single head outputting `2·sum(actions_dim)` (mean + raw-std interleaved). For discrete actions: one head per categorical sub-action (multi-discrete support — e.g. Minedojo). Stores the std and clip hyperparameters for use in `forward`. Note `flatten_dim=None` — the MLP does not collapse leading dims, so the actor preserves `[T, B, …]` shape.

---

## Line 783 — `Actor.forward`

```python
    def forward(
        self, state: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the mask to use on the actions.
                Default to None.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        pre_dist: List[Tensor] = [head(out) for head in self.mlp_heads]
        if self.is_continuous:
            mean, std = torch.chunk(pre_dist[0], 2, -1)
            if self.distribution == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5)
                std = F.softplus(std + self.init_std) + self.min_std
                actions_dist = Normal(mean, std)
                actions_dist = Independent(TransformedDistribution(actions_dist, TanhTransform()), 1)
            elif self.distribution == "normal":
                actions_dist = Normal(mean, std)
                actions_dist = Independent(actions_dist, 1)
            elif self.distribution == "scaled_normal":
                std = (self.max_std - self.min_std) * torch.sigmoid(std + self.init_std) + self.min_std
                dist = Normal(torch.tanh(mean), std)
                actions_dist = Independent(dist, 1)
            if not greedy:
                actions = actions_dist.rsample()
            else:
                sample = actions_dist.sample((100,))
                log_prob = actions_dist.log_prob(sample)
                actions = sample[log_prob.argmax(0)].view(1, 1, -1)
            if self._action_clip > 0.0:
                action_clip = torch.full_like(actions, self._action_clip)
                actions = actions * (action_clip / torch.maximum(action_clip, torch.abs(actions))).detach()
            actions = [actions]
            actions_dist = [actions_dist]
        else:
            actions_dist: List[Distribution] = []
            actions: List[Tensor] = []
            for logits in pre_dist:
                actions_dist.append(OneHotCategoricalStraightThrough(logits=self._uniform_mix(logits)))
                if not greedy:
                    actions.append(actions_dist[-1].rsample())
                else:
                    actions.append(actions_dist[-1].mode)
        return tuple(actions), tuple(actions_dist)
```

**What it does**: Build the action distribution and sample from it. **Continuous branch**: chunk the head output into `mean, std`; `scaled_normal` (DreamerV3 default) computes `std = (max_std - min_std)·σ(raw + init_std) + min_std` and `Normal(tanh(mean), std)`, then wraps in `Independent(..., 1)` so the per-dimension distributions sum log-probs over the action dim. The 5·tanh(x/5) saturation in `tanh_normal` keeps means in roughly `[-5, 5]`. Greedy continuous: take 100 samples, return the one with the highest log-prob (a soft mode approximation). **Discrete branch**: one `OneHotCategoricalStraightThrough` per action head; rsample uses straight-through (reparameterizable gradient through hard one-hot), greedy uses `.mode` (argmax one-hot). **Action clip**: `actions · clip / max(clip, |actions|)`, detached, scales actions back to `[-clip, clip]` without breaking gradients through the magnitude. Returns parallel tuples of actions and distributions.

---

## Line 839 — `Actor._uniform_mix`

```python
    def _uniform_mix(self, logits: Tensor) -> Tensor:
        if self._unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            logits = probs_to_logits(probs)
        return logits
```

**What it does**: Same uniform-mixing trick as `RSSM._uniform_mix`, but for discrete action distributions: `p ← (1-α)·p + α·(1/K)`. Unlike the RSSM version, the action logits are already `[…, action_dim]` so no reshape is needed. Keeps the policy from collapsing to a deterministic delta on any single action, which would zero gradients.

---

## Line 848 — `class MinedojoActor`

```python
class MinedojoActor(Actor):
```

**What it does**: Subclass of `Actor` specialised for Minedojo's hierarchical multi-discrete action space (functional/craft-smelt/destroy/equip-place actions). Only overrides `__init__` (slightly different defaults) and `forward` (to apply environment-supplied action masks). The class has no docstring in the source.

---

## Line 849 — `MinedojoActor.__init__`

```python
    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        is_continuous: bool,
        distribution_cfg: Dict[str, Any],
        init_std: float = 0,
        min_std: float = 0.1,
        dense_units: int = 1024,
        activation: nn.Module = nn.SiLU,
        mlp_layers: int = 5,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
        unimix: float = 0.01,
        action_clip: float = 1.0,
    ) -> None:
        super().__init__(
            latent_state_size=latent_state_size,
            actions_dim=actions_dim,
            is_continuous=is_continuous,
            distribution_cfg=distribution_cfg,
            init_std=init_std,
            min_std=min_std,
            dense_units=dense_units,
            activation=activation,
            mlp_layers=mlp_layers,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
            unimix=unimix,
            action_clip=action_clip,
        )
```

**What it does**: Forwards all arguments to `Actor.__init__`. The only differences from the base default are `init_std=0` and `min_std=0.1`. Notably, the parent's `max_std` is *not* exposed — Minedojo is discrete, so `scaled_normal` never fires.

---

## Line 881 — `MinedojoActor.forward`

```python
    def forward(
        self, state: Tensor, greedy: bool = True, mask: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to True.
            mask (Dict[str, Tensor], optional): the mask to apply to the actions.
                Default to None.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        actions_logits: List[Tensor] = [self._uniform_mix(head(out)) for head in self.mlp_heads]
        actions_dist: List[Distribution] = []
        actions: List[Tensor] = []
        functional_action = None
        for i, logits in enumerate(actions_logits):
            if mask is not None:
                if i == 0:
                    logits[torch.logical_not(mask["mask_action_type"].expand_as(logits))] = -torch.inf
                elif i == 1:
                    mask["mask_craft_smelt"] = mask["mask_craft_smelt"].expand_as(logits)
                    for t in range(functional_action.shape[0]):
                        for b in range(functional_action.shape[1]):
                            sampled_action = functional_action[t, b].item()
                            if sampled_action == 15:  # Craft action
                                logits[t, b][torch.logical_not(mask["mask_craft_smelt"][t, b])] = -torch.inf
                elif i == 2:
                    mask["mask_destroy"] = mask["mask_destroy"].expand_as(logits)
                    mask["mask_equip_place"] = mask["mask_equip_place"].expand_as(logits)
                    for t in range(functional_action.shape[0]):
                        for b in range(functional_action.shape[1]):
                            sampled_action = functional_action[t, b].item()
                            if sampled_action in (16, 17):  # Equip/Place action
                                logits[t, b][torch.logical_not(mask["mask_equip_place"][t, b])] = -torch.inf
                            elif sampled_action == 18:  # Destroy action
                                logits[t, b][torch.logical_not(mask["mask_destroy"][t, b])] = -torch.inf
            actions_dist.append(OneHotCategoricalStraightThrough(logits=logits))
            if not greedy:
                actions.append(actions_dist[-1].rsample())
            else:
                actions.append(actions_dist[-1].mode)
            if functional_action is None:
                functional_action = actions[0].argmax(dim=-1)  # [T, B]
        return tuple(actions), tuple(actions_dist)
```

**What it does**: Hierarchical multi-discrete action sampling. Head 0 is the *functional action type* (move, jump, craft, equip, destroy, etc.). Heads 1 and 2 are conditional sub-actions whose valid set depends on head 0. The mask flow: (1) Mask head-0 logits using `mask_action_type`. (2) Sample head 0 (`functional_action`). (3) For head 1 (craft/smelt argument), set logits to `-inf` for invalid items *only* when head 0 indicates Craft (id=15). (4) For head 2, similar branching for Equip/Place (16/17) and Destroy (18). The double-loop over `(t, b)` is necessary because each batch element conditions on its own sampled functional action — vectorising this would require gather-style masking. `OneHotCategoricalStraightThrough` is used uniformly so gradients flow through the masking.

---

## Line 935 — `build_agent`

```python
def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_state: Optional[Dict[str, Tensor]] = None,
    critic_state: Optional[Dict[str, Tensor]] = None,
    target_critic_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, PlayerDV3]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        actions_dim (Sequence[int]): the dimension of the actions.
        is_continuous (bool): whether or not the actions are continuous.
        cfg (DictConfig): the configs of DreamerV3.
        obs_space (Dict[str, Any]): the observation space.
        world_model_state (Dict[str, Tensor], optional): the state of the world model.
            Default to None.
        actor_state: (Dict[str, Tensor], optional): the state of the actor.
            Default to None.
        critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.
        target_critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and
        reward models and the continue model.
        The actor (_FabricModule).
        The critic (_FabricModule).
        The target critic (nn.Module).
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic

    # Sizes
    recurrent_state_size = world_model_cfg.recurrent_model.recurrent_state_size
    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    latent_state_size = stochastic_size + recurrent_state_size

    # Define models
    cnn_stages = int(np.log2(cfg.env.screen_size) - np.log2(4))
    cnn_encoder = (
        CNNEncoder(
            keys=cfg.algo.cnn_keys.encoder,
            input_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.encoder],
            image_size=obs_space[cfg.algo.cnn_keys.encoder[0]].shape[-2:],
            channels_multiplier=world_model_cfg.encoder.cnn_channels_multiplier,
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.encoder.cnn_layer_norm.cls),
            layer_norm_kw=world_model_cfg.encoder.cnn_layer_norm.kw,
            activation=hydra.utils.get_class(world_model_cfg.encoder.cnn_act),
            stages=cnn_stages,
        )
        if cfg.algo.cnn_keys.encoder is not None and len(cfg.algo.cnn_keys.encoder) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            keys=cfg.algo.mlp_keys.encoder,
            input_dims=[obs_space[k].shape[0] for k in cfg.algo.mlp_keys.encoder],
            mlp_layers=world_model_cfg.encoder.mlp_layers,
            dense_units=world_model_cfg.encoder.dense_units,
            activation=hydra.utils.get_class(world_model_cfg.encoder.dense_act),
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.encoder.mlp_layer_norm.cls),
            layer_norm_kw=world_model_cfg.encoder.mlp_layer_norm.kw,
        )
        if cfg.algo.mlp_keys.encoder is not None and len(cfg.algo.mlp_keys.encoder) > 0
        else None
    )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)

    recurrent_model = RecurrentModel(
        input_size=int(sum(actions_dim) + stochastic_size),
        recurrent_state_size=world_model_cfg.recurrent_model.recurrent_state_size,
        dense_units=world_model_cfg.recurrent_model.dense_units,
        layer_norm_cls=hydra.utils.get_class(world_model_cfg.recurrent_model.layer_norm.cls),
        layer_norm_kw=world_model_cfg.recurrent_model.layer_norm.kw,
    )
    represention_model_input_size = encoder.output_dim
    if not cfg.algo.world_model.decoupled_rssm:
        represention_model_input_size += recurrent_state_size
    representation_ln_cls = hydra.utils.get_class(world_model_cfg.representation_model.layer_norm.cls)
    representation_model = MLP(
        input_dims=represention_model_input_size,
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.representation_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.representation_model.dense_act),
        layer_args={"bias": representation_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[representation_ln_cls],
        norm_args=[
            {
                **world_model_cfg.representation_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.representation_model.hidden_size,
            }
        ],
    )
    transition_ln_cls = hydra.utils.get_class(world_model_cfg.transition_model.layer_norm.cls)
    transition_model = MLP(
        input_dims=recurrent_state_size,
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.transition_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.transition_model.dense_act),
        layer_args={"bias": transition_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[transition_ln_cls],
        norm_args=[
            {
                **world_model_cfg.transition_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.transition_model.hidden_size,
            }
        ],
    )

    if cfg.algo.world_model.decoupled_rssm:
        rssm_cls = DecoupledRSSM
    else:
        rssm_cls = RSSM
    rssm = rssm_cls(
        recurrent_model=recurrent_model.apply(init_weights),
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        distribution_cfg=cfg.distribution,
        discrete=world_model_cfg.discrete_size,
        unimix=cfg.algo.unimix,
        learnable_initial_recurrent_state=cfg.algo.world_model.learnable_initial_recurrent_state,
    ).to(fabric.device)

    cnn_decoder = (
        CNNDecoder(
            keys=cfg.algo.cnn_keys.decoder,
            output_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.decoder],
            channels_multiplier=world_model_cfg.observation_model.cnn_channels_multiplier,
            latent_state_size=latent_state_size,
            cnn_encoder_output_dim=cnn_encoder.output_dim,
            image_size=obs_space[cfg.algo.cnn_keys.decoder[0]].shape[-2:],
            activation=hydra.utils.get_class(world_model_cfg.observation_model.cnn_act),
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.observation_model.cnn_layer_norm.cls),
            layer_norm_kw=world_model_cfg.observation_model.mlp_layer_norm.kw,
            stages=cnn_stages,
        )
        if cfg.algo.cnn_keys.decoder is not None and len(cfg.algo.cnn_keys.decoder) > 0
        else None
    )
    mlp_decoder = (
        MLPDecoder(
            keys=cfg.algo.mlp_keys.decoder,
            output_dims=[obs_space[k].shape[0] for k in cfg.algo.mlp_keys.decoder],
            latent_state_size=latent_state_size,
            mlp_layers=world_model_cfg.observation_model.mlp_layers,
            dense_units=world_model_cfg.observation_model.dense_units,
            activation=hydra.utils.get_class(world_model_cfg.observation_model.dense_act),
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.observation_model.mlp_layer_norm.cls),
            layer_norm_kw=world_model_cfg.observation_model.mlp_layer_norm.kw,
        )
        if cfg.algo.mlp_keys.decoder is not None and len(cfg.algo.mlp_keys.decoder) > 0
        else None
    )
    observation_model = MultiDecoder(cnn_decoder, mlp_decoder)

    reward_ln_cls = hydra.utils.get_class(world_model_cfg.reward_model.layer_norm.cls)
    reward_model = MLP(
        input_dims=latent_state_size,
        output_dim=world_model_cfg.reward_model.bins,
        hidden_sizes=[world_model_cfg.reward_model.dense_units] * world_model_cfg.reward_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.reward_model.dense_act),
        layer_args={"bias": reward_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=reward_ln_cls,
        norm_args={
            **world_model_cfg.reward_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.reward_model.dense_units,
        },
    )

    discount_ln_cls = hydra.utils.get_class(world_model_cfg.discount_model.layer_norm.cls)
    continue_model = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[world_model_cfg.discount_model.dense_units] * world_model_cfg.discount_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.discount_model.dense_act),
        layer_args={"bias": discount_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=discount_ln_cls,
        norm_args={
            **world_model_cfg.discount_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.discount_model.dense_units,
        },
    )
    world_model = WorldModel(
        encoder.apply(init_weights),
        rssm,
        observation_model.apply(init_weights),
        reward_model.apply(init_weights),
        continue_model.apply(init_weights),
    )

    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    actor: Actor | MinedojoActor = actor_cls(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        dense_units=actor_cfg.dense_units,
        activation=hydra.utils.get_class(actor_cfg.dense_act),
        mlp_layers=actor_cfg.mlp_layers,
        distribution_cfg=cfg.distribution,
        layer_norm_cls=hydra.utils.get_class(actor_cfg.layer_norm.cls),
        layer_norm_kw=actor_cfg.layer_norm.kw,
        unimix=cfg.algo.unimix,
        action_clip=actor_cfg.action_clip,
    )

    critic_ln_cls = hydra.utils.get_class(critic_cfg.layer_norm.cls)
    critic = MLP(
        input_dims=latent_state_size,
        output_dim=critic_cfg.bins,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=hydra.utils.get_class(critic_cfg.dense_act),
        layer_args={"bias": critic_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=critic_ln_cls,
        norm_args={
            **critic_cfg.layer_norm.kw,
            "normalized_shape": critic_cfg.dense_units,
        },
    )
    actor.apply(init_weights)
    critic.apply(init_weights)

    if cfg.algo.hafner_initialization:
        actor.mlp_heads.apply(uniform_init_weights(1.0))
        critic.model[-1].apply(uniform_init_weights(0.0))
        rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
        rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
        world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
        if mlp_decoder is not None:
            mlp_decoder.heads.apply(uniform_init_weights(1.0))
        if cnn_decoder is not None:
            cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

    # Load models from checkpoint
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if actor_state:
        actor.load_state_dict(actor_state)
    if critic_state:
        critic.load_state_dict(critic_state)

    # Create the player agent
    fabric_player = get_single_device_fabric(fabric)
    player = PlayerDV3(
        copy.deepcopy(world_model.encoder),
        copy.deepcopy(world_model.rssm),
        copy.deepcopy(actor),
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        fabric_player.device,
        discrete_size=cfg.algo.world_model.discrete_size,
    )

    # Setup models with Fabric
    world_model.encoder = fabric.setup_module(world_model.encoder)
    world_model.observation_model = fabric.setup_module(world_model.observation_model)
    world_model.reward_model = fabric.setup_module(world_model.reward_model)
    world_model.rssm.recurrent_model = fabric.setup_module(world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = fabric.setup_module(world_model.rssm.representation_model)
    world_model.rssm.transition_model = fabric.setup_module(world_model.rssm.transition_model)
    if world_model.continue_model:
        world_model.continue_model = fabric.setup_module(world_model.continue_model)
    actor = fabric.setup_module(actor)
    critic = fabric.setup_module(critic)

    # Setup target critic with a SingleDeviceStrategy
    target_critic = copy.deepcopy(critic.module)
    if target_critic_state:
        target_critic.load_state_dict(target_critic_state)
    target_critic = fabric_player.setup_module(target_critic)

    # Setup the player agent with a single-device Fabric
    player.encoder = fabric_player.setup_module(player.encoder)
    player.rssm.recurrent_model = fabric_player.setup_module(player.rssm.recurrent_model)
    player.rssm.transition_model = fabric_player.setup_module(player.rssm.transition_model)
    player.rssm.representation_model = fabric_player.setup_module(player.rssm.representation_model)
    player.actor = fabric_player.setup_module(player.actor)

    # Tie weights between the agent and the player
    for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.parameters(), player.rssm.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(actor.parameters(), player.actor.parameters()):
        p.data = agent_p.data
    return world_model, actor, critic, target_critic, player
```

**What it does**: The single factory that wires the whole agent together. Walkthrough of the major phases:

1. **Read sizes from config** (`recurrent_state_size`, `stochastic_size = stoch·discrete`, `latent_state_size = stochastic + recurrent`). `cnn_stages = log2(screen_size) - log2(4)` — 4 for a 64×64 screen.

2. **Encoders** — Build `CNNEncoder` if `cnn_keys.encoder` is non-empty, `MLPEncoder` if `mlp_keys.encoder` is non-empty (either may be `None`). Wrap in [`MultiEncoder`](models.md) which concatenates their outputs.

3. **Recurrent, representation, transition models** — Built directly as `RecurrentModel` + two `MLP`s. Note the representation model input size depends on `decoupled_rssm`: it's `encoder.output_dim` alone in the decoupled case, else `+ recurrent_state_size`. The transition model always takes only `recurrent_state_size`. The two MLPs have a single hidden layer (`hidden_size`, LayerNorm + SiLU) and output `stochastic_size` logits (which will be reshaped to `[stoch, discrete]` inside `_uniform_mix`).

4. **RSSM** — Select `RSSM` or `DecoupledRSSM` based on config. `recurrent_model.apply(init_weights)` and the same for the two MLPs runs Hafner's Truncated-Normal initialization (see [`utils.md`](utils.md)). Move to `fabric.device`.

5. **Decoders** — Build `CNNDecoder` / `MLPDecoder` if their key lists are non-empty. Wrap in [`MultiDecoder`](models.md).

6. **Reward + continue heads** — Both are `MLP`s with `latent_state_size` input. Reward output is `world_model_cfg.reward_model.bins` (e.g. 255) — these are *logits over discrete value bins* that get wrapped in [`TwoHotEncodingDistribution`](distribution.md) downstream. Continue model has `output_dim=1` — a scalar logit consumed by [`BernoulliSafeMode`](distribution.md). Both have several hidden layers (`mlp_layers`, default 2) with LayerNorm + SiLU.

7. **Assemble `WorldModel`** — Container imported from DreamerV2 that bundles encoder, RSSM, observation_model, reward_model, continue_model. Each sub-module gets `init_weights` applied.

8. **Actor and critic** — `actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)` picks either `Actor` or `MinedojoActor` based on config. Critic is just an `MLP` outputting `critic_cfg.bins` logits — same twohot setup as the reward model. Both get `init_weights` applied.

9. **Hafner initialization** — When `hafner_initialization=True` (DreamerV3 default), the last layer of various heads gets *uniform-init* with specific scales. Critically: `critic.model[-1]` and `world_model.reward_model.model[-1]` use scale `0.0` (zero-init the final layer) — this matches Hafner's recipe that the twohot reward/value predictor starts at the uniform-over-bins distribution. Actor heads, transition/representation final layers, continue head, and decoder heads get scale `1.0`. **The zero-init of reward and critic heads is cascade target #27 in the project's fix list.**

10. **Optional checkpoint loading** — Restore weights into freshly-built modules if state dicts are provided.

11. **Build the player** — `copy.deepcopy` the encoder, RSSM, and actor for the player's own modules (they'll be tied via aliasing below). The player runs on a single-device fabric so multi-GPU rank-0 rollout works.

12. **Setup with Fabric** — Each top-level module is wrapped via `fabric.setup_module` so the optimizer/DDP/precision wrappers are in place. Target critic is built by `copy.deepcopy(critic.module)` (unwrap the FabricModule first) and wrapped under the *single-device* fabric so it lives on the rollout device. **The target critic is updated via polyak averaging** in the main loop (see [`utils_core.md`](utils_core.md) `polyak_update`).

13. **Weight tying** — The player's encoder / RSSM / actor are aliased to the trainable ones by `p.data = agent_p.data`. This is critical: the player must always see the latest weights without needing an explicit copy after every optimizer step.

Returns the five objects consumed by the main training loop: `(world_model, actor, critic, target_critic, player)`.
