---
title: "Sheeprl Reference: models.py"
source: tmp/sheeprl/sheeprl/models/models.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `models.py`

> **Source**: `tmp/sheeprl/sheeprl/models/models.py` — 525 lines, 38 def/class sections.
> **Purpose** (one-line): Generic network primitives: `MLP`, `CNN`, `DeCNN`, `NatureCNN`, `LayerNormGRUCell`, `MultiEncoder`, `MultiDecoder`, plus channel-aware `LayerNorm` variants — the building blocks composed by `agent.py` into the world model + actor + critic.
> **Imports from elsewhere in this index**: self-contained — pulls helpers (`create_layers`, `miniblock`, `cnn_forward`, `ArgsType`, `ModuleType`) from `sheeprl.utils.model`. The DreamerV3 `agent.py` instantiates these classes (`MLP`, `CNN`, `DeCNN`, `LayerNormGRUCell`, `MultiEncoder`, `MultiDecoder`, `LayerNormChannelLast`, `LayerNorm`) but this file does not import upward.

---

## Table of Contents

- [Lines 1–14 — Imports](#lines-114--imports)
- [Line 16 — `class MLP`](#line-16--class-mlp)
- [Line 46 — `MLP.__init__`](#line-46--mlp__init__)
- [Line 104 — `MLP.model` (property)](#line-104--mlpmodel-property)
- [Line 108 — `MLP.output_dim` (property)](#line-108--mlpoutput_dim-property)
- [Line 112 — `MLP.flatten_dim` (property)](#line-112--mlpflatten_dim-property)
- [Line 116 — `MLP.forward`](#line-116--mlpforward)
- [Line 122 — `class CNN`](#line-122--class-cnn)
- [Line 147 — `CNN.__init__`](#line-147--cnn__init__)
- [Line 193 — `CNN.model` (property)](#line-193--cnnmodel-property)
- [Line 197 — `CNN.output_dim` (property)](#line-197--cnnoutput_dim-property)
- [Line 201 — `CNN.forward`](#line-201--cnnforward)
- [Line 205 — `class DeCNN`](#line-205--class-decnn)
- [Line 230 — `DeCNN.__init__`](#line-230--decnn__init__)
- [Line 276 — `DeCNN.model` (property)](#line-276--decnnmodel-property)
- [Line 280 — `DeCNN.output_dim` (property)](#line-280--decnnoutput_dim-property)
- [Line 284 — `DeCNN.forward`](#line-284--decnnforward)
- [Line 288 — `class NatureCNN`](#line-288--class-naturecnn)
- [Line 301 — `NatureCNN.__init__`](#line-301--naturecnn__init__)
- [Line 322 — `NatureCNN.output_dim` (property)](#line-322--naturecnnoutput_dim-property)
- [Line 325 — `NatureCNN.forward`](#line-325--naturecnnforward)
- [Line 331 — `class LayerNormGRUCell`](#line-331--class-layernormgrucell)
- [Line 351 — `LayerNormGRUCell.__init__`](#line-351--layernormgrucell__init__)
- [Line 370 — `LayerNormGRUCell.forward`](#line-370--layernormgrucellforward)
- [Line 413 — `class MultiEncoder`](#line-413--class-multiencoder)
- [Line 414 — `MultiEncoder.__init__`](#line-414--multiencoder__init__)
- [Line 458 — `MultiEncoder.cnn_keys` (property)](#line-458--multiencodercnn_keys-property)
- [Line 462 — `MultiEncoder.mlp_keys` (property)](#line-462--multiencodermlp_keys-property)
- [Line 465 — `MultiEncoder.forward`](#line-465--multiencoderforward)
- [Line 478 — `class MultiDecoder`](#line-478--class-multidecoder)
- [Line 479 — `MultiDecoder.__init__`](#line-479--multidecoder__init__)
- [Line 491 — `MultiDecoder.cnn_keys` (property)](#line-491--multidecodercnn_keys-property)
- [Line 495 — `MultiDecoder.mlp_keys` (property)](#line-495--multidecodermlp_keys-property)
- [Line 498 — `MultiDecoder.forward`](#line-498--multidecoderforward)
- [Line 507 — `class LayerNormChannelLast`](#line-507--class-layernormchannellast)
- [Line 508 — `LayerNormChannelLast.__init__`](#line-508--layernormchannellast__init__)
- [Line 511 — `LayerNormChannelLast.forward`](#line-511--layernormchannellastforward)
- [Line 521 — `class LayerNorm`](#line-521--class-layernorm)
- [Line 522 — `LayerNorm.forward`](#line-522--layernormforward)

---

## Lines 1–14 — Imports

```python
"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""

import warnings
from math import prod
from typing import Any, Callable, Dict, Optional, Sequence, Union, no_type_check

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sheeprl.utils.model import ArgsType, ModuleType, cnn_forward, create_layers, miniblock
```

The file credits Tianshou for the original common-net design. It imports `warnings` (used to flag possibly-unflattened inputs in `MLP`), `prod` (collapse multi-dim input shapes to a single linear-input size), and typing primitives. PyTorch core (`torch`, `F`, `Tensor`, `nn`) supplies the layer base classes. The local `sheeprl.utils.model` helpers — `ArgsType` / `ModuleType` typedefs, `cnn_forward` (collapses batch/time dims for conv), `create_layers` (broadcasts a single layer-class into a per-layer list), and `miniblock` (linear/conv → dropout → norm → activation tuple) — are the engine that drives the loop bodies of `MLP`, `CNN`, and `DeCNN`. No DreamerV3-specific code lives here; this is a reusable model toolbox.

---

## Line 16 — `class MLP`

```python
class MLP(nn.Module):
    """Simple MLP backbone.

    Args:
        input_dims (Union[int, Sequence[int]]): dimensions of the input vector.
        output_dim (int, optional): dimension of the output vector. If set to None, there
            is no final linear layer. Else, a final linear layer is added.
            Defaults to None.
        hidden_sizes (Sequence[int], optional): shape of MLP passed in as a list, not including
            input_dims and output_dim.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
        flatten_dim (int, optional): whether to flatten input data. The flatten dimension starts from 1.
            Defaults to True.
    """
```

`MLP` is the project's universal multi-layer perceptron. The docstring is exhaustive: per-layer dropout / norm / activation can be configured by passing either a single module class (broadcast across all layers) or an explicit list matching `hidden_sizes` length. `output_dim=None` strips the final projection so callers can pass the penultimate hidden vector elsewhere (e.g., to a `Categorical` head). `flatten_dim` lets callers feed tensors with batch/time leading dims and have the MLP flatten everything from a given axis. The default activation is `nn.ReLU` here — DreamerV3's `agent.py` overrides this to `nn.SiLU`.

---

## Line 46 — `MLP.__init__`

```python
    def __init__(
        self,
        input_dims: Union[int, Sequence[int]],
        output_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = (),
        layer_args: Optional[ArgsType] = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        flatten_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_sizes)
        if num_layers < 1 and output_dim is None:
            raise ValueError("The number of layers should be at least 1.")

        if isinstance(input_dims, Sequence) and flatten_dim is None:
            warnings.warn(
                "input_dims is a sequence, but flatten_dim is not specified. "
                "Be careful to flatten the input data correctly before the forward."
            )

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        if isinstance(input_dims, int):
            input_dims = [input_dims]
        hidden_sizes = [prod(input_dims)] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, nn.Linear, l_args, drop, drop_args, norm, norm_args, activ, act_args)
        if output_dim is not None:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]

        self._output_dim = output_dim or hidden_sizes[-1]
        self._model = nn.Sequential(*model)
        self._flatten_dim = flatten_dim
```

The constructor first guards against a zero-layer no-output configuration. It uses `create_layers` to convert single-class arguments into per-layer lists (so dropout/norm/activation can vary by depth). `prod(input_dims)` collapses multi-dim inputs (e.g., flattened latent + action shapes) to a single linear-input width. The main loop emits a `miniblock` (Linear → optional dropout → optional norm → optional activation) per hidden layer. Crucially, if `output_dim is not None`, a **bare `nn.Linear`** is appended **without** activation or norm — this is the head used for reward/value/policy outputs, and `agent.py` zero-inits its weights downstream (cascade-target init for reward + critic heads).

---

## Line 104 — `MLP.model` (property)

```python
    @property
    def model(self) -> nn.Module:
        return self._model
```

Exposes the inner `nn.Sequential` so callers (or the matching `CNN`/`DeCNN`) can introspect / iterate the layer stack. Used by `agent.py` when it walks the head to apply Hafner's `uniform_init_weights(0.0)` to the terminal `nn.Linear`.

---

## Line 108 — `MLP.output_dim` (property)

```python
    @property
    def output_dim(self) -> int:
        return self._output_dim
```

Reports the final output width — either the explicit `output_dim` or the last hidden size if no head was added. `MultiEncoder` reads this to compute the concatenated latent dim shared with the RSSM.

---

## Line 112 — `MLP.flatten_dim` (property)

```python
    @property
    def flatten_dim(self) -> Optional[int]:
        return self._flatten_dim
```

Read-only accessor for the flatten starting axis (or `None`). The `forward` consults this to decide whether to call `.flatten(...)` before the first linear.

---

## Line 116 — `MLP.forward`

```python
    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        if self.flatten_dim is not None:
            obs = obs.flatten(self.flatten_dim)
        return self.model(obs)
```

Optional flatten then run the `nn.Sequential`. Decorated `@no_type_check` because `obs` is sometimes a dict-key-derived tensor whose dtype the type checker can't reconcile. The simplicity here is intentional — every layer's dropout/norm/activation is baked inside `nn.Sequential` so the forward is one line.

---

## Line 122 — `class CNN`

```python
class CNN(nn.Module):
    """Simple CNN backbone.

    Args:
        input_channels (int): dimensions of the input channels.
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN,
            including the output channels.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
    """
```

`CNN` is the convolutional counterpart of `MLP`. It composes 2-D conv layers (default `nn.Conv2d`, but `cnn_layer` can be swapped) into a `nn.Sequential` driven by the same `miniblock` helper. Per-layer dropout / norm / activation broadcasting matches `MLP`. DreamerV3 instantiates `CNN` for the pixel encoder, swapping `norm_layer=LayerNormChannelLast` (defined below) to keep LayerNorm semantics on NCHW feature maps.

---

## Line 147 — `CNN.__init__`

```python
    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        cnn_layer: ModuleType = nn.Conv2d,
        layer_args: ArgsType = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)
```

Mirrors `MLP.__init__` in structure: validate non-empty hidden list, broadcast per-layer dropout/norm/activation, iterate `miniblock` to assemble (Conv → dropout → norm → activation) tuples, store as `nn.Sequential`. `_output_dim` is the **channel count of the final conv** — *not* a flattened spatial size; that is computed by callers (see `NatureCNN`) via a probe forward pass. `layer_args` is the per-layer kernel / stride / padding dict — the user supplies this to control spatial down-sampling.

---

## Line 193 — `CNN.model` (property)

```python
    @property
    def model(self) -> nn.Module:
        return self._model
```

Same role as `MLP.model`: returns the inner `nn.Sequential` for introspection. `NatureCNN.__init__` uses this to probe spatial output dim and `cnn_forward` walks it directly.

---

## Line 197 — `CNN.output_dim` (property)

```python
    @property
    def output_dim(self) -> int:
        return self._output_dim
```

Returns last-layer channel count. `MultiEncoder` reads this when wiring the CNN branch but the **true latent dim** the RSSM sees is post-flatten; subclasses overwrite `_output_dim` once the spatial collapse width is known.

---

## Line 201 — `CNN.forward`

```python
    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        return self.model(obs)
```

A pure pass-through. The conv stack does its own NCHW handling; flattening is left to whatever feeds the resulting tensor into a linear head.

---

## Line 205 — `class DeCNN`

```python
class DeCNN(nn.Module):
    """Simple DeCNN backbone.

    Args:
        input_channels (int): dimensions of the input channels.
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN,
            including the output channels.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
    """
```

The transposed-convolution mirror of `CNN`. Defaults `cnn_layer=nn.ConvTranspose2d` so feature maps are *up-sampled*. Used by DreamerV3 to build the pixel-observation decoder branch of `MultiDecoder`, which reconstructs RGB observations from the world model's flattened latent + recurrent state.

---

## Line 230 — `DeCNN.__init__`

```python
    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int] = (),
        cnn_layer: ModuleType = nn.ConvTranspose2d,
        layer_args: ArgsType = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)
```

Structurally identical to `CNN.__init__`. The only meaningful change is the default `cnn_layer=nn.ConvTranspose2d`, which up-samples spatial dimensions per layer (controlled by per-layer `kernel_size` / `stride` / `output_padding` in `layer_args`). The DreamerV3 image decoder typically uses 4–5 transposed-conv layers to grow from a 1×1 (or 4×4) "spatial" latent up to the original observation size.

---

## Line 276 — `DeCNN.model` (property)

```python
    @property
    def model(self) -> nn.Module:
        return self._model
```

Returns the inner `nn.Sequential` of transposed convs. Same accessor pattern as `MLP.model` / `CNN.model`.

---

## Line 280 — `DeCNN.output_dim` (property)

```python
    @property
    def output_dim(self) -> int:
        return self._output_dim
```

Returns the channel count of the final transposed conv — typically the number of image channels (3 for RGB). Used by `MultiDecoder`-side observation reconstruction shape checks.

---

## Line 284 — `DeCNN.forward`

```python
    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        return self.model(obs)
```

Pass-through forward; spatial reshape from latent vector → 4-D feature map happens in the wrapper that owns this `DeCNN`.

---

## Line 288 — `class NatureCNN`

```python
class NatureCNN(CNN):
    """CNN from DQN Nature paper: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.

    Args:
        in_channels (int): the input channels to the first convolutional layer
        features_dim (int): the features dimension in output from the last convolutional layer
        screen_size (int, optional): the dimension of the input image as a single integer.
            Needed to extract the features and compute the output dimension after all the
            convolutional layers.
            Defaults to 64.
    """
```

The classic Mnih et al. DQN CNN, instantiated as a thin subclass of `CNN` with the canonical `[32, 64, 64]` channel ladder and `(8,4)`/`(4,2)`/`(3,1)` kernel/stride schedule. Includes a probe-forward to determine the post-conv flattened dim and an optional `nn.Linear` projection to a user-chosen `features_dim`. Not used by DreamerV3 (which uses Hafner's symmetric encoder), but kept here for the SAC / PPO algorithm families in `sheeprl`.

---

## Line 301 — `NatureCNN.__init__`

```python
    def __init__(self, in_channels: int, features_dim: int, screen_size: int = 64):
        super().__init__(
            in_channels,
            [32, 64, 64],
            layer_args=[
                {"kernel_size": 8, "stride": 4},
                {"kernel_size": 4, "stride": 2},
                {"kernel_size": 3, "stride": 1},
            ],
        )

        with torch.no_grad():
            x = self.model(torch.rand(1, in_channels, screen_size, screen_size, device=self.model[0].weight.device))
            out_dim = x.flatten(1).shape[1]
        self._output_dim = out_dim
        self.fc = None
        if features_dim is not None:
            self._output_dim = features_dim
            self.fc = nn.Linear(out_dim, features_dim)
```

Hands the DQN ladder + kernel/stride table to `CNN.__init__`. Then in a `torch.no_grad()` block it runs a single fake `(1, C, S, S)` tensor through the conv stack to learn the post-flatten width — this avoids hard-coding spatial arithmetic per `screen_size`. If `features_dim` is supplied, it builds the projection `nn.Linear(out_dim, features_dim)` and overwrites `_output_dim` so downstream sizing is consistent. Note: the probe uses `self.model[0].weight.device` to keep CPU/GPU placement matched.

---

## Line 322 — `NatureCNN.output_dim` (property)

```python
    @property
    def output_dim(self) -> int:
        return self._output_dim
```

Overrides `CNN.output_dim` to return the **post-flatten / post-FC** width rather than the last conv's channel count — so that `MultiEncoder` sees the actual feature size it'll receive.

---

## Line 325 — `NatureCNN.forward`

```python
    def forward(self, x: Tensor) -> Tensor:
        x = cnn_forward(self.model, x, input_dim=x.shape[-3:], output_dim=(-1,))
        x = F.relu(self.fc(x))
        return x
```

`cnn_forward` (from `utils.model`) flattens leading batch/time dims so the conv stack sees a true 4-D `(B*T, C, H, W)`, then reshapes back. Result is flattened to `(B*T, -1)`, run through `self.fc`, and ReLU-activated. The final ReLU is intentional — the consumer expects pre-head features, not a raw linear output.

---

## Line 331 — `class LayerNormGRUCell`

```python
class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm, taken
    from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.

    Args:
        input_size (int): the input size.
        hidden_size (int): the hidden state size
        bias (bool, optional): whether to apply a bias to the input projection.
            Defaults to True.
        batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
            Defaults to False.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to nn.Identiy.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {}.
    """
```

The DreamerV2/V3 recurrent cell, ported verbatim from Hafner's reference implementation. Differs from `torch.nn.GRUCell` in three ways: (1) the input + hidden are concatenated and projected through a **single** `nn.Linear` to 3× hidden, (2) a LayerNorm is applied to that joint projection, and (3) the update gate uses `sigmoid(update - 1)` to bias the cell toward retaining previous state (Hafner's "less reset" trick that stabilises long-rollout imagination). The "sequence-length-1" 3-D handling is for slot-by-slot rollouts in the RSSM.

---

## Line 351 — `LayerNormGRUCell.__init__`

```python
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm_cls: Callable[..., nn.Module] = nn.Identity,
        layer_norm_kw: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        # Avoid multiple values for the `normalized_shape` argument
        layer_norm_kw.pop("normalized_shape", None)
        self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)
```

Builds the single joint projection `nn.Linear(input + hidden → 3·hidden)` (one matmul for reset/cand/update gates combined) and the optional `LayerNorm(3·hidden_size)` on its output. `layer_norm_cls` defaults to `nn.Identity` so the cell is a plain "fused-gate GRU" by default; DreamerV3's RSSM passes `LayerNorm` (the wrapper defined at the bottom of this file) and `eps=1e-3` per Hafner's spec. The `pop("normalized_shape", None)` guard prevents the user from double-supplying the first positional arg.

---

## Line 370 — `LayerNormGRUCell.forward`

```python
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx
```

The recurrence body. Step 1 handles shape gymnastics: a 3-D `(B, 1, F)` or `(1, B, F)` input is squeezed to 2-D, a 3-D hidden is squeezed to 2-D, and unbatched inputs get unsqueezed to a singleton batch. Step 2 concatenates hidden + input and applies the fused linear + LayerNorm. Step 3 splits into the three gates: `reset` is a standard sigmoid, the candidate state is `tanh(reset * cand)` (gating happens **inside** the tanh, not on its output — Hafner's variant), and the update gate is `sigmoid(update - 1)` which biases the **prior** of update toward "keep old state" (≈ 0.27 vs 0.5 in vanilla GRU). The update mix `hx = update * cand + (1 - update) * hx` then produces the new hidden, and shapes are restored to match the input rank.

---

## Line 413 — `class MultiEncoder`

```python
class MultiEncoder(nn.Module):
```

The dict-observation encoder wrapper. Given a CNN encoder (for image keys) and/or an MLP encoder (for vector keys), it dispatches the dict to whichever sub-encoders are present and concatenates their outputs along the last axis. Either sub-encoder can be `None` (single-modality envs) but not both. This is the entry-point that converts heterogeneous observations into the single "encoded obs" vector consumed by the RSSM's posterior network.

---

## Line 414 — `MultiEncoder.__init__`

```python
    def __init__(
        self,
        cnn_encoder: ModuleType,
        mlp_encoder: ModuleType,
    ) -> None:
        super().__init__()
        if cnn_encoder is None and mlp_encoder is None:
            raise ValueError("There must be at least one encoder, both cnn and mlp encoders are None")
        self.has_cnn_encoder = False
        self.has_mlp_encoder = False
        if cnn_encoder is not None:
            if getattr(cnn_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(cnn_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
            self.has_cnn_encoder = True
        if mlp_encoder is not None:
            if getattr(mlp_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(mlp_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
            self.has_mlp_encoder = True
        self.has_both_encoders = self.has_cnn_encoder and self.has_mlp_encoder
        self.cnn_encoder = cnn_encoder
        self.mlp_encoder = mlp_encoder
        self.cnn_input_dim = self.cnn_encoder.input_dim if self.cnn_encoder is not None else None
        self.mlp_input_dim = self.mlp_encoder.input_dim if self.mlp_encoder is not None else None
        self.cnn_output_dim = self.cnn_encoder.output_dim if self.cnn_encoder is not None else 0
        self.mlp_output_dim = self.mlp_encoder.output_dim if self.mlp_encoder is not None else 0
        self.output_dim = self.cnn_output_dim + self.mlp_output_dim
```

Validates that at least one sub-encoder was supplied; each supplied sub-encoder must expose `input_dim` and `output_dim` attributes (so `MultiEncoder` can advertise a single concatenated `output_dim` without re-running a probe). Sets `has_cnn_encoder` / `has_mlp_encoder` / `has_both_encoders` flags used by `forward`. Importantly, the **sub-encoders themselves** are responsible for selecting the correct dict keys; `MultiEncoder` only orchestrates concatenation.

---

## Line 458 — `MultiEncoder.cnn_keys` (property)

```python
    @property
    def cnn_keys(self) -> Sequence[str]:
        return self.cnn_encoder.keys if self.cnn_encoder is not None else []
```

Surfaces the image observation keys this encoder consumes (e.g., `["rgb", "depth"]`). Used by the algorithm to slice the replay buffer correctly.

---

## Line 462 — `MultiEncoder.mlp_keys` (property)

```python
    @property
    def mlp_keys(self) -> Sequence[str]:
        return self.mlp_encoder.keys if self.mlp_encoder is not None else []
```

Vector observation keys (e.g., `["state", "proprio"]`). Same role as `cnn_keys` — used by buffer slicing and obs-space introspection.

---

## Line 465 — `MultiEncoder.forward`

```python
    def forward(self, obs: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        if self.has_cnn_encoder:
            cnn_out = self.cnn_encoder(obs, *args, **kwargs)
        if self.has_mlp_encoder:
            mlp_out = self.mlp_encoder(obs, *args, **kwargs)
        if self.has_both_encoders:
            return torch.cat((cnn_out, mlp_out), -1)
        elif self.has_cnn_encoder:
            return cnn_out
        else:
            return mlp_out
```

Dispatch on flags: invoke whichever sub-encoders exist with the full obs dict (each sub-encoder pulls its own keys), then concatenate on the last axis if both are present. The `*args, **kwargs` pass-through lets the algorithm forward extra hints (e.g., training-mode flags) to either sub-encoder without `MultiEncoder` knowing about them.

---

## Line 478 — `class MultiDecoder`

```python
class MultiDecoder(nn.Module):
```

The symmetric output side. Takes the world model's flattened latent + recurrent state and reconstructs a dict of observations by dispatching to a CNN decoder (image reconstructions) and/or an MLP decoder (vector reconstructions). Returns a dict so the consumer (loss code in `loss.py`) can index per-key reconstruction distributions.

---

## Line 479 — `MultiDecoder.__init__`

```python
    def __init__(
        self,
        cnn_decoder: ModuleType,
        mlp_decoder: ModuleType,
    ) -> None:
        super().__init__()
        if cnn_decoder is None and mlp_decoder is None:
            raise ValueError("There must be an decoder, both cnn and mlp decoders are None")
        self.cnn_decoder = cnn_decoder
        self.mlp_decoder = mlp_decoder
```

Much smaller surface than `MultiEncoder.__init__` — `MultiDecoder` does not pre-compute concat dims (the decoder side splits outputs, it doesn't merge inputs). Just stores the two sub-decoders and asserts at least one exists.

---

## Line 491 — `MultiDecoder.cnn_keys` (property)

```python
    @property
    def cnn_keys(self) -> Sequence[str]:
        return self.cnn_decoder.keys if self.cnn_decoder is not None else []
```

Image reconstruction keys produced by this decoder — must match the encoder side so the per-key reconstruction loss can be paired with ground truth.

---

## Line 495 — `MultiDecoder.mlp_keys` (property)

```python
    @property
    def mlp_keys(self) -> Sequence[str]:
        return self.mlp_decoder.keys if self.mlp_decoder is not None else []
```

Vector reconstruction keys. Same role as `cnn_keys` for the MLP side.

---

## Line 498 — `MultiDecoder.forward`

```python
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        reconstructed_obs = {}
        if self.cnn_decoder is not None:
            reconstructed_obs.update(self.cnn_decoder(x))
        if self.mlp_decoder is not None:
            reconstructed_obs.update(self.mlp_decoder(x))
        return reconstructed_obs
```

Each sub-decoder is expected to return its own dict of key → reconstruction; `MultiDecoder` simply `.update()`s them into a single combined dict. No concatenation is needed here because outputs are already keyed. The convention is that sub-decoders emit distribution **parameters** (not samples) so the loss code can compute `log_prob` against the ground-truth observation.

---

## Line 507 — `class LayerNormChannelLast`

```python
class LayerNormChannelLast(nn.LayerNorm):
```

A LayerNorm variant for **NCHW** conv feature maps. Standard `nn.LayerNorm` normalises the last axis, but in NCHW that is the width — semantically wrong. This wrapper permutes to NHWC, applies the parent `nn.LayerNorm` (which now correctly normalises over channels), and permutes back. Used as `norm_layer` inside DreamerV3's image-encoder `CNN` with `eps=1e-3`.

---

## Line 508 — `LayerNormChannelLast.__init__`

```python
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
```

Pure delegation — accepts everything `nn.LayerNorm` does (notably `normalized_shape=channels` and `eps`). Exists only to make the class explicit for `isinstance` checks and to attach the custom forward below.

---

## Line 511 — `LayerNormChannelLast.forward`

```python
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (NCHW), received {len(x.shape)}D instead: {x.shape}")
        input_dtype = x.dtype
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.to(input_dtype)
```

Asserts 4-D input, saves the input dtype (so AMP / bfloat16 inputs survive a fp32 LayerNorm round-trip), permutes `NCHW → NHWC`, calls `nn.LayerNorm.forward` (which normalises over the last axis = channels), permutes back, and casts to the original dtype. The dtype dance is critical for mixed-precision training: `nn.LayerNorm` always runs internal stats in fp32, so without the cast the output would silently promote and break a `torch.compile` graph.

---

## Line 521 — `class LayerNorm`

```python
class LayerNorm(nn.LayerNorm):
```

A trivial subclass of `nn.LayerNorm` that adds **only** a dtype-preserving forward (no shape permute — for 1-D / 2-D / 3-D tensors where the last axis already is the feature axis). DreamerV3 uses this inside `LayerNormGRUCell` (`layer_norm_cls=LayerNorm`, `eps=1e-3`) and inside the MLP heads of the world-model / actor / critic when `norm_layer=LayerNorm` is requested.

---

## Line 522 — `LayerNorm.forward`

```python
    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        out = super().forward(x)
        return out.to(input_dtype)
```

Save dtype → call standard `nn.LayerNorm` → cast back. Same AMP-survival pattern as `LayerNormChannelLast.forward`. This is the LayerNorm the rest of the codebase imports when it wants Hafner's DreamerV3 `eps=1e-3` LayerNorm without breaking bf16 / fp16 training.
