---
title: "Sheeprl Reference: buffers.py"
source: tmp/sheeprl/sheeprl/data/buffers.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `buffers.py`

> **Source**: `tmp/sheeprl/sheeprl/data/buffers.py` — 1180 lines, 55 def/class sections.
> **Purpose** (one-line): All replay-buffer machinery: `ReplayBuffer`, `SequentialReplayBuffer`, `EnvIndependentReplayBuffer`, `EpisodeBuffer`, sampling logic, memmap support, plus the `get_tensor` helper.
> **Imports from elsewhere in this index**: (none significant; mostly self-contained — depends only on `sheeprl.utils.memmap.MemmapArray` and `sheeprl.utils.utils.NUMPY_TO_TORCH_DTYPE_DICT`).

---

## Table of Contents

- [Lines 1–17 — Imports](#lines-117--imports)
- [Line 20 — `class ReplayBuffer`](#line-20--class-replaybuffer)
- [Line 23 — `ReplayBuffer.__init__`](#line-23--replaybuffer__init__)
- [Line 82 — `ReplayBuffer.buffer` (property)](#line-82--replaybufferbuffer-property)
- [Line 86 — `ReplayBuffer.buffer_size` (property)](#line-86--replaybufferbuffer_size-property)
- [Line 90 — `ReplayBuffer.full` (property)](#line-90--replaybufferfull-property)
- [Line 94 — `ReplayBuffer.n_envs` (property)](#line-94--replaybuffern_envs-property)
- [Line 98 — `ReplayBuffer.empty` (property)](#line-98--replaybufferempty-property)
- [Line 102 — `ReplayBuffer.is_memmap` (property)](#line-102--replaybufferis_memmap-property)
- [Line 105 — `ReplayBuffer.__len__`](#line-105--replaybuffer__len__)
- [Line 109 — `ReplayBuffer.to_tensor`](#line-109--replaybufferto_tensor)
- [Line 138 — `ReplayBuffer.add` (overload 1)](#line-138--replaybufferadd-overload-1)
- [Line 142 — `ReplayBuffer.add` (overload 2)](#line-142--replaybufferadd-overload-2)
- [Line 145 — `ReplayBuffer.add` (implementation)](#line-145--replaybufferadd-implementation)
- [Line 223 — `ReplayBuffer.sample`](#line-223--replaybuffersample)
- [Line 270 — `ReplayBuffer._get_samples`](#line-270--replaybuffer_get_samples)
- [Line 291 — `ReplayBuffer.sample_tensors`](#line-291--replaybuffersample_tensors)
- [Line 328 — `ReplayBuffer.__getitem__`](#line-328--replaybuffer__getitem__)
- [Line 335 — `ReplayBuffer.__setitem__`](#line-335--replaybuffer__setitem__)
- [Line 363 — `class SequentialReplayBuffer`](#line-363--class-sequentialreplaybuffer)
- [Line 366 — `SequentialReplayBuffer.__init__`](#line-366--sequentialreplaybuffer__init__)
- [Line 395 — `SequentialReplayBuffer.sample`](#line-395--sequentialreplaybuffersample)
- [Line 467 — `SequentialReplayBuffer._get_samples`](#line-467--sequentialreplaybuffer_get_samples)
- [Line 529 — `class EnvIndependentReplayBuffer`](#line-529--class-envindependentreplaybuffer)
- [Line 530 — `EnvIndependentReplayBuffer.__init__`](#line-530--envindependentreplaybuffer__init__)
- [Line 593 — `EnvIndependentReplayBuffer.buffer` (property)](#line-593--envindependentreplaybufferbuffer-property)
- [Line 597 — `EnvIndependentReplayBuffer.buffer_size` (property)](#line-597--envindependentreplaybufferbuffer_size-property)
- [Line 601 — `EnvIndependentReplayBuffer.full` (property)](#line-601--envindependentreplaybufferfull-property)
- [Line 605 — `EnvIndependentReplayBuffer.n_envs` (property)](#line-605--envindependentreplaybuffern_envs-property)
- [Line 609 — `EnvIndependentReplayBuffer.empty` (property)](#line-609--envindependentreplaybufferempty-property)
- [Line 613 — `EnvIndependentReplayBuffer.is_memmap` (property)](#line-613--envindependentreplaybufferis_memmap-property)
- [Line 616 — `EnvIndependentReplayBuffer.__len__`](#line-616--envindependentreplaybuffer__len__)
- [Line 620 — `EnvIndependentReplayBuffer.add` (overload 1)](#line-620--envindependentreplaybufferadd-overload-1)
- [Line 624 — `EnvIndependentReplayBuffer.add` (overload 2)](#line-624--envindependentreplaybufferadd-overload-2)
- [Line 627 — `EnvIndependentReplayBuffer.add` (implementation)](#line-627--envindependentreplaybufferadd-implementation)
- [Line 656 — `EnvIndependentReplayBuffer.sample`](#line-656--envindependentreplaybuffersample)
- [Line 702 — `EnvIndependentReplayBuffer.sample_tensors`](#line-702--envindependentreplaybuffersample_tensors)
- [Line 746 — `class EpisodeBuffer`](#line-746--class-episodebuffer)
- [Line 770 — `EpisodeBuffer.__init__`](#line-770--episodebuffer__init__)
- [Line 824 — `EpisodeBuffer.prioritize_ends` (property)](#line-824--episodebufferprioritize_ends-property)
- [Line 828 — `EpisodeBuffer.prioritize_ends` (setter)](#line-828--episodebufferprioritize_ends-setter)
- [Line 832 — `EpisodeBuffer.buffer` (property)](#line-832--episodebufferbuffer-property)
- [Line 836 — `EpisodeBuffer.obs_keys` (property)](#line-836--episodebufferobs_keys-property)
- [Line 840 — `EpisodeBuffer.n_envs` (property)](#line-840--episodebuffern_envs-property)
- [Line 844 — `EpisodeBuffer.buffer_size` (property)](#line-844--episodebufferbuffer_size-property)
- [Line 848 — `EpisodeBuffer.minimum_episode_length` (property)](#line-848--episodebufferminimum_episode_length-property)
- [Line 852 — `EpisodeBuffer.is_memmap` (property)](#line-852--episodebufferis_memmap-property)
- [Line 856 — `EpisodeBuffer.full` (property)](#line-856--episodebufferfull-property)
- [Line 859 — `EpisodeBuffer.__len__`](#line-859--episodebuffer__len__)
- [Line 863 — `EpisodeBuffer.add` (overload 1)](#line-863--episodebufferadd-overload-1)
- [Line 867 — `EpisodeBuffer.add` (overload 2)](#line-867--episodebufferadd-overload-2)
- [Line 875 — `EpisodeBuffer.add` (implementation)](#line-875--episodebufferadd-implementation)
- [Line 971 — `EpisodeBuffer._save_episode`](#line-971--episodebuffer_save_episode)
- [Line 1033 — `EpisodeBuffer.sample`](#line-1033--episodebuffersample)
- [Line 1123 — `EpisodeBuffer.sample_tensors`](#line-1123--episodebuffersample_tensors)
- [Line 1158 — `get_tensor`](#line-1158--get_tensor)

---

## Lines 1–17 — Imports

```python
from __future__ import annotations

import logging
import os
import shutil
import typing
import uuid
from itertools import compress
from pathlib import Path
from typing import Dict, Optional, Sequence, Type

import numpy as np
import torch
from torch import Tensor

from sheeprl.utils.memmap import MemmapArray
from sheeprl.utils.utils import NUMPY_TO_TORCH_DTYPE_DICT
```

**What it does**: Brings in stdlib utilities (`logging`, `os`, `shutil`, `uuid`, `compress`, `Path`), typing helpers, plus numpy and torch. The two project-local imports are `MemmapArray` (the disk-backed numpy-array wrapper that powers checkpointable buffers) and `NUMPY_TO_TORCH_DTYPE_DICT` (used by `get_tensor` to map numpy dtypes to torch dtypes when converting sampled arrays). `from __future__ import annotations` enables PEP 604 union syntax (`X | Y`) in annotations on older Python.

---

## Line 20 — `class ReplayBuffer`

```python
class ReplayBuffer:
    batch_axis: int = 1
    ...
```

**What it does**: Standard FIFO ring-buffer holding transitions for one or more environments. Internally a `Dict[str, np.ndarray | MemmapArray]` keyed by field name (`observations`, `actions`, `rewards`, …). Each array has shape `[buffer_size, n_envs, ...]`. Designed for flat (non-sequence) sampling — DreamerV3 itself uses the sequential subclass, but this class supplies the shared add/sample/memmap plumbing. The class attribute `batch_axis: int = 1` (line 21) tells `EnvIndependentReplayBuffer` along which axis to concatenate per-env sample dictionaries — for flat samples shaped `[n_samples, batch_size, ...]` the batch axis is `1`.

---

## Line 23 — `ReplayBuffer.__init__`

```python
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
        **kwargs,
    ):
        """A standard replay buffer implementation. Internally this is represented by a
        dictionary mapping string to numpy arrays. The first dimension of the arrays is the
        buffer size, while the second dimension is the number of environments.
        ...
        """
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        self._obs_keys = obs_keys
        self._memmap = memmap
        self._memmap_dir = memmap_dir
        self._memmap_mode = memmap_mode
        self._buf: Dict[str, np.ndarray | MemmapArray] = {}
        if self._memmap:
            if self._memmap_mode not in ("r+", "w+", "c", "copyonwrite", "readwrite", "write"):
                raise ValueError(
                    'Accepted values for memmap_mode are "r+", "readwrite", "w+", "write", "c" or '
                    '"copyonwrite". PyTorch does not support tensors backed by read-only '
                    'NumPy arrays, so "r" and "readonly" are not supported.'
                )
            if self._memmap_dir is None:
                raise ValueError(
                    "The buffer is set to be memory-mapped but the 'memmap_dir' attribute is None. "
                    "Set the 'memmap_dir' to a known directory.",
                )
            else:
                self._memmap_dir = Path(self._memmap_dir)
                self._memmap_dir.mkdir(parents=True, exist_ok=True)
        self._pos = 0
        self._full = False
        self._memmap_specs = {}
        self._rng: np.random.Generator = np.random.default_rng()
```

**What it does**: Stores capacity, env count, obs-key list, and memmap flags. Validates that memmap modes are writable (read-only modes are rejected because PyTorch refuses read-only-numpy-backed tensors) and creates the memmap directory if asked. Initialises `_pos` (ring-buffer write pointer), `_full` (set when wrap-around first happens), and a numpy `Generator` used by all sampling. The actual per-key arrays live in `self._buf` and are allocated lazily on the first `add()` call.

---

## Line 82 — `ReplayBuffer.buffer` (property)

```python
    @property
    def buffer(self) -> Dict[str, np.ndarray]:
        return self._buf
```

**What it does**: Returns the underlying dict mapping field name to its `[buffer_size, n_envs, ...]` array (numpy or memmap). Read-only accessor for use from outside; mutation still happens in-place via `add`/`__setitem__`.

---

## Line 86 — `ReplayBuffer.buffer_size` (property)

```python
    @property
    def buffer_size(self) -> int:
        return self._buffer_size
```

**What it does**: Returns the configured capacity (first dim of every stored array). Distinct from `__len__`, which also returns `buffer_size`; both currently report capacity rather than fill-level.

---

## Line 90 — `ReplayBuffer.full` (property)

```python
    @property
    def full(self) -> bool:
        return self._full
```

**What it does**: True once the ring buffer has wrapped at least once. Used by `sample` to decide whether to sample over `[0, buffer_size)` or only `[0, _pos)`.

---

## Line 94 — `ReplayBuffer.n_envs` (property)

```python
    @property
    def n_envs(self) -> int:
        return self._n_envs
```

**What it does**: Returns the second-dim size (number of parallel envs whose transitions share this buffer).

---

## Line 98 — `ReplayBuffer.empty` (property)

```python
    @property
    def empty(self) -> bool:
        return (self.buffer is not None and len(self.buffer) == 0) or self.buffer is None
```

**What it does**: True before any `add()` call — at that point `_buf` is still the empty dict initialised in `__init__`, since per-key arrays are lazily allocated.

---

## Line 102 — `ReplayBuffer.is_memmap` (property)

```python
    @property
    def is_memmap(self) -> bool:
        return self._memmap
```

**What it does**: Exposes the memmap flag set at construction time. Memmap mode means the buffer can be checkpointed and resumed from disk without re-allocating RAM.

---

## Line 105 — `ReplayBuffer.__len__`

```python
    def __len__(self) -> int:
        return self.buffer_size
```

**What it does**: Reports the capacity, not the live fill-level. Code that needs "how many transitions have I actually added?" should check `_pos` and `_full` directly; `len()` here is closer to "max addressable slot".

---

## Line 109 — `ReplayBuffer.to_tensor`

```python
    @torch.no_grad()
    def to_tensor(
        self,
        dtype: Optional[torch.dtype] = None,
        clone: bool = False,
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
    ) -> Dict[str, Tensor]:
        """Converts the replay buffer to a dictionary mapping string to torch.Tensor.
        ...
        """
        buf = {}
        for k, v in self.buffer.items():
            buf[k] = get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy)
        return buf
```

**What it does**: Walks every key in `_buf` and converts the underlying numpy / memmap array to a torch tensor via the module-level `get_tensor`. `@torch.no_grad()` strips autograd because the buffer is data, not a computation node. Mostly used for debugging / inspection — the hot path uses `sample_tensors`.

---

## Line 138 — `ReplayBuffer.add` (overload 1)

```python
    @typing.overload
    def add(self, data: "ReplayBuffer", validate_args: bool = False) -> None:
        ...
```

**What it does**: Type-checker-only signature declaring that `add()` accepts another `ReplayBuffer` (its `.buffer` dict is extracted inside the implementation). No runtime body — `typing.overload` is consumed by static analysers only.

---

## Line 142 — `ReplayBuffer.add` (overload 2)

```python
    @typing.overload
    def add(self, data: Dict[str, np.ndarray], validate_args: bool = False) -> None:
        ...
```

**What it does**: Second overload — `data` may be a raw dict of numpy arrays shaped `[sequence_length, n_envs, ...]`. Together overloads 1 and 2 give the IDE the union of accepted call shapes.

---

## Line 145 — `ReplayBuffer.add` (implementation)

```python
    def add(self, data: "ReplayBuffer" | Dict[str, np.ndarray], validate_args: bool = False) -> None:
        """Add data to the replay buffer. If the replay buffer is full, then the oldest data is overwritten.
        ...
        """
        if isinstance(data, ReplayBuffer):
            data = data.buffer
        if validate_args:
            if not isinstance(data, dict):
                raise ValueError(...)
            elif isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(v, np.ndarray):
                        raise ValueError(...)
            last_key = next(iter(data.keys()))
            last_batch_shape = next(iter(data.values())).shape[:2]
            for i, (k, v) in enumerate(data.items()):
                if len(v.shape) < 2:
                    raise RuntimeError(...)
                if i > 0:
                    current_key = k
                    current_batch_shape = v.shape[:2]
                    if current_batch_shape != last_batch_shape:
                        raise RuntimeError(...)
                    last_key = current_key
                    last_batch_shape = current_batch_shape
        data_len = next(iter(data.values())).shape[0]
        next_pos = (self._pos + data_len) % self._buffer_size
        if next_pos <= self._pos or (data_len > self._buffer_size and not self._full):
            idxes = np.array(list(range(self._pos, self._buffer_size)) + list(range(0, next_pos)))
        else:
            idxes = np.array(range(self._pos, next_pos))
        if data_len > self._buffer_size:
            data_to_store = {k: v[-self._buffer_size - next_pos :] for k, v in data.items()}
        else:
            data_to_store = data
        if self._memmap and self.empty:
            for k, v in data_to_store.items():
                self.buffer[k] = MemmapArray(
                    filename=Path(self._memmap_dir / f"{k}.memmap"),
                    dtype=v.dtype,
                    shape=(self._buffer_size, self._n_envs, *v.shape[2:]),
                    mode=self._memmap_mode,
                )
                self.buffer[k][idxes] = data_to_store[k]
        elif self.empty:
            for k, v in data_to_store.items():
                self.buffer[k] = np.empty(shape=(self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype)
                self.buffer[k][idxes] = data_to_store[k]
        else:
            for k, v in data_to_store.items():
                self.buffer[k][idxes] = data_to_store[k]
        if self._pos + data_len >= self._buffer_size:
            self._full = True
        self._pos = next_pos
```

**What it does**: The ring-buffer write path. Optional `validate_args` enforces shape congruence (`[T, n_envs, ...]` across every key). Computes `idxes` — the absolute slot indices into `[0, buffer_size)` accounting for wrap-around — and on first call allocates each key as either a `MemmapArray` (memmap mode) or `np.empty`. Subsequent calls scatter `data` into the existing arrays. Sets `_full=True` once the write would touch slot `buffer_size`, and advances `_pos` modulo capacity.

---

## Line 223 — `ReplayBuffer.sample`

```python
    def sample(
        self, batch_size: int, sample_next_obs: bool = False, clone: bool = False, n_samples: int = 1, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the replay buffer. If the replay buffer is not full, then the samples are taken
        from the first 'self.pos' elements. Otherwise, the samples are taken from all the elements.
        When 'sample_next_obs' is True we sample until 'self.pos - 1' to avoid sampling the last observation,
        which would be invalid.
        ...
        """
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(...)
        if not self._full and self._pos == 0:
            raise ValueError(...)
        if self._full:
            first_range_end = self._pos - 1 if sample_next_obs else self._pos
            second_range_end = self.buffer_size if first_range_end >= 0 else self.buffer_size + first_range_end
            valid_idxes = np.array(
                list(range(0, first_range_end)) + list(range(self._pos, second_range_end)), dtype=np.intp
            )
            batch_idxes = valid_idxes[
                self._rng.integers(0, len(valid_idxes), size=(batch_size * n_samples,), dtype=np.intp)
            ]
        else:
            max_pos_to_sample = self._pos - 1 if sample_next_obs else self._pos
            if max_pos_to_sample == 0:
                raise RuntimeError(...)
            batch_idxes = self._rng.integers(0, max_pos_to_sample, size=(batch_size * n_samples,), dtype=np.intp)
        return {
            k: v.reshape(n_samples, batch_size, *v.shape[1:])
            for k, v in self._get_samples(batch_idxes=batch_idxes, sample_next_obs=sample_next_obs, clone=clone).items()
        }
```

**What it does**: Builds a list of valid timestep indices excluding the slot at `_pos` (the "current write head" boundary that would yield a fake transition straddling the wrap). With `sample_next_obs=True` the index just before `_pos` is also excluded so the `next_obs` lookup doesn't read across the boundary. Picks `batch_size * n_samples` random indices, delegates to `_get_samples`, and reshapes each value back to `[n_samples, batch_size, ...]`.

---

## Line 270 — `ReplayBuffer._get_samples`

```python
    def _get_samples(
        self, batch_idxes: np.ndarray, sample_next_obs: bool = False, clone: bool = False
    ) -> Dict[str, np.ndarray]:
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        samples: Dict[str, np.ndarray] = {}
        env_idxes = self._rng.integers(0, self.n_envs, size=(len(batch_idxes),), dtype=np.intp)
        flattened_idxes = (batch_idxes * self.n_envs + env_idxes).flat
        if sample_next_obs:
            flattened_next_idxes = (((batch_idxes + 1) % self._buffer_size) * self.n_envs + env_idxes).flat
        for k, v in self.buffer.items():
            samples[k] = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
            if clone:
                samples[k] = samples[k].copy()
            if k in self._obs_keys and sample_next_obs:
                samples[f"next_{k}"] = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_next_idxes, axis=0)
                if clone:
                    samples[f"next_{k}"] = samples[f"next_{k}"].copy()
        return samples
```

**What it does**: Each batch element gets a uniformly random env index, fused with its time index into a flat `time * n_envs + env` linear index, and `np.take` pulls the rows out of a `[buffer_size * n_envs, ...]` reshape of each array. For `obs_keys` and `sample_next_obs=True`, also indexes `t+1` (with modulo wrap) to produce `next_<k>` entries. `clone=True` forces an independent copy (otherwise memmap reads stay backed by the file).

---

## Line 291 — `ReplayBuffer.sample_tensors`

```python
    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        clone: bool = False,
        sample_next_obs: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Sample elements from the replay buffer and convert them to torch tensors.
        ...
        """
        n_samples = kwargs.pop("n_samples", 1)
        samples = self.sample(
            batch_size=batch_size, sample_next_obs=sample_next_obs, clone=clone, n_samples=n_samples, **kwargs
        )
        return {
            k: get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy) for k, v in samples.items()
        }
```

**What it does**: Thin torch-side wrapper around `sample`. Calls `sample`, then converts each numpy/memmap value to a tensor on the requested `device`/`dtype` via `get_tensor`. Used by trainers as the primary entry point.

---

## Line 328 — `ReplayBuffer.__getitem__`

```python
    def __getitem__(self, key: str) -> np.ndarray | np.memmap | MemmapArray:
        if not isinstance(key, str):
            raise TypeError("'key' must be a string")
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        return self.buffer.get(key)
```

**What it does**: Dict-style key access — `buf["rewards"]` returns the full `[buffer_size, n_envs, ...]` array. Rejects non-string keys and an empty buffer so callers fail loudly rather than getting silent `None`s for typos.

---

## Line 335 — `ReplayBuffer.__setitem__`

```python
    def __setitem__(self, key: str, value: np.ndarray | np.memmap | MemmapArray) -> None:
        if not isinstance(value, (np.ndarray, MemmapArray)):
            raise ValueError(...)
        if self.empty:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        if value.shape[:2] != (self._buffer_size, self._n_envs):
            raise RuntimeError(
                "'value' must have at least two dimensions of dimension [buffer_size, n_envs, ...]. "
                f"Shape of 'value' is {value.shape}"
            )
        if self._memmap:
            if isinstance(value, np.ndarray):
                filename = Path(self._memmap_dir / f"{key}.memmap")
            elif isinstance(value, MemmapArray):
                filename = value.filename
            value_to_add = MemmapArray.from_array(value, filename=filename, mode=self._memmap_mode)
        else:
            if isinstance(value, np.ndarray):
                value_to_add = np.copy(value)
            elif isinstance(value, MemmapArray):
                value_to_add = np.copy(value.array)
        self.buffer.update({key: value_to_add})
```

**What it does**: Bulk-replace a whole field array. Enforces `shape[:2] == (buffer_size, n_envs)`. In memmap mode wraps the supplied array as a fresh `MemmapArray` (writing to `<memmap_dir>/<key>.memmap`); otherwise stores an independent `np.copy`. Mostly used for resuming a buffer from a checkpoint.

---

## Line 363 — `class SequentialReplayBuffer`

```python
class SequentialReplayBuffer(ReplayBuffer):
    batch_axis: int = 2
    ...
```

**What it does**: Subclass that samples contiguous time slices of length `sequence_length` (e.g. 64 in the smoke run, configured via `per_rank_sequence_length`). Reuses `ReplayBuffer.add` and storage layout; only `sample` / `_get_samples` change. This is the buffer **DreamerV3 actually consumes** for world-model training. The class attribute `batch_axis: int = 2` (line 364) overrides the parent's `1` because sequential samples have shape `[n_samples, sequence_length, batch_size, ...]`.

---

## Line 366 — `SequentialReplayBuffer.__init__`

```python
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
        **kwargs,
    ):
        """A sequential replay buffer implementation. ...
        """
        super().__init__(buffer_size, n_envs, obs_keys, memmap, memmap_dir, memmap_mode, **kwargs)
```

**What it does**: Pure delegation to `ReplayBuffer.__init__`. Storage layout is identical; only the sampling logic differs. The dedicated `__init__` exists mostly for the subclass-specific docstring.

---

## Line 395 — `SequentialReplayBuffer.sample`

```python
    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        sequence_length: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        ...
        """
        batch_dim = batch_size * n_samples
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(...)
        if not self.full and self._pos == 0:
            raise ValueError(...)
        if self._buf is None:
            raise RuntimeError(...)
        if not self.full and self._pos - sequence_length + 1 < 1:
            raise ValueError(f"Cannot sample a sequence of length {sequence_length}. Data added so far: {self._pos}")
        if self.full and sequence_length > self.__len__():
            raise ValueError(...)
        if self.full:
            first_range_end = self._pos - sequence_length + 1
            second_range_end = self.buffer_size if first_range_end >= 0 else self.buffer_size + first_range_end
            valid_idxes = np.array(
                list(range(0, first_range_end)) + list(range(self._pos, second_range_end)), dtype=np.intp
            )
            start_idxes = valid_idxes[self._rng.integers(0, len(valid_idxes), size=(batch_dim,), dtype=np.intp)]
        else:
            start_idxes = self._rng.integers(0, self._pos - sequence_length + 1, size=(batch_dim,), dtype=np.intp)
        chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(1, -1)
        idxes = (start_idxes.reshape(-1, 1) + chunk_length) % self.buffer_size
        return self._get_samples(
            idxes, batch_size, n_samples, sequence_length, sample_next_obs=sample_next_obs, clone=clone
        )
```

**What it does**: **Crucially, this does NOT respect episode boundaries** — a sampled length-L chunk may straddle a `done`. DreamerV3 relies on the stored `is_first` flag to reset the recurrent state at the right step inside the chunk. Valid start indices are restricted so the chunk `[start, start+L)` does not cross the write-head `_pos`. `chunk_length` broadcasts `(0,1,...,L-1)` against `start_idxes` to produce a `[batch_dim, sequence_length]` index matrix, wrapped modulo `buffer_size`.

---

## Line 467 — `SequentialReplayBuffer._get_samples`

```python
    def _get_samples(
        self,
        batch_idxes: np.ndarray,
        batch_size: int,
        n_samples: int,
        sequence_length: int,
        sample_next_obs: bool = False,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        batch_shape = (batch_size * n_samples, sequence_length)
        flattened_batch_idxes = np.ravel(batch_idxes)
        if self._n_envs == 1:
            env_idxes = np.zeros((np.prod(batch_shape),), dtype=np.intp)
        else:
            env_idxes = self._rng.integers(0, self.n_envs, size=(batch_shape[0],), dtype=np.intp)
            env_idxes = np.reshape(env_idxes, (-1, 1))
            env_idxes = np.tile(env_idxes, (1, sequence_length))
            env_idxes = np.ravel(env_idxes)
        flattened_idxes = (flattened_batch_idxes * self._n_envs + env_idxes).flat
        samples: Dict[str, np.ndarray] = {}
        for k, v in self.buffer.items():
            flattened_v = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
            batched_v = np.reshape(flattened_v, (n_samples, batch_size, sequence_length) + flattened_v.shape[1:])
            samples[k] = np.swapaxes(batched_v, axis1=1, axis2=2)
            if clone:
                samples[k] = samples[k].copy()
            if sample_next_obs:
                flattened_next_v = v[(flattened_batch_idxes + 1) % self._buffer_size, env_idxes]
                batched_next_v = np.reshape(
                    flattened_next_v, (n_samples, batch_size, sequence_length) + flattened_next_v.shape[1:]
                )
                samples[f"next_{k}"] = np.swapaxes(batched_next_v, axis1=1, axis2=2)
                if clone:
                    samples[f"next_{k}"] = samples[f"next_{k}"].copy()
        return samples
```

**What it does**: **Critical invariant: every step in a sequence shares the same env index** — `env_idxes` is drawn once per chunk (`batch_shape[0]`) then tiled `sequence_length` times. This is what guarantees that within a sampled length-L window the underlying time-series is from one env's trajectory. The flat-index trick (`t * n_envs + env`) lets a single `np.take` extract all `batch_dim * L` rows in one shot. Final layout is `[n_samples, sequence_length, batch_size, ...]` after the `swapaxes`.

---

## Line 529 — `class EnvIndependentReplayBuffer`

```python
class EnvIndependentReplayBuffer:
    ...
```

**What it does**: Wraps `N` independent sub-buffers (one per env), each constructed as `buffer_cls(n_envs=1, ...)`. Lets `dreamer-v3` keep per-env ring buffers — important when the N parallel envs have different episode lengths and you want sampled sequences to come from a single episode/env, not be smeared across envs. This is the buffer the DreamerV3 main loop actually instantiates (with `buffer_cls=SequentialReplayBuffer`).

---

## Line 530 — `EnvIndependentReplayBuffer.__init__`

```python
    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
        buffer_cls: Type[ReplayBuffer] = ReplayBuffer,
        **kwargs,
    ):
        """A replay buffer implementation that is composed of multiple independent replay buffers.
        ...
        """
        if buffer_size <= 0:
            raise ValueError(...)
        if n_envs <= 0:
            raise ValueError(...)
        if memmap:
            if memmap_mode not in ("r+", "w+", "c", "copyonwrite", "readwrite", "write"):
                raise ValueError(...)
            if memmap_dir is None:
                raise ValueError(...)
            else:
                memmap_dir = Path(memmap_dir)
                memmap_dir.mkdir(parents=True, exist_ok=True)
        self._buf: Sequence[ReplayBuffer] = [
            buffer_cls(
                buffer_size=buffer_size,
                n_envs=1,
                obs_keys=obs_keys,
                memmap=memmap,
                memmap_dir=memmap_dir / f"env_{i}" if memmap else None,
                memmap_mode=memmap_mode,
                **kwargs,
            )
            for i in range(n_envs)
        ]
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        self._rng: np.random.Generator = np.random.default_rng()
        self._concat_along_axis = buffer_cls.batch_axis
```

**What it does**: Validates inputs, builds `n_envs` sub-buffers each with `n_envs=1`, and gives each its own subdir (`<memmap_dir>/env_i/`) when memmapping. Caches `buffer_cls.batch_axis` as `_concat_along_axis` so `sample` knows where to concatenate per-env samples (1 for flat, 2 for sequential).

---

## Line 593 — `EnvIndependentReplayBuffer.buffer` (property)

```python
    @property
    def buffer(self) -> Sequence[ReplayBuffer]:
        return tuple(self._buf)
```

**What it does**: Returns the tuple of underlying sub-buffers (immutable view). Callers can drill into a single env's buffer via `buffer[env_idx]`.

---

## Line 597 — `EnvIndependentReplayBuffer.buffer_size` (property)

```python
    @property
    def buffer_size(self) -> int:
        return self._buffer_size
```

**What it does**: Capacity **per sub-buffer**, not the aggregate across envs. Total storage is `buffer_size * n_envs` transitions.

---

## Line 601 — `EnvIndependentReplayBuffer.full` (property)

```python
    @property
    def full(self) -> Sequence[bool]:
        return tuple([b.full for b in self.buffer])
```

**What it does**: Returns a per-env tuple of "has this sub-buffer wrapped at least once?" booleans. Each env fills at its own rate, so this is intentionally per-env, not aggregate.

---

## Line 605 — `EnvIndependentReplayBuffer.n_envs` (property)

```python
    @property
    def n_envs(self) -> int:
        return self._n_envs
```

**What it does**: Number of sub-buffers (= number of parallel envs).

---

## Line 609 — `EnvIndependentReplayBuffer.empty` (property)

```python
    @property
    def empty(self) -> Sequence[bool]:
        return tuple([b.empty for b in self.buffer])
```

**What it does**: Per-env emptiness flags. A given env is empty until its sub-buffer has received at least one `add()`.

---

## Line 613 — `EnvIndependentReplayBuffer.is_memmap` (property)

```python
    @property
    def is_memmap(self) -> Sequence[bool]:
        return tuple([b.is_memmap for b in self.buffer])
```

**What it does**: Per-env memmap flags. In normal use they're all True or all False, but the API exposes them per-env.

---

## Line 616 — `EnvIndependentReplayBuffer.__len__`

```python
    def __len__(self) -> int:
        return self.buffer_size
```

**What it does**: Returns the per-sub-buffer capacity. Same caveat as the base class — capacity, not fill level.

---

## Line 620 — `EnvIndependentReplayBuffer.add` (overload 1)

```python
    @typing.overload
    def add(self, data: "ReplayBuffer", validate_args: bool = False) -> None:
        ...
```

**What it does**: Type-checker overload for the `ReplayBuffer` argument form. Note this overload signature is incomplete relative to the implementation — it omits `indices`, but mypy will still see the implementation signature.

---

## Line 624 — `EnvIndependentReplayBuffer.add` (overload 2)

```python
    @typing.overload
    def add(self, data: Dict[str, np.ndarray], validate_args: bool = False) -> None:
        ...
```

**What it does**: Type-checker overload for the dict-of-arrays argument form.

---

## Line 627 — `EnvIndependentReplayBuffer.add` (implementation)

```python
    def add(
        self,
        data: "ReplayBuffer" | Dict[str, np.ndarray],
        indices: Optional[Sequence[int]] = None,
        validate_args: bool = False,
    ) -> None:
        """Add data to the replay buffers specified by the 'indices'. If 'indices' is None, then the data is added
        one for every environment. ...
        """
        if indices is None:
            indices = tuple(range(self.n_envs))
        elif len(indices) != next(iter(data.values())).shape[1]:
            raise ValueError(...)
        for env_data_idx, env_idx in enumerate(indices):
            env_data = {k: v[:, env_data_idx : env_data_idx + 1] for k, v in data.items()}
            self._buf[env_idx].add(env_data, validate_args=validate_args)
```

**What it does**: Splits a `[T, n_envs, ...]` data dict column-wise per env and routes each column to its own sub-buffer. The `indices` argument lets the caller add only a subset of envs (useful if only some envs stepped). Each sub-buffer sees a `[T, 1, ...]` slice — its private `n_envs=1`.

---

## Line 656 — `EnvIndependentReplayBuffer.sample`

```python
    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Samples data from the buffer. ...
        """
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(...)
        if self._buf is None:
            raise RuntimeError(...)
        bs_per_buf = np.bincount(self._rng.integers(0, self._n_envs, (batch_size,)))
        per_buf_samples = [
            b.sample(
                batch_size=bs,
                sample_next_obs=sample_next_obs,
                clone=clone,
                n_samples=n_samples,
                **kwargs,
            )
            for b, bs in zip(self._buf, bs_per_buf)
            if bs > 0
        ]
        samples = {}
        for k in per_buf_samples[0].keys():
            samples[k] = np.concatenate([s[k] for s in per_buf_samples], axis=self._concat_along_axis)
        return samples
```

**What it does**: Uses `np.bincount` over `batch_size` uniformly-drawn env IDs to split the batch across sub-buffers (each env gets `bs_per_buf[i]` rows on average). Each sub-buffer produces its own samples, and the per-env dicts are concatenated along `_concat_along_axis` (1 for flat, 2 for sequential) so the final batch interleaves envs along the batch dimension. Sub-buffers that drew zero are skipped.

---

## Line 702 — `EnvIndependentReplayBuffer.sample_tensors`

```python
    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Sample elements from the replay buffer and convert them to torch tensors.
        ...
        """
        samples = self.sample(
            batch_size=batch_size,
            sample_next_obs=sample_next_obs,
            clone=clone,
            n_samples=n_samples,
            **kwargs,
        )
        return {
            k: get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy) for k, v in samples.items()
        }
```

**What it does**: Mirror of `ReplayBuffer.sample_tensors` — calls `sample` and converts the resulting dict to torch tensors via `get_tensor`. This is the entry point DreamerV3's trainer calls each gradient step.

---

## Line 746 — `class EpisodeBuffer`

```python
class EpisodeBuffer:
    """A replay buffer that stores separately the episodes.
    ...
    """
    batch_axis: int = 2
    ...
```

**What it does**: Alternative storage strategy — instead of a fixed ring buffer over transitions, stores **complete episodes** (each as a `Dict[str, np.ndarray | MemmapArray]`). Sampled sequences are then drawn from inside a single episode, so they never cross a `done` boundary. Capacity is in transition-steps; oldest episodes are evicted FIFO when adding a new one would overflow. Supports `prioritize_ends` (Dreamer-style sampling that over-weights the tail of each episode). The class attribute `batch_axis: int = 2` (line 768) matches `SequentialReplayBuffer` because samples are `[n_samples, sequence_length, batch_size, ...]`.

---

## Line 770 — `EpisodeBuffer.__init__`

```python
    def __init__(
        self,
        buffer_size: int,
        minimum_episode_length: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        prioritize_ends: bool = False,
        memmap: bool = False,
        memmap_dir: str | os.PathLike | None = None,
        memmap_mode: str = "r+",
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(...)
        if minimum_episode_length <= 0:
            raise ValueError(...)
        if buffer_size < minimum_episode_length:
            raise ValueError(...)
        self._n_envs = n_envs
        self._obs_keys = obs_keys
        self._buffer_size = buffer_size
        self._minimum_episode_length = minimum_episode_length
        self._prioritize_ends = prioritize_ends
        self._open_episodes = [[] for _ in range(n_envs)]
        self._cum_lengths: Sequence[int] = []
        self._buf: Sequence[Dict[str, np.ndarray | MemmapArray]] = []
        self._memmap = memmap
        self._memmap_dir = memmap_dir
        self._memmap_mode = memmap_mode
        if self._memmap:
            if self._memmap_mode not in ("r+", "w+", "c", "copyonwrite", "readwrite", "write"):
                raise ValueError(...)
            if self._memmap_dir is None:
                raise ValueError(...)
            else:
                self._memmap_dir = Path(self._memmap_dir)
                self._memmap_dir.mkdir(parents=True, exist_ok=True)
```

**What it does**: Validates capacity / min-episode-length, then sets up two parallel data structures: `_open_episodes[env]` — list of in-progress chunks for each env (will be concatenated and saved once a `done` arrives) — and `_buf` — list of finished, saved episodes. `_cum_lengths` keeps the running total of saved transitions so `full` / sampling can be O(1) without iterating `_buf`. Memmap setup follows the same pattern as `ReplayBuffer`.

---

## Line 824 — `EpisodeBuffer.prioritize_ends` (property)

```python
    @property
    def prioritize_ends(self) -> bool:
        return self._prioritize_ends
```

**What it does**: Returns the prioritize-ends flag. Dreamer-style: when True, sampling lets the chunk start past `ep_len - sequence_length` so the chunk is "padded" at the tail with the final transition repeated (see `sample` for the exact mechanics).

---

## Line 828 — `EpisodeBuffer.prioritize_ends` (setter)

```python
    @prioritize_ends.setter
    def prioritize_ends(self, prioritize_ends: bool) -> None:
        self._prioritize_ends = prioritize_ends
```

**What it does**: Lets the trainer flip the flag at runtime (useful if e.g. only enabling end-prioritisation after some warm-up).

---

## Line 832 — `EpisodeBuffer.buffer` (property)

```python
    @property
    def buffer(self) -> Sequence[Dict[str, np.ndarray | MemmapArray]]:
        return self._buf
```

**What it does**: Returns the list of stored episodes. Each entry is a `Dict[str, array]` whose first axis is the episode length (variable across entries).

---

## Line 836 — `EpisodeBuffer.obs_keys` (property)

```python
    @property
    def obs_keys(self) -> Sequence[str]:
        return self._obs_keys
```

**What it does**: Returns the observation-key list — keys for which `next_<k>` rows are also emitted when sampling with `sample_next_obs=True`.

---

## Line 840 — `EpisodeBuffer.n_envs` (property)

```python
    @property
    def n_envs(self) -> int:
        return self._n_envs
```

**What it does**: Number of parallel envs feeding into this buffer. Each env has its own `_open_episodes[env]` slot.

---

## Line 844 — `EpisodeBuffer.buffer_size` (property)

```python
    @property
    def buffer_size(self) -> int:
        return self._buffer_size
```

**What it does**: Capacity in transitions across all episodes. Compared against `_cum_lengths[-1]` to decide when to evict.

---

## Line 848 — `EpisodeBuffer.minimum_episode_length` (property)

```python
    @property
    def minimum_episode_length(self) -> int:
        return self._minimum_episode_length
```

**What it does**: Returns the minimum length an episode must reach to be saved (shorter episodes raise on `_save_episode`). Lets sampling be confident that every episode in `_buf` is at least this long.

---

## Line 852 — `EpisodeBuffer.is_memmap` (property)

```python
    @property
    def is_memmap(self) -> bool:
        return self._memmap
```

**What it does**: Returns the memmap flag. Unlike `EnvIndependentReplayBuffer`, this is a single bool — `EpisodeBuffer` doesn't have per-env sub-buffers.

---

## Line 856 — `EpisodeBuffer.full` (property)

```python
    @property
    def full(self) -> bool:
        return self._cum_lengths[-1] + self._minimum_episode_length > self._buffer_size if len(self._buf) > 0 else False
```

**What it does**: True once adding even a minimum-length episode would overflow. The `+ minimum_episode_length` margin is the trigger that prompts `_save_episode` to evict the oldest stored episode(s) before appending a new one.

---

## Line 859 — `EpisodeBuffer.__len__`

```python
    def __len__(self) -> int:
        return self._cum_lengths[-1] if len(self._buf) > 0 else 0
```

**What it does**: Returns the total number of stored transition-steps (sum of all episode lengths). Unlike `ReplayBuffer.__len__`, this is the **fill level**, not the capacity.

---

## Line 863 — `EpisodeBuffer.add` (overload 1)

```python
    @typing.overload
    def add(self, data: "ReplayBuffer", env_idxes: Sequence[int] | None = None, validate_args: bool = False) -> None:
        ...
```

**What it does**: Type-checker overload — `add` accepts a `ReplayBuffer` (its `.buffer` dict is unpacked inside).

---

## Line 867 — `EpisodeBuffer.add` (overload 2)

```python
    @typing.overload
    def add(
        self,
        data: Dict[str, np.ndarray],
        env_idxes: Sequence[int] | None = None,
        validate_args: bool = False,
    ) -> None:
        ...
```

**What it does**: Type-checker overload — `add` also accepts a raw `Dict[str, np.ndarray]`. `env_idxes` lets the caller specify which envs each column of `data` belongs to.

---

## Line 875 — `EpisodeBuffer.add` (implementation)

```python
    def add(
        self,
        data: "ReplayBuffer" | Dict[str, np.ndarray],
        env_idxes: Sequence[int] | None = None,
        validate_args: bool = False,
    ) -> None:
        """Add data to the replay buffer in episodes. ...
        """
        if isinstance(data, ReplayBuffer):
            data = data.buffer
        if validate_args:
            if data is None:
                raise ValueError(...)
            if not isinstance(data, dict):
                raise ValueError(...)
            elif isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(v, np.ndarray):
                        raise ValueError(...)
            last_key = next(iter(data.keys()))
            last_batch_shape = next(iter(data.values())).shape[:2]
            for i, (k, v) in enumerate(data.items()):
                if len(v.shape) < 2:
                    raise RuntimeError(...)
                if i > 0:
                    current_key = k
                    current_batch_shape = v.shape[:2]
                    if current_batch_shape != last_batch_shape:
                        raise RuntimeError(...)
                    last_key = current_key
                    last_batch_shape = current_batch_shape
            if "terminated" not in data and "truncated" not in data:
                raise RuntimeError(...)
            if env_idxes is not None and (np.array(env_idxes) >= self._n_envs).any():
                raise ValueError(...)
        if env_idxes is None:
            env_idxes = range(self._n_envs)
        for i, env in enumerate(env_idxes):
            env_data = {k: v[:, i] for k, v in data.items()}
            done = np.logical_or(env_data["terminated"], env_data["truncated"])
            episode_ends = done.nonzero()[0].tolist()
            if len(episode_ends) == 0:
                self._open_episodes[env].append(env_data)
            else:
                episode_ends.append(len(done))
                start = 0
                for ep_end_idx in episode_ends:
                    stop = ep_end_idx
                    episode = {k: env_data[k][start : stop + 1] for k in env_data.keys()}
                    if len(np.logical_or(episode["terminated"], episode["truncated"])) > 0:
                        self._open_episodes[env].append(episode)
                    start = stop + 1
                    should_save = len(self._open_episodes[env]) > 0 and np.logical_or(
                        self._open_episodes[env][-1]["terminated"][-1], self._open_episodes[env][-1]["truncated"][-1]
                    )
                    if should_save:
                        self._save_episode(self._open_episodes[env])
                        self._open_episodes[env] = []
```

**What it does**: Episode-aware ingestion. For each env, computes `done = terminated OR truncated`, finds all `done` indices, and splits the incoming `[T, n_envs, ...]` chunk along those boundaries. Sub-chunks before/at each `done` are appended to that env's open-episode list and immediately flushed via `_save_episode` (which concatenates them into a single contiguous episode and stores it). A trailing sub-chunk after the last `done` (without its own `done`) stays in `_open_episodes` until the next `add()` brings the closing `done`. This is the only writer that distinguishes `terminated` from `truncated`, and it treats them identically (both end the episode).

---

## Line 971 — `EpisodeBuffer._save_episode`

```python
    def _save_episode(self, episode_chunks: Sequence[Dict[str, np.ndarray | MemmapArray]]) -> None:
        if len(episode_chunks) == 0:
            raise RuntimeError(...)
        episode = {k: [] for k in episode_chunks[0].keys()}
        for chunk in episode_chunks:
            for k in chunk.keys():
                episode[k].append(chunk[k])
        episode = {k: np.concatenate(v, axis=0) for k, v in episode.items()}
        ends = np.logical_or(episode["terminated"], episode["truncated"])
        ep_len = ends.shape[0]
        if len(ends.nonzero()[0]) != 1 or ends[-1] != 1:
            raise RuntimeError(f"The episode must contain exactly one done, got: {len(np.nonzero(ends))}")
        if ep_len < self._minimum_episode_length:
            raise RuntimeError(...)
        if ep_len > self._buffer_size:
            raise RuntimeError(...)
        if self.full or len(self) + ep_len > self._buffer_size:
            cum_lengths = np.array(self._cum_lengths)
            mask = (len(self) - cum_lengths + ep_len) <= self._buffer_size
            last_to_remove = mask.argmax()
            if self._memmap and self._memmap_dir is not None:
                for _ in range(last_to_remove + 1):
                    dirname = os.path.dirname(self._buf[0][next(iter(self._buf[0].keys()))].filename)
                    for v in self._buf[0].values():
                        del v
                    del self._buf[0]
                    try:
                        shutil.rmtree(dirname)
                    except Exception as e:
                        logging.error(e)
            else:
                self._buf = self._buf[last_to_remove + 1 :]
            cum_lengths = cum_lengths[last_to_remove + 1 :] - cum_lengths[last_to_remove]
            self._cum_lengths = cum_lengths.tolist()
        self._cum_lengths.append(len(self) + ep_len)
        episode_to_store = episode
        if self._memmap:
            episode_dir = self._memmap_dir / f"episode_{str(uuid.uuid4())}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            episode_to_store = {}
            for k, v in episode.items():
                path = Path(episode_dir / f"{k}.memmap")
                filename = str(path)
                episode_to_store[k] = MemmapArray(
                    filename=str(filename),
                    dtype=v.dtype,
                    shape=v.shape,
                    mode=self._memmap_mode,
                )
                episode_to_store[k][:] = episode[k]
        self._buf.append(episode_to_store)
```

**What it does**: Concatenates the open-episode chunks into one contiguous episode, then asserts exactly-one-done-at-the-end and `minimum_episode_length <= ep_len <= buffer_size`. If adding the new episode would overflow, evicts the oldest episodes (computing how many via `argmax` over the cumulative-length mask) and on memmap mode deletes each evicted episode's directory from disk via `shutil.rmtree`. Finally appends the new episode, materialised either as in-memory arrays or as a fresh `<memmap_dir>/episode_<uuid>/<key>.memmap` per key.

---

## Line 1033 — `EpisodeBuffer.sample`

```python
    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        n_samples: int = 1,
        clone: bool = False,
        sequence_length: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Sample trajectories from the replay buffer.
        ...
        """
        if batch_size <= 0:
            raise ValueError(...)
        if n_samples <= 0:
            raise ValueError(...)
        if sample_next_obs:
            valid_episode_idxes = np.array(self._cum_lengths) - np.array([0] + self._cum_lengths[:-1]) > sequence_length
        else:
            valid_episode_idxes = (
                np.array(self._cum_lengths) - np.array([0] + self._cum_lengths[:-1]) >= sequence_length
            )
        valid_episodes = list(compress(self._buf, valid_episode_idxes))
        if len(valid_episodes) == 0:
            raise RuntimeError(...)
        chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(1, -1)
        nsample_per_eps = np.bincount(np.random.randint(0, len(valid_episodes), (batch_size * n_samples,))).astype(
            np.intp
        )
        samples_per_eps = {k: [] for k in valid_episodes[0].keys()}
        if sample_next_obs:
            samples_per_eps.update({f"next_{k}": [] for k in self._obs_keys})
        for i, n in enumerate(nsample_per_eps):
            if n > 0:
                ep_len = np.logical_or(valid_episodes[i]["terminated"], valid_episodes[i]["truncated"]).shape[0]
                if sample_next_obs:
                    ep_len -= 1
                upper = ep_len - sequence_length + 1
                if self._prioritize_ends:
                    upper += sequence_length
                start_idxes = np.minimum(
                    np.random.randint(0, upper, size=(n,)).reshape(-1, 1), ep_len - sequence_length, dtype=np.intp
                )
                indices = start_idxes + chunk_length
                for k in valid_episodes[0].keys():
                    samples_per_eps[k].append(
                        np.take(valid_episodes[i][k], indices.flat, axis=0).reshape(
                            n, sequence_length, *valid_episodes[i][k].shape[1:]
                        )
                    )
                    if sample_next_obs and k in self._obs_keys:
                        samples_per_eps[f"next_{k}"].append(valid_episodes[i][k][indices + 1])
        samples = {}
        for k, v in samples_per_eps.items():
            if len(v) > 0:
                samples[k] = np.moveaxis(
                    np.concatenate(v, axis=0).reshape(n_samples, batch_size, sequence_length, *v[0].shape[2:]),
                    2,
                    1,
                )
                if clone:
                    samples[k] = samples[k].copy()
        return samples
```

**What it does**: Episode-bounded sampling. First filters out episodes shorter than `sequence_length` (or `+1` if next-obs is needed). Then bin-counts a random distribution of the `batch_size*n_samples` samples over the valid episodes — this gives episodes proportionally more draws if they are simply more numerous, not length-weighted. For each chosen episode, picks `n` start indices in `[0, upper)`; with `prioritize_ends=True`, `upper` is extended by `sequence_length` and then clipped via `np.minimum`, which has the effect of repeating the last valid start index so the **tail of the episode is over-sampled**. Concatenates per-episode slices and reshapes to `[n_samples, sequence_length, batch_size, ...]`.

---

## Line 1123 — `EpisodeBuffer.sample_tensors`

```python
    @torch.no_grad()
    def sample_tensors(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        n_samples: int = 1,
        clone: bool = False,
        sequence_length: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: str | torch.dtype = "cpu",
        from_numpy: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Sample elements from the replay buffer and convert them to torch tensors.
        ...
        """
        samples = self.sample(batch_size, sample_next_obs, n_samples, clone, sequence_length)
        return {
            k: get_tensor(v, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy) for k, v in samples.items()
        }
```

**What it does**: Calls `sample` and converts each value to a torch tensor via `get_tensor`. Same pattern as the corresponding method on the other buffer classes.

---

## Line 1158 — `get_tensor`

```python
def get_tensor(
    array: np.ndarray | MemmapArray,
    dtype: Optional[torch.dtype] = None,
    clone: bool = False,
    device: str | torch.dtype = "cpu",
    from_numpy: bool = False,
) -> Tensor:
    if isinstance(array, MemmapArray):
        array = array.array
    if clone:
        array = array.copy()
    if from_numpy:
        torch_v = torch.from_numpy(array).to(
            dtype=NUMPY_TO_TORCH_DTYPE_DICT[array.dtype] if dtype is None else dtype,
            device=device,
        )
    else:
        torch_v = torch.as_tensor(
            array,
            dtype=NUMPY_TO_TORCH_DTYPE_DICT[array.dtype] if dtype is None else dtype,
            device=device,
        )
    return torch_v
```

**What it does**: Module-level helper used by every `sample_tensors` / `to_tensor`. Unwraps a `MemmapArray` to its underlying numpy view, optionally copies, then converts to a torch tensor. `from_numpy=True` uses `torch.from_numpy` (zero-copy, shares storage) followed by a `.to(device, dtype)` move; `False` uses `torch.as_tensor` (may copy depending on dtype/device). Dtype defaults to `NUMPY_TO_TORCH_DTYPE_DICT[array.dtype]` if not specified.
