"""dreamer-srl replay buffer — SequentialReplayBuffer (in-memory only).

Ported from sheeprl@33b6366:sheeprl/data/buffers.py

DEVIATION D-004 (pre-declared): memmap / memmap_dir / memmap_mode arguments
are omitted entirely. The JAX buffer stores arrays in RAM only. This is not a
semantic deviation — memmap mode and in-memory mode produce the same stored
bytes; D-004 makes the omission an explicit PI-ratifiable entry.

Isolation rule (v2 Risks §13): this module does NOT import from
src.models.dreamer_v3_nnx, src.models.dreamer_v3_trainer, or any other file
in src.models.

Pure-functional design: SequentialReplayBuffer is a Python class (not a flax
pytree) because NumPy arrays are already mutable in-place and the buffer state
is never passed through JAX JIT. The add() and sample() methods mutate
self._buf, self._pos, self._full in-place — matching sheeprl's OOP pattern
exactly. The _sample_at_indices() method is added (not in sheeprl) to expose
explicit-index sampling, bypassing the PRNG, for CP3b's bit-identity tests.

Step 2 (Option M slice — GPU-mode buffer flag):
  device="cpu" (default) preserves all existing numpy behaviour exactly;
  device="gpu" stores arrays as jnp.ndarray and uses JAX ops for add/sample.
  See docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md §Step 2.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np


class SequentialReplayBuffer:
    """Sequential replay buffer — in-memory, pure-numpy, no memmap.

    Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L363-L526
    (SequentialReplayBuffer class, minus the memmap path — see D-004).

    Internally this is represented by a dictionary mapping string to numpy
    arrays. The first dimension of the arrays is the buffer size, while the
    second dimension is the number of environments. The sequentiality comes
    from the fact that the samples are sampled as sequences of consecutive
    elements.

    Args:
        buffer_size (int): the buffer size (number of time steps).
        n_envs (int, optional): the number of environments. Defaults to 1.
        obs_keys (Sequence[str], optional): names of observation keys.
            Defaults to ("observations",).
    """

    batch_axis: int = 2  # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L364

    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("observations",),
        device: str = "cpu",
    ) -> None:
        # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L50-L79 (ReplayBuffer.__init__)
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        if device not in ("cpu", "gpu"):
            raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
        self._buffer_size: int = buffer_size
        self._n_envs: int = n_envs
        self._obs_keys: Sequence[str] = obs_keys
        self._buf: Dict[str, Union[np.ndarray, jax.Array]] = {}
        self._pos: int = 0
        self._full: bool = False
        self._rng: np.random.Generator = np.random.default_rng()
        # Step 2: device flag — "cpu" preserves all existing numpy behaviour;
        # "gpu" uses jnp arrays and JAX-native add/sample ops.
        self._device: str = device
        self._on_gpu: bool = (device == "gpu")

    # ------------------------------------------------------------------
    # Properties (mirrors ReplayBuffer API used by the training loop)
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> Dict[str, np.ndarray]:
        """Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L81-L83"""
        return self._buf

    @property
    def buffer_size(self) -> int:
        """Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L85-L87"""
        return self._buffer_size

    @property
    def full(self) -> bool:
        """Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L89-L91"""
        return self._full

    @property
    def n_envs(self) -> int:
        """Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L93-L95"""
        return self._n_envs

    @property
    def empty(self) -> bool:
        """Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L97-L99"""
        return len(self._buf) == 0

    def __len__(self) -> int:
        """Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L105-L106"""
        return self._buffer_size

    # ------------------------------------------------------------------
    # add()
    # ------------------------------------------------------------------

    def add(
        self,
        data: Dict[str, np.ndarray],
        env_idxes: Optional[List[int]] = None,
        validate_args: bool = False,
    ) -> None:
        """Add data to the replay buffer (ring-buffer semantics).

        Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L145-L221
        (ReplayBuffer.add — the dict-input path only; ReplayBuffer-arg path
        is out of scope for dreamer-srl).

        Data must be a dict of numpy arrays with shape [sequence_length, n_envs, ...].
        If the buffer is full, oldest data is overwritten.

        CP7-P1 fix: when env_idxes is provided (non-None), data contains only
        len(env_idxes) env-columns and is written into those specific columns of
        the full buffer.  This enables the reset_data second write at done
        boundaries (sheeprl@33b6366:dreamer_v3.py:L650:
            rb.add(reset_data, dones_idxes, validate_args=...)).

        Args:
            data (Dict[str, np.ndarray]): transitions to add, each array shaped
                [sequence_length, n_envs, ...] when env_idxes is None, or
                [sequence_length, len(env_idxes), ...] when env_idxes is given.
            env_idxes (Optional[List[int]]): env column indices to write into.
                None → write all env columns (legacy behaviour, no shape change).
                Authorized by: docs/reviews/dreamer_srl_v2_cp7_driver_review.md §P1
                Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L193-L221
                              sheeprl@33b6366:dreamer_v3.py:L650
            validate_args (bool): if True, validate shapes. Defaults to False.
        """
        if validate_args:
            if not isinstance(data, dict):
                raise ValueError(
                    f"'data' must be a dictionary containing Numpy arrays, "
                    f"but 'data' is of type '{type(data)}'"
                )
            for k, v in data.items():
                if self._on_gpu:
                    # GPU mode accepts jax.Array or np.ndarray (will be stored as jnp)
                    if not isinstance(v, (np.ndarray, jax.Array)):
                        raise ValueError(
                            f"'data' must be a dictionary containing Numpy or JAX arrays. "
                            f"Found key '{k}' containing a value of type '{type(v)}'"
                        )
                else:
                    if not isinstance(v, np.ndarray):
                        raise ValueError(
                            f"'data' must be a dictionary containing Numpy arrays. "
                            f"Found key '{k}' containing a value of type '{type(v)}'"
                        )
            last_key = next(iter(data.keys()))
            last_batch_shape = next(iter(data.values())).shape[:2]
            for i, (k, v) in enumerate(data.items()):
                if len(v.shape) < 2:
                    raise RuntimeError(
                        "'data' must have at least 2 dimensions: [sequence_length, n_envs, ...]. "
                        f"Shape of '{k}' is {v.shape}"
                    )
                if i > 0:
                    current_key = k
                    current_batch_shape = v.shape[:2]
                    if current_batch_shape != last_batch_shape:
                        raise RuntimeError(
                            "Every array in 'data' must be congruent in the first 2 dimensions: "
                            f"found key '{last_key}' with shape '{last_batch_shape}' "
                            f"and '{current_key}' with shape '{current_batch_shape}'"
                        )
                    last_key = current_key
                    last_batch_shape = current_batch_shape

        # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L193-L221
        data_len = next(iter(data.values())).shape[0]
        next_pos = (self._pos + data_len) % self._buffer_size
        if next_pos <= self._pos or (data_len > self._buffer_size and not self._full):
            idxes = np.array(list(range(self._pos, self._buffer_size)) + list(range(0, next_pos)))
        else:
            idxes = np.array(range(self._pos, next_pos))
        if data_len > self._buffer_size:
            data_to_store = {k: v[-self._buffer_size - next_pos:] for k, v in data.items()}
        else:
            data_to_store = data

        if self._on_gpu:
            # GPU path: jnp arrays, functional .at[].set() updates.
            # Step 2 (Option M) — mirrors dreamer_v3_trainer.py:946-977.
            jax_idxes = jnp.array(idxes)
            if env_idxes is not None:
                # CP7-P1 subset write: broadcast idxes[:, None] x env_idxes[None, :]
                # to replace np.ix_(idxes, env_idxes).
                jax_env_idxes = jnp.array(env_idxes)
                if self.empty:
                    for k, v in data_to_store.items():
                        jv = jnp.asarray(v)
                        self._buf[k] = jnp.zeros(
                            shape=(self._buffer_size, self._n_envs, *jv.shape[2:]),
                            dtype=jv.dtype,
                        )
                        self._buf[k] = self._buf[k].at[
                            jax_idxes[:, None], jax_env_idxes[None, :]
                        ].set(jv)
                else:
                    for k, v in data_to_store.items():
                        jv = jnp.asarray(v)
                        self._buf[k] = self._buf[k].at[
                            jax_idxes[:, None], jax_env_idxes[None, :]
                        ].set(jv)
            elif self.empty:
                for k, v in data_to_store.items():
                    jv = jnp.asarray(v)
                    self._buf[k] = jnp.zeros(
                        shape=(self._buffer_size, self._n_envs, *jv.shape[2:]),
                        dtype=jv.dtype,
                    )
                    self._buf[k] = self._buf[k].at[jax_idxes].set(jv)
            else:
                for k, v in data_to_store.items():
                    jv = jnp.asarray(v)
                    self._buf[k] = self._buf[k].at[jax_idxes].set(jv)
        else:
            # CPU path: original numpy in-place writes, unchanged.
            if env_idxes is not None:
                # CP7-P1: per-env-subset write for reset_data at done boundaries.
                # data shape: [seq_len, len(env_idxes), ...]; write only into env columns env_idxes.
                # Non-selected env columns at this time slot are left as stale ring-buffer data
                # (acceptable — sequences are sampled per-env and non-done envs are not done here).
                # Ported from sheeprl@33b6366:dreamer_v3.py:L650
                #             sheeprl@33b6366:sheeprl/data/buffers.py:L193-L221 (add with env_idxes)
                if self.empty:
                    # Initialize buffer with zeros for all envs before writing subset.
                    # Use first key to determine trailing shape, then init full buffer.
                    for k, v in data_to_store.items():
                        self._buf[k] = np.zeros(
                            shape=(self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype
                        )
                        self._buf[k][np.ix_(idxes, env_idxes)] = data_to_store[k]
                else:
                    for k, v in data_to_store.items():
                        self._buf[k][np.ix_(idxes, env_idxes)] = data_to_store[k]
            elif self.empty:
                for k, v in data_to_store.items():
                    self._buf[k] = np.empty(
                        shape=(self._buffer_size, self._n_envs, *v.shape[2:]), dtype=v.dtype
                    )
                    self._buf[k][idxes] = data_to_store[k]
            else:
                for k, v in data_to_store.items():
                    self._buf[k][idxes] = data_to_store[k]
        if self._pos + data_len >= self._buffer_size:
            self._full = True
        self._pos = next_pos

    # ------------------------------------------------------------------
    # sample()
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        n_samples: int = 1,
        sequence_length: int = 1,
        key: Optional[jax.Array] = None,
    ) -> Dict[str, Union[np.ndarray, jax.Array]]:
        """Sample elements from the replay buffer in a sequential manner.

        Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L395-L465
        (SequentialReplayBuffer.sample).

        Sequences may straddle done-boundaries (item #2 in the high-risk
        table). The buffer does NOT exclude done indices from valid_idxes —
        only the chunk that would overlap self._pos is excluded. The
        is_first marker inside the sampled window is what CP4b relies on.

        Args:
            batch_size (int): Number of sequences to sample.
            sample_next_obs (bool): whether to sample the next observation.
                Defaults to False.
            clone (bool): whether to clone the sampled arrays. Defaults to False.
            n_samples (int): the number of samples to perform. Defaults to 1.
            sequence_length (int): the length of each sampled sequence.
                Defaults to 1.
            key (Optional[jax.Array]): JAX PRNG key. Required when device="gpu";
                ignored (may be None) when device="cpu". Step 2 addition.

        Returns:
            Dict[str, np.ndarray | jax.Array]: shape [n_samples, sequence_length, batch_size, ...]
        """
        # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L419-L465
        batch_dim = batch_size * n_samples

        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(
                f"'batch_size' ({batch_size}) and 'n_samples' ({n_samples}) must be both greater than 0"
            )
        if not self._full and self._pos == 0:
            raise ValueError(
                "No sample has been added to the buffer. "
                "Please add at least one sample calling 'self.add()'"
            )
        if self._buf is None or len(self._buf) == 0:
            raise RuntimeError("The buffer has not been initialized. Try to add some data first.")
        if not self._full and self._pos - sequence_length + 1 < 1:
            raise ValueError(
                f"Cannot sample a sequence of length {sequence_length}. "
                f"Data added so far: {self._pos}"
            )
        if self._full and sequence_length > self._buffer_size:
            raise ValueError(
                f"The sequence length ({sequence_length}) is greater than "
                f"the buffer size ({self._buffer_size})"
            )
        if self._on_gpu and key is None:
            raise ValueError(
                "A JAX PRNG key must be provided when device='gpu'. "
                "Pass key=jax.random.PRNGKey(...) to sample()."
            )

        # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L438-L456
        if self._on_gpu:
            # GPU path: use jax.random.randint for index sampling.
            # Step 2 (Option M) — mirrors dreamer_v3_trainer.py:997-1011.
            assert key is not None  # guaranteed by check above
            if self._full:
                first_range_end = self._pos - sequence_length + 1
                second_range_end = (
                    self._buffer_size if first_range_end >= 0
                    else self._buffer_size + first_range_end
                )
                valid_idxes = jnp.array(
                    list(range(0, first_range_end)) + list(range(self._pos, second_range_end)),
                    dtype=jnp.int32,
                )
                key, sample_key, env_key = jax.random.split(key, 3)
                rand_pos = jax.random.randint(
                    sample_key, shape=(batch_dim,), minval=0, maxval=len(valid_idxes),
                    dtype=jnp.int32,
                )
                start_idxes = valid_idxes[rand_pos]
            else:
                key, sample_key, env_key = jax.random.split(key, 3)
                start_idxes = jax.random.randint(
                    sample_key, shape=(batch_dim,),
                    minval=0, maxval=self._pos - sequence_length + 1,
                    dtype=jnp.int32,
                )
                # env_key already set by split above
            chunk_length = jnp.arange(sequence_length, dtype=jnp.int32).reshape(1, -1)
            idxes = (start_idxes.reshape(-1, 1) + chunk_length) % self._buffer_size
            return self._get_samples(
                idxes, batch_size, n_samples, sequence_length,
                sample_next_obs=sample_next_obs, clone=clone, gpu_env_key=env_key,
            )
        else:
            # CPU path: original numpy RNG — unchanged.
            if self._full:
                # Exclude the chunk (self._pos - sequence_length, self._pos) — invalid
                first_range_end = self._pos - sequence_length + 1
                second_range_end = (
                    self._buffer_size if first_range_end >= 0
                    else self._buffer_size + first_range_end
                )
                valid_idxes = np.array(
                    list(range(0, first_range_end)) + list(range(self._pos, second_range_end)),
                    dtype=np.intp,
                )
                start_idxes = valid_idxes[
                    self._rng.integers(0, len(valid_idxes), size=(batch_dim,), dtype=np.intp)
                ]
            else:
                start_idxes = self._rng.integers(
                    0, self._pos - sequence_length + 1, size=(batch_dim,), dtype=np.intp
                )

            # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L458-L465
            chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(1, -1)
            idxes = (start_idxes.reshape(-1, 1) + chunk_length) % self._buffer_size

            return self._get_samples(
                idxes, batch_size, n_samples, sequence_length,
                sample_next_obs=sample_next_obs, clone=clone,
            )

    # ------------------------------------------------------------------
    # _get_samples()
    # ------------------------------------------------------------------

    def _get_samples(
        self,
        batch_idxes: Union[np.ndarray, jax.Array],
        batch_size: int,
        n_samples: int,
        sequence_length: int,
        sample_next_obs: bool = False,
        clone: bool = False,
        gpu_env_key: Optional[jax.Array] = None,
    ) -> Dict[str, Union[np.ndarray, jax.Array]]:
        """Internal: retrieve samples given batch_idxes of shape [B*N, seq_len].

        Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L467-L526
        (SequentialReplayBuffer._get_samples).

        Env selection: for n_envs > 1, a single env index is sampled per
        sequence (item #1 in the high-risk table). The env index is tiled
        over the full sequence_length so the flat index selects within one
        env column — sequences never cross env boundaries.

        Args:
            batch_idxes: shape [batch_size * n_samples, sequence_length]
                — time indices for each element of each sequence.
            batch_size: number of sequences per sample.
            n_samples: number of samples.
            sequence_length: length of each sequence.
            sample_next_obs: whether to include next-obs. Defaults to False.
            clone: whether to clone output arrays. Defaults to False.
            gpu_env_key: JAX PRNG key for env-index sampling in GPU mode.
                None for CPU mode (uses self._rng). Step 2 addition.

        Returns:
            Dict[str, np.ndarray | jax.Array]: shape [n_samples, sequence_length, batch_size, ...]
        """
        # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L476-L526
        batch_shape = (batch_size * n_samples, sequence_length)

        if self._on_gpu:
            # GPU path: jnp operations; batch_idxes is already a jax.Array.
            # Step 2 (Option M).
            assert gpu_env_key is not None, "gpu_env_key required in GPU mode"
            flattened_batch_idxes = jnp.ravel(batch_idxes)

            if self._n_envs == 1:
                env_idxes = jnp.zeros((batch_shape[0] * sequence_length,), dtype=jnp.int32)
            else:
                env_idxes_per_seq = jax.random.randint(
                    gpu_env_key, shape=(batch_shape[0],), minval=0, maxval=self._n_envs,
                    dtype=jnp.int32,
                )
                env_idxes = jnp.reshape(env_idxes_per_seq, (-1, 1))
                env_idxes = jnp.tile(env_idxes, (1, sequence_length))
                env_idxes = jnp.ravel(env_idxes)

            flattened_idxes = flattened_batch_idxes * self._n_envs + env_idxes

            samples: Dict[str, Union[np.ndarray, jax.Array]] = {}
            for k, v in self._buf.items():
                flat_v = jnp.reshape(v, (-1, *v.shape[2:]))
                flattened_v = jnp.take(flat_v, flattened_idxes, axis=0)
                batched_v = jnp.reshape(
                    flattened_v,
                    (n_samples, batch_size, sequence_length) + flattened_v.shape[1:],
                )
                # [n_samples, batch_size, seq_len, ...] → [n_samples, seq_len, batch_size, ...]
                samples[k] = jnp.swapaxes(batched_v, axis1=1, axis2=2)
                if sample_next_obs and k in self._obs_keys:
                    next_idxes = (flattened_batch_idxes + 1) % self._buffer_size
                    flat_next_v = jnp.take(
                        flat_v,
                        next_idxes * self._n_envs + env_idxes,
                        axis=0,
                    )
                    batched_next_v = jnp.reshape(
                        flat_next_v,
                        (n_samples, batch_size, sequence_length) + flat_next_v.shape[1:],
                    )
                    samples[f"next_{k}"] = jnp.swapaxes(batched_next_v, axis1=1, axis2=2)
            return samples
        else:
            # CPU path: original numpy code — unchanged.
            # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L476-L526
            flattened_batch_idxes = np.ravel(batch_idxes)

            # Each sequence must come from the same environment (item #1)
            # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L480-L486
            if self._n_envs == 1:
                env_idxes = np.zeros((np.prod(batch_shape),), dtype=np.intp)
            else:
                env_idxes = self._rng.integers(0, self._n_envs, size=(batch_shape[0],), dtype=np.intp)
                env_idxes = np.reshape(env_idxes, (-1, 1))
                env_idxes = np.tile(env_idxes, (1, sequence_length))
                env_idxes = np.ravel(env_idxes)

            # Flatten indexes: flat_idx = time_idx * n_envs + env_idx
            # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L489
            flattened_idxes = (flattened_batch_idxes * self._n_envs + env_idxes).flat

            # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L492-L526
            cpu_samples: Dict[str, np.ndarray] = {}
            for k, v in self._buf.items():
                flattened_v = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
                batched_v = np.reshape(
                    flattened_v,
                    (n_samples, batch_size, sequence_length) + flattened_v.shape[1:],
                )
                # [n_samples, batch_size, seq_len, ...] → [n_samples, seq_len, batch_size, ...]
                cpu_samples[k] = np.swapaxes(batched_v, axis1=1, axis2=2)
                if clone:
                    cpu_samples[k] = cpu_samples[k].copy()
                if sample_next_obs and k in self._obs_keys:
                    flattened_next_v = v[
                        (flattened_batch_idxes + 1) % self._buffer_size, env_idxes
                    ]
                    batched_next_v = np.reshape(
                        flattened_next_v,
                        (n_samples, batch_size, sequence_length) + flattened_next_v.shape[1:],
                    )
                    cpu_samples[f"next_{k}"] = np.swapaxes(batched_next_v, axis1=1, axis2=2)
                    if clone:
                        cpu_samples[f"next_{k}"] = cpu_samples[f"next_{k}"].copy()
            return cpu_samples

    # ------------------------------------------------------------------
    # reset() — curriculum stage-boundary clear (not in sheeprl)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Mark the buffer empty without freeing the backing arrays.

        Matches train.py:1230-1231 (idx=0; size=0) cheap-clear: sample()
        gates on self._pos, so stale rows become unreachable. Used at
        curriculum stage boundaries to prevent cross-stage dynamics
        contamination of the world model.

        Note: self._buf retains its allocated arrays; they will be
        overwritten as new data is added. This avoids re-allocating large
        arrays at each stage boundary.
        """
        self._pos = 0
        self._full = False

    # ------------------------------------------------------------------
    # _sample_at_indices() — CP3b addition (not in sheeprl)
    # ------------------------------------------------------------------

    def _sample_at_indices(
        self,
        precomputed_start_idxes: np.ndarray,
        env_idxes: np.ndarray,
        sequence_length: int,
        batch_size: int,
        n_samples: int = 1,
        clone: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample using pre-computed time start indices and env indices.

        This method bypasses the internal PRNG entirely — both the time-index
        selection and the env-column selection are provided by the caller.
        Added for CP3b's bit-identity test 2: the caller pre-computes sheeprl's
        sample-index sequence (sheeprl._rng.integers outputs) and passes them
        here, making the sampled arrays directly comparable to sheeprl's
        _get_samples output without having to match cross-PRNG streams.

        **CPU-only**: raises NotImplementedError when device="gpu" (Step 2 design
        decision (a) — GPU mode buffers are not tested for bit-identity, which
        is a CPU-only concern; GPU-mode unit tests in test_gpu_buffer.py use
        sample() with a fixed JAX PRNG key instead).

        Args:
            precomputed_start_idxes: shape [batch_size * n_samples]
                — start time indices for each sequence.
            env_idxes: shape [batch_size * n_samples]
                — environment column index for each sequence (one per sequence,
                tiled internally to cover the full sequence_length).
            sequence_length (int): length of each sequence.
            batch_size (int): number of sequences per sample.
            n_samples (int): number of samples. Defaults to 1.
            clone (bool): whether to clone output arrays. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: shape [n_samples, sequence_length, batch_size, ...]
        """
        if self._on_gpu:
            raise NotImplementedError(
                "_sample_at_indices() is CPU-only (device='cpu'). "
                "For GPU mode, use sample(key=...) with a fixed JAX PRNG key."
            )
        chunk_length = np.arange(sequence_length, dtype=np.intp).reshape(1, -1)
        idxes = (precomputed_start_idxes.reshape(-1, 1) + chunk_length) % self._buffer_size

        batch_shape = (batch_size * n_samples, sequence_length)
        flattened_batch_idxes = np.ravel(idxes)

        # Tile env_idxes over sequence_length (same as sheeprl n_envs > 1 branch)
        env_idxes_tiled = np.reshape(env_idxes, (-1, 1))
        env_idxes_tiled = np.tile(env_idxes_tiled, (1, sequence_length))
        env_idxes_flat = np.ravel(env_idxes_tiled)

        flattened_idxes = (flattened_batch_idxes * self._n_envs + env_idxes_flat).flat

        samples: Dict[str, np.ndarray] = {}
        for k, v in self._buf.items():
            flattened_v = np.take(np.reshape(v, (-1, *v.shape[2:])), flattened_idxes, axis=0)
            batched_v = np.reshape(
                flattened_v,
                (n_samples, batch_size, sequence_length) + flattened_v.shape[1:],
            )
            samples[k] = np.swapaxes(batched_v, axis1=1, axis2=2)
            if clone:
                samples[k] = samples[k].copy()
        return samples
