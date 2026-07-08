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
        done_mask: Optional[np.ndarray] = None,
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

        Fix 3 (recompile-storm): when done_mask is provided (preferred path for the
        training loop), data has shape [sequence_length, n_envs, ...] (FULL width).
        Only the env columns where done_mask[e]=True are written; non-done columns
        at this time slot retain their existing ring-buffer values.  Pure numpy —
        no JAX ops involved.  This eliminates the variable-width env_idxes=dones_idxes
        path that previously caused XLA recompiles when data was shaped [1,R,...] for
        variable R.  Takes precedence over env_idxes when both are supplied.

        SUPERSEDED (WP-SRL P2, 2026-07-08): the training driver no longer calls
        this done_mask branch on a SHARED multi-env buffer — the partial-column
        write advanced the shared `_pos` past every non-done env's untouched
        row, punching hole rows into their histories (area report 04 D-03).
        The driver now routes done-boundary writes through
        EnvIndependentSequentialReplayBuffer.add(done_mask=...), which forwards
        full rows only to done envs' own sub-buffers. This branch is retained
        (covered by existing tests; still valid at n_envs == 1, e.g. the GPU
        buffer mode) but no multi-env driver path uses it.

        Args:
            data (Dict[str, np.ndarray]): transitions to add, each array shaped
                [sequence_length, n_envs, ...] when env_idxes is None or done_mask
                is provided, or [sequence_length, len(env_idxes), ...] when env_idxes
                is given.
            env_idxes (Optional[List[int]]): env column indices to write into.
                None → write all env columns (legacy behaviour, no shape change).
                Ignored when done_mask is provided.
                Authorized by: docs/reviews/dreamer_srl_v2_cp7_driver_review.md §P1
                Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L193-L221
                              sheeprl@33b6366:dreamer_v3.py:L650
            validate_args (bool): if True, validate shapes. Defaults to False.
            done_mask (Optional[np.ndarray]): boolean or float array of shape
                [n_envs]. When provided, data must be full-width [seq, n_envs, ...]
                and only columns where done_mask[e] is truthy are written.
                Fix 3 (recompile-storm): preferred over env_idxes for the reset_data
                boundary write in the training loop.
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

        # Fix 3 (recompile-storm): when done_mask is provided, convert it to env_idxes
        # and slice data_to_store to only the done columns.  Data arrives at fixed width
        # [seq, num_envs, ...]; we extract only the done columns here (pure numpy —
        # no JAX traces involved) so the rest of the write path is identical to the
        # existing env_idxes branch.
        if done_mask is not None:
            _done_cols = list(np.where(np.asarray(done_mask, dtype=bool))[0])
            if len(_done_cols) == 0:
                # No done envs this step — nothing to write (pos/full unchanged).
                return
            data_to_store = {k: v[:, _done_cols] for k, v in data_to_store.items()}
            env_idxes = _done_cols

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
    # ready_to_sample() / filled_size — WP-SRL P8/P2 driver conveniences
    # (shared API with EnvIndependentSequentialReplayBuffer so the driver
    # is class-agnostic; not in sheeprl, whose loop needs no such gate)
    # ------------------------------------------------------------------

    def ready_to_sample(self, sequence_length: int) -> bool:
        """True when a sequence of `sequence_length` can be sampled.

        WP-SRL P8: a wrapped-full ring buffer is sampleable even when `_pos`
        has cycled below `sequence_length` — the old driver gate
        (`buffer._pos >= seq_len`) skipped up to seq_len-1 owed gradient
        steps after every ring wrap (area report 04 D-07).

        Regression test:
            tests/algorithms/dreamer_srl/test_env_independent_buffer.py::test_ready_to_sample_after_wrap
        """
        return bool(self._full or self._pos >= sequence_length)

    @property
    def filled_size(self) -> int:
        """Number of stored transitions (capacity once the ring has wrapped)."""
        return int(self._buffer_size if self._full else self._pos)

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


def validate_per_env_capacity(
    per_env_buffer_size: int,
    sequence_length: int,
    *,
    configured_buffer_size: int,
    num_envs: int,
) -> None:
    """N1 (review_srl_parity_fixes.md): fail fast when per-env capacity < seq_len.

    With WP-SRL P2's sheeprl sizing (`buffer.size // num_envs`, done in the
    driver), a small configured buffer + many envs can leave each per-env
    sub-buffer smaller than `algo.per_rank_sequence_length`. In that state
    `ready_to_sample()` returns True once the ring wraps full, but every
    `sample()` raises ("sequence length greater than the buffer size",
    buffers.py:367-371) — the run crashes only at the FIRST post-prefill
    sample instead of at startup. The driver calls this at buffer
    construction (dreamer_srl_main.py, step 7) so the misconfiguration
    surfaces before any environment step is paid for.

    Regression test:
        tests/algorithms/dreamer_srl/test_env_independent_buffer.py::test_per_env_capacity_guard
    """
    if per_env_buffer_size < sequence_length:
        raise ValueError(
            f"per-env replay-buffer capacity {per_env_buffer_size} "
            f"(buffer.size {configured_buffer_size} // num_envs {num_envs}) is "
            f"smaller than algo.per_rank_sequence_length {sequence_length}: "
            f"every sample() would fail after prefill. Increase buffer.size to "
            f"at least {sequence_length * num_envs}, reduce num_envs, or reduce "
            f"the sequence length."
        )


class EnvIndependentSequentialReplayBuffer:
    """n_envs independent single-env SequentialReplayBuffers.

    Structural port of sheeprl@33b6366:sheeprl/data/buffers.py:L529-L699
    (EnvIndependentReplayBuffer with buffer_cls=SequentialReplayBuffer),
    minus the memmap trio (D-004) and torch tensor methods. WP-SRL P2:
    replaces the shared-write-head buffer whose done-mask reset writes
    punched hole rows into non-done envs' columns (area report 04 D-03 —
    probe: env-1 sampled sequence [21, 31, 0, 41]).

    Pinned semantics (each a regression-test assertion in
    tests/algorithms/dreamer_srl/test_env_independent_buffer.py):
      1. Sizing: each sub-buffer holds cfg.buffer.size // num_envs
         transitions (sheeprl dreamer_v3.py:478); division happens in the
         DRIVER, mirroring sheeprl.
      2. Regular add (all envs): env column e of `data` goes to sub-buffer
         e; every sub-buffer's own `_pos` advances by 1 (buffers.py:645-654).
      3. Reset write at done boundaries: routed ONLY to done envs'
         sub-buffers (dreamer_v3.py:650); non-done envs' heads do NOT move —
         no hole rows.
      4. Sample: batch allocated across sub-buffers via
         np.bincount(rng.integers(0, n_envs, (batch_size,))), per-sub-buffer
         sample, concat along the batch axis (axis 2 of
         [n_samples, seq_len, batch_size, ...]) (buffers.py:683-699).
         Per-env valid-index exclusion comes for free from each sub-buffer's
         own `_pos`.
      5. RNG: unseeded np.random.default_rng() for the bincount (matches
         sheeprl; parity row P-14's reproducibility caveat carries over).

    CPU-only: the opt-in GPU buffer mode (--buffer-device gpu, Option M)
    samples with traced jnp gathers; bincount allocation would produce
    variable per-sub-buffer shapes -> recompile storm. The driver therefore
    restricts GPU buffer mode to num_envs == 1, where the plain single-env
    SequentialReplayBuffer is kept (semantically identical to a
    wrapper-of-one). Multi-env GPU buffering is a declared non-goal of
    WP-SRL (fix_plan_srl_parity.md P2 design note).
    """

    batch_axis: int = 2  # concat axis, mirrors SequentialReplayBuffer.batch_axis

    def __init__(
        self,
        buffer_size: int,
        n_envs: int = 1,
        obs_keys: Sequence[str] = ("obs",),
    ) -> None:
        # Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L556-L590
        if buffer_size <= 0:
            raise ValueError(f"The buffer size must be greater than zero, got: {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"The number of environments must be greater than zero, got: {n_envs}")
        self._buf: List[SequentialReplayBuffer] = [
            SequentialReplayBuffer(
                buffer_size=buffer_size,
                n_envs=1,
                obs_keys=obs_keys,
                device="cpu",
            )
            for _ in range(n_envs)
        ]
        self._buffer_size = buffer_size
        self._n_envs = n_envs
        self._obs_keys = obs_keys
        # bincount RNG — unseeded like sheeprl (buffers.py:589; P-14 caveat)
        self._rng: np.random.Generator = np.random.default_rng()
        self._concat_along_axis = SequentialReplayBuffer.batch_axis
        self._on_gpu: bool = False  # driver's GPU-sample branch reads this

    # ------------------------------------------------------------------
    # Properties (sheeprl buffers.py:592-617)
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> Sequence[SequentialReplayBuffer]:
        return tuple(self._buf)

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def full(self) -> Sequence[bool]:
        return tuple(b.full for b in self._buf)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def empty(self) -> Sequence[bool]:
        return tuple(b.empty for b in self._buf)

    def __len__(self) -> int:
        return self._buffer_size

    # ------------------------------------------------------------------
    # add()
    # ------------------------------------------------------------------

    def add(
        self,
        data: Dict[str, np.ndarray],
        validate_args: bool = False,
        done_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Route full-width [seq, n_envs, ...] data to per-env sub-buffers.

        Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L627-L654.
        `done_mask` replaces sheeprl's `indices` argument (Fix-3 fixed-width
        convention): `data` is always full-width [seq, n_envs, ...]; when
        done_mask is provided only the truthy columns are routed — non-done
        sub-buffers are untouched (sheeprl dreamer_v3.py:650 — the
        no-hole-rows property, WP-SRL P2).
        """
        if done_mask is not None:
            indices = np.where(np.asarray(done_mask, dtype=bool))[0]
        else:
            indices = range(self._n_envs)
        for env_idx in indices:
            # sheeprl buffers.py:652-653: one column, width preserved
            env_data = {k: v[:, env_idx:env_idx + 1] for k, v in data.items()}
            self._buf[env_idx].add(env_data, validate_args=validate_args)

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
    ) -> Dict[str, np.ndarray]:
        """bincount batch allocation across sub-buffers + batch-axis concat.

        Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L656-L699.
        Output shape: [n_samples, sequence_length, batch_size, ...].
        """
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError(
                f"'batch_size' ({batch_size}) and 'n_samples' ({n_samples}) "
                "must be both greater than 0"
            )
        bs_per_buf = np.bincount(
            self._rng.integers(0, self._n_envs, (batch_size,)),
            minlength=self._n_envs,
        )
        per_buf_samples = [
            b.sample(
                batch_size=bs,
                sample_next_obs=sample_next_obs,
                clone=clone,
                n_samples=n_samples,
                sequence_length=sequence_length,
            )
            for b, bs in zip(self._buf, bs_per_buf)
            if bs > 0
        ]
        samples: Dict[str, np.ndarray] = {}
        for k in per_buf_samples[0].keys():
            samples[k] = np.concatenate(
                [s[k] for s in per_buf_samples], axis=self._concat_along_axis
            )
        return samples

    # ------------------------------------------------------------------
    # ready_to_sample() / filled_size / reset — driver conveniences
    # (shared API with SequentialReplayBuffer; not in sheeprl)
    # ------------------------------------------------------------------

    def ready_to_sample(self, sequence_length: int) -> bool:
        """WP-SRL P8 gate: every env column has >= sequence_length valid rows,
        counting wrapped buffers as full (`_full or _pos >= seq_len`)."""
        return all(b.ready_to_sample(sequence_length) for b in self._buf)

    @property
    def filled_size(self) -> int:
        """Total stored transitions across sub-buffers
        (continual-stage-swap logging, dreamer_srl_main.py)."""
        return int(sum(b.filled_size for b in self._buf))

    def reset(self) -> None:
        """Curriculum stage-boundary clear — every sub-buffer's head to 0."""
        for b in self._buf:
            b.reset()
